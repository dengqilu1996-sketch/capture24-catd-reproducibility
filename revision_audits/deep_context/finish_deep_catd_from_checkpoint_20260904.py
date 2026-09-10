"""Finish posterior export and CATD evaluation from the completed CNN checkpoint."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn import metrics

SRC = Path(r"D:/capture24_project_current/experiments/train_deep_catd_multitask_20260904.py")
spec = importlib.util.spec_from_file_location("exp", SRC)
exp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exp)
ROOT, DATA, FIELDS, OUT = exp.ROOT, exp.DATA, exp.FIELDS, exp.OUT
FINE, COARSE, INTENS = exp.FINE, exp.COARSE, exp.INTENS

z = np.load(FIELDS, allow_pickle=True)
valid = z["valid_annotation_mask"]
p = z["participant"][valid]
yf0 = z["y_willetts_specific2018"][valid]
yc0 = z["y_walmsley2020"][valid]
yi0 = z["y_met_intensity4"][valid]
X = np.load(DATA / "X.npy", mmap_mode="r")
fidx = {c: i for i, c in enumerate(FINE)}
cidx = {c: i for i, c in enumerate(COARSE)}
iidx = {c: i for i, c in enumerate(INTENS)}
yf = np.array([fidx[x] for x in yf0], dtype=np.int64)
yc = np.array([cidx[x] for x in yc0], dtype=np.int64)
yi = np.array([iidx[x] for x in yi0], dtype=np.int64)
train = np.isin(p, [f"P{i:03d}" for i in range(1, 81)])
val = np.isin(p, [f"P{i:03d}" for i in range(81, 101)])
test = np.isin(p, [f"P{i:03d}" for i in range(101, 152)])
ti, vi, si = np.flatnonzero(train), np.flatnonzero(val), np.flatnonzero(test)
va = exp.WindowDataset(X, vi, yf[vi], yc[vi], yi[vi])
te = exp.WindowDataset(X, si, yf[si], yc[si], yi[si])
dlva = DataLoader(va, batch_size=2048, shuffle=False, num_workers=0, pin_memory=True)
dlte = DataLoader(te, batch_size=2048, shuffle=False, num_workers=0, pin_memory=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = exp.MultiHeadCNN(len(FINE), len(COARSE), len(INTENS)).to(device)
ck = torch.load(OUT / "best.pt", map_location=device, weights_only=False)
model.load_state_dict(ck["model"])
losses = (
    nn.CrossEntropyLoss(weight=exp.weight(yf[ti], len(FINE)).to(device)),
    nn.CrossEntropyLoss(weight=exp.weight(yc[ti], len(COARSE)).to(device)),
    nn.CrossEntropyLoss(weight=exp.weight(yi[ti], len(INTENS)).to(device)),
)
valres, _, _ = exp.evaluate(model, dlva, device, losses, save_logits=True)
testres, _, _ = exp.evaluate(model, dlte, device, losses, save_logits=True)

def clean(x):
    return {k: v for k, v in x.items() if k != "logits"}

np.savez_compressed(
    OUT / "posteriors.npz", participant=p[vi], y_fine=yf0[vi],
    fine_logits=valres["logits"][0], coarse_logits=valres["logits"][1],
    intensity_logits=valres["logits"][2], val_index=vi,
)
np.savez_compressed(
    OUT / "posteriors_test.npz", participant=p[si], y_fine=yf0[si],
    fine_logits=testres["logits"][0], coarse_logits=testres["logits"][1],
    intensity_logits=testres["logits"][2], test_index=si,
)
json.dump({"best_epoch": int(ck["epoch"]), "val": clean(valres),
           "test": clean(testres), "classes": {"fine": FINE.tolist(),
           "coarse": COARSE.tolist(), "intensity": INTENS.tolist()}},
          open(OUT / "training_report.json", "w", encoding="utf-8"), indent=2)

def majority(src, target):
    m = np.zeros((len(FINE), len(target)), float)
    ix = {x: i for i, x in enumerate(target)}
    for a, b in zip(yf0[ti], src[ti]):
        m[fidx[a], ix[b]] += 1
    return m / np.maximum(m.sum(1, keepdims=True), 1)

mf2c, mf2i = majority(yc0, COARSE), majority(yi0, INTENS)
fine_to_coarse = COARSE[np.argmax(mf2c, axis=1)]
fine_to_intensity = INTENS[np.argmax(mf2i, axis=1)]
trm = exp.transition(yf0[ti], p[ti], FINE)
def soft(a):
    q = a - a.max(1, keepdims=True)
    q = np.exp(q)
    return q / q.sum(1, keepdims=True)

pv, pc, pi = soft(valres["logits"][0]), soft(valres["logits"][1]), soft(valres["logits"][2])
ps, pcs, pis = soft(testres["logits"][0]), soft(testres["logits"][1]), soft(testres["logits"][2])
def conflict_metrics(pred, coarse_hard, intensity_hard):
    pred_idx = np.array([fidx[x] for x in pred], dtype=int)
    coarse_conflict = fine_to_coarse[pred_idx] != COARSE[coarse_hard]
    intensity_conflict = fine_to_intensity[pred_idx] != INTENS[intensity_hard]
    return {
        "coarse_conflict_rate": float(coarse_conflict.mean()),
        "intensity_conflict_rate": float(intensity_conflict.mean()),
        "any_conflict_rate": float((coarse_conflict | intensity_conflict).mean()),
    }
grid = []
for a in [0, .25, .5, 1]:
    for b in [0, .25, .5, 1]:
        em = pv * np.power(np.maximum(pc @ mf2c.T, 1e-12), a) * np.power(np.maximum(pi @ mf2i.T, 1e-12), b)
        for g in [0, .5, 1, 2, 3, 4, 5, 6]:
            pred = exp.viterbi(em, p[vi], FINE, trm, g)
            grid.append({"alpha": a, "beta": b, "gamma": g,
                         "macro_f1": exp.macro(yf0[vi], pred),
                         "balanced_accuracy": metrics.balanced_accuracy_score(yf0[vi], pred),
                         **conflict_metrics(pred, valres["logits"][1].argmax(1), valres["logits"][2].argmax(1))})
gd = pd.DataFrame(grid)
bestm = gd.macro_f1.max()
eligible = gd[gd.macro_f1 >= .95 * bestm]
sel = eligible.sort_values(["any_conflict_rate", "intensity_conflict_rate", "macro_f1", "balanced_accuracy"], ascending=[True, True, False, False]).iloc[0]
to = gd[(gd.alpha == 0) & (gd.beta == 0)].sort_values(["macro_f1", "balanced_accuracy"], ascending=False).iloc[0]
ems = ps * np.power(np.maximum(pcs @ mf2c.T, 1e-12), float(sel.alpha)) * np.power(np.maximum(pis @ mf2i.T, 1e-12), float(sel.beta))
predc = exp.viterbi(ems, p[si], FINE, trm, float(sel.gamma))
predt = exp.viterbi(ps, p[si], FINE, trm, float(to.gamma))
predh = exp.viterbi(ps, p[si], FINE, trm, 1)
base = FINE[np.argmax(testres["logits"][0], axis=1)]
result = []
for name, pred in [("cnn", base), ("cnn_hmm", predh), ("cnn_temporal_only", predt), ("cnn_catd", predc)]:
    row = {"method": name,
        "alpha": float(sel.alpha) if name == "cnn_catd" else (0 if name == "cnn_temporal_only" else np.nan),
        "beta": float(sel.beta) if name == "cnn_catd" else (0 if name == "cnn_temporal_only" else np.nan),
        "gamma": float(sel.gamma) if name == "cnn_catd" else (float(to.gamma) if name == "cnn_temporal_only" else (1 if name == "cnn_hmm" else np.nan)),
        "macro_f1": exp.macro(yf0[si], pred),
        "balanced_accuracy": metrics.balanced_accuracy_score(yf0[si], pred)}
    row.update(conflict_metrics(pred, testres["logits"][1].argmax(1), testres["logits"][2].argmax(1)))
    result.append(row)
gd.to_csv(OUT / "validation_candidates.csv", index=False)
pd.DataFrame(result).to_csv(OUT / "locked_test_metrics.csv", index=False)
json.dump({"selection_rule": "conflict-priority; retain >=95% of validation maximum macro-F1, then minimize any conflict",
           "selected": sel.to_dict(), "temporal_only": to.to_dict(), "locked_rows": result},
          open(OUT / "catd_report.json", "w", encoding="utf-8"), indent=2)
print("val", clean(valres)); print("test", clean(testres)); print("selected", sel.to_dict())
print(pd.DataFrame(result).to_string(index=False))
