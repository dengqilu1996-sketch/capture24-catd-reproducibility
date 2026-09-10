"""Repeated participant-level RF retraining audit for CATD parameter stability.

Five independent 80/20 participant splits are drawn from P001--P100.  For
each split, fine, coarse, and MET-intensity Balanced RF heads are retrained
with the official tuned parameter files and a 300-tree audit cap (the formal
benchmark uses 3000 trees).  The split-level CATD operating point is selected
with the manuscript's 95% conflict-priority rule.  Only aggregate metrics and
selected parameters are written; no predictions are redistributed.
"""

from __future__ import annotations

import json
from pathlib import Path
import os
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from sklearn import metrics
import yaml


ROOT = Path(__file__).resolve().parents[1]
PREPARED = ROOT / "data" / "prepared_data_official_repro"
BCM = ROOT / "results" / "bcm"
BASELINES = ROOT / "results" / "baselines"
OFFICIAL_CODE = ROOT / "capture24_project" / "official_code" / "capture24"
LABEL_FIELDS = BCM / "capture24_derived_label_fields_20260623_152248.npz"
OPTIMISED = BASELINES / "optimised_params_rf_xgb_compat_full_merged"


def row_normalize(x, eps=1e-12):
    return x / np.maximum(x.sum(axis=1, keepdims=True), eps)


def map_matrix(source, target, source_classes, target_classes, mask, smoothing=1e-3):
    out = np.full((len(source_classes), len(target_classes)), smoothing, dtype="float64")
    si = {c: i for i, c in enumerate(source_classes)}
    ti = {c: i for i, c in enumerate(target_classes)}
    for s, t in zip(source[mask], target[mask]):
        if s in si and t in ti:
            out[si[s], ti[t]] += 1.0
    return row_normalize(out)


def transition_matrix(y, p, classes, train_ids, smoothing=1e-3):
    mask = np.isin(p, train_ids)
    idx = {c: i for i, c in enumerate(classes)}
    out = np.full((len(classes), len(classes)), smoothing, dtype="float64")
    for person in np.unique(p[mask]):
        seq = y[mask & (p == person)]
        for a, b in zip(seq[:-1], seq[1:]):
            if a in idx and b in idx:
                out[idx[a], idx[b]] += 1.0
    return row_normalize(out)


def viterbi(proba, participant, classes, trans, gamma):
    log_trans = gamma * np.log(np.maximum(trans, 1e-12))
    pred_idx = np.zeros(len(participant), dtype=np.int32)
    for person in np.unique(participant):
        rows = np.where(participant == person)[0]
        emit = np.log(np.maximum(proba[rows], 1e-12))
        n, k = emit.shape
        dp = np.zeros((n, k), dtype="float64")
        back = np.zeros((n, k), dtype=np.int32)
        dp[0] = emit[0]
        for t in range(1, n):
            scores = dp[t - 1][:, None] + log_trans
            back[t] = np.argmax(scores, axis=0)
            dp[t] = emit[t] + scores[back[t], np.arange(k)]
        path = np.zeros(n, dtype=np.int32)
        path[-1] = np.argmax(dp[-1])
        for t in range(n - 2, -1, -1):
            path[t] = back[t + 1, path[t + 1]]
        pred_idx[rows] = path
    return classes[pred_idx]


def macro_f1(y, pred):
    return float(metrics.f1_score(y, pred, average="macro", zero_division=0))


def conflict_rate(pred, classes, coarse_hard, intensity_hard, fine_to_coarse, fine_to_intensity):
    idx = {c: i for i, c in enumerate(classes)}
    pi = np.asarray([idx[x] for x in pred], dtype=np.int32)
    c = fine_to_coarse[pi] != coarse_hard
    i = fine_to_intensity[pi] != intensity_hard
    return float((c | i).mean())


def main():
    sys.path.insert(0, str(OFFICIAL_CODE))
    from classifier import Classifier

    X = pd.read_pickle(PREPARED / "X_feats.pkl").values
    P = np.load(PREPARED / "P.npy")
    y = np.load(PREPARED / "Y_WillettsSpecific2018.npy")
    y_coarse = np.load(PREPARED / "Y_Walmsley2020.npy")
    labels = np.load(LABEL_FIELDS, allow_pickle=True)
    y_intensity = labels["y_met_intensity4"]
    valid_annotation = labels["valid_annotation_mask"]
    derivation = np.asarray([f"P{i:03d}" for i in range(1, 101)])
    seeds = [1001, 1002, 1003, 1004, 1005]
    alphas = [0.0, 0.25, 0.5, 1.0]
    betas = [0.0, 0.25, 0.5, 1.0]
    gammas = [0.0, 0.5, 1.0, 2.0, 3.0, 4.0]
    rows = []

    for split_no, seed in enumerate(seeds, start=1):
        rng = np.random.default_rng(seed)
        ids = derivation.copy()
        rng.shuffle(ids)
        train_ids = sorted(ids[:80].tolist())
        val_ids = sorted(ids[80:].tolist())
        train_mask = np.isin(P, train_ids)
        val_mask = np.isin(P, val_ids)
        train_valid_intensity = train_mask & (y_intensity != "unknown")
        common_kwargs = {"n_jobs": 12, "n_estimators": 300}
        start = time.perf_counter()

        fine = Classifier("rf", seed=seed, optimisedir=str(OPTIMISED / "rf_WillettsSpecific2018.pkl"), **common_kwargs)
        fine.fit(X[train_mask], y[train_mask], P[train_mask])
        fine_proba = fine.predict_proba(X[val_mask])
        fine_classes = np.asarray(fine.window_classifier.model.classes_)

        coarse = Classifier("rf", seed=seed, optimisedir=str(OPTIMISED / "rf_Walmsley2020.pkl"), **common_kwargs)
        coarse.fit(X[train_mask], y_coarse[train_mask], P[train_mask])
        coarse_proba = coarse.predict_proba(X[val_mask])
        coarse_classes = np.asarray(coarse.window_classifier.model.classes_)

        intensity = Classifier("rf", seed=seed, optimisedir=str(PREPARED / "__no_optimised_params__"), **common_kwargs)
        intensity.fit(X[train_valid_intensity], y_intensity[train_valid_intensity], P[train_valid_intensity])
        intensity_proba = intensity.predict_proba(X[val_mask])
        intensity_classes = np.asarray(intensity.window_classifier.model.classes_)

        m_coarse = map_matrix(y, y_coarse, fine_classes, coarse_classes, train_mask, smoothing=1e-3)
        m_intensity = map_matrix(y, y_intensity, fine_classes, intensity_classes, train_valid_intensity, smoothing=1e-3)
        coarse_support = coarse_proba @ m_coarse.T
        intensity_support = intensity_proba @ m_intensity.T
        val_p = P[val_mask]
        val_y = y[val_mask]
        val_coarse_hard = coarse_classes[np.argmax(coarse_proba, axis=1)]
        val_intensity_hard = intensity_classes[np.argmax(intensity_proba, axis=1)]
        fine_to_coarse = coarse_classes[np.argmax(m_coarse, axis=1)]
        fine_to_intensity = intensity_classes[np.argmax(m_intensity, axis=1)]
        trans = transition_matrix(y, P, fine_classes, train_ids)

        candidates = []
        for alpha in alphas:
            for beta in betas:
                emission = row_normalize(
                    fine_proba
                    * np.power(np.maximum(coarse_support, 1e-12), alpha)
                    * np.power(np.maximum(intensity_support, 1e-12), beta)
                )
                for gamma in gammas:
                    pred = viterbi(emission, val_p, fine_classes, trans, gamma)
                    candidates.append({
                        "alpha": alpha,
                        "beta": beta,
                        "gamma": gamma,
                        "macro_f1": macro_f1(val_y, pred),
                        "any_conflict_rate": conflict_rate(pred, fine_classes, val_coarse_hard, val_intensity_hard, fine_to_coarse, fine_to_intensity),
                    })
        cand = pd.DataFrame(candidates)
        best_macro = float(cand["macro_f1"].max())
        selected = cand[cand["macro_f1"] >= best_macro * 0.95].sort_values(
            ["any_conflict_rate", "macro_f1"], ascending=[True, False]
        ).iloc[0]
        rows.append({
            "split": split_no,
            "seed": seed,
            "train_participants": len(train_ids),
            "validation_participants": len(val_ids),
            "selected_alpha": float(selected.alpha),
            "selected_beta": float(selected.beta),
            "selected_gamma": float(selected.gamma),
            "best_validation_macro_f1": best_macro,
            "selected_validation_macro_f1": float(selected.macro_f1),
            "selected_any_conflict_rate": float(selected.any_conflict_rate),
            "elapsed_seconds": time.perf_counter() - start,
            "n_estimators": 300,
        })
        print(rows[-1])

    df = pd.DataFrame(rows)
    out_csv = BCM / "repeated_rf_retraining_validation_20260905.csv"
    out_json = BCM / "repeated_rf_retraining_validation_20260905.json"
    out_md = BCM / "repeated_rf_retraining_validation_20260905.md"
    df.to_csv(out_csv, index=False)
    summary = {
        "n_splits": len(df),
        "selection_frequency": {
            f"{a:g},{b:g},{g:g}": int(((df.selected_alpha == a) & (df.selected_beta == b) & (df.selected_gamma == g)).sum())
            for a in alphas for b in betas for g in gammas
            if int(((df.selected_alpha == a) & (df.selected_beta == b) & (df.selected_gamma == g)).sum())
        },
        "selected_macro_f1_mean": float(df.selected_validation_macro_f1.mean()),
        "selected_macro_f1_min": float(df.selected_validation_macro_f1.min()),
        "selected_macro_f1_max": float(df.selected_validation_macro_f1.max()),
        "selected_any_conflict_mean": float(df.selected_any_conflict_rate.mean()),
        "audit_tree_count": 300,
        "note": "Retraining audit uses 300-tree RF heads for tractability; the formal locked benchmark uses 3000 trees. No predictions are saved.",
    }
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with out_md.open("w", encoding="utf-8") as f:
        f.write("# Repeated participant-level RF retraining audit\n\n")
        f.write("Five independent 80/20 participant splits were drawn from P001--P100. Fine, coarse, and MET-intensity RF heads were retrained for each split with a 300-tree audit cap; the formal benchmark uses 3000 trees. The 95% conflict-priority rule was then applied to each split. Only aggregate metrics and selected parameters are retained.\n\n")
        f.write(df.to_markdown(index=False))
        f.write("\n\n## Summary\n\n")
        f.write(json.dumps(summary, indent=2))
        f.write("\n")
    print(out_md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
