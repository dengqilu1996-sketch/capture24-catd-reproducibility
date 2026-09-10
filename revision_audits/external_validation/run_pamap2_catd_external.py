"""Participant-disjoint external CATD audit on PAMAP2 protocol data.

The experiment keeps the Capture-24 decoding logic but retrains all three RF
heads on PAMAP2.  No Capture-24 or PAMAP2 test participant is used for fitting
the mapping, transition matrix, or operating-point selection.  Raw data remain
local; this script writes only summary CSV/JSON outputs.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
from pathlib import Path

import numpy as np
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.metrics import balanced_accuracy_score, f1_score, matthews_corrcoef


SAMPLE_RATE = 100
WINDOW = 1000
GRAVITY = 9.80665
SMOOTH = 1e-3
EPS = 1e-12

# PAMAP2 protocol activities (optional activities are excluded because they
# have poor participant coverage).  The external fine task is intentionally
# dataset-specific; CATD is evaluated as a decoding mechanism, not by forcing
# Capture-24 class names onto PAMAP2.
ACTIVITY_IDS = (1, 2, 3, 4, 5, 6, 7, 12, 13, 16, 17, 24)
ACTIVITY_NAMES = {
    1: "lying", 2: "sitting", 3: "standing", 4: "walking",
    5: "running", 6: "cycling", 7: "nordic_walking",
    12: "ascending_stairs", 13: "descending_stairs",
    16: "vacuum_cleaning", 17: "ironing", 24: "rope_jumping",
}
FINE_TO_ID = {aid: i for i, aid in enumerate(ACTIVITY_IDS)}
FINE_NAMES = [ACTIVITY_NAMES[aid] for aid in ACTIVITY_IDS]

# Fixed before looking at test scores.  These groups are a semantic hierarchy
# for PAMAP2 and are not learned from the held-out participants.
COARSE_BY_ACTIVITY = {
    1: "postural", 2: "postural", 3: "postural",
    4: "locomotion", 5: "locomotion", 7: "locomotion",
    12: "locomotion", 13: "locomotion",
    16: "household", 17: "household",
    6: "sport", 24: "sport",
}
COARSE_NAMES = ["postural", "household", "locomotion", "sport"]
COARSE_TO_ID = {name: i for i, name in enumerate(COARSE_NAMES)}

# Same three non-sleep strata used in the prior fixed PAMAP2 stress test.
# Values correspond to fixed activity/MET groupings; no target score is used.
INTENSITY_BY_ACTIVITY = {
    1: "sedentary", 2: "sedentary",
    3: "light", 17: "light",
    4: "mvpa", 5: "mvpa", 6: "mvpa", 7: "mvpa",
    12: "mvpa", 13: "mvpa", 16: "mvpa", 24: "mvpa",
}
INTENSITY_NAMES = ["light", "mvpa", "sedentary"]
INTENSITY_TO_ID = {name: i for i, name in enumerate(INTENSITY_NAMES)}


def load_feature_module() -> object:
    here = Path(__file__).resolve().parent
    candidates = [
        here / "capture24_project" / "official_code" / "capture24" / "features.py",
        Path(r"D:\capture24_project_current\capture24_project\official_code\capture24\features.py"),
    ]
    feature_path = next((p for p in candidates if p.is_file()), None)
    if feature_path is None:
        raise FileNotFoundError("; ".join(map(str, candidates)))
    spec = importlib.util.spec_from_file_location("capture24_features", feature_path)
    if spec is None or spec.loader is None:
        raise ImportError(feature_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def segment_subject(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    raw = np.loadtxt(path, delimiter=",", skiprows=1, usecols=(0, 1, 4, 5, 6), dtype=np.float64)
    ts, activity, accel = raw[:, 0], raw[:, 1], raw[:, 2:5]
    valid = np.isfinite(ts) & np.isfinite(activity) & np.isfinite(accel).all(axis=1)
    valid &= np.isin(activity.astype(np.int64, copy=False), ACTIVITY_IDS)
    starts = np.ones(len(raw), dtype=bool)
    if len(raw) > 1:
        starts[1:] = (
            ~valid[1:] | ~valid[:-1]
            | (activity[1:] != activity[:-1])
            | (np.abs(np.diff(ts) - 0.01) > 0.001)
        )
    run_starts = np.flatnonzero(starts)
    run_ends = np.r_[run_starts[1:], len(raw)]
    xs, ys, cs, ins = [], [], [], []
    for start, end in zip(run_starts, run_ends, strict=True):
        if not valid[start]:
            continue
        aid = int(activity[start])
        n = ((end - start) // WINDOW) * WINDOW
        if n <= 0:
            continue
        x = accel[start:start + n].reshape(-1, WINDOW, 3) / GRAVITY
        xs.append(np.asarray(x, dtype=np.float32))
        ys.append(np.full(len(x), FINE_TO_ID[aid], dtype=np.int64))
        cs.append(np.full(len(x), COARSE_TO_ID[COARSE_BY_ACTIVITY[aid]], dtype=np.int64))
        ins.append(np.full(len(x), INTENSITY_TO_ID[INTENSITY_BY_ACTIVITY[aid]], dtype=np.int64))
    if not xs:
        raise RuntimeError(f"No usable windows in {path}")
    return np.concatenate(xs), np.concatenate(ys), np.concatenate(cs), np.concatenate(ins)


def make_features(x: np.ndarray, feature_module: object) -> np.ndarray:
    rows = []
    for window in x:
        values = feature_module.extract_features(window, sample_rate=SAMPLE_RATE)
        if not values:
            raise RuntimeError("Feature extraction returned an empty row")
        rows.append(list(values.values()))
    out = np.asarray(rows, dtype=np.float32)
    if out.shape[1] != 32 or not np.isfinite(out).all():
        raise RuntimeError(f"Unexpected feature matrix: {out.shape}")
    return out


def transition_matrix(y: np.ndarray, participants: np.ndarray, n_classes: int) -> np.ndarray:
    counts = np.full((n_classes, n_classes), SMOOTH, dtype=np.float64)
    for participant in np.unique(participants):
        idx = np.flatnonzero(participants == participant)
        seq = y[idx]
        for a, b in zip(seq[:-1], seq[1:]):
            counts[int(a), int(b)] += 1
    return counts / counts.sum(axis=1, keepdims=True)


def mapping_matrix(y_fine: np.ndarray, y_aux: np.ndarray, n_aux: int) -> np.ndarray:
    counts = np.full((len(FINE_NAMES), n_aux), SMOOTH, dtype=np.float64)
    for fine, aux in zip(y_fine, y_aux, strict=True):
        counts[int(fine), int(aux)] += 1
    return counts / counts.sum(axis=1, keepdims=True)


def decode(proba: np.ndarray, participants: np.ndarray, trans: np.ndarray, gamma: float, support_c: np.ndarray | None = None, support_i: np.ndarray | None = None, alpha: float = 0.0, beta: float = 0.0) -> np.ndarray:
    emission = np.maximum(proba, EPS).astype(np.float64)
    if support_c is not None:
        emission *= np.power(np.maximum(support_c, EPS), alpha)
    if support_i is not None:
        emission *= np.power(np.maximum(support_i, EPS), beta)
    emission /= np.maximum(emission.sum(axis=1, keepdims=True), EPS)
    log_trans = gamma * np.log(np.maximum(trans, EPS))
    out = np.empty(len(proba), dtype=np.int64)
    for participant in np.unique(participants):
        idx = np.flatnonzero(participants == participant)
        e = np.log(np.maximum(emission[idx], EPS))
        dp = np.empty_like(e)
        back = np.zeros((len(idx), e.shape[1]), dtype=np.int64)
        dp[0] = e[0]
        for t in range(1, len(idx)):
            score = dp[t - 1][:, None] + log_trans
            back[t] = np.argmax(score, axis=0)
            dp[t] = e[t] + np.max(score, axis=0)
        seq = np.empty(len(idx), dtype=np.int64)
        seq[-1] = int(np.argmax(dp[-1]))
        for t in range(len(idx) - 1, 0, -1):
            seq[t - 1] = back[t, seq[t]]
        out[idx] = seq
    return out


def metric_row(name: str, y: np.ndarray, pred: np.ndarray, coarse_true: np.ndarray, intensity_true: np.ndarray, coarse_map: np.ndarray, intensity_map: np.ndarray) -> dict:
    coarse_pred = coarse_map[pred]
    intensity_pred = intensity_map[pred]
    # Some PAMAP2 protocol activities are absent for individual held-out
    # participants.  Macro-F1 is therefore computed over classes supported by
    # that test partition; the full activity support is reported separately.
    supported = np.unique(y)
    return {
        "method": name,
        "n_windows": int(len(y)),
        "n_observed_fine_classes": int(len(supported)),
        "macro_f1": float(f1_score(y, pred, average="macro", labels=supported, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "mcc": float(matthews_corrcoef(y, pred)),
        "coarse_conflict": float(np.mean(coarse_pred != coarse_true)),
        "intensity_conflict": float(np.mean(intensity_pred != intensity_true)),
        "any_conflict": float(np.mean((coarse_pred != coarse_true) | (intensity_pred != intensity_true))),
    }


def fit_head(X: np.ndarray, y: np.ndarray, n_estimators: int, seed: int) -> BalancedRandomForestClassifier:
    model = BalancedRandomForestClassifier(
        n_estimators=n_estimators, bootstrap=True, replacement=True,
        sampling_strategy="not minority", oob_score=False,
        max_features="sqrt", n_jobs=12, random_state=seed,
    )
    model.fit(X, y)
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=Path(r"D:\claude项目\capture24_work2\capture24\external_data\pamap2\ProtocolCsv"))
    parser.add_argument("--output-dir", type=Path, default=Path(r"D:\claude项目\capture24_work2\capture24\external_validation\pamap2_catd_external"))
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite:
        raise SystemExit(f"Output exists: {args.output_dir}; use --overwrite")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    feature_module = load_feature_module()
    Xs, ys, cs, ins, ps = [], [], [], [], []
    for path in sorted(args.input_dir.glob("subject*.csv")):
        subject = int(path.stem.replace("subject", ""))
        if subject == 109:
            continue  # only six rope-jump windows; no complete external endpoint
        x, y, c, i = segment_subject(path)
        Xs.append(make_features(x, feature_module))
        ys.append(y); cs.append(c); ins.append(i); ps.append(np.full(len(y), subject, dtype=np.int64))
        print(f"subject {subject}: {len(y)} windows", flush=True)
    X = np.concatenate(Xs); y = np.concatenate(ys); c = np.concatenate(cs); intensity = np.concatenate(ins); participants = np.concatenate(ps)
    np.savez_compressed(args.output_dir / "pamap2_external_features.npz", X=X, y_fine=y, y_coarse=c, y_intensity=intensity, participant=participants)

    # Four fixed participant-disjoint splits. Each split uses five train, one
    # validation, and two locked-test participants; subject 109 is excluded.
    splits = [
        {"name": "fold1", "train": [104, 105, 106, 107, 108], "val": [103], "test": [101, 102]},
        {"name": "fold2", "train": [101, 102, 106, 107, 108], "val": [105], "test": [103, 104]},
        {"name": "fold3", "train": [101, 102, 103, 104, 108], "val": [107], "test": [105, 106]},
        {"name": "fold4", "train": [102, 103, 104, 105, 106], "val": [101], "test": [107, 108]},
    ]
    rows = []
    for split in splits:
        tr = np.isin(participants, split["train"]); va = np.isin(participants, split["val"]); te = np.isin(participants, split["test"])
        fine = fit_head(X[tr], y[tr], args.n_estimators, args.seed)
        coarse = fit_head(X[tr], c[tr], args.n_estimators, args.seed + 1)
        inten = fit_head(X[tr], intensity[tr], args.n_estimators, args.seed + 2)
        classes_f = np.asarray(fine.classes_); classes_c = np.asarray(coarse.classes_); classes_i = np.asarray(inten.classes_)
        # All core labels should be represented in training. If not, abort
        # rather than silently changing the external estimand.
        if len(classes_f) != len(FINE_NAMES) or len(classes_c) != len(COARSE_NAMES) or len(classes_i) != len(INTENSITY_NAMES):
            raise RuntimeError(f"Incomplete classes in {split['name']}: {classes_f}, {classes_c}, {classes_i}")
        pva = fine.predict_proba(X[va]); pcva = coarse.predict_proba(X[va]); piva = inten.predict_proba(X[va])
        pte = fine.predict_proba(X[te]); pcte = coarse.predict_proba(X[te]); pite = inten.predict_proba(X[te])
        map_c = mapping_matrix(y[tr], c[tr], len(COARSE_NAMES)); map_i = mapping_matrix(y[tr], intensity[tr], len(INTENSITY_NAMES))
        trans = transition_matrix(y[tr], participants[tr], len(FINE_NAMES))
        fine_to_c = np.argmax(map_c, axis=1); fine_to_i = np.argmax(map_i, axis=1)
        va_c_support = pcva @ map_c.T; va_i_support = piva @ map_i.T
        te_c_support = pcte @ map_c.T; te_i_support = pite @ map_i.T
        selected = None; max_f1 = -1.0
        candidates = []
        for alpha in (0.0, 0.25, 0.5, 1.0):
            for beta in (0.0, 0.25, 0.5, 1.0):
                for gamma in (0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0):
                    pred = decode(pva, participants[va], trans, gamma, va_c_support, va_i_support, alpha, beta)
                    f1 = f1_score(y[va], pred, average="macro", labels=np.unique(y[va]), zero_division=0)
                    cc = float(np.mean(fine_to_c[pred] != c[va])); ci = float(np.mean(fine_to_i[pred] != intensity[va]))
                    anyc = float(np.mean((fine_to_c[pred] != c[va]) | (fine_to_i[pred] != intensity[va])))
                    candidates.append((alpha, beta, gamma, float(f1), anyc, cc, ci))
                    max_f1 = max(max_f1, float(f1))
        eligible = [row for row in candidates if row[3] >= 0.95 * max_f1]
        selected = min(eligible, key=lambda row: (row[4], -row[3], row[0], row[1], row[2]))
        alpha, beta, gamma = selected[:3]
        # Baselines and CATD on the locked test participants.
        pred_rf = classes_f[np.argmax(pte, axis=1)]
        pred_temporal = decode(pte, participants[te], trans, 3.0)
        pred_catd = decode(pte, participants[te], trans, gamma, te_c_support, te_i_support, alpha, beta)
        for name, pred in (("RF", pred_rf), ("temporal_only", pred_temporal), ("CATD_conflict_priority", pred_catd)):
            row = metric_row(name, y[te], pred, c[te], intensity[te], fine_to_c, fine_to_i)
            row.update({"split": split["name"], "n_estimators": args.n_estimators, "alpha": alpha, "beta": beta, "gamma": gamma, "val_macro_f1": selected[3], "val_any_conflict": selected[4], "train_participants": ";".join(map(str, split["train"])), "val_participants": ";".join(map(str, split["val"])), "test_participants": ";".join(map(str, split["test"]))})
            rows.append(row)
        print(split["name"], "selected", (alpha, beta, gamma), "test CATD", rows[-1], flush=True)
    with (args.output_dir / "external_catd_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    metadata = {"dataset": "PAMAP2 protocol", "participants": [101,102,103,104,105,106,107,108], "fine_classes": FINE_NAMES, "coarse_classes": COARSE_NAMES, "intensity_classes": INTENSITY_NAMES, "n_windows": int(len(y)), "n_features": int(X.shape[1]), "n_estimators": args.n_estimators, "splits": splits, "mapping_source": "fixed PAMAP2 semantic activity map; mapping and transitions estimated on training participants only", "status": "complete"}
    (args.output_dir / "external_catd_manifest.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Saved {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
