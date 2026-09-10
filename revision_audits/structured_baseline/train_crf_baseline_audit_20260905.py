"""Train a compact linear-chain CRF structured baseline on Capture-24.

This is an aggregate-only audit.  To keep the structured baseline feasible on a
desktop, it uses the first 200 chronologically ordered windows per participant
for each split (P001--P080 train, P081--P100 validation, P101--P151 locked test).
No per-window predictions are written to disk.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn_crfsuite
from sklearn.metrics import (
    balanced_accuracy_score,
    cohen_kappa_score,
    f1_score,
    matthews_corrcoef,
)


ROOT = Path(r"D:\capture24_project_current")
DATA = ROOT / "data" / "prepared_data_official_repro"
OUT = ROOT / "results" / "structured_baseline_crf_20260905"
OUT.mkdir(parents=True, exist_ok=True)

CAP = 200
FEATURE_COLUMNS = [
    "avg",
    "std",
    "skew",
    "kurt",
    "min",
    "q25",
    "med",
    "q75",
    "max",
    "power",
    "f1",
    "f2",
    "f3",
    "p1",
    "p2",
    "p3",
]


def participant_ids(prefix: str, start: int, stop: int) -> list[str]:
    return [f"{prefix}{i:03d}" for i in range(start, stop + 1)]


def make_rows(participants: list[str], p: np.ndarray, cap: int) -> list[np.ndarray]:
    rows: list[np.ndarray] = []
    for pid in participants:
        idx = np.flatnonzero(p == pid)
        if idx.size == 0:
            continue
        # Spread the cap across each participant's recording so the audit does
        # not accidentally select only the first (often single-activity) block.
        if idx.size <= cap:
            rows.append(idx)
        else:
            rows.append(np.unique(np.linspace(0, idx.size - 1, cap, dtype=int)))
    return rows


def fit_bins(x: np.ndarray) -> list[np.ndarray]:
    # Quantile bins are fitted only on the training subset, avoiding leakage.
    qs = np.linspace(0.0, 1.0, 9)
    bins: list[np.ndarray] = []
    for j in range(x.shape[1]):
        edges = np.unique(np.nanquantile(x[:, j], qs))
        if edges.size < 2:
            edges = np.array([-np.inf, np.inf], dtype=float)
        else:
            edges = edges.astype(float)
            edges[0] = -np.inf
            edges[-1] = np.inf
        bins.append(edges)
    return bins


def sequence_features(x: np.ndarray, bins: list[np.ndarray]) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for row in x:
        d = {"bias": "1"}
        for j, edges in enumerate(bins):
            # searchsorted is robust to repeated quantiles removed above.
            b = int(np.searchsorted(edges[1:-1], float(row[j]), side="right"))
            d[f"f{j}"] = str(b)
        out.append(d)
    return out


def flatten(seqs: list[list[str]]) -> list[str]:
    return [v for seq in seqs for v in seq]


def main() -> None:
    started = time.time()
    xdf = pd.read_pickle(DATA / "X_feats.pkl")
    p = np.load(DATA / "P.npy", allow_pickle=True)
    y = np.load(DATA / "Y_WillettsSpecific2018.npy", allow_pickle=True)
    x = xdf[FEATURE_COLUMNS].to_numpy(dtype=np.float64)

    train_rows = make_rows(participant_ids("P", 1, 80), p, CAP)
    val_rows = make_rows(participant_ids("P", 81, 100), p, CAP)
    test_rows = make_rows(participant_ids("P", 101, 151), p, CAP)
    if len(train_rows) != 80 or len(val_rows) != 20 or len(test_rows) != 51:
        raise RuntimeError("Unexpected participant coverage in prepared arrays")

    train_flat = np.concatenate(train_rows)
    bins = fit_bins(x[train_flat])
    x_train = [sequence_features(x[idx], bins) for idx in train_rows]
    y_train = [y[idx].astype(str).tolist() for idx in train_rows]
    x_val = [sequence_features(x[idx], bins) for idx in val_rows]
    y_val = [y[idx].astype(str).tolist() for idx in val_rows]
    x_test = [sequence_features(x[idx], bins) for idx in test_rows]
    y_test = [y[idx].astype(str).tolist() for idx in test_rows]

    # A tiny fit catches broken pycrfsuite installations before the full run.
    dry = sklearn_crfsuite.CRF(
        algorithm="lbfgs", c1=0.1, c2=0.1, max_iterations=2,
        all_possible_transitions=True, verbose=False,
    )
    dry.fit([x_train[0][:10]], [y_train[0][:10]])

    crf = sklearn_crfsuite.CRF(
        algorithm="lbfgs",
        c1=0.1,
        c2=0.1,
        max_iterations=50,
        all_possible_transitions=True,
        verbose=False,
    )
    crf.fit(x_train, y_train)
    pred_val = crf.predict(x_val)
    pred_test = crf.predict(x_test)

    yv, pv = flatten(y_val), flatten(pred_val)
    yt, pt = flatten(y_test), flatten(pred_test)
    labels = sorted({label for seq in y_train for label in seq})

    def metrics(yt_: list[str], yp_: list[str]) -> dict[str, float]:
        return {
            "macro_f1": float(f1_score(yt_, yp_, labels=labels, average="macro", zero_division=0)),
            "balanced_accuracy": float(balanced_accuracy_score(yt_, yp_)),
            "mcc": float(matthews_corrcoef(yt_, yp_)),
            "cohen_kappa": float(cohen_kappa_score(yt_, yp_, labels=labels)),
        }

    result = {
        "method": "crf_structured",
        "algorithm": "linear-chain CRF (sklearn-crfsuite lbfgs)",
        "feature_columns": FEATURE_COLUMNS,
        "quantile_bins": 8,
        "window_cap_per_participant": CAP,
        "train_participants": "P001-P080",
        "validation_participants": "P081-P100",
        "locked_test_participants": "P101-P151",
        "n_train_windows": int(sum(len(z) for z in train_rows)),
        "n_validation_windows": int(sum(len(z) for z in val_rows)),
        "n_test_windows": int(sum(len(z) for z in test_rows)),
        "validation": metrics(yv, pv),
        "locked_test": metrics(yt, pt),
        "elapsed_seconds": float(time.time() - started),
        "prediction_files_written": False,
        "notes": (
            "Chronological 200-window/participant cap used for feasibility; "
            "this is a structured-baseline audit rather than a full-data retrain."
        ),
    }
    (OUT / "crf_audit_20260905.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    pd.DataFrame(
        [
            {"split": "validation", **result["validation"]},
            {"split": "locked_test", **result["locked_test"]},
        ]
    ).to_csv(OUT / "crf_audit_20260905.csv", index=False)
    lines = [
        "# Trained CRF structured-baseline audit (2026-09-05)",
        "",
        "A linear-chain CRF was trained on quantile-binned window features.",
        f"The audit uses {CAP} chronologically ordered, evenly spaced windows per participant: "
        "P001-P080 train, P081-P100 validation, P101-P151 locked test.",
        "No raw data or per-window predictions are included in the submission package.",
        "",
        "| Split | Macro-F1 | Balanced accuracy | MCC | Cohen's kappa |",
        "|---|---:|---:|---:|---:|",
    ]
    for split, vals in [("Validation", result["validation"]), ("Locked test", result["locked_test"])]:
        lines.append(
            f"| {split} | {vals['macro_f1']:.4f} | {vals['balanced_accuracy']:.4f} | "
            f"{vals['mcc']:.4f} | {vals['cohen_kappa']:.4f} |"
        )
    lines += [
        "",
        "The participant/window cap is disclosed because a full-data CRF retrain was not computationally proportionate to this audit.",
    ]
    (OUT / "crf_audit_20260905.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
