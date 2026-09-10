"""Quantify whether internal auxiliary conflict is associated with fine-label errors."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from make_aux_temporal_v2_ablation_table import conflict_support
from bootstrap_aux_temporal_v2_metrics import markdown_table


ROOT = Path(__file__).resolve().parents[1]
BCM = ROOT / "results" / "bcm"
OUT = ROOT / "paper_artifacts" / "tables"
CATD = BCM / "rf_aux_temporal_conflict_priority_locked_20260630_164943_locked_test_predictions.npz"
TEMPORAL = BCM / "rf_temporal_decoding_locked_20260630_160032_locked_test_predictions.npz"


def phi(error: np.ndarray, conflict: np.ndarray) -> float:
    a = int(np.sum(error & conflict))
    b = int(np.sum(error & ~conflict))
    c = int(np.sum(~error & conflict))
    d = int(np.sum(~error & ~conflict))
    den = np.sqrt(float((a + b) * (c + d) * (a + c) * (b + d)))
    return float((a * d - b * c) / den) if den else 0.0


def one_method(name: str, pred: np.ndarray, y: np.ndarray, support: tuple[np.ndarray, ...], classes: np.ndarray) -> list[dict[str, object]]:
    coarse_hard, intensity_hard, fine_to_coarse, fine_to_intensity = support
    idx = {c: i for i, c in enumerate(classes)}
    pred_idx = np.asarray([idx[p] for p in pred], dtype=np.int32)
    error = pred != y
    conflicts = {
        "coarse": fine_to_coarse[pred_idx] != coarse_hard,
        "intensity": fine_to_intensity[pred_idx] != intensity_hard,
    }
    conflicts["any"] = conflicts["coarse"] | conflicts["intensity"]
    rows = []
    for label, conflict in conflicts.items():
        if conflict.any():
            error_conflict = float(error[conflict].mean())
        else:
            error_conflict = float("nan")
        if (~conflict).any():
            error_no_conflict = float(error[~conflict].mean())
        else:
            error_no_conflict = float("nan")
        rows.append(
            {
                "method": name,
                "conflict_type": label,
                "conflict_rate": float(conflict.mean()),
                "fine_error_rate_if_conflict": error_conflict,
                "fine_error_rate_if_no_conflict": error_no_conflict,
                "error_rate_difference": error_conflict - error_no_conflict,
                "phi_error_conflict": phi(error, conflict),
                "n_windows": int(len(error)),
            }
        )
    return rows


def main() -> None:
    catd = np.load(CATD, allow_pickle=True)
    temporal = np.load(TEMPORAL, allow_pickle=True)
    if not np.array_equal(catd["participant"], temporal["participant"]):
        raise ValueError("Prediction row order differs between CATD and temporal-only files")
    participant = catd["participant"]
    y = catd["y_true"]
    classes = catd["classes"]
    support = conflict_support(participant, classes)
    rows = []
    rows += one_method("CATD (conflict-priority)", catd["pred_rf_probability_temporal"], y, support, classes)
    rows += one_method("temporal-only decoding", temporal["pred_rf_probability_temporal"], y, support, classes)
    rows += one_method("RF+HMM", catd["pred_rf_official_hmm"], y, support, classes)
    df = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "table_conflict_error_association_v2.csv", index=False)
    lines = [
        "# Conflict–error association diagnostic",
        "",
        "This is an internal validity diagnostic, not an external behavioural ground-truth validation. Auxiliary hard predictions and fine-label predictions are evaluated on the same locked P101–P151 rows; all fine-to-auxiliary mappings are estimated from P001–P100.",
        "",
        markdown_table(df),
        "",
        "`error_rate_difference` is the fine-label error rate among conflicting windows minus the error rate among non-conflicting windows. `phi_error_conflict` is the 2×2 phi coefficient.",
    ]
    (OUT / "table_conflict_error_association_v2.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(OUT / "table_conflict_error_association_v2.md")


if __name__ == "__main__":
    main()
