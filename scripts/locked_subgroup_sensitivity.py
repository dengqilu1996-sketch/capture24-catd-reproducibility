"""Locked-test subgroup sensitivity for CATD versus temporal-only decoding."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn import metrics

from make_aux_temporal_v2_ablation_table import conflict_support

ROOT = Path(__file__).resolve().parents[1]
BCM = ROOT / "results" / "bcm"
OUT = ROOT / "paper_artifacts" / "tables"
CATD = BCM / "rf_aux_temporal_conflict_priority_locked_20260630_164943_locked_test_predictions.npz"
TEMPORAL = BCM / "rf_temporal_decoding_locked_20260630_160032_locked_test_predictions.npz"


def main() -> None:
    z = np.load(CATD, allow_pickle=True)
    t = np.load(TEMPORAL, allow_pickle=True)
    participant = z["participant"]
    y = z["y_true"]
    classes = z["classes"]
    support = conflict_support(participant, classes)
    coarse_hard, intensity_hard, fine_to_coarse, fine_to_intensity = support
    idx = {c: i for i, c in enumerate(classes)}
    rows = []
    groups = {"P101-P125": (101, 125), "P126-P151": (126, 151)}
    for group, (lo, hi) in groups.items():
        mask = np.array([lo <= int(p[1:]) <= hi for p in participant])
        for method, pred in [("temporal-only decoding", t["pred_rf_probability_temporal"]), ("CATD (conflict-priority)", z["pred_rf_probability_temporal"])]:
            pi = np.asarray([idx[p] for p in pred[mask]], dtype=np.int32)
            coarse = fine_to_coarse[pi] != coarse_hard[mask]
            intensity = fine_to_intensity[pi] != intensity_hard[mask]
            rows.append({
                "group": group,
                "n_participants": int(len(np.unique(participant[mask]))),
                "method": method,
                "macro_f1": float(metrics.f1_score(y[mask], pred[mask], labels=classes, average="macro", zero_division=0)),
                "balanced_accuracy": float(metrics.balanced_accuracy_score(y[mask], pred[mask])),
                "intensity_conflict": float(intensity.mean()),
                "any_conflict": float((coarse | intensity).mean()),
            })
    df = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "table_locked_subgroup_sensitivity_v2.csv", index=False)
    lines = [
        "# Locked-test subgroup sensitivity",
        "",
        "The locked P101–P151 test set is split by participant ID into P101–P125 and P126–P151 after all parameters were selected. This is an evaluation-stability analysis, not a second tuning split.",
        "",
        "| group | n_participants | method | macro_f1 | balanced_accuracy | intensity_conflict | any_conflict |",
        "|---|---:|---|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append("| {group} | {n_participants} | {method} | {macro_f1:.4f} | {balanced_accuracy:.4f} | {intensity_conflict:.4f} | {any_conflict:.4f} |".format(**r))
    (OUT / "table_locked_subgroup_sensitivity_v2.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(OUT / "table_locked_subgroup_sensitivity_v2.md")


if __name__ == "__main__":
    main()
