"""Permutation null control for the internal auxiliary-conflict diagnostic."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from make_aux_temporal_v2_ablation_table import conflict_support


ROOT = Path(__file__).resolve().parents[1]
BCM = ROOT / "results" / "bcm"
OUT = ROOT / "paper_artifacts" / "tables"
PRED = BCM / "rf_aux_temporal_conflict_priority_locked_20260630_164943_locked_test_predictions.npz"
N_PERM = 1000
SEED = 20260711


def rates(pred: np.ndarray, classes: np.ndarray, coarse_hard: np.ndarray, intensity_hard: np.ndarray, fine_to_coarse: np.ndarray, fine_to_intensity: np.ndarray) -> tuple[float, float, float]:
    idx = {c: i for i, c in enumerate(classes)}
    pred_idx = np.asarray([idx[p] for p in pred], dtype=np.int32)
    coarse = fine_to_coarse[pred_idx] != coarse_hard
    intensity = fine_to_intensity[pred_idx] != intensity_hard
    return float(coarse.mean()), float(intensity.mean()), float((coarse | intensity).mean())


def count_matches(pred_idx: np.ndarray, hard: np.ndarray, fine_classes: np.ndarray, aux_classes: np.ndarray) -> np.ndarray:
    fine_idx = pred_idx
    aux_idx = {c: i for i, c in enumerate(aux_classes)}
    hard_idx = np.asarray([aux_idx[c] for c in hard], dtype=np.int32)
    counts = np.zeros((len(fine_classes), len(aux_classes)), dtype=np.int64)
    np.add.at(counts, (fine_idx, hard_idx), 1)
    return counts


def main() -> None:
    z = np.load(PRED, allow_pickle=True)
    participant = z["participant"]
    classes = z["classes"]
    support = conflict_support(participant, classes)
    coarse_hard, intensity_hard, fine_to_coarse, fine_to_intensity = support
    pred = z["pred_rf_probability_temporal"]
    observed = rates(pred, classes, coarse_hard, intensity_hard, fine_to_coarse, fine_to_intensity)

    idx = {c: i for i, c in enumerate(classes)}
    pred_idx = np.asarray([idx[p] for p in pred], dtype=np.int32)
    coarse_classes = np.unique(coarse_hard)
    intensity_classes = np.unique(intensity_hard)
    coarse_counts = count_matches(pred_idx, coarse_hard, classes, coarse_classes)
    intensity_counts = count_matches(pred_idx, intensity_hard, classes, intensity_classes)
    coarse_map_idx = {c: i for i, c in enumerate(coarse_classes)}
    intensity_map_idx = {c: i for i, c in enumerate(intensity_classes)}
    observed_coarse_map = np.asarray([coarse_map_idx[c] for c in fine_to_coarse], dtype=np.int32)
    observed_intensity_map = np.asarray([intensity_map_idx[c] for c in fine_to_intensity], dtype=np.int32)
    observed_coarse = np.asarray([fine_to_coarse[p] != c for p, c in zip(pred_idx, coarse_hard)])
    observed_intensity = np.asarray([fine_to_intensity[p] != c for p, c in zip(pred_idx, intensity_hard)])
    observed = (float(observed_coarse.mean()), float(observed_intensity.mean()), float(np.mean(observed_coarse | observed_intensity)))

    rng = np.random.default_rng(SEED)
    coarse_perm = rng.integers(0, len(coarse_classes), size=(N_PERM, len(classes)))
    intensity_perm = rng.integers(0, len(intensity_classes), size=(N_PERM, len(classes)))
    coarse_match = coarse_counts[np.arange(len(classes))[None, :], coarse_perm].sum(axis=1)
    intensity_match = intensity_counts[np.arange(len(classes))[None, :], intensity_perm].sum(axis=1)
    # Any-conflict is computed directly for the observed mapping; null any-conflict
    # uses the joint row counts over the two auxiliary labels.
    coarse_label_idx = np.asarray([coarse_map_idx[c] for c in coarse_hard], dtype=np.int32)
    intensity_label_idx = np.asarray([intensity_map_idx[c] for c in intensity_hard], dtype=np.int32)
    joint = np.zeros((len(classes), len(coarse_classes), len(intensity_classes)), dtype=np.int64)
    np.add.at(joint, (pred_idx, coarse_label_idx, intensity_label_idx), 1)
    any_match = np.zeros(N_PERM, dtype=np.int64)
    for i in range(N_PERM):
        any_match[i] = joint[np.arange(len(classes))[:, None], coarse_perm[i][:, None], intensity_perm[i][:, None]].sum()
    perm = np.column_stack((1 - coarse_match / len(pred), 1 - intensity_match / len(pred), 1 - any_match / len(pred)))

    labels = ["coarse conflict", "intensity conflict", "any conflict"]
    rows = []
    for j, label in enumerate(labels):
        null_values = perm[:, j]
        obs = observed[j]
        rows.append(
            {
                "metric": label,
                "observed_CATD": obs,
                "null_mean": float(null_values.mean()),
                "null_ci_low": float(np.percentile(null_values, 2.5)),
                "null_ci_high": float(np.percentile(null_values, 97.5)),
                "observed_minus_null_mean": obs - float(null_values.mean()),
                "permutation_lower_tail_p": float((np.sum(null_values <= obs) + 1) / (N_PERM + 1)),
            }
        )
    df = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "table_null_mapping_conflict_control_v2.csv", index=False)
    lines = [
        "# Permutation null control for auxiliary conflict",
        "",
        f"Locked-test CATD predictions; mapping permutations: {N_PERM}; seed: {SEED}.",
        "Each permutation independently shuffles the fine-to-coarse and fine-to-intensity mapping vectors while leaving predictions and auxiliary hard labels unchanged.",
        "",
        "| metric | observed_CATD | null_mean | null_ci_low | null_ci_high | observed_minus_null_mean | permutation_lower_tail_p |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append("| {metric} | {observed_CATD:.4f} | {null_mean:.4f} | {null_ci_low:.4f} | {null_ci_high:.4f} | {observed_minus_null_mean:.4f} | {permutation_lower_tail_p:.4f} |".format(**row))
    lines += [
        "",
        "This null tests whether the observed conflict rates are lower than expected under arbitrary fine-to-auxiliary mappings. It does not provide an independent behavioural ground-truth validation.",
    ]
    (OUT / "table_null_mapping_conflict_control_v2.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(OUT / "table_null_mapping_conflict_control_v2.md")


if __name__ == "__main__":
    main()
