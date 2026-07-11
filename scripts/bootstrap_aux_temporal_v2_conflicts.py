from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "paths.capture24.yaml"
BASELINE_DIR = ROOT / "results" / "baselines"
BCM_DIR = ROOT / "results" / "bcm"
OUT_DIR = ROOT / "paper_artifacts" / "tables"
PRED = BCM_DIR / "rf_aux_temporal_conflict_priority_locked_20260630_164943_locked_test_predictions.npz"
N_BOOT = 5000
SEED = 20260630


METHODS = {
    "RF baseline": "pred_rf_baseline",
    "RF+HMM": "pred_rf_official_hmm",
    "coarse/MET temporal decoding": "pred_rf_probability_temporal",
}


def latest(path: Path, pattern: str) -> Path:
    files = sorted(path.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files matching {pattern} in {path}")
    return files[-1]


def participant_range(start: int, end: int) -> list[str]:
    return [f"P{i:03d}" for i in range(start, end + 1)]


def row_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return x / np.maximum(x.sum(axis=1, keepdims=True), eps)


def map_matrix(
    source: np.ndarray,
    target: np.ndarray,
    source_classes: np.ndarray,
    target_classes: np.ndarray,
    mask: np.ndarray,
    smoothing: float = 1e-3,
) -> np.ndarray:
    mat = np.full((len(source_classes), len(target_classes)), smoothing, dtype="float64")
    src_index = {c: i for i, c in enumerate(source_classes)}
    tgt_index = {c: i for i, c in enumerate(target_classes)}
    for s, t in zip(source[mask], target[mask]):
        if s in src_index and t in tgt_index:
            mat[src_index[s], tgt_index[t]] += 1.0
    return row_normalize(mat)


def conflict_arrays(
    pred: np.ndarray,
    fine_classes: np.ndarray,
    coarse_hard: np.ndarray,
    intensity_hard: np.ndarray,
    fine_to_coarse: np.ndarray,
    fine_to_intensity: np.ndarray,
) -> dict[str, np.ndarray]:
    idx = {c: i for i, c in enumerate(fine_classes)}
    pred_idx = np.asarray([idx[p] for p in pred], dtype=np.int32)
    coarse = fine_to_coarse[pred_idx] != coarse_hard
    intensity = fine_to_intensity[pred_idx] != intensity_hard
    return {
        "coarse_conflict": coarse,
        "intensity_conflict": intensity,
        "any_conflict": coarse | intensity,
    }


def ci(values: np.ndarray) -> tuple[float, float]:
    return tuple(np.percentile(values, [2.5, 97.5]).astype(float))


def main() -> None:
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    prepared = Path(cfg["prepared_data"])
    if not prepared.exists():
        prepared = Path("D:/claude项目/capture24_work2/capture24/prepared_data")

    predz = np.load(PRED, allow_pickle=True)
    participant = predz["participant"]
    fine_classes = predz["classes"]

    y = np.load(prepared / "Y_WillettsSpecific2018.npy")
    y_coarse = np.load(prepared / "Y_Walmsley2020.npy")
    p_all = np.load(prepared / "P.npy")
    label_fields = np.load(latest(BCM_DIR, "capture24_derived_label_fields_*.npz"), allow_pickle=True)

    train100_valid_mask = np.isin(p_all, participant_range(1, 100)) & label_fields["valid_annotation_mask"]

    coarse_test = np.load(
        latest(BASELINE_DIR, "official_baselines_*rfcompat_xgbopt_full_Walmsley2020_predictions.npz"),
        allow_pickle=True,
    )
    intensity_test = np.load(latest(BCM_DIR, "auxiliary_xgb_*_y_met_intensity4_predictions.npz"), allow_pickle=True)

    if not np.array_equal(coarse_test["participant"], participant):
        raise ValueError("Coarse test participant order mismatch.")
    if not np.array_equal(intensity_test["participant"], participant):
        raise ValueError("Intensity test participant order mismatch.")

    coarse_classes = coarse_test["classes_rf_rfcompat_xgbopt_full"]
    intensity_classes = intensity_test["classes"]
    coarse_hard = coarse_classes[np.argmax(coarse_test["proba_rf_rfcompat_xgbopt_full"], axis=1)]
    intensity_hard = intensity_classes[np.argmax(intensity_test["proba"], axis=1)]

    fine_to_coarse = coarse_classes[
        np.argmax(map_matrix(y, y_coarse, fine_classes, coarse_classes, train100_valid_mask), axis=1)
    ]
    fine_to_intensity = intensity_classes[
        np.argmax(map_matrix(y, label_fields["y_met_intensity4"], fine_classes, intensity_classes, train100_valid_mask), axis=1)
    ]

    conflicts = {
        name: conflict_arrays(predz[key], fine_classes, coarse_hard, intensity_hard, fine_to_coarse, fine_to_intensity)
        for name, key in METHODS.items()
    }

    participants = np.unique(participant)
    part_indices = [np.where(participant == p)[0] for p in participants]
    rng = np.random.default_rng(SEED)

    point_rows = []
    for name, vals in conflicts.items():
        row = {"method": name}
        row.update({k: float(v.mean()) for k, v in vals.items()})
        point_rows.append(row)

    boot = {name: {k: [] for k in ("coarse_conflict", "intensity_conflict", "any_conflict")} for name in METHODS}
    deltas_vs_hmm = {k: [] for k in ("coarse_conflict", "intensity_conflict", "any_conflict")}
    deltas_vs_rf = {k: [] for k in ("coarse_conflict", "intensity_conflict", "any_conflict")}

    # For a rate, participant bootstrap can be computed from sampled numerator/denominator counts.
    participant_counts = {}
    for name, vals in conflicts.items():
        participant_counts[name] = {}
        for metric_name, arr in vals.items():
            num = np.asarray([arr[idx].sum() for idx in part_indices], dtype="float64")
            den = np.asarray([len(idx) for idx in part_indices], dtype="float64")
            participant_counts[name][metric_name] = (num, den)

    for _ in range(N_BOOT):
        sampled = rng.integers(0, len(participants), size=len(participants))
        scores = {}
        for name in METHODS:
            scores[name] = {}
            for metric_name in ("coarse_conflict", "intensity_conflict", "any_conflict"):
                num, den = participant_counts[name][metric_name]
                value = float(num[sampled].sum() / den[sampled].sum())
                scores[name][metric_name] = value
                boot[name][metric_name].append(value)
        for metric_name in ("coarse_conflict", "intensity_conflict", "any_conflict"):
            deltas_vs_hmm[metric_name].append(
                scores["coarse/MET temporal decoding"][metric_name] - scores["RF+HMM"][metric_name]
            )
            deltas_vs_rf[metric_name].append(
                scores["coarse/MET temporal decoding"][metric_name] - scores["RF baseline"][metric_name]
            )

    metric_rows = []
    for row in point_rows:
        out = dict(row)
        name = row["method"]
        for metric_name in ("coarse_conflict", "intensity_conflict", "any_conflict"):
            low, high = ci(np.asarray(boot[name][metric_name]))
            out[f"{metric_name}_ci_low"] = low
            out[f"{metric_name}_ci_high"] = high
        metric_rows.append(out)

    delta_rows = []
    point = {row["method"]: row for row in point_rows}
    for label, store, ref in [
        ("coarse/MET TD vs RF+HMM", deltas_vs_hmm, "RF+HMM"),
        ("coarse/MET TD vs RF baseline", deltas_vs_rf, "RF baseline"),
    ]:
        row = {"comparison": label}
        for metric_name in ("coarse_conflict", "intensity_conflict", "any_conflict"):
            point_delta = point["coarse/MET temporal decoding"][metric_name] - point[ref][metric_name]
            low, high = ci(np.asarray(store[metric_name]))
            row[f"delta_{metric_name}"] = point_delta
            row[f"delta_{metric_name}_ci_low"] = low
            row[f"delta_{metric_name}_ci_high"] = high
        delta_rows.append(row)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    metrics_df = pd.DataFrame(metric_rows)
    deltas_df = pd.DataFrame(delta_rows)
    metrics_df.to_csv(OUT_DIR / "table_aux_temporal_bootstrap_conflict_metrics_v2.csv", index=False)
    deltas_df.to_csv(OUT_DIR / "table_aux_temporal_bootstrap_conflict_deltas_v2.csv", index=False)

    md = OUT_DIR / "table_aux_temporal_bootstrap_conflict_ci_v2.md"
    with md.open("w", encoding="utf-8") as f:
        f.write("# Participant-Level Bootstrap Conflict CI: Auxiliary Temporal v2\n\n")
        f.write(f"Bootstrap unit: participant. Iterations: {N_BOOT}. Seed: {SEED}.\n\n")
        f.write("## Conflict Metrics\n\n")
        f.write(metrics_df.to_markdown(index=False, floatfmt=".4f"))
        f.write("\n\n## Deltas\n\n")
        f.write(deltas_df.to_markdown(index=False, floatfmt=".4f"))
        f.write("\n")

    print(f"Wrote {md}")


if __name__ == "__main__":
    main()
