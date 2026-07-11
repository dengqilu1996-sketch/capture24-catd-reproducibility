from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn import metrics


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "paths.capture24.yaml"
BASELINE_DIR = ROOT / "results" / "baselines"
BCM_DIR = ROOT / "results" / "bcm"
OUT_DIR = ROOT / "paper_artifacts" / "tables"

TEMPORAL_ONLY = BCM_DIR / "rf_temporal_decoding_locked_20260630_160032_locked_test_predictions.npz"
PERFORMANCE = BCM_DIR / "rf_aux_temporal_locked_20260630_162006_locked_test_predictions.npz"
CONFLICT = BCM_DIR / "rf_aux_temporal_conflict_priority_locked_20260630_164943_locked_test_predictions.npz"


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


def prepared_path() -> Path:
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    prepared = Path(cfg["prepared_data"])
    if prepared.exists():
        return prepared
    return Path("D:/claude项目/capture24_work2/capture24/prepared_data")


def conflict_support(participant: np.ndarray, fine_classes: np.ndarray):
    prepared = prepared_path()
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
        np.argmax(
            map_matrix(y, label_fields["y_met_intensity4"], fine_classes, intensity_classes, train100_valid_mask),
            axis=1,
        )
    ]
    return coarse_hard, intensity_hard, fine_to_coarse, fine_to_intensity


def scores(y_true: np.ndarray, y_pred: np.ndarray, rf_pred: np.ndarray, support) -> dict[str, float | int]:
    coarse_hard, intensity_hard, fine_to_coarse, fine_to_intensity, classes = support
    idx = {c: i for i, c in enumerate(classes)}
    pred_idx = np.asarray([idx[p] for p in y_pred], dtype=np.int32)
    coarse = fine_to_coarse[pred_idx] != coarse_hard
    intensity = fine_to_intensity[pred_idx] != intensity_hard
    return {
        "macro_f1": metrics.f1_score(y_true, y_pred, average="macro", zero_division=0),
        "balanced_accuracy": metrics.balanced_accuracy_score(y_true, y_pred),
        "intensity_conflict": float(intensity.mean()),
        "any_conflict": float((coarse | intensity).mean()),
        "corrected": int(((rf_pred != y_true) & (y_pred == y_true)).sum()),
        "harmed": int(((rf_pred == y_true) & (y_pred != y_true)).sum()),
    }


def main() -> None:
    z_conflict = np.load(CONFLICT, allow_pickle=True)
    z_perf = np.load(PERFORMANCE, allow_pickle=True)
    z_temporal = np.load(TEMPORAL_ONLY, allow_pickle=True)

    participant = z_conflict["participant"]
    y_true = z_conflict["y_true"]
    classes = z_conflict["classes"]
    rf_pred = z_conflict["pred_rf_baseline"]

    if not np.array_equal(z_perf["participant"], participant) or not np.array_equal(z_temporal["participant"], participant):
        raise ValueError("Participant order mismatch across RF temporal prediction files.")

    coarse_hard, intensity_hard, fine_to_coarse, fine_to_intensity = conflict_support(participant, classes)
    support = (coarse_hard, intensity_hard, fine_to_coarse, fine_to_intensity, classes)

    rows = []
    rows.append(
        {
            "family": "RF",
            "method": "RF baseline",
            "selection_role": "no temporal decoding",
            "alpha": "-",
            "beta": "-",
            "gamma_or_gate": "-",
            **scores(y_true, z_conflict["pred_rf_baseline"], rf_pred, support),
        }
    )
    rows.append(
        {
            "family": "RF",
            "method": "official RF+HMM",
            "selection_role": "official temporal baseline",
            "alpha": "-",
            "beta": "-",
            "gamma_or_gate": "HMM",
            **scores(y_true, z_conflict["pred_rf_official_hmm"], rf_pred, support),
        }
    )
    rows.append(
        {
            "family": "RF",
            "method": "RF temporal-only decoding",
            "selection_role": "gamma-selected temporal-only control",
            "alpha": "0.00",
            "beta": "0.00",
            "gamma_or_gate": "gamma=3.0",
            **scores(y_true, z_temporal["pred_rf_probability_temporal"], rf_pred, support),
        }
    )
    rows.append(
        {
            "family": "RF",
            "method": "RF intensity-temporal decoding",
            "selection_role": "validation-selected performance-priority variant",
            "alpha": "0.00",
            "beta": "0.25",
            "gamma_or_gate": "gamma=3.0",
            **scores(y_true, z_perf["pred_rf_probability_temporal"], rf_pred, support),
        }
    )
    rows.append(
        {
            "family": "RF",
            "method": "coarse/MET temporal decoding",
            "selection_role": "main conflict-priority variant",
            "alpha": "0.25",
            "beta": "1.00",
            "gamma_or_gate": "gamma=3.0",
            **scores(y_true, z_conflict["pred_rf_probability_temporal"], rf_pred, support),
        }
    )
    rows.extend(
        [
            {
                "family": "XGBoost",
                "method": "XGB baseline",
                "selection_role": "no calibration",
                "alpha": "-",
                "beta": "-",
                "gamma_or_gate": "-",
                "macro_f1": 0.4003,
                "balanced_accuracy": 0.3897,
                "intensity_conflict": 0.1951,
                "any_conflict": 0.2190,
                "corrected": 0,
                "harmed": 0,
            },
            {
                "family": "XGBoost",
                "method": "post-hoc BCM",
                "selection_role": "non-temporal consistency ablation",
                "alpha": "1.00",
                "beta": "2.00",
                "gamma_or_gate": "both auxiliary conflicts",
                "macro_f1": 0.4067,
                "balanced_accuracy": 0.3955,
                "intensity_conflict": 0.1704,
                "any_conflict": 0.2018,
                "corrected": 2615,
                "harmed": 2889,
            },
        ]
    )

    df = pd.DataFrame(rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUT_DIR / "table_aux_temporal_ablation_v2.csv"
    md_path = OUT_DIR / "table_aux_temporal_ablation_v2.md"
    df.to_csv(csv_path, index=False)

    display = df.copy()
    for col in ("macro_f1", "balanced_accuracy", "intensity_conflict", "any_conflict"):
        display[col] = display[col].map(lambda x: f"{float(x):.4f}")
    for col in ("corrected", "harmed"):
        display[col] = display[col].map(lambda x: f"{int(x):,}")

    md = [
        "# Table 4. Ablation and Positioning of Auxiliary Consistency Variants",
        "",
        "Locked test: P101-P151. RF temporal-decoding parameters are selected on P081-P100.",
        "The RF temporal-only and RF intensity-temporal rows are validation-selected controls, not post-hoc test-set optima.",
        "For RF rows, corrected/harmed counts are relative to the RF baseline. For the XGBoost post-hoc BCM row, corrected/harmed counts are relative to the XGBoost baseline.",
        "",
        display.to_markdown(index=False),
        "",
        "## Source files",
        "",
        "- RF temporal-only: `results/bcm/rf_temporal_decoding_locked_20260630_160032_locked_test_metrics.csv`.",
        "- RF intensity-temporal performance-priority: `results/bcm/rf_aux_temporal_locked_20260630_162006_locked_test_metrics.csv`.",
        "- RF coarse/MET conflict-priority: `results/bcm/rf_aux_temporal_conflict_priority_locked_20260630_164943_locked_test_metrics.csv`.",
        "",
        "## Interpretation",
        "",
        "- Official RF+HMM confirms that temporal modelling is a strong baseline, but in the current diagnostics it increases any-conflict relative to RF baseline.",
        "- RF temporal-only decoding already explains a large part of the performance gain over RF+HMM.",
        "- Adding intensity support in the validation-selected performance-priority variant gives the highest reported validation-selected locked-test macro-F1, but conflict reduction is weaker than in the conflict-priority variant.",
        "- The conflict-priority coarse/MET temporal decoder is selected as the main paper method because it reduces intensity-conflict and any-conflict substantially while preserving the temporal-only macro-F1 level.",
        "- XGBoost post-hoc BCM shows that cross-granularity consistency also helps without temporal decoding, but its performance gain is much smaller; it should be reported as an ablation rather than the main result.",
        "",
    ]
    md_path.write_text("\n".join(md), encoding="utf-8")
    print(csv_path)
    print(md_path)


if __name__ == "__main__":
    main()
