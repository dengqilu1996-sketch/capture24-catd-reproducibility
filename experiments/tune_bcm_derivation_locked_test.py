"""Tune BCM only inside the derivation split, then apply once to held-out test.

Internal tuning split:
- P001-P080: internal train for fine/coarse/intensity XGBoost heads.
- P081-P100: internal validation for selecting alpha/beta and optional gates.

Locked test application:
- Uses the already saved full-derivation prediction/support artifacts for
  P101-P151 and applies the selected validation configuration once.

This script uses the clean official Capture-24 Classifier wrapper and does not
modify or import code from the previous project.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import platform
import subprocess
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


DEFAULT_PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = DEFAULT_PROJECT_ROOT / "configs" / "paths.capture24.yaml"


def load_config(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def git_short_sha(path: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "--short", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def latest(path: Path, pattern: str) -> Path:
    files = sorted(path.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files matching {pattern} in {path}")
    return files[-1]


def resolved_path(value: str | None, fallback: Path) -> Path:
    if value:
        candidate = Path(value)
        if candidate.exists():
            return candidate
    return fallback


def participant_range(start: int, end: int) -> list[str]:
    return [f"P{i:03d}" for i in range(start, end + 1)]


def row_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return x / np.maximum(x.sum(axis=1, keepdims=True), eps)


def mapping_matrix(
    source: np.ndarray,
    target: np.ndarray,
    source_classes: np.ndarray,
    target_classes: np.ndarray,
    mask: np.ndarray,
    smoothing: float,
) -> np.ndarray:
    mat = np.full((len(source_classes), len(target_classes)), smoothing, dtype="float64")
    src_index = {c: i for i, c in enumerate(source_classes)}
    tgt_index = {c: i for i, c in enumerate(target_classes)}
    for s, t in zip(source[mask], target[mask]):
        if s in src_index and t in tgt_index:
            mat[src_index[s], tgt_index[t]] += 1.0
    return row_normalize(mat)


def predict_from_proba(proba: np.ndarray, classes: np.ndarray) -> np.ndarray:
    return classes[np.argmax(proba, axis=1)]


def top_margin(proba: np.ndarray) -> np.ndarray:
    top2 = np.partition(proba, -2, axis=1)[:, -2:]
    return top2[:, 1] - top2[:, 0]


def pred_index(pred: np.ndarray, classes: np.ndarray) -> np.ndarray:
    index = {c: i for i, c in enumerate(classes)}
    return np.asarray([index[p] for p in pred], dtype=np.int32)


def calibrated_proba(
    fine_proba: np.ndarray,
    coarse_support: np.ndarray,
    intensity_support: np.ndarray,
    alpha: float,
    beta: float,
    eps: float,
) -> np.ndarray:
    score = fine_proba.astype("float64").copy()
    score *= np.power(np.maximum(coarse_support, eps), alpha)
    score *= np.power(np.maximum(intensity_support, eps), beta)
    return row_normalize(score).astype("float32")


def metric_row(y_true: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    return {
        "balanced_accuracy": float(metrics.balanced_accuracy_score(y_true, pred)),
        "macro_f1": float(metrics.f1_score(y_true, pred, average="macro", zero_division=0)),
        "weighted_f1": float(metrics.f1_score(y_true, pred, average="weighted", zero_division=0)),
        "mcc": float(metrics.matthews_corrcoef(y_true, pred)),
        "kappa": float(metrics.cohen_kappa_score(y_true, pred)),
    }


def bootstrap_ci_by_participant(
    y_true: np.ndarray,
    pred: np.ndarray,
    participant: np.ndarray,
    nboots: int,
    seed: int,
) -> dict[str, float]:
    if nboots <= 0:
        return {}
    rng = np.random.default_rng(seed)
    groups = np.unique(participant)
    idx = np.arange(len(y_true))
    values = []
    for _ in range(nboots):
        chosen = rng.choice(groups, size=len(groups), replace=True)
        boot_idx = np.concatenate([idx[participant == g] for g in chosen])
        values.append(
            [
                metrics.balanced_accuracy_score(y_true[boot_idx], pred[boot_idx]),
                metrics.f1_score(y_true[boot_idx], pred[boot_idx], average="macro", zero_division=0),
                metrics.matthews_corrcoef(y_true[boot_idx], pred[boot_idx]),
                metrics.cohen_kappa_score(y_true[boot_idx], pred[boot_idx]),
            ]
        )
    boot = np.asarray(values)
    out = {}
    for name, col in [("balanced_accuracy", 0), ("macro_f1", 1), ("mcc", 2), ("kappa", 3)]:
        out[f"{name}_ci_low"] = float(np.percentile(boot[:, col], 2.5))
        out[f"{name}_ci_high"] = float(np.percentile(boot[:, col], 97.5))
    return out


def paired_delta_ci_by_participant(
    y_true: np.ndarray,
    baseline_pred: np.ndarray,
    candidate_pred: np.ndarray,
    participant: np.ndarray,
    nboots: int,
    seed: int,
) -> dict[str, float]:
    if nboots <= 0:
        return {}
    rng = np.random.default_rng(seed)
    groups = np.unique(participant)
    idx = np.arange(len(y_true))
    values = []
    for _ in range(nboots):
        chosen = rng.choice(groups, size=len(groups), replace=True)
        boot_idx = np.concatenate([idx[participant == g] for g in chosen])
        values.append(
            [
                metrics.f1_score(y_true[boot_idx], candidate_pred[boot_idx], average="macro", zero_division=0)
                - metrics.f1_score(y_true[boot_idx], baseline_pred[boot_idx], average="macro", zero_division=0),
                metrics.balanced_accuracy_score(y_true[boot_idx], candidate_pred[boot_idx])
                - metrics.balanced_accuracy_score(y_true[boot_idx], baseline_pred[boot_idx]),
            ]
        )
    boot = np.asarray(values)
    return {
        "macro_f1_delta_ci_low": float(np.percentile(boot[:, 0], 2.5)),
        "macro_f1_delta_ci_high": float(np.percentile(boot[:, 0], 97.5)),
        "balanced_accuracy_delta_ci_low": float(np.percentile(boot[:, 1], 2.5)),
        "balanced_accuracy_delta_ci_high": float(np.percentile(boot[:, 1], 97.5)),
    }


def conflict_metrics(
    pred: np.ndarray,
    fine_classes: np.ndarray,
    coarse_hard: np.ndarray,
    intensity_hard: np.ndarray,
    fine_to_coarse: np.ndarray,
    fine_to_intensity: np.ndarray,
    coarse_support: np.ndarray,
    intensity_support: np.ndarray,
) -> dict[str, float]:
    idx = pred_index(pred, fine_classes)
    rows = np.arange(len(pred))
    return {
        "coarse_conflict_rate": float(np.mean(fine_to_coarse[idx] != coarse_hard)),
        "intensity_conflict_rate": float(np.mean(fine_to_intensity[idx] != intensity_hard)),
        "any_conflict_rate": float(np.mean((fine_to_coarse[idx] != coarse_hard) | (fine_to_intensity[idx] != intensity_hard))),
        "mean_coarse_support_for_pred": float(np.mean(coarse_support[rows, idx])),
        "mean_intensity_support_for_pred": float(np.mean(intensity_support[rows, idx])),
    }


def class_order(classifier) -> np.ndarray:
    label_encoder = getattr(classifier.window_classifier, "le", None)
    if label_encoder is not None and hasattr(label_encoder, "classes_"):
        return np.asarray(label_encoder.classes_)
    window_model = getattr(classifier.window_classifier, "model", None)
    if window_model is not None and hasattr(window_model, "classes_"):
        return np.asarray(window_model.classes_)
    raise ValueError("Could not recover class order from official classifier.")


def train_predict_xgb(
    official_code: Path,
    prepared_data: Path,
    X_train: np.ndarray,
    y_train: np.ndarray,
    p_train: np.ndarray,
    X_eval: np.ndarray,
    p_eval: np.ndarray,
    seed: int,
    model_n_jobs: int,
    xgb_device: str,
    xgb_tree_method: str,
):
    sys.path.insert(0, str(official_code))
    from classifier import Classifier

    classifier = Classifier(
        "xgb",
        seed,
        optimisedir=str(prepared_data / "__no_optimised_params__"),
        n_jobs=model_n_jobs,
        device=xgb_device,
        tree_method=xgb_tree_method,
    )
    classifier.fit(X_train, y_train, p_train)
    proba = classifier.predict_proba(X_eval, p_eval)
    classes = class_order(classifier)
    pred = predict_from_proba(proba, classes)
    return pred, proba, classes


def gate_mask(
    gate: str,
    margin: np.ndarray,
    threshold: float,
    coarse_conflict: np.ndarray,
    intensity_conflict: np.ndarray,
) -> np.ndarray:
    if gate == "none":
        return np.ones(len(margin), dtype=bool)
    if gate == "any_aux_conflict":
        conflict = coarse_conflict | intensity_conflict
    elif gate == "coarse_conflict":
        conflict = coarse_conflict
    elif gate == "intensity_conflict":
        conflict = intensity_conflict
    elif gate == "both_aux_conflicts":
        conflict = coarse_conflict & intensity_conflict
    else:
        raise ValueError(f"Unknown gate: {gate}")
    return conflict & (margin <= threshold)


def evaluate_candidates(
    y_true: np.ndarray,
    fine_classes: np.ndarray,
    fine_proba: np.ndarray,
    base_pred: np.ndarray,
    coarse_support: np.ndarray,
    intensity_support: np.ndarray,
    coarse_hard: np.ndarray,
    intensity_hard: np.ndarray,
    fine_to_coarse: np.ndarray,
    fine_to_intensity: np.ndarray,
    alphas: list[float],
    betas: list[float],
    gates: list[str],
    thresholds: list[float],
    eps: float,
) -> pd.DataFrame:
    base_idx = pred_index(base_pred, fine_classes)
    base_coarse_conflict = fine_to_coarse[base_idx] != coarse_hard
    base_intensity_conflict = fine_to_intensity[base_idx] != intensity_hard
    margin = top_margin(fine_proba)

    rows = []
    baseline_metrics = metric_row(y_true, base_pred)
    baseline_conflicts = conflict_metrics(
        base_pred,
        fine_classes,
        coarse_hard,
        intensity_hard,
        fine_to_coarse,
        fine_to_intensity,
        coarse_support,
        intensity_support,
    )
    rows.append(
        {
            "candidate_id": "baseline",
            "alpha": 0.0,
            "beta": 0.0,
            "gate": "baseline",
            "threshold": np.nan,
            "gate_rate": 0.0,
            **baseline_metrics,
            **baseline_conflicts,
            "macro_f1_delta_vs_baseline": 0.0,
            "balanced_accuracy_delta_vs_baseline": 0.0,
        }
    )

    for alpha in alphas:
        for beta in betas:
            proba = calibrated_proba(fine_proba, coarse_support, intensity_support, alpha, beta, eps)
            bcm_pred = predict_from_proba(proba, fine_classes)
            for gate in gates:
                gate_thresholds = [np.nan] if gate == "none" else thresholds
                for threshold in gate_thresholds:
                    apply = gate_mask(
                        gate,
                        margin,
                        float(threshold) if not np.isnan(threshold) else 1.0,
                        base_coarse_conflict,
                        base_intensity_conflict,
                    )
                    pred = base_pred.copy()
                    pred[apply] = bcm_pred[apply]
                    m = metric_row(y_true, pred)
                    c = conflict_metrics(
                        pred,
                        fine_classes,
                        coarse_hard,
                        intensity_hard,
                        fine_to_coarse,
                        fine_to_intensity,
                        coarse_support,
                        intensity_support,
                    )
                    rows.append(
                        {
                            "candidate_id": f"a{alpha:g}_b{beta:g}_{gate}_{threshold if not np.isnan(threshold) else 'all'}",
                            "alpha": alpha,
                            "beta": beta,
                            "gate": gate,
                            "threshold": threshold,
                            "gate_rate": float(apply.mean()),
                            **m,
                            **c,
                            "macro_f1_delta_vs_baseline": m["macro_f1"] - baseline_metrics["macro_f1"],
                            "balanced_accuracy_delta_vs_baseline": m["balanced_accuracy"] - baseline_metrics["balanced_accuracy"],
                        }
                    )
    return pd.DataFrame(rows).sort_values(
        ["macro_f1", "balanced_accuracy", "any_conflict_rate", "gate_rate"],
        ascending=[False, False, True, True],
    )


def apply_candidate(
    selected: pd.Series,
    fine_classes: np.ndarray,
    fine_proba: np.ndarray,
    base_pred: np.ndarray,
    coarse_support: np.ndarray,
    intensity_support: np.ndarray,
    coarse_hard: np.ndarray,
    intensity_hard: np.ndarray,
    fine_to_coarse: np.ndarray,
    fine_to_intensity: np.ndarray,
    eps: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    proba = calibrated_proba(
        fine_proba,
        coarse_support,
        intensity_support,
        float(selected["alpha"]),
        float(selected["beta"]),
        eps,
    )
    bcm_pred = predict_from_proba(proba, fine_classes)
    base_idx = pred_index(base_pred, fine_classes)
    apply = gate_mask(
        str(selected["gate"]),
        top_margin(fine_proba),
        float(selected["threshold"]) if not pd.isna(selected["threshold"]) else 1.0,
        fine_to_coarse[base_idx] != coarse_hard,
        fine_to_intensity[base_idx] != intensity_hard,
    )
    pred = base_pred.copy()
    pred[apply] = bcm_pred[apply]
    gated_proba = fine_proba.copy()
    gated_proba[apply] = proba[apply]
    return pred, gated_proba, apply


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--label-fields-npz", type=Path, default=None)
    parser.add_argument("--locked-test-bcm-npz", type=Path, default=None)
    parser.add_argument("--coarse-test-npz", type=Path, default=None)
    parser.add_argument("--intensity-test-npz", type=Path, default=None)
    parser.add_argument("--coarse-test-proba-key", default="proba_xgb_gpu_proba")
    parser.add_argument("--alphas", default="0.5,1.0,1.5,2.0")
    parser.add_argument("--betas", default="0.5,1.0,1.5,2.0")
    parser.add_argument("--thresholds", default="0.1,0.2,0.3,0.5,1.0")
    parser.add_argument("--gates", default="none,any_aux_conflict,coarse_conflict,intensity_conflict,both_aux_conflicts")
    parser.add_argument("--run-id-prefix", default="bcm_derivation_tuned")
    parser.add_argument("--project-root", type=Path, default=None)
    parser.add_argument("--prepared-data", type=Path, default=None)
    parser.add_argument("--official-code", type=Path, default=None)
    parser.add_argument("--baseline-results", type=Path, default=None)
    parser.add_argument("--bcm-results", type=Path, default=None)
    parser.add_argument("--logs-dir", type=Path, default=None)
    parser.add_argument("--model-n-jobs", type=int, default=4)
    parser.add_argument("--xgb-device", default="cuda")
    parser.add_argument("--xgb-tree-method", default="hist")
    parser.add_argument("--nboots", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eps", type=float, default=1e-12)
    args = parser.parse_args()

    cfg = load_config(args.config)
    project_root = args.project_root or resolved_path(cfg.get("project_root"), DEFAULT_PROJECT_ROOT)
    prepared_data = args.prepared_data or resolved_path(cfg.get("prepared_data"), project_root / "data" / "prepared_data_official_repro")
    official_code = args.official_code or resolved_path(
        cfg.get("capture24_official_code"),
        project_root / "capture24_project" / "official_code" / "capture24",
    )
    baseline_dir = args.baseline_results or resolved_path(cfg.get("baseline_results"), project_root / "results" / "baselines")
    bcm_dir = args.bcm_results or resolved_path(cfg.get("bcm_results"), project_root / "results" / "bcm")
    logs_dir = args.logs_dir or resolved_path(cfg.get("logs"), project_root / "logs")
    paper_tables = project_root / "paper_artifacts" / "tables"
    bcm_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    paper_tables.mkdir(parents=True, exist_ok=True)

    label_fields_npz = args.label_fields_npz or latest(bcm_dir, "capture24_derived_label_fields_*.npz")
    locked_test_bcm_npz = args.locked_test_bcm_npz or latest(bcm_dir, "posthoc_bcm_v1_predictions_*.npz")
    coarse_test_npz = args.coarse_test_npz or latest(baseline_dir, "official_baselines_*gpu_proba_Walmsley2020_predictions.npz")
    intensity_test_npz = args.intensity_test_npz or latest(bcm_dir, "auxiliary_xgb_*_y_met_intensity4_predictions.npz")

    labels = np.load(label_fields_npz, allow_pickle=True)
    X = pd.read_pickle(prepared_data / "X_feats.pkl").values
    P = np.load(prepared_data / "P.npy")
    y_fine = labels["y_willetts_specific2018"]
    y_coarse = labels["y_walmsley2020"]
    y_intensity = labels["y_met_intensity4"]
    valid_intensity = labels["valid_annotation_mask"]

    internal_train_mask = np.isin(P, participant_range(1, 80))
    internal_val_mask = np.isin(P, participant_range(81, 100))
    full_derivation_mask = np.isin(P, participant_range(1, 100))
    if not internal_train_mask.any() or not internal_val_mask.any():
        raise ValueError("Internal derivation split masks are empty.")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"{args.run_id_prefix}_{timestamp}"
    start_time = time.time()

    X_train = X[internal_train_mask]
    P_train = P[internal_train_mask]
    X_val = X[internal_val_mask]
    P_val = P[internal_val_mask]
    y_val = y_fine[internal_val_mask]

    fine_pred, fine_proba, fine_classes = train_predict_xgb(
        official_code,
        prepared_data,
        X_train,
        y_fine[internal_train_mask],
        P_train,
        X_val,
        P_val,
        args.seed,
        args.model_n_jobs,
        args.xgb_device,
        args.xgb_tree_method,
    )
    coarse_pred, coarse_proba, coarse_classes = train_predict_xgb(
        official_code,
        prepared_data,
        X_train,
        y_coarse[internal_train_mask],
        P_train,
        X_val,
        P_val,
        args.seed,
        args.model_n_jobs,
        args.xgb_device,
        args.xgb_tree_method,
    )
    intensity_train_mask = internal_train_mask & valid_intensity
    intensity_pred, intensity_proba, intensity_classes = train_predict_xgb(
        official_code,
        prepared_data,
        X[intensity_train_mask],
        y_intensity[intensity_train_mask],
        P[intensity_train_mask],
        X_val,
        P_val,
        args.seed,
        args.model_n_jobs,
        args.xgb_device,
        args.xgb_tree_method,
    )

    internal_map_mask = internal_train_mask & valid_intensity
    fine_coarse_map = mapping_matrix(y_fine, y_coarse, fine_classes, coarse_classes, internal_map_mask, smoothing=1e-3)
    fine_intensity_map = mapping_matrix(y_fine, y_intensity, fine_classes, intensity_classes, internal_map_mask, smoothing=1e-3)
    val_coarse_support = coarse_proba @ fine_coarse_map.T
    val_intensity_support = intensity_proba @ fine_intensity_map.T
    val_fine_to_coarse = coarse_classes[np.argmax(fine_coarse_map, axis=1)]
    val_fine_to_intensity = intensity_classes[np.argmax(fine_intensity_map, axis=1)]

    alphas = [float(x.strip()) for x in args.alphas.split(",") if x.strip()]
    betas = [float(x.strip()) for x in args.betas.split(",") if x.strip()]
    thresholds = [float(x.strip()) for x in args.thresholds.split(",") if x.strip()]
    gates = [x.strip() for x in args.gates.split(",") if x.strip()]
    validation_candidates = evaluate_candidates(
        y_val,
        fine_classes,
        fine_proba,
        fine_pred,
        val_coarse_support,
        val_intensity_support,
        coarse_pred,
        intensity_pred,
        val_fine_to_coarse,
        val_fine_to_intensity,
        alphas,
        betas,
        gates,
        thresholds,
        args.eps,
    )
    selected = validation_candidates.iloc[0]

    locked = np.load(locked_test_bcm_npz, allow_pickle=True)
    coarse_test = np.load(coarse_test_npz)
    intensity_test = np.load(intensity_test_npz)
    test_participant = locked["participant"]
    test_y_true = locked["y_true"]
    test_fine_classes = locked["fine_classes"]
    test_fine_proba = locked["proba_xgb_fine_baseline"]
    test_base_pred = locked["pred_xgb_fine_baseline"]
    test_coarse_support = locked["coarse_support"]
    test_intensity_support = locked["intensity_support"]
    if not np.array_equal(test_participant, coarse_test["participant"]) or not np.array_equal(test_participant, intensity_test["participant"]):
        raise ValueError("Locked test artifact participant order mismatch.")
    if not np.array_equal(test_fine_classes, fine_classes):
        # XGBoost class order should match, but fail loudly if it does not.
        raise ValueError("Internal-validation and locked-test fine class orders differ.")

    full_map_mask = full_derivation_mask & valid_intensity
    test_coarse_classes = locked["coarse_classes"]
    test_intensity_classes = locked["intensity_classes"]
    full_fine_coarse_map = mapping_matrix(y_fine, y_coarse, test_fine_classes, test_coarse_classes, full_map_mask, smoothing=1e-3)
    full_fine_intensity_map = mapping_matrix(y_fine, y_intensity, test_fine_classes, test_intensity_classes, full_map_mask, smoothing=1e-3)
    test_fine_to_coarse = test_coarse_classes[np.argmax(full_fine_coarse_map, axis=1)]
    test_fine_to_intensity = test_intensity_classes[np.argmax(full_fine_intensity_map, axis=1)]
    test_coarse_hard = predict_from_proba(coarse_test[args.coarse_test_proba_key], test_coarse_classes)
    test_intensity_hard = predict_from_proba(intensity_test["proba"], test_intensity_classes)

    tuned_pred, tuned_proba, tuned_apply_mask = apply_candidate(
        selected,
        test_fine_classes,
        test_fine_proba,
        test_base_pred,
        test_coarse_support,
        test_intensity_support,
        test_coarse_hard,
        test_intensity_hard,
        test_fine_to_coarse,
        test_fine_to_intensity,
        args.eps,
    )

    metrics_rows = []
    for method, pred in [("xgb_fine_baseline", test_base_pred), ("bcm_derivation_tuned", tuned_pred)]:
        row = {"method": method, **metric_row(test_y_true, pred)}
        row.update(bootstrap_ci_by_participant(test_y_true, pred, test_participant, args.nboots, args.seed))
        row.update(
            conflict_metrics(
                pred,
                test_fine_classes,
                test_coarse_hard,
                test_intensity_hard,
                test_fine_to_coarse,
                test_fine_to_intensity,
                test_coarse_support,
                test_intensity_support,
            )
        )
        row["gate_rate"] = 0.0 if method == "xgb_fine_baseline" else float(tuned_apply_mask.mean())
        row["corrected"] = 0 if method == "xgb_fine_baseline" else int(((test_base_pred != test_y_true) & (tuned_pred == test_y_true)).sum())
        row["harmed"] = 0 if method == "xgb_fine_baseline" else int(((test_base_pred == test_y_true) & (tuned_pred != test_y_true)).sum())
        metrics_rows.append(row)
    metrics_df = pd.DataFrame(metrics_rows)
    tuned_metrics = metrics_df[metrics_df["method"] == "bcm_derivation_tuned"].iloc[0].to_dict()
    base_metrics = metrics_df[metrics_df["method"] == "xgb_fine_baseline"].iloc[0].to_dict()
    delta_row = {
        "comparison": "bcm_derivation_tuned - xgb_fine_baseline",
        "macro_f1_delta": tuned_metrics["macro_f1"] - base_metrics["macro_f1"],
        "balanced_accuracy_delta": tuned_metrics["balanced_accuracy"] - base_metrics["balanced_accuracy"],
        "coarse_conflict_delta": tuned_metrics["coarse_conflict_rate"] - base_metrics["coarse_conflict_rate"],
        "intensity_conflict_delta": tuned_metrics["intensity_conflict_rate"] - base_metrics["intensity_conflict_rate"],
    }
    delta_row.update(paired_delta_ci_by_participant(test_y_true, test_base_pred, tuned_pred, test_participant, args.nboots, args.seed + 1))

    validation_npz_path = bcm_dir / f"{run_id}_internal_validation_predictions.npz"
    np.savez_compressed(
        validation_npz_path,
        participant=P_val,
        y_true=y_val,
        fine_classes=fine_classes,
        coarse_classes=coarse_classes,
        intensity_classes=intensity_classes,
        proba_fine=fine_proba.astype("float32"),
        pred_fine=fine_pred,
        proba_coarse=coarse_proba.astype("float32"),
        pred_coarse=coarse_pred,
        proba_intensity=intensity_proba.astype("float32"),
        pred_intensity=intensity_pred,
        coarse_support=val_coarse_support.astype("float32"),
        intensity_support=val_intensity_support.astype("float32"),
    )
    locked_pred_path = bcm_dir / f"{run_id}_locked_test_predictions.npz"
    np.savez_compressed(
        locked_pred_path,
        participant=test_participant,
        y_true=test_y_true,
        fine_classes=test_fine_classes,
        pred_xgb_fine_baseline=test_base_pred,
        pred_bcm_derivation_tuned=tuned_pred,
        proba_bcm_derivation_tuned=tuned_proba.astype("float32"),
        tuned_apply_mask=tuned_apply_mask,
    )
    validation_candidates_path = bcm_dir / f"{run_id}_validation_candidates.csv"
    selected_path = bcm_dir / f"{run_id}_selected_config.csv"
    metrics_path = bcm_dir / f"{run_id}_locked_test_metrics.csv"
    deltas_path = bcm_dir / f"{run_id}_locked_test_deltas.csv"
    metadata_path = logs_dir / f"{run_id}.json"
    report_path = bcm_dir / f"{args.run_id_prefix}_report.md"
    table_copy = paper_tables / "table_bcm_derivation_tuned_locked_test.csv"

    validation_candidates.to_csv(validation_candidates_path, index=False)
    selected.to_frame().T.to_csv(selected_path, index=False)
    metrics_df.to_csv(metrics_path, index=False)
    metrics_df.to_csv(table_copy, index=False)
    pd.DataFrame([delta_row]).to_csv(deltas_path, index=False)

    metadata = {
        "run_id": run_id,
        "created_at": timestamp,
        "python": sys.executable,
        "python_version": platform.python_version(),
        "official_code": str(official_code),
        "official_code_commit": git_short_sha(official_code),
        "prepared_data": str(prepared_data),
        "label_fields_npz": str(label_fields_npz),
        "locked_test_bcm_npz": str(locked_test_bcm_npz),
        "coarse_test_npz": str(coarse_test_npz),
        "intensity_test_npz": str(intensity_test_npz),
        "internal_train": "P001-P080",
        "internal_validation": "P081-P100",
        "locked_test": "P101-P151",
        "xgb_device": args.xgb_device,
        "xgb_tree_method": args.xgb_tree_method,
        "coarse_test_proba_key": args.coarse_test_proba_key,
        "model_n_jobs": args.model_n_jobs,
        "nboots": args.nboots,
        "seed": args.seed,
        "elapsed_seconds": time.time() - start_time,
        "selected_config": selected.to_dict(),
        "outputs": {
            "validation_predictions": str(validation_npz_path),
            "locked_test_predictions": str(locked_pred_path),
            "validation_candidates": str(validation_candidates_path),
            "selected_config": str(selected_path),
            "locked_test_metrics": str(metrics_path),
            "locked_test_deltas": str(deltas_path),
            "paper_table_copy": str(table_copy),
            "report": str(report_path),
        },
    }
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    with report_path.open("a", encoding="utf-8") as f:
        f.write(f"\n\n## {run_id}\n\n")
        f.write("- Internal tuning split: P001-P080 train, P081-P100 validation.\n")
        f.write("- Locked held-out test: P101-P151, using the selected validation configuration once.\n")
        f.write(f"- Selected config CSV: `{selected_path}`\n")
        f.write(f"- Validation candidates: `{validation_candidates_path}`\n")
        f.write(f"- Locked test metrics: `{metrics_path}`\n")
        f.write(f"- Locked test deltas: `{deltas_path}`\n")
        f.write(f"- Locked test predictions: `{locked_pred_path}`\n")
        f.write(f"- Metadata: `{metadata_path}`\n\n")
        f.write("### Selected Validation Configuration\n\n")
        f.write(selected.to_frame().T.to_markdown(index=False))
        f.write("\n\n### Locked Test Metrics\n\n")
        f.write(metrics_df.to_markdown(index=False))
        f.write("\n\n### Locked Test Paired Delta\n\n")
        f.write(pd.DataFrame([delta_row]).to_markdown(index=False))
        f.write("\n")

    print(report_path)
    print(selected.to_frame().T.to_string(index=False))
    print(metrics_df.to_string(index=False))
    print(pd.DataFrame([delta_row]).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
