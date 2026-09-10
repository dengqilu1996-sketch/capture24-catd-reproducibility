"""Formal RF auxiliary-aware temporal decoding with derivation-only tuning.

Internal validation:
- Train official RF on P001-P080.
- Predict probabilities on P081-P100.
- Select alpha, beta, gamma for participant-wise probability Viterbi decoding.

Locked test:
- Use saved full-derivation RF probabilities on P101-P151.
- Apply the selected gamma once.
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import os
import sys

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


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def latest(path: Path, pattern: str) -> Path:
    files = sorted(path.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files matching {pattern} in {path}")
    return files[-1]


def participant_range(start: int, end: int) -> list[str]:
    return [f"P{i:03d}" for i in range(start, end + 1)]


def row_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return x / np.maximum(x.sum(axis=1, keepdims=True), eps)


def transition_matrix(y: np.ndarray, participant: np.ndarray, classes: np.ndarray, train_ids: list[str], smoothing: float) -> np.ndarray:
    mask = np.isin(participant, train_ids)
    index = {c: i for i, c in enumerate(classes)}
    mat = np.full((len(classes), len(classes)), smoothing, dtype="float64")
    for p in np.unique(participant[mask]):
        seq = y[mask & (participant == p)]
        if len(seq) < 2:
            continue
        for a, b in zip(seq[:-1], seq[1:]):
            if a in index and b in index:
                mat[index[a], index[b]] += 1.0
    return row_normalize(mat)


def viterbi_decode_proba(proba: np.ndarray, participant: np.ndarray, classes: np.ndarray, trans: np.ndarray, gamma: float, eps: float) -> np.ndarray:
    log_trans = gamma * np.log(np.maximum(trans, eps))
    pred_idx = np.zeros(len(participant), dtype=np.int32)
    for p in np.unique(participant):
        idx = np.where(participant == p)[0]
        emit = np.log(np.maximum(proba[idx], eps))
        n, k = emit.shape
        dp = np.zeros((n, k), dtype="float64")
        back = np.zeros((n, k), dtype=np.int32)
        dp[0] = emit[0]
        for t in range(1, n):
            scores = dp[t - 1][:, None] + log_trans
            back[t] = np.argmax(scores, axis=0)
            dp[t] = emit[t] + scores[back[t], np.arange(k)]
        path = np.zeros(n, dtype=np.int32)
        path[-1] = int(np.argmax(dp[-1]))
        for t in range(n - 2, -1, -1):
            path[t] = back[t + 1, path[t + 1]]
        pred_idx[idx] = path
    return classes[pred_idx]


def metric_row(y_true: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    return {
        "balanced_accuracy": float(metrics.balanced_accuracy_score(y_true, pred)),
        "macro_f1": float(metrics.f1_score(y_true, pred, average="macro", zero_division=0)),
        "weighted_f1": float(metrics.f1_score(y_true, pred, average="weighted", zero_division=0)),
        "mcc": float(metrics.matthews_corrcoef(y_true, pred)),
        "kappa": float(metrics.cohen_kappa_score(y_true, pred)),
    }


def conflict_row(
    pred: np.ndarray,
    fine_classes: np.ndarray,
    coarse_hard: np.ndarray,
    intensity_hard: np.ndarray,
    fine_to_coarse: np.ndarray,
    fine_to_intensity: np.ndarray,
) -> dict[str, float]:
    idx = {c: i for i, c in enumerate(fine_classes)}
    pred_idx = np.asarray([idx[p] for p in pred], dtype=np.int32)
    coarse_conflict = fine_to_coarse[pred_idx] != coarse_hard
    intensity_conflict = fine_to_intensity[pred_idx] != intensity_hard
    return {
        "coarse_conflict_rate": float(coarse_conflict.mean()),
        "intensity_conflict_rate": float(intensity_conflict.mean()),
        "any_conflict_rate": float((coarse_conflict | intensity_conflict).mean()),
    }


def parse_grid(value: str) -> list[float]:
    return [float(x.strip()) for x in value.split(",") if x.strip()]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--alphas", default="0,0.25,0.5,1")
    parser.add_argument("--betas", default="0,0.25,0.5,1")
    parser.add_argument("--gammas", default="0,0.25,0.5,0.75,1,1.5,2,2.5,3,4,5,6,8,10")
    parser.add_argument("--smoothing", type=float, default=1e-3)
    parser.add_argument("--model-n-jobs", type=int, default=12)
    parser.add_argument("--n-estimators-override", type=int, default=None, help="Debug only; leave unset for formal RF.")
    parser.add_argument(
        "--selection-mode",
        choices=["performance", "conflict_priority"],
        default="performance",
        help="performance maximizes validation macro-F1; conflict_priority keeps high macro-F1 then minimizes any conflict.",
    )
    parser.add_argument(
        "--macro-retention",
        type=float,
        default=0.95,
        help="For conflict_priority, retain candidates with validation macro-F1 >= best_macro * this value.",
    )
    parser.add_argument("--output-prefix", default="rf_temporal_decoding_locked")
    args = parser.parse_args()

    cfg = load_config(args.config)
    root = DEFAULT_PROJECT_ROOT
    official_code = Path(cfg.get("capture24_official_code", ""))
    if not official_code.exists():
        official_code = root / "capture24_project" / "official_code" / "capture24"
    prepared_data = Path(cfg["prepared_data"])
    if not prepared_data.exists():
        prepared_data = Path("D:/claude项目/capture24_work2/capture24/prepared_data")
    optimisedir = root / "results" / "baselines" / "optimised_params_rf_xgb_compat_full_merged"
    baseline_dir = root / "results" / "baselines"
    bcm_dir = root / "results" / "bcm"
    tables_dir = root / "paper_artifacts" / "tables"
    bcm_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(official_code))
    from classifier import Classifier

    X = pd.read_pickle(prepared_data / "X_feats.pkl").values
    P = np.load(prepared_data / "P.npy")
    y = np.load(prepared_data / "Y_WillettsSpecific2018.npy")
    y_coarse = np.load(prepared_data / "Y_Walmsley2020.npy")

    internal_train_ids = participant_range(1, 80)
    internal_val_ids = participant_range(81, 100)
    train_mask = np.isin(P, internal_train_ids)
    val_mask = np.isin(P, internal_val_ids)

    kwargs = {"n_jobs": args.model_n_jobs}
    if args.n_estimators_override is not None:
        kwargs["n_estimators"] = args.n_estimators_override
    fine_clf = Classifier(
        "rf",
        seed=42,
        optimisedir=str(optimisedir / "rf_WillettsSpecific2018.pkl"),
        **kwargs,
    )
    fine_clf.fit(X[train_mask], y[train_mask], P[train_mask])
    val_proba = fine_clf.predict_proba(X[val_mask])
    val_classes = np.asarray(fine_clf.window_classifier.model.classes_)
    val_y = y[val_mask]
    val_p = P[val_mask]

    coarse_clf = Classifier(
        "rf",
        seed=42,
        optimisedir=str(optimisedir / "rf_Walmsley2020.pkl"),
        **kwargs,
    )
    coarse_clf.fit(X[train_mask], y_coarse[train_mask], P[train_mask])
    val_coarse_proba = coarse_clf.predict_proba(X[val_mask])
    val_coarse_classes = np.asarray(coarse_clf.window_classifier.model.classes_)

    # Reuse existing internal validation intensity head from the BCM tuning run.
    bcm_dir = root / "results" / "bcm"
    val_aux_npz = latest(bcm_dir, "posthoc_calibration_v2_locked_base_*_internal_validation_predictions.npz")
    val_aux = np.load(val_aux_npz, allow_pickle=True)
    if not np.array_equal(val_aux["participant"], val_p) or not np.array_equal(val_aux["y_true"], val_y):
        raise ValueError("Internal validation intensity artifact is not aligned with RF validation split.")
    val_intensity_proba = val_aux["proba_intensity"]
    val_intensity_classes = val_aux["intensity_classes"]

    def map_matrix(source: np.ndarray, target: np.ndarray, source_classes: np.ndarray, target_classes: np.ndarray, mask: np.ndarray) -> np.ndarray:
        mat = np.full((len(source_classes), len(target_classes)), args.smoothing, dtype="float64")
        src_index = {c: i for i, c in enumerate(source_classes)}
        tgt_index = {c: i for i, c in enumerate(target_classes)}
        for s, t in zip(source[mask], target[mask]):
            if s in src_index and t in tgt_index:
                mat[src_index[s], tgt_index[t]] += 1.0
        return row_normalize(mat)

    label_fields = np.load(latest(bcm_dir, "capture24_derived_label_fields_*.npz"), allow_pickle=True)
    train80_valid_mask = train_mask & label_fields["valid_annotation_mask"]
    train100_valid_mask = np.isin(P, participant_range(1, 100)) & label_fields["valid_annotation_mask"]
    val_coarse_support = val_coarse_proba @ map_matrix(y, y_coarse, val_classes, val_coarse_classes, train80_valid_mask).T
    val_intensity_support = val_intensity_proba @ map_matrix(y, label_fields["y_met_intensity4"], val_classes, val_intensity_classes, train80_valid_mask).T
    val_coarse_hard = val_coarse_classes[np.argmax(val_coarse_proba, axis=1)]
    val_intensity_hard = val_intensity_classes[np.argmax(val_intensity_proba, axis=1)]
    val_fine_to_coarse = val_coarse_classes[
        np.argmax(map_matrix(y, y_coarse, val_classes, val_coarse_classes, train80_valid_mask), axis=1)
    ]
    val_fine_to_intensity = val_intensity_classes[
        np.argmax(map_matrix(y, label_fields["y_met_intensity4"], val_classes, val_intensity_classes, train80_valid_mask), axis=1)
    ]

    trans_val = transition_matrix(y, P, val_classes, internal_train_ids, args.smoothing)
    rows = []
    for alpha in parse_grid(args.alphas):
        for beta in parse_grid(args.betas):
            val_emission = row_normalize(
                val_proba
                * np.power(np.maximum(val_coarse_support, 1e-12), alpha)
                * np.power(np.maximum(val_intensity_support, 1e-12), beta)
            )
            for gamma in parse_grid(args.gammas):
                pred = viterbi_decode_proba(val_emission, val_p, val_classes, trans_val, gamma, 1e-12)
                row = {"alpha": alpha, "beta": beta, "gamma": gamma}
                row.update(metric_row(val_y, pred))
                row.update(
                    conflict_row(
                        pred,
                        val_classes,
                        val_coarse_hard,
                        val_intensity_hard,
                        val_fine_to_coarse,
                        val_fine_to_intensity,
                    )
                )
                rows.append(row)
    val_df = pd.DataFrame(rows).sort_values(["macro_f1", "balanced_accuracy"], ascending=False)
    if args.selection_mode == "performance":
        selected = val_df.iloc[0]
    else:
        best_macro = float(val_df["macro_f1"].max())
        eligible = val_df[val_df["macro_f1"] >= best_macro * args.macro_retention].copy()
        selected = eligible.sort_values(
            ["any_conflict_rate", "intensity_conflict_rate", "macro_f1", "balanced_accuracy"],
            ascending=[True, True, False, False],
        ).iloc[0]

    test_npz = latest(baseline_dir, "official_baselines_*rfcompat_xgbopt_full_WillettsSpecific2018_predictions.npz")
    test = np.load(test_npz, allow_pickle=True)
    test_classes = test["classes_rf_rfcompat_xgbopt_full"]
    if not np.array_equal(test_classes, val_classes):
        raise ValueError("Internal RF and locked RF class order mismatch.")
    trans_test = transition_matrix(y, P, test_classes, participant_range(1, 100), args.smoothing)
    coarse_test = np.load(latest(baseline_dir, "official_baselines_*rfcompat_xgbopt_full_Walmsley2020_predictions.npz"), allow_pickle=True)
    intensity_test = np.load(latest(bcm_dir, "auxiliary_xgb_*_y_met_intensity4_predictions.npz"), allow_pickle=True)
    test_coarse_support = coarse_test["proba_rf_rfcompat_xgbopt_full"] @ map_matrix(
        y,
        y_coarse,
        test_classes,
        coarse_test["classes_rf_rfcompat_xgbopt_full"],
        train100_valid_mask,
    ).T
    test_intensity_support = intensity_test["proba"] @ map_matrix(
        y,
        label_fields["y_met_intensity4"],
        test_classes,
        intensity_test["classes"],
        train100_valid_mask,
    ).T
    test_coarse_hard = coarse_test["classes_rf_rfcompat_xgbopt_full"][np.argmax(coarse_test["proba_rf_rfcompat_xgbopt_full"], axis=1)]
    test_intensity_hard = intensity_test["classes"][np.argmax(intensity_test["proba"], axis=1)]
    test_fine_to_coarse = coarse_test["classes_rf_rfcompat_xgbopt_full"][
        np.argmax(
            map_matrix(
                y,
                y_coarse,
                test_classes,
                coarse_test["classes_rf_rfcompat_xgbopt_full"],
                train100_valid_mask,
            ),
            axis=1,
        )
    ]
    test_fine_to_intensity = intensity_test["classes"][
        np.argmax(
            map_matrix(
                y,
                label_fields["y_met_intensity4"],
                test_classes,
                intensity_test["classes"],
                train100_valid_mask,
            ),
            axis=1,
        )
    ]
    test_emission = row_normalize(
        test["proba_rf_rfcompat_xgbopt_full"]
        * np.power(np.maximum(test_coarse_support, 1e-12), float(selected["alpha"]))
        * np.power(np.maximum(test_intensity_support, 1e-12), float(selected["beta"]))
    )
    test_temporal = viterbi_decode_proba(
        test_emission,
        test["participant"],
        test_classes,
        trans_test,
        float(selected["gamma"]),
        1e-12,
    )
    y_test = test["y_true"]
    locked_rows = []
    for method, pred in [
        ("rf_baseline", test["pred_rf_rfcompat_xgbopt_full"]),
        ("rf_official_hmm", test["pred_rf_hmm_rfcompat_xgbopt_full"]),
        ("rf_probability_temporal", test_temporal),
    ]:
        row = {
            "method": method,
            "alpha": float(selected["alpha"]) if method == "rf_probability_temporal" else np.nan,
            "beta": float(selected["beta"]) if method == "rf_probability_temporal" else np.nan,
            "gamma": float(selected["gamma"]) if method == "rf_probability_temporal" else np.nan,
        }
        row.update(metric_row(y_test, pred))
        row.update(
            conflict_row(
                pred,
                test_classes,
                test_coarse_hard,
                test_intensity_hard,
                test_fine_to_coarse,
                test_fine_to_intensity,
            )
        )
        row["changed_vs_rf_baseline"] = int(np.sum(pred != test["pred_rf_rfcompat_xgbopt_full"]))
        row["corrected_vs_rf_baseline"] = int(((test["pred_rf_rfcompat_xgbopt_full"] != y_test) & (pred == y_test)).sum())
        row["harmed_vs_rf_baseline"] = int(((test["pred_rf_rfcompat_xgbopt_full"] == y_test) & (pred != y_test)).sum())
        locked_rows.append(row)
    locked_df = pd.DataFrame(locked_rows)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = f"{args.output_prefix}_{timestamp}"
    val_path = bcm_dir / f"{prefix}_validation_gamma_candidates.csv"
    metrics_path = bcm_dir / f"{prefix}_locked_test_metrics.csv"
    pred_path = bcm_dir / f"{prefix}_locked_test_predictions.npz"
    report_path = bcm_dir / f"{args.output_prefix}_report.md"
    table_path = tables_dir / f"table_{args.output_prefix}_locked_test.csv"
    val_df.to_csv(val_path, index=False)
    locked_df.to_csv(metrics_path, index=False)
    locked_df.to_csv(table_path, index=False)
    np.savez_compressed(
        pred_path,
        participant=test["participant"],
        y_true=y_test,
        classes=test_classes,
        pred_rf_baseline=test["pred_rf_rfcompat_xgbopt_full"],
        pred_rf_official_hmm=test["pred_rf_hmm_rfcompat_xgbopt_full"],
        pred_rf_probability_temporal=test_temporal,
        proba_rf_aux_temporal_emission=test_emission.astype("float32"),
        selected_gamma=np.asarray(float(selected["gamma"])),
        selected_alpha=np.asarray(float(selected["alpha"])),
        selected_beta=np.asarray(float(selected["beta"])),
    )
    with report_path.open("w", encoding="utf-8") as f:
        f.write("# RF Probability Temporal Decoding Locked Test\n\n")
        f.write("Alpha, beta, and gamma are selected on P081-P100 after training RF on P001-P080, then applied once to saved full-derivation RF probabilities on P101-P151.\n\n")
        f.write(f"- validation candidates: `{val_path}`\n")
        f.write(f"- locked metrics: `{metrics_path}`\n")
        f.write(f"- predictions: `{pred_path}`\n\n")
        f.write(f"- selection mode: `{args.selection_mode}`\n")
        f.write(f"- macro retention: `{args.macro_retention}`\n\n")
        f.write("## Selected Gamma\n\n")
        f.write(selected.to_frame().T.to_markdown(index=False))
        f.write("\n\n## Locked Test Metrics\n\n")
        f.write(locked_df.to_markdown(index=False))
        f.write("\n")
    print(report_path)
    print("SELECTED")
    print(selected.to_string())
    print("LOCKED TEST")
    print(locked_df.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
