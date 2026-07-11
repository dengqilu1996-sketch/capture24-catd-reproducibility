"""Audit experiment: gamma-tuned official RF+HMM baseline.

The official Capture-24 RF+HMM baseline uses hard RF predictions as HMM
observations and a calibrated HMM score with an implicit transition weight of
1. This script keeps that mechanism fixed and tunes only a multiplicative
gamma on the transition log-probability, using P001-P080/P081-P100 validation
and one locked P001-P100/P101-P151 evaluation.
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


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OFFICIAL_CODE = PROJECT_ROOT / "capture24_project" / "official_code" / "capture24"
PREPARED_DATA = PROJECT_ROOT / "data" / "prepared_data_official_repro"
OPTIMISEDIR = PROJECT_ROOT / "results" / "baselines" / "optimised_params_rf_xgb_compat_full_merged"
BASELINE_DIR = PROJECT_ROOT / "results" / "baselines"
OUT_DIR = PROJECT_ROOT / "results" / "bcm"
AUDIT_DIR = PROJECT_ROOT / "paper_artifacts" / "audits"


def participant_range(start: int, end: int) -> list[str]:
    return [f"P{i:03d}" for i in range(start, end + 1)]


def ordered_unique(values: np.ndarray) -> np.ndarray:
    seen = set()
    out = []
    for value in values:
        if value not in seen:
            seen.add(value)
            out.append(value)
    return np.asarray(out)


def parse_grid(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def latest(path: Path, pattern: str) -> Path | None:
    files = sorted(path.glob(pattern))
    return files[-1] if files else None


def metric_row(y_true: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    return {
        "balanced_accuracy": float(metrics.balanced_accuracy_score(y_true, pred)),
        "macro_f1": float(metrics.f1_score(y_true, pred, average="macro", zero_division=0)),
        "weighted_f1": float(metrics.f1_score(y_true, pred, average="weighted", zero_division=0)),
        "mcc": float(metrics.matthews_corrcoef(y_true, pred)),
        "kappa": float(metrics.cohen_kappa_score(y_true, pred)),
    }


def gamma_viterbi(
    observations: np.ndarray,
    groups: np.ndarray,
    prior: np.ndarray,
    emission: np.ndarray,
    transition: np.ndarray,
    labels: np.ndarray,
    gamma: float,
    eps: float = 1e-16,
) -> np.ndarray:
    """Official hard-observation HMM Viterbi with gamma * log(transition)."""
    label_to_idx = {label: i for i, label in enumerate(labels)}
    pred = np.empty_like(observations)
    log_prior = np.log(prior + eps)
    log_emission = np.log(emission + eps)
    log_transition = gamma * np.log(transition + eps)

    for group in ordered_unique(groups):
        idx = np.where(groups == group)[0]
        obs_idx = np.asarray([label_to_idx[label] for label in observations[idx]], dtype=np.int32)
        n_obs = len(obs_idx)
        n_labels = len(labels)
        scores = np.zeros((n_obs, n_labels), dtype="float64")
        back = np.zeros((n_obs, n_labels), dtype=np.int32)

        scores[0] = log_prior + log_emission[:, obs_idx[0]]
        for t in range(1, n_obs):
            step = scores[t - 1][:, None] + log_transition
            back[t] = np.argmax(step, axis=0)
            scores[t] = log_emission[:, obs_idx[t]] + step[back[t], np.arange(n_labels)]

        path = np.zeros(n_obs, dtype=np.int32)
        path[-1] = int(np.argmax(scores[-1]))
        for t in range(n_obs - 2, -1, -1):
            path[t] = back[t + 1, path[t + 1]]
        pred[idx] = labels[path]

    return pred


def official_hmm_params(classifier) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    smoother = classifier.smoother
    return (
        np.asarray(smoother.startprob),
        np.asarray(smoother.emissionprob),
        np.asarray(smoother.transmat),
        np.asarray(smoother.labels),
    )


def rows_for_gammas(
    classifier,
    X_eval: np.ndarray,
    y_eval: np.ndarray,
    p_eval: np.ndarray,
    gammas: list[float],
) -> tuple[pd.DataFrame, dict[float, np.ndarray], np.ndarray]:
    rf_pred = classifier.window_classifier.predict(X_eval)
    prior, emission, transition, labels = official_hmm_params(classifier)
    pred_by_gamma = {}
    rows = []
    for gamma in gammas:
        pred = gamma_viterbi(rf_pred, p_eval, prior, emission, transition, labels, gamma)
        pred_by_gamma[gamma] = pred
        row = {"gamma": gamma}
        row.update(metric_row(y_eval, pred))
        row["changed_vs_rf_window"] = int(np.sum(pred != rf_pred))
        rows.append(row)

    official_pred = classifier.smoother.predict(rf_pred, p_eval)
    if 1.0 in pred_by_gamma:
        gamma1_diff = int(np.sum(pred_by_gamma[1.0] != official_pred))
        for row in rows:
            if row["gamma"] == 1.0:
                row["diff_vs_official_hmm_predict"] = gamma1_diff
            else:
                row["diff_vs_official_hmm_predict"] = np.nan
    return pd.DataFrame(rows), pred_by_gamma, rf_pred


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared-data", type=Path, default=PREPARED_DATA)
    parser.add_argument("--optimisedir", type=Path, default=OPTIMISEDIR)
    parser.add_argument("--gammas", default="0,0.25,0.5,0.75,1,1.5,2,2.5,3,4,5,6,8,10")
    parser.add_argument("--model-n-jobs", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-estimators-override", type=int, default=None, help="Debug only; leave unset for formal RF.")
    args = parser.parse_args()

    sys.path.insert(0, str(OFFICIAL_CODE))
    from classifier import Classifier

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)

    X = pd.read_pickle(args.prepared_data / "X_feats.pkl").values
    P = np.load(args.prepared_data / "P.npy")
    y = np.load(args.prepared_data / "Y_WillettsSpecific2018.npy")
    gammas = parse_grid(args.gammas)
    if 1.0 not in gammas:
        gammas.append(1.0)
        gammas = sorted(gammas)

    kwargs = {"n_jobs": args.model_n_jobs}
    if args.n_estimators_override is not None:
        kwargs["n_estimators"] = args.n_estimators_override
    optimised_path = args.optimisedir / "rf_hmm_WillettsSpecific2018.pkl"

    train80 = np.isin(P, participant_range(1, 80))
    val = np.isin(P, participant_range(81, 100))
    train100 = np.isin(P, participant_range(1, 100))
    test = np.isin(P, participant_range(101, 151))

    val_clf = Classifier("rf_hmm", args.seed, optimisedir=str(optimised_path), **kwargs)
    val_clf.fit(X[train80], y[train80], P[train80])
    val_df, val_preds, val_rf_pred = rows_for_gammas(val_clf, X[val], y[val], P[val], gammas)
    val_df = val_df.sort_values(["macro_f1", "balanced_accuracy"], ascending=False)
    selected = val_df.iloc[0]
    selected_gamma = float(selected["gamma"])

    test_clf = Classifier("rf_hmm", args.seed, optimisedir=str(optimised_path), **kwargs)
    test_clf.fit(X[train100], y[train100], P[train100])
    test_df, test_preds, test_rf_pred = rows_for_gammas(test_clf, X[test], y[test], P[test], gammas)
    test_df.insert(0, "selected_on_validation", test_df["gamma"].eq(selected_gamma))
    test_df = test_df.sort_values(["selected_on_validation", "gamma"], ascending=[False, True])

    selected_pred = test_preds[selected_gamma]
    saved_locked = latest(BASELINE_DIR, "official_baselines_*rfcompat_xgbopt_full_WillettsSpecific2018_predictions.npz")
    saved_diff = None
    saved_rf_diff = None
    if saved_locked is not None:
        saved = np.load(saved_locked, allow_pickle=True)
        saved_diff = int(np.sum(test_preds[1.0] != saved["pred_rf_hmm_rfcompat_xgbopt_full"]))
        saved_rf_diff = int(np.sum(test_rf_pred != saved["pred_rf_rfcompat_xgbopt_full"]))

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = f"official_rf_hmm_gamma_tuned_{timestamp}"
    val_path = OUT_DIR / f"{prefix}_validation_gamma_candidates.csv"
    test_path = OUT_DIR / f"{prefix}_locked_test_metrics.csv"
    pred_path = OUT_DIR / f"{prefix}_locked_test_predictions.npz"
    report_path = OUT_DIR / "official_rf_hmm_gamma_tuned_report.md"
    audit_path = AUDIT_DIR / "rf_hmm_gamma_tuning_audit_20260701.md"

    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)
    np.savez_compressed(
        pred_path,
        participant=P[test],
        y_true=y[test],
        pred_rf_window=test_rf_pred,
        pred_official_hmm_gamma1=test_preds[1.0],
        pred_official_hmm_gamma_tuned=selected_pred,
        selected_gamma=np.asarray(selected_gamma),
        gammas=np.asarray(gammas),
    )

    selected_test = test_df[test_df["gamma"].eq(selected_gamma)].iloc[0]
    gamma1_test = test_df[test_df["gamma"].eq(1.0)].iloc[0]
    with report_path.open("w", encoding="utf-8") as f:
        f.write("# Official RF+HMM Gamma-Tuning Audit\n\n")
        f.write("This audit keeps the official hard-observation RF+HMM mechanism fixed and tunes only `gamma * log(transition)`.\n\n")
        f.write(f"- validation candidates: `{val_path}`\n")
        f.write(f"- locked-test metrics: `{test_path}`\n")
        f.write(f"- locked-test predictions: `{pred_path}`\n")
        f.write(f"- selected gamma: `{selected_gamma}`\n")
        if saved_diff is not None:
            f.write(f"- gamma=1 diff vs saved official RF+HMM locked predictions: `{saved_diff}` windows\n")
            f.write(f"- RF window diff vs saved RF locked predictions: `{saved_rf_diff}` windows\n")
        f.write("\n## Selected Validation Row\n\n")
        f.write(selected.to_frame().T.to_markdown(index=False))
        f.write("\n\n## Locked-Test Rows\n\n")
        f.write(test_df.to_markdown(index=False))
        f.write("\n")

    with audit_path.open("w", encoding="utf-8") as f:
        f.write("# RF+HMM Transition-Weight Fairness Audit (2026-07-01)\n\n")
        f.write("## Question\n\n")
        f.write("Does the official Capture-24 RF+HMM baseline tune the HMM transition weight, and how much does a fair gamma-tuned official RF+HMM baseline explain the reported temporal-decoding gain?\n\n")
        f.write("## Code-Level Finding\n\n")
        f.write("The clean official code has no transition-weight hyperparameter. `hmm.py` adds `log(emission)` and `log(transition)` directly in Viterbi decoding, so the implicit transition weight is gamma=1. `classifier.py` constructs `HMM()` without a gamma argument.\n\n")
        f.write("## Audit Experiment\n\n")
        f.write("I trained official `Classifier('rf_hmm')` on P001-P080, selected gamma on P081-P100, then retrained on P001-P100 and evaluated once on locked P101-P151. The RF model, OOB emission estimation, hard RF observations, and transition matrix are the official mechanism; only gamma is tuned.\n\n")
        f.write(f"- validation candidates: `{val_path}`\n")
        f.write(f"- locked-test metrics: `{test_path}`\n")
        f.write(f"- selected gamma: `{selected_gamma}`\n\n")
        f.write("## Key Numbers\n\n")
        f.write(f"- validation selected macro-F1: {float(selected['macro_f1']):.6f}\n")
        f.write(f"- locked gamma=1 official-style macro-F1: {float(gamma1_test['macro_f1']):.6f}\n")
        f.write(f"- locked gamma-tuned official-style macro-F1: {float(selected_test['macro_f1']):.6f}\n")
        if saved_diff is not None:
            f.write(f"- gamma=1 diff vs saved official RF+HMM prediction artifact: {saved_diff} windows\n")
            f.write(f"- RF window diff vs saved RF prediction artifact: {saved_rf_diff} windows\n")
        f.write("\n## Interpretation Placeholder\n\n")
        f.write("Use this audit to decide whether the manuscript needs to replace the current official RF+HMM comparator with a gamma-tuned RF+HMM comparator or reframe the method as a consistency/conflit diagnostic built on top of tuned temporal smoothing.\n")

    print(report_path)
    print("SELECTED VALIDATION")
    print(selected.to_string())
    print("LOCKED GAMMA=1")
    print(gamma1_test.to_string())
    print("LOCKED SELECTED")
    print(selected_test.to_string())
    if saved_diff is not None:
        print(f"diff_vs_saved_official_hmm_gamma1={saved_diff}")
        print(f"diff_vs_saved_rf_window={saved_rf_diff}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
