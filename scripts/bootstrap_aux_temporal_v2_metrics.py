from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn import metrics


ROOT = Path(__file__).resolve().parents[1]
PRED = ROOT / "results" / "bcm" / "rf_aux_temporal_conflict_priority_locked_20260630_164943_locked_test_predictions.npz"
TEMPORAL_ONLY_PRED = ROOT / "results" / "bcm" / "rf_temporal_decoding_locked_20260630_160032_locked_test_predictions.npz"
OUT_DIR = ROOT / "paper_artifacts" / "tables"
N_BOOT = 5000
SEED = 20260630


METHODS = {
    "RF baseline": "pred_rf_baseline",
    "RF+HMM": "pred_rf_official_hmm",
    "CATD (conflict-priority)": "pred_rf_probability_temporal",
}


def score(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "macro_f1": float(metrics.f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "balanced_accuracy": float(metrics.balanced_accuracy_score(y_true, y_pred)),
    }


def scores_from_confusion(cm: np.ndarray) -> dict[str, float]:
    tp = np.diag(cm).astype("float64")
    support = cm.sum(axis=1).astype("float64")
    pred_support = cm.sum(axis=0).astype("float64")
    recall = np.divide(tp, support, out=np.zeros_like(tp), where=support > 0)
    precision = np.divide(tp, pred_support, out=np.zeros_like(tp), where=pred_support > 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros_like(tp), where=(precision + recall) > 0)
    valid = support > 0
    return {
        "macro_f1": float(f1[valid].mean()),
        "balanced_accuracy": float(recall[valid].mean()),
    }


def ci(values: np.ndarray) -> tuple[float, float]:
    return tuple(np.percentile(values, [2.5, 97.5]).astype(float))


def markdown_table(frame: pd.DataFrame) -> str:
    """Render a small markdown table without requiring the optional tabulate package."""
    headers = list(frame.columns)
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    for row in frame.itertuples(index=False, name=None):
        cells = []
        for value in row:
            cells.append(f"{value:.4f}" if isinstance(value, (float, np.floating)) else str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main() -> None:
    z = np.load(PRED, allow_pickle=True)
    temporal_only_z = np.load(TEMPORAL_ONLY_PRED, allow_pickle=True)
    participant = z["participant"]
    y_true = z["y_true"]
    if not (np.array_equal(participant, temporal_only_z["participant"]) and np.array_equal(y_true, temporal_only_z["y_true"])):
        raise ValueError("CATD and temporal-only prediction files do not share the same locked-test rows")
    participants = np.unique(participant)
    rng = np.random.default_rng(SEED)

    base_rows = []
    point_scores = {name: score(y_true, z[key]) for name, key in METHODS.items()}
    point_scores["temporal-only decoding"] = score(y_true, temporal_only_z["pred_rf_probability_temporal"])
    for name, vals in point_scores.items():
        row = {"method": name}
        row.update(vals)
        base_rows.append(row)

    method_preds = {name: z[key] for name, key in METHODS.items()}
    method_preds["temporal-only decoding"] = temporal_only_z["pred_rf_probability_temporal"]
    boot = {name: {"macro_f1": [], "balanced_accuracy": []} for name in method_preds}
    deltas_vs_hmm = {"macro_f1": [], "balanced_accuracy": []}
    deltas_vs_rf = {"macro_f1": [], "balanced_accuracy": []}
    deltas_vs_temporal_only = {"macro_f1": [], "balanced_accuracy": []}

    classes = z["classes"]
    participant_confusions = {name: [] for name in method_preds}
    for p in participants:
        idx = participant == p
        for name, pred in method_preds.items():
            participant_confusions[name].append(metrics.confusion_matrix(y_true[idx], pred[idx], labels=classes))
    participant_confusions = {
        name: np.stack(mats, axis=0).astype("int64") for name, mats in participant_confusions.items()
    }

    for _ in range(N_BOOT):
        sampled_idx = rng.integers(0, len(participants), size=len(participants))
        scores = {}
        for name in method_preds:
            cm = participant_confusions[name][sampled_idx].sum(axis=0)
            scores[name] = scores_from_confusion(cm)
            for metric_name, value in scores[name].items():
                boot[name][metric_name].append(value)
        for metric_name in ("macro_f1", "balanced_accuracy"):
            deltas_vs_hmm[metric_name].append(
                scores["CATD (conflict-priority)"][metric_name] - scores["RF+HMM"][metric_name]
            )
            deltas_vs_rf[metric_name].append(
                scores["CATD (conflict-priority)"][metric_name] - scores["RF baseline"][metric_name]
            )
            deltas_vs_temporal_only[metric_name].append(
                scores["CATD (conflict-priority)"][metric_name] - scores["temporal-only decoding"][metric_name]
            )

    metric_rows = []
    for row in base_rows:
        name = row["method"]
        out = dict(row)
        for metric_name in ("macro_f1", "balanced_accuracy"):
            low, high = ci(np.asarray(boot[name][metric_name]))
            out[f"{metric_name}_ci_low"] = low
            out[f"{metric_name}_ci_high"] = high
        metric_rows.append(out)

    delta_rows = []
    point = point_scores
    for label, store, ref in [
        ("CATD vs RF+HMM", deltas_vs_hmm, "RF+HMM"),
        ("CATD vs RF baseline", deltas_vs_rf, "RF baseline"),
        ("CATD vs temporal-only decoding", deltas_vs_temporal_only, "temporal-only decoding"),
    ]:
        row = {"comparison": label}
        for metric_name in ("macro_f1", "balanced_accuracy"):
            point_delta = point["CATD (conflict-priority)"][metric_name] - point[ref][metric_name]
            low, high = ci(np.asarray(store[metric_name]))
            row[f"delta_{metric_name}"] = point_delta
            row[f"delta_{metric_name}_ci_low"] = low
            row[f"delta_{metric_name}_ci_high"] = high
        delta_rows.append(row)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    metrics_df = pd.DataFrame(metric_rows)
    deltas_df = pd.DataFrame(delta_rows)
    metrics_csv = OUT_DIR / "table_aux_temporal_bootstrap_metrics_v2.csv"
    deltas_csv = OUT_DIR / "table_aux_temporal_bootstrap_deltas_v2.csv"
    metrics_df.to_csv(metrics_csv, index=False)
    deltas_df.to_csv(deltas_csv, index=False)

    md = OUT_DIR / "table_aux_temporal_bootstrap_ci_v2.md"
    with md.open("w", encoding="utf-8") as f:
        f.write("# Participant-Level Bootstrap CI: Auxiliary Temporal v2\n\n")
        f.write(f"Bootstrap unit: participant. Iterations: {N_BOOT}. Seed: {SEED}.\n\n")
        f.write("## Method Metrics\n\n")
        f.write(markdown_table(metrics_df))
        f.write("\n\n## Deltas\n\n")
        f.write(markdown_table(deltas_df))
        f.write("\n")

    print(f"Wrote {metrics_csv}")
    print(f"Wrote {deltas_csv}")
    print(f"Wrote {md}")


if __name__ == "__main__":
    main()
