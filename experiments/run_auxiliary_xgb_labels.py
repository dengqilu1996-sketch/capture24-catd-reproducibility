"""Train GPU XGBoost auxiliary label heads for BCM/post-hoc calibration.

This uses the clean official Capture-24 Classifier wrapper, but labels are
derived project artifacts such as MET-derived intensity. The official code is
not modified.
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


def latest_label_fields(results_bcm: Path) -> Path:
    files = sorted(results_bcm.glob("capture24_derived_label_fields_*.npz"))
    if not files:
        raise FileNotFoundError(f"No derived label field NPZ found in {results_bcm}")
    return files[-1]


def train_test_split(P: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    test_ids = [f"P{i}" for i in range(101, 152)]
    mask_test = np.isin(P, test_ids)
    return ~mask_test, mask_test


def metric_rows(metric_dict: dict[str, str], run_id: str, label_field: str, model: str, elapsed: float) -> list[dict[str, object]]:
    rows = []
    for metric_name, metric_value in metric_dict.items():
        point, interval = metric_value.split(" ", 1)
        low, high = interval.strip("()").split(",")
        rows.append(
            {
                "run_id": run_id,
                "label_field": label_field,
                "model": model,
                "metric": metric_name,
                "value": float(point),
                "ci_low": float(low),
                "ci_high": float(high),
                "elapsed_seconds": elapsed,
            }
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--label-fields", default="y_met_intensity4")
    parser.add_argument("--label-fields-npz", type=Path, default=None)
    parser.add_argument("--model", default="xgb")
    parser.add_argument("--xgb-device", default="cuda")
    parser.add_argument("--xgb-tree-method", default="hist")
    parser.add_argument("--model-n-jobs", type=int, default=4)
    parser.add_argument("--nboots", type=int, default=100)
    parser.add_argument("--n-jobs", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = load_config(args.config)
    project_root = Path(cfg["project_root"])
    prepared_data = Path(cfg["prepared_data"])
    official_code = Path(cfg["capture24_official_code"])
    results_bcm = Path(cfg["bcm_results"])
    logs_dir = Path(cfg["logs"])
    results_bcm.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(official_code))
    from classifier import Classifier
    from eval import metrics_report

    label_npz = args.label_fields_npz or latest_label_fields(results_bcm)
    label_fields = [x.strip() for x in args.label_fields.split(",") if x.strip()]

    X = pd.read_pickle(prepared_data / "X_feats.pkl").values
    P = np.load(prepared_data / "P.npy")
    labels = np.load(label_npz, allow_pickle=True)
    train_mask, test_mask = train_test_split(P)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"auxiliary_xgb_{timestamp}"
    all_rows = []
    outputs = {}

    for field in label_fields:
        y = labels[field]
        valid_mask = y != "unknown"
        train_idx = train_mask & valid_mask
        test_idx_eval = test_mask & valid_mask
        test_idx_all = test_mask

        model_kwargs = {
            "n_jobs": args.model_n_jobs,
            "device": args.xgb_device,
            "tree_method": args.xgb_tree_method,
        }
        classifier = Classifier(args.model, args.seed, optimisedir=str(prepared_data / "__no_optimised_params__"), **model_kwargs)

        start = time.time()
        classifier.fit(X[train_idx], y[train_idx], P[train_idx])
        pred_all = classifier.predict(X[test_idx_all], P[test_idx_all])
        proba_all = classifier.predict_proba(X[test_idx_all], P[test_idx_all])
        elapsed = time.time() - start

        pred_eval = pred_all[valid_mask[test_idx_all]]
        y_eval = y[test_idx_eval]
        p_eval = P[test_idx_eval]
        metrics = metrics_report(
            y_eval,
            pred_eval,
            p_eval,
            tag=f"{field}/{args.model}_gpu",
            nboots=args.nboots,
            n_jobs=args.n_jobs,
            verbose=True,
        )
        all_rows.extend(metric_rows(metrics, run_id, field, f"{args.model}_gpu", elapsed))

        classes = getattr(classifier.window_classifier.le, "classes_", None)
        pred_path = results_bcm / f"{run_id}_{field}_predictions.npz"
        np.savez_compressed(
            pred_path,
            participant=P[test_idx_all],
            y_true=y[test_idx_all],
            valid_eval_mask=valid_mask[test_idx_all],
            pred=pred_all,
            proba=proba_all,
            classes=np.asarray(classes),
        )
        outputs[field] = str(pred_path)

    metrics_path = results_bcm / f"{run_id}_metrics.csv"
    pd.DataFrame(all_rows).to_csv(metrics_path, index=False)
    metadata = {
        "run_id": run_id,
        "created_at": timestamp,
        "project_root": str(project_root),
        "official_code": str(official_code),
        "official_code_commit": git_short_sha(official_code),
        "prepared_data": str(prepared_data),
        "label_fields_npz": str(label_npz),
        "label_fields": label_fields,
        "model": args.model,
        "xgb_device": args.xgb_device,
        "xgb_tree_method": args.xgb_tree_method,
        "python": sys.executable,
        "python_version": platform.python_version(),
        "outputs": {**outputs, "metrics": str(metrics_path)},
    }
    metadata_path = logs_dir / f"{run_id}.json"
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    report_path = results_bcm / "auxiliary_xgb_label_heads_report.md"
    with report_path.open("a", encoding="utf-8") as f:
        f.write(f"\n\n## {run_id}\n\n")
        f.write(f"- Metrics CSV: `{metrics_path}`\n")
        f.write(f"- Metadata JSON: `{metadata_path}`\n")
        for field, path in outputs.items():
            f.write(f"- `{field}` predictions: `{path}`\n")
        f.write("\n")
        f.write(pd.DataFrame(all_rows).pivot_table(index=["label_field", "model"], columns="metric", values="value", aggfunc="first").reset_index().to_markdown(index=False))
        f.write("\n")

    print(metrics_path)
    print(metadata_path)
    for path in outputs.values():
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
