"""Run Capture-24 official RF/XGBoost/HMM baselines and save artifacts.

This wrapper intentionally imports the clean official Capture-24 code from
capture24_project/official_code/capture24. It does not import any Python code
from the previous project; only prepared_data arrays are used as input.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

# Keep numerical libraries from spawning extra hidden thread pools. Model-level
# parallelism is controlled explicitly with --model-n-jobs.
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


def parse_csv_arg(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def git_short_sha(path: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "--short", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def resolve_existing_path(path: Path, fallback: Path) -> Path:
    return path if path.exists() else fallback


def package_versions() -> dict[str, str]:
    import importlib.metadata as md

    packages = [
        "numpy",
        "pandas",
        "scikit-learn",
        "imbalanced-learn",
        "xgboost",
        "hyperopt",
        "joblib",
        "PyYAML",
    ]
    versions = {}
    for pkg in packages:
        try:
            versions[pkg] = md.version(pkg)
        except md.PackageNotFoundError:
            versions[pkg] = "not-installed"
    return versions


def split_metric_interval(value: str) -> tuple[float, float, float]:
    # Official eval returns strings like "0.543 (0.501, 0.581)".
    point, interval = value.split(" ", 1)
    low, high = interval.strip("()").split(",")
    return float(point), float(low), float(high)


def write_metrics_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--prepared-data", type=Path, default=None)
    parser.add_argument("--optimisedir", type=Path, default=None)
    parser.add_argument("--labels", default="Walmsley2020,WillettsSpecific2018")
    parser.add_argument("--models", default="rf,rf_hmm,xgb,xgb_hmm")
    parser.add_argument("--nboots", type=int, default=100)
    parser.add_argument("--n-jobs", type=int, default=12)
    parser.add_argument("--model-n-jobs", type=int, default=4)
    parser.add_argument("--xgb-device", default=None, help="Optional XGBoost device, e.g. cuda or cpu.")
    parser.add_argument("--xgb-tree-method", default=None, help="Optional XGBoost tree_method, e.g. hist.")
    parser.add_argument("--run-tag", default="", help="Optional tag appended to run id and model names.")
    parser.add_argument("--save-proba", action="store_true", help="Save predict_proba arrays and class order in NPZ artifacts.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    project_root = DEFAULT_PROJECT_ROOT
    official_code = resolve_existing_path(
        Path(cfg["capture24_official_code"]),
        DEFAULT_PROJECT_ROOT / "capture24_project" / "official_code" / "capture24",
    )
    prepared_data = args.prepared_data if args.prepared_data is not None else Path(cfg["prepared_data"])
    baseline_results = DEFAULT_PROJECT_ROOT / "results" / "baselines"
    logs_dir = DEFAULT_PROJECT_ROOT / "logs" / "capture24_official_repro"
    baseline_results.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(official_code))
    from benchmark import train_test_split
    from classifier import Classifier
    from eval import metrics_report

    labels = parse_csv_arg(args.labels)
    models = parse_csv_arg(args.models)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"official_baselines_{timestamp}"
    if args.run_tag:
        run_id += f"_{args.run_tag}"
    metrics_path = baseline_results / f"{run_id}_metrics.csv"
    log_json_path = logs_dir / f"{run_id}.json"

    X = pd.read_pickle(prepared_data / "X_feats.pkl").values
    P = np.load(prepared_data / "P.npy")
    ys = {label: np.load(prepared_data / f"Y_{label}.npy") for label in labels}

    if args.smoke_test:
        rng = np.random.default_rng(args.seed)
        sample_size = max(1, int(0.01 * len(P)))
        idx = rng.integers(len(P), size=sample_size)
        X = X[idx]
        P = P[idx]
        ys = {label: y[idx] for label, y in ys.items()}
        run_id += "_smoke"

    train_mask, test_mask = train_test_split(P)
    X_train = X[train_mask]
    X_test = X[test_mask]
    P_train = P[train_mask]
    P_test = P[test_mask]

    metrics_rows: list[dict[str, object]] = []
    run_log: dict[str, object] = {
        "run_id": run_id,
        "timestamp": timestamp,
        "command": " ".join(sys.argv),
        "python": sys.executable,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "package_versions": package_versions(),
        "project_root": str(project_root),
        "official_code": str(official_code),
        "official_code_commit": git_short_sha(official_code),
        "prepared_data": str(prepared_data),
        "labels": labels,
        "models": models,
        "seed": args.seed,
        "nboots": args.nboots,
        "n_jobs": args.n_jobs,
        "model_n_jobs": args.model_n_jobs,
        "xgb_device": args.xgb_device,
        "xgb_tree_method": args.xgb_tree_method,
        "run_tag": args.run_tag,
        "save_proba": args.save_proba,
        "smoke_test": args.smoke_test,
        "n_windows": int(len(P)),
        "n_train_windows": int(train_mask.sum()),
        "n_test_windows": int(test_mask.sum()),
        "n_train_participants": int(len(np.unique(P_train))),
        "n_test_participants": int(len(np.unique(P_test))),
        "outputs": {},
    }

    for label in labels:
        y_train = ys[label][train_mask]
        y_test = ys[label][test_mask]
        prediction_artifacts: dict[str, np.ndarray] = {}

        for model in models:
            model_label = f"{model}_{args.run_tag}" if args.run_tag else model
            model_kwargs = {"n_jobs": args.model_n_jobs}
            if "xgb" in model.lower():
                if args.xgb_device:
                    model_kwargs["device"] = args.xgb_device
                if args.xgb_tree_method:
                    model_kwargs["tree_method"] = args.xgb_tree_method

            start = time.time()
            if args.optimisedir is not None:
                optimised_path = args.optimisedir / f"{model}_{label}.pkl"
            else:
                optimised_path = prepared_data / "__no_optimised_params__" / f"{model}_{label}.pkl"

            classifier = Classifier(
                model,
                args.seed,
                optimisedir=str(optimised_path),
                **model_kwargs,
            )
            classifier.fit(X_train, y_train, P_train)
            y_pred = classifier.predict(X_test, P_test)
            elapsed_seconds = time.time() - start

            prediction_artifacts[f"pred_{model_label}"] = y_pred
            if args.save_proba:
                proba = classifier.predict_proba(X_test, P_test)
                prediction_artifacts[f"proba_{model_label}"] = proba

                classes = None
                label_encoder = getattr(classifier.window_classifier, "le", None)
                window_model = getattr(classifier.window_classifier, "model", None)
                if label_encoder is not None and hasattr(label_encoder, "classes_"):
                    classes = label_encoder.classes_
                elif window_model is not None and hasattr(window_model, "classes_"):
                    classes = window_model.classes_
                elif getattr(classifier.smoother, "labels", None) is not None:
                    classes = classifier.smoother.labels
                if classes is not None:
                    prediction_artifacts[f"classes_{model_label}"] = np.asarray(classes)

            metric_dict = metrics_report(
                y_test,
                y_pred,
                P_test,
                tag=f"{label}/{model_label}",
                nboots=args.nboots,
                n_jobs=args.n_jobs,
                verbose=True,
            )

            for metric_name, metric_value in metric_dict.items():
                point, ci_low, ci_high = split_metric_interval(metric_value)
                metrics_rows.append(
                    {
                        "run_id": run_id,
                        "label_set": label,
                        "model": model_label,
                        "base_model": model,
                        "metric": metric_name,
                        "value": point,
                        "ci_low": ci_low,
                        "ci_high": ci_high,
                        "elapsed_seconds": elapsed_seconds,
                        "n_train_windows": int(len(y_train)),
                        "n_test_windows": int(len(y_test)),
                        "n_train_participants": int(len(np.unique(P_train))),
                        "n_test_participants": int(len(np.unique(P_test))),
                    }
                )

            write_metrics_csv(metrics_path, metrics_rows)
            run_log["outputs"]["metrics_csv_partial"] = str(metrics_path)
            with log_json_path.open("w", encoding="utf-8") as f:
                json.dump(run_log, f, indent=2, ensure_ascii=False)

        pred_path = baseline_results / f"{run_id}_{label}_predictions.npz"
        np.savez_compressed(
            pred_path,
            y_true=y_test,
            participant=P_test,
            **prediction_artifacts,
        )
        run_log["outputs"][f"{label}_predictions"] = str(pred_path)

    write_metrics_csv(metrics_path, metrics_rows)

    run_log["outputs"]["metrics_csv"] = str(metrics_path)
    with log_json_path.open("w", encoding="utf-8") as f:
        json.dump(run_log, f, indent=2, ensure_ascii=False)

    log_md_path = logs_dir / "reproduction_official_benchmark.md"
    summary = pd.DataFrame(metrics_rows)
    compact = summary.pivot_table(
        index=["label_set", "model"],
        columns="metric",
        values="value",
        aggfunc="first",
    ).reset_index()

    with log_md_path.open("a", encoding="utf-8") as f:
        f.write(f"\n\n## {run_id}\n\n")
        f.write(f"- Official code commit: `{run_log['official_code_commit']}`\n")
        f.write(f"- Python: `{sys.executable}` ({platform.python_version()})\n")
        f.write(f"- Prepared data: `{prepared_data}`\n")
        f.write(f"- Split: P001-P100 train/derivation, P101-P151 test\n")
        f.write(f"- Windows: train={train_mask.sum()}, test={test_mask.sum()}\n")
        f.write(f"- Metrics CSV: `{metrics_path}`\n")
        f.write(f"- Run metadata JSON: `{log_json_path}`\n\n")
        f.write(compact.to_markdown(index=False))
        f.write("\n")

    print(f"Saved metrics: {metrics_path}")
    print(f"Saved run metadata: {log_json_path}")
    print(f"Updated reproduction log: {log_md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
