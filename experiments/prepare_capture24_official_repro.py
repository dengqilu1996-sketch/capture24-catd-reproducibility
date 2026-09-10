"""Generate Capture-24 prepared_data with the clean official preprocessing code.

The official CLI downloads capture24.zip before checking whether the extracted
CSV files already exist. In this project the raw CSV files are already present,
so this wrapper imports and calls the official preprocessing function directly.
It does not use legacy project Python code.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OFFICIAL_ROOT = PROJECT_ROOT / "capture24_project" / "official_code" / "capture24"
DEFAULT_RAW_PARENT = Path(r"D:\claude项目\capture24_work2\capture24\data")
DEFAULT_OUTDIR = PROJECT_ROOT / "data" / "prepared_data_official_repro"
DEFAULT_LOGDIR = PROJECT_ROOT / "logs" / "capture24_official_repro"


def git_commit(path: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "--short", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unavailable"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-parent", type=Path, default=DEFAULT_RAW_PARENT)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--annots", default="Walmsley2020,WillettsSpecific2018")
    parser.add_argument("--winsec", type=int, default=10)
    parser.add_argument("--n-jobs", type=int, default=8)
    parser.add_argument("--joblib-backend", default="threading")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--logdir", type=Path, default=DEFAULT_LOGDIR)
    args = parser.parse_args()

    if not (args.raw_parent / "capture24").exists():
        raise FileNotFoundError(f"Expected raw capture24 directory under {args.raw_parent}")
    if not (args.raw_parent / "capture24" / "annotation-label-dictionary.csv").exists():
        raise FileNotFoundError("annotation-label-dictionary.csv not found in raw capture24 directory")

    sys.path.insert(0, str(OFFICIAL_ROOT))
    from joblib import parallel_backend
    from prepare_data import load_all_and_make_windows

    args.logdir.mkdir(parents=True, exist_ok=True)
    args.outdir.mkdir(parents=True, exist_ok=True)

    run_meta = {
        "started_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "official_code": str(OFFICIAL_ROOT),
        "official_commit": git_commit(OFFICIAL_ROOT),
        "raw_parent": str(args.raw_parent),
        "outdir": str(args.outdir),
        "annots": args.annots.split(","),
        "winsec": args.winsec,
        "n_jobs": args.n_jobs,
        "joblib_backend": args.joblib_backend,
        "overwrite": args.overwrite,
        "python": sys.executable,
        "python_version": platform.python_version(),
    }
    meta_path = args.logdir / "prepare_capture24_official_repro_metadata.json"
    meta_path.write_text(json.dumps(run_meta, indent=2, ensure_ascii=False), encoding="utf-8")

    with parallel_backend(args.joblib_backend):
        load_all_and_make_windows(
            str(args.raw_parent),
            args.annots.split(","),
            str(args.outdir),
            args.n_jobs,
            overwrite=args.overwrite,
            winsec=args.winsec,
        )

    run_meta["finished_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
    meta_path.write_text(json.dumps(run_meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote prepared_data to {args.outdir}")
    print(f"Wrote metadata to {meta_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
