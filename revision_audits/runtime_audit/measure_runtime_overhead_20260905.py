"""Measure post-training RF/HMM/CATD decoding time and Windows RSS.

The measurement uses the locked Capture-24 test split (P101--P151) and the
same saved RF/auxiliary probabilities used by the paper.  It deliberately
does not retrain a model or write prediction arrays; only a small timing CSV
and Markdown report are written to ``results/bcm``.
"""

from __future__ import annotations

import ctypes
from ctypes import wintypes
from pathlib import Path
import os
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
BASELINES = RESULTS / "baselines"
BCM = RESULTS / "bcm"
PREPARED = ROOT / "data" / "prepared_data_official_repro"


class _PROCESS_MEMORY_COUNTERS(ctypes.Structure):
    _fields_ = [
        ("cb", wintypes.DWORD),
        ("PageFaultCount", wintypes.DWORD),
        ("PeakWorkingSetSize", ctypes.c_size_t),
        ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t),
        ("PeakPagefileUsage", ctypes.c_size_t),
    ]


def memory_mb() -> tuple[float, float]:
    """Return (current RSS, peak RSS) in decimal MB on Windows."""
    counters = _PROCESS_MEMORY_COUNTERS()
    counters.cb = ctypes.sizeof(counters)
    psapi = ctypes.WinDLL("psapi.dll", use_last_error=True)
    fn = psapi.GetProcessMemoryInfo
    fn.argtypes = [wintypes.HANDLE, ctypes.POINTER(_PROCESS_MEMORY_COUNTERS), wintypes.DWORD]
    fn.restype = wintypes.BOOL
    ok = fn(ctypes.windll.kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb)
    if not ok:
        return float("nan"), float("nan")
    return counters.WorkingSetSize / 1e6, counters.PeakWorkingSetSize / 1e6


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


def transition_matrix(y, participant, classes, train_ids, smoothing=1e-3):
    mask = np.isin(participant, train_ids)
    index = {c: i for i, c in enumerate(classes)}
    mat = np.full((len(classes), len(classes)), smoothing, dtype="float64")
    for p in np.unique(participant[mask]):
        seq = y[mask & (participant == p)]
        for a, b in zip(seq[:-1], seq[1:]):
            if a in index and b in index:
                mat[index[a], index[b]] += 1.0
    return row_normalize(mat)


def viterbi_decode_proba(proba, participant, classes, trans, gamma, eps=1e-12):
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


def timed(label, fn, repeats=3):
    # Warm up allocation/cache effects, then report median and range.
    fn()
    values = []
    rss_before, _ = memory_mb()
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        values.append(time.perf_counter() - start)
    rss_after, peak = memory_mb()
    return {
        "stage": label,
        "repeats": repeats,
        "median_seconds": float(np.median(values)),
        "min_seconds": float(np.min(values)),
        "max_seconds": float(np.max(values)),
        "rss_delta_mb": float(rss_after - rss_before),
        "peak_rss_mb": float(peak),
    }


def latest(pattern_dir: Path, pattern: str) -> Path:
    files = sorted(pattern_dir.glob(pattern))
    if not files:
        raise FileNotFoundError(pattern)
    return files[-1]


def main() -> int:
    rf_npz = latest(BASELINES, "official_baselines_*rfcompat_xgbopt_full_WillettsSpecific2018_predictions.npz")
    coarse_npz = latest(BASELINES, "official_baselines_*rfcompat_xgbopt_full_Walmsley2020_predictions.npz")
    intensity_npz = latest(BCM, "auxiliary_xgb_*_y_met_intensity4_predictions.npz")
    labels_npz = latest(BCM, "capture24_derived_label_fields_*.npz")
    saved_npz = latest(BCM, "rf_temporal_current_*_locked_test_predictions.npz")

    rf = np.load(rf_npz, allow_pickle=True)
    coarse = np.load(coarse_npz, allow_pickle=True)
    intensity = np.load(intensity_npz, allow_pickle=True)
    labels = np.load(labels_npz, allow_pickle=True)
    saved = np.load(saved_npz, allow_pickle=True)
    P = np.load(PREPARED / "P.npy")
    y = np.load(PREPARED / "Y_WillettsSpecific2018.npy")
    y_coarse = np.load(PREPARED / "Y_Walmsley2020.npy")

    # Check that all saved locked arrays use the same row order.
    if not np.array_equal(rf["participant"], saved["participant"]):
        raise ValueError("RF and CATD participant order differ")
    fine_classes = rf["classes_rf_rfcompat_xgbopt_full"]
    coarse_classes = coarse["classes_rf_rfcompat_xgbopt_full"]
    intensity_classes = intensity["classes"]
    train100 = np.isin(P, [f"P{i:03d}" for i in range(1, 101)])
    train100_valid = train100 & labels["valid_annotation_mask"]
    fine_to_coarse = map_matrix(y, y_coarse, fine_classes, coarse_classes, train100_valid)
    fine_to_intensity = map_matrix(y, labels["y_met_intensity4"], fine_classes, intensity_classes, train100_valid)
    trans = transition_matrix(y, P, fine_classes, [f"P{i:03d}" for i in range(1, 101)])
    alpha = float(saved["selected_alpha"])
    beta = float(saved["selected_beta"])
    gamma = float(saved["selected_gamma"])
    fine_proba = rf["proba_rf_rfcompat_xgbopt_full"].astype("float64", copy=False)
    coarse_proba = coarse["proba_rf_rfcompat_xgbopt_full"].astype("float64", copy=False)
    intensity_proba = intensity["proba"].astype("float64", copy=False)
    participant = rf["participant"]

    def build_catd_emission():
        coarse_support = coarse_proba @ fine_to_coarse.T
        intensity_support = intensity_proba @ fine_to_intensity.T
        return row_normalize(
            fine_proba
            * np.power(np.maximum(coarse_support, 1e-12), alpha)
            * np.power(np.maximum(intensity_support, 1e-12), beta)
        )

    catd_emission = build_catd_emission()
    rows = []
    rows.append(timed("CATD support projection + emission", build_catd_emission))
    rows.append(timed("RF+HMM Viterbi decode (control)", lambda: viterbi_decode_proba(fine_proba, participant, fine_classes, trans, gamma)))
    rows.append(timed("CATD Viterbi decode", lambda: viterbi_decode_proba(catd_emission, participant, fine_classes, trans, gamma)))

    out_csv = BCM / "runtime_overhead_capture24_20260905.csv"
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        f.write("stage,repeats,median_seconds,min_seconds,max_seconds,rss_delta_mb,peak_rss_mb\n")
        for r in rows:
            f.write(",".join(str(r[k]) for k in ["stage", "repeats", "median_seconds", "min_seconds", "max_seconds", "rss_delta_mb", "peak_rss_mb"]) + "\n")
    # Use the formal full 32-feature RF/XGBoost run rather than a later smoke
    # file when linking the end-to-end context values.
    end_to_end_csv = BASELINES / "official_baselines_20260629_164856_rfcompat_xgbopt_full_metrics.csv"
    report = BCM / "runtime_overhead_capture24_20260905.md"
    with report.open("w", encoding="utf-8") as f:
        f.write("# Capture-24 runtime and memory audit\n\n")
        f.write("The post-training measurements below use the locked P101--P151 rows and the exact saved probabilities used for the manuscript. Each stage has one warm-up and three timed repeats; medians are reported. RSS is the Windows process working-set peak of this measurement process.\n\n")
        f.write("## Post-training decoding\n\n")
        f.write("| Stage | Median (s) | Min--max (s) | Peak RSS (MB) |\n|---|---:|---:|---:|\n")
        for r in rows:
            f.write(f"| {r['stage']} | {r['median_seconds']:.4f} | {r['min_seconds']:.4f}--{r['max_seconds']:.4f} | {r['peak_rss_mb']:.1f} |\n")
        f.write("\nThe official end-to-end baseline log (model fitting plus prediction) is retained separately in `" + str(end_to_end_csv) + "`; its elapsed values are not conflated with the post-training timings above.\n")
    print(report)
    print(out_csv)
    for r in rows:
        print(r)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
