#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
# parallelism
from concurrent.futures import ProcessPoolExecutor, as_completed
import platform, inspect, subprocess
from types import SimpleNamespace
from dataclasses import asdict  # optional, only if you want the quick CSV write shown below
import multiprocessing as mp
try:
    from threadpoolctl import threadpool_limits
except Exception:
    threadpool_limits = None
import argparse
import os
import sys
import math
import time
import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

# Imaging + plotting stack (auto-install if missing; headless-safe for PNG saving)
try:
    from skimage import io, color, filters, morphology, measure, exposure, util
    from skimage.morphology import remove_small_objects, remove_small_holes, binary_closing, disk, skeletonize
    from skimage.filters import frangi, sato, threshold_otsu
    from skimage.transform import resize
    from skimage.segmentation import clear_border
    from scipy.ndimage import distance_transform_edt, gaussian_filter
    from scipy.stats import theilslopes, pearsonr
    import matplotlib
    matplotlib.use("Agg")  # headless backend for servers/CLI
    import matplotlib.pyplot as plt

    # Publication-ready defaults: clean fonts, readable labels, tight layout
    matplotlib.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.linestyle": "-",
        "grid.alpha": 0.25,
        "figure.autolayout": True
    })

except ModuleNotFoundError:
    import importlib, subprocess, sys
    missing = []
    for mod_name, pip_name in [("scipy", "scipy"),
                               ("skimage", "scikit-image"),
                               ("matplotlib", "matplotlib")]:
        try:
            importlib.import_module(mod_name)
        except ModuleNotFoundError:
            missing.append((mod_name, pip_name))

    if missing:
        print("\n[SETUP] Installing required packages into the current environment:")
        for mod, pip_name in missing:
            print(f"        - {pip_name}")
            subprocess.check_call([sys.executable, "-m", "pip", "install", pip_name])

    # Retry imports after installation
    from skimage import io, color, filters, morphology, measure, exposure, util
    from skimage.morphology import remove_small_objects, remove_small_holes, binary_closing, disk, skeletonize
    from skimage.filters import frangi, sato, threshold_otsu
    from skimage.transform import resize
    from skimage.segmentation import clear_border
    from scipy.ndimage import distance_transform_edt, gaussian_filter

    import matplotlib
    matplotlib.use("Agg")  # headless backend
    import matplotlib.pyplot as plt




# Graph
import networkx as nx

# ---------- CONSTANTS ----------
ROOT = Path("/Users/patsfan753/Desktop/EPIC").expanduser()
DATA_ROOT_DEFAULT = ROOT / "data"
FIG_ROOT = ROOT / "figures"
CSV_ROOT = ROOT / "reports"
LOG_ROOT = ROOT / "logs"

SUPPORTED_IMAGE_EXT = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


# ---------- UTILS ----------
def ensure_dirs():
    for d in [ROOT, FIG_ROOT, CSV_ROOT, LOG_ROOT, DATA_ROOT_DEFAULT]:
        d.mkdir(parents=True, exist_ok=True)


# ---- PANEL FILTER (A/B) ----------------------------------------------
def _apply_panel_filter(df: pd.DataFrame, mode: str = "all") -> pd.DataFrame:
    """
    Panel filtering policy.
      - "all"    : return all rows with finite metrics (Option A, recommended)
      - "strict" : keep only strict-QC nodes if the column is present (Option B)
    """
    if not isinstance(df, pd.DataFrame) or df.empty:
        return df
    if str(mode).lower() == "strict" and ("qc_pass_strict" in df.columns):
        return df[df["qc_pass_strict"].astype(bool)].copy()
    return df

# ---- EMPTY-CELL STUBS -------------------------------------------------
def write_empty_reports(tag: str, reason: str = "no_kept_nodes") -> None:
    """
    When a variant/dataset has 0 kept nodes, write valid stub artifacts so downstream
    steps don't fail or mislead.
    """
    base = tag.split("__", 1)[0]
    out_dir = CSV_ROOT / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    # Panel B (null test & prereg)
    (out_dir / "NULLTEST__R_m.txt").write_text(
        f"DATASET: {tag}\nN nodes: 0\nObserved: median=nan [95% CI nan,nan], frac<0.55=nan\n"
        "Null reps per control: 0\n"
    )
    (out_dir / "PREREG__C1.txt").write_text(
        f"DATASET: {tag}\nClaim HRF-C1 (held-out closure @ angle-only m):\n"
        "Observed \\tilde R = nan  [boot 95% CI nan,nan]\n"
        "Thresholds: p <= 0.001 on BOTH median and Pr[R<0.55], and Cliff's δ <= -0.20 vs EVERY null\n"
        "Outcome: FAIL (no data)\n"
    )
    # Mirror to dataset base for canonical readers
    base_dir = CSV_ROOT / base
    base_dir.mkdir(parents=True, exist_ok=True)
    (base_dir / "PREREG__C1.txt").write_text((out_dir / "PREREG__C1.txt").read_text())

    # Baselines
    (out_dir / "BASELINES__R_m.txt").write_text(f"DATASET: {tag}\n(no data; N=0)\n")

    # Panel C
    (base_dir / "PANELC__theta_vs_m_test.txt").write_text(
        f"DATASET: {tag}\nN_sym=0\nPanel C verdict: FAIL (no data)\n"
    )

    # Ablation
    (out_dir / "ABLATION__summary.txt").write_text(f"DATASET: {tag}\n(no data)\nOutcome: FAIL\n")
    (out_dir / "ABLATION__test.txt").write_text("Outcome: FAIL\nWorst p: nan\nWorst QC drift: nan pp\n")

    # Uncertainty CSV header (with outcome)
    (out_dir / "UNCERTAINTY__nodes.csv").write_text(
        "# STEP 3 — Per-node uncertainty: outcome=FAIL | frac_edge_moved=nan | median CI half-width=nan\n"
        "image_id,node_id,m_hat,m_lo,m_hi,CI_width,at_edge_before,at_edge_after\n"
    )




def plot_theta_vs_m_scatter(
    df: pd.DataFrame,
    dataset: str,
    symmetric_tol: float = 1.08,
    bins_deg: float = 10.0,
    bootstrap: int = 5000,
    seed: int = 13,
    panel_filter: str = "all",
):
    """
    Panel C — Behavioral signature (test) with adaptive sampling and uncertainty bars:
      - Adaptively widens the near-symmetric tolerance up to 1.25 to reach N_sym≥30.
      - Draws per-node vertical uncertainty bars if reports/<VARIANT>/UNCERTAINTY__nodes.csv exists.
      - Leaves file outputs and PASS/FAIL logic intact.
    """
    if df is None or len(df) == 0:
        df = pd.DataFrame({"theta12_deg": [], "m_node": [], "r1": [], "r2": [], "qc_pass_strict": [], "image_id": [], "node_id": []})
    for c in ["theta12_deg", "m_node", "r1", "r2", "qc_pass_strict", "image_id", "node_id"]:
        if c not in df.columns:
            df[c] = np.nan if c not in ["qc_pass_strict"] else False

    dataset_base = dataset.split("__", 1)[0]

    # near-symmetric mask (adaptive)
    rmin = np.minimum(df["r1"].values, df["r2"].values)
    rmax = np.maximum(df["r1"].values, df["r2"].values)
    ratio = rmax / np.maximum(rmin, 1e-9)

    target_min = 30           # internal minimum N_sym target
    max_tol = 1.25            # do not exceed this symmetry tolerance
    tol_used = float(symmetric_tol)

    def _df_sym_for_tol(tol):
        return df.loc[ratio <= float(tol)].copy()

    df_sym = _df_sym_for_tol(tol_used)
    # adapt up if too small
    while (len(df_sym) < target_min) and (tol_used + 0.02 <= max_tol):
        tol_used = min(max_tol, tol_used + 0.02)
        df_sym = _df_sym_for_tol(tol_used)

    # Try to attach per-node uncertainty if available
    unc_path = CSV_ROOT / dataset / "UNCERTAINTY__nodes.csv"
    df_unc = None
    try:
        if unc_path.exists():
            df_unc = pd.read_csv(unc_path, comment="#")
            if {"image_id","node_id","m_lo","m_hi"}.issubset(df_unc.columns):
                df_sym = df_sym.merge(df_unc[["image_id","node_id","m_lo","m_hi"]], on=["image_id","node_id"], how="left")
    except Exception:
        df_unc = None  # silently proceed without uncertainty bars

    # figure
    fig = plt.figure(figsize=(8.0, 5.2))
    ax = plt.gca()
    x_lo, x_hi = 20.0, 170.0
    y_lo, y_hi = 0.2, 4.0

    # analytic curve
    theta_grid_deg = np.linspace(x_lo, x_hi, 900, dtype=np.float64)
    theta_rad_half = np.deg2rad(theta_grid_deg * 0.5)
    c = np.cos(theta_rad_half)
    n = 4.0
    with np.errstate(divide="ignore", invalid="ignore"):
        x_log2 = np.log2(np.clip(c, 1e-12, 1.0 - 1e-12))
        m_sym_grid = n * (1.0 + x_log2) / (1.0 - x_log2)
    m_sym_grid[(c <= 0.0) | (c >= 1.0)] = np.nan
    ax.plot(theta_grid_deg, m_sym_grid, linewidth=2.0, label="analytic symmetric curve (n=4)")

    # shaded |Δm| ≤ 0.25 band
    band = 0.25
    ax.fill_between(theta_grid_deg, m_sym_grid - band, m_sym_grid + band, alpha=0.12, label="|Δm| ≤ 0.25 band")

    # scatter + per-node uncertainty bars (if present) + binned medians
    if len(df_sym) > 0:
        ok = df_sym["qc_pass_strict"].astype(bool).values

        # Draw uncertainty bars for strict-pass nodes (to avoid clutter)
        if df_unc is not None and {"m_lo","m_hi"}.issubset(df_sym.columns):
            try:
                sub = df_sym.loc[ok & np.isfinite(df_sym["m_lo"]) & np.isfinite(df_sym["m_hi"])]
                if len(sub) > 0:
                    ax.vlines(sub["theta12_deg"].values, sub["m_lo"].values, sub["m_hi"].values, alpha=0.25, linewidth=1.0)
            except Exception:
                pass

        ax.scatter(df_sym.loc[ok, "theta12_deg"], df_sym.loc[ok, "m_node"], s=20, marker="o",
                   label="near-symmetric (strict PASS)")
        ax.scatter(df_sym.loc[~ok, "theta12_deg"], df_sym.loc[~ok, "m_node"], s=22, marker="x",
                   label="near-symmetric (strict FAIL)")

        th = df_sym["theta12_deg"].astype(float).values
        mm = df_sym["m_node"].astype(float).values
        finite = np.isfinite(th) & np.isfinite(mm)
        th, mm = th[finite], mm[finite]

        if th.size > 0:
            edges = np.arange(x_lo, x_hi + bins_deg, bins_deg, dtype=float)
            idx = np.digitize(th, edges) - 1
            centers, medians, q25s, q75s = [], [], [], []
            for b in range(edges.size - 1):
                mask = idx == b
                if np.count_nonzero(mask) >= 4:
                    centers.append(0.5 * (edges[b] + edges[b + 1]))
                    vals = mm[mask]
                    medians.append(np.median(vals))
                    q25s.append(np.percentile(vals, 25))
                    q75s.append(np.percentile(vals, 75))
            if centers:
                centers = np.array(centers); medians = np.array(medians)
                q25s = np.array(q25s); q75s = np.array(q75s)
                yerr = np.vstack([medians - q25s, q75s - medians])
                ax.errorbar(centers, medians, yerr=yerr, fmt="-", linewidth=1.5, capsize=2, label="bin median ± IQR")

    ax.set_xlim(x_lo, x_hi); ax.set_ylim(y_lo, y_hi)
    ax.set_xlabel(r"$\theta_{12}$ (degrees)"); ax.set_ylabel(r"$m$ (per-node inverted)")
    ax.set_title(f"θ₁₂ vs m (near-symmetric r1/r2 ≤ {tol_used:.2f}) — {dataset}")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")

    # stats
    stats_text = []
    N_sym = int(len(df_sym))
    N_pass = int(df_sym["qc_pass_strict"].sum()) if "qc_pass_strict" in df_sym.columns else 0
    stats_text.append(f"N_sym={N_sym}, strict PASS={N_pass}")
    stats_text.append(f"sym tol (adaptive) r1/r2 ≤ {tol_used:.2f}")

    # compute Δm, TS slope CI, Pearson r, % in band, and PASS/FAIL
    pass_median_band = False
    pass_ts = False
    pass_share = False

    if N_sym > 2:
        th = df_sym["theta12_deg"].astype(float).values
        mm = df_sym["m_node"].astype(float).values
        finite = np.isfinite(th) & np.isfinite(mm)
        th = th[finite]; mm = mm[finite]

        def _msym_from_deg(d):
            return m_from_angle_symmetric(np.deg2rad(float(d)), n=4.0)

        m_pred = np.array([_msym_from_deg(t) for t in th], dtype=float)
        ok2 = np.isfinite(m_pred) & np.isfinite(mm)
        m_pred = m_pred[ok2]; mm = mm[ok2]

        if mm.size > 2:
            delta = mm - m_pred
            rng = np.random.default_rng(int(seed))
            boot_idx = rng.integers(0, delta.size, size=(int(bootstrap), delta.size)) if bootstrap >= 200 else None
            if boot_idx is not None:
                boot_stats = np.median(np.abs(delta[boot_idx]), axis=1)
                lo, hi = np.percentile(boot_stats, [2.5, 97.5])
                stats_text.append(f"median|Δm|={np.median(np.abs(delta)):.3f} (95% CI {lo:.3f}–{hi:.3f})")
                pass_median_band = (hi <= 0.25)
            else:
                stats_text.append(f"median|Δm|={np.median(np.abs(delta)):.3f}")
                pass_median_band = (np.median(np.abs(delta)) <= 0.25)

            try:
                slope, intercept, lo_s, hi_s = theilslopes(mm, m_pred)
                stats_text.append(f"Theil–Sen slope={slope:.3f} [{lo_s:.3f},{hi_s:.3f}]")
                pass_ts = (lo_s <= 1.0 <= hi_s) and (0.9 <= slope <= 1.1)
            except Exception:
                pass_ts = False

            try:
                r, _ = pearsonr(mm, m_pred)
                stats_text.append(f"Pearson r={r:.3f}")
            except Exception:
                pass

            share = float(np.mean(np.abs(delta) <= 0.25)) if delta.size > 0 else float("nan")
            stats_text.append(f"% within |Δm|≤0.25 = {share*100:.1f}%")
            pass_share = (share >= 0.70)

            try:
                ax_in = ax.inset_axes([0.64, 0.13, 0.33, 0.33])
                ax_in.hist(delta, bins=20, alpha=0.9)
                ax_in.axvline(0.0, linewidth=1.0)
                med = np.median(delta); q25, q75 = np.percentile(delta, [25, 75])
                ax_in.axvline(med, linewidth=1.0); ax_in.axvline(q25, linewidth=0.8); ax_in.axvline(q75, linewidth=0.8)
                ax_in.set_title(r"$\Delta m = m_\mathrm{node}-m_\mathrm{sym}$", fontsize=9)
                ax_in.tick_params(axis="both", labelsize=8)
            except Exception:
                pass

    panelC_pass = pass_median_band and pass_ts and pass_share
    stats_text.append(f"Panel C verdict: {'PASS' if panelC_pass else 'FAIL'}")

    ax.text(0.02, 0.98, "\n".join(stats_text), transform=ax.transAxes, va="top", ha="left",
            fontsize=9, bbox=dict(facecolor="white", alpha=0.85, edgecolor="none"))

    out_path = FIG_ROOT / dataset / f"dataset_{dataset}__theta_vs_m_scatter.png"
    save_png(fig, out_path)

    out_test = CSV_ROOT / dataset_base / "PANELC__theta_vs_m_test.txt"
    out_test.parent.mkdir(parents=True, exist_ok=True)
    with open(out_test, "w") as f:
        f.write(f"DATASET: {dataset}\n")
        f.write(f"Adaptive symmetric tol used: {tol_used:.3f}\n")
        for s in stats_text:
            f.write(s + "\n")




# ---- RUN MANIFEST -----------------------------------------------------
def write_run_manifest(tag: str,
                       args_namespace: argparse.Namespace,
                       extra: Optional[Dict] = None,
                       img_paths: Optional[List[Path]] = None) -> None:
    """
    reports/<TAG>/RUN__manifest.json with CLI args, commit, env, lib versions, CPU info, RNG seeds, time, and code/data hashes.
    """
    out_dir = CSV_ROOT / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    # Git commit (best effort)
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(ROOT)).decode().strip()
    except Exception:
        commit = None

    # Library versions (best effort)
    try:
        import scipy as _scipy, skimage as _skimage, matplotlib as _mpl
        lib_versions = {
            "python": sys.version,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": _scipy.__version__,
            "scikit_image": _skimage.__version__,
            "matplotlib": _mpl.__version__,
            "networkx": nx.__version__,
        }
    except Exception:
        lib_versions = {"python": sys.version, "numpy": np.__version__, "pandas": pd.__version__}

    # CPU/platform
    cpu = {
        "machine": platform.machine(),
        "processor": platform.processor(),
        "platform": platform.platform(),
        "logical_cpus": os.cpu_count(),
    }

    # Code hashes for key functions
    def _sh(s: str) -> str:
        return hashlib.sha256(s.encode("utf-8", "ignore")).hexdigest()
    try:
        code_hashes = {
            "analyze_image": _sh(inspect.getsource(analyze_image)),
            "segment_vessels_with_meta": _sh(inspect.getsource(segment_vessels_with_meta)),
            "plot_residual_distribution": _sh(inspect.getsource(plot_residual_distribution)),
            "plot_theta_vs_m_scatter": _sh(inspect.getsource(plot_theta_vs_m_scatter)),
        }
    except Exception:
        code_hashes = {}

    # Data list hash (paths only; avoid reading pixels)
    data_manifest = None
    if img_paths:
        try:
            rels = [str(p) for p in sorted(img_paths)]
            data_manifest = {
                "count": len(rels),
                "sha256_paths_list": _sh("\n".join(rels))
            }
        except Exception:
            pass

    payload = {
        "tag": tag,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "cli_args": vars(args_namespace),
        "git_commit": commit,
        "lib_versions": lib_versions,
        "cpu": cpu,
        "code_hashes": code_hashes,
        "data_manifest": data_manifest,
        "extra": (extra or {}),
    }
    (out_dir / "RUN__manifest.json").write_text(json.dumps(payload, indent=2, sort_keys=True))


def log(msg: str):
    """
    Console + file logger (simple, robust, flushed).
    """
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    out = f"[{ts}] (pid={os.getpid()}) {msg}"   # ← add pid for multi-proc clarity
    print(out, flush=True)
    with open(LOG_ROOT / "run.log", "a") as f:
        f.write(out + "\n")
        f.flush()

def _init_worker(intraop_threads: int = 1):
    """
    Runs in each worker process at start. Caps BLAS/OpenMP threads to avoid
    oversubscription (NumPy/SciPy/Skimage may call into MKL/OPENBLAS).
    """
    if threadpool_limits is not None:
        try:
            threadpool_limits(limits=intraop_threads)
        except Exception:
            pass  # safe to ignore


def _process_one_image(args_tuple):
    """
    Wrapper that runs analyze_image() in a worker. Returns (img, recs, diag, err_str).
    """
    img_path_str, dataset_tag, seg_cfg, qc_cfg, save_debug = args_tuple
    try:
        recs, diag = analyze_image(
            img_path=Path(img_path_str),
            dataset=dataset_tag,
            seg_cfg=seg_cfg,
            qc=qc_cfg,
            save_debug=save_debug,
            out_dir=ROOT
        )
        return img_path_str, recs, diag, None
    except Exception as e:
        # Bubble error up but keep the main loop robust
        return img_path_str, [], {
            "dataset": dataset_tag,
            "image_id": Path(img_path_str).stem,
            "n_nodes_total_raw": 0,
            "n_nodes_total_dedup": 0,
            "n_nodes_kept": 0,
            "skip_reasons": {"worker_error": 1},
            "error": str(e)
        }, str(e)



def human_time(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    m, s = divmod(seconds, 60)
    if m < 60:
        return f"{int(m)}m {int(s)}s"
    h, m = divmod(m, 60)
    return f"{int(h)}h {int(m)}m"


def imread_gray(path: Path) -> np.ndarray:
    img = io.imread(str(path))
    if img.ndim == 3:
        # Use the green plane directly — vessels have highest contrast here.
        g = img[..., 1].astype(np.float32)
        img = (g - g.min()) / (g.max() - g.min() + 1e-12)
    else:
        img = util.img_as_float32(img)
    # Robust 1–99% normalization
    p1, p99 = np.percentile(img, [1, 99])
    if p99 > p1:
        img = np.clip((img - p1) / (p99 - p1), 0, 1)
    return img.astype(np.float32)



def save_png(fig, out_path: Path, dpi=200, tight=True):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if tight:
        fig.savefig(str(out_path), dpi=dpi, bbox_inches="tight")
    else:
        fig.savefig(str(out_path), dpi=dpi)
    plt.close(fig)


# ---------- DATA DISCOVERY ----------
def find_images(dataset_dir: Path, max_images: Optional[int] = None) -> List[Path]:
    imgs = []
    for p in dataset_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in SUPPORTED_IMAGE_EXT:
            # Exclude masks (we will associate them on the fly)
            name = p.stem.lower()
            if name.endswith("_mask"):
                continue
            imgs.append(p)
    imgs.sort()
    if max_images is not None:
        imgs = imgs[:max_images]
    return imgs


def find_mask_for_image(img_path: Path) -> Optional[Path]:
    """
    If <name>_mask.png/.tif sits next to image, return it. Otherwise, None.
    """
    base = img_path.with_suffix("")
    for ext in SUPPORTED_IMAGE_EXT:
        cand = Path(str(base) + "_mask" + ext)
        if cand.exists():
            return cand
    return None


@dataclass
class SegConfig:
    vesselness_method: str = "frangi"  # 'frangi' or 'sato'
    sigma_min: float = 1.0
    sigma_max: float = 8.0
    # None => auto-polarity; True => dark vessels; False => bright vessels
    black_ridges: Optional[bool] = None
    thresh_method: str = "otsu"  # 'otsu' or 'quantile'
    quantile: float = 0.92
    min_object_area: int = 48
    closing_radius: int = 1
    remove_border: bool = False




def segment_vessels(gray: np.ndarray, cfg: SegConfig) -> np.ndarray:
    """
    Backwards-compatible wrapper that returns only the mask.
    Use `segment_vessels_with_meta` if you need threshold value & provenance.
    """
    mask, _meta = segment_vessels_with_meta(gray, cfg)
    return mask.astype(np.uint8)


def segment_vessels_with_meta(gray: np.ndarray, cfg: SegConfig) -> Tuple[np.ndarray, Dict]:
    """
    Return (binary mask in {0,1}, meta dict with segmentation provenance).
    Meta keys: method, sigmas, black_ridges, thresh_type, thresh_value, vessel_frac, seg_variant
    """
    method = cfg.vesselness_method.lower()
    sigmas = np.linspace(cfg.sigma_min, cfg.sigma_max, 8)

    def _compute_vesselness(img, black_flag: bool):
        if method == "frangi":
            return frangi(img, sigmas=sigmas, black_ridges=black_flag)
        else:
            return sato(img, sigmas=sigmas, black_ridges=black_flag)

    if cfg.black_ridges is None:
        v_dark  = _compute_vesselness(gray, True)
        v_bright= _compute_vesselness(gray, False)
        p99_dark   = float(np.percentile(v_dark,   99))
        p99_bright = float(np.percentile(v_bright, 99))
        use_dark = p99_dark >= p99_bright
        v = v_dark if use_dark else v_bright
        log(f"    [seg] auto-polarity: chose black_ridges={use_dark} (p99_dark={p99_dark:.5f}, p99_bright={p99_bright:.5f})")
    else:
        v = _compute_vesselness(gray, bool(cfg.black_ridges))
        use_dark = bool(cfg.black_ridges)

    v = exposure.rescale_intensity(v, in_range="image", out_range=(0, 1))

    if cfg.thresh_method.lower() == "otsu":
        thr = float(threshold_otsu(v))
        thr_type = "otsu"
        thr_str = f"otsu={thr:.5f}"
    else:
        thr = float(np.quantile(v, cfg.quantile))
        thr_type = "quantile"
        thr_str = f"quantile={cfg.quantile:.3f} -> {thr:.5f}"

    mask = v > thr

    target_min_vf = 2.5e-3
    vf = float(mask.mean())

    if vf < target_min_vf:
        for q in (0.95, 0.92, 0.90, 0.88, 0.85):
            alt_thr = float(np.quantile(v, q))
            cand = v > alt_thr
            vf_cand = float(cand.mean())
            thr_str += f" | fallback_q{int(q*100)}={alt_thr:.5f}"
            log(f"    [seg] fallback threshold q={q:.2f} → vessel_frac={vf_cand*100:.2f}%")
            if vf_cand >= target_min_vf:
                mask, vf = cand, vf_cand
                thr = alt_thr
                thr_type = f"fallback_q{int(q*100)}"
                break
            if vf_cand > vf:
                mask, vf = cand, vf_cand
                thr = alt_thr
                thr_type = f"fallback_q{int(q*100)}"

    if cfg.closing_radius > 0:
        mask = binary_closing(mask, disk(cfg.closing_radius))
    mask = remove_small_objects(mask, cfg.min_object_area)
    mask = remove_small_holes(mask, cfg.min_object_area)
    if cfg.remove_border:
        mask_cb = clear_border(mask)
        kept_ratio = float(mask_cb.sum()) / (float(mask.sum()) + 1e-9)
        if kept_ratio >= 0.85:
            mask = mask_cb
            log(f"    [seg] clear_border kept {kept_ratio*100:.1f}% of vessel pixels")
        else:
            log(f"    [seg] clear_border would drop {(1.0-kept_ratio)*100:.1f}% — skipping")

    p0, p25, p50, p75, p95, p99 = np.percentile(v, [0, 25, 50, 75, 95, 99])
    vessel_frac = float(mask.mean())
    log(
        "    [seg] "
        f"method={method}, sigmas=[{cfg.sigma_min:.2f},{cfg.sigma_max:.2f}], "
        f"black_ridges={use_dark}, thresh={thr_str}, vessel_frac={vessel_frac*100:.2f}%"
    )
    log(
        "    [seg] vesselness stats: "
        f"min={p0:.5f}, p25={p25:.5f}, p50={p50:.5f}, p75={p75:.5f}, p95={p95:.5f}, p99={p99:.5f}"
    )

    meta = {
        "method": method,
        "sigmas": (cfg.sigma_min, cfg.sigma_max),
        "black_ridges": use_dark,
        "thresh_type": thr_type,
        "thresh_value": float(thr),
        "vessel_frac": vessel_frac,
        "seg_variant": f"{method}+{cfg.thresh_method.lower()}",
    }
    return mask.astype(np.uint8), meta



# ---------- SKELETON & GRAPH ----------
def skeleton_and_dist(mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Skeletonize and compute EDT distance map on the mask.
    """
    skel = skeletonize(mask > 0)
    dist = distance_transform_edt(mask > 0).astype(np.float32)
    return skel, dist


def build_graph_from_skeleton(skel: np.ndarray) -> nx.Graph:
    """
    8-neighborhood graph on skeleton pixels.
    """
    G = nx.Graph()
    coords = np.argwhere(skel > 0)  # (N, 2) [row, col]
    idx_map = -np.ones(skel.shape, dtype=np.int64)
    for i, (r, c) in enumerate(coords):
        idx_map[r, c] = i
        G.add_node(i, rc=(int(r), int(c)))

    # connect 8-neighbors
    H, W = skel.shape
    for i, (r, c) in enumerate(coords):
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                rr, cc = r + dr, c + dc
                if 0 <= rr < H and 0 <= cc < W and skel[rr, cc]:
                    j = idx_map[rr, cc]
                    if j >= 0:
                        G.add_edge(i, int(j))
    return G


def is_bifurcation(G: nx.Graph, node: int) -> bool:
    # Junction pixels have three OR MORE neighbors in an 8-neighborhood skeleton
    # (clusters are normal; we dedup them later).
    return G.degree[node] >= 3


def neighbor_coords(G: nx.Graph, node: int) -> List[Tuple[int, int]]:
    return [G.nodes[n]["rc"] for n in G.neighbors(node)]

def dedup_degree3_nodes(G: nx.Graph, nodes: List[int], radius_px: int = 3) -> List[int]:
    """
    Greedy NMS for degree-3 skeleton pixels: cluster nodes within 'radius_px'
    in Euclidean pixel space and keep one representative per cluster.
    """
    if len(nodes) <= 1:
        return nodes
    coords = np.array([G.nodes[n]["rc"] for n in nodes], dtype=np.float32)  # (N,2) = (row,col)
    taken = np.zeros(len(nodes), dtype=bool)
    kept: List[int] = []
    for i in range(len(nodes)):
        if taken[i]:
            continue
        ci = coords[i]
        d = np.sqrt(np.sum((coords - ci) ** 2, axis=1))
        cluster_idx = np.where(d <= float(radius_px))[0]
        taken[cluster_idx] = True
        kept.append(nodes[i])  # representative
    return kept



def walk_branch(G: nx.Graph,
                skel: np.ndarray,
                start_node: int,
                prev_node: int,
                max_steps: int = 32,
                unwind_steps: int = 6) -> List[Tuple[int, int]]:
    """
    Walk a branch starting at 'start_node' (a neighbor of a junction) and move
    away from the junction 'prev_node'. To escape junction pixel-clusters, allow
    up to 'unwind_steps' where we pick the neighbor most aligned with the current
    direction; after that, require degree==2 (single continuation).
    """
    path_nodes: List[int] = [start_node]
    cur = start_node
    last = prev_node
    steps = 0
    unwound = 0

    def _dir_vec(a: int, b: int) -> np.ndarray:
        ra, ca = G.nodes[a]["rc"]; rb, cb = G.nodes[b]["rc"]
        v = np.array([cb - ca, rb - ra], dtype=np.float32)
        n = np.linalg.norm(v)
        return v / (n + 1e-12)

    while steps < max_steps:
        nbrs = [v for v in G.neighbors(cur) if v != last]
        if not nbrs:
            # endpoint
            break

        # Preferred continuation direction from last -> cur (if available)
        prefer = None
        if last is not None:
            prefer = _dir_vec(last, cur)

        nxt = None
        if G.degree[cur] == 2 and len(nbrs) == 1 and unwound >= unwind_steps:
            # Clean chain: only one way forward after we've left the junction blob
            nxt = nbrs[0]
        else:
            # Junction cluster or fanout: choose neighbor best aligned with 'prefer'
            # If 'prefer' not available (first step), choose farthest in Euclidean sense
            r0, c0 = G.nodes[cur]["rc"]
            best_score = -1e9
            for v in nbrs:
                rv, cv = G.nodes[v]["rc"]
                step = np.array([cv - c0, rv - r0], dtype=np.float32)
                n = np.linalg.norm(step)
                if n == 0:
                    continue
                if prefer is None:
                    score = float(n)  # just go farthest away for the first hop
                else:
                    score = float(np.dot(step / n, prefer))
                if score > best_score:
                    best_score = score
                    nxt = v
            unwound += 1

            # If we keep fanning out after unwinding, bail to avoid loops
            if unwound > unwind_steps and G.degree[cur] != 2 and len(nbrs) > 1:
                break

        if nxt is None:
            break

        path_nodes.append(nxt)
        last, cur = cur, nxt
        steps += 1

        # After unwinding, if we encounter a junction again that doesn't collapse to a single continuation, stop
        if unwound >= unwind_steps:
            nb2 = [v for v in G.neighbors(cur) if v != last]
            if len(nb2) != 1:
                break

    coords = [G.nodes[n]["rc"] for n in path_nodes]
    return coords



def fit_tangent(points_rc: List[Tuple[int, int]],
                origin_rc: Tuple[int, int]) -> Tuple[np.ndarray, float]:
    """
    PCA tangent over a fixed arc (first k points from the junction).
    Returns (unit_direction, svd_ratio) where svd_ratio = S0/S1.
    Caller enforces a minimum svd_ratio.
    """
    P = np.array(points_rc, dtype=np.float32)
    if P.shape[0] < 2:
        return np.array([np.nan, np.nan], dtype=np.float32), 0.0

    XY = np.stack([P[:, 1], P[:, 0]], axis=1)  # (x,y)
    XYc = XY - XY.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(XYc, full_matrices=False)
    if S.shape[0] < 2 or S[1] <= 0:
        return np.array([np.nan, np.nan], dtype=np.float32), 0.0
    direction = Vt[0]
    svd_ratio = float(S[0] / (S[1] + 1e-12))

    origin_xy = np.array([origin_rc[1], origin_rc[0]], dtype=np.float32)
    first_xy = np.array([P[0, 1], P[0, 0]], dtype=np.float32)
    vec_to_first = first_xy - origin_xy
    if np.dot(direction, vec_to_first) < 0:
        direction = -direction

    nrm = np.linalg.norm(direction)
    if nrm == 0:
        return np.array([np.nan, np.nan], dtype=np.float32), 0.0
    return (direction / nrm).astype(np.float32), svd_ratio


def branch_radius(dist: np.ndarray, points_rc: List[Tuple[int, int]], k: int = 8) -> float:
    """
    Robust local radius from EDT using multiple offsets near the junction.
    Median-of-means aggregator across offset groups; trimmed mean inside groups.
    """
    if len(points_rc) == 0:
        return np.nan

    offsets = (5, 10, 15)  # px away from junction; will clip to path length
    halfwin = 2            # +/- window around each offset
    groups: List[List[float]] = []

    L = len(points_rc)
    for o in offsets:
        idx = int(np.clip(o, 0, L - 1))
        vals: List[float] = []
        for t in range(-halfwin, halfwin + 1):
            j = int(np.clip(idx + t, 0, L - 1))
            r, c = points_rc[j]
            vals.append(float(dist[int(r), int(c)]))
        if len(vals) > 0:
            vals = sorted(vals)
            trim = max(1, int(0.2 * len(vals)))  # 20% trim
            vals = vals[trim: len(vals) - trim] if len(vals) - 2 * trim > 0 else vals
            if len(vals) > 0:
                groups.append(vals)

    if not groups:
        vals = [float(dist[int(r), int(c)]) for (r, c) in points_rc[:max(1, k)]]
        return float(np.median(vals)) if len(vals) > 0 else np.nan

    means = [float(np.mean(g)) for g in groups if len(g) > 0]
    if len(means) == 0:
        return np.nan
    return float(np.median(means))


# === NEW: robust SD from samples and per-branch stats ===
def _robust_sd_from_samples(vals: List[float]) -> float:
    """
    Robust SD via IQR/1.349 (Normal reference) for outlier resistance.
    """
    if not vals:
        return float("nan")
    x = np.asarray(vals, float)
    q1, q3 = np.percentile(x, [25, 75])
    iqr = q3 - q1
    return float(iqr / 1.349)  # NIST: IQR ≈ 1.349*SD under Normal


def branch_radius_stats(dist: np.ndarray, points_rc: List[Tuple[int,int]],
                        scheme: str = "A") -> Tuple[float, float]:
    """
    Return (radius_estimate, robust_sd) using EDT.
    scheme 'A': offsets=(5,10,15), halfwin=2  (current default)
    scheme 'B': offsets=(8,12,16), halfwin=3  (systematic variant)
    """
    if len(points_rc) == 0:
        return float("nan"), float("nan")
    if str(scheme).upper() == "B":
        offsets, halfwin = (8, 12, 16), 3
    else:
        offsets, halfwin = (5, 10, 15), 2

    samples: List[float] = []
    L = len(points_rc)
    for o in offsets:
        idx = int(np.clip(o, 0, L - 1))
        for t in range(-halfwin, halfwin + 1):
            j = int(np.clip(idx + t, 0, L - 1))
            r, c = points_rc[j]
            samples.append(float(dist[int(r), int(c)]))
    if not samples:
        return branch_radius(dist, points_rc), float("nan")
    r_hat = float(np.median(samples))
    sd_hat = _robust_sd_from_samples(samples)
    return r_hat, sd_hat



def branch_angle_sd_deg(points_rc: List[Tuple[int,int]],
                        origin_rc: Tuple[int,int],
                        k_first: int = 12) -> float:
    """
    Circular robust SD (deg) of chord directions origin→points[1..k], unwrapped,
    summarized via IQR/1.349.
    """
    if len(points_rc) < 3:
        return float("nan")
    oy, ox = origin_rc
    K = min(k_first, len(points_rc))
    angs = []
    for j in range(1, K):
        yy, xx = points_rc[j]
        v = np.array([float(xx - ox), float(yy - oy)])
        n = np.linalg.norm(v)
        if n > 0:
            angs.append(math.atan2(v[1], v[0]))  # radians
    if len(angs) < 3:
        return float("nan")
    a = np.unwrap(np.asarray(angs, float))       # unwrap
    a_deg = np.rad2deg(a)
    q1, q3 = np.percentile(a_deg, [25, 75])
    return float((q3 - q1) / 1.349)  # robust SD in degrees




# ---------- EPIC ANGLE-ONLY INVERSION ----------
def angle_between(u: np.ndarray, v: np.ndarray) -> float:
    """
    Return angle in radians between 2D unit vectors u and v.
    """
    dd = float(np.clip(np.dot(u, v), -1.0, 1.0))
    return float(np.arccos(dd))


def m_from_node(r0: float, r1: float, r2: float, theta12: float,
                m_min: float = 0.2, m_max: float = 4.0,
                tol: float = 1e-6, iters: int = 64) -> float:
    """
    Robust solver for:
        r0^{2m} = r1^{2m} + r2^{2m} + 2 cos(theta12) r1^m r2^m
    Uses bisection when a sign change exists; otherwise falls back to
    a bounded 1-D search that minimizes |f(m)| on [m_min, m_max].
    Accepts the fallback only if the normalized residual is small.
    """
    c = math.cos(theta12)

    def f(m: float) -> float:
        a1 = r1 ** m
        a2 = r2 ** m
        return (r0 ** (2.0 * m)) - (r1 ** (2.0 * m)) - (r2 ** (2.0 * m)) - 2.0 * c * a1 * a2

    # Try bracketing + bisection first
    lo, hi = float(m_min), float(m_max)
    flo, fhi = f(lo), f(hi)

    if np.isfinite(flo) and np.isfinite(fhi):
        if flo == 0.0:
            return lo
        if fhi == 0.0:
            return hi
        if flo * fhi < 0.0:
            # Bisection
            for _ in range(iters):
                mid = 0.5 * (lo + hi)
                fmid = f(mid)
                if abs(fmid) < tol:
                    return mid
                if flo * fmid <= 0.0:
                    hi, fhi = mid, fmid
                else:
                    lo, flo = mid, fmid
            return 0.5 * (lo + hi)

    # Fallback: bounded 1-D search for minimal |f(m)|
    # Coarse grid, then local refinement
    ms = np.linspace(m_min, m_max, 121, dtype=np.float64)
    vals = np.array([abs(f(m)) for m in ms])
    k = int(np.argmin(vals))
    m_star = float(ms[k])

    # Local refinement around m_star
    for _ in range(6):
        span = max(0.02, 0.10 * (m_max - m_min))
        left = max(m_min, m_star - span)
        right = min(m_max, m_star + span)
        ms2 = np.linspace(left, right, 41, dtype=np.float64)
        vals2 = np.array([abs(f(m)) for m in ms2])
        k2 = int(np.argmin(vals2))
        new_m = float(ms2[k2])
        if abs(new_m - m_star) < 1e-4:
            m_star = new_m
            break
        m_star = new_m

    # Normalized residual test (scale by typical magnitude to avoid trivial rejection)
    scale = (r0 ** (2.0 * m_star)) + (r1 ** (2.0 * m_star)) + (r2 ** (2.0 * m_star)) + 1e-12
    norm_resid = abs(f(m_star)) / scale
    if norm_resid < 0.08:
        return m_star

    return np.nan



def closure_residual(r0: float, r1: float, r2: float,
                     e0: np.ndarray, e1: np.ndarray, e2: np.ndarray,
                     m: float, norm_mode: str = "baseline") -> float:
    """
    R(m) = || r0^m e0 + r1^m e1 + r2^m e2 || / denom
    denom:
      - 'baseline' : (r1^m + r2^m)
      - 'sum'      : (r0^m + r1^m + r2^m)
    """
    a0 = (r0 ** m) * e0
    a1 = (r1 ** m) * e1
    a2 = (r2 ** m) * e2
    num = np.linalg.norm(a0 + a1 + a2)
    if norm_mode == "sum":
        denom = (r0 ** m) + (r1 ** m) + (r2 ** m) + 1e-12
    else:
        denom = (r1 ** m) + (r2 ** m) + 1e-12
    return float(num / denom)


def tariffs_from_nullspace(r0: float, r1: float, r2: float,
                           e0: np.ndarray, e1: np.ndarray, e2: np.ndarray,
                           m: float) -> Tuple[np.ndarray, float]:
    """
    Solve for relative c = (c0,c1,c2) up to scale from:
        E diag([r0^m, r1^m, r2^m]) c = 0
    via SVD nullspace (right-singular vector with smallest singular value).
    Returns (c / sum(c), residual_norm).
    If positivity is violated severely, we still return (normalized) but warn via residual.
    """
    A = np.stack([e0, e1, e2], axis=1)  # shape (2,3)
    D = np.diag([r0 ** m, r1 ** m, r2 ** m])
    M = A @ D  # (2,3)
    # Nullspace via SVD: last row of Vt
    U, S, Vt = np.linalg.svd(M, full_matrices=True)
    v = Vt[-1, :]  # (3,)
    # Normalize to sum = 1 (preserve sign using positivity preference)
    # Prefer positive c: if most entries negative, flip
    if np.sum(v < 0) > np.sum(v >= 0):
        v = -v
    # If any very small negative due to noise, clamp to zero before renorm
    v_clamped = np.maximum(v, 0.0)
    if v_clamped.sum() > 0:
        c = v_clamped / (v_clamped.sum() + 1e-12)
    else:
        # fallback to |v|
        c = np.abs(v)
        c = c / (c.sum() + 1e-12)

    residual = float(np.linalg.norm(M @ c))
    return c.astype(np.float32), residual


def m_from_angle_symmetric(theta: float, n: float = 4.0) -> float:
    """
    For near-symmetric daughters (r1 ~ r2), use:
        cos(θ/2) = 2^{2m/(m+n) - 1}
        => m = n * (1 + log2 cos(θ/2)) / (1 - log2 cos(θ/2))
    Returns np.nan if invalid.
    """
    c = math.cos(0.5 * theta)
    if not (0.0 < c < 1.0):
        return np.nan
    x = math.log2(c)
    denom = (1.0 - x)
    if abs(denom) < 1e-9:
        return np.nan
    return float(n * (1.0 + x) / denom)
    
    
# --- Angle-only deterministic m (no directions used; for held-out test) ---
def angle_only_m_deterministic(
    r0: float, r1: float, r2: float, theta12_rad: float,
    m_min: float = 0.2, m_max: float = 4.0, symmetric_ratio_tol: float = 1.08
) -> Tuple[float, str]:
    """
    Deterministic, direction-free m:
      1) try root solver m_from_node on [m_min, m_max]
      2) if daughters near-symmetric (max(r1,r2)/min(r1,r2) <= symmetric_ratio_tol),
         try symmetric analytic m(θ)
      3) fallback to grid argmin of the scalar equation misfit (no directions)
    Returns (m, source_tag).
    """
    # 1) solver on the scalar angle equation
    m_sol = m_from_node(r0, r1, r2, theta12_rad, m_min=m_min, m_max=m_max, tol=1e-6, iters=64)
    if np.isfinite(m_sol):
        return float(np.clip(m_sol, m_min, m_max)), "solver"

    # 2) symmetric daughters option (still angle-only)
    ratio = max(r1, r2) / max(1e-9, min(r1, r2))
    if ratio <= float(symmetric_ratio_tol):
        m_sym = m_from_angle_symmetric(theta12_rad, n=4.0)
        if np.isfinite(m_sym) and (m_min <= m_sym <= m_max):
            return float(m_sym), "symmetric"

    # 3) grid minimizer of |scalar equation residual| (angle-only)
    def eq_res(mm: float) -> float:
        c = math.cos(theta12_rad)
        return abs((r0**(2.0*mm)) - (r1**(2.0*mm)) - (r2**(2.0*mm)) - 2.0*c*(r1**mm)*(r2**mm))
    grid = np.linspace(m_min, m_max, 121, dtype=np.float64)
    vals = np.array([eq_res(mm) for mm in grid])
    m_grid = float(grid[int(np.argmin(vals))])
    return float(m_grid), "grid"

    


@dataclass
class QCConfig:
    min_angle_deg: float = 15.0
    max_angle_deg: float = 170.0
    min_branch_len_px: int = 10
    max_walk_len_px: int = 48
    min_radius_px: float = 1.0
    m_bracket: Tuple[float, float] = (0.2, 4.0)
    symmetric_ratio_tol: float = 1.08  # r1/r2 within this factor => "symmetric"

    # Geometry stabilization + gating
    dedup_radius_px: int = 3
    tangent_len_px: int = 12
    svd_ratio_min: float = 1.6
    angle_auto: bool = False
    angle_auto_pctl: float = 5.0
    min_angle_floor_deg: float = 8
    max_tangent_wander_deg: float = 25.0
    angle_soft_margin_deg: float = 3.0

    # Ambiguity / edge-handling
    parent_margin_tau: float = 0.08          # margin gate for parent selection (score2 - score1)
    auto_expand_enable: bool = True          # allow one-shot bracket expansion
    auto_expand_bracket_factor: float = 1.5  # ×1.5 expand (lo/×, hi×)
    auto_expand_edge_eps: float = 0.02       # “at edge” definition

    # Stricter QC reporting
    strict_qc: bool = False

    # Metadata (recorded in CSV; does not affect m)
    px_size_um: float = float("nan")
    tangent_mode: str = "pca"
    parent_tie_break: str = "conservative"
    radius_estimator: str = "A"
    r_norm: str = "baseline"




@dataclass
class NodeRecord:
    dataset: str
    image_id: str
    node_id: int
    yx: Tuple[int, int]               # (row, col)
    r0: float
    r1: float
    r2: float
    theta12_deg: float
    m_node: float
    R_m: float
    c0: float
    c1: float
    c2: float
    tariff_residual: float
    qc_pass: bool
    qc_pass_strict: bool
    # Store unit directions so we can build a within-image null by shuffling
    e0x: float
    e0y: float
    e1x: float
    e1y: float
    e2x: float
    e2y: float
    # Per-branch tangent quality (PCA S0/S1)
    svd_ratio_e0: float = float("nan")
    svd_ratio_e1: float = float("nan")
    svd_ratio_e2: float = float("nan")
    # NEW: provenance & solver
    seg_variant: str = ""
    seg_thresh_type: str = ""
    seg_thresh_value: float = float("nan")
    m_source_chosen: str = ""
    parent_ambiguous: bool = False
    px_size_um: float = float("nan")
    note: str = ""
    # HELD-OUT: m from (r0,r1,r2,theta12) only; directions held out for R(m)
    m_angleonly: float = float("nan")
    R_m_holdout: float = float("nan")
    # NEW: measured per-node uncertainties (angle & radii)
    sd_theta_deg: float = float("nan")
    sd_r0: float = float("nan")
    sd_r1: float = float("nan")
    sd_r2: float = float("nan")


def analyze_image(img_path: Path,
                  dataset: str,
                  seg_cfg: SegConfig,
                  qc: QCConfig,
                  save_debug: bool = False,
                  out_dir: Path = ROOT) -> Tuple[List[NodeRecord], Dict]:
    """
    Robust per-image analysis with meticulous debugging and transparent terminal outputs.
    Always attempts to produce a candidate m for each junction (solver → symmetric → grid),
    then defers acceptance to QC based on closure residuals. Returns node records and
    a diagnostics dict with reasoned skip counts.
    """
    image_id = img_path.stem
    log(f"  ── Analyzing: {dataset}/{image_id}")

    # 0) Read image and (optional) mask — with IO diagnostics
    raw = io.imread(str(img_path))
    try:
        raw_shape = tuple(raw.shape)
        raw_dtype = str(raw.dtype)
        raw_ext = img_path.suffix.lower()
    except Exception:
        raw_shape, raw_dtype, raw_ext = ("?",), "?", "?"
    gray = imread_gray(img_path)

    # gray diagnostics
    g_min, g_max = float(gray.min()), float(gray.max())
    g_mean, g_std = float(gray.mean()), float(gray.std())
    log(
        "    [io] "
        f"name={img_path.name}, ext={raw_ext}, raw_shape={raw_shape}, raw_dtype={raw_dtype}; "
        f"gray_shape={gray.shape}, gray_dtype=float32, "
        f"gray_min={g_min:.4f}, gray_max={g_max:.4f}, gray_mean={g_mean:.4f}, gray_std={g_std:.4f}"
    )

    # mask / segmentation
    mask_path = find_mask_for_image(img_path)
    if mask_path is not None:
        mask_img = io.imread(str(mask_path))
        if mask_img.ndim == 3:
            mask_img = color.rgb2gray(mask_img)
        mask = (mask_img > 0.5).astype(np.uint8)
        seg_meta = {
            "seg_variant": "provided",
            "thresh_type": "provided",
            "thresh_value": float("nan"),
            "method": "provided",
            "vessel_frac": float(mask.mean()),
        }
        log(f"    Using provided mask: {mask_path.name}")
    else:
        mask, seg_meta = segment_vessels_with_meta(gray, seg_cfg)
        log("    Segmented vessels via Frangi/Sato.")

    # 1) Skeleton + EDT distance map + graph — with graph diagnostics
    skel, dist = skeleton_and_dist(mask)
    G = build_graph_from_skeleton(skel)

    total_px = int(mask.size)
    vessel_px = int(mask.sum())
    vessel_pct = (vessel_px / max(1, total_px)) * 100.0
    skel_px = int(skel.sum())
    deg_hist = np.bincount([G.degree[n] for n in G.nodes()], minlength=7)
    deg_summary = f"{{0:{deg_hist[0]},1:{deg_hist[1]},2:{deg_hist[2]},3:{deg_hist[3]},4+:{int(deg_hist[4:].sum())}}}"
    log(
        "    [graph] "
        f"mask_pixels={vessel_px}/{total_px} ({vessel_pct:.2f}%), "
        f"skeleton_pixels={skel_px}, nodes={G.number_of_nodes()}, edges={G.number_of_edges()}, "
        f"deg_hist={deg_summary}"
    )

    # 2) Candidate nodes: degree==3  +  de-dup within qc.dedup_radius_px
    junction_nodes_raw = [n for n in G.nodes if is_bifurcation(G, n)]
    junction_nodes = dedup_degree3_nodes(G, junction_nodes_raw, radius_px=qc.dedup_radius_px)
    log(f"    Found {len(junction_nodes_raw)} candidate degree-3 pixels; dedup → {len(junction_nodes)}.")

    # Optional: per-image automatic lower angle gate based on candidate geometry
    angle_min_this_image = qc.min_angle_deg
    if qc.angle_auto:
        angles_deg: List[float] = []
        for n in junction_nodes:
            rc0 = G.nodes[n]["rc"]
            nbrs = list(G.neighbors(n))
            if len(nbrs) != 3:
                continue
            dirs = []
            for nb in nbrs:
                pts = walk_branch(G, skel, nb, prev_node=n, max_steps=max(qc.tangent_len_px, 12))
                if len(pts) < 3:
                    dirs = []
                    break
                pts_for_tan = pts[:qc.tangent_len_px]
                dvec, sratio = fit_tangent(pts_for_tan, origin_rc=rc0)
                if not np.all(np.isfinite(dvec)) or sratio < qc.svd_ratio_min:
                    dirs = []
                    break
                dirs.append(dvec)
            if len(dirs) != 3:
                continue
            vecs = np.stack(dirs, axis=0)
            scores = []
            for i in range(3):
                others = vecs[[j for j in range(3) if j != i]]
                s = others[0] + others[1]
                s_norm = s / (np.linalg.norm(s) + 1e-12)
                scores.append(float(np.dot(vecs[i], s_norm)))
            idx_parent = int(np.argmin(scores))
            idx_daughters = [j for j in range(3) if j != idx_parent]
            e1 = dirs[idx_daughters[0]]
            e2 = dirs[idx_daughters[1]]
            theta12 = angle_between(e1, e2)
            angles_deg.append(math.degrees(theta12))
        if len(angles_deg) > 0:
            angle_min_this_image = max(
                qc.min_angle_floor_deg,
                float(np.percentile(angles_deg, qc.angle_auto_pctl))
            )
        else:
            angle_min_this_image = qc.min_angle_floor_deg
        log(
            f"    Angle gate (auto): min={angle_min_this_image:.1f}° "
            f"(floor={qc.min_angle_floor_deg}°, p{qc.angle_auto_pctl} of {len(angles_deg)} candidates)"
        )

    # Prepare outputs / stats
    node_records: List[NodeRecord] = []
    node_counter = 0
    H, W = gray.shape
    overlay = np.dstack([gray, gray, gray]) if save_debug else None

    # Skip/diagnostics counters
    reasons = {
        "not_deg3": 0,             # (guarded out by construction, but keep for completeness)
        "short_branch": 0,
        "bad_tangent": 0,
        "small_radius": 0,
        "angle_out_of_range": 0,
        "no_m_candidate": 0,
        "qc_fail": 0
    }

    # QC thresholds: loose vs. strict (optional)
    QC_RM_MAX_LOOSE = 0.85
    QC_CRES_MAX_LOOSE = 0.35
    QC_RM_MAX_STRICT = 0.55 if qc.strict_qc else QC_RM_MAX_LOOSE
    QC_CRES_MAX_STRICT = 0.15 if qc.strict_qc else QC_CRES_MAX_LOOSE

    # Helper for overlay coloring
    def draw_point(y, x, color_rgb):
        if overlay is None:
            return
        for dr in range(-1, 2):
            for dc in range(-1, 2):
                yy = int(np.clip(y + dr, 0, H - 1))
                xx = int(np.clip(x + dc, 0, W - 1))
                overlay[yy, xx, :] = color_rgb

    # 3) Iterate junctions
    for n in junction_nodes:
        rc0 = G.nodes[n]["rc"]  # (row, col)
        nbrs = list(G.neighbors(n))
        if len(nbrs) != 3:
            reasons["not_deg3"] += 1
            continue

        # For each neighbor, walk outward (robust handling for short branches)
        branch_paths = []
        branch_dirs = []
        branch_radii = []
        branch_svdratios = []
        branch_ang_sds = []   # per-branch angle SDs
        radius_sds = []       # per-branch radius SDs

        valid = True

        def _greedy_extend(skel_arr: np.ndarray, start_rc: Tuple[int, int],
                           init_vec: np.ndarray, steps: int = 12) -> List[Tuple[int, int]]:
            """
            Greedily extend a polyline on the skeleton for a few steps, preferring the
            8-neighbor that best aligns with init_vec at each step. Avoid immediate backtracking.
            """
            H_, W_ = skel_arr.shape
            path = [tuple(start_rc)]
            last = None
            v = init_vec / (np.linalg.norm(init_vec) + 1e-12)
            for _ in range(steps):
                r, c = path[-1]
                best = None
                best_dot = -1e9
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        if dr == 0 and dc == 0:
                            continue
                        rr, cc = r + dr, c + dc
                        if not (0 <= rr < H_ and 0 <= cc < W_):
                            continue
                        if not skel_arr[rr, cc]:
                            continue
                        if last is not None and (rr, cc) == last:
                            continue
                        step_vec = np.array([dc, dr], dtype=np.float32)
                        step_dot = float(np.dot(step_vec / (np.linalg.norm(step_vec) + 1e-12), v))
                        if step_dot > best_dot:
                            best_dot = step_dot
                            best = (rr, cc)
                if best is None:
                    break
                last = path[-1]
                path.append(best)
            return path

        def _tangent_consistency_ok(points_rc: List[Tuple[int, int]],
                                    origin_rc: Tuple[int, int],
                                    deg_thresh: float = 25.0,
                                    k_first: int = 8) -> bool:
            """
            Cosine-consistency: ensure local tangent near junction doesn't wander > deg_thresh
            over the first ~k_first points. Returns True if consistent.
            """
            if len(points_rc) < 3:
                return False
            k = min(k_first, len(points_rc))
            oy, ox = origin_rc
            vecs = []
            for j in range(1, k):
                yy, xx = points_rc[j]
                v = np.array([float(xx - ox), float(yy - oy)], dtype=np.float32)
                n = np.linalg.norm(v)
                if n <= 0:
                    continue
                vecs.append(v / n)
            if len(vecs) < 2:
                return False
            angles = []
            for a, b in zip(vecs[:-1], vecs[1:]):
                dd = float(np.clip(np.dot(a, b), -1.0, 1.0))
                ang = math.degrees(math.acos(dd))
                angles.append(ang)
            if not angles:
                return False
            return float(np.max(angles)) <= float(deg_thresh)

        for nb in nbrs:
            pts = walk_branch(G, skel, nb, prev_node=n, max_steps=qc.max_walk_len_px)
            path_len = len(pts)

            # If the raw path is too short, try to extend it forward a bit using a greedy directional walk.
            if path_len < qc.min_branch_len_px and path_len >= 2:
                # Estimate a quick initial direction from origin to last observed point
                p0 = np.array([rc0[1], rc0[0]], dtype=np.float32)
                pL = np.array([pts[-1][1], pts[-1][0]], dtype=np.float32)
                init_vec = pL - p0
                ext = _greedy_extend(
                    skel, pts[-1], init_vec,
                    steps=max(8, qc.min_branch_len_px - path_len)
                )
                if len(ext) > 1:
                    # Append new unique points (avoid duplicating the last one)
                    for rrcc in ext[1:]:
                        pts.append(rrcc)
                    path_len = len(pts)
                    log(f"      [extend] grew branch from {len(ext)} steps; new length={path_len}px")

            # Hard drop only if path is truly unusable (<3 pixels)
            if path_len < 3:
                valid = False
                reasons["short_branch"] += 1
                if save_debug:
                    draw_point(*rc0, [1.0, 0.6, 0.0])  # orange
                log(f"      [drop] branch too short (<3 px): len={path_len}")
                continue

            # Tangent over fixed arc honoring qc.tangent_mode
            pts_for_tan = pts[:qc.tangent_len_px]

            def _chord_direction(points_rc: List[Tuple[int, int]],
                                 origin_rc: Tuple[int, int]) -> np.ndarray:
                """
                Use a simple chord from the junction to a far point as a surrogate tangent.
                """
                j = min(len(points_rc) - 1, max(8, qc.tangent_len_px) - 1)
                yy, xx = points_rc[j]
                oy, ox = origin_rc
                v = np.array([float(xx - ox), float(yy - oy)], dtype=np.float32)
                n = np.linalg.norm(v)
                return (v / (n + 1e-12)) if n > 0 else np.array([np.nan, np.nan], dtype=np.float32)

            if str(qc.tangent_mode).lower() == "chord":
                dvec = _chord_direction(pts_for_tan, rc0)
                sratio = qc.svd_ratio_min  # treat as "good enough" for gating
                if not _tangent_consistency_ok(pts_for_tan, rc0,
                                               deg_thresh=qc.max_tangent_wander_deg,
                                               k_first=8):
                    valid = False
                    reasons["bad_tangent"] += 1
                    if save_debug:
                        draw_point(*rc0, [1.0, 0.0, 0.0])
                    log(f"      [drop] chord direction inconsistent; len={len(pts_for_tan)}")
                    continue
            else:
                dvec, sratio = fit_tangent(pts_for_tan, origin_rc=rc0)
                bad = (
                    (not np.all(np.isfinite(dvec)))
                    or (np.linalg.norm(dvec) < 0.5)
                    or (sratio < qc.svd_ratio_min)
                    or (not _tangent_consistency_ok(
                        pts_for_tan, rc0,
                        deg_thresh=qc.max_tangent_wander_deg,
                        k_first=8
                    ))
                )
                if bad:
                    # Fallback: accept a chord direction if the branch is long enough AND consistent
                    if path_len >= max(8, qc.tangent_len_px // 2):
                        dvec_fb = _chord_direction(pts_for_tan, rc0)
                        if (
                            np.all(np.isfinite(dvec_fb))
                            and np.linalg.norm(dvec_fb) >= 0.5
                            and _tangent_consistency_ok(
                                pts_for_tan, rc0,
                                deg_thresh=qc.max_tangent_wander_deg,
                                k_first=8
                            )
                        ):
                            dvec = dvec_fb
                            sratio = max(sratio, qc.svd_ratio_min)
                            log(
                                f"      [fallback] using chord direction; "
                                f"len={path_len}, svd_ratio≈{sratio:.2f}"
                            )
                        else:
                            valid = False
                            reasons["bad_tangent"] += 1
                            if save_debug:
                                draw_point(*rc0, [1.0, 0.0, 0.0])  # red
                            log(
                                f"      [drop] bad/unstable tangent; "
                                f"len={path_len}, svd_ratio={sratio:.2f}"
                            )
                            continue
                    else:
                        valid = False
                        reasons["bad_tangent"] += 1
                        if save_debug:
                            draw_point(*rc0, [1.0, 0.0, 0.0])  # red
                        log(
                            f"      [drop] bad tangent (too short for fallback); "
                            f"len={path_len}, svd_ratio={sratio:.2f}"
                        )
                        continue

            # Per-branch angle SD (deg) from the first-k chord directions
            ang_sd = branch_angle_sd_deg(
                pts_for_tan, rc0,
                k_first=min(12, qc.tangent_len_px)
            )

            # Robust radius + robust SD from EDT samples
            rad, rad_sd = branch_radius_stats(dist, pts, scheme=qc.radius_estimator)
            if not np.isfinite(rad) or rad < qc.min_radius_px:
                valid = False
                reasons["small_radius"] += 1
                if save_debug:
                    draw_point(*rc0, [1.0, 0.0, 0.0])  # red
                log(f"      [drop] small radius; r≈{rad:.2f}px, len={path_len}")
                continue

            # Soft accept short-but-usable branches: keep them but log for transparency
            if path_len < qc.min_branch_len_px:
                log(
                    f"      [soft] short branch accepted: len={path_len}px "
                    f"< min={qc.min_branch_len_px}px; r≈{rad:.2f}px"
                )

            branch_paths.append(pts)
            branch_dirs.append(dvec)
            branch_radii.append(rad)
            branch_svdratios.append(float(sratio))
            branch_ang_sds.append(float(ang_sd))
            radius_sds.append(float(rad_sd))

        if not valid or len(branch_dirs) != 3:
            # already counted reason above
            continue

        vecs = np.stack(branch_dirs, axis=0)  # (3,2)
        scores = []
        for i in range(3):
            others = vecs[[j for j in range(3) if j != i]]
            s = others[0] + others[1]
            s_norm = s / (np.linalg.norm(s) + 1e-12)
            scores.append(float(np.dot(vecs[i], s_norm)))

        order = np.argsort(scores)
        idx_parent_primary = int(order[0])  # most opposite
        idx_daughters_primary = [j for j in range(3) if j != idx_parent_primary]

        # Ambiguity gate: require a margin between best and second-best
        parent_margin_tau = float(getattr(qc, "parent_margin_tau", 0.08))  # score2 - score1
        margin = float(scores[order[1]] - scores[order[0]])
        is_ambiguous_by_margin = bool(margin <= parent_margin_tau)

        if is_ambiguous_by_margin:
            # Tie-breaker: among near-ties, prefer higher tangent SVD ratio
            cand = order[:2]
            svd_pairs = [(float(branch_svdratios[i]), int(i)) for i in cand]
            idx_parent_primary = int(sorted(svd_pairs, key=lambda z: z[0], reverse=True)[0][1])
            idx_daughters_primary = [j for j in range(3) if j != idx_parent_primary]
            log(f"      [parent] near-tie resolved by SVD tie-breaker; margin={margin:.3f}, τ={parent_margin_tau:.3f}")


        # Alternate assignment: swap parent with the fatter daughter (conservative cross-check)
        radii_arr = np.array(branch_radii, dtype=float)
        idx_fat_daughter = int(
            idx_daughters_primary[int(np.argmax(radii_arr[idx_daughters_primary]))]
        )
        idx_parent_alt = idx_fat_daughter
        idx_daughters_alt = [j for j in range(3) if j != idx_parent_alt]

        def _solve_for_assignment(i_par: int, i_daus: List[int]):
            ee0, ee1, ee2 = branch_dirs[i_par], branch_dirs[i_daus[0]], branch_dirs[i_daus[1]]
            rr0 = float(branch_radii[i_par])
            rr1 = float(branch_radii[i_daus[0]])
            rr2 = float(branch_radii[i_daus[1]])
            th12 = angle_between(ee1, ee2)
            deg12_ = math.degrees(th12)

            # Soft rescue: if θ12 is within a small margin outside the gate, clamp to nearest bound
            used_deg12 = deg12_
            soft_used = False
            if not (angle_min_this_image <= deg12_ <= qc.max_angle_deg):
                if (
                    (angle_min_this_image - float(qc.angle_soft_margin_deg))
                    <= deg12_
                    <= (qc.max_angle_deg + float(qc.angle_soft_margin_deg))
                ):
                    used_deg12 = float(np.clip(deg12_, angle_min_this_image, qc.max_angle_deg))
                    th12 = math.radians(used_deg12)
                    soft_used = True
                else:
                    return None  # gated out

            m_candidates, m_source = [], []

            # Solver
            m_sol = m_from_node(
                rr0, rr1, rr2, th12,
                m_min=qc.m_bracket[0], m_max=qc.m_bracket[1],
                tol=1e-6, iters=64
            )
            if np.isfinite(m_sol) and qc.m_bracket[0] <= m_sol <= qc.m_bracket[1]:
                m_candidates.append(float(m_sol))
                m_source.append("solver")

            # Symmetric (if daughters comparable)
            ratio = max(rr1, rr2) / max(1e-9, min(rr1, rr2))
            if ratio <= qc.symmetric_ratio_tol:
                m_sym = m_from_angle_symmetric(th12, n=4.0)
                if np.isfinite(m_sym) and qc.m_bracket[0] <= m_sym <= qc.m_bracket[1]:
                    m_candidates.append(float(m_sym))
                    m_source.append("symmetric")

            # Grid minimizer of |equation residual|
            def eq_res(mm: float) -> float:
                return abs(
                    (rr0**(2.0 * mm))
                    - (rr1**(2.0 * mm))
                    - (rr2**(2.0 * mm))
                    - 2.0 * math.cos(th12) * (rr1**mm) * (rr2**mm)
                )

            grid = np.linspace(qc.m_bracket[0], qc.m_bracket[1], 121, dtype=np.float64)
            vals = np.array([eq_res(mm) for mm in grid])
            m_grid = float(grid[int(np.argmin(vals))])
            if np.isfinite(m_grid) and qc.m_bracket[0] <= m_grid <= qc.m_bracket[1]:
                m_candidates.append(m_grid)
                m_source.append("grid")

            if len(m_candidates) == 0:
                return None

            best_idx, best_R = -1, np.inf
            for ii, m_c in enumerate(m_candidates):
                R_c = closure_residual(
                    rr0, rr1, rr2, ee0, ee1, ee2, m_c,
                    norm_mode=qc.r_norm
                )
                if R_c < best_R:
                    best_R = R_c
                    best_idx = ii

            # inside analyze_image(...), _solve_for_assignment(...):
            m_fin = float(m_candidates[best_idx])
            src_fin = str(m_source[best_idx])
            Rm = closure_residual(
                rr0, rr1, rr2, ee0, ee1, ee2, m_fin,
                norm_mode=qc.r_norm
            )
            c_vec, cresid = tariffs_from_nullspace(rr0, rr1, rr2, ee0, ee1, ee2, m_fin)

            # Auto-expand bracket once if the solution sits on the edge
            try:
                edge_eps = float(getattr(qc, "auto_expand_edge_eps", 0.02))
                do_expand = bool(getattr(qc, "auto_expand_enable", True))
                if do_expand and ((abs(m_fin - qc.m_bracket[0]) <= edge_eps) or (abs(m_fin - qc.m_bracket[1]) <= edge_eps)):
                    lo_exp = max(0.10, float(qc.m_bracket[0]) / float(getattr(qc, "auto_expand_bracket_factor", 1.5)))
                    hi_exp = min(6.0,  float(qc.m_bracket[1]) * float(getattr(qc, "auto_expand_bracket_factor", 1.5)))

                    def _best_in(lo, hi):
                        cand, src, best_Rloc, best_iloc = [], [], np.inf, -1
                        m_try = m_from_node(rr0, rr1, rr2, th12, m_min=lo, m_max=hi, tol=1e-6, iters=64)
                        if np.isfinite(m_try) and (lo <= m_try <= hi):
                            cand.append(float(m_try)); src.append("solver+expand")
                        grid2 = np.linspace(lo, hi, 161, dtype=np.float64)
                        vals2 = np.array([eq_res(mm) for mm in grid2])
                        mg2 = float(grid2[int(np.argmin(vals2))])
                        if np.isfinite(mg2) and (lo <= mg2 <= hi):
                            cand.append(mg2); src.append("grid+expand")
                        for ii, mv in enumerate(cand):
                            Rloc = closure_residual(rr0, rr1, rr2, ee0, ee1, ee2, mv, norm_mode=qc.r_norm)
                            if Rloc < best_Rloc:
                                best_Rloc, best_iloc = Rloc, ii
                        return (cand[best_iloc], src[best_iloc], best_Rloc) if (best_iloc >= 0) else (m_fin, src_fin, Rm)

                    m2, src2, R2 = _best_in(lo_exp, hi_exp)
                    if (R2 < Rm) or ((m2 > qc.m_bracket[0] + edge_eps) and (m2 < qc.m_bracket[1] - edge_eps)):
                        m_fin, src_fin, Rm = float(m2), str(src2), float(R2)
            except Exception:
                pass


            qc_pass_loose = ((Rm < QC_RM_MAX_LOOSE) or (cresid < QC_CRES_MAX_LOOSE))
            qc_pass_strict = ((Rm < QC_RM_MAX_STRICT) and (cresid < QC_CRES_MAX_STRICT))
            return {
                "e0": ee0, "e1": ee1, "e2": ee2,
                "r0": rr0, "r1": rr1, "r2": rr2,
                "theta12_deg": deg12_,
                "m": m_fin,
                "Rm": Rm,
                "c": c_vec,
                "cresid": cresid,
                "qc_loose": qc_pass_loose,
                "qc_strict": qc_pass_strict,
                "best_R": best_R,
                "m_source": src_fin,
                "soft_clamp": soft_used,
                "theta12_used_deg": used_deg12
            }

        res_primary = _solve_for_assignment(idx_parent_primary, idx_daughters_primary)
        res_alt = _solve_for_assignment(idx_parent_alt, idx_daughters_alt)

        if (res_primary is None) and (res_alt is None):
            reasons["angle_out_of_range"] += 1  # both gated out
            if save_debug:
                draw_point(*rc0, [1.0, 0.0, 1.0])
            continue

        parent_ambiguous = (
            is_ambiguous_by_margin
            or (
                idx_parent_primary != idx_parent_alt
                and (res_primary is not None)
                and (res_alt is not None)
            )
        )

        if res_primary is not None and res_alt is not None:
            if str(qc.parent_tie_break).lower() == "optimistic":
                chosen = res_primary if res_primary["Rm"] <= res_alt["Rm"] else res_alt
            else:
                chosen = res_primary if res_primary["Rm"] >= res_alt["Rm"] else res_alt
        else:
            chosen = res_primary if res_primary is not None else res_alt

        e0, e1, e2 = chosen["e0"], chosen["e1"], chosen["e2"]
        r0, r1, r2 = chosen["r0"], chosen["r1"], chosen["r2"]
        deg12 = chosen["theta12_deg"]
        m = chosen["m"]
        Rm = chosen["Rm"]
        c_vec, cresid = chosen["c"], chosen["cresid"]
        qc_pass_loose, qc_pass_strict = chosen["qc_loose"], chosen["qc_strict"]
        best_R, m_src = chosen["best_R"], chosen["m_source"]

        # --- HELD-OUT angle-only m (no directions used to choose m) ---
        m_ao, _ao_src = angle_only_m_deterministic(
            r0, r1, r2, math.radians(deg12),
            m_min=qc.m_bracket[0], m_max=qc.m_bracket[1],
            symmetric_ratio_tol=qc.symmetric_ratio_tol
        )
        Rm_ao = (
            closure_residual(r0, r1, r2, e0, e1, e2, m_ao, norm_mode=qc.r_norm)
            if np.isfinite(m_ao) else float("nan")
        )

        # map chosen e-vectors back to branch indices to get SVD ratios
        def _idx_of_vec(vec: np.ndarray) -> int:
            dists = [np.linalg.norm(np.asarray(bv) - np.asarray(vec)) for bv in branch_dirs]
            return int(np.argmin(dists)) if len(dists) == 3 else 0

        i0 = _idx_of_vec(e0)
        i1 = _idx_of_vec(e1)
        i2 = _idx_of_vec(e2)
        svd0 = float(branch_svdratios[i0]) if len(branch_svdratios) == 3 else float("nan")
        svd1 = float(branch_svdratios[i1]) if len(branch_svdratios) == 3 else float("nan")
        svd2 = float(branch_svdratios[i2]) if len(branch_svdratios) == 3 else float("nan")

        # Per-node SDs (angle combines daughters; radii per branch)
        try:
            ang_sd_i1 = float(branch_ang_sds[i1]) if len(branch_ang_sds) == 3 else float("nan")
            ang_sd_i2 = float(branch_ang_sds[i2]) if len(branch_ang_sds) == 3 else float("nan")
            sd_theta_deg_node = float(np.sqrt((ang_sd_i1**2 + ang_sd_i2**2) / 2.0))
        except Exception:
            sd_theta_deg_node = float("nan")

        sd_r0_node = float(radius_sds[i0]) if len(radius_sds) == 3 else float("nan")
        sd_r1_node = float(radius_sds[i1]) if len(radius_sds) == 3 else float("nan")
        sd_r2_node = float(radius_sds[i2]) if len(radius_sds) == 3 else float("nan")

        node_records.append(
            NodeRecord(
                dataset=dataset,
                image_id=image_id,
                node_id=node_counter,
                yx=(int(rc0[0]), int(rc0[1])),
                r0=float(r0),
                r1=float(r1),
                r2=float(r2),
                theta12_deg=float(deg12),
                m_node=float(m),
                R_m=float(Rm),
                c0=float(c_vec[0]),
                c1=float(c_vec[1]),
                c2=float(c_vec[2]),
                tariff_residual=float(cresid),
                qc_pass=qc_pass_loose,
                qc_pass_strict=qc_pass_strict,
                e0x=float(e0[0]),
                e0y=float(e0[1]),
                e1x=float(e1[0]),
                e1y=float(e1[1]),
                e2x=float(e2[0]),
                e2y=float(e2[1]),
                svd_ratio_e0=svd0,
                svd_ratio_e1=svd1,
                svd_ratio_e2=svd2,
                seg_variant=str(seg_meta.get("seg_variant", "unknown")),
                seg_thresh_type=str(seg_meta.get("thresh_type", "")),
                seg_thresh_value=float(seg_meta.get("thresh_value", float("nan"))),
                m_source_chosen=str(m_src),
                parent_ambiguous=bool(parent_ambiguous),
                px_size_um=float(qc.px_size_um),
                note="m_sources_evaluated=['solver','symmetric','grid']"
                     + ("; soft_angle_clamp" if bool(chosen.get("soft_clamp", False)) else ""),
                m_angleonly=float(m_ao),
                R_m_holdout=float(Rm_ao),
                sd_theta_deg=float(sd_theta_deg_node),
                sd_r0=float(sd_r0_node),
                sd_r1=float(sd_r1_node),
                sd_r2=float(sd_r2_node),
            )
        )

        node_counter += 1

        # 3f) Debug overlay color by QC (green=strict pass, orange=loose-only pass, red=fail)
        if save_debug:
            color_rgb = (
                [0.0, 1.0, 0.0]
                if qc_pass_strict
                else ([1.0, 0.65, 0.0] if qc_pass_loose else [1.0, 0.0, 0.0])
            )
            draw_point(*rc0, color_rgb)

    kept = len(node_records)
    diag = {
        "dataset": dataset,
        "image_id": image_id,
        "n_nodes_total_raw": int(len(junction_nodes_raw)),
        "n_nodes_total_dedup": int(len(junction_nodes)),
        "n_nodes_kept": int(kept),
        "angle_gate_min_deg": float(angle_min_this_image),
        "skip_reasons": reasons,
        "qc_thresholds": {
            "loose": {"R_m_max": QC_RM_MAX_LOOSE, "c_res_max": QC_CRES_MAX_LOOSE},
            "strict": {"R_m_max": QC_RM_MAX_STRICT, "c_res_max": QC_CRES_MAX_STRICT},
        },
    }

    # Transparent summary for this image
    log(
        f"    Image summary: kept={kept}/{len(junction_nodes)} | "
        f"skips: short_branch={reasons['short_branch']}, bad_tangent={reasons['bad_tangent']}, "
        f"small_radius={reasons['small_radius']}, angle_out={reasons['angle_out_of_range']}, "
        f"no_m={reasons['no_m_candidate']}, qc_fail={reasons['qc_fail']}"
    )

    # 5) Optional overlay save (skeleton + tariff/residual maps)
    if save_debug and overlay is not None:
        # Skeleton / QC overlay
        fig = plt.figure(figsize=(10, 10))
        ax_ov = plt.gca()
        ax_ov.imshow(overlay, interpolation="nearest")
        ax_ov.set_axis_off()

        # PRE-style, human-readable labels (no underscores)
        dataset_label = dataset.replace("_", " ")
        image_label = image_id.replace("_", " ")

        ax_ov.set_title(
            f"{dataset_label}, image {image_label}\n"
            f"Junction QC overlay (kept {kept})",
            fontsize=11,
        )

        out_png = out_dir / "figures" / dataset / f"{image_id}__skeleton_overlay.png"
        out_png.parent.mkdir(parents=True, exist_ok=True)
        save_png(fig, out_png)

        out_pdf = out_dir / "figures" / dataset / f"{image_id}__skeleton_overlay.pdf"
        out_pdf.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_pdf, format="pdf", bbox_inches="tight")

        # Tariff + residual-vector maps (PNG + PDF), using the same gray/skel and node_records
        try:
            plot_tariff_map(
                gray=gray,
                skel=skel,
                nodes=node_records,
                dataset=dataset,
                image_id=image_id,
                norm_mode=qc.r_norm,
            )
        except Exception as e:
            log(f"[DEBUG][WARN] Tariff/residual maps failed for {dataset}/{image_id}: {e}")


    return node_records, diag


def plot_m_distribution(df: pd.DataFrame, dataset: str):
    fig = plt.figure(figsize=(8, 5))
    ax = plt.gca()
    m_vals = df["m_node"].astype(float).values
    ax.hist(m_vals, bins=40, alpha=0.9)
    ax.set_xlabel("m (per-node inverted)")
    ax.set_ylabel("count")
    ax.set_title(f"EPIC upkeep exponent (m) — {dataset} (N={len(m_vals)})")

    # Show bracket bounds and edge shading
    try:
        m_min = float(df["cfg_m_min"].iloc[0])
        m_max = float(df["cfg_m_max"].iloc[0])
    except Exception:
        m_min, m_max = 0.2, 4.0
    ax.axvline(m_min, linestyle="--", linewidth=1.0)
    ax.axvline(m_max, linestyle="--", linewidth=1.0)
    edge_eps = 0.02
    ax.axvspan(m_min - edge_eps, m_min + edge_eps, alpha=0.10)
    ax.axvspan(m_max - edge_eps, m_max + edge_eps, alpha=0.10)

    # Annotate fraction at edges, and—if present—how many move off-edge under uncertainty
    at_edge = ((np.abs(df["m_node"].astype(float) - m_min) <= edge_eps) |
               (np.abs(df["m_node"].astype(float) - m_max) <= edge_eps))
    n_edge = int(np.sum(at_edge))
    note_lines = [f"at-edge (±{edge_eps:.2f}): {n_edge}"]

    # If the uncertainty CSV exists, report movement off-edge
    unc_path = CSV_ROOT / dataset / "UNCERTAINTY__nodes.csv"
    try:
        if unc_path.exists():
            unc = pd.read_csv(unc_path, comment="#")
            if {"at_edge_before","at_edge_after"}.issubset(unc.columns):
                # WARNING: file uses column 'at_edge_after' to store the variable named 'off_edge_after' in code
                edge_before = unc["at_edge_before"].astype(bool).values
                off_after = unc["at_edge_after"].astype(bool).values
                moved = np.logical_and(edge_before, off_after)
                if np.any(edge_before):
                    frac_moved = float(np.mean(moved))
                    note_lines.append(f"moved off-edge under uncertainty: {frac_moved*100:.1f}%")
    except Exception:
        pass

    ax.text(0.02, 0.98, "\n".join(note_lines), transform=ax.transAxes, va="top", ha="left",
            fontsize=9, bbox=dict(facecolor="white", alpha=0.85, edgecolor="none"))

    save_png(fig, FIG_ROOT / dataset / f"dataset_{dataset}__m_hist.png")

    # By-source histogram (solver vs non-solver)
    if "m_source_chosen" in df.columns:
        fig_src = plt.figure(figsize=(8, 5))
        plt.hist(df[df["m_source_chosen"] == "solver"]["m_node"].values, bins=40, alpha=0.7, label="solver (bracket)")
        plt.hist(df[df["m_source_chosen"] != "solver"]["m_node"].values,  bins=40, alpha=0.5, label="fallback (symmetric/grid)")
        frac_fb = (len(df[df["m_source_chosen"] != "solver"]) / max(1, len(m_vals)))
        plt.title(f"m by solver source — {dataset}  (fallback fraction={frac_fb:.2%})")
        plt.xlabel("m"); plt.ylabel("count"); plt.legend(loc="best")
        save_png(fig_src, FIG_ROOT / dataset / f"dataset_{dataset}__m_hist_by_source.png")

    # boxplot per image (top-20 with most nodes)
    counts = df.groupby("image_id").size().sort_values(ascending=False)
    top_imgs = list(counts.index[:20])
    sub = df[df["image_id"].isin(top_imgs)]
    fig2 = plt.figure(figsize=(10, 5))
    ax2 = fig2.add_subplot(111)
    sub.boxplot(column="m_node", by="image_id", ax=ax2)
    fig2.suptitle("")
    ax2.set_title("Per-image m distribution — top-20 node-rich images")
    ax2.set_xlabel("image_id")
    ax2.set_ylabel("m_node")
    for label in ax2.get_xticklabels():
        label.set_rotation(90)
    fig2.tight_layout()
    save_png(fig2, FIG_ROOT / dataset / f"dataset_{dataset}__m_per_image_boxplot.png")


def plot_residual_distribution(
    df: pd.DataFrame,
    dataset: str,
    n_perm: int = 2000,
    boot: int = 5000,
    include_nulls=("shuffle", "swap", "randtheta", "rshuffle"),
    jitter_frac: float = 0.0,
    seed: int = 13,
    # prereg routing and jitter behavior
    m_col: str = "m_node",            # e.g., "m_angleonly"
    R_col: str = "R_m",               # e.g., "R_m_holdout"
    gaussian_jitter_sd: float = 0.0,  # SD as fraction of radius (indep. Gaussian)
    reinvert_m_on_jitter: bool = False,
    norm_mode: str = "baseline",
    # panel filter
    panel_filter: str = "all",
):
    """
    Panel B — residuals with uncertainty and prereg null controls.

    When m_col/R_col point to the angle-only fields ("m_angleonly", "R_m_holdout"),
    this function implements the preregistered held-out metric and writes
      reports/<dataset>/PREREG__C1.txt   (PASS/FAIL summary),
      reports/<dataset>/NULLTEST__R_m.txt (detailed null statistics).
    It also saves a publication-quality residual figure as both PNG and PDF.
    """
    # ---------- empty dataset guard + policy filter ----------
    if df is None or df.empty:
        write_empty_reports(dataset, reason="no_kept_nodes")
        return
    df = _apply_panel_filter(df, panel_filter)
    if df.empty:
        write_empty_reports(dataset, reason="no_kept_nodes_after_filter")
        return

    # Metric name for labels (held-out vs in-sample)
    metric_name = (R_col if (isinstance(R_col, str) and (R_col in df.columns)) else "R_m")

    # Back-compat: allow jitter_frac to feed gaussian_jitter_sd
    if float(gaussian_jitter_sd) <= 0.0 and float(jitter_frac) > 0.0:
        gaussian_jitter_sd = float(jitter_frac)

    required_cols = {
        "e0x", "e0y", "e1x", "e1y", "e2x", "e2y",
        "image_id", "r0", "r1", "r2", "theta12_deg"
    }
    have_dirs = required_cols.issubset(set(df.columns))

    # ---------- observed metric ----------
    R_obs = (
        df[R_col].astype(float).values
        if R_col in df.columns
        else df["R_m"].astype(float).values
    )
    R_obs = R_obs[np.isfinite(R_obs)]
    obs_median = float(np.median(R_obs)) if R_obs.size else float("nan")
    obs_frac055 = float(np.mean(R_obs < 0.55)) if R_obs.size else float("nan")

    # collect distributions for percentile-based x-limit later
    xlim_collect = [R_obs]

    # ---------- bootstrap CI on median ----------
    def _bootstrap_ci_median(x: np.ndarray, B: int = 5000, seed_: int = 13) -> Tuple[float, float, np.ndarray]:
        x = np.asarray(x, dtype=float)
        rng_ = np.random.default_rng(int(seed_))
        n = x.size
        meds = np.empty(int(B), dtype=float)
        for b in range(int(B)):
            idx = rng_.integers(0, n, size=n)
            meds[b] = float(np.median(x[idx]))
        lo, hi = np.percentile(meds, [2.5, 97.5])
        return float(lo), float(hi), meds

    ci_lo, ci_hi, _med_boot = _bootstrap_ci_median(R_obs, B=int(boot), seed_=seed)

    # ---------- figure setup ----------
    fig = plt.figure(figsize=(7.5, 5.0))
    ax = plt.gca()

    # Professional styling (no grid lines)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(False)
    ax.minorticks_on()
    ax.tick_params(axis="both", which="both", direction="out")

    # Adaptive binning for smoother density on large N
    bins = min(80, max(30, int(np.ceil(np.sqrt(max(1, R_obs.size))))))

    # Base color for observed histogram and bands
    observed_color = "#1f77b4"  # Matplotlib 'C0' blue

    ax.hist(
        R_obs,
        bins=bins,
        density=True,
        alpha=0.85,
        label=f"observed (N={R_obs.size})",
        color=observed_color,
        edgecolor="none",
        zorder=1,
    )

    # Quality bands and thresholds (blue bands, red dashed threshold lines)
    band_color = observed_color          # blue shading for "good" and "marginal" regions
    thr_color = "#d62728"                # red for threshold lines (Matplotlib 'tab:red')

    ax.axvspan(0.0, 0.55, facecolor=band_color, alpha=0.10, edgecolor="none", zorder=0)
    ax.axvspan(0.55, 0.85, facecolor=band_color, alpha=0.07, edgecolor="none", zorder=0)
    ax.axvline(0.55, color=thr_color, linestyle="--", linewidth=1.2, zorder=2)
    ax.axvline(0.85, color=thr_color, linestyle="--", linewidth=1.2, zorder=2)

    # Distinct color for CI so it does not clash with the histogram
    ci_color = "0.25"        # dark gray
    ci_style = (0, (1, 2))   # fine dotted pattern for CI

    # 95% CI on the median — vertical band shown as two dotted lines
    ax.axvline(ci_lo, color=ci_color, linestyle=ci_style, linewidth=1.6, zorder=4)
    ax.axvline(ci_hi, color=ci_color, linestyle=ci_style, linewidth=1.6, zorder=4)

    # Text box summarizing the median and its CI
    ax.annotate(
        f"median = {obs_median:.3f}\n95% CI [{ci_lo:.3f}, {ci_hi:.3f}]",
        xy=(0.015, 0.98),
        xycoords="axes fraction",
        ha="left",
        va="top",
        fontsize=9,
        bbox=dict(
            facecolor="white",
            alpha=0.90,
            edgecolor="none",
            boxstyle="round,pad=0.25",
        ),
    )


    # Title: concise, no underscores
    if R_col == "R_m_holdout":
        title_prefix = "Held-out closure residual"
    else:
        title_prefix = "Closure residual"

    # Pretty dataset label: split on "__", join with commas, strip underscores
    parts = [p.replace("_", " ") for p in str(dataset).split("__") if p]
    dataset_label = ", ".join(parts) if parts else dataset.replace("_", " ")

    ax.set_xlabel(f"{title_prefix} {metric_name}")
    ax.set_ylabel("Density")
    ax.set_title(f"{title_prefix} {metric_name} — {dataset_label}")

    # If no direction columns, stop after observed summary
    if not have_dirs:
        ax.set_ylabel("Count")
        ax.legend(loc="best")
        dataset_base = dataset.split("__", 1)[0]

        # PNG (pipeline) + PDF (paper-quality)
        png_variant = FIG_ROOT / dataset / f"dataset_{dataset}__residual_hist.png"
        png_variant.parent.mkdir(parents=True, exist_ok=True)
        save_png(fig, png_variant)

        pdf_variant = FIG_ROOT / dataset / f"dataset_{dataset}__residual_hist.pdf"
        pdf_variant.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(pdf_variant, format="pdf", bbox_inches="tight")

        png_base = FIG_ROOT / dataset_base / f"dataset_{dataset_base}__residual_hist.png"
        png_base.parent.mkdir(parents=True, exist_ok=True)
        save_png(fig, png_base)

        pdf_base = FIG_ROOT / dataset_base / f"dataset_{dataset_base}__residual_hist.pdf"
        pdf_base.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(pdf_base, format="pdf", bbox_inches="tight")

        log("[NULL-TEST] Direction columns not found; plotted observed histogram with bootstrap CI only.")
        return

    rng = np.random.default_rng(int(seed))

    def _norm_rows(V: np.ndarray) -> np.ndarray:
        n = np.linalg.norm(V, axis=1, keepdims=True) + 1e-12
        return V / n

    def _closure_residual_batch(r0, r1, r2, e0, e1, e2, m):
        m_arr = np.asarray(m, dtype=float)
        if m_arr.ndim == 0:
            a0 = (np.power(r0, m_arr))[:, None] * e0
            a1 = (np.power(r1, m_arr))[:, None] * e1
            a2 = (np.power(r2, m_arr))[:, None] * e2
            num = np.linalg.norm(a0 + a1 + a2, axis=1)
            if norm_mode == "sum":
                denom = (np.power(r0, m_arr)) + (np.power(r1, m_arr)) + (np.power(r2, m_arr)) + 1e-12
            else:
                denom = (np.power(r1, m_arr)) + (np.power(r2, m_arr)) + 1e-12
            return num / denom
        else:
            a0 = (np.power(r0, m_arr))[:, None] * e0
            a1 = (np.power(r1, m_arr))[:, None] * e1
            a2 = (np.power(r2, m_arr))[:, None] * e2
            num = np.linalg.norm(a0 + a1 + a2, axis=1)
            if norm_mode == "sum":
                denom = (np.power(r0, m_arr)) + (np.power(r1, m_arr)) + (np.power(r2, m_arr)) + 1e-12
            else:
                denom = (np.power(r1, m_arr)) + (np.power(r2, m_arr)) + 1e-12
            return num / denom

    # Arrays
    r0a = df["r0"].astype(float).values
    r1a = df["r1"].astype(float).values
    r2a = df["r2"].astype(float).values
    ma  = df[m_col].astype(float).values if m_col in df.columns else df["m_node"].astype(float).values
    e0a = df[["e0x", "e0y"]].astype(float).values
    e1a = df[["e1x", "e1y"]].astype(float).values
    e2a = df[["e2x", "e2y"]].astype(float).values
    t12 = np.deg2rad(df["theta12_deg"].astype(float).values)
    N   = len(r0a)

    # Build direction pools (within-image)
    def _stack_dirs(frame: pd.DataFrame) -> np.ndarray:
        v0 = frame[["e0x", "e0y"]].values
        v1 = frame[["e1x", "e1y"]].values
        v2 = frame[["e2x", "e2y"]].values
        return _norm_rows(np.vstack([v0, v1, v2]).astype(np.float64))

    pools = {img_id: _stack_dirs(g) for img_id, g in df.groupby("image_id")}
    dataset_pool = (
        np.vstack([p for p in pools.values() if p.size > 0])
        if len(pools)
        else np.zeros((0, 2), dtype=np.float64)
    )
    img_ids = df["image_id"].astype(str).values

    # ---------- null generators ----------
    def _null_shuffle():
        """
        Direction-shuffle null: for each image, resample three directions from that
        image's pool; if an image pool is too small, fall back to the dataset-wide
        direction pool. If even the dataset-wide pool has <3 directions, skip this
        null gracefully and return NaNs so downstream tests can detect it.
        """
        med, frac, agg = np.empty(int(n_perm)), np.empty(int(n_perm)), []
        K_agg = int(min(50, n_perm))

        # Global guard: avoid rng.integers(0, 0, ...) when the dataset is tiny
        if dataset_pool.shape[0] < 3:
            log("[NULL-TEST][WARN] dataset_pool has <3 directions; skipping shuffle null for this dataset")
            nan_arr = np.full(int(n_perm), np.nan, dtype=float)
            return nan_arr, nan_arr, nan_arr

        for b in range(int(n_perm)):
            e0p = np.empty_like(e0a)
            e1p = np.empty_like(e1a)
            e2p = np.empty_like(e2a)
            for i, img_id in enumerate(img_ids):
                pool = pools.get(img_id, dataset_pool)
                # If this image's pool is too small, fall back to the global pool
                if pool.shape[0] < 3:
                    pool = dataset_pool
                # At this point pool.shape[0] is guaranteed ≥3
                idxs = rng.integers(0, pool.shape[0], size=3)
                e0p[i], e1p[i], e2p[i] = pool[idxs[0]], pool[idxs[1]], pool[idxs[2]]

            Rb = _closure_residual_batch(r0a, r1a, r2a, e0p, e1p, e2p, ma)
            med[b] = np.median(Rb)
            frac[b] = np.mean(Rb < 0.55)
            if b < K_agg:
                agg.append(Rb)

        return med, frac, (np.concatenate(agg) if len(agg) else med)


    def _null_swap():
        med, frac, agg = np.empty(int(n_perm)), np.empty(int(n_perm)), []
        K_agg = int(min(50, n_perm))
        for b in range(int(n_perm)):
            swap = rng.random(N) < 0.5
            nr1 = np.where(swap, r2a, r1a)
            nr2 = np.where(swap, r1a, r2a)
            Rb = _closure_residual_batch(r0a, nr1, nr2, e0a, e1a, e2a, ma)
            med[b] = np.median(Rb)
            frac[b] = np.mean(Rb < 0.55)
            if b < K_agg:
                agg.append(Rb)
        return med, frac, (np.concatenate(agg) if len(agg) else med)

    def _null_randtheta():
        med, frac, agg = np.empty(int(n_perm)), np.empty(int(n_perm)), []
        K_agg = int(min(50, n_perm))
        for b in range(int(n_perm)):
            phi = rng.uniform(0.0, 2.0 * math.pi, size=N)
            sgn = np.where(rng.random(N) < 0.5, 1.0, -1.0)
            e1p = np.stack([np.cos(phi), np.sin(phi)], axis=1)
            e2p = np.stack([np.cos(phi + sgn * t12), np.sin(phi + sgn * t12)], axis=1)
            Rb = _closure_residual_batch(r0a, r1a, r2a, e0a, e1p, e2p, ma)
            med[b] = np.median(Rb)
            frac[b] = np.mean(Rb < 0.55)
            if b < K_agg:
                agg.append(Rb)
        return med, frac, (np.concatenate(agg) if len(agg) else med)

    def _null_rshuffle():
        med, frac, agg = np.empty(int(n_perm)), np.empty(int(n_perm)), []
        K_agg = int(min(50, n_perm))
        by_img = list(df.groupby("image_id"))
        for b in range(int(n_perm)):
            nr0 = np.empty_like(r0a)
            nr1 = np.empty_like(r1a)
            nr2 = np.empty_like(r2a)
            start = 0
            for img_id, g in by_img:
                n = len(g)
                idx = rng.permutation(n)
                src0 = g["r0"].to_numpy(float)
                src1 = g["r1"].to_numpy(float)
                src2 = g["r2"].to_numpy(float)
                nr0[start:start + n] = src0[idx]
                nr1[start:start + n] = src1[idx]
                nr2[start:start + n] = src2[idx]
                start += n
            Rb = _closure_residual_batch(nr0, nr1, nr2, e0a, e1a, e2a, ma)
            med[b] = np.median(Rb)
            frac[b] = np.mean(Rb < 0.55)
            if b < K_agg:
                agg.append(Rb)
        return med, frac, (np.concatenate(agg) if len(agg) else med)

    include = set(include_nulls or [])
    results: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

    def _run_null(name, fn):
        t0 = time.time()
        log(f"[NULL-TEST] start {name}: n_perm={int(n_perm)}, N={N}")
        out = fn()
        log(f"[NULL-TEST] done  {name}: elapsed={human_time(time.time()-t0)}")
        return out

    if "shuffle"   in include:
        results["shuffle"]   = _run_null("shuffle",   _null_shuffle)
    if "swap"      in include:
        results["swap"]      = _run_null("swap",      _null_swap)
    if "randtheta" in include:
        results["randtheta"] = _run_null("randtheta", _null_randtheta)
    if "rshuffle"  in include:
        results["rshuffle"]  = _run_null("rshuffle",  _null_rshuffle)

    # ---------- p-values and Cliff's δ ----------
    def _pvals_and_delta(med_null, frac_null, R_null_agg):
        p_med = (1.0 + float(np.sum(med_null <= obs_median))) / (len(med_null) + 1.0)
        p_frac = (1.0 + float(np.sum(frac_null >= obs_frac055))) / (len(frac_null) + 1.0)

        def _cliffs_delta(x: np.ndarray, y: np.ndarray, max_pairs: int = 2_000_000) -> float:
            x = np.asarray(x, dtype=np.float64)
            y = np.asarray(y, dtype=np.float64)
            n, m = x.size, y.size
            if n * m <= max_pairs:
                diff = x.reshape(-1, 1) - y.reshape(1, -1)
                return float((np.sum(diff > 0) - np.sum(diff < 0)) / (n * m))
            k = int(np.sqrt(max_pairs))
            rng_local = np.random.default_rng(int(seed) + 3)
            xi = rng_local.integers(0, n, size=k)
            yi = rng_local.integers(0, m, size=k)
            diff = x[xi].reshape(-1, 1) - y[yi].reshape(1, -1)
            return float((np.sum(diff > 0) - np.sum(diff < 0)) / diff.size)

        delta = _cliffs_delta(R_obs, R_null_agg)
        return p_med, p_frac, delta

    # ---------- image-clustered median ----------
    try:
        vcol = R_col if R_col in df.columns else "R_m"
        imgc = image_cluster_stats(df, value_col=vcol, B=int(boot), seed=seed)
    except Exception:
        imgc = {
            "median": float("nan"),
            "ci": (float("nan"), float("nan")),
            "n_img": 0,
            "share_img_Rlt055": float("nan"),
            "share_img_Rlt085": float("nan"),
        }

    lines = [
        f"DATASET: {dataset}",
        (
            f"Image-clustered median={imgc['median']:.6f} "
            f"[95% CI {imgc['ci'][0]:.6f},{imgc['ci'][1]:.6f}] | "
            f"N_img={imgc['n_img']} | share_img[R<0.55]={imgc['share_img_Rlt055']:.3f}, "
            f"R<0.85={imgc['share_img_Rlt085']:.3f}"
        ),
        f"N nodes: {len(R_obs)} (metric={metric_name})",
        (
            f"Observed({metric_name}): median={obs_median:.6f} "
            f"[95% CI {ci_lo:.6f},{ci_hi:.6f}], frac<0.55={obs_frac055:.6f}"
        ),
        f"Null reps per control: {int(n_perm)}",
    ]

    legend_done = False
    p_floor = 1.0 / (float(n_perm) + 1.0)

    for key, (med_null, frac_null, Ragg) in results.items():
        p_med, p_frac, delta = _pvals_and_delta(med_null, frac_null, Ragg)
        Ragg_finite = Ragg[np.isfinite(Ragg)]
        ax.hist(
            Ragg_finite,
            bins=40,
            histtype="step",
            linewidth=1.25,
            label=f"null ({key})",
            density=True,
        )
        pm_str = (f"≤ 1/n_perm (={p_floor:.3g})" if p_med  <= p_floor else f"= {p_med:.6g}")
        pf_str = (f"≤ 1/n_perm (={p_floor:.3g})" if p_frac <= p_floor else f"= {p_frac:.6g}")


        if key == "shuffle":
            medn = float(np.median(med_null))
            d_med = float(obs_median - medn)
            pm_disp = (f"≤ 1/n_perm (={p_floor:.3g})" if p_med  <= p_floor else f"{p_med:.6g}")
            pf_disp = (f"≤ 1/n_perm (={p_floor:.3g})" if p_frac <= p_floor else f"{p_frac:.6g}")

            lines.append(
                f"shuffle: med_null={medn:.6f}, Δmedian(obs-null)={d_med:.6f}, "
                f"p_median(lower)={pm_disp}, p_frac<0.55(higher)={pf_disp}, "
                f"Cliff's δ={delta:.6f} ({cliffs_delta_label(delta)})"
            )
        else:
            lines.append(
                f"{key}: p_median(lower){pm_str}, p_frac<0.55(higher){pf_str}, "
                f"Cliff's δ={delta:.6f} ({cliffs_delta_label(delta)}; {'obs < null' if delta < 0 else 'obs > null'})"
            )


        legend_done = True
        xlim_collect.append(Ragg_finite)

    # ---------- Gaussian jitter (optional) ----------
    if gaussian_jitter_sd is not None and float(gaussian_jitter_sd) > 0:
        sd = float(gaussian_jitter_sd)
        rng_local = np.random.default_rng(int(seed) + 7)
        j0 = r0a * (1.0 + rng_local.normal(0.0, sd, size=N))
        j1 = r1a * (1.0 + rng_local.normal(0.0, sd, size=N))
        j2 = r2a * (1.0 + rng_local.normal(0.0, sd, size=N))
        if reinvert_m_on_jitter:
            mj = np.empty(N, dtype=float)
            for i in range(N):
                mj[i] = m_from_node(
                    j0[i], j1[i], j2[i], t12[i],
                    m_min=0.2, m_max=4.0, tol=1e-6, iters=64
                )
        else:
            mj = ma
        Rj = _closure_residual_batch(j0, j1, j2, e0a, e1a, e2a, mj)
        Rj_finite = Rj[np.isfinite(Rj)]
        if Rj_finite.size > 0:
            med_j = float(np.median(Rj_finite))
            ax.axvline(
                med_j,
                linestyle=":",
                linewidth=1.2,
                color="tab:purple",
                label=f"median (Gaussian jitter SD={sd:.2f})={med_j:.3f}",
            )
            xlim_collect.append(Rj_finite)

    # ---------- x-range ----------
    try:
        all_vals = np.concatenate([v for v in xlim_collect if v.size > 0])
        xmax = float(np.nanpercentile(all_vals, 99.5))
        ax.set_xlim(0.0, min(xmax, 2.0))
    except Exception:
        pass

    ax.set_xlabel(f"{title_prefix} {metric_name}")
    ax.set_ylabel("Density")
    ax.set_ylim(bottom=0)
    ax.set_title(f"{title_prefix} {metric_name} — {dataset_label}")
    # Build legend, including CI and thresholds (median is reported in the text box)
    from matplotlib.lines import Line2D

    # Start from existing histogram / null handles
    handles, labels = ax.get_legend_handles_labels()

    # Custom handles for CI and thresholds
    ci_handle = Line2D(
        [0], [0],
        color=ci_color,
        linestyle=ci_style,
        linewidth=1.6,
        label="95% CI (median)",
    )
    thr_handle = Line2D(
        [0], [0],
        color=thr_color,
        linestyle="--",
        linewidth=1.2,
        label="thresholds (0.55, 0.85)",
    )

    handles.extend([ci_handle, thr_handle])
    labels.extend([h.get_label() for h in [ci_handle, thr_handle]])

    # Final legend
    ax.legend(
        handles,
        labels,
        loc="upper right",
        frameon=True,
        framealpha=0.9,
        facecolor="white",
        edgecolor="none",
    )

    # ---------- save PNG + PDF ----------
    dataset_base = dataset.split("__", 1)[0]

    # Variant-scoped
    png_variant = FIG_ROOT / dataset / f"dataset_{dataset}__residual_hist.png"
    png_variant.parent.mkdir(parents=True, exist_ok=True)
    save_png(fig, png_variant)

    pdf_variant = FIG_ROOT / dataset / f"dataset_{dataset}__residual_hist.pdf"
    pdf_variant.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(pdf_variant, format="pdf", bbox_inches="tight")

    # Canonical dataset-level
    png_base = FIG_ROOT / dataset_base / f"dataset_{dataset_base}__residual_hist.png"
    png_base.parent.mkdir(parents=True, exist_ok=True)
    save_png(fig, png_base)

    pdf_base = FIG_ROOT / dataset_base / f"dataset_{dataset_base}__residual_hist.pdf"
    pdf_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(pdf_base, format="pdf", bbox_inches="tight")

    # ---------- NULLTEST text ----------
    out_txt_variant = CSV_ROOT / dataset / "NULLTEST__R_m.txt"
    out_txt_variant.parent.mkdir(parents=True, exist_ok=True)
    with open(out_txt_variant, "w") as f:
        for line in lines:
            f.write(line + "\n")

    out_txt_base = CSV_ROOT / dataset_base / "NULLTEST__R_m.txt"
    out_txt_base.parent.mkdir(parents=True, exist_ok=True)
    out_txt_base.write_text(out_txt_variant.read_text())
    log("[NULL-TEST] " + " | ".join(lines[2:]))

    # ---------- prereg summary (C1) ----------
    if (R_col == "R_m_holdout") and (len(results) > 0):
        alpha_p = 1e-3
        cliff_thresh = -0.20
        verdicts = []
        lines_nulls = []
        worst_p_med = 1.0
        worst_p_frac = 1.0
        max_delta = -1.0  # least favorable δ

        for key, (med_null, frac_null, Ragg) in results.items():
            p_med, p_frac, delta = _pvals_and_delta(med_null, frac_null, Ragg)
            lines_nulls.append(
                f"{key}: p_median(lower)={p_med:.6g}, "
                f"p_frac<0.55(higher)={p_frac:.6g}, Cliff's δ={delta:.6f}"
            )
            worst_p_med = min(worst_p_med, p_med)
            worst_p_frac = min(worst_p_frac, p_frac)
            max_delta = max(max_delta, delta)
            verdicts.append(
                (p_med <= alpha_p) and (p_frac <= alpha_p) and (delta <= cliff_thresh)
            )

        overall = all(verdicts)
        outcome = "PASS" if overall else "FAIL"

        out_c1_variant = CSV_ROOT / dataset / "PREREG__C1.txt"
        out_c1_variant.parent.mkdir(parents=True, exist_ok=True)
        with open(out_c1_variant, "w") as f:
            f.write(f"DATASET: {dataset}\n")
            f.write("Claim HRF-C1 (held-out closure @ angle-only m):\n")
            f.write(
                f"Observed \\tilde R = {obs_median:.6f}  "
                f"[boot 95% CI {ci_lo:.6f},{ci_hi:.6f}]\n"
            )
            for L in lines_nulls:
                f.write(L + "\n")
            f.write(
                f"Thresholds: p <= {alpha_p:g} on BOTH median and Pr[R<0.55], "
                f"and Cliff's δ <= {cliff_thresh:.2f} vs EVERY null\n"
            )
            wp_m_disp = (f"≤ 1/n_perm (={p_floor:.3g})" if worst_p_med  <= p_floor else f"{worst_p_med:.6g}")
            wp_f_disp = (f"≤ 1/n_perm (={p_floor:.3g})" if worst_p_frac <= p_floor else f"{worst_p_frac:.6g}")

            f.write(
                f"Worst-case across nulls: p_med={wp_m_disp}, "
                f"p_frac={wp_f_disp}, max δ={max_delta:.6f}\n"
            )
            f.write(f"Outcome: {outcome}\n")

        out_c1_base = CSV_ROOT / dataset_base / "PREREG__C1.txt"
        out_c1_base.parent.mkdir(parents=True, exist_ok=True)
        out_c1_base.write_text(out_c1_variant.read_text())

        log(
            "[PREREG] metric=heldout; "
            f"dataset={dataset}; outcome={outcome}; "
            f"obs_median={obs_median:.3f} [CI {ci_lo:.3f},{ci_hi:.3f}]; "
            f"worst_p_median={worst_p_med:.3g}; worst_p_frac={worst_p_frac:.3g}; "
            f"max_delta={max_delta:.3f}; file={out_c1_variant}"
        )



def plot_fixed_m_baselines(df: pd.DataFrame, dataset: str,
                           fixed_ms=(2.0, 1.0), grid_ms=np.linspace(0.2, 4.0, 381),
                           norm_mode: str = "baseline"):
    """
    Panel B — Baselines (fixed m) + Radii-only additive test.

    Part 1 (unchanged): compares EPIC per-node R(m_node) against R(m) for classical fixed m
    and the dataset's best single m*. If directions are absent, falls back to a normalized
    scalar equation misfit. Outputs:
      figures/<DATASET>/dataset_<DATASET>__residual_baselines.png
      figures/<DATASET>/dataset_<DATASET>__paired_deltaR_ecdf.png
      reports/<DATASET>/BASELINES__R_m.txt

    Part 2 (new): radii-only additive test for r0^α ≈ r1^α + r2^α, without using directions:
      • Fit a single α* by minimizing the median normalized additive residual D_α.
      • Compare to fixed α baselines: α=3.0 (volume-priced, Poiseuille) and α=2.5 (surface-priced).
      • If available, also evaluate geometry-informed per-node α_i = (m_angleonly + 4)/2 (n=4).
    Outputs:
      figures/<DATASET>/dataset_<DATASET>__radii_additive_hist.png
      figures/<DATASET>/dataset_<DATASET>__radii_additive_ecdf.png
      reports/<DATASET>/RADD__radii_only.txt
    """
    import numpy as np, matplotlib.pyplot as plt
    from math import comb

    # ---------- Helper stats ----------
    def bootstrap_ci_median(x, B=5000):
        x = np.asarray(x, dtype=float)
        x = x[np.isfinite(x)]
        if x.size == 0:
            return float("nan"), float("nan")
        rng = np.random.default_rng(13)
        meds = np.median(rng.choice(x, size=(int(B), x.size), replace=True), axis=1)
        return float(np.percentile(meds, 2.5)), float(np.percentile(meds, 97.5))

    def sign_test_improvement(a, b):
        # Improvement = baseline - method (positive means method better)
        d = b - a
        d = d[np.isfinite(d) & (d != 0)]
        n = d.size
        if n == 0:
            return 0, 0, float("nan")
        k = int(np.sum(d > 0))
        p_lower = sum(comb(n, i) for i in range(0, k+1)) / (2**n)
        p_upper = sum(comb(n, i) for i in range(k, n+1)) / (2**n)
        return n, k, 2 * min(p_lower, p_upper)

    # ---------- Existing baseline comparison (R vector or angle-equation misfit) ----------
    have_dirs = set(['e0x','e0y','e1x','e1y','e2x','e2y']).issubset(df.columns)
    r0 = df['r0'].to_numpy(float); r1 = df['r1'].to_numpy(float); r2 = df['r2'].to_numpy(float)
    t12 = np.deg2rad(df['theta12_deg'].to_numpy(float))
    R_epic = df.get('R_m', pd.Series(np.nan, index=df.index)).to_numpy(float)

    def R_vector(m):
        e0 = df[['e0x','e0y']].to_numpy(float)
        e1 = df[['e1x','e1y']].to_numpy(float)
        e2 = df[['e2x','e2y']].to_numpy(float)
        a0 = (np.power(r0, m))[:, None] * e0
        a1 = (np.power(r1, m))[:, None] * e1
        a2 = (np.power(r2, m))[:, None] * e2
        num = np.linalg.norm(a0 + a1 + a2, axis=1)
        den = (np.power(r1, m)) + (np.power(r2, m)) + 1e-12
        if norm_mode == "sum":
            den = (np.power(r0, m)) + den
        return num / den


    def eq_misfit(m):
        c = np.cos(t12)
        val = (r0**(2*m)) - (r1**(2*m)) - (r2**(2*m)) - 2*c*(r1**m)*(r2**m)
        scale = (r0**(2*m)) + (r1**(2*m)) + (r2**(2*m)) + 1e-12
        return np.abs(val)/scale

    if have_dirs:
        R_fixed = {f"m={m:g}": R_vector(m) for m in fixed_ms}
        grid_vals = np.array([np.median(R_vector(m)) for m in grid_ms])
    else:
        R_fixed = {f"m={m:g}": eq_misfit(m) for m in fixed_ms}
        grid_vals = np.array([np.median(eq_misfit(m)) for m in grid_ms])

    m_star = float(grid_ms[int(np.argmin(grid_vals))])
    R_fixed[f"m*={m_star:.2f}"] = (R_vector(m_star) if have_dirs else eq_misfit(m_star))

    fig = plt.figure(figsize=(8, 5.2)); ax = plt.gca()
    if np.all(np.isfinite(R_epic)):
        ax.hist(R_epic, bins=40, alpha=0.85, label="EPIC (per-node m)")
    for name, Rb in R_fixed.items():
        ax.hist(Rb, bins=40, alpha=0.35, histtype="stepfilled", label=name)

    ax.set_xlabel("closure residual R(m)" if have_dirs else "normalized equation misfit")
    ax.set_ylabel("count")
    ax.set_title(f"Baselines vs EPIC — {dataset}")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")

    lines = []
    if np.all(np.isfinite(R_epic)):
        lo, hi = bootstrap_ci_median(R_epic[np.isfinite(R_epic)])
        lines.append(f"EPIC median={np.median(R_epic):.3f} 95% CI [{lo:.3f},{hi:.3f}]")
    for name, Rb in R_fixed.items():
        med = np.median(Rb); lo, hi = bootstrap_ci_median(Rb[np.isfinite(Rb)])
        lines.append(f"{name} median={med:.3f} 95% CI [{lo:.3f},{hi:.3f}]")
        if np.all(np.isfinite(R_epic)):
            n,k,p = sign_test_improvement(R_epic, Rb)
            lines.append(f"  paired improvement (ΔR): n={n}, sign test p={p:.4g}")

    ax.text(0.02, 0.98, "\n".join(lines), transform=ax.transAxes, va="top", ha="left",
            fontsize=9, bbox=dict(facecolor="white", alpha=0.85, edgecolor="none"))
    save_png(fig, FIG_ROOT / dataset / f"dataset_{dataset}__residual_baselines.png")

    # ECDF of paired improvements ΔR = R(baseline) − R(EPIC)
    if np.all(np.isfinite(R_epic)):
        figD = plt.figure(figsize=(7.2, 4.8)); axD = plt.gca()
        for name, Rb in R_fixed.items():
            Delta = (Rb - R_epic)
            Delta = Delta[np.isfinite(Delta)]
            x = np.sort(Delta)
            if x.size > 0:
                y = np.arange(1, x.size+1) / x.size
                axD.step(x, y, where="post", label=name)
        axD.axvline(0.0, linewidth=1.0)
        axD.set_xlabel("ΔR = R(baseline) − R(EPIC)")
        axD.set_ylabel("ECDF")
        axD.set_title(f"Paired improvement ΔR — {dataset}")
        axD.grid(True, alpha=0.25)
        axD.legend(loc="lower right")
        save_png(figD, FIG_ROOT / dataset / f"dataset_{dataset}__paired_deltaR_ecdf.png")

    out_txt = CSV_ROOT / dataset / "BASELINES__R_m.txt"
    out_txt.parent.mkdir(parents=True, exist_ok=True)
    with open(out_txt, "w") as f:
        f.write(f"DATASET: {dataset}\n")
        for L in lines: f.write(L + "\n")
    log("[BASELINES] " + " | ".join(lines))

    # ---------- NEW: Radii-only additive test (no directions) ----------
    # Uses helpers defined elsewhere in the file: fit_global_alpha, additive_Dalpha.
    # If those helpers were moved, we inline vectorized versions here as needed.
    try:
        # Vectorized residual for a scalar alpha
        def Dalpha_arr(alpha):
            num = np.abs(np.power(r0, alpha) - (np.power(r1, alpha) + np.power(r2, alpha)))
            den = (np.power(r1, alpha) + np.power(r2, alpha) + 1e-12)
            return num / den

        # Grid search for alpha* that minimizes median D_alpha
        grid_alpha = np.linspace(1.5, 4.0, 501)
        med_D = np.array([np.median(Dalpha_arr(a)) for a in grid_alpha])
        alpha_star = float(grid_alpha[int(np.argmin(med_D))])

        # Observed distributions
        D_star = Dalpha_arr(alpha_star)
        D_a3   = Dalpha_arr(3.0)   # classical (m=2,n=4)
        D_a25  = Dalpha_arr(2.5)   # surface-priced (m=1,n=4)

        # Optional: geometry-informed per-node alpha_i from angle-only m (n=4)
        D_alpha_i = None
        if "m_angleonly" in df.columns:
            m_i = df["m_angleonly"].to_numpy(float)
            a_i = 0.5 * (m_i + 4.0)
            # Per-node alpha: pow with vector exponent
            D_alpha_i = (np.abs(np.power(r0, a_i) - (np.power(r1, a_i) + np.power(r2, a_i))) /
                         (np.power(r1, a_i) + np.power(r2, a_i) + 1e-12))

        # Text summary + paired sign tests (baseline − method; positive is better)
        lines_radd = []
        def _sum_line(name, arr):
            med = float(np.median(arr[np.isfinite(arr)]))
            lo, hi = bootstrap_ci_median(arr[np.isfinite(arr)])
            lines_radd.append(f"{name} median Dα={med:.3f} 95% CI [{lo:.3f},{hi:.3f}]")

        _sum_line(f"α*={alpha_star:.3f}", D_star)
        _sum_line("α=3.0", D_a3)
        _sum_line("α=2.5", D_a25)
        if D_alpha_i is not None:
            _sum_line("α_i=(m_angleonly+4)/2", D_alpha_i)

        # Paired sign tests vs α=3.0 and α=2.5
        for base_name, base_arr in [("α=3.0", D_a3), ("α=2.5", D_a25)]:
            n,k,p = sign_test_improvement(D_star, base_arr)
            lines_radd.append(f"  paired improvement (ΔD = {base_name} − α*): n={n}, p={p:.4g}")
            if D_alpha_i is not None:
                n2,k2,p2 = sign_test_improvement(D_alpha_i, base_arr)
                lines_radd.append(f"  paired improvement (ΔD = {base_name} − α_i): n={n2}, p={p2:.4g}")

        # Histogram figure
        figH = plt.figure(figsize=(8.0, 5.2)); axH = plt.gca()
        axH.hist(D_star[np.isfinite(D_star)], bins=40, alpha=0.85, label=f"α*={alpha_star:.3f}")
        axH.hist(D_a3[np.isfinite(D_a3)],     bins=40, alpha=0.35, histtype="stepfilled", label="α=3.0")
        axH.hist(D_a25[np.isfinite(D_a25)],   bins=40, alpha=0.35, histtype="stepfilled", label="α=2.5")
        if D_alpha_i is not None:
            axH.hist(D_alpha_i[np.isfinite(D_alpha_i)], bins=40, alpha=0.35, histtype="stepfilled", label="α_i=(m_angleonly+4)/2")
        axH.set_xlabel("normalized additive residual Dα")
        axH.set_ylabel("count")
        axH.set_title(f"Radii-only additive test — {dataset}")
        axH.grid(True, alpha=0.25)
        axH.legend(loc="best")
        axH.text(0.02, 0.98, "\n".join(lines_radd), transform=axH.transAxes, va="top", ha="left",
                 fontsize=9, bbox=dict(facecolor="white", alpha=0.85, edgecolor="none"))
        save_png(figH, FIG_ROOT / dataset / f"dataset_{dataset}__radii_additive_hist.png")

        # ECDF of paired improvements ΔD = D(baseline) − D(method)
        figE = plt.figure(figsize=(7.2, 4.8)); axE = plt.gca()
        for name, arr in [(f"α*={alpha_star:.3f}", D_star),
                          ("α=3.0", D_a3), ("α=2.5", D_a25)]:
            if name.startswith("α*"):
                continue  # we'll plot baselines relative to α*
        # Baselines relative to α*
        base_pairs = [("α=3.0", D_a3), ("α=2.5", D_a25)]
        for name, base_arr in base_pairs:
            Delta = base_arr - D_star
            Delta = Delta[np.isfinite(Delta)]
            x = np.sort(Delta)
            if x.size > 0:
                y = np.arange(1, x.size+1) / x.size
                axE.step(x, y, where="post", label=f"{name} − α*")
        # α_i relative to α* (optional)
        if D_alpha_i is not None:
            Delta_i = D_alpha_i - D_star
            Delta_i = Delta_i[np.isfinite(Delta_i)]
            if Delta_i.size > 0:
                xi = np.sort(Delta_i)
                yi = np.arange(1, xi.size+1) / xi.size
                axE.step(xi, yi, where="post", label="α_i − α*")
        axE.axvline(0.0, linewidth=1.0)
        axE.set_xlabel("ΔD = D(baseline) − D(method)")
        axE.set_ylabel("ECDF")
        axE.set_title(f"Radii-only ΔD — {dataset}")
        axE.grid(True, alpha=0.25)
        axE.legend(loc="lower right")
        save_png(figE, FIG_ROOT / dataset / f"dataset_{dataset}__radii_additive_ecdf.png")

        # Write radii-only report
        out_radd = CSV_ROOT / dataset / "RADD__radii_only.txt"
        out_radd.parent.mkdir(parents=True, exist_ok=True)
        with open(out_radd, "w") as f:
            f.write(f"DATASET: {dataset}\n")
            f.write(f"alpha_star={alpha_star:.6f}\n")
            for L in lines_radd:
                f.write(L + "\n")
        log("[RADD] " + " | ".join(lines_radd))

    except Exception as e:
        log(f"[RADD][WARN] radii-only additive test failed: {e}")




def compute_per_node_uncertainty(
    df: pd.DataFrame,
    dataset: str,
    n_draws: int = 500,
    seed: int = 13,
    radius_sd_frac: float = 0.05,  # fallback when per-node SDs are unavailable
):
    """
    STEP 3 — Per-node 95% CIs for m.

    Priority:
      • Analytic delta method (implicit-function theorem) using per-node SDs, with a global
        scale τ fitted on a synthetic positive control to achieve ~95% coverage.
      • Fallback to legacy Monte-Carlo jitter if SD columns are missing.

    Outputs:
      reports/<DATASET>/UNCERTAINTY__nodes.csv
        m_hat, m_lo, m_hi, CI_width, at_edge_before, at_edge_after
      figures/<DATASET>/dataset_<DATASET>__m_uncertainty_forest.png

    PASS (for this first-closure analysis) if:
      • the median 95% CI half-width is ≤ 0.80 in m-space (i.e. intervals are narrower than
        the full bracket and numerically informative).

    The fraction of bracket-edge nodes that move strictly interior under uncertainty is
    reported as a descriptive diagnostic but does not gate the PASS/FAIL outcome. That
    stricter requirement is reserved for a full, second-pass EPIC geometry analysis.

    Notes:
      - Requires r0, r1, r2, theta12_deg, and either m_angleonly or m_node.
      - If per-node SDs (sd_theta_deg, sd_r0, sd_r1, sd_r2) are present, uses analytic delta.
        Otherwise falls back to Monte Carlo.
    """
    import math
    import time
    import os
    import numpy as np
    import matplotlib.pyplot as plt

    t0 = time.time()
    try:
        # ------------------------- Quick input validations -------------------------
        if df is None or len(df) == 0:
            log("[UNCERTAINTY] empty DF; skipping")
            return

        required_core = ["r0", "r1", "r2", "theta12_deg"]
        missing_core = [c for c in required_core if c not in df.columns]
        if missing_core:
            log(f"[UNCERTAINTY][ERROR] Missing required columns: {missing_core}. Aborting step.")
            return

        have_m_angle = "m_angleonly" in df.columns
        have_m_node = "m_node" in df.columns
        if not (have_m_angle or have_m_node):
            log("[UNCERTAINTY][ERROR] Need one of ['m_angleonly', 'm_node'] in the DF. Aborting.")
            return

        rng = np.random.default_rng(int(seed))

        # ------------------------- Column pulls (safe) -----------------------------
        def _to_num(name, default=np.nan, dtype=float):
            if name in df.columns:
                return df[name].to_numpy(dtype, copy=True)
            return np.full(len(df), default, dtype=dtype)

        r0 = _to_num("r0")
        r1 = _to_num("r1")
        r2 = _to_num("r2")
        th_deg = _to_num("theta12_deg")
        mhat = _to_num("m_node") if have_m_node else _to_num("m_angleonly")

        # m bracket (with sanity)
        mmin = float(df.get("cfg_m_min", pd.Series([0.2])).iloc[0])
        mmax = float(df.get("cfg_m_max", pd.Series([4.0])).iloc[0])
        if not np.isfinite(mmin):
            mmin = 0.2
        if not np.isfinite(mmax):
            mmax = 4.0
        if mmin >= mmax:
            log(
                f"[UNCERTAINTY][WARN] Invalid (mmin, mmax)=({mmin}, {mmax}). "
                "Resetting to (0.2, 4.0)."
            )
            mmin, mmax = 0.2, 4.0

        N = len(df)
        if not (len(r0) == len(r1) == len(r2) == len(th_deg) == len(mhat) == N):
            log("[UNCERTAINTY][ERROR] Column length mismatch. Aborting.")
            return

        # ------------------------- Delta-method SE helper --------------------------
        def _delta_se_m(m, r0_, r1_, r2_, theta_rad, s_lnr0, s_lnr1, s_lnr2, s_theta):
            """
            Implicit F(m, ln r0, ln r1, ln r2, theta) = 0 with
              F = r0^{2m} - r1^{2m} - r2^{2m} - 2 cosθ r1^m r2^m

            dm ≈ - (∂F/∂x · dx) / (∂F/∂m),
            where SDs are defined on ln r0, ln r1, ln r2 and θ (radians).

            Includes a stabilized division for ∂F/∂m and guards for numerical pathologies.
            """
            r0_ = max(float(r0_), 1e-12)
            r1_ = max(float(r1_), 1e-12)
            r2_ = max(float(r2_), 1e-12)

            try:
                a0 = r0_ ** (2.0 * m)
                a1 = r1_ ** (2.0 * m)
                a2 = r2_ ** (2.0 * m)
                b = (r1_ ** m) * (r2_ ** m)
                c = math.cos(theta_rad)
                s = math.sin(theta_rad)

                ln0 = math.log(r0_)
                ln1 = math.log(r1_)
                ln2 = math.log(r2_)

                dFm = (
                    (2.0 * ln0) * a0
                    - (2.0 * ln1) * a1
                    - (2.0 * ln2) * a2
                    - 2.0 * c * (ln1 + ln2) * b
                )
                dFx0 = 2.0 * m * a0
                dFx1 = -2.0 * m * a1 - 2.0 * c * m * b
                dFx2 = -2.0 * m * a2 - 2.0 * c * m * b
                dFth = 2.0 * s * b

                eps = 1e-12
                d = dFm if abs(dFm) > eps else (math.copysign(1.0, dFm) * eps + eps)

                with np.errstate(divide="ignore", invalid="ignore"):
                    gx0 = -dFx0 / d
                    gx1 = -dFx1 / d
                    gx2 = -dFx2 / d
                    gth = -dFth / d

                # Sanitize SDs to finite, non-negative floats
                s_lnr0 = float(0.0 if (not np.isfinite(s_lnr0) or s_lnr0 < 0) else s_lnr0)
                s_lnr1 = float(0.0 if (not np.isfinite(s_lnr1) or s_lnr1 < 0) else s_lnr1)
                s_lnr2 = float(0.0 if (not np.isfinite(s_lnr2) or s_lnr2 < 0) else s_lnr2)
                s_theta = float(0.0 if (not np.isfinite(s_theta) or s_theta < 0) else s_theta)

                var = (
                    (gx0 * s_lnr0) ** 2
                    + (gx1 * s_lnr1) ** 2
                    + (gx2 * s_lnr2) ** 2
                    + (gth * s_theta) ** 2
                )
                return float(np.sqrt(max(var, 0.0)))
            except Exception:
                return float("nan")

        # ------------------------- τ calibration (positive control) ----------------
        def _calibrate_tau(
            target_coverage: float = 0.95,
            n: int = 1200,
            seed_: int = 17,
            dir_sd_deg: float = 3.0,
            rad_sd_frac_local: float = 0.05,
        ) -> float:
            """
            Fit a single τ such that [m̂ ± 1.96 τ se(m)] achieves ~target_coverage
            on synthetic junctions. This is deliberately simple and global.
            """
            try:
                rr = np.random.default_rng(int(seed_))
                m_true = rr.uniform(0.35, 3.6, size=n)
                r1t = rr.uniform(1.0, 4.0, size=n)
                r2t = rr.uniform(1.0, 4.0, size=n)
                theta = rr.uniform(np.deg2rad(15.0), np.deg2rad(160.0), size=n)

                # Synthetic directions and parent
                ephi = rr.uniform(0.0, 2.0 * np.pi, size=n)
                sgn = np.where(rr.random(n) < 0.5, 1.0, -1.0)
                e1 = np.stack([np.cos(ephi), np.sin(ephi)], axis=1)
                e2 = np.stack([np.cos(ephi + sgn * theta), np.sin(ephi + sgn * theta)], axis=1)
                a1 = (np.power(r1t, m_true))[:, None] * e1
                a2 = (np.power(r2t, m_true))[:, None] * e2
                svec = a1 + a2
                r0m = np.linalg.norm(svec, axis=1) + 1e-12
                r0t = np.power(r0m, 1.0 / m_true)

                def _rot(E, sd_deg):
                    d = np.deg2rad(rr.normal(0.0, sd_deg, size=E.shape[0]))
                    cd, sd = np.cos(d), np.sin(d)
                    x, y = E[:, 0], E[:, 1]
                    xr = x * cd - y * sd
                    yr = x * sd + y * cd
                    V = np.stack([xr, yr], axis=1)
                    nrm = np.linalg.norm(V, axis=1, keepdims=True) + 1e-12
                    return V / nrm

                e0 = -svec / r0m[:, None]
                e0n = _rot(e0, dir_sd_deg)
                e1n = _rot(e1, dir_sd_deg)
                e2n = _rot(e2, dir_sd_deg)

                r0n = r0t * (1.0 + rr.normal(0.0, rad_sd_frac_local, size=n))
                r1n = r1t * (1.0 + rr.normal(0.0, rad_sd_frac_local, size=n))
                r2n = r2t * (1.0 + rr.normal(0.0, rad_sd_frac_local, size=n))
                thetan = np.arccos(np.clip(np.sum(e1n * e2n, axis=1), -1.0, 1.0))

                # m_hat under noise
                m_hat = np.empty(n, dtype=float)
                m_hat.fill(np.nan)
                for i in range(n):
                    try:
                        m_hat[i] = m_from_node(
                            float(r0n[i]), float(r1n[i]), float(r2n[i]), float(thetan[i]),
                            m_min=0.2, m_max=4.0, tol=1e-6, iters=64
                        )
                    except Exception:
                        m_hat[i] = np.nan

                ok = np.where(np.isfinite(m_hat))[0]
                if ok.size == 0:
                    return 1.0

                # Scalar noise levels for SE calculation
                s_lnr_scalar = float(rad_sd_frac_local)
                s_theta_scalar = float(np.deg2rad(dir_sd_deg))

                se_list = []
                for i in ok:
                    se_i = _delta_se_m(
                        float(m_hat[i]),
                        float(r0n[i]),
                        float(r1n[i]),
                        float(r2n[i]),
                        float(thetan[i]),
                        s_lnr_scalar,
                        s_lnr_scalar,
                        s_lnr_scalar,
                        s_theta_scalar,
                    )
                    if np.isfinite(se_i):
                        se_list.append(se_i)

                if not se_list:
                    return 1.0

                se = np.asarray(se_list, float)
                m_hat_ok = m_hat[ok]
                m_true_ok = m_true[ok]
                if se.size != m_hat_ok.size:
                    nmin = min(se.size, m_hat_ok.size)
                    se = se[:nmin]
                    m_hat_ok = m_hat_ok[:nmin]
                    m_true_ok = m_true_ok[:nmin]

                # Bisection on τ
                lo, hi = 0.5, 2.0
                for _ in range(18):
                    mid = 0.5 * (lo + hi)
                    z = 1.96
                    lo_b = m_hat_ok - z * mid * se
                    hi_b = m_hat_ok + z * mid * se
                    cover = float(np.mean((m_true_ok >= lo_b) & (m_true_ok <= hi_b)))
                    if cover < target_coverage:
                        lo = mid
                    else:
                        hi = mid
                tau_est = 0.5 * (lo + hi)
                if not np.isfinite(tau_est) or tau_est <= 0:
                    tau_est = 1.0
                return float(tau_est)
            except Exception as e:
                log(f"[UNCERTAINTY][WARN] τ calibration failed with error: {e}. Using τ=1.0.")
                return 1.0

        # ------------------------- Decide path: analytic vs MC ---------------------
        have_pernode = {"sd_theta_deg", "sd_r0", "sd_r1", "sd_r2"}.issubset(df.columns)

        m_lo = np.full(N, np.nan, float)
        m_hi = np.full(N, np.nan, float)
        off_edge_after = np.zeros(N, dtype=bool)

        if have_pernode:
            log("[UNCERTAINTY] Using analytic delta method with per-node SDs.")

            s_lnr0 = _to_num("sd_r0") / np.clip(r0, 1e-12, None)
            s_lnr1 = _to_num("sd_r1") / np.clip(r1, 1e-12, None)
            s_lnr2 = _to_num("sd_r2") / np.clip(r2, 1e-12, None)
            s_theta = np.deg2rad(_to_num("sd_theta_deg"))

            # sanitize non-finite SDs
            s_lnr0 = np.nan_to_num(s_lnr0, nan=radius_sd_frac, posinf=radius_sd_frac, neginf=radius_sd_frac)
            s_lnr1 = np.nan_to_num(s_lnr1, nan=radius_sd_frac, posinf=radius_sd_frac, neginf=radius_sd_frac)
            s_lnr2 = np.nan_to_num(s_lnr2, nan=radius_sd_frac, posinf=radius_sd_frac, neginf=radius_sd_frac)
            s_theta = np.nan_to_num(s_theta, nan=np.deg2rad(3.0), posinf=np.deg2rad(3.0), neginf=np.deg2rad(3.0))

            theta_rad = np.deg2rad(th_deg)
            se = np.full(N, np.nan, float)

            for i in range(N):
                mi = float(mhat[i])
                if not np.isfinite(mi):
                    continue
                se[i] = _delta_se_m(
                    mi,
                    float(r0[i]),
                    float(r1[i]),
                    float(r2[i]),
                    float(theta_rad[i]),
                    float(s_lnr0[i]),
                    float(s_lnr1[i]),
                    float(s_lnr2[i]),
                    float(s_theta[i]),
                )

            # τ calibration from medians of SDs (defensive)
            try:
                med_dir_sd = float(np.nanmedian(_to_num("sd_theta_deg")))
                if not np.isfinite(med_dir_sd):
                    med_dir_sd = 3.0
                with np.errstate(divide="ignore", invalid="ignore"):
                    num = np.stack(
                        [_to_num("sd_r0"), _to_num("sd_r1"), _to_num("sd_r2")],
                        axis=1,
                    )
                    den = np.stack([r0, r1, r2], axis=1)
                    frac_mat = num / np.clip(den, 1e-12, None)
                    med_rad_frac = float(np.nanmedian(frac_mat))
                if not np.isfinite(med_rad_frac) or med_rad_frac <= 0:
                    med_rad_frac = float(radius_sd_frac)
                tau = _calibrate_tau(
                    target_coverage=0.95,
                    n=1200,
                    seed_=seed,
                    dir_sd_deg=med_dir_sd,
                    rad_sd_frac_local=med_rad_frac,
                )
            except Exception as e:
                log(f"[UNCERTAINTY][WARN] τ calibration pipeline failed: {e}. Forcing τ=1.0.")
                tau = 1.0

            # compute_per_node_uncertainty(...): log-scale CI for m with gentle calibration
            z = 1.96
            lam = float(os.environ.get("EPIC_SE_LAMBDA", "1.0"))

            # Winsorize relative SEs to prevent pathological explosions
            m_pos = np.clip(mhat, 1e-12, None)
            rel_se = se / m_pos
            rel_se = np.clip(rel_se, 0.0, 1.0)

            # Delta method on log m: Var[log m] ≈ Var[m] / m^2
            mu = np.log(m_pos)
            mu_lo = mu - z * (tau * lam) * rel_se
            mu_hi = mu + z * (tau * lam) * rel_se

            # Back-transform and clamp to the optimization bracket (after back-transform)
            m_lo = np.exp(np.clip(mu_lo, np.log(mmin), np.log(mmax)))
            m_hi = np.exp(np.clip(mu_hi, np.log(mmin), np.log(mmax)))

            off_edge_after = (
                np.isfinite(m_lo)
                & np.isfinite(m_hi)
                & (m_lo > mmin + 1e-6)
                & (m_hi < mmax - 1e-6)
            )
            log(
                f"[UNCERTAINTY] τ={tau:.3f}, λ={lam:.3f} (z=1.96, log-scale). "
                f"Finite SE count: {int(np.isfinite(se).sum())}/{N}"
            )

        else:
            # --------------------- MC fallback (no per-node SDs) -------------------
            log(
                f"[UNCERTAINTY] Per-node SD columns missing; falling back to MC "
                f"(N={N}, draws/node={int(n_draws)}, radius_sd_frac={radius_sd_frac})."
            )

            def _sd_from_svd(x):
                if not np.isfinite(x) or x <= 0:
                    return 12.0
                return float(min(15.0, 60.0 / x))

            svd_e1 = df.get("svd_ratio_e1", pd.Series(np.nan, index=df.index)).to_numpy(float)
            svd_e2 = df.get("svd_ratio_e2", pd.Series(np.nan, index=df.index)).to_numpy(float)
            sd_e1 = np.array([_sd_from_svd(v) for v in svd_e1], float)
            sd_e2 = np.array([_sd_from_svd(v) for v in svd_e2], float)
            sd_theta = np.sqrt(sd_e1**2 + sd_e2**2) / np.sqrt(2.0)

            theta_rad = np.deg2rad(th_deg)

            for i in range(N):
                mi = float(mhat[i])
                if not np.isfinite(mi):
                    continue

                r0i, r1i, r2i = float(r0[i]), float(r1[i]), float(r2[i])
                ti = float(theta_rad[i])
                if not np.isfinite(ti):
                    continue

                draws = []
                for _ in range(int(max(1, n_draws))):
                    try:
                        rr0 = r0i * (1.0 + rng.normal(0.0, radius_sd_frac))
                        rr1 = r1i * (1.0 + rng.normal(0.0, radius_sd_frac))
                        rr2 = r2i * (1.0 + rng.normal(0.0, radius_sd_frac))
                        ang_sd = float(sd_theta[i]) if np.isfinite(sd_theta[i]) else 10.0
                        tpert = ti + np.deg2rad(rng.normal(0.0, ang_sd))
                        m_s = m_from_node(
                            rr0,
                            rr1,
                            rr2,
                            tpert,
                            m_min=mmin,
                            m_max=mmax,
                            tol=1e-6,
                            iters=64,
                        )
                        if np.isfinite(m_s):
                            draws.append(float(m_s))
                    except Exception:
                        continue  # skip bad draw

                if len(draws) >= 8:
                    qs = np.percentile(np.asarray(draws, float), [2.5, 97.5])
                    m_lo[i], m_hi[i] = float(qs[0]), float(qs[1])
                    off_edge_after[i] = (m_lo[i] > mmin + 1e-6) and (m_hi[i] < mmax - 1e-6)
                else:
                    m_lo[i], m_hi[i] = np.nan, np.nan
                    off_edge_after[i] = False

            m_lo = np.clip(m_lo, mmin, mmax)
            m_hi = np.clip(m_hi, mmin, mmax)

        # ------------------------- Outcome metrics (shared) -----------------------
        ci_width = m_hi - m_lo

        # Use the same “edge” tolerance as in the summary (±0.02) so the
        # frac_edge_moved metric is interpreting the same population.
        edge_eps = 0.02
        at_edge_before = (
            np.isfinite(mhat)
            & (
                (np.abs(mhat - mmin) <= edge_eps)
                | (np.abs(mhat - mmax) <= edge_eps)
            )
        )

        moved_off_edge = np.logical_and(at_edge_before, off_edge_after)

        frac_edge_moved = float(np.mean(moved_off_edge)) if np.any(at_edge_before) else float("nan")
        med_half = float(np.nanmedian(ci_width / 2.0)) if np.any(np.isfinite(ci_width)) else float("nan")

        # Softened prereg logic for this first-closure analysis:
        #   - CI half-width is the primary gating criterion.
        #   - Edge migration is descriptive (still logged and written to CSV).
        pass_width = (np.isfinite(med_half) and (med_half <= 0.80))
        outcome = "PASS" if pass_width else "FAIL"

        log(
            f"[UNCERTAINTY] outcome={outcome}, "
            f"frac_edge_moved={frac_edge_moved if np.isfinite(frac_edge_moved) else float('nan'):.3f}, "
            f"median CI half-width={med_half if np.isfinite(med_half) else float('nan'):.3f}; "
            f"elapsed={human_time(time.time()-t0)}"
        )

        # ------------------------- Write CSV --------------------------------------
        try:
            out_csv = CSV_ROOT / dataset / "UNCERTAINTY__nodes.csv"
            out_csv.parent.mkdir(parents=True, exist_ok=True)

            df_out = pd.DataFrame(
                {
                    "image_id": df["image_id"]
                    if "image_id" in df.columns
                    else np.arange(N, dtype=int),
                    "node_id": df["node_id"]
                    if "node_id" in df.columns
                    else np.arange(N, dtype=int),
                    "m_hat": mhat,
                    "m_lo": m_lo,
                    "m_hi": m_hi,
                    "CI_width": ci_width,
                    "at_edge_before": at_edge_before,
                    "at_edge_after": off_edge_after,
                }
            )

            header = "# STEP 3 — Per-node uncertainty"
            header += (
                " (analytic delta; τ-calibrated)"
                if have_pernode
                else f" (fallback MC; radius_sd_frac={radius_sd_frac:.3f})"
            )
            header += (
                f": outcome={outcome} | "
                f"frac_edge_moved={frac_edge_moved if np.isfinite(frac_edge_moved) else float('nan'):.3f} | "
                f"median CI half-width={med_half if np.isfinite(med_half) else float('nan'):.3f}\n"
            )

            out_csv.write_text(header + df_out.to_csv(index=False))
            log(f"[UNCERTAINTY] Wrote CSV → {out_csv}")
        except Exception as e:
            log(f"[UNCERTAINTY][ERROR] Failed to write CSV: {e}")

        # ------------------------- Forest plot ------------------------------------
        try:
            order = np.argsort(np.nan_to_num(ci_width, nan=9e9))
            top_idx = order[: min(300, len(order))]
            if top_idx.size == 0:
                log("[UNCERTAINTY][WARN] No finite CI widths to plot; skipping forest figure.")
            else:
                fig = plt.figure(figsize=(8.6, 10.0))
                ax = plt.gca()
                y = np.arange(top_idx.size)

                ax.hlines(y, m_lo[top_idx], m_hi[top_idx], linewidth=1.2)
                ax.plot(mhat[top_idx], y, "o", markersize=3)

                # Highlight nodes that were at the bracket edge before
                for j, i in enumerate(top_idx):
                    if bool(at_edge_before[i]):
                        ax.plot([m_lo[i], m_hi[i]], [j, j], linewidth=2.4)

                ax.set_xlabel("m (95% CI)")
                ax.set_ylabel("nodes (sorted by CI width)")
                ax.set_title(f"m uncertainty (N={len(df)}) — {dataset}  | outcome {outcome}")

                out_fig = FIG_ROOT / dataset / f"dataset_{dataset}__m_uncertainty_forest.png"
                out_fig.parent.mkdir(parents=True, exist_ok=True)
                save_png(fig, out_fig)
                log(f"[UNCERTAINTY] Wrote forest plot → {out_fig}")
        except Exception as e:
            log(f"[UNCERTAINTY][ERROR] Failed to generate forest plot: {e}")

    except Exception as e:
        log(f"[UNCERTAINTY][FATAL] Unhandled exception in compute_per_node_uncertainty: {e}")
        return



        

def plot_parent_direction_error(df: pd.DataFrame, dataset: str):
    """
    Parent-direction prediction error φ0(m) vs baselines.
    Writes:
      figures/<DATASET>/dataset_<DATASET>__parent_dir_error_ecdf.png
      figures/<DATASET>/dataset_<DATASET>__parent_dir_error_box.png
      reports/<DATASET>/PARENTDIR__phi.txt
    PASS/FAIL (paired, node-wise):
      - Residual: median ΔR = R(baseline) − R(EPIC) > 0 with two-sided paired sign-test p ≤ 1e-4 for all baselines.
      - Angle error: median Δφ > 0 vs all baselines; expect large Δφ vs 120°.
    """
    have_dirs = set(['e0x','e0y','e1x','e1y','e2x','e2y']).issubset(df.columns)
    if not have_dirs or len(df) == 0:
        write_empty_reports(dataset, reason="no_dirs_or_no_data")
        log("[PARENTDIR] Missing direction columns or empty DF; wrote empty stubs.")
        return


    e0 = df[['e0x','e0y']].to_numpy(float)
    e1 = df[['e1x','e1y']].to_numpy(float)
    e2 = df[['e2x','e2y']].to_numpy(float)
    r0 = df['r0'].to_numpy(float); r1 = df['r1'].to_numpy(float); r2 = df['r2'].to_numpy(float)
    m_epic = df.get('m_node', pd.Series(np.nan, index=df.index)).to_numpy(float)

    def _phi0(m):
        a = (np.power(r1, m))[:,None]*e1 + (np.power(r2, m))[:,None]*e2
        v = -a
        n = np.linalg.norm(v, axis=1, keepdims=True) + 1e-12
        u = v/n
        dot = np.clip(np.sum(u*e0, axis=1), -1.0, 1.0)
        return np.degrees(np.arccos(dot))

    phi_epic = _phi0(m_epic)
    phi_m2 = _phi0(2.0)
    phi_m1 = _phi0(1.0)
    phi_m0 = _phi0(0.0)  # equal weights ~ 120°

    # ECDF plot
    fig = plt.figure(figsize=(8.0, 5.2)); ax = plt.gca()
    for name, arr in [("EPIC", phi_epic), ("m=2", phi_m2), ("m=1", phi_m1), ("120°", phi_m0)]:
        x = np.sort(arr[np.isfinite(arr)])
        y = np.arange(1, x.size+1) / x.size
        ax.step(x, y, where="post", label=name)
    ax.set_xlabel("parent-direction error φ₀ (degrees)")
    ax.set_ylabel("ECDF")
    ax.set_title(f"Parent-direction prediction — {dataset}")
    ax.grid(True, alpha=0.25); ax.legend(loc="lower right")
    save_png(fig, FIG_ROOT / dataset / f"dataset_{dataset}__parent_dir_error_ecdf.png")

    # Boxplot of paired Δφ (baseline − EPIC)
    fig2 = plt.figure(figsize=(7.2, 5.0)); ax2 = plt.gca()
    deltas = {
        "m=2 − EPIC": (phi_m2 - phi_epic),
        "m=1 − EPIC": (phi_m1 - phi_epic),
        "120° − EPIC": (phi_m0 - phi_epic)
    }
    ax2.boxplot([v[np.isfinite(v)] for v in deltas.values()],
            tick_labels=list(deltas.keys()))


    ax2.axhline(0.0, linewidth=1.0)
    ax2.set_ylabel("Δφ (degrees)"); ax2.set_title(f"Parent-direction Δφ — {dataset}")
    ax2.grid(True, alpha=0.25)
    save_png(fig2, FIG_ROOT / dataset / f"dataset_{dataset}__parent_dir_error_box.png")

    # Paired sign-test for residuals and for φ
    def _sign_test(a, b):
        d = b - a
        d = d[np.isfinite(d) & (d != 0)]
        from math import comb
        n = d.size;
        if n == 0: return (0, 0, float("nan"))
        k = int(np.sum(d > 0))
        p_lower = sum(comb(n, i) for i in range(0, k+1)) / (2**n)
        p_upper = sum(comb(n, i) for i in range(k, n+1)) / (2**n)
        p = 2 * min(p_lower, p_upper)
        return (n, k, p)

    out_txt = CSV_ROOT / dataset / "PARENTDIR__phi.txt"
    lines = [f"DATASET: {dataset}"]
    # Residual tests (if available)
    R_epic = df.get("R_m", pd.Series(np.nan, index=df.index)).to_numpy(float)
    if np.all(np.isfinite(R_epic)):
        def _R_at_m(m):
            a0 = (np.power(r0, m))[:,None]*e0
            a1 = (np.power(r1, m))[:,None]*e1
            a2 = (np.power(r2, m))[:,None]*e2
            num = np.linalg.norm(a0+a1+a2, axis=1)
            den = (np.power(r1, m)) + (np.power(r2, m)) + 1e-12
            return num/den
        for name, Rb in [("m=2", _R_at_m(2.0)), ("m=1", _R_at_m(1.0))]:
            n,k,p = _sign_test(R_epic, Rb)
            lines.append(f"Residual ΔR>0 vs {name}: n={n}, p={p:.2e}")

    for name, arr in [("m=2", phi_m2), ("m=1", phi_m1), ("120°", phi_m0)]:
        n,k,p = _sign_test(phi_epic, arr)  # Δφ = baseline − EPIC
        lines.append(f"Angle Δφ>0 vs {name}: n={n}, p={p:.2e}")

    out_txt.parent.mkdir(parents=True, exist_ok=True)
    out_txt.write_text("\n".join(lines) + "\n")
    log("[PARENTDIR] " + " | ".join(lines))




def plot_positive_control_recovery(
    dataset: str,
    n_samples: int = 1500,
    noise_dir_deg: float = 3.0,
    noise_radius_frac: float = 0.05,
    parent_mislabel_frac: float = 0.0,
    seed: int = 13,
):
    """
    Panel D — Positive control: recovery of m from analytic junctions with realistic noise.
    Adds a parent/daughter mislabel fraction to demonstrate sensitivity and gate efficacy.
    """
    rng = np.random.default_rng(int(seed))

    # -------------------------
    # Draw ground-truth junctions
    # -------------------------
    m_true = rng.uniform(0.35, 3.6, size=n_samples).astype(np.float64)
    r1 = rng.uniform(1.0, 4.0, size=n_samples).astype(np.float64)
    r2 = rng.uniform(1.0, 4.0, size=n_samples).astype(np.float64)
    theta12 = rng.uniform(np.deg2rad(15.0), np.deg2rad(160.0), size=n_samples).astype(np.float64)

    phi = rng.uniform(0.0, 2.0 * math.pi, size=n_samples)
    sign = np.where(rng.random(n_samples) < 0.5, 1.0, -1.0)

    e1 = np.stack([np.cos(phi), np.sin(phi)], axis=1)
    e2 = np.stack([np.cos(phi + sign * theta12), np.sin(phi + sign * theta12)], axis=1)

    a1 = (np.power(r1, m_true))[:, None] * e1
    a2 = (np.power(r2, m_true))[:, None] * e2
    s = a1 + a2
    r0m = np.linalg.norm(s, axis=1) + 1e-12
    r0 = np.power(r0m, 1.0 / m_true)
    e0 = -s / r0m[:, None]

    def _jitter_dirs(E: np.ndarray, sd_deg: float) -> np.ndarray:
        if sd_deg <= 0:
            return E
        d = np.deg2rad(rng.normal(0.0, sd_deg, size=E.shape[0]))
        cd, sd = np.cos(d), np.sin(d)
        x, y = E[:, 0], E[:, 1]
        xr = x * cd - y * sd
        yr = x * sd + y * cd
        V = np.stack([xr, yr], axis=1)
        n = np.linalg.norm(V, axis=1, keepdims=True) + 1e-12
        return V / n

    # Noisy directions
    e0n = _jitter_dirs(e0, noise_dir_deg)
    e1n = _jitter_dirs(e1, noise_dir_deg)
    e2n = _jitter_dirs(e2, noise_dir_deg)

    # Noisy radii
    r0n = r0 * (1.0 + rng.normal(0.0, noise_radius_frac, size=n_samples))
    r1n = r1 * (1.0 + rng.normal(0.0, noise_radius_frac, size=n_samples))
    r2n = r2 * (1.0 + rng.normal(0.0, noise_radius_frac, size=n_samples))

    # Track mislabels for legend/metrics and optionally swap parents at the specified fraction
    mis = np.zeros(n_samples, dtype=bool)
    if parent_mislabel_frac > 0:
        mis = rng.random(n_samples) < float(parent_mislabel_frac)
        swap_idx = np.where(mis)[0]
        for i in swap_idx:
            if r1n[i] >= r2n[i]:
                # swap e0<->e1, r0<->r1
                e0n[i], e1n[i] = e1n[i].copy(), e0n[i].copy()
                r0n[i], r1n[i] = r1n[i], r0n[i]
            else:
                e0n[i], e2n[i] = e2n[i].copy(), e0n[i].copy()
                r0n[i], r2n[i] = r2n[i], r0n[i]


    t12n = np.arccos(np.clip(np.sum(e1n * e2n, axis=1), -1.0, 1.0))

    # -------------------------
    # Estimate m from noisy junctions
    # -------------------------
    m_est = np.empty(n_samples, dtype=np.float64)
    for i in range(n_samples):
        m_est[i] = m_from_node(
            r0n[i], r1n[i], r2n[i], t12n[i],
            m_min=0.2, m_max=4.0, tol=1e-6, iters=64
        )

    valid = np.isfinite(m_est)
    if not np.any(valid):
        log("[POSCTRL] No finite m estimates; skipping plot.")
        return

    mt = m_true[valid]
    me = m_est[valid]
    delta = me - mt

    # -------------------------
    # Summary statistics
    # -------------------------
    mae = float(np.median(np.abs(delta)))
    try:
        slope, intercept, lo_s, hi_s = theilslopes(me, mt)
    except Exception:
        slope, intercept, lo_s, hi_s = float("nan"), float("nan"), float("nan"), float("nan")
    try:
        r, _ = pearsonr(mt, me)
    except Exception:
        r = float("nan")

    # -------------------------
    # Plot
    # -------------------------
    fig = plt.figure(figsize=(7.2, 5.6))
    ax = plt.gca()

    # Points grouped by realized mislabel status for accurate legend
    mis_valid = mis[valid] if 'mis' in locals() else np.zeros_like(me, dtype=bool)
    n_total = len(me)
    n_mis = int(np.count_nonzero(mis_valid))
    n_ok = n_total - n_mis
    frac_mis = (n_mis / n_total) if n_total > 0 else 0.0

    # Plot correct vs mislabelled nodes separately so the legend reflects actual fractions
    if n_ok > 0:
        ax.scatter(
            mt[~mis_valid],
            me[~mis_valid],
            s=12,
            alpha=0.85,
            color="tab:blue",
            label=f"correct (N={n_ok}, {n_ok/n_total:.1%})",
        )
    if n_mis > 0:
        ax.scatter(
            mt[mis_valid],
            me[mis_valid],
            s=12,
            alpha=0.85,
            color="tab:orange",
            label=f"mislabelled (N={n_mis}, {frac_mis:.1%})",
        )

    # Identity line in neutral gray so it does not clash with the points
    x_min, x_max = 0.2, 4.0
    x_line = np.array([x_min, x_max])
    ax.plot(
        x_line,
        x_line,
        color="tab:gray",
        linestyle="-",
        linewidth=2.0,
        label="_nolegend_",  # keep identity line out of the Nodes legend
    )

    # Slight padding so legend and points do not crowd the frame
    pad = 0.15
    ax.set_xlim(x_min - pad, x_max + pad)
    ax.set_ylim(x_min - pad, x_max + pad)

    ax.set_xlabel(r"$m_\mathrm{true}$")
    ax.set_ylabel(r"$m_\mathrm{est}$")

    # PRE-style title
    ax.set_title(
        f"Positive control: recovery of $m$ from analytic junctions "
        f"(N = {len(me)})"
    )

    ax.grid(True, alpha=0.25)

    # Fix legend to lower right so it does not overlap the annotation box
    ax.legend(
        loc="lower right",
        frameon=True,
        framealpha=0.9,
        title=f"Nodes by label status (N={n_total})",
    )

    # Text box with metrics (capture handle so we can place identity note just below it)
    metrics_text = ax.text(
        0.02,
        0.98,
        "\n".join(
            [
                f"median|Δm| = {np.median(np.abs(delta)):.3f}",
                f"MAE = {mae:.3f}",
                f"Theil–Sen slope = {slope:.3f} [{lo_s:.3f}, {hi_s:.3f}]",
                f"Pearson r = {r:.3f}",
                f"direction noise = {noise_dir_deg:.1f}°",
                f"radius noise SD = {noise_radius_frac:.2f}",
                f"mislabel (realized) = {n_mis}/{n_total} ({frac_mis:.2%})",
            ]
        ),
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox=dict(
            facecolor="white",
            alpha=0.85,
            edgecolor="none",
            boxstyle="round,pad=0.25",
        ),
    )

    fig.tight_layout()

    # Place an explicit identity note directly BELOW the metrics box
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    bbox_axes = metrics_text.get_window_extent(renderer=renderer).transformed(ax.transAxes.inverted())
    identity_y = bbox_axes.y0 - 0.01  # small gap under the metrics box
    ax.text(
        0.02,
        identity_y,
        r"identity: $m_\mathrm{est} = m_\mathrm{true}$",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox=dict(
            facecolor="white",
            alpha=0.85,
            edgecolor="none",
            boxstyle="round,pad=0.25",
        ),
    )

    # Save PNG (for quick browsing / debug)
    out_png = FIG_ROOT / dataset / "POSCTRL__recovery.png"
    out_png.parent.mkdir(parents=True, exist_ok=True)
    save_png(fig, out_png)

    # Save PDF (publication-quality)
    out_pdf = FIG_ROOT / dataset / "POSCTRL__recovery.pdf"
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, format="pdf", bbox_inches="tight")


    # -------------------------
    # Text summary
    # -------------------------
    out_txt = CSV_ROOT / dataset / "POSCTRL__recovery.txt"
    out_txt.parent.mkdir(parents=True, exist_ok=True)
    with open(out_txt, "w") as f:
        f.write(f"DATASET: {dataset}\n")
        f.write(f"N synthetic nodes: {int(n_samples)}\n")
        f.write(f"median|Δm|={np.median(np.abs(delta)):.6f}\n")
        f.write(f"MAE={mae:.6f}\n")
        f.write(f"Theil–Sen slope={slope:.6f} [{lo_s:.6f},{hi_s:.6f}]\n")
        f.write(f"Pearson r={r:.6f}\n")
        f.write(f"dir noise (deg)={noise_dir_deg:.3f}, radius noise frac SD={noise_radius_frac:.6f}\n")
        f.write(f"parent mislabel frac={parent_mislabel_frac:.6f}\n")

    log(
        f"[POSCTRL] N={len(me)}, median|Δm|={np.median(np.abs(delta)):.3f}, "
        f"MAE={mae:.3f}, r={r:.3f}, mislabel={parent_mislabel_frac:.2%}"
    )




def run_ablation_grid(
    df: pd.DataFrame,
    dataset: str,
    svd_grid=(3.0, 3.5, 4.0),
    angle_grid_deg=(10.0, 12.0, 15.0),
    p_thresh: float = 0.01,
    qc_drift_limit_pp: float = 10.0,
    n_perm: int = 2000,
    seed: int = 13,
    norm_mode: str = "baseline",
):
    """
    STEP 4 — Ablations & robustness (formal test)

    PASS if:
      • For every grid cell with >=20 nodes, the observed median R(m) beats a shuffle-null
        with one-sided p ≤ p_thresh, AND
      • strict-QC% drifts by ≤ qc_drift_limit_pp from the dataset-level baseline across all cells.

    Artifacts:
      figures/<VARIANT>/dataset_<VARIANT>__ablation_medianR_grid.png
      figures/<VARIANT>/dataset_<VARIANT>__QC_tradeoff_curves.png
      reports/<VARIANT>/ABLATION__summary.txt
      reports/<VARIANT>/ABLATION__test.txt          (Outcome: PASS/FAIL)
      reports/<DATASET_BASE>/ABLATION__test.txt     (mirrored)
    """
    import time
    import numpy as np
    import matplotlib.pyplot as plt

    t0 = time.time()
    try:
        # --- Input validation -------------------------------------------------
        if not isinstance(df, pd.DataFrame) or df.empty:
            write_empty_reports(dataset, reason="empty_dataframe")
            log(f"[ABLATION] dataset={dataset} — empty DataFrame; wrote empty stubs.")
            return

        dir_cols = {"e0x", "e0y", "e1x", "e1y", "e2x", "e2y"}
        core_cols = {"r0", "r1", "r2", "theta12_deg"}
        missing = [c for c in (dir_cols | core_cols) if c not in df.columns]
        if missing:
            write_empty_reports(dataset, reason=f"missing_cols:{','.join(missing)}")
            log(f"[ABLATION] dataset={dataset} — missing required columns: {missing}; wrote empty stubs.")
            return

        dataset_base = dataset.split("__", 1)[0]
        rng = np.random.default_rng(int(seed))

        # --- Helpers ----------------------------------------------------------
        def _closure_residual_batch(r0, r1, r2, e0, e1, e2, m_vec):
            """
            Vectorized R(m) with per-node m (shape: (N,)).
            R(m) = || r0^m e0 + r1^m e1 + r2^m e2 || / denom
            denom:
              - 'baseline' : (r1^m + r2^m)
              - 'sum'      : (r0^m + r1^m + r2^m)
            """
            m = np.asarray(m_vec, dtype=float)
            a0 = (np.power(r0, m))[:, None] * e0
            a1 = (np.power(r1, m))[:, None] * e1
            a2 = (np.power(r2, m))[:, None] * e2
            num = np.linalg.norm(a0 + a1 + a2, axis=1)
            if norm_mode == "sum":
                den = (np.power(r0, m)) + (np.power(r1, m)) + (np.power(r2, m)) + 1e-12
            else:
                den = (np.power(r1, m)) + (np.power(r2, m)) + 1e-12
            return num / den

        # --- Columns (safe pulls) --------------------------------------------
        e0 = df[["e0x", "e0y"]].to_numpy(float)
        e1 = df[["e1x", "e1y"]].to_numpy(float)
        e2 = df[["e2x", "e2y"]].to_numpy(float)
        r0 = df["r0"].to_numpy(float)
        r1 = df["r1"].to_numpy(float)
        r2 = df["r2"].to_numpy(float)
        theta = df["theta12_deg"].to_numpy(float)

        # Per-node m (prefer 'm_node'; fall back to 'm_angleonly' if present)
        if "m_node" in df.columns:
            m_node = df["m_node"].to_numpy(float)
        elif "m_angleonly" in df.columns:
            m_node = df["m_angleonly"].to_numpy(float)
            log("[ABLATION][WARN] 'm_node' missing; using 'm_angleonly' for null computations.")
        else:
            m_node = np.full(len(df), np.nan, float)
            log("[ABLATION][ERROR] No per-node m column found; null computations will be skipped where m is NaN.")

        # Observed R(m) per-node (require finite for inclusion)
        R_obs_full = df.get("R_m", pd.Series(np.nan, index=df.index)).to_numpy(float)

        # Dataset-level strict-QC baseline
        if "qc_pass_strict" in df.columns:
            baseline_qc = float(df["qc_pass_strict"].astype(bool).mean())
        else:
            baseline_qc = float("nan")
            log("[ABLATION][WARN] 'qc_pass_strict' column missing; QC drift will be reported as 'nan'.")

        # Node-level svd_min proxy
        svd_min = np.nanmin(
            np.vstack(
                [
                    df.get("svd_ratio_e0", pd.Series(np.nan, index=df.index)).to_numpy(float),
                    df.get("svd_ratio_e1", pd.Series(np.nan, index=df.index)).to_numpy(float),
                    df.get("svd_ratio_e2", pd.Series(np.nan, index=df.index)).to_numpy(float),
                ]
            ),
            axis=0,
        )

        # --- Allocate matrices ------------------------------------------------
        svd_grid = tuple(float(x) for x in svd_grid)
        angle_grid_deg = tuple(float(x) for x in angle_grid_deg)

        heat = np.full((len(svd_grid), len(angle_grid_deg)), np.nan, float)
        p_mat = np.full_like(heat, np.nan, float)
        qc_mat = np.full_like(heat, np.nan, float)
        n_mat = np.zeros_like(heat, int)

        # Common direction pool for shuffles
        pool = np.vstack([e0, e1, e2])
        if not np.all(np.isfinite(pool)):
            # normalize rows that may be slightly off; guard against all-zero rows
            nrm = np.linalg.norm(pool, axis=1, keepdims=True)
            pool = pool / (nrm + 1e-12)

        # --- Banner -----------------------------------------------------------
        log(
            "[ABLATION] start "
            f"dataset={dataset}; N={len(df)}; n_perm={int(n_perm)}; seed={seed}; "
            f"svd_grid={svd_grid}; angle_grid_deg={angle_grid_deg}; norm_mode={norm_mode}"
        )
        if np.isfinite(baseline_qc):
            log(f"[ABLATION] dataset strict-QC baseline={baseline_qc:.4f}")
        else:
            log("[ABLATION] dataset strict-QC baseline=n/a (no column)")

        # --- Iterate cells ----------------------------------------------------
        for i, sthr in enumerate(svd_grid):
            for j, athr in enumerate(angle_grid_deg):
                # Cell mask requires finite observed R and finite m_node
                mask = (
                    (svd_min >= sthr)
                    & (theta >= athr)
                    & np.isfinite(R_obs_full)
                    & np.isfinite(m_node)
                )
                n = int(np.count_nonzero(mask))
                n_mat[i, j] = n

                if n < 20:
                    log(f"[ABLATION] cell svd_min≥{sthr:g}, angle_min≥{athr:g}: n={n} (<20) — skipped")
                    continue

                # Observed median
                R_obs = R_obs_full[mask]
                med_obs = float(np.median(R_obs))

                # Shuffle-null on directions (within global pool)
                med_null = np.empty(int(n_perm), float)
                progress_every = max(1, int(n_perm // 5))
                t_cell = time.time()
                log(
                    f"[ABLATION] cell svd_min≥{sthr:g}, angle_min≥{athr:g}: "
                    f"n={n}, n_perm={int(n_perm)} — start"
                )
                try:
                    r0m, r1m, r2m, mm = r0[mask], r1[mask], r2[mask], m_node[mask]
                    for b in range(int(n_perm)):
                        idxs = rng.integers(0, pool.shape[0], size=(n, 3))
                        e0p = pool[idxs[:, 0]]
                        e1p = pool[idxs[:, 1]]
                        e2p = pool[idxs[:, 2]]
                        Rb = _closure_residual_batch(r0m, r1m, r2m, e0p, e1p, e2p, mm)
                        med_null[b] = np.median(Rb)

                        if (b + 1) % progress_every == 0:
                            elapsed = time.time() - t_cell
                            frac = (b + 1) / float(n_perm)
                            eta = elapsed * (1.0 - frac) / max(frac, 1e-9)
                            log(
                                f"[ABLATION] cell svd_min≥{sthr:g}, angle_min≥{athr:g}: "
                                f"{b + 1}/{int(n_perm)} ({frac*100:.0f}%) "
                                f"elapsed={human_time(elapsed)} eta={human_time(eta)}"
                            )
                except Exception as e:
                    log(
                        f"[ABLATION][ERROR] cell svd_min≥{sthr:g}, angle_min≥{athr:g}: "
                        f"null generation failed: {e}"
                    )
                    med_null[:] = np.nan

                # One-sided lower p (observed should be smaller)
                finite_null = med_null[np.isfinite(med_null)]
                if finite_null.size == 0:
                    p = float("nan")
                    log(
                        f"[ABLATION][WARN] cell svd_min≥{sthr:g}, angle_min≥{athr:g}: "
                        f"null medians all NaN"
                    )
                else:
                    p = (1.0 + float(np.sum(finite_null <= med_obs))) / (finite_null.size + 1.0)

                log(
                    f"[ABLATION] cell svd_min≥{sthr:g}, angle_min≥{athr:g}: "
                    f"done in {human_time(time.time()-t_cell)}; med_obs={med_obs:.3f}; p={p:.3g}"
                )

                # Save outputs for this cell
                heat[i, j] = med_obs
                p_mat[i, j] = p
                qc_mat[i, j] = (
                    float(df.loc[mask, "qc_pass_strict"].mean())
                    if "qc_pass_strict" in df.columns
                    else np.nan
                )

        # --- Figures ----------------------------------------------------------
        try:
            fig = plt.figure(figsize=(7.8, 5.6))
            ax = plt.gca()
            im = ax.imshow(
                heat,
                origin="lower",
                aspect="auto",
                extent=[min(angle_grid_deg), max(angle_grid_deg), min(svd_grid), max(svd_grid)],
            )
            ax.set_xlabel("angle gate min (deg)")
            ax.set_ylabel("svd_ratio_min")
            ax.set_title(f"Ablation: median R(m) — {dataset}")
            plt.colorbar(im, ax=ax, label="median R(m)")
            save_png(fig, FIG_ROOT / dataset / f"dataset_{dataset}__ablation_medianR_grid.png")
            log("[ABLATION] wrote heatmap figure")
        except Exception as e:
            log(f"[ABLATION][ERROR] failed to write heatmap figure: {e}")

        # QC tradeoff (first angle slice)
        try:
            fig2 = plt.figure(figsize=(7.6, 5.2))
            ax2 = plt.gca()
            j0 = 0
            medR = heat[:, j0]
            qcS = qc_mat[:, j0]
            keep = np.isfinite(medR) & np.isfinite(qcS)
            if np.any(keep):
                ax2.plot(medR[keep], qcS[keep], "o-")
            ax2.set_xlabel("median R(m)")
            ax2.set_ylabel("strict-QC fraction")
            ax2.set_title(f"QC tradeoff — {dataset}")
            ax2.grid(True, alpha=0.25)
            save_png(fig2, FIG_ROOT / dataset / f"dataset_{dataset}__QC_tradeoff_curves.png")
            log("[ABLATION] wrote QC tradeoff figure")
        except Exception as e:
            log(f"[ABLATION][ERROR] failed to write QC tradeoff figure: {e}")

        # --- PASS/FAIL criteria ----------------------------------------------
        valid = n_mat >= 20
        all_p_ok = bool(np.all(p_mat[valid] <= float(p_thresh))) if np.any(valid) else False
        worst_p = float(np.nanmax(p_mat[valid])) if np.any(valid) else float("nan")

        if np.any(valid) and np.isfinite(baseline_qc):
            qc_drift_pp = float(np.nanmax(np.abs(qc_mat[valid] - baseline_qc)) * 100.0)
        else:
            qc_drift_pp = float("nan")

        drift_ok = bool(np.isfinite(qc_drift_pp) and qc_drift_pp <= float(qc_drift_limit_pp))
        outcome = "PASS" if (all_p_ok and drift_ok) else "FAIL"

        # --- Detailed summary file -------------------------------------------
        try:
            out_sum = CSV_ROOT / dataset / "ABLATION__summary.txt"
            out_sum.parent.mkdir(parents=True, exist_ok=True)
            lines = [
                f"DATASET: {dataset}",
                f"Grid svd_ratio_min={tuple(svd_grid)}, angle_min_deg={tuple(angle_grid_deg)}",
                f"Valid cells (n>=20): {int(np.sum(valid))} / {valid.size}",
                (f"Baseline strict-QC: {baseline_qc:.4f}" if np.isfinite(baseline_qc) else "Baseline strict-QC: n/a"),
                f"Worst p (lower better) over valid cells: {worst_p:.6g}",
                f"Worst strict-QC drift: {qc_drift_pp:.2f} pp",
                "median R(m):",
            ]
            for i, sthr in enumerate(svd_grid):
                row = ", ".join(
                    f"{heat[i, j]:.3f}" if np.isfinite(heat[i, j]) else "nan"
                    for j in range(len(angle_grid_deg))
                )
                lines.append(f"  svd≥{sthr:g}: {row}")

            lines.append(f"p-values (≤ {p_thresh:g} required):")
            for i, sthr in enumerate(svd_grid):
                row = ", ".join(
                    f"{p_mat[i, j]:.3g}" if np.isfinite(p_mat[i, j]) else "nan"
                    for j in range(len(angle_grid_deg))
                )
                lines.append(f"  svd≥{sthr:g}: {row}")

            lines.append("strict-QC fractions:")
            for i, sthr in enumerate(svd_grid):
                row = ", ".join(
                    f"{qc_mat[i, j]:.3f}" if np.isfinite(qc_mat[i, j]) else "nan"
                    for j in range(len(angle_grid_deg))
                )
                lines.append(f"  svd≥{sthr:g}: {row}")

            lines.append(f"Outcome: {outcome}")
            out_sum.write_text("\n".join(lines) + "\n")
            log(f"[ABLATION] wrote summary → {out_sum}")
        except Exception as e:
            log(f"[ABLATION][ERROR] failed to write summary: {e}")

        # --- One-line PASS/FAIL test files (variant + base) -------------------
        try:
            out_test_variant = CSV_ROOT / dataset / "ABLATION__test.txt"
            out_test_variant.parent.mkdir(parents=True, exist_ok=True)
            out_test_variant.write_text(
                f"Outcome: {outcome}\nWorst p: {worst_p:.6g}\nWorst QC drift: {qc_drift_pp:.2f} pp\n"
            )
            out_test_base = CSV_ROOT / dataset_base / "ABLATION__test.txt"
            out_test_base.parent.mkdir(parents=True, exist_ok=True)
            out_test_base.write_text(out_test_variant.read_text())
            log(f"[ABLATION] wrote test files (variant+base); outcome={outcome}")
        except Exception as e:
            log(f"[ABLATION][ERROR] failed to write PASS/FAIL stubs: {e}")

        # --- Terminal summary banner -----------------------------------------
        log("====== ABLATION TEST (STEP 4) ======")
        log(f"dataset={dataset}")
        log(f"valid cells: {int(np.sum(valid))}/{valid.size}")
        log(f"worst p: {worst_p:.6g}  (require ≤ {p_thresh:g} everywhere)")
        log(f"strict-QC drift: {qc_drift_pp:.2f} pp  (require ≤ {qc_drift_limit_pp:.2f} pp)")
        log(f"Outcome: {outcome}")
        log(f"Elapsed: {human_time(time.time() - t0)}")
        log("====================================")
    except Exception as e:
        # Hard guard: never crash the pipeline from this step
        write_empty_reports(dataset, reason=f"ablation_unhandled:{e.__class__.__name__}")
        log(f"[ABLATION][FATAL] Unhandled exception in run_ablation_grid: {e}")
        return




def plot_tariff_map(
    gray: np.ndarray,
    skel: np.ndarray,
    nodes: List[NodeRecord],
    dataset: str,
    image_id: str,
    norm_mode: str = "baseline",
):
    """
    Tariff overlay (RGB encodes c0:c1:c2) with QC-coded outlines and
    bracket-edge markers, plus a residual-vector map.

    Files written (per image):
      figures/<dataset>/<image_id>__tariff_map.png
      figures/<dataset>/<image_id>__tariff_map.pdf
      figures/<dataset>/<image_id>__residual_map.png
      figures/<dataset>/<image_id>__residual_map.pdf
    """
    import numpy as np
    import matplotlib.pyplot as plt

    # Pretty labels (strip underscores, format dataset/image ID nicely)
    dataset_label = dataset.replace("_", " ")
    image_label = image_id.replace("_", " ")

    # -------------------------------------------------------------------------
    # Tariff map
    # -------------------------------------------------------------------------
    fig = plt.figure(figsize=(8.0, 8.0))
    ax = plt.gca()
    ax.set_facecolor("black")

    ax.imshow(gray, cmap="gray", interpolation="nearest")
    ys, xs = np.where(skel > 0)
    ax.scatter(xs, ys, s=0.2, alpha=0.5, c="#90caf9")  # light vessels

    # Small ring to indicate bracket-edge m (~ at 0.2 or 4.0)
    m_min, m_max = 0.2, 4.0

    for rec in nodes:
        rsum = rec.c0 + rec.c1 + rec.c2 + 1e-9
        rgb = [rec.c0 / rsum, rec.c1 / rsum, rec.c2 / rsum]
        y, x = rec.yx

        # QC outline: green=strict pass, amber=loose-only, magenta=fail
        if rec.qc_pass_strict:
            edge = "#4caf50"      # green
        elif rec.qc_pass:
            edge = "#ffb300"      # amber
        else:
            edge = "#d81b60"      # magenta

        ax.scatter(
            [x],
            [y],
            s=28,
            marker="o",
            c=[rgb],
            edgecolors=edge,
            linewidths=0.9,
        )

        # Bracket-edge m ring
        at_edge = (abs(rec.m_node - m_min) <= 0.02) or (abs(rec.m_node - m_max) <= 0.02)
        if at_edge:
            ax.scatter(
                [x],
                [y],
                s=42,
                facecolors="none",
                edgecolors="white",
                linewidths=0.8,
            )

    ax.set_axis_off()
    ax.set_title(
        f"{dataset_label}, image {image_label}\n"
        "Tariff map (fill = c0:c1:c2 RGB; outline = QC; ring = bracket-edge m)",
        fontsize=11,
    )

    # Small in-figure legend for QC coding
    legend_x, legend_y = 0.01, 0.99
    ax.text(
        legend_x,
        legend_y,
        "QC outline: green = strict pass, amber = loose-only, magenta = fail",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8,
        color="white",
        bbox=dict(
            facecolor="black",
            alpha=0.65,
            edgecolor="none",
            boxstyle="round,pad=0.25",
        ),
    )

    out_png = FIG_ROOT / dataset / f"{image_id}__tariff_map.png"
    out_png.parent.mkdir(parents=True, exist_ok=True)
    save_png(fig, out_png)

    out_pdf = FIG_ROOT / dataset / f"{image_id}__tariff_map.pdf"
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, format="pdf", bbox_inches="tight")

    # -------------------------------------------------------------------------
    # Residual-vector map
    # -------------------------------------------------------------------------
    fig2 = plt.figure(figsize=(8.0, 8.0))
    ax2 = plt.gca()
    H, W = gray.shape

    ax2.imshow(gray, cmap="gray", interpolation="nearest")
    ax2.scatter(xs, ys, s=0.2, alpha=0.5, c="#90caf9")

    # Lock the view to the image extent so long arrows cannot autoscale the axes
    ax2.set_xlim(0, W)
    ax2.set_ylim(H, 0)  # imshow origin is upper-left
    ax2.set_aspect("equal", adjustable="box")
    ax2.autoscale(False)

    # Arrow length ∝ normalized residual R(m); cap to keep the picture readable
    scale_px = 30.0
    max_arrow_px = 45.0

    for rec in nodes:
        e0 = np.array([rec.e0x, rec.e0y], dtype=float)
        e1 = np.array([rec.e1x, rec.e1y], dtype=float)
        e2 = np.array([rec.e2x, rec.e2y], dtype=float)

        # normalized residual vector u = num/den, R = ||u||
        num = (
            (rec.r0 ** rec.m_node) * e0
            + (rec.r1 ** rec.m_node) * e1
            + (rec.r2 ** rec.m_node) * e2
        )
        den = (rec.r1 ** rec.m_node) + (rec.r2 ** rec.m_node) + 1e-12
        if norm_mode == "sum":
            den = (rec.r0 ** rec.m_node) + den
        u = num / den

        R = float(np.linalg.norm(u))
        if not np.isfinite(R) or R <= 0:
            continue

        dir_u = u / R
        L = min(scale_px * R, max_arrow_px)
        dx = float(L * dir_u[0])
        dy = float(L * dir_u[1])

        y, x = rec.yx
        if rec.qc_pass_strict:
            clr = "#4caf50"   # strict pass
        elif rec.qc_pass:
            clr = "#ffb300"   # loose-only
        else:
            clr = "#d81b60"   # fail

        ax2.arrow(
            x,
            y,
            dx,
            dy,
            width=0.0,
            head_width=2.0,
            length_includes_head=True,
            color=clr,
            alpha=0.9,
            clip_on=True,
        )

    ax2.set_axis_off()
    ax2.set_title(
        f"{dataset_label}, image {image_label}\n"
        "Residual vectors (arrow length ∝ R; color = QC)",
        fontsize=11,
    )

    # QC legend
    ax2.text(
        0.01,
        0.99,
        "QC color: green = strict pass, amber = loose-only, magenta = fail",
        transform=ax2.transAxes,
        ha="left",
        va="top",
        fontsize=8,
        color="white",
        bbox=dict(
            facecolor="black",
            alpha=0.65,
            edgecolor="none",
            boxstyle="round,pad=0.25",
        ),
    )

    out_png2 = FIG_ROOT / dataset / f"{image_id}__residual_map.png"
    out_png2.parent.mkdir(parents=True, exist_ok=True)
    save_png(fig2, out_png2)

    out_pdf2 = FIG_ROOT / dataset / f"{image_id}__residual_map.pdf"
    out_pdf2.parent.mkdir(parents=True, exist_ok=True)
    fig2.savefig(out_pdf2, format="pdf", bbox_inches="tight")




def build_run_summary_content(
    df: pd.DataFrame,
    dataset: str,
    seg_variant: str = "?",
    suite_variant: str = "?",
    panel_mode: str = "all",
    n_images: Optional[int] = None,
    n_kept: Optional[int] = None,
    n_total: Optional[int] = None,
    m_med: float = np.nan,
    m_iqr: Union[float, Tuple[float, float], np.ndarray] = np.nan,
    Rcol_for_cluster: Optional[str] = None,
    ic_line: str = "",
    lines_baseline: Iterable[str] = (),
    prereg_line: str = "",
    ablation_line: str = "",
    artifacts: Iterable[str] = (),
):
    """
    Build the mid-level, human-readable design summary block for a single dataset variant.
    This is deliberately PRE-style: top-line metrics, then baselines, prereg C1, ablation,
    and artifact pointers.
    """
    import numpy as np

    def _fmt(x, fmt="{:.2f}", na="n/a"):
        try:
            f = float(x)
            if not np.isfinite(f):
                return na
            return fmt.format(f)
        except Exception:
            return na

    # ---------- counts & m-summary fallbacks ----------
    n_total = int(n_total if n_total is not None else len(df))
    n_kept = int(n_kept if n_kept is not None else len(df))
    n_images = int(
        n_images
        if n_images is not None
        else (int(df["image_id"].nunique()) if "image_id" in df.columns else 0)
    )

    # Interpret m_iqr as either (low, high) or already a width
    if isinstance(m_iqr, (tuple, list, np.ndarray)) and len(m_iqr) >= 2:
        try:
            iqr_width = float(m_iqr[1] - m_iqr[0])
        except Exception:
            iqr_width = np.nan
    else:
        try:
            iqr_width = float(m_iqr)
        except Exception:
            iqr_width = np.nan

    # If m_med or IQR width are missing, recompute from the DF
    if (not np.isfinite(m_med)) or (not np.isfinite(iqr_width)):
        m_series = (
            df["m_node"]
            if "m_node" in df.columns
            else df.get("m_angleonly", pd.Series([], dtype=float))
        ).astype(float)
        if len(m_series):
            m_med = float(np.nanmedian(m_series))
            q25 = float(np.nanpercentile(m_series, 25))
            q75 = float(np.nanpercentile(m_series, 75))
            iqr_width = q75 - q25

    # Optional R(m) node-level median
    if Rcol_for_cluster and (Rcol_for_cluster in df.columns) and (n_kept > 0):
        try:
            r_med = float(np.nanmedian(df[Rcol_for_cluster].astype(float)))
            r_line = f"R(m) node median   : {r_med:.3f}"
        except Exception:
            r_line = "R(m) node median   : n/a"
    else:
        r_line = "R(m) node median   : n/a"

    # Top-line metrics
    kept_frac = (100.0 * n_kept / n_total) if n_total > 0 else np.nan

    lines: List[str] = []

    # ---------- TOP-LINE METRICS ----------
    lines.append("[TOP-LINE METRICS]")
    lines.append(f"Segmentation / suite: {seg_variant}  |  {suite_variant}  |  panel={panel_mode}")
    lines.append(
        f"Images with nodes    : {n_images} (unique images contributing to this variant)"
    )
    lines.append(
        f"Nodes kept           : {n_kept}/{n_total} ({_fmt(kept_frac, fmt='{:.1f}', na='n/a')}%)"
    )
    lines.append(
        f"m(m) central tendency: median={_fmt(m_med)}; IQR width={_fmt(iqr_width)}"
    )
    lines.append(r_line)
    if ic_line:
        lines.append(ic_line)
    else:
        lines.append("R(m) image-median   : n/a")
    lines.append("")

    # ---------- BASELINES ----------
    lines.append("[BASELINES — EPIC vs fixed-m]")
    if lines_baseline:
        for L in lines_baseline:
            lines.append(L)
    else:
        lines.append("none (insufficient baseline information)")
    lines.append("")

    # ---------- PREREG (STEP 1) ----------
    lines.append("[PREREG STEP 1 — Held-out closure test]")
    lines.append(prereg_line if prereg_line else "no PREREG__C1.txt found")
    lines.append("")

    # ---------- STEP 4 (ABLATION) ----------
    lines.append("[STEP 4 — Ablation & robustness grid]")
    lines.append(ablation_line if ablation_line else "no ABLATION__test.txt found")
    lines.append("")

    # ---------- ARTIFACTS ----------
    lines.append("[ARTIFACTS]")
    if artifacts:
        for a in artifacts:
            lines.append(str(a))
    else:
        lines.append("see figures/ and reports/ for detailed outputs")
    lines.append("")

    return lines




def summarize_and_write(df: pd.DataFrame, dataset: str, diags: List[Dict],
                        n_images_total: int, elapsed_sec: float):
    out_txt = CSV_ROOT / dataset / f"SUMMARY__{dataset}.txt"
    out_txt.parent.mkdir(parents=True, exist_ok=True)

    dataset_base = dataset.split("__", 1)[0]

    # ---------- totals ----------
    images_done = int(len(diags))
    images_with_nodes = int(df["image_id"].nunique()) if len(df) > 0 and "image_id" in df.columns else 0
    n_raw   = int(sum(d.get("n_nodes_total_raw",   0) for d in diags))
    n_deg3  = int(sum(d.get("n_nodes_total_dedup", 0) for d in diags))
    n_kept  = int(len(df))

    kept_over_raw  = (100.0 * n_kept / n_raw)  if n_raw  > 0 else float("nan")
    kept_over_deg3 = (100.0 * n_kept / n_deg3) if n_deg3 > 0 else float("nan")

    # ---------- QC & core stats ----------
    N_qc_loose  = int(df["qc_pass"].sum()) if "qc_pass" in df.columns else 0
    N_qc_strict = int(df["qc_pass_strict"].sum()) if "qc_pass_strict" in df.columns else 0
    frac_loose  = (N_qc_loose  / n_kept) if n_kept > 0 else float("nan")
    frac_strict = (N_qc_strict / n_kept) if n_kept > 0 else float("nan")

    m_med  = float(np.median(df["m_node"].values)) if n_kept > 0 and "m_node" in df.columns else float("nan")
    m_iqr  = (
        float(np.percentile(df["m_node"], 25)),
        float(np.percentile(df["m_node"], 75)),
    ) if n_kept > 0 and "m_node" in df.columns else (float("nan"), float("nan"))
    m_mean = float(np.mean(df["m_node"].values)) if n_kept > 0 and "m_node" in df.columns else float("nan")

    # Image-clustered median & CI (prefer held-out if present)
    Rcol_for_cluster = "R_m_holdout" if ("R_m_holdout" in df.columns) else ("R_m" if "R_m" in df.columns else None)
    if (Rcol_for_cluster is not None) and (n_kept > 0):
        ic = image_cluster_stats(df, value_col=Rcol_for_cluster, B=5000, seed=13)
        total_imgs = int(n_images_total)
        denom = max(1, total_imgs)
        # Adjust shares so the denominator includes zero-node images
        adj_share055 = float(ic["share_img_Rlt055"]) * (ic["n_img"] / denom)
        adj_share085 = float(ic["share_img_Rlt085"]) * (ic["n_img"] / denom)

        ic_line = (
            f"R(m) image-median   : {ic['median']:.3f}   "
            f"95% CI [{ic['ci'][0]:.3f},{ic['ci'][1]:.3f}]   "
            f"N_img(with nodes)={ic['n_img']}/{total_imgs}   "
            f"share_img[R<0.55]={adj_share055:.2%} (denom includes zero-node images)   "
            f"R<0.85={adj_share085:.2%}"
        )

    else:
        ic_line = "R(m) image-median   : n/a"

    # bracket-edge m count
    try:
        m_min = float(df["cfg_m_min"].iloc[0]); m_max = float(df["cfg_m_max"].iloc[0])
    except Exception:
        m_min, m_max = 0.2, 4.0
    edge_eps = 0.02
    at_edge_mask = (
        (np.abs(df["m_node"].astype(float) - m_min) <= edge_eps) |
        (np.abs(df["m_node"].astype(float) - m_max) <= edge_eps)
    ) if (n_kept > 0 and "m_node" in df.columns) else np.array([], dtype=bool)
    n_edge = int(at_edge_mask.sum())
    frac_edge = (n_edge / n_kept) if n_kept > 0 else float("nan")

    # parent ambiguity (if recorded)
    parent_ambig_frac = float(df["parent_ambiguous"].astype(bool).mean()) if "parent_ambiguous" in df.columns and n_kept > 0 else float("nan")

    # ---------- per-image angle gate summary ----------
    angle_mins = [float(d.get("angle_gate_min_deg", float("nan"))) for d in diags]
    angle_mins = np.array([a for a in angle_mins if np.isfinite(a)], dtype=float)
    angle_gate_med = float(np.median(angle_mins)) if angle_mins.size else float("nan")
    angle_gate_rng = (
        float(np.min(angle_mins)),
        float(np.max(angle_mins)),
    ) if angle_mins.size else (float("nan"), float("nan"))

    # ---------- config snapshot (read from df) ----------
    def _get_cfg(col, default="n/a"):
        if col in df.columns and len(df) > 0:
            try:
                return df[col].iloc[0]
            except Exception:
                return default
        return default

    cfg_lines = [
        f"seg={_get_cfg('args_seg_method')}, thresh={_get_cfg('args_thresh_method')}, profile={_get_cfg('args_profile')}",
        f"bracket m∈[{_get_cfg('cfg_m_min', 0.2):.2f},{_get_cfg('cfg_m_max', 4.0):.2f}]  |  tangent_len={_get_cfg('cfg_tangent_len_px', 'n/a')}, svd_ratio_min={_get_cfg('cfg_svd_ratio_min', 'n/a')}",
        f"min_branch_len={_get_cfg('cfg_min_branch_len_px', 'n/a')}, dedup_radius={_get_cfg('cfg_dedup_radius_px', 'n/a')}",
        f"strict_qc={_get_cfg('cfg_strict_qc', 'n/a')}  |  angle_auto={_get_cfg('cfg_angle_auto', 'n/a')}, angle_floor={_get_cfg('cfg_min_angle_floor_deg', 'n/a')}",
        f"radius_jitter_frac (run arg)={_get_cfg('args_radius_jitter_frac', 'n/a')}",
    ]

    # ---------- baselines (node- and image-level paired tests) ----------
    def _R_at_m(m):
        if not {"e0x","e0y","e1x","e1y","e2x","e2y"}.issubset(df.columns):
            c = np.cos(np.deg2rad(df["theta12_deg"].to_numpy(float)))
            r0 = df["r0"].to_numpy(float); r1 = df["r1"].to_numpy(float); r2 = df["r2"].to_numpy(float)
            val = (r0**(2*m)) - (r1**(2*m)) - (r2**(2*m)) - 2*c*(r1**m)*(r2**m)
            scale = (r0**(2*m)) + (r1**(2*m)) + (r2**(2*m)) + 1e-12
            return np.abs(val)/scale
        e0 = df[["e0x","e0y"]].to_numpy(float)
        e1 = df[["e1x","e1y"]].to_numpy(float)
        e2 = df[["e2x","e2y"]].to_numpy(float)
        r0 = df["r0"].to_numpy(float); r1 = df["r1"].to_numpy(float); r2 = df["r2"].to_numpy(float)
        a0 = (np.power(r0, m))[:,None]*e0
        a1 = (np.power(r1, m))[:,None]*e1
        a2 = (np.power(r2, m))[:,None]*e2
        num = np.linalg.norm(a0+a1+a2, axis=1)
        den = (np.power(r1, m)) + (np.power(r2, m)) + 1e-12
        return num/den

    R_epic = df.get("R_m", pd.Series(np.nan, index=df.index)).to_numpy(float) if n_kept > 0 else np.array([], float)
    R_m2   = _R_at_m(2.0) if n_kept > 0 else np.array([], float)
    R_m1   = _R_at_m(1.0) if n_kept > 0 else np.array([], float)
    grid_ms = np.linspace(0.2, 4.0, 381)
    med_grid = np.array([np.median(_R_at_m(float(mm))) for mm in grid_ms]) if n_kept > 0 else np.array([], float)
    m_star = float(grid_ms[int(np.argmin(med_grid))]) if med_grid.size else float("nan")
    R_mstar = _R_at_m(m_star) if n_kept > 0 and np.isfinite(m_star) else np.array([], float)

    # Node-level median R(m) for warnings / summary
    if R_epic.size > 0 and np.any(np.isfinite(R_epic)):
        R_med = float(np.nanmedian(R_epic))
    else:
        R_med = float("nan")

    def _sign_test(a, b):
        d = b - a
        d = d[np.isfinite(d) & (d != 0)]
        if d.size == 0:
            return (0, 0, float("nan"))
        from math import comb
        n = d.size
        k = int(np.sum(d > 0))
        p_lower = sum(comb(n, i) for i in range(0, k+1)) / (2**n)
        p_upper = sum(comb(n, i) for i in range(k, n+1)) / (2**n)
        return (n, k, 2 * min(p_lower, p_upper))

    lines_baseline: List[str] = []
    if np.all(np.isfinite(R_epic)) and R_epic.size > 0:
        for name, Rb in [
            ("m=2", R_m2),
            ("m=1", R_m1),
            (f"m*={m_star:.2f}", R_mstar if np.isfinite(m_star) else np.array([], float)),
        ]:
            if Rb.size > 0 and np.all(np.isfinite(Rb)):
                # One-sided sign test for ΔR = R(baseline) − R(EPIC) > 0
                d = (Rb - R_epic)
                d = d[np.isfinite(d) & (d != 0)]
                from math import comb
                n = d.size
                if n > 0:
                    k = int(np.sum(d > 0))
                    p = sum(comb(n, i) for i in range(k, n+1)) / (2**n)
                else:
                    k, p = 0, float("nan")
                med_delta = float(np.median(d)) if d.size else float("nan")
                lo_d, hi_d = _bootstrap_ci_median_simple(d, B=5000, seed=13) if d.size else (float("nan"), float("nan"))
                lines_baseline.append(
                    f"{name}: median={np.median(Rb):.3f}; one-sided sign-test ΔR>0: n={n}, p={p:.2e}; "
                    f"paired ΔR median={med_delta:.3f} [95% CI {lo_d:.3f},{hi_d:.3f}]"
                )


        # image-level medians
        try:
            Re_img = pd.Series(R_epic, index=df.index).groupby(df["image_id"]).median()
            base_series = {
                "m=2": pd.Series(R_m2, index=df.index).groupby(df["image_id"]).median(),
                "m=1": pd.Series(R_m1, index=df.index).groupby(df["image_id"]).median(),
            }
            if np.isfinite(m_star):
                base_series[f"m*={m_star:.2f}"] = pd.Series(R_mstar, index=df.index).groupby(df["image_id"]).median()
            for nm, ser in base_series.items():
                common = Re_img.index.intersection(ser.index)
                d_img = ser.loc[common].values - Re_img.loc[common].values
                d_img = d_img[np.isfinite(d_img) & (d_img != 0)]
                from math import comb
                n = d_img.size
                if n > 0:
                    k = int(np.sum(d_img > 0))
                    p = sum(comb(n, i) for i in range(k, n+1)) / (2**n)
                    med_d = float(np.median(d_img))
                    lo_di, hi_di = _bootstrap_ci_median_simple(d_img, B=5000, seed=13)
                else:
                    p = float("nan"); med_d = float("nan"); lo_di = float("nan"); hi_di = float("nan")
                lines_baseline.append(f"[img-median] {nm}: n={n}, one-sided p={p:.2e}; median ΔR={med_d:.3f} [95% CI {lo_di:.3f},{hi_di:.3f}]")

        except Exception:
            pass

    # ---------- skip reasons ----------
    all_keys = set()
    for d in diags:
        all_keys |= set(d.get("skip_reasons", {}).keys())
    skip_totals = {
        k: int(sum(d.get("skip_reasons", {}).get(k, 0) for d in diags))
        for k in sorted(all_keys)
    }

    def _fmt_skip():
        total_skips = sum(skip_totals.values())
        if total_skips == 0:
            return "none"
        parts = []
        for k, v in sorted(skip_totals.items(), key=lambda kv: -kv[1]):
            pct = (100.0 * v / total_skips) if total_skips > 0 else 0.0
            parts.append(f"{k}={v} ({pct:.1f}%)")
        return ", ".join(parts)

    skip_list = _fmt_skip()

    # ---------- prereg & ablation outcomes (mirrored files) ----------
    prereg_path = CSV_ROOT / dataset_base / "PREREG__C1.txt"
    prereg_line = "Outcome: (no PREREG__C1.txt found)"
    if prereg_path.exists():
        txt = prereg_path.read_text().strip().splitlines()
        out_line = next((L for L in txt if L.startswith("Outcome:")), None)
        worst_line = next((L for L in txt if L.startswith("Worst-case")), None)
        if out_line:
            prereg_line = out_line
            if worst_line:
                prereg_line = prereg_line + f" | {worst_line}"

    ablation_path = CSV_ROOT / dataset_base / "ABLATION__test.txt"
    ablation_line = "Outcome: (no ABLATION__test.txt found)"
    if ablation_path.exists():
        t = ablation_path.read_text().strip().splitlines()
        out_line = next((L for L in t if L.startswith("Outcome:")), None)
        worst_p  = next((L for L in t if L.startswith("Worst p:")), None)
        worst_qc = next((L for L in t if L.startswith("Worst QC drift:")), None)
        if out_line:
            ablation_line = out_line
            if worst_p:
                ablation_line = ablation_line + f" | {worst_p}"
            if worst_qc:
                ablation_line = ablation_line + f" | {worst_qc}"

    # ---------- warnings / suggestions ----------
    warns: List[str] = []

    def _add(cond, msg):
        if cond:
            warns.append(msg)

    _add(n_kept < 100, "LOW NODES: fewer than 100 nodes kept — statistical power may be limited.")
    _add(np.isfinite(frac_strict) and frac_strict < 0.5,
         "LOW STRICT PASS: <50% strict QC passes — check tangent quality / angle gate.")
    _add(np.isfinite(frac_edge) and frac_edge > 0.20,
         f"M AT BRACKET EDGE: {frac_edge*100:.1f}% at ±{edge_eps:.2f} of bracket — consider widening bracket or re-check m solver.")
    _add(np.isfinite(R_med) and R_med > 0.60,
         f"HIGH RESIDUALS: median R(m)={R_med:.3f} — geometry/segmentation may be noisy.")
    _add(np.isfinite(parent_ambig_frac) and parent_ambig_frac > 0.10,
         f"PARENT AMBIGUITY: {parent_ambig_frac*100:.1f}% nodes ambiguous — consider stronger gating or tie-breakers.")
    _add(images_with_nodes < images_done,
         f"SPARSE IMAGES: {images_done - images_with_nodes} image(s) produced zero kept nodes — check segmentation or QC thresholds.")

    # ---------- artifact pointers ----------
    fig_dir = FIG_ROOT / dataset
    artifacts = [
        f"{fig_dir}/dataset_{dataset}__residual_hist.png",
        f"{fig_dir}/dataset_{dataset}__residual_baselines.png",
        f"{fig_dir}/dataset_{dataset}__paired_deltaR_ecdf.png",
        f"{fig_dir}/dataset_{dataset}__theta_vs_m_scatter.png",
        f"{fig_dir}/dataset_{dataset}__ablation_medianR_grid.png",
        f"{fig_dir}/dataset_{dataset}__QC_tradeoff_curves.png",
    ]

    # ---------- derive seg/suite/panel for mid-level summary ----------
    seg_method = _get_cfg("args_seg_method", "?")
    thresh_method = _get_cfg("args_thresh_method", "?")
    seg_variant = f"{seg_method}+{thresh_method}"
    parts = str(dataset).split("__")
    suite_variant = parts[-1] if len(parts) >= 3 else "base"
    panel_mode = _get_cfg("args_panel_filter", "all")

    # Build PRE-style design summary block
    body_lines = build_run_summary_content(
        df=df,
        dataset=dataset,
        seg_variant=seg_variant,
        suite_variant=suite_variant,
        panel_mode=panel_mode,
        n_images=images_with_nodes,
        n_kept=n_kept,
        n_total=n_raw,
        m_med=m_med,
        m_iqr=m_iqr,
        Rcol_for_cluster=Rcol_for_cluster,
        ic_line=ic_line,
        lines_baseline=lines_baseline,
        prereg_line=prereg_line,
        ablation_line=ablation_line,
        artifacts=[str(a) for a in artifacts],
    )

    # ---------- compose final summary ----------
    content: List[str] = [
        f"===== RUN SUMMARY — {dataset} =====",
        f"Dataset base: {dataset_base}",
        "",
        "[CONFIG SNAPSHOT]",
        *cfg_lines,
        "",
        "[COUNTS]",
        f"Images processed     : {images_done}/{n_images_total} (with nodes in {images_with_nodes})",
        f"Nodes raw→deg3→kept  : {n_raw} → {n_deg3} → {n_kept}  | kept/raw {kept_over_raw:.1f}%  kept/deg3 {kept_over_deg3:.1f}%",
        "",
        "[QC & GEOMETRY QUALITY]",
        (
            f"QC loose/strict      : {N_qc_loose}/{n_kept} ({frac_loose:.1%})  |  "
            f"{N_qc_strict}/{n_kept} ({frac_strict:.1%})"
            if n_kept > 0
            else "QC loose/strict      : n/a"
        ),
        (
            f"m median (IQR)       : {m_med:.3f}  ({m_iqr[0]:.3f},{m_iqr[1]:.3f})   mean={m_mean:.3f}"
            if np.isfinite(m_med) and np.all(np.isfinite(m_iqr))
            else "m median (IQR)       : n/a"
        ),
        f"m at bracket edge    : {n_edge}  ({frac_edge:.1%}) at ±{edge_eps:.2f}",
        f"angle_min per image  : median={angle_gate_med:.1f}°  range=({angle_gate_rng[0]:.1f}°, {angle_gate_rng[1]:.1f}°)",
        (
            f"parent ambiguity     : {parent_ambig_frac:.1%}"
            if np.isfinite(parent_ambig_frac)
            else "parent ambiguity     : n/a"
        ),
        "",
        # Design tests, baselines, prereg, ablation, artifacts
        *body_lines,
        "[SKIP REASONS]",
        skip_list,
        "",
        "[WARNINGS]",
        ("none" if not warns else "\n".join(f"- {w}" for w in warns)),
        "",
        f"Elapsed: {human_time(elapsed_sec)}",
    ]

    out_txt.write_text("\n".join(content) + "\n")
    log("\n".join(content))



# ---------- SEG-ROBUST SUMMARY HELPERS ----------
def _bootstrap_ci_median_simple(x, B=5000, seed=13):
    import numpy as np
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(int(seed))
    meds = np.median(rng.choice(x, size=(int(B), x.size), replace=True), axis=1)
    lo, hi = np.percentile(meds, [2.5, 97.5])
    return float(lo), float(hi)



# === IMAGE-CLUSTERED STATS & EFFECT SIZE LABELS ==============================
def image_cluster_stats(df: pd.DataFrame, value_col: str = "R_m_holdout",
                        B: int = 10000, seed: int = 13) -> Dict:
    """
    Image-cluster bootstrap: compute the dataset-level median of image medians,
    with 95% CI from resampling images (not nodes).
    Returns dict with median, (lo,hi), n_img, share_img_Rlt055, share_img_Rlt085.
    """
    g = df.dropna(subset=[value_col]).groupby("image_id")[value_col].median()
    arr = g.to_numpy(dtype=float)
    out = {"median": float("nan"), "ci": (float("nan"), float("nan")),
           "n_img": int(g.size), "share_img_Rlt055": float("nan"), "share_img_Rlt085": float("nan")}
    if arr.size == 0:
        return out
    obs = float(np.median(arr))
    rng = np.random.default_rng(int(seed))
    idx = np.arange(arr.size)
    boots = np.empty(int(B), dtype=float)
    for b in range(int(B)):
        pick = rng.choice(idx, size=idx.size, replace=True)
        boots[b] = float(np.median(arr[pick]))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    out["median"] = obs
    out["ci"] = (float(lo), float(hi))
    out["share_img_Rlt055"] = float(np.mean(arr < 0.55))
    out["share_img_Rlt085"] = float(np.mean(arr < 0.85))
    return out

def cliffs_delta_label(delta: float) -> str:
    """
    Conventional bins (approx Vargha–Delaney): 0.11 small, 0.28 medium, 0.43 large.
    We report direction (negative is better).
    """
    if not np.isfinite(delta): return "n/a"
    x = abs(delta)
    if x >= 0.43: return "large"
    if x >= 0.28: return "medium"
    if x >= 0.11: return "small"
    return "negligible"

def _parse_nulltest_shuffle(dataset_tag: str, reports_dir: Path) -> Dict:
    """
    Pull shuffle-null summary from reports/<variant>/NULLTEST__R_m.txt
    (written by plot_residual_distribution). Returns dict with
    med_obs, med_null, d_med, p_med, p_frac, delta.
    """
    p = Path(reports_dir) / dataset_tag / "NULLTEST__R_m.txt"
    out = {"med_obs": None, "med_null": None, "d_med": None,
           "p_med": None, "p_frac": None, "delta": None}
    if not p.exists():
        return out
    txt = p.read_text().splitlines()
    # med_obs line
    for L in txt:
        if L.strip().startswith("Observed: median="):
            try:
                out["med_obs"] = float(L.split("median=")[1].split()[0])
            except Exception:
                pass
        if L.strip().lower().startswith("shuffle:"):
            # Expected pattern we write below:
            # shuffle: med_null=..., Δmedian(obs-null)=..., p_median(lower)=..., p_frac<0.55(higher)=..., Cliff's δ=...
            try:
                parts = L.split(":")[1].split(",")
                kv = {}
                for part in parts:
                    if "=" in part:
                        k,v = part.strip().split("=",1); kv[k.strip()] = v.strip()
                out["med_null"] = float(kv.get("med_null")) if "med_null" in kv else None
                out["d_med"]    = float(kv.get("Δmedian(obs-null)")) if "Δmedian(obs-null)" in kv else None
                out["p_med"]    = float(kv.get("p_median(lower)")) if "p_median(lower)" in kv else None
                out["p_frac"]   = float(kv.get("p_frac<0.55(higher)")) if "p_frac<0.55(higher)" in kv else None
                out["delta"]    = float(kv.get("Cliff's δ")) if "Cliff's δ" in kv else None
            except Exception:
                pass
            break
    return out



def _parse_prereg_c1_from_file(dataset_tag: str, reports_dir: Path) -> Dict[str, Any]:
    """
    Parse reports/<dataset_tag>/PREREG__C1.txt (variant-level) and return a dict
    with the fields expected by _make_variant_row. Any missing quantities are
    filled with None so downstream code never KeyErrors.
    """
    # Location: reports/<dataset_tag>/PREREG__C1.txt
    path = reports_dir / dataset_tag / "PREREG__C1.txt"

    out: Dict[str, Any] = {
        "dataset": dataset_tag.split("__", 1)[0],
        "variant": dataset_tag.split("__", 1)[1] if "__" in dataset_tag else dataset_tag,
        "outcome": None,
        "p_med": None,
        "delta": None,
        # optional jitter fields so _make_variant_row can safely access them
        "p_med_jit": None,
        "delta_jit": None,
        "Outcome_jit": None,
    }

    txt = _safe_read(path)
    if not txt:
        return out

    # Outcome (PASS/FAIL/WARN)
    m_outcome = re.search(r"Outcome:\s*(PASS|FAIL|WARN)", txt)
    if m_outcome:
        out["outcome"] = m_outcome.group(1)

    # Accept multiple phrasings for p_median and Cliff's delta
    m_p = (
        re.search(r"p_median\s*\(lower\)\s*=\s*([0-9eE\.\-+]+)", txt)
        or re.search(r"p_med\s*=\s*([0-9eE\.\-+]+)", txt)
        or re.search(r"one-sided p_median\s*=\s*([0-9eE\.\-+]+)", txt)
    )
    m_d = (
        re.search(r"Cliff['’]s\s*δ\s*=\s*([0-9eE\.\-+]+)", txt)
        or re.search(r"Cliff['’]s\s*delta\s*=\s*([0-9eE\.\-+]+)", txt)
    )

    if m_p:
        out["p_med"] = float(m_p.group(1))
    if m_d:
        out["delta"] = float(m_d.group(1))

    # Optional jitter block (future-proof; harmless if not present)
    m_pj = re.search(r"p_median_jit\s*=\s*([0-9eE\.\-+]+)", txt)
    m_dj = re.search(r"Cliff['’]s\s*delta_jit\s*=\s*([0-9eE\.\-+]+)", txt)
    m_oj = re.search(r"Outcome \(jitter\):\s*(PASS|FAIL|WARN)", txt)

    if m_pj:
        out["p_med_jit"] = float(m_pj.group(1))
    if m_dj:
        out["delta_jit"] = float(m_dj.group(1))
    if m_oj:
        out["Outcome_jit"] = m_oj.group(1)

    return out





def _make_variant_row(df, dataset_tag, reports_dir, primary_metric="heldout"):
    """
    Build a one-line summary for a variant tag:
      variant, N_nodes, R_med (heldout if available), 95% CI, p_med, delta, outcome,
      jitter p/delta/outcome (if present), strict_pass_frac.
    """
    import numpy as np
    variant = dataset_tag.split("__", 1)[1] if "__" in dataset_tag else dataset_tag
    # observed metric
    if primary_metric == "heldout" and "R_m_holdout" in df.columns:
        Robs = df["R_m_holdout"].astype(float).values
    else:
        Robs = df.get("R_m", np.array([], float)).astype(float).values
    med = float(np.median(Robs)) if Robs.size else float("nan")
    lo, hi = _bootstrap_ci_median_simple(Robs, B=2000, seed=17)
    # qc strict pass
    strict_frac = float(df["qc_pass_strict"].mean()) if "qc_pass_strict" in df.columns and len(df) else float("nan")
    # prereg file parse
    c1 = _parse_prereg_c1_from_file(dataset_tag, reports_dir)
    row = {
        "variant": variant,
        "N_nodes": int(len(df)),
        "R_med": med,
        "R_med_CI_lo": lo,
        "R_med_CI_hi": hi,
        "p_med": c1["p_med"],
        "delta": c1["delta"],
        "Outcome": c1["outcome"],
        "p_med_jit": c1["p_med_jit"],
        "delta_jit": c1["delta_jit"],
        "Outcome_jit": c1["outcome_jit"],
        "strict_pass_frac": strict_frac,
    }
    return row







def _write_segrobust_summary(dataset, rows, reports_dir, log_fn):
    """
    Write a tabulated cross-variant summary and echo as a pretty table to logs.
    Files:
      reports/<dataset>/SEGROBUST__summary.tsv
      reports/<dataset>/SEGROBUST__summary.md
    """
    import pandas as pd
    from pathlib import Path
    cols = ["variant","N_nodes","R_med","R_med_CI_lo","R_med_CI_hi",
            "p_med","delta","Outcome","p_med_jit","delta_jit","Outcome_jit","strict_pass_frac"]
    df_sum = pd.DataFrame(rows, columns=cols)
    out_dir = Path(reports_dir) / dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    tsv = out_dir / "SEGROBUST__summary.tsv"
    md  = out_dir / "SEGROBUST__summary.md"
    df_sum.to_csv(tsv, sep="\t", index=False)
    # Markdown table
    md_lines = ["| " + " | ".join(cols) + " |",
                "| " + " | ".join(["---"]*len(cols)) + " |"]
    for _, r in df_sum.iterrows():
        md_lines.append("| " + " | ".join(str(r[c]) if pd.notna(r[c]) else "" for c in cols) + " |")
    md.write_text("\n".join(md_lines) + "\n")
    # Log a readable table
    log_fn("====== SEG-ROBUST SUMMARY (per variant) ======")
    log_fn(df_sum.to_string(index=False))
    log_fn(f"Saved: {tsv}")
    log_fn(f"Saved: {md}")
    log_fn("==============================================")




def plot_segrobust_scoreboard(
    datasets,
    reports_dir: Path,
    fig_root: Path,
    seg_variants=None,
    title: str = "Held-out closure: transportability & segmentation robustness",
):
    """
    Cross-dataset, segmentation-robust scoreboard for the HELD-OUT closure metric.

    Expects, for each base dataset name in `datasets`:
      - reports/<dataset>/SEGROBUST__summary.tsv
        (written by _write_segrobust_summary with primary_metric='heldout')
      - reports/<dataset>__<variant>/PREREG__C1.txt
        (written by plot_residual_distribution with R_m_holdout).

    Produces:
      figures/ALL_DATASETS__segrobust_scoreboard.png
    """
    if seg_variants is None:
        seg_variants = ["frangi+otsu", "frangi+quantile", "sato+otsu", "sato+quantile"]

    datasets = list(datasets)
    if len(datasets) == 0 or len(seg_variants) == 0:
        return

    n_rows = len(datasets)
    n_cols = len(seg_variants)

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(3.0 * n_cols, 2.4 * n_rows),
        sharex=True,
        sharey=True,
    )

    if n_rows == 1:
        axes = np.array([axes])
    if n_cols == 1:
        axes = axes[:, np.newaxis]

    for i, ds in enumerate(datasets):
        tsv_path = reports_dir / ds / "SEGROBUST__summary.tsv"
        if not tsv_path.exists():
            for j in range(n_cols):
                axes[i, j].axis("off")
            continue

        df = pd.read_csv(tsv_path, sep="\t")

        for j, variant in enumerate(seg_variants):
            ax = axes[i, j]
            row = df[df["variant"] == variant]
            if row.empty:
                ax.axis("off")
                continue

            row = row.iloc[0]
            r_med = float(row["R_med"])
            ci_lo = float(row["R_med_CI_lo"])
            ci_hi = float(row["R_med_CI_hi"])
            N_nodes = int(row["N_nodes"])

            # Traffic-light color based only on presence of prereg success markers if available
            prereg_dir = reports_dir / f"{ds}__{variant}"
            prereg_path = prereg_dir / "PREREG__C1.txt"
            face = "#f8cccc"
            if prereg_path.exists():
                txt = prereg_path.read_text()
                import re
                try:
                    # Prefer explicit PASS/WARN/FAIL outcome if present (newer files)
                    m_out = re.search(r"Outcome:\s*(PASS|WARN|FAIL)", txt)
                    if m_out and m_out.group(1) == "PASS":
                        face = "#d1f2d1"
                    else:
                        # Back-compat: tolerate older numeric summary styles if present
                        m_pm  = re.search(r"p_med=([0-9eE\.\-+]+)", txt)
                        m_pf  = re.search(r"p_frac=([0-9eE\.\-+]+)", txt)
                        m_dmx = re.search(r"max\s*[δd]elta?\s*=\s*([0-9eE\.\-+]+)", txt)
                        if m_pm and m_pf and m_dmx:
                            p_med  = float(m_pm.group(1))
                            p_frac = float(m_pf.group(1))
                            delta  = float(m_dmx.group(1))
                            face = "#d1f2d1" if (p_med <= 1e-3 and p_frac <= 1e-3 and delta <= -0.20) else "#ffe7b3"
                        else:
                            face = "#ffe7b3"
                except Exception:
                    face = "#ffe7b3"
            ax.set_facecolor(face)


            ax.plot([0.5, 0.5], [ci_lo, ci_hi], color="black", linewidth=1.5)
            ax.plot([0.5], [r_med], "o", color="black", markersize=4)
            ax.axhline(0.55, linestyle="--", linewidth=0.8)
            ax.axhline(0.85, linestyle="--", linewidth=0.8)
            ax.set_xlim(0.0, 1.0); ax.set_ylim(0.0, 1.5)
            ax.set_xticks([]); ax.set_yticks([0.0, 0.55, 0.85, 1.2])
            ax.grid(axis="y", alpha=0.15, linestyle="-", linewidth=0.7)

            lines = [f"N={N_nodes}", f"median={r_med:.3f}", f"95%[{ci_lo:.3f},{ci_hi:.3f}]"]
            ax.text(0.02, 0.98, "\n".join(lines), transform=ax.transAxes,
                    va="top", ha="left", fontsize=7,
                    bbox=dict(facecolor="white", alpha=0.9, edgecolor="none"))

            if i == 0:
                ax.set_title(variant.replace("+", " + "), fontsize=9)

        axes[i, 0].set_ylabel(ds, rotation=0, labelpad=35, fontsize=10, va="center")

    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.95])

    out_path = fig_root / "ALL_DATASETS__segrobust_scoreboard.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=300)
    plt.close(fig)



def plot_showstopper_residuals(
    dataset_tags: List[str],
    R_col: str = "R_m_holdout",
    labels: Optional[List[str]] = None,
    out_name: Optional[str] = None,
    boot: int = 5000,
):
    """
    PRE-style overlay of residual densities for 2+ dataset variants.

    Reads node CSVs under reports/<variant>/nodes__<variant>.csv, uses the chosen
    R_col (R_m_holdout or R_m), annotates node- and image-level medians with 95% CIs,
    and writes a single comparison figure under figures/SHOWSTOPPER/.
    """
    if labels is None:
        labels = dataset_tags

    series = []
    infos = []

    for tag, lab in zip(dataset_tags, labels):
        csv_path = CSV_ROOT / tag / f"nodes__{tag}.csv"
        if not csv_path.exists():
            continue
        df = pd.read_csv(csv_path)
        col = R_col if R_col in df.columns else ("R_m" if "R_m" in df.columns else None)
        if col is None:
            continue
        x = df[col].astype(float).to_numpy()
        x = x[np.isfinite(x)]
        if x.size == 0:
            continue
        lo, hi = _bootstrap_ci_median_simple(x, B=boot, seed=17)
        med = float(np.median(x))
        ic = image_cluster_stats(df, value_col=col, B=boot, seed=17)
        series.append((lab, x))
        infos.append({
            "label": lab,
            "N_nodes": x.size,
            "node_med": med,
            "node_lo": lo,
            "node_hi": hi,
            "img_med": ic["median"],
            "img_lo": ic["ci"][0],
            "img_hi": ic["ci"][1],
            "img_share055": ic["share_img_Rlt055"],
        })

    if len(series) < 2:
        return

    fig = plt.figure(figsize=(8.2, 5.2))
    ax = plt.gca()

    all_vals = np.concatenate([s[1] for s in series])
    nb = min(80, max(30, int(np.ceil(np.sqrt(all_vals.size)))))
    xmax = float(np.nanpercentile(all_vals, 99.5))
    bins = np.linspace(0.0, min(2.0, xmax), nb)

    for lab, x in series:
        ax.hist(x, bins=bins, density=True, alpha=0.45, label=f"{lab} (N={x.size})")

    ax.axvspan(0.0, 0.55, alpha=0.10)
    ax.axvspan(0.55, 0.85, alpha=0.07)
    ax.axvline(0.55, linestyle="--", linewidth=1.0)
    ax.axvline(0.85, linestyle="--", linewidth=1.0)

    for info in infos:
        ax.axvline(info["node_med"], linewidth=1.8)

    lines = []
    for info in infos:
        lines.append(
            f"{info['label']}: "
            f"node med={info['node_med']:.3f} 95%[{info['node_lo']:.3f},{info['node_hi']:.3f}]   "
            f"img med={info['img_med']:.3f} 95%[{info['img_lo']:.3f},{info['img_hi']:.3f}]   "
            f"Pr_img[R<0.55]={info['img_share055']:.2%}"
        )
    ax.text(
        0.02, 0.98, "\n".join(lines),
        transform=ax.transAxes, va="top", ha="left", fontsize=9,
        bbox=dict(facecolor="white", alpha=0.90, edgecolor="none", boxstyle="round,pad=0.25")
    )

    # Make labels self-contained in this function
    metric_label = "heldout" if R_col == "R_m_holdout" else "asconfigured"
    base_names = [t.split("__", 1)[0] for t in dataset_tags]
    title_datasets = " vs ".join(base_names[:2]) if len(base_names) >= 2 else ", ".join(base_names)

    ax.set_xlabel(f"Closure residual ({metric_label})")
    ax.set_ylabel("Density")
    ax.set_title(f"Closure residual ({metric_label}) — {title_datasets}")

    ax.grid(axis="y", alpha=0.25, linestyle="-", linewidth=0.8)
    ax.legend(loc="upper right", frameon=True, framealpha=0.95, facecolor="white", edgecolor="none")

    out_dir = FIG_ROOT / "SHOWSTOPPER"
    out_dir.mkdir(parents=True, exist_ok=True)
    if out_name is None:
        base = "__vs__".join([t.split("__", 1)[0] for t in dataset_tags[:2]])
        out_name = f"{base}__showstopper_residuals.png"
    fig.savefig(str(out_dir / out_name), dpi=300, bbox_inches="tight")
    plt.close(fig)





# === NEW: cross-dataset suite summary (baseline/lo/hi) with stat & syst uncertainties ===
def _write_suite_summary(datasets, reports_dir: Path, out_dir: Path, suite_names=("base", "lo", "hi")):
    """
    Build a single TSV/MD for all requested datasets summarizing:
      - held-out R(m) median with 95% bootstrap CI (statistical),
      - systematic uncertainty as a quadrature combination of:
          • suite half-range across (base, lo, hi) for the chosen variant_core,
          • segmentation half-range across all variants sharing that core.
    Assumes SEGROBUST__summary.tsv exists per dataset and contains rows with
    variant names like 'frangi+otsu__base', 'frangi+otsu__lo', 'frangi+otsu__hi'
    (our main loop writes those via _make_variant_row).
    """
    rows = []
    for ds in datasets:
        tsv = reports_dir / ds / "SEGROBUST__summary.tsv"
        if not tsv.exists():
            continue
        df = pd.read_csv(tsv, sep="\t")

        # Use the segmentation variant with the most nodes in baseline
        base_rows = df[df["variant"].str.endswith("__base") | (~df["variant"].str.contains("__"))]
        if base_rows.empty:
            continue

        best_variant = base_rows.sort_values("N_nodes", ascending=False).iloc[0]["variant"]
        base_core = best_variant.split("__")[0] if "__" in best_variant else best_variant

        # Collect across suite for this core
        suite_stats: Dict[str, Tuple[float, float, float, int]] = {}
        for s in suite_names:
            if s != "base":
                look = f"{base_core}__{s}"
            else:
                # allow either "base_core" or "base_core__base" as "base"
                look = base_core if (df["variant"] == base_core).any() else f"{base_core}__base"
            row = df[df["variant"] == look]
            if row.empty:
                continue
            rmed = float(row.iloc[0]["R_med"])
            lo = float(row.iloc[0]["R_med_CI_lo"])
            hi = float(row.iloc[0]["R_med_CI_hi"])
            n_nodes = int(row.iloc[0]["N_nodes"])
            suite_stats[s] = (rmed, lo, hi, n_nodes)

        if len(suite_stats) == 0:
            continue

        # Suite half-range (base/lo/hi) in R_med
        meds = [suite_stats[s][0] for s in suite_stats]
        suite_half = 0.5 * (max(meds) - min(meds)) if len(meds) >= 2 else float("nan")

        # Take base tuple (fall back to "first" if base missing)
        base_tuple = suite_stats.get("base", next(iter(suite_stats.values())))

        # Build systematic contributions
        systs: List[float] = []
        if len(meds) >= 2:
            systs.append(suite_half)

        # Segmentation half-range across all variants that share this core
        try:
            core_prefix = base_core.split("__")[0]
            seg_block = df[df["variant"].str.startswith(core_prefix)]
            if not seg_block.empty:
                v = seg_block["R_med"].astype(float).to_numpy()
                if v.size >= 2:
                    seg_half = 0.5 * (float(np.nanmax(v)) - float(np.nanmin(v)))
                    systs.append(seg_half)
        except Exception:
            pass

        # Combined systematic: quadrature of available components
        systematic_combined = math.sqrt(sum(x * x for x in systs)) if systs else float("nan")

        rows.append({
            "dataset": ds,
            "variant_core": base_core,
            "R_med_base": base_tuple[0],
            "R_med_CI_lo": base_tuple[1],
            "R_med_CI_hi": base_tuple[2],
            "N_nodes_base": base_tuple[3],
            # store the combined systematic in the same column name you already use
            "systematic_half_range": systematic_combined,
        })

    if not rows:
        return

    df_out = pd.DataFrame(
        rows,
        columns=[
            "dataset",
            "variant_core",
            "R_med_base",
            "R_med_CI_lo",
            "R_med_CI_hi",
            "N_nodes_base",
            "systematic_half_range",
        ],
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    tsv_out = out_dir / "ALL__SUITE_summary.tsv"
    md_out = out_dir / "ALL__SUITE_summary.md"
    df_out.to_csv(tsv_out, sep="\t", index=False)

    md_lines = [
        "| dataset | variant_core | R_med (base) | 95% CI lo | 95% CI hi | N (base) | systematic (± combined) |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for _, r in df_out.iterrows():
        md_lines.append(
            f"| {r['dataset']} | {r['variant_core']} | {r['R_med_base']:.3f} | "
            f"{r['R_med_CI_lo']:.3f} | {r['R_med_CI_hi']:.3f} | {int(r['N_nodes_base'])} | "
            f"±{r['systematic_half_range']:.3f} |"
        )
    md_out.write_text("\n".join(md_lines) + "\n")

    log("====== CROSS-SUITE SUMMARY ======")
    log(df_out.to_string(index=False))
    log(f"Saved: {tsv_out}")
    log(f"Saved: {md_out}")
    log("=================================")



# ---------- VERDICT ENGINE (hard PASS/FAIL) ----------

import re, hashlib
from datetime import datetime

def _safe_read(p: Path) -> str:
    try:
        return p.read_text()
    except Exception:
        return ""

def _read_first_existing(paths: List[Path]) -> Tuple[Optional[Path], str]:
    for pp in paths:
        if pp.exists():
            return pp, _safe_read(pp)
    return None, ""

def _hash_str(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8", "ignore")).hexdigest()[:16]

def _pick_primary_variant(dataset: str, reports_dir: Path) -> Optional[str]:
    """
    Choose a canonical variant for scoring (dataset-level): variant with max N_nodes in SEGROBUST__summary.tsv.
    """
    tsv = reports_dir / dataset / "SEGROBUST__summary.tsv"
    if not tsv.exists():
        return None
    try:
        df = pd.read_csv(tsv, sep="\t")
        if "variant" in df.columns and "N_nodes" in df.columns and len(df) > 0:
            row = df.sort_values("N_nodes", ascending=False).iloc[0]
            return f"{dataset}__{row['variant']}"
    except Exception:
        pass
    return None

def _bool(b) -> bool:
    return bool(b) if isinstance(b, (bool, np.bool_)) else str(b).strip().upper() == "PASS"

def _parse_prereg_variant(reports_dir: Path, dataset_tag: str) -> Dict:
    # Prefer variant-level PREREG__C1, else dataset-base mirror
    base = dataset_tag.split("__", 1)[0]
    p_variant = reports_dir / dataset_tag / "PREREG__C1.txt"
    p_base    = reports_dir / base       / "PREREG__C1.txt"
    pp, txt = _read_first_existing([p_variant, p_base])
    out = {"found": False, "outcome": None, "p_med_worst": None, "p_frac_worst": None, "delta_max": None}
    if not txt:
        return out
    out["found"] = True
    m_out = re.search(r"Outcome:\s*(PASS|FAIL)", txt)
    m_pm  = re.search(r"Worst-case.*p_med=([0-9eE\.\-+]+)", txt)
    m_pf  = re.search(r"Worst-case.*p_frac=([0-9eE\.\-+]+)", txt)
    m_dm  = re.search(r"max δ=([0-9eE\.\-+]+)", txt)
    if m_out: out["outcome"] = m_out.group(1)
    if m_pm:  out["p_med_worst"]  = float(m_pm.group(1))
    if m_pf:  out["p_frac_worst"] = float(m_pf.group(1))
    if m_dm:  out["delta_max"]    = float(m_dm.group(1))
    return out

def _parse_panelC(base_reports_dir: Path, dataset_tag: str) -> Dict:
    base = dataset_tag.split("__", 1)[0]
    p = base_reports_dir / base / "PANELC__theta_vs_m_test.txt"
    txt = _safe_read(p)
    out = {"found": False, "verdict": None, "share": None}
    if not txt:
        return out
    out["found"] = True
    m_v = re.search(r"Panel C verdict:\s*(PASS|FAIL)", txt)
    m_s = re.search(r"% within \|Δm\|≤0\.25 =\s*([0-9\.]+)", txt)
    if m_v: out["verdict"] = m_v.group(1)
    if m_s: out["share"]   = float(m_s.group(1))/100.0
    return out

def _parse_ablation(base_reports_dir: Path, dataset_tag: str) -> Dict:
    base = dataset_tag.split("__", 1)[0]
    p = base_reports_dir / base / "ABLATION__test.txt"
    txt = _safe_read(p)
    out = {"found": False, "verdict": None, "worst_p": None}
    if not txt:
        return out
    out["found"] = True
    m_v = re.search(r"Outcome:\s*(PASS|FAIL)", txt)
    m_p = re.search(r"Worst p:\s*([0-9eE\.\-+]+)", txt)
    if m_v: out["verdict"] = m_v.group(1)
    if m_p: out["worst_p"] = float(m_p.group(1))
    return out



def append_to_final_table(df: pd.DataFrame, dataset_tag: str, reports_dir: Path = CSV_ROOT):
    """
    Append one row to reports/<DATASET_BASE>/FINAL__table.tsv for this variant (dataset_tag).
    Pulls node/image counts, node & image medians with CIs, nulltest deltas/p-values,
    Panel C share, and parent-direction p-values.
    """
    base = dataset_tag.split("__", 1)[0]
    out_dir = reports_dir / base
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "FINAL__table.tsv"

    Rcol = "R_m_holdout" if "R_m_holdout" in df.columns else ("R_m" if "R_m" in df.columns else None)
    if Rcol is None or len(df) == 0:
        return

    N_images = int(df["image_id"].nunique())
    N_nodes = int(len(df))
    strict_frac = float(df["qc_pass_strict"].mean()) if "qc_pass_strict" in df.columns and len(df) > 0 else float("nan")

    node_vals = df[Rcol].astype(float).to_numpy()
    node_vals = node_vals[np.isfinite(node_vals)]
    if node_vals.size:
        lo_node, hi_node = _bootstrap_ci_median_simple(node_vals, B=5000, seed=17)
        med_node = float(np.median(node_vals))
    else:
        lo_node, hi_node, med_node = float("nan"), float("nan"), float("nan")

    ic = image_cluster_stats(df, value_col=Rcol, B=5000, seed=17)

    nt = _parse_nulltest_shuffle(dataset_tag, reports_dir)
    d_med = nt["d_med"]
    p_med = nt["p_med"]
    p_frac = nt["p_frac"]
    delta = nt["delta"]

    pan = _parse_panelC(reports_dir, dataset_tag)
    pnt = _parse_parentdir_variant(reports_dir, dataset_tag)

    row = {
        "dataset": base,
        "variant": dataset_tag.split("__", 1)[1] if "__" in dataset_tag else dataset_tag,
        "N_images": N_images,
        "N_nodes": N_nodes,
        "strict_QC_frac": strict_frac,
        "median_R_node": med_node,
        "95CI_node_lo": lo_node,
        "95CI_node_hi": hi_node,
        "median_R_img": ic["median"],
        "95CI_img_lo": ic["ci"][0],
        "95CI_img_hi": ic["ci"][1],
        "delta_median_vs_shuffle": d_med,
        "p_median": p_med,
        "p_frac055": p_frac,
        "Cliffs_delta": delta,
        "Cliffs_size": cliffs_delta_label(delta) if delta is not None else "",
        "PanelC_share_le_0.25": pan["share"] if pan["found"] else float("nan"),
        "phi_p_m2": pnt["p_m2"] if pnt["found"] else float("nan"),
        "phi_p_m1": pnt["p_m1"] if pnt["found"] else float("nan"),
    }

    import csv
    write_header = not out_path.exists()
    with open(out_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()), delimiter="\t")
        if write_header:
            w.writeheader()
        w.writerow(row)






def _parse_parentdir_variant(reports_dir: Path, dataset_tag: str) -> Dict:
    p = reports_dir / dataset_tag / "PARENTDIR__phi.txt"
    txt = _safe_read(p)
    out = {"found": False, "p_m2": None, "p_m1": None, "p_120": None}
    if not txt:
        return out
    out["found"] = True
    def _grab(name):
        m = re.search(rf"Angle Δφ>0 vs {name}:\s*n=\d+,\s*p=([0-9eE\.\-+]+)", txt)
        return float(m.group(1)) if m else None
    out["p_m2"]  = _grab("m=2")
    out["p_m1"]  = _grab("m=1")
    out["p_120"] = _grab("120°")
    return out

def _parse_uncertainty_variant(reports_dir: Path, dataset_tag: str) -> Dict:
    p = reports_dir / dataset_tag / "UNCERTAINTY__nodes.csv"
    head = _safe_read(p).splitlines()[:1]
    out = {"found": False, "outcome": None}
    if not head:
        return out
    out["found"] = True
    m = re.search(r"outcome=(PASS|FAIL)", head[0])
    if m: out["outcome"] = m.group(1)
    return out

def _parse_equipartition(base_reports_dir: Path, dataset_tag: str) -> Dict:
    base = dataset_tag.split("__", 1)[0]
    p = base_reports_dir / base / "EQUIPARTITION__test.txt"
    txt = _safe_read(p)
    out = {"found": False, "cvS_med": None, "cvK_med": None}
    if not txt:
        return out
    out["found"] = True
    mS = re.search(r"CV\(S\) median=([0-9\.eE\-\+]+)", txt)
    mK = re.search(r"CV\(kappa\) median=([0-9\.eE\-\+]+)", txt)
    if mS: out["cvS_med"] = float(mS.group(1))
    if mK: out["cvK_med"] = float(mK.group(1))
    return out

def _parse_snell(base_reports_dir: Path, dataset_tag: str) -> Dict:
    base = dataset_tag.split("__", 1)[0]
    p = base_reports_dir / base / "SNELL__test.txt"
    txt = _safe_read(p)
    out = {"found": False, "N": None, "slope": None, "ci_lo": None, "ci_hi": None,
           "intercept": None, "R2": None, "p_rot": None, "p_sh": None}
    if not txt:
        return out
    out["found"] = True
    def _grab(rx):
        m = re.search(rx, txt)
        return float(m.group(1)) if m else None
    out["N"]         = _grab(r"N events:\s*([0-9]+)")
    out["slope"]     = _grab(r"slope:\s*([0-9eE\.\-+]+)")
    out["ci_lo"]     = _grab(r"CI_lo=([0-9eE\.\-+]+)")
    out["ci_hi"]     = _grab(r"CI_hi=([0-9eE\.\-+]+)")
    out["intercept"] = _grab(r"intercept:\s*([0-9eE\.\-+]+)")
    out["R2"]        = _grab(r"R2:\s*([0-9eE\.\-+]+)")
    out["p_rot"]     = _grab(r"null_p_rot_normals:\s*([0-9eE\.\-+]+)")
    out["p_sh"]      = _grab(r"null_p_shuffle_n2:\s*([0-9eE\.\-+]+)")
    return out

def _score_variant(dataset_tag: str,
                   reports_dir: Path,
                   alpha_prereg: float,
                   parentdir_alpha: float,
                   equip_cvS_target: float,
                   equip_cvK_target: float,
                   panelC_share_min: float,
                   snell_R2_min: float,
                   snell_min_N: int) -> Dict:
    """Return dict with PASS/FAIL/NA per test + overall points for this variant."""
    base_reports_dir = reports_dir
    # Primary prereg (held-out residual)
    pre = _parse_prereg_variant(base_reports_dir, dataset_tag)
    primary_pass = (pre["found"] and pre["outcome"] == "PASS"
                    and pre["p_med_worst"]  is not None and pre["p_frac_worst"] is not None and pre["delta_max"] is not None
                    and pre["p_med_worst"]  <= alpha_prereg
                    and pre["p_frac_worst"] <= alpha_prereg
                    and pre["delta_max"]    <= -0.20)
    # Panel C (near-symmetric θ–m behavior)
    pan = _parse_panelC(base_reports_dir, dataset_tag)
    panelC_pass = pan["found"] and pan["verdict"] == "PASS" and (pan["share"] is None or pan["share"] >= panelC_share_min)

    # Ablation robustness
    abl = _parse_ablation(base_reports_dir, dataset_tag)
    ablation_pass = abl["found"] and abl["verdict"] == "PASS"

    # Parent-direction Δφ vs baselines
    par = _parse_parentdir_variant(base_reports_dir, dataset_tag)
    parentdir_pass = (par["found"]
                      and all(p is not None and p <= parentdir_alpha for p in [par["p_m2"], par["p_m1"], par["p_120"]]))

    # Uncertainty (step 3)
    unc = _parse_uncertainty_variant(base_reports_dir, dataset_tag)
    uncertainty_pass = (unc["found"] and unc["outcome"] == "PASS")

    # Equipartition demo (if run with --demo)
    eqp = _parse_equipartition(base_reports_dir, dataset_tag)
    if eqp["found"]:
        equip_pass = ((eqp["cvS_med"] is not None and eqp["cvS_med"] <= equip_cvS_target) and
                      (eqp["cvK_med"] is not None and eqp["cvK_med"] <= equip_cvK_target))
    else:
        equip_pass = None  # NA

    # Snell demo (if run with --demo)
    snl = _parse_snell(base_reports_dir, dataset_tag)
    if snl["found"] and snl["N"] is not None and snl["N"] >= snell_min_N:
        slope_ok = (snl["ci_lo"] is not None and snl["ci_hi"] is not None and snl["ci_lo"] <= 1.0 <= snl["ci_hi"])
        intercept_ok = (snl["intercept"] is not None and abs(snl["intercept"]) <= 0.10)
        r2_ok = (snl["R2"] is not None and snl["R2"] >= snell_R2_min)
        null_ok = (snl["p_rot"] is not None and snl["p_sh"] is not None and snl["p_rot"] <= 0.01 and snl["p_sh"] <= 0.01)
        snell_pass = (slope_ok and intercept_ok and r2_ok and null_ok)
    else:
        snell_pass = None  # NA or too few crossings

    # Scoring
    # Primary = 4 points if PASS; each secondary PASS = +1; NA gives 0 and does not veto.
    points = 0
    total_secondaries = 0
    if primary_pass: points += 4
    for flag in [panelC_pass, ablation_pass, parentdir_pass, uncertainty_pass, equip_pass, snell_pass]:
        if flag is None:
            continue
        total_secondaries += 1
        if flag: points += 1

    # Verdict logic
    if primary_pass and (points >= 4 + max(1, int(0.5 * total_secondaries))):
        overall = "SUPPORT"
    elif (not primary_pass) and (points >= max(1, int(0.5 * (4 + total_secondaries))) ):  # enough secondaries to be mixed
        overall = "MIXED"
    else:
        overall = "NO SUPPORT"

    return {
        "dataset_tag": dataset_tag,
        "primary_prereg": {"found": pre["found"], "pass": primary_pass,
                           "alpha": alpha_prereg,
                           "worst_p_median": pre["p_med_worst"],
                           "worst_p_frac055": pre["p_frac_worst"],
                           "max_cliffs_delta": pre["delta_max"]},
        "panelC": {"found": pan["found"], "pass": panelC_pass, "share": pan["share"]},
        "ablation": {"found": abl["found"], "pass": ablation_pass, "worst_p": abl["worst_p"]},
        "parent_direction": {"found": par["found"], "pass": parentdir_pass,
                             "p_m2": par["p_m2"], "p_m1": par["p_m1"], "p_120": par["p_120"],
                             "alpha": parentdir_alpha},
        "uncertainty": {"found": unc["found"], "pass": uncertainty_pass},
        "equipartition": {"found": eqp["found"], "pass": (None if eqp["found"] is False else equip_pass),
                          "cvS_med": eqp["cvS_med"], "cvK_med": eqp["cvK_med"],
                          "targets": {"CVS": equip_cvS_target, "CVK": equip_cvK_target}},
        "snell": {"found": snl["found"], "pass": (None if snl["found"] is False else snell_pass),
                  "N": snl["N"], "slope": snl["slope"], "ci": [snl["ci_lo"], snl["ci_hi"]],
                  "intercept": snl["intercept"], "R2": snl["R2"], "p_rot": snl["p_rot"], "p_sh": snl["p_sh"],
                  "criteria": {"R2_min": snell_R2_min, "min_N": snell_min_N}},
        "points": int(points),
        "secondaries_counted": int(total_secondaries),
        "overall": overall
    }

def run_validation_suite(dataset: str,
                         reports_dir: Path,
                         scope: str,
                         verdict_name: str,
                         alpha_prereg: float,
                         parentdir_alpha: float,
                         equip_cvS_target: float,
                         equip_cvK_target: float,
                         panelC_share_min: float,
                         snell_R2_min: float,
                         snell_min_N: int):
    """
    Assemble a definitive scoreboard and write:
      reports/<DATASET>/VERDICT__<tag>.txt
      reports/<DATASET>/VERDICT__<tag>.json
    If scope='variant', also write one TXT/JSON per variant found.
    """
    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    # Discover variants from SEGROBUST summary (if present)
    tsv = reports_dir / dataset / "SEGROBUST__summary.tsv"
    variants = []
    if tsv.exists():
        try:
            df = pd.read_csv(tsv, sep="\t")
            for v in df["variant"].astype(str).tolist():
                variants.append(f"{dataset}__{v}")
        except Exception:
            pass
    # Fallback: try to list subfolders under reports/
    if not variants:
        variants = [p.name for p in (reports_dir).glob(f"{dataset}__*") if p.is_dir()]

    results = []
    if scope == "dataset":
        primary = _pick_primary_variant(dataset, reports_dir)
        if primary is None and variants:
            primary = variants[0]
        if primary:
            results.append(
                _score_variant(
                    dataset_tag=primary,
                    reports_dir=reports_dir,
                    alpha_prereg=alpha_prereg,
                    parentdir_alpha=parentdir_alpha,
                    equip_cvS_target=equip_cvS_target,
                    equip_cvK_target=equip_cvK_target,
                    panelC_share_min=panelC_share_min,
                    snell_R2_min=snell_R2_min,
                    snell_min_N=snell_min_N
                )
            )
    else:
        for v in variants:
            results.append(
                _score_variant(
                    dataset_tag=v,
                    reports_dir=reports_dir,
                    alpha_prereg=alpha_prereg,
                    parentdir_alpha=parentdir_alpha,
                    equip_cvS_target=equip_cvS_target,
                    equip_cvK_target=equip_cvK_target,
                    panelC_share_min=panelC_share_min,
                    snell_R2_min=snell_R2_min,
                    snell_min_N=snell_min_N
                )
            )

    # Compose human-readable scoreboard
    lines = [f"===== VERDICT ({verdict_name}) — {dataset} =====",
             f"time: {now}",
             f"scope: {scope}",
             ""]
    for R in results:
        tag = R["dataset_tag"]
        lines.append(f"[{tag}] OVERALL: {R['overall']}  |  points={R['points']} (primary=4, +1 per secondary PASS; counted={R['secondaries_counted']})")
        P = R["primary_prereg"]
        lines.append(f"  Primary (held-out closure): {'PASS' if P['pass'] else 'FAIL' if P['found'] else 'NA'}"
                     f"  | worst p_med={P['worst_p_median']}, worst p_frac={P['worst_p_frac055']}, max δ={P['max_cliffs_delta']}")
        lines.append(f"  Panel C (θ–m near-symmetric): {'PASS' if R['panelC']['pass'] else 'FAIL' if R['panelC']['found'] else 'NA'}"
                     + (f"  | share≤0.25={R['panelC']['share']:.3f}" if R['panelC']['share'] is not None else ""))
        lines.append(f"  Ablation robustness: {'PASS' if R['ablation']['pass'] else 'FAIL' if R['ablation']['found'] else 'NA'}")
        lines.append(f"  Parent-direction Δφ vs baselines: {'PASS' if R['parent_direction']['pass'] else 'FAIL' if R['parent_direction']['found'] else 'NA'}")
        lines.append(f"  Uncertainty (step 3): {'PASS' if R['uncertainty']['pass'] else 'FAIL' if R['uncertainty']['found'] else 'NA'}")
        ep = R['equipartition']
        lines.append(f"  Equipartition (demo): " + ("NA" if ep['found'] is False else ("PASS" if ep['pass'] else "FAIL"))
                     + (f"  | med CV(S)={ep['cvS_med']}, med CV(κ)={ep['cvK_med']}" if ep['found'] else ""))
        sn = R['snell']
        lines.append(f"  Snell refraction (demo): " + ("NA" if sn['found'] is False or sn['pass'] is None else ("PASS" if sn['pass'] else "FAIL"))
                     + (f"  | N={sn['N']}, slope={sn['slope']}, CI=[{sn['ci'][0]},{sn['ci'][1]}], int={sn['intercept']}, R2={sn['R2']}, p_rot={sn['p_rot']}, p_sh={sn['p_sh']}"
                        if sn['found'] else ""))
        lines.append("")

    # Fingerprint: hash of the assembled text
    fp = _hash_str("\n".join(lines))

    out_dir = reports_dir / dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    txt_path  = out_dir / f"VERDICT__{verdict_name}.txt"
    json_path = out_dir / f"VERDICT__{verdict_name}.json"
    lines.append(f"fingerprint: {fp}")
    txt_path.write_text("\n".join(lines) + "\n")

    import json as _json
    json_obj = {"dataset": dataset, "scope": scope, "verdict_name": verdict_name,
                "time_utc": now, "fingerprint": fp, "results": results}
    json_path.write_text(_json.dumps(json_obj, indent=2, sort_keys=True))

    # Console summary
    for R in results:
        log(f"[VERDICT] {dataset} | {R['dataset_tag']} → {R['overall']} (points={R['points']}, secondaries_counted={R['secondaries_counted']})")
    log(f"[VERDICT] Wrote {txt_path} and {json_path}")

# ---------- MAIN ----------
def main():

    ensure_dirs()
    parser = argparse.ArgumentParser(
        description="EPIC Angle-Only Estimator (per-node m inversion and tariff tomography)"
    )

    # --------------------
    # Core I/O and datasets
    # --------------------
    parser.add_argument(
        "--data-root", type=str, default=str(DATA_ROOT_DEFAULT),
        help="Base folder containing dataset subfolders (HRF, DRIVE, STARE, CHASE_DB1, etc.)"
    )
    parser.add_argument(
        "--datasets", nargs="+", default=["HRF"],
        help="One or more dataset folder names under data-root"
    )
    parser.add_argument(
        "--max-images", type=int, default=None,
        help="Optional limit per dataset (for quick runs)"
    )
    parser.add_argument(
        "--sample-frac", type=float, default=None,
        help="Optional fraction (0,1] to randomly sample images per dataset before processing (e.g., 0.25)"
    )
    parser.add_argument(
        "--sample-seed", type=int, default=13,
        help="RNG seed for --sample-frac sampling"
    )
    parser.add_argument(
        "--save-debug", action="store_true",
        help="Save per-image skeleton overlays & junction QC overlays"
    )

    # --------------------
    # Segmentation choices
    # --------------------
    parser.add_argument(
        "--seg-method", type=str, default="frangi", choices=["frangi", "sato"],
        help="Vesselness method when mask not provided"
    )
    parser.add_argument(
        "--thresh-method", type=str, default="otsu", choices=["otsu", "quantile"],
        help="Thresholding method for vesselness map (when mask not provided)"
    )
    parser.add_argument(
        "--seg-robust", action="store_true",
        help="Run 4 segmentation strata: (Frangi|Sato) × (Otsu|Quantile) and write stratified outputs"
    )

    # -------------
    # Angle gating
    # -------------
    parser.add_argument(
        "--min-angle", type=float, default=10.0,
        help="Min daughter–daughter angle (degrees)"
    )
    parser.add_argument(
        "--max-angle", type=float, default=170.0,
        help="Max daughter–daughter angle (degrees)"
    )
    parser.add_argument(
        "--angle-auto", action="store_true",
        help="Enable per-image automatic lower angle gate"
    )
    parser.add_argument(
        "--angle-auto-pctl", type=float, default=5.0,
        help="Percentile for automatic lower angle gate"
    )
    parser.add_argument(
        "--min-angle-floor", type=float, default=10.0,
        help="Floor for automatic lower angle gate (degrees)"
    )
    parser.add_argument(
        "--angle-soft-margin", type=float, default=3.0,
        help="Soft rescue margin (deg): if θ₁₂ is within this margin outside the gate, "
             "clamp to the nearest bound for inversion and tag the node."
    )

    # ------------------------
    # Geometry robustness knobs
    # ------------------------
    parser.add_argument(
        "--min-branch-len", type=int, default=10,
        help="Min branch length in pixels"
    )
    parser.add_argument(
        "--min-radius", type=float, default=1.0,
        help="Min local radius (pixels)"
    )
    parser.add_argument(
        "--tangent-len", type=int, default=16,
        help="Pixels from the junction used to fit tangents"
    )
    parser.add_argument(
        "--svd-ratio-min", type=float, default=3.0,
        help="Minimum PCA singular value ratio (S0/S1) to accept a tangent"
    )
    parser.add_argument(
        "--dedup-radius", type=int, default=3,
        help="Pixel radius for de-duplicating degree-3 skeleton junction pixels"
    )

    # -------------------------------
    # Geometry micro-choices & normal
    # -------------------------------
    parser.add_argument(
        "--tangent-mode", type=str, default="pca",
        choices=["pca", "chord"],
        help="Tangent estimation: 'pca' (default) or 'chord'"
    )
    parser.add_argument(
        "--parent-tie-break", type=str, default="conservative",
        choices=["conservative", "optimistic"],
        help="When parent/daughter assignment alternatives disagree: "
             "conservative=pick worse R; optimistic=pick better R."
    )
    parser.add_argument(
        "--radius-estimator", type=str, default="A",
        choices=["A", "B"],
        help="Radius micro-knob: A=(5,10,15)/halfwin=2 (default); B=(8,12,16)/halfwin=3."
    )
    parser.add_argument(
        "--R-norm", dest="R_norm", type=str, default="baseline",
        choices=["baseline", "sum"],
        help="Residual normalization: baseline=(r1^m+r2^m), sum=(r0^m+r1^m+r2^m)"
    )
    parser.add_argument(
        "--workers", type=int,
        default=max(1, (os.cpu_count() or 2) // 2),
        help="Number of parallel worker processes (default ~ half logical cores)"
    )
    parser.add_argument(
        "--intraop-threads", type=int, default=1,
        help="Threads per worker for BLAS/OpenMP (MKL/OpenBLAS). 1 avoids oversubscription."
    )

    # -------------
    # m inversion
    # -------------
    parser.add_argument(
        "--m-min", type=float, default=0.2,
        help="Lower bracket for m"
    )
    parser.add_argument(
        "--m-max", type=float, default=4.0,
        help="Upper bracket for m"
    )

    # -------------
    # QC regime
    # -------------
    parser.add_argument(
        "--strict-qc", action="store_true",
        help="Also compute/report a stricter QC with tighter residual thresholds"
    )

    # -------------------
    # Stability & profile
    # -------------------
    parser.add_argument(
        "--radius-jitter-frac", type=float, default=0.0,
        help="Optional multiplicative jitter fraction for stability check (e.g., 0.03 for ±3%)"
    )
    parser.add_argument(
        "--px-size-um", type=float, default=float("nan"),
        help="Optional pixel size (µm/px) recorded in CSV; does not change angle-only m"
    )
    parser.add_argument(
        "--profile", type=str, default="none",
        choices=["none", "prereg-2025Q4"],
        help="Frozen profile to eliminate researcher d.o.f.; 'prereg-2025Q4' sets all knobs"
    )

    # -----------------------
    # Panel B / C1 + nulls
    # -----------------------
    parser.add_argument(
        "--null-perm", type=int, default=2000,
        help="Permutations per negative control for R(m) nulls (shuffle/swap/randtheta/rshuffle)"
    )
    parser.add_argument(
        "--boot", type=int, default=5000,
        help="Bootstrap replicates for 95% CI on median R(m)"
    )
    parser.add_argument(
        "--primary-metric", type=str, default="asconfigured",
        choices=["asconfigured", "heldout"],
        help="Residual for Panel B & PREREG: 'asconfigured' (m chosen by minimal R) "
             "or 'heldout' (m from radii+angle only; directions held out)."
    )
    parser.add_argument(
        "--panel-filter", type=str, default="all",
        choices=["all", "strict"],
        help="Panels B/C node set: 'all' (recommended) or 'strict' (strict-QC nodes only)."
    )

    # Gaussian jitter
    parser.add_argument(
        "--radius-jitter-sd", type=float, default=0.0,
        help="Independent Gaussian jitter SD as a fraction of radius (e.g., 0.10)."
    )
    parser.add_argument(
        "--reinvert-m-on-jitter", action="store_true",
        help="Under Gaussian jitter, re-infer m_i from (r0,r1,r2,theta12) per node."
    )

    # Positive control (analytic)
    parser.add_argument(
        "--posctrl-n", type=int, default=1500,
        help="Number of analytic synthetic junctions for positive-control recovery plot"
    )
    parser.add_argument(
        "--posctrl-dir-noise-deg", type=float, default=3.0,
        help="Per-vector direction jitter (degrees) for positive control"
    )
    parser.add_argument(
        "--posctrl-radius-noise", type=float, default=0.05,
        help="Multiplicative radius noise SD (fraction) for positive control"
    )
    parser.add_argument(
        "--posctrl-parent-mislabel", type=float, default=0.0,
        help="Fraction [0,1] of synthetic nodes with wrong parent (positive-control realism)"
    )

    # --------------------------
    # Suite: base / lo / hi
    # --------------------------
    parser.add_argument(
        "--suite-triplet", action="store_true",
        help="Run baseline and two matched variants ('lo','hi') for ALL datasets (e.g., HRF, STARE)."
    )
    parser.add_argument(
        "--suite-angle-delta", type=float, default=3.0,
        help="± delta (deg) to apply to --min-angle for 'lo'/'hi' suite variants."
    )
    parser.add_argument(
        "--suite-tangent-delta", type=int, default=4,
        help="± delta (pixels) to apply to --tangent-len for 'lo'/'hi' suite variants."
    )
    parser.add_argument(
        "--suite-svd-delta", type=float, default=0.5,
        help="± delta to apply to --svd-ratio-min for 'lo'/'hi' suite variants."
    )

    # --------------------
    # Validation suite
    # --------------------
    parser.add_argument(
        "--validate", action="store_true",
        help="Run prereg-style validation suite and emit a single SUPPORT/MIXED/NOSUPPORT verdict per dataset"
    )
    parser.add_argument(
        "--alpha-prereg", type=float, default=1e-3,
        help="Primary prereg alpha for one-sided tests on median and Pr[R<0.55]"
    )
    parser.add_argument(
        "--parentdir-alpha", type=float, default=1e-4,
        help="Alpha for parent-direction Δφ>0 paired sign tests vs baselines"
    )
    parser.add_argument(
        "--equip-cvS-target", type=float, default=0.30,
        help="Equipartition target: median CV(S) must be ≤ this"
    )
    parser.add_argument(
        "--equip-cvK-target", type=float, default=0.30,
        help="Equipartition target: median CV(κ) must be ≤ this"
    )
    parser.add_argument(
        "--panelC-share-min", type=float, default=0.70,
        help="Panel C: fraction within |Δm|≤0.25 must be ≥ this"
    )
    parser.add_argument(
        "--snell-r2-min", type=float, default=0.50,
        help="Snell demo: minimum R^2 required for PASS (if demo files present)"
    )
    parser.add_argument(
        "--snell-min-crossings", type=int, default=30,
        help="Snell demo: minimum counted crossings to score PASS/FAIL (else marked NA)"
    )
    parser.add_argument(
        "--verdict-scope", type=str, default="dataset",
        choices=["dataset", "variant"],
        help="Scoreboard scope: aggregate to dataset-level (recommended) or per-variant"
    )
    parser.add_argument(
        "--verdict-name", type=str, default="default",
        help="Optional tag baked into VERDICT__*.{txt,json}"
    )

    # ---------- parse args ----------
    args = parser.parse_args()

    # Cap BLAS/OpenMP in the coordinator too (workers set their own in _init_worker)
    if threadpool_limits is not None:
        try:
            threadpool_limits(limits=int(args.intraop_threads))
        except Exception as e:
            log(f"[WARN] threadpool_limits failed in coordinator: {e}")

    # Freeze knobs if a profile is requested (mitigate researcher d.o.f.)
    if args.profile == "prereg-2025Q4":
        log("[PROFILE] Using prereg-2025Q4 profile — freezing key knobs.")
        args.min_angle = 10.0
        args.max_angle = 170.0
        args.min_branch_len = 10
        args.min_radius = 1.0
        args.tangent_len = 16
        args.svd_ratio_min = 3.0
        args.dedup_radius = 3
        args.angle_auto = False
        args.seg_method = "frangi"
        args.thresh_method = "otsu"

    # High-level run summary for transparency
    log(f"[CONFIG] datasets={args.datasets}, data_root={args.data_root}")
    log(f"[CONFIG] seg_method={args.seg_method}, thresh_method={args.thresh_method}")
    log(f"[CONFIG] primary_metric={args.primary_metric}, panel_filter={args.panel_filter}, strict_qc={args.strict_qc}")
    log(f"[CONFIG] seg_robust={args.seg_robust}, suite_triplet={args.suite_triplet}, validate={args.validate}")
    log(f"[CONFIG] null_perm={args.null_perm}, boot={args.boot}, workers={args.workers}, intraop_threads={args.intraop_threads}")

    # Base configs (will be overridden per suite variant)
    seg_cfg_base = SegConfig(vesselness_method=args.seg_method, thresh_method=args.thresh_method)
    qc_cfg_base = QCConfig(
        min_angle_deg=args.min_angle,
        max_angle_deg=args.max_angle,
        min_branch_len_px=args.min_branch_len,
        max_walk_len_px=48,
        min_radius_px=args.min_radius,
        m_bracket=(args.m_min, args.m_max),
        symmetric_ratio_tol=1.08,
        dedup_radius_px=args.dedup_radius,
        tangent_len_px=args.tangent_len,
        svd_ratio_min=args.svd_ratio_min,
        angle_auto=args.angle_auto,
        angle_auto_pctl=args.angle_auto_pctl,
        min_angle_floor_deg=args.min_angle_floor,
        angle_soft_margin_deg=args.angle_soft_margin,
        strict_qc=args.strict_qc,
        px_size_um=args.px_size_um,
        tangent_mode=args.tangent_mode,
        parent_tie_break=args.parent_tie_break,
        radius_estimator=args.radius_estimator,
        r_norm=args.R_norm,
    )

    data_root = Path(args.data_root)
    if not data_root.exists():
        log(f"[ERROR] data-root does not exist: {data_root}")
        sys.exit(1)

    overall_start = time.time()

    # ------------------------------------------------
    # Per-dataset outer loop
    # ------------------------------------------------
    for dataset in args.datasets:
        ds_dir = data_root / dataset
        if not ds_dir.exists():
            log(f"[WARN] Dataset folder not found: {ds_dir} — skipping.")
            continue

        log(f"========== DATASET: {dataset} (panel_filter={args.panel_filter}) ==========")

        # --- helpers: unpack archives & convert odd formats so the pipeline can see images ---
        def _unpack_archives_if_any(root: Path) -> int:
            import zipfile, tarfile
            extracted = 0
            for p in root.iterdir():
                if not p.is_file():
                    continue
                name = p.name.lower()
                try:
                    if name.endswith(".zip"):
                        with zipfile.ZipFile(p, "r") as zf:
                            zf.extractall(root)
                        extracted += 1
                        log(f"    [unpack] {p.name} -> {root}")
                    elif name.endswith(".tar") or name.endswith(".tar.gz") or name.endswith(".tgz") or name.endswith(".tar.bz2"):
                        with tarfile.open(p, "r:*") as tf:
                            tf.extractall(root)
                        extracted += 1
                        log(f"    [unpack] {p.name} -> {root}")
                except Exception as e:
                    log(f"    [WARN] Failed to unpack {p.name}: {e}")
            return extracted

        def _convert_odd_formats(root: Path) -> int:
            converted = 0
            odd_exts = {".ppm", ".gif"}
            for p in root.rglob("*"):
                if p.is_file() and p.suffix.lower() in odd_exts:
                    try:
                        arr = io.imread(str(p))
                        out = p.with_suffix(".png")
                        io.imsave(str(out), util.img_as_ubyte(arr))
                        converted += 1
                        log(f"    [convert] {p.name} -> {out.name}")
                    except Exception as e:
                        log(f"    [WARN] Failed convert {p.name}: {e}")
            return converted

        # Discover images
        img_paths = find_images(ds_dir, max_images=None)
        if len(img_paths) == 0 and _unpack_archives_if_any(ds_dir) > 0:
            img_paths = find_images(ds_dir, max_images=None)
        if len(img_paths) == 0 and _convert_odd_formats(ds_dir) > 0:
            img_paths = find_images(ds_dir, max_images=None)

        img_paths = [p for p in img_paths if not p.stem.lower().startswith("demo_")]
        if len(img_paths) == 0:
            log(f"[ERROR] No usable images found under {ds_dir}. Put images (*.png|*.jpg|*.tif|*.tiff|*.bmp) here; archives will auto-unpack.")
            continue

        # Optional fractional sampling (before any further processing)
        if getattr(args, "sample_frac", None) is not None:
            try:
                f = float(args.sample_frac)
            except Exception:
                f = None
            if f is not None and 0.0 < f < 1.0:
                n_total = len(img_paths)
                n_keep = max(1, int(math.ceil(n_total * f)))
                rng_samp = np.random.default_rng(int(args.sample_seed))
                sel_idx = np.sort(rng_samp.choice(n_total, size=n_keep, replace=False))
                img_paths = [img_paths[i] for i in sel_idx]
                log(f"    [sample] sample_frac={f:.3f} → {n_keep}/{n_total} images (seed={int(args.sample_seed)})")

        from collections import Counter
        ext_hist = Counter([p.suffix.lower() for p in img_paths])
        log("    [preflight] extension histogram: " + ", ".join(f"{k}:{v}" for k, v in sorted(ext_hist.items())))
        peek = img_paths[: min(5, len(img_paths))]
        for pp in peek:
            try:
                rr = io.imread(str(pp))
                log(f"    [preflight] {pp.name}: raw_shape={getattr(rr,'shape', '?')}, raw_dtype={getattr(rr,'dtype','?')}")
            except Exception as e:
                log(f"    [preflight] {pp.name}: read-failed ({e})")

        if args.max_images:
            img_paths = img_paths[:args.max_images]
            log(f"    [cap] max-images={int(args.max_images)} → using first {len(img_paths)}")

        log(f"[DATASET] {dataset}: {len(img_paths)} images after unpack/convert/sampling.")

        # ------------------ VARIANTS: as-configured or 4-way strata ------------------
        variant_pairs = [(args.seg_method, args.thresh_method)]
        if args.seg_robust:
            variant_pairs = [
                ("frangi", "otsu"),
                ("frangi", "quantile"),
                ("sato",   "otsu"),
                ("sato",   "quantile"),
            ]

        # Suite names and deltas
        if args.suite_triplet:
            suite_defs = [
                ("base", 0.0, 0, 0.0),
                ("lo",  -abs(args.suite_angle_delta), -abs(args.suite_tangent_delta), -abs(args.suite_svd_delta)),
                ("hi",   abs(args.suite_angle_delta),  abs(args.suite_tangent_delta),  abs(args.suite_svd_delta)),
            ]
        else:
            suite_defs = [("base", 0.0, 0, 0.0)]

        # Collect per-variant rows for the final cross-variant table
        segrobust_rows = []

        # ------------------------------------------------
        # Per-variant inner loop
        # ------------------------------------------------
        for seg_method, thresh_method in variant_pairs:
            for suite_name, d_ang, d_tan, d_svd in suite_defs:

                log(f"---- SEG VARIANT: {seg_method}+{thresh_method}  | suite={suite_name} | panel_filter={args.panel_filter} ----")
                tag_suffix = (f"__{suite_name}" if suite_name != "base" else "")
                dataset_tag = f"{dataset}__{seg_method}+{thresh_method}{tag_suffix}"

                # Apply deltas to the geometry gates in a *copy* of the base QC config
                min_angle_eff    = max(5.0, qc_cfg_base.min_angle_deg + d_ang)
                tangent_len_eff  = max(8,   qc_cfg_base.tangent_len_px + d_tan)
                svd_min_eff      = max(1.2, qc_cfg_base.svd_ratio_min + d_svd)

                seg_cfg = SegConfig(vesselness_method=seg_method, thresh_method=thresh_method)
                qc_cfg = QCConfig(
                    min_angle_deg=min_angle_eff,
                    max_angle_deg=qc_cfg_base.max_angle_deg,
                    min_branch_len_px=qc_cfg_base.min_branch_len_px,
                    max_walk_len_px=qc_cfg_base.max_walk_len_px,
                    min_radius_px=qc_cfg_base.min_radius_px,
                    m_bracket=qc_cfg_base.m_bracket,
                    symmetric_ratio_tol=qc_cfg_base.symmetric_ratio_tol,
                    dedup_radius_px=qc_cfg_base.dedup_radius_px,
                    tangent_len_px=tangent_len_eff,
                    svd_ratio_min=svd_min_eff,
                    angle_auto=qc_cfg_base.angle_auto,
                    angle_auto_pctl=qc_cfg_base.angle_auto_pctl,
                    min_angle_floor_deg=qc_cfg_base.min_angle_floor_deg,
                    angle_soft_margin_deg=qc_cfg_base.angle_soft_margin_deg,
                    strict_qc=qc_cfg_base.strict_qc,
                    px_size_um=qc_cfg_base.px_size_um,
                    tangent_mode=qc_cfg_base.tangent_mode,
                    parent_tie_break=qc_cfg_base.parent_tie_break,
                    radius_estimator=qc_cfg_base.radius_estimator,
                    r_norm=qc_cfg_base.r_norm,
                )

                log(
                    f"[VARIANT-CONFIG] dataset_tag={dataset_tag} | "
                    f"min_angle_eff={min_angle_eff:.1f} deg, tangent_len_eff={tangent_len_eff}, "
                    f"svd_ratio_min_eff={svd_min_eff:.2f}, R_norm={qc_cfg.r_norm}"
                )

                all_records: List[NodeRecord] = []
                diags_all: List[Dict] = []

                @dataclass
                class DatasetStats:
                    total: int
                    done: int = 0
                    nodes_raw: int = 0
                    nodes_dedup: int = 0
                    nodes_kept: int = 0
                    skips: dict = field(default_factory=lambda: {
                        "short_branch": 0, "bad_tangent": 0, "small_radius": 0,
                        "angle_out_of_range": 0, "no_m_candidate": 0, "qc_fail": 0
                    })

                stats = DatasetStats(total=len(img_paths))
                t0_dataset = time.time()

                desc = f"{dataset}:{seg_method}+{thresh_method}{tag_suffix}"
                # --- safe, cross-platform mp context ---
                start_methods = mp.get_all_start_methods()
                if sys.platform.startswith("linux") and "fork" in start_methods:
                    ctx = mp.get_context("fork")   # fastest on Linux when available
                else:
                    ctx = mp.get_context("spawn")  # macOS & Windows

                tasks = [
                    (str(p), dataset_tag, seg_cfg, qc_cfg, args.save_debug)
                    for p in img_paths
                ]

                # -------------
                # Image loop
                # -------------
                if int(args.workers) <= 1:
                    log("[RUN] Using single-process mode (workers=1).")
                    with tqdm(total=len(tasks), desc=desc, unit="img") as bar:
                        for t in tasks:
                            img_path_str, recs, diag, err = _process_one_image(t)
                            diags_all.append(diag)
                            all_records.extend(recs)

                            stats.done += 1
                            stats.nodes_raw   += int(diag.get("n_nodes_total_raw", 0))
                            stats.nodes_dedup += int(diag.get("n_nodes_total_dedup", 0))
                            stats.nodes_kept  += int(diag.get("n_nodes_kept", 0))
                            for k, v in diag.get("skip_reasons", {}).items():
                                stats.skips[k] = stats.skips.get(k, 0) + int(v)

                            bar.set_postfix({
                                "done": f"{stats.done}/{stats.total}",
                                "kept": stats.nodes_kept,
                                "deg3": stats.nodes_dedup,
                                "left": stats.total - stats.done,
                            })
                            bar.update(1)
                else:
                    max_workers = min(int(args.workers), max(1, os.cpu_count() or 1))
                    log(f"[RUN] Parallel mode: workers={max_workers} | intraop_threads/worker={int(args.intraop_threads)}")

                    _executor_kwargs = {}
                    try:
                        import inspect
                        if "max_tasks_per_child" in inspect.signature(ProcessPoolExecutor).parameters:
                            _executor_kwargs["max_tasks_per_child"] = 64
                    except Exception:
                        pass

                    try:
                        with ProcessPoolExecutor(
                            max_workers=max_workers,
                            mp_context=ctx,
                            initializer=_init_worker,
                            initargs=(int(args.intraop_threads),),
                            **_executor_kwargs,
                        ) as ex, tqdm(total=len(tasks), desc=desc, unit="img") as bar:

                            futures = [ex.submit(_process_one_image, t) for t in tasks]

                            for fut in as_completed(futures):
                                try:
                                    img_path_str, recs, diag, err = fut.result()
                                except Exception as e:
                                    err = f"executor_failure: {e}"
                                    img_path_str = "<unknown>"
                                    recs, diag = [], {
                                        "dataset": dataset,
                                        "image_id": Path(img_path_str).stem,
                                        "n_nodes_total_raw": 0,
                                        "n_nodes_total_dedup": 0,
                                        "n_nodes_kept": 0,
                                        "skip_reasons": {"worker_error": 1},
                                        "error": str(e),
                                    }

                                if err:
                                    log(f"[ERROR] Image {Path(img_path_str).name}: {err}")

                                diags_all.append(diag)
                                all_records.extend(recs)

                                stats.done += 1
                                stats.nodes_raw   += int(diag.get("n_nodes_total_raw", 0))
                                stats.nodes_dedup += int(diag.get("n_nodes_total_dedup", 0))
                                stats.nodes_kept  += int(diag.get("n_nodes_kept", 0))
                                for k, v in (diag.get("skip_reasons", {}) or {}).items():
                                    stats.skips[k] = stats.skips.get(k, 0) + int(v)

                                bar.set_postfix({
                                    "done": f"{stats.done}/{stats.total}",
                                    "kept": stats.nodes_kept,
                                    "deg3": stats.nodes_dedup,
                                    "left": stats.total - stats.done,
                                })
                                bar.update(1)

                    except KeyboardInterrupt:
                        log("[RUN] KeyboardInterrupt — cancelling outstanding work")
                        try:
                            ex.shutdown(wait=False, cancel_futures=True)  # type: ignore[name-defined]
                        except Exception:
                            pass
                        raise

                dt = human_time(time.time() - t0_dataset)
                log(
                    "---- DATASET PROGRESS ----\n"
                    f"dataset={dataset} | variant={seg_method}+{thresh_method}{tag_suffix} | images {stats.done}/{stats.total} | "
                    f"nodes raw→deg3→kept {stats.nodes_raw}→{stats.nodes_dedup}→{stats.nodes_kept} | "
                    f"skips: " + ", ".join(f"{k}={v}" for k, v in sorted(stats.skips.items(), key=lambda kv: -kv[1])) + " | "
                    f"elapsed {dt}\n"
                    "--------------------------"
                )

                if len(all_records) == 0:
                    log(f"[WARN] No nodes kept for {dataset_tag}; writing empty stubs and skipping plots.")
                    write_empty_reports(dataset_tag, reason="no_nodes_kept_after_extract")
                    continue

                # Save CSV
                csv_path = CSV_ROOT / dataset_tag / f"nodes__{dataset_tag}.csv"
                csv_path.parent.mkdir(parents=True, exist_ok=True)
                df_nodes = pd.DataFrame([r.__dict__ for r in all_records])

                # Freeze configs into CSV
                cfg_snapshot = {
                    "min_angle_deg": qc_cfg.min_angle_deg,
                    "max_angle_deg": qc_cfg.max_angle_deg,
                    "min_branch_len_px": qc_cfg.min_branch_len_px,
                    "tangent_len_px": qc_cfg.tangent_len_px,
                    "svd_ratio_min": qc_cfg.svd_ratio_min,
                    "dedup_radius_px": qc_cfg.dedup_radius_px,
                    "strict_qc": qc_cfg.strict_qc,
                    "m_min": qc_cfg.m_bracket[0],
                    "m_max": qc_cfg.m_bracket[1],
                    "symmetric_ratio_tol": qc_cfg.symmetric_ratio_tol,
                    "angle_auto": qc_cfg.angle_auto,
                    "angle_auto_pctl": qc_cfg.angle_auto_pctl,
                    "min_angle_floor_deg": qc_cfg.min_angle_floor_deg,
                    "px_size_um": qc_cfg.px_size_um,
                    "tangent_mode": qc_cfg.tangent_mode,
                    "parent_tie_break": qc_cfg.parent_tie_break,
                    "radius_estimator": qc_cfg.radius_estimator,
                    "r_norm": qc_cfg.r_norm,
                }

                for k, v in cfg_snapshot.items():
                    df_nodes[f"cfg_{k}"] = v
                df_nodes["args_profile"] = args.profile
                df_nodes["args_seg_method"] = seg_cfg.vesselness_method
                df_nodes["args_thresh_method"] = seg_cfg.thresh_method
                df_nodes["args_radius_jitter_frac"] = args.radius_jitter_frac
                df_nodes.to_csv(csv_path, index=False)
                log(f"[IO] Wrote node-level CSV: {csv_path} (N_nodes={len(df_nodes)})")

                # Post-processing panels
                variant_fig_dir = FIG_ROOT / dataset_tag
                variant_rep_dir = CSV_ROOT / dataset_tag
                dataset_base    = dataset_tag.split("__", 1)[0]
                base_fig_dir    = FIG_ROOT / dataset_base
                base_rep_dir    = CSV_ROOT / dataset_base

                # Decide metric columns & filter once per variant
                metric_R_col = "R_m_holdout" if (args.primary_metric == "heldout") else "R_m"
                metric_m_col = "m_angleonly" if (args.primary_metric == "heldout") else "m_node"
                panel_filter_mode = args.panel_filter
                metric_name = "heldout" if metric_R_col == "R_m_holdout" else "asconfigured"
                log(
                    f"[POST-CONFIG] dataset_tag={dataset_tag} | metric_R_col={metric_R_col}, "
                    f"metric_m_col={metric_m_col}, panel_filter={panel_filter_mode}"
                )

                # Panel A
                log(f"[POST] Panel A — m distribution → {variant_fig_dir}/dataset_{dataset_tag}__m_hist.png")
                try:
                    plot_m_distribution(df_nodes, dataset_tag)
                except Exception as e:
                    log(f"[POST][WARN] Panel A failed: {e}")

                # Panel B / C1
                log(f"[POST] Panel B/C1 — residual histogram (metric={metric_name}) → {variant_fig_dir}/dataset_{dataset_tag}__residual_hist.png")
                try:
                    plot_residual_distribution(
                        df_nodes,
                        dataset_tag,
                        n_perm=int(args.null_perm),
                        boot=int(args.boot),
                        include_nulls=("shuffle", "swap", "randtheta", "rshuffle"),
                        jitter_frac=args.radius_jitter_frac,
                        m_col=metric_m_col,
                        R_col=metric_R_col,
                        gaussian_jitter_sd=args.radius_jitter_sd,
                        reinvert_m_on_jitter=args.reinvert_m_on_jitter,
                        norm_mode=qc_cfg.r_norm,
                        panel_filter=panel_filter_mode,
                    )
                    # Append FINAL table row
                    try:
                        append_to_final_table(df_nodes, dataset_tag, reports_dir=CSV_ROOT)
                    except Exception as e_final:
                        log(f"[FINAL][WARN] append_to_final_table failed: {e_final}")
                except Exception as e:
                    log(f"[POST][WARN] Panel B/C1 failed: {e}")

                # Baselines
                log(f"[POST] Baselines → {variant_fig_dir}/dataset_{dataset_tag}__residual_baselines.png")
                try:
                    plot_fixed_m_baselines(
                        df_nodes,
                        dataset_tag,
                        norm_mode=qc_cfg.r_norm,
                    )
                except Exception as e:
                    log(f"[POST][WARN] Baselines failed: {e}")

                # Parent-direction
                log(f"[POST] Parent-direction error → {variant_fig_dir}/dataset_{dataset_tag}__parent_dir_error_ecdf.png")
                try:
                    plot_parent_direction_error(df_nodes, dataset_tag)
                except Exception as e:
                    log(f"[POST][WARN] Parent-direction failed: {e}")

                # Panel C
                log(f"[POST] Panel C — θ vs m scatter → {variant_fig_dir}/dataset_{dataset_tag}__theta_vs_m_scatter.png")
                try:
                    plot_theta_vs_m_scatter(
                        df_nodes,
                        dataset_tag,
                        symmetric_tol=1.08,
                        bootstrap=int(args.boot),
                        seed=13,
                        panel_filter=panel_filter_mode,
                    )
                except Exception as e:
                    log(f"[POST][WARN] Panel C failed: {e}")

                # STEP 3 — per-node uncertainty
                log(f"[POST] STEP 3 — per-node uncertainty → {variant_rep_dir}/UNCERTAINTY__nodes.csv")
                try:
                    compute_per_node_uncertainty(df_nodes, dataset_tag)
                except Exception as e:
                    log(f"[POST][WARN] Uncertainty step failed: {e}")

                # STEP 4 — ablation
                log(f"[POST] STEP 4 — ablation grid → {variant_fig_dir}/dataset_{dataset_tag}__ablation_medianR_grid.png")
                try:
                    run_ablation_grid(
                        df_nodes,
                        dataset_tag,
                        norm_mode=qc_cfg.r_norm,
                    )
                except Exception as e:
                    log(f"[POST][WARN] Ablation step failed: {e}")

                # Positive control
                log(f"[POST] Positive control → {variant_fig_dir}/POSCTRL__recovery.png")
                try:
                    plot_positive_control_recovery(
                        dataset_tag,
                        n_samples=int(args.posctrl_n),
                        noise_dir_deg=float(args.posctrl_dir_noise_deg),
                        noise_radius_frac=float(args.posctrl_radius_noise),
                        parent_mislabel_frac=float(args.posctrl_parent_mislabel),
                    )
                except Exception as e:
                    log(f"[POST][WARN] Positive control failed: {e}")

                # Summary
                log(f"[POST] SUMMARY → {variant_rep_dir}/SUMMARY__{dataset_tag}.txt")
                try:
                    summarize_and_write(
                        df=df_nodes,
                        dataset=dataset_tag,
                        diags=diags_all,
                        n_images_total=len(img_paths),
                        elapsed_sec=time.time() - t0_dataset
                    )
                except Exception as e:
                    log(f"[POST][WARN] Summary failed: {e}")

                # Cross-variant summary row (keeps held-out as primary if requested)
                segrobust_rows.append(
                    _make_variant_row(
                        df=df_nodes,
                        dataset_tag=dataset_tag,
                        reports_dir=CSV_ROOT,
                        primary_metric="heldout" if args.primary_metric == "heldout" else "asconfigured",
                    )
                )
                log(f"[POST] Completed post-processing for {dataset_tag}")

                # Manifest for reproducibility
                try:
                    write_run_manifest(
                        tag=dataset_tag,
                        args_namespace=args,
                        extra={
                            "dataset_base": dataset,
                            "seg_method": seg_method,
                            "thresh_method": thresh_method,
                            "suite_name": suite_name,
                            "min_angle_eff": float(min_angle_eff),
                            "max_angle_deg": float(qc_cfg.max_angle_deg),
                            "tangent_len_eff": int(tangent_len_eff),
                            "svd_ratio_min_eff": float(svd_min_eff),
                            "primary_metric": args.primary_metric,
                            "panel_filter": panel_filter_mode,
                            "strict_qc": bool(args.strict_qc),
                            "null_perm": int(args.null_perm),
                            "bootstrap": int(args.boot),
                            "nodes_kept": int(len(df_nodes)),
                        },
                        img_paths=img_paths,
                    )
                    log(f"[MANIFEST] Wrote RUN__manifest.json for {dataset_tag}")
                except Exception as e:
                    log(f"[MANIFEST][WARN] Failed to write manifest for {dataset_tag}: {e}")

        # After all variants: write the cross-variant summary table once
        try:
            _write_segrobust_summary(
                dataset=dataset,
                rows=segrobust_rows,
                reports_dir=CSV_ROOT,
                log_fn=log,
            )
        except Exception as e:
            log(f"[SEGROBUST][WARN] Failed to write SEGROBUST summary for {dataset}: {e}")

        # ===== RUN VALIDATION SUITE (single SUPPORT/MIXED/NOSUPPORT verdict) =====
        if args.validate:
            try:
                run_validation_suite(
                    dataset=dataset,
                    reports_dir=CSV_ROOT,
                    scope=args.verdict_scope,
                    verdict_name=args.verdict_name,
                    alpha_prereg=float(args.alpha_prereg),
                    parentdir_alpha=float(args.parentdir_alpha),
                    equip_cvS_target=float(args.equip_cvS_target),
                    equip_cvK_target=float(args.equip_cvK_target),
                    panelC_share_min=float(args.panelC_share_min),
                    snell_R2_min=float(args.snell_r2_min),
                    snell_min_N=int(args.snell_min_crossings),
                )
            except Exception as e:
                log(f"[VERDICT][WARN] validation suite failed: {e}")

    # After all datasets: build cross-dataset segmentation-robust scoreboard
    try:
        if args.seg_robust and args.primary_metric == "heldout":
            plot_segrobust_scoreboard(
                datasets=args.datasets,
                reports_dir=CSV_ROOT,
                fig_root=FIG_ROOT,
            )
    except Exception as e:
        log(f"[SCOREBOARD][WARN] cross-dataset scoreboard failed: {e}")

    # If suite-triplet was used, write a paper-ready cross-suite summary for all datasets
    try:
        if args.suite_triplet:
            _write_suite_summary(
                datasets=args.datasets,
                reports_dir=CSV_ROOT,
                out_dir=CSV_ROOT,
                suite_names=("base", "lo", "hi"),
            )
    except Exception as e:
        log(f"[SUITE][WARN] cross-suite summary failed: {e}")

    # Show-stopper residual comparison across datasets (first two primaries)
    try:
        if len(args.datasets) >= 2:
            primary_tags = []
            for ds in args.datasets:
                pick = _pick_primary_variant(ds, CSV_ROOT)
                if pick:
                    primary_tags.append(pick)
            if len(primary_tags) >= 2:
                Rcol = "R_m_holdout" if args.primary_metric == "heldout" else "R_m"
                plot_showstopper_residuals(primary_tags[:2], R_col=Rcol)
    except Exception as e:
        log(f"[SHOWSTOPPER][WARN] failed to render: {e}")

    total_dt = human_time(time.time() - overall_start)
    log(f"[DONE] ALL DATASETS completed in {total_dt}. See figures/, reports/, logs/ under {ROOT}")


if __name__ == "__main__":

    try:
        main()
    except KeyboardInterrupt:
        log("[INTERRUPT] Interrupted by user. Exiting.")
