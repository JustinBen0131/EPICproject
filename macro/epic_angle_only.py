#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

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


def log(msg: str):
    """
    Console + file logger (simple, robust, flushed).
    """
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    out = f"[{ts}] {msg}"
    print(out, flush=True)
    with open(LOG_ROOT / "run.log", "a") as f:
        f.write(out + "\n")
        f.flush()


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
                     m: float) -> float:
    """
    R(m) = || r0^m e0 + r1^m e1 + r2^m e2 || / (r1^m + r2^m)
    """
    a0 = (r0 ** m) * e0
    a1 = (r1 ** m) * e1
    a2 = (r2 ** m) * e2
    num = np.linalg.norm(a0 + a1 + a2)
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

    


# ---------- PIPELINE ----------
@dataclass
class QCConfig:
    min_angle_deg: float = 15.0
    max_angle_deg: float = 170.0
    min_branch_len_px: int = 10
    max_walk_len_px: int = 48
    min_radius_px: float = 1.0
    m_bracket: Tuple[float, float] = (0.2, 4.0)
    symmetric_ratio_tol: float = 1.08  # r1/r2 within this factor => "symmetric"

    # NEW: geometry stabilization + gating
    dedup_radius_px: int = 3                 # cluster degree-3 pixels within this radius
    tangent_len_px: int = 12                 # pixels used to fit branch tangents from junction
    svd_ratio_min: float = 1.6               # PCA quality: S0/S1 must be >= this
    angle_auto: bool = False                 # enable per-image auto lower angle bound
    angle_auto_pctl: float = 5.0             # percentile for auto lower bound
    min_angle_floor_deg: float = 8           # floor for auto lower bound (deg)
    max_tangent_wander_deg: float = 25.0     # NEW: cosine-consistency gate over first ~8 points
    angle_soft_margin_deg: float = 3.0       # NEW: allow rescuing nodes within this many degrees outside the gate

    # NEW: stricter QC reporting
    strict_qc: bool = False                  # compute/report strict QC in addition to loose

    # NEW: metadata (recorded in CSV; does not affect m)
    px_size_um: float = float("nan")         # optional pixel size (µm/px)




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
        raw_dtype  = str(raw.dtype)
        raw_ext    = img_path.suffix.lower()
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

    total_px   = int(mask.size)
    vessel_px  = int(mask.sum())
    vessel_pct = (vessel_px / max(1, total_px)) * 100.0
    skel_px    = int(skel.sum())
    deg_hist   = np.bincount([G.degree[n] for n in G.nodes()], minlength=7)
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
            angle_min_this_image = max(qc.min_angle_floor_deg, float(np.percentile(angles_deg, qc.angle_auto_pctl)))
        else:
            angle_min_this_image = qc.min_angle_floor_deg
        log(f"    Angle gate (auto): min={angle_min_this_image:.1f}° (floor={qc.min_angle_floor_deg}°, p{qc.angle_auto_pctl} of {len(angles_deg)} candidates)")


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
    QC_RM_MAX_LOOSE   = 0.85
    QC_CRES_MAX_LOOSE = 0.35
    QC_RM_MAX_STRICT   = 0.55 if qc.strict_qc else QC_RM_MAX_LOOSE
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
        branch_dirs  = []
        branch_radii = []
        branch_svdratios = []

        valid = True

        def _greedy_extend(skel_arr: np.ndarray, start_rc: Tuple[int,int], init_vec: np.ndarray, steps: int = 12) -> List[Tuple[int,int]]:
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

        def _tangent_consistency_ok(points_rc: List[Tuple[int,int]],
                                    origin_rc: Tuple[int,int],
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
                ext = _greedy_extend(skel, pts[-1], init_vec, steps=max(8, qc.min_branch_len_px - path_len))
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

            # Tangent over fixed arc + PCA quality (with robust fallback)
            pts_for_tan = pts[:qc.tangent_len_px]
            dvec, sratio = fit_tangent(pts_for_tan, origin_rc=rc0)

            def _chord_direction(points_rc, origin_rc):
                # use the farthest available point within the window as a chord
                j = min(len(points_rc) - 1, max(8, qc.tangent_len_px) - 1)
                yy, xx = points_rc[j]
                oy, ox = origin_rc
                v = np.array([float(xx - ox), float(yy - oy)], dtype=np.float32)
                n = np.linalg.norm(v)
                return (v / (n + 1e-12)) if n > 0 else np.array([np.nan, np.nan], dtype=np.float32)

            bad = (not np.all(np.isfinite(dvec))) or (np.linalg.norm(dvec) < 0.5) or (sratio < qc.svd_ratio_min) \
                  or (not _tangent_consistency_ok(pts_for_tan, rc0, deg_thresh=qc.max_tangent_wander_deg, k_first=8))
            if bad:
                # Fallback: accept a chord direction if the branch is long enough AND consistent
                if path_len >= max(8, qc.tangent_len_px // 2):
                    dvec_fb = _chord_direction(pts_for_tan, rc0)
                    if np.all(np.isfinite(dvec_fb)) and np.linalg.norm(dvec_fb) >= 0.5 \
                       and _tangent_consistency_ok(pts_for_tan, rc0, deg_thresh=qc.max_tangent_wander_deg, k_first=8):
                        dvec = dvec_fb
                        sratio = max(sratio, qc.svd_ratio_min)
                        log(f"      [fallback] using chord direction; len={path_len}, svd_ratio≈{sratio:.2f}")
                    else:
                        valid = False
                        reasons["bad_tangent"] += 1
                        if save_debug:
                            draw_point(*rc0, [1.0, 0.0, 0.0])  # red
                        log(f"      [drop] bad/unstable tangent; len={path_len}, svd_ratio={sratio:.2f}")
                        continue
                else:
                    valid = False
                    reasons["bad_tangent"] += 1
                    if save_debug:
                        draw_point(*rc0, [1.0, 0.0, 0.0])  # red
                    log(f"      [drop] bad tangent (too short for fallback); len={path_len}, svd_ratio={sratio:.2f}")
                    continue


            # Robust radius
            rad = branch_radius(dist, pts, k=min(8, max(3, path_len)))
            if not np.isfinite(rad) or rad < qc.min_radius_px:
                valid = False
                reasons["small_radius"] += 1
                if save_debug:
                    draw_point(*rc0, [1.0, 0.0, 0.0])  # red
                log(f"      [drop] small radius; r≈{rad:.2f}px, len={path_len}")
                continue


            # Soft accept short-but-usable branches: keep them but log for transparency
            if path_len < qc.min_branch_len_px:
                log(f"      [soft] short branch accepted: len={path_len}px < min={qc.min_branch_len_px}px; r≈{rad:.2f}px")

            branch_paths.append(pts)
            branch_dirs.append(dvec)
            branch_radii.append(rad)
            branch_svdratios.append(float(sratio))

        if not valid or len(branch_dirs) != 3:
            # already counted reason above
            continue

        # 3b) Parent/daughter assignment by direction opposition (primary)
        vecs = np.stack(branch_dirs, axis=0)  # (3,2)
        scores = []
        for i in range(3):
            others = vecs[[j for j in range(3) if j != i]]
            s = others[0] + others[1]
            s_norm = s / (np.linalg.norm(s) + 1e-12)
            scores.append(float(np.dot(vecs[i], s_norm)))
        idx_parent_primary = int(np.argmin(scores))  # most opposite to the other two
        idx_daughters_primary = [j for j in range(3) if j != idx_parent_primary]

        # Alternate assignment: swap parent with the fatter daughter (conservative cross-check)
        radii_arr = np.array(branch_radii, dtype=float)
        idx_fat_daughter = int(idx_daughters_primary[int(np.argmax(radii_arr[idx_daughters_primary]))])
        idx_parent_alt = idx_fat_daughter
        idx_daughters_alt = [j for j in range(3) if j != idx_parent_alt]

        def _solve_for_assignment(i_par: int, i_daus: List[int]):
            ee0, ee1, ee2 = branch_dirs[i_par], branch_dirs[i_daus[0]], branch_dirs[i_daus[1]]
            rr0, rr1, rr2 = float(branch_radii[i_par]), float(branch_radii[i_daus[0]]), float(branch_radii[i_daus[1]])
            th12 = angle_between(ee1, ee2)
            deg12_ = math.degrees(th12)

            # Soft rescue: if θ12 is within a small margin outside the gate, clamp to the nearest bound
            used_deg12 = deg12_
            soft_used = False
            if not (angle_min_this_image <= deg12_ <= qc.max_angle_deg):
                if (angle_min_this_image - float(qc.angle_soft_margin_deg)) <= deg12_ <= (qc.max_angle_deg + float(qc.angle_soft_margin_deg)):
                    used_deg12 = float(np.clip(deg12_, angle_min_this_image, qc.max_angle_deg))
                    th12 = math.radians(used_deg12)
                    soft_used = True
                else:
                    return None  # gated out

            m_candidates, m_source = [], []

            # Solver
            m_sol = m_from_node(rr0, rr1, rr2, th12,
                                m_min=qc.m_bracket[0], m_max=qc.m_bracket[1],
                                tol=1e-6, iters=64)
            if np.isfinite(m_sol) and qc.m_bracket[0] <= m_sol <= qc.m_bracket[1]:
                m_candidates.append(float(m_sol)); m_source.append("solver")

            # Symmetric (if daughters comparable)
            ratio = max(rr1, rr2) / max(1e-9, min(rr1, rr2))
            if ratio <= qc.symmetric_ratio_tol:
                m_sym = m_from_angle_symmetric(th12, n=4.0)
                if np.isfinite(m_sym) and qc.m_bracket[0] <= m_sym <= qc.m_bracket[1]:
                    m_candidates.append(float(m_sym)); m_source.append("symmetric")

            # Grid minimizer of |equation residual|
            def eq_res(mm: float) -> float:
                return abs((rr0**(2.0*mm)) - (rr1**(2.0*mm)) - (rr2**(2.0*mm)) - 2.0*math.cos(th12)*(rr1**mm)*(rr2**mm))
            grid = np.linspace(qc.m_bracket[0], qc.m_bracket[1], 121, dtype=np.float64)
            vals = np.array([eq_res(mm) for mm in grid])
            m_grid = float(grid[int(np.argmin(vals))])
            if np.isfinite(m_grid) and qc.m_bracket[0] <= m_grid <= qc.m_bracket[1]:
                m_candidates.append(m_grid); m_source.append("grid")

            if len(m_candidates) == 0:
                return None

            best_idx, best_R = -1, np.inf
            for ii, m_c in enumerate(m_candidates):
                R_c = closure_residual(rr0, rr1, rr2, ee0, ee1, ee2, m_c)
                if R_c < best_R:
                    best_R = R_c; best_idx = ii
            m_fin = float(m_candidates[best_idx]); src_fin = str(m_source[best_idx])
            Rm = closure_residual(rr0, rr1, rr2, ee0, ee1, ee2, m_fin)
            c_vec, cresid = tariffs_from_nullspace(rr0, rr1, rr2, ee0, ee1, ee2, m_fin)

            qc_pass_loose  = ((Rm < QC_RM_MAX_LOOSE) or (cresid < QC_CRES_MAX_LOOSE))
            qc_pass_strict = ((Rm < QC_RM_MAX_STRICT) and (cresid < QC_CRES_MAX_STRICT))
            return {
                "e0": ee0, "e1": ee1, "e2": ee2, "r0": rr0, "r1": rr1, "r2": rr2,
                "theta12_deg": deg12_, "m": m_fin, "Rm": Rm, "c": c_vec, "cresid": cresid,
                "qc_loose": qc_pass_loose, "qc_strict": qc_pass_strict, "best_R": best_R,
                "m_source": src_fin, "soft_clamp": soft_used, "theta12_used_deg": used_deg12
            }

        res_primary = _solve_for_assignment(idx_parent_primary, idx_daughters_primary)
        res_alt     = _solve_for_assignment(idx_parent_alt, idx_daughters_alt)

        if (res_primary is None) and (res_alt is None):
            reasons["angle_out_of_range"] += 1  # both gated out
            if save_debug:
                draw_point(*rc0, [1.0, 0.0, 1.0])
            continue

        # If both exist and disagree on who is parent, keep the WORSE R(m) (conservative)
        parent_ambiguous = (idx_parent_primary != idx_parent_alt) and (res_primary is not None) and (res_alt is not None)
        chosen = None
        if res_primary is not None and res_alt is not None:
            chosen = res_primary if res_primary["Rm"] >= res_alt["Rm"] else res_alt
        else:
            chosen = res_primary if res_primary is not None else res_alt

        e0, e1, e2 = chosen["e0"], chosen["e1"], chosen["e2"]
        r0, r1, r2 = chosen["r0"], chosen["r1"], chosen["r2"]
        deg12, m, Rm = chosen["theta12_deg"], chosen["m"], chosen["Rm"]
        c_vec, cresid = chosen["c"], chosen["cresid"]
        qc_pass_loose, qc_pass_strict = chosen["qc_loose"], chosen["qc_strict"]
        best_R, m_src = chosen["best_R"], chosen["m_source"]
        
        # --- HELD-OUT angle-only m (no directions used to choose m) ---
        m_ao, _ao_src = angle_only_m_deterministic(
            r0, r1, r2, math.radians(deg12),
            m_min=qc.m_bracket[0], m_max=qc.m_bracket[1],
            symmetric_ratio_tol=qc.symmetric_ratio_tol
        )
        Rm_ao = closure_residual(r0, r1, r2, e0, e1, e2, m_ao) if np.isfinite(m_ao) else float("nan")


        # map chosen e-vectors back to branch indices to get SVD ratios
        def _idx_of_vec(vec):
            dists = [np.linalg.norm(np.asarray(bv) - np.asarray(vec)) for bv in branch_dirs]
            return int(np.argmin(dists)) if len(dists) == 3 else 0
        i0 = _idx_of_vec(e0); i1 = _idx_of_vec(e1); i2 = _idx_of_vec(e2)
        svd0 = float(branch_svdratios[i0]) if len(branch_svdratios) == 3 else float("nan")
        svd1 = float(branch_svdratios[i1]) if len(branch_svdratios) == 3 else float("nan")
        svd2 = float(branch_svdratios[i2]) if len(branch_svdratios) == 3 else float("nan")

        node_records.append(NodeRecord(
            dataset=dataset,
            image_id=image_id,
            node_id=node_counter,
            yx=(int(rc0[0]), int(rc0[1])),
            r0=float(r0), r1=float(r1), r2=float(r2),
            theta12_deg=float(deg12),
            m_node=float(m),
            R_m=float(Rm),
            c0=float(c_vec[0]), c1=float(c_vec[1]), c2=float(c_vec[2]),
            tariff_residual=float(cresid),
            qc_pass=qc_pass_loose,
            qc_pass_strict=qc_pass_strict,
            e0x=float(e0[0]), e0y=float(e0[1]),
            e1x=float(e1[0]), e1y=float(e1[1]),
            e2x=float(e2[0]), e2y=float(e2[1]),
            svd_ratio_e0=svd0, svd_ratio_e1=svd1, svd_ratio_e2=svd2,
            seg_variant=str(seg_meta.get("seg_variant", "unknown")),
            seg_thresh_type=str(seg_meta.get("thresh_type", "")),
            seg_thresh_value=float(seg_meta.get("thresh_value", float("nan"))),
            m_source_chosen=str(m_src),
            parent_ambiguous=bool(parent_ambiguous),
            px_size_um=float(qc.px_size_um),
            note=f"m_sources_evaluated=['solver','symmetric','grid']" + ("; soft_angle_clamp" if bool(chosen.get('soft_clamp', False)) else ""),
            m_angleonly=float(m_ao),
            R_m_holdout=float(Rm_ao),
        ))


        node_counter += 1

        # 3f) Debug overlay color by QC (green=strict pass, orange=loose-only pass, red=fail)
        if save_debug:
            color_rgb = [0.0, 1.0, 0.0] if qc_pass_strict else ([1.0, 0.65, 0.0] if qc_pass_loose else [1.0, 0.0, 0.0])
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
            "loose":  {"R_m_max": QC_RM_MAX_LOOSE,  "c_res_max": QC_CRES_MAX_LOOSE},
            "strict": {"R_m_max": QC_RM_MAX_STRICT, "c_res_max": QC_CRES_MAX_STRICT},
        },
    }

    # Transparent summary for this image
    log(f"    Image summary: kept={kept}/{len(junction_nodes)} | "
        f"skips: short_branch={reasons['short_branch']}, bad_tangent={reasons['bad_tangent']}, "
        f"small_radius={reasons['small_radius']}, angle_out={reasons['angle_out_of_range']}, "
        f"no_m={reasons['no_m_candidate']}, qc_fail={reasons['qc_fail']}")

    # 5) Optional overlay save
    if save_debug and overlay is not None:
        fig = plt.figure(figsize=(10, 10))
        plt.imshow(overlay)
        plt.axis('off')
        plt.title(f"{dataset}/{image_id} — junction QC overlay (kept {kept})")
        save_png(fig, (out_dir / "figures" / dataset / f"{image_id}__skeleton_overlay.png"))

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
    # NEW: prereg routing and jitter behavior
    m_col: str = "m_node",            # e.g., "m_angleonly"
    R_col: str = "R_m",               # e.g., "R_m_holdout"
    gaussian_jitter_sd: float = 0.0,  # SD as fraction of radius (indep. Gaussian)
    reinvert_m_on_jitter: bool = False,
):
    """
    Panel B — Residuals with uncertainty and rich controls.

    When m_col/R_col point to the angle-only fields ("m_angleonly", "R_m_holdout"),
    this function implements the preregistered held-out metric and writes
    reports/<dataset>/PREREG__C1.txt (PASS/FAIL), using the within-image direction-shuffle null.

    Also supports Gaussian radius jitter (SD = fraction × radius) with optional
    re-inversion of m_i from (r0,r1,r2,theta12) under jitter.
    """
    # Back-compat: if gaussian_jitter_sd not provided but jitter_frac was passed,
    # route jitter_frac into gaussian_jitter_sd so the dotted median is drawn.
    if float(gaussian_jitter_sd) <= 0.0 and float(jitter_frac) > 0.0:
        gaussian_jitter_sd = float(jitter_frac)

    required_cols = {"e0x","e0y","e1x","e1y","e2x","e2y","image_id","r0","r1","r2","theta12_deg"}

    have_dirs = required_cols.issubset(set(df.columns))

    # Choose observed column(s)
    R_obs = (df[R_col].astype(float).values if R_col in df.columns else df["R_m"].astype(float).values)
    R_obs = R_obs[np.isfinite(R_obs)]
    obs_median = float(np.median(R_obs)) if R_obs.size else float("nan")
    obs_frac055 = float(np.mean(R_obs < 0.55)) if R_obs.size else float("nan")
    # collect distributions for percentile-based x-limit later
    xlim_collect = [R_obs]

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

    fig = plt.figure(figsize=(7.5, 5.0))
    ax = plt.gca()

    # Professional styling
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.25, linestyle="-", linewidth=0.8)
    ax.minorticks_on()
    ax.tick_params(axis="both", which="both", direction="out")

    # Adaptive binning for smoother density on large N
    bins = min(80, max(30, int(np.ceil(np.sqrt(max(1, R_obs.size))))))
    ax.hist(R_obs, bins=bins, density=True, alpha=0.85, label=f"observed (N={R_obs.size})")

    # Quality bands and thresholds
    ax.axvspan(0.0, 0.55, alpha=0.10)
    ax.axvspan(0.55, 0.85, alpha=0.07)
    ax.axvline(0.55, linestyle="--", linewidth=1.0)
    ax.axvline(0.85, linestyle="--", linewidth=1.0)

    # Observed median and 95% CI — lines plus anchored label that cannot run off-canvas
    ax.axvline(obs_median, linewidth=2.0)
    ax.axvline(ci_lo, linestyle=":", linewidth=1.2)
    ax.axvline(ci_hi, linestyle=":", linewidth=1.2)
    ax.annotate(
        f"median = {obs_median:.3f}\n95% CI [{ci_lo:.3f}, {ci_hi:.3f}]",
        xy=(0.015, 0.98), xycoords="axes fraction",
        ha="left", va="top", fontsize=9,
        bbox=dict(facecolor="white", alpha=0.90, edgecolor="none", boxstyle="round,pad=0.25"),
    )

    ax.set_xlabel("Closure residual R(m)")
    ax.set_ylabel("Density")
    ax.set_title(f"Closure residual R(m) — {dataset}")

    if not have_dirs:
        ax.set_xlabel("closure residual R(m)")
        ax.set_ylabel("count")
        ax.set_title(f"Closure residual R(m) — {dataset}")
        ax.legend(loc="best")
        save_png(fig, FIG_ROOT / dataset / f"dataset_{dataset}__residual_hist.png")
        log("[NULL-TEST] Direction columns not found; plotted observed histogram with bootstrap CI only.")
        return

    rng = np.random.default_rng(int(seed))

    def _norm_rows(V: np.ndarray) -> np.ndarray:
        n = np.linalg.norm(V, axis=1, keepdims=True) + 1e-12
        return V / n

    # Accept scalar or per-node m
    def _closure_residual_batch(r0, r1, r2, e0, e1, e2, m):
        m_arr = np.asarray(m, dtype=float)
        if m_arr.ndim == 0:
            a0 = (np.power(r0, m_arr))[:, None] * e0
            a1 = (np.power(r1, m_arr))[:, None] * e1
            a2 = (np.power(r2, m_arr))[:, None] * e2
            num = np.linalg.norm(a0 + a1 + a2, axis=1)
            denom = (np.power(r1, m_arr)) + (np.power(r2, m_arr)) + 1e-12
            return num / denom
        else:
            a0 = (np.power(r0, m_arr))[:, None] * e0
            a1 = (np.power(r1, m_arr))[:, None] * e1
            a2 = (np.power(r2, m_arr))[:, None] * e2
            num = np.linalg.norm(a0 + a1 + a2, axis=1)
            denom = (np.power(r1, m_arr)) + (np.power(r2, m_arr)) + 1e-12
            return num / denom

    # Arrays
    r0a = df["r0"].astype(float).values
    r1a = df["r1"].astype(float).values
    r2a = df["r2"].astype(float).values
    ma  = df[m_col].astype(float).values if m_col in df.columns else df["m_node"].astype(float).values
    e0a = df[["e0x","e0y"]].astype(float).values
    e1a = df[["e1x","e1y"]].astype(float).values
    e2a = df[["e2x","e2y"]].astype(float).values
    t12 = np.deg2rad(df["theta12_deg"].astype(float).values)
    N   = len(r0a)

    # Build direction pools (within-image)
    def _stack_dirs(frame: pd.DataFrame) -> np.ndarray:
        v0 = frame[["e0x","e0y"]].values
        v1 = frame[["e1x","e1y"]].values
        v2 = frame[["e2x","e2y"]].values
        return _norm_rows(np.vstack([v0, v1, v2]).astype(np.float64))

    pools = {img_id: _stack_dirs(g) for img_id, g in df.groupby("image_id")}
    dataset_pool = np.vstack([p for p in pools.values() if p.size > 0]) if len(pools) else np.zeros((0,2), dtype=np.float64)
    img_ids = df["image_id"].astype(str).values

    # Nulls
    def _null_shuffle():
        med, frac, agg = np.empty(int(n_perm)), np.empty(int(n_perm)), []
        K_agg = int(min(50, n_perm))
        for b in range(int(n_perm)):
            e0p = np.empty_like(e0a); e1p = np.empty_like(e1a); e2p = np.empty_like(e2a)
            for i, img_id in enumerate(img_ids):
                pool = pools.get(img_id, dataset_pool)
                if pool.shape[0] < 3: pool = dataset_pool
                idxs = rng.integers(0, pool.shape[0], size=3)
                e0p[i], e1p[i], e2p[i] = pool[idxs[0]], pool[idxs[1]], pool[idxs[2]]
            Rb = _closure_residual_batch(r0a, r1a, r2a, e0p, e1p, e2p, ma)
            med[b] = np.median(Rb); frac[b] = np.mean(Rb < 0.55)
            if b < K_agg: agg.append(Rb)
        return med, frac, (np.concatenate(agg) if len(agg) else med)

    def _null_swap():
        med, frac, agg = np.empty(int(n_perm)), np.empty(int(n_perm)), []
        K_agg = int(min(50, n_perm))
        for b in range(int(n_perm)):
            swap = rng.random(N) < 0.5
            nr1 = np.where(swap, r2a, r1a)
            nr2 = np.where(swap, r1a, r2a)
            Rb = _closure_residual_batch(r0a, nr1, nr2, e0a, e1a, e2a, ma)
            med[b] = np.median(Rb); frac[b] = np.mean(Rb < 0.55)
            if b < K_agg: agg.append(Rb)
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
            med[b] = np.median(Rb); frac[b] = np.mean(Rb < 0.55)
            if b < K_agg: agg.append(Rb)
        return med, frac, (np.concatenate(agg) if len(agg) else med)

    def _null_rshuffle():
        med, frac, agg = np.empty(int(n_perm)), np.empty(int(n_perm)), []
        K_agg = int(min(50, n_perm))
        by_img = list(df.groupby("image_id"))
        for b in range(int(n_perm)):
            nr0 = np.empty_like(r0a); nr1 = np.empty_like(r1a); nr2 = np.empty_like(r2a)
            start = 0
            for img_id, g in by_img:
                n = len(g)
                idx = rng.permutation(n)
                nr0[start:start+n] = g["r0"].to_numpy(float)[idx]
                nr1[start:start+n] = g["r1"].to_numpy(float)[idx]
                nr2[start:start+n] = g["r2"].to_numpy(float)[idx]
                start += n
            Rb = _closure_residual_batch(nr0, nr1, nr2, e0a, e1a, e2a, ma)
            med[b] = np.median(Rb); frac[b] = np.mean(Rb < 0.55)
            if b < K_agg: agg.append(Rb)
        return med, frac, (np.concatenate(agg) if len(agg) else med)

    include = set(include_nulls or [])
    results = {}

    def _run_null(name, fn):
        t0 = time.time()
        log(f"[NULL-TEST] start {name}: n_perm={int(n_perm)}, N={N}")
        out = fn()
        log(f"[NULL-TEST] done  {name}: elapsed={human_time(time.time()-t0)}")
        return out

    if "shuffle"   in include: results["shuffle"]   = _run_null("shuffle",   _null_shuffle)
    if "swap"      in include: results["swap"]      = _run_null("swap",      _null_swap)
    if "randtheta" in include: results["randtheta"] = _run_null("randtheta", _null_randtheta)
    if "rshuffle"  in include: results["rshuffle"]  = _run_null("rshuffle",  _null_rshuffle)

    # Standard one-sided p-values and Cliff's delta (P(x>y) - P(x<y))
    def _pvals_and_delta(med_null, frac_null, R_null_agg):
        p_med = (1.0 + float(np.sum(med_null <= obs_median))) / (len(med_null) + 1.0)
        p_frac = (1.0 + float(np.sum(frac_null >= obs_frac055))) / (len(frac_null) + 1.0)
        def _cliffs_delta(x: np.ndarray, y: np.ndarray, max_pairs: int = 2_000_000) -> float:
            x = np.asarray(x, dtype=np.float64); y = np.asarray(y, dtype=np.float64)
            n, m = x.size, y.size
            if n * m <= max_pairs:
                diff = x.reshape(-1, 1) - y.reshape(1, -1)
                return float((np.sum(diff > 0) - np.sum(diff < 0)) / (n * m))
            k = int(np.sqrt(max_pairs))
            rng_local = np.random.default_rng(int(seed) + 3)
            xi = rng_local.integers(0, n, size=k); yi = rng_local.integers(0, m, size=k)
            diff = x[xi].reshape(-1, 1) - y[yi].reshape(1, -1)
            return float((np.sum(diff > 0) - np.sum(diff < 0)) / diff.size)
        delta = _cliffs_delta(R_obs, R_null_agg)
        return p_med, p_frac, delta

    lines = [f"DATASET: {dataset}",
             f"N nodes: {len(R_obs)}",
             f"Observed: median={obs_median:.6f} [95% CI {ci_lo:.6f},{ci_hi:.6f}], frac<0.55={obs_frac055:.6f}",
             f"Null reps per control: {int(n_perm)}"]

    legend_done = False
    for key, (med_null, frac_null, Ragg) in results.items():
        p_med, p_frac, delta = _pvals_and_delta(med_null, frac_null, Ragg)
        Ragg_finite = Ragg[np.isfinite(Ragg)]
        ax.hist(Ragg_finite, bins=40, histtype="step", linewidth=1.25, label=f"null ({key})", density=True)
        pm_str = "< 1e-12" if p_med < 1e-12 else f"= {p_med:.6g}"
        pf_str = "< 1e-12" if p_frac < 1e-12 else f"= {p_frac:.6g}"
        lines.append(f"{key}: p_median(lower){pm_str}, p_frac<0.55(higher){pf_str}, Cliff's δ={delta:.6f}")
        legend_done = True
        xlim_collect.append(Ragg_finite)

    # --- Gaussian jitter (independent per radius), optional re-inversion of m under jitter ---
    if gaussian_jitter_sd is not None and float(gaussian_jitter_sd) > 0:
        sd = float(gaussian_jitter_sd)
        rng_local = np.random.default_rng(int(seed) + 7)
        j0 = r0a * (1.0 + rng_local.normal(0.0, sd, size=N))
        j1 = r1a * (1.0 + rng_local.normal(0.0, sd, size=N))
        j2 = r2a * (1.0 + rng_local.normal(0.0, sd, size=N))
        if reinvert_m_on_jitter:
            mj = np.empty(N, dtype=float)
            for i in range(N):
                mj[i] = m_from_node(j0[i], j1[i], j2[i], t12[i], m_min=0.2, m_max=4.0, tol=1e-6, iters=64)
        else:
            mj = ma
        Rj = _closure_residual_batch(j0, j1, j2, e0a, e1a, e2a, mj)
        Rj_finite = Rj[np.isfinite(Rj)]
        if Rj_finite.size > 0:
            med_j = float(np.median(Rj_finite))
            ax.axvline(med_j, linestyle=":", linewidth=1.2,
                       label=f"median (Gaussian jitter SD={sd:.2f})={med_j:.3f}")
            xlim_collect.append(Rj_finite)

    # Set a readable x-range (cap at 99.5th percentile; hard cap at 2.0)
    try:
        all_vals = np.concatenate([v for v in xlim_collect if v.size > 0])
        xmax = float(np.nanpercentile(all_vals, 99.5))
        ax.set_xlim(0.0, min(xmax, 2.0))
    except Exception:
        pass

    ax.set_xlabel("closure residual R(m)")
    ax.set_ylabel("density")
    ax.set_ylim(bottom=0)
    ax.set_title(f"Closure residual R(m) — {dataset}")
    if legend_done or (gaussian_jitter_sd and gaussian_jitter_sd > 0):
        ax.legend(
            loc="upper right",
            frameon=True, framealpha=0.9,
            facecolor="white", edgecolor="none"
        )


    # Save under variant tag AND under base dataset for canonical filenames
    dataset_base = dataset.split("__", 1)[0]

    # Variant-scoped figure
    save_png(fig, FIG_ROOT / dataset / f"dataset_{dataset}__residual_hist.png")
    # Canonical dataset-level figure
    save_png(fig, FIG_ROOT / dataset_base / f"dataset_{dataset_base}__residual_hist.png")

    # Variant-scoped NULLTEST
    out_txt_variant = CSV_ROOT / dataset / "NULLTEST__R_m.txt"
    out_txt_variant.parent.mkdir(parents=True, exist_ok=True)
    with open(out_txt_variant, "w") as f:
        for line in lines:
            f.write(line + "\n")

    # Mirror to dataset-level NULLTEST
    out_txt_base = CSV_ROOT / dataset_base / "NULLTEST__R_m.txt"
    out_txt_base.parent.mkdir(parents=True, exist_ok=True)
    (out_txt_base).write_text((out_txt_variant).read_text())
    log("[NULL-TEST] " + " | ".join(lines[2:]))

    # --- PREREG report: held-out metric must beat ALL nulls on both tests, with δ ≤ -0.20 ---
    if (R_col == "R_m_holdout") and (len(results) > 0):
        alpha_p = 1e-3
        cliff_thresh = -0.20
        verdicts = []
        lines_nulls = []
        worst_p_med = 1.0
        worst_p_frac = 1.0
        max_delta = -1.0  # most positive (least favorable) delta
        for key, (med_null, frac_null, Ragg) in results.items():
            p_med, p_frac, delta = _pvals_and_delta(med_null, frac_null, Ragg)
            lines_nulls.append(f"{key}: p_median(lower)={p_med:.6g}, p_frac<0.55(higher)={p_frac:.6g}, Cliff's δ={delta:.6f}")
            worst_p_med = min(worst_p_med, p_med)
            worst_p_frac = min(worst_p_frac, p_frac)
            max_delta = max(max_delta, delta)
            verdicts.append((p_med <= alpha_p) and (p_frac <= alpha_p) and (delta <= cliff_thresh))
        overall = all(verdicts)
        outcome = "PASS" if overall else "FAIL"

        # Write variant-level prereg file
        out_c1_variant = CSV_ROOT / dataset / "PREREG__C1.txt"
        out_c1_variant.parent.mkdir(parents=True, exist_ok=True)
        with open(out_c1_variant, "w") as f:
            f.write(f"DATASET: {dataset}\n")
            f.write("Claim HRF-C1 (held-out closure @ angle-only m):\n")
            f.write(f"Observed \\tilde R = {obs_median:.6f}  [boot 95% CI {ci_lo:.6f},{ci_hi:.6f}]\n")
            for L in lines_nulls:
                f.write(L + "\n")
            f.write(f"Thresholds: p <= {alpha_p:g} on BOTH median and Pr[R<0.55], and Cliff's δ <= {cliff_thresh:.2f} vs EVERY null\n")
            f.write(f"Worst-case across nulls: p_med={worst_p_med:.6g}, p_frac={worst_p_frac:.6g}, max δ={max_delta:.6f}\n")
            f.write(f"Outcome: {outcome}\n")

        # Also mirror to base dataset folder (for canonical “HRF/*” files)
        dataset_base = dataset.split("__", 1)[0]
        out_c1_base = CSV_ROOT / dataset_base / "PREREG__C1.txt"
        out_c1_base.parent.mkdir(parents=True, exist_ok=True)
        (out_c1_base).write_text((out_c1_variant).read_text())

        # Explicit console summary of prereg result (clear PASS/FAIL line)
        log(
            "[PREREG] metric=heldout; "
            f"dataset={dataset}; outcome={outcome}; "
            f"obs_median={obs_median:.3f} [CI {ci_lo:.3f},{ci_hi:.3f}]; "
            f"worst_p_median={worst_p_med:.3g}; worst_p_frac={worst_p_frac:.3g}; "
            f"max_delta={max_delta:.3f}; file={out_c1_variant}"
        )



def plot_fixed_m_baselines(df: pd.DataFrame, dataset: str,
                           fixed_ms=(2.0, 1.0), grid_ms=np.linspace(0.2, 4.0, 381)):
    """
    Panel B2 — Baselines (fixed m).
    Compares EPIC per-node R(m_node) against R(m) for classical fixed m and the dataset's best single m*.
    If directions are absent, falls back to a normalized scalar equation misfit.
    Outputs:
      figures/<DATASET>/dataset_<DATASET>__residual_baselines.png
      reports/<DATASET>/BASELINES__R_m.txt
    """
    import numpy as np, matplotlib.pyplot as plt
    from math import comb

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
        denom = (np.power(r1, m)) + (np.power(r2, m)) + 1e-12
        return num / denom

    def eq_misfit(m):
        c = np.cos(t12)
        val = (r0**(2*m)) - (r1**(2*m)) - (r2**(2*m)) - 2*c*(r1**m)*(r2**m)
        scale = (r0**(2*m)) + (r1**(2*m)) + (r2**(2*m)) + 1e-12
        return np.abs(val)/scale

    def bootstrap_ci_median(x, B=5000):
        rng = np.random.default_rng(13)
        meds = np.median(rng.choice(x, size=(B, x.size), replace=True), axis=1)
        return float(np.percentile(meds, 2.5)), float(np.percentile(meds, 97.5))

    def sign_test_improvement(a, b):
        d = b - a
        d = d[np.isfinite(d) & (d != 0)]
        n = d.size
        k = int(np.sum(d > 0))
        # two-sided exact sign test
        p_lower = sum(comb(n, i) for i in range(0, k+1)) / (2**n)
        p_upper = sum(comb(n, i) for i in range(k, n+1)) / (2**n)
        return n, k, 2 * min(p_lower, p_upper)

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

    # NEW: ECDF of paired improvements ΔR = R(baseline) − R(EPIC)
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




def compute_per_node_uncertainty(
    df: pd.DataFrame,
    dataset: str,
    n_draws: int = 500,
    seed: int = 13,
    radius_sd_frac: float = 0.05,  # conservative default when local EDT variability not recorded
):
    """
    STEP 3 — Monte-Carlo per-node uncertainty on m (angle-only inversion):
      - Jitter radii by multiplicative Gaussian SD = radius_sd_frac * r.
      - Jitter angles via per-branch tangent SVD ratio: SD_deg = min(15, 60/svd_ratio).
      - For each redraw, recompute m from (r0,r1,r2,theta12).
      - Writes:
          reports/<DATASET>/UNCERTAINTY__nodes.csv  (m_hat, m_lo, m_hi, CI_width, at_edge_before, at_edge_after)
          figures/<DATASET>/dataset_<DATASET>__m_uncertainty_forest.png
      - Adds summary lines into SUMMARY__*.txt are produced by summarize_and_write (reads counts).
    Pass/Fail (reported at end of CSV header):
      - ≥ 60% of bracket-edge nodes move off the edge after uncertainty;
      - Median CI half-width ≤ 0.35.
    """
    if len(df) == 0:
        log("[UNCERTAINTY] empty DF; skipping")
        return

    rng = np.random.default_rng(int(seed))
    dataset_base = dataset.split("__", 1)[0]

    # pull fields
    r0 = df["r0"].to_numpy(float); r1 = df["r1"].to_numpy(float); r2 = df["r2"].to_numpy(float)
    th_deg = df["theta12_deg"].to_numpy(float)
    mhat = df["m_angleonly"].to_numpy(float) if "m_angleonly" in df.columns else df["m_node"].to_numpy(float)
    mmin = float(df.get("cfg_m_min", pd.Series([0.2])).iloc[0]); mmax = float(df.get("cfg_m_max", pd.Series([4.0])).iloc[0])

    # per-branch SVD-based angle SD (deg) with conservative fallback
    def _sd_from_svd(x):
        if not np.isfinite(x) or x <= 0: return 12.0
        return float(min(15.0, 60.0 / x))
    sd_e1 = np.array([_sd_from_svd(v) for v in df.get("svd_ratio_e1", pd.Series(np.nan, index=df.index)).to_numpy(float)])
    sd_e2 = np.array([_sd_from_svd(v) for v in df.get("svd_ratio_e2", pd.Series(np.nan, index=df.index)).to_numpy(float)])
    # combine into theta SD (sum of two small rotations in quadrature)
    sd_theta = np.sqrt(sd_e1**2 + sd_e2**2) / np.sqrt(2.0)

    N = len(df)
    m_lo = np.empty(N, float); m_hi = np.empty(N, float)
    off_edge_after = np.zeros(N, dtype=bool)


    # progress heartbeat
    log(f"[UNCERTAINTY] start: N={N}, draws/node={int(n_draws)}, radius_sd_frac={radius_sd_frac}")
    t0_unc = time.time()
    next_tick = 0
    tick_every = max(1, N // 20)  # ~5% steps

    for i in range(N):
        if not np.isfinite(mhat[i]):
            m_lo[i] = np.nan; m_hi[i] = np.nan
            continue

        r0i, r1i, r2i = r0[i], r1[i], r2[i]
        theta_i = np.deg2rad(th_deg[i])
        draws = []
        for _ in range(int(n_draws)):
            rr0 = r0i * (1.0 + rng.normal(0.0, radius_sd_frac))
            rr1 = r1i * (1.0 + rng.normal(0.0, radius_sd_frac))
            rr2 = r2i * (1.0 + rng.normal(0.0, radius_sd_frac))
            tpert = theta_i + np.deg2rad(rng.normal(0.0, sd_theta[i] if np.isfinite(sd_theta[i]) else 10.0))
            m_s = m_from_node(rr0, rr1, rr2, tpert, m_min=mmin, m_max=mmax, tol=1e-6, iters=64)
            if np.isfinite(m_s):
                draws.append(m_s)

        if len(draws) >= 8:
            qs = np.percentile(draws, [2.5, 97.5])
            m_lo[i], m_hi[i] = float(qs[0]), float(qs[1])
            off_edge_after[i] = (m_lo[i] > mmin + 1e-6) and (m_hi[i] < mmax - 1e-6)
        else:
            m_lo[i], m_hi[i] = np.nan, np.nan
            off_edge_after[i] = False

        # heartbeat log every ~5%
        if i >= next_tick:
            elapsed = time.time() - t0_unc
            frac = (i + 1) / float(N)
            eta = elapsed * (1.0 - frac) / max(frac, 1e-9)
            log(f"[UNCERTAINTY] progress {i+1}/{N} ({frac*100:.0f}%) elapsed={human_time(elapsed)} eta={human_time(eta)}")
            next_tick += tick_every

    log(f"[UNCERTAINTY] done: total={human_time(time.time()-t0_unc)}")

    ci_width = (m_hi - m_lo)
    at_edge_before = ((np.abs(mhat - mmin) <= 1e-6) | (np.abs(mhat - mmax) <= 1e-6))
    moved_off_edge = np.logical_and(at_edge_before, off_edge_after)
    frac_edge_moved = float(np.mean(moved_off_edge)) if np.any(at_edge_before) else float("nan")
    med_half = float(np.nanmedian(ci_width/2.0))

    # PASS/FAIL
    pass_edge = (np.isfinite(frac_edge_moved) and (frac_edge_moved >= 0.60))
    pass_width = (np.isfinite(med_half) and (med_half <= 0.35))
    outcome = "PASS" if (pass_edge and pass_width) else "FAIL"

    # write CSV
    out_csv = CSV_ROOT / dataset / "UNCERTAINTY__nodes.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df_out = pd.DataFrame({
        "image_id": df["image_id"],
        "node_id": df["node_id"],
        "m_hat": mhat,
        "m_lo": m_lo,
        "m_hi": m_hi,
        "CI_width": ci_width,
        "at_edge_before": at_edge_before,
        "at_edge_after": off_edge_after
    })
    header = (f"# STEP 3 — Per-node uncertainty: outcome={outcome} | "
              f"frac_edge_moved={frac_edge_moved:.3f} | median CI half-width={med_half:.3f}\n")
    out_csv.write_text(header + df_out.to_csv(index=False))

    # forest plot
    order = np.argsort(np.nan_to_num(ci_width, nan=9e9))
    top_idx = order[:min(300, len(order))]
    fig = plt.figure(figsize=(8.5, 10.0)); ax = plt.gca()
    y = np.arange(top_idx.size)
    ax.hlines(y, m_lo[top_idx], m_hi[top_idx], linewidth=1.2)
    ax.plot(mhat[top_idx], y, "o", markersize=3)
    # mark edge nodes
    for j, i in enumerate(top_idx):
        if at_edge_before[i]:
            ax.plot([m_lo[i], m_hi[i]], [j, j], linewidth=2.4)
    ax.set_xlabel("m (95% CI)"); ax.set_ylabel("nodes (sorted by CI width)")
    ax.set_title(f"m uncertainty (N={len(df)}) — {dataset}  | outcome {outcome}")
    save_png(fig, FIG_ROOT / dataset / f"dataset_{dataset}__m_uncertainty_forest.png")

    log(f"[UNCERTAINTY] outcome={outcome}, frac_edge_moved={frac_edge_moved:.3f}, median CI half-width={med_half:.3f}")



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
        log("[PARENTDIR] Missing direction columns; skipping.")
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
    ax2.boxplot([v[np.isfinite(v)] for v in deltas.values()], labels=list(deltas.keys()))
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
    Panel D — Positive control: analytic junctions with realistic noise.
    Adds a parent/daughter mislabel fraction to demonstrate sensitivity and gate efficacy.
    """
    rng = np.random.default_rng(int(seed))

    # Draw truths
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
    e0 = - s / r0m[:, None]

    def _jitter_dirs(E: np.ndarray, sd_deg: float) -> np.ndarray:
        if sd_deg <= 0: return E
        d = np.deg2rad(rng.normal(0.0, sd_deg, size=E.shape[0]))
        cd, sd = np.cos(d), np.sin(d)
        x, y = E[:, 0], E[:, 1]
        xr = x * cd - y * sd
        yr = x * sd + y * cd
        V = np.stack([xr, yr], axis=1)
        n = np.linalg.norm(V, axis=1, keepdims=True) + 1e-12
        return V / n

    e0n = _jitter_dirs(e0, noise_dir_deg)
    e1n = _jitter_dirs(e1, noise_dir_deg)
    e2n = _jitter_dirs(e2, noise_dir_deg)

    r0n = r0 * (1.0 + rng.normal(0.0, noise_radius_frac, size=n_samples))
    r1n = r1 * (1.0 + rng.normal(0.0, noise_radius_frac, size=n_samples))
    r2n = r2 * (1.0 + rng.normal(0.0, noise_radius_frac, size=n_samples))

    # Optional: mislabel parent at random fraction
    if parent_mislabel_frac > 0:
        mis = rng.random(n_samples) < float(parent_mislabel_frac)
        # swap parent with the fatter daughter (mimic realistic failure)
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

    m_est = np.empty(n_samples, dtype=np.float64)
    for i in range(n_samples):
        m_est[i] = m_from_node(r0n[i], r1n[i], r2n[i], t12n[i], m_min=0.2, m_max=4.0, tol=1e-6, iters=64)

    valid = np.isfinite(m_est)
    if not np.any(valid):
        log("[POSCTRL] No finite m estimates; skipping plot.")
        return

    mt = m_true[valid]
    me = m_est[valid]
    delta = me - mt

    mae = float(np.median(np.abs(delta)))
    try:
        slope, intercept, lo_s, hi_s = theilslopes(me, mt)
    except Exception:
        slope, intercept, lo_s, hi_s = float("nan"), float("nan"), float("nan"), float("nan")
    try:
        r, _ = pearsonr(mt, me)
    except Exception:
        r = float("nan")

    fig = plt.figure(figsize=(7.0, 5.2))
    ax = plt.gca()
    ax.scatter(mt, me, s=12, alpha=0.85, label="nodes")
    ax.plot([0.2, 4.0], [0.2, 4.0], linewidth=2.0, label="identity")
    ax.set_xlim(0.2, 4.0); ax.set_ylim(0.2, 4.0)
    ax.set_xlabel("m (true)")
    ax.set_ylabel("m (estimated)")
    ax.set_title(f"Positive control — analytic junctions (N={len(me)})")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")

    ax.text(0.02, 0.98,
            "\n".join([
                f"median|Δm|={np.median(np.abs(delta)):.3f}",
                f"MAE={mae:.3f}",
                f"Theil–Sen slope={slope:.3f} [{lo_s:.3f},{hi_s:.3f}]",
                f"Pearson r={r:.3f}",
                f"dir noise={noise_dir_deg:.1f}°, radius noise SD={noise_radius_frac:.2f}",
                f"parent mislabel={parent_mislabel_frac:.2%}"
            ]),
            transform=ax.transAxes, va="top", ha="left",
            fontsize=9, bbox=dict(facecolor="white", alpha=0.85, edgecolor="none"))

    save_png(fig, FIG_ROOT / dataset / f"POSCTRL__recovery.png")

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
    log(f"[POSCTRL] N={len(me)}, median|Δm|={np.median(np.abs(delta)):.3f}, MAE={mae:.3f}, r={r:.3f}, mislabel={parent_mislabel_frac:.2%}")






def plot_theta_vs_m_scatter(
    df: pd.DataFrame,
    dataset: str,
    symmetric_tol: float = 1.08,
    bins_deg: float = 10.0,
    bootstrap: int = 5000,
    seed: int = 13,
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




def run_ablation_grid(df: pd.DataFrame, dataset: str,
                      svd_grid=(3.0, 3.5, 4.0),
                      angle_grid_deg=(10.0, 12.0, 15.0),
                      p_thresh: float = 0.01,
                      qc_drift_limit_pp: float = 10.0,
                      n_perm: int = 2000,
                      seed: int = 13):
    """
    STEP 4 — Ablations & robustness (formal test)
      PASS if:
        • For every grid cell with >=20 nodes, the observed median R(m) beats a shuffle-null with one-sided p ≤ p_thresh, AND
        • strict-QC% drifts by ≤ qc_drift_limit_pp from the dataset-level baseline across all cells.

      Outputs:
        figures/<VARIANT>/dataset_<VARIANT>__ablation_medianR_grid.png
        figures/<VARIANT>/dataset_<VARIANT>__QC_tradeoff_curves.png
        reports/<VARIANT>/ABLATION__summary.txt
        reports/<VARIANT>/ABLATION__test.txt          (Outcome: PASS/FAIL)
        reports/<DATASET_BASE>/ABLATION__test.txt     (mirrored)
      Also prints a concise PASS/FAIL block to terminal.
    """
    have_dirs = {"e0x","e0y","e1x","e1y","e2x","e2y"}.issubset(df.columns)
    if not have_dirs or len(df) == 0:
        log("[ABLATION] Missing direction columns; skipping.")
        return

    import numpy as np
    rng = np.random.default_rng(int(seed))
    dataset_base = dataset.split("__", 1)[0]

    # helper
    def _closure_residual_batch(r0, r1, r2, e0, e1, e2, m):
        a0 = (np.power(r0, m))[:, None] * e0
        a1 = (np.power(r1, m))[:, None] * e1
        a2 = (np.power(r2, m))[:, None] * e2
        num = np.linalg.norm(a0 + a1 + a2, axis=1)
        den = (np.power(r1, m)) + (np.power(r2, m)) + 1e-12
        return num / den

    # arrays
    e0 = df[["e0x","e0y"]].to_numpy(float)
    e1 = df[["e1x","e1y"]].to_numpy(float)
    e2 = df[["e2x","e2y"]].to_numpy(float)
    r0 = df["r0"].to_numpy(float); r1 = df["r1"].to_numpy(float); r2 = df["r2"].to_numpy(float)
    theta = df["theta12_deg"].to_numpy(float)
    m_node = df.get("m_node", pd.Series(np.nan, index=df.index)).to_numpy(float)
    R_obs_full = df.get("R_m", pd.Series(np.nan, index=df.index)).to_numpy(float)

    # dataset-level strict-QC baseline
    qc_col = df["qc_pass_strict"] if "qc_pass_strict" in df.columns else pd.Series([], dtype=bool)
    baseline_qc = float(qc_col.mean()) if len(qc_col) else float("nan")

    # node-level svd_min proxy (min of the three per node)
    svd_min = np.nanmin(np.vstack([
        df.get("svd_ratio_e0", pd.Series(np.nan, index=df.index)).to_numpy(float),
        df.get("svd_ratio_e1", pd.Series(np.nan, index=df.index)).to_numpy(float),
        df.get("svd_ratio_e2", pd.Series(np.nan, index=df.index)).to_numpy(float)
    ]), axis=0)

    # matrices to fill
    heat = np.full((len(svd_grid), len(angle_grid_deg)), np.nan, float)
    p_mat = np.full_like(heat, np.nan, float)
    qc_mat = np.full_like(heat, np.nan, float)
    n_mat  = np.zeros_like(heat, int)

    # common direction pool for shuffles
    pool = np.vstack([e0, e1, e2])

    for i, sthr in enumerate(svd_grid):
        for j, athr in enumerate(angle_grid_deg):
            mask = (svd_min >= float(sthr)) & (theta >= float(athr)) & np.isfinite(R_obs_full)
            n = int(np.count_nonzero(mask))
            n_mat[i, j] = n
            if n < 20:
                continue

            R_obs = R_obs_full[mask]
            med_obs = float(np.median(R_obs))

            med_null = np.empty(int(n_perm), float)
            t0_perm = time.time()
            log(f"[ABLATION] cell svd_min≥{sthr:g}, angle_min≥{athr:g}: n={n}, n_perm={int(n_perm)} — start")
            for b in range(int(n_perm)):
                idxs = rng.integers(0, pool.shape[0], size=(n, 3))
                e0p = pool[idxs[:,0]]; e1p = pool[idxs[:,1]]; e2p = pool[idxs[:,2]]
                Rb = _closure_residual_batch(r0[mask], r1[mask], r2[mask], e0p, e1p, e2p, m_node[mask])
                med_null[b] = np.median(Rb)
                if (b + 1) % max(1, int(n_perm // 5)) == 0:
                    elapsed = time.time() - t0_perm
                    frac = (b + 1) / float(n_perm)
                    eta  = elapsed * (1.0 - frac) / max(frac, 1e-9)
                    log(f"[ABLATION] cell svd_min≥{sthr:g}, angle_min≥{athr:g}: {b+1}/{int(n_perm)} ({frac*100:.0f}%) elapsed={human_time(elapsed)} eta={human_time(eta)}")
            p = (1.0 + float(np.sum(med_null <= med_obs))) / (len(med_null) + 1.0)  # lower is better
            log(f"[ABLATION] cell svd_min≥{sthr:g}, angle_min≥{athr:g}: done in {human_time(time.time()-t0_perm)}; med_obs={med_obs:.3f}; p={p:.3g}")


            heat[i, j] = med_obs
            p_mat[i, j] = p
            qc_mat[i, j] = float(df.loc[mask, "qc_pass_strict"].mean()) if "qc_pass_strict" in df.columns else np.nan

    # figures (same visuals)
    fig = plt.figure(figsize=(7.8, 5.6)); ax = plt.gca()
    im = ax.imshow(heat, origin="lower", aspect="auto",
                   extent=[min(angle_grid_deg), max(angle_grid_deg), min(svd_grid), max(svd_grid)])
    ax.set_xlabel("angle gate min (deg)"); ax.set_ylabel("svd_ratio_min")
    ax.set_title(f"Ablation: median R(m) — {dataset}")
    plt.colorbar(im, ax=ax, label="median R(m)")
    save_png(fig, FIG_ROOT / dataset / f"dataset_{dataset}__ablation_medianR_grid.png")

    # QC tradeoff (use first angle slice for a clean curve)
    fig2 = plt.figure(figsize=(7.6, 5.2)); ax2 = plt.gca()
    try:
        j0 = 0
        medR = heat[:, j0]; qcS = qc_mat[:, j0]
        keep = np.isfinite(medR) & np.isfinite(qcS)
        ax2.plot(medR[keep], qcS[keep], "o-")
    except Exception:
        pass
    ax2.set_xlabel("median R(m)"); ax2.set_ylabel("strict-QC fraction")
    ax2.set_title(f"QC tradeoff — {dataset}")
    ax2.grid(True, alpha=0.25)
    save_png(fig2, FIG_ROOT / dataset / f"dataset_{dataset}__QC_tradeoff_curves.png")

    # PASS/FAIL criteria
    valid = n_mat >= 20
    all_p_ok = bool(np.all(p_mat[valid] <= float(p_thresh))) if np.any(valid) else False
    worst_p = float(np.nanmax(p_mat[valid])) if np.any(valid) else float("nan")

    # QC drift vs baseline (percentage points)
    qc_drift_pp = np.nanmax(np.abs(qc_mat[valid] - baseline_qc)) * 100.0 if np.any(valid) and np.isfinite(baseline_qc) else float("nan")
    drift_ok = bool(np.isfinite(qc_drift_pp) and qc_drift_pp <= float(qc_drift_limit_pp))

    outcome = "PASS" if (all_p_ok and drift_ok) else "FAIL"

    # Write detailed summary
    out_sum = CSV_ROOT / dataset / "ABLATION__summary.txt"
    lines = [
        f"DATASET: {dataset}",
        f"Grid svd_ratio_min={tuple(svd_grid)}, angle_min_deg={tuple(angle_grid_deg)}",
        f"Valid cells (n>=20): {int(np.sum(valid))} / {valid.size}",
        f"Baseline strict-QC: {baseline_qc:.4f}" if np.isfinite(baseline_qc) else "Baseline strict-QC: n/a",
        f"Worst p (lower better) over valid cells: {worst_p:.6g}",
        f"Worst strict-QC drift: {qc_drift_pp:.2f} pp"
    ]
    # matrix prints
    lines.append("median R(m):")
    for i, sthr in enumerate(svd_grid):
        lines.append(f"  svd≥{sthr:g}: " + ", ".join(f"{heat[i,j]:.3f}" if np.isfinite(heat[i,j]) else "nan" for j in range(len(angle_grid_deg))))
    lines.append(f"p-values (≤ {p_thresh:g} required):")
    for i, sthr in enumerate(svd_grid):
        lines.append(f"  svd≥{sthr:g}: " + ", ".join(f"{p_mat[i,j]:.3g}" if np.isfinite(p_mat[i,j]) else "nan" for j in range(len(angle_grid_deg))))
    lines.append("strict-QC fractions:")
    for i, sthr in enumerate(svd_grid):
        lines.append(f"  svd≥{sthr:g}: " + ", ".join(f"{qc_mat[i,j]:.3f}" if np.isfinite(qc_mat[i,j]) else "nan" for j in range(len(angle_grid_deg))))
    lines.append(f"Outcome: {outcome}")
    out_sum.parent.mkdir(parents=True, exist_ok=True)
    out_sum.write_text("\n".join(lines) + "\n")

    # One-line PASS/FAIL test file (variant + base)
    out_test_variant = CSV_ROOT / dataset / "ABLATION__test.txt"
    out_test_variant.write_text(f"Outcome: {outcome}\nWorst p: {worst_p:.6g}\nWorst QC drift: {qc_drift_pp:.2f} pp\n")
    out_test_base = CSV_ROOT / dataset_base / "ABLATION__test.txt"
    out_test_base.parent.mkdir(parents=True, exist_ok=True)
    out_test_base.write_text(out_test_variant.read_text())

    # Terminal summary
    log("====== ABLATION TEST (STEP 4) ======")
    log(f"dataset={dataset}")
    log(f"valid cells: {int(np.sum(valid))}/{valid.size}")
    log(f"worst p: {worst_p:.6g}  (require ≤ {p_thresh:g} everywhere)")
    log(f"strict-QC drift: {qc_drift_pp:.2f} pp  (require ≤ {qc_drift_limit_pp:.2f} pp)")
    log(f"Outcome: {outcome}")
    log("====================================")



def plot_tariff_map(gray: np.ndarray, skel: np.ndarray,
                    nodes: List[NodeRecord],
                    dataset: str, image_id: str):
    """
    Tariff overlay (RGB encodes c0:c1:c2), with QC outline and bracket-edge markers.
    Also emits a residual-vector performance map alongside the tariff map.

    Files written:
      figures/<dataset>/<image_id>__tariff_map.png
      figures/<dataset>/<image_id>__residual_map.png
    """
    import numpy as np
    import matplotlib.pyplot as plt

    # --- Tariff map ---
    fig = plt.figure(figsize=(10, 10)); ax = plt.gca()
    ax.imshow(gray, cmap="gray")
    ys, xs = np.where(skel > 0)
    ax.scatter(xs, ys, s=0.1, alpha=0.6)

    # Use a small ring to indicate bracket-edge m (~at 0.2 or 4.0)
    m_min, m_max = 0.2, 4.0
    for rec in nodes:
        rsum = rec.c0 + rec.c1 + rec.c2 + 1e-9
        rgb = [rec.c0 / rsum, rec.c1 / rsum, rec.c2 / rsum]
        y, x = rec.yx

        # QC outline: green=strict pass, orange=loose-only, red=fail
        if rec.qc_pass_strict:
            edge = "lime"
        elif rec.qc_pass:
            edge = "orange"
        else:
            edge = "red"

        ax.scatter([x], [y], s=28, marker='o', c=[rgb], edgecolors=edge, linewidths=0.8)

        # Bracket-edge m ring
        at_edge = (abs(rec.m_node - m_min) <= 0.02) or (abs(rec.m_node - m_max) <= 0.02)
        if at_edge:
            ax.scatter([x], [y], s=44, facecolors='none', edgecolors='black', linewidths=0.6)

    ax.set_axis_off()
    ax.set_title(f"{dataset}/{image_id} — tariffs (RGB) • outline=QC • ring=bracket-edge m")
    save_png(fig, FIG_ROOT / dataset / f"{image_id}__tariff_map.png")

    # --- Residual-vector map ---
    fig2 = plt.figure(figsize=(10, 10)); ax2 = plt.gca()
    H, W = gray.shape
    ax2.imshow(gray, cmap="gray")
    ax2.scatter(xs, ys, s=0.1, alpha=0.6)

    # Lock the view to the image extent so long arrows cannot autoscale the axes
    ax2.set_xlim(0, W)
    ax2.set_ylim(H, 0)  # imshow origin is upper-left
    ax2.set_aspect('equal', adjustable='box')
    ax2.autoscale(False)

    # Arrow length ∝ normalized residual R(m); cap to keep the picture readable
    scale_px = 30.0
    max_arrow_px = 45.0

    for rec in nodes:
        e0 = np.array([rec.e0x, rec.e0y], dtype=float)
        e1 = np.array([rec.e1x, rec.e1y], dtype=float)
        e2 = np.array([rec.e2x, rec.e2y], dtype=float)

        # normalized residual vector u = num/den, R = ||u||
        num = (rec.r0 ** rec.m_node) * e0 + (rec.r1 ** rec.m_node) * e1 + (rec.r2 ** rec.m_node) * e2
        den = (rec.r1 ** rec.m_node) + (rec.r2 ** rec.m_node) + 1e-12
        u = num / den
        R = float(np.linalg.norm(u))
        if not np.isfinite(R) or R <= 0:
            continue
        dir_u = u / R
        L = min(scale_px * R, max_arrow_px)
        dx = float(L * dir_u[0])
        dy = float(L * dir_u[1])

        y, x = rec.yx
        clr = "lime" if rec.qc_pass_strict else ("orange" if rec.qc_pass else "red")
        ax2.arrow(x, y, dx, dy,
                  width=0.0, head_width=2.0, length_includes_head=True,
                  color=clr, alpha=0.9, clip_on=True)

    ax2.set_axis_off()
    ax2.set_title(f"{dataset}/{image_id} — residual vectors (color=QC)")
    save_png(fig2, FIG_ROOT / dataset / f"{image_id}__residual_map.png")



def summarize_and_write(df: pd.DataFrame, dataset: str, diags: List[Dict],
                        n_images_total: int, elapsed_sec: float):
    out_txt = CSV_ROOT / dataset / f"SUMMARY__{dataset}.txt"
    out_txt.parent.mkdir(parents=True, exist_ok=True)

    dataset_base = dataset.split("__", 1)[0]

    # ---------- totals ----------
    images_done = int(len(diags))
    images_with_nodes = int(df["image_id"].nunique()) if len(df) > 0 else 0
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

    m_med  = float(np.median(df["m_node"].values)) if n_kept > 0 else float("nan")
    m_iqr  = (float(np.percentile(df["m_node"], 25)), float(np.percentile(df["m_node"], 75))) if n_kept > 0 else (float("nan"), float("nan"))
    m_mean = float(np.mean(df["m_node"].values)) if n_kept > 0 else float("nan")
    R_med  = float(np.median(df["R_m"].values))    if ("R_m" in df.columns and n_kept > 0) else float("nan")

    # bracket-edge m count
    try:
        m_min = float(df["cfg_m_min"].iloc[0]); m_max = float(df["cfg_m_max"].iloc[0])
    except Exception:
        m_min, m_max = 0.2, 4.0
    edge_eps = 0.02
    at_edge_mask = ((np.abs(df["m_node"].astype(float) - m_min) <= edge_eps) |
                    (np.abs(df["m_node"].astype(float) - m_max) <= edge_eps)) if n_kept > 0 else np.array([], dtype=bool)
    n_edge = int(at_edge_mask.sum())
    frac_edge = (n_edge / n_kept) if n_kept > 0 else float("nan")

    # parent ambiguity (if recorded)
    parent_ambig_frac = float(df["parent_ambiguous"].astype(bool).mean()) if "parent_ambiguous" in df.columns and n_kept > 0 else float("nan")

    # ---------- per-image angle gate summary ----------
    angle_mins = [float(d.get("angle_gate_min_deg", float("nan"))) for d in diags]
    angle_mins = np.array([a for a in angle_mins if np.isfinite(a)], dtype=float)
    angle_gate_med = float(np.median(angle_mins)) if angle_mins.size else float("nan")
    angle_gate_rng = (float(np.min(angle_mins)), float(np.max(angle_mins))) if angle_mins.size else (float("nan"), float("nan"))

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
        f"radius_jitter_frac (run arg)={_get_cfg('args_radius_jitter_frac', 'n/a')}"
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
    grid_ms = np.linspace(0.2, 4.0, 381); med_grid = np.array([np.median(_R_at_m(float(mm))) for mm in grid_ms]) if n_kept > 0 else np.array([], float)
    m_star = float(grid_ms[int(np.argmin(med_grid))]) if med_grid.size else float("nan")
    R_mstar = _R_at_m(m_star) if n_kept > 0 and np.isfinite(m_star) else np.array([], float)

    def _sign_test(a, b):
        d = b - a
        d = d[np.isfinite(d) & (d != 0)]
        if d.size == 0: return (0, 0, float("nan"))
        from math import comb
        n = d.size; k = int(np.sum(d > 0))
        p_lower = sum(comb(n, i) for i in range(0, k+1)) / (2**n)
        p_upper = sum(comb(n, i) for i in range(k, n+1)) / (2**n)
        return (n, k, 2 * min(p_lower, p_upper))

    lines_baseline = []
    if np.all(np.isfinite(R_epic)) and R_epic.size > 0:
        for name, Rb in [("m=2", R_m2), ("m=1", R_m1), (f"m*={m_star:.2f}", R_mstar if np.isfinite(m_star) else np.array([], float))]:
            if Rb.size > 0 and np.all(np.isfinite(Rb)):
                n,k,p = _sign_test(R_epic, Rb); lines_baseline.append(f"{name}: median={np.median(Rb):.3f}, paired ΔR>0 n={n}, p={p:.2e}")

        # image-level medians
        try:
            Re_img = pd.Series(R_epic, index=df.index).groupby(df['image_id']).median()
            base_series = {"m=2": pd.Series(R_m2, index=df.index).groupby(df['image_id']).median(),
                           "m=1": pd.Series(R_m1, index=df.index).groupby(df['image_id']).median()}
            if np.isfinite(m_star):
                base_series[f"m*={m_star:.2f}"] = pd.Series(R_mstar, index=df.index).groupby(df['image_id']).median()
            for nm, ser in base_series.items():
                common = Re_img.index.intersection(ser.index)
                n,k,p = _sign_test(Re_img.loc[common].values, ser.loc[common].values)
                lines_baseline.append(f"[img-median] {nm}: n={n}, p={p:.2e}")
        except Exception:
            pass

    # ---------- skip reasons ----------
    all_keys = set()
    for d in diags: all_keys |= set(d.get("skip_reasons", {}).keys())
    skip_totals = {k: int(sum(d.get("skip_reasons", {}).get(k, 0) for d in diags)) for k in sorted(all_keys)}
    def _fmt_skip():
        total_skips = sum(skip_totals.values())
        if total_skips == 0: return "none"
        parts = []
        for k, v in sorted(skip_totals.items(), key=lambda kv: -kv[1]):
            pct = (100.0 * v / total_skips) if total_skips > 0 else 0.0
            parts.append(f"{k}={v} ({pct:.1f}%)")
        return ", ".join(parts)
    skip_list = _fmt_skip()

    # ---------- prereg & ablation outcomes (mirrored files) ----------
    prereg_path = CSV_ROOT / dataset_base / "PREREG__C1.txt"
    prereg_line = "STEP 1 — held-out closure: (no PREREG__C1.txt found)"
    if prereg_path.exists():
        txt = prereg_path.read_text().strip().splitlines()
        out_line = next((L for L in txt if L.startswith("Outcome:")), None)
        worst_line = next((L for L in txt if L.startswith("Worst-case")), None)
        if out_line:
            prereg_line = f"STEP 1 — held-out closure: {out_line.split(':',1)[1].strip()}" + (f" | {worst_line}" if worst_line else "")

    ablation_path = CSV_ROOT / dataset_base / "ABLATION__test.txt"
    ablation_line = "STEP 4 — ablation: (no ABLATION__test.txt found)"
    if ablation_path.exists():
        t = ablation_path.read_text().strip().splitlines()
        out_line = next((L for L in t if L.startswith("Outcome:")), None)
        worst_p  = next((L for L in t if L.startswith("Worst p:")), None)
        worst_qc = next((L for L in t if L.startswith("Worst QC drift:")), None)
        if out_line:
            ablation_line = f"STEP 4 — ablation: {out_line.split(':',1)[1].strip()}" + (f" | {worst_p}" if worst_p else "") + (f" | {worst_qc}" if worst_qc else "")

    # ---------- warnings / suggestions ----------
    warns = []
    def _add(cond, msg):
        if cond: warns.append(msg)

    _add(n_kept < 100,            "LOW NODES: fewer than 100 nodes kept — statistical power may be limited.")
    _add(frac_strict is not np.nan and frac_strict < 0.5, "LOW STRICT PASS: <50% strict QC passes — check tangent quality / angle gate.")
    _add(np.isfinite(frac_edge) and frac_edge > 0.20,     f"M AT BRACKET EDGE: {frac_edge*100:.1f}% at ±{edge_eps} of bracket — consider widening bracket or re-check m solver.")
    _add(np.isfinite(R_med) and R_med > 0.60,             f"HIGH RESIDUALS: median R(m)={R_med:.3f} — geometry/segmentation may be noisy.")
    _add(np.isfinite(parent_ambig_frac) and parent_ambig_frac > 0.10, f"PARENT AMBIGUITY: {parent_ambig_frac*100:.1f}% nodes ambiguous — consider stronger gating or tie-breakers.")
    _add(images_with_nodes < images_done,                 f"SPARSE IMAGES: {images_done - images_with_nodes} image(s) produced zero kept nodes — check segmentation or QC thresholds.")

    # ---------- artifact pointers ----------
    fig_dir = FIG_ROOT / dataset
    artifacts = [
        f"{fig_dir}/dataset_{dataset}__residual_hist.png",
        f"{fig_dir}/dataset_{dataset}__residual_baselines.png",
        f"{fig_dir}/dataset_{dataset}__paired_deltaR_ecdf.png",
        f"{fig_dir}/dataset_{dataset}__theta_vs_m_scatter.png",
        f"{fig_dir}/dataset_{dataset}__ablation_medianR_grid.png",
        f"{fig_dir}/dataset_{dataset}__QC_tradeoff_curves.png"
    ]

    # ---------- compose summary ----------
    content = [
        f"===== RUN SUMMARY — {dataset} =====",
        f"Dataset base: {dataset_base}",
        "",
        "[CONFIG]",
        *cfg_lines,
        "",
        "[COUNTS]",
        f"Images processed   : {images_done}/{n_images_total} (with nodes in {images_with_nodes})",
        f"Nodes raw→deg3→kept: {n_raw} → {n_deg3} → {n_kept}  | kept/raw {kept_over_raw:.1f}%  kept/deg3 {kept_over_deg3:.1f}%",
        "",
        "[QC & METRICS]",
        f"QC loose/strict    : {N_qc_loose}/{n_kept} ({frac_loose:.1%})  |  {N_qc_strict}/{n_kept} ({frac_strict:.1%})" if n_kept>0 else "QC loose/strict    : n/a",
        f"m median (IQR)     : {m_med:.3f}  ({m_iqr[0]:.3f},{m_iqr[1]:.3f})   mean={m_mean:.3f}",
        f"R(m) median        : {R_med:.3f}" if np.isfinite(R_med) else "R(m) median        : n/a",
        f"m at bracket edge  : {n_edge}  ({frac_edge:.1%}) at ±{edge_eps:.2f}",
        f"angle_min per image: median={angle_gate_med:.1f}°  range=({angle_gate_rng[0]:.1f}°, {angle_gate_rng[1]:.1f}°)",
        f"parent ambiguity   : {parent_ambig_frac:.1%}" if np.isfinite(parent_ambig_frac) else "parent ambiguity   : n/a",
        "",
        "[BASELINES]",
        (" | ".join(lines_baseline) if lines_baseline else "n/a"),
        "",
        "[SKIP REASONS]",
        skip_list,
        "",
        "[PREREG/ABLATION]",
        prereg_line,
        ablation_line,
        "",
        "[ARTIFACTS]",
        *[str(p) for p in artifacts],
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

def _parse_prereg_c1_from_file(dataset_tag, base_reports_dir):
    """
    Parse reports/<dataset_tag>/PREREG__C1.txt if present.
    Returns dict with keys: p_med, delta, outcome, p_med_jit, delta_jit, outcome_jit.
    Missing values -> None.
    """
    import re
    from pathlib import Path
    out = {"p_med": None, "delta": None, "outcome": None,
           "p_med_jit": None, "delta_jit": None, "outcome_jit": None}
    p = Path(base_reports_dir) / dataset_tag / "PREREG__C1.txt"
    if not p.exists():
        return out
    txt = p.read_text()
    m_p = re.search(r"one-sided p_median\s*=\s*([0-9eE\.\-+]+)", txt)
    m_d = re.search(r"Cliff's delta\s*=\s*([0-9eE\.\-+]+)", txt)
    m_o = re.search(r"Outcome:\s*(PASS|FAIL)", txt)
    if m_p: out["p_med"] = float(m_p.group(1))
    if m_d: out["delta"] = float(m_d.group(1))
    if m_o: out["outcome"] = m_o.group(1)
    # Jitter block (if present)
    m_pj = re.search(r"p_median_jit\s*=\s*([0-9eE\.\-+]+)", txt)
    m_dj = re.search(r"Cliff's delta_jit\s*=\s*([0-9eE\.\-+]+)", txt)
    m_oj = re.search(r"Outcome \(jitter\):\s*(PASS|FAIL)", txt)
    if m_pj: out["p_med_jit"] = float(m_pj.group(1))
    if m_dj: out["delta_jit"] = float(m_dj.group(1))
    if m_oj: out["outcome_jit"] = m_oj.group(1)
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
        # The four prereg segmentation strata used by --seg-robust
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

    # Normalize axes array shapes for 1-row/1-col cases
    if n_rows == 1:
        axes = np.array([axes])
    if n_cols == 1:
        axes = axes[:, np.newaxis]

    for i, ds in enumerate(datasets):
        tsv_path = reports_dir / ds / "SEGROBUST__summary.tsv"
        if not tsv_path.exists():
            # No summary for this dataset: blank row
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

            # Pull prereg stats (worst-case p-values and Cliff's δ) for this dataset+variant
            p_med = None
            p_frac = None
            delta = None
            prereg_dir = reports_dir / f"{ds}__{variant}"
            prereg_path = prereg_dir / "PREREG__C1.txt"
            if prereg_path.exists():
                txt = prereg_path.read_text()

                def _grab(pattern: str):
                    m = re.search(pattern, txt)
                    return float(m.group(1)) if m else None

                # Uses the "Worst-case across nulls: p_med=..., p_frac=..., max δ=..." line
                p_med  = _grab(r"p_med=([0-9eE\.\-+]+)")
                p_frac = _grab(r"p_frac=([0-9eE\.\-+]+)")
                delta  = _grab(r"max δ=([0-9eE\.\-+]+)")

            # Traffic-light coloring
            if (
                (p_med is not None)
                and (p_frac is not None)
                and (delta is not None)
                and (p_med <= 1e-3)
                and (p_frac <= 1e-3)
                and (delta <= -0.20)
            ):
                face = "#d1f2d1"   # green: prereg thresholds all met
            elif (p_med is not None) or (p_frac is not None) or (delta is not None):
                face = "#ffe7b3"   # amber: partial / mixed evidence
            else:
                face = "#f8cccc"   # red: missing prereg or failed tests
            ax.set_facecolor(face)

            # Median + 95% CI whisker at x=0.5
            ax.plot([0.5, 0.5], [ci_lo, ci_hi], color="black", linewidth=1.5)
            ax.plot([0.5], [r_med], "o", color="black", markersize=4)

            # Quality bands (strict / loose)
            ax.axhline(0.55, linestyle="--", linewidth=0.8)
            ax.axhline(0.85, linestyle="--", linewidth=0.8)

            ax.set_xlim(0.0, 1.0)
            ax.set_ylim(0.0, 1.5)
            ax.set_xticks([])
            ax.set_yticks([0.0, 0.55, 0.85, 1.2])
            ax.grid(axis="y", alpha=0.15, linestyle="-", linewidth=0.7)

            # Text badge
            lines = [
                f"N={N_nodes}",
                f"median={r_med:.3f}",
                f"95%[{ci_lo:.3f},{ci_hi:.3f}]",
            ]
            if p_med is not None:
                lines.append(f"p̃={p_med:.1e}")
            if p_frac is not None:
                lines.append(f"p<0.55={p_frac:.1e}")
            if delta is not None:
                lines.append(f"δ={delta:.2f}")

            ax.text(
                0.02,
                0.98,
                "\n".join(lines),
                transform=ax.transAxes,
                va="top",
                ha="left",
                fontsize=7,
                bbox=dict(facecolor="white", alpha=0.9, edgecolor="none"),
            )

            if i == 0:
                ax.set_title(variant.replace("+", " + "), fontsize=9)

        # Row label with dataset name
        axes[i, 0].set_ylabel(ds, rotation=0, labelpad=35, fontsize=10, va="center")

    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.95])

    out_path = fig_root / "ALL_DATASETS__segrobust_scoreboard.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=300)
    plt.close(fig)



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
    parser = argparse.ArgumentParser(description="EPIC Angle-Only Estimator (per-node m inversion and tariff tomography)")

    parser.add_argument("--data-root", type=str, default=str(DATA_ROOT_DEFAULT),
                        help="Base folder containing dataset subfolders (HRF, DRIVE, STARE, CHASE_DB1, etc.)")
    parser.add_argument("--datasets", nargs="+", default=["HRF"],
                            help="One or more dataset folder names under data-root")
    parser.add_argument("--max-images", type=int, default=None,
                            help="Optional limit per dataset (for quick runs)")
    parser.add_argument("--sample-frac", type=float, default=None,
                            help="Optional fraction (0,1] to randomly sample images per dataset before processing (e.g., 0.25)")
    parser.add_argument("--sample-seed", type=int, default=13,
                            help="RNG seed for --sample-frac sampling")
    parser.add_argument("--save-debug", action="store_true",
                            help="Save per-image skeleton overlays & tariff maps")

    parser.add_argument("--seg-method", type=str, default="frangi", choices=["frangi", "sato"],
                        help="Vesselness method when mask not provided")
    parser.add_argument("--thresh-method", type=str, default="otsu", choices=["otsu", "quantile"],
                        help="Thresholding method for vesselness map (when mask not provided)")
    parser.add_argument("--seg-robust", action="store_true",
                        help="Run 4 segmentation strata: (Frangi|Sato) × (Otsu|Quantile) and write stratified outputs")


    # Angle gating
    parser.add_argument("--min-angle", type=float, default=10.0, help="Min daughter-daughter angle (deg)")
    parser.add_argument("--max-angle", type=float, default=170.0, help="Max daughter-daughter angle (deg)")
    parser.add_argument("--angle-auto", action="store_true", help="Enable per-image automatic lower angle gate")
    parser.add_argument("--angle-auto-pctl", type=float, default=5.0, help="Percentile for automatic lower angle gate")
    parser.add_argument("--min-angle-floor", type=float, default=10.0, help="Floor for automatic lower angle gate (deg)")
    parser.add_argument(
        "--angle-soft-margin",
        type=float,
        default=3.0,
        help="Soft rescue margin (deg): if θ₁₂ is within this margin outside the gate, clamp to the nearest bound for inversion and tag the node."
    )


    # Geometry robustness
    parser.add_argument("--min-branch-len", type=int, default=10, help="Min branch length in pixels")
    parser.add_argument("--min-radius", type=float, default=1.0, help="Min local radius (pixels)")
    parser.add_argument("--tangent-len", type=int, default=16, help="Pixels from the junction used to fit tangents")
    parser.add_argument("--svd-ratio-min", type=float, default=3.0, help="Minimum PCA singular value ratio (S0/S1) to accept a tangent")
    parser.add_argument("--dedup-radius", type=int, default=3, help="Pixel radius for de-duplicating degree-3 junction pixels")

    # m inversion
    parser.add_argument("--m-min", type=float, default=0.2, help="Lower bracket for m")
    parser.add_argument("--m-max", type=float, default=4.0, help="Upper bracket for m")

    # QC regime
    parser.add_argument("--strict-qc", action="store_true",
                        help="Also compute/report a stricter QC with tighter residual thresholds")

    # Stability & profiles
    parser.add_argument("--radius-jitter-frac", type=float, default=0.0,
                        help="Optional multiplicative jitter fraction for stability check (e.g., 0.03 for ±3%)")
    parser.add_argument("--px-size-um", type=float, default=float("nan"),
                        help="Optional pixel size (µm/px) recorded in CSV; does not change angle-only m")
    parser.add_argument("--profile", type=str, default="none", choices=["none", "prereg-2025Q4"],
                        help="Frozen profile to eliminate researcher d.o.f.; 'prereg-2025Q4' sets all knobs")

    # Panel B nulls + uncertainty + prereg knobs
    parser.add_argument("--null-perm", type=int, default=2000,
                        help="Permutations per negative control for R(m) nulls (shuffle/swap/randtheta/rshuffle)")
    parser.add_argument("--boot", type=int, default=5000,
                        help="Bootstrap replicates for 95% CI on median R(m)")

    # Primary metric routing for Panel B / prereg:
    parser.add_argument("--primary-metric", type=str, default="asconfigured",
                        choices=["asconfigured", "heldout"],
                        help="Which residual to plot and test in Panel B & PREREG report: "
                             "'asconfigured' (legacy, m chosen by minimal R) or "
                             "'heldout' (m from radii+angle only; directions held out).")

    # Gaussian jitter (independent per radius, SD = fraction × radius). When enabled,
    # plot_residual_distribution can re-infer m_i under jitter if requested.
    parser.add_argument("--radius-jitter-sd", type=float, default=0.0,
                        help="Independent Gaussian jitter SD as a fraction of radius (e.g., 0.10).")
    parser.add_argument("--reinvert-m-on-jitter", action="store_true",
                        help="Under Gaussian jitter, re-infer m_i from (r0,r1,r2,theta12) per node.")

    # Positive control (analytic)
    parser.add_argument("--posctrl-n", type=int, default=1500,
                        help="Number of analytic synthetic junctions for positive-control recovery plot")
    parser.add_argument("--posctrl-dir-noise-deg", type=float, default=3.0,
                        help="Per-vector direction jitter (degrees) for positive control")
    parser.add_argument("--posctrl-radius-noise", type=float, default=0.05,
                        help="Multiplicative radius noise SD (fraction) for positive control")
    parser.add_argument("--posctrl-parent-mislabel", type=float, default=0.0,
                        help="Fraction [0,1] of synthetic nodes with wrong parent (positive-control realism)")

    # ===== VALIDATION SUITE (hard PASS/FAIL) =====
    parser.add_argument("--validate", action="store_true",
                        help="Run prereg-style validation suite and emit a single SUPPORT/MIXED/NOSUPPORT verdict per dataset")
    parser.add_argument("--alpha-prereg", type=float, default=1e-3,
                        help="Primary prereg alpha for one-sided tests on median and Pr[R<0.55]")
    parser.add_argument("--parentdir-alpha", type=float, default=1e-4,
                        help="Alpha for parent-direction Δφ>0 paired sign tests vs baselines")
    parser.add_argument("--equip-cvS-target", type=float, default=0.30,
                        help="Equipartition target: median CV(S) must be ≤ this")
    parser.add_argument("--equip-cvK-target", type=float, default=0.30,
                        help="Equipartition target: median CV(κ) must be ≤ this")
    parser.add_argument("--panelC-share-min", type=float, default=0.70,
                        help="Panel C: fraction within |Δm|≤0.25 must be ≥ this")
    parser.add_argument("--snell-r2-min", type=float, default=0.50,
                        help="Snell demo: minimum R^2 required for PASS (if demo files present)")
    parser.add_argument("--snell-min-crossings", type=int, default=30,
                        help="Snell demo: minimum counted crossings to score PASS/FAIL (else marked NA)")
    parser.add_argument("--verdict-scope", type=str, default="dataset",
                        choices=["dataset", "variant"],
                        help="Scoreboard scope: aggregate to dataset-level (recommended) or per-variant")
    parser.add_argument("--verdict-name", type=str, default="default",
                        help="Optional tag baked into VERDICT__*.{txt,json}")

    args = parser.parse_args()

    # Freeze knobs if a profile is requested (mitigate researcher d.o.f.)
    if args.profile == "prereg-2025Q4":
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


    # Configs
    seg_cfg = SegConfig(vesselness_method=args.seg_method)
    qc_cfg = QCConfig(
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
        strict_qc=args.strict_qc
    )


    data_root = Path(args.data_root)
    if not data_root.exists():
        log(f"[ERROR] data-root does not exist: {data_root}")
        sys.exit(1)

    overall_start = time.time()

    for dataset in args.datasets:
        ds_dir = data_root / dataset
        if not ds_dir.exists():
            log(f"[WARN] Dataset folder not found: {ds_dir} — skipping.")
            continue

        log(f"========== DATASET: {dataset} ==========")

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
                n_keep  = max(1, int(math.ceil(n_total * f)))
                rng_samp = np.random.default_rng(int(args.sample_seed))
                sel_idx = np.sort(rng_samp.choice(n_total, size=n_keep, replace=False))
                img_paths = [img_paths[i] for i in sel_idx]
                log(f"    [sample] sample_frac={f:.3f} → {n_keep}/{n_total} images (seed={int(args.sample_seed)})")

        from collections import Counter
        ext_hist = Counter([p.suffix.lower() for p in img_paths])
        log("    [preflight] extension histogram: " + ", ".join(f"{k}:{v}" for k,v in sorted(ext_hist.items())))
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

        log(f"Found {len(img_paths)} images (after unpack/conversion and sampling).")


        # ------------------ VARIANTS: as-configured or 4-way strata ------------------
        variant_pairs = [(args.seg_method, args.thresh_method)]
        if args.seg_robust:
            variant_pairs = [("frangi", "otsu"), ("frangi", "quantile"),
                             ("sato", "otsu"),   ("sato", "quantile")]

        # Collect per-variant rows for the final cross-variant table
        segrobust_rows = []

        for seg_method, thresh_method in variant_pairs:
            log(f"---- SEG VARIANT: {seg_method}+{thresh_method} ----")
            dataset_tag = f"{dataset}__{seg_method}+{thresh_method}"

            seg_cfg = SegConfig(vesselness_method=seg_method, thresh_method=thresh_method)
            qc_cfg = QCConfig(
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

            with tqdm(total=stats.total, desc=f"{dataset}:{seg_method}+{thresh_method}", unit="img") as bar:
                for img_path in img_paths:
                    try:
                        recs, diag = analyze_image(
                            img_path=img_path,
                            dataset=dataset_tag,   # variant-scoped overlays & node metadata
                            seg_cfg=seg_cfg,
                            qc=qc_cfg,
                            save_debug=args.save_debug,
                            out_dir=ROOT
                        )
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

                        if args.save_debug and len(recs) > 0:
                            gray = imread_gray(img_path)
                            mask_path = find_mask_for_image(img_path)
                            if mask_path is not None:
                                mask = io.imread(str(mask_path))
                                if mask.ndim == 3:
                                    mask = color.rgb2gray(mask)
                                mask = (mask > 0.5).astype(np.uint8)
                            else:
                                mask = segment_vessels(gray, seg_cfg)
                            skel, _ = skeleton_and_dist(mask)
                            plot_tariff_map(gray, skel, recs, dataset_tag, img_path.stem)

                    except Exception as e:
                        log(f"[ERROR] Image {img_path.name}: {e}")
                        stats.done += 1
                        bar.set_postfix({
                            "done": f"{stats.done}/{stats.total}",
                            "kept": stats.nodes_kept,
                            "deg3": stats.nodes_dedup,
                            "left": stats.total - stats.done,
                        })
                        bar.update(1)

            dt = human_time(time.time() - t0_dataset)
            log(
                "---- DATASET PROGRESS ----\n"
                f"dataset={dataset} | variant={seg_method}+{thresh_method} | images {stats.done}/{stats.total} | "
                f"nodes raw→deg3→kept {stats.nodes_raw}→{stats.nodes_dedup}→{stats.nodes_kept} | "
                f"skips: " + ", ".join(f"{k}={v}" for k, v in sorted(stats.skips.items(), key=lambda kv: -kv[1])) + " | "
                f"elapsed {dt}\n"
                "--------------------------"
            )

            if len(all_records) == 0:
                log(f"[WARN] No nodes kept for {dataset_tag}; skipping plots/summary.")
                continue

            # Save CSV
            csv_path = CSV_ROOT / dataset_tag / f"nodes__{dataset_tag}.csv"
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            df = pd.DataFrame([r.__dict__ for r in all_records])

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
            }
            for k, v in cfg_snapshot.items():
                df[f"cfg_{k}"] = v
            df["args_profile"] = args.profile
            df["args_seg_method"] = seg_cfg.vesselness_method
            df["args_thresh_method"] = seg_cfg.thresh_method
            df["args_radius_jitter_frac"] = args.radius_jitter_frac

            df.to_csv(csv_path, index=False)
            log(f"Wrote node-level CSV: {csv_path}")

            # Plots & prereg selection (transparent, stage-by-stage)
            variant_fig_dir = FIG_ROOT / dataset_tag
            variant_rep_dir = CSV_ROOT / dataset_tag
            dataset_base    = dataset_tag.split("__", 1)[0]
            base_fig_dir    = FIG_ROOT / dataset_base
            base_rep_dir    = CSV_ROOT / dataset_base

            log(f"[POST] Starting post-processing for {dataset_tag}")
            log(f"[POST] Panel A — m distribution → {variant_fig_dir}/dataset_{dataset_tag}__m_hist.png")
            try:
                plot_m_distribution(df, dataset_tag)
            except Exception as e:
                log(f"[POST][WARN] Panel A failed: {e}")

            m_col = "m_node" if args.primary_metric == "asconfigured" else "m_angleonly"
            R_col = "R_m"    if args.primary_metric == "asconfigured" else "R_m_holdout"
            metric_name = "heldout" if R_col == "R_m_holdout" else "asconfigured"
            log(
                f"[POST] Panel B/C1 — residual histogram (metric={metric_name}) "
                f"→ {variant_fig_dir}/dataset_{dataset_tag}__residual_hist.png; "
                f"nulls report → {variant_rep_dir}/NULLTEST__R_m.txt; prereg (if heldout) → {variant_rep_dir}/PREREG__C1.txt"
            )
            try:
                plot_residual_distribution(
                    df,
                    dataset_tag,
                    n_perm=args.null_perm,
                    boot=args.boot,
                    include_nulls=("shuffle", "swap", "randtheta", "rshuffle"),
                    jitter_frac=args.radius_jitter_frac,
                    m_col=m_col,
                    R_col=R_col,
                    gaussian_jitter_sd=args.radius_jitter_sd,
                    reinvert_m_on_jitter=args.reinvert_m_on_jitter
                )
            except Exception as e:
                log(f"[POST][WARN] Panel B/C1 failed: {e}")

            log(
                f"[POST] Baselines — histograms and paired ΔR ECDF "
                f"→ {variant_fig_dir}/dataset_{dataset_tag}__residual_baselines.png, "
                f"{variant_fig_dir}/dataset_{dataset_tag}__paired_deltaR_ecdf.png; "
                f"text → {variant_rep_dir}/BASELINES__R_m.txt"
            )
            try:
                plot_fixed_m_baselines(df, dataset_tag)
            except Exception as e:
                log(f"[POST][WARN] Baselines failed: {e}")

            log(f"[POST] Parent-direction error (φ₀) → {variant_fig_dir}/dataset_{dataset_tag}__parent_dir_error_ecdf.png")
            try:
                plot_parent_direction_error(df, dataset_tag)   # STEP 5
            except Exception as e:
                log(f"[POST][WARN] Parent-direction failed: {e}")

            log(f"[POST] Panel C — θ vs m scatter → {variant_fig_dir}/dataset_{dataset_tag}__theta_vs_m_scatter.png; test → {base_rep_dir}/PANELC__theta_vs_m_test.txt")
            try:
                plot_theta_vs_m_scatter(df, dataset_tag, symmetric_tol=1.08)
            except Exception as e:
                log(f"[POST][WARN] Panel C failed: {e}")

            log(f"[POST] STEP 3 — per-node uncertainty (if enabled) → {variant_rep_dir}/UNCERTAINTY__nodes.csv")
            try:
                compute_per_node_uncertainty(df, dataset_tag)    # STEP 3
            except Exception as e:
                log(f"[POST][WARN] Uncertainty step failed: {e}")

            log(f"[POST] STEP 4 — ablation grid → {variant_fig_dir}/dataset_{dataset_tag}__ablation_medianR_grid.png; test → {variant_rep_dir}/ABLATION__test.txt")
            try:
                run_ablation_grid(df, dataset_tag)               # STEP 4 (test)
            except Exception as e:
                log(f"[POST][WARN] Ablation step failed: {e}")

            log(f"[POST] Positive control → {variant_fig_dir}/POSCTRL__recovery.png; text → {variant_rep_dir}/POSCTRL__recovery.txt")
            try:
                plot_positive_control_recovery(
                    dataset_tag,
                    n_samples=args.posctrl_n,
                    noise_dir_deg=args.posctrl_dir_noise_deg,
                    noise_radius_frac=args.posctrl_radius_noise,
                    parent_mislabel_frac=args.posctrl_parent_mislabel,
                )
            except Exception as e:
                log(f"[POST][WARN] Positive control failed: {e}")

            log(f"[POST] SUMMARY → {variant_rep_dir}/SUMMARY__{dataset_tag}.txt")
            try:
                summarize_and_write(
                    df=df,
                    dataset=dataset_tag,
                    diags=diags_all,
                    n_images_total=len(img_paths),
                    elapsed_sec=time.time() - t0_dataset
                )
            except Exception as e:
                log(f"[POST][WARN] Summary failed: {e}")

            # Append a one-line row for the cross-variant table (heldout as primary by your command)
            segrobust_rows.append(
                _make_variant_row(
                    df=df,
                    dataset_tag=dataset_tag,
                    reports_dir=CSV_ROOT,
                    primary_metric="heldout" if args.primary_metric == "heldout" else "asconfigured"
                )
            )
            log(f"[POST] Completed post-processing for {dataset_tag}")

        # After all variants: write the cross-variant summary table once
        _write_segrobust_summary(
            dataset=dataset,
            rows=segrobust_rows,
            reports_dir=CSV_ROOT,
            log_fn=log
        )

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
    # (only when segmentation strata are enabled and the held-out metric is used)
    try:
        if args.seg_robust and args.primary_metric == "heldout":
            plot_segrobust_scoreboard(
                datasets=args.datasets,
                reports_dir=CSV_ROOT,
                fig_root=FIG_ROOT,
            )
    except Exception as e:
        log(f"[SCOREBOARD][WARN] cross-dataset scoreboard failed: {e}")

    total_dt = human_time(time.time() - overall_start)
    log(f"ALL DONE in {total_dt}. See figures/, reports/, logs/ under {ROOT}")


if __name__ == "__main__":

    try:
        main()
    except KeyboardInterrupt:
        log("Interrupted by user. Exiting.")

