#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_epic.py — Quick, summary-only runner for arXiv v2 (plus full plotting pipeline if needed).

Quick mode (what you asked for)
-------------------------------
Usage:
  ./plot_epic.py HRF_sato_otsu [DATA_ROOT]
or
  ./plot_epic.py HRF__sato+otsu [DATA_ROOT]

Effect:
  • Normalizes tag (e.g., HRF_sato_otsu -> HRF__sato+otsu).
  • Looks for files first in DATA_ROOT (default='.'), then in DATA_ROOT/reports/<TAG>/.
  • Reads:
      nodes__<TAG>.csv
      PREREG__C1.txt
      ABLATION__test.txt
      BASELINES__R_m.txt
      PARENTDIR__phi.txt
      SUMMARY__<TAG>.txt
      UNCERTAINTY__nodes.csv (optional)
  • Computes medians, bootstrap CIs, edge counts, QC shares, etc.
  • Prints a concise terminal summary and writes PAPER_SUMMARY__<TAG>.md.
  • Produces NO figures.

Full pipeline (unchanged, optional)
-----------------------------------
You can still run the plotting pipeline with flags if you want:

  python plot_epic.py --data-root /path/to/EPIC --datasets HRF \
    --variants sato+otsu --primary-metric heldout --fig-size single \
    --ext pdf png --scoreboard

Author
------
EPIC publication helper — v2 quick summary focused.
"""

from __future__ import annotations
import argparse
import json
import math
import os
import re
import sys
import textwrap
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# Matplotlib is kept for compatibility with the full pipeline; not used in quick mode.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ----------------------- Defaults & Style (for optional full pipeline) -----------------------

ROOT_DEFAULT = Path(".").expanduser()
REPORTS = "reports"
FIGURES_PUB = "figures_pub"

PUB_RCPARAMS = {
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "font.size": 9.5,
    "font.sans-serif": ["DejaVu Sans"],
    "axes.titlesize": 10.5,
    "axes.labelsize": 10.0,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": "-",
    "grid.linewidth": 0.7,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "legend.frameon": True,
    "legend.framealpha": 0.9,
}
FIGSIZE_PRESETS = {"single": (3.5, 2.4), "onehalf": (4.75, 3.0), "double": (7.2, 4.6)}

# ----------------------- Utilities -----------------------

def human_time(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    m, s = divmod(seconds, 60)
    if m < 60:
        return f"{int(m)}m {int(s)}s"
    h, m = divmod(m, 60)
    return f"{int(h)}h {int(m)}m"

def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p

def bootstrap_ci_median(x: np.ndarray, B: int = 5000, seed: int = 13) -> Tuple[float, float]:
    rng = np.random.default_rng(int(seed))
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan"), float("nan")
    meds = np.median(rng.choice(x, size=(int(B), x.size), replace=True), axis=1)
    lo, hi = np.percentile(meds, [2.5, 97.5])
    return float(lo), float(hi)

def normalize_tag(arg: str) -> str:
    """
    Accept "HRF_sato_otsu" or "HRF__sato+otsu" and return "HRF__sato+otsu".
    Rule: first underscore -> "__"; remaining underscores -> "+".
    """
    if "__" in arg:
        return arg
    if "_" in arg:
        parts = arg.split("_", 1)
        ds = parts[0]
        rest = parts[1].replace("_", "+")
        return f"{ds}__{rest}"
    return arg

def probable_bases(data_root: Path, tag: str) -> List[Path]:
    """
    Search order for files:
      1) data_root/
      2) data_root/reports/<TAG>/
      3) parent_of_data_root/reports/<TAG>/  (e.g. ../reports/<TAG> when run from macro/)
    """
    bases: List[Path] = []

    # 1) data_root/
    bases.append(data_root)

    # 2) data_root/reports/<TAG>/
    bases.append(data_root / REPORTS / tag)

    # 3) parent_of_data_root/reports/<TAG>/
    parent_reports = data_root.parent / REPORTS / tag
    if parent_reports not in bases:
        bases.append(parent_reports)

    return bases


def find_file(data_root: Path, tag: str, filename: str) -> Optional[Path]:
    for base in probable_bases(data_root, tag):
        p = base / filename
        if p.exists():
            return p
    return None

def read_csv_relaxed(path: Path) -> pd.DataFrame:
    """
    Tolerant CSV reader: skips a leading commented line and coerces numerics.
    """
    if path is None or not path.exists():
        raise FileNotFoundError(f"Missing CSV: {path}")
    txt = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    from io import StringIO
    if txt and txt[0].lstrip().startswith("#"):
        df = pd.read_csv(StringIO("\n".join(txt[1:])))
    else:
        df = pd.read_csv(StringIO("\n".join(txt)))
    return df

def read_nodes_anywhere(data_root: Path, tag: str) -> pd.DataFrame:
    csvname = f"nodes__{tag}.csv"
    path = find_file(data_root, tag, csvname)
    if path is None:
        raise FileNotFoundError(f"Could not find {csvname} under {data_root} or {data_root}/{REPORTS}/{tag}")
    df = read_csv_relaxed(path)
    # Coerce expected columns (ignore if absent)
    for col in ["m_node","R_m","m_angleonly","R_m_holdout","theta12_deg","r0","r1","r2",
                "e0x","e0y","e1x","e1y","e2x","e2y",
                "svd_ratio_e0","svd_ratio_e1","svd_ratio_e2",
                "cfg_m_min","cfg_m_max"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in ["qc_pass","qc_pass_strict","parent_ambiguous"]:
        if col in df.columns:
            df[col] = df[col].astype(bool)
    if "cfg_m_min" not in df.columns: df["cfg_m_min"] = 0.2
    if "cfg_m_max" not in df.columns: df["cfg_m_max"] = 4.0
    return df

def maybe_read_text(data_root: Path, tag: str, filename: str) -> Optional[str]:
    p = find_file(data_root, tag, filename)
    return p.read_text(encoding="utf-8", errors="ignore") if p else None

# ----------------------- Quick summary logic -----------------------

def compute_core_stats(df: pd.DataFrame, primary_metric: str = "heldout", boot: int = 5000) -> Dict[str, float]:
    n = len(df)
    strict = float(df["qc_pass_strict"].mean()) if "qc_pass_strict" in df.columns and n else float("nan")
    loose  = float(df["qc_pass"].mean())        if "qc_pass" in df.columns and n else float("nan")
    m_med  = float(np.nanmedian(df["m_node"])) if "m_node" in df.columns and n else float("nan")
    if "m_node" in df.columns and n:
        m_q25 = float(np.nanpercentile(df["m_node"], 25))
        m_q75 = float(np.nanpercentile(df["m_node"], 75))
    else:
        m_q25 = m_q75 = float("nan")
    m_min = float(df["cfg_m_min"].iloc[0]) if "cfg_m_min" in df.columns and n else 0.2
    m_max = float(df["cfg_m_max"].iloc[0]) if "cfg_m_max" in df.columns and n else 4.0
    at_edge = int((((df.get("m_node", pd.Series([])) - m_min).abs() <= 0.02) | ((df.get("m_node", pd.Series([])) - m_max).abs() <= 0.02)).sum()) if n else 0
    parent_ambig = float(df["parent_ambiguous"].mean()) if "parent_ambiguous" in df.columns and n else float("nan")

    R_col = "R_m_holdout" if primary_metric == "heldout" and "R_m_holdout" in df.columns else "R_m"
    R_vals = df.get(R_col, pd.Series([], dtype=float)).astype(float).to_numpy()
    R_vals = R_vals[np.isfinite(R_vals)]
    R_med  = float(np.median(R_vals)) if R_vals.size else float("nan")
    R_lo, R_hi = bootstrap_ci_median(R_vals, B=int(boot), seed=13) if R_vals.size else (float("nan"), float("nan"))

    return dict(
        n=n, strict=strict, loose=loose,
        m_med=m_med, m_q25=m_q25, m_q75=m_q75, m_min=m_min, m_max=m_max, at_edge=at_edge,
        parent_ambig=parent_ambig,
        R_med=R_med, R_lo=R_lo, R_hi=R_hi, R_col=R_col
    )

def parse_prereg_c1(txt: Optional[str]) -> Dict[str, Optional[str]]:
    d = dict(outcome=None, R_obs=None, R_ci=None, worst_p_med=None, worst_p_frac=None, worst_delta=None)
    if not txt: return d
    m_out = re.search(r"Outcome:\s*(PASS|FAIL)", txt)
    if m_out: d["outcome"] = m_out.group(1)
    m_R   = re.search(r"Observed\s*[\~\u007E]?\s*R\s*=\s*([0-9\.\-eE]+).*?\[boot\s*95% CI\s*([0-9\.\-eE]+),\s*([0-9\.\-eE]+)\]", txt)
    if m_R:
        d["R_obs"] = f"{float(m_R.group(1)):.6f}"
        d["R_ci"]  = f"[{float(m_R.group(2)):.6f}, {float(m_R.group(3)):.6f}]"
    # Support the format in your text file
    if not d["R_obs"]:
        m_R2 = re.search(r"Observed\s*\\tilde R\s*=\s*([0-9\.\-eE]+).*?\[boot\s*95% CI\s*([0-9\.\-eE]+),\s*([0-9\.\-eE]+)\]", txt)
        if m_R2:
            d["R_obs"] = f"{float(m_R2.group(1)):.6f}"
            d["R_ci"]  = f"[{float(m_R2.group(2)):.6f}, {float(m_R2.group(3)):.6f}]"
    m_wpmed = re.search(r"p_med(?:ian)?=([0-9eE\.\-+]+)", txt)
    m_wpfr  = re.search(r"p_frac[^\=]*=\s*([0-9eE\.\-+]+)", txt)
    m_wdel  = re.search(r"Cliff['’]s?\s*δ\s*=\s*([0-9eE\.\-+]+)|max\s*δ\s*=\s*([0-9eE\.\-+]+)", txt)
    if m_wpmed: d["worst_p_med"] = m_wpmed.group(1)
    if m_wpfr:  d["worst_p_frac"] = m_wpfr.group(1)
    if m_wdel:
        d["worst_delta"] = m_wdel.group(1) or m_wdel.group(2)
    return d

def parse_simple_outcome(txt: Optional[str]) -> Optional[str]:
    if not txt: return None
    m = re.search(r"Outcome:\s*(PASS|FAIL)", txt)
    return m.group(1) if m else None

def parse_baselines(txt: Optional[str]) -> Dict[str, Optional[str]]:
    """
    Expect lines like:
      m=2: median=0.262, ... | m=1: median=0.218, ... | m*=1.18: median=0.215, ...
    Returns a compact dict of medians if we can parse them.
    """
    out = dict(m2=None, m1=None, mstar=None)
    if not txt: return out
    m2 = re.search(r"m=2:\s*median=([0-9\.\-eE]+)", txt)
    m1 = re.search(r"m=1:\s*median=([0-9\.\-eE]+)", txt)
    ms = re.search(r"m\*\s*=\s*([0-9\.\-eE]+)\s*:\s*median=([0-9\.\-eE]+)", txt)
    if m2: out["m2"] = m2.group(1)
    if m1: out["m1"] = m1.group(1)
    if ms: out["mstar"] = f"m*={ms.group(1)} → median={ms.group(2)}"
    return out

def write_markdown_summary(tag: str,
                           core: Dict[str, float],
                           prereg: Dict[str, Optional[str]],
                           ablation_outcome: Optional[str],
                           baselines: Dict[str, Optional[str]],
                           parentdir_txt: Optional[str],
                           run_summary_txt: Optional[str],
                           out_dir: Path) -> Path:
    md = out_dir / f"PAPER_SUMMARY__{tag}.md"
    lines = []
    lines.append(f"# EPIC Paper Summary — `{tag}`")
    lines.append("")
    lines.append(f"- **Nodes**: {core['n']}")
    if math.isfinite(core["strict"]): lines.append(f"- **QC (strict / loose)**: {core['strict']:.1%} / {core['loose']:.1%}")
    if math.isfinite(core["m_med"]):  lines.append(f"- **m median (IQR)**: {core['m_med']:.3f} ({core['m_q25']:.3f}, {core['m_q75']:.3f})")
    if math.isfinite(core["R_med"]):  lines.append(f"- **Residual ({core['R_col']}) median**: {core['R_med']:.3f}  95% CI [{core['R_lo']:.3f}, {core['R_hi']:.3f}]")
    lines.append(f"- **m at bracket edge (±0.02)**: {core['at_edge']} / {core['n']}")
    if math.isfinite(core["parent_ambig"]): lines.append(f"- **Parent ambiguity**: {core['parent_ambig']:.1%}")
    lines.append("")
    lines.append("## Tests & Verdicts")
    if prereg["outcome"]:
        extras = []
        if prereg["R_obs"] and prereg["R_ci"]:
            extras.append(f"observed \\~R={prereg['R_obs']} 95% CI {prereg['R_ci']}")
        if prereg["worst_p_med"] and prereg["worst_p_frac"] and prereg["worst_delta"]:
            extras.append(f"worst p_med={prereg['worst_p_med']}, p_frac={prereg['worst_p_frac']}, max δ={prereg['worst_delta']}")
        tail = " | " + " | ".join(extras) if extras else ""
        lines.append(f"- **PREREG (heldout)**: {prereg['outcome']}{tail}")
    else:
        lines.append(f"- **PREREG (heldout)**: n/a")
    lines.append(f"- **Ablation**: {ablation_outcome or 'n/a'}")
    if any(v for v in baselines.values()):
        m2 = f"m=2 → median={baselines['m2']}" if baselines["m2"] else None
        m1 = f"m=1 → median={baselines['m1']}" if baselines["m1"] else None
        ms = baselines["mstar"]
        bline = " | ".join([s for s in [m2, m1, ms] if s])
        if bline:
            lines.append(f"- **Baselines**: {bline}")
    lines.append("")
    if parentdir_txt:
        lines.append("## Parent-direction notes")
        lines.append("```")
        # keep it short
        lines.append("\n".join(parentdir_txt.strip().splitlines()[:12]))
        lines.append("```")
        lines.append("")
    if run_summary_txt:
        lines.append("## Run log (truncated)")
        lines.append("```")
        lines.append("\n".join(run_summary_txt.strip().splitlines()[:30]))
        lines.append("```")
        lines.append("")
    md.write_text("\n".join(lines), encoding="utf-8")
    return md

def quick_run_single(tag_arg: str, data_root: Optional[str] = None) -> int:
    t0 = time.time()
    tag = normalize_tag(tag_arg)
    base = Path(data_root).expanduser().resolve() if data_root else Path(".").resolve()

    # Load core nodes
    try:
        df = read_nodes_anywhere(base, tag)
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        return 2

    # Compute summary numbers
    core = compute_core_stats(df, primary_metric="heldout", boot=5000)

    # Parse supportive artifacts (best-effort)
    prereg_txt   = maybe_read_text(base, tag, "PREREG__C1.txt")
    ablation_txt = maybe_read_text(base, tag, "ABLATION__test.txt") or maybe_read_text(base, tag, "ABLATION__summary.txt")
    baselines_txt= maybe_read_text(base, tag, "BASELINES__R_m.txt")
    parentdir_txt= maybe_read_text(base, tag, "PARENTDIR__phi.txt")
    run_summary  = maybe_read_text(base, tag, f"SUMMARY__{tag}.txt")

    prereg = parse_prereg_c1(prereg_txt)
    abl_out= parse_simple_outcome(ablation_txt)
    base_dict = parse_baselines(baselines_txt)

    # Print concise terminal summary
    print(f"===== EPIC Quick Summary — {tag} =====")
    print(f"Nodes: {core['n']}")
    if math.isfinite(core["strict"]):
        print(f"QC strict: {core['strict']:.1%}   |   QC loose: {core['loose']:.1%}")
    if math.isfinite(core["m_med"]):
        print(f"m median (IQR): {core['m_med']:.3f}  ({core['m_q25']:.3f}, {core['m_q75']:.3f})")
    if math.isfinite(core["R_med"]):
        print(f"Residual ({core['R_col']}): median={core['R_med']:.3f}  95% CI [{core['R_lo']:.3f}, {core['R_hi']:.3f}]")
    print(f"m at bracket edge (±0.02): {core['at_edge']} / {core['n']}")
    if math.isfinite(core["parent_ambig"]):
        print(f"Parent ambiguity: {core['parent_ambig']:.1%}")

    if prereg["outcome"]:
        tail = []
        if prereg["R_obs"] and prereg["R_ci"]:
            tail.append(f"~R={prereg['R_obs']} 95% CI {prereg['R_ci']}")
        if prereg["worst_p_med"] and prereg["worst_p_frac"] and prereg["worst_delta"]:
            tail.append(f"worst p_med={prereg['worst_p_med']}, p_frac={prereg['worst_p_frac']}, max δ={prereg['worst_delta']}")
        print(f"PREREG (heldout): {prereg['outcome']}" + ((" | " + " | ".join(tail)) if tail else ""))
    if abl_out:
        print(f"Ablation: {abl_out}")
    if any(v for v in base_dict.values()):
        m2 = f"m=2 → median={base_dict['m2']}" if base_dict["m2"] else None
        m1 = f"m=1 → median={base_dict['m1']}" if base_dict["m1"] else None
        ms = base_dict["mstar"]
        bline = " | ".join([s for s in [m2, m1, ms] if s])
        if bline:
            print(f"Baselines: {bline}")

    # Write markdown summary next to where we found the nodes CSV (base or reports/<tag>)
    out_dir = Path(".")
    md_path = write_markdown_summary(
        tag=tag,
        core=core,
        prereg=prereg,
        ablation_outcome=abl_out,
        baselines=base_dict,
        parentdir_txt=parentdir_txt,
        run_summary_txt=run_summary,
        out_dir=out_dir
    )
    print(f"[OK] Wrote {md_path}  |  elapsed {human_time(time.time()-t0)}")
    return 0

# ======================= Optional: full plotting pipeline (unchanged APIs) =======================

def set_pub_style():
    matplotlib.rcParams.update(PUB_RCPARAMS)

def new_fig(figsize_key: str = "single") -> Tuple[plt.Figure, plt.Axes]:
    sz = FIGSIZE_PRESETS.get(figsize_key, FIGSIZE_PRESETS["single"])
    fig = plt.figure(figsize=sz)
    ax = fig.gca()
    return fig, ax

def save_figure(fig: plt.Figure, out_base: Path, extensions: Sequence[str]):
    out_base.parent.mkdir(parents=True, exist_ok=True)
    for ext in extensions:
        path = out_base.with_suffix("." + ext.lower())
        fig.savefig(str(path))
    plt.close(fig)

def discover_variants(reports_dir: Path, dataset: str, explicit_variants: Optional[Sequence[str]]) -> List[str]:
    ds = str(dataset)
    tags = []
    if explicit_variants:
        for v in explicit_variants:
            tag = f"{ds}__{v}"
            if (reports_dir / tag / f"nodes__{tag}.csv").exists():
                tags.append(tag)
    else:
        for p in (reports_dir).glob(f"{ds}__*"):
            if not p.is_dir():
                continue
            nodes = list(p.glob(f"nodes__{p.name}.csv"))
            if nodes:
                tags.append(p.name)
    base_csv = reports_dir / ds / f"nodes__{ds}.csv"
    if not tags and base_csv.exists():
        tags.append(ds)
    return sorted(tags)

def read_nodes_csv(reports_dir: Path, dataset_tag: str) -> pd.DataFrame:
    csv_path = reports_dir / dataset_tag / f"nodes__{dataset_tag}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing nodes CSV: {csv_path}")
    df = pd.read_csv(csv_path)
    for col in ["m_node","R_m","m_angleonly","R_m_holdout","theta12_deg","r0","r1","r2",
                "e0x","e0y","e1x","e1y","e2x","e2y",
                "svd_ratio_e0","svd_ratio_e1","svd_ratio_e2",
                "cfg_m_min","cfg_m_max"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in ["qc_pass","qc_pass_strict","parent_ambiguous"]:
        if col in df.columns:
            df[col] = df[col].astype(bool)
    if "cfg_m_min" not in df.columns: df["cfg_m_min"] = 0.2
    if "cfg_m_max" not in df.columns: df["cfg_m_max"] = 4.0
    return df

def closure_residual_batch(r0, r1, r2, e0, e1, e2, m):
    r0 = np.asarray(r0, float); r1 = np.asarray(r1, float); r2 = np.asarray(r2, float)
    e0 = np.asarray(e0, float); e1 = np.asarray(e1, float); e2 = np.asarray(e2, float)
    m  = np.asarray(m,  float)
    a0 = (np.power(r0, m))[:, None] * e0
    a1 = (np.power(r1, m))[:, None] * e1
    a2 = (np.power(r2, m))[:, None] * e2
    num = np.linalg.norm(a0 + a1 + a2, axis=1)
    den = (np.power(r1, m)) + (np.power(r2, m)) + 1e-12
    return num / den

def parent_direction_error_phi0(r0, r1, r2, e0, e1, e2, m) -> np.ndarray:
    r0 = np.asarray(r0, float); r1 = np.asarray(r1, float); r2 = np.asarray(r2, float)
    e0 = np.asarray(e0, float); e1 = np.asarray(e1, float); e2 = np.asarray(e2, float)
    m  = np.asarray(m,  float)
    a  = (np.power(r1, m))[:, None] * e1 + (np.power(r2, m))[:, None] * e2
    v  = -a
    n  = np.linalg.norm(v, axis=1, keepdims=True) + 1e-12
    u  = v / n
    dots = np.clip(np.sum(u * e0, axis=1), -1.0, 1.0)
    return np.degrees(np.arccos(dots))

def symmetric_m_from_theta(theta_deg: np.ndarray, n: float = 4.0) -> np.ndarray:
    t = np.radians(theta_deg * 0.5)
    c = np.cos(t)
    x = np.log2(np.clip(c, 1e-12, 1.0 - 1e-12))
    denom = (1.0 - x)
    with np.errstate(divide="ignore", invalid="ignore"):
        m = n * (1.0 + x) / denom
    m[(c <= 0.0) | (c >= 1.0)] = np.nan
    return m

def set_pub_style():
    matplotlib.rcParams.update(PUB_RCPARAMS)

def panel_A_m_hist(df: pd.DataFrame, out_dir: Path, dataset_tag: str,
                   figsize_key: str, exts: Sequence[str]):
    fig, ax = new_fig(figsize_key)
    m_vals = df["m_node"].astype(float).to_numpy()
    m_vals = m_vals[np.isfinite(m_vals)]
    bins = min(60, max(30, int(np.ceil(np.sqrt(max(1, m_vals.size))))))
    ax.hist(m_vals, bins=bins, alpha=0.9)
    m_min = float(df["cfg_m_min"].iloc[0]); m_max = float(df["cfg_m_max"].iloc[0])
    ax.axvline(m_min, linestyle="--", linewidth=1.0)
    ax.axvline(m_max, linestyle="--", linewidth=1.0)
    at_edge = ((np.abs(df["m_node"] - m_min) <= 0.02) | (np.abs(df["m_node"] - m_max) <= 0.02))
    ax.set_xlabel("EPIC upkeep exponent m (per node)")
    ax.set_ylabel("count")
    ax.set_title(f"Panel A — m distribution — {dataset_tag}  (N={m_vals.size})")
    note = f"at-edge (±0.02): {int(at_edge.sum())} / {len(df)}"
    ax.text(0.02, 0.98, note, transform=ax.transAxes, va="top", ha="left",
            bbox=dict(facecolor="white", alpha=0.90, edgecolor="none", boxstyle="round,pad=0.25"))
    save_figure(fig, out_dir / f"panelA__m_hist__{dataset_tag}", exts)

def panel_B_residual_hist(df: pd.DataFrame, out_dir: Path, dataset_tag: str,
                          primary_metric: str, figsize_key: str, exts: Sequence[str],
                          boot: int = 5000):
    R_col = "R_m_holdout" if primary_metric == "heldout" else "R_m"
    if R_col not in df.columns:
        R_col = "R_m"
    R = df[R_col].astype(float).to_numpy()
    R = R[np.isfinite(R)]
    fig, ax = new_fig(figsize_key)
    bins = min(80, max(30, int(np.ceil(np.sqrt(max(1, R.size))))))
    ax.hist(R, bins=bins, density=False, alpha=0.9, label=f"observed (N={R.size})")
    ax.axvspan(0.0, 0.55, alpha=0.10)
    ax.axvspan(0.55, 0.85, alpha=0.07)
    ax.axvline(0.55, linestyle="--", linewidth=1.0)
    ax.axvline(0.85, linestyle="--", linewidth=1.0)
    med = float(np.median(R)) if R.size else float("nan")
    lo, hi = bootstrap_ci_median(R, B=int(boot), seed=13)
    ax.axvline(med, linewidth=2.0, label=f"median={med:.3f}")
    ax.axvline(lo, linestyle=":", linewidth=1.2)
    ax.axvline(hi, linestyle=":", linewidth=1.2)
    ax.annotate(f"median = {med:.3f}\n95% CI [{lo:.3f}, {hi:.3f}]",
                xy=(0.015, 0.98), xycoords="axes fraction", ha="left", va="top",
                bbox=dict(facecolor="white", alpha=0.90, edgecolor="none", boxstyle="round,pad=0.25"))
    ax.set_xlabel("closure residual R(m)")
    ax.set_ylabel("count")
    ax.set_title(f"Panel B — residuals ({primary_metric}) — {dataset_tag}")
    ax.legend(loc="upper right")
    try:
        xmax = float(np.nanpercentile(R, 99.5))
        ax.set_xlim(0.0, min(2.0, xmax))
    except Exception:
        pass
    save_figure(fig, out_dir / f"panelB__residual_hist__{primary_metric}__{dataset_tag}", exts)
    return med, lo, hi

def figure_baselines(df: pd.DataFrame, out_dir: Path, dataset_tag: str,
                     figsize_key: str, exts: Sequence[str]):
    have_dirs = all(c in df.columns for c in ["e0x","e0y","e1x","e1y","e2x","e2y"])
    r0 = df["r0"].to_numpy(float); r1 = df["r1"].to_numpy(float); r2 = df["r2"].to_numpy(float)
    e0 = df[["e0x","e0y"]].to_numpy(float) if have_dirs else None
    e1 = df[["e1x","e1y"]].to_numpy(float) if have_dirs else None
    e2 = df[["e2x","e2y"]].to_numpy(float) if have_dirs else None
    t12= np.deg2rad(df["theta12_deg"].to_numpy(float)) if "theta12_deg" in df.columns else None

    def closure_resid(m):
        if have_dirs:
            return closure_residual_batch(r0, r1, r2, e0, e1, e2, m)
        else:
            c = np.cos(t12)
            val = (r0**(2*m)) - (r1**(2*m)) - (r2**(2*m)) - 2*c*(r1**m)*(r2**m)
            scale = (r0**(2*m)) + (r1**(2*m)) + (r2**(2*m)) + 1e-12
            return np.abs(val)/scale

    R_epic = df.get("R_m", pd.Series(np.nan, index=df.index)).to_numpy(float)
    grid_ms = np.linspace(0.2, 4.0, 381)
    med_grid = np.array([np.median(closure_resid(float(mm))) for mm in grid_ms])
    m_star = float(grid_ms[int(np.argmin(med_grid))])

    fig, ax = new_fig(figsize_key)
    if np.all(np.isfinite(R_epic)):
        ax.hist(R_epic, bins=40, alpha=0.85, label="EPIC (per-node m)")
    for m_fixed in (2.0, 1.0, m_star):
        Rb = closure_resid(m_fixed)
        label = f"m={m_fixed:g}" if m_fixed in (1.0, 2.0) else f"m*={m_star:.2f}"
        ax.hist(Rb, bins=40, alpha=0.35, histtype="stepfilled", label=label)
    ax.set_xlabel("closure residual R(m)" if have_dirs else "normalized equation misfit")
    ax.set_ylabel("count")
    ax.set_title(f"Baselines vs EPIC — {dataset_tag}")
    ax.legend(loc="best")
    save_figure(fig, out_dir / f"baselines__hist__{dataset_tag}", exts)

    if np.all(np.isfinite(R_epic)):
        fig2, ax2 = new_fig(figsize_key)
        for m_fixed in (2.0, 1.0, m_star):
            Rb = closure_resid(m_fixed)
            dR = (Rb - R_epic)
            dR = dR[np.isfinite(dR)]
            x = np.sort(dR)
            y = np.arange(1, x.size+1) / x.size if x.size else np.array([])
            label = f"m={m_fixed:g}" if m_fixed in (1.0, 2.0) else f"m*={m_star:.2f}"
            if x.size:
                ax2.step(x, y, where="post", label=label)
        ax2.axvline(0.0, linewidth=1.0)
        ax2.set_xlabel("ΔR = R(baseline) − R(EPIC)")
        ax2.set_ylabel("ECDF")
        ax2.set_title(f"Paired improvement ΔR — {dataset_tag}")
        ax2.legend(loc="lower right")
        save_figure(fig2, out_dir / f"baselines__paired_ecdf__{dataset_tag}", exts)

def panel_C_theta_vs_m(df: pd.DataFrame, out_dir: Path, dataset_tag: str,
                       figsize_key: str, exts: Sequence[str],
                       symmetric_tol: float = 1.08):
    if not {"theta12_deg","m_node","r1","r2"}.issubset(df.columns):
        return
    rmin = np.minimum(df["r1"].to_numpy(float), df["r2"].to_numpy(float))
    rmax = np.maximum(df["r1"].to_numpy(float), df["r2"].to_numpy(float))
    ratio = rmax / np.maximum(rmin, 1e-9)

    def sub_for_tol(tol):
        mask = ratio <= float(tol)
        return df.loc[mask].copy()

    tol = float(symmetric_tol)
    sub = sub_for_tol(tol)
    while (len(sub) < 30) and (tol + 0.02 <= 1.25):
        tol = min(1.25, tol + 0.02)
        sub = sub_for_tol(tol)

    fig, ax = new_fig(figsize_key)
    x_lo, x_hi = 20.0, 170.0
    theta_grid = np.linspace(x_lo, x_hi, 900)
    m_sym_curve = symmetric_m_from_theta(theta_grid, n=4.0)
    ax.plot(theta_grid, m_sym_curve, linewidth=2.0, label="analytic symmetric (n=4)")
    band = 0.25
    ax.fill_between(theta_grid, m_sym_curve - band, m_sym_curve + band, alpha=0.12, label="|Δm| ≤ 0.25")

    ok = np.isfinite(sub["theta12_deg"]) & np.isfinite(sub["m_node"])
    th = sub.loc[ok, "theta12_deg"].to_numpy(float)
    mm = sub.loc[ok, "m_node"].to_numpy(float)
    qc_strict = sub.loc[ok, "qc_pass_strict"].astype(bool).to_numpy() if "qc_pass_strict" in sub.columns else np.zeros_like(mm, dtype=bool)
    ax.scatter(th[qc_strict], mm[qc_strict], s=20, marker="o", label="near-symmetric (strict PASS)")
    ax.scatter(th[~qc_strict], mm[~qc_strict], s=22, marker="x", label="near-symmetric (strict FAIL)")

    if th.size > 0:
        edges = np.arange(x_lo, x_hi+10, 10.0)
        idx = np.digitize(th, edges) - 1
        centers, med, q25, q75 = [], [], [], []
        for b in range(edges.size - 1):
            msk = idx == b
            if np.count_nonzero(msk) >= 4:
                centers.append(0.5*(edges[b]+edges[b+1]))
                vals = mm[msk]
                med.append(np.median(vals)); q25.append(np.percentile(vals, 25)); q75.append(np.percentile(vals, 75))
        if centers:
            centers = np.array(centers); med = np.array(med); q25 = np.array(q25); q75 = np.array(q75)
            yerr = np.vstack([med - q25, q75 - med])
            ax.errorbar(centers, med, yerr=yerr, fmt="-", linewidth=1.5, capsize=2, label="bin median ± IQR")

    ax.set_xlim(x_lo, x_hi); ax.set_ylim(0.2, 4.0)
    ax.set_xlabel(r"$\theta_{12}$ (degrees)")
    ax.set_ylabel(r"$m$ (per-node inverted)")
    ax.set_title(f"Panel C — θ–m near-symmetric (r1/r2 ≤ {tol:.2f}) — {dataset_tag}")
    ax.legend(loc="best")
    save_figure(fig, out_dir / f"panelC__theta_vs_m__{dataset_tag}", exts)

def figure_parent_direction(df: pd.DataFrame, out_dir: Path, dataset_tag: str,
                            figsize_key: str, exts: Sequence[str]):
    have_dirs = all(c in df.columns for c in ["e0x","e0y","e1x","e1y","e2x","e2y"])
    if not have_dirs:
        return
    e0 = df[["e0x","e0y"]].to_numpy(float)
    e1 = df[["e1x","e1y"]].to_numpy(float)
    e2 = df[["e2x","e2y"]].to_numpy(float)
    r0 = df["r0"].to_numpy(float); r1 = df["r1"].to_numpy(float); r2 = df["r2"].to_numpy(float)
    m_epic = df.get("m_node", pd.Series(np.nan, index=df.index)).to_numpy(float)

    def phi(m): return parent_direction_error_phi0(r0, r1, r2, e0, e1, e2, m)

    fig, ax = new_fig(figsize_key)
    for name, arr in [("EPIC", phi(m_epic)), ("m=2", phi(2.0)), ("m=1", phi(1.0)), ("120°", phi(0.0))]:
        x = np.sort(arr[np.isfinite(arr)])
        y = np.arange(1, x.size+1) / x.size if x.size else np.array([])
        if x.size:
            ax.step(x, y, where="post", label=name)
    ax.set_xlabel("parent-direction error φ₀ (degrees)")
    ax.set_ylabel("ECDF")
    ax.set_title(f"Parent-direction prediction — {dataset_tag}")
    ax.legend(loc="lower right")
    save_figure(fig, out_dir / f"parentdir__phi0_ecdf__{dataset_tag}", exts)

    fig2, ax2 = new_fig(figsize_key)
    phi_e = phi(m_epic)
    deltas = [("m=2 − EPIC", phi(2.0) - phi_e), ("m=1 − EPIC", phi(1.0) - phi_e), ("120° − EPIC", phi(0.0) - phi_e)]
    data = [d[1][np.isfinite(d[1])] for d in deltas]
    if any(len(x) for x in data):
        ax2.boxplot(data, labels=[d[0] for d in deltas])
        ax2.axhline(0.0, linewidth=1.0)
        ax2.set_ylabel("Δφ (degrees)")
        ax2.set_title(f"Parent-direction Δφ — {dataset_tag}")
        save_figure(fig2, out_dir / f"parentdir__delta_phi_box__{dataset_tag}", exts)
    else:
        plt.close(fig2)

def segrobust_scoreboard(datasets: Sequence[str], reports_dir: Path, out_dir: Path,
                         figsize_key: str, exts: Sequence[str]):
    variants = ["frangi+otsu", "frangi+quantile", "sato+otsu", "sato+quantile"]
    rows = []
    for ds in datasets:
        tsv = reports_dir / ds / "SEGROBUST__summary.tsv"
        if not tsv.exists():
            rows.append(None)
            continue
        df = pd.read_csv(tsv, sep="\t")
        rows.append(df)

    fig = plt.figure(figsize=(FIGSIZE_PRESETS["double"][0], FIGSIZE_PRESETS["double"][1] * max(1, len(datasets)/2)))
    axes = fig.subplots(len(datasets), len(variants), squeeze=False, sharex=True, sharey=True)

    for i, ds in enumerate(datasets):
        df = rows[i]
        for j, var in enumerate(variants):
            ax = axes[i, j]
            if df is None or df.empty or var not in df["variant"].astype(str).values:
                ax.axis("off")
                continue
            row = df[df["variant"] == var].iloc[0]
            r_med = float(row["R_med"]); ci_lo = float(row["R_med_CI_lo"]); ci_hi = float(row["R_med_CI_hi"])
            N = int(row["N_nodes"])
            ax.plot([0.5, 0.5], [ci_lo, ci_hi], color="black", linewidth=1.5)
            ax.plot([0.5], [r_med], "o", color="black", markersize=4)
            ax.axhline(0.55, linestyle="--", linewidth=0.8)
            ax.axhline(0.85, linestyle="--", linewidth=0.8)
            ax.set_xlim(0.0, 1.0); ax.set_ylim(0.0, 1.5); ax.set_xticks([])
            ax.grid(axis="y", alpha=0.15, linestyle="-", linewidth=0.7)
            ax.text(0.02, 0.98, f"N={N}\nmedian={r_med:.3f}\n95%[{ci_lo:.3f},{ci_hi:.3f}]",
                    transform=ax.transAxes, va="top", ha="left",
                    bbox=dict(facecolor="white", alpha=0.9, edgecolor="none"), fontsize=8)
            if i == 0:
                ax.set_title(var.replace("+", " + "), fontsize=9)
        axes[i, 0].set_ylabel(ds, rotation=0, labelpad=28, fontsize=10, va="center")

    fig.suptitle("Held-out closure: transportability & segmentation robustness", fontsize=12)
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.95])
    save_figure(fig, out_dir / "ALL_DATASETS__segrobust_scoreboard", exts)

def terminal_and_markdown_summary(
    df: pd.DataFrame,
    reports_dir: Path,
    out_dir_pub: Path,
    dataset_tag: str,
    primary_metric: str,
    med_residual: float,
    med_lo: float,
    med_hi: float,
    exts: Sequence[str]
):
    ds_base = dataset_tag.split("__", 1)[0]
    def first_of(*paths) -> Optional[str]:
        for p in paths:
            if p.exists():
                return p.read_text()
        return None
    prereg_base = reports_dir / ds_base / "PREREG__C1.txt"
    prereg_var  = reports_dir / dataset_tag / "PREREG__C1.txt"
    ablation_base = reports_dir / ds_base / "ABLATION__test.txt"
    panelC_base   = reports_dir / ds_base / "PANELC__theta_vs_m_test.txt"
    prereg_txt = first_of(prereg_var, prereg_base)
    abl_txt    = first_of(ablation_base)
    pan_txt    = first_of(panelC_base)

    def grab(regex: str, text: Optional[str]) -> Optional[str]:
        if not text: return None
        m = re.search(regex, text)
        return m.group(1) if m else None

    prereg_outcome = grab(r"Outcome:\s*(PASS|FAIL)", prereg_txt)
    prereg_worst_p_med = grab(r"Worst.*p_med=([0-9eE\.\-+]+)", prereg_txt)
    prereg_worst_p_frac= grab(r"Worst.*p_frac=([0-9eE\.\-+]+)", prereg_txt)
    prereg_max_delta   = grab(r"max δ=([0-9eE\.\-+]+)", prereg_txt)

    abl_outcome = grab(r"Outcome:\s*(PASS|FAIL)", abl_txt)
    pan_verdict = grab(r"Panel C verdict:\s*(PASS|FAIL)", pan_txt)
    pan_share   = grab(r"% within \|Δm\|≤0\.25\s*=\s*([0-9\.]+)", pan_txt)

    n = len(df)
    strict = float(df["qc_pass_strict"].mean()) if "qc_pass_strict" in df.columns and n else float("nan")
    loose  = float(df["qc_pass"].mean())        if "qc_pass" in df.columns and n else float("nan")
    m_med = float(np.median(df["m_node"])) if n else float("nan")
    m_iqr = (float(np.percentile(df["m_node"], 25)), float(np.percentile(df["m_node"], 75))) if n else (float("nan"), float("nan"))
    m_min = float(df["cfg_m_min"].iloc[0]); m_max = float(df["cfg_m_max"].iloc[0])
    at_edge = ((np.abs(df["m_node"] - m_min) <= 0.02) | (np.abs(df["m_node"] - m_max) <= 0.02)).sum()
    parent_ambig = float(df["parent_ambiguous"].mean()) if "parent_ambiguous" in df.columns and n else float("nan")

    lines = []
    lines.append(f"===== SUMMARY — {dataset_tag} =====")
    lines.append(f"N nodes: {n}")
    lines.append(f"QC strict: {strict:.1%}   QC loose: {loose:.1%}")
    lines.append(f"m median (IQR): {m_med:.3f}  ({m_iqr[0]:.3f},{m_iqr[1]:.3f})")
    lines.append(f"Residual ({primary_metric}) median: {med_residual:.3f}  [95% CI {med_lo:.3f}, {med_hi:.3f}]")
    lines.append(f"m at bracket edge (±0.02): {at_edge} / {n}")
    if np.isfinite(parent_ambig): lines.append(f"Parent ambiguity: {parent_ambig:.1%}")
    if prereg_outcome:
        lines.append(f"PREREG (heldout): {prereg_outcome} | worst p_med={prereg_worst_p_med}, p_frac={prereg_worst_p_frac}, max δ={prereg_max_delta}")
    if abl_outcome:
        lines.append(f"Ablation: {abl_outcome}")
    if pan_verdict:
        lines.append(f"Panel C: {pan_verdict}" + (f" | share≤0.25={float(pan_share)/100.0:.1%}" if pan_share else ""))
    print("\n".join(lines))

    md = out_dir_pub / f"PAPER_SUMMARY__{dataset_tag}.md"
    md_lines = [
        f"# EPIC Paper Summary — `{dataset_tag}`",
        "",
        f"- **Nodes**: {n}",
        f"- **QC (strict / loose)**: {strict:.1%} / {loose:.1%}",
        f"- **m median (IQR)**: {m_med:.3f} ({m_iqr[0]:.3f}, {m_iqr[1]:.3f})",
        f"- **Residual ({primary_metric}) median**: {med_residual:.3f}  95% CI [{med_lo:.3f}, {med_hi:.3f}]",
        f"- **m at bracket edge (±0.02)**: {at_edge} / {n}",
        f"- **Parent ambiguity**: {(f'{parent_ambig:.1%}' if np.isfinite(parent_ambig) else 'n/a')}",
        "",
        "## Tests & Verdicts",
        f"- **PREREG (heldout)**: {prereg_outcome or 'n/a'}"
        + (f"  | worst p_med={prereg_worst_p_med}, p_frac={prereg_worst_p_frac}, max δ={prereg_max_delta}" if prereg_outcome else ""),
        f"- **Ablation**: {abl_outcome or 'n/a'}",
        f"- **Panel C**: n/a",
        "",
    ]
    md.write_text("\n".join(md_lines), encoding="utf-8")

# ----------------------- CLI -----------------------

def parse_args_full() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="EPIC plots & summaries")
    p.add_argument("--data-root", type=str, default=str(ROOT_DEFAULT),
                   help="Root containing reports/ and figures/ (default: %(default)s)")
    p.add_argument("--datasets", nargs="+", required=True,
                   help="Datasets (e.g., HRF DRIVE STARE CHASE_DB1)")
    p.add_argument("--variants", nargs="*", default=None,
                   help="Restrict to segmentation variants (e.g., frangi+otsu sato+quantile).")
    p.add_argument("--primary-metric", choices=["heldout","asconfigured"], default="heldout",
                   help="Residual to use for Panel B & median CI (default: %(default)s)")
    p.add_argument("--fig-size", choices=list(FIGSIZE_PRESETS.keys()), default="single",
                   help="Figure size preset")
    p.add_argument("--ext", nargs="+", default=["pdf", "png"],
                   help="Figure export extensions")
    p.add_argument("--boot", type=int, default=5000, help="Bootstrap reps for median CI")
    p.add_argument("--scoreboard", action="store_true",
                   help="Also generate cross-dataset segmentation-robust scoreboard if summary TSVs exist")
    return p.parse_args()

def process_variant(reports_dir: Path, figures_pub_root: Path, dataset_tag: str,
                    fig_size_key: str, exts: Sequence[str], primary_metric: str, boot: int):
    set_pub_style()
    t0 = time.time()
    out_pub = ensure_dir(figures_pub_root / dataset_tag)
    df = read_nodes_csv(reports_dir, dataset_tag)
    # Panels / Figures
    med, lo, hi = 0.0, 0.0, 0.0
    panel_A_m_hist(df, out_pub, dataset_tag, fig_size_key, exts)
    med, lo, hi = panel_B_residual_hist(df, out_pub, dataset_tag, primary_metric, fig_size_key, exts, boot=boot)
    figure_baselines(df, out_pub, dataset_tag, fig_size_key, exts)
    panel_C_theta_vs_m(df, out_pub, dataset_tag, fig_size_key, exts)
    figure_parent_direction(df, out_pub, dataset_tag, fig_size_key, exts)
    # Terminal+MD summary
    terminal_and_markdown_summary(df, reports_dir, out_pub, dataset_tag, primary_metric, med, lo, hi, exts)
    print(f"[{dataset_tag}] figures → {out_pub}  | elapsed {human_time(time.time()-t0)}")

def main():
    # QUICK MODE: one positional arg (tag) or two (tag, data_root)
    if len(sys.argv) >= 2 and not sys.argv[1].startswith("-"):
        tag_arg = sys.argv[1]
        data_root = sys.argv[2] if len(sys.argv) >= 3 and not sys.argv[2].startswith("-") else None
        sys.exit(quick_run_single(tag_arg, data_root))

    # Otherwise, fall back to the full plotting pipeline (optional)
    args = parse_args_full()
    root = Path(args.data_root).expanduser()
    reports_dir = root / REPORTS
    if not reports_dir.exists():
        print(f"[ERROR] reports/ not found under {root}", file=sys.stderr)
        sys.exit(2)
    figures_pub_root = ensure_dir(root / FIGURES_PUB)

    overall_start = time.time()
    all_tags: List[str] = []
    for ds in args.datasets:
        tags = discover_variants(reports_dir, ds, args.variants)
        if not tags:
            print(f"[WARN] No variants found for dataset '{ds}' in {reports_dir}")
            continue
        for tag in tags:
            try:
                process_variant(reports_dir, figures_pub_root, tag, args.fig_size, args.ext, args.primary_metric, args.boot)
                all_tags.append(tag)
            except Exception as e:
                print(f"[ERROR] {tag}: {e}", file=sys.stderr)

    if args.scoreboard:
        try:
            out_dir = ensure_dir(figures_pub_root)
            segrobust_scoreboard(args.datasets, reports_dir, out_dir, args.fig_size, args.ext)
            print(f"[SCOREBOARD] → {out_dir/'ALL_DATASETS__segrobust_scoreboard'}.[{','.join(args.ext)}]")
        except Exception as e:
            print(f"[SCOREBOARD][WARN] failed: {e}", file=sys.stderr)

    print(f"ALL DONE in {human_time(time.time()-overall_start)}")

if __name__ == "__main__":
    main()
