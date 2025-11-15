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

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
import argparse
import math
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


# ---------------------------- Utilities ----------------------------

def read_csv_relaxed(path: Path) -> pd.DataFrame:
    """Tolerant CSV reader: skips a leading commented line if present."""
    txt = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    from io import StringIO
    if txt and txt[0].lstrip().startswith("#"):
        return pd.read_csv(StringIO("\n".join(txt[1:])))
    return pd.read_csv(StringIO("\n".join(txt)))


def maybe_read_text(*paths: Path) -> Optional[str]:
    """Return first readable file's text among the provided paths."""
    for p in paths:
        if p and p.exists():
            try:
                return p.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue
    return None


def bootstrap_ci_median(x: np.ndarray, B: int = 5000, seed: int = 13) -> Tuple[float, float]:
    """Bootstrap 95% CI for the median."""
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(int(seed))
    meds = np.median(rng.choice(x, size=(int(B), x.size), replace=True), axis=1)
    lo, hi = np.percentile(meds, [2.5, 97.5])
    return float(lo), float(hi)


def bootstrap_ci_delta_median(x: np.ndarray, y: np.ndarray, B: int = 5000, seed: int = 13) -> Tuple[float, float]:
    """Bootstrap 95% CI for Δ median = median(x) − median(y) with independent resampling."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    x = x[np.isfinite(x)];     y = y[np.isfinite(y)]
    if x.size == 0 or y.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(int(seed))
    meds_x = np.median(rng.choice(x, size=(int(B), x.size), replace=True), axis=1)
    meds_y = np.median(rng.choice(y, size=(int(B), y.size), replace=True), axis=1)
    deltas = meds_x - meds_y
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    return float(lo), float(hi)


def fmt_pct(x: Optional[float]) -> str:
    return f"{100.0 * float(x):.1f}%" if x is not None and np.isfinite(x) else "n/a"


def fmt_num(x: Optional[float], nd: int = 3) -> str:
    return f"{float(x):.{nd}f}" if x is not None and np.isfinite(x) else "n/a"


def human_time(seconds: float) -> str:
    if seconds < 60: return f"{seconds:.1f}s"
    m, s = divmod(seconds, 60)
    if m < 60: return f"{int(m)}m {int(s)}s"
    h, m = divmod(m, 60); return f"{int(h)}h {int(m)}m"


# --------------------- Discovery & Data Reading --------------------

def discover_dataset_variants(reports_dir: Path,
                              datasets: Optional[Sequence[str]] = None,
                              variants: Optional[Sequence[str]] = None) -> Dict[str, List[str]]:
    """
    Map datasets to a sorted list of discovered variants under reports/.
    Expects folders named: <DATASET>__<seg>+<thresh>
    """
    mapping: Dict[str, List[str]] = {}
    for p in reports_dir.iterdir():
        if not p.is_dir(): continue
        name = p.name
        if "__" in name and "+" in name:
            ds, var = name.split("__", 1)
            if datasets and ds not in datasets: continue
            if variants and var not in variants: continue
            mapping.setdefault(ds, []).append(var)
    for ds in mapping:
        mapping[ds] = sorted(mapping[ds])
    if datasets:
        for ds in datasets:
            mapping.setdefault(ds, [])
    return dict(sorted(mapping.items()))


def read_nodes_csv(reports_dir: Path, dataset: str, variant: str) -> pd.DataFrame:
    tag = f"{dataset}__{variant}"
    csv_path = reports_dir / tag / f"nodes__{tag}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing CSV: {csv_path}")
    df = read_csv_relaxed(csv_path)
    # Coerce types for known columns
    for col in ["R_m_holdout","R_m","m_node","theta12_deg","r1","r2",
                "cfg_m_min","cfg_m_max","parent_ambiguous"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in ["qc_pass","qc_pass_strict","parent_ambiguous"]:
        if col in df.columns:
            df[col] = df[col].astype(bool)
    return df


# --------------------------- Parsers -------------------------------

def parse_prereg_c1(txt: Optional[str]) -> Dict[str, Optional[str]]:
    d = dict(outcome=None, R_obs=None, R_ci=None, worst_p_med=None, worst_p_frac=None, max_delta=None)
    if not txt: return d
    m_out = re.search(r"Outcome:\s*(PASS|FAIL)", txt); d["outcome"] = m_out.group(1) if m_out else None
    m_R = re.search(r"Observed\s*[~\u007E]?\s*R\s*=\s*([0-9\.\-eE]+).*?\[boot\s*95% CI\s*([0-9\.\-eE]+),\s*([0-9\.\-eE]+)\]", txt)
    if m_R:
        d["R_obs"] = f"{float(m_R.group(1)):.6f}"
        d["R_ci"]  = f"[{float(m_R.group(2)):.6f}, {float(m_R.group(3)):.6f}]"
    m_wpmed = re.search(r"p_med(?:ian)?=([0-9eE\.\-+]+)", txt)
    m_wpfr  = re.search(r"p_frac[^\=]*=\s*([0-9eE\.\-+]+)", txt)
    m_wdel  = re.search(r"Cliff['’]s?\s*δ\s*=\s*([0-9eE\.\-+]+)|max\s*δ\s*=\s*([0-9eE\.\-+]+)", txt)
    d["worst_p_med"]  = m_wpmed.group(1) if m_wpmed else None
    d["worst_p_frac"] = m_wpfr.group(1) if m_wpfr else None
    d["max_delta"]    = (m_wdel.group(1) or m_wdel.group(2)) if m_wdel else None
    return d


def parse_simple_outcome(txt: Optional[str]) -> Optional[str]:
    if not txt: return None
    m = re.search(r"Outcome:\s*(PASS|FAIL)", txt)
    return m.group(1) if m else None


def parse_baselines(txt: Optional[str]) -> Dict[str, Optional[float]]:
    """
    Look for: 'm=2: median=0.xxx', 'm=1: median=0.xxx'
    """
    out = dict(m2=None, m1=None)
    if not txt: return out
    m2 = re.search(r"m=2:\s*median=([0-9\.\-eE]+)", txt)
    m1 = re.search(r"m=1:\s*median=([0-9\.\-eE]+)", txt)
    if m2: out["m2"] = float(m2.group(1))
    if m1: out["m1"] = float(m1.group(1))
    return out


# --------------------------- Core stats ----------------------------

@dataclass
class VariantRow:
    dataset: str
    variant: str
    R_col: str
    N: int
    R_med: float
    R_lo: float
    R_hi: float
    R_p25: float
    R_p75: float
    R_p95: float
    R_share_055: float
    R_share_085: float
    qc_strict: Optional[float]
    qc_loose: Optional[float]
    m_med: Optional[float]
    m_q25: Optional[float]
    m_q75: Optional[float]
    m_at_edge_count: Optional[int]
    m_at_edge_share: Optional[float]
    parent_ambig_share: Optional[float]
    theta_med: Optional[float]
    theta_q25: Optional[float]
    theta_q75: Optional[float]
    R_med_asconf: Optional[float]
    R_lo_asconf: Optional[float]
    R_hi_asconf: Optional[float]
    R_med_heldout: Optional[float]
    R_lo_heldout: Optional[float]
    R_hi_heldout: Optional[float]
    ablation_outcome: Optional[str]
    prereg_outcome: Optional[str]
    prereg_R_obs: Optional[str]
    prereg_R_ci: Optional[str]
    prereg_worst_p_med: Optional[str]
    prereg_worst_p_frac: Optional[str]
    prereg_max_delta: Optional[str]
    baseline_m1: Optional[float]
    baseline_m2: Optional[float]
    R_values: np.ndarray  # keep for pairwise Δ bootstraps


def compute_row(df: pd.DataFrame,
                reports_dir: Path,
                dataset: str,
                variant: str,
                prefer_metric: str,
                boot: int) -> VariantRow:
    # Choose metric column
    has_holdout = "R_m_holdout" in df.columns
    if prefer_metric == "asconfigured":
        R_col = "R_m"
    elif prefer_metric == "heldout":
        R_col = "R_m_holdout" if has_holdout else "R_m"
    else:  # auto
        R_col = "R_m_holdout" if has_holdout else "R_m"

    R = pd.to_numeric(df.get(R_col, pd.Series([], dtype=float)), errors="coerce").to_numpy()
    R = R[np.isfinite(R)]
    N = int(R.size)
    if N:
        R_med = float(np.median(R)); R_lo, R_hi = bootstrap_ci_median(R, B=int(boot), seed=13)
        R_p25 = float(np.percentile(R, 25)); R_p75 = float(np.percentile(R, 75)); R_p95 = float(np.percentile(R, 95))
        R_share_055 = float((R <= 0.55).mean()); R_share_085 = float((R <= 0.85).mean())
    else:
        R_med = R_lo = R_hi = R_p25 = R_p75 = R_p95 = R_share_055 = R_share_085 = float("nan")

    # QC
    qc_strict = float(df["qc_pass_strict"].mean()) if "qc_pass_strict" in df.columns and len(df) else None
    qc_loose  = float(df["qc_pass"].mean())        if "qc_pass" in df.columns and len(df) else None

    # m stats
    mvals = pd.to_numeric(df.get("m_node", pd.Series([], dtype=float)), errors="coerce")
    m_med = float(np.nanmedian(mvals)) if mvals.size else None
    m_q25 = float(np.nanpercentile(mvals, 25)) if mvals.size else None
    m_q75 = float(np.nanpercentile(mvals, 75)) if mvals.size else None
    mmin  = float(df["cfg_m_min"].iloc[0]) if "cfg_m_min" in df.columns and len(df) else 0.2
    mmax  = float(df["cfg_m_max"].iloc[0]) if "cfg_m_max" in df.columns and len(df) else 4.0
    if mvals.size:
        at_edge = int((((mvals - mmin).abs() <= 0.02) | ((mvals - mmax).abs() <= 0.02)).sum())
        at_edge_share = float(at_edge) / float(len(mvals))
    else:
        at_edge = None; at_edge_share = None

    parent_ambig_share = float(df["parent_ambiguous"].mean()) if "parent_ambiguous" in df.columns and len(df) else None

    # Theta
    thetas = pd.to_numeric(df.get("theta12_deg", pd.Series([], dtype=float)), errors="coerce").to_numpy()
    thetas = thetas[np.isfinite(thetas)]
    theta_med = float(np.median(thetas)) if thetas.size else None
    theta_q25 = float(np.percentile(thetas, 25)) if thetas.size else None
    theta_q75 = float(np.percentile(thetas, 75)) if thetas.size else None

    # As-configured vs held-out (if both exist)
    R_med_asconf = R_lo_asconf = R_hi_asconf = None
    R_med_heldout = R_lo_heldout = R_hi_heldout = None
    if "R_m" in df.columns:
        Ra = pd.to_numeric(df["R_m"], errors="coerce").dropna().to_numpy()
        if Ra.size:
            R_med_asconf = float(np.median(Ra)); R_lo_asconf, R_hi_asconf = bootstrap_ci_median(Ra, B=int(boot), seed=13)
    if "R_m_holdout" in df.columns:
        Rh = pd.to_numeric(df["R_m_holdout"], errors="coerce").dropna().to_numpy()
        if Rh.size:
            R_med_heldout = float(np.median(Rh)); R_lo_heldout, R_hi_heldout = bootstrap_ci_median(Rh, B=int(boot), seed=13)

    # Best-effort: parse text artifacts (variant-level preferred, then dataset-level)
    var_dir = reports_dir / f"{dataset}__{variant}"
    ds_dir  = reports_dir / dataset
    prereg_txt   = maybe_read_text(var_dir / "PREREG__C1.txt",   ds_dir / "PREREG__C1.txt")
    ablation_txt = maybe_read_text(var_dir / "ABLATION__test.txt", ds_dir / "ABLATION__test.txt")
    baselines_txt= maybe_read_text(var_dir / "BASELINES__R_m.txt", ds_dir / "BASELINES__R_m.txt")

    prereg = parse_prereg_c1(prereg_txt)
    ablation_outcome = parse_simple_outcome(ablation_txt)
    baselines = parse_baselines(baselines_txt)

    return VariantRow(
        dataset=dataset, variant=variant, R_col=R_col, N=N,
        R_med=R_med, R_lo=R_lo, R_hi=R_hi,
        R_p25=R_p25, R_p75=R_p75, R_p95=R_p95,
        R_share_055=R_share_055, R_share_085=R_share_085,
        qc_strict=qc_strict, qc_loose=qc_loose,
        m_med=m_med, m_q25=m_q25, m_q75=m_q75,
        m_at_edge_count=at_edge, m_at_edge_share=at_edge_share,
        parent_ambig_share=parent_ambig_share,
        theta_med=theta_med, theta_q25=theta_q25, theta_q75=theta_q75,
        R_med_asconf=R_med_asconf, R_lo_asconf=R_lo_asconf, R_hi_asconf=R_hi_asconf,
        R_med_heldout=R_med_heldout, R_lo_heldout=R_lo_heldout, R_hi_heldout=R_hi_heldout,
        ablation_outcome=ablation_outcome,
        prereg_outcome=prereg["outcome"],
        prereg_R_obs=prereg["R_obs"], prereg_R_ci=prereg["R_ci"],
        prereg_worst_p_med=prereg["worst_p_med"],
        prereg_worst_p_frac=prereg["worst_p_frac"],
        prereg_max_delta=prereg["max_delta"],
        baseline_m1=baselines["m1"], baseline_m2=baselines["m2"],
        R_values=R
    )


# ------------------------- Printing helpers -----------------------

def df_print(df: pd.DataFrame, title: str):
    print(title)
    if df.empty:
        print("(no rows)\n")
    else:
        print(df.to_string(index=False))
        print("")


def rows_to_core_df(rows: List[VariantRow],
                    pairwise: bool,
                    boot: int) -> pd.DataFrame:
    # Determine winner (by R_med asc)
    rows_sorted = sorted(rows, key=lambda r: (math.inf if not np.isfinite(r.R_med) else r.R_med, r.variant))
    winner = rows_sorted[0] if rows_sorted else None

    # Build core table
    out = []
    for rank, r in enumerate(rows_sorted, 1):
        delta, d_lo, d_hi, star = (None, None, None, "")
        if winner and np.isfinite(winner.R_med) and np.isfinite(r.R_med):
            delta = r.R_med - winner.R_med
            if pairwise and r is not winner and r.N > 0 and winner.N > 0:
                d_lo, d_hi = bootstrap_ci_delta_median(r.R_values, winner.R_values, B=int(boot), seed=13)
                star = "" if (d_lo <= 0.0 <= d_hi) else "*"
        out.append(dict(
            rank=rank,
            variant=r.variant,
            N=r.N,
            R_med=r.R_med, R_lo=r.R_lo, R_hi=r.R_hi,
            R_IQR=r.R_p75 - r.R_p25 if np.isfinite(r.R_p75) and np.isfinite(r.R_p25) else float("nan"),
            R_p95=r.R_p95,
            share_le_0p55=r.R_share_055, share_le_0p85=r.R_share_085,
            delta_vs_best=delta, delta_CI_lo=d_lo, delta_CI_hi=d_hi, delta_sig=star
        ))
    return pd.DataFrame(out)


def rows_to_qc_df(rows: List[VariantRow]) -> pd.DataFrame:
    out = []
    for r in sorted(rows, key=lambda x: x.variant):
        out.append(dict(
            variant=r.variant,
            QC_strict=r.qc_strict, QC_loose=r.qc_loose,
            m_med=r.m_med,
            m_IQR=(r.m_q75 - r.m_q25) if (r.m_q75 is not None and r.m_q25 is not None) else None,
            m_at_edge_count=r.m_at_edge_count,
            m_at_edge_share=r.m_at_edge_share,
            parent_ambig=r.parent_ambig_share,
            theta_med=r.theta_med,
            theta_IQR=(r.theta_q75 - r.theta_q25) if (r.theta_q75 is not None and r.theta_q25 is not None) else None
        ))
    return pd.DataFrame(out)


def rows_to_heldout_vs_asconf_df(rows: List[VariantRow]) -> pd.DataFrame:
    out = []
    for r in sorted(rows, key=lambda x: x.variant):
        out.append(dict(
            variant=r.variant,
            R_med_heldout=r.R_med_heldout, R_CI_heldout_lo=r.R_lo_heldout, R_CI_heldout_hi=r.R_hi_heldout,
            R_med_asconfigured=r.R_med_asconf, R_CI_asconf_lo=r.R_lo_asconf, R_CI_asconf_hi=r.R_hi_asconf
        ))
    return pd.DataFrame(out)


def rows_to_tests_df(rows: List[VariantRow]) -> pd.DataFrame:
    out = []
    for r in sorted(rows, key=lambda x: x.variant):
        out.append(dict(
            variant=r.variant,
            PREREG=r.prereg_outcome,
            PREREG_R=r.prereg_R_obs,
            PREREG_CI=r.prereg_R_ci,
            PREREG_worst_p_med=r.prereg_worst_p_med,
            PREREG_worst_p_frac=r.prereg_worst_p_frac,
            PREREG_max_delta=r.prereg_max_delta,
            Ablation=r.ablation_outcome,
            Baseline_m1=r.baseline_m1,
            Baseline_m2=r.baseline_m2
        ))
    return pd.DataFrame(out)


# ----------------------- Cross-dataset pivots ----------------------

def core_rows_to_pivots(all_rows: List[VariantRow]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return (pivot_med, pivot_N) with datasets as rows and variants as columns."""
    df = pd.DataFrame([dict(dataset=r.dataset, variant=r.variant, R_med=r.R_med, N=r.N) for r in all_rows])
    if df.empty:
        return pd.DataFrame(), pd.DataFrame()
    p_med = df.pivot_table(index="dataset", columns="variant", values="R_med", aggfunc="first")
    p_N   = df.pivot_table(index="dataset", columns="variant", values="N",     aggfunc="first")
    # sort both by columns alphabetically for stability
    p_med = p_med.reindex(sorted(p_med.columns), axis=1)
    p_N   = p_N.reindex(sorted(p_N.columns), axis=1)
    return p_med, p_N


def winners_and_win_counts(all_rows: List[VariantRow]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if not all_rows:
        return pd.DataFrame(), pd.DataFrame()
    winners = []
    by_ds: Dict[str, List[VariantRow]] = {}
    for r in all_rows:
        by_ds.setdefault(r.dataset, []).append(r)
    for ds, rows in by_ds.items():
        rows_sorted = sorted(rows, key=lambda x: (math.inf if not np.isfinite(x.R_med) else x.R_med, x.variant))
        if rows_sorted:
            w = rows_sorted[0]
            winners.append(dict(dataset=ds, winner=w.variant, R_med=w.R_med, N=w.N))
    winners_df = pd.DataFrame(winners).sort_values("dataset") if winners else pd.DataFrame()
    win_counts = winners_df["winner"].value_counts().rename_axis("variant").reset_index(name="wins") if not winners_df.empty else pd.DataFrame()
    return winners_df, win_counts


# ------------------------------- CLI -------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Exhaustive, tabulated comparison of EPIC segmentation variants.")
    p.add_argument("--data-root", type=str, default=".", help="EPIC root containing reports/ (default: '.')")
    p.add_argument("--datasets", nargs="+", default=None, help="Restrict to datasets (e.g., HRF STARE CHASE_DB1)")
    p.add_argument("--variants", nargs="+", default=None, help="Restrict to variants (e.g., frangi+otsu sato+quantile)")
    p.add_argument("--primary-metric", choices=["auto","heldout","asconfigured"], default="auto",
                   help="auto uses R_m_holdout if present, otherwise R_m (default: auto)")
    p.add_argument("--pairwise", action="store_true", help="Also bootstrap Δ median vs winner (95% CI); '*' if CI excludes 0.")
    p.add_argument("--boot", type=int, default=5000, help="Bootstrap reps for medians/Δ (default: 5000)")
    p.add_argument("--save-tsv", action="store_true", help="Write all printed tables as TSV under reports/.")
    return p.parse_args()


# -------------------------------- Main -----------------------------

def main():
    t0 = time.time()
    args = parse_args()
    root = Path(args.data_root).expanduser().resolve()
    reports_dir = root / "reports"
    if not reports_dir.exists():
        print(f"[ERROR] reports/ not found under {root}", file=sys.stderr)
        sys.exit(2)

    mapping = discover_dataset_variants(reports_dir, args.datasets, args.variants)
    if not mapping:
        print("[WARN] No dataset×variant folders discovered in reports/.")
        sys.exit(0)

    all_rows: List[VariantRow] = []

    for ds, vars_for_ds in mapping.items():
        if not vars_for_ds:
            print(f"[WARN] Dataset '{ds}' has no matching variants."); print("")
            continue

        # Compute rows
        ds_rows: List[VariantRow] = []
        for v in vars_for_ds:
            try:
                df = read_nodes_csv(reports_dir, ds, v)
                ds_rows.append(compute_row(df, reports_dir, ds, v, args.primary_metric, args.boot))
            except Exception as e:
                print(f"[WARN] {ds}/{v}: {e}", file=sys.stderr)

        if not ds_rows:
            print(f"=== {ds} ==="); print("(no readable variants)\n")
            continue

        all_rows.extend(ds_rows)

        # Tables for this dataset
        core_df   = rows_to_core_df(ds_rows, pairwise=args.pairwise, boot=args.boot)
        qc_df     = rows_to_qc_df(ds_rows)
        hvac_df   = rows_to_heldout_vs_asconf_df(ds_rows)
        tests_df  = rows_to_tests_df(ds_rows)

        # Pretty print
        df_print(core_df.assign(
            R_med=core_df["R_med"].map(lambda x: fmt_num(x)),
            R_lo=core_df["R_lo"].map(lambda x: fmt_num(x)),
            R_hi=core_df["R_hi"].map(lambda x: fmt_num(x)),
            R_IQR=core_df["R_IQR"].map(lambda x: fmt_num(x)),
            R_p95=core_df["R_p95"].map(lambda x: fmt_num(x)),
            share_le_0p55=core_df["share_le_0p55"].map(fmt_pct),
            share_le_0p85=core_df["share_le_0p85"].map(fmt_pct),
            delta_vs_best=core_df["delta_vs_best"].map(lambda x: "n/a" if x is None or not np.isfinite(x) else f"{x:+.3f}"),
            delta_CI=core_df.apply(lambda r:
                                   ("[" + (fmt_num(r["delta_CI_lo"]) if r["delta_CI_lo"] is not None else "n/a")
                                    + ", " + (fmt_num(r["delta_CI_hi"]) if r["delta_CI_hi"] is not None else "n/a") + "]"
                                    + (r["delta_sig"] or "")) if pd.notna(r["delta_CI_lo"]) else "",
                                   axis=1)
        ).drop(columns=["delta_CI_lo","delta_CI_hi","delta_sig"]),
                 title=f"=== {ds} — CORE performance (metric: auto=R_m_holdout if present else R_m) ===")

        df_print(qc_df.assign(
            QC_strict=qc_df["QC_strict"].map(fmt_pct),
            QC_loose=qc_df["QC_loose"].map(fmt_pct),
            m_med=qc_df["m_med"].map(fmt_num),
            m_IQR=qc_df["m_IQR"].map(fmt_num),
            m_at_edge_share=qc_df["m_at_edge_share"].map(fmt_pct),
            parent_ambig=qc_df["parent_ambig"].map(fmt_pct),
            theta_med=qc_df["theta_med"].map(fmt_num),
            theta_IQR=qc_df["theta_IQR"].map(fmt_num),
        ), title=f"--- {ds} — QC & parameter summary ---")

        df_print(hvac_df.assign(
            R_med_heldout=hvac_df["R_med_heldout"].map(fmt_num),
            R_CI_heldout_lo=hvac_df["R_CI_heldout_lo"].map(fmt_num),
            R_CI_heldout_hi=hvac_df["R_CI_heldout_hi"].map(fmt_num),
            R_med_asconfigured=hvac_df["R_med_asconfigured"].map(fmt_num),
            R_CI_asconf_lo=hvac_df["R_CI_asconf_lo"].map(fmt_num),
            R_CI_asconf_hi=hvac_df["R_CI_asconf_hi"].map(fmt_num),
        ), title=f"--- {ds} — Held-out vs As-configured (if both available) ---")

        df_print(tests_df, title=f"--- {ds} — Tests & baselines (best-effort parsed) ---")

        # Optional TSVs
        if args.save_tsv:
            out_dir = reports_dir / ds
            out_dir.mkdir(parents=True, exist_ok=True)
            core_df.to_csv(out_dir / "SEGVAR__core.tsv", sep="\t", index=False)
            qc_df.to_csv(out_dir / "SEGVAR__qc.tsv", sep="\t", index=False)
            hvac_df.to_csv(out_dir / "SEGVAR__heldout_vs_asconfigured.tsv", sep="\t", index=False)
            tests_df.to_csv(out_dir / "SEGVAR__tests_baselines.tsv", sep="\t", index=False)

    # Cross-dataset pivots and winners
    p_med, p_N = core_rows_to_pivots(all_rows)
    if not p_med.empty:
        df_print(p_med.applymap(lambda x: fmt_num(x)), title="=== Cross-dataset pivot — residual median ===")
    if not p_N.empty:
        df_print(p_N.astype("Int64"), title="=== Cross-dataset pivot — N nodes ===")

    winners_df, win_counts_df = winners_and_win_counts(all_rows)
    if not winners_df.empty:
        df_print(winners_df.assign(R_med=winners_df["R_med"].map(fmt_num)), title="=== Winners per dataset ===")
    if not win_counts_df.empty:
        df_print(win_counts_df, title="=== Variant win counts (across datasets) ===")

    if args.save_tsv:
        if not p_med.empty:
            (reports_dir / "ALL_DATASETS__pivot_medians.tsv").write_text(p_med.to_csv(sep="\t"), encoding="utf-8")
        if not p_N.empty:
            (reports_dir / "ALL_DATASETS__pivot_N.tsv").write_text(p_N.to_csv(sep="\t"), encoding="utf-8")
        if not winners_df.empty:
            winners_df.to_csv(reports_dir / "ALL_DATASETS__winners.tsv", sep="\t", index=False)
        if not win_counts_df.empty:
            win_counts_df.to_csv(reports_dir / "ALL_DATASETS__variant_win_counts.tsv", sep="\t", index=False)
        print(f"[OK] Wrote TSVs under {reports_dir}")

    print(f"Done in {human_time(time.time() - t0)}")


if __name__ == "__main__":
    main()
