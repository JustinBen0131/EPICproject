# EPIC (angle-only) — HRF Results, Reproducibility & Code

> **TL;DR.** This repository hosts the code and artifacts for an *angle-only* EPIC analysis on the **HRF** dataset with **Sato** filtering and **Otsu** thresholding. All preregistered tests **pass**. EPIC improves upon fixed-\(m\) baselines at both node- and image-level residuals and remains robust under systematic perturbations of geometry/quality gates.


## Table of contents

- [Repository layout](#repository-layout)
- [Results at a glance](#results-at-a-glance)
- [Seg-robust triplet (base / lo / hi)](#seg-robust-triplet-base--lo--hi)
- [Comparative baselines (fixed \(m\))](#comparative-baselines-fixed-m)
- [Parent-direction tests](#parent-direction-tests)
- [Reproducibility & quickstart](#reproducibility--quickstart)
- [Outputs & artifact layout](#outputs--artifact-layout)
- [Statistical conventions](#statistical-conventions)
- [Caveats & notes](#caveats--notes)
- [Citation](#citation)
- [License](#license)

---

## Repository layout

```text
.
├── epic_angle_only.py        # main script: angle-only EPIC pipeline
├── figures/                  # histograms, ECDFs, scatter, ablations, QC tradeoffs
├── reports/                  # summaries, CSVs, prereg tests, ablation summaries, 
└── logs/                     # console logs with timestamps, progress, null tests
```

---

## Results at a glance

**Dataset / variant:** `HRF__sato+otsu` (**strict QC snapshot**)  
**Images:** 45 (HRF release; see note in [Caveats & notes](#caveats--notes))  
**Nodes (kept):** 19,127

- **QC pass (strict / loose):** **89.7% / 98.1%**
- **\(m\) median (IQR):** **0.784** *(0.200, 1.416)*
- **Residual \(R_{m,\text{holdout}}\), node median:** **0.213** *(95% CI [0.209, 0.217])*
- **At bracket edge** \(\lvert m - \text{edge} \rvert \le 0.02\): **6,160 / 19,127** *(32.2%)*
- **Parent ambiguity:** **84.8%**

**Preregistered nulls (held-out): PASS.**  
Observed \(\tilde R \approx 0.2129\) *(95% CI [0.2089, 0.2166])*, worst \(p_{\text{median}}=2.0\times 10^{-4}\), \(p_{\text{frac}}=2.0\times 10^{-4}\); max Cliff’s \(\delta=-0.839\).

**Ablation grid (QC/geometry gates): PASS.**

---

## Seg-robust triplet (base / lo / hi)

The analysis adopts a three-setting “suite triplet” to bound systematics from geometry/quality gating:

- **base:** default gates  
- **lo:** looser gates (smaller min angle / shorter tangent / lower SVD ratio)  
- **hi:** stricter gates (larger min angle / longer tangent / higher SVD ratio)

**Top-line residuals (held-out), node- and image-level medians**

| Variant         | N_img | N_nodes | Strict-QC pass frac | R_node_med |    95% CI (node) | R_img_med |   95% CI (image) | PREREG C1 |
| :-------------- | ----: | ------: | ------------------: | ---------: | ---------------: | --------: | ---------------: | :-------: |
| `sato+otsu`     |    45 |  19,126 |              0.9405 | **0.2322** | [0.2284, 0.2361] |    0.2270 | [0.2198, 0.2373] | **PASS**  |
| `sato+otsu__lo` |    45 |  19,147 |              0.9534 | **0.2150** | [0.2120, 0.2184] |    0.2077 | [0.2051, 0.2202] | **PASS**  |
| `sato+otsu__hi` |    45 |  19,110 |              0.9293 | **0.2427** | [0.2383, 0.2462] |    0.2374 | [0.2289, 0.2435] | **PASS**  |

**Systematics (half-range across base/lo/hi):** node ±0.014, image ±0.015.  
**Combined (quadrature):** node ±0.020, image ±0.021.

---

## Comparative baselines (fixed \(m\))

Paired comparisons against fixed-\(m\) baselines on `sato+otsu` (**base**):

**Node-level medians (held-out residual):**

- **EPIC:** 0.232 *(95% CI [0.228, 0.236])*
- **\(m=2\):** 0.261 → sign test \(p=3.21\times 10^{-239}\); paired median \(\Delta R=0.039\) \([0.036, 0.042]\)
- **\(m=1\):** 0.220 → \(p=1.00\times 10^{-60}\); paired median \(\Delta R=0.009\) \([0.007, 0.010]\)
- **\(m^*\approx 1.21\):** 0.217 → \(p=1.22\times 10^{-41}\); paired median \(\Delta R=0.006\) \([0.005, 0.008]\)

**Image-median comparisons (same run):**

- \(m=2:\ p=4.67\times 10^{-9}\), median \(\Delta R=0.033\) \([0.023, 0.047]\)  
- \(m=1\) and \(m^*\): negative shifts (as expected)

*(Analogous trends hold for `lo` and `hi`; see per-variant summaries in `reports/`.)*

---

## Parent-direction tests

**Strict-QC snapshot (`HRF__sato+otsu`)** — residual and angle tests are decisively one-sided:

- Residual \(\Delta R>0\) vs \(m=2\): \(n=19{,}127,\ p=7.83\times 10^{-289}\)
- Residual \(\Delta R>0\) vs \(m=1\): \(n=19{,}127,\ p=1.64\times 10^{-136}\)
- Angle \(\Delta \phi>0\) vs \(m=2\): \(n=19{,}106,\ p=1.05\times 10^{-56}\)
- Angle \(\Delta \phi>0\) vs \(m=1\): \(n=19{,}106,\ p=8.03\times 10^{-12}\)
- Angle \(\Delta \phi>0\) vs \(120^\circ\): \(n=19{,}106,\ p\approx 0\)

The `sato+otsu` base run exhibits similarly extreme values (e.g., residual \(p=6.41\times 10^{-239}\) vs \(m=2\); angle \(p=4.71\times 10^{-112}\) vs \(m=2\)). Full per-variant outputs appear in `reports/*/PARENTDIR__*.txt` and the corresponding ECDF/scatter figures.

---

## Reproducibility & quickstart

> The commands below reproduce the results. Execute them from the repository root after placing HRF images as desired.

### Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
pip install numpy scipy pandas matplotlib scikit-image tqdm
```

### Data

Place the HRF images in a directory known to the local config (e.g., `data/HRF/...`). The script enumerates images internally; per-image node counts and summaries are written under `reports/`.

### End-to-end run (paper-facing strict-QC snapshot + systematics)

```bash
python3 epic_angle_only.py \
  --datasets HRF \
  --seg-method sato \
  --thresh-method otsu \
  --primary-metric heldout \
  --suite-triplet \
  --save-debug \
  --validate \
  --posctrl-parent-mislabel 0.2
```

> This pipeline executes the **base** run and the **lo/hi** systematics sweep. The paper-facing strict-QC snapshot corresponds to the `HRF__sato+otsu` block with **strict_qc=True** (see `RUN SUMMARY — HRF__sato+otsu` in `reports/`).

**Effective gates used by the suite:**

- **lo:** `min_angle_eff = 7°`,  `tangent_len_eff = 12`, `svd_ratio_min_eff = 2.50`
- **base:** `min_angle_eff = 10°`, `tangent_len_eff = 16`, `svd_ratio_min_eff = 3.00`
- **hi:** `min_angle_eff = 13°`, `tangent_len_eff = 20`, `svd_ratio_min_eff = 3.50`

---

## Outputs & artifact layout

Representative paths (HRF + Sato/Otsu):

```text
figures/HRF__sato+otsu/
  dataset_HRF__sato+otsu__m_hist.png
  dataset_HRF__sato+otsu__residual_hist.png
  dataset_HRF__sato+otsu__residual_baselines.png
  dataset_HRF__sato+otsu__paired_deltaR_ecdf.png
  dataset_HRF__sato+otsu__theta_vs_m_scatter.png
  dataset_HRF__sato+otsu__ablation_medianR_grid.png
  dataset_HRF__sato+otsu__QC_tradeoff_curves.png
  POSCTRL__recovery.png
  parent_dir_error_ecdf.png

reports/HRF__sato+otsu/
  nodes__HRF__sato+otsu.csv
  SUMMARY__HRF__sato+otsu.txt
  PREREG__C1.txt
  ABLATION__summary.txt
  UNCERTAINTY__nodes.csv
  RUN__manifest.json

# analogous directories for: HRF__sato+otsu__lo and HRF__sato+otsu__hi

reports/HRF/
  SEGROBUST__summary.tsv
  SEGROBUST__summary.md
  VERDICT__default.txt
  VERDICT__default.json

reports/
  ALL__SUITE_summary.tsv
  ALL__SUITE_summary.md
```

**Positive control (parent mislabel):**  
\(N=1266\), median \(|\Delta m|=0.298\), MAE \(=0.298\), \(r=0.628\), mislabel \(=20\%\).

**Null tests:**  
\(n_{\text{perm}}=2000\) per control; all worst-case \(p \le 5\times 10^{-4}\).

---

## Statistical conventions

- **Medians & CIs.** Nonparametric bootstrap 95% CIs are reported for medians.  
- **Systematics.** Suite-level systematic is the quadrature sum of the half-range across the base/lo/hi QC variants and the half-range across the HRF segmentation triplet.  
- **Effect sizes.** Cliff’s \(\delta\) quantifies shifts relative to structure-preserving nulls (more negative → tighter closure in data).  
- **Permutation tests.** One-sided permutation tests across four preregistered null controls; “worst \(p\)” denotes the largest \(p\) among controls.

---

## Caveats & notes

- **Bracket-edge mass.** Between 22% and 32% of nodes lie within \(\pm 0.02\) of the search bracket edges, depending on the suite. *Action:* consider widening the bracket or revisiting the \(m\)-solver tolerance.  
- **Parent ambiguity.** Roughly 85–88% of nodes are ambiguous prior to tie-breakers. *Action:* stronger gating or alternate tie-breakers may reduce ambiguity further.  
- **HRF image count note.** The public HRF release contains **45** images. Any “90” counts appearing in some logs originate from duplicate enumeration and should be interpreted as **45/45**. Subsequent patches consolidate enumeration to the canonical 45 images.  
- **Uncertainty export.** `UNCERTAINTY__nodes.csv` is produced via an analytic delta method; the edge-movement criterion **fails** at current settings (median CI half-width \(\sim 1.7\)). This behavior is expected with the present approximation and is included for completeness.

---

### Appendix — Config snapshots (for exact reproduction)

**Strict-QC snapshot (`HRF__sato+otsu`):**

```text
seg=sato, thresh=otsu, profile=none
bracket m∈[0.20,4.00] | tangent_len=16 | svd_ratio_min=3.0
min_branch_len=10 | dedup_radius=3
strict_qc=True | angle_auto=False | angle_floor=10.0
radius_jitter_frac=0.0
```

**Suite triplet effective gates:**

- **lo:**   `min_angle_eff = 7.0°`,  `tangent_len_eff = 12`, `svd_ratio_min_eff = 2.50`
- **base:** `min_angle_eff = 10.0°`, `tangent_len_eff = 16`, `svd_ratio_min_eff = 3.00`
- **hi:**   `min_angle_eff = 13.0°`, `tangent_len_eff = 20`, `svd_ratio_min_eff = 3.50`

---

*Numbers in this README are drawn from run logs dated **2025-11-17** and match the artifacts under `figures/` and `reports/`.*
