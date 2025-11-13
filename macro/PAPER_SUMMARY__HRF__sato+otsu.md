# EPIC Paper Summary — `HRF__sato+otsu`

- **Nodes**: 19127
- **QC (strict / loose)**: 89.7% / 98.1%
- **m median (IQR)**: 0.784 (0.200, 1.416)
- **Residual (R_m_holdout) median**: 0.213  95% CI [0.209, 0.217]
- **m at bracket edge (±0.02)**: 6160 / 19127
- **Parent ambiguity**: 84.8%

## Tests & Verdicts
- **PREREG (heldout)**: PASS | observed \~R=0.212897 95% CI [0.208941, 0.216562] | worst p_med=0.00019996, p_frac=0.00019996, max δ=-0.838987
- **Ablation**: PASS

## Parent-direction notes
```
DATASET: HRF__sato+otsu
Residual ΔR>0 vs m=2: n=19127, p=7.83e-289
Residual ΔR>0 vs m=1: n=19127, p=1.64e-136
Angle Δφ>0 vs m=2: n=19106, p=1.05e-56
Angle Δφ>0 vs m=1: n=19106, p=8.03e-12
Angle Δφ>0 vs 120°: n=19106, p=0.00e+00
```

## Run log (truncated)
```
===== RUN SUMMARY — HRF__sato+otsu =====
Dataset base: HRF

[CONFIG]
seg=sato, thresh=otsu, profile=none
bracket m∈[0.20,4.00]  |  tangent_len=16, svd_ratio_min=3.0
min_branch_len=10, dedup_radius=3
strict_qc=True  |  angle_auto=False, angle_floor=10.0
radius_jitter_frac (run arg)=0.0

[COUNTS]
Images processed   : 90/90 (with nodes in 45)
Nodes raw→deg3→kept: 71599 → 23245 → 19127  | kept/raw 26.7%  kept/deg3 82.3%

[QC & METRICS]
QC loose/strict    : 18762/19127 (98.1%)  |  17165/19127 (89.7%)
m median (IQR)     : 0.784  (0.200,1.416)   mean=0.959
R(m) median        : 0.210
m at bracket edge  : 6160  (32.2%) at ±0.02
angle_min per image: median=10.0°  range=(10.0°, 10.0°)
parent ambiguity   : 84.8%

[BASELINES]
m=2: median=0.262, paired ΔR>0 n=19127, p=7.83e-289 | m=1: median=0.218, paired ΔR>0 n=19127, p=1.64e-136 | m*=1.18: median=0.215, paired ΔR>0 n=19127, p=4.62e-99 | [img-median] m=2: n=45, p=8.65e-10 | [img-median] m=1: n=45, p=6.57e-05 | [img-median] m*=1.18: n=45, p=1.61e-02

[SKIP REASONS]
not_deg3=2895 (68.7%), bad_tangent=1175 (27.9%), angle_out_of_range=77 (1.8%), short_branch=65 (1.5%), no_m_candidate=0 (0.0%), qc_fail=0 (0.0%), small_radius=0 (0.0%)

[PREREG/ABLATION]
STEP 1 — held-out closure: PASS | Worst-case across nulls: p_med=0.00019996, p_frac=0.00019996, max δ=-0.282865
```
