# Testing a theory of branching networks with retinal vessels

Why do blood vessels divide at particular angles and with particular ratios of parent to daughter width? Justin Bennett’s two sole-authored preprints connect the geometry of a junction to the competing costs of transporting fluid and maintaining the network. The first develops an entropy–information-cost formulation. The second uses a single index to relate Murray-type width scaling, branched-transport costs and junction angles.

This repository contains Bennett’s computational test using retinal images. The pipeline extracts vessel centerlines and bifurcations and tests geometric predictions on held-out measurements. The base segmentation table reports 19,126 branching points from 45 images. Several points therefore come from the same image. The reproduction guide reports node- and image-level summaries, controls and sensitivity to segmentation choices.

## Read or reproduce the work

- [Entropy–information-cost preprint](https://arxiv.org/abs/2511.04022) · [Single-index theory preprint](https://arxiv.org/abs/2511.19915)
- [Technical reproduction guide](macro/README.md) gives the dataset requirements, command line, parameters, held-out tests and sensitivity analyses. [Python implementation](macro/epic_angle_only.py) · [Reports by segmentation setting](reports)
- The image study tests predictions about measured junction geometry. The range of results across extraction settings and the uncertainty diagnostics are reported alongside the primary analysis.

## Subsequent literature

A paper in the *Journal of the Royal Society Interface* cites the single-index theory. Riccardo Marchesi subsequently studied the homogeneous two-term cost class and called it the “Murray–Bennett family.” These publications document how other authors used the theoretical work. [Journal article](https://doi.org/10.1098/rsif.2026.0331) · [Marchesi’s treatment](https://arxiv.org/abs/2603.13687)
