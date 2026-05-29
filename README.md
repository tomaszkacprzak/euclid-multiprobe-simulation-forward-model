
# euclid-multiprobe-simulation-forward-model
[![arXiv](https://img.shields.io/badge/arXiv-2511.04681-b31b1b.svg)](https://arxiv.org/abs/2511.04681)

This repository provides a pipeline to forward model Euclid-like weak lensing and galaxy clustering mocks from cosmological simulations:
- **Input:** Full-sky probe maps (weak lensing signal, intrinsic alignments, and dark matter density) from the [CosmoGridV1](http://www.cosmogrid.ai/) simulation suite [[Kacprzak & Fluri et al. 2022](https://arxiv.org/abs/2209.04662)] projected using [`cosmogridv11`](https://github.com/CosmoGridCollab/cosmogridv11)
- **Output:** Self-consistent Euclid-like weak lensing (convergence) and galaxy clustering (linear bias) maps with realistic survey properties, following [[Thomsen et al. 2025](https://arxiv.org/abs/2511.04681)]
- **Deep Learning Integration**: The data storage and loading are designed to work in conjunction with the training pipeline for mutual information maximizing graph convolutional neural networks in [`euclid-deep-lss`](https://github.com/tomaszkacprzak/euclid-deep-lss/)

![](data/figures/combined_moll+gnom.png)

## Installation

Requires Python >= 3.8, TensorFlow >= 2.0, and TensorFlow-Probability.

**Step 1: Install this package**

*On HPC clusters with pre-installed TensorFlow* (recommended):
```bash
pip install -e .
```

*On systems without TensorFlow*:
```bash
pip install -e .[tf]
```

Use the first option when TensorFlow is available via system modules (e.g., `module load tensorflow`) to preserve optimized GPU/MPI configurations.

## Repository Structure

### `msfm`
- `msfm/apps` - Production scripts for parallel mock generation using [`esub-epipe`](https://cosmo-gitlab.phys.ethz.ch/cosmo_public/esub-epipe) for submission
- `msfm/utils` - Helper functions
- `msfm/fiducial_pipeline.py` and `msfm/grid_pipeline.py` - Data generators for neural network training

### `configs`
Configuration files for cosmological and astrophysical paremeter priors, fixed survey properties, forward-modeling choices, and other analysis settings.

### `data`
CosmoGridV1 properties, survey masks, and catalog ellipticities used in the shape noise generation. Note: `DESY3_noise_v11.h5` exceeds the repo's file size limit and must be generated from the source galaxy catalog via `notebooks/noise_file.ipynb`.

### `notebooks`
Notebooks for generating contents of the `data` directory.

### `pipelines`
Submission commands for distributed HPC execution via [`esub-epipe`](https://cosmo-gitlab.phys.ethz.ch/cosmo_public/esub-epipe).

## Companion Repositories
- Informative map-level neural summary statistics: [`y3-deep-lss`](https://github.com/des-science/y3-deep-lss)
- Simulation-based inference: [`multiprobe-simulation-inference`](https://github.com/des-science/multiprobe-simulation-inference)
