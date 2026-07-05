<div align="center">
<h1><strong>Depth-progressive 3D seismic velocity model building via 2D generative diffusion models</strong></h1>
<h4>Shijun Cheng, Randy Harsuko, Mohammad H. Taufik, and Tariq Alkhalifah</h3>
<h4><em>DeepWave Consortium, King Abdullah University of Science and Technology (KAUST)</em></h4>
<p><em>Corresponding author: Shijun Cheng (<a href="mailto:sjcheng.academic@gmail.com">sjcheng.academic@gmail.com</a>)</em></p>
</div>

## Overview

Three-dimensional (3D) velocity model building (VMB) is essential for seismic imaging but remains computationally prohibitive for deep-learning approaches that operate directly in 3D. DiffVMB3D addresses this challenge by reformulating 3D VMB as a 2D generative diffusion problem through a **depth-as-channel** representation: the depth dimension of each 3D velocity patch is mapped to the channel dimension of a standard 2D U-Net, eliminating the need for 3D convolutions entirely.

The method builds velocity models progressively from shallow to deep via a **depth-progressive recursive inference** procedure. Starting from a known or estimated shallow velocity, the diffusion model generates successively deeper patches, with each prediction conditioned on the previously generated (shallower) patch. The network accepts multiple optional conditioning inputs, like structural attributes (e.g., seismic images) and well-log velocities, which are handled gracefully through **classifier-free guidance** training. Overlapping patches are merged using a **depth-attenuated Gaussian blending** scheme that accounts for the recursive error propagation inherent in the top-down generation strategy.

## Project structure

This repository is organized as follows:

* **diffvmb3d/**: Python library containing routines for 3D velocity model building using 2D generative diffusion models.
* **dataset/**: Folder to store training and test datasets (downloaded from Zenodo).
* **checkpoints/**: Folder to store trained model weights (downloaded from Zenodo).

## Supplementary files

To ensure reproducibility, we provide the synthetic dataset for training and testing, as well as our trained model weights on Zenodo.

### Training and testing dataset

Download `train.zip` and `test.zip` from [Zenodo](https://doi.org/10.5281/zenodo.21206427), then extract the contents:

```bash
# Download and extract training data (5,500 3D velocity models, .npz format)
unzip train.zip -d dataset/train/

# Download and extract test data (4 benchmark models, .mat format)
unzip test.zip -d dataset/test/
```

The training data contains 5,500 3D velocity models of size 128 × 128 × 128, each stored as a `.npz` file with keys `v3d` (velocity), `ref` (reflectivity), and `seis` (synthetic seismic image). The test data contains the four benchmark models (Overthrust, SEAM Arid, SEG/EAGE, Marmousi) as `.mat` files with the same keys. For full details on dataset generation, see the Zenodo record.

### Trained model

Download `trained_model.zip` from [Zenodo](https://doi.org/10.5281/zenodo.21206427), then extract:

```bash
# Download and extract trained model weights (EMA, decay=0.999, 200K iterations)
unzip trained_model.zip -d checkpoints/
```

## Getting started :space_invader: :robot:

To ensure reproducibility of the results, we suggest using the `environment.yml` file when creating an environment.

Simply run:
```bash
./install_env.sh
```
It will take some time, if at the end you see the word `Done!` on your terminal you are ready to go. Activate the environment by typing:
```bash
conda activate diffvmb3d
```
After that you can simply install your package:
```bash
pip install .
```
or in developer mode:
```bash
pip install -e .
```

## Running code :page_facing_up:

When you have downloaded the supplementary files and have installed the environment, you can run the training and inference code.

### Training

```bash
python train.py
```

Key training arguments (see `train.py` for the full list):
| Argument | Default | Description |
|---|---|---|
| `--batch_size` | 24 | Training batch size |
| `--lr` | 1e-4 | Learning rate (AdamW) |
| `--ema_rate` | 0.999 | EMA decay rate |
| `--wellcond_drop` | 0.05 | CFG dropout probability for well constraint |
| `--refcond_drop` | 0.05 | CFG dropout probability for structural attribute |

### Inference

```bash
python sample.py
```

Key inference arguments (see `sample.py` for the full list):
| Argument | Default | Description |
|---|---|---|
| `--model_path` | `./checkpoints/trained_model.pt` | Path to trained model |
| `--use_ddim` | True | Use DDIM sampling (10-step) |
| `--batch_size` | 50 | Ensemble size for mean/std estimation |
| `--use_well` | False | Enable well-log velocity conditioning |
| `--use_ref` | False | Enable structural attribute conditioning |
| `--use_wellguide` | False | Enable well gradient guidance |
| `--scale_factor` | 20 | Guidance gradient step size |

**Disclaimer:** All experiments have been carried on a Intel(R) Xeon(R) CPU @ 2.10GHz equipped with a single NVIDIA GeForce A100 GPU. Different environment configurations may be required for different combinations of workstation and GPU. If your graphics card does not support large batch size training, please reduce the `--batch_size` argument accordingly.

## Acknowledgements

This implementation is motivated by the paper [Improved Denoising Diffusion Probabilistic Models](https://arxiv.org/pdf/2102.09672) and the code is adapted from their [repository](https://github.com/openai/improved-diffusion). We are grateful for their open-source code.

## Cite us

Cheng, S., Harsuko, R., Taufik, M. H., and Alkhalifah, T. (2025). Depth-progressive 3D seismic velocity model building via 2D generative diffusion models.