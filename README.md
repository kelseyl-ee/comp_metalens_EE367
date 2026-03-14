# comp_metalens

This repository implements a hybrid optical–digital computational camera based on metasurface optics for image classification. The optical frontend simulates imaging through designed metalens phase profiles, while a neural network backend performs classification.

The optical model uses **Angular Spectrum Method (ASM) propagation** to compute the point spread functions (PSFs) produced by a given metalens phase profile. These PSFs are then used to simulate image formation.

Two optical hardware configurations are supported:
- **Metalens array**
- **Spatially multiplexed metalens singlet**

For image formation, two modeling assumptions can be used:
- **Shift-invariant PSF model** – a single PSF is applied across the entire image.
- **Spatially varying PSF model** – PSFs vary across the field of view and are computed for multiple field points.

The full workflow consists of three main stages:

1. **Initial phase design** – A CNN is trained on the dataset and its learned kernels are converted into target PSFs. The Gerchberg–Saxton algorithm combined with ASM is then used to compute metalens phase profiles that approximate these target PSFs.

2. **GS-phase evaluation** – Dataset images are passed through the optical forward model using the GS-designed phase profiles, and the backend fully connected (FC) classifier is retrained using these optically generated features.

3. **End-to-end optimization** – The metalens phase profiles and FC weights are jointly optimized to improve classification performance.

Additional notebooks provide demonstrations of the optical model, including PSF generation and image formation under the different optical configurations.


## Summary of Runnable Files

The following files can be run without changing any inputs:

**Optical model:**
1. `asm_full_opt/modal_test/psf_conv.ipynb` — Calculates PSFs for either shift-invariant or spatially varying modes. Also demonstrates image formation via PSF convolution for array and singlet configurations.
2. `asm_full_opt/modal_test/full_imager.ipynb` — Shows imaging through designed phase profiles for array and singlet configurations under both shift-invariant and spatially varying PSF models.

**Optimization pipeline:**
1. `initial_phase/make_initial_phase.py` — Computes starting phase profiles by training a CNN, converting its kernels into target PSFs, and generating initial phase profiles using the Gerchberg–Saxton algorithm.
2. `asm_full_opt/gs_classification.py` — Performs classification on measured images produced by GS-designed lens phases and retrains the backend FC weights on these measured images.
3. `asm_full_opt/full_optimize.py` — Performs end-to-end optimization of the phase profiles and FC weights. This script is computationally expensive, so running only a few iterations is recommended.

**Results:**
1. `asm_full_opt/plots/results_singlet.ipynb` - Shows designed phase, PSFs, example images, and classification accuracy; currently configured for MNIST
2. `asm_full_opt/plots/results_array.ipynb` - Shows designed phase, PSFs, example images, and classification accuracy; currently configured for MNIST


## Optical Models
Detailed descriptions inside Jupyter notebook files. 


## Generate starting phase via Gerchberg Saxton algorithm
To compute starting phase profiles, run:

```bash
python comp_metalens/initial_phase/make_initial_phase.py
```

(Default: **MNIST**, multiplexed singlet configuration.)

The code runs the following pipeline:
1. calls `convN_FC.py` to train a CNN with one convolutional layer on the dataset and save learned kernels 
2. calls `kernel_to_psf.py` to transform kernels into target PSFs of the lens
3. calls `ideal_psf_classification.py` to calculate classification accuracy by passing optical model through target PSFs and then applying pretrained FC weights
4. calls `contruct_phase_gs.py` to run Gerchberg-Saxton algorithm to inverse design lens phase profile given a target PSF

Visual outputs stored under ```comp_metalens/store_ouputs```
- `MNIST_8x7x7_target_psf.png`
- `MNIST_8x7x7_phase_init.png`
- **`MNIST_8x7x7_phase_init_wPSF.png` shows designed phase and their PSFs compared to target PSFs**


## Evaluate GS-phase performance and retrain backend FC layer

**Must run `initial_phase/make_initial_phase.py` first!!**
To evaluate performance of GS phase and retrain FC weights, run:

```bash
python comp_metalens/asm_full_opt/gs_classification.py
```

This script will:
1. Image the MNIST dataset through designed GS phase using the optical forward model. This will take a few minutes the first time.
2. Use digital pretrained FC weights to calculate an initial classification performance.
3. Based on the images taken, retrain backend FC weights and evaluate on test data.


## End-to-end optimization
**Must run `initial_phase/make_initial_phase.py` and  `asm_full_opt/gs_classification.py` first!!**

To perform end-to-end optimization updating phase and FC weights simultaneously, run:

```bash
python comp_metalens/asm_full_opt/full_opt_forward.py
```
Default is a few iterations for tuning after GS phase design and retrained FC weights.