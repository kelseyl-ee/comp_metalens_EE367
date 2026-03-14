import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os
from pathlib import Path

from convN_FC import MultiKernelCNN
import convN_FC
import kernel_to_psf
import construct_phase_gs
import ideal_psf_classification


# --------------------- Input parameters ---------------------- #
dataset = "Fashion" # MNIST or Fashion 
input_hw = (28, 28) if dataset in ["MNIST", "Fashion"] else (32, 32)
num_kernels = 8
kernel_size = 7

config_mode = "multiplex" # "array" or "multiplex" -- MAY NEED TO TUNE GRID AND LENS SIZE WHEN SWITCHING
per_row = 4 # only used if config_mode == "multiplex"
gap = 75 # only used if config_mode == "multiplex"

batch_N = 128
use_pool = True
lr = 8e-4
num_epoch = 15

# Computing target PSF
upsample=4 if config_mode == "array" else 2
mode='bilinear' # nearest or bilinear

# Computing initial phase
num_masks = 1 if config_mode == "multiplex" else 2*num_kernels
phase_guess="hyperbolic" # "random" or "hyperbolic"
num_iter=70

# Classify with ideal kernel images?
class_w_ideal_PSF = True

tag = f"{dataset}_{num_kernels}x{kernel_size}x{kernel_size}"

save_name_kernel      = f"{tag}_kernels.pt"
save_name_trained_fc  = f"{tag}_fc_final.pt"
save_name_psf         = f"{tag}_target_psf.pt"
save_name_init_phase  = f"{tag}_phase_init.pt"

# -------------------------------------------------------------- #
#                        Train kernels                           #
# -------------------------------------------------------------- #
# if save_name_kernel and save_name_trained_fc doesn't exist in the folder ../store_outputs,
# then call main of convN_FC with the correct inputs

out_dir = Path(__file__).resolve().parents[1] / "store_outputs"
os.makedirs(out_dir, exist_ok=True)

kernel_path = os.path.join(out_dir, save_name_kernel)
fc_path     = os.path.join(out_dir, save_name_trained_fc)

if not (os.path.exists(kernel_path) and os.path.exists(fc_path)):
    print("\nTrained kernels / FC not found — training CNN...")

    convN_FC.main(
        dataset=dataset,
        num_kernels=num_kernels,
        kernel_size=kernel_size,
        use_pool=use_pool,
        batch_size=batch_N,
        lr=lr,
        num_epochs=num_epoch,
        input_hw=input_hw,
        save_name_kernel=kernel_path,
        save_name_fc=fc_path,
    )

else:
    print("\nFound existing trained kernels and FC — skipping training.")
    print("Kernel:", kernel_path)
    print("FC:", fc_path)


# -------------------------------------------------------------- #
#                     Scale to Ideal PSFs                        #
# -------------------------------------------------------------- #
# if save_name_psf doesn't exist in the folder ../store_outputs,
# then call main of kernel_to_psf with the correct inputs

out_dir = Path(__file__).resolve().parents[1] / "store_outputs"
os.makedirs(out_dir, exist_ok=True)

psf_path = os.path.join(out_dir, save_name_psf)

if not (os.path.exists(psf_path)):
    print("\nTarget PSF not found, running kernel_to_psf...")

    kernel_to_psf.main(
        kernel_path,
        upsample=upsample,
        mode=mode,
        config_mode=config_mode,
        per_row=per_row,
        gap=gap,
        save_name=psf_path       
    )

else:
    print("\nFound existing target PSF — skipping kernel_to_psf")
    print("PSF:", psf_path)


# -------------------------------------------------------------- #
#                    Classify w Ideal PSF                        #
# -------------------------------------------------------------- #

if class_w_ideal_PSF:
    out_dir = Path(__file__).resolve().parents[1] / "store_outputs"
    os.makedirs(out_dir, exist_ok=True)

    psf_path = os.path.join(out_dir, save_name_psf)
    fc_path = os.path.join(out_dir, save_name_trained_fc)

    if config_mode == "multiplex":
        centers_path = os.path.join(out_dir, f"{tag}_target_psf_centers.pt")
        if not os.path.exists(centers_path):
            raise ValueError(f"Expected centers file at {centers_path} for multiplex mode. Please run kernel_to_psf to generate it.")
    else:        
        centers_path = None

    ideal_psf_classification.main(
        dataset=dataset,
        fc_file_name=fc_path,
        target_psf_file_name=psf_path,
        num_kernels=num_kernels,
        kernel_size=kernel_size,
        centers_file_name=centers_path
    )


# -------------------------------------------------------------- #
#                 Solve starting phase profiles                  #
# -------------------------------------------------------------- #
# if save_name_init_phase doesn't exist in the folder ../store_outputs,
# then call main of contruct_phase_gs with the correct inputs

out_dir = Path(__file__).resolve().parents[1] / "store_outputs"
os.makedirs(out_dir, exist_ok=True)

phase_path = os.path.join(out_dir, save_name_init_phase)

if not (os.path.exists(phase_path)):
    print("\nInit phase not found, running construct_phase_gs...")

    construct_phase_gs.main(
        psf_path,
        num_masks=num_masks,
        phase_guess=phase_guess,
        num_iter=num_iter,
        save_name=phase_path    
    )

else:
    print("\nFound existing init phase — skipping construct_phase_gs")
    print("Init phase:", phase_path)