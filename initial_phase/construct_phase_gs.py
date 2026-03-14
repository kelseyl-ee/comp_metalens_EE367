import numpy as np
import matplotlib.pyplot as plt
from scipy.special import j1
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import os

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1] / "asm_full_opt"))
import config
from phase_mask import PhaseMask
from asm_prop import ASMPropagator
from pixel_map import PixelMap
from psf_conv import PSFConv


MM = 1E-3
UM = 1E-6
NM = 1E-9


class PhaseGS(nn.Module):
    """
    Takes input desired PSFs [K, N, N] and computes corresponding phase profiles [K, N, N]
    """

    def __init__(self, config, phase, asm, psf_ideal=None):

        super().__init__()
        self.device_ = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype_  = torch.float32

        self.phase = phase
        self.asm = asm

        if psf_ideal.ndim == 4:
            psf_ideal = psf_ideal.squeeze(1)
        self.psf_ideal = psf_ideal

        self.register_buffer("X", asm.X)
        self.register_buffer("Y", asm.Y)
        self.register_buffer("A_mask", phase.A) # circular mask of 1 and 0

        self.z = config.Z
        self.grid_N = config.GRID_N

        self.init_g()
        self.error_history = []
    

    def init_g(self, eps=1e-12):
        """
        Initialize pupil-plane complex field U [K, N, N]
        """
        A_mask = self.A_mask.to(device=self.device_, dtype=self.dtype_)          # [N,N]
        A = A_mask / torch.sqrt((A_mask**2).sum() + eps)
        self.A = A
        g_init = self.phase.apply(A)
        self.g = g_init

        return g_init

    
    def psf_amp_constraint(self, eps=1e-12):
        """
        g, psf_ideal should be [K, N, N]. apply ASM to starting g then apply image plane constraint.
        """
        g = self.g
        psf_ideal=self.psf_ideal

        if g.ndim != 3 or psf_ideal.ndim != 3:
            raise ValueError(f"g and psf_ideal must be [K,N,N]. Got {tuple(g.shape)} and {tuple(psf_ideal.shape)}")
        if g.shape != psf_ideal.shape:
            raise ValueError(f"Shape mismatch: g {tuple(g.shape)} vs psf_ideal {tuple(psf_ideal.shape)}")

        psf, Gz = self.asm(phase_mask=self.phase, normalize=True, return_field=True)
        Gz = self._collapse_to_knn(Gz)

        self.G = Gz

        # --- Image-plane amplitude constraint ---
        A_target = torch.sqrt(torch.clamp(psf_ideal, min=0.0))  # [K,N,N] amplitude target
        phase_z = torch.angle(Gz)                               # keep current phase

        Gz_prime = A_target * torch.exp(1j * phase_z)
        Gz_prime = self._collapse_to_knn(Gz_prime)  

        self.G_prime = Gz_prime

        return Gz_prime

    
    def pupil_amp_constraint(self, eps=1e-12):

        H_back = torch.conj(self.asm.H)

        _, g_prime = self.asm(
            phase_mask=self.phase,
            U0_stack=self.G_prime,          
            H=H_back,
            return_field=True,
            apply_phase=False,
            normalize=False                 
        )
        g_prime = self._collapse_to_knn(g_prime)
        self.g_prime = g_prime

        new_phase = torch.angle(g_prime)
        self.phase.phi = new_phase * self.A_mask

        g_iter = self.phase.apply(self.A)
        g_iter = self._collapse_to_knn(g_iter)
        self.g = g_iter

        return self.g


    def run_gs(self, num_iters=50):
        """
        Runs the Gerchberg-Saxton iterative loop.
        """
        for i in range(num_iters):
            self.psf_amp_constraint()
            self.pupil_amp_constraint()

            with torch.no_grad():
                current_psf = self.asm(phase_mask=self.phase, normalize=True)
                # RMSE = sqrt(mean((psf - ideal)^2))
                rmse = torch.sqrt(F.mse_loss(current_psf, self.psf_ideal))
                self.error_history.append(rmse.item())

            if (i + 1) % 20 == 0:
                print(f"Iteration {i+1}/{num_iters} - RMSE: {rmse.item():.2e}")

        return self.phase.phi



    def _collapse_to_knn(self, x):
        if x.ndim == 4:
            if x.shape[0] == 1:
                x = x.squeeze(0)   # [1,K,N,N] -> [K,N,N]
            elif x.shape[1] == 1:
                x = x.squeeze(1)   # [K,1,N,N] -> [K,N,N]
            else:
                raise ValueError(f"Expected a singleton in dim 0 or 1, got shape {tuple(x.shape)}")
        if x.ndim != 3:
            raise ValueError(f"Expected [K,N,N], got shape {tuple(x.shape)}")
        return x



def main(
    target_psf_file_name,
    num_masks=16,
    phase_guess="hyperbolic",
    num_iter=200,
    save_name="Default_MNIST_7x7_init_phase.pt"
):
    # ---- Load learned kernels ----
    asm = ASMPropagator(config)

    phase = PhaseMask(
        config,
        num_masks=num_masks,
        init=phase_guess,
        X=asm.X,
        Y=asm.Y,    
        trainable=False,
    )

    phi = phase()

    target_psf = torch.load(target_psf_file_name, map_location="cpu")
    gs = PhaseGS(config, phase, asm, target_psf)

    print(f"\nStarting GS Phase Retrieval for {num_iter} iterations...")

    new_phis = gs.run_gs(num_iters=num_iter)

    with torch.no_grad():
        final_psf, Ufinal = gs.asm(phase_mask=gs.phase, normalize=True, return_field=True)
    
    # print("Ufinal intensity sum:", (Ufinal[5].abs()**2).sum().item())
    torch.save(gs.phase.phi.cpu(), save_name)
    print(f"Saved init phase to {save_name}")
    

    # ---- visualization ----
    # plot 1
    phi = gs.phase.phi.detach().cpu()
    K = phi.shape[0]
    if K == 1:
        plt.figure(figsize=(5,5))
        plt.imshow(phi[0].T, cmap="twilight", origin="lower")
        plt.axis("off")
    else:
        cols = 4
        rows = math.ceil(K / cols)

        fig, axes = plt.subplots(rows, cols, figsize=(2*cols, 2*rows))
        axes = axes.flatten()

        for k in range(K):
            axes[k].imshow(phi[k].T, cmap="twilight", origin="lower")
            # axes[k].set_title(f"Phase {k}", fontsize=9)
            axes[k].axis("off")

        for k in range(K, len(axes)):
            axes[k].axis("off")

    plt.tight_layout()
    png_name = os.path.splitext(save_name)[0] + ".png"
    plt.savefig(png_name, dpi=300, bbox_inches="tight")
    plt.show()
        
    # plot 2

    num_display = min(16, gs.phase.phi.shape[0])
    if num_display == 1:
        fig, axes = plt.subplots(3, num_display, figsize=(2.5*num_display, 7.5))
    else:
        fig, axes = plt.subplots(3, num_display, figsize=(1*num_display, 3))

    axes = np.array(axes)
    if axes.ndim == 1:              # happens when num_display == 1
        axes = axes.reshape(3, 1)   

    with torch.no_grad():
        final_psf = gs.asm(phase_mask=gs.phase, normalize=True)

    for k in range(num_display):
        axes[0,k].imshow(gs.phase.phi[k].T.cpu(), cmap="twilight", origin="lower")
        axes[1,k].imshow(final_psf[k].T.cpu(), cmap="inferno", origin="lower")
        axes[2,k].imshow(gs.psf_ideal[k].T.cpu(), cmap="inferno", origin="lower")

        # axes[0,k].set_title(f"K{k} Phase", fontsize=9)
        # axes[1,k].set_title(f"K{k} Opt", fontsize=9)
        # axes[2,k].set_title(f"K{k} Target", fontsize=9)

        for i in range(3): axes[i,k].axis("off")

    plt.tight_layout()
    png_name = os.path.splitext(save_name)[0] + "_wPSF.png"
    plt.savefig(png_name, dpi=300, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    main()


