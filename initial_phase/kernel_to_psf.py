import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
import math

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1] / "asm_full_opt"))
import config

MM = 1E-3
UM = 1E-6
NM = 1E-9

class Kernel2PSF(nn.Module):
    """
    Takes some desired kernel and scale them to ideal PSFs
    """

    def __init__(self, config):

        super().__init__()
        self.device_ = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype_  = torch.float32

        self.grid_N = config.GRID_N

    
    def split_kernels(self, kernels):
        if kernels.dim() == 4 and kernels.shape[1] == 1:
            kernels = kernels.squeeze(1)
        self.kernel_pos  = torch.clamp(kernels, min=0.0)
        self.kernel_neg  = torch.clamp(-kernels, min=0.0)
        self.kernel_diff = self.kernel_pos - self.kernel_neg

        K, H, W = kernels.shape
        raw_kernels = torch.empty((2 * K, H, W), dtype=kernels.dtype, device=kernels.device)
        raw_kernels[0::2] = self.kernel_pos
        raw_kernels[1::2] = self.kernel_neg
        self.raw_kernels = raw_kernels

        return raw_kernels
    

    def upsample_and_center_kernels(self, raw_kernels, upsample, grid_N=None, mode="nearest", align_corners=False):
        """
        raw_kernels: (M, N, N)
        upsample: int scale factor
        mode: "nearest" (pixel replication) or "bilinear" (smooth)
        align_corners: only used for bilinear; keep False unless you have a reason

        Returns:
            centered: (M, grid_N, grid_N)
        """
        raw_kernels = raw_kernels.to(device=self.device_, dtype=self.dtype_)

        if raw_kernels.ndim != 3:
            raise ValueError(f"raw_kernels must be (M,N,N). Got {tuple(raw_kernels.shape)}")

        M, N1, N2 = raw_kernels.shape

        mode = mode.lower()
        if mode == "nearest":
            up = raw_kernels.repeat_interleave(upsample, dim=1).repeat_interleave(upsample, dim=2)
        elif mode == "bilinear":
            # F.interpolate expects [M, C, H, W]
            up = F.interpolate(
                raw_kernels.unsqueeze(1),                 # [M,1,N,N]
                scale_factor=upsample,
                mode="bilinear",
                align_corners=align_corners
            ).squeeze(1)                                  # [M, N*upsample, N*upsample]
        else:
            raise ValueError(f"mode must be 'nearest' or 'bilinear'. Got '{mode}'")

        grid_N = self.grid_N if grid_N is None else grid_N
        Ku = up.shape[-1]
        if Ku > grid_N:
            raise ValueError(f"Upsampled kernel is {Ku}x{Ku}, larger than grid_N={grid_N}")

        centered = torch.zeros((M, grid_N, grid_N), dtype=up.dtype, device=up.device)

        start = (grid_N - Ku) // 2
        end = start + Ku
        centered[:, start:end, start:end] = up

        eps = 1e-12
        centered = centered / (centered.sum(dim=(-2, -1), keepdim=True) + eps)  # [M,1,1]

        centered = torch.rot90(centered, k=-1, dims=(-2, -1))

        return centered
    


    def rearrange_kernels(self, raw_kernels, per_row: int):
        """ 
        rearrange split kernels for PSF grid
        """
        if raw_kernels.dim() == 4 and raw_kernels.shape[1] == 1:
            raw_kernels = raw_kernels.squeeze(1)  

        N, H, W = raw_kernels.shape
        K = N // 2

        pos = raw_kernels[0::2].flip(0)  
        neg = raw_kernels[1::2].flip(0)  

        out = torch.empty_like(raw_kernels)

        i = 0
        start = 0
        while start < K:
            end = min(start + per_row, K)
            n = end - start
            out[i:i+n] = neg[start:end]
            i += n
            out[i:i+n] = pos[start:end]
            i += n
            start = end

        return out


    def stitch_and_center(self, kernels, upsample=1, upsample_mode="nearest", per_row=4, gap=10):
        # kernels: (K, n, n)
        if kernels.ndim != 3:
            raise ValueError(f"Expected (K,n,n). Got {tuple(kernels.shape)}")
        K, n, n2 = kernels.shape
        kernels = kernels.transpose(-2, -1).flip(-2)

        kernels = self.upsample_and_center_kernels(kernels, upsample=upsample, grid_N=n*upsample, mode=upsample_mode)  
        K, n, n2 = kernels.shape

        rows = math.ceil(K / per_row)
        H = rows * n + (rows - 1) * gap
        W = per_row * n + (per_row - 1) * gap

        stitched = torch.zeros((H, W), dtype=kernels.dtype, device=kernels.device)
        centers_in_stitched = torch.empty((K, 2), dtype=torch.long, device=kernels.device)

        for idx in range(K):
            r, c = divmod(idx, per_row)
            y0 = r * (n + gap)
            x0 = c * (n + gap)
            stitched[y0:y0+n, x0:x0+n] = kernels[idx]
            centers_in_stitched[idx] = torch.tensor([y0 + n//2 - 1, x0 + n//2 - 1], device=kernels.device)

        grid_N = self.grid_N
        if H > grid_N or W > grid_N:
            raise ValueError(f"Stitched image {H}x{W} larger than grid {grid_N}x{grid_N}")

        out = torch.zeros((grid_N, grid_N), dtype=kernels.dtype, device=kernels.device)
        y0 = (grid_N - H) // 2
        x0 = (grid_N - W) // 2
        out[y0:y0+H, x0:x0+W] = stitched
        out = out.unsqueeze(0)

        centers_in_grid = centers_in_stitched + torch.tensor([y0, x0], device=kernels.device)

        return out, centers_in_grid



def plot_target_psfs(psf_grid, file_name=None):
    """
    psf_grid: [2K, H, W] interleaved [pos0, neg0, pos1, neg1, ...]
    """
    psf = psf_grid.detach().cpu()
    N = psf.shape[0]          
    cols = 4
    rows = math.ceil(N / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(2*cols, 2*rows))
    axes = axes.flatten()

    for i in range(N):
        label = f"pos {i//2}" if i % 2 == 0 else f"neg {i//2}"
        axes[i].imshow(psf[i].T, cmap="gray", origin="lower")
        axes[i].set_title(label, fontsize=9)
        axes[i].axis("off")

    for i in range(N, len(axes)):
        axes[i].axis("off")

    plt.tight_layout()

    if file_name is not None:
        png_name = os.path.splitext(file_name)[0] + ".png"
        plt.savefig(png_name, dpi=300, bbox_inches="tight")

    plt.show()



def main(
    kernel_file_name,
    upsample=2,
    mode="nearest", # nearest or bilinear
    config_mode="array",
    per_row=None,
    gap=None,
    save_name="Default_MNIST_target_psf_7x7.pt"
):
    # ---- Load learned kernels ----
    kernels = torch.load(kernel_file_name, map_location="cpu")
    kernels = kernels.squeeze(1)  # eg. [8,1,7,7] -> [8,7,7]
    print("kernels shape:", kernels.shape)

    # ---- Kernel → PSF pipeline ----
    k2psf = Kernel2PSF(config)

    if config_mode == "array":
        raw_kernels = k2psf.split_kernels(kernels)
        print("raw_kernels shape:", raw_kernels.shape)

        psf_grid = k2psf.upsample_and_center_kernels(
            raw_kernels,
            upsample=upsample,
            mode=mode,
        )
        print("psf_grid shape:", psf_grid.shape)

        torch.save(psf_grid.cpu(), save_name)
        print(f"Saved psf_grid to {save_name}")

        plot_target_psfs(psf_grid, file_name=save_name)

    elif config_mode == "multiplex":
        pr = per_row if per_row is not None else 4

        raw_kernels = k2psf.split_kernels(kernels)
        print("raw_kernels shape:", raw_kernels.shape)

        raw_kernels = k2psf.rearrange_kernels(raw_kernels, per_row=pr)

        stitched, centers = k2psf.stitch_and_center(
            raw_kernels,
            upsample=upsample,
            per_row=pr,
            gap=gap,
        )
        stitched = stitched[0].T.flip(1).unsqueeze(0)

        print("Stitched PSF shape:", stitched.shape)
        print("Centers shape:", centers.shape)

        # ---- save both ----
        torch.save(stitched.cpu(), save_name)

        centers_path = save_name.replace(".pt", "_centers.pt")
        torch.save(centers.cpu() if torch.is_tensor(centers) else centers, centers_path)

        print(f"Saved stitched PSF to {save_name}")
        print(f"Saved centers to {centers_path}")

        plt.figure(figsize=(4, 4))
        plt.imshow(stitched[0].T.cpu(), cmap="gray", origin="lower")
        plt.title("Target PSF")
        plt.axis("off")
        plt.tight_layout()

        png_name = os.path.splitext(save_name)[0] + ".png"
        plt.savefig(png_name, dpi=300, bbox_inches="tight")
        plt.show()

    else:
        raise ValueError(f"Unknown config_mode: {config_mode}")


if __name__ == "__main__":
    main("MNIST_kernels_7x7.pt")

