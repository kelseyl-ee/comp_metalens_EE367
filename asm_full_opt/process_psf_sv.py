import torch
import torch.nn as nn
import torch.nn.functional as F

import config



class PSFProcessor(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.device_ = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype_  = torch.float32

        self.grid_N = config.GRID_N
        self.H_obj = config.H_OBJ
        self.W_obj = config.W_OBJ
        self.dx = config.PIX_SIZE
        self.block_size = config.BLOCK_SIZE
        self.psf_window_N = config.PSF_WINDOW_N
        self.tukey_alpha = config.TUKEY_ALPHA  # Tukey window alpha parameter
        self.window_overlap = None

        self.build_window()


    def compute_window_overlap(self, i_c=None, j_c=None, P=None):
        """
        1D overlap (in sensor pixels) between neighboring windows.
        """
        if i_c is None or j_c is None or P is None:
            return self.window_overlap

        Wn = int(self.psf_window_N)
        B = int(self.block_size)
        nBH = (int(self.H_obj) + B - 1) // B
        nBW = (int(self.W_obj) + B - 1) // B

        if int(P) != (nBH * nBW):
            return self.window_overlap

        # p ordering is u-major: p = block_u * nBH + block_v
        i_grid = i_c.reshape(nBW, nBH)
        j_grid = j_c.reshape(nBW, nBH)

        strides = []
        if nBW > 1:
            du = (i_grid[1:, :] - i_grid[:-1, :]).abs().reshape(-1)
            du = du[du > 0]
            if du.numel() > 0:
                strides.append(du)
        if nBH > 1:
            dv = (j_grid[:, 1:] - j_grid[:, :-1]).abs().reshape(-1)
            dv = dv[dv > 0]
            if dv.numel() > 0:
                strides.append(dv)

        if len(strides) == 0:
            return self.window_overlap

        stride_1d = torch.cat(strides).median().item()
        overlap = int(max(0, round(Wn - stride_1d)))
        self.window_overlap = overlap
        return overlap


    def build_window(self):
        """
        Build a 2D apodization window for the cropped PSF.
        Saved as self.W2d with shape [Wn, Wn].
        """
        Wn = int(self.psf_window_N)
        a = float(self.tukey_alpha)

        # Guard rails:
        # - alpha <= 0 means no apodization (all ones window)
        # - Wn <= 1 avoids division by zero in x normalization
        if Wn <= 1 or a <= 0.0:
            W2d = torch.ones((Wn, Wn), device=self.device_, dtype=self.dtype_)
        else:
            # Standard Tukey in [0, 1]
            a = min(a, 1.0)
            n = torch.arange(Wn, device=self.device_, dtype=self.dtype_)
            x = n / (Wn - 1)
            w1 = torch.ones_like(x)
            left = x < (a / 2)
            right = x > (1 - a / 2)
            w1[left] = 0.5 * (1 + torch.cos(2 * torch.pi * (x[left] / a - 0.5)))
            w1[right] = 0.5 * (1 + torch.cos(2 * torch.pi * ((x[right] - 1) / a + 0.5)))
            W2d = w1[:, None] * w1[None, :]

        self.register_buffer("W2d", W2d)

        return self.W2d


    def crop_center(self, psf_stack, pixel_map, waves, uv_samples=None, hfov_deg=None):
        """
        Crop PSFs around per-field predicted centers from wave angles.
        takes psf_stack either [P, N, N] or [L, P, N, N].

        Returns
        -------
        torch.Tensor
            Cropped PSFs with shape [P, Wn, Wn] or [L, P, Wn, Wn].
        """
        psf = torch.as_tensor(psf_stack, device=self.device_, dtype=self.dtype_)
        if psf.ndim == 3:
            has_lens_dim = False
            P, N, _ = psf.shape
            psf_flat = psf  # [P,N,N]
        elif psf.ndim == 4:
            has_lens_dim = True
            L, P, N, _ = psf.shape
            psf_flat = psf.reshape(L * P, N, N)  # [L*P,N,N]
        else:
            raise ValueError(f"psf_stack must have ndim 3 or 4, got shape {tuple(psf.shape)}")

        Wn = int(self.psf_window_N)

        if uv_samples is None:
            uv_samples = waves.uv_samples
        uv_samples = torch.as_tensor(uv_samples, device=pixel_map.device_, dtype=pixel_map.dtype_).reshape(P, 2)

        x_m, y_m = pixel_map.angles_to_sensor_xy(waves.theta_x, waves.theta_y)
        if x_m.numel() != P or y_m.numel() != P:
            raise ValueError(
                f"waves.theta_x/theta_y length must match P={P}, got "
                f"{x_m.numel()} and {y_m.numel()}"
            )

        # meters -> *pixel centers* on the PSF grid 
        dx = float(self.dx)
        c = (N - 1) / 2.0  # optical axis pixel center (float)
        i_c = c + (x_m / dx)  
        j_c = c + (y_m / dx)   

        # Update overlap in the correct sensor-pixel space.
        if self.window_overlap is None:
            self.compute_window_overlap(i_c=i_c, j_c=j_c, P=P)

        # Build local crop offsets (pixel units)
        half = (Wn - 1) / 2.0
        offs = torch.arange(Wn, device=self.device_, dtype=self.dtype_) - half  # [Wn]
        dI, dJ = torch.meshgrid(offs, offs, indexing="ij")  # each [Wn,Wn]

        # Absolute sample positions on full grid
        I = i_c.view(P, 1, 1) + dI.view(1, Wn, Wn)  # [P,Wn,Wn]
        J = j_c.view(P, 1, 1) + dJ.view(1, Wn, Wn)  # [P,Wn,Wn]

        # Normalize to [-1,1] for grid_sample (align_corners=True convention)
        j_norm = (2.0 * J / (N - 1)) - 1.0
        i_norm = (2.0 * I / (N - 1)) - 1.0
        grid = torch.stack([j_norm, i_norm], dim=-1)  # [P,Wn,Wn,2] = (x,y) = (col,row)

        # If we have multiple lenses, reuse the same [P,Wn,Wn,2] grid for each lens.
        if has_lens_dim:
            grid = grid.unsqueeze(0).expand(L, -1, -1, -1, -1).reshape(L * P, Wn, Wn, 2)

        # Sample crops (bilinear handles subpixel centers)
        psf_in = psf_flat.unsqueeze(1)  # [P,1,N,N] or [L*P,1,N,N]
        psf_crop = F.grid_sample(
            psf_in, grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True
        ).squeeze(1)  # [P,Wn,Wn] or [L*P,Wn,Wn]

        if has_lens_dim:
            psf_crop = psf_crop.reshape(L, P, Wn, Wn)

        return psf_crop


    def splat_crops_on_sensor(self, psf_crop, pixel_map, waves, index=0, frame_stride=1):
        """
        Splat cropped PSFs back onto the full sensor plane and mark crop footprints. 
        (for visualization only.)
        """
        crop = torch.as_tensor(psf_crop, device=self.device_, dtype=self.dtype_)
        if crop.ndim == 3:
            P, Wn, _ = crop.shape
            crop_use = crop
        elif crop.ndim == 4:
            L, P, Wn, _ = crop.shape
            if index < 0 or index >= L:
                raise ValueError(f"index out of range for L={L}: got {index}")
            crop_use = crop[index]
        else:
            raise ValueError(f"psf_crop must have ndim 3 or 4, got shape {tuple(crop.shape)}")

        if not hasattr(waves, "X") or not hasattr(waves, "Y"):
            raise ValueError("waves must provide sensor grids waves.X and waves.Y.")
        X, Y = waves.X, waves.Y

        X = torch.as_tensor(X, device=self.device_, dtype=self.dtype_)
        Y = torch.as_tensor(Y, device=self.device_, dtype=self.dtype_)

        N = int(X.shape[0])
        if X.shape[1] != N:
            raise ValueError(f"Sensor grid must be square [N,N], got {tuple(X.shape)}")

        x_m, y_m = pixel_map.angles_to_sensor_xy(waves.theta_x, waves.theta_y)
        x0 = X.min()
        y0 = Y.min()
        dx = float(self.dx)
        i_c = torch.round((x_m - x0) / dx).long()
        j_c = torch.round((y_m - y0) / dx).long()

        half = Wn // 2

        splat = torch.zeros((N, N), device=self.device_, dtype=self.dtype_)
        frame = torch.zeros((N, N), device=self.device_, dtype=self.dtype_)

        for p_idx in range(P):
            i0 = int(i_c[p_idx].item()) - half
            j0 = int(j_c[p_idx].item()) - half
            i1 = i0 + Wn
            j1 = j0 + Wn

            di0 = max(0, i0)
            dj0 = max(0, j0)
            di1 = min(N, i1)
            dj1 = min(N, j1)
            if di0 >= di1 or dj0 >= dj1:
                continue

            ci0 = di0 - i0
            cj0 = dj0 - j0
            ci1 = ci0 + (di1 - di0)
            cj1 = cj0 + (dj1 - dj0)

            patch = crop_use[p_idx, ci0:ci1, cj0:cj1]
            splat[di0:di1, dj0:dj1] += patch

            if (p_idx % max(1, int(frame_stride))) == 0:
                frame[di0, dj0:dj1] = 1.0
                frame[di1 - 1, dj0:dj1] = 1.0
                frame[di0:di1, dj0] = 1.0
                frame[di0:di1, dj1 - 1] = 1.0

        return splat, frame


    def normalize(self, psf_crop, eps=1e-12):
        psf = torch.as_tensor(psf_crop, device=self.device_, dtype=self.dtype_)
        denom = psf.sum(dim=(-2, -1), keepdim=True) + eps
        return psf / denom


    def apply_apodization(self, psf_crop):
        psf = torch.as_tensor(psf_crop, device=self.device_, dtype=self.dtype_)
        return psf * self.W2d


    def forward(self, psf_stack, pixel_map, gw, uv_samples=None, hfov_deg=None):
        """
        Full preprocessing pipeline:
            full PSFs -> center crop -> window -> optional renormalize

        Returns
        -------
        psf_out : torch.Tensor, shape [L, P, Wn, Wn]
        """
        psf_crop = self.crop_center(psf_stack, pixel_map, gw, uv_samples=uv_samples, hfov_deg=hfov_deg)
        if float(self.tukey_alpha) > 0.0:
            psf_crop = self.apply_apodization(psf_crop)
            psf_crop = self.normalize(psf_crop)

        return psf_crop
