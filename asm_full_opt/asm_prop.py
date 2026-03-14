import math
import torch
import torch.nn as nn


class ASMPropagator(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.device_ = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype_  = torch.float32

        self.wavl = config.WAVL
        self.grid_N = config.GRID_N
        self.dx = config.PIX_SIZE
        self.z = config.Z

        self.build_spatial_grid()
        self.build_frequency_grids()
        self.build_transfer_function()


    def build_spatial_grid(self):
        """
        Builds a centered spatial grid at the metalens plane.
        """
        N = self.grid_N
        dx = self.dx
        coords = (torch.arange(N, device=self.device_, dtype=self.dtype_)
                - (N // 2)) * dx

        x = coords.clone()
        y = coords.clone()
        X, Y = torch.meshgrid(x, y, indexing="ij")

        self.x = x
        self.y = y
        self.X = X
        self.Y = Y

        return x, y, X, Y



    def build_frequency_grids(self):
        """
        Build spatial-frequency grids matching torch.fft.fft2 ordering.
        """
        N = self.grid_N
        dx = self.dx

        fx = torch.fft.fftfreq(N, d=dx).to(device=self.device_, dtype=self.dtype_)  # cycles/m
        fy = torch.fft.fftfreq(N, d=dx).to(device=self.device_, dtype=self.dtype_)  # cycles/m

        FX, FY = torch.meshgrid(fx, fy, indexing="ij")  # [N,N]

        self.register_buffer("fx", fx)
        self.register_buffer("fy", fy)
        self.register_buffer("FX", FX)
        self.register_buffer("FY", FY)

        return fx, fy, FX, FY



    def generate_on_axis_plane_wave(self, normalize=True):
        """
        Generate a single on-axis plane wave U0(x,y) = A * exp(i*phase).
        """
        N = self.grid_N
        U0 = torch.ones((N, N), device=self.device_, dtype=torch.complex64)
        if normalize:
            eps = 1e-12
            power = (U0.abs() ** 2).sum() + eps
            U0 = U0 / torch.sqrt(power)
        return U0



    def build_transfer_function(self, z=None, evanescent=True, store=True):
        """
        Build ASM transfer function H = exp(i*kz*z).

        If z is None, uses self.z.
        """
        if z is None:
            z = self.z

        k0 = (2.0 * math.pi) / self.wavl

        kx = (2.0 * math.pi) * self.FX  # rad/m
        ky = (2.0 * math.pi) * self.FY  # rad/m

        kz_sq = (k0 * k0) - (kx * kx + ky * ky)

        if evanescent:
            kz = torch.sqrt(kz_sq.to(torch.complex64))
            H  = torch.exp(1j * kz * z)
        else:
            kz = torch.sqrt(torch.clamp(kz_sq, min=0.0))
            H = torch.exp(1j * kz.to(torch.float32) * z)

        if store:
            self.register_buffer("H", H)

        return H


    
    def forward(self, phase_mask, U0_stack=None, batch_size=None, normalize=False, 
                return_field=False, H=None, apply_phase=True):
        
        # 1. Standardize U0 to [P, N, N]
        U0 = torch.as_tensor(U0_stack if U0_stack is not None 
                             else self.generate_on_axis_plane_wave(), device=self.device_)
        if U0.ndim == 2:
            U0 = U0.unsqueeze(0)
        if U0.ndim != 3:
            raise ValueError(f"U0_stack must be 2D or 3D, got shape {tuple(U0.shape)}")
        if not torch.is_complex(U0):
            U0 = U0.to(torch.complex64)
            
        H = self.H if H is None else H.to(U0.device)
        if not torch.is_complex(H):
            H = H.to(torch.complex64)

        P = U0.shape[0] # Number of field points
        chunk_size = batch_size or P
        
        all_psfs, all_fields = [], []

        for i in range(0, P, chunk_size):
            U0_chunk = U0[i : i + chunk_size] # [chunk, N, N]
            
            if apply_phase:
                # Build pure transmission masks (phase/aperture only) using a unit field.
                # This avoids accidentally multiplying by the input field twice.
                unit_field = torch.ones_like(U0_chunk[0])
                masks = phase_mask.apply(unit_field)
                if masks.ndim == 2:
                    masks = masks.unsqueeze(0)
                U_lens = masks.unsqueeze(1) * U0_chunk.unsqueeze(0)
            else:
                U_lens = U0_chunk.unsqueeze(0) # Results in [1, chunk, N, N]

            # 2. Propagate
            Uz_chunk = torch.fft.ifft2(torch.fft.fft2(U_lens) * H)
            psf_chunk = Uz_chunk.abs()**2

            if normalize:
                psf_chunk = psf_chunk / (psf_chunk.sum(dim=(-2, -1), keepdim=True) + 1e-12)

            all_psfs.append(psf_chunk)
            if return_field: all_fields.append(Uz_chunk)

        # 3. Final Assembly: Cat along dim 1 (the field/chunk dimension)
        psf_final = torch.cat(all_psfs, dim=1) 
        
        if return_field:
            return psf_final, torch.cat(all_fields, dim=1)
        
        return psf_final
