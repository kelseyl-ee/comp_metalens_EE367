import numpy as np
import matplotlib.pyplot as plt
from scipy.special import j1
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.special import j1

import config

MM = 1E-3
UM = 1E-6
NM = 1E-9


class PhaseMask(nn.Module):

    def __init__(
        self,
        config,
        num_masks=1,
        init="hyperbolic",
        custom=None,
        noise_std=0.0,
        defocus_max=0*UM,
        X=None,
        Y=None,
        wrap_phase=False,
        trainable=True,
        use_aperture=True,
        test_orientation=False, # If true, quadrant test pattern
    ):
        super().__init__()

        self.device_ = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype_  = torch.float32

        self.wavl = config.WAVL  
        self.grid_N = config.GRID_N
        self.efl = config.EFL
        self.lens_D = config.LENS_D
        self.hfov = config.HFOV
        self.dx = config.PIX_SIZE

        self.num_masks = num_masks
        self.defocus_max = defocus_max
        self.wrap_phase = wrap_phase
        self.use_aperture = use_aperture

        if X is None and Y is None:
            self.build_spatial_grid()
        else:
            self.x = None  
            self.y = None
            self.register_buffer("X", X)
            self.register_buffer("Y", Y)

        # Build aperture mask (1 inside, 0 outside)
        R = torch.sqrt(self.X * self.X + self.Y * self.Y)
        A = (R <= (self.lens_D / 2.0)).to(self.dtype_)
        if test_orientation:
            # mask = (self.X > 0) & (self.Y > 0) & (self.Y < self.lens_D/4.0)
            mask = (torch.abs(self.Y) > 0) & (torch.abs(self.Y) > self.lens_D/8.0)
            A[mask] = 0
        self.register_buffer("A", A)

        # Initialize phase
        init = init.lower()
        phi0 = self._make_init_phase(init=init, custom=custom)

        if noise_std > 0.0:
            phi0 = phi0 + torch.randn_like(phi0) * noise_std

        if self.use_aperture:
            phi0 = phi0 * self.A

        if self.wrap_phase:
            phi0 = self._wrap(phi0)

        if trainable:
            self.phi = nn.Parameter(phi0)
        else:
            self.register_buffer("phi", phi0)
    
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

        self.register_buffer("X", X)
        self.register_buffer("Y", Y)

        return x, y, X, Y


    def _make_init_phase(self, init, custom=None):
        K = self.num_masks

        if init == "hyperbolic":
            dz = torch.linspace(-self.defocus_max, self.defocus_max, K, device=self.device_, dtype=self.dtype_,)
            f_list = self.efl + dz
            phi = torch.stack([self.hyperbolic_phase(self.X, self.Y, self.wavl, f.item()) for f in f_list],dim=0)
        elif init == "random":
            phi = (2 * math.pi) * torch.rand((K, self.grid_N, self.grid_N), device=self.device_, dtype=self.dtype_
            ) - math.pi
        elif init in ("zeros", "zero"):
            phi = torch.zeros((K, self.grid_N, self.grid_N), device=self.device_, dtype=self.dtype_)
        elif init == "custom":
            custom = torch.as_tensor(custom)
            phi = custom.to(device=self.device_, dtype=self.dtype_)
        else:
            raise ValueError(...)

        return phi



    @staticmethod
    def hyperbolic_phase(X, Y, wavl, efl):
        """
        Ideal focusing phase of a thin lens (hyperbolic) for a focus at distance f.

        phi(x,y) = -k * (sqrt(x^2+y^2+f^2) - f)
        """
        k = (2.0 * math.pi) / wavl
        r2 = X * X + Y * Y
        opd = torch.sqrt(r2 + efl * efl) - efl
        return -k * opd
    

    @staticmethod
    def _wrap(phi):
        return torch.atan2(torch.sin(phi), torch.cos(phi))


    def forward(self):
        phi = self.phi

        if self.wrap_phase:
            phi = self._wrap(phi)

        if self.use_aperture:
            phi = phi * self.A[None, :, :]

        return phi


    
    def apply(self, U: torch.Tensor) -> torch.Tensor:
        """
        Apply phase mask(s) to a single complex field U.

        U:   (N, N) or (K, N, N) complex
        out: (K, N, N) complex
        """
        U_2d = U.squeeze() if U.ndim == 3 else U

        phi = self.forward().to(device=U.device)   
        phase = torch.exp(1j * phi)                

        if self.use_aperture:
            A = self.A.to(device=U.device)         
            phase = phase * A[None, :, :]          

        return U_2d[None, :, :] * phase               # (K, N, N)


