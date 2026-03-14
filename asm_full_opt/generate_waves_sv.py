import numpy as np
import matplotlib.pyplot as plt
import config
from scipy.special import j1
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.special import j1


MM = 1E-3
UM = 1E-6
NM = 1E-9

class GenerateWaves(nn.Module):
    
    def __init__(self, config, pixel_map, X=None, Y=None):
        super().__init__()
        self.device_ = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype_  = torch.float32

        self.pixel_map = pixel_map

        self.wavl = config.WAVL  
        self.grid_N = config.GRID_N
        self.efl = config.EFL
        self.lens_D = config.LENS_D
        self.hfov = config.HFOV
        self.dx = config.PIX_SIZE

        self.H_obj = config.H_OBJ
        self.W_obj = config.W_OBJ
        self.field_strategy = config.FIELD_STRATEGY
        self.block_size = config.BLOCK_SIZE

        if X is None and Y is None:
            self.build_spatial_grid()
        else:
            self.x = None  
            self.y = None
            self.register_buffer("X", X)
            self.register_buffer("Y", Y)
        
    

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
    

    
    def make_plane_waves(self, kx, ky, X=None, Y=None):
        # Use self.X if none provided; handles device/dtype automatically
        X = self.X if X is None else X
        Y = self.Y if Y is None else Y

        # Shape kx/ky to [P, 1, 1] for broadcasting across the [N, N] grid
        phase = kx.view(-1, 1, 1) * X + ky.view(-1, 1, 1) * Y

        # 1j * phase creates a complex64 tensor automatically
        return torch.exp(-1j * phase)

    
    
    def generate_plane_wave_stack(self, strategy=None, block_size=None, hfov_deg=None):
        """
        Calls PixelMap to determine sampling, then generates the physical fields.
        """
        strategy = strategy or self.field_strategy
        block_size = block_size or self.block_size
        hfov_deg = hfov_deg or self.hfov
        
        # 1. Coordinate/Math logic
        uv_samples, pixel_to_sample = self.pixel_map.sample_field_points(
            strategy=strategy, 
            block_size=block_size
        )
        
        theta_x, theta_y = self.pixel_map.uv_to_angles(
            uv_samples, 
            hfov_deg=hfov_deg, 
            store=False
        )
        
        kx, ky = self.pixel_map.angles_to_k(theta_x, theta_y)

        # 2. Store as buffers (so they move with the model to GPU/CPU)
        self.register_buffer("uv_samples", uv_samples, persistent=False)
        self.register_buffer("pixel_to_sample", pixel_to_sample, persistent=False)
        self.register_buffer("theta_x", theta_x, persistent=False)
        self.register_buffer("theta_y", theta_y, persistent=False)
        self.register_buffer("kx", kx, persistent=False)
        self.register_buffer("ky", ky, persistent=False)

        # 3. Generate the actual complex wavefronts
        U0 = self.make_plane_waves(kx, ky)
        self.register_buffer("U0", U0, persistent=False)

        return U0