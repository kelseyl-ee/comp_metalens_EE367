import numpy as np
import matplotlib.pyplot as plt
from scipy.special import j1
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from phase_mask import PhaseMask
from asm_prop import ASMPropagator
import config


MM = 1E-3
UM = 1E-6
NM = 1E-9

class PixelMap(nn.Module):
    """
    Utility class to map object pixels -> angles -> sensor shifts/pixels.
    """

    def __init__(self, config, X=None, Y=None):
        super().__init__()
        self.device_ = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype_  = torch.float32

        self.H_obj = int(config.H_OBJ)
        self.W_obj = int(config.W_OBJ)
        self.hfov = config.HFOV
        self.efl = config.EFL

        self.wavl = config.WAVL  
        self.grid_N = config.GRID_N
        self.lens_D = config.LENS_D
        self.dx = config.PIX_SIZE

        self.field_strategy = config.FIELD_STRATEGY
        self.block_size = config.BLOCK_SIZE
        self.register_buffer("uv_samples", None, persistent=False)
        self.register_buffer("pixel_to_sample", None, persistent=False)

        if X is None and Y is None:
            self.build_spatial_grid()
        else:
            self.x = None  
            self.y = None
            self.register_buffer("X", X)
            self.register_buffer("Y", Y)

    ## ------------------- Forward mapping -------------------------------
  
    def pixel_uv_grid(self, H=None, W=None, flatten=True):
        """
        Returns uv coordinates for every object pixel center.
        """
        if H is None:
            H = self.H_obj
        if W is None:
            W = self.W_obj

        u = torch.arange(W, device=self.device_, dtype=self.dtype_) + 0.5  
        v = torch.arange(H, device=self.device_, dtype=self.dtype_) + 0.5  

        # full grid of centers
        U, V = torch.meshgrid(u, v, indexing="ij")  
        uv_hw = torch.stack([U, V], dim=-1)

        if flatten:
            return uv_hw.reshape(-1, 2)          # [H*W, 2]
        return uv_hw
    
    
    def uv_to_angles(self, uv_samples, hfov_deg=None, store=True):
            """
            Map object-plane sample coordinates (u,v) to field angles (theta_x, theta_y),
            enforcing: (u,v) = (0.5, 0.5) -> ( +HFOV/sqrt(2), -HFOV/sqrt(2) )
                    (u,v) = (W-0.5, 0.5) -> ( -HFOV/sqrt(2), -HFOV/sqrt(2) )

            Returns
            -------
            theta_x : torch.Tensor, shape [P]
            theta_y : torch.Tensor, shape [P]
                Angles in radians.
            """
            if hfov_deg is None:
                hfov_deg = float(self.hfov)

            uv = torch.as_tensor(uv_samples, device=self.device_, dtype=self.dtype_)

            u = uv[:, 0]
            v = uv[:, 1]
            H = float(self.H_obj)
            W = float(self.W_obj)

            # Center in pixel-center coords 
            u_c = W / 2.0
            v_c = H / 2.0
            du = u - u_c
            dv = v - v_c

            # Normalize so that u=0.5 -> -1 and u=W-0.5 -> +1 (same for v)
            r_u = (W / 2.0) - 0.5
            r_v = (H / 2.0) - 0.5
            u_n = du / r_u
            v_n = dv / r_v

            theta_diag = math.radians(hfov_deg)
            a = theta_diag / math.sqrt(2.0)

            theta_x = torch.atan(-u_n * math.tan(a))
            theta_y = torch.atan( v_n * math.tan(a))
            if store:
                self.theta_x = theta_x
                self.theta_y = theta_y

            return theta_x, theta_y


    def angles_to_sensor_xy(self, theta_x, theta_y, efl_m=None):
        """
        Convert angles -> sensor-plane shifts using:
            x = f * tan(theta_x)
            y = f * tan(theta_y)
        """
        f = float(self.efl if efl_m is None else efl_m)

        tx = torch.as_tensor(theta_x, device=self.device_, dtype=self.dtype_)
        ty = torch.as_tensor(theta_y, device=self.device_, dtype=self.dtype_)

        sensor_x = -f * torch.tan(tx)
        sensor_y = -f * torch.tan(ty)

        return sensor_x, sensor_y


    def map_obj_to_sensor_xy(self, H=None, W=None, hfov_deg=None, efl_m=None):
        if H is None:
            H = self.H_obj
        if W is None:
            W = self.W_obj

        if hfov_deg is None:
            hfov_deg = float(self.hfov)
        
        if efl_m is None:
            efl_m = float(self.efl)

        uv = self.pixel_uv_grid(H=H, W=W)
        theta_x, theta_y = self.uv_to_angles(uv_samples=uv, hfov_deg=hfov_deg)
        sx, sy = self.angles_to_sensor_xy(theta_x=theta_x, theta_y=theta_y, efl_m=efl_m)  

        return sx, sy 
    

    def field_points_to_hit_map(self, sensor_x, sensor_y, clamp=True, test_orientation=True):
        """
        Debug-only: return an occupancy map on the sensor grid (1 if any point hits that pixel).

        Returns
        -------
        hit_map : torch.Tensor, shape [N, N], dtype float32/float64
        """
        sx = torch.as_tensor(sensor_x, device=self.device_, dtype=self.dtype_).reshape(-1)
        sy = torch.as_tensor(sensor_y, device=self.device_, dtype=self.dtype_).reshape(-1)

        if test_orientation:
            sx = sx[2:]
            sy = sy[2:]

        N = self.grid_N
        dx = self.dx
        x0 = self.X.min()
        y0 = self.Y.min()

        ix = torch.round((sx - x0) / dx).long()
        iy = torch.round((sy - y0) / dx).long()

        if clamp:
            m = (ix >= 0) & (ix < N) & (iy >= 0) & (iy < N)
            ix, iy = ix[m], iy[m]

        hit_map = torch.zeros((N, N), device=self.device_, dtype=self.dtype_)
        hit_map.index_put_((ix, iy), torch.ones_like(ix, dtype=self.dtype_), accumulate=True)

        # Convert "counts" to {0,1} occupancy
        hit_map = (hit_map > 0).to(self.dtype_)
        return hit_map

    
    ## ------------------- Backward mapping ------------------------------

    def sensor_xy_to_angles(self, X=None, Y=None, efl_m=None):
        if X is None:
            X = self.X
        if Y is None:
            Y = self.Y
        
        f = float(self.efl if efl_m is None else efl_m)

        X = torch.as_tensor(X, device=self.device_, dtype=self.dtype_)
        Y = torch.as_tensor(Y, device=self.device_, dtype=self.dtype_)

        theta_x = torch.atan(X / f)
        theta_y = torch.atan(Y / f)

        return theta_x, theta_y
    

    def angles_to_object_pix(self, theta_x, theta_y, hfov_deg=None):
        """
        Map angles -> object pixel-center coordinates (u, v), angle-based.

        Conventions enforced:
        (theta_x, theta_y) = (+a, +a) -> (u, v) = (H-0.5, 0.5)
        (theta_x, theta_y) = (+a, -a) -> (u, v) = (0.5, 0.5)
        where a = hfov / sqrt(2)

        If angles fall outside the FOV, (u, v) are set to 0.
        """
        if hfov_deg is None:
            hfov_deg = float(self.hfov)

        hfov_deg = float(hfov_deg)
        hfov = math.radians(hfov_deg)

        tx = torch.as_tensor(theta_x, device=self.device_, dtype=self.dtype_)
        ty = torch.as_tensor(theta_y, device=self.device_, dtype=self.dtype_)

        H = float(self.H_obj)
        W = float(self.W_obj)

        a = hfov / math.sqrt(2.0)
        tan_a = math.tan(a)

        # Normalized coordinates (angle-based)
        u_n = torch.tan(ty) / tan_a
        v_n = -torch.tan(tx) / tan_a
        valid = (u_n >= -1.0) & (u_n <= 1.0) & (v_n >= -1.0) & (v_n <= 1.0)

        # Map to pixel-center coordinates
        u_c = H / 2.0
        v_c = W / 2.0
        r_u = (H / 2.0) - 0.5
        r_v = (W / 2.0) - 0.5

        u = u_c + u_n * r_u
        v = v_c + v_n * r_v

        # Clamp outside-FOV rays to zero
        u = torch.where(valid, u, torch.zeros_like(u))
        v = torch.where(valid, v, torch.zeros_like(v))

        return u, v, valid


    def build_obj_sampling_grid(
        self,
        X=None,
        Y=None,
        hfov=None,
        efl_m=None,
        transpose_sensor_grid=False,
        store=True,
    ):
        """
        Build grid_sample grid that maps *sensor pixels* -> *object sampling coords*.

        Returns
        -------
        grid : [1, N, N, 2] float tensor for grid_sample (xg, yg) in [-1,1]
        valid: [1, 1, N, N] float mask (1=in-FOV, 0=out-of-FOV)
        """
        if X is None:
            X = self.X
        if Y is None:
            Y = self.Y

        X = torch.as_tensor(X, device=self.device_, dtype=self.dtype_)
        Y = torch.as_tensor(Y, device=self.device_, dtype=self.dtype_)

        # Optional: fix orientation for grid_sample so last dim is x (cols) and second last is y (rows)
        # If your X,Y came from meshgrid(indexing="ij"), transposing usually makes plotting/sampling intuitive.
        if transpose_sensor_grid:
            X = X.transpose(0, 1)
            Y = Y.transpose(0, 1)

        if hfov is None:
            hfov = float(self.hfov)
        if efl_m is None:
            efl_m = float(self.efl)

        theta_x, theta_y = self.sensor_xy_to_angles(X=X, Y=Y, efl_m=efl_m)
        u, v, valid_bool = self.angles_to_object_pix(theta_x, theta_y, hfov_deg=hfov)

        H = float(self.H_obj)
        W = float(self.W_obj)

        # grid_sample coords
        xg = 2.0 * (v - 0.5) / (W - 1.0) - 1.0  # horizontal
        yg = 2.0 * (u - 0.5) / (H - 1.0) - 1.0  # vertical

        # For out-of-FOV rays
        xg = torch.where(valid_bool, xg, torch.full_like(xg, -2.0))
        yg = torch.where(valid_bool, yg, torch.full_like(yg, -2.0))

        grid = torch.stack([xg, yg], dim=-1)          # [N, N, 2]
        grid = grid.unsqueeze(0)                      # [1, N, N, 2]

        valid = valid_bool.to(self.dtype_).unsqueeze(0).unsqueeze(0)  # [1,1,N,N]

        if store:
            # cache as buffers so they move with .to(device)
            self.register_buffer("obj_sampling_grid", grid)
            self.register_buffer("obj_valid_mask", valid)

        return grid, valid


    def render_sensor_ideal(
        self,
        obj,
        grid=None,
        padding_mode="zeros",
        align_corners=True,
    ):
        """
        Render ideal sensor image by sampling object via cached grid_sample grid.
        """
        if grid is None:
            if not hasattr(self, "obj_sampling_grid"):
                self.obj_sampling_grid, valid = self.build_obj_sampling_grid()
            grid = self.obj_sampling_grid

        obj = torch.as_tensor(obj, device=self.device_, dtype=self.dtype_)

        # Standardize obj to [B,1,H,W]
        if obj.ndim == 2:
            obj = obj.unsqueeze(0).unsqueeze(0)
        elif obj.ndim == 3:
            obj = obj.unsqueeze(1)
        elif obj.ndim == 4:
            pass
        else:
            raise ValueError(f"obj must have ndim 2, 3, or 4, got {obj.ndim}")

        B = obj.shape[0]

        # Expand grid to batch
        if grid.shape[0] == 1:
            grid_b = grid.expand(B, -1, -1, -1)
        elif grid.shape[0] == B:
            grid_b = grid
        else:
            raise ValueError(f"grid batch dim must be 1 or B. Got grid.shape[0]={grid.shape[0]}, B={B}")

        sensor_ideal = F.grid_sample(
            obj,
            grid_b,
            mode="bilinear",
            padding_mode=padding_mode,
            align_corners=align_corners,
        )
        return sensor_ideal
    
    ## ------------------- Spatially varying ------------------------------   

    def sample_field_points(self, strategy=None, block_size=None):
        """
        Determines PSF sample locations and creates a pixel-to-sample routing map.
        Uses the pixel-center convention from pixel_uv_grid.
        """
        H, W = self.H_obj, self.W_obj
        strategy = (strategy or self.field_strategy).lower()
        B = block_size or self.block_size

        if strategy == "full":
            uv_samples = self.pixel_uv_grid(H=H, W=W, flatten=True)
            
            # pixel_to_sample mapping: p = u * H + v
            u_idx = torch.arange(W, device=self.device_, dtype=torch.long).view(1, W)
            v_idx = torch.arange(H, device=self.device_, dtype=torch.long).view(H, 1)
            pixel_to_sample = u_idx * H + v_idx

        elif strategy == "block":
            # Calculate number of blocks
            nBH = (H + B - 1) // B
            nBW = (W + B - 1) // B

            # Calculate block centers in pixel-coordinates
            u_c = []
            for j in range(nBW):
                u0, u1 = j * B, min((j + 1) * B - 1, W - 1)
                u_c.append((u0 + u1 + 1) / 2.0) # Midpoint of pixel centers
            
            v_c = []
            for i in range(nBH):
                v0, v1 = i * B, min((i + 1) * B - 1, H - 1)
                v_c.append((v0 + v1 + 1) / 2.0)

            u_centers = torch.tensor(u_c, device=self.device_, dtype=self.dtype_)
            v_centers = torch.tensor(v_c, device=self.device_, dtype=self.dtype_)

            # Create the sample list [P, 2]
            U_c, V_c = torch.meshgrid(u_centers, v_centers, indexing="ij")
            uv_samples = torch.stack([U_c.reshape(-1), V_c.reshape(-1)], dim=-1)

            # Map every pixel to its block index
            v_idx = torch.arange(H, device=self.device_, dtype=torch.long).view(H, 1)
            u_idx = torch.arange(W, device=self.device_, dtype=torch.long).view(1, W)

            block_v = torch.div(v_idx, B, rounding_mode="floor")
            block_u = torch.div(u_idx, B, rounding_mode="floor")
            
            # Indexing: p = block_u * nBH + block_v
            pixel_to_sample = block_u * nBH + block_v
        
        else:
            raise ValueError(f"Strategy '{strategy}' not recognized.")

        # Store and return
        self.uv_samples = uv_samples
        self.pixel_to_sample = pixel_to_sample
        return uv_samples, pixel_to_sample 
    

    def angles_to_k(self, theta_x, theta_y):
        """
        Convert field angles to transverse wavevectors (kx, ky).
        """
        wavl = float(self.wavl)
        k0 = (2.0 * math.pi) / wavl

        tx = torch.as_tensor(theta_x, device=self.device_, dtype=self.dtype_)
        ty = torch.as_tensor(theta_y, device=self.device_, dtype=self.dtype_)

        kx = k0 * torch.sin(tx)
        ky = k0 * torch.sin(ty)

        return kx, ky
    

    #--------------------------------------------------------------------
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
