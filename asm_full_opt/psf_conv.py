import numpy as np
import matplotlib.pyplot as plt
from scipy.special import j1
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import config


MM = 1E-3
UM = 1E-6
NM = 1E-9

class PSFConv(nn.Module):
    """
    Takes input object and on-axis PSFs, returns image convolved w PSFs 
    """

    def __init__(self, 
        config, 
        pixel_map,
        psf_processor=None,
        X=None, 
        Y=None,
        ):

        super().__init__()
        self.device_ = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype_  = torch.float32

        self.pixel_map = pixel_map
        self.psf_processor = psf_processor

        self.H_obj = int(config.H_OBJ)
        self.W_obj = int(config.W_OBJ)
        self.hfov = config.HFOV
        self.efl = config.EFL

        self.wavl = config.WAVL  
        self.grid_N = config.GRID_N
        self.lens_D = config.LENS_D
        self.dx = config.PIX_SIZE

        if X is None and Y is None:
            self.build_spatial_grid()
        else:
            self.x = None  
            self.y = None
            self.register_buffer("X", X)
            self.register_buffer("Y", Y)


    # ---------------- Shift invariant image formation --------------

    def render_sensor_ideal(self, obj):
        pm = self.pixel_map
        im_sensor_ideal = pm.render_sensor_ideal(obj)
        return im_sensor_ideal


    def make_otfs(self, psfs, normalize=True, eps=1e-12):
        """
        Can support psf stack [L, P, N, N]
        """
        psfs = torch.as_tensor(psfs, device=self.device_, dtype=self.dtype_)

        if psfs.ndim not in (3, 4):
            raise ValueError(f"psf must be (K, N, N) or (L, P, N, N), got shape {tuple(psfs.shape)}")

        if normalize:
            psfs = psfs / (psfs.sum(dim=(-2, -1), keepdim=True) + eps)

        psf0 = torch.fft.ifftshift(psfs, dim=(-2, -1))
        otfs = torch.fft.fft2(psf0, dim=(-2, -1))

        return otfs

    
    def shift_inv_sensor_image(self, obj, psfs):
        """
        Render final sensor image:
            object -> ideal sensor image -> PSF convolution
        """
        # 1) ideal sensor image (geometry only)
        im_ideal = self.render_sensor_ideal(obj)           # [B,1,N,N]

        # 2) OTF from PSF
        otfs = self.make_otfs(psfs)                        # [K,N,N] or [L,P,N,N] 
        if otfs.ndim == 4:
            if otfs.shape[1] != 1:
                raise ValueError(
                    f"4D psfs/otfs imply spatial variation; require P==1 for this model, got shape {tuple(otfs.shape)}"
                )
            otfs = otfs.squeeze(1)                         # [L,N,N]

        # 3) FFT-based convolution
        F_img = torch.fft.fft2(im_ideal, dim=(-2, -1))  # [B, 1, N, N]
        imgs = torch.fft.ifft2(
            F_img * otfs[None, :, :, :],                 # [B, K, N, N]
            dim=(-2, -1)
        ).real

        return imgs


    # ---------------- Spatially varying image formation --------------

    def render_sensor_ideal_sv(self, obj):
        pm = self.pixel_map
        im_sensor_ideal = pm.render_sensor_ideal(obj)   # [B,1,N,N]
        im_sensor_ideal = torch.rot90(im_sensor_ideal, k=1, dims=(-2, -1))
        return im_sensor_ideal
    

    def prep_psfs(self, psfs):
        """
        Reorder PSFs for spatially varying model.

        Input
            psfs : [L, P, Hp, Wp]
        Output
            n : grid dimension
            psfs_reordered : [L, n, n, Hp, Wp]
        """
        psfs = torch.as_tensor(psfs, device=self.device_, dtype=self.dtype_)
        L, P, Hp, Wp = psfs.shape

        n = int(round(P ** 0.5))
        if n * n != P:
            raise ValueError(f"P must be a perfect square for n x n tiling, got P={P}")
        psfs_col_major = psfs.reshape(L, n, n, Hp, Wp)
        psfs_reordered = psfs_col_major.transpose(1, 2)
        psfs_reordered = torch.rot90(psfs_reordered, k=1, dims=(-2, -1)).contiguous()

        return n, psfs_reordered
        
    
    def space_variant_convolution2(self, target_images, psfs, n, overlap, normalize=1):
        """
        Simulate captured images using spatially varying PSFs.

        Parameters
        ----------
        target_images : torch.Tensor Ideal sensor images [B,1,H,W]
        psfs : torch.Tensor Reordered PSFs [L,n,n,Hp,Wp]
        n : int Number of segments per dimension
        overlap : int Overlap between segments
        normalize : int Whether to normalize by window accumulation

        Returns
        -------
        captured_images : torch.Tensor [B,L,H,W]
        """

        B, _, H, W = target_images.shape
        L, n1, n2, Hp, Wp = psfs.shape

        if n1 != n or n2 != n:
            raise ValueError(f"PSF grid mismatch: got {psfs.shape}, expected n={n}")

        device = target_images.device
        dtype = target_images.dtype

        H_s = H // n
        W_s = W // n

        captured_images = torch.zeros(B, L, H, W, device=device, dtype=dtype)
        normalization_image = torch.zeros(H, W, device=device, dtype=dtype)

        for i in range(n):
            for j in range(n):

                h0 = max(0, i * H_s - overlap // 2)
                h1 = min(H, (i + 1) * H_s + overlap // 2)
                w0 = max(0, j * W_s - overlap // 2)
                w1 = min(W, (j + 1) * W_s + overlap // 2)

                target_segment = target_images[:,0,h0:h1,w0:w1]   # [B,Hs,Ws]

                window = self.generate_window(
                    target_segment.shape[1],
                    target_segment.shape[2],
                    device=device,
                    dtype=dtype
                )

                for l in range(L):

                    psf = psfs[l,i,j]

                    captured_segment = self.simulate_image(target_segment, psf)  # [B,Hs,Ws]

                    captured_segment = captured_segment * window

                    captured_images[:,l,h0:h1,w0:w1] += captured_segment

                normalization_image[h0:h1,w0:w1] += window

        if normalize:
            captured_images /= normalization_image.clamp_min(1e-12)[None,None]

        # Replace NaNs with 0
        captured_images = torch.nan_to_num(captured_images, nan=0.0)

        return captured_images


    def simulate_image(self, target, psf):
        """
        Convolve target image with PSF using FFT.

        target : [B,H,W]
        psf    : [Hp,Wp]

        returns
        -------
        [B,H,W]
        """

        B, H, W = target.shape
        Hp, Wp = psf.shape

        padded_height = H + Hp
        padded_width = W + Wp

        psf_pad_h = padded_height - Hp
        psf_pad_w = padded_width - Wp

        target_pad_h = padded_height - H
        target_pad_w = padded_width - W

        target_padded = torch.nn.functional.pad(
            target,
            (
                target_pad_w//2, target_pad_w - target_pad_w//2,
                target_pad_h//2, target_pad_h - target_pad_h//2
            )
        )

        psf_padded = torch.nn.functional.pad(
            psf,
            (
                psf_pad_w//2, psf_pad_w - psf_pad_w//2,
                psf_pad_h//2, psf_pad_h - psf_pad_h//2
            )
        )

        target_fft = torch.fft.fftshift(torch.fft.fftn(torch.fft.ifftshift(target_padded, dim=(-2,-1)), dim=(-2,-1)), dim=(-2,-1))
        psf_fft = torch.fft.fftshift(torch.fft.fftn(torch.fft.ifftshift(psf_padded, dim=(-2,-1)), dim=(-2,-1)), dim=(-2,-1))

        convolved_fft = target_fft * psf_fft

        convolved_image_padded = torch.fft.fftshift(
            torch.fft.ifftn(torch.fft.ifftshift(convolved_fft, dim=(-2,-1)), dim=(-2,-1)),
            dim=(-2,-1)
        ).real

        convolved_image = convolved_image_padded[
            :,
            target_pad_h//2 : target_pad_h//2 + H,
            target_pad_w//2 : target_pad_w//2 + W
        ]

        return convolved_image
    

    def generate_window(self, H_s, W_s, device=None, dtype=None):
        """
        Generate Gaussian blending window.
        """

        y = torch.linspace(-H_s/2, H_s/2, H_s, device=device, dtype=dtype)
        x = torch.linspace(-W_s/2, W_s/2, W_s, device=device, dtype=dtype)

        y, x = torch.meshgrid(y, x, indexing="ij")

        sigma = min(H_s, W_s) / 4

        window = torch.exp(-0.5 * ((x / sigma)**2 + (y / sigma)**2))

        return window
    

    def rotate_back(self, captured_imgs):
        return torch.rot90(captured_imgs, k=-1, dims=(-2, -1)).contiguous()


    def sv_sensor_image(self, obj, psfs):
        im_sensor_ideal = self.render_sensor_ideal_sv(obj)
        n, psfs_reordered = self.prep_psfs(psfs)
        overlap = self.psf_processor.window_overlap

        captured_imgs = self.space_variant_convolution2(im_sensor_ideal, psfs_reordered, n, overlap)
        captured_imgs = self.rotate_back(captured_imgs)
        
        return captured_imgs



    #-------------------------- Sensor image ----------------------------

    def sensor_image(self, obj, psf):

        if psf.ndim == 4 and psf.shape[1] == 1:
            return self.shift_inv_sensor_image(obj, psf)
        
        if psf.ndim == 4 and psf.shape[1] > 1:
            return self.sv_sensor_image(obj, psf)
        
        else:
            raise ValueError(
            f"Invalid psf shape {tuple(psf.shape)}. Expected [L,1,H,W] for shift-invariant "
            f"or [L,P,H,W] with P>1 for spatially varying PSFs."
        )       
        

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
    
    
