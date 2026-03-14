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

class PostProcess(nn.Module):
    """
    Takes result images and process through lightweight digital backend. 
    """

    def __init__(self, 
        config, 
        pixel_map,
        mode="array",
        centers=None,
        X=None, 
        Y=None,
        ):

        super().__init__()
        self.device_ = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype_  = torch.float32

        self.pixel_map = pixel_map
        self.centers = centers
        self.mode = mode

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
        
        self.compute_img_crop()


    def compute_img_crop(self):
        """
        Compute and cache the tight bbox around the in-FoV region defined by
        self.pixel_map.obj_valid_mask.
        """
        if not hasattr(self.pixel_map, "obj_valid_mask"):
            self.pixel_map.build_obj_sampling_grid(store=True)

        valid = self.pixel_map.obj_valid_mask  # expected [1,1,N,N]
        valid = torch.as_tensor(valid, device=self.device_)

        if valid.ndim == 4:
            v2d = valid[0, 0] > 0  # [N,N] bool
        elif valid.ndim == 2:
            v2d = valid > 0
        else:
            raise ValueError(f"obj_valid_mask should be [1,1,N,N] or [N,N]. Got {tuple(valid.shape)}")

        ys, xs = torch.where(v2d)
        if ys.numel() == 0:
            raise ValueError("obj_valid_mask has no valid pixels (all zeros).")

        y0 = int(ys.min().item())
        y1 = int(ys.max().item()) + 1  # exclusive
        x0 = int(xs.min().item())
        x1 = int(xs.max().item()) + 1  # exclusive

        self.valid_img_crop = (y0, y1, x0, x1)
        return self.valid_img_crop


    def multiple_crops(self, store=True):
        """
        Compute crop windows (y0,y1,x0,x1) centered at each (y,x) in self.centers,
        using the crop size from self.valid_img_crop. Clamps to [0, grid_N].
        """
        if not hasattr(self, "valid_img_crop"):
            self.compute_img_crop()

        y0, y1, x0, x1 = self.valid_img_crop
        h = y1 - y0
        w = x1 - x0

        if self.centers is None:
            raise ValueError("self.centers is None — crop centers were not set before calling multiple_crops().")

        centers = torch.as_tensor(self.centers, device=self.device_)
        centers = centers.long()

        K = centers.shape[0]
        crops = torch.empty((K, 4), dtype=torch.long, device=self.device_)  # (y0,y1,x0,x1)

        for i in range(K):
            cy, cx = centers[i].tolist()

            yy0 = cy - h // 2
            xx0 = cx - w // 2
            yy1 = yy0 + h
            xx1 = xx0 + w

            # clamp to image bounds [0, grid_N]
            if yy0 < 0:
                yy1 -= yy0
                yy0 = 0
            if xx0 < 0:
                xx1 -= xx0
                xx0 = 0
            if yy1 > self.grid_N:
                d = yy1 - self.grid_N
                yy0 -= d
                yy1 = self.grid_N
            if xx1 > self.grid_N:
                d = xx1 - self.grid_N
                xx0 -= d
                xx1 = self.grid_N

            # final safety clamp
            yy0 = max(0, yy0); xx0 = max(0, xx0)
            yy1 = min(self.grid_N, yy1); xx1 = min(self.grid_N, xx1)

            crops[i] = torch.tensor([yy0+1, yy1+1, xx0+1, xx1+1], device=self.device_)

        if store:
            self.crop_grid = crops

        return crops


    def crop_imgs(self, imgs):
        """
        imgs: [B, K, N, N]

        - mode="array" (K>1): crop the same FoV bbox for every kernel -> [B, K, Hc, Wc]
        - mode="multiplex" (K==1): crop around each center using self.center_crops -> [B, Kcrops, Hc, Wc]
        """
        imgs = imgs.to(device=self.device_, dtype=self.dtype_)
        imgs = torch.rot90(imgs, k=3, dims=(-2, -1))  # rotate spatial dims 
        B, K, N, N2 = imgs.shape

        if imgs.ndim != 4:
            raise ValueError(f"Expected imgs [B,K,N,N]. Got {tuple(imgs.shape)}")

        if not hasattr(self, "valid_img_crop"):
            self.compute_img_crop()
        y0, y1, x0, x1 = self.valid_img_crop
        Hc, Wc = (y1 - y0), (x1 - x0)

        # ---- array mode: same crop for each channel/kernel ----
        if self.mode == "array" and K > 1:
            return imgs[:, :, y0:y1, x0:x1]

        # ---- multiplex mode: K==1, crop around each center ----
        if not hasattr(self, "crop_grid"):
            self.multiple_crops(store=True)  
        
        crop_grid = self.crop_grid  # [Kcrops, 4]
        Kcrops = crop_grid.shape[0]
        if K != 1:
            raise ValueError(f"mode='multiplex' expects imgs with K=1. Got K={K}")

        base = imgs[:, 0]  # [B, N, N]
        yy0 = crop_grid[:, 0]
        yy1 = crop_grid[:, 1]
        xx0 = crop_grid[:, 2]
        xx1 = crop_grid[:, 3]

        dy = torch.arange(Hc, device=base.device)
        dx = torch.arange(Wc, device=base.device)

        Y = yy0[:, None, None] + dy[None, :, None]   
        X = xx0[:, None, None] + dx[None, None, :] 

        Y = Y.expand(Kcrops, Hc, Wc)
        X = X.expand(Kcrops, Hc, Wc)

        b = torch.arange(B, device=base.device)[:, None, None, None]
        out = base[b, Y[None, ...], X[None, ...]]  # [B, Kcrops, h, w]

        return out



    def downsample_imgs(self, imgs, H=None, W=None):
        """
        Downsample images to (self.H_obj, self.W_obj).
        """
        imgs = imgs.to(device=self.device_, dtype=self.dtype_)

        if imgs.ndim != 4:
            raise ValueError(f"Expected imgs with shape [B,K,H,W]. Got {tuple(imgs.shape)}")
        
        H_out = self.H_obj if H is None else int(H)
        W_out = self.W_obj if W is None else int(W)

        imgs_ds = F.interpolate(imgs, size=(H_out, W_out), mode="area")

        return imgs_ds



    def subtract_imgs(self, imgs):
        """
        imgs: [B, K, H, W]
        If multiplex:
            reorder from BLOCK order → INTERLEAVED order, then subtract.
        If array:
            assume already interleaved and just subtract.
        """
        if imgs.ndim != 4:
            raise ValueError(f"Expected imgs [B,K,H,W]. Got {tuple(imgs.shape)}")

        # ---- multiplex: first undo block ordering ----
        if self.mode == "multiplex" and self.centers is not None:
            centers = torch.as_tensor(self.centers, device=imgs.device)
            first_y = centers[0, 0]
            per_row = int((centers[:, 0] == first_y).sum().item())

            B, K, H, W = imgs.shape
            pos_list, neg_list = [], []
            start = 0
            while start < K:
                n = min(per_row, (K - start) // 2)
                if n <= 0:
                    break
                pos_list.append(imgs[:, start:start+n])
                neg_list.append(imgs[:, start+n:start+2*n])
                start += 2 * n

            pos = torch.cat(pos_list, dim=1)
            neg = torch.cat(neg_list, dim=1)

        else:
            # already interleaved
            pos = imgs[:, 0::2]
            neg = imgs[:, 1::2]

        fm = pos - neg
        return fm
     


    def forward(self, imgs, subtract=True, H=None, W=None):
        crop = self.crop_imgs(imgs)
        imgs_ds = self.downsample_imgs(crop)
        
        if subtract:
            fm = self.subtract_imgs(imgs_ds)
            imgs_pp = fm
        else:
            imgs_pp = imgs_ds

        return imgs_pp
