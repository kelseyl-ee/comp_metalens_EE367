import numpy as np
import matplotlib.pyplot as plt
from scipy.special import j1
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm



import config



MM = 1E-3
UM = 1E-6
NM = 1E-9


class FullOptForward(nn.Module):

    def __init__(self, config, phase, asm, conv, pp, pm=None, waves=None, psf_processor=None):

        super().__init__()
        self.device_ = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype_  = torch.float32

        self.config = config
        self.phase = phase
        self.asm = asm
        self.conv = conv
        self.pp = pp

        # For spatially varying model
        self.pm = pm
        self.waves = waves
        self.psf_processor = psf_processor

        if self.waves is not None:
            self.waves.generate_plane_wave_stack()

  
    
    def optical_forward(self, objs, *, normalize_psf=True, out_psfs=False):
        """
        objs: expected [B, 1, H, W] 
        returns imgs: [B, L, 255, 255] 
        """
        if not self.config.SV:
            psfs = self.asm(phase_mask=self.phase, normalize=normalize_psf)
        else:
            psfs_sv = self.asm(phase_mask=self.phase, U0_stack=self.waves.U0, normalize=normalize_psf)
            psfs = self.psf_processor(psfs_sv, self.pm, self.waves)

        imgs = self.conv.sensor_image(objs, psfs)
        if out_psfs:
            return imgs, psfs
        return imgs


    def img2fm(self, imgs):
        fm = self.pp(imgs)
        return fm
    

    def forward_features(self, x):
        """
        Grad-enabled version of extract_features for training phase + FC end-to-end.
        """
        imgs = self.optical_forward(x)   # must be differentiable
        fm = self.img2fm(imgs)
        fm = F.max_pool2d(fm, 2)
        fm = fm.flatten(1)
        return fm



    @torch.no_grad()
    def extract_features(self, dataset, batch_optical=500):
        """
        ds yields x: e.g. [B,1,28,28] in [0,1]
        returns X: e.g. [N, 8*14*14], y: [N]
        """
        loader = DataLoader(dataset, batch_size=batch_optical, shuffle=False, num_workers=0)
        feats, labels = [], []
        self.eval()

        img_examples = None

        for x, y in tqdm(loader, total=len(loader), desc="extract features"):
            x = x.to(self.device_)
            y = y.to(self.device_)

            imgs = self.optical_forward(x)
            if img_examples is None:
                img_examples = self.pp.crop_imgs(imgs[12:22].detach().cpu())

            fm = self.img2fm(imgs)
            fm = F.max_pool2d(fm, 2)                      # eg. [B,8,14,14]
            fm = fm.flatten(1)                            # eg. [B,1568]

            feats.append(fm.detach().cpu())
            labels.append(y.detach().cpu())

        X = torch.cat(feats, dim=0)
        y = torch.cat(labels, dim=0)
        return X, y, img_examples    

