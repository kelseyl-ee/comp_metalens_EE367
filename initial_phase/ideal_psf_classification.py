import os, sys
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

sys.path.append("asm_full_opt")
from asm_prop import ASMPropagator
from pixel_map import PixelMap
from psf_conv import PSFConv
from im_postprocess import PostProcess
import config


sys.path.append("store_outputs")
from log_results import log_result



def get_dataset(name: str):
    name = name.upper()
    if name == "MNIST":
        tfm = transforms.ToTensor()
        ds = datasets.MNIST(root="./data", train=False, download=False, transform=tfm)
    elif name in ("FASHION", "FASHIONMNIST", "FASHION_MNIST"):
        tfm = transforms.ToTensor()
        ds = datasets.FashionMNIST(root="./data", train=False, download=False, transform=tfm)
    elif name in ("CIFAR_G", "CIFAR10_G", "CIFAR_GRAY"):
        tfm = transforms.Compose([transforms.Grayscale(num_output_channels=1), transforms.ToTensor()])
        ds = datasets.CIFAR10(root="./data", train=False, download=False, transform=tfm)
    else:
        raise ValueError(f"Unknown dataset '{name}'. Use 'MNIST', 'Fashion', or 'CIFAR_G'.")
    return ds


def main(dataset, fc_file_name, target_psf_file_name, num_kernels, kernel_size, centers_file_name=None):
    # ---- optics / conv objects ----
    asm = ASMPropagator(config)
    pm = PixelMap(config, asm.X, asm.Y)
    conv = PSFConv(config, pm, asm.X, asm.Y)

    if centers_file_name is not None:
        centers = torch.load(centers_file_name, map_location="cpu")
        pp = PostProcess(config, pixel_map=pm, X=asm.X, Y=asm.Y, centers=centers, mode="multiplex")
    else:
        pp = PostProcess(config, pixel_map=pm, X=asm.X, Y=asm.Y)

    # ---- load weights ----
    target_psf = torch.load(target_psf_file_name, map_location="cpu")

    if target_psf.ndim == 3:
        target_psf = target_psf.unsqueeze(1)

    pretrained_FC = torch.load(fc_file_name, map_location="cpu")
    W, b = pretrained_FC["weight"], pretrained_FC["bias"]

    # ---- dataset ----
    ds = get_dataset(dataset)
    loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=0)

    all_imgs, all_labels = [], []
    with torch.no_grad():
        for x, y in tqdm(loader, desc=f"Evaluating {dataset}", total=len(loader)):
            imgs = conv.sensor_image(x, target_psf)
            all_imgs.append(pp(imgs).cpu())
            all_labels.append(y.cpu())

    all_imgs = torch.cat(all_imgs, dim=0)      # [N,C,H,W]
    all_labels = torch.cat(all_labels, dim=0)  # [N]

    # ---- pool + (optional) relu + fc ----
    x = F.max_pool2d(all_imgs, 2)
    x = F.relu(x)
    x = x.flatten(1)

    logits = x @ W.t() + b
    preds = logits.argmax(dim=1)
    acc = (preds == all_labels).float().mean().item() * 100.0
    print(f"Test accuracy ({dataset}): {acc:.2f}%")

    # ---- logging (define these from your loaded PSF/kernel) ----
    method = "ideal PSF + pool + FC"
    log_result(dataset, method, num_kernels, kernel_size, acc)

