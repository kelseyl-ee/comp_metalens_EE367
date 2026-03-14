import math
import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split, TensorDataset

import sys
from pathlib import Path
sys.path.append("store_outputs")
from log_results import log_result

from phase_mask import PhaseMask
from asm_prop import ASMPropagator
from pixel_map import PixelMap
from generate_waves_sv import GenerateWaves
from process_psf_sv import PSFProcessor
from psf_conv import PSFConv
from im_postprocess import PostProcess
from full_opt_forward import FullOptForward
from retrain_FC import RetrainFC
import config


# --------------------- Inputs ---------------------- #

DATASET = "Fashion"   # or "Fashion" or "CIFAR_G"
CONFIG_MODE = "multiplex"

NUM_KERNELS = 8
KERNEL_SIZE = 7

# BASE = f"store_outputs/{DATASET}_{KERNEL_SIZE}x{KERNEL_SIZE}_{CONFIG_MODE}"
BASE = f"store_outputs"

PHASE_PATH  = f"{BASE}/{DATASET}_{NUM_KERNELS}x{KERNEL_SIZE}x{KERNEL_SIZE}_phase_init.pt"
CENTERS_PATH = f"{BASE}/{DATASET}_{NUM_KERNELS}x{KERNEL_SIZE}x{KERNEL_SIZE}_target_psf_centers.pt"
FEATURE_TAG = f"{DATASET}_{NUM_KERNELS}x{KERNEL_SIZE}x{KERNEL_SIZE}"
OUT_FC_PATH = f"{BASE}/{DATASET}_{NUM_KERNELS}x{KERNEL_SIZE}x{KERNEL_SIZE}_fc_retrained.pt"
FC_PRE_PATH = f"{BASE}/{DATASET}_{NUM_KERNELS}x{KERNEL_SIZE}x{KERNEL_SIZE}_fc_final.pt"

BATCH_OPTICAL = 256   # optical forward batch size
BATCH_FC = 64        # FC training batch size
EPOCHS = 30
LR = 5e-3
WEIGHT_DECAY = 1e-9

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_cached_features(cache_dir, split, tag, compute_fn):
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    X_path = cache_dir / f"X_{split}_{tag}.pt"
    y_path = cache_dir / f"y_{split}_{tag}.pt"
    img_pt_path  = cache_dir / f"img_{split}_{tag}.pt"     
    img_png_path = cache_dir / f"img_{split}_{tag}_ex0.png"

    if X_path.exists() and y_path.exists():
        print(f"Loading cached {split} ({tag})")
        X = torch.load(X_path, map_location="cpu")
        y = torch.load(y_path, map_location="cpu")
        return X, y

    print(f"Computing {split} ({tag})")
    X, y, img_examples = compute_fn()

    torch.save(X, X_path)
    torch.save(y, y_path)
    # torch.save(img_examples.detach().cpu(), img_pt_path)

    # ---- save sample plot ----
    ex0 = img_examples[0] 
    C = ex0.shape[0]
    cols = math.ceil(C / 2)

    plt.figure(figsize=(cols * 1.5, 3))
    for c in range(C):
        plt.subplot(2, cols, c + 1)
        plt.imshow(ex0[c], cmap="gray")
        plt.title(f"ch {c}", fontsize=9)
        plt.axis("off")

    plt.suptitle(f"{split} img_examples ex0 ({tag})")
    plt.tight_layout()
    plt.savefig(img_png_path, dpi=150)
    plt.close()

    return X, y

# -------------------------------------------------- #


def main():
    # ---- load phase ----
    phase_init = torch.load(PHASE_PATH, map_location="cpu")

    # ---- build optical pipeline ----
    asm = ASMPropagator(config)
    phase = PhaseMask(config, init="custom", custom=phase_init, X=asm.X, Y=asm.Y)
    pm = PixelMap(config, asm.X, asm.Y)
    waves = GenerateWaves(config, pm, X=asm.X, Y=asm.Y)
    processor = PSFProcessor(config)
    conv = PSFConv(config, pm, processor, asm.X, asm.Y)

    if CONFIG_MODE == "multiplex":
        centers = torch.load(CENTERS_PATH, map_location="cpu")
        pp = PostProcess(config, pixel_map=pm, mode="multiplex",centers=centers, X=asm.X, Y=asm.Y)
    else:
        pp = PostProcess(config, pixel_map=pm, X=asm.X, Y=asm.Y)

    model_opt = FullOptForward(config, phase, asm, conv, pp, pm, waves, processor).to(DEVICE)
    
    # ---- datasets ----
    transform = transforms.ToTensor()

    if DATASET == "MNIST":
        train_dataset = torchvision.datasets.MNIST(root="./data", train=True, download=True, transform=transform)
        test_dataset = torchvision.datasets.MNIST(root="./data", train=False, download=True, transform=transform)
    elif DATASET == "Fashion":
        train_dataset = torchvision.datasets.FashionMNIST(root="./data", train=True, download=True, transform=transform)
        test_dataset = torchvision.datasets.FashionMNIST(root="./data", train=False, download=True, transform=transform)
    elif DATASET == "CIFAR_G":
        # CIFAR-10 converted to grayscale
        transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
        ])
        train_dataset = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
        test_dataset = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)
    else:
        raise ValueError(f"Unknown dataset: {DATASET}")

    X_train, y_train = get_cached_features(
        "store_outputs/feature_cache",
        "train",
        tag=FEATURE_TAG,
        compute_fn=lambda: model_opt.extract_features(train_dataset, batch_optical=BATCH_OPTICAL)
    )

    X_test, y_test = get_cached_features(
        "store_outputs/feature_cache",
        "test",
        tag=FEATURE_TAG,
        compute_fn=lambda: model_opt.extract_features(test_dataset, batch_optical=BATCH_OPTICAL)
    )

    D = X_train.shape[1]
    print(f"Feature dim D={D} | X_train {tuple(X_train.shape)} | X_test {tuple(X_test.shape)}")


    # ---- evaluate pretrained FC ----
    state = torch.load(FC_PRE_PATH, map_location=DEVICE)
    W, b = state["weight"], state["bias"]

    with torch.no_grad():
        X_test = F.relu(X_test)  
        logits = X_test @ W.T + b
        preds = logits.argmax(dim=1)

    acc = (preds == y_test).float().mean().item() * 100
    print(f"Pretrained FC test accuracy: {acc:.2f}%")

    log_result(
        dataset_name=DATASET,
        method="GS PSF + FC",
        num_kernels=NUM_KERNELS,
        kernel_size=KERNEL_SIZE,
        accuracy=acc,
    )

    # ---- retrain FC  ----
    clf = RetrainFC(D).to(DEVICE)
    opt = torch.optim.Adam(clf.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    loss_fn = nn.CrossEntropyLoss()

    train_loader = DataLoader(
        TensorDataset(X_train, y_train),
        batch_size=BATCH_FC,
        shuffle=True,
        num_workers=0,
    )

    test_loader = DataLoader(
        TensorDataset(X_test, y_test),
        batch_size=1024,
        shuffle=False,
        num_workers=0,
    )

    for epoch in range(1, EPOCHS + 1):
        clf.train()
        total_loss = 0.0

        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)

            opt.zero_grad()
            loss = loss_fn(clf(xb), yb)
            loss.backward()
            opt.step()

            total_loss += loss.item()

        tr_acc = clf.eval_acc(train_loader)

        if epoch % 10 == 0 or epoch == 1:
            print(
                f"Epoch {epoch:02d}/{EPOCHS} | "
                f"loss {total_loss/len(train_loader):.4f} | "
                f"train {tr_acc*100:.2f}% "
            )

    clf.eval()
    with torch.no_grad():
        X_test = F.relu(X_test) 
        logits = clf(X_test)            # [N, C]
        preds = logits.argmax(dim=1)    # [N]

    acc = (preds == y_test).float().mean().item() * 100
    print(f"Retrained FC test accuracy: {acc:.2f}%")

    log_result(
        dataset_name=DATASET,
        method="GS PSF + Retrained FC",
        num_kernels=NUM_KERNELS,
        kernel_size=KERNEL_SIZE,
        accuracy=acc,
    )

    # ---- save FC weights  ----
    torch.save(
        {
            "weight": clf.fc.weight.detach().cpu(),
            "bias": clf.fc.bias.detach().cpu(),
        },
        OUT_FC_PATH,
    )
    print(f"Saved retrained FC to: {OUT_FC_PATH}")



if __name__ == "__main__":
    main()
