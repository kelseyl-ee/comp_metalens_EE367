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

BASE = f"store_outputs"

OUT_PHASE_DIR = Path(BASE) / "phase_checkpoints"
OUT_PHASE_DIR.mkdir(parents=True, exist_ok=True)

PHASE_PATH  = f"{BASE}/{DATASET}_{NUM_KERNELS}x{KERNEL_SIZE}x{KERNEL_SIZE}_phase_init.pt"
CENTERS_PATH = f"{BASE}/{DATASET}_{NUM_KERNELS}x{KERNEL_SIZE}x{KERNEL_SIZE}_target_psf_centers.pt"
FC_PATH    = f"{BASE}/{DATASET}_{NUM_KERNELS}x{KERNEL_SIZE}x{KERNEL_SIZE}_fc_final.pt"
FEATURE_TAG = f"{DATASET}_{NUM_KERNELS}x{KERNEL_SIZE}x{KERNEL_SIZE}"
OUT_FC_PATH = f"{BASE}/{DATASET}_{NUM_KERNELS}x{KERNEL_SIZE}x{KERNEL_SIZE}_fc_trained.pt"
OUT_PHASE_PATH = f"{BASE}/{DATASET}_{NUM_KERNELS}x{KERNEL_SIZE}x{KERNEL_SIZE}_phase_trained.pt"

BATCH_EVAL = 256
BATCH = 64 
EPOCHS = 2
LR_PHASE = 5e-4
LR_FC = 3e-3
WEIGHT_DECAY = 0.0

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --------------------- Some helper functions ---------------------- #

def make_loader(ds, bs, shuffle):
    return DataLoader(ds, batch_size=bs, shuffle=shuffle, num_workers=0)

@torch.no_grad()
def eval_acc(model_opt, clf, loader, device):
    model_opt.eval()
    clf.eval()
    correct = total = 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        fm = model_opt.forward_features(xb)
        logits = clf(F.relu(fm))
        pred = logits.argmax(dim=1)
        correct += (pred == yb).sum().item()
        total += yb.numel()
    return correct / total

@torch.no_grad()
def eval_loss(model_opt, clf, loss_fn, loader, device):
    model_opt.eval()
    clf.eval()
    total = 0.0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        fm = model_opt.forward_features(xb)
        logits = clf(F.relu(fm))
        total += loss_fn(logits, yb).item()
    return total / len(loader)


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

    # Split train into train/val
    VAL_FRAC = 0.1   # 10% val

    N = len(train_dataset)
    n_val = int(VAL_FRAC * N)
    n_train = N - n_val

    train_ds, val_ds = random_split(
        train_dataset,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(0),
        )


    DEBUG_FRAC = 0.05  # use 10% of data
    def subsample(ds, frac):
        n = int(len(ds) * frac)
        return torch.utils.data.Subset(ds, range(n))

    # train_ds_dbg = subsample(train_ds, DEBUG_FRAC)
    # val_ds_dbg   = subsample(val_ds, DEBUG_FRAC)
    # test_ds_dbg  = subsample(test_dataset, DEBUG_FRAC)

    # train_loader = make_loader(train_ds_dbg, BATCH, True)
    # val_loader   = make_loader(val_ds_dbg, BATCH_EVAL, False)
    # test_loader  = make_loader(test_ds_dbg, BATCH_EVAL, False)
    # print(f"\nData loaded (DEBUG) | train {len(train_ds_dbg)} | val {len(val_ds_dbg)} | test {len(test_ds_dbg)} | batch {BATCH}/{BATCH_EVAL}")


    train_loader = make_loader(train_ds, BATCH, True)
    val_loader   = make_loader(val_ds, BATCH_EVAL, False)
    test_loader  = make_loader(test_dataset, BATCH_EVAL, False)
    print(f"\nData loaded | train {len(train_ds)} | val {len(val_ds)} | test {len(test_dataset)} | batch {BATCH}/{BATCH_EVAL}")


    # ---------- Set up optimizer and loss ---------- #

    H_pool = 14
    W_pool = 14
    D = NUM_KERNELS * H_pool * W_pool

    clf = RetrainFC(D).to(DEVICE)

    # load pretrained FC 
    state = torch.load(FC_PATH, map_location=DEVICE)
    clf.fc.weight.data.copy_(state["weight"])
    clf.fc.bias.data.copy_(state["bias"])

    opt = torch.optim.Adam(
        [
            {"params": model_opt.phase.parameters(), "lr": LR_PHASE},
            {"params": clf.parameters(),             "lr": LR_FC},
        ],
        weight_decay=WEIGHT_DECAY,
    )
    loss_fn = nn.CrossEntropyLoss()

    # ---- phase snapshots (for plotting later) ----
    phase_snaps = []         
    phase_titles = []        

    with torch.no_grad():
        phi0 = model_opt.phase().detach().cpu()   # "effective" phase (wrap/aperture applied if enabled)
    phase_snaps.append(phi0)
    phase_titles.append("init")

    # ---------- Training loop ---------- #

    batch_loss_hist = []
    batch_step_hist = []
    val_loss_hist   = []
    train_loss_hist = []
    val_acc_hist    = []

    global_step = 0

    for epoch in range(1, EPOCHS + 1):
        model_opt.train()
        clf.train()

        running = 0.0
        LOG_EVERY = 20  
        

        num_batches = len(train_loader)
        for i, (xb, yb) in enumerate(train_loader):
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)

            opt.zero_grad(set_to_none=True)
            fm = model_opt.forward_features(xb)
            logits = clf(F.relu(fm))
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()

            global_step += 1
            running += loss.item()

            if i % LOG_EVERY == 0:
                avg = running / (i + 1)
                batch_loss_hist.append(avg)
                batch_step_hist.append(global_step)
                print(f"[epoch {epoch}] batch {i+1}/{len(train_loader)} | loss {loss.item():.4f} | avg {avg:.4f}")

        
        avg_train_loss = running / len(train_loader)
        train_loss_hist.append(avg_train_loss)
        avg_val_loss = eval_loss(model_opt, clf, loss_fn, val_loader, DEVICE)
        val_loss_hist.append(avg_val_loss)

        va = eval_acc(model_opt, clf, val_loader, DEVICE)
        val_acc_hist.append(va)

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"loss {running/len(train_loader):.4f} | "
            f"val loss {avg_val_loss:.4f} | "
            f"val {va*100:.2f}%"
        )

        phase_epoch_path = OUT_PHASE_DIR / f"{DATASET}_{NUM_KERNELS}x{KERNEL_SIZE}x{KERNEL_SIZE}_phase_epoch{epoch:03d}.pt"
        torch.save(model_opt.phase.phi.detach().cpu(), phase_epoch_path)
        print("Saved phase:", phase_epoch_path)

        # ---- also store for final plotting (effective phase) ----
        with torch.no_grad():
            phi_eff = model_opt.phase().detach().cpu()
        phase_snaps.append(phi_eff)
        phase_titles.append(f"ep{epoch:03d}")



    # ------  Evaluate on test set ------ #

    test_acc = eval_acc(model_opt, clf, test_loader, DEVICE) * 100
    print(f"Final test accuracy: {test_acc:.2f}%")

    log_result(
        dataset_name=DATASET,
        method="End-to-end",
        num_kernels=NUM_KERNELS,
        kernel_size=KERNEL_SIZE,
        accuracy=test_acc,
    )

    # save results
    torch.save(
        {"weight": clf.fc.weight.detach().cpu(), "bias": clf.fc.bias.detach().cpu()},
        OUT_FC_PATH,
    )
    torch.save(model_opt.phase.phi.detach().cpu(), OUT_PHASE_PATH)
    print("Saved:", OUT_FC_PATH)
    print("Saved:", OUT_PHASE_PATH)

    # ---- plot phase evolution (all snapshots) ----

    K = 1   # number of phase masks

    # select snapshots: init + every other epoch
    snap_indices = list(range(0, len(phase_snaps), 2))
    phases_sel   = [phase_snaps[i] for i in snap_indices]
    titles_sel   = [phase_titles[i] for i in snap_indices]

    n_cols = len(phases_sel)
    n_rows = K

    plt.figure(figsize=(n_cols * 2.0, n_rows * 2.0))

    for r in range(K):                 # each phase mask
        for c, (phi_all, title) in enumerate(zip(phases_sel, titles_sel)):
            if phi_all.ndim == 2:
                phi = phi_all
            else:
                idx_mask = min(r, phi_all.shape[0] - 1)
                phi = phi_all[idx_mask]

            idx = r * n_cols + c + 1
            plt.subplot(n_rows, n_cols, idx)
            phi_wrapped = ((phi + torch.pi) % (2 * torch.pi)) - torch.pi
            plt.imshow(phi_wrapped.T, cmap="twilight", origin="lower", vmin=-torch.pi, vmax=torch.pi)

            if r == 0:
                plt.title(title, fontsize=9)
            if c == 0:
                plt.ylabel(f"k={r}", fontsize=9)
            plt.axis("off")

    plt.suptitle(
        f"Phase evolution ({K} masks) | {DATASET} {NUM_KERNELS}x{KERNEL_SIZE}x{KERNEL_SIZE}",
        y=0.995
    )
    plt.tight_layout()

    fig_path = Path(BASE) / f"{DATASET}_{NUM_KERNELS}x{KERNEL_SIZE}x{KERNEL_SIZE}_phase_evolution_all_k.png"
    plt.savefig(fig_path, dpi=150)
    # plt.show()
    plt.close()

    print("Saved phase evolution plot:", fig_path)


    # ---- plot loss curves ----

    plt.figure(figsize=(6, 4))

    # batch-level running avg (x = global step)
    if len(batch_step_hist) > 0:
        plt.plot(batch_step_hist, batch_loss_hist, color='gray', linewidth=1, label="train loss (batch running avg)")

    # epoch-level losses (x = step at end of each epoch)
    steps_per_epoch = len(train_loader)
    epoch_steps = [steps_per_epoch * e for e in range(1, len(train_loss_hist) + 1)]
    plt.plot(epoch_steps, train_loss_hist, color='blue', label="train loss (epoch avg)")
    plt.plot(epoch_steps, val_loss_hist, color='red', label="val loss (epoch avg)")

    plt.xlabel("Training step")
    plt.ylabel("Cross-entropy loss")
    plt.title(f"{DATASET} loss curves")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.yscale("log")
    plt.tight_layout()

    # show or save
    loss_fig_path = Path(BASE) / f"{DATASET}_{NUM_KERNELS}x{KERNEL_SIZE}x{KERNEL_SIZE}_loss_curves.png"
    plt.savefig(loss_fig_path, dpi=150)
    # plt.show()
    print("Saved loss plot:", loss_fig_path)

    # ---- plot validation accuracy curve ----
    plt.figure(figsize=(6, 4))

    epochs = list(range(1, len(val_acc_hist) + 1))
    plt.plot(epochs, np.array(val_acc_hist) * 100, color='blue', marker="s", label="val acc")

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(f"{DATASET} validation accuracy")
    plt.ylim(50, 100)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    acc_fig_path = Path(BASE) / f"{DATASET}_{NUM_KERNELS}x{KERNEL_SIZE}x{KERNEL_SIZE}_val_accuracy.png"
    plt.savefig(acc_fig_path, dpi=150)
    # plt.show()
    print("Saved validation accuracy plot:", acc_fig_path)





if __name__ == "__main__":
    main()




