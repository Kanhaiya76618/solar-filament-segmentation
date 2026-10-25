"""U-Net training on native-resolution crops with Dice + BCE against the soft annotator target."""
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.config import fold_tag, repo_path, resolve_device
from src.data.cache import load_disks
from src.data.unet_dataset import CropDataset, worker_init
from src.models.unet import build_unet, save_unet


def dice_bce(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    logits = logits.float()
    bce = F.binary_cross_entropy_with_logits(logits, target)
    p = torch.sigmoid(logits)
    dice = 1 - (2 * (p * target).sum() + 1) / (p.sum() + target.sum() + 1)
    return bce + dice


def unet_weights_path(cfg: dict, fold: int) -> Path:
    return repo_path(cfg["weights_dir"]) / f"unet_{cfg['unet']['encoder']}_{fold_tag(fold)}.pt"


def train_unet(cfg: dict, fold: int) -> Path:
    """Trains on every fold except ``fold`` (all data when fold < 0); saves the last epoch."""
    c, cache = cfg["unet"], Path(cfg["data"]["cache_dir"])
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    folds = pd.read_csv(cache / "folds.csv")
    ids = folds["image_id"] if fold < 0 else folds.loc[folds["fold"] != fold, "image_id"]
    device = resolve_device(cfg["device"])
    cuda = device.startswith("cuda")

    model = build_unet(c["encoder"], c["encoder_weights"]).to(device)
    if cuda and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    dataset = CropDataset(cache, ids.tolist(), load_disks(cache), c["crop"], c["samples_per_epoch"],
                          c["positive_fraction"], augment=True, seed=cfg["seed"])
    loader = DataLoader(dataset, batch_size=c["batch_size"], num_workers=c["num_workers"],
                        worker_init_fn=worker_init if c["num_workers"] > 0 else None,
                        pin_memory=cuda, drop_last=True)
    opt = torch.optim.AdamW(model.parameters(), lr=c["lr"], weight_decay=c["weight_decay"])
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=c["lr"], total_steps=c["epochs"] * len(loader),
                                                pct_start=0.05)
    amp = bool(c["amp"]) and cuda
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    print(f"[unet] {fold_tag(fold)}: {len(ids)} images, {c['epochs']} epochs x {len(loader)} steps on {device}")
    start = time.time()
    for epoch in range(c["epochs"]):
        model.train()
        losses = []
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
                loss = dice_bce(model(x), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            losses.append(loss.item())
        print(f"[unet] epoch {epoch + 1}/{c['epochs']} loss {np.mean(losses):.4f} "
              f"lr {sched.get_last_lr()[0]:.2e} {time.time() - start:.0f}s", flush=True)

    path = unet_weights_path(cfg, fold)
    save_unet(model, path, c["encoder"])
    print(f"[unet] saved {path}")
    return path
