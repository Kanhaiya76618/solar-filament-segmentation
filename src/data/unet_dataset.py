"""Random native-resolution crops for U-Net training, read from the memory-mapped cache."""
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from src.data.preprocess import Disk, model_input


class CropDataset(Dataset):
    """Each item is a random crop: (x: 2xCxC float32, y: 1xCxC soft target in [0, 1]).

    A share ``positive_fraction`` of crops is centred on a random filament pixel, the rest on a
    random point of the solar disk. Augmentations: the 8 flips/rotations of the square, plus
    gain, offset, gamma, Gaussian noise and blur on the intensity channel (on-disk pixels only).
    Geometric resampling is avoided because it blurs few-pixel-wide filaments.
    """

    def __init__(self, cache: Path, image_ids: list[str], disks: dict[str, Disk], crop: int = 512,
                 samples_per_epoch: int = 4000, positive_fraction: float = 0.7, augment: bool = True,
                 seed: int = 0):
        self.cache, self.ids, self.disks = Path(cache), list(image_ids), disks
        self.crop, self.n, self.pos_frac, self.augment = crop, samples_per_epoch, positive_fraction, augment
        self.rng = np.random.default_rng(seed)
        self._corr, self._target, self._pos = {}, {}, {}

    def __len__(self) -> int:
        return self.n

    def _arrays(self, image_id: str):
        if image_id not in self._corr:  # opened lazily, so each worker process has its own handles
            self._corr[image_id] = np.load(self.cache / "corr" / f"{image_id}.npy", mmap_mode="r")
            self._target[image_id] = np.load(self.cache / "target" / f"{image_id}.npy", mmap_mode="r")
            self._pos[image_id] = np.load(self.cache / "pos" / f"{image_id}.npy")
        return self._corr[image_id], self._target[image_id], self._pos[image_id]

    def __getitem__(self, index: int):
        rng = self.rng
        image_id = self.ids[rng.integers(len(self.ids))]
        corr, target, pos = self._arrays(image_id)
        disk = self.disks[image_id]
        h, w = corr.shape
        if len(pos) and rng.random() < self.pos_frac:
            cy, cx = divmod(int(pos[rng.integers(len(pos))]), w)
        else:
            a, rr = rng.uniform(0, 2 * np.pi), disk.r * np.sqrt(rng.uniform(0, 1))
            cy, cx = int(disk.cy + rr * np.sin(a)), int(disk.cx + rr * np.cos(a))
        c = self.crop
        y0 = int(np.clip(cy - c // 2 + rng.integers(-c // 4, c // 4 + 1), 0, h - c))
        x0 = int(np.clip(cx - c // 2 + rng.integers(-c // 4, c // 4 + 1), 0, w - c))

        x = model_input(np.asarray(corr[y0:y0 + c, x0:x0 + c]), disk, y0, x0)
        y = np.asarray(target[y0:y0 + c, x0:x0 + c], dtype=np.float32)[None] / 255.0
        if self.augment:
            x, y = self._augment(x, y, rng)
        return torch.from_numpy(np.ascontiguousarray(x)), torch.from_numpy(np.ascontiguousarray(y))

    @staticmethod
    def _augment(x: np.ndarray, y: np.ndarray, rng: np.random.Generator):
        k = int(rng.integers(4))
        x, y = np.rot90(x, k, axes=(1, 2)), np.rot90(y, k, axes=(1, 2))
        if rng.random() < 0.5:
            x, y = x[:, :, ::-1], y[:, :, ::-1]
        x = x.copy()
        on_disk = x[1] < 1.0 / 1.5  # radius channel is rho / 1.5
        img = x[0]
        img = np.clip(img * rng.uniform(0.85, 1.15) + rng.uniform(-0.05, 0.05), 0, 1) ** rng.uniform(0.8, 1.25)
        if rng.random() < 0.3:
            img = cv2.GaussianBlur(img.astype(np.float32), (0, 0), rng.uniform(0.5, 1.2))
        img = img + rng.normal(0, rng.uniform(0, 0.02), img.shape)
        x[0] = np.where(on_disk, np.clip(img, 0, 1), 0.0)
        return x, y


def worker_init(worker_id: int) -> None:
    """Gives each DataLoader worker its own random stream (derived from torch's seed)."""
    info = torch.utils.data.get_worker_info()
    info.dataset.rng = np.random.default_rng(torch.initial_seed() % 2 ** 32)
