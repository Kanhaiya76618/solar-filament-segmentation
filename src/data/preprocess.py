"""
Solar-disk fitting and limb-darkening correction for 8-bit GONG H-alpha JPEGs.

1. ``fit_disk`` finds a rough disk by thresholding, then refits the limb at the steepest radial
   intensity drop along 720 rays. A plain threshold overestimates the radius because GONG frames
   have a bright halo just outside the limb.
2. ``correct`` divides the image by its median intensity in thin concentric rings, so the quiet
   Sun is ~1 everywhere and filaments stand out as dips below 1, from disk centre to limb.
3. ``model_input`` stacks the corrected image and the radial distance (in solar radii) as the
   two network input channels.
"""
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class Disk:
    cx: float
    cy: float
    r: float


def read_gray(path: str | Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise OSError(f"cannot read image {path}")
    return img


def _fit_circle(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    """Algebraic least-squares circle fit, refitted twice without outliers (> 3 robust sigma)."""
    keep = np.ones(len(x), dtype=bool)
    for _ in range(3):
        a = np.c_[x[keep], y[keep], np.ones(keep.sum())]
        d, e, f = np.linalg.lstsq(a, -(x[keep] ** 2 + y[keep] ** 2), rcond=None)[0]
        cx, cy = -d / 2, -e / 2
        r = float(np.sqrt(max(cx ** 2 + cy ** 2 - f, 1.0)))
        resid = np.abs(np.hypot(x - cx, y - cy) - r)
        keep = resid < 3 * 1.4826 * np.median(resid[keep]) + 0.5
    return float(cx), float(cy), r


def fit_disk(img: np.ndarray, n_angles: int = 720) -> Disk:
    h, w = img.shape
    f = img.astype(np.float32)
    rough = (f > 0.25 * float(np.percentile(f, 99.5))).astype(np.uint8)
    n, _, stats, centroids = cv2.connectedComponentsWithStats(rough, connectivity=8)
    if n < 2:
        return Disk(w / 2, h / 2, 0.43 * min(h, w))
    k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    cx, cy = (float(v) for v in centroids[k])
    r = float(np.sqrt(stats[k, cv2.CC_STAT_AREA] / np.pi))

    smooth = cv2.GaussianBlur(f, (0, 0), 2.0)
    theta = np.linspace(0, 2 * np.pi, n_angles, endpoint=False)
    cos, sin = np.cos(theta)[:, None], np.sin(theta)[:, None]
    for _ in range(2):
        radii = np.arange(0.85 * r, 1.15 * r, 0.5, dtype=np.float32)[None, :]
        rays = cv2.remap(smooth, (cx + cos * radii).astype(np.float32), (cy + sin * radii).astype(np.float32),
                         cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        edge = radii[0, np.argmin(np.diff(rays, axis=1), axis=1)] + 0.25  # midpoint of the steepest drop
        cx, cy, r = _fit_circle(cx + cos[:, 0] * edge, cy + sin[:, 0] * edge)
    return Disk(cx, cy, r)


def radius_map(disk: Disk, shape: tuple[int, int], y0: int = 0, x0: int = 0) -> np.ndarray:
    """Distance from disk centre in solar radii for the window starting at (y0, x0)."""
    ys = np.arange(y0, y0 + shape[0], dtype=np.float32)[:, None]
    xs = np.arange(x0, x0 + shape[1], dtype=np.float32)[None, :]
    return np.sqrt((xs - disk.cx) ** 2 + (ys - disk.cy) ** 2) / disk.r


def limb_profile(img: np.ndarray, disk: Disk, bin_width: float = 0.005, step: int = 2):
    """Median intensity in rings of ``bin_width`` solar radii (on a subsampled grid), lightly smoothed."""
    sub = img[::step, ::step].astype(np.float32)
    rho = radius_map(Disk(disk.cx / step, disk.cy / step, disk.r / step), sub.shape)
    inside = rho < 1.0
    n_bins = int(np.ceil(1.0 / bin_width))
    bins = np.minimum((rho[inside] / bin_width).astype(np.int64), n_bins - 1)
    vals = sub[inside]
    order = np.argsort(bins, kind="stable")
    bins, vals = bins[order], vals[order]
    edges = np.searchsorted(bins, np.arange(n_bins + 1))
    prof = np.array([np.median(vals[a:b]) if b > a else np.nan for a, b in zip(edges[:-1], edges[1:])])
    centers = (np.arange(n_bins) + 0.5) * bin_width
    ok = np.isfinite(prof)
    prof = np.interp(centers, centers[ok], prof[ok])
    prof = np.convolve(np.pad(prof, 1, mode="edge"), np.ones(3) / 3, mode="valid")
    return centers, prof


def correct(img: np.ndarray, disk: Disk, bin_width: float = 0.005, corr_max: float = 1.5) -> np.ndarray:
    """Limb-darkening-corrected image as uint8: 255 * clip(I / ring_median / corr_max, 0, 1), 0 off disk."""
    centers, prof = limb_profile(img, disk, bin_width)
    rho = radius_map(disk, img.shape)
    expected = np.interp(rho, centers, prof).astype(np.float32)
    ratio = img.astype(np.float32) / np.maximum(expected, 1.0)
    out = np.clip(ratio / corr_max, 0.0, 1.0) * 255.0
    out[rho >= 1.0] = 0
    return out.astype(np.uint8)


def preprocess(img: np.ndarray, bin_width: float = 0.005, corr_max: float = 1.5) -> tuple[np.ndarray, Disk]:
    disk = fit_disk(img)
    return correct(img, disk, bin_width, corr_max), disk


def model_input(corr: np.ndarray, disk: Disk, y0: int = 0, x0: int = 0) -> np.ndarray:
    """(2, h, w) float32 network input for a window of the corrected image starting at (y0, x0)."""
    rho = radius_map(disk, corr.shape, y0, x0)
    return np.stack([corr.astype(np.float32) / 255.0, np.clip(rho, 0.0, 1.5) / 1.5])


def disk_mask(disk: Disk, shape: tuple[int, int] = (2048, 2048)) -> np.ndarray:
    return radius_map(disk, shape) < 1.0
