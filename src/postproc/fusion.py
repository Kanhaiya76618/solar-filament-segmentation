"""
Turns a U-Net probability map and detector masks into disjoint filament instances.

The U-Net decides which pixels are filament (sharp, native-resolution boundaries); the detectors
decide which of those pixels belong together (one filament, even across faint gaps):

1. Threshold the probability map, close small gaps, fill holes, clip to the solar disk.
2. Visit detections from most to least confident. Each claims the still-unclaimed filament pixels
   inside its mask grown by ``claim_dilation`` px. Claims below ``min_area`` px are dropped.
3. Optionally keep large U-Net components that no detection claimed (``unclaimed_min_area``).
4. Optionally shrink or grow every instance by ``boundary_offset`` px, keeping them disjoint.

Instances are disjoint by construction, which the Kaggle scorer requires.
"""
from dataclasses import dataclass

import cv2
import numpy as np
from pycocotools import mask as mask_util
from scipy import ndimage


@dataclass
class Instance:
    score: float
    y0: int
    x0: int
    mask: np.ndarray  # bool crop
    source: str = "det"  # "det" (claimed by a detection) or "unet" (unclaimed U-Net component)

    @property
    def area(self) -> int:
        return int(self.mask.sum())

    def full(self, shape: tuple[int, int]) -> np.ndarray:
        out = np.zeros(shape, dtype=np.uint8)
        h, w = self.mask.shape
        out[self.y0:self.y0 + h, self.x0:self.x0 + w] = self.mask
        return out


def _disk(radius: int) -> np.ndarray:
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))


def semantic_mask(prob: np.ndarray, on_disk: np.ndarray, threshold: float, close_radius: int) -> np.ndarray:
    sem = (prob > threshold).astype(np.uint8)
    if close_radius > 0:
        sem = cv2.morphologyEx(sem, cv2.MORPH_CLOSE, _disk(close_radius))
    return ndimage.binary_fill_holes(sem) & on_disk


def fuse(prob: np.ndarray, on_disk: np.ndarray, detections: list, p: dict) -> list[Instance]:
    """Steps 1-3 above. ``detections`` is a list of (score, RLE dict) at native resolution."""
    sem = semantic_mask(prob, on_disk, p["unet_threshold"], p["close_radius"])
    h, w = sem.shape
    claimed = np.zeros_like(sem)
    d = int(p["claim_dilation"])
    grow = _disk(d) if d > 0 else None
    out: list[Instance] = []

    for score, rle in sorted(detections, key=lambda t: -t[0]):
        if score < p["det_threshold"]:
            break
        bx, by, bw, bh = mask_util.toBbox(rle)
        y0, x0 = max(0, int(by) - d - 1), max(0, int(bx) - d - 1)
        y1, x1 = min(h, int(np.ceil(by + bh)) + d + 1), min(w, int(np.ceil(bx + bw)) + d + 1)
        det = mask_util.decode(rle)[y0:y1, x0:x1]
        region = cv2.dilate(det, grow) > 0 if grow is not None else det > 0
        claim = region & sem[y0:y1, x0:x1] & ~claimed[y0:y1, x0:x1]
        if claim.sum() < max(p["min_area"], 1):
            continue
        claimed[y0:y1, x0:x1] |= claim
        out.append(Instance(score, y0, x0, claim, "det"))

    if p.get("unclaimed_min_area", 0) > 0:
        rest = (sem & ~claimed).astype(np.uint8)
        n, labels, stats, _ = cv2.connectedComponentsWithStats(rest, connectivity=8)
        for i in range(1, n):
            x0, y0, bw, bh, area = stats[i]
            if area < p["unclaimed_min_area"]:
                continue
            m = labels[y0:y0 + bh, x0:x0 + bw] == i
            mean_prob = float(prob[y0:y0 + bh, x0:x0 + bw][m].mean())
            if mean_prob >= p.get("unclaimed_min_prob", 0.0):
                out.append(Instance(mean_prob, int(y0), int(x0), m, "unet"))
    return out


def select(instances: list[Instance], p: dict) -> list[Instance]:
    """Applies the score/area thresholds of ``p`` to an instance list from a more permissive ``fuse``."""
    keep = []
    for inst in instances:
        if inst.source == "det":
            ok = inst.score >= p["det_threshold"] and inst.area >= p["min_area"]
        else:
            ok = (p["unclaimed_min_area"] > 0 and inst.area >= p["unclaimed_min_area"]
                  and inst.score >= p["unclaimed_min_prob"])
        if ok:
            keep.append(inst)
    return keep


def apply_boundary_offset(instances: list[Instance], offset: int, on_disk: np.ndarray) -> list[Instance]:
    """Erodes (offset < 0) or dilates (offset > 0) every instance; dilation stays disjoint by score."""
    if offset == 0:
        return instances
    cross = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    if offset < 0:
        out = []
        for inst in instances:
            m = cv2.erode(inst.mask.astype(np.uint8), cross, iterations=-offset) > 0
            if m.any():
                out.append(Instance(inst.score, inst.y0, inst.x0, m, inst.source))
        return out

    h, w = on_disk.shape
    claimed = np.zeros((h, w), dtype=bool)
    out = []
    for inst in sorted(instances, key=lambda i: -i.score):
        mh, mw = inst.mask.shape
        y0, x0 = max(0, inst.y0 - offset), max(0, inst.x0 - offset)
        y1, x1 = min(h, inst.y0 + mh + offset), min(w, inst.x0 + mw + offset)
        m = np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)
        m[inst.y0 - y0:inst.y0 - y0 + mh, inst.x0 - x0:inst.x0 - x0 + mw] = inst.mask
        m = (cv2.dilate(m, cross, iterations=offset) > 0) & on_disk[y0:y1, x0:x1] & ~claimed[y0:y1, x0:x1]
        claimed[y0:y1, x0:x1] |= m
        out.append(Instance(inst.score, y0, x0, m, inst.source))
    return out


def postprocess(prob: np.ndarray, on_disk: np.ndarray, detections: list, p: dict) -> list[Instance]:
    """The exact pipeline used for submissions: fuse, then threshold, then boundary offset."""
    instances = select(fuse(prob, on_disk, detections, p), p)
    return apply_boundary_offset(instances, int(p.get("boundary_offset", 0)), on_disk)
