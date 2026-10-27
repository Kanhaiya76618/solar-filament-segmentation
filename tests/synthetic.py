"""
Synthetic GONG-like data for tests and smoke runs: limb-darkened full-disk images with a bright
halo, thin curved dark filaments (annotated), round dark sunspots (not annotated) and noise,
written in the competition's folder layout with a COCO-style annotation JSON.
"""
import json
from datetime import datetime, timedelta
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from src.data.annotations import polygon_to_rle

SIZE = 2048


def sun_image(rng: np.random.Generator, filaments: list[np.ndarray], cx: float, cy: float, r: float,
              n_spots: int = 2, halo: tuple[float, float, float] | None = None) -> np.ndarray:
    """``halo = (dx, dy, radius)``: a bright scattered-light annulus centred at (cx+dx, cy+dy)
    that ends in a sharp edge, as in many real GONG frames."""
    ys, xs = np.mgrid[:SIZE, :SIZE].astype(np.float32)
    d = np.hypot(xs - cx, ys - cy)
    rho = np.clip(d / r, 0, 1)
    outside = 35 * np.exp(-(d - r) / 25)
    if halo is not None:
        outside = np.where(np.hypot(xs - cx - halo[0], ys - cy - halo[1]) < halo[2], 55.0, outside)
    img = np.where(d <= r, 170 * (1 - 0.5 * (1 - np.sqrt(1 - rho ** 2))), outside)
    for m in filaments:
        img[m] *= 0.65
    for _ in range(n_spots):
        a, rr = rng.uniform(0, 2 * np.pi), r * 0.6 * np.sqrt(rng.uniform())
        spot = np.hypot(xs - (cx + rr * np.cos(a)), ys - (cy + rr * np.sin(a))) < rng.uniform(6, 14)
        img[spot] *= 0.4
    img = cv2.GaussianBlur(img.astype(np.float32), (0, 0), 1.0) + rng.normal(0, 3, img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)


def random_filament(rng: np.random.Generator, cx: float, cy: float, r: float) -> np.ndarray:
    """A thin curved band (quadratic Bezier, 3-8 px wide, 80-300 px long) inside the disk."""
    a, rr = rng.uniform(0, 2 * np.pi), r * 0.8 * np.sqrt(rng.uniform())
    p0 = np.array([cx + rr * np.cos(a), cy + rr * np.sin(a)])
    heading, length = rng.uniform(0, np.pi), rng.uniform(80, 300)
    p2 = p0 + length * np.array([np.cos(heading), np.sin(heading)])
    p1 = (p0 + p2) / 2 + rng.normal(0, length / 4, 2)
    t = np.linspace(0, 1, 60)[:, None]
    curve = ((1 - t) ** 2 * p0 + 2 * (1 - t) * t * p1 + t ** 2 * p2).astype(np.int32)
    canvas = np.zeros((SIZE, SIZE), dtype=np.uint8)
    cv2.polylines(canvas, [curve], False, 1, thickness=int(rng.integers(3, 9)))
    return canvas.astype(bool)


def polygon(mask: np.ndarray) -> list[float]:
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    pts = max(contours, key=len)[:, 0, :].astype(float)
    pts = np.vstack([pts, pts[:1]])  # closed path, as in MAGFiLO
    return pts.ravel().tolist()


def make_dataset(root: Path, n_train: int = 8, n_test: int = 3, seed: int = 0) -> pd.DataFrame:
    """Writes ``root/MAGFiLO_1.0_Kaggle_2026``; returns the test ground truth in evaluator format.

    Every third training image gets a second annotator who skips one filament and draws the
    others one pixel wider, to mimic annotator disagreement.
    """
    rng = np.random.default_rng(seed)
    base = root / "MAGFiLO_1.0_Kaggle_2026"
    (base / "train" / "train_images").mkdir(parents=True, exist_ok=True)
    (base / "test" / "test_images").mkdir(parents=True, exist_ok=True)
    images, annotations, test_rows = [], [], []
    start = datetime(2015, 1, 1, 12, 0, 0)

    for k in range(n_train + n_test):
        image_id = (start + timedelta(days=3 * k, minutes=int(rng.integers(0, 600)))).strftime("%Y%m%d%H%M%S") \
            + "MBLCTU"[k % 6] + "h"
        cx, cy, r = 1024 + rng.uniform(-60, 60), 1024 + rng.uniform(-60, 60), rng.uniform(895, 912)
        fils = [random_filament(rng, cx, cy, r) for _ in range(int(rng.integers(3, 8)))]
        fils = [f for i, f in enumerate(fils) if not any((f & g).any() for g in fils[:i])]  # keep disjoint
        is_test = k >= n_train
        split = "test" if is_test else "train"
        halo = (rng.uniform(-40, 40), rng.uniform(-40, 40), rng.uniform(980, 1060)) if k % 2 else None
        cv2.imwrite(str(base / split / f"{split}_images" / f"{image_id}.jpeg"),
                    sun_image(rng, fils, cx, cy, r, halo=halo), [cv2.IMWRITE_JPEG_QUALITY, 95])

        annotators = ["010101"] + (["010102"] if (not is_test and k % 3 == 0) else [])
        for annotator in annotators:
            record = f"{annotator}-{image_id}"
            drawn = fils
            if annotator == "010102":
                kernel = np.ones((3, 3), np.uint8)
                drawn = [cv2.dilate(f.astype(np.uint8), kernel).astype(bool) for f in fils[1:]]
            for n, f in enumerate(drawn, 1):
                poly = polygon(f)
                if is_test:
                    test_rows.append((f"{record}_{n}", polygon_to_rle([poly], SIZE, SIZE)))
                    continue
                ys, xs = np.nonzero(f)
                annotations.append({"id": f"{record}-{n}", "image_id": record, "category_id": 1,
                                    "segmentation": [poly], "area": float(f.sum()), "iscrowd": 0,
                                    "bbox": [float(xs.min()), float(ys.min()), float(np.ptp(xs) + 1),
                                             float(np.ptp(ys) + 1)], "spine": []})
            if not is_test:
                images.append({"id": record, "width": SIZE, "height": SIZE, "file_name": f"{image_id}.jpeg"})

    coco = {"info": {}, "licenses": [], "categories": [{"id": 1, "name": "Left", "supercategory": "filament"}],
            "images": images, "annotations": annotations}
    (base / "train" / "MAGFiLO_1.0_Annotations_kaggle2026_train.json").write_text(json.dumps(coco))
    return pd.DataFrame(test_rows, columns=["filament_id", "segmentation_rle"])


if __name__ == "__main__":
    import sys

    out = Path(sys.argv[1])
    gt = make_dataset(out, int(sys.argv[2]) if len(sys.argv) > 2 else 8, int(sys.argv[3]) if len(sys.argv) > 3 else 3)
    gt.to_csv(out / "test_gt.csv", index=False)
    print(f"wrote synthetic data to {out}; test ground truth in {out / 'test_gt.csv'}")
