"""
Builds the on-disk training cache from the competition data.

Layout under ``data.cache_dir``:

    gt.pkl                 ground truth table from the annotation JSON
    folds.csv              image_id, n_records, group, fold
    disks.csv              image_id, cx, cy, r (fitted solar disk)
    corr/<image>.npy       limb-corrected uint8 image (memory-mapped by the U-Net loader)
    target/<image>.npy     uint8 mean of the annotator masks * 255 (soft U-Net target)
    pos/<image>.npy        flat indices of filament pixels (for sampling filament-centred crops)
    png/<image>.png        corrected image for YOLO
    yolo/images|labels     one YOLO sample per annotator record (images are symlinks)
"""
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import yaml
from pycocotools import mask as mask_util

from src.data.annotations import load_annotations
from src.data.paths import DataPaths, list_images
from src.data.preprocess import Disk, preprocess, read_gray
from src.data.splits import make_folds

MAX_POSITIVES = 50_000


def _cache_image(job: tuple) -> tuple[str, float, float, float]:
    image_id, src, records, cache, bin_width, corr_max, seed = job
    corr, disk = preprocess(read_gray(src), bin_width, corr_max)
    np.save(cache / "corr" / f"{image_id}.npy", corr)
    cv2.imwrite(str(cache / "png" / f"{image_id}.png"), corr)

    h, w = corr.shape
    target = np.zeros((h, w), dtype=np.float32)
    for rles in records:  # one entry per annotator record: that annotator's filaments
        union = np.zeros((h, w), dtype=bool)
        for counts in rles:
            union |= mask_util.decode({"size": [h, w], "counts": counts}).astype(bool)
        target += union
    target /= max(len(records), 1)
    np.save(cache / "target" / f"{image_id}.npy", np.round(target * 255).astype(np.uint8))

    pos = np.flatnonzero(target > 0)
    if len(pos) > MAX_POSITIVES:
        pos = np.random.default_rng(seed).choice(pos, MAX_POSITIVES, replace=False)
    np.save(cache / "pos" / f"{image_id}.npy", pos.astype(np.int64))
    return image_id, disk.cx, disk.cy, disk.r


def build_train_cache(cfg: dict, paths: DataPaths) -> Path:
    cache = Path(cfg["data"]["cache_dir"])
    for sub in ["corr", "target", "pos", "png"]:
        (cache / sub).mkdir(parents=True, exist_ok=True)

    gt = load_annotations(paths.train_json)
    gt.to_pickle(cache / "gt.pkl")
    folds = make_folds(gt, cfg["data"]["n_folds"], cfg["data"]["fold_gap_days"], cfg["seed"])
    folds.to_csv(cache / "folds.csv", index=False)

    images = list_images(paths.train_images)
    missing = sorted(set(gt["image_id"]) - set(images))
    if missing:
        raise FileNotFoundError(f"{len(missing)} annotated images not found, e.g. {missing[:3]}")

    disks_csv = cache / "disks.csv"
    done = pd.read_csv(disks_csv) if disks_csv.exists() else pd.DataFrame(columns=["image_id", "cx", "cy", "r"])
    todo = [i for i in sorted(gt["image_id"].unique()) if i not in set(done["image_id"])
            or not (cache / "pos" / f"{i}.npy").exists()]
    p = cfg["preprocess"]
    records = {i: [g["segmentation_rle"].tolist() for _, g in rows.groupby("annotator_image")]
               for i, rows in gt.groupby("image_id")}
    jobs = [(i, images[i], records[i], cache, p["limb_bin"], p["corr_max"], cfg["seed"]) for i in todo]
    print(f"[cache] {len(done)} images cached, preprocessing {len(jobs)}")
    with ProcessPoolExecutor(max_workers=max(1, cfg["data"]["workers"])) as pool:
        rows = list(pool.map(_cache_image, jobs, chunksize=4))
    disks = pd.concat([done, pd.DataFrame(rows, columns=["image_id", "cx", "cy", "r"])])
    disks.drop_duplicates("image_id", keep="last").to_csv(disks_csv, index=False)
    return cache


def load_disks(cache: Path) -> dict[str, Disk]:
    df = pd.read_csv(cache / "disks.csv")
    return {r.image_id: Disk(r.cx, r.cy, r.r) for r in df.itertuples()}


def export_yolo(cache: Path, fold: int) -> Path:
    """YOLO segmentation dataset with one sample per annotator record; returns the data YAML.

    Every record is its own sample, so the detector sees each annotator's notion of where one
    filament ends. Fold ``fold`` is the validation list (with ``fold < 0`` it is only a monitor).
    """
    gt = pd.read_pickle(cache / "gt.pkl")
    folds = pd.read_csv(cache / "folds.csv")
    img_dir, lbl_dir = cache / "yolo" / "images", cache / "yolo" / "labels"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    for record, rows in gt.groupby("annotator_image"):
        link = img_dir / f"{record}.png"
        if not link.exists():
            os.symlink(cache / "png" / f"{rows['image_id'].iloc[0]}.png", link)
        lines = []
        for poly in rows["polygon"]:
            xy = np.clip(np.asarray(poly, dtype=np.float64).reshape(-1, 2) / 2048.0, 0.0, 1.0)
            if len(xy) >= 3:
                lines.append("0 " + " ".join(f"{v:.6f}" for v in xy.ravel()))
        (lbl_dir / f"{record}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))

    record_fold = gt.drop_duplicates("annotator_image").merge(folds[["image_id", "fold"]], on="image_id")
    val_fold = fold if fold >= 0 else 0
    train = record_fold if fold < 0 else record_fold[record_fold["fold"] != fold]
    val = record_fold[record_fold["fold"] == val_fold]
    tag = "all" if fold < 0 else f"fold{fold}"
    lists = {}
    for name, part in [("train", train), ("val", val)]:
        lists[name] = cache / "yolo" / f"{name}_{tag}.txt"
        lists[name].write_text("".join(f"{img_dir / r}.png\n" for r in part["annotator_image"]))
    data_yaml = cache / "yolo" / f"data_{tag}.yaml"
    data_yaml.write_text(yaml.safe_dump({"path": str(cache / "yolo"), "train": str(lists["train"]),
                                         "val": str(lists["val"]), "names": {0: "filament"}}))
    return data_yaml
