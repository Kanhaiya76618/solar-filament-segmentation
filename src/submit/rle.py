"""
RLE encoding, overlap removal and submission checks for the Kaggle CSV.

The scorer rejects a submission (status ERROR) when two predicted filaments of one image share
a pixel, so every submission goes through ``paint_disjoint`` and ``check_submission``.
"""
import csv
import re
from pathlib import Path

import numpy as np
import pandas as pd
from pycocotools import mask as mask_util

HEIGHT = WIDTH = 2048
IMAGE_ID = re.compile(r"^\d{14}[A-Z]h$")


def encode_mask(mask: np.ndarray, verify: bool = True) -> str:
    """Binary (H, W) mask -> COCO RLE counts string, as the submission expects.

    pycocotools reads the array column-major. A C-ordered array still encodes, but decodes
    transposed and scores near zero, so the array is made Fortran-contiguous first and, by
    default, decoded again to confirm the round trip.
    """
    m = np.asfortranarray(mask.astype(np.uint8))
    counts = mask_util.encode(m)["counts"].decode("ascii")
    if verify and not np.array_equal(decode_rle(counts, *m.shape), m):
        raise RuntimeError("RLE round trip failed")
    return counts


def decode_rle(counts: str, height: int = HEIGHT, width: int = WIDTH) -> np.ndarray:
    return mask_util.decode({"size": [height, width], "counts": counts})


def paint_disjoint(masks: list[np.ndarray], scores: list[float], min_area: int = 0) -> list[int]:
    """Makes masks pixel-disjoint in place, highest score first; returns indices of kept masks.

    Each mask keeps only the pixels no higher-scoring mask has claimed. Masks left with fewer
    than ``min_area`` pixels are dropped, since such remnants are almost always false positives.
    """
    claimed = np.zeros(masks[0].shape, dtype=bool) if masks else None
    kept = []
    for i in np.argsort(-np.asarray(scores, dtype=float), kind="stable"):
        m = masks[i].astype(bool) & ~claimed
        if m.sum() < max(min_area, 1):
            continue
        masks[i] = m
        claimed |= m
        kept.append(int(i))
    return kept


def write_submission(rows: list[tuple[str, str]], path: str | Path) -> None:
    """Writes (filament_id, rle_counts) rows. Fails instead of quoting if a field needs quotes."""
    with open(path, "w", newline="") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_NONE)
        writer.writerow(["filament_id", "segmentation_rle"])
        writer.writerows(rows)


def check_submission(sub: str | Path | pd.DataFrame, image_ids: set[str] | None = None) -> list[str]:
    """Returns a list of problems (empty when the submission is valid).

    Checks the columns, unique ids, the ``<image>_<suffix>`` id format, known image ids, that
    every RLE decodes to a non-empty 2048x2048 mask, and that masks of one image are disjoint.
    """
    df = sub if isinstance(sub, pd.DataFrame) else pd.read_csv(sub, dtype=str, keep_default_na=False)
    if list(df.columns) != ["filament_id", "segmentation_rle"]:
        return [f"columns must be filament_id, segmentation_rle; got {list(df.columns)}"]

    problems = []
    if df["filament_id"].duplicated().any():
        problems.append(f"{df['filament_id'].duplicated().sum()} duplicated filament_id values")

    stems = df["filament_id"].str.split("_", n=1).str[0]
    bad = df.loc[~stems.str.match(IMAGE_ID) | ~df["filament_id"].str.contains("_"), "filament_id"]
    problems += [f"malformed filament_id: {fid}" for fid in bad.head(5)]
    if image_ids is not None:
        problems += [f"unknown image id: {s}" for s in sorted(set(stems) - set(image_ids))[:5]]

    for stem, rows in df.groupby(stems, sort=False):
        occupied = np.zeros((HEIGHT, WIDTH), dtype=np.uint16)
        for fid, counts in zip(rows["filament_id"], rows["segmentation_rle"]):
            try:
                m = decode_rle(counts)
            except Exception as e:  # pycocotools raises plain Exceptions on bad strings
                problems.append(f"{fid}: RLE does not decode ({e})")
                continue
            if m.shape != (HEIGHT, WIDTH) or not m.any():
                problems.append(f"{fid}: empty or wrongly sized mask")
            occupied += m
        if (occupied > 1).any():
            problems.append(f"{stem}: predicted filaments overlap ({int((occupied > 1).sum())} px)")
    return problems
