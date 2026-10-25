"""
Loads the MAGFiLO COCO-style training JSON into the submission-style ground-truth table.

Polygons are rasterized with pycocotools (``frPyObjects`` + ``merge``), the same way the
organizers derive ``area`` and, presumably, the hidden test masks. Use this rasterization for
training targets as well: filaments are only a few pixels wide, so a half-pixel convention
change (e.g. switching to cv2.fillPoly) visibly lowers IoU.
"""
import json
from pathlib import Path

import pandas as pd
from pycocotools import mask as mask_util


def _flat_polygon(segmentation: list) -> list[float]:
    """The single polygon of an annotation as a flat [x0, y0, x1, y1, ...] list."""
    if segmentation and isinstance(segmentation[0], list):
        return max(segmentation, key=len)
    return segmentation


def polygon_to_rle(segmentation: list, height: int, width: int) -> str:
    if segmentation and not isinstance(segmentation[0], list):  # a single flat [x0, y0, ...] path
        segmentation = [segmentation]
    rles = mask_util.frPyObjects(segmentation, height, width)
    return mask_util.merge(rles)["counts"].decode("ascii")


def load_annotations(json_path: str | Path) -> pd.DataFrame:
    """One row per annotated filament.

    Columns: filament_id (``<annotator>-<image>_<n>``, the format the evaluator expects),
    segmentation_rle, annotator_image, annotator, image_id (physical image, e.g.
    ``20160920230134Lh``), file_name, polygon (flat [x0, y0, ...] list), area, bbox, spine,
    category_id.
    """
    coco = json.loads(Path(json_path).read_text())
    images = {str(im["id"]): im for im in coco["images"]}

    rows, counter = [], {}
    for ann in coco["annotations"]:
        if not ann.get("segmentation"):
            continue
        im = images[str(ann["image_id"])]
        image_id = Path(im["file_name"]).stem  # the file may be .jpeg while the JSON says .jpg
        annotator = str(im["id"]).split("-", maxsplit=1)[0] if "-" in str(im["id"]) else "000000"
        annotator_image = f"{annotator}-{image_id}"
        counter[annotator_image] = counter.get(annotator_image, 0) + 1
        rows.append({
            "filament_id": f"{annotator_image}_{counter[annotator_image]}",
            "segmentation_rle": polygon_to_rle(ann["segmentation"], im["height"], im["width"]),
            "annotator_image": annotator_image,
            "annotator": annotator,
            "image_id": image_id,
            "file_name": im["file_name"],
            "polygon": _flat_polygon(ann["segmentation"]),
            "area": ann.get("area"),
            "bbox": ann.get("bbox"),
            "spine": ann.get("spine"),
            "category_id": ann.get("category_id"),
        })
    return pd.DataFrame(rows)
