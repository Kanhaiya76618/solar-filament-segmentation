"""YOLO11-seg detections mapped back to native 2048x2048 resolution and merged across detectors."""
import cv2
import numpy as np
from pycocotools import mask as mask_util

Detection = tuple[float, dict]  # (confidence, COCO RLE dict at native resolution)


def _to_native(masks: np.ndarray, orig_shape: tuple[int, int]) -> list[np.ndarray]:
    """Undoes Ultralytics' letterbox (scale + centred padding) for (N, h, w) masks."""
    _, h_in, w_in = masks.shape
    h, w = orig_shape
    gain = min(h_in / h, w_in / w)
    pad_w, pad_h = (w_in - w * gain) / 2, (h_in - h * gain) / 2
    top, left = int(round(pad_h - 0.1)), int(round(pad_w - 0.1))
    bottom, right = int(round(h_in - pad_h + 0.1)), int(round(w_in - pad_w + 0.1))
    return [cv2.resize(m[top:bottom, left:right], (w, h), interpolation=cv2.INTER_LINEAR) > 0.5
            for m in masks.astype(np.float32)]


def detect(model, corr: np.ndarray, imgsz: int, conf: float, iou: float, max_det: int, device) -> list[Detection]:
    """Runs one detector on the corrected image (fed as 3 identical channels, like training)."""
    image = np.repeat(corr[:, :, None], 3, axis=2)
    result = model.predict(image, imgsz=imgsz, conf=conf, iou=iou, max_det=max_det, device=device,
                           retina_masks=False, verbose=False)[0]
    if result.masks is None or len(result.boxes) == 0:
        return []
    scores = result.boxes.conf.cpu().numpy()
    masks = _to_native(result.masks.data.cpu().numpy(), result.orig_shape)
    return [(float(s), mask_util.encode(np.asfortranarray(m.astype(np.uint8))))
            for s, m in zip(scores, masks) if m.any()]


def merge_detectors(per_model: list[list[Detection]]) -> list[Detection]:
    """Merges detections of several detectors: masks with IoU > 0.5 are the same filament.

    The most confident mask of a cluster is kept, and its score is the mean confidence over all
    detectors, counting 0 for a detector that missed it. With one detector this is the identity.
    """
    if len(per_model) == 1:
        return sorted(per_model[0], key=lambda d: -d[0])
    clusters: list[dict] = []  # {"rle", "best", "sum", "models"}
    for m, dets in enumerate(per_model):
        for score, rle in sorted(dets, key=lambda d: -d[0]):
            best = None
            if clusters:
                ious = np.asarray(mask_util.iou([rle], [c["rle"] for c in clusters], [0] * len(clusters)))[0]
                for k in np.argsort(-ious):
                    if ious[k] <= 0.5:
                        break
                    if m not in clusters[k]["models"]:
                        best = k
                        break
            if best is None:
                clusters.append({"rle": rle, "best": score, "sum": score, "models": {m}})
            else:
                c = clusters[best]
                c["sum"] += score
                c["models"].add(m)
                if score > c["best"]:
                    c["rle"], c["best"] = rle, score
    n = len(per_model)
    return sorted(((c["sum"] / n, c["rle"]) for c in clusters), key=lambda d: -d[0])
