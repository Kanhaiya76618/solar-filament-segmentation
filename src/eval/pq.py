"""
Panoptic Quality evaluator that reproduces the organizers' self-evaluation notebook.

It follows the same rules as the notebook (kept verbatim in tests/reference_official.py):

- Ground-truth ids look like ``<annotator>-<image>_<n>``; prediction ids look like ``<image>_<n>``.
- Every annotator record is scored on its own against all predictions for its physical image.
- A pair is a true positive when IoU > 0.5 (strictly). TP, FP and FN are pooled over all records.
- Predictions for an image that has no ground-truth record are ignored.

Instead of decoding every mask to a 2048x2048 layer, it works on the RLE strings with
pycocotools, which is fast enough to call inside threshold sweeps. IoU and Dice are computed
in float32 from exact integer areas, as in the notebook, so the scores agree bit for bit.
"""
from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
import pandas as pd
from pycocotools import mask as mask_util

HEIGHT = WIDTH = 2048
IOU_THRESHOLD = 0.5


@dataclass
class PQResult:
    pq: float
    sq: float
    rq: float
    sum_iou: float
    tp: int
    fp: int
    fn: int
    records: pd.DataFrame          # one row per annotator record: n_gt, n_pred, tp, fp, fn, sum_iou
    pair_ious: np.ndarray          # every GT/prediction pair with IoU > 0 (notebook IoU histogram)
    pair_dices: np.ndarray         # same pairs, Dice (notebook Dice histogram)
    gt_match_counts: list[int]     # per GT filament: predictions touching it (1:n plot)
    pred_match_counts: list[int]   # per prediction and record: GT filaments touching it (n:1 plot)

    def summary(self) -> str:
        splits = sum(c > 1 for c in self.gt_match_counts)
        merges = sum(c > 1 for c in self.pred_match_counts)
        return (f"PQ {self.pq:.4f} = SQ {self.sq:.4f} x RQ {self.rq:.4f} | "
                f"TP {self.tp}  FP {self.fp}  FN {self.fn} | "
                f"GT touched by >1 pred: {splits}  preds touching >1 GT: {merges}")


def _rle_objs(rles: list[str], height: int = HEIGHT, width: int = WIDTH) -> list[dict]:
    return [{"size": [height, width], "counts": r} for r in rles]


def overlap_matrices(gt_rles: list[str], pred_rles: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """(n_gt, n_pred) float32 IoU and Dice matrices, equal to the notebook's get_overlap_matrices."""
    n_gt, n_pred = len(gt_rles), len(pred_rles)
    iou = np.zeros((n_gt, n_pred), dtype=np.float32)
    dice = np.zeros((n_gt, n_pred), dtype=np.float32)
    if n_gt == 0 or n_pred == 0:
        return iou, dice

    gt, pred = _rle_objs(gt_rles), _rle_objs(pred_rles)
    gt_area = mask_util.area(gt).astype(np.int64)
    pred_area = mask_util.area(pred).astype(np.int64)
    # Double-precision IoU from pycocotools is only used to find the overlapping pairs.
    touching = np.asarray(mask_util.iou(pred, gt, [0] * n_gt)).reshape(n_pred, n_gt).T > 0

    for i, j in zip(*np.nonzero(touching)):
        inter = int(mask_util.area(mask_util.merge([gt[i], pred[j]], intersect=True)))
        union = gt_area[i] + pred_area[j] - inter
        iou[i, j] = np.float32(inter) / np.float32(union)
        dice[i, j] = np.float32(2 * inter) / np.float32(gt_area[i] + pred_area[j])
    return iou, dice


def _iter_records(gt_df: pd.DataFrame, pred_df: pd.DataFrame) -> Iterator[tuple]:
    """Yields (annotator_image, image_id, gt_ids, pred_ids, iou, dice) per annotator record."""
    gt_keys = gt_df["filament_id"].str.split("_", n=1).str[0]
    gt_groups = gt_df.groupby(gt_keys, sort=False)

    pred_groups: dict[str, pd.DataFrame] = {}
    if len(pred_df):
        pred_keys = pred_df["filament_id"].str.split("_", n=1).str[0]
        pred_groups = dict(tuple(pred_df.groupby(pred_keys, sort=False)))
    no_preds = pred_df.iloc[0:0]

    for annotator_image, gt_rows in gt_groups:
        image_id = annotator_image.split("-", maxsplit=1)[1]
        pred_rows = pred_groups.get(image_id, no_preds)
        iou, dice = overlap_matrices(gt_rows["segmentation_rle"].tolist(),
                                     pred_rows["segmentation_rle"].tolist())
        yield (annotator_image, image_id, gt_rows["filament_id"].tolist(),
               pred_rows["filament_id"].tolist(), iou, dice)


def evaluate(gt_df: pd.DataFrame, pred_df: pd.DataFrame) -> PQResult:
    """Scores predictions against every annotator record, exactly like the official notebook.

    Both frames have the submission columns ``filament_id`` and ``segmentation_rle``.
    """
    tp_iou_scores: list[float] = []
    fp = fn = 0
    records, pair_ious, pair_dices = [], [], []
    gt_match_counts: list[int] = []
    pred_match_counts: list[int] = []

    for annotator_image, image_id, gt_ids, pred_ids, iou, dice in _iter_records(gt_df, pred_df):
        n_gt, n_pred = len(gt_ids), len(pred_ids)
        if n_pred == 0:
            rec_tp, rec_fp, rec_fn, rec_iou = 0, 0, n_gt, 0.0
            gt_match_counts.extend([0] * n_gt)
        else:
            hit = iou > IOU_THRESHOLD
            hits = iou[hit].tolist()
            tp_iou_scores.extend(hits)
            rec_tp, rec_iou = len(hits), sum(hits)
            rec_fp = int((hit.sum(axis=0) == 0).sum())
            rec_fn = int((hit.sum(axis=1) == 0).sum())

            touching = iou > 0
            gt_match_counts.extend(touching.sum(axis=1).tolist())
            pred_match_counts.extend(touching.sum(axis=0).tolist())
            pair_ious.append(iou[touching])
            pair_dices.append(dice[touching])

        fp += rec_fp
        fn += rec_fn
        records.append((annotator_image, image_id, n_gt, n_pred, rec_tp, rec_fp, rec_fn, rec_iou))

    tp = len(tp_iou_scores)
    sum_iou = sum(tp_iou_scores)
    denominator = tp + 0.5 * fp + 0.5 * fn
    return PQResult(
        pq=sum_iou / denominator if denominator > 0 else 0.0,
        sq=sum_iou / tp if tp else 0.0,
        rq=tp / denominator if denominator > 0 else 0.0,
        sum_iou=sum_iou, tp=tp, fp=fp, fn=fn,
        records=pd.DataFrame(records, columns=["annotator_image", "image_id", "n_gt", "n_pred",
                                               "tp", "fp", "fn", "sum_iou"]),
        pair_ious=np.concatenate(pair_ious) if pair_ious else np.zeros(0, np.float32),
        pair_dices=np.concatenate(pair_dices) if pair_dices else np.zeros(0, np.float32),
        gt_match_counts=gt_match_counts,
        pred_match_counts=pred_match_counts,
    )


def prediction_targets(gt_df: pd.DataFrame, pred_df: pd.DataFrame) -> pd.DataFrame:
    """Per-prediction training targets for an instance scorer.

    For each predicted filament, over all annotator records of its image:

    - ``match_rate``: fraction of records in which it is a true positive (IoU > 0.5);
    - ``expected_iou``: mean of IoU * [IoU > 0.5], i.e. what it adds to the PQ numerator per record;
    - ``best_iou``: mean of its best IoU with any GT filament, matched or not.

    Keeping a prediction always adds 0.5 per record to the PQ denominator, so it raises the
    score only when ``expected_iou > PQ / 2`` (see docs/PLAN.md, "PQ-optimal selection").
    """
    acc: dict[str, list[float]] = {}
    for _, _, _, pred_ids, iou, _ in _iter_records(gt_df, pred_df):
        best = iou.max(axis=0) if len(iou) else np.zeros(len(pred_ids), np.float32)
        for fid, b in zip(pred_ids, best.tolist()):
            row = acc.setdefault(fid, [0, 0.0, 0.0, 0.0])
            row[0] += 1
            row[1] += b > IOU_THRESHOLD
            row[2] += b if b > IOU_THRESHOLD else 0.0
            row[3] += b

    out = pd.DataFrame({"filament_id": pred_df["filament_id"]})
    stats = pd.DataFrame.from_dict(acc, orient="index",
                                   columns=["n_records", "n_match", "sum_matched_iou", "sum_best_iou"])
    out = out.join(stats, on="filament_id")
    n = out["n_records"]
    out["match_rate"] = out["n_match"] / n
    out["expected_iou"] = out["sum_matched_iou"] / n
    out["best_iou"] = out["sum_best_iou"] / n
    return out[["filament_id", "n_records", "match_rate", "expected_iou", "best_iou"]]
