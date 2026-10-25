import json

import numpy as np
import pandas as pd
import pytest

from src.data.annotations import load_annotations
from src.eval.pq import evaluate, prediction_targets
from src.submit.rle import check_submission, decode_rle, encode_mask, paint_disjoint, write_submission
from tests import reference_official as official

H = W = 2048


def rect(y0, x0, h, w):
    m = np.zeros((H, W), dtype=np.uint8)
    m[y0:y0 + h, x0:x0 + w] = 1
    return m


def filament(rng, cy, cx, length, width, angle):
    """Thin rotated ellipse, a rough stand-in for a filament."""
    y, x = np.ogrid[:H, :W]
    c, s = np.cos(angle), np.sin(angle)
    u = (x - cx) * c + (y - cy) * s
    v = -(x - cx) * s + (y - cy) * c
    return ((u / length) ** 2 + (v / width) ** 2 <= 1).astype(np.uint8)


def frame(rows):
    return pd.DataFrame([(fid, encode_mask(m)) for fid, m in rows],
                        columns=["filament_id", "segmentation_rle"])


def official_scores(gt_df, pred_df):
    overlap = official.get_overlap_df(gt_df, pred_df)
    return official.get_pq_score(overlap), official.m2n_counts(overlap), overlap


def test_perfect_match():
    m = rect(100, 100, 10, 40)
    res = evaluate(frame([("010101-20150125172714Mh_1", m)]), frame([("20150125172714Mh_1", m)]))
    assert (res.pq, res.tp, res.fp, res.fn) == (1.0, 1, 0, 0)


def test_iou_exactly_half_is_not_a_match():
    gt = frame([("010101-20150125172714Mh_1", rect(100, 100, 10, 10))])
    res = evaluate(gt, frame([("20150125172714Mh_1", rect(100, 100, 10, 5))]))
    assert (res.tp, res.fp, res.fn, res.pq) == (0, 1, 1, 0.0)
    res = evaluate(gt, frame([("20150125172714Mh_1", rect(100, 100, 10, 6))]))
    assert res.tp == 1 and res.pq == pytest.approx(0.6 / 1.0)


def test_split_and_merge_show_up_in_diagnostics():
    split = evaluate(frame([("010101-20150125172714Mh_1", rect(100, 100, 10, 20))]),
                     frame([("20150125172714Mh_1", rect(100, 100, 10, 10)),
                            ("20150125172714Mh_2", rect(100, 110, 10, 10))]))
    assert (split.tp, split.fp, split.fn) == (0, 2, 1)
    assert split.gt_match_counts == [2]

    merge = evaluate(frame([("010101-20150125172714Mh_1", rect(100, 100, 10, 10)),
                            ("010101-20150125172714Mh_2", rect(100, 110, 10, 10))]),
                     frame([("20150125172714Mh_1", rect(100, 100, 10, 20))]))
    assert (merge.tp, merge.fp, merge.fn) == (0, 1, 2)
    assert merge.pred_match_counts == [2]


def test_each_annotator_record_is_scored_separately():
    a, b = rect(100, 100, 10, 40), rect(500, 500, 10, 40)
    gt = frame([("010101-20150125172714Mh_1", a),
                ("010102-20150125172714Mh_1", a), ("010102-20150125172714Mh_2", b)])
    pred = frame([("20150125172714Mh_1", a), ("20150125172714Mh_2", rect(900, 900, 8, 8))])
    res = evaluate(gt, pred)
    # record 010101: TP a, FP blob. record 010102: TP a, FN b, FP blob.
    assert (res.tp, res.fp, res.fn) == (2, 2, 1)
    assert res.pq == pytest.approx(2 / (2 + 0.5 * 2 + 0.5 * 1))

    targets = prediction_targets(gt, pred).set_index("filament_id")
    assert targets.loc["20150125172714Mh_1", "match_rate"] == 1.0
    assert targets.loc["20150125172714Mh_1", "expected_iou"] == 1.0
    assert targets.loc["20150125172714Mh_2", "expected_iou"] == 0.0
    assert targets.loc["20150125172714Mh_2", "n_records"] == 2


def test_matches_official_notebook_on_random_scenes():
    rng = np.random.default_rng(0)
    gt_rows, pred_rows = [], []
    images = [f"2015012517{i:02d}14Mh" for i in range(5)]
    for k, image in enumerate(images):
        shapes = [(rng.uniform(300, 1700), rng.uniform(300, 1700), rng.uniform(20, 120),
                   rng.uniform(2, 8), rng.uniform(0, np.pi)) for _ in range(rng.integers(2, 6))]
        for annotator in ["010101", "010102"][: 1 + k % 2]:
            for n, (cy, cx, ln, wd, ang) in enumerate(shapes, start=1):
                jitter = rng.normal(0, 1.5, 2) if annotator == "010102" else (0, 0)
                gt_rows.append((f"{annotator}-{image}_{n}",
                                filament(rng, cy + jitter[0], cx + jitter[1], ln, wd, ang)))
        if k == 4:
            continue  # an image without predictions: all its GT become false negatives
        n = 0
        for cy, cx, ln, wd, ang in shapes:
            n += 1
            # Perturb position and width so IoUs straddle the 0.5 threshold.
            pred_rows.append((f"{image}_{n}", filament(rng, cy + rng.normal(0, 3), cx + rng.normal(0, 3),
                                                       ln * rng.uniform(0.6, 1.2),
                                                       wd * rng.uniform(0.5, 1.6), ang)))
        n += 1
        pred_rows.append((f"{image}_{n}", filament(rng, 1900, 1900, 15, 4, 0.3)))  # stray FP
    pred_rows.append(("20990101000000Mh_1", rect(10, 10, 5, 5)))  # image with no GT is ignored

    gt_df, pred_df = frame(gt_rows), frame(pred_rows)
    ref_pq, (ref_gt_counts, ref_pred_counts), overlap = official_scores(gt_df, pred_df)
    res = evaluate(gt_df, pred_df)

    assert res.pq == ref_pq  # bit-for-bit
    assert res.gt_match_counts == ref_gt_counts
    assert res.pred_match_counts == ref_pred_counts
    ref_ious = np.concatenate([m.numpy().ravel() for m in overlap["iou_matrix"]])
    np.testing.assert_array_equal(np.sort(res.pair_ious), np.sort(ref_ious[ref_ious > 0]))
    assert 0 < res.tp and 0 < res.fp and 0 < res.fn  # the scene exercises all three cases


def test_rle_round_trip_catches_transposition():
    m = rect(10, 300, 5, 900)  # asymmetric: transposing it changes the mask
    c_ordered = np.ascontiguousarray(m)
    counts = encode_mask(c_ordered)
    assert np.array_equal(decode_rle(counts), m)
    assert not np.array_equal(decode_rle(counts).T, m)


def test_paint_disjoint_and_submission_check(tmp_path):
    masks = [rect(100, 100, 10, 50), rect(100, 120, 10, 50), rect(100, 130, 2, 2)]
    kept = paint_disjoint(masks, scores=[0.9, 0.8, 0.1], min_area=20)
    assert kept == [0, 1]  # the low-score mask lost all its pixels and is dropped
    assert not (masks[0].astype(bool) & masks[1].astype(bool)).any()

    path = tmp_path / "submission.csv"
    write_submission([(f"20150125172714Mh_{i + 1}", encode_mask(masks[i])) for i in kept], path)
    assert check_submission(path, image_ids={"20150125172714Mh"}) == []

    overlapping = frame([("20150125172714Mh_1", rect(0, 0, 10, 10)),
                         ("20150125172714Mh_2", rect(5, 5, 10, 10)),
                         ("bad-id_1", rect(50, 50, 3, 3))])
    problems = check_submission(overlapping)
    assert any("overlap" in p for p in problems)
    assert any("malformed" in p for p in problems)


def test_load_annotations_rasterizes_polygons_with_pycocotools(tmp_path):
    square = [100.0, 100.0, 140.0, 100.0, 140.0, 110.0, 100.0, 110.0, 100.0, 100.0]
    coco = {
        "images": [{"id": "010101-20150125172714Mh", "width": W, "height": H,
                    "file_name": "20150125172714Mh.jpeg"}],
        "annotations": [{"id": "a", "image_id": "010101-20150125172714Mh", "category_id": 1,
                         "segmentation": [square], "area": 400.0, "spine": [], "iscrowd": 0}],
    }
    path = tmp_path / "train.json"
    path.write_text(json.dumps(coco))
    gt = load_annotations(path)
    assert gt.loc[0, "filament_id"] == "010101-20150125172714Mh_1"
    assert gt.loc[0, "image_id"] == "20150125172714Mh"
    assert decode_rle(gt.loc[0, "segmentation_rle"]).sum() == 400

    pred = gt[["filament_id", "segmentation_rle"]].assign(
        filament_id=lambda d: d["filament_id"].str.split("-", n=1).str[1])
    assert evaluate(gt[["filament_id", "segmentation_rle"]], pred).pq == 1.0
