import numpy as np
import pandas as pd
import pytest
from pycocotools import mask as mask_util

from src.config import load_config
from src.data.cache import build_train_cache, export_yolo, load_disks
from src.data.paths import find_data
from src.data.preprocess import correct, disk_mask, fit_disk, radius_map
from src.data.splits import make_folds
from src.infer.yolo import _to_native, merge_detectors
from src.postproc.fusion import apply_boundary_offset, fuse, postprocess, select
from tests.synthetic import random_filament, sun_image

PARAMS = dict(unet_threshold=0.5, close_radius=0, det_threshold=0.3, claim_dilation=2, min_area=10,
              boundary_offset=0, unclaimed_min_area=0, unclaimed_min_prob=0.0)


def rle(mask):
    return mask_util.encode(np.asfortranarray(mask.astype(np.uint8)))


def band(y0, x0, h, w, shape=(2048, 2048)):
    m = np.zeros(shape, dtype=bool)
    m[y0:y0 + h, x0:x0 + w] = True
    return m


@pytest.mark.parametrize("cx, cy, halo", [(1023.0, 1024.0, (0, 0, 1035)), (980.0, 1060.0, (-30, 25, 1000)),
                                           (1050.0, 990.0, None)])
def test_disk_fit_ignores_bright_halo(cx, cy, halo):
    """Regression: on frames with a bright scattered-light annulus, the old fit locked onto its
    outer edge (r ~ 960-1035 instead of ~904) for about 18% of the training images."""
    img = sun_image(np.random.default_rng(0), [], cx, cy, 904.0, halo=halo)
    disk = fit_disk(img)
    assert abs(disk.cx - cx) < 1.5 and abs(disk.cy - cy) < 1.5 and abs(disk.r - 904.0) < 2.0


@pytest.fixture(scope="module")
def synthetic_sun():
    rng = np.random.default_rng(3)
    cx, cy, r = 1031.5, 1017.0, 896.0
    fils = [random_filament(rng, cx, cy, r) for _ in range(5)]
    return sun_image(rng, fils, cx, cy, r), (cx, cy, r), fils


def test_disk_fit_and_limb_correction(synthetic_sun):
    img, (cx, cy, r), fils = synthetic_sun
    disk = fit_disk(img)
    assert abs(disk.cx - cx) < 1.5 and abs(disk.cy - cy) < 1.5 and abs(disk.r - r) < 2.0

    corr = correct(img, disk)
    rho = radius_map(disk, img.shape)
    quiet = ~np.any(fils, axis=0)
    for lo in [0.0, 0.5, 0.9]:  # quiet Sun maps to ~255 / 1.5 = 170 from centre to limb
        ring = quiet & (rho >= lo) & (rho < lo + 0.05)
        assert abs(np.median(corr[ring]) - 170) < 8
    assert np.median(corr[np.any(fils, axis=0) & (rho < 0.9)]) < 0.8 * 170  # filaments stay dark
    assert corr[rho >= 1].max() == 0


def test_folds_keep_records_and_nearby_days_together():
    gt = pd.DataFrame({
        "image_id": ["20150101120000Mh", "20150101120000Mh", "20150101180000Bh", "20150110120000Lh",
                     "20150120120000Ch", "20150130120000Th"],
        "annotator_image": ["010101-20150101120000Mh", "010102-20150101120000Mh", "010101-20150101180000Bh",
                            "010201-20150110120000Lh", "010301-20150120120000Ch", "010401-20150130120000Th"],
    })
    folds = make_folds(gt, n_folds=2).set_index("image_id")
    assert folds.loc["20150101120000Mh", "fold"] == folds.loc["20150101180000Bh", "fold"]  # 6 h apart
    assert folds.loc["20150101120000Mh", "n_records"] == 2
    assert set(folds["fold"]) == {0, 1}


def test_fusion_claims_are_disjoint_and_follow_detections():
    prob = np.zeros((2048, 2048), np.float32)
    prob[band(100, 100, 6, 200)] = 0.9   # columns 100-299 ...
    prob[band(100, 310, 6, 100)] = 0.9   # ... a 10-px gap, then columns 310-409
    prob[band(400, 400, 6, 50)] = 0.9    # a third segment that no detector sees
    on_disk = np.ones_like(prob, dtype=bool)
    # Detection boxes (grown by claim_dilation=2): columns 96-259 and 248-421.
    dets = [(0.8, rle(band(98, 98, 10, 160))), (0.6, rle(band(98, 250, 10, 170))), (0.1, rle(band(0, 0, 5, 5)))]

    inst = fuse(prob, on_disk, dets, PARAMS)
    assert [round(i.score, 2) for i in inst] == [0.8, 0.6]  # the 0.1 detection is below threshold
    a, b = (i.full((2048, 2048)).astype(bool) for i in inst)
    assert not (a & b).any()
    assert a.sum() == 6 * 160                  # columns 100-259
    assert b.sum() == 6 * 40 + 6 * 100         # the rest of segment 1 plus segment 2, across the gap

    with_unclaimed = fuse(prob, on_disk, dets, dict(PARAMS, unclaimed_min_area=100, unclaimed_min_prob=0.5))
    assert [i.source for i in with_unclaimed] == ["det", "det", "unet"]
    assert select(with_unclaimed, dict(PARAMS, unclaimed_min_area=400, unclaimed_min_prob=0.5)) == with_unclaimed[:2]

    grown = apply_boundary_offset(inst, 1, on_disk)
    g = [i.full((2048, 2048)).astype(bool) for i in grown]
    assert not (g[0] & g[1]).any() and g[0].sum() > a.sum()
    shrunk = postprocess(prob, on_disk, dets, dict(PARAMS, boundary_offset=-1))
    assert all(s.area < i.area for s, i in zip(shrunk, inst))


def test_unet_tiling_and_tta_put_pixels_back_in_place(synthetic_sun):
    """With a pointwise 'network', the blended 8-view prediction must equal the pointwise map."""
    import torch

    from src.data.preprocess import Disk
    from src.infer.unet import predict_prob

    class Pointwise(torch.nn.Module):
        def forward(self, x):
            return 10 * (x[:, :1] - 0.5) + 3 * x[:, 1:2]  # uses both channels; no spatial mixing

    img, (cx, cy, r), _ = synthetic_sun
    disk = Disk(cx, cy, r)
    corr = correct(img, disk)
    prob = predict_prob([Pointwise()], corr, disk, "cpu", tile=1024, overlap=256, tta=True, batch=8)
    rho = radius_map(disk, img.shape)
    expected = 1 / (1 + np.exp(-(10 * (corr / 255.0 - 0.5) + 3 * np.clip(rho, 0, 1.5) / 1.5)))
    inside = rho < 1
    np.testing.assert_allclose(prob[inside], expected[inside], atol=1e-4)
    assert prob[~inside].max() == 0


def test_detector_merge_and_letterbox_inversion():
    m1, m2 = band(100, 100, 10, 100), band(500, 500, 10, 100)
    merged = merge_detectors([[(0.9, rle(m1)), (0.4, rle(m2))], [(0.7, rle(m1))]])
    assert [round(s, 2) for s, _ in merged] == [0.8, 0.2]  # mean confidence, 0 for a miss

    native = np.zeros((2048, 1024), bool)
    native[1000:1040, 200:600] = True
    small = np.zeros((1, 512, 512), np.float32)  # letterboxed: scale 0.25, 128 px padding left/right
    small[0, 250:260, 128 + 50:128 + 150] = 1
    back = _to_native(small, (2048, 1024))[0]
    assert np.abs(back.astype(int) - native.astype(int)).sum() < 0.05 * native.sum()


def test_cache_and_yolo_export(tmp_path):
    from tests.synthetic import make_dataset

    make_dataset(tmp_path, n_train=4, n_test=1, seed=1)
    cfg = load_config(overrides=[f"data.cache_dir={tmp_path / 'cache'}", f"data.root={tmp_path / 'MAGFiLO_1.0_Kaggle_2026'}",
                                 "data.n_folds=2", "data.workers=1"])
    cache = build_train_cache(cfg, find_data(cfg["data"]["root"]))
    gt = pd.read_pickle(cache / "gt.pkl")
    multi = gt.groupby("image_id")["annotator_image"].nunique().idxmax()
    target = np.load(cache / "target" / f"{multi}.npy")
    assert {0, 255}.issubset(set(np.unique(target))) and (target == 128).any()  # annotators disagree
    disk = load_disks(cache)[multi]
    assert 880 < disk.r < 930 and disk_mask(disk).sum() > 0

    data_yaml = export_yolo(cache, fold=0)
    labels = list((cache / "yolo" / "labels").glob("*.txt"))
    assert len(labels) == gt["annotator_image"].nunique()
    first = labels[0].read_text().splitlines()[0].split()
    assert first[0] == "0" and all(0.0 <= float(v) <= 1.0 for v in first[1:])
    assert data_yaml.exists()
