"""
End-to-end steps shared by the CLI and the notebook:

- ``prepare``: preprocess the training images, build folds and the training cache;
- ``validate``: predict a held-out fold, score it exactly like Kaggle, tune post-processing;
- ``predict``: predict the test images and write a checked ``submission.csv``.
"""
import itertools
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import fold_tag, repo_path, resolve_device
from src.data.cache import build_train_cache, load_disks
from src.data.paths import find_data, list_images
from src.data.preprocess import Disk, disk_mask, preprocess, read_gray
from src.eval.pq import PQResult, evaluate
from src.infer.unet import predict_prob
from src.infer.yolo import detect, merge_detectors
from src.models.unet import load_unet
from src.postproc.fusion import Instance, apply_boundary_offset, fuse, postprocess, select
from src.submit.rle import check_submission, encode_mask, write_submission
from src.train.unet import unet_weights_path
from src.train.yolo import yolo_device, yolo_weights_path

SHAPE = (2048, 2048)


def prepare(cfg: dict) -> Path:
    return build_train_cache(cfg, find_data(cfg["data"]["root"]))


def postproc_params(cfg: dict) -> dict:
    """Config defaults, overridden by the tuned ``weights/postproc.json`` when it exists."""
    params = dict(cfg["postproc"])
    tuned = repo_path(cfg["weights_dir"]) / "postproc.json"
    if tuned.exists():
        params.update(json.loads(tuned.read_text()))
    return params


class Predictor:
    """The U-Nets and detectors trained without the given folds (``-1`` = trained on all data)."""

    def __init__(self, cfg: dict, folds: list[int]):
        from ultralytics import YOLO

        self.cfg = cfg
        self.device = resolve_device(cfg["device"])
        self.yolo_device = yolo_device(cfg)
        unet_paths = [unet_weights_path(cfg, f) for f in folds]
        yolo_paths = [(yolo_weights_path(cfg, m["name"], f), m["imgsz"]) for f in folds for m in cfg["yolo"]["models"]]
        missing = [str(p) for p in unet_paths if not p.exists()] + [str(p) for p, _ in yolo_paths if not p.exists()]
        if missing:
            raise FileNotFoundError(f"missing weights: {missing}")
        self.unets = [load_unet(p, self.device) for p in unet_paths]
        self.detectors = [(YOLO(str(p), task="segment"), imgsz) for p, imgsz in yolo_paths]

    def raw(self, corr: np.ndarray, disk: Disk) -> tuple[np.ndarray, list]:
        """U-Net probability map and merged detections for one limb-corrected image."""
        u, y = self.cfg["unet"], self.cfg["yolo"]
        prob = predict_prob(self.unets, corr, disk, self.device, u["infer_tile"], u["infer_overlap"],
                            u["tta"], u["infer_batch"])
        dets = merge_detectors([detect(model, corr, imgsz, y["conf"], y["iou"], y["max_det"], self.yolo_device)
                                for model, imgsz in self.detectors])
        return prob, dets


def _rows(image_id: str, instances: list[Instance]) -> list[tuple[str, str]]:
    ordered = sorted(instances, key=lambda i: -i.score)
    return [(f"{image_id}_{n}", encode_mask(inst.full(SHAPE), verify=False)) for n, inst in enumerate(ordered, 1)]


def _score(raws: dict, gt: pd.DataFrame, p: dict) -> tuple[PQResult, pd.DataFrame]:
    rows = []
    for image_id, (prob, disk, dets) in raws.items():
        rows += _rows(image_id, postprocess(prob, disk_mask(disk, SHAPE), dets, p))
    pred = pd.DataFrame(rows, columns=["filament_id", "segmentation_rle"])
    return evaluate(gt, pred), pred


def tune(raws: dict, gt: pd.DataFrame, cfg: dict) -> tuple[dict, pd.DataFrame]:
    """Grid search of the post-processing on a validation fold.

    For each (U-Net threshold, claim dilation) the instances are built once with the loosest
    thresholds and encoded; the stricter score/area thresholds are then applied by filtering.
    That ignores the rare pixel a dropped claim would have passed to a weaker detection, so the
    winner is re-scored exactly (also for each boundary offset) before it is returned.
    """
    grid, base = cfg["tune"], dict(cfg["postproc"])
    positive_areas = [a for a in grid["unclaimed_min_area"] if a > 0]
    table = []
    for t_unet, dil, close in itertools.product(grid["unet_threshold"], grid["claim_dilation"],
                                                grid.get("close_radius", [base["close_radius"]])):
        loose = dict(base, unet_threshold=t_unet, claim_dilation=dil, close_radius=close,
                     det_threshold=min(grid["det_threshold"]),
                     min_area=min(grid["min_area"]), unclaimed_min_area=min(positive_areas, default=0),
                     unclaimed_min_prob=min(grid["unclaimed_min_prob"]))
        cands = {i: [(inst, encode_mask(inst.full(SHAPE), verify=False))
                     for inst in fuse(prob, disk_mask(disk, SHAPE), dets, loose)]
                 for i, (prob, disk, dets) in raws.items()}
        combos = {(t, a, ua, up if ua > 0 else None) for t, a, ua, up in itertools.product(
            grid["det_threshold"], grid["min_area"], grid["unclaimed_min_area"], grid["unclaimed_min_prob"])}
        for t, a, ua, up in sorted(combos, key=str):
            p = dict(loose, det_threshold=t, min_area=a, unclaimed_min_area=ua, unclaimed_min_prob=up or 0.0)
            rows = []
            for image_id, items in cands.items():
                kept = {id(inst) for inst in select([inst for inst, _ in items], p)}
                ordered = sorted((it for it in items if id(it[0]) in kept), key=lambda it: -it[0].score)
                rows += [(f"{image_id}_{n}", rle) for n, (_, rle) in enumerate(ordered, 1)]
            res = evaluate(gt, pd.DataFrame(rows, columns=["filament_id", "segmentation_rle"]))
            table.append({**{k: p[k] for k in ["unet_threshold", "close_radius", "claim_dilation", "det_threshold",
                                                "min_area", "unclaimed_min_area", "unclaimed_min_prob"]},
                          "boundary_offset": 0, "pq": res.pq, "tp": res.tp, "fp": res.fp, "fn": res.fn})
        print(f"[tune] unet_threshold={t_unet} claim_dilation={dil} close_radius={close}: best so far "
              f"{max(r['pq'] for r in table):.4f}", flush=True)

    best = dict(base, **{k: v for k, v in max(table, key=lambda r: r["pq"]).items()
                         if k not in {"pq", "tp", "fp", "fn"}})
    exact = []
    for offset in grid["boundary_offset"]:
        p = dict(best, boundary_offset=offset)
        res, _ = _score(raws, gt, p)
        exact.append((res.pq, offset))
        print(f"[tune] exact PQ with boundary_offset={offset}: {res.pq:.4f}", flush=True)
    best["boundary_offset"] = max(exact)[1]
    return best, pd.DataFrame(table).sort_values("pq", ascending=False)


def validate(cfg: dict, fold: int) -> dict:
    """Scores the models trained without ``fold`` on that fold, tunes and saves the post-processing."""
    if fold < 0:
        raise ValueError("validation needs a held-out fold (fold >= 0)")
    cache = Path(cfg["data"]["cache_dir"])
    gt = pd.read_pickle(cache / "gt.pkl")
    folds = pd.read_csv(cache / "folds.csv")
    disks = load_disks(cache)
    val_ids = folds.loc[folds["fold"] == fold, "image_id"].tolist()
    gt_val = gt[gt["image_id"].isin(val_ids)][["filament_id", "segmentation_rle"]].reset_index(drop=True)

    predictor = Predictor(cfg, [fold])
    raws, start = {}, time.time()
    for k, image_id in enumerate(val_ids, 1):
        corr = np.load(cache / "corr" / f"{image_id}.npy")
        prob, dets = predictor.raw(corr, disks[image_id])
        raws[image_id] = (prob.astype(np.float16), disks[image_id], dets)
        if k % 20 == 0 or k == len(val_ids):
            print(f"[validate] predicted {k}/{len(val_ids)} images ({time.time() - start:.0f}s)", flush=True)

    base = dict(cfg["postproc"])
    base_res, _ = _score(raws, gt_val, base)
    print(f"[validate] default post-processing: {base_res.summary()}")
    best, table = tune(raws, gt_val, cfg)
    best_res, oof = _score(raws, gt_val, best)
    print(f"[validate] tuned post-processing:   {best_res.summary()}")

    out_dir = repo_path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = fold_tag(fold)
    table.to_csv(out_dir / f"tune_{tag}.csv", index=False)
    oof.to_csv(out_dir / f"oof_{tag}.csv", index=False)
    gt_val.to_csv(out_dir / f"gt_{tag}.csv", index=False)
    tuned = {k: best[k] for k in cfg["postproc"]}
    (out_dir / f"postproc_{tag}.json").write_text(json.dumps(tuned, indent=2))
    weights = repo_path(cfg["weights_dir"])
    weights.mkdir(parents=True, exist_ok=True)
    (weights / "postproc.json").write_text(json.dumps(tuned, indent=2))
    report = {"fold": fold, "n_images": len(val_ids), "n_records": int(gt_val["filament_id"].str.split("_").str[0].nunique()),
              "default": {"pq": base_res.pq, "sq": base_res.sq, "rq": base_res.rq,
                          "tp": base_res.tp, "fp": base_res.fp, "fn": base_res.fn},
              "tuned": {"pq": best_res.pq, "sq": best_res.sq, "rq": best_res.rq,
                        "tp": best_res.tp, "fp": best_res.fp, "fn": best_res.fn},
              "params": tuned}
    (out_dir / f"validation_{tag}.json").write_text(json.dumps(report, indent=2))
    return report


def predict(cfg: dict, folds: list[int], out_csv: str | Path) -> Path:
    """Predicts every test image with the models of ``folds`` and writes a checked submission."""
    paths = find_data(cfg["data"]["root"])
    images = list_images(paths.test_images)
    params = postproc_params(cfg)
    print(f"[predict] {len(images)} test images, folds {folds}, post-processing {params}")
    predictor = Predictor(cfg, folds)
    rows, seconds = [], []
    for k, (image_id, path) in enumerate(images.items(), 1):
        start = time.time()
        corr, disk = preprocess(read_gray(path), cfg["preprocess"]["limb_bin"], cfg["preprocess"]["corr_max"])
        prob, dets = predictor.raw(corr, disk)
        rows += _rows(image_id, postprocess(prob, disk_mask(disk, corr.shape), dets, params))
        seconds.append(time.time() - start)
        if k % 20 == 0 or k == len(images):
            print(f"[predict] {k}/{len(images)} images, {len(rows)} filaments, "
                  f"{np.mean(seconds):.1f}s per image", flush=True)

    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    write_submission(rows, out_csv)
    problems = check_submission(out_csv, image_ids=set(images))
    if problems:
        raise RuntimeError("invalid submission:\n" + "\n".join(problems[:20]))
    print(f"[predict] wrote {out_csv}: {len(rows)} filaments on {len(images)} images "
          f"({len(rows) / max(len(images), 1):.1f} per image); submission check passed")
    return out_csv
