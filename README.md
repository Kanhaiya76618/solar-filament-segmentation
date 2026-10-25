# Solar filament instance segmentation (Solar Filament Segmentation Challenge 2026)

This repository segments every solar filament in full-disk GONG H-alpha images as a separate
instance mask, and writes the competition's `submission.csv`. It is built for the challenge's
Panoptic Quality (PQ) metric: filaments must be complete (thin barbs included), must not be split
into fragments, and must not be merged with their neighbours.

## Method

```
GONG JPEG (2048x2048)
  -> disk fit (steepest limb edge) + limb-darkening correction + radial-distance channel
  -> U-Net (ResNet34) on native-resolution tiles, 8-fold TTA      -> filament probability map
  -> YOLO11-seg, trained on each annotator's labels separately     -> filament instances and confidences
  -> fusion: detections claim U-Net pixels in order of confidence  -> disjoint filament masks
  -> post-processing thresholds tuned for PQ on a held-out fold    -> submission.csv (COCO RLE)
```

- **Why two models.** Filaments are often only 3–8 px wide, so a 1-px boundary error already
  costs about 25% IoU. The U-Net predicts pixels at full resolution. Detector masks are too coarse
  for that, but detectors are good at deciding which pixels belong to one filament, even across
  faint gaps. So the detector decides grouping, and the U-Net decides the boundary.
- **Annotator disagreement.** Many images were annotated independently by 2–3 experts, and the
  metric scores each annotator separately. The U-Net learns the mean of the annotators' masks, and
  the detector sees every annotator's labels as a separate sample.
- **Leakage-safe validation.** All annotations of an image stay in one fold, and images taken less
  than a day apart share a fold, because they show the same filaments.
- **Exact metric.** `src/eval/pq.py` reproduces the organizers' self-evaluation notebook bit for bit
  (tested against a verbatim copy) and is fast enough for post-processing grid searches.

Results: *fill in after training* (validation PQ / SQ / RQ, public leaderboard).

## Running it

### On Kaggle (recommended)

1. New notebook: **GPU T4×2**, **Internet on**. Add the competition data.
2. Make the code available: push this repository to GitHub and set `REPO_URL` in the first code
   cell, or upload the folder as a Kaggle dataset and attach it.
3. Open `notebooks/pipeline.ipynb` and run all cells.
   - `MODE = "train"`: about 3–4 h; writes `weights/` and `submission.csv`.
   - `MODE = "reproduce"`: predicts only, using the committed weights.

### With the command line

```bash
python -m src.cli prepare
```

```bash
python -m src.cli train-unet --fold 0
```

```bash
python -m src.cli train-yolo --fold 0
```

```bash
python -m src.cli validate --fold 0
```

```bash
python -m src.cli predict --folds 0 --out submission.csv
```

- `--fold k` trains without fold k; `--fold -1` trains on all data.
- `--folds 0 1 2` ensembles the models of several folds.
- `--set key=value` overrides any entry of `configs/default.yaml`.

### Tests

```bash
python -m pytest -q
```

The tests cover:
- equality with the official scoring code;
- RLE encoding;
- overlap removal and submission checks;
- disk fitting and limb correction;
- folds;
- fusion;
- detector merging;
- tile/TTA inversion;
- the training cache.

`python -m tests.synthetic <dir>` writes a small synthetic dataset in the competition layout for
smoke runs.

## Repository layout

| Path | Contents |
|---|---|
| `configs/default.yaml` | All hyperparameters and post-processing defaults |
| `src/data/` | Annotation loading, disk fit and limb correction, folds, training cache, crop dataset |
| `src/models/unet.py` | U-Net construction and fp16 checkpoints |
| `src/train/` | U-Net and YOLO training |
| `src/infer/` | Tiled U-Net inference with TTA; YOLO inference and cross-detector merging |
| `src/postproc/fusion.py` | Pixel/instance fusion, thresholds, boundary offset |
| `src/eval/` | Exact PQ evaluator, per-prediction targets, diagnostic plots |
| `src/submit/rle.py` | RLE encoding, overlap removal, CSV writer and checker |
| `src/pipeline.py`, `src/cli.py` | `prepare`, `validate` (tuning), `predict` |
| `notebooks/pipeline.ipynb` | The whole pipeline, end to end |
| `weights/` | Trained models and `postproc.json`, loaded by the notebook |
| `docs/PLAN.md` | Research notes, verified competition facts and the improvement plan |

## Data and rules

- Training uses only the competition's training images and annotations, plus ImageNet (U-Net
  encoder) and COCO (YOLO) pretrained weights.
- Inference uses only the supplied test image.
- No public MAGFiLO annotations of test images are used.
