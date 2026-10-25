# Solar Filament Segmentation Challenge 2026: verified master plan

Written October 25, 2026. 21 days remain before the deadline (Nov 15, 11:59 pm CDT).

I checked this plan against the primary sources: the organizers' scoring notebook, the Data tab,
the host's rulings in the discussion forum, the public notebooks that the Google report cites,
the strongest public reproducible repository, and the EdgeAttNet paper. Section 9 lists every source.

---

## 0. Summary

1. **The public leaderboard above ~0.45 is contaminated by leaked labels.** All 180 test images
   appear, with full annotations, in the public MAGFiLO v1.0 release. The 0.55 cluster of about 30
   teams comes from a leaked CSV or leak-trained weights. The host says *"we take a score of 0.41
   very seriously"* and will clean the leaderboard after the deadline. The Google report's gates
   ("classical ≥ 0.45", "YOLO ≥ 0.55") cannot be reached legitimately. **Real target: 0.42–0.48
   on local OOF, which corresponds to about 0.38–0.43 on the leaderboard.**
2. **Human annotators agree with each other at only PQ ≈ 0.34** (measured over 296
   multi-annotator observations). A model scoring 0.42 already beats annotator-vs-annotator
   agreement, because it predicts a "consensus" filament.
3. **Start from the one public recipe with verified numbers**, rather than from an
   architecture nobody has tested: a native-resolution ResNet34 U-Net for pixels, YOLO11-seg
   detectors to decide which pixels form one filament, and score-ordered disjoint painting. It
   scored 0.421 validation PQ (ameyypawar/filament-seg).
4. **Most of the remaining PQ is won after the network.** Section 3 gives an exact rule for when
   a predicted filament helps PQ: *keep it only if its expected matched IoU exceeds PQ/2*.
   Implement it with a small learned instance scorer trained on out-of-fold predictions.
5. **The rules allow three things that most competitors do not use:** the manual
   **spine** annotations, **temporally neighbouring GONG frames during training** (the host:
   *"we like such ideas"*), and **unlabelled GONG images for pretraining**. Test-time inference
   must use only the single supplied image.
6. **The qualitative 30% decides the prize, and the host is explicitly suspicious of AI-written
   reports.** Every component you ship must be one you can explain and ablate. The simpler,
   measured pipeline below is easier to defend than the Google report's eleven-component design.
7. **A verified evaluator is already in this folder** (`src/eval/pq.py`). It matches the
   organizers' notebook bit for bit and runs about 600× faster. That speed makes
   threshold and post-processing sweeps cheap.

---

## 1. Verified facts

| Topic | Verified fact | Source |
|---|---|---|
| Data package | 888 files (750.97 MB): 707 train JPEGs, 180 test JPEGs, and one COCO JSON. Images are 2048×2048 8-bit grayscale converted from FITS. | Kaggle Data tab |
| Train statistics | 707 observations, 1,154 annotator records, 8,199 filaments. The median filament is **1,228 px**, and **41% are under 1,000 px**. | ameyypawar/filament-seg |
| Multi-annotator | 296 of 707 train observations have 2–3 independent annotator records. One annotator marked 409 filaments and another marked 378 on the same images, and **23%** of the first set do not appear in the second. | Discussion 729809 |
| Human ceiling | **Annotator vs annotator PQ = 0.343** (296 observations). Annotator vs annotator mean Dice = 0.687. | ameyypawar; discussion 729809 |
| GT format | Each filament is one closed polygon (`iscrowd` = 0). The JSON also has `spine` (one connected polyline), `bbox`, `area` (computed by pycocotools from the polygon) and `category_id` (chirality). | Data tab |
| Allowed supervision | The host says you **may use spine, bbox, area and category_id**. What you may not use is any external information that leaks test ground truth. | Discussion 736543 (host) |
| Scoring | The match rule is `iou_matrix > 0.5` (strict). Each annotator record is scored separately against all predictions for its image. TP, FP and FN are pooled across records (micro-averaged). Predictions on images with no GT record are ignored. Diagnostics: IoU and Dice histograms over pairs with IoU > 0, and 1:n / n:1 counts over pairs with IoU > 0. | Self-evaluation notebook v6, reproduced in `tests/reference_official.py` |
| Overlaps | The Kaggle scorer **rejects** submissions with overlapping masks (status ERROR). Host: *"we do not accept overlapping predictions."* | Discussion 735377 (host) |
| Leak | All 180 test filenames appear with annotations in public MAGFiLO v1.0, and 139 are byte-identical. The host: the leaderboard *"is only a preliminary way of filtering"*. | Discussion 728062 |
| Legit frontier | Honest reports cluster at local 0.46–0.49 and leaderboard 0.40–0.41. The host: *"we take a score of 0.41 very seriously."* Scores above ~0.45 are attributed to the leak. | Discussions 743958, 745880, 741892 |
| Local→LB gap | Two independent teams report a gap of about −0.05 to −0.06 (0.46→0.40; 0.421→0.37). | 745880; ameyypawar |
| Temporal data | Neighbouring frames are fine **for training**. At inference, the model may use **only the supplied test image**. | Discussion 743385 (host) |
| Unlabelled GONG | Self-supervised pretraining on extra unlabelled GONG H-alpha images is allowed. | Discussion 737215 (host) |
| Reproducibility | Weights must be **stored in the repo and loaded in the notebook**, otherwise the entry is ineligible. | Host, Oct 23 comment in 724503 |
| AI use | AI is allowed, but *"prize-eligible participants must fully understand every component"*. A report *"generated by AI without meaningful human oversight undermine[s] our confidence"*. | Host, 724503 |
| Missing labels | The host is interested in all filaments, but tiny detections lacking morphology are uninteresting. FPs cost PQ, and judges also read the m:n plots. | Discussion 735256 (host) |
| GONG archive | 1-minute cadence per site. FITS are in `gong2.nso.edu/ftp/HA/haf/`; 2048² JPEGs (wavelet-processed) are in `/ftp/HA/hag/`. Disks are remapped to a fixed radius (the page gives both 860 and 900 px). | NSO H-alpha product page |

---

## 2. Audit of the Google Deep Research report

### What it got right (keep)
- IoU > 0.5 is strict, scoring is per annotator record, TP/FP/FN are micro-averaged, and masks must be disjoint.
- Fortran-order RLE with a round-trip check, no quotes, and no rows for empty images.
- Validation grouped by physical image, with one annotator sampled per epoch.
- The input is an 8-bit JPEG (not FITS), so limb-darkening correction and disk masking are needed.
- Scoring tiny satellite fragments as FPs also pollutes the IoU histogram and the 1:n plot (confirmed in the notebook code).

### What it got wrong (fix)

| Claim in the report | Reality | Consequence |
|---|---|---|
| "Classical baseline scores 0.46 on public LB (vmerckle)" | That notebook submits **120 masks per image** (21,597 rows). The "0.46" was scored under the pre-Aug-12 Dice metric. Under PQ, a tuned threshold baseline scores **0.08** (ameyypawar). | The phase-1 gate "classical ≥ 0.45" is impossible. A classical baseline is only for plumbing. |
| "YOLO-seg pipeline (hdjojo) reaches 0.55" | 0.55 is the leak cluster. That notebook loads weights named `magfilo_…` of unknown provenance. It also sorts masks by **class id** (`boxes.cls`, all zero), not by confidence, so overlap priority is arbitrary. | Do not use those weights or that CSV. Do not chase 0.55. |
| "Hacking = uploading duplicate overlapping masks" | The scorer **rejects** overlaps. The real shortcut is the public test annotations. | Disjointness is a validity requirement, not an anti-hack. |
| "Derive spine from masks to obey the no-metadata rule" | The host explicitly allows the manual **spine**, bbox, area and category. | Use the real spine as auxiliary supervision. |
| "3.5 is a typo for 0.35" | Probably, but the host wrote 0.3, 0.35, 3.5 and 0.41 in different places. | The only actionable signal: an honest 0.35–0.45 is valued. |
| Centroid-offset head with Z = 256 | The centroid of a curved filament often lies **outside** it. Offsets of hundreds of pixels on 6-px-wide objects are hard to regress. Omnipose (Cutler et al. 2022) was created because centroid-style flows fail on elongated cells. | Use detector ownership. If a dense grouping head is needed, use distance-field flows (Omnipose-style), not centroids. |
| "EG-MHSA outperforms transformers on thin barbs" | The paper reports **pixel** mIoU (0.6451 vs 0.5724 for U-Net) for **from-scratch** models **without augmentation**. It has no instance or PQ evaluation and no barb metric. | Worth at most one ablation. It is not proven to help PQ. |
| YOLO11x-seg at 2048 as Branch 2 | YOLO masks come from prototypes at **1/4 of input resolution**. At imgsz 1536, one prototype cell is 5.3 native pixels, which is too coarse for 6-px filaments. ameyypawar found detector masks *"too coarse to submit alone"*. | Use YOLO s/m for ownership and confidence, and the U-Net for pixels. |
| Four heads + EG-MHSA + YOLO11x + steerable closing + offset watershed + Optuna + Hungarian consensus in 21 days | No ablation evidence for any of it on this task. It is also hard to explain to judges who penalise uncritical AI use. | Ship a smaller, measured system and add parts only when ablations pay. |
| Frangi channel and 3rd-order polynomial "cloud" surface | Plausible, but untested on this data. | Optional ablations, not core. |

---

## 3. The PQ mathematics that should drive every decision

Notation: for a candidate filament *i* on an image with *M* annotator records, let
`r_i = mean over records a of IoU_ia · [IoU_ia > 0.5]`.
`src/eval/pq.py:prediction_targets` computes this as `expected_iou`.

**3.1 Marginal value of one prediction.** Adding a prediction always adds 0.5 per record to the
denominator: either it matches (TP + 1, FN − 1) or it does not (FP + 1). It adds `r_i` per record
to the numerator on average. So PQ rises if and only if

```
r_i  >  PQ / 2
```

With PQ ≈ 0.42, keep a candidate only if its *expected matched IoU* exceeds about 0.21. If matched
IoUs are typically about 0.65, this means a match probability above about 0.32. That is roughly
consistent with the 0.30 detection-confidence cut ameyypawar found empirically, though confidence is
not a calibrated probability, so treat this only as a sanity check. The rule replaces hand-tuned
confidence and minimum-area thresholds with a single quantity you can estimate.

**3.2 Optimising the whole ratio (Dinkelbach's method).** PQ = S/D is a ratio, so maximise
`S − λ·D` with λ fixed, then update λ ← S/D and repeat. This converges in 2–3 iterations.
For a fixed λ the decision decouples per candidate (keep if `r̂_i > λ/2`). Use it for
three decisions:
- **keep or drop** a candidate: keep if `r̂_i > λ/2`;
- **merge or split** fragments A and B: merge if `r̂_AB − λ/2 > max(r̂_A − λ/2, 0) + max(r̂_B − λ/2, 0)`;
- **which of two overlapping candidates owns the shared pixels**: paint in order of `r̂`.

**3.3 Thin objects make IoU fragile.** For a filament of width *w*, a boundary error of δ px on
both sides gives IoU ≈ w/(w + 2δ). With w = 6: δ = 1 gives **0.75** and δ = 2 gives **0.60**. A 1-px
sideways shift gives (w − 1)/(w + 1) = **0.71**. Consequences:
- predict pixels at **native 2048 resolution** (tiles), never downsample-and-upsample;
- rasterise training polygons **exactly as pycocotools does** (`src/data/annotations.py`), because a
  half-pixel convention change is a systematic δ;
- tune a **global boundary offset** (erode or dilate by −1/0/+1 px, and the U-Net threshold)
  directly on OOF PQ. This is now cheap.

**3.4 Annotator disagreement.** About a quarter of filaments are marked by only some of the annotators.
Training on a randomly sampled annotator per epoch teaches the model the *probability* that a
filament is annotated, which is what 3.1 needs. Do not union annotators, because that merges
filaments one expert split.

---

## 4. Pipeline

### 4.1 Base system: reproduce first (target OOF ≈ 0.40–0.42)
This follows ameyypawar/filament-seg. Each step below has published ablation numbers.

1. **Disk fit.** Fit the limb at the sharpest radial edge. A thresholded disk is too large by
   39 px at the median, because GONG has a bright halo. Zero everything off the disk.
2. **Limb-darkening correction.** Divide by the median intensity in concentric rings
   (EdgeAttNet uses the same radial flattening).
3. **Input channels:** (a) corrected intensity, (b) distance from disk centre in solar radii.
4. **Pixel model.** U-Net (segmentation_models_pytorch, ResNet34 ImageNet encoder) on
   **512 px native-resolution crops**, Dice + BCE, about 30 epochs. Infer with Hann-blended
   overlapping tiles and D4 TTA (8 flips and rotations). **Use the last epoch**, which generalised
   better than the best-validation epoch.
5. **Ownership models.** YOLO11-seg at several resolutions (11s@1024, 11m@1280, 11m@1536),
   trained with **each annotator record as a separate sample**. Merge detections across models when
   IoU > 0.5, average the confidences, and count a miss as 0.
6. **Fusion.** Threshold the U-Net at 0.5, close with a 3-px disk, fill holes and clip to the disk.
   Process detections from most to least confident; each claims the unclaimed U-Net pixels inside
   its mask dilated by 4 px. Drop claims under 200 px. This gives disjoint instances by
   construction.
7. **Encode** with `src/submit/rle.py` and run `check_submission` before every upload.

Published ablation: U-Net alone 0.371, detectors alone 0.383, fused 0.418–0.421.

### 4.2 Upgrades in priority order

Add one at a time. Keep an upgrade only if it raises OOF PQ by at least +0.005 on two folds.
Expected gains are **hypotheses** until measured.

**Tier 1: post-processing (CPU only, highest value per day)**

- **U1. Candidate pool + learned instance scorer + Dinkelbach threshold.**
  - Candidates are every fused instance plus every U-Net connected component that **no detector
    claimed**. The base system discards these, losing recall on filaments YOLO missed.
  - For each candidate, compute features: area, skeleton length, mean width, elongation, mean and
    max U-Net probability, detector max confidence, number of detectors that agree, contrast
    against the local background, radial position ρ, and number of connected parts.
  - Target: `expected_iou` from `prediction_targets`, computed on OOF predictions.
  - Fit a LightGBM regressor with grouped CV, then select candidates with the rule from 3.2.
  - This is the same idea as Mask Scoring R-CNN's MaskIoU head (Huang et al., CVPR 2019), done as
    cheap tabular learning.
- **U2. Boundary calibration.** Sweep the global erode/dilate offset (−1, 0, +1, +2), the U-Net
  threshold and the detector dilation (2–6 px) on OOF PQ.
- **U3. Fragment merging.** Merge two candidates when they lie inside one detector envelope or when
  the spine probability connects them (see U5). Accept the merge only if the 3.2 merge rule says yes.
  This directly attacks the 1:n bars.

**Tier 2: model upgrades (GPU)**

- **U4. Soft multi-annotator pixel target.** Train the U-Net on the mean of the available annotator
  masks, or sample one annotator per epoch. Compare both. This calibrates the probabilities that U1
  consumes.
- **U5. Spine auxiliary head.** Add a second output channel supervised by the **manual spine**,
  dilated to about 3 px (host-approved). Use it to bridge gaps inside one filament (a geodesic path
  along high spine probability) and as a scorer feature. This targets the "structural continuity"
  challenge named in the competition text.
- **U6. Neighbour-frame augmentation.** The host approved this for training only.
  - The Sun rotates about 13.2°/day, which is about 0.14 px/min at disk centre. Frames ±1–5 min from
    the same site therefore have the **same filaments within 1 px**, but different seeing, noise
    and cloud.
  - This gives 5–10× more real training images with the same labels.
  - **Check first** that a `hag` archive JPEG of a training frame matches the competition JPEG
    (byte-identical or histogram-equivalent). If it does not, convert the `haf` FITS with a mapping
    fitted on the exact training frames, or drop U6.
  - This is also a strong, original point for the report.
- **U7. Detector resolution.** Try YOLO11m-seg at 2048 and `retina_masks=True`. Ownership quality
  may improve even though the masks stay coarse.
- **U8. Crop size and context.** Compare 768 or 1024 crops against 512 for continuity of long filaments.

**Tier 3: only if Tier 1–2 are done by Nov 5**

- **U9.** EdgeAttNet-style edge-guided bottleneck attention (Q = K = features + learned edge map,
  4 heads, no positional encoding) as a single ablation. Judges from the organizing lab may value
  this connection, but the paper's evidence is pixel-level only.
- **U10.** Copy-paste augmentation of filaments onto quiet-Sun regions (Ghiasi et al., CVPR 2021).
- **U11.** Self-supervised pretraining on unlabelled GONG. This is allowed, but probably too slow for 21 days.

**Do not build:** Mask2Former, centroid-offset heads, steerable orientation closing, Optuna over a
dozen coupled thresholds, or Hungarian consensus across architectures. Each costs days, has no
evidence on this task, and is hard to defend in a 4-page report.

### 4.3 Ensembling
- Average U-Net probabilities across folds, seeds and TTA views. These come from the same
  architecture, so pixel averaging is safe.
- Merge detector outputs across models by IoU > 0.5, as in the base system.
- The scorer is trained on single-fold OOF outputs, but the 5-fold ensemble on test is slightly more
  confident. Re-check λ on a held-out fold that was scored with an ensemble of the other folds, or
  shade λ upward by about 0.01–0.02.

---

## 5. Validation protocol

1. **Groups:** one group per physical image (all annotator records together). Then merge groups
   whose observations fall within **±1 day**, even across sites, because the same filaments are
   visible. Check the group sizes. If 72-h single-linkage chains the dataset into a few huge
   clusters, fall back to day-level groups.
2. **5 folds** with StratifiedGroupKFold, stratified by year and site. Fold 0 is the fast
   ablation fold. Use all 5 folds for the final OOF scorer and λ.
3. **Score exactly like Kaggle:** one prediction set per physical image, scored against every
   annotator record with `evaluate()`.
4. **Avoid optimistic tuning:** fit post-processing parameters on 4 folds' OOF and report PQ on
   the 5th (cross-fitting). Report both numbers.
5. **Leaderboard calibration:** submit at most about 3 times in total (base, after Tier 1, final).
   Expect LB ≈ local − 0.05. Never tune on the leaderboard: it is half the test set and contaminated.
6. **Audit numbers to compute on day 1:**
   - record and filament counts (707 / 1,154 / 8,199);
   - annotator multiplicity;
   - area, length (skeleton) and width (area ÷ length) distributions;
   - filaments per image;
   - per-site intensity statistics;
   - **time from each test image to its nearest training image**, from the filename timestamps, so you
     can see whether your fold gaps mimic the test relationship.

---

## 6. Compute plan

Your Mac (Apple M5, 16 GB) handles data preparation, the evaluator, post-processing sweeps, the
LightGBM scorer and inference sanity checks. Train on Kaggle GPUs (30 h/week, T4×2 or P100, about
90 GPU-hours until the deadline) and Colab if needed.

| Job | Rough cost per run | Runs |
|---|---|---|
| U-Net R34, 512 crops, 30 epochs | ~1–2 h | 5 folds + 2–4 ablations |
| YOLO11m-seg @1536, 80 epochs | ~2–3 h | 5 folds + 1–2 ablations |
| YOLO11s-seg @1024 | ~1 h | 5 folds |
| Test inference with TTA + 5 folds | < 1 h | 2–3 |

These are estimates; time the first fold-0 runs and re-plan. Run folds in parallel Kaggle sessions.

---

## 7. Schedule (Oct 25 – Nov 15)

| Dates | Work | Gate |
|---|---|---|
| **Oct 25–26** | Join the competition and download the data. Create the GitHub repo and copy this folder in. Run the tests. Load the JSON with `load_annotations`. Run the audit (section 5.6) and build the folds. Download 3 `hag` JPEGs of training frames and compare them with the competition JPEGs (decides U6). | Evaluator scores GT against itself at 1.0; folds are fixed and saved |
| **Oct 27–29** | Preprocessing (disk fit, limb correction, radius channel). Fold-0 U-Net and YOLO11m@1536. Base fusion. First OOF PQ. **One** LB submission to measure the gap. Start folds 1–4 in background sessions. | Fold-0 PQ ≥ 0.38 |
| **Oct 30–Nov 2** | Tier 1 on fold 0: candidate pool, scorer prototype, boundary sweep, fragment merging. Neighbour-frame download script, if the Oct 26 check passed. | +0.01–0.03 over base on fold 0 |
| **Nov 3–5** | Tier 2 ablations on fold 0, confirmed on fold 1: soft targets, spine head, neighbour frames, YOLO@2048, crop size. **Freeze the architecture on Nov 5.** | Each kept upgrade gives ≥ +0.005 on two folds |
| **Nov 6–8** | Train all 5 folds of the frozen configuration. Build the full OOF candidate table, train the scorer with grouped CV, and run Dinkelbach for λ. Compute the cross-fitted OOF PQ. | Final OOF PQ known, cross-fitted |
| **Nov 9–10** | Test inference with the 5-fold ensemble. Run `check_submission`. Make 1–2 LB submissions. Save weights as **fp16, each file < 50 MB**, without Git LFS. Make the notebook run end to end. | Valid CSV; LB within ~0.06 of OOF |
| **Nov 11–12** | Clean-room reproduction: a fresh Kaggle notebook clones the public repo and regenerates `submission.csv`, which should score ≥ 0.9999 PQ against the submitted CSV. Make figures and measure runtime per image. | Reproduction passes |
| **Nov 13–14** | **Write the 4-page report yourself** (section 8). Write the README and a final code review. | Report is complete |
| **Nov 15** | Make the repo public and submit the Google form **by noon CDT**, leaving the evening as buffer. | Submitted |

**If you fall behind:** drop Tier 2 entirely. The base system plus U1–U3 is still a strong,
defensible entry.

---

## 8. Rules, integrity and deliverables checklist

**Never:**
- download or use full MAGFiLO v1.0 annotations (all 180 test images are in them);
- use any CSV, weights or Kaggle "Models" entry of unknown training data (for example the 0.55 CSV or `magfilo_*` YOLO weights);
- feed neighbouring frames to the model at inference;
- submit overlapping masks.

**Allowed:**
- the training JSON's masks, spine, bbox, area and category;
- unlabelled GONG frames for pretraining, and neighbouring frames for training;
- ImageNet- or COCO-pretrained backbones;
- AI assistance, as long as you understand every line.

**Repository** (the code-quality score is 30% of the rubric together with the report and morphology):
- a modular `src/` tree (data / models / postproc / eval / submit), tests, and `configs/` with every
  threshold;
- `requirements.txt` with exact versions, captured from the Kaggle image you actually used
  (`pip freeze`);
- `notebooks/pipeline.ipynb`, which loads the weights from the repo and reproduces
  `submission.csv` with no retraining;
- weights committed as fp16 state_dicts under 100 MB each (GitHub's hard limit). Avoid Git LFS,
  because its bandwidth quota can break judges' clones.

**Report** (4 pages, Overleaf template). The host penalises reports written by AI without oversight,
so write it yourself in your own words. Content:
1. a pipeline figure;
2. a table of PQ, SQ and RQ per ablation (base → each upgrade);
3. the official IoU and Dice histograms and the 1:n / n:1 plot, base versus final;
4. the human-agreement context (PQ 0.343) and why the model exceeds it;
5. the PQ/2 selection rule with its one-line derivation;
6. runtime per image and hardware;
7. qualitative overlays: barbs, a filament near the limb, a sunspot rejected, and a failure case;
8. the neighbour-frame idea, if used.

Expect to be asked to justify any component.

---

## 9. Code in this folder

The base system of section 4.1 is implemented end to end. See `README.md` and
`notebooks/pipeline.ipynb`. It covers:
- preprocessing (disk fit, limb correction, radius channel);
- leakage-safe folds;
- U-Net training and tiled TTA inference;
- YOLO11-seg training and inference;
- detection-claims fusion;
- a PQ-tuned post-processing grid, including the unclaimed-component candidates of U1 and the
  boundary offset of U2;
- a checked `submission.csv`.

The evaluation building blocks:

| Path | What it does | Status |
|---|---|---|
| `src/eval/pq.py` | `evaluate()` reproduces the official notebook (PQ, SQ, RQ, TP/FP/FN, histogram data, 1:n / n:1 counts) from RLE strings. `prediction_targets()` gives per-prediction `expected_iou` and `match_rate` for the scorer (U1). | Tested: bit-for-bit equal to the notebook on random scenes; about 600× faster (0.01 s vs 6.7 s on 20 synthetic images) |
| `src/submit/rle.py` | `encode_mask` (Fortran order + round-trip check), `paint_disjoint` (score-ordered de-overlap with a minimum area), `write_submission` (fails rather than quoting), `check_submission` (columns, ids, decoding, empty masks, overlaps) | Tested |
| `src/data/annotations.py` | Loads the competition JSON into evaluator-format GT, rasterising polygons with pycocotools exactly like the organizers | Tested on a synthetic JSON. **Run it on the real JSON first** (the Data tab says `image_id` is an int in annotations but a string in images; the loader handles both). |
| `tests/reference_official.py` | Verbatim scoring code from the organizers' notebook (Apache 2.0), used as the test oracle | — |
| `tests/test_eval_and_rle.py` | Tests for exact match, IoU = 0.5 not matching, split, merge, multi-annotator scoring, targets, notebook equivalence, transposition, de-overlap, the submission checker and polygon rasterisation | 8/8 pass |

Run the tests (Python 3.10+) from the project folder:

```bash
uv venv --python 3.12 .venv
```

```bash
uv pip install numpy pandas pycocotools torch pytest
```

```bash
.venv/bin/python -m pytest -q
```

Score a validation fold:

```python
from src.data.annotations import load_annotations
from src.eval.pq import evaluate

gt = load_annotations("MAGFiLO_1.0_Kaggle_2026/train/MAGFiLO_1.0_Annotations_kaggle2026_train.json")
val_gt = gt[gt.image_id.isin(val_image_ids)][["filament_id", "segmentation_rle"]]
result = evaluate(val_gt, pred_df)   # pred_df: filament_id = "<image_id>_<n>", segmentation_rle
print(result.summary())
```

### Not verified (check these yourself)
- The annotator multiplicity of the hidden test GT, and therefore exactly how big the local→LB gap will be.
- Whether `hag` archive JPEGs equal the competition JPEGs (decides U6).
- The GPU-hour estimates in section 6.
- Third-party claims of a local PQ of 0.49. Treat ~0.46 local / ~0.41 LB as the honest frontier.

---

## 10. Sources

**Competition (Kaggle):**
- [Data description](https://www.kaggle.com/competitions/filament-segmentation-2026/data)
- [Leaderboard](https://www.kaggle.com/competitions/filament-segmentation-2026/leaderboard)
- [Self-evaluation notebook](https://www.kaggle.com/code/azimahmadzadeh/self-evaluation-notebook)
- Discussion threads:
  - [Organizers' Remarks (724503)](https://www.kaggle.com/competitions/filament-segmentation-2026/discussion/724503)
  - [Test set overlap with public dataset (728062)](https://www.kaggle.com/competitions/filament-segmentation-2026/discussion/728062)
  - [Using additional MAGFiLO annotations (736543)](https://www.kaggle.com/competitions/filament-segmentation-2026/discussion/736543)
  - [Overlapping predictions get ERROR (735377)](https://www.kaggle.com/competitions/filament-segmentation-2026/discussion/735377)
  - [Neighbouring GONG frames (743385)](https://www.kaggle.com/competitions/filament-segmentation-2026/discussion/743385)
  - [Unlabelled GONG pretraining (737215)](https://www.kaggle.com/competitions/filament-segmentation-2026/discussion/737215)
  - [Public metric measurements (729809)](https://www.kaggle.com/competitions/filament-segmentation-2026/discussion/729809)
  - [Missing filaments in annotations (735256)](https://www.kaggle.com/competitions/filament-segmentation-2026/discussion/735256)
  - [Huge gap on the leaderboard (743958)](https://www.kaggle.com/competitions/filament-segmentation-2026/discussion/743958)
  - [Realistic scores (745880)](https://www.kaggle.com/competitions/filament-segmentation-2026/discussion/745880)
  - [Score frontier (741892)](https://www.kaggle.com/competitions/filament-segmentation-2026/discussion/741892)

**Public notebooks:**
- [hdjojo/solar-filament-seg-inference](https://www.kaggle.com/code/hdjojo/solar-filament-seg-inference)
- [vmerckle/simple-dark-filament-segmentation-0-46](https://www.kaggle.com/code/vmerckle/simple-dark-filament-segmentation-0-46)

**Reproducible baseline:**
- [ameyypawar/filament-seg](https://github.com/ameyypawar/filament-seg)

**Papers:**
- [EdgeAttNet, arXiv:2509.02964](https://arxiv.org/abs/2509.02964) (Solomon, Martens, Liu, Angryk, 2025)
- MAGFiLO, Ahmadzadeh et al., *Scientific Data* 11 (2024), [10.1038/s41597-024-03876-y](https://doi.org/10.1038/s41597-024-03876-y)
- Panoptic Segmentation, Kirillov et al., CVPR 2019, [10.1109/CVPR.2019.00963](https://doi.org/10.1109/CVPR.2019.00963) (Theorem 1: unique matching for IoU > 0.5)
- Omnipose, Cutler et al., *Nature Methods* 2022, [10.1038/s41592-022-01639-4](https://doi.org/10.1038/s41592-022-01639-4)
- Skeleton Recall Loss, Kirchhoff et al., ECCV 2024, [arXiv:2404.03010](https://arxiv.org/abs/2404.03010). Note that a 2025 replication ([arXiv:2508.11374](https://arxiv.org/abs/2508.11374)) found no gain over vanilla losses, so do not rely on topology losses.
- Diercke et al., *A&A* 686 (2024), [arXiv:2402.15407](https://arxiv.org/abs/2402.15407): YOLO boxes plus a U-Net for filaments, a detector-then-pixels precedent on GONG data
- Mask Scoring R-CNN, Huang et al., CVPR 2019 (learned mask-IoU scoring)
- Copy-Paste augmentation, Ghiasi et al., CVPR 2021
- Dinkelbach, *Management Science* 13 (1967): fractional programming

**GONG archive:**
- [NSO H-alpha product page](https://nso.edu/nisp/products/h-alpha/)
