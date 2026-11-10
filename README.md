# Land-cover change in Phnom Penh from monthly Sentinel-1/2 time series

CS230 (Stanford, Autumn 2026) team project. Given 72 months (Jan 2017 – Dec 2022) of Sentinel-2 + Sentinel-1
imagery for a 320 m × 320 m patch, classify what happened to it:

| Class | Meaning |
|---|---|
| `stable` | no lasting change (includes "always built-up" and permanent water) |
| `seasonal` | recurring flooding / water in most years, no permanent conversion |
| `urban` | new built-up area that persists |

**Status (2026-11-09):** the full experiment ladder (all 12 models x 3 seeds) has finished on AWS SageMaker. All scores are still on **weak labels**, pending the gold (human-verified) set the team is preparing. A self-checked inference bundle (weights, normalisation stats, M0 baselines) is in [`artifacts/`](artifacts/), see section 5b.

---

## 1. Data

Not included in the repo (≈2.5 GB, owned by the team's Google Drive: `patches/s2`, plus `labels.csv`).

| Item | Details |
|---|---|
| Grid | 80 × 80 patches over Phnom Penh, id `phnom_penh_{col}_{row}` (col 1499–1578, row 3954–4033). Row `r+1` is north of row `r`; col `c+1` is east of col `c` (verified from image-seam continuity). |
| Patch file | `int16` array `(72, 8, 32, 32)` = months × channels × 10 m pixels |
| Channels | 0–3: S2 B2, B3, B4, B8 (reflectance × 1e4, 0 = no clear obs) · 4–5: S1 VV, VH (dB × 100) · 6: S2 obs count · 7: S1 obs count |
| Labels | `labels.csv`, 6400 patches. Rule-based **weak labels** derived from Dynamic World by a teammate (`reason` column). 2669 ambiguous `drop` patches are excluded, leaving **3731** labelled patches (1710 stable / 1069 seasonal / 952 urban). |
| Pre-processing | Cloudy S2 months forward/backward filled per pixel; NDVI and NDWI added → 10 channels; per-channel z-score from training patches. |

Note: the Drive export only contains the 3731 labelled patches, so in the context-window models (§3) neighbours
that are `drop` patches are zero-filled.

## 2. Evaluation protocol

- **Spatial block split**: 8 × 8-patch blocks, randomly assigned 70 / 15 / 15 % to train / val / test (seed 0).
- **Buffer**: train/val patches within 1 patch of any test patch are removed (229 patches) to avoid
  spatial-adjacency leakage. Final sizes: train 2389, val 499, test 614.
- Model selection (best epoch) on **val** macro-F1; test is evaluated once.
- Metrics: test macro-F1 with 1000× bootstrap 95 % CI; per-class F1; macro-F1 on **interior** patches (all 8 neighbours
  share the label) vs **boundary** patches (≥ 50 % of neighbours differ). The error analysis showed that most errors
  happen on boundaries (city fringe, river banks), so this split is reported separately.
- 3 seeds per model (seed-0 pass runs first for every model, then seeds 1, 2).
- `results/preds/*.csv` stores class probabilities for **every** patch, so the gold set can be scored later without
  retraining.

The exact split is in `results/split.csv` (`split` ∈ train / val / test / buffer, plus `nb_diff` = fraction of
neighbours with a different label).

## 3. Models

| Name | Input | Model | Question it answers |
|---|---|---|---|
| `M0_rf`, `M0_hgb` | patch-mean series → 150 hand-crafted stats (annual means, first/last year, trend, seasonality) | Random Forest / Gradient Boosting | strong non-deep baseline |
| `M1_gru` (`_delta`) | patch-mean series 72 × 10 (+ difference from first-year mean) | 2-layer BiGRU + attention pooling | does a learned temporal model beat hand-crafted stats? |
| `M2_cnn_k1` (`_delta`) | full 72 × 10 × 32 × 32 cube | per-frame 6-layer CNN → temporal Transformer (2 layers, d=128, month + position embeddings) | does spatial structure inside the patch help? |
| `M2_cnn_k3_delta`, `M2_cnn_k5_delta` | 3 × 3 / 5 × 5 neighbourhood window (96 / 160 px) | same, features pooled globally + over the centre patch | does surrounding context help? |
| `R18_scratch_delta` vs `R18_ssl4eo_delta` | 32 × 32 frames upsampled × 2 | ResNet-18 frame encoder (random init vs **SSL4EO-S12 MoCo** Sentinel-2 weights; B2/B3/B4/B8 conv1 filters copied) → temporal Transformer | does remote-sensing pretraining help? |
| `Presto_scratch` vs `Presto_pretrained` | 16 sub-patch (8 × 8 px) mean series, 72 months = 3 × 24-month chunks; missing bands masked | **Presto** encoder (pretrained S1/S2 pixel-time-series foundation model) + MLP head on mean/std of embeddings | does a time-series foundation model help? |

`_delta` = 8 extra channels: value minus the patch's first-year mean (targets the "always built-up patch predicted as
urban" error found in the milestone error analysis).

Training: AdamW (wd 0.05), one-cycle LR, label smoothing 0.1, √-inverse-frequency class weights, mixed precision,
random flips / 90° rotations for image models. Hyper-parameters per model are in the `EXPS` list in
[`src/run_all.py`](src/run_all.py).

## 4. Results (weak-label test set, mean ± sd over 3 seeds)

Test macro-F1 is mean ± sd over seeds 0, 1, 2; the CI column is the mean of the per-seed 95 % bootstrap CIs; interior / boundary F1 are seed means. *Seed ens.* = test macro-F1 of the 3-seed averaged probabilities. Per-class F1 and confusion matrices: [`results/RESULTS.md`](results/RESULTS.md).

| Model | Test macro-F1 | 95 % CI (mean) | Interior F1 | Boundary F1 | Seed ens. |
|---|---|---|---|---|---|
| `M0_rf` | 0.958 ± 0.002 | [0.940, 0.974] | 0.970 | 0.814 | 0.958 |
| `M0_hgb` | 0.963 ± 0.000 | [0.948, 0.977] | 0.974 | 0.809 | 0.963 |
| `M1_gru` | 0.961 ± 0.001 | [0.943, 0.976] | 0.967 | 0.846 | 0.967 |
| `M1_gru_delta` | 0.953 ± 0.001 | [0.933, 0.970] | 0.957 | 0.840 | 0.959 |
| `M2_cnn_k1` | 0.977 ± 0.003 | [0.962, 0.990] | 0.980 | 0.945 | 0.985 |
| `M2_cnn_k1_delta` | 0.980 ± 0.003 | [0.968, 0.991] | 0.987 | 0.908 | 0.980 |
| `M2_cnn_k3_delta` | 0.971 ± 0.005 | [0.956, 0.984] | 0.989 | 0.830 | 0.976 |
| `M2_cnn_k5_delta` | 0.974 ± 0.004 | [0.960, 0.985] | 0.990 | 0.832 | 0.978 |
| `R18_scratch_delta` | 0.972 ± 0.006 | [0.956, 0.986] | 0.984 | 0.881 | 0.986 |
| `R18_ssl4eo_delta` | 0.969 ± 0.006 | [0.953, 0.983] | 0.981 | 0.845 | 0.978 |
| `Presto_scratch` | 0.957 ± 0.004 | [0.938, 0.974] | 0.965 | 0.866 | 0.963 |
| `Presto_pretrained` | 0.970 ± 0.004 | [0.954, 0.984] | 0.981 | 0.900 | 0.973 |

Observations (only from the numbers above):

1. **M0 vs M1 vs M2.** The frame-CNN M2 family (0.971-0.980) is ~1.5-2.5 points above the hand-crafted M0 baselines (0.958 / 0.963) and the patch-mean BiGRU M1 (0.961 / 0.953). M0 and M1 are indistinguishable. The M2 vs M0 gap is several seed-sd, but the bootstrap CIs still overlap (e.g. M2_cnn_k1_delta [0.968, 0.991] vs M0_hgb [0.948, 0.977]), so it is a consistent trend on this test set rather than a proven gap.
2. **Context windows did not help.** 3x3 (0.971 ± 0.005) and 5x5 (0.974 ± 0.004) are not better than the single-patch M2_cnn_k1_delta (0.980 ± 0.003). Interior F1 is marginally higher (0.989 / 0.990 vs 0.987) but boundary F1 drops (0.830 / 0.832 vs 0.908), which is consistent with neighbouring "drop" patches being zero-filled; the overall differences are within ~1-2 seed sd.
3. **ResNet18: SSL4EO pretraining gave no gain** (0.969 ± 0.006 vs 0.972 ± 0.006 from scratch; within seed sd).
4. **Presto: pretraining helps** (0.970 ± 0.004 vs 0.957 ± 0.004, ~3 sd), the only clear pretraining effect here, but CIs overlap ([0.954, 0.984] vs [0.938, 0.974]) and pretrained Presto does not beat the scratch M2 CNNs.
5. **Difference-from-first-year inputs: no consistent effect.** M2: 0.980 vs 0.977 (within sd); M1: 0.953 vs 0.961 (lower).
6. **Interior vs boundary.** All models score 0.957-0.990 on interior patches but 0.80-0.95 on boundary patches (n ~ 62, so this column is noisy; one patch moves it by ~1-2 points). Best boundary F1: M2_cnn_k1 (0.945). Seed ensembling adds 0-1.4 points for the deep models (best: R18_scratch_delta 0.986, M2_cnn_k1 0.985).
7. These are weak labels generated from Dynamic World statistics; a high score shows agreement with the rule, not necessarily correct change detection. The gold set will answer that.

Milestone numbers are **not** directly comparable: this repo uses a new split with a leakage buffer.

## 5. Reproduce

### Option A — any machine with a GPU

```bash
pip install torch torchvision scikit-learn pandas einops
# put the s2 zips (or loose .npy files) and labels.csv in data/raw/
python src/run_all.py --raw data/raw --out runs/local --work /tmp/work            # everything, 3 seeds
python src/run_all.py --raw data/raw --out runs/local --only M2_cnn_k1 --seeds 0   # a single model
python src/run_all.py --raw data/raw --out runs/smoke --smoke 1                    # 400 patches, 1 epoch
python scripts/summarize.py runs/local
```

Runs are resumable: anything already in `<out>/results.jsonl` is skipped. Needs ≈ 8 GB RAM (all 3731 patches are held
in memory). The SSL4EO weights are downloaded from Hugging Face on first use; Presto weights ship in
`src/presto_default_model.pt` (MIT licence, from [nasaharvest/presto](https://github.com/nasaharvest/presto)).

### Option B — AWS SageMaker (what we used)

One `ml.g5.xlarge` (A10G 24 GB), PyTorch 2.5.1 DLC, on-demand ≈ $1.41/h.

```bash
# in AWS CloudShell (us-west-2)
aws s3 cp <zips + labels.csv> s3://<bucket>/raw/
tar czf cs230_code.tar.gz -C src . && aws s3 cp cs230_code.tar.gz s3://<bucket>/code/
python3 aws/launch.py full 20 seeds=0,1,2      # job name prefix, max hours, hyper-parameters → run_all.py args
aws/mon.sh <job-name> 40                       # status + last log lines
aws s3 sync s3://<bucket>/runs/full/ckpt/ results/   # results are checkpointed to S3 continuously
```

Edit the bucket / role ARN at the top of `aws/launch.py`. A role with `AmazonSageMakerFullAccess` and S3 access is
required.

## 5b. Evaluate on a new, human-labelled dataset

`artifacts/` holds everything needed for inference: `norm_stats.json` (the exact normalisation used in
training), seed-0 weights of every deep model (`*_s0.pt`), the refitted M0 baselines (`M0_*.joblib`) and
`export_selfcheck.json` (re-predicting the original test split with these files reproduces the training-time
probabilities).

New data must use the same export: `<prefix>_{col}_{row}.npy`, `int16 (72, 8, 32, 32)`, same channels and
scaling, 72 monthly composites starting in **January**. The 3 × 3 / 5 × 5 models also need the neighbouring
patches in the same folder (missing neighbours are zero-filled).

```bash
# gold.csv: patch_id,label[,lat,lon]   label ∈ stable|seasonal|urban (other values, e.g. uncertain, are not scored)
python src/predict.py predict --data new_patches/ --labels gold.csv --artifacts artifacts --out eval_gold
# -> eval_gold/METRICS.md (macro-F1 + bootstrap CI, per-class F1, confusion), preds_<model>.csv

# or on SageMaker (upload new_patches/ incl. gold.csv to s3://<bucket>/gold/)
python3 aws/launch_any.py gold 1 predict.py ml.g5.2xlarge \
    new=s3://<bucket>/gold/ artifacts=s3://<bucket>/runs/export/ckpt/ -- mode=predict labels=gold.csv out=/opt/ml/checkpoints
```

If the gold patches are a subset of the original 3731, no inference is needed: join `gold.csv` with
`results/preds/<run>.csv` on `patch_id` (this also gives all 3 seeds). Only patches in the **test** split
(`results/split.csv`) give an unbiased score.

`artifacts/` is regenerated with `python src/predict.py export --raw data/raw --weights results/checkpoints
--out artifacts`.

## 6. Repository layout

```
src/
  run_all.py          experiment runner (data → split → all models → metrics → summary)
  predict.py          export inference artifacts / evaluate saved models on new labelled patches
  data.py             patch loading, cloud filling, neighbour windows, spatial split, M0 features
  models.py           GRU, CNN + temporal Transformer, ResNet-18 (SSL4EO), Presto classifier
  presto_vendor.py    single-file Presto (MIT, nasaharvest/presto)
  presto_default_model.pt
aws/                  SageMaker launch + monitor scripts
scripts/summarize.py  results.jsonl → summary.csv + RESULTS.md
sync_results.sh      pull latest results/logs/weights from S3 and rebuild RESULTS.md
results/
  results.jsonl       one line per run: config, val/test metrics, confusion matrix
  summary.csv         per-model aggregates over seeds
  RESULTS.md          human-readable table
  split.csv           exact train/val/test/buffer assignment
  preds/<run>.csv     class probabilities for every patch
  log.txt             training log (per-epoch loss / val F1)
  checkpoints/        seed-0 model weights (.pt)
artifacts/            inference bundle: norm_stats.json, *_s0.pt, M0_*.joblib, export_selfcheck.json
```

## 7. Known limitations / next steps

- Weak labels only → gold-set evaluation (stratified, blind, from test blocks) is the next step.
- Context-window models see zero-filled `drop` neighbours; exporting all 6400 patches would fix this.
- Single spatial split; block-level cross-validation would give tighter uncertainty.
- Year range (2017–2022) inferred from `onset_year` and the export file names; confirm with the data owner.
