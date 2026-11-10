# Transfer test: Vientiane, human-labelled (2026-11-09)

Models trained **only on Phnom Penh weak labels** (seed-0 weights from `artifacts/`), evaluated on 285
human-labelled Vientiane patches. No Vientiane data was used for training or model selection.

## Data

- `vientiane_qc_labels.csv`: 300 QC patches sampled **100 per weak-label class** (stratified on the rule label,
  not random). Columns: `weak_label` (same Dynamic World rule as Phnom Penh), `human_label` (one annotator, blind;
  the annotator's class `persistent` = `urban`), `lat`, `lon`.
- 15 patches marked `unsure` are excluded → **285 scored patches** (human: 110 stable / 77 seasonal / 98 urban).
- Run: SageMaker job `vientiane-1110-050658`, `src/predict.py predict` (see README §5b). Presto used lat/lon
  extrapolated from the Phnom Penh grid (~0.07° lon error); the true coordinates are in the csv.

## 1. How good is the weak-label rule itself? (rule vs human, n = 285)

| rule \ human | seasonal | stable | urban |
|---|---|---|---|
| seasonal | 70 | **21** | 1 |
| stable | 7 | 83 | 4 |
| urban | 0 | 6 | 93 |

Macro-F1 **0.862**, accuracy 0.863, Cohen's κ 0.795; 39 disagreements. Per-class F1: stable 0.814,
seasonal 0.828, urban 0.944. Main rule error: stable patches called seasonal.

## 2. Models vs human labels and vs the rule

| Model | F1 vs human [95% CI] | F1 vs rule | On the 39 rule≠human patches: agrees with human / rule |
|---|---|---|---|
| `R18_ssl4eo_delta` | **0.785** [0.733, 0.830] | 0.867 | 7 / 30 |
| `M2_cnn_k5_delta` | 0.769 [0.720, 0.816] | 0.862 | 6 / 33 |
| `M2_cnn_k3_delta` | 0.759 [0.706, 0.806] | 0.856 | 5 / 33 |
| `M2_cnn_k1` | 0.752 [0.697, 0.799] | 0.825 | 9 / 30 |
| `R18_scratch_delta` | 0.722 [0.666, 0.776] | 0.765 | 14 / 25 |
| `Presto_pretrained` | 0.707 [0.654, 0.758] | 0.801 | 5 / 33 |
| `M1_gru` | 0.701 [0.647, 0.749] | 0.762 | 10 / 28 |
| `M2_cnn_k1_delta` | 0.669 [0.608, 0.728] | 0.672 | 20 / 18 |
| `Presto_scratch` | 0.616 [0.554, 0.675] | 0.645 | 16 / 22 |
| `M0_hgb` | 0.574 [0.517, 0.628] | 0.624 | 9 / 26 |
| `M1_gru_delta` | 0.511 [0.449, 0.569] | 0.583 | 7 / 30 |
| `M0_rf` | 0.472 [0.409, 0.528] | 0.522 | 11 / 26 |

Full metrics and confusion matrices: `METRICS.md`, `metrics.json`; per-patch probabilities: `preds_<model>.csv`.

## Observations

1. **Cross-city transfer separates models that look identical in-domain.** On Phnom Penh all models score
   0.95–0.98; on Vientiane hand-crafted features (M0) drop to 0.47–0.57, the temporal-only GRU to 0.70, and
   spatial CNNs reach 0.75–0.79.
2. **Pretraining helps transfer** (single seed, CIs overlap): ResNet-18 SSL4EO 0.785 vs scratch 0.722;
   Presto pretrained 0.707 vs scratch 0.616. In-domain the pretrained and scratch versions were tied.
3. **Difference-from-first-year inputs hurt transfer** (M2 0.752 → 0.669, GRU 0.701 → 0.511) although they were
   neutral in Phnom Penh — hypothesis (untested): the first-year baseline level differs between cities.
4. **The models learned the rule, not the human notion of change.** Where rule and human disagree, the best
   models side with the rule 30–33 times out of 39. Their agreement with the rule (≈0.86) equals the rule's own
   agreement with humans (0.862).
5. For the best model the gap from in-domain to human labels splits roughly in half:
   Phnom Penh weak 0.97 → Vientiane weak 0.867 (**domain shift ≈ −0.10**) → Vientiane human 0.785
   (**rule-vs-human label gap ≈ −0.08**). Training on Vientiane weak labels can at best close the first part
   (ceiling ≈ 0.86 vs human); exceeding the rule needs better labels.

## Caveats

- The QC sample is stratified on the weak label and excludes `unsure`, which favours the rule; numbers are not
  city-wide accuracy estimates.
- One annotator, no inter-annotator agreement yet.
- Single seed per model; with n = 285 the CI half-width is ≈ 0.05, so differences < 0.05 between models are not
  significant without a paired test.
