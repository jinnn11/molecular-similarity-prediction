# Molecule Similarity Prediction

Tri-modal contrastive pretraining (2D images, 3D conformers, 1D fingerprints) on ~292K GEOM-Drugs molecules, followed by frozen-encoder similarity regression on 200 expert-labeled molecule pairs. The downstream stage uses cosine similarities across modalities and a tiny regression head to predict a human similarity score in the 0–1 range.

**Result: the learned embeddings do not outperform standard fingerprint similarity.** Averaging the three embedding cosines (no training on labels) reaches Pearson 0.90 with expert scores. ECFP4 Tanimoto reaches 0.88, and the paired difference is not significant (95% CI −0.01 to +0.05). The best trained head (`cosine_mlp`, 0.91) was picked from eleven downstream configurations tried on these same pairs. It matches FCFP4 and atom-pair Tanimoto, which involve no learning at all.

An earlier version of this README claimed the model outperformed "Tanimoto" (0.92 vs 0.85). That baseline was ROCS TanimotoCombo, a 3D shape-overlap score, and it is the weakest of the baselines below. See [Results](#results).

## Architecture

### Pretraining
![Pretraining diagram](diagram/diagram_pretraining.png)

### Similarity Regression (Frozen Encoders)
![Similarity Regression](diagram/diagram_similarity_boxed.png)

The final prediction head is a lightweight MLP operating on cosine similarity outputs from the frozen encoders:

- Inputs: cosine similarities computed from frozen 2D/3D/1D embeddings.
- Architecture: `Linear(3 → 16) → ReLU → Linear(16 → 1)` (regression, no sigmoid).

## Highlights
- Tri-tower contrastive pretraining (ResNet-18, SchNet, MLP)
- Frozen-encoder downstream evaluation with cosine similarities
- Downstream evaluation on 200 expert-labeled pairs with 5-fold cross-validation
- Experiment tracking with plots, logs, and JSON summaries

## Data
- Pretraining: GEOM-Drugs (drugs_crude.msgpack) with one conformer per molecule and atom-count filtering.
- Downstream: 200 labeled similarity pairs in `dataset_Similarity_Prediction/`.
- The raw datasets and generated run artifacts are intentionally not versioned in this portfolio snapshot.

## Setup

Recommended Python: 3.10

```
conda env create -f environment.yml
conda activate simpred
```

Alternative:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Quick Validation

```bash
python3 -m pytest
python3 finetuning/cosine_baseline.py --help
python3 results/aggregate_results.py --help
```

## Reproducing The Project

You will need to supply the two local data sources below before running the full pipeline:

- GEOM-Drugs archive for pretraining, passed to `pretraining/precompute_drugs.py --archive ...`
- The expert-labeled downstream CSVs and associated 3D conformers under `dataset_Similarity_Prediction/`

The repository keeps the source code, diagrams, and curated summary results in `results/`, but does not commit raw datasets, precomputed tensors, or fold-by-fold training artifacts.

## Precompute (Pretraining Inputs)

```
python pretraining/precompute_drugs.py \
  --archive /path/to/drugs_crude.msgpack \
  --out pretraining_data/drugs \
  --max-atoms 100 \
  --image-size 384
```

## Pretraining

```
python pretraining/train_pretrain.py \
  --data-dir pretraining_data/drugs \
  --image-size 384 \
  --batch-size 64 \
  --accumulation-steps 4 \
  --epochs 100 \
  --device cuda
```

Optional: evaluate cosine signal during pretraining

```
python pretraining/train_pretrain.py \
  --data-dir pretraining_data/drugs \
  --image-size 384 \
  --batch-size 64 \
  --accumulation-steps 4 \
  --epochs 100 \
  --device cuda \
  --eval-csv dataset_Similarity_Prediction/original_training_set/original_training_set.csv \
  --eval-csv dataset_Similarity_Prediction/new_dataset/new_dataset.csv \
  --eval-every 5
```

Outputs land in `outputs/pretrain_runs/run_*/`.

## Feature Extraction (Frozen Encoders)

```
python finetuning/extract_features.py \
  --csv dataset_Similarity_Prediction/original_training_set/original_training_set.csv \
  --csv dataset_Similarity_Prediction/new_dataset/new_dataset.csv \
  --weights-dir outputs/pretrain_runs/<RUN_ID> \
  --image-size 384 \
  --out outputs/finetune_features/features.npz
```

## Similarity Regression (Cosine Baselines)

Average cosine:

```
python finetuning/cosine_baseline.py \
  --features outputs/finetune_features/features.npz \
  --mode avg \
  --experiment-name cosine_avg
```

Linear cosine with Tanimoto:

```
python finetuning/cosine_baseline.py \
  --features outputs/finetune_features/features.npz \
  --mode linear \
  --include-tanimoto \
  --experiment-name cosine_linear_tani
```

All results are saved under `outputs/finetune_runs/<experiment_name>/run_*/`.

## Aggregate Results

```
python results/aggregate_results.py
```

Produces:
- `results/summary_table.csv`
- `results/summary_table.md`
- `results/metrics_overview.png`
- `results/pearson_vs_tanimoto.png`

In these files, "tanimoto" means ROCS TanimotoCombo, not fingerprint Tanimoto. See [Results](#results) for the fingerprint baselines.

## Repository Layout

- `pretraining/`: precompute + contrastive training
- `finetuning/`: feature extraction + similarity regression
- `results/`: aggregated result tables and plots
- `diagram/`: architecture diagrams
- `tests/`: lightweight validation for utility code
- `outputs/`: local run artifacts, ignored by git

## Reproducibility

- Most scripts accept `--seed`.
- Run configs and metrics are saved alongside each run.

## Notes

- The downstream stage is best described as **feature extraction + similarity regression** because encoders are frozen.
- If you unfreeze encoders, it becomes true finetuning.

## Results

All numbers are on the 200 expert-labeled pairs (100 in each CSV). Trained heads are scored on pooled out-of-fold predictions from 5-fold CV. Brackets are 95% bootstrap intervals from 10,000 resamples of the pairs. The same resamples are used for every predictor, so differences are paired. The intervals do not account for having picked the best head on these same pairs.

### Fingerprint baselines vs learned embeddings

| Predictor | Fit to labels | Pearson | Spearman |
| --- | --- | --- | --- |
| ROCS TanimotoCombo (3D shape + color; ships with the dataset) | No | 0.855 [0.816, 0.889] | 0.851 [0.810, 0.881] |
| ECFP4 Tanimoto (RDKit, 2048 bits) | No | 0.883 [0.852, 0.911] | 0.871 [0.834, 0.897] |
| CDK Extended Tanimoto (ships with the dataset) | No | 0.885 [0.854, 0.913] | 0.863 [0.824, 0.890] |
| FCFP4 Tanimoto | No | 0.914 [0.887, 0.935] | 0.896 [0.865, 0.916] |
| Atom-pair Tanimoto | No | 0.914 [0.892, 0.934] | 0.903 [0.869, 0.924] |
| `cosine_avg`: mean of the 2D/3D/1D embedding cosines | No | 0.899 [0.875, 0.921] | 0.896 [0.866, 0.917] |
| `cosine_mlp`: MLP on the three cosines | Yes | 0.912 [0.886, 0.934] | 0.884 [0.850, 0.907] |

| Paired difference | Δ Pearson | Δ Spearman |
| --- | --- | --- |
| `cosine_avg` − ECFP4 | +0.016 [−0.012, +0.045] | +0.025 [−0.001, +0.052] |
| `cosine_mlp` − ECFP4 | +0.028 [+0.002, +0.055] | +0.013 [−0.013, +0.040] |
| `cosine_avg` − ROCS TanimotoCombo | +0.044 [+0.010, +0.082] | +0.045 [+0.011, +0.083] |

- `cosine_mlp`'s Pearson interval clears zero by 0.002. Its Spearman interval does not, and neither interval reflects that it was the best of eleven downstream configurations.
- FCFP4 and atom-pair were the best of six fingerprints tried, so they carry the same selection bias. ECFP4 is the pre-specified baseline.
- Pearson is unchanged by linear rescaling, so a raw Tanimoto score and a trained regressor can be compared directly on correlation.

### Per CSV

`cosine_avg` has no parameters fit to labels, so each CSV is an independent test of it. `new_dataset` has 25 pairs of each type: 2D-similar or 2D-dissimilar, crossed with 3D-similar or 3D-dissimilar.

| Predictor (Pearson) | `original_training_set` | `new_dataset` |
| --- | --- | --- |
| ECFP4 Tanimoto | 0.893 | 0.897 |
| CDK Extended Tanimoto | 0.935 | 0.830 |
| ROCS TanimotoCombo | 0.923 | 0.799 |
| `cosine_avg` | 0.927 | 0.866 |
| `cosine_mlp` (out-of-fold) | 0.942 | 0.877 |

- On `new_dataset`, `cosine_avg` scores below ECFP4: Δ Pearson −0.031 [−0.070, +0.007], Δ Spearman −0.039 [−0.084, −0.001].
- On `original_training_set`, it scores above ECFP4 (Δ Pearson +0.034 [−0.004, +0.074]) and level with the two Tanimoto scores that ship with the dataset (vs ROCS: +0.004 [−0.024, +0.033]).
- The pooled ROCS figure (0.855) is lower than both per-CSV figures because the relationship between that score and the labels differs between the two files.

### Each modality, and what pretraining contributes

Pearson of each embedding cosine on all 200 pairs, compared with the same encoders before GEOM-Drugs pretraining. Inputs are identical (same depictions, fingerprints and conformers). Random-init values are means over 5 seeds.

| Embedding | Random init | ImageNet init | After pretraining |
| --- | --- | --- | --- |
| 2D (ResNet-18 on depictions) | 0.379 | 0.683 | 0.889 |
| 3D (SchNet on conformers) | 0.555 | – | 0.874 |
| 1D (MLP on ECFP4 bits) | 0.873 | – | 0.891 |
| Mean of the three | 0.884 | 0.894 (ImageNet 2D, random 1D/3D) | 0.899 |

- Pretraining makes the 2D and 3D embeddings track expert scores far better than random or ImageNet weights.
- The 1D tower does not improve on its input: the plain cosine of the ECFP4 bits it reads scores 0.901.
- The fused score is dominated by the fingerprint signal. With no GEOM-Drugs pretraining, the mean of the three cosines already scores 0.894, against 0.899 pretrained. On `new_dataset`, the untrained mean (0.905–0.915) scores higher than the pretrained one (0.866).
- Dropping any one modality from the mean moves Pearson by at most 0.01 (1D+2D 0.899, 1D+3D 0.895, 2D+3D 0.889).

### Molecule-level leakage

58 of the 309 unique molecules appear in more than one pair. Rerunning CV with folds grouped by shared molecules (120 groups, mean over 10 fold assignments) barely moves the trained heads:

| Head (Pearson) | Pair-split CV | Molecule-grouped CV |
| --- | --- | --- |
| `cosine_linear` | 0.898 | 0.895 |
| `cosine_mlp` | 0.917 | 0.910 |
| `cosine_linear_tani` | 0.915 | 0.909 |

The heads see only three or four similarity scores per pair, so there is little to memorize.

### Evaluation hygiene

- The pretraining run behind these numbers (`run_20260122_211530`) was launched without `--eval-csv`, so the 200 pairs were not monitored during pretraining. Its last checkpoint is used.
- All eleven downstream configurations were compared on the same 200 pairs, so the best one's score is optimistically biased. `cosine_avg` has nothing fit to labels.

## Limitations

- **Small evaluation set.** With 200 pairs, the intervals above are 0.04–0.07 wide (wider per CSV), too wide to separate most of the predictors.
- **Pair-specific 3D inputs.** The 3D tower reads `best_rocs_conformer_*.pdb`, conformers that ROCS chose to maximize overlap within each specific pair. They would not exist when screening new compounds, and pretraining used each molecule's lowest-energy GEOM conformer instead. The pretrained encoders have not been re-evaluated with independently generated conformers (`extract_features.py --no-pdb`).
- **One pretraining run.** All results come from a single pretraining seed.

## Conclusions

- On these 200 expert-labeled pairs, tri-modal contrastive embeddings do not outperform standard 2D fingerprint similarity. The label-free cosine average is statistically indistinguishable from ECFP4 Tanimoto, and the best trained head matches FCFP4 and atom-pair Tanimoto.
- The earlier claim of beating Tanimoto (0.92 vs 0.85) compared the best of eleven configurations against ROCS TanimotoCombo, the weakest baseline tested.
- Contrastive pretraining raises the 2D and 3D embeddings' agreement with experts (0.38 → 0.89 and 0.56 → 0.87 Pearson), but fusing them adds little beyond the fingerprint signal.
- Siamese heads trained on the full embeddings (`train_siamese.py`) overfit on 200 pairs (Pearson 0.35–0.47).
