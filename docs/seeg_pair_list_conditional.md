# SEEG pair-list conditional drifting pipeline

This document summarizes the implementation added for the deadline-oriented
prototype.  It uses the released natural-image VAE temporarily; replacing it
with an SEEG-specific VAE is the highest-priority follow-up.

## 1. Build and validate pair lists

`dataset/pair_list_spectrogram.py` groups every row of
`reduced_pair_table.csv` by `patient_channel`.  A unique target channel becomes
one record and all of its `pair` entries become a variable-length condition
set.  Construction validates required columns, different target/donor
patients, matching regions, coordinate-derived distances, image existence,
and conflicting target metadata.  Duplicate pair channels are removed and
conditions are sorted by distance and channel ID.

Expected full-image layout:

```text
spectrogram_dataset/
  PID1/PID1_A10.png
  PID9/PID9_C10.png
```

Create a reproducible, fully patient-disjoint split:

```bash
python -m dataset.pair_list_spectrogram \
  --table-path ~/reduced_pair_table.csv \
  --image-root ~/spectrogram_dataset \
  --output ~/seeg_patient_split.json \
  --seed 42
```

With 65 patients, a 70/15/15 split is approximately 46/10/9 patients.  The
dataset filters both target and donor roles to the selected split.  Targets
whose condition list becomes empty are omitted.  Inspect the retained target
and pair-list counts before training; patient count alone does not guarantee
that every split retains enough graph edges.

## 2. Load variable-length condition sets

The loader pads or truncates each condition list to `max_conditions` and
returns:

```text
source                 [B,K,C,H,W]
condition_mask         [B,K]
condition_coord        [B,K,3]
condition_distance     [B,K,1]
target_true            [B,C,H,W]
target_ref             [B,1,C,H,W]
negative               [B,N,C,H,W]
target_coord           [B,3]
```

Negatives are targets from different brain regions.  Samples with uncertain
compatibility are not automatically declared negative.

## 3. Temporary natural-image VAE

The released `pcuenq/sd-vae-ft-mse-flax` encodes target, conditions, and
negatives. Pixel tensors are RGB, resized to 256, and normalized to `[-1,1]`;
the VAE produces 32x32x4 latents. Do not encode every variable-length pair list
online during training: that materializes `batch × conditions` full-resolution
VAE activations alongside the generator and MAE and can request tens of GB of
GPU memory. Build the cache once, one image at a time:

```bash
CUDA_VISIBLE_DEVICES=0 python -m dataset.pair_list_latent_cache \
  --table-path ~/reduced_pair_table.csv \
  --image-root ~/spectrogram_dataset \
  --target-root ~/spectrogram_pair_list_latent_cache
```

The config reads those `.npy` latents through `dataset.latent_root`, so the VAE
is not invoked by DataLoader batches. Generator and drifting losses then run in
latent space. Always inspect VAE reconstructions before a long run.
If clinically important spectrogram structure is lost, stop and train a
domain-specific SEEG VAE rather than tuning the generator around a lossy
representation.

The first cache build downloads `pcuenq/sd-vae-ft-mse-flax` from Hugging Face.
The unauthenticated-request message is not an error; set `HF_TOKEN` if the
server's rate limit is a concern. Diffusers also warns that its Flax VAE class
is deprecated. This repository currently requires that class, but the cache
command invokes it outside the long training loop and warning output is
rate-limited. Pin the tested Diffusers version rather than upgrading across its
eventual Flax removal.

The debug config intentionally starts with batch size 1, four conditions, one
negative, and one generated sample. After it completes both training and
validation steps, increase one dimension at a time while watching `nvidia-smi`.
Two visible GPUs do not combine their memory when the command is restricted to
one device with `CUDA_VISIBLE_DEVICES=0`.

## 4. Inject the condition set

Each condition latent is processed by the same two-layer convolutional
encoder.  Its embedding is added to an MLP embedding of:

```text
condition coordinate (3)
relative condition-to-target coordinate (3)
distance (1)
```

Coordinates and distances are divided by `dataset.coordinate_scale` (100 by
default) before the metadata MLP so millimetre-scale values do not dominate
the image embedding.

A learned masked-attention pool produces one permutation-invariant context
vector.  This vector is added to the existing DiT conditioning vector.  The
spatial DiT input remains latent Gaussian noise; pair-list images are not
concatenated as an arbitrary number of channels.

## 5. Condition-aware drifting field

`drift_loss` accepts `positive_mask[B,Cg,Cp]`.  The mask gates the positive
affinity before attractive/repulsive coefficients, force aggregation, and
force normalization.  The first prototype uses the exact true target as the
only compatible positive.  Global generated/negative repulsion remains
unchanged, matching the paper-aligned initial formulation.

The optimized loss is:

```text
lambda_drift * compatibility-masked drift
+ lambda_pair * target latent L1
```

The target is used only during training/validation loss calculation.  It is
not an inference input.

## 6. Run the debug experiment

Update the three paths in
`configs/gen/seeg_pair_list_conditional_debug.yaml`, then run:

```bash
CUDA_VISIBLE_DEVICES=0 python main.py --gen \
  --config configs/gen/seeg_pair_list_conditional_debug.yaml \
  --workdir runs/seeg_pair_list_conditional_debug
```

This is a 2,000-step plumbing/stability run, not a final scientific training
schedule.  Confirm finite gradients, inspect VAE reconstructions and generated
spectrograms, and audit validation patients before increasing the budget.

## 7. Inference

For a held-out target channel, inference receives its region/coordinate and
the spectrograms plus metadata of its in-split pair list.  It does not receive
the true target. Generate from one held-out target's pair list with:

```bash
CUDA_VISIBLE_DEVICES=0 python sample_pair_list.py \
  --init-from runs/seeg_pair_list_conditional_debug/params_ema \
  --config configs/gen/seeg_pair_list_conditional_debug.yaml \
  --target-id 'PID1,A10' \
  --outdir runs/seeg_pair_list_samples/PID1_A10 \
  --num-samples 16
```

The true target may be loaded separately afterward to calculate MAE, SSIM,
marginal spectral errors, and comparison plots.

## 8. Required follow-ups

1. Save and report retained targets/pair counts per patient split.
2. Train a spectrogram-specific VAE and compare reconstruction fidelity.
3. Add region-stratified reconstruction metrics and condition-shuffling tests.
4. Compare exact-target masking with broader spatial compatibility masks.
5. Add cross-attention over condition tokens only after pooled conditioning is
   stable.
