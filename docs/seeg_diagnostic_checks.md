# SEEG diagnostic checks

Run commands from the repository root with the training environment activated.
First edit `configs/gen/seeg_pair_list_conditional_debug.yaml` so
`dataset.split_manifest` is `~/RAE_first_exp/patient_split.json`. Keep the existing
image, cache and pair-table paths. These checks do not require rebuilding latents
or changing the architecture, VAE, feature encoder or loss weights.

## 1. Audit condition coverage

```bash
python -m tools.seeg_checks audit \
  --config configs/gen/seeg_pair_list_conditional_debug.yaml \
  --output runs/seeg_condition_audit.json
```

The console summarizes each split. The JSON records each target channel's image
count, eligible donor-channel count, distinct donor patients and selected slots.
Zero-condition targets are counted here but omitted by training. Compare the
histograms across splits, and inspect `test.targets["HHFU073,A6"]`. Two channels
from one donor patient provide less diversity than two channels from two patients.
Patient-disjoint filtering remains enforced. Increasing max_conditions cannot
create additional eligible channels.

## 2. Fit a small, fixed training subset

```bash
python -m tools.seeg_checks overfit-config \
  --config configs/gen/seeg_pair_list_conditional_debug.yaml \
  --output configs/gen/seeg_overfit_local.yaml --targets 32 --steps 5000

python -m tools.seeg_checks audit \
  --config configs/gen/seeg_overfit_local.yaml \
  --output runs/seeg_overfit_audit.json

CUDA_VISIBLE_DEVICES=0 python main.py --gen \
  --config configs/gen/seeg_overfit_local.yaml \
  --workdir runs/seeg_overfit_32
```

Use a new work directory; this starts from scratch. The subset is selected from
eligible training targets using dataset.seed. Donor channels, donor segments and
negative sampling are reproducible for each target. Cached latents bypass pixel
augmentation. The generated noise remains stochastic. Negatives also come from
the small subset, so at least two regions must be selected; if construction
rejects a one-region subset, change dataset.seed or increase --targets.

Validation still uses held-out patients; it is not an overfit score. The preset
raises validation to 32 batches, while preserving batch size. Inspect training
latent L1 and decoded examples of the selected training targets. The audit's
`overfit_targets` lists exact filenames, conditions and inference seeds.

For a selected target, use its listed target_id, target_segment and inference_seed:

```bash
CUDA_VISIBLE_DEVICES=0 python sample_pair_list.py \
  --init-from runs/seeg_overfit_32/params_ema \
  --config configs/gen/seeg_overfit_local.yaml \
  --target-id 'PATIENT,CHANNEL' --target-segment 'EXACT_FILENAME.png' \
  --split train --training-condition-selection --seed INFERENCE_SEED \
  --num-samples 4 --batch-size 1 --outdir runs/seeg_overfit_sample
```

The training-condition-selection flag reproduces the subset's seeded donor-channel
ordering. Without it, ordinary inference selects nearest donor channels.
Expect a substantial training L1 reduction and recovery of visible structure.
If these fail, inspect optimization, latent scales and loss balance before paying
for a long run. Success establishes small-data fit, not held-out generalization.

## 3. Compare raw and EMA parameters at the same step

Export both from a retained training checkpoint, for example the completed 2000-step run:

```bash
python -m tools.seeg_checks export \
  --checkpoint-dir runs/seeg_segments_debug/checkpoints --step 2000 \
  --weights raw --output runs/seeg_weights_2000_raw

python -m tools.seeg_checks export \
  --checkpoint-dir runs/seeg_segments_debug/checkpoints --step 2000 \
  --weights ema --output runs/seeg_weights_2000_ema
```

Both directories use the existing artifact loader format; metadata distinguishes
raw from EMA despite the format's `ema_params.msgpack` filename. Loading requires
the same model config used for training. For overfit checkpoints, use that run's
checkpoint directory and config instead.

```bash
CUDA_VISIBLE_DEVICES=0 python sample_pair_list.py \
  --init-from runs/seeg_weights_2000_raw \
  --config configs/gen/seeg_pair_list_conditional_debug.yaml \
  --target-id 'HHFU073,A6' --target-segment 'EXACT_FILENAME.png' \
  --split test --seed 42 --num-samples 4 --batch-size 1 \
  --outdir runs/seeg_compare_raw
```

Repeat with `--init-from runs/seeg_weights_2000_ema` and
`--outdir runs/seeg_compare_ema`, keeping all other arguments identical.
Raw samples substantially better than EMA suggest short-run EMA lag. If both
are smooth, EMA alone does not explain the failure.

## 4. Test whether image conditions affect generation

Repeat the raw command above with these changes:

```text
--shuffle-condition-images
--outdir runs/seeg_compare_shuffled
```

Keep checkpoint, seed, batch size, target and sample count identical. This replaces
condition images with seeded random images from the same split, retaining the
original mask, coordinates and distances. It excludes the exact target image,
but replacements can come from the same region/channel. `conditions.txt` records
actual images. This is an image-condition sensitivity test, not an ablation of
coordinate conditioning or a guaranteed incompatible-region test.

Compare corresponding sample numbers. Nearly unchanged output suggests weak use
of image conditions, though similar replacement images can also make this test
inconclusive. Repeat on several targets and seeds. Changed output shows sensitivity;
it does not alone demonstrate that the original conditions improve accuracy.

## 5. Decide whether to run longer

If small-data fit works and VAE reconstructions preserve relevant structure,
raise both `train.total_steps` and `optimizer.lr_schedule.total_steps` in a copy
of the full-data config. Use at least 32 validation batches and inspect a fixed
panel of targets periodically. At batch size one, 50,000 usable targets require
roughly 50,000 steps per target traversal; use the audit's retained_segments count
rather than the raw image count. The original 2000-step run is a smoke test.

Monitor latent L1, ordinary and EMA decoded examples, and held-out performance.
`semantic_loss` is a detached feature-similarity monitor, not an optimized term;
about 0.693 means similar average positive/negative similarity. `loss_drift` near
one is expected from its normalization. Neither curve alone diagnoses success.

If small-data fit succeeds but held-out samples remain smooth, evaluate whether
independent cross-patient segments supply enough information to predict exact
target texture. Exact-target L1 can favor average predictions. If the VAE itself
loses clinically relevant detail across representative images, then assess a
spectrogram-specific encoder. Avoid changing all these factors together.
