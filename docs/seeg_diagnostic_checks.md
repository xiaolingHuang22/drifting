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

## 6. Batch-check the completed 5000-step overfit run

These commands reuse existing checkpoints and caches; no training is required.
Use the exact overfit config and its audit from the run being investigated.

```bash
python -m tools.seeg_checks export \
  --checkpoint-dir runs/seeg_overfit_32/checkpoints --step 5000 \
  --weights raw --output runs/seeg_overfit_5000_raw

python -m tools.seeg_checks export \
  --checkpoint-dir runs/seeg_overfit_32/checkpoints --step 5000 \
  --weights ema --output runs/seeg_overfit_5000_ema

CUDA_VISIBLE_DEVICES=0 python -m tools.seeg_compare \
  --config configs/gen/seeg_overfit_local.yaml \
  --audit runs/seeg_overfit_audit.json \
  --raw runs/seeg_overfit_5000_raw --ema runs/seeg_overfit_5000_ema \
  --targets 4 --num-samples 4 --outdir runs/seeg_overfit_comparison
```

Adjust the audit path if you saved it elsewhere. This command launches separate
inference processes for raw, EMA and raw-with-shuffled-images for each selected
training target. Each uses identical noise seeds, batch size one and the audited
fixed donor selection. It preserves coordinates/distances in the shuffled case.
Outputs are organized by target and variant, including conditions.txt, PNGs and
unquantized latent arrays. Existing output files at those paths are overwritten;
use a fresh output directory for each experiment.

`comparison.json` reports generated latent L1 averaged over samples against the
exact cached target, plus zero-latent, training-subset mean-latent and median-latent
L1 baselines. The mean/median use all audited overfit targets, not the held-out
set. These are memorization sanity baselines; they are not independent validation
scores. Median is included because elementwise L1 favors the median. The saved
mean and median arrays can also be decoded with the existing VAE if needed.

If generated L1 does not beat these constant baselines and outputs are smooth,
the run has not demonstrated target-specific fitting. A small raw/shuffled latent
difference supports weak image sensitivity on those examples; there is no universal
threshold. Compare it with target magnitudes and repeat across targets. A larger
difference shows sensitivity, not necessarily useful conditioning.

## 7. Record conditioning gradients and actual optimizer updates

In a copy of the overfit YAML, set:

```yaml
train:
  log_update_diagnostics: true
```

Retain all the other train settings; do not replace the whole train section.
Run in a new directory:

```bash
CUDA_VISIBLE_DEVICES=0 python main.py --gen \
  --config configs/gen/seeg_overfit_diagnostics_local.yaml \
  --workdir runs/seeg_overfit_diagnostics
```

Save your edited copy at the config path above first. Each metrics.jsonl record
now includes `condition_diag/MODULE/grad_norm`, `param_norm`, `update_norm` and
`relative_update`. Modules cover both condition convolutions, image projection,
metadata projection, attention score and output projection. Gradients are measured
before clipping; update norms are actual new-minus-old parameters after the
optimizer, including weight decay. A nonzero update with zero gradients can arise
from weight decay or optimizer momentum and does not establish useful learning.

Persistent zero gradients require investigating the gradient path; persistently
small relative updates can explain slow adaptation. Nonzero gradients alone do not
prove useful condition use. The model currently scales its pooled condition signal
by 0.02; this change records diagnostics without altering that scale, architecture,
loss weights or encoders. Assess the measurements before changing those factors.

## 8. Paired-L1-only fitting experiment (original VAE)

Generate a fresh config from the exact 32-target config used previously. This
preserves its seed, subset, model, VAE, optimizer settings and fixed donors, while
setting lambda_pair=1 and all other objective weights to zero. Do not resume a
mixed-objective checkpoint for this comparison.

```bash
python -m tools.seeg_checks overfit-config \
  --config configs/gen/seeg_overfit_local.yaml \
  --output configs/gen/seeg_pair_only_local.yaml \
  --targets 32 --steps 5000 --pair-only

python -m tools.seeg_checks audit \
  --config configs/gen/seeg_pair_only_local.yaml \
  --output runs/seeg_pair_only_audit.json

CUDA_VISIBLE_DEVICES=0 python main.py --gen \
  --config configs/gen/seeg_pair_only_local.yaml \
  --workdir runs/seeg_pair_only_32
```

Feature diagnostics still run, so semantic_loss may be logged and this does not
eliminate feature-extractor memory use. Only paired latent L1 contributes to the
optimized objective. Use loss_pair, raw decoded samples and constant baselines
to judge fitting; validation is still held out and need not improve during
memorization. Conditioning update diagnostics are enabled automatically.

Export the completed raw and EMA weights:

```bash
python -m tools.seeg_checks export \
  --checkpoint-dir runs/seeg_pair_only_32/checkpoints --step 5000 \
  --weights raw --output runs/seeg_pair_only_raw
python -m tools.seeg_checks export \
  --checkpoint-dir runs/seeg_pair_only_32/checkpoints --step 5000 \
  --weights ema --output runs/seeg_pair_only_ema
CUDA_VISIBLE_DEVICES=0 python -m tools.seeg_compare \
  --config configs/gen/seeg_pair_only_local.yaml \
  --audit runs/seeg_pair_only_audit.json \
  --raw runs/seeg_pair_only_raw --ema runs/seeg_pair_only_ema \
  --targets 4 --num-samples 4 --outdir runs/seeg_pair_only_comparison
```

If it fits substantially better than the mixed loss, investigate objective
balance. If it still fails, isolate noise and dataset size with the following
second experiment. This changes two factors and is an isolation check, not a
clean attribution of which factor helped. Use fresh directories.

```bash
python -m tools.seeg_checks overfit-config \
  --config configs/gen/seeg_overfit_local.yaml \
  --output configs/gen/seeg_pair_only_fixed_local.yaml \
  --targets 4 --steps 5000 --pair-only --fixed-noise-seed 123
python -m tools.seeg_checks audit \
  --config configs/gen/seeg_pair_only_fixed_local.yaml \
  --output runs/seeg_pair_only_fixed_audit.json
CUDA_VISIBLE_DEVICES=0 python main.py --gen \
  --config configs/gen/seeg_pair_only_fixed_local.yaml \
  --workdir runs/seeg_pair_only_fixed_4
```

This reuses one generator noise key at every step and forces batch size and
samples-per-target to one. Validation also uses this key; donors remain fixed.
The subset must contain at least two regions for the existing negative loader;
if rejected, increase targets or change dataset.seed and regenerate the audit.
Export raw/EMA as above using this run's checkpoint directory. Run seeg_compare
with this config/audit, those exports, a fresh outdir, and `--num-samples 1`.
It automatically passes the matching fixed noise key to inference. For manual
sample_pair_list.py calls add `--fixed-noise-seed 123 --num-samples 1 --batch-size 1`;
retain the audited --seed separately to reproduce donors.

These experiments leave condition scale/initialization and the original VAE
unchanged. Success with four fixed-noise targets establishes memorization only.
If needed, separate the two factors next: 32 targets with fixed noise, and four
with ordinary noise. Do not judge either experiment solely by semantic_loss.

## 9. Separate target count from noise variation

The four-target fixed-noise run fits well in latent space. First confirm its
raw decoded samples, then change one factor at a time. Keep the original VAE,
model, paired-only loss, dataset.seed, fixed donor selection, batch size one and
5000 steps. Use the exact fixed-four config from the successful run as the base.
Equal steps give different presentations per target (1250 for four versus about
156 for 32); this is an equal-compute comparison, not equal exposure per target.

### A. Decode the completed fixed-four run (no retraining)

```bash
python -m tools.seeg_checks export \
  --checkpoint-dir runs/seeg_pair_only_fixed_4/checkpoints --step 5000 \
  --weights raw --output runs/seeg_fixed4_raw
python -m tools.seeg_checks export \
  --checkpoint-dir runs/seeg_pair_only_fixed_4/checkpoints --step 5000 \
  --weights ema --output runs/seeg_fixed4_ema
CUDA_VISIBLE_DEVICES=0 python -m tools.seeg_compare \
  --config configs/gen/seeg_pair_only_fixed_local.yaml \
  --audit runs/seeg_pair_only_fixed_audit.json \
  --raw runs/seeg_fixed4_raw --ema runs/seeg_fixed4_ema \
  --targets 4 --num-samples 1 --outdir runs/seeg_fixed4_comparison
```

The comparison automatically reuses the fixed noise seed from the config. Inspect
raw samples and their latent L1 first; EMA can lag. Compare correct and shuffled
conditions to establish image sensitivity. Low raw latent error with detailed
reconstructions verifies the memorization result visually.

### B. Train 32 targets with the same fixed noise

```bash
python -m tools.seeg_checks overfit-config \
  --config configs/gen/seeg_pair_only_fixed_local.yaml \
  --output configs/gen/seeg_pair_only_fixed32_local.yaml \
  --targets 32 --steps 5000 --pair-only --fixed-noise-seed 123
python -m tools.seeg_checks audit \
  --config configs/gen/seeg_pair_only_fixed32_local.yaml \
  --output runs/seeg_pair_only_fixed32_audit.json
CUDA_VISIBLE_DEVICES=0 python main.py --gen \
  --config configs/gen/seeg_pair_only_fixed32_local.yaml \
  --workdir runs/seeg_pair_only_fixed_32
```

Use the successful run's actual fixed noise seed if it differs from 123.

### C. Train four targets with varying noise

```bash
python -m tools.seeg_checks overfit-config \
  --config configs/gen/seeg_pair_only_fixed_local.yaml \
  --output configs/gen/seeg_pair_only_varying4_local.yaml \
  --targets 4 --steps 5000 --pair-only --varying-noise
python -m tools.seeg_checks audit \
  --config configs/gen/seeg_pair_only_varying4_local.yaml \
  --output runs/seeg_pair_only_varying4_audit.json
CUDA_VISIBLE_DEVICES=0 python main.py --gen \
  --config configs/gen/seeg_pair_only_varying4_local.yaml \
  --workdir runs/seeg_pair_only_varying_4
```

`--varying-noise` explicitly removes an inherited fixed_noise_seed; simply omitting
--fixed-noise-seed preserves an existing config value. Fixed and varying flags
cannot be supplied together. Varying noise changes both Gaussian input noise and
random noise-label conditioning, as in ordinary training. Target/donor selections
remain fixed. Each run starts from scratch in its own work directory.

### D. Compare the two new runs

For each run, export raw and EMA at step 5000 using the same commands as A, replacing
the checkpoint directory and output artifact names. Then use these matching sets:

| Run | Config | Audit | Samples |
| --- | --- | --- | --- |
| Fixed 32 | seeg_pair_only_fixed32_local.yaml | seeg_pair_only_fixed32_audit.json | --num-samples 1 |
| Varying 4 | seeg_pair_only_varying4_local.yaml | seeg_pair_only_varying4_audit.json | --num-samples 4 |

Configs live in configs/gen and audits in runs, as above. Pass the matching raw/EMA
exports to tools.seeg_compare and use separate output directories. Compare
training loss_pair, raw decoded structure, constant baselines and correct/shuffled
latent differences. Validation need not improve in these memorization experiments.

If fixed32 succeeds but varying4 fails, noise invariance is the leading difficulty.
If fixed32 fails but varying4 succeeds, larger-subset fitting or exposure is the
leading difficulty. If both succeed, their combination deserves investigation.
If both fail, the fixed-four success is a narrow memorization result. No new
paper-derived architecture or loss is introduced in these controls.
