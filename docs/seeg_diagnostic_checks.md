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

## 10. Extend fixed32 to match small-subset training exposure

Fixed32 is improving and beats constant baselines, while varying4 fits well.
Continue the existing paired-only fixed32 run to 40000 TOTAL steps (35000 more
from step 5000). This gives roughly 1250 presentations per target, comparable to
four targets over 5000 steps. Keep the original VAE and all experiment settings.
No cache rebuild or model modification is required.

### A. Generate the extension config and resume

```bash
python -m tools.seeg_checks extend-config \
  --config configs/gen/seeg_pair_only_fixed32_local.yaml \
  --output configs/gen/seeg_pair_only_fixed32_40k_local.yaml \
  --steps 40000 --resume-min-step 5000

CUDA_VISIBLE_DEVICES=0 python main.py --gen \
  --config configs/gen/seeg_pair_only_fixed32_40k_local.yaml \
  --workdir runs/seeg_pair_only_fixed_32
```

Use the SAME workdir that contains the completed fixed32 `checkpoints/`, not a
new empty directory. The training loop restores model, EMA and optimizer state
from its latest checkpoint. It logs the restored step and refuses to start if
it is below 5000 or already at/above 40000. It resumes the latest available step,
not necessarily exactly 5000. Dataset iteration restarts on process launch;
this is optimizer-state continuation, not a bit-identical replay of sample order.

The helper preserves subset/donor seeds, noise key, architecture, loss weights,
learning rate, diagnostics and validation settings. It extends the constant LR
schedule and retains checkpoints every 5000 steps, with up to eight recent ones.
It rejects nonconstant schedules to avoid silently changing their trajectory.
Do not use --init-from EMA exports as a substitute for full-state resumption.
If you want a separate run directory, copy the existing checkpoints directory
there first and point --workdir to its parent. Keep the original config/audit
and checkpoint copies if you need an immutable record of the earlier experiment.

### B. Compare intermediate checkpoints

Repeat at 10000, 20000 and 40000 (or 5000-step intervals if needed). Here is 10000:

```bash
python -m tools.seeg_checks export \
  --checkpoint-dir runs/seeg_pair_only_fixed_32/checkpoints --step 10000 \
  --weights raw --output runs/seeg_fixed32_10k_raw
python -m tools.seeg_checks export \
  --checkpoint-dir runs/seeg_pair_only_fixed_32/checkpoints --step 10000 \
  --weights ema --output runs/seeg_fixed32_10k_ema
CUDA_VISIBLE_DEVICES=0 python -m tools.seeg_compare \
  --config configs/gen/seeg_pair_only_fixed32_40k_local.yaml \
  --audit runs/seeg_pair_only_fixed32_audit.json \
  --raw runs/seeg_fixed32_10k_raw --ema runs/seeg_fixed32_10k_ema \
  --targets 4 --num-samples 1 --outdir runs/seeg_fixed32_10k_comparison
```

Change --step and all artifact/output names together for each checkpoint.
Use the existing fixed32 audit: the subset and donors have not changed. Run
inference after checkpoint saving completes; use a separate GPU if running it
concurrently with training. To evaluate every training target, use --targets 32.

### C. Decide what to do next

Track loss_pair, raw decoded structure, generated latent L1 versus the same
mean/median baselines, and correct-versus-shuffled differences. Validation stays
patient-held-out and should be interpreted separately from memorization.
If raw fitting continues improving, finish the exposure-matched run. If it
plateaus without recovering structure, investigate optimization and condition
capacity before increasing dataset size. Do not infer success from semantic_loss
alone. A successful fixed32 fit supports a subsequent 32-target varying-noise
experiment at comparable exposure; it does not itself establish generalization.

## 11. Train varying-noise 32 targets at matched exposure

Generate BOTH controlled experiment configs from the exact fixed32 config used
for the 40k run. This retains the subset seed, donor choices, model, original VAE,
learning rate and optimizer settings. It removes the fixed noise key, clears
inherited resume requirements, uses batch size one and 40000 total steps, and
enables conditioning diagnostics. The two generated configs differ only in
lambda_drift (0 versus 0.1). No new paper-derived method is introduced.

```bash
python -m tools.seeg_checks matched-configs \
  --config configs/gen/seeg_pair_only_fixed32_40k_local.yaml \
  --output configs/gen/matched32 --steps 40000 --drift-weight 0.1

python -m tools.seeg_checks audit \
  --config configs/gen/matched32/seeg_varying32_pair_40k_local.yaml \
  --output runs/seeg_varying32_pair_audit.json

CUDA_VISIBLE_DEVICES=0 python main.py --gen \
  --config configs/gen/matched32/seeg_varying32_pair_40k_local.yaml \
  --workdir runs/seeg_varying32_pair_40k
```

Use a NEW EMPTY workdir for this run, not the fixed32 directory. This is a
from-scratch comparison: initialization and dataset seeds match the other config.
The helper refuses to overwrite existing configs. Neither experiment initializes
from fixed32 weights. Subsequent invocations in the same workdir resume its own
latest checkpoint normally. Varying noise changes both spatial Gaussian noise and
noise-label conditioning. Donor selections remain fixed.

At 10k, 20k and 40k, export raw/EMA and compare. Here is the final checkpoint:

```bash
python -m tools.seeg_checks export \
  --checkpoint-dir runs/seeg_varying32_pair_40k/checkpoints --step 40000 \
  --weights raw --output runs/seeg_varying32_pair_40k_raw
python -m tools.seeg_checks export \
  --checkpoint-dir runs/seeg_varying32_pair_40k/checkpoints --step 40000 \
  --weights ema --output runs/seeg_varying32_pair_40k_ema
CUDA_VISIBLE_DEVICES=0 python -m tools.seeg_compare \
  --config configs/gen/matched32/seeg_varying32_pair_40k_local.yaml \
  --audit runs/seeg_varying32_pair_audit.json \
  --raw runs/seeg_varying32_pair_40k_raw --ema runs/seeg_varying32_pair_40k_ema \
  --targets 32 --num-samples 4 --outdir runs/seeg_varying32_pair_40k_comparison
```

Use unique artifact/output names for intermediate steps. Four samples test more
than one noise draw but are not a full diversity evaluation. Compare training L1,
per-target raw/EMA errors, structures and condition sensitivity against fixed32.
If varying32 fits less well, noise invariance remains an optimization difficulty.

## 12. Reintroduce drifting with a matched from-scratch control

Use the second config generated in Section 11. It sets lambda_drift=0.1,
lambda_pair=1, lambda_condition=0 and lambda_tv=0; all other settings match the
varying32 paired-only config. The 0.1 weight reproduces the earlier mixed objective
and is a controlled starting point, not an established optimum. It uses the
existing compatibility-masked drifting implementation and original VAE.

```bash
python -m tools.seeg_checks audit \
  --config configs/gen/matched32/seeg_varying32_drift_40k_local.yaml \
  --output runs/seeg_varying32_drift_audit.json
CUDA_VISIBLE_DEVICES=0 python main.py --gen \
  --config configs/gen/matched32/seeg_varying32_drift_40k_local.yaml \
  --workdir runs/seeg_varying32_drift_40k
```

Use a separate NEW EMPTY workdir. Do not warm-start this controlled comparison
from paired-only weights: that would mix objective and training-history changes.
You may run it after Section 11 or independently on another GPU. Audit target
lists and donor paths should match between the two runs.

```bash
python -m tools.seeg_checks export \
  --checkpoint-dir runs/seeg_varying32_drift_40k/checkpoints --step 40000 \
  --weights raw --output runs/seeg_varying32_drift_40k_raw
python -m tools.seeg_checks export \
  --checkpoint-dir runs/seeg_varying32_drift_40k/checkpoints --step 40000 \
  --weights ema --output runs/seeg_varying32_drift_40k_ema
CUDA_VISIBLE_DEVICES=0 python -m tools.seeg_compare \
  --config configs/gen/matched32/seeg_varying32_drift_40k_local.yaml \
  --audit runs/seeg_varying32_drift_audit.json \
  --raw runs/seeg_varying32_drift_40k_raw --ema runs/seeg_varying32_drift_40k_ema \
  --targets 32 --num-samples 4 --outdir runs/seeg_varying32_drift_40k_comparison
```

Compare these runs at equal steps using loss_pair (NOT total loss, whose terms
differ), raw/EMA samples, baseline errors, shuffled-condition sensitivity and
validation separately. Normalized drift loss near one is not a progress measure.
Better training texture or memorization alone does not establish held-out
improvement. If drifting slows fitting, quantify the tradeoff before changing its
weight. The held-out donor coverage limitation remains in both experiments;
these 32-target controls do not resolve the evaluation protocol or justify claims
of patient generalization.

## 13. Generalization with drifting and a training-patient reference library

Drifting remains active: its feature-space force includes attraction toward
compatible positive features and repulsion involving generated/negative features.
Paired latent L1 remains an additional supervised term. This experiment keeps
lambda_drift=0.1 and lambda_pair=1, the original VAE, and the existing architecture.

The new `dataset.donor_protocol: train_reference` uses training patients as donors
for ALL splits. Target patients remain in their original train/val/test split.
The pair table's existing same-region eligibility and coordinates still apply;
self-patient pairs are excluded. Validation/test images never enter training as
conditions. Their target images are used for evaluation only. This differs from
the older within_split protocol, which remains the default for old configs.

```bash
python -m tools.seeg_checks generalization-config \
  --config configs/gen/matched32/seeg_varying32_drift_40k_local.yaml \
  --output configs/gen/seeg_generalization_local.yaml \
  --steps 100000 --drift-weight 0.1
python -m tools.seeg_checks audit \
  --config configs/gen/seeg_generalization_local.yaml \
  --output runs/seeg_generalization_audit.json
CUDA_VISIBLE_DEVICES=0 python main.py --gen \
  --config configs/gen/seeg_generalization_local.yaml \
  --workdir runs/seeg_generalization
```

Use a new workdir. This starts from scratch on ALL eligible training images,
removes the 32-target cap and fixed noise, and resamples donor channels/segments
during training. Validation selections are deterministic. The original pair-table,
image, cache and split paths are inherited; inspect them first. No cache rebuild
is needed if all referenced images already have latents. Training-reference
coverage should be audited rather than assumed. No new pairing edges are invented.
With approximately 30k eligible training targets and batch size one, 100k steps
is roughly three target traversals, a pilot budget rather than a final schedule.
Image sampling gives patients with more segments more training weight.

The region-incompatible negatives still come from target records in the selected
split. Thus validation loss uses held-out targets/negatives for scoring, not for
optimizer updates. This protocol assumes a training-patient reference library is
available at deployment; it is not a completely isolated held-out donor population.

## 14. Held-out evaluation beyond latent L1

Use validation to choose checkpoints/settings. Keep test evaluation for the final
chosen model. The new evaluation command creates a seeded panel with up to four
eligible target images per patient and reuses it across checkpoints. It reports
patients with no eligible targets. Save the panel and config with the experiment.

First export a completed checkpoint (example: 20000; change as appropriate):

```bash
python -m tools.seeg_checks export \
  --checkpoint-dir runs/seeg_generalization/checkpoints --step 20000 \
  --weights ema --output runs/seeg_generalization_20k_ema
CUDA_VISIBLE_DEVICES=0 python -m tools.seeg_evaluate \
  --config configs/gen/seeg_generalization_local.yaml \
  --init-from runs/seeg_generalization_20k_ema \
  --split val --panel runs/seeg_generalization_val_panel.json \
  --targets-per-patient 4 --num-samples 4 --seed 42 \
  --outdir runs/seeg_generalization_val_20k
```

For later checkpoints reuse exactly the same --panel, config, sample count and
seed; use different artifact/output directories. Inference seed and donor choices
are fixed per panel entry. The command runs a separate inference process per target
so large panels are slow. Cached latents and segmented directory layout are required.
To create a larger panel use a NEW panel path and higher --targets-per-patient;
once created, the existing panel controls selection regardless of that argument.

Run a matched shuffled-condition evaluation by repeating the same command with
`--shuffle-condition-images` and a different --outdir. This retains metadata and
noise seeds while replacing images from training donors. Keep generated conditions.txt
files to audit selection. Sensitivity alone does not demonstrate improved accuracy.

`evaluation.json` includes per-sample and per-target measurements, target averages,
equal-weight patient averages and equal-weight region averages. Report patient
averages to avoid domination by patients with many images. It measures:

- Latent L1 against the cached target.
- Image MAE, RMSE, PSNR (higher is better), and grayscale local SSIM (higher is
  better, Gaussian sigma 1.5; cropped five-pixel border, population covariance).
- Row/column mean-intensity profile errors and gradient error to monitor broad
  structure and edges. These are DISPLAY-SPACE proxies, not physical SEEG band
  power or clinical event metrics; interpreting axes requires plot metadata.
- MAE against the target's VAE reconstruction, separately from the original image.
- Pairwise generated-latent L1 diversity across noise draws. Larger diversity is
  not automatically better and must be interpreted with fidelity.
- VAE reconstruction metrics against the original, as a representation reference.

Scores average all generated samples; no best-of-N target selection is performed.
The PSNR implementation caps perfect-match values at 120 dB. SSIM is implemented
with scipy (already required by JAX); no new model weights are downloaded. SSIM
follows the standard local luminance/contrast/structure formulation of Wang et al.,
"Image Quality Assessment: From Error Visibility to Structural Similarity", IEEE
TIP 2004, https://doi.org/10.1109/TIP.2004.825095. This is an evaluation metric,
not a change to the drifting model or training objective.

After selecting a checkpoint, use --split test and a NEW test panel/output path.
Do not compare new train_reference scores directly with old within_split scores
as if only the model changed. Exact target segment texture may not be predictable
from unrelated patient recordings; interpret reconstruction error alongside
structure, condition controls, and eventually clinically relevant signal metrics.

## 15. Diagnose the stalled full-data run on training and validation panels

The 100k run plateaued on BOTH training and validation latent L1. Do not extend it
unchanged. The successful small runs used repeated targets and fixed donor images;
the full run changed target count, visits per target and donor variability together.
First evaluate TRAINING targets from the existing checkpoint, not just validation.
The evaluator now supports --split train and respects the config's training subset.

```bash
CUDA_VISIBLE_DEVICES=0 python -m tools.seeg_evaluate \
  --config configs/gen/seeg_generalization_local.yaml \
  --init-from runs/seeg_generalization_100k_ema \
  --split train --panel runs/seeg_generalization_train_panel.json \
  --targets-per-patient 1 --num-samples 4 --seed 42 \
  --outdir runs/seeg_generalization_train_100k
```

For the same raw checkpoint, export --weights raw from checkpoints at step 100000
and repeat with the SAME panel in another outdir. Repeat on the existing validation
panel with correct and --shuffle-condition-images inputs. Keep test untouched.
These require no retraining or cache rebuild. Panels with one training example per
eligible patient give a quick fitting check, not exhaustive training-set performance.

Evaluation now also reports zero-latent, TRAINING-mean-latent and selected-donor-mean
latent L1. The training mean is streamed over eligible training target latents only,
respecting any target cap; no validation/test targets enter this baseline. It can
read many .npy files and take time, but does not materialize the full cache in RAM.
The donor baseline averages the actual selected images saved in conditions.txt,
so it changes under shuffled conditions. Neither latent mean is a physical mixture
of signals or a guaranteed plausible spectrogram. Baselines are latent-space checks;
existing image metrics remain unchanged. All baseline values appear in per-sample,
target/patient/region summaries in evaluation.json.

For fixed-condition training subsets, the evaluator reproduces the audited view-zero
training donors with a separate condition seed. Generator noise seeds stay fixed
per evaluation panel. For resampled training conditions, it selects a reproducible
training-style donor draw. Held-out evaluation continues selecting nearest eligible
donor channels and a seeded segment. Keep this difference in mind when comparing
training to validation. Panels are not the same as minibatches in training logs.

## 16. Isolate donor resampling before another full-data run

Generate three matched, drift-active 32-target experiments from the stalled full
config. Keep the original VAE, architecture, learning rate, drift weight and paired
L1. Donors are training patients and generator noise varies normally in ALL three.

```bash
python -m tools.seeg_recovery \
  --config configs/gen/seeg_generalization_local.yaml \
  --output configs/gen/recovery32 --stage conditions --steps 40000 --views 4
```

The new condition_sampling setting controls ONLY training donor selection:

| Config | Donor inputs on repeated visits |
| --- | --- |
| seeg_recovery32_fixed.yaml | One reproducible condition set per target |
| seeg_recovery32_resample.yaml | Fresh random donor channels/segments each visit |
| seeg_recovery32_bank.yaml | Randomly choose among four reproducible sets per target |

All variants fix negative sampling per target to prevent it confounding the donor
comparison; they differ ONLY in condition_sampling. Bank view zero reproduces the
fixed donor set. Other views can overlap or duplicate when donors have few images;
"four" is a cap on seeded views, not a guarantee of four distinct sets. This bounded
augmentation is a proposed experiment, not an established improvement or a new
paper-derived algorithm. Training views never include validation/test patients.

Audit each config (example for fixed):

```bash
python -m tools.seeg_checks audit \
  --config configs/gen/recovery32/seeg_recovery32_fixed.yaml \
  --output runs/seeg_recovery32_fixed_audit.json
CUDA_VISIBLE_DEVICES=0 python main.py --gen \
  --config configs/gen/recovery32/seeg_recovery32_fixed.yaml \
  --workdir runs/seeg_recovery32_fixed
```

Repeat by replacing every "fixed" in those paths with "resample", then "bank".
Use NEW EMPTY workdirs. Start with fixed/resample; bank is useful if refreshed donors
hurt fitting. Re-running old experiments exactly is not intended: these new controls
use training-reference donors and controlled negatives, with identical settings
between the three new variants. Audit overfit_targets lists view-zero conditions;
it does not claim those are every random draw or every bank view.

At 10k and 40k, export raw/EMA using Section 12's export pattern with the matching
checkpoint directory. Use tools.seeg_compare with the matching config and audit,
--targets 32 --num-samples 4, and a distinct output directory. It evaluates view-zero
donors on the training targets, including for the bank model. Use tools.seeg_evaluate
on training/val panels as in Sections 14-15 for baseline and image measurements.
Use the SAME saved validation panel across all variants. Banks must also be checked
on held-out targets with ordinary donor selection; view-zero memorization alone
is not success at generalization.

The new configs use a seeded patient-balanced validation cap of 64 eligible images,
cache the same validation batches for every evaluation, and reuse a validation-only
noise seed. They report val/* using EMA weights. Training loss remains raw-weight
stochastic minibatch loss, so train and val curves have different estimators.
These validation settings do NOT alter training noise, images or gradients. Old
configs retain their original behavior unless the new fields are enabled.

Validation panel/noise controls live in train.forward_dict:
`stable_validation: true`, `validation_noise_seed: 2026`, `validation_use_ema: true`.
The validation loss frequency is 1000 steps. Dataset.val_max_targets=64 provides
patient-balanced cap selection; eligible patients missing pair-table edges remain
missing and should be reported by the audit. Stable validation makes comparisons
less noisy, not necessarily lower. Training resampling and bank draw order remain
stochastic and are not bit-identically replayed on resume.

The automatic loss_curve.png now plots paired latent L1 when available. The
separate semantic_loss_curve.png preserves the detached semantic monitor. This
plotting change applies to all runs; historical plots can be regenerated by
calling WandbLogger.save_loss_plot on their metrics directory. Do not optimize
for visual smoothness of curves or treat normalized drift_loss near one as failure.

Interpret the control before scaling:

- Fixed fits, resample fails: donor variability is implicated; test bank as a bridge.
- Fixed and resample both fit: the full-run failure is more likely scale/exposure or
  heterogeneous data, not resampling alone.
- Bank fits training but not validation: it is another memorization result; do not
  promote it based on training losses alone.
- Fixed no longer fits: stop and inspect subset, conditioning/negative controls and
  data before launching a larger run.

## 17. Scale gradually with patient-balanced training subsets

Only after Section 16 shows learning, generate a 256-target pilot, rather than
jumping directly to all ~30k images. This gives eligible training patients roughly
equal target representation in the subset; it does not change or combine patient
splits. Full data remain available as the reference donor library.

```bash
python -m tools.seeg_recovery \
  --config configs/gen/seeg_generalization_local.yaml \
  --output configs/gen/recovery256 --stage scale --steps 100000 --views 4
python -m tools.seeg_checks audit \
  --config configs/gen/recovery256/seeg_recovery256_bank.yaml \
  --output runs/seeg_recovery256_bank_audit.json
CUDA_VISIBLE_DEVICES=0 python main.py --gen \
  --config configs/gen/recovery256/seeg_recovery256_bank.yaml \
  --workdir runs/seeg_recovery256_bank
```

A matched seeg_recovery256_resample.yaml is generated too. Run it separately if
Section 16 supports that comparison; do not automatically spend two full runs.
Both keep drifting active. No new encoder, condition scale, feature architecture,
LR schedule or objective weight change is included, so the candidate sampling
improvement can be evaluated on its own. Use a fresh workdir for each run.

100k steps give about 390 presentations per target here, still below the ~1250
in fixed32 at 40k. Compare checkpoints at 10k/40k/100k. If larger-subset fitting
is merely slower, account for exposure before claiming a capacity failure. Evaluate
all 256 training targets when practical (tools.seeg_compare --targets 256) and the
same validation panel. Choose settings on patient-averaged validation metrics,
condition-shuffling controls and structure, not a best-looking target. If training
improves while validation does not, prioritize predictive information in donors,
recording-state metadata and task definition before more memorization training.
Arbitrary cross-patient segments may not predict an exact target's fine texture;
no sampling change guarantees the validation L1 will approach reconstruction error.

## 18. Generate controlled drifting-field sample experiments

The dataset remains `~/RAE_first_exp/{train,val,test}/PATIENT/CHANNEL/*.png`.
No filenames, split assignments, graph edges, encoder or latent representation
change. Existing complete caches can be reused. The next question is whether
single generated/positive examples and different-region real negatives limit the
conditional drifting field. These are controls, not a promised generalization fix.

Generate five configs from the exact recovery256 bank config used previously:

```bash
python -m tools.seeg_field_experiments \
  --config configs/gen/recovery256/seeg_recovery256_bank.yaml \
  --output configs/gen/field256 --targets 256 --steps 100000 \
  --generated 4 --positives 4 --negative-weight 0.25
```

The helper preserves the selected base's donor bank, condition views, target-subset
seed/balancing, VAE, model, optimizer/LR, stable validation settings and drift weight.
It keeps lambda_pair=1 and lambda_drift positive. All runs start from scratch.
The following changes are sequential controls:

| Config suffix | Generated per condition | Positive slots | Explicit real-negative weight |
| --- | --- | --- | --- |
| a_anchor | 1 | 1 | 1 |
| b_multigen | 4 | 1 | 1 |
| c_multipos | 4 | 4 | 1 |
| d_softneg | 4 | 4 | 0.25 |
| e_norealneg | 4 | 4 | 0 |

Files are named seeg_field_SUFFIX.yaml. A versus B isolates generated count; B
versus C isolates positive count; C versus D/E isolates explicit-real-negative
weight. D is optional; do not launch all five full runs automatically. Run A/B
first, then C, then D or E depending on results and budget. A is a matched control
and can also be compared with the existing recovery256 baseline, but use the new
run if comparing exact initialization/history. Warm-starting from old checkpoints
would change the comparison. Use new empty workdirs.

Positive slot zero is ALWAYS the exact target segment. Additional positives are
distinct images from the SAME patient's SAME channel folder, selected reproducibly
per anchor. They may cover different two-minute intervals; they are samples of
that channel's resting-state distribution, not synchronized copies of the anchor.
The exact target remains the ONLY L1 anchor for every generated sample. Conditions
remain other-patient neighbors from the pair table; positive images are loss inputs,
never generator conditions. Validation/test patients never provide training positives.

If fewer images exist, the remaining slots contain masked padding; duplicates
are not counted as extra positives. For example, two available images with k_pos=4
produce positive_mask=[1,1,0,0]. Padding contributes neither to field affinities nor
to the semantic monitor. `dataset.positive_sampling: fixed` makes the new comparison
reproducible; `resample` is supported for a later, separately controlled experiment.

A training target cap limits anchor images, NOT the same-channel positive library.
C/D/E can use other training segments of the selected channels outside the 256
anchors. This expansion is part of the intended positive-pool change and must be
reported, not described as identical supervised data exposure. The audit now reports
anchor and available positive-image counts, singleton channels, generated count,
negative weight, and selected positive_paths/masks for capped training targets.

All configs set dataset.val_k_pos=1: validation keeps the exact target as its single
positive, with the same examples/noise/EMA protocol. Training can therefore have
multiple positive segments without granting extra held-out positive segments to
validation scoring. Existing configs default to k_pos=1; their behavior is preserved.

## 19. Run the generated-sample and positive-pool controls

Example A (repeat for B by replacing a_anchor with b_multigen in every path):

```bash
python -m tools.seeg_checks audit \
  --config configs/gen/field256/seeg_field_a_anchor.yaml \
  --output runs/seeg_field_a_anchor_audit.json
CUDA_VISIBLE_DEVICES=0 python main.py --gen \
  --config configs/gen/field256/seeg_field_a_anchor.yaml \
  --workdir runs/seeg_field_a_anchor
```

For C, use seeg_field_c_multipos.yaml and a matching new workdir/audit. Check its
positive pool counts before training. More generated/positive samples increase
feature-extractor activation memory and runtime. Run a short smoke check first.
If four generated samples do not fit GPU memory, regenerate ALL experiment configs
in a new directory with --generated 2 and keep --positives 4. If positive features
also do not fit, use --positives 2 consistently for C/D/E. Do not silently change
batch size, model or loss weights in only one variant to handle memory.

Report both optimizer steps and forward_kimg: B-E generate four samples per target
visit versus A's one. Equal steps preserve anchor visits, not compute or generated
sample exposure. Initial comparisons at 10k/20k are useful for deciding whether
longer runs are justified; select checkpoints on the same validation panel rather
than choosing the last checkpoint just because training error is lower.

Example checkpoint exports/comparison for C at 20000:

```bash
python -m tools.seeg_checks export \
  --checkpoint-dir runs/seeg_field_c_multipos/checkpoints --step 20000 \
  --weights raw --output runs/seeg_field_c_20k_raw
python -m tools.seeg_checks export \
  --checkpoint-dir runs/seeg_field_c_multipos/checkpoints --step 20000 \
  --weights ema --output runs/seeg_field_c_20k_ema
CUDA_VISIBLE_DEVICES=0 python -m tools.seeg_compare \
  --config configs/gen/field256/seeg_field_c_multipos.yaml \
  --audit runs/seeg_field_c_multipos_audit.json \
  --raw runs/seeg_field_c_20k_raw --ema runs/seeg_field_c_20k_ema \
  --targets 32 --num-samples 4 --outdir runs/seeg_field_c_20k_comparison
```

This is a TRAINING-target comparison; --targets 256 evaluates all anchors. Use
matching paths for A/B/D/E and for other checkpoints. New training metrics include
positive_count (actual unmasked positives), generated_per_condition,
real_negative_weight, and gen_pair_distance when multiple samples are generated.
Semantic_loss remains a detached feature-similarity monitor; positive padding is
excluded. Changing positive sets changes that monitor's reference distribution,
so compare actual target L1 and validation image/structure metrics too.

## 20. Test explicit negatives while retaining drifting

C retains the existing different-region real negatives. D reduces their weight;
E sets it to zero. In ALL of B-E, distinct generated samples still repel one
another and are attracted toward positive samples. Thus E is NOT L1-only training.
Generated repulsion is unchanged by real_negative_weight. Do not set generated
count to one in E: that would remove the distinct generated companions this
control is designed to retain.

```bash
python -m tools.seeg_checks audit \
  --config configs/gen/field256/seeg_field_e_norealneg.yaml \
  --output runs/seeg_field_e_norealneg_audit.json
CUDA_VISIBLE_DEVICES=0 python main.py --gen \
  --config configs/gen/field256/seeg_field_e_norealneg.yaml \
  --workdir runs/seeg_field_e_norealneg
```

D runs identically using d_softneg paths. Settings live in
train.loss_kwargs.real_negative_weight. This weight changes affinity and scale
calculations; because the field is normalized, 0.25 does NOT promise one quarter
of the final gradient. Zero-weight real negatives are masked before softmax and
excluded from field scaling/forces. Their images are still loaded for the detached
semantic monitor, so E does not eliminate all negative-image memory/I/O. The
monitor can still compare to them even though they no longer supervise the field.

Use the SAME saved validation panel for all variants, as in Section 14:

```bash
CUDA_VISIBLE_DEVICES=0 python -m tools.seeg_evaluate \
  --config configs/gen/field256/seeg_field_c_multipos.yaml \
  --init-from runs/seeg_field_c_20k_ema \
  --split val --panel runs/seeg_generalization_val_panel.json \
  --targets-per-patient 4 --num-samples 4 --seed 42 \
  --outdir runs/seeg_field_c_val_20k
```

Repeat with each variant's config/export and unique outdir, then repeat with
--shuffle-condition-images. Reuse the existing patient-held-out panel; do not
select a panel based on which targets look good. Compare patient-macro latent L1,
SSIM/profile errors, VAE references, mean/donor baselines and generated diversity.
Do not compare total training loss as if it were the same objective across variants.
The in-loop validation sample count changes with generated_per_condition; offline
seeg_evaluate with the same --num-samples/seed/batch size gives the stricter common
measurement. Validation L1 still measures the exact target segment, not whether
other plausible resting-state samples are physiologically acceptable.

Improved training texture without improved held-out measurements is still
memorization. These controls test whether positive/negative construction contributes
to that gap; they do not assume it is the sole cause. Keep the original VAE,
patient split, reference-donor protocol and confirmed resting-state assumption.
No new paper-derived architecture is introduced.
