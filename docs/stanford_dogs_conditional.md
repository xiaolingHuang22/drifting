# Stanford Dogs conditional drifting experiment

## Data layout

Use the same five breed folders in all three splits:

```text
StanfordDogs_split/
├── train/<breed>/*.{jpg,jpeg,png}
├── val/<breed>/*.{jpg,jpeg,png}
└── test/<breed>/*.{jpg,jpeg,png}
```

With 150 images per breed and a 70/15/15 split, put 105 images in
`train`, 22 in `val`, and 23 in `test` for every breed. Do not copy one
image into more than one split.

The conditional ImageFolder dataset makes every image in a split a target.
For each target it dynamically:

1. scales each image to `[0,1]`, resizes it to `dataset.resolution ×
   dataset.resolution`, and applies torchvision augmentation only for training;
2. selects `k_conditions` other same-breed images and concatenates their RGB
   tensors on the channel axis into `source`;
3. repeats each target `condition_sets_per_target` times so independent
   condition sets form separate training examples;
4. selects `k_positive` additional same-breed images as the independent
   positive reference set;
5. selects `k_neg` images from other breeds as negatives; and
6. returns the target as `target_true` for evaluation or an optional pair loss.

Validation and test condition selection is deterministic. Training selection
is randomized by DataLoader worker seeds.

## Training

Edit `dataset.data_path` in
`configs/gen/stanford_dogs_conditional_debug.yaml`, then run:

```bash
CUDA_VISIBLE_DEVICES=0 python main.py \
  --gen \
  --config configs/gen/stanford_dogs_conditional_debug.yaml \
  --workdir runs/stanford_dogs_conditional_debug
```

The debug config trains for 2,000 steps, validates every 100 steps, and saves
every 500 steps. Breed labels are not passed to the generator: its class label
is always zero, so breed information must come from the concatenated images.
With four RGB conditions, the source has 12 channels; concatenating three
noise channels gives the generator a 15-channel input. Generated images and
all data tensors use the `[0,1]` pixel range and are 32×32 in this debug setup.

The drift objective is evaluated in the pretrained MAE feature space rather
than directly on flattened RGB pixels. The latter is inappropriate for these
spatially unaligned photographs: minimizing raw-pixel drift can reward coarse
patch/color averages instead of recognizable dogs. The first run therefore
downloads `hf://mae_pixel_640` into `HF_ROOT`. To deliberately reproduce the
raw-pixel ablation, set `feature.include_raw_global: true` and
`feature.use_mae: false` (and clear `feature.mae_path`).

If samples become more block-like while `loss_0.2/global` decreases, do not
interpret that metric as evidence that image quality improved. It only reports
the magnitude of the drift field in the named feature space. In particular,
the old `global` metric measured flattened pixel vectors and could improve
while perceptual quality became worse. With the default config below, MAE
feature names replace `global` in the per-feature log keys.

## Test generation

After training, choose one held-out target and sample conditions from the same
test breed folder:

```bash
CUDA_VISIBLE_DEVICES=0 python sample_only.py \
  --init-from runs/stanford_dogs_conditional_debug/params_ema \
  --outdir runs/stanford_dogs_conditional_debug_test/chihuahua \
  --num-samples 32 \
  --batch-size 8 \
  --class-id 0 \
  --condition-dir ~/StanfordDogs_split/test/n02085620-Chihuahua \
  --num-conditions 4 \
  --condition-seed 42 \
  --target-img ~/StanfordDogs_split/test/n02085620-Chihuahua/example.jpg \
  --hsdp-dim 1
```

`target-img` is excluded from the sampled condition set and is not sent to the
generator. It is only saved into the output directory for comparison. The
output directory also contains:

- `conditions.txt`: exact condition files selected;
- `condition_00_*.png`, etc.: each resized condition whose channels are sent to
  the generator;
- `target_held_out.png`: transformed target used only for comparison; and
- generated `sample_*.png` images.

The individual previews are saved for auditing. The model no longer receives a
pixel average: condition 0 occupies source channels 0–2, condition 1 occupies
channels 3–5, and so on. The order is exactly the order in `conditions.txt`.

At the end of training, the offline logger reads the emitted `loss` and
`val/loss` records and writes `WORKDIR/log/loss_curve.png`. Its points are the
same buffered logging steps stored in `WORKDIR/log/metrics.jsonl`.

## Compare saved checkpoints

`params_ema/` contains only the latest exported EMA parameters. To sample an
earlier resumable checkpoint, point `--init-from` at `checkpoints/`, provide
the original training YAML so the generator architecture can be rebuilt, and
select the saved step explicitly:

```bash
CUDA_VISIBLE_DEVICES=0 python sample_only.py \
  --init-from runs/stanford_dogs_conditional_debug/checkpoints \
  --config configs/gen/stanford_dogs_conditional_debug.yaml \
  --checkpoint-step 500 \
  --outdir runs/checkpoint_comparison/step_500 \
  --num-samples 8 \
  --batch-size 8 \
  --seed 123 \
  --class-id 0 \
  --condition-dir ~/StanfordDogs_split/test/n02085620-Chihuahua \
  --num-conditions 4 \
  --condition-seed 123 \
  --hsdp-dim 1
```

Repeat with another retained step and a different output directory. Raw
training checkpoints contain both current and EMA parameters; the sampler uses
`ema_params`, matching normal sampling from `params_ema/`. Keep `--seed`,
`--condition-seed`, and all condition arguments identical for a meaningful
checkpoint comparison.

The sampler also copies `dataset.num_classes` from this YAML into the rebuilt
generator. Training injects that value outside the YAML `model` block, while a
raw checkpoint does not carry model metadata. Omitting it would reconstruct
the default 1,001-row class embedding instead of this experiment's one-row
embedding and cause a Flax `ScopeParamShapeError`.

For a condition-use check, keep `--seed` fixed and repeat the command with a
different breed's `--condition-dir`. The output breed should change with the
conditions. Do not use images from `train` or `val` for this final test.
