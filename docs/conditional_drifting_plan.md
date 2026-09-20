# Conditional drifting plan for paired SEEG spectrograms

This note sketches the first conditional-drifting prototype for paired SEEG spectrogram data.
It is intentionally scoped as a patch plan because the current training path is ImageFolder/class-label driven, while the conditional task needs a paired manifest and a generator input that depends on a source spectrogram.

## 1. Pair notation and tensors

Use one training record per pair:

```text
pair = (ps, pr, ls, lr)
```

where:

- `ps`: patient that provides the source signal.
- `pr`: patient that provides the reference signal at the destination location.
- `ls`: exact coordinate/location available for `ps`, matched to `lr` and in the same brain region.
- `lr`: exact destination coordinate/location that exists for `pr` and may be missing for `ps` at inference time.
- `ps_ls`: source spectrogram condition, shape `[H, W, C]`.
- `ps_lr`: true target for supervised training when available, shape `[H, W, C]`.
- `pr_lr`: reference/positive sample at the destination location, shape `[H, W, C]`.
- `ps_not_lr`: negative samples from `ps` at locations other than `lr`, shape `[K_neg, H, W, C]`.

The minimal dataloader batch should produce:

```python
{
    "source": ps_ls,             # [B, H, W, C]
    "target_true": ps_lr,        # [B, H, W, C], optional at inference
    "target_ref": pr_lr,         # [B, H, W, C]
    "negative": ps_not_lr,       # [B, K_neg, H, W, C]
    "source_coord": ls,          # [B, coord_dim]
    "target_coord": lr,          # [B, coord_dim]
}
```

If `ps_lr` is not available for a training pair, omit or mask the supervised term for that item.

## 2. Original drifting loss

The existing loss receives generated features `gen`, positive features `fixed_pos`, and negative features `fixed_neg`.
It builds the target bank:

```math
T = [\operatorname{stopgrad}(g), n, p]
```

where `g` is generated features, `n` is negative features, and `p` is positive features. For each radius `R`, it computes distance logits:

```math
\ell_R = -\frac{D(g, T)}{R \cdot \operatorname{scale}}
```

turns them into symmetrized affinities, separates positive and negative blocks, computes a force `F_R`, normalizes it, and creates a stop-gradient goal:

```math
\tilde{g}_{goal} = \operatorname{stopgrad}\left(\tilde{g} + \sum_{R \in R_{list}} \frac{F_R}{\sqrt{\operatorname{mean}(F_R^2)}}\right)
```

The optimized loss is:

```math
\mathcal{L}_{drift} = \mathbb{E}\left[\|\tilde{g} - \tilde{g}_{goal}\|_2^2\right]
```

## 3. Conditional objective for the proposed prototype

Your proposed channel-concat generator is:

```math
x_{gen} = G_\theta([e, ps_{ls}], lr)
```

where `e` is random noise and `[e, ps_ls]` is channel-wise concatenation before the patch embedding.
For a first prototype, keep `lr`/`ls` coordinate conditioning optional but strongly recommended, because `ps_ls` alone may be ambiguous if multiple destination coordinates are possible.

Feature extraction:

```math
g = \phi(x_{gen}),\quad p = \phi(pr_{lr}),\quad n = \phi(ps_{not\_lr})
```

Conditional drift term:

```math
\mathcal{L}_{cond\_drift}
= \mathcal{L}_{drift}(g, fixed\_pos=p, fixed\_neg=n)
```

Optional supervised target term, only when `ps_lr` exists:

```math
\mathcal{L}_{pair}
= \|\phi(x_{gen}) - \phi(ps_{lr})\|_1
```

or in pixel space:

```math
\mathcal{L}_{pair\_pixel}
= \|x_{gen} - ps_{lr}\|_1
```

Total loss:

```math
\mathcal{L}
= \lambda_d \mathcal{L}_{cond\_drift}
+ \lambda_p \mathcal{L}_{pair}
```

Start with a small `lambda_p` (or even `0`) if the paired target is noisy or inconsistently aligned.

## 4. Dataset patch sketch

Add a new file, for example `dataset/paired_spectrogram.py`:

```python
class PairedSpectrogramDataset(torch.utils.data.Dataset):
    def __init__(self, manifest_path, transform=None, k_neg=4):
        self.rows = load_jsonl_or_csv(manifest_path)
        self.transform = transform
        self.k_neg = k_neg

    def __getitem__(self, idx):
        row = self.rows[idx]
        source = load_image(row["ps_ls_path"])
        target_ref = load_image(row["pr_lr_path"])
        target_true = load_image(row["ps_lr_path"]) if row.get("ps_lr_path") else zeros_like(source)
        negatives = [load_image(p) for p in sample_negatives(row["ps_not_lr_paths"], self.k_neg)]
        return {
            "source": self.transform(source),
            "target_true": self.transform(target_true),
            "target_ref": self.transform(target_ref),
            "negative": torch.stack([self.transform(x) for x in negatives]),
            "source_coord": torch.tensor(row["ls"], dtype=torch.float32),
            "target_coord": torch.tensor(row["lr"], dtype=torch.float32),
            "has_target_true": torch.tensor(bool(row.get("ps_lr_path")), dtype=torch.float32),
        }
```

Then update `utils/model_builder.py` so `dataset.mode: paired_spectrogram` calls this loader instead of the ImageFolder loader.

## 5. Generator patch sketch

The current generator creates random noise internally and passes it into `generate_image`. For channel-wise concatenation, add optional `source` input and split `in_channels` into noise channels plus condition channels.

Config idea:

```yaml
model:
  in_channels: 6       # noise RGB/3 + source RGB/3
  out_channels: 3
  noise_in_channels: 3
  source_in_channels: 3
  use_coord_cond: true
```

Sketch inside `DitGen.__call__`:

```python
def __call__(self, c, cfg_scale=1.0, source=None, source_coord=None, target_coord=None, ...):
    B = c.shape[0]
    noise = normal((B, H, W, self.noise_in_channels))
    if source is not None:
        x = jnp.concatenate([noise, source], axis=-1)
    else:
        x = noise

    cond = self.c_cfg_noise_to_cond(c, cfg_scale, noise_labels)
    if self.use_coord_cond:
        coord_cond = self.coord_embed(jnp.concatenate([source_coord, target_coord], axis=-1))
        cond = cond + coord_cond[:, None, :]

    samples = self.generate_image(x, cond, deterministic=deterministic)
    return {"samples": samples}
```

For the first prototype, `c` can be a dummy label or a coarse region/site ID. The exact `ls/lr` coordinate embedding should carry the continuous location information.

## 6. Training-step patch sketch

Add a separate training function rather than changing the existing one immediately:

```python
def train_step_conditional(state, batch, feature_params, feature_apply, ...):
    source = batch["source"]              # ps_ls
    target_ref = batch["target_ref"]      # pr_lr, positive
    target_true = batch["target_true"]    # ps_lr, optional supervised target
    negative = batch["negative"]          # ps_not_lr

    def loss_fn(params):
        gen = state.apply_fn(
            {"params": params},
            train=True,
            rngs=prepare_rng(rng_step, ["noise"]),
            c=batch.get("site_id", dummy_site_id),
            source=source,
            source_coord=batch["source_coord"],
            target_coord=batch["target_coord"],
            cfg_scale=1.0,
        )["samples"]

        gen_feat = feature_apply(feature_params, gen, **activation_kwargs)
        pos_feat = feature_apply(feature_params, target_ref, **activation_kwargs)
        neg_feat = feature_apply(feature_params, rearrange(negative, "b k h w c -> (b k) h w c"), **activation_kwargs)

        # Reshape feature trees to [B, C, S] to match drift_loss.
        drift, drift_info = drift_loss(gen=gen_feat, fixed_pos=pos_feat, fixed_neg=neg_feat, **loss_kwargs)

        pair = feature_l1(gen, target_true, mask=batch["has_target_true"])
        loss = lambda_drift * drift.mean() + lambda_pair * pair
        return loss, {"loss_drift": drift.mean(), "loss_pair": pair, **drift_info}
```

Keep the original `train_step` unchanged until the conditional pipeline is stable.

## 7. Negative-sampling caveat

Using `ps_not_lr` as negatives is a reasonable first experiment, but it is a strong assumption: the model will be pushed away from the source patient's other measured sites. That may help location specificity, but it may also suppress patient-specific style. Alternative negatives to compare later:

1. `pr` or other patients at non-`lr` locations.
2. Same coarse brain region but different exact coordinate.
3. Random mismatched `(patient, location)` pairs.
4. A mixture of all of the above with weights.

## 8. Minimal config sketch

```yaml
dataset:
  mode: paired_spectrogram
  manifest_train: /path/to/train_pairs.jsonl
  manifest_val: /path/to/val_pairs.jsonl
  k_neg: 4
  batch_size: 64
  eval_batch_size: 64

model:
  in_channels: 6
  noise_in_channels: 3
  source_in_channels: 3
  out_channels: 3
  use_coord_cond: true

train:
  conditional: true
  lambda_drift: 1.0
  lambda_pair: 0.1
  loss_kwargs:
    R_list: [0.2]
```

## 9. Recommended implementation order

1. Implement `PairedSpectrogramDataset` and verify shapes only.
2. Add `dataset.mode: paired_spectrogram` branch in `model_builder`.
3. Add generator `source` argument and channel concat path.
4. Add `train_step_conditional` with paired pixel/feature loss only.
5. Add conditional drift with `pr_lr` positives and `ps_not_lr` negatives.
6. Add validation/inference for a held-out pair.

This order gives a working checkpoint at each stage and avoids debugging dataset, model, and loss changes all at once.
