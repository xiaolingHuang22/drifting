"""Generate a held-out SEEG target from its variable-length pair list."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from PIL import Image
from torchvision import transforms

from dataset.pair_list_spectrogram import build_pair_list_records, latent_cache_path, load_patient_split, select_conditions
from dataset.vae import vae_enc_decode
from sample_only import _to_uint8_images
from utils.env import HF_ROOT
from utils.hsdp_util import set_global_mesh
from utils.init_util import load_generator_model_and_params
from utils.misc import load_config, prepare_rng, run_init


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--init-from", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--target-id", required=True, help="For example: PID1,A10")
    parser.add_argument("--target-segment", help="Exact target image filename; required for channels with multiple segments.")
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--training-condition-selection", action="store_true", help="Use seeded training donor-channel ordering for fixed-subset checks.")
    parser.add_argument("--shuffle-condition-images", action="store_true", help="Replace donor images with seeded random in-split images while preserving condition metadata.")
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--num-samples", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint-step", type=int, default=None)
    parser.add_argument("--hsdp-dim", type=int, default=1)
    args = parser.parse_args()

    run_init()
    set_global_mesh(args.hsdp_dim)
    config = load_config(args.config)
    model_config = dict(config.model)
    model_config["num_classes"] = int(config.dataset.num_classes)
    model, params, _ = load_generator_model_and_params(
        args.init_from,
        hf_cache_dir=HF_ROOT,
        model_config=model_config,
        checkpoint_step=args.checkpoint_step,
    )

    image_root = Path(config.dataset.image_root).expanduser().resolve()
    manifest = load_patient_split(config.dataset.split_manifest)
    segmented = any((image_root / split).is_dir() for split in ("train", "val", "test"))
    records = build_pair_list_records(config.dataset.table_path, image_root,
                                      config.dataset.split_manifest if segmented else None)
    allowed = set(manifest[args.split])
    matches = [r for r in records if r["target_patient"] in allowed
               and (r["target_id"] == args.target_id or r.get("channel_id") == args.target_id)
               and (not args.target_segment or Path(r["target_path"]).name == args.target_segment)]
    if len(matches) != 1:
        raise ValueError(f"Expected one target in {args.split}, found {len(matches)}. Specify --target-segment for a multi-segment channel.")
    record = dict(matches[0])
    record["conditions"] = [c for c in record["conditions"] if c["patient"] in allowed]
    if not record["conditions"]:
        raise ValueError("No donor channels remain in the requested patient split.")
    max_conditions = int(config.dataset.get("max_conditions", 8))
    selected = select_conditions(record, max_conditions, random.Random(args.seed), training=args.training_condition_selection)
    if args.shuffle_condition_images:
        pool = sorted({path for r in records if r["target_patient"] in allowed
                       for c in r["conditions"] if c["patient"] in allowed
                       for path in c.get("paths", [c["path"]])}
                      - {r["target_path"] for r in matches})
        if not pool:
            raise ValueError("No in-split images available for condition shuffling.")
        shuffle_rng = random.Random(args.seed + 100000)
        selected = [dict(c, path=shuffle_rng.choice([p for p in pool if p != c["path"]] or pool)) for c in selected]
    latent_root_value = config.dataset.get("latent_root", None)
    latent_root = Path(latent_root_value).expanduser().resolve() if latent_root_value else None
    if latent_root is not None:
        condition_latents = []
        for item in selected:
            image_path = Path(item["path"])
            latent_path = latent_cache_path(latent_root, image_path, image_root)
            if not latent_path.is_file():
                raise FileNotFoundError(
                    f"Missing cached condition latent: {latent_path}. Build it with "
                    "python -m dataset.pair_list_latent_cache."
                )
            latent = np.load(latent_path, allow_pickle=False)
            if latent.ndim != 3 or latent.shape[-1] != 4:
                raise ValueError(f"Expected HWC latent with 4 channels, got {latent.shape}: {latent_path}")
            condition_latents.append(latent.astype(np.float32, copy=False))
        while len(condition_latents) < max_conditions:
            condition_latents.append(np.zeros_like(condition_latents[0]))
        encoded = jnp.asarray(np.stack(condition_latents))
    else:
        transform = transforms.Compose([
            transforms.Resize(
                (int(config.dataset.resolution), int(config.dataset.resolution)),
                antialias=True,
            ),
            transforms.ToTensor(),
            transforms.Normalize([.5] * 3, [.5] * 3),
        ])
        pixel_conditions = []
        for item in selected:
            with Image.open(item["path"]) as image:
                pixel_conditions.append(
                    np.asarray(transform(image.convert("RGB")), dtype=np.float32)
                )
        while len(pixel_conditions) < max_conditions:
            pixel_conditions.append(np.zeros_like(pixel_conditions[0]))
        pixels = jnp.asarray(np.stack(pixel_conditions))
        encode_fn, _ = vae_enc_decode()
        encoded = encode_fn(pixels, jax.random.PRNGKey(args.seed))
        if encoded.shape[1] in (1, 3, 4):
            encoded = encoded.transpose(0, 2, 3, 1)
    source_single = encoded[None]
    mask_single = jnp.asarray(
        [[1.0] * len(selected) + [0.0] * (max_conditions - len(selected))],
        dtype=jnp.float32,
    )
    target_coord = jnp.asarray([record["target_coord"]], dtype=jnp.float32)
    coordinate_scale = float(config.dataset.get("coordinate_scale", 100.0))
    target_coord = target_coord / coordinate_scale
    coords = np.zeros((max_conditions, 3), dtype=np.float32)
    distances = np.zeros((max_conditions, 1), dtype=np.float32)
    for index, item in enumerate(selected):
        coords[index] = np.asarray(item["coord"]) / coordinate_scale
        distances[index, 0] = item["distance"] / coordinate_scale
    metadata_single = jnp.asarray(
        np.concatenate([coords, coords - np.asarray(target_coord[0]), distances], axis=-1)[None]
    )

    def generate(labels, source, mask, metadata, target_coord_value, model_params, rng):
        return model.apply(
            {"params": model_params}, c=labels, cfg_scale=1.0, source=source,
            source_coord=jnp.zeros_like(target_coord_value), target_coord=target_coord_value,
            condition_mask=mask, condition_metadata=metadata,
            rngs=prepare_rng(rng, ["noise"]), train=False,
        )["samples"]

    # Keep model parameters as an explicit JIT argument. Capturing device arrays
    # in a closure can retain a stale PJRT device handle when CUDA visibility is
    # restricted for inference (for example CUDA_VISIBLE_DEVICES=0).
    generate_jit = jax.jit(generate)
    _, decode_fn = vae_enc_decode()
    outdir = Path(args.outdir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    # These files are evaluation references only; the true target is never
    # passed to the generator. The reconstruction separates VAE information
    # loss from generator-training failure.
    target_image_path = Path(record["target_path"])
    with Image.open(target_image_path) as target_image:
        target_image.convert("RGB").save(outdir / "target_original.png")
    if latent_root is not None:
        target_latent_path = (
            latent_cache_path(latent_root, target_image_path, image_root)
        )
        if not target_latent_path.is_file():
            raise FileNotFoundError(f"Missing cached target latent: {target_latent_path}")
        target_latent = np.load(target_latent_path, allow_pickle=False)
        target_reconstruction = decode_fn(
            jnp.asarray(target_latent[None]).transpose(0, 3, 1, 2)
        )
        target_reconstruction = jnp.clip((target_reconstruction + 1) / 2, 0, 1)
        target_reconstruction = _to_uint8_images(
            np.asarray(target_reconstruction)
        )[0]
        Image.fromarray(target_reconstruction).save(
            outdir / "target_vae_reconstruction.png"
        )
    (outdir / "conditions.txt").write_text(
        "\n".join(item["path"] for item in selected) + "\n", encoding="utf-8"
    )
    saved = 0
    while saved < args.num_samples:
        batch = min(args.batch_size, args.num_samples - saved)
        latents = generate_jit(
            jnp.zeros((batch,), dtype=jnp.int32),
            jnp.repeat(source_single, batch, axis=0),
            jnp.repeat(mask_single, batch, axis=0),
            jnp.repeat(metadata_single, batch, axis=0),
            jnp.repeat(target_coord, batch, axis=0),
            params,
            jax.random.PRNGKey(args.seed + saved),
        )
        # Decode outside the generator JIT. This avoids capturing the large VAE
        # parameter tree in the same compiled closure and lowers peak memory.
        decoded = decode_fn(latents.transpose(0, 3, 1, 2))
        images = jnp.clip((decoded + 1) / 2, 0, 1)
        images = _to_uint8_images(np.asarray(images))
        for index, image in enumerate(images):
            Image.fromarray(image).save(outdir / f"sample_{saved + index:06d}.png")
        saved += batch
    print(f"Saved {saved} samples for {args.target_id} to {outdir}")


if __name__ == "__main__":
    main()
