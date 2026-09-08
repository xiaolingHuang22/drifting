"""Generate and save images without running FID/IS/PR evaluation.

Examples:
  # mixed-class cycling mode (default)
  python sample_only.py --init-from runs/gen_imagenet256 --outdir runs/samples_mix

  # fixed class mode
  python sample_only.py --init-from runs/gen_imagenet256 --outdir runs/samples_cls3 --class-id 3

  # conditional mode: channel-concatenate four images from one test breed folder
  python sample_only.py --init-from runs/dogs/params_ema --outdir runs/dogs_test \
    --condition-dir StanfordDogs_split/test/n02085620-Chihuahua --num-conditions 4
"""

from __future__ import annotations

import argparse
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from PIL import Image

from dataset.dataset import get_postprocess_fn
from dataset.conditional_imagefolder import (
    IMAGE_EXTENSIONS,
    build_conditional_image_transform,
    compute_dataset_min_max,
)
from inference import _is_latent, generate_step
from utils.env import HF_ROOT
from utils.hsdp_util import ddp_shard, set_global_mesh
from utils.init_util import load_generator_model_and_params
from utils.misc import load_config, prepare_rng, run_init


def _pick_labels(
    *,
    start: int,
    batch_size: int,
    class_id: int | None,
    cycle_classes: bool,
    num_classes: int,
) -> jnp.ndarray:
    if class_id is not None:
        if class_id < 0 or class_id >= num_classes:
            raise ValueError(f"class_id must be in [0, {num_classes - 1}], got {class_id}.")
        return jnp.full((batch_size,), class_id, dtype=jnp.int32)

    if cycle_classes:
        return (jnp.arange(start, start + batch_size, dtype=jnp.int32) % num_classes).astype(jnp.int32)

    # default fallback: class 0
    return jnp.zeros((batch_size,), dtype=jnp.int32)


def _to_uint8_images(images: np.ndarray) -> np.ndarray:
    # Supports either BHWC or BCHW output.
    if images.ndim != 4:
        raise ValueError(f"Expected 4D output, got shape {images.shape}.")
    if images.shape[1] in (1, 3):  # BCHW -> BHWC
        images = np.transpose(images, (0, 2, 3, 1))
    images = np.asarray(images, dtype=np.float32)
    bad = ~np.isfinite(images)
    if np.any(bad):
        print(f"[warn] sample_only received non-finite pixels: {int(bad.sum())}")
    images = np.nan_to_num(images, nan=0.0, posinf=1.0, neginf=0.0)
    images = np.clip(images, 0.0, 1.0)
    return (images * 255.0).astype(np.uint8)


def _load_condition_image(
    path: str | Path,
    *,
    image_size: int,
    channels: int,
    pixel_range: str,
    normalization_min: float = 0.0,
    normalization_max: float = 1.0,
) -> jnp.ndarray:
    """Load one condition image with the validation transform used in training."""
    if channels not in (1, 3):
        raise ValueError(
            "sample_only.py currently supports source_in_channels of 1 or 3, "
            f"got {channels}."
        )
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Condition image does not exist: {path}")
    transform = build_conditional_image_transform(
        image_size,
        split="test",
        use_aug=False,
        use_hflip=False,
        normalization_min=normalization_min,
        normalization_max=normalization_max,
    )
    with Image.open(path) as image:
        tensor = transform(image.convert("L" if channels == 1 else "RGB"))
    arr = np.asarray(tensor, dtype=np.float32).transpose(1, 2, 0)
    if pixel_range == "minus_one_one":
        arr = arr * 2.0 - 1.0
    elif pixel_range != "zero_one":
        raise ValueError(f"Unsupported model.pixel_range={pixel_range!r}.")
    return jnp.asarray(arr[None, ...], dtype=jnp.float32)


def _select_condition_paths(args: argparse.Namespace) -> list[Path]:
    """Resolve explicit conditions or sample conditions from a test breed folder."""
    if args.num_conditions <= 0:
        raise ValueError(f"--num-conditions must be positive, got {args.num_conditions}.")
    if args.source_img:
        paths = [Path(path).expanduser().resolve() for path in args.source_img]
    else:
        condition_dir = args.condition_dir
        if condition_dir is None and args.target_img is not None:
            condition_dir = str(Path(args.target_img).expanduser().resolve().parent)
        if condition_dir is None:
            raise ValueError(
                "Conditional checkpoints require --source-img, --condition-dir, "
                "or --target-img (whose parent breed folder is used)."
            )
        directory = Path(condition_dir).expanduser().resolve()
        if not directory.is_dir():
            raise FileNotFoundError(f"Condition directory does not exist: {directory}")
        target_path = (
            Path(args.target_img).expanduser().resolve()
            if args.target_img is not None
            else None
        )
        candidates = sorted(
            path
            for path in directory.rglob("*")
            if path.is_file()
            and path.suffix.lower() in IMAGE_EXTENSIONS
            and path.resolve() != target_path
        )
        if len(candidates) < args.num_conditions:
            raise ValueError(
                f"Need {args.num_conditions} condition images, but found "
                f"{len(candidates)} eligible files in {directory}."
            )
        rng = np.random.default_rng(args.condition_seed)
        indices = rng.choice(len(candidates), size=args.num_conditions, replace=False)
        paths = [candidates[int(index)] for index in indices]

    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Condition images do not exist: {missing}")
    if not paths:
        raise ValueError("At least one condition image is required.")
    if args.target_img is not None:
        target_path = Path(args.target_img).expanduser().resolve()
        if target_path in paths:
            raise ValueError(
                f"Held-out target {target_path} must not also be a condition image."
            )
    return paths


def conditional_generate_step(
    labels,
    source,
    source_coord,
    target_coord,
    params,
    rng,
    apply_fn,
    postprocess_fn,
    cfg_scale=1.0,
):
    """Generate conditional samples from source spectrograms and coordinates."""
    latent_samples = apply_fn(
        {"params": params},
        train=False,
        rngs=prepare_rng(rng, ["noise"]),
        c=labels,
        cfg_scale=cfg_scale,
        source=source,
        source_coord=source_coord,
        target_coord=target_coord,
    )["samples"]
    latent_samples = jax.tree_util.tree_map(
        lambda x: jax.lax.with_sharding_constraint(x, ddp_shard()),
        latent_samples,
    )
    return postprocess_fn(latent_samples)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate-only sampler (no FID).")
    parser.add_argument("--init-from", required=True, help="Local artifact dir or hf:// model id.")
    parser.add_argument(
        "--config",
        default=None,
        help=(
            "Training YAML used to reconstruct the model when --init-from points "
            "to a raw checkpoints directory."
        ),
    )
    parser.add_argument(
        "--checkpoint-step",
        type=int,
        default=None,
        help="Load this exact Flax checkpoint step from --init-from.",
    )
    parser.add_argument("--outdir", default="runs/sample_only", help="Output image directory.")
    parser.add_argument("--num-samples", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--cfg-scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--class-id",
        type=int,
        default=None,
        help="Fixed class label to sample. If omitted, labels cycle 0..num_classes-1.",
    )
    parser.add_argument(
        "--cycle-classes",
        action="store_true",
        help="Cycle labels across classes when --class-id is not provided.",
    )
    parser.add_argument("--hsdp-dim", type=int, default=None)
    condition_group = parser.add_mutually_exclusive_group()
    condition_group.add_argument(
        "--source-img",
        nargs="+",
        default=None,
        help="One or more explicit condition images; their channels are concatenated.",
    )
    condition_group.add_argument(
        "--condition-dir",
        default=None,
        help="Breed directory to sample condition images from, usually under test/.",
    )
    parser.add_argument("--num-conditions", type=int, default=4)
    parser.add_argument("--condition-seed", type=int, default=42)
    parser.add_argument(
        "--target-img",
        default=None,
        help=(
            "Optional held-out target for visual comparison. It is excluded from "
            "condition-dir sampling and is never passed to the generator."
        ),
    )
    parser.add_argument(
        "--source-coord",
        nargs=3,
        type=float,
        default=None,
        metavar=("X", "Y", "Z"),
        help="Source coordinate ls for conditional coordinate conditioning.",
    )
    parser.add_argument(
        "--target-coord",
        nargs=3,
        type=float,
        default=None,
        metavar=("X", "Y", "Z"),
        help="Destination coordinate lr for conditional coordinate conditioning.",
    )
    args = parser.parse_args()

    run_init()
    hsdp = args.hsdp_dim or min(8, jax.local_device_count() * jax.process_count())
    set_global_mesh(hsdp)

    config_model = None
    normalization_min, normalization_max = 0.0, 1.0
    if args.config is not None:
        config = load_config(args.config)
        config_model = dict(config.model)
        if str(config.dataset.get("mode", "")).lower() == "conditional_imagefolder":
            normalization_min, normalization_max = compute_dataset_min_max(
                config.dataset.data_path
            )
        # Training injects dataset.num_classes when constructing DitGen rather
        # than storing it under config.model. Raw checkpoints have no artifact
        # metadata, so reproduce that injection before rebuilding the model.
        config_model["num_classes"] = int(config.dataset.num_classes)
    model, params, metadata = load_generator_model_and_params(
        args.init_from,
        hf_cache_dir=HF_ROOT,
        model_config=config_model,
        checkpoint_step=args.checkpoint_step,
    )
    model_cfg = dict(metadata.get("model_config", {}) or {})
    num_classes = int(model_cfg.get("num_classes", 1000))
    latent = _is_latent(metadata)
    pixel_range = str(model_cfg.get("pixel_range", "minus_one_one"))
    if not latent and pixel_range == "zero_one":
        postprocess_fn = lambda images: jnp.clip(images, 0, 1).transpose(0, 3, 1, 2)
    else:
        postprocess_fn = get_postprocess_fn(use_aug=False, use_latent=False, use_cache=latent)
    gen_step_jit = jax.jit(
        lambda batch, params, rng, cfg_scale: generate_step(
            batch,
            params=params,
            rng=rng,
            apply_fn=model.apply,
            postprocess_fn=postprocess_fn,
            cfg_scale=cfg_scale,
        )
    )
    conditional = (
        int(model_cfg.get("source_in_channels", 0)) > 0
        or args.source_img is not None
        or args.condition_dir is not None
        or args.target_img is not None
    )
    if conditional:
        source_in_channels = int(model_cfg.get("source_in_channels", 3))
        if bool(model_cfg.get("use_coord_cond", False)) and (
            args.source_coord is None or args.target_coord is None
        ):
            raise ValueError(
                "--source-coord and --target-coord are required when "
                "model.use_coord_cond=true."
            )
        image_size = int(model_cfg.get("input_size", 256))
        condition_paths = _select_condition_paths(args)
        condition_channels = 3
        if args.config is not None:
            condition_channels = int(config.dataset.get("condition_channels", 3))
        expected_source_channels = len(condition_paths) * condition_channels
        if expected_source_channels != source_in_channels:
            raise ValueError(
                f"Selected {len(condition_paths)} conditions with {condition_channels} "
                f"channels each, producing {expected_source_channels} source channels; "
                f"the checkpoint expects {source_in_channels}."
            )
        condition_tensors = [
            _load_condition_image(
                path,
                image_size=image_size,
                channels=condition_channels,
                pixel_range=pixel_range,
                normalization_min=normalization_min,
                normalization_max=normalization_max,
            )
            for path in condition_paths
        ]
        source_single = jnp.concatenate(condition_tensors, axis=-1)
        coord_dim = int(model_cfg.get("coord_dim", 3))
        source_coord_single = jnp.asarray(
            [args.source_coord or [0.0] * coord_dim],
            dtype=jnp.float32,
        )
        target_coord_single = jnp.asarray(
            [args.target_coord or [0.0] * coord_dim],
            dtype=jnp.float32,
        )
        cond_step_jit = jax.jit(
            lambda labels, source, source_coord, target_coord, params, rng, cfg_scale: conditional_generate_step(
                labels,
                source,
                source_coord,
                target_coord,
                params=params,
                rng=rng,
                apply_fn=model.apply,
                postprocess_fn=postprocess_fn,
                cfg_scale=cfg_scale,
            )
        )

    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    if conditional:
        (outdir / "conditions.txt").write_text(
            "\n".join(str(path) for path in condition_paths) + "\n",
            encoding="utf-8",
        )
        # Save exactly the transformed tensors that were concatenated. This makes
        # it possible to distinguish surprising source content from an image
        # selection or preprocessing bug.
        for index, (condition_path, condition_tensor) in enumerate(
            zip(condition_paths, condition_tensors)
        ):
            preview_tensor = (
                (condition_tensor + 1.0) / 2.0
                if pixel_range == "minus_one_one"
                else condition_tensor
            )
            condition_image = _to_uint8_images(np.asarray(preview_tensor))[0]
            safe_stem = "".join(
                char if char.isalnum() or char in {"-", "_"} else "_"
                for char in condition_path.stem
            )
            Image.fromarray(condition_image).save(
                outdir / f"condition_{index:02d}_{safe_stem}.png"
            )
        if args.target_img is not None:
            target_preview = _load_condition_image(
                args.target_img,
                image_size=image_size,
                channels=condition_channels,
                pixel_range=pixel_range,
                normalization_min=normalization_min,
                normalization_max=normalization_max,
            )
            if pixel_range == "minus_one_one":
                target_preview = (target_preview + 1.0) / 2.0
            target_preview = _to_uint8_images(np.asarray(target_preview))[0]
            Image.fromarray(target_preview).save(outdir / "target_held_out.png")

    saved = 0
    sample_idx = 0
    while saved < args.num_samples:
        cur_bsz = min(args.batch_size, args.num_samples - saved)
        labels = _pick_labels(
            start=sample_idx,
            batch_size=cur_bsz,
            class_id=args.class_id,
            cycle_classes=args.cycle_classes or args.class_id is None,
            num_classes=num_classes,
        )
        rng = jax.random.PRNGKey(args.seed + sample_idx)
        if conditional:
            source = jnp.repeat(source_single, cur_bsz, axis=0)
            source_coord = jnp.repeat(source_coord_single, cur_bsz, axis=0)
            target_coord = jnp.repeat(target_coord_single, cur_bsz, axis=0)
            images = cond_step_jit(
                labels,
                source,
                source_coord,
                target_coord,
                params=params,
                rng=rng,
                cfg_scale=args.cfg_scale,
            )
        else:
            # generate_step ignores batch[0] and only uses labels.
            dummy_images = jnp.zeros((cur_bsz, 1, 1, 1), dtype=jnp.float32)
            batch = (dummy_images, labels)
            images = gen_step_jit(batch, params=params, rng=rng, cfg_scale=args.cfg_scale)
        images = _to_uint8_images(np.asarray(images))

        for i in range(cur_bsz):
            class_label = int(labels[i])
            Image.fromarray(images[i]).save(outdir / f"sample_{saved + i:06d}_class{class_label}.png")

        saved += cur_bsz
        sample_idx += cur_bsz

    print(f"Saved {saved} images to {outdir}")


if __name__ == "__main__":
    main()
