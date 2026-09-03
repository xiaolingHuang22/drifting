from pathlib import Path

import jax
import optax

from dataset.dataset import create_imagenet_split
from dataset.conditional_imagefolder import create_conditional_imagefolder_split
from dataset.paired_spectrogram import create_paired_spectrogram_split
from utils.logging import WandbLogger
from utils.misc import EasyDict


def create_learning_rate_fn(
    learning_rate,
    warmup_steps,
    total_steps,
    lr_schedule="const",
):
    """Create warmup + main learning-rate schedule."""
    warmup_init_value = min(1e-6, learning_rate)
    warmup_fn = optax.linear_schedule(
        init_value=warmup_init_value,
        end_value=learning_rate,
        transition_steps=warmup_steps,
    )
    if lr_schedule in ["cosine", "cos"]:
        cosine_steps = max(total_steps - warmup_steps, 1)
        schedule_fn = optax.cosine_decay_schedule(
            init_value=learning_rate,
            decay_steps=cosine_steps,
            alpha=1e-6,
        )
    elif lr_schedule == "const":
        schedule_fn = optax.constant_schedule(value=learning_rate)
    else:
        raise NotImplementedError(lr_schedule)

    return optax.join_schedules(
        schedules=[warmup_fn, schedule_fn],
        boundaries=[warmup_steps],
    )


def build_model_dict(config, model_class, *, workdir: str = "runs"):
    """Build model, datasets, optimizer, and logger from config."""
    print("Building model...")
    model = model_class(
        num_classes=config.dataset.num_classes,
        **config.model,
    )

    print("Building dataset...")
    batch_size_per_node = config.dataset.batch_size // jax.process_count()
    resolution = int(config.dataset.resolution)
    use_aug = bool(config.dataset.get("use_aug", False))
    use_hflip = bool(config.dataset.get("use_hflip", True))
    use_latent = bool(config.dataset.get("use_latent", False))
    use_cache = bool(config.dataset.get("use_cache", False))

    dataset_mode = str(config.dataset.get("mode", "imagenet")).lower()
    dataset_kwargs = dict(config.dataset.get("kwargs", {}))
    if dataset_mode in {"imagenet", "imagefolder"}:
        train_loader, preprocess_fn, postprocess_fn = create_imagenet_split(
            resolution=resolution,
            use_aug=use_aug,
            use_hflip=use_hflip,
            use_latent=use_latent,
            use_cache=use_cache,
            batch_size=batch_size_per_node,
            split="train",
            **dataset_kwargs,
        )

        eval_loader, _, _ = create_imagenet_split(
            resolution=resolution,
            use_aug=use_aug,
            use_hflip=use_hflip,
            use_latent=use_latent,
            use_cache=use_cache,
            batch_size=config.dataset.eval_batch_size // jax.process_count(),
            split="val",
            **dataset_kwargs,
        )
        dataset_name = f"imagenet{resolution}"
    elif dataset_mode == "conditional_imagefolder":
        if use_latent or use_cache:
            raise ValueError(
                "dataset.mode=conditional_imagefolder currently supports pixel-space loading only."
            )
        data_path = config.dataset.get("data_path", None)
        if data_path is None:
            raise ValueError(
                "dataset.data_path is required when dataset.mode=conditional_imagefolder."
            )
        k_conditions = int(config.dataset.get("k_conditions", 4))
        k_positive = int(config.dataset.get("k_positive", 4))
        k_neg = int(config.dataset.get("k_neg", 4))
        split_seed = int(config.dataset.get("seed", config.train.get("seed", 42)))
        train_loader, preprocess_fn, postprocess_fn = create_conditional_imagefolder_split(
            data_path=data_path,
            resolution=resolution,
            use_aug=use_aug,
            use_hflip=use_hflip,
            batch_size=batch_size_per_node,
            split="train",
            k_conditions=k_conditions,
            k_positive=k_positive,
            k_neg=k_neg,
            seed=split_seed,
            **dataset_kwargs,
        )
        eval_loader, _, _ = create_conditional_imagefolder_split(
            data_path=data_path,
            resolution=resolution,
            use_aug=False,
            use_hflip=False,
            batch_size=config.dataset.eval_batch_size // jax.process_count(),
            split="val",
            k_conditions=k_conditions,
            k_positive=k_positive,
            k_neg=k_neg,
            seed=split_seed,
            **dataset_kwargs,
        )
        if train_loader.dataset.class_names != eval_loader.dataset.class_names:
            raise ValueError(
                "conditional_imagefolder train/val class folders must match exactly: "
                f"train={train_loader.dataset.class_names}, "
                f"val={eval_loader.dataset.class_names}."
            )
        dataset_name = f"conditional_imagefolder{resolution}"
    elif dataset_mode == "paired_spectrogram":
        if use_latent or use_cache:
            raise ValueError(
                "dataset.mode=paired_spectrogram currently supports pixel-space loading only."
            )
        residual_path = config.dataset.get("residual_path", None)
        train_table_path = config.dataset.get(
            "train_table_path",
            config.dataset.get("table_path", None),
        )
        val_table_path = config.dataset.get("val_table_path", train_table_path)
        if residual_path is None:
            raise ValueError(
                "dataset.residual_path is required when dataset.mode=paired_spectrogram."
            )
        if train_table_path is None:
            raise ValueError(
                "dataset.train_table_path or dataset.table_path is required "
                "when dataset.mode=paired_spectrogram."
            )
        k_neg = int(config.dataset.get("k_neg", config.train.get("neg_per_sample", 4)))
        coordinate_tolerance = float(config.dataset.get("coordinate_tolerance", 1e-5))
        train_loader, preprocess_fn, postprocess_fn = create_paired_spectrogram_split(
            residual_path=residual_path,
            table_path=train_table_path,
            resolution=resolution,
            use_aug=use_aug,
            use_hflip=use_hflip,
            batch_size=batch_size_per_node,
            split="train",
            k_neg=k_neg,
            coordinate_tolerance=coordinate_tolerance,
            **dataset_kwargs,
        )
        eval_loader, _, _ = create_paired_spectrogram_split(
            residual_path=residual_path,
            table_path=val_table_path,
            resolution=resolution,
            use_aug=False,
            use_hflip=False,
            batch_size=config.dataset.eval_batch_size // jax.process_count(),
            split="val",
            k_neg=k_neg,
            coordinate_tolerance=coordinate_tolerance,
            **dataset_kwargs,
        )
        dataset_name = f"paired_spectrogram{resolution}"
    else:
        raise ValueError(f"Unsupported dataset.mode={dataset_mode!r}.")

    learning_rate_fn = create_learning_rate_fn(**config.optimizer.lr_schedule)

    optimizer = optax.adamw(
        learning_rate=learning_rate_fn,
        weight_decay=config.optimizer.get("weight_decay", 0.0),
        b1=config.optimizer.adam_b1,
        b2=config.optimizer.adam_b2,
    )

    logger = WandbLogger()
    w_cfg = EasyDict(dict(config.get("logging", {})))
    use_wandb = bool(w_cfg.get("use_wandb", config.get("use_wandb", True)))
    if "use_wandb" in w_cfg:
        del w_cfg["use_wandb"]
    output_root = Path(workdir).resolve()
    logger.set_logging(
        config=config,
        use_wandb=use_wandb,
        workdir=str(output_root),
        **w_cfg,
    )

    return EasyDict(
        model=model,
        optimizer=optimizer,
        logger=logger,
        eval_loader=eval_loader,
        train_loader=train_loader,
        dataset_name=dataset_name,
        preprocess_fn=preprocess_fn,
        postprocess_fn=postprocess_fn,
        train=config.train,
        learning_rate_fn=learning_rate_fn,
        feature=config.get("feature", {}),
    )
