"""Conditional ImageFolder dataset with averaged same-class conditions."""

from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import Callable

import jax
import jax.numpy as jnp
import torch
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from dataset.dataset import _build_transforms, worker_init_fn
from utils.logging import log_for_0


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class ConditionalImageFolderDataset(torch.utils.data.Dataset):
    """Build conditional examples from class-organized image folders.

    Every image is a target. For that target, the dataset samples
    ``k_conditions`` other images from the same class and averages their
    transformed tensors into one source condition. ``k_positive`` further
    same-class images are used as ``target_ref`` for the drifting attraction
    term, while
    ``k_neg`` images from other classes form the negative set.

    The expected root is one split of an ImageFolder-style dataset, for example
    ``StanfordDogs_split/train/<breed>/*.jpg``.
    """

    def __init__(
        self,
        root: str | Path,
        *,
        transform: Callable[[Image.Image], torch.Tensor],
        k_conditions: int = 4,
        k_positive: int = 4,
        k_neg: int = 4,
        deterministic: bool = False,
        seed: int = 42,
    ) -> None:
        self.root = Path(root).expanduser().resolve()
        self.transform = transform
        self.k_conditions = int(k_conditions)
        self.k_positive = int(k_positive)
        self.k_neg = int(k_neg)
        self.deterministic = bool(deterministic)
        self.seed = int(seed)

        if not self.root.is_dir():
            raise FileNotFoundError(f"Conditional image split does not exist: {self.root}")
        if self.k_conditions <= 0:
            raise ValueError(f"k_conditions must be positive, got {self.k_conditions}.")
        if self.k_positive <= 0:
            raise ValueError(f"k_positive must be positive, got {self.k_positive}.")
        if self.k_neg <= 0:
            raise ValueError(f"k_neg must be positive, got {self.k_neg}.")

        self.class_names = sorted(path.name for path in self.root.iterdir() if path.is_dir())
        if len(self.class_names) < 2:
            raise ValueError(
                f"Conditional training needs at least two class folders for negatives; "
                f"found {len(self.class_names)} under {self.root}."
            )

        self.class_to_idx = {name: index for index, name in enumerate(self.class_names)}
        self.images_by_class: dict[int, list[Path]] = {}
        self.targets: list[tuple[Path, int]] = []
        for class_name in self.class_names:
            class_idx = self.class_to_idx[class_name]
            paths = sorted(
                path
                for path in (self.root / class_name).rglob("*")
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
            )
            minimum = 1 + self.k_conditions + self.k_positive
            if len(paths) < minimum:
                raise ValueError(
                    f"Class {class_name!r} has {len(paths)} images; at least {minimum} "
                    f"are required for k_conditions={self.k_conditions} and "
                    f"k_positive={self.k_positive}."
                )
            self.images_by_class[class_idx] = paths
            self.targets.extend((path, class_idx) for path in paths)

        self.negative_pool = {
            class_idx: [
                path
                for other_idx, paths in self.images_by_class.items()
                if other_idx != class_idx
                for path in paths
            ]
            for class_idx in self.images_by_class
        }

    def __len__(self) -> int:
        return len(self.targets)

    def _generator(self, idx: int) -> torch.Generator | None:
        if not self.deterministic:
            return None
        generator = torch.Generator()
        generator.manual_seed(self.seed + idx)
        return generator

    @staticmethod
    def _sample_indices(
        population: int,
        count: int,
        *,
        generator: torch.Generator | None,
    ) -> list[int]:
        if population <= 0:
            raise ValueError("Cannot sample from an empty image pool.")
        if population >= count:
            return torch.randperm(population, generator=generator)[:count].tolist()
        return torch.randint(population, (count,), generator=generator).tolist()

    def _load_image(self, path: Path) -> torch.Tensor:
        with Image.open(path) as image:
            tensor = self.transform(image.convert("RGB"))
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(
                "ConditionalImageFolderDataset transform must return a torch.Tensor, "
                f"got {type(tensor).__name__}."
            )
        return tensor

    def __getitem__(self, idx: int) -> dict[str, object]:
        target_path, class_idx = self.targets[idx]
        generator = self._generator(idx)

        same_class_pool = [
            path for path in self.images_by_class[class_idx] if path != target_path
        ]
        # Reserve independent same-breed images for the drift positive set.
        selected = self._sample_indices(
            len(same_class_pool),
            self.k_conditions + self.k_positive,
            generator=generator,
        )
        condition_paths = [same_class_pool[index] for index in selected[: self.k_conditions]]
        target_ref_paths = [
            same_class_pool[index]
            for index in selected[self.k_conditions :]
        ]

        negative_pool = self.negative_pool[class_idx]
        negative_indices = self._sample_indices(
            len(negative_pool),
            self.k_neg,
            generator=generator,
        )
        negative_paths = [negative_pool[index] for index in negative_indices]

        conditions = torch.stack([self._load_image(path) for path in condition_paths])
        source = conditions.mean(dim=0)
        target_true = self._load_image(target_path)
        target_ref = torch.stack([self._load_image(path) for path in target_ref_paths])
        negative = torch.stack([self._load_image(path) for path in negative_paths])

        return {
            "source": source,
            "target_ref": target_ref,
            "target_true": target_true,
            "has_target_true": torch.tensor(1.0, dtype=torch.float32),
            "negative": negative,
            # The conditional train-step currently accepts coordinate keys. Dogs
            # have no anatomical coordinates, and use_coord_cond should be false.
            "source_coord": torch.zeros(3, dtype=torch.float32),
            "target_coord": torch.zeros(3, dtype=torch.float32),
            # Keep the generator class label constant so breed information must
            # come from the image condition rather than a class embedding.
            "label": torch.tensor(0, dtype=torch.int64),
            "breed_index": torch.tensor(class_idx, dtype=torch.int64),
            "breed": self.class_names[class_idx],
            "target_path": str(target_path),
        }


def create_conditional_imagefolder_split(
    *,
    data_path: str | Path,
    resolution: int,
    batch_size: int,
    split: str,
    k_conditions: int = 4,
    k_positive: int = 4,
    k_neg: int = 4,
    use_aug: bool = False,
    use_hflip: bool = False,
    seed: int = 42,
    num_workers: int = 4,
    prefetch_factor: int = 2,
    pin_memory: bool = False,
):
    """Create a conditional ImageFolder loader for one train/val/test split."""
    transform = _build_transforms(
        resolution=resolution,
        use_aug=use_aug,
        split=split,
        use_hflip=use_hflip,
    )
    dataset = ConditionalImageFolderDataset(
        Path(data_path) / split,
        transform=transform,
        k_conditions=k_conditions,
        k_positive=k_positive,
        k_neg=k_neg,
        deterministic=(split != "train"),
        seed=seed,
    )
    log_for_0(dataset)

    rank = jax.process_index()
    sampler = DistributedSampler(
        dataset,
        num_replicas=jax.process_count(),
        rank=rank,
        shuffle=(split == "train"),
        seed=seed,
    )
    loader_kwargs = {
        "dataset": dataset,
        "batch_size": batch_size,
        "drop_last": (split == "train"),
        "worker_init_fn": partial(worker_init_fn, rank=rank),
        "sampler": sampler,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": num_workers > 0,
    }
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = prefetch_factor
        loader_kwargs["multiprocessing_context"] = "spawn"
    loader = DataLoader(**loader_kwargs)

    def _to_bhwc(tensor):
        return jnp.asarray(tensor).transpose(0, 2, 3, 1)

    def _to_bkhwc(tensor):
        return jnp.asarray(tensor).transpose(0, 1, 3, 4, 2)

    def preprocess_fn(batch, rng=jax.random.PRNGKey(0)):
        del rng
        target_ref = _to_bkhwc(batch["target_ref"])
        return {
            "images": target_ref[:, 0],
            "labels": jnp.asarray(batch["label"], dtype=jnp.int32),
            "source": _to_bhwc(batch["source"]),
            "target_ref": target_ref,
            "target_true": _to_bhwc(batch["target_true"]),
            "has_target_true": jnp.asarray(batch["has_target_true"], dtype=jnp.float32),
            "negative": _to_bkhwc(batch["negative"]),
            "source_coord": jnp.asarray(batch["source_coord"], dtype=jnp.float32),
            "target_coord": jnp.asarray(batch["target_coord"], dtype=jnp.float32),
            "breed_index": jnp.asarray(batch["breed_index"], dtype=jnp.int32),
        }

    def postprocess_fn(images):
        return jnp.clip((images + 1) / 2, 0, 1).transpose(0, 3, 1, 2)

    return loader, preprocess_fn, postprocess_fn
