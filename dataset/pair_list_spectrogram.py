"""Patient-disjoint pair-list SEEG dataset for conditional drifting.

The CSV is grouped by ``patient_channel``.  Each unique target channel becomes
one example and all corresponding ``pair`` rows become its variable-length
condition set.  Split manifests are patient based so a held-out patient cannot
appear as either a target or a donor in another split.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
from collections import defaultdict
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms

from dataset.dataset import worker_init_fn


REQUIRED_COLUMNS = {
    "patient_channel", "location", "x1", "y1", "z1", "pair",
    "pair_location", "x2", "y2", "z2", "distance",
}


def _channel(value: str) -> tuple[str, str]:
    parts = [part.strip() for part in str(value).split(",", 1)]
    if len(parts) != 2 or not all(parts):
        raise ValueError(f"Expected 'patient,channel', got {value!r}.")
    return parts[0], parts[1]


def load_patient_split(path):
    manifest = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    seen = set()
    for split in ("train", "val", "test"):
        patients = manifest.get(split)
        if not isinstance(patients, list) or not all(isinstance(p, str) for p in patients):
            raise ValueError(f"Split {split} must contain a list of patient identifiers.")
        if len(set(patients)) != len(patients) or seen.intersection(patients):
            raise ValueError(f"Duplicate or overlapping patients in split {split}.")
        seen.update(patients)
    return manifest


def image_paths(root, patient, channel, manifest=None):
    if manifest is not None:
        split = next((s for s in ("train", "val", "test") if patient in manifest[s]), None)
        if split is None:
            raise ValueError(f"Pair-table patient {patient!r} is absent from the split manifest.")
        # Patient and channel identifiers come from directories, never filename parsing.
        folder = root / split / patient / channel
        paths = sorted(p for p in folder.glob("*") if p.suffix.lower() in (".png", ".jpg", ".jpeg") and p.is_file())
    else:
        paths = [root / patient / f"{patient}_{channel}.png"]
        paths = [p for p in paths if p.is_file()]
    if not paths:
        raise FileNotFoundError(f"No spectrogram images for {patient},{channel} under {root}")
    return [str(p) for p in paths]


def latent_cache_path(latent_root, image_path, image_root=None):
    image_path = Path(image_path)
    relative = image_path.relative_to(Path(image_root).expanduser().resolve()) if image_root else Path(image_path.parent.name) / image_path.name
    return Path(latent_root) / relative.with_suffix(".npy")


def select_conditions(record, max_conditions, rng, training=False):
    conditions = list(record["conditions"])
    if training:
        rng.shuffle(conditions)
    selected = []
    for condition in conditions[:max_conditions]:
        item = dict(condition)
        item["path"] = rng.choice(item.get("paths", [item["path"]]))
        selected.append(item)
    return selected


def build_pair_list_records(table_path: str | Path, image_root: str | Path, split_manifest=None):
    """Group channel metadata, then expand segmented channels into image records."""
    table_path = Path(table_path).expanduser().resolve()
    image_root = Path(image_root).expanduser().resolve()
    manifest = load_patient_split(split_manifest) if split_manifest else None
    if manifest is not None:
        for split in ("train", "val", "test"):
            for folder in (image_root / split).iterdir():
                if folder.is_dir() and folder.name not in manifest[split]:
                    raise ValueError(f"Patient folder {folder} disagrees with the split manifest.")
    path_index = {}
    def paths(patient, channel):
        key = (patient, channel)
        if key not in path_index:
            path_index[key] = image_paths(image_root, patient, channel, manifest)
        return path_index[key]
    grouped = {}
    with table_path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        missing = REQUIRED_COLUMNS - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"Pair table is missing columns: {sorted(missing)}")
        for line_no, row in enumerate(reader, start=2):
            tp, tc = _channel(row["patient_channel"])
            cp, cc = _channel(row["pair"])
            if tp == cp:
                raise ValueError(f"Line {line_no}: pair patients must differ ({tp}).")
            if row["location"].strip() != row["pair_location"].strip():
                raise ValueError(f"Line {line_no}: target/pair brain regions differ.")
            target_coord = tuple(float(row[key]) for key in ("x1", "y1", "z1"))
            cond_coord = tuple(float(row[key]) for key in ("x2", "y2", "z2"))
            distance = float(row["distance"])
            calculated = math.dist(target_coord, cond_coord)
            if not math.isclose(distance, calculated, rel_tol=2e-2, abs_tol=0.25):
                raise ValueError(
                    f"Line {line_no}: distance={distance} disagrees with coordinates ({calculated})."
                )
            target_id = f"{tp},{tc}"
            record = grouped.setdefault(
                target_id,
                {
                    "target_id": target_id,
                    "target_patient": tp,
                    "target_channel": tc,
                    "region": row["location"].strip(),
                    "target_coord": target_coord,
                    "target_path": paths(tp, tc)[0],
                    "target_paths": paths(tp, tc),
                    "conditions": {},
                },
            )
            if tuple(record["target_coord"]) != target_coord or record["region"] != row["location"].strip():
                raise ValueError(f"Conflicting target metadata for {target_id}.")
            condition_id = f"{cp},{cc}"
            record["conditions"][condition_id] = {
                "channel_id": condition_id,
                "patient": cp,
                "channel": cc,
                "coord": cond_coord,
                "distance": distance,
                "path": paths(cp, cc)[0],
                "paths": paths(cp, cc),
            }
    records = []
    for record in grouped.values():
        record = dict(record)
        record["conditions"] = sorted(
            record["conditions"].values(), key=lambda item: (item["distance"], item["channel_id"])
        )
        if record["conditions"]:
            for path in record["target_paths"]:
                item = dict(record, target_path=path, channel_id=record["target_id"])
                if manifest is not None:
                    item["target_id"] = f'{record["target_id"]}/{Path(path).name}'
                    match = re.search(r"_(\d+)_(\d+)$", Path(path).stem)
                    if match:
                        item["segment_seconds"] = tuple(map(int, match.groups()))
                records.append(item)
    return sorted(records, key=lambda item: item["target_id"])


def create_patient_split_manifest(
    records,
    output_path: str | Path,
    *,
    seed: int = 42,
    train_fraction: float = 0.70,
    val_fraction: float = 0.15,
):
    """Create a fully patient-disjoint split for target and donor roles."""
    patients = sorted(
        {r["target_patient"] for r in records}
        | {c["patient"] for r in records for c in r["conditions"]}
    )
    rng = random.Random(seed)
    rng.shuffle(patients)
    n = len(patients)
    n_train = max(1, round(n * train_fraction))
    n_val = max(1, round(n * val_fraction))
    if n_train + n_val >= n:
        n_val = max(1, n - n_train - 1)
    manifest = {
        "seed": seed,
        "protocol": "fully_patient_disjoint",
        "train": sorted(patients[:n_train]),
        "val": sorted(patients[n_train:n_train + n_val]),
        "test": sorted(patients[n_train + n_val:]),
    }
    output_path = Path(output_path).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


class PairListSpectrogramDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        records,
        split_patients,
        *,
        max_conditions=8,
        k_neg=4,
        resolution=256,
        latent_root=None,
        training=False,
        image_root=None,
        max_targets=None,
        fixed_conditions=False,
        seed=42,
    ):
        self.fixed_conditions = bool(fixed_conditions)
        self.image_root = image_root
        self.training = bool(training)
        self.seed = int(seed)
        self.max_conditions = int(max_conditions)
        self.k_neg = int(k_neg)
        self.latent_root = (
            Path(latent_root).expanduser().resolve() if latent_root else None
        )
        allowed = set(split_patients)
        self.records = []
        for record in records:
            if record["target_patient"] not in allowed:
                continue
            filtered = [c for c in record["conditions"] if c["patient"] in allowed]
            if filtered:
                item = dict(record)
                item["conditions"] = filtered
                self.records.append(item)
        if not self.records:
            raise ValueError("No pair-list records remain after patient-disjoint filtering.")
        if max_targets is not None:
            if int(max_targets) <= 0:
                raise ValueError("max_targets must be positive.")
            # Seeded sample across patients/channels, rather than the first channel.
            self.records = random.Random(self.seed).sample(self.records, min(int(max_targets), len(self.records)))
        self.regions = sorted({r["region"] for r in self.records})
        self.region_to_idx = {name: i for i, name in enumerate(self.regions)}
        # Store one pool per region, rather than one O(N) list per image.
        self.records_by_region = {
            region: [r for r in self.records if r["region"] == region]
            for region in self.regions
        }
        if len(self.regions) < 2:
            raise ValueError("Need at least two brain regions to sample incompatible negatives.")
        ops = [transforms.Resize((resolution, resolution), antialias=True)]
        if training:
            ops += [transforms.RandomApply([transforms.ColorJitter(brightness=.1, contrast=.1)], p=.5)]
        ops += [transforms.ToTensor(), transforms.Normalize([.5] * 3, [.5] * 3)]
        self.transform = transforms.Compose(ops)

    def __len__(self):
        return len(self.records)

    def _load(self, path):
        if self.latent_root is not None:
            image_path = Path(path)
            latent_path = latent_cache_path(self.latent_root, image_path, self.image_root)
            if not latent_path.is_file():
                raise FileNotFoundError(
                    f"Missing cached VAE latent for {image_path}: {latent_path}. "
                    "Build the cache with python -m dataset.pair_list_latent_cache."
                )
            latent = np.load(latent_path, allow_pickle=False)
            if latent.ndim != 3 or latent.shape[-1] != 4:
                raise ValueError(f"Expected HWC latent with 4 channels, got {latent.shape}: {latent_path}")
            return torch.from_numpy(latent.transpose(2, 0, 1).copy()).float()
        with Image.open(path) as image:
            return self.transform(image.convert("RGB"))

    def __getitem__(self, index):
        record = self.records[index]
        rng = random.Random() if self.training and not self.fixed_conditions else random.Random(self.seed + index)
        conditions = select_conditions(record, self.max_conditions, rng, self.training)
        target = self._load(record["target_path"])
        condition_images = torch.zeros(
            self.max_conditions, target.shape[0], target.shape[1], target.shape[2]
        )
        condition_coords = torch.zeros(self.max_conditions, 3)
        condition_distances = torch.zeros(self.max_conditions, 1)
        condition_mask = torch.zeros(self.max_conditions)
        for i, condition in enumerate(conditions):
            condition_images[i] = self._load(condition["path"])
            condition_coords[i] = torch.tensor(condition["coord"], dtype=torch.float32)
            condition_distances[i, 0] = float(condition["distance"])
            condition_mask[i] = 1.0
        other_regions = [r for r in self.regions if r != record["region"]]
        weights = [len(self.records_by_region[r]) for r in other_regions]
        negatives = [rng.choice(self.records_by_region[rng.choices(other_regions, weights=weights)[0]]) for _ in range(self.k_neg)]
        return {
            "source": condition_images,
            "condition_mask": condition_mask,
            "condition_coord": condition_coords,
            "condition_distance": condition_distances,
            "target_true": target,
            "target_ref": target.unsqueeze(0),
            "negative": torch.stack([self._load(x["target_path"]) for x in negatives]),
            "source_coord": condition_coords[0],
            "target_coord": torch.tensor(record["target_coord"], dtype=torch.float32),
            "region_id": torch.tensor(self.region_to_idx[record["region"]], dtype=torch.int64),
            "labels": torch.tensor(0, dtype=torch.int64),
            "positive_mask": torch.ones(1, dtype=torch.float32),
            "target_id": record["target_id"],
        }


def create_pair_list_spectrogram_split(
    *, table_path, image_root, split_manifest, split, resolution, batch_size,
    max_conditions=8, k_neg=4, use_natural_vae=True, latent_root=None,
    coordinate_scale=100.0, seed=42,
    num_workers=0, prefetch_factor=2, pin_memory=False,
    train_max_targets=None, fixed_conditions=False,
):
    records = build_pair_list_records(table_path, image_root, split_manifest if any((Path(image_root).expanduser() / s).is_dir() for s in ("train", "val", "test")) else None)
    manifest = load_patient_split(split_manifest)
    ds = PairListSpectrogramDataset(
        records, manifest[split], max_conditions=max_conditions, k_neg=k_neg,
        resolution=resolution, latent_root=latent_root,
        image_root=image_root,
        max_targets=train_max_targets if split == "train" else None,
        fixed_conditions=fixed_conditions,
        training=(split == "train"), seed=seed,
    )
    sampler = DistributedSampler(
        ds, num_replicas=jax.process_count(), rank=jax.process_index(), shuffle=(split == "train")
    )
    kwargs = dict(
        dataset=ds, batch_size=batch_size, sampler=sampler, drop_last=(split == "train"),
        num_workers=num_workers, pin_memory=pin_memory,
        worker_init_fn=partial(worker_init_fn, rank=jax.process_index()),
        persistent_workers=(num_workers > 0),
    )
    if num_workers > 0:
        kwargs.update(prefetch_factor=prefetch_factor, multiprocessing_context="spawn")
    loader = DataLoader(**kwargs)
    encode_fn = decode_fn = None
    if latent_root and use_natural_vae:
        raise ValueError("Use either latent_root or online use_natural_vae, not both.")
    if use_natural_vae:
        from dataset.vae import vae_enc_decode
        encode_fn, decode_fn = vae_enc_decode()

    def _encode_bchw(value, rng):
        encoded = encode_fn(value, rng) if encode_fn else jnp.asarray(value)
        # Diffusers returns BCHW; generator convention is BHWC.
        if encoded.ndim == 4 and encoded.shape[1] in (1, 3, 4):
            encoded = encoded.transpose(0, 2, 3, 1)
        return encoded

    def preprocess_fn(batch, rng=jax.random.PRNGKey(0)):
        source = jnp.asarray(batch["source"])
        b, k, c, h, w = source.shape
        keys = jax.random.split(rng, 3)
        source = _encode_bchw(source.reshape(b * k, c, h, w), keys[0])
        source = source.reshape(b, k, *source.shape[1:])
        target = _encode_bchw(jnp.asarray(batch["target_true"]), keys[1])
        neg = jnp.asarray(batch["negative"])
        bn, kn, cn, hn, wn = neg.shape
        neg = _encode_bchw(neg.reshape(bn * kn, cn, hn, wn), keys[2]).reshape(bn, kn, *target.shape[1:])
        return {
            "source": source,
            "condition_mask": jnp.asarray(batch["condition_mask"], dtype=jnp.float32),
            "condition_coord": jnp.asarray(batch["condition_coord"], dtype=jnp.float32) / coordinate_scale,
            "condition_distance": jnp.asarray(batch["condition_distance"], dtype=jnp.float32) / coordinate_scale,
            "target_true": target,
            "target_ref": target[:, None],
            "negative": neg,
            "source_coord": jnp.asarray(batch["source_coord"], dtype=jnp.float32) / coordinate_scale,
            "target_coord": jnp.asarray(batch["target_coord"], dtype=jnp.float32) / coordinate_scale,
            "region_id": jnp.asarray(batch["region_id"], dtype=jnp.int32),
            "labels": jnp.zeros((b,), dtype=jnp.int32),
            "positive_mask": jnp.asarray(batch["positive_mask"], dtype=jnp.float32),
            "has_target_true": jnp.ones((b,), dtype=jnp.float32),
        }

    def postprocess_fn(images):
        if decode_fn:
            decoded = decode_fn(images.transpose(0, 3, 1, 2))
            return jnp.clip((decoded + 1) / 2, 0, 1)
        return jnp.clip((images + 1) / 2, 0, 1).transpose(0, 3, 1, 2)

    return loader, preprocess_fn, postprocess_fn


def main():
    parser = argparse.ArgumentParser(description="Build patient-disjoint SEEG pair-list split manifest.")
    parser.add_argument("--table-path", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    records = build_pair_list_records(args.table_path, args.image_root)
    manifest = create_patient_split_manifest(records, args.output, seed=args.seed)
    summary = {key: len(value) if isinstance(value, list) else value for key, value in manifest.items()}
    for split in ("train", "val", "test"):
        allowed = set(manifest[split])
        retained = [
            (record, [c for c in record["conditions"] if c["patient"] in allowed])
            for record in records
            if record["target_patient"] in allowed
        ]
        retained = [(record, conditions) for record, conditions in retained if conditions]
        summary[f"{split}_targets"] = len(retained)
        summary[f"{split}_pair_edges"] = sum(len(conditions) for _, conditions in retained)
        summary[f"{split}_regions"] = len({record["region"] for record, _ in retained})
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
