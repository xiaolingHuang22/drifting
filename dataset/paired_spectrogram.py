"""Paired SEEG spectrogram dataset for conditional drifting experiments."""

from __future__ import annotations

import csv
from functools import partial
from pathlib import Path
from typing import Callable, Dict, List, Mapping, Sequence, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms

from dataset.dataset import _build_transforms, worker_init_fn
from utils.logging import log_for_0


REQUIRED_COLUMNS = {
    "patient_channel",
    "pair",
    "x1",
    "y1",
    "z1",
    "x2",
    "y2",
    "z2",
}


def _split_patient_channel(value: str, *, column: str, row_number: int) -> Tuple[str, str]:
    """Parse one ``patient,channel`` CSV field."""
    parts = str(value).split(",", 1)
    if len(parts) != 2 or not all(part.strip() for part in parts):
        raise ValueError(
            f"Row {row_number}: column {column!r} must contain "
            f"'patient,channel', got {value!r}."
        )
    return parts[0].strip(), parts[1].strip()


def _coord_from_row(
    row: Mapping[str, str],
    names: Sequence[str],
    *,
    row_number: int,
) -> Tuple[float, float, float]:
    """Read one three-dimensional coordinate from a CSV row."""
    try:
        return tuple(float(row[name]) for name in names)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"Row {row_number}: coordinates {tuple(names)!r} must be numeric."
        ) from error


class PairedSpectrogramDataset(torch.utils.data.Dataset):
    """Load source/reference pairs and source-patient negative samples.

    The CSV schema follows the notation from ``docs/conditional_drifting_plan.md``:

    - ``patient_channel``: ``ps,source_channel`` at coordinate ``(x1, y1, z1)``.
    - ``pair``: ``pr,reference_channel`` at coordinate ``(x2, y2, z2)``.

    Negative samples are spectrograms from ``ps`` whose known coordinate differs
    from the reference coordinate. Coordinates are learned from every source and
    reference occurrence in the CSV. Files without a known coordinate are not
    used as negatives, because they cannot safely be classified as ``ps_not_lr``.

    Returned image tensors use CHW layout and are normalized to ``[-1, 1]`` by
    the default transform. A DataLoader therefore returns ``BCHW`` sources and
    references, and ``B,K,C,H,W`` negatives.
    """

    def __init__(
        self,
        residual_path: str | Path,
        table_path: str | Path,
        k_neg: int = 4,
        transform: Callable[[Image.Image], torch.Tensor] | None = None,
        coordinate_tolerance: float = 1e-5,
    ) -> None:
        self.residual_folder = Path(residual_path).expanduser().resolve()
        self.table_path = Path(table_path).expanduser().resolve()
        self.k_neg = int(k_neg)
        self.coordinate_tolerance = float(coordinate_tolerance)
        self.transform = transform or transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )

        if self.k_neg <= 0:
            raise ValueError(f"k_neg must be positive, got {self.k_neg}.")
        if not self.residual_folder.is_dir():
            raise FileNotFoundError(
                f"Residual folder does not exist: {self.residual_folder}"
            )
        if not self.table_path.is_file():
            raise FileNotFoundError(f"Pair table does not exist: {self.table_path}")

        self.rows = self._read_rows()
        self.channel_coordinates = self._build_channel_coordinate_index()
        self.negative_candidates = self._build_negative_candidate_index()

    def _read_rows(self) -> List[Dict[str, object]]:
        rows: List[Dict[str, object]] = []
        with self.table_path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            fieldnames = set(reader.fieldnames or ())
            missing = REQUIRED_COLUMNS - fieldnames
            if missing:
                raise ValueError(
                    f"Pair table {self.table_path} is missing columns: "
                    f"{sorted(missing)}."
                )

            for row_number, raw_row in enumerate(reader, start=2):
                source_patient, source_channel = _split_patient_channel(
                    raw_row["patient_channel"],
                    column="patient_channel",
                    row_number=row_number,
                )
                ref_patient, ref_channel = _split_patient_channel(
                    raw_row["pair"],
                    column="pair",
                    row_number=row_number,
                )
                source_coord = _coord_from_row(
                    raw_row, ("x1", "y1", "z1"), row_number=row_number
                )
                ref_coord = _coord_from_row(
                    raw_row, ("x2", "y2", "z2"), row_number=row_number
                )
                target_true_value = (
                    raw_row.get("target_true")
                    or raw_row.get("ps_lr")
                    or raw_row.get("target_patient_channel")
                    or ""
                )
                target_true_patient = ""
                target_true_channel = ""
                if str(target_true_value).strip():
                    target_true_patient, target_true_channel = _split_patient_channel(
                        target_true_value,
                        column="target_true",
                        row_number=row_number,
                    )
                rows.append(
                    {
                        "source_patient": source_patient,
                        "source_channel": source_channel,
                        "ref_patient": ref_patient,
                        "ref_channel": ref_channel,
                        "target_true_patient": target_true_patient,
                        "target_true_channel": target_true_channel,
                        "source_coord": source_coord,
                        "ref_coord": ref_coord,
                        "row_number": row_number,
                    }
                )

        if not rows:
            raise ValueError(f"Pair table contains no data rows: {self.table_path}")
        return rows

    def _build_channel_coordinate_index(
        self,
    ) -> Dict[Tuple[str, str], Tuple[float, float, float]]:
        index: Dict[Tuple[str, str], Tuple[float, float, float]] = {}
        for row in self.rows:
            entries = [
                (
                    str(row["source_patient"]),
                    str(row["source_channel"]),
                    row["source_coord"],
                ),
                (
                    str(row["ref_patient"]),
                    str(row["ref_channel"]),
                    row["ref_coord"],
                ),
            ]
            if row.get("target_true_patient") and row.get("target_true_channel"):
                entries.append(
                    (
                        str(row["target_true_patient"]),
                        str(row["target_true_channel"]),
                        row["ref_coord"],
                    )
                )
            for patient, channel, coord_value in entries:
                coord = tuple(float(value) for value in coord_value)
                key = (patient, channel)
                previous = index.get(key)
                if previous is not None and not np.allclose(
                    previous,
                    coord,
                    atol=self.coordinate_tolerance,
                    rtol=0.0,
                ):
                    raise ValueError(
                        f"Conflicting coordinates for patient/channel {key}: "
                        f"{previous} and {coord}."
                    )
                index[key] = coord
        return index

    def _image_path(self, patient: str, channel: str) -> Path:
        return (
            self.residual_folder
            / patient
            / f"{patient}_{channel}_residual.png"
        )

    @staticmethod
    def _channel_from_path(path: Path, patient: str) -> str | None:
        prefix = f"{patient}_"
        suffix = "_residual.png"
        name = path.name
        if not name.startswith(prefix) or not name.endswith(suffix):
            return None
        channel = name[len(prefix) : -len(suffix)]
        return channel or None

    def _build_negative_candidate_index(self) -> Dict[str, List[Tuple[Path, np.ndarray]]]:
        patients = {str(row["source_patient"]) for row in self.rows}
        candidates: Dict[str, List[Tuple[Path, np.ndarray]]] = {}
        for patient in patients:
            patient_dir = self.residual_folder / patient
            if not patient_dir.is_dir():
                raise FileNotFoundError(
                    f"Source-patient folder does not exist: {patient_dir}"
                )

            patient_candidates: List[Tuple[Path, np.ndarray]] = []
            for path in sorted(patient_dir.glob(f"{patient}_*_residual.png")):
                channel = self._channel_from_path(path, patient)
                coord = self.channel_coordinates.get((patient, channel or ""))
                if channel is not None and coord is not None:
                    patient_candidates.append(
                        (path, np.asarray(coord, dtype=np.float32))
                    )
            candidates[patient] = patient_candidates
        return candidates

    def _load_image(self, path: Path) -> torch.Tensor:
        if not path.is_file():
            raise FileNotFoundError(f"Spectrogram image does not exist: {path}")
        with Image.open(path) as image:
            image = image.convert("RGB")
            tensor = self.transform(image)
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(
                "PairedSpectrogramDataset transform must return a torch.Tensor, "
                f"got {type(tensor).__name__}."
            )
        return tensor

    def _sample_negative_paths(
        self,
        source_patient: str,
        ref_coord: Tuple[float, float, float],
    ) -> List[Path]:
        destination = np.asarray(ref_coord, dtype=np.float32)
        candidates = [
            path
            for path, coord in self.negative_candidates[source_patient]
            if not np.allclose(
                coord,
                destination,
                atol=self.coordinate_tolerance,
                rtol=0.0,
            )
        ]
        if not candidates:
            raise ValueError(
                f"No valid negative samples found for patient {source_patient!r}; "
                "the CSV must provide coordinates for at least one channel whose "
                "coordinate differs from the reference location."
            )

        if len(candidates) >= self.k_neg:
            indices = torch.randperm(len(candidates))[: self.k_neg].tolist()
        else:
            indices = torch.randint(
                low=0,
                high=len(candidates),
                size=(self.k_neg,),
            ).tolist()
        return [candidates[index] for index in indices]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        row = self.rows[idx]
        source_patient = str(row["source_patient"])
        source_channel = str(row["source_channel"])
        ref_patient = str(row["ref_patient"])
        ref_channel = str(row["ref_channel"])
        target_true_patient = str(row.get("target_true_patient", ""))
        target_true_channel = str(row.get("target_true_channel", ""))
        source_coord = tuple(float(value) for value in row["source_coord"])
        ref_coord = tuple(float(value) for value in row["ref_coord"])

        source_path = self._image_path(source_patient, source_channel)
        ref_path = self._image_path(ref_patient, ref_channel)
        negative_paths = self._sample_negative_paths(source_patient, ref_coord)

        source = self._load_image(source_path)
        target_ref = self._load_image(ref_path)
        has_target_true = bool(target_true_patient and target_true_channel)
        if has_target_true:
            target_true = self._load_image(self._image_path(target_true_patient, target_true_channel))
        else:
            target_true = torch.zeros_like(source)
        negative = torch.stack(
            [self._load_image(path) for path in negative_paths],
            dim=0,
        )

        return {
            "source": source,
            "target_ref": target_ref,
            "target_true": target_true,
            "has_target_true": torch.tensor(float(has_target_true), dtype=torch.float32),
            "negative": negative,
            "source_coord": torch.tensor(source_coord, dtype=torch.float32),
            "target_coord": torch.tensor(ref_coord, dtype=torch.float32),
            "source_patient": source_patient,
            "source_channel": source_channel,
            "ref_patient": ref_patient,
            "ref_channel": ref_channel,
            "target_true_patient": target_true_patient,
            "target_true_channel": target_true_channel,
        }


def create_paired_spectrogram_split(
    *,
    residual_path: str | Path,
    table_path: str | Path,
    resolution: int,
    batch_size: int,
    split: str,
    k_neg: int = 4,
    use_aug: bool = False,
    use_hflip: bool = False,
    coordinate_tolerance: float = 1e-5,
    num_workers: int = 4,
    prefetch_factor: int = 2,
    pin_memory: bool = False,
):
    """Create a paired-spectrogram DataLoader and preprocessing functions.

    The returned ``preprocess_fn`` keeps all paired fields so future conditional
    training code can consume them. For compatibility with the current
    ImageFolder-oriented training loop, ``images``/``labels`` are also provided:

    - ``images`` is set to ``target_ref`` (``pr_lr``) in BHWC layout.
    - ``labels`` is a zero class label because this paired dataset is not
      ImageNet-class-conditioned.

    Args:
        residual_path: root folder containing ``patient/patient_channel_residual.png``.
        table_path: CSV with paired rows for the requested split.
        resolution: output image resolution for the transform.
        batch_size: per-process DataLoader batch size.
        split: split name, usually ``train`` or ``val``.
        k_neg: number of ``ps_not_lr`` negatives returned per row.
        use_aug: whether to use train-time image augmentation.
        use_hflip: whether to allow horizontal flips in the image transform.
        coordinate_tolerance: tolerance when excluding negatives at ``lr``.
        num_workers: DataLoader workers.
        prefetch_factor: DataLoader prefetch factor when workers are enabled.
        pin_memory: DataLoader pin-memory flag.

    Returns:
        ``(loader, preprocess_fn, postprocess_fn)`` compatible with
        ``build_model_dict``.
    """
    transform = _build_transforms(
        resolution=resolution,
        use_aug=use_aug,
        split=split,
        use_hflip=use_hflip,
    )
    ds = PairedSpectrogramDataset(
        residual_path=residual_path,
        table_path=table_path,
        k_neg=k_neg,
        transform=transform,
        coordinate_tolerance=coordinate_tolerance,
    )
    log_for_0(ds)

    rank = jax.process_index()
    sampler = DistributedSampler(
        ds,
        num_replicas=jax.process_count(),
        rank=rank,
        shuffle=(split == "train"),
    )
    loader_kwargs = {
        "dataset": ds,
        "batch_size": batch_size,
        "drop_last": (split == "train"),
        "worker_init_fn": partial(worker_init_fn, rank=rank),
        "sampler": sampler,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": True if num_workers > 0 else False,
    }
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = prefetch_factor
        loader_kwargs["multiprocessing_context"] = "spawn"
    loader = DataLoader(**loader_kwargs)

    def _to_bhwc(tensor):
        return jnp.array(tensor).transpose(0, 2, 3, 1)

    def _to_bkhwc(tensor):
        return jnp.array(tensor).transpose(0, 1, 3, 4, 2)

    def preprocess_fn(batch, rng=jax.random.PRNGKey(0)):
        del rng
        target_ref = _to_bhwc(batch["target_ref"])
        source = _to_bhwc(batch["source"])
        negative = _to_bkhwc(batch["negative"])
        labels = jnp.zeros((target_ref.shape[0],), dtype=jnp.int32)
        out = {
            "images": target_ref,
            "labels": labels,
            "source": source,
            "target_ref": target_ref,
            "negative": negative,
            "source_coord": jnp.array(batch["source_coord"], dtype=jnp.float32),
            "target_coord": jnp.array(batch["target_coord"], dtype=jnp.float32),
        }
        if "target_true" in batch:
            out["target_true"] = _to_bhwc(batch["target_true"])
        if "has_target_true" in batch:
            out["has_target_true"] = jnp.array(batch["has_target_true"], dtype=jnp.float32)
        return out

    def postprocess_fn(images):
        return jnp.clip((images + 1) / 2, 0, 1).transpose(0, 3, 1, 2)

    return loader, preprocess_fn, postprocess_fn
