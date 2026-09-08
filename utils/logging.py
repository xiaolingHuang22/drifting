from __future__ import annotations

import json
import os
import hashlib
import math
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import jax
import numpy as np
from absl import logging as absl_logging
from PIL import Image, ImageDraw


def is_rank_zero() -> bool:
    return jax.process_index() == 0


def log_for_0(msg, *args, **kwargs):
    if is_rank_zero():
        absl_logging.info(msg, *args, **kwargs)


def log_for_all(msg):
    absl_logging.info("[Rank %s] %s", jax.process_index(), msg)


class WandbLogger:
    def __init__(self) -> None:
        self.step = 0
        self.use_wandb = True
        self.log_every_k = 1
        self.console_log = True
        self.console_preview_keys = None
        self._buffer: Dict[str, float] = {}
        self._count: Dict[str, int] = {}
        self.offline_dir = Path("log")
        self._wandb = None

    def set_logging(
        self,
        project: Optional[str] = None,
        config: Optional[Any] = None,
        entity: Optional[str] = None,
        name: Optional[str] = None,
        use_wandb: bool = True,
        offline_dir: str = "log",
        workdir: Optional[str] = None,
        log_every_k: int = 1,
        console_log: bool = True,
        console_preview_keys: Optional[Sequence[str]] = None,
        allow_resume: bool = True,
        **kwargs,
    ) -> None:
        self.use_wandb = bool(use_wandb)
        self.log_every_k = int(log_every_k)
        self.console_log = bool(console_log)
        self.console_preview_keys = list(console_preview_keys) if console_preview_keys is not None else None
        workdir_path = Path(workdir).resolve() if workdir else None
        resolved_offline_dir = workdir_path / "log" if (workdir_path is not None and not self.use_wandb) else Path(offline_dir)
        self.offline_dir = resolved_offline_dir
        self.offline_dir.mkdir(parents=True, exist_ok=True)

        if not is_rank_zero():
            return

        if self.use_wandb:
            import wandb
            self._wandb = wandb
            default_run_id = ""
            if workdir_path is not None:
                default_run_id = hashlib.sha1(str(workdir_path).encode("utf-8")).hexdigest()[:16]
            run_id = kwargs.pop("run_id", None) or default_run_id
            init_kwargs = dict(project=project, entity=entity, name=name, config=config, mode="online", reinit=True)
            if allow_resume:
                init_kwargs["resume"] = "allow"
                if run_id:
                    init_kwargs["id"] = run_id
            init_kwargs.update(kwargs)
            wandb.init(**init_kwargs)

    def set_step(self, step: int) -> None:
        self.step = int(step)

    def _flush_buffer(self) -> None:
        if not self._buffer:
            return
        reduced = {k: (self._buffer[k] / max(1, self._count.get(k, 1))) for k in self._buffer.keys()}
        if self.console_log and is_rank_zero():
            default_preview_keys = (
                "loss",
                "val/loss",
                "lr",
                "g_norm",
                "best_fid",
                "best_cfg",
            )
            configured_keys = self.console_preview_keys or default_preview_keys
            preview_keys = [k for k in configured_keys if k in reduced]
            if preview_keys:
                preview = " ".join(f"{k}={reduced[k]:.6g}" for k in preview_keys)
                # Use plain stdout to ensure visibility regardless of absl log level.
                print(f"[train] step={self.step} {preview}", flush=True)
        if self._wandb is not None:
            self._wandb.log(reduced, step=self.step)
        else:
            p = self.offline_dir / "metrics.jsonl"
            with p.open("a", encoding="utf-8") as f:
                f.write(json.dumps({"step": self.step, **reduced}, ensure_ascii=False) + "\n")
        self._buffer.clear()
        self._count.clear()

    def log_dict(self, d: Dict[str, Any]) -> None:
        if not is_rank_zero():
            return
        reduced = {}
        for k, v in d.items():
            if isinstance(v, (jax.Array, np.ndarray)):
                v = float(np.asarray(v).mean())
            if isinstance(v, (int, float, np.floating, np.integer)):
                reduced[k] = float(v)
        for k, v in reduced.items():
            self._buffer[k] = self._buffer.get(k, 0.0) + float(v)
            self._count[k] = self._count.get(k, 0) + 1
        if self.log_every_k <= 1 or (self.step % self.log_every_k == 0):
            self._flush_buffer()

    def log_dict_dir(self, prefix: str, d: Dict[str, Any]) -> None:
        """Log a dict with keys namespaced by prefix."""
        self.log_dict({f"{prefix}/{k}": v for k, v in d.items()})

    @staticmethod
    def _normalize_images(images) -> np.ndarray:
        arr = np.asarray(images)
        if arr.ndim == 3:
            arr = arr[None, ...]
        if arr.ndim != 4:
            raise ValueError(f"Expected image batch with 3 or 4 dims, got shape {arr.shape}")
        if arr.shape[1] in (1, 3) and arr.shape[-1] not in (1, 3):
            arr = np.transpose(arr, (0, 2, 3, 1))
        if arr.shape[-1] == 1:
            arr = np.repeat(arr, 3, axis=-1)
        if arr.shape[-1] != 3:
            raise ValueError(f"Expected channel-last image batch with 3 channels, got shape {arr.shape}")
        if arr.dtype != np.uint8:
            # Logger inputs usually come in as float images in [0, 1]. Wandb and
            # PIL are more predictable with uint8 RGB, so normalize once here and
            # keep the downstream logging paths dtype-stable.
            arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0)
            arr = np.clip(arr, 0.0, 1.0)
            arr = (arr * 255.0).astype(np.uint8)
        return arr

    @staticmethod
    def _make_grid_image(images: np.ndarray, rows: int = 8) -> Image.Image:
        rows = max(1, int(rows))
        pil_imgs = [Image.fromarray(img) for img in images]
        cols = max(1, int(math.ceil(len(pil_imgs) / rows)))
        w, h = pil_imgs[0].size
        total = rows * cols
        if len(pil_imgs) < total:
            blank = Image.new("RGB", (w, h), color=(0, 0, 0))
            pil_imgs += [blank] * (total - len(pil_imgs))
        grid = Image.new("RGB", (cols * w, rows * h))
        for idx, img in enumerate(pil_imgs):
            row = idx % rows
            col = idx // rows
            grid.paste(img, (col * w, row * h))
        return grid

    def log_image(self, name: str, images) -> None:
        if not is_rank_zero():
            return
        arr = self._normalize_images(images)
        grid_img = self._make_grid_image(arr)
        if self._wandb is not None:
            self._wandb.log({name: [self._wandb.Image(img) for img in arr]}, step=self.step)
            self._wandb.log({f"{name}_grid": self._wandb.Image(grid_img)}, step=self.step)
            return
        out = self.offline_dir / "images"
        out.mkdir(parents=True, exist_ok=True)
        grid_img.save(out / f"{name.replace('/', '_')}_step{self.step}.jpg", format="JPEG")

    def finish(self) -> None:
        self._flush_buffer()
        if self._wandb is not None and is_rank_zero():
            self._wandb.finish()

    def save_loss_plot(self, filename: str = "loss_curve.png") -> Optional[Path]:
        """Plot logged train/validation losses from the offline JSONL history."""
        if not is_rank_zero():
            return None
        metrics_path = self.offline_dir / "metrics.jsonl"
        if not metrics_path.is_file():
            log_for_0("Cannot plot losses; metrics file does not exist: %s", metrics_path)
            return None

        objective_train_points: list[tuple[int, float]] = []
        objective_val_points: list[tuple[int, float]] = []
        semantic_train_points: list[tuple[int, float]] = []
        semantic_val_points: list[tuple[int, float]] = []
        with metrics_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                try:
                    record = json.loads(line)
                    step = int(record["step"])
                except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                    continue
                for key, points in (
                    ("loss", objective_train_points),
                    ("val/loss", objective_val_points),
                    ("semantic_loss", semantic_train_points),
                    ("val/semantic_loss", semantic_val_points),
                ):
                    value = record.get(key)
                    if isinstance(value, (int, float)) and math.isfinite(value):
                        points.append((step, float(value)))

        if semantic_train_points:
            train_points = semantic_train_points
            val_points = semantic_val_points
            train_label = "train/semantic_loss"
            val_label = "val/semantic_loss"
            title = "Training and validation semantic monitor loss"
        else:
            train_points = objective_train_points
            val_points = objective_val_points
            train_label = "train/loss"
            val_label = "val/loss"
            title = "Training and validation loss"

        all_points = train_points + val_points
        if not all_points:
            log_for_0("Cannot plot losses; no finite loss values found in %s", metrics_path)
            return None

        width, height = 1000, 600
        left, right, top, bottom = 90, 30, 45, 70
        image = Image.new("RGB", (width, height), "white")
        draw = ImageDraw.Draw(image)
        x_min = min(step for step, _ in all_points)
        x_max = max(step for step, _ in all_points)
        y_min = min(loss for _, loss in all_points)
        y_max = max(loss for _, loss in all_points)
        if x_max == x_min:
            x_max = x_min + 1
        if y_max == y_min:
            margin = max(abs(y_min) * 0.05, 1e-6)
            y_min -= margin
            y_max += margin

        plot_width = width - left - right
        plot_height = height - top - bottom

        def xy(point: tuple[int, float]) -> tuple[float, float]:
            step, loss = point
            x = left + (step - x_min) / (x_max - x_min) * plot_width
            y = top + (y_max - loss) / (y_max - y_min) * plot_height
            return x, y

        draw.line((left, top, left, top + plot_height), fill="black", width=2)
        draw.line((left, top + plot_height, left + plot_width, top + plot_height), fill="black", width=2)
        for index in range(6):
            fraction = index / 5
            y = top + fraction * plot_height
            value = y_max - fraction * (y_max - y_min)
            draw.line((left, y, left + plot_width, y), fill=(225, 225, 225), width=1)
            draw.text((5, y - 7), f"{value:.6g}", fill="black")
            x = left + fraction * plot_width
            step = round(x_min + fraction * (x_max - x_min))
            draw.text((x - 15, top + plot_height + 10), str(step), fill="black")

        if len(train_points) >= 2:
            draw.line([xy(point) for point in train_points], fill=(31, 119, 180), width=3)
        elif train_points:
            x, y = xy(train_points[0])
            draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=(31, 119, 180))
        if len(val_points) >= 2:
            draw.line([xy(point) for point in val_points], fill=(214, 39, 40), width=3)
        elif val_points:
            x, y = xy(val_points[0])
            draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=(214, 39, 40))

        draw.text((left, 15), title, fill="black")
        draw.text((left + plot_width / 2 - 20, height - 25), "step", fill="black")
        draw.text((left + 15, 28), train_label, fill=(31, 119, 180))
        draw.text((left + 180, 28), val_label, fill=(214, 39, 40))
        output_path = self.offline_dir / filename
        image.save(output_path)
        log_for_0("Saved loss plot to %s", output_path)
        return output_path


class NullLogger:
    @staticmethod
    def log_dict(*args, **kwargs):
        return None

    @staticmethod
    def log_image(*args, **kwargs):
        return None

    @staticmethod
    def finish(*args, **kwargs):
        return None

    @staticmethod
    def save_loss_plot(*args, **kwargs):
        return None
