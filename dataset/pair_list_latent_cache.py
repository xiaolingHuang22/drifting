"""Precompute Stable-Diffusion VAE latents for pair-list spectrogram PNGs."""

from __future__ import annotations

import argparse
from pathlib import Path

import jax
import numpy as np
from PIL import Image
from torchvision import transforms

from dataset.pair_list_spectrogram import build_pair_list_records, latent_cache_path
from dataset.vae import vae_enc_decode
from utils.hsdp_util import set_global_mesh
from utils.misc import run_init


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table-path", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--target-root", required=True)
    parser.add_argument("--split-manifest", default=None)
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    run_init()
    set_global_mesh(1)
    image_root = Path(args.image_root).expanduser().resolve()
    target_root = Path(args.target_root).expanduser().resolve()
    records = build_pair_list_records(args.table_path, image_root, args.split_manifest)
    paths = {Path(record["target_path"]) for record in records}
    paths.update(Path(path) for record in records for item in record["conditions"] for path in item.get("paths", [item["path"]]))
    transform = transforms.Compose([
        transforms.Resize((args.resolution, args.resolution), antialias=True),
        transforms.ToTensor(),
        transforms.Normalize([.5] * 3, [.5] * 3),
    ])
    encode_fn, _ = vae_enc_decode()
    written = skipped = 0
    for index, image_path in enumerate(sorted(paths)):
        output = latent_cache_path(target_root, image_path, image_root)
        if output.exists() and not args.overwrite:
            skipped += 1
            continue
        output.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(image_path) as image:
            pixels = np.asarray(transform(image.convert("RGB")), dtype=np.float32)[None]
        # A batch of one deliberately avoids the huge B*K online-VAE allocation
        # that can coexist poorly with the generator and frozen MAE on one GPU.
        latent = encode_fn(pixels, jax.random.PRNGKey(index))
        latent = np.asarray(jax.device_get(latent), dtype=np.float32)
        if latent.ndim != 4:
            raise ValueError(f"Unexpected VAE latent shape {latent.shape} for {image_path}")
        if latent.shape[1] == 4:
            latent = latent.transpose(0, 2, 3, 1)
        np.save(output, latent[0], allow_pickle=False)
        written += 1
        if written % 100 == 0:
            print(f"cached={written} skipped={skipped} total={len(paths)}", flush=True)
    print(f"Latent cache complete: cached={written}, skipped={skipped}, root={target_root}")


if __name__ == "__main__":
    main()
