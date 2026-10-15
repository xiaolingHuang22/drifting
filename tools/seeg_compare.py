"""Compare raw/EMA and shuffled conditions on audited overfit targets."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import yaml

from dataset.pair_list_spectrogram import latent_cache_path


def load_latent(path):
    value = np.load(path, allow_pickle=False)
    if value.dtype == np.dtype('V2'):
        # NumPy's .npy format loses the name of ml_dtypes.bfloat16 and
        # restores its two-byte payload as void. These generated latent files
        # were written directly from JAX bfloat16 arrays.
        import ml_dtypes
        value = value.view(ml_dtypes.bfloat16)
    return value.astype(np.float32)


def latent_scores(target, mean, median, generated):
    return {name: float(np.abs(value - target).mean()) for name, value in
            [('zero_l1', np.zeros_like(target)), ('train_mean_l1', mean),
             ('train_median_l1', median), ('generated_l1', generated)]}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True)
    p.add_argument('--audit', required=True)
    p.add_argument('--raw', required=True)
    p.add_argument('--ema', required=True)
    p.add_argument('--outdir', required=True)
    p.add_argument('--targets', type=int, default=4)
    p.add_argument('--num-samples', type=int, default=4)
    a = p.parse_args()
    if a.targets < 1 or a.num_samples < 1:
        p.error('targets and num-samples must be positive')
    cfg = yaml.safe_load(Path(a.config).expanduser().read_text())
    fixed_noise = cfg.get('train', {}).get('forward_dict', {}).get('fixed_noise_seed')
    if fixed_noise is not None and a.num_samples != 1:
        p.error('Fixed-noise experiments require --num-samples 1')
    ds = cfg['dataset']
    targets = json.loads(Path(a.audit).expanduser().read_text()).get('overfit_targets', [])
    if not targets:
        raise ValueError('Use an audit generated with the same overfit config used for training.')
    image_root = Path(ds['image_root']).expanduser().resolve()
    def target_latent(t):
        patient, channel = t['target_id'].split(',', 1)
        path = image_root / 'train' / patient / channel / t['target_segment']
        return load_latent(latent_cache_path(Path(ds['latent_root']).expanduser(), path, image_root))
    latents = np.stack([target_latent(t) for t in targets])
    mean, median = latents.mean(0), np.median(latents, axis=0)
    out = Path(a.outdir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / 'train_mean_latent.npy', mean)
    np.save(out / 'train_median_latent.npy', median)
    report = []
    for i, t in enumerate(targets[:a.targets]):
        item = {'target': t, 'variants': {}}
        generated = {}
        for variant, artifact, shuffled in [('raw', a.raw, False), ('ema', a.ema, False), ('raw_shuffled', a.raw, True)]:
            folder = out / f'target_{i:02d}' / variant
            cmd = [sys.executable, 'sample_pair_list.py', '--init-from', artifact,
                   '--config', a.config, '--target-id', t['target_id'],
                   '--target-segment', t['target_segment'], '--split', 'train',
                   '--training-condition-selection', '--seed', str(t['inference_seed']),
                   '--num-samples', str(a.num_samples), '--batch-size', '1', '--outdir', str(folder)]
            if fixed_noise is not None:
                cmd += ['--fixed-noise-seed', str(fixed_noise)]
            if shuffled:
                cmd.append('--shuffle-condition-images')
            subprocess.run(cmd, check=True)
            generated[variant] = np.stack([load_latent(folder / f'latent_{n:06d}.npy') for n in range(a.num_samples)])
            item['variants'][variant] = latent_scores(latents[i], mean, median, generated[variant])
        difference = np.abs(generated['raw'] - generated['raw_shuffled'])
        item['raw_shuffle_latent_mae'] = float(difference.mean())
        item['raw_shuffle_relative_to_target_abs_mean'] = float(difference.mean() / max(float(np.abs(latents[i]).mean()), 1e-12))
        report.append(item)
        (out / 'comparison.json').write_text(json.dumps(report, indent=2))
    print(f'Report: {out / "comparison.json"}')


if __name__ == '__main__':
    main()
