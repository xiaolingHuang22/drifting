"""Evaluate a saved held-out panel with training-reference conditions."""
import argparse
import json
import random
from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter
import yaml

from dataset.pair_list_spectrogram import (
    build_pair_list_records, load_patient_split, donor_patients_for_split,
    eligible_conditions, latent_cache_path,
)
from tools.seeg_compare import load_latent


def image_metrics(reference, prediction):
    # Images in [0,1]. These are display-space metrics, not physical SEEG power.
    x, y = reference.mean(axis=-1), prediction.mean(axis=-1)
    diff = prediction - reference
    mse = float(np.mean(diff ** 2))
    mx, my = gaussian_filter(x, 1.5), gaussian_filter(y, 1.5)
    vx = np.maximum(gaussian_filter(x*x, 1.5) - mx*mx, 0)
    vy = np.maximum(gaussian_filter(y*y, 1.5) - my*my, 0)
    cov = gaussian_filter(x*y, 1.5) - mx*my
    ssim = ((2*mx*my + .01**2)*(2*cov + .03**2) /
            ((mx*mx+my*my+.01**2)*(vx+vy+.03**2)))
    return {'image_mae': float(np.abs(diff).mean()), 'image_rmse': mse**.5,
            'psnr_db': float(-10*np.log10(max(mse, 1e-12))),
            'ssim_gray': float(ssim[5:-5, 5:-5].mean()),
            'row_profile_mae': float(np.abs(x.mean(1)-y.mean(1)).mean()),
            'column_profile_mae': float(np.abs(x.mean(0)-y.mean(0)).mean()),
            'gradient_mae': float(np.mean([np.abs(np.diff(x, axis=a)-np.diff(y, axis=a)).mean() for a in (0, 1)]))}


def average(rows):
    return {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True)
    p.add_argument('--init-from', required=True)
    p.add_argument('--split', choices=['val', 'test'], default='val')
    p.add_argument('--panel', required=True, help='Created once then reused across checkpoints.')
    p.add_argument('--targets-per-patient', type=int, default=4)
    p.add_argument('--num-samples', type=int, default=4)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--outdir', required=True)
    p.add_argument('--shuffle-condition-images', action='store_true')
    a = p.parse_args()
    if a.targets_per_patient < 1 or a.num_samples < 1:
        p.error('Counts must be positive')
    ds = yaml.safe_load(Path(a.config).expanduser().read_text())['dataset']
    root = Path(ds['image_root']).expanduser().resolve()
    manifest = load_patient_split(ds['split_manifest'])
    donors = donor_patients_for_split(manifest, a.split, ds.get('donor_protocol', 'within_split'))
    records = build_pair_list_records(ds['table_path'], root, ds['split_manifest'])
    candidates = {r['target_id']: r for r in records if r['target_patient'] in manifest[a.split]
                  and eligible_conditions(r, donors)}
    panel_path = Path(a.panel).expanduser()
    if panel_path.exists():
        panel = json.loads(panel_path.read_text())
        if panel.get('donor_protocol', ds.get('donor_protocol', 'within_split')) != ds.get('donor_protocol', 'within_split'):
            raise ValueError('Panel donor protocol mismatch')
        if panel['split'] != a.split:
            raise ValueError('Panel split mismatch')
    else:
        chosen = []
        for patient in sorted(manifest[a.split]):
            pool = sorted(k for k,r in candidates.items() if r['target_patient'] == patient)
            rng = random.Random(f'{a.seed}:{patient}')
            chosen.extend(rng.sample(pool, min(len(pool), a.targets_per_patient)))
        panel = {'split': a.split, 'seed': a.seed, 'donor_protocol': ds.get('donor_protocol', 'within_split'), 'target_ids': chosen}
        panel_path.parent.mkdir(parents=True, exist_ok=True)
        panel_path.write_text(json.dumps(panel, indent=2))
    if not panel['target_ids']:
        raise ValueError('No eligible targets; inspect donor coverage audit')
    out = Path(a.outdir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    report = {'split': a.split, 'donor_protocol': ds.get('donor_protocol', 'within_split'),
              'checkpoint': a.init_from, 'shuffled': a.shuffle_condition_images,
              'patients_without_eligible_targets': [pat for pat in manifest[a.split] if not any(r['target_patient']==pat for r in candidates.values())],
              'targets': []}
    def pixels(path):
        return np.asarray(Image.open(path).convert('RGB').resize((int(ds['resolution']), int(ds['resolution']))), dtype=np.float32)/255.
    for i, key in enumerate(panel['target_ids']):
        if key not in candidates:
            raise ValueError(f'Panel target no longer eligible: {key}')
        r = candidates[key]
        folder = out / f'target_{i:03d}'
        cmd = [sys.executable, 'sample_pair_list.py', '--config', a.config,
               '--init-from', a.init_from, '--target-id', r.get('channel_id', key),
               '--target-segment', Path(r['target_path']).name, '--split', a.split,
               '--seed', str(panel['seed']+i), '--num-samples', str(a.num_samples),
               '--batch-size', '1', '--outdir', str(folder)]
        if a.shuffle_condition_images:
            cmd.append('--shuffle-condition-images')
        subprocess.run(cmd, check=True)
        target = pixels(r['target_path'])
        reconstruction = pixels(folder/'target_vae_reconstruction.png')
        target_latent = load_latent(latent_cache_path(Path(ds['latent_root']).expanduser(), r['target_path'], root))
        sample_metrics, generated = [], []
        for j in range(a.num_samples):
            scores = image_metrics(target, pixels(folder/f'sample_{j:06d}.png'))
            scores['reconstruction_image_mae'] = float(np.abs(pixels(folder/f'sample_{j:06d}.png')-reconstruction).mean())
            latent = load_latent(folder/f'latent_{j:06d}.npy')
            scores['latent_l1'] = float(np.abs(latent-target_latent).mean())
            generated.append(latent)
            sample_metrics.append(scores)
        diversity = [float(np.abs(generated[j]-generated[k]).mean()) for j in range(len(generated)) for k in range(j)]
        row = {'target_id': key, 'patient': r['target_patient'], 'region': r['region'],
               'eligible_donor_channels': len(eligible_conditions(r, donors)),
               'samples': sample_metrics, 'mean': average(sample_metrics),
               'vae_reference': image_metrics(target, reconstruction),
               'latent_sample_diversity_l1': float(np.mean(diversity)) if diversity else None}
        report['targets'].append(row)
        report['target_macro'] = average([x['mean'] for x in report['targets']])
        patients = sorted({x['patient'] for x in report['targets']})
        regions = sorted({x['region'] for x in report['targets']})
        report['region_macro'] = average([average([x['mean'] for x in report['targets'] if x['region']==reg]) for reg in regions])
        report['patient_macro'] = average([average([x['mean'] for x in report['targets'] if x['patient']==pat]) for pat in patients])
        (out/'evaluation.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report['patient_macro'], indent=2))


if __name__ == '__main__':
    main()
