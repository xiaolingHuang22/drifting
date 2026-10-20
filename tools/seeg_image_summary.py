"""Summarize saved raw/EMA/shuffled comparison PNGs without GPU inference."""
import argparse
import csv
import json
from pathlib import Path
import re

import numpy as np
from PIL import Image

from utils.spectrogram_metrics import image_metrics

VARIANTS = ('raw', 'ema', 'raw_shuffled')
REFERENCES = {'original': 'target_original.png', 'vae_reconstruction': 'target_vae_reconstruction.png'}


def pixels(path):
    with Image.open(path) as image:
        return np.asarray(image.convert('RGB'), dtype=np.float64) / 255.


def quality(reference, prediction):
    metrics = image_metrics(reference, prediction)
    return {'mse': metrics['image_mse'], 'rmse': metrics['image_rmse'],
            'ssim_gray': metrics['ssim_gray'], 'psnr_db': metrics['psnr_db']}


def summarize_comparison(directory, num_samples=None, target_dirs=None):
    directory = Path(directory).expanduser().resolve()
    manifest_path = directory/'comparison_manifest.json'
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest.get('status') != 'complete':
            raise ValueError('Comparison manifest is incomplete; finish inference before summarizing')
        if target_dirs is None:
            target_dirs = manifest['target_directories']
        if num_samples is None:
            num_samples = manifest['num_samples']
    if num_samples is not None and num_samples < 1:
        raise ValueError('num_samples must be positive')
    targets = ([directory/name for name in target_dirs] if target_dirs is not None
               else sorted(p for p in directory.glob('target_*') if p.is_dir()))
    if not targets:
        raise ValueError(f'No target_* directories found under {directory}')
    report = {'comparison_directory': str(directory), 'target_count': len(targets),
              'pixel_range': [0,1], 'aggregation': 'Arithmetic mean of per-image metrics; every generated image has equal weight. No best-of-N selection.',
              'ssim': 'Grayscale channel mean, Gaussian sigma 1.5, population covariance, 5-pixel border crop; data range 1.',
              'psnr': 'Per-image PSNR in dB, perfect matches capped at 120 dB.',
              'sample_selection': 'Manifest or explicit count' if num_samples is not None else 'All sample_XXXXXX.png files present (legacy directory without manifest).',
              'variants': {}}
    for variant in VARIANTS:
        rows = {name:[] for name in REFERENCES}
        target_scores = {name:[] for name in REFERENCES}
        pixels_total = 0
        squared_error = {name:0. for name in REFERENCES}
        counts = {}
        for target in targets:
            folder = target/variant
            samples = ([folder/f'sample_{i:06d}.png' for i in range(num_samples)] if num_samples is not None
                       else sorted(p for p in folder.glob('sample_*.png') if re.fullmatch(r'sample_\d{6}\.png',p.name)))
            if not samples or any(not p.is_file() for p in samples):
                raise ValueError(f'Missing expected generated PNGs in {folder}')
            reference = {name:pixels(folder/filename) for name,filename in REFERENCES.items()}
            per_target = {name:[] for name in REFERENCES}
            for sample in samples:
                generated = pixels(sample)
                pixels_total += generated.size
                for name,image in reference.items():
                    result = quality(image,generated)
                    rows[name].append(result)
                    per_target[name].append(result)
                    squared_error[name] += result['mse'] * generated.size
            counts[target.name] = len(samples)
            for name in REFERENCES:
                target_scores[name].append({k:float(np.mean([r[k] for r in per_target[name]])) for k in per_target[name][0]})
        result = {'image_count': sum(counts.values()), 'target_count':len(counts), 'samples_per_target':counts, 'references':{}}
        for name in REFERENCES:
            pooled_mse = squared_error[name]/pixels_total
            result['references'][name] = {
                'mean_per_image': {k:float(np.mean([r[k] for r in rows[name]])) for k in rows[name][0]},
                'std_per_image': {k:float(np.std([r[k] for r in rows[name]])) for k in rows[name][0]},
                'target_macro': {k:float(np.mean([r[k] for r in target_scores[name]])) for k in target_scores[name][0]},
                'pooled_mse': pooled_mse, 'pooled_rmse':float(np.sqrt(pooled_mse)),
                'pooled_psnr_db':float(-10*np.log10(max(pooled_mse,1e-12)))}
        report['variants'][variant] = result
    return report


def write_summary(directory, output=None, num_samples=None, target_dirs=None):
    report = summarize_comparison(directory,num_samples,target_dirs)
    path = Path(output).expanduser() if output else Path(directory).expanduser()/'image_metrics_summary.json'
    if path.suffix.lower() != '.json':
        raise ValueError('Summary output must end in .json')
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    csv_path = path.with_suffix('.csv')
    fields = ['variant','reference','image_count','target_count','mean_mse','mean_rmse','mean_ssim_gray','mean_psnr_db','pooled_mse','pooled_rmse','pooled_psnr_db']
    with csv_path.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields)
        writer.writeheader()
        for variant,data in report['variants'].items():
            for reference,stats in data['references'].items():
                writer.writerow(dict(variant=variant,reference=reference,image_count=data['image_count'],target_count=data['target_count'],
                    **{'mean_'+k:v for k,v in stats['mean_per_image'].items()},
                    **{k:stats[k] for k in ['pooled_mse','pooled_rmse','pooled_psnr_db']}))
    print(f'Image metrics: {path}\nCSV: {csv_path}')
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--comparison-dir',required=True)
    p.add_argument('--output',help='Optional .json path; a matching CSV is also written.')
    p.add_argument('--num-samples',type=int,help='For legacy outputs, score only samples 0..N-1 and ignore stale higher-numbered PNGs.')
    a=p.parse_args()
    write_summary(a.comparison_dir,a.output,a.num_samples)


if __name__=='__main__':
    main()
