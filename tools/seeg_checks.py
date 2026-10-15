"""Audit donor coverage, prepare a fixed-data experiment, or export checkpoint weights."""
import argparse
import json
from pathlib import Path

import yaml


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    audit = sub.add_parser('audit')
    audit.add_argument('--config', required=True)
    audit.add_argument('--output', required=True)
    config = sub.add_parser('overfit-config')
    config.add_argument('--config', required=True)
    config.add_argument('--output', required=True)
    config.add_argument('--targets', type=int, default=32)
    config.add_argument('--steps', type=int, default=5000)
    config.add_argument('--pair-only', action='store_true')
    config.add_argument('--fixed-noise-seed', type=int)
    export = sub.add_parser('export')
    export.add_argument('--checkpoint-dir', required=True)
    export.add_argument('--step', type=int, required=True)
    export.add_argument('--weights', choices=['raw', 'ema'], required=True)
    export.add_argument('--output', required=True)
    args = parser.parse_args()
    output = Path(args.output).expanduser()
    if args.command == 'export':
        from flax import serialization
        from flax.training import checkpoints
        restored = checkpoints.restore_checkpoint(str(Path(args.checkpoint_dir).expanduser().resolve()), target=None, step=args.step)
        key = 'params' if args.weights == 'raw' else 'ema_params'
        if not isinstance(restored, dict) or key not in restored:
            raise ValueError(f'Checkpoint {args.step} does not contain {key}.')
        output.mkdir(parents=True, exist_ok=True)
        # Artifact loader uses this filename for both variants; metadata identifies weights.
        (output / 'ema_params.msgpack').write_bytes(serialization.msgpack_serialize(restored[key]))
        (output / 'metadata.json').write_text(json.dumps({'step': args.step, 'weights': args.weights, 'kind': 'gen'}, indent=2))
        return
    cfg = yaml.safe_load(Path(args.config).expanduser().read_text())
    output.parent.mkdir(parents=True, exist_ok=True)
    if args.command == 'overfit-config':
        if args.targets <= 0 or args.steps <= 0:
            raise ValueError('Targets and steps must be positive.')
        cfg['dataset'].update(train_max_targets=args.targets, fixed_conditions=True)
        cfg['train'].update(total_steps=args.steps, init_from='', val_loss_batches=32)
        if args.pair_only:
            cfg['train'].update(lambda_drift=0.0, lambda_pair=1.0,
                                lambda_condition=0.0, lambda_tv=0.0,
                                log_update_diagnostics=True)
        if args.fixed_noise_seed is not None:
            cfg['train'].setdefault('forward_dict', {}).update(
                fixed_noise_seed=args.fixed_noise_seed, gen_per_label=1)
            cfg['dataset'].update(batch_size=1, eval_batch_size=1)
            cfg['train']['train_batch_size'] = 1
        cfg['optimizer']['lr_schedule']['total_steps'] = args.steps
        output.write_text(yaml.safe_dump(cfg, sort_keys=False))
        return
    from dataset.pair_list_spectrogram import build_pair_list_records, load_patient_split
    ds = cfg['dataset']
    root = Path(ds['image_root']).expanduser()
    segmented = any((root / s).is_dir() for s in ('train', 'val', 'test'))
    records = build_pair_list_records(ds['table_path'], root, ds['split_manifest'] if segmented else None)
    manifest = load_patient_split(ds['split_manifest'])
    result = {}
    for split in ('train', 'val', 'test'):
        allowed = set(manifest[split])
        targets = {}
        for r in records:
            if r['target_patient'] not in allowed:
                continue
            key = r.get('channel_id', r['target_id'])
            if key not in targets:
                donors = [c for c in r['conditions'] if c['patient'] in allowed]
                targets[key] = {'target_segments': 0, 'eligible_channels': len(donors),
                                'donor_patients': sorted({c['patient'] for c in donors}),
                                'selected_slots': min(len(donors), ds.get('max_conditions', 8))}
            targets[key]['target_segments'] += 1
        histogram = {}
        for t in targets.values():
            n = str(t['eligible_channels'])
            histogram[n] = histogram.get(n, 0) + 1
        result[split] = {'patients': len(allowed), 'target_channels': len(targets),
                         'retained_segments': sum(t['target_segments'] for t in targets.values() if t['eligible_channels']),
                         'eligible_channel_histogram': histogram, 'targets': targets}
    if ds.get('train_max_targets'):
        from dataset.pair_list_spectrogram import PairListSpectrogramDataset, select_conditions
        import random
        subset = PairListSpectrogramDataset(records, manifest['train'],
                    max_targets=ds['train_max_targets'], seed=ds.get('seed', 42),
                    fixed_conditions=True)
        result['overfit_targets'] = [
            {'target_id': r.get('channel_id', r['target_id']),
             'target_segment': Path(r['target_path']).name,
             'conditions': [c['path'] for c in select_conditions(r, ds.get('max_conditions', 8),
                            random.Random(ds.get('seed', 42) + i), training=True)],
             'inference_seed': ds.get('seed', 42) + i}
            for i, r in enumerate(subset.records)]
    output.write_text(json.dumps(result, indent=2))
    print(json.dumps({s: {k:v for k,v in summary.items() if k != 'targets'} for s, summary in result.items() if s in ('train', 'val', 'test')}, indent=2))


if __name__ == '__main__':
    main()
