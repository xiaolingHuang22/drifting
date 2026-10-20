"""Audit donor coverage, prepare a fixed-data experiment, or export checkpoint weights."""
import argparse
import copy
import math
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
    noise = config.add_mutually_exclusive_group()
    noise.add_argument('--fixed-noise-seed', type=int)
    noise.add_argument('--varying-noise', action='store_true',
                       help='Remove an inherited fixed noise seed explicitly.')
    matched = sub.add_parser('matched-configs')
    matched.add_argument('--config', required=True)
    matched.add_argument('--output', required=True, help='Directory for paired-only and mixed configs.')
    matched.add_argument('--steps', type=int, default=40000)
    matched.add_argument('--drift-weight', type=float, default=0.1)
    general = sub.add_parser('generalization-config')
    general.add_argument('--config', required=True)
    general.add_argument('--output', required=True)
    general.add_argument('--steps', type=int, default=100000)
    general.add_argument('--drift-weight', type=float, default=0.1)
    extend = sub.add_parser('extend-config')
    extend.add_argument('--config', required=True)
    extend.add_argument('--output', required=True)
    extend.add_argument('--steps', type=int, default=40000)
    extend.add_argument('--resume-min-step', type=int, default=5000)
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
    if args.command == 'generalization-config':
        if args.steps <= 0 or not math.isfinite(args.drift_weight) or args.drift_weight <= 0:
            raise ValueError('Steps and drift weight must be positive and finite.')
        cfg['dataset'].update(donor_protocol='train_reference', train_max_targets=None,
                              fixed_conditions=False)
        cfg['train'].update(total_steps=args.steps, init_from='', resume_min_step=0,
                            lambda_drift=args.drift_weight, lambda_pair=1.,
                            lambda_condition=0., lambda_tv=0., val_loss_batches=64,
                            save_per_step=5000, keep_every=5000, keep_last=8,
                            log_update_diagnostics=True)
        cfg['train'].setdefault('forward_dict', {}).pop('fixed_noise_seed', None)
        cfg['optimizer']['lr_schedule']['total_steps'] = args.steps
        output.write_text(yaml.safe_dump(cfg, sort_keys=False))
        return
    if args.command == 'matched-configs':
        if args.steps <= 0 or not math.isfinite(args.drift_weight) or args.drift_weight <= 0:
            raise ValueError('Steps and drift weight must be positive and finite.')
        cfg['dataset'].update(train_max_targets=32, fixed_conditions=True,
                              batch_size=1, eval_batch_size=1)
        cfg['train'].update(total_steps=args.steps, init_from='', resume_min_step=0,
                            train_batch_size=1, val_loss_batches=32,
                            lambda_pair=1.0, lambda_condition=0.0, lambda_tv=0.0,
                            log_update_diagnostics=True, save_per_step=5000,
                            keep_every=5000, keep_last=8)
        cfg['train'].setdefault('forward_dict', {}).pop('fixed_noise_seed', None)
        cfg['train']['forward_dict']['gen_per_label'] = 1
        cfg['optimizer']['lr_schedule']['total_steps'] = args.steps
        output.mkdir(parents=True, exist_ok=True)
        variants = [('varying32_pair', 0.0), ('varying32_drift', args.drift_weight)]
        for name, _ in variants:
            path = output / f'seeg_{name}_40k_local.yaml'
            if path.exists():
                raise FileExistsError(f'Refusing to overwrite experiment config: {path}')
        for name, weight in variants:
            variant = copy.deepcopy(cfg)
            variant['train']['lambda_drift'] = weight
            path = output / f'seeg_{name}_40k_local.yaml'
            if path.exists():
                raise FileExistsError(f'Refusing to overwrite experiment config: {path}')
            path.write_text(yaml.safe_dump(variant, sort_keys=False))
            print(path)
        return
    if args.command == 'extend-config':
        if args.resume_min_step <= 0 or args.steps <= args.resume_min_step:
            raise ValueError('Extension total steps must exceed a positive resume minimum.')
        if cfg['optimizer']['lr_schedule'].get('lr_schedule', 'const') != 'const':
            raise ValueError('This controlled extension requires the existing constant LR schedule.')
        cfg['train'].update(total_steps=args.steps, resume_min_step=args.resume_min_step,
                            save_per_step=5000, keep_every=5000, keep_last=8)
        cfg['optimizer']['lr_schedule']['total_steps'] = args.steps
        output.write_text(yaml.safe_dump(cfg, sort_keys=False))
        return
    if args.command == 'overfit-config':
        if args.targets <= 0 or args.steps <= 0:
            raise ValueError('Targets and steps must be positive.')
        cfg['dataset'].update(train_max_targets=args.targets, fixed_conditions=True)
        cfg['train'].update(total_steps=args.steps, init_from='', resume_min_step=0, val_loss_batches=32)
        if args.pair_only:
            cfg['train'].update(lambda_drift=0.0, lambda_pair=1.0,
                                lambda_condition=0.0, lambda_tv=0.0,
                                log_update_diagnostics=True)
        if args.varying_noise:
            cfg['train'].setdefault('forward_dict', {}).pop('fixed_noise_seed', None)
        if args.fixed_noise_seed is not None:
            cfg['train'].setdefault('forward_dict', {}).update(
                fixed_noise_seed=args.fixed_noise_seed, gen_per_label=1)
            cfg['dataset'].update(batch_size=1, eval_batch_size=1)
            cfg['train']['train_batch_size'] = 1
        cfg['optimizer']['lr_schedule']['total_steps'] = args.steps
        output.write_text(yaml.safe_dump(cfg, sort_keys=False))
        return
    from dataset.pair_list_spectrogram import build_pair_list_records, load_patient_split, donor_patients_for_split, eligible_conditions, select_positive_paths, select_target_records
    ds = cfg['dataset']
    root = Path(ds['image_root']).expanduser()
    segmented = any((root / s).is_dir() for s in ('train', 'val', 'test'))
    records = build_pair_list_records(ds['table_path'], root, ds['split_manifest'] if segmented else None)
    manifest = load_patient_split(ds['split_manifest'])
    result = {}
    for split in ('train', 'val', 'test'):
        allowed = set(manifest[split])
        donor_allowed = donor_patients_for_split(manifest, split, ds.get('donor_protocol', 'within_split'))
        targets = {}
        for r in records:
            if r['target_patient'] not in allowed:
                continue
            key = r.get('channel_id', r['target_id'])
            if key not in targets:
                donors = eligible_conditions(r, donor_allowed)
                targets[key] = {'target_segments': 0, 'eligible_channels': len(donors),
                                'donor_patients': sorted({c['patient'] for c in donors}),
                                'selected_slots': min(len(donors), ds.get('max_conditions', 8))}
            targets[key]['target_segments'] += 1
        histogram = {}
        for t in targets.values():
            n = str(t['eligible_channels'])
            histogram[n] = histogram.get(n, 0) + 1
        result[split] = {'donor_protocol': ds.get('donor_protocol', 'within_split'), 'patients': len(allowed), 'target_channels': len(targets),
                         'retained_segments': sum(t['target_segments'] for t in targets.values() if t['eligible_channels']),
                         'eligible_channel_histogram': histogram, 'targets': targets}
    import random
    training_donors = donor_patients_for_split(manifest, 'train', ds.get('donor_protocol', 'within_split'))
    anchors = select_target_records([r for r in records if r['target_patient'] in manifest['train']
                                    and eligible_conditions(r, training_donors)],
                                   ds.get('train_max_targets'), ds.get('seed',42),
                                   ds.get('train_patient_balanced',False))
    result['field_sampling'] = {
        'generated_per_condition': cfg.get('train',{}).get('forward_dict',{}).get('gen_per_label',1),
        'k_pos': ds.get('k_pos',1), 'val_k_pos': ds.get('val_k_pos',ds.get('k_pos',1)),
        'positive_sampling': ds.get('positive_sampling','fixed'),
        'real_negative_weight': cfg.get('train',{}).get('loss_kwargs',{}).get('real_negative_weight',1.),
        'training_anchor_images': len(anchors),
        'available_training_positive_images': len({p for r in anchors for p in r.get('target_paths',[r['target_path']])}),
        'anchor_channels_with_one_segment': len({r.get('channel_id',r['target_id']) for r in anchors if len(r.get('target_paths',[r['target_path']])) == 1}),
    }
    if ds.get('train_max_targets'):
        from dataset.pair_list_spectrogram import PairListSpectrogramDataset, select_conditions
        import random
        subset = PairListSpectrogramDataset(records, manifest['train'],
                    max_targets=ds['train_max_targets'], seed=ds.get('seed', 42),
                    fixed_conditions=True,
                    patient_balanced=bool(ds.get('train_patient_balanced', False)),
                    donor_patients=donor_patients_for_split(manifest, 'train', ds.get('donor_protocol', 'within_split')))
        result['overfit_targets'] = [
            {'target_id': r.get('channel_id', r['target_id']),
             'target_segment': Path(r['target_path']).name,
             'conditions': [c['path'] for c in select_conditions(r, ds.get('max_conditions', 8),
                            random.Random(ds.get('seed', 42) + i), training=True)],
             'inference_seed': ds.get('seed', 42) + i,
             'positive_paths': select_positive_paths(r, ds.get('k_pos',1), random.Random(f"{ds.get('seed',42)}:{r['target_id']}:positive"))[0],
             'positive_mask': select_positive_paths(r, ds.get('k_pos',1), random.Random(f"{ds.get('seed',42)}:{r['target_id']}:positive"))[1],
             'condition_sampling': ds.get('condition_sampling', 'legacy'),
             'conditions_are_view_zero': True}
            for i, r in enumerate(subset.records)]
    output.write_text(json.dumps(result, indent=2))
    print(json.dumps({s: {k:v for k,v in summary.items() if k != 'targets'} for s, summary in result.items() if s in ('train', 'val', 'test')}, indent=2))


if __name__ == '__main__':
    main()
