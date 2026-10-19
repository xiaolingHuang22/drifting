"""Generate drift-active controls for donor variability and staged scaling."""
import argparse
import copy
import math
from pathlib import Path

import yaml


def recovery_configs(base, stage='conditions', steps=None, views=4):
    if views < 1:
        raise ValueError('views must be positive')
    drift = float(base['train'].get('lambda_drift', .1))
    if not math.isfinite(drift) or drift <= 0:
        raise ValueError('Recovery experiments require active drifting in the base config.')
    if stage not in ('conditions', 'scale'):
        raise ValueError('Unknown stage')
    steps = steps if steps is not None else (40000 if stage == 'conditions' else 100000)
    if steps <= 0:
        raise ValueError('steps must be positive')
    count = 32 if stage == 'conditions' else 256
    modes = ('fixed', 'resample', 'bank') if stage == 'conditions' else ('resample', 'bank')
    common = copy.deepcopy(base)
    common['dataset'].update(train_max_targets=count, donor_protocol='train_reference',
        fixed_conditions=False, fixed_negatives=True, condition_views=views,
        train_patient_balanced=(stage == 'scale'), val_max_targets=64,
        batch_size=1, eval_batch_size=1)
    common['train'].update(total_steps=steps, train_batch_size=1, init_from='', resume_min_step=0,
        lambda_drift=drift, lambda_pair=1., lambda_condition=0., lambda_tv=0.,
        enable_val_loss=True, val_loss_per_step=1000, val_loss_batches=64,
        log_gen_diagnostics=True, log_update_diagnostics=True,
        save_per_step=5000, keep_every=5000, keep_last=8)
    forward = common['train'].setdefault('forward_dict', {})
    forward.pop('fixed_noise_seed', None)
    forward.update(gen_per_label=1, stable_validation=True, validation_noise_seed=2026,
                   validation_use_ema=True)
    common['optimizer']['lr_schedule']['total_steps'] = steps
    return {f'seeg_recovery{count}_{mode}.yaml': dict_with_sampling(common, mode) for mode in modes}


def dict_with_sampling(common, mode):
    config = copy.deepcopy(common)
    config['dataset']['condition_sampling'] = mode
    return config


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--stage', choices=['conditions', 'scale'], default='conditions')
    p.add_argument('--steps', type=int)
    p.add_argument('--views', type=int, default=4)
    a = p.parse_args()
    base = yaml.safe_load(Path(a.config).expanduser().read_text())
    configs = recovery_configs(base, a.stage, a.steps, a.views)
    output = Path(a.output).expanduser()
    if any((output/name).exists() for name in configs):
        raise FileExistsError('Use a fresh config output directory; refusing to overwrite.')
    output.mkdir(parents=True, exist_ok=True)
    for name, config in configs.items():
        path = output/name
        path.write_text(yaml.safe_dump(config, sort_keys=False))
        print(path)


if __name__ == '__main__':
    main()
