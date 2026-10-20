"""Matched drift-active experiments for generated, positive and negative samples."""
import argparse
import copy
import math
from pathlib import Path

import yaml


def field_configs(base, targets=256, steps=100000, generated=4, positives=4, negative_weight=.25):
    if targets < 1 or steps < 1 or generated < 2 or positives < 2:
        raise ValueError('Positive targets/steps and at least two generated/positive samples required')
    if not math.isfinite(negative_weight) or not 0 < negative_weight < 1:
        raise ValueError('Reduced real-negative weight must lie strictly between zero and one')
    drift = float(base['train'].get('lambda_drift', .1))
    if not math.isfinite(drift) or drift <= 0:
        raise ValueError('These experiments require drifting to remain active')
    common = copy.deepcopy(base)
    common['dataset'].update(train_max_targets=targets, k_neg=1, k_pos=1,
        positive_sampling='fixed', donor_protocol='train_reference',
        batch_size=1, eval_batch_size=1)
    common['train'].update(total_steps=steps, train_batch_size=1, init_from='', resume_min_step=0,
        lambda_drift=drift, lambda_pair=1., lambda_condition=0., lambda_tv=0.,
        save_per_step=5000, keep_every=5000, keep_last=8,
        log_gen_diagnostics=True, log_update_diagnostics=True)
    forward = common['train'].setdefault('forward_dict', {})
    forward.pop('fixed_noise_seed', None)
    # Stable validation keeps SINGLE exact-target positives in every variant.
    forward.update(stable_validation=True, validation_noise_seed=2026, validation_use_ema=True)
    common['optimizer']['lr_schedule']['total_steps'] = steps
    result = {}
    for name, g, p, w in [('a_anchor',1,1,1.), ('b_multigen',generated,1,1.),
                           ('c_multipos',generated,positives,1.),
                           ('d_softneg',generated,positives,negative_weight),
                           ('e_norealneg',generated,positives,0.)]:
        cfg = copy.deepcopy(common)
        cfg['dataset']['k_pos'] = p
        cfg['dataset']['val_k_pos'] = 1
        cfg['train']['forward_dict']['gen_per_label'] = g
        cfg['train'].setdefault('loss_kwargs', {})['real_negative_weight'] = w
        result[f'seeg_field_{name}.yaml'] = cfg
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--targets', type=int, default=256)
    p.add_argument('--steps', type=int, default=100000)
    p.add_argument('--generated', type=int, default=4)
    p.add_argument('--positives', type=int, default=4)
    p.add_argument('--negative-weight', type=float, default=.25)
    a = p.parse_args()
    configs = field_configs(yaml.safe_load(Path(a.config).expanduser().read_text()),
                           a.targets, a.steps, a.generated, a.positives, a.negative_weight)
    output = Path(a.output).expanduser()
    if any((output/name).exists() for name in configs):
        raise FileExistsError('Use a new config output directory')
    output.mkdir(parents=True, exist_ok=True)
    for name, cfg in configs.items():
        path = output/name
        path.write_text(yaml.safe_dump(cfg, sort_keys=False))
        print(path)


if __name__ == '__main__':
    main()
