import unittest
import tempfile
from pathlib import Path
import ml_dtypes
import json
import yaml
from unittest.mock import patch
import jax.numpy as jnp
import numpy as np
from utils.condition_diagnostics import module_update_metrics
from tools.seeg_compare import latent_scores, load_latent


class DiagnosticTests(unittest.TestCase):
    def test_actual_update_and_module_filter(self):
        old = {'condition_conv_1': {'w': jnp.array([3., 4.])}, 'other': {'w': jnp.ones(1)}}
        grads = {'condition_conv_1': {'w': jnp.array([0., 2.])}, 'other': {'w': jnp.ones(1)}}
        new = {'condition_conv_1': {'w': jnp.array([3., 5.])}, 'other': {'w': jnp.ones(1)}}
        metrics = module_update_metrics(old, grads, new)
        prefix = 'condition_diag/condition_conv_1/'
        self.assertAlmostEqual(float(metrics[prefix + 'grad_norm']), 2.)
        self.assertAlmostEqual(float(metrics[prefix + 'update_norm']), 1.)
        self.assertAlmostEqual(float(metrics[prefix + 'relative_update']), .2)
        self.assertEqual(len(metrics), 4)

    def test_pair_only_config(self):
        from tools.seeg_checks import main
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory) / 'base.yaml'
            out = Path(directory) / 'out.yaml'
            base.write_text(yaml.safe_dump({'dataset': {'image_root': '~/RAE_first_exp'},
                'train': {'lambda_drift': .1, 'forward_dict': {'gen_per_label': 1}},
                'model': {'in_channels': 4}, 'optimizer': {'lr_schedule': {}}}))
            with patch('sys.argv', ['checks', 'overfit-config', '--config', str(base),
                '--output', str(out), '--pair-only', '--targets', '4', '--fixed-noise-seed', '123']):
                main()
            config = yaml.safe_load(out.read_text())
            self.assertEqual(config['train']['lambda_drift'], 0.)
            self.assertEqual(config['train']['lambda_pair'], 1.)
            self.assertEqual(config['train']['forward_dict']['fixed_noise_seed'], 123)
            self.assertEqual(config['dataset']['train_max_targets'], 4)
            self.assertEqual(config['model'], {'in_channels': 4})
            varying = Path(directory) / 'varying.yaml'
            with patch('sys.argv', ['checks', 'overfit-config', '--config', str(out),
                '--output', str(varying), '--pair-only', '--targets', '4', '--varying-noise']):
                main()
            changed = yaml.safe_load(varying.read_text())
            self.assertNotIn('fixed_noise_seed', changed['train']['forward_dict'])
            self.assertEqual(changed['dataset']['batch_size'], 1)
            self.assertEqual(changed['model'], config['model'])

    def test_extension_preserves_experiment(self):
        from tools.seeg_checks import main
        with tempfile.TemporaryDirectory() as directory:
            base, out = Path(directory) / 'base.yaml', Path(directory) / 'out.yaml'
            original = {'dataset': {'train_max_targets': 32, 'seed': 42},
                        'train': {'total_steps': 5000, 'lambda_pair': 1.,
                                  'forward_dict': {'fixed_noise_seed': 123}},
                        'optimizer': {'lr_schedule': {'lr_schedule': 'const', 'learning_rate': .0001}}}
            base.write_text(yaml.safe_dump(original))
            with patch('sys.argv', ['checks', 'extend-config', '--config', str(base), '--output', str(out)]):
                main()
            cfg = yaml.safe_load(out.read_text())
            self.assertEqual(cfg['dataset'], original['dataset'])
            self.assertEqual(cfg['train']['forward_dict'], original['train']['forward_dict'])
            self.assertEqual(cfg['train']['resume_min_step'], 5000)
            self.assertEqual(cfg['train']['total_steps'], 40000)
            self.assertEqual(cfg['train']['keep_every'], 5000)

    def test_bfloat16_saved_latents(self):
        values = np.array([0., 1., -2., .125], dtype=ml_dtypes.bfloat16)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'latent.npy'
            np.save(path, values)
            self.assertEqual(np.load(path).dtype, np.dtype('V2'))
            loaded = load_latent(path)
            self.assertEqual(loaded.dtype, np.float32)
            np.testing.assert_array_equal(loaded, [0., 1., -2., .125])
            np.save(path, values.astype(np.float32))
            np.testing.assert_array_equal(load_latent(path), loaded)

    def test_latent_baselines(self):
        target = np.ones((2, 2, 4))
        result = latent_scores(target, target * .5, target, np.stack([target, target]))
        self.assertEqual(result, {'zero_l1': 1., 'train_mean_l1': .5,
                                  'train_median_l1': 0., 'generated_l1': 0.})


if __name__ == '__main__':
    unittest.main()
