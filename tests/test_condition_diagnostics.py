import unittest
import jax.numpy as jnp
import numpy as np
from utils.condition_diagnostics import module_update_metrics
from tools.seeg_compare import latent_scores


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

    def test_latent_baselines(self):
        target = np.ones((2, 2, 4))
        result = latent_scores(target, target * .5, target, np.stack([target, target]))
        self.assertEqual(result, {'zero_l1': 1., 'train_mean_l1': .5,
                                  'train_median_l1': 0., 'generated_l1': 0.})


if __name__ == '__main__':
    unittest.main()
