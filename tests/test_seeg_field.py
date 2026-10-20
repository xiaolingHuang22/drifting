import copy
import random
import unittest
from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np

from drift_loss import drift_loss
from dataset.pair_list_spectrogram import select_positive_paths
from tools.seeg_field_experiments import field_configs


class FieldTests(unittest.TestCase):
    def test_channel_positives_unique_anchor_and_padding(self):
        record = {'target_path':'train/P/A/anchor.png',
                  'target_paths':['train/P/A/anchor.png','train/P/A/other.png']}
        paths, mask = select_positive_paths(record,4,random.Random(42))
        self.assertEqual(paths[:2],['train/P/A/anchor.png','train/P/A/other.png'])
        self.assertEqual(mask,[1.,1.,0.,0.])
        self.assertEqual(paths[2:],[record['target_path']]*2)
        self.assertEqual(select_positive_paths(record,1,random.Random(42)),([record['target_path']],[1.]))
        with self.assertRaises(ValueError):
            select_positive_paths(dict(record,target_paths=['test/Q/A/other.png']),4,random.Random(0))

    def gradients(self, pos, neg=None, **kwargs):
        gen=jnp.array([[[.1,.3],[.5,.2],[.3,.8],[.9,.6]]])
        return jax.grad(lambda x: drift_loss(x,pos,neg,R_list=(.2,),**kwargs)[0].mean())(gen)

    def test_zero_real_negatives_equal_absent_negatives(self):
        pos=jnp.array([[[.2,.7],[.6,.3]]])
        neg=jnp.array([[[7.,-9.]]])
        a=self.gradients(pos,neg,real_negative_weight=0.)
        b=self.gradients(pos,None)
        np.testing.assert_allclose(a,b,rtol=1e-5,atol=1e-5)
        self.assertGreater(float(jnp.linalg.norm(a)),0.)
        self.assertTrue(np.isfinite(a).all())

    def test_masked_positive_padding_does_not_change_field(self):
        pos=jnp.array([[[.2,.7]]])
        neg=jnp.array([[[.8,-.2]]])
        padded=jnp.concatenate([pos,jnp.array([[[100.,-50.]]])],axis=1)
        a=self.gradients(pos,neg)
        b=self.gradients(padded,neg,weight_pos=jnp.array([[1.,0.]]),
                         positive_mask=jnp.array([[[1.,0.]]*4]))
        np.testing.assert_allclose(a,b,rtol=1e-5,atol=1e-5)

    def test_conditional_update_with_multiple_masked_positives(self):
        import optax
        # Training initializes distributed runtime at import; unit tests already
        # have a local JAX backend, so initialize only the single-device mesh.
        with patch("utils.misc.run_init"):
            from train import TrainState, train_step_conditional
        from utils.hsdp_util import set_global_mesh
        set_global_mesh(1)
        params={'image':jnp.full((2,2,4),.1)}
        def apply(variables,c,rngs,**kwargs):
            return {'samples':jnp.broadcast_to(variables['params']['image'],(len(c),2,2,4))
                    +.01*jax.random.normal(rngs['noise'],(len(c),2,2,4))}
        def features(params,images,**kwargs):
            return {'raw':images.reshape(len(images),1,-1)}
        state=TrainState.create(apply_fn=apply,params=params,ema_params=params,tx=optax.sgd(.001))
        batch={'source':jnp.ones((1,2,2,2,4)),'target_true':jnp.full((1,2,2,4),.3),
               'target_ref':jnp.stack([jnp.full((1,2,2,4),v) for v in [.3,.5,.3,.3]],axis=1),
               'negative':jnp.full((1,1,2,2,4),-.2),'source_coord':jnp.zeros((1,3)),
               'target_coord':jnp.zeros((1,3)),'labels':jnp.zeros(1,dtype=jnp.int32),
               'positive_mask':jnp.array([[1.,1.,0.,0.]]),'has_target_true':jnp.ones(1)}
        new,metrics=train_step_conditional(state,batch,{},features,jax.random.PRNGKey(0),
            learning_rate_fn=lambda step:jnp.array(.001),gen_per_label=4,
            loss_kwargs={'R_list':(.2,),'real_negative_weight':0.},lambda_pair=1.,lambda_drift=.1)
        self.assertTrue(all(jnp.isfinite(x).all() for x in jax.tree_util.tree_leaves(new.params)))
        self.assertEqual(float(metrics['positive_count']),2.)
        self.assertEqual(float(metrics['generated_per_condition']),4.)
        self.assertEqual(float(metrics['real_negative_weight']),0.)
        self.assertGreater(float(jnp.linalg.norm(new.params['image']-params['image'])),0.)

    def test_experiment_changes_one_factor(self):
        base={'dataset':{'seed':42,'condition_sampling':'bank'},
              'train':{'lambda_drift':.1,'forward_dict':{},'loss_kwargs':{'R_list':[.2]}},
              'optimizer':{'lr_schedule':{'learning_rate':.0001}},'model':{'in_channels':4}}
        original=copy.deepcopy(base)
        configs=list(field_configs(base).values())
        self.assertEqual(base,original)
        def diff(a,b):
            differences=[]
            def walk(x,y,path=''):
                for key in x:
                    if isinstance(x[key],dict):walk(x[key],y[key],path+key+'.')
                    elif x[key]!=y[key]:differences.append(path+key)
            walk(a,b)
            return differences
        self.assertEqual(diff(configs[0],configs[1]),['train.forward_dict.gen_per_label'])
        self.assertEqual(diff(configs[1],configs[2]),['dataset.k_pos'])
        for cfg in configs[3:]:
            self.assertEqual(diff(configs[2],cfg),['train.loss_kwargs.real_negative_weight'])
        self.assertTrue(all(c['train']['lambda_drift']>.0 for c in configs))
        self.assertTrue(all(c['dataset']['val_k_pos']==1 for c in configs))


if __name__ == '__main__':
    unittest.main()
