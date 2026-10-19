import copy
import unittest
import tempfile
import json
import yaml
from PIL import Image
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from dataset.pair_list_spectrogram import (
    PairListSpectrogramDataset, select_target_records, condition_view_rng, select_conditions,
)
from tools.seeg_recovery import recovery_configs
from tools.seeg_evaluate import training_latent_mean


class RecoveryTests(unittest.TestCase):
    def records(self):
        conditions = [dict(patient='donor', path=str(10+i), paths=[str(10+i), str(30+i)],
                           coord=(i,0,0), distance=i+1) for i in range(8)]
        return [dict(target_id=f'p,A{i}/x.png', target_patient='p', region=f'R{i}',
                     target_path=str(i), target_coord=(0,0,0), conditions=conditions) for i in range(2)]

    def dataset(self, mode):
        ds = PairListSpectrogramDataset(self.records(), ['p'], donor_patients=['donor'],
            max_conditions=3, k_neg=1, training=True, condition_sampling=mode,
            condition_views=4, fixed_negatives=True)
        ds._load = lambda path: torch.full((4,2,2), float(path))
        return ds

    def test_bank_reproducibility_and_negative_control(self):
        bank, fixed, resampled = [self.dataset(mode) for mode in ['bank','fixed','resample']]
        with patch('dataset.pair_list_spectrogram.random.randrange', return_value=0):
            zero = bank[0]
        np.testing.assert_array_equal(zero['source'], fixed[0]['source'])
        for view in range(4):
            with patch('dataset.pair_list_spectrogram.random.randrange', return_value=view):
                a, b = bank[0], bank[0]
            np.testing.assert_array_equal(a['source'], b['source'])
            np.testing.assert_array_equal(a['negative'], zero['negative'])
        np.testing.assert_array_equal(resampled[0]['negative'], zero['negative'])
        views = [select_conditions(self.records()[0], 3,
                 condition_view_rng(42,0,'p,A0/x.png',v), True) for v in range(4)]
        self.assertTrue(any(v != views[0] for v in views[1:]))

    def test_balanced_selection_and_training_only_baseline(self):
        records = [dict(target_patient=p, target_path=f'{p}{i}') for p in ['p1','p2','p3'] for i in range(10)]
        subset = select_target_records(records, 6, 42, True)
        self.assertEqual([sum(r['target_patient']==p for r in subset) for p in ['p1','p2','p3']], [2,2,2])
        self.assertEqual(subset, select_target_records(records,6,42,True))
        self.assertEqual(select_target_records(records,None), records)
        mean = training_latent_mean([{'target_path':'a'},{'target_path':'b'}],
                                    lambda path: np.full((2,2,4), 1 if path=='a' else 3))
        np.testing.assert_array_equal(mean, np.full((2,2,4),2))

    def test_training_and_validation_evaluation_baselines(self):
        from tools.seeg_evaluate import main
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            image_root=root/'images'
            latent_root=root/'latents'
            paths={}
            for split,patient,value in [('train','p',.1),('train','d',.2),('val','q',.4)]:
                relative=Path(split)/patient/'A'/'segment.png'
                path=image_root/relative
                path.parent.mkdir(parents=True,exist_ok=True)
                Image.new('RGB',(32,32),(100,100,100)).save(path)
                latent=latent_root/relative.with_suffix('.npy')
                latent.parent.mkdir(parents=True,exist_ok=True)
                np.save(latent,np.full((2,2,4),value,dtype=np.float32))
                paths[patient]=str(path)
            records=[]
            for patient,donor in [('p','d'),('q','p')]:
                records.append({'target_id':patient+',A/segment.png','channel_id':patient+',A',
                    'target_patient':patient,'target_path':paths[patient],'region':'R',
                    'conditions':[{'patient':donor,'path':paths[donor]}]})
            manifest=root/'split.json'
            manifest.write_text(json.dumps({'train':['p','d'],'val':['q'],'test':[]}))
            cfg=root/'config.yaml'
            cfg.write_text(yaml.safe_dump({'dataset':{'image_root':str(image_root),
                'latent_root':str(latent_root),'split_manifest':str(manifest),'table_path':'unused',
                'resolution':32,'donor_protocol':'train_reference','train_max_targets':1,
                'condition_sampling':'fixed','seed':42},'train':{}}))
            commands=[]
            def generate(cmd,check):
                commands.append(cmd)
                patient=cmd[cmd.index('--target-id')+1].split(',')[0]
                folder=Path(cmd[cmd.index('--outdir')+1]);folder.mkdir(parents=True)
                for filename in ['target_vae_reconstruction.png','sample_000000.png']:
                    Image.open(paths[patient]).save(folder/filename)
                relative=Path(paths[patient]).relative_to(image_root).with_suffix('.npy')
                np.save(folder/'latent_000000.npy',np.load(latent_root/relative))
                (folder/'conditions.txt').write_text(paths['d' if patient=='p' else 'p']+'\n')
            for split in ['train','val']:
                out=root/split
                with patch('tools.seeg_evaluate.build_pair_list_records',return_value=records), patch('tools.seeg_evaluate.subprocess.run',side_effect=generate), patch('sys.argv',[
                    'eval','--config',str(cfg),'--init-from','fake','--split',split,
                    '--panel',str(root/(split+'_panel.json')),'--targets-per-patient','1',
                    '--num-samples','1','--outdir',str(out)]):
                    main()
                result=json.loads((out/'evaluation.json').read_text())
                self.assertEqual(result['patient_macro']['latent_l1'],0.)
                np.testing.assert_allclose(np.load(out/'training_mean_latent.npy'),.1)
            self.assertIn('--condition-seed',commands[0])
            self.assertNotIn('--condition-seed',commands[1])

    def test_pair_plot_and_separate_semantic_monitor(self):
        from utils.logging import WandbLogger
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            (root/'metrics.jsonl').write_text('\n'.join(json.dumps({
                'step':i, 'loss_pair':1/i, 'val/loss_pair':.5,
                'semantic_loss':.7, 'val/semantic_loss':.69}) for i in (1,2,3)))
            logger=WandbLogger()
            logger.offline_dir=root
            self.assertEqual(logger.save_loss_plot(),root/'loss_curve.png')
            self.assertTrue((root/'semantic_loss_curve.png').is_file())

    def test_controlled_configs_and_preserved_base(self):
        base = {'dataset':{'seed':42}, 'train':{'lambda_drift':.1, 'forward_dict':{'fixed_noise_seed':123}},
                'model':{'in_channels':4}, 'optimizer':{'lr_schedule':{'learning_rate':.0001}}}
        original = copy.deepcopy(base)
        configs = recovery_configs(base)
        self.assertEqual(base,original)
        values = list(configs.values())
        reference = copy.deepcopy(values[0])
        reference['dataset'].pop('condition_sampling')
        for cfg in values:
            other=copy.deepcopy(cfg)
            mode=other['dataset'].pop('condition_sampling')
            self.assertIn(mode, ['fixed','resample','bank'])
            self.assertEqual(other,reference)
            self.assertGreater(cfg['train']['lambda_drift'],0)
            self.assertTrue(cfg['train']['forward_dict']['stable_validation'])
            self.assertNotIn('fixed_noise_seed',cfg['train']['forward_dict'])
            self.assertEqual(cfg['model'],base['model'])
        scale = recovery_configs(base,stage='scale')
        self.assertTrue(all(c['dataset']['train_max_targets']==256 and c['dataset']['train_patient_balanced'] for c in scale.values()))
        with self.assertRaises(ValueError):
            recovery_configs(dict(base,train={'lambda_drift':0}))


if __name__ == '__main__':
    unittest.main()
