import csv
import json
import tempfile
import unittest
from unittest.mock import patch
import yaml
from pathlib import Path

import numpy as np
from PIL import Image

from tools.seeg_image_summary import summarize_comparison, write_summary
from utils.spectrogram_metrics import image_metrics


class ImageSummaryTests(unittest.TestCase):
    def fixture(self,root):
        for target,count in [('target_00',1),('target_01',3)]:
            for variant in ['raw','ema','raw_shuffled']:
                folder=root/target/variant
                folder.mkdir(parents=True)
                Image.new('RGB',(32,32),(0,0,0)).save(folder/'target_original.png')
                Image.new('RGB',(32,32),(128,128,128)).save(folder/'target_vae_reconstruction.png')
                for i in range(count):
                    value = 0 if target=='target_00' else 255
                    Image.new('RGB',(32,32),(value,value,value)).save(folder/f'sample_{i:06d}.png')

    def test_average_is_per_image_and_pooled_is_distinct(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            self.fixture(root)
            report=write_summary(root)
            raw=report['variants']['raw']
            self.assertEqual(raw['image_count'],4)
            self.assertEqual(raw['target_count'],2)
            stats=raw['references']['original']
            self.assertAlmostEqual(stats['mean_per_image']['mse'],.75)
            self.assertAlmostEqual(stats['mean_per_image']['rmse'],.75)
            self.assertAlmostEqual(stats['target_macro']['mse'],.5)
            self.assertAlmostEqual(stats['pooled_rmse'],np.sqrt(.75))
            self.assertAlmostEqual(stats['mean_per_image']['psnr_db'],30.)
            with (root/'image_metrics_summary.csv').open() as stream:
                self.assertEqual(len(list(csv.DictReader(stream))),6)
            json.loads((root/'image_metrics_summary.json').read_text())

    def test_manifest_ignores_stale_samples_and_targets(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            self.fixture(root)
            (root/'comparison_manifest.json').write_text(json.dumps({
                'status':'complete','num_samples':1,'target_directories':['target_00']}))
            result=summarize_comparison(root)
            self.assertEqual(result['target_count'],1)
            self.assertEqual(result['variants']['ema']['image_count'],1)
            self.assertEqual(result['variants']['ema']['references']['original']['mean_per_image']['mse'],0.)
            (root/'comparison_manifest.json').write_text(json.dumps({'status':'running'}))
            with self.assertRaisesRegex(ValueError,'incomplete'):
                summarize_comparison(root)

    def test_missing_sample_is_not_silently_skipped(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            self.fixture(root)
            with self.assertRaisesRegex(ValueError,'Missing expected'):
                summarize_comparison(root,num_samples=4)

    def test_comparison_automatically_exports_image_averages(self):
        from tools.seeg_compare import main
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            images=root/'images';cache=root/'cache';out=root/'comparison'
            relative=Path('train/P/A/segment.png')
            (images/relative).parent.mkdir(parents=True)
            Image.new('RGB',(32,32),(64,64,64)).save(images/relative)
            latent=cache/relative.with_suffix('.npy');latent.parent.mkdir(parents=True)
            np.save(latent,np.zeros((2,2,4),dtype=np.float32))
            config=root/'config.yaml'
            config.write_text(yaml.safe_dump({'dataset':{'image_root':str(images),'latent_root':str(cache)}}))
            audit=root/'audit.json'
            audit.write_text(json.dumps({'overfit_targets':[{'target_id':'P,A',
                'target_segment':'segment.png','inference_seed':42}]}))
            def generate(cmd,check):
                folder=Path(cmd[cmd.index('--outdir')+1]);folder.mkdir(parents=True)
                value=128 if '--shuffle-condition-images' in cmd else (64 if cmd[cmd.index('--init-from')+1]=='raw' else 96)
                for filename in ['target_original.png','target_vae_reconstruction.png']:
                    Image.new('RGB',(32,32),(64,64,64)).save(folder/filename)
                for i in range(2):
                    Image.new('RGB',(32,32),(value,value,value)).save(folder/f'sample_{i:06d}.png')
                    np.save(folder/f'latent_{i:06d}.npy',np.zeros((2,2,4),dtype=np.float32))
            with patch('tools.seeg_compare.subprocess.run',side_effect=generate), patch('sys.argv',[
                'compare','--config',str(config),'--audit',str(audit),'--raw','raw','--ema','ema',
                '--num-samples','2','--targets','1','--outdir',str(out)]):
                main()
            report=json.loads((out/'image_metrics_summary.json').read_text())
            self.assertEqual(report['variants']['raw']['image_count'],2)
            self.assertEqual(report['variants']['raw']['references']['original']['mean_per_image']['mse'],0.)
            self.assertGreater(report['variants']['raw_shuffled']['references']['original']['mean_per_image']['mse'],0.)
            self.assertEqual(json.loads((out/'comparison_manifest.json').read_text())['status'],'complete')
            self.assertTrue((out/'comparison.json').is_file())

    def test_dimensions_are_checked(self):
        with self.assertRaises(ValueError):
            image_metrics(np.zeros((32,32,3)),np.zeros((16,16,3)))


if __name__=='__main__':
    unittest.main()
