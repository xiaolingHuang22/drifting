import csv
import json
import random
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

import numpy as np
from PIL import Image

from dataset.pair_list_spectrogram import (
    PairListSpectrogramDataset, build_pair_list_records,
    latent_cache_path, load_patient_split, select_conditions,
)


class SegmentDatasetTest(unittest.TestCase):
    def test_segments_splits_cache_and_sampling(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            images = root / 'images'
            patients = ['HHFU059', 'FirstName_LastName', 'VAL1', 'VAL2', 'TEST1', 'TEST2']
            manifest = {'train': patients[:2], 'val': patients[2:4], 'test': patients[4:]}
            split_path = root / 'split.json'
            split_path.write_text(json.dumps(manifest))
            fields = ['patient_channel', 'location', 'x1', 'y1', 'z1', 'pair', 'pair_location', 'x2', 'y2', 'z2', 'distance']
            table = root / 'pairs.csv'
            with table.open('w', newline='') as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                for split, pair in manifest.items():
                    for patient, donor in (pair, pair[::-1]):
                        for channel, region in [('A1', 'region1'), ('A7', 'region2')]:
                            folder = images / split / patient / channel
                            folder.mkdir(parents=True)
                            for start in [1320, 2400]:
                                Image.new('RGB', (256, 256)).save(folder / f'{channel}_{patient}-DT0120HK_{start}_{start+120}.png')
                            writer.writerow(dict(zip(fields, [f'{patient},{channel}', region, 0, 0, 0, f'{donor},{channel}', region, 1, 0, 0, 1])))
                    # Cross-split donor must be excluded by the dataset.
                    if split == 'train':
                        writer.writerow(dict(zip(fields, [f'{pair[0]},A1', 'region1', 0, 0, 0, 'VAL1,A1', 'region1', 1, 0, 0, 1])))
            records = build_pair_list_records(table, images, split_path)
            self.assertEqual(len(records), 24)
            self.assertEqual(records[0]['segment_seconds'], (1320, 1440))
            cache = root / 'cache'
            for path in images.rglob('*.png'):
                output = latent_cache_path(cache, path, images)
                output.parent.mkdir(parents=True, exist_ok=True)
                np.save(output, np.zeros((32, 32, 4), dtype=np.float32))
            ds = PairListSpectrogramDataset(records, manifest['train'], max_conditions=4, k_neg=1, latent_root=cache, image_root=images)
            self.assertEqual(len(ds), 8)
            self.assertTrue(all(c['patient'] in manifest['train'] for r in ds.records for c in r['conditions']))
            fixed = PairListSpectrogramDataset(records, manifest['train'], max_conditions=4,
                        k_neg=1, latent_root=cache, image_root=images, training=True,
                        fixed_conditions=True, max_targets=8, seed=42)
            self.assertEqual(len(fixed), 8)
            self.assertTrue(np.array_equal(fixed[0]['source'], fixed[0]['source']))
            self.assertEqual([r['target_id'] for r in fixed.records],
                [r['target_id'] for r in PairListSpectrogramDataset(records, manifest['train'],
                     max_targets=8, seed=42).records])
            sample = ds[0]
            self.assertEqual(tuple(sample['source'].shape), (4, 4, 32, 32))
            self.assertEqual(tuple(sample['target_true'].shape), (4, 32, 32))
            self.assertEqual(tuple(sample['negative'].shape), (1, 4, 32, 32))
            self.assertEqual(select_conditions(ds.records[0], 4, random.Random(7)), select_conditions(ds.records[0], 4, random.Random(7)))
            self.assertEqual(len({latent_cache_path(cache, p, images) for p in images.rglob('*.png')}), 24)
            import yaml
            from tools.seeg_checks import main
            config_path = root / 'config.yaml'
            config_path.write_text(yaml.safe_dump({'dataset': {
                'image_root': str(images), 'table_path': str(table),
                'split_manifest': str(split_path), 'train_max_targets': 8,
                'max_conditions': 4, 'seed': 42}}))
            report_path = root / 'audit.json'
            with patch('sys.argv', ['checks', 'audit', '--config', str(config_path), '--output', str(report_path)]):
                main()
            report = json.loads(report_path.read_text())
            self.assertEqual(report['train']['retained_segments'], 8)
            self.assertEqual(len(report['overfit_targets']), 8)
            manifest['val'].append(manifest['train'][0])
            split_path.write_text(json.dumps(manifest))
            with self.assertRaises(ValueError):
                load_patient_split(split_path)


if __name__ == '__main__':
    unittest.main()
