import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import torch
from tdgnet import TraceSample
from tdgnet.data import save_trace

ROOT = Path(__file__).resolve().parents[1]


class CLITests(unittest.TestCase):
    def test_train_evaluate_uses_checkpoint_config_and_skips_untrained_head(self):
        environment = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for split in ['train', 'val', 'test']:
                for i in range(2):
                    s = TraceSample([torch.eye(3)] * 2, [torch.ones(3, 2)] * 2,
                                    torch.tensor([False, True, True]), float(i),
                                    torch.tensor([0.0, 1.0]))
                    save_trace(s, root / split / f'trace_{i}.pt')
            config = root / 'config.yaml'
            config.write_text('model:\n  hidden_dim: 8\n  msg_hidden_dim: 5\n  num_temporal_heads: 2\n'
                              '  num_mp_layers: 2\n  pooling: response\ntrain:\n  epochs: 1\n  batch_size: 2\n  device: null\n')
            subprocess.run([sys.executable, str(ROOT/'scripts/train.py'), '--config', str(config),
                            '--train_dir', str(root/'train'), '--val_dir', str(root/'val'),
                            '--run_dir', str(root/'run')], cwd=ROOT, env=environment,
                           check=True, capture_output=True, text=True)
            completed = subprocess.run([sys.executable, str(ROOT/'scripts/evaluate.py'),
                                        '--checkpoint', str(root/'run/best.pt'), '--test_dir', str(root/'test')],
                                       cwd=ROOT, env=environment, check=True, capture_output=True, text=True)
            result = json.loads(completed.stdout)
            self.assertEqual(result['trained_heads'], ['response'])
            self.assertEqual(result['response_n'], 2)
            self.assertIsNone(result['token_auroc'])
            self.assertEqual(result['token_n'], 0)

    def test_synthetic_quickstart(self):
        completed = subprocess.run([sys.executable, str(ROOT/'examples/quickstart.py')],
                                   cwd=ROOT, check=True, capture_output=True, text=True)
        self.assertIn('Synthetic CPU example completed.', completed.stdout)
        self.assertIn('Checkpoint round-trip: passed', completed.stdout)
