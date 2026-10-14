import tempfile
import unittest
from pathlib import Path

from tdgnet.config import deep_merge, load_configs, load_yaml


class ConfigTests(unittest.TestCase):
    def test_recursive_merge_preserves_unrelated_fields(self):
        source = {'model': {'tau': 0.05, 'hidden_dim': 128}, 'train': {'seed': 42}}
        merged = deep_merge(source, {'model': {'tau': 0.03}})
        self.assertEqual(merged['model'], {'tau': 0.03, 'hidden_dim': 128})
        self.assertEqual(source['model']['tau'], 0.05)
        self.assertEqual(merged['train']['seed'], 42)

    def test_later_config_wins(self):
        with tempfile.TemporaryDirectory() as directory:
            a, b = Path(directory) / 'a.yaml', Path(directory) / 'b.yaml'
            a.write_text('model:\n  tau: 0.05\n  hidden_dim: 128\n')
            b.write_text('model:\n  tau: 0.03\n')
            self.assertEqual(load_configs([a, b])['model'], {'tau': 0.03, 'hidden_dim': 128})

    def test_config_must_be_mapping(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory) / 'bad.yaml'
            p.write_text('- one\n- two\n')
            with self.assertRaises(ValueError):
                load_yaml(p)
