import tempfile
import unittest
from pathlib import Path

import torch

from tdgnet import TraceSample
from tdgnet.data import load_trace, save_trace
from tdgnet.storage import graphs_from_sparse_sample, load_shard, save_shard


class StorageTests(unittest.TestCase):
    def make_sample(self):
        a = torch.tensor([[0.9699999, 0.0300001], [0.0499999, 0.9500001]])
        return TraceSample([a], [torch.tensor([[1.234567, 2.0], [3.0, 4.0]])],
                           torch.tensor([False, True]), 1.0, torch.tensor([0.0]))

    def test_dense_roundtrip_preserves_supervision(self):
        s = self.make_sample()
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory) / 'trace.pt'
            save_trace(s, p)
            restored = load_trace(p)
        torch.testing.assert_close(s.features[0], restored.features[0])
        torch.testing.assert_close(s.token_labels, restored.token_labels)

    def test_sparse_roundtrip_preserves_threshold_precision(self):
        s = self.make_sample()
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory) / 'shard_00000.pt'
            save_shard([s], p, tau_min=0.03)
            restored = load_shard(p)[0]
        self.assertTrue(torch.equal(s.attentions[0], restored.attentions[0]))
        self.assertTrue(torch.equal(s.features[0], restored.features[0]))
        index, _ = graphs_from_sparse_sample(restored, tau=0.03)[0]
        self.assertIn([1, 0], index.T.tolist())
        with self.assertRaises(ValueError):
            graphs_from_sparse_sample(restored, tau=0.02)

    def test_training_rejects_unavailable_attention_threshold(self):
        from tdgnet import TDGNet
        from tdgnet.train import train_tdgnet
        a, b = self.make_sample(), self.make_sample()
        a.label, b.label = 0.0, 1.0
        a.metadata['attention_storage_tau_min'] = 0.07
        with self.assertRaisesRegex(ValueError, 'below the stored'):
            train_tdgnet(TDGNet(2), [a, b], [a, b], epochs=1)
