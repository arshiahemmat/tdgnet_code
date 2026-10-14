import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F

from tdgnet import TDGNet, TraceSample, collate_traces, load_checkpoint, save_checkpoint
from tdgnet.train import evaluate_response_auroc, evaluate_token_auroc, train_tdgnet


def samples():
    result = []
    for label in [0.0, 1.0]:
        result.append(TraceSample([torch.eye(3)] * 2, [torch.ones(3, 2)] * 2,
                                  torch.tensor([False, True, True]), label,
                                  torch.tensor([0.0, 1.0])))
    return result


class TrainingTests(unittest.TestCase):
    def test_token_loss_uses_token_targets(self):
        model = TDGNet(2, hidden_dim=4)
        with torch.no_grad():
            for p in model.token_mlp.parameters():
                p.zero_()
            model.token_mlp.net[-1].bias.fill_(2.0)
        expected = F.binary_cross_entropy_with_logits(torch.tensor([2.0, 2.0]), torch.tensor([0.0, 1.0])).item()
        data = samples()
        for item in data:
            item.label = 1.0  # Deliberately different from the mixed token targets.
        result = train_tdgnet(model, data, data, task='token', epochs=1, batch_size=2, lr=0)
        self.assertAlmostEqual(result.history[0]['train_loss'], expected, places=6)
        self.assertEqual(result.model.trained_heads, ['token'])
        self.assertIsNone(model.pred_mlp.net[-1].weight.grad)

    def test_joint_training_updates_both_heads(self):
        model = TDGNet(2, hidden_dim=4)
        train_tdgnet(model, samples(), samples(), task='joint', epochs=1, batch_size=2)
        self.assertEqual(model.trained_heads, ['response', 'token'])
        self.assertTrue(torch.isfinite(model.token_mlp.net[-1].weight.grad).all())

    def test_response_training_leaves_token_head_untrained(self):
        model = TDGNet(2, hidden_dim=4)
        train_tdgnet(model, samples(), samples(), epochs=1, batch_size=2)
        self.assertEqual(model.trained_heads, ['response'])
        with self.assertRaisesRegex(ValueError, 'not been trained'):
            evaluate_token_auroc(model, samples())

    def test_untrained_response_head_cannot_be_evaluated(self):
        with self.assertRaisesRegex(ValueError, 'not been trained'):
            evaluate_response_auroc(TDGNet(2), samples())

    def test_validation_requires_both_classes(self):
        with self.assertRaisesRegex(ValueError, 'both classes'):
            train_tdgnet(TDGNet(2), samples(), samples()[:1], epochs=1)

    def test_checkpoint_roundtrip_covers_nondefault_architecture(self):
        model = TDGNet(2, hidden_dim=8, msg_hidden_dim=5, dropout=0.1, tau=0.1,
                      num_mp_layers=3, num_temporal_heads=4,
                      include_self_loops=False, pooling='response').eval()
        model.mark_trained(response=True)
        batch = collate_traces(samples())
        args = {k: batch[k] for k in ['features', 'attentions', 'response_mask', 'node_mask']}
        with torch.no_grad():
            before = model(**args)
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory) / 'model.pt'
            save_checkpoint(model, p)
            restored = load_checkpoint(p)
        self.assertEqual(model.get_config(), restored.get_config())
        self.assertEqual(restored.trained_heads, ['response'])
        self.assertFalse(any(k.startswith('msg_mlp.') for k in restored.state_dict()))
        with torch.no_grad():
            after = restored(**args)
        torch.testing.assert_close(before.response_logit, after.response_logit)
        torch.testing.assert_close(before.token_logits, after.token_logits)

    def test_unlabelled_response_can_receive_token_training(self):
        s = samples()
        for x in s:
            x.label = None
        result = train_tdgnet(TDGNet(2, hidden_dim=4), s, s, task='token', epochs=1)
        self.assertEqual(result.selection_head, 'token')
