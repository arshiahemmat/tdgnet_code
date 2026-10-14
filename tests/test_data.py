import unittest

import torch

from tdgnet import TraceSample, collate_traces, select_trace_keyframes


def sample(n=3, label=1.0, token_labels=None):
    return TraceSample([torch.eye(n)] * 2, [torch.ones(n, 2)] * 2,
                       torch.arange(n) >= 1, label, token_labels)


class DataTests(unittest.TestCase):
    def test_full_and_response_only_targets_align(self):
        a = sample(3, token_labels=torch.tensor([1.0, 0.0, 1.0]))
        b = sample(4, token_labels=torch.tensor([1.0, 0.0, 1.0]))
        batch = collate_traces([a, b])
        self.assertTrue(torch.isnan(batch['token_labels'][:, 0]).all())
        self.assertTrue(torch.isnan(batch['token_labels'][0, 3]))
        torch.testing.assert_close(batch['token_labels'][0, 1:3], torch.tensor([0.0, 1.0]))
        torch.testing.assert_close(batch['token_labels'][1, 1:4], torch.tensor([1.0, 0.0, 1.0]))

    def test_missing_supervision_stays_missing(self):
        batch = collate_traces([sample(label=None)])
        self.assertTrue(torch.isnan(batch['labels']).all())
        self.assertTrue(torch.isnan(batch['token_labels']).all())

    def test_keyframes_are_selected_from_full_states(self):
        s = select_trace_keyframes([torch.eye(2)] * 17,
                                  [torch.full((2, 2), float(i)) for i in range(17)],
                                  torch.ones(2, dtype=torch.bool))
        self.assertEqual([int(x[0, 0]) for x in s.features], [0, 8, 12, 16])
        self.assertEqual(s.metadata['diffusion_times'], [16, 8, 4, 0])

    def test_reject_wrong_feature_width_and_labels(self):
        with self.assertRaises(ValueError):
            TraceSample([torch.eye(2)] * 2, [torch.ones(2, 2), torch.ones(2, 3)],
                        torch.ones(2, dtype=torch.bool))
        with self.assertRaises(ValueError):
            sample(label=-1)
        with self.assertRaises(ValueError):
            sample(token_labels=torch.tensor([0.0, 2.0]))

    def test_response_mask_must_be_a_vector(self):
        with self.assertRaises(ValueError):
            TraceSample([torch.eye(2)], [torch.ones(2, 2)], torch.ones(1, 2, dtype=torch.bool))
