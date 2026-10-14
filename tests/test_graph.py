import unittest

import torch

from tdgnet.graph import build_graph_from_attention, default_keyframe_indices, zero_padding_attention


class GraphTests(unittest.TestCase):
    def test_query_key_direction_and_self_loops(self):
        a = torch.tensor([[0.7, 0.3], [0.1, 0.9]])
        index, attr = build_graph_from_attention(a, tau=0.2)
        edges = {tuple(pair) for pair in index.T.tolist()}
        self.assertEqual(edges, {(0, 0), (1, 0), (1, 1)})
        for pair, value in zip(index.T.tolist(), attr.flatten().tolist()):
            self.assertAlmostEqual(value, a[pair[1], pair[0]].item())

    def test_strict_threshold_and_optional_no_self_loops(self):
        a = torch.tensor([[0.5, 0.25], [0.3, 0.5]])
        index, _ = build_graph_from_attention(a, tau=0.25, include_self_loops=False)
        self.assertEqual(index.T.tolist(), [[0, 1]])

    def test_empty_graph_shapes(self):
        index, attr = build_graph_from_attention(torch.zeros(3, 3))
        self.assertEqual(tuple(index.shape), (2, 0))
        self.assertEqual(tuple(attr.shape), (0, 1))

    def test_padding_cannot_send_or_receive(self):
        a = zero_padding_attention(torch.ones(3, 3), torch.tensor([True, False, True]))
        index, _ = build_graph_from_attention(a)
        self.assertNotIn(1, index.flatten().tolist())

    def test_keyframes_include_true_endpoint(self):
        self.assertEqual(default_keyframe_indices(16), [0, 8, 12, 16])
        self.assertEqual(default_keyframe_indices(32), [0, 16, 24, 32])
        self.assertEqual(default_keyframe_indices(5), [0, 3, 4, 5])
        self.assertEqual(default_keyframe_indices(1), [0, 1])
        with self.assertRaises(ValueError):
            default_keyframe_indices(0)

    def test_invalid_attention_fails(self):
        for a in [torch.full((2, 2), float('nan')), torch.full((2, 2), -0.1)]:
            with self.assertRaises(ValueError):
                build_graph_from_attention(a)
