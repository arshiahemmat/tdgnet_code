"""Compare the default detector to independent NumPy equations."""
import unittest

import numpy as np
import torch

from tdgnet import TDGNet


def linear(x, module):
    return x @ module.weight.detach().cpu().numpy().T + module.bias.detach().cpu().numpy()


def mlp(x, module):
    return linear(np.maximum(linear(x, module.net[0]), 0), module.net[2])


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def numpy_reference(model, features, attentions):
    k, n, _ = features.shape
    memory = np.zeros((n, model.hidden_dim))
    memories, projected = [], []
    for step in range(k):
        h = linear(features[step], model.node_proj)
        messages = np.zeros_like(memory)
        for i in range(n):
            incoming = []
            for j in range(n):
                if attentions[step, i, j] > model.tau:
                    incoming.append(mlp(np.concatenate([h[j], h[i], [attentions[step, i, j]]]),
                                        model.msg_mlps[0]))
            if incoming:
                messages[i] = np.mean(incoming, axis=0)
        cell = model.gru
        gi = messages @ cell.weight_ih.detach().numpy().T + cell.bias_ih.detach().numpy()
        gh = memory @ cell.weight_hh.detach().numpy().T + cell.bias_hh.detach().numpy()
        ir, iz, inn = np.split(gi, 3, axis=1)
        hr, hz, hn = np.split(gh, 3, axis=1)
        reset, update = sigmoid(ir + hr), sigmoid(iz + hz)
        candidate = np.tanh(inn + reset * hn)
        memory = (1 - update) * candidate + update * memory
        memories.append(memory.copy())
        projected.append(linear(memory, model.z_proj))
    scores = np.stack(memories) @ model.q.detach().numpy()
    weights = np.exp(scores - scores.max(axis=0, keepdims=True))
    alpha = weights / weights.sum(axis=0, keepdims=True)
    z = (alpha[..., None] * np.stack(projected)).sum(axis=0)
    return z, alpha, mlp(z.mean(axis=0), model.pred_mlp).item(), mlp(z, model.token_mlp).ravel()


class EquationTests(unittest.TestCase):
    def test_default_forward_matches_independent_equations(self):
        torch.manual_seed(17)
        model = TDGNet(2, hidden_dim=2, tau=0.15).eval()
        x = torch.tensor([[[0.1, 0.7], [0.3, -0.2], [0.8, 0.6]],
                          [[0.4, 0.2], [-0.5, 0.9], [0.2, 0.1]]])
        a = torch.tensor([[[0.4, 0.6, 0.0], [0.2, 0.2, 0.6], [0.1, 0.8, 0.1]],
                          [[0.2, 0.3, 0.5], [0.6, 0.2, 0.2], [0.7, 0.2, 0.1]]])
        mask = torch.tensor([[False, True, True]])
        out = model(x[None], a[None], response_mask=mask)
        z, alpha, response, tokens = numpy_reference(model, x.numpy(), a.numpy())
        np.testing.assert_allclose(out.node_embeddings.detach().numpy()[0], z, atol=1e-6)
        np.testing.assert_allclose(out.alpha.detach().numpy()[0], alpha, atol=1e-6)
        self.assertAlmostEqual(out.response_logit.item(), response, places=6)
        tokens[0] = 0
        np.testing.assert_allclose(out.token_logits.detach().numpy()[0], tokens, atol=1e-6)
        out.response_logit.sum().backward()
        self.assertTrue(any(p.grad is not None for p in model.parameters()))
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))

    def test_temporal_softmax_is_over_time(self):
        model = TDGNet(2, hidden_dim=2)
        with torch.no_grad():
            model.q.copy_(torch.tensor([1.0, 0.0]))
        s = torch.tensor([[[0.0, 0.0], [1.0, 0.0]], [[2.0, 0.0], [-1.0, 0.0]]])
        values = torch.tensor([[[1.0, 3.0], [2.0, 4.0]], [[5.0, 7.0], [6.0, 8.0]]])
        z, alpha = model._temporal_pool(s, values)
        expected = torch.tensor([[1/(1+np.exp(2)), 1/(1+np.exp(-2))],
                                 [1/(1+np.exp(-2)), 1/(1+np.exp(2))]], dtype=torch.float32)
        torch.testing.assert_close(alpha, expected)
        torch.testing.assert_close(z, (expected[..., None] * values).sum(0))

    def test_padding_and_response_pooling_are_explicit(self):
        torch.manual_seed(4)
        model = TDGNet(2, hidden_dim=4, pooling='response').eval()
        x, a = torch.randn(1, 2, 3, 2), torch.eye(3).repeat(1, 2, 1, 1)
        response = torch.tensor([[False, True, False]])
        nodes = torch.tensor([[True, True, False]])
        out = model(x, a, response_mask=response, node_mask=nodes)
        self.assertEqual(out.node_embeddings[0, 2].abs().sum().item(), 0)
        self.assertEqual(out.alpha[0, :, 2].abs().sum().item(), 0)
        torch.testing.assert_close(out.response_logit[0], model.pred_mlp(out.node_embeddings[0, 1]).squeeze(-1))

    def test_precomputed_padding_edges_are_removed(self):
        torch.manual_seed(9)
        model = TDGNet(2, hidden_dim=4).eval()
        x, a = torch.randn(1, 1, 3, 2), torch.eye(3)[None, None]
        nodes = torch.tensor([[True, True, False]])
        dense = model(x, a, node_mask=nodes)
        indices = [[torch.tensor([[0, 1, 2, 2], [0, 1, 0, 2]])]]
        attributes = [[torch.ones(4, 1)]]
        sparse = model(x, edge_indices=indices, edge_attrs=attributes, node_mask=nodes)
        torch.testing.assert_close(dense.response_logit, sparse.response_logit)
