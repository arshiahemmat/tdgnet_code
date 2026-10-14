"""TDGNet detector: temporal message passing with GRU memory and keyframe attention."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from tdgnet.graph import build_temporal_graph, zero_padding_attention


class MLP(nn.Module):
    def __init__(self, dims: Sequence[int], dropout: float = 0.0):
        super().__init__()
        if len(dims) < 2:
            raise ValueError("dims must have at least input and output sizes")
        layers: List[nn.Module] = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.ReLU())
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


@dataclass
class TDGNetOutput:
    """Model outputs for response-level and optional token-level heads."""

    response_logit: Tensor  # [B]
    token_logits: Tensor  # [B, N]
    alpha: Tensor  # [B, K, N] temporal attention (mean over heads)
    node_embeddings: Tensor  # [B, N, H] trajectory embeddings z after keyframe aggregation


class TDGNet(nn.Module):
    """
    Temporal Dynamic Graph Network (paper Sec. 3.3 / Algorithm 1).

    The default single message layer and single temporal query implement the
    recurrence and readout in Algorithm 1. Mean-pooling includes every valid
    graph node. Response-only pooling, stacked residual message layers and
    multiple averaged temporal queries are explicit options.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 128,
        msg_hidden_dim: Optional[int] = None,
        dropout: float = 0.0,
        tau: float = 0.05,
        edge_dim: int = 1,
        num_mp_layers: int = 1,
        num_temporal_heads: int = 1,
        disable_edges: bool = False,
        include_self_loops: bool = True,
        pooling: str = "all",
    ):
        super().__init__()
        if in_dim <= 0 or hidden_dim <= 0:
            raise ValueError("in_dim and hidden_dim must be positive")
        if num_mp_layers < 1:
            raise ValueError("num_mp_layers must be >= 1")
        if num_temporal_heads < 1:
            raise ValueError("num_temporal_heads must be >= 1")
        if not 0 < tau < 1 or not 0 <= dropout < 1:
            raise ValueError("require 0 < tau < 1 and 0 <= dropout < 1")
        if edge_dim < 1 or (msg_hidden_dim is not None and msg_hidden_dim < 1):
            raise ValueError("edge and message dimensions must be positive")
        if pooling not in ("all", "response"):
            raise ValueError("pooling must be 'all' or 'response'")
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.tau = tau
        self.edge_dim = edge_dim
        self.num_mp_layers = num_mp_layers
        self.num_temporal_heads = num_temporal_heads
        self.disable_edges = disable_edges
        self.include_self_loops = include_self_loops
        self.pooling = pooling
        msg_hidden = msg_hidden_dim if msg_hidden_dim is not None else hidden_dim
        self.msg_hidden_dim = msg_hidden

        self.node_proj = nn.Linear(in_dim, hidden_dim)
        self.msg_mlps = nn.ModuleList(
            [
                MLP([2 * hidden_dim + edge_dim, msg_hidden, hidden_dim], dropout=dropout)
                for _ in range(num_mp_layers)
            ]
        )
        self.gru = nn.GRUCell(input_size=hidden_dim, hidden_size=hidden_dim)
        self.z_proj = nn.Linear(hidden_dim, hidden_dim)
        # Shape [H] when one head (exact Alg. 1); [M, H] when M > 1.
        if num_temporal_heads == 1:
            self.q = nn.Parameter(torch.randn(hidden_dim) * 0.02)
        else:
            self.q = nn.Parameter(torch.randn(num_temporal_heads, hidden_dim) * 0.02)
        self.pred_mlp = MLP([hidden_dim, hidden_dim, 1], dropout=dropout)
        self.token_mlp = MLP([hidden_dim, hidden_dim, 1], dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("_trained_heads", torch.zeros(2, dtype=torch.bool))

    @property
    def msg_mlp(self) -> nn.Module:
        return self.msg_mlps[0]

    @property
    def trained_heads(self) -> List[str]:
        return [name for i, name in enumerate(("response", "token"))
                if bool(self._trained_heads[i].item())]

    def mark_trained(self, *, response: bool = False, token: bool = False) -> None:
        """Record which heads have received an optimization update."""
        if response:
            self._trained_heads[0] = True
        if token:
            self._trained_heads[1] = True

    def get_config(self) -> dict:
        """Constructor options needed to reload this architecture."""
        return dict(in_dim=self.in_dim, hidden_dim=self.hidden_dim,
                    msg_hidden_dim=self.msg_hidden_dim, dropout=self.dropout.p,
                    tau=self.tau, edge_dim=self.edge_dim,
                    num_mp_layers=self.num_mp_layers,
                    num_temporal_heads=self.num_temporal_heads,
                    disable_edges=self.disable_edges,
                    include_self_loops=self.include_self_loops, pooling=self.pooling)

    def _mean_messages(
        self,
        h: Tensor,
        edge_index: Tensor,
        edge_attr: Tensor,
        msg_mlp: nn.Module,
    ) -> Tensor:
        """m_i = mean_j MLP([h_j; h_i; e_ji]); 0 if no in-neighbours."""
        n, hdim = h.shape
        if self.disable_edges or edge_index.numel() == 0:
            return h.new_zeros((n, hdim))

        src, dst = edge_index[0], edge_index[1]
        if edge_attr.dim() == 1:
            edge_attr = edge_attr.unsqueeze(-1)
        msg_in = torch.cat([h[src], h[dst], edge_attr], dim=-1)
        msgs = msg_mlp(msg_in)

        agg = h.new_zeros((n, hdim))
        agg.index_add_(0, dst, msgs)
        counts = h.new_zeros((n, 1))
        counts.index_add_(0, dst, torch.ones(msgs.size(0), 1, device=h.device, dtype=h.dtype))
        return agg / counts.clamp(min=1.0)

    def _message_pass_stack(
        self,
        h: Tensor,
        edge_index: Tensor,
        edge_attr: Tensor,
    ) -> Tensor:
        """
        Apply ``num_mp_layers`` mean-message layers.

        Layer 0 matches Algorithm 1 (message only, no residual into the GRU input).
        Layers 1..L-1 update node states with a residual before the next message:
        ``h <- h + m``, then ``m`` from the last layer is returned for the GRU.
        """
        m = self._mean_messages(h, edge_index, edge_attr, self.msg_mlps[0])
        for layer in range(1, self.num_mp_layers):
            h = h + m
            m = self._mean_messages(h, edge_index, edge_attr, self.msg_mlps[layer])
        return m

    def _temporal_pool(
        self,
        S: Tensor,
        Z: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """
        Multi-head temporal attention over keyframes.

        Scores from memory ``S``; values from ``Z``. Returns ``z [N,H]`` and
        mean-over-heads ``alpha [K,N]``.
        """
        if self.num_temporal_heads == 1:
            q = self.q  # [H]
            scores = torch.einsum("knh,h->kn", S, q)
            alpha = torch.softmax(scores, dim=0)
            z = (alpha.unsqueeze(-1) * Z).sum(dim=0)
            return z, alpha

        # q: [M, H]; scores: [M, K, N]
        scores = torch.einsum("knh,mh->mkn", S, self.q)
        alpha_m = torch.softmax(scores, dim=1)  # over keyframes
        z_m = (alpha_m.unsqueeze(-1) * Z.unsqueeze(0)).sum(dim=1)  # [M, N, H]
        z = z_m.mean(dim=0)
        alpha = alpha_m.mean(dim=0)  # [K, N]
        return z, alpha

    def _encode_sample(
        self,
        features: Tensor,
        graphs: List[Tuple[Tensor, Tensor]],
        node_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        k, n, _ = features.shape
        if len(graphs) != k:
            raise ValueError(f"expected {k} graphs, got {len(graphs)}")

        device = features.device
        s = features.new_zeros((n, self.hidden_dim), dtype=self.gru.weight_hh.dtype)
        s_stack: List[Tensor] = []
        z_stack: List[Tensor] = []

        for t in range(k):
            h = self.node_proj(features[t].to(dtype=self.node_proj.weight.dtype))
            h = self.dropout(h)
            edge_index, edge_attr = graphs[t]
            edge_index = edge_index.to(device)
            edge_attr = edge_attr.to(device=device, dtype=h.dtype)
            m = self._message_pass_stack(h, edge_index, edge_attr)
            s = self.gru(m, s)
            if node_mask is not None:
                s = s * node_mask.unsqueeze(-1).to(dtype=s.dtype)
            z_t = self.z_proj(s)
            s_stack.append(s)
            z_stack.append(z_t)

        S = torch.stack(s_stack, dim=0)
        Z = torch.stack(z_stack, dim=0)
        z, alpha = self._temporal_pool(S, Z)
        if node_mask is not None:
            z = z * node_mask[:, None].to(z.dtype)
            alpha = alpha * node_mask[None, :].to(alpha.dtype)
        return z, alpha, S

    def forward(
        self,
        features: Tensor,
        attentions: Optional[Tensor] = None,
        edge_indices: Optional[List[List[Tensor]]] = None,
        edge_attrs: Optional[List[List[Tensor]]] = None,
        response_mask: Optional[Tensor] = None,
        node_mask: Optional[Tensor] = None,
        tau: Optional[float] = None,
    ) -> TDGNetOutput:
        if features.dim() != 4:
            raise ValueError(f"features must be [B, K, N, in_dim], got {tuple(features.shape)}")
        b, k, n, in_dim = features.shape
        if min(b, k, n) < 1:
            raise ValueError("batch, keyframe and node dimensions must be non-empty")
        if in_dim != self.in_dim:
            raise ValueError(f"expected in_dim={self.in_dim}, got {in_dim}")

        if node_mask is None:
            node_mask = torch.ones(b, n, dtype=torch.bool, device=features.device)
        if response_mask is None:
            response_mask = node_mask.clone()
        for name, mask in (("node_mask", node_mask), ("response_mask", response_mask)):
            if mask.shape != (b, n) or mask.dtype != torch.bool or mask.device != features.device:
                raise ValueError(f"{name} must be bool [B,N] on the feature device")
        if attentions is not None and attentions.shape != (b, k, n, n):
            raise ValueError("attentions must be [B,K,N,N]")
        if (edge_indices is None) != (edge_attrs is None):
            raise ValueError("edge_indices and edge_attrs must be supplied together")
        if edge_indices is not None and (len(edge_indices) != b or len(edge_attrs) != b):
            raise ValueError("precomputed graphs must have one entry per sample")

        thr = self.tau if tau is None else tau
        response_logits: List[Tensor] = []
        token_logits: List[Tensor] = []
        alphas: List[Tensor] = []
        node_embeddings: List[Tensor] = []

        for bi in range(b):
            if edge_indices is not None and edge_attrs is not None:
                if len(edge_indices[bi]) != k or len(edge_attrs[bi]) != k:
                    raise ValueError("one graph is required per keyframe")
                graphs = list(zip(edge_indices[bi], edge_attrs[bi]))
                cleaned = []
                for index, attr in graphs:
                    if index.dim() != 2 or index.size(0) != 2 or index.dtype != torch.long:
                        raise ValueError("edge_index must be int64 [2,E]")
                    if attr.shape != (index.size(1), self.edge_dim):
                        raise ValueError("edge_attr must be [E,edge_dim]")
                    index, attr = index.to(features.device), attr.to(features.device)
                    if index.numel() and ((index < 0).any() or (index >= n).any()):
                        raise ValueError("edge index out of range")
                    valid = node_mask[bi, index[0]] & node_mask[bi, index[1]]
                    if not self.include_self_loops:
                        valid = valid & (index[0] != index[1])
                    cleaned.append((index[:, valid], attr[valid]))
                graphs = cleaned
            else:
                if attentions is None:
                    raise ValueError("Provide attentions or (edge_indices, edge_attrs)")
                masked = [
                    zero_padding_attention(attentions[bi, t], node_mask[bi])
                    for t in range(k)
                ]
                if self.edge_dim != 1:
                    raise ValueError("attention-derived edges have edge_dim=1")
                graphs = build_temporal_graph(masked, tau=thr,
                                              include_self_loops=self.include_self_loops)
            if self.disable_edges:
                graphs = [
                    (
                        torch.empty((2, 0), dtype=torch.long, device=features.device),
                        torch.empty((0, self.edge_dim), dtype=features.dtype, device=features.device),
                    )
                    for _ in graphs
                ]
            z, alpha, _ = self._encode_sample(
                features[bi],
                graphs,
                node_mask=node_mask[bi],
            )
            token_mask = response_mask[bi] & node_mask[bi]
            pool_mask = node_mask[bi] if self.pooling == "all" else token_mask
            if pool_mask.any():
                g = z[pool_mask].mean(dim=0)
            else:
                raise ValueError("each sample needs at least one node for pooling")
            response_logits.append(self.pred_mlp(g).squeeze(-1))
            tok = self.token_mlp(z).squeeze(-1)
            tok = tok.masked_fill(~token_mask, 0.0)
            token_logits.append(tok)
            alphas.append(alpha)
            node_embeddings.append(z)

        return TDGNetOutput(
            response_logit=torch.stack(response_logits, dim=0),
            token_logits=torch.stack(token_logits, dim=0),
            alpha=torch.stack(alphas, dim=0),
            node_embeddings=torch.stack(node_embeddings, dim=0),
        )


__all__ = ["TDGNet", "TDGNetOutput", "MLP"]
