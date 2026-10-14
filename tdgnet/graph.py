"""Temporal graph construction from keyframe attention matrices."""
from __future__ import annotations

from typing import List, Sequence, Tuple

import torch
from torch import Tensor

from tdgnet.keyframes import default_keyframe_indices


def zero_padding_attention(attention: Tensor, node_mask: Tensor) -> Tensor:
    """
    Zero attention rows and columns for padding nodes (``node_mask`` False).

    Padding tokens then cannot send or receive messages after thresholding.
    """
    if attention.dim() != 2 or attention.size(0) != attention.size(1):
        raise ValueError(f"attention must be square [N, N], got {tuple(attention.shape)}")
    if node_mask.shape != (attention.size(0),):
        raise ValueError("node_mask must have shape [N]")
    mask = node_mask.to(device=attention.device, dtype=torch.bool)
    a = attention.clone()
    pad = ~mask
    a = a.masked_fill(pad.unsqueeze(1), 0)
    a = a.masked_fill(pad.unsqueeze(0), 0)
    return a


def build_graph_from_attention(
    attention: Tensor,
    tau: float = 0.05,
    *,
    include_self_loops: bool = True,
) -> Tuple[Tensor, Tensor]:
    """
    Build a directed message graph from one head-averaged attention matrix.

    Paper Eq. 2: edge ``j -> i`` exists iff ``A[i, j] > tau`` (token ``i`` attends
    to ``j``; the message flows from ``j`` into ``i``). Edge feature
    ``e_ji = A[i, j]``. Self-loops are retained by default, as in Eq. 2.

    Args:
        attention: ``[N, N]`` with rows = queries, cols = keys.
        tau: attention threshold.

    Returns:
        edge_index: ``[2, E]`` where row 0 = source ``j``, row 1 = target ``i``.
        edge_attr: ``[E, 1]`` edge features.
    """
    if attention.dim() != 2 or attention.size(0) != attention.size(1):
        raise ValueError(f"attention must be square [N, N], got {tuple(attention.shape)}")

    if not 0 < tau < 1:
        raise ValueError("tau must lie strictly between 0 and 1")
    if not attention.is_floating_point() or not torch.isfinite(attention).all():
        raise ValueError("attention must contain finite floating-point scores")
    if (attention < 0).any() or (attention > 1).any():
        raise ValueError("attention scores must lie in [0, 1]")
    a = attention.detach()
    n = a.size(0)
    mask = a > tau
    if not include_self_loops:
        mask = mask & ~torch.eye(n, dtype=torch.bool, device=a.device)
    # nonzero yields (query i, key j) pairs where A[i, j] > tau.
    qi, kj = mask.nonzero(as_tuple=True)
    # Message edge j -> i : source=j (key), target=i (query).
    edge_index = torch.stack([kj, qi], dim=0).long()
    if edge_index.numel() == 0:
        edge_attr = torch.empty((0, 1), dtype=a.dtype, device=a.device)
    else:
        edge_attr = a[qi, kj].unsqueeze(-1).to(dtype=a.dtype)
    return edge_index, edge_attr


def build_temporal_graph(
    attn_keyframes: Sequence[Tensor],
    tau: float = 0.05,
    *,
    include_self_loops: bool = True,
) -> List[Tuple[Tensor, Tensor]]:
    """
    Build one graph per keyframe.

    Args:
        attn_keyframes: sequence of ``K`` tensors, each ``[N, N]``.
        tau: attention threshold (default 0.05).

    Returns:
        List of ``(edge_index, edge_attr)`` pairs, length ``K``.
    """
    if not attn_keyframes:
        raise ValueError("attn_keyframes must be non-empty")
    return [build_graph_from_attention(a, tau=tau, include_self_loops=include_self_loops)
            for a in attn_keyframes]


__all__ = [
    "zero_padding_attention",
    "default_keyframe_indices",
    "build_graph_from_attention",
    "build_temporal_graph",
]
