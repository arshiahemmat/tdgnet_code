"""Trace sample containers, I/O helpers, and class balancing."""
from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import torch
from torch import Tensor

from tdgnet.graph import default_keyframe_indices


@dataclass
class TraceSample:
    """
    One diffusion trace with keyframe attentions and node features.

    Attributes:
        attentions: list of ``K`` tensors ``[N, N]`` (head-averaged final-layer attn).
        features: list of ``K`` tensors ``[N, in_dim]`` (final-layer hidden states).
        response_mask: ``[N]`` bool; True for response tokens (prompt = False).
        label: optional response-level label (1 = hallucination).
        token_labels: optional labels ``[N]`` or response-only labels ``[R]``;
            NaN marks positions without supervision.
        metadata: sampler / backbone fields (steps, gen_length, ...).
    """

    attentions: List[Tensor]
    features: List[Tensor]
    response_mask: Tensor
    label: Optional[float] = None
    token_labels: Optional[Tensor] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if len(self.attentions) != len(self.features):
            raise ValueError("attentions and features must have the same number of keyframes")
        if not self.attentions:
            raise ValueError("TraceSample must contain at least one keyframe")
        n = self.attentions[0].shape[0]
        if n < 1 or self.features[0].dim() != 2 or self.features[0].shape[1] < 1:
            raise ValueError("features must be non-empty [N,D]")
        d = self.features[0].shape[1]
        for a, x in zip(self.attentions, self.features):
            if a.shape != (n, n):
                raise ValueError(f"attention shape {tuple(a.shape)} != ({n}, {n})")
            if x.shape != (n, d):
                raise ValueError("all features must have the same [N,D] shape")
            if not a.is_floating_point() or not x.is_floating_point():
                raise ValueError("attentions and features must be floating-point")
            if not torch.isfinite(a).all() or not torch.isfinite(x).all():
                raise ValueError("attentions and features must be finite")
            if (a < 0).any() or (a > 1).any():
                raise ValueError("attention scores must lie in [0,1]")
        if self.response_mask.shape != (n,) or self.response_mask.dtype != torch.bool:
            raise ValueError("response_mask must be bool [N]")
        if self.label is not None and float(self.label) not in (0.0, 1.0):
            raise ValueError("label must be 0, 1 or None")
        if self.token_labels is not None:
            r = int(self.response_mask.sum().item())
            if self.token_labels.dim() != 1 or self.token_labels.numel() not in (n, r):
                raise ValueError("token_labels must be [N] or response-only [R]")
            t = self.token_labels.float()
            if ((~torch.isnan(t)) & (t != 0) & (t != 1)).any():
                raise ValueError("token labels must be 0, 1 or NaN")
        json.dumps(self.metadata, allow_nan=False)

    @property
    def num_nodes(self) -> int:
        return int(self.attentions[0].shape[0])

    @property
    def num_keyframes(self) -> int:
        return len(self.attentions)

    @property
    def in_dim(self) -> int:
        return int(self.features[0].shape[-1])

    def to(self, device: Union[str, torch.device]) -> "TraceSample":
        return TraceSample(
            attentions=[a.to(device) for a in self.attentions],
            features=[f.to(device) for f in self.features],
            response_mask=self.response_mask.to(device),
            label=self.label,
            token_labels=None if self.token_labels is None else self.token_labels.to(device),
            metadata=dict(self.metadata),
        )


def select_trace_keyframes(
    attentions: Sequence[Tensor], features: Sequence[Tensor],
    response_mask: Tensor, *, label: Optional[float] = None,
    token_labels: Optional[Tensor] = None, metadata: Optional[Dict[str, Any]] = None,
) -> TraceSample:
    """Select the paper's keyframes from a full trajectory of T+1 states."""
    if len(attentions) != len(features) or len(attentions) < 2:
        raise ValueError("supply matching T+1 attention and feature states, including both endpoints")
    steps = len(attentions) - 1
    indices = default_keyframe_indices(steps)
    return TraceSample([attentions[i] for i in indices], [features[i] for i in indices],
                       response_mask, label, token_labels,
                       {**(metadata or {}), "steps": steps, "keyframe_indices": indices,
                        "diffusion_times": [steps - i for i in indices]})


def save_trace(sample: TraceSample, path: Union[str, Path]) -> None:
    """Save a TraceSample to a ``.pt`` file (tensors preserved)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "attentions": sample.attentions,
        "features": sample.features,
        "response_mask": sample.response_mask,
        "label": sample.label,
        "token_labels": sample.token_labels,
        "metadata": sample.metadata,
    }
    torch.save(payload, path)


def load_trace(path: Union[str, Path], map_location: str = "cpu") -> TraceSample:
    """Load a TraceSample from a ``.pt`` file."""
    payload = torch.load(path, map_location=map_location, weights_only=True)
    return TraceSample(
        attentions=list(payload["attentions"]),
        features=list(payload["features"]),
        response_mask=payload["response_mask"],
        label=payload.get("label"),
        token_labels=payload.get("token_labels"),
        metadata=dict(payload.get("metadata") or {}),
    )


def save_traces_meta(samples: Sequence[TraceSample], path: Union[str, Path]) -> None:
    """Write JSON metadata (labels only) for a list of traces — no tensors."""
    path = Path(path)
    rows = []
    for i, s in enumerate(samples):
        rows.append(
            {
                "index": i,
                "label": s.label,
                "num_nodes": s.num_nodes,
                "num_keyframes": s.num_keyframes,
                "in_dim": s.in_dim,
                "metadata": s.metadata,
            }
        )
    path.write_text(json.dumps(rows, indent=2))


def print_label_balance(samples: Sequence[TraceSample], split_name: str = "split") -> Dict[str, float]:
    """Print and return label counts / fractions for a split."""
    labels = [s.label for s in samples if s.label is not None]
    n = len(labels)
    if n == 0:
        print(f"[{split_name}] no labels present ({len(samples)} samples)")
        return {"n": 0, "pos": 0, "neg": 0, "pos_frac": float("nan")}
    pos = sum(1 for y in labels if float(y) >= 0.5)
    neg = n - pos
    frac = pos / n
    print(f"[{split_name}] n={n}  hallucinated(1)={pos}  factual(0)={neg}  pos_frac={frac:.4f}")
    return {"n": n, "pos": pos, "neg": neg, "pos_frac": frac}


def undersample_to_balance(
    samples: Sequence[TraceSample],
    seed: int = 42,
) -> List[TraceSample]:
    """
    Undersample the majority class to 50/50.

    Samples without a label are dropped. If either class is empty, returns a
    shuffled copy of the labelled samples unchanged.
    """
    rng = random.Random(seed)
    pos = [s for s in samples if s.label is not None and float(s.label) >= 0.5]
    neg = [s for s in samples if s.label is not None and float(s.label) < 0.5]
    if not pos or not neg:
        out = list(pos or neg)
        rng.shuffle(out)
        return out
    m = min(len(pos), len(neg))
    pos_s = rng.sample(pos, m)
    neg_s = rng.sample(neg, m)
    out = pos_s + neg_s
    rng.shuffle(out)
    return out


def collate_traces(
    samples: Sequence[TraceSample],
    device: Union[str, torch.device] = "cpu",
) -> Dict[str, Tensor]:
    """
    Pad a batch of traces to a common ``N`` (and require a common ``K``, ``in_dim``).

    Returns tensors:
        features ``[B, K, N, in_dim]``, attentions ``[B, K, N, N]``,
        response_mask ``[B, N]``, node_mask ``[B, N]``, labels ``[B]`` and
        token_labels ``[B,N]`` (NaN at missing, prompt and padding positions).
    """
    if not samples:
        raise ValueError("empty batch")
    k = samples[0].num_keyframes
    in_dim = samples[0].in_dim
    for s in samples:
        if s.num_keyframes != k or s.in_dim != in_dim:
            raise ValueError("all samples in a batch must share K and in_dim")
    max_n = max(s.num_nodes for s in samples)
    b = len(samples)

    features = torch.zeros(b, k, max_n, in_dim, device=device)
    attentions = torch.zeros(b, k, max_n, max_n, device=device)
    response_mask = torch.zeros(b, max_n, dtype=torch.bool, device=device)
    node_mask = torch.zeros(b, max_n, dtype=torch.bool, device=device)
    labels = torch.full((b,), float("nan"), device=device)
    token_labels = torch.full((b, max_n), float("nan"), device=device)

    for i, s in enumerate(samples):
        n = s.num_nodes
        node_mask[i, :n] = True
        response_mask[i, :n] = s.response_mask.to(device=device, dtype=torch.bool)
        for t in range(k):
            features[i, t, :n] = s.features[t].to(device)
            attentions[i, t, :n, :n] = s.attentions[t].to(device)
        if s.label is not None:
            labels[i] = float(s.label)
        if s.token_labels is not None:
            target = s.token_labels.to(device=device, dtype=torch.float32)
            if target.numel() == n:
                token_labels[i, :n] = target
            else:
                response_indices = response_mask[i, :n].nonzero(as_tuple=True)[0]
                token_labels[i, response_indices] = target
    token_labels = token_labels.masked_fill(~(response_mask & node_mask), float("nan"))

    return {
        "features": features,
        "attentions": attentions,
        "response_mask": response_mask,
        "node_mask": node_mask,
        "labels": labels,
        "token_labels": token_labels,
    }


__all__ = [
    "TraceSample",
    "select_trace_keyframes",
    "save_trace",
    "load_trace",
    "save_traces_meta",
    "print_label_balance",
    "undersample_to_balance",
    "collate_traces",
]
