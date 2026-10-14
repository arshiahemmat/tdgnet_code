"""Sparse attention / hidden-state shard I/O for extracted traces."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch
from torch import Tensor

from tdgnet.data import TraceSample
from tdgnet.graph import build_graph_from_attention

DEFAULT_TAU_MIN = 0.03


def sparsify_attention(attention: Tensor, tau_min: float = DEFAULT_TAU_MIN) -> Dict[str, Tensor]:
    """
    Store entries with value > tau_min, including self-loops.

    Values retain their original dtype to avoid changing threshold decisions.
    Graph thresholds must be greater than or equal to tau_min.
    """
    if attention.dim() != 2 or attention.size(0) != attention.size(1):
        raise ValueError("attention must be square")
    if not 0 <= tau_min < 1:
        raise ValueError("tau_min must lie in [0,1)")
    n = attention.size(0)
    mask = attention > tau_min
    qi, kj = mask.nonzero(as_tuple=True)
    values = attention[qi, kj].detach().cpu()
    indices = torch.stack([qi, kj], dim=1).to(dtype=torch.int32, device="cpu")
    return {"indices": indices, "values": values, "n": torch.tensor(n, dtype=torch.int32)}


def densify_attention(sparse: Dict[str, Tensor]) -> Tensor:
    n = int(sparse["n"].item()) if torch.is_tensor(sparse["n"]) else int(sparse["n"])
    a = torch.zeros(n, n, dtype=sparse["values"].dtype)
    if sparse["indices"].numel() == 0:
        return a
    idx = sparse["indices"].long()
    a[idx[:, 0], idx[:, 1]] = sparse["values"]
    return a


def estimate_bytes_per_sample(
    n: int,
    k: int,
    in_dim: int,
    *,
    attn_density: float = 0.05,
    tau_min: float = DEFAULT_TAU_MIN,
) -> Dict[str, float]:
    """Rough disk usage estimate for float32 features and attention values."""
    hidden_bytes = k * n * in_dim * 4
    # indices int32 (8 bytes/nnz) + values float32 (4)
    nnz = attn_density * n * n * k
    attn_bytes = nnz * (8 + 4)
    misc = 4096
    total = hidden_bytes + attn_bytes + misc
    return {
        "hidden_bytes": hidden_bytes,
        "attn_bytes": attn_bytes,
        "total_bytes": total,
        "total_mb": total / (1024**2),
        "tau_min": tau_min,
        "assumed_attn_density": attn_density,
    }


def pack_trace(
    sample: TraceSample,
    *,
    tau_min: float = DEFAULT_TAU_MIN,
) -> Dict[str, Any]:
    sparse_attns = [sparsify_attention(a, tau_min=tau_min) for a in sample.attentions]
    stored_min = float(sample.metadata.get("attention_storage_tau_min", 0.0))
    if tau_min < stored_min:
        raise ValueError("cannot restore attention entries below an existing storage threshold")
    features = [f.detach().cpu() for f in sample.features]
    return {
        "sparse_attentions": sparse_attns,
        "features": features,
        "response_mask": sample.response_mask.cpu().bool(),
        "label": sample.label,
        "token_labels": None if sample.token_labels is None else sample.token_labels.cpu(),
        "metadata": dict(sample.metadata),
        "tau_min": tau_min,
    }


def unpack_trace(payload: Dict[str, Any]) -> TraceSample:
    attns = [densify_attention(s) for s in payload["sparse_attentions"]]
    feats = list(payload["features"])
    return TraceSample(
        attentions=attns,
        features=feats,
        response_mask=payload["response_mask"].bool(),
        label=payload.get("label"),
        token_labels=payload.get("token_labels"),
        metadata={**(payload.get("metadata") or {}),
                  "attention_storage_tau_min": float(payload["tau_min"])},
    )


def save_shard(samples: Sequence[TraceSample], path: Union[str, Path], *, tau_min: float = DEFAULT_TAU_MIN) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    packed = [pack_trace(s, tau_min=tau_min) for s in samples]
    torch.save({"samples": packed, "tau_min": tau_min}, path)


def load_shard(path: Union[str, Path]) -> List[TraceSample]:
    blob = torch.load(path, map_location="cpu", weights_only=True)
    return [unpack_trace(p) for p in blob["samples"]]


def load_trace_dir(path: Union[str, Path]) -> List[TraceSample]:
    """Load all ``shard_*.pt`` files from a directory (sorted), else loose ``*.pt`` traces."""
    from tdgnet.data import load_trace

    path = Path(path)
    if not path.is_dir():
        raise ValueError(f"trace directory does not exist: {path}")
    shards = sorted(path.glob("shard_*.pt"))
    if shards:
        out: List[TraceSample] = []
        for p in shards:
            out.extend(load_shard(p))
        return out
    return [load_trace(p) for p in sorted(path.glob("*.pt"))]


def shard_path(out_dir: Path, shard_id: int) -> Path:
    return Path(out_dir) / f"shard_{shard_id:05d}.pt"


def graphs_from_sparse_sample(
    sample: TraceSample,
    tau: float = 0.05,
) -> List[Tuple[Tensor, Tensor]]:
    """Build graphs at an arbitrary tau >= tau_min without re-extraction."""
    minimum = float(sample.metadata.get("attention_storage_tau_min", 0.0))
    if tau < minimum:
        raise ValueError(f"tau must be >= stored threshold {minimum}")
    return [build_graph_from_attention(a, tau=tau) for a in sample.attentions]


def print_storage_estimate(n: int, k: int, in_dim: int) -> None:
    est = estimate_bytes_per_sample(n, k, in_dim)
    print(
        f"[storage] per-sample estimate (N={n}, K={k}, d={in_dim}): "
        f"~{est['total_mb']:.2f} MB "
        f"(hiddens={est['hidden_bytes']/1e6:.2f} MB, "
        f"sparse_attn≈{est['attn_bytes']/1e6:.2f} MB at density={est['assumed_attn_density']})"
    )


__all__ = [
    "DEFAULT_TAU_MIN",
    "sparsify_attention",
    "densify_attention",
    "pack_trace",
    "unpack_trace",
    "save_shard",
    "load_shard",
    "load_trace_dir",
    "shard_path",
    "estimate_bytes_per_sample",
    "print_storage_estimate",
    "graphs_from_sparse_sample",
]
