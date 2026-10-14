"""Model checkpoints with complete architecture settings and trained-head state."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch

from tdgnet.model import TDGNet


def save_checkpoint(model: TDGNet, path: Union[str, Path], *,
                    metadata: Optional[Dict[str, Any]] = None) -> None:
    info = dict(metadata or {})
    json.dumps(info, allow_nan=False)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"format_version": 1, "model_config": model.get_config(),
                "model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                "metadata": info}, path)


def load_checkpoint(path: Union[str, Path], *, device: Union[str, torch.device] = "cpu") -> TDGNet:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("format_version") != 1:
        raise ValueError("unsupported checkpoint format")
    model = TDGNet(**payload["model_config"])
    model.load_state_dict(payload["model"], strict=True)
    return model.to(device).eval()


__all__ = ["save_checkpoint", "load_checkpoint"]
