"""YAML config loading with recursive merges."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Union


def _require_yaml():
    try:
        import yaml
    except ImportError as exc:
        raise ImportError(
            "PyYAML is required for configs. Install with: pip install pyyaml"
        ) from exc
    return yaml


def load_yaml(path: Union[str, Path]) -> Dict[str, Any]:
    yaml = _require_yaml()
    with Path(path).open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"config root must be a mapping: {path}")
    return data


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_configs(paths: List[Union[str, Path]]) -> Dict[str, Any]:
    cfg: Dict[str, Any] = {}
    for p in paths:
        cfg = deep_merge(cfg, load_yaml(p))
    return cfg


__all__ = ["load_yaml", "deep_merge", "load_configs"]
