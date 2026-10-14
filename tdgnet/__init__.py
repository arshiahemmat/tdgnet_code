"""TDGNet: temporal dynamic graphs for hallucination detection in diffusion LMs."""
from importlib import import_module

_EXPORTS = {
    "TDGNet": ("tdgnet.model", "TDGNet"),
    "TDGNetOutput": ("tdgnet.model", "TDGNetOutput"),
    "TraceSample": ("tdgnet.data", "TraceSample"),
    "collate_traces": ("tdgnet.data", "collate_traces"),
    "select_trace_keyframes": ("tdgnet.data", "select_trace_keyframes"),
    "undersample_to_balance": ("tdgnet.data", "undersample_to_balance"),
    "build_graph_from_attention": ("tdgnet.graph", "build_graph_from_attention"),
    "build_temporal_graph": ("tdgnet.graph", "build_temporal_graph"),
    "default_keyframe_indices": ("tdgnet.keyframes", "default_keyframe_indices"),
    "load_checkpoint": ("tdgnet.checkpoint", "load_checkpoint"),
    "save_checkpoint": ("tdgnet.checkpoint", "save_checkpoint"),
}

__all__ = list(_EXPORTS)
__version__ = "0.1.0"


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f"module 'tdgnet' has no attribute {name!r}")
    module, attribute = _EXPORTS[name]
    value = getattr(import_module(module), attribute)
    globals()[name] = value
    return value
