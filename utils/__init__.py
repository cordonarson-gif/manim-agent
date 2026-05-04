from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORTS: dict[str, tuple[str, str]] = {
    "ExperimentLogger": ("experiment_logger", "ExperimentLogger"),
    "inject_bounding_boxes": ("manim_injector", "inject_bounding_boxes"),
    "build_deepseek_chat_model": ("model_provider", "build_deepseek_chat_model"),
    "build_qwen_vision_model": ("model_provider", "build_qwen_vision_model"),
}

__all__ = list(_EXPORTS)


def __getattr__(name: str) -> Any:
    """Lazily import helpers to avoid unnecessary model dependency imports."""

    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_name, attr_name = _EXPORTS[name]
    value = getattr(import_module(f".{module_name}", __name__), attr_name)
    globals()[name] = value
    return value
