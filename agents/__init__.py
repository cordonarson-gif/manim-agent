from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORTS: dict[str, tuple[str, str]] = {
    "ast_reviewer_node": ("ast_reviewer", "ast_reviewer_node"),
    "planner_node": ("planner", "planner_node"),
    "coder_node": ("coder", "coder_node"),
    "execution_node": ("execution", "execution_node"),
    "vision_critic_node": ("vision_critic", "vision_critic_node"),
}

__all__ = list(_EXPORTS)


def __getattr__(name: str) -> Any:
    """Lazily import agent nodes so lightweight modules stay testable."""

    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_name, attr_name = _EXPORTS[name]
    value = getattr(import_module(f".{module_name}", __name__), attr_name)
    globals()[name] = value
    return value
