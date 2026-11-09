from __future__ import annotations

from typing import Any

from .state import DataAgentState

__all__ = ["DataAgentState", "compiled_agent", "build_agent"]


def __getattr__(name: str) -> Any:  # PEP 562：惰性代理
    if name in ("compiled_agent", "build_agent"):
        from . import builder

        return getattr(builder, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
