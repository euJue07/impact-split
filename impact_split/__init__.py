from __future__ import annotations

import importlib
from typing import Any

from impact_split.plots import InteractiveForceGraph, interactive_force_graph
from impact_split.splitter import ImpactSplitter

__all__ = ["ImpactSplitter", "InteractiveForceGraph", "interactive_force_graph", "config"]


def __getattr__(name: str) -> Any:
    if name == "config":
        return importlib.import_module("impact_split.config")
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)
