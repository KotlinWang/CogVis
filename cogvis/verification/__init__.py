"""ARF verification stage: region filtering and the Gate Adapter."""

from __future__ import annotations

__all__ = ["AdaptiveRegionFilter", "GateAdapter"]


def __getattr__(name: str):
    if name == "AdaptiveRegionFilter":
        from .adaptive_region_filter import AdaptiveRegionFilter

        return AdaptiveRegionFilter
    if name == "GateAdapter":
        from .gate_adapter import GateAdapter

        return GateAdapter
    raise AttributeError(name)
