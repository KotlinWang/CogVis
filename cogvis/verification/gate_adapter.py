"""Gate Adapter, corresponding to A_gate in paper Eq. (9)."""

from __future__ import annotations

from cogvis._legacy import runtime_symbol

_ComponentGateHead = runtime_symbol(
    "train_cacdd_ratio_threshold_component_gate", "ComponentGateHead"
)

__all__ = ["GateAdapter"]


class GateAdapter(_ComponentGateHead):
    """Paper-named subclass of the locked region-reliability classifier."""
