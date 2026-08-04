"""Change Adapter, corresponding to A_chg in paper Eq. (1)-(3)."""

from __future__ import annotations

from cogvis._legacy import runtime_symbol

_TemporalGateFPNAdapter = runtime_symbol("temporal_lora_adapter", "TemporalGateFPNAdapter")

__all__ = ["ChangeAdapter"]


class ChangeAdapter(_TemporalGateFPNAdapter):
    """Paper-named subclass with an unchanged state-dict and forward path."""
