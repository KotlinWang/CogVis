"""Adaptive Region Filter (ARF), paper Eq. (9)."""

from __future__ import annotations

from cogvis._legacy import runtime_symbol

_component_gate_features = runtime_symbol(
    "train_cacdd_ratio_threshold_component_gate", "component_gate_features"
)
_component_gate_filter = runtime_symbol(
    "train_cacdd_ratio_threshold_component_gate", "component_gate_filter"
)

__all__ = ["AdaptiveRegionFilter"]


class AdaptiveRegionFilter:
    """Paper-level facade for candidate-region verification."""

    encode_candidate_region = staticmethod(_component_gate_features)

    @staticmethod
    def verify_candidate_regions(*args, **kwargs):
        """Retain only ARF candidates that satisfy the Gate Adapter rule."""

        return _component_gate_filter(*args, **kwargs)

    # Earlier facade names remain import-compatible.
    build_region_features = encode_candidate_region
    filter_candidate_regions = verify_candidate_regions
