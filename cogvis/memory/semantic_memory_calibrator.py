"""Semantic Memory Calibrator (SMC), paper Eq. (4)-(8)."""

from __future__ import annotations

from cogvis._legacy import runtime_symbol

_build_memory_key = runtime_symbol("clip_memory_threshold", "build_memory_key")
_retrieve_tau = runtime_symbol("clip_memory_threshold", "retrieve_tau")
_memory_rectified_score = runtime_symbol("memory_ovcd_readout", "memory_rectified_score")
_extract_class_scores = runtime_symbol("posterior_ovcd", "extract_class_scores")
_build_posterior_inputs = runtime_symbol("posterior_ovcd", "build_posterior_inputs")

__all__ = ["SemanticMemoryCalibrator"]


class SemanticMemoryCalibrator:
    """Paper-level facade over the locked semantic-memory operations."""

    build_memory_key = staticmethod(_build_memory_key)
    construct_semantic_change_response = staticmethod(_memory_rectified_score)
    extract_query_responses = staticmethod(_extract_class_scores)

    @staticmethod
    def retrieve_memory_anchor(*args, **kwargs):
        """Retrieve the top-K memory anchor tau-bar for one image-query pair."""

        return _retrieve_tau(*args, **kwargs)

    @staticmethod
    def build_competitive_posteriors(*args, **kwargs):
        """Build the competing prompt posteriors used to form R_c."""

        return _build_posterior_inputs(*args, **kwargs)

    # Earlier facade names remain import-compatible.
    build_query_key = build_memory_key
    calibrate_response = construct_semantic_change_response
    retrieve_adaptive_threshold = retrieve_memory_anchor
    compose_query_response = build_competitive_posteriors
