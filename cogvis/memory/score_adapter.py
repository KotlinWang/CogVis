"""Score Adapter, corresponding to A_score in paper Eq. (8)."""

from __future__ import annotations

from cogvis._legacy import runtime_symbol

_CorrectionHead = runtime_symbol("clip_memory_threshold", "CorrectionHead")

__all__ = ["ScoreAdapter"]


class ScoreAdapter(_CorrectionHead):
    """Paper-named subclass of the locked threshold-correction head."""
