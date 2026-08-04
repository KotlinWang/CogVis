"""SMC semantic-memory stage: response calibration and the Score Adapter."""

from __future__ import annotations

__all__ = ["SemanticMemoryCalibrator", "ScoreAdapter"]


def __getattr__(name: str):
    if name == "SemanticMemoryCalibrator":
        from .semantic_memory_calibrator import SemanticMemoryCalibrator

        return SemanticMemoryCalibrator
    if name == "ScoreAdapter":
        from .score_adapter import ScoreAdapter

        return ScoreAdapter
    raise AttributeError(name)
