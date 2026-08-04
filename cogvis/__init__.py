"""Paper-aligned public API for the locked CogVis implementation.

Imports are lazy so command construction and integrity checks do not initialize
the frozen vision stack.
"""

from __future__ import annotations

from importlib import import_module

__version__ = "2026.7.11"
__all__ = [
    "CogVis",
    "SceneChangePerceptron",
    "SemanticMemoryCalibrator",
    "AdaptiveRegionFilter",
    "ChangeAdapter",
    "ScoreAdapter",
    "GateAdapter",
]

_PUBLIC = {
    "CogVis": ("cogvis.pipeline", "CogVis"),
    "SceneChangePerceptron": ("cogvis.perception.scene_change_perceptron", "SceneChangePerceptron"),
    "SemanticMemoryCalibrator": ("cogvis.memory.semantic_memory_calibrator", "SemanticMemoryCalibrator"),
    "AdaptiveRegionFilter": ("cogvis.verification.adaptive_region_filter", "AdaptiveRegionFilter"),
    "ChangeAdapter": ("cogvis.perception.change_adapter", "ChangeAdapter"),
    "ScoreAdapter": ("cogvis.memory.score_adapter", "ScoreAdapter"),
    "GateAdapter": ("cogvis.verification.gate_adapter", "GateAdapter"),
}


def __getattr__(name: str):
    if name not in _PUBLIC:
        raise AttributeError(name)
    module_name, symbol_name = _PUBLIC[name]
    value = getattr(import_module(module_name), symbol_name)
    globals()[name] = value
    return value
