"""Compatibility imports for the former flat CogVis module namespace."""

__all__ = ["SceneChangePerceptron", "SemanticMemoryCalibrator", "AdaptiveRegionFilter"]


def __getattr__(name: str):
    if name == "SceneChangePerceptron":
        from .scene_change_perceptron import SceneChangePerceptron

        return SceneChangePerceptron
    if name == "SemanticMemoryCalibrator":
        from .semantic_memory_calibrator import SemanticMemoryCalibrator

        return SemanticMemoryCalibrator
    if name == "AdaptiveRegionFilter":
        from .adaptive_region_filter import AdaptiveRegionFilter

        return AdaptiveRegionFilter
    raise AttributeError(name)
