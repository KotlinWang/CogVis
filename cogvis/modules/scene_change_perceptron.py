"""Compatibility import for :mod:`cogvis.perception.scene_change_perceptron`."""

from cogvis.perception.scene_change_perceptron import (
    SceneChangePerceptron,
    compute_change_prior,
    compute_query_shared_change_prior,
    perceive_scene_change,
)

__all__ = [
    "SceneChangePerceptron",
    "compute_change_prior",
    "compute_query_shared_change_prior",
    "perceive_scene_change",
]
