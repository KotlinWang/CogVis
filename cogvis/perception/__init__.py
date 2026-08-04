"""SCP perception stage: the shared change prior and Change Adapter."""

from __future__ import annotations

__all__ = ["SceneChangePerceptron", "ChangeAdapter"]


def __getattr__(name: str):
    if name == "SceneChangePerceptron":
        from .scene_change_perceptron import SceneChangePerceptron

        return SceneChangePerceptron
    if name == "ChangeAdapter":
        from .change_adapter import ChangeAdapter

        return ChangeAdapter
    raise AttributeError(name)
