"""Compatibility imports for the former flat CogVis adapter namespace."""

__all__ = ["ChangeAdapter", "ScoreAdapter", "GateAdapter"]


def __getattr__(name: str):
    if name == "ChangeAdapter":
        from .change_adapter import ChangeAdapter

        return ChangeAdapter
    if name == "ScoreAdapter":
        from .score_adapter import ScoreAdapter

        return ScoreAdapter
    if name == "GateAdapter":
        from .gate_adapter import GateAdapter

        return GateAdapter
    raise AttributeError(name)
