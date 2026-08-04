"""Scene Change Perceptron (SCP), paper Eq. (1)-(3)."""

from __future__ import annotations

from cogvis._legacy import runtime_symbol

_SemanticTemporalEvidenceEncoder = runtime_symbol(
    "change_module", "SemanticTemporalEvidenceEncoder"
)

__all__ = [
    "SceneChangePerceptron",
    "compute_change_prior",
    "compute_query_shared_change_prior",
    "perceive_scene_change",
]


class SceneChangePerceptron(_SemanticTemporalEvidenceEncoder):
    """Paper-named, checkpoint-compatible SCP implementation.

    The locked paper checkpoint selects ``temporal_gate_v4`` with active FPN
    level ``2`` and gate level ``2``. Constructor defaults are inherited for
    compatibility; load checkpoint metadata before reproducing paper results.
    """


def compute_change_prior(perceptron, fpn_t1, fpn_t2, output_size=None):
    """Compute the category-agnostic prior B shared by every query."""

    return perceptron(fpn_t1, fpn_t2, output_size=output_size)


# The explicit aliases make the query-sharing role visible without changing
# callers of the earlier public facade.
compute_query_shared_change_prior = compute_change_prior
perceive_scene_change = compute_change_prior
