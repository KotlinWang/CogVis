"""Cross-temporal memory readout for CogVis OVCD experiments.

This module is intentionally independent from the current best evaluator. It
keeps the frozen SAM3 posterior path unchanged and only replaces the final
class-score readout with a memory-inspired temporal rectification. The design
is meant for side-by-side comparison against ``eval_ovcd_benchmark.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from posterior_ovcd import (
    build_posterior_inputs,
    compute_score_threshold,
    decode_ovcd_score,
    extract_class_scores,
)


@dataclass
class MemoryReadoutConfig:
    """Hyperparameters for memory-rectified posterior change scores."""

    delta_weight: float = 0.65
    transition_weight: float = 0.35
    bcd_alpha: float = 1.0
    bcd_floor: float = 0.15
    stable_suppress: float = 0.45
    stable_power: float = 1.0
    support_boost: float = 0.10
    presence_delta_weight: float = 0.15
    robust_normalize: bool = False
    q_low: float = 0.02
    q_high: float = 0.98


def _getattr(args, name: str, default):
    return getattr(args, name, default) if args is not None else default


def config_from_args(args=None) -> MemoryReadoutConfig:
    return MemoryReadoutConfig(
        delta_weight=float(_getattr(args, "memory_delta_weight", 0.65)),
        transition_weight=float(_getattr(args, "memory_transition_weight", 0.35)),
        bcd_alpha=float(_getattr(args, "memory_bcd_alpha", 1.0)),
        bcd_floor=float(_getattr(args, "memory_bcd_floor", 0.15)),
        stable_suppress=float(_getattr(args, "memory_stable_suppress", 0.45)),
        stable_power=float(_getattr(args, "memory_stable_power", 1.0)),
        support_boost=float(_getattr(args, "memory_support_boost", 0.10)),
        presence_delta_weight=float(_getattr(args, "memory_presence_delta_weight", 0.15)),
        robust_normalize=bool(_getattr(args, "memory_robust_normalize", False)),
        q_low=float(_getattr(args, "memory_q_low", 0.02)),
        q_high=float(_getattr(args, "memory_q_high", 0.98)),
    )


def _robust_normalize(score: torch.Tensor, q_low: float, q_high: float) -> torch.Tensor:
    q_low = min(max(float(q_low), 0.0), 0.49)
    q_high = min(max(float(q_high), 0.51), 1.0)
    lo = torch.quantile(score.float(), q_low)
    hi = torch.quantile(score.float(), q_high)
    return ((score.float() - lo) / (hi - lo + 1e-6)).clamp(0, 1)


def memory_rectified_score(
    posterior_a: torch.Tensor,
    posterior_b: torch.Tensor,
    bcd_score: Optional[torch.Tensor],
    presence_a: Optional[torch.Tensor] = None,
    presence_b: Optional[torch.Tensor] = None,
    config: Optional[MemoryReadoutConfig] = None,
) -> torch.Tensor:
    """Return a cross-temporal memory score for one queried class.

    The first timestamp acts as memory and the second timestamp acts as the
    incoming observation, but the evidence is symmetric: both disappearance and
    emergence are kept. Stable high-confidence semantics are suppressed when
    the category-agnostic BCD prior says the region is unchanged.
    """
    cfg = config or MemoryReadoutConfig()
    pa = posterior_a.float().clamp(0, 1)
    pb = posterior_b.float().clamp(0, 1)
    delta = (pb - pa).abs()
    emergence = pb * (1.0 - pa)
    disappearance = pa * (1.0 - pb)
    transition = torch.maximum(emergence, disappearance)
    stable = torch.minimum(pa, pb)
    support = torch.maximum(pa, pb)

    score = cfg.delta_weight * delta + cfg.transition_weight * transition

    if bcd_score is not None:
        bcd = bcd_score.float().clamp(0, 1)
        gate = cfg.bcd_floor + (1.0 - cfg.bcd_floor) * bcd.pow(cfg.bcd_alpha)
        stable_penalty = stable.pow(cfg.stable_power) * (1.0 - bcd).clamp(0, 1)
        keep = (1.0 - cfg.stable_suppress * stable_penalty).clamp(0, 1)
        score = score * gate * keep
        if cfg.support_boost > 0:
            score = torch.maximum(score, cfg.support_boost * support * bcd)

    if presence_a is not None and presence_b is not None and cfg.presence_delta_weight > 0:
        presence_delta = (presence_b.float() - presence_a.float()).abs().clamp(0, 1)
        score = score * (1.0 + cfg.presence_delta_weight * presence_delta)

    score = score.clamp(0, 1)
    if cfg.robust_normalize:
        score = _robust_normalize(score, cfg.q_low, cfg.q_high)
    return score.clamp(0, 1)


def _threshold_for(
    class_thresholds: Dict[str, float],
    dataset: str,
    class_name: str,
    fallback: Optional[float],
) -> Optional[float]:
    return class_thresholds.get(f"{dataset}:{class_name}", class_thresholds.get(class_name, fallback))


def _optional_abs_threshold(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    value = float(value)
    if value < 0:
        return None
    return value


def _bcd_mass_foreground_ratio(bcd_score: torch.Tensor, memory_args) -> float:
    bcd = bcd_score.float().clamp(0, 1)
    bcd_threshold = compute_score_threshold(
        bcd,
        mode=str(_getattr(memory_args, "bcd_mass_threshold_mode", "adaptive")),
        percentile=float(_getattr(memory_args, "bcd_mass_threshold_percentile", 85.0)),
        threshold_abs=_optional_abs_threshold(_getattr(memory_args, "bcd_mass_threshold_abs", None)),
        std_factor=float(_getattr(memory_args, "bcd_mass_threshold_std_factor", 1.0)),
        threshold_min=float(_getattr(memory_args, "bcd_mass_threshold_min", 0.0)),
        threshold_max=float(_getattr(memory_args, "bcd_mass_threshold_max", 1.0)),
        threshold_max_foreground_ratio=0.0,
    )
    ratio = float((bcd > bcd_threshold).float().mean().detach().cpu())
    ratio *= float(_getattr(memory_args, "bcd_mass_ratio_scale", 1.0))
    return float(np.clip(
        ratio,
        float(_getattr(memory_args, "bcd_mass_ratio_min", 0.002)),
        float(_getattr(memory_args, "bcd_mass_ratio_max", 0.35)),
    ))


def _decode_foreground_ratio(args, memory_args, bcd_score: Optional[torch.Tensor]) -> float:
    base_ratio = float(_getattr(args, "threshold_max_foreground_ratio", 0.0) or 0.0)
    if str(_getattr(memory_args, "decode_calibration", "none")) != "bcd_mass":
        return base_ratio
    if bcd_score is None:
        return base_ratio
    ratio = _bcd_mass_foreground_ratio(bcd_score, memory_args)
    if base_ratio > 0:
        ratio = min(ratio, base_ratio)
    return ratio


@torch.no_grad()
def build_memory_posterior_masks(
    processor,
    args,
    dataset: str,
    meta: dict,
    class_names: Sequence[str],
    prompt_bank: Dict[str, List[str]],
    confusers: Dict[str, List[str]],
    class_thresholds: Dict[str, float],
    image_mean: Optional[np.ndarray] = None,
    memory_args=None,
) -> Tuple[Dict[str, np.ndarray], Dict[str, float]]:
    """Drop-in replacement for eval_ovcd_benchmark.build_posterior_masks."""
    class_names = list(class_names)
    cfg = config_from_args(memory_args)
    scores_a, presence_a = extract_class_scores(processor, meta["state_a"], prompt_bank, class_names)
    scores_b, presence_b = extract_class_scores(processor, meta["state_b"], prompt_bank, class_names)
    posterior = build_posterior_inputs(
        scores_a,
        scores_b,
        presence_a,
        presence_b,
        class_names=class_names,
        confusers=confusers,
        rho=args.rho,
    )

    pred_masks: Dict[str, np.ndarray] = {}
    thresholds: Dict[str, float] = {}
    for cls_idx, cls_name in enumerate(class_names):
        score = memory_rectified_score(
            posterior.posterior_a[cls_idx],
            posterior.posterior_b[cls_idx],
            None if args.disable_bcd_gate else meta["bcd_score"],
            presence_a=presence_a[cls_idx],
            presence_b=presence_b[cls_idx],
            config=cfg,
        )

        threshold_abs = _threshold_for(class_thresholds, dataset, cls_name, args.threshold_abs)
        presence = torch.maximum(presence_a[cls_idx], presence_b[cls_idx])
        max_foreground_ratio = _decode_foreground_ratio(
            args,
            memory_args,
            None if args.disable_bcd_gate else meta["bcd_score"],
        )
        pred_mask, thd = decode_ovcd_score(
            score,
            threshold_mode=args.threshold_mode,
            threshold_percentile=args.threshold_percentile,
            threshold_abs=threshold_abs,
            threshold_std_factor=args.threshold_std_factor,
            threshold_edge_weight=args.threshold_edge_weight,
            threshold_edge_percentile=args.threshold_edge_percentile,
            threshold_edge_dilate=args.threshold_edge_dilate,
            threshold_edge_min_pixels=args.threshold_edge_min_pixels,
            threshold_min=args.threshold_min,
            threshold_max=args.threshold_max,
            threshold_scale=args.threshold_scale,
            threshold_bias=args.threshold_bias,
            threshold_max_foreground_ratio=max_foreground_ratio,
            threshold_candidate_percentile=args.threshold_candidate_percentile,
            min_score=args.min_score,
            presence=presence,
            presence_min=args.presence_min,
            region_mode=args.region_mode,
            image_np=image_mean,
            slic_segments=args.slic_segments,
            region_score=args.region_score,
            min_region_area=args.min_region_area,
            max_region_area=args.max_region_area,
            topk_ratio=args.region_topk_ratio,
            acf_conf_percentile=args.acf_conf_percentile,
            acf_conf_lambda=args.acf_conf_lambda,
            acf_conf_min=args.acf_conf_min,
            acf_conf_max=args.acf_conf_max,
            acf_cv_max=args.acf_cv_max,
        )
        pred_masks[cls_name] = pred_mask.detach().cpu().numpy().astype(np.int64)
        thresholds[cls_name] = float(thd.detach().cpu())
    return pred_masks, thresholds
