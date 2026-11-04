"""Train and evaluate CLIP-guided Memory-Corrected Thresholding (CMCT).

This entrypoint is deliberately separate from the current CogVis evaluator.
It uses only CA-CDD labels to build a threshold memory bank and correction
head, then evaluates target OVCD datasets without target-label calibration.
"""

from __future__ import annotations

import argparse
import json
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

import eval_ovcd_benchmark as bench
import train_cacdd_ratio_threshold_component_gate as base
from clip_memory_threshold import (
    CMCT_STAT_NAMES,
    ClipFeatureExtractor,
    CorrectionHead,
    build_memory_key,
    correction_features,
    retrieve_tau,
    score_clip_stats,
)
from memory_ovcd_readout import memory_rectified_score
from posterior_ovcd import build_posterior_inputs, decode_ovcd_score, extract_class_scores


FIXED_THRESHOLD_REFERENCE = {
    "SCSCD:Bareland": 0.04,
    "SCSCD:Water": 0.20,
    "SCSCD:Building": 0.20,
    "SCSCD:Structure": 0.02,
    "SCSCD:Farmland": 0.08,
    "SCSCD:Vegetation": 0.08,
    "SCSCD:Road": 0.30,
    "SECOND:Building": 0.30,
    "SECOND:Tree": 0.06,
    "SECOND:Water": 0.15,
    "SECOND:Grass": 0.06,
    "SECOND:Ground": 0.06,
    "SECOND:Playground": 0.20,
    "CLCD-512:Bareland": 0.12,
    "CLCD-512:Grass": 0.12,
    "CLCD-512:Road": 0.12,
    "CLCD-512:Tree": 0.12,
    "CLCD-512:Water": 0.12,
    "CLCD-512:Cropland": 0.12,
    "CLCD-512:Building": 0.12,
    "DSIFN-512:Building": 0.10,
    "LEVIR-CD-512:Building": 0.35,
    "WHU-CD-512:Building": 0.35,
}


def _prediction_file(args, dataset: str, item: dict, class_name: str | None = None) -> Path | None:
    output_root = str(getattr(args, "prediction_output_dir", "") or "").strip()
    if not output_root:
        return None
    directory = Path(output_root) / dataset
    if class_name is not None:
        directory = directory / class_name
    directory.mkdir(parents=True, exist_ok=True)
    return directory / f"{Path(str(item['name'])).stem}.png"


def export_semantic_prediction(args, dataset: str, item: dict, class_names, pred_by_method) -> None:
    method = str(getattr(args, "prediction_method", "clip_corrected_tau_component_gate"))
    predictions = pred_by_method.get(method)
    if predictions is None:
        raise KeyError(f"Prediction export method is unavailable: {method}")
    for class_name in class_names:
        path = _prediction_file(args, dataset, item, class_name)
        if path is None:
            return
        mask = np.asarray(predictions[class_name]).astype(bool)
        Image.fromarray(mask.astype(np.uint8) * 255).save(path)


def export_binary_prediction(args, dataset: str, item: dict, pred_by_method) -> None:
    method = str(getattr(args, "prediction_method", "clip_corrected_tau_component_gate"))
    prediction = pred_by_method.get(method)
    if prediction is None:
        raise KeyError(f"Prediction export method is unavailable: {method}")
    path = _prediction_file(args, dataset, item)
    if path is None:
        return
    mask = np.asarray(prediction).astype(bool)
    Image.fromarray(mask.astype(np.uint8) * 255).save(path)

CMCT_CLASS_CONDITION_NAMES = [
    "class_tau_mean",
    "class_tau_std",
    "class_tau_q50",
    "class_tau_q75",
    "class_tau_q90",
    "class_tau_max",
    "class_tau_count_ratio",
    "global_tau_mean",
    "global_tau_q75",
    "global_tau_q90",
    "tau_mem_minus_class_mean",
    "tau_mem_minus_class_q75",
    "tau_mem_minus_class_q90",
    "class_mean_over_global_mean",
    "class_q90_over_global_q90",
]

CMCT_COMPONENT_RELIABILITY_NAMES = [
    "component_count_log",
    "component_area_ratio",
    "component_largest_area_ratio",
    "component_gate_prob_mean",
    "component_gate_prob_min",
    "component_gate_prob_max",
    "component_gate_prob_area_mean",
    "component_keep_area_ratio",
    "component_low_prob_area_ratio",
    "component_keep_count_ratio",
    "component_bcd_support_mean",
    "component_score_margin_mean",
]

CMCT_SEMANTIC_RELATION_NAMES = [
    "semantic_top1_sim",
    "semantic_top2_sim",
    "semantic_top1_margin",
    "semantic_sim_mean",
    "semantic_sim_std",
    "semantic_entropy",
    "semantic_weighted_tau_mean",
    "semantic_weighted_tau_std",
    "semantic_weighted_tau_q50",
    "semantic_weighted_tau_q75",
    "semantic_weighted_tau_q90",
    "semantic_weighted_ratio_mean",
    "semantic_weighted_ratio_q75",
    "semantic_weighted_ratio_q90",
    "semantic_weighted_precision_risk",
    "semantic_weighted_recall_risk",
    "semantic_top1_tau_mean",
    "semantic_top1_tau_std",
    "semantic_top1_tau_q50",
    "semantic_top1_tau_q75",
    "semantic_top1_tau_q90",
    "semantic_top1_ratio_mean",
    "semantic_top1_ratio_q75",
    "semantic_top1_precision_risk",
    "semantic_top1_recall_risk",
    "semantic_top1_count_ratio",
]

CMCT_ORACLE_METRIC_NAMES = ["oracle_iou", "oracle_f1", "oracle_precision", "oracle_recall"]

CMCT_META_THRESHOLD_SCALAR_NAMES = [
    "meta_tau_mem",
    "meta_tau_corr",
    "meta_adaptive_tau",
    "meta_tau_corr_minus_mem",
    "meta_tau_corr_minus_adaptive",
    "meta_tau_mem_minus_adaptive",
    "meta_tau_corr_over_adaptive",
    "meta_tau_mem_over_adaptive",
    "meta_ratio_mem",
    "meta_uncertainty",
    "meta_trust",
]

CMCT_CLASS_RESIDUAL_ADAPTER_STATS = ["mean", "q50", "q75", "q90"]
CMCT_SCORE_CALIBRATION_MODES = ["shift_q95", "affine_q50_q95", "affine_q75_q95", "bcd_product"]

COMPONENT_VERIFIER_CLIP_FEATURE_NAMES = [
    "clip_before_sim",
    "clip_after_sim",
    "clip_scene_sim",
    "clip_best_state_sim",
    "clip_before_margin",
    "clip_after_margin",
    "clip_scene_margin",
    "clip_best_margin",
    "clip_after_minus_before",
    "clip_scene_minus_before",
]

COMPONENT_VERIFIER_FEATURE_NAMES = list(base.COMPONENT_GATE_FEATURE_NAMES) + COMPONENT_VERIFIER_CLIP_FEATURE_NAMES

SUPPRESSED_RECOVERY_EXTRA_FEATURE_NAMES = [
    "corr_cover_ratio",
    "gate_cover_ratio",
    "suppressed_area_ratio",
    "suppressed_over_image",
    "component_gate_keep_prob",
    "tau_corr",
    "relaxed_tau",
    "tau_relax_delta",
    "score_support_lost_ratio",
    "bcd_support_lost_ratio",
]

PRAT_REGION_EXTRA_FEATURE_NAMES = [
    "prat_semantic_mean",
    "prat_semantic_q75",
    "prat_semantic_q90",
    "prat_semantic_max",
    "prat_structure_mean",
    "prat_structure_q75",
    "prat_structure_q90",
    "prat_structure_max",
    "prat_score_support_mean",
    "prat_score_support_q75",
    "prat_semantic_structure_product_mean",
    "prat_semantic_structure_product_q75",
    "prat_component_area_ratio",
]

SUPPRESSED_RECOVERY_FEATURE_NAMES = list(base.COMPONENT_GATE_FEATURE_NAMES) + SUPPRESSED_RECOVERY_EXTRA_FEATURE_NAMES


def suppressed_recovery_feature_names(args=None) -> List[str]:
    names = list(SUPPRESSED_RECOVERY_FEATURE_NAMES)
    if args is not None and bool(getattr(args, "cmct_suppressed_recovery_enable_prat_region_features", False)):
        names += list(PRAT_REGION_EXTRA_FEATURE_NAMES)
    if args is not None and bool(getattr(args, "cmct_suppressed_recovery_enable_clip_features", False)):
        names += list(COMPONENT_VERIFIER_CLIP_FEATURE_NAMES)
    return names


def suppressed_recovery_feature_dim(args=None) -> int:
    return len(suppressed_recovery_feature_names(args))


SOURCE_PROMPTS = OrderedDict([
    ("Building", ["building", "roof", "house", "construction"]),
    ("Tree", ["tree", "forest"]),
    ("Water", ["water", "river", "pond"]),
    ("Grass", ["grass", "low vegetation"]),
    ("Ground", ["ground", "bare ground", "surface"]),
    ("Playground", ["playground", "sports field"]),
    ("Bareland", ["bareland", "barren ground", "soil"]),
    ("Road", ["road"]),
    ("Cropland", ["cropland", "farmland"]),
    ("Farmland", ["farmland", "cropland", "terrace"]),
    ("Vegetation", ["vegetation", "grass", "tree", "forest"]),
    ("Structure", ["structure", "greenhouse", "construction"]),
])


def source_prompt_spec(args) -> Tuple[List[str], Dict[str, List[str]], Dict[str, List[str]]]:
    requested = [x.strip() for x in str(args.clip_memory_source_classes).split(",") if x.strip()]
    if not requested or requested == ["union_remote"]:
        class_names = list(SOURCE_PROMPTS.keys())
    else:
        missing = [name for name in requested if name not in SOURCE_PROMPTS]
        if missing:
            raise ValueError(f"Unknown source classes for CMCT: {missing}")
        class_names = requested
    if int(args.clip_memory_max_source_classes) > 0:
        class_names = class_names[: int(args.clip_memory_max_source_classes)]
    prompt_bank = {name: list(SOURCE_PROMPTS[name]) for name in class_names}
    confusers = {name: [other for other in class_names if other != name] for name in class_names}
    return class_names, prompt_bank, confusers


def prompt_text_for_class(class_name: str, prompt_bank: Dict[str, List[str]]) -> str:
    prompts = prompt_bank.get(class_name) or [class_name]
    return str(prompts[0])


def parse_threshold_grid(spec: str) -> List[float]:
    spec = str(spec).strip()
    if ":" in spec:
        start, stop, step = [float(x) for x in spec.split(":")]
        values = []
        cur = start
        while cur <= stop + 1e-9:
            values.append(float(cur))
            cur += step
        return values
    return [float(x.strip()) for x in spec.split(",") if x.strip()]


def _clipper_text_embedding(clipper: ClipFeatureExtractor, text: str) -> np.ndarray:
    cache = getattr(clipper, "_cmct_text_cache", None)
    if cache is None:
        cache = {}
        setattr(clipper, "_cmct_text_cache", cache)
    text = str(text)
    if text not in cache:
        cache[text] = clipper.encode_text(text)
    return np.asarray(cache[text], dtype=np.float32)


def _softmax_np(values: np.ndarray, temperature: float) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if values.size == 0:
        return values
    temp = max(float(temperature), 1e-6)
    logits = values / temp
    logits = logits - float(np.max(logits))
    exp = np.exp(logits)
    return (exp / max(float(exp.sum()), 1e-8)).astype(np.float32)


def _source_class_tau_stats(bank: Dict[str, object], source_names: Sequence[str]):
    cache_key = tuple(str(name) for name in source_names)
    cache = bank.get("_source_class_tau_stats_cache")
    if isinstance(cache, dict) and cache.get("key") == cache_key:
        return cache["value"]

    memory_taus = np.asarray(bank.get("memory_taus", np.zeros((0,), dtype=np.float32)), dtype=np.float32)
    memory_ratios = np.asarray(bank.get("memory_ratios", np.zeros_like(memory_taus)), dtype=np.float32)
    memory_oracle_metrics = np.asarray(
        bank.get("memory_oracle_metrics", np.zeros((memory_taus.shape[0], len(CMCT_ORACLE_METRIC_NAMES)), dtype=np.float32)),
        dtype=np.float32,
    )
    if memory_ratios.shape[0] != memory_taus.shape[0]:
        memory_ratios = np.zeros_like(memory_taus)
    if memory_oracle_metrics.ndim != 2 or memory_oracle_metrics.shape[0] != memory_taus.shape[0]:
        memory_oracle_metrics = np.zeros((memory_taus.shape[0], len(CMCT_ORACLE_METRIC_NAMES)), dtype=np.float32)
    metas = list(bank.get("memory_meta", []))
    global_taus = memory_taus if memory_taus.size > 0 else np.zeros((1,), dtype=np.float32)
    global_ratios = memory_ratios if memory_ratios.size > 0 else np.zeros((1,), dtype=np.float32)
    global_mean, global_std, global_q50, global_q75, global_q90, _ = _tau_distribution_features(global_taus)
    global_ratio_mean, global_ratio_std, global_ratio_q50, global_ratio_q75, global_ratio_q90, _ = _tau_distribution_features(global_ratios)
    means, stds, q50s, q75s, q90s, counts = [], [], [], [], [], []
    ratio_means, ratio_stds, ratio_q50s, ratio_q75s, ratio_q90s = [], [], [], [], []
    precision_risks, recall_risks = [], []
    total = max(float(memory_taus.size), 1.0)
    for name in source_names:
        idx = [
            i for i, meta in enumerate(metas)
            if i < memory_taus.shape[0] and str(meta.get("class_name", "")) == str(name)
        ]
        if idx:
            idx_np = np.asarray(idx, dtype=np.int64)
            values = memory_taus[idx_np]
            ratios = memory_ratios[idx_np]
            metrics = memory_oracle_metrics[idx_np] if memory_oracle_metrics.shape[0] == memory_taus.shape[0] else np.zeros((len(idx), len(CMCT_ORACLE_METRIC_NAMES)), dtype=np.float32)
            mean, std, q50, q75, q90, _ = _tau_distribution_features(values)
            ratio_mean, ratio_std, ratio_q50, ratio_q75, ratio_q90, _ = _tau_distribution_features(ratios)
            count_ratio = float(len(idx)) / total
            if metrics.shape[1] >= len(CMCT_ORACLE_METRIC_NAMES) and float(metrics.sum()) > 0.0:
                precision = float(metrics[:, CMCT_ORACLE_METRIC_NAMES.index("oracle_precision")].mean())
                recall = float(metrics[:, CMCT_ORACLE_METRIC_NAMES.index("oracle_recall")].mean())
                precision_risk = float(1.0 - np.clip(precision, 0.0, 1.0))
                recall_risk = float(1.0 - np.clip(recall, 0.0, 1.0))
            else:
                precision_risk = float(np.clip(ratio_q90 * max(0.0, global_q50 - q50 + 0.05), 0.0, 1.0))
                recall_risk = float(np.clip(max(0.0, q50 - global_q50) + max(0.0, global_ratio_mean - ratio_mean), 0.0, 1.0))
        else:
            mean, std, q50, q75, q90, count_ratio = global_mean, global_std, global_q50, global_q75, global_q90, 0.0
            ratio_mean, ratio_std, ratio_q50, ratio_q75, ratio_q90 = global_ratio_mean, global_ratio_std, global_ratio_q50, global_ratio_q75, global_ratio_q90
            precision_risk, recall_risk = 0.0, 0.0
        means.append(mean)
        stds.append(std)
        q50s.append(q50)
        q75s.append(q75)
        q90s.append(q90)
        counts.append(count_ratio)
        ratio_means.append(ratio_mean)
        ratio_stds.append(ratio_std)
        ratio_q50s.append(ratio_q50)
        ratio_q75s.append(ratio_q75)
        ratio_q90s.append(ratio_q90)
        precision_risks.append(precision_risk)
        recall_risks.append(recall_risk)
    stats = (
        np.asarray(means, dtype=np.float32),
        np.asarray(stds, dtype=np.float32),
        np.asarray(q50s, dtype=np.float32),
        np.asarray(q75s, dtype=np.float32),
        np.asarray(q90s, dtype=np.float32),
        np.asarray(counts, dtype=np.float32),
        np.asarray(ratio_means, dtype=np.float32),
        np.asarray(ratio_stds, dtype=np.float32),
        np.asarray(ratio_q50s, dtype=np.float32),
        np.asarray(ratio_q75s, dtype=np.float32),
        np.asarray(ratio_q90s, dtype=np.float32),
        np.asarray(precision_risks, dtype=np.float32),
        np.asarray(recall_risks, dtype=np.float32),
    )
    bank["_source_class_tau_stats_cache"] = {"key": cache_key, "value": stats}
    return stats


def semantic_relation_features_from_embedding(args, bank: Dict[str, object], query_embedding: np.ndarray) -> np.ndarray:
    if not bool(getattr(args, "cmct_enable_semantic_relation", False)):
        return np.zeros((0,), dtype=np.float32)
    source_names = list(bank.get("source_class_names", []))
    source_embeddings = np.asarray(bank.get("source_class_text_embeddings", np.zeros((0, 0), dtype=np.float32)), dtype=np.float32)
    query_embedding = np.asarray(query_embedding, dtype=np.float32)
    if (
        not source_names
        or source_embeddings.ndim != 2
        or source_embeddings.shape[0] == 0
        or query_embedding.size != source_embeddings.shape[1]
    ):
        return np.zeros((len(CMCT_SEMANTIC_RELATION_NAMES),), dtype=np.float32)

    query_norm = query_embedding / max(float(np.linalg.norm(query_embedding)), 1e-8)
    source_norm = source_embeddings / np.maximum(np.linalg.norm(source_embeddings, axis=1, keepdims=True), 1e-8)
    sims = np.matmul(source_norm, query_norm).astype(np.float32)
    order = np.argsort(-sims)
    top1 = int(order[0])
    top2 = int(order[1]) if order.size > 1 else top1
    weights = _softmax_np(sims, float(getattr(args, "cmct_semantic_relation_temperature", 0.10)))
    entropy = 0.0
    if weights.size > 1:
        entropy = float(-(weights * np.log(np.maximum(weights, 1e-8))).sum() / np.log(float(weights.size)))
    (
        tau_means,
        tau_stds,
        tau_q50s,
        tau_q75s,
        tau_q90s,
        count_ratios,
        ratio_means,
        ratio_stds,
        ratio_q50s,
        ratio_q75s,
        ratio_q90s,
        precision_risks,
        recall_risks,
    ) = _source_class_tau_stats(bank, source_names)
    return np.asarray([
        float(sims[top1]),
        float(sims[top2]),
        float(sims[top1] - sims[top2]),
        float(sims.mean()),
        float(sims.std()),
        entropy,
        float((weights * tau_means).sum()),
        float((weights * tau_stds).sum()),
        float((weights * tau_q50s).sum()),
        float((weights * tau_q75s).sum()),
        float((weights * tau_q90s).sum()),
        float((weights * ratio_means).sum()),
        float((weights * ratio_q75s).sum()),
        float((weights * ratio_q90s).sum()),
        float((weights * precision_risks).sum()),
        float((weights * recall_risks).sum()),
        float(tau_means[top1]),
        float(tau_stds[top1]),
        float(tau_q50s[top1]),
        float(tau_q75s[top1]),
        float(tau_q90s[top1]),
        float(ratio_means[top1]),
        float(ratio_q75s[top1]),
        float(precision_risks[top1]),
        float(recall_risks[top1]),
        float(count_ratios[top1]),
    ], dtype=np.float32)


def semantic_relation_features_for_class(args, bank: Dict[str, object], cls_name: Optional[str]) -> np.ndarray:
    if not bool(getattr(args, "cmct_enable_semantic_relation", False)):
        return np.zeros((0,), dtype=np.float32)
    source_names = list(bank.get("source_class_names", []))
    source_embeddings = np.asarray(bank.get("source_class_text_embeddings", np.zeros((0, 0), dtype=np.float32)), dtype=np.float32)
    if cls_name in source_names and source_embeddings.ndim == 2 and source_embeddings.shape[0] == len(source_names):
        return semantic_relation_features_from_embedding(args, bank, source_embeddings[source_names.index(cls_name)])
    return np.zeros((len(CMCT_SEMANTIC_RELATION_NAMES),), dtype=np.float32)


def semantic_relation_features_for_prompt(
    args,
    bank: Dict[str, object],
    clipper: ClipFeatureExtractor,
    cls_name: str,
    prompt_bank: Dict[str, List[str]],
) -> np.ndarray:
    if not bool(getattr(args, "cmct_enable_semantic_relation", False)):
        return np.zeros((0,), dtype=np.float32)
    query_embedding = _clipper_text_embedding(clipper, prompt_text_for_class(cls_name, prompt_bank))
    return semantic_relation_features_from_embedding(args, bank, query_embedding)


def _oracle_target_mask(score_np: np.ndarray, gt_np: np.ndarray, args) -> np.ndarray:
    mode = str(getattr(args, "cmct_oracle_positive_mode", "binary"))
    if mode == "binary":
        return gt_np
    if mode != "class_topq":
        raise ValueError(f"Unknown cmct_oracle_positive_mode: {mode}")
    changed_scores = score_np[gt_np]
    if changed_scores.size == 0:
        return gt_np
    cutoff = float(np.quantile(changed_scores, float(args.cmct_oracle_positive_quantile)))
    cutoff = max(cutoff, float(args.cmct_oracle_positive_min_score))
    target = gt_np & (score_np >= cutoff)
    if int(target.sum()) < int(args.cmct_oracle_positive_min_pixels):
        return gt_np
    return target


def oracle_threshold_for_score(score: torch.Tensor, gt: np.ndarray, args) -> float:
    score_np = score.detach().float().clamp(0, 1).cpu().numpy().astype(np.float32).reshape(-1)
    gt_np = gt.astype(bool).reshape(-1)
    gt_np = _oracle_target_mask(score_np, gt_np, args)
    if score_np.size == 0 or int(gt_np.sum()) == 0:
        return float(args.cmct_threshold_min)
    best_threshold = float(args.cmct_threshold_min)
    best_value = -float("inf")
    for threshold in parse_threshold_grid(args.cmct_threshold_grid):
        threshold = float(np.clip(threshold, float(args.cmct_threshold_min), float(args.cmct_threshold_max)))
        pred = score_np > threshold
        tp = float((pred & gt_np).sum())
        fp = float((pred & ~gt_np).sum())
        fn = float((~pred & gt_np).sum())
        precision = tp / max(tp + fp, 1e-8)
        recall = tp / max(tp + fn, 1e-8)
        iou = tp / max(tp + fp + fn, 1e-8)
        f1 = 2.0 * precision * recall / max(precision + recall, 1e-8)
        value = iou if args.cmct_threshold_target_metric == "IoU" else f1
        if recall < float(args.cmct_min_recall):
            value -= float(args.cmct_recall_penalty) * (float(args.cmct_min_recall) - recall)
        if precision < float(args.cmct_min_precision):
            value -= float(args.cmct_precision_penalty) * (float(args.cmct_min_precision) - precision)
        if value > best_value:
            best_value = value
            best_threshold = threshold
    return float(best_threshold)


def oracle_ratio_for_score(score: torch.Tensor, tau: float) -> float:
    score_np = score.detach().float().clamp(0, 1).cpu().numpy().astype(np.float32)
    if score_np.size == 0:
        return 0.0
    return float((score_np > float(tau)).mean())


def oracle_metrics_for_score(score: torch.Tensor, gt: np.ndarray, tau: float) -> np.ndarray:
    score_np = score.detach().float().clamp(0, 1).cpu().numpy().astype(np.float32).reshape(-1)
    gt_np = gt.astype(bool).reshape(-1)
    if score_np.size == 0:
        return np.zeros((len(CMCT_ORACLE_METRIC_NAMES),), dtype=np.float32)
    pred = score_np > float(tau)
    tp = float((pred & gt_np).sum())
    fp = float((pred & ~gt_np).sum())
    fn = float((~pred & gt_np).sum())
    precision = tp / max(tp + fp, 1e-8)
    recall = tp / max(tp + fn, 1e-8)
    iou = tp / max(tp + fp + fn, 1e-8)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-8)
    return np.asarray([iou, f1, precision, recall], dtype=np.float32)


def _budgeted_tau(score: torch.Tensor, tau: float, budget_ratio: Optional[float], args) -> float:
    if not bool(getattr(args, "cmct_enable_memory_ratio_budget", False)):
        return float(tau)
    if budget_ratio is None or not np.isfinite(float(budget_ratio)) or float(budget_ratio) <= 0:
        return float(tau)
    ratio = float(budget_ratio) * float(getattr(args, "cmct_memory_ratio_scale", 1.0))
    ratio = float(np.clip(
        ratio,
        float(getattr(args, "cmct_memory_ratio_min", 1e-4)),
        float(getattr(args, "cmct_memory_ratio_max", 0.50)),
    ))
    mass_threshold = float(torch.quantile(score.float().clamp(0, 1), 1.0 - ratio).detach().cpu())
    return float(np.clip(
        max(float(tau), mass_threshold),
        float(args.cmct_threshold_min),
        float(args.cmct_threshold_max),
    ))


def decode_fixed_tau(
    score: torch.Tensor,
    presence: torch.Tensor,
    tau: float,
    args,
    budget_ratio: Optional[float] = None,
) -> Tuple[np.ndarray, float]:
    tau = _budgeted_tau(score, tau, budget_ratio, args)
    pred_mask, threshold = decode_ovcd_score(
        score,
        threshold_mode=args.threshold_mode,
        threshold_percentile=args.threshold_percentile,
        threshold_abs=float(tau),
        threshold_std_factor=args.threshold_std_factor,
        threshold_edge_weight=args.threshold_edge_weight,
        threshold_edge_percentile=args.threshold_edge_percentile,
        threshold_edge_dilate=args.threshold_edge_dilate,
        threshold_edge_min_pixels=args.threshold_edge_min_pixels,
        threshold_min=args.threshold_min,
        threshold_max=args.threshold_max,
        threshold_scale=1.0,
        threshold_bias=0.0,
        threshold_max_foreground_ratio=0.0,
        threshold_candidate_percentile=args.threshold_candidate_percentile,
        min_score=args.min_score,
        presence=presence,
        presence_min=args.presence_min,
        region_mode=args.region_mode,
        image_np=None,
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
    return pred_mask.detach().cpu().numpy().astype(np.int64), float(threshold.detach().cpu())


@torch.no_grad()
def build_memory_rows(
    args,
    items: Sequence[dict],
    processor,
    change_module,
    device,
    clipper: ClipFeatureExtractor,
    component_model=None,
):
    class_names, prompt_bank, confusers = source_prompt_spec(args)
    source_text_embeddings = {
        name: _clipper_text_embedding(clipper, prompt_text_for_class(name, prompt_bank))
        for name in class_names
    }
    cfg = base._memory_config_from_args(args)
    rows = []
    stats = {
        "source_classes": class_names,
        "num_images": 0,
        "num_rows": 0,
        "tau_mean": 0.0,
        "tau_min": 0.0,
        "tau_max": 0.0,
    }
    for item in tqdm(items, desc="collect-CMCT-memory", dynamic_ncols=True):
        meta, _, _, size_hw = bench.predict_change(processor, change_module, item, args, device)
        gt = bench.read_binary_mask(item["label"], size_hw).astype(bool)
        pair_embedding = clipper.encode_pair(item)
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
        for cls_idx, cls_name in enumerate(class_names):
            score = memory_rectified_score(
                posterior.posterior_a[cls_idx],
                posterior.posterior_b[cls_idx],
                None if args.disable_bcd_gate else meta["bcd_score"],
                presence_a=presence_a[cls_idx],
                presence_b=presence_b[cls_idx],
                config=cfg,
            )
            presence = torch.maximum(presence_a[cls_idx], presence_b[cls_idx])
            component_features = np.zeros((0,), dtype=np.float32)
            if bool(getattr(args, "cmct_enable_component_reliability", False)):
                base_mask, base_threshold = base.decode_class_score(score, presence, 0.0, args)
                component_features = component_reliability_features(
                    args,
                    base_mask,
                    score,
                    meta["bcd_score"],
                    base_threshold,
                    component_model,
                    device,
                )
            score_stats = score_clip_stats(score, meta["bcd_score"], presence)
            text_embedding = source_text_embeddings[cls_name]
            key = build_memory_key(
                pair_embedding,
                text_embedding,
                score_stats,
                scene_weight=args.clip_memory_scene_weight,
                delta_weight=args.clip_memory_delta_weight,
                text_weight=args.clip_memory_text_weight,
                stats_weight=args.clip_memory_stats_weight,
            )
            tau = oracle_threshold_for_score(score, gt, args)
            ratio = oracle_ratio_for_score(score, tau)
            oracle_metrics = oracle_metrics_for_score(score, gt, tau)
            rows.append({
                "key": key,
                "stats": score_stats,
                "tau": float(tau),
                "ratio": float(ratio),
                "oracle_metrics": oracle_metrics,
                "component_features": component_features,
                "source_text_embedding": text_embedding,
                "class_name": cls_name,
                "item_name": item.get("name", ""),
            })
        stats["num_images"] += 1
    taus = [row["tau"] for row in rows]
    if taus:
        stats.update({
            "num_rows": len(rows),
            "tau_mean": float(np.mean(taus)),
            "tau_min": float(np.min(taus)),
            "tau_max": float(np.max(taus)),
        })
    return rows, stats


def rows_to_arrays(rows: Sequence[dict]) -> Dict[str, object]:
    if not rows:
        return {
            "memory_keys": np.zeros((0, 1), dtype=np.float32),
            "memory_taus": np.zeros((0,), dtype=np.float32),
            "memory_ratios": np.zeros((0,), dtype=np.float32),
            "memory_stats": np.zeros((0, len(CMCT_STAT_NAMES)), dtype=np.float32),
            "memory_oracle_metrics": np.zeros((0, len(CMCT_ORACLE_METRIC_NAMES)), dtype=np.float32),
            "memory_component_features": np.zeros((0, len(CMCT_COMPONENT_RELIABILITY_NAMES)), dtype=np.float32),
            "source_class_names": [],
            "source_class_text_embeddings": np.zeros((0, 0), dtype=np.float32),
            "memory_meta": [],
        }
    component_rows = [
        np.asarray(row.get("component_features", np.zeros((0,), dtype=np.float32)), dtype=np.float32)
        for row in rows
    ]
    if component_rows and component_rows[0].size > 0:
        memory_component_features = np.stack(component_rows, axis=0).astype(np.float32)
    else:
        memory_component_features = np.zeros((len(rows), 0), dtype=np.float32)
    oracle_metric_rows = [
        np.asarray(row.get("oracle_metrics", np.zeros((len(CMCT_ORACLE_METRIC_NAMES),), dtype=np.float32)), dtype=np.float32)
        for row in rows
    ]
    if oracle_metric_rows:
        memory_oracle_metrics = np.stack(oracle_metric_rows, axis=0).astype(np.float32)
    else:
        memory_oracle_metrics = np.zeros((len(rows), len(CMCT_ORACLE_METRIC_NAMES)), dtype=np.float32)
    source_names = []
    source_embeddings = []
    seen = set()
    for row in rows:
        name = str(row["class_name"])
        if name in seen:
            continue
        seen.add(name)
        source_names.append(name)
        source_embeddings.append(np.asarray(row.get("source_text_embedding", np.zeros((0,), dtype=np.float32)), dtype=np.float32))
    if source_embeddings and source_embeddings[0].size > 0:
        source_class_text_embeddings = np.stack(source_embeddings, axis=0).astype(np.float32)
    else:
        source_class_text_embeddings = np.zeros((0, 0), dtype=np.float32)
    return {
        "memory_keys": np.stack([row["key"] for row in rows], axis=0).astype(np.float32),
        "memory_taus": np.asarray([row["tau"] for row in rows], dtype=np.float32),
        "memory_ratios": np.asarray([row.get("ratio", 0.0) for row in rows], dtype=np.float32),
        "memory_stats": np.stack([row["stats"] for row in rows], axis=0).astype(np.float32),
        "memory_oracle_metrics": memory_oracle_metrics,
        "memory_component_features": memory_component_features,
        "source_class_names": source_names,
        "source_class_text_embeddings": source_class_text_embeddings,
        "memory_meta": [{"class_name": row["class_name"], "item_name": row["item_name"]} for row in rows],
    }


def _tau_distribution_features(values: np.ndarray) -> Tuple[float, float, float, float, float, float]:
    values = np.asarray(values, dtype=np.float32)
    if values.size == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    q50, q75, q90 = np.quantile(values, [0.50, 0.75, 0.90])
    return (
        float(values.mean()),
        float(values.std()),
        float(q50),
        float(q75),
        float(q90),
        float(values.max()),
    )


def class_condition_features(
    args,
    bank: Dict[str, object],
    cls_name: Optional[str],
    tau_mem: float,
    exclude_index: Optional[int] = None,
) -> np.ndarray:
    if not bool(getattr(args, "cmct_enable_class_residual", False)):
        return np.zeros((0,), dtype=np.float32)

    memory_taus = np.asarray(bank["memory_taus"], dtype=np.float32)
    metas = list(bank.get("memory_meta", []))
    if memory_taus.size == 0 or not metas or not cls_name:
        return np.zeros((len(CMCT_CLASS_CONDITION_NAMES),), dtype=np.float32)

    valid = np.ones((memory_taus.shape[0],), dtype=bool)
    if exclude_index is not None and 0 <= int(exclude_index) < valid.shape[0]:
        valid[int(exclude_index)] = False
    valid_taus = memory_taus[valid]
    if valid_taus.size == 0:
        valid_taus = memory_taus
        valid = np.ones((memory_taus.shape[0],), dtype=bool)

    allowed = _memory_class_aliases(str(cls_name))
    class_idx = np.asarray([
        i for i, meta in enumerate(metas)
        if valid[i] and str(meta.get("class_name", "")) in allowed
    ], dtype=np.int64)
    if class_idx.size < int(getattr(args, "cmct_class_residual_min_rows", 8)):
        class_taus = valid_taus
    else:
        class_taus = memory_taus[class_idx]

    class_mean, class_std, class_q50, class_q75, class_q90, class_max = _tau_distribution_features(class_taus)
    global_mean, _, _, global_q75, global_q90, _ = _tau_distribution_features(valid_taus)
    eps = 1e-6
    count_ratio = float(class_taus.size) / max(float(valid_taus.size), 1.0)
    return np.asarray([
        class_mean,
        class_std,
        class_q50,
        class_q75,
        class_q90,
        class_max,
        count_ratio,
        global_mean,
        global_q75,
        global_q90,
        float(tau_mem) - class_mean,
        float(tau_mem) - class_q75,
        float(tau_mem) - class_q90,
        class_mean / max(global_mean, eps),
        class_q90 / max(global_q90, eps),
    ], dtype=np.float32)


def cmct_correction_features(
    args,
    bank: Dict[str, object],
    stats: np.ndarray,
    tau_mem: float,
    retrieval_diag: Dict[str, float],
    cls_name: Optional[str] = None,
    exclude_index: Optional[int] = None,
    component_features: Optional[np.ndarray] = None,
    semantic_features: Optional[np.ndarray] = None,
) -> np.ndarray:
    base_features = correction_features(stats, tau_mem, retrieval_diag)
    class_features = class_condition_features(args, bank, cls_name, tau_mem, exclude_index=exclude_index)
    rows = [base_features]
    if class_features.size > 0:
        rows.append(class_features)
    if bool(getattr(args, "cmct_enable_component_reliability", False)):
        if component_features is None:
            component_features = np.zeros((len(CMCT_COMPONENT_RELIABILITY_NAMES),), dtype=np.float32)
        rows.append(np.asarray(component_features, dtype=np.float32))
    if bool(getattr(args, "cmct_enable_semantic_relation", False)):
        if semantic_features is None:
            semantic_features = np.zeros((len(CMCT_SEMANTIC_RELATION_NAMES),), dtype=np.float32)
        rows.append(np.asarray(semantic_features, dtype=np.float32))
    return np.concatenate(rows, axis=0).astype(np.float32)


def cmct_correction_feature_names(args) -> List[str]:
    names = list(CMCT_STAT_NAMES) + ["tau_mem", "retrieval_top1_sim", "retrieval_mean_sim"]
    if bool(getattr(args, "cmct_enable_class_residual", False)):
        names += list(CMCT_CLASS_CONDITION_NAMES)
    if bool(getattr(args, "cmct_enable_component_reliability", False)):
        names += list(CMCT_COMPONENT_RELIABILITY_NAMES)
    if bool(getattr(args, "cmct_enable_semantic_relation", False)):
        names += list(CMCT_SEMANTIC_RELATION_NAMES)
    return names


def cmct_correction_feature_dim(args) -> int:
    return len(cmct_correction_feature_names(args))


def make_query(args, item, cls_name: str, prompt_bank, pair_embedding, clipper, score, bcd_score, presence):
    stats = score_clip_stats(score, bcd_score, presence)
    text_embedding = clipper.encode_text(prompt_text_for_class(cls_name, prompt_bank))
    key = build_memory_key(
        pair_embedding,
        text_embedding,
        stats,
        scene_weight=args.clip_memory_scene_weight,
        delta_weight=args.clip_memory_delta_weight,
        text_weight=args.clip_memory_text_weight,
        stats_weight=args.clip_memory_stats_weight,
    )
    return key, stats


def train_correction_head(args, bank: Dict[str, object], device):
    memory_keys = np.asarray(bank["memory_keys"], dtype=np.float32)
    memory_taus = np.asarray(bank["memory_taus"], dtype=np.float32)
    memory_stats = np.asarray(bank["memory_stats"], dtype=np.float32)
    memory_component_features = np.asarray(
        bank.get("memory_component_features", np.zeros((memory_taus.shape[0], 0), dtype=np.float32)),
        dtype=np.float32,
    )
    memory_meta = list(bank.get("memory_meta", []))
    input_dim = cmct_correction_feature_dim(args)
    model = CorrectionHead(
        input_dim,
        hidden_dim=args.clip_corrector_hidden_dim,
        dropout=args.clip_corrector_dropout,
        max_delta=args.clip_corrector_max_delta,
    ).to(device)
    if memory_taus.size <= 1:
        return model, {"num_rows": int(memory_taus.size), "trained": False}

    x_rows = []
    y_rows = []
    for idx in range(memory_taus.shape[0]):
        tau_mem, diag = retrieve_tau(
            memory_keys[idx],
            memory_keys,
            memory_taus,
            topk=args.clip_memory_topk,
            temperature=args.clip_memory_temperature,
            exclude_index=idx,
        )
        cls_name = None
        if idx < len(memory_meta):
            cls_name = str(memory_meta[idx].get("class_name", ""))
        x_rows.append(cmct_correction_features(
            args,
            bank,
            memory_stats[idx],
            tau_mem,
            diag,
            cls_name=cls_name,
            exclude_index=idx,
            component_features=memory_component_features[idx] if memory_component_features.shape[1] > 0 else None,
            semantic_features=semantic_relation_features_for_class(args, bank, cls_name),
        ))
        y_rows.append(float(memory_taus[idx] - tau_mem))
    x = torch.tensor(np.stack(x_rows, axis=0), dtype=torch.float32, device=device)
    y = torch.tensor(np.asarray(y_rows, dtype=np.float32), dtype=torch.float32, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.clip_corrector_lr, weight_decay=args.clip_corrector_weight_decay)
    for _ in range(int(args.clip_corrector_epochs)):
        model.train()
        perm = torch.randperm(x.shape[0], device=device)
        for start in range(0, x.shape[0], int(args.clip_corrector_batch_size)):
            idx = perm[start:start + int(args.clip_corrector_batch_size)]
            pred = model(x[idx])
            loss = F.smooth_l1_loss(pred, y[idx])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    model.eval()
    with torch.no_grad():
        pred = model(x)
        loss = F.smooth_l1_loss(pred, y)
    return model, {
        "num_rows": int(memory_taus.size),
        "trained": True,
        "target_delta_mean": float(y.mean().detach().cpu()),
        "target_delta_min": float(y.min().detach().cpu()),
        "target_delta_max": float(y.max().detach().cpu()),
        "pred_delta_mean": float(pred.mean().detach().cpu()),
        "smooth_l1": float(loss.detach().cpu()),
        "class_conditioned": bool(getattr(args, "cmct_enable_class_residual", False)),
        "input_dim": int(input_dim),
    }


def _memory_class_aliases(cls_name: str) -> set:
    name = str(cls_name)
    aliases = {
        "Bareland": {"Bareland", "Ground"},
        "Ground": {"Ground", "Bareland"},
        "Cropland": {"Cropland", "Farmland"},
        "Farmland": {"Farmland", "Cropland"},
        "Grass": {"Grass", "Vegetation"},
        "Vegetation": {"Vegetation", "Grass", "Tree"},
        "Structure": {"Structure", "Building"},
    }
    return set(aliases.get(name, {name}))


def _class_filtered_memory(args, bank, cls_name: Optional[str]):
    memory_keys = np.asarray(bank["memory_keys"], dtype=np.float32)
    memory_taus = np.asarray(bank["memory_taus"], dtype=np.float32)
    memory_ratios = np.asarray(bank.get("memory_ratios", np.zeros_like(memory_taus)), dtype=np.float32)
    mode = str(getattr(args, "clip_memory_class_filter", "none"))
    diag = {
        "class_filter": mode,
        "candidate_rows": int(memory_taus.shape[0]),
        "class_filtered": 0,
    }
    if mode == "none" or not cls_name:
        return memory_keys, memory_taus, memory_ratios, diag

    metas = list(bank.get("memory_meta", []))
    if not metas:
        return memory_keys, memory_taus, memory_ratios, diag
    if mode == "exact":
        allowed = {str(cls_name)}
    elif mode == "alias":
        allowed = _memory_class_aliases(str(cls_name))
    else:
        raise ValueError(f"Unknown clip_memory_class_filter: {mode}")
    idx = [
        i for i, meta in enumerate(metas)
        if str(meta.get("class_name", "")) in allowed
    ]
    if len(idx) < int(args.clip_memory_class_filter_min_rows):
        diag["class_filter_fallback"] = 1
        diag["matched_rows"] = int(len(idx))
        return memory_keys, memory_taus, memory_ratios, diag
    idx_np = np.asarray(idx, dtype=np.int64)
    diag["candidate_rows"] = int(idx_np.shape[0])
    diag["matched_rows"] = int(idx_np.shape[0])
    diag["class_filtered"] = 1
    return memory_keys[idx_np], memory_taus[idx_np], memory_ratios[idx_np], diag


@torch.no_grad()
def predict_cmct_thresholds(
    args,
    bank,
    correction_model,
    key: np.ndarray,
    stats: np.ndarray,
    device,
    cls_name: Optional[str] = None,
    component_features: Optional[np.ndarray] = None,
    semantic_features: Optional[np.ndarray] = None,
):
    memory_keys, memory_taus, memory_ratios, filter_diag = _class_filtered_memory(args, bank, cls_name)
    tau_mem, diag = retrieve_tau(
        key,
        memory_keys,
        memory_taus,
        topk=args.clip_memory_topk,
        temperature=args.clip_memory_temperature,
    )
    ratio_mem = None
    if bool(getattr(args, "cmct_enable_memory_ratio_budget", False)) and memory_ratios.size == memory_taus.size and memory_ratios.size > 0:
        ratio_mem, ratio_diag = retrieve_tau(
            key,
            memory_keys,
            memory_ratios,
            topk=args.clip_memory_topk,
            temperature=args.clip_memory_temperature,
        )
        diag["ratio_mem"] = float(ratio_mem)
        diag["ratio_top1_sim"] = float(ratio_diag.get("top1_sim", 0.0))
    diag.update(filter_diag)
    tau_corr = tau_mem
    if correction_model is not None:
        x = torch.tensor(
            cmct_correction_features(
                args,
                bank,
                stats,
                tau_mem,
                diag,
                cls_name=cls_name,
                component_features=component_features,
                semantic_features=semantic_features,
            ),
            dtype=torch.float32,
            device=device,
        ).view(1, -1)
        delta = float(correction_model(x).detach().cpu().reshape(()))
        raw_delta = float(delta)
        if bool(getattr(args, "cmct_semantic_delta_confidence_gate", False)) and semantic_features is not None:
            semantic_conf = _semantic_relation_confidence(args, semantic_features)
            min_gate = float(np.clip(getattr(args, "cmct_semantic_delta_min_gate", 0.35), 0.0, 1.0))
            delta_gate = min_gate + (1.0 - min_gate) * float(np.clip(semantic_conf, 0.0, 1.0))
            delta *= float(delta_gate)
            diag["raw_delta_tau"] = float(raw_delta)
            diag["semantic_delta_gate"] = float(delta_gate)
            diag["semantic_delta_confidence"] = float(semantic_conf)
        tau_corr = tau_mem + delta
        diag["delta_tau"] = float(delta)
    tau_mem = float(np.clip(tau_mem, float(args.cmct_threshold_min), float(args.cmct_threshold_max)))
    tau_corr = float(np.clip(tau_corr, float(args.cmct_threshold_min), float(args.cmct_threshold_max)))
    return tau_mem, tau_corr, diag


def apply_adaptive_floor(tau: float, adaptive_threshold: float, args) -> float:
    if float(args.cmct_adaptive_floor_scale) <= 0:
        return float(tau)
    floor = float(adaptive_threshold) * float(args.cmct_adaptive_floor_scale)
    return float(np.clip(max(float(tau), floor), float(args.cmct_threshold_min), float(args.cmct_threshold_max)))


def apply_class_prior_floor(tau: float, args, bank, cls_name: Optional[str]) -> float:
    quantile = float(getattr(args, "cmct_class_prior_floor_quantile", 0.0))
    scale = float(getattr(args, "cmct_class_prior_floor_scale", 1.0))
    if quantile <= 0.0 or scale <= 0.0 or not cls_name:
        return float(tau)
    metas = list(bank.get("memory_meta", []))
    if not metas:
        return float(tau)
    allowed = _memory_class_aliases(str(cls_name))
    idx = [
        i for i, meta in enumerate(metas)
        if str(meta.get("class_name", "")) in allowed
    ]
    if len(idx) < int(getattr(args, "cmct_class_prior_floor_min_rows", 8)):
        return float(tau)
    memory_taus = np.asarray(bank["memory_taus"], dtype=np.float32)
    floor = float(np.quantile(memory_taus[np.asarray(idx, dtype=np.int64)], quantile)) * scale
    return float(np.clip(max(float(tau), floor), float(args.cmct_threshold_min), float(args.cmct_threshold_max)))


def _semantic_prior_stat_value(
    args,
    semantic_features: Optional[np.ndarray],
    stat: Optional[str] = None,
) -> float:
    if semantic_features is None:
        return 0.0
    values = np.asarray(semantic_features, dtype=np.float32)
    if values.size == 0:
        return 0.0
    mode = str(stat if stat is not None else getattr(args, "cmct_semantic_prior_floor_stat", "weighted_q90"))
    weighted_q75 = _feature_value_by_name(
        CMCT_SEMANTIC_RELATION_NAMES,
        values,
        "semantic_weighted_tau_q75",
        0.0,
    )
    weighted_q50 = _feature_value_by_name(
        CMCT_SEMANTIC_RELATION_NAMES,
        values,
        "semantic_weighted_tau_q50",
        0.0,
    )
    weighted_q90 = _feature_value_by_name(
        CMCT_SEMANTIC_RELATION_NAMES,
        values,
        "semantic_weighted_tau_q90",
        0.0,
    )
    top1_q50 = _feature_value_by_name(
        CMCT_SEMANTIC_RELATION_NAMES,
        values,
        "semantic_top1_tau_q50",
        0.0,
    )
    top1_q75 = _feature_value_by_name(
        CMCT_SEMANTIC_RELATION_NAMES,
        values,
        "semantic_top1_tau_q75",
        0.0,
    )
    top1_q90 = _feature_value_by_name(
        CMCT_SEMANTIC_RELATION_NAMES,
        values,
        "semantic_top1_tau_q90",
        0.0,
    )
    if mode == "weighted_q50":
        return float(weighted_q50)
    if mode == "weighted_q75":
        return float(weighted_q75)
    if mode == "top1_q75":
        return float(top1_q75)
    if mode == "top1_q50":
        return float(top1_q50)
    if mode == "top1_q90":
        return float(top1_q90)
    if mode == "mix_q50_q75":
        mix = float(getattr(args, "cmct_semantic_prior_floor_q90_mix", 0.50))
        mix = float(np.clip(mix, 0.0, 1.0))
        return float((1.0 - mix) * weighted_q50 + mix * weighted_q75)
    if mode == "mix_q75_q90":
        mix = float(getattr(args, "cmct_semantic_prior_floor_q90_mix", 0.50))
        mix = float(np.clip(mix, 0.0, 1.0))
        return float((1.0 - mix) * weighted_q75 + mix * weighted_q90)
    return float(weighted_q90)


def apply_semantic_prior_floor(
    tau: float,
    args,
    semantic_features: Optional[np.ndarray],
    stat: Optional[str] = None,
    scale: Optional[float] = None,
    strength: Optional[float] = None,
) -> Tuple[float, Dict[str, float]]:
    if not bool(getattr(args, "cmct_enable_semantic_prior_floor", False)):
        return float(tau), {"enabled": False}
    values = np.asarray(semantic_features if semantic_features is not None else [], dtype=np.float32)
    if values.size == 0:
        return float(tau), {"enabled": True, "applied": False, "reason": "empty_semantic_features"}

    top1_sim = _feature_value_by_name(CMCT_SEMANTIC_RELATION_NAMES, values, "semantic_top1_sim", 0.0)
    margin = _feature_value_by_name(CMCT_SEMANTIC_RELATION_NAMES, values, "semantic_top1_margin", 0.0)
    entropy = _feature_value_by_name(CMCT_SEMANTIC_RELATION_NAMES, values, "semantic_entropy", 1.0)
    min_sim = float(getattr(args, "cmct_semantic_prior_floor_min_top1_sim", -1.0))
    min_margin = float(getattr(args, "cmct_semantic_prior_floor_min_margin", -1.0))
    max_entropy = float(getattr(args, "cmct_semantic_prior_floor_max_entropy", 1.1))
    if top1_sim < min_sim or margin < min_margin or entropy > max_entropy:
        return float(tau), {
            "enabled": True,
            "applied": False,
            "reason": "semantic_confidence_gate",
            "top1_sim": float(top1_sim),
            "margin": float(margin),
            "entropy": float(entropy),
        }

    prior = _semantic_prior_stat_value(args, values, stat=stat)
    if prior <= 0.0:
        return float(tau), {"enabled": True, "applied": False, "reason": "nonpositive_prior"}
    scale_value = float(scale if scale is not None else getattr(args, "cmct_semantic_prior_floor_scale", 1.0))
    strength_value = float(strength if strength is not None else getattr(args, "cmct_semantic_prior_floor_strength", 1.0))
    strength_value = float(np.clip(strength_value, 0.0, 1.0))
    floor = float(prior) * scale_value
    updated = float(tau) + strength_value * max(0.0, floor - float(tau))
    updated = float(np.clip(updated, float(args.cmct_threshold_min), float(args.cmct_threshold_max)))
    return updated, {
        "enabled": True,
        "applied": bool(updated > float(tau) + 1e-8),
        "prior": float(prior),
        "floor": float(floor),
        "strength": float(strength_value),
        "scale": float(scale_value),
        "delta": float(updated - float(tau)),
        "top1_sim": float(top1_sim),
        "margin": float(margin),
        "entropy": float(entropy),
    }


def apply_semantic_prior_anchor(
    tau: float,
    args,
    semantic_features: Optional[np.ndarray],
) -> Tuple[float, Dict[str, float]]:
    if not bool(getattr(args, "cmct_enable_semantic_prior_anchor", False)):
        return float(tau), {"enabled": False}
    values = np.asarray(semantic_features if semantic_features is not None else [], dtype=np.float32)
    if values.size == 0:
        return float(tau), {"enabled": True, "applied": False, "reason": "empty_semantic_features"}

    top1_sim = _feature_value_by_name(CMCT_SEMANTIC_RELATION_NAMES, values, "semantic_top1_sim", 0.0)
    margin = _feature_value_by_name(CMCT_SEMANTIC_RELATION_NAMES, values, "semantic_top1_margin", 0.0)
    entropy = _feature_value_by_name(CMCT_SEMANTIC_RELATION_NAMES, values, "semantic_entropy", 1.0)
    min_sim = float(getattr(args, "cmct_semantic_prior_anchor_min_top1_sim", -1.0))
    min_margin = float(getattr(args, "cmct_semantic_prior_anchor_min_margin", -1.0))
    max_entropy = float(getattr(args, "cmct_semantic_prior_anchor_max_entropy", 1.1))
    if top1_sim < min_sim or margin < min_margin or entropy > max_entropy:
        return float(tau), {
            "enabled": True,
            "applied": False,
            "reason": "semantic_confidence_gate",
            "top1_sim": float(top1_sim),
            "margin": float(margin),
            "entropy": float(entropy),
        }

    stat = str(getattr(args, "cmct_semantic_prior_anchor_stat", "mix_q75_q90"))
    prior = _semantic_prior_stat_value(args, values, stat=stat)
    if prior <= 0.0:
        return float(tau), {"enabled": True, "applied": False, "reason": "nonpositive_prior"}

    scale = float(getattr(args, "cmct_semantic_prior_anchor_scale", 1.0))
    target = float(np.clip(
        float(prior) * scale,
        float(args.cmct_threshold_min),
        float(args.cmct_threshold_max),
    ))
    direction = str(getattr(args, "cmct_semantic_prior_anchor_direction", "both"))
    delta = float(target - float(tau))
    if direction == "lower" and delta >= 0.0:
        return float(tau), {
            "enabled": True,
            "applied": False,
            "reason": "not_lowering",
            "prior": float(prior),
            "target": float(target),
            "delta_to_anchor": float(delta),
        }
    if direction == "raise" and delta <= 0.0:
        return float(tau), {
            "enabled": True,
            "applied": False,
            "reason": "not_raising",
            "prior": float(prior),
            "target": float(target),
            "delta_to_anchor": float(delta),
        }

    confidence = _semantic_relation_confidence(args, values)
    confidence_power = max(float(getattr(args, "cmct_semantic_prior_anchor_confidence_power", 1.0)), 0.0)
    confidence = float(np.clip(confidence, 0.0, 1.0)) ** confidence_power
    min_gate = float(np.clip(getattr(args, "cmct_semantic_prior_anchor_min_gate", 0.0), 0.0, 1.0))
    gate = min_gate + (1.0 - min_gate) * confidence
    strength = float(np.clip(getattr(args, "cmct_semantic_prior_anchor_strength", 0.5), 0.0, 1.0))
    effective_strength = float(np.clip(strength * gate, 0.0, 1.0))
    updated = float(tau) + effective_strength * delta
    updated = float(np.clip(updated, float(args.cmct_threshold_min), float(args.cmct_threshold_max)))
    return updated, {
        "enabled": True,
        "applied": bool(abs(updated - float(tau)) > 1e-8),
        "prior": float(prior),
        "target": float(target),
        "delta_to_anchor": float(delta),
        "delta": float(updated - float(tau)),
        "strength": float(strength),
        "gate": float(gate),
        "effective_strength": float(effective_strength),
        "confidence": float(confidence),
        "top1_sim": float(top1_sim),
        "margin": float(margin),
        "entropy": float(entropy),
    }


def _component_local_block_classes(args) -> set:
    value = str(getattr(args, "cmct_component_local_block_classes", "") or "")
    return {x.strip() for x in value.split(",") if x.strip()}


def _component_local_class_blocked(args, cls_name: Optional[str]) -> bool:
    if not cls_name:
        return False
    blocked = _component_local_block_classes(args)
    if not blocked:
        return False
    aliases = _memory_class_aliases(str(cls_name))
    return bool(aliases & blocked)


def _component_bbox_yxyx(region: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    ys, xs = np.where(region.astype(bool))
    if ys.size == 0 or xs.size == 0:
        return None
    return int(ys.min()), int(xs.min()), int(ys.max()) + 1, int(xs.max()) + 1


def _component_clip_similarity_feature_values(
    args,
    region: np.ndarray,
    item: Optional[dict],
    clipper: Optional[ClipFeatureExtractor],
    class_text_embedding: Optional[np.ndarray],
    class_text_embeddings: Optional[np.ndarray],
    class_text_index: Optional[int] = None,
) -> np.ndarray:
    if item is None or clipper is None or class_text_embedding is None:
        return np.zeros((len(COMPONENT_VERIFIER_CLIP_FEATURE_NAMES),), dtype=np.float32)
    bbox = _component_bbox_yxyx(region)
    if bbox is None:
        return np.zeros((len(COMPONENT_VERIFIER_CLIP_FEATURE_NAMES),), dtype=np.float32)
    emb = clipper.encode_region_pair(
        item,
        bbox,
        region.shape,
        pad=int(getattr(args, "cmct_component_clip_pad", 4)),
    )
    text = np.asarray(class_text_embedding, dtype=np.float32)
    text = text / max(float(np.linalg.norm(text)), 1e-8)
    all_text = None
    if class_text_embeddings is not None:
        all_text = np.asarray(class_text_embeddings, dtype=np.float32)
        if all_text.ndim == 2 and all_text.shape[0] > 1 and all_text.shape[1] == text.shape[0]:
            all_text = all_text / np.maximum(np.linalg.norm(all_text, axis=1, keepdims=True), 1e-8)
        else:
            all_text = None

    sims = []
    margins = []
    for image_embedding in (emb.before, emb.after, emb.scene):
        image_embedding = np.asarray(image_embedding, dtype=np.float32)
        image_embedding = image_embedding / max(float(np.linalg.norm(image_embedding)), 1e-8)
        sim = float(np.dot(image_embedding, text))
        margin = 0.0
        if all_text is not None:
            all_sims = np.matmul(all_text, image_embedding)
            if class_text_index is not None and 0 <= int(class_text_index) < all_sims.shape[0]:
                current_index = int(class_text_index)
                sim = float(all_sims[current_index])
                other = np.delete(all_sims, current_index)
                other_max = float(other.max()) if other.size > 0 else sim
            else:
                order = np.argsort(-all_sims)
                other_max = float(all_sims[order[1]]) if order.size > 1 else sim
            margin = float(sim - other_max)
        sims.append(sim)
        margins.append(margin)

    before_sim, after_sim, scene_sim = sims
    before_margin, after_margin, scene_margin = margins
    best_index = int(np.argmax(np.asarray(sims, dtype=np.float32)))
    best_state_sim = float(sims[best_index])
    best_margin = float(margins[best_index])
    return np.asarray([
        float(before_sim),
        float(after_sim),
        float(scene_sim),
        best_state_sim,
        float(before_margin),
        float(after_margin),
        float(scene_margin),
        best_margin,
        float(after_sim - before_sim),
        float(scene_sim - before_sim),
    ], dtype=np.float32)


def component_verifier_feature_row(
    args,
    base_row: Sequence[float],
    region: np.ndarray,
    item: Optional[dict],
    clipper: Optional[ClipFeatureExtractor],
    class_text_embedding: Optional[np.ndarray],
    class_text_embeddings: Optional[np.ndarray],
    class_text_index: Optional[int] = None,
) -> np.ndarray:
    clip_features = _component_clip_similarity_feature_values(
        args,
        region,
        item,
        clipper,
        class_text_embedding,
        class_text_embeddings,
        class_text_index=class_text_index,
    )
    return np.concatenate([np.asarray(base_row, dtype=np.float32), clip_features], axis=0).astype(np.float32)


def dense_semantic_delta_np(posterior_a: torch.Tensor, posterior_b: torch.Tensor, cls_idx: int) -> np.ndarray:
    """Competition-aware dense posterior delta for one class."""

    pa = posterior_a.detach().float().cpu()
    pb = posterior_b.detach().float().cpu()
    idx = int(cls_idx)
    p1_cls = pa[idx].numpy().astype(np.float32)
    p2_cls = pb[idx].numpy().astype(np.float32)
    if pa.shape[0] > 1:
        other = [i for i in range(int(pa.shape[0])) if i != idx]
        p1_neg = torch.max(pa[other], dim=0).values.numpy().astype(np.float32)
        p2_neg = torch.max(pb[other], dim=0).values.numpy().astype(np.float32)
    else:
        p1_neg = np.zeros_like(p1_cls, dtype=np.float32)
        p2_neg = np.zeros_like(p2_cls, dtype=np.float32)
    p1_cal = np.clip(p1_cls - p1_neg, 0.0, 1.0)
    p2_cal = np.clip(p2_cls - p2_neg, 0.0, 1.0)
    return np.maximum(p2_cal * (1.0 - p1_cal), p1_cal * (1.0 - p2_cal)).astype(np.float32)


def prat_region_extra_feature_values(
    region: np.ndarray,
    semantic_delta_np: Optional[np.ndarray],
    structure_support_np: Optional[np.ndarray],
    score_support_np: Optional[np.ndarray],
) -> np.ndarray:
    region = np.asarray(region).astype(bool)
    if (
        semantic_delta_np is None
        or structure_support_np is None
        or score_support_np is None
        or not region.any()
    ):
        return np.zeros((len(PRAT_REGION_EXTRA_FEATURE_NAMES),), dtype=np.float32)

    sem = np.asarray(semantic_delta_np, dtype=np.float32)[region]
    struct = np.asarray(structure_support_np, dtype=np.float32)[region]
    score_sup = np.asarray(score_support_np, dtype=np.float32)[region]
    product = sem * struct
    area_ratio = float(region.mean())
    values = [
        float(sem.mean()),
        float(np.quantile(sem, 0.75)),
        float(np.quantile(sem, 0.90)),
        float(sem.max()),
        float(struct.mean()),
        float(np.quantile(struct, 0.75)),
        float(np.quantile(struct, 0.90)),
        float(struct.max()),
        float(score_sup.mean()),
        float(np.quantile(score_sup, 0.75)),
        float(product.mean()),
        float(np.quantile(product, 0.75)),
        area_ratio,
    ]
    return np.asarray(values, dtype=np.float32)


def suppressed_recovery_feature_row(
    base_row: Sequence[float],
    region: np.ndarray,
    corr_bool: np.ndarray,
    gate_bool: np.ndarray,
    feature_score_np: np.ndarray,
    bcd_np: np.ndarray,
    feature_score_threshold: float,
    bcd_threshold_value: float,
    keep_prob: float,
    tau_corr: float,
    relaxed_tau: float,
    prat_region_features: Optional[np.ndarray] = None,
) -> np.ndarray:
    region = region.astype(bool)
    corr_bool = corr_bool.astype(bool)
    gate_bool = gate_bool.astype(bool)
    area = float(region.sum())
    image_area = float(region.size)
    if area <= 0:
        extra = np.zeros((len(SUPPRESSED_RECOVERY_EXTRA_FEATURE_NAMES),), dtype=np.float32)
        return np.concatenate([np.asarray(base_row, dtype=np.float32), extra], axis=0).astype(np.float32)
    suppressed = region & ~gate_bool
    suppressed_area = float(suppressed.sum())
    if suppressed_area > 0:
        score_lost = float((feature_score_np[suppressed] > float(feature_score_threshold)).mean())
        bcd_lost = float((bcd_np[suppressed] > float(bcd_threshold_value)).mean())
    else:
        score_lost = 0.0
        bcd_lost = 0.0
    extra = np.asarray([
        float((region & corr_bool).sum()) / max(area, 1.0),
        float((region & gate_bool).sum()) / max(area, 1.0),
        suppressed_area / max(area, 1.0),
        suppressed_area / max(image_area, 1.0),
        float(keep_prob),
        float(tau_corr),
        float(relaxed_tau),
        float(tau_corr) - float(relaxed_tau),
        score_lost,
        bcd_lost,
    ], dtype=np.float32)
    parts = [np.asarray(base_row, dtype=np.float32), extra]
    if prat_region_features is not None:
        parts.append(np.asarray(prat_region_features, dtype=np.float32))
    return np.concatenate(parts, axis=0).astype(np.float32)


def _component_clip_verify(
    args,
    region: np.ndarray,
    item: Optional[dict],
    clipper: Optional[ClipFeatureExtractor],
    class_text_embedding: Optional[np.ndarray],
    class_text_embeddings: Optional[np.ndarray],
    class_text_index: Optional[int] = None,
) -> Tuple[bool, Dict[str, float]]:
    if not bool(getattr(args, "cmct_enable_component_clip_verify", False)):
        return True, {"enabled": False}
    if item is None or clipper is None or class_text_embedding is None:
        return True, {"enabled": True, "applied": False, "reason": "missing_clip_context"}
    area = int(region.astype(bool).sum())
    min_area = int(getattr(args, "cmct_component_clip_min_area", 64))
    if area < min_area:
        return True, {"enabled": True, "applied": False, "reason": "small_component", "area": float(area)}
    bbox = _component_bbox_yxyx(region)
    if bbox is None:
        return False, {"enabled": True, "applied": True, "reason": "empty_bbox"}

    emb = clipper.encode_region_pair(
        item,
        bbox,
        region.shape,
        pad=int(getattr(args, "cmct_component_clip_pad", 4)),
    )
    text = np.asarray(class_text_embedding, dtype=np.float32)
    text = text / max(float(np.linalg.norm(text)), 1e-8)
    image_embeddings = [emb.before, emb.after, emb.scene]
    if str(getattr(args, "cmct_component_clip_embedding", "max_state")) == "scene":
        image_embeddings = [emb.scene]
    elif str(getattr(args, "cmct_component_clip_embedding", "max_state")) == "after":
        image_embeddings = [emb.after]

    sims = []
    margins = []
    all_text = None
    if class_text_embeddings is not None:
        all_text = np.asarray(class_text_embeddings, dtype=np.float32)
        if all_text.ndim == 2 and all_text.shape[0] > 1 and all_text.shape[1] == text.shape[0]:
            all_text = all_text / np.maximum(np.linalg.norm(all_text, axis=1, keepdims=True), 1e-8)
        else:
            all_text = None
    for image_embedding in image_embeddings:
        image_embedding = np.asarray(image_embedding, dtype=np.float32)
        image_embedding = image_embedding / max(float(np.linalg.norm(image_embedding)), 1e-8)
        sim = float(np.dot(image_embedding, text))
        sims.append(sim)
        if all_text is not None:
            all_sims = np.matmul(all_text, image_embedding)
            if class_text_index is not None and 0 <= int(class_text_index) < all_sims.shape[0]:
                current_index = int(class_text_index)
                sim = float(all_sims[current_index])
                sims[-1] = sim
                other = np.delete(all_sims, current_index)
                other_max = float(other.max()) if other.size > 0 else sim
            else:
                order = np.argsort(-all_sims)
                other_max = float(all_sims[order[1]]) if order.size > 1 else sim
            margins.append(sim - other_max)
    best_sim = float(max(sims)) if sims else 0.0
    best_margin = float(max(margins)) if margins else 0.0
    min_sim = float(getattr(args, "cmct_component_clip_min_sim", -1.0))
    min_margin = float(getattr(args, "cmct_component_clip_min_margin", -1.0))
    passed = bool(best_sim >= min_sim and best_margin >= min_margin)
    return passed, {
        "enabled": True,
        "applied": True,
        "passed": passed,
        "sim": best_sim,
        "margin": best_margin,
        "area": float(area),
    }


@torch.no_grad()
def component_reliability_features(
    args,
    pred_mask: np.ndarray,
    score: torch.Tensor,
    bcd_score: torch.Tensor,
    score_threshold: float,
    component_model,
    device,
) -> np.ndarray:
    if not bool(getattr(args, "cmct_enable_component_reliability", False)):
        return np.zeros((0,), dtype=np.float32)

    pred_bool = pred_mask.astype(bool)
    if not pred_bool.any():
        return np.zeros((len(CMCT_COMPONENT_RELIABILITY_NAMES),), dtype=np.float32)

    bcd = bcd_score.float().clamp(0, 1)
    bcd_threshold = base.component_gate_threshold(bcd, args)
    bcd_threshold_value = float(bcd_threshold.detach().cpu())
    bcd_np = bcd.detach().cpu().numpy()
    score_np = score.detach().float().cpu().numpy()
    feature_score_np, feature_score_threshold = base.component_gate_feature_inputs(
        score_np,
        bcd_np,
        float(score_threshold),
        bcd_threshold_value,
        args,
    )

    labels_cc, num = base.connected_components(pred_bool)
    image_area = float(pred_bool.size)
    areas = []
    probs = []
    bcd_support = []
    score_margins = []
    keep_threshold = float(getattr(args, "component_gate_keep_threshold", 0.50))

    if component_model is not None:
        component_model.eval()
    for lab in range(1, num + 1):
        region = labels_cc == lab
        area_pixels = int(region.sum())
        if area_pixels < int(getattr(args, "component_gate_min_area", 8)):
            continue
        if int(getattr(args, "component_gate_max_area", 0)) > 0 and area_pixels > int(args.component_gate_max_area):
            continue
        row = base.component_gate_features(
            region,
            feature_score_np,
            bcd_np,
            feature_score_threshold,
            bcd_threshold_value,
            topk_ratio=float(args.component_gate_topk_ratio),
        )
        if component_model is None:
            keep_prob = 0.5
        else:
            x = torch.tensor(row, dtype=torch.float32, device=device).view(1, -1)
            keep_prob = float(torch.sigmoid(component_model(x)).detach().cpu().reshape(()))
        areas.append(float(area_pixels) / max(image_area, 1.0))
        probs.append(float(keep_prob))
        bcd_support.append(float(row[12]))
        score_margins.append(float(row[17]))

    if not areas:
        return np.zeros((len(CMCT_COMPONENT_RELIABILITY_NAMES),), dtype=np.float32)

    area_np = np.asarray(areas, dtype=np.float32)
    prob_np = np.asarray(probs, dtype=np.float32)
    total_area = float(area_np.sum())
    keep_mask = prob_np >= keep_threshold
    low_mask = ~keep_mask
    return np.asarray([
        float(np.log1p(len(areas))),
        total_area,
        float(area_np.max()),
        float(prob_np.mean()),
        float(prob_np.min()),
        float(prob_np.max()),
        float((prob_np * area_np).sum() / max(total_area, 1e-8)),
        float(area_np[keep_mask].sum()) if keep_mask.any() else 0.0,
        float(area_np[low_mask].sum()) if low_mask.any() else 0.0,
        float(keep_mask.mean()),
        float(np.mean(bcd_support)) if bcd_support else 0.0,
        float(np.mean(score_margins)) if score_margins else 0.0,
    ], dtype=np.float32)


@torch.no_grad()
def component_local_refine_mask(
    args,
    score: torch.Tensor,
    bcd_score: torch.Tensor,
    presence: torch.Tensor,
    tau_corr: float,
    component_model,
    device,
    cls_name: Optional[str] = None,
    item: Optional[dict] = None,
    clipper: Optional[ClipFeatureExtractor] = None,
    class_text_embedding: Optional[np.ndarray] = None,
    class_text_embeddings: Optional[np.ndarray] = None,
    class_text_index: Optional[int] = None,
    posterior_a: Optional[torch.Tensor] = None,
    posterior_b: Optional[torch.Tensor] = None,
    component_verifier_model=None,
    suppressed_recovery_model=None,
) -> Tuple[np.ndarray, float, Dict[str, float]]:
    corr_mask, corr_threshold = decode_fixed_tau(score, presence, tau_corr, args)
    if not bool(getattr(args, "cmct_enable_component_local_refine", False)) or component_model is None:
        return corr_mask.astype(np.int64), float(corr_threshold), {"enabled": False}
    if _component_local_class_blocked(args, cls_name):
        gate_mask = base.component_gate_filter(
            corr_mask,
            score,
            bcd_score,
            corr_threshold,
            args,
            component_model,
            device,
        )
        return gate_mask.astype(np.int64), float(corr_threshold), {
            "enabled": True,
            "class_blocked": True,
        }

    relax_scale = float(getattr(args, "cmct_component_local_relax_scale", 0.75))
    relax_delta = float(getattr(args, "cmct_component_local_relax_delta", 0.0))
    relaxed_tau = float(tau_corr) * relax_scale - relax_delta
    relaxed_tau = float(np.clip(relaxed_tau, float(args.cmct_threshold_min), float(args.cmct_threshold_max)))
    relaxed_mask, relaxed_threshold = decode_fixed_tau(score, presence, relaxed_tau, args)
    relaxed_bool = relaxed_mask.astype(bool)
    corr_bool = corr_mask.astype(bool)
    if not relaxed_bool.any():
        return corr_mask.astype(np.int64), float(corr_threshold), {
            "enabled": True,
            "relaxed_threshold": float(relaxed_threshold),
            "components": 0,
            "kept_components": 0,
            "added_components": 0,
            "added_area_ratio": 0.0,
        }

    bcd = bcd_score.float().clamp(0, 1)
    bcd_threshold = base.component_gate_threshold(bcd, args)
    bcd_threshold_value = float(bcd_threshold.detach().cpu())
    bcd_np = bcd.detach().cpu().numpy()
    score_np = score.detach().float().cpu().numpy()
    gate_mask = base.component_gate_filter(
        corr_mask,
        score,
        bcd_score,
        corr_threshold,
        args,
        component_model,
        device,
    )
    gate_bool = gate_mask.astype(bool)
    feature_score_np, feature_score_threshold = base.component_gate_feature_inputs(
        score_np,
        bcd_np,
        float(corr_threshold),
        bcd_threshold_value,
        args,
    )
    use_prat_region_features = bool(getattr(args, "cmct_suppressed_recovery_enable_prat_region_features", False))
    semantic_delta_np = None
    structure_support_np = None
    score_support_np = None
    if use_prat_region_features and posterior_a is not None and posterior_b is not None and class_text_index is not None:
        semantic_delta_np = dense_semantic_delta_np(posterior_a, posterior_b, int(class_text_index))
        denom = max(float(corr_threshold) - float(relaxed_threshold), 1e-6)
        score_support_np = np.clip((score_np - float(relaxed_threshold)) / denom, 0.0, 1.0).astype(np.float32)
        structure_support_np = (np.clip(bcd_np, 0.0, 1.0) * score_support_np).astype(np.float32)

    labels_cc, num = base.connected_components(relaxed_bool)
    out = np.zeros_like(relaxed_bool, dtype=bool)
    keep_threshold = float(getattr(args, "component_gate_keep_threshold", 0.50))
    add_threshold = float(getattr(args, "cmct_component_local_add_threshold", 0.70))
    expand_threshold = float(getattr(args, "cmct_component_local_expand_threshold", add_threshold))
    min_bcd_support = float(getattr(args, "cmct_component_local_min_bcd_support", 0.05))
    min_score_support = float(getattr(args, "cmct_component_local_min_score_support", 0.0))
    min_score_topk_margin = float(getattr(args, "cmct_component_local_min_score_topk_margin", -1.0))
    protect_area = float(getattr(args, "component_gate_protect_area_ratio", 0.0))
    image_area = float(relaxed_bool.size)
    kept_components = 0
    added_components = 0
    filtered_components = 0
    added_area = 0.0
    prob_sum = 0.0
    clip_checked = 0
    clip_rejected = 0
    clip_sim_sum = 0.0
    clip_margin_sum = 0.0
    verifier_checked = 0
    verifier_rejected = 0
    verifier_prob_sum = 0.0

    component_model.eval()
    if component_verifier_model is not None:
        component_verifier_model.eval()
    if suppressed_recovery_model is not None:
        suppressed_recovery_model.eval()
    suppressed_checked = 0
    suppressed_rejected = 0
    suppressed_prob_sum = 0.0
    for lab in range(1, num + 1):
        region = labels_cc == lab
        area = int(region.sum())
        if area < int(getattr(args, "component_gate_min_area", 8)):
            continue
        if int(getattr(args, "component_gate_max_area", 0)) > 0 and area > int(args.component_gate_max_area):
            continue

        corr_region = region & corr_bool
        corr_area = int(corr_region.sum())
        row = base.component_gate_features(
            region,
            feature_score_np,
            bcd_np,
            feature_score_threshold,
            bcd_threshold_value,
            topk_ratio=float(args.component_gate_topk_ratio),
        )
        x = torch.tensor(row, dtype=torch.float32, device=device).view(1, -1)
        keep_prob = float(torch.sigmoid(component_model(x)).detach().cpu().reshape(()))
        prob_sum += keep_prob

        area_ratio = float(area) / max(image_area, 1.0)
        protected = protect_area > 0 and area_ratio >= protect_area
        score_values = score_np[region]
        strong_semantic = _region_topk_mean_np(score_values, float(args.component_gate_topk_ratio)) >= (
            float(corr_threshold) * float(args.component_gate_strong_score_scale)
        )
        keep_existing = (
            corr_area > 0
            and (
                protected
                or keep_prob >= keep_threshold
                or (bool(args.component_gate_keep_strong_semantic) and strong_semantic)
            )
        )
        if keep_existing:
            out[corr_region] = True
            kept_components += 1

        can_add = (
            keep_prob >= add_threshold
            and float(row[12]) >= min_bcd_support
            and float(row[13]) >= min_score_support
            and float(row[17]) >= min_score_topk_margin
        )
        if corr_area > 0:
            can_add = can_add and keep_prob >= expand_threshold
        if can_add and bool(getattr(args, "cmct_enable_component_clip_verify", False)):
            clip_ok, clip_diag = _component_clip_verify(
                args,
                region,
                item,
                clipper,
                class_text_embedding,
                class_text_embeddings,
                class_text_index=class_text_index,
            )
            if bool(clip_diag.get("applied", False)):
                clip_checked += 1
                clip_sim_sum += float(clip_diag.get("sim", 0.0))
                clip_margin_sum += float(clip_diag.get("margin", 0.0))
                if not clip_ok:
                    clip_rejected += 1
            can_add = can_add and clip_ok
        if can_add and bool(getattr(args, "cmct_enable_component_verifier", False)) and component_verifier_model is not None:
            verifier_row = component_verifier_feature_row(
                args,
                row,
                region,
                item,
                clipper,
                class_text_embedding,
                class_text_embeddings,
                class_text_index=class_text_index,
            )
            x_verifier = torch.tensor(verifier_row, dtype=torch.float32, device=device).view(1, -1)
            verifier_prob = float(torch.sigmoid(component_verifier_model(x_verifier)).detach().cpu().reshape(()))
            verifier_checked += 1
            verifier_prob_sum += verifier_prob
            if verifier_prob < float(getattr(args, "cmct_component_verifier_keep_threshold", 0.50)):
                verifier_rejected += 1
                can_add = False
        if can_add and bool(getattr(args, "cmct_enable_suppressed_component_recovery", False)) and suppressed_recovery_model is not None:
            suppressed_region = region & ~gate_bool
            suppressed_pixels = int(suppressed_region.sum())
            suppressed_ratio = float(suppressed_pixels) / max(float(area), 1.0)
            min_pixels = int(getattr(args, "cmct_suppressed_recovery_min_suppressed_pixels", 8))
            min_ratio = float(getattr(args, "cmct_suppressed_recovery_min_suppressed_ratio", 0.05))
            if suppressed_pixels < min_pixels or suppressed_ratio < min_ratio:
                suppressed_rejected += 1
                can_add = False
            else:
                recovery_row = suppressed_recovery_feature_row(
                    row,
                    region,
                    corr_bool,
                    gate_bool,
                    feature_score_np,
                    bcd_np,
                    feature_score_threshold,
                    bcd_threshold_value,
                    keep_prob,
                    float(tau_corr),
                    float(relaxed_tau),
                    prat_region_features=prat_region_extra_feature_values(
                        region,
                        semantic_delta_np,
                        structure_support_np,
                        score_support_np,
                    ) if use_prat_region_features else None,
                )
                if bool(getattr(args, "cmct_suppressed_recovery_enable_clip_features", False)):
                    clip_features = _component_clip_similarity_feature_values(
                        args,
                        region,
                        item,
                        clipper,
                        class_text_embedding,
                        class_text_embeddings,
                        class_text_index=class_text_index,
                    )
                    recovery_row = np.concatenate([recovery_row, clip_features], axis=0).astype(np.float32)
                x_recovery = torch.tensor(recovery_row, dtype=torch.float32, device=device).view(1, -1)
                recovery_prob = float(torch.sigmoid(suppressed_recovery_model(x_recovery)).detach().cpu().reshape(()))
                suppressed_checked += 1
                suppressed_prob_sum += recovery_prob
                if recovery_prob < float(getattr(args, "cmct_suppressed_recovery_keep_threshold", 0.50)):
                    suppressed_rejected += 1
                    can_add = False
        if can_add:
            before = int(out.sum())
            out[region] = True
            added = int(out.sum()) - before
            if added > 0:
                added_components += 1
                added_area += float(added)

        if not keep_existing and not can_add:
            filtered_components += 1

    return out.astype(np.int64), float(corr_threshold), {
        "enabled": True,
        "relaxed_threshold": float(relaxed_threshold),
        "components": int(num),
        "kept_components": int(kept_components),
        "added_components": int(added_components),
        "filtered_components": int(filtered_components),
        "added_area_ratio": float(added_area / max(image_area, 1.0)),
        "mean_keep_prob": float(prob_sum / max(float(num), 1.0)),
        "clip_checked_components": int(clip_checked),
        "clip_rejected_components": int(clip_rejected),
        "clip_mean_sim": float(clip_sim_sum / max(float(clip_checked), 1.0)),
        "clip_mean_margin": float(clip_margin_sum / max(float(clip_checked), 1.0)),
        "verifier_checked_components": int(verifier_checked),
        "verifier_rejected_components": int(verifier_rejected),
        "verifier_mean_prob": float(verifier_prob_sum / max(float(verifier_checked), 1.0)),
        "suppressed_recovery_checked_components": int(suppressed_checked),
        "suppressed_recovery_rejected_components": int(suppressed_rejected),
        "suppressed_recovery_mean_prob": float(suppressed_prob_sum / max(float(suppressed_checked), 1.0)),
    }


def _region_topk_mean_np(values: np.ndarray, topk_ratio: float) -> float:
    values = np.asarray(values, dtype=np.float32)
    if values.size == 0:
        return 0.0
    k = max(1, int(values.size * float(topk_ratio)))
    return float(np.partition(values, -k)[-k:].mean())


def seg2change_support_masks_from_scores(
    args,
    class_names: Sequence[str],
    scores_a: torch.Tensor,
    scores_b: torch.Tensor,
    bcd_mask_np: np.ndarray,
) -> Tuple[Dict[str, np.ndarray], Dict[str, float]]:
    """Build Seg2Change-style endpoint support from existing SAM3 class scores."""

    if not class_names:
        return {}, {"enabled": True, "reason": "no_classes"}
    score_a_np = scores_a.detach().float().cpu().numpy().astype(np.float32)
    score_b_np = scores_b.detach().float().cpu().numpy().astype(np.float32)
    bcd_bool = np.asarray(bcd_mask_np).astype(bool)
    mode = str(getattr(args, "cmct_seg2change_support_mode", "hard_argmax"))
    conf_th = float(getattr(args, "cmct_seg2change_confidence_threshold", 0.10))
    use_bcd = not bool(getattr(args, "cmct_seg2change_disable_bcd_support", False))

    support = {}
    if mode == "soft_score":
        for idx, cls_name in enumerate(class_names):
            mask = (score_a_np[idx] >= conf_th) | (score_b_np[idx] >= conf_th)
            if use_bcd:
                mask &= bcd_bool
            support[cls_name] = mask.astype(np.int64)
    else:
        conf_a = score_a_np.max(axis=0)
        conf_b = score_b_np.max(axis=0)
        idx_a = score_a_np.argmax(axis=0)
        idx_b = score_b_np.argmax(axis=0)
        for idx, cls_name in enumerate(class_names):
            mask = ((idx_a == idx) & (conf_a >= conf_th)) | ((idx_b == idx) & (conf_b >= conf_th))
            if use_bcd:
                mask &= bcd_bool
            support[cls_name] = mask.astype(np.int64)

    area_ratios = [float(np.asarray(support[cls]).astype(bool).mean()) for cls in class_names]
    return support, {
        "enabled": True,
        "mode_hard_argmax": float(mode == "hard_argmax"),
        "mode_soft_score": float(mode == "soft_score"),
        "confidence_threshold": float(conf_th),
        "bcd_support_enabled": float(use_bcd),
        "bcd_area_ratio": float(bcd_bool.mean()),
        "support_area_ratio_mean": float(np.mean(area_ratios)) if area_ratios else 0.0,
        "support_area_ratio_max": float(np.max(area_ratios)) if area_ratios else 0.0,
    }


def seg2change_complement_component_gate(
    args,
    current_mask: np.ndarray,
    score: torch.Tensor,
    bcd_score: torch.Tensor,
    corr_threshold: float,
    support_mask: np.ndarray,
    component_model,
    device,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """Recover near-threshold CMCT components when Seg2Change endpoint evidence supports them."""

    out = np.asarray(current_mask).astype(bool).copy()
    support = np.asarray(support_mask).astype(bool)
    if out.shape != support.shape:
        raise ValueError(f"Seg2Change support shape {support.shape} does not match mask shape {out.shape}")

    score_np = score.detach().float().cpu().numpy().astype(np.float32)
    bcd_np = bcd_score.detach().float().cpu().numpy().astype(np.float32)
    image_area = float(max(out.size, 1))
    margin = max(float(getattr(args, "cmct_seg2change_recovery_tau_margin", 0.03)), 0.0)
    relaxed_threshold = max(float(corr_threshold) - margin, 0.0)
    min_bcd = float(getattr(args, "cmct_seg2change_recovery_min_bcd", 0.0))
    candidate = support & (score_np >= relaxed_threshold)
    if min_bcd > 0:
        candidate &= bcd_np >= min_bcd
    recovery = candidate & ~out
    raw_recovery_pixels = int(recovery.sum())

    max_area_ratio = float(getattr(args, "cmct_seg2change_recovery_max_area_ratio", 0.0))
    if max_area_ratio > 0 and raw_recovery_pixels > 0:
        max_pixels = max(1, int(round(max_area_ratio * image_area)))
        if raw_recovery_pixels > max_pixels:
            values = score_np[recovery]
            cutoff = float(np.partition(values, -max_pixels)[-max_pixels])
            recovery &= score_np >= cutoff

    capped_recovery_pixels = int(recovery.sum())
    use_gate = (
        component_model is not None
        and not bool(getattr(args, "cmct_seg2change_recovery_disable_component_gate", False))
    )
    recovery_gate_threshold = float(getattr(args, "cmct_seg2change_recovery_gate_keep_threshold", -1.0))
    if recovery.any() and use_gate:
        gate_args = args
        if recovery_gate_threshold >= 0:
            gate_args = argparse.Namespace(**vars(args))
            gate_args.component_gate_keep_threshold = recovery_gate_threshold
        recovery = base.component_gate_filter(
            recovery.astype(np.int64),
            score,
            bcd_score,
            relaxed_threshold,
            gate_args,
            component_model,
            device,
        ).astype(bool)
    gated_recovery_pixels = int(recovery.sum())

    before = int(out.sum())
    out |= recovery
    added_pixels = int(out.sum()) - before
    suppressed_pixels = 0
    if bool(getattr(args, "cmct_seg2change_enable_suppression", False)):
        suppress_margin = max(float(getattr(args, "cmct_seg2change_suppress_tau_margin", 0.01)), 0.0)
        suppress_bcd = float(getattr(args, "cmct_seg2change_suppress_max_bcd", 0.20))
        suppress = out & ~support & (score_np <= float(corr_threshold) + suppress_margin) & (bcd_np <= suppress_bcd)
        suppressed_pixels = int(suppress.sum())
        out[suppress] = False

    return out.astype(np.int64), {
        "enabled": True,
        "support_area_ratio": float(support.mean()),
        "current_area_ratio": float(np.asarray(current_mask).astype(bool).mean()),
        "relaxed_threshold": float(relaxed_threshold),
        "threshold": float(corr_threshold),
        "recovery_margin": float(margin),
        "recovery_raw_area_ratio": float(raw_recovery_pixels / image_area),
        "recovery_capped_area_ratio": float(capped_recovery_pixels / image_area),
        "recovery_added_area_ratio": float(added_pixels / image_area),
        "recovery_gate_keep_ratio": float(gated_recovery_pixels / max(float(capped_recovery_pixels), 1.0)),
        "recovery_component_gate_enabled": float(use_gate),
        "recovery_gate_keep_threshold": float(
            recovery_gate_threshold if use_gate else -1.0
        ),
        "suppressed_area_ratio": float(suppressed_pixels / image_area),
        "final_area_ratio": float(out.mean()),
    }


def prat_lite_component_gate(
    args,
    current_mask: np.ndarray,
    score: torch.Tensor,
    bcd_score: torch.Tensor,
    corr_threshold: float,
    posterior_a: torch.Tensor,
    posterior_b: torch.Tensor,
    cls_idx: int,
    component_model,
    device,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """Recover near-threshold components with dense posterior and structure support.

    This is intentionally source-free at inference time: it does not use target
    labels, and it preserves the current component-gated mask unless a missing
    region has semantic endpoint evidence and BCD/structure support.
    """

    out = np.asarray(current_mask).astype(bool).copy()
    score_np = score.detach().float().cpu().numpy().astype(np.float32)
    bcd_np = bcd_score.detach().float().clamp(0, 1).cpu().numpy().astype(np.float32)
    image_area = float(max(out.size, 1))

    pa = posterior_a.detach().float().cpu()
    pb = posterior_b.detach().float().cpu()
    p1_cls = pa[int(cls_idx)].numpy().astype(np.float32)
    p2_cls = pb[int(cls_idx)].numpy().astype(np.float32)
    if pa.shape[0] > 1:
        other = [i for i in range(int(pa.shape[0])) if i != int(cls_idx)]
        p1_neg = torch.max(pa[other], dim=0).values.numpy().astype(np.float32)
        p2_neg = torch.max(pb[other], dim=0).values.numpy().astype(np.float32)
    else:
        p1_neg = np.zeros_like(p1_cls, dtype=np.float32)
        p2_neg = np.zeros_like(p2_cls, dtype=np.float32)

    # Competition-aware dense posterior delta, following the CoRegOVCD spirit.
    p1_cal = np.clip(p1_cls - p1_neg, 0.0, 1.0)
    p2_cal = np.clip(p2_cls - p2_neg, 0.0, 1.0)
    semantic_delta = np.maximum(p2_cal * (1.0 - p1_cal), p1_cal * (1.0 - p2_cal)).astype(np.float32)

    margin = max(float(getattr(args, "cmct_prat_recovery_tau_margin", 0.04)), 0.0)
    relaxed_threshold = max(float(corr_threshold) - margin, 0.0)
    semantic_min = float(getattr(args, "cmct_prat_semantic_min", 0.10))
    structure_min = float(getattr(args, "cmct_prat_structure_min", 0.20))
    min_bcd = float(getattr(args, "cmct_prat_min_bcd", 0.0))
    min_score_q = float(getattr(args, "cmct_prat_region_min_score_q75", -1.0))
    min_sem_q = float(getattr(args, "cmct_prat_region_min_semantic_q75", -1.0))
    min_struct_q = float(getattr(args, "cmct_prat_region_min_structure_q75", -1.0))
    if min_score_q < 0:
        min_score_q = relaxed_threshold
    if min_sem_q < 0:
        min_sem_q = semantic_min
    if min_struct_q < 0:
        min_struct_q = structure_min

    denom = max(float(corr_threshold) - relaxed_threshold, 1e-6)
    score_support = np.clip((score_np - relaxed_threshold) / denom, 0.0, 1.0).astype(np.float32)
    structure_support = (np.clip(bcd_np, 0.0, 1.0) * score_support).astype(np.float32)

    candidate = (
        (~out)
        & (score_np >= relaxed_threshold)
        & (semantic_delta >= semantic_min)
        & (structure_support >= structure_min)
    )
    if min_bcd > 0:
        candidate &= bcd_np >= min_bcd
    raw_candidate_pixels = int(candidate.sum())

    labels_cc, num = base.connected_components(candidate)
    recovery = np.zeros_like(out, dtype=bool)
    min_area = int(getattr(args, "component_gate_min_area", 8))
    max_area = int(getattr(args, "component_gate_max_area", 0))
    accepted_components = 0
    rejected_components = 0
    score_q75_values = []
    semantic_q75_values = []
    structure_q75_values = []
    bcd_q75_values = []
    for lab in range(1, num + 1):
        region = labels_cc == lab
        area = int(region.sum())
        if area < min_area:
            continue
        if max_area > 0 and area > max_area:
            continue
        score_values = score_np[region]
        semantic_values = semantic_delta[region]
        structure_values = structure_support[region]
        bcd_values = bcd_np[region]
        score_q75 = float(np.quantile(score_values, 0.75)) if score_values.size else 0.0
        semantic_q75 = float(np.quantile(semantic_values, 0.75)) if semantic_values.size else 0.0
        structure_q75 = float(np.quantile(structure_values, 0.75)) if structure_values.size else 0.0
        bcd_q75 = float(np.quantile(bcd_values, 0.75)) if bcd_values.size else 0.0
        keep = (
            score_q75 >= min_score_q
            and semantic_q75 >= min_sem_q
            and structure_q75 >= min_struct_q
        )
        if keep:
            recovery[region] = True
            accepted_components += 1
            score_q75_values.append(score_q75)
            semantic_q75_values.append(semantic_q75)
            structure_q75_values.append(structure_q75)
            bcd_q75_values.append(bcd_q75)
        else:
            rejected_components += 1

    component_recovery_pixels = int(recovery.sum())
    max_area_ratio = float(getattr(args, "cmct_prat_max_added_ratio", 0.015))
    if max_area_ratio > 0 and component_recovery_pixels > 0:
        max_pixels = max(1, int(round(max_area_ratio * image_area)))
        if component_recovery_pixels > max_pixels:
            values = (score_np + semantic_delta + structure_support)[recovery]
            cutoff = float(np.partition(values, -max_pixels)[-max_pixels])
            recovery &= (score_np + semantic_delta + structure_support) >= cutoff

    capped_recovery_pixels = int(recovery.sum())
    use_gate = (
        component_model is not None
        and not bool(getattr(args, "cmct_prat_disable_component_gate", False))
    )
    recovery_gate_threshold = float(getattr(args, "cmct_prat_recovery_gate_keep_threshold", -1.0))
    if recovery.any() and use_gate:
        gate_args = args
        if recovery_gate_threshold >= 0:
            gate_args = argparse.Namespace(**vars(args))
            gate_args.component_gate_keep_threshold = recovery_gate_threshold
        recovery = base.component_gate_filter(
            recovery.astype(np.int64),
            score,
            bcd_score,
            relaxed_threshold,
            gate_args,
            component_model,
            device,
        ).astype(bool)
    gated_recovery_pixels = int(recovery.sum())

    before = int(out.sum())
    current_area_ratio = float(before / image_area)
    out |= recovery
    added_pixels = int(out.sum()) - before
    return out.astype(np.int64), {
        "enabled": True,
        "threshold": float(corr_threshold),
        "relaxed_threshold": float(relaxed_threshold),
        "recovery_margin": float(margin),
        "semantic_delta_mean": float(semantic_delta.mean()),
        "semantic_delta_q75": float(np.quantile(semantic_delta, 0.75)),
        "structure_support_mean": float(structure_support.mean()),
        "structure_support_q75": float(np.quantile(structure_support, 0.75)),
        "current_area_ratio": current_area_ratio,
        "candidate_area_ratio": float(raw_candidate_pixels / image_area),
        "component_recovery_area_ratio": float(component_recovery_pixels / image_area),
        "capped_recovery_area_ratio": float(capped_recovery_pixels / image_area),
        "recovery_added_area_ratio": float(added_pixels / image_area),
        "accepted_components": float(accepted_components),
        "rejected_components": float(rejected_components),
        "accepted_score_q75_mean": float(np.mean(score_q75_values)) if score_q75_values else 0.0,
        "accepted_semantic_q75_mean": float(np.mean(semantic_q75_values)) if semantic_q75_values else 0.0,
        "accepted_structure_q75_mean": float(np.mean(structure_q75_values)) if structure_q75_values else 0.0,
        "accepted_bcd_q75_mean": float(np.mean(bcd_q75_values)) if bcd_q75_values else 0.0,
        "recovery_gate_keep_ratio": float(gated_recovery_pixels / max(float(capped_recovery_pixels), 1.0)),
        "recovery_component_gate_enabled": float(use_gate),
        "recovery_gate_keep_threshold": float(recovery_gate_threshold if use_gate else -1.0),
        "final_area_ratio": float(out.mean()),
    }


def multiclass_component_competition_gate(
    args,
    class_names: Sequence[str],
    score_maps: Dict[str, np.ndarray],
    thresholds: Dict[str, float],
    candidate_masks: Dict[str, np.ndarray],
    bcd_score: torch.Tensor,
    endpoint_maps_a: Optional[Dict[str, np.ndarray]] = None,
    endpoint_maps_b: Optional[Dict[str, np.ndarray]] = None,
) -> Tuple[Dict[str, np.ndarray], Dict[str, float]]:
    """Resolve class-by-class OVCD masks into one component-level multi-class map."""

    if not class_names:
        return {}, {"enabled": True, "reason": "no_classes"}
    first_mask = next(iter(candidate_masks.values()), None)
    if first_mask is None:
        return {}, {"enabled": True, "reason": "no_candidates"}

    shape_hw = first_mask.shape
    out = {cls: np.zeros(shape_hw, dtype=np.int64) for cls in class_names}
    candidate_bool = {
        cls: np.asarray(candidate_masks.get(cls, np.zeros(shape_hw, dtype=np.int64))).astype(bool)
        for cls in class_names
    }
    union = np.zeros(shape_hw, dtype=bool)
    for mask in candidate_bool.values():
        union |= mask
    if not union.any():
        return out, {
            "enabled": True,
            "components": 0,
            "accepted_components": 0,
            "rejected_components": 0,
            "accepted_area_ratio": 0.0,
        }

    bcd = bcd_score.float().clamp(0, 1)
    bcd_threshold = base.component_gate_threshold(bcd, args)
    bcd_threshold_value = float(bcd_threshold.detach().cpu())
    bcd_np = bcd.detach().cpu().numpy()

    mode = str(getattr(args, "cmct_multiclass_mode", "component_union"))
    if mode == "endpoint_topk":
        candidate_stack = np.stack([candidate_bool[cls] for cls in class_names], axis=0)
        if endpoint_maps_a is None or endpoint_maps_b is None:
            return {cls: candidate_bool[cls].astype(np.int64) for cls in class_names}, {
                "enabled": True,
                "endpoint_topk_mode": True,
                "reason": "missing_endpoint_maps",
            }
        endpoint_a = np.stack([
            np.asarray(endpoint_maps_a[cls], dtype=np.float32) for cls in class_names
        ], axis=0)
        endpoint_b = np.stack([
            np.asarray(endpoint_maps_b[cls], dtype=np.float32) for cls in class_names
        ], axis=0)
        topk = max(1, min(int(getattr(args, "cmct_multiclass_endpoint_topk", 1)), len(class_names)))
        order_a = np.argsort(-endpoint_a, axis=0)[:topk]
        order_b = np.argsort(-endpoint_b, axis=0)[:topk]
        allowed = np.zeros_like(candidate_stack, dtype=bool)
        yy, xx = np.indices(shape_hw)
        for rank in range(topk):
            allowed[order_a[rank], yy, xx] = True
            allowed[order_b[rank], yy, xx] = True
        slack = float(getattr(args, "cmct_multiclass_endpoint_margin_slack", 0.0))
        if slack > 0:
            kth_a = np.take_along_axis(endpoint_a, order_a[-1][None, :, :], axis=0)[0]
            kth_b = np.take_along_axis(endpoint_b, order_b[-1][None, :, :], axis=0)[0]
            allowed |= endpoint_a >= (kth_a[None, :, :] - slack)
            allowed |= endpoint_b >= (kth_b[None, :, :] - slack)
        if bool(getattr(args, "cmct_multiclass_endpoint_keep_unambiguous", False)):
            unambiguous = candidate_stack.sum(axis=0) <= 1
            allowed |= candidate_stack & unambiguous[None, :, :]
        out_stack = candidate_stack & allowed
        out = {cls: out_stack[idx].astype(np.int64) for idx, cls in enumerate(class_names)}
        overlap_pixels_before = int((candidate_stack.sum(axis=0) > 1).sum())
        overlap_pixels_after = int((out_stack.sum(axis=0) > 1).sum())
        suppressed_pixels = int((candidate_stack & ~out_stack).sum())
        return out, {
            "enabled": True,
            "endpoint_topk_mode": True,
            "endpoint_topk": int(topk),
            "accepted_area_ratio": float(out_stack.any(axis=0).mean()),
            "overlap_pixels_before": int(overlap_pixels_before),
            "overlap_area_ratio_before": float(overlap_pixels_before / max(float(union.size), 1.0)),
            "overlap_pixels_after": int(overlap_pixels_after),
            "overlap_area_ratio_after": float(overlap_pixels_after / max(float(union.size), 1.0)),
            "suppressed_pixels": int(suppressed_pixels),
            "suppressed_area_ratio": float(suppressed_pixels / max(float(union.size), 1.0)),
        }
    if mode == "soft_suppress":
        score_stack = np.stack([
            np.asarray(score_maps[cls], dtype=np.float32) for cls in class_names
        ], axis=0)
        threshold_stack = np.asarray([
            float(thresholds.get(cls, 0.0)) for cls in class_names
        ], dtype=np.float32)[:, None, None]
        candidate_stack = np.stack([candidate_bool[cls] for cls in class_names], axis=0)
        score_mode = str(getattr(args, "cmct_multiclass_pixel_score_mode", "delta"))
        if score_mode == "raw":
            evidence = score_stack
            suppress_below = float(getattr(args, "cmct_multiclass_suppress_below_delta", 0.0))
        elif score_mode == "ratio":
            evidence = score_stack / np.maximum(threshold_stack, 1e-6)
            suppress_below = float(getattr(args, "cmct_multiclass_suppress_below_ratio", 1.0))
        else:
            evidence = score_stack - threshold_stack
            suppress_below = float(getattr(args, "cmct_multiclass_suppress_below_delta", 0.0))
        masked_evidence = np.where(candidate_stack, evidence, -1e6).astype(np.float32)
        top1_val = masked_evidence.max(axis=0)
        candidate_count = candidate_stack.sum(axis=0)
        conflict = candidate_count > 1
        min_margin = float(getattr(args, "cmct_multiclass_min_margin", 0.05))
        suppressed_pixels = 0
        out = {cls: candidate_bool[cls].astype(np.int64) for cls in class_names}
        for idx, cls in enumerate(class_names):
            cls_evidence = masked_evidence[idx]
            suppress = (
                conflict
                & candidate_stack[idx]
                & ((top1_val - cls_evidence) >= min_margin)
                & (cls_evidence < suppress_below)
            )
            if suppress.any():
                out[cls][suppress] = 0
                suppressed_pixels += int(suppress.sum())
        overlap_pixels_before = int((candidate_stack.sum(axis=0) > 1).sum())
        out_stack = np.stack([out[cls].astype(bool) for cls in class_names], axis=0)
        overlap_pixels_after = int((out_stack.sum(axis=0) > 1).sum())
        return out, {
            "enabled": True,
            "soft_suppress_mode": True,
            "components": int(num) if "num" in locals() else 0,
            "accepted_components": 0,
            "rejected_components": 0,
            "accepted_area_ratio": float(out_stack.any(axis=0).mean()),
            "overlap_pixels_before": int(overlap_pixels_before),
            "overlap_area_ratio_before": float(overlap_pixels_before / max(float(union.size), 1.0)),
            "overlap_pixels_after": int(overlap_pixels_after),
            "overlap_area_ratio_after": float(overlap_pixels_after / max(float(union.size), 1.0)),
            "suppressed_pixels": int(suppressed_pixels),
            "suppressed_area_ratio": float(suppressed_pixels / max(float(union.size), 1.0)),
        }
    if mode == "pixel_competition":
        score_stack = np.stack([
            np.asarray(score_maps[cls], dtype=np.float32) for cls in class_names
        ], axis=0)
        threshold_stack = np.asarray([
            float(thresholds.get(cls, 0.0)) for cls in class_names
        ], dtype=np.float32)[:, None, None]
        candidate_stack = np.stack([candidate_bool[cls] for cls in class_names], axis=0)
        score_mode = str(getattr(args, "cmct_multiclass_pixel_score_mode", "delta"))
        if score_mode == "raw":
            evidence = score_stack
        elif score_mode == "ratio":
            evidence = score_stack / np.maximum(threshold_stack, 1e-6)
        else:
            evidence = score_stack - threshold_stack
        evidence = np.where(candidate_stack, evidence, -1e6).astype(np.float32)
        top1_idx = np.argmax(evidence, axis=0)
        top1_val = np.take_along_axis(evidence, top1_idx[None, :, :], axis=0)[0]
        if len(class_names) > 1:
            top2_val = np.partition(evidence, -2, axis=0)[-2]
        else:
            top2_val = np.full_like(top1_val, -1e6)
        margin = top1_val - top2_val
        min_delta = float(getattr(args, "cmct_multiclass_min_delta", 0.0))
        min_margin = float(getattr(args, "cmct_multiclass_min_margin", 0.0))
        candidate_count = candidate_stack.sum(axis=0)
        conflict = candidate_count > 1
        accepted = union.copy()
        if bool(getattr(args, "cmct_multiclass_pixel_require_positive_delta", False)):
            accepted &= top1_val >= min_delta
        accepted[conflict] &= margin[conflict] >= min_margin

        labels_cc, num = base.connected_components(accepted)
        min_area = int(getattr(args, "component_gate_min_area", 8))
        max_area = int(getattr(args, "component_gate_max_area", 0))
        min_bcd_support = float(getattr(args, "cmct_multiclass_min_bcd_support", 0.05))
        accepted_components = 0
        rejected_components = 0
        accepted_area = 0.0
        margin_sum = 0.0
        delta_sum = 0.0
        bcd_support_sum = 0.0
        for lab in range(1, num + 1):
            region = labels_cc == lab
            area = int(region.sum())
            if area < min_area:
                accepted[region] = False
                rejected_components += 1
                continue
            if max_area > 0 and area > max_area:
                accepted[region] = False
                rejected_components += 1
                continue
            bcd_support = float((bcd_np[region] > bcd_threshold_value).mean()) if area > 0 else 0.0
            if bcd_support < min_bcd_support:
                accepted[region] = False
                rejected_components += 1
                continue
            accepted_components += 1
            accepted_area += float(area)
            margin_sum += float(margin[region].mean())
            delta_sum += float(top1_val[region].mean())
            bcd_support_sum += bcd_support

        for idx, cls in enumerate(class_names):
            out[cls][accepted & (top1_idx == idx)] = 1
        overlap_pixels_before = int((candidate_stack.sum(axis=0) > 1).sum())
        return out, {
            "enabled": True,
            "pixel_mode": True,
            "components": int(num),
            "accepted_components": int(accepted_components),
            "rejected_components": int(rejected_components),
            "accepted_area_ratio": float(accepted_area / max(float(union.size), 1.0)),
            "overlap_pixels_before": int(overlap_pixels_before),
            "overlap_area_ratio_before": float(overlap_pixels_before / max(float(union.size), 1.0)),
            "mean_margin": float(margin_sum / max(float(accepted_components), 1.0)),
            "mean_delta": float(delta_sum / max(float(accepted_components), 1.0)),
            "mean_bcd_support": float(bcd_support_sum / max(float(accepted_components), 1.0)),
        }

    labels_cc, num = base.connected_components(union)
    image_area = float(union.size)
    topk_ratio = float(getattr(args, "cmct_multiclass_topk_ratio", args.component_gate_topk_ratio))
    min_delta = float(getattr(args, "cmct_multiclass_min_delta", 0.0))
    min_margin = float(getattr(args, "cmct_multiclass_min_margin", 0.0))
    min_cover = float(getattr(args, "cmct_multiclass_min_class_cover", 0.05))
    min_bcd_support = float(getattr(args, "cmct_multiclass_min_bcd_support", 0.05))
    support_weight = float(getattr(args, "cmct_multiclass_support_weight", 0.0))
    over_threshold_weight = float(getattr(args, "cmct_multiclass_over_threshold_weight", 0.0))
    assign_full_component = bool(getattr(args, "cmct_multiclass_assign_full_component", False))
    min_area = int(getattr(args, "component_gate_min_area", 8))
    max_area = int(getattr(args, "component_gate_max_area", 0))

    accepted_components = 0
    rejected_components = 0
    accepted_area = 0.0
    margin_sum = 0.0
    delta_sum = 0.0
    cover_sum = 0.0
    bcd_support_sum = 0.0
    overlap_pixels_before = 0

    stacked_candidate = np.stack([candidate_bool[cls] for cls in class_names], axis=0)
    overlap_pixels_before = int((stacked_candidate.sum(axis=0) > 1).sum())

    for lab in range(1, num + 1):
        region = labels_cc == lab
        area = int(region.sum())
        if area < min_area:
            continue
        if max_area > 0 and area > max_area:
            continue
        bcd_support = float((bcd_np[region] > bcd_threshold_value).mean()) if area > 0 else 0.0
        if bcd_support < min_bcd_support:
            rejected_components += 1
            continue

        evidences = []
        for cls in class_names:
            score_np = np.asarray(score_maps[cls], dtype=np.float32)
            threshold = float(thresholds.get(cls, 0.0))
            cls_values = score_np[region]
            cls_mask = candidate_bool[cls]
            cover = float((region & cls_mask).sum()) / max(float(area), 1.0)
            topk_mean = _region_topk_mean_np(cls_values, topk_ratio)
            over_ratio = float((cls_values > threshold).mean()) if cls_values.size > 0 else 0.0
            delta = float(topk_mean - threshold)
            evidence = delta + support_weight * cover + over_threshold_weight * over_ratio
            evidences.append((evidence, delta, cover, over_ratio, cls))
        evidences.sort(key=lambda x: x[0], reverse=True)
        top1 = evidences[0]
        top2 = evidences[1] if len(evidences) > 1 else (top1[0], top1[1], top1[2], top1[3], top1[4])
        margin = float(top1[0] - top2[0])
        top1_delta = float(top1[1])
        top1_cover = float(top1[2])
        top_cls = top1[4]

        if top1_delta < min_delta or margin < min_margin or top1_cover < min_cover:
            rejected_components += 1
            continue

        if assign_full_component:
            assign_region = region
        else:
            assign_region = region & candidate_bool[top_cls]
            if not assign_region.any():
                score_np = np.asarray(score_maps[top_cls], dtype=np.float32)
                assign_region = region & (score_np > float(thresholds.get(top_cls, 0.0)))
        if not assign_region.any():
            rejected_components += 1
            continue

        out[top_cls][assign_region] = 1
        accepted_components += 1
        accepted_area += float(assign_region.sum())
        margin_sum += margin
        delta_sum += top1_delta
        cover_sum += top1_cover
        bcd_support_sum += bcd_support

    return out, {
        "enabled": True,
        "components": int(num),
        "accepted_components": int(accepted_components),
        "rejected_components": int(rejected_components),
        "accepted_area_ratio": float(accepted_area / max(image_area, 1.0)),
        "overlap_pixels_before": int(overlap_pixels_before),
        "overlap_area_ratio_before": float(overlap_pixels_before / max(image_area, 1.0)),
        "mean_margin": float(margin_sum / max(float(accepted_components), 1.0)),
        "mean_delta": float(delta_sum / max(float(accepted_components), 1.0)),
        "mean_cover": float(cover_sum / max(float(accepted_components), 1.0)),
        "mean_bcd_support": float(bcd_support_sum / max(float(accepted_components), 1.0)),
    }


def _feature_value_by_name(names: Sequence[str], values: Optional[np.ndarray], name: str, default: float = 0.0) -> float:
    if values is None:
        return float(default)
    values = np.asarray(values, dtype=np.float32)
    try:
        idx = list(names).index(name)
    except ValueError:
        return float(default)
    if idx >= values.size:
        return float(default)
    value = float(values[idx])
    return value if np.isfinite(value) else float(default)


def cmct_uncertainty_score(
    args,
    stats: np.ndarray,
    diag: Dict[str, float],
    component_features: Optional[np.ndarray] = None,
    semantic_features: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    if not bool(getattr(args, "cmct_enable_uncertainty_calibration", False)):
        return {"uncertainty": 0.0, "trust": 1.0}

    top1_sim = float(diag.get("top1_sim", 0.0))
    mean_sim = float(diag.get("mean_sim", top1_sim))
    retrieval_low = float(getattr(args, "cmct_uncertainty_retrieval_low", 0.10))
    retrieval_high = float(getattr(args, "cmct_uncertainty_retrieval_high", 0.35))
    retrieval_conf = float(np.clip((top1_sim - retrieval_low) / max(retrieval_high - retrieval_low, 1e-6), 0.0, 1.0))
    mean_conf = float(np.clip((mean_sim - retrieval_low) / max(retrieval_high - retrieval_low, 1e-6), 0.0, 1.0))
    retrieval_unc = 1.0 - 0.7 * retrieval_conf - 0.3 * mean_conf

    score_q50 = _feature_value_by_name(CMCT_STAT_NAMES, stats, "score_q50")
    score_q95 = _feature_value_by_name(CMCT_STAT_NAMES, stats, "score_q95")
    score_std = _feature_value_by_name(CMCT_STAT_NAMES, stats, "score_std")
    score_span = max(0.0, float(score_q95 - score_q50))
    score_conf = 0.5 * float(np.clip(score_span / float(getattr(args, "cmct_uncertainty_score_span_high", 0.20)), 0.0, 1.0))
    score_conf += 0.5 * float(np.clip(score_std / float(getattr(args, "cmct_uncertainty_score_std_high", 0.12)), 0.0, 1.0))
    score_unc = 1.0 - score_conf

    bcd_mean = _feature_value_by_name(CMCT_STAT_NAMES, stats, "bcd_mean")
    product_q85 = _feature_value_by_name(CMCT_STAT_NAMES, stats, "score_bcd_product_q85")
    bcd_conf = 0.5 * float(np.clip(bcd_mean / float(getattr(args, "cmct_uncertainty_bcd_mean_high", 0.20)), 0.0, 1.0))
    bcd_conf += 0.5 * float(np.clip(product_q85 / float(getattr(args, "cmct_uncertainty_product_high", 0.10)), 0.0, 1.0))
    bcd_unc = 1.0 - bcd_conf

    component_unc = 0.0
    component_weight = 0.0
    if component_features is not None and np.asarray(component_features).size > 0:
        low_area = _feature_value_by_name(CMCT_COMPONENT_RELIABILITY_NAMES, component_features, "component_low_prob_area_ratio")
        keep_ratio = _feature_value_by_name(CMCT_COMPONENT_RELIABILITY_NAMES, component_features, "component_keep_count_ratio", 1.0)
        gate_prob = _feature_value_by_name(CMCT_COMPONENT_RELIABILITY_NAMES, component_features, "component_gate_prob_area_mean", 0.5)
        component_unc = float(np.clip(0.5 * low_area + 0.25 * (1.0 - keep_ratio) + 0.25 * (1.0 - gate_prob), 0.0, 1.0))
        component_weight = 1.0

    semantic_unc = 0.0
    semantic_weight = 0.0
    if semantic_features is not None and np.asarray(semantic_features).size > 0:
        semantic_entropy = _feature_value_by_name(CMCT_SEMANTIC_RELATION_NAMES, semantic_features, "semantic_entropy")
        semantic_margin = _feature_value_by_name(CMCT_SEMANTIC_RELATION_NAMES, semantic_features, "semantic_top1_margin")
        margin_unc = 1.0 - float(np.clip(semantic_margin / float(getattr(args, "cmct_uncertainty_semantic_margin_high", 0.05)), 0.0, 1.0))
        semantic_unc = float(np.clip(0.6 * semantic_entropy + 0.4 * margin_unc, 0.0, 1.0))
        semantic_weight = 1.0

    weights = [
        float(getattr(args, "cmct_uncertainty_retrieval_weight", 1.0)),
        float(getattr(args, "cmct_uncertainty_score_weight", 0.75)),
        float(getattr(args, "cmct_uncertainty_bcd_weight", 0.75)),
        float(getattr(args, "cmct_uncertainty_component_weight", 0.50)) * component_weight,
        float(getattr(args, "cmct_uncertainty_semantic_weight", 0.50)) * semantic_weight,
    ]
    values = [retrieval_unc, score_unc, bcd_unc, component_unc, semantic_unc]
    denom = max(float(sum(weights)), 1e-8)
    uncertainty = float(np.clip(sum(w * v for w, v in zip(weights, values)) / denom, 0.0, 1.0))
    trust = 1.0 - uncertainty
    return {
        "uncertainty": uncertainty,
        "trust": float(np.clip(trust, 0.0, 1.0)),
        "retrieval_unc": float(retrieval_unc),
        "score_unc": float(score_unc),
        "bcd_unc": float(bcd_unc),
        "component_unc": float(component_unc),
        "semantic_unc": float(semantic_unc),
    }


def apply_uncertainty_calibration(
    tau: float,
    adaptive_threshold: float,
    args,
    stats: np.ndarray,
    diag: Dict[str, float],
    component_features: Optional[np.ndarray] = None,
    semantic_features: Optional[np.ndarray] = None,
) -> Tuple[float, Dict[str, float]]:
    if not bool(getattr(args, "cmct_enable_uncertainty_calibration", False)):
        return float(tau), {"enabled": False}
    scores = cmct_uncertainty_score(args, stats, diag, component_features, semantic_features)
    strength = float(getattr(args, "cmct_uncertainty_strength", 0.50))
    min_mix = float(getattr(args, "cmct_uncertainty_min_mix", 0.0))
    max_mix = float(getattr(args, "cmct_uncertainty_max_mix", 1.0))
    base_mix = float(np.clip(min_mix + strength * scores["uncertainty"], min_mix, max_mix))
    calibrated = (1.0 - base_mix) * float(tau) + base_mix * float(adaptive_threshold)
    calibrated = float(np.clip(calibrated, float(args.cmct_threshold_min), float(args.cmct_threshold_max)))
    scores.update({
        "enabled": True,
        "base_mix": base_mix,
        "tau_before_uncertainty": float(tau),
        "adaptive_threshold": float(adaptive_threshold),
        "tau_after_uncertainty": calibrated,
    })
    return calibrated, scores


def _class_proto_source_quantiles(bank: Dict[str, object], quantile: float) -> Tuple[List[str], np.ndarray, np.ndarray]:
    source_names = list(bank.get("source_class_names", []))
    memory_taus = np.asarray(bank.get("memory_taus", np.zeros((0,), dtype=np.float32)), dtype=np.float32)
    metas = list(bank.get("memory_meta", []))
    if memory_taus.size == 0:
        return source_names, np.zeros((len(source_names),), dtype=np.float32), np.zeros((len(source_names),), dtype=np.float32)
    global_tau = float(np.quantile(memory_taus, float(quantile)))
    values = []
    counts = []
    for name in source_names:
        idx = [
            i for i, meta in enumerate(metas)
            if i < memory_taus.shape[0] and str(meta.get("class_name", "")) == str(name)
        ]
        if idx:
            class_values = memory_taus[np.asarray(idx, dtype=np.int64)]
            values.append(float(np.quantile(class_values, float(quantile))))
            counts.append(float(len(idx)))
        else:
            values.append(global_tau)
            counts.append(0.0)
    return source_names, np.asarray(values, dtype=np.float32), np.asarray(counts, dtype=np.float32)


def class_prototype_threshold(
    args,
    bank: Dict[str, object],
    clipper: Optional[ClipFeatureExtractor],
    cls_name: Optional[str],
    prompt_bank: Optional[Dict[str, List[str]]],
) -> Tuple[float, Dict[str, float]]:
    memory_taus = np.asarray(bank.get("memory_taus", np.zeros((0,), dtype=np.float32)), dtype=np.float32)
    if memory_taus.size == 0:
        return float(getattr(args, "cmct_threshold_min", 0.02)), {"proto_available": 0.0}

    quantile = float(getattr(args, "cmct_class_proto_quantile", 0.75))
    quantile = float(np.clip(quantile, 0.0, 1.0))
    source_names, source_quantiles, source_counts = _class_proto_source_quantiles(bank, quantile)
    global_proto = float(np.quantile(memory_taus, quantile))
    proto_sem = global_proto
    semantic_conf = 0.0
    semantic_top1_sim = 0.0
    semantic_margin = 0.0

    source_embeddings = np.asarray(bank.get("source_class_text_embeddings", np.zeros((0, 0), dtype=np.float32)), dtype=np.float32)
    if (
        clipper is not None
        and cls_name
        and prompt_bank is not None
        and source_names
        and source_embeddings.ndim == 2
        and source_embeddings.shape[0] == len(source_names)
        and source_embeddings.shape[1] > 0
    ):
        query_embedding = _clipper_text_embedding(clipper, prompt_text_for_class(str(cls_name), prompt_bank))
        sims = np.asarray(source_embeddings @ query_embedding.astype(np.float32), dtype=np.float32)
        weights = _softmax_np(sims, float(getattr(args, "cmct_class_proto_temperature", 0.10)))
        proto_sem = float(np.sum(weights * source_quantiles))
        if sims.size > 0:
            order = np.argsort(sims)[::-1]
            semantic_top1_sim = float(sims[order[0]])
            if order.size > 1:
                semantic_margin = float(sims[order[0]] - sims[order[1]])
            semantic_conf = 0.5 * float(weights[order[0]])
            semantic_conf += 0.5 * float(np.clip(
                semantic_margin / max(float(getattr(args, "cmct_class_proto_semantic_margin_high", 0.05)), 1e-6),
                0.0,
                1.0,
            ))

    exact_tau = None
    exact_rows = 0
    if cls_name:
        metas = list(bank.get("memory_meta", []))
        allowed = _memory_class_aliases(str(cls_name))
        idx = [
            i for i, meta in enumerate(metas)
            if i < memory_taus.shape[0] and str(meta.get("class_name", "")) in allowed
        ]
        exact_rows = len(idx)
        if len(idx) >= int(getattr(args, "cmct_class_proto_min_rows", 4)):
            exact_tau = float(np.quantile(memory_taus[np.asarray(idx, dtype=np.int64)], quantile))

    if exact_tau is not None:
        exact_weight = float(np.clip(getattr(args, "cmct_class_proto_exact_weight", 0.70), 0.0, 1.0))
        proto = exact_weight * exact_tau + (1.0 - exact_weight) * proto_sem
        proto_conf = max(semantic_conf, 0.75)
    else:
        proto = proto_sem
        proto_conf = semantic_conf

    proto = float(np.clip(proto, float(args.cmct_threshold_min), float(args.cmct_threshold_max)))
    return proto, {
        "proto_available": 1.0,
        "proto_tau": proto,
        "proto_semantic_tau": float(proto_sem),
        "proto_exact_tau": float(exact_tau) if exact_tau is not None else -1.0,
        "proto_exact_rows": float(exact_rows),
        "proto_conf": float(np.clip(proto_conf, 0.0, 1.0)),
        "proto_semantic_conf": float(np.clip(semantic_conf, 0.0, 1.0)),
        "proto_semantic_top1_sim": semantic_top1_sim,
        "proto_semantic_margin": semantic_margin,
        "proto_source_count": float(len(source_names)),
        "proto_source_rows_mean": float(source_counts.mean()) if source_counts.size > 0 else 0.0,
    }


def class_prototype_reliability(
    args,
    retrieval_diag: Dict[str, float],
    proto_diag: Dict[str, float],
    component_features: Optional[np.ndarray] = None,
) -> float:
    top1_sim = float(retrieval_diag.get("top1_sim", 0.0))
    retrieval_low = float(getattr(args, "cmct_class_proto_retrieval_low", 0.10))
    retrieval_high = float(getattr(args, "cmct_class_proto_retrieval_high", 0.35))
    retrieval_conf = float(np.clip((top1_sim - retrieval_low) / max(retrieval_high - retrieval_low, 1e-6), 0.0, 1.0))
    semantic_conf = float(np.clip(proto_diag.get("proto_conf", 0.0), 0.0, 1.0))

    component_conf = 0.0
    component_weight = 0.0
    if component_features is not None and np.asarray(component_features).size > 0:
        gate_prob = _feature_value_by_name(CMCT_COMPONENT_RELIABILITY_NAMES, component_features, "component_gate_prob_area_mean", 0.5)
        keep_area = _feature_value_by_name(CMCT_COMPONENT_RELIABILITY_NAMES, component_features, "component_keep_area_ratio", 0.0)
        low_area = _feature_value_by_name(CMCT_COMPONENT_RELIABILITY_NAMES, component_features, "component_low_prob_area_ratio", 0.0)
        component_conf = float(np.clip(0.5 * gate_prob + 0.3 * keep_area + 0.2 * (1.0 - low_area), 0.0, 1.0))
        component_weight = 1.0

    weights = [
        float(getattr(args, "cmct_class_proto_retrieval_weight", 1.0)),
        float(getattr(args, "cmct_class_proto_semantic_weight", 1.0)),
        float(getattr(args, "cmct_class_proto_component_weight", 0.50)) * component_weight,
    ]
    values = [retrieval_conf, semantic_conf, component_conf]
    denom = max(float(sum(weights)), 1e-8)
    return float(np.clip(sum(w * v for w, v in zip(weights, values)) / denom, 0.0, 1.0))


def apply_class_prototype_threshold(
    tau: float,
    args,
    bank: Dict[str, object],
    clipper: Optional[ClipFeatureExtractor],
    cls_name: Optional[str],
    prompt_bank: Optional[Dict[str, List[str]]],
    retrieval_diag: Dict[str, float],
    component_features: Optional[np.ndarray] = None,
) -> Tuple[float, Dict[str, float]]:
    if not bool(getattr(args, "cmct_enable_class_prototype_threshold", False)):
        return float(tau), {"enabled": False}
    proto, proto_diag = class_prototype_threshold(args, bank, clipper, cls_name, prompt_bank)
    reliability = class_prototype_reliability(args, retrieval_diag, proto_diag, component_features=component_features)
    reliability_floor = float(np.clip(getattr(args, "cmct_class_proto_reliability_floor", 0.0), 0.0, 1.0))
    reliability = reliability_floor + (1.0 - reliability_floor) * reliability
    alpha = float(np.clip(getattr(args, "cmct_class_proto_mix", 0.50), 0.0, 1.0)) * reliability

    direction = str(getattr(args, "cmct_class_proto_direction", "interp"))
    if direction == "floor":
        target_tau = max(float(tau), float(proto))
    elif direction == "cap":
        target_tau = min(float(tau), float(proto))
    elif direction == "interp":
        target_tau = float(proto)
    else:
        raise ValueError(f"Unknown cmct_class_proto_direction: {direction}")
    tau_out = (1.0 - alpha) * float(tau) + alpha * float(target_tau)
    tau_out = float(np.clip(tau_out, float(args.cmct_threshold_min), float(args.cmct_threshold_max)))
    proto_diag.update({
        "enabled": True,
        "proto_reliability": float(reliability),
        "proto_alpha": float(alpha),
        "proto_direction": direction,
        "tau_before_proto": float(tau),
        "tau_after_proto": tau_out,
    })
    return tau_out, proto_diag


def cmct_meta_feature_names(args) -> List[str]:
    return cmct_correction_feature_names(args) + list(CMCT_META_THRESHOLD_SCALAR_NAMES)


def cmct_meta_feature_dim(args) -> int:
    return len(cmct_meta_feature_names(args))


def cmct_meta_features(
    args,
    bank: Dict[str, object],
    stats: np.ndarray,
    tau_mem: float,
    tau_corr: float,
    adaptive_threshold: float,
    retrieval_diag: Dict[str, float],
    cls_name: Optional[str] = None,
    component_features: Optional[np.ndarray] = None,
    semantic_features: Optional[np.ndarray] = None,
    uncertainty_diag: Optional[Dict[str, float]] = None,
) -> np.ndarray:
    correction_row = cmct_correction_features(
        args,
        bank,
        stats,
        tau_mem,
        retrieval_diag,
        cls_name=cls_name,
        component_features=component_features,
        semantic_features=semantic_features,
    )
    eps = 1e-6
    uncertainty_diag = uncertainty_diag or {}
    adaptive = max(float(adaptive_threshold), eps)
    scalars = np.asarray([
        float(tau_mem),
        float(tau_corr),
        float(adaptive_threshold),
        float(tau_corr) - float(tau_mem),
        float(tau_corr) - float(adaptive_threshold),
        float(tau_mem) - float(adaptive_threshold),
        float(tau_corr) / adaptive,
        float(tau_mem) / adaptive,
        float(retrieval_diag.get("ratio_mem", 0.0)),
        float(uncertainty_diag.get("uncertainty", 0.0)),
        float(uncertainty_diag.get("trust", 1.0)),
    ], dtype=np.float32)
    return np.concatenate([correction_row, scalars], axis=0).astype(np.float32)


@torch.no_grad()
def predict_meta_threshold(
    args,
    meta_model,
    bank,
    stats: np.ndarray,
    tau_mem: float,
    tau_corr: float,
    adaptive_threshold: float,
    retrieval_diag: Dict[str, float],
    device,
    cls_name: Optional[str] = None,
    component_features: Optional[np.ndarray] = None,
    semantic_features: Optional[np.ndarray] = None,
    uncertainty_diag: Optional[Dict[str, float]] = None,
) -> Tuple[float, Dict[str, float]]:
    if meta_model is None or not bool(getattr(args, "cmct_enable_meta_threshold", False)):
        return float(tau_corr), {"enabled": False}
    x = torch.tensor(
        cmct_meta_features(
            args,
            bank,
            stats,
            tau_mem,
            tau_corr,
            adaptive_threshold,
            retrieval_diag,
            cls_name=cls_name,
            component_features=component_features,
            semantic_features=semantic_features,
            uncertainty_diag=uncertainty_diag,
        ),
        dtype=torch.float32,
        device=device,
    ).view(1, -1)
    delta = float(meta_model(x).detach().cpu().reshape(()))
    tau_meta = float(np.clip(
        float(tau_corr) + delta,
        float(args.cmct_threshold_min),
        float(args.cmct_threshold_max),
    ))
    return tau_meta, {
        "enabled": True,
        "meta_delta_tau": float(delta),
        "tau_before_meta": float(tau_corr),
        "tau_after_meta": float(tau_meta),
    }


def _component_reliability_confidence(component_features: Optional[np.ndarray]) -> float:
    if component_features is None or np.asarray(component_features).size == 0:
        return 0.5
    values = np.asarray(component_features, dtype=np.float32)
    gate_prob = _feature_value_by_name(CMCT_COMPONENT_RELIABILITY_NAMES, values, "component_gate_prob_area_mean", 0.5)
    keep_area = _feature_value_by_name(CMCT_COMPONENT_RELIABILITY_NAMES, values, "component_keep_area_ratio", 0.0)
    low_area = _feature_value_by_name(CMCT_COMPONENT_RELIABILITY_NAMES, values, "component_low_prob_area_ratio", 0.0)
    bcd_support = _feature_value_by_name(CMCT_COMPONENT_RELIABILITY_NAMES, values, "component_bcd_support_mean", 0.0)
    score_margin = _feature_value_by_name(CMCT_COMPONENT_RELIABILITY_NAMES, values, "component_score_margin_mean", 0.0)
    return float(np.clip(
        0.35 * gate_prob
        + 0.25 * keep_area
        + 0.20 * (1.0 - low_area)
        + 0.10 * bcd_support
        + 0.10 * np.clip(score_margin, 0.0, 1.0),
        0.0,
        1.0,
    ))


def _semantic_relation_confidence(args, semantic_features: Optional[np.ndarray]) -> float:
    if semantic_features is None or np.asarray(semantic_features).size == 0:
        return 0.5
    values = np.asarray(semantic_features, dtype=np.float32)
    margin = _feature_value_by_name(CMCT_SEMANTIC_RELATION_NAMES, values, "semantic_top1_margin", 0.0)
    entropy = _feature_value_by_name(CMCT_SEMANTIC_RELATION_NAMES, values, "semantic_entropy", 1.0)
    margin_high = max(float(getattr(
        args,
        "cmct_semantic_delta_margin_high",
        getattr(args, "cmct_class_residual_adapter_semantic_margin_high", 0.05),
    )), 1e-6)
    margin_conf = float(np.clip(margin / margin_high, 0.0, 1.0))
    entropy_conf = float(np.clip(1.0 - entropy, 0.0, 1.0))
    return float(np.clip(0.55 * margin_conf + 0.45 * entropy_conf, 0.0, 1.0))


def _score_class_residual_adapter_candidate(metrics: Dict[str, float], args) -> float:
    metric_name = str(getattr(args, "cmct_class_residual_adapter_select_metric", "IoU"))
    value = float(metrics.get(metric_name, 0.0))
    min_recall = float(getattr(args, "cmct_class_residual_adapter_min_recall", 0.45))
    recall = float(metrics.get("Recall", 0.0))
    if recall < min_recall:
        value -= float(getattr(args, "cmct_class_residual_adapter_recall_penalty", 1.0)) * (min_recall - recall)
    min_precision = float(getattr(args, "cmct_class_residual_adapter_min_precision", 0.0))
    precision = float(metrics.get("Pre", 0.0))
    if precision < min_precision:
        value -= float(getattr(args, "cmct_class_residual_adapter_precision_penalty", 1.0)) * (min_precision - precision)
    return float(value)


def _score_calibration_reference(
    args,
    bank: Dict[str, object],
    clipper: Optional[ClipFeatureExtractor],
    cls_name: Optional[str],
    prompt_bank: Optional[Dict[str, List[str]]],
) -> Dict[str, float]:
    source_names = list(bank.get("source_class_names", []))
    memory_stats = np.asarray(bank.get("memory_stats", np.zeros((0, len(CMCT_STAT_NAMES)), dtype=np.float32)), dtype=np.float32)
    metas = list(bank.get("memory_meta", []))
    if not source_names or memory_stats.ndim != 2 or memory_stats.shape[0] == 0:
        return {"available": 0.0}

    global_profile = {
        name: float(np.mean(memory_stats[:, idx]))
        for idx, name in enumerate(CMCT_STAT_NAMES)
        if idx < memory_stats.shape[1]
    }
    profiles = []
    counts = []
    for source_name in source_names:
        idx = [
            i for i, meta in enumerate(metas)
            if i < memory_stats.shape[0] and str(meta.get("class_name", "")) == str(source_name)
        ]
        if idx:
            arr = memory_stats[np.asarray(idx, dtype=np.int64)]
            profile = {
                name: float(np.mean(arr[:, stat_idx]))
                for stat_idx, name in enumerate(CMCT_STAT_NAMES)
                if stat_idx < arr.shape[1]
            }
            counts.append(float(len(idx)))
        else:
            profile = dict(global_profile)
            counts.append(0.0)
        profiles.append(profile)

    semantic_profile = dict(global_profile)
    top1_sim = 0.0
    margin = 0.0
    if (
        clipper is not None
        and cls_name
        and prompt_bank is not None
        and source_names
        and len(profiles) == len(source_names)
    ):
        source_embeddings = np.asarray(bank.get("source_class_text_embeddings", np.zeros((0, 0), dtype=np.float32)), dtype=np.float32)
        if source_embeddings.ndim == 2 and source_embeddings.shape[0] == len(source_names) and source_embeddings.shape[1] > 0:
            query = _clipper_text_embedding(clipper, prompt_text_for_class(str(cls_name), prompt_bank))
            query = query / max(float(np.linalg.norm(query)), 1e-8)
            source_norm = source_embeddings / np.maximum(np.linalg.norm(source_embeddings, axis=1, keepdims=True), 1e-8)
            sims = np.matmul(source_norm, query.astype(np.float32)).astype(np.float32)
            weights = _softmax_np(sims, float(getattr(args, "cmct_score_calibration_temperature", 0.10)))
            semantic_profile = {}
            for stat_name in CMCT_STAT_NAMES:
                values = np.asarray([profile.get(stat_name, global_profile.get(stat_name, 0.0)) for profile in profiles], dtype=np.float32)
                semantic_profile[stat_name] = float(np.sum(weights * values))
            if sims.size > 0:
                order = np.argsort(sims)[::-1]
                top1_sim = float(sims[order[0]])
                if order.size > 1:
                    margin = float(sims[order[0]] - sims[order[1]])

    exact_profile = None
    exact_rows = 0.0
    if cls_name:
        allowed = _memory_class_aliases(str(cls_name))
        matched = [i for i, name in enumerate(source_names) if str(name) in allowed]
        if matched:
            exact_rows = max(float(counts[i]) for i in matched)
            exact_idx = matched[int(np.argmax([counts[i] for i in matched]))]
            if exact_rows >= float(getattr(args, "cmct_score_calibration_min_rows", 4)):
                exact_profile = profiles[exact_idx]

    exact_weight = float(np.clip(getattr(args, "cmct_score_calibration_exact_weight", 0.70), 0.0, 1.0))
    if exact_profile is None:
        exact_weight = 0.0
        exact_profile = semantic_profile
    out = {"available": 1.0, "exact_rows": float(exact_rows), "top1_sim": float(top1_sim), "semantic_margin": float(margin)}
    for stat_name in CMCT_STAT_NAMES:
        out[f"ref_{stat_name}"] = float(
            exact_weight * exact_profile.get(stat_name, semantic_profile.get(stat_name, 0.0))
            + (1.0 - exact_weight) * semantic_profile.get(stat_name, global_profile.get(stat_name, 0.0))
        )
    return out


def _logit_float(value: float, eps: float = 1e-4) -> float:
    value = float(np.clip(float(value), eps, 1.0 - eps))
    return float(np.log(value / (1.0 - value)))


def _score_calibration_score_candidate(metrics: Dict[str, float], args) -> float:
    metric_name = str(getattr(args, "cmct_score_calibration_select_metric", "IoU"))
    value = float(metrics.get(metric_name, 0.0))
    min_recall = float(getattr(args, "cmct_score_calibration_min_recall", 0.45))
    recall = float(metrics.get("Recall", 0.0))
    if recall < min_recall:
        value -= float(getattr(args, "cmct_score_calibration_recall_penalty", 1.0)) * (min_recall - recall)
    min_precision = float(getattr(args, "cmct_score_calibration_min_precision", 0.0))
    precision = float(metrics.get("Pre", 0.0))
    if precision < min_precision:
        value -= float(getattr(args, "cmct_score_calibration_precision_penalty", 1.0)) * (min_precision - precision)
    return float(value)


def apply_score_calibration(
    score: torch.Tensor,
    bcd_score: Optional[torch.Tensor],
    args,
    bank: Dict[str, object],
    clipper: Optional[ClipFeatureExtractor],
    cls_name: Optional[str],
    prompt_bank: Optional[Dict[str, List[str]]],
    stats: np.ndarray,
    semantic_features: Optional[np.ndarray] = None,
    component_features: Optional[np.ndarray] = None,
    mode: Optional[str] = None,
    strength: Optional[float] = None,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    if not bool(getattr(args, "cmct_enable_score_calibration", False)):
        return score, {"enabled": False}
    mode_value = str(mode or getattr(args, "cmct_score_calibration_mode", "shift_q95"))
    strength_value = float(np.clip(
        float(strength if strength is not None else getattr(args, "cmct_score_calibration_strength", 0.0)),
        0.0,
        1.0,
    ))
    if strength_value <= 0.0:
        return score, {"enabled": True, "strength": 0.0}

    if mode_value == "bcd_product":
        if bcd_score is None:
            return score, {"enabled": True, "strength": strength_value, "available": 0.0, "reason_bcd_missing": 1.0}
        bcd = bcd_score.float().clamp(0, 1)
        floor = float(np.clip(getattr(args, "cmct_score_calibration_bcd_floor", 0.05), 0.0, 1.0))
        alpha = max(float(getattr(args, "cmct_score_calibration_bcd_alpha", 1.0)), 1e-6)
        target = score.float().clamp(0, 1) * (floor + (1.0 - floor) * bcd.pow(alpha))
        calibrated = ((1.0 - strength_value) * score.float().clamp(0, 1) + strength_value * target).clamp(0, 1)
        return calibrated, {
            "enabled": True,
            "available": 1.0,
            "mode_bcd_product": 1.0,
            "strength": float(strength_value),
            "effective_strength": float(strength_value),
            "bcd_floor": float(floor),
            "bcd_alpha": float(alpha),
            "bcd_mean": float(bcd.mean().detach().cpu()),
        }

    ref = _score_calibration_reference(args, bank, clipper, cls_name, prompt_bank)
    if float(ref.get("available", 0.0)) <= 0.0:
        return score, {"enabled": True, "strength": strength_value, "available": 0.0}

    if mode_value == "shift_q95":
        low_name, high_name = "score_q50", "score_q95"
        use_affine = False
    elif mode_value == "affine_q50_q95":
        low_name, high_name = "score_q50", "score_q95"
        use_affine = True
    elif mode_value == "affine_q75_q95":
        low_name, high_name = "score_q75", "score_q95"
        use_affine = True
    else:
        raise ValueError(f"Unknown cmct_score_calibration_mode: {mode_value}")

    cur_low = _feature_value_by_name(CMCT_STAT_NAMES, stats, low_name)
    cur_high = _feature_value_by_name(CMCT_STAT_NAMES, stats, high_name)
    ref_low = float(ref.get(f"ref_{low_name}", cur_low))
    ref_high = float(ref.get(f"ref_{high_name}", cur_high))
    eps = float(getattr(args, "cmct_score_calibration_eps", 1e-4))
    cur_low_l = _logit_float(cur_low, eps)
    cur_high_l = _logit_float(cur_high, eps)
    ref_low_l = _logit_float(ref_low, eps)
    ref_high_l = _logit_float(ref_high, eps)
    raw_shift = float(np.clip(
        ref_high_l - cur_high_l,
        -float(getattr(args, "cmct_score_calibration_max_shift", 2.0)),
        float(getattr(args, "cmct_score_calibration_max_shift", 2.0)),
    ))
    raw_scale = 1.0
    if use_affine:
        raw_scale = (ref_high_l - ref_low_l) / max(cur_high_l - cur_low_l, 1e-6)
        raw_scale = float(np.clip(
            raw_scale,
            float(getattr(args, "cmct_score_calibration_scale_min", 0.50)),
            float(getattr(args, "cmct_score_calibration_scale_max", 2.00)),
        ))

    effective_strength = strength_value
    if bool(getattr(args, "cmct_score_calibration_confidence_gate", False)):
        semantic_conf = _semantic_relation_confidence(args, semantic_features)
        component_conf = _component_reliability_confidence(component_features)
        min_conf = float(np.clip(getattr(args, "cmct_score_calibration_min_confidence", 0.35), 0.0, 1.0))
        semantic_gate = min_conf + (1.0 - min_conf) * semantic_conf
        if raw_shift >= 0.0:
            component_gate = min_conf + (1.0 - min_conf) * component_conf
        else:
            component_gate = min_conf + (1.0 - min_conf) * (1.0 - component_conf)
        effective_strength *= float(np.clip(semantic_gate * component_gate, 0.0, 1.0))

    score_safe = score.float().clamp(eps, 1.0 - eps)
    logit_score = torch.logit(score_safe)
    if use_affine:
        target_logit = ref_low_l + raw_scale * (logit_score - cur_low_l)
    else:
        target_logit = logit_score + raw_shift
    out_logit = (1.0 - effective_strength) * logit_score + effective_strength * target_logit
    calibrated = torch.sigmoid(out_logit).clamp(0.0, 1.0)
    return calibrated, {
        "enabled": True,
        "available": 1.0,
        "mode_shift_q95": float(mode_value == "shift_q95"),
        "mode_affine": float(use_affine),
        "strength": float(strength_value),
        "effective_strength": float(effective_strength),
        "cur_low": float(cur_low),
        "cur_high": float(cur_high),
        "ref_low": float(ref_low),
        "ref_high": float(ref_high),
        "raw_shift": float(raw_shift),
        "raw_scale": float(raw_scale),
        "exact_rows": float(ref.get("exact_rows", 0.0)),
        "top1_sim": float(ref.get("top1_sim", 0.0)),
        "semantic_margin": float(ref.get("semantic_margin", 0.0)),
    }


@torch.no_grad()
def select_score_calibration(
    args,
    val_items: Sequence[dict],
    processor,
    change_module,
    device,
    clipper,
    bank,
    correction_model,
    component_model=None,
) -> Dict[str, object]:
    if not bool(getattr(args, "cmct_enable_score_calibration", False)):
        return {"enabled": False, "reason": "score_calibration_disabled"}
    if not bool(getattr(args, "cmct_score_calibration_auto_select", False)):
        return {
            "enabled": True,
            "selected": False,
            "mode": str(getattr(args, "cmct_score_calibration_mode", "shift_q95")),
            "strength": float(getattr(args, "cmct_score_calibration_strength", 0.0)),
            "reason": "auto_select_disabled",
        }
    if len(val_items) == 0:
        return {"enabled": True, "selected": False, "reason": "empty_source_val"}

    modes = [
        x.strip()
        for x in str(getattr(args, "cmct_score_calibration_mode_grid", "shift_q95,affine_q75_q95")).split(",")
        if x.strip() in CMCT_SCORE_CALIBRATION_MODES
    ]
    strengths = [float(x) for x in base.parse_ratio_grid(args.cmct_score_calibration_strength_grid)]
    candidates = [(mode, float(np.clip(strength, 0.0, 1.0))) for mode in modes for strength in strengths]
    if not candidates:
        return {"enabled": True, "selected": False, "reason": "empty_candidate_grid"}

    source_names, prompt_bank, confusers = source_prompt_spec(args)
    selection_method = "clip_corrected_tau_component_gate" if bool(args.enable_component_gate) and component_model is not None else "clip_corrected_tau_softgate"
    cfg = base._memory_config_from_args(args)
    if correction_model is not None:
        correction_model.eval()
    if component_model is not None:
        component_model.eval()
    confusions = OrderedDict((candidate, np.zeros((2, 2), dtype=np.float64)) for candidate in candidates)

    for item in tqdm(val_items, desc="select-score-calibration", dynamic_ncols=True):
        meta, _, _, size_hw = bench.predict_change(processor, change_module, item, args, device)
        gt = bench.read_binary_mask(item["label"], size_hw).astype(np.int64)
        pair_embedding = clipper.encode_pair(item)
        scores_a, presence_a = extract_class_scores(processor, meta["state_a"], prompt_bank, source_names)
        scores_b, presence_b = extract_class_scores(processor, meta["state_b"], prompt_bank, source_names)
        posterior = build_posterior_inputs(
            scores_a,
            scores_b,
            presence_a,
            presence_b,
            class_names=source_names,
            confusers=confusers,
            rho=args.rho,
        )
        pred_by_candidate = OrderedDict((candidate, np.zeros(size_hw, dtype=np.int64)) for candidate in candidates)
        for cls_idx, cls_name in enumerate(source_names):
            score = memory_rectified_score(
                posterior.posterior_a[cls_idx],
                posterior.posterior_b[cls_idx],
                None if args.disable_bcd_gate else meta["bcd_score"],
                presence_a=presence_a[cls_idx],
                presence_b=presence_b[cls_idx],
                config=cfg,
            )
            presence = torch.maximum(presence_a[cls_idx], presence_b[cls_idx])
            base_mask, base_threshold = base.decode_class_score(score, presence, 0.0, args)
            component_features = component_reliability_features(
                args,
                base_mask,
                score,
                meta["bcd_score"],
                base_threshold,
                component_model,
                device,
            )
            semantic_features = semantic_relation_features_for_prompt(args, bank, clipper, cls_name, prompt_bank)
            key, stat_values = make_query(args, item, cls_name, prompt_bank, pair_embedding, clipper, score, meta["bcd_score"], presence)
            tau_mem, tau_corr, diag = predict_cmct_thresholds(
                args,
                bank,
                correction_model,
                key,
                stat_values,
                device,
                cls_name=cls_name,
                component_features=component_features,
                semantic_features=semantic_features,
            )
            tau_corr = apply_adaptive_floor(tau_corr, base_threshold, args)
            tau_corr = apply_class_prior_floor(tau_corr, args, bank, cls_name)
            tau_corr, semantic_prior_diag = apply_semantic_prior_floor(tau_corr, args, semantic_features)
            diag.update({f"semantic_prior_{k}": v for k, v in semantic_prior_diag.items() if isinstance(v, (int, float, bool))})
            tau_corr, uncertainty_diag = apply_uncertainty_calibration(
                tau_corr,
                base_threshold,
                args,
                stat_values,
                diag,
                component_features=component_features,
                semantic_features=semantic_features,
            )
            diag.update({f"uncertainty_{k}": v for k, v in uncertainty_diag.items() if isinstance(v, (int, float, bool))})
            ratio_mem = diag.get("ratio_mem") if bool(getattr(args, "cmct_enable_memory_ratio_budget", False)) else None
            for candidate_mode, candidate_strength in candidates:
                calibrated_score, _ = apply_score_calibration(
                    score,
                    meta["bcd_score"],
                    args,
                    bank,
                    clipper,
                    cls_name,
                    prompt_bank,
                    stat_values,
                    semantic_features=semantic_features,
                    component_features=component_features,
                    mode=candidate_mode,
                    strength=candidate_strength,
                )
                mask = _decode_ratio_selection_mask(
                    args,
                    calibrated_score,
                    meta["bcd_score"],
                    presence,
                    tau_corr,
                    ratio_mem,
                    selection_method,
                    component_model,
                    device,
                )
                pred_by_candidate[(candidate_mode, candidate_strength)] |= mask.astype(np.int64)
        for candidate, pred in pred_by_candidate.items():
            confusions[candidate] += bench.confusion_binary(gt, pred.astype(np.int64))

    entries = []
    best_entry = None
    best_score = -float("inf")
    for candidate_mode, candidate_strength in candidates:
        cm = confusions[(candidate_mode, candidate_strength)]
        metrics = {k: float(v) for k, v in bench.metrics_from_confusion(cm).items()}
        score_value = _score_calibration_score_candidate(metrics, args)
        entry = {
            "mode": str(candidate_mode),
            "strength": float(candidate_strength),
            "score": float(score_value),
            "metrics": metrics,
            "confusion": cm.tolist(),
        }
        entries.append(entry)
        if score_value > best_score:
            best_score = score_value
            best_entry = entry

    if best_entry is None:
        return {"enabled": True, "selected": False, "reason": "selection_failed", "candidates": entries}
    args.cmct_score_calibration_mode = str(best_entry["mode"])
    args.cmct_score_calibration_strength = float(best_entry["strength"])
    print(
        "Selected score calibration: "
        f"mode={args.cmct_score_calibration_mode}, "
        f"strength={args.cmct_score_calibration_strength:.4f}, "
        f"source_val_score={best_entry['score']:.4f}"
    )
    return {
        "enabled": True,
        "selected": True,
        "metric": str(getattr(args, "cmct_score_calibration_select_metric", "IoU")),
        "selection_method": selection_method,
        "num_val_items": int(len(val_items)),
        "source_classes": source_names,
        "selected_candidate": best_entry,
        "candidates": entries,
    }


def _class_residual_adapter_value(
    args,
    adapter: Dict[str, object],
    bank: Dict[str, object],
    clipper: Optional[ClipFeatureExtractor],
    cls_name: Optional[str],
    prompt_bank: Optional[Dict[str, List[str]]],
    semantic_features: Optional[np.ndarray],
    component_features: Optional[np.ndarray],
    stat: Optional[str] = None,
    mix: Optional[float] = None,
    direction: Optional[str] = None,
) -> Tuple[float, Dict[str, float]]:
    source_names = list(adapter.get("source_class_names", []))
    class_values = adapter.get("class_residuals", {})
    if not source_names or not class_values:
        return 0.0, {"adapter_available": 0.0}

    stat_name = str(stat or getattr(args, "cmct_class_residual_adapter_stat", "q50"))
    if stat_name not in CMCT_CLASS_RESIDUAL_ADAPTER_STATS:
        stat_name = "q50"
    mix_value = float(np.clip(
        float(mix if mix is not None else getattr(args, "cmct_class_residual_adapter_mix", 1.0)),
        0.0,
        1.0,
    ))
    direction_value = str(direction or getattr(args, "cmct_class_residual_adapter_direction", "signed"))

    residual_by_name = []
    counts = []
    global_stats = adapter.get("global_residual", {})
    global_value = float(global_stats.get(stat_name, 0.0)) if isinstance(global_stats, dict) else 0.0
    for name in source_names:
        stats = class_values.get(name, {}) if isinstance(class_values, dict) else {}
        residual_by_name.append(float(stats.get(stat_name, global_value)))
        counts.append(float(stats.get("count", 0.0)))
    residuals = np.asarray(residual_by_name, dtype=np.float32)

    exact_delta = None
    exact_rows = 0.0
    if cls_name and str(cls_name) in class_values:
        stats = class_values.get(str(cls_name), {})
        exact_rows = float(stats.get("count", 0.0))
        if exact_rows >= float(getattr(args, "cmct_class_residual_adapter_min_rows", 2)):
            exact_delta = float(stats.get(stat_name, global_value))

    semantic_delta = global_value
    top1_sim = 0.0
    semantic_margin = 0.0
    if (
        clipper is not None
        and cls_name
        and prompt_bank is not None
        and source_names
        and residuals.size == len(source_names)
    ):
        source_embeddings = np.asarray(bank.get("source_class_text_embeddings", np.zeros((0, 0), dtype=np.float32)), dtype=np.float32)
        if source_embeddings.ndim == 2 and source_embeddings.shape[0] == len(source_names) and source_embeddings.shape[1] > 0:
            query = _clipper_text_embedding(clipper, prompt_text_for_class(str(cls_name), prompt_bank))
            query = query / max(float(np.linalg.norm(query)), 1e-8)
            source_norm = source_embeddings / np.maximum(np.linalg.norm(source_embeddings, axis=1, keepdims=True), 1e-8)
            sims = np.matmul(source_norm, query.astype(np.float32)).astype(np.float32)
            weights = _softmax_np(sims, float(getattr(args, "cmct_class_residual_adapter_temperature", 0.10)))
            semantic_delta = float(np.sum(weights * residuals))
            if sims.size > 0:
                order = np.argsort(sims)[::-1]
                top1_sim = float(sims[order[0]])
                if order.size > 1:
                    semantic_margin = float(sims[order[0]] - sims[order[1]])

    if exact_delta is not None:
        exact_weight = float(np.clip(getattr(args, "cmct_class_residual_adapter_exact_weight", 0.70), 0.0, 1.0))
        raw_delta = exact_weight * float(exact_delta) + (1.0 - exact_weight) * float(semantic_delta)
    else:
        raw_delta = float(semantic_delta)

    if direction_value == "positive":
        raw_delta = max(raw_delta, 0.0)
    elif direction_value == "negative":
        raw_delta = min(raw_delta, 0.0)
    elif direction_value != "signed":
        raise ValueError(f"Unknown cmct_class_residual_adapter_direction: {direction_value}")

    semantic_conf = _semantic_relation_confidence(args, semantic_features)
    component_conf = _component_reliability_confidence(component_features)
    if raw_delta >= 0.0:
        component_gate = 1.0 + float(getattr(args, "cmct_class_residual_adapter_positive_risk_boost", 0.0)) * (1.0 - component_conf)
    else:
        component_gate = component_conf
    reliability = float(np.clip(semantic_conf * component_gate, 0.0, 1.0))
    max_delta = float(getattr(args, "cmct_class_residual_adapter_max_delta", 0.20))
    delta = float(np.clip(mix_value * reliability * raw_delta, -max_delta, max_delta))
    return delta, {
        "adapter_available": 1.0,
        "adapter_raw_delta": float(raw_delta),
        "adapter_delta": float(delta),
        "adapter_mix": float(mix_value),
        "adapter_stat": float(CMCT_CLASS_RESIDUAL_ADAPTER_STATS.index(stat_name)),
        "adapter_direction_positive": float(direction_value == "positive"),
        "adapter_direction_negative": float(direction_value == "negative"),
        "adapter_semantic_delta": float(semantic_delta),
        "adapter_exact_delta": float(exact_delta) if exact_delta is not None else 0.0,
        "adapter_exact_rows": float(exact_rows),
        "adapter_semantic_conf": float(semantic_conf),
        "adapter_component_conf": float(component_conf),
        "adapter_reliability": float(reliability),
        "adapter_top1_sim": float(top1_sim),
        "adapter_semantic_margin": float(semantic_margin),
    }


def apply_class_residual_adapter(
    tau: float,
    args,
    adapter: Optional[Dict[str, object]],
    bank: Dict[str, object],
    clipper: Optional[ClipFeatureExtractor],
    cls_name: Optional[str],
    prompt_bank: Optional[Dict[str, List[str]]],
    semantic_features: Optional[np.ndarray],
    component_features: Optional[np.ndarray],
    stat: Optional[str] = None,
    mix: Optional[float] = None,
    direction: Optional[str] = None,
) -> Tuple[float, Dict[str, float]]:
    if not bool(getattr(args, "cmct_enable_class_residual_adapter", False)):
        return float(tau), {"enabled": False}
    if adapter is None or not bool(adapter.get("enabled", False)):
        return float(tau), {"enabled": True, "adapter_available": 0.0}
    delta, diag = _class_residual_adapter_value(
        args,
        adapter,
        bank,
        clipper,
        cls_name,
        prompt_bank,
        semantic_features,
        component_features,
        stat=stat,
        mix=mix,
        direction=direction,
    )
    tau_out = float(np.clip(
        float(tau) + float(delta),
        float(args.cmct_threshold_min),
        float(args.cmct_threshold_max),
    ))
    diag.update({
        "enabled": True,
        "tau_before_adapter": float(tau),
        "tau_after_adapter": float(tau_out),
    })
    return tau_out, diag


@torch.no_grad()
def collect_class_residual_adapter(
    args,
    val_items: Sequence[dict],
    processor,
    change_module,
    device,
    clipper,
    bank,
    correction_model,
    component_model=None,
) -> Dict[str, object]:
    source_names, prompt_bank, confusers = source_prompt_spec(args)
    residual_rows = OrderedDict((name, []) for name in source_names)
    global_rows = []
    if correction_model is not None:
        correction_model.eval()
    if component_model is not None:
        component_model.eval()
    cfg = base._memory_config_from_args(args)
    for item in tqdm(val_items, desc="collect-class-residual-adapter", dynamic_ncols=True):
        meta, _, _, size_hw = bench.predict_change(processor, change_module, item, args, device)
        gt = bench.read_binary_mask(item["label"], size_hw).astype(np.int64)
        pair_embedding = clipper.encode_pair(item)
        scores_a, presence_a = extract_class_scores(processor, meta["state_a"], prompt_bank, source_names)
        scores_b, presence_b = extract_class_scores(processor, meta["state_b"], prompt_bank, source_names)
        posterior = build_posterior_inputs(
            scores_a,
            scores_b,
            presence_a,
            presence_b,
            class_names=source_names,
            confusers=confusers,
            rho=args.rho,
        )
        for cls_idx, cls_name in enumerate(source_names):
            score = memory_rectified_score(
                posterior.posterior_a[cls_idx],
                posterior.posterior_b[cls_idx],
                None if args.disable_bcd_gate else meta["bcd_score"],
                presence_a=presence_a[cls_idx],
                presence_b=presence_b[cls_idx],
                config=cfg,
            )
            presence = torch.maximum(presence_a[cls_idx], presence_b[cls_idx])
            base_mask, base_threshold = base.decode_class_score(score, presence, 0.0, args)
            component_features = component_reliability_features(
                args,
                base_mask,
                score,
                meta["bcd_score"],
                base_threshold,
                component_model,
                device,
            )
            semantic_features = semantic_relation_features_for_prompt(args, bank, clipper, cls_name, prompt_bank)
            key, stat_values = make_query(args, item, cls_name, prompt_bank, pair_embedding, clipper, score, meta["bcd_score"], presence)
            tau_mem, tau_corr, diag = predict_cmct_thresholds(
                args,
                bank,
                correction_model,
                key,
                stat_values,
                device,
                cls_name=cls_name,
                component_features=component_features,
                semantic_features=semantic_features,
            )
            tau_corr = apply_adaptive_floor(tau_corr, base_threshold, args)
            tau_corr = apply_class_prior_floor(tau_corr, args, bank, cls_name)
            tau_corr, _ = apply_semantic_prior_floor(tau_corr, args, semantic_features)
            tau_corr, _ = apply_uncertainty_calibration(
                tau_corr,
                base_threshold,
                args,
                stat_values,
                diag,
                component_features=component_features,
                semantic_features=semantic_features,
            )
            target_tau = oracle_threshold_for_score(score, gt, args)
            residual = float(target_tau) - float(tau_corr)
            residual_rows[cls_name].append(residual)
            global_rows.append(residual)

    def summarize(values: Sequence[float]) -> Dict[str, float]:
        arr = np.asarray(list(values), dtype=np.float32)
        if arr.size == 0:
            return {"count": 0.0, "mean": 0.0, "q50": 0.0, "q75": 0.0, "q90": 0.0}
        return {
            "count": float(arr.size),
            "mean": float(arr.mean()),
            "q50": float(np.quantile(arr, 0.50)),
            "q75": float(np.quantile(arr, 0.75)),
            "q90": float(np.quantile(arr, 0.90)),
        }

    class_residuals = OrderedDict((name, summarize(values)) for name, values in residual_rows.items())
    return {
        "enabled": True,
        "num_val_items": int(len(val_items)),
        "num_rows": int(len(global_rows)),
        "source_class_names": source_names,
        "global_residual": summarize(global_rows),
        "class_residuals": class_residuals,
    }


@torch.no_grad()
def select_class_residual_adapter(
    args,
    val_items: Sequence[dict],
    processor,
    change_module,
    device,
    clipper,
    bank,
    correction_model,
    component_model=None,
) -> Dict[str, object]:
    if not bool(getattr(args, "cmct_enable_class_residual_adapter", False)):
        return {"enabled": False, "reason": "class_residual_adapter_disabled"}
    adapter = collect_class_residual_adapter(
        args,
        val_items,
        processor,
        change_module,
        device,
        clipper,
        bank,
        correction_model,
        component_model,
    )
    if not bool(getattr(args, "cmct_class_residual_adapter_auto_select", False)):
        adapter.update({
            "selected": False,
            "stat": str(getattr(args, "cmct_class_residual_adapter_stat", "q50")),
            "mix": float(getattr(args, "cmct_class_residual_adapter_mix", 1.0)),
            "direction": str(getattr(args, "cmct_class_residual_adapter_direction", "signed")),
            "reason": "auto_select_disabled",
        })
        return adapter
    if len(val_items) == 0:
        adapter.update({"selected": False, "reason": "empty_source_val"})
        return adapter

    stats = [
        x.strip()
        for x in str(getattr(args, "cmct_class_residual_adapter_stat_grid", "q50,q75")).split(",")
        if x.strip() in CMCT_CLASS_RESIDUAL_ADAPTER_STATS
    ]
    mixes = [float(x) for x in base.parse_ratio_grid(args.cmct_class_residual_adapter_mix_grid)]
    directions = [
        x.strip()
        for x in str(getattr(args, "cmct_class_residual_adapter_direction_grid", "signed,positive")).split(",")
        if x.strip() in {"signed", "positive", "negative"}
    ]
    candidates = [(stat, float(np.clip(mix, 0.0, 1.0)), direction) for stat in stats for mix in mixes for direction in directions]
    if not candidates:
        adapter.update({"selected": False, "reason": "empty_candidate_grid"})
        return adapter

    source_names, prompt_bank, confusers = source_prompt_spec(args)
    selection_method = "clip_corrected_tau_component_gate" if bool(args.enable_component_gate) and component_model is not None else "clip_corrected_tau_softgate"
    cfg = base._memory_config_from_args(args)
    if correction_model is not None:
        correction_model.eval()
    if component_model is not None:
        component_model.eval()
    confusions = OrderedDict((candidate, np.zeros((2, 2), dtype=np.float64)) for candidate in candidates)

    for item in tqdm(val_items, desc="select-class-residual-adapter", dynamic_ncols=True):
        meta, _, _, size_hw = bench.predict_change(processor, change_module, item, args, device)
        gt = bench.read_binary_mask(item["label"], size_hw).astype(np.int64)
        pair_embedding = clipper.encode_pair(item)
        scores_a, presence_a = extract_class_scores(processor, meta["state_a"], prompt_bank, source_names)
        scores_b, presence_b = extract_class_scores(processor, meta["state_b"], prompt_bank, source_names)
        posterior = build_posterior_inputs(
            scores_a,
            scores_b,
            presence_a,
            presence_b,
            class_names=source_names,
            confusers=confusers,
            rho=args.rho,
        )
        pred_by_candidate = OrderedDict((candidate, np.zeros(size_hw, dtype=np.int64)) for candidate in candidates)
        for cls_idx, cls_name in enumerate(source_names):
            score = memory_rectified_score(
                posterior.posterior_a[cls_idx],
                posterior.posterior_b[cls_idx],
                None if args.disable_bcd_gate else meta["bcd_score"],
                presence_a=presence_a[cls_idx],
                presence_b=presence_b[cls_idx],
                config=cfg,
            )
            presence = torch.maximum(presence_a[cls_idx], presence_b[cls_idx])
            base_mask, base_threshold = base.decode_class_score(score, presence, 0.0, args)
            component_features = component_reliability_features(
                args,
                base_mask,
                score,
                meta["bcd_score"],
                base_threshold,
                component_model,
                device,
            )
            semantic_features = semantic_relation_features_for_prompt(args, bank, clipper, cls_name, prompt_bank)
            key, stat_values = make_query(args, item, cls_name, prompt_bank, pair_embedding, clipper, score, meta["bcd_score"], presence)
            tau_mem, tau_corr, diag = predict_cmct_thresholds(
                args,
                bank,
                correction_model,
                key,
                stat_values,
                device,
                cls_name=cls_name,
                component_features=component_features,
                semantic_features=semantic_features,
            )
            tau_corr = apply_adaptive_floor(tau_corr, base_threshold, args)
            tau_corr = apply_class_prior_floor(tau_corr, args, bank, cls_name)
            tau_corr, semantic_prior_diag = apply_semantic_prior_floor(tau_corr, args, semantic_features)
            diag.update({f"semantic_prior_{k}": v for k, v in semantic_prior_diag.items() if isinstance(v, (int, float, bool))})
            tau_corr, uncertainty_diag = apply_uncertainty_calibration(
                tau_corr,
                base_threshold,
                args,
                stat_values,
                diag,
                component_features=component_features,
                semantic_features=semantic_features,
            )
            diag.update({f"uncertainty_{k}": v for k, v in uncertainty_diag.items() if isinstance(v, (int, float, bool))})
            ratio_mem = diag.get("ratio_mem") if bool(getattr(args, "cmct_enable_memory_ratio_budget", False)) else None
            for candidate_stat, candidate_mix, candidate_direction in candidates:
                tau_candidate, _ = apply_class_residual_adapter(
                    tau_corr,
                    args,
                    adapter,
                    bank,
                    clipper,
                    cls_name,
                    prompt_bank,
                    semantic_features,
                    component_features,
                    stat=candidate_stat,
                    mix=candidate_mix,
                    direction=candidate_direction,
                )
                mask = _decode_ratio_selection_mask(
                    args,
                    score,
                    meta["bcd_score"],
                    presence,
                    tau_candidate,
                    ratio_mem,
                    selection_method,
                    component_model,
                    device,
                )
                pred_by_candidate[(candidate_stat, candidate_mix, candidate_direction)] |= mask.astype(np.int64)
        for candidate, pred in pred_by_candidate.items():
            confusions[candidate] += bench.confusion_binary(gt, pred.astype(np.int64))

    entries = []
    best_entry = None
    best_score = -float("inf")
    for candidate_stat, candidate_mix, candidate_direction in candidates:
        cm = confusions[(candidate_stat, candidate_mix, candidate_direction)]
        metrics = {k: float(v) for k, v in bench.metrics_from_confusion(cm).items()}
        score_value = _score_class_residual_adapter_candidate(metrics, args)
        entry = {
            "stat": str(candidate_stat),
            "mix": float(candidate_mix),
            "direction": str(candidate_direction),
            "score": float(score_value),
            "metrics": metrics,
            "confusion": cm.tolist(),
        }
        entries.append(entry)
        if score_value > best_score:
            best_score = score_value
            best_entry = entry

    if best_entry is None:
        adapter.update({"selected": False, "reason": "selection_failed", "candidates": entries})
        return adapter
    args.cmct_class_residual_adapter_stat = str(best_entry["stat"])
    args.cmct_class_residual_adapter_mix = float(best_entry["mix"])
    args.cmct_class_residual_adapter_direction = str(best_entry["direction"])
    print(
        "Selected class residual adapter: "
        f"stat={args.cmct_class_residual_adapter_stat}, "
        f"mix={args.cmct_class_residual_adapter_mix:.4f}, "
        f"direction={args.cmct_class_residual_adapter_direction}, "
        f"source_val_score={best_entry['score']:.4f}"
    )
    adapter.update({
        "selected": True,
        "metric": str(getattr(args, "cmct_class_residual_adapter_select_metric", "IoU")),
        "selection_method": selection_method,
        "selected_candidate": best_entry,
        "candidates": entries,
    })
    return adapter


def _resolve_ratio_selection_method(args, component_model) -> str:
    method = str(getattr(args, "cmct_memory_ratio_select_method", "auto"))
    if method != "auto":
        return method
    if bool(getattr(args, "enable_component_gate", False)) and component_model is not None:
        return "clip_corrected_tau_component_gate"
    return "clip_corrected_tau_softgate"


def _score_ratio_budget_candidate(metrics: Dict[str, float], args) -> float:
    metric_name = str(getattr(args, "cmct_memory_ratio_select_metric", "IoU"))
    value = float(metrics.get(metric_name, 0.0))
    min_recall = float(getattr(args, "cmct_memory_ratio_min_recall", 0.45))
    recall = float(metrics.get("Recall", 0.0))
    if recall < min_recall:
        value -= float(getattr(args, "cmct_memory_ratio_recall_penalty", 1.0)) * (min_recall - recall)
    return float(value)


def _decode_ratio_selection_mask(
    args,
    score: torch.Tensor,
    bcd_score: torch.Tensor,
    presence: torch.Tensor,
    tau_corr: float,
    ratio_mem: Optional[float],
    selection_method: str,
    component_model,
    device,
) -> np.ndarray:
    corr_mask, corr_threshold = decode_fixed_tau(score, presence, tau_corr, args, budget_ratio=ratio_mem)
    if selection_method == "clip_corrected_tau":
        return corr_mask
    if selection_method == "clip_corrected_tau_softgate":
        return base.bcd_soft_gate_filter(corr_mask, score, bcd_score, corr_threshold, args)
    if selection_method == "clip_corrected_tau_component_gate":
        if component_model is None:
            return base.bcd_soft_gate_filter(corr_mask, score, bcd_score, corr_threshold, args)
        return base.component_gate_filter(
            corr_mask,
            score,
            bcd_score,
            corr_threshold,
            args,
            component_model,
            device,
        )
    raise ValueError(f"Unknown cmct_memory_ratio_select_method: {selection_method}")


def _resolve_class_proto_selection_method(args, component_model) -> str:
    method = str(getattr(args, "cmct_class_proto_select_method", "auto"))
    if method != "auto":
        return method
    if bool(getattr(args, "enable_component_gate", False)) and component_model is not None:
        return "clip_corrected_tau_component_gate"
    return "clip_corrected_tau_softgate"


def _resolve_semantic_prior_selection_method(args, component_model) -> str:
    method = str(getattr(args, "cmct_semantic_prior_select_method", "auto"))
    if method != "auto":
        return method
    if bool(getattr(args, "enable_component_gate", False)) and component_model is not None:
        return "clip_corrected_tau_component_gate"
    return "clip_corrected_tau_softgate"


def _score_semantic_prior_candidate(metrics: Dict[str, float], args) -> float:
    metric_name = str(getattr(args, "cmct_semantic_prior_select_metric", "IoU"))
    value = float(metrics.get(metric_name, 0.0))
    min_recall = float(getattr(args, "cmct_semantic_prior_min_recall", 0.45))
    recall = float(metrics.get("Recall", 0.0))
    if recall < min_recall:
        value -= float(getattr(args, "cmct_semantic_prior_recall_penalty", 1.0)) * (min_recall - recall)
    min_precision = float(getattr(args, "cmct_semantic_prior_min_precision", 0.0))
    precision = float(metrics.get("Pre", 0.0))
    if precision < min_precision:
        value -= float(getattr(args, "cmct_semantic_prior_precision_penalty", 1.0)) * (min_precision - precision)
    return float(value)


@torch.no_grad()
def select_semantic_prior_floor(
    args,
    val_items: Sequence[dict],
    processor,
    change_module,
    device,
    clipper,
    bank,
    correction_model,
    component_model=None,
    component_verifier_model=None,
) -> Dict[str, object]:
    if not bool(getattr(args, "cmct_enable_semantic_prior_floor", False)):
        return {"enabled": False, "reason": "semantic_prior_floor_disabled"}
    if not bool(getattr(args, "cmct_semantic_prior_auto_select", False)):
        return {
            "enabled": True,
            "selected": False,
            "reason": "auto_select_disabled",
            "stat": str(getattr(args, "cmct_semantic_prior_floor_stat", "weighted_q90")),
            "strength": float(getattr(args, "cmct_semantic_prior_floor_strength", 1.0)),
            "scale": float(getattr(args, "cmct_semantic_prior_floor_scale", 1.0)),
            "selection_method": _resolve_semantic_prior_selection_method(args, component_model),
        }
    if len(val_items) == 0:
        return {"enabled": True, "selected": False, "reason": "empty_source_val"}

    stats = [
        x.strip()
        for x in str(getattr(args, "cmct_semantic_prior_floor_stat_grid", "weighted_q75,weighted_q90,mix_q75_q90")).split(",")
        if x.strip()
    ]
    allowed_stats = {"weighted_q75", "weighted_q90", "top1_q75", "mix_q75_q90"}
    stats = [x for x in stats if x in allowed_stats]
    strengths = [float(x) for x in base.parse_ratio_grid(args.cmct_semantic_prior_floor_strength_grid)]
    scales = [float(x) for x in base.parse_ratio_grid(args.cmct_semantic_prior_floor_scale_grid)]
    candidates = [
        (stat, float(np.clip(strength, 0.0, 1.0)), float(scale))
        for stat in stats
        for strength in strengths
        for scale in scales
    ]
    if not candidates:
        return {"enabled": True, "selected": False, "reason": "empty_candidate_grid"}

    cfg = base._memory_config_from_args(args)
    class_names, prompt_bank, confusers = source_prompt_spec(args)
    selection_method = _resolve_semantic_prior_selection_method(args, component_model)
    if correction_model is not None:
        correction_model.eval()
    if component_model is not None:
        component_model.eval()

    old_stat = str(args.cmct_semantic_prior_floor_stat)
    old_strength = float(args.cmct_semantic_prior_floor_strength)
    old_scale = float(args.cmct_semantic_prior_floor_scale)
    confusions = OrderedDict((candidate, np.zeros((2, 2), dtype=np.float64)) for candidate in candidates)

    for item in tqdm(val_items, desc="select-semantic-prior", dynamic_ncols=True):
        meta, _, _, size_hw = bench.predict_change(processor, change_module, item, args, device)
        gt = bench.read_binary_mask(item["label"], size_hw).astype(np.int64)
        pair_embedding = clipper.encode_pair(item)
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
        pred_by_candidate = OrderedDict((candidate, np.zeros(size_hw, dtype=np.int64)) for candidate in candidates)
        for cls_idx, cls_name in enumerate(class_names):
            score = memory_rectified_score(
                posterior.posterior_a[cls_idx],
                posterior.posterior_b[cls_idx],
                None if args.disable_bcd_gate else meta["bcd_score"],
                presence_a=presence_a[cls_idx],
                presence_b=presence_b[cls_idx],
                config=cfg,
            )
            presence = torch.maximum(presence_a[cls_idx], presence_b[cls_idx])
            base_mask, base_threshold = base.decode_class_score(score, presence, 0.0, args)
            component_features = component_reliability_features(
                args,
                base_mask,
                score,
                meta["bcd_score"],
                base_threshold,
                component_model,
                device,
            )
            semantic_features = semantic_relation_features_for_prompt(args, bank, clipper, cls_name, prompt_bank)
            key, stat_values = make_query(args, item, cls_name, prompt_bank, pair_embedding, clipper, score, meta["bcd_score"], presence)
            _, tau_corr, diag = predict_cmct_thresholds(
                args,
                bank,
                correction_model,
                key,
                stat_values,
                device,
                cls_name=cls_name,
                component_features=component_features,
                semantic_features=semantic_features,
            )
            tau_corr = apply_adaptive_floor(tau_corr, base_threshold, args)
            tau_corr = apply_class_prior_floor(tau_corr, args, bank, cls_name)
            for candidate_stat, strength, scale in candidates:
                candidate_tau, _ = apply_semantic_prior_floor(
                    tau_corr,
                    args,
                    semantic_features,
                    stat=candidate_stat,
                    strength=strength,
                    scale=scale,
                )
                mask = _decode_ratio_selection_mask(
                    args,
                    score,
                    meta["bcd_score"],
                    presence,
                    candidate_tau,
                    None,
                    selection_method,
                    component_model,
                    device,
                )
                pred_by_candidate[(candidate_stat, strength, scale)] |= mask.astype(np.int64)

        for candidate, pred in pred_by_candidate.items():
            confusions[candidate] += bench.confusion_binary(gt, pred.astype(np.int64))

    entries = []
    best_entry = None
    best_score = -float("inf")
    for candidate_stat, strength, scale in candidates:
        cm = confusions[(candidate_stat, strength, scale)]
        metrics = {k: float(v) for k, v in bench.metrics_from_confusion(cm).items()}
        score_value = _score_semantic_prior_candidate(metrics, args)
        entry = {
            "stat": str(candidate_stat),
            "strength": float(strength),
            "scale": float(scale),
            "score": float(score_value),
            "metrics": metrics,
            "confusion": cm.tolist(),
        }
        entries.append(entry)
        if score_value > best_score:
            best_score = score_value
            best_entry = entry

    if best_entry is None:
        args.cmct_semantic_prior_floor_stat = old_stat
        args.cmct_semantic_prior_floor_strength = old_strength
        args.cmct_semantic_prior_floor_scale = old_scale
        return {"enabled": True, "selected": False, "reason": "selection_failed", "candidates": entries}

    args.cmct_semantic_prior_floor_stat = str(best_entry["stat"])
    args.cmct_semantic_prior_floor_strength = float(best_entry["strength"])
    args.cmct_semantic_prior_floor_scale = float(best_entry["scale"])
    print(
        "Selected semantic prior floor: "
        f"stat={args.cmct_semantic_prior_floor_stat}, "
        f"strength={args.cmct_semantic_prior_floor_strength:.4f}, "
        f"scale={args.cmct_semantic_prior_floor_scale:.4f}, "
        f"source_val_score={best_entry['score']:.4f}"
    )
    return {
        "enabled": True,
        "selected": True,
        "metric": str(getattr(args, "cmct_semantic_prior_select_metric", "IoU")),
        "selection_method": selection_method,
        "num_val_items": int(len(val_items)),
        "source_classes": class_names,
        "old_stat": old_stat,
        "old_strength": float(old_strength),
        "old_scale": float(old_scale),
        "selected": best_entry,
        "candidates": entries,
    }


def update_cmct_checkpoint_semantic_prior_selection(args, selection: Dict[str, object]) -> None:
    if not selection or not bool(selection.get("enabled")) or not bool(selection.get("selected")):
        return
    if not args.output_checkpoint:
        return
    path = Path(args.output_checkpoint)
    source_path = path if path.exists() else (Path(args.load_checkpoint) if args.load_checkpoint else None)
    if source_path is None or not source_path.exists():
        return
    ckpt = torch.load(source_path, map_location="cpu", weights_only=False)
    config = dict(ckpt.get("config", {}))
    config["cmct_enable_semantic_prior_floor"] = bool(getattr(args, "cmct_enable_semantic_prior_floor", False))
    config["cmct_semantic_prior_floor_stat"] = str(args.cmct_semantic_prior_floor_stat)
    config["cmct_semantic_prior_floor_strength"] = float(args.cmct_semantic_prior_floor_strength)
    config["cmct_semantic_prior_floor_scale"] = float(args.cmct_semantic_prior_floor_scale)
    config["cmct_semantic_prior_auto_select"] = bool(getattr(args, "cmct_semantic_prior_auto_select", False))
    ckpt["config"] = config
    ckpt["cmct_semantic_prior_selection"] = selection
    torch.save(ckpt, path)


def _score_class_proto_candidate(metrics: Dict[str, float], args) -> float:
    metric_name = str(getattr(args, "cmct_class_proto_select_metric", "IoU"))
    value = float(metrics.get(metric_name, 0.0))
    min_recall = float(getattr(args, "cmct_class_proto_min_recall", 0.45))
    recall = float(metrics.get("Recall", 0.0))
    if recall < min_recall:
        value -= float(getattr(args, "cmct_class_proto_recall_penalty", 1.0)) * (min_recall - recall)
    min_precision = float(getattr(args, "cmct_class_proto_min_precision", 0.0))
    precision = float(metrics.get("Pre", 0.0))
    if precision < min_precision:
        value -= float(getattr(args, "cmct_class_proto_precision_penalty", 1.0)) * (min_precision - precision)
    return float(value)


@torch.no_grad()
def select_class_prototype_threshold(
    args,
    val_items: Sequence[dict],
    processor,
    change_module,
    device,
    clipper,
    bank,
    correction_model,
    component_model=None,
    component_verifier_model=None,
) -> Dict[str, object]:
    if not bool(getattr(args, "cmct_enable_class_prototype_threshold", False)):
        return {"enabled": False, "reason": "class_prototype_threshold_disabled"}
    if not bool(getattr(args, "cmct_class_proto_auto_select", False)):
        return {
            "enabled": True,
            "selected": False,
            "reason": "auto_select_disabled",
            "quantile": float(getattr(args, "cmct_class_proto_quantile", 0.75)),
            "mix": float(getattr(args, "cmct_class_proto_mix", 0.50)),
            "direction": str(getattr(args, "cmct_class_proto_direction", "interp")),
        }
    if len(val_items) == 0:
        return {"enabled": True, "selected": False, "reason": "empty_source_val"}

    quantiles = [float(x) for x in base.parse_ratio_grid(args.cmct_class_proto_quantile_grid)]
    mixes = [float(x) for x in base.parse_ratio_grid(args.cmct_class_proto_mix_grid)]
    directions = [x.strip() for x in str(args.cmct_class_proto_direction_grid).split(",") if x.strip()]
    candidates = [
        (float(np.clip(q, 0.0, 1.0)), float(np.clip(mix, 0.0, 1.0)), direction)
        for q in quantiles
        for mix in mixes
        for direction in directions
        if direction in {"interp", "floor", "cap"}
    ]
    if not candidates:
        return {"enabled": True, "selected": False, "reason": "empty_candidate_grid"}

    cfg = base._memory_config_from_args(args)
    class_names, prompt_bank, confusers = source_prompt_spec(args)
    selection_method = _resolve_class_proto_selection_method(args, component_model)
    if correction_model is not None:
        correction_model.eval()
    if component_model is not None:
        component_model.eval()

    old_quantile = float(args.cmct_class_proto_quantile)
    old_mix = float(args.cmct_class_proto_mix)
    old_direction = str(args.cmct_class_proto_direction)
    confusions = OrderedDict((candidate, np.zeros((2, 2), dtype=np.float64)) for candidate in candidates)

    for item in tqdm(val_items, desc="select-class-prototype", dynamic_ncols=True):
        meta, _, _, size_hw = bench.predict_change(processor, change_module, item, args, device)
        gt = bench.read_binary_mask(item["label"], size_hw).astype(np.int64)
        pair_embedding = clipper.encode_pair(item)
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
        pred_by_candidate = OrderedDict(
            (candidate, np.zeros(size_hw, dtype=np.int64))
            for candidate in candidates
        )
        for cls_idx, cls_name in enumerate(class_names):
            score = memory_rectified_score(
                posterior.posterior_a[cls_idx],
                posterior.posterior_b[cls_idx],
                None if args.disable_bcd_gate else meta["bcd_score"],
                presence_a=presence_a[cls_idx],
                presence_b=presence_b[cls_idx],
                config=cfg,
            )
            presence = torch.maximum(presence_a[cls_idx], presence_b[cls_idx])
            base_mask, base_threshold = base.decode_class_score(score, presence, 0.0, args)
            component_features = component_reliability_features(
                args,
                base_mask,
                score,
                meta["bcd_score"],
                base_threshold,
                component_model,
                device,
            )
            semantic_features = semantic_relation_features_for_prompt(args, bank, clipper, cls_name, prompt_bank)
            key, stats = make_query(args, item, cls_name, prompt_bank, pair_embedding, clipper, score, meta["bcd_score"], presence)
            _, tau_corr, diag = predict_cmct_thresholds(
                args,
                bank,
                correction_model,
                key,
                stats,
                device,
                cls_name=cls_name,
                component_features=component_features,
                semantic_features=semantic_features,
            )
            tau_corr = apply_adaptive_floor(tau_corr, base_threshold, args)
            tau_corr = apply_class_prior_floor(tau_corr, args, bank, cls_name)
            tau_corr, semantic_prior_diag = apply_semantic_prior_floor(tau_corr, args, semantic_features)
            diag.update({f"semantic_prior_{k}": v for k, v in semantic_prior_diag.items() if isinstance(v, (int, float, bool))})
            tau_corr, uncertainty_diag = apply_uncertainty_calibration(
                tau_corr,
                base_threshold,
                args,
                stats,
                diag,
                component_features=component_features,
                semantic_features=semantic_features,
            )
            diag.update({f"uncertainty_{k}": v for k, v in uncertainty_diag.items() if isinstance(v, (int, float, bool))})
            ratio_mem = diag.get("ratio_mem") if bool(getattr(args, "cmct_enable_memory_ratio_budget", False)) else None

            for quantile, mix, direction in candidates:
                args.cmct_class_proto_quantile = float(quantile)
                args.cmct_class_proto_mix = float(mix)
                args.cmct_class_proto_direction = str(direction)
                tau_proto, _ = apply_class_prototype_threshold(
                    tau_corr,
                    args,
                    bank,
                    clipper,
                    cls_name,
                    prompt_bank,
                    diag,
                    component_features=component_features,
                )
                mask = _decode_ratio_selection_mask(
                    args,
                    score,
                    meta["bcd_score"],
                    presence,
                    tau_proto,
                    ratio_mem,
                    selection_method,
                    component_model,
                    device,
                )
                pred_by_candidate[(quantile, mix, direction)] |= mask.astype(np.int64)

        for candidate, pred in pred_by_candidate.items():
            confusions[candidate] += bench.confusion_binary(gt, pred.astype(np.int64))

    entries = []
    best_entry = None
    best_score = -float("inf")
    for quantile, mix, direction in candidates:
        cm = confusions[(quantile, mix, direction)]
        metrics = {k: float(v) for k, v in bench.metrics_from_confusion(cm).items()}
        score_value = _score_class_proto_candidate(metrics, args)
        entry = {
            "quantile": float(quantile),
            "mix": float(mix),
            "direction": str(direction),
            "score": float(score_value),
            "metrics": metrics,
            "confusion": cm.tolist(),
        }
        entries.append(entry)
        if score_value > best_score:
            best_score = score_value
            best_entry = entry

    if best_entry is None:
        args.cmct_class_proto_quantile = old_quantile
        args.cmct_class_proto_mix = old_mix
        args.cmct_class_proto_direction = old_direction
        return {"enabled": True, "selected": False, "reason": "selection_failed", "candidates": entries}

    args.cmct_class_proto_quantile = float(best_entry["quantile"])
    args.cmct_class_proto_mix = float(best_entry["mix"])
    args.cmct_class_proto_direction = str(best_entry["direction"])
    print(
        "Selected class prototype threshold: "
        f"quantile={args.cmct_class_proto_quantile:.4f}, "
        f"mix={args.cmct_class_proto_mix:.4f}, "
        f"direction={args.cmct_class_proto_direction}, "
        f"method={selection_method}, "
        f"source_val_score={best_entry['score']:.4f}"
    )
    return {
        "enabled": True,
        "selected": True,
        "method": selection_method,
        "metric": str(getattr(args, "cmct_class_proto_select_metric", "IoU")),
        "num_val_items": int(len(val_items)),
        "source_classes": class_names,
        "old_quantile": float(old_quantile),
        "old_mix": float(old_mix),
        "old_direction": old_direction,
        "selected": best_entry,
        "candidates": entries,
    }


def update_cmct_checkpoint_class_proto_selection(args, selection: Dict[str, object]) -> None:
    if not selection or not bool(selection.get("enabled")) or not bool(selection.get("selected")):
        return
    if not args.output_checkpoint:
        return
    path = Path(args.output_checkpoint)
    source_path = path if path.exists() else (Path(args.load_checkpoint) if args.load_checkpoint else None)
    if source_path is None or not source_path.exists():
        return
    ckpt = torch.load(source_path, map_location="cpu", weights_only=False)
    config = dict(ckpt.get("config", {}))
    config["cmct_enable_class_prototype_threshold"] = bool(getattr(args, "cmct_enable_class_prototype_threshold", False))
    config["cmct_class_proto_quantile"] = float(args.cmct_class_proto_quantile)
    config["cmct_class_proto_mix"] = float(args.cmct_class_proto_mix)
    config["cmct_class_proto_direction"] = str(args.cmct_class_proto_direction)
    config["cmct_class_proto_auto_select"] = bool(getattr(args, "cmct_class_proto_auto_select", False))
    ckpt["config"] = config
    ckpt["cmct_class_proto_selection"] = selection
    torch.save(ckpt, path)


def _score_component_local_candidate(metrics: Dict[str, float], args) -> float:
    metric_name = str(getattr(args, "cmct_component_local_select_metric", "IoU"))
    value = float(metrics.get(metric_name, 0.0))
    min_recall = float(getattr(args, "cmct_component_local_min_recall", 0.45))
    recall = float(metrics.get("Recall", 0.0))
    if recall < min_recall:
        value -= float(getattr(args, "cmct_component_local_recall_penalty", 1.0)) * (min_recall - recall)
    min_precision = float(getattr(args, "cmct_component_local_min_precision", 0.0))
    precision = float(metrics.get("Pre", 0.0))
    if precision < min_precision:
        value -= float(getattr(args, "cmct_component_local_precision_penalty", 1.0)) * (min_precision - precision)
    return float(value)


@torch.no_grad()
def select_component_local_refinement(
    args,
    val_items: Sequence[dict],
    processor,
    change_module,
    device,
    clipper,
    bank,
    correction_model,
    component_model=None,
    component_verifier_model=None,
) -> Dict[str, object]:
    if not bool(getattr(args, "cmct_enable_component_local_refine", False)):
        return {"enabled": False, "reason": "component_local_refine_disabled"}
    if component_model is None:
        return {"enabled": True, "selected": False, "reason": "missing_component_model"}
    if not bool(getattr(args, "cmct_component_local_auto_select", False)):
        return {
            "enabled": True,
            "selected": False,
            "reason": "auto_select_disabled",
            "relax_scale": float(getattr(args, "cmct_component_local_relax_scale", 0.75)),
            "add_threshold": float(getattr(args, "cmct_component_local_add_threshold", 0.70)),
            "expand_threshold": float(getattr(args, "cmct_component_local_expand_threshold", 0.70)),
            "min_score_support": float(getattr(args, "cmct_component_local_min_score_support", 0.0)),
            "min_score_topk_margin": float(getattr(args, "cmct_component_local_min_score_topk_margin", -1.0)),
            "clip_min_sim": float(getattr(args, "cmct_component_clip_min_sim", -1.0)),
            "clip_min_margin": float(getattr(args, "cmct_component_clip_min_margin", -1.0)),
        }
    if len(val_items) == 0:
        return {"enabled": True, "selected": False, "reason": "empty_source_val"}

    relax_scales = [float(x) for x in base.parse_ratio_grid(args.cmct_component_local_relax_scale_grid)]
    add_thresholds = [float(x) for x in base.parse_ratio_grid(args.cmct_component_local_add_threshold_grid)]
    expand_thresholds = [float(x) for x in base.parse_ratio_grid(args.cmct_component_local_expand_threshold_grid)]
    score_supports = [float(x) for x in base.parse_ratio_grid(args.cmct_component_local_min_score_support_grid)]
    score_margins = [float(x) for x in base.parse_ratio_grid(args.cmct_component_local_min_score_topk_margin_grid)]
    if bool(getattr(args, "cmct_enable_component_clip_verify", False)):
        clip_sims = [float(x) for x in base.parse_ratio_grid(args.cmct_component_clip_min_sim_grid)]
        clip_margins = [float(x) for x in base.parse_ratio_grid(args.cmct_component_clip_min_margin_grid)]
    else:
        clip_sims = [float(getattr(args, "cmct_component_clip_min_sim", -1.0))]
        clip_margins = [float(getattr(args, "cmct_component_clip_min_margin", -1.0))]
    candidates = [
        (
            float(np.clip(relax, 0.05, 1.0)),
            float(np.clip(add, 0.0, 1.0)),
            float(np.clip(expand, 0.0, 1.0)),
            float(np.clip(score_support, 0.0, 1.0)),
            float(score_margin),
            float(clip_sim),
            float(clip_margin),
        )
        for relax in relax_scales
        for add in add_thresholds
        for expand in expand_thresholds
        for score_support in score_supports
        for score_margin in score_margins
        for clip_sim in clip_sims
        for clip_margin in clip_margins
    ]
    if not candidates:
        return {"enabled": True, "selected": False, "reason": "empty_candidate_grid"}

    cfg = base._memory_config_from_args(args)
    class_names, prompt_bank, confusers = source_prompt_spec(args)
    if correction_model is not None:
        correction_model.eval()
    component_model.eval()

    old_relax = float(args.cmct_component_local_relax_scale)
    old_add = float(args.cmct_component_local_add_threshold)
    old_expand = float(args.cmct_component_local_expand_threshold)
    old_score_support = float(args.cmct_component_local_min_score_support)
    old_score_margin = float(args.cmct_component_local_min_score_topk_margin)
    old_clip_sim = float(args.cmct_component_clip_min_sim)
    old_clip_margin = float(args.cmct_component_clip_min_margin)
    old_block_classes = str(getattr(args, "cmct_component_local_block_classes", ""))
    args.cmct_component_local_block_classes = ""
    confusions = OrderedDict((candidate, np.zeros((2, 2), dtype=np.float64)) for candidate in candidates)
    class_gate_confusions = OrderedDict((cls_name, np.zeros((2, 2), dtype=np.float64)) for cls_name in class_names)
    class_candidate_confusions = OrderedDict(
        (
            cls_name,
            OrderedDict((candidate, np.zeros((2, 2), dtype=np.float64)) for candidate in candidates),
        )
        for cls_name in class_names
    )

    for item in tqdm(val_items, desc="select-component-local", dynamic_ncols=True):
        meta, _, _, size_hw = bench.predict_change(processor, change_module, item, args, device)
        gt = bench.read_binary_mask(item["label"], size_hw).astype(np.int64)
        pair_embedding = clipper.encode_pair(item)
        scores_a, presence_a = extract_class_scores(processor, meta["state_a"], prompt_bank, class_names)
        scores_b, presence_b = extract_class_scores(processor, meta["state_b"], prompt_bank, class_names)
        class_text_embeddings = np.stack([
            _clipper_text_embedding(clipper, prompt_text_for_class(name, prompt_bank))
            for name in class_names
        ], axis=0).astype(np.float32)
        posterior = build_posterior_inputs(
            scores_a,
            scores_b,
            presence_a,
            presence_b,
            class_names=class_names,
            confusers=confusers,
            rho=args.rho,
        )
        pred_by_candidate = OrderedDict(
            (candidate, np.zeros(size_hw, dtype=np.int64))
            for candidate in candidates
        )
        for cls_idx, cls_name in enumerate(class_names):
            score = memory_rectified_score(
                posterior.posterior_a[cls_idx],
                posterior.posterior_b[cls_idx],
                None if args.disable_bcd_gate else meta["bcd_score"],
                presence_a=presence_a[cls_idx],
                presence_b=presence_b[cls_idx],
                config=cfg,
            )
            presence = torch.maximum(presence_a[cls_idx], presence_b[cls_idx])
            base_mask, base_threshold = base.decode_class_score(score, presence, 0.0, args)
            component_features = component_reliability_features(
                args,
                base_mask,
                score,
                meta["bcd_score"],
                base_threshold,
                component_model,
                device,
            )
            semantic_features = semantic_relation_features_for_prompt(args, bank, clipper, cls_name, prompt_bank)
            key, stats = make_query(args, item, cls_name, prompt_bank, pair_embedding, clipper, score, meta["bcd_score"], presence)
            _, tau_corr, diag = predict_cmct_thresholds(
                args,
                bank,
                correction_model,
                key,
                stats,
                device,
                cls_name=cls_name,
                component_features=component_features,
                semantic_features=semantic_features,
            )
            tau_corr = apply_adaptive_floor(tau_corr, base_threshold, args)
            tau_corr = apply_class_prior_floor(tau_corr, args, bank, cls_name)
            tau_corr, semantic_prior_diag = apply_semantic_prior_floor(tau_corr, args, semantic_features)
            diag.update({f"semantic_prior_{k}": v for k, v in semantic_prior_diag.items() if isinstance(v, (int, float, bool))})
            tau_corr, uncertainty_diag = apply_uncertainty_calibration(
                tau_corr,
                base_threshold,
                args,
                stats,
                diag,
                component_features=component_features,
                semantic_features=semantic_features,
            )
            diag.update({f"uncertainty_{k}": v for k, v in uncertainty_diag.items() if isinstance(v, (int, float, bool))})
            corr_mask, corr_threshold = decode_fixed_tau(score, presence, tau_corr, args)
            gate_mask = base.component_gate_filter(
                corr_mask,
                score,
                meta["bcd_score"],
                corr_threshold,
                args,
                component_model,
                device,
            )
            class_gate_confusions[cls_name] += bench.confusion_binary(gt, gate_mask.astype(np.int64))

            class_text_embedding = class_text_embeddings[cls_idx]
            for relax, add, expand, score_support, score_margin, clip_sim, clip_margin in candidates:
                args.cmct_component_local_relax_scale = float(relax)
                args.cmct_component_local_add_threshold = float(add)
                args.cmct_component_local_expand_threshold = float(expand)
                args.cmct_component_local_min_score_support = float(score_support)
                args.cmct_component_local_min_score_topk_margin = float(score_margin)
                args.cmct_component_clip_min_sim = float(clip_sim)
                args.cmct_component_clip_min_margin = float(clip_margin)
                mask, _, _ = component_local_refine_mask(
                    args,
                    score,
                    meta["bcd_score"],
                    presence,
                    tau_corr,
                    component_model,
                    device,
                    cls_name=cls_name,
                    item=item,
                    clipper=clipper,
                    class_text_embedding=class_text_embedding,
                    class_text_embeddings=class_text_embeddings,
                    class_text_index=cls_idx,
                    component_verifier_model=component_verifier_model,
                )
                candidate_key = (relax, add, expand, score_support, score_margin, clip_sim, clip_margin)
                mask_int = mask.astype(np.int64)
                pred_by_candidate[candidate_key] |= mask_int
                class_candidate_confusions[cls_name][candidate_key] += bench.confusion_binary(gt, mask_int)

        for candidate, pred in pred_by_candidate.items():
            confusions[candidate] += bench.confusion_binary(gt, pred.astype(np.int64))

    entries = []
    best_entry = None
    best_score = -float("inf")
    for relax, add, expand, score_support, score_margin, clip_sim, clip_margin in candidates:
        cm = confusions[(relax, add, expand, score_support, score_margin, clip_sim, clip_margin)]
        metrics = {k: float(v) for k, v in bench.metrics_from_confusion(cm).items()}
        score_value = _score_component_local_candidate(metrics, args)
        entry = {
            "relax_scale": float(relax),
            "add_threshold": float(add),
            "expand_threshold": float(expand),
            "min_score_support": float(score_support),
            "min_score_topk_margin": float(score_margin),
            "clip_min_sim": float(clip_sim),
            "clip_min_margin": float(clip_margin),
            "score": float(score_value),
            "metrics": metrics,
            "confusion": cm.tolist(),
        }
        entries.append(entry)
        if score_value > best_score:
            best_score = score_value
            best_entry = entry

    if best_entry is None:
        args.cmct_component_local_relax_scale = old_relax
        args.cmct_component_local_add_threshold = old_add
        args.cmct_component_local_expand_threshold = old_expand
        args.cmct_component_local_min_score_support = old_score_support
        args.cmct_component_local_min_score_topk_margin = old_score_margin
        args.cmct_component_clip_min_sim = old_clip_sim
        args.cmct_component_clip_min_margin = old_clip_margin
        args.cmct_component_local_block_classes = old_block_classes
        return {"enabled": True, "selected": False, "reason": "selection_failed", "candidates": entries}

    args.cmct_component_local_relax_scale = float(best_entry["relax_scale"])
    args.cmct_component_local_add_threshold = float(best_entry["add_threshold"])
    args.cmct_component_local_expand_threshold = float(best_entry["expand_threshold"])
    args.cmct_component_local_min_score_support = float(best_entry["min_score_support"])
    args.cmct_component_local_min_score_topk_margin = float(best_entry["min_score_topk_margin"])
    args.cmct_component_clip_min_sim = float(best_entry["clip_min_sim"])
    args.cmct_component_clip_min_margin = float(best_entry["clip_min_margin"])
    best_candidate = (
        float(best_entry["relax_scale"]),
        float(best_entry["add_threshold"]),
        float(best_entry["expand_threshold"]),
        float(best_entry["min_score_support"]),
        float(best_entry["min_score_topk_margin"]),
        float(best_entry["clip_min_sim"]),
        float(best_entry["clip_min_margin"]),
    )
    class_policy = {
        "enabled": bool(getattr(args, "cmct_component_local_class_auto_block", False)),
        "blocked_classes": [],
        "class_entries": [],
    }
    if bool(getattr(args, "cmct_component_local_class_auto_block", False)):
        min_gain = float(getattr(args, "cmct_component_local_class_min_gain", 0.0))
        blocked_classes = []
        for cls_name in class_names:
            gate_metrics = {k: float(v) for k, v in bench.metrics_from_confusion(class_gate_confusions[cls_name]).items()}
            local_metrics = {
                k: float(v)
                for k, v in bench.metrics_from_confusion(class_candidate_confusions[cls_name][best_candidate]).items()
            }
            gate_score = _score_component_local_candidate(gate_metrics, args)
            local_score = _score_component_local_candidate(local_metrics, args)
            blocked = bool(local_score < gate_score + min_gain)
            if blocked:
                blocked_classes.append(str(cls_name))
            class_policy["class_entries"].append({
                "class_name": str(cls_name),
                "blocked": blocked,
                "gate_score": float(gate_score),
                "local_score": float(local_score),
                "gate_metrics": gate_metrics,
                "local_metrics": local_metrics,
            })
        args.cmct_component_local_block_classes = ",".join(blocked_classes)
        class_policy["blocked_classes"] = blocked_classes
    else:
        args.cmct_component_local_block_classes = old_block_classes
    print(
        "Selected component local refinement: "
        f"relax_scale={args.cmct_component_local_relax_scale:.4f}, "
        f"add_threshold={args.cmct_component_local_add_threshold:.4f}, "
        f"expand_threshold={args.cmct_component_local_expand_threshold:.4f}, "
        f"min_score_support={args.cmct_component_local_min_score_support:.4f}, "
        f"min_score_topk_margin={args.cmct_component_local_min_score_topk_margin:.4f}, "
        f"clip_min_sim={args.cmct_component_clip_min_sim:.4f}, "
        f"clip_min_margin={args.cmct_component_clip_min_margin:.4f}, "
        f"blocked_classes={args.cmct_component_local_block_classes or 'none'}, "
        f"source_val_score={best_entry['score']:.4f}"
    )
    return {
        "enabled": True,
        "selected": True,
        "metric": str(getattr(args, "cmct_component_local_select_metric", "IoU")),
        "num_val_items": int(len(val_items)),
        "source_classes": class_names,
        "old_relax_scale": float(old_relax),
        "old_add_threshold": float(old_add),
        "old_expand_threshold": float(old_expand),
        "old_min_score_support": float(old_score_support),
        "old_min_score_topk_margin": float(old_score_margin),
        "old_clip_min_sim": float(old_clip_sim),
        "old_clip_min_margin": float(old_clip_margin),
        "old_block_classes": old_block_classes,
        "class_policy": class_policy,
        "selected": best_entry,
        "candidates": entries,
    }


def update_cmct_checkpoint_component_local_selection(args, selection: Dict[str, object]) -> None:
    if not selection or not bool(selection.get("enabled")) or not bool(selection.get("selected")):
        return
    if not args.output_checkpoint:
        return
    path = Path(args.output_checkpoint)
    source_path = path if path.exists() else (Path(args.load_checkpoint) if args.load_checkpoint else None)
    if source_path is None or not source_path.exists():
        return
    ckpt = torch.load(source_path, map_location="cpu", weights_only=False)
    config = dict(ckpt.get("config", {}))
    config["cmct_enable_component_local_refine"] = bool(getattr(args, "cmct_enable_component_local_refine", False))
    config["cmct_component_local_relax_scale"] = float(args.cmct_component_local_relax_scale)
    config["cmct_component_local_add_threshold"] = float(args.cmct_component_local_add_threshold)
    config["cmct_component_local_expand_threshold"] = float(args.cmct_component_local_expand_threshold)
    config["cmct_component_local_min_score_support"] = float(args.cmct_component_local_min_score_support)
    config["cmct_component_local_min_score_topk_margin"] = float(args.cmct_component_local_min_score_topk_margin)
    config["cmct_enable_component_clip_verify"] = bool(getattr(args, "cmct_enable_component_clip_verify", False))
    config["cmct_component_clip_min_sim"] = float(getattr(args, "cmct_component_clip_min_sim", -1.0))
    config["cmct_component_clip_min_margin"] = float(getattr(args, "cmct_component_clip_min_margin", -1.0))
    config["cmct_component_clip_min_area"] = int(getattr(args, "cmct_component_clip_min_area", 64))
    config["cmct_enable_component_verifier"] = bool(getattr(args, "cmct_enable_component_verifier", False))
    config["cmct_component_verifier_keep_threshold"] = float(getattr(args, "cmct_component_verifier_keep_threshold", 0.50))
    config["cmct_component_local_class_auto_block"] = bool(getattr(args, "cmct_component_local_class_auto_block", False))
    config["cmct_component_local_block_classes"] = str(getattr(args, "cmct_component_local_block_classes", ""))
    config["cmct_component_local_class_min_gain"] = float(getattr(args, "cmct_component_local_class_min_gain", 0.0))
    config["cmct_component_local_auto_select"] = bool(getattr(args, "cmct_component_local_auto_select", False))
    ckpt["config"] = config
    ckpt["cmct_component_local_selection"] = selection
    torch.save(ckpt, path)


@torch.no_grad()
def select_memory_ratio_budget(
    args,
    val_items: Sequence[dict],
    processor,
    change_module,
    device,
    clipper,
    bank,
    correction_model,
    component_model=None,
) -> Dict[str, object]:
    if not bool(getattr(args, "cmct_enable_memory_ratio_budget", False)):
        return {"enabled": False, "reason": "memory_ratio_budget_disabled"}
    if not bool(getattr(args, "cmct_memory_ratio_auto_select", False)):
        return {
            "enabled": False,
            "reason": "auto_select_disabled",
            "scale": float(getattr(args, "cmct_memory_ratio_scale", 1.0)),
            "max": float(getattr(args, "cmct_memory_ratio_max", 0.50)),
        }
    if len(val_items) == 0:
        return {"enabled": True, "selected": False, "reason": "empty_source_val"}

    scales = [float(x) for x in base.parse_ratio_grid(args.cmct_memory_ratio_scale_grid)]
    max_values = [float(x) for x in base.parse_ratio_grid(args.cmct_memory_ratio_max_grid)]
    ratio_min = float(getattr(args, "cmct_memory_ratio_min", 1e-4))
    candidates = [
        (scale, max_value)
        for scale in scales
        for max_value in max_values
        if scale > 0 and max_value >= ratio_min
    ]
    if not candidates:
        return {"enabled": True, "selected": False, "reason": "empty_candidate_grid"}

    cfg = base._memory_config_from_args(args)
    class_names, prompt_bank, confusers = source_prompt_spec(args)
    selection_method = _resolve_ratio_selection_method(args, component_model)
    if correction_model is not None:
        correction_model.eval()
    if component_model is not None:
        component_model.eval()

    old_scale = float(args.cmct_memory_ratio_scale)
    old_max = float(args.cmct_memory_ratio_max)
    confusions = OrderedDict((candidate, np.zeros((2, 2), dtype=np.float64)) for candidate in candidates)

    for item in tqdm(val_items, desc="select-ratio-budget", dynamic_ncols=True):
        meta, _, _, size_hw = bench.predict_change(processor, change_module, item, args, device)
        gt = bench.read_binary_mask(item["label"], size_hw).astype(np.int64)
        pair_embedding = clipper.encode_pair(item)
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
        pred_by_candidate = OrderedDict(
            (candidate, np.zeros(size_hw, dtype=np.int64))
            for candidate in candidates
        )
        for cls_idx, cls_name in enumerate(class_names):
            score = memory_rectified_score(
                posterior.posterior_a[cls_idx],
                posterior.posterior_b[cls_idx],
                None if args.disable_bcd_gate else meta["bcd_score"],
                presence_a=presence_a[cls_idx],
                presence_b=presence_b[cls_idx],
                config=cfg,
            )
            presence = torch.maximum(presence_a[cls_idx], presence_b[cls_idx])
            base_mask, base_threshold = base.decode_class_score(score, presence, 0.0, args)
            component_features = component_reliability_features(
                args,
                base_mask,
                score,
                meta["bcd_score"],
                base_threshold,
                component_model,
                device,
            )
            semantic_features = semantic_relation_features_for_prompt(args, bank, clipper, cls_name, prompt_bank)
            key, stats = make_query(args, item, cls_name, prompt_bank, pair_embedding, clipper, score, meta["bcd_score"], presence)
            _, tau_corr, diag = predict_cmct_thresholds(
                args,
                bank,
                correction_model,
                key,
                stats,
                device,
                cls_name=cls_name,
                component_features=component_features,
                semantic_features=semantic_features,
            )
            tau_corr = apply_adaptive_floor(tau_corr, base_threshold, args)
            tau_corr = apply_class_prior_floor(tau_corr, args, bank, cls_name)
            tau_corr, semantic_prior_diag = apply_semantic_prior_floor(tau_corr, args, semantic_features)
            diag.update({f"semantic_prior_{k}": v for k, v in semantic_prior_diag.items() if isinstance(v, (int, float, bool))})
            tau_corr, uncertainty_diag = apply_uncertainty_calibration(
                tau_corr,
                base_threshold,
                args,
                stats,
                diag,
                component_features=component_features,
                semantic_features=semantic_features,
            )
            diag.update({f"uncertainty_{k}": v for k, v in uncertainty_diag.items() if isinstance(v, (int, float, bool))})
            ratio_mem = diag.get("ratio_mem")

            for scale, max_value in candidates:
                args.cmct_memory_ratio_scale = float(scale)
                args.cmct_memory_ratio_max = float(max_value)
                mask = _decode_ratio_selection_mask(
                    args,
                    score,
                    meta["bcd_score"],
                    presence,
                    tau_corr,
                    ratio_mem,
                    selection_method,
                    component_model,
                    device,
                )
                pred_by_candidate[(scale, max_value)] |= mask.astype(np.int64)

        for candidate, pred in pred_by_candidate.items():
            confusions[candidate] += bench.confusion_binary(gt, pred.astype(np.int64))

    entries = []
    best_entry = None
    best_score = -float("inf")
    for scale, max_value in candidates:
        cm = confusions[(scale, max_value)]
        metrics = {k: float(v) for k, v in bench.metrics_from_confusion(cm).items()}
        score_value = _score_ratio_budget_candidate(metrics, args)
        entry = {
            "scale": float(scale),
            "max": float(max_value),
            "score": float(score_value),
            "metrics": metrics,
            "confusion": cm.tolist(),
        }
        entries.append(entry)
        if score_value > best_score:
            best_score = score_value
            best_entry = entry

    if best_entry is None:
        args.cmct_memory_ratio_scale = old_scale
        args.cmct_memory_ratio_max = old_max
        return {"enabled": True, "selected": False, "reason": "selection_failed", "candidates": entries}

    args.cmct_memory_ratio_scale = float(best_entry["scale"])
    args.cmct_memory_ratio_max = float(best_entry["max"])
    print(
        "Selected memory ratio budget: "
        f"scale={args.cmct_memory_ratio_scale:.4f}, "
        f"max={args.cmct_memory_ratio_max:.4f}, "
        f"method={selection_method}, "
        f"source_val_score={best_entry['score']:.4f}"
    )
    return {
        "enabled": True,
        "selected": True,
        "method": selection_method,
        "metric": str(getattr(args, "cmct_memory_ratio_select_metric", "IoU")),
        "num_val_items": int(len(val_items)),
        "source_classes": class_names,
        "old_scale": float(old_scale),
        "old_max": float(old_max),
        "selected": best_entry,
        "candidates": entries,
    }


def update_cmct_checkpoint_ratio_selection(args, selection: Dict[str, object]) -> None:
    if not selection or not bool(selection.get("enabled")) or not bool(selection.get("selected")):
        return
    if not args.output_checkpoint:
        return
    path = Path(args.output_checkpoint)
    if not path.exists():
        return
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    config = dict(ckpt.get("config", {}))
    config["cmct_memory_ratio_scale"] = float(args.cmct_memory_ratio_scale)
    config["cmct_memory_ratio_max"] = float(args.cmct_memory_ratio_max)
    config["cmct_memory_ratio_auto_select"] = bool(getattr(args, "cmct_memory_ratio_auto_select", False))
    ckpt["config"] = config
    ckpt["cmct_memory_ratio_selection"] = selection
    torch.save(ckpt, path)


def _score_uncertainty_candidate(metrics: Dict[str, float], args) -> float:
    metric_name = str(getattr(args, "cmct_uncertainty_select_metric", "IoU"))
    value = float(metrics.get(metric_name, 0.0))
    min_recall = float(getattr(args, "cmct_uncertainty_min_recall", 0.45))
    recall = float(metrics.get("Recall", 0.0))
    if recall < min_recall:
        value -= float(getattr(args, "cmct_uncertainty_recall_penalty", 1.0)) * (min_recall - recall)
    return float(value)


@torch.no_grad()
def select_uncertainty_calibration(
    args,
    val_items: Sequence[dict],
    processor,
    change_module,
    device,
    clipper,
    bank,
    correction_model,
    component_model=None,
) -> Dict[str, object]:
    if not bool(getattr(args, "cmct_enable_uncertainty_calibration", False)):
        return {"enabled": False, "reason": "uncertainty_calibration_disabled"}
    if not bool(getattr(args, "cmct_uncertainty_auto_select", False)):
        return {
            "enabled": True,
            "selected": False,
            "reason": "auto_select_disabled",
            "strength": float(getattr(args, "cmct_uncertainty_strength", 0.50)),
            "min_mix": float(getattr(args, "cmct_uncertainty_min_mix", 0.0)),
            "max_mix": float(getattr(args, "cmct_uncertainty_max_mix", 1.0)),
        }
    if len(val_items) == 0:
        return {"enabled": True, "selected": False, "reason": "empty_source_val"}

    strengths = [float(x) for x in base.parse_ratio_grid(args.cmct_uncertainty_strength_grid)]
    min_mixes = [float(x) for x in base.parse_ratio_grid(args.cmct_uncertainty_min_mix_grid)]
    max_mix = float(getattr(args, "cmct_uncertainty_max_mix", 1.0))
    candidates = [
        (strength, min_mix)
        for strength in strengths
        for min_mix in min_mixes
        if strength >= 0.0 and 0.0 <= min_mix <= max_mix
    ]
    if not candidates:
        return {"enabled": True, "selected": False, "reason": "empty_candidate_grid"}

    cfg = base._memory_config_from_args(args)
    class_names, prompt_bank, confusers = source_prompt_spec(args)
    selection_method = _resolve_ratio_selection_method(args, component_model)
    if correction_model is not None:
        correction_model.eval()
    if component_model is not None:
        component_model.eval()

    old_strength = float(args.cmct_uncertainty_strength)
    old_min_mix = float(args.cmct_uncertainty_min_mix)
    old_max_mix = float(args.cmct_uncertainty_max_mix)
    args.cmct_uncertainty_max_mix = max_mix
    confusions = OrderedDict((candidate, np.zeros((2, 2), dtype=np.float64)) for candidate in candidates)

    for item in tqdm(val_items, desc="select-uncertainty-calibration", dynamic_ncols=True):
        meta, _, _, size_hw = bench.predict_change(processor, change_module, item, args, device)
        gt = bench.read_binary_mask(item["label"], size_hw).astype(np.int64)
        pair_embedding = clipper.encode_pair(item)
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
        pred_by_candidate = OrderedDict(
            (candidate, np.zeros(size_hw, dtype=np.int64))
            for candidate in candidates
        )
        for cls_idx, cls_name in enumerate(class_names):
            score = memory_rectified_score(
                posterior.posterior_a[cls_idx],
                posterior.posterior_b[cls_idx],
                None if args.disable_bcd_gate else meta["bcd_score"],
                presence_a=presence_a[cls_idx],
                presence_b=presence_b[cls_idx],
                config=cfg,
            )
            presence = torch.maximum(presence_a[cls_idx], presence_b[cls_idx])
            base_mask, base_threshold = base.decode_class_score(score, presence, 0.0, args)
            component_features = component_reliability_features(
                args,
                base_mask,
                score,
                meta["bcd_score"],
                base_threshold,
                component_model,
                device,
            )
            semantic_features = semantic_relation_features_for_prompt(args, bank, clipper, cls_name, prompt_bank)
            key, stats = make_query(args, item, cls_name, prompt_bank, pair_embedding, clipper, score, meta["bcd_score"], presence)
            _, tau_corr, diag = predict_cmct_thresholds(
                args,
                bank,
                correction_model,
                key,
                stats,
                device,
                cls_name=cls_name,
                component_features=component_features,
                semantic_features=semantic_features,
            )
            tau_corr = apply_adaptive_floor(tau_corr, base_threshold, args)
            tau_corr = apply_class_prior_floor(tau_corr, args, bank, cls_name)
            tau_corr, semantic_prior_diag = apply_semantic_prior_floor(tau_corr, args, semantic_features)
            diag.update({f"semantic_prior_{k}": v for k, v in semantic_prior_diag.items() if isinstance(v, (int, float, bool))})
            ratio_mem = diag.get("ratio_mem") if bool(getattr(args, "cmct_enable_memory_ratio_budget", False)) else None

            for strength, min_mix in candidates:
                args.cmct_uncertainty_strength = float(strength)
                args.cmct_uncertainty_min_mix = float(min_mix)
                calibrated_tau, _ = apply_uncertainty_calibration(
                    tau_corr,
                    base_threshold,
                    args,
                    stats,
                    diag,
                    component_features=component_features,
                    semantic_features=semantic_features,
                )
                mask = _decode_ratio_selection_mask(
                    args,
                    score,
                    meta["bcd_score"],
                    presence,
                    calibrated_tau,
                    ratio_mem,
                    selection_method,
                    component_model,
                    device,
                )
                pred_by_candidate[(strength, min_mix)] |= mask.astype(np.int64)

        for candidate, pred in pred_by_candidate.items():
            confusions[candidate] += bench.confusion_binary(gt, pred.astype(np.int64))

    entries = []
    best_entry = None
    best_score = -float("inf")
    for strength, min_mix in candidates:
        cm = confusions[(strength, min_mix)]
        metrics = {k: float(v) for k, v in bench.metrics_from_confusion(cm).items()}
        score_value = _score_uncertainty_candidate(metrics, args)
        entry = {
            "strength": float(strength),
            "min_mix": float(min_mix),
            "max_mix": float(max_mix),
            "score": float(score_value),
            "metrics": metrics,
            "confusion": cm.tolist(),
        }
        entries.append(entry)
        if score_value > best_score:
            best_score = score_value
            best_entry = entry

    if best_entry is None:
        args.cmct_uncertainty_strength = old_strength
        args.cmct_uncertainty_min_mix = old_min_mix
        args.cmct_uncertainty_max_mix = old_max_mix
        return {"enabled": True, "selected": False, "reason": "selection_failed", "candidates": entries}

    args.cmct_uncertainty_strength = float(best_entry["strength"])
    args.cmct_uncertainty_min_mix = float(best_entry["min_mix"])
    args.cmct_uncertainty_max_mix = float(best_entry["max_mix"])
    print(
        "Selected uncertainty calibration: "
        f"strength={args.cmct_uncertainty_strength:.4f}, "
        f"min_mix={args.cmct_uncertainty_min_mix:.4f}, "
        f"max_mix={args.cmct_uncertainty_max_mix:.4f}, "
        f"method={selection_method}, "
        f"source_val_score={best_entry['score']:.4f}"
    )
    return {
        "enabled": True,
        "selected": True,
        "method": selection_method,
        "metric": str(getattr(args, "cmct_uncertainty_select_metric", "IoU")),
        "num_val_items": int(len(val_items)),
        "source_classes": class_names,
        "old_strength": float(old_strength),
        "old_min_mix": float(old_min_mix),
        "old_max_mix": float(old_max_mix),
        "selected": best_entry,
        "candidates": entries,
    }


def update_cmct_checkpoint_uncertainty_selection(args, selection: Dict[str, object]) -> None:
    if not selection or not bool(selection.get("enabled")) or not bool(selection.get("selected")):
        return
    if not args.output_checkpoint:
        return
    path = Path(args.output_checkpoint)
    if not path.exists():
        return
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    config = dict(ckpt.get("config", {}))
    config["cmct_uncertainty_strength"] = float(args.cmct_uncertainty_strength)
    config["cmct_uncertainty_min_mix"] = float(args.cmct_uncertainty_min_mix)
    config["cmct_uncertainty_max_mix"] = float(args.cmct_uncertainty_max_mix)
    config["cmct_uncertainty_auto_select"] = bool(getattr(args, "cmct_uncertainty_auto_select", False))
    ckpt["config"] = config
    ckpt["cmct_uncertainty_selection"] = selection
    torch.save(ckpt, path)


def _build_meta_threshold_rows(
    args,
    val_items: Sequence[dict],
    processor,
    change_module,
    device,
    clipper,
    bank,
    correction_model,
    component_model=None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, object]]:
    cfg = base._memory_config_from_args(args)
    class_names, prompt_bank, confusers = source_prompt_spec(args)
    if correction_model is not None:
        correction_model.eval()
    if component_model is not None:
        component_model.eval()

    x_rows = []
    y_rows = []
    tau_corr_rows = []
    target_rows = []
    for item in tqdm(val_items, desc="collect-meta-threshold", dynamic_ncols=True):
        meta, _, _, size_hw = bench.predict_change(processor, change_module, item, args, device)
        gt = bench.read_binary_mask(item["label"], size_hw).astype(bool)
        pair_embedding = clipper.encode_pair(item)
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
        for cls_idx, cls_name in enumerate(class_names):
            score = memory_rectified_score(
                posterior.posterior_a[cls_idx],
                posterior.posterior_b[cls_idx],
                None if args.disable_bcd_gate else meta["bcd_score"],
                presence_a=presence_a[cls_idx],
                presence_b=presence_b[cls_idx],
                config=cfg,
            )
            presence = torch.maximum(presence_a[cls_idx], presence_b[cls_idx])
            base_mask, base_threshold = base.decode_class_score(score, presence, 0.0, args)
            component_features = component_reliability_features(
                args,
                base_mask,
                score,
                meta["bcd_score"],
                base_threshold,
                component_model,
                device,
            )
            semantic_features = semantic_relation_features_for_prompt(args, bank, clipper, cls_name, prompt_bank)
            key, stats = make_query(args, item, cls_name, prompt_bank, pair_embedding, clipper, score, meta["bcd_score"], presence)
            tau_mem, tau_corr, diag = predict_cmct_thresholds(
                args,
                bank,
                correction_model,
                key,
                stats,
                device,
                cls_name=cls_name,
                component_features=component_features,
                semantic_features=semantic_features,
            )
            tau_mem = apply_adaptive_floor(tau_mem, base_threshold, args)
            tau_corr = apply_adaptive_floor(tau_corr, base_threshold, args)
            tau_mem = apply_class_prior_floor(tau_mem, args, bank, cls_name)
            tau_corr = apply_class_prior_floor(tau_corr, args, bank, cls_name)
            tau_corr, semantic_prior_diag = apply_semantic_prior_floor(tau_corr, args, semantic_features)
            diag.update({f"semantic_prior_{k}": v for k, v in semantic_prior_diag.items() if isinstance(v, (int, float, bool))})
            tau_corr, uncertainty_diag = apply_uncertainty_calibration(
                tau_corr,
                base_threshold,
                args,
                stats,
                diag,
                component_features=component_features,
                semantic_features=semantic_features,
            )
            diag.update({f"uncertainty_{k}": v for k, v in uncertainty_diag.items() if isinstance(v, (int, float, bool))})

            target_tau = oracle_threshold_for_score(score, gt, args)
            feature_row = cmct_meta_features(
                args,
                bank,
                stats,
                tau_mem,
                tau_corr,
                base_threshold,
                diag,
                cls_name=cls_name,
                component_features=component_features,
                semantic_features=semantic_features,
                uncertainty_diag=uncertainty_diag,
            )
            x_rows.append(feature_row)
            y_rows.append(float(target_tau) - float(tau_corr))
            tau_corr_rows.append(float(tau_corr))
            target_rows.append(float(target_tau))

    input_dim = cmct_meta_feature_dim(args)
    if not x_rows:
        return (
            np.zeros((0, input_dim), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            {"num_rows": 0, "num_val_items": int(len(val_items))},
        )
    x_np = np.stack(x_rows, axis=0).astype(np.float32)
    y_np = np.asarray(y_rows, dtype=np.float32)
    tau_corr_np = np.asarray(tau_corr_rows, dtype=np.float32)
    target_np = np.asarray(target_rows, dtype=np.float32)
    stats = {
        "num_rows": int(x_np.shape[0]),
        "num_val_items": int(len(val_items)),
        "target_tau_mean": float(target_np.mean()),
        "target_tau_min": float(target_np.min()),
        "target_tau_max": float(target_np.max()),
        "pre_meta_mae": float(np.mean(np.abs(tau_corr_np - target_np))),
        "target_delta_mean": float(y_np.mean()),
        "target_delta_min": float(y_np.min()),
        "target_delta_max": float(y_np.max()),
    }
    return x_np, y_np, tau_corr_np, stats


def train_meta_threshold_head(
    args,
    val_items: Sequence[dict],
    processor,
    change_module,
    device,
    clipper,
    bank,
    correction_model,
    component_model=None,
):
    input_dim = cmct_meta_feature_dim(args)
    model = CorrectionHead(
        input_dim,
        hidden_dim=int(getattr(args, "cmct_meta_hidden_dim", 64)),
        dropout=float(getattr(args, "cmct_meta_dropout", 0.05)),
        max_delta=float(getattr(args, "cmct_meta_max_delta", 0.20)),
    ).to(device)
    if len(val_items) == 0:
        return model, {"enabled": True, "trained": False, "reason": "empty_source_val", "input_dim": int(input_dim)}

    x_np, y_np, tau_corr_np, stats = _build_meta_threshold_rows(
        args,
        val_items,
        processor,
        change_module,
        device,
        clipper,
        bank,
        correction_model,
        component_model,
    )
    if x_np.shape[0] <= 1:
        stats.update({"enabled": True, "trained": False, "reason": "not_enough_rows", "input_dim": int(input_dim)})
        return model, stats

    x = torch.tensor(x_np, dtype=torch.float32, device=device)
    y = torch.tensor(y_np, dtype=torch.float32, device=device)
    tau_corr = torch.tensor(tau_corr_np, dtype=torch.float32, device=device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(getattr(args, "cmct_meta_lr", 1e-3)),
        weight_decay=float(getattr(args, "cmct_meta_weight_decay", 1e-4)),
    )
    batch_size = max(int(getattr(args, "cmct_meta_batch_size", 64)), 1)
    for _ in range(int(getattr(args, "cmct_meta_epochs", 80))):
        model.train()
        perm = torch.randperm(x.shape[0], device=device)
        for start in range(0, x.shape[0], batch_size):
            idx = perm[start:start + batch_size]
            pred = model(x[idx])
            loss = F.smooth_l1_loss(pred, y[idx])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        pred = model(x)
        train_loss = F.smooth_l1_loss(pred, y)
        target_tau = tau_corr + y
        pred_tau = torch.clamp(
            tau_corr + pred,
            float(args.cmct_threshold_min),
            float(args.cmct_threshold_max),
        )
        post_mae = torch.mean(torch.abs(pred_tau - target_tau))
    stats.update({
        "enabled": True,
        "trained": True,
        "input_dim": int(input_dim),
        "smooth_l1": float(train_loss.detach().cpu()),
        "post_meta_mae": float(post_mae.detach().cpu()),
        "pred_delta_mean": float(pred.mean().detach().cpu()),
        "pred_delta_min": float(pred.min().detach().cpu()),
        "pred_delta_max": float(pred.max().detach().cpu()),
    })
    return model, stats


def update_cmct_checkpoint_meta_threshold(args, meta_model, stats: Dict[str, object]) -> None:
    if meta_model is None or not args.output_checkpoint:
        return
    path = Path(args.output_checkpoint)
    source_path = path if path.exists() else (Path(args.load_checkpoint) if args.load_checkpoint else None)
    if source_path is None or not source_path.exists():
        return
    ckpt = torch.load(source_path, map_location="cpu", weights_only=False)
    config = dict(ckpt.get("config", {}))
    config.update({
        "cmct_enable_meta_threshold": bool(getattr(args, "cmct_enable_meta_threshold", False)),
        "cmct_meta_hidden_dim": int(getattr(args, "cmct_meta_hidden_dim", 64)),
        "cmct_meta_dropout": float(getattr(args, "cmct_meta_dropout", 0.05)),
        "cmct_meta_epochs": int(getattr(args, "cmct_meta_epochs", 80)),
        "cmct_meta_batch_size": int(getattr(args, "cmct_meta_batch_size", 64)),
        "cmct_meta_lr": float(getattr(args, "cmct_meta_lr", 1e-3)),
        "cmct_meta_weight_decay": float(getattr(args, "cmct_meta_weight_decay", 1e-4)),
        "cmct_meta_max_delta": float(getattr(args, "cmct_meta_max_delta", 0.20)),
    })
    ckpt["config"] = config
    ckpt["meta_threshold_head"] = meta_model.state_dict()
    ckpt["cmct_meta_threshold_stats"] = stats
    ckpt["cmct_meta_feature_names"] = cmct_meta_feature_names(args)
    torch.save(ckpt, path)
    print(f"Saved CMCT meta threshold checkpoint: {path}")


def load_or_train_meta_threshold(
    args,
    val_items: Sequence[dict],
    processor,
    change_module,
    device,
    clipper,
    bank,
    correction_model,
    component_model=None,
):
    if not bool(getattr(args, "cmct_enable_meta_threshold", False)):
        return None, {"enabled": False, "reason": "meta_threshold_disabled"}

    input_dim = cmct_meta_feature_dim(args)
    if args.load_checkpoint and not bool(getattr(args, "cmct_meta_force_retrain", False)):
        ckpt = torch.load(args.load_checkpoint, map_location=device, weights_only=False)
        state = ckpt.get("meta_threshold_head")
        names = ckpt.get("cmct_meta_feature_names", [])
        if state is not None and len(names) == input_dim:
            model = CorrectionHead(
                input_dim,
                hidden_dim=int(ckpt.get("config", {}).get("cmct_meta_hidden_dim", getattr(args, "cmct_meta_hidden_dim", 64))),
                dropout=float(ckpt.get("config", {}).get("cmct_meta_dropout", getattr(args, "cmct_meta_dropout", 0.05))),
                max_delta=float(ckpt.get("config", {}).get("cmct_meta_max_delta", getattr(args, "cmct_meta_max_delta", 0.20))),
            ).to(device)
            model.load_state_dict(state)
            model.eval()
            stats = dict(ckpt.get("cmct_meta_threshold_stats", {}))
            stats.update({"enabled": True, "loaded_from_checkpoint": True, "input_dim": int(input_dim)})
            return model, stats

    model, stats = train_meta_threshold_head(
        args,
        val_items,
        processor,
        change_module,
        device,
        clipper,
        bank,
        correction_model,
        component_model,
    )
    update_cmct_checkpoint_meta_threshold(args, model, stats)
    return model, stats


def _append_threshold_ref_error(errors, method: str, dataset: str, cls_name: str, threshold: float):
    ref = FIXED_THRESHOLD_REFERENCE.get(f"{dataset}:{cls_name}")
    if ref is not None:
        errors[method].append(abs(float(threshold) - float(ref)))


def _summarize_threshold_logs(threshold_logs, errors):
    out = {}
    for method, values in threshold_logs.items():
        out[method] = {
            "mean": float(np.mean(values)) if values else 0.0,
            "min": float(np.min(values)) if values else 0.0,
            "max": float(np.max(values)) if values else 0.0,
            "fixed_ref_mae": float(np.mean(errors.get(method, []))) if errors.get(method) else None,
        }
    return out


def _append_class_threshold_log(class_logs, class_errors, method: str, dataset: str, cls_name: str, threshold: float):
    if method not in class_logs or cls_name not in class_logs[method]:
        return
    class_logs[method][cls_name].append(float(threshold))
    ref = FIXED_THRESHOLD_REFERENCE.get(f"{dataset}:{cls_name}")
    if ref is not None:
        class_errors[method][cls_name].append(abs(float(threshold) - float(ref)))


def _summarize_class_threshold_logs(class_logs, class_errors):
    out = OrderedDict()
    for method, per_class in class_logs.items():
        out[method] = OrderedDict()
        for cls_name, values in per_class.items():
            errors = class_errors.get(method, {}).get(cls_name, [])
            out[method][cls_name] = {
                "mean": float(np.mean(values)) if values else 0.0,
                "min": float(np.min(values)) if values else 0.0,
                "max": float(np.max(values)) if values else 0.0,
                "fixed_ref_mae": float(np.mean(errors)) if errors else None,
            }
    return out


@torch.no_grad()
def evaluate_semantic_targets(
    args,
    datasets,
    items_by_dataset,
    processor,
    change_module,
    device,
    clipper,
    bank,
    correction_model,
    component_model=None,
    component_verifier_model=None,
    suppressed_recovery_model=None,
    meta_model=None,
    class_residual_adapter=None,
    score_calibration_selection=None,
):
    cfg = base._memory_config_from_args(args)
    if correction_model is not None:
        correction_model.eval()
    if meta_model is not None:
        meta_model.eval()
    if component_verifier_model is not None:
        component_verifier_model.eval()
    if suppressed_recovery_model is not None:
        suppressed_recovery_model.eval()
    out = OrderedDict()
    method_names = [
        "adaptive_no_cap",
        "bcd_soft_gate",
        "clip_knn_tau",
        "clip_corrected_tau",
        "clip_corrected_tau_softgate",
    ]
    use_local = bool(getattr(args, "cmct_enable_component_local_refine", False))
    if use_local:
        method_names.append("clip_component_local_tau")
    use_proto = bool(getattr(args, "cmct_enable_class_prototype_threshold", False))
    if use_proto:
        method_names += [
            "clip_proto_tau",
            "clip_proto_tau_softgate",
        ]
    use_meta = bool(getattr(args, "cmct_enable_meta_threshold", False)) and meta_model is not None
    if use_meta:
        method_names += [
            "clip_meta_tau",
            "clip_meta_tau_softgate",
        ]
    use_adapter = bool(getattr(args, "cmct_enable_class_residual_adapter", False)) and class_residual_adapter is not None
    if use_adapter:
        method_names += [
            "clip_class_adapter_tau",
            "clip_class_adapter_tau_softgate",
        ]
    use_score_cal = bool(getattr(args, "cmct_enable_score_calibration", False))
    if use_score_cal:
        method_names += [
            "clip_score_calibrated_tau",
            "clip_score_calibrated_tau_softgate",
        ]
    use_semantic_prior_branch = bool(getattr(args, "cmct_enable_semantic_relation", False))
    if use_semantic_prior_branch:
        method_names += [
            "clip_semantic_prior_tau",
            "clip_semantic_prior_tau_softgate",
        ]
    use_semantic_anchor = bool(getattr(args, "cmct_enable_semantic_prior_anchor", False))
    if use_semantic_anchor:
        method_names += [
            "clip_semantic_anchor_tau",
            "clip_semantic_anchor_tau_softgate",
        ]
    use_seg2change_complement = (
        bool(args.enable_component_gate)
        and bool(getattr(args, "cmct_enable_seg2change_complement", False))
    )
    use_prat_lite = (
        bool(args.enable_component_gate)
        and bool(getattr(args, "cmct_enable_prat_lite", False))
    )
    if bool(args.enable_component_gate):
        method_names.append("clip_corrected_tau_component_gate")
        if bool(getattr(args, "cmct_enable_multilabel_onepass", False)):
            method_names.append("clip_multilabel_onepass_component_gate")
        if use_seg2change_complement:
            method_names.append("clip_seg2change_complement_component_gate")
        if use_prat_lite:
            method_names.append("clip_prat_lite_component_gate")
        if use_semantic_prior_branch:
            method_names.append("clip_semantic_prior_tau_component_gate")
        if use_semantic_anchor:
            method_names.append("clip_semantic_anchor_tau_component_gate")
        if use_proto:
            method_names.append("clip_proto_tau_component_gate")
        if use_meta:
            method_names.append("clip_meta_tau_component_gate")
        if use_adapter:
            method_names.append("clip_class_adapter_tau_component_gate")
        if use_score_cal:
            method_names.append("clip_score_calibrated_tau_component_gate")
    use_multiclass_competition = bool(getattr(args, "cmct_enable_multiclass_competition", False))
    if use_multiclass_competition:
        method_names.append("clip_multiclass_competition_gate")
    for dataset in datasets:
        spec = base.dataset_spec(dataset)
        class_names = spec["class_names"]
        accumulators = OrderedDict((name, base.new_accumulator(class_names)) for name in method_names)
        threshold_logs = OrderedDict((name, []) for name in method_names)
        ref_errors = OrderedDict((name, []) for name in method_names)
        class_threshold_logs = OrderedDict(
            (name, OrderedDict((cls, []) for cls in class_names))
            for name in method_names
        )
        class_ref_errors = OrderedDict(
            (name, OrderedDict((cls, []) for cls in class_names))
            for name in method_names
        )
        retrieval_logs = []
        multiclass_logs = []
        seg2change_logs = []
        prat_logs = []
        for item in tqdm(items_by_dataset[dataset], desc=f"eval-CMCT-{dataset}", dynamic_ncols=True):
            meta, _, bcd_np, size_hw = bench.predict_change(processor, change_module, item, args, device)
            gt_masks = base.build_gt_masks(item, size_hw, spec)
            gt_cd = bench.read_binary_mask(item["gt_cd"], size_hw)
            bcd_cm = bench.confusion_binary(gt_cd, bcd_np)
            pair_embedding = clipper.encode_pair(item)
            scores_a, presence_a = extract_class_scores(processor, meta["state_a"], spec["prompt_bank"], class_names)
            scores_b, presence_b = extract_class_scores(processor, meta["state_b"], spec["prompt_bank"], class_names)
            seg2change_support_by_class = {}
            if use_seg2change_complement:
                seg2change_support_by_class, support_diag = seg2change_support_masks_from_scores(
                    args,
                    class_names,
                    scores_a,
                    scores_b,
                    bcd_np,
                )
                seg2change_logs.append(support_diag)
            class_text_embeddings = np.stack([
                _clipper_text_embedding(clipper, prompt_text_for_class(name, spec["prompt_bank"]))
                for name in class_names
            ], axis=0).astype(np.float32)
            posterior = build_posterior_inputs(
                scores_a,
                scores_b,
                presence_a,
                presence_b,
                class_names=class_names,
                confusers=spec["confusers"],
                rho=args.rho,
            )
            pred_by_method = {
                name: {cls: np.zeros(size_hw, dtype=np.int64) for cls in class_names}
                for name in method_names
            }
            multiclass_score_maps = {}
            multiclass_thresholds = {}
            multiclass_candidate_masks = {}
            multiclass_endpoint_maps_a = {}
            multiclass_endpoint_maps_b = {}
            for cls_idx, cls_name in enumerate(class_names):
                score = memory_rectified_score(
                    posterior.posterior_a[cls_idx],
                    posterior.posterior_b[cls_idx],
                    None if args.disable_bcd_gate else meta["bcd_score"],
                    presence_a=presence_a[cls_idx],
                    presence_b=presence_b[cls_idx],
                    config=cfg,
                )
                presence = torch.maximum(presence_a[cls_idx], presence_b[cls_idx])
                if use_multiclass_competition:
                    multiclass_score_maps[cls_name] = score.detach().float().cpu().numpy().astype(np.float32)
                    multiclass_endpoint_maps_a[cls_name] = posterior.posterior_a[cls_idx].detach().float().cpu().numpy().astype(np.float32)
                    multiclass_endpoint_maps_b[cls_name] = posterior.posterior_b[cls_idx].detach().float().cpu().numpy().astype(np.float32)

                base_mask, base_threshold = base.decode_class_score(score, presence, 0.0, args)
                pred_by_method["adaptive_no_cap"][cls_name] = base_mask
                threshold_logs["adaptive_no_cap"].append(base_threshold)
                _append_threshold_ref_error(ref_errors, "adaptive_no_cap", dataset, cls_name, base_threshold)
                _append_class_threshold_log(class_threshold_logs, class_ref_errors, "adaptive_no_cap", dataset, cls_name, base_threshold)

                soft_mask = base.bcd_soft_gate_filter(base_mask, score, meta["bcd_score"], base_threshold, args)
                pred_by_method["bcd_soft_gate"][cls_name] = soft_mask
                threshold_logs["bcd_soft_gate"].append(base_threshold)
                _append_threshold_ref_error(ref_errors, "bcd_soft_gate", dataset, cls_name, base_threshold)
                _append_class_threshold_log(class_threshold_logs, class_ref_errors, "bcd_soft_gate", dataset, cls_name, base_threshold)

                component_features = component_reliability_features(
                    args,
                    base_mask,
                    score,
                    meta["bcd_score"],
                    base_threshold,
                    component_model,
                    device,
                )
                semantic_features = semantic_relation_features_for_prompt(args, bank, clipper, cls_name, spec["prompt_bank"])
                key, stats = make_query(args, item, cls_name, spec["prompt_bank"], pair_embedding, clipper, score, meta["bcd_score"], presence)
                tau_mem, tau_corr, diag = predict_cmct_thresholds(
                    args,
                    bank,
                    correction_model,
                    key,
                    stats,
                    device,
                    cls_name=cls_name,
                    component_features=component_features,
                    semantic_features=semantic_features,
                )
                retrieval_logs.append(diag)
                tau_mem = apply_adaptive_floor(tau_mem, base_threshold, args)
                tau_corr = apply_adaptive_floor(tau_corr, base_threshold, args)
                tau_mem = apply_class_prior_floor(tau_mem, args, bank, cls_name)
                tau_corr = apply_class_prior_floor(tau_corr, args, bank, cls_name)
                tau_corr, semantic_prior_diag = apply_semantic_prior_floor(tau_corr, args, semantic_features)
                diag.update({f"semantic_prior_{k}": v for k, v in semantic_prior_diag.items() if isinstance(v, (int, float, bool))})
                tau_corr, uncertainty_diag = apply_uncertainty_calibration(
                    tau_corr,
                    base_threshold,
                    args,
                    stats,
                    diag,
                    component_features=component_features,
                    semantic_features=semantic_features,
                )
                diag.update({f"uncertainty_{k}": v for k, v in uncertainty_diag.items() if isinstance(v, (int, float, bool))})

                ratio_mem = diag.get("ratio_mem") if bool(getattr(args, "cmct_enable_memory_ratio_budget", False)) else None
                knn_mask, knn_threshold = decode_fixed_tau(score, presence, tau_mem, args, budget_ratio=ratio_mem)
                pred_by_method["clip_knn_tau"][cls_name] = knn_mask
                threshold_logs["clip_knn_tau"].append(knn_threshold)
                _append_threshold_ref_error(ref_errors, "clip_knn_tau", dataset, cls_name, knn_threshold)
                _append_class_threshold_log(class_threshold_logs, class_ref_errors, "clip_knn_tau", dataset, cls_name, knn_threshold)

                corr_mask, corr_threshold = decode_fixed_tau(score, presence, tau_corr, args, budget_ratio=ratio_mem)
                pred_by_method["clip_corrected_tau"][cls_name] = corr_mask
                threshold_logs["clip_corrected_tau"].append(corr_threshold)
                _append_threshold_ref_error(ref_errors, "clip_corrected_tau", dataset, cls_name, corr_threshold)
                _append_class_threshold_log(class_threshold_logs, class_ref_errors, "clip_corrected_tau", dataset, cls_name, corr_threshold)
                if use_multiclass_competition:
                    multiclass_thresholds[cls_name] = float(corr_threshold)
                    multiclass_candidate_masks[cls_name] = corr_mask

                corr_soft = base.bcd_soft_gate_filter(corr_mask, score, meta["bcd_score"], corr_threshold, args)
                pred_by_method["clip_corrected_tau_softgate"][cls_name] = corr_soft
                threshold_logs["clip_corrected_tau_softgate"].append(corr_threshold)
                _append_threshold_ref_error(ref_errors, "clip_corrected_tau_softgate", dataset, cls_name, corr_threshold)
                _append_class_threshold_log(class_threshold_logs, class_ref_errors, "clip_corrected_tau_softgate", dataset, cls_name, corr_threshold)
                if use_semantic_prior_branch:
                    pred_by_method["clip_semantic_prior_tau"][cls_name] = corr_mask
                    threshold_logs["clip_semantic_prior_tau"].append(corr_threshold)
                    _append_threshold_ref_error(ref_errors, "clip_semantic_prior_tau", dataset, cls_name, corr_threshold)
                    _append_class_threshold_log(class_threshold_logs, class_ref_errors, "clip_semantic_prior_tau", dataset, cls_name, corr_threshold)
                    pred_by_method["clip_semantic_prior_tau_softgate"][cls_name] = corr_soft
                    threshold_logs["clip_semantic_prior_tau_softgate"].append(corr_threshold)
                    _append_threshold_ref_error(ref_errors, "clip_semantic_prior_tau_softgate", dataset, cls_name, corr_threshold)
                    _append_class_threshold_log(class_threshold_logs, class_ref_errors, "clip_semantic_prior_tau_softgate", dataset, cls_name, corr_threshold)
                if use_semantic_anchor:
                    tau_anchor, anchor_diag = apply_semantic_prior_anchor(tau_corr, args, semantic_features)
                    diag.update({f"semantic_anchor_{k}": v for k, v in anchor_diag.items() if isinstance(v, (int, float, bool))})
                    anchor_mask, anchor_threshold = decode_fixed_tau(score, presence, tau_anchor, args, budget_ratio=ratio_mem)
                    pred_by_method["clip_semantic_anchor_tau"][cls_name] = anchor_mask
                    threshold_logs["clip_semantic_anchor_tau"].append(anchor_threshold)
                    _append_threshold_ref_error(ref_errors, "clip_semantic_anchor_tau", dataset, cls_name, anchor_threshold)
                    _append_class_threshold_log(class_threshold_logs, class_ref_errors, "clip_semantic_anchor_tau", dataset, cls_name, anchor_threshold)
                    anchor_soft = base.bcd_soft_gate_filter(anchor_mask, score, meta["bcd_score"], anchor_threshold, args)
                    pred_by_method["clip_semantic_anchor_tau_softgate"][cls_name] = anchor_soft
                    threshold_logs["clip_semantic_anchor_tau_softgate"].append(anchor_threshold)
                    _append_threshold_ref_error(ref_errors, "clip_semantic_anchor_tau_softgate", dataset, cls_name, anchor_threshold)
                    _append_class_threshold_log(class_threshold_logs, class_ref_errors, "clip_semantic_anchor_tau_softgate", dataset, cls_name, anchor_threshold)
                if use_score_cal:
                    calibrated_score, score_cal_diag = apply_score_calibration(
                        score,
                        meta["bcd_score"],
                        args,
                        bank,
                        clipper,
                        cls_name,
                        spec["prompt_bank"],
                        stats,
                        semantic_features=semantic_features,
                        component_features=component_features,
                    )
                    diag.update({f"score_cal_{k}": v for k, v in score_cal_diag.items() if isinstance(v, (int, float, bool))})
                    score_cal_mask, score_cal_threshold = decode_fixed_tau(calibrated_score, presence, tau_corr, args, budget_ratio=ratio_mem)
                    pred_by_method["clip_score_calibrated_tau"][cls_name] = score_cal_mask
                    threshold_logs["clip_score_calibrated_tau"].append(score_cal_threshold)
                    _append_threshold_ref_error(ref_errors, "clip_score_calibrated_tau", dataset, cls_name, score_cal_threshold)
                    _append_class_threshold_log(class_threshold_logs, class_ref_errors, "clip_score_calibrated_tau", dataset, cls_name, score_cal_threshold)
                    score_cal_soft = base.bcd_soft_gate_filter(score_cal_mask, calibrated_score, meta["bcd_score"], score_cal_threshold, args)
                    pred_by_method["clip_score_calibrated_tau_softgate"][cls_name] = score_cal_soft
                    threshold_logs["clip_score_calibrated_tau_softgate"].append(score_cal_threshold)
                    _append_threshold_ref_error(ref_errors, "clip_score_calibrated_tau_softgate", dataset, cls_name, score_cal_threshold)
                    _append_class_threshold_log(class_threshold_logs, class_ref_errors, "clip_score_calibrated_tau_softgate", dataset, cls_name, score_cal_threshold)
                if use_adapter:
                    tau_adapter, adapter_diag = apply_class_residual_adapter(
                        tau_corr,
                        args,
                        class_residual_adapter,
                        bank,
                        clipper,
                        cls_name,
                        spec["prompt_bank"],
                        semantic_features,
                        component_features,
                    )
                    diag.update({f"class_adapter_{k}": v for k, v in adapter_diag.items() if isinstance(v, (int, float, bool))})
                    adapter_mask, adapter_threshold = decode_fixed_tau(score, presence, tau_adapter, args, budget_ratio=ratio_mem)
                    pred_by_method["clip_class_adapter_tau"][cls_name] = adapter_mask
                    threshold_logs["clip_class_adapter_tau"].append(adapter_threshold)
                    _append_threshold_ref_error(ref_errors, "clip_class_adapter_tau", dataset, cls_name, adapter_threshold)
                    _append_class_threshold_log(class_threshold_logs, class_ref_errors, "clip_class_adapter_tau", dataset, cls_name, adapter_threshold)
                    adapter_soft = base.bcd_soft_gate_filter(adapter_mask, score, meta["bcd_score"], adapter_threshold, args)
                    pred_by_method["clip_class_adapter_tau_softgate"][cls_name] = adapter_soft
                    threshold_logs["clip_class_adapter_tau_softgate"].append(adapter_threshold)
                    _append_threshold_ref_error(ref_errors, "clip_class_adapter_tau_softgate", dataset, cls_name, adapter_threshold)
                    _append_class_threshold_log(class_threshold_logs, class_ref_errors, "clip_class_adapter_tau_softgate", dataset, cls_name, adapter_threshold)
                if use_local:
                    class_text_embedding = class_text_embeddings[cls_idx]
                    local_mask, local_threshold, local_diag = component_local_refine_mask(
                        args,
                        score,
                        meta["bcd_score"],
                        presence,
                        tau_corr,
                        component_model,
                        device,
                        cls_name=cls_name,
                        item=item,
                        clipper=clipper,
                        class_text_embedding=class_text_embedding,
                        class_text_embeddings=class_text_embeddings,
                        class_text_index=cls_idx,
                        posterior_a=posterior.posterior_a,
                        posterior_b=posterior.posterior_b,
                        component_verifier_model=component_verifier_model,
                        suppressed_recovery_model=suppressed_recovery_model,
                    )
                    diag.update({f"local_{k}": v for k, v in local_diag.items() if isinstance(v, (int, float, bool))})
                    pred_by_method["clip_component_local_tau"][cls_name] = local_mask
                    threshold_logs["clip_component_local_tau"].append(local_threshold)
                    _append_threshold_ref_error(ref_errors, "clip_component_local_tau", dataset, cls_name, local_threshold)
                    _append_class_threshold_log(class_threshold_logs, class_ref_errors, "clip_component_local_tau", dataset, cls_name, local_threshold)
                if use_proto:
                    tau_proto, proto_diag = apply_class_prototype_threshold(
                        tau_corr,
                        args,
                        bank,
                        clipper,
                        cls_name,
                        spec["prompt_bank"],
                        diag,
                        component_features=component_features,
                    )
                    diag.update({k: v for k, v in proto_diag.items() if isinstance(v, (int, float, bool))})
                    proto_mask, proto_threshold = decode_fixed_tau(score, presence, tau_proto, args, budget_ratio=ratio_mem)
                    pred_by_method["clip_proto_tau"][cls_name] = proto_mask
                    threshold_logs["clip_proto_tau"].append(proto_threshold)
                    _append_threshold_ref_error(ref_errors, "clip_proto_tau", dataset, cls_name, proto_threshold)
                    _append_class_threshold_log(class_threshold_logs, class_ref_errors, "clip_proto_tau", dataset, cls_name, proto_threshold)
                    proto_soft = base.bcd_soft_gate_filter(proto_mask, score, meta["bcd_score"], proto_threshold, args)
                    pred_by_method["clip_proto_tau_softgate"][cls_name] = proto_soft
                    threshold_logs["clip_proto_tau_softgate"].append(proto_threshold)
                    _append_threshold_ref_error(ref_errors, "clip_proto_tau_softgate", dataset, cls_name, proto_threshold)
                    _append_class_threshold_log(class_threshold_logs, class_ref_errors, "clip_proto_tau_softgate", dataset, cls_name, proto_threshold)
                if use_meta:
                    tau_meta, meta_diag = predict_meta_threshold(
                        args,
                        meta_model,
                        bank,
                        stats,
                        tau_mem,
                        tau_corr,
                        base_threshold,
                        diag,
                        device,
                        cls_name=cls_name,
                        component_features=component_features,
                        semantic_features=semantic_features,
                        uncertainty_diag=uncertainty_diag,
                    )
                    diag.update({k: v for k, v in meta_diag.items() if isinstance(v, (int, float, bool))})
                    meta_mask, meta_threshold = decode_fixed_tau(score, presence, tau_meta, args, budget_ratio=ratio_mem)
                    pred_by_method["clip_meta_tau"][cls_name] = meta_mask
                    threshold_logs["clip_meta_tau"].append(meta_threshold)
                    _append_threshold_ref_error(ref_errors, "clip_meta_tau", dataset, cls_name, meta_threshold)
                    _append_class_threshold_log(class_threshold_logs, class_ref_errors, "clip_meta_tau", dataset, cls_name, meta_threshold)
                    meta_soft = base.bcd_soft_gate_filter(meta_mask, score, meta["bcd_score"], meta_threshold, args)
                    pred_by_method["clip_meta_tau_softgate"][cls_name] = meta_soft
                    threshold_logs["clip_meta_tau_softgate"].append(meta_threshold)
                    _append_threshold_ref_error(ref_errors, "clip_meta_tau_softgate", dataset, cls_name, meta_threshold)
                    _append_class_threshold_log(class_threshold_logs, class_ref_errors, "clip_meta_tau_softgate", dataset, cls_name, meta_threshold)
                if bool(args.enable_component_gate):
                    comp_mask = base.component_gate_filter(
                        corr_mask,
                        score,
                        meta["bcd_score"],
                        corr_threshold,
                        args,
                        component_model,
                        device,
                    )
                    pred_by_method["clip_corrected_tau_component_gate"][cls_name] = comp_mask
                    threshold_logs["clip_corrected_tau_component_gate"].append(corr_threshold)
                    _append_threshold_ref_error(ref_errors, "clip_corrected_tau_component_gate", dataset, cls_name, corr_threshold)
                    _append_class_threshold_log(class_threshold_logs, class_ref_errors, "clip_corrected_tau_component_gate", dataset, cls_name, corr_threshold)
                    if bool(getattr(args, "cmct_enable_multilabel_onepass", False)):
                        pred_by_method["clip_multilabel_onepass_component_gate"][cls_name] = comp_mask
                        threshold_logs["clip_multilabel_onepass_component_gate"].append(corr_threshold)
                        _append_threshold_ref_error(ref_errors, "clip_multilabel_onepass_component_gate", dataset, cls_name, corr_threshold)
                        _append_class_threshold_log(
                            class_threshold_logs,
                            class_ref_errors,
                            "clip_multilabel_onepass_component_gate",
                            dataset,
                            cls_name,
                            corr_threshold,
                        )
                    if use_seg2change_complement:
                        seg_support = seg2change_support_by_class.get(
                            cls_name,
                            np.zeros(size_hw, dtype=np.int64),
                        )
                        seg_comp_mask, seg_diag = seg2change_complement_component_gate(
                            args,
                            comp_mask,
                            score,
                            meta["bcd_score"],
                            corr_threshold,
                            seg_support,
                            component_model,
                            device,
                        )
                        pred_by_method["clip_seg2change_complement_component_gate"][cls_name] = seg_comp_mask
                        threshold_logs["clip_seg2change_complement_component_gate"].append(corr_threshold)
                        _append_threshold_ref_error(
                            ref_errors,
                            "clip_seg2change_complement_component_gate",
                            dataset,
                            cls_name,
                            corr_threshold,
                        )
                        _append_class_threshold_log(
                            class_threshold_logs,
                            class_ref_errors,
                            "clip_seg2change_complement_component_gate",
                            dataset,
                            cls_name,
                            corr_threshold,
                        )
                        seg2change_logs.append(seg_diag)
                    if use_prat_lite:
                        prat_comp_mask, prat_diag = prat_lite_component_gate(
                            args,
                            comp_mask,
                            score,
                            meta["bcd_score"],
                            corr_threshold,
                            posterior.posterior_a,
                            posterior.posterior_b,
                            cls_idx,
                            component_model,
                            device,
                        )
                        pred_by_method["clip_prat_lite_component_gate"][cls_name] = prat_comp_mask
                        threshold_logs["clip_prat_lite_component_gate"].append(corr_threshold)
                        _append_threshold_ref_error(
                            ref_errors,
                            "clip_prat_lite_component_gate",
                            dataset,
                            cls_name,
                            corr_threshold,
                        )
                        _append_class_threshold_log(
                            class_threshold_logs,
                            class_ref_errors,
                            "clip_prat_lite_component_gate",
                            dataset,
                            cls_name,
                            corr_threshold,
                        )
                        prat_logs.append(prat_diag)
                    if use_multiclass_competition:
                        multiclass_candidate_masks[cls_name] = comp_mask
                    if use_semantic_prior_branch:
                        pred_by_method["clip_semantic_prior_tau_component_gate"][cls_name] = comp_mask
                        threshold_logs["clip_semantic_prior_tau_component_gate"].append(corr_threshold)
                        _append_threshold_ref_error(ref_errors, "clip_semantic_prior_tau_component_gate", dataset, cls_name, corr_threshold)
                        _append_class_threshold_log(class_threshold_logs, class_ref_errors, "clip_semantic_prior_tau_component_gate", dataset, cls_name, corr_threshold)
                    if use_semantic_anchor:
                        anchor_comp_mask = base.component_gate_filter(
                            anchor_mask,
                            score,
                            meta["bcd_score"],
                            anchor_threshold,
                            args,
                            component_model,
                            device,
                        )
                        pred_by_method["clip_semantic_anchor_tau_component_gate"][cls_name] = anchor_comp_mask
                        threshold_logs["clip_semantic_anchor_tau_component_gate"].append(anchor_threshold)
                        _append_threshold_ref_error(ref_errors, "clip_semantic_anchor_tau_component_gate", dataset, cls_name, anchor_threshold)
                        _append_class_threshold_log(class_threshold_logs, class_ref_errors, "clip_semantic_anchor_tau_component_gate", dataset, cls_name, anchor_threshold)
                    if use_proto:
                        proto_comp_mask = base.component_gate_filter(
                            proto_mask,
                            score,
                            meta["bcd_score"],
                            proto_threshold,
                            args,
                            component_model,
                            device,
                        )
                        pred_by_method["clip_proto_tau_component_gate"][cls_name] = proto_comp_mask
                        threshold_logs["clip_proto_tau_component_gate"].append(proto_threshold)
                        _append_threshold_ref_error(ref_errors, "clip_proto_tau_component_gate", dataset, cls_name, proto_threshold)
                        _append_class_threshold_log(class_threshold_logs, class_ref_errors, "clip_proto_tau_component_gate", dataset, cls_name, proto_threshold)
                    if use_meta:
                        meta_comp_mask = base.component_gate_filter(
                            meta_mask,
                            score,
                            meta["bcd_score"],
                            meta_threshold,
                            args,
                            component_model,
                            device,
                        )
                        pred_by_method["clip_meta_tau_component_gate"][cls_name] = meta_comp_mask
                        threshold_logs["clip_meta_tau_component_gate"].append(meta_threshold)
                        _append_threshold_ref_error(ref_errors, "clip_meta_tau_component_gate", dataset, cls_name, meta_threshold)
                        _append_class_threshold_log(class_threshold_logs, class_ref_errors, "clip_meta_tau_component_gate", dataset, cls_name, meta_threshold)
                    if use_adapter:
                        adapter_comp_mask = base.component_gate_filter(
                            adapter_mask,
                            score,
                            meta["bcd_score"],
                            adapter_threshold,
                            args,
                            component_model,
                            device,
                        )
                        pred_by_method["clip_class_adapter_tau_component_gate"][cls_name] = adapter_comp_mask
                        threshold_logs["clip_class_adapter_tau_component_gate"].append(adapter_threshold)
                        _append_threshold_ref_error(ref_errors, "clip_class_adapter_tau_component_gate", dataset, cls_name, adapter_threshold)
                        _append_class_threshold_log(class_threshold_logs, class_ref_errors, "clip_class_adapter_tau_component_gate", dataset, cls_name, adapter_threshold)
                    if use_score_cal:
                        score_cal_comp_mask = base.component_gate_filter(
                            score_cal_mask,
                            calibrated_score,
                            meta["bcd_score"],
                            score_cal_threshold,
                            args,
                            component_model,
                            device,
                        )
                        pred_by_method["clip_score_calibrated_tau_component_gate"][cls_name] = score_cal_comp_mask
                        threshold_logs["clip_score_calibrated_tau_component_gate"].append(score_cal_threshold)
                        _append_threshold_ref_error(ref_errors, "clip_score_calibrated_tau_component_gate", dataset, cls_name, score_cal_threshold)
                        _append_class_threshold_log(class_threshold_logs, class_ref_errors, "clip_score_calibrated_tau_component_gate", dataset, cls_name, score_cal_threshold)

            if use_multiclass_competition:
                mc_masks, mc_diag = multiclass_component_competition_gate(
                    args,
                    class_names,
                    multiclass_score_maps,
                    multiclass_thresholds,
                    multiclass_candidate_masks,
                    meta["bcd_score"],
                    endpoint_maps_a=multiclass_endpoint_maps_a,
                    endpoint_maps_b=multiclass_endpoint_maps_b,
                )
                pred_by_method["clip_multiclass_competition_gate"] = mc_masks
                multiclass_logs.append(mc_diag)
                for cls_name in class_names:
                    threshold = float(multiclass_thresholds.get(cls_name, 0.0))
                    threshold_logs["clip_multiclass_competition_gate"].append(threshold)
                    _append_threshold_ref_error(ref_errors, "clip_multiclass_competition_gate", dataset, cls_name, threshold)
                    _append_class_threshold_log(
                        class_threshold_logs,
                        class_ref_errors,
                        "clip_multiclass_competition_gate",
                        dataset,
                        cls_name,
                        threshold,
                    )

            export_semantic_prediction(args, dataset, item, class_names, pred_by_method)
            for method, pred_masks in pred_by_method.items():
                base.add_prediction(accumulators[method], class_names, gt_masks, pred_masks, bcd_cm)
        dataset_out = OrderedDict()
        threshold_summary = _summarize_threshold_logs(threshold_logs, ref_errors)
        class_threshold_summary = _summarize_class_threshold_logs(class_threshold_logs, class_ref_errors)
        for method, acc in accumulators.items():
            result = base.finalize_accumulator(acc)
            result["threshold_summary"] = threshold_summary[method]
            result["class_threshold_summary"] = class_threshold_summary[method]
            dataset_out[method] = result
        if retrieval_logs:
            dataset_out["clip_memory_retrieval_summary"] = {
                "top1_sim_mean": float(np.mean([x.get("top1_sim", 0.0) for x in retrieval_logs])),
                "mean_sim_mean": float(np.mean([x.get("mean_sim", 0.0) for x in retrieval_logs])),
                "delta_tau_mean": float(np.mean([x.get("delta_tau", 0.0) for x in retrieval_logs])),
            }
        if multiclass_logs:
            numeric_keys = sorted({
                key
                for entry in multiclass_logs
                for key, value in entry.items()
                if isinstance(value, (int, float, bool))
            })
            dataset_out["clip_multiclass_competition_summary"] = {
                key: float(np.mean([float(entry.get(key, 0.0)) for entry in multiclass_logs]))
                for key in numeric_keys
            }
        if seg2change_logs:
            numeric_keys = sorted({
                key
                for entry in seg2change_logs
                for key, value in entry.items()
                if isinstance(value, (int, float, bool))
            })
            dataset_out["clip_seg2change_complement_summary"] = {
                key: float(np.mean([float(entry[key]) for entry in seg2change_logs if key in entry]))
                for key in numeric_keys
            }
        if prat_logs:
            numeric_keys = sorted({
                key
                for entry in prat_logs
                for key, value in entry.items()
                if isinstance(value, (int, float, bool))
            })
            dataset_out["clip_prat_lite_summary"] = {
                key: float(np.mean([float(entry[key]) for entry in prat_logs if key in entry]))
                for key in numeric_keys
            }
        out[dataset] = dataset_out
    return out


@torch.no_grad()
def evaluate_binary_targets(
    args,
    datasets,
    items_by_dataset,
    processor,
    change_module,
    device,
    clipper,
    bank,
    correction_model,
    component_model=None,
    component_verifier_model=None,
    suppressed_recovery_model=None,
    meta_model=None,
    class_residual_adapter=None,
    score_calibration_selection=None,
):
    cfg = base._memory_config_from_args(args)
    if correction_model is not None:
        correction_model.eval()
    if meta_model is not None:
        meta_model.eval()
    if component_verifier_model is not None:
        component_verifier_model.eval()
    if suppressed_recovery_model is not None:
        suppressed_recovery_model.eval()
    out = OrderedDict()
    method_names = [
        "adaptive_no_cap",
        "bcd_soft_gate",
        "clip_knn_tau",
        "clip_corrected_tau",
        "clip_corrected_tau_softgate",
    ]
    use_local = bool(getattr(args, "cmct_enable_component_local_refine", False))
    if use_local:
        method_names.append("clip_component_local_tau")
    use_proto = bool(getattr(args, "cmct_enable_class_prototype_threshold", False))
    if use_proto:
        method_names += [
            "clip_proto_tau",
            "clip_proto_tau_softgate",
        ]
    use_meta = bool(getattr(args, "cmct_enable_meta_threshold", False)) and meta_model is not None
    if use_meta:
        method_names += [
            "clip_meta_tau",
            "clip_meta_tau_softgate",
        ]
    use_adapter = bool(getattr(args, "cmct_enable_class_residual_adapter", False)) and class_residual_adapter is not None
    if use_adapter:
        method_names += [
            "clip_class_adapter_tau",
            "clip_class_adapter_tau_softgate",
        ]
    use_score_cal = bool(getattr(args, "cmct_enable_score_calibration", False))
    if use_score_cal:
        method_names += [
            "clip_score_calibrated_tau",
            "clip_score_calibrated_tau_softgate",
        ]
    use_semantic_prior_branch = bool(getattr(args, "cmct_enable_semantic_relation", False))
    if use_semantic_prior_branch:
        method_names += [
            "clip_semantic_prior_tau",
            "clip_semantic_prior_tau_softgate",
        ]
    if bool(args.enable_component_gate):
        method_names.append("clip_corrected_tau_component_gate")
        if use_semantic_prior_branch:
            method_names.append("clip_semantic_prior_tau_component_gate")
        if use_proto:
            method_names.append("clip_proto_tau_component_gate")
        if use_meta:
            method_names.append("clip_meta_tau_component_gate")
        if use_adapter:
            method_names.append("clip_class_adapter_tau_component_gate")
        if use_score_cal:
            method_names.append("clip_score_calibrated_tau_component_gate")
    for dataset in datasets:
        binary_spec = bench.BINARY_DATASETS[dataset]
        posterior_specs = binary_spec.get("posterior_specs")
        if posterior_specs is None:
            continue
        class_names = list(posterior_specs.keys())
        target_names = set(binary_spec["posterior_targets"])
        prompt_bank = bench.specs_to_prompt_bank(posterior_specs)
        confusers = bench.specs_to_confusers(posterior_specs)
        accumulators = OrderedDict((name, {
            "samples": 0,
            "task_confusion": np.zeros((2, 2), dtype=np.int64),
            "bcd_confusion": np.zeros((2, 2), dtype=np.int64),
        }) for name in method_names)
        threshold_logs = OrderedDict((name, []) for name in method_names)
        ref_errors = OrderedDict((name, []) for name in method_names)
        for item in tqdm(items_by_dataset[dataset], desc=f"eval-CMCT-{dataset}", dynamic_ncols=True):
            meta, _, bcd_np, size_hw = bench.predict_change(processor, change_module, item, args, device)
            gt = bench.read_binary_mask(item["label"], size_hw)
            bcd_cm = bench.confusion_binary(gt, bcd_np)
            pair_embedding = clipper.encode_pair(item)
            scores_a, presence_a = extract_class_scores(processor, meta["state_a"], prompt_bank, class_names)
            scores_b, presence_b = extract_class_scores(processor, meta["state_b"], prompt_bank, class_names)
            class_text_embeddings = np.stack([
                _clipper_text_embedding(clipper, prompt_text_for_class(name, prompt_bank))
                for name in class_names
            ], axis=0).astype(np.float32)
            posterior = build_posterior_inputs(
                scores_a,
                scores_b,
                presence_a,
                presence_b,
                class_names=class_names,
                confusers=confusers,
                rho=args.rho,
            )
            pred_by_method = OrderedDict((name, np.zeros(size_hw, dtype=np.int64)) for name in method_names)
            for cls_idx, cls_name in enumerate(class_names):
                if cls_name not in target_names:
                    continue
                score = memory_rectified_score(
                    posterior.posterior_a[cls_idx],
                    posterior.posterior_b[cls_idx],
                    None if args.disable_bcd_gate else meta["bcd_score"],
                    presence_a=presence_a[cls_idx],
                    presence_b=presence_b[cls_idx],
                    config=cfg,
                )
                presence = torch.maximum(presence_a[cls_idx], presence_b[cls_idx])
                base_mask, base_threshold = base.decode_class_score(score, presence, 0.0, args)
                pred_by_method["adaptive_no_cap"] |= base_mask
                threshold_logs["adaptive_no_cap"].append(base_threshold)
                _append_threshold_ref_error(ref_errors, "adaptive_no_cap", dataset, cls_name, base_threshold)
                soft_mask = base.bcd_soft_gate_filter(base_mask, score, meta["bcd_score"], base_threshold, args)
                pred_by_method["bcd_soft_gate"] |= soft_mask
                threshold_logs["bcd_soft_gate"].append(base_threshold)
                _append_threshold_ref_error(ref_errors, "bcd_soft_gate", dataset, cls_name, base_threshold)

                component_features = component_reliability_features(
                    args,
                    base_mask,
                    score,
                    meta["bcd_score"],
                    base_threshold,
                    component_model,
                    device,
                )
                semantic_features = semantic_relation_features_for_prompt(args, bank, clipper, cls_name, prompt_bank)
                key, stats = make_query(args, item, cls_name, prompt_bank, pair_embedding, clipper, score, meta["bcd_score"], presence)
                tau_mem, tau_corr, diag = predict_cmct_thresholds(
                    args,
                    bank,
                    correction_model,
                    key,
                    stats,
                    device,
                    cls_name=cls_name,
                    component_features=component_features,
                    semantic_features=semantic_features,
                )
                tau_mem = apply_adaptive_floor(tau_mem, base_threshold, args)
                tau_corr = apply_adaptive_floor(tau_corr, base_threshold, args)
                tau_mem = apply_class_prior_floor(tau_mem, args, bank, cls_name)
                tau_corr = apply_class_prior_floor(tau_corr, args, bank, cls_name)
                tau_corr, semantic_prior_diag = apply_semantic_prior_floor(tau_corr, args, semantic_features)
                diag.update({f"semantic_prior_{k}": v for k, v in semantic_prior_diag.items() if isinstance(v, (int, float, bool))})
                tau_corr, uncertainty_diag = apply_uncertainty_calibration(
                    tau_corr,
                    base_threshold,
                    args,
                    stats,
                    diag,
                    component_features=component_features,
                    semantic_features=semantic_features,
                )
                diag.update({f"uncertainty_{k}": v for k, v in uncertainty_diag.items() if isinstance(v, (int, float, bool))})
                ratio_mem = diag.get("ratio_mem") if bool(getattr(args, "cmct_enable_memory_ratio_budget", False)) else None
                knn_mask, knn_threshold = decode_fixed_tau(score, presence, tau_mem, args, budget_ratio=ratio_mem)
                pred_by_method["clip_knn_tau"] |= knn_mask
                threshold_logs["clip_knn_tau"].append(knn_threshold)
                _append_threshold_ref_error(ref_errors, "clip_knn_tau", dataset, cls_name, knn_threshold)
                corr_mask, corr_threshold = decode_fixed_tau(score, presence, tau_corr, args, budget_ratio=ratio_mem)
                pred_by_method["clip_corrected_tau"] |= corr_mask
                threshold_logs["clip_corrected_tau"].append(corr_threshold)
                _append_threshold_ref_error(ref_errors, "clip_corrected_tau", dataset, cls_name, corr_threshold)
                corr_soft = base.bcd_soft_gate_filter(corr_mask, score, meta["bcd_score"], corr_threshold, args)
                pred_by_method["clip_corrected_tau_softgate"] |= corr_soft
                threshold_logs["clip_corrected_tau_softgate"].append(corr_threshold)
                _append_threshold_ref_error(ref_errors, "clip_corrected_tau_softgate", dataset, cls_name, corr_threshold)
                if use_semantic_prior_branch:
                    pred_by_method["clip_semantic_prior_tau"] |= corr_mask
                    threshold_logs["clip_semantic_prior_tau"].append(corr_threshold)
                    _append_threshold_ref_error(ref_errors, "clip_semantic_prior_tau", dataset, cls_name, corr_threshold)
                    pred_by_method["clip_semantic_prior_tau_softgate"] |= corr_soft
                    threshold_logs["clip_semantic_prior_tau_softgate"].append(corr_threshold)
                    _append_threshold_ref_error(ref_errors, "clip_semantic_prior_tau_softgate", dataset, cls_name, corr_threshold)
                if use_score_cal:
                    calibrated_score, score_cal_diag = apply_score_calibration(
                        score,
                        meta["bcd_score"],
                        args,
                        bank,
                        clipper,
                        cls_name,
                        prompt_bank,
                        stats,
                        semantic_features=semantic_features,
                        component_features=component_features,
                    )
                    diag.update({f"score_cal_{k}": v for k, v in score_cal_diag.items() if isinstance(v, (int, float, bool))})
                    score_cal_mask, score_cal_threshold = decode_fixed_tau(calibrated_score, presence, tau_corr, args, budget_ratio=ratio_mem)
                    pred_by_method["clip_score_calibrated_tau"] |= score_cal_mask
                    threshold_logs["clip_score_calibrated_tau"].append(score_cal_threshold)
                    _append_threshold_ref_error(ref_errors, "clip_score_calibrated_tau", dataset, cls_name, score_cal_threshold)
                    score_cal_soft = base.bcd_soft_gate_filter(score_cal_mask, calibrated_score, meta["bcd_score"], score_cal_threshold, args)
                    pred_by_method["clip_score_calibrated_tau_softgate"] |= score_cal_soft
                    threshold_logs["clip_score_calibrated_tau_softgate"].append(score_cal_threshold)
                    _append_threshold_ref_error(ref_errors, "clip_score_calibrated_tau_softgate", dataset, cls_name, score_cal_threshold)
                if use_adapter:
                    tau_adapter, adapter_diag = apply_class_residual_adapter(
                        tau_corr,
                        args,
                        class_residual_adapter,
                        bank,
                        clipper,
                        cls_name,
                        prompt_bank,
                        semantic_features,
                        component_features,
                    )
                    diag.update({f"class_adapter_{k}": v for k, v in adapter_diag.items() if isinstance(v, (int, float, bool))})
                    adapter_mask, adapter_threshold = decode_fixed_tau(score, presence, tau_adapter, args, budget_ratio=ratio_mem)
                    pred_by_method["clip_class_adapter_tau"] |= adapter_mask
                    threshold_logs["clip_class_adapter_tau"].append(adapter_threshold)
                    _append_threshold_ref_error(ref_errors, "clip_class_adapter_tau", dataset, cls_name, adapter_threshold)
                    adapter_soft = base.bcd_soft_gate_filter(adapter_mask, score, meta["bcd_score"], adapter_threshold, args)
                    pred_by_method["clip_class_adapter_tau_softgate"] |= adapter_soft
                    threshold_logs["clip_class_adapter_tau_softgate"].append(adapter_threshold)
                    _append_threshold_ref_error(ref_errors, "clip_class_adapter_tau_softgate", dataset, cls_name, adapter_threshold)
                if use_local:
                    class_text_embedding = class_text_embeddings[cls_idx]
                    local_mask, local_threshold, local_diag = component_local_refine_mask(
                        args,
                        score,
                        meta["bcd_score"],
                        presence,
                        tau_corr,
                        component_model,
                        device,
                        cls_name=cls_name,
                        item=item,
                        clipper=clipper,
                        class_text_embedding=class_text_embedding,
                        class_text_embeddings=class_text_embeddings,
                        class_text_index=cls_idx,
                        posterior_a=posterior.posterior_a,
                        posterior_b=posterior.posterior_b,
                        component_verifier_model=component_verifier_model,
                        suppressed_recovery_model=suppressed_recovery_model,
                    )
                    diag.update({f"local_{k}": v for k, v in local_diag.items() if isinstance(v, (int, float, bool))})
                    pred_by_method["clip_component_local_tau"] |= local_mask
                    threshold_logs["clip_component_local_tau"].append(local_threshold)
                    _append_threshold_ref_error(ref_errors, "clip_component_local_tau", dataset, cls_name, local_threshold)
                if use_proto:
                    tau_proto, proto_diag = apply_class_prototype_threshold(
                        tau_corr,
                        args,
                        bank,
                        clipper,
                        cls_name,
                        prompt_bank,
                        diag,
                        component_features=component_features,
                    )
                    diag.update({k: v for k, v in proto_diag.items() if isinstance(v, (int, float, bool))})
                    proto_mask, proto_threshold = decode_fixed_tau(score, presence, tau_proto, args, budget_ratio=ratio_mem)
                    pred_by_method["clip_proto_tau"] |= proto_mask
                    threshold_logs["clip_proto_tau"].append(proto_threshold)
                    _append_threshold_ref_error(ref_errors, "clip_proto_tau", dataset, cls_name, proto_threshold)
                    proto_soft = base.bcd_soft_gate_filter(proto_mask, score, meta["bcd_score"], proto_threshold, args)
                    pred_by_method["clip_proto_tau_softgate"] |= proto_soft
                    threshold_logs["clip_proto_tau_softgate"].append(proto_threshold)
                    _append_threshold_ref_error(ref_errors, "clip_proto_tau_softgate", dataset, cls_name, proto_threshold)
                if use_meta:
                    tau_meta, meta_diag = predict_meta_threshold(
                        args,
                        meta_model,
                        bank,
                        stats,
                        tau_mem,
                        tau_corr,
                        base_threshold,
                        diag,
                        device,
                        cls_name=cls_name,
                        component_features=component_features,
                        semantic_features=semantic_features,
                        uncertainty_diag=uncertainty_diag,
                    )
                    diag.update({k: v for k, v in meta_diag.items() if isinstance(v, (int, float, bool))})
                    meta_mask, meta_threshold = decode_fixed_tau(score, presence, tau_meta, args, budget_ratio=ratio_mem)
                    pred_by_method["clip_meta_tau"] |= meta_mask
                    threshold_logs["clip_meta_tau"].append(meta_threshold)
                    _append_threshold_ref_error(ref_errors, "clip_meta_tau", dataset, cls_name, meta_threshold)
                    meta_soft = base.bcd_soft_gate_filter(meta_mask, score, meta["bcd_score"], meta_threshold, args)
                    pred_by_method["clip_meta_tau_softgate"] |= meta_soft
                    threshold_logs["clip_meta_tau_softgate"].append(meta_threshold)
                    _append_threshold_ref_error(ref_errors, "clip_meta_tau_softgate", dataset, cls_name, meta_threshold)
                if bool(args.enable_component_gate):
                    comp_mask = base.component_gate_filter(
                        corr_mask,
                        score,
                        meta["bcd_score"],
                        corr_threshold,
                        args,
                        component_model,
                        device,
                    )
                    pred_by_method["clip_corrected_tau_component_gate"] |= comp_mask
                    threshold_logs["clip_corrected_tau_component_gate"].append(corr_threshold)
                    _append_threshold_ref_error(ref_errors, "clip_corrected_tau_component_gate", dataset, cls_name, corr_threshold)
                    if use_semantic_prior_branch:
                        pred_by_method["clip_semantic_prior_tau_component_gate"] |= comp_mask
                        threshold_logs["clip_semantic_prior_tau_component_gate"].append(corr_threshold)
                        _append_threshold_ref_error(ref_errors, "clip_semantic_prior_tau_component_gate", dataset, cls_name, corr_threshold)
                    if use_proto:
                        proto_comp_mask = base.component_gate_filter(
                            proto_mask,
                            score,
                            meta["bcd_score"],
                            proto_threshold,
                            args,
                            component_model,
                            device,
                        )
                        pred_by_method["clip_proto_tau_component_gate"] |= proto_comp_mask
                        threshold_logs["clip_proto_tau_component_gate"].append(proto_threshold)
                        _append_threshold_ref_error(ref_errors, "clip_proto_tau_component_gate", dataset, cls_name, proto_threshold)
                    if use_meta:
                        meta_comp_mask = base.component_gate_filter(
                            meta_mask,
                            score,
                            meta["bcd_score"],
                            meta_threshold,
                            args,
                            component_model,
                            device,
                        )
                        pred_by_method["clip_meta_tau_component_gate"] |= meta_comp_mask
                        threshold_logs["clip_meta_tau_component_gate"].append(meta_threshold)
                        _append_threshold_ref_error(ref_errors, "clip_meta_tau_component_gate", dataset, cls_name, meta_threshold)
                    if use_adapter:
                        adapter_comp_mask = base.component_gate_filter(
                            adapter_mask,
                            score,
                            meta["bcd_score"],
                            adapter_threshold,
                            args,
                            component_model,
                            device,
                        )
                        pred_by_method["clip_class_adapter_tau_component_gate"] |= adapter_comp_mask
                        threshold_logs["clip_class_adapter_tau_component_gate"].append(adapter_threshold)
                        _append_threshold_ref_error(ref_errors, "clip_class_adapter_tau_component_gate", dataset, cls_name, adapter_threshold)
                    if use_score_cal:
                        score_cal_comp_mask = base.component_gate_filter(
                            score_cal_mask,
                            calibrated_score,
                            meta["bcd_score"],
                            score_cal_threshold,
                            args,
                            component_model,
                            device,
                        )
                        pred_by_method["clip_score_calibrated_tau_component_gate"] |= score_cal_comp_mask
                        threshold_logs["clip_score_calibrated_tau_component_gate"].append(score_cal_threshold)
                        _append_threshold_ref_error(ref_errors, "clip_score_calibrated_tau_component_gate", dataset, cls_name, score_cal_threshold)

            export_binary_prediction(args, dataset, item, pred_by_method)
            for method, pred in pred_by_method.items():
                accumulators[method]["samples"] += 1
                accumulators[method]["bcd_confusion"] += bcd_cm
                accumulators[method]["task_confusion"] += bench.confusion_binary(gt, pred.astype(np.int64))
        dataset_out = OrderedDict()
        threshold_summary = _summarize_threshold_logs(threshold_logs, ref_errors)
        for method, acc in accumulators.items():
            dataset_out[method] = {
                "samples": int(acc["samples"]),
                "task_confusion": acc["task_confusion"].tolist(),
                "bcd_confusion": acc["bcd_confusion"].tolist(),
                "task": {k: float(v) for k, v in bench.metrics_from_confusion(acc["task_confusion"]).items()},
                "bcd": {k: float(v) for k, v in bench.metrics_from_confusion(acc["bcd_confusion"]).items()},
                "threshold_summary": threshold_summary[method],
            }
        out[dataset] = dataset_out
    return out


def load_or_train_cmct(args, train_items, processor, change_module, device, clipper, component_model=None):
    if args.load_checkpoint:
        ckpt = torch.load(args.load_checkpoint, map_location=device, weights_only=False)
        memory_taus = np.asarray(ckpt["memory_taus"], dtype=np.float32)
        memory_component_features = np.asarray(
            ckpt.get("memory_component_features", np.zeros((memory_taus.shape[0], 0), dtype=np.float32)),
            dtype=np.float32,
        )
        source_class_names = list(ckpt.get("source_class_names", []))
        source_class_text_embeddings = np.asarray(
            ckpt.get("source_class_text_embeddings", np.zeros((0, 0), dtype=np.float32)),
            dtype=np.float32,
        )
        if bool(getattr(args, "cmct_enable_component_reliability", False)) and (
            memory_component_features.ndim != 2
            or memory_component_features.shape[1] != len(CMCT_COMPONENT_RELIABILITY_NAMES)
        ):
            raise ValueError(
                "Loaded CMCT checkpoint does not contain component reliability features. "
                "Rebuild the CMCT checkpoint without --load_checkpoint when enabling "
                "--cmct_enable_component_reliability."
            )
        if bool(getattr(args, "cmct_enable_semantic_relation", False)) and (
            not source_class_names
            or source_class_text_embeddings.ndim != 2
            or source_class_text_embeddings.shape[0] != len(source_class_names)
        ):
            raise ValueError(
                "Loaded CMCT checkpoint does not contain source class text embeddings. "
                "Rebuild the CMCT checkpoint without --load_checkpoint when enabling "
                "--cmct_enable_semantic_relation."
            )
        bank = {
            "memory_keys": np.asarray(ckpt["memory_keys"], dtype=np.float32),
            "memory_taus": memory_taus,
            "memory_ratios": np.asarray(ckpt.get("memory_ratios", np.zeros_like(ckpt["memory_taus"])), dtype=np.float32),
            "memory_stats": np.asarray(ckpt["memory_stats"], dtype=np.float32),
            "memory_oracle_metrics": np.asarray(
                ckpt.get("memory_oracle_metrics", np.zeros((memory_taus.shape[0], len(CMCT_ORACLE_METRIC_NAMES)), dtype=np.float32)),
                dtype=np.float32,
            ),
            "memory_component_features": memory_component_features,
            "source_class_names": source_class_names,
            "source_class_text_embeddings": source_class_text_embeddings,
            "memory_meta": list(ckpt.get("memory_meta", [])),
        }
        input_dim = cmct_correction_feature_dim(args)
        first_weight = ckpt["correction_head"].get("net.0.weight")
        checkpoint_input_dim = int(first_weight.shape[1]) if first_weight is not None else 0
        should_retrain = checkpoint_input_dim != int(input_dim)
        if should_retrain:
            model, corrector_stats = train_correction_head(args, bank, device)
            corrector_stats["retrained_from_checkpoint"] = str(args.load_checkpoint)
            corrector_stats["checkpoint_input_dim"] = int(checkpoint_input_dim)
            corrector_stats["requested_input_dim"] = int(input_dim)
            memory_stats = dict(ckpt.get("clip_memory_stats", {}))
            if args.output_checkpoint:
                torch.save({
                    "memory_keys": bank["memory_keys"],
                    "memory_taus": bank["memory_taus"],
                    "memory_ratios": bank["memory_ratios"],
                    "memory_stats": bank["memory_stats"],
                    "memory_oracle_metrics": bank["memory_oracle_metrics"],
                    "memory_component_features": bank["memory_component_features"],
                    "source_class_names": bank["source_class_names"],
                    "source_class_text_embeddings": bank["source_class_text_embeddings"],
                    "memory_meta": bank["memory_meta"],
                    "correction_head": model.state_dict(),
                    "config": vars(args),
                    "clip_memory_stats": memory_stats,
                    "clip_corrector_stats": corrector_stats,
                    "stat_names": CMCT_STAT_NAMES,
                    "class_condition_names": CMCT_CLASS_CONDITION_NAMES if bool(args.cmct_enable_class_residual) else [],
                    "component_reliability_names": CMCT_COMPONENT_RELIABILITY_NAMES if bool(args.cmct_enable_component_reliability) else [],
                    "semantic_relation_names": CMCT_SEMANTIC_RELATION_NAMES if bool(args.cmct_enable_semantic_relation) else [],
                    "oracle_metric_names": CMCT_ORACLE_METRIC_NAMES,
                }, args.output_checkpoint)
                print(f"Saved CMCT checkpoint: {args.output_checkpoint}")
            return bank, model, memory_stats, corrector_stats
        model = CorrectionHead(
            input_dim,
            hidden_dim=int(ckpt["config"].get("clip_corrector_hidden_dim", args.clip_corrector_hidden_dim)),
            dropout=float(ckpt["config"].get("clip_corrector_dropout", args.clip_corrector_dropout)),
            max_delta=float(ckpt["config"].get("clip_corrector_max_delta", args.clip_corrector_max_delta)),
        ).to(device)
        model.load_state_dict(ckpt["correction_head"])
        model.eval()
        corrector_stats = dict(ckpt.get("clip_corrector_stats", {}))
        corrector_stats["loaded_from_checkpoint"] = True
        corrector_stats["input_dim"] = int(input_dim)
        return bank, model, dict(ckpt.get("clip_memory_stats", {})), corrector_stats

    rows, memory_stats = build_memory_rows(args, train_items, processor, change_module, device, clipper, component_model)
    bank = rows_to_arrays(rows)
    model, corrector_stats = train_correction_head(args, bank, device)
    if args.output_checkpoint:
        torch.save({
            "memory_keys": bank["memory_keys"],
            "memory_taus": bank["memory_taus"],
            "memory_ratios": bank["memory_ratios"],
            "memory_stats": bank["memory_stats"],
            "memory_oracle_metrics": bank["memory_oracle_metrics"],
            "memory_component_features": bank["memory_component_features"],
            "source_class_names": bank["source_class_names"],
            "source_class_text_embeddings": bank["source_class_text_embeddings"],
            "memory_meta": bank["memory_meta"],
            "correction_head": model.state_dict(),
            "config": vars(args),
            "clip_memory_stats": memory_stats,
            "clip_corrector_stats": corrector_stats,
            "stat_names": CMCT_STAT_NAMES,
            "class_condition_names": CMCT_CLASS_CONDITION_NAMES if bool(args.cmct_enable_class_residual) else [],
            "component_reliability_names": CMCT_COMPONENT_RELIABILITY_NAMES if bool(args.cmct_enable_component_reliability) else [],
            "semantic_relation_names": CMCT_SEMANTIC_RELATION_NAMES if bool(args.cmct_enable_semantic_relation) else [],
            "oracle_metric_names": CMCT_ORACLE_METRIC_NAMES,
        }, args.output_checkpoint)
        print(f"Saved CMCT checkpoint: {args.output_checkpoint}")
    return bank, model, memory_stats, corrector_stats


@torch.no_grad()
def collect_cacdd_component_verifier_examples(args, items: Sequence[dict], processor, change_module, device, clipper):
    feats = []
    labels_out = []
    stats = {
        "candidate_source": str(args.component_gate_candidate_source),
        "candidate_components": 0,
        "positive_components": 0,
        "negative_components": 0,
    }
    if str(args.component_gate_candidate_source) != "semantic_proxy":
        return torch.empty((0, len(COMPONENT_VERIFIER_FEATURE_NAMES))), torch.empty((0,)), {
            **stats,
            "reason": "component_verifier_requires_semantic_proxy",
        }

    source_class_names, source_prompt_bank, source_confusers = base.source_component_prompt_spec(args)
    class_text_embeddings = np.stack([
        _clipper_text_embedding(clipper, prompt_text_for_class(name, source_prompt_bank))
        for name in source_class_names
    ], axis=0).astype(np.float32)
    stats["source_prompt_dataset"] = str(args.component_gate_source_prompt_dataset)
    stats["source_classes"] = source_class_names
    stats["class_component_counts"] = {name: 0 for name in source_class_names}
    cfg = base._memory_config_from_args(args)

    for item in tqdm(items, desc="collect-CMCT-component-verifier", dynamic_ncols=True):
        meta, _, _, size_hw = bench.predict_change(processor, change_module, item, args, device)
        gt = bench.read_binary_mask(item["label"], size_hw)
        bcd = meta["bcd_score"].float().clamp(0, 1)
        bcd_th = base.component_gate_threshold(bcd, args)
        bcd_th_value = float(bcd_th.detach().cpu())
        bcd_np = bcd.detach().cpu().numpy()
        scores_a, presence_a = extract_class_scores(
            processor, meta["state_a"], source_prompt_bank, source_class_names,
        )
        scores_b, presence_b = extract_class_scores(
            processor, meta["state_b"], source_prompt_bank, source_class_names,
        )
        posterior = build_posterior_inputs(
            scores_a,
            scores_b,
            presence_a,
            presence_b,
            class_names=source_class_names,
            confusers=source_confusers,
            rho=args.rho,
        )
        for cls_idx, cls_name in enumerate(source_class_names):
            score = memory_rectified_score(
                posterior.posterior_a[cls_idx],
                posterior.posterior_b[cls_idx],
                None if args.disable_bcd_gate else meta["bcd_score"],
                presence_a=presence_a[cls_idx],
                presence_b=presence_b[cls_idx],
                config=cfg,
            )
            presence = torch.maximum(presence_a[cls_idx], presence_b[cls_idx])
            candidate_np, score_threshold = base.decode_class_score(score, presence, 0.0, args)
            candidate_threshold = float(score_threshold)
            relax_scale = float(getattr(args, "cmct_component_verifier_candidate_relax_scale", 1.0))
            if relax_scale > 0.0 and abs(relax_scale - 1.0) > 1e-6:
                candidate_np, _ = decode_fixed_tau(
                    score,
                    presence,
                    float(score_threshold) * relax_scale,
                    args,
                )
            labels_cc, num = base.connected_components(candidate_np.astype(bool))
            score_np = score.detach().float().cpu().numpy()
            feature_score_np, feature_score_threshold = base.component_gate_feature_inputs(
                score_np,
                bcd_np,
                candidate_threshold,
                bcd_th_value,
                args,
            )
            class_text_embedding = class_text_embeddings[cls_idx]
            for lab in range(1, num + 1):
                region = labels_cc == lab
                area = int(region.sum())
                if area < int(args.component_gate_min_area):
                    continue
                if int(args.component_gate_max_area) > 0 and area > int(args.component_gate_max_area):
                    continue
                base_row = base.component_gate_features(
                    region,
                    feature_score_np,
                    bcd_np,
                    feature_score_threshold,
                    bcd_th_value,
                    topk_ratio=float(args.component_gate_topk_ratio),
                )
                row = component_verifier_feature_row(
                    args,
                    base_row,
                    region,
                    item,
                    clipper,
                    class_text_embedding,
                    class_text_embeddings,
                    class_text_index=cls_idx,
                )
                label = base.label_component_gate(region, gt, args)
                feats.append(row)
                labels_out.append(label)
                stats["candidate_components"] += 1
                stats["class_component_counts"][cls_name] += 1
                if label > 0.5:
                    stats["positive_components"] += 1
                else:
                    stats["negative_components"] += 1

    if not feats:
        return torch.empty((0, len(COMPONENT_VERIFIER_FEATURE_NAMES))), torch.empty((0,)), stats
    stats["positive_ratio"] = float(np.mean(labels_out))
    return torch.tensor(np.stack(feats, axis=0), dtype=torch.float32), torch.tensor(labels_out, dtype=torch.float32), stats


def train_component_verifier(args, x: torch.Tensor, y: torch.Tensor, device):
    model = base.ComponentGateHead(
        len(COMPONENT_VERIFIER_FEATURE_NAMES),
        hidden_dim=int(args.cmct_component_verifier_hidden_dim),
        dropout=float(args.cmct_component_verifier_dropout),
    ).to(device)
    if x.numel() == 0:
        return model, {"num_components": 0, "positive_ratio": 0.0}
    x = x.to(device)
    y = y.to(device)
    pos = y.sum().clamp_min(1.0)
    neg = (y.numel() - y.sum()).clamp_min(1.0)
    pos_weight = (neg / pos * float(args.cmct_component_verifier_positive_weight_scale)).clamp(
        0.5,
        float(args.cmct_component_verifier_max_pos_weight),
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(args.cmct_component_verifier_lr),
        weight_decay=float(args.cmct_component_verifier_weight_decay),
    )
    for _ in range(int(args.cmct_component_verifier_epochs)):
        model.train()
        perm = torch.randperm(x.shape[0], device=device)
        for start in range(0, x.shape[0], int(args.cmct_component_verifier_batch_size)):
            idx = perm[start:start + int(args.cmct_component_verifier_batch_size)]
            logits = model(x[idx])
            loss = F.binary_cross_entropy_with_logits(logits, y[idx], pos_weight=pos_weight)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    model.eval()
    with torch.no_grad():
        probs = torch.sigmoid(model(x))
    return model, {
        "num_components": int(y.numel()),
        "positive_ratio": float(y.mean().detach().cpu()),
        "pos_weight": float(pos_weight.detach().cpu()),
        "prob_mean": float(probs.mean().detach().cpu()),
        "prob_min": float(probs.min().detach().cpu()),
        "prob_max": float(probs.max().detach().cpu()),
    }


def select_component_verifier_threshold(args, model, x: torch.Tensor, y: torch.Tensor, device):
    if x.numel() == 0:
        return float(max(float(args.cmct_component_verifier_keep_threshold), 0.5)), {"threshold_mode": "fallback_empty"}
    x = x.to(device)
    y = y.to(device)
    rows = []
    best = None
    model.eval()
    with torch.no_grad():
        probs = torch.sigmoid(model(x))
        for threshold in base.parse_ratio_grid(args.cmct_component_verifier_threshold_grid):
            threshold = float(threshold)
            pred = probs >= threshold
            tp = ((pred > 0) & (y > 0.5)).sum().float()
            fp = ((pred > 0) & (y <= 0.5)).sum().float()
            fn = ((pred == 0) & (y > 0.5)).sum().float()
            precision = tp / (tp + fp).clamp_min(1.0)
            recall = tp / (tp + fn).clamp_min(1.0)
            f1 = 2.0 * precision * recall / (precision + recall).clamp_min(1e-8)
            iou = tp / (tp + fp + fn).clamp_min(1.0)
            metric = iou if str(args.cmct_component_verifier_select_metric) == "IoU" else f1
            if float(args.cmct_component_verifier_min_recall) > 0:
                metric = metric - float(args.cmct_component_verifier_recall_penalty) * torch.relu(
                    torch.tensor(float(args.cmct_component_verifier_min_recall), device=device) - recall
                )
            if float(args.cmct_component_verifier_min_precision) > 0:
                metric = metric - float(args.cmct_component_verifier_precision_penalty) * torch.relu(
                    torch.tensor(float(args.cmct_component_verifier_min_precision), device=device) - precision
                )
            row = {
                "threshold": threshold,
                "IoU": float(iou.detach().cpu()),
                "F1": float(f1.detach().cpu()),
                "Pre": float(precision.detach().cpu()),
                "Recall": float(recall.detach().cpu()),
                "score": float(metric.detach().cpu()),
            }
            rows.append(row)
            if best is None or row["score"] > best["score"]:
                best = row
    if best is None:
        return 0.5, {"threshold_mode": "fallback_no_grid"}
    return float(best["threshold"]), {
        "threshold_mode": "source_validation",
        "selected": best,
        "grid": rows,
    }


def load_or_train_component_verifier(args, all_source_items, processor, change_module, device, clipper):
    if not bool(getattr(args, "cmct_enable_component_verifier", False)):
        return None, {}, {}
    if str(args.component_gate_candidate_source) != "semantic_proxy":
        return None, {"enabled": True, "trained": False, "reason": "requires_semantic_proxy"}, {}
    if args.cmct_component_verifier_checkpoint:
        ckpt = torch.load(args.cmct_component_verifier_checkpoint, map_location=device, weights_only=False)
        model = base.ComponentGateHead(
            len(COMPONENT_VERIFIER_FEATURE_NAMES),
            hidden_dim=int(ckpt.get("config", {}).get("cmct_component_verifier_hidden_dim", args.cmct_component_verifier_hidden_dim)),
            dropout=float(ckpt.get("config", {}).get("cmct_component_verifier_dropout", args.cmct_component_verifier_dropout)),
        ).to(device)
        model.load_state_dict(ckpt["component_verifier_head"])
        model.eval()
        stats = dict(ckpt.get("component_verifier_stats", {}))
        collect_stats = dict(ckpt.get("component_verifier_collect_stats", {}))
        if float(args.cmct_component_verifier_keep_threshold) < 0 and "keep_threshold" in stats:
            args.cmct_component_verifier_keep_threshold = float(stats["keep_threshold"])
        stats["loaded_from_checkpoint"] = True
        return model, stats, collect_stats

    train_limit = int(getattr(args, "cmct_component_verifier_train_limit", 0))
    if train_limit <= 0:
        train_limit = int(args.train_limit)
    if train_limit <= 0:
        train_limit = 32
    train_items = base.split_items(all_source_items, train_limit, int(args.cmct_component_verifier_train_offset))
    x_train, y_train, collect_stats = collect_cacdd_component_verifier_examples(
        args,
        train_items,
        processor,
        change_module,
        device,
        clipper,
    )
    model, stats = train_component_verifier(args, x_train, y_train, device)
    if float(args.cmct_component_verifier_keep_threshold) < 0:
        val_limit = int(getattr(args, "cmct_component_verifier_val_limit", 0))
        if val_limit <= 0:
            val_limit = max(1, min(train_limit, int(args.source_val_limit) if int(args.source_val_limit) > 0 else 8))
        val_items = base.split_items(all_source_items, val_limit, int(args.cmct_component_verifier_val_offset))
        x_val, y_val, val_collect_stats = collect_cacdd_component_verifier_examples(
            args,
            val_items,
            processor,
            change_module,
            device,
            clipper,
        )
        selected_threshold, threshold_stats = select_component_verifier_threshold(args, model, x_val, y_val, device)
        args.cmct_component_verifier_keep_threshold = float(selected_threshold)
        stats["keep_threshold"] = float(selected_threshold)
        stats["threshold_selection"] = threshold_stats
        collect_stats["source_val_for_threshold"] = val_collect_stats
    else:
        stats["keep_threshold"] = float(args.cmct_component_verifier_keep_threshold)
    stats["loaded_from_checkpoint"] = False
    if args.output_component_verifier_checkpoint:
        torch.save({
            "component_verifier_head": model.state_dict(),
            "component_verifier_feature_names": COMPONENT_VERIFIER_FEATURE_NAMES,
            "component_verifier_stats": stats,
            "component_verifier_collect_stats": collect_stats,
            "config": vars(args),
        }, args.output_component_verifier_checkpoint)
        print(f"Saved component verifier checkpoint: {args.output_component_verifier_checkpoint}")
    return model, stats, collect_stats


@torch.no_grad()
def collect_cacdd_suppressed_recovery_examples(
    args,
    items: Sequence[dict],
    processor,
    change_module,
    device,
    clipper,
    bank,
    correction_model,
    component_model,
):
    feats = []
    labels_out = []
    stats = {
        "candidate_components": 0,
        "suppressed_components": 0,
        "positive_components": 0,
        "negative_components": 0,
        "skipped_no_suppression": 0,
    }
    if component_model is None:
        return torch.empty((0, suppressed_recovery_feature_dim(args))), torch.empty((0,)), {
            **stats,
            "reason": "requires_component_gate_model",
        }

    class_names, prompt_bank, confusers = source_prompt_spec(args)
    class_text_embeddings = None
    if bool(getattr(args, "cmct_suppressed_recovery_enable_clip_features", False)):
        class_text_embeddings = np.stack([
            _clipper_text_embedding(clipper, prompt_text_for_class(name, prompt_bank))
            for name in class_names
        ], axis=0).astype(np.float32)
    stats["source_classes"] = class_names
    stats["class_component_counts"] = {name: 0 for name in class_names}
    cfg = base._memory_config_from_args(args)
    component_model.eval()
    if correction_model is not None:
        correction_model.eval()

    for item in tqdm(items, desc="collect-suppressed-recovery", dynamic_ncols=True):
        meta, _, _, size_hw = bench.predict_change(processor, change_module, item, args, device)
        gt = bench.read_binary_mask(item["label"], size_hw).astype(bool)
        pair_embedding = clipper.encode_pair(item)
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
        bcd = meta["bcd_score"].float().clamp(0, 1)
        bcd_threshold = base.component_gate_threshold(bcd, args)
        bcd_threshold_value = float(bcd_threshold.detach().cpu())
        bcd_np = bcd.detach().cpu().numpy()
        for cls_idx, cls_name in enumerate(class_names):
            score = memory_rectified_score(
                posterior.posterior_a[cls_idx],
                posterior.posterior_b[cls_idx],
                None if args.disable_bcd_gate else meta["bcd_score"],
                presence_a=presence_a[cls_idx],
                presence_b=presence_b[cls_idx],
                config=cfg,
            )
            presence = torch.maximum(presence_a[cls_idx], presence_b[cls_idx])
            base_mask, base_threshold = base.decode_class_score(score, presence, 0.0, args)
            component_features = component_reliability_features(
                args,
                base_mask,
                score,
                meta["bcd_score"],
                base_threshold,
                component_model,
                device,
            )
            semantic_features = semantic_relation_features_for_prompt(args, bank, clipper, cls_name, prompt_bank)
            key, query_stats = make_query(args, item, cls_name, prompt_bank, pair_embedding, clipper, score, meta["bcd_score"], presence)
            tau_mem, tau_corr, diag = predict_cmct_thresholds(
                args,
                bank,
                correction_model,
                key,
                query_stats,
                device,
                cls_name=cls_name,
                component_features=component_features,
                semantic_features=semantic_features,
            )
            tau_mem = apply_adaptive_floor(tau_mem, base_threshold, args)
            tau_corr = apply_adaptive_floor(tau_corr, base_threshold, args)
            tau_mem = apply_class_prior_floor(tau_mem, args, bank, cls_name)
            tau_corr = apply_class_prior_floor(tau_corr, args, bank, cls_name)
            tau_corr, _ = apply_semantic_prior_floor(tau_corr, args, semantic_features)
            tau_corr, _ = apply_uncertainty_calibration(
                tau_corr,
                base_threshold,
                args,
                query_stats,
                diag,
                component_features=component_features,
                semantic_features=semantic_features,
            )
            ratio_mem = diag.get("ratio_mem") if bool(getattr(args, "cmct_enable_memory_ratio_budget", False)) else None
            corr_mask, corr_threshold = decode_fixed_tau(score, presence, tau_corr, args, budget_ratio=ratio_mem)
            gate_mask = base.component_gate_filter(
                corr_mask,
                score,
                meta["bcd_score"],
                corr_threshold,
                args,
                component_model,
                device,
            )
            gate_bool = gate_mask.astype(bool)
            relaxed_tau = float(tau_corr) * float(getattr(args, "cmct_suppressed_recovery_candidate_relax_scale", 0.80))
            relaxed_tau -= float(getattr(args, "cmct_suppressed_recovery_candidate_relax_delta", 0.0))
            relaxed_tau = float(np.clip(relaxed_tau, float(args.cmct_threshold_min), float(args.cmct_threshold_max)))
            relaxed_mask, _ = decode_fixed_tau(score, presence, relaxed_tau, args)
            relaxed_bool = relaxed_mask.astype(bool)
            if not relaxed_bool.any():
                continue

            score_np = score.detach().float().cpu().numpy()
            feature_score_np, feature_score_threshold = base.component_gate_feature_inputs(
                score_np,
                bcd_np,
                float(corr_threshold),
                bcd_threshold_value,
                args,
            )
            use_prat_region_features = bool(getattr(args, "cmct_suppressed_recovery_enable_prat_region_features", False))
            semantic_delta_np = None
            structure_support_np = None
            score_support_np = None
            if use_prat_region_features:
                semantic_delta_np = dense_semantic_delta_np(posterior.posterior_a, posterior.posterior_b, cls_idx)
                denom = max(float(corr_threshold) - float(relaxed_tau), 1e-6)
                score_support_np = np.clip((score_np - float(relaxed_tau)) / denom, 0.0, 1.0).astype(np.float32)
                structure_support_np = (np.clip(bcd_np, 0.0, 1.0) * score_support_np).astype(np.float32)
            labels_cc, num = base.connected_components(relaxed_bool)
            corr_bool = corr_mask.astype(bool)
            stats["candidate_components"] += int(num)
            for lab in range(1, num + 1):
                region = labels_cc == lab
                area = int(region.sum())
                if area < int(getattr(args, "component_gate_min_area", 8)):
                    continue
                if int(getattr(args, "component_gate_max_area", 0)) > 0 and area > int(args.component_gate_max_area):
                    continue
                suppressed_region = region & ~gate_bool
                suppressed_pixels = int(suppressed_region.sum())
                suppressed_ratio = float(suppressed_pixels) / max(float(area), 1.0)
                if (
                    suppressed_pixels < int(getattr(args, "cmct_suppressed_recovery_min_suppressed_pixels", 8))
                    or suppressed_ratio < float(getattr(args, "cmct_suppressed_recovery_min_suppressed_ratio", 0.05))
                ):
                    stats["skipped_no_suppression"] += 1
                    continue
                base_row = base.component_gate_features(
                    region,
                    feature_score_np,
                    bcd_np,
                    feature_score_threshold,
                    bcd_threshold_value,
                    topk_ratio=float(args.component_gate_topk_ratio),
                )
                x_gate = torch.tensor(base_row, dtype=torch.float32, device=device).view(1, -1)
                keep_prob = float(torch.sigmoid(component_model(x_gate)).detach().cpu().reshape(()))
                row = suppressed_recovery_feature_row(
                    base_row,
                    region,
                    corr_bool,
                    gate_bool,
                    feature_score_np,
                    bcd_np,
                    feature_score_threshold,
                    bcd_threshold_value,
                    keep_prob,
                    float(tau_corr),
                    float(relaxed_tau),
                    prat_region_features=prat_region_extra_feature_values(
                        region,
                        semantic_delta_np,
                        structure_support_np,
                        score_support_np,
                    ) if use_prat_region_features else None,
                )
                if bool(getattr(args, "cmct_suppressed_recovery_enable_clip_features", False)):
                    class_text_embedding = class_text_embeddings[cls_idx] if class_text_embeddings is not None else None
                    clip_features = _component_clip_similarity_feature_values(
                        args,
                        region,
                        item,
                        clipper,
                        class_text_embedding,
                        class_text_embeddings,
                        class_text_index=cls_idx,
                    )
                    row = np.concatenate([row, clip_features], axis=0).astype(np.float32)
                label = base.label_component_gate(suppressed_region, gt, args)
                feats.append(row)
                labels_out.append(label)
                stats["suppressed_components"] += 1
                stats["class_component_counts"][cls_name] += 1
                if label > 0.5:
                    stats["positive_components"] += 1
                else:
                    stats["negative_components"] += 1

    if not feats:
        return torch.empty((0, suppressed_recovery_feature_dim(args))), torch.empty((0,)), stats
    stats["positive_ratio"] = float(np.mean(labels_out))
    return torch.tensor(np.stack(feats, axis=0), dtype=torch.float32), torch.tensor(labels_out, dtype=torch.float32), stats


def train_suppressed_recovery(args, x: torch.Tensor, y: torch.Tensor, device):
    model = base.ComponentGateHead(
        suppressed_recovery_feature_dim(args),
        hidden_dim=int(args.cmct_suppressed_recovery_hidden_dim),
        dropout=float(args.cmct_suppressed_recovery_dropout),
    ).to(device)
    if x.numel() == 0:
        return model, {"num_components": 0, "positive_ratio": 0.0, "trained": False}
    x = x.to(device)
    y = y.to(device)
    pos = y.sum().clamp_min(1.0)
    neg = (y.numel() - y.sum()).clamp_min(1.0)
    pos_weight = (neg / pos * float(args.cmct_suppressed_recovery_positive_weight_scale)).clamp(
        0.5,
        float(args.cmct_suppressed_recovery_max_pos_weight),
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(args.cmct_suppressed_recovery_lr),
        weight_decay=float(args.cmct_suppressed_recovery_weight_decay),
    )
    for _ in range(int(args.cmct_suppressed_recovery_epochs)):
        model.train()
        perm = torch.randperm(x.shape[0], device=device)
        for start in range(0, x.shape[0], int(args.cmct_suppressed_recovery_batch_size)):
            idx = perm[start:start + int(args.cmct_suppressed_recovery_batch_size)]
            logits = model(x[idx])
            loss = F.binary_cross_entropy_with_logits(logits, y[idx], pos_weight=pos_weight)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    model.eval()
    with torch.no_grad():
        probs = torch.sigmoid(model(x))
    return model, {
        "num_components": int(y.numel()),
        "positive_ratio": float(y.mean().detach().cpu()),
        "pos_weight": float(pos_weight.detach().cpu()),
        "prob_mean": float(probs.mean().detach().cpu()),
        "prob_min": float(probs.min().detach().cpu()),
        "prob_max": float(probs.max().detach().cpu()),
        "trained": True,
    }


def select_suppressed_recovery_threshold(args, model, x: torch.Tensor, y: torch.Tensor, device):
    if x.numel() == 0:
        return float(max(float(args.cmct_suppressed_recovery_keep_threshold), 0.5)), {"threshold_mode": "fallback_empty"}
    x = x.to(device)
    y = y.to(device)
    rows = []
    best = None
    model.eval()
    with torch.no_grad():
        probs = torch.sigmoid(model(x))
        for threshold in base.parse_ratio_grid(args.cmct_suppressed_recovery_threshold_grid):
            threshold = float(threshold)
            pred = probs >= threshold
            tp = ((pred > 0) & (y > 0.5)).sum().float()
            fp = ((pred > 0) & (y <= 0.5)).sum().float()
            fn = ((pred == 0) & (y > 0.5)).sum().float()
            precision = tp / (tp + fp).clamp_min(1.0)
            recall = tp / (tp + fn).clamp_min(1.0)
            f1 = 2.0 * precision * recall / (precision + recall).clamp_min(1e-8)
            iou = tp / (tp + fp + fn).clamp_min(1.0)
            metric = iou if str(args.cmct_suppressed_recovery_select_metric) == "IoU" else f1
            if float(args.cmct_suppressed_recovery_min_recall) > 0:
                metric = metric - float(args.cmct_suppressed_recovery_recall_penalty) * torch.relu(
                    torch.tensor(float(args.cmct_suppressed_recovery_min_recall), device=device) - recall
                )
            if float(args.cmct_suppressed_recovery_min_precision) > 0:
                metric = metric - float(args.cmct_suppressed_recovery_precision_penalty) * torch.relu(
                    torch.tensor(float(args.cmct_suppressed_recovery_min_precision), device=device) - precision
                )
            row = {
                "threshold": threshold,
                "IoU": float(iou.detach().cpu()),
                "F1": float(f1.detach().cpu()),
                "Pre": float(precision.detach().cpu()),
                "Recall": float(recall.detach().cpu()),
                "score": float(metric.detach().cpu()),
            }
            rows.append(row)
            if best is None or row["score"] > best["score"]:
                best = row
    if best is None:
        return 0.5, {"threshold_mode": "fallback_no_grid"}
    return float(best["threshold"]), {
        "threshold_mode": "source_validation",
        "selected": best,
        "grid": rows,
    }


def load_or_train_suppressed_recovery(
    args,
    all_source_items,
    processor,
    change_module,
    device,
    clipper,
    bank,
    correction_model,
    component_model,
):
    if not bool(getattr(args, "cmct_enable_suppressed_component_recovery", False)):
        return None, {}, {}
    if args.cmct_suppressed_recovery_checkpoint:
        ckpt = torch.load(args.cmct_suppressed_recovery_checkpoint, map_location=device, weights_only=False)
        checkpoint_feature_names = ckpt.get("suppressed_recovery_feature_names", suppressed_recovery_feature_names(args))
        checkpoint_input_dim = len(checkpoint_feature_names)
        requested_input_dim = suppressed_recovery_feature_dim(args)
        if checkpoint_input_dim != requested_input_dim:
            raise ValueError(
                "Suppressed recovery checkpoint feature dimension mismatch: "
                f"checkpoint={checkpoint_input_dim}, requested={requested_input_dim}. "
                "Retrain or load a checkpoint produced with the same "
                "--cmct_suppressed_recovery_enable_clip_features setting."
            )
        model = base.ComponentGateHead(
            checkpoint_input_dim,
            hidden_dim=int(ckpt.get("config", {}).get("cmct_suppressed_recovery_hidden_dim", args.cmct_suppressed_recovery_hidden_dim)),
            dropout=float(ckpt.get("config", {}).get("cmct_suppressed_recovery_dropout", args.cmct_suppressed_recovery_dropout)),
        ).to(device)
        model.load_state_dict(ckpt["suppressed_recovery_head"])
        model.eval()
        stats = dict(ckpt.get("suppressed_recovery_stats", {}))
        collect_stats = dict(ckpt.get("suppressed_recovery_collect_stats", {}))
        if float(args.cmct_suppressed_recovery_keep_threshold) < 0 and "keep_threshold" in stats:
            args.cmct_suppressed_recovery_keep_threshold = float(stats["keep_threshold"])
        stats["loaded_from_checkpoint"] = True
        return model, stats, collect_stats
    if component_model is None:
        return None, {"enabled": True, "trained": False, "reason": "requires_component_gate_model"}, {}

    train_limit = int(getattr(args, "cmct_suppressed_recovery_train_limit", 0))
    if train_limit <= 0:
        train_limit = int(args.train_limit)
    if train_limit <= 0:
        train_limit = 32
    train_items = base.split_items(all_source_items, train_limit, int(args.cmct_suppressed_recovery_train_offset))
    x_train, y_train, collect_stats = collect_cacdd_suppressed_recovery_examples(
        args,
        train_items,
        processor,
        change_module,
        device,
        clipper,
        bank,
        correction_model,
        component_model,
    )
    model, stats = train_suppressed_recovery(args, x_train, y_train, device)
    if float(args.cmct_suppressed_recovery_keep_threshold) < 0:
        val_limit = int(getattr(args, "cmct_suppressed_recovery_val_limit", 0))
        if val_limit <= 0:
            val_limit = max(1, min(train_limit, int(args.source_val_limit) if int(args.source_val_limit) > 0 else 8))
        val_items = base.split_items(all_source_items, val_limit, int(args.cmct_suppressed_recovery_val_offset))
        x_val, y_val, val_collect_stats = collect_cacdd_suppressed_recovery_examples(
            args,
            val_items,
            processor,
            change_module,
            device,
            clipper,
            bank,
            correction_model,
            component_model,
        )
        selected_threshold, threshold_stats = select_suppressed_recovery_threshold(args, model, x_val, y_val, device)
        args.cmct_suppressed_recovery_keep_threshold = float(selected_threshold)
        stats["keep_threshold"] = float(selected_threshold)
        stats["threshold_selection"] = threshold_stats
        collect_stats["source_val_for_threshold"] = val_collect_stats
    else:
        stats["keep_threshold"] = float(args.cmct_suppressed_recovery_keep_threshold)
    stats["loaded_from_checkpoint"] = False
    if args.output_suppressed_recovery_checkpoint:
        torch.save({
            "suppressed_recovery_head": model.state_dict(),
            "suppressed_recovery_feature_names": suppressed_recovery_feature_names(args),
            "suppressed_recovery_stats": stats,
            "suppressed_recovery_collect_stats": collect_stats,
            "config": vars(args),
        }, args.output_suppressed_recovery_checkpoint)
        print(f"Saved suppressed recovery checkpoint: {args.output_suppressed_recovery_checkpoint}")
    return model, stats, collect_stats


def load_or_train_component_gate(args, all_source_items, processor, change_module, device):
    if (
        not bool(args.enable_component_gate)
        and not bool(getattr(args, "cmct_enable_component_reliability", False))
        and not bool(getattr(args, "cmct_enable_component_local_refine", False))
        and not bool(getattr(args, "cmct_enable_suppressed_component_recovery", False))
    ):
        return None, {}, {}
    if args.component_gate_checkpoint:
        ckpt = torch.load(args.component_gate_checkpoint, map_location=device, weights_only=False)
        model = base.ComponentGateHead(
            len(base.COMPONENT_GATE_FEATURE_NAMES),
            hidden_dim=int(ckpt.get("config", {}).get("component_gate_hidden_dim", args.component_gate_hidden_dim)),
            dropout=float(ckpt.get("config", {}).get("component_gate_dropout", args.component_gate_dropout)),
        ).to(device)
        model.load_state_dict(ckpt["component_gate_head"])
        model.eval()
        stats = dict(ckpt.get("component_gate_stats", {}))
        collect_stats = dict(ckpt.get("component_gate_collect_stats", {}))
        if float(args.component_gate_keep_threshold) < 0 and "keep_threshold" in stats:
            args.component_gate_keep_threshold = float(stats["keep_threshold"])
        stats["loaded_from_checkpoint"] = True
        return model, stats, collect_stats

    train_limit = int(args.component_gate_train_limit)
    if train_limit <= 0:
        train_limit = int(args.train_limit)
    if train_limit <= 0:
        train_limit = 32
    train_items = base.split_items(all_source_items, train_limit, int(args.component_gate_train_offset))
    x_gate, y_gate, collect_stats = base.collect_cacdd_component_gate_examples(
        args, train_items, processor, change_module, device,
    )
    model, stats = base.train_component_gate(args, x_gate, y_gate, device)
    if float(args.component_gate_keep_threshold) < 0:
        val_limit = int(args.component_gate_val_limit)
        if val_limit <= 0:
            val_limit = max(1, min(train_limit, int(args.source_val_limit) if int(args.source_val_limit) > 0 else 8))
        val_items = base.split_items(all_source_items, val_limit, int(args.component_gate_val_offset))
        x_val, y_val, val_collect_stats = base.collect_cacdd_component_gate_examples(
            args, val_items, processor, change_module, device,
        )
        selected_threshold, threshold_stats = base.select_component_gate_threshold(args, model, x_val, y_val, device)
        args.component_gate_keep_threshold = float(selected_threshold)
        stats["keep_threshold"] = float(selected_threshold)
        stats["threshold_selection"] = threshold_stats
        collect_stats["source_val_for_threshold"] = val_collect_stats
    stats["loaded_from_checkpoint"] = False
    if args.output_component_gate_checkpoint:
        torch.save({
            "component_gate_head": model.state_dict(),
            "component_gate_feature_names": base.COMPONENT_GATE_FEATURE_NAMES,
            "component_gate_stats": stats,
            "component_gate_collect_stats": collect_stats,
            "config": vars(args),
        }, args.output_component_gate_checkpoint)
        print(f"Saved component gate checkpoint: {args.output_component_gate_checkpoint}")
    return model, stats, collect_stats


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--benchmark_root", type=str, default="data/OVCD_Benchmark")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--sam3_checkpoint", type=str, default="weights/sam3.pt")
    parser.add_argument("--output_json", type=str, required=True)
    parser.add_argument("--prediction_output_dir", type=str, default="")
    parser.add_argument(
        "--prediction_method",
        type=str,
        default="clip_corrected_tau_component_gate",
    )
    parser.add_argument("--output_checkpoint", type=str, default=None)
    parser.add_argument("--load_checkpoint", type=str, default=None)
    parser.add_argument("--source_dataset", type=str, default="CACDD")
    parser.add_argument("--target_datasets", type=str, default="SECOND,SCSCD")
    parser.add_argument("--train_limit", type=int, default=4)
    parser.add_argument("--train_offset", type=int, default=0)
    parser.add_argument("--source_val_limit", type=int, default=2)
    parser.add_argument("--source_val_offset", type=int, default=128)
    parser.add_argument("--eval_limit", type=int, default=0)
    parser.add_argument("--eval_offset", type=int, default=0)

    parser.add_argument("--module_type", type=str, default="auto", choices=["auto", "temporal_lora", "temporal_gate_v4", "mdfm"])
    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--lora_alpha", type=float, default=16.0)
    parser.add_argument("--lora_dropout", type=float, default=0.0)
    parser.add_argument("--gamma_init", type=float, default=0.05)
    parser.add_argument("--head_hidden_dim", type=int, default=96)
    parser.add_argument("--max_gamma", type=float, default=0.2)
    parser.add_argument("--residual_clip_ratio", type=float, default=0.25)
    parser.add_argument("--gain_init", type=float, default=1.0)
    parser.add_argument("--gain_min", type=float, default=0.75)
    parser.add_argument("--gain_max", type=float, default=1.25)
    parser.add_argument("--gate_floor", type=float, default=0.02)
    parser.add_argument("--gate_temperature", type=float, default=1.0)
    parser.add_argument("--v4_active_levels", type=str, default="all")
    parser.add_argument("--v4_gate_level", type=int, default=0)

    parser.add_argument("--sam3_confidence_threshold", type=float, default=0.1)
    parser.add_argument("--rho", type=float, default=2.0)
    parser.add_argument("--disable_bcd_gate", action="store_true")
    parser.add_argument("--threshold_mode", type=str, default="adaptive",
                        choices=["percentile", "mean_std", "otsu", "otsu_edge", "candidate_otsu", "adaptive"])
    parser.add_argument("--threshold_percentile", type=float, default=85.0)
    parser.add_argument("--threshold_std_factor", type=float, default=1.0)
    parser.add_argument("--threshold_edge_weight", type=float, default=0.5)
    parser.add_argument("--threshold_edge_percentile", type=float, default=80.0)
    parser.add_argument("--threshold_edge_dilate", type=int, default=1)
    parser.add_argument("--threshold_edge_min_pixels", type=int, default=64)
    parser.add_argument("--threshold_min", type=float, default=0.0)
    parser.add_argument("--threshold_max", type=float, default=1.0)
    parser.add_argument("--threshold_scale", type=float, default=1.0)
    parser.add_argument("--threshold_bias", type=float, default=0.0)
    parser.add_argument("--threshold_max_foreground_ratio", type=float, default=0.0)
    parser.add_argument("--threshold_candidate_percentile", type=float, default=80.0)
    parser.add_argument("--presence_min", type=float, default=0.0)
    parser.add_argument("--min_score", type=float, default=0.01)
    parser.add_argument("--region_mode", type=str, default="component",
                        choices=["none", "component", "slic", "acf_component", "acf_region"])
    parser.add_argument("--region_score", type=str, default="mean", choices=["mean", "max", "topk_mean"])
    parser.add_argument("--region_topk_ratio", type=float, default=0.2)
    parser.add_argument("--min_region_area", type=int, default=4)
    parser.add_argument("--max_region_area", type=int, default=262144)
    parser.add_argument("--slic_segments", type=int, default=512)
    parser.add_argument("--acf_conf_percentile", type=float, default=50.0)
    parser.add_argument("--acf_conf_lambda", type=float, default=1.0)
    parser.add_argument("--acf_conf_min", type=float, default=0.0)
    parser.add_argument("--acf_conf_max", type=float, default=1.0)
    parser.add_argument("--acf_cv_max", type=float, default=2.0)

    parser.add_argument("--bcd_score_mode", type=str, default="head",
                        choices=["auto", "head", "hybrid", "fused", "residual", "feature"])
    parser.add_argument("--bcd_threshold_mode", type=str, default="percentile",
                        choices=["percentile", "mean_std", "otsu", "otsu_edge", "candidate_otsu", "adaptive"])
    parser.add_argument("--bcd_threshold_percentile", type=float, default=85.0)
    parser.add_argument("--bcd_threshold_abs", type=float, default=None)
    parser.add_argument("--bcd_threshold_std_factor", type=float, default=1.0)

    parser.add_argument("--bcd_soft_gate_threshold_mode", type=str, default="adaptive",
                        choices=["percentile", "mean_std", "otsu", "otsu_edge", "candidate_otsu", "adaptive"])
    parser.add_argument("--bcd_soft_gate_threshold_percentile", type=float, default=85.0)
    parser.add_argument("--bcd_soft_gate_threshold_abs", type=float, default=None)
    parser.add_argument("--bcd_soft_gate_threshold_std_factor", type=float, default=1.0)
    parser.add_argument("--bcd_soft_gate_threshold_min", type=float, default=0.0)
    parser.add_argument("--bcd_soft_gate_threshold_max", type=float, default=1.0)
    parser.add_argument("--bcd_soft_gate_topk_ratio", type=float, default=0.2)
    parser.add_argument("--bcd_soft_gate_threshold_scale", type=float, default=0.60)
    parser.add_argument("--bcd_soft_gate_min_overlap", type=float, default=0.015)
    parser.add_argument("--bcd_soft_gate_strong_score_scale", type=float, default=1.15)
    parser.add_argument("--bcd_soft_gate_min_area", type=int, default=8)
    parser.add_argument("--bcd_soft_gate_max_area", type=int, default=262144)
    parser.add_argument("--bcd_soft_gate_protect_area_ratio", type=float, default=0.02)

    parser.add_argument("--memory_delta_weight", type=float, default=0.55)
    parser.add_argument("--memory_transition_weight", type=float, default=0.25)
    parser.add_argument("--memory_bcd_alpha", type=float, default=1.0)
    parser.add_argument("--memory_bcd_floor", type=float, default=0.10)
    parser.add_argument("--memory_stable_suppress", type=float, default=0.45)
    parser.add_argument("--memory_stable_power", type=float, default=1.0)
    parser.add_argument("--memory_support_boost", type=float, default=0.0)
    parser.add_argument("--memory_presence_delta_weight", type=float, default=0.15)
    parser.add_argument("--memory_robust_normalize", action="store_true")
    parser.add_argument("--memory_q_low", type=float, default=0.02)
    parser.add_argument("--memory_q_high", type=float, default=0.98)


def add_cmct_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--clip_model", type=str, default="ViT-B/16")
    parser.add_argument("--clip_cache_dir", type=str, default="weights")
    parser.add_argument("--clip_memory_source_classes", type=str, default="union_remote")
    parser.add_argument("--clip_memory_max_source_classes", type=int, default=0)
    parser.add_argument("--clip_memory_topk", type=int, default=16)
    parser.add_argument("--clip_memory_temperature", type=float, default=0.07)
    parser.add_argument("--clip_memory_scene_weight", type=float, default=1.0)
    parser.add_argument("--clip_memory_delta_weight", type=float, default=0.7)
    parser.add_argument("--clip_memory_text_weight", type=float, default=1.0)
    parser.add_argument("--clip_memory_stats_weight", type=float, default=0.25)
    parser.add_argument("--clip_memory_class_filter", type=str, default="none", choices=["none", "exact", "alias"])
    parser.add_argument("--clip_memory_class_filter_min_rows", type=int, default=8)
    parser.add_argument("--clip_corrector_hidden_dim", type=int, default=64)
    parser.add_argument("--clip_corrector_dropout", type=float, default=0.05)
    parser.add_argument("--clip_corrector_epochs", type=int, default=60)
    parser.add_argument("--clip_corrector_batch_size", type=int, default=64)
    parser.add_argument("--clip_corrector_lr", type=float, default=1e-3)
    parser.add_argument("--clip_corrector_weight_decay", type=float, default=1e-4)
    parser.add_argument("--clip_corrector_max_delta", type=float, default=0.20)
    parser.add_argument("--cmct_threshold_grid", type=str, default="0.02:0.80:0.02")
    parser.add_argument("--cmct_threshold_target_metric", type=str, default="IoU", choices=["IoU", "F1"])
    parser.add_argument("--cmct_min_recall", type=float, default=0.35)
    parser.add_argument("--cmct_recall_penalty", type=float, default=1.0)
    parser.add_argument("--cmct_min_precision", type=float, default=0.0)
    parser.add_argument("--cmct_precision_penalty", type=float, default=1.0)
    parser.add_argument("--cmct_threshold_min", type=float, default=0.02)
    parser.add_argument("--cmct_threshold_max", type=float, default=0.80)
    parser.add_argument("--cmct_adaptive_floor_scale", type=float, default=0.0)
    parser.add_argument("--cmct_class_prior_floor_quantile", type=float, default=0.0)
    parser.add_argument("--cmct_class_prior_floor_scale", type=float, default=1.0)
    parser.add_argument("--cmct_class_prior_floor_min_rows", type=int, default=8)
    parser.add_argument("--cmct_enable_class_residual", action="store_true")
    parser.add_argument("--cmct_class_residual_min_rows", type=int, default=8)
    parser.add_argument("--cmct_enable_component_reliability", action="store_true")
    parser.add_argument("--cmct_enable_semantic_relation", action="store_true")
    parser.add_argument("--cmct_semantic_relation_temperature", type=float, default=0.10)
    parser.add_argument("--cmct_semantic_delta_confidence_gate", action="store_true")
    parser.add_argument("--cmct_semantic_delta_min_gate", type=float, default=0.35)
    parser.add_argument("--cmct_semantic_delta_margin_high", type=float, default=0.05)
    parser.add_argument("--cmct_enable_semantic_prior_floor", action="store_true")
    parser.add_argument("--cmct_semantic_prior_auto_select", action="store_true")
    parser.add_argument("--cmct_semantic_prior_floor_stat", type=str, default="weighted_q90", choices=["weighted_q75", "weighted_q90", "top1_q75", "mix_q75_q90"])
    parser.add_argument("--cmct_semantic_prior_floor_stat_grid", type=str, default="weighted_q75,weighted_q90,mix_q75_q90")
    parser.add_argument("--cmct_semantic_prior_floor_strength", type=float, default=1.0)
    parser.add_argument("--cmct_semantic_prior_floor_strength_grid", type=str, default="0.0,0.25,0.50,0.75,1.0")
    parser.add_argument("--cmct_semantic_prior_floor_scale", type=float, default=1.0)
    parser.add_argument("--cmct_semantic_prior_floor_scale_grid", type=str, default="0.90,1.00,1.10")
    parser.add_argument("--cmct_semantic_prior_floor_q90_mix", type=float, default=0.50)
    parser.add_argument("--cmct_semantic_prior_floor_min_top1_sim", type=float, default=-1.0)
    parser.add_argument("--cmct_semantic_prior_floor_min_margin", type=float, default=-1.0)
    parser.add_argument("--cmct_semantic_prior_floor_max_entropy", type=float, default=1.1)
    parser.add_argument("--cmct_enable_semantic_prior_anchor", action="store_true")
    parser.add_argument(
        "--cmct_semantic_prior_anchor_stat",
        type=str,
        default="mix_q75_q90",
        choices=["weighted_q50", "weighted_q75", "weighted_q90", "top1_q50", "top1_q75", "top1_q90", "mix_q50_q75", "mix_q75_q90"],
    )
    parser.add_argument("--cmct_semantic_prior_anchor_strength", type=float, default=0.50)
    parser.add_argument("--cmct_semantic_prior_anchor_scale", type=float, default=1.0)
    parser.add_argument("--cmct_semantic_prior_anchor_min_gate", type=float, default=0.0)
    parser.add_argument("--cmct_semantic_prior_anchor_confidence_power", type=float, default=1.0)
    parser.add_argument("--cmct_semantic_prior_anchor_min_top1_sim", type=float, default=-1.0)
    parser.add_argument("--cmct_semantic_prior_anchor_min_margin", type=float, default=-1.0)
    parser.add_argument("--cmct_semantic_prior_anchor_max_entropy", type=float, default=1.1)
    parser.add_argument("--cmct_semantic_prior_anchor_direction", type=str, default="both", choices=["both", "lower", "raise"])
    parser.add_argument("--cmct_semantic_prior_select_method", type=str, default="auto",
                        choices=["auto", "clip_corrected_tau", "clip_corrected_tau_softgate", "clip_corrected_tau_component_gate"])
    parser.add_argument("--cmct_semantic_prior_select_metric", type=str, default="IoU", choices=["IoU", "F1"])
    parser.add_argument("--cmct_semantic_prior_min_recall", type=float, default=0.45)
    parser.add_argument("--cmct_semantic_prior_recall_penalty", type=float, default=1.0)
    parser.add_argument("--cmct_semantic_prior_min_precision", type=float, default=0.0)
    parser.add_argument("--cmct_semantic_prior_precision_penalty", type=float, default=1.0)
    parser.add_argument("--cmct_enable_uncertainty_calibration", action="store_true")
    parser.add_argument("--cmct_uncertainty_strength", type=float, default=0.50)
    parser.add_argument("--cmct_uncertainty_min_mix", type=float, default=0.0)
    parser.add_argument("--cmct_uncertainty_max_mix", type=float, default=1.0)
    parser.add_argument("--cmct_uncertainty_auto_select", action="store_true")
    parser.add_argument("--cmct_uncertainty_strength_grid", type=str, default="0.0,0.25,0.50,0.75,1.0")
    parser.add_argument("--cmct_uncertainty_min_mix_grid", type=str, default="0.0,0.10,0.20")
    parser.add_argument("--cmct_uncertainty_select_metric", type=str, default="IoU", choices=["IoU", "F1"])
    parser.add_argument("--cmct_uncertainty_min_recall", type=float, default=0.45)
    parser.add_argument("--cmct_uncertainty_recall_penalty", type=float, default=1.0)
    parser.add_argument("--cmct_uncertainty_retrieval_low", type=float, default=0.10)
    parser.add_argument("--cmct_uncertainty_retrieval_high", type=float, default=0.35)
    parser.add_argument("--cmct_uncertainty_score_span_high", type=float, default=0.20)
    parser.add_argument("--cmct_uncertainty_score_std_high", type=float, default=0.12)
    parser.add_argument("--cmct_uncertainty_bcd_mean_high", type=float, default=0.20)
    parser.add_argument("--cmct_uncertainty_product_high", type=float, default=0.10)
    parser.add_argument("--cmct_uncertainty_semantic_margin_high", type=float, default=0.05)
    parser.add_argument("--cmct_uncertainty_retrieval_weight", type=float, default=1.0)
    parser.add_argument("--cmct_uncertainty_score_weight", type=float, default=0.75)
    parser.add_argument("--cmct_uncertainty_bcd_weight", type=float, default=0.75)
    parser.add_argument("--cmct_uncertainty_component_weight", type=float, default=0.50)
    parser.add_argument("--cmct_uncertainty_semantic_weight", type=float, default=0.50)
    parser.add_argument("--cmct_enable_component_local_refine", action="store_true")
    parser.add_argument("--cmct_component_local_auto_select", action="store_true")
    parser.add_argument("--cmct_component_local_relax_scale", type=float, default=0.75)
    parser.add_argument("--cmct_component_local_relax_scale_grid", type=str, default="0.50,0.65,0.80,0.90")
    parser.add_argument("--cmct_component_local_relax_delta", type=float, default=0.0)
    parser.add_argument("--cmct_component_local_add_threshold", type=float, default=0.70)
    parser.add_argument("--cmct_component_local_add_threshold_grid", type=str, default="0.60,0.70,0.80")
    parser.add_argument("--cmct_component_local_expand_threshold", type=float, default=0.70)
    parser.add_argument("--cmct_component_local_expand_threshold_grid", type=str, default="0.60,0.70,0.80")
    parser.add_argument("--cmct_component_local_min_bcd_support", type=float, default=0.05)
    parser.add_argument("--cmct_component_local_min_score_support", type=float, default=0.0)
    parser.add_argument("--cmct_component_local_min_score_support_grid", type=str, default="0.0,0.25,0.50")
    parser.add_argument("--cmct_component_local_min_score_topk_margin", type=float, default=-1.0)
    parser.add_argument("--cmct_component_local_min_score_topk_margin_grid", type=str, default="-1.0,0.0,0.02")
    parser.add_argument("--cmct_component_local_select_metric", type=str, default="IoU", choices=["IoU", "F1"])
    parser.add_argument("--cmct_component_local_min_recall", type=float, default=0.45)
    parser.add_argument("--cmct_component_local_recall_penalty", type=float, default=1.0)
    parser.add_argument("--cmct_component_local_min_precision", type=float, default=0.0)
    parser.add_argument("--cmct_component_local_precision_penalty", type=float, default=1.0)
    parser.add_argument("--cmct_component_local_class_auto_block", action="store_true")
    parser.add_argument("--cmct_component_local_block_classes", type=str, default="")
    parser.add_argument("--cmct_component_local_class_min_gain", type=float, default=0.0)
    parser.add_argument("--cmct_enable_component_clip_verify", action="store_true")
    parser.add_argument("--cmct_component_clip_min_sim", type=float, default=-1.0)
    parser.add_argument("--cmct_component_clip_min_sim_grid", type=str, default="-1.0")
    parser.add_argument("--cmct_component_clip_min_margin", type=float, default=-1.0)
    parser.add_argument("--cmct_component_clip_min_margin_grid", type=str, default="-1.0")
    parser.add_argument("--cmct_component_clip_min_area", type=int, default=64)
    parser.add_argument("--cmct_component_clip_pad", type=int, default=4)
    parser.add_argument("--cmct_component_clip_embedding", type=str, default="max_state", choices=["max_state", "after", "scene"])
    parser.add_argument("--cmct_enable_component_verifier", action="store_true")
    parser.add_argument("--cmct_component_verifier_checkpoint", type=str, default=None)
    parser.add_argument("--output_component_verifier_checkpoint", type=str, default=None)
    parser.add_argument("--cmct_component_verifier_train_limit", type=int, default=0)
    parser.add_argument("--cmct_component_verifier_train_offset", type=int, default=0)
    parser.add_argument("--cmct_component_verifier_val_limit", type=int, default=0)
    parser.add_argument("--cmct_component_verifier_val_offset", type=int, default=128)
    parser.add_argument("--cmct_component_verifier_hidden_dim", type=int, default=96)
    parser.add_argument("--cmct_component_verifier_dropout", type=float, default=0.05)
    parser.add_argument("--cmct_component_verifier_epochs", type=int, default=30)
    parser.add_argument("--cmct_component_verifier_batch_size", type=int, default=256)
    parser.add_argument("--cmct_component_verifier_lr", type=float, default=1e-3)
    parser.add_argument("--cmct_component_verifier_weight_decay", type=float, default=1e-4)
    parser.add_argument("--cmct_component_verifier_positive_weight_scale", type=float, default=1.0)
    parser.add_argument("--cmct_component_verifier_max_pos_weight", type=float, default=12.0)
    parser.add_argument("--cmct_component_verifier_keep_threshold", type=float, default=-1.0)
    parser.add_argument("--cmct_component_verifier_threshold_grid", type=str, default="0.20:0.80:0.05")
    parser.add_argument("--cmct_component_verifier_select_metric", type=str, default="IoU", choices=["IoU", "F1"])
    parser.add_argument("--cmct_component_verifier_min_recall", type=float, default=0.35)
    parser.add_argument("--cmct_component_verifier_recall_penalty", type=float, default=1.0)
    parser.add_argument("--cmct_component_verifier_min_precision", type=float, default=0.0)
    parser.add_argument("--cmct_component_verifier_precision_penalty", type=float, default=1.0)
    parser.add_argument("--cmct_component_verifier_candidate_relax_scale", type=float, default=1.0)
    parser.add_argument("--cmct_enable_suppressed_component_recovery", action="store_true")
    parser.add_argument("--cmct_suppressed_recovery_enable_clip_features", action="store_true")
    parser.add_argument("--cmct_suppressed_recovery_enable_prat_region_features", action="store_true")
    parser.add_argument("--cmct_suppressed_recovery_checkpoint", type=str, default=None)
    parser.add_argument("--output_suppressed_recovery_checkpoint", type=str, default=None)
    parser.add_argument("--cmct_suppressed_recovery_train_limit", type=int, default=0)
    parser.add_argument("--cmct_suppressed_recovery_train_offset", type=int, default=0)
    parser.add_argument("--cmct_suppressed_recovery_val_limit", type=int, default=0)
    parser.add_argument("--cmct_suppressed_recovery_val_offset", type=int, default=128)
    parser.add_argument("--cmct_suppressed_recovery_hidden_dim", type=int, default=64)
    parser.add_argument("--cmct_suppressed_recovery_dropout", type=float, default=0.05)
    parser.add_argument("--cmct_suppressed_recovery_epochs", type=int, default=30)
    parser.add_argument("--cmct_suppressed_recovery_batch_size", type=int, default=256)
    parser.add_argument("--cmct_suppressed_recovery_lr", type=float, default=1e-3)
    parser.add_argument("--cmct_suppressed_recovery_weight_decay", type=float, default=1e-4)
    parser.add_argument("--cmct_suppressed_recovery_positive_weight_scale", type=float, default=1.0)
    parser.add_argument("--cmct_suppressed_recovery_max_pos_weight", type=float, default=12.0)
    parser.add_argument("--cmct_suppressed_recovery_keep_threshold", type=float, default=-1.0)
    parser.add_argument("--cmct_suppressed_recovery_threshold_grid", type=str, default="0.20:0.80:0.05")
    parser.add_argument("--cmct_suppressed_recovery_select_metric", type=str, default="IoU", choices=["IoU", "F1"])
    parser.add_argument("--cmct_suppressed_recovery_min_recall", type=float, default=0.35)
    parser.add_argument("--cmct_suppressed_recovery_recall_penalty", type=float, default=1.0)
    parser.add_argument("--cmct_suppressed_recovery_min_precision", type=float, default=0.0)
    parser.add_argument("--cmct_suppressed_recovery_precision_penalty", type=float, default=1.0)
    parser.add_argument("--cmct_suppressed_recovery_candidate_relax_scale", type=float, default=0.80)
    parser.add_argument("--cmct_suppressed_recovery_candidate_relax_delta", type=float, default=0.0)
    parser.add_argument("--cmct_suppressed_recovery_min_suppressed_pixels", type=int, default=8)
    parser.add_argument("--cmct_suppressed_recovery_min_suppressed_ratio", type=float, default=0.05)
    parser.add_argument("--cmct_enable_class_prototype_threshold", action="store_true")
    parser.add_argument("--cmct_class_proto_auto_select", action="store_true")
    parser.add_argument("--cmct_class_proto_quantile", type=float, default=0.75)
    parser.add_argument("--cmct_class_proto_quantile_grid", type=str, default="0.50,0.75,0.90")
    parser.add_argument("--cmct_class_proto_mix", type=float, default=0.50)
    parser.add_argument("--cmct_class_proto_mix_grid", type=str, default="0.0,0.25,0.50,0.75,1.0")
    parser.add_argument("--cmct_class_proto_direction", type=str, default="interp", choices=["interp", "floor", "cap"])
    parser.add_argument("--cmct_class_proto_direction_grid", type=str, default="interp,floor,cap")
    parser.add_argument("--cmct_class_proto_exact_weight", type=float, default=0.70)
    parser.add_argument("--cmct_class_proto_min_rows", type=int, default=4)
    parser.add_argument("--cmct_class_proto_temperature", type=float, default=0.10)
    parser.add_argument("--cmct_class_proto_semantic_margin_high", type=float, default=0.05)
    parser.add_argument("--cmct_class_proto_reliability_floor", type=float, default=0.0)
    parser.add_argument("--cmct_class_proto_retrieval_low", type=float, default=0.10)
    parser.add_argument("--cmct_class_proto_retrieval_high", type=float, default=0.35)
    parser.add_argument("--cmct_class_proto_retrieval_weight", type=float, default=1.0)
    parser.add_argument("--cmct_class_proto_semantic_weight", type=float, default=1.0)
    parser.add_argument("--cmct_class_proto_component_weight", type=float, default=0.50)
    parser.add_argument("--cmct_class_proto_select_metric", type=str, default="IoU", choices=["IoU", "F1"])
    parser.add_argument("--cmct_class_proto_select_method", type=str, default="auto",
                        choices=["auto", "clip_corrected_tau", "clip_corrected_tau_softgate", "clip_corrected_tau_component_gate"])
    parser.add_argument("--cmct_class_proto_min_recall", type=float, default=0.45)
    parser.add_argument("--cmct_class_proto_recall_penalty", type=float, default=1.0)
    parser.add_argument("--cmct_class_proto_min_precision", type=float, default=0.0)
    parser.add_argument("--cmct_class_proto_precision_penalty", type=float, default=1.0)
    parser.add_argument("--cmct_enable_meta_threshold", action="store_true")
    parser.add_argument("--cmct_meta_force_retrain", action="store_true")
    parser.add_argument("--cmct_meta_hidden_dim", type=int, default=64)
    parser.add_argument("--cmct_meta_dropout", type=float, default=0.05)
    parser.add_argument("--cmct_meta_epochs", type=int, default=80)
    parser.add_argument("--cmct_meta_batch_size", type=int, default=64)
    parser.add_argument("--cmct_meta_lr", type=float, default=1e-3)
    parser.add_argument("--cmct_meta_weight_decay", type=float, default=1e-4)
    parser.add_argument("--cmct_meta_max_delta", type=float, default=0.20)
    parser.add_argument("--cmct_enable_class_residual_adapter", action="store_true")
    parser.add_argument("--cmct_class_residual_adapter_auto_select", action="store_true")
    parser.add_argument("--cmct_class_residual_adapter_stat", type=str, default="q50", choices=CMCT_CLASS_RESIDUAL_ADAPTER_STATS)
    parser.add_argument("--cmct_class_residual_adapter_stat_grid", type=str, default="q50,q75")
    parser.add_argument("--cmct_class_residual_adapter_mix", type=float, default=1.0)
    parser.add_argument("--cmct_class_residual_adapter_mix_grid", type=str, default="0.25,0.50,0.75,1.0")
    parser.add_argument("--cmct_class_residual_adapter_direction", type=str, default="signed",
                        choices=["signed", "positive", "negative"])
    parser.add_argument("--cmct_class_residual_adapter_direction_grid", type=str, default="signed,positive")
    parser.add_argument("--cmct_class_residual_adapter_temperature", type=float, default=0.10)
    parser.add_argument("--cmct_class_residual_adapter_exact_weight", type=float, default=0.70)
    parser.add_argument("--cmct_class_residual_adapter_min_rows", type=int, default=2)
    parser.add_argument("--cmct_class_residual_adapter_max_delta", type=float, default=0.20)
    parser.add_argument("--cmct_class_residual_adapter_semantic_margin_high", type=float, default=0.05)
    parser.add_argument("--cmct_class_residual_adapter_positive_risk_boost", type=float, default=0.0)
    parser.add_argument("--cmct_class_residual_adapter_select_metric", type=str, default="IoU", choices=["IoU", "F1"])
    parser.add_argument("--cmct_class_residual_adapter_min_recall", type=float, default=0.45)
    parser.add_argument("--cmct_class_residual_adapter_recall_penalty", type=float, default=1.0)
    parser.add_argument("--cmct_class_residual_adapter_min_precision", type=float, default=0.0)
    parser.add_argument("--cmct_class_residual_adapter_precision_penalty", type=float, default=1.0)
    parser.add_argument("--cmct_enable_score_calibration", action="store_true")
    parser.add_argument("--cmct_score_calibration_auto_select", action="store_true")
    parser.add_argument("--cmct_score_calibration_mode", type=str, default="shift_q95",
                        choices=CMCT_SCORE_CALIBRATION_MODES)
    parser.add_argument("--cmct_score_calibration_mode_grid", type=str, default="shift_q95,affine_q75_q95")
    parser.add_argument("--cmct_score_calibration_strength", type=float, default=0.0)
    parser.add_argument("--cmct_score_calibration_strength_grid", type=str, default="0.0,0.25,0.50,0.75,1.0")
    parser.add_argument("--cmct_score_calibration_temperature", type=float, default=0.10)
    parser.add_argument("--cmct_score_calibration_exact_weight", type=float, default=0.70)
    parser.add_argument("--cmct_score_calibration_min_rows", type=int, default=4)
    parser.add_argument("--cmct_score_calibration_eps", type=float, default=1e-4)
    parser.add_argument("--cmct_score_calibration_max_shift", type=float, default=2.0)
    parser.add_argument("--cmct_score_calibration_scale_min", type=float, default=0.50)
    parser.add_argument("--cmct_score_calibration_scale_max", type=float, default=2.00)
    parser.add_argument("--cmct_score_calibration_bcd_floor", type=float, default=0.05)
    parser.add_argument("--cmct_score_calibration_bcd_alpha", type=float, default=1.0)
    parser.add_argument("--cmct_score_calibration_confidence_gate", action="store_true")
    parser.add_argument("--cmct_score_calibration_min_confidence", type=float, default=0.35)
    parser.add_argument("--cmct_score_calibration_select_metric", type=str, default="IoU", choices=["IoU", "F1"])
    parser.add_argument("--cmct_score_calibration_min_recall", type=float, default=0.45)
    parser.add_argument("--cmct_score_calibration_recall_penalty", type=float, default=1.0)
    parser.add_argument("--cmct_score_calibration_min_precision", type=float, default=0.0)
    parser.add_argument("--cmct_score_calibration_precision_penalty", type=float, default=1.0)
    parser.add_argument("--cmct_enable_memory_ratio_budget", action="store_true")
    parser.add_argument("--cmct_memory_ratio_scale", type=float, default=1.0)
    parser.add_argument("--cmct_memory_ratio_min", type=float, default=1e-4)
    parser.add_argument("--cmct_memory_ratio_max", type=float, default=0.50)
    parser.add_argument("--cmct_memory_ratio_auto_select", action="store_true")
    parser.add_argument("--cmct_memory_ratio_scale_grid", type=str, default="1.0,1.25,1.5,2.0,2.5,3.0")
    parser.add_argument("--cmct_memory_ratio_max_grid", type=str, default="0.25,0.35,0.50")
    parser.add_argument("--cmct_memory_ratio_select_metric", type=str, default="IoU", choices=["IoU", "F1"])
    parser.add_argument("--cmct_memory_ratio_select_method", type=str, default="auto",
                        choices=["auto", "clip_corrected_tau", "clip_corrected_tau_softgate", "clip_corrected_tau_component_gate"])
    parser.add_argument("--cmct_memory_ratio_min_recall", type=float, default=0.45)
    parser.add_argument("--cmct_memory_ratio_recall_penalty", type=float, default=1.0)
    parser.add_argument("--cmct_oracle_positive_mode", type=str, default="binary", choices=["binary", "class_topq"])
    parser.add_argument("--cmct_oracle_positive_quantile", type=float, default=0.70)
    parser.add_argument("--cmct_oracle_positive_min_pixels", type=int, default=32)
    parser.add_argument("--cmct_oracle_positive_min_score", type=float, default=0.0)
    parser.add_argument("--enable_component_gate", action="store_true")
    parser.add_argument("--component_gate_checkpoint", type=str, default=None)
    parser.add_argument("--output_component_gate_checkpoint", type=str, default=None)
    parser.add_argument("--component_gate_train_limit", type=int, default=0)
    parser.add_argument("--component_gate_train_offset", type=int, default=0)
    parser.add_argument("--component_gate_val_limit", type=int, default=0)
    parser.add_argument("--component_gate_val_offset", type=int, default=128)
    parser.add_argument("--component_gate_candidate_source", type=str, default="semantic_proxy",
                        choices=["bcd", "semantic_proxy"])
    parser.add_argument("--component_gate_source_prompt_dataset", type=str, default="CLCD-512")
    parser.add_argument("--component_gate_source_classes", type=str,
                        default="Bareland,Grass,Road,Tree,Water,Cropland,Building")
    parser.add_argument("--component_gate_max_source_classes", type=int, default=0)
    parser.add_argument("--component_gate_bcd_threshold_mode", type=str, default="adaptive",
                        choices=["percentile", "mean_std", "otsu", "otsu_edge", "candidate_otsu", "adaptive"])
    parser.add_argument("--component_gate_bcd_threshold_percentile", type=float, default=85.0)
    parser.add_argument("--component_gate_bcd_threshold_abs", type=float, default=None)
    parser.add_argument("--component_gate_bcd_threshold_std_factor", type=float, default=1.0)
    parser.add_argument("--component_gate_bcd_threshold_min", type=float, default=0.0)
    parser.add_argument("--component_gate_bcd_threshold_max", type=float, default=1.0)
    parser.add_argument("--component_gate_feature_mode", type=str, default="full",
                        choices=["full", "bcd_only"])
    parser.add_argument("--component_gate_topk_ratio", type=float, default=0.2)
    parser.add_argument("--component_gate_min_area", type=int, default=8)
    parser.add_argument("--component_gate_max_area", type=int, default=262144)
    parser.add_argument("--component_gate_protect_area_ratio", type=float, default=0.04)
    parser.add_argument("--component_gate_label_overlap", type=float, default=0.30)
    parser.add_argument("--component_gate_label_gt_cover", type=float, default=0.05)
    parser.add_argument("--component_gate_label_min_pixels", type=int, default=8)
    parser.add_argument("--component_gate_hidden_dim", type=int, default=64)
    parser.add_argument("--component_gate_dropout", type=float, default=0.05)
    parser.add_argument("--component_gate_epochs", type=int, default=30)
    parser.add_argument("--component_gate_batch_size", type=int, default=256)
    parser.add_argument("--component_gate_lr", type=float, default=1e-3)
    parser.add_argument("--component_gate_weight_decay", type=float, default=1e-4)
    parser.add_argument("--component_gate_positive_weight_scale", type=float, default=1.0)
    parser.add_argument("--component_gate_max_pos_weight", type=float, default=12.0)
    parser.add_argument("--component_gate_keep_threshold", type=float, default=0.50)
    parser.add_argument("--component_gate_threshold_grid", type=str, default="0.20:0.80:0.05")
    parser.add_argument("--component_gate_select_metric", type=str, default="IoU", choices=["IoU", "F1"])
    parser.add_argument("--component_gate_min_recall", type=float, default=0.35)
    parser.add_argument("--component_gate_recall_penalty", type=float, default=1.0)
    parser.add_argument("--component_gate_strong_score_scale", type=float, default=1.15)
    parser.add_argument("--component_gate_keep_strong_semantic", action="store_true")
    parser.add_argument("--cmct_enable_multilabel_onepass", action="store_true")
    parser.add_argument("--cmct_enable_seg2change_complement", action="store_true")
    parser.add_argument("--cmct_seg2change_support_mode", type=str, default="hard_argmax",
                        choices=["hard_argmax", "soft_score"])
    parser.add_argument("--cmct_seg2change_confidence_threshold", type=float, default=0.10)
    parser.add_argument("--cmct_seg2change_disable_bcd_support", action="store_true")
    parser.add_argument("--cmct_seg2change_recovery_tau_margin", type=float, default=0.03)
    parser.add_argument("--cmct_seg2change_recovery_min_bcd", type=float, default=0.0)
    parser.add_argument("--cmct_seg2change_recovery_max_area_ratio", type=float, default=0.02)
    parser.add_argument("--cmct_seg2change_recovery_disable_component_gate", action="store_true")
    parser.add_argument("--cmct_seg2change_recovery_gate_keep_threshold", type=float, default=-1.0)
    parser.add_argument("--cmct_seg2change_enable_suppression", action="store_true")
    parser.add_argument("--cmct_seg2change_suppress_tau_margin", type=float, default=0.01)
    parser.add_argument("--cmct_seg2change_suppress_max_bcd", type=float, default=0.20)
    parser.add_argument("--cmct_enable_prat_lite", action="store_true")
    parser.add_argument("--cmct_prat_recovery_tau_margin", type=float, default=0.04)
    parser.add_argument("--cmct_prat_semantic_min", type=float, default=0.10)
    parser.add_argument("--cmct_prat_structure_min", type=float, default=0.20)
    parser.add_argument("--cmct_prat_min_bcd", type=float, default=0.0)
    parser.add_argument("--cmct_prat_region_min_score_q75", type=float, default=-1.0)
    parser.add_argument("--cmct_prat_region_min_semantic_q75", type=float, default=-1.0)
    parser.add_argument("--cmct_prat_region_min_structure_q75", type=float, default=-1.0)
    parser.add_argument("--cmct_prat_max_added_ratio", type=float, default=0.015)
    parser.add_argument("--cmct_prat_disable_component_gate", action="store_true")
    parser.add_argument("--cmct_prat_recovery_gate_keep_threshold", type=float, default=-1.0)
    parser.add_argument("--cmct_enable_multiclass_competition", action="store_true")
    parser.add_argument("--cmct_multiclass_mode", type=str, default="component_union",
                        choices=["component_union", "pixel_competition", "soft_suppress", "endpoint_topk"])
    parser.add_argument("--cmct_multiclass_pixel_score_mode", type=str, default="delta",
                        choices=["delta", "raw", "ratio"])
    parser.add_argument("--cmct_multiclass_topk_ratio", type=float, default=0.20)
    parser.add_argument("--cmct_multiclass_min_delta", type=float, default=0.0)
    parser.add_argument("--cmct_multiclass_min_margin", type=float, default=0.0)
    parser.add_argument("--cmct_multiclass_min_class_cover", type=float, default=0.05)
    parser.add_argument("--cmct_multiclass_min_bcd_support", type=float, default=0.05)
    parser.add_argument("--cmct_multiclass_support_weight", type=float, default=0.0)
    parser.add_argument("--cmct_multiclass_over_threshold_weight", type=float, default=0.0)
    parser.add_argument("--cmct_multiclass_assign_full_component", action="store_true")
    parser.add_argument("--cmct_multiclass_pixel_require_positive_delta", action="store_true")
    parser.add_argument("--cmct_multiclass_suppress_below_delta", type=float, default=0.0)
    parser.add_argument("--cmct_multiclass_suppress_below_ratio", type=float, default=1.0)
    parser.add_argument("--cmct_multiclass_endpoint_topk", type=int, default=1)
    parser.add_argument("--cmct_multiclass_endpoint_margin_slack", type=float, default=0.0)
    parser.add_argument("--cmct_multiclass_endpoint_keep_unambiguous", action="store_true")
    parser.add_argument("--seed", type=int, default=2026)


def main() -> None:
    parser = argparse.ArgumentParser(description="CLIP-guided Memory-Corrected Thresholding for strict OVCD")
    add_common_args(parser)
    add_cmct_args(parser)
    args = parser.parse_args()

    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))

    device = bench.select_device()
    print(f"Using device: {bench.describe_device(device)}")
    processor, change_module, module_type = base.load_models(args, device)
    print(f"Loaded change module: {module_type}")
    clipper = ClipFeatureExtractor(args.clip_model, args.clip_cache_dir, device=device)
    print(f"Loaded CLIP: {args.clip_model}")

    root = Path(args.benchmark_root)
    source_spec = bench.BINARY_DATASETS[args.source_dataset]
    all_source = bench.list_binary_items(root, source_spec)
    train_items = base.split_items(all_source, args.train_limit, args.train_offset)
    source_val_items = base.split_items(all_source, args.source_val_limit, args.source_val_offset)
    print(f"{args.source_dataset}: train {len(train_items)} / val {len(source_val_items)} / all {len(all_source)}")

    component_model, component_gate_stats, component_gate_collect_stats = load_or_train_component_gate(
        args, all_source, processor, change_module, device,
    )
    component_verifier_model, component_verifier_stats, component_verifier_collect_stats = load_or_train_component_verifier(
        args, all_source, processor, change_module, device, clipper,
    )
    bank, correction_model, clip_memory_stats, clip_corrector_stats = load_or_train_cmct(
        args, train_items, processor, change_module, device, clipper, component_model,
    )
    suppressed_recovery_model, suppressed_recovery_stats, suppressed_recovery_collect_stats = load_or_train_suppressed_recovery(
        args, all_source, processor, change_module, device, clipper, bank, correction_model, component_model,
    )
    semantic_prior_selection = select_semantic_prior_floor(
        args, source_val_items, processor, change_module, device,
        clipper, bank, correction_model, component_model,
    )
    update_cmct_checkpoint_semantic_prior_selection(args, semantic_prior_selection)
    uncertainty_selection = select_uncertainty_calibration(
        args, source_val_items, processor, change_module, device,
        clipper, bank, correction_model, component_model,
    )
    update_cmct_checkpoint_uncertainty_selection(args, uncertainty_selection)
    ratio_budget_selection = select_memory_ratio_budget(
        args, source_val_items, processor, change_module, device,
        clipper, bank, correction_model, component_model,
    )
    update_cmct_checkpoint_ratio_selection(args, ratio_budget_selection)
    class_proto_selection = select_class_prototype_threshold(
        args, source_val_items, processor, change_module, device,
        clipper, bank, correction_model, component_model,
    )
    update_cmct_checkpoint_class_proto_selection(args, class_proto_selection)
    score_calibration_selection = select_score_calibration(
        args, source_val_items, processor, change_module, device,
        clipper, bank, correction_model, component_model,
    )
    class_residual_adapter = select_class_residual_adapter(
        args, source_val_items, processor, change_module, device,
        clipper, bank, correction_model, component_model,
    )
    component_local_selection = select_component_local_refinement(
        args, source_val_items, processor, change_module, device,
        clipper, bank, correction_model, component_model, component_verifier_model,
    )
    update_cmct_checkpoint_component_local_selection(args, component_local_selection)
    meta_model, meta_threshold_stats = load_or_train_meta_threshold(
        args, source_val_items, processor, change_module, device,
        clipper, bank, correction_model, component_model,
    )

    targets = [x.strip() for x in args.target_datasets.split(",") if x.strip()]
    semantic_targets = []
    binary_targets = []
    semantic_items_by_dataset = {}
    binary_items_by_dataset = {}
    for dataset in targets:
        if dataset in ("SECOND", "SCSCD"):
            all_items = bench.list_semantic_items(root, dataset)
            semantic_items_by_dataset[dataset] = base.split_items(all_items, args.eval_limit, args.eval_offset)
            semantic_targets.append(dataset)
            print(f"{dataset}: eval {len(semantic_items_by_dataset[dataset])} / all {len(all_items)}")
        elif dataset in bench.BINARY_DATASETS:
            all_items = bench.list_binary_items(root, bench.BINARY_DATASETS[dataset])
            binary_items_by_dataset[dataset] = base.split_items(all_items, args.eval_limit, args.eval_offset)
            binary_targets.append(dataset)
            print(f"{dataset}: eval {len(binary_items_by_dataset[dataset])} / all {len(all_items)}")
        else:
            raise ValueError(f"Unknown target dataset: {dataset}")

    target_results = OrderedDict()
    if semantic_targets:
        target_results.update(evaluate_semantic_targets(
            args, semantic_targets, semantic_items_by_dataset, processor, change_module, device,
            clipper, bank, correction_model, component_model, component_verifier_model, suppressed_recovery_model, meta_model,
            class_residual_adapter, score_calibration_selection,
        ))
    if binary_targets:
        target_results.update(evaluate_binary_targets(
            args, binary_targets, binary_items_by_dataset, processor, change_module, device,
            clipper, bank, correction_model, component_model, component_verifier_model, suppressed_recovery_model, meta_model,
            class_residual_adapter, score_calibration_selection,
        ))

    out = {
        "config": vars(args),
        "module_type": module_type,
        "stat_names": CMCT_STAT_NAMES,
        "class_condition_names": CMCT_CLASS_CONDITION_NAMES if bool(args.cmct_enable_class_residual) else [],
        "component_reliability_names": CMCT_COMPONENT_RELIABILITY_NAMES if bool(args.cmct_enable_component_reliability) else [],
        "semantic_relation_names": CMCT_SEMANTIC_RELATION_NAMES if bool(args.cmct_enable_semantic_relation) else [],
        "oracle_metric_names": CMCT_ORACLE_METRIC_NAMES,
        "clip_memory_stats": clip_memory_stats,
        "clip_corrector_stats": clip_corrector_stats,
        "component_gate_feature_names": base.COMPONENT_GATE_FEATURE_NAMES if bool(args.enable_component_gate) else [],
        "component_gate_stats": component_gate_stats,
        "component_gate_collect_stats": component_gate_collect_stats,
        "component_verifier_feature_names": COMPONENT_VERIFIER_FEATURE_NAMES if bool(getattr(args, "cmct_enable_component_verifier", False)) else [],
        "component_verifier_stats": component_verifier_stats,
        "component_verifier_collect_stats": component_verifier_collect_stats,
        "suppressed_recovery_feature_names": suppressed_recovery_feature_names(args) if bool(getattr(args, "cmct_enable_suppressed_component_recovery", False)) else [],
        "suppressed_recovery_stats": suppressed_recovery_stats,
        "suppressed_recovery_collect_stats": suppressed_recovery_collect_stats,
        "cmct_semantic_prior_selection": semantic_prior_selection,
        "cmct_uncertainty_selection": uncertainty_selection,
        "cmct_memory_ratio_selection": ratio_budget_selection,
        "cmct_class_proto_selection": class_proto_selection,
        "cmct_score_calibration_selection": score_calibration_selection,
        "cmct_class_residual_adapter": class_residual_adapter,
        "cmct_component_local_selection": component_local_selection,
        "cmct_meta_threshold_stats": meta_threshold_stats,
        "cmct_meta_feature_names": cmct_meta_feature_names(args) if bool(getattr(args, "cmct_enable_meta_threshold", False)) else [],
        "target_results": target_results,
    }
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved metrics: {args.output_json}")


if __name__ == "__main__":
    main()
