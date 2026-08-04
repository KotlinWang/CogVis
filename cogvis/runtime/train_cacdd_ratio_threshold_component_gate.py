"""Train a CACDD-sourced adaptive foreground-ratio threshold for OVCD.

This experiment replaces dataset/class fixed thresholds with a source-trained
image-level foreground budget.  The ratio head is trained only on binary CACDD
labels from the source split.  At target inference time it sees no target
labels; it predicts a per-image changed-area ratio from the BCD score
distribution and uses that ratio as ``threshold_max_foreground_ratio`` for
open-vocabulary class-score decoding.
"""

from __future__ import annotations

import argparse
import json
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

import eval_ovcd_benchmark as bench
from memory_ovcd_readout import MemoryReadoutConfig, memory_rectified_score
from posterior_ovcd import (
    build_posterior_inputs,
    compute_score_threshold,
    decode_ovcd_score,
    extract_class_scores,
)
from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
from search_memory_readout import (
    add_prediction,
    bcd_mass_foreground_ratio,
    build_gt_masks,
    connected_components,
    dataset_spec,
    finalize_accumulator,
    new_accumulator,
)


RATIO_FEATURE_NAMES = [
    "score_mean",
    "score_std",
    "score_min",
    "score_max",
    "score_q50",
    "score_q75",
    "score_q85",
    "score_q90",
    "score_q95",
    "score_q98",
    "score_top1_mean",
    "score_top5_mean",
    "ratio_above_mean",
    "ratio_above_mean_std",
    "ratio_above_q85",
    "ratio_above_adaptive",
    "bcd_mass_ratio",
    "adaptive_threshold",
    "adaptive_component_count_log",
    "adaptive_largest_component_ratio",
    "adaptive_component_area_mean",
]
BCD_MASS_RATIO_FEATURE_INDEX = RATIO_FEATURE_NAMES.index("bcd_mass_ratio")

COMPONENT_GATE_FEATURE_NAMES = [
    "area_ratio",
    "sqrt_area_ratio",
    "bbox_area_ratio",
    "extent",
    "score_mean",
    "score_std",
    "score_max",
    "score_topk",
    "bcd_mean",
    "bcd_std",
    "bcd_max",
    "bcd_topk",
    "bcd_support_ratio",
    "score_support_ratio",
    "score_bcd_product_mean",
    "score_threshold",
    "bcd_threshold",
    "score_topk_margin",
    "bcd_topk_margin",
]


class RatioThresholdHead(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 64, dropout: float = 0.05):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class ComponentGateHead(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 64, dropout: float = 0.05):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def split_items(items: Sequence[dict], limit: int, offset: int = 0) -> List[dict]:
    offset = max(int(offset), 0)
    selected = list(items)[offset:]
    if int(limit) > 0:
        selected = selected[: int(limit)]
    return selected


def parse_ratio_grid(spec: str) -> List[float]:
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


def ratio_from_logit(logit: torch.Tensor, args, features: Optional[torch.Tensor] = None) -> torch.Tensor:
    lo = float(args.ratio_min)
    hi = float(args.ratio_max)
    if args.ratio_output_mode == "residual_bcd_mass":
        if features is None:
            raise ValueError("features are required for ratio_output_mode=residual_bcd_mass")
        base = features[..., BCD_MASS_RATIO_FEATURE_INDEX].float().clamp(lo, hi)
        delta = float(args.ratio_residual_max_log_scale) * torch.tanh(logit)
        return (base * torch.exp(delta)).clamp(lo, hi)
    return lo + (hi - lo) * torch.sigmoid(logit)


def _top_mean(values: np.ndarray, ratio: float) -> float:
    if values.size == 0:
        return 0.0
    k = max(1, int(values.size * float(ratio)))
    return float(np.partition(values, -k)[-k:].mean())


def score_distribution_features(score: torch.Tensor, args) -> List[float]:
    score = score.float().clamp(0, 1)
    flat = score.detach().cpu().numpy().astype(np.float32).reshape(-1)
    if flat.size == 0:
        return [0.0] * len(RATIO_FEATURE_NAMES)

    q50, q75, q85, q90, q95, q98 = np.quantile(flat, [0.50, 0.75, 0.85, 0.90, 0.95, 0.98])
    mean = float(flat.mean())
    std = float(flat.std())
    adaptive_th = compute_score_threshold(
        score,
        mode=args.bcd_mass_threshold_mode,
        percentile=args.bcd_mass_threshold_percentile,
        threshold_abs=_optional_abs_threshold(args.bcd_mass_threshold_abs),
        std_factor=args.bcd_mass_threshold_std_factor,
        threshold_min=args.bcd_mass_threshold_min,
        threshold_max=args.bcd_mass_threshold_max,
        threshold_max_foreground_ratio=0.0,
    )
    adaptive_value = float(adaptive_th.detach().cpu())
    bcd_mass_ratio = float(np.clip(
        float((flat > adaptive_value).mean()) * float(args.bcd_mass_ratio_scale),
        float(args.bcd_mass_ratio_min),
        float(args.bcd_mass_ratio_max),
    ))
    adaptive_mask = (flat > adaptive_value).reshape(score.shape[-2], score.shape[-1])
    labels, num = connected_components(adaptive_mask)
    component_areas = []
    for lab in range(1, num + 1):
        area = float((labels == lab).sum()) / float(flat.size)
        if area > 0:
            component_areas.append(area)
    largest = max(component_areas) if component_areas else 0.0
    area_mean = float(np.mean(component_areas)) if component_areas else 0.0

    return [
        mean,
        std,
        float(flat.min()),
        float(flat.max()),
        float(q50),
        float(q75),
        float(q85),
        float(q90),
        float(q95),
        float(q98),
        _top_mean(flat, 0.01),
        _top_mean(flat, 0.05),
        float((flat > mean).mean()),
        float((flat > mean + std).mean()),
        float((flat > q85).mean()),
        float((flat > adaptive_value).mean()),
        bcd_mass_ratio,
        adaptive_value,
        float(np.log1p(num)),
        float(largest),
        float(area_mean),
    ]


def _optional_abs_threshold(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    value = float(value)
    if value < 0:
        return None
    return value


def confusion_from_binary(gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    return bench.confusion_binary(gt.astype(np.int64), pred.astype(np.int64))


def oracle_ratio_for_score(score: torch.Tensor, gt: np.ndarray, args) -> float:
    flat = score.detach().float().cpu().numpy().astype(np.float32).reshape(-1)
    gt_flat = gt.astype(bool).reshape(-1)
    if flat.size == 0:
        return float(args.ratio_min)
    if int(gt_flat.sum()) == 0:
        return float(args.ratio_min)

    best_ratio = float(args.ratio_min)
    best_score = -float("inf")
    for ratio in parse_ratio_grid(args.ratio_grid):
        ratio = float(np.clip(ratio, float(args.ratio_min), float(args.ratio_max)))
        threshold = float(np.quantile(flat, 1.0 - ratio))
        pred = flat > threshold
        tp = float((pred & gt_flat).sum())
        fp = float((pred & ~gt_flat).sum())
        fn = float((~pred & gt_flat).sum())
        precision = tp / max(tp + fp, 1e-8)
        recall = tp / max(tp + fn, 1e-8)
        iou = tp / max(tp + fp + fn, 1e-8)
        f1 = 2.0 * precision * recall / max(precision + recall, 1e-8)
        score_value = iou if args.ratio_target_metric == "IoU" else f1
        if recall < float(args.ratio_min_recall):
            score_value -= float(args.ratio_recall_penalty) * (float(args.ratio_min_recall) - recall)
        if score_value > best_score:
            best_score = score_value
            best_ratio = ratio
    return float(best_ratio)


def mask_from_ratio(score: torch.Tensor, ratio: float) -> np.ndarray:
    flat = score.detach().float().cpu().numpy().astype(np.float32).reshape(-1)
    if flat.size == 0:
        return np.zeros(tuple(score.shape[-2:]), dtype=np.int64)
    ratio = float(np.clip(ratio, 1e-6, 1.0))
    threshold = float(np.quantile(flat, 1.0 - ratio))
    return (flat > threshold).reshape(tuple(score.shape[-2:])).astype(np.int64)


@torch.no_grad()
def collect_cacdd_examples(args, items: Sequence[dict], processor, change_module, device):
    x_rows = []
    y_rows = []
    heuristic_cms = {
        "oracle_ratio": np.zeros((2, 2), dtype=np.int64),
        "bcd_mass_ratio": np.zeros((2, 2), dtype=np.int64),
    }
    target_ratios = []
    for item in tqdm(items, desc="collect-CACDD", dynamic_ncols=True):
        meta, _, _, size_hw = bench.predict_change(processor, change_module, item, args, device)
        gt = bench.read_binary_mask(item["label"], size_hw)
        features = score_distribution_features(meta["bcd_score"], args)
        target_ratio = oracle_ratio_for_score(meta["bcd_score"], gt, args)
        mass_ratio = bcd_mass_foreground_ratio(meta["bcd_score"], args)
        x_rows.append(features)
        y_rows.append(target_ratio)
        target_ratios.append(target_ratio)
        heuristic_cms["oracle_ratio"] += confusion_from_binary(gt, mask_from_ratio(meta["bcd_score"], target_ratio))
        heuristic_cms["bcd_mass_ratio"] += confusion_from_binary(gt, mask_from_ratio(meta["bcd_score"], mass_ratio))
    if not x_rows:
        return torch.empty((0, len(RATIO_FEATURE_NAMES))), torch.empty((0,)), {}, []
    return (
        torch.tensor(x_rows, dtype=torch.float32),
        torch.tensor(y_rows, dtype=torch.float32),
        {name: {k: float(v) for k, v in bench.metrics_from_confusion(cm).items()} for name, cm in heuristic_cms.items()},
        target_ratios,
    )


def train_ratio_head(args, x: torch.Tensor, y: torch.Tensor, device):
    model = RatioThresholdHead(
        len(RATIO_FEATURE_NAMES),
        hidden_dim=args.ratio_hidden_dim,
        dropout=args.ratio_dropout,
    ).to(device)
    if x.numel() == 0:
        return model, {"num_images": 0}

    x = x.to(device)
    y = y.to(device).clamp(float(args.ratio_min), float(args.ratio_max))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.ratio_lr, weight_decay=args.ratio_weight_decay)
    for _ in range(args.ratio_epochs):
        model.train()
        perm = torch.randperm(x.shape[0], device=device)
        for start in range(0, x.shape[0], args.ratio_batch_size):
            idx = perm[start:start + args.ratio_batch_size]
            pred = ratio_from_logit(model(x[idx]), args, x[idx])
            loss = F.smooth_l1_loss(torch.log(pred + 1e-6), torch.log(y[idx] + 1e-6))
            if args.ratio_output_mode == "residual_bcd_mass" and float(args.ratio_residual_l1_weight) > 0:
                base = x[idx, BCD_MASS_RATIO_FEATURE_INDEX].to(device).clamp(float(args.ratio_min), float(args.ratio_max))
                loss = loss + float(args.ratio_residual_l1_weight) * torch.abs(torch.log(pred + 1e-6) - torch.log(base + 1e-6)).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        pred = ratio_from_logit(model(x), args, x)
        loss = F.smooth_l1_loss(torch.log(pred + 1e-6), torch.log(y + 1e-6))
    return model, {
        "num_images": int(y.numel()),
        "target_mean": float(y.mean().detach().cpu()),
        "target_min": float(y.min().detach().cpu()),
        "target_max": float(y.max().detach().cpu()),
        "pred_mean": float(pred.mean().detach().cpu()),
        "pred_min": float(pred.min().detach().cpu()),
        "pred_max": float(pred.max().detach().cpu()),
        "log_smooth_l1": float(loss.detach().cpu()),
        "ratio_output_mode": str(args.ratio_output_mode),
    }


@torch.no_grad()
def evaluate_cacdd_ratio(args, items: Sequence[dict], processor, change_module, device, model):
    out_cms = OrderedDict([
        ("bcd_default", np.zeros((2, 2), dtype=np.int64)),
        ("bcd_mass_ratio", np.zeros((2, 2), dtype=np.int64)),
        ("cacdd_learned_ratio", np.zeros((2, 2), dtype=np.int64)),
        ("oracle_ratio", np.zeros((2, 2), dtype=np.int64)),
    ])
    ratios = OrderedDict((name, []) for name in out_cms.keys())
    model.eval()
    for item in tqdm(items, desc="eval-CACDD", dynamic_ncols=True):
        meta, _, bcd_np, size_hw = bench.predict_change(processor, change_module, item, args, device)
        gt = bench.read_binary_mask(item["label"], size_hw)
        feature = torch.tensor(score_distribution_features(meta["bcd_score"], args), dtype=torch.float32, device=device).view(1, -1)
        learned_ratio = float(ratio_from_logit(model(feature), args, feature).detach().cpu().reshape(()))
        mass_ratio = bcd_mass_foreground_ratio(meta["bcd_score"], args)
        oracle_ratio = oracle_ratio_for_score(meta["bcd_score"], gt, args)
        preds = {
            "bcd_default": bcd_np,
            "bcd_mass_ratio": mask_from_ratio(meta["bcd_score"], mass_ratio),
            "cacdd_learned_ratio": mask_from_ratio(meta["bcd_score"], learned_ratio),
            "oracle_ratio": mask_from_ratio(meta["bcd_score"], oracle_ratio),
        }
        ratio_values = {
            "bcd_default": float((bcd_np > 0).mean()),
            "bcd_mass_ratio": float(mass_ratio),
            "cacdd_learned_ratio": float(learned_ratio),
            "oracle_ratio": float(oracle_ratio),
        }
        for name, pred in preds.items():
            out_cms[name] += confusion_from_binary(gt, pred)
            ratios[name].append(ratio_values[name])
    return OrderedDict(
        (name, {
            "metrics": {k: float(v) for k, v in bench.metrics_from_confusion(cm).items()},
            "ratio_mean": float(np.mean(ratios[name])) if ratios[name] else 0.0,
            "ratio_min": float(np.min(ratios[name])) if ratios[name] else 0.0,
            "ratio_max": float(np.max(ratios[name])) if ratios[name] else 0.0,
        })
        for name, cm in out_cms.items()
    )


def _memory_config_from_args(args) -> MemoryReadoutConfig:
    return MemoryReadoutConfig(
        delta_weight=args.memory_delta_weight,
        transition_weight=args.memory_transition_weight,
        bcd_alpha=args.memory_bcd_alpha,
        bcd_floor=args.memory_bcd_floor,
        stable_suppress=args.memory_stable_suppress,
        stable_power=args.memory_stable_power,
        support_boost=args.memory_support_boost,
        presence_delta_weight=args.memory_presence_delta_weight,
        robust_normalize=args.memory_robust_normalize,
        q_low=args.memory_q_low,
        q_high=args.memory_q_high,
    )


def decode_class_score(score, presence, ratio: float, args):
    pred_mask, threshold = decode_ovcd_score(
        score,
        threshold_mode=args.threshold_mode,
        threshold_percentile=args.threshold_percentile,
        threshold_abs=None,
        threshold_std_factor=args.threshold_std_factor,
        threshold_edge_weight=args.threshold_edge_weight,
        threshold_edge_percentile=args.threshold_edge_percentile,
        threshold_edge_dilate=args.threshold_edge_dilate,
        threshold_edge_min_pixels=args.threshold_edge_min_pixels,
        threshold_min=args.threshold_min,
        threshold_max=args.threshold_max,
        threshold_scale=args.threshold_scale,
        threshold_bias=args.threshold_bias,
        threshold_max_foreground_ratio=ratio,
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


def _region_topk_mean(values: np.ndarray, topk_ratio: float) -> float:
    if values.size == 0:
        return 0.0
    k = max(1, int(values.size * float(topk_ratio)))
    return float(np.partition(values, -k)[-k:].mean())


def component_gate_threshold(bcd_score: torch.Tensor, args) -> torch.Tensor:
    return compute_score_threshold(
        bcd_score.float().clamp(0, 1),
        mode=args.component_gate_bcd_threshold_mode,
        percentile=args.component_gate_bcd_threshold_percentile,
        threshold_abs=_optional_abs_threshold(args.component_gate_bcd_threshold_abs),
        std_factor=args.component_gate_bcd_threshold_std_factor,
        threshold_min=args.component_gate_bcd_threshold_min,
        threshold_max=args.component_gate_bcd_threshold_max,
        threshold_max_foreground_ratio=0.0,
    )


def component_gate_features(region: np.ndarray, score_np: np.ndarray, bcd_np: np.ndarray,
                            score_threshold: float, bcd_threshold_value: float,
                            topk_ratio: float = 0.2) -> List[float]:
    h, w = region.shape
    area = float(region.sum())
    if area <= 0:
        return [0.0] * len(COMPONENT_GATE_FEATURE_NAMES)
    ys, xs = np.where(region)
    bbox_area = float((ys.max() - ys.min() + 1) * (xs.max() - xs.min() + 1))
    image_area = float(h * w)
    score_values = score_np[region]
    bcd_values = bcd_np[region]
    score_topk = _region_topk_mean(score_values, topk_ratio)
    bcd_topk = _region_topk_mean(bcd_values, topk_ratio)
    return [
        area / image_area,
        float(np.sqrt(area / image_area)),
        bbox_area / image_area,
        area / max(bbox_area, 1.0),
        float(score_values.mean()),
        float(score_values.std()),
        float(score_values.max()),
        float(score_topk),
        float(bcd_values.mean()),
        float(bcd_values.std()),
        float(bcd_values.max()),
        float(bcd_topk),
        float((bcd_values > bcd_threshold_value).mean()),
        float((score_values > score_threshold).mean()),
        float((score_values * bcd_values).mean()),
        float(score_threshold),
        float(bcd_threshold_value),
        float(score_topk - score_threshold),
        float(bcd_topk - bcd_threshold_value),
    ]


def component_gate_feature_inputs(score_np: np.ndarray, bcd_np: np.ndarray,
                                  score_threshold: float, bcd_threshold_value: float,
                                  args) -> Tuple[np.ndarray, float]:
    if args.component_gate_feature_mode == "bcd_only":
        return bcd_np, float(bcd_threshold_value)
    return score_np, float(score_threshold)


def source_component_prompt_spec(args):
    dataset = str(args.component_gate_source_prompt_dataset)
    if dataset not in bench.BINARY_DATASETS:
        raise ValueError(f"Unknown component gate source prompt dataset: {dataset}")
    specs = bench.BINARY_DATASETS[dataset].get("posterior_specs")
    if specs is None:
        raise ValueError(f"{dataset} does not provide posterior_specs for semantic proxy components")
    class_names = list(specs.keys())
    requested = [x.strip() for x in str(args.component_gate_source_classes).split(",") if x.strip()]
    if requested:
        missing = [name for name in requested if name not in specs]
        if missing:
            raise ValueError(f"Unknown source component classes for {dataset}: {missing}")
        class_names = requested
    if int(args.component_gate_max_source_classes) > 0:
        class_names = class_names[: int(args.component_gate_max_source_classes)]
    selected_specs = OrderedDict((name, specs[name]) for name in class_names)
    return (
        class_names,
        bench.specs_to_prompt_bank(selected_specs),
        bench.specs_to_confusers(selected_specs),
    )


def label_component_gate(region: np.ndarray, gt_mask: np.ndarray, args) -> float:
    gt = gt_mask.astype(bool)
    inter = float(np.logical_and(region, gt).sum())
    area = float(region.sum())
    gt_area = float(gt.sum())
    comp_overlap = inter / max(area, 1.0)
    gt_cover = inter / max(gt_area, 1.0)
    positive = (
        inter >= float(args.component_gate_label_min_pixels)
        and (
            comp_overlap >= float(args.component_gate_label_overlap)
            or gt_cover >= float(args.component_gate_label_gt_cover)
        )
    )
    return 1.0 if positive else 0.0


@torch.no_grad()
def collect_cacdd_component_gate_examples(args, items: Sequence[dict], processor, change_module, device):
    feats = []
    labels_out = []
    stats = {
        "candidate_source": str(args.component_gate_candidate_source),
        "candidate_components": 0,
        "positive_components": 0,
        "negative_components": 0,
    }
    semantic_source = str(args.component_gate_candidate_source) == "semantic_proxy"
    if semantic_source:
        source_class_names, source_prompt_bank, source_confusers = source_component_prompt_spec(args)
        stats["source_prompt_dataset"] = str(args.component_gate_source_prompt_dataset)
        stats["source_classes"] = source_class_names
        stats["class_component_counts"] = {name: 0 for name in source_class_names}
        cfg = _memory_config_from_args(args)
    for item in tqdm(items, desc="collect-CACDD-component-gate", dynamic_ncols=True):
        meta, _, _, size_hw = bench.predict_change(processor, change_module, item, args, device)
        gt = bench.read_binary_mask(item["label"], size_hw)
        bcd = meta["bcd_score"].float().clamp(0, 1)
        bcd_th = component_gate_threshold(bcd, args)
        bcd_th_value = float(bcd_th.detach().cpu())
        bcd_np = bcd.detach().cpu().numpy()

        component_regions: List[Tuple[np.ndarray, torch.Tensor, float, Optional[str]]] = []
        if semantic_source:
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
                candidate_np, score_threshold = decode_class_score(score, presence, 0.0, args)
                labels_cc, num = connected_components(candidate_np.astype(bool))
                for lab in range(1, num + 1):
                    component_regions.append((labels_cc == lab, score, float(score_threshold), cls_name))
        else:
            candidate_np = bcd_np > bcd_th_value
            labels_cc, num = connected_components(candidate_np)
            for lab in range(1, num + 1):
                component_regions.append((labels_cc == lab, bcd, bcd_th_value, None))

        for region, score_tensor, score_threshold_value, cls_name in component_regions:
            area = int(region.sum())
            if area < int(args.component_gate_min_area):
                continue
            if int(args.component_gate_max_area) > 0 and area > int(args.component_gate_max_area):
                continue
            score_np = score_tensor.detach().float().cpu().numpy()
            feature_score_np, feature_score_threshold = component_gate_feature_inputs(
                score_np, bcd_np, float(score_threshold_value), bcd_th_value, args,
            )
            row = component_gate_features(
                region,
                feature_score_np,
                bcd_np,
                feature_score_threshold,
                bcd_th_value,
                topk_ratio=float(args.component_gate_topk_ratio),
            )
            label = label_component_gate(region, gt, args)
            feats.append(row)
            labels_out.append(label)
            stats["candidate_components"] += 1
            if cls_name is not None:
                stats["class_component_counts"][cls_name] += 1
            if label > 0.5:
                stats["positive_components"] += 1
            else:
                stats["negative_components"] += 1
    if not feats:
        return torch.empty((0, len(COMPONENT_GATE_FEATURE_NAMES))), torch.empty((0,)), stats
    stats["positive_ratio"] = float(np.mean(labels_out))
    return torch.tensor(feats, dtype=torch.float32), torch.tensor(labels_out, dtype=torch.float32), stats


def select_component_gate_threshold(args, model: nn.Module, x: torch.Tensor, y: torch.Tensor, device):
    if x.numel() == 0:
        return float(max(args.component_gate_keep_threshold, 0.5)), {"threshold_mode": "fallback_empty"}
    x = x.to(device)
    y = y.to(device)
    values = parse_ratio_grid(args.component_gate_threshold_grid)
    best = None
    rows = []
    model.eval()
    with torch.no_grad():
        probs = torch.sigmoid(model(x))
        for threshold in values:
            threshold = float(threshold)
            pred = probs >= threshold
            tp = ((pred > 0) & (y > 0.5)).sum().float()
            fp = ((pred > 0) & (y <= 0.5)).sum().float()
            fn = ((pred == 0) & (y > 0.5)).sum().float()
            precision = tp / (tp + fp).clamp_min(1.0)
            recall = tp / (tp + fn).clamp_min(1.0)
            f1 = 2.0 * precision * recall / (precision + recall).clamp_min(1e-8)
            iou = tp / (tp + fp + fn).clamp_min(1.0)
            metric = iou if args.component_gate_select_metric == "IoU" else f1
            if float(args.component_gate_min_recall) > 0:
                metric = metric - float(args.component_gate_recall_penalty) * torch.relu(
                    torch.tensor(float(args.component_gate_min_recall), device=device) - recall
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


def train_component_gate(args, x: torch.Tensor, y: torch.Tensor, device):
    model = ComponentGateHead(
        len(COMPONENT_GATE_FEATURE_NAMES),
        hidden_dim=args.component_gate_hidden_dim,
        dropout=args.component_gate_dropout,
    ).to(device)
    if x.numel() == 0:
        return model, {"num_components": 0, "positive_ratio": 0.0}
    x = x.to(device)
    y = y.to(device)
    pos = y.sum().clamp_min(1.0)
    neg = (y.numel() - y.sum()).clamp_min(1.0)
    pos_weight = (neg / pos * float(args.component_gate_positive_weight_scale)).clamp(0.5, args.component_gate_max_pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.component_gate_lr, weight_decay=args.component_gate_weight_decay)
    for _ in range(args.component_gate_epochs):
        model.train()
        perm = torch.randperm(x.shape[0], device=device)
        for start in range(0, x.shape[0], args.component_gate_batch_size):
            idx = perm[start:start + args.component_gate_batch_size]
            logits = model(x[idx])
            loss = F.binary_cross_entropy_with_logits(logits, y[idx], pos_weight=pos_weight)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    model.eval()
    with torch.no_grad():
        logits = model(x)
        probs = torch.sigmoid(logits)
        pred = probs >= float(args.component_gate_keep_threshold)
        tp = ((pred > 0) & (y > 0.5)).sum().float()
        fp = ((pred > 0) & (y <= 0.5)).sum().float()
        fn = ((pred == 0) & (y > 0.5)).sum().float()
        precision = tp / (tp + fp).clamp_min(1.0)
        recall = tp / (tp + fn).clamp_min(1.0)
    return model, {
        "num_components": int(y.numel()),
        "positive_ratio": float(y.mean().detach().cpu()),
        "pos_weight": float(pos_weight.detach().cpu()),
        "prob_mean": float(probs.mean().detach().cpu()),
        "prob_min": float(probs.min().detach().cpu()),
        "prob_max": float(probs.max().detach().cpu()),
        "train_precision_at_keep_threshold": float(precision.detach().cpu()),
        "train_recall_at_keep_threshold": float(recall.detach().cpu()),
        "keep_threshold": float(args.component_gate_keep_threshold),
    }


@torch.no_grad()
def component_gate_filter(pred_mask: np.ndarray, score: torch.Tensor, bcd_score: torch.Tensor,
                          score_threshold: float, args, component_model: Optional[nn.Module],
                          device) -> np.ndarray:
    if component_model is None:
        return pred_mask.astype(np.int64)
    pred_bool = pred_mask.astype(bool)
    if not pred_bool.any():
        return pred_mask.astype(np.int64)

    bcd = bcd_score.float().clamp(0, 1)
    bcd_th = component_gate_threshold(bcd, args)
    bcd_th_value = float(bcd_th.detach().cpu())
    bcd_np = bcd.detach().cpu().numpy()
    score_np = score.detach().float().cpu().numpy()
    feature_score_np, feature_score_threshold = component_gate_feature_inputs(
        score_np, bcd_np, float(score_threshold), bcd_th_value, args,
    )

    labels_cc, num = connected_components(pred_bool)
    out = pred_bool.copy()
    for lab in range(1, num + 1):
        region = labels_cc == lab
        area = int(region.sum())
        if area < int(args.component_gate_min_area):
            continue
        if float(args.component_gate_protect_area_ratio) > 0 and (
            float(area) / float(pred_bool.size)
        ) >= float(args.component_gate_protect_area_ratio):
            continue
        if int(args.component_gate_max_area) > 0 and area > int(args.component_gate_max_area):
            continue
        score_values = score_np[region]
        strong_semantic = _region_topk_mean(score_values, args.component_gate_topk_ratio) >= (
            float(score_threshold) * float(args.component_gate_strong_score_scale)
        )
        if bool(args.component_gate_keep_strong_semantic) and strong_semantic:
            continue
        row = component_gate_features(
            region,
            feature_score_np,
            bcd_np,
            feature_score_threshold,
            bcd_th_value,
            topk_ratio=float(args.component_gate_topk_ratio),
        )
        x = torch.tensor(row, dtype=torch.float32, device=device).view(1, -1)
        keep_prob = float(torch.sigmoid(component_model(x)).detach().cpu().reshape(()))
        if keep_prob < float(args.component_gate_keep_threshold):
            out[region] = False
    return out.astype(np.int64)


def bcd_soft_gate_filter(pred_mask: np.ndarray, score: torch.Tensor, bcd_score: torch.Tensor,
                         score_threshold: float, args) -> np.ndarray:
    """Conservatively suppress weak components unsupported by BCD evidence.

    Unlike a hard foreground-ratio cap, this keeps the no-cap adaptive mask as
    the base prediction and only removes components that are weak in both the
    semantic change score and the category-agnostic BCD score. This is intended
    to preserve recall while still reducing obvious hallucinated fragments.
    """
    pred_bool = pred_mask.astype(bool)
    if not pred_bool.any():
        return pred_mask.astype(np.int64)

    bcd = bcd_score.float().clamp(0, 1)
    bcd_threshold = compute_score_threshold(
        bcd,
        mode=args.bcd_soft_gate_threshold_mode,
        percentile=args.bcd_soft_gate_threshold_percentile,
        threshold_abs=_optional_abs_threshold(args.bcd_soft_gate_threshold_abs),
        std_factor=args.bcd_soft_gate_threshold_std_factor,
        threshold_min=args.bcd_soft_gate_threshold_min,
        threshold_max=args.bcd_soft_gate_threshold_max,
        threshold_max_foreground_ratio=0.0,
    )
    bcd_threshold_value = float(bcd_threshold.detach().cpu())
    bcd_np = bcd.detach().cpu().numpy()
    score_np = score.detach().float().cpu().numpy()

    labels, num = connected_components(pred_bool)
    out = pred_bool.copy()
    for lab in range(1, num + 1):
        region = labels == lab
        area = int(region.sum())
        if area < int(args.bcd_soft_gate_min_area):
            continue
        if float(args.bcd_soft_gate_protect_area_ratio) > 0 and (
            float(area) / float(pred_bool.size)
        ) >= float(args.bcd_soft_gate_protect_area_ratio):
            continue
        if int(args.bcd_soft_gate_max_area) > 0 and area > int(args.bcd_soft_gate_max_area):
            continue

        bcd_values = bcd_np[region]
        score_values = score_np[region]
        bcd_overlap = float((bcd_values > bcd_threshold_value).mean())
        bcd_conf = _region_topk_mean(bcd_values, args.bcd_soft_gate_topk_ratio)
        score_conf = _region_topk_mean(score_values, args.bcd_soft_gate_topk_ratio)

        has_bcd_support = (
            bcd_overlap >= float(args.bcd_soft_gate_min_overlap)
            or bcd_conf >= bcd_threshold_value * float(args.bcd_soft_gate_threshold_scale)
        )
        strong_semantic = score_conf >= float(score_threshold) * float(args.bcd_soft_gate_strong_score_scale)
        if not (has_bcd_support or strong_semantic):
            out[region] = False
    return out.astype(np.int64)


def class_budget_weights(posterior, presence_a: torch.Tensor, presence_b: torch.Tensor, args) -> np.ndarray:
    num_classes = int(posterior.posterior_a.shape[0])
    if num_classes <= 0:
        return np.zeros((0,), dtype=np.float32)
    if args.class_budget_mode == "none":
        return np.ones((num_classes,), dtype=np.float32)

    presence = torch.maximum(presence_a.float(), presence_b.float()).clamp(0, 1)
    support = torch.maximum(posterior.posterior_a.float(), posterior.posterior_b.float()).flatten(1).mean(dim=1)
    signal = (presence * support).detach().cpu().numpy().astype(np.float32)
    signal = np.power(np.clip(signal, 0.0, None), float(args.class_budget_power))
    if float(signal.sum()) <= 1e-8:
        weights = np.ones((num_classes,), dtype=np.float32) / float(num_classes)
    else:
        weights = signal / float(signal.sum())
    floor = float(np.clip(args.class_budget_uniform_floor, 0.0, 1.0))
    if floor > 0:
        weights = (1.0 - floor) * weights + floor / float(num_classes)
    return weights.astype(np.float32)


def class_ratio_from_budget(base_ratio: float, class_idx: int, weights: np.ndarray, args) -> float:
    base_ratio = float(base_ratio)
    if base_ratio <= 0 or args.class_budget_mode == "none":
        return base_ratio
    if weights.size == 0:
        return base_ratio
    idx = int(np.clip(class_idx, 0, weights.size - 1))
    ratio = base_ratio * float(args.class_budget_multiplier) * float(weights[idx])
    return float(np.clip(ratio, float(args.ratio_min), float(args.semantic_ratio_max)))


@torch.no_grad()
def evaluate_semantic_targets(args, datasets, items_by_dataset, processor, change_module, device, model,
                              component_model: Optional[nn.Module] = None):
    cfg = _memory_config_from_args(args)
    model.eval()
    if component_model is not None:
        component_model.eval()
    out = OrderedDict()
    for dataset in datasets:
        spec = dataset_spec(dataset)
        class_names = spec["class_names"]
        method_names = ["adaptive_no_cap", "bcd_soft_gate"]
        if bool(args.enable_component_gate):
            method_names.append("cacdd_component_gate")
        method_names.extend(["bcd_mass_ratio", "cacdd_learned_ratio"])
        accumulators = OrderedDict((name, new_accumulator(class_names)) for name in method_names)
        ratio_logs = OrderedDict((name, []) for name in accumulators.keys())
        threshold_logs = OrderedDict((name, []) for name in accumulators.keys())
        for item in tqdm(items_by_dataset[dataset], desc=f"eval-{dataset}", dynamic_ncols=True):
            meta, _, bcd_np, size_hw = bench.predict_change(processor, change_module, item, args, device)
            gt_masks = build_gt_masks(item, size_hw, spec)
            gt_cd = bench.read_binary_mask(item["gt_cd"], size_hw)
            bcd_cm = bench.confusion_binary(gt_cd, bcd_np)
            learned_feature = torch.tensor(
                score_distribution_features(meta["bcd_score"], args),
                dtype=torch.float32,
                device=device,
            ).view(1, -1)
            learned_ratio = float(ratio_from_logit(model(learned_feature), args, learned_feature).detach().cpu().reshape(()))
            learned_ratio = float(np.clip(learned_ratio * float(args.semantic_ratio_scale), float(args.ratio_min), float(args.semantic_ratio_max)))
            mass_ratio = float(np.clip(
                bcd_mass_foreground_ratio(meta["bcd_score"], args) * float(args.semantic_ratio_scale),
                float(args.ratio_min),
                float(args.semantic_ratio_max),
            ))
            ratio_by_method = {
                "adaptive_no_cap": 0.0,
                "bcd_soft_gate": 0.0,
            }
            if bool(args.enable_component_gate):
                ratio_by_method["cacdd_component_gate"] = 0.0
            ratio_by_method["bcd_mass_ratio"] = mass_ratio
            ratio_by_method["cacdd_learned_ratio"] = learned_ratio

            scores_a, presence_a = extract_class_scores(processor, meta["state_a"], spec["prompt_bank"], class_names)
            scores_b, presence_b = extract_class_scores(processor, meta["state_b"], spec["prompt_bank"], class_names)
            posterior = build_posterior_inputs(
                scores_a,
                scores_b,
                presence_a,
                presence_b,
                class_names=class_names,
                confusers=spec["confusers"],
                rho=args.rho,
            )
            budget_weights = class_budget_weights(posterior, presence_a, presence_b, args)

            pred_by_method = {
                name: {cls: np.zeros(size_hw, dtype=np.int64) for cls in class_names}
                for name in accumulators.keys()
            }
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
                for method, ratio in ratio_by_method.items():
                    class_ratio = class_ratio_from_budget(ratio, cls_idx, budget_weights, args)
                    pred_mask, threshold = decode_class_score(score, presence, class_ratio, args)
                    if method == "bcd_soft_gate":
                        pred_mask = bcd_soft_gate_filter(pred_mask, score, meta["bcd_score"], threshold, args)
                    elif method == "cacdd_component_gate":
                        pred_mask = component_gate_filter(
                            pred_mask, score, meta["bcd_score"], threshold, args, component_model, device,
                        )
                    pred_by_method[method][cls_name] = pred_mask
                    threshold_logs[method].append(threshold)
            for method, pred_masks in pred_by_method.items():
                add_prediction(accumulators[method], class_names, gt_masks, pred_masks, bcd_cm)
                ratio_logs[method].append(ratio_by_method[method])
        dataset_out = OrderedDict()
        for method, acc in accumulators.items():
            result = finalize_accumulator(acc)
            result["ratio_summary"] = {
                "mean": float(np.mean(ratio_logs[method])) if ratio_logs[method] else 0.0,
                "min": float(np.min(ratio_logs[method])) if ratio_logs[method] else 0.0,
                "max": float(np.max(ratio_logs[method])) if ratio_logs[method] else 0.0,
            }
            result["threshold_summary"] = {
                "mean": float(np.mean(threshold_logs[method])) if threshold_logs[method] else 0.0,
                "min": float(np.min(threshold_logs[method])) if threshold_logs[method] else 0.0,
                "max": float(np.max(threshold_logs[method])) if threshold_logs[method] else 0.0,
            }
            dataset_out[method] = result
        out[dataset] = dataset_out
    return out


@torch.no_grad()
def evaluate_binary_targets(args, datasets, items_by_dataset, processor, change_module, device, model,
                            component_model: Optional[nn.Module] = None):
    cfg = _memory_config_from_args(args)
    model.eval()
    if component_model is not None:
        component_model.eval()
    out = OrderedDict()
    for dataset in datasets:
        binary_spec = bench.BINARY_DATASETS[dataset]
        accumulators = OrderedDict()
        ratio_logs = OrderedDict()
        threshold_logs = OrderedDict()
        method_names = ["adaptive_no_cap", "bcd_soft_gate"]
        if bool(args.enable_component_gate):
            method_names.append("cacdd_component_gate")
        method_names.extend(["bcd_mass_ratio", "cacdd_learned_ratio"])
        for method in method_names:
            accumulators[method] = {
                "samples": 0,
                "task_confusion": np.zeros((2, 2), dtype=np.int64),
                "bcd_confusion": np.zeros((2, 2), dtype=np.int64),
            }
            ratio_logs[method] = []
            threshold_logs[method] = []

        posterior_specs = binary_spec.get("posterior_specs")
        if posterior_specs is not None:
            class_names = list(posterior_specs.keys())
            prompt_bank = bench.specs_to_prompt_bank(posterior_specs)
            confusers = bench.specs_to_confusers(posterior_specs)
            target_names = list(binary_spec["posterior_targets"])
        else:
            class_names = []
            prompt_bank = {}
            confusers = {}
            target_names = []

        for item in tqdm(items_by_dataset[dataset], desc=f"eval-{dataset}", dynamic_ncols=True):
            meta, _, bcd_np, size_hw = bench.predict_change(processor, change_module, item, args, device)
            gt = bench.read_binary_mask(item["label"], size_hw)
            bcd_cm = bench.confusion_binary(gt, bcd_np)
            learned_feature = torch.tensor(
                score_distribution_features(meta["bcd_score"], args),
                dtype=torch.float32,
                device=device,
            ).view(1, -1)
            learned_ratio = float(ratio_from_logit(model(learned_feature), args, learned_feature).detach().cpu().reshape(()))
            learned_ratio = float(np.clip(learned_ratio * float(args.semantic_ratio_scale), float(args.ratio_min), float(args.semantic_ratio_max)))
            mass_ratio = float(np.clip(
                bcd_mass_foreground_ratio(meta["bcd_score"], args) * float(args.semantic_ratio_scale),
                float(args.ratio_min),
                float(args.semantic_ratio_max),
            ))
            ratio_by_method = {
                "adaptive_no_cap": 0.0,
                "bcd_soft_gate": 0.0,
            }
            if bool(args.enable_component_gate):
                ratio_by_method["cacdd_component_gate"] = 0.0
            ratio_by_method["bcd_mass_ratio"] = mass_ratio
            ratio_by_method["cacdd_learned_ratio"] = learned_ratio

            pred_by_method = OrderedDict()
            if posterior_specs is None:
                for method, ratio in ratio_by_method.items():
                    if method == "adaptive_no_cap":
                        pred_by_method[method] = bcd_np.astype(np.int64)
                    elif method == "bcd_soft_gate":
                        pred_by_method[method] = bcd_np.astype(np.int64)
                    elif method == "cacdd_component_gate":
                        pred_by_method[method] = bcd_np.astype(np.int64)
                    else:
                        pred_by_method[method] = mask_from_ratio(meta["bcd_score"], ratio)
                    threshold_logs[method].append(0.0)
            else:
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
                budget_weights = class_budget_weights(posterior, presence_a, presence_b, args)
                pred_by_method = {
                    method: np.zeros(size_hw, dtype=np.int64)
                    for method in accumulators.keys()
                }
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
                    for method, ratio in ratio_by_method.items():
                        class_ratio = class_ratio_from_budget(ratio, cls_idx, budget_weights, args)
                        pred_mask, threshold = decode_class_score(score, presence, class_ratio, args)
                        if method == "bcd_soft_gate":
                            pred_mask = bcd_soft_gate_filter(pred_mask, score, meta["bcd_score"], threshold, args)
                        elif method == "cacdd_component_gate":
                            pred_mask = component_gate_filter(
                                pred_mask, score, meta["bcd_score"], threshold, args, component_model, device,
                            )
                        pred_by_method[method] |= pred_mask.astype(np.int64)
                        threshold_logs[method].append(threshold)

            for method, pred in pred_by_method.items():
                accumulators[method]["samples"] += 1
                accumulators[method]["bcd_confusion"] += bcd_cm
                accumulators[method]["task_confusion"] += bench.confusion_binary(gt, pred.astype(np.int64))
                ratio_logs[method].append(ratio_by_method[method])

        dataset_out = OrderedDict()
        for method, acc in accumulators.items():
            dataset_out[method] = {
                "samples": int(acc["samples"]),
                "task_confusion": acc["task_confusion"].tolist(),
                "bcd_confusion": acc["bcd_confusion"].tolist(),
                "task": {k: float(v) for k, v in bench.metrics_from_confusion(acc["task_confusion"]).items()},
                "bcd": {k: float(v) for k, v in bench.metrics_from_confusion(acc["bcd_confusion"]).items()},
                "ratio_summary": {
                    "mean": float(np.mean(ratio_logs[method])) if ratio_logs[method] else 0.0,
                    "min": float(np.min(ratio_logs[method])) if ratio_logs[method] else 0.0,
                    "max": float(np.max(ratio_logs[method])) if ratio_logs[method] else 0.0,
                },
                "threshold_summary": {
                    "mean": float(np.mean(threshold_logs[method])) if threshold_logs[method] else 0.0,
                    "min": float(np.min(threshold_logs[method])) if threshold_logs[method] else 0.0,
                    "max": float(np.max(threshold_logs[method])) if threshold_logs[method] else 0.0,
                },
            }
        out[dataset] = dataset_out
    return out


def load_models(args, device):
    change_module, module_type = bench.load_change_module(args, device)
    bpe_path = Path(__file__).resolve().parent / "sam3/assets/bpe_simple_vocab_16e6.txt.gz"
    sam3_model = build_sam3_image_model(
        bpe_path=str(bpe_path),
        checkpoint_path=args.sam3_checkpoint,
        device=str(device),
    )
    processor = Sam3Processor(
        sam3_model,
        confidence_threshold=args.sam3_confidence_threshold,
        device=str(device),
    )
    return processor, change_module, module_type


def main() -> None:
    parser = argparse.ArgumentParser(description="CACDD-sourced adaptive ratio threshold for CogVis OVCD")
    parser.add_argument("--benchmark_root", type=str, default="data/OVCD_Benchmark")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--sam3_checkpoint", type=str, default="weights/sam3.pt")
    parser.add_argument("--output_json", type=str, required=True)
    parser.add_argument("--output_checkpoint", type=str, default=None)
    parser.add_argument("--load_checkpoint", type=str, default=None)
    parser.add_argument("--source_dataset", type=str, default="CACDD")
    parser.add_argument("--target_datasets", type=str, default="SECOND,SCSCD")
    parser.add_argument("--train_limit", type=int, default=16)
    parser.add_argument("--train_offset", type=int, default=0)
    parser.add_argument("--source_val_limit", type=int, default=8)
    parser.add_argument("--source_val_offset", type=int, default=128)
    parser.add_argument("--eval_limit", type=int, default=8)
    parser.add_argument("--eval_offset", type=int, default=128)

    parser.add_argument("--module_type", type=str, default="auto",
                        choices=["auto", "temporal_lora", "temporal_gate_v4", "mdfm"])
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

    parser.add_argument("--bcd_mass_threshold_mode", type=str, default="adaptive",
                        choices=["percentile", "mean_std", "otsu", "otsu_edge", "candidate_otsu", "adaptive"])
    parser.add_argument("--bcd_mass_threshold_percentile", type=float, default=85.0)
    parser.add_argument("--bcd_mass_threshold_abs", type=float, default=None)
    parser.add_argument("--bcd_mass_threshold_std_factor", type=float, default=1.0)
    parser.add_argument("--bcd_mass_threshold_min", type=float, default=0.0)
    parser.add_argument("--bcd_mass_threshold_max", type=float, default=1.0)
    parser.add_argument("--bcd_mass_ratio_scale", type=float, default=1.0)
    parser.add_argument("--bcd_mass_ratio_min", type=float, default=0.002)
    parser.add_argument("--bcd_mass_ratio_max", type=float, default=0.35)
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
    parser.add_argument("--enable_component_gate", action="store_true")
    parser.add_argument("--component_gate_retrain", action="store_true")
    parser.add_argument("--component_gate_candidate_source", type=str, default="bcd",
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
    parser.add_argument("--component_gate_protect_area_ratio", type=float, default=0.02)
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

    parser.add_argument("--ratio_grid", type=str, default="0.002:0.350:0.008")
    parser.add_argument("--ratio_target_metric", type=str, default="IoU", choices=["IoU", "F1"])
    parser.add_argument("--ratio_min_recall", type=float, default=0.35)
    parser.add_argument("--ratio_recall_penalty", type=float, default=1.0)
    parser.add_argument("--ratio_min", type=float, default=0.002)
    parser.add_argument("--ratio_max", type=float, default=0.35)
    parser.add_argument("--ratio_output_mode", type=str, default="absolute",
                        choices=["absolute", "residual_bcd_mass"])
    parser.add_argument("--ratio_residual_max_log_scale", type=float, default=0.35)
    parser.add_argument("--ratio_residual_l1_weight", type=float, default=0.05)
    parser.add_argument("--semantic_ratio_scale", type=float, default=1.0)
    parser.add_argument("--semantic_ratio_max", type=float, default=0.35)
    parser.add_argument("--class_budget_mode", type=str, default="presence_mass",
                        choices=["none", "presence_mass"])
    parser.add_argument("--class_budget_multiplier", type=float, default=1.5)
    parser.add_argument("--class_budget_uniform_floor", type=float, default=0.20)
    parser.add_argument("--class_budget_power", type=float, default=1.0)
    parser.add_argument("--ratio_hidden_dim", type=int, default=64)
    parser.add_argument("--ratio_dropout", type=float, default=0.05)
    parser.add_argument("--ratio_epochs", type=int, default=80)
    parser.add_argument("--ratio_batch_size", type=int, default=16)
    parser.add_argument("--ratio_lr", type=float, default=1e-3)
    parser.add_argument("--ratio_weight_decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))

    device = bench.select_device()
    print(f"Using device: {bench.describe_device(device)}")
    processor, change_module, module_type = load_models(args, device)
    print(f"Loaded change module: {module_type}")

    root = Path(args.benchmark_root)
    source_spec = bench.BINARY_DATASETS[args.source_dataset]
    all_source = bench.list_binary_items(root, source_spec)
    train_items = split_items(all_source, args.train_limit, args.train_offset)
    source_val_items = split_items(all_source, args.source_val_limit, args.source_val_offset)
    print(f"{args.source_dataset}: train {len(train_items)} / val {len(source_val_items)} / all {len(all_source)}")

    train_stats = {}
    source_collect_stats = {}
    ckpt = None
    if args.load_checkpoint:
        ckpt = torch.load(args.load_checkpoint, map_location=device, weights_only=False)
        model = RatioThresholdHead(
            len(RATIO_FEATURE_NAMES),
            hidden_dim=int(ckpt["config"].get("ratio_hidden_dim", args.ratio_hidden_dim)),
            dropout=float(ckpt["config"].get("ratio_dropout", args.ratio_dropout)),
        ).to(device)
        model.load_state_dict(ckpt["ratio_head"])
        model.eval()
        train_stats = dict(ckpt.get("train_stats", {}))
        source_collect_stats = dict(ckpt.get("source_collect_stats", {}))
    else:
        x_train, y_train, source_collect_stats, target_ratios = collect_cacdd_examples(
            args, train_items, processor, change_module, device,
        )
        model, train_stats = train_ratio_head(args, x_train, y_train, device)
        train_stats["target_ratio_values"] = [float(v) for v in target_ratios]
        if args.output_checkpoint:
            torch.save(
                {
                    "ratio_head": model.state_dict(),
                    "config": vars(args),
                    "feature_names": RATIO_FEATURE_NAMES,
                    "train_stats": train_stats,
                    "source_collect_stats": source_collect_stats,
                },
                args.output_checkpoint,
            )
            print(f"Saved checkpoint: {args.output_checkpoint}")

    component_model = None
    component_gate_stats = {}
    component_gate_collect_stats = {}
    if bool(args.enable_component_gate):
        loaded_component_gate = False
        if ckpt is not None and "component_gate_head" in ckpt and not bool(args.component_gate_retrain):
            component_model = ComponentGateHead(
                len(COMPONENT_GATE_FEATURE_NAMES),
                hidden_dim=int(ckpt.get("config", {}).get("component_gate_hidden_dim", args.component_gate_hidden_dim)),
                dropout=float(ckpt.get("config", {}).get("component_gate_dropout", args.component_gate_dropout)),
            ).to(device)
            component_model.load_state_dict(ckpt["component_gate_head"])
            component_model.eval()
            component_gate_stats = dict(ckpt.get("component_gate_stats", {}))
            component_gate_collect_stats = dict(ckpt.get("component_gate_collect_stats", {}))
            if float(args.component_gate_keep_threshold) < 0 and "keep_threshold" in component_gate_stats:
                args.component_gate_keep_threshold = float(component_gate_stats["keep_threshold"])
            loaded_component_gate = True
        if not loaded_component_gate:
            x_gate, y_gate, component_gate_collect_stats = collect_cacdd_component_gate_examples(
                args, train_items, processor, change_module, device,
            )
            component_model, component_gate_stats = train_component_gate(args, x_gate, y_gate, device)
            if float(args.component_gate_keep_threshold) < 0:
                x_gate_val, y_gate_val, val_collect_stats = collect_cacdd_component_gate_examples(
                    args, source_val_items, processor, change_module, device,
                )
                selected_threshold, threshold_stats = select_component_gate_threshold(
                    args,
                    component_model,
                    x_gate_val if x_gate_val.numel() > 0 else x_gate,
                    y_gate_val if y_gate_val.numel() > 0 else y_gate,
                    device,
                )
                args.component_gate_keep_threshold = float(selected_threshold)
                component_gate_stats["keep_threshold"] = float(selected_threshold)
                component_gate_stats["threshold_selection"] = threshold_stats
                component_gate_collect_stats["source_val_for_threshold"] = val_collect_stats
        component_gate_stats["loaded_from_checkpoint"] = bool(loaded_component_gate)

    if args.output_checkpoint:
        checkpoint_payload = {
            "ratio_head": model.state_dict(),
            "config": vars(args),
            "feature_names": RATIO_FEATURE_NAMES,
            "train_stats": train_stats,
            "source_collect_stats": source_collect_stats,
        }
        if component_model is not None:
            checkpoint_payload.update({
                "component_gate_head": component_model.state_dict(),
                "component_gate_feature_names": COMPONENT_GATE_FEATURE_NAMES,
                "component_gate_stats": component_gate_stats,
                "component_gate_collect_stats": component_gate_collect_stats,
            })
        torch.save(checkpoint_payload, args.output_checkpoint)
        print(f"Saved checkpoint: {args.output_checkpoint}")

    source_val_results = evaluate_cacdd_ratio(args, source_val_items, processor, change_module, device, model)

    targets = [x.strip() for x in args.target_datasets.split(",") if x.strip()]
    semantic_targets = []
    binary_targets = []
    semantic_items_by_dataset = {}
    binary_items_by_dataset = {}
    for dataset in targets:
        if dataset in ("SECOND", "SCSCD"):
            all_items = bench.list_semantic_items(root, dataset)
            semantic_items_by_dataset[dataset] = split_items(all_items, args.eval_limit, args.eval_offset)
            semantic_targets.append(dataset)
            print(f"{dataset}: eval {len(semantic_items_by_dataset[dataset])} / all {len(all_items)}")
        elif dataset in bench.BINARY_DATASETS:
            all_items = bench.list_binary_items(root, bench.BINARY_DATASETS[dataset])
            binary_items_by_dataset[dataset] = split_items(all_items, args.eval_limit, args.eval_offset)
            binary_targets.append(dataset)
            print(f"{dataset}: eval {len(binary_items_by_dataset[dataset])} / all {len(all_items)}")
        else:
            raise ValueError(f"Unknown target dataset: {dataset}")
    target_results = OrderedDict()
    if semantic_targets:
        target_results.update(evaluate_semantic_targets(
            args, semantic_targets, semantic_items_by_dataset, processor, change_module, device, model, component_model,
        ))
    if binary_targets:
        target_results.update(evaluate_binary_targets(
            args, binary_targets, binary_items_by_dataset, processor, change_module, device, model, component_model,
        ))

    out = {
        "config": vars(args),
        "module_type": module_type,
        "feature_names": RATIO_FEATURE_NAMES,
        "component_gate_feature_names": COMPONENT_GATE_FEATURE_NAMES if bool(args.enable_component_gate) else [],
        "train_stats": train_stats,
        "source_collect_stats": source_collect_stats,
        "component_gate_stats": component_gate_stats,
        "component_gate_collect_stats": component_gate_collect_stats,
        "source_val_results": source_val_results,
        "target_results": target_results,
    }
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved metrics: {args.output_json}")


if __name__ == "__main__":
    main()
