"""Small-grid search for memory-rectified CogVis OVCD readout.

The script is designed as an independent experiment entrypoint. It does not
modify the current best evaluator. For each image pair, it runs SAM3 and the
CogVis change prior once, caches posterior tensors in memory, and evaluates
many memory-readout hyperparameter settings on top of the same forward pass.
"""

from __future__ import annotations

import argparse
import json
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
from tqdm import tqdm

import eval_ovcd_benchmark as bench
from memory_ovcd_readout import MemoryReadoutConfig, memory_rectified_score
from posterior_ovcd import (
    apply_bcd_soft_gate,
    build_posterior_inputs,
    compute_score_threshold,
    decode_ovcd_score,
    extract_class_scores,
)
from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor


DEFAULT_CLASS_THRESHOLDS = (
    "SCSCD:Bareland=0.04,SCSCD:Water=0.20,SCSCD:Building=0.20,"
    "SCSCD:Structure=0.02,SCSCD:Farmland=0.08,SCSCD:Vegetation=0.08,SCSCD:Road=0.30,"
    "SECOND:Building=0.30,SECOND:Tree=0.06,SECOND:Water=0.15,"
    "SECOND:Grass=0.06,SECOND:Ground=0.06,SECOND:Playground=0.20"
)


def parse_float_list(value: str) -> List[float]:
    return [float(v.strip()) for v in value.split(",") if v.strip()]


def dataset_spec(dataset: str):
    if dataset == "SECOND":
        return {
            "class_names": bench.second_class_names(),
            "prompt_bank": bench.second_prompt_bank(),
            "confusers": bench.second_confusers(),
            "gt_ids": bench.second_gt_ids(),
            "cd_masked": False,
        }
    if dataset == "SCSCD":
        specs = bench.SCSCD_POSTERIOR_SPECS
        return {
            "class_names": list(specs.keys()),
            "prompt_bank": bench.specs_to_prompt_bank(specs),
            "confusers": bench.specs_to_confusers(specs),
            "gt_ids": bench.specs_to_gt_ids(specs),
            "cd_masked": True,
        }
    raise ValueError(f"Unsupported dataset for memory search: {dataset}")


def make_memory_configs(args) -> OrderedDict[str, MemoryReadoutConfig]:
    configs: OrderedDict[str, MemoryReadoutConfig] = OrderedDict()
    for dw in parse_float_list(args.memory_delta_weights):
        for tw in parse_float_list(args.memory_transition_weights):
            for floor in parse_float_list(args.memory_bcd_floors):
                for suppress in parse_float_list(args.memory_stable_suppresses):
                    for boost in parse_float_list(args.memory_support_boosts):
                        name = (
                            f"mem_dw{dw:.2f}_tw{tw:.2f}_floor{floor:.2f}"
                            f"_sup{suppress:.2f}_boost{boost:.2f}"
                        )
                        configs[name] = MemoryReadoutConfig(
                            delta_weight=dw,
                            transition_weight=tw,
                            bcd_alpha=args.memory_bcd_alpha,
                            bcd_floor=floor,
                            stable_suppress=suppress,
                            stable_power=args.memory_stable_power,
                            support_boost=boost,
                            presence_delta_weight=args.memory_presence_delta_weight,
                            robust_normalize=args.memory_robust_normalize,
                            q_low=args.memory_q_low,
                            q_high=args.memory_q_high,
                        )
    return configs


def new_accumulator(class_names: Sequence[str]) -> dict:
    return {
        "samples": 0,
        "class_confusions": {name: np.zeros((2, 2), dtype=np.int64) for name in class_names},
        "bcd_confusion": np.zeros((2, 2), dtype=np.int64),
    }


def add_prediction(
    acc: dict,
    class_names: Sequence[str],
    gt_masks: Dict[str, np.ndarray],
    pred_masks: Dict[str, np.ndarray],
    bcd_cm: np.ndarray,
):
    acc["samples"] += 1
    acc["bcd_confusion"] += bcd_cm
    for cls_name in class_names:
        acc["class_confusions"][cls_name] += bench.confusion_binary(
            gt_masks[cls_name].astype(np.int64),
            pred_masks[cls_name].astype(np.int64),
        )


def finalize_accumulator(acc: dict) -> dict:
    class_metrics = OrderedDict()
    all_metrics = {k: [] for k in ["IoU", "F1", "Pre", "Recall", "OA", "Kappa"]}
    for cls_name, cm in acc["class_confusions"].items():
        metrics = bench.metrics_from_confusion(cm)
        class_metrics[cls_name] = {k: float(v) for k, v in metrics.items()}
        for key, value in metrics.items():
            all_metrics[key].append(value)
    average = {k: float(np.mean(v)) for k, v in all_metrics.items()}
    return {
        "samples": int(acc["samples"]),
        "average": average,
        "bcd": {k: float(v) for k, v in bench.metrics_from_confusion(acc["bcd_confusion"]).items()},
        "class_metrics": class_metrics,
        "class_confusions": {k: v.tolist() for k, v in acc["class_confusions"].items()},
        "bcd_confusion": acc["bcd_confusion"].tolist(),
    }


def build_gt_masks(item: dict, size_hw, spec: dict) -> Dict[str, np.ndarray]:
    gt_t1 = bench.read_label(item["gt_t1"], size_hw)
    gt_t2 = bench.read_label(item["gt_t2"], size_hw)
    gt_cd = bench.read_binary_mask(item["gt_cd"], size_hw).astype(bool)
    out = {}
    for cls_name, ids in spec["gt_ids"].items():
        gt = np.zeros(size_hw, dtype=bool)
        for gid in ids:
            gt |= (gt_t1 == gid) | (gt_t2 == gid)
        if spec["cd_masked"]:
            gt &= gt_cd
        out[cls_name] = gt
    return out


def _optional_abs_threshold(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    value = float(value)
    if value < 0:
        return None
    return value


def bcd_mass_foreground_ratio(bcd_score: torch.Tensor, args) -> float:
    """Estimate a label-free per-image foreground cap from the BCD prior."""
    bcd = bcd_score.float().clamp(0, 1)
    bcd_threshold = compute_score_threshold(
        bcd,
        mode=args.bcd_mass_threshold_mode,
        percentile=args.bcd_mass_threshold_percentile,
        threshold_abs=_optional_abs_threshold(args.bcd_mass_threshold_abs),
        std_factor=args.bcd_mass_threshold_std_factor,
        threshold_min=args.bcd_mass_threshold_min,
        threshold_max=args.bcd_mass_threshold_max,
        threshold_max_foreground_ratio=0.0,
    )
    ratio = float((bcd > bcd_threshold).float().mean().detach().cpu())
    ratio *= float(args.bcd_mass_ratio_scale)
    if args.threshold_max_foreground_ratio and args.threshold_max_foreground_ratio > 0:
        ratio = min(ratio, float(args.threshold_max_foreground_ratio))
    return float(np.clip(ratio, float(args.bcd_mass_ratio_min), float(args.bcd_mass_ratio_max)))


def connected_components(mask: np.ndarray):
    try:
        import cv2

        num, labels = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
        return labels.astype(np.int32), int(num - 1)
    except Exception:
        try:
            from scipy import ndimage

            labels, num = ndimage.label(mask.astype(bool), structure=np.ones((3, 3), dtype=np.uint8))
            return labels.astype(np.int32), int(num)
        except Exception:
            return mask.astype(np.int32), int(mask.any())


def region_stat(values: np.ndarray, stat: str, topk_ratio: float) -> float:
    if values.size == 0:
        return 0.0
    if stat == "max":
        return float(values.max())
    if stat == "topk_mean":
        k = max(1, int(values.size * float(topk_ratio)))
        return float(np.partition(values, -k)[-k:].mean())
    return float(values.mean())


def bcd_component_filter(pred_mask: np.ndarray, score: torch.Tensor, bcd_score: torch.Tensor, args) -> np.ndarray:
    """Filter predicted components using label-free BCD support consistency."""
    pred_bool = pred_mask.astype(bool)
    if not pred_bool.any():
        return pred_mask

    bcd = bcd_score.float().clamp(0, 1)
    bcd_threshold = compute_score_threshold(
        bcd,
        mode=args.bcd_component_threshold_mode,
        percentile=args.bcd_component_threshold_percentile,
        threshold_abs=_optional_abs_threshold(args.bcd_component_threshold_abs),
        std_factor=args.bcd_component_threshold_std_factor,
        threshold_min=args.bcd_component_threshold_min,
        threshold_max=args.bcd_component_threshold_max,
        threshold_max_foreground_ratio=0.0,
    )
    bcd_threshold_value = float(bcd_threshold.detach().cpu())
    bcd_np = bcd.detach().cpu().numpy()
    score_np = score.detach().float().cpu().numpy()

    labels, num = connected_components(pred_bool)
    out = np.zeros_like(pred_bool, dtype=np.uint8)
    for lab in range(1, num + 1):
        region = labels == lab
        area = int(region.sum())
        if area < int(args.bcd_component_min_area):
            continue
        if args.bcd_component_max_area > 0 and area > int(args.bcd_component_max_area):
            continue

        bcd_values = bcd_np[region]
        score_values = score_np[region]
        bcd_support_ratio = float((bcd_values > bcd_threshold_value).mean())
        bcd_conf = region_stat(
            bcd_values,
            args.bcd_component_stat,
            args.bcd_component_topk_ratio,
        )
        score_conf = region_stat(
            score_values,
            args.bcd_component_score_stat,
            args.bcd_component_topk_ratio,
        )

        keep_by_overlap = bcd_support_ratio >= float(args.bcd_component_min_overlap)
        keep_by_conf = bcd_conf >= bcd_threshold_value * float(args.bcd_component_threshold_scale)
        keep_by_score = score_conf >= float(args.bcd_component_min_score)
        if (keep_by_overlap or keep_by_conf) and keep_by_score:
            out[region] = 1
    return out.astype(np.int64)


def decode_score_map(
    score,
    args,
    dataset: str,
    cls_name: str,
    class_thresholds,
    image_mean=None,
    bcd_score=None,
    presence=None,
    use_bcd_mass: bool = False,
    use_bcd_component: bool = False,
):
    threshold_abs = bench.threshold_for(class_thresholds, dataset, cls_name, args.threshold_abs)
    max_foreground_ratio = args.threshold_max_foreground_ratio
    if use_bcd_mass and bcd_score is not None:
        max_foreground_ratio = bcd_mass_foreground_ratio(bcd_score, args)
    pred_mask, _ = decode_ovcd_score(
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
    pred_np = pred_mask.detach().cpu().numpy().astype(np.int64)
    if use_bcd_component and bcd_score is not None:
        pred_np = bcd_component_filter(pred_np, score, bcd_score, args)
    return pred_np


def evaluate_dataset(args, dataset: str, processor, change_module, device) -> dict:
    root = Path(args.benchmark_root)
    items = bench.split_items(
        bench.list_semantic_items(root, dataset),
        chunk_index=0,
        num_chunks=1,
        limit=args.limit,
    )
    spec = dataset_spec(dataset)
    class_names = spec["class_names"]
    class_thresholds = bench.parse_class_thresholds(args.class_thresholds)
    memory_configs = make_memory_configs(args)
    accumulators = OrderedDict([("original", new_accumulator(class_names))])
    for name in memory_configs:
        accumulators[name] = new_accumulator(class_names)

    for item in tqdm(items, desc=f"search-{dataset}", dynamic_ncols=True):
        meta, _, bcd_np, size_hw = bench.predict_change(processor, change_module, item, args, device)
        gt_cd = bench.read_binary_mask(item["gt_cd"], size_hw)
        bcd_cm = bench.confusion_binary(gt_cd, bcd_np)
        gt_masks = build_gt_masks(item, size_hw, spec)

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

        original_preds = {}
        for cls_idx, cls_name in enumerate(class_names):
            score = posterior.delta[cls_idx]
            if not args.disable_bcd_gate:
                score = apply_bcd_soft_gate(
                    score,
                    meta["bcd_score"],
                    alpha=args.bcd_alpha,
                    floor=args.bcd_floor,
                )
            presence = torch.maximum(presence_a[cls_idx], presence_b[cls_idx])
            original_preds[cls_name] = decode_score_map(
                score,
                args,
                dataset,
                cls_name,
                class_thresholds,
                bcd_score=None if args.disable_bcd_gate else meta["bcd_score"],
                presence=presence,
                use_bcd_mass=args.decode_calibration in ("bcd_mass", "bcd_mass_component") and args.calibration_scope in ("all", "original"),
                use_bcd_component=args.decode_calibration in ("bcd_component", "bcd_mass_component") and args.calibration_scope in ("all", "original"),
            )
        add_prediction(accumulators["original"], class_names, gt_masks, original_preds, bcd_cm)

        for cfg_name, cfg in memory_configs.items():
            memory_preds = {}
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
                memory_preds[cls_name] = decode_score_map(
                    score,
                    args,
                    dataset,
                    cls_name,
                    class_thresholds,
                    bcd_score=None if args.disable_bcd_gate else meta["bcd_score"],
                    presence=presence,
                    use_bcd_mass=args.decode_calibration in ("bcd_mass", "bcd_mass_component") and args.calibration_scope in ("all", "memory"),
                    use_bcd_component=args.decode_calibration in ("bcd_component", "bcd_mass_component") and args.calibration_scope in ("all", "memory"),
                )
            add_prediction(accumulators[cfg_name], class_names, gt_masks, memory_preds, bcd_cm)

    results = OrderedDict((name, finalize_accumulator(acc)) for name, acc in accumulators.items())
    ranked = sorted(results.items(), key=lambda kv: kv[1]["average"]["IoU"], reverse=True)
    return {
        "dataset": dataset,
        "num_items": len(items),
        "ranked": [
            {
                "name": name,
                "mIoU": metrics["average"]["IoU"],
                "mF1": metrics["average"]["F1"],
                "mPre": metrics["average"]["Pre"],
                "mRecall": metrics["average"]["Recall"],
                "mKappa": metrics["average"]["Kappa"],
            }
            for name, metrics in ranked
        ],
        "results": results,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Search memory-rectified OVCD readout settings")
    parser.add_argument("--benchmark_root", type=str, default="data/OVCD_Benchmark")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--sam3_checkpoint", type=str, default="weights/sam3.pt")
    parser.add_argument("--datasets", type=str, default="SECOND,SCSCD")
    parser.add_argument("--limit", type=int, default=32)
    parser.add_argument("--output_json", type=str, required=True)
    parser.add_argument("--class_thresholds", type=str, default=DEFAULT_CLASS_THRESHOLDS)

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
    parser.add_argument("--bcd_alpha", type=float, default=1.0)
    parser.add_argument("--bcd_floor", type=float, default=0.1)
    parser.add_argument("--threshold_mode", type=str, default="percentile",
                        choices=["percentile", "mean_std", "otsu", "otsu_edge", "candidate_otsu", "adaptive"])
    parser.add_argument("--threshold_percentile", type=float, default=85.0)
    parser.add_argument("--threshold_abs", type=float, default=None)
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
    parser.add_argument("--decode_calibration", type=str, default="none",
                        choices=["none", "bcd_mass", "bcd_component", "bcd_mass_component"])
    parser.add_argument("--calibration_scope", type=str, default="all", choices=["all", "original", "memory"])
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
    parser.add_argument("--bcd_component_threshold_mode", type=str, default="adaptive",
                        choices=["percentile", "mean_std", "otsu", "otsu_edge", "candidate_otsu", "adaptive"])
    parser.add_argument("--bcd_component_threshold_percentile", type=float, default=85.0)
    parser.add_argument("--bcd_component_threshold_abs", type=float, default=None)
    parser.add_argument("--bcd_component_threshold_std_factor", type=float, default=1.0)
    parser.add_argument("--bcd_component_threshold_min", type=float, default=0.0)
    parser.add_argument("--bcd_component_threshold_max", type=float, default=1.0)
    parser.add_argument("--bcd_component_stat", type=str, default="topk_mean", choices=["mean", "max", "topk_mean"])
    parser.add_argument("--bcd_component_score_stat", type=str, default="topk_mean", choices=["mean", "max", "topk_mean"])
    parser.add_argument("--bcd_component_topk_ratio", type=float, default=0.2)
    parser.add_argument("--bcd_component_threshold_scale", type=float, default=0.65)
    parser.add_argument("--bcd_component_min_overlap", type=float, default=0.03)
    parser.add_argument("--bcd_component_min_score", type=float, default=0.01)
    parser.add_argument("--bcd_component_min_area", type=int, default=4)
    parser.add_argument("--bcd_component_max_area", type=int, default=262144)
    parser.add_argument("--region_mode", type=str, default="component",
                        choices=["none", "component", "slic", "acf_component", "acf_region"])
    parser.add_argument("--region_score", type=str, default="mean", choices=["mean", "max", "topk_mean"])
    parser.add_argument("--region_topk_ratio", type=float, default=0.2)
    parser.add_argument("--min_region_area", type=int, default=4)
    parser.add_argument("--max_region_area", type=int, default=262144)
    parser.add_argument("--slic_segments", type=int, default=512)
    parser.add_argument("--bcd_score_mode", type=str, default="head",
                        choices=["auto", "head", "hybrid", "fused", "residual", "feature"])
    parser.add_argument("--bcd_threshold_mode", type=str, default="percentile",
                        choices=["percentile", "mean_std", "otsu", "otsu_edge", "candidate_otsu", "adaptive"])
    parser.add_argument("--bcd_threshold_percentile", type=float, default=85.0)
    parser.add_argument("--bcd_threshold_abs", type=float, default=0.65)
    parser.add_argument("--bcd_threshold_std_factor", type=float, default=1.0)
    parser.add_argument("--acf_conf_percentile", type=float, default=50.0)
    parser.add_argument("--acf_conf_lambda", type=float, default=1.0)
    parser.add_argument("--acf_conf_min", type=float, default=0.0)
    parser.add_argument("--acf_conf_max", type=float, default=1.0)
    parser.add_argument("--acf_cv_max", type=float, default=2.0)

    parser.add_argument("--memory_delta_weights", type=str, default="0.55,0.65,0.75")
    parser.add_argument("--memory_transition_weights", type=str, default="0.25,0.35")
    parser.add_argument("--memory_bcd_floors", type=str, default="0.10,0.15")
    parser.add_argument("--memory_stable_suppresses", type=str, default="0.25,0.45,0.65")
    parser.add_argument("--memory_support_boosts", type=str, default="0.00,0.10")
    parser.add_argument("--memory_bcd_alpha", type=float, default=1.0)
    parser.add_argument("--memory_stable_power", type=float, default=1.0)
    parser.add_argument("--memory_presence_delta_weight", type=float, default=0.15)
    parser.add_argument("--memory_robust_normalize", action="store_true")
    parser.add_argument("--memory_q_low", type=float, default=0.02)
    parser.add_argument("--memory_q_high", type=float, default=0.98)
    return parser.parse_args()


def main():
    args = parse_args()
    device = bench.select_device()
    print(f"Using device: {bench.describe_device(device)}")
    change_module, module_type = bench.load_change_module(args, device)
    print(f"Loaded change module: {module_type}")

    bpe_path = Path(__file__).resolve().parent / "sam3/assets/bpe_simple_vocab_16e6.txt.gz"
    model = build_sam3_image_model(
        bpe_path=str(bpe_path),
        checkpoint_path=args.sam3_checkpoint,
        device=str(device),
    )
    processor = Sam3Processor(model, confidence_threshold=args.sam3_confidence_threshold, device=str(device))

    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    out = {
        "config": vars(args),
        "module_type": module_type,
        "datasets": OrderedDict(),
    }
    for dataset in datasets:
        out["datasets"][dataset] = evaluate_dataset(args, dataset, processor, change_module, device)

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump(out, f, indent=2)

    print("\n## Memory readout search summary")
    for dataset, data in out["datasets"].items():
        print(f"[{dataset}] samples={data['num_items']}")
        for row in data["ranked"][:8]:
            print(
                f"  {row['name']:<55} "
                f"mIoU={row['mIoU']*100:.2f} mF1={row['mF1']*100:.2f} "
                f"mPre={row['mPre']*100:.2f} mRec={row['mRecall']*100:.2f} "
                f"K={row['mKappa']:.4f}"
            )
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
