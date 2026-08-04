"""Evaluate the current CMCT+component-gate OVCD model on xBD BDA.

This script keeps the strict OVCD model fixed and adds an xBD adapter:
building localization is produced from SAM3 building support plus the
CMCT+Gate damage candidate, while damage severity is assigned with zero-shot
SAM3 text prompts.  The primary BDA score follows the ChangeMamba/xBD-style
0.3 localization F1 + 0.7 harmonic damage F1 protocol.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

import eval_ovcd_benchmark as bench  # noqa: E402
import train_cacdd_ratio_threshold_component_gate as base  # noqa: E402
import train_clip_memory_threshold as cmct  # noqa: E402
from clip_memory_threshold import ClipFeatureExtractor, ClipPairEmbedding  # noqa: E402
from memory_ovcd_readout import memory_rectified_score  # noqa: E402
from posterior_ovcd import build_posterior_inputs, competitive_posteriors, extract_class_scores  # noqa: E402


DEFAULT_REPO = Path(
    os.environ.get("COGVIS_ROOT", str(Path(__file__).resolve().parents[2]))
).expanduser()
DEFAULT_CKPT_DIR = Path(
    os.environ.get("COGVIS_CHECKPOINT_DIR", str(DEFAULT_REPO / "weights"))
).expanduser()
DEFAULT_XBD_ROOT = Path(
    os.environ.get("COGVIS_XBD_ROOT", str(DEFAULT_REPO / "data" / "xBD" / "test"))
).expanduser()

XBD_CLASS_NAMES = OrderedDict([
    (0, "background"),
    (1, "no-damage"),
    (2, "minor-damage"),
    (3, "major-damage"),
    (4, "destroyed"),
])

DAMAGE_PROMPTS = OrderedDict([
    ("no-damage", [
        "undamaged building",
        "intact building",
        "building with no damage",
    ]),
    ("minor-damage", [
        "minor damaged building",
        "slightly damaged building",
        "building with minor damage",
    ]),
    ("major-damage", [
        "major damaged building",
        "heavily damaged building",
        "severely damaged building",
    ]),
    ("destroyed", [
        "destroyed building",
        "collapsed building",
        "building rubble",
    ]),
])

BUILDING_PROMPTS = OrderedDict([
    ("Building", ["building", "roof", "house"]),
])

DAMAGE_CONFUSERS = {
    name: [other for other in DAMAGE_PROMPTS.keys() if other != name]
    for name in DAMAGE_PROMPTS.keys()
}

TARGET_VALUE_MAP = {
    "background": 0,
    "no-damage": 1,
    "no damage": 1,
    "minor-damage": 2,
    "minor damage": 2,
    "major-damage": 3,
    "major damage": 3,
    "destroyed": 4,
}

IGNORE_INDEX = 255
IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff")


def _set_parser_default(parser: argparse.ArgumentParser, dest: str, value, required: Optional[bool] = None) -> None:
    for action in parser._actions:
        if action.dest == dest:
            action.default = value
            if required is not None:
                action.required = required
            return


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Zero-shot xBD BDA evaluation for the current CMCT+Gate OVCD model.",
    )
    cmct.add_common_args(parser)
    cmct.add_cmct_args(parser)

    parser.add_argument("--xbd_root", type=str, default=str(DEFAULT_XBD_ROOT))
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--num_chunks", type=int, default=1)
    parser.add_argument("--chunk_index", type=int, default=0)
    parser.add_argument("--building_support_threshold", type=float, default=0.10)
    parser.add_argument("--damage_confidence_threshold", type=float, default=0.10)
    parser.add_argument("--serious_damage_threshold", type=float, default=0.20)
    parser.add_argument("--tile_size", type=int, default=512)
    parser.add_argument("--tile_stride", type=int, default=512)
    parser.add_argument("--save_vis", action="store_true", default=True)
    parser.add_argument("--no_save_vis", action="store_false", dest="save_vis")
    parser.add_argument("--save_pred_clf", action="store_true", default=True)
    parser.add_argument("--no_save_pred_clf", action="store_false", dest="save_pred_clf")
    parser.add_argument("--merge_glob", type=str, default=None)
    parser.add_argument("--disable_component_gate", action="store_true")

    _set_parser_default(parser, "checkpoint", str(DEFAULT_CKPT_DIR / "final.pt"), required=False)
    _set_parser_default(parser, "sam3_checkpoint", str(DEFAULT_REPO / "weights/sam3/sam3.pt"))
    _set_parser_default(parser, "output_json", str(DEFAULT_REPO / "outputs/xbd_bda_cmct/smoke/metrics.json"), required=False)
    _set_parser_default(parser, "load_checkpoint", str(DEFAULT_CKPT_DIR / "cmct_fullcacdd_prec35_floor075_train4968_val4968_epoch100.pt"))
    _set_parser_default(parser, "target_datasets", "xBD")
    _set_parser_default(parser, "train_limit", 0)
    _set_parser_default(parser, "source_val_limit", 0)
    _set_parser_default(parser, "eval_limit", 2)
    _set_parser_default(parser, "eval_offset", 0)
    _set_parser_default(parser, "bcd_score_mode", "head")
    _set_parser_default(parser, "cmct_adaptive_floor_scale", 0.75)
    _set_parser_default(parser, "enable_component_gate", True)
    _set_parser_default(
        parser,
        "component_gate_checkpoint",
        str(DEFAULT_CKPT_DIR / "cmct_component_gate_fullcacdd_semproxy_train4968_val4968_epoch100.pt"),
    )
    _set_parser_default(parser, "component_gate_keep_threshold", 0.60)
    return parser


def jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return value


def list_xbd_items(root: Path) -> List[dict]:
    image_dir = root / "images"
    target_dir = root / "targets"
    label_dir = root / "labels"
    pre_paths = []
    for ext in IMAGE_EXTS:
        pre_paths.extend(image_dir.glob(f"*_pre_disaster{ext}"))
    items = []
    for pre in sorted(pre_paths):
        stem = pre.stem.replace("_pre_disaster", "")
        post = None
        for ext in IMAGE_EXTS:
            candidate = image_dir / f"{stem}_post_disaster{ext}"
            if candidate.exists():
                post = candidate
                break
        if post is None:
            continue
        target = None
        for ext in IMAGE_EXTS:
            candidate = target_dir / f"{stem}_post_disaster_target{ext}"
            if candidate.exists():
                target = candidate
                break
        post_label = label_dir / f"{stem}_post_disaster.json"
        pre_label = label_dir / f"{stem}_pre_disaster.json"
        items.append({
            "name": stem,
            "a": str(pre),
            "b": str(post),
            "target": str(target) if target is not None else None,
            "post_label": str(post_label) if post_label.exists() else None,
            "pre_label": str(pre_label) if pre_label.exists() else None,
        })
    return items


def split_items(items: Sequence[dict], limit: int, offset: int, num_chunks: int, chunk_index: int) -> List[dict]:
    selected = list(items)
    if int(offset) > 0:
        selected = selected[int(offset):]
    if int(limit) > 0:
        selected = selected[: int(limit)]
    num_chunks = max(int(num_chunks), 1)
    chunk_index = int(chunk_index)
    if num_chunks > 1:
        selected = [item for idx, item in enumerate(selected) if idx % num_chunks == chunk_index]
    return selected


def _target_values_to_canonical(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.ndim == 3:
        arr = arr[..., 0]
    out = arr.astype(np.uint8, copy=True)
    invalid = ~np.isin(out, np.array([0, 1, 2, 3, 4], dtype=np.uint8))
    out[invalid] = IGNORE_INDEX
    return out


def _parse_wkt_polygon(wkt: str) -> List[List[Tuple[float, float]]]:
    if not wkt:
        return []
    rings = []
    for body in re.findall(r"\(\((.*?)\)\)", str(wkt)):
        coords = []
        for pair in body.split(","):
            nums = pair.strip().split()
            if len(nums) < 2:
                continue
            try:
                coords.append((float(nums[0]), float(nums[1])))
            except ValueError:
                continue
        if len(coords) >= 3:
            rings.append(coords)
    return rings


def rasterize_xbd_json(label_path: Path, size_hw: Tuple[int, int]) -> np.ndarray:
    h, w = size_hw
    target = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(target)
    data = json.loads(label_path.read_text())
    features = data.get("features", {}).get("xy", [])
    for feature in features:
        props = feature.get("properties", {})
        subtype = str(props.get("subtype", "")).lower().strip()
        value = TARGET_VALUE_MAP.get(subtype, IGNORE_INDEX)
        wkt = feature.get("wkt", "")
        for ring in _parse_wkt_polygon(wkt):
            draw.polygon(ring, fill=int(value))
    return np.array(target, dtype=np.uint8)


def read_gt_mask(item: dict, size_hw: Tuple[int, int]) -> np.ndarray:
    target = item.get("target")
    if target:
        arr = np.array(Image.open(target))
        arr = _target_values_to_canonical(arr)
    elif item.get("post_label"):
        arr = rasterize_xbd_json(Path(item["post_label"]), size_hw)
    else:
        raise FileNotFoundError(f"No target or post label for {item['name']}")
    if arr.shape[:2] != tuple(size_hw):
        arr_img = Image.fromarray(arr)
        arr = np.array(arr_img.resize((size_hw[1], size_hw[0]), resample=Image.Resampling.NEAREST))
    return arr.astype(np.uint8)


def colorize_mask(mask: np.ndarray) -> np.ndarray:
    colors = np.array([
        [0, 0, 0],
        [45, 170, 90],
        [245, 210, 70],
        [235, 130, 40],
        [210, 45, 45],
        [180, 0, 180],
    ], dtype=np.uint8)
    out = np.zeros((*mask.shape, 3), dtype=np.uint8)
    safe = mask.copy()
    safe[safe == IGNORE_INDEX] = 5
    safe = np.clip(safe, 0, 5)
    out[:] = colors[safe]
    return out


def _l2_normalize_np(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    norm = float(np.linalg.norm(x))
    if norm < eps:
        return np.zeros_like(x, dtype=np.float32)
    return (x / norm).astype(np.float32)


def encode_pair_from_pil(clipper: ClipFeatureExtractor, img_a: Image.Image, img_b: Image.Image) -> ClipPairEmbedding:
    emb_a = clipper.encode_pil(img_a.convert("RGB"))
    emb_b = clipper.encode_pil(img_b.convert("RGB"))
    scene = _l2_normalize_np(0.5 * (emb_a + emb_b))
    delta = _l2_normalize_np(np.abs(emb_b - emb_a))
    return ClipPairEmbedding(scene=scene, delta=delta)


@torch.no_grad()
def predict_change_from_pil(processor, change_module, img_a: Image.Image, img_b: Image.Image, args, device):
    img_a = img_a.convert("RGB")
    img_b = img_b.convert("RGB")
    h, w = img_a.size[1], img_a.size[0]
    state_a = processor.set_image(img_a)
    state_b = processor.set_image(img_b)
    fpn_a = [f.to(device) for f in state_a["backbone_out"]["backbone_fpn"]]
    fpn_b = [f.to(device) for f in state_b["backbone_out"]["backbone_fpn"]]
    outputs = change_module(fpn_a, fpn_b, output_size=(h, w))
    bcd, score, threshold = bench.compute_bcd(
        outputs["change_fpn"],
        fpn_a,
        fpn_b,
        h,
        w,
        threshold_percentile=args.bcd_threshold_percentile,
        threshold_abs=args.bcd_threshold_abs,
        score_mode=args.bcd_score_mode,
        threshold_mode=args.bcd_threshold_mode,
        threshold_std_factor=args.bcd_threshold_std_factor,
        change_logits=outputs.get("change_logits"),
        threshold_edge_weight=args.threshold_edge_weight,
        threshold_edge_percentile=args.threshold_edge_percentile,
        threshold_edge_dilate=args.threshold_edge_dilate,
        threshold_edge_min_pixels=args.threshold_edge_min_pixels,
        threshold_min=args.threshold_min,
        threshold_max=args.threshold_max,
        threshold_scale=args.threshold_scale,
        threshold_bias=args.threshold_bias,
        threshold_max_foreground_ratio=args.threshold_max_foreground_ratio,
        threshold_candidate_percentile=args.threshold_candidate_percentile,
    )
    meta = {
        "state_a": state_a,
        "state_b": state_b,
        "bcd_score": score,
        "threshold": float(threshold.detach().cpu()),
        "score_mean": float(score.float().mean().detach().cpu()),
    }
    return meta, bcd, bcd.detach().cpu().numpy().astype(np.int64), (h, w)


def confusion_multiclass(gt: np.ndarray, pred: np.ndarray, num_classes: int, ignore_index: int = IGNORE_INDEX) -> np.ndarray:
    mask = (gt != ignore_index) & (gt >= 0) & (gt < num_classes) & (pred >= 0) & (pred < num_classes)
    if not mask.any():
        return np.zeros((num_classes, num_classes), dtype=np.int64)
    label = num_classes * gt[mask].astype(np.int64) + pred[mask].astype(np.int64)
    return np.bincount(label, minlength=num_classes * num_classes).reshape(num_classes, num_classes)


def binary_confusion(gt: np.ndarray, pred: np.ndarray, valid: np.ndarray) -> np.ndarray:
    gt_bin = (gt > 0) & valid
    pred_bin = (pred > 0) & valid
    label = (2 * gt_bin.astype(np.int64) + pred_bin.astype(np.int64)).reshape(-1)
    return np.bincount(label, minlength=4).reshape(2, 2)


def class_metrics_from_confusion(cm: np.ndarray) -> Dict[str, object]:
    cm = np.asarray(cm, dtype=np.float64)
    eps = 1e-12
    per_class = OrderedDict()
    ious = []
    f1s = []
    precisions = []
    recalls = []
    for idx in range(cm.shape[0]):
        tp = cm[idx, idx]
        fp = cm[:, idx].sum() - tp
        fn = cm[idx, :].sum() - tp
        precision = tp / (tp + fp + eps)
        recall = tp / (tp + fn + eps)
        f1 = 2.0 * precision * recall / (precision + recall + eps)
        iou = tp / (tp + fp + fn + eps)
        name = XBD_CLASS_NAMES.get(idx, f"class_{idx}")
        per_class[name] = {
            "IoU": float(iou),
            "F1": float(f1),
            "Precision": float(precision),
            "Recall": float(recall),
            "Support": int(cm[idx, :].sum()),
            "Predicted": int(cm[:, idx].sum()),
        }
        ious.append(float(iou))
        f1s.append(float(f1))
        precisions.append(float(precision))
        recalls.append(float(recall))
    total = cm.sum()
    oa = float(np.trace(cm) / (total + eps))
    return {
        "OA": oa,
        "mIoU": float(np.mean(ious)) if ious else 0.0,
        "mean_F1": float(np.mean(f1s)) if f1s else 0.0,
        "mean_Precision": float(np.mean(precisions)) if precisions else 0.0,
        "mean_Recall": float(np.mean(recalls)) if recalls else 0.0,
        "per_class": per_class,
    }


def change_mamba_damage_metrics(clf_cm4: np.ndarray) -> Dict[str, object]:
    cm = np.asarray(clf_cm4, dtype=np.float64)
    eps = 1e-12
    per_class = OrderedDict()
    f1s = []
    for idx, name in enumerate(list(XBD_CLASS_NAMES.values())[1:]):
        tp = cm[idx, idx]
        fp = cm[:, idx].sum() - tp
        fn = cm[idx, :].sum() - tp
        precision = tp / (tp + fp + eps)
        recall = tp / (tp + fn + eps)
        f1 = 2.0 * precision * recall / (precision + recall + eps)
        per_class[name] = {
            "F1": float(f1),
            "Precision": float(precision),
            "Recall": float(recall),
            "Support": int(cm[idx, :].sum()),
            "Predicted": int(cm[:, idx].sum()),
        }
        f1s.append(float(f1))
    harmonic = len(f1s) / (sum(1.0 / (f + eps) for f in f1s) + eps) if f1s else 0.0
    macro = float(np.mean(f1s)) if f1s else 0.0
    macro_3cls = float(np.mean(f1s[1:])) if len(f1s) > 1 else 0.0
    return {
        "F1_damage_harmonic_4cls": float(harmonic),
        "F1_damage_macro_4cls": macro,
        "F1_damage_macro_3cls": macro_3cls,
        "per_damage_class": per_class,
    }


def classification_confusion_4(gt: np.ndarray, pred_clf_1_to_4: np.ndarray) -> np.ndarray:
    mask = (gt >= 1) & (gt <= 4)
    if not mask.any():
        return np.zeros((4, 4), dtype=np.int64)
    gt4 = gt[mask].astype(np.int64) - 1
    pred4 = np.clip(pred_clf_1_to_4[mask].astype(np.int64), 1, 4) - 1
    label = 4 * gt4 + pred4
    return np.bincount(label, minlength=16).reshape(4, 4)


def finalize_metrics(cm5: np.ndarray, loc_cm: np.ndarray, clf_cm4: np.ndarray) -> Dict[str, object]:
    tn, fp = loc_cm[0, 0], loc_cm[0, 1]
    fn, tp = loc_cm[1, 0], loc_cm[1, 1]
    loc_precision = tp / (tp + fp + 1e-12)
    loc_recall = tp / (tp + fn + 1e-12)
    loc_f1 = 2.0 * loc_precision * loc_recall / (loc_precision + loc_recall + 1e-12)
    loc_iou = tp / (tp + fp + fn + 1e-12)
    damage = change_mamba_damage_metrics(clf_cm4)
    class_metrics = class_metrics_from_confusion(cm5)
    score_changemamba = 0.3 * float(loc_f1) + 0.7 * float(damage["F1_damage_harmonic_4cls"])
    score_macro = 0.3 * float(loc_f1) + 0.7 * float(damage["F1_damage_macro_4cls"])
    return {
        "Score_ChangeMamba": float(score_changemamba),
        "Score_xView2_macro": float(score_macro),
        "F1_loc": float(loc_f1),
        "IoU_loc": float(loc_iou),
        "Precision_loc": float(loc_precision),
        "Recall_loc": float(loc_recall),
        **damage,
        **class_metrics,
        "confusion_5cls": cm5.astype(int).tolist(),
        "loc_confusion": loc_cm.astype(int).tolist(),
        "damage_confusion_4cls": clf_cm4.astype(int).tolist(),
    }


@torch.no_grad()
def predict_xbd_sample(
    args,
    item: dict,
    processor,
    change_module,
    device,
    clipper,
    bank,
    correction_model,
    component_model,
    image_a: Optional[Image.Image] = None,
    image_b: Optional[Image.Image] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, float]]:
    cfg = base._memory_config_from_args(args)
    if image_a is None or image_b is None:
        image_a = Image.open(item["a"]).convert("RGB")
        image_b = Image.open(item["b"]).convert("RGB")
        pair_embedding = clipper.encode_pair(item)
    else:
        image_a = image_a.convert("RGB")
        image_b = image_b.convert("RGB")
        pair_embedding = encode_pair_from_pil(clipper, image_a, image_b)
    meta, _, _, size_hw = predict_change_from_pil(processor, change_module, image_a, image_b, args, device)
    class_names = ["Building"]
    scores_a, presence_a = extract_class_scores(processor, meta["state_a"], BUILDING_PROMPTS, class_names)
    scores_b, presence_b = extract_class_scores(processor, meta["state_b"], BUILDING_PROMPTS, class_names)
    posterior = build_posterior_inputs(
        scores_a,
        scores_b,
        presence_a,
        presence_b,
        class_names=class_names,
        confusers={"Building": []},
        rho=float(args.rho),
    )

    score = memory_rectified_score(
        posterior.posterior_a[0],
        posterior.posterior_b[0],
        None if bool(args.disable_bcd_gate) else meta["bcd_score"],
        presence_a=presence_a[0],
        presence_b=presence_b[0],
        config=cfg,
    )
    presence = torch.maximum(presence_a[0], presence_b[0])
    base_mask, base_threshold = base.decode_class_score(score, presence, 0.0, args)
    component_features = cmct.component_reliability_features(
        args,
        base_mask,
        score,
        meta["bcd_score"],
        base_threshold,
        component_model,
        device,
    )
    semantic_features = cmct.semantic_relation_features_for_prompt(
        args,
        bank,
        clipper,
        "Building",
        BUILDING_PROMPTS,
    )
    key, stats = cmct.make_query(args, item, "Building", BUILDING_PROMPTS, pair_embedding, clipper, score, meta["bcd_score"], presence)
    tau_mem, tau_corr, diag = cmct.predict_cmct_thresholds(
        args,
        bank,
        correction_model,
        key,
        stats,
        device,
        cls_name="Building",
        component_features=component_features,
        semantic_features=semantic_features,
    )
    tau_mem = cmct.apply_adaptive_floor(tau_mem, base_threshold, args)
    tau_corr = cmct.apply_adaptive_floor(tau_corr, base_threshold, args)
    tau_mem = cmct.apply_class_prior_floor(tau_mem, args, bank, "Building")
    tau_corr = cmct.apply_class_prior_floor(tau_corr, args, bank, "Building")
    tau_corr, semantic_prior_diag = cmct.apply_semantic_prior_floor(tau_corr, args, semantic_features)
    diag.update({f"semantic_prior_{k}": v for k, v in semantic_prior_diag.items() if isinstance(v, (int, float, bool))})
    tau_corr, uncertainty_diag = cmct.apply_uncertainty_calibration(
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
    corr_mask, corr_threshold = cmct.decode_fixed_tau(score, presence, tau_corr, args, budget_ratio=ratio_mem)
    damage_candidate = base.component_gate_filter(
        corr_mask,
        score,
        meta["bcd_score"],
        corr_threshold,
        args,
        component_model,
        device,
    ).astype(bool)

    building_score = torch.maximum(scores_a[0], scores_b[0]).detach().float().cpu().numpy()
    building_support = building_score >= float(args.building_support_threshold)
    building_support = building_support | damage_candidate

    damage_class_names = list(DAMAGE_PROMPTS.keys())
    damage_scores_b, _ = extract_class_scores(processor, meta["state_b"], DAMAGE_PROMPTS, damage_class_names)
    damage_post = competitive_posteriors(
        damage_scores_b,
        damage_class_names,
        DAMAGE_CONFUSERS,
        rho=float(args.rho),
    ).detach().float().cpu().numpy()

    damage_only_scores = damage_post[1:4]
    damage_max = damage_only_scores.max(axis=0)
    damage_arg = damage_only_scores.argmax(axis=0) + 2
    serious_scores = damage_post[2:4]
    serious_max = serious_scores.max(axis=0)
    serious_arg = serious_scores.argmax(axis=0) + 3

    pred_clf = np.ones(size_hw, dtype=np.uint8)
    use_damage = damage_candidate & (damage_max >= float(args.damage_confidence_threshold))
    pred_clf[use_damage] = damage_arg[use_damage].astype(np.uint8)
    use_serious = (~damage_candidate) & (serious_max >= float(args.serious_damage_threshold))
    pred_clf[use_serious] = serious_arg[use_serious].astype(np.uint8)

    pred_full = np.zeros(size_hw, dtype=np.uint8)
    pred_full[building_support] = pred_clf[building_support]
    pred_loc = building_support.astype(np.uint8)

    diagnostics = {
        "base_threshold": float(base_threshold),
        "tau_mem": float(tau_mem),
        "tau_corr": float(tau_corr),
        "corr_threshold": float(corr_threshold),
        "building_support_ratio": float(building_support.mean()),
        "damage_candidate_ratio": float(damage_candidate.mean()),
        "mean_building_score": float(np.mean(building_score)),
        "mean_damage_max": float(np.mean(damage_max)),
    }
    for key_diag, value in diag.items():
        if isinstance(value, (int, float, bool)) and math.isfinite(float(value)):
            diagnostics[f"cmct_{key_diag}"] = float(value)
    return pred_full, pred_clf, pred_loc, diagnostics


def tile_starts(length: int, tile_size: int, stride: int) -> List[int]:
    if tile_size <= 0 or tile_size >= length:
        return [0]
    stride = max(int(stride), 1)
    starts = list(range(0, max(length - tile_size + 1, 1), stride))
    last = max(length - tile_size, 0)
    if starts[-1] != last:
        starts.append(last)
    return starts


@torch.no_grad()
def predict_xbd_sample_tiled(
    args,
    item: dict,
    processor,
    change_module,
    device,
    clipper,
    bank,
    correction_model,
    component_model,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, float]]:
    with Image.open(item["a"]) as img_a_src:
        img_a = img_a_src.convert("RGB")
    with Image.open(item["b"]) as img_b_src:
        img_b = img_b_src.convert("RGB")
    w, h = img_a.size
    tile_size = int(getattr(args, "tile_size", 512))
    tile_stride = int(getattr(args, "tile_stride", tile_size))
    if tile_size <= 0 or (tile_size >= max(h, w) and tile_stride >= max(h, w)):
        return predict_xbd_sample(
            args,
            item,
            processor,
            change_module,
            device,
            clipper,
            bank,
            correction_model,
            component_model,
            image_a=img_a,
            image_b=img_b,
        )

    ys = tile_starts(h, tile_size, tile_stride)
    xs = tile_starts(w, tile_size, tile_stride)
    full_votes = np.zeros((5, h, w), dtype=np.float32)
    clf_votes = np.zeros((4, h, w), dtype=np.float32)
    loc_votes = np.zeros((h, w), dtype=np.float32)
    cover = np.zeros((h, w), dtype=np.float32)
    tile_diags = []
    for y0 in ys:
        for x0 in xs:
            x1 = min(x0 + tile_size, w)
            y1 = min(y0 + tile_size, h)
            box = (x0, y0, x1, y1)
            tile_a = img_a.crop(box)
            tile_b = img_b.crop(box)
            tile_pred, tile_clf, tile_loc, diag = predict_xbd_sample(
                args,
                item,
                processor,
                change_module,
                device,
                clipper,
                bank,
                correction_model,
                component_model,
                image_a=tile_a,
                image_b=tile_b,
            )
            th, tw = tile_pred.shape
            target = np.s_[y0:y0 + th, x0:x0 + tw]
            for cls_id in range(5):
                full_votes[(cls_id,) + target] += (tile_pred == cls_id).astype(np.float32)
            for cls_id in range(1, 5):
                clf_votes[(cls_id - 1,) + target] += (tile_clf == cls_id).astype(np.float32)
            loc_votes[target] += tile_loc.astype(np.float32)
            cover[target] += 1.0
            tile_diags.append(diag)

    pred_full = full_votes.argmax(axis=0).astype(np.uint8)
    pred_clf = (clf_votes.argmax(axis=0) + 1).astype(np.uint8)
    pred_loc = (loc_votes >= (0.5 * np.maximum(cover, 1.0))).astype(np.uint8)
    pred_full[pred_loc == 0] = 0
    pred_full[(pred_loc > 0) & (pred_full == 0)] = pred_clf[(pred_loc > 0) & (pred_full == 0)]
    diagnostics = {
        "tile_size": float(tile_size),
        "tile_stride": float(tile_stride),
        "tile_count": float(len(tile_diags)),
        "building_support_ratio": float((pred_full > 0).mean()),
        "damage_candidate_ratio": float(np.mean([d.get("damage_candidate_ratio", 0.0) for d in tile_diags])) if tile_diags else 0.0,
        "tau_corr_mean": float(np.mean([d.get("tau_corr", 0.0) for d in tile_diags])) if tile_diags else 0.0,
        "tau_corr_min": float(np.min([d.get("tau_corr", 0.0) for d in tile_diags])) if tile_diags else 0.0,
        "tau_corr_max": float(np.max([d.get("tau_corr", 0.0) for d in tile_diags])) if tile_diags else 0.0,
    }
    return pred_full, pred_clf, pred_loc, diagnostics


def write_summary(path: Path, metrics: Dict[str, object], sample_count: int, config: dict) -> None:
    lines = [
        "# xBD BDA CMCT Evaluation",
        "",
        f"- Samples: {sample_count}",
        f"- Score_ChangeMamba: {metrics['Score_ChangeMamba'] * 100:.2f}",
        f"- F1_loc: {metrics['F1_loc'] * 100:.2f}",
        f"- F1_damage_harmonic_4cls: {metrics['F1_damage_harmonic_4cls'] * 100:.2f}",
        f"- F1_damage_macro_4cls: {metrics['F1_damage_macro_4cls'] * 100:.2f}",
        f"- F1_damage_macro_3cls: {metrics['F1_damage_macro_3cls'] * 100:.2f}",
        f"- mIoU_5cls: {metrics['mIoU'] * 100:.2f}",
        "",
        "## Per-class F1",
    ]
    for name, entry in metrics["per_class"].items():
        lines.append(f"- {name}: F1={entry['F1'] * 100:.2f}, IoU={entry['IoU'] * 100:.2f}, Recall={entry['Recall'] * 100:.2f}")
    lines.extend([
        "",
        "## Fixed zero-shot thresholds",
        f"- building_support_threshold: {config['building_support_threshold']}",
        f"- damage_confidence_threshold: {config['damage_confidence_threshold']}",
        f"- serious_damage_threshold: {config['serious_damage_threshold']}",
        f"- tile_size: {config.get('tile_size')}",
        f"- tile_stride: {config.get('tile_stride')}",
    ])
    path.write_text("\n".join(lines) + "\n")


def evaluate(args) -> Dict[str, object]:
    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))

    output_json = Path(args.output_json)
    output_dir = Path(args.output_dir) if args.output_dir else output_json.parent
    pred_dir = output_dir / "pred_raw"
    pred_clf_dir = output_dir / "pred_clf_raw"
    vis_dir = output_dir / "vis"
    pred_dir.mkdir(parents=True, exist_ok=True)
    if bool(args.save_pred_clf):
        pred_clf_dir.mkdir(parents=True, exist_ok=True)
    if bool(args.save_vis):
        vis_dir.mkdir(parents=True, exist_ok=True)

    device = bench.select_device()
    print(f"Using device: {bench.describe_device(device)}")
    processor, change_module, module_type = base.load_models(args, device)
    print(f"Loaded change module: {module_type}")
    clipper = ClipFeatureExtractor(args.clip_model, args.clip_cache_dir, device=device)
    print(f"Loaded CLIP: {args.clip_model}")

    component_model, component_gate_stats, _ = cmct.load_or_train_component_gate(args, [], processor, change_module, device)
    bank, correction_model, clip_memory_stats, clip_corrector_stats = cmct.load_or_train_cmct(
        args,
        [],
        processor,
        change_module,
        device,
        clipper,
        component_model,
    )

    all_items = list_xbd_items(Path(args.xbd_root))
    items = split_items(all_items, args.eval_limit, args.eval_offset, args.num_chunks, args.chunk_index)
    print(f"xBD: eval {len(items)} / all {len(all_items)} samples")

    cm5 = np.zeros((5, 5), dtype=np.int64)
    loc_cm = np.zeros((2, 2), dtype=np.int64)
    clf_cm4 = np.zeros((4, 4), dtype=np.int64)
    diagnostics = []
    target_uniques = OrderedDict()

    for item in tqdm(items, desc="eval-xBD-BDA", dynamic_ncols=True):
        pred_full, pred_clf, pred_loc, diag = predict_xbd_sample_tiled(
            args,
            item,
            processor,
            change_module,
            device,
            clipper,
            bank,
            correction_model,
            component_model,
        )
        gt = read_gt_mask(item, pred_full.shape)
        valid = gt != IGNORE_INDEX
        cm5 += confusion_multiclass(gt, pred_full, 5)
        loc_cm += binary_confusion(gt, pred_loc, valid)
        clf_cm4 += classification_confusion_4(gt, pred_clf)
        unique, counts = np.unique(gt, return_counts=True)
        for value, count in zip(unique.tolist(), counts.tolist()):
            target_uniques[str(int(value))] = int(target_uniques.get(str(int(value)), 0)) + int(count)

        Image.fromarray(pred_full).save(pred_dir / f"{item['name']}_pred.png")
        if bool(args.save_pred_clf):
            Image.fromarray(pred_clf).save(pred_clf_dir / f"{item['name']}_pred_clf.png")
        if bool(args.save_vis):
            Image.fromarray(colorize_mask(gt)).save(vis_dir / f"{item['name']}_gt_color.png")
            Image.fromarray(colorize_mask(pred_full)).save(vis_dir / f"{item['name']}_pred_color.png")
        diagnostics.append({"name": item["name"], **diag})

    metrics = finalize_metrics(cm5, loc_cm, clf_cm4)
    out = {
        "config": vars(args),
        "module_type": module_type,
        "samples": len(items),
        "total_samples": len(all_items),
        "target_unique_pixel_counts": target_uniques,
        "clip_memory_stats": clip_memory_stats,
        "clip_corrector_stats": clip_corrector_stats,
        "component_gate_stats": component_gate_stats,
        "diagnostics": diagnostics,
        "metrics": metrics,
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(jsonable(out), indent=2))
    np.save(output_dir / "confusion_matrix.npy", cm5)
    (output_dir / "config.json").write_text(json.dumps(jsonable(vars(args)), indent=2))
    write_summary(output_dir / "summary.md", metrics, len(items), vars(args))
    print(f"Saved metrics: {output_json}")
    print(f"Score_ChangeMamba: {metrics['Score_ChangeMamba'] * 100:.2f}")
    print(f"F1_loc: {metrics['F1_loc'] * 100:.2f}")
    print(f"F1_damage_harmonic_4cls: {metrics['F1_damage_harmonic_4cls'] * 100:.2f}")
    return out


def merge_outputs(args) -> Dict[str, object]:
    import glob

    paths = sorted(Path(p) for p in glob.glob(str(args.merge_glob)))
    if not paths:
        raise SystemExit(f"No metrics files matched --merge_glob: {args.merge_glob}")
    cm5 = np.zeros((5, 5), dtype=np.int64)
    loc_cm = np.zeros((2, 2), dtype=np.int64)
    clf_cm4 = np.zeros((4, 4), dtype=np.int64)
    samples = 0
    total_samples = 0
    chunk_paths = []
    target_uniques = OrderedDict()
    for path in paths:
        data = json.loads(path.read_text())
        metrics = data["metrics"]
        cm5 += np.asarray(metrics["confusion_5cls"], dtype=np.int64)
        loc_cm += np.asarray(metrics["loc_confusion"], dtype=np.int64)
        clf_cm4 += np.asarray(metrics["damage_confusion_4cls"], dtype=np.int64)
        samples += int(data.get("samples", 0))
        total_samples = max(total_samples, int(data.get("total_samples", 0)))
        chunk_paths.append(str(path))
        for key, value in data.get("target_unique_pixel_counts", {}).items():
            target_uniques[str(key)] = int(target_uniques.get(str(key), 0)) + int(value)
    merged_metrics = finalize_metrics(cm5, loc_cm, clf_cm4)
    output_json = Path(args.output_json)
    output_dir = Path(args.output_dir) if args.output_dir else output_json.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    merge_config = {
        "merge_glob": args.merge_glob,
        "output_json": str(output_json),
        "output_dir": str(output_dir),
        "tile_size": args.tile_size,
        "tile_stride": args.tile_stride,
        "building_support_threshold": args.building_support_threshold,
        "damage_confidence_threshold": args.damage_confidence_threshold,
        "serious_damage_threshold": args.serious_damage_threshold,
    }
    out = {
        "config": merge_config,
        "samples": samples,
        "total_samples": total_samples,
        "num_chunks": len(paths),
        "chunk_paths": chunk_paths,
        "target_unique_pixel_counts": target_uniques,
        "metrics": merged_metrics,
    }
    output_json.write_text(json.dumps(jsonable(out), indent=2))
    np.save(output_dir / "confusion_matrix.npy", cm5)
    (output_dir / "config.json").write_text(json.dumps(jsonable(merge_config), indent=2))
    write_summary(output_dir / "summary.md", merged_metrics, samples, merge_config)
    print(f"Merged {len(paths)} chunks: {samples}/{total_samples} samples")
    print(f"Saved merged metrics: {output_json}")
    print(f"Score_ChangeMamba: {merged_metrics['Score_ChangeMamba'] * 100:.2f}")
    print(f"F1_loc: {merged_metrics['F1_loc'] * 100:.2f}")
    print(f"F1_damage_harmonic_4cls: {merged_metrics['F1_damage_harmonic_4cls'] * 100:.2f}")
    return out


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if bool(getattr(args, "disable_component_gate", False)):
        args.enable_component_gate = False
    if args.merge_glob:
        merge_outputs(args)
    else:
        evaluate(args)


if __name__ == "__main__":
    main()
