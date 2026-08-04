"""Evaluate CogMem-OVCD/STEE evidence on the OVCD_Benchmark datasets.

The task metrics follow the Seg2Change-style protocol:

    pred_task = BCD & (semantic_A(target) | semantic_B(target))

For binary CD datasets without semantic labels, the task target is the dataset
binary change mask.  CACDD is reported as BCD-only because Seg2Change does not
define a class-specific OVCD target for it.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

from eval_change_module import (  # noqa: E402
    compute_bcd,
    confusion_binary,
    metrics_from_confusion,
)
from ovcd_prompts import (  # noqa: E402
    second_class_names,
    second_confusers,
    second_gt_ids,
    second_prompt_bank,
)
from posterior_ovcd import (  # noqa: E402
    apply_bcd_soft_gate,
    build_posterior_inputs,
    decode_ovcd_score,
    extract_class_scores,
)
from sam3 import build_sam3_image_model  # noqa: E402
from sam3.model.sam3_image_processor import Sam3Processor  # noqa: E402
from change_module import (  # noqa: E402
    SemanticTemporalEvidenceEncoder,
    infer_module_type_from_state_dict,
    module_kwargs_from_checkpoint,
    normalize_state_dict_for_stee,
)
from utils.gpu import describe_device, select_device  # noqa: E402


IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff")


SECOND_CLASSES = OrderedDict([
    ("Building", {"gt_ids": [5], "seg_ids": [8]}),
    ("Tree", {"gt_ids": [3], "seg_ids": [5]}),
    ("Water", {"gt_ids": [4], "seg_ids": [6]}),
    ("Grass", {"gt_ids": [1], "seg_ids": [2, 7]}),
    ("Ground", {"gt_ids": [2], "seg_ids": [1]}),
    ("Playground", {"gt_ids": [6], "seg_ids": [3]}),
])

SCSCD_CLASSES = OrderedDict([
    ("Bareland", {"gt_ids": [0], "seg_ids": [0]}),
    ("Water", {"gt_ids": [1], "seg_ids": [1]}),
    ("Building", {"gt_ids": [2], "seg_ids": [2]}),
    ("Structure", {"gt_ids": [3], "seg_ids": [3]}),
    ("Farmland", {"gt_ids": [4], "seg_ids": [4]}),
    ("Vegetation", {"gt_ids": [5], "seg_ids": [5]}),
    ("Road", {"gt_ids": [6], "seg_ids": [6]}),
])

SECOND_NAME_LIST = [
    "background",
    "bareland,barren,ground",
    "grass",
    "sports field",
    "car",
    "tree,forest",
    "water,river",
    "cropland",
    "building,roof,house",
]

SCSCD_NAME_LIST = [
    "bareland,barren,ground,floor,soil",
    "water,river,pond",
    "building,roof,house",
    "structure,construction,greenhouse",
    "farmland,cropland,terrace",
    "grass,vegetation,tree,forest",
    "road",
]

LANDCOVER_NAME_LIST = [
    "background",
    "bareland,barren",
    "grass",
    "road",
    "car",
    "tree,forest",
    "water,river",
    "cropland",
    "building,roof,house",
]

DSIFN_NAME_LIST = [
    "background",
    "building,roof,house,garden,playground,construction,apartment,residential,materials",
    "tree,forest",
    "water,river",
]


SCSCD_POSTERIOR_SPECS = OrderedDict([
    ("Bareland", {
        "gt_ids": [0],
        "prompts": ["bareland", "barren ground", "soil", "ground"],
        "confusers": ["Road", "Farmland", "Building", "Structure"],
    }),
    ("Water", {
        "gt_ids": [1],
        "prompts": ["water", "river", "pond"],
        "confusers": ["Bareland", "Road", "Farmland"],
    }),
    ("Building", {
        "gt_ids": [2],
        "prompts": ["building", "roof", "house"],
        "confusers": ["Structure", "Road", "Bareland"],
    }),
    ("Structure", {
        "gt_ids": [3],
        "prompts": ["structure", "construction", "greenhouse"],
        "confusers": ["Building", "Farmland", "Bareland"],
    }),
    ("Farmland", {
        "gt_ids": [4],
        "prompts": ["farmland", "cropland", "terrace"],
        "confusers": ["Vegetation", "Bareland", "Road"],
    }),
    ("Vegetation", {
        "gt_ids": [5],
        "prompts": ["grass", "vegetation", "tree", "forest"],
        "confusers": ["Farmland", "Bareland", "Road"],
    }),
    ("Road", {
        "gt_ids": [6],
        "prompts": ["road"],
        "confusers": ["Bareland", "Building", "Structure"],
    }),
])

LANDCOVER_POSTERIOR_SPECS = OrderedDict([
    ("Bareland", {"prompts": ["bareland", "barren ground", "ground", "soil"]}),
    ("Grass", {"prompts": ["grass", "low vegetation"]}),
    ("Road", {"prompts": ["road"]}),
    ("Tree", {"prompts": ["tree", "forest"]}),
    ("Water", {"prompts": ["water", "river"]}),
    ("Cropland", {"prompts": ["cropland", "farmland"]}),
    ("Building", {"prompts": ["building", "roof", "house"]}),
])

DSIFN_POSTERIOR_SPECS = OrderedDict([
    ("Building", {
        "prompts": ["building", "roof", "house", "construction", "residential building"],
        "confusers": ["Tree", "Water"],
    }),
    ("Tree", {
        "prompts": ["tree", "forest"],
        "confusers": ["Building", "Water"],
    }),
    ("Water", {
        "prompts": ["water", "river"],
        "confusers": ["Building", "Tree"],
    }),
])


def _with_all_confusers(specs: OrderedDict) -> OrderedDict:
    names = list(specs.keys())
    out = OrderedDict()
    for name, spec in specs.items():
        merged = dict(spec)
        merged.setdefault("confusers", [other for other in names if other != name])
        out[name] = merged
    return out


BINARY_DATASETS = OrderedDict([
    ("CACDD", {
        "root": "CACDD",
        "layout": "flat",
        "task": "bcd_only",
        "name_list": None,
        "target_ids": None,
        "posterior_specs": None,
        "posterior_targets": None,
    }),
    ("CLCD-512", {
        "root": "CLCD-512",
        "layout": "test_ab_label",
        "task": "seg2change",
        "name_list": LANDCOVER_NAME_LIST,
        "target_ids": [1, 2, 3, 5, 6, 7, 8],
        "posterior_specs": _with_all_confusers(LANDCOVER_POSTERIOR_SPECS),
        "posterior_targets": ["Bareland", "Grass", "Road", "Tree", "Water", "Cropland", "Building"],
    }),
    ("DSIFN-512", {
        "root": "DSIFN-512",
        "layout": "test_ab_label",
        "task": "seg2change",
        "name_list": DSIFN_NAME_LIST,
        "target_ids": [1],
        "posterior_specs": DSIFN_POSTERIOR_SPECS,
        "posterior_targets": ["Building"],
    }),
    ("LEVIR-CD-512", {
        "root": "LEVIR-CD-512",
        "layout": "test_ab_label",
        "task": "seg2change",
        "name_list": LANDCOVER_NAME_LIST,
        "target_ids": [8],
        "posterior_specs": _with_all_confusers(LANDCOVER_POSTERIOR_SPECS),
        "posterior_targets": ["Building"],
    }),
    ("WHU-CD-512", {
        "root": "WHU-CD-512",
        "layout": "test_ab_label",
        "task": "seg2change",
        "name_list": LANDCOVER_NAME_LIST,
        "target_ids": [8],
        "posterior_specs": _with_all_confusers(LANDCOVER_POSTERIOR_SPECS),
        "posterior_targets": ["Building"],
    }),
])


def get_cls_idx_from_list(name_list: Sequence[str]) -> Tuple[List[str], List[int]]:
    query_words, query_idx = [], []
    for cls_idx, names in enumerate(name_list):
        for name in names.split(","):
            query_words.append(name.strip())
            query_idx.append(cls_idx)
    return query_words, query_idx


def _as_hw_logits(tensor: torch.Tensor, h: int, w: int) -> torch.Tensor:
    tensor = tensor.float()
    if tensor.ndim == 4:
        if tensor.shape[-2:] != (h, w):
            tensor = F.interpolate(tensor, size=(h, w), mode="bilinear", align_corners=False)
        tensor = tensor.squeeze(0).squeeze(0)
    elif tensor.ndim == 3:
        if tensor.shape[-2:] != (h, w):
            tensor = F.interpolate(tensor.unsqueeze(0), size=(h, w), mode="bilinear", align_corners=False).squeeze(0)
        tensor = tensor.squeeze(0)
    elif tensor.ndim == 2:
        if tensor.shape != (h, w):
            tensor = F.interpolate(
                tensor.view(1, 1, *tensor.shape),
                size=(h, w),
                mode="bilinear",
                align_corners=False,
            ).squeeze()
    else:
        raise ValueError(f"Cannot convert tensor with shape {tuple(tensor.shape)} to HxW")
    return tensor


@torch.no_grad()
def segearth_predict_from_state(
    processor: Sam3Processor,
    state: dict,
    name_list: Sequence[str],
    confidence_threshold: float = 0.1,
    low_conf_label: int = 0,
) -> torch.Tensor:
    """Run SAM3 text decoding and return a class-index semantic map."""
    h = state["original_height"]
    w = state["original_width"]
    device = torch.device(processor.device)
    query_words, query_idx_list = get_cls_idx_from_list(name_list)
    query_idx = torch.tensor(query_idx_list, dtype=torch.int64, device=device)
    num_cls = max(query_idx_list) + 1
    seg_logits = torch.zeros((len(query_words), h, w), device=device)

    autocast_enabled = device.type == "cuda"
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=autocast_enabled):
        for q_idx, q_word in enumerate(query_words):
            processor.reset_all_prompts(state)
            state = processor.set_text_prompt(state=state, prompt=q_word)

            if state["masks_logits"].shape[0] > 0:
                for inst_id in range(state["masks_logits"].shape[0]):
                    inst_logits = state["masks_logits"][inst_id].squeeze()
                    inst_score = state["object_score"][inst_id]
                    inst_logits = _as_hw_logits(inst_logits, h, w)
                    seg_logits[q_idx] = torch.maximum(seg_logits[q_idx], inst_logits * inst_score)

            sem_logits = _as_hw_logits(state["semantic_mask_logits"], h, w)
            seg_logits[q_idx] = torch.maximum(seg_logits[q_idx], sem_logits)
            seg_logits[q_idx] = seg_logits[q_idx] * state["presence_score"]

    if num_cls != len(query_words):
        cls_index = F.one_hot(query_idx, num_classes=num_cls).T.view(num_cls, len(query_words), 1, 1).float()
        cls_index = cls_index.to(seg_logits.device)
        seg_logits = (seg_logits.unsqueeze(0) * cls_index).max(1)[0]

    max_vals, seg_pred = seg_logits.max(dim=0)
    seg_pred[max_vals < confidence_threshold] = int(low_conf_label)
    return seg_pred


def load_change_module(args, device: torch.device) -> Tuple[SemanticTemporalEvidenceEncoder, str]:
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    state_dict = ckpt["change_module"]
    module_type = args.module_type
    if module_type == "auto":
        module_type = ckpt.get("module_type") or infer_module_type_from_state_dict(state_dict)
    change_module = SemanticTemporalEvidenceEncoder(
        dims=[64, 96, 128],
        module_type=module_type,
        **module_kwargs_from_checkpoint(ckpt, args),
    ).to(device)
    state_dict = normalize_state_dict_for_stee(state_dict, module_type)
    change_module.load_state_dict(state_dict)
    change_module.eval()
    return change_module, module_type


def image_files(path: Path) -> List[Path]:
    return sorted([p for p in path.iterdir() if p.suffix.lower() in IMAGE_EXTS])


def find_by_stem(directory: Path, name: str) -> Path:
    direct = directory / name
    if direct.exists():
        return direct
    stem = Path(name).stem
    for ext in IMAGE_EXTS:
        cand = directory / f"{stem}{ext}"
        if cand.exists():
            return cand
    raise FileNotFoundError(f"Cannot find file for {name} in {directory}")


def list_binary_items(root: Path, spec: dict) -> List[dict]:
    ds_root = root / spec["root"]
    if spec["layout"] == "flat":
        a_dir, b_dir, label_dir = ds_root / "A", ds_root / "B", ds_root / "label"
    elif spec["layout"] == "test_ab_label":
        a_dir, b_dir, label_dir = ds_root / "test" / "A", ds_root / "test" / "B", ds_root / "test" / "label"
    else:
        raise ValueError(f"Unknown binary layout: {spec['layout']}")
    items = []
    for a_path in image_files(a_dir):
        items.append({
            "name": a_path.name,
            "a": str(a_path),
            "b": str(find_by_stem(b_dir, a_path.name)),
            "label": str(find_by_stem(label_dir, a_path.name)),
        })
    return items


def list_semantic_items(root: Path, dataset: str) -> List[dict]:
    ds_root = root / dataset / "test"
    t1_dir, t2_dir = ds_root / "T1", ds_root / "T2"
    items = []
    for a_path in image_files(t1_dir):
        items.append({
            "name": a_path.name,
            "a": str(a_path),
            "b": str(find_by_stem(t2_dir, a_path.name)),
            "gt_cd": str(find_by_stem(ds_root / "GT_CD", a_path.name)),
            "gt_t1": str(find_by_stem(ds_root / "GT_T1", a_path.name)),
            "gt_t2": str(find_by_stem(ds_root / "GT_T2", a_path.name)),
        })
    return items


def split_items(items: List[dict], chunk_index: int, num_chunks: int, limit: int) -> List[dict]:
    if limit > 0:
        items = items[:limit]
    if num_chunks <= 1:
        return items
    return [item for idx, item in enumerate(items) if idx % num_chunks == chunk_index]


def read_binary_mask(path: str, size_hw: Tuple[int, int]) -> np.ndarray:
    h, w = size_hw
    img = Image.open(path).convert("L")
    if img.size != (w, h):
        img = img.resize((w, h), Image.NEAREST)
    arr = np.asarray(img)
    return (arr > 0).astype(np.int64)


def read_label(path: str, size_hw: Tuple[int, int]) -> np.ndarray:
    h, w = size_hw
    img = Image.open(path).convert("L")
    if img.size != (w, h):
        img = img.resize((w, h), Image.NEAREST)
    return np.asarray(img)


def mask_for_ids(label: np.ndarray, ids: Iterable[int]) -> np.ndarray:
    out = np.zeros(label.shape, dtype=bool)
    for idx in ids:
        out |= label == int(idx)
    return out


def support_from_seg(seg_a: np.ndarray, seg_b: np.ndarray, target_ids: Iterable[int]) -> np.ndarray:
    support = np.zeros(seg_a.shape, dtype=bool)
    for idx in target_ids:
        support |= (seg_a == int(idx)) | (seg_b == int(idx))
    return support


def specs_to_prompt_bank(specs: OrderedDict) -> Dict[str, List[str]]:
    return {name: list(spec["prompts"]) for name, spec in specs.items()}


def specs_to_confusers(specs: OrderedDict) -> Dict[str, List[str]]:
    names = list(specs.keys())
    return {
        name: list(spec.get("confusers") or [other for other in names if other != name])
        for name, spec in specs.items()
    }


def specs_to_gt_ids(specs: OrderedDict) -> Dict[str, List[int]]:
    return {name: list(spec["gt_ids"]) for name, spec in specs.items()}


def parse_class_thresholds(spec: str) -> Dict[str, float]:
    """Parse thresholds; keys may be Class or Dataset:Class."""
    out = {}
    if not spec:
        return out
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"Invalid class threshold item: {item}")
        name, value = item.split("=", 1)
        out[name.strip()] = float(value)
    return out


def threshold_for(class_thresholds: Dict[str, float], dataset: str, class_name: str, fallback: Optional[float]) -> Optional[float]:
    return class_thresholds.get(f"{dataset}:{class_name}", class_thresholds.get(class_name, fallback))


@torch.no_grad()
def predict_change(
    processor: Sam3Processor,
    change_module: SemanticTemporalEvidenceEncoder,
    item: dict,
    args,
    device: torch.device,
) -> Tuple[dict, torch.Tensor, np.ndarray, Tuple[int, int]]:
    img_a = Image.open(item["a"]).convert("RGB")
    img_b = Image.open(item["b"]).convert("RGB")
    h, w = img_a.size[1], img_a.size[0]
    state_a = processor.set_image(img_a)
    state_b = processor.set_image(img_b)
    fpn_a = [f.to(device) for f in state_a["backbone_out"]["backbone_fpn"]]
    fpn_b = [f.to(device) for f in state_b["backbone_out"]["backbone_fpn"]]
    outputs = change_module(fpn_a, fpn_b, output_size=(h, w))
    bcd, score, threshold = compute_bcd(
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


def init_dataset_result(mode: str, class_names: Optional[Sequence[str]] = None) -> dict:
    out = {
        "mode": mode,
        "samples": 0,
        "task_confusion": np.zeros((2, 2), dtype=np.int64),
        "bcd_confusion": np.zeros((2, 2), dtype=np.int64),
        "thresholds": [],
    }
    if class_names:
        out["class_confusions"] = {name: np.zeros((2, 2), dtype=np.int64) for name in class_names}
    return out


def finalize_dataset_result(raw: dict) -> dict:
    out = {
        "mode": raw["mode"],
        "samples": int(raw["samples"]),
        "threshold_mean": float(np.mean(raw["thresholds"])) if raw["thresholds"] else None,
        "threshold_std": float(np.std(raw["thresholds"])) if raw["thresholds"] else None,
        "bcd_confusion": raw["bcd_confusion"].tolist(),
        "bcd": {k: float(v) for k, v in metrics_from_confusion(raw["bcd_confusion"]).items()},
    }
    if "class_confusions" in raw:
        all_metrics = {k: [] for k in ["IoU", "F1", "Pre", "Recall", "OA", "Kappa"]}
        class_metrics = OrderedDict()
        for name, cm in raw["class_confusions"].items():
            m = metrics_from_confusion(cm)
            class_metrics[name] = {k: float(v) for k, v in m.items()}
            for key in all_metrics:
                all_metrics[key].append(m[key])
        out["class_confusions"] = {k: v.tolist() for k, v in raw["class_confusions"].items()}
        out["class_metrics"] = class_metrics
        out["average"] = {k: float(np.mean(v)) for k, v in all_metrics.items()}
    else:
        out["task_confusion"] = raw["task_confusion"].tolist()
        out["task"] = {k: float(v) for k, v in metrics_from_confusion(raw["task_confusion"]).items()}
    return out


def evaluate_binary_dataset(args, dataset: str, spec: dict, items: List[dict], processor, change_module, device) -> dict:
    raw = init_dataset_result(mode=spec["task"])
    low_conf_label = 0
    for item in tqdm(items, desc=f"{dataset}", dynamic_ncols=True):
        meta, bcd, bcd_np, size_hw = predict_change(processor, change_module, item, args, device)
        gt = read_binary_mask(item["label"], size_hw)
        raw["bcd_confusion"] += confusion_binary(gt, bcd_np)
        raw["thresholds"].append(meta["threshold"])
        raw["samples"] += 1

        if spec["task"] == "bcd_only":
            pred_task = bcd_np
        else:
            seg_a = segearth_predict_from_state(
                processor,
                meta["state_a"],
                spec["name_list"],
                confidence_threshold=args.semantic_confidence_threshold,
                low_conf_label=low_conf_label,
            ).detach().cpu().numpy()
            seg_b = segearth_predict_from_state(
                processor,
                meta["state_b"],
                spec["name_list"],
                confidence_threshold=args.semantic_confidence_threshold,
                low_conf_label=low_conf_label,
            ).detach().cpu().numpy()
            support = support_from_seg(seg_a, seg_b, spec["target_ids"])
            pred_task = (bcd_np.astype(bool) & support).astype(np.int64)
        raw["task_confusion"] += confusion_binary(gt, pred_task)
    return raw


def evaluate_second(args, items: List[dict], processor, change_module, device) -> dict:
    raw = init_dataset_result(mode="semantic_querywise", class_names=list(SECOND_CLASSES.keys()))
    for item in tqdm(items, desc="SECOND", dynamic_ncols=True):
        meta, _, bcd_np, size_hw = predict_change(processor, change_module, item, args, device)
        gt_cd = read_binary_mask(item["gt_cd"], size_hw)
        raw["bcd_confusion"] += confusion_binary(gt_cd, bcd_np)
        raw["thresholds"].append(meta["threshold"])
        raw["samples"] += 1

        seg_a = segearth_predict_from_state(
            processor, meta["state_a"], SECOND_NAME_LIST,
            confidence_threshold=args.semantic_confidence_threshold, low_conf_label=0,
        ).detach().cpu().numpy()
        seg_b = segearth_predict_from_state(
            processor, meta["state_b"], SECOND_NAME_LIST,
            confidence_threshold=args.semantic_confidence_threshold, low_conf_label=0,
        ).detach().cpu().numpy()
        gt_t1 = read_label(item["gt_t1"], size_hw)
        gt_t2 = read_label(item["gt_t2"], size_hw)

        for name, cls in SECOND_CLASSES.items():
            support = support_from_seg(seg_a, seg_b, cls["seg_ids"])
            pred = (bcd_np.astype(bool) & support).astype(np.int64)
            gt = (mask_for_ids(gt_t1, cls["gt_ids"]) | mask_for_ids(gt_t2, cls["gt_ids"])).astype(np.int64)
            raw["class_confusions"][name] += confusion_binary(gt, pred)
    return raw


def evaluate_scscd(args, items: List[dict], processor, change_module, device) -> dict:
    raw = init_dataset_result(mode="semantic_querywise_cd_masked", class_names=list(SCSCD_CLASSES.keys()))
    for item in tqdm(items, desc="SCSCD", dynamic_ncols=True):
        meta, _, bcd_np, size_hw = predict_change(processor, change_module, item, args, device)
        gt_cd = read_binary_mask(item["gt_cd"], size_hw)
        raw["bcd_confusion"] += confusion_binary(gt_cd, bcd_np)
        raw["thresholds"].append(meta["threshold"])
        raw["samples"] += 1

        seg_a = segearth_predict_from_state(
            processor, meta["state_a"], SCSCD_NAME_LIST,
            confidence_threshold=args.semantic_confidence_threshold, low_conf_label=-1,
        ).detach().cpu().numpy()
        seg_b = segearth_predict_from_state(
            processor, meta["state_b"], SCSCD_NAME_LIST,
            confidence_threshold=args.semantic_confidence_threshold, low_conf_label=-1,
        ).detach().cpu().numpy()
        gt_t1 = read_label(item["gt_t1"], size_hw)
        gt_t2 = read_label(item["gt_t2"], size_hw)

        for name, cls in SCSCD_CLASSES.items():
            support = support_from_seg(seg_a, seg_b, cls["seg_ids"])
            pred = (bcd_np.astype(bool) & support).astype(np.int64)
            gt = (
                (mask_for_ids(gt_t1, cls["gt_ids"]) | mask_for_ids(gt_t2, cls["gt_ids"]))
                & gt_cd.astype(bool)
            ).astype(np.int64)
            raw["class_confusions"][name] += confusion_binary(gt, pred)
    return raw


@torch.no_grad()
def build_posterior_masks(
    processor,
    args,
    dataset: str,
    meta: dict,
    class_names: Sequence[str],
    prompt_bank: Dict[str, List[str]],
    confusers: Dict[str, List[str]],
    class_thresholds: Dict[str, float],
    image_mean: Optional[np.ndarray] = None,
) -> Tuple[Dict[str, np.ndarray], Dict[str, float]]:
    class_names = list(class_names)
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
        score = posterior.delta[cls_idx]
        if not args.disable_bcd_gate:
            score = apply_bcd_soft_gate(
                score,
                meta["bcd_score"],
                alpha=args.bcd_alpha,
                floor=args.bcd_floor,
            )
        if args.support_blend_weight > 0:
            if args.support_source == "raw":
                support = torch.maximum(posterior.scores_a[cls_idx], posterior.scores_b[cls_idx])
            else:
                support = torch.maximum(posterior.posterior_a[cls_idx], posterior.posterior_b[cls_idx])
            score = torch.maximum(
                score,
                (support * meta["bcd_score"].float().clamp(0, 1) * args.support_blend_weight).clamp(0, 1),
            )

        presence = torch.maximum(presence_a[cls_idx], presence_b[cls_idx])
        threshold_abs = threshold_for(class_thresholds, dataset, cls_name, args.threshold_abs)
        threshold_mode = args.threshold_mode
        pred_mask, thd = decode_ovcd_score(
            score,
            threshold_mode=threshold_mode,
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
            threshold_max_foreground_ratio=args.threshold_max_foreground_ratio,
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


def evaluate_binary_dataset_posterior(args, dataset: str, spec: dict, items: List[dict], processor, change_module, device) -> dict:
    mode = "bcd_only" if spec["posterior_specs"] is None else "posterior_binary"
    raw = init_dataset_result(mode=mode)
    class_thresholds = parse_class_thresholds(args.class_thresholds)
    for item in tqdm(items, desc=f"{dataset}", dynamic_ncols=True):
        meta, _, bcd_np, size_hw = predict_change(processor, change_module, item, args, device)
        gt = read_binary_mask(item["label"], size_hw)
        raw["bcd_confusion"] += confusion_binary(gt, bcd_np)
        raw["thresholds"].append(meta["threshold"])
        raw["samples"] += 1

        if spec["posterior_specs"] is None:
            pred_task = bcd_np
        else:
            specs = spec["posterior_specs"]
            class_names = list(specs.keys())
            pred_masks, _ = build_posterior_masks(
                processor,
                args,
                dataset,
                meta,
                class_names,
                specs_to_prompt_bank(specs),
                specs_to_confusers(specs),
                class_thresholds,
            )
            pred_task = np.zeros(size_hw, dtype=np.int64)
            for cls_name in spec["posterior_targets"]:
                pred_task |= pred_masks[cls_name]
        raw["task_confusion"] += confusion_binary(gt, pred_task.astype(np.int64))
    return raw


def evaluate_second_posterior(args, items: List[dict], processor, change_module, device) -> dict:
    class_names = second_class_names()
    prompt_bank = second_prompt_bank()
    confusers = second_confusers()
    gt_ids = second_gt_ids()
    class_thresholds = parse_class_thresholds(args.class_thresholds)
    raw = init_dataset_result(mode="posterior_querywise", class_names=class_names)
    for item in tqdm(items, desc="SECOND", dynamic_ncols=True):
        meta, _, bcd_np, size_hw = predict_change(processor, change_module, item, args, device)
        gt_cd = read_binary_mask(item["gt_cd"], size_hw)
        raw["bcd_confusion"] += confusion_binary(gt_cd, bcd_np)
        raw["thresholds"].append(meta["threshold"])
        raw["samples"] += 1

        pred_masks, _ = build_posterior_masks(
            processor,
            args,
            "SECOND",
            meta,
            class_names,
            prompt_bank,
            confusers,
            class_thresholds,
        )
        gt_t1 = read_label(item["gt_t1"], size_hw)
        gt_t2 = read_label(item["gt_t2"], size_hw)
        for cls_name in class_names:
            gt = np.zeros(size_hw, dtype=bool)
            for gid in gt_ids[cls_name]:
                gt |= (gt_t1 == gid) | (gt_t2 == gid)
            raw["class_confusions"][cls_name] += confusion_binary(
                gt.astype(np.int64),
                pred_masks[cls_name].astype(np.int64),
            )
    return raw


def evaluate_scscd_posterior(args, items: List[dict], processor, change_module, device) -> dict:
    specs = SCSCD_POSTERIOR_SPECS
    class_names = list(specs.keys())
    prompt_bank = specs_to_prompt_bank(specs)
    confusers = specs_to_confusers(specs)
    gt_ids = specs_to_gt_ids(specs)
    class_thresholds = parse_class_thresholds(args.class_thresholds)
    raw = init_dataset_result(mode="posterior_querywise_cd_masked", class_names=class_names)
    for item in tqdm(items, desc="SCSCD", dynamic_ncols=True):
        meta, _, bcd_np, size_hw = predict_change(processor, change_module, item, args, device)
        gt_cd = read_binary_mask(item["gt_cd"], size_hw)
        raw["bcd_confusion"] += confusion_binary(gt_cd, bcd_np)
        raw["thresholds"].append(meta["threshold"])
        raw["samples"] += 1

        pred_masks, _ = build_posterior_masks(
            processor,
            args,
            "SCSCD",
            meta,
            class_names,
            prompt_bank,
            confusers,
            class_thresholds,
        )
        gt_t1 = read_label(item["gt_t1"], size_hw)
        gt_t2 = read_label(item["gt_t2"], size_hw)
        for cls_name in class_names:
            gt = np.zeros(size_hw, dtype=bool)
            for gid in gt_ids[cls_name]:
                gt |= (gt_t1 == gid) | (gt_t2 == gid)
            gt &= gt_cd.astype(bool)
            raw["class_confusions"][cls_name] += confusion_binary(
                gt.astype(np.int64),
                pred_masks[cls_name].astype(np.int64),
            )
    return raw


def raw_to_jsonable(raw_results: Dict[str, dict]) -> dict:
    return {name: finalize_dataset_result(raw) for name, raw in raw_results.items()}


def merge_results(args) -> None:
    merged: Dict[str, dict] = {}
    shard_paths = sorted(Path(args.merge_dir).glob(args.merge_glob))
    if not shard_paths:
        raise SystemExit(f"No shard files found: {args.merge_dir}/{args.merge_glob}")

    for path in shard_paths:
        data = json.load(open(path))
        for ds, result in data["datasets"].items():
            target = merged.setdefault(ds, {
                "mode": result["mode"],
                "samples": 0,
                "thresholds": [],
                "bcd_confusion": np.zeros((2, 2), dtype=np.int64),
            })
            target["samples"] += int(result["samples"])
            if result.get("threshold_mean") is not None and result["samples"] > 0:
                # Shard-level threshold means are only diagnostic; merge them
                # approximately by repeating one weighted representative.
                target["thresholds"].extend([float(result["threshold_mean"])] * int(result["samples"]))
            target["bcd_confusion"] += np.asarray(result["bcd_confusion"], dtype=np.int64)
            if "class_confusions" in result:
                class_target = target.setdefault("class_confusions", {})
                for cls_name, cm in result["class_confusions"].items():
                    class_target.setdefault(cls_name, np.zeros((2, 2), dtype=np.int64))
                    class_target[cls_name] += np.asarray(cm, dtype=np.int64)
            else:
                target.setdefault("task_confusion", np.zeros((2, 2), dtype=np.int64))
                target["task_confusion"] += np.asarray(result["task_confusion"], dtype=np.int64)

    out = {
        "merged_from": [str(p) for p in shard_paths],
        "datasets": raw_to_jsonable(merged),
    }
    out["summary"] = build_summary(out["datasets"])
    os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(out, f, indent=2)
    print_summary(out["datasets"])
    print(f"Saved merged metrics: {args.output_json}")


def pct_metrics(metrics: dict) -> dict:
    return {k: (v * 100 if k != "Kappa" else v) for k, v in metrics.items()}


def build_summary(datasets: Dict[str, dict]) -> dict:
    summary = OrderedDict()
    for name, data in datasets.items():
        if "average" in data:
            summary[name] = {
                "samples": data["samples"],
                "task": pct_metrics(data["average"]),
                "bcd": pct_metrics(data["bcd"]),
            }
        else:
            summary[name] = {
                "samples": data["samples"],
                "task": pct_metrics(data["task"]),
                "bcd": pct_metrics(data["bcd"]),
            }
    return summary


def print_summary(datasets: Dict[str, dict], title: str = "OVCD_Benchmark Summary") -> None:
    print("\n" + "=" * 96)
    print(title)
    print("=" * 96)
    print(f"{'Dataset':<16} {'Samples':>7} {'Task IoU':>9} {'Task F1':>9} {'Task Pre':>9} "
          f"{'Task Rec':>9} {'Task OA':>9} {'Task Kap':>9} {'BCD F1':>9}")
    print("-" * 96)
    for name, data in datasets.items():
        task = data.get("average", data.get("task"))
        bcd = data["bcd"]
        print(f"{name:<16} {data['samples']:7d} {task['IoU']*100:9.2f} {task['F1']*100:9.2f} "
              f"{task['Pre']*100:9.2f} {task['Recall']*100:9.2f} {task['OA']*100:9.2f} "
              f"{task['Kappa']:9.4f} {bcd['F1']*100:9.2f}")


def evaluate(args) -> None:
    device = select_device()
    print(f"Using device: {describe_device(device)}")
    change_module, module_type = load_change_module(args, device)
    print(f"Loaded change module: {module_type}")

    bpe_path = project_root / "sam3/assets/bpe_simple_vocab_16e6.txt.gz"
    model = build_sam3_image_model(
        bpe_path=str(bpe_path),
        checkpoint_path=args.sam3_checkpoint,
        device=str(device),
    )
    processor = Sam3Processor(
        model,
        confidence_threshold=args.sam3_confidence_threshold,
        device=str(device),
    )

    benchmark_root = Path(args.benchmark_root)
    requested = [x.strip() for x in args.datasets.split(",") if x.strip()]
    if requested == ["all"]:
        requested = list(BINARY_DATASETS.keys()) + ["SCSCD", "SECOND"]

    raw_results = OrderedDict()
    for dataset in requested:
        if dataset in BINARY_DATASETS:
            all_items = list_binary_items(benchmark_root, BINARY_DATASETS[dataset])
            items = split_items(all_items, args.chunk_index, args.num_chunks, args.limit)
            print(f"{dataset}: evaluating {len(items)}/{len(all_items)} samples")
            if args.prediction_mode == "posterior":
                raw_results[dataset] = evaluate_binary_dataset_posterior(
                    args, dataset, BINARY_DATASETS[dataset], items, processor, change_module, device)
            else:
                raw_results[dataset] = evaluate_binary_dataset(
                    args, dataset, BINARY_DATASETS[dataset], items, processor, change_module, device)
        elif dataset == "SECOND":
            all_items = list_semantic_items(benchmark_root, "SECOND")
            items = split_items(all_items, args.chunk_index, args.num_chunks, args.limit)
            print(f"SECOND: evaluating {len(items)}/{len(all_items)} samples")
            if args.prediction_mode == "posterior":
                raw_results[dataset] = evaluate_second_posterior(args, items, processor, change_module, device)
            else:
                raw_results[dataset] = evaluate_second(args, items, processor, change_module, device)
        elif dataset == "SCSCD":
            all_items = list_semantic_items(benchmark_root, "SCSCD")
            items = split_items(all_items, args.chunk_index, args.num_chunks, args.limit)
            print(f"SCSCD: evaluating {len(items)}/{len(all_items)} samples")
            if args.prediction_mode == "posterior":
                raw_results[dataset] = evaluate_scscd_posterior(args, items, processor, change_module, device)
            else:
                raw_results[dataset] = evaluate_scscd(args, items, processor, change_module, device)
        else:
            raise ValueError(f"Unknown dataset: {dataset}")

    out = {
        "config": vars(args),
        "module_type": module_type,
        "datasets": raw_to_jsonable(raw_results),
    }
    out["summary"] = build_summary(out["datasets"])
    print_summary(out["datasets"], title=f"OVCD_Benchmark {args.prediction_mode} Summary")
    if args.output_json:
        os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)
        with open(args.output_json, "w") as f:
            json.dump(out, f, indent=2)
        print(f"Saved metrics: {args.output_json}")


def main() -> None:
    parser = argparse.ArgumentParser(description="CogMem-OVCD / STEE OVCD_Benchmark evaluation")
    parser.add_argument("--benchmark_root", type=str, default="data/OVCD_Benchmark")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--sam3_checkpoint", type=str, default="weights/sam3.pt")
    parser.add_argument("--output_json", type=str, default=None)
    parser.add_argument("--datasets", type=str, default="all",
                        help="Comma-separated datasets or all")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--chunk_index", type=int, default=0)
    parser.add_argument("--num_chunks", type=int, default=1)
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
    parser.add_argument("--semantic_confidence_threshold", type=float, default=0.1)
    parser.add_argument("--prediction_mode", type=str, default="seg2change",
                        choices=["seg2change", "posterior"],
                        help="seg2change uses BCD & semantic OR; posterior uses query-conditioned posterior scores.")
    parser.add_argument("--rho", type=float, default=2.0)
    parser.add_argument("--disable_bcd_gate", action="store_true")
    parser.add_argument("--bcd_alpha", type=float, default=1.0)
    parser.add_argument("--bcd_floor", type=float, default=0.1)
    parser.add_argument("--support_blend_weight", type=float, default=0.0)
    parser.add_argument("--support_source", type=str, default="posterior", choices=["posterior", "raw"])
    parser.add_argument("--threshold_mode", type=str, default="percentile",
                        choices=["percentile", "mean_std", "otsu", "otsu_edge", "candidate_otsu", "adaptive"])
    parser.add_argument("--threshold_percentile", type=float, default=85.0)
    parser.add_argument("--threshold_abs", type=float, default=None)
    parser.add_argument("--class_thresholds", type=str, default="")
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
    parser.add_argument("--region_mode", type=str, default="component", choices=["none", "component", "slic", "acf_component", "acf_region"])
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
    parser.add_argument("--bcd_threshold_abs", type=float, default=None)
    parser.add_argument("--bcd_threshold_std_factor", type=float, default=1.0)
    parser.add_argument("--acf_conf_percentile", type=float, default=50.0)
    parser.add_argument("--acf_conf_lambda", type=float, default=1.0)
    parser.add_argument("--acf_conf_min", type=float, default=0.0)
    parser.add_argument("--acf_conf_max", type=float, default=1.0)
    parser.add_argument("--acf_cv_max", type=float, default=2.0)
    parser.add_argument("--merge_dir", type=str, default=None,
                        help="Merge existing shard JSON files instead of running inference.")
    parser.add_argument("--merge_glob", type=str, default="eval_shard*.json")
    args = parser.parse_args()

    if args.merge_dir:
        if not args.output_json:
            raise SystemExit("--output_json is required with --merge_dir")
        merge_results(args)
    else:
        evaluate(args)


if __name__ == "__main__":
    main()
