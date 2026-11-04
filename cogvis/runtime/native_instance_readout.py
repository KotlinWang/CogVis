"""Native xBD instance readout for the frozen CogVis BDA forward.

The frozen pixel route (:mod:`evaluate_xbd_bda_cmct`) returns only the 5-class
``pred_full`` / ``pred_clf`` / ``pred_loc`` PNGs.  Everything that carries the
*instance* identity and the *continuous* evidence (the SAM3 text-prompt masks
and their object scores, the change score, the competition-normalised SAM3
text-state scores) is discarded before it can be exported.

This module turns those native outputs into per-instance records without
inventing anything:

* an instance is one **SAM3 text-prompt mask** produced on the PRE image from
  the frozen ``building`` prompt bank of the installed xBD profile (the
  ``collapse_heavy`` CLI profile has five prompts), kept at the resolution the
  SAM3 processor emitted it in and mapped back onto the native PRE grid through
  its tile origin. No connected-component post-processing, no morphological
  "shape repair", and no union of the change-candidate branch with the
  building-prompt branch;
* the localization score is the native SAM3 object score of that mask;
* the damage state and the damage scores are read out of the *same* forward's
  ``pred_clf`` (4-class BDA map) and ``damage_state_scores`` (competition
  normalised SAM3 text-state scores computed on the POST image).

Thresholds are deliberately limited to the ones the frozen route already uses:
the SAM3 processor confidence threshold (its own default) plus one fixed
class-agnostic mask-IoU NMS threshold that only de-duplicates prompts from the
frozen building bank (five prompts under the original ``collapse_heavy`` xBD
CLI profile). There is no area filter and no per-dataset tuning.

The score definitions below are *ranking scores*, not calibrated
probabilities.  ``damage_state_scores`` is the output of
``competitive_posteriors`` (target-vs-confuser competition), so it is reported
as ``state_score`` / ``damage_query_score`` and never as a probability.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

#: Fixed class-agnostic mask-IoU threshold used only to de-duplicate the
#: prompts of the installed SAM3 ``building`` bank inside one tile.
NMS_MASK_IOU = 0.5

#: Frozen category ids of the published xBD three-grade instance reference.
CATEGORY_BY_GRADE = {"intact": 1, "damaged": 2, "destroyed": 3}

#: The frozen BDA forward labels buildings with 1..4; xBD ``minor-damage`` and
#: ``major-damage`` are merged into a single ``damaged`` grade by the published
#: reference, so the readout merges them here as well.
GRADE_BY_PRED_CLF = {1: "intact", 2: "damaged", 3: "damaged", 4: "destroyed"}

#: Index layout of ``damage_state_scores`` (the order of ``DAMAGE_PROMPTS``).
STATE_INDEX = {"no-damage": 0, "minor-damage": 1, "major-damage": 2, "destroyed": 3}

DAMAGE_STATE_SCORE_DEFINITION = {
    "intact": "mask mean of damage_state_scores[no-damage]",
    "damaged": "mask mean of max(damage_state_scores[minor-damage], damage_state_scores[major-damage])",
    "destroyed": "mask mean of damage_state_scores[destroyed]",
    "damage_query": "mask mean of max(damage_state_scores[minor-damage], "
                    "damage_state_scores[major-damage], damage_state_scores[destroyed])",
}

STATE_PROVENANCE = (
    "grade = majority of the frozen BDA forward's pred_clf 4-class map inside the native SAM3 "
    "mask (minor-damage and major-damage merged into damaged per the published reference); "
    "ties resolve to the smallest label id"
)


# --------------------------------------------------------------------------- #
# geometry helpers
# --------------------------------------------------------------------------- #
def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    """Intersection over union of two boolean masks of identical shape."""
    a = np.asarray(a, dtype=bool)
    b = np.asarray(b, dtype=bool)
    if a.shape != b.shape:
        raise ValueError(f"mask shapes differ: {a.shape} vs {b.shape}")
    inter = int(np.logical_and(a, b).sum())
    if inter == 0:
        return 0.0
    union = int(np.logical_or(a, b).sum())
    return float(inter) / float(union) if union else 0.0


def nms_keep(masks: Sequence[np.ndarray], scores: Sequence[float],
             iou_threshold: float = NMS_MASK_IOU) -> List[int]:
    """Greedy class-agnostic mask NMS; returns kept indices.

    Ordering is deterministic: descending score, ties broken by ascending index.
    """
    masks = list(masks)
    scores = [float(value) for value in scores]
    if len(masks) != len(scores):
        raise ValueError("masks and scores must have the same length")
    order = sorted(range(len(masks)), key=lambda index: (-scores[index], index))
    kept: List[int] = []
    for index in order:
        if all(mask_iou(masks[index], masks[other]) <= float(iou_threshold) for other in kept):
            kept.append(index)
    return kept


def bbox_xyxy(mask: np.ndarray) -> Tuple[int, int, int, int]:
    """Inclusive-exclusive bounding box of a non-empty boolean mask."""
    rows = np.flatnonzero(np.asarray(mask, dtype=bool).any(axis=1))
    cols = np.flatnonzero(np.asarray(mask, dtype=bool).any(axis=0))
    if rows.size == 0 or cols.size == 0:
        raise ValueError("cannot box an empty mask")
    return int(cols[0]), int(rows[0]), int(cols[-1]) + 1, int(rows[-1]) + 1


def _region_mean(values: np.ndarray, mask: np.ndarray) -> float:
    selected = np.asarray(values, dtype=np.float64)[np.asarray(mask, dtype=bool)]
    if selected.size == 0:
        raise ValueError("cannot average an empty region")
    return float(selected.mean())


# --------------------------------------------------------------------------- #
# RLE (COCO conventions)
# --------------------------------------------------------------------------- #
def _run_lengths(flat: np.ndarray) -> List[int]:
    """COCO uncompressed RLE counts: alternating runs, starting with background.

    The first count is always the number of leading ``False`` pixels (possibly
    0), because ``pycocotools.mask.decode`` starts from ``value = False``.
    """
    flat = np.asarray(flat, dtype=bool)
    size = int(flat.size)
    if size == 0:
        return []
    changes = np.flatnonzero(flat[1:] != flat[:-1]) + 1
    bounds = np.concatenate((np.array([0], dtype=np.int64), changes, np.array([size], dtype=np.int64)))
    lengths = np.diff(bounds).astype(np.int64).tolist()
    if bool(flat[0]):
        lengths = [0, *lengths]
    return [int(value) for value in lengths]


def rle_encode(mask: np.ndarray) -> Dict[str, object]:
    """Uncompressed COCO RLE, column-major, counts as an int list."""
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 2:
        raise ValueError(f"expected a 2-D mask, got shape {mask.shape}")
    flat = np.asfortranarray(mask).reshape(-1, order="F")
    return {"size": [int(mask.shape[0]), int(mask.shape[1])], "counts": _run_lengths(flat)}


def rle_decode(encoding: Dict[str, object]) -> np.ndarray:
    """Inverse of :func:`rle_encode`."""
    height, width = (int(value) for value in encoding["size"])
    flat = np.zeros(height * width, dtype=bool)
    position = 0
    value = False
    for count in encoding["counts"]:
        count = int(count)
        if value and count:
            flat[position:position + count] = True
        position += count
        value = not value
    return flat.reshape((height, width), order="F")


def compressed_rle(mask: np.ndarray) -> Dict[str, object]:
    """Compressed COCO RLE for a COCO detection submission.

    ``pycocotools`` is imported lazily so this module stays importable (and
    testable) in a numpy-only environment.
    """
    try:
        from pycocotools import mask as coco_mask
    except ImportError as error:  # pragma: no cover - environment guard
        raise RuntimeError(
            "pycocotools is required to write compressed COCO RLE submissions; "
            "install it in the run environment"
        ) from error
    encoded = coco_mask.encode(np.asfortranarray(np.asarray(mask, dtype=np.uint8)))
    counts = encoded["counts"]
    if isinstance(counts, bytes):
        counts = counts.decode("ascii")
    mask = np.asarray(mask, dtype=bool)
    return {"size": [int(mask.shape[0]), int(mask.shape[1])], "counts": counts}


# --------------------------------------------------------------------------- #
# per-instance readout
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class NativeInstance:
    """One native SAM3 PRE building-prompt mask with its read-out scores."""

    pair_id: str
    image_id: int
    split: str
    instance_index: int
    prompt: str
    class_name: str
    tile_origin: Tuple[int, int]
    grade: str
    mask: np.ndarray
    area: int
    bbox_xyxy: Tuple[int, int, int, int]
    localization_score: float
    damage_query_score: float
    state_scores: Dict[str, float]
    support_overlap: float

    @property
    def uid(self) -> str:
        return f"{self.pair_id}#{self.instance_index:05d}"

    @property
    def grade_score(self) -> float:
        return float(self.state_scores[self.grade])

    @property
    def category_id(self) -> int:
        return CATEGORY_BY_GRADE[self.grade]

    def instances_row(self, method: str, dataset_id: str = "xbd") -> Dict[str, object]:
        """Neutral per-instance JSONL row (same key set as the frozen export)."""
        return {
            "method": method,
            "dataset_id": dataset_id,
            "split": self.split,
            "pair_id": self.pair_id,
            "image_id": self.image_id,
            "uid": self.uid,
            "prompt": self.prompt,
            "class_name": self.class_name,
            "tile_origin": [int(self.tile_origin[0]), int(self.tile_origin[1])],
            "state": self.grade,
            "score": self.grade_score,
            "localization_score": self.localization_score,
            "damage_query_score": self.damage_query_score,
            "state_scores": {key: float(value) for key, value in sorted(self.state_scores.items())},
            "support_overlap": self.support_overlap,
            "area": int(self.area),
            "bbox_xyxy": [int(value) for value in self.bbox_xyxy],
            "mask_rle": rle_encode(self.mask),
        }

    def to_detection(self, category_id: int, score: float, encode_mask=None) -> Dict[str, object]:
        """One COCO detection record on the native PRE grid."""
        if encode_mask is None:
            encode_mask = compressed_rle
        x0, y0, x1, y1 = self.bbox_xyxy
        return {
            "image_id": int(self.image_id),
            "category_id": int(category_id),
            "score": float(score),
            "segmentation": encode_mask(self.mask),
            "bbox": [x0, y0, x1 - x0, y1 - y0],
        }


def grade_from_pred_clf(pred_clf: np.ndarray, mask: np.ndarray) -> str:
    """Majority 4-class label of the BDA map inside the mask, merged to 3 grades."""
    pred_clf = np.asarray(pred_clf)
    mask = np.asarray(mask, dtype=bool)
    if pred_clf.shape != mask.shape:
        raise ValueError(f"pred_clf shape {pred_clf.shape} != mask shape {mask.shape}")
    labels = pred_clf[mask].astype(np.int64).ravel()
    if labels.size == 0:
        raise ValueError("cannot grade an empty mask")
    valid = np.isin(labels, np.array([1, 2, 3, 4], dtype=np.int64))
    if not bool(valid.any()):
        raise ValueError(f"pred_clf holds no 1..4 label inside the mask: {np.unique(labels)[:8].tolist()}")
    counts = np.bincount(labels[valid], minlength=5)
    majority = int(np.argmax(counts))  # ties resolve to the smallest label id
    return GRADE_BY_PRED_CLF[majority]


def state_scores_for_mask(mask: np.ndarray, damage_state_scores: np.ndarray) -> Dict[str, float]:
    """Mask-mean state scores for the three published grades."""
    scores = np.asarray(damage_state_scores, dtype=np.float32)
    if scores.ndim != 3 or scores.shape[0] != 4:
        raise ValueError(f"damage_state_scores must be (4,H,W), got {scores.shape}")
    if scores.shape[1:] != mask.shape:
        raise ValueError(f"damage_state_scores grid {scores.shape[1:]} != mask grid {mask.shape}")
    minor_or_major = np.maximum(scores[STATE_INDEX["minor-damage"]], scores[STATE_INDEX["major-damage"]])
    return {
        "intact": _region_mean(scores[STATE_INDEX["no-damage"]], mask),
        "damaged": _region_mean(minor_or_major, mask),
        "destroyed": _region_mean(scores[STATE_INDEX["destroyed"]], mask),
    }


def damage_query_score_for_mask(mask: np.ndarray, damage_state_scores: np.ndarray) -> float:
    """Single-query ``damaged or worse`` ranking score for any proposal."""
    scores = np.asarray(damage_state_scores, dtype=np.float32)
    if scores.ndim != 3 or scores.shape[0] != 4:
        raise ValueError(f"damage_state_scores must be (4,H,W), got {scores.shape}")
    if scores.shape[1:] != mask.shape:
        raise ValueError(f"damage_state_scores grid {scores.shape[1:]} != mask grid {mask.shape}")
    damaged_or_worse = scores[1:4].max(axis=0)
    return _region_mean(damaged_or_worse, mask)


def _place_tile_mask(local_mask: np.ndarray, origin: Tuple[int, int], size_hw: Tuple[int, int]) -> np.ndarray:
    """Map a tile-local mask onto the native PRE grid through its tile origin."""
    x0, y0 = int(origin[0]), int(origin[1])
    local_mask = np.asarray(local_mask, dtype=bool)
    tile_height, tile_width = local_mask.shape
    full_height, full_width = int(size_hw[0]), int(size_hw[1])
    if x0 < 0 or y0 < 0 or x0 + tile_width > full_width or y0 + tile_height > full_height:
        raise ValueError(
            f"tile mask {local_mask.shape} at origin {(x0, y0)} does not fit the native grid {size_hw}"
        )
    placed = np.zeros((full_height, full_width), dtype=bool)
    placed[y0:y0 + tile_height, x0:x0 + tile_width] = local_mask
    return placed


def build_native_instances(
    *,
    pair_id: str,
    image_id: int,
    split: str,
    size_hw: Tuple[int, int],
    prompt_captures: Sequence[Dict[str, object]],
    pred_clf: np.ndarray,
    damage_state_scores: np.ndarray,
    valid: Optional[np.ndarray] = None,
    building_support: Optional[np.ndarray] = None,
    nms_iou: float = NMS_MASK_IOU,
) -> Tuple[List[NativeInstance], Dict[str, object]]:
    """Build the per-instance records of one pair from the native forward output.

    ``prompt_captures`` is the raw list produced by the shared forward: one
    record per (tile, prompt) with the SAM3 masks/scores exactly as emitted at
    ``Sam3Processor.set_text_prompt`` time.  Captures are grouped by tile origin
    so that NMS only ever compares prompts inside one tile: tiles are
    non-overlapping, and a building that the tiling cuts in two stays two
    instances instead of being stitched back together.
    """
    size_hw = (int(size_hw[0]), int(size_hw[1]))
    stats: Dict[str, object] = {
        "prompt_captures": len(list(prompt_captures)),
        "masks_before_nms": 0,
        "masks_after_nms": 0,
        "instances_dropped_outside_domain": 0,
        "tile_origins": [],
        "nms_mask_iou": float(nms_iou),
    }

    groups: Dict[Tuple[int, int], List[Dict[str, object]]] = {}
    for capture in prompt_captures:
        origin = (int(capture["origin"][0]), int(capture["origin"][1]))
        groups.setdefault(origin, []).append(capture)

    instances: List[NativeInstance] = []
    for origin in sorted(groups):
        stats["tile_origins"].append([origin[0], origin[1]])
        placed: List[np.ndarray] = []
        prompts: List[str] = []
        class_names: List[str] = []
        scores: List[float] = []
        for capture in groups[origin]:
            masks = np.asarray(capture["masks"], dtype=bool)
            capture_scores = np.asarray(capture["scores"], dtype=np.float64).reshape(-1)
            if masks.ndim == 2:
                masks = masks[None, ...]
            if masks.shape[0] != capture_scores.size:
                raise ValueError(
                    f"capture for prompt {capture['prompt']!r} has {masks.shape[0]} masks "
                    f"but {capture_scores.size} scores"
                )
            for index in range(masks.shape[0]):
                stats["masks_before_nms"] = int(stats["masks_before_nms"]) + 1
                mask = _place_tile_mask(masks[index], origin, size_hw)
                if valid is not None:
                    mask = mask & np.asarray(valid, dtype=bool)
                if not bool(mask.any()):
                    stats["instances_dropped_outside_domain"] = (
                        int(stats["instances_dropped_outside_domain"]) + 1
                    )
                    continue
                placed.append(mask)
                prompts.append(str(capture["prompt"]))
                class_names.append(str(capture.get("class_name", "Building")))
                scores.append(float(capture_scores[index]))
        if not placed:
            continue
        kept = nms_keep(placed, scores, iou_threshold=float(nms_iou))
        stats["masks_after_nms"] = int(stats["masks_after_nms"]) + len(kept)
        for index in kept:
            mask = placed[index]
            state_scores = state_scores_for_mask(mask, damage_state_scores)
            grade = grade_from_pred_clf(pred_clf, mask)
            support_overlap = (
                float(np.asarray(building_support, dtype=bool)[mask].mean())
                if building_support is not None
                else float("nan")
            )
            instances.append(
                NativeInstance(
                    pair_id=pair_id,
                    image_id=int(image_id),
                    split=split,
                    instance_index=len(instances),
                    prompt=prompts[index],
                    class_name=class_names[index],
                    tile_origin=origin,
                    grade=grade,
                    mask=mask,
                    area=int(mask.sum()),
                    bbox_xyxy=bbox_xyxy(mask),
                    localization_score=float(scores[index]),
                    damage_query_score=damage_query_score_for_mask(mask, damage_state_scores),
                    state_scores=state_scores,
                    support_overlap=support_overlap,
                )
            )
    return instances, stats


def submission_records(instances: Iterable[NativeInstance], variant: str,
                       encode_mask=None) -> List[Dict[str, object]]:
    """COCO detection records for one evaluation variant.

    ``three_grade``  - published category per grade, ranked by the grade score.
    ``damaged_query``- every proposal under the single query category 1, ranked
                       by the ``damaged or worse`` score.
    ``class_agnostic`` - every proposal under category 1, ranked by the *native
                       localization* score only (never a damage score).
    """
    records: List[Dict[str, object]] = []
    for instance in instances:
        if variant == "three_grade":
            records.append(instance.to_detection(instance.category_id, instance.grade_score, encode_mask))
        elif variant == "damaged_query":
            records.append(instance.to_detection(1, instance.damage_query_score, encode_mask))
        elif variant == "class_agnostic":
            records.append(instance.to_detection(1, instance.localization_score, encode_mask))
        else:
            raise ValueError(f"unknown submission variant {variant!r}")
    records.sort(key=lambda record: (-float(record["score"]), int(record["image_id"])))
    return records


def write_json(path, payload) -> None:
    from pathlib import Path

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
