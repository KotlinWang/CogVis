"""Torch-free evidence plumbing of the CogVis xBD BDA forward.

``evaluate_xbd_bda_cmct`` records native evidence per tile; this module owns the
pure-numpy half of that contract (allocating the full-image buffers, flattening
the raw SAM3 grounding snapshots and placing one tile's evidence onto the native
PRE grid) so it can be reasoned about and tested without torch or a GPU.

Tiles are non-overlapping whenever evidence is requested (the forward refuses
overlapping tiles), so placement is exact and no blending rule is invented.
``pred_clf`` is deliberately *not* stitched here: the tiled route already derives
one image-level ``pred_clf`` by tile voting and that array is what the frozen
pixel route writes to disk, so it is carried through verbatim.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

CONTINUOUS_EVIDENCE_KEYS = ("change_score", "bcd_score", "building_score")
BOOLEAN_EVIDENCE_KEYS = ("damage_candidate", "building_support")


def new_image_evidence(size_hw: Tuple[int, int]) -> Dict[str, object]:
    """Empty full-image evidence buffers for one pair, on the native PRE grid."""
    height, width = int(size_hw[0]), int(size_hw[1])
    record: Dict[str, object] = {
        "size_hw": (height, width),
        "tiles": [],
        "prompt_instances": [],
        "thresholds": [],
        "damage_state_scores": np.zeros((4, height, width), dtype=np.float32),
    }
    for key in CONTINUOUS_EVIDENCE_KEYS:
        record[key] = np.zeros((height, width), dtype=np.float32)
    for key in BOOLEAN_EVIDENCE_KEYS:
        record[key] = np.zeros((height, width), dtype=bool)
    return record


def numpy_prompt_instances(
    captures: Optional[Dict[str, list]],
    origin: Tuple[int, int],
) -> List[Dict[str, object]]:
    """Flatten raw per-prompt SAM3 grounding snapshots into plain numpy records.

    ``captures`` maps a class name to the snapshots that
    ``posterior_ovcd.prompt_score_from_state(capture=...)`` appended while the
    prompt was still live in the SAM3 state.  ``origin`` is the tile origin on the
    native PRE grid, which is what lets a tile-local mask be placed later.
    """
    if not captures:
        return []
    flat: List[Dict[str, object]] = []
    for class_name, records in captures.items():
        for record in records:
            flat.append({
                "prompt": str(record["prompt"]),
                "class_name": str(class_name),
                "origin": [int(origin[0]), int(origin[1])],
                "masks": np.asarray(record["masks"], dtype=bool),
                "scores": np.asarray(record["scores"], dtype=np.float32).reshape(-1),
                "boxes": np.asarray(record["boxes"], dtype=np.float32).reshape(-1, 4),
            })
    return flat


def accumulate_tile_evidence(
    evidence: Dict[str, object],
    tile_evidence: Dict[str, object],
    origin: Tuple[int, int],
    size_hw: Tuple[int, int],
) -> None:
    """Place one tile's native evidence onto the full-image grid."""
    x0, y0 = int(origin[0]), int(origin[1])
    tile_height, tile_width = int(size_hw[0]), int(size_hw[1])
    height, width = evidence["size_hw"]
    if x0 < 0 or y0 < 0 or x0 + tile_width > width or y0 + tile_height > height:
        raise ValueError(
            f"tile {size_hw} at origin {(x0, y0)} does not fit the native grid {(height, width)}"
        )
    target = np.s_[y0:y0 + tile_height, x0:x0 + tile_width]
    evidence["tiles"].append({"origin": [x0, y0], "size_hw": [tile_height, tile_width]})
    for key in CONTINUOUS_EVIDENCE_KEYS:
        evidence[key][target] = np.asarray(tile_evidence[key], dtype=np.float32)
    evidence["damage_state_scores"][..., y0:y0 + tile_height, x0:x0 + tile_width] = np.asarray(
        tile_evidence["damage_state_scores"], dtype=np.float32
    )
    for key in BOOLEAN_EVIDENCE_KEYS:
        evidence[key][target] = np.asarray(tile_evidence[key], dtype=bool)
    evidence["prompt_instances"].extend(
        numpy_prompt_instances(tile_evidence.pop("prompt_captures", None), origin)
    )
    evidence["thresholds"].append(tile_evidence["thresholds"][0])


def finish_single_image_evidence(evidence: Dict[str, object], origin: Tuple[int, int]) -> None:
    """Finish the evidence of a pair that ran without tiling.

    ``thresholds`` already holds the single tile's record, set by
    ``predict_xbd_sample`` itself.
    """
    height, width = evidence["size_hw"]
    evidence["prompt_instances"] = numpy_prompt_instances(
        evidence.pop("prompt_captures", None), origin
    )
    evidence["tiles"] = [{"origin": [int(origin[0]), int(origin[1])], "size_hw": [height, width]}]
