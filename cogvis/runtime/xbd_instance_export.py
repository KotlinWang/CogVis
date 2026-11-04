#!/usr/bin/env python3
"""Export the native CogVis xBD BDA forward as BRIGHT-style instance predictions.

Why this entry point exists
---------------------------
The frozen pixel route (:mod:`evaluate_xbd_bda_cmct`) reduces the forward to
three 8-bit PNGs per pair.  Instance-level comparison against the published xBD
three-grade COCO reference needs the *objects* and the *continuous* evidence,
which the PNG route discards.  This driver runs the **same frozen forward**
(the very same ``predict_xbd_sample_tiled`` and the same argparse defaults as
the pixel route) with an opt-in evidence sink, and writes:

``<method>.instances.jsonl``
    neutral per-instance rows: native SAM3 PRE mask (uncompressed COCO RLE),
    native localization score, per-grade damage scores, state, provenance.
``<method>.three_grade.predictions.json``
    COCO detections, category 1/2/3, ranked by the grade score.
``<method>.damaged_query.predictions.json``
    COCO detections, single query category 1, **every** proposal ranked by the
    ``damaged or worse`` score.  It is a ranking of the proposal set, never a
    filter of the three-grade argmax.
``<method>.class_agnostic.predictions.json``
    COCO detections, category 1, ranked by the *native SAM3 localization*
    score only (never a damage score).
``<method>.coverage.json``
    one record per input manifest row, including explicit ``instances: 0``
    rows, so a genuine empty scene can never be confused with a pair that was
    never run.
``<method>.provenance.json``
    the resolved configuration, the instance policy, and the weight provenance.

Inputs
------
Imagery comes from a **materialised input manifest** (``--manifest``): one JSONL
row per frozen pair giving ``pre_rgb`` / ``post_rgb`` / ``valid_mask`` ``.npy``
paths.  Materialisation is a separate CPU step owned by the run
(``xbd_instance_inputs.materialize_published_pair`` drives the published reader);
this inference route therefore has exactly one way to obtain pixels and does not
import the reader, and it never reads an answer-key field.

What an "instance" is here
--------------------------
One SAM3 text-prompt mask produced on the PRE image by the SAM3 ``building``
prompt bank of the installed xBD prompt profile, de-duplicated across those
prompts by one fixed class-agnostic mask-IoU NMS.  There is **no**
connected-component step, **no** union with the change-candidate branch, and no
damage-state filter: the union ``building_support = building_prompt | damage_candidate``
that the pixel route uses for ``pred_loc`` is only *recorded* (``support_overlap``)
and never used to form an instance.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import eval_ovcd_benchmark as bench  # noqa: E402
import evaluate_xbd_bda_cmct as xbd_eval  # noqa: E402
import train_cacdd_ratio_threshold_component_gate as base  # noqa: E402
import train_clip_memory_threshold as cmct  # noqa: E402
from clip_memory_threshold import ClipFeatureExtractor  # noqa: E402
from cmct_checkpoint_preflight import preflight_frozen_cmct_checkpoint  # noqa: E402
from native_instance_readout import (  # noqa: E402
    CATEGORY_BY_GRADE,
    DAMAGE_STATE_SCORE_DEFINITION,
    NMS_MASK_IOU,
    STATE_PROVENANCE,
    build_native_instances,
    submission_records,
)
from xbd_instance_inputs import (  # noqa: E402
    MATERIALIZED_INPUT_FIELDS,
    load_input_manifest,
    load_materialized_pair,
    select_rows,
)

METHOD_NAME = "cogvis_xbd_bda_instance_adapted"
DEFAULT_SPLIT = "test_cross_event"
SUBMISSION_VARIANTS = ("three_grade", "damaged_query", "class_agnostic")

INSTANCE_POLICY = {
    "instance_source": "SAM3 text-prompt masks on the PRE image from the frozen building prompt bank of "
                       "the installed xBD prompt profile (see frozen_route.prompt_banks.building)",
    "mask_resolution": "native: SAM3 emits masks on the tile grid it saw; mapped back to the native "
                       "PRE grid through the tile origin with no resampling",
    "prompt_deduplication": f"one fixed class-agnostic mask-IoU NMS at {NMS_MASK_IOU} across the prompts "
                            "of the building bank inside one tile",
    "sam3_instance_threshold": "Sam3Processor default confidence threshold (read from the live processor "
                               "and reported in provenance['frozen_route']['args'] sampling the same "
                               "value the model used)",
    "no_connected_components": True,
    "no_union_for_instances": True,
    "no_area_filter": True,
    "no_domain_tuning": True,
    "tile_boundary_policy": "tiles are non-overlapping (tile_stride == tile_size), so a building cut by a "
                            "tile boundary stays two instances; this is a disclosed tile-adaptation "
                            "boundary and no stitching is applied",
    "damage_state_source": STATE_PROVENANCE,
    "damage_state_field": "the same forward's pred_clf 4-class map, read inside the native mask",
}

SCORE_DEFINITIONS = {
    "localization_score": "native SAM3 object score (sigmoid(pred_logits) * presence) of the mask itself; "
                          "used only by the class_agnostic localization submission",
    "three_grade_score": "mask mean of the state score of the instance's graded state",
    "damaged_query_score": DAMAGE_STATE_SCORE_DEFINITION["damage_query"],
    "state_scores": {key: value for key, value in DAMAGE_STATE_SCORE_DEFINITION.items() if key != "damage_query"},
    "not_a_probability": "damage scores are competition-normalised SAM3 text-state scores "
                         "(competitive_posteriors target-vs-confuser), not calibrated probabilities",
}


def build_parser() -> argparse.ArgumentParser:
    """Frozen pixel-route parser plus the instance-export arguments."""
    parser = xbd_eval.build_parser()
    parser.description = __doc__
    parser.add_argument("--manifest", required=True,
                        help="materialised input manifest JSONL (pre_rgb / post_rgb / valid_mask paths)")
    parser.add_argument("--instance-output-dir", required=True, type=Path,
                        help="directory that receives the instance export (distinct from the inherited "
                             "--output_dir, which the pixel route uses for its PNGs)")
    parser.add_argument("--method-name", default=METHOD_NAME)
    parser.add_argument("--split", default=DEFAULT_SPLIT)
    parser.add_argument("--xbd_prompt_profile", default="collapse_heavy")
    parser.add_argument("--xbd_prompt_profile_json", default=str(xbd_eval.DEFAULT_REPO / "prompts.json"))
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--offset", type=int, default=0)
    return parser


# --------------------------------------------------------------------------- #
# output writers
# --------------------------------------------------------------------------- #
def write_instances_jsonl(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True, allow_nan=True) + "\n")


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=True), encoding="utf-8")


def weight_provenance(paths: Sequence[object]) -> Dict[str, object]:
    out: Dict[str, object] = {}
    for raw in paths:
        if not raw:
            continue
        path = Path(str(raw))
        entry: Dict[str, object] = {"path": str(path), "exists": path.exists()}
        if path.exists():
            stat = path.stat()
            entry.update(size_bytes=int(stat.st_size),
                         mtime_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(stat.st_mtime)))
        out[str(path)] = entry
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))
    started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    output_dir = Path(args.instance_output_dir)
    method = str(args.method_name)

    all_rows = load_input_manifest(Path(args.manifest))
    rows = select_rows(all_rows, args.limit, args.offset)
    if not rows:
        raise ValueError("the selection is empty; refusing to write a report over zero frozen pairs")

    # The frozen pixel route installs its xBD prompt profile *before* running the
    # evaluation entry point; the instance export must install the very same
    # profile, otherwise the masks, the damage state scores and therefore every
    # exported score would silently come from a different prompt bank.
    import evaluate_xbd_bda_prompt_profiles as prompt_profiles

    profile = prompt_profiles._load_profile(
        str(args.xbd_prompt_profile), str(args.xbd_prompt_profile_json)
    )
    prompt_profiles._install_profile(str(args.xbd_prompt_profile), profile)
    prompt_banks = {
        "building": {name: list(prompts) for name, prompts in xbd_eval.BUILDING_PROMPTS.items()},
        "damage": {name: list(prompts) for name, prompts in xbd_eval.DAMAGE_PROMPTS.items()},
    }

    # The argparse defaults of the frozen entry point point at a weights dir inside
    # the source tree; a run must prove the four real weight paths resolve before a
    # single tensor is loaded.
    required_weights = {
        "change adapter (--checkpoint)": args.checkpoint,
        "score adapter (--load_checkpoint)": args.load_checkpoint,
        "component gate (--component_gate_checkpoint)": args.component_gate_checkpoint,
        "frozen SAM3 (--sam3_checkpoint)": args.sam3_checkpoint,
        "frozen CLIP (--clip_cache_dir/ViT-B-16.pt)": Path(args.clip_cache_dir) / "ViT-B-16.pt",
    }
    missing_weights = [f"{name}: {path}" for name, path in required_weights.items()
                       if not Path(str(path)).exists()]
    if missing_weights:
        raise FileNotFoundError(
            "the frozen weights of this run are incomplete, so this export would not be the frozen "
            "model:\n  - " + "\n  - ".join(missing_weights)
            + "\npass the runtime weight paths explicitly (see runtime/env.sh)"
        )

    cmct_input_dim = cmct.cmct_correction_feature_dim(args)
    checkpoint_input_dim = preflight_frozen_cmct_checkpoint(
        args.load_checkpoint,
        cmct_input_dim,
        torch.load,
    )
    print(
        f"CMCT checkpoint preflight: input_dim={checkpoint_input_dim}; no retraining path can be entered",
        flush=True,
    )

    device = bench.select_device()
    print(f"Using device: {bench.describe_device(device)}", flush=True)
    processor, change_module, module_type = base.load_models(args, device)
    clipper = ClipFeatureExtractor(args.clip_model, args.clip_cache_dir, device=device)
    component_model, _, _ = cmct.load_or_train_component_gate(args, [], processor, change_module, device)
    bank, correction_model, _, corrector_stats = cmct.load_or_train_cmct(
        args, [], processor, change_module, device, clipper, component_model
    )
    print(f"Loaded change module: {module_type}; no training performed", flush=True)

    model_provenance = {
        "weights": weight_provenance([
            args.checkpoint, args.sam3_checkpoint, args.load_checkpoint, args.component_gate_checkpoint,
        ]),
        "module_type": module_type,
        "sam3_confidence_threshold": float(getattr(processor, "confidence_threshold", float("nan"))),
        "corrector_stats": {key: value for key, value in corrector_stats.items()
                            if isinstance(value, (int, float, str, bool))},
    }

    coverage: List[Dict[str, object]] = []
    instance_rows: List[Dict[str, object]] = []
    detections: Dict[str, List[Dict[str, object]]] = {variant: [] for variant in SUBMISSION_VARIANTS}
    described: List[str] = []

    for row in tqdm(rows, desc="Instance export", dynamic_ncols=True):
        pair_id = str(row["pair_id"])
        pre_image, post_image, valid, dtype_notes = load_materialized_pair(row)
        width, height = pre_image.size
        described.append(pair_id)
        record: Dict[str, object] = {
            "pair_id": pair_id,
            "image_id": int(row["image_id"]),
            "dataset_id": str(row.get("dataset_id", "xbd")),
            "event_group_id": row.get("event_group_id"),
            "width": int(width),
            "height": int(height),
            "valid_pixels": int(valid.sum()),
            "input_dtypes": dtype_notes,
        }
        item = {"name": pair_id}
        evidence: Dict[str, object] = {"size_hw": (int(height), int(width))}
        pred_full, pred_clf, pred_loc, diag = xbd_eval.predict_xbd_sample_tiled(
            args,
            item,
            processor,
            change_module,
            device,
            clipper,
            bank,
            correction_model,
            component_model,
            evidence=evidence,
            images=(pre_image, post_image),
        )
        stitched_support = np.asarray(evidence["building_support"], dtype=bool)
        if not np.array_equal(stitched_support, np.asarray(pred_loc, dtype=bool)):
            raise RuntimeError(
                f"{pair_id}: the stitched native building_support does not reproduce the frozen route's "
                "pred_loc; the evidence path has drifted from the pixel route and the instance export "
                "would not describe the frozen model"
            )

        instances, inst_stats = build_native_instances(
            pair_id=pair_id,
            image_id=int(row["image_id"]),
            split=str(args.split),
            size_hw=(int(height), int(width)),
            prompt_captures=evidence["prompt_instances"],
            pred_clf=np.asarray(evidence["pred_clf"], dtype=np.uint8),
            damage_state_scores=np.asarray(evidence["damage_state_scores"], dtype=np.float32),
            valid=valid,
            building_support=stitched_support,
            nms_iou=NMS_MASK_IOU,
        )
        for instance in instances:
            instance_rows.append(instance.instances_row(method))
        for variant in SUBMISSION_VARIANTS:
            detections[variant].extend(submission_records(instances, variant))

        grades: Dict[str, int] = {name: 0 for name in CATEGORY_BY_GRADE}
        prompts: Dict[str, int] = {}
        for instance in instances:
            grades[instance.grade] += 1
            prompts[instance.prompt] = prompts.get(instance.prompt, 0) + 1
        record.update({
            "instances": len(instances),
            "grades": grades,
            "prompt_counts": prompts,
            "masks_before_nms": inst_stats["masks_before_nms"],
            "masks_after_nms": inst_stats["masks_after_nms"],
            "instances_dropped_outside_domain": inst_stats["instances_dropped_outside_domain"],
            "tile_origins": inst_stats["tile_origins"],
            "building_support_matches_pred_loc": True,
            "building_support_pixels": int(stitched_support.sum()),
            "damage_candidate_pixels": int(np.asarray(evidence["damage_candidate"], dtype=bool).sum()),
            "pred_full_nonzero_pixels": int((np.asarray(pred_full) > 0).sum()),
            "thresholds": evidence["thresholds"],
            "tile_size": int(args.tile_size),
            "tile_stride": int(args.tile_stride),
        })
        coverage.append(record)

    if len(described) != len(rows):
        raise RuntimeError(
            f"only {len(described)} of {len(rows)} selected frozen pairs were processed; "
            "a partial export must not be written as if it covered the selection"
        )

    summary = {
        "method": method,
        "manifest": str(args.manifest),
        "manifest_rows": len(all_rows),
        "pairs_selected": len(rows),
        "pairs_processed": len(described),
        "selection_first_pair_id": str(rows[0]["pair_id"]),
        "selection_last_pair_id": str(rows[-1]["pair_id"]),
        "images_with_zero_instances": int(sum(1 for row in coverage if row.get("instances") == 0)),
        "instances_total": int(len(instance_rows)),
        "instances_by_grade": {
            name: int(sum(1 for row in instance_rows if row["state"] == name)) for name in CATEGORY_BY_GRADE
        },
        "records": coverage,
    }

    provenance = {
        "method": method,
        "semantic_components": {
            "change_localization_branch": "CMCT memory-rectified change score with a query-adapted "
                                          "threshold and the component gate",
            "grade_head": "SAM3 text-state scores on the POST image over the DAMAGE_PROMPTS bank, "
                          "competition normalised against the other damage states",
        },
        "training_provenance": {
            "change_adapter": "checkpoint named final.pt from checkpoints_v4_full_100epoch_final_20260508",
            "score_adapter": "cmct_fullcacdd_prec35_floor075_train4968_val4968_epoch100.pt",
            "gate_adapter": "cmct_component_gate_fullcacdd_semproxy_train4968_val4968_epoch100.pt",
            "statement": "the change adapter, the CMCT threshold head and the component gate were trained "
                         "on the CACDD source domain; SAM3 and CLIP stay frozen; no xBD training and no "
                         "xBD fine-tuning is involved",
            "unknowns": "the overlap between the CACDD training imagery and the xBD test pairs has not "
                        "been audited and is reported as unknown, not as zero",
            "no_training_this_run": True,
        },
        "instance_policy": INSTANCE_POLICY,
        "score_definitions": SCORE_DEFINITIONS,
        "input_contract": {
            "route": "materialised input manifest: one .npy per pre_rgb / post_rgb / valid_mask, "
                     "native PRE grid, 8-bit RGB",
            "materialisation": "owned by the run; xbd_instance_inputs.materialize_published_pair drives "
                               "the published reader Benchmark.inputs(pair_id, oracle_assets=False) on CPU",
            "domain": "the materialised valid_mask; instance masks are clipped to it and an instance with "
                      "no valid pixel is dropped and counted in coverage.json",
            "gt_fields_read": [],
            "manifest_fields_read": list(MATERIALIZED_INPUT_FIELDS),
        },
        "frozen_route": {
            "entry": "cogvis/runtime/evaluate_xbd_bda_cmct.py::predict_xbd_sample_tiled",
            "prompt_profile": str(args.xbd_prompt_profile),
            "prompt_banks": prompt_banks,
            "args": {key: (str(value) if isinstance(value, Path) else value)
                     for key, value in sorted(vars(args).items())},
        },
        "model": model_provenance,
        "run": {
            "started_utc": started,
            "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "host": platform.node(),
            "python": sys.version.split()[0],
            "argv": sys.argv,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>"),
        },
    }

    write_instances_jsonl(output_dir / f"{method}.instances.jsonl", instance_rows)
    for variant in SUBMISSION_VARIANTS:
        write_json(output_dir / f"{method}.{variant}.predictions.json", detections[variant])
    write_json(output_dir / f"{method}.coverage.json", summary)
    write_json(output_dir / f"{method}.provenance.json", provenance)
    print(f"pairs={len(described)} instances={len(instance_rows)} output={output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
