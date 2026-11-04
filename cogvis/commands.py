"""Exact command contract for the locked 2026-07-11 paper configuration.

The 2026-11-03 release makes the paper's large-region protection threshold
(``eta_a = 0.04``) explicit on every route; all other flags are unchanged.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterable

from .paths import (
    CHANGE_ADAPTER_CHECKPOINT,
    CLIP_CACHE_DIR,
    GATE_ADAPTER_CHECKPOINT,
    LOCKED_RUNTIME,
    OVCD_BENCHMARK_ROOT,
    SAM3_CHECKPOINT,
    SCORE_ADAPTER_CHECKPOINT,
    SEMANTIC_PROMPTS,
    XBD_PROMPTS,
    XBD_ROOT,
)

SEMANTIC_DATASETS = (
    "SECOND",
    "SCSCD",
    "CLCD-512",
    "DSIFN-512",
    "LEVIR-CD-512",
    "WHU-CD-512",
)


XBD_FROZEN_MODEL_ARGS = (
    "--checkpoint",
    "--sam3_checkpoint",
    "--load_checkpoint",
    "--target_datasets",
    "--train_limit",
    "--source_val_limit",
    "--bcd_score_mode",
    "--cmct_adaptive_floor_scale",
    "--clip_cache_dir",
    "--enable_component_gate",
    "--component_gate_checkpoint",
    "--component_gate_keep_threshold",
    "--component_gate_protect_area_ratio",
    "--building_support_threshold",
    "--damage_confidence_threshold",
    "--serious_damage_threshold",
    "--tile_size",
    "--tile_stride",
    "--seed",
)


def xbd_frozen_model_args() -> list[str]:
    """The frozen xBD BDA model/threshold flags of the instance export.

    ``xbd_command`` keeps its original literal argument block: the pixel route is
    the frozen reference and must not be rewritten.  ``tests/test_xbd_instance_export.py``
    asserts that both commands still agree on every one of these flags, so the
    two entry points cannot drift into running different model configurations.
    """
    return [
        "--checkpoint", str(CHANGE_ADAPTER_CHECKPOINT),
        "--sam3_checkpoint", str(SAM3_CHECKPOINT),
        "--load_checkpoint", str(SCORE_ADAPTER_CHECKPOINT),
        "--target_datasets", "xBD",
        "--train_limit", "0",
        "--source_val_limit", "0",
        "--bcd_score_mode", "head",
        "--cmct_adaptive_floor_scale", "0.75",
        "--clip_cache_dir", str(CLIP_CACHE_DIR),
        "--enable_component_gate",
        "--component_gate_checkpoint", str(GATE_ADAPTER_CHECKPOINT),
        "--component_gate_keep_threshold", "0.60",
        "--component_gate_protect_area_ratio", "0.04",
        "--building_support_threshold", "0.10",
        "--damage_confidence_threshold", "0.10",
        "--serious_damage_threshold", "0.20",
        "--tile_size", "512",
        "--tile_stride", "512",
        "--seed", "2026",
    ]


def semantic_command(
    dataset: str,
    output_json: Path,
    prediction_output_dir: Path | None = None,
    eval_limit: int = 0,
    eval_offset: int = 0,
    benchmark_root: Path = OVCD_BENCHMARK_ROOT,
    python: str = sys.executable,
    extra_args: Iterable[str] = (),
) -> list[str]:
    if dataset not in SEMANTIC_DATASETS:
        raise ValueError(f"Unsupported paper dataset {dataset!r}; expected one of {SEMANTIC_DATASETS}")
    cmd = [
        python,
        str(LOCKED_RUNTIME / "old_adaptive_ovcd_prompt_profiles_with_predictions_20260711.py"),
        "--ovcd_prompt_profile", "paper_fixed_20260711",
        "--ovcd_prompt_profile_json", str(SEMANTIC_PROMPTS),
        "--ovcd_prompt_dataset", dataset,
        "--benchmark_root", str(benchmark_root),
        "--checkpoint", str(CHANGE_ADAPTER_CHECKPOINT),
        "--sam3_checkpoint", str(SAM3_CHECKPOINT),
        "--output_json", str(output_json),
        "--prediction_method", "clip_corrected_tau_component_gate",
        "--load_checkpoint", str(SCORE_ADAPTER_CHECKPOINT),
        "--source_dataset", "CACDD",
        "--target_datasets", dataset,
        "--train_limit", "0",
        "--source_val_limit", "0",
        "--eval_limit", str(eval_limit),
        "--eval_offset", str(eval_offset),
        "--bcd_score_mode", "head",
        "--cmct_adaptive_floor_scale", "0.75",
        "--clip_cache_dir", str(CLIP_CACHE_DIR),
        "--enable_component_gate",
        "--component_gate_checkpoint", str(GATE_ADAPTER_CHECKPOINT),
        "--component_gate_candidate_source", "semantic_proxy",
        "--component_gate_source_prompt_dataset", "CLCD-512",
        "--component_gate_source_classes", "Bareland,Grass,Road,Tree,Water,Cropland,Building",
        "--component_gate_feature_mode", "full",
        "--component_gate_keep_threshold", "0.60",
        "--component_gate_protect_area_ratio", "0.04",
        "--cmct_enable_multilabel_onepass",
        "--seed", "2026",
    ]
    if prediction_output_dir is not None:
        cmd.extend(["--prediction_output_dir", str(prediction_output_dir)])
    cmd.extend(extra_args)
    return cmd


def xbd_command(
    output_json: Path,
    output_dir: Path,
    eval_limit: int = 0,
    eval_offset: int = 0,
    num_chunks: int = 1,
    chunk_index: int = 0,
    xbd_root: Path = XBD_ROOT,
    python: str = sys.executable,
    save_predictions: bool = False,
    extra_args: Iterable[str] = (),
) -> list[str]:
    cmd = [
        python,
        str(LOCKED_RUNTIME / "evaluate_xbd_bda_prompt_profiles.py"),
        "--xbd_prompt_profile", "collapse_heavy",
        "--xbd_prompt_profile_json", str(XBD_PROMPTS),
        "--checkpoint", str(CHANGE_ADAPTER_CHECKPOINT),
        "--sam3_checkpoint", str(SAM3_CHECKPOINT),
        "--output_json", str(output_json),
        "--output_dir", str(output_dir),
        "--xbd_root", str(xbd_root),
        "--load_checkpoint", str(SCORE_ADAPTER_CHECKPOINT),
        "--target_datasets", "xBD",
        "--train_limit", "0",
        "--source_val_limit", "0",
        "--eval_limit", str(eval_limit),
        "--eval_offset", str(eval_offset),
        "--num_chunks", str(num_chunks),
        "--chunk_index", str(chunk_index),
        "--bcd_score_mode", "head",
        "--cmct_adaptive_floor_scale", "0.75",
        "--clip_cache_dir", str(CLIP_CACHE_DIR),
        "--enable_component_gate",
        "--component_gate_checkpoint", str(GATE_ADAPTER_CHECKPOINT),
        "--component_gate_keep_threshold", "0.60",
        "--component_gate_protect_area_ratio", "0.04",
        "--building_support_threshold", "0.10",
        "--damage_confidence_threshold", "0.10",
        "--serious_damage_threshold", "0.20",
        "--tile_size", "512",
        "--tile_stride", "512",
        "--seed", "2026",
    ]
    if not save_predictions:
        cmd.extend(["--no_save_vis", "--no_save_pred_clf"])
    cmd.extend(extra_args)
    return cmd


def xbd_instance_command(
    manifest: Path,
    output_dir: Path,
    method_name: str = "cogvis_xbd_bda_instance_adapted",
    split: str = "test_cross_event",
    limit: int = 0,
    offset: int = 0,
    python: str = sys.executable,
    extra_args: Iterable[str] = (),
) -> list[str]:
    """Native instance export over the frozen BDA forward (same model flags).

    ``manifest`` is the materialised input manifest (pre_rgb / post_rgb /
    valid_mask ``.npy`` paths), not the published pair manifest: decoding belongs
    to the materialisation step, so this route has a single input source.
    """
    return [
        python,
        str(LOCKED_RUNTIME / "xbd_instance_export.py"),
        "--xbd_prompt_profile", "collapse_heavy",
        "--xbd_prompt_profile_json", str(XBD_PROMPTS),
        *xbd_frozen_model_args(),
        "--manifest", str(manifest),
        "--instance-output-dir", str(output_dir),
        "--method-name", method_name,
        "--split", split,
        "--limit", str(limit),
        "--offset", str(offset),
        *extra_args,
    ]


def xbd_merge_command(
    chunk_glob: str,
    output_json: Path,
    output_dir: Path,
    python: str = sys.executable,
) -> list[str]:
    return [
        python,
        str(LOCKED_RUNTIME / "evaluate_xbd_bda_prompt_profiles.py"),
        "--xbd_prompt_profile", "collapse_heavy",
        "--xbd_prompt_profile_json", str(XBD_PROMPTS),
        "--merge_glob", chunk_glob,
        "--output_json", str(output_json),
        "--output_dir", str(output_dir),
    ]
