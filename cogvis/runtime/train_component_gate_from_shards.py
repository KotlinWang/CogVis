from __future__ import annotations

import argparse
import glob
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import torch

import eval_ovcd_benchmark as bench
import train_cacdd_ratio_threshold_component_gate as base
from train_clip_memory_threshold import add_cmct_args, add_common_args


def _merge_stats(stats_list: list[dict[str, Any]]) -> dict[str, Any]:
    if not stats_list:
        return {}
    merged = deepcopy(stats_list[0])
    for key in ("candidate_components", "positive_components", "negative_components"):
        merged[key] = int(sum(int(s.get(key, 0)) for s in stats_list))
    total = max(int(merged.get("candidate_components", 0)), 1)
    merged["positive_ratio"] = float(int(merged.get("positive_components", 0)) / total)

    class_counts: dict[str, int] = {}
    for stats in stats_list:
        for name, value in stats.get("class_component_counts", {}).items():
            class_counts[name] = class_counts.get(name, 0) + int(value)
    if class_counts:
        merged["class_component_counts"] = class_counts
    merged["merged_shards"] = len(stats_list)
    return merged


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a component gate from pre-collected shards.")
    add_common_args(parser)
    add_cmct_args(parser)
    parser.add_argument("--shards_glob", type=str, required=True)
    parser.add_argument("--val_shards_glob", type=str, default=None)
    args = parser.parse_args()
    if not args.output_component_gate_checkpoint:
        raise SystemExit("--output_component_gate_checkpoint is required")

    shard_paths = sorted(glob.glob(args.shards_glob))
    if not shard_paths:
        raise SystemExit(f"No training shards matched: {args.shards_glob}")
    val_paths = sorted(glob.glob(args.val_shards_glob)) if args.val_shards_glob else shard_paths
    if not val_paths:
        raise SystemExit(f"No validation shards matched: {args.val_shards_glob}")

    train_chunks = [torch.load(path, map_location="cpu", weights_only=False) for path in shard_paths]
    val_chunks = [torch.load(path, map_location="cpu", weights_only=False) for path in val_paths]
    x_gate = torch.cat([chunk["x"].float() for chunk in train_chunks], dim=0)
    y_gate = torch.cat([chunk["y"].float() for chunk in train_chunks], dim=0)
    x_val = torch.cat([chunk["x"].float() for chunk in val_chunks], dim=0)
    y_val = torch.cat([chunk["y"].float() for chunk in val_chunks], dim=0)

    feature_names = train_chunks[0].get("component_gate_feature_names", base.COMPONENT_GATE_FEATURE_NAMES)
    if list(feature_names) != list(base.COMPONENT_GATE_FEATURE_NAMES):
        raise SystemExit("Shard feature names do not match current component-gate feature definition.")

    device = bench.select_device()
    print(f"Using device: {bench.describe_device(device)}")
    print(f"Train features: {tuple(x_gate.shape)}, positives={int(y_gate.sum().item())}")
    print(f"Val features: {tuple(x_val.shape)}, positives={int(y_val.sum().item())}")

    model, stats = base.train_component_gate(args, x_gate, y_gate, device)
    if float(args.component_gate_keep_threshold) < 0:
        selected_threshold, threshold_stats = base.select_component_gate_threshold(
            args, model, x_val, y_val, device,
        )
        args.component_gate_keep_threshold = float(selected_threshold)
        stats["keep_threshold"] = float(selected_threshold)
        stats["threshold_selection"] = threshold_stats
    else:
        stats["keep_threshold"] = float(args.component_gate_keep_threshold)
    stats["loaded_from_checkpoint"] = False

    collect_stats = _merge_stats([chunk.get("component_gate_collect_stats", {}) for chunk in train_chunks])
    collect_stats["source_val_for_threshold"] = _merge_stats(
        [chunk.get("component_gate_collect_stats", {}) for chunk in val_chunks]
    )
    collect_stats["train_shards"] = shard_paths
    collect_stats["val_shards"] = val_paths

    output = Path(args.output_component_gate_checkpoint)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "component_gate_head": model.state_dict(),
        "component_gate_feature_names": base.COMPONENT_GATE_FEATURE_NAMES,
        "component_gate_stats": stats,
        "component_gate_collect_stats": collect_stats,
        "config": vars(args),
    }, output)
    print(f"Saved component gate checkpoint: {output}")
    print(json.dumps({
        "checkpoint": str(output),
        "component_gate_stats": stats,
        "component_gate_collect_stats": collect_stats,
    }, indent=2))


if __name__ == "__main__":
    main()
