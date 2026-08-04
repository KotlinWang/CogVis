from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

import eval_ovcd_benchmark as bench
import train_cacdd_ratio_threshold_component_gate as base
from train_clip_memory_threshold import add_cmct_args, add_common_args


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect a CA-CDD component-gate feature shard.")
    add_common_args(parser)
    add_cmct_args(parser)
    parser.add_argument("--shard_output", type=str, required=True)
    args = parser.parse_args()

    device = bench.select_device()
    print(f"Using device: {bench.describe_device(device)}")
    processor, change_module, module_type = base.load_models(args, device)
    print(f"Loaded change module: {module_type}")

    root = Path(args.benchmark_root)
    source_spec = bench.BINARY_DATASETS[args.source_dataset]
    all_source = bench.list_binary_items(root, source_spec)
    items = base.split_items(
        all_source,
        int(args.component_gate_train_limit),
        int(args.component_gate_train_offset),
    )
    print(
        f"{args.source_dataset}: shard {len(items)} / all {len(all_source)} "
        f"(offset={args.component_gate_train_offset}, limit={args.component_gate_train_limit})"
    )

    x_gate, y_gate, collect_stats = base.collect_cacdd_component_gate_examples(
        args, items, processor, change_module, device,
    )
    out = {
        "x": x_gate.cpu(),
        "y": y_gate.cpu(),
        "component_gate_collect_stats": collect_stats,
        "component_gate_feature_names": base.COMPONENT_GATE_FEATURE_NAMES,
        "config": vars(args),
        "module_type": module_type,
    }
    output = Path(args.shard_output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, output)
    summary = {
        "shard_output": str(output),
        "features": list(x_gate.shape),
        "labels": list(y_gate.shape),
        "stats": collect_stats,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
