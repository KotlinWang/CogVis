from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

import eval_ovcd_benchmark as bench
import train_cacdd_ratio_threshold_component_gate as base
from train_clip_memory_threshold import (
    ClipFeatureExtractor,
    add_cmct_args,
    add_common_args,
    build_memory_rows,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect a CA-CDD CMCT memory-row shard.")
    add_common_args(parser)
    add_cmct_args(parser)
    parser.add_argument("--shard_output", type=str, required=True)
    args = parser.parse_args()

    device = bench.select_device()
    print(f"Using device: {bench.describe_device(device)}")
    processor, change_module, module_type = base.load_models(args, device)
    print(f"Loaded change module: {module_type}")
    clipper = ClipFeatureExtractor(args.clip_model, args.clip_cache_dir, device=device)
    print(f"Loaded CLIP: {args.clip_model}")

    root = Path(args.benchmark_root)
    source_spec = bench.BINARY_DATASETS[args.source_dataset]
    all_source = bench.list_binary_items(root, source_spec)
    items = base.split_items(all_source, int(args.train_limit), int(args.train_offset))
    print(
        f"{args.source_dataset}: CMCT shard {len(items)} / all {len(all_source)} "
        f"(offset={args.train_offset}, limit={args.train_limit})"
    )

    rows, memory_stats = build_memory_rows(
        args,
        items,
        processor,
        change_module,
        device,
        clipper,
        component_model=None,
    )
    out = {
        "rows": rows,
        "clip_memory_stats": memory_stats,
        "config": vars(args),
        "module_type": module_type,
    }
    output = Path(args.shard_output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, output)
    print(json.dumps({
        "shard_output": str(output),
        "num_rows": int(len(rows)),
        "clip_memory_stats": memory_stats,
    }, indent=2))


if __name__ == "__main__":
    main()
