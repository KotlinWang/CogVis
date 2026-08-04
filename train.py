#!/usr/bin/env python3
"""Train CogVis Change, Score, and Gate Adapters in the paper's stage order."""

from __future__ import annotations

import argparse
import json
import os
import random
import runpy
import subprocess
import sys
from pathlib import Path
from typing import Iterable

from cogvis.paths import (
    CHANGE_ADAPTER_CHECKPOINT,
    CLIP_CACHE_DIR,
    LOCKED_RUNTIME,
    OVCD_BENCHMARK_ROOT,
    RESULTS_DIR,
    SAM3_CHECKPOINT,
)


ROOT = Path(__file__).resolve().parent
DEFAULT_CACHE_ROOT = Path(
    os.environ.get("COGVIS_FPN_CACHE_ROOT", str(ROOT / "weights"))
).expanduser()
DEFAULT_TRAIN_ROOT = RESULTS_DIR / "training"
SOURCE_CLASSES = "Bareland,Grass,Road,Tree,Water,Cropland,Building"


def _seeded_worker(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--seed", type=int, required=True)
    args, remainder = parser.parse_known_args(argv)
    if remainder[:1] == ["--"]:
        remainder = remainder[1:]
    if not remainder:
        raise SystemExit("missing locked runtime script")

    import numpy as np
    import torch

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True, warn_only=True)

    script = Path(remainder[0]).resolve()
    sys.path.insert(0, str(script.parent))
    original = sys.argv
    sys.argv = [str(script), *remainder[1:]]
    try:
        runpy.run_path(str(script), run_name="__main__")
    finally:
        sys.argv = original
    return 0


def _seeded_command(
    python: str,
    seed: int,
    script: str,
    arguments: Iterable[str],
) -> list[str]:
    return [
        python,
        str(Path(__file__).resolve()),
        "_seeded",
        "--seed",
        str(seed),
        "--",
        str(LOCKED_RUNTIME / script),
        *arguments,
    ]


def _run(commands: list[list[str]], *, dry_run: bool, device: str | None, seed: int) -> None:
    if dry_run:
        print(json.dumps(commands, ensure_ascii=False, indent=2))
        return
    env = os.environ.copy()
    env["PYTHONHASHSEED"] = str(seed)
    env.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    if device is not None:
        env["CUDA_VISIBLE_DEVICES"] = device
    for command in commands:
        subprocess.run(command, check=True, env=env)


def _ensure_link(link: Path, target: Path) -> None:
    target = target.expanduser().resolve()
    if not target.exists():
        raise FileNotFoundError(target)
    if os.path.lexists(link):
        if link.is_symlink() and link.resolve() == target:
            return
        raise FileExistsError(f"refusing to replace existing input path: {link}")
    link.parent.mkdir(parents=True, exist_ok=True)
    link.symlink_to(target, target_is_directory=target.is_dir())


def _prepare_change_inputs(output_dir: Path, cache_root: Path, label_dir: Path) -> tuple[Path, Path]:
    input_root = output_dir / "_inputs"
    cache_view = input_root / "cache"
    dataset_view = input_root / "dataset"
    _ensure_link(cache_view / "fpn_cache", cache_root / "fpn_cache")
    _ensure_link(dataset_view / "train" / "label", label_dir)
    return cache_view, dataset_view


def _change_paths(output_dir: Path) -> tuple[Path, Path]:
    return output_dir / "_inputs" / "cache", output_dir / "_inputs" / "dataset"


def change_adapter_command(
    *,
    python: str,
    seed: int,
    output_dir: Path,
    sam3_checkpoint: Path,
    samples: int,
    epochs: int,
    batch_size: int,
    workers: int,
) -> list[str]:
    cache_view, dataset_view = _change_paths(output_dir)
    arguments = [
        "--phase", "train",
        "--sam3_checkpoint", str(sam3_checkpoint),
        "--dataset_root", str(dataset_view),
        "--cache_dir", str(cache_view),
        "--save_dir", str(output_dir),
        "--crop_size", "512",
        "--batch_size", str(batch_size),
        "--num_workers", str(workers),
        "--prefetch_factor", "1",
        "--epochs", str(epochs),
        "--save_every", "10",
        "--max_samples", str(samples),
        "--accum_steps", "1",
        "--lr", "0.0005",
        "--weight_decay", "0.01",
        "--module_type", "temporal_gate_v4",
        "--lora_rank", "16",
        "--lora_alpha", "16.0",
        "--head_hidden_dim", "96",
        "--residual_clip_ratio", "0.25",
        "--gain_init", "1.0",
        "--gain_min", "0.75",
        "--gain_max", "1.25",
        "--gate_floor", "0.02",
        "--gate_temperature", "1.0",
        "--v4_active_levels", "2",
        "--v4_gate_level", "2",
        "--v4_loss_levels", "active",
        "--native_logit_training",
        "--head_lr_mult", "10.0",
        "--lambda_feature", "0.10",
        "--lambda_change", "1.50",
        "--lambda_mask_prior", "0.25",
        "--lambda_logit_consist", "0.10",
        "--mask_positive_margin", "0.08",
        "--mask_negative_margin", "0.02",
        "--supervise_at_logit_resolution",
        "--amp",
        "--amp_dtype", "bf16",
        "--cast_fpn_to_amp",
        "--load_fpn_levels", "active",
        "--no_shuffle",
        "--disable_tqdm",
    ]
    return _seeded_command(python, seed, "train_change_adapter.py", arguments)


def _collection_ranges(samples: int, shards: int) -> list[tuple[int, int]]:
    if samples < 1 or shards < 1:
        raise ValueError("samples and shards must be positive")
    base, extra = divmod(samples, shards)
    ranges = []
    offset = 0
    for index in range(shards):
        limit = base + (1 if index < extra else 0)
        if limit:
            ranges.append((offset, limit))
        offset += limit
    return ranges


def _collector_common(
    benchmark_root: Path,
    change_checkpoint: Path,
    sam3_checkpoint: Path,
    clip_cache_dir: Path,
    output_json: Path,
    seed: int,
) -> list[str]:
    return [
        "--benchmark_root", str(benchmark_root),
        "--checkpoint", str(change_checkpoint),
        "--sam3_checkpoint", str(sam3_checkpoint),
        "--clip_cache_dir", str(clip_cache_dir),
        "--output_json", str(output_json),
        "--source_dataset", "CACDD",
        "--target_datasets", "SECOND",
        "--bcd_score_mode", "head",
        "--seed", str(seed),
    ]


def score_adapter_plan(
    *,
    python: str,
    seed: int,
    benchmark_root: Path,
    change_checkpoint: Path,
    sam3_checkpoint: Path,
    clip_cache_dir: Path,
    output_dir: Path,
    samples: int,
    shards: int,
    epochs: int,
    shards_glob: str | None,
    reuse_shards: bool,
) -> list[list[str]]:
    commands: list[list[str]] = []
    shard_dir = output_dir / "shards"
    if shards_glob is None:
        shards_glob = str(shard_dir / "score_shard_*.pt")
        if not reuse_shards:
            for index, (offset, limit) in enumerate(_collection_ranges(samples, shards)):
                output = shard_dir / f"score_shard_{index:02d}.pt"
                arguments = _collector_common(
                    benchmark_root,
                    change_checkpoint,
                    sam3_checkpoint,
                    clip_cache_dir,
                    shard_dir / f"collect_{index:02d}.json",
                    seed,
                )
                arguments.extend([
                    "--shard_output", str(output),
                    "--train_limit", str(limit),
                    "--train_offset", str(offset),
                    "--source_val_limit", "0",
                    "--eval_limit", "0",
                    "--cmct_min_recall", "0.35",
                    "--cmct_min_precision", "0.35",
                    "--cmct_threshold_grid", "0.02:0.80:0.02",
                    "--cmct_threshold_min", "0.02",
                    "--cmct_threshold_max", "0.80",
                    "--cmct_adaptive_floor_scale", "0.75",
                    "--clip_memory_source_classes", "union_remote",
                    "--clip_memory_topk", "16",
                    "--clip_memory_temperature", "0.07",
                    "--clip_memory_scene_weight", "1.0",
                    "--clip_memory_delta_weight", "0.7",
                    "--clip_memory_text_weight", "1.0",
                    "--clip_memory_stats_weight", "0.25",
                ])
                commands.append(
                    _seeded_command(python, seed, "collect_cmct_memory_shard.py", arguments)
                )

    fit_args = _collector_common(
        benchmark_root,
        change_checkpoint,
        sam3_checkpoint,
        clip_cache_dir,
        output_dir / "fit.json",
        seed,
    )
    fit_args.extend([
        "--output_checkpoint", str(output_dir / "score_adapter.pt"),
        "--shards_glob", shards_glob,
        "--train_limit", str(samples),
        "--train_offset", "0",
        "--source_val_limit", str(samples),
        "--source_val_offset", "0",
        "--eval_limit", "0",
        "--cmct_min_recall", "0.35",
        "--cmct_min_precision", "0.35",
        "--cmct_threshold_grid", "0.02:0.80:0.02",
        "--cmct_threshold_min", "0.02",
        "--cmct_threshold_max", "0.80",
        "--cmct_adaptive_floor_scale", "0.75",
        "--clip_memory_source_classes", "union_remote",
        "--clip_memory_topk", "16",
        "--clip_memory_temperature", "0.07",
        "--clip_memory_scene_weight", "1.0",
        "--clip_memory_delta_weight", "0.7",
        "--clip_memory_text_weight", "1.0",
        "--clip_memory_stats_weight", "0.25",
        "--clip_corrector_epochs", str(epochs),
        "--clip_corrector_batch_size", "128",
        "--clip_corrector_hidden_dim", "64",
        "--clip_corrector_dropout", "0.05",
        "--clip_corrector_lr", "0.001",
        "--clip_corrector_weight_decay", "0.0001",
        "--clip_corrector_max_delta", "0.20",
        "--cmct_retrieval_batch_size", "256",
    ])
    commands.append(_seeded_command(python, seed, "train_cmct_from_memory_shards.py", fit_args))
    return commands


def gate_adapter_plan(
    *,
    python: str,
    seed: int,
    benchmark_root: Path,
    change_checkpoint: Path,
    sam3_checkpoint: Path,
    clip_cache_dir: Path,
    output_dir: Path,
    samples: int,
    shards: int,
    epochs: int,
    shards_glob: str | None,
    reuse_shards: bool,
) -> list[list[str]]:
    commands: list[list[str]] = []
    shard_dir = output_dir / "shards"
    if shards_glob is None:
        shards_glob = str(shard_dir / "gate_shard_*.pt")
        if not reuse_shards:
            for index, (offset, limit) in enumerate(_collection_ranges(samples, shards)):
                output = shard_dir / f"gate_shard_{index:02d}.pt"
                arguments = _collector_common(
                    benchmark_root,
                    change_checkpoint,
                    sam3_checkpoint,
                    clip_cache_dir,
                    shard_dir / f"collect_{index:02d}.json",
                    seed,
                )
                arguments.extend([
                    "--shard_output", str(output),
                    "--train_limit", "0",
                    "--source_val_limit", "0",
                    "--eval_limit", "0",
                    "--component_gate_train_limit", str(limit),
                    "--component_gate_train_offset", str(offset),
                    "--component_gate_candidate_source", "semantic_proxy",
                    "--component_gate_source_prompt_dataset", "CLCD-512",
                    "--component_gate_source_classes", SOURCE_CLASSES,
                    "--component_gate_feature_mode", "full",
                    "--component_gate_bcd_threshold_mode", "adaptive",
                    "--component_gate_topk_ratio", "0.2",
                    "--component_gate_min_area", "8",
                    "--component_gate_max_area", "262144",
                ])
                commands.append(
                    _seeded_command(python, seed, "collect_component_gate_shard.py", arguments)
                )

    fit_args = _collector_common(
        benchmark_root,
        change_checkpoint,
        sam3_checkpoint,
        clip_cache_dir,
        output_dir / "fit.json",
        seed,
    )
    fit_args.extend([
        "--output_component_gate_checkpoint", str(output_dir / "gate_adapter.pt"),
        "--shards_glob", shards_glob,
        "--val_shards_glob", shards_glob,
        "--train_limit", "0",
        "--source_val_limit", "0",
        "--eval_limit", "1",
        "--eval_offset", "0",
        "--component_gate_candidate_source", "semantic_proxy",
        "--component_gate_source_prompt_dataset", "CLCD-512",
        "--component_gate_source_classes", SOURCE_CLASSES,
        "--component_gate_feature_mode", "full",
        "--component_gate_epochs", str(epochs),
        "--component_gate_batch_size", "4096",
        "--component_gate_lr", "0.001",
        "--component_gate_weight_decay", "0.0001",
        "--component_gate_keep_threshold", "-1",
        "--component_gate_threshold_grid", "0.20:0.80:0.05",
        "--component_gate_select_metric", "IoU",
        "--component_gate_min_recall", "0.45",
    ])
    commands.append(
        _seeded_command(python, seed, "train_component_gate_from_shards.py", fit_args)
    )
    return commands


# Public helpers follow the paper's three adapter names. The short historical
# names remain aliases so external launchers keep their prior behavior.
change_command = change_adapter_command
score_plan = score_adapter_plan
gate_plan = gate_adapter_plan


def _add_execution_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--device", help="CUDA_VISIBLE_DEVICES value, for example 0")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--dry-run", action="store_true")


def _add_collection_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--benchmark-root", type=Path, default=OVCD_BENCHMARK_ROOT)
    parser.add_argument("--change-checkpoint", type=Path, default=CHANGE_ADAPTER_CHECKPOINT)
    parser.add_argument("--sam3-checkpoint", type=Path, default=SAM3_CHECKPOINT)
    parser.add_argument("--clip-cache-dir", type=Path, default=CLIP_CACHE_DIR)
    parser.add_argument("--samples", type=int, default=4968)
    parser.add_argument("--shards", type=int, default=8)
    parser.add_argument("--shards-glob")
    parser.add_argument("--reuse-shards", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="adapter", required=True)

    change_adapter = sub.add_parser(
        "change-adapter",
        aliases=["change"],
        help="optimize the SCP Change Adapter",
    )
    _add_execution_args(change_adapter)
    change_adapter.add_argument("--benchmark-root", type=Path, default=OVCD_BENCHMARK_ROOT)
    change_adapter.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    change_adapter.add_argument("--label-dir", type=Path)
    change_adapter.add_argument("--sam3-checkpoint", type=Path, default=SAM3_CHECKPOINT)
    change_adapter.add_argument("--output-dir", type=Path, default=DEFAULT_TRAIN_ROOT / "change")
    change_adapter.add_argument("--samples", type=int, default=4968)
    change_adapter.add_argument("--epochs", type=int, default=100)
    change_adapter.add_argument("--batch-size", type=int, default=16)
    change_adapter.add_argument("--workers", type=int, default=4)

    score_adapter = sub.add_parser(
        "score-adapter",
        aliases=["score"],
        help="construct SMC memory and optimize the Score Adapter",
    )
    _add_execution_args(score_adapter)
    _add_collection_args(score_adapter)
    score_adapter.add_argument("--output-dir", type=Path, default=DEFAULT_TRAIN_ROOT / "score")
    score_adapter.add_argument("--epochs", type=int, default=100)

    gate_adapter = sub.add_parser(
        "gate-adapter",
        aliases=["gate"],
        help="collect ARF candidates and optimize the Gate Adapter",
    )
    _add_execution_args(gate_adapter)
    _add_collection_args(gate_adapter)
    gate_adapter.add_argument("--output-dir", type=Path, default=DEFAULT_TRAIN_ROOT / "gate")
    gate_adapter.add_argument("--epochs", type=int, default=100)

    all_adapters = sub.add_parser(
        "all-adapters",
        aliases=["all"],
        help="optimize Change, Score, and Gate Adapters sequentially",
    )
    _add_execution_args(all_adapters)
    all_adapters.add_argument("--benchmark-root", type=Path, default=OVCD_BENCHMARK_ROOT)
    all_adapters.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    all_adapters.add_argument("--label-dir", type=Path)
    all_adapters.add_argument("--sam3-checkpoint", type=Path, default=SAM3_CHECKPOINT)
    all_adapters.add_argument("--clip-cache-dir", type=Path, default=CLIP_CACHE_DIR)
    all_adapters.add_argument("--output-dir", type=Path, default=DEFAULT_TRAIN_ROOT / "all")
    all_adapters.add_argument("--samples", type=int, default=4968)
    all_adapters.add_argument("--shards", type=int, default=8)
    all_adapters.add_argument("--change-epochs", type=int, default=100)
    all_adapters.add_argument("--score-epochs", type=int, default=100)
    all_adapters.add_argument("--gate-epochs", type=int, default=100)
    all_adapters.add_argument("--batch-size", type=int, default=16)
    all_adapters.add_argument("--workers", type=int, default=4)
    all_adapters.add_argument("--reuse-shards", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    commands: list[list[str]]
    args.adapter = {
        "change": "change-adapter",
        "score": "score-adapter",
        "gate": "gate-adapter",
        "all": "all-adapters",
    }.get(args.adapter, args.adapter)

    if args.adapter == "change-adapter":
        label_dir = args.label_dir or args.benchmark_root / "CACDD" / "label"
        if not args.dry_run:
            _prepare_change_inputs(args.output_dir, args.cache_root, label_dir)
        commands = [change_adapter_command(
            python=args.python,
            seed=args.seed,
            output_dir=args.output_dir,
            sam3_checkpoint=args.sam3_checkpoint,
            samples=args.samples,
            epochs=args.epochs,
            batch_size=args.batch_size,
            workers=args.workers,
        )]
    elif args.adapter == "score-adapter":
        commands = score_adapter_plan(
            python=args.python,
            seed=args.seed,
            benchmark_root=args.benchmark_root,
            change_checkpoint=args.change_checkpoint,
            sam3_checkpoint=args.sam3_checkpoint,
            clip_cache_dir=args.clip_cache_dir,
            output_dir=args.output_dir,
            samples=args.samples,
            shards=args.shards,
            epochs=args.epochs,
            shards_glob=args.shards_glob,
            reuse_shards=args.reuse_shards,
        )
    elif args.adapter == "gate-adapter":
        commands = gate_adapter_plan(
            python=args.python,
            seed=args.seed,
            benchmark_root=args.benchmark_root,
            change_checkpoint=args.change_checkpoint,
            sam3_checkpoint=args.sam3_checkpoint,
            clip_cache_dir=args.clip_cache_dir,
            output_dir=args.output_dir,
            samples=args.samples,
            shards=args.shards,
            epochs=args.epochs,
            shards_glob=args.shards_glob,
            reuse_shards=args.reuse_shards,
        )
    else:
        change_dir = args.output_dir / "change"
        label_dir = args.label_dir or args.benchmark_root / "CACDD" / "label"
        if not args.dry_run:
            _prepare_change_inputs(change_dir, args.cache_root, label_dir)
        change_checkpoint = change_dir / "final.pt"
        commands = [change_adapter_command(
            python=args.python,
            seed=args.seed,
            output_dir=change_dir,
            sam3_checkpoint=args.sam3_checkpoint,
            samples=args.samples,
            epochs=args.change_epochs,
            batch_size=args.batch_size,
            workers=args.workers,
        )]
        commands.extend(score_adapter_plan(
            python=args.python,
            seed=args.seed,
            benchmark_root=args.benchmark_root,
            change_checkpoint=change_checkpoint,
            sam3_checkpoint=args.sam3_checkpoint,
            clip_cache_dir=args.clip_cache_dir,
            output_dir=args.output_dir / "score",
            samples=args.samples,
            shards=args.shards,
            epochs=args.score_epochs,
            shards_glob=None,
            reuse_shards=args.reuse_shards,
        ))
        commands.extend(gate_adapter_plan(
            python=args.python,
            seed=args.seed,
            benchmark_root=args.benchmark_root,
            change_checkpoint=change_checkpoint,
            sam3_checkpoint=args.sam3_checkpoint,
            clip_cache_dir=args.clip_cache_dir,
            output_dir=args.output_dir / "gate",
            samples=args.samples,
            shards=args.shards,
            epochs=args.gate_epochs,
            shards_glob=None,
            reuse_shards=args.reuse_shards,
        ))

    _run(commands, dry_run=args.dry_run, device=args.device, seed=args.seed)
    return 0


if __name__ == "__main__":
    if sys.argv[1:2] == ["_seeded"]:
        raise SystemExit(_seeded_worker(sys.argv[2:]))
    raise SystemExit(main())
