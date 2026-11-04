#!/usr/bin/env python3
"""Run the paper-locked CogVis pipeline on all seven supported datasets."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from cogvis.commands import SEMANTIC_DATASETS
from cogvis.paths import OVCD_BENCHMARK_ROOT, RESULTS_DIR, XBD_ROOT
from cogvis.pipeline import CogVis


DATASETS = (*SEMANTIC_DATASETS, "xBD", "xBD-instances", "all")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=DATASETS)
    parser.add_argument("--benchmark-root", type=Path, default=OVCD_BENCHMARK_ROOT)
    parser.add_argument("--xbd-root", type=Path, default=XBD_ROOT)
    parser.add_argument("--output-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--eval-limit", type=int, default=0)
    parser.add_argument("--eval-offset", type=int, default=0)
    parser.add_argument("--num-chunks", type=int)
    parser.add_argument("--chunk-index", type=int)
    parser.add_argument("--save-predictions", action="store_true")
    parser.add_argument("--manifest", type=Path,
                        help="materialised input manifest JSONL (pre_rgb/post_rgb/valid_mask .npy paths); "
                             "required by --dataset xBD-instances")
    parser.add_argument("--instance-method", default="cogvis_xbd_bda_instance_adapted")
    parser.add_argument("--instance-split", default="test_cross_event")
    parser.add_argument("--device", help="CUDA_VISIBLE_DEVICES value, for example 0")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def semantic_run(
    runner: CogVis,
    dataset: str,
    output_root: Path,
    args: argparse.Namespace,
):
    dataset_dir = output_root / dataset
    prediction_dir = dataset_dir / "predictions" if args.save_predictions else None
    return runner.run_semantic(
        dataset=dataset,
        output_json=dataset_dir / "metrics.json",
        prediction_output_dir=prediction_dir,
        eval_limit=args.eval_limit,
        eval_offset=args.eval_offset,
        benchmark_root=args.benchmark_root,
        dry_run=args.dry_run,
    )


def xbd_runs(
    runner: CogVis,
    output_root: Path,
    args: argparse.Namespace,
) -> list[object]:
    xbd_dir = output_root / "xBD"
    num_chunks = args.num_chunks or (7 if args.dataset == "all" else 1)
    if num_chunks < 1:
        raise ValueError("--num-chunks must be at least 1")
    if args.chunk_index is not None:
        if not 0 <= args.chunk_index < num_chunks:
            raise ValueError("--chunk-index must be in [0, num_chunks)")
        indices = [args.chunk_index]
    else:
        indices = list(range(num_chunks))

    results: list[object] = []
    for index in indices:
        single_run = num_chunks == 1 and args.chunk_index is None
        chunk_dir = xbd_dir if single_run else xbd_dir / "chunks" / f"chunk_{index:02d}"
        output_json = (
            xbd_dir / "metrics.json"
            if single_run
            else xbd_dir / "chunks" / f"chunk_{index:02d}.json"
        )
        results.append(
            runner.run_xbd(
                output_json=output_json,
                output_dir=chunk_dir,
                eval_limit=args.eval_limit,
                eval_offset=args.eval_offset,
                num_chunks=num_chunks,
                chunk_index=index,
                xbd_root=args.xbd_root,
                save_predictions=args.save_predictions,
                dry_run=args.dry_run,
            )
        )

    if args.chunk_index is None and num_chunks > 1:
        results.append(
            runner.merge_xbd(
                chunk_glob=str(xbd_dir / "chunks" / "chunk_*.json"),
                output_json=xbd_dir / "metrics.json",
                output_dir=xbd_dir / "merged",
                dry_run=args.dry_run,
            )
        )
    return results


def instance_run(
    runner: CogVis,
    output_root: Path,
    args: argparse.Namespace,
):
    if args.manifest is None:
        raise SystemExit(
            "--dataset xBD-instances needs --manifest (materialised input manifest JSONL with "
            "pre_rgb / post_rgb / valid_mask .npy paths)"
        )
    return runner.run_xbd_instances(
        manifest=args.manifest,
        output_dir=output_root / "xBD-instances" / args.instance_method,
        method_name=args.instance_method,
        split=args.instance_split,
        limit=args.eval_limit,
        offset=args.eval_offset,
        dry_run=args.dry_run,
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    runner = CogVis(python=args.python, cuda_visible_devices=args.device)
    results: list[object] = []

    if args.dataset in SEMANTIC_DATASETS:
        results.append(semantic_run(runner, args.dataset, args.output_dir, args))
    elif args.dataset == "xBD":
        results.extend(xbd_runs(runner, args.output_dir, args))
    elif args.dataset == "xBD-instances":
        results.append(instance_run(runner, args.output_dir, args))
    else:
        for dataset in SEMANTIC_DATASETS:
            results.append(semantic_run(runner, dataset, args.output_dir, args))
        results.extend(xbd_runs(runner, args.output_dir, args))

    if args.dry_run:
        print(json.dumps(results, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
