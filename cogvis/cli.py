"""Command-line interface for paper-locked CogVis inference."""

from __future__ import annotations

import argparse
import json
import shlex
import sys
from pathlib import Path

from .commands import SEMANTIC_DATASETS
from .paths import RESULTS_DIR
from .pipeline import CogVis


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CogVis paper-locked inference")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--device", default=None, help="CUDA_VISIBLE_DEVICES value, e.g. 0")
    parser.add_argument("--dry-run", action="store_true")
    sub = parser.add_subparsers(dest="task", required=True)

    semantic = sub.add_parser("semantic", help="SECOND/SCSCD/binary-CD evaluation")
    semantic.add_argument("--dataset", required=True, choices=SEMANTIC_DATASETS)
    semantic.add_argument("--output-json", type=Path)
    semantic.add_argument("--prediction-output-dir", type=Path)
    semantic.add_argument("--eval-limit", type=int, default=0)
    semantic.add_argument("--eval-offset", type=int, default=0)
    semantic.add_argument("runtime_args", nargs=argparse.REMAINDER)

    xbd = sub.add_parser("xbd", help="xBD building damage assessment")
    xbd.add_argument("--output-json", type=Path)
    xbd.add_argument("--output-dir", type=Path)
    xbd.add_argument("--eval-limit", type=int, default=0)
    xbd.add_argument("--eval-offset", type=int, default=0)
    xbd.add_argument("--num-chunks", type=int, default=1)
    xbd.add_argument("--chunk-index", type=int, default=0)
    xbd.add_argument("--save-predictions", action="store_true")
    xbd.add_argument("runtime_args", nargs=argparse.REMAINDER)

    merge = sub.add_parser("xbd-merge", help="merge xBD chunk metrics")
    merge.add_argument("--chunk-glob", required=True)
    merge.add_argument("--output-json", type=Path, required=True)
    merge.add_argument("--output-dir", type=Path, required=True)
    return parser


def _clean_runtime_args(values: list[str]) -> list[str]:
    return values[1:] if values[:1] == ["--"] else values


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    runner = CogVis(python=args.python, cuda_visible_devices=args.device)
    if args.task == "semantic":
        stem = args.dataset.lower().replace("-", "_")
        output_json = args.output_json or RESULTS_DIR / "runs" / stem / "metrics.json"
        result = runner.run_semantic(
            dataset=args.dataset,
            output_json=output_json,
            prediction_output_dir=args.prediction_output_dir,
            eval_limit=args.eval_limit,
            eval_offset=args.eval_offset,
            extra_args=_clean_runtime_args(args.runtime_args),
            dry_run=args.dry_run,
        )
    elif args.task == "xbd":
        output_dir = args.output_dir or RESULTS_DIR / "runs" / "xbd"
        output_json = args.output_json or output_dir / "metrics.json"
        result = runner.run_xbd(
            output_json=output_json,
            output_dir=output_dir,
            eval_limit=args.eval_limit,
            eval_offset=args.eval_offset,
            num_chunks=args.num_chunks,
            chunk_index=args.chunk_index,
            save_predictions=args.save_predictions,
            extra_args=_clean_runtime_args(args.runtime_args),
            dry_run=args.dry_run,
        )
    else:
        result = runner.merge_xbd(
            chunk_glob=args.chunk_glob,
            output_json=args.output_json,
            output_dir=args.output_dir,
            dry_run=args.dry_run,
        )
    if args.dry_run:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
