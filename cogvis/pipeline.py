"""Perception-memory-verification execution facade."""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from .commands import semantic_command, xbd_command, xbd_merge_command
from .paths import OVCD_BENCHMARK_ROOT, XBD_ROOT
from .paths import require_runtime_layout


@dataclass(frozen=True)
class CogVis:
    """Run the locked CogVis perception-memory-verification pipeline."""

    python: str
    cuda_visible_devices: str | None = None

    def _run(
        self, command: Sequence[str], dry_run: bool = False
    ) -> subprocess.CompletedProcess | list[str]:
        if dry_run:
            return list(command)
        require_runtime_layout()
        env = os.environ.copy()
        if self.cuda_visible_devices is not None:
            env["CUDA_VISIBLE_DEVICES"] = self.cuda_visible_devices
        return subprocess.run(list(command), check=True, env=env)

    def run_semantic(
        self,
        dataset: str,
        output_json: Path,
        prediction_output_dir: Path | None = None,
        eval_limit: int = 0,
        eval_offset: int = 0,
        benchmark_root: Path = OVCD_BENCHMARK_ROOT,
        extra_args: Sequence[str] = (),
        dry_run: bool = False,
    ):
        if not dry_run:
            output_json.parent.mkdir(parents=True, exist_ok=True)
            if prediction_output_dir is not None:
                prediction_output_dir.mkdir(parents=True, exist_ok=True)
        return self._run(
            semantic_command(
                dataset=dataset,
                output_json=output_json,
                prediction_output_dir=prediction_output_dir,
                eval_limit=eval_limit,
                eval_offset=eval_offset,
                benchmark_root=benchmark_root,
                python=self.python,
                extra_args=extra_args,
            ),
            dry_run=dry_run,
        )

    def run_xbd(
        self,
        output_json: Path,
        output_dir: Path,
        eval_limit: int = 0,
        eval_offset: int = 0,
        num_chunks: int = 1,
        chunk_index: int = 0,
        xbd_root: Path = XBD_ROOT,
        save_predictions: bool = False,
        extra_args: Sequence[str] = (),
        dry_run: bool = False,
    ):
        if not dry_run:
            output_json.parent.mkdir(parents=True, exist_ok=True)
            output_dir.mkdir(parents=True, exist_ok=True)
        return self._run(
            xbd_command(
                output_json=output_json,
                output_dir=output_dir,
                eval_limit=eval_limit,
                eval_offset=eval_offset,
                num_chunks=num_chunks,
                chunk_index=chunk_index,
                xbd_root=xbd_root,
                python=self.python,
                save_predictions=save_predictions,
                extra_args=extra_args,
            ),
            dry_run=dry_run,
        )

    def merge_xbd(
        self,
        chunk_glob: str,
        output_json: Path,
        output_dir: Path,
        dry_run: bool = False,
    ):
        command = xbd_merge_command(
            chunk_glob=chunk_glob,
            output_json=output_json,
            output_dir=output_dir,
            python=self.python,
        )
        if dry_run:
            return list(command)
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)
        env = os.environ.copy()
        return subprocess.run(command, check=True, env=env)
