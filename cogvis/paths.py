"""Paths for the compact, relocatable CogVis source tree."""

from __future__ import annotations

import os
from pathlib import Path


def _root_from_env() -> Path:
    value = os.environ.get("COGVIS_ROOT")
    if value:
        return Path(value).expanduser().resolve()
    return Path(__file__).resolve().parents[1]


PACKAGE_ROOT = _root_from_env()
SOURCE_ROOT = Path(__file__).resolve().parent


def _path_from_env(name: str, default: Path) -> Path:
    value = os.environ.get(name)
    if not value:
        return default
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = PACKAGE_ROOT / path
    return path.resolve()


LOCKED_RUNTIME = _path_from_env("COGVIS_RUNTIME", SOURCE_ROOT / "runtime")
# Preserve the internal name used by older callers without exposing it as the
# canonical paper-facing name.
LEGACY_RUNTIME = LOCKED_RUNTIME
WEIGHT_DIR = _path_from_env("COGVIS_WEIGHT_DIR", PACKAGE_ROOT / "weights")
RESULTS_DIR = _path_from_env("COGVIS_RESULTS_DIR", PACKAGE_ROOT / "outputs")
PROMPTS = _path_from_env("COGVIS_PROMPTS", PACKAGE_ROOT / "prompts.json")

# Both retained loaders accept the same top-level profiles mapping.
SEMANTIC_PROMPTS = PROMPTS
XBD_PROMPTS = PROMPTS

CHANGE_ADAPTER_CHECKPOINT = _path_from_env(
    "COGVIS_CHANGE_ADAPTER_CHECKPOINT", WEIGHT_DIR / "change_adapter.pt"
)
SCORE_ADAPTER_CHECKPOINT = _path_from_env(
    "COGVIS_SCORE_ADAPTER_CHECKPOINT", WEIGHT_DIR / "score_adapter.pt"
)
GATE_ADAPTER_CHECKPOINT = _path_from_env(
    "COGVIS_GATE_ADAPTER_CHECKPOINT", WEIGHT_DIR / "gate_adapter.pt"
)
# Compatibility aliases retain the original external configuration contract.
SEMANTIC_MEMORY_CHECKPOINT = SCORE_ADAPTER_CHECKPOINT
ADAPTIVE_REGION_FILTER_CHECKPOINT = GATE_ADAPTER_CHECKPOINT
SAM3_CHECKPOINT = _path_from_env("COGVIS_SAM3_CHECKPOINT", WEIGHT_DIR / "sam3.pt")
CLIP_CACHE_DIR = _path_from_env("COGVIS_CLIP_CACHE_DIR", WEIGHT_DIR)

DEFAULT_DATA_ROOT = PACKAGE_ROOT / "data"
OVCD_BENCHMARK_ROOT = _path_from_env(
    "COGVIS_OVCD_BENCHMARK_ROOT", DEFAULT_DATA_ROOT / "OVCD_Benchmark"
)
XBD_ROOT = _path_from_env("COGVIS_XBD_ROOT", DEFAULT_DATA_ROOT / "xBD" / "test")


def require_runtime_layout() -> None:
    required = {
        "runtime": LOCKED_RUNTIME,
        "prompts": PROMPTS,
        "Change Adapter checkpoint": CHANGE_ADAPTER_CHECKPOINT,
        "Score Adapter checkpoint": SCORE_ADAPTER_CHECKPOINT,
        "Gate Adapter checkpoint": GATE_ADAPTER_CHECKPOINT,
        "SAM3 checkpoint": SAM3_CHECKPOINT,
        "CLIP checkpoint": CLIP_CACHE_DIR / "ViT-B-16.pt",
    }
    missing = [f"{label}: {path}" for label, path in required.items() if not path.exists()]
    if missing:
        raise FileNotFoundError("CogVis runtime is incomplete:\n" + "\n".join(missing))
