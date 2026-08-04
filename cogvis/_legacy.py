"""Controlled access to the frozen CogVis runtime implementation."""

from __future__ import annotations

import importlib
import sys
from types import ModuleType

try:
    from .paths import LOCKED_RUNTIME
except ImportError:  # The GitHub-layout package calls this path LEGACY_RUNTIME.
    from .paths import LEGACY_RUNTIME as LOCKED_RUNTIME


def import_locked_runtime(module_name: str) -> ModuleType:
    runtime_path = str(LOCKED_RUNTIME)
    if runtime_path not in sys.path:
        sys.path.insert(0, runtime_path)
    return importlib.import_module(module_name)


def runtime_symbol(module_name: str, symbol_name: str):
    return getattr(import_locked_runtime(module_name), symbol_name)


# Compatibility aliases for callers that imported this internal helper.
import_legacy = import_locked_runtime
legacy_symbol = runtime_symbol
