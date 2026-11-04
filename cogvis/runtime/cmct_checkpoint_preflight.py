"""CPU-only compatibility gate for the frozen CMCT correction head.

This module deliberately does not import torch. The caller supplies its
``torch.load`` function so this guard can be tested without loading a model or
starting any training path.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable


def preflight_frozen_cmct_checkpoint(
    checkpoint_path: str | Path,
    requested_input_dim: int,
    load_checkpoint: Callable,
) -> int:
    """Read the checkpoint on CPU and refuse a feature-dimension mismatch.

    ``load_or_train_cmct`` silently trains a replacement correction head if the
    input dimension differs. This check must run before SAM3/model construction
    and before that loader so the frozen export can never initiate training.
    Returns the checkpoint input dimension on success.
    """
    checkpoint = load_checkpoint(
        str(checkpoint_path), map_location="cpu", weights_only=False,
    )
    try:
        weight = checkpoint["correction_head"]["net.0.weight"]
        checkpoint_input_dim = int(weight.shape[1])
    except (KeyError, IndexError, TypeError, AttributeError) as error:
        raise RuntimeError(
            f"CMCT checkpoint {checkpoint_path} has no readable correction_head net.0.weight; "
            "refusing to construct a model or enter load_or_train_cmct"
        ) from error
    requested_input_dim = int(requested_input_dim)
    if checkpoint_input_dim != requested_input_dim:
        raise RuntimeError(
            "frozen CMCT correction input dimension mismatch: "
            f"checkpoint={checkpoint_input_dim}, requested={requested_input_dim}. "
            "The legacy loader would train a replacement head; refusing before SAM3/model load."
        )
    return checkpoint_input_dim
