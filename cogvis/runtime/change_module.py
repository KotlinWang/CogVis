import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from model.ovcd.change_blocks import MDFM
from model.ovcd.model_aggregator import EffAggregatorLayer
from temporal_lora_adapter import TemporalGateFPNAdapter, TemporalLoRAFPNAdapter


class FPNLevelProcessor(nn.Module):
    """
    Process one FPN level: extract difference between T1/T2, convert to decoder-compatible format.

    Like FeatUp translates low-res features to high-res features,
    this translates bi-temporal features to change-aware features
    that SAM3 decoder can directly consume.
    """

    def __init__(self, in_dim, out_dim):
        super().__init__()
        # Project 256-dim FPN to working dimension
        self.adapter_T1 = nn.Conv2d(256, in_dim, kernel_size=1, stride=1, padding=0)
        self.adapter_T2 = nn.Conv2d(256, in_dim, kernel_size=1, stride=1, padding=0)

        # Extract difference between T1 and T2
        self.mdfm = MDFM(in_dim, out_dim)

        # Refine difference with cross-temporal attention
        self.aggregator = EffAggregatorLayer(out_dim)

        # Convert back to 256-dim FPN space (decoder-compatible format)
        self.proj_out = nn.Sequential(
            nn.Conv2d(out_dim, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(8, 256),
            nn.ReLU(inplace=True),
        )

    def forward(self, fpn_T1, fpn_T2):
        """
        Args:
            fpn_T1: [B, 256, H, W] - T1 FPN feature at this level
            fpn_T2: [B, 256, H, W] - T2 FPN feature at this level
        Returns:
            change_feature: [B, 256, H, W] - decoder-compatible change feature
        """
        # Project to working dimension
        feat_T1 = self.adapter_T1(fpn_T1)
        feat_T2 = self.adapter_T2(fpn_T2)

        # Extract and refine difference
        diff = self.mdfm(feat_T1, feat_T2)
        diff = self.aggregator(diff, feat_T1, feat_T2)

        # Convert to decoder-compatible 256-dim format
        change_feature = self.proj_out(diff)

        return change_feature


class MDFMFPNAdapter(nn.Module):
    """
    Feature translator: converts bi-temporal SAM3 FPN features into
    change-aware features that SAM3 decoder can directly process.

    Analogous to FeatUp:
      FeatUp:    low-res features  → high-res features (for downstream)
      STEE: T1/T2 FPN features → change-aware features (for SAM3 decoder)

    Input:  fpn_T1, fpn_T2 (each list of 3 tensors: [B,256,288,288], [B,256,144,144], [B,256,72,72])
    Output: change_fpn (list of 3 tensors, same shapes, decoder-compatible)
    """

    def __init__(self, dims=None):
        super().__init__()
        if dims is None:
            dims = [64, 96, 128]

        self.level0 = FPNLevelProcessor(dims[0], dims[0])  # 288x288
        self.level1 = FPNLevelProcessor(dims[1], dims[1])  # 144x144
        self.level2 = FPNLevelProcessor(dims[2], dims[2])  # 72x72

        # Residual scaling: starts near identity so decoder still works at init
        self.gamma = nn.Parameter(torch.tensor(0.1))

    def forward(self, fpn_T1, fpn_T2, output_size=None):
        """
        Args:
            fpn_T1: list of 3 tensors [B,256,H_i,W_i]
            fpn_T2: list of 3 tensors [B,256,H_i,W_i]
        Returns:
            change_fpn: list of 3 tensors [B,256,H_i,W_i]
                        Decoder-compatible features encoding T1/T2 differences.
        """
        diff_0 = self.level0(fpn_T1[0], fpn_T2[0])
        diff_1 = self.level1(fpn_T1[1], fpn_T2[1])
        diff_2 = self.level2(fpn_T1[2], fpn_T2[2])

        # Residual: base feature + learned change signal
        # Use T1 as base (arbitrary choice, could also use average)
        change_fpn = [
            fpn_T1[0] + self.gamma * diff_0,
            fpn_T1[1] + self.gamma * diff_1,
            fpn_T1[2] + self.gamma * diff_2,
        ]

        return {
            "change_fpn": change_fpn,
            "change_logits": None,
            "deltas": [
                change_fpn[0] - fpn_T1[0],
                change_fpn[1] - fpn_T1[1],
                change_fpn[2] - fpn_T1[2],
            ],
        }


class SemanticTemporalEvidenceEncoder(nn.Module):
    """
    Semantic-Temporal Evidence Encoder (STEE).

    STEE is the change-evidence extraction module in CogMem-OVCD. It keeps the
    SAM3 encoder/decoder frozen and learns a decoder-compatible temporal
    adapter that produces change-aware FPN features and category-agnostic BCD
    logits.

    module_type='temporal_lora' is the v2 design:
      Frozen SAM3 FPN + low-rank temporal residual + auxiliary BCD head.

    module_type='temporal_gate_v4' is the recommended v4 design:
      Frozen SAM3 FPN + low-rank temporal residual + spatial change gate.

    module_type='mdfm' keeps the previous MDFM+EffAggregator adapter for
    compatibility with older checkpoints.
    """

    def __init__(self, dims=None, module_type="temporal_lora", lora_rank=16,
                 lora_alpha=16.0, lora_dropout=0.0, gamma_init=0.05,
                 head_hidden_dim=96, max_gamma=0.2, residual_clip_ratio=0.25,
                 gain_init=1.0, gain_min=0.75, gain_max=1.25,
                 gate_floor=0.02, gate_temperature=1.0,
                 v4_active_levels=(0, 1, 2), v4_gate_level=0):
        super().__init__()
        self.module_type = module_type
        if module_type == "mdfm":
            self.adapter = MDFMFPNAdapter(dims=dims)
        elif module_type == "temporal_lora":
            self.adapter = TemporalLoRAFPNAdapter(
                rank=lora_rank,
                alpha=lora_alpha,
                dropout=lora_dropout,
                gamma_init=gamma_init,
                head_hidden_dim=head_hidden_dim,
                max_gamma=max_gamma,
                residual_clip_ratio=residual_clip_ratio,
            )
        elif module_type == "temporal_gate_v4":
            self.adapter = TemporalGateFPNAdapter(
                rank=lora_rank,
                alpha=lora_alpha,
                dropout=lora_dropout,
                gain_init=gain_init,
                gain_min=gain_min,
                gain_max=gain_max,
                gate_floor=gate_floor,
                gate_temperature=gate_temperature,
                head_hidden_dim=head_hidden_dim,
                residual_clip_ratio=residual_clip_ratio,
                active_levels=v4_active_levels,
                gate_level=v4_gate_level,
            )
        else:
            raise ValueError(f"Unknown module_type: {module_type}")

    @property
    def gamma(self):
        return self.adapter.gamma

    def forward(self, fpn_T1, fpn_T2, output_size=None):
        return self.adapter(fpn_T1, fpn_T2, output_size=output_size)


# Public alias for callers that refer to the Change Adapter directly.
ChangeAdapterModule = SemanticTemporalEvidenceEncoder


def infer_module_type_from_state_dict(state_dict):
    keys = list(state_dict.keys())
    if any("change_fuse" in k or "gain_raw" in k or ".delta." in k for k in keys):
        return "temporal_gate_v4"
    if any(k.startswith("adapter.levels.") or ".down.weight" in k for k in keys):
        return "temporal_lora"
    return "mdfm"


def normalize_state_dict_for_stee(state_dict, module_type):
    """
    Support older MDFM checkpoints that were saved before STEE got
    the `adapter.` wrapper prefix.
    """
    if module_type == "mdfm" and not any(k.startswith("adapter.") for k in state_dict):
        return {f"adapter.{k}": v for k, v in state_dict.items()}
    return state_dict


normalize_state_dict_for_change_adapter = normalize_state_dict_for_stee


def module_kwargs_from_checkpoint(ckpt=None, args=None):
    """
    Resolve STEE constructor kwargs from checkpoint metadata.

    Newer checkpoints store v4 gate parameters explicitly. Older checkpoints
    fall back to CLI arguments so existing runs stay loadable.
    """
    ckpt = ckpt or {}

    def pick(name, default):
        if name in ckpt and ckpt[name] is not None:
            return ckpt[name]
        if args is not None and hasattr(args, name):
            return getattr(args, name)
        return default

    return {
        "lora_rank": pick("lora_rank", 16),
        "lora_alpha": pick("lora_alpha", 16.0),
        "lora_dropout": pick("lora_dropout", 0.0),
        "gamma_init": pick("gamma_init", 0.05),
        "head_hidden_dim": pick("head_hidden_dim", 96),
        "max_gamma": pick("max_gamma", 0.2),
        "residual_clip_ratio": pick("residual_clip_ratio", 0.25),
        "gain_init": pick("gain_init", 1.0),
        "gain_min": pick("gain_min", 0.75),
        "gain_max": pick("gain_max", 1.25),
        "gate_floor": pick("gate_floor", 0.02),
        "gate_temperature": pick("gate_temperature", 1.0),
        "v4_active_levels": pick("v4_active_levels", (0, 1, 2)),
        "v4_gate_level": pick("v4_gate_level", 0),
    }
