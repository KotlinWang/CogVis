import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalLoRALevel(nn.Module):
    """
    Low-rank temporal residual for one SAM3 FPN level.

    This follows the LoRA principle: keep the frozen SAM3 FPN feature as the
    base representation and learn only a small, low-rank update that carries
    bi-temporal change evidence.
    """

    def __init__(self, channels=256, rank=16, alpha=16.0, dropout=0.0):
        super().__init__()
        self.rank = rank
        self.scale = alpha / max(rank, 1)
        in_channels = channels * 4

        self.down = nn.Conv2d(in_channels, rank, kernel_size=1, bias=False)
        self.act = nn.GELU()
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.up = nn.Conv2d(rank, channels, kernel_size=1, bias=False)

        # LoRA starts as a no-op so the frozen decoder initially receives a
        # standard SAM3 FPN feature.
        nn.init.kaiming_uniform_(self.down.weight, a=5 ** 0.5)
        nn.init.zeros_(self.up.weight)

    def forward(self, feat_t1, feat_t2):
        signed_diff = feat_t2 - feat_t1
        abs_diff = signed_diff.abs()
        prod = feat_t1 * feat_t2
        x = torch.cat([feat_t1, feat_t2, abs_diff, prod], dim=1)
        delta = self.up(self.dropout(self.act(self.down(x)))) * self.scale
        return delta


def _spatial_energy(feat):
    return feat.float().square().mean(dim=1, keepdim=True).sqrt()


def _normalize_map(map_tensor):
    eps = 1e-6
    denom = map_tensor.mean(dim=(-2, -1), keepdim=True).detach().clamp_min(eps)
    return map_tensor / denom


def _parse_levels(levels, default=(0, 1, 2)):
    if levels is None:
        return tuple(default)
    if isinstance(levels, str):
        value = levels.strip().lower()
        if value in ("", "all"):
            return (0, 1, 2)
        parsed = tuple(int(x.strip()) for x in value.split(",") if x.strip())
    else:
        parsed = tuple(int(x) for x in levels)
    if not parsed:
        raise ValueError("At least one v4 active level is required")
    for level in parsed:
        if level < 0 or level > 2:
            raise ValueError(f"Invalid FPN level: {level}")
    return tuple(sorted(set(parsed)))


class TemporalGateLevel(nn.Module):
    """
    One temporal LoRA level plus change-stat extraction for v4.

    The LoRA branch still learns the low-rank temporal residual, while the
    auxiliary statistics are used by the fused change gate.
    """

    def __init__(self, channels=256, rank=16, alpha=16.0, dropout=0.0):
        super().__init__()
        self.delta = TemporalLoRALevel(
            channels=channels,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
        )

    def forward(self, feat_t1, feat_t2):
        delta = self.delta(feat_t1, feat_t2)
        signed_diff = feat_t2 - feat_t1
        raw_diff = signed_diff.abs().mean(dim=1, keepdim=True)
        delta_mag = delta.float().norm(dim=1, keepdim=True)
        base_mag = 0.5 * (_spatial_energy(feat_t1) + _spatial_energy(feat_t2))
        cos_diff = (1 - F.cosine_similarity(feat_t1, feat_t2, dim=1, eps=1e-6).unsqueeze(1)).clamp(0, 2) / 2

        stats = torch.cat([
            _normalize_map(raw_diff),
            _normalize_map(delta_mag),
            _normalize_map(base_mag),
            cos_diff,
        ], dim=1)
        return delta, stats


class AuxiliaryChangeHead(nn.Module):
    """
    Category-agnostic change head trained with binary change masks.

    It reads LoRA residuals, not semantic class logits. Its job is only BCD:
    changed vs unchanged.
    """

    def __init__(self, channels=256, hidden_dim=96):
        super().__init__()
        self.proj = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(channels, hidden_dim, kernel_size=1, bias=False),
                nn.GroupNorm(8, hidden_dim),
                nn.GELU(),
            )
            for _ in range(3)
        ])
        self.fuse = nn.Sequential(
            nn.Conv2d(hidden_dim * 3, hidden_dim, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim // 2, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, hidden_dim // 2),
            nn.GELU(),
            nn.Conv2d(hidden_dim // 2, 1, kernel_size=1),
        )

    def forward(self, deltas, output_size=None):
        target_size = deltas[0].shape[-2:]
        feats = []
        for delta, proj in zip(deltas, self.proj):
            feat = proj(delta)
            if feat.shape[-2:] != target_size:
                feat = F.interpolate(feat, size=target_size, mode="bilinear", align_corners=False)
            feats.append(feat)
        logits = self.fuse(torch.cat(feats, dim=1))
        if output_size is not None and logits.shape[-2:] != output_size:
            logits = F.interpolate(logits, size=output_size, mode="bilinear", align_corners=False)
        return logits


def _clip_residual_by_rms(residual, base, max_ratio):
    if max_ratio is None or max_ratio <= 0:
        return residual
    eps = 1e-6
    with torch.no_grad():
        residual_rms = residual.detach().float().square().mean(dim=(1, 2, 3), keepdim=True).sqrt()
        base_rms = base.detach().float().square().mean(dim=(1, 2, 3), keepdim=True).sqrt()
        scale = (max_ratio * base_rms / (residual_rms + eps)).clamp(max=1.0)
    return residual * scale.to(device=residual.device, dtype=residual.dtype)


class TemporalLoRAFPNAdapter(nn.Module):
    """
    SAM3-FPN-compatible temporal adapter with an auxiliary change head.

    Returns:
        dict with:
          change_fpn: list of SAM3-decoder-compatible FPN tensors
          change_logits: [B, 1, H, W] category-agnostic BCD logits
          deltas: low-rank temporal residuals per FPN level
    """

    def __init__(self, channels=256, rank=16, alpha=16.0, dropout=0.0,
                 gamma_init=0.05, head_hidden_dim=96, max_gamma=0.2,
                 residual_clip_ratio=0.25):
        super().__init__()
        self.levels = nn.ModuleList([
            TemporalLoRALevel(channels=channels, rank=rank, alpha=alpha, dropout=dropout)
            for _ in range(3)
        ])
        self.change_head = AuxiliaryChangeHead(channels=channels, hidden_dim=head_hidden_dim)
        self.gamma = nn.Parameter(torch.tensor(float(gamma_init)))
        self.max_gamma = max_gamma
        self.residual_clip_ratio = residual_clip_ratio

    @property
    def effective_gamma(self):
        if self.max_gamma is None or self.max_gamma <= 0:
            return self.gamma
        return self.gamma.clamp(-self.max_gamma, self.max_gamma)

    def forward(self, fpn_t1, fpn_t2, output_size=None):
        deltas = [level(a, b) for level, a, b in zip(self.levels, fpn_t1, fpn_t2)]
        gamma = self.effective_gamma
        residuals = [
            _clip_residual_by_rms(gamma * delta, base, self.residual_clip_ratio)
            for base, delta in zip(fpn_t1, deltas)
        ]
        change_fpn = [base + residual for base, residual in zip(fpn_t1, residuals)]
        change_logits = self.change_head(deltas, output_size=output_size)
        return {
            "change_fpn": change_fpn,
            "change_logits": change_logits,
            "deltas": deltas,
            "residuals": residuals,
            "effective_gamma": gamma,
        }


class TemporalGateFPNAdapter(nn.Module):
    """
    v4 adapter: space-gated temporal residuals.

    The fused change logit is used twice:
      1. as the explicit category-agnostic change head
      2. as a spatial gate that modulates LoRA residual injection per level

    This removes the collapse-prone global gamma from the critical path.
    """

    def __init__(
        self,
        channels=256,
        rank=16,
        alpha=16.0,
        dropout=0.0,
        gain_init=1.0,
        gain_min=0.75,
        gain_max=1.25,
        gate_floor=0.02,
        gate_temperature=1.0,
        head_hidden_dim=96,
        residual_clip_ratio=0.25,
        active_levels=(0, 1, 2),
        gate_level=0,
    ):
        super().__init__()
        self.active_levels = _parse_levels(active_levels)
        self.gate_level = int(gate_level)
        if self.gate_level < 0 or self.gate_level > 2:
            raise ValueError(f"Invalid gate_level: {gate_level}")
        self.levels = nn.ModuleList([
            TemporalGateLevel(channels=channels, rank=rank, alpha=alpha, dropout=dropout)
            for _ in range(3)
        ])
        stat_channels = 4 * len(self.active_levels)
        self.change_fuse = nn.Sequential(
            nn.Conv2d(stat_channels, head_hidden_dim, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, head_hidden_dim),
            nn.GELU(),
            nn.Conv2d(head_hidden_dim, head_hidden_dim, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, head_hidden_dim),
            nn.GELU(),
            nn.Conv2d(head_hidden_dim, 1, kernel_size=1),
        )
        self.gain_raw = nn.Parameter(torch.tensor(float(0.0)))
        self.gain_min = float(gain_min)
        self.gain_max = float(gain_max)
        self.gate_floor = float(gate_floor)
        self.gate_temperature = float(gate_temperature)
        self.residual_clip_ratio = residual_clip_ratio

        if gain_max <= gain_min:
            raise ValueError(f"Invalid gain bounds: min={gain_min}, max={gain_max}")
        init = float(gain_init)
        init = min(max(init, gain_min + 1e-4), gain_max - 1e-4)
        init_ratio = (init - gain_min) / (gain_max - gain_min)
        init_ratio = min(max(init_ratio, 1e-4), 1 - 1e-4)
        raw = torch.log(torch.tensor(init_ratio / (1 - init_ratio)))
        with torch.no_grad():
            self.gain_raw.copy_(raw)
            final = self.change_fuse[-1]
            nn.init.normal_(final.weight, mean=0.0, std=1e-3)
            nn.init.constant_(final.bias, -1.0)

    @property
    def gamma(self):
        gain = torch.sigmoid(self.gain_raw)
        return self.gain_min + (self.gain_max - self.gain_min) * gain

    @property
    def effective_gamma(self):
        return self.gamma

    def _gate_from_logits(self, logits):
        gate = torch.sigmoid(logits.float() / max(self.gate_temperature, 1e-6))
        gate = self.gate_floor + (1.0 - self.gate_floor) * gate
        return gate.to(dtype=logits.dtype)

    def forward(self, fpn_t1, fpn_t2, output_size=None):
        active_deltas = []
        deltas_by_level = [None, None, None]
        stat_maps = []
        for idx, (level, feat_t1, feat_t2) in enumerate(zip(self.levels, fpn_t1, fpn_t2)):
            if idx not in self.active_levels:
                continue
            if feat_t1 is None or feat_t2 is None:
                raise ValueError(f"v4 active level {idx} was not loaded")
            delta, stats = level(feat_t1, feat_t2)
            active_deltas.append(delta)
            deltas_by_level[idx] = delta
            stat_maps.append(stats)

        if fpn_t1[self.gate_level] is None:
            raise ValueError(f"v4 gate level {self.gate_level} was not loaded")
        target_size = fpn_t1[self.gate_level].shape[-2:]
        fused_stats = []
        for stats in stat_maps:
            if stats.shape[-2:] != target_size:
                stats = F.interpolate(stats, size=target_size, mode="bilinear", align_corners=False)
            fused_stats.append(stats)
        change_logits = self.change_fuse(torch.cat(fused_stats, dim=1))
        if output_size is not None and change_logits.shape[-2:] != output_size:
            change_logits_out = F.interpolate(change_logits, size=output_size, mode="bilinear", align_corners=False)
        else:
            change_logits_out = change_logits

        gate_full = self._gate_from_logits(change_logits)
        gamma = self.effective_gamma
        change_fpn = []
        residuals = []
        for idx, base in enumerate(fpn_t1):
            if base is None:
                change_fpn.append(None)
                residuals.append(None)
                continue
            delta = deltas_by_level[idx]
            if delta is None:
                residual = torch.zeros_like(base)
                change_fpn.append(base)
                residuals.append(residual)
                continue
            gate_level = gate_full
            if gate_level.shape[-2:] != delta.shape[-2:]:
                gate_level = F.interpolate(gate_level, size=delta.shape[-2:], mode="bilinear", align_corners=False)
            residual = _clip_residual_by_rms(gamma * gate_level * delta, base, self.residual_clip_ratio)
            change_fpn.append(base + residual)
            residuals.append(residual)

        return {
            "change_fpn": change_fpn,
            "change_logits": change_logits_out,
            "deltas": active_deltas,
            "residuals": residuals,
            "gate_map": gate_full if output_size is None or gate_full.shape[-2:] == output_size else F.interpolate(
                gate_full, size=output_size, mode="bilinear", align_corners=False),
            "effective_gamma": gamma,
        }
