"""CLIP-guided memory retrieval utilities for CMCT experiments.

The module is intentionally small and side-effect free.  It provides only
feature extraction, memory-key construction, KNN threshold retrieval, and a
light correction head.  Dataset loading and OVCD decoding stay in the caller.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image


CMCT_STAT_NAMES = [
    "score_mean",
    "score_std",
    "score_q50",
    "score_q75",
    "score_q85",
    "score_q90",
    "score_q95",
    "score_q98",
    "score_max",
    "score_ratio_above_mean",
    "score_ratio_above_q85",
    "bcd_mean",
    "bcd_std",
    "bcd_q75",
    "bcd_q85",
    "bcd_q95",
    "bcd_max",
    "score_bcd_product_mean",
    "score_bcd_product_q85",
    "presence",
]


@dataclass
class ClipPairEmbedding:
    scene: np.ndarray
    delta: np.ndarray


@dataclass
class ClipRegionEmbedding:
    before: np.ndarray
    after: np.ndarray
    scene: np.ndarray
    delta: np.ndarray


def _l2_normalize_np(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    norm = float(np.linalg.norm(x))
    if norm < eps:
        return np.zeros_like(x, dtype=np.float32)
    return (x / norm).astype(np.float32)


def _safe_quantiles(values: np.ndarray, qs: Sequence[float]) -> List[float]:
    if values.size == 0:
        return [0.0 for _ in qs]
    return [float(v) for v in np.quantile(values, qs)]


class ClipFeatureExtractor:
    """Thin wrapper around OpenAI CLIP with text and image-pair caching."""

    def __init__(
        self,
        model_name: str = "ViT-B/16",
        cache_dir: str = "weights",
        device: Optional[torch.device] = None,
    ):
        import clip

        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model, self.preprocess = clip.load(
            model_name,
            device=str(self.device),
            download_root=cache_dir,
            jit=False,
        )
        self.model.eval()
        self._clip = clip
        self.model_name = model_name

    @torch.no_grad()
    @lru_cache(maxsize=4096)
    def encode_text(self, text: str) -> np.ndarray:
        tokens = self._clip.tokenize([str(text)]).to(self.device)
        feat = self.model.encode_text(tokens).float()
        feat = feat / feat.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        return feat[0].detach().cpu().numpy().astype(np.float32)

    @torch.no_grad()
    @lru_cache(maxsize=2048)
    def encode_image(self, path: str) -> np.ndarray:
        image = Image.open(path).convert("RGB")
        return self.encode_pil(image)

    @torch.no_grad()
    def encode_pil(self, image: Image.Image) -> np.ndarray:
        tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        feat = self.model.encode_image(tensor).float()
        feat = feat / feat.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        return feat[0].detach().cpu().numpy().astype(np.float32)

    @torch.no_grad()
    @lru_cache(maxsize=8192)
    def encode_image_crop(self, path: str, box_xyxy: Tuple[int, int, int, int]) -> np.ndarray:
        image = Image.open(path).convert("RGB")
        width, height = image.size
        x0, y0, x1, y1 = [int(v) for v in box_xyxy]
        x0 = max(0, min(x0, width - 1))
        y0 = max(0, min(y0, height - 1))
        x1 = max(x0 + 1, min(x1, width))
        y1 = max(y0 + 1, min(y1, height))
        return self.encode_pil(image.crop((x0, y0, x1, y1)))

    def encode_pair(self, item: dict) -> ClipPairEmbedding:
        emb_a = self.encode_image(str(item["a"]))
        emb_b = self.encode_image(str(item["b"]))
        scene = _l2_normalize_np(0.5 * (emb_a + emb_b))
        delta = _l2_normalize_np(np.abs(emb_b - emb_a))
        return ClipPairEmbedding(scene=scene, delta=delta)

    def encode_region_pair(
        self,
        item: dict,
        box_yxyx: Tuple[int, int, int, int],
        mask_hw: Tuple[int, int],
        pad: int = 4,
    ) -> ClipRegionEmbedding:
        path_a = str(item["a"])
        path_b = str(item["b"])
        with Image.open(path_a) as image:
            width, height = image.size
        mask_h, mask_w = int(mask_hw[0]), int(mask_hw[1])
        y0, x0, y1, x1 = [int(v) for v in box_yxyx]
        pad = max(int(pad), 0)
        y0 -= pad
        x0 -= pad
        y1 += pad
        x1 += pad
        sx = float(width) / max(float(mask_w), 1.0)
        sy = float(height) / max(float(mask_h), 1.0)
        crop_box = (
            int(np.floor(x0 * sx)),
            int(np.floor(y0 * sy)),
            int(np.ceil(x1 * sx)),
            int(np.ceil(y1 * sy)),
        )
        emb_a = self.encode_image_crop(path_a, crop_box)
        emb_b = self.encode_image_crop(path_b, crop_box)
        scene = _l2_normalize_np(0.5 * (emb_a + emb_b))
        delta = _l2_normalize_np(np.abs(emb_b - emb_a))
        return ClipRegionEmbedding(before=emb_a, after=emb_b, scene=scene, delta=delta)


class CorrectionHead(nn.Module):
    """Predict a bounded delta added to the retrieved memory threshold."""

    def __init__(self, in_dim: int, hidden_dim: int = 64, dropout: float = 0.05, max_delta: float = 0.20):
        super().__init__()
        self.max_delta = float(max_delta)
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.max_delta * torch.tanh(self.net(x).squeeze(-1))


def score_clip_stats(score: torch.Tensor, bcd_score: torch.Tensor, presence: Optional[torch.Tensor] = None) -> np.ndarray:
    """Build compact source/target statistics for one image-class score map."""

    score_np = score.detach().float().clamp(0, 1).cpu().numpy().astype(np.float32).reshape(-1)
    bcd_np = bcd_score.detach().float().clamp(0, 1).cpu().numpy().astype(np.float32).reshape(-1)
    if score_np.size == 0:
        return np.zeros((len(CMCT_STAT_NAMES),), dtype=np.float32)

    sq50, sq75, sq85, sq90, sq95, sq98 = _safe_quantiles(score_np, [0.50, 0.75, 0.85, 0.90, 0.95, 0.98])
    bq75, bq85, bq95 = _safe_quantiles(bcd_np, [0.75, 0.85, 0.95])
    product = score_np * bcd_np
    pq85 = _safe_quantiles(product, [0.85])[0]
    score_mean = float(score_np.mean())
    score_std = float(score_np.std())
    presence_value = 0.0
    if presence is not None:
        presence_value = float(presence.detach().float().clamp(0, 1).mean().cpu())
    return np.asarray([
        score_mean,
        score_std,
        sq50,
        sq75,
        sq85,
        sq90,
        sq95,
        sq98,
        float(score_np.max()),
        float((score_np > score_mean).mean()),
        float((score_np > sq85).mean()),
        float(bcd_np.mean()),
        float(bcd_np.std()),
        bq75,
        bq85,
        bq95,
        float(bcd_np.max()),
        float(product.mean()),
        pq85,
        presence_value,
    ], dtype=np.float32)


def build_memory_key(
    pair_embedding: ClipPairEmbedding,
    text_embedding: np.ndarray,
    stats: np.ndarray,
    scene_weight: float = 1.0,
    delta_weight: float = 0.7,
    text_weight: float = 1.0,
    stats_weight: float = 0.25,
) -> np.ndarray:
    """Construct a normalized key for scene-class-statistics retrieval."""

    parts = [
        float(scene_weight) * _l2_normalize_np(pair_embedding.scene),
        float(delta_weight) * _l2_normalize_np(pair_embedding.delta),
        float(text_weight) * _l2_normalize_np(text_embedding),
        float(stats_weight) * np.asarray(stats, dtype=np.float32),
    ]
    return _l2_normalize_np(np.concatenate(parts, axis=0))


def retrieve_tau(
    query_key: np.ndarray,
    memory_keys: np.ndarray,
    memory_taus: np.ndarray,
    topk: int = 16,
    temperature: float = 0.07,
    exclude_index: Optional[int] = None,
) -> Tuple[float, Dict[str, float]]:
    """Return a CLIP-memory weighted threshold and retrieval diagnostics."""

    if memory_keys.size == 0 or memory_taus.size == 0:
        return 0.5, {"top1_sim": 0.0, "mean_sim": 0.0, "topk": 0}
    sims = np.asarray(memory_keys @ np.asarray(query_key, dtype=np.float32), dtype=np.float32)
    if exclude_index is not None and 0 <= int(exclude_index) < sims.shape[0]:
        sims[int(exclude_index)] = -np.inf
    finite = np.isfinite(sims)
    if not finite.any():
        return float(np.mean(memory_taus)), {"top1_sim": 0.0, "mean_sim": 0.0, "topk": 0}
    k = min(max(int(topk), 1), int(finite.sum()))
    idx = np.argpartition(sims, -k)[-k:]
    idx = idx[np.argsort(sims[idx])[::-1]]
    selected_sims = sims[idx]
    selected_taus = memory_taus[idx].astype(np.float32)
    temp = max(float(temperature), 1e-4)
    logits = selected_sims / temp
    logits = logits - float(np.max(logits))
    weights = np.exp(logits)
    weights = weights / max(float(weights.sum()), 1e-8)
    tau = float(np.sum(weights * selected_taus))
    return tau, {
        "top1_sim": float(selected_sims[0]),
        "mean_sim": float(np.mean(selected_sims)),
        "topk": int(k),
    }


def correction_features(stats: np.ndarray, tau_mem: float, retrieval_diag: Dict[str, float]) -> np.ndarray:
    return np.concatenate([
        np.asarray(stats, dtype=np.float32),
        np.asarray([
            float(tau_mem),
            float(retrieval_diag.get("top1_sim", 0.0)),
            float(retrieval_diag.get("mean_sim", 0.0)),
        ], dtype=np.float32),
    ], axis=0)


def stack_or_empty(rows: Iterable[np.ndarray], dim: int) -> np.ndarray:
    rows = list(rows)
    if not rows:
        return np.zeros((0, dim), dtype=np.float32)
    return np.stack(rows, axis=0).astype(np.float32)
