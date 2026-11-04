"""Text-conditioned posterior OVCD utilities.

This module compares SAM3 text-conditioned concept posteriors across time.
It deliberately operates after SAM3 grounding/decoding, so it does not
depend on injecting modified FPN features into the frozen SAM3 decoder.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F


@dataclass
class PosteriorInputs:
    scores_a: torch.Tensor
    scores_b: torch.Tensor
    presence_a: torch.Tensor
    presence_b: torch.Tensor
    posterior_a: torch.Tensor
    posterior_b: torch.Tensor
    delta: torch.Tensor


def _as_hw(tensor: torch.Tensor, h: int, w: int) -> torch.Tensor:
    tensor = tensor.float()
    if tensor.ndim == 4:
        if tensor.shape[-2:] != (h, w):
            tensor = F.interpolate(tensor, size=(h, w), mode="bilinear", align_corners=False)
        tensor = tensor.squeeze(0).squeeze(0)
    elif tensor.ndim == 3:
        if tensor.shape[-2:] != (h, w):
            tensor = F.interpolate(tensor.unsqueeze(0), size=(h, w), mode="bilinear", align_corners=False).squeeze(0)
        tensor = tensor.squeeze(0)
    elif tensor.ndim == 2:
        if tensor.shape != (h, w):
            tensor = F.interpolate(tensor.view(1, 1, *tensor.shape), size=(h, w), mode="bilinear", align_corners=False).squeeze()
    else:
        raise ValueError(f"Cannot convert tensor with shape {tuple(tensor.shape)} to HxW")
    return tensor.clamp(0, 1)


def prompt_score_from_state(
    processor,
    state: Dict,
    prompt: str,
    capture: Optional[List[Dict[str, object]]] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Run one SAM3 prompt and return a dense concept score plus presence.

    Sam3Processor stores probabilities despite names such as masks_logits and
    semantic_mask_logits. object_score is already presence-gated in this local
    SAM3 implementation, so we apply presence to the semantic branch and use
    object_score directly for the instance branch.

    ``capture`` (opt-in) records the raw per-mask SAM3 grounding output of this
    prompt.  It must be read here and not later: ``set_text_prompt`` overwrites
    ``state["masks"]``/``state["scores"]``/``state["boxes"]`` on the very next
    prompt, so any later reader would only see the last prompt of the bank.
    """
    h = state["original_height"]
    w = state["original_width"]
    device = torch.device(processor.device)

    processor.reset_all_prompts(state)
    state = processor.set_text_prompt(state=state, prompt=prompt)

    if capture is not None:
        capture.append(_capture_grounding(state, prompt))

    presence = state["presence_score"].float().reshape(()).clamp(0, 1)

    sem = _as_hw(state["semantic_mask_logits"], h, w) * presence
    inst = torch.zeros((h, w), device=device, dtype=torch.float32)
    masks = state.get("masks_logits")
    obj_scores = state.get("object_score")
    if masks is not None and obj_scores is not None and masks.numel() > 0:
        masks = masks.float()
        if masks.ndim == 4:
            masks = masks.squeeze(1)
        if masks.shape[-2:] != (h, w):
            masks = F.interpolate(masks.unsqueeze(1), size=(h, w), mode="bilinear", align_corners=False).squeeze(1)
        obj_scores = obj_scores.float().view(-1, 1, 1).clamp(0, 1)
        inst = (masks.clamp(0, 1) * obj_scores).amax(dim=0)

    score = torch.maximum(sem, inst).clamp(0, 1)
    return score, presence


@torch.no_grad()
def _capture_grounding(state: Dict, prompt: str) -> Dict[str, object]:
    """Snapshot the raw SAM3 grounding output of the prompt currently in ``state``."""
    masks = state.get("masks")
    scores = state.get("object_score")
    boxes = state.get("boxes")
    if masks is None or scores is None:
        return {
            "prompt": prompt,
            "masks": torch.zeros((0, state["original_height"], state["original_width"]), dtype=torch.bool),
            "scores": torch.zeros((0,), dtype=torch.float32),
            "boxes": torch.zeros((0, 4), dtype=torch.float32),
        }
    if masks.ndim == 4:
        masks = masks.squeeze(1)
    return {
        "prompt": prompt,
        "masks": masks.detach().to(dtype=torch.bool).cpu(),
        "scores": scores.detach().float().reshape(-1).cpu(),
        "boxes": (boxes.detach().float().reshape(-1, 4).cpu() if boxes is not None
                  else torch.zeros((int(masks.shape[0]), 4), dtype=torch.float32)),
    }


@torch.no_grad()
def extract_class_scores(
    processor,
    state: Dict,
    prompt_bank: Dict[str, List[str]],
    class_names: Optional[Iterable[str]] = None,
    capture: Optional[Dict[str, List[Dict[str, object]]]] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return class-level SAM3 fused scores [C,H,W] and presence [C].

    ``capture`` (opt-in) maps each class name to the raw per-prompt grounding
    snapshots of that class, taken while the prompt is still live in ``state``.
    """
    if class_names is None:
        class_names = list(prompt_bank.keys())
    class_names = list(class_names)
    h = state["original_height"]
    w = state["original_width"]
    device = torch.device(processor.device)

    class_scores = []
    class_presence = []
    for name in class_names:
        score_c = torch.zeros((h, w), device=device, dtype=torch.float32)
        presence_c = torch.zeros((), device=device, dtype=torch.float32)
        class_capture = None if capture is None else capture.setdefault(name, [])
        for prompt in prompt_bank[name]:
            score_p, presence_p = prompt_score_from_state(
                processor, state, prompt, capture=class_capture,
            )
            score_c = torch.maximum(score_c, score_p)
            presence_c = torch.maximum(presence_c, presence_p)
        class_scores.append(score_c)
        class_presence.append(presence_c)

    return torch.stack(class_scores, dim=0), torch.stack(class_presence, dim=0)


def competitive_posteriors(
    scores: torch.Tensor,
    class_names: List[str],
    confusers: Dict[str, List[str]],
    rho: float = 1.5,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Calibrate concept scores by target-vs-confuser competition."""
    name_to_idx = {name: idx for idx, name in enumerate(class_names)}
    calibrated = []
    for idx, name in enumerate(class_names):
        conf_names = [c for c in confusers.get(name, []) if c in name_to_idx and c != name]
        if not conf_names:
            conf_idxs = [j for j in range(len(class_names)) if j != idx]
        else:
            conf_idxs = [name_to_idx[c] for c in conf_names]
        target = scores[idx]
        if conf_idxs:
            competitor = scores[conf_idxs].amax(dim=0)
        else:
            competitor = torch.zeros_like(target)
        dominance = target / (target + competitor + eps)
        calibrated.append((target * dominance.clamp(0, 1).pow(rho)).clamp(0, 1))
    return torch.stack(calibrated, dim=0)


def build_posterior_inputs(
    scores_a: torch.Tensor,
    scores_b: torch.Tensor,
    presence_a: torch.Tensor,
    presence_b: torch.Tensor,
    class_names: List[str],
    confusers: Dict[str, List[str]],
    rho: float,
) -> PosteriorInputs:
    posterior_a = competitive_posteriors(scores_a, class_names, confusers, rho=rho)
    posterior_b = competitive_posteriors(scores_b, class_names, confusers, rho=rho)
    delta = (posterior_b - posterior_a).abs().clamp(0, 1)
    return PosteriorInputs(
        scores_a=scores_a,
        scores_b=scores_b,
        presence_a=presence_a,
        presence_b=presence_b,
        posterior_a=posterior_a,
        posterior_b=posterior_b,
        delta=delta,
    )


def apply_bcd_soft_gate(
    score: torch.Tensor,
    bcd_score: Optional[torch.Tensor],
    alpha: float = 1.0,
    floor: float = 0.2,
) -> torch.Tensor:
    """Gate posterior delta by BCD without hard intersection."""
    if bcd_score is None:
        return score
    gate = bcd_score.float().clamp(0, 1).pow(alpha)
    if floor > 0:
        gate = floor + (1.0 - floor) * gate
    return (score.float() * gate).clamp(0, 1)


def compute_score_threshold(
    score: torch.Tensor,
    mode: str = "percentile",
    percentile: float = 85.0,
    threshold_abs: Optional[float] = None,
    std_factor: float = 1.0,
    edge_weight: float = 0.5,
    edge_percentile: float = 80.0,
    edge_dilate: int = 1,
    edge_min_pixels: int = 64,
    threshold_min: float = 0.0,
    threshold_max: float = 1.0,
    threshold_scale: float = 1.0,
    threshold_bias: float = 0.0,
    threshold_max_foreground_ratio: float = 0.0,
    threshold_candidate_percentile: float = 80.0,
) -> torch.Tensor:
    score = score.float().clamp(0, 1)
    threshold_min = float(threshold_min)
    threshold_max = float(threshold_max)
    def finish(threshold: torch.Tensor) -> torch.Tensor:
        threshold = threshold * float(threshold_scale) + float(threshold_bias)
        if threshold_max_foreground_ratio and threshold_max_foreground_ratio > 0:
            ratio = float(np.clip(threshold_max_foreground_ratio, 1e-6, 1.0))
            mass_threshold = torch.quantile(score, 1.0 - ratio)
            threshold = torch.maximum(threshold, mass_threshold)
        return threshold.clamp(threshold_min, threshold_max)

    if threshold_abs is not None:
        return torch.tensor(float(threshold_abs), device=score.device, dtype=score.dtype).clamp(threshold_min, threshold_max)
    if mode == "percentile":
        return finish(torch.quantile(score, percentile / 100.0))
    if mode == "mean_std":
        return finish(score.mean() + std_factor * score.std())
    if mode == "otsu":
        return finish(_otsu_threshold(score))
    if mode == "otsu_edge":
        return finish(_edge_aware_otsu_threshold(
            score,
            edge_weight=edge_weight,
            edge_percentile=edge_percentile,
            edge_dilate=edge_dilate,
            edge_min_pixels=edge_min_pixels,
        ))
    if mode == "candidate_otsu":
        support_cut = torch.quantile(score, float(threshold_candidate_percentile) / 100.0)
        candidates = score[score >= support_cut]
        return finish(_otsu_threshold(candidates))
    if mode == "adaptive":
        # Robust per-image threshold: combine global/edge Otsu with distribution statistics.
        otsu_edge = _edge_aware_otsu_threshold(
            score,
            edge_weight=edge_weight,
            edge_percentile=edge_percentile,
            edge_dilate=edge_dilate,
            edge_min_pixels=edge_min_pixels,
        )
        pct = torch.quantile(score, percentile / 100.0)
        mean_std = (score.mean() + std_factor * score.std()).clamp(0, 1)
        return finish(torch.median(torch.stack([otsu_edge, pct, mean_std])))
    raise ValueError(f"Unknown threshold mode: {mode}")


def _otsu_threshold(values: torch.Tensor, bins: int = 128) -> torch.Tensor:
    values = values.float().clamp(0, 1)
    values = values[torch.isfinite(values)]
    if values.numel() == 0:
        return torch.tensor(1.0, device=values.device, dtype=torch.float32)
    if float((values.max() - values.min()).detach().cpu()) < 1e-6:
        return values.mean()
    hist = torch.histc(values, bins=bins, min=0, max=1)
    prob = hist / (hist.sum() + 1e-12)
    centers = torch.linspace(0, 1, steps=bins, device=values.device, dtype=values.dtype)
    omega = torch.cumsum(prob, dim=0)
    mu = torch.cumsum(prob * centers, dim=0)
    mu_t = mu[-1]
    sigma_b = (mu_t * omega - mu).pow(2) / (omega * (1 - omega) + 1e-12)
    return centers[torch.argmax(sigma_b)]


def _score_edge_mask(
    score: torch.Tensor,
    edge_percentile: float = 80.0,
    edge_dilate: int = 1,
    edge_min_pixels: int = 64,
) -> Optional[torch.Tensor]:
    """Extract a stable high-gradient neighborhood from a score map."""
    score_4d = score.float().view(1, 1, *score.shape[-2:])
    sobel_x = torch.tensor(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
        device=score.device,
        dtype=score.dtype,
    ).view(1, 1, 3, 3)
    sobel_y = torch.tensor(
        [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]],
        device=score.device,
        dtype=score.dtype,
    ).view(1, 1, 3, 3)
    gx = F.conv2d(score_4d, sobel_x, padding=1)
    gy = F.conv2d(score_4d, sobel_y, padding=1)
    grad = torch.sqrt(gx.pow(2) + gy.pow(2)).squeeze(0).squeeze(0)
    if float(grad.max().detach().cpu()) < 1e-8:
        return None
    edge_th = torch.quantile(grad.flatten(), float(edge_percentile) / 100.0)
    edge = grad > edge_th
    if edge_dilate > 0:
        kernel = 2 * int(edge_dilate) + 1
        edge = F.max_pool2d(edge.float().view(1, 1, *edge.shape), kernel_size=kernel, stride=1, padding=int(edge_dilate))
        edge = edge.squeeze(0).squeeze(0).bool()
    if int(edge.sum().detach().cpu()) < int(edge_min_pixels):
        return None
    return edge


def _edge_aware_otsu_threshold(
    score: torch.Tensor,
    edge_weight: float = 0.5,
    edge_percentile: float = 80.0,
    edge_dilate: int = 1,
    edge_min_pixels: int = 64,
) -> torch.Tensor:
    global_th = _otsu_threshold(score)
    edge = _score_edge_mask(score, edge_percentile=edge_percentile, edge_dilate=edge_dilate, edge_min_pixels=edge_min_pixels)
    if edge is None:
        return global_th
    edge_th = _otsu_threshold(score[edge])
    weight = float(np.clip(edge_weight, 0.0, 1.0))
    return ((1.0 - weight) * global_th + weight * edge_th).clamp(0, 1)


def _connected_components(mask: np.ndarray) -> Tuple[np.ndarray, int]:
    try:
        import cv2

        num, labels = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
        return labels, num - 1
    except Exception:
        try:
            from scipy import ndimage

            labels, num = ndimage.label(mask.astype(bool), structure=np.ones((3, 3), dtype=np.uint8))
            return labels.astype(np.int32), int(num)
        except Exception:
            return mask.astype(np.int32), int(mask.any())


def _region_stat(values: np.ndarray, stat: str = "mean", topk_ratio: float = 0.2) -> float:
    if values.size == 0:
        return 0.0
    if stat == "max":
        return float(values.max())
    if stat == "topk_mean":
        k = max(1, int(values.size * topk_ratio))
        return float(np.partition(values, -k)[-k:].mean())
    return float(values.mean())


def component_consensus(
    score: torch.Tensor,
    threshold: torch.Tensor,
    min_area: int = 4,
    max_area: int = 262144,
    score_stat: str = "mean",
    topk_ratio: float = 0.2,
) -> torch.Tensor:
    """Remove tiny/noisy thresholded components using region-level scores."""
    score_np = score.detach().float().cpu().numpy()
    base = score_np > float(threshold.detach().cpu())
    labels, num = _connected_components(base)
    out = np.zeros_like(base, dtype=np.uint8)
    for lab in range(1, num + 1):
        region = labels == lab
        area = int(region.sum())
        if area < min_area or area > max_area:
            continue
        if _region_stat(score_np[region], score_stat, topk_ratio) > float(threshold.detach().cpu()):
            out[region] = 1
    return torch.from_numpy(out).to(score.device).bool()


def acf_component_consensus(
    score: torch.Tensor,
    threshold: torch.Tensor,
    min_area: int = 4,
    max_area: int = 262144,
    score_stat: str = "mean",
    topk_ratio: float = 0.2,
    conf_percentile: float = 50.0,
    conf_lambda: float = 1.0,
    conf_min: float = 0.0,
    conf_max: float = 1.0,
    cv_max: float = 2.0,
) -> torch.Tensor:
    """AdaptOVCD-style confidence and spatial consistency filtering."""
    score_np = score.detach().float().cpu().numpy()
    threshold_value = float(threshold.detach().cpu())
    base = score_np > threshold_value
    positive = score_np[base]
    if positive.size == 0:
        return torch.zeros_like(score, dtype=torch.bool)

    conf_lambda = max(float(conf_lambda), 1e-6)
    conf_th = float(np.quantile(positive, float(conf_percentile) / 100.0) / conf_lambda)
    conf_th = float(np.clip(conf_th, float(conf_min), float(conf_max)))

    labels, num = _connected_components(base)
    out = np.zeros_like(base, dtype=np.uint8)
    for lab in range(1, num + 1):
        region = labels == lab
        area = int(region.sum())
        if area < min_area or area > max_area:
            continue
        values = score_np[region]
        mean_conf = float(values.mean()) if values.size else 0.0
        cv = float(values.std() / (mean_conf + 1e-6)) if values.size else float("inf")
        if cv > float(cv_max):
            continue
        if mean_conf < conf_th:
            continue
        if _region_stat(values, score_stat, topk_ratio) > threshold_value:
            out[region] = 1
    return torch.from_numpy(out).to(score.device).bool()


def acf_region_consensus(
    score: torch.Tensor,
    seed_threshold: torch.Tensor,
    min_area: int = 4,
    max_area: int = 262144,
    score_stat: str = "topk_mean",
    topk_ratio: float = 0.2,
    conf_percentile: float = 70.0,
    conf_lambda: float = 1.0,
    conf_min: float = 0.0,
    conf_max: float = 1.0,
    cv_max: float = 2.0,
) -> torch.Tensor:
    """Region-level adaptive filtering inspired by AdaptOVCD ACF and StructFilter.

    The seed threshold only proposes candidate components. The final decision
    is made over component-level confidence statistics, which is less sensitive
    to sparse SAM3 posterior pixels than direct pixel Otsu.
    """
    score_np = score.detach().float().cpu().numpy()
    seed_value = float(seed_threshold.detach().cpu())
    base = score_np > seed_value
    labels, num = _connected_components(base)

    regions = []
    region_scores = []
    for lab in range(1, num + 1):
        region = labels == lab
        area = int(region.sum())
        if area < min_area or area > max_area:
            continue
        values = score_np[region]
        mean_conf = float(values.mean()) if values.size else 0.0
        cv = float(values.std() / (mean_conf + 1e-6)) if values.size else float("inf")
        if cv > float(cv_max):
            continue
        stat_value = _region_stat(values, score_stat, topk_ratio)
        regions.append(region)
        region_scores.append(stat_value)

    if not region_scores:
        return torch.zeros_like(score, dtype=torch.bool)

    conf_lambda = max(float(conf_lambda), 1e-6)
    region_scores_np = np.asarray(region_scores, dtype=np.float32)
    adaptive_conf = float(np.quantile(region_scores_np, float(conf_percentile) / 100.0) / conf_lambda)
    adaptive_conf = float(np.clip(adaptive_conf, float(conf_min), float(conf_max)))
    keep_threshold = max(seed_value, adaptive_conf)

    out = np.zeros_like(base, dtype=np.uint8)
    for region, stat_value in zip(regions, region_scores):
        if stat_value >= keep_threshold:
            out[region] = 1
    return torch.from_numpy(out).to(score.device).bool()


def slic_consensus(
    score: torch.Tensor,
    image_np: np.ndarray,
    threshold: torch.Tensor,
    n_segments: int = 512,
    min_area: int = 4,
    max_area: int = 262144,
    score_stat: str = "mean",
    topk_ratio: float = 0.2,
) -> torch.Tensor:
    try:
        from skimage.segmentation import slic
    except Exception:
        return component_consensus(score, threshold, min_area, max_area, score_stat, topk_ratio)

    score_np = score.detach().float().cpu().numpy()
    labels = slic(image_np, n_segments=n_segments, compactness=10.0, start_label=1, channel_axis=-1)
    out = np.zeros(score_np.shape, dtype=np.uint8)
    for lab in np.unique(labels):
        region = labels == lab
        area = int(region.sum())
        if area < min_area or area > max_area:
            continue
        if _region_stat(score_np[region], score_stat, topk_ratio) > float(threshold.detach().cpu()):
            out[region] = 1
    return torch.from_numpy(out).to(score.device).bool()


def decode_ovcd_score(
    score: torch.Tensor,
    threshold_mode: str = "percentile",
    threshold_percentile: float = 85.0,
    threshold_abs: Optional[float] = None,
    threshold_std_factor: float = 1.0,
    threshold_edge_weight: float = 0.5,
    threshold_edge_percentile: float = 80.0,
    threshold_edge_dilate: int = 1,
    threshold_edge_min_pixels: int = 64,
    threshold_min: float = 0.0,
    threshold_max: float = 1.0,
    threshold_scale: float = 1.0,
    threshold_bias: float = 0.0,
    threshold_max_foreground_ratio: float = 0.0,
    threshold_candidate_percentile: float = 80.0,
    min_score: float = 0.02,
    presence: Optional[torch.Tensor] = None,
    presence_min: float = 0.0,
    region_mode: str = "component",
    image_np: Optional[np.ndarray] = None,
    slic_segments: int = 512,
    region_score: str = "mean",
    min_region_area: int = 4,
    max_region_area: int = 262144,
    topk_ratio: float = 0.2,
    acf_conf_percentile: float = 50.0,
    acf_conf_lambda: float = 1.0,
    acf_conf_min: float = 0.0,
    acf_conf_max: float = 1.0,
    acf_cv_max: float = 2.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Decode a class score map to a binary mask."""
    score = score.float().clamp(0, 1)
    threshold = compute_score_threshold(
        score, mode=threshold_mode, percentile=threshold_percentile,
        threshold_abs=threshold_abs, std_factor=threshold_std_factor,
        edge_weight=threshold_edge_weight,
        edge_percentile=threshold_edge_percentile,
        edge_dilate=threshold_edge_dilate,
        edge_min_pixels=threshold_edge_min_pixels,
        threshold_min=threshold_min,
        threshold_max=threshold_max,
        threshold_scale=threshold_scale,
        threshold_bias=threshold_bias,
        threshold_max_foreground_ratio=threshold_max_foreground_ratio,
        threshold_candidate_percentile=threshold_candidate_percentile)
    if score.max() < min_score:
        return torch.zeros_like(score, dtype=torch.bool), threshold
    if presence is not None and float(presence.detach().cpu()) < presence_min:
        return torch.zeros_like(score, dtype=torch.bool), threshold

    if region_mode == "none":
        return score > threshold, threshold
    if region_mode == "slic" and image_np is not None:
        return slic_consensus(
            score, image_np, threshold, n_segments=slic_segments,
            min_area=min_region_area, max_area=max_region_area,
            score_stat=region_score, topk_ratio=topk_ratio), threshold
    if region_mode == "component":
        return component_consensus(
            score, threshold, min_area=min_region_area,
            max_area=max_region_area, score_stat=region_score,
            topk_ratio=topk_ratio), threshold
    if region_mode == "acf_component":
        return acf_component_consensus(
            score,
            threshold,
            min_area=min_region_area,
            max_area=max_region_area,
            score_stat=region_score,
            topk_ratio=topk_ratio,
            conf_percentile=acf_conf_percentile,
            conf_lambda=acf_conf_lambda,
            conf_min=acf_conf_min,
            conf_max=acf_conf_max,
            cv_max=acf_cv_max,
        ), threshold
    if region_mode == "acf_region":
        return acf_region_consensus(
            score,
            threshold,
            min_area=min_region_area,
            max_area=max_region_area,
            score_stat=region_score,
            topk_ratio=topk_ratio,
            conf_percentile=acf_conf_percentile,
            conf_lambda=acf_conf_lambda,
            conf_min=acf_conf_min,
            conf_max=acf_conf_max,
            cv_max=acf_cv_max,
        ), threshold
    raise ValueError(f"Unknown region_mode: {region_mode}")
