import torch
import torch.nn as nn
import torch.nn.functional as F


def cosine_loss(pred, target):
    """Cosine similarity loss (1 - cosine_sim). Lower is better."""
    pred_flat = pred.flatten(2)
    target_flat = target.flatten(2)
    cos_sim = F.cosine_similarity(pred_flat, target_flat, dim=1)
    return (1 - cos_sim).mean()


def reconstruction_loss(pred, target):
    """MSE loss: change features should preserve base feature structure."""
    return F.mse_loss(pred, target)


def total_variation_loss(feat):
    """Total variation for spatial smoothness."""
    diff_h = feat[:, :, 1:, :] - feat[:, :, :-1, :]
    diff_w = feat[:, :, :, 1:] - feat[:, :, :, :-1]
    return diff_h.square().mean() + diff_w.square().mean()


def change_aware_contrastive_loss(change_fpn, base_fpn, cos_sim_original):
    """
    Change-aware contrastive loss using original T1/T2 similarity as proxy.

    The residual (change_fpn - base_fpn) represents the learned change signal.
    - Where T1≈T2 (unchanged, high cos_sim): residual should be small
    - Where T1≠T2 (changed, low cos_sim): residual can be large

    Args:
        change_fpn: [B, C, H, W] output of change module
        base_fpn: [B, C, H, W] base feature (T1)
        cos_sim_original: [B, 1, H, W] cosine similarity of original T1/T2 features
    """
    residual = change_fpn - base_fpn  # the learned change signal

    # Residual magnitude per pixel
    residual_mag = residual.norm(dim=1, keepdim=True)  # [B, 1, H, W]

    # Where unchanged (high cos_sim): penalize large residuals
    # Where changed (low cos_sim): allow large residuals
    unchanged_weight = cos_sim_original.clamp(0, 1)  # high = unchanged
    loss = (unchanged_weight * residual_mag).mean()

    return loss


def change_prior_alignment_loss(change_fpn, base_fpn, fpn_T1, fpn_T2):
    """
    Align residual strength with the frozen backbone's bi-temporal dissimilarity.

    The previous contrastive term only suppressed residuals in unchanged regions,
    which makes a near-zero residual a valid local optimum. This term gives the
    module a positive target: residuals should be spatially stronger where the
    original T1/T2 FPN features disagree.
    """
    residual_mag = (change_fpn - base_fpn).norm(dim=1, keepdim=True)

    with torch.no_grad():
        target = (1 - F.cosine_similarity(fpn_T1, fpn_T2, dim=1).unsqueeze(1)).clamp(0, 2) / 2

    eps = 1e-6
    residual_norm = residual_mag / (residual_mag.mean(dim=(-2, -1), keepdim=True).detach() + eps)
    target_norm = target / (target.mean(dim=(-2, -1), keepdim=True) + eps)
    return F.smooth_l1_loss(residual_norm, target_norm)


def multi_level_change_prior_alignment_loss(change_fpn, fpn_T1, fpn_T2):
    """
    Positive anti-collapse objective for all FPN levels.

    Reconstruction and unchanged-region penalties are necessary to keep the
    output on the SAM3 feature manifold, but alone they make a near-zero
    residual attractive. This loss explicitly asks residual energy to follow
    the frozen encoder's bi-temporal dissimilarity prior.
    """
    total = 0
    eps = 1e-6
    for pred, base, feat_a, feat_b in zip(change_fpn, fpn_T1, fpn_T1, fpn_T2):
        residual_mag = (pred - base).norm(dim=1, keepdim=True)
        with torch.no_grad():
            target = (1 - F.cosine_similarity(feat_a, feat_b, dim=1).unsqueeze(1)).clamp(0, 2) / 2
            target_norm = target / (target.mean(dim=(-2, -1), keepdim=True) + eps)
            hi = torch.quantile(target.flatten(1), 0.80, dim=1).view(-1, 1, 1, 1)
            lo = torch.quantile(target.flatten(1), 0.50, dim=1).view(-1, 1, 1, 1)
            pos_mask = (target >= hi).float()
            neg_mask = (target <= lo).float()

        residual_norm = residual_mag / (residual_mag.mean(dim=(-2, -1), keepdim=True).detach() + eps)
        shape_loss = F.smooth_l1_loss(residual_norm, target_norm)

        # Encourage non-trivial residuals at high-prior locations while keeping
        # low-prior areas quiet. This is the main guard against zero residuals.
        pos_loss = (pos_mask * F.relu(1.0 - residual_norm)).sum() / (pos_mask.sum() + eps)
        neg_loss = (neg_mask * F.relu(residual_norm - 0.5)).sum() / (neg_mask.sum() + eps)
        total = total + shape_loss + 0.5 * pos_loss + 0.25 * neg_loss
    return total / len(change_fpn)


def feature_distribution_loss(change_fpn, fpn_T1, fpn_T2):
    """
    Keep translated features on the frozen SAM3 FPN manifold.

    This is a lightweight decoder-compatibility proxy: the decoder expects FPN
    statistics similar to the frozen encoder. We match per-channel mean/std to
    the average bi-temporal FPN statistics instead of only reconstructing T1.
    """
    total = 0
    for pred, feat_a, feat_b in zip(change_fpn, fpn_T1, fpn_T2):
        target = 0.5 * (feat_a + feat_b)
        pred_mean = pred.mean(dim=(-2, -1))
        pred_std = pred.std(dim=(-2, -1), unbiased=False)
        target_mean = target.mean(dim=(-2, -1)).detach()
        target_std = target.std(dim=(-2, -1), unbiased=False).detach()
        total = total + F.l1_loss(pred_mean, target_mean) + F.l1_loss(pred_std, target_std)
    return total / len(change_fpn)


def residual_ratio_loss(change_fpn, fpn_T1, max_residual_ratio=0.25):
    """
    Penalize decoder-facing residuals that get too large relative to frozen FPN.

    Frozen SAM3 decoder compatibility depends more on bounded residual energy
    than on the absolute residual value. This loss makes feature injection fail
    closed: the BCD head can still learn change, while decoder input stays near
    the original SAM3 feature manifold.
    """
    total = 0
    eps = 1e-6
    for pred, base in zip(change_fpn, fpn_T1):
        residual = pred - base
        residual_power = residual.float().square().mean(dim=(1, 2, 3))
        base_power = base.float().square().mean(dim=(1, 2, 3)).detach()
        ratio_sq = residual_power / (base_power + eps)
        total = total + F.relu(ratio_sq - max_residual_ratio ** 2).mean()
    return total / len(change_fpn)


def residual_mask_prior_loss(
    change_fpn,
    fpn_T1,
    masks,
    positive_margin=0.12,
    negative_margin=0.02,
):
    """
    Encourage residual energy to be non-trivial in GT changed regions and
    small in unchanged regions.

    This adds the missing positive scale signal that the current scale-free
    prior terms do not provide.
    """
    if masks.ndim == 3:
        masks = masks.unsqueeze(1)
    masks = masks.float()
    total = 0
    eps = 1e-6
    for pred, base in zip(change_fpn, fpn_T1):
        target = F.interpolate(masks, size=pred.shape[-2:], mode="nearest")
        residual = pred - base
        residual_mag = residual.float().norm(dim=1, keepdim=True)
        base_rms = base.float().square().mean(dim=(1, 2, 3), keepdim=True).sqrt().detach()
        residual_norm = residual_mag / (base_rms + eps)
        pos_mask = (target > 0.5).float()
        neg_mask = (target <= 0.5).float()
        pos_loss = (pos_mask * F.relu(positive_margin - residual_norm)).sum() / (pos_mask.sum() + eps)
        neg_loss = (neg_mask * F.relu(residual_norm - negative_margin)).sum() / (neg_mask.sum() + eps)
        total = total + pos_loss + 0.25 * neg_loss
    return total / len(change_fpn)


def change_logits_consistency_loss(change_logits, change_logits_jit, jitter_params, feature_jitter):
    """
    Align change logits across jittered feature views.

    The original logits are transformed into the jittered frame, then compared
    against the prediction from jittered features. This stabilizes the BCD
    prior used at inference time.
    """
    if change_logits is None or change_logits_jit is None:
        return None
    ref = feature_jitter.transform_tensor(change_logits.detach(), jitter_params, inverse=False)
    if ref.shape[-2:] != change_logits_jit.shape[-2:]:
        ref = F.interpolate(ref, size=change_logits_jit.shape[-2:], mode="bilinear", align_corners=False)
    ref_prob = torch.sigmoid(ref.float())
    jit_prob = torch.sigmoid(change_logits_jit.float())
    return F.smooth_l1_loss(jit_prob, ref_prob)


@torch.no_grad()
def decoder_compatibility_metrics(change_fpn, fpn_T1, fpn_T2):
    """Batch metrics for checking whether change_fpn remains decoder-compatible."""
    eps = 1e-6
    residual_ratios = []
    mean_drifts = []
    std_drifts = []
    cos_t1 = []
    cos_t2 = []

    for pred, feat_a, feat_b in zip(change_fpn, fpn_T1, fpn_T2):
        residual = pred - feat_a
        residual_rms = residual.float().square().mean(dim=(1, 2, 3)).sqrt()
        base_rms = feat_a.float().square().mean(dim=(1, 2, 3)).sqrt()
        residual_ratios.append((residual_rms / (base_rms + eps)).mean())

        target = 0.5 * (feat_a + feat_b)
        pred_mean = pred.mean(dim=(-2, -1))
        pred_std = pred.std(dim=(-2, -1), unbiased=False)
        target_mean = target.mean(dim=(-2, -1))
        target_std = target.std(dim=(-2, -1), unbiased=False)
        mean_drifts.append((pred_mean - target_mean).abs().mean())
        std_drifts.append((pred_std - target_std).abs().mean())

        cos_t1.append(F.cosine_similarity(pred.flatten(2), feat_a.flatten(2), dim=1).mean())
        cos_t2.append(F.cosine_similarity(pred.flatten(2), feat_b.flatten(2), dim=1).mean())

    return {
        'residual_ratio': torch.stack(residual_ratios).mean(),
        'mean_drift': torch.stack(mean_drifts).mean(),
        'std_drift': torch.stack(std_drifts).mean(),
        'cos_t1': torch.stack(cos_t1).mean(),
        'cos_t2': torch.stack(cos_t2).mean(),
    }


@torch.no_grad()
def change_head_separation_metrics(change_logits, target):
    """Check whether the BCD head assigns higher scores to changed pixels."""
    if target.ndim == 3:
        target = target.unsqueeze(1)
    if change_logits.shape[-2:] != target.shape[-2:]:
        change_logits = F.interpolate(
            change_logits, size=target.shape[-2:], mode='bilinear', align_corners=False)
    score = torch.sigmoid(change_logits.float())
    target = target.float()
    pos = target > 0.5
    neg = ~pos
    pos_score = score[pos].mean() if pos.any() else score.new_tensor(0.0)
    neg_score = score[neg].mean() if neg.any() else score.new_tensor(0.0)
    return {
        'score_pos': pos_score,
        'score_neg': neg_score,
        'score_gap': pos_score - neg_score,
        'pos_frac': target.mean(),
    }


@torch.no_grad()
def gate_separation_metrics(gate_map, target):
    """Check whether a spatial gate is higher on changed pixels."""
    if target.ndim == 3:
        target = target.unsqueeze(1)
    if gate_map.shape[-2:] != target.shape[-2:]:
        gate_map = F.interpolate(gate_map, size=target.shape[-2:], mode='bilinear', align_corners=False)
    gate = gate_map.float().clamp(0, 1)
    target = target.float()
    pos = target > 0.5
    neg = ~pos
    pos_score = gate[pos].mean() if pos.any() else gate.new_tensor(0.0)
    neg_score = gate[neg].mean() if neg.any() else gate.new_tensor(0.0)
    return {
        'gate_pos': pos_score,
        'gate_neg': neg_score,
        'gate_gap': pos_score - neg_score,
        'gate_mean': gate.mean(),
    }


def dice_loss_with_logits(logits, target, eps=1e-6):
    probs = torch.sigmoid(logits)
    target = target.float()
    dims = tuple(range(1, probs.ndim))
    intersection = (probs * target).sum(dim=dims)
    union = probs.sum(dim=dims) + target.sum(dim=dims)
    return (1 - (2 * intersection + eps) / (union + eps)).mean()


def balanced_bce_with_logits(logits, target, max_pos_weight=20.0):
    target = target.float()
    pos = target.sum()
    neg = target.numel() - pos
    pos_weight = (neg / (pos + 1e-6)).clamp(1.0, max_pos_weight).to(logits.device)
    return F.binary_cross_entropy_with_logits(logits, target, pos_weight=pos_weight)


def change_supervision_loss(
    change_logits,
    target,
    lambda_bce=1.0,
    lambda_dice=1.0,
    resize_target_to_logits=False,
):
    """
    Supervised category-agnostic change objective.

    This is the part FeatUp/SimFeatUp do not provide. It teaches the temporal
    adapter which bi-temporal differences are valid land-cover changes.
    """
    if target.ndim == 3:
        target = target.unsqueeze(1)
    target = target.float()
    if change_logits.shape[-2:] != target.shape[-2:]:
        if resize_target_to_logits:
            target = F.interpolate(target, size=change_logits.shape[-2:], mode='nearest')
        else:
            change_logits = F.interpolate(
                change_logits, size=target.shape[-2:], mode='bilinear', align_corners=False)
    bce = balanced_bce_with_logits(change_logits, target)
    dice = dice_loss_with_logits(change_logits, target)
    total = lambda_bce * bce + lambda_dice * dice
    return total, {'bce': bce, 'dice': dice, 'change_total': total}


def compute_original_cos_sim(fpn_T1, fpn_T2, level=-1):
    """
    Compute per-pixel cosine similarity between original T1/T2 features.
    Used as soft proxy for changed/unchanged regions.
    """
    cos_sim = F.cosine_similarity(fpn_T1[level], fpn_T2[level], dim=1).unsqueeze(1)
    return cos_sim


class STEEConsistencyLoss(nn.Module):
    """
    Training loss for the Semantic-Temporal Evidence Encoder (STEE).

    Inspired by FeatUp's multi-view consistency approach:
    - L_consist: change features should be consistent across jittered views
    - L_recon: change features should preserve base feature structure
    - L_tv: change signal should be spatially smooth
    - L_contrast: change signal should be suppressed where T1≈T2
    - L_prior: residual magnitude should follow the frozen feature-change prior
    - L_dist: translated FPN statistics should remain SAM3-decoder compatible
    - L_mask_prior: residual energy should be non-trivial in GT change regions
    - L_logit_consist: explicit change logits should stay stable under jitter
    """

    def __init__(self, lambda_consist=1.0, lambda_recon=0.1,
                 lambda_tv=0.05, lambda_contrast=0.5,
                 lambda_prior=1.0, lambda_dist=0.05,
                 lambda_decoder=0.1, lambda_mask_prior=0.5,
                 lambda_logit_consist=0.1, mask_positive_margin=0.12,
                 mask_negative_margin=0.02,
                 max_residual_ratio=0.25):
        super().__init__()
        self.lambda_consist = lambda_consist
        self.lambda_recon = lambda_recon
        self.lambda_tv = lambda_tv
        self.lambda_contrast = lambda_contrast
        self.lambda_prior = lambda_prior
        self.lambda_dist = lambda_dist
        self.lambda_decoder = lambda_decoder
        self.lambda_mask_prior = lambda_mask_prior
        self.lambda_logit_consist = lambda_logit_consist
        self.mask_positive_margin = mask_positive_margin
        self.mask_negative_margin = mask_negative_margin
        self.max_residual_ratio = max_residual_ratio

    def forward(self, change_fpn, change_fpn_jit, fpn_T1, fpn_T2,
                jitter_params, feature_jitter, masks=None,
                change_logits=None, change_logits_jit=None, weights=None):
        """
        Args:
            change_fpn: list of 3 tensors - change module output (unjittered)
            change_fpn_jit: list of 3 tensors - change module output (from jittered input)
            fpn_T1, fpn_T2: original cached FPN features
            jitter_params: dict from FeatureJitter
            feature_jitter: FeatureJitter instance for undo
            masks: optional binary change mask supervision
            change_logits: optional supervised change logits from unjittered input
            change_logits_jit: optional change logits from jittered input
            weights: optional dict overriding the default internal loss weights
        """
        losses = {}

        # L1: Multi-view consistency (FeatUp core idea)
        # Jittered → unjittered should match direct output
        change_fpn_undo = feature_jitter._apply_undo(change_fpn_jit, jitter_params)
        L_consist = 0
        for i in range(len(change_fpn)):
            L_consist += cosine_loss(change_fpn_undo[i], change_fpn[i].detach())
        L_consist /= len(change_fpn)
        losses['consist'] = L_consist

        # L2: Reconstruction (preserve base feature structure)
        L_recon = 0
        for i in range(len(change_fpn)):
            L_recon += reconstruction_loss(change_fpn[i], fpn_T1[i])
        L_recon /= len(change_fpn)
        losses['recon'] = L_recon

        # L3: Total variation (spatial smoothness of change signal)
        L_tv = 0
        for i in range(len(change_fpn)):
            residual = change_fpn[i] - fpn_T1[i]
            L_tv += total_variation_loss(residual)
        L_tv /= len(change_fpn)
        losses['tv'] = L_tv

        # L4: Change-aware contrastive
        # Change signal should be strong only where T1≠T2
        cos_sim = compute_original_cos_sim(fpn_T1, fpn_T2, level=-1)
        L_contrast = change_aware_contrastive_loss(
            change_fpn[-1], fpn_T1[-1], cos_sim)
        losses['contrast'] = L_contrast

        # L5: Positive change prior alignment
        L_prior = multi_level_change_prior_alignment_loss(change_fpn, fpn_T1, fpn_T2)
        losses['prior'] = L_prior

        # L6: Decoder-compatibility proxy in feature-statistics space
        L_dist = feature_distribution_loss(change_fpn, fpn_T1, fpn_T2)
        losses['dist'] = L_dist

        # L7: Harder decoder compatibility guard in residual-energy space
        L_decoder = residual_ratio_loss(
            change_fpn, fpn_T1, max_residual_ratio=self.max_residual_ratio)
        losses['decoder'] = L_decoder

        # L8: Positive residual-scale prior from GT change masks.
        L_mask_prior = None
        if masks is not None and self.lambda_mask_prior > 0:
            L_mask_prior = residual_mask_prior_loss(
                change_fpn,
                fpn_T1,
                masks,
                positive_margin=self.mask_positive_margin,
                negative_margin=self.mask_negative_margin,
            )
            losses['mask_prior'] = L_mask_prior

        # L9: Keep the explicit BCD head stable under feature-space jitter.
        L_logit_consist = None
        if change_logits is not None and change_logits_jit is not None and self.lambda_logit_consist > 0:
            L_logit_consist = change_logits_consistency_loss(
                change_logits,
                change_logits_jit,
                jitter_params,
                feature_jitter,
            )
            losses['logit_consist'] = L_logit_consist

        # Total
        weights = weights or {}
        w_consist = weights.get("consist", self.lambda_consist)
        w_recon = weights.get("recon", self.lambda_recon)
        w_tv = weights.get("tv", self.lambda_tv)
        w_contrast = weights.get("contrast", self.lambda_contrast)
        w_prior = weights.get("prior", self.lambda_prior)
        w_dist = weights.get("dist", self.lambda_dist)
        w_decoder = weights.get("decoder", self.lambda_decoder)
        w_mask_prior = weights.get("mask_prior", self.lambda_mask_prior)
        w_logit_consist = weights.get("logit_consist", self.lambda_logit_consist)

        total = (w_consist * L_consist +
                 w_recon * L_recon +
                 w_tv * L_tv +
                 w_contrast * L_contrast +
                 w_prior * L_prior +
                 w_dist * L_dist +
                 w_decoder * L_decoder)
        if L_mask_prior is not None:
            total = total + w_mask_prior * L_mask_prior
        if L_logit_consist is not None:
            total = total + w_logit_consist * L_logit_consist
        losses['total'] = total

        return total, losses


# Public alias for callers that refer to the Change Adapter loss directly.
ChangeAdapterLoss = STEEConsistencyLoss
