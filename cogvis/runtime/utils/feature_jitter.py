import torch
import torch.nn.functional as F
import random


class FeatureJitter:
    """
    Feature-level spatial jittering for unsupervised consistency training.
    Applies small spatial transforms (pad+crop, flip) consistently to all FPN levels.
    The same jitter is applied to both T1 and T2 features.

    The sampled transform is defined at the highest-resolution FPN level and
    scaled for lower-resolution levels. Using the same absolute pixel offset at
    288x288 and 72x72 would create different image-space transforms.
    """

    def __init__(self, max_pad=4, flip_prob=0.5):
        self.max_pad = max_pad
        self.flip_prob = flip_prob

    def jitter_pair(self, fpn_T1, fpn_T2):
        """
        Apply the same jitter to both T1 and T2 features.

        Args:
            fpn_T1: list of [B, C, H_i, W_i]
            fpn_T2: list of [B, C, H_i, W_i]
        Returns:
            jittered_T1, jittered_T2: jittered features
            jitter_params: for undo
        """
        base_pad = random.randint(0, self.max_pad)
        do_flip = random.random() < self.flip_prob

        base_feat = next((feat for feat in fpn_T1 if feat is not None), None)
        if base_feat is None:
            raise ValueError("FeatureJitter received no loaded FPN tensors")
        _, _, base_h, base_w = base_feat.shape
        if base_pad > 0:
            base_offset_h = random.randint(0, 2 * base_pad)
            base_offset_w = random.randint(0, 2 * base_pad)
        else:
            base_offset_h = 0
            base_offset_w = 0

        level_params = []
        for feat in fpn_T1:
            if feat is None:
                level_params.append(None)
                continue
            _, _, h, w = feat.shape
            scale_h = h / base_h
            scale_w = w / base_w
            pad_h = int(round(base_pad * scale_h))
            pad_w = int(round(base_pad * scale_w))
            offset_h = int(round(base_offset_h * scale_h))
            offset_w = int(round(base_offset_w * scale_w))
            offset_h = min(max(offset_h, 0), 2 * pad_h)
            offset_w = min(max(offset_w, 0), 2 * pad_w)
            level_params.append((pad_h, pad_w, offset_h, offset_w))

        jitter_params = {
            "pad_pixels": base_pad,
            "do_flip": do_flip,
            "crop_offsets": [None if p is None else (p[2], p[3]) for p in level_params],
            "level_params": level_params,
            "base_shape": (base_h, base_w),
            "base_crop_offset": (base_offset_h, base_offset_w),
        }

        jittered_T1 = self._apply_jitter(fpn_T1, jitter_params)
        jittered_T2 = self._apply_jitter(fpn_T2, jitter_params)

        return jittered_T1, jittered_T2, jitter_params

    def _resolve_level_params(self, height, width, jitter_params):
        base_h, base_w = jitter_params.get("base_shape", (height, width))
        base_pad = jitter_params["pad_pixels"]
        base_oh, base_ow = jitter_params.get("base_crop_offset", (0, 0))

        scale_h = height / max(base_h, 1)
        scale_w = width / max(base_w, 1)
        pad_h = int(round(base_pad * scale_h))
        pad_w = int(round(base_pad * scale_w))
        offset_h = int(round(base_oh * scale_h))
        offset_w = int(round(base_ow * scale_w))
        offset_h = min(max(offset_h, 0), 2 * pad_h)
        offset_w = min(max(offset_w, 0), 2 * pad_w)
        return pad_h, pad_w, offset_h, offset_w

    def _transform_single(self, feat, jitter_params, inverse=False, level_index=None):
        do_flip = jitter_params["do_flip"]
        level_params = jitter_params.get("level_params")
        if (
            level_params is not None
            and level_index is not None
            and level_index < len(level_params)
            and level_params[level_index] is not None
        ):
            pad_h, pad_w, oh, ow = level_params[level_index]
        else:
            height, width = feat.shape[-2:]
            pad_h, pad_w, oh, ow = self._resolve_level_params(height, width, jitter_params)

        H, W = feat.shape[-2:]
        if inverse:
            if do_flip:
                feat = torch.flip(feat, dims=[-1])

            if pad_h > 0 or pad_w > 0:
                feat = F.pad(feat, [pad_w, pad_w, pad_h, pad_h], mode='replicate')
                feat = feat[:, :, 2 * pad_h - oh:2 * pad_h - oh + H,
                            2 * pad_w - ow:2 * pad_w - ow + W]
            return feat

        if pad_h > 0 or pad_w > 0:
            feat = F.pad(feat, [pad_w, pad_w, pad_h, pad_h], mode='replicate')
            feat = feat[:, :, oh:oh + H, ow:ow + W]

        if do_flip:
            feat = torch.flip(feat, dims=[-1])

        return feat

    def transform_tensor(self, tensor, jitter_params, inverse=False):
        """
        Apply the same spatial jitter (or inverse jitter) to a dense tensor.
        Useful for aligning change logits across jittered views.
        """
        if tensor.ndim != 4:
            raise ValueError(f"Expected a 4D tensor, got shape {tuple(tensor.shape)}")
        return self._transform_single(tensor, jitter_params, inverse=inverse, level_index=None)

    def _apply_jitter(self, fpn_list, jitter_params):
        jittered = []
        for i, feat in enumerate(fpn_list):
            if feat is None:
                jittered.append(None)
                continue
            jittered.append(self._transform_single(feat, jitter_params, inverse=False, level_index=i))

        return jittered

    def _apply_undo(self, fpn_list, jitter_params):
        """
        Undo spatial jitter for consistency loss comparison.
        Undoes flip (exact) and crop (approximate — border artifacts from replicate padding).
        """
        unjittered = []
        for i, feat in enumerate(fpn_list):
            if feat is None:
                unjittered.append(None)
                continue
            unjittered.append(self._transform_single(feat, jitter_params, inverse=True, level_index=i))

        return unjittered
