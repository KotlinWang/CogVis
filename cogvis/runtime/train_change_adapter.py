"""
STEE Training

Two modes:
  --phase cache : pre-extract SAM3 FPN features to disk
  --phase train : train SemanticTemporalEvidenceEncoder

Training supports two data modes:
  cached (default): read pre-extracted FPN features from disk
  online (--no_cache): load images, extract FPN features from SAM3 backbone on-the-fly
"""

import os
import sys
import argparse
import json
import time
from pathlib import Path
from contextlib import nullcontext

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from PIL import Image
import numpy as np

project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

from change_module import (
    SemanticTemporalEvidenceEncoder,
    infer_module_type_from_state_dict,
    module_kwargs_from_checkpoint,
    normalize_state_dict_for_stee,
)
from losses import (
    STEEConsistencyLoss,
    change_supervision_loss,
    decoder_compatibility_metrics,
    change_head_separation_metrics,
    gate_separation_metrics,
)
from eval_change_module import compute_bcd, confusion_binary, metrics_from_confusion
from utils.feature_jitter import FeatureJitter
from utils.gpu import select_device, describe_device

from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
from dataset.ovcd import OVCDDataset


# ──────────────────────────────────────────────
# Phase 1: Cache FPN features (optional)
# ──────────────────────────────────────────────

def cache_features(args):
    print("=" * 60)
    print("Phase 1: Caching FPN features from SAM3 encoder")
    print("=" * 60)

    device = select_device()
    print(f"Using device: {describe_device(device)}")

    bpe_path = os.path.join(project_root, "sam3/assets/bpe_simple_vocab_16e6.txt.gz")
    model = build_sam3_image_model(
        bpe_path=bpe_path,
        checkpoint_path=args.sam3_checkpoint,
        device=str(device)
    )
    processor = Sam3Processor(model, confidence_threshold=0.1, device=str(device))

    trainset = OVCDDataset('train', args.dataset_root, "CA-CDD", args.crop_size)

    cache_dir = os.path.join(args.cache_dir, 'fpn_cache')
    os.makedirs(cache_dir, exist_ok=True)

    max_samples = args.max_samples if args.max_samples > 0 else len(trainset)

    with torch.no_grad():
        for idx in tqdm(range(min(max_samples, len(trainset))), desc="Caching"):
            img_id = trainset.ids[idx]
            imgA_path = os.path.join(trainset.root, trainset.A, img_id)
            imgB_path = os.path.join(trainset.root, trainset.B, img_id)

            base_name = os.path.splitext(img_id)[0]
            cache_path = os.path.join(cache_dir, f'{base_name}.pt')
            if os.path.exists(cache_path):
                continue

            imgA_pil = Image.open(imgA_path).convert('RGB')
            imgB_pil = Image.open(imgB_path).convert('RGB')

            state_A = processor.set_image(imgA_pil)
            state_B = processor.set_image(imgB_pil)

            fpn_A = [f.detach().cpu().squeeze(0) for f in state_A["backbone_out"]["backbone_fpn"]]
            fpn_B = [f.detach().cpu().squeeze(0) for f in state_B["backbone_out"]["backbone_fpn"]]

            _save_checkpoint({'fpn_A': fpn_A, 'fpn_B': fpn_B}, cache_path)

    print(f"Cached to {cache_dir}, total: {min(max_samples, len(trainset))}")


# ──────────────────────────────────────────────
# Datasets
# ──────────────────────────────────────────────

def _load_binary_mask(label_dir, base_name, crop_size):
    for ext in ('.png', '.jpg', '.jpeg', '.tif', '.tiff'):
        path = os.path.join(label_dir, base_name + ext)
        if os.path.exists(path):
            mask = Image.open(path).convert('L').resize((crop_size, crop_size), Image.NEAREST)
            mask = torch.from_numpy((np.array(mask) > 0).astype(np.float32)).unsqueeze(0)
            return mask
    raise FileNotFoundError(f"Cannot find binary change label for {base_name} in {label_dir}")


def _normalize_cache_file_name(name):
    name = os.path.basename(str(name).strip())
    if not name:
        return ''
    base, ext = os.path.splitext(name)
    if ext.lower() == '.pt':
        return name
    return base + '.pt'


def _load_cache_file_set(path):
    if not path:
        return None
    with open(path, 'r') as f:
        names = {_normalize_cache_file_name(line) for line in f if line.strip()}
    names.discard('')
    return names


def _parse_ram_cache_levels(value):
    if value is None:
        return []
    value = value.strip().lower()
    if value in ('', 'none', 'false', 'off'):
        return []
    if value == 'all':
        return [0, 1, 2]
    levels = []
    for part in value.split(','):
        part = part.strip()
        if not part:
            continue
        level = int(part)
        if level < 0 or level > 2:
            raise ValueError(f"Invalid FPN level for --ram_cache_levels: {level}")
        levels.append(level)
    return sorted(set(levels))


def _parse_fpn_levels(value, active_levels=None):
    if value is None:
        return [0, 1, 2]
    value = str(value).strip().lower()
    if value in ('', 'all'):
        return [0, 1, 2]
    if value == 'active':
        return _parse_fpn_levels(active_levels if active_levels is not None else 'all')
    levels = []
    for part in value.split(','):
        part = part.strip()
        if not part:
            continue
        level = int(part)
        if level < 0 or level > 2:
            raise ValueError(f"Invalid FPN level: {level}")
        levels.append(level)
    if not levels:
        raise ValueError(f"No FPN levels selected from {value!r}")
    return sorted(set(levels))


def _select_fpn_levels(fpn_list, levels):
    return [fpn_list[i] for i in levels]


def _first_loaded_feature_shape(fpn_list):
    for feat in fpn_list:
        if feat is not None:
            return feat.shape[-2:]
    raise ValueError("No FPN levels were loaded")


def _mem_available_bytes():
    try:
        with open('/proc/meminfo', 'r') as f:
            for line in f:
                if line.startswith('MemAvailable:'):
                    return int(line.split()[1]) * 1024
    except OSError:
        return None
    return None


class CachedFPNDataset(torch.utils.data.Dataset):
    def __init__(self, cache_dir, dataset_root=None, crop_size=512, return_mask=True,
                 max_samples=0, ram_cache_levels=None, ram_cache_max_gb=0.0,
                 include_file=None, exclude_file=None, load_levels=None,
                 label_dir=None, offset=0):
        self.return_mask = return_mask
        self.crop_size = crop_size
        self.label_dir = None
        self._ram_a = {}
        self._ram_b = {}
        self.indices = None
        self.load_levels = set(_parse_fpn_levels(load_levels or 'all'))
        include_names = _load_cache_file_set(include_file)
        exclude_names = _load_cache_file_set(exclude_file) or set()

        def keep_file(name):
            if include_names is not None and name not in include_names:
                return False
            return name not in exclude_names

        if label_dir is not None:
            self.label_dir = str(label_dir)
        elif dataset_root is not None:
            self.label_dir = os.path.join(dataset_root, 'train', 'label')
        mmap_dir = os.path.join(cache_dir, 'fpn_mmap')
        if os.path.isdir(mmap_dir):
            self.mmap_dir = mmap_dir
            meta = torch.load(os.path.join(mmap_dir, 'meta.pt'), weights_only=False)
            mmap_n = meta['n']
            self.shapes = meta['shapes']
            all_files = meta.get('files', [f'{i:05d}.pt' for i in range(mmap_n)])
            self.mmap_dtype = np.dtype(meta.get('dtype', 'float32'))
            self.indices = [i for i, name in enumerate(all_files) if keep_file(name)]
            if offset > 0:
                self.indices = self.indices[int(offset):]
            if max_samples > 0:
                self.indices = self.indices[:max_samples]
            self.files = [all_files[i] for i in self.indices]
            self.n = len(self.indices)
            self._mmaps_a = [np.memmap(os.path.join(mmap_dir, f'fpn_A_{i}.dat'), dtype=self.mmap_dtype, mode='r', shape=(mmap_n, *s))
                              for i, s in enumerate(self.shapes)]
            self._mmaps_b = [np.memmap(os.path.join(mmap_dir, f'fpn_B_{i}.dat'), dtype=self.mmap_dtype, mode='r', shape=(mmap_n, *s))
                              for i, s in enumerate(self.shapes)]
            self._use_mmap = True
        else:
            self.cache_dir = os.path.join(cache_dir, 'fpn_cache')
            self.files = sorted([f for f in os.listdir(self.cache_dir) if f.endswith('.pt') and keep_file(f)])
            if offset > 0:
                self.files = self.files[int(offset):]
            if max_samples > 0:
                self.files = self.files[:max_samples]
            self._use_mmap = False

        ram_cache_levels = ram_cache_levels or []
        if ram_cache_levels:
            self._preload_ram_levels(ram_cache_levels, ram_cache_max_gb)

    def __len__(self):
        return self.n if self._use_mmap else len(self.files)

    def _preload_ram_levels(self, levels, ram_cache_max_gb):
        if not self._use_mmap:
            raise ValueError("--ram_cache_levels requires an mmap cache prepared before training.")

        bytes_needed = 0
        for level in levels:
            level_shape = self.shapes[level]
            bytes_needed += 2 * self.n * int(np.prod(level_shape)) * self.mmap_dtype.itemsize
        gb_needed = bytes_needed / (1024 ** 3)
        if ram_cache_max_gb > 0 and gb_needed > ram_cache_max_gb:
            raise MemoryError(
                f"Requested RAM cache levels {levels} need {gb_needed:.1f} GB, "
                f"larger than --ram_cache_max_gb={ram_cache_max_gb:.1f} GB."
            )

        available = _mem_available_bytes()
        if available is not None and bytes_needed > available * 0.80:
            raise MemoryError(
                f"Requested RAM cache levels {levels} need {gb_needed:.1f} GB, "
                f"but MemAvailable is only {available / (1024 ** 3):.1f} GB. "
                "Use fewer levels, smaller --max_samples, or build/use mmap only."
            )

        print(f"Preloading FPN levels {levels} into RAM: {gb_needed:.1f} GB")
        runs = []
        i = 0
        while i < len(self.indices):
            start_row = self.indices[i]
            j = i + 1
            while j < len(self.indices) and self.indices[j] == self.indices[j - 1] + 1:
                j += 1
            runs.append((i, j, start_row, self.indices[j - 1] + 1))
            i = j
        print(f"RAM preload contiguous row runs: {len(runs)}")
        for level in tqdm(levels, desc="RAM cache levels"):
            shape = (self.n, *self.shapes[level])
            ram_a = np.empty(shape, dtype=self.mmap_dtype)
            ram_b = np.empty(shape, dtype=self.mmap_dtype)
            for dst_start, dst_end, src_start, src_end in runs:
                ram_a[dst_start:dst_end] = self._mmaps_a[level][src_start:src_end]
                ram_b[dst_start:dst_end] = self._mmaps_b[level][src_start:src_end]
            self._ram_a[level] = ram_a
            self._ram_b[level] = ram_b

    def __getitem__(self, idx):
        if self._use_mmap:
            row_idx = self.indices[idx]
            fpn_A = []
            fpn_B = []
            for level, mmap in enumerate(self._mmaps_a):
                if level not in self.load_levels:
                    fpn_A.append(None)
                    continue
                source = self._ram_a.get(level)
                fpn_A.append(torch.from_numpy(source[idx] if source is not None else mmap[row_idx].copy()))
            for level, mmap in enumerate(self._mmaps_b):
                if level not in self.load_levels:
                    fpn_B.append(None)
                    continue
                source = self._ram_b.get(level)
                fpn_B.append(torch.from_numpy(source[idx] if source is not None else mmap[row_idx].copy()))
            file_name = self.files[idx]
        else:
            file_name = self.files[idx]
            data = torch.load(os.path.join(self.cache_dir, file_name), weights_only=False)
            fpn_A, fpn_B = data['fpn_A'], data['fpn_B']
            fpn_A = [feat if i in self.load_levels else None for i, feat in enumerate(fpn_A)]
            fpn_B = [feat if i in self.load_levels else None for i, feat in enumerate(fpn_B)]
        if not self.return_mask:
            return fpn_A, fpn_B
        if self.label_dir is None:
            raise ValueError("dataset_root is required when return_mask=True")
        base_name = os.path.splitext(file_name)[0]
        mask = _load_binary_mask(self.label_dir, base_name, self.crop_size)
        return fpn_A, fpn_B, mask


class OnlineDataset(Dataset):
    """Returns image paths for online FPN extraction during training."""
    def __init__(self, trainset, max_samples=0):
        self.trainset = trainset
        self.ids = trainset.ids[:max_samples] if max_samples > 0 else trainset.ids

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        imgA_path = os.path.join(self.trainset.root, self.trainset.A, img_id)
        imgB_path = os.path.join(self.trainset.root, self.trainset.B, img_id)
        mask_path = os.path.join(self.trainset.root, self.trainset.label, img_id)
        imgA = Image.open(imgA_path).convert('RGB').resize((self.trainset.size, self.trainset.size), Image.BILINEAR)
        imgB = Image.open(imgB_path).convert('RGB').resize((self.trainset.size, self.trainset.size), Image.BILINEAR)
        mask = Image.open(mask_path).convert('L').resize((self.trainset.size, self.trainset.size), Image.NEAREST)
        mask = torch.from_numpy((np.array(mask) > 0).astype(np.float32)).unsqueeze(0)
        return imgA, imgB, mask


def collate_pil(batch):
    """Keep PIL images as a list."""
    imgA, imgB, masks = zip(*batch)
    return list(imgA), list(imgB), torch.stack(masks)


def collate_fpns(batch):
    fpn_A_batch = [item[0] for item in batch]
    fpn_B_batch = [item[1] for item in batch]
    stacked_A = [
        None if fpn_A_batch[0][i] is None else torch.stack([fpn[i] for fpn in fpn_A_batch])
        for i in range(3)
    ]
    stacked_B = [
        None if fpn_B_batch[0][i] is None else torch.stack([fpn[i] for fpn in fpn_B_batch])
        for i in range(3)
    ]
    if len(batch[0]) > 2:
        masks = torch.stack([item[2] for item in batch])
        return stacked_A, stacked_B, masks
    return stacked_A, stacked_B


def _lerp(a, b, t):
    return a * (1 - t) + b * t


def resolve_training_schedule(args, epoch):
    """
    Stage-aware loss weighting for v4.

    v4 is easier to optimize if we first make the gate learn change cues,
    then progressively tighten compatibility regularizers.
    """
    base = {
        "feature_scale": args.lambda_feature,
        "change_scale": args.lambda_change,
        "sparse_scale": args.lambda_sparse,
        "criterion_weights": {},
    }
    if args.module_type != "temporal_gate_v4":
        return base

    total_epochs = max(int(args.epochs), 1)
    warmup_end = max(1, int(round(total_epochs * 0.25)))
    refine_end = max(warmup_end + 1, int(round(total_epochs * 0.70)))

    stage1 = {
        "consist": 0.80,
        "recon": 0.05,
        "tv": 0.02,
        "contrast": 0.25,
        "prior": 0.35,
        "dist": 0.00,
        "decoder": 0.00,
        "mask_prior": 1.50,
        "logit_consist": 0.10,
    }
    stage2 = {
        "consist": 1.00,
        "recon": 0.08,
        "tv": 0.04,
        "contrast": 0.40,
        "prior": 0.80,
        "dist": 0.02,
        "decoder": 0.03,
        "mask_prior": 1.00,
        "logit_consist": 0.10,
    }
    stage3 = {
        "consist": 1.00,
        "recon": 0.10,
        "tv": 0.05,
        "contrast": 0.50,
        "prior": 1.00,
        "dist": 0.05,
        "decoder": 0.05,
        "mask_prior": 0.60,
        "logit_consist": 0.10,
    }

    if epoch < warmup_end:
        return {
            "feature_scale": args.lambda_feature * 0.25,
            "change_scale": args.lambda_change * 1.25,
            "sparse_scale": args.lambda_sparse * 0.50,
            "criterion_weights": stage1,
        }

    if epoch < refine_end:
        t = (epoch - warmup_end) / max(refine_end - warmup_end, 1)
        return {
            "feature_scale": _lerp(args.lambda_feature * 0.25, args.lambda_feature * 0.60, t),
            "change_scale": _lerp(args.lambda_change * 1.25, args.lambda_change * 1.00, t),
            "sparse_scale": _lerp(args.lambda_sparse * 0.50, args.lambda_sparse, t),
            "criterion_weights": {k: _lerp(stage1[k], stage2[k], t) for k in stage1},
        }

    t = (epoch - refine_end) / max(total_epochs - refine_end, 1)
    return {
        "feature_scale": _lerp(args.lambda_feature * 0.60, args.lambda_feature * 0.90, t),
        "change_scale": _lerp(args.lambda_change * 1.00, args.lambda_change * 0.85, t),
        "sparse_scale": _lerp(args.lambda_sparse, args.lambda_sparse * 1.50, t),
        "criterion_weights": {k: _lerp(stage2[k], stage3[k], t) for k in stage2},
    }


def _checkpoint_payload(args, change_module, epoch=None, extra=None):
    payload = {
        'framework_name': 'CogMem-OVCD',
        'module_name': 'STEE',
        'module_full_name': 'Semantic-Temporal Evidence Encoder',
        'change_module': change_module.state_dict(),
        'gamma': change_module.gamma.item(),
        'module_type': args.module_type,
        'lora_rank': args.lora_rank,
        'lora_alpha': args.lora_alpha,
        'lora_dropout': args.lora_dropout,
        'gamma_init': args.gamma_init,
        'head_hidden_dim': args.head_hidden_dim,
        'max_gamma': args.max_gamma,
        'residual_clip_ratio': args.residual_clip_ratio,
        'gain_init': args.gain_init,
        'gain_min': args.gain_min,
        'gain_max': args.gain_max,
        'gate_floor': args.gate_floor,
        'gate_temperature': args.gate_temperature,
        'v4_active_levels': args.v4_active_levels,
        'v4_gate_level': args.v4_gate_level,
        'v4_loss_levels': args.v4_loss_levels,
        'native_logit_training': args.native_logit_training,
        'head_lr_mult': args.head_lr_mult,
        'lambda_mask_prior': args.lambda_mask_prior,
        'lambda_logit_consist': args.lambda_logit_consist,
        'mask_positive_margin': args.mask_positive_margin,
        'mask_negative_margin': args.mask_negative_margin,
        'supervise_at_logit_resolution': args.supervise_at_logit_resolution,
        'amp': args.amp,
        'amp_dtype': args.amp_dtype,
    }
    if epoch is not None:
        payload['epoch'] = int(epoch)
    if extra:
        payload.update(extra)
    return payload


def _make_cached_validation_loader(args, load_fpn_levels):
    if not args.val_cache_dir or not args.val_label_dir:
        return None
    val_dataset = CachedFPNDataset(
        args.val_cache_dir,
        dataset_root=None,
        label_dir=args.val_label_dir,
        crop_size=args.crop_size,
        return_mask=True,
        max_samples=args.val_limit,
        offset=args.val_offset,
        load_levels=load_fpn_levels,
    )
    loader_kwargs = {
        'batch_size': args.val_batch_size,
        'shuffle': False,
        'num_workers': args.val_num_workers,
        'collate_fn': collate_fpns,
        'pin_memory': args.val_num_workers > 0,
    }
    if args.val_num_workers > 0:
        loader_kwargs['persistent_workers'] = True
        loader_kwargs['prefetch_factor'] = args.prefetch_factor
    print(
        f"Validation: cached FPN at {args.val_cache_dir}, labels={args.val_label_dir}, "
        f"samples={len(val_dataset)}, batch={args.val_batch_size}, metric={args.best_metric}",
        flush=True,
    )
    return DataLoader(val_dataset, **loader_kwargs)


@torch.no_grad()
def _evaluate_cached_scp(change_module, args, val_loader, device, fpn_device_dtype, amp_enabled, amp_dtype):
    was_training = change_module.training
    change_module.eval()
    cm = np.zeros((2, 2), dtype=np.int64)
    thresholds = []
    samples = 0
    autocast_ctx = (
        torch.amp.autocast(device_type='cuda', dtype=amp_dtype, enabled=True)
        if amp_enabled and device.type == 'cuda' else nullcontext()
    )
    for batch in val_loader:
        fpn_A = [
            None if f is None else f.to(device, dtype=fpn_device_dtype, non_blocking=True)
            for f in batch[0]
        ]
        fpn_B = [
            None if f is None else f.to(device, dtype=fpn_device_dtype, non_blocking=True)
            for f in batch[1]
        ]
        masks = batch[2].to(device, non_blocking=True)
        with autocast_ctx:
            outputs = change_module(fpn_A, fpn_B, output_size=(args.crop_size, args.crop_size))
        for i in range(masks.shape[0]):
            change_fpn_i = [None if f is None else f[i:i + 1] for f in outputs['change_fpn']]
            fpn_A_i = [None if f is None else f[i:i + 1] for f in fpn_A]
            fpn_B_i = [None if f is None else f[i:i + 1] for f in fpn_B]
            logits = outputs.get('change_logits')
            logits_i = None if logits is None else logits[i:i + 1]
            bcd, _score, threshold = compute_bcd(
                change_fpn_i,
                fpn_A_i,
                fpn_B_i,
                args.crop_size,
                args.crop_size,
                threshold_percentile=args.bcd_threshold_percentile,
                threshold_abs=args.bcd_threshold_abs,
                score_mode=args.bcd_score_mode,
                threshold_mode=args.bcd_threshold_mode,
                threshold_std_factor=args.bcd_threshold_std_factor,
                change_logits=logits_i,
                threshold_edge_weight=args.threshold_edge_weight,
                threshold_edge_percentile=args.threshold_edge_percentile,
                threshold_edge_dilate=args.threshold_edge_dilate,
                threshold_edge_min_pixels=args.threshold_edge_min_pixels,
                threshold_min=args.threshold_min,
                threshold_max=args.threshold_max,
                threshold_scale=args.threshold_scale,
                threshold_bias=args.threshold_bias,
                threshold_max_foreground_ratio=args.threshold_max_foreground_ratio,
                threshold_candidate_percentile=args.threshold_candidate_percentile,
            )
            gt = (masks[i].squeeze(0).detach().cpu().numpy() > 0).astype(np.int64)
            pred = (bcd.detach().cpu().numpy() > 0).astype(np.int64)
            cm += confusion_binary(gt, pred)
            thresholds.append(float(threshold.detach().cpu()))
            samples += 1
    if was_training:
        change_module.train()
    metrics = {k: float(v) for k, v in metrics_from_confusion(cm).items()}
    return {
        'samples': int(samples),
        'confusion': cm.tolist(),
        'metrics': metrics,
        'threshold_mean': float(np.mean(thresholds)) if thresholds else None,
        'threshold_std': float(np.std(thresholds)) if thresholds else None,
    }


def _is_better_metric(current, best, metric, tie_metric):
    if best is None:
        return True
    cur_m = float(current['metrics'].get(metric, -1.0))
    best_m = float(best['metrics'].get(metric, -1.0))
    if cur_m > best_m + 1e-12:
        return True
    if abs(cur_m - best_m) <= 1e-12:
        cur_t = float(current['metrics'].get(tie_metric, -1.0))
        best_t = float(best['metrics'].get(tie_metric, -1.0))
        return cur_t > best_t + 1e-12
    return False


# ──────────────────────────────────────────────
# Phase 2: Train change module
# ──────────────────────────────────────────────

def train(args):
    print("=" * 60)
    print("Phase 2: Training STEE module")
    print("=" * 60)

    device = select_device()
    print(f"Using device: {describe_device(device)}")
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision('high')

    amp_enabled = bool(args.amp and device.type == 'cuda')
    amp_dtype = torch.float16 if args.amp_dtype == 'fp16' else torch.bfloat16
    fpn_device_dtype = amp_dtype if amp_enabled and args.cast_fpn_to_amp else torch.float32
    if amp_enabled:
        print(f"AMP: enabled, dtype={args.amp_dtype}, cast_fpn_to_amp={args.cast_fpn_to_amp}")
    else:
        print("AMP: disabled")

    start_epoch = 0
    ckpt = None
    state_dict = None
    if args.resume and os.path.exists(args.resume):
        print(f"Resuming from {args.resume}")
        ckpt = torch.load(args.resume, weights_only=False)
        state_dict = ckpt['change_module']
        ckpt_module_type = ckpt.get('module_type') or infer_module_type_from_state_dict(state_dict)
        if ckpt_module_type and ckpt_module_type != args.module_type:
            raise ValueError(
                f"Checkpoint module_type={ckpt_module_type}, but requested --module_type {args.module_type}")
        start_epoch = ckpt.get('epoch', 0)
        print(f"  Resumed at epoch {start_epoch}")

    # Change module (trainable)
    change_module = SemanticTemporalEvidenceEncoder(
        dims=[64, 96, 128],
        module_type=args.module_type if ckpt is None else (ckpt.get('module_type') or infer_module_type_from_state_dict(state_dict)),
        **module_kwargs_from_checkpoint(ckpt, args),
    ).to(device)
    if ckpt is not None:
        state_dict = normalize_state_dict_for_stee(state_dict, args.module_type)
        change_module.load_state_dict(state_dict)

    criterion = STEEConsistencyLoss(
        lambda_consist=args.lambda_consist,
        lambda_recon=args.lambda_recon,
        lambda_tv=args.lambda_tv,
        lambda_contrast=args.lambda_contrast,
        lambda_prior=args.lambda_prior,
        lambda_dist=args.lambda_dist,
        lambda_decoder=args.lambda_decoder,
        lambda_mask_prior=args.lambda_mask_prior,
        lambda_logit_consist=args.lambda_logit_consist,
        mask_positive_margin=args.mask_positive_margin,
        mask_negative_margin=args.mask_negative_margin,
        max_residual_ratio=args.max_residual_ratio,
    )
    jitter = FeatureJitter(max_pad=4, flip_prob=0.5)

    if args.module_type == "temporal_gate_v4" and args.head_lr_mult != 1.0:
        fuse_params = []
        other_params = []
        for name, param in change_module.named_parameters():
            if not param.requires_grad:
                continue
            if "change_fuse" in name:
                fuse_params.append(param)
            else:
                other_params.append(param)
        optimizer = torch.optim.AdamW(
            [
                {"params": other_params, "lr": args.lr},
                {"params": fuse_params, "lr": args.lr * args.head_lr_mult},
            ],
            weight_decay=args.weight_decay,
        )
        print(f"Optimizer: v4 change_fuse lr multiplier={args.head_lr_mult}")
    else:
        optimizer = torch.optim.AdamW(change_module.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler('cuda', enabled=amp_enabled and amp_dtype == torch.float16)

    load_fpn_levels = args.load_fpn_levels
    if args.module_type == "temporal_gate_v4" and str(load_fpn_levels).strip().lower() == "active":
        load_fpn_levels = args.v4_active_levels

    # Data: cached vs online
    online = getattr(args, 'no_cache', False)
    if online:
        print("Data mode: ONLINE (extracting FPN features on-the-fly from SAM3 backbone)")
        bpe_path = os.path.join(project_root, "sam3/assets/bpe_simple_vocab_16e6.txt.gz")
        sam3_model = build_sam3_image_model(
            bpe_path=bpe_path,
            checkpoint_path=args.sam3_checkpoint, device=str(device)
        )
        processor = Sam3Processor(sam3_model, confidence_threshold=0.1, device=str(device))
        trainset = OVCDDataset('train', args.dataset_root, "CA-CDD", args.crop_size)
        online_dataset = OnlineDataset(trainset, max_samples=args.max_samples)
        trainloader = DataLoader(
            online_dataset, batch_size=args.batch_size, shuffle=not args.no_shuffle,
            num_workers=0, pin_memory=False, collate_fn=collate_pil
        )
        n_samples = len(online_dataset)
    else:
        print("Data mode: CACHED (reading pre-extracted FPN features from disk)")
        cached_dataset = CachedFPNDataset(
            args.cache_dir,
            dataset_root=args.dataset_root,
            crop_size=args.crop_size,
            return_mask=not args.no_change_supervision,
            max_samples=args.max_samples,
            ram_cache_levels=_parse_ram_cache_levels(args.ram_cache_levels),
            ram_cache_max_gb=args.ram_cache_max_gb,
            include_file=args.include_file,
            exclude_file=args.exclude_file,
            load_levels=load_fpn_levels,
        )
        loader_kwargs = {
            'batch_size': args.batch_size,
            'shuffle': not args.no_shuffle,
            'num_workers': args.num_workers,
            'collate_fn': collate_fpns,
            'pin_memory': args.num_workers > 0,
        }
        if args.num_workers > 0:
            loader_kwargs['persistent_workers'] = True
            loader_kwargs['prefetch_factor'] = args.prefetch_factor
        trainloader = DataLoader(cached_dataset, **loader_kwargs)
        n_samples = len(cached_dataset)
        print(
            f"DataLoader: shuffle={not args.no_shuffle}, workers={args.num_workers}, "
            f"prefetch_factor={args.prefetch_factor if args.num_workers > 0 else 'n/a'}"
        )
        if getattr(cached_dataset, '_use_mmap', False):
            print(f"Cache source: mmap ({cached_dataset.mmap_dtype}) at {cached_dataset.mmap_dir}")
            if cached_dataset._ram_a:
                print(f"RAM cached levels: {sorted(cached_dataset._ram_a.keys())}")
            print(f"Loaded FPN levels: {sorted(cached_dataset.load_levels)}")
        else:
            print(f"Cache source: torch .pt files at {cached_dataset.cache_dir}")
            print(f"Loaded FPN levels: {sorted(cached_dataset.load_levels)}")

    val_loader = _make_cached_validation_loader(args, load_fpn_levels)
    best_validation = None
    best_checkpoint_path = args.best_checkpoint_path or os.path.join(args.save_dir, 'best.pt')

    accum_steps = args.accum_steps
    effective_batch = args.batch_size * accum_steps
    print(f"Samples: {n_samples}, Batch: {args.batch_size}, Accum: {accum_steps}, Effective: {effective_batch}, Epochs: {args.epochs}")
    loss_levels = _parse_fpn_levels(
        args.v4_loss_levels if args.module_type == "temporal_gate_v4" else "all",
        active_levels=args.v4_active_levels,
    )
    if args.module_type == "temporal_gate_v4":
        print(
            f"v4 config: active_levels={_parse_fpn_levels(args.v4_active_levels)}, "
            f"gate_level={args.v4_gate_level}, loss_levels={loss_levels}, "
            f"native_logit_training={args.native_logit_training}"
        )

    for epoch in range(start_epoch, args.epochs):
        epoch_start_time = time.perf_counter()
        change_module.train()
        schedule = resolve_training_schedule(args, epoch)
        epoch_losses = {}
        optimizer.zero_grad()

        pbar = tqdm(
            trainloader,
            desc=f"Epoch {epoch+1}/{args.epochs}",
            mininterval=args.tqdm_mininterval,
            disable=args.disable_tqdm,
        )
        for step, batch in enumerate(pbar):
            if online:
                imgA, imgB, masks = batch
                masks = masks.to(device, non_blocking=True)
                # Extract FPN features with frozen SAM3 backbone
                with torch.no_grad():
                    fpn_A, fpn_B = [], []
                    for i in range(len(imgA)):
                        sA = processor.set_image(imgA[i])
                        sB = processor.set_image(imgB[i])
                        fpn_A.append([f.squeeze(0) for f in sA["backbone_out"]["backbone_fpn"]])
                        fpn_B.append([f.squeeze(0) for f in sB["backbone_out"]["backbone_fpn"]])
                    # Stack to batch
                    fpn_A = [torch.stack([f[i] for f in fpn_A]).to(device, dtype=torch.float32) for i in range(3)]
                    fpn_B = [torch.stack([f[i] for f in fpn_B]).to(device, dtype=torch.float32) for i in range(3)]
            else:
                fpn_A = [
                    None if f is None else f.to(device, dtype=fpn_device_dtype, non_blocking=True)
                    for f in batch[0]
                ]
                fpn_B = [
                    None if f is None else f.to(device, dtype=fpn_device_dtype, non_blocking=True)
                    for f in batch[1]
                ]
                masks = batch[2].to(device, non_blocking=True) if len(batch) > 2 else None

            autocast_ctx = (
                torch.amp.autocast(device_type='cuda', dtype=amp_dtype, enabled=True)
                if amp_enabled else nullcontext()
            )
            with autocast_ctx:
                # Forward (unjittered)
                if args.native_logit_training and args.module_type == "temporal_gate_v4":
                    output_size = None
                elif masks is not None and not args.supervise_at_logit_resolution:
                    output_size = masks.shape[-2:]
                else:
                    output_size = _first_loaded_feature_shape(fpn_A)
                outputs = change_module(fpn_A, fpn_B, output_size=output_size)
                change_fpn = outputs['change_fpn']

                # Forward (jittered)
                fpn_A_jit, fpn_B_jit, jitter_params = jitter.jitter_pair(fpn_A, fpn_B)
                outputs_jit = change_module(fpn_A_jit, fpn_B_jit, output_size=output_size)
                change_fpn_jit = outputs_jit['change_fpn']

                change_fpn_loss = _select_fpn_levels(change_fpn, loss_levels)
                change_fpn_jit_loss = _select_fpn_levels(change_fpn_jit, loss_levels)
                fpn_A_loss = _select_fpn_levels(fpn_A, loss_levels)
                fpn_B_loss = _select_fpn_levels(fpn_B, loss_levels)
                feature_loss, loss_dict = criterion(
                    change_fpn_loss, change_fpn_jit_loss,
                    fpn_A_loss, fpn_B_loss,
                    jitter_params, jitter,
                    masks=masks,
                    change_logits=outputs.get('change_logits'),
                    change_logits_jit=outputs_jit.get('change_logits'),
                    weights=schedule["criterion_weights"],
                )
                total_loss = schedule["feature_scale"] * feature_loss
                loss_dict['feature_total'] = feature_loss

                if not args.no_change_supervision:
                    if outputs.get('change_logits') is None:
                        raise ValueError(
                            "Change supervision requires a module with change_logits; "
                            "use --module_type temporal_lora or temporal_gate_v4"
                        )
                    change_loss, change_loss_dict = change_supervision_loss(
                        outputs['change_logits'], masks,
                        lambda_bce=args.lambda_bce,
                        lambda_dice=args.lambda_dice,
                        resize_target_to_logits=args.supervise_at_logit_resolution,
                    )
                    total_loss = total_loss + schedule["change_scale"] * change_loss
                    for k, v in change_loss_dict.items():
                        loss_dict[k] = v
                    for k, v in change_head_separation_metrics(outputs['change_logits'], masks).items():
                        loss_dict[k] = v
                if outputs.get("gate_map") is not None and masks is not None:
                    for k, v in gate_separation_metrics(outputs["gate_map"], masks).items():
                        loss_dict[k] = v

                if schedule["sparse_scale"] > 0:
                    sparse_loss = 0
                    for delta in outputs.get('deltas', []):
                        sparse_loss = sparse_loss + delta.abs().mean()
                    sparse_loss = sparse_loss / max(len(outputs.get('deltas', [])), 1)
                    total_loss = total_loss + schedule["sparse_scale"] * sparse_loss
                    loss_dict['sparse'] = sparse_loss
                for k, v in decoder_compatibility_metrics(change_fpn_loss, fpn_A_loss, fpn_B_loss).items():
                    loss_dict[k] = v
                loss_dict['total'] = total_loss

            scaled_loss = total_loss / accum_steps
            if scaler.is_enabled():
                scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()

            for k, v in loss_dict.items():
                epoch_losses[k] = epoch_losses.get(k, 0) + v.item()

            if (step + 1) % accum_steps == 0 or (step + 1) == len(trainloader):
                if scaler.is_enabled():
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(change_module.parameters(), max_norm=1.0)
                if scaler.is_enabled():
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()

            if not args.disable_tqdm:
                pbar.set_postfix(loss=total_loss.item())

        scheduler.step()

        n = len(trainloader)
        epoch_elapsed = time.perf_counter() - epoch_start_time
        samples_per_sec = n_samples / max(epoch_elapsed, 1e-6)
        print(f"\nEpoch {epoch+1}/{args.epochs}:")
        for k, v in epoch_losses.items():
            print(f"  {k}: {v / n:.4f}")
        print(f"  gamma: {change_module.gamma.item():.4f}")
        print(f"  lr: {scheduler.get_last_lr()[0]:.6f}")
        print(f"  epoch_time_sec: {epoch_elapsed:.2f}")
        print(f"  train_samples_per_sec: {samples_per_sec:.2f}")

        if val_loader is not None and (epoch + 1) % max(int(args.val_every), 1) == 0:
            val_result = _evaluate_cached_scp(
                change_module,
                args,
                val_loader,
                device,
                fpn_device_dtype,
                amp_enabled,
                amp_dtype,
            )
            val_result['epoch'] = epoch + 1
            val_metrics = val_result['metrics']
            print(
                f"  val_{args.best_metric}: {val_metrics.get(args.best_metric, 0.0):.6f} "
                f"val_{args.best_tie_metric}: {val_metrics.get(args.best_tie_metric, 0.0):.6f} "
                f"val_Recall: {val_metrics.get('Recall', 0.0):.6f}",
                flush=True,
            )
            if _is_better_metric(val_result, best_validation, args.best_metric, args.best_tie_metric):
                best_validation = val_result
                os.makedirs(args.save_dir, exist_ok=True)
                best_payload = _checkpoint_payload(
                    args,
                    change_module,
                    epoch=epoch + 1,
                    extra={
                        'selection_protocol': 'online_cached_source_validation',
                        'selection_metric': args.best_metric,
                        'selection_tie_metric': args.best_tie_metric,
                        'validation': val_result,
                    },
                )
                _save_checkpoint(best_payload, best_checkpoint_path)
                metrics_path = os.path.join(args.save_dir, 'best_metrics.json')
                with open(metrics_path, 'w') as f:
                    json.dump({
                        'best_checkpoint': best_checkpoint_path,
                        'epoch': epoch + 1,
                        'metric': args.best_metric,
                        'tie_metric': args.best_tie_metric,
                        'validation': val_result,
                    }, f, indent=2)
                print(f"  Saved best: {best_checkpoint_path}", flush=True)

        if args.save_every > 0 and (epoch + 1) % args.save_every == 0:
            save_path = os.path.join(args.save_dir, f'checkpoint_epoch{epoch+1}.pt')
            os.makedirs(args.save_dir, exist_ok=True)
            _save_checkpoint(_checkpoint_payload(args, change_module, epoch=epoch + 1), save_path)
            print(f"  Saved: {save_path}")

    final_payload = _checkpoint_payload(args, change_module, epoch=args.epochs)
    final_path = os.path.join(args.save_dir, 'stee_temporal_gate_v4_final.pt')
    _save_checkpoint(final_payload, final_path)
    print(f"\nSTEE final: {final_path}")
    legacy_final_path = os.path.join(args.save_dir, 'final.pt')
    if legacy_final_path != final_path:
        _save_checkpoint(final_payload, legacy_final_path)
        print(f"Legacy final alias: {legacy_final_path}")


def _save_checkpoint(state, path, retries=3):
    for attempt in range(retries):
        try:
            tmp = path + '.tmp'
            torch.save(state, tmp, _use_new_zipfile_serialization=False)
            os.replace(tmp, path)
            return
        except Exception as e:
            if attempt < retries - 1:
                print(f"  Save failed (attempt {attempt+1}): {e}, retrying...")
                import time; time.sleep(2)
            else:
                raise


def main():
    parser = argparse.ArgumentParser(description='STEE Training')
    parser.add_argument('--phase', type=str, default='train',
                        choices=['cache', 'train', 'all'])
    parser.add_argument('--sam3_checkpoint', type=str, default='weights/sam3/sam3.pt')
    parser.add_argument('--dataset_root', type=str, default='./data/')
    parser.add_argument('--cache_dir', type=str, default='weights')
    parser.add_argument('--save_dir', type=str, default='./checkpoints')
    parser.add_argument('--crop_size', type=int, default=512)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--prefetch_factor', type=int, default=1)
    parser.add_argument('--tqdm_mininterval', type=float, default=10.0,
                        help='Minimum seconds between tqdm log refreshes.')
    parser.add_argument('--disable_tqdm', action='store_true',
                        help='Disable per-step tqdm progress output and keep epoch summaries only.')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--save_every', type=int, default=10)
    parser.add_argument('--val_cache_dir', type=str, default='',
                        help='Optional cached FPN directory for source validation best-checkpoint selection.')
    parser.add_argument('--val_label_dir', type=str, default='',
                        help='Optional label directory paired with --val_cache_dir.')
    parser.add_argument('--val_batch_size', type=int, default=32)
    parser.add_argument('--val_num_workers', type=int, default=2)
    parser.add_argument('--val_offset', type=int, default=0)
    parser.add_argument('--val_limit', type=int, default=0)
    parser.add_argument('--val_every', type=int, default=1)
    parser.add_argument('--best_checkpoint_path', type=str, default='',
                        help='Path for the continuously maintained best checkpoint. Defaults to save_dir/best.pt.')
    parser.add_argument('--best_metric', type=str, default='F1')
    parser.add_argument('--best_tie_metric', type=str, default='IoU')
    parser.add_argument('--bcd_score_mode', type=str, default='head')
    parser.add_argument('--bcd_threshold_mode', type=str, default='percentile')
    parser.add_argument('--bcd_threshold_percentile', type=float, default=85.0)
    parser.add_argument('--bcd_threshold_abs', type=float, default=None)
    parser.add_argument('--bcd_threshold_std_factor', type=float, default=1.0)
    parser.add_argument('--threshold_edge_weight', type=float, default=0.5)
    parser.add_argument('--threshold_edge_percentile', type=float, default=80.0)
    parser.add_argument('--threshold_edge_dilate', type=int, default=1)
    parser.add_argument('--threshold_edge_min_pixels', type=int, default=64)
    parser.add_argument('--threshold_min', type=float, default=0.0)
    parser.add_argument('--threshold_max', type=float, default=1.0)
    parser.add_argument('--threshold_scale', type=float, default=1.0)
    parser.add_argument('--threshold_bias', type=float, default=0.0)
    parser.add_argument('--threshold_max_foreground_ratio', type=float, default=0.0)
    parser.add_argument('--threshold_candidate_percentile', type=float, default=80.0)
    parser.add_argument('--accum_steps', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-2)
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--max_samples', type=int, default=0)
    parser.add_argument('--include_file', type=str, default=None,
                        help='Optional text file of cached sample names to include. Accepts .pt or image names.')
    parser.add_argument('--exclude_file', type=str, default=None,
                        help='Optional text file of cached sample names to exclude from cached training.')
    parser.add_argument('--no_shuffle', action='store_true',
                        help='Disable sample shuffling. Faster for huge cached FPN files on spinning/RAID disks.')
    parser.add_argument('--ram_cache_levels', type=str, default='',
                        help='Comma-separated FPN levels to preload into CPU RAM from mmap, e.g. "2", "1,2", or "all".')
    parser.add_argument('--ram_cache_max_gb', type=float, default=0.0,
                        help='Safety limit for --ram_cache_levels. 0 means only MemAvailable safety is enforced.')
    parser.add_argument('--load_fpn_levels', type=str, default='all',
                        help='Cached training levels to load, e.g. "2", "1,2", "all", or v4-only "active".')
    parser.add_argument('--amp', action=argparse.BooleanOptionalAction, default=False,
                        help='Enable CUDA autocast mixed precision during training.')
    parser.add_argument('--amp_dtype', type=str, default='fp16', choices=['fp16', 'bf16'])
    parser.add_argument('--cast_fpn_to_amp', action=argparse.BooleanOptionalAction, default=True,
                        help='Move cached FPN tensors to GPU in AMP dtype when --amp is enabled.')
    parser.add_argument('--no_cache', action='store_true',
                        help='Extract FPN features online from SAM3 backbone instead of reading cached .pt files')
    parser.add_argument('--module_type', type=str, default='temporal_gate_v4',
                        choices=['temporal_lora', 'temporal_gate_v4', 'mdfm'],
                        help='temporal_gate_v4 is the recommended STEE design; temporal_lora keeps v2 compatibility.')
    parser.add_argument('--lora_rank', type=int, default=16)
    parser.add_argument('--lora_alpha', type=float, default=16.0)
    parser.add_argument('--lora_dropout', type=float, default=0.0)
    parser.add_argument('--gamma_init', type=float, default=0.05,
                        help='Initial residual scale. Small values keep frozen SAM3 decoder inputs near original FPN.')
    parser.add_argument('--head_hidden_dim', type=int, default=96)
    parser.add_argument('--max_gamma', type=float, default=0.2,
                        help='Clamp residual injection scale to protect the frozen SAM3 decoder.')
    parser.add_argument('--residual_clip_ratio', type=float, default=0.25,
                        help='Clamp decoder-facing residual RMS to this fraction of frozen FPN RMS.')
    parser.add_argument('--gain_init', type=float, default=1.0,
                        help='Initial gain for v4 spatial gate injection.')
    parser.add_argument('--gain_min', type=float, default=0.75,
                        help='Lower bound for v4 effective gamma.')
    parser.add_argument('--gain_max', type=float, default=1.25,
                        help='Upper bound for v4 effective gamma.')
    parser.add_argument('--gate_floor', type=float, default=0.02,
                        help='Minimum spatial gate value for v4 residual injection.')
    parser.add_argument('--gate_temperature', type=float, default=1.0,
                        help='Temperature used to sharpen v4 spatial gates.')
    parser.add_argument('--v4_active_levels', type=str, default='all',
                        help='FPN levels used by v4 gate/residual branches, e.g. "2", "1,2", or "all".')
    parser.add_argument('--v4_gate_level', type=int, default=0,
                        help='FPN level resolution used by v4 change_fuse before optional output upsampling.')
    parser.add_argument('--v4_loss_levels', type=str, default='active',
                        help='FPN levels used by feature regularization for v4. "active" follows --v4_active_levels.')
    parser.add_argument('--native_logit_training', action=argparse.BooleanOptionalAction, default=False,
                        help='For v4, train change logits at native gate resolution instead of upsampling to level0.')
    parser.add_argument('--head_lr_mult', type=float, default=5.0,
                        help='LR multiplier for v4 change_fuse parameters.')
    parser.add_argument('--lambda_consist', type=float, default=1.0)
    parser.add_argument('--lambda_recon', type=float, default=0.1)
    parser.add_argument('--lambda_tv', type=float, default=0.05)
    parser.add_argument('--lambda_contrast', type=float, default=0.5)
    parser.add_argument('--lambda_prior', type=float, default=1.0,
                        help='Positive anti-collapse alignment to frozen bi-temporal feature difference')
    parser.add_argument('--lambda_dist', type=float, default=0.05,
                        help='FPN distribution matching for SAM3 decoder compatibility')
    parser.add_argument('--lambda_decoder', type=float, default=0.1,
                        help='Residual-ratio penalty for frozen decoder compatibility.')
    parser.add_argument('--lambda_mask_prior', type=float, default=0.5,
                        help='Positive residual-scale prior aligned with GT change masks.')
    parser.add_argument('--lambda_logit_consist', type=float, default=0.1,
                        help='Jitter consistency weight for the explicit change head.')
    parser.add_argument('--mask_positive_margin', type=float, default=0.12,
                        help='Target residual scale inside changed regions.')
    parser.add_argument('--mask_negative_margin', type=float, default=0.02,
                        help='Target residual scale inside unchanged regions.')
    parser.add_argument('--max_residual_ratio', type=float, default=0.25,
                        help='Target upper bound used by --lambda_decoder.')
    parser.add_argument('--lambda_feature', type=float, default=0.2,
                        help='Weight for FeatUp/SimFeatUp-style feature regularization.')
    parser.add_argument('--lambda_change', type=float, default=1.0,
                        help='Weight for supervised category-agnostic BCD loss.')
    parser.add_argument('--lambda_bce', type=float, default=1.0)
    parser.add_argument('--lambda_dice', type=float, default=1.0)
    parser.add_argument('--supervise_at_logit_resolution', action=argparse.BooleanOptionalAction, default=True,
                        help='Resize GT masks to the change-head logit resolution during training.')
    parser.add_argument('--lambda_sparse', type=float, default=1e-4,
                        help='Small L1 penalty on LoRA residuals to protect decoder compatibility.')
    parser.add_argument('--no_change_supervision', action='store_true',
                        help='Disable binary mask supervision; not recommended for OVCD, kept for ablations.')

    args = parser.parse_args()

    if args.phase in ['cache', 'all']:
        cache_features(args)
    if args.phase in ['train', 'all']:
        train(args)


if __name__ == '__main__':
    main()
