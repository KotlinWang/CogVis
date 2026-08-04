"""
STEE Evaluation (Seg2Change Protocol)

All-class-at-once evaluation: one backbone pass per image, then per-class Seg2Change.

Seg2Change protocol:
  pred_class = BCD & (pred_A_class | pred_B_class)
  gt_class   = (GT_T1 == class_id) | (GT_T2 == class_id)

BCD = residual magnitude thresholding from change module output.
"""

import os
import sys
import argparse
from pathlib import Path
import json

import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from PIL import Image
from torch.utils.data import DataLoader

project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

from change_module import (
    SemanticTemporalEvidenceEncoder,
    infer_module_type_from_state_dict,
    module_kwargs_from_checkpoint,
    normalize_state_dict_for_stee,
)
from utils.gpu import select_device, describe_device
from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
from dataset.ovcd import OVCDDataset
from posterior_ovcd import compute_score_threshold


# SECOND class mapping: display_name -> (class_ids_in_seg, gt_class_ids).
# GT ids follow Seg2Change/dataset/ovcd.py:
# background=0, grass=1, ground=2, tree=3, water=4, building=5, playground=6.
# The segmentation ids follow the prompt list below:
# background=0, ground=1, grass=2, sports field=3, car=4,
# tree=5, water=6, cropland=7, building=8.
SECOND_CLASSES = {
    'Building':    ([8],       [5]),
    'Tree':        ([5],       [3]),
    'Water':       ([6],       [4]),
    'Grass':       ([2, 7],    [1]),
    'Ground':      ([1],       [2]),
    'Playground':  ([3],       [6]),
}


def get_cls_idx_from_list(name_list):
    query_words, query_idx = [], []
    for cls_idx, names in enumerate(name_list):
        for name in names.split(','):
            query_words.append(name.strip())
            query_idx.append(cls_idx)
    return query_words, query_idx


def segearth_predict_from_state(processor, state, query_words, query_idx, num_cls):
    """Run SAM3 text decoding from an existing image/backbone state."""
    h = state["original_height"]
    w = state["original_width"]
    num_queries = len(query_words)
    device = torch.device(processor.device)
    seg_logits = torch.zeros((num_queries, h, w), device=device)

    with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
        for q_idx, q_word in enumerate(query_words):
            processor.reset_all_prompts(state)
            state = processor.set_text_prompt(state=state, prompt=q_word)

            # Instance decoder
            if state['masks_logits'].shape[0] > 0:
                for inst_id in range(state['masks_logits'].shape[0]):
                    inst_logits = state['masks_logits'][inst_id].squeeze()
                    inst_score = state['object_score'][inst_id]
                    if inst_logits.shape != (h, w):
                        inst_logits = F.interpolate(
                            inst_logits.view(1, 1, *inst_logits.shape),
                            size=(h, w), mode='bilinear', align_corners=False
                        ).squeeze()
                    seg_logits[q_idx] = torch.max(seg_logits[q_idx], inst_logits * inst_score)

            # Semantic decoder
            sem_logits = state['semantic_mask_logits']
            if sem_logits.shape != (h, w):
                sem_logits = F.interpolate(
                    sem_logits, size=(h, w), mode='bilinear', align_corners=False
                ).squeeze()
            seg_logits[q_idx] = torch.max(seg_logits[q_idx], sem_logits)

            # Presence score
            seg_logits[q_idx] = seg_logits[q_idx] * state["presence_score"]

    # Aggregate synonyms → class-level logits
    if num_cls != num_queries:
        cls_index = F.one_hot(query_idx).T.view(num_cls, num_queries, 1, 1).float().to(seg_logits.device)
        seg_logits = (seg_logits.unsqueeze(0) * cls_index).max(1)[0]

    # Argmax per pixel, threshold low-confidence to background
    seg_pred = torch.argmax(seg_logits, dim=0)
    max_vals = seg_logits.max(0)[0]
    seg_pred[max_vals < 0.1] = 0

    return seg_pred


def segearth_predict_all_classes(processor, image, query_words, query_idx, num_cls):
    """Run SegEarth-OV3 from an image, using its original frozen SAM3 features."""
    state = processor.set_image(image)
    return segearth_predict_from_state(processor, state, query_words, query_idx, num_cls)


def clone_state_with_change_fpn(state, change_fpn):
    """
    Create a SAM3 processor state whose visual FPN is replaced by STEE output.

    This is the critical path for testing whether STEE really produces
    decoder-compatible features instead of only a standalone residual score.
    Positional encodings and metadata stay from the original image state.
    """
    new_state = dict(state)
    backbone_out = dict(state["backbone_out"])
    original_fpn = backbone_out["backbone_fpn"]
    injected_fpn = [
        feat.to(device=ref.device, dtype=ref.dtype)
        for feat, ref in zip(change_fpn, original_fpn)
    ]
    backbone_out["backbone_fpn"] = injected_fpn
    backbone_out["vision_features"] = injected_fpn[-1]
    for key in ("language_features", "language_mask", "language_embeds"):
        backbone_out.pop(key, None)
    new_state["backbone_out"] = backbone_out
    for key in ("geometric_prompt", "boxes", "masks", "masks_logits", "scores",
                "semantic_mask_logits", "presence_score", "object_score"):
        new_state.pop(key, None)
    return new_state


def confusion_binary(gt, pred):
    """2x2 binary confusion matrix via bincount. Layout: [[TN, FP], [FN, TP]]"""
    label = (2 * gt.astype(np.int64) + pred.astype(np.int64)).ravel()
    cm = np.bincount(label, minlength=4).reshape(2, 2)
    return cm


def metrics_from_confusion(cm):
    """Compute IoU, F1, Precision, Recall, OA, Kappa from 2x2 confusion matrix."""
    tn, fp = cm[0, 0], cm[0, 1]
    fn, tp = cm[1, 0], cm[1, 1]
    total = cm.sum()

    iou = tp / (tp + fp + fn + 1e-12)
    precision = tp / (tp + fp + 1e-12)
    recall = tp / (tp + fn + 1e-12)
    f1 = 2 * precision * recall / (precision + recall + 1e-12)
    oa = (tp + tn) / (total + 1e-12)

    # Kappa
    po = oa
    pe = ((tp + fp) * (tp + fn) + (tn + fn) * (fp + tn)) / (total * total + 1e-12)
    kappa = (po - pe) / (1 - pe + 1e-12)

    return {'IoU': iou, 'F1': f1, 'Pre': precision, 'Recall': recall, 'OA': oa, 'Kappa': kappa}


def compute_change_score(change_fpn, fpn_A, fpn_B, H, W, score_mode='fused'):
    """Compute a dense change score from learned residuals and frozen feature difference."""
    residual = change_fpn[-1] - fpn_A[-1]  # [1, 256, h, w]
    residual_mag = residual.norm(dim=1).squeeze(0)  # [h, w]

    cos_diff = 1 - F.cosine_similarity(fpn_A[-1], fpn_B[-1], dim=1).squeeze(0)

    def _normalize(score):
        score = score.float()
        lo = torch.quantile(score, 0.02)
        hi = torch.quantile(score, 0.98)
        return ((score - lo) / (hi - lo + 1e-6)).clamp(0, 1)

    residual_score = _normalize(residual_mag)
    feature_score = _normalize(cos_diff)

    if score_mode == 'residual':
        score_small = residual_score
    elif score_mode == 'feature':
        score_small = feature_score
    elif score_mode == 'fused':
        score_small = 0.6 * residual_score + 0.4 * feature_score
    else:
        raise ValueError(f"Unknown score_mode: {score_mode}")

    return F.interpolate(
        score_small.unsqueeze(0).unsqueeze(0),
        size=(H, W), mode='bilinear', align_corners=False
    ).squeeze(0).squeeze(0)


def compute_head_score(change_logits, H, W):
    """Convert the explicit change head to a dense [H, W] probability map."""
    score = torch.sigmoid(change_logits.float())
    if score.shape[-2:] != (H, W):
        score = F.interpolate(score, size=(H, W), mode='bilinear', align_corners=False)
    return score.squeeze(0).squeeze(0)


def compute_threshold(score, threshold_mode='percentile', threshold_percentile=85,
                      threshold_abs=None, threshold_std_factor=1.0,
                      threshold_edge_weight=0.5, threshold_edge_percentile=80.0,
                      threshold_edge_dilate=1, threshold_edge_min_pixels=64,
                      threshold_min=0.0, threshold_max=1.0,
                      threshold_scale=1.0, threshold_bias=0.0,
                      threshold_max_foreground_ratio=0.0,
                      threshold_candidate_percentile=80.0):
    """Compute an adaptive threshold on a normalized [0, 1] score map."""
    return compute_score_threshold(
        score,
        mode=threshold_mode,
        percentile=threshold_percentile,
        threshold_abs=threshold_abs,
        std_factor=threshold_std_factor,
        edge_weight=threshold_edge_weight,
        edge_percentile=threshold_edge_percentile,
        edge_dilate=threshold_edge_dilate,
        edge_min_pixels=threshold_edge_min_pixels,
        threshold_min=threshold_min,
        threshold_max=threshold_max,
        threshold_scale=threshold_scale,
        threshold_bias=threshold_bias,
        threshold_max_foreground_ratio=threshold_max_foreground_ratio,
        threshold_candidate_percentile=threshold_candidate_percentile,
    )


def compute_bcd(change_fpn, fpn_A, fpn_B, H, W, threshold_percentile=85,
                threshold_abs=None, score_mode='auto', threshold_mode='percentile',
                threshold_std_factor=1.0, change_logits=None,
                threshold_edge_weight=0.5, threshold_edge_percentile=80.0,
                threshold_edge_dilate=1, threshold_edge_min_pixels=64,
                threshold_min=0.0, threshold_max=1.0,
                threshold_scale=1.0, threshold_bias=0.0,
                threshold_max_foreground_ratio=0.0,
                threshold_candidate_percentile=80.0):
    """BCD from a normalized dense change score, upsampled to (H, W)."""
    if score_mode == 'auto':
        score_mode = 'head' if change_logits is not None else 'fused'

    if score_mode == 'head':
        if change_logits is None:
            raise ValueError("--score_mode head requires a checkpoint with change_logits")
        score = compute_head_score(change_logits, H, W)
    elif score_mode == 'hybrid':
        if change_logits is None:
            raise ValueError("--score_mode hybrid requires a checkpoint with change_logits")
        head_score = compute_head_score(change_logits, H, W)
        dense_score = compute_change_score(change_fpn, fpn_A, fpn_B, H, W, score_mode='fused')
        score = 0.5 * head_score + 0.5 * dense_score
    else:
        score = compute_change_score(change_fpn, fpn_A, fpn_B, H, W, score_mode=score_mode)

    threshold = compute_threshold(
        score, threshold_mode=threshold_mode,
        threshold_percentile=threshold_percentile,
        threshold_abs=threshold_abs,
        threshold_std_factor=threshold_std_factor,
        threshold_edge_weight=threshold_edge_weight,
        threshold_edge_percentile=threshold_edge_percentile,
        threshold_edge_dilate=threshold_edge_dilate,
        threshold_edge_min_pixels=threshold_edge_min_pixels,
        threshold_min=threshold_min,
        threshold_max=threshold_max,
        threshold_scale=threshold_scale,
        threshold_bias=threshold_bias,
        threshold_max_foreground_ratio=threshold_max_foreground_ratio,
        threshold_candidate_percentile=threshold_candidate_percentile)
    bcd = (score > threshold).long()
    return bcd, score, threshold


def evaluate(args):
    print("=" * 60)
    print("STEE Evaluation (All-class Seg2Change)")
    print("=" * 60)

    device = select_device()
    print(f"Using device: {describe_device(device)}")

    # Load trained STEE module. Older checkpoints are inferred
    # as MDFM so old eval runs remain reproducible.
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    state_dict = ckpt['change_module']
    module_type = args.module_type
    if module_type == 'auto':
        module_type = ckpt.get('module_type') or infer_module_type_from_state_dict(state_dict)
    change_module = SemanticTemporalEvidenceEncoder(
        dims=[64, 96, 128],
        module_type=module_type,
        **module_kwargs_from_checkpoint(ckpt, args),
    ).to(device)
    state_dict = normalize_state_dict_for_stee(state_dict, module_type)
    change_module.load_state_dict(state_dict)
    change_module.eval()
    print(f"Loaded change module: {module_type}")

    # Build SAM3
    bpe_path = os.path.join(project_root, "sam3/assets/bpe_simple_vocab_16e6.txt.gz")
    model = build_sam3_image_model(
        bpe_path=bpe_path,
        checkpoint_path=args.sam3_checkpoint, device=str(device)
    )
    processor = Sam3Processor(model, confidence_threshold=0.1, device=str(device))

    # All class prompts for SegEarth (all classes at once)
    name_list = ['background', 'bareland,barren,ground', 'grass', 'sports field', 'car',
                 'tree,forest', 'water,river', 'cropland', 'building,roof,house']
    query_words, query_idx_list = get_cls_idx_from_list(name_list)
    query_idx = torch.tensor(query_idx_list, dtype=torch.int64).to(device)
    num_cls = max(query_idx_list) + 1

    # SECOND dataset (all samples, no per-class filter)
    second_root = args.dataset_root
    if not (Path(second_root) / 'T1').is_dir():
        second_root = str(Path(second_root) / 'SECOND')

    # classname is required by OVCDDataset but we load raw GT masks ourselves
    valset = OVCDDataset('test', second_root, "SECOND", args.crop_size, classname='building')
    if args.chunk_file:
        with open(args.chunk_file, 'r') as f:
            keep = {line.strip() for line in f if line.strip()}
        valset.ids = [x for x in valset.ids if x in keep]
    # GT paths are under second_root/test/
    gt_root = os.path.join(second_root, 'test')
    if args.limit > 0:
        valset.ids = valset.ids[:args.limit]
    valloader = DataLoader(valset, batch_size=1, shuffle=False, num_workers=2)
    print(f"SECOND test samples: {len(valset)}")

    # Per-class confusion matrices
    class_names = list(SECOND_CLASSES.keys())
    class_cms = {name: np.zeros((2, 2), dtype=np.int64) for name in class_names}
    bcd_cm = np.zeros((2, 2), dtype=np.int64)
    results = {
        'config': {
            'checkpoint': args.checkpoint,
            'dataset_root': second_root,
            'threshold_percentile': args.threshold_percentile,
            'threshold_abs': args.threshold_abs,
            'threshold_mode': args.threshold_mode,
            'threshold_std_factor': args.threshold_std_factor,
            'score_mode': args.score_mode,
            'module_type': module_type,
            'semantic_source': args.semantic_source,
            'limit': args.limit,
            'chunk_file': args.chunk_file,
            'chunk_id': args.chunk_id,
        },
        'class_confusions': {},
        'bcd_confusion': None,
        'class_metrics': {},
        'average': {},
        'bcd': {},
        'thresholds': [],
    }

    for imgA, imgB, mask, imgA_path, imgB_path, _ in tqdm(valloader, desc="Evaluating"):
        imgA_pil = Image.open(imgA_path[0]).convert('RGB')
        imgB_pil = Image.open(imgB_path[0]).convert('RGB')
        H, W = imgA_pil.size[1], imgA_pil.size[0]

        # 1. Extract FPN features + change module (once per sample)
        state_A = processor.set_image(imgA_pil)
        state_B = processor.set_image(imgB_pil)
        fpn_A = [f.to(device) for f in state_A["backbone_out"]["backbone_fpn"]]
        fpn_B = [f.to(device) for f in state_B["backbone_out"]["backbone_fpn"]]

        with torch.no_grad():
            outputs = change_module(fpn_A, fpn_B, output_size=(H, W))
            change_fpn = outputs['change_fpn']
            change_logits = outputs.get('change_logits')
            bcd, change_score, threshold = compute_bcd(
                change_fpn, fpn_A, fpn_B, H, W,
                args.threshold_percentile, args.threshold_abs, args.score_mode,
                args.threshold_mode, args.threshold_std_factor,
                change_logits=change_logits,
                threshold_edge_weight=args.threshold_edge_weight,
                threshold_edge_percentile=args.threshold_edge_percentile,
                threshold_edge_dilate=args.threshold_edge_dilate,
                threshold_edge_min_pixels=args.threshold_edge_min_pixels,
                threshold_min=args.threshold_min,
                threshold_max=args.threshold_max,
                threshold_scale=args.threshold_scale,
                threshold_bias=args.threshold_bias,
                threshold_max_foreground_ratio=args.threshold_max_foreground_ratio,
                threshold_candidate_percentile=args.threshold_candidate_percentile)
            results['thresholds'].append(float(threshold.detach().cpu()))

        # 2. Semantic predictions. original = Seg2Change-style T1/T2 indexing.
            # change_decoder = decode text prompts from STEE-translated FPN.
        seg_A = seg_B = seg_change = None
        if args.semantic_source in ['original', 'hybrid']:
            seg_A = segearth_predict_from_state(processor, state_A, query_words, query_idx, num_cls)
            seg_B = segearth_predict_from_state(processor, state_B, query_words, query_idx, num_cls)
        if args.semantic_source in ['change_decoder', 'hybrid']:
            change_state = clone_state_with_change_fpn(state_A, change_fpn)
            seg_change = segearth_predict_from_state(processor, change_state, query_words, query_idx, num_cls)

        bcd_np = bcd.cpu().numpy()

        # 3. BCD evaluation against GT_CD
        gt_cd = np.array(Image.open(
            os.path.join(gt_root, 'GT_CD', os.path.basename(imgA_path[0]))
        ).convert('L'))
        gt_binary = (gt_cd > 0).astype(np.int64)
        bcd_cm += confusion_binary(gt_binary, bcd_np)

        # 4. Per-class Seg2Change evaluation
        gt_t1 = np.array(Image.open(
            os.path.join(gt_root, 'GT_T1', os.path.basename(imgA_path[0]))
        ).convert('L'))
        gt_t2 = np.array(Image.open(
            os.path.join(gt_root, 'GT_T2', os.path.basename(imgA_path[0]))
        ).convert('L'))

        seg_A_np = None if seg_A is None else seg_A.cpu().numpy()
        seg_B_np = None if seg_B is None else seg_B.cpu().numpy()
        seg_change_np = None if seg_change is None else seg_change.cpu().numpy()

        for cls_name, (seg_ids, gt_ids) in SECOND_CLASSES.items():
            # Semantic support can come from original T1/T2 masks, STEE
            # decoder masks, or their union depending on semantic_source.
            pred_support = np.zeros((H, W), dtype=bool)
            for sid in seg_ids:
                if seg_A_np is not None:
                    pred_support |= (seg_A_np == sid)
                if seg_B_np is not None:
                    pred_support |= (seg_B_np == sid)
                if seg_change_np is not None:
                    pred_support |= (seg_change_np == sid)

            # OVCD readout: class prediction is change localization gated by
            # the configured open-vocabulary semantic support.
            pred = (bcd_np & pred_support).astype(np.int64)

            # GT: class present in either T1 or T2
            gt = np.zeros((H, W), dtype=bool)
            for gid in gt_ids:
                gt |= (gt_t1 == gid) | (gt_t2 == gid)
            gt = gt.astype(np.int64)

            class_cms[cls_name] += confusion_binary(gt, pred)

    # 5. Print results
    print("\n" + "=" * 70)
    print("SECOND Seg2Change Results")
    print("=" * 70)
    print(f"{'Task':<15} {'IoU':>7} {'F1':>7} {'Pre':>7} {'Recall':>7} {'OA':>7} {'Kappa':>7}")
    print("-" * 70)

    all_metrics = {m: [] for m in ['IoU', 'F1', 'Pre', 'Recall', 'OA', 'Kappa']}

    for cls_name in class_names:
        m = metrics_from_confusion(class_cms[cls_name])
        results['class_confusions'][cls_name] = class_cms[cls_name].tolist()
        results['class_metrics'][cls_name] = {k: float(v) for k, v in m.items()}
        print(f"SECOND-{cls_name:<10} {m['IoU']*100:7.2f} {m['F1']*100:7.2f} "
              f"{m['Pre']*100:7.2f} {m['Recall']*100:7.2f} {m['OA']*100:7.2f} {m['Kappa']:7.4f}")
        for k in all_metrics:
            all_metrics[k].append(m[k])

    # Average
    print("-" * 70)
    avg = {k: np.mean(v) for k, v in all_metrics.items()}
    results['average'] = {k: float(v) for k, v in avg.items()}
    print(f"{'SECOND-ALL':<15} {avg['IoU']*100:7.2f} {avg['F1']*100:7.2f} "
          f"{avg['Pre']*100:7.2f} {avg['Recall']*100:7.2f} {avg['OA']*100:7.2f} {avg['Kappa']:7.4f}")

    # BCD
    print("\n" + "=" * 70)
    print("BCD (Binary Change Detection)")
    print("=" * 70)
    bcd_m = metrics_from_confusion(bcd_cm)
    results['bcd_confusion'] = bcd_cm.tolist()
    results['bcd'] = {k: float(v) for k, v in bcd_m.items()}
    print(f"{'BCD':<15} {bcd_m['IoU']*100:7.2f} {bcd_m['F1']*100:7.2f} "
          f"{bcd_m['Pre']*100:7.2f} {bcd_m['Recall']*100:7.2f} {bcd_m['OA']*100:7.2f} {bcd_m['Kappa']:7.4f}")

    if args.output_json:
        os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)
        with open(args.output_json, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved metrics: {args.output_json}")


def main():
    parser = argparse.ArgumentParser(description='STEE Evaluation')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--sam3_checkpoint', type=str, default='weights/sam3/sam3.pt')
    parser.add_argument('--dataset_root', type=str, default='./data/')
    parser.add_argument('--crop_size', type=int, default=512)
    parser.add_argument('--module_type', type=str, default='auto',
                        choices=['auto', 'temporal_lora', 'temporal_gate_v4', 'mdfm'],
                        help='auto infers adapter type from checkpoint metadata/state_dict.')
    parser.add_argument('--lora_rank', type=int, default=16)
    parser.add_argument('--lora_alpha', type=float, default=16.0)
    parser.add_argument('--lora_dropout', type=float, default=0.0)
    parser.add_argument('--gamma_init', type=float, default=0.05)
    parser.add_argument('--head_hidden_dim', type=int, default=96)
    parser.add_argument('--max_gamma', type=float, default=0.2)
    parser.add_argument('--residual_clip_ratio', type=float, default=0.25)
    parser.add_argument('--gain_init', type=float, default=1.0)
    parser.add_argument('--gain_min', type=float, default=0.75)
    parser.add_argument('--gain_max', type=float, default=1.25)
    parser.add_argument('--gate_floor', type=float, default=0.02)
    parser.add_argument('--gate_temperature', type=float, default=1.0)
    parser.add_argument('--v4_active_levels', type=str, default='all')
    parser.add_argument('--v4_gate_level', type=int, default=0)
    parser.add_argument('--threshold_percentile', type=int, default=85,
                        help='Percentile for adaptive change thresholding')
    parser.add_argument('--threshold_abs', type=float, default=None,
                        help='Absolute threshold on normalized change score. Overrides percentile when set.')
    parser.add_argument('--threshold_mode', type=str, default='percentile',
                        choices=['percentile', 'mean_std', 'otsu', 'otsu_edge', 'candidate_otsu', 'adaptive'],
                        help='Adaptive thresholding rule for the normalized change score.')
    parser.add_argument('--threshold_std_factor', type=float, default=1.0,
                        help='Std multiplier when --threshold_mode mean_std is used.')
    parser.add_argument('--threshold_edge_weight', type=float, default=0.5,
                        help='Edge Otsu blend weight for --threshold_mode otsu_edge/adaptive.')
    parser.add_argument('--threshold_edge_percentile', type=float, default=80.0,
                        help='Gradient percentile used to define score-edge neighborhoods.')
    parser.add_argument('--threshold_edge_dilate', type=int, default=1,
                        help='Dilation radius for score-edge neighborhoods.')
    parser.add_argument('--threshold_edge_min_pixels', type=int, default=64,
                        help='Fallback to global Otsu when edge pixels are fewer than this.')
    parser.add_argument('--threshold_min', type=float, default=0.0,
                        help='Lower clamp for adaptive thresholds.')
    parser.add_argument('--threshold_max', type=float, default=1.0,
                        help='Upper clamp for adaptive thresholds.')
    parser.add_argument('--threshold_scale', type=float, default=1.0,
                        help='Global scale applied to adaptive thresholds.')
    parser.add_argument('--threshold_bias', type=float, default=0.0,
                        help='Global additive bias applied to adaptive thresholds.')
    parser.add_argument('--threshold_max_foreground_ratio', type=float, default=0.0,
                        help='Raise adaptive thresholds to keep predicted foreground below this ratio.')
    parser.add_argument('--threshold_candidate_percentile', type=float, default=80.0,
                        help='Low cutoff percentile for candidate_otsu thresholding.')
    parser.add_argument('--score_mode', type=str, default='auto',
                        choices=['auto', 'head', 'hybrid', 'fused', 'residual', 'feature'],
                        help='Change score source used for BCD. auto uses supervised change head when available.')
    parser.add_argument('--semantic_source', type=str, default='hybrid',
                        choices=['original', 'change_decoder', 'hybrid'],
                        help='Semantic support source: original T1/T2, STEE-decoded FPN, or union.')
    parser.add_argument('--limit', type=int, default=0,
                        help='Limit samples for quick testing (0 = all)')
    parser.add_argument('--chunk_file', type=str, default=None,
                        help='Optional text file containing image names for this eval shard.')
    parser.add_argument('--chunk_id', type=int, default=0,
                        help='Eval shard id used in logs/output naming.')
    parser.add_argument('--output_json', type=str, default=None,
                        help='Optional path to write metrics and confusion matrices as JSON.')
    args = parser.parse_args()
    evaluate(args)


if __name__ == '__main__':
    main()
