from __future__ import annotations

import argparse
import glob
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

import eval_ovcd_benchmark as bench
from train_clip_memory_threshold import (
    CMCT_CLASS_CONDITION_NAMES,
    CMCT_COMPONENT_RELIABILITY_NAMES,
    CMCT_ORACLE_METRIC_NAMES,
    CMCT_SEMANTIC_RELATION_NAMES,
    CMCT_STAT_NAMES,
    CorrectionHead,
    add_cmct_args,
    add_common_args,
    cmct_correction_feature_dim,
    cmct_correction_features,
    rows_to_arrays,
    semantic_relation_features_for_class,
)


def _merge_memory_stats(stats_list: list[dict[str, Any]], shard_paths: list[str]) -> dict[str, Any]:
    if not stats_list:
        return {}
    merged = deepcopy(stats_list[0])
    total_rows = int(sum(int(s.get("num_rows", 0)) for s in stats_list))
    total_images = int(sum(int(s.get("num_images", 0)) for s in stats_list))
    merged["num_images"] = total_images
    merged["num_rows"] = total_rows
    if total_rows > 0:
        merged["tau_mean"] = float(
            sum(float(s.get("tau_mean", 0.0)) * int(s.get("num_rows", 0)) for s in stats_list) / total_rows
        )
        merged["tau_min"] = float(min(float(s.get("tau_min", 0.0)) for s in stats_list if int(s.get("num_rows", 0)) > 0))
        merged["tau_max"] = float(max(float(s.get("tau_max", 0.0)) for s in stats_list if int(s.get("num_rows", 0)) > 0))
    merged["merged_shards"] = len(stats_list)
    merged["train_shards"] = shard_paths
    return merged


@torch.no_grad()
def _batched_leave_one_out_retrieval(
    memory_keys_np: np.ndarray,
    memory_taus_np: np.ndarray,
    *,
    topk: int,
    temperature: float,
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    memory_keys = torch.tensor(memory_keys_np, dtype=torch.float32, device=device)
    memory_taus = torch.tensor(memory_taus_np, dtype=torch.float32, device=device)
    n_rows = int(memory_taus.shape[0])
    if n_rows <= 1:
        tau = torch.full((n_rows,), float(memory_taus.mean().detach().cpu()) if n_rows else 0.5, device=device)
        zero = torch.zeros((n_rows,), dtype=torch.float32, device=device)
        return tau.cpu().numpy(), zero.cpu().numpy(), zero.cpu().numpy()

    k = min(max(int(topk), 1), n_rows - 1)
    temp = max(float(temperature), 1e-4)
    tau_mem = torch.empty((n_rows,), dtype=torch.float32, device=device)
    top1_sim = torch.empty((n_rows,), dtype=torch.float32, device=device)
    mean_sim = torch.empty((n_rows,), dtype=torch.float32, device=device)
    batch_size = max(int(batch_size), 1)

    for start in tqdm(range(0, n_rows, batch_size), desc="batched-LOO-retrieval", dynamic_ncols=True):
        end = min(start + batch_size, n_rows)
        sims = memory_keys[start:end] @ memory_keys.T
        diag_rows = torch.arange(end - start, device=device)
        diag_cols = torch.arange(start, end, device=device)
        sims[diag_rows, diag_cols] = -torch.inf
        selected_sims, selected_idx = torch.topk(sims, k=k, dim=1, largest=True, sorted=True)
        weights = torch.softmax(selected_sims / temp, dim=1)
        selected_taus = memory_taus[selected_idx]
        tau_mem[start:end] = (weights * selected_taus).sum(dim=1)
        top1_sim[start:end] = selected_sims[:, 0]
        mean_sim[start:end] = selected_sims.mean(dim=1)

    return tau_mem.cpu().numpy(), top1_sim.cpu().numpy(), mean_sim.cpu().numpy()


def _train_correction_head_batched(args, bank: dict[str, Any], device: torch.device):
    memory_keys = np.asarray(bank["memory_keys"], dtype=np.float32)
    memory_taus = np.asarray(bank["memory_taus"], dtype=np.float32)
    memory_stats = np.asarray(bank["memory_stats"], dtype=np.float32)
    memory_component_features = np.asarray(
        bank.get("memory_component_features", np.zeros((memory_taus.shape[0], 0), dtype=np.float32)),
        dtype=np.float32,
    )
    memory_meta = list(bank.get("memory_meta", []))
    input_dim = cmct_correction_feature_dim(args)
    model = CorrectionHead(
        input_dim,
        hidden_dim=args.clip_corrector_hidden_dim,
        dropout=args.clip_corrector_dropout,
        max_delta=args.clip_corrector_max_delta,
    ).to(device)
    if memory_taus.size <= 1:
        return model, {"num_rows": int(memory_taus.size), "trained": False}

    retrieval_batch_size = int(getattr(args, "cmct_retrieval_batch_size", 256))
    tau_mem, top1_sim, mean_sim = _batched_leave_one_out_retrieval(
        memory_keys,
        memory_taus,
        topk=args.clip_memory_topk,
        temperature=args.clip_memory_temperature,
        device=device,
        batch_size=retrieval_batch_size,
    )

    x_rows = []
    y_rows = []
    for idx in tqdm(range(memory_taus.shape[0]), desc="build-corrector-features", dynamic_ncols=True):
        cls_name = ""
        if idx < len(memory_meta):
            cls_name = str(memory_meta[idx].get("class_name", ""))
        diag = {
            "top1_sim": float(top1_sim[idx]),
            "mean_sim": float(mean_sim[idx]),
            "topk": int(min(max(int(args.clip_memory_topk), 1), memory_taus.shape[0] - 1)),
        }
        x_rows.append(cmct_correction_features(
            args,
            bank,
            memory_stats[idx],
            float(tau_mem[idx]),
            diag,
            cls_name=cls_name,
            exclude_index=idx,
            component_features=memory_component_features[idx] if memory_component_features.shape[1] > 0 else None,
            semantic_features=semantic_relation_features_for_class(args, bank, cls_name),
        ))
        y_rows.append(float(memory_taus[idx] - tau_mem[idx]))

    x = torch.tensor(np.stack(x_rows, axis=0), dtype=torch.float32, device=device)
    y = torch.tensor(np.asarray(y_rows, dtype=np.float32), dtype=torch.float32, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.clip_corrector_lr, weight_decay=args.clip_corrector_weight_decay)
    for _ in tqdm(range(int(args.clip_corrector_epochs)), desc="train-correction-head", dynamic_ncols=True):
        model.train()
        perm = torch.randperm(x.shape[0], device=device)
        for start in range(0, x.shape[0], int(args.clip_corrector_batch_size)):
            idx = perm[start:start + int(args.clip_corrector_batch_size)]
            pred = model(x[idx])
            loss = F.smooth_l1_loss(pred, y[idx])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        pred = model(x)
        loss = F.smooth_l1_loss(pred, y)
    return model, {
        "num_rows": int(memory_taus.size),
        "trained": True,
        "target_delta_mean": float(y.mean().detach().cpu()),
        "target_delta_min": float(y.min().detach().cpu()),
        "target_delta_max": float(y.max().detach().cpu()),
        "pred_delta_mean": float(pred.mean().detach().cpu()),
        "smooth_l1": float(loss.detach().cpu()),
        "class_conditioned": bool(getattr(args, "cmct_enable_class_residual", False)),
        "input_dim": int(input_dim),
        "retrieval_batch_size": int(retrieval_batch_size),
        "clip_corrector_batch_size": int(args.clip_corrector_batch_size),
        "clip_corrector_epochs": int(args.clip_corrector_epochs),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a CMCT correction checkpoint from pre-collected memory shards.")
    add_common_args(parser)
    add_cmct_args(parser)
    parser.add_argument("--shards_glob", type=str, required=True)
    parser.add_argument("--cmct_retrieval_batch_size", type=int, default=256)
    args = parser.parse_args()
    if not args.output_checkpoint:
        raise SystemExit("--output_checkpoint is required")

    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))

    shard_paths = sorted(glob.glob(args.shards_glob))
    if not shard_paths:
        raise SystemExit(f"No CMCT memory shards matched: {args.shards_glob}")
    chunks = [torch.load(path, map_location="cpu", weights_only=False) for path in shard_paths]
    rows = []
    for chunk in chunks:
        rows.extend(chunk["rows"])
    print(f"Loaded CMCT memory rows: {len(rows)} from {len(shard_paths)} shards")

    bank = rows_to_arrays(rows)
    device = bench.select_device()
    print(f"Using device: {bench.describe_device(device)}")
    model, corrector_stats = _train_correction_head_batched(args, bank, device)
    memory_stats = _merge_memory_stats([chunk.get("clip_memory_stats", {}) for chunk in chunks], shard_paths)
    memory_stats["source_val_same_as_train"] = True
    memory_stats["source_train_limit_effective"] = int(memory_stats.get("num_images", 0))
    memory_stats["source_val_limit_effective"] = int(memory_stats.get("num_images", 0))

    output = Path(args.output_checkpoint)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "memory_keys": bank["memory_keys"],
        "memory_taus": bank["memory_taus"],
        "memory_ratios": bank["memory_ratios"],
        "memory_stats": bank["memory_stats"],
        "memory_oracle_metrics": bank["memory_oracle_metrics"],
        "memory_component_features": bank["memory_component_features"],
        "source_class_names": bank["source_class_names"],
        "source_class_text_embeddings": bank["source_class_text_embeddings"],
        "memory_meta": bank["memory_meta"],
        "correction_head": model.state_dict(),
        "config": vars(args),
        "clip_memory_stats": memory_stats,
        "clip_corrector_stats": corrector_stats,
        "stat_names": CMCT_STAT_NAMES,
        "class_condition_names": CMCT_CLASS_CONDITION_NAMES if bool(args.cmct_enable_class_residual) else [],
        "component_reliability_names": CMCT_COMPONENT_RELIABILITY_NAMES if bool(args.cmct_enable_component_reliability) else [],
        "semantic_relation_names": CMCT_SEMANTIC_RELATION_NAMES if bool(args.cmct_enable_semantic_relation) else [],
        "oracle_metric_names": CMCT_ORACLE_METRIC_NAMES,
    }, output)
    print(f"Saved CMCT checkpoint: {output}")
    print(json.dumps({
        "checkpoint": str(output),
        "clip_memory_stats": memory_stats,
        "clip_corrector_stats": corrector_stats,
    }, indent=2))


if __name__ == "__main__":
    main()
