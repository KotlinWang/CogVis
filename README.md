# CogVis

CogVis is the paper-aligned implementation of the perception-memory-verification pipeline for open-vocabulary change detection (OVCD) and xBD damage assessment. This directory follows the compact layout of the validated 2026-07-11 package: five top-level files and two required directories contain the complete source, the seven-dataset prompts, all three adapter training paths, inference, and the locked weights.

The shipped inference configuration is the paper configuration with the large-region protection threshold **`eta_a = 0.04` locked explicitly**. `cogvis/commands.py` passes `--component_gate_protect_area_ratio 0.04` on the semantic, xBD, and xBD-instance routes, and the runtime parser defaults are also `0.04`, so the released code cannot silently fall back to the historical `0.02` value.

## Layout

```text
CogVis/
|-- cogvis/
|   |-- perception/     SCP and the Change Adapter (A_chg)
|   |-- memory/         SMC and the Score Adapter (A_score)
|   |-- verification/   ARF and the Gate Adapter (A_gate)
|   `-- runtime/        frozen training/inference implementation
|-- weights/            Change/Score/Gate, SAM3, and CLIP weights
|-- prompts.json        prompts for all seven datasets
|-- train.py            sequential adapter training
|-- infer.py            seven-dataset inference
|-- README.md
`-- requirements.txt
```

Generated training and inference outputs are written to `outputs/` only after a command is run.

The canonical public source follows the three decision stages in the paper:

| Paper term | Source |
|---|---|
| Scene Change Perceptron (SCP) | `cogvis/perception/scene_change_perceptron.py` |
| Change Adapter (A_chg) | `cogvis/perception/change_adapter.py` |
| Semantic Memory Calibrator (SMC) | `cogvis/memory/semantic_memory_calibrator.py` |
| Score Adapter (A_score) | `cogvis/memory/score_adapter.py` |
| Adaptive Region Filter (ARF) | `cogvis/verification/adaptive_region_filter.py` |
| Gate Adapter (A_gate) | `cogvis/verification/gate_adapter.py` |

`cogvis/modules/` and `cogvis/adapters/` remain as import-only compatibility aliases and contain no duplicate implementation or alternative experimental branch. `cogvis/runtime/` retains the minimum frozen implementation required by the paper-named facades.

## Locked configuration

| Symbol | Argument | Value |
|---|---|---:|
| `K_mem` | `--clip_memory_topk` | 16 |
| `T_m` | `--clip_memory_temperature` | 0.07 |
| `tau_r` | `--component_gate_keep_threshold` | 0.60 |
| `eta_a` | `--component_gate_protect_area_ratio` | **0.04** |
| adaptive floor | `--cmct_adaptive_floor_scale` | 0.75 |
| `lambda_abs` | `--memory_delta_weight` | 0.55 |
| `lambda_dir` | `--memory_transition_weight` | 0.25 |
| `epsilon_g` | `--memory_bcd_floor` | 0.10 |
| `lambda_s` | `--memory_stable_suppress` | 0.45 |
| presence weight | `--memory_presence_delta_weight` | 0.15 |
| memory bank | CA-CDD source images | 100% (4,968) |
| stability seeds | `--seed` | 2026-2030 |

The implemented response is

```text
q_c   = Q_c * [Q_c / (Q_c + C_c + 1e-6)]^2
D_c   = 0.55 * |q_c^2 - q_c^1| + 0.25 * max(q_c^2 * (1 - q_c^1), q_c^1 * (1 - q_c^2))
G_c   = 0.10 + 0.90 * B
S_c   = clip(1 - 0.45 * min(q_c^1, q_c^2) * (1 - B), 0, 1)
p_c   = 1 + 0.15 * |pi_c^2 - pi_c^1|
R_c   = clip(D_c * G_c * S_c * p_c, 0, 1)
```

`eta_a = 0.04` means that a candidate connected region covering at least 4% of the image and containing correlated change pixels is retained without the Gate probability test. Set it to `0.00` only for the protection ablation; see below.

## Environment

The verified server environment is Ubuntu, Python 3.10.20, CUDA 12.1, PyTorch 2.5.1+cu121, and an NVIDIA RTX 3090.

```bash
conda create -n cogvis python=3.10 -y
conda activate cogvis
python -m pip install -r requirements.txt
```

The default dataset paths point to the server 107 benchmark layout. On another machine, pass `--benchmark-root` and `--xbd-root` explicitly, or set `COGVIS_OVCD_BENCHMARK_ROOT` and `COGVIS_XBD_ROOT`.

The expected OVCD benchmark root contains `CACDD`, `SECOND`, `SCSCD`, `CLCD-512`, `DSIFN-512`, `LEVIR-CD-512`, and `WHU-CD-512`. The xBD argument points to its `test` directory containing `images`, `labels`, and `targets`.

## Weights

All five weight files are required in `weights/` (about 4.1 GB total). The local release package ships them, but the GitHub repository does not track them: ordinary Git blobs cannot carry the large SAM3 and CLIP files. Place the files in `weights/` from the validated weight store, Git LFS, or the release assets. The validated hashes are:

| File | SHA-256 |
|---|---|
| `change_adapter.pt` | `a3fb0e6d59943b9b66f407f84e22ce6b9875bbe567cef75cfec604d05b5f9844` |
| `score_adapter.pt` | `bf65af28ef4d1d51bd67f822f64a8547a15cb89efaeb90fb37f01519540589c8` |
| `gate_adapter.pt` | `a713dc5e14ffa46a0ecf1612e3c58e9a7d72ef0a2bfeccb1abccbb9197700fa5` |
| `sam3.pt` | `9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e` |
| `ViT-B-16.pt` | `5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f` |

The repository's `.gitignore` excludes `weights/`. For publication, attach these files to a GitHub Release or store them with Git LFS, and preserve the filenames and hashes above; ordinary GitHub blobs cannot carry the large SAM3 and CLIP files.

## Inference

Run one of the seven datasets:

```bash
python infer.py --dataset SECOND --device 0
python infer.py --dataset SCSCD --device 0
python infer.py --dataset CLCD-512 --device 0
python infer.py --dataset DSIFN-512 --device 0
python infer.py --dataset LEVIR-CD-512 --device 0
python infer.py --dataset WHU-CD-512 --device 0
python infer.py --dataset xBD --num-chunks 7 --device 0
```

Run all seven and merge the seven xBD chunks:

```bash
python infer.py --dataset all --device 0
```

For a fast contract check that does not initialize the models or write outputs:

```bash
python infer.py --dataset all --dry-run
```

Use `--save-predictions` for semantic prediction PNGs. xBD always writes its raw prediction PNG because the retained evaluator uses those files as part of its output contract. A bare runtime evaluator call now defaults to the complete evaluation split (`--eval_limit 0`); pass a positive limit only for smoke tests.

### Optional native xBD instance export

The pixel five-state xBD mIoU remains the paper metric. The optional instance route reuses exactly the same frozen model and threshold flags and exports native instance evidence:

```bash
python infer.py --dataset xBD-instances \
  --manifest /path/to/materialized_inputs.jsonl \
  --device 0
```

Each manifest line provides `pre_rgb`, `post_rgb`, and `valid_mask` `.npy` paths. The route writes per-instance evidence and does not change the pixel metric.

## Training

The three paper adapters are trained through one entry point in the paper's required order: Change, Score, then Gate. The defaults lock the 4,968-image CA-CDD recipe, eight 621-image shards, 100 epochs, and seed 2026.

```bash
python train.py change-adapter --device 0
python train.py score-adapter --device 0
python train.py gate-adapter --device 0
python train.py all-adapters --device 0
```

The former short commands `change`, `score`, `gate`, and `all` remain accepted aliases. Output directory names and checkpoint filenames remain unchanged to preserve prior run compatibility.

`change-adapter` optimizes the SCP Change Adapter and automatically constructs a read-only cache view containing only `/data/kotlin/weights/fpn_cache`. This is intentional: the newer FP16 `fpn_mmap` cache is not part of the locked best-checkpoint lineage. The Change recipe uses `temporal_gate_v4`, active/gate level 2, BF16 AMP, batch 16, learning rate `5e-4`, and the paper loss weights.

`score-adapter` constructs the SMC memory from eight shards and fits the Score Adapter with batch 128. `gate-adapter` collects ARF candidate-region shards and fits the Gate Adapter with batch 4096, the threshold grid `0.20:0.80:0.05`, and the locked `eta_a = 0.04`. Existing shards can be reused:

```bash
python train.py score-adapter --shards-glob '/path/to/score_shards/*.pt' --reuse-shards --device 0
python train.py gate-adapter --shards-glob '/path/to/gate_shards/*.pt' --reuse-shards --device 0
```

Inspect the complete commands without creating files:

```bash
python train.py all-adapters --dry-run
```

Fresh outputs are placed under `outputs/training/`. To evaluate newly trained adapters without replacing the locked weights:

```bash
COGVIS_CHANGE_ADAPTER_CHECKPOINT=/path/to/change/final.pt \
COGVIS_SCORE_ADAPTER_CHECKPOINT=/path/to/score/score_adapter.pt \
COGVIS_GATE_ADAPTER_CHECKPOINT=/path/to/gate/gate_adapter.pt \
python infer.py --dataset DSIFN-512 --device 0
```

## Reproduction status

The shipped checkpoints are the validated 2026-07-11 weights and reproduce the retained inference path. The compact entry point produces the full seven-dataset command set; the only intentional change from the 2026-07-11 command contract is the explicit `eta_a = 0.04` protection flag. A fresh full DSIFN-512 run on server 107 produced IoU `0.4852350242608666`, and fresh DSIFN/xBD canaries matched their reference prediction hashes.

The manuscript Table 1 values come from the original single-checkpoint run:

| Dataset | Metric | Value |
|---|---:|---:|
| SECOND | mIoU | 32.48 |
| SCSCD | mIoU | 25.36 |
| CLCD-512 | IoU | 33.67 |
| DSIFN-512 | IoU | 48.50 |
| LEVIR-CD-512 | IoU | 72.55 |
| WHU-CD-512 | IoU | 75.86 |
| xBD | mIoU | 32.68 |

The 2026-10-28 full-data stability series retrained all three adapters for five independent seeds (2026-2030) under the locked `keep = 0.60 / protect = 0.04 / floor = 0.75` protocol and evaluated every target test set with each of the same five models:

| Dataset | Metric | Test pairs/seed | Mean +/- sample SD |
|---|---:|---:|---:|
| SECOND | mIoU | 1,694 | 32.36 +/- 0.14 |
| SCSCD | mIoU | 322 | 26.38 +/- 0.35 |
| CLCD-512 | IoU | 120 | 34.70 +/- 1.12 |
| DSIFN-512 | IoU | 48 | 50.36 +/- 0.38 |
| LEVIR-CD-512 | IoU | 512 | 73.87 +/- 0.24 |
| WHU-CD-512 | IoU | 660 | 78.23 +/- 0.52 |
| xBD | mIoU | 933 | 32.62 +/- 0.05 |

The standard deviations are sample standard deviations (`ddof=1`, percentage points). This is a new locked-protocol repetition series, not a pure seed replay of the original submission checkpoint: the original SCP metadata is incomplete and the historical Gate record used a different configuration. It therefore quantifies training repeatability and should not be used to reinterpret the original table's single-seed margins as statistically significant.

### Large-region protection ablation

The `eta_a = 0.04` value is kept as the global default. Disabling the exemption (`eta_a = 0.00`) on the same five checkpoints and full test sets gives:

| Dataset | `eta_a = 0.04` | `eta_a = 0.00` | Delta |
|---|---:|---:|---:|
| SECOND | 32.36 | 32.42 | +0.05 |
| SCSCD | 26.38 | 26.78 | +0.41 |
| CLCD-512 | 34.70 | 34.77 | +0.08 |
| DSIFN-512 | 50.36 | 49.87 | -0.48 |
| LEVIR-CD-512 | 73.87 | 73.87 | 0.00 |
| WHU-CD-512 | 78.23 | 78.27 | +0.04 |
| xBD | 32.62 | 32.62 | 0.00 |

The two large effects cancel (SCSCD improves by 0.41, DSIFN drops by 0.48); the unweighted mean change is +0.01 percentage points. The exemption is therefore retained as a fixed design choice rather than tuned per dataset on the test sets.

To reproduce the `eta_a = 0.00` row without editing files, use the Python facade:

```python
from pathlib import Path
from cogvis.pipeline import CogVis

runner = CogVis()
runner.run_semantic(
    dataset="SCSCD",
    output_json=Path("outputs/protect0/SCSCD/metrics.json"),
    extra_args=["--component_gate_protect_area_ratio", "0.00"],
)
```

The training code is executable and seeds all three frozen runtime trainers through one deterministic wrapper. Exact bitwise recovery of the historical best weights cannot be claimed: the original Score shards were deleted, the original Gate shards were replaced by a statistically matching recollection, and the retained memory trainer evolved after the final checkpoint was produced. New full training must therefore be evaluated before claiming that it reaches the historical table values.

Release verification (2026-11-03): the source compiles; `python infer.py --dataset all --dry-run` expands to six semantic commands, seven xBD chunk commands, and one merge, with `protect = 0.04` on every gated route; the 56 targeted contract tests pass with the pixel-route golden fixture updated for the explicit flag; and all five weight hashes match `weights/SHA256SUMS` in the local release package.
