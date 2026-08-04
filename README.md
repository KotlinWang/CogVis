# 🌍 CogVis: Must Open-Vocabulary Change Detection Perceive the Scene Anew for Every Query?

![Python](https://img.shields.io/badge/Python-3.10-3776AB?logo=python&logoColor=white) ![PyTorch](https://img.shields.io/badge/PyTorch-2.5.1-EE4C2C?logo=pytorch&logoColor=white) ![CUDA](https://img.shields.io/badge/CUDA-12.1-76B900?logo=nvidia&logoColor=white) ![Benchmarks](https://img.shields.io/badge/Benchmarks-7-2F6BFF) ![Adapters](https://img.shields.io/badge/Trainable_Adapters-3-A23B72)

## 📰 News

* **`Aug. 04, 2026`**: The CogVis repository now provides seven-benchmark inference, staged adapter training, and paper-locked prompt profiles.

## ⭐ Overview

![CogVis overview](./assets/cogvis_framework.png)

CogVis is a cognitive memory-guided framework for Open-Vocabulary Change Detection (OVCD). It reorganizes OVCD into a **perception-memory-verification** process: the Scene Change Perceptron extracts reusable temporal evidence once per image pair, the Semantic Memory Calibrator adapts the decision boundary for each text query, and the Adaptive Region Filter verifies candidate regions. SAM3 and CLIP remain frozen, while only three lightweight adapters are trained.

> **Perceive the scene once. Recall the right boundary. Verify every region.**

## ✨ Highlights

| **7 / 7** | **+28.50%** | **0.230 FPS** | **10.03 s** |
|:---:|:---:|:---:|:---:|
| Best reported result on every evaluated benchmark | Higher throughput than the next-fastest method | 4.82 GiB peak memory | Inference with 10 queried categories |

* **Query-shared perception:** temporal evidence is computed once and reused across categories.
* **Memory-guided calibration:** each image-query pair receives an adaptive threshold.
* **Reliable prediction:** semantic, temporal, and structural evidence jointly filter candidate regions.
* **Cross-task evaluation:** semantic CD, binary CD, and building-damage assessment are covered in one framework.

## 🎯 Model Zoo

Model weights are intentionally excluded from Git. Prepare the following checkpoints and place them under `weights/` before running CogVis:

| Component | Used by | Checkpoint | Status |
|:---|:---|:---|:---:|
| Change Adapter | Scene Change Perceptron | `change_adapter.pt` | Required |
| Score Adapter | Semantic Memory Calibrator | `score_adapter.pt` | Required |
| Gate Adapter | Adaptive Region Filter | `gate_adapter.pt` | Required |
| SAM3 | Frozen visual backbone | `sam3.pt` | Required |
| CLIP ViT-B/16 | Frozen text encoder and retrieval | `ViT-B-16.pt` | Required |

If the checkpoint files are stored elsewhere, set `COGVIS_WEIGHT_DIR=/path/to/weights`.

## 🏆 Main Results

The values below are reported in the accompanying anonymous manuscript.

| Task | Dataset | Metric | Prior best | **CogVis** | Gain |
|:---|:---|:---:|---:|---:|---:|
| Semantic CD | SECOND | mIoU | 31.67 | **32.48** | **+0.81** |
| Semantic CD | SCSCD | mIoU | 23.22 | **25.36** | **+2.14** |
| Binary CD | CLCD | IoU | 31.48 | **33.67** | **+2.19** |
| Binary CD | DSIFN | IoU | 47.55 | **48.50** | **+0.95** |
| Binary CD | LEVIR-CD | IoU | 72.50 | **72.55** | **+0.05** |
| Binary CD | WHU-CD | IoU | 75.72 | **75.86** | **+0.14** |
| Damage assessment | xBD | mIoU | 24.51 | **32.68** | **+8.17** |

![Qualitative comparison across seven benchmarks](./assets/qualitative_results.png)

### ⚡ Efficiency and Query Scaling

On an RTX 3090 with 512 x 512 inputs, CogVis reaches **0.230 FPS** with **4.82 GiB** peak memory. Sharing temporal perception improves throughput by **28.50%** over the next-fastest compared method. At 10 queried categories, CogVis requires **10.03 s**, compared with **12.56-956.60 s** for category-wise baselines.

![Efficiency and query-scaling analysis](./assets/efficiency_scaling.png)

## 🛰️ Dataset Preparation

<details open>
<summary><strong>Expected directory structure</strong></summary>

CogVis evaluates six semantic or binary change-detection datasets and the xBD building-damage benchmark. CA-CDD is used only for adapter optimization and memory construction.

```text
CogVis/
|-- data/
|   |-- OVCD_Benchmark/
|   |   |-- CACDD/
|   |   |-- SECOND/
|   |   |-- SCSCD/
|   |   |-- CLCD-512/
|   |   |-- DSIFN-512/
|   |   |-- LEVIR-CD-512/
|   |   `-- WHU-CD-512/
|   `-- xBD/
|       `-- test/
|           |-- images/
|           |-- labels/
|           `-- targets/
`-- weights/
    |-- change_adapter.pt
    |-- score_adapter.pt
    |-- gate_adapter.pt
    |-- sam3.pt
    `-- ViT-B-16.pt
```

Set custom dataset locations when the data is stored outside the repository:

```bash
export COGVIS_OVCD_BENCHMARK_ROOT=/path/to/OVCD_Benchmark
export COGVIS_XBD_ROOT=/path/to/xBD/test
```

</details>

## 💻 Installation

<details open>
<summary><strong>Create the CogVis environment</strong></summary>

CogVis is tested with Ubuntu, an NVIDIA GPU, CUDA 12.1, and Python 3.10.

```bash
conda create -n cogvis python=3.10 -y
conda activate cogvis
python -m pip install -r requirements.txt
```

Run all commands below from the CogVis repository root.

Validate the inference and training command contracts without initializing models or writing predictions:

```bash
python infer.py --dataset all --dry-run
python train.py all-adapters --dry-run
```

</details>

## 🔍 Inference

### Evaluate all seven benchmarks

This command runs all six semantic or binary datasets and evaluates xBD in seven chunks before merging its metrics:

```bash
python infer.py \
  --dataset all \
  --device 0 \
  --output-dir outputs/inference
```

### Evaluate one semantic dataset

```bash
python infer.py \
  --dataset SECOND \
  --device 0 \
  --save-predictions \
  --output-dir outputs/second
```

Supported names are `SECOND`, `SCSCD`, `CLCD-512`, `DSIFN-512`, `LEVIR-CD-512`, and `WHU-CD-512`.

### Evaluate xBD

```bash
python infer.py \
  --dataset xBD \
  --num-chunks 7 \
  --device 0 \
  --output-dir outputs/xbd
```

| Output | Location |
|:---|:---|
| Per-dataset metrics | `outputs/inference/<dataset>/metrics.json` |
| Merged xBD metrics | `outputs/inference/xBD/metrics.json` |
| Saved predictions | `<output-dir>/<dataset>/predictions/` |

Use `python infer.py --help` for evaluation limits, offsets, chunk selection, and custom dataset roots.

## 🧠 Training

The Change, Score, and Gate Adapters are trained sequentially. The default paper recipe uses **4,968 samples**, **8 shards**, **100 epochs per adapter**, and **seed 2026**.

The Change Adapter requires an FPN cache at `<cache-root>/fpn_cache`. Its labels default to `<benchmark-root>/CACDD/label` unless `--label-dir` is supplied.

```bash
python train.py all-adapters \
  --benchmark-root "$COGVIS_OVCD_BENCHMARK_ROOT" \
  --cache-root /path/to/cache-parent \
  --clip-cache-dir "${COGVIS_WEIGHT_DIR:-$PWD/weights}" \
  --device 0 \
  --output-dir outputs/training/all
```

The resulting checkpoints are:

```text
outputs/training/all/change/final.pt
outputs/training/all/score/score_adapter.pt
outputs/training/all/gate/gate_adapter.pt
```

Evaluate newly trained adapters without replacing the provided checkpoints:

```bash
COGVIS_CHANGE_ADAPTER_CHECKPOINT="$PWD/outputs/training/all/change/final.pt" \
COGVIS_SCORE_ADAPTER_CHECKPOINT="$PWD/outputs/training/all/score/score_adapter.pt" \
COGVIS_GATE_ADAPTER_CHECKPOINT="$PWD/outputs/training/all/gate/gate_adapter.pt" \
python infer.py \
  --dataset all \
  --device 0 \
  --output-dir outputs/trained-inference
```

Individual stages are available as `change-adapter`, `score-adapter`, and `gate-adapter`.

## ⚙️ Configuration

<details>
<summary><strong>Environment-variable overrides</strong></summary>

| Variable | Purpose | Default |
|:---|:---|:---|
| `COGVIS_OVCD_BENCHMARK_ROOT` | CA-CDD and six evaluation datasets | `data/OVCD_Benchmark` |
| `COGVIS_XBD_ROOT` | xBD test split | `data/xBD/test` |
| `COGVIS_WEIGHT_DIR` | Adapter and foundation-model weights | `weights` |
| `COGVIS_RESULTS_DIR` | Inference and training outputs | `outputs` |
| `COGVIS_PROMPTS` | Paper-locked prompt profiles | `prompts.json` |
| `COGVIS_CHANGE_ADAPTER_CHECKPOINT` | Change Adapter override | `weights/change_adapter.pt` |
| `COGVIS_SCORE_ADAPTER_CHECKPOINT` | Score Adapter override | `weights/score_adapter.pt` |
| `COGVIS_GATE_ADAPTER_CHECKPOINT` | Gate Adapter override | `weights/gate_adapter.pt` |
| `COGVIS_SAM3_CHECKPOINT` | SAM3 checkpoint override | `weights/sam3.pt` |
| `COGVIS_CLIP_CACHE_DIR` | CLIP checkpoint directory | `weights` |

</details>

## 🗂️ Project Structure

<details>
<summary><strong>Source tree</strong></summary>

```text
CogVis/
|-- cogvis/
|   |-- adapters/       # Change, Score, and Gate adapters
|   |-- perception/     # Scene Change Perceptron
|   |-- memory/         # Semantic Memory Calibrator
|   |-- verification/   # Adaptive Region Filter
|   `-- runtime/        # Paper-locked runtime
|-- assets/             # README figures
|-- infer.py            # Seven-benchmark inference entry point
|-- train.py            # Staged adapter training
|-- prompts.json        # Fixed prompt profiles
|-- requirements.txt
`-- weights/
```

</details>

## 🤝 Acknowledgement

CogVis builds on frozen SAM3 visual features and CLIP ViT-B/16 text embeddings. We thank the authors and maintainers of the foundation models, change-detection benchmarks, and open-source libraries that make this reproduction possible.

## 📄 Citation

This repository accompanies an anonymous submission. Citation metadata will be added after publication.
