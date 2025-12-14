# SV-SCN: Single-View Shape Completion Network

A license-clean ML pipeline for reconstructing complete 3D furniture models from partial point clouds.

## Overview

SV-SCN takes a partial 3D point cloud (derived from single-view depth estimation) and predicts the complete 3D shape. This is designed for the **Photo → 3D Asset Platform** targeting furniture sellers.

### Key Features

- **License-clean**: All production code uses MIT/Apache-2.0/BSD licensed components
- **Class-conditioned**: Learns shape priors for chairs, stools, and tables
- **Confidence scoring**: Detects unreliable predictions with fallback handling
- **AR-ready exports**: GLB, OBJ, and PLY formats

## Project Structure

```
frozo-3d-model/
├── svscn/                      # Main ML package
│   ├── config.py              # Hyperparameters
│   ├── data/                  # Dataset pipeline
│   │   ├── shapenet.py       # ShapeNet loader
│   │   ├── download.py       # Objaverse downloader
│   │   ├── preprocess.py     # Mesh → point cloud
│   │   ├── augment.py        # Partial view generation
│   │   └── dataset.py        # PyTorch Dataset
│   ├── models/               # Neural networks
│   │   ├── encoder.py        # PointNet encoder
│   │   ├── decoder.py        # FoldingNet decoder
│   │   ├── svscn.py          # Full model
│   │   └── losses.py         # Chamfer Distance
│   ├── training/             # Training loop
│   │   └── trainer.py
│   └── inference/            # Inference pipeline
│       ├── predictor.py      # Prediction + confidence
│       ├── mesh_utils.py     # Point cloud → mesh
│       └── export.py         # 3D format exports
├── research/                  # SAM-3D benchmarking (internal only)
├── scripts/                   # CLI tools
│   ├── train.py
│   └── infer.py
├── notebooks/
│   └── train_colab.ipynb
├── requirements.txt
└── README.md
```

## Quick Start

### 1. Installation

```bash
# Clone repository
git clone <repo-url>
cd frozo-3d-model

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Prepare Data

```bash
# Option A: Create placeholder data for testing
python -m svscn.data.shapenet --placeholder --output_dir data/shapenet

# Option B: Use ShapeNet (requires registration at shapenet.org)
export SHAPENET_PATH=/path/to/ShapeNetCore.v2
python -m svscn.data.shapenet --output_dir data/shapenet

# Process meshes to point clouds
python -m svscn.data.preprocess --input_dir data/shapenet --output_dir data/processed

# Generate training pairs
python -m svscn.data.augment --input_dir data/processed --output_dir data/training
```

### 3. Train Model

```bash
# Local training
python scripts/train.py \
    --data_dir data/training \
    --epochs 150 \
    --batch_size 32

# Or use Colab notebook
# Open notebooks/train_colab.ipynb in Google Colab
```

### 4. Run Inference

```bash
# Point cloud output
python scripts/infer.py \
    --checkpoint checkpoints/best.pt \
    --input sample_partial.npy \
    --output result.npy \
    --class_id 0  # 0=chair, 1=stool, 2=table

# Mesh output (for AR)
python scripts/infer.py \
    --checkpoint checkpoints/best.pt \
    --input sample_partial.npy \
    --output result.glb \
    --export_mesh
```

## Model Architecture

```
Partial Point Cloud (2048 points) + Class ID
    ↓
PointNet Encoder → Global Feature (512-dim)
    ↓
Class Embedding (64-dim) → Concatenate
    ↓
FoldingNet Decoder (2D grid folding)
    ↓
Complete Point Cloud (8192 points)
```

### Training Configuration (per ML Training Spec)

| Parameter | Value |
|-----------|-------|
| Input points | 2048 |
| Output points | 8192 |
| Optimizer | Adam |
| Learning rate | 1e-3 → 1e-4 (cosine) |
| Batch size | 32 |
| Epochs | 150 |
| Loss | Chamfer Distance |

## Dataset

Target mix: **75% Objaverse + 25% ShapeNet**

| Stage | Samples |
|-------|---------|
| Baseline | 5,000 |
| Stable v1 | 10,000 |

### Strict Filtering

All meshes undergo validation:
- ✓ License check (CC-BY/CC-0 only for Objaverse)
- ✓ Mesh integrity (no holes, self-intersections)
- ✓ Reasonable vertex count (100 - 200k)
- ✓ Aspect ratio check

## Research Lab (Internal Only)

The `research/` directory contains SAM-3D benchmarking tools for quality comparison.

> ⚠️ **SAM-3D is licensed under SAM License (Meta).**  
> Research outputs are for internal metrics only.  
> **Do NOT use SAM-3D outputs in production.**

```bash
# Setup (requires 32GB+ GPU)
cd research
./setup_sam3d.sh

# Run benchmark
python benchmark.py --input_dir samples/

# Compare with SV-SCN
python compare.py --svscn_checkpoint ../checkpoints/best.pt
```

## Acceptance Criteria

A model is accepted if:
- ✅ No visible holes in AR at 1m distance
- ✅ Footprint error ≤ ±20%
- ✅ Backside geometry visually plausible
- ✅ Export succeeds 100%

## License

- **Production code (svscn/)**: MIT License
- **Research code (research/)**: For internal use only (references SAM License materials)

## References

- SAM 3D Paper: [Meta AI Research](https://ai.meta.com/research/publications/sam-3d-3dfy-anything-in-images/)
- PointNet: [arXiv:1612.00593](https://arxiv.org/abs/1612.00593)
- FoldingNet: [arXiv:1712.07262](https://arxiv.org/abs/1712.07262)
- ShapeNet: [shapenet.org](https://shapenet.org/)
- Objaverse: [objaverse.allenai.org](https://objaverse.allenai.org/)
