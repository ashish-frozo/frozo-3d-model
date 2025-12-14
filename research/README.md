# ⚠️ SAM-3D Research Lab - INTERNAL USE ONLY ⚠️

This directory contains integration with Meta's SAM-3D Objects for **research benchmarking only**.

## License Warning

> **SAM-3D Objects is licensed under the [SAM License](https://github.com/facebookresearch/sam-3d-objects/blob/main/LICENSE) (Meta).**
>
> - ✅ Use for internal research and benchmarking
> - ✅ Generate quality metrics and failure analysis
> - ❌ **DO NOT** use SAM-3D outputs in production
> - ❌ **DO NOT** include SAM-3D code/weights in shipped product
> - ❌ **DO NOT** train production models on SAM-3D outputs

## Purpose

1. **Establish quality upper-bound**: See what state-of-the-art can achieve
2. **Identify failure modes**: Document where SAM-3D struggles  
3. **Extract metrics**: Define acceptance criteria for SV-SCN
4. **Compare**: Measure SV-SCN performance relative to SAM-3D

## Hardware Requirements

- **GPU**: NVIDIA with ≥32GB VRAM (A100, H100, or equivalent)
- **OS**: Linux 64-bit
- **Storage**: ~50GB for checkpoints

## Directory Structure

```
research/
├── README.md              # This file
├── setup_sam3d.sh         # Environment and checkpoint setup
├── benchmark.py           # Run SAM-3D on test images
├── compare.py             # Compare SV-SCN vs SAM-3D
├── metrics/               # Output metrics JSONs
└── samples/               # Test images for benchmarking
```

## Quick Start

```bash
# 1. Setup environment (requires HuggingFace login)
./setup_sam3d.sh

# 2. Run benchmark on furniture images
python benchmark.py --input_dir samples/ --output_dir metrics/

# 3. Compare with SV-SCN (after training)
python compare.py --svscn_checkpoint ../checkpoints/sv_scn_v0.1.0.pt
```

## Output Format

Benchmark results are saved as JSON:

```json
{
  "image_id": "chair_001",
  "sam3d_metrics": {
    "inference_time_ms": 10234,
    "num_vertices": 45678,
    "num_faces": 91234
  },
  "quality_notes": "Clean reconstruction, minor back artifacts"
}
```

## Integration with Production Pipeline

**Allowed data flow:**
```
SAM-3D → Metrics JSON → Acceptance criteria → SV-SCN training targets
```

**NOT allowed:**
```
SAM-3D → 3D outputs → Production pipeline  ❌
SAM-3D → Model weights → SV-SCN            ❌
```
