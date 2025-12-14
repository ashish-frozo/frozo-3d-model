"""
SAM-3D Benchmark Pipeline

Runs SAM-3D Objects on test furniture images to establish quality upper-bound.
Outputs metrics JSON for comparison with SV-SCN.

⚠️  FOR INTERNAL RESEARCH USE ONLY - NOT FOR PRODUCTION

Usage:
    python benchmark.py --input_dir samples/ --output_dir metrics/
"""

import os
import sys
import json
import time
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime

import numpy as np

# Lazy imports for SAM-3D (may not be available)
SAM3D_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Result from a single SAM-3D inference."""
    
    image_id: str
    image_path: str
    timestamp: str
    
    # Timing
    inference_time_ms: float
    
    # Output stats
    num_points: Optional[int] = None
    num_vertices: Optional[int] = None
    num_faces: Optional[int] = None
    
    # Quality indicators
    has_output: bool = False
    error_message: Optional[str] = None
    
    # Bounding box (for scale reference)
    bbox_min: Optional[List[float]] = None
    bbox_max: Optional[List[float]] = None
    
    # Notes
    quality_notes: str = ""


@dataclass
class BenchmarkSummary:
    """Summary of all benchmark runs."""
    
    total_images: int
    successful: int
    failed: int
    
    # Timing stats
    mean_inference_ms: float
    median_inference_ms: float
    min_inference_ms: float
    max_inference_ms: float
    
    # Output stats
    mean_vertices: float
    mean_faces: float
    
    # Failure analysis
    failure_reasons: Dict[str, int]
    
    # Metadata
    benchmark_date: str
    sam3d_version: str = "unknown"


def check_sam3d_available() -> bool:
    """Check if SAM-3D is properly installed."""
    global SAM3D_AVAILABLE
    
    try:
        # Check if sam-3d-objects is in path
        sam3d_path = Path(__file__).parent / "sam-3d-objects"
        if sam3d_path.exists():
            sys.path.insert(0, str(sam3d_path))
            sys.path.insert(0, str(sam3d_path / "notebook"))
        
        from inference import Inference, load_image, load_single_mask
        SAM3D_AVAILABLE = True
        return True
    except ImportError as e:
        logger.warning(f"SAM-3D not available: {e}")
        logger.warning("Run setup_sam3d.sh first to install SAM-3D Objects.")
        return False


def create_mock_mask(image_shape: tuple) -> np.ndarray:
    """
    Create a simple center-focused mask for testing.
    In production, you'd use SAM or manual masks.
    """
    h, w = image_shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    
    # Create elliptical mask in center (70% of image)
    center_y, center_x = h // 2, w // 2
    radius_y, radius_x = int(h * 0.35), int(w * 0.35)
    
    y, x = np.ogrid[:h, :w]
    ellipse = ((x - center_x) / radius_x) ** 2 + ((y - center_y) / radius_y) ** 2 <= 1
    mask[ellipse] = 255
    
    return mask


def run_sam3d_inference(
    image_path: Path,
    mask: Optional[np.ndarray] = None,
    inference_engine: Any = None
) -> BenchmarkResult:
    """
    Run SAM-3D inference on a single image.
    
    Args:
        image_path: Path to input image
        mask: Optional binary mask (will auto-generate if None)
        inference_engine: SAM-3D Inference object
    
    Returns:
        BenchmarkResult with metrics
    """
    image_id = image_path.stem
    timestamp = datetime.now().isoformat()
    
    result = BenchmarkResult(
        image_id=image_id,
        image_path=str(image_path),
        timestamp=timestamp,
        inference_time_ms=0.0
    )
    
    if not SAM3D_AVAILABLE or inference_engine is None:
        result.error_message = "SAM-3D not available"
        return result
    
    try:
        from inference import load_image
        
        # Load image
        image = load_image(str(image_path))
        
        # Generate mask if not provided
        if mask is None:
            mask = create_mock_mask(image.shape)
        
        # Run inference
        start_time = time.perf_counter()
        output = inference_engine(image, mask, seed=42)
        end_time = time.perf_counter()
        
        result.inference_time_ms = (end_time - start_time) * 1000
        result.has_output = True
        
        # Extract metrics from output
        if "gs" in output:
            gs = output["gs"]
            # Gaussian splat metrics
            if hasattr(gs, "means"):
                result.num_points = len(gs.means)
                points = gs.means.cpu().numpy()
                result.bbox_min = points.min(axis=0).tolist()
                result.bbox_max = points.max(axis=0).tolist()
        
        if "mesh" in output:
            mesh = output["mesh"]
            if hasattr(mesh, "vertices"):
                result.num_vertices = len(mesh.vertices)
            if hasattr(mesh, "faces"):
                result.num_faces = len(mesh.faces)
        
        result.quality_notes = "Inference completed successfully"
        
    except Exception as e:
        result.error_message = str(e)
        logger.error(f"Inference failed for {image_id}: {e}")
    
    return result


def run_benchmark(
    input_dir: Path,
    output_dir: Path,
    config_path: Optional[Path] = None,
    max_images: Optional[int] = None
) -> BenchmarkSummary:
    """
    Run benchmark on all images in input directory.
    
    Args:
        input_dir: Directory containing test images
        output_dir: Directory to save results
        config_path: Path to SAM-3D config (uses default if None)
        max_images: Limit number of images (for testing)
    
    Returns:
        BenchmarkSummary with aggregate metrics
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all images
    image_extensions = {".jpg", ".jpeg", ".png", ".webp"}
    images = [
        f for f in input_dir.iterdir()
        if f.suffix.lower() in image_extensions
    ]
    
    if max_images:
        images = images[:max_images]
    
    if not images:
        logger.warning(f"No images found in {input_dir}")
        return None
    
    logger.info(f"Found {len(images)} images to benchmark")
    
    # Initialize SAM-3D if available
    inference_engine = None
    sam3d_version = "not_installed"
    
    if check_sam3d_available():
        try:
            from inference import Inference
            
            if config_path is None:
                config_path = Path(__file__).parent / "sam-3d-objects/checkpoints/hf/pipeline.yaml"
            
            if config_path.exists():
                logger.info(f"Loading SAM-3D from {config_path}")
                inference_engine = Inference(str(config_path), compile=False)
                sam3d_version = "hf_checkpoint"
            else:
                logger.warning(f"Config not found: {config_path}")
                logger.warning("Run download_checkpoints.sh first")
        except Exception as e:
            logger.error(f"Failed to load SAM-3D: {e}")
    
    # Run benchmark
    results: List[BenchmarkResult] = []
    
    for i, image_path in enumerate(images):
        logger.info(f"Processing [{i+1}/{len(images)}]: {image_path.name}")
        result = run_sam3d_inference(image_path, inference_engine=inference_engine)
        results.append(result)
        
        # Save individual result
        result_path = output_dir / f"{result.image_id}.json"
        with open(result_path, "w") as f:
            json.dump(asdict(result), f, indent=2)
    
    # Compute summary
    successful = [r for r in results if r.has_output]
    failed = [r for r in results if not r.has_output]
    
    inference_times = [r.inference_time_ms for r in successful if r.inference_time_ms > 0]
    vertices = [r.num_vertices for r in successful if r.num_vertices]
    faces = [r.num_faces for r in successful if r.num_faces]
    
    # Count failure reasons
    failure_reasons: Dict[str, int] = {}
    for r in failed:
        reason = r.error_message or "unknown"
        failure_reasons[reason] = failure_reasons.get(reason, 0) + 1
    
    summary = BenchmarkSummary(
        total_images=len(results),
        successful=len(successful),
        failed=len(failed),
        mean_inference_ms=np.mean(inference_times) if inference_times else 0,
        median_inference_ms=np.median(inference_times) if inference_times else 0,
        min_inference_ms=min(inference_times) if inference_times else 0,
        max_inference_ms=max(inference_times) if inference_times else 0,
        mean_vertices=np.mean(vertices) if vertices else 0,
        mean_faces=np.mean(faces) if faces else 0,
        failure_reasons=failure_reasons,
        benchmark_date=datetime.now().isoformat(),
        sam3d_version=sam3d_version
    )
    
    # Save summary
    summary_path = output_dir / "benchmark_summary.json"
    with open(summary_path, "w") as f:
        json.dump(asdict(summary), f, indent=2)
    
    logger.info(f"Benchmark complete. Results saved to {output_dir}")
    logger.info(f"  Successful: {summary.successful}/{summary.total_images}")
    logger.info(f"  Mean inference time: {summary.mean_inference_ms:.1f}ms")
    
    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Run SAM-3D benchmark on furniture images"
    )
    parser.add_argument(
        "--input_dir",
        type=Path,
        default=Path("samples"),
        help="Directory containing test images"
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("metrics"),
        help="Directory to save benchmark results"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to SAM-3D pipeline.yaml config"
    )
    parser.add_argument(
        "--max_images",
        type=int,
        default=None,
        help="Limit number of images to process"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s"
    )
    
    # Print warning
    print("\n" + "=" * 60)
    print("⚠️  SAM-3D BENCHMARK - INTERNAL RESEARCH ONLY")
    print("   Do not use outputs in production!")
    print("=" * 60 + "\n")
    
    # Run benchmark
    summary = run_benchmark(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        config_path=args.config,
        max_images=args.max_images
    )
    
    if summary:
        print("\nBenchmark Summary:")
        print(f"  Total images: {summary.total_images}")
        print(f"  Successful: {summary.successful}")
        print(f"  Failed: {summary.failed}")
        if summary.mean_inference_ms > 0:
            print(f"  Mean inference: {summary.mean_inference_ms:.1f}ms")


if __name__ == "__main__":
    main()
