"""
SV-SCN vs SAM-3D Comparison Tool

Compares SV-SCN outputs against SAM-3D baseline to measure relative quality.
Helps identify where SV-SCN needs improvement.

⚠️  FOR INTERNAL RESEARCH USE ONLY - NOT FOR PRODUCTION

Usage:
    python compare.py --svscn_checkpoint ../checkpoints/sv_scn_v0.1.0.pt
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict, field
from datetime import datetime

import numpy as np

# Add parent directory for svscn imports
sys.path.insert(0, str(Path(__file__).parent.parent))

logger = logging.getLogger(__name__)


@dataclass
class ComparisonMetrics:
    """Metrics comparing SV-SCN to SAM-3D for a single sample."""
    
    image_id: str
    
    # SV-SCN metrics
    svscn_inference_ms: float = 0.0
    svscn_num_points: int = 0
    svscn_chamfer_self: float = 0.0  # Reconstruction error
    svscn_confidence: float = 0.0
    
    # SAM-3D baseline metrics (from benchmark)
    sam3d_inference_ms: float = 0.0
    sam3d_num_vertices: int = 0
    
    # Comparison metrics
    speedup_ratio: float = 0.0  # SAM-3D time / SV-SCN time
    point_ratio: float = 0.0   # SV-SCN points / SAM-3D vertices
    
    # Quality assessment
    visual_quality_notes: str = ""
    failure_mode: Optional[str] = None


@dataclass
class ComparisonSummary:
    """Aggregate comparison across all samples."""
    
    total_samples: int
    svscn_successful: int
    sam3d_successful: int
    both_successful: int
    
    # Speed comparison
    mean_speedup: float
    median_speedup: float
    
    # Quality buckets
    svscn_acceptable: int  # Meets AR quality bar
    svscn_needs_improvement: int
    svscn_failed: int
    
    # Common failure modes
    failure_mode_counts: Dict[str, int] = field(default_factory=dict)
    
    # Recommendations
    priority_improvements: List[str] = field(default_factory=list)
    
    # Metadata
    comparison_date: str = ""
    svscn_version: str = ""
    sam3d_version: str = ""


def load_sam3d_metrics(metrics_dir: Path) -> Dict[str, Dict]:
    """Load SAM-3D benchmark results."""
    results = {}
    
    for f in metrics_dir.glob("*.json"):
        if f.name == "benchmark_summary.json":
            continue
        
        with open(f) as fp:
            data = json.load(fp)
            results[data["image_id"]] = data
    
    return results


def compute_chamfer_distance(
    points1: np.ndarray,
    points2: np.ndarray
) -> float:
    """
    Compute Chamfer Distance between two point clouds.
    
    Args:
        points1: (N, 3) array
        points2: (M, 3) array
    
    Returns:
        Chamfer distance (lower is better)
    """
    from scipy.spatial import cKDTree
    
    # Build KD-trees
    tree1 = cKDTree(points1)
    tree2 = cKDTree(points2)
    
    # Find nearest neighbors
    dist1, _ = tree2.query(points1, k=1)  # For each point in 1, find nearest in 2
    dist2, _ = tree1.query(points2, k=1)  # For each point in 2, find nearest in 1
    
    # Chamfer = mean of squared distances in both directions
    chamfer = np.mean(dist1 ** 2) + np.mean(dist2 ** 2)
    
    return chamfer


def load_svscn_model(checkpoint_path: Path):
    """Load trained SV-SCN model."""
    try:
        import torch
        from svscn.models import SVSCN
        from svscn.config import default_config
        
        model = SVSCN(num_classes=default_config.model.NUM_CLASSES)
        
        if checkpoint_path.exists():
            state_dict = torch.load(checkpoint_path, map_location="cpu")
            if "model_state_dict" in state_dict:
                model.load_state_dict(state_dict["model_state_dict"])
            else:
                model.load_state_dict(state_dict)
            logger.info(f"Loaded SV-SCN from {checkpoint_path}")
        else:
            logger.warning(f"Checkpoint not found: {checkpoint_path}")
            logger.warning("Using randomly initialized model")
        
        model.eval()
        return model
    
    except ImportError as e:
        logger.error(f"Failed to import SV-SCN: {e}")
        return None


def run_svscn_inference(
    model,
    partial_points: np.ndarray,
    class_id: int
) -> Tuple[np.ndarray, float, float]:
    """
    Run SV-SCN inference.
    
    Returns:
        (completed_points, inference_time_ms, confidence)
    """
    import torch
    import time
    
    device = next(model.parameters()).device
    
    # Prepare input
    partial = torch.from_numpy(partial_points).float().unsqueeze(0).to(device)
    cls = torch.tensor([class_id]).to(device)
    
    # Run inference
    start = time.perf_counter()
    with torch.no_grad():
        output = model(partial, cls)
    end = time.perf_counter()
    
    # Compute confidence (based on reconstruction error)
    completed = output.cpu().numpy()[0]
    
    # Simple confidence: inverse of self-reconstruction error
    # (In production, use the trained confidence estimator)
    chamfer = compute_chamfer_distance(completed[:2048], partial_points)
    confidence = max(0, 1.0 - chamfer * 10)  # Scale factor TBD
    
    inference_ms = (end - start) * 1000
    
    return completed, inference_ms, confidence


def assess_quality(
    svscn_points: np.ndarray,
    partial_points: np.ndarray,
    confidence: float
) -> Tuple[str, Optional[str]]:
    """
    Assess quality of SV-SCN output.
    
    Returns:
        (quality_notes, failure_mode or None)
    """
    failure_mode = None
    notes = []
    
    # Check point count
    if len(svscn_points) < 8000:
        notes.append("Low point count")
        failure_mode = "incomplete_output"
    
    # Check confidence
    if confidence < 0.6:
        notes.append(f"Low confidence ({confidence:.2f})")
        failure_mode = failure_mode or "low_confidence"
    
    # Check bounding box (should be roughly unit cube)
    bbox_min = svscn_points.min(axis=0)
    bbox_max = svscn_points.max(axis=0)
    bbox_size = bbox_max - bbox_min
    
    if np.any(bbox_size > 2.0):
        notes.append("Oversized output")
        failure_mode = failure_mode or "scale_error"
    
    if np.any(bbox_size < 0.1):
        notes.append("Undersized output")
        failure_mode = failure_mode or "scale_error"
    
    # Check for degenerate geometry (all points too close)
    if np.std(svscn_points) < 0.05:
        notes.append("Collapsed geometry")
        failure_mode = failure_mode or "degenerate"
    
    if not notes:
        notes.append("Acceptable quality")
    
    return "; ".join(notes), failure_mode


def run_comparison(
    svscn_checkpoint: Path,
    sam3d_metrics_dir: Path,
    test_data_dir: Path,
    output_dir: Path
) -> ComparisonSummary:
    """
    Run full comparison between SV-SCN and SAM-3D.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load SAM-3D baseline metrics
    sam3d_metrics = load_sam3d_metrics(sam3d_metrics_dir)
    logger.info(f"Loaded {len(sam3d_metrics)} SAM-3D results")
    
    if not sam3d_metrics:
        logger.warning("No SAM-3D metrics found. Run benchmark.py first.")
    
    # Load SV-SCN model
    model = load_svscn_model(svscn_checkpoint)
    
    if model is None:
        logger.error("Failed to load SV-SCN model")
        return None
    
    # Find test samples (partial point clouds)
    test_files = list(test_data_dir.glob("*_partial.npy"))
    logger.info(f"Found {len(test_files)} test samples")
    
    if not test_files:
        logger.warning(f"No test samples found in {test_data_dir}")
        logger.info("Creating synthetic test samples...")
        # Create dummy samples for testing
        test_files = create_dummy_test_samples(test_data_dir)
    
    # Run comparison
    results: List[ComparisonMetrics] = []
    
    for test_file in test_files:
        image_id = test_file.stem.replace("_partial", "")
        logger.info(f"Comparing: {image_id}")
        
        # Load partial point cloud
        partial_points = np.load(test_file)
        
        # Load class ID (default to chair=0)
        class_file = test_file.parent / f"{image_id}_class.txt"
        class_id = 0
        if class_file.exists():
            class_id = int(class_file.read_text().strip())
        
        # Run SV-SCN
        completed, svscn_ms, confidence = run_svscn_inference(
            model, partial_points, class_id
        )
        
        # Compute self-reconstruction error
        chamfer = compute_chamfer_distance(completed[:2048], partial_points)
        
        # Assess quality
        quality_notes, failure_mode = assess_quality(
            completed, partial_points, confidence
        )
        
        # Get SAM-3D baseline
        sam3d = sam3d_metrics.get(image_id, {})
        sam3d_ms = sam3d.get("inference_time_ms", 0)
        sam3d_verts = sam3d.get("num_vertices", 0)
        
        # Compute comparison metrics
        speedup = sam3d_ms / svscn_ms if svscn_ms > 0 else 0
        point_ratio = len(completed) / sam3d_verts if sam3d_verts > 0 else 0
        
        result = ComparisonMetrics(
            image_id=image_id,
            svscn_inference_ms=svscn_ms,
            svscn_num_points=len(completed),
            svscn_chamfer_self=chamfer,
            svscn_confidence=confidence,
            sam3d_inference_ms=sam3d_ms,
            sam3d_num_vertices=sam3d_verts,
            speedup_ratio=speedup,
            point_ratio=point_ratio,
            visual_quality_notes=quality_notes,
            failure_mode=failure_mode
        )
        
        results.append(result)
        
        # Save individual comparison
        result_path = output_dir / f"{image_id}_comparison.json"
        with open(result_path, "w") as f:
            json.dump(asdict(result), f, indent=2)
    
    # Compute summary
    svscn_success = [r for r in results if r.failure_mode is None]
    sam3d_success = [r for r in results if r.sam3d_inference_ms > 0]
    both_success = [r for r in svscn_success if r.sam3d_inference_ms > 0]
    
    speedups = [r.speedup_ratio for r in both_success if r.speedup_ratio > 0]
    
    # Categorize by quality
    acceptable = [r for r in results if r.failure_mode is None and r.svscn_confidence >= 0.6]
    needs_improvement = [r for r in results if r.failure_mode is None and r.svscn_confidence < 0.6]
    failed = [r for r in results if r.failure_mode is not None]
    
    # Count failure modes
    failure_counts: Dict[str, int] = {}
    for r in failed:
        mode = r.failure_mode or "unknown"
        failure_counts[mode] = failure_counts.get(mode, 0) + 1
    
    # Generate recommendations
    recommendations = []
    if failure_counts.get("low_confidence", 0) > len(results) * 0.2:
        recommendations.append("Increase training data diversity")
    if failure_counts.get("scale_error", 0) > 0:
        recommendations.append("Review normalization pipeline")
    if failure_counts.get("degenerate", 0) > 0:
        recommendations.append("Check decoder architecture")
    if len(acceptable) < len(results) * 0.5:
        recommendations.append("Model needs more training epochs")
    
    # Get versions
    svscn_version = svscn_checkpoint.stem if svscn_checkpoint.exists() else "not_found"
    
    summary = ComparisonSummary(
        total_samples=len(results),
        svscn_successful=len(svscn_success),
        sam3d_successful=len(sam3d_success),
        both_successful=len(both_success),
        mean_speedup=np.mean(speedups) if speedups else 0,
        median_speedup=np.median(speedups) if speedups else 0,
        svscn_acceptable=len(acceptable),
        svscn_needs_improvement=len(needs_improvement),
        svscn_failed=len(failed),
        failure_mode_counts=failure_counts,
        priority_improvements=recommendations,
        comparison_date=datetime.now().isoformat(),
        svscn_version=svscn_version,
        sam3d_version="hf"
    )
    
    # Save summary
    summary_path = output_dir / "comparison_summary.json"
    with open(summary_path, "w") as f:
        json.dump(asdict(summary), f, indent=2)
    
    # Print report
    print_comparison_report(summary)
    
    return summary


def create_dummy_test_samples(test_dir: Path, num_samples: int = 5) -> List[Path]:
    """Create dummy test samples for pipeline validation."""
    test_dir = Path(test_dir)
    test_dir.mkdir(parents=True, exist_ok=True)
    
    files = []
    for i in range(num_samples):
        # Create random partial point cloud (simulating chair-like shape)
        points = np.random.randn(2048, 3) * 0.3
        points[:, 1] += 0.5  # Shift up
        
        filename = test_dir / f"sample_{i:03d}_partial.npy"
        np.save(filename, points.astype(np.float32))
        files.append(filename)
        
        # Save class (chair = 0)
        class_file = test_dir / f"sample_{i:03d}_class.txt"
        class_file.write_text("0")
    
    logger.info(f"Created {num_samples} dummy test samples in {test_dir}")
    return files


def print_comparison_report(summary: ComparisonSummary):
    """Print formatted comparison report."""
    print("\n" + "=" * 60)
    print("  SV-SCN vs SAM-3D Comparison Report")
    print("=" * 60)
    print(f"\nDate: {summary.comparison_date}")
    print(f"SV-SCN Version: {summary.svscn_version}")
    print(f"SAM-3D Version: {summary.sam3d_version}")
    
    print("\n--- Sample Counts ---")
    print(f"Total samples: {summary.total_samples}")
    print(f"SV-SCN successful: {summary.svscn_successful}")
    print(f"SAM-3D successful: {summary.sam3d_successful}")
    print(f"Both successful: {summary.both_successful}")
    
    print("\n--- Quality Assessment ---")
    print(f"Acceptable (AR-ready): {summary.svscn_acceptable}")
    print(f"Needs improvement: {summary.svscn_needs_improvement}")
    print(f"Failed: {summary.svscn_failed}")
    
    if summary.mean_speedup > 0:
        print("\n--- Speed Comparison ---")
        print(f"Mean speedup over SAM-3D: {summary.mean_speedup:.1f}x")
        print(f"Median speedup: {summary.median_speedup:.1f}x")
    
    if summary.failure_mode_counts:
        print("\n--- Failure Modes ---")
        for mode, count in sorted(summary.failure_mode_counts.items(), key=lambda x: -x[1]):
            print(f"  {mode}: {count}")
    
    if summary.priority_improvements:
        print("\n--- Recommendations ---")
        for rec in summary.priority_improvements:
            print(f"  • {rec}")
    
    print("\n" + "=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Compare SV-SCN outputs against SAM-3D baseline"
    )
    parser.add_argument(
        "--svscn_checkpoint",
        type=Path,
        required=True,
        help="Path to SV-SCN model checkpoint"
    )
    parser.add_argument(
        "--sam3d_metrics",
        type=Path,
        default=Path("metrics"),
        help="Directory containing SAM-3D benchmark results"
    )
    parser.add_argument(
        "--test_data",
        type=Path,
        default=Path("test_samples"),
        help="Directory containing test partial point clouds"
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("comparisons"),
        help="Directory to save comparison results"
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
    print("⚠️  SV-SCN vs SAM-3D COMPARISON - INTERNAL RESEARCH ONLY")
    print("=" * 60 + "\n")
    
    # Run comparison
    run_comparison(
        svscn_checkpoint=args.svscn_checkpoint,
        sam3d_metrics_dir=args.sam3d_metrics,
        test_data_dir=args.test_data,
        output_dir=args.output_dir
    )


if __name__ == "__main__":
    main()
