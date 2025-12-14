#!/usr/bin/env python
"""
SV-SCN Inference Script

Run inference on partial point clouds to generate complete 3D models.

Usage:
    python scripts/infer.py \
        --checkpoint checkpoints/best.pt \
        --input sample_partial.npy \
        --output result.glb \
        --class_id 0

With mesh export:
    python scripts/infer.py \
        --checkpoint checkpoints/best.pt \
        --input sample_partial.npy \
        --output result.glb \
        --export_mesh \
        --class_id 0
"""

import sys
import argparse
import logging
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from svscn.inference import Predictor, point_cloud_to_mesh, export_mesh
from svscn.inference.export import export_pointcloud, export_for_ar

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run SV-SCN inference on partial point cloud",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Input partial point cloud (.npy)"
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output file path"
    )
    parser.add_argument(
        "--class_id",
        type=int,
        default=0,
        choices=[0, 1, 2],
        help="Object class: 0=chair, 1=stool, 2=table"
    )
    parser.add_argument(
        "--export_mesh",
        action="store_true",
        help="Convert to mesh before export"
    )
    parser.add_argument(
        "--export_ar",
        action="store_true",
        help="Export AR-optimized formats (GLB + USDZ)"
    )
    parser.add_argument(
        "--mesh_method",
        type=str,
        default="ball_pivoting",
        choices=["ball_pivoting", "poisson"],
        help="Mesh reconstruction method"
    )
    parser.add_argument(
        "--no_fallback",
        action="store_true",
        help="Disable symmetry fallback for low confidence"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device (cuda/cpu)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose output"
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(message)s"
    )
    
    # Check inputs
    if not args.checkpoint.exists():
        logger.error(f"Checkpoint not found: {args.checkpoint}")
        return 1
    
    if not args.input.exists():
        logger.error(f"Input file not found: {args.input}")
        return 1
    
    # Load predictor
    logger.info(f"Loading model from {args.checkpoint}")
    predictor = Predictor.load(args.checkpoint, device=args.device)
    
    # Load input
    logger.info(f"Loading input: {args.input}")
    partial = np.load(args.input)
    logger.info(f"  Shape: {partial.shape}")
    
    # Class names
    class_names = ["chair", "stool", "table"]
    logger.info(f"  Class: {class_names[args.class_id]}")
    
    # Run inference
    logger.info("Running inference...")
    result = predictor.predict(
        partial,
        args.class_id,
        use_fallback=not args.no_fallback
    )
    
    logger.info(f"  Confidence: {result.confidence:.2f}")
    logger.info(f"  Used fallback: {result.used_fallback}")
    logger.info(f"  Output shape: {result.completed.shape}")
    
    # Export
    output_path = args.output
    
    if args.export_ar:
        # Export all AR formats
        logger.info("Exporting AR-optimized formats...")
        
        # First convert to mesh
        mesh_result = point_cloud_to_mesh(
            result.completed,
            method=args.mesh_method
        )
        
        if mesh_result:
            exports = export_for_ar(
                mesh_result,
                output_path.parent,
                name=output_path.stem
            )
            
            for fmt, path in exports.items():
                logger.info(f"  {fmt.upper()}: {path}")
        else:
            logger.error("Mesh reconstruction failed")
            return 1
    
    elif args.export_mesh:
        # Export as mesh
        logger.info(f"Converting to mesh ({args.mesh_method})...")
        mesh_result = point_cloud_to_mesh(
            result.completed,
            method=args.mesh_method
        )
        
        if mesh_result:
            logger.info(f"  Vertices: {mesh_result.num_vertices}")
            logger.info(f"  Faces: {mesh_result.num_faces}")
            
            if export_mesh(mesh_result, output_path):
                logger.info(f"Saved mesh to: {output_path}")
            else:
                logger.error("Mesh export failed")
                return 1
        else:
            logger.error("Mesh reconstruction failed")
            return 1
    
    else:
        # Export as point cloud
        suffix = output_path.suffix.lower()
        
        if suffix == ".npy":
            np.save(output_path, result.completed)
            logger.info(f"Saved point cloud to: {output_path}")
        else:
            if export_pointcloud(result.completed, output_path):
                logger.info(f"Saved point cloud to: {output_path}")
            else:
                logger.error("Point cloud export failed")
                return 1
    
    # Print summary
    print("\n" + "=" * 50)
    print("Inference Complete!")
    print("=" * 50)
    print(f"Class: {result.class_name}")
    print(f"Confidence: {result.confidence:.2f}")
    print(f"Output: {output_path}")
    
    if result.confidence < 0.6:
        print("\n⚠️  Low confidence detected!")
        if result.used_fallback:
            print("   Symmetry fallback was used.")
        else:
            print("   Consider reviewing output quality.")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
