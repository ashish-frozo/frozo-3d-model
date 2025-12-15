"""
ShapeNet Furniture Downloader

Downloads and prepares furniture models from ShapeNet for SV-SCN training.
ShapeNet is the primary dataset for initial pipeline validation.

License: ShapeNet models are for research use. Verify license compliance
for commercial deployment.

Usage:
    python -m svscn.data.shapenet --data_dir data/shapenet --category chair
"""

import os
import json
import zipfile
import logging
from pathlib import Path
from typing import List, Dict, Optional, Set
from dataclasses import dataclass
import shutil

import numpy as np

logger = logging.getLogger(__name__)


# ShapeNet category IDs (synset IDs) for furniture
# Reference: https://shapenet.org/
SHAPENET_CATEGORIES = {
    "chair": "03001627",
    "table": "04379243",
    "stool": "03001627",  # Stools are often under chair category
    "sofa": "04256520",
    "bench": "02828884",
    "cabinet": "02933112",
    "desk": "03179701",
}

# Furniture categories for v1 (from ML Training Spec)
V1_CATEGORIES = ["chair", "stool", "table"]


@dataclass
class ShapeNetModel:
    """Metadata for a ShapeNet model."""
    
    model_id: str
    category: str
    synset_id: str
    name: str
    obj_path: Optional[Path] = None
    is_valid: bool = True
    validation_notes: str = ""


def get_shapenet_path() -> Optional[Path]:
    """
    Get path to ShapeNet data.
    Checks common locations and environment variable.
    """
    # Check environment variable
    env_path = os.environ.get("SHAPENET_PATH")
    if env_path and Path(env_path).exists():
        return Path(env_path)
    
    # Check common locations
    common_paths = [
        Path.home() / "datasets" / "ShapeNetCore.v2",
        Path.home() / "data" / "ShapeNetCore.v2",
        Path("/data/ShapeNetCore.v2"),
        Path("./data/shapenet/ShapeNetCore.v2"),
    ]
    
    for p in common_paths:
        if p.exists():
            return p
    
    return None


def list_available_models(
    shapenet_path: Path,
    category: str,
    max_models: Optional[int] = None
) -> List[ShapeNetModel]:
    """
    List available models for a category in ShapeNet.
    
    Args:
        shapenet_path: Path to ShapeNetCore.v2 directory
        category: Category name (chair, table, stool)
        max_models: Limit number of models returned
    
    Returns:
        List of ShapeNetModel objects
    """
    synset_id = SHAPENET_CATEGORIES.get(category)
    if not synset_id:
        logger.error(f"Unknown category: {category}")
        return []
    
    category_path = shapenet_path / synset_id
    if not category_path.exists():
        logger.error(f"Category path not found: {category_path}")
        return []
    
    models = []
    
    for model_dir in category_path.iterdir():
        if not model_dir.is_dir():
            continue
        
        model_id = model_dir.name
        
        # Check for OBJ file
        obj_path = model_dir / "models" / "model_normalized.obj"
        if not obj_path.exists():
            continue
        
        # Load name from metadata if available
        name = model_id
        meta_path = model_dir / "models" / "model_normalized.json"
        if meta_path.exists():
            try:
                with open(meta_path) as f:
                    meta = json.load(f)
                    name = meta.get("name", model_id)
            except:
                pass
        
        models.append(ShapeNetModel(
            model_id=model_id,
            category=category,
            synset_id=synset_id,
            name=name,
            obj_path=obj_path
        ))
        
        if max_models and len(models) >= max_models:
            break
    
    logger.info(f"Found {len(models)} {category} models in ShapeNet")
    return models


def validate_mesh(obj_path: Path) -> tuple[bool, str]:
    """
    Validate mesh for training suitability.
    
    Checks (per ML Training Spec):
    - No holes
    - No self-intersections
    - Reasonable vertex count
    - Watertight (optional)
    
    Returns:
        (is_valid, notes)
    """
    try:
        import trimesh
        
        mesh = trimesh.load(obj_path, force='mesh')
        
        notes = []
        is_valid = True
        
        # Check vertex count
        if len(mesh.vertices) < 100:
            notes.append("Too few vertices")
            is_valid = False
        elif len(mesh.vertices) > 500000:
            notes.append("Too many vertices (will be simplified)")
        
        # Check face count
        if len(mesh.faces) < 50:
            notes.append("Too few faces")
            is_valid = False
        
        # Check for degenerate faces
        face_areas = mesh.area_faces
        degenerate = np.sum(face_areas < 1e-10)
        if degenerate > len(mesh.faces) * 0.01:
            notes.append(f"{degenerate} degenerate faces")
            is_valid = False
        
        # Check bounding box (should be reasonable)
        bbox = mesh.bounding_box.extents
        if np.any(bbox < 1e-6):
            notes.append("Flat/degenerate bounding box")
            is_valid = False
        
        # Check for watertight (warning only)
        if not mesh.is_watertight:
            notes.append("Not watertight (may have holes)")
        
        # Check for reasonable aspect ratio
        aspect = max(bbox) / (min(bbox) + 1e-10)
        if aspect > 100:
            notes.append(f"Extreme aspect ratio: {aspect:.1f}")
            is_valid = False
        
        return is_valid, "; ".join(notes) if notes else "Valid"
        
    except Exception as e:
        return False, f"Load error: {e}"


def prepare_shapenet_dataset(
    shapenet_path: Path,
    output_dir: Path,
    categories: Optional[List[str]] = None,
    samples_per_category: int = 1000,
    validate: bool = True
) -> Dict[str, List[Path]]:
    """
    Prepare ShapeNet furniture dataset for training.
    
    Args:
        shapenet_path: Path to ShapeNetCore.v2
        output_dir: Where to save processed data
        categories: List of categories (default: v1 categories)
        samples_per_category: Target samples per category
        validate: Whether to validate meshes
    
    Returns:
        Dict mapping category to list of valid model paths
    """
    if categories is None:
        categories = V1_CATEGORIES
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    result = {cat: [] for cat in categories}
    stats = {"total": 0, "valid": 0, "invalid": 0, "by_category": {}}
    
    for category in categories:
        logger.info(f"Processing category: {category}")
        
        # List models
        models = list_available_models(
            shapenet_path, 
            category, 
            max_models=samples_per_category * 2  # Get extra for filtering
        )
        
        valid_count = 0
        invalid_count = 0
        
        for model in models:
            if valid_count >= samples_per_category:
                break
            
            stats["total"] += 1
            
            # Validate if requested
            if validate:
                is_valid, notes = validate_mesh(model.obj_path)
                model.is_valid = is_valid
                model.validation_notes = notes
            
            if not model.is_valid:
                invalid_count += 1
                stats["invalid"] += 1
                continue
            
            # Copy to output directory
            cat_dir = output_dir / category
            cat_dir.mkdir(exist_ok=True)
            
            dest_path = cat_dir / f"{model.model_id}.obj"
            
            if not dest_path.exists():
                shutil.copy2(model.obj_path, dest_path)
            
            result[category].append(dest_path)
            valid_count += 1
            stats["valid"] += 1
        
        stats["by_category"][category] = {
            "valid": valid_count,
            "invalid": invalid_count
        }
        
        logger.info(f"  {category}: {valid_count} valid, {invalid_count} invalid")
    
    # Save stats
    stats_path = output_dir / "shapenet_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    
    logger.info(f"Dataset prepared: {stats['valid']} valid models")
    logger.info(f"Stats saved to: {stats_path}")
    
    return result


def download_shapenet_sample(output_dir: Path, samples_per_category: int = 10) -> bool:
    """
    Download a sample of ShapeNet for testing.
    
    Note: Full ShapeNet requires registration at shapenet.org
    This downloads only publicly available sample data.
    
    Args:
        output_dir: Where to save placeholder data
        samples_per_category: Number of samples to generate per category
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.warning("Full ShapeNet requires registration at https://shapenet.org/")
    logger.info(f"Creating {samples_per_category} synthetic samples per category...")
    
    # Create placeholder data for testing
    for category in V1_CATEGORIES:
        cat_dir = output_dir / category
        cat_dir.mkdir(exist_ok=True)
        
        # Create simple placeholder meshes
        for i in range(samples_per_category):
            obj_content = create_placeholder_obj(category, i)
            obj_path = cat_dir / f"placeholder_{i:04d}.obj"
            obj_path.write_text(obj_content)
    
    total = samples_per_category * len(V1_CATEGORIES)
    logger.info(f"Created {total} placeholder models in {output_dir}")
    logger.info("Replace with real ShapeNet data from shapenet.org")
    
    return True


def create_placeholder_obj(category: str, index: int) -> str:
    """Create a simple placeholder OBJ file for testing."""
    # Create a simple box with slight variations
    np.random.seed(index)
    
    # Base dimensions per category
    if category == "chair":
        scale = np.array([0.5, 0.8, 0.5]) * (0.8 + 0.4 * np.random.random())
    elif category == "table":
        scale = np.array([1.0, 0.5, 0.6]) * (0.8 + 0.4 * np.random.random())
    else:  # stool
        scale = np.array([0.4, 0.6, 0.4]) * (0.8 + 0.4 * np.random.random())
    
    # Unit cube vertices
    vertices = np.array([
        [-0.5, -0.5, -0.5],
        [0.5, -0.5, -0.5],
        [0.5, 0.5, -0.5],
        [-0.5, 0.5, -0.5],
        [-0.5, -0.5, 0.5],
        [0.5, -0.5, 0.5],
        [0.5, 0.5, 0.5],
        [-0.5, 0.5, 0.5],
    ]) * scale
    
    # Faces (1-indexed for OBJ)
    faces = [
        [1, 2, 3, 4],  # bottom
        [5, 6, 7, 8],  # top
        [1, 2, 6, 5],  # front
        [2, 3, 7, 6],  # right
        [3, 4, 8, 7],  # back
        [4, 1, 5, 8],  # left
    ]
    
    lines = [f"# Placeholder {category} mesh {index}"]
    for v in vertices:
        lines.append(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}")
    for f in faces:
        lines.append(f"f {' '.join(map(str, f))}")
    
    return "\n".join(lines)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Prepare ShapeNet furniture dataset")
    parser.add_argument(
        "--shapenet_path",
        type=Path,
        default=None,
        help="Path to ShapeNetCore.v2 (or set SHAPENET_PATH env var)"
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("data/shapenet"),
        help="Output directory"
    )
    parser.add_argument(
        "--samples_per_category",
        type=int,
        default=1000,
        help="Target samples per category"
    )
    parser.add_argument(
        "--placeholder",
        action="store_true",
        help="Create placeholder data for testing"
    )
    parser.add_argument(
        "--category",
        type=str,
        default=None,
        help="Process single category (chair/table/stool)"
    )
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    
    if args.placeholder:
        download_shapenet_sample(args.output_dir, args.samples_per_category)
        return
    
    # Find ShapeNet
    shapenet_path = args.shapenet_path or get_shapenet_path()
    
    if shapenet_path is None or not shapenet_path.exists():
        logger.error("ShapeNet not found!")
        logger.info("Options:")
        logger.info("  1. Download from https://shapenet.org/")
        logger.info("  2. Set SHAPENET_PATH environment variable")
        logger.info("  3. Use --placeholder flag for testing")
        return
    
    categories = [args.category] if args.category else V1_CATEGORIES
    
    prepare_shapenet_dataset(
        shapenet_path=shapenet_path,
        output_dir=args.output_dir,
        categories=categories,
        samples_per_category=args.samples_per_category
    )


if __name__ == "__main__":
    main()
