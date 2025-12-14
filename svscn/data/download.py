"""
Objaverse Furniture Downloader

Downloads and filters furniture models from Objaverse.
Secondary dataset source (targets 75% of final mix).

Strict filtering applied:
- CC-BY/CC-0 license only
- Mesh validation (no holes, self-intersections)
- Furniture keyword matching
- Quality score filtering

Usage:
    python -m svscn.data.objaverse_download --data_dir data/objaverse
"""

import os
import json
import logging
from pathlib import Path
from typing import List, Dict, Optional, Set, Tuple
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib

import numpy as np
from tqdm import tqdm

logger = logging.getLogger(__name__)


# Furniture-related keywords for strict filtering
FURNITURE_KEYWORDS = {
    "chair": {
        "required": ["chair"],
        "optional": ["seat", "armchair", "office", "dining", "wooden", "leather"],
        "exclude": ["wheelchair", "chairlift", "chairperson", "chairman"]
    },
    "stool": {
        "required": ["stool"],
        "optional": ["bar", "step", "kitchen", "wooden"],
        "exclude": ["footstool"]  # footstools are too simple
    },
    "table": {
        "required": ["table"],
        "optional": ["desk", "coffee", "side", "dining", "wooden", "glass"],
        "exclude": ["tablet", "table_tennis", "timetable", "vegetable", "tablecloth"]
    }
}

# Allowed licenses (commercial-safe)
ALLOWED_LICENSES = {
    "by",         # CC-BY
    "by-sa",      # CC-BY-SA 
    "cc0",        # Public domain
    "cc-by",
    "cc-by-4.0",
    "cc-by-3.0",
    "cc-0",
    "public domain",
    "0",
}

# Rejected licenses
REJECTED_LICENSES = {
    "nc",         # Non-commercial
    "nd",         # No derivatives
    "all rights reserved",
}


@dataclass
class ObjaverseModel:
    """Metadata for an Objaverse model."""
    
    uid: str
    name: str
    category: str
    license: str
    tags: List[str]
    
    # Quality metrics
    quality_score: float = 0.0
    keyword_score: float = 0.0
    
    # Validation status
    is_valid: bool = False
    validation_notes: str = ""
    
    # Paths
    source_url: Optional[str] = None
    local_path: Optional[Path] = None


@dataclass
class FilterStats:
    """Statistics from filtering process."""
    
    total_scanned: int = 0
    license_rejected: int = 0
    keyword_rejected: int = 0
    quality_rejected: int = 0
    mesh_invalid: int = 0
    accepted: int = 0
    
    by_category: Dict[str, int] = None
    
    def __post_init__(self):
        if self.by_category is None:
            self.by_category = {}


def check_license_clean(license_str: str) -> Tuple[bool, str]:
    """
    Check if license is commercially usable.
    
    Returns:
        (is_clean, reason)
    """
    if not license_str:
        return False, "No license specified"
    
    license_lower = license_str.lower()
    
    # Check for rejected licenses first
    for reject in REJECTED_LICENSES:
        if reject in license_lower:
            return False, f"Contains '{reject}'"
    
    # Check for allowed licenses
    for allowed in ALLOWED_LICENSES:
        if allowed in license_lower:
            return True, f"Allowed: {allowed}"
    
    return False, f"Unknown license: {license_str}"


def compute_keyword_score(
    name: str,
    tags: List[str],
    category: str
) -> Tuple[float, bool]:
    """
    Compute how well a model matches furniture category.
    
    Returns:
        (score 0-1, passes_filter)
    """
    keywords = FURNITURE_KEYWORDS.get(category)
    if not keywords:
        return 0.0, False
    
    text = (name + " " + " ".join(tags)).lower()
    
    # Check exclusions first
    for exclude in keywords.get("exclude", []):
        if exclude in text:
            return 0.0, False
    
    # Must have at least one required keyword
    has_required = any(req in text for req in keywords["required"])
    if not has_required:
        return 0.0, False
    
    # Score based on optional keywords
    optional_hits = sum(1 for opt in keywords.get("optional", []) if opt in text)
    optional_score = optional_hits / max(len(keywords.get("optional", [])), 1)
    
    # Final score: 0.5 base for having required, up to 1.0 with optionals
    score = 0.5 + 0.5 * optional_score
    
    return score, True


def validate_objaverse_mesh(mesh_path: Path) -> Tuple[bool, str, float]:
    """
    Validate Objaverse mesh with strict criteria.
    
    Returns:
        (is_valid, notes, quality_score)
    """
    try:
        import trimesh
        
        mesh = trimesh.load(mesh_path, force='mesh')
        
        notes = []
        quality = 1.0
        is_valid = True
        
        # --- Strict validation criteria ---
        
        # 1. Vertex count
        if len(mesh.vertices) < 100:
            notes.append("Too few vertices (<100)")
            is_valid = False
        elif len(mesh.vertices) > 200000:
            notes.append("Very high vertex count (>200k)")
            quality -= 0.2
        
        # 2. Face count
        if len(mesh.faces) < 50:
            notes.append("Too few faces (<50)")
            is_valid = False
        
        # 3. Degenerate faces
        face_areas = mesh.area_faces
        degenerate_ratio = np.sum(face_areas < 1e-10) / len(face_areas)
        if degenerate_ratio > 0.05:
            notes.append(f"{degenerate_ratio*100:.1f}% degenerate faces")
            is_valid = False
        elif degenerate_ratio > 0.01:
            quality -= 0.1
        
        # 4. Bounding box sanity
        bbox = mesh.bounding_box.extents
        if np.any(bbox < 1e-6):
            notes.append("Degenerate bounding box")
            is_valid = False
        
        # 5. Aspect ratio
        aspect = max(bbox) / (min(bbox) + 1e-10)
        if aspect > 50:
            notes.append(f"Extreme aspect ratio ({aspect:.1f})")
            is_valid = False
        elif aspect > 20:
            quality -= 0.1
        
        # 6. Watertight check
        if not mesh.is_watertight:
            notes.append("Not watertight (has holes)")
            quality -= 0.2
        
        # 7. Self-intersections (expensive, sample-based)
        # Note: Full check is expensive, so we do a quick heuristic
        if hasattr(mesh, 'is_volume') and not mesh.is_volume:
            notes.append("Not a valid volume")
            quality -= 0.1
        
        # 8. Connected components
        try:
            components = mesh.split(only_watertight=False)
            if len(components) > 10:
                notes.append(f"Too many components ({len(components)})")
                is_valid = False
            elif len(components) > 3:
                quality -= 0.1
        except:
            pass
        
        quality = max(0.0, min(1.0, quality))
        
        return is_valid, "; ".join(notes) if notes else "Valid", quality
        
    except Exception as e:
        return False, f"Load error: {e}", 0.0


def filter_objaverse_annotations(
    categories: List[str],
    max_per_category: int = 2000,
    min_quality_score: float = 0.5
) -> Dict[str, List[ObjaverseModel]]:
    """
    Filter Objaverse annotations for suitable furniture models.
    
    Args:
        categories: List of categories to filter for
        max_per_category: Maximum models per category
        min_quality_score: Minimum keyword score to accept
    
    Returns:
        Dict mapping category to list of candidate models
    """
    try:
        import objaverse
    except ImportError:
        logger.error("objaverse package not installed. Run: pip install objaverse")
        return {}
    
    logger.info("Loading Objaverse annotations...")
    annotations = objaverse.load_annotations()
    
    stats = FilterStats()
    result = {cat: [] for cat in categories}
    
    for uid, info in tqdm(annotations.items(), desc="Filtering"):
        stats.total_scanned += 1
        
        # Check license
        license_str = info.get("license", "")
        is_clean, license_reason = check_license_clean(license_str)
        if not is_clean:
            stats.license_rejected += 1
            continue
        
        # Get metadata
        name = info.get("name", "").strip()
        tags = [t.strip().lower() for t in info.get("tags", [])]
        
        # Check each category
        for category in categories:
            if len(result[category]) >= max_per_category:
                continue
            
            score, passes = compute_keyword_score(name, tags, category)
            
            if not passes:
                continue
            
            if score < min_quality_score:
                stats.keyword_rejected += 1
                continue
            
            model = ObjaverseModel(
                uid=uid,
                name=name,
                category=category,
                license=license_str,
                tags=tags,
                keyword_score=score
            )
            
            result[category].append(model)
            stats.accepted += 1
            break  # Only assign to one category
    
    for cat, models in result.items():
        stats.by_category[cat] = len(models)
        logger.info(f"{cat}: {len(models)} candidates")
    
    logger.info(f"Filter stats: {stats.total_scanned} scanned, "
                f"{stats.accepted} accepted, "
                f"{stats.license_rejected} license rejected")
    
    return result


def download_and_validate(
    candidates: Dict[str, List[ObjaverseModel]],
    output_dir: Path,
    samples_per_category: int = 1000,
    validate_meshes: bool = True,
    num_workers: int = 4
) -> Dict[str, List[Path]]:
    """
    Download candidates and validate meshes.
    
    Args:
        candidates: Dict from filter_objaverse_annotations
        output_dir: Where to save validated models
        samples_per_category: Target valid samples per category
        validate_meshes: Whether to run mesh validation
        num_workers: Download parallelism
    
    Returns:
        Dict mapping category to list of valid model paths
    """
    try:
        import objaverse
    except ImportError:
        logger.error("objaverse package not installed")
        return {}
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    result = {cat: [] for cat in candidates.keys()}
    stats = FilterStats()
    
    for category, models in candidates.items():
        logger.info(f"Processing {category}: {len(models)} candidates")
        
        cat_dir = output_dir / category
        cat_dir.mkdir(exist_ok=True)
        
        # Sort by keyword score (best first)
        models_sorted = sorted(models, key=lambda m: -m.keyword_score)
        
        # Get UIDs to download
        uids_to_download = [m.uid for m in models_sorted[:samples_per_category * 2]]
        
        if not uids_to_download:
            continue
        
        # Download
        logger.info(f"Downloading {len(uids_to_download)} models...")
        downloaded = objaverse.load_objects(
            uids=uids_to_download,
            download_processes=num_workers
        )
        
        # Validate and copy
        valid_count = 0
        
        for model in models_sorted:
            if valid_count >= samples_per_category:
                break
            
            if model.uid not in downloaded:
                continue
            
            source_path = Path(downloaded[model.uid])
            
            # Validate mesh
            if validate_meshes:
                is_valid, notes, quality = validate_objaverse_mesh(source_path)
                model.is_valid = is_valid
                model.validation_notes = notes
                model.quality_score = quality
                
                if not is_valid:
                    stats.mesh_invalid += 1
                    continue
            else:
                model.is_valid = True
                model.quality_score = model.keyword_score
            
            # Copy to output
            dest_path = cat_dir / f"{model.uid}.glb"
            if not dest_path.exists():
                import shutil
                shutil.copy2(source_path, dest_path)
            
            model.local_path = dest_path
            result[category].append(dest_path)
            valid_count += 1
        
        stats.by_category[category] = valid_count
        logger.info(f"  {category}: {valid_count} valid models")
    
    # Save metadata
    metadata = {
        "stats": asdict(stats),
        "categories": {
            cat: [str(p) for p in paths] 
            for cat, paths in result.items()
        }
    }
    
    meta_path = output_dir / "objaverse_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    
    return result


def prepare_objaverse_dataset(
    output_dir: Path,
    categories: Optional[List[str]] = None,
    samples_per_category: int = 1000,
    validate: bool = True
) -> Dict[str, List[Path]]:
    """
    Full pipeline: filter, download, validate Objaverse furniture.
    
    Args:
        output_dir: Output directory
        categories: Categories to download (default: v1)
        samples_per_category: Target per category
        validate: Whether to validate meshes
    
    Returns:
        Dict mapping category to valid model paths
    """
    if categories is None:
        from ..config import default_config
        categories = default_config.data.FURNITURE_CLASSES
    
    # Step 1: Filter annotations
    candidates = filter_objaverse_annotations(
        categories=categories,
        max_per_category=samples_per_category * 3  # Extra for filtering
    )
    
    # Step 2: Download and validate
    result = download_and_validate(
        candidates=candidates,
        output_dir=output_dir,
        samples_per_category=samples_per_category,
        validate_meshes=validate
    )
    
    return result


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Download and filter Objaverse furniture models"
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("data/objaverse"),
        help="Output directory"
    )
    parser.add_argument(
        "--samples_per_category",
        type=int,
        default=1000,
        help="Target samples per category"
    )
    parser.add_argument(
        "--category",
        type=str,
        default=None,
        help="Process single category"
    )
    parser.add_argument(
        "--no_validate",
        action="store_true",
        help="Skip mesh validation"
    )
    parser.add_argument(
        "--filter_only",
        action="store_true",
        help="Only run annotation filtering (no download)"
    )
    
    args = parser.parse_args()
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s"
    )
    
    categories = [args.category] if args.category else ["chair", "stool", "table"]
    
    if args.filter_only:
        candidates = filter_objaverse_annotations(
            categories=categories,
            max_per_category=args.samples_per_category * 3
        )
        
        # Save candidates
        output_path = args.output_dir / "candidates.json"
        args.output_dir.mkdir(parents=True, exist_ok=True)
        
        candidates_dict = {
            cat: [asdict(m) for m in models]
            for cat, models in candidates.items()
        }
        
        with open(output_path, "w") as f:
            json.dump(candidates_dict, f, indent=2)
        
        logger.info(f"Candidates saved to {output_path}")
        return
    
    prepare_objaverse_dataset(
        output_dir=args.output_dir,
        categories=categories,
        samples_per_category=args.samples_per_category,
        validate=not args.no_validate
    )


if __name__ == "__main__":
    main()
