"""
Combined Dataset Manager

Manages the combined ShapeNet + Objaverse dataset with:
- 75% Objaverse / 25% ShapeNet target mix
- Unified preprocessing
- Split management (train/val/test with no object overlap)

Usage:
    python -m svscn.data.dataset_manager --prepare --output_dir data/combined
"""

import os
import json
import logging
import random
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Set
from dataclasses import dataclass, asdict, field
from collections import defaultdict
import hashlib

import numpy as np

from .shapenet import (
    prepare_shapenet_dataset, 
    get_shapenet_path,
    V1_CATEGORIES as SHAPENET_CATEGORIES
)
from .download import prepare_objaverse_dataset

logger = logging.getLogger(__name__)


@dataclass
class DatasetSplit:
    """Information about a dataset split."""
    
    name: str  # train, val, test
    samples: List[str]  # List of sample IDs
    by_category: Dict[str, List[str]] = field(default_factory=dict)
    by_source: Dict[str, List[str]] = field(default_factory=dict)


@dataclass
class CombinedDatasetConfig:
    """Configuration for combined dataset."""
    
    # Target mix
    objaverse_ratio: float = 0.75  # 75% Objaverse
    shapenet_ratio: float = 0.25   # 25% ShapeNet
    
    # Split ratios (per ML Training Spec)
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    
    # Categories
    categories: List[str] = field(
        default_factory=lambda: ["chair", "stool", "table"]
    )
    
    # Target sizes
    target_total: int = 5000  # Baseline
    min_per_category: int = 1000
    
    # Random seed for reproducibility
    seed: int = 42


def compute_sample_id(source: str, category: str, filename: str) -> str:
    """Generate unique sample ID."""
    # Use hash to create consistent, short ID
    content = f"{source}:{category}:{filename}"
    hash_val = hashlib.md5(content.encode()).hexdigest()[:8]
    return f"{source[:2]}_{category[:2]}_{hash_val}"


def create_splits(
    samples: Dict[str, List[Dict]],
    config: CombinedDatasetConfig
) -> Tuple[DatasetSplit, DatasetSplit, DatasetSplit]:
    """
    Create train/val/test splits with no object overlap.
    
    Stratified by category and source to maintain ratios.
    """
    random.seed(config.seed)
    np.random.seed(config.seed)
    
    # Group samples by category
    by_category = defaultdict(list)
    for source, source_samples in samples.items():
        for sample in source_samples:
            cat = sample["category"]
            sample["source"] = source
            by_category[cat].append(sample)
    
    # Initialize splits
    train = DatasetSplit(name="train", samples=[], by_category={}, by_source={})
    val = DatasetSplit(name="val", samples=[], by_category={}, by_source={})
    test = DatasetSplit(name="test", samples=[], by_category={}, by_source={})
    
    for split in [train, val, test]:
        for cat in config.categories:
            split.by_category[cat] = []
        split.by_source["shapenet"] = []
        split.by_source["objaverse"] = []
    
    # Split each category
    for cat, cat_samples in by_category.items():
        random.shuffle(cat_samples)
        
        n = len(cat_samples)
        n_train = int(n * config.train_ratio)
        n_val = int(n * config.val_ratio)
        
        train_samples = cat_samples[:n_train]
        val_samples = cat_samples[n_train:n_train + n_val]
        test_samples = cat_samples[n_train + n_val:]
        
        # Add to splits
        for sample in train_samples:
            sid = sample["id"]
            train.samples.append(sid)
            train.by_category[cat].append(sid)
            train.by_source[sample["source"]].append(sid)
        
        for sample in val_samples:
            sid = sample["id"]
            val.samples.append(sid)
            val.by_category[cat].append(sid)
            val.by_source[sample["source"]].append(sid)
        
        for sample in test_samples:
            sid = sample["id"]
            test.samples.append(sid)
            test.by_category[cat].append(sid)
            test.by_source[sample["source"]].append(sid)
    
    return train, val, test


def prepare_combined_dataset(
    output_dir: Path,
    shapenet_path: Optional[Path] = None,
    config: Optional[CombinedDatasetConfig] = None
) -> Dict:
    """
    Prepare combined ShapeNet + Objaverse dataset.
    
    Pipeline:
    1. Prepare ShapeNet (25% target)
    2. Prepare Objaverse (75% target)
    3. Combine and create splits
    4. Save metadata
    
    Args:
        output_dir: Where to save combined dataset
        shapenet_path: Path to ShapeNetCore.v2
        config: Dataset configuration
    
    Returns:
        Dataset metadata dict
    """
    if config is None:
        config = CombinedDatasetConfig()
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Calculate target samples per source
    target_per_category = config.target_total // len(config.categories)
    shapenet_per_cat = int(target_per_category * config.shapenet_ratio)
    objaverse_per_cat = int(target_per_category * config.objaverse_ratio)
    
    logger.info(f"Target per category: {target_per_category}")
    logger.info(f"  ShapeNet: {shapenet_per_cat} ({config.shapenet_ratio*100:.0f}%)")
    logger.info(f"  Objaverse: {objaverse_per_cat} ({config.objaverse_ratio*100:.0f}%)")
    
    samples = {"shapenet": [], "objaverse": []}
    
    # --- Step 1: Prepare ShapeNet ---
    logger.info("=" * 50)
    logger.info("Step 1: Preparing ShapeNet dataset")
    logger.info("=" * 50)
    
    shapenet_path = shapenet_path or get_shapenet_path()
    
    if shapenet_path and shapenet_path.exists():
        shapenet_dir = output_dir / "raw" / "shapenet"
        shapenet_models = prepare_shapenet_dataset(
            shapenet_path=shapenet_path,
            output_dir=shapenet_dir,
            categories=config.categories,
            samples_per_category=shapenet_per_cat
        )
        
        for cat, paths in shapenet_models.items():
            for path in paths:
                sample_id = compute_sample_id("shapenet", cat, path.name)
                samples["shapenet"].append({
                    "id": sample_id,
                    "category": cat,
                    "path": str(path),
                    "filename": path.name
                })
    else:
        logger.warning("ShapeNet not found. Creating placeholder data...")
        from .shapenet import download_shapenet_sample
        shapenet_dir = output_dir / "raw" / "shapenet"
        download_shapenet_sample(shapenet_dir)
        
        for cat in config.categories:
            cat_dir = shapenet_dir / cat
            if cat_dir.exists():
                for path in cat_dir.glob("*.obj"):
                    sample_id = compute_sample_id("shapenet", cat, path.name)
                    samples["shapenet"].append({
                        "id": sample_id,
                        "category": cat,
                        "path": str(path),
                        "filename": path.name
                    })
    
    logger.info(f"ShapeNet: {len(samples['shapenet'])} samples")
    
    # --- Step 2: Prepare Objaverse ---
    logger.info("=" * 50)
    logger.info("Step 2: Preparing Objaverse dataset")
    logger.info("=" * 50)
    
    try:
        objaverse_dir = output_dir / "raw" / "objaverse"
        objaverse_models = prepare_objaverse_dataset(
            output_dir=objaverse_dir,
            categories=config.categories,
            samples_per_category=objaverse_per_cat,
            validate=True
        )
        
        for cat, paths in objaverse_models.items():
            for path in paths:
                sample_id = compute_sample_id("objaverse", cat, path.name)
                samples["objaverse"].append({
                    "id": sample_id,
                    "category": cat,
                    "path": str(path),
                    "filename": path.name
                })
    except Exception as e:
        logger.warning(f"Objaverse preparation failed: {e}")
        logger.warning("Continuing with ShapeNet only")
    
    logger.info(f"Objaverse: {len(samples['objaverse'])} samples")
    
    # --- Step 3: Create splits ---
    logger.info("=" * 50)
    logger.info("Step 3: Creating train/val/test splits")
    logger.info("=" * 50)
    
    train_split, val_split, test_split = create_splits(samples, config)
    
    # --- Step 4: Save metadata ---
    logger.info("=" * 50)
    logger.info("Step 4: Saving metadata")
    logger.info("=" * 50)
    
    # Create sample lookup
    sample_lookup = {}
    for source, source_samples in samples.items():
        for sample in source_samples:
            sample_lookup[sample["id"]] = sample
    
    metadata = {
        "config": asdict(config),
        "stats": {
            "total": len(sample_lookup),
            "by_source": {
                "shapenet": len(samples["shapenet"]),
                "objaverse": len(samples["objaverse"])
            },
            "by_category": {
                cat: sum(1 for s in sample_lookup.values() if s["category"] == cat)
                for cat in config.categories
            },
            "actual_ratio": {
                "shapenet": len(samples["shapenet"]) / max(len(sample_lookup), 1),
                "objaverse": len(samples["objaverse"]) / max(len(sample_lookup), 1)
            }
        },
        "splits": {
            "train": asdict(train_split),
            "val": asdict(val_split),
            "test": asdict(test_split)
        },
        "samples": sample_lookup
    }
    
    # Save metadata
    meta_path = output_dir / "dataset_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    
    # Save split files (for easy loading)
    splits_dir = output_dir / "splits"
    splits_dir.mkdir(exist_ok=True)
    
    for split in [train_split, val_split, test_split]:
        split_path = splits_dir / f"{split.name}.json"
        with open(split_path, "w") as f:
            json.dump(asdict(split), f, indent=2)
    
    # Print summary
    print("\n" + "=" * 50)
    print("Dataset Preparation Complete")
    print("=" * 50)
    print(f"\nTotal samples: {len(sample_lookup)}")
    print(f"\nBy source:")
    print(f"  ShapeNet:  {len(samples['shapenet']):4d} ({len(samples['shapenet'])/max(len(sample_lookup),1)*100:.1f}%)")
    print(f"  Objaverse: {len(samples['objaverse']):4d} ({len(samples['objaverse'])/max(len(sample_lookup),1)*100:.1f}%)")
    print(f"\nBy category:")
    for cat in config.categories:
        count = sum(1 for s in sample_lookup.values() if s["category"] == cat)
        print(f"  {cat}: {count}")
    print(f"\nSplits:")
    print(f"  Train: {len(train_split.samples)}")
    print(f"  Val:   {len(val_split.samples)}")
    print(f"  Test:  {len(test_split.samples)}")
    print(f"\nMetadata saved to: {meta_path}")
    
    return metadata


def load_dataset_metadata(dataset_dir: Path) -> Dict:
    """Load dataset metadata from prepared dataset."""
    meta_path = Path(dataset_dir) / "dataset_metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Dataset metadata not found: {meta_path}")
    
    with open(meta_path) as f:
        return json.load(f)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Prepare combined ShapeNet + Objaverse dataset"
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("data/combined"),
        help="Output directory"
    )
    parser.add_argument(
        "--shapenet_path",
        type=Path,
        default=None,
        help="Path to ShapeNetCore.v2"
    )
    parser.add_argument(
        "--total_samples",
        type=int,
        default=5000,
        help="Target total samples (baseline: 5000)"
    )
    parser.add_argument(
        "--objaverse_ratio",
        type=float,
        default=0.75,
        help="Objaverse ratio (default: 0.75)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    
    args = parser.parse_args()
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s"
    )
    
    config = CombinedDatasetConfig(
        objaverse_ratio=args.objaverse_ratio,
        shapenet_ratio=1.0 - args.objaverse_ratio,
        target_total=args.total_samples,
        seed=args.seed
    )
    
    prepare_combined_dataset(
        output_dir=args.output_dir,
        shapenet_path=args.shapenet_path,
        config=config
    )


if __name__ == "__main__":
    main()
