"""
PyTorch Dataset for SV-SCN Training

Loads (partial, full, class) training tuples.
Handles train/val/test splits per ML Training Spec.

Usage:
    from svscn.data import FurnitureDataset, create_data_loaders
    train_loader, val_loader, test_loader = create_data_loaders("data/processed")
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from ..config import default_config
from .augment import augment_rotation, augment_jitter, augment_scale

logger = logging.getLogger(__name__)


# Class mapping
CLASS_TO_ID = {
    "chair": 0,
    "stool": 1,
    "table": 2,
}

ID_TO_CLASS = {v: k for k, v in CLASS_TO_ID.items()}


class FurnitureDataset(Dataset):
    """
    PyTorch Dataset for furniture point cloud completion.
    
    Loads training pairs of (partial, full, class_id).
    """
    
    def __init__(
        self,
        data_dir: Path,
        split: str = "train",
        num_input_points: int = 2048,
        num_output_points: int = 8192,
        augment: bool = True,
        class_filter: Optional[List[str]] = None
    ):
        """
        Args:
            data_dir: Base data directory (contains splits/, partial/, full/)
            split: One of "train", "val", "test"
            num_input_points: Partial cloud size
            num_output_points: Full cloud size
            augment: Whether to apply augmentation (train only)
            class_filter: Optional list of classes to include
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.num_input_points = num_input_points
        self.num_output_points = num_output_points
        self.augment = augment and (split == "train")
        self.class_filter = set(class_filter) if class_filter else None
        
        self.samples = self._load_samples()
        
        logger.info(f"Loaded {len(self.samples)} {split} samples")
    
    def _load_samples(self) -> List[Dict]:
        """Load sample metadata from split file."""
        samples = []
        
        # Try loading from split file
        split_path = self.data_dir / "splits" / f"{self.split}.json"
        
        if split_path.exists():
            with open(split_path) as f:
                split_data = json.load(f)
            
            # Load full metadata
            meta_path = self.data_dir / "dataset_metadata.json"
            with open(meta_path) as f:
                metadata = json.load(f)
            
            sample_lookup = metadata.get("samples", {})
            
            for sample_id in split_data.get("samples", []):
                sample_info = sample_lookup.get(sample_id, {})
                category = sample_info.get("category", "chair")
                
                if self.class_filter and category not in self.class_filter:
                    continue
                
                samples.append({
                    "id": sample_id,
                    "category": category,
                    "class_id": CLASS_TO_ID.get(category, 0),
                    "source": sample_info.get("source", "unknown")
                })
        else:
            # Fallback: scan directories
            samples = self._scan_directories()
        
        return samples
    
    def _scan_directories(self) -> List[Dict]:
        """Scan data directories if no split file exists."""
        samples = []
        
        partial_dir = self.data_dir / "partial"
        full_dir = self.data_dir / "full"
        
        if not partial_dir.exists():
            # Try alternative structure: category subdirs
            for category in CLASS_TO_ID.keys():
                cat_dir = self.data_dir / category
                if cat_dir.exists():
                    for npy_file in cat_dir.glob("*_partial.npy"):
                        sample_id = npy_file.stem.replace("_partial", "")
                        
                        if self.class_filter and category not in self.class_filter:
                            continue
                        
                        samples.append({
                            "id": sample_id,
                            "category": category,
                            "class_id": CLASS_TO_ID[category],
                            "partial_path": str(npy_file),
                            "full_path": str(npy_file.with_name(f"{sample_id}_full.npy"))
                        })
        else:
            for npy_file in partial_dir.glob("*_partial.npy"):
                sample_id = npy_file.stem.replace("_partial", "")
                
                # Try to infer category from filename
                category = "chair"  # default
                for cat in CLASS_TO_ID.keys():
                    if cat in sample_id.lower():
                        category = cat
                        break
                
                if self.class_filter and category not in self.class_filter:
                    continue
                
                samples.append({
                    "id": sample_id,
                    "category": category,
                    "class_id": CLASS_TO_ID.get(category, 0),
                    "partial_path": str(npy_file),
                    "full_path": str(full_dir / f"{sample_id}_full.npy")
                })
        
        return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        
        # Load point clouds
        partial = self._load_points(sample, "partial")
        full = self._load_points(sample, "full")
        
        # Apply augmentation
        if self.augment:
            partial, full = self._apply_augmentation(partial, full)
        
        # Ensure correct sizes
        partial = self._ensure_size(partial, self.num_input_points)
        full = self._ensure_size(full, self.num_output_points)
        
        # Create one-hot class encoding
        num_classes = len(CLASS_TO_ID)
        class_onehot = torch.zeros(num_classes)
        class_onehot[sample["class_id"]] = 1.0
        
        return {
            "partial": torch.from_numpy(partial).float(),
            "full": torch.from_numpy(full).float(),
            "class_id": torch.tensor(sample["class_id"]).long(),
            "class_onehot": class_onehot,
            "sample_id": sample["id"]
        }
    
    def _load_points(self, sample: Dict, point_type: str) -> np.ndarray:
        """Load point cloud from file."""
        # Try direct path first
        if f"{point_type}_path" in sample:
            path = Path(sample[f"{point_type}_path"])
            if path.exists():
                return np.load(path)
        
        # Try standard structure
        path = self.data_dir / point_type / f"{sample['id']}_{point_type}.npy"
        if path.exists():
            return np.load(path)
        
        # Try category structure
        path = self.data_dir / sample["category"] / f"{sample['id']}_{point_type}.npy"
        if path.exists():
            return np.load(path)
        
        # Fallback: generate random points (for testing)
        logger.warning(f"Could not find {point_type} for {sample['id']}, using random")
        size = self.num_input_points if point_type == "partial" else self.num_output_points
        return np.random.randn(size, 3).astype(np.float32) * 0.3
    
    def _apply_augmentation(
        self,
        partial: np.ndarray,
        full: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply consistent augmentation to both point clouds."""
        # Random rotation (same for both)
        angle = np.random.uniform(0, 2 * np.pi)
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        rotation = np.array([
            [cos_a, 0, sin_a],
            [0, 1, 0],
            [-sin_a, 0, cos_a]
        ])
        
        partial = partial @ rotation.T
        full = full @ rotation.T
        
        # Random scale (same for both)
        scale = np.random.uniform(0.9, 1.1)
        partial = partial * scale
        full = full * scale
        
        # Jitter (independent)
        partial = partial + np.random.randn(*partial.shape) * 0.005
        full = full + np.random.randn(*full.shape) * 0.002
        
        return partial.astype(np.float32), full.astype(np.float32)
    
    def _ensure_size(self, points: np.ndarray, target: int) -> np.ndarray:
        """Ensure point cloud has exact target size."""
        n = len(points)
        
        if n == target:
            return points
        elif n > target:
            indices = np.random.choice(n, target, replace=False)
            return points[indices]
        else:
            shortage = target - n
            extra_indices = np.random.choice(n, shortage, replace=True)
            extra = points[extra_indices] + np.random.randn(shortage, 3) * 0.001
            return np.vstack([points, extra]).astype(np.float32)


def create_data_loaders(
    data_dir: Path,
    batch_size: int = 32,
    num_workers: int = 4,
    config: Optional[object] = None
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, val, test data loaders.
    
    Args:
        data_dir: Base data directory
        batch_size: Batch size
        num_workers: DataLoader workers
        config: Optional config object
    
    Returns:
        (train_loader, val_loader, test_loader)
    """
    if config is None:
        config = default_config
    
    # Create datasets
    train_dataset = FurnitureDataset(
        data_dir,
        split="train",
        num_input_points=config.model.INPUT_POINTS,
        num_output_points=config.model.OUTPUT_POINTS,
        augment=True
    )
    
    val_dataset = FurnitureDataset(
        data_dir,
        split="val",
        num_input_points=config.model.INPUT_POINTS,
        num_output_points=config.model.OUTPUT_POINTS,
        augment=False
    )
    
    test_dataset = FurnitureDataset(
        data_dir,
        split="test",
        num_input_points=config.model.INPUT_POINTS,
        num_output_points=config.model.OUTPUT_POINTS,
        augment=False
    )
    
    # Create loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Custom collate function for point cloud batches."""
    return {
        "partial": torch.stack([b["partial"] for b in batch]),
        "full": torch.stack([b["full"] for b in batch]),
        "class_id": torch.stack([b["class_id"] for b in batch]),
        "class_onehot": torch.stack([b["class_onehot"] for b in batch]),
        "sample_ids": [b["sample_id"] for b in batch]
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test dataset loading")
    parser.add_argument("--data_dir", type=Path, required=True)
    parser.add_argument("--split", type=str, default="train")
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    
    dataset = FurnitureDataset(args.data_dir, split=args.split)
    
    print(f"Dataset size: {len(dataset)}")
    
    if len(dataset) > 0:
        sample = dataset[0]
        print(f"Sample keys: {sample.keys()}")
        print(f"Partial shape: {sample['partial'].shape}")
        print(f"Full shape: {sample['full'].shape}")
        print(f"Class ID: {sample['class_id']}")
        print(f"Class onehot: {sample['class_onehot']}")
