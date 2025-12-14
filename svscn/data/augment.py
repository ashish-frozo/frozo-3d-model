"""
Partial View Generator

Generates partial point clouds from complete ones, simulating single-view capture.
Per ML Training Spec:
- Camera elevations: 15°, 30°, 45°
- Azimuth: random
- Hidden point removal
- Random point dropout (≤5%)

Usage:
    from svscn.data.augment import PartialViewGenerator
    generator = PartialViewGenerator()
    partial = generator.generate_partial(full_points)
"""

import logging
from pathlib import Path
from typing import Tuple, Optional, List
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class PartialViewConfig:
    """Configuration for partial view generation."""
    
    # Camera settings (per ML Training Spec)
    camera_elevations: List[int] = None  # degrees
    azimuth_range: Tuple[int, int] = (0, 360)
    
    # Point cloud sizes
    input_points: int = 8192   # Full cloud
    output_points: int = 2048  # Partial cloud
    
    # Hidden point removal
    camera_distance: float = 2.0  # Distance from origin
    hpr_radius: float = 100.0     # HPR radius parameter
    
    # Augmentation
    random_dropout_max: float = 0.05  # ≤5%
    add_noise: bool = True
    noise_std: float = 0.005
    
    # Random rotation augmentation
    random_rotation: bool = True
    
    def __post_init__(self):
        if self.camera_elevations is None:
            self.camera_elevations = [15, 30, 45]


class PartialViewGenerator:
    """
    Generates partial point clouds simulating single-view depth capture.
    
    Uses Hidden Point Removal (HPR) algorithm to determine visibility
    from a given camera viewpoint.
    """
    
    def __init__(self, config: Optional[PartialViewConfig] = None):
        self.config = config or PartialViewConfig()
    
    def generate_partial(
        self,
        full_points: np.ndarray,
        elevation: Optional[float] = None,
        azimuth: Optional[float] = None,
        return_camera: bool = False
    ) -> np.ndarray:
        """
        Generate partial point cloud from full cloud.
        
        Args:
            full_points: (N, 3) full point cloud
            elevation: Camera elevation in degrees (random if None)
            azimuth: Camera azimuth in degrees (random if None)
            return_camera: Whether to return camera position
        
        Returns:
            (M, 3) partial point cloud, or tuple with camera if return_camera=True
        """
        # Random camera position if not specified
        if elevation is None:
            elevation = np.random.choice(self.config.camera_elevations)
        if azimuth is None:
            azimuth = np.random.uniform(*self.config.azimuth_range)
        
        # Convert to radians
        elev_rad = np.radians(elevation)
        azim_rad = np.radians(azimuth)
        
        # Compute camera position
        distance = self.config.camera_distance
        camera = np.array([
            distance * np.cos(elev_rad) * np.cos(azim_rad),
            distance * np.cos(elev_rad) * np.sin(azim_rad),
            distance * np.sin(elev_rad)
        ])
        
        # Apply Hidden Point Removal
        visible_indices = self._hidden_point_removal(full_points, camera)
        partial_points = full_points[visible_indices].copy()
        
        # Apply random dropout
        partial_points = self._apply_dropout(partial_points)
        
        # Add noise
        if self.config.add_noise:
            partial_points = self._add_noise(partial_points)
        
        # Resample to target size
        partial_points = self._resample(partial_points, self.config.output_points)
        
        if return_camera:
            return partial_points, camera
        return partial_points
    
    def _hidden_point_removal(
        self,
        points: np.ndarray,
        camera: np.ndarray
    ) -> np.ndarray:
        """
        Hidden Point Removal algorithm.
        
        Determines which points are visible from the camera position.
        Based on "Direct Visibility of Point Sets" (Katz et al., 2007)
        """
        # Translate points so camera is at origin
        translated = points - camera
        
        # Compute distances
        distances = np.linalg.norm(translated, axis=1)
        
        # Spherical flip transformation
        R = self.config.hpr_radius
        
        # Avoid division by zero
        distances = np.maximum(distances, 1e-10)
        
        # Flip points to sphere
        flipped = translated + 2 * (R - distances[:, np.newaxis]) * (translated / distances[:, np.newaxis])
        
        # Add camera (origin) to flipped points
        flipped_with_cam = np.vstack([[[0, 0, 0]], flipped])
        
        # Compute convex hull
        try:
            from scipy.spatial import ConvexHull
            hull = ConvexHull(flipped_with_cam)
            
            # Visible points are on the hull (excluding camera vertex 0)
            hull_vertices = set(hull.vertices)
            hull_vertices.discard(0)  # Remove camera
            
            visible_indices = np.array([v - 1 for v in hull_vertices if v > 0])
            
            return visible_indices
            
        except Exception as e:
            logger.warning(f"Convex hull failed: {e}, using fallback")
            return self._visibility_fallback(points, camera)
    
    def _visibility_fallback(
        self,
        points: np.ndarray,
        camera: np.ndarray
    ) -> np.ndarray:
        """
        Fallback visibility estimation using simple dot product.
        
        Points facing the camera are considered visible.
        """
        # Estimate normals (approximate using local neighbors)
        from scipy.spatial import cKDTree
        
        tree = cKDTree(points)
        
        visible = []
        
        for i, point in enumerate(points):
            # Get local neighbors
            _, idx = tree.query(point, k=10)
            neighbors = points[idx]
            
            # Estimate normal via PCA
            centered = neighbors - neighbors.mean(axis=0)
            try:
                _, _, Vt = np.linalg.svd(centered)
                normal = Vt[-1]  # Smallest singular vector
            except:
                normal = np.array([0, 0, 1])
            
            # Check if facing camera
            to_camera = camera - point
            to_camera = to_camera / (np.linalg.norm(to_camera) + 1e-10)
            
            # Make normal face camera
            if np.dot(normal, to_camera) < 0:
                normal = -normal
            
            # Visibility based on dot product
            facing = np.dot(normal, to_camera)
            
            if facing > 0.1:  # Threshold for visibility
                visible.append(i)
        
        return np.array(visible)
    
    def _apply_dropout(self, points: np.ndarray) -> np.ndarray:
        """Apply random point dropout (≤5%)."""
        if self.config.random_dropout_max <= 0:
            return points
        
        dropout_ratio = np.random.uniform(0, self.config.random_dropout_max)
        keep_ratio = 1.0 - dropout_ratio
        
        n_keep = max(int(len(points) * keep_ratio), 1)
        indices = np.random.choice(len(points), n_keep, replace=False)
        
        return points[indices]
    
    def _add_noise(self, points: np.ndarray) -> np.ndarray:
        """Add Gaussian noise to simulate sensor noise."""
        noise = np.random.randn(*points.shape) * self.config.noise_std
        return points + noise
    
    def _resample(self, points: np.ndarray, target: int) -> np.ndarray:
        """Resample to target number of points."""
        n = len(points)
        
        if n >= target:
            indices = np.random.choice(n, target, replace=False)
            return points[indices].astype(np.float32)
        else:
            # Upsample with jitter
            shortage = target - n
            extra_indices = np.random.choice(n, shortage, replace=True)
            extra_points = points[extra_indices] + np.random.randn(shortage, 3) * 0.001
            combined = np.vstack([points, extra_points])
            return combined.astype(np.float32)
    
    def generate_multi_view(
        self,
        full_points: np.ndarray,
        num_views: int = 3
    ) -> List[Tuple[np.ndarray, dict]]:
        """
        Generate multiple partial views of the same object.
        
        Useful for data augmentation during training.
        
        Returns:
            List of (partial_points, metadata) tuples
        """
        views = []
        
        for _ in range(num_views):
            elevation = np.random.choice(self.config.camera_elevations)
            azimuth = np.random.uniform(*self.config.azimuth_range)
            
            partial, camera = self.generate_partial(
                full_points,
                elevation=elevation,
                azimuth=azimuth,
                return_camera=True
            )
            
            metadata = {
                "elevation": elevation,
                "azimuth": azimuth,
                "camera": camera.tolist(),
                "num_points": len(partial)
            }
            
            views.append((partial, metadata))
        
        return views


def augment_rotation(points: np.ndarray) -> np.ndarray:
    """Apply random rotation augmentation."""
    # Random rotation around Y axis (up)
    angle = np.random.uniform(0, 2 * np.pi)
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    
    rotation = np.array([
        [cos_a, 0, sin_a],
        [0, 1, 0],
        [-sin_a, 0, cos_a]
    ])
    
    return points @ rotation.T


def augment_jitter(points: np.ndarray, sigma: float = 0.01) -> np.ndarray:
    """Add jitter augmentation."""
    noise = np.random.randn(*points.shape) * sigma
    return points + noise


def augment_scale(points: np.ndarray, scale_range: Tuple[float, float] = (0.8, 1.2)) -> np.ndarray:
    """Apply random scale augmentation."""
    scale = np.random.uniform(*scale_range)
    return points * scale


def generate_training_pair(
    full_points: np.ndarray,
    generator: Optional[PartialViewGenerator] = None,
    apply_augmentation: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a (partial, full) training pair.
    
    Args:
        full_points: Complete point cloud
        generator: PartialViewGenerator instance
        apply_augmentation: Whether to apply augmentation
    
    Returns:
        (partial_points, full_points) tuple
    """
    if generator is None:
        generator = PartialViewGenerator()
    
    # Apply same augmentation to both
    if apply_augmentation:
        full_points = augment_rotation(full_points.copy())
        full_points = augment_jitter(full_points)
        full_points = augment_scale(full_points)
    
    # Generate partial view
    partial_points = generator.generate_partial(full_points)
    
    return partial_points, full_points


def process_to_training_data(
    full_clouds_dir: Path,
    output_dir: Path,
    views_per_object: int = 3
) -> int:
    """
    Process full point clouds to training pairs.
    
    Args:
        full_clouds_dir: Directory containing full .npy files
        output_dir: Where to save training pairs
        views_per_object: Number of partial views per object
    
    Returns:
        Number of training pairs created
    """
    full_clouds_dir = Path(full_clouds_dir)
    output_dir = Path(output_dir)
    
    partial_dir = output_dir / "partial"
    full_dir = output_dir / "full"
    partial_dir.mkdir(parents=True, exist_ok=True)
    full_dir.mkdir(parents=True, exist_ok=True)
    
    generator = PartialViewGenerator()
    
    npy_files = list(full_clouds_dir.rglob("*.npy"))
    pair_count = 0
    
    for npy_path in npy_files:
        full_points = np.load(npy_path)
        
        # Get relative path for organizing
        rel_path = npy_path.relative_to(full_clouds_dir)
        stem = rel_path.stem
        
        for view_idx in range(views_per_object):
            partial, full = generate_training_pair(full_points, generator)
            
            pair_name = f"{stem}_v{view_idx}"
            
            np.save(partial_dir / f"{pair_name}_partial.npy", partial)
            np.save(full_dir / f"{pair_name}_full.npy", full)
            
            pair_count += 1
    
    logger.info(f"Created {pair_count} training pairs")
    return pair_count


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate training pairs")
    parser.add_argument("--input_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--views", type=int, default=3)
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    
    process_to_training_data(args.input_dir, args.output_dir, args.views)
