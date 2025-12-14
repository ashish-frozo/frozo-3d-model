"""
Loss Functions for SV-SCN

Implements:
- Chamfer Distance (primary) - per ML Training Spec
- Symmetry Loss (optional)
- Earth Mover's Distance (optional, more expensive)

Usage:
    from svscn.models import chamfer_distance, symmetry_loss
    loss = chamfer_distance(pred, target)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


def pairwise_distances(
    x: torch.Tensor,
    y: torch.Tensor
) -> torch.Tensor:
    """
    Compute pairwise L2 distances between two point sets.
    
    Args:
        x: (B, N, 3) first point set
        y: (B, M, 3) second point set
    
    Returns:
        (B, N, M) pairwise distance matrix
    """
    # Using the formula: ||x - y||^2 = ||x||^2 + ||y||^2 - 2*x·y
    
    # (B, N)
    xx = torch.sum(x ** 2, dim=2)
    # (B, M)
    yy = torch.sum(y ** 2, dim=2)
    # (B, N, M)
    xy = torch.bmm(x, y.transpose(1, 2))
    
    # (B, N, M)
    distances = xx.unsqueeze(2) + yy.unsqueeze(1) - 2 * xy
    
    # Clamp for numerical stability (avoid negative due to float errors)
    distances = torch.clamp(distances, min=0.0)
    
    return distances


def chamfer_distance(
    pred: torch.Tensor,
    target: torch.Tensor,
    reduce: str = "mean"
) -> torch.Tensor:
    """
    Compute Chamfer Distance between predicted and target point clouds.
    
    Chamfer Distance = mean(min_y ||x - y||^2 for x in pred) +
                       mean(min_x ||x - y||^2 for y in target)
    
    Args:
        pred: (B, N, 3) predicted point cloud
        target: (B, M, 3) target point cloud
        reduce: "mean", "sum", or "none"
    
    Returns:
        Chamfer distance (scalar if reduce="mean" or "sum", else (B,))
    """
    # Compute pairwise squared distances
    dist_matrix = pairwise_distances(pred, target)  # (B, N, M)
    
    # For each point in pred, find min distance to target
    min_pred_to_target = dist_matrix.min(dim=2)[0]  # (B, N)
    
    # For each point in target, find min distance to pred
    min_target_to_pred = dist_matrix.min(dim=1)[0]  # (B, M)
    
    # Chamfer per sample
    chamfer_per_sample = min_pred_to_target.mean(dim=1) + min_target_to_pred.mean(dim=1)  # (B,)
    
    if reduce == "mean":
        return chamfer_per_sample.mean()
    elif reduce == "sum":
        return chamfer_per_sample.sum()
    else:
        return chamfer_per_sample


def chamfer_distance_l1(
    pred: torch.Tensor,
    target: torch.Tensor,
    reduce: str = "mean"
) -> torch.Tensor:
    """
    Chamfer Distance using L1 norm instead of L2.
    
    Can be more robust to outliers.
    """
    dist_matrix = pairwise_distances(pred, target)
    dist_matrix = torch.sqrt(dist_matrix + 1e-8)  # L1 via sqrt of L2
    
    min_pred_to_target = dist_matrix.min(dim=2)[0]
    min_target_to_pred = dist_matrix.min(dim=1)[0]
    
    chamfer_per_sample = min_pred_to_target.mean(dim=1) + min_target_to_pred.mean(dim=1)
    
    if reduce == "mean":
        return chamfer_per_sample.mean()
    elif reduce == "sum":
        return chamfer_per_sample.sum()
    else:
        return chamfer_per_sample


def symmetry_loss(
    points: torch.Tensor,
    axis: int = 0,
    reduce: str = "mean"
) -> torch.Tensor:
    """
    Symmetry loss to encourage left-right symmetry.
    
    Useful for furniture like chairs and tables which are often symmetric.
    
    Args:
        points: (B, N, 3) point cloud
        axis: Axis for symmetry (0=x, 1=y, 2=z)
        reduce: "mean" or "sum"
    
    Returns:
        Symmetry loss
    """
    # Mirror points across axis
    mirrored = points.clone()
    mirrored[:, :, axis] = -mirrored[:, :, axis]
    
    # Compute Chamfer between original and mirrored
    # Lower = more symmetric
    sym_loss = chamfer_distance(points, mirrored, reduce=reduce)
    
    return sym_loss


def density_loss(
    points: torch.Tensor,
    k: int = 10,
    reduce: str = "mean"
) -> torch.Tensor:
    """
    Density uniformity loss.
    
    Encourages uniform point distribution by penalizing
    variance in local densities.
    
    Args:
        points: (B, N, 3) point cloud
        k: Number of neighbors for density estimation
        reduce: "mean" or "sum"
    """
    from scipy.spatial import cKDTree
    
    # This is expensive and not differentiable
    # For training, we use a differentiable approximation
    
    batch_size, num_points, _ = points.shape
    
    losses = []
    for b in range(batch_size):
        pts = points[b].detach().cpu().numpy()
        tree = cKDTree(pts)
        
        # Get k nearest neighbors for each point
        distances, _ = tree.query(pts, k=k+1)  # +1 to exclude self
        
        # Local density = 1 / mean_distance
        mean_dist = distances[:, 1:].mean(axis=1)  # Exclude self
        
        # Penalize variance in densities
        density_var = mean_dist.var()
        losses.append(density_var)
    
    loss = torch.tensor(losses, device=points.device)
    
    if reduce == "mean":
        return loss.mean()
    return loss.sum()


class ChamferLoss(nn.Module):
    """
    Chamfer Distance loss module.
    
    Wrapper for use in training pipelines.
    """
    
    def __init__(
        self,
        use_l1: bool = False,
        symmetry_weight: float = 0.0,
        symmetry_axis: int = 0
    ):
        super().__init__()
        
        self.use_l1 = use_l1
        self.symmetry_weight = symmetry_weight
        self.symmetry_axis = symmetry_axis
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute loss with optional symmetry term.
        
        Returns:
            (total_loss, loss_dict with components)
        """
        # Main Chamfer loss
        if self.use_l1:
            cd_loss = chamfer_distance_l1(pred, target)
        else:
            cd_loss = chamfer_distance(pred, target)
        
        total_loss = cd_loss
        loss_dict = {"chamfer": cd_loss.item()}
        
        # Optional symmetry loss
        if self.symmetry_weight > 0:
            sym_loss = symmetry_loss(pred, axis=self.symmetry_axis)
            total_loss = total_loss + self.symmetry_weight * sym_loss
            loss_dict["symmetry"] = sym_loss.item()
        
        loss_dict["total"] = total_loss.item()
        
        return total_loss, loss_dict


def coverage_ratio(
    pred: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.02
) -> torch.Tensor:
    """
    Compute coverage ratio (what fraction of target is covered by pred).
    
    Useful for evaluation.
    
    Args:
        pred: (B, N, 3) predicted
        target: (B, M, 3) target
        threshold: Distance threshold for coverage
    
    Returns:
        (B,) coverage ratio per sample
    """
    dist_matrix = pairwise_distances(pred, target)
    dist_matrix = torch.sqrt(dist_matrix + 1e-8)
    
    # For each target point, min distance to pred
    min_dist = dist_matrix.min(dim=1)[0]  # (B, M)
    
    # Fraction of target points within threshold
    covered = (min_dist < threshold).float()
    coverage = covered.mean(dim=1)
    
    return coverage


if __name__ == "__main__":
    # Test losses
    batch_size = 4
    num_pred = 8192
    num_target = 8192
    
    pred = torch.randn(batch_size, num_pred, 3)
    target = torch.randn(batch_size, num_target, 3)
    
    # Test Chamfer
    cd = chamfer_distance(pred, target)
    print(f"Chamfer Distance: {cd.item():.6f}")
    
    # Test L1 Chamfer
    cd_l1 = chamfer_distance_l1(pred, target)
    print(f"Chamfer L1: {cd_l1.item():.6f}")
    
    # Test symmetry
    sym = symmetry_loss(pred, axis=0)
    print(f"Symmetry Loss: {sym.item():.6f}")
    
    # Test with identical pointsы
    identical = chamfer_distance(pred, pred)
    print(f"Chamfer (identical): {identical.item():.6f}")  # Should be ~0
    
    # Test coverage
    cov = coverage_ratio(pred, target)
    print(f"Coverage: {cov.mean().item():.4f}")
    
    # Test loss module
    loss_fn = ChamferLoss(symmetry_weight=0.1)
    total, components = loss_fn(pred, target)
    print(f"Total loss: {total.item():.6f}")
    print(f"Components: {components}")
