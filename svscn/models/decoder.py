"""
FoldingNet-Style Decoder for SV-SCN

Decodes global feature into completed point cloud.
Architecture (per ML Training Spec):
- Generate 2D grid (91x91 → sample to 8192)
- Tile global feature
- MLP: 514 → 512 → 512 → 256 → 3

The decoder "folds" a 2D template into the target 3D shape.

Usage:
    from svscn.models import FoldingNetDecoder
    decoder = FoldingNetDecoder()
    points = decoder(features)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from ..config import default_config


class FoldingLayer(nn.Module):
    """
    Single folding layer that transforms 2D + feature to 3D.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: list,
        output_dim: int = 3
    ):
        super().__init__()
        
        dims = [input_dim] + hidden_dims + [output_dim]
        
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Conv1d(dims[i], dims[i+1], 1))
            if i < len(dims) - 2:  # No activation on output
                layers.append(nn.BatchNorm1d(dims[i+1]))
                layers.append(nn.ReLU(inplace=True))
        
        self.layers = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, N) input features
        
        Returns:
            (B, 3, N) 3D point offsets
        """
        return self.layers(x)


class FoldingNetDecoder(nn.Module):
    """
    FoldingNet-style decoder for point cloud generation.
    
    Generates points by:
    1. Creating a 2D grid template
    2. Concatenating with global feature
    3. Folding the template into 3D shape via MLP
    """
    
    def __init__(
        self,
        global_feature_dim: int = 512,
        output_points: int = 8192,
        grid_size: int = 91,  # 91x91 = 8281 points
        decoder_dims: list = None
    ):
        """
        Args:
            global_feature_dim: Input feature dimension
            output_points: Number of output points
            grid_size: 2D grid size (grid_size^2 ≈ output_points)
            decoder_dims: Hidden MLP dimensions
        """
        super().__init__()
        
        if decoder_dims is None:
            decoder_dims = [512, 512, 256]
        
        self.global_feature_dim = global_feature_dim
        self.output_points = output_points
        self.grid_size = grid_size
        
        # Number of grid points
        self.num_grid_points = grid_size * grid_size
        
        # Create fixed 2D grid template (will be sampled)
        self.register_buffer("grid_template", self._create_grid())
        
        # Folding layers
        # Input: global_feature (512) + grid_coords (2) = 514
        input_dim = global_feature_dim + 2
        
        # First folding: 2D → coarse 3D
        self.fold1 = FoldingLayer(
            input_dim=input_dim,
            hidden_dims=[512, 512, 256],
            output_dim=3
        )
        
        # Second folding: refine 3D
        # Input: feature (512) + coarse 3D (3) = 515
        self.fold2 = FoldingLayer(
            input_dim=global_feature_dim + 3,
            hidden_dims=[512, 256, 128],
            output_dim=3
        )
    
    def _create_grid(self) -> torch.Tensor:
        """Create 2D grid template in [-1, 1] range."""
        # Create grid
        x = torch.linspace(-1, 1, self.grid_size)
        y = torch.linspace(-1, 1, self.grid_size)
        
        # Meshgrid
        grid_x, grid_y = torch.meshgrid(x, y, indexing='ij')
        
        # Stack and flatten: (grid_size^2, 2)
        grid = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=1)
        
        return grid
    
    def _sample_grid(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Sample grid points to target output size."""
        grid = self.grid_template.to(device)
        
        if self.num_grid_points > self.output_points:
            # Subsample
            indices = torch.randperm(self.num_grid_points)[:self.output_points]
            grid = grid[indices]
        elif self.num_grid_points < self.output_points:
            # Oversample with jitter
            shortage = self.output_points - self.num_grid_points
            extra_indices = torch.randint(0, self.num_grid_points, (shortage,))
            extra = grid[extra_indices] + torch.randn(shortage, 2, device=device) * 0.01
            grid = torch.cat([grid, extra], dim=0)
        
        # Expand for batch: (N, 2) → (B, N, 2)
        grid = grid.unsqueeze(0).expand(batch_size, -1, -1)
        
        return grid
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Decode global feature to point cloud.
        
        Args:
            features: (B, global_feature_dim) global features
        
        Returns:
            (B, output_points, 3) generated point cloud
        """
        batch_size = features.shape[0]
        device = features.device
        
        # Get grid
        grid = self._sample_grid(batch_size, device)  # (B, N, 2)
        num_points = grid.shape[1]
        
        # Tile features: (B, C) → (B, N, C)
        features_tiled = features.unsqueeze(1).expand(-1, num_points, -1)
        
        # Concat features + grid: (B, N, C+2)
        x = torch.cat([features_tiled, grid], dim=2)
        
        # Transpose for Conv1d: (B, N, C+2) → (B, C+2, N)
        x = x.transpose(1, 2).contiguous()
        
        # First folding: 2D → coarse 3D
        coarse = self.fold1(x)  # (B, 3, N)
        
        # Second folding: refine
        # Concat features with coarse output
        features_t = features.unsqueeze(2).expand(-1, -1, num_points)  # (B, C, N)
        refine_input = torch.cat([features_t, coarse], dim=1)  # (B, C+3, N)
        
        refined = self.fold2(refine_input)  # (B, 3, N)
        
        # Add residual from coarse
        output = coarse + refined
        
        # Transpose back: (B, 3, N) → (B, N, 3)
        output = output.transpose(1, 2).contiguous()
        
        return output


class SimpleFoldingDecoder(nn.Module):
    """
    Simplified single-stage folding decoder.
    
    Faster training, slightly less refined output.
    """
    
    def __init__(
        self,
        global_feature_dim: int = 512,
        output_points: int = 8192,
        hidden_dims: list = None
    ):
        super().__init__()
        
        if hidden_dims is None:
            hidden_dims = [512, 512, 256]
        
        self.global_feature_dim = global_feature_dim
        self.output_points = output_points
        
        # Use square grid closest to output_points
        self.grid_size = int(math.ceil(math.sqrt(output_points)))
        
        # Create grid
        self.register_buffer("grid", self._create_grid())
        
        # Single MLP
        dims = [global_feature_dim + 2] + hidden_dims + [3]
        
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            if i < len(dims) - 2:
                layers.append(nn.ReLU(inplace=True))
        
        self.mlp = nn.Sequential(*layers)
    
    def _create_grid(self) -> torch.Tensor:
        x = torch.linspace(-1, 1, self.grid_size)
        y = torch.linspace(-1, 1, self.grid_size)
        grid_x, grid_y = torch.meshgrid(x, y, indexing='ij')
        grid = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=1)
        return grid[:self.output_points]  # Trim to exact size
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        batch_size = features.shape[0]
        device = features.device
        
        # Get grid
        grid = self.grid.to(device)
        num_points = grid.shape[0]
        
        # Pad if needed
        if num_points < self.output_points:
            pad = self.output_points - num_points
            extra = grid[:pad] + torch.randn(pad, 2, device=device) * 0.01
            grid = torch.cat([grid, extra], dim=0)
            num_points = self.output_points
        
        # Expand grid for batch
        grid = grid.unsqueeze(0).expand(batch_size, -1, -1)  # (B, N, 2)
        
        # Tile features
        features_tiled = features.unsqueeze(1).expand(-1, num_points, -1)  # (B, N, C)
        
        # Concat and transform
        x = torch.cat([features_tiled, grid], dim=2)  # (B, N, C+2)
        
        output = self.mlp(x)  # (B, N, 3)
        
        return output


if __name__ == "__main__":
    # Test decoder
    decoder = FoldingNetDecoder()
    
    batch_size = 4
    feature_dim = 512
    
    features = torch.randn(batch_size, feature_dim)
    
    points = decoder(features)
    
    print(f"Input shape: {features.shape}")
    print(f"Output shape: {points.shape}")
    print(f"Expected: ({batch_size}, {decoder.output_points}, 3)")
    
    # Test simple decoder
    simple_decoder = SimpleFoldingDecoder()
    simple_points = simple_decoder(features)
    print(f"Simple decoder output: {simple_points.shape}")
