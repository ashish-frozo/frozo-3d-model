"""
PointNet-Style Encoder for SV-SCN

Encodes partial point cloud into global feature vector.
Architecture (per ML Training Spec):
- SharedMLP: 3 → 64 → 128 → 256 → 512
- MaxPool across points
- Class conditioning via concatenation

Usage:
    from svscn.models import PointNetEncoder
    encoder = PointNetEncoder(num_classes=3)
    features = encoder(partial_points, class_ids)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from ..config import default_config


class SharedMLP(nn.Module):
    """
    Shared MLP applied to each point independently.
    
    Each layer: Linear → BatchNorm → ReLU
    """
    
    def __init__(
        self,
        channels: list,
        use_bn: bool = True,
        use_activation: bool = True
    ):
        super().__init__()
        
        layers = []
        for i in range(len(channels) - 1):
            layers.append(nn.Conv1d(channels[i], channels[i+1], 1))
            if use_bn:
                layers.append(nn.BatchNorm1d(channels[i+1]))
            if use_activation and i < len(channels) - 2:  # No activation on last
                layers.append(nn.ReLU(inplace=True))
        
        # Final activation
        if use_activation:
            layers.append(nn.ReLU(inplace=True))
        
        self.layers = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, N) point features
        
        Returns:
            (B, C', N) transformed features
        """
        return self.layers(x)


class PointNetEncoder(nn.Module):
    """
    PointNet-style encoder with class conditioning.
    
    Extracts global feature from partial point cloud and conditions
    on object class (chair/stool/table) to learn class-specific priors.
    """
    
    def __init__(
        self,
        num_classes: int = 3,
        input_dim: int = 3,
        encoder_dims: list = None,
        global_feature_dim: int = 512,
        class_embed_dim: int = 64
    ):
        """
        Args:
            num_classes: Number of object classes
            input_dim: Input point dimension (3 for xyz)
            encoder_dims: MLP dimensions [64, 128, 256, 512]
            global_feature_dim: Output feature dimension
            class_embed_dim: Class embedding dimension
        """
        super().__init__()
        
        if encoder_dims is None:
            encoder_dims = default_config.model.ENCODER_DIMS
        
        self.num_classes = num_classes
        self.global_feature_dim = global_feature_dim
        self.class_embed_dim = class_embed_dim
        
        # Point feature extraction
        # SharedMLP: 3 → 64 → 128 → 256 → 512
        all_dims = [input_dim] + encoder_dims
        self.point_mlp = SharedMLP(all_dims)
        
        # Class embedding
        self.class_embed = nn.Embedding(num_classes, class_embed_dim)
        
        # Feature fusion (global + class → final)
        fusion_input_dim = encoder_dims[-1] + class_embed_dim
        self.fusion_mlp = nn.Sequential(
            nn.Linear(fusion_input_dim, global_feature_dim),
            nn.BatchNorm1d(global_feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(global_feature_dim, global_feature_dim),
            nn.BatchNorm1d(global_feature_dim),
            nn.ReLU(inplace=True)
        )
    
    def forward(
        self,
        points: torch.Tensor,
        class_ids: torch.Tensor
    ) -> torch.Tensor:
        """
        Encode partial point cloud with class conditioning.
        
        Args:
            points: (B, N, 3) partial point cloud
            class_ids: (B,) class indices
        
        Returns:
            (B, global_feature_dim) class-conditioned global feature
        """
        batch_size = points.shape[0]
        
        # Transpose for Conv1d: (B, N, 3) → (B, 3, N)
        x = points.transpose(1, 2).contiguous()
        
        # Extract point features
        x = self.point_mlp(x)  # (B, 512, N)
        
        # Global max pooling
        global_feat = x.max(dim=2)[0]  # (B, 512)
        
        # Get class embedding
        class_feat = self.class_embed(class_ids)  # (B, class_embed_dim)
        
        # Fuse global and class features
        fused = torch.cat([global_feat, class_feat], dim=1)  # (B, 512+64)
        
        # Final feature transformation
        output = self.fusion_mlp(fused)  # (B, global_feature_dim)
        
        return output
    
    def encode_points_only(self, points: torch.Tensor) -> torch.Tensor:
        """
        Encode points without class conditioning.
        Useful for inference when class is unknown.
        """
        x = points.transpose(1, 2).contiguous()
        x = self.point_mlp(x)
        global_feat = x.max(dim=2)[0]
        return global_feat


class TNet(nn.Module):
    """
    Transformation Network (T-Net) for input/feature alignment.
    
    Optional module that learns a transformation matrix to
    canonicalize the input point cloud orientation.
    """
    
    def __init__(self, k: int = 3):
        super().__init__()
        
        self.k = k
        
        self.conv = nn.Sequential(
            nn.Conv1d(k, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, 1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
        )
        
        self.fc = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Linear(128, k * k)
        )
        
        # Initialize to identity
        self.fc[-1].weight.data.zero_()
        self.fc[-1].bias.data.copy_(torch.eye(k).view(-1))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, k, N) input features
        
        Returns:
            (B, k, k) transformation matrix
        """
        batch_size = x.shape[0]
        
        x = self.conv(x)
        x = x.max(dim=2)[0]
        
        transform = self.fc(x)
        transform = transform.view(batch_size, self.k, self.k)
        
        return transform


class PointNetEncoderWithTNet(PointNetEncoder):
    """
    PointNet encoder with input transformation network.
    
    Adds a T-Net to learn canonical orientation, which can help
    with rotation invariance.
    """
    
    def __init__(self, *args, use_tnet: bool = True, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.use_tnet = use_tnet
        if use_tnet:
            self.tnet = TNet(k=3)
    
    def forward(
        self,
        points: torch.Tensor,
        class_ids: torch.Tensor
    ) -> torch.Tensor:
        batch_size = points.shape[0]
        
        # Apply input transformation
        if self.use_tnet:
            x = points.transpose(1, 2).contiguous()
            transform = self.tnet(x)
            points = torch.bmm(points, transform)
        
        # Continue with standard encoding
        return super().forward(points, class_ids)


if __name__ == "__main__":
    # Test encoder
    encoder = PointNetEncoder(num_classes=3)
    
    batch_size = 4
    num_points = 2048
    
    points = torch.randn(batch_size, num_points, 3)
    class_ids = torch.randint(0, 3, (batch_size,))
    
    features = encoder(points, class_ids)
    
    print(f"Input shape: {points.shape}")
    print(f"Output shape: {features.shape}")
    print(f"Expected: ({batch_size}, {encoder.global_feature_dim})")
    
    # Test with T-Net
    encoder_tnet = PointNetEncoderWithTNet(num_classes=3, use_tnet=True)
    features_tnet = encoder_tnet(points, class_ids)
    print(f"With T-Net output shape: {features_tnet.shape}")
