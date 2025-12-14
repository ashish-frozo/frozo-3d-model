"""
SV-SCN: Single-View Shape Completion Network

Complete model combining PointNet encoder and FoldingNet decoder.
Per ML Training Spec:
- Input: 2048 point partial cloud + class ID
- Output: 8192 point complete cloud

Usage:
    from svscn.models import SVSCN
    model = SVSCN(num_classes=3)
    completed = model(partial_points, class_ids)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple

from .encoder import PointNetEncoder
from .decoder import FoldingNetDecoder, SimpleFoldingDecoder
from ..config import default_config


class SVSCN(nn.Module):
    """
    Single-View Shape Completion Network.
    
    Reconstructs complete 3D point clouds from partial observations,
    conditioned on object class (chair/stool/table).
    
    Architecture:
        partial_cloud + class_id
            ↓
        PointNetEncoder → global_feature (512-dim)
            ↓
        FoldingNetDecoder → complete_cloud (8192 points)
    """
    
    def __init__(
        self,
        num_classes: int = 3,
        input_points: int = 2048,
        output_points: int = 8192,
        global_feature_dim: int = 512,
        class_embed_dim: int = 64,
        encoder_dims: list = None,
        use_simple_decoder: bool = False
    ):
        """
        Args:
            num_classes: Number of object classes
            input_points: Expected partial cloud size
            output_points: Output complete cloud size
            global_feature_dim: Encoder output dimension
            class_embed_dim: Class embedding dimension
            encoder_dims: Encoder MLP dimensions
            use_simple_decoder: Use simplified single-stage decoder
        """
        super().__init__()
        
        self.num_classes = num_classes
        self.input_points = input_points
        self.output_points = output_points
        
        # Encoder
        self.encoder = PointNetEncoder(
            num_classes=num_classes,
            input_dim=3,
            encoder_dims=encoder_dims,
            global_feature_dim=global_feature_dim,
            class_embed_dim=class_embed_dim
        )
        
        # Decoder
        if use_simple_decoder:
            self.decoder = SimpleFoldingDecoder(
                global_feature_dim=global_feature_dim,
                output_points=output_points
            )
        else:
            self.decoder = FoldingNetDecoder(
                global_feature_dim=global_feature_dim,
                output_points=output_points
            )
    
    def forward(
        self,
        partial: torch.Tensor,
        class_id: torch.Tensor
    ) -> torch.Tensor:
        """
        Complete a partial point cloud.
        
        Args:
            partial: (B, N, 3) partial point cloud
            class_id: (B,) class indices
        
        Returns:
            (B, output_points, 3) completed point cloud
        """
        # Encode
        features = self.encoder(partial, class_id)  # (B, 512)
        
        # Decode
        completed = self.decoder(features)  # (B, output_points, 3)
        
        return completed
    
    def encode(
        self,
        partial: torch.Tensor,
        class_id: torch.Tensor
    ) -> torch.Tensor:
        """Get latent representation only."""
        return self.encoder(partial, class_id)
    
    def decode(self, features: torch.Tensor) -> torch.Tensor:
        """Decode from latent representation."""
        return self.decoder(features)


class SVSCNWithConfidence(nn.Module):
    """
    SV-SCN with confidence estimation head.
    
    Per ML Training Spec:
    - confidence < 0.6 → trigger fallback
    """
    
    def __init__(
        self,
        num_classes: int = 3,
        **kwargs
    ):
        super().__init__()
        
        # Main model
        self.svscn = SVSCN(num_classes=num_classes, **kwargs)
        
        # Confidence head
        self.confidence_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            nn.Sigmoid()  # Output in [0, 1]
        )
    
    def forward(
        self,
        partial: torch.Tensor,
        class_id: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Complete with confidence score.
        
        Returns:
            (completed, confidence) tuple
        """
        # Get features
        features = self.svscn.encoder(partial, class_id)
        
        # Generate completion
        completed = self.svscn.decoder(features)
        
        # Estimate confidence
        confidence = self.confidence_head(features).squeeze(-1)
        
        return completed, confidence
    
    def predict_with_fallback(
        self,
        partial: torch.Tensor,
        class_id: torch.Tensor,
        confidence_threshold: float = 0.6
    ) -> Dict[str, torch.Tensor]:
        """
        Predict with fallback for low-confidence samples.
        
        Returns dict with:
            - completed: completed point cloud
            - confidence: confidence scores
            - used_fallback: boolean mask
        """
        completed, confidence = self.forward(partial, class_id)
        
        # Identify low-confidence samples
        low_conf_mask = confidence < confidence_threshold
        
        # Apply symmetry fallback for low confidence
        if low_conf_mask.any():
            fallback = self._symmetry_fallback(partial)
            completed[low_conf_mask] = fallback[low_conf_mask]
        
        return {
            "completed": completed,
            "confidence": confidence,
            "used_fallback": low_conf_mask
        }
    
    def _symmetry_fallback(
        self,
        partial: torch.Tensor
    ) -> torch.Tensor:
        """
        Simple symmetry-based completion fallback.
        
        Mirrors the partial cloud across the X axis.
        """
        # Mirror across X
        mirrored = partial.clone()
        mirrored[:, :, 0] = -mirrored[:, :, 0]
        
        # Combine original and mirrored
        combined = torch.cat([partial, mirrored], dim=1)
        
        # Subsample to target size
        batch_size = partial.shape[0]
        n = combined.shape[1]
        target = self.svscn.output_points
        
        if n > target:
            indices = torch.randperm(n)[:target]
            combined = combined[:, indices, :]
        elif n < target:
            shortage = target - n
            extra_indices = torch.randint(0, n, (shortage,))
            extra = combined[:, extra_indices, :] + torch.randn(batch_size, shortage, 3, device=partial.device) * 0.01
            combined = torch.cat([combined, extra], dim=1)
        
        return combined


def create_model(
    config: Optional[object] = None,
    with_confidence: bool = False
) -> nn.Module:
    """
    Factory function to create SV-SCN model.
    
    Args:
        config: Configuration object
        with_confidence: Include confidence estimation
    
    Returns:
        Model instance
    """
    if config is None:
        config = default_config
    
    kwargs = {
        "num_classes": config.model.NUM_CLASSES,
        "input_points": config.model.INPUT_POINTS,
        "output_points": config.model.OUTPUT_POINTS,
        "global_feature_dim": config.model.GLOBAL_FEATURE_DIM,
        "class_embed_dim": config.model.CLASS_EMBED_DIM,
        "encoder_dims": config.model.ENCODER_DIMS,
    }
    
    if with_confidence:
        return SVSCNWithConfidence(**kwargs)
    else:
        return SVSCN(**kwargs)


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test model
    model = SVSCN(num_classes=3)
    
    batch_size = 4
    num_input = 2048
    
    partial = torch.randn(batch_size, num_input, 3)
    class_ids = torch.randint(0, 3, (batch_size,))
    
    # Forward pass
    completed = model(partial, class_ids)
    
    print(f"Model: SVSCN")
    print(f"Parameters: {count_parameters(model):,}")
    print(f"Input shape: {partial.shape}")
    print(f"Output shape: {completed.shape}")
    print(f"Expected output: ({batch_size}, {model.output_points}, 3)")
    
    # Test with confidence
    model_conf = SVSCNWithConfidence(num_classes=3)
    completed, confidence = model_conf(partial, class_ids)
    
    print(f"\nModel with confidence: SVSCNWithConfidence")
    print(f"Parameters: {count_parameters(model_conf):,}")
    print(f"Completed shape: {completed.shape}")
    print(f"Confidence shape: {confidence.shape}")
    print(f"Confidence values: {confidence}")
