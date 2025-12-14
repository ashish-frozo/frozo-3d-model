"""
Unit tests for SV-SCN pipeline.

Run with: pytest tests/ -v
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestConfig:
    """Test configuration loading."""
    
    def test_default_config_exists(self):
        from svscn.config import default_config
        
        assert default_config is not None
        assert default_config.model.INPUT_POINTS == 2048
        assert default_config.model.OUTPUT_POINTS == 8192
        assert default_config.training.EPOCHS == 150
    
    def test_config_classes(self):
        from svscn.config import default_config
        
        assert len(default_config.data.FURNITURE_CLASSES) == 3
        assert "chair" in default_config.data.FURNITURE_CLASSES


class TestEncoder:
    """Test PointNet encoder."""
    
    def test_encoder_forward(self):
        from svscn.models.encoder import PointNetEncoder
        
        encoder = PointNetEncoder(num_classes=3)
        
        batch_size = 4
        num_points = 2048
        
        points = torch.randn(batch_size, num_points, 3)
        class_ids = torch.randint(0, 3, (batch_size,))
        
        features = encoder(points, class_ids)
        
        assert features.shape == (batch_size, 512)
    
    def test_encoder_deterministic(self):
        from svscn.models.encoder import PointNetEncoder
        
        encoder = PointNetEncoder(num_classes=3)
        encoder.eval()
        
        points = torch.randn(2, 2048, 3)
        class_ids = torch.tensor([0, 1])
        
        with torch.no_grad():
            feat1 = encoder(points, class_ids)
            feat2 = encoder(points, class_ids)
        
        assert torch.allclose(feat1, feat2)


class TestDecoder:
    """Test FoldingNet decoder."""
    
    def test_decoder_forward(self):
        from svscn.models.decoder import FoldingNetDecoder
        
        decoder = FoldingNetDecoder()
        
        batch_size = 4
        features = torch.randn(batch_size, 512)
        
        points = decoder(features)
        
        assert points.shape == (batch_size, 8192, 3)
    
    def test_simple_decoder(self):
        from svscn.models.decoder import SimpleFoldingDecoder
        
        decoder = SimpleFoldingDecoder()
        
        features = torch.randn(2, 512)
        points = decoder(features)
        
        assert points.shape == (2, 8192, 3)


class TestSVSCN:
    """Test complete model."""
    
    def test_svscn_forward(self):
        from svscn.models import SVSCN
        
        model = SVSCN(num_classes=3)
        
        batch_size = 4
        partial = torch.randn(batch_size, 2048, 3)
        class_ids = torch.randint(0, 3, (batch_size,))
        
        completed = model(partial, class_ids)
        
        assert completed.shape == (batch_size, 8192, 3)
    
    def test_svscn_parameter_count(self):
        from svscn.models import SVSCN
        
        model = SVSCN(num_classes=3)
        
        num_params = sum(p.numel() for p in model.parameters())
        
        # Should be reasonable size (not huge)
        assert num_params > 100000  # At least 100k params
        assert num_params < 50000000  # Less than 50M params


class TestLosses:
    """Test loss functions."""
    
    def test_chamfer_distance(self):
        from svscn.models.losses import chamfer_distance
        
        pred = torch.randn(4, 8192, 3)
        target = torch.randn(4, 8192, 3)
        
        loss = chamfer_distance(pred, target)
        
        assert loss.ndim == 0  # Scalar
        assert loss.item() > 0
    
    def test_chamfer_identical(self):
        from svscn.models.losses import chamfer_distance
        
        points = torch.randn(2, 8192, 3)
        
        loss = chamfer_distance(points, points)
        
        # Should be approximately 0 for identical clouds
        assert loss.item() < 1e-5
    
    def test_symmetry_loss(self):
        from svscn.models.losses import symmetry_loss
        
        # Create symmetric point cloud
        points = torch.randn(2, 1000, 3)
        points = torch.cat([points, -points[:, :, 0:1]], dim=2)[:, :, :3]
        
        # Make it symmetric
        mirrored = points.clone()
        mirrored[:, :, 0] = -mirrored[:, :, 0]
        symmetric = torch.cat([points[:, :500, :], mirrored[:, :500, :]], dim=1)
        
        # Non-symmetric
        asymmetric = torch.randn(2, 1000, 3)
        
        sym_loss = symmetry_loss(symmetric)
        asym_loss = symmetry_loss(asymmetric)
        
        # Symmetric should have lower loss
        assert sym_loss.item() < asym_loss.item()


class TestPreprocessing:
    """Test data preprocessing."""
    
    def test_normalize_points(self):
        from svscn.data.preprocess import PointCloudPreprocessor
        
        processor = PointCloudPreprocessor()
        
        # Create unnormalized points
        points = np.random.randn(10000, 3) * 10 + 5
        
        # Process
        processed = processor._preprocess_pipeline(points, 8192)
        
        assert processed is not None
        assert len(processed) == 8192
        
        # Should be normalized to unit cube
        assert processed.max() <= 1.0
        assert processed.min() >= -1.0


class TestPartialViewGeneration:
    """Test partial view generation."""
    
    def test_generate_partial(self):
        from svscn.data.augment import PartialViewGenerator
        
        generator = PartialViewGenerator()
        
        # Create full point cloud (unit sphere)
        theta = np.random.uniform(0, 2*np.pi, 8192)
        phi = np.random.uniform(0, np.pi, 8192)
        r = 0.5
        
        full = np.stack([
            r * np.sin(phi) * np.cos(theta),
            r * np.sin(phi) * np.sin(theta),
            r * np.cos(phi)
        ], axis=1).astype(np.float32)
        
        partial = generator.generate_partial(full)
        
        assert partial.shape == (2048, 3)
    
    def test_multiple_views(self):
        from svscn.data.augment import PartialViewGenerator
        
        generator = PartialViewGenerator()
        
        full = np.random.randn(8192, 3).astype(np.float32)
        
        views = generator.generate_multi_view(full, num_views=3)
        
        assert len(views) == 3
        for view, meta in views:
            assert view.shape == (2048, 3)
            assert "elevation" in meta
            assert "azimuth" in meta


class TestDataset:
    """Test PyTorch dataset."""
    
    def test_dataset_creation(self):
        from svscn.data.dataset import CLASS_TO_ID, ID_TO_CLASS
        
        assert CLASS_TO_ID["chair"] == 0
        assert CLASS_TO_ID["stool"] == 1
        assert CLASS_TO_ID["table"] == 2
        
        assert ID_TO_CLASS[0] == "chair"


class TestInference:
    """Test inference pipeline."""
    
    def test_predictor_predict(self):
        from svscn.models import SVSCN
        from svscn.inference import Predictor
        
        model = SVSCN(num_classes=3)
        predictor = Predictor(model, device="cpu")
        
        partial = np.random.randn(2048, 3).astype(np.float32)
        
        result = predictor.predict(partial, class_id=0)
        
        assert result.completed.shape == (8192, 3)
        assert 0.0 <= result.confidence <= 1.0
        assert result.class_name == "chair"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
