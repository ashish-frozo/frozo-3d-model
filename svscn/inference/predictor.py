"""
SV-SCN Inference Pipeline

Complete inference with:
- Model loading
- Confidence estimation  
- Fallback handling
- Batch processing

Usage:
    from svscn.inference import Predictor
    predictor = Predictor.load("checkpoints/best.pt")
    result = predictor.predict(partial_points, class_id)
"""

import logging
from pathlib import Path
from typing import Optional, Dict, List, Union, Tuple
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from ..config import default_config
from ..models import SVSCN
from ..models.losses import chamfer_distance

logger = logging.getLogger(__name__)


@dataclass
class PredictionResult:
    """Result from a single prediction."""
    
    # Point clouds
    completed: np.ndarray  # (N, 3) completed point cloud
    partial: np.ndarray    # Input partial cloud
    
    # Metadata
    class_id: int
    class_name: str
    
    # Confidence
    confidence: float
    used_fallback: bool
    
    # Quality metrics
    reconstruction_error: float = 0.0


class Predictor:
    """
    SV-SCN inference wrapper.
    
    Handles model loading, preprocessing, confidence estimation,
    and fallback logic.
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: Optional[str] = None,
        confidence_threshold: float = 0.6
    ):
        """
        Args:
            model: Loaded SV-SCN model
            device: cuda or cpu
            confidence_threshold: Below this → use fallback
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.model.eval()
        
        self.confidence_threshold = confidence_threshold
        
        # Class mapping
        self.class_names = ["chair", "stool", "table"]
        
        # Calibration for confidence (learned from validation)
        self.error_scale = 0.1  # Scale factor for error → confidence
    
    @classmethod
    def load(
        cls,
        checkpoint_path: Union[str, Path],
        device: Optional[str] = None
    ) -> "Predictor":
        """
        Load predictor from checkpoint.
        
        Args:
            checkpoint_path: Path to .pt checkpoint
            device: cuda or cpu
        
        Returns:
            Predictor instance
        """
        checkpoint_path = Path(checkpoint_path)
        
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Extract model config from checkpoint or use defaults
        num_classes = checkpoint.get("num_classes", 3)
        input_points = checkpoint.get("input_points", 2048)
        output_points = checkpoint.get("output_points", 8192)
        
        # Create model
        model = SVSCN(
            num_classes=num_classes,
            input_points=input_points,
            output_points=output_points
        )
        
        # Load weights
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        model.load_state_dict(state_dict)
        
        logger.info(f"Loaded model from {checkpoint_path}")
        
        return cls(model, device)
    
    @torch.no_grad()
    def predict(
        self,
        partial: np.ndarray,
        class_id: int,
        use_fallback: bool = True
    ) -> PredictionResult:
        """
        Predict complete point cloud from partial.
        
        Args:
            partial: (N, 3) partial point cloud
            class_id: Object class (0=chair, 1=stool, 2=table)
            use_fallback: Whether to use fallback for low confidence
        
        Returns:
            PredictionResult with completed cloud and metrics
        """
        # Preprocess
        partial_tensor = self._preprocess(partial)
        class_tensor = torch.tensor([class_id], device=self.device)
        
        # Forward pass
        completed_tensor = self.model(partial_tensor, class_tensor)
        
        # Compute confidence
        confidence, recon_error = self._estimate_confidence(
            partial_tensor, completed_tensor
        )
        
        # Apply fallback if needed
        used_fallback = False
        if use_fallback and confidence < self.confidence_threshold:
            logger.info(f"Low confidence ({confidence:.2f}), using fallback")
            completed_tensor = self._symmetry_fallback(partial_tensor)
            used_fallback = True
        
        # Convert to numpy
        completed = completed_tensor.cpu().numpy()[0]
        
        return PredictionResult(
            completed=completed,
            partial=partial,
            class_id=class_id,
            class_name=self.class_names[class_id],
            confidence=confidence,
            used_fallback=used_fallback,
            reconstruction_error=recon_error
        )
    
    @torch.no_grad()
    def predict_batch(
        self,
        partials: List[np.ndarray],
        class_ids: List[int]
    ) -> List[PredictionResult]:
        """
        Batch prediction for multiple samples.
        """
        results = []
        
        # Stack inputs
        batch_partial = torch.stack([
            self._preprocess(p).squeeze(0) for p in partials
        ])
        batch_class = torch.tensor(class_ids, device=self.device)
        
        # Forward
        batch_completed = self.model(batch_partial, batch_class)
        
        # Process each
        for i, (partial, class_id) in enumerate(zip(partials, class_ids)):
            completed = batch_completed[i:i+1]
            partial_t = batch_partial[i:i+1]
            
            confidence, error = self._estimate_confidence(partial_t, completed)
            
            used_fallback = False
            if confidence < self.confidence_threshold:
                completed = self._symmetry_fallback(partial_t)
                used_fallback = True
            
            results.append(PredictionResult(
                completed=completed.cpu().numpy()[0],
                partial=partial,
                class_id=class_id,
                class_name=self.class_names[class_id],
                confidence=confidence,
                used_fallback=used_fallback,
                reconstruction_error=error
            ))
        
        return results
    
    def _preprocess(self, points: np.ndarray) -> torch.Tensor:
        """Preprocess input points."""
        # Ensure correct shape
        if points.ndim == 2:
            points = points[np.newaxis, ...]  # Add batch dim
        
        # Convert to tensor
        tensor = torch.from_numpy(points.astype(np.float32))
        tensor = tensor.to(self.device)
        
        # Ensure correct number of points
        target = self.model.input_points
        n = tensor.shape[1]
        
        if n > target:
            indices = torch.randperm(n)[:target]
            tensor = tensor[:, indices, :]
        elif n < target:
            shortage = target - n
            extra_indices = torch.randint(0, n, (shortage,))
            extra = tensor[:, extra_indices, :] + torch.randn(1, shortage, 3, device=self.device) * 0.001
            tensor = torch.cat([tensor, extra], dim=1)
        
        return tensor
    
    def _estimate_confidence(
        self,
        partial: torch.Tensor,
        completed: torch.Tensor
    ) -> Tuple[float, float]:
        """
        Estimate prediction confidence.
        
        Based on reconstruction error between input partial
        and corresponding region in completed cloud.
        
        Returns:
            (confidence 0-1, reconstruction_error)
        """
        # Sample points from completed that should match partial
        # (Using Chamfer distance as proxy)
        
        # Compute distance from partial to completed
        # Lower = partial is well-represented in completed
        error = chamfer_distance(partial, completed[:, :partial.shape[1], :])
        error = error.item()
        
        # Convert error to confidence
        # confidence = 1 / (1 + error * scale)
        confidence = 1.0 / (1.0 + error * self.error_scale)
        confidence = max(0.0, min(1.0, confidence))
        
        return confidence, error
    
    def _symmetry_fallback(self, partial: torch.Tensor) -> torch.Tensor:
        """
        Symmetry-based fallback completion.
        
        Mirrors partial cloud across X axis and combines.
        """
        # Mirror across X
        mirrored = partial.clone()
        mirrored[:, :, 0] = -mirrored[:, :, 0]
        
        # Combine
        combined = torch.cat([partial, mirrored], dim=1)
        
        # Subsample to output size
        target = self.model.output_points
        n = combined.shape[1]
        
        if n > target:
            indices = torch.randperm(n)[:target]
            combined = combined[:, indices, :]
        elif n < target:
            shortage = target - n
            extra_indices = torch.randint(0, n, (shortage,), device=self.device)
            extra = combined[:, extra_indices, :] + torch.randn(1, shortage, 3, device=self.device) * 0.01
            combined = torch.cat([combined, extra], dim=1)
        
        return combined
    
    def calibrate_confidence(
        self,
        val_loader,
        target_accuracy: float = 0.9
    ):
        """
        Calibrate confidence threshold using validation data.
        
        Adjusts threshold so that `target_accuracy` of predictions
        above threshold are actually good.
        """
        errors = []
        
        for batch in val_loader:
            partial = batch["partial"].to(self.device)
            full = batch["full"].to(self.device)
            class_id = batch["class_id"].to(self.device)
            
            pred = self.model(partial, class_id)
            
            # Compute per-sample error
            batch_errors = chamfer_distance(pred, full, reduce="none")
            errors.extend(batch_errors.cpu().numpy().tolist())
        
        # Find error threshold for target accuracy
        errors = np.array(errors)
        threshold_error = np.percentile(errors, target_accuracy * 100)
        
        # Update scale factor
        # confidence = 1 / (1 + error * scale)
        # We want confidence = 0.6 at threshold_error
        # 0.6 = 1 / (1 + threshold_error * scale)
        # scale = (1/0.6 - 1) / threshold_error
        self.error_scale = (1.0 / 0.6 - 1) / max(threshold_error, 1e-6)
        
        logger.info(f"Calibrated confidence: error_scale = {self.error_scale:.2f}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run SV-SCN inference")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--class_id", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("output.npy"))
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    
    # Load predictor
    predictor = Predictor.load(args.checkpoint)
    
    # Load input
    partial = np.load(args.input)
    
    # Predict
    result = predictor.predict(partial, args.class_id)
    
    print(f"Class: {result.class_name}")
    print(f"Confidence: {result.confidence:.2f}")
    print(f"Used fallback: {result.used_fallback}")
    print(f"Output shape: {result.completed.shape}")
    
    # Save
    np.save(args.output, result.completed)
    print(f"Saved to: {args.output}")
