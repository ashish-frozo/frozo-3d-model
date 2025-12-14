"""
SV-SCN Trainer

Complete training loop per ML Training Spec:
- Optimizer: Adam
- Learning rate: 1e-3 → 1e-4 (cosine annealing)
- Batch size: 32
- Epochs: 150
- Loss: Chamfer Distance

Usage:
    from svscn.training import Trainer
    trainer = Trainer(model, train_loader, val_loader)
    trainer.train(epochs=150)
"""

import os
import json
import logging
import time
from pathlib import Path
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from ..config import default_config
from ..models.losses import chamfer_distance, ChamferLoss

logger = logging.getLogger(__name__)


@dataclass
class TrainingState:
    """Checkpoint state for resuming training."""
    
    epoch: int
    best_val_loss: float
    train_losses: List[float]
    val_losses: List[float]
    learning_rates: List[float]


@dataclass
class TrainingMetrics:
    """Metrics from a single epoch."""
    
    epoch: int
    train_loss: float
    val_loss: float
    learning_rate: float
    epoch_time_sec: float
    
    # Loss components
    train_chamfer: float = 0.0
    val_chamfer: float = 0.0
    train_symmetry: float = 0.0
    val_symmetry: float = 0.0


class Trainer:
    """
    SV-SCN Training Manager.
    
    Handles training loop, validation, checkpointing, and logging.
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Optional[object] = None,
        checkpoint_dir: Optional[Path] = None,
        log_dir: Optional[Path] = None,
        device: Optional[str] = None
    ):
        """
        Args:
            model: SV-SCN model instance
            train_loader: Training data loader
            val_loader: Validation data loader
            config: Training configuration
            checkpoint_dir: Where to save checkpoints
            log_dir: TensorBoard log directory
            device: cuda or cpu
        """
        self.config = config or default_config.training
        
        # Device setup
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        
        # Model
        self.model = model.to(self.device)
        
        # Data
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        # Directories
        self.checkpoint_dir = Path(checkpoint_dir or self.config.CHECKPOINT_DIR)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.log_dir = Path(log_dir or self.config.LOG_DIR)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Optimizer (per ML Training Spec: Adam)
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.config.LEARNING_RATE_INIT,
            weight_decay=self.config.WEIGHT_DECAY
        )
        
        # Learning rate scheduler (cosine annealing: 1e-3 → 1e-4)
        # With optional warmup
        main_scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=self.config.EPOCHS - self.config.WARMUP_EPOCHS,
            eta_min=self.config.LEARNING_RATE_FINAL
        )
        
        if self.config.WARMUP_EPOCHS > 0:
            warmup_scheduler = LinearLR(
                self.optimizer,
                start_factor=0.1,
                end_factor=1.0,
                total_iters=self.config.WARMUP_EPOCHS
            )
            self.scheduler = SequentialLR(
                self.optimizer,
                schedulers=[warmup_scheduler, main_scheduler],
                milestones=[self.config.WARMUP_EPOCHS]
            )
        else:
            self.scheduler = main_scheduler
        
        # Loss function
        self.loss_fn = ChamferLoss(symmetry_weight=0.0)
        
        # TensorBoard
        self.writer = SummaryWriter(log_dir=self.log_dir)
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.train_losses: List[float] = []
        self.val_losses: List[float] = []
        self.learning_rates: List[float] = []
        
        # Failure sample collection
        self.failure_samples: List[Dict] = []
    
    def train(
        self,
        epochs: Optional[int] = None,
        resume_from: Optional[Path] = None
    ) -> Dict:
        """
        Run training loop.
        
        Args:
            epochs: Number of epochs (default from config)
            resume_from: Checkpoint to resume from
        
        Returns:
            Training summary dict
        """
        epochs = epochs or self.config.EPOCHS
        
        # Resume if checkpoint provided
        if resume_from:
            self.load_checkpoint(resume_from)
        
        logger.info(f"Starting training for {epochs} epochs")
        logger.info(f"Device: {self.device}")
        logger.info(f"Train samples: {len(self.train_loader.dataset)}")
        logger.info(f"Val samples: {len(self.val_loader.dataset)}")
        
        start_epoch = self.current_epoch
        
        for epoch in range(start_epoch, epochs):
            self.current_epoch = epoch
            epoch_start = time.time()
            
            # Train
            train_loss, train_components = self._train_epoch()
            
            # Validate
            val_loss, val_components = self._validate()
            
            # Get current LR
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Step scheduler
            self.scheduler.step()
            
            # Record metrics
            epoch_time = time.time() - epoch_start
            
            metrics = TrainingMetrics(
                epoch=epoch,
                train_loss=train_loss,
                val_loss=val_loss,
                learning_rate=current_lr,
                epoch_time_sec=epoch_time,
                train_chamfer=train_components.get("chamfer", 0),
                val_chamfer=val_components.get("chamfer", 0)
            )
            
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.learning_rates.append(current_lr)
            
            # Log
            self._log_metrics(metrics)
            
            # Checkpoint
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self._save_best_checkpoint()
            
            if (epoch + 1) % self.config.SAVE_EVERY_N_EPOCHS == 0:
                self._save_checkpoint(f"epoch_{epoch:03d}")
            
            # Log progress
            logger.info(
                f"Epoch {epoch:3d}/{epochs} | "
                f"Train: {train_loss:.6f} | Val: {val_loss:.6f} | "
                f"LR: {current_lr:.2e} | Time: {epoch_time:.1f}s"
            )
        
        # Final save
        self._save_checkpoint("final")
        self._save_training_summary()
        
        self.writer.close()
        
        return {
            "best_val_loss": self.best_val_loss,
            "final_train_loss": self.train_losses[-1],
            "total_epochs": epochs
        }
    
    def _train_epoch(self) -> Tuple[float, Dict]:
        """Run one training epoch."""
        self.model.train()
        
        total_loss = 0.0
        total_components = {"chamfer": 0.0, "symmetry": 0.0}
        num_batches = 0
        
        for batch_idx, batch in enumerate(self.train_loader):
            # Move to device
            partial = batch["partial"].to(self.device)
            full = batch["full"].to(self.device)
            class_id = batch["class_id"].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            pred = self.model(partial, class_id)
            
            # Compute loss
            loss, components = self.loss_fn(pred, full)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if self.config.GRADIENT_CLIP_NORM > 0:
                nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.GRADIENT_CLIP_NORM
                )
            
            self.optimizer.step()
            
            # Accumulate
            total_loss += loss.item()
            for k, v in components.items():
                total_components[k] = total_components.get(k, 0) + v
            num_batches += 1
            
            # Log step
            if (batch_idx + 1) % self.config.LOG_EVERY_N_STEPS == 0:
                step = self.current_epoch * len(self.train_loader) + batch_idx
                self.writer.add_scalar("train/step_loss", loss.item(), step)
        
        avg_loss = total_loss / num_batches
        avg_components = {k: v / num_batches for k, v in total_components.items()}
        
        return avg_loss, avg_components
    
    @torch.no_grad()
    def _validate(self) -> Tuple[float, Dict]:
        """Run validation."""
        self.model.eval()
        
        total_loss = 0.0
        total_components = {"chamfer": 0.0, "symmetry": 0.0}
        num_batches = 0
        
        for batch in self.val_loader:
            partial = batch["partial"].to(self.device)
            full = batch["full"].to(self.device)
            class_id = batch["class_id"].to(self.device)
            
            # Forward
            pred = self.model(partial, class_id)
            
            # Loss
            loss, components = self.loss_fn(pred, full)
            
            total_loss += loss.item()
            for k, v in components.items():
                total_components[k] = total_components.get(k, 0) + v
            num_batches += 1
            
            # Collect failure samples
            if loss.item() > self.best_val_loss * 2:
                self.failure_samples.append({
                    "sample_ids": batch.get("sample_ids", []),
                    "loss": loss.item()
                })
        
        avg_loss = total_loss / max(num_batches, 1)
        avg_components = {k: v / max(num_batches, 1) for k, v in total_components.items()}
        
        return avg_loss, avg_components
    
    def _log_metrics(self, metrics: TrainingMetrics):
        """Log metrics to TensorBoard."""
        epoch = metrics.epoch
        
        self.writer.add_scalar("train/loss", metrics.train_loss, epoch)
        self.writer.add_scalar("val/loss", metrics.val_loss, epoch)
        self.writer.add_scalar("train/chamfer", metrics.train_chamfer, epoch)
        self.writer.add_scalar("val/chamfer", metrics.val_chamfer, epoch)
        self.writer.add_scalar("lr", metrics.learning_rate, epoch)
        self.writer.add_scalar("epoch_time", metrics.epoch_time_sec, epoch)
    
    def _save_checkpoint(self, name: str):
        """Save training checkpoint."""
        path = self.checkpoint_dir / f"{name}.pt"
        
        torch.save({
            "epoch": self.current_epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_val_loss": self.best_val_loss,
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "learning_rates": self.learning_rates,
            "config": asdict(self.config) if hasattr(self.config, '__dataclass_fields__') else {}
        }, path)
        
        logger.debug(f"Saved checkpoint: {path}")
        
        # Cleanup old checkpoints
        self._cleanup_checkpoints()
    
    def _save_best_checkpoint(self):
        """Save best model checkpoint."""
        path = self.checkpoint_dir / "best.pt"
        
        torch.save({
            "epoch": self.current_epoch,
            "model_state_dict": self.model.state_dict(),
            "best_val_loss": self.best_val_loss,
        }, path)
        
        logger.info(f"New best model saved (val_loss: {self.best_val_loss:.6f})")
    
    def _cleanup_checkpoints(self):
        """Remove old checkpoints, keeping only the last N."""
        keep = self.config.KEEP_LAST_N_CHECKPOINTS
        
        checkpoints = sorted(
            self.checkpoint_dir.glob("epoch_*.pt"),
            key=lambda p: p.stat().st_mtime
        )
        
        for ckpt in checkpoints[:-keep]:
            ckpt.unlink()
    
    def load_checkpoint(self, path: Path):
        """Load training checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        
        self.current_epoch = checkpoint["epoch"] + 1
        self.best_val_loss = checkpoint["best_val_loss"]
        self.train_losses = checkpoint.get("train_losses", [])
        self.val_losses = checkpoint.get("val_losses", [])
        self.learning_rates = checkpoint.get("learning_rates", [])
        
        logger.info(f"Resumed from epoch {self.current_epoch}")
    
    def _save_training_summary(self):
        """Save training summary JSON."""
        summary = {
            "total_epochs": self.current_epoch + 1,
            "best_val_loss": self.best_val_loss,
            "final_train_loss": self.train_losses[-1] if self.train_losses else 0,
            "final_val_loss": self.val_losses[-1] if self.val_losses else 0,
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "learning_rates": self.learning_rates,
            "failure_samples": self.failure_samples[:100],  # Limit
            "timestamp": datetime.now().isoformat()
        }
        
        path = self.checkpoint_dir / "training_summary.json"
        with open(path, "w") as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Training summary saved: {path}")


def train_model(
    data_dir: Path,
    checkpoint_dir: Optional[Path] = None,
    epochs: int = 150,
    batch_size: int = 32,
    device: Optional[str] = None
) -> Dict:
    """
    Convenience function to train SV-SCN.
    
    Args:
        data_dir: Path to processed dataset
        checkpoint_dir: Where to save checkpoints
        epochs: Number of training epochs
        batch_size: Batch size
        device: cuda or cpu
    
    Returns:
        Training summary
    """
    from ..models import SVSCN
    from ..data import create_data_loaders
    
    # Create model
    model = SVSCN(num_classes=3)
    
    # Create data loaders
    train_loader, val_loader, _ = create_data_loaders(
        data_dir,
        batch_size=batch_size
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        checkpoint_dir=checkpoint_dir,
        device=device
    )
    
    # Train
    return trainer.train(epochs=epochs)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train SV-SCN model")
    parser.add_argument("--data_dir", type=Path, required=True)
    parser.add_argument("--checkpoint_dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", type=str, default=None)
    
    args = parser.parse_args()
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s"
    )
    
    summary = train_model(
        data_dir=args.data_dir,
        checkpoint_dir=args.checkpoint_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        device=args.device
    )
    
    print(f"\nTraining complete!")
    print(f"Best validation loss: {summary['best_val_loss']:.6f}")
