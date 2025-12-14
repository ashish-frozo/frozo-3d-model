#!/usr/bin/env python
"""
SV-SCN Training Script

Command-line interface for training the SV-SCN model.

Usage:
    python scripts/train.py --data_dir data/processed --epochs 150

Full options:
    python scripts/train.py \
        --data_dir data/processed \
        --checkpoint_dir checkpoints \
        --log_dir logs \
        --epochs 150 \
        --batch_size 32 \
        --lr 1e-3 \
        --device cuda
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch

from svscn.config import default_config
from svscn.models import SVSCN
from svscn.data import create_data_loaders
from svscn.training import Trainer

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train SV-SCN model for point cloud completion",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Data
    parser.add_argument(
        "--data_dir",
        type=Path,
        required=True,
        help="Path to processed dataset"
    )
    
    # Output
    parser.add_argument(
        "--checkpoint_dir",
        type=Path,
        default=Path("checkpoints"),
        help="Directory for saving checkpoints"
    )
    parser.add_argument(
        "--log_dir",
        type=Path,
        default=Path("logs"),
        help="TensorBoard log directory"
    )
    parser.add_argument(
        "--exp_name",
        type=str,
        default=None,
        help="Experiment name (auto-generated if not provided)"
    )
    
    # Training
    parser.add_argument(
        "--epochs",
        type=int,
        default=150,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Initial learning rate"
    )
    parser.add_argument(
        "--lr_min",
        type=float,
        default=1e-4,
        help="Minimum learning rate"
    )
    parser.add_argument(
        "--warmup_epochs",
        type=int,
        default=5,
        help="Number of warmup epochs"
    )
    
    # Model
    parser.add_argument(
        "--num_classes",
        type=int,
        default=3,
        help="Number of object classes"
    )
    parser.add_argument(
        "--input_points",
        type=int,
        default=2048,
        help="Input partial cloud size"
    )
    parser.add_argument(
        "--output_points",
        type=int,
        default=8192,
        help="Output complete cloud size"
    )
    
    # Misc
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device (cuda/cpu, auto-detect if not set)"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="DataLoader workers"
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Resume from checkpoint"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose logging"
    )
    
    return parser.parse_args()


def setup_experiment(args):
    """Setup experiment directories and config."""
    
    # Generate experiment name if not provided
    if args.exp_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.exp_name = f"svscn_{timestamp}"
    
    # Create directories
    exp_checkpoint_dir = args.checkpoint_dir / args.exp_name
    exp_log_dir = args.log_dir / args.exp_name
    
    exp_checkpoint_dir.mkdir(parents=True, exist_ok=True)
    exp_log_dir.mkdir(parents=True, exist_ok=True)
    
    # Update config
    config = default_config.training
    config.EPOCHS = args.epochs
    config.BATCH_SIZE = args.batch_size
    config.LEARNING_RATE_INIT = args.lr
    config.LEARNING_RATE_FINAL = args.lr_min
    config.WARMUP_EPOCHS = args.warmup_epochs
    config.CHECKPOINT_DIR = exp_checkpoint_dir
    config.LOG_DIR = exp_log_dir
    
    return config, exp_checkpoint_dir, exp_log_dir


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    import random
    import numpy as np
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    args = parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(args.checkpoint_dir / "train.log")
        ]
    )
    
    # Set seed
    set_seed(args.seed)
    
    # Setup experiment
    config, checkpoint_dir, log_dir = setup_experiment(args)
    
    # Device
    if args.device:
        device = args.device
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    logger.info("=" * 60)
    logger.info("SV-SCN Training")
    logger.info("=" * 60)
    logger.info(f"Experiment: {args.exp_name}")
    logger.info(f"Device: {device}")
    logger.info(f"Data: {args.data_dir}")
    logger.info(f"Checkpoints: {checkpoint_dir}")
    logger.info(f"Logs: {log_dir}")
    logger.info("")
    
    # Check data directory
    if not args.data_dir.exists():
        logger.error(f"Data directory not found: {args.data_dir}")
        logger.info("Run data preparation first:")
        logger.info("  python -m svscn.data.dataset_manager --prepare --output_dir data/combined")
        return 1
    
    # Create model
    logger.info("Creating model...")
    model = SVSCN(
        num_classes=args.num_classes,
        input_points=args.input_points,
        output_points=args.output_points
    )
    
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {num_params:,}")
    
    # Create data loaders
    logger.info("Loading data...")
    try:
        train_loader, val_loader, test_loader = create_data_loaders(
            args.data_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers
        )
        logger.info(f"Train batches: {len(train_loader)}")
        logger.info(f"Val batches: {len(val_loader)}")
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        logger.info("Make sure data is prepared correctly")
        return 1
    
    # Create trainer
    logger.info("Initializing trainer...")
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        checkpoint_dir=checkpoint_dir,
        log_dir=log_dir,
        device=device
    )
    
    # Resume if specified
    if args.resume:
        if args.resume.exists():
            logger.info(f"Resuming from: {args.resume}")
            trainer.load_checkpoint(args.resume)
        else:
            logger.warning(f"Checkpoint not found: {args.resume}")
    
    # Train
    logger.info("")
    logger.info("Starting training...")
    logger.info(f"Epochs: {args.epochs}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Learning rate: {args.lr} → {args.lr_min}")
    logger.info("")
    
    try:
        summary = trainer.train(epochs=args.epochs)
        
        logger.info("")
        logger.info("=" * 60)
        logger.info("Training Complete!")
        logger.info("=" * 60)
        logger.info(f"Best validation loss: {summary['best_val_loss']:.6f}")
        logger.info(f"Final training loss: {summary['final_train_loss']:.6f}")
        logger.info(f"Checkpoints saved to: {checkpoint_dir}")
        
        return 0
        
    except KeyboardInterrupt:
        logger.info("\nTraining interrupted by user")
        logger.info("Saving checkpoint...")
        trainer._save_checkpoint("interrupted")
        return 1
    except Exception as e:
        logger.exception(f"Training failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
