"""
Configuration for SV-SCN training and inference.
All hyperparameters defined per ML Training Spec v1.0.
"""

from dataclasses import dataclass, field
from typing import List, Tuple
from pathlib import Path


@dataclass
class DataConfig:
    """Dataset configuration."""
    
    # Supported classes (frozen for v1)
    FURNITURE_CLASSES: List[str] = field(
        default_factory=lambda: ["chair", "stool", "table"]
    )
    NUM_CLASSES: int = 3
    
    # Point cloud sizes (frozen for v1)
    NUM_INPUT_POINTS: int = 2048   # Partial PC
    NUM_OUTPUT_POINTS: int = 8192  # Complete PC
    
    # Data split (objects must not overlap)
    TRAIN_SPLIT: float = 0.8
    VAL_SPLIT: float = 0.1
    TEST_SPLIT: float = 0.1
    
    # Camera settings for partial view generation
    CAMERA_ELEVATIONS: List[int] = field(
        default_factory=lambda: [15, 30, 45]  # degrees
    )
    AZIMUTH_RANGE: Tuple[int, int] = (0, 360)  # random
    
    # Dataset size targets
    BASELINE_SAMPLES: int = 5000
    STABLE_V1_SAMPLES: int = 10000
    
    # Paths
    RAW_DATA_DIR: Path = Path("data/raw")
    PROCESSED_DATA_DIR: Path = Path("data/processed")
    
    # Preprocessing
    RANDOM_DROPOUT_MAX: float = 0.05  # ≤5% random point dropout


@dataclass
class ModelConfig:
    """Model architecture configuration (frozen for v1)."""
    
    # Input/Output sizes
    INPUT_POINTS: int = 2048
    OUTPUT_POINTS: int = 8192
    INPUT_DIM: int = 3  # x, y, z
    
    # Encoder (PointNet-style)
    ENCODER_DIMS: List[int] = field(
        default_factory=lambda: [64, 128, 256, 512]
    )
    GLOBAL_FEATURE_DIM: int = 512
    
    # Decoder (FoldingNet-style)
    DECODER_DIMS: List[int] = field(
        default_factory=lambda: [512, 512, 256, 3]
    )
    FOLDING_GRID_SIZE: int = 91  # 91x91 ≈ 8281 → sample to 8192
    
    # Class conditioning
    NUM_CLASSES: int = 3
    CLASS_EMBED_DIM: int = 64


@dataclass
class TrainingConfig:
    """Training configuration per ML Training Spec."""
    
    # Optimizer
    OPTIMIZER: str = "adam"
    LEARNING_RATE_INIT: float = 1e-3
    LEARNING_RATE_FINAL: float = 1e-4
    WEIGHT_DECAY: float = 0.0
    
    # Training
    BATCH_SIZE: int = 32
    EPOCHS: int = 150
    GRADIENT_CLIP_NORM: float = 1.0
    
    # Scheduler
    LR_SCHEDULER: str = "cosine"  # cosine annealing
    WARMUP_EPOCHS: int = 5
    
    # Checkpointing
    CHECKPOINT_DIR: Path = Path("checkpoints")
    SAVE_EVERY_N_EPOCHS: int = 10
    KEEP_LAST_N_CHECKPOINTS: int = 3
    
    # Logging
    LOG_DIR: Path = Path("logs")
    LOG_EVERY_N_STEPS: int = 50
    
    # Validation
    VAL_EVERY_N_EPOCHS: int = 1


@dataclass
class InferenceConfig:
    """Inference and confidence estimation configuration."""
    
    # Confidence thresholds
    CONFIDENCE_THRESHOLD: float = 0.6  # Below this → fallback
    
    # Mesh reconstruction
    BALL_PIVOT_RADII: List[float] = field(
        default_factory=lambda: [0.01, 0.02, 0.04]
    )
    POISSON_DEPTH: int = 9
    
    # Export formats
    SUPPORTED_FORMATS: List[str] = field(
        default_factory=lambda: ["obj", "glb", "ply"]
    )


@dataclass
class Config:
    """Master configuration combining all sub-configs."""
    
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    
    # Versioning
    VERSION: str = "sv_scn_v0.1.0"
    
    # Device
    DEVICE: str = "cuda"  # or "cpu"
    
    # Random seed for reproducibility
    SEED: int = 42


# Default config instance
default_config = Config()
