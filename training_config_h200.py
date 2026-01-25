"""
H200 Training Configuration for ARACHNE-X Avatar
Optimized for 141GB HBM3e with fp8, LoRA, and advanced distributed training.
"""

import torch
import json
from dataclasses import dataclass, asdict
from typing import Optional, List


@dataclass
class H200TrainingConfig:
    """H200-optimized training configuration for ARACHNE-X."""
    
    # Model Architecture
    model_name: str = "ARACHNE-X-Avatar"
    hidden_size: int = 3072  # Increased from 1536 for H200
    num_attention_heads: int = 32  # Increased from 16
    num_layers: int = 48  # Increased from baseline
    intermediate_size: int = 12288
    vocab_size: int = 32000
    
    # Precision & Quantization (fp8 for H200)
    dtype: str = "bfloat16"  # Base precision
    use_fp8: bool = True
    fp8_format: str = "OCP"  # Open Computation Project format
    use_gradient_checkpointing: bool = True
    use_flash_attention_2: bool = True
    
    # Batch & Gradient Settings
    batch_size: int = 64  # 2-3x increase for H200
    gradient_accumulation_steps: int = 2  # Can be lower due to batch size
    gradient_accumulation_dtype: str = "float32"
    max_grad_norm: float = 1.0
    
    # Learning Rate & Optimization
    learning_rate: float = 1e-4
    lr_scheduler: str = "cosine"
    warmup_steps: int = 5000
    warmup_ratio: float = 0.05
    max_steps: int = 500000
    weight_decay: float = 0.01
    
    # Adam Optimizer Settings
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    
    # Distributed Training (Multi-GPU)
    distributed_backend: str = "nccl"  # NVIDIA NCCL for H200
    num_processes: int = 8  # Assume 8x H200s for pod
    world_size: int = 8
    
    # Context Parallelism (for long sequences)
    use_context_parallel: bool = True
    cp_split_hw: tuple = (2, 2)  # Can go (4, 4) on multiple H200s
    
    # LoRA Fine-tuning (Avatar-specific)
    use_lora: bool = True
    lora_rank: int = 256  # High rank for avatar quality
    lora_alpha: int = 512
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = None
    
    # Audio Processing
    audio_model_name: str = "facebook/wav2vec2-xls-r-2b"
    audio_embedding_dim: int = 2560
    use_multi_stream_audio: bool = True
    
    # Loss Weights
    lip_sync_weight: float = 0.25
    identity_weight: float = 0.15
    temporal_weight: float = 0.10
    expression_weight: float = 0.10
    perceptual_weight: float = 0.40
    
    # Data Settings
    video_resolution: int = 1024
    video_fps: int = 30
    num_frames: int = 96
    crop_size: int = 512
    
    # Augmentation
    use_augmentation: bool = True
    augmentation_p: float = 0.5
    
    # Checkpoint & Logging
    save_steps: int = 1000
    eval_steps: int = 500
    log_steps: int = 50
    save_total_limit: int = 3
    resume_from_checkpoint: Optional[str] = None
    
    # Hardware Optimization for H200
    h200_specific: dict = None
    
    def __post_init__(self):
        if self.lora_target_modules is None:
            self.lora_target_modules = [
                "to_q", "to_k", "to_v", "to_out.0",
                "net.0", "net.2", "dense_h_to_4h", "dense_4h_to_h"
            ]
        
        # H200-specific optimizations
        if self.h200_specific is None:
            self.h200_specific = {
                "enable_tensor_core": True,
                "enable_sparsity": True,
                "memory_pool_fraction": 0.8,
                "enable_graphs": True,
                "cudnn_benchmark": True,
                "max_hbm_percent": 90  # Use 90% of 141GB HBM3e
            }
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        config_dict = asdict(self)
        config_dict['lora_target_modules'] = list(config_dict['lora_target_modules'])
        return config_dict
    
    def save_json(self, path: str) -> None:
        """Save configuration to JSON."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def from_json(cls, path: str):
        """Load configuration from JSON."""
        with open(path, 'r') as f:
            config_dict = json.load(f)
        config_dict['lora_target_modules'] = config_dict.get('lora_target_modules', cls().lora_target_modules)
        return cls(**config_dict)


@dataclass
class H200DataConfig:
    """Data pipeline configuration for H200 training."""
    
    # Dataset paths
    dataset_root: str = "/mnt/datasets/avatar-training"
    train_split: float = 0.9
    val_split: float = 0.05
    test_split: float = 0.05
    
    # Data loading
    num_workers: int = 16  # High for H200
    prefetch_factor: int = 2
    pin_memory: bool = True
    persistent_workers: bool = True
    
    # Frame sampling strategy
    sampling_strategy: str = "uniform"  # or "random", "action_based"
    sample_frames_per_video: int = 96
    stride: int = 1
    
    # Audio settings
    audio_sample_rate: int = 16000
    audio_length_ms: int = 5000  # 5 seconds
    
    # Preprocessing
    normalize_video: bool = True
    video_mean: List[float] = None
    video_std: List[float] = None
    
    def __post_init__(self):
        if self.video_mean is None:
            self.video_mean = [0.485, 0.456, 0.406]  # ImageNet
        if self.video_std is None:
            self.video_std = [0.229, 0.224, 0.225]


@dataclass
class H200LoRAConfig:
    """LoRA-specific configuration for ARACHNE-X fine-tuning."""
    
    # LoRA architecture
    rank: int = 256
    alpha: int = 512
    dropout: float = 0.05
    
    # Target modules for avatar face generation
    target_modules: List[str] = None
    
    # Optimization
    use_rslora: bool = True  # Rank-Stabilized LoRA
    use_dora: bool = False   # Can enable for better performance
    
    # Training
    lora_learning_rate: float = 5e-4
    use_bias_correction: bool = True
    
    def __post_init__(self):
        if self.target_modules is None:
            self.target_modules = [
                # Attention layers
                "attn.to_q", "attn.to_k", "attn.to_v",
                "attn.to_out.0",
                "cross_attn.to_q", "cross_attn.to_k", 
                "cross_attn.to_v", "cross_attn.to_out.0",
                # Avatar-specific
                "audio_cross_attn.q_proj", "audio_cross_attn.k_proj",
                "audio_cross_attn.v_proj", "audio_cross_attn.out_proj",
                # FFN layers
                "net.0", "net.2"
            ]


def create_h200_config(
    num_gpus: int = 8,
    batch_size_per_gpu: int = 8,
    use_lora: bool = True,
    training_steps: int = 500000
) -> H200TrainingConfig:
    """Factory function to create H200 config with custom parameters."""
    config = H200TrainingConfig(
        batch_size=batch_size_per_gpu,
        world_size=num_gpus,
        num_processes=num_gpus,
        max_steps=training_steps,
        use_lora=use_lora
    )
    
    # Adjust batch size and accumulation steps
    config.batch_size = batch_size_per_gpu
    config.gradient_accumulation_steps = max(1, 128 // (batch_size_per_gpu * num_gpus))
    
    return config


# Pre-configured profiles
PROFILE_H200_SINGLE = H200TrainingConfig(
    num_processes=1,
    world_size=1,
    batch_size=16,
    cp_split_hw=(1, 1)
)

PROFILE_H200_DUAL = H200TrainingConfig(
    num_processes=2,
    world_size=2,
    batch_size=32,
    cp_split_hw=(1, 2)
)

PROFILE_H200_POD = H200TrainingConfig(
    num_processes=8,
    world_size=8,
    batch_size=64,
    cp_split_hw=(2, 2)
)

PROFILE_H200_MEGA = H200TrainingConfig(
    num_processes=16,
    world_size=16,
    batch_size=128,
    cp_split_hw=(4, 2),
    num_layers=64,
    hidden_size=4096
)


# H200 Memory Optimization Settings
H200_MEMORY_OPTIMIZATIONS = {
    "enable_gradient_checkpointing": True,
    "enable_flash_attention": True,
    "enable_memory_efficient_attention": True,
    "enable_fused_ops": True,
    "dtype": torch.bfloat16,
    "use_fp8_compute": True,
    "fused_softmax": True,
    "fused_gelu": True,
    "memory_fraction": 0.85,  # Use 85% of 141GB HBM3e
}
