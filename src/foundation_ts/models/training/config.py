from dataclasses import dataclass, field
from typing import Literal


@dataclass
class DatasetConfig:
    dataset_path: str
    seq_max_len: int = 4096
    seq_stride: int = 4096
    normalization_func: Literal["max", "zero"] = "zero"


@dataclass
class ModelConfig:
    hidden_size: int
    n_decoder_layers: int
    patch: bool = False
    patch_len: int = 32
    patch_stride: int = 32
    num_experts: int = 4
    num_expert_layers: int = 1
    k: int = 2
    n_head: int = 8
    d_ff: int | None = None
    d_expert: int | None = None
    horizons: list[int] = field(default_factory=lambda: [1, 8, 32, 64])


@dataclass
class TrainingConfig:
    model_config: ModelConfig
    epochs: int = 1
    steps_per_epoch: int = 100_000
    batch_size: int = 1024
    learning_rate: float = 1e-3
    weight_decay: float = 1e-1
    beta1: float = 0.9
    beta2: float = 0.95
    aux_loss_weight: float = 0.02
    max_grad_norm: float | None = 1.0
    drop_last: bool = True
    device: str | None = None
    seed: int | None = 42
    warmup_steps: int = 10_000
    use_bf16: bool = True
    use_amp: bool = True
    num_workers: int = 4
    prefetch_factor: int = 4
    pin_memory: bool = True
    grad_accum_steps: int = 1
    val_split: float = 0.01
    val_max_batches: int = 10
    ood_val_dataset_path: str | None = None
    ood_val_max_batches: int = 10
    log_every: int = 50
    val_every: int = 1000
    checkpoint_every: int = 2000
    checkpoint_dir: str = "checkpoints"
    resume_from_checkpoint: str | None = None
    profile: bool = False
    profile_dir: str | None = None
    compile: bool = False
    log_perf_metrics: bool = False
    mfu_peak_tflops: float | None = None


@dataclass
class RunnerConfig:
    dataset_config: DatasetConfig
    train_config: TrainingConfig
