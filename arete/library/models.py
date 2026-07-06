from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class AudioConfig:
    sample_rate: int = 44100
    channels: int = 1
    chunk_seconds: float = 2.5
    hop_seconds: float = 1.0


@dataclass
class ModelConfig:
    name: str = "WaveformUNet"
    in_channels: int = 1
    base_channels: int = 32
    depth: int = 5
    kernel_size: int = 15
    use_ema: bool = True
    ema_decay: float = 0.999


@dataclass
class DegradationConfig:
    mp3_bitrates: list[int] = field(default_factory=lambda: [64, 96, 128, 192])
    aac_bitrates: list[int] = field(default_factory=lambda: [64, 96, 128])
    opus_bitrates: list[int] = field(default_factory=lambda: [48, 64, 96])
    resample_rates: list[int] = field(default_factory=lambda: [16000, 22050])
    prob_lowpass: float = 0.3
    lowpass_cutoff_hz: int = 8000
    prob_stereo_collapse: float = 0.2
    prob_clipping: float = 0.1
    clipping_threshold: float = 0.9


@dataclass
class LossConfig:
    lambda_l1: float = 0.3
    lambda_mr_stft: float = 0.5
    lambda_mel: float = 0.2
    lambda_highband: float = 0.1
    highband_cutoff_hz: int = 8000
    mr_stft_fft_sizes: list[int] = field(default_factory=lambda: [512, 1024, 2048])
    mr_stft_hop_sizes: list[int] = field(default_factory=lambda: [128, 256, 512])
    mr_stft_win_sizes: list[int] = field(default_factory=lambda: [512, 1024, 2048])
    mel_n_mels: int = 80


@dataclass
class TrainingConfig:
    epochs: int = 100
    batch_size: int = 16
    learning_rate: float = 3.0e-4
    lr_scheduler: str = "cosine"
    warmup_epochs: int = 5
    grad_clip: float = 1.0
    mixed_precision: bool = True
    num_workers: int = 4
    val_every_n_epochs: int = 5
    save_every_n_epochs: int = 10
    log_dir: str = "runs"
    checkpoint_dir: str = "checkpoints"
