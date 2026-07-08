from typing import TypedDict


class AudioConfig(TypedDict):
    sample_rate: int
    channels: int
    chunk_seconds: float
    hop_seconds: float


class DegradationConfig(TypedDict):
    mp3_bitrates: list[int]
    aac_bitrates: list[int]
    opus_bitrates: list[int]
    resample_rates: list[int]
    prob_lowpass: float
    lowpass_cutoff_hz: int
    prob_stereo_collapse: float
    prob_clipping: float
    clipping_threshold: float


class ModelConfig(TypedDict):
    name: str
    in_channels: int
    base_channels: int
    depth: int
    kernel_size: int
    use_ema: bool
    ema_decay: float


class LossConfig(TypedDict):
    lambda_l1: float
    lambda_mr_stft: float
    lambda_mel: float
    lambda_highband: float
    highband_cutoff_hz: int
    mr_stft_fft_sizes: list[int]
    mr_stft_hop_sizes: list[int]
    mr_stft_win_sizes: list[int]
    mel_n_mels: int


class TrainingConfig(TypedDict):
    epochs: int
    batch_size: int
    learning_rate: float
    lr_scheduler: str
    warmup_epochs: int
    grad_clip: float
    mixed_precision: bool
    num_workers: int
    val_every_n_epochs: int
    save_every_n_epochs: int
    log_dir: str
    checkpoint_dir: str


class DataConfig(TypedDict):
    train_split: float
    seed: int


class TrainMetrics(TypedDict):
    l_time: float
    l_mr_stft: float
    l_mel: float
    l_highband: float
    total: float


class CheckpointState(TypedDict, total=False):
    epoch: int | str
    model: dict
    optimizer: dict
    scheduler: dict
    ema: dict | None
