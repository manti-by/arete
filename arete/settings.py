import os
from pathlib import Path

from arete.library.types import AudioConfig, DataConfig, DegradationConfig, LossConfig, ModelConfig, TrainingConfig


BASE_PATH = Path(__file__).resolve().parent.parent

LOG_PATH = os.environ.get("LOG_PATH", BASE_PATH / "arete.log")

LOGGING: dict = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "standard": {
            "format": "%(asctime)s %(levelname)-6s: %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S",
        }
    },
    "handlers": {
        "console": {
            "level": "DEBUG",
            "class": "logging.StreamHandler",
            "formatter": "standard",
        },
        "file": {
            "level": "INFO",
            "class": "logging.FileHandler",
            "filename": str(LOG_PATH),
            "formatter": "standard",
        },
    },
    "loggers": {
        "": {"handlers": ["console", "file"], "level": "DEBUG", "propagate": True},
    },
}

AUDIO: AudioConfig = {
    "sample_rate": 44100,
    "channels": 1,
    "chunk_seconds": 2.5,
    "hop_seconds": 1.0,
}

DEGRADATIONS: DegradationConfig = {
    "mp3_bitrates": [64, 96, 128, 192],
    "aac_bitrates": [64, 96, 128],
    "opus_bitrates": [48, 64, 96],
    "resample_rates": [16000, 22050],
    "prob_lowpass": 0.3,
    "lowpass_cutoff_hz": 8000,
    "prob_stereo_collapse": 0.2,
    "prob_clipping": 0.1,
    "clipping_threshold": 0.9,
}

MODEL: ModelConfig = {
    "name": "WaveformUNet",
    "in_channels": 1,
    "base_channels": 32,
    "depth": 5,
    "kernel_size": 15,
    "use_ema": True,
    "ema_decay": 0.999,
}

LOSS: LossConfig = {
    "lambda_l1": 0.3,
    "lambda_mr_stft": 0.5,
    "lambda_mel": 0.2,
    "lambda_highband": 0.1,
    "highband_cutoff_hz": 8000,
    "mr_stft_fft_sizes": [512, 1024, 2048],
    "mr_stft_hop_sizes": [128, 256, 512],
    "mr_stft_win_sizes": [512, 1024, 2048],
    "mel_n_mels": 80,
}

TRAINING: TrainingConfig = {
    "epochs": 100,
    "batch_size": 4,
    "learning_rate": 3.0e-4,
    "lr_scheduler": "cosine",
    "warmup_epochs": 5,
    "grad_clip": 1.0,
    "mixed_precision": True,
    "num_workers": 4,
    "val_every_n_epochs": 5,
    "save_every_n_epochs": 10,
    "log_dir": "runs/",
    "checkpoint_dir": "checkpoints/",
}

DATA: DataConfig = {
    "train_split": 0.9,
    "seed": 42,
}
