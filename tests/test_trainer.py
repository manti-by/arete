import torch
from torch.utils.data import DataLoader

from audio_upscaler.models import WaveformUNet
from audio_upscaler.training import Trainer


def _dummy_loader(batch_size: int = 2, n_batches: int = 3) -> DataLoader:
    """Create a DataLoader yielding random (degraded, clean) pairs."""
    data = [(torch.randn(1, 16000), torch.randn(1, 16000)) for _ in range(batch_size * n_batches)]
    return DataLoader(data, batch_size=batch_size)


CONFIG = {
    "audio": {
        "sample_rate": 16000,
        "channels": 1,
        "chunk_seconds": 1.0,
    },
    "model": {
        "use_ema": True,
        "ema_decay": 0.999,
    },
    "training": {
        "epochs": 1,
        "batch_size": 2,
        "learning_rate": 3.0e-4,
        "lr_scheduler": "cosine",
        "warmup_epochs": 0,
        "grad_clip": 1.0,
        "mixed_precision": False,
        "num_workers": 0,
        "val_every_n_epochs": 1,
        "save_every_n_epochs": 1,
        "log_dir": "runs",
        "checkpoint_dir": "checkpoints",
    },
    "loss": {
        "lambda_l1": 0.3,
        "lambda_mr_stft": 0.5,
        "lambda_mel": 0.2,
        "lambda_highband": 0.1,
        "highband_cutoff_hz": 4000,
        "mr_stft_fft_sizes": [512, 1024],
        "mr_stft_hop_sizes": [128, 256],
        "mr_stft_win_sizes": [512, 1024],
        "mel_n_mels": 40,
    },
}


class TestTrainer:
    def test_init(self) -> None:
        model = WaveformUNet(in_channels=1, base_channels=8, depth=2)
        train_loader = _dummy_loader()
        val_loader = _dummy_loader(n_batches=1)
        trainer = Trainer(model, train_loader, val_loader, CONFIG, device="cpu")
        assert trainer.epochs == 1
        assert trainer.grad_clip == 1.0
        assert trainer.ema is not None
        assert trainer.criterion is not None

    def test_init_no_ema(self) -> None:
        cfg = {**CONFIG, "model": {**CONFIG["model"], "use_ema": False}}
        model = WaveformUNet(in_channels=1, base_channels=8, depth=2)
        train_loader = _dummy_loader()
        val_loader = _dummy_loader(n_batches=1)
        trainer = Trainer(model, train_loader, val_loader, cfg, device="cpu")
        assert trainer.ema is None

    def test_init_mixed_precision(self) -> None:
        cfg = {**CONFIG}
        cfg["training"] = {**CONFIG["training"], "mixed_precision": True}
        model = WaveformUNet(in_channels=1, base_channels=8, depth=2)
        train_loader = _dummy_loader()
        val_loader = _dummy_loader(n_batches=1)
        trainer = Trainer(model, train_loader, val_loader, cfg, device="cpu")
        assert trainer.scaler.is_enabled() is False  # disabled on cpu regardless

    def test_fit_one_epoch(self) -> None:
        model = WaveformUNet(in_channels=1, base_channels=8, depth=2)
        train_loader = _dummy_loader()
        val_loader = _dummy_loader(n_batches=1)
        trainer = Trainer(model, train_loader, val_loader, CONFIG, device="cpu")
        trainer.fit()
        # After training, checkpoint should be saved
        ckpt = trainer.ckpt_dir / "checkpoint_epoch_final.pt"
        assert ckpt.exists()
        ckpt.unlink(missing_ok=True)
        ckpt2 = trainer.ckpt_dir / "checkpoint_epoch_1.pt"
        ckpt2.unlink(missing_ok=True)

    def test_save_checkpoint(self) -> None:
        model = WaveformUNet(in_channels=1, base_channels=8, depth=2)
        train_loader = _dummy_loader()
        val_loader = _dummy_loader(n_batches=1)
        trainer = Trainer(model, train_loader, val_loader, CONFIG, device="cpu")
        trainer._save_checkpoint(42)
        ckpt = trainer.ckpt_dir / "checkpoint_epoch_42.pt"
        assert ckpt.exists()
        state = torch.load(ckpt, map_location="cpu")
        assert "model" in state
        assert "optimizer" in state
        assert "epoch" in state
        ckpt.unlink(missing_ok=True)
