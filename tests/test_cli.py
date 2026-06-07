from click.testing import CliRunner

from audio_upscaler.cli import main


class TestCLI:
    def test_info(self) -> None:
        runner = CliRunner()
        result = runner.invoke(main, ["info"])
        assert result.exit_code == 0
        assert "WaveformUNet" in result.output
        assert "STFTUNet" in result.output
        assert "parameters" in result.output.lower()

    def test_info_custom_config(self, tmp_path):
        config = tmp_path / "config.yaml"
        config.write_text("""
audio:
  sample_rate: 22050
  channels: 1
  chunk_seconds: 1.0
  hop_seconds: 0.5
model:
  name: WaveformUNet
  in_channels: 1
  base_channels: 8
  depth: 3
  kernel_size: 7
  use_ema: true
  ema_decay: 0.999
loss:
  lambda_l1: 0.3
  lambda_mr_stft: 0.5
  lambda_mel: 0.2
  lambda_highband: 0.1
  highband_cutoff_hz: 8000
  mr_stft_fft_sizes: [512, 1024, 2048]
  mr_stft_hop_sizes: [128, 256, 512]
  mr_stft_win_sizes: [512, 1024, 2048]
  mel_n_mels: 80
training:
  epochs: 1
  batch_size: 2
  learning_rate: 3.0e-4
  lr_scheduler: cosine
  warmup_epochs: 0
  grad_clip: 1.0
  mixed_precision: false
  num_workers: 0
  val_every_n_epochs: 5
  save_every_n_epochs: 10
  log_dir: runs/
  checkpoint_dir: checkpoints/
data:
  raw_dir: data/raw
  processed_dir: data/processed
  train_split: 0.9
  seed: 42
""")
        runner = CliRunner()
        result = runner.invoke(main, ["info", "--config", str(config)])
        assert result.exit_code == 0
        assert "WaveformUNet" in result.output

    def test_info_help(self) -> None:
        runner = CliRunner()
        result = runner.invoke(main, ["--help"])
        assert result.exit_code == 0
        assert "Audio Upscaler" in result.output
