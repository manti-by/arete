from pathlib import Path

import soundfile as sf
import torch
import torch.nn as nn

from arete.services import Enhancer


class IdentityModel(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class TestEnhancer:
    def test_init(self) -> None:
        model = IdentityModel()
        enh = Enhancer(model, sample_rate=16000, chunk_seconds=1.0, hop_seconds=0.5)
        assert enh.sample_rate == 16000
        assert enh.chunk_len == 16000
        assert enh.hop_len == 8000

    def test_enhance_waveform_shape(self) -> None:
        model = IdentityModel()
        enh = Enhancer(model, sample_rate=16000, chunk_seconds=0.5, hop_seconds=0.25)
        wav = torch.randn(1, 16000)
        out = enh.enhance_waveform(wav)
        assert out.shape == wav.shape

    def test_enhance_waveform_identity(self) -> None:
        model = IdentityModel()
        enh = Enhancer(model, sample_rate=16000, chunk_seconds=0.5, hop_seconds=0.25)
        wav = torch.randn(1, 16000) * 0.1
        out = enh.enhance_waveform(wav)
        assert out.shape == wav.shape
        assert torch.allclose(out[:, 1:], wav[:, 1:], atol=1e-4)

    def test_enhance_waveform_stereo(self) -> None:
        model = IdentityModel()
        enh = Enhancer(model, sample_rate=16000, chunk_seconds=0.5, hop_seconds=0.25)
        wav = torch.randn(2, 16000) * 0.1
        out = enh.enhance_waveform(wav)
        assert out.shape == wav.shape
        assert torch.allclose(out[:, 1:], wav[:, 1:], atol=1e-4)

    def test_enhance_waveform_long_audio(self) -> None:
        model = IdentityModel()
        enh = Enhancer(model, sample_rate=16000, chunk_seconds=0.3, hop_seconds=0.1)
        wav = torch.randn(1, 48000) * 0.1
        out = enh.enhance_waveform(wav)
        assert out.shape == wav.shape

    def test_enhance_file(self, tmp_path: Path) -> None:
        model = IdentityModel()
        enh = Enhancer(model, sample_rate=16000, chunk_seconds=1.0, hop_seconds=0.5)

        input_path = tmp_path / "input.wav"
        output_path = tmp_path / "output.wav"
        wav_np = torch.randn(1, 16000).numpy().astype("float32")
        sf.write(str(input_path), wav_np.T, 16000)

        enh.enhance_file(str(input_path), str(output_path))
        assert output_path.exists()

        loaded, _ = sf.read(str(output_path))
        assert loaded.ndim in (1, 2)

    def test_from_checkpoint(self, tmp_path: Path) -> None:
        model = IdentityModel()
        ckpt_path = tmp_path / "model.pt"
        torch.save({"model": model.state_dict()}, ckpt_path)

        enh = Enhancer.from_checkpoint(
            checkpoint_path=str(ckpt_path),
            model=IdentityModel(),
            sample_rate=16000,
            chunk_seconds=0.5,
            hop_seconds=0.25,
            device="cpu",
        )
        assert enh.sample_rate == 16000
        wav = torch.randn(1, 8000) * 0.1
        out = enh.enhance_waveform(wav)
        assert out.shape == wav.shape

    def test_from_checkpoint_ema(self, tmp_path: Path) -> None:
        model = IdentityModel()
        ckpt_path = tmp_path / "model_ema.pt"
        torch.save({"model": model.state_dict(), "ema": model.state_dict()}, ckpt_path)

        enh = Enhancer.from_checkpoint(
            checkpoint_path=str(ckpt_path),
            model=IdentityModel(),
            sample_rate=16000,
            chunk_seconds=0.5,
            hop_seconds=0.25,
            device="cpu",
            use_ema=True,
        )
        wav = torch.randn(1, 8000) * 0.1
        out = enh.enhance_waveform(wav)
        assert out.shape == wav.shape

    def test_enhance_file_resamples(self, tmp_path: Path) -> None:
        model = IdentityModel()
        enh = Enhancer(model, sample_rate=16000, chunk_seconds=0.5, hop_seconds=0.25)

        input_path = tmp_path / "input_48k.wav"
        output_path = tmp_path / "output_48k.wav"
        wav_np = torch.randn(1, 48000).numpy().astype("float32")
        sf.write(str(input_path), wav_np.T, 48000)

        enh.enhance_file(str(input_path), str(output_path))
        assert output_path.exists()
