import torch

from audio_upscaler.baselines import harmonic_bwe, upsample_baseline, wiener_denoise


SR = 22050
SAMPLES = SR * 2


def dummy_audio(channels: int = 1) -> torch.Tensor:
    return torch.randn(channels, SAMPLES) * 0.1


class TestDSPBaselines:
    def test_upsample_shape(self) -> None:
        x = dummy_audio()
        y = upsample_baseline(x, SR, target_sr=8000)
        assert y.shape == x.shape

    def test_upsample_stereo(self) -> None:
        x = dummy_audio(channels=2)
        y = upsample_baseline(x, SR, target_sr=8000)
        assert y.shape == x.shape

    def test_wiener_shape(self) -> None:
        x = dummy_audio()
        y = wiener_denoise(x, n_fft=512, hop_length=128)
        assert y.shape == x.shape

    def test_wiener_stereo(self) -> None:
        x = dummy_audio(channels=2)
        y = wiener_denoise(x, n_fft=512, hop_length=128)
        assert y.shape == x.shape

    def test_harmonic_bwe_shape(self) -> None:
        x = dummy_audio()
        y = harmonic_bwe(x, SR, n_fft=512, hop_length=128)
        assert y.shape == x.shape

    def test_harmonic_bwe_stereo(self) -> None:
        x = dummy_audio(channels=2)
        y = harmonic_bwe(x, SR, n_fft=512, hop_length=128)
        assert y.shape == x.shape

    def test_harmonic_bwe_source_above_target(self) -> None:
        x = dummy_audio()
        y = harmonic_bwe(x, SR, n_fft=512, hop_length=128, source_max_hz=16000, target_max_hz=8000)
        assert y.shape == x.shape

    def test_harmonic_bwe_nyquist_limit(self) -> None:
        x = dummy_audio()
        y = harmonic_bwe(x, SR, n_fft=512, hop_length=128, source_max_hz=20000, target_max_hz=22050)
        assert y.shape == x.shape
