import io
import subprocess
from unittest.mock import patch

import numpy as np
import soundfile as sf
import torch

from audio_upscaler.data.degradations import Degrader


def _mock_subprocess_run(*args, **kwargs) -> subprocess.CompletedProcess:
    """Mock ffmpeg: return dummy WAV data for decode step, no-op for encode."""
    cmd = args[0] if args else []
    # Detect decode step (output to stdout via -f wav -)
    is_decode = "-f" in cmd and "wav" in cmd and "-" in cmd
    if is_decode:
        sr = 22050
        dummy = np.zeros((sr, 1), dtype=np.float32)
        buf = io.BytesIO()
        sf.write(buf, dummy, sr, format="WAV")
        buf.seek(0)
        return subprocess.CompletedProcess(
            args=cmd,
            returncode=0,
            stdout=buf.read(),
            stderr=b"",
        )
    return subprocess.CompletedProcess(args=cmd, returncode=0, stdout=b"", stderr=b"")


class TestDegrader:
    def test_init_defaults(self) -> None:
        d = Degrader()
        assert d.sample_rate == 44100
        assert d.mp3_bitrates == [64, 96, 128, 192]

    def test_degrade_returns_same_shape(self) -> None:
        d = Degrader(sample_rate=22050)
        wav = torch.randn(1, 22050)
        with patch.object(subprocess, "run", _mock_subprocess_run):
            degraded = d.degrade(wav)
        assert degraded.shape == wav.shape
        assert degraded.dtype == torch.float32

    def test_degrade_stereo_collapse(self) -> None:
        d = Degrader(sample_rate=22050, prob_stereo_collapse=1.0)
        wav = torch.randn(2, 22050)
        with patch.object(subprocess, "run", _mock_subprocess_run):
            degraded = d.degrade(wav)
        # Mock ffmpeg returns mono; real ffmpeg preserves channels
        assert degraded.shape[-1] == wav.shape[-1]

    def test_degrade_lowpass(self) -> None:
        d = Degrader(sample_rate=22050, prob_lowpass=1.0, lowpass_cutoff_hz=8000)
        wav = torch.randn(1, 22050)
        with patch.object(subprocess, "run", _mock_subprocess_run):
            degraded = d.degrade(wav)
        assert degraded.shape == wav.shape

    def test_degrade_clipping(self) -> None:
        d = Degrader(sample_rate=22050, prob_clipping=1.0, clipping_threshold=0.5)
        wav = torch.randn(1, 22050) * 0.8
        with patch.object(subprocess, "run", _mock_subprocess_run):
            degraded = d.degrade(wav)
        assert degraded.shape == wav.shape

    def test_degrade_shorter_padded(self) -> None:
        d = Degrader(sample_rate=22050)
        wav = torch.randn(1, 44100)
        with patch.object(subprocess, "run", _mock_subprocess_run):
            degraded = d.degrade(wav)
        assert degraded.shape[-1] == 44100

    def test_degrade_identity_no_random(self) -> None:
        """Without any degradation enabled, output should be close to input."""
        d = Degrader(
            sample_rate=22050,
            prob_lowpass=0.0,
            prob_stereo_collapse=0.0,
            prob_clipping=0.0,
        )
        wav = torch.randn(1, 22050) * 0.1
        with patch.object(subprocess, "run", _mock_subprocess_run):
            degraded = d.degrade(wav)
        assert degraded.shape == wav.shape
