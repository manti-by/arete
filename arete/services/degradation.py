import io
import random
import subprocess
import tempfile
from pathlib import Path
from typing import ClassVar

import numpy as np
import soundfile as sf
import torch


_rng = random.SystemRandom()


class Degrader:
    CODEC_CHOICES: ClassVar[list[str]] = ["mp3", "aac", "opus"]

    def __init__(
        self,
        sample_rate: int = 44100,
        mp3_bitrates: list[int] | None = None,
        aac_bitrates: list[int] | None = None,
        opus_bitrates: list[int] | None = None,
        resample_rates: list[int] | None = None,
        prob_lowpass: float = 0.3,
        lowpass_cutoff_hz: int = 8000,
        prob_stereo_collapse: float = 0.2,
        prob_clipping: float = 0.1,
        clipping_threshold: float = 0.9,
    ) -> None:
        self.sample_rate = sample_rate
        self.mp3_bitrates = mp3_bitrates or [64, 96, 128, 192]
        self.aac_bitrates = aac_bitrates or [64, 96, 128]
        self.opus_bitrates = opus_bitrates or [48, 64, 96]
        self.resample_rates = resample_rates or [16000, 22050]
        self.prob_lowpass = prob_lowpass
        self.lowpass_cutoff_hz = lowpass_cutoff_hz
        self.prob_stereo_collapse = prob_stereo_collapse
        self.prob_clipping = prob_clipping
        self.clipping_threshold = clipping_threshold

    def degrade(self, waveform: torch.Tensor) -> torch.Tensor:
        audio_np = waveform.numpy().T

        if audio_np.shape[1] > 1 and _rng.random() < self.prob_stereo_collapse:
            mono = audio_np.mean(axis=1, keepdims=True)
            audio_np = np.concatenate([mono, mono], axis=1)

        apply_lowpass = _rng.random() < self.prob_lowpass
        codec = _rng.choice(self.CODEC_CHOICES)
        audio_np = self.codec_degrade(audio_np, codec, apply_lowpass)

        if _rng.random() < 0.3:
            target_sr = _rng.choice(self.resample_rates)
            audio_np = self.resample(audio_np, target_sr)

        if _rng.random() < self.prob_clipping:
            audio_np = np.clip(audio_np, -self.clipping_threshold, self.clipping_threshold)
            audio_np /= self.clipping_threshold

        degraded = torch.from_numpy(audio_np.T.copy()).float()
        if degraded.shape[-1] < waveform.shape[-1]:
            pad = waveform.shape[-1] - degraded.shape[-1]
            degraded = torch.nn.functional.pad(degraded, (0, pad))
        elif degraded.shape[-1] > waveform.shape[-1]:
            degraded = degraded[..., : waveform.shape[-1]]

        return degraded

    def build_ffmpeg_filters(self, codec: str, bitrate: int, apply_lowpass: bool) -> tuple[list[str], list[str], str]:
        in_filters: list[str] = []
        if apply_lowpass:
            in_filters.append(f"lowpass=f={self.lowpass_cutoff_hz}")

        if codec == "mp3":
            codec_args = ["-c:a", "libmp3lame", "-b:a", f"{bitrate}k"]
            ext = "mp3"
        elif codec == "aac":
            codec_args = ["-c:a", "aac", "-b:a", f"{bitrate}k"]
            ext = "aac"
        else:
            codec_args = ["-c:a", "libopus", "-b:a", f"{bitrate}k"]
            ext = "ogg"

        return in_filters, codec_args, ext

    def codec_degrade(self, audio_np: np.ndarray, codec: str, apply_lowpass: bool) -> np.ndarray:
        bitrates = {
            "mp3": self.mp3_bitrates,
            "aac": self.aac_bitrates,
            "opus": self.opus_bitrates,
        }[codec]
        bitrate = _rng.choice(bitrates)
        in_filters, codec_args, ext = self.build_ffmpeg_filters(codec, bitrate, apply_lowpass)

        n_channels = audio_np.shape[1]
        filter_str = ",".join(in_filters) if in_filters else "anull"

        with tempfile.NamedTemporaryFile(suffix=f".{ext}", delete=False) as tmp_out:
            tmp_path = tmp_out.name

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_in:
            sf.write(tmp_in.name, audio_np, self.sample_rate)
            cmd = [
                "ffmpeg",
                "-y",
                "-i",
                tmp_in.name,
                "-af",
                filter_str,
                *codec_args,
                tmp_path,
            ]
            try:
                subprocess.run(cmd, capture_output=True, check=True)
            except Exception:
                Path(tmp_path).unlink(missing_ok=True)
                Path(tmp_in.name).unlink(missing_ok=True)
                raise

        cmd2 = ["ffmpeg", "-y", "-i", tmp_path, "-ar", str(self.sample_rate), "-ac", str(n_channels), "-f", "wav", "-"]
        try:
            result = subprocess.run(cmd2, capture_output=True, check=True)
        except Exception:
            Path(tmp_path).unlink(missing_ok=True)
            Path(tmp_in.name).unlink(missing_ok=True)
            raise

        audio_out, _ = sf.read(io.BytesIO(result.stdout))
        if audio_out.ndim == 1:
            audio_out = audio_out[:, np.newaxis]

        Path(tmp_path).unlink(missing_ok=True)
        Path(tmp_in.name).unlink(missing_ok=True)

        return audio_out.astype(np.float32)

    def resample(self, audio_np: np.ndarray, target_sr: int) -> np.ndarray:
        n_channels = audio_np.shape[1]

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f_in:
            sf.write(f_in.name, audio_np, self.sample_rate)

            cmd_down = [
                "ffmpeg",
                "-y",
                "-i",
                f_in.name,
                "-ar",
                str(target_sr),
                "-ac",
                str(n_channels),
                "-f",
                "wav",
                "-",
            ]
            try:
                r = subprocess.run(cmd_down, capture_output=True, check=True)
                mid, _ = sf.read(io.BytesIO(r.stdout))
                if mid.ndim == 1:
                    mid = mid[:, np.newaxis]
            except Exception:
                Path(f_in.name).unlink(missing_ok=True)
                raise

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f_mid:
            sf.write(f_mid.name, mid, target_sr)
            cmd_up = [
                "ffmpeg",
                "-y",
                "-i",
                f_mid.name,
                "-ar",
                str(self.sample_rate),
                "-ac",
                str(n_channels),
                "-f",
                "wav",
                "-",
            ]
            try:
                r = subprocess.run(cmd_up, capture_output=True, check=True)
                out, _ = sf.read(io.BytesIO(r.stdout))
                if out.ndim == 1:
                    out = out[:, np.newaxis]
            except Exception:
                Path(f_mid.name).unlink(missing_ok=True)
                Path(f_in.name).unlink(missing_ok=True)
                raise

        Path(f_mid.name).unlink(missing_ok=True)
        Path(f_in.name).unlink(missing_ok=True)
        return out.astype(np.float32)
