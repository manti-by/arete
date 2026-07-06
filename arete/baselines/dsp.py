from __future__ import annotations

import numpy as np
import torch
import torchaudio.functional as taf


def upsample_baseline(
    waveform: torch.Tensor,
    sample_rate: int,
    target_sr: int = 44100,
) -> torch.Tensor:
    down = taf.resample(waveform, sample_rate, target_sr)
    up = taf.resample(down, target_sr, sample_rate)
    if up.shape[-1] < waveform.shape[-1]:
        up = torch.nn.functional.pad(up, (0, waveform.shape[-1] - up.shape[-1]))
    return up[..., : waveform.shape[-1]]


def compute_spectral_subtraction(
    waveform: np.ndarray,
    n_fft: int = 2048,
    hop_length: int = 512,
    noise_frames: int = 5,
    alpha: float = 1.5,
) -> np.ndarray:
    window = np.hanning(n_fft)
    spec = np.array(
        [np.fft.rfft(waveform[i : i + n_fft] * window) for i in range(0, len(waveform) - n_fft, hop_length)]
    )

    mag = np.abs(spec)
    phase = np.angle(spec)
    noise_mag = mag[:noise_frames].mean(axis=0, keepdims=True)
    mag_clean = np.maximum(mag - alpha * noise_mag, 0.0)
    spec_clean = mag_clean * np.exp(1j * phase)

    n_t = len(waveform)
    reconstructed = np.zeros(n_t)
    count = np.zeros(n_t)
    for i, frame_spec in enumerate(spec_clean):
        start = i * hop_length
        end = start + n_fft
        if end > n_t:
            break
        frame = np.fft.irfft(frame_spec)[:n_fft]
        reconstructed[start:end] += frame * window
        count[start:end] += window

    count = np.where(count < 1e-8, 1.0, count)
    return reconstructed / count


def wiener_denoise(
    waveform: torch.Tensor,
    n_fft: int = 2048,
    hop_length: int = 512,
    noise_frames: int = 5,
    alpha: float = 1.5,
) -> torch.Tensor:
    n_ch, _ = waveform.shape
    out_channels = []

    for ch in range(n_ch):
        wav = waveform[ch].numpy()
        reconstructed = compute_spectral_subtraction(wav, n_fft, hop_length, noise_frames, alpha)
        out_channels.append(torch.from_numpy(reconstructed.astype(np.float32)))

    return torch.stack(out_channels)


def harmonic_bwe(
    waveform: torch.Tensor,
    sample_rate: int,
    n_fft: int = 2048,
    hop_length: int = 512,
    source_max_hz: int = 8000,
    target_max_hz: int = 20000,
) -> torch.Tensor:
    n_ch, n_t = waveform.shape
    max_bin = n_fft // 2 + 1
    source_bin = int(source_max_hz / (sample_rate / 2) * max_bin)
    source_bin = min(source_bin, max_bin)
    target_bin = int(target_max_hz / (sample_rate / 2) * max_bin)
    target_bin = min(target_bin, max_bin)
    target_bin = max(target_bin, source_bin + 1)

    out_channels = []
    for ch in range(n_ch):
        wav_np = waveform[ch].numpy()
        window = np.hanning(n_fft)
        frames = []

        for start in range(0, n_t - n_fft, hop_length):
            frame = wav_np[start : start + n_fft] * window
            spec = np.fft.rfft(frame)
            mag = np.abs(spec)
            phase = np.angle(spec)

            low_mag = mag[:source_bin]
            needed = target_bin - source_bin
            repeats = int(np.ceil(needed / source_bin))
            extension = np.tile(low_mag, repeats)[:needed] * 0.5

            mag[source_bin:target_bin] = np.maximum(mag[source_bin:target_bin], extension)
            spec_new = mag * np.exp(1j * phase)
            frames.append(np.fft.irfft(spec_new)[:n_fft])

        out = np.zeros(n_t)
        cnt = np.zeros(n_t)
        for i, fr in enumerate(frames):
            s = i * hop_length
            e = s + n_fft
            if e > n_t:
                break
            out[s:e] += fr * window
            cnt[s:e] += window

        cnt = np.where(cnt < 1e-8, 1.0, cnt)
        out /= cnt
        out_channels.append(torch.from_numpy(out.astype(np.float32)))

    return torch.stack(out_channels)
