from __future__ import annotations

import logging
from pathlib import Path

import torch
import torch.nn as nn
import torchaudio


logger = logging.getLogger(__name__)


class Enhancer:
    def __init__(
        self,
        model: nn.Module,
        sample_rate: int = 44100,
        chunk_seconds: float = 2.5,
        hop_seconds: float = 1.0,
        device: str = "cpu",
    ) -> None:
        self.model = model.to(device).eval()
        self.sample_rate = sample_rate
        self.chunk_len = int(chunk_seconds * sample_rate)
        self.hop_len = int(hop_seconds * sample_rate)
        self.device = device

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        model: nn.Module,
        sample_rate: int = 44100,
        chunk_seconds: float = 2.5,
        hop_seconds: float = 1.0,
        device: str = "cpu",
        use_ema: bool = True,
    ) -> Enhancer:
        state = torch.load(checkpoint_path, map_location=device, weights_only=True)
        key = "ema" if (use_ema and "ema" in state) else "model"
        model.load_state_dict(state[key])
        return cls(model, sample_rate, chunk_seconds, hop_seconds, device)

    @torch.no_grad()
    def enhance_waveform(self, waveform: torch.Tensor) -> torch.Tensor:
        n_ch, n_t = waveform.shape
        output = torch.zeros(n_ch, n_t)
        weight = torch.zeros(n_t)
        window = torch.hann_window(self.chunk_len)

        start = 0
        while start < n_t:
            end = min(start + self.chunk_len, n_t)
            chunk = waveform[:, start:end]

            if chunk.shape[-1] < self.chunk_len:
                pad = self.chunk_len - chunk.shape[-1]
                chunk = torch.nn.functional.pad(chunk, (0, pad))

            chunk_device = chunk.unsqueeze(0).to(self.device)
            pred = self.model(chunk_device).squeeze(0).cpu()

            length = end - start
            output[:, start:end] += pred[:, :length] * window[:length]
            weight[start:end] += window[:length]

            start += self.hop_len

        weight = weight.clamp(min=1e-8)
        output = output / weight
        return output

    def enhance_file(self, input_path: str | Path, output_path: str | Path) -> None:
        input_path = Path(input_path)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        waveform, sr = torchaudio.load(str(input_path))
        if sr != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sr, self.sample_rate)

        enhanced = self.enhance_waveform(waveform)
        torchaudio.save(str(output_path), enhanced, self.sample_rate)
        logger.info("Saved enhanced audio: %s", output_path)
