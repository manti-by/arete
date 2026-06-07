"""Combined audio restoration loss.

L = λ1·L_time + λ2·L_MR-STFT + λ3·L_mel + λ4·L_highband

References:
    - auraloss: https://github.com/csteinmetz1/auraloss
    - Steinmetz & Reiss, "auraloss: Audio focused loss functions in PyTorch" (2020)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


try:
    from auraloss.freq import MelSTFTLoss, MultiResolutionSTFTLoss

    _AURALOSS_AVAILABLE = True
except ImportError:
    _AURALOSS_AVAILABLE = False


class _FallbackMRSTFTLoss(nn.Module):
    """Minimal multi-resolution STFT loss when auraloss is not installed."""

    def __init__(self, fft_sizes: list[int], hop_sizes: list[int], win_sizes: list[int]) -> None:
        super().__init__()
        self.params = list(zip(fft_sizes, hop_sizes, win_sizes, strict=True))

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = pred.squeeze(1)
        target = target.squeeze(1)
        losses = []
        for n_fft, hop, win in self.params:
            p_spec = torch.stft(pred, n_fft, hop, win, return_complex=True).abs()
            t_spec = torch.stft(target, n_fft, hop, win, return_complex=True).abs()
            losses.append(F.l1_loss(p_spec, t_spec))
        return torch.stack(losses).mean()


class CombinedAudioLoss(nn.Module):
    """Weighted combination of time-domain L1, MR-STFT, Mel-STFT, and high-band losses.

    Args:
        sample_rate:        audio sample rate (needed for Mel loss)
        lambda_l1:          weight for time-domain L1
        lambda_mr_stft:     weight for multi-resolution STFT loss
        lambda_mel:         weight for Mel-STFT loss
        lambda_highband:    weight for high-frequency band loss
        highband_cutoff_hz: cutoff frequency defining the high band
        fft_sizes:          FFT sizes for MR-STFT
        hop_sizes:          hop sizes for MR-STFT
        win_sizes:          window sizes for MR-STFT
        n_mels:             number of mel filterbanks
    """

    def __init__(
        self,
        sample_rate: int = 44100,
        lambda_l1: float = 0.3,
        lambda_mr_stft: float = 0.5,
        lambda_mel: float = 0.2,
        lambda_highband: float = 0.1,
        highband_cutoff_hz: int = 8000,
        fft_sizes: list[int] | None = None,
        hop_sizes: list[int] | None = None,
        win_sizes: list[int] | None = None,
        n_mels: int = 80,
    ) -> None:
        super().__init__()
        self.lambda_l1 = lambda_l1
        self.lambda_mr_stft = lambda_mr_stft
        self.lambda_mel = lambda_mel
        self.lambda_highband = lambda_highband
        self.sample_rate = sample_rate

        fft_sizes = fft_sizes or [512, 1024, 2048]
        hop_sizes = hop_sizes or [128, 256, 512]
        win_sizes = win_sizes or [512, 1024, 2048]

        # High-band cutoff bin at n_fft=2048; clamp to avoid empty slice
        hb_n_bins = fft_sizes[-1] // 2 + 1
        self.hb_bin = min(
            int(highband_cutoff_hz / (sample_rate / 2) * hb_n_bins),
            hb_n_bins - 1,
        )
        self.hb_n_fft = fft_sizes[-1]
        self.hb_hop = hop_sizes[-1]

        if _AURALOSS_AVAILABLE:
            self.mr_stft_loss = MultiResolutionSTFTLoss(
                fft_sizes=fft_sizes,
                hop_sizes=hop_sizes,
                win_lengths=win_sizes,
                scale="mel",
                n_bins=n_mels,
                sample_rate=sample_rate,
                perceptual_weighting=True,
            )
            self.mel_loss = MelSTFTLoss(
                sample_rate=sample_rate,
                fft_size=fft_sizes[-1],
                hop_size=hop_sizes[-1],
                win_length=win_sizes[-1],
                n_mels=n_mels,
            )
        else:
            self.mr_stft_loss = _FallbackMRSTFTLoss(fft_sizes, hop_sizes, win_sizes)
            self.mel_loss = None

    def _highband_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """L1 loss on the high-frequency portion of the magnitude spectrogram."""
        p = pred.squeeze(1)
        t = target.squeeze(1)
        p_spec = torch.stft(p, self.hb_n_fft, self.hb_hop, return_complex=True).abs()
        t_spec = torch.stft(t, self.hb_n_fft, self.hb_hop, return_complex=True).abs()
        return F.l1_loss(p_spec[:, self.hb_bin :], t_spec[:, self.hb_bin :])

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute combined loss.

        Args:
            pred:   (B, C, T) predicted waveform
            target: (B, C, T) clean target waveform

        Returns:
            total_loss, dict of component losses
        """
        l_time = F.l1_loss(pred, target)
        l_mr = self.mr_stft_loss(pred, target)
        l_hb = self._highband_loss(pred, target)

        l_mel = self.mel_loss(pred, target) if self.mel_loss is not None else l_mr

        total = (
            self.lambda_l1 * l_time
            + self.lambda_mr_stft * l_mr
            + self.lambda_mel * l_mel
            + self.lambda_highband * l_hb
        )

        return total, {
            "l_time": l_time.item(),
            "l_mr_stft": l_mr.item(),
            "l_mel": l_mel.item(),
            "l_highband": l_hb.item(),
            "total": total.item(),
        }
