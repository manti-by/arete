import torch
import torch.nn as nn
import torch.nn.functional as F


try:
    from auraloss.freq import MelSTFTLoss, MultiResolutionSTFTLoss

    AURALOSS_AVAILABLE = True
except ImportError:
    AURALOSS_AVAILABLE = False


def highband_loss(pred: torch.Tensor, target: torch.Tensor, n_fft: int, hop: int, cutoff_bin: int) -> torch.Tensor:
    p = pred.squeeze(1)
    t = target.squeeze(1)
    window = torch.hann_window(n_fft, device=p.device)
    p_spec = torch.stft(p, n_fft, hop, window=window, return_complex=True).abs()
    t_spec = torch.stft(t, n_fft, hop, window=window, return_complex=True).abs()
    return F.l1_loss(p_spec[:, cutoff_bin:], t_spec[:, cutoff_bin:])


class FallbackMRSTFTLoss(nn.Module):
    def __init__(self, fft_sizes: list[int], hop_sizes: list[int], win_sizes: list[int]) -> None:
        super().__init__()
        self.params = list(zip(fft_sizes, hop_sizes, win_sizes, strict=True))

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = pred.squeeze(1)
        target = target.squeeze(1)
        losses = []
        for n_fft, hop, win in self.params:
            window = torch.hann_window(win, device=pred.device)
            p_spec = torch.stft(pred, n_fft, hop, win, window=window, return_complex=True).abs()
            t_spec = torch.stft(target, n_fft, hop, win, window=window, return_complex=True).abs()
            losses.append(F.l1_loss(p_spec, t_spec))
        return torch.stack(losses).mean()


class CombinedAudioLoss(nn.Module):
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

        hb_n_bins = fft_sizes[-1] // 2 + 1
        self.hb_bin = min(
            int(highband_cutoff_hz / (sample_rate / 2) * hb_n_bins),
            hb_n_bins - 1,
        )
        self.hb_n_fft = fft_sizes[-1]
        self.hb_hop = hop_sizes[-1]

        if AURALOSS_AVAILABLE:
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
            self.mr_stft_loss = FallbackMRSTFTLoss(fft_sizes, hop_sizes, win_sizes)
            self.mel_loss = None

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
        l_time = F.l1_loss(pred, target)
        l_mr = self.mr_stft_loss(pred, target)
        l_hb = highband_loss(pred, target, self.hb_n_fft, self.hb_hop, self.hb_bin)
        l_mel = self.mel_loss(pred, target) if self.mel_loss is not None else l_mr

        total = (
            self.lambda_l1 * l_time + self.lambda_mr_stft * l_mr + self.lambda_mel * l_mel + self.lambda_highband * l_hb
        )

        return total, {
            "l_time": l_time.item(),
            "l_mr_stft": l_mr.item(),
            "l_mel": l_mel.item(),
            "l_highband": l_hb.item(),
            "total": total.item(),
        }
