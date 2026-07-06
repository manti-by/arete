from __future__ import annotations

import torch
import torch.nn as nn
import torchaudio.transforms as T


class DoubleConv2d(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class STFTUNet(nn.Module):
    def __init__(
        self,
        n_fft: int = 2048,
        hop_length: int = 512,
        sample_rate: int = 44100,
        base_ch: int = 32,
        depth: int = 4,
    ) -> None:
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length

        self.stft = T.Spectrogram(n_fft=n_fft, hop_length=hop_length, power=None)
        self.griffin_lim = T.GriffinLim(n_fft=n_fft, hop_length=hop_length)

        self.depth = depth
        enc_chs = [1] + [base_ch * (2**i) for i in range(depth)]
        self.encoders = nn.ModuleList([DoubleConv2d(enc_chs[i], enc_chs[i + 1]) for i in range(depth)])
        self.pool = nn.MaxPool2d(2)

        self.bottleneck = DoubleConv2d(enc_chs[-1], enc_chs[-1] * 2)

        dec_chs = list(reversed(enc_chs[1:]))
        self.up_convs = nn.ModuleList(
            [nn.ConvTranspose2d(enc_chs[-1] * 2 if i == 0 else dec_chs[i - 1], dec_chs[i], 2, 2) for i in range(depth)]
        )
        self.decoders = nn.ModuleList([DoubleConv2d(dec_chs[i] + dec_chs[i], dec_chs[i]) for i in range(depth)])

        self.out_conv = nn.Conv2d(dec_chs[-1], 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        *_, n_t = x.shape
        wav = x.squeeze(1)

        spec_complex = self.stft(wav)
        mag = spec_complex.abs()
        phase = spec_complex.angle()

        log_mag = torch.log1p(mag).unsqueeze(1)

        skips: list[torch.Tensor] = []
        h = log_mag
        for enc in self.encoders:
            h = enc(h)
            skips.append(h)
            h = self.pool(h)

        h = self.bottleneck(h)

        for up, dec, skip in zip(self.up_convs, self.decoders, reversed(skips), strict=True):
            h = up(h)
            dy = skip.shape[2] - h.shape[2]
            dx = skip.shape[3] - h.shape[3]
            if dy > 0 or dx > 0:
                h = torch.nn.functional.pad(h, (0, dx, 0, dy))
            h = torch.cat([h, skip], dim=1)
            h = dec(h)

        pred_log_mag = self.out_conv(h).squeeze(1)
        pred_mag = torch.expm1(pred_log_mag.clamp(min=-10))

        pred_complex = torch.polar(pred_mag, phase)
        restored = torch.istft(
            pred_complex,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            length=n_t,
        )
        return restored.unsqueeze(1)
