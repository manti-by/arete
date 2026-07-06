from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel: int = 15, groups: int = 1) -> None:
        super().__init__()
        pad = kernel // 2
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel, padding=pad, groups=groups, bias=False),
            nn.GroupNorm(min(8, out_ch), out_ch),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class EncoderBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel: int = 15, stride: int = 2) -> None:
        super().__init__()
        self.conv = ConvBlock(in_ch, out_ch, kernel)
        self.down = nn.Conv1d(out_ch, out_ch, kernel_size=stride, stride=stride, bias=False)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        skip = self.conv(x)
        return self.down(skip), skip


class DecoderBlock(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, kernel: int = 15, stride: int = 2) -> None:
        super().__init__()
        self.up = nn.ConvTranspose1d(in_ch, in_ch, kernel_size=stride, stride=stride, bias=False)
        self.conv = ConvBlock(in_ch + skip_ch, out_ch, kernel)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[-1] != skip.shape[-1]:
            x = F.pad(x, (0, skip.shape[-1] - x.shape[-1]))
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class WaveformUNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 32,
        depth: int = 5,
        kernel_size: int = 15,
        stride: int = 2,
    ) -> None:
        super().__init__()
        self.depth = depth
        ch = [base_channels * (2**i) for i in range(depth)]

        self.encoders = nn.ModuleList()
        self.encoders.append(EncoderBlock(in_channels, ch[0], kernel_size, stride))
        for i in range(1, depth):
            self.encoders.append(EncoderBlock(ch[i - 1], ch[i], kernel_size, stride))

        self.bottleneck = ConvBlock(ch[-1], ch[-1], kernel_size)

        self.decoders = nn.ModuleList()
        for i in range(depth - 1, 0, -1):
            self.decoders.append(DecoderBlock(ch[i], ch[i - 1], ch[i - 1], kernel_size, stride))
        self.decoders.append(DecoderBlock(ch[0], in_channels, in_channels, kernel_size, stride))

        self.out_conv = nn.Conv1d(in_channels, in_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips: list[torch.Tensor] = []
        h = x

        for enc in self.encoders:
            h, skip = enc(h)
            skips.append(skip)

        h = self.bottleneck(h)

        skip_inputs = [*list(reversed(skips[:-1])), x]
        for dec, skip in zip(self.decoders, skip_inputs, strict=True):
            h = dec(h, skip)

        residual = self.out_conv(h)
        if residual.shape[-1] != x.shape[-1]:
            residual = F.pad(residual, (0, x.shape[-1] - residual.shape[-1]))

        return x + residual
