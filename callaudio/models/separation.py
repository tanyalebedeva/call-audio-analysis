"""Модели отделения речи от фона.

UNetMask   - частотная область: маска на амплитудной спектрограмме STFT, фаза смеси.
ConvTasNet - временная область: обучаемый энкодер вместо STFT, маска в латентном пространстве.

Обе модели принимают смесь (B, T) и возвращают оценку чистой речи (B, T) той же длины,
поэтому для них подходит любая функция потерь из losses.py.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ===================== U-Net на спектрограмме =====================

class _ConvBlock(nn.Module):
    def __init__(self, c_in: int, c_out: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(c_in, c_out, 3, padding=1), nn.BatchNorm2d(c_out), nn.LeakyReLU(0.2),
            nn.Conv2d(c_out, c_out, 3, padding=1), nn.BatchNorm2d(c_out), nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        return self.net(x)


class UNetMask(nn.Module):
    def __init__(self, n_fft: int = 512, hop: int = 128, channels=(16, 32, 64, 128)):
        super().__init__()
        self.n_fft, self.hop = n_fft, hop
        self.register_buffer("window", torch.hann_window(n_fft), persistent=False)
        self.downs = nn.ModuleList()
        c_prev = 1
        for c in channels:
            self.downs.append(_ConvBlock(c_prev, c))
            c_prev = c
        self.bottleneck = _ConvBlock(channels[-1], channels[-1] * 2)
        self.ups = nn.ModuleList()
        self.up_blocks = nn.ModuleList()
        c_prev = channels[-1] * 2
        for c in reversed(channels):
            self.ups.append(nn.ConvTranspose2d(c_prev, c, 2, stride=2))
            self.up_blocks.append(_ConvBlock(c * 2, c))
            c_prev = c
        self.head = nn.Conv2d(channels[0], 1, 1)
        self.factor = 2 ** len(channels)

    def stft(self, wav: torch.Tensor) -> torch.Tensor:
        return torch.stft(wav, self.n_fft, self.hop, window=self.window, return_complex=True)

    def forward(self, mix: torch.Tensor, return_mask: bool = False):
        length = mix.shape[-1]
        with torch.autocast(device_type=mix.device.type, enabled=False):
            spec = self.stft(mix.float())                          # (B, F, T) комплексная
        mag = spec.abs()
        x = torch.log1p(mag).unsqueeze(1)                          # (B, 1, F, T)
        # дополняем до кратного 2^depth, чтобы pooling/upsampling сошлись
        f, t = x.shape[-2:]
        pf, pt = (-f) % self.factor, (-t) % self.factor
        x = F.pad(x, (0, pt, 0, pf))
        skips = []
        for down in self.downs:
            x = down(x)
            skips.append(x)
            x = F.max_pool2d(x, 2)
        x = self.bottleneck(x)
        for up, block, skip in zip(self.ups, self.up_blocks, reversed(skips)):
            x = block(torch.cat([up(x), skip], dim=1))
        mask = torch.sigmoid(self.head(x))[:, 0, :f, :t]           # маска 0..1 той же формы, что mag
        with torch.autocast(device_type=mix.device.type, enabled=False):
            est_spec = spec * mask.float()                         # амплитуда × маска, фаза смеси
            est = torch.istft(est_spec, self.n_fft, self.hop, window=self.window, length=length)
        return (est, mask) if return_mask else est


# ===================== Conv-TasNet =====================

class GlobalLayerNorm(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(1, channels, 1))
        self.beta = nn.Parameter(torch.zeros(1, channels, 1))

    def forward(self, x):
        mean = x.mean(dim=(1, 2), keepdim=True)
        var = ((x - mean) ** 2).mean(dim=(1, 2), keepdim=True)
        return self.gamma * (x - mean) / torch.sqrt(var + 1e-8) + self.beta


class TCNBlock(nn.Module):
    """1×1 conv → PReLU → gLN → depthwise dilated conv → PReLU → gLN → 1×1 conv (остаток и skip)."""

    def __init__(self, bottleneck: int, hidden: int, kernel: int, dilation: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(bottleneck, hidden, 1), nn.PReLU(), GlobalLayerNorm(hidden),
            nn.Conv1d(hidden, hidden, kernel, dilation=dilation, padding=dilation * (kernel - 1) // 2, groups=hidden),
            nn.PReLU(), GlobalLayerNorm(hidden),
        )
        self.res = nn.Conv1d(hidden, bottleneck, 1)
        self.skip = nn.Conv1d(hidden, bottleneck, 1)

    def forward(self, x):
        y = self.net(x)
        return x + self.res(y), self.skip(y)


class ConvTasNet(nn.Module):
    """Уменьшенная Conv-TasNet (Luo & Mesgarani, 2019) с одним выходом - речью.
    Фон получается как смесь минус речь."""

    def __init__(self, n_filters: int = 128, kernel_size: int = 16, bottleneck: int = 128, hidden: int = 256,
                 tcn_kernel: int = 3, blocks: int = 6, repeats: int = 2):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = kernel_size // 2
        self.encoder = nn.Conv1d(1, n_filters, kernel_size, stride=self.stride, bias=False)
        self.norm = GlobalLayerNorm(n_filters)
        self.in_proj = nn.Conv1d(n_filters, bottleneck, 1)
        self.tcn = nn.ModuleList(
            [TCNBlock(bottleneck, hidden, tcn_kernel, 2 ** b) for _ in range(repeats) for b in range(blocks)]
        )
        self.mask = nn.Sequential(nn.PReLU(), nn.Conv1d(bottleneck, n_filters, 1), nn.ReLU())
        self.decoder = nn.ConvTranspose1d(n_filters, 1, kernel_size, stride=self.stride, bias=False)

    def encode(self, mix: torch.Tensor) -> torch.Tensor:
        """Обучаемое латентное представление смеси: (B, T) -> (B, N, frames)."""
        return F.relu(self.encoder(mix.unsqueeze(1)))

    def forward(self, mix: torch.Tensor) -> torch.Tensor:
        length = mix.shape[-1]
        pad = (self.stride - (length - self.kernel_size) % self.stride) % self.stride
        x = F.pad(mix, (0, pad))
        w = self.encode(x)
        y = self.in_proj(self.norm(w))
        skip_sum = 0.0
        for block in self.tcn:
            y, s = block(y)
            skip_sum = skip_sum + s
        m = self.mask(skip_sum)
        est = self.decoder(w * m).squeeze(1)
        return est[..., :length]


def build_separation_model(name: str, **kwargs) -> nn.Module:
    name = name.lower()
    if name == "unet":
        return UNetMask(**kwargs)
    if name in ("convtasnet", "tasnet"):
        return ConvTasNet(**kwargs)
    raise ValueError(f"Неизвестная модель разделения: {name}")
