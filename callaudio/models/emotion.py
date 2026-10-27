"""Классификаторы эмоций на четырёх представлениях аудио.

WaveCNN         - сырой сигнал (waveform), 1D-свёртки, сеть сама учит фильтры.
MelCNN          - лог-мел-спектрограмма, 2D-свёртки + SpecAugment.
SSLClassifier   - скрытые слои самообученной модели (HuBERT / wav2vec2) с обучаемыми весами слоёв.
LatentClassifier - латентные векторы VAE из модуля 3.

У каждой модели есть forward(wav) -> логиты и embed(wav) -> эмбеддинг предпоследнего слоя
(он нужен для визуализации пространства признаков по эпохам).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..audio import EMOTION_MEL, LogMel, vocoder_mel


class AttentiveStatsPooling(nn.Module):
    """Взвешенное среднее и стандартное отклонение по времени: (B, C, T) -> (B, 2C).
    Модель сама решает, какие кадры важнее для эмоции (например, ударные слоги)."""

    def __init__(self, channels: int, hidden: int = 128):
        super().__init__()
        self.attn = nn.Sequential(nn.Conv1d(channels, hidden, 1), nn.Tanh(), nn.Conv1d(hidden, channels, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = torch.softmax(self.attn(x), dim=-1)
        mean = (x * w).sum(-1)
        std = torch.sqrt(((x - mean[..., None]) ** 2 * w).sum(-1).clamp(min=1e-6))
        return torch.cat([mean, std], dim=-1)


class _Head(nn.Module):
    def __init__(self, in_dim: int, emb_dim: int, num_classes: int, dropout: float):
        super().__init__()
        self.embed = nn.Sequential(nn.Linear(in_dim, emb_dim), nn.BatchNorm1d(emb_dim), nn.ReLU(), nn.Dropout(dropout))
        self.classifier = nn.Linear(emb_dim, num_classes)


# ===================== waveform =====================

class WaveCNN(nn.Module):
    """В духе M5 (Dai et al., 2017): широкое первое ядро ~5 мс играет роль банка фильтров."""

    def __init__(self, num_classes: int = 4, channels=(64, 128, 128, 256), emb_dim: int = 128, dropout: float = 0.3):
        super().__init__()
        layers = [nn.Conv1d(1, channels[0], 80, stride=4), nn.BatchNorm1d(channels[0]), nn.ReLU(), nn.MaxPool1d(4)]
        c_prev = channels[0]
        for c in channels[1:]:
            layers += [nn.Conv1d(c_prev, c, 3, padding=1), nn.BatchNorm1d(c), nn.ReLU(), nn.MaxPool1d(4)]
            c_prev = c
        self.features = nn.Sequential(*layers)
        self.pool = AttentiveStatsPooling(c_prev)
        self.head = _Head(2 * c_prev, emb_dim, num_classes, dropout)

    def embed(self, wav: torch.Tensor) -> torch.Tensor:
        wav = wav / (wav.abs().amax(dim=-1, keepdim=True) + 1e-6)
        return self.head.embed(self.pool(self.features(wav.unsqueeze(1))))

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        return self.head.classifier(self.embed(wav))


# ===================== мел-спектрограмма =====================

class SpecAugment(nn.Module):
    """Маскирование случайных полос частот и отрезков времени - только при обучении."""

    def __init__(self, freq_mask: int = 8, time_mask: int = 40, n_masks: int = 2):
        super().__init__()
        self.freq_mask, self.time_mask, self.n_masks = freq_mask, time_mask, n_masks

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, F, T)
        if not self.training:
            return x
        x = x.clone()
        b, f, t = x.shape
        for _ in range(self.n_masks):
            fw = torch.randint(0, self.freq_mask + 1, (b,), device=x.device)
            f0 = (torch.rand(b, device=x.device) * (f - fw)).long()
            tw = torch.randint(0, self.time_mask + 1, (b,), device=x.device)
            t0 = (torch.rand(b, device=x.device) * (t - tw).clamp(min=1)).long()
            fr = torch.arange(f, device=x.device)[None]
            tr = torch.arange(t, device=x.device)[None]
            fmask = (fr >= f0[:, None]) & (fr < (f0 + fw)[:, None])
            tmask = (tr >= t0[:, None]) & (tr < (t0 + tw)[:, None])
            x = x.masked_fill(fmask[:, :, None] | tmask[:, None, :], 0.0)
        return x


class _ResBlock(nn.Module):
    def __init__(self, c_in: int, c_out: int, stride: int):
        super().__init__()
        self.conv1 = nn.Conv2d(c_in, c_out, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(c_out)
        self.conv2 = nn.Conv2d(c_out, c_out, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(c_out)
        self.short = (nn.Sequential(nn.Conv2d(c_in, c_out, 1, stride=stride, bias=False), nn.BatchNorm2d(c_out))
                      if stride != 1 or c_in != c_out else nn.Identity())

    def forward(self, x):
        y = F.relu(self.bn1(self.conv1(x)))
        return F.relu(self.bn2(self.conv2(y)) + self.short(x))


class MelCNN(nn.Module):
    def __init__(self, num_classes: int = 4, channels=(32, 64, 128, 256), emb_dim: int = 128,
                 dropout: float = 0.3, spec_augment: bool = True):
        super().__init__()
        self.frontend = LogMel(EMOTION_MEL)
        self.norm = nn.InstanceNorm1d(EMOTION_MEL["n_mels"])  # нормировка каждой мел-полосы по времени
        self.augment = SpecAugment() if spec_augment else nn.Identity()
        layers, c_prev = [], 1
        for i, c in enumerate(channels):
            layers.append(_ResBlock(c_prev, c, stride=1 if i == 0 else 2))
            c_prev = c
        self.body = nn.Sequential(*layers)
        self.pool = AttentiveStatsPooling(c_prev)
        self.head = _Head(2 * c_prev, emb_dim, num_classes, dropout)

    def embed(self, wav: torch.Tensor) -> torch.Tensor:
        x = self.augment(self.norm(self.frontend(wav)))       # (B, 64, T)
        h = self.body(x.unsqueeze(1)).mean(dim=2)             # усредняем по частоте -> (B, C, T')
        return self.head.embed(self.pool(h))

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        return self.head.classifier(self.embed(wav))


# ===================== самообученная модель =====================

class SSLClassifier(nn.Module):
    """Голова поверх HuBERT / wav2vec2.

    Скрытые состояния всех слоёв складываются с обучаемыми весами (softmax): после обучения
    веса показывают, на какой глубине модели лучше всего закодирована эмоция.
    freeze=True - обучаем только голову; unfreeze_top=N - дообучаем ещё N верхних слоёв."""

    def __init__(self, num_classes: int = 4, pretrained: str = "facebook/hubert-base-ls960",
                 freeze: bool = True, unfreeze_top: int = 0, emb_dim: int = 256, dropout: float = 0.3):
        super().__init__()
        from transformers import AutoModel

        self.backbone = AutoModel.from_pretrained(pretrained, layerdrop=0.0)
        if hasattr(self.backbone, "feature_extractor"):
            self.backbone.feature_extractor._freeze_parameters()  # CNN-кодировщик сигнала всегда заморожен
        n_layers = self.backbone.config.num_hidden_layers
        hidden = self.backbone.config.hidden_size
        for p in self.backbone.parameters():
            p.requires_grad = not freeze
        if freeze and unfreeze_top > 0:
            for layer in self.backbone.encoder.layers[-unfreeze_top:]:
                for p in layer.parameters():
                    p.requires_grad = True
        if hasattr(self.backbone, "feature_extractor"):
            for p in self.backbone.feature_extractor.parameters():
                p.requires_grad = False
        self.layer_weights = nn.Parameter(torch.zeros(n_layers + 1))
        self.pool = AttentiveStatsPooling(hidden)
        self.head = _Head(2 * hidden, emb_dim, num_classes, dropout)

    def layer_importance(self) -> torch.Tensor:
        return torch.softmax(self.layer_weights.detach(), dim=0)

    def embed(self, wav: torch.Tensor) -> torch.Tensor:
        wav = (wav - wav.mean(-1, keepdim=True)) / (wav.std(-1, keepdim=True) + 1e-6)
        out = self.backbone(wav, output_hidden_states=True)
        hs = torch.stack(out.hidden_states, dim=0)                     # (L+1, B, T, D)
        w = torch.softmax(self.layer_weights[: hs.shape[0]], dim=0)[:, None, None, None]
        h = (w * hs).sum(0).transpose(1, 2)                            # (B, D, T)
        return self.head.embed(self.pool(h))

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        return self.head.classifier(self.embed(wav))


# ===================== латенты VAE =====================

class LatentClassifier(nn.Module):
    """Режет запись на окна по 64 кадра (~1 с), кодирует каждое замороженным VAE
    и классифицирует среднее и разброс латентов по окнам."""

    def __init__(self, vae: nn.Module, mel_mean: float, mel_std: float, num_classes: int = 4,
                 emb_dim: int = 128, dropout: float = 0.3):
        super().__init__()
        from .vae import N_FRAMES

        self.vae = vae.eval()
        for p in self.vae.parameters():
            p.requires_grad = False
        self.n_frames = N_FRAMES
        self.mel_mean, self.mel_std = mel_mean, mel_std
        self.head = _Head(2 * vae.latent_dim, emb_dim, num_classes, dropout)

    def train(self, mode: bool = True):
        super().train(mode)
        self.vae.eval()  # VAE всегда в режиме инференса (BatchNorm не обновляется)
        return self

    def latents(self, wav: torch.Tensor) -> torch.Tensor:
        mel = (vocoder_mel(wav) - self.mel_mean) / self.mel_std           # (B, 80, T)
        t = mel.shape[-1] // self.n_frames * self.n_frames
        if t == 0:
            mel = F.pad(mel, (0, self.n_frames - mel.shape[-1]))
            t = self.n_frames
        windows = mel[..., :t].unfold(-1, self.n_frames, self.n_frames)    # (B, 80, W, 64)
        b, f, n_w, fr = windows.shape
        windows = windows.permute(0, 2, 1, 3).reshape(b * n_w, f, fr)
        with torch.no_grad():
            mu, _ = self.vae.encode(windows)
        return mu.view(b, n_w, -1)

    def embed(self, wav: torch.Tensor) -> torch.Tensor:
        z = self.latents(wav)
        stats = torch.cat([z.mean(1), z.std(1, unbiased=False)], dim=-1)
        return self.head.embed(stats)

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        return self.head.classifier(self.embed(wav))
