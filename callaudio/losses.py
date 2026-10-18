"""Функции потерь всех трёх модулей.

Разделение (все принимают оценку и эталон как waveform (B, T)):
    SpectralL1Loss, NegSISDRLoss, MultiResolutionSTFTLoss
Классификация эмоций:
    cross-entropy (обычная и взвешенная), FocalLoss
VAE:
    vae_loss с выбором MSE / L1 и весом KL (β) с разогревом
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ===================== разделение =====================

def si_sdr(est: torch.Tensor, ref: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Scale-invariant SDR в дБ для каждого примера батча: (B, T) -> (B,).
    Сигналы центрируются, оценка проецируется на эталон - громкость не влияет."""
    est = est - est.mean(dim=-1, keepdim=True)
    ref = ref - ref.mean(dim=-1, keepdim=True)
    alpha = (est * ref).sum(-1, keepdim=True) / (ref.pow(2).sum(-1, keepdim=True) + eps)
    target = alpha * ref
    noise = est - target
    return 10 * torch.log10(target.pow(2).sum(-1) / (noise.pow(2).sum(-1) + eps) + eps)


class NegSISDRLoss(nn.Module):
    def forward(self, est: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        return -si_sdr(est, ref).mean()


def _stft_mag(x: torch.Tensor, n_fft: int, hop: int, win: int) -> torch.Tensor:
    window = torch.hann_window(win, device=x.device, dtype=x.dtype)
    spec = torch.stft(x, n_fft=n_fft, hop_length=hop, win_length=win, window=window, return_complex=True)
    return spec.abs().clamp(min=1e-7)


class SpectralL1Loss(nn.Module):
    """L1 по амплитудному спектру STFT. Игнорирует фазу."""

    def __init__(self, n_fft: int = 512, hop: int = 128):
        super().__init__()
        self.n_fft, self.hop = n_fft, hop

    def forward(self, est: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        with torch.autocast(device_type=est.device.type, enabled=False):
            e = _stft_mag(est.float(), self.n_fft, self.hop, self.n_fft)
            r = _stft_mag(ref.float(), self.n_fft, self.hop, self.n_fft)
            return F.l1_loss(e, r)


class MultiResolutionSTFTLoss(nn.Module):
    """Сумма потерь на нескольких разрешениях STFT:
    спектральная сходимость ||R| - |E||_F / ||R||_F + L1 логарифмов амплитуд."""

    def __init__(self, resolutions=((512, 128, 512), (1024, 256, 1024), (2048, 512, 2048))):
        super().__init__()
        self.resolutions = resolutions

    def forward(self, est: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        total = 0.0
        with torch.autocast(device_type=est.device.type, enabled=False):
            for n_fft, hop, win in self.resolutions:
                e = _stft_mag(est.float(), n_fft, hop, win)
                r = _stft_mag(ref.float(), n_fft, hop, win)
                sc = torch.linalg.norm(r - e, dim=(-2, -1)) / (torch.linalg.norm(r, dim=(-2, -1)) + 1e-7)
                mag = F.l1_loss(torch.log(e), torch.log(r))
                total = total + sc.mean() + mag
        return total / len(self.resolutions)


def build_separation_loss(name: str, **kwargs) -> nn.Module:
    name = name.lower()
    if name in ("l1_spec", "spectral_l1"):
        return SpectralL1Loss(**kwargs)
    if name in ("si_sdr", "sisdr"):
        return NegSISDRLoss()
    if name in ("mrstft", "multi_res_stft"):
        return MultiResolutionSTFTLoss(**kwargs)
    raise ValueError(f"Неизвестная функция потерь разделения: {name}")


# ===================== классификация =====================

class FocalLoss(nn.Module):
    """Focal loss: (1 - p_t)^γ · CE. При γ = 0 совпадает с cross-entropy.
    Уменьшает вклад лёгких примеров - у нас это обычно нейтральные фразы."""

    def __init__(self, gamma: float = 2.0, weight: torch.Tensor | None = None, label_smoothing: float = 0.0):
        super().__init__()
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.register_buffer("weight", weight if weight is not None else None, persistent=False)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, target, weight=self.weight, reduction="none", label_smoothing=self.label_smoothing)
        p_t = torch.softmax(logits, dim=-1).gather(1, target[:, None]).squeeze(1)
        return ((1 - p_t) ** self.gamma * ce).mean()


def class_weights(labels: list[int], num_classes: int) -> torch.Tensor:
    """Веса обратно пропорциональны частоте класса, в среднем равны 1."""
    counts = torch.bincount(torch.tensor(labels), minlength=num_classes).float().clamp(min=1)
    w = counts.sum() / (num_classes * counts)
    return w / w.mean()


def build_classification_loss(name: str, labels: list[int] | None = None, num_classes: int | None = None,
                              gamma: float = 2.0, label_smoothing: float = 0.0) -> nn.Module:
    name = name.lower()
    weight = None
    if name in ("weighted_ce", "focal_weighted") and labels is not None:
        weight = class_weights(labels, num_classes)
    if name in ("ce", "weighted_ce"):
        return nn.CrossEntropyLoss(weight=weight, label_smoothing=label_smoothing)
    if name in ("focal", "focal_weighted"):
        return FocalLoss(gamma=gamma, weight=weight, label_smoothing=label_smoothing)
    raise ValueError(f"Неизвестная функция потерь классификации: {name}")


# ===================== VAE =====================

def kl_divergence(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """KL(q(z|x) || N(0, I)) для каждого примера, суммой по размерностям латента."""
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1)


def vae_loss(recon: torch.Tensor, target: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor,
             beta: float = 1.0, recon_type: str = "mse") -> tuple[torch.Tensor, dict]:
    """Потеря VAE = реконструкция (сумма по пикселям) + β · KL, среднее по батчу."""
    if recon_type == "mse":
        rec = F.mse_loss(recon, target, reduction="none")
    elif recon_type == "l1":
        rec = F.l1_loss(recon, target, reduction="none")
    else:
        raise ValueError(recon_type)
    rec = rec.flatten(1).sum(-1).mean()
    kl = kl_divergence(mu, logvar).mean()
    return rec + beta * kl, {"recon": rec.item(), "kl": kl.item(), "beta": beta}


def beta_schedule(epoch: int, step_in_epoch: float, beta: float, warmup_epochs: float) -> float:
    """Линейный разогрев β от 0 до целевого значения (KL annealing) - против коллапса латента."""
    if warmup_epochs <= 0:
        return beta
    progress = (epoch + step_in_epoch) / warmup_epochs
    return beta * min(1.0, progress)
