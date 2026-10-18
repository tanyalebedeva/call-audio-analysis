"""Свёрточный вариационный автоэнкодер на лог-мел-спектрограммах (80 × 64 кадра ≈ 1 с).

Если задан num_classes > 0, модель становится условной (CVAE): метка слова подаётся
и в энкодер, и в декодер. Тогда можно заказать слово и менять манеру произнесения через z.
"""
from __future__ import annotations

import torch
import torch.nn as nn

N_MELS, N_FRAMES = 80, 64


class ConvVAE(nn.Module):
    def __init__(self, latent_dim: int = 32, channels=(32, 64, 128, 256), num_classes: int = 0, class_dim: int = 16):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        enc, c_prev = [], 1
        for c in channels:
            enc += [nn.Conv2d(c_prev, c, 4, stride=2, padding=1), nn.BatchNorm2d(c), nn.LeakyReLU(0.2)]
            c_prev = c
        self.encoder = nn.Sequential(*enc)
        self.feat_shape = (channels[-1], N_MELS // 2 ** len(channels), N_FRAMES // 2 ** len(channels))  # (256, 5, 4)
        feat_dim = channels[-1] * self.feat_shape[1] * self.feat_shape[2]
        cond = class_dim if num_classes > 0 else 0
        self.class_emb = nn.Embedding(num_classes, class_dim) if num_classes > 0 else None
        self.fc_mu = nn.Linear(feat_dim + cond, latent_dim)
        self.fc_logvar = nn.Linear(feat_dim + cond, latent_dim)
        self.fc_dec = nn.Linear(latent_dim + cond, feat_dim)
        dec = []
        rev = list(reversed(channels))
        for c_in, c_out in zip(rev, rev[1:] + [rev[-1]]):
            dec += [nn.ConvTranspose2d(c_in, c_out, 4, stride=2, padding=1), nn.BatchNorm2d(c_out), nn.LeakyReLU(0.2)]
        self.decoder = nn.Sequential(*dec, nn.Conv2d(rev[-1], 1, 3, padding=1))

    def _cond(self, labels: torch.Tensor | None):
        if self.class_emb is None:
            return None
        if labels is None:
            raise ValueError("Условному VAE нужны метки классов")
        return self.class_emb(labels)

    def encode(self, x: torch.Tensor, labels: torch.Tensor | None = None):
        """x: (B, 80, 64) нормированная лог-мел -> (mu, logvar)."""
        h = self.encoder(x.unsqueeze(1)).flatten(1)
        c = self._cond(labels)
        if c is not None:
            h = torch.cat([h, c], dim=-1)
        return self.fc_mu(h), self.fc_logvar(h).clamp(-10, 10)

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Трюк репараметризации: z = mu + σ·ε, градиент проходит через mu и σ."""
        return mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)

    def decode(self, z: torch.Tensor, labels: torch.Tensor | None = None) -> torch.Tensor:
        c = self._cond(labels)
        if c is not None:
            z = torch.cat([z, c], dim=-1)
        h = self.fc_dec(z).view(-1, *self.feat_shape)
        return self.decoder(h).squeeze(1)

    def forward(self, x: torch.Tensor, labels: torch.Tensor | None = None):
        mu, logvar = self.encode(x, labels)
        z = self.reparameterize(mu, logvar) if self.training else mu
        return self.decode(z, labels), mu, logvar

    @torch.no_grad()
    def sample(self, n: int, labels: torch.Tensor | None = None, device="cpu") -> torch.Tensor:
        z = torch.randn(n, self.latent_dim, device=device)
        return self.decode(z, labels)

    @torch.no_grad()
    def interpolate(self, x_a: torch.Tensor, x_b: torch.Tensor, steps: int = 8,
                    label_a=None, label_b=None) -> torch.Tensor:
        """Линейная интерполяция между двумя спектрограммами в латентном пространстве."""
        mu_a, _ = self.encode(x_a[None], label_a)
        mu_b, _ = self.encode(x_b[None], label_b)
        alphas = torch.linspace(0, 1, steps, device=x_a.device)[:, None]
        z = (1 - alphas) * mu_a + alphas * mu_b
        labels = None
        if self.class_emb is not None:
            labels = torch.where(alphas[:, 0] < 0.5, label_a.expand(steps), label_b.expand(steps))
        return self.decode(z, labels)
