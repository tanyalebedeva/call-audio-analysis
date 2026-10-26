"""Синтетические сигналы для быстрой проверки кода без скачивания датасетов.

Это не замена реальным данным: синтетика нужна, чтобы за минуту убедиться,
что обучение, чекпойнты, метрики и визуализация работают, — до долгих прогонов в Colab.
"""
from __future__ import annotations

import numpy as np

SR = 16000


def speech_like(rng: np.random.Generator, seconds: float, f0: float = 140.0, f0_var: float = 0.15,
                syllable_rate: float = 4.0, energy: float = 0.3) -> np.ndarray:
    """Гармонический сигнал с плавающей основной частотой и «слоговой» огибающей."""
    n = int(seconds * SR)
    t = np.arange(n) / SR
    contour = f0 * (1 + f0_var * np.sin(2 * np.pi * rng.uniform(0.3, 1.0) * t + rng.uniform(0, 6.28)))
    phase = 2 * np.pi * np.cumsum(contour) / SR
    wav = sum((0.6 ** k) * np.sin((k + 1) * phase) for k in range(6))
    env = 0.5 * (1 + np.sin(2 * np.pi * syllable_rate * t + rng.uniform(0, 6.28)))
    wav = wav * env ** 2
    return (energy * wav / (np.abs(wav).max() + 1e-6)).astype(np.float32)


def noise_like(rng: np.random.Generator, seconds: float, kind: str = "noise") -> np.ndarray:
    n = int(seconds * SR)
    if kind == "music":
        t = np.arange(n) / SR
        notes = rng.uniform(200, 900, size=3)
        wav = sum(np.sin(2 * np.pi * f * t) for f in notes) * (0.6 + 0.4 * np.sin(2 * np.pi * 2 * t))
    else:
        white = rng.standard_normal(n)
        wav = np.cumsum(white)  # «красноватый» шум
        wav = wav - np.convolve(wav, np.ones(200) / 200, mode="same")
    return (0.3 * wav / (np.abs(wav).max() + 1e-6)).astype(np.float32)


# Синтетические «эмоции»: классы различаются высотой, разбросом тона, темпом и громкостью.
EMOTION_PROFILES = {
    0: dict(f0=130, f0_var=0.05, syllable_rate=3.5, energy=0.25),  # neutral
    1: dict(f0=210, f0_var=0.25, syllable_rate=6.0, energy=0.6),   # angry
    2: dict(f0=230, f0_var=0.35, syllable_rate=4.5, energy=0.4),   # positive
    3: dict(f0=110, f0_var=0.08, syllable_rate=2.0, energy=0.15),  # sad
}


def emotion_like(rng: np.random.Generator, label: int, seconds: float) -> np.ndarray:
    p = dict(EMOTION_PROFILES[label % len(EMOTION_PROFILES)])
    p["f0"] *= rng.uniform(0.85, 1.15)
    p["syllable_rate"] *= rng.uniform(0.85, 1.15)
    return speech_like(rng, seconds, **p)


def word_like(rng: np.random.Generator, label: int, seconds: float = 1.0) -> np.ndarray:
    """Синтетическое «слово»: два тона, частоты зависят от класса."""
    n = int(seconds * SR)
    t = np.arange(n) / SR
    f1, f2 = 200 + 60 * label, 700 + 90 * ((label * 7) % 10)
    half = n // 2
    wav = np.concatenate([np.sin(2 * np.pi * f1 * t[:half]), np.sin(2 * np.pi * f2 * t[half:])])
    wav *= np.hanning(n) * rng.uniform(0.3, 0.8)
    return (wav + 0.01 * rng.standard_normal(n)).astype(np.float32)
