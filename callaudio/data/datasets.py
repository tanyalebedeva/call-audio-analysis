"""Датасеты PyTorch для трёх модулей.

Все читают манифесты JSONL (см. data/prepare.py) или генерируют синтетику (synthetic=True).
"""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from ..audio import SAMPLE_RATE, fix_length, load_audio, mix_at_snr, telephone_channel, vocoder_mel
from ..utils import read_manifest
from . import synthetic as syn


def _rng(deterministic: bool, seed: int, index: int) -> np.random.Generator:
    """Для теста — фиксированный генератор на каждый индекс (смеси одинаковы при каждом запуске),
    для обучения — новый случайный (каждая эпоха видит новые смеси)."""
    if deterministic:
        return np.random.default_rng(seed * 1_000_003 + index)
    return np.random.default_rng()


def _load_segment(row: dict, seconds: float | None, rng: np.random.Generator, loop: bool = False) -> np.ndarray:
    """Случайный кусок длиной seconds (или весь файл, если seconds=None)."""
    if seconds is None:
        return load_audio(row["path"])
    length = int(seconds * SAMPLE_RATE)
    duration = float(row.get("duration", 0) or 0)
    start = rng.uniform(0, duration - seconds) if duration > seconds else 0.0
    wav = load_audio(row["path"], start=start, duration=seconds)
    if loop and 0 < len(wav) < length:
        wav = np.tile(wav, int(np.ceil(length / len(wav))))
    return fix_length(wav, length)


# ===================== модуль 1: смеси речи и фона =====================

class SeparationDataset(Dataset):
    """Смешивает речь и фон «на лету» с заданным SNR.

    segment_sec=None — целые высказывания (для оценки и WER, batch_size=1).
    snr_grid — при deterministic=True SNR берётся по кругу из сетки, чтобы корзины SNR были равными.
    """

    def __init__(self, speech_manifest: str | None = None, noise_manifest: str | None = None,
                 segment_sec: float | None = 4.0, snr_range=(-5.0, 15.0), snr_grid=None,
                 epoch_size: int | None = None, deterministic: bool = False, seed: int = 0,
                 telephone_prob: float = 0.0, max_sec: float = 20.0, synthetic: bool = False):
        self.synthetic = synthetic
        self.speech = [] if synthetic else read_manifest(speech_manifest)
        self.noise = [] if synthetic else read_manifest(noise_manifest)
        self.segment_sec = segment_sec
        self.snr_range = snr_range
        self.snr_grid = list(snr_grid) if snr_grid else None
        self.deterministic = deterministic
        self.seed = seed
        self.telephone_prob = telephone_prob
        self.max_sec = max_sec
        self.epoch_size = epoch_size or (len(self.speech) if not synthetic else 256)

    def __len__(self) -> int:
        return self.epoch_size

    def _speech(self, rng, index):
        if self.synthetic:
            return syn.speech_like(rng, self.segment_sec or 3.0, f0=rng.uniform(90, 250)), ""
        row = self.speech[index % len(self.speech)] if self.deterministic else self.speech[rng.integers(len(self.speech))]
        seconds = self.segment_sec
        if seconds is None and float(row.get("duration", 0) or 0) > self.max_sec:
            seconds = self.max_sec
        for _ in range(5):  # пропускаем почти беззвучные куски
            wav = _load_segment(row, seconds, rng)
            if np.sqrt(np.mean(wav ** 2)) > 1e-3 or self.deterministic:
                break
        return wav, row.get("text") or ""

    def _noise(self, rng, length):
        if self.synthetic:
            kind = "music" if rng.random() < 0.5 else "noise"
            return syn.noise_like(rng, length / SAMPLE_RATE, kind), kind
        row = self.noise[rng.integers(len(self.noise))]
        wav = _load_segment(row, length / SAMPLE_RATE, rng, loop=True)
        return wav, row.get("kind", "noise")

    def __getitem__(self, index: int) -> dict:
        rng = _rng(self.deterministic, self.seed, index)
        speech, text = self._speech(rng, index)
        noise, kind = self._noise(rng, len(speech))
        if self.snr_grid:
            snr = float(self.snr_grid[index % len(self.snr_grid)])
        else:
            snr = float(rng.uniform(*self.snr_range))
        mix, _ = mix_at_snr(speech, noise, snr)
        if self.telephone_prob > 0 and rng.random() < self.telephone_prob:
            mix, speech = telephone_channel(mix), telephone_channel(speech)
        peak = np.abs(mix).max()
        if peak > 0.95:  # общий масштаб, чтобы смесь не клиппировала
            mix, speech = mix * 0.95 / peak, speech * 0.95 / peak
        return {
            "mix": torch.from_numpy(mix.astype(np.float32)),
            "clean": torch.from_numpy(speech.astype(np.float32)),
            "snr": snr, "noise_kind": kind, "text": text,
        }


# ===================== модуль 2: эмоции =====================

class EmotionDataset(Dataset):
    """Записи с метками эмоций. Возвращает waveform фиксированной длины max_sec.

    noise_manifest + noise_prob — аугментация шумом при обучении;
    fixed_snr — оценка устойчивости: каждая запись зашумляется с этим SNR (детерминированно)."""

    def __init__(self, manifest: str | None = None, labels=("neutral", "angry", "positive", "sad"),
                 max_sec: float = 4.0, train: bool = True, noise_manifest: str | None = None,
                 noise_prob: float = 0.0, snr_range=(0.0, 20.0), fixed_snr: float | None = None,
                 seed: int = 0, synthetic: bool = False, synthetic_size: int = 512):
        self.labels = list(labels)
        self.label2id = {l: i for i, l in enumerate(self.labels)}
        self.synthetic = synthetic
        if synthetic:
            rng = np.random.default_rng(seed)
            self.rows = [{"label": self.labels[int(rng.integers(len(self.labels)))], "seed": int(i)}
                         for i in range(synthetic_size)]
        else:
            self.rows = [r for r in read_manifest(manifest) if r["label"] in self.label2id]
        self.length = int(max_sec * SAMPLE_RATE)
        self.max_sec = max_sec
        self.train = train
        self.noise = read_manifest(noise_manifest) if noise_manifest else []
        self.noise_prob = noise_prob
        self.snr_range = snr_range
        self.fixed_snr = fixed_snr
        self.seed = seed

    def __len__(self) -> int:
        return len(self.rows)

    def targets(self) -> list[int]:
        return [self.label2id[r["label"]] for r in self.rows]

    def _load(self, row, rng) -> np.ndarray:
        if self.synthetic:
            r = np.random.default_rng(row["seed"] + self.seed * 7919)
            return syn.emotion_like(r, self.label2id[row["label"]], self.max_sec)
        wav = load_audio(row["path"])
        if len(wav) > self.length:
            offset = int(rng.integers(0, len(wav) - self.length)) if self.train else (len(wav) - self.length) // 2
            return wav[offset:offset + self.length]
        return fix_length(wav, self.length)

    def __getitem__(self, index: int) -> dict:
        row = self.rows[index]
        rng = _rng(not self.train, self.seed, index)
        wav = self._load(row, rng)
        add_noise = self.noise and (self.fixed_snr is not None or (self.train and rng.random() < self.noise_prob))
        if add_noise:
            nrow = self.noise[rng.integers(len(self.noise))]
            noise = _load_segment(nrow, self.max_sec, rng, loop=True)
            snr = self.fixed_snr if self.fixed_snr is not None else rng.uniform(*self.snr_range)
            if np.sqrt(np.mean(wav ** 2)) > 1e-4:
                wav, _ = mix_at_snr(wav, noise, snr)
        return {"wav": torch.from_numpy(wav.astype(np.float32)),
                "label": self.label2id[row["label"]], "index": index}


# ===================== модуль 3: короткие слова для VAE =====================

class CommandsMelDataset(Dataset):
    """Односекундные слова -> лог-мел 80 × 64 в формате вокодера HiFi-GAN."""

    PAD_VALUE = float(np.log(1e-5))

    def __init__(self, manifest: str | None = None, words=None, synthetic: bool = False,
                 synthetic_size: int = 512, seed: int = 0):
        from ..models.vae import N_FRAMES

        self.n_frames = N_FRAMES
        self.synthetic = synthetic
        if synthetic:
            self.words = list(words or [f"w{i}" for i in range(6)])
            rng = np.random.default_rng(seed)
            self.rows = [{"label": self.words[int(rng.integers(len(self.words)))], "seed": i}
                         for i in range(synthetic_size)]
        else:
            rows = read_manifest(manifest)
            self.words = list(words) if words else sorted({r["label"] for r in rows})
            self.rows = [r for r in rows if r["label"] in self.words]
        self.word2id = {w: i for i, w in enumerate(self.words)}
        self.seed = seed

    def __len__(self) -> int:
        return len(self.rows)

    def wav(self, index: int) -> np.ndarray:
        row = self.rows[index]
        if self.synthetic:
            return syn.word_like(np.random.default_rng(row["seed"] + self.seed * 7919), self.word2id[row["label"]])
        return fix_length(load_audio(row["path"]), SAMPLE_RATE)

    def __getitem__(self, index: int) -> dict:
        wav = torch.from_numpy(self.wav(index))
        mel = vocoder_mel(wav[None])[0]                       # (80, 63)
        if mel.shape[-1] < self.n_frames:
            mel = torch.nn.functional.pad(mel, (0, self.n_frames - mel.shape[-1]), value=self.PAD_VALUE)
        return {"mel": mel[:, : self.n_frames], "label": self.word2id[self.rows[index]["label"]], "index": index}
