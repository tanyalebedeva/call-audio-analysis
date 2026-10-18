"""Загрузка аудио и представления: waveform, STFT, лог-мел-спектрограмма."""
from __future__ import annotations

import shutil
import subprocess

import numpy as np
import torch
import torch.nn as nn

SAMPLE_RATE = 16000

# Параметры мел-спектрограммы, совпадающие с вокодером
# speechbrain/tts-hifigan-libritts-16kHz — чтобы спектрограммы VAE можно было озвучить.
VOCODER_MEL = dict(
    sample_rate=SAMPLE_RATE, n_fft=1024, win_length=1024, hop_length=256,
    f_min=0.0, f_max=8000.0, n_mels=80, power=1.0, norm="slaney", mel_scale="slaney",
)

# Мел-спектрограмма для классификатора эмоций: окно 25 мс, шаг 10 мс, 64 фильтра.
EMOTION_MEL = dict(
    sample_rate=SAMPLE_RATE, n_fft=400, win_length=400, hop_length=160,
    f_min=20.0, f_max=8000.0, n_mels=64, power=2.0,
)


def resample(wav: np.ndarray, orig_sr: int, target_sr: int = SAMPLE_RATE) -> np.ndarray:
    if orig_sr == target_sr:
        return wav
    try:
        import torchaudio.functional as AF

        return AF.resample(torch.from_numpy(wav).float(), orig_sr, target_sr).numpy()
    except Exception:
        from math import gcd

        from scipy.signal import resample_poly

        g = gcd(orig_sr, target_sr)
        return resample_poly(wav, target_sr // g, orig_sr // g).astype(np.float32)


def _load_with_ffmpeg(path: str, sr: int) -> np.ndarray:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("Для mp3/ogg/m4a нужен ffmpeg (в Colab он уже установлен).")
    cmd = ["ffmpeg", "-nostdin", "-v", "error", "-i", str(path), "-f", "f32le", "-ac", "1", "-ar", str(sr), "-"]
    out = subprocess.run(cmd, capture_output=True, check=True).stdout
    return np.frombuffer(out, dtype=np.float32).copy()


def load_audio(path: str, sr: int = SAMPLE_RATE, start: float | None = None, duration: float | None = None) -> np.ndarray:
    """Возвращает моно float32 в [-1, 1] с частотой sr.
    wav/flac читаются через soundfile (можно читать кусок), остальное — через ffmpeg."""
    import soundfile as sf

    try:
        info = sf.info(str(path))
        frame_start = int((start or 0.0) * info.samplerate)
        frames = int(duration * info.samplerate) if duration is not None else -1
        wav, file_sr = sf.read(str(path), start=frame_start, frames=frames, dtype="float32", always_2d=True)
        wav = wav.mean(axis=1)
        return resample(wav, file_sr, sr).astype(np.float32)
    except Exception:
        wav = _load_with_ffmpeg(path, sr)
        if start is not None or duration is not None:
            s = int((start or 0.0) * sr)
            e = s + int(duration * sr) if duration is not None else None
            wav = wav[s:e]
        return wav


def fix_length(wav: np.ndarray, length: int, offset: int = 0) -> np.ndarray:
    """Обрезает с позиции offset или дополняет нулями до length отсчётов."""
    if len(wav) >= length:
        return wav[offset:offset + length]
    return np.pad(wav, (0, length - len(wav)))


def rms(x: torch.Tensor | np.ndarray, eps: float = 1e-8):
    if isinstance(x, np.ndarray):
        return float(np.sqrt(np.mean(x ** 2) + eps))
    return torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + eps)


def mix_at_snr(speech: np.ndarray, noise: np.ndarray, snr_db: float) -> tuple[np.ndarray, np.ndarray]:
    """Масштабирует шум так, чтобы отношение сигнал/шум было ровно snr_db.
    Возвращает (смесь, масштабированный шум)."""
    gain = rms(speech) / (rms(noise) * 10 ** (snr_db / 20))
    noise = noise * gain
    return speech + noise, noise


def telephone_channel(wav: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Грубая имитация телефонного канала: полоса до 4 кГц (через 8 кГц) и μ-law-квантование."""
    narrow = resample(wav, sr, 8000)
    t = torch.from_numpy(narrow).float().clamp(-1, 1)
    mu = 255.0
    enc = torch.sign(t) * torch.log1p(mu * t.abs()) / np.log1p(mu)
    enc = torch.round((enc + 1) / 2 * mu) / mu * 2 - 1
    dec = torch.sign(enc) * ((1 + mu) ** enc.abs() - 1) / mu
    return resample(dec.numpy(), 8000, sr)[: len(wav)].astype(np.float32)


class LogMel(nn.Module):
    """Лог-мел-спектрограмма как слой сети: считается на GPU вместе с моделью."""

    def __init__(self, params: dict | None = None, log_floor: float = 1e-5):
        super().__init__()
        import torchaudio.transforms as T

        self.mel = T.MelSpectrogram(**(params or EMOTION_MEL))
        self.log_floor = log_floor

    def forward(self, wav: torch.Tensor) -> torch.Tensor:  # (B, T) -> (B, n_mels, frames)
        with torch.autocast(device_type=wav.device.type, enabled=False):
            return torch.log(torch.clamp(self.mel(wav.float()), min=self.log_floor))


def vocoder_mel(wav: torch.Tensor) -> torch.Tensor:
    """Мел-спектрограмма в формате HiFi-GAN SpeechBrain: (B, T) -> (B, 80, frames)."""
    import torchaudio.transforms as T

    mel = T.MelSpectrogram(**VOCODER_MEL).to(wav.device)(wav.float())
    return torch.log(torch.clamp(mel, min=1e-5))
