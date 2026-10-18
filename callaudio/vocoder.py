"""Перевод лог-мел-спектрограммы обратно в звук: Griffin-Lim (без обучения) и HiFi-GAN (нейросетевой)."""
from __future__ import annotations

import torch

from .audio import VOCODER_MEL

_hifigan = None


def griffin_lim(log_mel: torch.Tensor, n_iter: int = 64) -> torch.Tensor:
    """(80, T) или (B, 80, T) лог-мел -> waveform. Фаза подбирается итеративно, слышны «металлические» артефакты."""
    import torchaudio.transforms as T

    squeeze = log_mel.dim() == 2
    mel = torch.exp(log_mel if not squeeze else log_mel[None]).float().cpu()
    n_stft = VOCODER_MEL["n_fft"] // 2 + 1
    inv = T.InverseMelScale(n_stft=n_stft, n_mels=VOCODER_MEL["n_mels"], sample_rate=VOCODER_MEL["sample_rate"],
                            f_min=VOCODER_MEL["f_min"], f_max=VOCODER_MEL["f_max"],
                            norm=VOCODER_MEL["norm"], mel_scale=VOCODER_MEL["mel_scale"])
    gl = T.GriffinLim(n_fft=VOCODER_MEL["n_fft"], win_length=VOCODER_MEL["win_length"],
                      hop_length=VOCODER_MEL["hop_length"], power=1.0, n_iter=n_iter)
    wav = gl(inv(mel).clamp(min=0))
    return wav[0] if squeeze else wav


def hifigan(log_mel: torch.Tensor, savedir: str = "pretrained/hifigan-16k") -> torch.Tensor:
    """Предобученный HiFi-GAN из SpeechBrain (Apache-2.0). Ожидает ровно такие же параметры мел, как VOCODER_MEL."""
    global _hifigan
    if _hifigan is None:
        from speechbrain.inference.vocoders import HIFIGAN

        _hifigan = HIFIGAN.from_hparams(source="speechbrain/tts-hifigan-libritts-16kHz", savedir=savedir)
    squeeze = log_mel.dim() == 2
    mel = (log_mel[None] if squeeze else log_mel).float().cpu()
    wav = _hifigan.decode_batch(mel).squeeze(1)
    return wav[0] if squeeze else wav


def vocode(log_mel: torch.Tensor, method: str = "hifigan") -> torch.Tensor:
    if method == "hifigan":
        try:
            return hifigan(log_mel)
        except Exception as e:
            print(f"[vocoder] HiFi-GAN недоступен ({e}), использую Griffin-Lim")
    return griffin_lim(log_mel)
