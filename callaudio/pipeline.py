"""Конвейер анализа звонка: очистка -> нарезка на реплики (VAD) -> расшифровка (Whisper) -> эмоции.

    from callaudio.pipeline import CallAnalyzer
    analyzer = CallAnalyzer(enhancer_ckpt="runs/sep_unet_sisdr/best.pt", emotion_ckpt="runs/emo_hubert_ft_ce/best.pt")
    result = analyzer.analyze("call.ogg")
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field

import numpy as np
import torch

from .audio import SAMPLE_RATE, load_audio

NEGATIVE = {"angry", "sad"}
HF_EMOTION_MODEL = "xbgoose/hubert-large-speech-emotion-recognition-russian-dusha-finetuned"
HF_LABELS = ["neutral", "angry", "positive", "sad", "other"]


@dataclass
class Segment:
    start: float
    end: float
    text: str = ""
    emotion: str = ""
    confidence: float = 0.0
    probs: dict = field(default_factory=dict)


class CallAnalyzer:
    def __init__(self, enhancer_ckpt: str | None = None, emotion_ckpt: str | None = None,
                 use_hf_emotion_fallback: bool = True, whisper_size: str = "small", device: str = "auto"):
        from .tasks import load_trained
        from .utils import get_device

        self.device = get_device(device)
        self.enhancer = load_trained(enhancer_ckpt, self.device)[0] if enhancer_ckpt else None
        self.emotion_model, self.labels, self.hf_emotion = None, [], None
        self.emotion_sec = 4.0
        if emotion_ckpt:
            model, task, ck = load_trained(emotion_ckpt, self.device)
            self.emotion_model, self.labels = model, task.labels
            self.emotion_sec = ck["cfg"]["data"].get("max_sec", 4.0)
        elif use_hf_emotion_fallback:
            self._load_hf_emotion()
        self.whisper_size = whisper_size
        self._whisper = None
        self._vad = None

    # ---------- загрузка моделей по требованию ----------

    def _load_hf_emotion(self):
        """Запасной вариант: открытая HuBERT, дообученная на Dusha (Apache-2.0)."""
        from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

        self.hf_extractor = AutoFeatureExtractor.from_pretrained(HF_EMOTION_MODEL)
        self.hf_emotion = AutoModelForAudioClassification.from_pretrained(HF_EMOTION_MODEL).to(self.device).eval()
        id2label = getattr(self.hf_emotion.config, "id2label", None) or {}
        self.labels = [id2label.get(i, HF_LABELS[i]) if not str(id2label.get(i, "")).startswith("LABEL_") else HF_LABELS[i]
                       for i in range(self.hf_emotion.config.num_labels)]

    @property
    def whisper(self):
        if self._whisper is None:
            from faster_whisper import WhisperModel

            cuda = self.device.type == "cuda"
            self._whisper = WhisperModel(self.whisper_size, device="cuda" if cuda else "cpu",
                                         compute_type="float16" if cuda else "int8")
        return self._whisper

    @property
    def vad(self):
        if self._vad is None:
            from silero_vad import get_speech_timestamps, load_silero_vad

            self._vad = (load_silero_vad(), get_speech_timestamps)
        return self._vad

    # ---------- шаги ----------

    @torch.no_grad()
    def enhance(self, wav: np.ndarray) -> np.ndarray:
        from .evaluate import enhance_long

        if self.enhancer is None:
            return wav
        return enhance_long(self.enhancer, torch.from_numpy(wav), self.device).numpy()

    def segment(self, wav: np.ndarray, max_seg_sec: float = 15.0, min_gap_sec: float = 0.4) -> list[tuple[float, float]]:
        model, get_ts = self.vad
        ts = get_ts(torch.from_numpy(wav), model, sampling_rate=SAMPLE_RATE, return_seconds=True)
        merged: list[list[float]] = []
        for t in ts:  # склеиваем куски с короткими паузами между ними
            if merged and t["start"] - merged[-1][1] < min_gap_sec and t["end"] - merged[-1][0] <= max_seg_sec:
                merged[-1][1] = t["end"]
            else:
                merged.append([t["start"], t["end"]])
        out = []
        for s, e in merged:  # длинные реплики режем на части не длиннее max_seg_sec
            while e - s > max_seg_sec:
                out.append((s, s + max_seg_sec))
                s += max_seg_sec
            out.append((s, e))
        return out

    def transcribe(self, seg: np.ndarray) -> str:
        parts, _ = self.whisper.transcribe(seg.astype(np.float32), language="ru", beam_size=1, vad_filter=False)
        return " ".join(p.text.strip() for p in parts).strip()

    @torch.no_grad()
    def emotion(self, seg: np.ndarray) -> dict[str, float]:
        """Длинные реплики режутся на окна длиной как при обучении, вероятности усредняются."""
        if self.hf_emotion is not None:
            inputs = self.hf_extractor(seg, sampling_rate=SAMPLE_RATE, return_tensors="pt").to(self.device)
            probs = torch.softmax(self.hf_emotion(**inputs).logits, -1)[0].cpu().numpy()
            return dict(zip(self.labels, probs.tolist()))
        if self.emotion_model is None:
            return {}
        n = int(self.emotion_sec * SAMPLE_RATE)
        windows = [seg[i:i + n] for i in range(0, max(1, len(seg) - n // 2), n)] or [seg]
        batch = torch.stack([torch.from_numpy(np.pad(w, (0, max(0, n - len(w))))[:n]) for w in windows]).to(self.device)
        probs = torch.softmax(self.emotion_model(batch).float(), -1).mean(0).cpu().numpy()
        return dict(zip(self.labels, probs.tolist()))

    # ---------- всё вместе ----------

    def analyze(self, path: str, enhance: bool = True, transcribe: bool = True, max_sec: float = 180.0) -> dict:
        timings = {}
        t = time.time()
        wav = load_audio(path)[: int(max_sec * SAMPLE_RATE)]
        timings["загрузка"] = time.time() - t

        t = time.time()
        clean = self.enhance(wav) if enhance else wav
        timings["очистка"] = time.time() - t

        t = time.time()
        spans = self.segment(clean)
        timings["нарезка"] = time.time() - t

        segments, t_asr, t_emo = [], 0.0, 0.0
        for s, e in spans:
            seg = clean[int(s * SAMPLE_RATE): int(e * SAMPLE_RATE)]
            item = Segment(start=round(s, 2), end=round(e, 2))
            if transcribe:
                t0 = time.time()
                item.text = self.transcribe(seg)
                t_asr += time.time() - t0
            t0 = time.time()
            probs = self.emotion(seg)
            t_emo += time.time() - t0
            if probs:
                item.emotion = max(probs, key=probs.get)
                item.confidence = round(probs[item.emotion], 3)
                item.probs = {k: round(v, 3) for k, v in probs.items()}
            segments.append(item)
        timings["расшифровка"], timings["эмоции"] = t_asr, t_emo

        total = sum(sg.end - sg.start for sg in segments) or 1.0
        negative = sum(sg.end - sg.start for sg in segments if sg.emotion in NEGATIVE) / total
        return {
            "segments": [asdict(sg) for sg in segments],
            "summary": {"duration_sec": round(len(wav) / SAMPLE_RATE, 1), "segments": len(segments),
                        "negative_share": round(negative, 3)},
            "timings": {k: round(v, 2) for k, v in timings.items()},
            "wav": wav, "clean": clean,
        }
