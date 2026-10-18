"""Метрики качества: разделение, классификация, распознавание речи."""
from __future__ import annotations

import re

import numpy as np
import torch


# ---------- классификация ----------

def classification_metrics(targets, preds, num_classes: int) -> dict:
    """UA (balanced accuracy) - средняя полнота по классам, WA - обычная точность,
    macro F1 - среднее F1 по классам. При дисбалансе главная метрика - macro F1."""
    targets, preds = np.asarray(targets), np.asarray(preds)
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(targets, preds):
        cm[t, p] += 1
    tp = np.diag(cm).astype(float)
    support = cm.sum(1)
    predicted = cm.sum(0)
    recall = np.divide(tp, support, out=np.zeros_like(tp), where=support > 0)
    precision = np.divide(tp, predicted, out=np.zeros_like(tp), where=predicted > 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros_like(tp), where=(precision + recall) > 0)
    present = support > 0
    return {
        "ua": float(recall[present].mean()) if present.any() else 0.0,
        "wa": float(tp.sum() / max(cm.sum(), 1)),
        "macro_f1": float(f1[present].mean()) if present.any() else 0.0,
        "per_class_f1": f1.tolist(),
        "confusion": cm.tolist(),
    }


# ---------- разделение ----------

def separation_metrics(est: torch.Tensor, clean: torch.Tensor, mix: torch.Tensor, sr: int = 16000,
                       with_perceptual: bool = True) -> dict:
    """SI-SDR, SI-SDRi и (если установлены пакеты pesq / pystoi) PESQ и STOI для одного примера (T,)."""
    from .losses import si_sdr

    est, clean, mix = est.float().cpu()[None], clean.float().cpu()[None], mix.float().cpu()[None]
    out = {"si_sdr": si_sdr(est, clean).item(), "si_sdr_mix": si_sdr(mix, clean).item()}
    out["si_sdri"] = out["si_sdr"] - out["si_sdr_mix"]
    if with_perceptual:
        try:
            from torchmetrics.functional.audio import (
                perceptual_evaluation_speech_quality as pesq,
                short_time_objective_intelligibility as stoi,
            )

            out["pesq"] = float(pesq(est[0], clean[0], sr, "wb"))
            out["pesq_mix"] = float(pesq(mix[0], clean[0], sr, "wb"))
            out["stoi"] = float(stoi(est[0], clean[0], sr))
            out["stoi_mix"] = float(stoi(mix[0], clean[0], sr))
        except Exception as e:  # pesq может упасть на тишине - пропускаем пример
            out["perceptual_error"] = str(e)[:80]
    return out


# ---------- распознавание речи ----------

def normalize_text(text: str) -> str:
    """Нижний регистр, ё → е, без пунктуации - иначе «Карту!» и «карту» засчитаются как ошибка."""
    text = text.lower().replace("ё", "е")
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def wer(references: list[str], hypotheses: list[str]) -> float:
    import jiwer

    refs = [normalize_text(r) for r in references]
    hyps = [normalize_text(h) for h in hypotheses]
    pairs = [(r, h) for r, h in zip(refs, hyps) if r]
    if not pairs:
        return float("nan")
    return float(jiwer.wer([p[0] for p in pairs], [p[1] for p in pairs]))


def silhouette(embeddings: np.ndarray, labels: np.ndarray) -> float:
    """Насколько хорошо разделены классы в исходном (не 2D) пространстве: от −1 до 1."""
    from sklearn.metrics import silhouette_score

    if len(np.unique(labels)) < 2:
        return float("nan")
    return float(silhouette_score(embeddings, labels, metric="cosine"))
