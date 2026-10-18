"""Веб-демо системы анализа звонков (Gradio).

Локально или в Colab:    python app/app.py --share
Пути к моделям задаются переменными окружения или аргументами:
    ENHANCER_CKPT   чекпойнт модуля 1 (очистка)
    EMOTION_CKPT    чекпойнт модуля 2 (эмоции); если не задан — открытая HuBERT, дообученная на Dusha
    VAE_CKPT        чекпойнт модуля 3 (VAE)
    UMAP_PKL        viz/umap_final.pkl прогона эмоций — для карты эмоций
"""
from __future__ import annotations

import argparse
import csv
import os
import pickle
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import gradio as gr  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

from callaudio.audio import SAMPLE_RATE, load_audio  # noqa: E402

EMOTION_RU = {"neutral": "нейтрально", "angry": "злость", "positive": "позитив", "sad": "грусть", "other": "другое"}
EMOTION_COLORS = {"neutral": "#9aa0a6", "angry": "#d93025", "positive": "#1e8e3e", "sad": "#1a73e8", "other": "#f9ab00"}

ARGS = None
_analyzer = None
_vae = None


def analyzer():
    global _analyzer
    if _analyzer is None:
        from callaudio.pipeline import CallAnalyzer

        _analyzer = CallAnalyzer(enhancer_ckpt=ARGS.enhancer or None, emotion_ckpt=ARGS.emotion or None,
                                 whisper_size=ARGS.whisper)
    return _analyzer


# ---------- вкладка 1: анализ звонка ----------

def spectrogram_figure(wav: np.ndarray, clean: np.ndarray):
    fig, axes = plt.subplots(2, 1, figsize=(9, 4.2), sharex=True)
    for ax, x, title in ((axes[0], wav, "До очистки"), (axes[1], clean, "После очистки")):
        ax.specgram(x + 1e-9, NFFT=512, Fs=SAMPLE_RATE, noverlap=384, cmap="magma", vmin=-120)
        ax.set_ylabel("Гц")
        ax.set_title(title, fontsize=10)
    axes[1].set_xlabel("секунды")
    fig.tight_layout()
    return fig


def timeline_figure(segments: list[dict], duration: float):
    fig, ax = plt.subplots(figsize=(9, 1.4))
    for s in segments:
        ax.barh(0, s["end"] - s["start"], left=s["start"], color=EMOTION_COLORS.get(s["emotion"], "#ccc"), height=0.6)
    ax.set_xlim(0, max(duration, 1))
    ax.set_yticks([])
    ax.set_xlabel("секунды")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in EMOTION_COLORS.values()]
    ax.legend(handles, [EMOTION_RU[k] for k in EMOTION_COLORS], ncol=5, fontsize=8, loc="upper center",
              bbox_to_anchor=(0.5, 1.6), frameon=False)
    fig.tight_layout()
    return fig


def run_analysis(path, enhance, transcribe):
    if not path:
        raise gr.Error("Загрузите или запишите аудио")
    res = analyzer().analyze(path, enhance=enhance, transcribe=transcribe, max_sec=ARGS.max_sec)
    wav, clean = res["wav"], res["clean"]
    rows = [[f"{s['start']:.1f}–{s['end']:.1f}", s["text"], EMOTION_RU.get(s["emotion"], s["emotion"]),
             f"{s['confidence']:.2f}"] for s in res["segments"]]
    csv_path = Path(tempfile.mkdtemp()) / "report.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["start", "end", "text", "emotion", "confidence"])
        for s in res["segments"]:
            w.writerow([s["start"], s["end"], s["text"], s["emotion"], s["confidence"]])
    sm, tm = res["summary"], res["timings"]
    summary = (f"**Длительность:** {sm['duration_sec']} с · **реплик:** {sm['segments']} · "
               f"**доля негатива (злость + грусть):** {sm['negative_share']:.0%}\n\n"
               "**Время шагов, с:** " + ", ".join(f"{k} {v}" for k, v in tm.items()))
    removed = wav - clean if enhance else np.zeros_like(wav)
    return ((SAMPLE_RATE, wav), (SAMPLE_RATE, clean), (SAMPLE_RATE, removed),
            spectrogram_figure(wav, clean), timeline_figure(res["segments"], sm["duration_sec"]),
            rows, summary, str(csv_path))


# ---------- вкладка 2: латентное пространство ----------

def vae():
    global _vae
    if _vae is None:
        from callaudio.tasks import load_trained

        if not ARGS.vae:
            raise gr.Error("Не задан VAE_CKPT")
        model, task, _ = load_trained(ARGS.vae)
        if model.num_classes > 0:
            raise gr.Error("Для интерполяции между записями нужен безусловный VAE (vae_mse)")
        _vae = (model, task)
    return _vae


def word_mel(path):
    from callaudio.audio import fix_length, vocoder_mel
    from callaudio.data.datasets import CommandsMelDataset

    wav = load_audio(path)
    energy = np.convolve(wav ** 2, np.ones(1600) / 1600, mode="same")  # окно 1 с вокруг самого громкого места
    center = int(np.argmax(energy))
    start = max(0, min(center - SAMPLE_RATE // 2, len(wav) - SAMPLE_RATE))
    wav = fix_length(wav[start:start + SAMPLE_RATE], SAMPLE_RATE)
    mel = vocoder_mel(torch.from_numpy(wav)[None])[0]
    return torch.nn.functional.pad(mel, (0, 64 - mel.shape[-1]), value=CommandsMelDataset.PAD_VALUE)[:, :64]


def run_interpolation(path_a, path_b, steps, vocoder_name):
    from callaudio.vocoder import vocode

    if not path_a or not path_b:
        raise gr.Error("Нужны две записи коротких слов")
    model, task = vae()
    xa, xb = task.norm(word_mel(path_a)), task.norm(word_mel(path_b))
    mels = task.denorm(model.interpolate(xa, xb, steps=int(steps)))
    fig, axes = plt.subplots(1, len(mels), figsize=(1.5 * len(mels), 2.3), sharey=True)
    audio = []
    for k, (ax, m) in enumerate(zip(axes, mels)):
        ax.imshow(m.numpy(), origin="lower", aspect="auto", cmap="magma")
        ax.set_title(f"{k / (len(mels) - 1):.2f}", fontsize=8)
        ax.set_xticks([])
        audio += [vocode(m, vocoder_name).numpy(), np.zeros(SAMPLE_RATE // 4, dtype=np.float32)]
    fig.tight_layout()
    return fig, (SAMPLE_RATE, np.concatenate(audio))


# ---------- вкладка 3: карта эмоций ----------

def run_map(path):
    import plotly.graph_objects as go

    if not ARGS.umap or not Path(ARGS.umap).exists():
        raise gr.Error("Не задан UMAP_PKL (viz/umap_final.pkl прогона эмоций)")
    with open(ARGS.umap, "rb") as f:
        m = pickle.load(f)
    fig = go.Figure()
    for k, name in enumerate(m["names"]):
        sel = m["labels"] == k
        fig.add_trace(go.Scattergl(x=m["coords"][sel, 0], y=m["coords"][sel, 1], mode="markers", name=EMOTION_RU.get(name, name),
                                   marker=dict(size=4, opacity=0.5, color=EMOTION_COLORS.get(name))))
    if path:
        a = analyzer()
        if a.emotion_model is None:
            raise gr.Error("Для проекции своих реплик нужен свой чекпойнт эмоций (EMOTION_CKPT)")
        res = a.analyze(path, transcribe=False, max_sec=ARGS.max_sec)
        n = int(a.emotion_sec * SAMPLE_RATE)
        embs = []
        for s in res["segments"]:
            seg = res["clean"][int(s["start"] * SAMPLE_RATE): int(s["end"] * SAMPLE_RATE)][:n]
            x = torch.from_numpy(np.pad(seg, (0, max(0, n - len(seg)))))[None].to(a.device)
            with torch.no_grad():
                embs.append(a.emotion_model.embed(x).float().cpu().numpy()[0])
        if embs:
            xy = m["reducer"].transform((np.stack(embs) - m["mean"]) / m["std"])
            fig.add_trace(go.Scatter(x=xy[:, 0], y=xy[:, 1], mode="markers+text", name="ваши реплики",
                                     text=[f"{s['start']:.0f}с" for s in res["segments"]], textposition="top center",
                                     marker=dict(size=12, symbol="star", color="black")))
    fig.update_layout(height=520, xaxis_visible=False, yaxis_visible=False, legend_title_text="эмоция")
    return fig


def build_ui():
    with gr.Blocks(title="Анализ звонков") as demo:
        gr.Markdown("# Анализ звонков\nОчистка речи от фона, расшифровка, эмоции по репликам и латентное пространство речи. "
                    "Используйте только свои записи или открытые данные.")
        with gr.Tab("Анализ звонка"):
            with gr.Row():
                inp = gr.Audio(sources=["upload", "microphone"], type="filepath", label="Запись (wav, mp3, ogg, m4a)")
                with gr.Column():
                    enh = gr.Checkbox(value=True, label="Очищать от фона")
                    asr = gr.Checkbox(value=True, label="Расшифровывать (Whisper)")
                    btn = gr.Button("Анализировать", variant="primary")
            summary = gr.Markdown()
            with gr.Row():
                a_orig = gr.Audio(label="Исходная")
                a_clean = gr.Audio(label="Очищенная")
                a_removed = gr.Audio(label="Что удалено")
            spec = gr.Plot(label="Спектрограммы")
            timeline = gr.Plot(label="Эмоции во времени")
            table = gr.Dataframe(headers=["время, с", "текст", "эмоция", "уверенность"], wrap=True)
            report = gr.File(label="Отчёт CSV")
            btn.click(run_analysis, [inp, enh, asr], [a_orig, a_clean, a_removed, spec, timeline, table, summary, report])
        with gr.Tab("Латентное пространство"):
            gr.Markdown("Запишите два коротких слова (например, «yes» и «no»): модель плавно перейдёт от одного к другому.")
            with gr.Row():
                wa = gr.Audio(sources=["upload", "microphone"], type="filepath", label="Слово A")
                wb = gr.Audio(sources=["upload", "microphone"], type="filepath", label="Слово B")
            with gr.Row():
                steps = gr.Slider(3, 12, value=8, step=1, label="Шагов")
                voc = gr.Radio(["hifigan", "griffin_lim"], value="hifigan", label="Вокодер")
                ibtn = gr.Button("Интерполировать", variant="primary")
            igrid = gr.Plot()
            iaudio = gr.Audio(label="Все шаги подряд")
            ibtn.click(run_interpolation, [wa, wb, steps, voc], [igrid, iaudio])
        with gr.Tab("Карта эмоций"):
            gr.Markdown("Пространство эмбеддингов классификатора эмоций (UMAP). Загрузите запись — её реплики появятся на карте.")
            mfile = gr.Audio(sources=["upload"], type="filepath", label="Запись (необязательно)")
            mbtn = gr.Button("Показать карту", variant="primary")
            mplot = gr.Plot()
            mbtn.click(run_map, [mfile], [mplot])
    return demo


def main():
    global ARGS
    p = argparse.ArgumentParser()
    p.add_argument("--enhancer", default=os.getenv("ENHANCER_CKPT"))
    p.add_argument("--emotion", default=os.getenv("EMOTION_CKPT"))
    p.add_argument("--vae", default=os.getenv("VAE_CKPT"))
    p.add_argument("--umap", default=os.getenv("UMAP_PKL"))
    p.add_argument("--whisper", default=os.getenv("WHISPER_SIZE", "small"))
    p.add_argument("--max-sec", type=float, default=float(os.getenv("MAX_SEC", 180)))
    p.add_argument("--share", action="store_true", help="публичная ссылка (удобно в Colab)")
    ARGS = p.parse_args()
    build_ui().launch(share=ARGS.share)


if __name__ == "__main__":
    main()
