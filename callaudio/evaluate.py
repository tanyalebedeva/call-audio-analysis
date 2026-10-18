"""Оценка обученных моделей на тестовой выборке. Результаты кладутся рядом с чекпойнтом.

    python -m callaudio.evaluate separation --ckpt runs/sep_unet_sisdr/best.pt --wer
    python -m callaudio.evaluate emotion    --ckpt runs/emo_mel_ce/best.pt
    python -m callaudio.evaluate emotion    --ckpt runs/emo_mel_ce/best.pt --noise-snr 5 --enhancer runs/sep_unet_sisdr/best.pt
    python -m callaudio.evaluate vae        --ckpt runs/vae_mse/best.pt --pair yes,no
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import DataLoader

from .audio import SAMPLE_RATE
from .data import CommandsMelDataset, EmotionDataset, SeparationDataset
from .metrics import classification_metrics, separation_metrics, silhouette, wer
from .tasks import load_trained
from .utils import get_device


def _save_json(obj, path: Path):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[evaluate] результаты: {path}")


def _mean_table(rows: list[dict], key: str) -> dict:
    groups = defaultdict(list)
    for r in rows:
        groups[r[key]].append(r)
    table = {}
    for g, items in sorted(groups.items()):
        metrics = {}
        for m in items[0]:
            vals = [it[m] for it in items if isinstance(it.get(m), (int, float)) and m not in (key, "snr")]
            if vals:
                metrics[m] = float(np.mean(vals))
        metrics["n"] = len(items)
        table[str(g)] = metrics
    return table


# ===================== модуль 1 =====================

@torch.no_grad()
def enhance_long(model, wav: torch.Tensor, device, chunk_sec: float = 10.0, overlap_sec: float = 1.0) -> torch.Tensor:
    """Обработка длинной записи кусками с перекрытием и плавной склейкой - экономит память."""
    wav = wav.float()
    n, chunk, hop = wav.shape[-1], int(chunk_sec * SAMPLE_RATE), int((chunk_sec - overlap_sec) * SAMPLE_RATE)
    if n <= chunk:
        return model(wav[None].to(device))[0].float().cpu()
    out = torch.zeros(n)
    weight = torch.zeros(n)
    fade = torch.hann_window(chunk, periodic=False).clamp(min=1e-3)
    for start in range(0, n, hop):
        seg = wav[start:start + chunk]
        est = model(seg[None].to(device))[0].float().cpu()
        w = fade[: len(seg)]
        out[start:start + len(seg)] += est * w
        weight[start:start + len(seg)] += w
        if start + chunk >= n:
            break
    return out / weight.clamp(min=1e-3)


def evaluate_separation(a):
    device = get_device()
    model, task, ck = load_trained(a.ckpt, device)
    d = ck["cfg"]["data"]
    ds = SeparationDataset(a.speech_manifest or d.get("speech_test", d.get("speech_val")),
                           a.noise_manifest or d.get("noise_test", d.get("noise_val")),
                           segment_sec=None, deterministic=True, seed=2, snr_grid=(-5, 0, 5, 10, 15),
                           epoch_size=a.limit, synthetic=d.get("synthetic", False))
    if a.limit is None:
        ds.epoch_size = len(ds.speech) if ds.speech else 50
    out_dir = Path(a.ckpt).parent
    examples = out_dir / "examples"
    asr = None
    if a.wer:
        from faster_whisper import WhisperModel

        asr = WhisperModel(a.whisper, device="cuda" if device.type == "cuda" else "cpu",
                           compute_type="float16" if device.type == "cuda" else "int8")

    def transcribe(x: torch.Tensor) -> str:
        segments, _ = asr.transcribe(x.numpy().astype(np.float32), language="ru", beam_size=1)
        return " ".join(s.text for s in segments)

    rows, refs, hyp = [], [], defaultdict(list)
    for i in range(len(ds)):
        item = ds[i]
        est = enhance_long(model, item["mix"], device)
        r = separation_metrics(est, item["clean"], item["mix"], with_perceptual=not a.no_perceptual)
        r.update(snr=item["snr"], noise_kind=item["noise_kind"])
        if asr is not None and item["text"]:
            refs.append(item["text"])
            for cond, x in (("clean", item["clean"]), ("noisy", item["mix"]), ("enhanced", est)):
                h = transcribe(x)
                hyp[cond].append(h)
                r[f"wer_{cond}"] = wer([item["text"]], [h])
        rows.append(r)
        if i < a.save_examples:
            examples.mkdir(exist_ok=True)
            for name, x in (("mix", item["mix"]), ("enhanced", est), ("clean", item["clean"])):
                sf.write(examples / f"{i:02d}_snr{int(item['snr'])}_{name}.wav", x.numpy(), SAMPLE_RATE)
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(ds)}")
    result = {
        "checkpoint": str(a.ckpt), "n": len(rows),
        "overall": _mean_table([{**r, "all": "all"} for r in rows], "all")["all"],
        "by_snr": _mean_table(rows, "snr"),
        "by_noise_kind": _mean_table(rows, "noise_kind"),
    }
    if refs:
        result["wer"] = {cond: wer(refs, hs) for cond, hs in hyp.items()}
    _save_json(result, out_dir / "results_separation.json")
    return result


# ===================== модуль 2 =====================

def evaluate_emotion(a):
    device = get_device()
    model, task, ck = load_trained(a.ckpt, device)
    d = ck["cfg"]["data"]
    ds = EmotionDataset(a.manifest or d.get("test", d.get("val")), labels=task.labels, max_sec=d.get("max_sec", 4.0),
                        train=False, seed=3, noise_manifest=a.noise_manifest or (d.get("noise_test") if a.noise_snr is not None else None),
                        fixed_snr=a.noise_snr, synthetic=d.get("synthetic", False),
                        synthetic_size=d.get("synthetic_size", 512) // 4)
    enhancer = None
    if a.enhancer:
        enhancer, _, _ = load_trained(a.enhancer, device)

    class _Wrapped(torch.nn.Module):  # очистка -> классификация
        def __init__(self, enh, clf):
            super().__init__()
            self.enh, self.clf = enh, clf

        def forward(self, wav):
            return self.clf(self.enh(wav))

    clf = _Wrapped(enhancer, model) if enhancer is not None else model
    loader = DataLoader(ds, batch_size=a.batch_size, num_workers=2)
    preds, targets, probs = task.predict(clf, loader, device)
    m = classification_metrics(targets, preds, num_classes=len(task.labels))
    m["labels"] = task.labels
    m["condition"] = {"noise_snr": a.noise_snr, "enhancer": a.enhancer}
    if hasattr(model, "layer_importance"):
        m["layer_importance"] = model.layer_importance().cpu().tolist()
    suffix = ""
    if a.noise_snr is not None:
        suffix += f"_snr{int(a.noise_snr)}"
    if a.enhancer:
        suffix += "_enhanced"
    out_dir = Path(a.ckpt).parent
    _save_json(m, out_dir / f"results_emotion{suffix}.json")
    _plot_confusion(np.array(m["confusion"]), task.labels, out_dir / f"confusion{suffix}.png",
                    title=f"macro F1 = {m['macro_f1']:.3f}, UA = {m['ua']:.3f}")
    return m


def _plot_confusion(cm: np.ndarray, labels, path: Path, title: str = ""):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    norm = cm / cm.sum(1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(labels)), labels, rotation=30)
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("предсказано")
    ax.set_ylabel("на самом деле")
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f"{norm[i, j]:.2f}", ha="center", va="center", color="white" if norm[i, j] > 0.5 else "black")
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ===================== модуль 3 =====================

def _features(mels: torch.Tensor) -> np.ndarray:
    """Грубые признаки для «судьи»: спектрограмма, уменьшенная в 4 раза по обеим осям."""
    return torch.nn.functional.avg_pool2d(mels[:, None], 4).flatten(1).numpy()


@torch.no_grad()
def evaluate_vae(a):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    from .vocoder import vocode

    device = get_device()
    model, task, ck = load_trained(a.ckpt, device)
    d = ck["cfg"]["data"]
    synthetic = d.get("synthetic", False)
    train = CommandsMelDataset(d.get("train"), words=task.words, synthetic=synthetic,
                               synthetic_size=d.get("synthetic_size", 512))
    test = CommandsMelDataset(a.manifest or d.get("test", d.get("val")), words=task.words, synthetic=synthetic,
                              synthetic_size=d.get("synthetic_size", 512) // 4, seed=2)
    cond = model.num_classes > 0

    def encode_all(ds, limit):
        idx = np.random.default_rng(0).choice(len(ds), size=min(limit, len(ds)), replace=False)
        loader = DataLoader(torch.utils.data.Subset(ds, idx.tolist()), batch_size=128)
        mels, mus, recons, labels = [], [], [], []
        for b in loader:
            x = task.norm(b["mel"].to(device))
            lab = b["label"].to(device) if cond else None
            recon, mu, _ = model(x, lab)
            mels.append(b["mel"])
            mus.append(mu.cpu())
            recons.append(task.denorm(recon).cpu())
            labels.append(b["label"])
        return torch.cat(mels), torch.cat(mus).numpy(), torch.cat(recons), torch.cat(labels).numpy()

    tr_mel, tr_mu, _, tr_y = encode_all(train, a.limit)
    te_mel, te_mu, te_rec, te_y = encode_all(test, a.limit // 4)
    result = {"checkpoint": str(a.ckpt), "words": task.words, "conditional": cond}
    result["recon_mse_logmel"] = float(((te_rec - te_mel) ** 2).mean())
    # насколько латент «знает» слово: линейный классификатор и silhouette
    probe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(tr_mu, tr_y)
    result["latent_linear_probe_acc"] = float(probe.score(te_mu, te_y))
    result["latent_silhouette"] = silhouette(te_mu, te_y)
    # «судья»: классификатор слов по настоящим спектрограммам проверяет реконструкции и генерации
    judge = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000)).fit(_features(tr_mel), tr_y)
    result["judge_acc_real"] = float(judge.score(_features(te_mel), te_y))
    result["judge_acc_recon"] = float(judge.score(_features(te_rec), te_y))
    if cond:
        n = 20
        labels = torch.arange(len(task.words), device=device).repeat_interleave(n)
        gen = task.denorm(model.sample(len(labels), labels, device=device)).cpu()
        result["judge_acc_generated"] = float(judge.score(_features(gen), labels.cpu().numpy()))

    out_dir = Path(a.ckpt).parent / "vae_examples"
    out_dir.mkdir(exist_ok=True)
    # интерполяция между двумя словами
    w_a, w_b = (a.pair.split(",") if a.pair else task.words[:2])
    ia = next(i for i, r in enumerate(test.rows) if r["label"] == w_a)
    ib = next(i for i, r in enumerate(test.rows) if r["label"] == w_b)
    xa, xb = task.norm(test[ia]["mel"].to(device)), task.norm(test[ib]["mel"].to(device))
    la = torch.tensor([test.word2id[w_a]], device=device) if cond else None
    lb = torch.tensor([test.word2id[w_b]], device=device) if cond else None
    path = task.denorm(model.interpolate(xa, xb, steps=a.steps, label_a=la, label_b=lb)).cpu()
    proba = judge.predict_proba(_features(path))
    classes = list(judge.classes_)
    result["interpolation"] = {
        "from": w_a, "to": w_b,
        "p_from": proba[:, classes.index(test.word2id[w_a])].round(3).tolist(),
        "p_to": proba[:, classes.index(test.word2id[w_b])].round(3).tolist(),
    }
    _plot_interpolation(path, w_a, w_b, out_dir / f"interpolation_{w_a}_{w_b}.png")
    audio = []
    for step in path:
        audio += [vocode(step, a.vocoder).numpy(), np.zeros(SAMPLE_RATE // 4, dtype=np.float32)]
    sf.write(out_dir / f"interpolation_{w_a}_{w_b}_{a.vocoder}.wav", np.concatenate(audio), SAMPLE_RATE)
    for k in range(min(3, len(te_mel))):
        sf.write(out_dir / f"recon_{k}_original.wav", vocode(te_mel[k], a.vocoder).numpy(), SAMPLE_RATE)
        sf.write(out_dir / f"recon_{k}_vae.wav", vocode(te_rec[k], a.vocoder).numpy(), SAMPLE_RATE)
    _save_json(result, Path(a.ckpt).parent / "results_vae.json")
    return result


def _plot_interpolation(mels: torch.Tensor, w_a: str, w_b: str, path: Path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(mels), figsize=(1.6 * len(mels), 2.4), sharey=True)
    for k, (ax, m) in enumerate(zip(axes, mels)):
        ax.imshow(m.numpy(), origin="lower", aspect="auto", cmap="magma")
        ax.set_title(f"{k / (len(mels) - 1):.2f}", fontsize=8)
        ax.set_xticks([])
    axes[0].set_ylabel("мел-полоса")
    fig.suptitle(f"Интерполяция в латентном пространстве: «{w_a}» → «{w_b}»", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="task", required=True)
    s = sub.add_parser("separation")
    s.add_argument("--ckpt", required=True)
    s.add_argument("--speech-manifest")
    s.add_argument("--noise-manifest")
    s.add_argument("--limit", type=int, default=None, help="сколько высказываний оценить (по умолчанию все)")
    s.add_argument("--wer", action="store_true", help="считать WER Whisper на чистой, зашумлённой и очищенной речи")
    s.add_argument("--whisper", default="small")
    s.add_argument("--no-perceptual", action="store_true", help="не считать PESQ/STOI (быстрее)")
    s.add_argument("--save-examples", type=int, default=5)
    e = sub.add_parser("emotion")
    e.add_argument("--ckpt", required=True)
    e.add_argument("--manifest")
    e.add_argument("--noise-manifest")
    e.add_argument("--noise-snr", type=float, default=None)
    e.add_argument("--enhancer", default=None, help="чекпойнт модели очистки из модуля 1")
    e.add_argument("--batch-size", type=int, default=32)
    v = sub.add_parser("vae")
    v.add_argument("--ckpt", required=True)
    v.add_argument("--manifest")
    v.add_argument("--limit", type=int, default=4000)
    v.add_argument("--pair", default=None, help="два слова через запятую для интерполяции, например yes,no")
    v.add_argument("--steps", type=int, default=8)
    v.add_argument("--vocoder", choices=["hifigan", "griffin_lim"], default="hifigan")
    a = p.parse_args()
    {"separation": evaluate_separation, "emotion": evaluate_emotion, "vae": evaluate_vae}[a.task](a)


if __name__ == "__main__":
    main()
