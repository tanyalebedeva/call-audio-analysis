"""Визуализация пространства признаков по эпохам обучения.

    python -m callaudio.viz --run runs/emo_mel_ce
    python -m callaudio.viz --run runs/vae_mse --method tsne

Берёт embeddings/epoch_XXX.npz, сохранённые train.py, и строит:
* кадры 2D-проекции для каждой эпохи и анимацию GIF;
* график silhouette score по эпохам (считается в исходном пространстве, а не по 2D-картинке);
* интерактивный HTML последней эпохи (plotly);
* umap_final.pkl — обученный UMAP для карты эмоций в демо.
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np

from .metrics import silhouette


def _standardize(x: np.ndarray) -> np.ndarray:
    return (x - x.mean(0)) / (x.std(0) + 1e-6)


def _align(coords: np.ndarray, reference: np.ndarray | None) -> np.ndarray:
    """Поворот/отражение кадра к предыдущему (Procrustes), чтобы анимация не «прыгала»."""
    coords = coords - coords.mean(0)
    coords = coords / (np.linalg.norm(coords) / np.sqrt(len(coords)) + 1e-9)
    if reference is None:
        return coords
    from scipy.linalg import orthogonal_procrustes

    r, _ = orthogonal_procrustes(coords, reference)
    return coords @ r


def project(emb: np.ndarray, method: str, init: np.ndarray | None = None, seed: int = 0):
    x = _standardize(emb)
    if method == "umap":
        import umap

        reducer = umap.UMAP(n_components=2, n_neighbors=30, min_dist=0.2, metric="cosine", random_state=seed,
                            init=init if init is not None else "spectral")
        return reducer.fit_transform(x), reducer
    if method == "tsne":
        from sklearn.manifold import TSNE

        return TSNE(n_components=2, init=init if init is not None else "pca", perplexity=30, random_state=seed).fit_transform(x), None
    if method == "pca":
        from sklearn.decomposition import PCA

        return PCA(n_components=2, random_state=seed).fit_transform(x), None
    raise ValueError(method)


def visualize_run(run: str | Path, method: str = "umap", every: int = 1, fps: int = 3) -> dict:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    run = Path(run)
    files = sorted((run / "embeddings").glob("epoch_*.npz"))[::every]
    if not files:
        raise FileNotFoundError(f"Нет эмбеддингов в {run / 'embeddings'} — сначала обучите модель")
    out = run / "viz"
    out.mkdir(exist_ok=True)

    frames, sil, prev, reducer, last = [], [], None, None, None
    for f in files:
        data = np.load(f, allow_pickle=True)
        emb, labels = data["emb"], data["labels"]
        names = list(data["names"]) if "names" in data and len(data["names"]) else [str(i) for i in np.unique(labels)]
        coords, reducer = project(emb, method, init=prev if method == "umap" else None)
        coords = _align(coords, prev)
        prev = coords
        epoch = int(f.stem.split("_")[1])
        s = silhouette(emb, labels)
        sil.append((epoch, s))
        frames.append((epoch, coords, labels, s))
        last = (emb, labels, names, coords)
        print(f"  эпоха {epoch}: silhouette = {s:.3f}")

    names = last[2]
    colors = plt.get_cmap("tab10")
    lim = np.abs(np.concatenate([c for _, c, _, _ in frames])).max() * 1.05

    def draw(ax, epoch, coords, labels, s):
        ax.clear()
        for k in np.unique(labels):
            m = labels == k
            ax.scatter(coords[m, 0], coords[m, 1], s=6, alpha=0.6, color=colors(int(k) % 10),
                       label=names[int(k)] if int(k) < len(names) else str(k))
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"{run.name} · эпоха {epoch} · silhouette {s:.3f}", fontsize=10)
        ax.legend(loc="upper right", fontsize=7, markerscale=2, frameon=False)

    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    for epoch, coords, labels, s in frames:
        draw(ax, epoch, coords, labels, s)
        fig.savefig(out / f"{method}_epoch_{epoch:03d}.png", dpi=130)
    anim = FuncAnimation(fig, lambda i: draw(ax, *frames[i]), frames=len(frames))
    anim.save(out / f"{method}_evolution.gif", writer=PillowWriter(fps=fps))
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 3))
    ax.plot([e for e, _ in sil], [s for _, s in sil], marker="o")
    ax.set_xlabel("эпоха")
    ax.set_ylabel("silhouette (косинус)")
    ax.set_title("Разделимость классов в пространстве эмбеддингов", fontsize=10)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "silhouette.png", dpi=150)
    plt.close(fig)

    emb, labels, names, coords = last
    try:
        import plotly.express as px

        fig = px.scatter(x=coords[:, 0], y=coords[:, 1], color=[names[int(k)] for k in labels], opacity=0.7,
                         title=f"{run.name}: пространство признаков, последняя эпоха")
        fig.write_html(out / f"{method}_final.html")
    except ImportError:
        pass
    if reducer is not None:
        with open(out / "umap_final.pkl", "wb") as f:
            pickle.dump({"reducer": reducer, "mean": emb.mean(0), "std": emb.std(0) + 1e-6,
                         "coords": coords, "labels": labels, "names": names}, f)
    np.savetxt(out / "silhouette.csv", np.array(sil), delimiter=",", header="epoch,silhouette", comments="")
    print(f"[viz] готово: {out}")
    return {"silhouette": sil, "out": str(out)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--method", choices=["umap", "tsne", "pca"], default="umap")
    p.add_argument("--every", type=int, default=1, help="брать каждую N-ю эпоху")
    p.add_argument("--fps", type=int, default=3)
    a = p.parse_args()
    visualize_run(a.run, a.method, a.every, a.fps)


if __name__ == "__main__":
    main()
