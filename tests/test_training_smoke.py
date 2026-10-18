"""Короткие прогоны обучения на синтетике: проверяют цикл, чекпойнты, продолжение и эмбеддинги."""
import json

import numpy as np

from callaudio.tasks import load_trained
from callaudio.train import train
from callaudio.utils import load_config


def _cfg(path, tmp_path, name, **extra):
    overrides = ["data.synthetic=true", "num_workers=0", "amp=false", f"out_dir={tmp_path / name}",
                 "log_every=1000", "epochs=2", "probe_size=64"]
    overrides += [f"{k}={v}" for k, v in extra.items()]
    return load_config(path, overrides)


def test_separation_trains_and_resumes(tmp_path):
    cfg = _cfg("configs/separation/unet_sisdr.yaml", tmp_path, "sep", **{
        "data.epoch_size": 16, "data.val_size": 8, "data.segment_sec": 1.0, "batch_size": 4})
    train(cfg)
    run = tmp_path / "sep"
    assert (run / "best.pt").exists() and (run / "last.pt").exists()
    cfg["epochs"] = 3  # продолжение с последней эпохи
    train(cfg)
    lines = (run / "metrics.jsonl").read_text().splitlines()
    assert [json.loads(l)["epoch"] for l in lines] == [1, 2, 3]
    model, task, _ = load_trained(run / "best.pt")
    assert type(model).__name__ == "UNetMask"


def test_emotion_trains_and_dumps_embeddings(tmp_path):
    cfg = _cfg("configs/emotion/mel_cnn.yaml", tmp_path, "emo", **{
        "data.synthetic_size": 64, "data.max_sec": 1.0, "batch_size": 8, "loss.name": "focal_weighted"})
    train(cfg)
    emb = np.load(tmp_path / "emo" / "embeddings" / "epoch_002.npz")
    assert emb["emb"].shape[1] == 128 and len(emb["labels"]) == len(emb["emb"])


def test_vae_and_latent_classifier(tmp_path):
    vae_cfg = _cfg("configs/vae/vae_mse.yaml", tmp_path, "vae", **{"data.synthetic_size": 64, "batch_size": 16})
    train(vae_cfg)
    emo_cfg = _cfg("configs/emotion/latent_vae.yaml", tmp_path, "emo_latent", **{
        "data.synthetic_size": 32, "data.max_sec": 2.0, "batch_size": 8,
        "model.vae_checkpoint": str(tmp_path / "vae" / "best.pt")})
    train(emo_cfg)
    model, task, _ = load_trained(tmp_path / "emo_latent" / "best.pt")
    assert type(model).__name__ == "LatentClassifier"
