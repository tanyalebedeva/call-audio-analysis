"""Общий цикл обучения для всех модулей.

    python -m callaudio.train --config configs/separation/unet_sisdr.yaml
    python -m callaudio.train --config configs/emotion/mel_cnn.yaml --set loss.name=focal out_dir=runs/emo_mel_focal

Особенности для Colab:
* чекпойнт last.pt пишется после каждой эпохи; повторный запуск той же команды продолжит обучение;
* max_minutes — мягкий лимит: обучение остановится и сохранится до того, как Colab оборвёт сессию;
* после каждой эпохи сохраняются эмбеддинги фиксированной выборки (embeddings/epoch_XXX.npz)
  для визуализации пространства признаков.
"""
from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .tasks import build_task
from .utils import (AverageMeter, JsonlLogger, count_parameters, get_device, load_checkpoint, load_config,
                    save_checkpoint, save_config, set_seed)


def warmup_cosine(total_steps: int, warmup_steps: int):
    def fn(step):
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))
    return fn


def train(cfg: dict) -> dict:
    out = Path(cfg["out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    set_seed(cfg.get("seed", 0))
    device = get_device(cfg.get("device", "auto"))
    task = build_task(cfg)
    train_ds, val_ds = task.build_datasets()

    resume_path = out / "last.pt"
    ck = load_checkpoint(resume_path) if cfg.get("resume", True) and resume_path.exists() else None
    model = task.build_model(extra=ck.get("extra") if ck else None).to(device)
    n_params, n_train = count_parameters(model), count_parameters(model, trainable_only=True)
    print(f"[train] {cfg['task']} · {type(model).__name__} · параметров {n_params / 1e6:.2f} M "
          f"(обучаемых {n_train / 1e6:.2f} M) · устройство {device}")

    workers = cfg.get("num_workers", 2)
    train_loader = DataLoader(train_ds, batch_size=cfg.get("batch_size", 16), shuffle=True, num_workers=workers,
                              drop_last=True, pin_memory=device.type == "cuda", persistent_workers=workers > 0)
    val_loader = DataLoader(val_ds, batch_size=cfg.get("eval_batch_size", cfg.get("batch_size", 16)),
                            shuffle=False, num_workers=workers)

    epochs = cfg.get("epochs", 20)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=float(cfg.get("lr", 1e-3)), weight_decay=float(cfg.get("weight_decay", 1e-4)))
    total = epochs * len(train_loader)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, warmup_cosine(total, int(cfg.get("warmup_frac", 0.05) * total)))
    use_amp = bool(cfg.get("amp", True)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    start_epoch, best = 0, -float("inf")
    if ck:
        model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        scheduler.load_state_dict(ck["scheduler"])
        scaler.load_state_dict(ck["scaler"])
        start_epoch, best = ck["epoch"] + 1, ck["best_score"]
        print(f"[train] продолжаю с эпохи {start_epoch + 1}, лучший score {best:.4f}")
    else:
        save_config(cfg, out / "config.yaml")

    logger = JsonlLogger(out / "metrics.jsonl", use_wandb=cfg.get("wandb", {}).get("enabled", False),
                         wandb_cfg={"project": cfg.get("wandb", {}).get("project", "call-audio-analysis"),
                                    "name": out.name, "config": cfg, "id": out.name})
    t0 = time.time()
    max_minutes = cfg.get("max_minutes")
    log_every = cfg.get("log_every", 50)
    grad_clip = cfg.get("grad_clip", 5.0)

    for epoch in range(start_epoch, epochs):
        model.train()
        meter = AverageMeter()
        for step, batch in enumerate(train_loader):
            batch = {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v for k, v in batch.items()}
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                loss, logs = task.loss(model, batch, epoch, step / len(train_loader))
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(params, grad_clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            meter.update(logs, n=1)
            if (step + 1) % log_every == 0:
                avg = meter.averages()
                print(f"  эпоха {epoch + 1} шаг {step + 1}/{len(train_loader)} " +
                      " ".join(f"{k}={v:.4f}" for k, v in avg.items()))

        val = task.validate(model, val_loader, device)
        record = {"epoch": epoch + 1, "lr": scheduler.get_last_lr()[0],
                  **{f"train_{k}": v for k, v in meter.averages().items()},
                  **{f"val_{k}": v for k, v in val.items() if isinstance(v, (int, float))}}
        logger.log(record)
        print(f"[эпоха {epoch + 1}/{epochs}] " + " ".join(f"{k}={v:.4f}" for k, v in record.items()
                                                         if isinstance(v, float) and k != "lr"))

        probe = task.probe(model, val_ds, device, size=cfg.get("probe_size", 2000))
        if probe is not None:
            (out / "embeddings").mkdir(exist_ok=True)
            np.savez_compressed(out / "embeddings" / f"epoch_{epoch + 1:03d}.npz", emb=probe[0], labels=probe[1],
                                names=np.array(task.probe_labels() or []))

        state = dict(model=model.state_dict(), optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
                     scaler=scaler.state_dict(), epoch=epoch, best_score=max(best, val["score"]), cfg=cfg,
                     extra=task.extra_state())
        if val["score"] > best:
            best = val["score"]
            save_checkpoint(out / "best.pt", **state)
            print(f"  ↑ новый лучший score {best:.4f}, сохранён best.pt")
        save_checkpoint(out / "last.pt", **state)

        if max_minutes and (time.time() - t0) / 60 > max_minutes and epoch + 1 < epochs:
            print(f"[train] достигнут лимит {max_minutes} мин — остановка. Запустите ту же команду, чтобы продолжить.")
            break
    return {"best_score": best, "out_dir": str(out)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--set", nargs="*", default=[], help="переопределения: key.sub=value")
    a = p.parse_args()
    cfg = load_config(a.config, a.set)
    result = train(cfg)
    print(f"[train] готово: {result}")


if __name__ == "__main__":
    main()
