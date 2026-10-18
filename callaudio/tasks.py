"""Описание трёх задач для общего цикла обучения (train.py).

Каждая задача умеет: собрать датасеты и модель, посчитать потерю на батче,
провалидироваться (метрика `score`, больше - лучше) и выдать эмбеддинги
фиксированной выборки для визуализации пространства признаков.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from . import losses as L
from .data import CommandsMelDataset, EmotionDataset, SeparationDataset
from .metrics import classification_metrics
from .models import ConvVAE, LatentClassifier, MelCNN, SSLClassifier, WaveCNN, build_separation_model
from .utils import load_checkpoint


def _to(batch: dict, device) -> dict:
    return {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v for k, v in batch.items()}


class BaseTask:
    name = "base"

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.data_cfg = cfg.get("data", {})
        self.model_cfg = cfg.get("model", {})
        self.loss_cfg = cfg.get("loss", {})
        self.criterion: nn.Module | None = None

    # переопределяются в задачах
    def build_datasets(self):
        raise NotImplementedError

    def build_model(self, extra: dict | None = None) -> nn.Module:
        raise NotImplementedError

    def loss(self, model, batch, epoch: int, frac: float):
        raise NotImplementedError

    def validate(self, model, loader, device) -> dict:
        raise NotImplementedError

    def extra_state(self) -> dict:
        return {}

    def probe_labels(self) -> list[str] | None:
        return None

    @torch.no_grad()
    def probe(self, model, dataset, device, size: int = 2000):
        """Эмбеддинги одной и той же сбалансированной выборки после каждой эпохи."""
        return None

    @staticmethod
    def balanced_indices(targets: list[int], size: int, seed: int = 0) -> list[int]:
        rng = np.random.default_rng(seed)
        targets = np.asarray(targets)
        classes = np.unique(targets)
        per = max(1, size // len(classes))
        idx = []
        for c in classes:
            members = np.flatnonzero(targets == c)
            idx += rng.choice(members, size=min(per, len(members)), replace=False).tolist()
        return sorted(idx)


# ===================== модуль 1 =====================

class SeparationTask(BaseTask):
    name = "separation"

    def build_datasets(self):
        d = self.data_cfg
        common = dict(segment_sec=d.get("segment_sec", 4.0), snr_range=tuple(d.get("snr_range", (-5, 15))),
                      telephone_prob=d.get("telephone_prob", 0.0), synthetic=d.get("synthetic", False))
        train = SeparationDataset(d.get("speech_train"), d.get("noise_train"), epoch_size=d.get("epoch_size", 4000), **common)
        val = SeparationDataset(d.get("speech_val"), d.get("noise_val"), epoch_size=d.get("val_size", 300),
                                deterministic=True, seed=1, snr_grid=d.get("snr_grid", (-5, 0, 5, 10, 15)), **common)
        return train, val

    def build_model(self, extra=None):
        params = {k: v for k, v in self.model_cfg.items() if k != "name"}
        model = build_separation_model(self.model_cfg.get("name", "unet"), **params)
        params_l = {k: v for k, v in self.loss_cfg.items() if k != "name"}
        self.criterion = L.build_separation_loss(self.loss_cfg.get("name", "si_sdr"), **params_l)
        return model

    def loss(self, model, batch, epoch, frac):
        est = model(batch["mix"])
        loss = self.criterion(est, batch["clean"])
        return loss, {"loss": loss.item(), "si_sdr": L.si_sdr(est.detach().float(), batch["clean"]).mean().item()}

    @torch.no_grad()
    def validate(self, model, loader, device):
        model.eval()
        est_sdr, mix_sdr = [], []
        for batch in loader:
            batch = _to(batch, device)
            est = model(batch["mix"]).float()
            est_sdr.append(L.si_sdr(est, batch["clean"]).cpu())
            mix_sdr.append(L.si_sdr(batch["mix"], batch["clean"]).cpu())
        est_sdr, mix_sdr = torch.cat(est_sdr), torch.cat(mix_sdr)
        sdri = (est_sdr - mix_sdr).mean().item()
        return {"si_sdr": est_sdr.mean().item(), "si_sdr_mix": mix_sdr.mean().item(), "si_sdri": sdri, "score": sdri}


# ===================== модуль 2 =====================

class EmotionTask(BaseTask):
    name = "emotion"

    def __init__(self, cfg):
        super().__init__(cfg)
        self.labels = list(self.data_cfg.get("labels", ["neutral", "angry", "positive", "sad"]))
        self.train_targets: list[int] = []
        self.vae_extra: dict = {}

    def build_datasets(self):
        d = self.data_cfg
        common = dict(labels=self.labels, max_sec=d.get("max_sec", 4.0), synthetic=d.get("synthetic", False))
        train = EmotionDataset(d.get("train"), train=True, noise_manifest=d.get("noise_manifest"),
                               noise_prob=d.get("noise_prob", 0.0), snr_range=tuple(d.get("snr_range", (0, 20))),
                               synthetic_size=d.get("synthetic_size", 512), **common)
        val = EmotionDataset(d.get("val"), train=False, seed=1, synthetic_size=d.get("synthetic_size", 512) // 4, **common)
        self.train_targets = train.targets()
        return train, val

    def build_model(self, extra=None):
        m = dict(self.model_cfg)
        name = m.pop("name", "mel_cnn")
        n = len(self.labels)
        if name == "wave_cnn":
            model = WaveCNN(num_classes=n, **m)
        elif name == "mel_cnn":
            model = MelCNN(num_classes=n, **m)
        elif name == "ssl":
            model = SSLClassifier(num_classes=n, **m)
        elif name == "latent":
            vae_ckpt = m.pop("vae_checkpoint", None)
            if extra and "vae_cfg" in extra:  # загрузка из сохранённого чекпойнта классификатора
                self.vae_extra = {k: extra[k] for k in ("vae_cfg", "mel_mean", "mel_std")}
                vae = ConvVAE(**extra["vae_cfg"])
            else:
                ck = load_checkpoint(vae_ckpt)
                vae_cfg = {k: v for k, v in ck["cfg"]["model"].items() if k != "name"}
                if vae_cfg.pop("conditional", False):
                    raise ValueError("Для признаков нужен безусловный VAE (conditional: false)")
                vae = ConvVAE(**vae_cfg)
                vae.load_state_dict(ck["model"])
                self.vae_extra = {"vae_cfg": vae_cfg, "mel_mean": ck["extra"]["mel_mean"], "mel_std": ck["extra"]["mel_std"]}
            model = LatentClassifier(vae, self.vae_extra["mel_mean"], self.vae_extra["mel_std"], num_classes=n, **m)
        else:
            raise ValueError(f"Неизвестная модель эмоций: {name}")
        self.criterion = L.build_classification_loss(
            self.loss_cfg.get("name", "ce"), labels=self.train_targets or None, num_classes=n,
            gamma=self.loss_cfg.get("gamma", 2.0), label_smoothing=self.loss_cfg.get("label_smoothing", 0.0))
        return model

    def loss(self, model, batch, epoch, frac):
        logits = model(batch["wav"])
        loss = self.criterion(logits.float(), batch["label"])
        acc = (logits.argmax(-1) == batch["label"]).float().mean().item()
        return loss, {"loss": loss.item(), "acc": acc}

    @torch.no_grad()
    def predict(self, model, loader, device):
        model.eval()
        preds, targets, probs = [], [], []
        for batch in loader:
            batch = _to(batch, device)
            logits = model(batch["wav"]).float()
            probs.append(torch.softmax(logits, -1).cpu())
            preds.append(logits.argmax(-1).cpu())
            targets.append(batch["label"].cpu())
        return torch.cat(preds).numpy(), torch.cat(targets).numpy(), torch.cat(probs).numpy()

    def validate(self, model, loader, device):
        preds, targets, _ = self.predict(model, loader, device)
        m = classification_metrics(targets, preds, num_classes=len(self.labels))
        out = {"ua": m["ua"], "wa": m["wa"], "macro_f1": m["macro_f1"], "score": m["macro_f1"]}
        if isinstance(model, SSLClassifier):
            w = model.layer_importance().cpu().numpy()
            out["best_layer"] = int(w.argmax())
        return out

    def extra_state(self):
        return {"labels": self.labels, **self.vae_extra}

    def probe_labels(self):
        return self.labels

    @torch.no_grad()
    def probe(self, model, dataset, device, size=2000):
        idx = self.balanced_indices(dataset.targets(), size)
        loader = DataLoader(Subset(dataset, idx), batch_size=self.cfg.get("eval_batch_size", 32), num_workers=0)
        model.eval()
        embs, labels = [], []
        for batch in loader:
            batch = _to(batch, device)
            embs.append(model.embed(batch["wav"]).float().cpu())
            labels.append(batch["label"].cpu())
        return torch.cat(embs).numpy(), torch.cat(labels).numpy()


# ===================== модуль 3 =====================

class VAETask(BaseTask):
    name = "vae"

    def __init__(self, cfg):
        super().__init__(cfg)
        self.mel_mean = cfg.get("data", {}).get("mel_mean")
        self.mel_std = cfg.get("data", {}).get("mel_std")
        self.words: list[str] = []

    def build_datasets(self):
        d = self.data_cfg
        train = CommandsMelDataset(d.get("train"), words=d.get("words"), synthetic=d.get("synthetic", False),
                                   synthetic_size=d.get("synthetic_size", 512))
        val = CommandsMelDataset(d.get("val"), words=train.words, synthetic=d.get("synthetic", False),
                                 synthetic_size=d.get("synthetic_size", 512) // 4, seed=1)
        self.words = train.words
        if self.mel_mean is None:  # глобальные среднее и разброс лог-мел по части обучающей выборки
            rng = np.random.default_rng(0)
            idx = rng.choice(len(train), size=min(len(train), d.get("stats_samples", 500)), replace=False)
            mels = torch.stack([train[int(i)]["mel"] for i in idx])
            self.mel_mean, self.mel_std = float(mels.mean()), float(mels.std())
            print(f"[vae] статистика лог-мел: mean={self.mel_mean:.3f}, std={self.mel_std:.3f}")
        return train, val

    def norm(self, mel):
        return (mel - self.mel_mean) / self.mel_std

    def denorm(self, mel):
        return mel * self.mel_std + self.mel_mean

    def build_model(self, extra=None):
        if extra:
            self.mel_mean, self.mel_std = extra.get("mel_mean", self.mel_mean), extra.get("mel_std", self.mel_std)
            self.words = extra.get("words", self.words)
        m = dict(self.model_cfg)
        m.pop("name", None)
        conditional = m.pop("conditional", False)
        return ConvVAE(num_classes=len(self.words) if conditional else 0, **m)

    def _labels(self, model, batch):
        return batch["label"] if model.num_classes > 0 else None

    def loss(self, model, batch, epoch, frac):
        x = self.norm(batch["mel"])
        recon, mu, logvar = model(x, self._labels(model, batch))
        beta = L.beta_schedule(epoch, frac, self.loss_cfg.get("beta", 1.0), self.loss_cfg.get("warmup_epochs", 5))
        loss, logs = L.vae_loss(recon.float(), x, mu.float(), logvar.float(), beta=beta,
                                recon_type=self.loss_cfg.get("recon", "mse"))
        return loss, {"loss": loss.item(), **logs}

    @torch.no_grad()
    def validate(self, model, loader, device):
        model.eval()
        mse, kl, n = 0.0, 0.0, 0
        for batch in loader:
            batch = _to(batch, device)
            x = self.norm(batch["mel"])
            recon, mu, logvar = model(x, self._labels(model, batch))
            mse += torch.mean((recon.float() - x) ** 2, dim=(1, 2)).sum().item()
            kl += L.kl_divergence(mu.float(), logvar.float()).sum().item()
            n += x.shape[0]
        return {"recon_mse": mse / n, "kl": kl / n, "score": -mse / n}

    def extra_state(self):
        return {"mel_mean": self.mel_mean, "mel_std": self.mel_std, "words": self.words}

    def probe_labels(self):
        return self.words

    @torch.no_grad()
    def probe(self, model, dataset, device, size=2000):
        targets = [dataset.word2id[r["label"]] for r in dataset.rows]
        idx = self.balanced_indices(targets, size)
        loader = DataLoader(Subset(dataset, idx), batch_size=128, num_workers=0)
        model.eval()
        embs, labels = [], []
        for batch in loader:
            batch = _to(batch, device)
            mu, _ = model.encode(self.norm(batch["mel"]), self._labels(model, batch))
            embs.append(mu.float().cpu())
            labels.append(batch["label"].cpu())
        return torch.cat(embs).numpy(), torch.cat(labels).numpy()


TASKS = {"separation": SeparationTask, "emotion": EmotionTask, "vae": VAETask}


def build_task(cfg: dict) -> BaseTask:
    return TASKS[cfg["task"]](cfg)


def load_trained(checkpoint: str | Path, device="cpu", which: str = "model"):
    """Восстанавливает (модель, задачу) из чекпойнта для оценки и демо."""
    ck = load_checkpoint(checkpoint)
    task = build_task(ck["cfg"])
    extra = ck.get("extra", {})
    if isinstance(task, EmotionTask):
        task.labels = extra.get("labels", task.labels)
    model = task.build_model(extra=extra)
    model.load_state_dict(ck[which])
    return model.to(device).eval(), task, ck
