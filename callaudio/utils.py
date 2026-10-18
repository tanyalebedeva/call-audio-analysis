"""Общие утилиты: конфиги, seed, чекпойнты, логирование."""
from __future__ import annotations

import copy
import json
import os
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


# ---------- конфиги ----------

def load_config(path: str | os.PathLike, overrides: list[str] | None = None) -> dict:
    """Читает YAML. Поддерживает наследование через ключ `base:` и переопределения `a.b=c`."""
    path = Path(path)
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    if "base" in cfg:
        base = load_config(path.parent / cfg.pop("base"))
        cfg = deep_merge(base, cfg)
    for item in overrides or []:
        key, value = item.split("=", 1)
        set_by_path(cfg, key, yaml.safe_load(value))
    return resolve_placeholders(cfg)


def resolve_placeholders(cfg: dict) -> dict:
    """Подставляет {data_root} и {runs_root} в строки конфига — пути задаются в одном месте."""
    roots = {k: str(cfg[k]) for k in ("data_root", "runs_root") if k in cfg}

    def walk(node):
        if isinstance(node, dict):
            return {k: walk(v) for k, v in node.items()}
        if isinstance(node, list):
            return [walk(v) for v in node]
        if isinstance(node, str) and "{" in node:
            for k, v in roots.items():
                node = node.replace("{" + k + "}", v)
        return node

    return walk(cfg)


def deep_merge(base: dict, other: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in other.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def set_by_path(cfg: dict, dotted: str, value: Any) -> None:
    keys = dotted.split(".")
    node = cfg
    for k in keys[:-1]:
        node = node.setdefault(k, {})
    node[keys[-1]] = value


def save_config(cfg: dict, path: str | os.PathLike) -> None:
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)


# ---------- воспроизводимость ----------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(name: str = "auto") -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def count_parameters(model: torch.nn.Module, trainable_only: bool = False) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad or not trainable_only)


# ---------- чекпойнты ----------

def save_checkpoint(path: str | os.PathLike, **state) -> None:
    """Атомарная запись: сначала во временный файл, потом переименование.
    Если Colab оборвёт сессию посреди записи, прошлый чекпойнт не испортится."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, tmp)
    os.replace(tmp, path)


def load_checkpoint(path: str | os.PathLike, map_location="cpu") -> dict:
    return torch.load(path, map_location=map_location, weights_only=False)


# ---------- логирование ----------

class JsonlLogger:
    def __init__(self, path: str | os.PathLike, use_wandb: bool = False, wandb_cfg: dict | None = None):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.wandb = None
        if use_wandb:
            try:
                import wandb

                self.wandb = wandb.init(**(wandb_cfg or {}), resume="allow")
            except Exception as e:  # W&B не обязателен
                print(f"[logger] W&B отключён: {e}")

    def log(self, record: dict) -> None:
        record = {"time": time.time(), **record}
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        if self.wandb is not None:
            self.wandb.log({k: v for k, v in record.items() if isinstance(v, (int, float))})


class AverageMeter:
    def __init__(self):
        self.sums: dict[str, float] = {}
        self.counts: dict[str, int] = {}

    def update(self, values: dict[str, float], n: int = 1) -> None:
        for k, v in values.items():
            self.sums[k] = self.sums.get(k, 0.0) + float(v) * n
            self.counts[k] = self.counts.get(k, 0) + n

    def averages(self) -> dict[str, float]:
        return {k: self.sums[k] / max(self.counts[k], 1) for k in self.sums}


def read_manifest(path: str | os.PathLike) -> list[dict]:
    """Манифест — JSONL: одна строка = одна запись (path, duration, label, text, ...)."""
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_manifest(rows: list[dict], path: str | os.PathLike) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
