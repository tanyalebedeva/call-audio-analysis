"""Сводные таблицы всех экспериментов для пояснительной записки.

    python -m callaudio.report --runs /content/drive/MyDrive/coursework/runs --out reports

Собирает results_*.json и лучшие значения из metrics.jsonl каждого прогона
и пишет markdown- и CSV-таблицы: reports/separation.md, emotion.md, vae.md.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import yaml


def _best_val(run: Path) -> dict:
    path = run / "metrics.jsonl"
    if not path.exists():
        return {}
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    best = max(rows, key=lambda r: r.get("val_score", float("-inf")))
    return {"epochs": len(rows), "best_epoch": best["epoch"], **{k: v for k, v in best.items() if k.startswith("val_")}}


def _write(rows: list[dict], columns: list[str], out: Path, name: str, title: str):
    if not rows:
        return
    out.mkdir(parents=True, exist_ok=True)
    with open(out / f"{name}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    def fmt(v):
        return f"{v:.3f}" if isinstance(v, float) else ("" if v is None else str(v))

    lines = [f"## {title}", "", "| " + " | ".join(columns) + " |", "|" + " --- |" * len(columns)]
    lines += ["| " + " | ".join(fmt(r.get(c)) for c in columns) + " |" for r in rows]
    (out / f"{name}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines) + "\n")


def build_report(runs_root: str | Path, out: str | Path):
    runs_root, out = Path(runs_root), Path(out)
    sep, emo, vae = [], [], []
    for run in sorted(p for p in runs_root.iterdir() if (p / "config.yaml").exists()):
        cfg = yaml.safe_load((run / "config.yaml").read_text(encoding="utf-8"))
        base = {"run": run.name, "model": cfg.get("model", {}).get("name"), "loss": cfg.get("loss", {}).get("name"),
                **_best_val(run)}
        if cfg["task"] == "separation":
            r = run / "results_separation.json"
            if r.exists():
                res = json.loads(r.read_text(encoding="utf-8"))
                o = res["overall"]
                base.update({k: o.get(k) for k in ("si_sdri", "pesq", "pesq_mix", "stoi", "stoi_mix")})
                for snr, m in res["by_snr"].items():
                    base[f"si_sdri@{snr}dB"] = m.get("si_sdri")
                if "wer" in res:
                    base.update({f"wer_{k}": v for k, v in res["wer"].items()})
            sep.append(base)
        elif cfg["task"] == "emotion":
            for f in sorted(run.glob("results_emotion*.json")):
                res = json.loads(f.read_text(encoding="utf-8"))
                cond = f.stem.replace("results_emotion", "") or "clean"
                emo.append({**base, "condition": cond.strip("_"), "ua": res["ua"], "wa": res["wa"], "macro_f1": res["macro_f1"]})
            if not list(run.glob("results_emotion*.json")):
                emo.append(base)
        elif cfg["task"] == "vae":
            r = run / "results_vae.json"
            if r.exists():
                base.update({k: v for k, v in json.loads(r.read_text(encoding="utf-8")).items() if isinstance(v, float)})
            base["beta"] = cfg.get("loss", {}).get("beta")
            base["recon"] = cfg.get("loss", {}).get("recon")
            vae.append(base)

    snr_cols = sorted({k for r in sep for k in r if k.startswith("si_sdri@")}, key=lambda s: float(s.split("@")[1][:-2]))
    _write(sep, ["run", "model", "loss", "best_epoch", "val_si_sdri", "si_sdri", "pesq", "pesq_mix", "stoi", "stoi_mix",
                 *snr_cols, "wer_clean", "wer_noisy", "wer_enhanced"], out, "separation", "Модуль 1: отделение речи от фона")
    _write(emo, ["run", "model", "loss", "condition", "best_epoch", "ua", "wa", "macro_f1"], out, "emotion",
           "Модуль 2: распознавание эмоций")
    _write(vae, ["run", "recon", "beta", "best_epoch", "recon_mse_logmel", "latent_linear_probe_acc", "latent_silhouette",
                 "judge_acc_real", "judge_acc_recon", "judge_acc_generated"], out, "vae", "Модуль 3: VAE")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", required=True)
    p.add_argument("--out", default="reports")
    a = p.parse_args()
    build_report(a.runs, a.out)


if __name__ == "__main__":
    main()
