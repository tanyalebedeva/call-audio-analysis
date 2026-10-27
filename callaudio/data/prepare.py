"""Скачивание и подготовка датасетов. Запускается в Colab:
 
    python -m callaudio.data.prepare golos    --out /content/drive/MyDrive/coursework/data
    python -m callaudio.data.prepare musan    --out ... --music-files 300
    python -m callaudio.data.prepare dusha    --out ... --per-class 6000
    python -m callaudio.data.prepare dusha_windows --out ...   # окна по 1 с для VAE на русской речи
    python -m callaudio.data.prepare commands --out ... --words core
    python -m callaudio.data.prepare folder   --audio-dir my_calls --out ... --name mycalls
 
Каждая команда кладёт аудио (16 кГц, моно, FLAC) и манифесты JSONL:
    {"path": ..., "duration": ..., "label"/"text"/"kind": ...}
Разбиение train/val/test делается по файлам, чтобы один и тот же файл не попал в обе части.
"""
from __future__ import annotations
 
import argparse
import io
import json
import random
import shutil
import tarfile
from collections import Counter, defaultdict
from pathlib import Path
 
import numpy as np
import soundfile as sf
 
from ..audio import SAMPLE_RATE, load_audio, resample
from ..utils import write_manifest
 
GOLOS_TEST_URL = "https://sc.link/Kqr"  # test.tar, ~1,3 ГБ, ~12,6 ч речи с расшифровками
GOLOS_HF = {"crowd": "bond005/sberdevices_golos_10h_crowd", "farfield": "bond005/sberdevices_golos_100h_farfield"}
MUSAN_URL = "https://openslr.trmal.net/resources/17/musan.tar.gz"
MUSAN_MIRROR = "https://openslr.elda.org/resources/17/musan.tar.gz"
COMMANDS_URL = "http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz"
CORE_WORDS = ["yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go"]
DUSHA_HF = "xbgoose/dusha"
EMOTIONS = ["neutral", "angry", "positive", "sad"]
 
 
# ---------- общие помощники ----------
 
def download(url: str, dest: Path, mirrors: tuple[str, ...] = ()) -> Path:
    import requests
 
    if dest.exists() and dest.stat().st_size > 0:
        print(f"[skip] уже скачано: {dest}")
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    last_err = None
    for u in (url, *mirrors):
        try:
            print(f"[download] {u}")
            with requests.get(u, stream=True, timeout=60, headers={"User-Agent": "Mozilla/5.0"}) as r:
                r.raise_for_status()
                total = int(r.headers.get("content-length", 0))
                done = 0
                tmp = dest.with_suffix(dest.suffix + ".part")
                with open(tmp, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1 << 20):
                        f.write(chunk)
                        done += len(chunk)
                        if total and done % (200 << 20) < (1 << 20):
                            print(f"  {done / 2**30:.1f} / {total / 2**30:.1f} ГБ")
                if done == 0 or (total and done < total):
                    tmp.unlink(missing_ok=True)
                    raise IOError(f"скачано {done} из {total} байт")
                tmp.rename(dest)
                return dest
        except Exception as e:  # пробуем зеркало
            last_err = e
            print(f"  ошибка: {e}")
    raise RuntimeError(f"Не удалось скачать {url}: {last_err}")
 
 
def save_flac(wav: np.ndarray, path: Path) -> float:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), wav, SAMPLE_RATE, format="FLAC")
    return round(len(wav) / SAMPLE_RATE, 3)
 
 
def split_rows(rows: list[dict], seed: int = 0, val: float = 0.1, test: float = 0.1) -> dict[str, list[dict]]:
    rows = rows[:]
    random.Random(seed).shuffle(rows)
    n_test, n_val = int(len(rows) * test), int(len(rows) * val)
    return {"test": rows[:n_test], "val": rows[n_test:n_test + n_val], "train": rows[n_test + n_val:]}
 
 
def write_splits(splits: dict[str, list[dict]], out: Path, prefix: str) -> None:
    for name, rows in splits.items():
        write_manifest(rows, out / f"{prefix}_{name}.jsonl")
        print(f"[manifest] {prefix}_{name}.jsonl: {len(rows)} записей, "
              f"{sum(r.get('duration', 0) for r in rows) / 3600:.2f} ч")
 
 
def convert_tree(files: list[Path], src_root: Path, dst_root: Path) -> dict[Path, float]:
    """Конвертирует аудио в 16 кГц моно FLAC, сохраняя структуру папок."""
    out = {}
    for i, f in enumerate(files):
        dst = dst_root / f.relative_to(src_root).with_suffix(".flac")
        if not dst.exists():
            try:
                out[dst] = save_flac(load_audio(str(f)), dst)
            except Exception as e:
                print(f"  пропуск {f}: {e}")
                continue
        else:
            out[dst] = round(sf.info(str(dst)).duration, 3)
        if (i + 1) % 1000 == 0:
            print(f"  {i + 1} / {len(files)}")
    return out
 
 
# ---------- Golos ----------
 
def prepare_golos(out: Path, work: Path) -> None:
    """Тестовая часть Golos: речь с расшифровками для смесей и WER."""
    tar_path = work / "golos_test.tar"
    extract_dir = work / "golos_test"
    try:
        download(GOLOS_TEST_URL, tar_path)
        if not extract_dir.exists():
            print("[extract] golos_test.tar")
            with tarfile.open(tar_path) as t:
                t.extractall(extract_dir)
    except Exception as e:
        print(f"[golos] архив недоступен: {e}")
        tar_path.unlink(missing_ok=True)
        shutil.rmtree(extract_dir, ignore_errors=True)
    if not extract_dir.exists() or not any(extract_dir.rglob("*.wav")):
        prepare_golos_hf(out)
        return
    texts = {}
    for manifest in extract_dir.rglob("*.jsonl"):  # манифесты NeMo: audio_filepath, text, duration
        for line in open(manifest, encoding="utf-8"):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "audio_filepath" in r:
                texts[(manifest.parent / r["audio_filepath"]).resolve()] = r.get("text", "")
    wavs = sorted(extract_dir.rglob("*.wav"))
    print(f"[golos] найдено {len(wavs)} wav, расшифровок: {len(texts)}")
    durations = convert_tree(wavs, extract_dir, out / "golos")
    rows = []
    for wav in wavs:
        dst = out / "golos" / wav.relative_to(extract_dir).with_suffix(".flac")
        if dst in durations:
            rows.append({"path": str(dst), "duration": durations[dst],
                         "text": texts.get(wav.resolve(), ""), "domain": "farfield" if "farfield" in str(wav) else "crowd"})
    write_splits(split_rows(rows), out / "manifests", "speech")
 
 
def prepare_golos_hf(out: Path) -> None:
    """Тестовая часть Golos с Hugging Face."""
    from datasets import Audio, load_dataset
 
    rows = []
    for domain, name in GOLOS_HF.items():
        ds = load_dataset(name, split="test", streaming=True).cast_column("audio", Audio(decode=False))
        n = 0
        for i, ex in enumerate(ds):
            text = ex.get("transcription") or ex.get("text") or ""
            try:
                wav, sr = sf.read(io.BytesIO(ex["audio"]["bytes"]), dtype="float32", always_2d=True)
            except Exception:
                continue
            path = out / "golos" / domain / f"{i:06d}.flac"
            if path.exists():
                duration = round(sf.info(str(path)).duration, 3)
            else:
                duration = save_flac(resample(wav.mean(axis=1), sr, SAMPLE_RATE), path)
            rows.append({"path": str(path), "duration": duration, "text": text, "domain": domain})
            n += 1
        print(f"[golos] {domain}: {n} записей")
    write_splits(split_rows(rows), out / "manifests", "speech")
 
 
# ---------- MUSAN ----------
 
def prepare_musan(out: Path, work: Path, music_files: int = 300) -> None:
    """Шум и музыка из MUSAN. Речевую часть не берём — она конфликтует с задачей «речь против фона»."""
    tar_path = download(MUSAN_URL, work / "musan.tar.gz", mirrors=(MUSAN_MIRROR,))
    extract_dir = work / "musan_extracted"
    if not extract_dir.exists():
        print("[extract] musan: noise и music (это несколько минут)")
        with tarfile.open(tar_path) as t:
            members = [m for m in t.getmembers() if m.name.startswith(("musan/noise", "musan/music"))]
            t.extractall(extract_dir, members=members)
    root = extract_dir / "musan"
    noise = sorted((root / "noise").rglob("*.wav"))
    music = sorted((root / "music").rglob("*.wav"))
    random.Random(0).shuffle(music)
    music = music[:music_files]
    print(f"[musan] шум: {len(noise)} файлов, музыка: {len(music)} файлов")
    durations = convert_tree(noise + music, root, out / "musan")
    rows = [{"path": str(p), "duration": d, "kind": "music" if "/music/" in str(p) else "noise"}
            for p, d in durations.items()]
    write_splits(split_rows(rows), out / "manifests", "noise")
 
 
# ---------- Dusha ----------
 
def prepare_dusha(out: Path, per_class: int = 6000, test_per_class: int = 1000, max_sec: float = 8.0,
                  labels=EMOTIONS) -> None:
    """Подмножество Dusha с Hugging Face. Аудио декодируем сами (decode=False),
    чтобы не зависеть от аудио-бэкенда библиотеки datasets."""
    from datasets import Audio, load_dataset
 
    for split, cap in (("train", per_class), ("test", test_per_class)):
        ds = load_dataset(DUSHA_HF, split=split, streaming=True).cast_column("audio", Audio(decode=False))
        counts: Counter = Counter()
        rows = []
        for i, ex in enumerate(ds):
            label = str(ex["emotion"]).lower()
            if label not in labels or counts[label] >= cap:
                if all(counts[l] >= cap for l in labels):
                    break
                continue
            audio = ex["audio"]
            try:
                wav, sr = sf.read(io.BytesIO(audio["bytes"]), dtype="float32", always_2d=True)
            except Exception:
                continue
            wav = resample(wav.mean(axis=1), sr, SAMPLE_RATE)[: int(max_sec * SAMPLE_RATE)]
            if len(wav) < SAMPLE_RATE // 2:  # короче 0,5 с — мало информации об эмоции
                continue
            path = out / "dusha" / split / label / f"{i:07d}.flac"
            rows.append({"path": str(path), "duration": save_flac(wav, path), "label": label})
            counts[label] += 1
            if sum(counts.values()) % 2000 == 0:
                print(f"  {split}: {dict(counts)}")
        print(f"[dusha] {split}: {dict(counts)}")
        if split == "train":
            # валидация — 10 % train со стратификацией по классам
            by_label = defaultdict(list)
            for r in rows:
                by_label[r["label"]].append(r)
            train, val = [], []
            for label_rows in by_label.values():
                random.Random(0).shuffle(label_rows)
                k = max(1, len(label_rows) // 10)
                val += label_rows[:k]
                train += label_rows[k:]
            write_splits({"train": train, "val": val}, out / "manifests", "emotion")
        else:
            write_splits({"test": rows}, out / "manifests", "emotion")
 
 
# ---------- Speech Commands ----------
 
def prepare_commands(out: Path, work: Path, words: str = "core") -> None:
    tar_path = download(COMMANDS_URL, work / "speech_commands_v0.02.tar.gz")
    extract_dir = work / "speech_commands"
    if not extract_dir.exists():
        print("[extract] speech commands")
        extract_dir.mkdir(parents=True)
        with tarfile.open(tar_path) as t:
            t.extractall(extract_dir)
    val_list = set((extract_dir / "validation_list.txt").read_text().split())
    test_list = set((extract_dir / "testing_list.txt").read_text().split())
    wanted = CORE_WORDS if words == "core" else None
    splits = {"train": [], "val": [], "test": []}
    for wav in sorted(extract_dir.glob("*/*.wav")):
        word = wav.parent.name
        if word.startswith("_") or (wanted and word not in wanted):
            continue
        rel = f"{word}/{wav.name}"
        split = "test" if rel in test_list else "val" if rel in val_list else "train"
        splits[split].append({"path": str(wav), "duration": 1.0, "label": word,
                              "speaker": wav.name.split("_")[0]})
    # исходные файлы уже 16 кГц wav — копируем в out, чтобы данные лежали на Drive
    for rows in splits.values():
        for r in rows:
            dst = out / "commands" / Path(r["path"]).relative_to(extract_dir)
            if not dst.exists():
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(r["path"], dst)
            r["path"] = str(dst)
    write_splits(splits, out / "manifests", "commands")
 
 
# ---------- окна Dusha для VAE ----------
 
def prepare_dusha_windows(out: Path, files_per_class: int = 2000, windows_per_file: int = 3) -> None:
    """Режет записи Dusha на окна по 1 с — данные для VAE, латенты которого идут признаками в модуль 2.
    Метка окна = эмоция исходной записи (по ней раскрашивается латентное пространство)."""
    from ..utils import read_manifest
 
    for split in ("train", "val"):
        rows = read_manifest(out / "manifests" / f"emotion_{split}.jsonl")
        per_class: Counter = Counter()
        cap = files_per_class if split == "train" else max(1, files_per_class // 10)
        result = []
        for i, r in enumerate(rows):
            if per_class[r["label"]] >= cap:
                continue
            wav = load_audio(r["path"])
            starts = list(range(0, max(1, len(wav) - SAMPLE_RATE + 1), SAMPLE_RATE))[:windows_per_file]
            kept = 0
            for k, s in enumerate(starts):
                seg = wav[s:s + SAMPLE_RATE]
                if len(seg) < SAMPLE_RATE or np.sqrt(np.mean(seg ** 2)) < 1e-3:  # пропускаем тишину
                    continue
                path = out / "dusha_windows" / split / r["label"] / f"{i:07d}_{k}.flac"
                result.append({"path": str(path), "duration": save_flac(seg, path), "label": r["label"]})
                kept += 1
            per_class[r["label"]] += int(kept > 0)
        write_manifest(result, out / "manifests" / f"dusha_windows_{split}.jsonl")
        print(f"[dusha_windows] {split}: {len(result)} окон, файлов по классам {dict(per_class)}")
 
 
# ---------- свои файлы ----------
 
def prepare_folder(audio_dir: Path, out: Path, name: str, label_from_folder: bool = True) -> None:
    """Манифест для своих записей. Если файлы лежат в папках по классам (angry/x.wav),
    имя папки становится меткой. Подходит и для оригинального Dusha с частями crowd / podcast."""
    files = sorted(p for p in audio_dir.rglob("*") if p.suffix.lower() in {".wav", ".flac", ".mp3", ".ogg", ".m4a"})
    durations = convert_tree(files, audio_dir, out / name)
    rows = []
    for f in files:
        dst = out / name / f.relative_to(audio_dir).with_suffix(".flac")
        if dst in durations:
            row = {"path": str(dst), "duration": durations[dst]}
            if label_from_folder:
                row["label"] = f.parent.name
            rows.append(row)
    write_manifest(rows, out / "manifests" / f"{name}.jsonl")
    print(f"[folder] {name}.jsonl: {len(rows)} записей")
 
 
def main() -> None:
    p = argparse.ArgumentParser(description="Подготовка данных курсовой")
    sub = p.add_subparsers(dest="cmd", required=True)
    for cmd in ("golos", "musan", "dusha", "dusha_windows", "commands", "folder"):
        sp = sub.add_parser(cmd)
        sp.add_argument("--out", required=True, type=Path, help="куда класть данные (папка на Google Drive)")
        sp.add_argument("--work", type=Path, default=Path("/content/work"), help="временная папка для архивов")
        if cmd == "musan":
            sp.add_argument("--music-files", type=int, default=300)
        if cmd == "dusha":
            sp.add_argument("--per-class", type=int, default=6000)
            sp.add_argument("--test-per-class", type=int, default=1000)
        if cmd == "commands":
            sp.add_argument("--words", choices=["core", "all"], default="core")
        if cmd == "folder":
            sp.add_argument("--audio-dir", type=Path, required=True)
            sp.add_argument("--name", required=True)
            sp.add_argument("--no-labels", action="store_true")
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    if a.cmd == "golos":
        prepare_golos(a.out, a.work)
    elif a.cmd == "musan":
        prepare_musan(a.out, a.work, a.music_files)
    elif a.cmd == "dusha":
        prepare_dusha(a.out, a.per_class, a.test_per_class)
    elif a.cmd == "dusha_windows":
        prepare_dusha_windows(a.out)
    elif a.cmd == "commands":
        prepare_commands(a.out, a.work, a.words)
    elif a.cmd == "folder":
        prepare_folder(a.audio_dir, a.out, a.name, not a.no_labels)
 
 
if __name__ == "__main__":
    main()