"""多言語の会話ベンチマーク(FLEURS 合成2人会話 → アプリと同じパイプライン → 指標)。

FLEURS(google/fleurs, CC BY 4.0)は同じ文 id を各言語で読み上げた並行コーパス。
話者A(言語X)と話者B(言語Y)の発話を交互に並べ(早い交代0.2〜0.4秒・間0.6〜1.5秒・
同じ人の連続発話も混ぜる)、部屋の雑音と音量の揺れを足した会話 WAV と正解 JSON を作り、
アプリと同じ区切り(AutoGain → silero VAD → StreamSegmenter(cut_s=0.18) →
雑音ゲート/幻覚守卫 → Endpointer(rejoin))と リモートサーバーの耳・翻訳で通す。

指標(ペア別・言語別): 言語判定の正解率、ASR の CER(ja/zh/ko、句読点除去)/WER(en)、
zh 出力の繁体字混入数、翻訳 chrF(文字6-gram, beta=2、自前実装)、区切り(2人を跨ぐ=cross、
1ターンが複数に割れる=split、同じ人の2文が1単位=merged、取りこぼし=missing、
正解なしの単位=spurious)、話し終わり→訳してよい時刻/翻訳時間/字幕まで、
翻訳失敗(空・（翻訳できませんでした）)、素通し(訳の言語 == 原文の言語)。

実行(リポジトリ直下。FLEURS は信頼しないデータなので -I で読む):
  .venv/bin/python -I scripts/eval_conv.py build --fleurs DIR --pair ja-en --seed 1 --dur 180
  .venv/bin/python -I scripts/eval_conv.py run   --conv DIR/convs/ja-en_s1.json --out DIR/results
  .venv/bin/python -I scripts/eval_conv.py report DIR/results/*/summary.json
  .venv/bin/python -I scripts/eval_conv.py all   --fleurs DIR --pairs ja-en,ja-zh,ja-ko,en-zh --conds quiet,venue
条件(--cond): quiet = 2人・静かな部屋(-45dBFS の雑音) / venue = 騒がしい会場・複数人
  (1言語2人、話者の音量差±6dB、交代時の重なり0.2〜0.4秒、FLEURS 6〜10声のざわめきを SNR 5〜10dB)。
venue の追加指標: ゴースト単位(正解の声と重ならない単位 = ざわめきを拾った等)、
  ざわめき混入(同時刻の背景の文の言葉が聞き取りに入った単位)、同じ言語の別人を1単位にした数。
DIR/dl_<fleurs言語>/ に test.tsv と test.tar.gz(HF: data/<lang>/test.tsv, data/<lang>/audio/test.tar.gz)。
tar は展開せず、必要な wav だけをメモリで読んで DIR/clips/ に16kHz mono で書き直す。
リモートサーバーの耳/翻訳(SSH 転送 12346/12345)を使う。耳が落ちていたら scripts/remote_tunnel.sh。
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import math
import os
import random
import re
import sys
import tarfile
import time
import unicodedata
import wave
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402

SR = 16000
FLEURS = {"ja": "ja_jp", "en": "en_us", "zh": "cmn_hans_cn", "ko": "ko_kr"}
FAIL_TEXT = "（翻訳できませんでした）"

csv.field_size_limit(10_000_000)


# ====================================================================== データ

def load_tsv(fleurs_dir: Path, lang: str) -> list[dict]:
    """FLEURS の test.tsv: id, file_name, raw_transcription, transcription, chars, num_samples, gender。"""
    p = fleurs_dir / f"dl_{FLEURS[lang]}" / "test.tsv"
    rows = []
    with open(p, encoding="utf-8", newline="") as f:
        for r in csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE):
            if len(r) < 7 or not r[0].isdigit():
                continue
            fn = os.path.basename(r[1])
            if not re.fullmatch(r"[0-9A-Za-z_\-]+\.wav", fn):
                continue
            rows.append({"sid": int(r[0]), "file": fn, "text": r[2].strip(),
                         "n": int(r[5]) if r[5].isdigit() else 0, "gender": r[6].strip()})
    return rows


def wav_bytes_to_16k(b: bytes) -> np.ndarray:
    """RIFF/WAVE を自前で読む(FLEURS は float32 = format 3 で、標準の wave は読めない)。"""
    import struct
    if b[:4] != b"RIFF" or b[8:12] != b"WAVE":
        raise ValueError("WAV ではない")
    pos, fmt, raw = 12, None, None
    while pos + 8 <= len(b):
        cid, size = b[pos:pos + 4], struct.unpack("<I", b[pos + 4:pos + 8])[0]
        body = b[pos + 8:pos + 8 + size]
        if cid == b"fmt ":
            tag, ch, sr, _br, _ba, bits = struct.unpack("<HHIIHH", body[:16])
            if tag == 0xFFFE and len(body) >= 26:
                tag = struct.unpack("<H", body[24:26])[0]
            fmt = (tag, ch, sr, bits)
        elif cid == b"data":
            raw = body
        pos += 8 + size + (size & 1)
    if fmt is None or raw is None:
        raise ValueError("fmt/data が無い")
    tag, ch, sr, bits = fmt
    if tag == 1 and bits == 16:
        x = np.frombuffer(raw[:len(raw) // 2 * 2], dtype="<i2").astype(np.float32) / 32768.0
    elif tag == 1 and bits == 32:
        x = np.frombuffer(raw[:len(raw) // 4 * 4], dtype="<i4").astype(np.float32) / 2147483648.0
    elif tag == 3 and bits == 32:
        x = np.frombuffer(raw[:len(raw) // 4 * 4], dtype="<f4").astype(np.float32)
    else:
        raise ValueError(f"未対応の形式 tag={tag} bits={bits}")
    x = np.nan_to_num(x)
    if ch > 1:
        x = x.reshape(-1, ch).mean(axis=1)
    if sr != SR:
        x = resample(x, sr, SR)
    return x.astype(np.float32)


def resample(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    """FFT による帯域制限リサンプル(scipy 無し)。FLEURS は元々16kHzなので通常は通らない。"""
    n_out = int(round(len(x) * sr_out / sr_in))
    X = np.fft.rfft(x)
    keep = min(len(X), n_out // 2 + 1)
    Y = np.zeros(n_out // 2 + 1, dtype=complex)
    Y[:keep] = X[:keep]
    return np.fft.irfft(Y, n_out) * (n_out / len(x))


def fetch_clips(fleurs_dir: Path, lang: str, files: set[str]) -> dict[str, Path]:
    """必要な wav だけを tar から(展開せず)読み、clips/<lang>/ に16kHz mono で書く。"""
    out_dir = fleurs_dir / "clips" / lang
    out_dir.mkdir(parents=True, exist_ok=True)
    got = {f: out_dir / f for f in files if (out_dir / f).exists()}
    need = set(files) - set(got)
    if need:
        tp = fleurs_dir / f"dl_{FLEURS[lang]}" / "test.tar.gz"
        with tarfile.open(tp, "r:gz") as tf:
            for m in tf:
                if not m.isreg():
                    continue
                base = os.path.basename(m.name)
                if base not in need:
                    continue
                fobj = tf.extractfile(m)
                if fobj is None or m.size > 50_000_000:
                    continue
                x = wav_bytes_to_16k(fobj.read())
                write_wav(out_dir / base, x)
                got[base] = out_dir / base
                need.discard(base)
                if not need:
                    break
    if need:
        raise RuntimeError(f"{lang}: tar に無い wav {sorted(need)[:3]}…")
    return got


def read_wav(path) -> np.ndarray:
    with wave.open(str(path), "rb") as w:
        assert w.getframerate() == SR and w.getnchannels() == 1 and w.getsampwidth() == 2
        return np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(np.float32) / 32768.0


def write_wav(path, x: np.ndarray) -> None:
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes((np.clip(x, -1, 1) * 32767.0).astype("<i2").tobytes())


def trim_silence(x: np.ndarray, rel_db: float = -35.0, pad_s: float = 0.05) -> np.ndarray:
    """FLEURS の前後の無音(録音の頭・尻)を落とす。間の長さを正確に制御するため。"""
    fr = int(0.02 * SR)
    n = len(x) // fr
    if n < 3:
        return x
    r = np.sqrt(np.mean(np.square(x[:n * fr].reshape(n, fr), dtype=np.float64), axis=1))
    thr = max(r.max() * 10 ** (rel_db / 20), np.percentile(r, 10) * 3.0, 1e-4)
    on = np.where(r > thr)[0]
    if not len(on):
        return x
    a = max(0, on[0] * fr - int(pad_s * SR))
    b = min(len(x), (on[-1] + 1) * fr + int(pad_s * SR))
    return x[a:b]


def active_rms(x: np.ndarray) -> float:
    fr = int(0.02 * SR)
    n = len(x) // fr
    r = np.sqrt(np.mean(np.square(x[:n * fr].reshape(n, fr), dtype=np.float64), axis=1))
    act = r[r > r.max() * 0.1]
    return float(np.sqrt(np.mean(np.square(act)))) if len(act) else float(r.mean() + 1e-9)


def pink_noise(n: int, rng: np.random.Generator) -> np.ndarray:
    w = rng.standard_normal(n)
    X = np.fft.rfft(w)
    f = np.arange(len(X), dtype=np.float64)
    f[0] = 1.0
    y = np.fft.irfft(X / np.sqrt(f), n)
    return (y / (np.sqrt(np.mean(y ** 2)) + 1e-12)).astype(np.float32)


def build(fleurs_dir: Path, pair: str, seed: int, dur_s: float, out_dir: Path,
          min_raw_s: float = 2.5, max_raw_s: float = 14.0, p_same: float = 0.15) -> Path:
    X, Y = pair.split("-")
    rng = random.Random(seed)
    nrng = np.random.default_rng(seed)
    rows = {L: load_tsv(fleurs_dir, L) for L in (X, Y)}
    by_sid: dict[str, dict[int, list[dict]]] = {}
    text_of: dict[str, dict[int, str]] = {}
    for L, rs in rows.items():
        by_sid[L], text_of[L] = {}, {}
        for r in rs:
            text_of[L].setdefault(r["sid"], r["text"])
            if min_raw_s * SR <= r["n"] <= max_raw_s * SR:
                by_sid[L].setdefault(r["sid"], []).append(r)
    common = sorted(set(text_of[X]) & set(text_of[Y]))
    usable = {L: [s for s in common if s in by_sid[L]] for L in (X, Y)}
    print(f"[build] {pair}: 共通の文id {len(common)} / 使える {X}:{len(usable[X])} {Y}:{len(usable[Y])}")

    # 話者ごとに性別を決めて寄せる(FLEURS は話者idが無い。同じ人っぽさの近似)
    gender = {X: rng.choice(["MALE", "FEMALE"]), Y: rng.choice(["MALE", "FEMALE"])}
    used: set[int] = set()
    plan: list[dict] = []
    spk = 0  # 0=A(X) 1=B(Y)
    est = 1.0
    while est < dur_s:
        L = (X, Y)[spk]
        cand = [s for s in usable[L] if s not in used]
        if not cand:
            break
        sid = rng.choice(cand)
        used.add(sid)
        opts = by_sid[L][sid]
        pref = [r for r in opts if r["gender"] == gender[L]] or opts
        r = rng.choice(pref)
        plan.append({"spk": "AB"[spk], "lang": L, "sid": sid, "file": r["file"], "text": r["text"],
                     "gender": r["gender"],
                     "ref": {X: text_of[X][sid], Y: text_of[Y][sid]}})
        est += r["n"] / SR * 0.85 + 0.8
        if not (plan and len(plan) >= 2 and plan[-2]["spk"] == plan[-1]["spk"]) \
                and rng.random() < p_same:
            continue  # 同じ人がもう1文(連続は2文まで)
        spk = 1 - spk

    clips = {}
    for L in (X, Y):
        clips.update({(L, f): p for f, p in fetch_clips(
            fleurs_dir, L, {t["file"] for t in plan if t["lang"] == L}).items()})

    base_db = {"A": -24.0 + rng.uniform(-2, 2), "B": -28.0 + rng.uniform(-2, 2)}
    t = 1.0
    pieces = []
    for i, tr in enumerate(plan):
        x = trim_silence(read_wav(clips[(tr["lang"], tr["file"])]))
        g_db = base_db[tr["spk"]] + rng.uniform(-3.0, 3.0)
        x = x * (10 ** (g_db / 20) / max(active_rms(x), 1e-6))
        if i > 0:
            if tr["spk"] == plan[i - 1]["spk"]:
                gap, kind = rng.uniform(0.4, 1.2), "same"
            else:
                u = rng.random()
                if u < 0.45:
                    gap, kind = rng.uniform(0.2, 0.4), "quick"
                elif u < 0.6:
                    gap, kind = rng.uniform(0.4, 0.6), "mid"
                else:
                    gap, kind = rng.uniform(0.6, 1.5), "pause"
            t += gap
        else:
            gap, kind = 0.0, "first"
        tr.update({"idx": i, "start": round(t, 3), "end": round(t + len(x) / SR, 3),
                   "gain_db": round(g_db, 1), "gap_before": round(gap, 3), "gap_kind": kind})
        pieces.append((int(round(t * SR)), x))
        t += len(x) / SR
    total = int((t + 2.0) * SR)
    mix = np.zeros(total, dtype=np.float32)
    for a, x in pieces:
        mix[a:a + len(x)] += x
    noise_db = -45.0 + rng.uniform(-2, 2)
    noise = 0.8 * pink_noise(total, nrng) + 0.2 * nrng.standard_normal(total).astype(np.float32)
    noise *= 10 ** (noise_db / 20) / (np.sqrt(np.mean(noise ** 2)) + 1e-12)
    mix += noise
    peak = float(np.max(np.abs(mix)))
    if peak > 0.98:
        mix *= 0.98 / peak

    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{pair}_s{seed}"
    write_wav(out_dir / f"{stem}.wav", mix)
    meta = {"pair": pair, "seed": seed, "wav": f"{stem}.wav", "duration": round(total / SR, 2),
            "noise_dbfs": round(noise_db, 1), "speaker_level_dbfs": base_db,
            "gender": gender, "source": "google/fleurs test (CC BY 4.0)", "turns": plan}
    jp = out_dir / f"{stem}.json"
    jp.write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    kinds = {}
    for tr in plan:
        kinds[tr["gap_kind"]] = kinds.get(tr["gap_kind"], 0) + 1
    print(f"[build] {jp.name}: {total / SR:.1f}s / {len(plan)}ターン / 間 {kinds}")
    return jp


def lowpass(x: np.ndarray, cutoff: float = 3000.0) -> np.ndarray:
    """遠くの人の声(高域が減る)の近似。FFT で cutoff 以上を緩やかに落とす。"""
    X = np.fft.rfft(x)
    f = np.fft.rfftfreq(len(x), 1.0 / SR)
    X *= 1.0 / np.sqrt(1.0 + (f / cutoff) ** 4)
    return np.fft.irfft(X, len(x)).astype(np.float32)


def build_venue(fleurs_dir: Path, pair: str, seed: int, dur_s: float, out_dir: Path,
                n_speakers: int = 4, min_raw_s: float = 2.5, max_raw_s: float = 14.0,
                p_same: float = 0.15, p_overlap: float = 0.25) -> Path:
    """騒がしい会場・複数人: 1言語2人(計4人。--speakers 3 なら X 2人 + Y 1人)、
    話者ごとの音量差 ±6dB(マイクに近い/遠い)、話者交代の時々の重なり(0.2〜0.4秒)、
    背景のざわめき(正解に使わない FLEURS 発話 6〜10本を4言語から同時に流す。
    前景の声に対して SNR 5〜10dB、遠くの声として高域を落とす)+ 部屋の雑音。"""
    X, Y = pair.split("-")
    rng = random.Random(seed * 1000 + 7)
    nrng = np.random.default_rng(seed * 1000 + 7)
    all_l = ["ja", "en", "zh", "ko"]
    rows = {L: load_tsv(fleurs_dir, L) for L in all_l}
    by_sid: dict[str, dict[int, list[dict]]] = {}
    text_of: dict[str, dict[int, str]] = {}
    for L, rs in rows.items():
        by_sid[L], text_of[L] = {}, {}
        for r in rs:
            text_of[L].setdefault(r["sid"], r["text"])
            if min_raw_s * SR <= r["n"] <= max_raw_s * SR:
                by_sid[L].setdefault(r["sid"], []).append(r)
    common = sorted(set(text_of[X]) & set(text_of[Y]))
    usable = {L: [s for s in common if s in by_sid[L]] for L in (X, Y)}

    # 話者: 言語ごとに男女を変えて2人(FLEURS は話者id無し → 性別と音量で別人を近似)
    spk_list: list[dict] = []
    for li, L in enumerate((X, Y)):
        n = 2 if (li == 0 or n_speakers >= 4) else 1
        gs = rng.sample(["MALE", "FEMALE"], 2)
        for k in range(n):
            spk_list.append({"name": f"{'AB'[li]}{k + 1}", "lang": L, "gender": gs[k % 2],
                             "level_db": round(-26.0 + rng.uniform(-6.0, 6.0), 1)})
    used: set[int] = set()
    used_files: set[tuple[str, str]] = set()
    plan: list[dict] = []
    cur = rng.choice(spk_list)
    est = 1.0
    while est < dur_s:
        L = cur["lang"]
        cand = [s for s in usable[L] if s not in used]
        if not cand:
            break
        sid = rng.choice(cand)
        used.add(sid)
        opts = by_sid[L][sid]
        pref = [r for r in opts if r["gender"] == cur["gender"]] or opts
        r = rng.choice(pref)
        used_files.add((L, r["file"]))
        plan.append({"spk": cur["name"], "lang": L, "sid": sid, "file": r["file"],
                     "text": r["text"], "gender": r["gender"],
                     "ref": {X: text_of[X][sid], Y: text_of[Y][sid]}})
        est += r["n"] / SR * 0.85 + 0.6
        if not (len(plan) >= 2 and plan[-2]["spk"] == plan[-1]["spk"]) and rng.random() < p_same:
            continue
        others = [s for s in spk_list if s["name"] != cur["name"]]
        other_lang = [s for s in others if s["lang"] != cur["lang"]]
        same_lang = [s for s in others if s["lang"] == cur["lang"]]
        cur = rng.choice(other_lang if (not same_lang or rng.random() < 0.7) else same_lang)

    # 背景のざわめき: 正解に使っていない文id・ファイルから、4言語を順に回して 6〜10 本の声
    n_streams = rng.randint(6, 10)
    gold_sids = {t["sid"] for t in plan}
    streams = []
    for k in range(n_streams):
        L = all_l[k % 4]
        pool = [r for s, rs in by_sid[L].items() if s not in gold_sids for r in rs
                if (L, r["file"]) not in used_files]
        picks = rng.sample(pool, 3)
        for p in picks:
            used_files.add((L, p["file"]))
        streams.append({"lang": L, "utts": picks})

    clips = {}
    for L in all_l:
        need = {f for (l2, f) in used_files if l2 == L}
        if need:
            clips.update({(L, f): p for f, p in fetch_clips(fleurs_dir, L, need).items()})

    spk_of = {s["name"]: s for s in spk_list}
    t = 1.0
    pieces = []
    for i, tr in enumerate(plan):
        x = trim_silence(read_wav(clips[(tr["lang"], tr["file"])]))
        g_db = spk_of[tr["spk"]]["level_db"] + rng.uniform(-2.0, 2.0)
        x = x * (10 ** (g_db / 20) / max(active_rms(x), 1e-6))
        if i > 0:
            prev_len = plan[i - 1]["end"] - plan[i - 1]["start"]
            if tr["spk"] == plan[i - 1]["spk"]:
                gap, kind = rng.uniform(0.4, 1.2), "same"
            else:
                u = rng.random()
                if u < p_overlap and prev_len > 1.5:
                    gap, kind = -rng.uniform(0.2, 0.4), "overlap"
                elif u < p_overlap + 0.35:
                    gap, kind = rng.uniform(0.2, 0.4), "quick"
                elif u < p_overlap + 0.45:
                    gap, kind = rng.uniform(0.4, 0.6), "mid"
                else:
                    gap, kind = rng.uniform(0.6, 1.5), "pause"
            t += gap
        else:
            gap, kind = 0.0, "first"
        tr.update({"idx": i, "start": round(t, 3), "end": round(t + len(x) / SR, 3),
                   "gain_db": round(g_db, 1), "gap_before": round(gap, 3), "gap_kind": kind})
        pieces.append((int(round(t * SR)), x))
        t += len(x) / SR
    total = int((t + 2.0) * SR)
    fg = np.zeros(total, dtype=np.float32)
    for a, x in pieces:
        fg[a:a + len(x)] += x

    # ざわめき: 各声は自分の3文を間0.2〜0.8秒で繰り返し、開始位置はずらす
    babble = np.zeros(total, dtype=np.float32)
    bab_segs = []
    for k, st in enumerate(streams):
        lvl = 10 ** (rng.uniform(-3.0, 3.0) / 20)
        pos = -int(rng.uniform(0, 8.0) * SR)
        j = rng.randrange(3)
        while pos < total:
            u = st["utts"][j % 3]
            x = trim_silence(read_wav(clips[(st["lang"], u["file"])]))
            x = lowpass(x) * (lvl * 0.05 / max(active_rms(x), 1e-6))
            a, b = max(0, pos), min(total, pos + len(x))
            if b > a:
                babble[a:b] += x[a - pos:b - pos]
                bab_segs.append({"stream": k, "lang": st["lang"], "sid": u["sid"], "text": u["text"],
                                 "start": round(a / SR, 3), "end": round(b / SR, 3)})
            pos += len(x) + int(rng.uniform(0.2, 0.8) * SR)
            j += 1
    fg_db = 10 * math.log10(np.mean([10 ** (tr["gain_db"] / 10) for tr in plan]))
    snr = rng.uniform(5.0, 10.0)
    b_rms = float(np.sqrt(np.mean(babble ** 2))) + 1e-12
    babble *= 10 ** ((fg_db - snr) / 20) / b_rms
    noise_db = -45.0 + rng.uniform(-2, 2)
    noise = 0.8 * pink_noise(total, nrng) + 0.2 * nrng.standard_normal(total).astype(np.float32)
    noise *= 10 ** (noise_db / 20) / (np.sqrt(np.mean(noise ** 2)) + 1e-12)
    mix = fg + babble + noise
    peak = float(np.max(np.abs(mix)))
    if peak > 0.98:
        mix *= 0.98 / peak

    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{pair}_venue_s{seed}"
    write_wav(out_dir / f"{stem}.wav", mix)
    meta = {"pair": pair, "seed": seed, "cond": "venue", "wav": f"{stem}.wav",
            "duration": round(total / SR, 2), "noise_dbfs": round(noise_db, 1),
            "speakers": spk_list, "babble_streams": n_streams,
            "babble_langs": [s["lang"] for s in streams],
            "snr_db": round(snr, 1), "fg_level_dbfs": round(fg_db, 1),
            "source": "google/fleurs test (CC BY 4.0)", "turns": plan, "babble": bab_segs}
    jp = out_dir / f"{stem}.json"
    jp.write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    kinds: dict[str, int] = {}
    for tr in plan:
        kinds[tr["gap_kind"]] = kinds.get(tr["gap_kind"], 0) + 1
    print(f"[build] {jp.name}: {total / SR:.1f}s / {len(plan)}ターン / 話者 "
          f"{[(s['name'], s['level_db']) for s in spk_list]} / ざわめき{n_streams}声 SNR {snr:.1f}dB / 間 {kinds}")
    return jp


# ====================================================================== パイプライン

def git_state() -> str:
    """どの版のアプリで測ったか(HEAD と、src/server の未コミット変更の有無)。"""
    import subprocess
    root = Path(__file__).resolve().parents[1]
    try:
        head = subprocess.run(["git", "-C", str(root), "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, timeout=5).stdout.strip()
        dirty = subprocess.run(["git", "-C", str(root), "status", "--porcelain", "src", "server"],
                               capture_output=True, text=True, timeout=5).stdout.split()
        return head + ("+dirty:" + ",".join(x for x in dirty if "/" in x) if dirty else "")
    except Exception:  # noqa: BLE001
        return "?"


class _SpecSim:
    """アプリの Speculator(speculate.py)の update/take/_show の規則だけを音声の時計で再現する。

    update(pending, emitted): emitted の先読みは確定(reserved)、それ以外で pending でない
    未確定の先読みは取り消し。pending が新しければ先読みを始める。大字幕に予想訳が出るのは
    その先読みが「今待っている断片」(_latest)である間だけ → superseded に替わった時刻を残す。"""

    def __init__(self):
        self.jobs: dict = {}
        self.log: list = []
        self.latest = None

    def update(self, pending, emitted, now: float, rng=None, src: str = "pending") -> None:
        em = {tuple(e) for e in emitted}
        for k, j in list(self.jobs.items()):
            if k in em:
                j["reserved"] = True
            elif not j["reserved"] and k != pending:
                j["cancelled"] = round(now, 3)
                del self.jobs[k]
        if pending is not None and pending[0] and pending not in self.jobs:
            j = {"text": pending[0], "lang": pending[1], "t0": round(now, 3), "src": src,
                 "lo": rng[0] if rng else None, "hi": rng[1] if rng else None,
                 "reserved": False, "superseded": None, "cancelled": None, "unit": None}
            self.jobs[pending] = j
            self.log.append(j)
        if pending != self.latest:
            for j in self.log:
                if (j["text"], j["lang"]) == self.latest and j["superseded"] is None:
                    j["superseded"] = round(now, 3)
            self.latest = pending

    def take(self, key, unit_idx: int):
        j = self.jobs.pop(tuple(key), None)
        if j is not None:
            j["unit"] = unit_idx
        if self.latest == tuple(key):
            self.latest = None
        return j


class _NoLocal:
    """リモートサーバーの耳が落ちたとき Mac の whisper に落とさない(重い計算は remote だけ)。"""
    fails = 0

    def transcribe(self, samples):
        _NoLocal.fails += 1
        return "", "", 0.0


def run_pipeline(conv_json: Path, out_root: Path, vad_threshold: float | None = None,
                 topic: str = "", stt_beam: int | None = None, mac: bool = False,
                 mac_ear: bool = False, hedge: float = 0.0,
                 mac_model: str | None = None, preview: float = 0.0) -> Path:
    git_at_start = git_state()   # src を読み込む「前」の版(終わりの版と違えば測定中に変わった)
    from tsuyaku.audio import AutoGain, StreamSegmenter, rms
    from tsuyaku.config import engine_candidates, load_settings
    from tsuyaku.guards import (echoes_prompt, hallucinated_out_of_pair, looks_truncated,
                                stt_prompt)
    from tsuyaku.langdetect import norm
    from tsuyaku.segmenter import Endpointer
    from tsuyaku.stt import RemoteSTT
    from tsuyaku.translate import (build_messages, out_lang, pick_model, translate_stream, warmup)
    try:
        from tsuyaku.guards import wrong_language
    except ImportError:  # 古い版のアプリ
        wrong_language = None
    import sherpa_onnx

    meta = json.loads(conv_json.read_text(encoding="utf-8"))
    pair = meta["pair"]
    pair_t = tuple(pair.split("-"))
    audio = read_wav(conv_json.parent / meta["wav"])
    s = load_settings()
    vth = s.vad_threshold if vad_threshold is None else vad_threshold

    _NoLocal.fails = 0
    if mac or mac_ear:
        # このMacだけ(11/06)。mac_ear = 耳だけこのMac(SenseVoice)、翻訳は remote: 耳 = SenseVoice、翻訳 = このMacの LM Studio
        from tsuyaku.stt import SenseVoiceSTT
        stt = RemoteSTT(None, SenseVoiceSTT(s.sense_voice_dir))
        stt.local.langs = pair_t
    elif hedge > 0:
        # アプリと同じ: リモートサーバーを hedge 秒待ち、来なければこのMacの SenseVoice(11/06)
        from tsuyaku.stt import SenseVoiceSTT
        stt = RemoteSTT(s.stt_url, SenseVoiceSTT(s.sense_voice_dir), hedge_s=hedge)
        stt.local.langs = pair_t
        if preview > 0:
            # アプリ(1d00fb3〜)と同じ: リモートサーバーが preview 秒で返らなければ、このMacの耳の
            # 結果で予想訳を先に出す(on_preview)。確定は リモートサーバーのまま
            stt.preview_s = preview
    else:
        stt = RemoteSTT(s.stt_url, _NoLocal())
    stt.langs = pair_t
    stt.prompt = stt_prompt(topic)
    if stt_beam:
        # 実験: リモートサーバーの耳に ?beam=N を付ける(サーバ 0aad8ff〜。アプリは既定 beam=1)
        _post0 = stt._post
        stt._post = lambda url, _p=_post0, **kw: _p(
            url + ("&" if "?" in url else "?") + f"beam={int(stt_beam)}", **kw)
    # アプリの SpeechCapture(fine=True = リモートサーバーの耳)と同じ区切り
    conf = sherpa_onnx.VadModelConfig()
    conf.silero_vad.model = s.silero_vad
    conf.silero_vad.threshold = vth
    conf.silero_vad.min_silence_duration = 0.2
    conf.silero_vad.min_speech_duration = s.min_speech_s
    conf.silero_vad.max_speech_duration = max(30.0, s.max_speech_s)
    conf.sample_rate = SR
    vad = sherpa_onnx.VoiceActivityDetector(conf, buffer_size_in_seconds=60)
    agc = AutoGain(SR)
    sg = StreamSegmenter(SR, min_speech_s=s.min_speech_s, min_silence_s=0.2,
                         cut_s=0.18, max_s=s.max_speech_s)
    import inspect
    push_takes_prob = "prob" in inspect.signature(Endpointer.push).parameters
    # 予想訳(preview, 11/06): on_preview が呼ばれた壁時計と文字を記録し、あとで音声の時計に直す
    pv_wall: list = []
    takes_pv = "on_preview" in inspect.signature(stt.transcribe).parameters

    def _on_preview(text, lang):
        pv_wall.append((time.time(), text, lang))

    def hear(a, with_pv):
        if preview > 0 and with_pv and takes_pv:
            return stt.transcribe(a, on_preview=_on_preview)
        return stt.transcribe(a)

    # アプリと同じく、繋ぎ直しの聞き直しでも予想訳を出す(rejoin=lambda a: hear(a, preview))
    ep = Endpointer(rejoin=lambda a: hear(a, True),
                    hold_open_s=s.hold_open_s if s.hold_open_s > 0 else 0.0)

    # ---- 音声の時計で再生。STT ワーカーの所要時間(実測)ぶん仮想時計を進める ----
    events: list[dict] = []   # 区間ごとの記録
    units: list[dict] = []
    queue: list[tuple] = []
    busy = 0.0
    last_tick = 0.0
    stt_calls = [0, 0.0]
    stt_retries = [0]

    def timed_stt(seg, with_pv=False):
        # リモートサーバーの耳が一時的に落ちた(再起動など)ときは、Mac に落とさず待って聞き直す。
        # 落ちていた間の待ち時間は遅延に入れない(評価の中断であってアプリの遅延ではない)。
        for k in range(30):
            before = _NoLocal.fails
            t0 = time.time()
            r = hear(seg, with_pv)
            dt = time.time() - t0
            if _NoLocal.fails == before:
                break
            stt_retries[0] += 1
            stt._down_at = None
            if k in (0, 5, 15):
                os.system(f"bash {Path(__file__).resolve().parent / 'remote_tunnel.sh'} >/dev/null 2>&1")
            time.sleep(2.0)
        stt_calls[0] += 1
        stt_calls[1] += dt
        return r, dt, t0

    # 先読み翻訳(11/06)をアプリの Speculator.update/take と同じ規則で音声の時計の上で再現する
    spec = _SpecSim()
    n_preview = [0, 0]   # [このMacの耳が予想を返した回数, そのうち予想訳を始めた回数]

    def pend_info():
        key = ep.pending() if hasattr(ep, "pending") else None
        p = getattr(ep, "_p", None)
        return key, ((p.lo, p.hi) if p is not None else None)

    def apply_previews(base_v: float, base_w: float, rng) -> None:
        """記録した on_preview を音声の時計に直して先読みに入れる(アプリの preview() と同じ条件)。"""
        for w, text, lang in pv_wall:
            n_preview[0] += 1
            if lang in pair_t and len(norm(text)) >= 2:
                n_preview[1] += 1
                spec.update((text, lang), [], base_v + (w - base_w), rng, "preview")
        pv_wall.clear()

    def add_units(us, ready_fn, now):
        key, rng = pend_info()
        spec.update(key, [(u.text, u.lang) for u in us], now, rng)
        for u in us:
            j = spec.take((u.text, u.lang), len(units))
            units.append({"text": u.text, "lang": u.lang, "lo": u.lo, "hi": u.hi,
                          "speech_lo": u.speech_lo, "parts": u.parts,
                          "ready": ready_fn(u),
                          "spec_t": j["t0"] if j else None, "spec_src": j["src"] if j else None})

    def process_queue(t_now: float) -> None:
        nonlocal busy
        if not queue or busy > t_now:
            return
        pending = list(queue)
        queue.clear()
        start = max(busy, pending[0][0])
        dropped = len(pending) >= 3   # アプリの pick_segments: 3個以上詰まったら最新1個だけ
        if dropped:
            for it in pending[:-1]:
                events.append({"lo": it[2], "hi": it[3], "skip": "backlog"})
            pending = pending[-1:]
            ep.flush()
        for t_det, seg, lo, hi, pre in pending:
            start = max(start, t_det)
            gate = min(s.noise_rms_floor, max(2.0 * sg.noise, 0.0015))
            ev = {"lo": round(lo, 3), "hi": round(hi, 3), "pre": round(pre, 3),
                  "t_det": round(t_det, 3)}
            events.append(ev)
            if rms(seg) < gate:
                ev["skip"] = "rms"
                continue
            # アプリ: 続き待ちの断片が無いときだけ予想訳を出す(あれば繋ぎ直しの方で出す)
            pend0 = ep.pending() if hasattr(ep, "pending") else None
            (text, lang, prob), dt, w0 = timed_stt(seg, pend0 is None)
            apply_previews(start, w0, (lo, hi))
            now = start + dt
            ev.update({"text": text, "lang": lang, "prob": round(float(prob), 3),
                       "stt_s": round(dt, 3), "where": stt.last_where})
            start = now
            if not text or not norm(text) or not lang:
                ev["skip"] = "empty"
                continue
            if echoes_prompt(text, stt.prompt):
                ev["skip"] = "echo_prompt"
                continue
            if hallucinated_out_of_pair(pair_t, lang, text, prob=prob):
                ev["skip"] = "halluc"
                continue
            _k0, rng0 = pend_info()
            t1 = time.time()
            kw = {"prob": prob} if push_takes_prob else {}   # アプリと同じく prob を渡す(対応版のみ)
            us = ep.push(seg, lo, hi, text, lang, now=now, pre=pre, **kw)
            dt2 = time.time() - t1
            ev["push_s"] = round(dt2, 3)
            apply_previews(now, t1, (rng0[0] if rng0 else lo, hi))   # 繋ぎ直しの聞き直し中の予想
            add_units(us, lambda u, _n=now, _d=dt2: (u.t_ready if u.t_ready < _n else _n + _d),
                      now + dt2)
            start = now + dt2
        busy = start

    chunk = 512
    for i in range(0, len(audio) + SR, chunk):
        ch = audio[i:i + chunk] if i < len(audio) else np.zeros(chunk, np.float32)
        t = (i + chunk) / SR
        g = agc(ch)
        vad.accept_waveform(g)
        while not vad.empty():
            vad.pop()
        for seg, lo, hi, pre in sg.feed(g, vad.is_speech_detected()):
            queue.append((t, seg, lo, hi, pre))
        process_queue(t)
        if busy <= t and not queue and t - last_tick >= 0.1:
            last_tick = t
            add_units(ep.tick(t, speaking_since=sg.speaking_since()), lambda u: u.t_ready, t)
    t_end = len(audio) / SR + 1.0
    while queue:
        process_queue(max(busy, queue[0][0]))
    add_units(ep.flush(), lambda u: max(u.t_ready, busy), busy)
    print(f"[run] {pair}: 区間 {len(events)} / 単位 {len(units)} / STT {stt_calls[0]}回 "
          f"平均{stt_calls[1] / max(1, stt_calls[0]):.2f}s / Mac落ち {_NoLocal.fails}")

    # ---- 翻訳: アプリの _utterance と同じ(retry・破綻時は空履歴で再試行・失敗表示) ----
    _lab, base, pref = engine_candidates("mac" if mac else "remote", s.host, None, mac_model)[0]
    model = pick_model(base, pref)
    warmup(base, model)
    history: list[dict] = []
    for u in units:
        msgs = build_messages(pair_t, topic, history, u["text"], u["lang"])
        first = [0.0]

        def on_delta(_d, _f=first):
            if not _f[0]:
                _f[0] = time.time()
        t0 = time.time()
        trans, tries, err = "", 0, ""
        for k in range(2):           # translate_with_retry と同じ: 空/例外なら0.4秒後に1回
            tries += 1
            try:
                trans = translate_stream(base, model, msgs, on_delta)
            except Exception as e:  # noqa: BLE001
                err = f"{type(e).__name__}: {str(e)[:80]}"
                trans = ""
            if trans:
                break
            if k == 0:
                time.sleep(0.4)
        lang_retry = False
        tgt = out_lang(pair_t, u["lang"])
        if trans and wrong_language is not None and wrong_language(u["text"], trans, tgt):
            # アプリ(6c5db02〜)と同じ: 訳の言語違い/写しは空履歴で1回やり直す
            lang_retry = True
            try:
                retry = translate_stream(base, model,
                                         build_messages(pair_t, topic, [], u["text"], u["lang"]))
            except Exception:  # noqa: BLE001
                retry = ""
            if retry and not wrong_language(u["text"], retry, tgt):
                trans = retry
        trunc_retry = False
        if trans and looks_truncated(u["text"], trans):
            trunc_retry = True
            try:
                retry = translate_stream(base, model,
                                         build_messages(pair_t, topic, [], u["text"], u["lang"]))
            except Exception:  # noqa: BLE001
                retry = ""
            if retry and not looks_truncated(u["text"], retry):
                trans = retry
        t_tr = time.time() - t0
        failed = (not trans) or looks_truncated(u["text"], trans)
        u.update({"trans": FAIL_TEXT if failed else trans, "fail": failed, "tr_s": round(t_tr, 3),
                  "ttft_s": round(first[0] - t0, 3) if first[0] else None, "tries": tries,
                  "trunc_retry": trunc_retry, "lang_retry": lang_retry, "tr_err": err, "model": model})
        if not failed:
            history.append({"lang": u["lang"], "orig": u["text"], "trans": trans})
    # 翻訳ワーカーは1本: 前の訳が終わるまで次は始まらない
    tb = 0.0
    tbs = 0.0
    for u in units:
        st = max(u["ready"], tb)
        u["tr_start"] = st
        u["tr_end"] = st + u["tr_s"]
        tb = u["tr_end"]
        # 先読みあり(アプリ 11/06〜): 待ちに入った時点の文字のまま確定したら、訳は
        # 待ち始めから別の流れで走っている。確定訳が出るのは「確定」「訳の完了」
        # 「前の文の表示」の遅い方。予想訳(仮)が見えるのは待ち始め+初字。
        sts = max(u["ready"], tbs)
        if u.get("spec_t") is not None:
            u["tr_end_spec"] = max(sts, u["spec_t"] + u["tr_s"])
            u["first_spec"] = u["spec_t"] + (u["ttft_s"] or u["tr_s"])
        else:
            u["tr_end_spec"] = sts + u["tr_s"]
            u["first_spec"] = sts + (u["ttft_s"] or u["tr_s"])
        tbs = u["tr_end_spec"]

    # 先読み・予想訳の「見えた時刻」: 外れた先読みも、次の断片に替わる前に初字が届けば
    # 大字幕に一度は出ている(Speculator._show は _latest の間だけ出す)。外れた分は実際に
    # 訳して初字までの時間を測る(当たった分は確定訳と同じ依頼なのでその初字を使う)。
    for j in spec.log:
        if j.get("unit") is not None and units[j["unit"]].get("ttft_s") is not None:
            uu = units[j["unit"]]
            j["ttft"] = uu["ttft_s"]
            j["result"] = uu["trans"]
            continue
        hist = [{"lang": uu["lang"], "orig": uu["text"], "trans": uu["trans"]}
                for uu in units if uu["ready"] < j["t0"] and not uu["fail"]]
        first = [0.0]

        def on_d(_d, _f=first):
            if not _f[0]:
                _f[0] = time.time()
        t0 = time.time()
        try:
            j["result"] = translate_stream(base, model,
                                           build_messages(pair_t, topic, hist, j["text"], j["lang"]),
                                           on_d)
        except Exception as e:  # noqa: BLE001
            j["result"] = ""
            j["err"] = f"{type(e).__name__}"
        j["ttft"] = round(first[0] - t0, 3) if first[0] else None
    for j in spec.log:
        sh = j["t0"] + j["ttft"] if j.get("ttft") is not None else None
        j["visible"] = sh is not None and (j["superseded"] is None or sh < j["superseded"])
        j["shown_t"] = sh if j["visible"] else None
        # どの単位の音声か(先読みの音声範囲が単位の範囲に入る)
        j["in_unit"] = None
        if j["lo"] is not None:
            for k, uu in enumerate(units):
                if uu["lo"] - 0.05 <= j["lo"] and j["hi"] <= uu["hi"] + 0.05:
                    j["in_unit"] = k
                    break
    for k, u in enumerate(units):
        cands = [(u["first_spec"], u.get("spec_src") or "final")]
        cands += [(j["shown_t"], j["src"] + "_guess") for j in spec.log
                  if j["visible"] and j["in_unit"] == k]
        u["first_any"], u["first_any_src"] = min(cands)

    out = out_root / conv_json.stem
    out.mkdir(parents=True, exist_ok=True)
    run_meta = {"conv": str(conv_json), "pair": pair, "model": model,
                "stt_url": "mac:sensevoice" if (mac or mac_ear) else s.stt_url,
                "vad_threshold": vth, "hold_open_s": s.hold_open_s, "topic": topic,
                "stt_beam": stt_beam or 1,
                "mac_fallbacks": _NoLocal.fails - stt_retries[0], "stt_retries": stt_retries[0],
                "git": git_at_start, "git_end": git_state(), "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                "hedge_s": hedge, "preview_s": preview,
                "stt_previewed": getattr(stt, "previewed", 0), "stt_hedged": getattr(stt, "hedged", 0),
                "preview_returned": n_preview[0], "preview_started": n_preview[1],
                "stt_calls": stt_calls[0]}
    (out / "run.json").write_text(json.dumps({"meta": run_meta, "events": events, "units": units,
                                              "spec_jobs": spec.log},
                                             ensure_ascii=False, indent=1), encoding="utf-8")
    return evaluate(conv_json, out)


# ====================================================================== 指標

_TRAD_ONLY = set(
    "們這個來時會說國對學過還開關頭點發經問電話門長車東書見聽後機現實為與從讓動體當無產進樣應"
    "義麼沒錢歡氣記處覺種邊認將萬變業條總區議裡裏辦愛兒幾歲號視聲讀寫醫藥場題師親陽陰嗎"
    "們間難買賣錯雖係聯絡員單價總結號響聲覽廣歷區際隨隊險陸陳張馬鳥魚黃齊龍龜傳們憶戰鐵銀"
    "鐘錄鍵門閉閱際雜雞離雲電靜韓頁項順須預領頭顏願類風飛飯飲館驗髮鬥鬧鬱魯鮮麗麥"
    "團圖圓園聖報場塊壞壓夠夢奪奮媽嬰學寧寶實審寫導將專尋對層屬岡島嶺崗帶幫廠廢廣"
    "強彈歸復徑從徵態慣慮慶憲應懷戲戶擊擇擔擁據擴擺擾攝數斷時晉書會東條來楊極榮樂標樣"
    "橋檢權歐歲歷殺殼氣漢沒淚淨涼減測濟濕滿漁潔潤澤灣燈營爭爺狀獨獲獸現環產畫異當療發"
    "盡盤睜確礦禮禪種稱穩窮競筆節範篤簡籃類紀約紅級紙純紛細終組結絕給統絲經綠維網緊線練"
    "縣總績織繼續罰羅羥義習聞聯聲職聽肅腦腳臉臨興舉舊艙艦藝藥蘇蘭處號蟲術衛衝補裝製複覺"
    "觀規視親覽觸計訂認討訓記訪設許評詞試詩話該詳誌語誤說請論諸謂講謝證識譜議護讀變讓豐"
    "貝負財貢貨質販貧購貴貸費賀資賓賞賢賣賤賦質賴趙趕跡軍軌較載輕輛輸辦農這進遠運過違適遲"
    "選遺邊鄉鄭醫釋針鈴鉛銅銷鋼錢鍋鎮長閃閉開閒間閣隊陽階際隨險隻雙雜難電靈韻響頁頂項順"
    "頒預領頻題額顏顯風颱飄飛飯飲飼飽養餐餘館馬駐騎騙驗驚體髮鬆魚鳥鳴鴨鴻鵝鷹鹽麥麵黨齒")


def traditional_chars(text: str) -> list[str]:
    return sorted({c for c in text if c in _TRAD_ONLY})


def _strip(text: str) -> str:
    t = unicodedata.normalize("NFKC", text or "").lower()
    return "".join(c for c in t if not unicodedata.category(c)[0] in "PSZC")


def asr_tokens(text: str, lang: str) -> list[str]:
    if lang == "en":
        t = unicodedata.normalize("NFKC", text or "").lower()
        t = re.sub(r"[^\w\s']", " ", t)
        return [w.strip("'") for w in t.split() if w.strip("'")]
    return list(_strip(text))


def edit_distance(a: list, b: list) -> int:
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, y in enumerate(b, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y))
        prev = cur
    return prev[-1]


def chrf_stats(hyp: str, ref: str, n: int = 6) -> list[tuple[int, int, int]]:
    """各 n(1..6) の (一致数, hyp の n-gram 数, ref の n-gram 数)。空白は除く(sacrebleu と同じ)。"""
    h = re.sub(r"\s+", "", hyp or "")
    r = re.sub(r"\s+", "", ref or "")
    out = []
    for k in range(1, n + 1):
        hc: dict[str, int] = {}
        rc: dict[str, int] = {}
        for i in range(len(h) - k + 1):
            hc[h[i:i + k]] = hc.get(h[i:i + k], 0) + 1
        for i in range(len(r) - k + 1):
            rc[r[i:i + k]] = rc.get(r[i:i + k], 0) + 1
        m = sum(min(c, rc.get(g, 0)) for g, c in hc.items())
        out.append((m, sum(hc.values()), sum(rc.values())))
    return out


def chrf_from_stats(stats: list[tuple[int, int, int]], beta: float = 2.0) -> float:
    """sacrebleu の chrF と同じ: n ごとの P/R を平均してから F_beta。0〜100。"""
    ps, rs = [], []
    for m, nh, nr in stats:
        if nh > 0 and nr > 0:
            ps.append(m / nh)
            rs.append(m / nr)
        elif nr > 0:
            ps.append(0.0)
            rs.append(0.0)
    if not ps:
        return 0.0
    p, r = sum(ps) / len(ps), sum(rs) / len(rs)
    if p + r == 0:
        return 0.0
    b2 = beta * beta
    return 100.0 * (1 + b2) * p * r / (b2 * p + r)


def chrf(hyp: str, ref: str) -> float:
    return chrf_from_stats(chrf_stats(hyp, ref))


def add_stats(acc: list | None, st: list) -> list:
    if acc is None:
        return [list(x) for x in st]
    for a, x in zip(acc, st):
        for i in range(3):
            a[i] += x[i]
    return acc


def out_text_lang(text: str, expect: str) -> str:
    """訳文の言語。漢字だけの文は文字種で ja/zh を決められないので:
    zh が期待なら zh。ja が期待でも、かなが1字も無い漢字だけの文は5字以上なら zh とみなす
    (日本語の訳文はほぼ必ずかなを含む。「以上」「了解」のような短い語だけ ja のまま)。"""
    from tsuyaku.langdetect import HAN, KANA, _LATIN_WORD, detect_lang, norm
    d = detect_lang(text)
    if d in ("ja", "zh") and HAN.search(text) and not KANA.search(text) \
            and not _LATIN_WORD.search(text):
        if expect == "zh":
            return "zh"
        return "ja" if len(norm(text)) <= 4 else "zh"
    return d


def copy_ratio(trans: str, src: str) -> float:
    """訳が原文の写しか(正規化した文字の一致率、0〜1)。"""
    import difflib
    a, b = _strip(trans), _strip(src)
    if not a or not b:
        return 0.0
    return difflib.SequenceMatcher(None, a, b, autojunk=False).ratio()


def unit_precision(hyp: str, gold: str, lang: str, min_block: int = 1) -> float:
    """聞き取った語(字)のうち正解にあるものの割合。低い = 幻覚・別物。"""
    import difflib
    h, g = asr_tokens(hyp, lang), asr_tokens(gold, lang)
    if not h:
        return 1.0
    sm = difflib.SequenceMatcher(None, h, g, autojunk=False)
    # min_block: CJK/ko で「ました」等の1〜2字の偶然一致を数えない(en は語なので1のまま)
    mb = 1 if lang == "en" else min_block
    return sum(b.size for b in sm.get_matching_blocks() if b.size >= mb) / len(h)


def _join(parts: list[str], lang: str) -> str:
    sep = " " if lang in ("en", "ko") else ""
    return sep.join(p for p in parts if p)


def _pct(xs: list[float], q: float) -> float | None:
    if not xs:
        return None
    xs = sorted(xs)
    return round(xs[min(len(xs) - 1, int(math.ceil(q * len(xs))) - 1)], 2)


def _has_run(seq: list, sub: list) -> bool:
    n = len(sub)
    return any(seq[i:i + n] == sub for i in range(len(seq) - n + 1))


def babble_leak(u: dict, babble: list, turns: list) -> list[str]:
    """単位の聞き取りに、同時刻の背景の声(ざわめき)の言葉が混ざったか。

    単位の時間(前後0.5秒の余裕)と重なる背景の発話ごとに、その言語の単位で
    (en=語、他=字)聞き取りと背景の文の共通の連続部分を探し、
    en 3語以上 / 他 4字以上で、かつ正解(重なるターンの文)には無いものを混入とする。"""
    out: list[str] = []
    a, b = u["speech_lo"] - 0.5, u["hi"] + 0.5
    import difflib
    for bg in babble:
        if bg["end"] <= a or bg["start"] >= b:
            continue
        L = bg["lang"]
        h = asr_tokens(u["text"], L)
        bt = asr_tokens(bg["text"], L)
        g = asr_tokens(_join([turns[i]["text"] for i in u.get("turns", [])], L), L)
        thr = 3 if L == "en" else 4
        sm = difflib.SequenceMatcher(None, h, bt, autojunk=False)
        for blk in sm.get_matching_blocks():
            if blk.size >= thr:
                run = h[blk.a:blk.a + blk.size]
                if not _has_run(g, run):
                    out.append((" " if L == "en" else "").join(run))
    return out


def evaluate(conv_json: Path, out: Path) -> Path:
    from tsuyaku.translate import out_lang
    meta = json.loads(conv_json.read_text(encoding="utf-8"))
    run = json.loads((out / "run.json").read_text(encoding="utf-8"))
    pair = meta["pair"]
    pair_t = tuple(pair.split("-"))
    turns = meta["turns"]
    units = run["units"]

    # ---- 単位 ↔ 正解ターンの対応(時間の重なり) ----
    for t in turns:
        t["units"] = []
    for k, u in enumerate(units):
        a, b = u["speech_lo"], u["hi"]
        ov = []
        for t in turns:
            o = min(b, t["end"]) - max(a, t["start"])
            dur = t["end"] - t["start"]
            if o >= 0.2 or (o > 0 and o >= 0.3 * dur):
                ov.append((o, t["idx"]))
        u["turns"] = [i for _, i in sorted(ov, key=lambda x: x[1])]
        u["dom"] = max(ov)[1] if ov else None
        for _, i in ov:
            turns[i]["units"].append(k)
        spk = {turns[i]["spk"] for i in u["turns"]}
        u["seg"] = ("spurious" if not ov else "cross" if len(spk) > 1
                    else "merged" if len(u["turns"]) > 1 else "ok")
        # cross のうち、同じ言語の別人どうし(言語で切れない)か
        u["cross_same_lang"] = (u["seg"] == "cross"
                                and len({turns[i]["lang"] for i in u["turns"]}) == 1)
        u["leak"] = babble_leak(u, meta.get("babble") or [], turns)
        if u["dom"] is not None:
            gl = turns[u["dom"]]["lang"]
            u["gold_lang"] = gl
            u["lid_ok"] = (u["lang"] == gl)
            tgt = out_lang(pair_t, gl)
            u["gold_src"] = _join([turns[i]["text"] for i in u["turns"]], gl)
            u["gold_ref"] = _join([turns[i]["ref"][tgt] for i in u["turns"]], tgt)
            u["chrf"] = 0.0 if u["fail"] else round(chrf(u["trans"], u["gold_ref"]), 1)
            ht, rt = asr_tokens(u["text"], gl), asr_tokens(u["gold_src"], gl)
            u["cer"] = round(edit_distance(ht, rt) / max(1, len(rt)), 3)
        exp_out = out_lang(pair_t, u["lang"]) if u["lang"] else ""
        ol = out_text_lang(u["trans"], exp_out) if not u["fail"] else ""
        u["out_lang"] = ol
        u["copy"] = (not u["fail"]) and copy_ratio(u["trans"], u["text"]) >= 0.7
        u["passthrough"] = (bool(ol) and ol == u["lang"]) or u["copy"]
        u["junk"] = u.get("dom") is not None and unit_precision(
            u["text"], u["gold_src"], u["gold_lang"]) < 0.3
        u["wrong_target"] = bool(ol) and not u["passthrough"] and ol != exp_out
        if u.get("dom") is not None:
            gold_end = min(u["hi"], max(turns[i]["end"] for i in u["turns"]))
            u["lat_ready"] = round(u["ready"] - gold_end, 3)
            u["lat_caption"] = round(u["tr_end"] - gold_end, 3)
            u["lat_first"] = (round(u["tr_start"] + u["ttft_s"] - gold_end, 3)
                              if u.get("ttft_s") is not None else None)
            if u.get("tr_end_spec") is not None:
                u["lat_caption_spec"] = round(u["tr_end_spec"] - gold_end, 3)
                u["lat_first_spec"] = round(u["first_spec"] - gold_end, 3)
            if u.get("first_any") is not None:
                u["lat_first_any"] = round(u["first_any"] - gold_end, 3)

    # ---- ターン単位の ASR / 翻訳(そのターンが主の単位をつなぐ) ----
    langs = list(pair_t)
    per = {L: {"turns": 0, "missing": 0, "split": 0, "asr_err": 0, "asr_n": 0,
               "asr_err_clean": 0, "asr_n_clean": 0, "chrf": None, "chrf_clean": None,
               "chrf_sent": [], "units": 0, "lid_ok": 0, "trad_units": 0, "zh_units": 0,
               "fail": 0, "passthrough": 0, "wrong_target": 0, "lat_ready": [],
               "lat_caption": [], "tr_s": [], "lat_first": [], "lat_caption_spec": [],
               "lat_first_spec": [], "spec_hit": 0} for L in langs}
    turn_rows = []
    for t in turns:
        L = t["lang"]
        P = per[L]
        P["turns"] += 1
        mine = [units[k] for k in sorted(set(t["units"])) if units[k]["dom"] == t["idx"]]
        if not t["units"]:
            P["missing"] += 1
        if len(set(t["units"])) > 1:
            P["split"] += 1
        hyp = _join([u["text"] for u in mine], L)
        ref_t = asr_tokens(t["text"], L)
        e = edit_distance(asr_tokens(hyp, L), ref_t)
        P["asr_err"] += e
        P["asr_n"] += len(ref_t)
        clean = len(set(t["units"])) == 1 and len(mine) == 1 and mine[0]["seg"] == "ok"
        if clean:
            P["asr_err_clean"] += e
            P["asr_n_clean"] += len(ref_t)
        tgt = out_lang(pair_t, L)
        trans = _join([u["trans"] for u in mine if not u["fail"]], tgt)
        st = chrf_stats(trans, t["ref"][tgt])
        P["chrf"] = add_stats(P["chrf"], st)
        if clean:
            P["chrf_clean"] = add_stats(P["chrf_clean"], st)
        P["chrf_sent"].append(chrf_from_stats(st))
        turn_rows.append({"idx": t["idx"], "spk": t["spk"], "lang": L, "gap": t["gap_kind"],
                          "units": sorted(set(t["units"])), "hyp": hyp, "ref": t["text"],
                          "trans": trans, "trans_ref": t["ref"][tgt],
                          "err_rate": round(e / max(1, len(ref_t)), 3),
                          "chrf": round(chrf_from_stats(st), 1), "clean": clean})
    seg_counts = {"ok": 0, "cross": 0, "merged": 0, "spurious": 0}
    junk = [{"text": u["text"], "lang": u["lang"], "gold": u.get("gold_lang"),
             "trans": u["trans"]} for u in units if u.get("junk")]
    # 捨てた区間のうち本物の発話と重なるもの(rms/空/幻覚判定/詰まり + Endpointer が捨てた断片)
    dropped: dict = {}
    dropped_real: list = []
    for ev in run["events"]:
        reason = ev.get("skip")
        if not reason:
            covered = any(u["lo"] <= ev["hi"] - 0.05 and u["hi"] >= ev["lo"] + 0.05 for u in units)
            if covered or not ev.get("text"):
                continue
            reason = "ep_drop"
        s_lo = ev["lo"] + ev.get("pre", 0.0)
        best, bt = 0.0, None
        for t in turns:
            o = min(ev["hi"], t["end"]) - max(s_lo, t["start"])
            if o > best:
                best, bt = o, t
        real = best >= 0.2
        d = dropped.setdefault(reason, [0, 0])
        d[0] += 1
        d[1] += int(real)
        if real:
            dropped_real.append({"reason": reason, "text": ev.get("text", ""), "prob": ev.get("prob"),
                                 "dur": round(ev["hi"] - s_lo, 2), "at": round(s_lo, 1),
                                 "gold_lang": bt["lang"], "gold": bt["text"],
                                 "turn_missing": not bt["units"],
                                 # 捨てた文字のうち正解にある割合(0.3未満 = 幻覚を捨てた。None = 文字なし)
                                 "match": (round(unit_precision(ev["text"], bt["text"], bt["lang"], min_block=3), 2)
                                           if ev.get("text") else None)})
    ghosts = [{"text": u["text"], "lang": u["lang"], "trans": u["trans"], "fail": u["fail"],
               "at": round(u["speech_lo"], 1), "leak": u["leak"]} for u in units if u["seg"] == "spurious"]
    leaks = [{"text": u["text"], "leak": u["leak"], "seg": u["seg"], "at": round(u["speech_lo"], 1)}
             for u in units if u["leak"]]
    for u in units:
        seg_counts[u["seg"]] += 1
        if u.get("dom") is None:
            continue
        P = per[u["gold_lang"]]
        P["units"] += 1
        P["lid_ok"] += int(u["lid_ok"])
        P["fail"] += int(u["fail"])
        P["passthrough"] += int(u["passthrough"])
        P["wrong_target"] += int(u["wrong_target"])
        for key in ("lat_ready", "lat_caption", "lat_first", "lat_caption_spec", "lat_first_spec"):
            if u.get(key) is not None:
                P[key].append(u[key])
        P["spec_hit"] += u.get("spec_t") is not None
        P["tr_s"].append(u["tr_s"])
    for u in units:  # 繁体字: zh と判定された出力 / zh への訳
        if u["lang"] == "zh" and u.get("gold_lang") in per:
            per[u["gold_lang"]]["zh_units"] += 1
            if traditional_chars(u["text"]):
                per[u["gold_lang"]]["trad_units"] += 1
    trad_trans = sum(1 for u in units if not u["fail"] and u.get("out_lang") == "zh"
                     and traditional_chars(u["trans"]))

    summary = {"pair": pair, "conv": meta["wav"], "duration": meta["duration"],
               "turns": len(turns), "units": len(units), "seg": seg_counts,
               "model": run["meta"]["model"], "vad_threshold": run["meta"]["vad_threshold"],
               "mac_fallbacks": run["meta"]["mac_fallbacks"],
               "trad_in_zh_translations": trad_trans, "junk_units": junk,
               "cond": meta.get("cond", "quiet"), "snr_db": meta.get("snr_db"),
               "git": run["meta"].get("git", "?"), "stt_retries": run["meta"].get("stt_retries", 0),
               "speakers": len(meta.get("speakers") or []) or 2,
               "overlap_turns": sum(1 for t in turns if t.get("gap_kind") == "overlap"),
               "cross_same_lang": sum(1 for u in units if u.get("cross_same_lang")),
               "ghost_units": ghosts, "leak_units": leaks,
               "dropped": dropped, "dropped_real": dropped_real,
               "stt_beam": run["meta"].get("stt_beam", 1),
               "gold_speech_s": round(sum(t["end"] - t["start"] for t in turns), 1),
               "skips": {}, "lang": {}}
    for ev in run["events"]:
        if ev.get("skip"):
            summary["skips"][ev["skip"]] = summary["skips"].get(ev["skip"], 0) + 1
    all_ready, all_cap, all_tr, all_caps, all_firsts = [], [], [], [], []
    for L, P in per.items():
        tgt = out_lang(pair_t, L)
        summary["lang"][L] = {
            "turns": P["turns"], "missing": P["missing"], "split": P["split"],
            "metric": "WER" if L == "en" else "CER",
            "asr_err": round(100 * P["asr_err"] / max(1, P["asr_n"]), 1),
            "asr_err_clean": round(100 * P["asr_err_clean"] / max(1, P["asr_n_clean"]), 1)
            if P["asr_n_clean"] else None,
            "dir": f"{L}->{tgt}",
            "chrf": round(chrf_from_stats(P["chrf"]), 1) if P["chrf"] else None,
            "chrf_clean": round(chrf_from_stats(P["chrf_clean"]), 1) if P["chrf_clean"] else None,
            "chrf_sent_mean": round(sum(P["chrf_sent"]) / max(1, len(P["chrf_sent"])), 1),
            "units": P["units"], "lid_acc": round(100 * P["lid_ok"] / max(1, P["units"]), 1),
            "fail": P["fail"], "passthrough": P["passthrough"], "wrong_target": P["wrong_target"],
            "zh_hyp_units": P["zh_units"], "trad_hyp_units": P["trad_units"],
            "lat_ready_med": _pct(P["lat_ready"], 0.5), "lat_ready_p90": _pct(P["lat_ready"], 0.9),
            "tr_s_med": _pct(P["tr_s"], 0.5), "tr_s_p90": _pct(P["tr_s"], 0.9),
            "lat_first_med": _pct(P["lat_first"], 0.5),
            "lat_caption_med": _pct(P["lat_caption"], 0.5),
            "lat_caption_p90": _pct(P["lat_caption"], 0.9),
            "lat_caption_max": _pct(P["lat_caption"], 1.0),
            "spec_hit": P["spec_hit"],
            "lat_caption_spec_med": _pct(P["lat_caption_spec"], 0.5),
            "lat_caption_spec_p90": _pct(P["lat_caption_spec"], 0.9),
            "lat_first_spec_med": _pct(P["lat_first_spec"], 0.5),
            "lat_first_spec_p90": _pct(P["lat_first_spec"], 0.9)}
        all_ready += P["lat_ready"]
        all_cap += P["lat_caption"]
        all_tr += P["tr_s"]
        all_caps += P["lat_caption_spec"]
        all_firsts += P["lat_first_spec"]
    summary["lat_ready_med"] = _pct(all_ready, 0.5)
    summary["lat_caption_med"] = _pct(all_cap, 0.5)
    summary["lat_caption_p90"] = _pct(all_cap, 0.9)
    summary["tr_s_med"] = _pct(all_tr, 0.5)
    summary["lat_caption_spec_med"] = _pct(all_caps, 0.5)
    summary["lat_caption_spec_p90"] = _pct(all_caps, 0.9)
    summary["lat_first_spec_med"] = _pct(all_firsts, 0.5)
    summary["lat_first_spec_p90"] = _pct(all_firsts, 0.9)
    summary["spec_hits"] = sum(1 for u in units if u.get("spec_t") is not None)
    # 予想訳(preview, 11/06): このMacの耳の文字で始めた先読み
    jobs = run.get("spec_jobs", [])
    pj = [j for j in jobs if j["src"] == "preview"]
    fa = [u["lat_first_any"] for u in units if u.get("lat_first_any") is not None]
    gq = []
    for j in pj:
        k = j.get("in_unit")
        if j.get("visible") and k is not None and units[k].get("gold_ref") and j.get("result"):
            gq.append((chrf(j["result"], units[k]["gold_ref"]), units[k]["chrf"]))
    rm = run["meta"]
    summary["preview"] = {
        "preview_s": rm.get("preview_s", 0.0), "hedge_s": rm.get("hedge_s", 0.0),
        "stt_calls": rm.get("stt_calls"), "previewed": rm.get("stt_previewed", 0),
        "returned": rm.get("preview_returned", 0), "started": len(pj),
        "hits": sum(1 for j in pj if j.get("unit") is not None),
        "visible": sum(1 for j in pj if j.get("visible")),
        "orphan_visible": sum(1 for j in pj if j.get("visible") and j.get("in_unit") is None),
        "orphan_examples": [j["text"][:24] for j in pj if j.get("visible") and j.get("in_unit") is None][:6],
        "first_any_src": {s_: sum(1 for u in units if u.get("lat_first_any") is not None
                                  and u.get("first_any_src") == s_)
                          for s_ in sorted({u.get("first_any_src") for u in units if u.get("first_any_src")})},
        "guess_chrf": round(sum(a for a, _ in gq) / len(gq), 1) if gq else None,
        "final_chrf_same_units": round(sum(b for _, b in gq) / len(gq), 1) if gq else None,
        "guess_n": len(gq), "stt_hedged": rm.get("stt_hedged", 0)}
    summary["lat_first_any_med"] = _pct(fa, 0.5)
    summary["lat_first_any_p90"] = _pct(fa, 0.9)
    summary["split_turns"] = sum(P["split"] for P in per.values())
    summary["missing_turns"] = sum(P["missing"] for P in per.values())

    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1),
                                      encoding="utf-8")
    (out / "units.json").write_text(json.dumps({"units": units, "turns": turn_rows},
                                               ensure_ascii=False, indent=1), encoding="utf-8")
    lines = [format_summary(summary), "", "== 単位ごと(時刻は会話内の秒) =="]
    for k, u in enumerate(units):
        lines.append(
            f"[u{k:02d}] {u['speech_lo']:6.1f}-{u['hi']:6.1f} {u['lang'] or '-'}"
            f"(正解{u.get('gold_lang', '-')}) seg={u['seg']} turns={u['turns']} parts={u['parts']}"
            f" ready+{u.get('lat_ready')} 字幕+{u.get('lat_caption')} chrF={u.get('chrf')}"
            f" CER={u.get('cer')}{' 素通し' if u['passthrough'] else ''}"
            f"{' 失敗' if u['fail'] else ''}")
        lines.append(f"      聞: {u['text']}")
        lines.append(f"      正: {u.get('gold_src', '')}")
        lines.append(f"      訳: {u['trans']}")
        lines.append(f"      参: {u.get('gold_ref', '')}")
    # ワーストはターン単位(1ターンが割れた断片を個別に並べると同じ文ばかりになる)
    lines += ["", "== ワースト5(ターンの chrF。単位の言語/区切りつき) =="]
    worst = sorted(turn_rows, key=lambda r: (r["chrf"], -r["err_rate"]))[:5]
    for r in worst:
        us = [units[k] for k in r["units"]]
        tags = ", ".join(f"u{k}:{units[k]['lang'] or '-'}/{units[k]['seg']}"
                         f"{'/素通し' if units[k]['passthrough'] else ''}"
                         f"{'/失敗' if units[k]['fail'] else ''}" for k in r["units"])
        lines.append(f"* turn{r['idx']} {r['spk']}:{r['lang']} 間={r['gap']} chrF={r['chrf']} "
                     f"誤り率={r['err_rate']} 単位[{tags or '無し'}]")
        lines.append(f"    正: {r['ref']}\n    聞: {' | '.join(u['text'] for u in us)}\n"
                     f"    参: {r['trans_ref']}\n    訳: {' | '.join(u['trans'] for u in us)}")
    (out / "report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    w0 = next(i for i, x in enumerate(lines) if x.startswith("== ワースト5"))
    print("\n".join(lines[:1] + lines[w0:]))
    return out / "summary.json"


def format_summary(sm: dict) -> str:
    seg = sm["seg"]
    cond = sm.get("cond", "quiet")
    extra = (f", 話者{sm.get('speakers')}人, SNR {sm.get('snr_db')}dB, 重なり{sm.get('overlap_turns')}"
             if cond == "venue" else "")
    head = (f"### {sm['pair']} [{cond}{extra}] ({sm['conv']}, {sm['duration']}s, ターン{sm['turns']}, "
            f"単位{sm['units']}, model={sm['model']}, vad={sm['vad_threshold']}, app={sm.get('git', '?')})\n"
            f"区切り: ok {seg['ok']} / cross {seg['cross']} / merged {seg['merged']} / "
            f"spurious {seg['spurious']} / split(ターン) {sm['split_turns']} / "
            f"missing(ターン) {sm['missing_turns']} / 捨てた区間 {sm['skips']} / "
            f"zh訳の繁体字 {sm['trad_in_zh_translations']} / Mac落ち {sm['mac_fallbacks']}\n"
            f"幻覚・別物の単位(聞いた字の7割以上が正解に無い) {len(sm.get('junk_units', []))}: "
            f"{' / '.join(j['text'][:20] for j in sm.get('junk_units', []))}\n"
            f"ゴースト単位(正解の声と重ならない) {len(sm.get('ghost_units', []))}"
            f"(訳が出た {sum(1 for g in sm.get('ghost_units', []) if not g['fail'])}): "
            f"{' / '.join(g['text'][:20] for g in sm.get('ghost_units', [])[:8])}\n"
            f"ざわめき混入の単位 {len(sm.get('leak_units', []))}: "
            f"{' / '.join(','.join(x['leak'])[:24] for x in sm.get('leak_units', [])[:8])} / "
            f"同じ言語の別人を1単位に {sm.get('cross_same_lang', 0)}\n"
            f"捨てた区間 {{理由: [全体, 本物の発話と重なる]}} {sm.get('dropped', {})}: "
            f"{' / '.join(d['text'][:16] or '(無音扱い)' for d in sm.get('dropped_real', [])[:8])}\n"
            f"遅延(中央値): 訳してよい+{sm['lat_ready_med']}s 翻訳{sm['tr_s_med']}s "
            f"字幕+{sm['lat_caption_med']}s (p90 {sm['lat_caption_p90']}s)"
            + (f"\n先読みあり: 的中{sm.get('spec_hits')}/{sm['units']}単位 / 確定字幕+"
               f"{sm['lat_caption_spec_med']}s (p90 {sm['lat_caption_spec_p90']}s) / 最初の訳(予想含む)+"
               f"{sm['lat_first_spec_med']}s (p90 {sm['lat_first_spec_p90']}s)"
               if sm.get("lat_caption_spec_med") is not None else "")
            + (f"\n予想訳(このMacの耳 {sm['preview']['preview_s']}s): 先出し{sm['preview']['returned']}回"
               f"/耳{sm['preview']['stt_calls']}回 → 訳を開始{sm['preview']['started']} 見えた{sm['preview']['visible']}"
               f" 当たり{sm['preview']['hits']} 単位外{sm['preview']['orphan_visible']} / 最初の訳(予想訳含む)+"
               f"{sm.get('lat_first_any_med')}s (p90 {sm.get('lat_first_any_p90')}s) / 最初の訳の出どころ "
               f"{sm['preview']['first_any_src']} / 予想訳 chrF {sm['preview']['guess_chrf']} vs 同じ単位の確定 "
               f"{sm['preview']['final_chrf_same_units']} (n={sm['preview']['guess_n']}) / 4秒切れ {sm['preview']['stt_hedged']}"
               if sm.get("preview") else ""))
    rows = ["| 言語 | ターン | 欠落 | 分割 | 言語判定 | ASR誤り(全体/きれい) | 方向 | chrF(全体/きれい) "
            "| 失敗 | 素通し | 別言語 | zh繁体 | 訳してよい中央/p90 | 翻訳中央 | 字幕中央/p90/最大 |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for L, d in sm["lang"].items():
        rows.append(
            f"| {L} | {d['turns']} | {d['missing']} | {d['split']} | {d['lid_acc']}% ({d['units']}) "
            f"| {d['metric']} {d['asr_err']}% / {d['asr_err_clean']}% | {d['dir']} "
            f"| {d['chrf']} / {d['chrf_clean']} | {d['fail']} | {d['passthrough']} | {d['wrong_target']} "
            f"| {d['trad_hyp_units']}/{d['zh_hyp_units']} | {d['lat_ready_med']}/{d['lat_ready_p90']} "
            f"| {d['tr_s_med']} | {d['lat_caption_med']}/{d['lat_caption_p90']}/{d['lat_caption_max']} |")
    return head + "\n" + "\n".join(rows)


def report(paths: list[Path], out: Path | None = None) -> None:
    txt = "\n\n".join(format_summary(json.loads(p.read_text(encoding="utf-8"))) for p in paths)
    print(txt)
    if out:
        out.write_text(txt + "\n", encoding="utf-8")


def compare(paths: list[Path], out: Path | None = None) -> None:
    """複数の run(summary.json)を 条件×方向 でまとめる(ターン/単位を合算して計算し直す)。"""
    from tsuyaku.translate import out_lang
    agg: dict = {}
    extra: dict = {}
    for sp in paths:
        sm = json.loads(sp.read_text(encoding="utf-8"))
        d = json.loads((sp.parent / "units.json").read_text(encoding="utf-8"))
        cond, pair = sm.get("cond", "quiet"), sm["pair"]
        pair_t = tuple(pair.split("-"))
        e = extra.setdefault((cond, pair), {"runs": 0, "units": 0, "ghost": 0, "ghost_tr": 0,
                                            "leak": 0, "cross": 0, "cross_same": 0, "speech_s": 0.0,
                                            "skips": {}})
        e["runs"] += 1
        e["units"] += sm["units"]
        e["ghost"] += len(sm.get("ghost_units", []))
        e["ghost_tr"] += sum(1 for g in sm.get("ghost_units", []) if not g["fail"])
        e["leak"] += len(sm.get("leak_units", []))
        e["cross"] += sm["seg"]["cross"]
        e["cross_same"] += sm.get("cross_same_lang", 0)
        e["speech_s"] += sm.get("gold_speech_s", 0.0)
        for k, v in sm.get("skips", {}).items():
            e["skips"][k] = e["skips"].get(k, 0) + v
        for r in d["turns"]:
            L = r["lang"]
            a = agg.setdefault((cond, pair, L), {"turns": 0, "missing": 0, "split": 0, "err": 0,
                                                 "n": 0, "chrf": None, "units": 0, "lid": 0,
                                                 "pt": 0, "wt": 0, "fail": 0, "junk": 0, "cap": []})
            a["turns"] += 1
            a["missing"] += int(not r["units"])
            a["split"] += int(len(r["units"]) > 1)
            rt = asr_tokens(r["ref"], L)
            a["err"] += edit_distance(asr_tokens(r["hyp"], L), rt)
            a["n"] += len(rt)
            a["chrf"] = add_stats(a["chrf"], chrf_stats(r["trans"], r["trans_ref"]))
        for u in d["units"]:
            if u.get("gold_lang") is None:
                continue
            a = agg[(cond, pair, u["gold_lang"])]
            a["units"] += 1
            a["lid"] += int(u["lid_ok"])
            a["pt"] += int(u["passthrough"])
            a["wt"] += int(u["wrong_target"])
            a["fail"] += int(u["fail"])
            a["junk"] += int(bool(u.get("junk")))
            if u.get("lat_caption") is not None:
                a["cap"].append(u["lat_caption"])
    rows = ["| 条件 | 方向 | ターン | 欠落 | 分割 | 言語判定 | ASR誤り | chrF | 素通し+別言語 | 失敗 | 幻覚・別物 "
            "| 字幕 中央/p90 |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for (cond, pair, L), a in sorted(agg.items()):
        tgt = out_lang(tuple(pair.split("-")), L)
        rows.append(
            f"| {cond} | {pair}:{L}->{tgt} | {a['turns']} | {a['missing']} | {a['split']} "
            f"| {a['lid']}/{a['units']} | {'WER' if L == 'en' else 'CER'} {100 * a['err'] / max(1, a['n']):.1f}% "
            f"| {chrf_from_stats(a['chrf']):.1f} | {a['pt'] + a['wt']} | {a['fail']} | {a['junk']} "
            f"| {_pct(a['cap'], 0.5)}/{_pct(a['cap'], 0.9)} |")
    rows += ["", "| 条件 | ペア | run数 | 単位 | cross(同じ言語の別人) | ゴースト(訳が出た) | ざわめき混入 | 捨てた区間 |",
             "|---|---|---|---|---|---|---|---|"]
    for (cond, pair), e in sorted(extra.items()):
        rows.append(f"| {cond} | {pair} | {e['runs']} | {e['units']} | {e['cross']}({e['cross_same']}) "
                    f"| {e['ghost']}({e['ghost_tr']}) | {e['leak']} | {e['skips']} |")
    txt = "\n".join(rows)
    print(txt)
    if out:
        out.write_text(txt + "\n", encoding="utf-8")


def _pool(res_dir: Path) -> dict:
    """results ディレクトリ → {(pair, cond): 合算した指標}(seed をまたいで合算)。"""
    out: dict = {}
    for sp in sorted(res_dir.glob("*/summary.json")):
        sm = json.loads(sp.read_text(encoding="utf-8"))
        d = json.loads((sp.parent / "units.json").read_text(encoding="utf-8"))
        k = (sm["pair"], sm.get("cond", "quiet"))
        a = out.setdefault(k, {"runs": [], "units": 0, "junk": 0, "split": 0, "missing": 0,
                               "turns": 0, "drop_real": 0, "drop_junk": 0, "drop_silent": 0, "ghost": 0, "cross": 0, "chrf": None,
                               "chrf_dir": {}, "cap": [], "pt": 0, "lid_bad": 0, "stt": [],
                               "cap_spec": [], "first_any": [], "pv": [0, 0, 0, 0]})
        a["runs"].append(sp.parent.name)
        a["units"] += sm["units"]
        a["junk"] += len(sm.get("junk_units", []))
        a["split"] += sm["split_turns"]
        a["missing"] += sm["missing_turns"]
        a["turns"] += sm["turns"]
        for dr in sm.get("dropped_real", []):
            m = dr.get("match")
            key = "drop_real" if (m is not None and m >= 0.3) else ("drop_junk" if m is not None else "drop_silent")
            a[key] = a.get(key, 0) + 1
        a["ghost"] += len(sm.get("ghost_units", []))
        a["cross"] += sm["seg"]["cross"]
        for r in d["turns"]:
            st = chrf_stats(r["trans"], r["trans_ref"])
            a["chrf"] = add_stats(a["chrf"], st)
            a["chrf_dir"][r["lang"]] = add_stats(a["chrf_dir"].get(r["lang"]), st)
        for u in d["units"]:
            if u.get("lat_caption") is not None:
                a["cap"].append(u["lat_caption"])
            if u.get("lat_caption_spec") is not None:
                a["cap_spec"].append(u["lat_caption_spec"])
            if u.get("lat_first_any") is not None:
                a["first_any"].append(u["lat_first_any"])
            a["pt"] += int(u["passthrough"] or u["wrong_target"])
            a["lid_bad"] += int(u.get("gold_lang") is not None and not u["lid_ok"])
        run = json.loads((sp.parent / "run.json").read_text(encoding="utf-8"))
        a["stt"] += [e["stt_s"] for e in run["events"] if e.get("stt_s")]
        pv = sm.get("preview") or {}
        for i_, key_ in enumerate(("returned", "visible", "hits", "orphan_visible")):
            a["pv"][i_] += pv.get(key_, 0) or 0
    return out


def ab(dir_a: Path, dir_b: Path, out: Path | None = None) -> None:
    """A/B 比較: ペア×条件ごとに A → B(seed 合算)。"""
    A, B = _pool(dir_a), _pool(dir_b)
    rows = [f"A={dir_a.name} / B={dir_b.name}(seed 合算。欠落=単位が1つも無いターン、"
            "本物を捨てた=本物の発話と0.2秒以上重なる区間を捨て、その文字の3割以上が正解にあった数。"
            "幻覚を捨てた=重なるが文字が正解にほぼ無い。文字なし=rms/空で捨てた)", "",
            "| ペア | 条件 | ターン | 単位 | 幻覚・別物 | 分割 | 欠落 | 本物を捨てた | 幻覚を捨てた | 文字なしで捨てた | ゴースト | cross "
            "| 言語判定ミス | 素通し・別言語 | chrF | 字幕 中央/p90 | 耳 中央/p90 "
            "| 確定字幕(先読み) 中央/p90 | 最初の訳(予想含む) 中央/p90 | 予想 先出し/見えた/当たり/単位外 |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    def ab_s(x, y):
        return f"{x} → {y}" if x != y else f"{x}"
    for k in sorted(set(A) | set(B)):
        a, b = A.get(k), B.get(k)
        if not a or not b:
            rows.append(f"| {k[0]} | {k[1]} | (片方なし: A={bool(a)} B={bool(b)}) |")
            continue
        f = lambda z: f"{chrf_from_stats(z['chrf']):.1f}"
        cap = lambda z: f"{_pct(z['cap'], 0.5)}/{_pct(z['cap'], 0.9)}"
        stt = lambda z: f"{_pct(z['stt'], 0.5)}/{_pct(z['stt'], 0.9)}"
        pp = lambda z, k: f"{_pct(z[k], 0.5)}/{_pct(z[k], 0.9)}"
        rows.append(
            f"| {k[0]} | {k[1]} | {a['turns']} | {ab_s(a['units'], b['units'])} "
            f"| {ab_s(a['junk'], b['junk'])} | {ab_s(a['split'], b['split'])} "
            f"| {ab_s(a['missing'], b['missing'])} | {ab_s(a['drop_real'], b['drop_real'])} "
            f"| {ab_s(a['drop_junk'], b['drop_junk'])} | {ab_s(a['drop_silent'], b['drop_silent'])} "
            f"| {ab_s(a['ghost'], b['ghost'])} | {ab_s(a['cross'], b['cross'])} "
            f"| {ab_s(a['lid_bad'], b['lid_bad'])} | {ab_s(a['pt'], b['pt'])} "
            f"| {ab_s(f(a), f(b))} | {ab_s(cap(a), cap(b))} | {ab_s(stt(a), stt(b))} "
            f"| {ab_s(pp(a, 'cap_spec'), pp(b, 'cap_spec'))} "
            f"| {ab_s(pp(a, 'first_any'), pp(b, 'first_any'))} "
            f"| {ab_s('/'.join(map(str, a['pv'])), '/'.join(map(str, b['pv'])))} |")
    rows += ["", "| ペア | 条件 | 方向別 chrF(原文の言語) A → B |", "|---|---|---|"]
    for k in sorted(set(A) & set(B)):
        parts = [f"{L}: {chrf_from_stats(A[k]['chrf_dir'][L]):.1f} → {chrf_from_stats(B[k]['chrf_dir'][L]):.1f}"
                 for L in A[k]["chrf_dir"] if L in B[k]["chrf_dir"]]
        rows.append(f"| {k[0]} | {k[1]} | {' / '.join(parts)} |")
    txt = "\n".join(rows)
    print(txt)
    if out:
        out.write_text(txt + "\n", encoding="utf-8")


# ====================================================================== CLI

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--fleurs", type=Path, required=True)
    b.add_argument("--pair", required=True)
    b.add_argument("--seed", type=int, default=1)
    b.add_argument("--dur", type=float, default=180.0)
    b.add_argument("--out", type=Path)
    b.add_argument("--cond", choices=["quiet", "venue"], default="quiet")
    b.add_argument("--speakers", type=int, default=4, help="venue の話者数(3 or 4)")
    r = sub.add_parser("run")
    r.add_argument("--conv", type=Path, required=True)
    r.add_argument("--out", type=Path, required=True)
    r.add_argument("--vad", type=float, default=None, help="VAD しきい(既定: config.json の vad_threshold)")
    r.add_argument("--topic", default="")
    r.add_argument("--stt-beam", type=int, default=None, help="リモートサーバーの耳に ?beam=N(実験)")
    r.add_argument("--mac", action="store_true", help="このMacだけ(耳=SenseVoice, 翻訳=LM Studio)")
    r.add_argument("--hedge", type=float, default=0.0, help="リモートサーバーの耳をこの秒数だけ待ち、来なければ SenseVoice")
    r.add_argument("--preview", type=float, default=0.0,
                   help="--hedge と併用: リモートサーバーがこの秒数で返らなければこのMacの耳で予想訳を先に出す")
    r.add_argument("--mac-ear", action="store_true", help="耳だけこのMac(SenseVoice)、翻訳は remote")
    r.add_argument("--mac-model", default=None, help="このMacの翻訳モデル(既定 config の ENGINE_MODELS)")
    e = sub.add_parser("eval", help="run.json から指標だけ計算し直す")
    e.add_argument("--conv", type=Path, required=True)
    e.add_argument("--out", type=Path, required=True)
    p = sub.add_parser("report")
    p.add_argument("summaries", type=Path, nargs="+")
    p.add_argument("--save", type=Path)
    c = sub.add_parser("compare", help="条件×方向で合算した表")
    c.add_argument("summaries", type=Path, nargs="+")
    c.add_argument("--save", type=Path)
    x = sub.add_parser("ab", help="2つの results ディレクトリの A/B 比較")
    x.add_argument("--a", type=Path, required=True)
    x.add_argument("--b", type=Path, required=True)
    x.add_argument("--save", type=Path)
    a = sub.add_parser("all")
    a.add_argument("--fleurs", type=Path, required=True)
    a.add_argument("--pairs", default="ja-en,ja-zh,ja-ko,en-zh")
    a.add_argument("--seed", type=int, default=1)
    a.add_argument("--dur", type=float, default=180.0)
    a.add_argument("--vad", type=float, default=None)
    a.add_argument("--conds", default="quiet,venue")
    a.add_argument("--speakers", type=int, default=4)
    args = ap.parse_args()

    def make(pr, seed, dur, out, cond, speakers):
        if cond == "venue":
            return build_venue(args.fleurs, pr, seed, dur, out, n_speakers=speakers)
        return build(args.fleurs, pr, seed, dur, out)

    if args.cmd == "build":
        make(args.pair, args.seed, args.dur, args.out or args.fleurs / "convs", args.cond, args.speakers)
    elif args.cmd == "run":
        run_pipeline(args.conv, args.out, args.vad, args.topic, args.stt_beam,
                     mac=args.mac, mac_model=args.mac_model, mac_ear=args.mac_ear, hedge=args.hedge,
                     preview=args.preview)
    elif args.cmd == "eval":
        evaluate(args.conv, args.out / args.conv.stem)
    elif args.cmd == "report":
        report(args.summaries, args.save)
    elif args.cmd == "compare":
        compare(args.summaries, args.save)
    elif args.cmd == "ab":
        ab(args.a, args.b, args.save)
    elif args.cmd == "all":
        sums = []
        for cond in args.conds.split(","):
            for pr in args.pairs.split(","):
                cj = make(pr, args.seed, args.dur, args.fleurs / "convs", cond, args.speakers)
                sums.append(run_pipeline(cj, args.fleurs / "results", args.vad))
        report(sums, args.fleurs / "results" / f"baseline_s{args.seed}.md")
        compare(sums, args.fleurs / "results" / f"compare_s{args.seed}.md")


if __name__ == "__main__":
    main()
