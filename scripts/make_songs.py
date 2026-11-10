"""本物に近い歌の試験音(11/08 リーダー「VOICEVOX の歌う機能で作った歌に伴奏を付けて、誤って字幕に出る量を測り直す」)。

旋律・歌詞・伴奏はすべてここで作る(著作権のない自作)。歌は VOICEVOX ENGINE の歌う機能
(/sing_frame_audio_query で先生役 6000 が声の高さと長さを決め、/frame_synthesis で歌い手が歌う)。
伴奏は和音(I–V–vi–IV)のピアノっぽい音 + ベース + キックとハイハット。歌声に対する伴奏の大きさを
0(アカペラ)/ 0.5 / 1 / 2 倍で混ぜる。正解は「字幕なし」。

実行(リモートサーバー で。VOICEVOX は自分のコンテナ、例 127.0.0.1:50121):
  python -I scripts/make_songs.py --url http://127.0.0.1:50121 --out eval/long/songs
"""
from __future__ import annotations

import argparse
import json
import wave
from pathlib import Path

import numpy as np
import requests

SR = 16000
VV_SR = 24000
FPS = VV_SR / 256              # VOICEVOX の歌の1フレーム = 256 サンプル(24kHz)
TEACHER = 6000                 # 波音リツ(歌の先生役)
SINGERS = {"metan": 3002, "zunda": 3003, "ryusei": 3013}
# 自作の歌詞(1音に1拍。小さい「っ」や拗音は使わない)
LYRICS = [
    "あさのひかりがまどをてらす", "きみのこえがきこえるよ", "とおいまちへあるいていこう",
    "かぜがやさしくせなかをおす", "ゆめのつづきをふたりでみよう", "よるのそらにほしがひかる",
    "あめのにおいがまちをつつむ", "あしたもまたここであおう",
]
SCALE = [0, 2, 4, 5, 7, 9, 11]          # 長音階
CHORDS = [[0, 4, 7], [7, 11, 14], [9, 12, 16], [5, 9, 12]]   # I V vi IV


def melody(n: int, rng, base: int) -> list[int]:
    """音階の上をなめらかに動く旋律(1拍ずつの音の高さ、MIDI)。"""
    deg, out = 2, []
    for i in range(n):
        deg = int(np.clip(deg + rng.choice([-2, -1, -1, 0, 1, 1, 2]), 0, 9))
        if i == n - 1:
            deg = 0 if rng.random() < 0.5 else 4          # 終わりは主音か属音
        out.append(base + 12 * (deg // 7) + SCALE[deg % 7])
    return out


def sing(url: str, singer: int, lines: list[str], bpm: float, rng, base: int) -> np.ndarray:
    beat = 60.0 / bpm
    parts = []
    for line in lines:
        keys = melody(len(line), rng, base)
        notes = [{"key": None, "frame_length": int(FPS * beat), "lyric": ""}]
        for i, (mora, k) in enumerate(zip(line, keys)):
            ln = beat * (2 if i == len(line) - 1 else rng.choice([0.5, 1, 1, 1, 1.5]))
            notes.append({"key": int(k), "frame_length": max(4, int(FPS * ln)), "lyric": mora})
        notes.append({"key": None, "frame_length": int(FPS * beat), "lyric": ""})
        q = requests.post(f"{url}/sing_frame_audio_query", params={"speaker": TEACHER},
                          json={"notes": notes}, timeout=120)
        q.raise_for_status()
        w = requests.post(f"{url}/frame_synthesis", params={"speaker": singer},
                          json=q.json(), timeout=300)
        w.raise_for_status()
        a = np.frombuffer(w.content[44:], np.int16).astype(np.float32) / 32768
        t_old = np.arange(len(a)) / VV_SR
        t_new = np.arange(int(len(a) * SR / VV_SR)) / SR
        parts.append(np.interp(t_new, t_old, a).astype(np.float32))
    return np.concatenate(parts)


def accomp(n: int, bpm: float, base: int, rng) -> np.ndarray:
    """和音・ベース・キック・ハイハット(1小節 = 4拍で和音が替わる)。"""
    beat = int(SR * 60.0 / bpm)
    out = np.zeros(n, np.float32)
    for b0 in range(0, n, beat):
        bar = (b0 // beat) // 4
        ch = CHORDS[bar % 4]
        m = min(beat, n - b0)
        t = np.arange(m) / SR
        env = np.exp(-2.5 * t)
        for iv in ch:                                     # ピアノっぽい和音(倍音つき)
            f = 440 * 2 ** ((base - 12 + iv - 69) / 12)
            out[b0:b0 + m] += 0.25 * env * (np.sin(2 * np.pi * f * t) + 0.4 * np.sin(4 * np.pi * f * t))
        fb = 440 * 2 ** ((base - 24 + ch[0] - 69) / 12)   # ベース
        out[b0:b0 + m] += 0.35 * np.exp(-1.5 * t) * np.sin(2 * np.pi * fb * t)
        if (b0 // beat) % 2 == 0:                         # キック
            k = min(m, int(0.12 * SR))
            tk = np.arange(k) / SR
            out[b0:b0 + k] += 0.8 * np.exp(-30 * tk) * np.sin(2 * np.pi * (60 + 80 * np.exp(-40 * tk)) * tk)
        h0 = b0 + beat // 2                               # 裏拍のハイハット
        if h0 < n:
            k = min(n - h0, int(0.04 * SR))
            out[h0:h0 + k] += 0.15 * rng.standard_normal(k).astype(np.float32) * np.exp(-np.arange(k) / 80)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:50121")
    ap.add_argument("--out", type=Path, default=Path("eval/long/songs"))
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / ".gitignore").write_text("*\n")
    for si, (sname, sid) in enumerate(SINGERS.items()):
        for bpm in (84, 120):
            rng = np.random.default_rng(100 * si + bpm)
            base = {"metan": 64, "zunda": 66, "ryusei": 52}[sname]
            lines = [LYRICS[(si * 3 + j) % len(LYRICS)] for j in range(5)]
            v = sing(a.url, sid, lines, bpm, rng, base)
            acc = accomp(len(v), bpm, base, rng)
            acc = acc / np.sqrt(np.mean(acc ** 2)) * np.sqrt(np.mean(v ** 2))
            for lvl in (0.0, 0.5, 1.0, 2.0):
                y = v + lvl * acc
                y = 0.1 * y / np.sqrt(np.mean(y ** 2))
                name = f"song_{sname}_{bpm}_m{lvl:g}"
                with wave.open(str(a.out / f"{name}.wav"), "wb") as w:
                    w.setnchannels(1)
                    w.setsampwidth(2)
                    w.setframerate(SR)
                    w.writeframes((np.clip(y, -1, 1) * 32767).astype(np.int16).tobytes())
                (a.out / f"{name}.json").write_text(json.dumps(
                    {"lang": "ja", "tgt": "zh", "sents": [], "dur": len(y) / SR, "voice": si,
                     "pace": str(bpm), "bgm": "song"}, ensure_ascii=False), encoding="utf-8")
                print(name, round(len(y) / SR, 1), flush=True)


if __name__ == "__main__":
    main()
