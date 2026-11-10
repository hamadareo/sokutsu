"""切れ目のない長い独白(zh・ja)で、長い発話を区切る前後を比べる(11/08)。

見つけた問題(リーダーが実記録で見た型、このMac = SenseVoice + gemma-3-4b):
  切れ目なく話し続けると 9〜12.5 秒が1単位になり、訳 101 字・読み上げ +3.7 秒、次の文は 14.9 秒遅れで
  読み上げを飛ばした。途中字幕は 6.5 秒あたりから伸びなくなった。

やり方(時計の上で再生。アプリの部品をそのまま使う):
  build  作った対訳の独白(eval/long/mono.json)を macOS の say で読ませ、文と文の間 0.12 秒で繋いだ WAV を作る。
         BGM あり版は和音+打楽器っぽい雑音を声の RMS の bgm 倍で重ねる(動画の音のつもり)。音は git に入れない。
  run    WAV を 512 サンプルずつ SpeechCapture.on_chunk(AutoGain → VAD → StreamSegmenter → 途中字幕)へ流し、
         STT ワーカーの規則(区間を SenseVoice で聞く → Endpointer、手が空いたら tick と途中字幕、
         途中字幕から長い発話を区切る _long_cut)を音声の時計で回す。認識の手間は 1回 0.15 秒 + 音声1秒あたり 0.01 秒。
         出た単位(文字・lo・hi・出た時刻)を JSON に残す。--split 0 が今まで(区切らない)、6 が区切る。
         --keep-bgm-text 0 は SenseVoice の BGM 判定で文字を捨てる今までの規則(途中字幕が止まる)。
  report 単位の長さ、聞き取りの誤り(CER: 単位を繋いだ文字 vs 正解)、文ごとの声の遅れ(文を言い終わってから、
         その文を含む単位の読み上げが鳴り始めるまで。p50/p90)、飛ばした文、訳の chrF(--translate があれば)。
         読み上げは eval_pace.simulate(auto+listen = 話のペースに合わせた速さ・読み上げ中も聞く)で回す。
         訳の長さは対訳の字数の比で見積もる(--translate があれば本当の訳)。訳にかかる時間は
         初字 0.3 秒 + 1字 1/40 秒(このMac の gemma-3-4b の目安)。

実行(リポジトリ直下):
  .venv/bin/python -I scripts/eval_long.py build --out eval/long/clips
  .venv/bin/python -I scripts/eval_long.py run --clips eval/long/clips --split 0 --keep-bgm-text 0 --out R/A
  .venv/bin/python -I scripts/eval_long.py run --clips eval/long/clips --split 6 --out R/B
  .venv/bin/python -I scripts/eval_long.py report --a R/A --b R/B [--translate http://127.0.0.1:1234/v1 --model gemma-3-4b]
ユーザーのアプリ(-m tsuyaku)が動いている間は run・--translate を流さない(このMacの CPU と LLM を使う)。
"""
from __future__ import annotations

import argparse
import json
import math
import os
import queue
import subprocess
import sys
import tempfile
import time
import wave
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

SR = 16000
MONO = ROOT / "eval" / "long" / "mono.json"
VOICES = {"zh": ["Tingting", "Eddy (中国語（中国本土）)"], "ja": ["Kyoko", "Eddy (日本語（日本）)"]}
RATES = {"normal": None, "fast": 1.3}      # say の既定の速さに掛ける数(None = 既定)
BGMS = {"clean": 0.0, "bgm": 0.6}
GAP_S = 0.12
STT_COST = (0.15, 0.01)                    # 認識1回の手間 = a + b × 音声秒
TR_TTFT, TR_CPS = 0.3, 40.0                # 訳: 初字 + 1秒あたりの字数


# ---------------------------------------------------------------- build

def _say(text: str, voice: str, rate: "float | None", path: str) -> np.ndarray:
    cmd = ["say", "-v", voice, "-o", path, "--data-format=LEI16@16000"]
    if rate:
        cmd += ["-r", str(int(185 * rate))]
    subprocess.run(cmd + [text], check=True)
    with wave.open(path) as w:
        a = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768
    nz = np.where(np.abs(a) > 0.01)[0]
    return a[max(0, nz[0] - 800): nz[-1] + 800] if len(nz) else a


def _bgm(n: int, seed: int) -> np.ndarray:
    t = np.arange(n) / SR
    rng = np.random.default_rng(seed)
    m = sum(np.sin(2 * np.pi * f * t * (1 + 0.05 * np.sin(0.3 * t + k)))
            for k, f in enumerate([220, 277, 330, 440, 554])) / 5
    m = m + 0.6 * rng.standard_normal(n) * (np.sin(2 * np.pi * 2 * t) > 0.6)
    return (m / np.sqrt(np.mean(m ** 2))).astype(np.float32)


def build(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / ".gitignore").write_text("*\n")
    mono = json.loads(MONO.read_text(encoding="utf-8"))
    with tempfile.TemporaryDirectory() as td:
        for topic, d in mono.items():
            if topic.startswith("_"):
                continue
            for lang in ("zh", "ja"):
                other = "ja" if lang == "zh" else "zh"
                for vi, voice in enumerate(VOICES[lang]):
                    for pace, rate in RATES.items():
                        parts, sents, t = [], [], 0.0
                        for i, s in enumerate(d[lang]):
                            a = _say(s, voice, rate, os.path.join(td, f"s{i}.wav"))
                            sents.append({"text": s, "ref": d[other][i], "start": t,
                                          "end": t + len(a) / SR})
                            parts += [a, np.zeros(int(SR * GAP_S), np.float32)]
                            t += len(a) / SR + GAP_S
                        x = np.concatenate(parts)
                        for bname, k in BGMS.items():
                            y = x + k * _bgm(len(x), vi) * np.sqrt(np.mean(x ** 2)) if k else x
                            name = f"{topic}_{lang}_v{vi}_{pace}_{bname}"
                            with wave.open(str(out / f"{name}.wav"), "wb") as w:
                                w.setnchannels(1)
                                w.setsampwidth(2)
                                w.setframerate(SR)
                                w.writeframes((np.clip(y, -1, 1) * 32767).astype(np.int16).tobytes())
                            (out / f"{name}.json").write_text(json.dumps(
                                {"lang": lang, "tgt": other, "sents": sents, "dur": len(y) / SR,
                                 "voice": vi, "pace": pace, "bgm": bname}, ensure_ascii=False, indent=1),
                                encoding="utf-8")
                            print(name, round(len(y) / SR, 1), flush=True)


# 誤って字幕に出るかを見る音(11/08 リーダー: 「Speech でなくても6字以上なら残す」で歌詞・拍手が字幕に乗らないか)。
# どれも作った音。正解は「字幕なし」。歌は say の声にビブラートと BGM を重ね、歌詞は作った文。
LYRICS = {"zh": "我们一起走过春天和夏天，风吹过城市的街道，我会一直等你回来，在那个下雨的晚上",
          "ja": "春も夏も一緒に歩いた道、風が街を吹き抜けていく、ずっと待っているよ、あの雨の夜に"}


def _vibrato(x: np.ndarray, depth_s: float = 0.0035, hz: float = 5.5) -> np.ndarray:
    t = np.arange(len(x)) / SR
    warp = t + depth_s * np.sin(2 * np.pi * hz * t)
    return np.interp(warp * SR, np.arange(len(x)), x).astype(np.float32)


def _melody(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = np.zeros(n, np.float32)
    note = int(0.35 * SR)
    for i in range(0, n, note):
        f = 220 * 2 ** (rng.integers(0, 12) / 12)
        t = np.arange(min(note, n - i)) / SR
        env = np.exp(-3 * t)
        out[i:i + len(t)] += (np.sin(2 * np.pi * f * t) + 0.3 * np.sin(4 * np.pi * f * t)) * env
    return out + 0.5 * _bgm(n, seed + 7)


def build_fp(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / ".gitignore").write_text("*\n")
    rng = np.random.default_rng(3)
    clips: dict = {}
    n = int(30 * SR)
    clips["music_chords"] = 0.1 * _bgm(n, 11)
    clips["music_melody"] = 0.1 * _melody(n, 12) / np.sqrt(np.mean(_melody(n, 12) ** 2))
    # 拍手: 短い破裂音がたくさん重なる / ざわめき: 何人もの声を小さく重ねる
    clap = np.zeros(n, np.float32)
    for _ in range(4000):
        i = int(rng.integers(0, n - 400))
        clap[i:i + 400] += rng.standard_normal(400).astype(np.float32) * np.exp(-np.arange(400) / 60)
    clips["applause"] = 0.1 * clap / np.sqrt(np.mean(clap ** 2))
    with tempfile.TemporaryDirectory() as td:
        bab = np.zeros(n, np.float32)
        lines = [s for d in json.loads(MONO.read_text(encoding="utf-8")).values() if isinstance(d, dict)
                 for lang in ("zh", "ja") for s in d[lang]]
        for k in range(8):
            v = VOICES["zh" if k % 2 else "ja"][k % 2 and 1 or 0]
            a = np.concatenate([_say(lines[(k * 3 + j) % len(lines)], v, None, os.path.join(td, "b.wav"))
                                for j in range(4)])
            off = int(rng.integers(0, SR * 3))
            seg = a[: n - off]
            bab[off:off + len(seg)] += seg
        bab = np.convolve(bab, np.ones(8) / 8, mode="same")       # 少しこもらせる(離れた所の声)
        clips["murmur"] = 0.05 * bab / np.sqrt(np.mean(bab ** 2))
        for lang in ("zh", "ja"):
            v = _say(LYRICS[lang] + "，" + LYRICS[lang], VOICES[lang][0], 0.65, os.path.join(td, "l.wav"))
            v = _vibrato(v)
            v = np.pad(v, (SR, 0))[:n]
            v = np.pad(v, (0, n - len(v)))
            mus = _melody(n, 21)
            mus = mus / np.sqrt(np.mean(mus ** 2)) * np.sqrt(np.mean(v ** 2))
            for lvl in (1.0, 2.0):            # 曲の大きさ(歌声の何倍)
                y = v + lvl * mus
                clips[f"song_{lang}_m{lvl:g}"] = 0.1 * y / np.sqrt(np.mean(y ** 2))
    for name, y in clips.items():
        with wave.open(str(out / f"{name}.wav"), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(SR)
            w.writeframes((np.clip(y, -1, 1) * 32767).astype(np.int16).tobytes())
        lang = "ja" if "_ja_" in name else "zh"
        (out / f"{name}.json").write_text(json.dumps(
            {"lang": lang, "tgt": "ja" if lang == "zh" else "zh", "sents": [], "dur": len(y) / SR,
             "voice": 0, "pace": "-", "bgm": "fp"}, ensure_ascii=False), encoding="utf-8")
        print(name, round(len(y) / SR, 1), flush=True)


def report_fp(dirs: list) -> None:
    """字幕に出た単位の数と字数(正解は0)。"""
    names = sorted({p.stem for d in dirs for p in Path(d).glob("*.json")})
    print("| 音 | " + " | ".join(Path(d).name for d in dirs) + " |")
    print("|---|" + "---|" * len(dirs))
    for nm in names:
        cells = []
        for d in dirs:
            p = Path(d) / f"{nm}.json"
            if not p.exists():
                cells.append("—")
                continue
            us = json.loads(p.read_text())["units"]
            cells.append(f"{len(us)}単位 {sum(len(_norm(u['text'])) for u in us)}字")
        print(f"| {nm} | " + " | ".join(cells) + " |")


# ---------------------------------------------------------------- run

def _no_audio_device_ok() -> None:
    """remote(音の機器が無い)でも audio.py を読めるように。評価はマイクを開かないので sounddevice は使わない。"""
    try:
        import sounddevice  # noqa: F401
    except Exception:  # noqa: BLE001 - PortAudio が無い
        import types
        sys.modules["sounddevice"] = types.ModuleType("sounddevice")


def app_running() -> bool:
    """ユーザーのアプリ(-m tsuyaku)が動いているか。動いていたら評価を止める(CPU・LLM を取り合わない)。"""
    r = subprocess.run(["ps", "ax", "-o", "command"], capture_output=True, text=True)
    return any(" -m tsuyaku" in ln and "eval_" not in ln for ln in r.stdout.splitlines())


def replay(wav: Path, meta: dict, split_s: float, keep_bgm_text, ear, s,
           old_hold: bool = False) -> dict:
    """1本の WAV を音声の時計で流し、出た単位を返す。"""
    _no_audio_device_ok()
    from tsuyaku.audio import SpeechCapture
    from tsuyaku.segmenter import Endpointer, head_text, merge_segments, pick_cut_at
    head_from_interim = os.environ.get("EVAL_LONG_REHEAR", "") != "1"   # 1 = 前半を聞き直す(前の版)

    with wave.open(str(wav)) as w:
        x = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768
    x = np.concatenate([x, np.zeros(int(SR * 2.0), np.float32)])
    pair = (meta["lang"], meta["tgt"])
    ear.langs = pair
    ear._last_speech_t = -1e9
    clock = [0.0]
    ear.clock = lambda: clock[0]          # 話し声の続きの判定は音声の時計で
    orig_keep = ear.bgm_keep
    if not keep_bgm_text:      # 今までの規則: Speech 以外の出来事は文字ごと捨てる
        ear.bgm_keep = "off"
    elif isinstance(keep_bgm_text, str):
        ear.bgm_keep = keep_bgm_text
    try:
        q: "queue.Queue" = queue.Queue()
        cap = SpeechCapture(s.silero_vad, q, SR, s.min_speech_s, s.max_speech_s, s.min_silence_s,
                            vad_threshold=s.vad_threshold, fine=True)
        cap.interim_s = s.interim_s
        busy = [0.0]           # この時刻まで STT ワーカーは手がふさがっている
        stats = {"stt": 0, "interim": 0, "cuts": 0, "interim_frozen": 0}

        def cost(a) -> float:
            return STT_COST[0] + STT_COST[1] * len(a) / SR

        def hear(a, span=None, cont=False):
            stats["stt"] += 1
            busy[0] = max(busy[0], clock[0]) + cost(a)
            return ear.transcribe(a, cont=cont)

        ep = Endpointer(rejoin=hear, hold_open_s=s.hold_open_s if s.hold_open_s > 0 else 0.0)
        if old_hold:           # 今まで: 言い終わった短い断片も話し中は最大 3 秒延長
            ep.speak_cap_done_s = ep.speak_cap_s
        units = []
        last_interim = [""]

        def emit(us):
            for u in us:
                units.append({"text": u.text, "lang": u.lang, "lo": u.lo, "hi": u.hi,
                              "at": max(busy[0], clock[0]), "parts": u.parts})

        step = 512
        next_idle = 0.0
        for i in range(0, len(x), step):
            cap.on_chunk(x[i:i + step])
            clock[0] = (i + step) / SR
            if clock[0] < busy[0]:
                continue
            if not q.empty():
                pend = []
                while not q.empty():
                    pend.append(q.get_nowait())
                if len(pend) >= 3:
                    pend = [merge_segments(pend, SR)]
                for item in pend:
                    seg, lo, hi, pre = item[:4]
                    ct = cap.segmenter.cut_text(hi) if head_from_interim else None
                    cont = cap.segmenter.starts_at_cut(lo)
                    text, lang, prob = (ct[0], ct[1], 1.0) if ct else hear(seg, cont=cont)
                    if not text or not lang:
                        continue
                    cut = cap.segmenter.was_cut(hi)
                    emit(ep.push(seg, lo, hi, text, lang, now=max(busy[0], clock[0]), pre=pre,
                                 prob=prob, cut=cut))
                continue
            if clock[0] < next_idle:
                continue
            next_idle = clock[0] + 0.1
            emit(ep.tick(clock[0], speaking_since=cap.segmenter.speaking_since()))
            it = cap.take_interim()
            if it is None:
                continue
            stats["interim"] += 1
            busy[0] = clock[0] + cost(it[0])
            text, lang, _p, toks, stamps = ear.transcribe_stamped(
                it[0], cont=cap.segmenter.starts_at_cut(it[2]))
            if not text:
                stats["interim_frozen"] += 1     # 途中字幕が出ない(前の字幕のまま止まる)
                continue
            if split_s > 0 and lang in pair and toks:
                rc = pick_cut_at(toks, stamps, len(it[0]) / SR, split_s=split_s,
                                 comma_s=s.long_comma_s, breath_s=s.long_breath_s)
                if rc is not None and cap.segmenter.request_cut(it[2] + rc[0], text=head_text(toks, rc[1]),
                                                                 lang=lang):
                    stats["cuts"] += 1
        emit(ep.flush())
    finally:
        ear.bgm_keep = orig_keep
    return {"units": units, "stats": stats}


def run(clips: Path, out: Path, split_s: float, keep_bgm_text, only: str,
        old_hold: bool = False, guard: bool = True) -> None:
    from tsuyaku.config import Settings
    from tsuyaku.stt import SenseVoiceSTT
    s = Settings()
    if os.environ.get("EVAL_LONG_COMMA"):
        s.long_comma_s = float(os.environ["EVAL_LONG_COMMA"])
    m = ROOT / "models"
    s.silero_vad = str(m / "silero_vad.onnx")
    ear = SenseVoiceSTT(str(next(m.glob("sherpa-onnx-sense-voice-*"))), num_threads=2)
    out.mkdir(parents=True, exist_ok=True)
    for js in sorted(clips.glob("*.json")):
        if only and only not in js.stem:
            continue
        if guard and app_running():
            print("ユーザーのアプリが動いているので止めます", flush=True)
            sys.exit(3)
        meta = json.loads(js.read_text(encoding="utf-8"))
        t0 = time.time()
        r = replay(js.with_suffix(".wav"), meta, split_s, keep_bgm_text, ear, s, old_hold)
        r["meta"] = meta
        (out / js.name).write_text(json.dumps(r, ensure_ascii=False, indent=1), encoding="utf-8")
        d = [u["hi"] - u["lo"] for u in r["units"]]
        print(f"{js.stem}: {len(d)}単位 最長{max(d) if d else 0:.1f}秒 区切り{r['stats']['cuts']} "
              f"止まった途中字幕{r['stats']['interim_frozen']}/{r['stats']['interim']} "
              f"({time.time() - t0:.1f}秒)", flush=True)


# ---------------------------------------------------------------- report

def _norm(t: str) -> str:
    from tsuyaku.langdetect import norm
    return norm(t).replace(" ", "")


def cer(hyp: str, ref: str) -> float:
    a, b = _norm(hyp), _norm(ref)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb))
        prev = cur
    return prev[-1] / max(1, len(b))


def chrf(hyp: str, ref: str, n: int = 6, beta: float = 2.0) -> float:
    h, r = _norm(hyp), _norm(ref)
    ps, rs = [], []
    for k in range(1, n + 1):
        hc: dict = {}
        rc: dict = {}
        for i in range(len(h) - k + 1):
            hc[h[i:i + k]] = hc.get(h[i:i + k], 0) + 1
        for i in range(len(r) - k + 1):
            rc[r[i:i + k]] = rc.get(r[i:i + k], 0) + 1
        m = sum(min(c, rc.get(g, 0)) for g, c in hc.items())
        if hc:
            ps.append(m / sum(hc.values()))
        if rc:
            rs.append(m / sum(rc.values()))
    p = sum(ps) / len(ps) if ps else 0.0
    rr = sum(rs) / len(rs) if rs else 0.0
    return 0.0 if p + rr == 0 else 100 * (1 + beta ** 2) * p * rr / (beta ** 2 * p + rr)


def translate(units: list, src: str, tgt: str, url: str, model: str) -> None:
    """単位ごとにこのMacの LLM で訳す(アプリと同じ指示: build_messages、前の文脈つき)。"""
    import requests
    from tsuyaku.translate import build_messages
    hist: list = []
    for u in units:
        msgs = build_messages((src, tgt), "", hist, u["text"], src)
        t0 = time.time()
        r = requests.post(url.rstrip("/") + "/chat/completions",
                          json={"model": model, "messages": msgs, "temperature": 0.0, "max_tokens": 400},
                          timeout=120)
        r.raise_for_status()
        u["tr"] = r.json()["choices"][0]["message"]["content"].strip()
        u["tr_s"] = time.time() - t0
        hist.append({"lang": src, "orig": u["text"], "trans": u["tr"]})
        del hist[:-4]


def score(res: dict, tr_cache: "dict | None" = None) -> dict:
    import eval_pace as EP
    meta, units = res["meta"], [dict(u) for u in res["units"]]
    src, tgt = meta["lang"], meta["tgt"]
    sents = meta["sents"]
    ref_src = "".join(s["text"] for s in sents)
    ref_tr = "".join(s["ref"] for s in sents)
    ratio = len(_norm(ref_tr)) / max(1, len(_norm(ref_src)))
    for u in units:
        if "tr" not in u:
            u["tr"] = "あ" * max(1, round(len(_norm(u["text"])) * ratio))
        # 訳にかかる時間は見積もり(初字 + 字数 ÷ 速さ)で揃える。本当に測った秒(tr_s)は LLM の読み込みや
        # 他の作業との取り合いで方式ごとに大きくぶれた(同じ単位で 1.3 秒と 3.7 秒)ので、遅れの比較には使わない
        trs = TR_TTFT + len(_norm(u["tr"])) / TR_CPS
        u.update({"start": u["lo"], "end": u["hi"], "dur": max(0.5, u["hi"] - u["lo"] - 0.3),
                  "ready": u["at"] + trs})
    table = EP.speed_table(ROOT / "eval" / "pace" / "tts_speed.json")
    caps = {tuple(k.split("|")): v for k, v in json.loads(
        (ROOT / "eval" / "pace" / "caps.json").read_text()).items()} \
        if (ROOT / "eval" / "pace" / "caps.json").exists() else {}
    sim = EP.simulate(units, tgt, table, "auto+listen", caps)
    # 文ごと: 言い終わり → その終わりを含む単位の声が鳴り始めるまで
    lags, skipped = [], 0
    for s_ in sents:
        e = s_["end"]
        u = next((u for u in units if u["lo"] - 0.3 <= e - 0.15 <= u["hi"] + 0.3), None)
        if u is None or u.get("skipped") or "voice_at" not in u:
            skipped += 1
            continue
        lags.append(u["voice_at"] - e)
    lags.sort()
    pct = lambda p: round(lags[min(len(lags) - 1, int(math.ceil(p * len(lags))) - 1)], 1) if lags else None
    durs = sorted(u["hi"] - u["lo"] for u in units)
    hyp_src = "".join(u["text"] for u in units)
    out = {"units": len(units), "dur_max": round(durs[-1], 1) if durs else 0,
           "dur_p50": round(durs[len(durs) // 2], 1) if durs else 0,
           "cer": round(100 * cer(hyp_src, ref_src), 1),
           "lag50": pct(0.5), "lag90": pct(0.9), "sent_skipped": skipped, "n_sent": len(sents),
           "unit_lag50": sim["lag50"], "unit_skips": sim["skips"], "sp_avg": sim["sp_avg"]}
    if any("tr_s" in u for u in units):
        out["chrf"] = round(chrf("".join(u["tr"] for u in units), ref_tr), 1)
    return out


def report(a: Path, b: Path, url: str, model: str, save: "Path | None") -> None:
    rows = []
    agg: dict = {}
    for ja in sorted(a.glob("*.json")):
        jb = b / ja.name
        if not jb.exists():
            continue
        ra, rb = json.loads(ja.read_text()), json.loads(jb.read_text())
        if url:
            if app_running():
                print("ユーザーのアプリが動いているので訳をやめます", flush=True)
                sys.exit(3)
            for r, p in ((ra, ja), (rb, jb)):
                if not all("tr" in u for u in r["units"]):
                    translate(r["units"], r["meta"]["lang"], r["meta"]["tgt"], url, model)
                    p.write_text(json.dumps(r, ensure_ascii=False, indent=1), encoding="utf-8")
        sa, sb = score(ra), score(rb)
        rows.append((ja.stem, sa, sb))
        key = (ra["meta"]["lang"], ra["meta"]["bgm"])
        for tag, sc in (("A", sa), ("B", sb)):
            g = agg.setdefault(key + (tag,), {})
            for k, v in sc.items():
                if isinstance(v, (int, float)) and v is not None:
                    g.setdefault(k, []).append(v)
    lines = ["| 独白 | 方式 | 単位 | 最長/中央(秒) | CER | 文の声の遅れ p50/p90 | 飛ばした文 | chrF |",
             "|---|---|---|---|---|---|---|---|"]
    for name, sa, sb in rows:
        for tag, sc in (("A 今まで", sa), ("B 区切る", sb)):
            lines.append(f"| {name} | {tag} | {sc['units']} | {sc['dur_max']}/{sc['dur_p50']} | {sc['cer']}% | "
                         f"{sc['lag50']}/{sc['lag90']}秒 | {sc['sent_skipped']}/{sc['n_sent']} | "
                         f"{sc.get('chrf', '—')} |")
    lines += ["", "| 言語 | 音 | 方式 | 単位の最長(平均) | CER(平均) | 文の声の遅れ p50/p90(平均) | 飛ばした文(計) | chrF(平均) |",
              "|---|---|---|---|---|---|---|---|"]
    mean = lambda v: round(sum(v) / len(v), 1) if v else "—"
    for (lang, bgm, tag), g in sorted(agg.items()):
        lines.append(f"| {lang} | {bgm} | {tag} | {mean(g.get('dur_max', []))} | {mean(g.get('cer', []))}% | "
                     f"{mean(g.get('lag50', []))}/{mean(g.get('lag90', []))}秒 | "
                     f"{sum(g.get('sent_skipped', []))}/{sum(g.get('n_sent', []))} | {mean(g.get('chrf', []))} |")
    text = "\n".join(lines)
    print(text)
    if save:
        save.write_text(text + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    b = sp.add_parser("build")
    b.add_argument("--out", type=Path, default=ROOT / "eval" / "long" / "clips")
    bf = sp.add_parser("build-fp")
    bf.add_argument("--out", type=Path, default=ROOT / "eval" / "long" / "fp")
    rf = sp.add_parser("report-fp")
    rf.add_argument("dirs", nargs="+")
    r = sp.add_parser("run")
    r.add_argument("--clips", type=Path, default=ROOT / "eval" / "long" / "clips")
    r.add_argument("--out", type=Path, required=True)
    r.add_argument("--split", type=float, default=6.0)
    r.add_argument("--keep-bgm-text", default="head", help="off / loose(6字以上)/ head(頭が話し声なら)")
    r.add_argument("--only", default="")
    r.add_argument("--comma", type=float, default=0.0, help="読点で切る秒(0 = 設定の long_comma_s)")
    r.add_argument("--old", type=int, default=0, help="1 = 今まで(区切らない・BGM の文字を捨てる・3秒延長)")
    r.add_argument("--no-guard", action="store_true")
    p = sp.add_parser("report")
    p.add_argument("--a", type=Path, required=True)
    p.add_argument("--b", type=Path, required=True)
    p.add_argument("--translate", default="")
    p.add_argument("--model", default="google/gemma-3-4b")
    p.add_argument("--save", type=Path)
    a = ap.parse_args()
    if a.cmd == "build":
        build(a.out)
    elif a.cmd == "build-fp":
        build_fp(a.out)
    elif a.cmd == "report-fp":
        report_fp(a.dirs)
    elif a.cmd == "run":
        if a.comma:
            os.environ["EVAL_LONG_COMMA"] = str(a.comma)
        if a.old:
            run(a.clips, a.out, 0.0, False, a.only, old_hold=True, guard=not a.no_guard)
        else:
            kb = {"0": False, "1": "loose"}.get(a.keep_bgm_text, a.keep_bgm_text)
            run(a.clips, a.out, a.split, False if kb == "off" else kb, a.only, guard=not a.no_guard)
    else:
        report(a.a, a.b, a.translate, a.model, a.save)


if __name__ == "__main__":
    main()
