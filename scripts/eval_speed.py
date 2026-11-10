"""応答速度の実時間評価(11/08 speed): 作った会話の音声を、アプリそのもの(Bridge._conversation)に
マイクの代わりに実時間で流し、話し終わりから画面・声までの遅れを区間ごとに測る。

なぜ: ユーザー 11/08「Mac と リモートサーバー サーバーどちらを選択しても快適に使えるように。特に応答速度を早くして」。
eval_live.py(AMI 英会話)・eval_conv.py(FLEURS を音声の時計で模擬)は字幕までは測るが、声の鳴り始め・
リモートサーバーの冷えた状態(ゲートが llama-server を止めている・耳が止まっている)を測らない。

材料: eval/speed/sents.json(作った対訳。人の発言ではない)。日↔中・日↔英 の2人の会話を作る。
  声(入力): 日本語 = この Mac の VOICEVOX 2(アプリの読み上げ 13 とは別の声)、中国語 = Kokoro 50、
  英語 = Kokoro 11。ふつう(×1.0、相手の読み上げを聞いてから話す間)と速い(×1.3、間 0.7 秒)。
  部屋の小さな雑音(-50dBFS)を足す。

差し替えるのは eval_live.py と同じ3つ(マイク → WAV を実時間で流す FileCapture、画面 → 記録、記録の
置き場所)に加えて、読み上げの「鳴らす」所だけ(合成はアプリのまま、afplay の代わりに音の長さだけ待つ)。
読み上げ中も聞く(listen_while_speaking=always: 入力はファイルなので回り込まない)。

指標(話し終わり = 作った音声の声の終わり。すべてそこからの秒):
  原文   = 画面にその文の原文(6割以上)が初めて出た(途中字幕・続き待ち・確定のどれでも)
  訳初字 = その文の原文と一緒に訳の字が初めて出た(予想訳・確定訳の流し込み)
  確定   = その文の確定(utterance)が出そろった
  声     = その文の訳の読み上げが鳴り始めた
  品質: 確定の原文の CER(ja/zh)/WER(en)、確定訳の chrF(作った対訳が参照)
  起動: 「会話を始める」→ マイクを開ける(boot)まで

実行(リポジトリ直下):
  .venv/bin/python scripts/eval_speed.py build --out DIR
  .venv/bin/python scripts/eval_speed.py run --clip DIR/ja-zh_normal.json --engine mac --out RES [--tag x]
  .venv/bin/python scripts/eval_speed.py report RES/*/summary.json
生成物(音声・記録)は git に入れない。eval/speed/ には md と sents.json だけ。
"""
from __future__ import annotations

import argparse
import difflib
import io
import json
import math
import os
import random
import re
import subprocess
import sys
import tempfile
import threading
import time
import unicodedata
import wave
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SR = 16000
SENTS = ROOT / "eval" / "speed" / "sents.json"
VOICE_IN = {"ja": "voicevox:2", "zh": "kokoro:50", "en": "kokoro:11"}


def _pct(xs, q):
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return None
    return round(xs[min(len(xs) - 1, max(0, int(math.ceil(q * len(xs))) - 1))], 2)


def read_wav(path) -> np.ndarray:
    with wave.open(str(path), "rb") as w:
        assert w.getframerate() == SR and w.getsampwidth() == 2
        x = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(np.float32) / 32768.0
        if w.getnchannels() > 1:
            x = x.reshape(-1, w.getnchannels()).mean(axis=1)
    return x


def write_wav(path, x) -> None:
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes((np.clip(x, -1, 1) * 32767).astype("<i2").tobytes())


def app_running() -> bool:
    """ユーザーのアプリ(python -m tsuyaku)が動いているか。"""
    ps = subprocess.run(["ps", "ax", "-o", "command"], capture_output=True, text=True).stdout
    # シェルの包み(他の作業の grep の文字列)は数えない: python の実行ファイルで -m tsuyaku の時だけ
    return any(re.match(r"^\S*[Pp]ython[\d.]*\s(.*\s)?-m tsuyaku(\s|$)", ln) for ln in ps.splitlines())


# ====================================================================== build

def _to16k(wav_bytes: bytes) -> np.ndarray:
    """合成の WAV(24kHz など)→ 16kHz mono float(macOS の afconvert で変換)。"""
    with tempfile.TemporaryDirectory() as d:
        a, b = Path(d) / "a.wav", Path(d) / "b.wav"
        a.write_bytes(wav_bytes)
        subprocess.run(["afconvert", "-f", "WAVE", "-d", "LEI16@16000", "-c", "1", str(a), str(b)],
                       check=True, capture_output=True)
        return read_wav(b)


def _trim(x: np.ndarray, th: float = 0.01) -> np.ndarray:
    idx = np.where(np.abs(x) > th)[0]
    if len(idx) == 0:
        return x
    return x[max(0, idx[0] - 160): idx[-1] + 160]


def build(out: Path, seed: int = 1) -> None:
    sys.path.insert(0, str(ROOT / "src"))
    from tsuyaku.config import load_settings
    from tsuyaku.tts import TTS
    s = load_settings()
    tts = TTS(s.tts_dir_kokoro, "", s.tts_sids, "say", voices=dict(VOICE_IN),
              speeds={}, voicevox_url=s.voicevox_url)
    sents = json.loads(SENTS.read_text(encoding="utf-8"))["sents"]
    out.mkdir(parents=True, exist_ok=True)
    cache = out / "tts"
    cache.mkdir(exist_ok=True)

    def voice(text: str, lang: str, speed: float) -> np.ndarray:
        import hashlib
        key = f"{lang}_{speed:.2f}_{hashlib.md5((text + VOICE_IN[lang]).encode()).hexdigest()[:10]}.wav"
        p = cache / key
        if not p.exists():
            wav = tts._synth(text, lang, speed)
            assert wav, f"合成できない: {lang} {text}"
            write_wav(p, _trim(_to16k(wav)))
        return read_wav(p)

    for other in ("zh", "en"):
        for cond, speed in (("normal", 1.0), ("fast", 1.3)):
            rng = random.Random(f"{seed}-{other}-{cond}")
            order = list(sents)
            rng.shuffle(order)
            pieces: list[np.ndarray] = [np.zeros(int(1.5 * SR), np.float32)]
            t = 1.5
            utts = []
            for k, sent in enumerate(order):
                lang = "ja" if k % 2 == 0 else other
                tgt = other if lang == "ja" else "ja"
                x = voice(sent[lang], lang, speed) * (10 ** (rng.uniform(-3, 0) / 20))
                dur = len(x) / SR
                utts.append({"i": k, "sid": sent["id"], "kind": sent["kind"], "lang": lang, "tgt": tgt,
                             "text": sent[lang], "ref": sent[tgt], "s": round(t, 3),
                             "e": round(t + dur, 3)})
                pieces.append(x)
                t += dur
                if cond == "normal":
                    # 相手の訳の読み上げを聞いてから話す(訳の声 ≒ 原文の長さ + 1秒強)
                    gap = 1.5 + 1.1 * dur + rng.uniform(0, 0.8)
                else:
                    gap = rng.uniform(0.6, 0.9)   # 速い掛け合い(読み上げを待たない)
                pieces.append(np.zeros(int(gap * SR), np.float32))
                t += gap
            x = np.concatenate(pieces)
            x = x + np.random.default_rng(seed).standard_normal(len(x)).astype(np.float32) * 10 ** (-50 / 20)
            name = f"ja-{other}_{cond}"
            write_wav(out / f"{name}.wav", x)
            meta = {"pair": ["ja", other], "cond": cond, "speed": speed, "wav": f"{name}.wav",
                    "dur": round(len(x) / SR, 2), "utts": utts}
            (out / f"{name}.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1),
                                              encoding="utf-8")
            print(f"[build] {name}: {len(utts)}発話 {len(x) / SR:.0f}秒", flush=True)


# ====================================================================== run

def gate_status() -> dict:
    """リモートサーバーの 翻訳ゲートウェイ の状態(llama-server を起こさない /gate/status)と耳の応答。"""
    import requests
    st: dict = {}
    try:
        st["gate"] = requests.get("http://127.0.0.1:12345/gate/status", timeout=3).json()
    except Exception as e:  # noqa: BLE001
        st["gate"] = {"error": type(e).__name__}
    try:
        st["ear"] = requests.get("http://127.0.0.1:12346/health", timeout=2).status_code == 200
    except Exception:  # noqa: BLE001
        st["ear"] = False
    return st


def zoom_running() -> bool:
    """Zoom が動いている(ユーザーが会議中かもしれない): Mac の重い測定はしない(11/08 リーダー)。"""
    ps = subprocess.run(["ps", "ax", "-o", "command"], capture_output=True, text=True).stdout
    # 行の頭がアプリ本体のときだけ(ほかの作業のシェルの文字列に同じ語が入っていても数えない)
    return any(ln.startswith("/Applications/zoom.us.app/Contents/MacOS/") for ln in ps.splitlines())


def others_busy(cpu: float = 30.0, lmstudio: bool = True) -> list[str]:
    """ほかの作業の重い評価(scripts/eval_*.py が CPU を cpu% 以上)。遅れの測定が狂うので空くのを待つ。"""
    ps = subprocess.run(["ps", "ax", "-o", "pid=,%cpu=,command="], capture_output=True, text=True).stdout
    me = os.getpid()
    out = []
    for ln in ps.splitlines():
        parts = ln.split(None, 2)
        if len(parts) < 3 or int(parts[0]) == me:
            continue
        if re.match(r"^\S*[Pp]ython", parts[2]) and "scripts/eval_" in parts[2] and (
                float(parts[1]) >= cpu or "scripts/eval_speed.py" in parts[2]):   # 自分の別の測定は CPU によらず
            out.append(parts[2][parts[2].index("scripts/eval_"):][:80])
    # このMacの LM Studio(同時に1件)を使っているほかの作業(CPU はほぼ使わないので上では見えない)。
    # リモートサーバーの測定でも待つ: LM Studio が訳している間は、メモリの帯域を取り合って Kokoro の合成が 4〜5倍遅い
    # (15字の中国語 0.6〜0.9秒 → 3.9秒、11/08 実測)
    try:
        if not lmstudio:
            return out
        ls = subprocess.run(["lsof", "-nP", "-iTCP:1234", "-sTCP:ESTABLISHED", "-t"],
                            capture_output=True, text=True, timeout=5).stdout.split()
        for pid in set(ls):
            if int(pid) == me:
                continue
            cmd = subprocess.run(["ps", "-o", "command=", "-p", pid], capture_output=True, text=True).stdout
            if "LM Studio" not in cmd and ".lmstudio" not in cmd and cmd.strip():
                out.append("LM Studio を使用中: " + cmd.strip()[cmd.find("scripts/"):][:60])
    except Exception:  # noqa: BLE001
        pass
    return out


def run(clip_json: Path, out_root: Path, engine: str, src: Path | None, sets: list[str],
        tag: str = "", tail_s: float = 10.0, wait_quiet: float = 1800.0, preload: bool = False) -> Path:
    if app_running():
        raise SystemExit("ユーザーのアプリが動いているので流さない")
    t_w = time.time()
    while time.time() - t_w < wait_quiet:
        busy = others_busy(lmstudio=True) + others_busy(lmstudio=True)   # 1回だけの瞬間値で決めない
        if zoom_running():
            busy.append("Zoom")
        if engine == "remote":   # リモートサーバーの Qwen をほかの作業が使っている(混むと訳の遅れが揺れる)
            # 短い依頼を続けて流す判定は、1回だけ見ると間に当たって空きに見える(11/08 夜): 10秒の間に5回見る
            for _k in range(5):
                act = (gate_status().get("gate") or {}).get("active_requests")
                if act:
                    busy.append(f"リモートサーバーの 翻訳ゲートウェイ を使用中 {act}件")
                    break
                time.sleep(2)
        if os.getloadavg()[0] > 6.0:   # Mac 全体の負荷(8コア)
            busy.append(f"負荷 {os.getloadavg()[0]:.1f}")
        if not busy:
            break
        print(f"[run] ほかの評価が動いている({busy[0]}…): 20秒待つ", flush=True)
        time.sleep(20)
    else:
        raise SystemExit(f"ほかの作業が {wait_quiet:.0f} 秒たっても終わらないので流さない")
    busy_at_start = others_busy(lmstudio=True)
    sys.path.insert(0, str((src or ROOT / "src").resolve()))
    import tsuyaku.__main__ as app
    from tsuyaku import audio as audio_mod

    meta = json.loads(clip_json.read_text(encoding="utf-8"))
    x = read_wav(clip_json.parent / meta["wav"])
    out = out_root / f"{clip_json.stem}_{engine}{('_' + tag) if tag else ''}"
    out.mkdir(parents=True, exist_ok=True)
    rec: list = []
    lock = threading.Lock()

    def note(kind, **kw):
        with lock:
            rec.append((time.time(), kind, kw))

    pre = gate_status() if engine != "mac" else {}
    feed_done = threading.Event()
    clock: dict = {}
    stop_feed = threading.Event()

    class FileCapture(audio_mod.SpeechCapture):
        def start(self):
            clock["boot_done"] = time.time()
            self._thread = threading.Thread(target=self._run_file, daemon=True)
            self._thread.start()

        def _run_file(self):
            rng = np.random.default_rng(0)
            n = len(x)
            tail = int(tail_s * SR)
            t0 = time.time()
            clock["t0"] = t0
            i = 0
            while i < n + tail and not self._stop.is_set() and not stop_feed.is_set():
                d = t0 + (i + 512) / SR - time.time()
                if d > 0:
                    time.sleep(d)
                if i < n:
                    ch = x[i:i + 512]
                    if len(ch) < 512:
                        ch = np.concatenate([ch, np.zeros(512 - len(ch), np.float32)])
                else:
                    ch = (rng.standard_normal(512) * 3e-4).astype(np.float32)
                    if not feed_done.is_set():
                        feed_done.set()
                self.on_chunk(ch)
                i += 512
            feed_done.set()

    class StubTTS(app.TTS):
        """合成はアプリのまま。鳴らす代わりに音の長さだけ待つ(評価で部屋に声を出さない)。"""

        def _synth(self, text, lang, speed=None):
            t = time.time()
            r = super()._synth(text, lang, speed)
            note("synth", dt=round(time.time() - t, 3), n=len(text), lang=lang)
            return r

        def _play_wav(self, data):
            with wave.open(io.BytesIO(data)) as w:
                dur = w.getnframes() / float(w.getframerate())
            self._started()
            note("play", dur=round(dur, 2))
            t = time.time()
            while time.time() - t < dur and not self._stop.is_set():
                time.sleep(0.02)
            self.last_play_s = getattr(self, "last_play_s", 0.0) + (time.time() - t)
            return True

        def _say_macos(self, text, name, speed=1.0):
            return self._play_wav(_silence_wav(len(text) / 6.0))

        def prepare(self, text, lang, speed=None, *a, **k):
            # 予想訳(先読み)の作り置きか: 新しい版は guess=True、古い版は Speculator の担当から呼ばれる
            import inspect
            guess = bool(k.get("guess")) or any(f.function == "_work" for f in inspect.stack()[1:6])
            note("tts_prepare", text=text, lang=lang, speed=speed, guess=guess, early=bool(k.get("early")))
            return super().prepare(text, lang, speed, *a, **k)

        def say(self, text, lang, speed=None):
            note("tts_say", text=text, lang=lang, speed=speed)
            return super().say(text, lang, speed)

    class UIRec:
        def put(self, evt):
            note("ui", evt=list(evt))

    app.SpeechCapture = FileCapture
    app.TTS = StubTTS
    b = app.Bridge()
    b.s.engine = engine
    b.s.listen_while_speaking = "always"
    b.s.log_dir = str(out / "applog")
    for kv in sets:
        k, v = kv.split("=", 1)
        cur = getattr(b.s, k)
        setattr(b.s, k, type(cur)(json.loads(v)) if not isinstance(cur, str) else v)
    b.ui = UIRec()
    busy_seen: list = []

    def watch_busy():   # 流している間、ほかの重い評価が動いたかを10秒ごとに記録(結果の信頼性)
        while not feed_done.is_set():
            o = others_busy(lmstudio=True)
            if o:
                busy_seen.append(o[0])
            feed_done.wait(10)
    threading.Thread(target=watch_busy, daemon=True).start()
    logf = (out / "app.log").open("w", encoding="utf-8")
    old = sys.stdout
    sys.stdout = _Tee(logf)
    pair = "-".join(meta["pair"])
    if preload and hasattr(b, "_preload"):
        # アプリを開いた時の先読み(11/08 speed)。起動画面にいる間に済む(ここでは終わるまで待ってから始める)
        t_p = time.time()
        b._preload()
        print(f"[run] 先読み {time.time() - t_p:.1f}秒", flush=True)
    th = threading.Thread(target=b._conversation, args=(pair, "", "interpret"), daemon=True)
    t_start = time.time()
    th.start()
    aborted = ""
    try:
        while not feed_done.wait(1.0):
            if not th.is_alive():
                raise RuntimeError("会話が始まらなかった(app.log)")
            if app_running() or zoom_running():
                aborted = "ユーザーのアプリか Zoom が起きたので止めた"
                stop_feed.set()
                break
        if not aborted:
            time.sleep(3.0)
            deadline = time.time() + 25
            while time.time() < deadline:
                with lock:
                    last = max((t for t, k, _ in rec if k in ("ui", "play")), default=0)
                if time.time() - last > 5.0 and b.tts_q.empty():
                    break
                time.sleep(0.5)
    finally:
        b.running.clear()
        th.join(timeout=15)
        sys.stdout = old
        logf.close()
    t0 = clock.get("t0", t_start)
    events = [{"t": round(t - t0, 3), "kind": k, **kw} for t, k, kw in rec]
    run_meta = {"clip": str(clip_json), "engine": engine, "engine_used": b.engine_used,
                "model": b.model_used, "sets": sets, "tag": tag, "src": str(src or ROOT / "src"),
                "git": _git(src or ROOT), "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                "boot_s": round(clock.get("boot_done", t_start) - t_start, 2), "preload": preload,
                "pre": pre, "aborted": aborted, "busy_at_start": busy_at_start,
                "busy_during": busy_seen,
                "spec": (None if b.spec is None else {"hits": b.spec.hits, "misses": b.spec.misses})}
    (out / "events.json").write_text(json.dumps({"meta": run_meta, "events": events},
                                                ensure_ascii=False, indent=0), encoding="utf-8")
    if aborted:
        raise SystemExit(aborted)
    return evaluate(clip_json, out)


def _silence_wav(sec: float) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(b"\0\0" * int(sec * SR))
    return buf.getvalue()


class _Tee:
    def __init__(self, f):
        self.f = f
        self.lock = threading.Lock()

    def write(self, s):
        with self.lock:
            self.f.write(s)
        return len(s)

    def flush(self):
        self.f.flush()


def _git(p: Path) -> str:
    try:
        r = subprocess.run(["git", "-C", str(p), "rev-parse", "--short", "HEAD"],
                           capture_output=True, text=True, timeout=5)
        d = subprocess.run(["git", "-C", str(p), "status", "--porcelain", "src"],
                           capture_output=True, text=True, timeout=5).stdout.strip()
        return r.stdout.strip() + ("+dirty" if d else "")
    except Exception:  # noqa: BLE001
        return "?"


# ====================================================================== 指標

def _strip(text: str) -> str:
    t = unicodedata.normalize("NFKC", text or "").lower()
    return "".join(c for c in t if unicodedata.category(c)[0] not in "PSZC")


def _cover(part: str, whole: str) -> float:
    """part の字のうち whole に順に見つかる割合(part が whole の一部か)。"""
    a, b = _strip(part), _strip(whole)
    if not a or not b:
        return 0.0
    m = sum(k.size for k in difflib.SequenceMatcher(None, a, b, autojunk=False).get_matching_blocks())
    return m / len(a)


def _recall(part: str, whole: str) -> float:
    """whole の字のうち part に出ている割合。"""
    a, b = _strip(part), _strip(whole)
    if not a or not b:
        return 0.0
    m = sum(k.size for k in difflib.SequenceMatcher(None, a, b, autojunk=False).get_matching_blocks())
    return m / len(b)


def _ec():
    sys.path.insert(0, str(ROOT / "scripts"))
    sys.path.insert(0, str(ROOT / "src"))
    import eval_conv
    return eval_conv


def evaluate(clip_json: Path, out: Path) -> Path:
    ec = _ec()
    meta = json.loads(clip_json.read_text(encoding="utf-8"))
    run = json.loads((out / "events.json").read_text(encoding="utf-8"))
    ev = run["events"]
    uis = [(e["t"], e["evt"]) for e in ev if e["kind"] == "ui"]
    shows = []          # (t, orig, trans, kind)
    utts = []           # (t, payload)
    voice = {}          # id -> t(鳴り始め)
    skipped = set()
    for t, evt in uis:
        if evt[0] == "partial" and len(evt) >= 4:
            shows.append((t, evt[1] or "", evt[2] or "", "partial"))
        elif evt[0] == "utterance":
            d = evt[1]
            shows.append((t, d.get("orig", ""), d.get("trans", ""), "utt"))
            utts.append((t, d))
        elif evt[0] == "voice":
            d = evt[1]
            if d.get("skipped"):
                skipped.add(d.get("id"))
            elif d.get("id") is not None and d["id"] not in voice:
                voice[d["id"]] = t
    gold = meta["utts"]
    rows = []
    used_utt: set = set()
    hyp_stats: dict = {}
    asr: dict = {}
    for gi, g in enumerate(gold):
        nxt = gold[gi + 1]["s"] if gi + 1 < len(gold) else g["e"] + 30
        lo, hi = g["s"], max(nxt, g["e"]) + 25
        src_t = tr_t = None
        for t, o, tr, _k in shows:
            if t < lo or t > hi:
                continue
            if src_t is None and _recall(o, g["text"]) >= 0.6 and _cover(o, g["text"]) >= 0.5:
                src_t = t
            if (tr_t is None and tr and tr not in ("…", "...") and _cover(o, g["text"]) >= 0.5
                    and _recall(o, g["text"]) >= 0.3):
                tr_t = t
        mine = []
        got = 0.0
        for k, (t, d) in enumerate(utts):
            if k in used_utt or t < lo or t > hi:
                continue
            if _cover(d.get("orig", ""), g["text"]) >= 0.6:
                mine.append((k, t, d))
                used_utt.add(k)
                got = _recall("".join(x[2].get("orig", "") for x in mine), g["text"])
                if got >= 0.85:
                    break
        fin_t = mine[-1][1] if mine else None
        vids = [d.get("id") for _k, _t, d in mine]
        v_t = min((voice[i] for i in vids if i in voice), default=None)
        hyp_o = ("" if g["lang"] in ("ja", "zh") else " ").join(d.get("orig", "") for _k, _t, d in mine)
        hyp_t = ("" if g["tgt"] in ("ja", "zh") else " ").join(d.get("trans", "") for _k, _t, d in mine)
        st = ec.chrf_stats(hyp_t, g["ref"])
        dir_ = f"{g['lang']}>{g['tgt']}"
        hyp_stats[dir_] = ec.add_stats(hyp_stats.get(dir_), st)
        rt = ec.asr_tokens(g["text"], g["lang"])
        ht = ec.asr_tokens(hyp_o, g["lang"])
        a = asr.setdefault(g["lang"], [0, 0])
        a[0] += ec.edit_distance(ht, rt)
        a[1] += len(rt)
        e = g["e"]

        def rel(t):
            return None if t is None else round(t - e, 2)
        rows.append({"i": g["i"], "kind": g["kind"], "dir": dir_, "dur": round(g["e"] - g["s"], 2),
                     "src": rel(src_t), "tr": rel(tr_t), "fin": rel(fin_t), "voice": rel(v_t),
                     "n_utt": len(mine), "skipped": any(i in skipped for i in vids),
                     "orig": hyp_o, "trans": hyp_t, "chrf": round(ec.chrf(hyp_t, g["ref"]), 1)})
    sm = summarize(rows)
    sm["chrf"] = {d: round(ec.chrf_from_stats(s), 1) for d, s in hyp_stats.items()}
    allst = None
    for s in hyp_stats.values():
        allst = ec.add_stats(allst, s)
    sm["chrf"]["all"] = round(ec.chrf_from_stats(allst), 1) if allst else 0.0
    sm["asr_err"] = {l: round(100.0 * a[0] / max(1, a[1]), 1) for l, a in asr.items()}
    sm["meta"] = run["meta"]
    sm["clip"] = clip_json.stem
    boots = [(e["t"], e["evt"][1].get("step")) for e in ev if e["kind"] == "ui" and e["evt"][0] == "boot"]
    sm["boot_steps"] = [[st, round(t - boots[0][0], 2)] for t, st in boots] if boots else []
    synth = [e["dt"] for e in ev if e["kind"] == "synth"]
    sm["synth"] = {"n": len(synth), "p50": _pct(synth, 0.5), "p90": _pct(synth, 0.9)}
    (out / "rows.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    (out / "summary.json").write_text(json.dumps(sm, ensure_ascii=False, indent=1), encoding="utf-8")
    print(fmt(sm), flush=True)
    return out / "summary.json"


def voice_replay(res: Path, src: Path | None, out: Path | None = None) -> dict:
    """声の側だけを同じ条件でくらべる(11/08 speed)。実時間の評価で記録した読み上げの呼び出し
    (予想訳の作り置き・確定訳の作り置き = 行列に入れた時刻・読む)を同じ時刻で、指定の版の TTS に
    もう一度流す(鳴らす代わりに音の長さだけ待つ)。耳・訳・回線の揺れが入らない。
    指標: 行列に入れてから鳴り始めるまで(前の読み上げの終わり待ちを含む)と、読む番が来てから鳴り始めるまで。"""
    sys.path.insert(0, str((src or ROOT / "src").resolve()))
    from tsuyaku.config import load_settings
    from tsuyaku import tts as tts_mod
    import queue as _q
    ev = json.loads((res / "events.json").read_text(encoding="utf-8"))["events"]
    calls = [e for e in ev if e["kind"] in ("tts_prepare",)]
    starts: dict = {}

    class Stub(tts_mod.TTS):
        def _play_wav(self, data):
            with wave.open(io.BytesIO(data)) as w:
                dur = w.getnframes() / float(w.getframerate())
            self._started()
            t = time.time()
            while time.time() - t < dur and not self._stop.is_set():
                time.sleep(0.01)
            self.last_play_s = getattr(self, "last_play_s", 0.0) + (time.time() - t)
            return True

    s = load_settings()
    t = Stub(s.tts_dir_kokoro, s.tts_dir_ko, s.tts_sids, s.tts_ja_engine, voices=s.tts_voice,
             speeds=s.tts_speed, voicevox_url=s.voicevox_url)
    for lang in ("ja", "en", "zh"):
        t._synth("はい。" if lang == "ja" else "OK.", lang)
    q: "_q.Queue" = _q.Queue()
    lat, wait = [], []
    done = threading.Event()

    def worker():
        while not (done.is_set() and q.empty()):
            try:
                k, text, lang, sp, t_enq = q.get(timeout=0.1)
            except _q.Empty:
                continue
            t_deq = time.time()

            def on_start(_k=k, _e=t_enq, _d=t_deq):
                lat.append(time.time() - _e)
                wait.append(time.time() - _d)
            t.on_start = on_start
            t.say(text, lang, sp)
    th = threading.Thread(target=worker, daemon=True)
    th.start()
    t0 = time.time()
    base = calls[0]["t"] if calls else 0.0
    for k, c in enumerate(calls):
        d = t0 + (c["t"] - base) - time.time()
        if d > 0:
            time.sleep(d)
        kw = {}
        names = tts_mod.TTS.prepare.__code__.co_varnames
        if c.get("early"):   # 訳の流し込み中の先作り: その仕組みの無い版では流さない
            if "early" in names:
                t.prepare(c["text"], c["lang"], c.get("speed"), early=True)
            continue
        if c.get("guess") and "guess" in names:
            kw["guess"] = True
        t.prepare(c["text"], c["lang"], c.get("speed"), **kw)
        if not c.get("guess"):
            q.put((k, c["text"], c["lang"], c.get("speed"), time.time()))
    done.set()
    th.join(timeout=120)
    r = {"n": len(lat), "enq_to_start": {"p50": _pct(lat, 0.5), "p90": _pct(lat, 0.9)},
         "turn_to_start": {"p50": _pct(wait, 0.5), "p90": _pct(wait, 0.9)},
         "guess_prepares": sum(1 for c in calls if c.get("guess")), "src": str(src or ROOT / "src")}
    print(json.dumps(r, ensure_ascii=False), flush=True)
    return r


KEYS = ("src", "tr", "fin", "voice")
NAMES = {"src": "原文", "tr": "訳初字", "fin": "訳確定", "voice": "声"}


def summarize(rows: list[dict]) -> dict:
    sm: dict = {"n": len(rows)}
    for k in KEYS:
        xs = [r[k] for r in rows if r[k] is not None]
        sm[k] = {"p50": _pct(xs, 0.5), "p90": _pct(xs, 0.9), "n": len(xs)}
    # 確定 → 声(読み上げを作って鳴らすまで。耳・訳の揺れに左右されない声の側だけの遅れ)
    gap = [r["voice"] - r["fin"] for r in rows
           if r["voice"] is not None and r["fin"] is not None and r["voice"] >= r["fin"] and not r["skipped"]]
    sm["gap"] = {"p50": _pct(gap, 0.5), "p90": _pct(gap, 0.9), "n": len(gap)}
    sm["missing_fin"] = sum(1 for r in rows if r["fin"] is None)
    sm["skipped"] = sum(1 for r in rows if r["skipped"])
    first = rows[0] if rows else {}
    sm["first"] = {k: first.get(k) for k in KEYS}
    for kind in ("short", "mid", "long"):
        rr = [r for r in rows if r["kind"] == kind]
        sm[kind] = {k: _pct([r[k] for r in rr], 0.5) for k in KEYS}
    return sm


def fmt(sm: dict) -> str:
    m = sm.get("meta", {})
    head = (f"{sm.get('clip')} {m.get('engine_used') or m.get('engine')} {m.get('model', '')} "
            f"boot {m.get('boot_s')}s tag={m.get('tag', '')}")
    cells = " | ".join(f"{NAMES[k]} {sm[k]['p50']}/{sm[k]['p90']}" for k in KEYS)
    if sm.get("gap"):
        cells += f" | 確定→声 {sm['gap']['p50']}/{sm['gap']['p90']}"
    return (f"{head}\n  {cells} | 欠け {sm['missing_fin']} 飛ばし {sm['skipped']} | "
            f"chrF {sm.get('chrf')} 誤り {sm.get('asr_err')} | 1文目 {sm['first']} | 合成 {sm.get('synth')}")


def report(paths: list[Path]) -> str:
    lines = ["| 音声 | 処理場所 | 版 | 起動 | 原文 p50/p90 | 訳初字 | 訳確定 | 声 | 確定→声 | 1文目 声 | 欠け/飛ばし | chrF | CER/WER |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for p in paths:
        sm = json.loads(Path(p).read_text(encoding="utf-8"))
        m = sm["meta"]
        cell = lambda k: f"{sm[k]['p50']}/{sm[k]['p90']}"  # noqa: E731
        lines.append(f"| {sm['clip']} | {m.get('engine_used') or m['engine']} | {m.get('tag') or m.get('git')} | "
                     f"{m.get('boot_s')} | {cell('src')} | {cell('tr')} | {cell('fin')} | {cell('voice')} | "
                     f"{cell('gap') if sm.get('gap') else '-'} | "
                     f"{sm['first'].get('voice')} | {sm['missing_fin']}/{sm['skipped']} | {sm['chrf'].get('all')} | "
                     f"{' '.join(f'{k}{v}' for k, v in sm['asr_err'].items())} |")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--out", type=Path, required=True)
    b.add_argument("--seed", type=int, default=1)
    r = sub.add_parser("run")
    r.add_argument("--clip", type=Path, required=True)
    r.add_argument("--out", type=Path, required=True)
    r.add_argument("--engine", choices=("mac", "remote"), required=True)
    r.add_argument("--src", type=Path)
    r.add_argument("--set", action="append", default=[])
    r.add_argument("--tag", default="")
    r.add_argument("--preload", action="store_true", help="アプリを開いた時の先読みを済ませてから始める")
    r.add_argument("--wait", type=float, default=1800.0, help="ほかの作業が終わるのを待つ上限(秒)。過ぎたら流さずに止める")
    e = sub.add_parser("eval")
    e.add_argument("--clip", type=Path, required=True)
    e.add_argument("--res", type=Path, required=True)
    v = sub.add_parser("voice")
    v.add_argument("--res", type=Path, required=True)
    v.add_argument("--src", type=Path)
    p = sub.add_parser("report")
    p.add_argument("paths", nargs="+", type=Path)
    a = ap.parse_args()
    if a.cmd == "build":
        build(a.out, a.seed)
    elif a.cmd == "run":
        run(a.clip, a.out, a.engine, a.src, a.set, a.tag, preload=a.preload, wait_quiet=a.wait)
    elif a.cmd == "eval":
        evaluate(a.clip, a.res)
    elif a.cmd == "voice":
        voice_replay(a.res, a.src)
    else:
        print(report(a.paths))


if __name__ == "__main__":
    main()
