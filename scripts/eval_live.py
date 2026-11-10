"""実時間の英会話ベンチマーク: 本物の会議の音声を、アプリそのもの(Bridge._conversation)に
マイクの代わりに実時間で流し、画面に出た時刻を測る(en→ja の通訳として)。

なぜ: 11/06 英会話の動画を流して「スピードについていけていない」。eval_conv.py は
FLEURS の読み上げ文(1文ずつ・間あり)を音声の時計で模擬するので、途切れずに話し続ける
本物の会話の遅れ(区間が12秒まで閉じない・続き待ちの聞き直し・途中字幕)が測れない。

やり方: アプリの _conversation をそのまま動かし、差し替えるのは3つだけ
  - SpeechCapture → FileCapture: マイクの代わりに WAV を 512 サンプルずつ実時間で流す
    (コールバックの中身はアプリと同じ: AutoGain → VAD → StreamSegmenter → 途中字幕)
  - 画面(ui キュー) → 時刻つきで全部記録
  - 会話の記録(logs/)は結果フォルダへ(本物の記録と混ぜない)
STT/翻訳/先読み/途中字幕/Endpointer はアプリのコードと設定(config.json)のまま。
リモートサーバーの耳・翻訳は SSH 転送(12346/12345)。家の上り回線を使うので、アプリを使っている
最中は走らせない(--no-wait を付けない限り、アプリが動いていて記録が直近5分に
更新されていたら待つ)。

材料: AMI 会議コーパス(CC BY 4.0)。手作業の書き起こし(単語ごとの時刻・話者)が正解。
  build: 会議の一部を切り出して clip WAV と正解 JSON を作る。参照訳 = 正解の書き起こしを
         同じ翻訳モデル(remote)で訳したもの(同じ話者の続きをまとめて、アプリと同じ文脈で)。
  run:   1クリップを実時間で流す(長さぶん時間がかかる)。
  eval:  記録から指標を計算し直す。
  ab:    2つの結果フォルダを比べる。

指標(画面に出た時刻は、表示された原文を正解の単語列に並べて(2語以上の一致)決める):
  語の遅れ = 正解の各語が「話し終わってから」初めて画面に出るまで(中央/p90)
             src = 原文(途中字幕・続き待ちの原文・予想・確定のどれでも)
             tr  = 訳(予想訳・確定訳の流し込み・確定)  fin = 確定字幕
  長い発話(同じ人が切れ目1秒未満で6秒以上): 話し始めから最初に原文/訳が出るまで
  WER(確定字幕の原文 vs 正解)、取りこぼし(確定に1度も出なかった語の割合)
  区切り: 単位の長さ(中央/p90/最大・8秒超の数)、3語以上の文が2単位以上に割れた割合
  翻訳: 30秒ごとの窓で 確定訳 vs 参照訳 の chrF(区切りが違っても比べられる)

実行(リポジトリ直下。AMI は信頼しないデータなので -I):
  .venv/bin/python -I scripts/eval_live.py build --zip ami_manual.zip --wav DIR/ES2004a.Mix-Headset.wav \\
      --start 300 --dur 180 --out eval/english/clips
  .venv/bin/python -I scripts/eval_live.py run --clip eval/english/clips/ES2004a_Mix-Headset_300.json \\
      --out eval/english/results/base [--src 別の版の src] [--set interim_s=0]
  .venv/bin/python -I scripts/eval_live.py ab --a eval/english/results/base --b eval/english/results/x
"""
from __future__ import annotations

import argparse
import difflib
import glob
import json
import math
import os
import re
import sys
import threading
import time
import wave
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SR = 16000
PAIR = ("ja", "en")
FAIL_TEXT = "（翻訳できませんでした）"
TOK = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")


def toks(text: str) -> list[str]:
    return TOK.findall((text or "").lower().replace("’", "'"))


def _pct(xs, q):
    if not xs:
        return None
    xs = sorted(xs)
    return round(xs[min(len(xs) - 1, max(0, int(math.ceil(q * len(xs))) - 1))], 2)


def read_wav(path) -> np.ndarray:
    with wave.open(str(path), "rb") as w:
        sr, ch, sw = w.getframerate(), w.getnchannels(), w.getsampwidth()
        raw = w.readframes(w.getnframes())
    assert sw == 2, "16bit のみ"
    x = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    if ch > 1:
        x = x.reshape(-1, ch).mean(axis=1)
    assert sr == SR, f"16kHz のみ({sr})"
    return x


def write_wav(path, x) -> None:
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes((np.clip(x, -1, 1) * 32767).astype("<i2").tobytes())


# ====================================================================== build

def ami_words(zip_path: Path, meeting: str) -> list[dict]:
    """AMI manual の words XML(話者ごと)→ 時刻順の語 [{spk,s,e,w,sent}]。句読点で文を切る。"""
    import xml.etree.ElementTree as ET
    out = []
    with zipfile.ZipFile(zip_path) as z:
        names = sorted(n for n in z.namelist()
                       if re.fullmatch(rf"words/{re.escape(meeting)}\.[A-Z]\.words\.xml", n))
        if not names:
            raise SystemExit(f"{meeting} の words が zip に無い")
        for n in names:
            spk = n.split(".")[-3]
            sid = 0
            root = ET.fromstring(z.read(n))
            pend_punc = None
            for w in root:
                tag = w.tag.split("}")[-1]
                if tag != "w" or w.get("starttime") is None:
                    continue
                t = (w.text or "").strip()
                if w.get("punc") == "true":
                    if out and out[-1]["spk"] == spk:
                        out[-1]["w"] += t
                    if t in (".", "?", "!"):
                        sid += 1
                    continue
                try:
                    s, e = float(w.get("starttime")), float(w.get("endtime"))
                except (TypeError, ValueError):
                    continue
                out.append({"spk": spk, "s": s, "e": e, "w": t, "sent": f"{spk}{sid}",
                            "trunc": w.get("trunc") == "true"})
    return sorted(out, key=lambda x: (x["s"], x["e"]))


def build(zip_path: Path, wav: Path, start: float, dur: float, out_dir: Path,
          ref: bool = True) -> Path:
    m = re.match(r"([A-Z]{2}\d{4}[a-z]?)\.(.+)\.wav$", wav.name)
    if not m:
        raise SystemExit("WAV 名は <会議>.<マイク>.wav")
    meeting, mic = m.group(1), m.group(2)
    x = read_wav(wav)
    a, b = int(start * SR), int((start + dur) * SR)
    clip = x[a:b]
    words = [dict(w, s=round(w["s"] - start, 3), e=round(w["e"] - start, 3))
             for w in ami_words(zip_path, meeting) if w["s"] >= start and w["e"] <= start + dur]
    # 文(話者ごと、句点まで)
    sents: dict[str, dict] = {}
    for w in words:
        d = sents.setdefault(w["sent"], {"spk": w["spk"], "s": w["s"], "e": w["e"], "words": []})
        d["e"] = max(d["e"], w["e"])
        d["words"].append(w["w"])
    sl = sorted(sents.values(), key=lambda d: d["s"])
    for d in sl:
        d["text"] = " ".join(d.pop("words"))
    # 参照訳の単位(ターン): 同じ話者の文が他の人に割り込まれず1.5秒以内に続くならまとめる
    turns: list[dict] = []
    for d in sl:
        if not toks(d["text"]):
            continue
        t = turns[-1] if turns else None
        if t and t["spk"] == d["spk"] and d["s"] - t["e"] < 1.5 and not any(
                o["spk"] != d["spk"] and t["e"] - 0.1 < o["s"] < d["s"] for o in sl):
            t["text"] += " " + d["text"]
            t["e"] = max(t["e"], d["e"])
        else:
            turns.append({"spk": d["spk"], "s": d["s"], "e": d["e"], "text": d["text"]})
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{meeting}_{mic}_{int(start)}"
    write_wav(out_dir / f"{stem}.wav", clip)
    meta = {"meeting": meeting, "mic": mic, "start": start, "dur": dur, "wav": f"{stem}.wav",
            "source": "AMI Meeting Corpus (CC BY 4.0), manual annotations 1.6.2",
            "words": words, "sentences": sl, "turns": turns}
    if ref:
        add_reference(meta)
    jp = out_dir / f"{stem}.json"
    jp.write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[build] {jp.name}: {dur:.0f}s / 語 {len(words)} / 文 {len(sl)} / ターン {len(turns)} "
          f"/ 話者 {sorted({w['spk'] for w in words})}")
    return jp


def add_reference(meta: dict) -> None:
    """参照訳: 正解の書き起こしをアプリと同じ翻訳(remote、同じ指示・直前4ターンの文脈)で訳す。"""
    sys.path.insert(0, str(ROOT / "src"))
    from tsuyaku.config import engine_candidates, load_settings
    from tsuyaku.translate import build_messages, pick_model, translate_with_retry, warmup
    s = load_settings()
    _lab, base, pref = engine_candidates("remote", s.host, None, None)[0]
    model = pick_model(base, pref)
    warmup(base, model)
    hist: list[dict] = []
    for t in meta["turns"]:
        tr = translate_with_retry(base, model, build_messages(PAIR, "", hist, t["text"], "en"))
        t["ref_ja"] = tr or ""
        if tr:
            hist.append({"lang": "en", "orig": t["text"], "trans": tr})
    meta["ref_model"] = model


# ====================================================================== run

def app_in_use(minutes: float = 5.0) -> bool:
    """ユーザーがアプリを使っている(プロセスがあり、記録が直近 minutes 分に更新された)。"""
    import subprocess
    ps = subprocess.run(["ps", "aux"], capture_output=True, text=True).stdout
    me = str(os.getpid())
    running = any(re.search(r"python.*-m tsuyaku", ln) and me not in ln.split()[1:2]
                  for ln in ps.splitlines())
    if not running:
        return False
    logs = sorted(glob.glob(str(ROOT / "logs" / "session-*.jsonl")), key=os.path.getmtime)
    return bool(logs) and time.time() - os.path.getmtime(logs[-1]) < minutes * 60


class _Tee:
    def __init__(self, f, other):
        self.f, self.o = f, other
        self.lock = threading.Lock()

    def write(self, s):
        with self.lock:
            self.f.write(s)
        return len(s)

    def flush(self):
        self.f.flush()


def run(clip_json: Path, out_root: Path, src: Path | None, sets: list[str],
        tail_s: float = 12.0, wait: bool = True) -> Path:
    if wait:
        while app_in_use():
            print("[run] アプリが使われている: 1分待つ", flush=True)
            time.sleep(60)
    sys.path.insert(0, str((src or ROOT / "src").resolve()))
    import tsuyaku.__main__ as app
    from tsuyaku import audio as audio_mod
    from tsuyaku import stt as stt_mod

    meta = json.loads(clip_json.read_text(encoding="utf-8"))
    x = read_wav(clip_json.parent / meta["wav"])
    out = out_root / clip_json.stem
    out.mkdir(parents=True, exist_ok=True)
    rec: list = []           # (壁時計, 種類, 中身)
    lock = threading.Lock()

    def note(kind, **kw):
        with lock:
            rec.append((time.time(), kind, kw))

    feed_done = threading.Event()
    clock = {}

    class _RecQ:
        """区間の行列(seg_q)に積まれた区間を記録する。"""

        def __init__(self, q):
            self.q = q

        def put(self, item):
            note("seg", lo=item[1], hi=item[2], pre=item[3])
            self.q.put(item)

        def __getattr__(self, k):
            return getattr(self.q, k)

    new_style = hasattr(audio_mod.SpeechCapture, "on_chunk")

    class FileCapture(audio_mod.SpeechCapture):
        """マイクの代わりに WAV を実時間で流す(1チャンクの処理はアプリと同じ on_chunk)。"""

        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            if new_style:
                self.out = _RecQ(self.out)

        def _maybe_interim(self, closed):
            had = self._interim
            super()._maybe_interim(closed)
            if new_style and self._interim is not None and self._interim is not had:
                note("interim_set", lo=self._interim[1], hi=self.segmenter._fed / self.sr)

        def start(self):
            self._thread = threading.Thread(target=self._run_file, daemon=True)
            self._thread.start()

        def _run_file(self):
            rng = np.random.default_rng(0)
            n = len(x)
            tail = int(tail_s * SR)
            t0 = time.time()
            clock["t0"] = t0
            note("feed_start")
            i = 0
            while i < n + tail and not self._stop.is_set():
                tgt = t0 + (i + 512) / SR
                d = tgt - time.time()
                if d > 0:
                    time.sleep(d)
                if i < n:
                    ch = x[i:i + 512]
                    if len(ch) < 512:
                        ch = np.concatenate([ch, np.zeros(512 - len(ch), np.float32)])
                else:   # 話の後: 部屋の小さな雑音
                    ch = (rng.standard_normal(512) * 3e-4).astype(np.float32)
                    if not feed_done.is_set() and i >= n:
                        note("audio_end")
                        feed_done.set()
                if new_style:
                    self.on_chunk(ch)
                else:
                    self._cb(ch)
                i += 512
            feed_done.set()

        def _cb(self, mono):
            # 古い版(on_chunk が無い)用: SpeechCapture._run の cb と同じ(読み上げ中の一時停止も含む)
            if self.paused.is_set() or self._stop.is_set():
                if not getattr(self, "_was_paused", False):
                    self._was_paused = True
                    for item in self.segmenter.flush_current():
                        note("seg", lo=item[1], hi=item[2], pre=item[3], flush=True)
                        self.out.put(item)
                return
            self._was_paused = False
            self.level = min(1.0, float(np.sqrt(np.mean(np.square(mono, dtype=np.float64)))) * 3.0)
            g = self.agc(mono)
            self.vad.accept_waveform(g)
            while not self.vad.empty():
                self.vad.pop()
            items = self.segmenter.feed(g, self.vad.is_speech_detected())
            self._clock = (self.segmenter._fed / self.sr, time.time())
            for item in items:
                note("seg", lo=item[1], hi=item[2], pre=item[3])
                self.out.put(item)
            had = self._interim
            audio_mod.SpeechCapture._maybe_interim(self, bool(items))
            if self._interim is not None and self._interim is not had:
                note("interim_set", lo=self._interim[1],
                     hi=self.segmenter._fed / self.sr)

    class RecEndpointer(app.Endpointer):
        def push(self, *a, **k):
            us = super().push(*a, **k)
            self._rec(us, "push")
            return us

        def tick(self, *a, **k):
            us = super().tick(*a, **k)
            self._rec(us, "tick")
            return us

        def flush(self):
            us = super().flush()
            self._rec(us, "flush")   # アプリは詰まったとき flush の結果を捨てる
            return us

        def _rec(self, us, how):
            for u in us:
                note("unit", how=how, text=u.text, lang=u.lang, lo=u.lo, hi=u.hi,
                     speech_lo=u.speech_lo, parts=u.parts)

    class _NoTTS:
        def __init__(self, *a, **k):
            self.on_start = None

        def prepare(self, *a, **k):
            pass

        def say(self, *a, **k):
            pass

        def stop(self):
            pass

    # 耳の呼び出しを記録(どのスレッド = 確定/途中字幕/予想の先出し)
    _rt = stt_mod.RemoteSTT.transcribe

    def rt(self, samples, *a, **k):
        t0 = time.time()
        r = _rt(self, samples, *a, **k)
        note("stt", dur=len(samples) / SR, dt=time.time() - t0, where=self.last_where,
             lang=r[1], text=r[0][:80], prob=float(r[2]))
        return r
    stt_mod.RemoteSTT.transcribe = rt
    if hasattr(stt_mod, "SenseVoiceSTT"):
        _sv = stt_mod.SenseVoiceSTT.transcribe

        def sv(self, samples, *a, **k):
            t0 = time.time()
            r = _sv(self, samples, *a, **k)
            note("mac_ear", dur=len(samples) / SR, dt=time.time() - t0, lang=r[1], text=r[0][:80])
            return r
        stt_mod.SenseVoiceSTT.transcribe = sv

    class UIRec:
        def put(self, evt):
            note("ui", evt=list(evt))

    _pick = app.pick_segments

    def pick(pending):
        out = _pick(pending)
        if len(out) < len(pending):   # 詰まり: 古い区間/訳を捨てた(取りこぼし)
            kind = "seg" if pending and isinstance(pending[0], tuple) and len(pending[0]) == 4 \
                and hasattr(pending[0][0], "shape") else "text"
            note("backlog_drop", what=kind, n=len(pending) - len(out))
        return out
    app.pick_segments = pick
    app.SpeechCapture = FileCapture
    app.Endpointer = RecEndpointer
    b = app.Bridge()
    for kv in sets:
        k, v = kv.split("=", 1)
        cur = getattr(b.s, k)
        setattr(b.s, k, type(cur)(json.loads(v)) if not isinstance(cur, str) else v)
    if not b.s.voice_on:
        app.TTS = _NoTTS   # 字幕だけ(ユーザーの今の使い方): 声のモデルも読まない
    b.s.log_dir = str(out / "applog")
    b.ui = UIRec()
    logf = (out / "app.log").open("w", encoding="utf-8")
    old_out = sys.stdout
    sys.stdout = _Tee(logf, old_out)
    th = threading.Thread(target=b._conversation, args=("-".join(PAIR), "", "interpret"),
                          daemon=True)
    t_start = time.time()
    th.start()
    try:
        while not feed_done.wait(1.0):
            if not th.is_alive():
                raise RuntimeError("会話が始まらなかった(app.log を見る)")
        time.sleep(4.0)    # 話の後の雑音の残り + 最後の訳
        deadline = time.time() + 20
        while time.time() < deadline:   # 訳が出きるまで(最大20秒)
            with lock:
                last = max((t for t, k, _ in rec if k in ("ui", "unit")), default=0)
            if time.time() - last > 4.0:
                break
            time.sleep(0.5)
    finally:
        b.running.clear()
        th.join(timeout=15)
        sys.stdout = old_out
        logf.close()
    t0 = clock.get("t0", t_start)
    src_git = _git(src or ROOT)
    events = [{"t": round(t - t0, 3), "kind": k, **kw} for t, k, kw in rec]
    run_meta = {"clip": str(clip_json), "src": str(src or ROOT / "src"), "git": src_git,
                "sets": sets, "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                "settings": {k: getattr(b.s, k) for k in (
                    "vad_threshold", "interim_s", "stt_hedge_s", "stt_preview_s", "speculate",
                    "voice_on", "hold_open_s", "max_speech_s", "engine")},
                "model": getattr(b, "model_used", ""), "engine": getattr(b, "engine_used", ""),
                "uplink": (None if getattr(b, "uplink", None) is None else
                           {"span_hits": getattr(b.stt, "span_hits", 0),
                            "span_miss": getattr(b.stt, "span_miss", 0),
                            "sent_bytes": b.uplink.sent_bytes})}
    (out / "events.json").write_text(json.dumps({"meta": run_meta, "events": events},
                                                ensure_ascii=False, indent=0), encoding="utf-8")
    return evaluate(clip_json, out)


def _git(p: Path) -> str:
    import subprocess
    try:
        r = subprocess.run(["git", "-C", str(p), "rev-parse", "--short", "HEAD"],
                           capture_output=True, text=True, timeout=5)
        d = subprocess.run(["git", "-C", str(p), "status", "--porcelain", "src"],
                           capture_output=True, text=True, timeout=5).stdout.strip()
        return r.stdout.strip() + ("+dirty" if d else "")
    except Exception:  # noqa: BLE001
        return "?"


# ====================================================================== 指標

def _chrf_mod():
    sys.path.insert(0, str(ROOT / "scripts"))
    import eval_conv
    return eval_conv


def mark_shown(gold: list[dict], shows: list[tuple[float, str]], before: float = 30.0) -> list:
    """shows = [(時刻, 表示した原文)] → 正解の各語が初めて出た時刻(出なければ None)。
    表示の原文を、その時刻までの直近 before 秒の正解の語列に並べ、2語以上続けて一致した語を
    「出た」とする(1語だけの一致は "yeah" などの偶然が多いので数えない)。"""
    gt = [toks(w["w"]) for w in gold]
    flat, owner = [], []
    for i, ts in enumerate(gt):
        for t in ts:
            flat.append(t)
            owner.append(i)
    starts = [gold[owner[k]]["s"] for k in range(len(flat))]
    ends = [gold[owner[k]]["e"] for k in range(len(flat))]
    first = [None] * len(gold)
    import bisect
    for T, text in sorted(shows):
        d = toks(text)
        if len(d) < 2:
            continue
        hi = bisect.bisect_right(starts, T + 0.5)
        lo = 0
        while lo < hi and ends[lo] < T - before:
            lo += 1
        win = flat[lo:hi]
        if not win:
            continue
        sm = difflib.SequenceMatcher(None, d, win, autojunk=False)
        for blk in sm.get_matching_blocks():
            if blk.size >= 2:
                for k in range(blk.b, blk.b + blk.size):
                    i = owner[lo + k]
                    if first[i] is None or T < first[i]:
                        first[i] = T
    return first


def evaluate(clip_json: Path, out: Path) -> Path:
    ec = _chrf_mod()
    meta = json.loads(clip_json.read_text(encoding="utf-8"))
    run = json.loads((out / "events.json").read_text(encoding="utf-8"))
    ev = run["events"]
    gold = [w for w in meta["words"] if toks(w["w"])]
    src_shows, tr_shows, fin_shows, finals = [], [], [], []
    for e in ev:
        if e["kind"] != "ui":
            continue
        k = e["evt"][0]
        if k == "partial":
            _k, text, trans, lang, *rest = e["evt"] + [None]
            if lang == "en" and text:
                src_shows.append((e["t"], text))
                if trans and trans != "…":
                    tr_shows.append((e["t"], text))
        elif k == "utterance":
            d = e["evt"][1]
            if d.get("lang") == "en" and d.get("orig"):
                src_shows.append((e["t"], d["orig"]))
                if d.get("trans") and d["trans"] != FAIL_TEXT:
                    tr_shows.append((e["t"], d["orig"]))
                    fin_shows.append((e["t"], d["orig"]))
            finals.append({"t": e["t"], **d})
    f_src = mark_shown(gold, src_shows)
    f_tr = mark_shown(gold, tr_shows)
    f_fin = mark_shown(gold, fin_shows)
    lag = lambda f: [round(f[i] - w["e"], 3) for i, w in enumerate(gold) if f[i] is not None]
    L_src, L_tr, L_fin = lag(f_src), lag(f_tr), lag(f_fin)
    n = len(gold)

    # 長い発話: 同じ人が切れ目1秒未満で6秒以上話し続けた区間
    runs = []
    by_spk: dict = {}
    for i, w in enumerate(gold):
        by_spk.setdefault(w["spk"], []).append(i)
    for spk, idx in by_spk.items():
        cur = [idx[0]]
        for i in idx[1:]:
            if gold[i]["s"] - gold[cur[-1]]["e"] < 1.0:
                cur.append(i)
            else:
                runs.append(cur)
                cur = [i]
        runs.append(cur)
    long_runs = [r for r in runs if gold[r[-1]]["e"] - gold[r[0]]["s"] >= 6.0]
    lr_src, lr_tr, lr_lag_src, lr_lag_tr = [], [], [], []
    for r in long_runs:
        s0 = gold[r[0]]["s"]
        a = [f_src[i] for i in r if f_src[i] is not None]
        b = [f_tr[i] for i in r if f_tr[i] is not None]
        lr_src.append(round(min(a) - s0, 2) if a else None)
        lr_tr.append(round(min(b) - s0, 2) if b else None)
        lr_lag_src += [f_src[i] - gold[i]["e"] for i in r if f_src[i] is not None]
        lr_lag_tr += [f_tr[i] - gold[i]["e"] for i in r if f_tr[i] is not None]

    # WER(確定字幕の原文を出た順につなぐ vs 正解を時刻順)
    hyp = [t for f in finals if f.get("lang") == "en" for t in toks(f.get("orig", ""))]
    ref = [t for w in gold for t in toks(w["w"])]
    wer = ec.edit_distance(hyp, ref) / max(1, len(ref))

    # 区切り(Endpointer が出した単位。flush で捨てられた分は別に数える)
    units = [e for e in ev if e["kind"] == "unit" and e["how"] != "flush"]
    flushed = [e for e in ev if e["kind"] == "unit" and e["how"] == "flush"]
    ulen = [u["hi"] - u["speech_lo"] for u in units]

    def unit_of(t):
        for k, u in enumerate(units):
            if u["speech_lo"] - 0.3 <= t <= u["hi"] + 0.3:
                return k
        return -1
    sent_units: dict = {}
    for w in gold:
        sent_units.setdefault(w["sent"], []).append(unit_of((w["s"] + w["e"]) / 2))
    big = [v for v in sent_units.values() if len(v) >= 3]
    split = sum(1 for v in big if len({k for k in v if k >= 0}) > 1)
    # 単位ごとの話者
    cross = 0
    for k, u in enumerate(units):
        spk: dict = {}
        for w in gold:
            if u["speech_lo"] - 0.1 <= (w["s"] + w["e"]) / 2 <= u["hi"] + 0.1:
                spk[w["spk"]] = spk.get(w["spk"], 0) + 1
        cross += sum(1 for v in spk.values() if v >= 2) >= 2

    # 翻訳: 30秒の窓ごとに 確定訳 vs 参照訳 の chrF(統計を足してから)
    umap: list = []
    used = set()
    for f in finals:
        for k, u in enumerate(units):
            if k not in used and u["text"] == f.get("orig"):
                used.add(k)
                umap.append(((u["speech_lo"] + u["hi"]) / 2, f.get("trans", "")))
                break
    W = 30.0
    nwin = int(math.ceil(meta["dur"] / W))
    acc = None
    win_scores = []
    for wi in range(nwin):
        a, b = wi * W, (wi + 1) * W
        h = "".join(t for m, t in umap if a <= m < b and t != FAIL_TEXT)
        r = "".join(t.get("ref_ja", "") for t in meta["turns"] if a <= (t["s"] + t["e"]) / 2 < b)
        if not r:
            continue
        st = ec.chrf_stats(h, r)
        acc = ec.add_stats(acc, st)
        win_scores.append(round(ec.chrf_from_stats(st), 1))

    # 予想訳の中身: 原文ごとに最後に出た予想訳を、原文が一番重なる確定字幕の訳と比べる(chrF)
    last_guess: dict = {}
    for e in ev:
        if e["kind"] == "ui" and e["evt"][0] == "partial" and len(e["evt"]) > 4 and e["evt"][4] \
                and e["evt"][2] not in ("…", ""):
            last_guess[e["evt"][1]] = e["evt"][2]
    agree = []
    fin_toks = [(toks(f.get("orig", "")), f.get("trans", "")) for f in finals
                if f.get("trans") and f.get("trans") != FAIL_TEXT]
    for g_orig, g_tr in last_guess.items():
        gt = toks(g_orig)
        if len(gt) < 3 or not fin_toks:
            continue
        best = max(fin_toks, key=lambda ft: difflib.SequenceMatcher(None, gt, ft[0], autojunk=False).ratio())
        if difflib.SequenceMatcher(None, gt, best[0], autojunk=False).ratio() >= 0.6:
            agree.append(ec.chrf(g_tr, best[1]))

    stt = [e for e in ev if e["kind"] == "stt"]
    リモートサーバー = [e["dt"] for e in stt if e["where"] == "remote"]
    mac = [e for e in ev if e["kind"] == "mac_ear"]
    sm = {
        "clip": clip_json.stem, "dur": meta["dur"], "words": n, "git": run["meta"]["git"],
        "sets": run["meta"]["sets"], "model": run["meta"].get("model"),
        "lag_src_med": _pct(L_src, .5), "lag_src_p90": _pct(L_src, .9),
        "lag_tr_med": _pct(L_tr, .5), "lag_tr_p90": _pct(L_tr, .9),
        "lag_fin_med": _pct(L_fin, .5), "lag_fin_p90": _pct(L_fin, .9),
        "cov_src": round(100 * len(L_src) / max(1, n), 1),
        "cov_tr": round(100 * len(L_tr) / max(1, n), 1),
        "cov_fin": round(100 * len(L_fin) / max(1, n), 1),
        "long_runs": len(long_runs),
        "long_first_src_med": _pct([v for v in lr_src if v is not None], .5),
        "long_first_src_p90": _pct([v for v in lr_src if v is not None], .9),
        "long_first_tr_med": _pct([v for v in lr_tr if v is not None], .5),
        "long_first_tr_p90": _pct([v for v in lr_tr if v is not None], .9),
        "long_lag_src_med": _pct(lr_lag_src, .5), "long_lag_tr_med": _pct(lr_lag_tr, .5),
        "long_lag_tr_p90": _pct(lr_lag_tr, .9),
        "wer": round(100 * wer, 1), "hyp_words": len(hyp), "ref_words": len(ref),
        "units": len(units), "flushed": len(flushed),
        "unit_len_med": _pct(ulen, .5), "unit_len_p90": _pct(ulen, .9), "unit_len_max": _pct(ulen, 1.0),
        "units_over8": sum(1 for v in ulen if v > 8.0),
        "sent_split": round(100 * split / max(1, len(big)), 1), "cross_units": cross,
        "chrf": round(ec.chrf_from_stats(acc), 1) if acc else None, "chrf_windows": win_scores,
        "finals": len(finals), "fail": sum(1 for f in finals if f.get("trans") == FAIL_TEXT),
        "stt_calls": len(stt), "stt_remote_med": _pct(remote, .5), "stt_remote_p90": _pct(remote, .9),
        "stt_mac": sum(1 for e in stt if e["where"] == "mac"),
        "segs": sum(1 for e in ev if e["kind"] == "seg"),
        "interim_set": sum(1 for e in ev if e["kind"] == "interim_set"),
        "mac_ear_calls": len(mac), "mac_ear_med": _pct([e["dt"] for e in mac], .5),
        "partials": sum(1 for e in ev if e["kind"] == "ui" and e["evt"][0] == "partial"),
        "guess_vs_final_chrf": round(sum(agree) / len(agree), 1) if agree else None,
        "guess_n": len(agree),
        "backlog_seg": sum(e["n"] for e in ev if e["kind"] == "backlog_drop" and e["what"] == "seg"),
        "backlog_text": sum(e["n"] for e in ev if e["kind"] == "backlog_drop" and e["what"] == "text"),
    }
    (out / "summary.json").write_text(json.dumps(sm, ensure_ascii=False, indent=1), encoding="utf-8")
    # 語ごとの表示時刻(原因調べ用)
    rows = [{"i": i, "spk": w["spk"], "s": w["s"], "e": w["e"], "w": w["w"],
             "src": f_src[i], "tr": f_tr[i], "fin": f_fin[i]} for i, w in enumerate(gold)]
    (out / "words.json").write_text(json.dumps(rows, ensure_ascii=False, indent=0), encoding="utf-8")
    print(fmt(sm))
    return out / "summary.json"


def fmt(sm: dict) -> str:
    return (f"### {sm['clip']} ({sm['dur']:.0f}s, 語{sm['words']}, app={sm['git']} {sm['sets']})\n"
            f"語の遅れ 中央/p90: 原文 {sm['lag_src_med']}/{sm['lag_src_p90']}s  訳 {sm['lag_tr_med']}/"
            f"{sm['lag_tr_p90']}s  確定 {sm['lag_fin_med']}/{sm['lag_fin_p90']}s  "
            f"(出た語の割合 原文{sm['cov_src']}% 訳{sm['cov_tr']}% 確定{sm['cov_fin']}%)\n"
            f"長い発話{sm['long_runs']}: 話し始め→最初の原文 {sm['long_first_src_med']}/{sm['long_first_src_p90']}s "
            f"→最初の訳 {sm['long_first_tr_med']}/{sm['long_first_tr_p90']}s / 語の遅れ 原文 {sm['long_lag_src_med']}s "
            f"訳 {sm['long_lag_tr_med']}/{sm['long_lag_tr_p90']}s\n"
            f"WER {sm['wer']}% (確定の語{sm['hyp_words']}/正解{sm['ref_words']}) / chrF(参照訳, 30秒窓) {sm['chrf']} "
            f"/ 単位{sm['units']}(捨て{sm['flushed']}) 長さ 中央{sm['unit_len_med']} p90 {sm['unit_len_p90']} "
            f"最大{sm['unit_len_max']} 8秒超{sm['units_over8']} / 文の分割 {sm['sent_split']}% / 2人入り {sm['cross_units']}\n"
            f"耳: リモートサーバー {sm['stt_calls'] - sm['stt_mac']}回 中央{sm['stt_remote_med']}s p90 {sm['stt_remote_p90']}s "
            f"/ Mac確定 {sm['stt_mac']} / Macの耳 {sm['mac_ear_calls']}回 {sm['mac_ear_med']}s / 区間 {sm['segs']} "
            f"/ 途中音声 {sm['interim_set']} / 失敗 {sm['fail']} / 詰まりで捨てた 区間{sm.get('backlog_seg', '-')}"
            f" 訳{sm.get('backlog_text', '-')} / 予想訳 vs 確定訳 chrF {sm.get('guess_vs_final_chrf')}"
            f" (n={sm.get('guess_n')})")


# ====================================================================== A/B

def pool(d: Path) -> dict:
    out = {}
    for sp in sorted(d.glob("*/summary.json")):
        out[sp.parent.name] = json.loads(sp.read_text(encoding="utf-8"))
    return out


def pooled(d: Path, clips: "set[str] | None" = None) -> dict:
    """クリップをまたいで語の遅れ等を合算(words.json から計算し直す)。"""
    ec = _chrf_mod()
    L = {"src": [], "tr": [], "fin": []}
    n = 0
    wer_e = wer_n = 0
    acc = None
    ulen = []
    over8 = units = 0
    lr_src, lr_tr = [], []
    リモートサーバー = []
    for sp in sorted(d.glob("*/summary.json")):
        if clips is not None and sp.parent.name not in clips:
            continue
        sm = json.loads(sp.read_text(encoding="utf-8"))
        rows = json.loads((sp.parent / "words.json").read_text(encoding="utf-8"))
        n += len(rows)
        for r in rows:
            for k in L:
                if r[k] is not None:
                    L[k].append(r[k] - r["e"])
        wer_e += sm["wer"] * sm["ref_words"] / 100
        wer_n += sm["ref_words"]
        units += sm["units"]
        over8 += sm["units_over8"]
        ev = json.loads((sp.parent / "events.json").read_text(encoding="utf-8"))["events"]
        ulen += [e["hi"] - e["speech_lo"] for e in ev if e["kind"] == "unit" and e["how"] != "flush"]
        リモートサーバー += [e["dt"] for e in ev if e["kind"] == "stt" and e["where"] == "remote"]
        clip = json.loads(Path(json.loads((sp.parent / "events.json").read_text(
            encoding="utf-8"))["meta"]["clip"]).read_text(encoding="utf-8"))
        # 窓 chrF は summary の再計算が要るので evaluate と同じ方法で統計を足す
        finals = [e["evt"][1] for e in ev if e["kind"] == "ui" and e["evt"][0] == "utterance"]
        us = [e for e in ev if e["kind"] == "unit" and e["how"] != "flush"]
        used, umap = set(), []
        for f in finals:
            for k, u in enumerate(us):
                if k not in used and u["text"] == f.get("orig"):
                    used.add(k)
                    umap.append(((u["speech_lo"] + u["hi"]) / 2, f.get("trans", "")))
                    break
        for wi in range(int(math.ceil(clip["dur"] / 30.0))):
            a, b = wi * 30.0, (wi + 1) * 30.0
            h = "".join(t for m, t in umap if a <= m < b and t != FAIL_TEXT)
            r = "".join(t.get("ref_ja", "") for t in clip["turns"] if a <= (t["s"] + t["e"]) / 2 < b)
            if r:
                acc = ec.add_stats(acc, ec.chrf_stats(h, r))
        if sm.get("long_first_src_med") is not None:
            pass
        wr = json.loads((sp.parent / "words.json").read_text(encoding="utf-8"))
        gold = clip["words"]
        # 長い発話の最初の表示(evaluate と同じ定義)
        by: dict = {}
        g2 = [w for w in gold if toks(w["w"])]
        for i, w in enumerate(g2):
            by.setdefault(w["spk"], []).append(i)
        for _spk, idx in by.items():
            cur = [idx[0]]
            groups = []
            for i in idx[1:]:
                if g2[i]["s"] - g2[cur[-1]]["e"] < 1.0:
                    cur.append(i)
                else:
                    groups.append(cur)
                    cur = [i]
            groups.append(cur)
            for r in groups:
                if g2[r[-1]]["e"] - g2[r[0]]["s"] < 6.0:
                    continue
                s0 = g2[r[0]]["s"]
                a_ = [wr[i]["src"] for i in r if wr[i]["src"] is not None]
                b_ = [wr[i]["tr"] for i in r if wr[i]["tr"] is not None]
                if a_:
                    lr_src.append(min(a_) - s0)
                if b_:
                    lr_tr.append(min(b_) - s0)
    return {"words": n, "cov": {k: round(100 * len(v) / max(1, n), 1) for k, v in L.items()},
            "lag": {k: (_pct(v, .5), _pct(v, .9)) for k, v in L.items()},
            "wer": round(100 * wer_e / max(1, wer_n), 1), "chrf": round(ec.chrf_from_stats(acc), 1) if acc else None,
            "units": units, "over8": over8, "ulen": (_pct(ulen, .5), _pct(ulen, .9), _pct(ulen, 1.0)),
            "long": len(lr_src), "long_src": (_pct(lr_src, .5), _pct(lr_src, .9)),
            "long_tr": (_pct(lr_tr, .5), _pct(lr_tr, .9)), "remote": (_pct(remote, .5), _pct(remote, .9))}


def ab(da: Path, db: Path, save: Path | None = None) -> str:
    A, B = pool(da), pool(db)
    common = set(A) & set(B)
    pa, pb = pooled(da, common), pooled(db, common)
    f = lambda t: f"{t[0]}/{t[1]}"

    def gq(D):
        v = [D[c].get("guess_vs_final_chrf") for c in common if D[c].get("guess_vs_final_chrf") is not None]
        return round(sum(v) / len(v), 1) if v else None
    rows = [f"A={da} / B={db}(共通クリップ {len(common)}: {', '.join(sorted(common))})", "",
            "| 指標 | A | B |", "|---|---|---|"]
    rows += [
        f"| 語の遅れ 原文 中央/p90 (s) | {f(pa['lag']['src'])} | {f(pb['lag']['src'])} |",
        f"| 語の遅れ 訳(予想含む) 中央/p90 | {f(pa['lag']['tr'])} | {f(pb['lag']['tr'])} |",
        f"| 語の遅れ 確定字幕 中央/p90 | {f(pa['lag']['fin'])} | {f(pb['lag']['fin'])} |",
        f"| 出た語の割合 原文/訳/確定 (%) | {pa['cov']['src']}/{pa['cov']['tr']}/{pa['cov']['fin']} "
        f"| {pb['cov']['src']}/{pb['cov']['tr']}/{pb['cov']['fin']} |",
        f"| 長い発話: 話し始め→最初の原文 中央/p90 | {f(pa['long_src'])} | {f(pb['long_src'])} |",
        f"| 長い発話: 話し始め→最初の訳 中央/p90 | {f(pa['long_tr'])} | {f(pb['long_tr'])} |",
        f"| WER 確定 (%) | {pa['wer']} | {pb['wer']} |",
        f"| chrF 確定訳 vs 参照訳(30秒窓) | {pa['chrf']} | {pb['chrf']} |",
        f"| 単位 数 / 8秒超 | {pa['units']} / {pa['over8']} | {pb['units']} / {pb['over8']} |",
        f"| 単位の長さ 中央/p90/最大 | {pa['ulen'][0]}/{pa['ulen'][1]}/{pa['ulen'][2]} "
        f"| {pb['ulen'][0]}/{pb['ulen'][1]}/{pb['ulen'][2]} |",
        f"| リモートサーバーの耳 往復 中央/p90 | {f(pa['remote'])} | {f(pb['remote'])} |",
        f"| 予想訳 vs 確定訳 chrF(クリップ平均) | {gq(A)} | {gq(B)} |",
    ]
    rows += ["", "| クリップ | 原文遅れ中央 A→B | 訳遅れ中央 A→B | 確定遅れ中央 A→B | WER A→B | chrF A→B |",
             "|---|---|---|---|---|---|"]
    for c in sorted(common):
        a, b = A[c], B[c]
        rows.append(f"| {c} | {a['lag_src_med']}→{b['lag_src_med']} | {a['lag_tr_med']}→{b['lag_tr_med']} "
                    f"| {a['lag_fin_med']}→{b['lag_fin_med']} | {a['wer']}→{b['wer']} | {a['chrf']}→{b['chrf']} |")
    txt = "\n".join(rows)
    print(txt)
    if save:
        save.write_text(txt + "\n", encoding="utf-8")
    return txt


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--zip", type=Path, required=True)
    b.add_argument("--wav", type=Path, required=True)
    b.add_argument("--start", type=float, required=True)
    b.add_argument("--dur", type=float, default=180.0)
    b.add_argument("--out", type=Path, default=ROOT / "eval" / "english" / "clips")
    b.add_argument("--no-ref", action="store_true")
    r = sub.add_parser("run")
    r.add_argument("--clip", type=Path, nargs="+", required=True)
    r.add_argument("--out", type=Path, required=True)
    r.add_argument("--src", type=Path, default=None, help="別の版の src(A/B 用の worktree)")
    r.add_argument("--set", action="append", default=[], help="設定の上書き key=値(JSON)")
    r.add_argument("--no-wait", action="store_true")
    e = sub.add_parser("eval")
    e.add_argument("--clip", type=Path, required=True)
    e.add_argument("--out", type=Path, required=True)
    x = sub.add_parser("ab")
    x.add_argument("--a", type=Path, required=True)
    x.add_argument("--b", type=Path, required=True)
    x.add_argument("--save", type=Path)
    a = ap.parse_args()
    if a.cmd == "build":
        build(a.zip, a.wav, a.start, a.dur, a.out, ref=not a.no_ref)
    elif a.cmd == "run":
        if len(a.clip) > 1:   # 1プロセス1クリップ(アプリのモジュールの差し替えを持ち越さない)
            import subprocess
            for c in a.clip:
                cmd = [sys.executable, "-I", __file__, "run", "--clip", str(c), "--out", str(a.out)]
                if a.src:
                    cmd += ["--src", str(a.src)]
                for s_ in a.set:
                    cmd += ["--set", s_]
                if a.no_wait:
                    cmd.append("--no-wait")
                subprocess.run(cmd, check=False)
        else:
            run(a.clip[0], a.out, a.src, a.set, wait=not a.no_wait)
    elif a.cmd == "eval":
        evaluate(a.clip, a.out / a.clip.stem)
    elif a.cmd == "ab":
        ab(a.a, a.b, a.save)


if __name__ == "__main__":
    main()
