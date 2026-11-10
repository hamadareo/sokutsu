"""Sokutsu(即通)— 対面多言語リアルタイム通訳アプリのエントリポイント。

起動すると起動画面(言語ペア+テーマ)を表示し、会話を始めると
マイク → VAD → whisper(STT) → ローカルLLM(翻訳) → 字幕+TTS
のパイプラインを回し続ける。翻訳の処理場所は起動画面の『翻訳の処理場所』で
選べ、既定の自動は リモートGPU(run.sh がSSH転送を自動で貼る)→ この Mac の
LM Studio の順に繋ぐ。クラウドAPIは使わない。
"""
from __future__ import annotations

import json
import os
import sys
import queue
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import requests
import webview

from .audio import rms, MicTest, SpeechCapture
from . import devices
from .config import (AUDIO_SOURCES, ENGINE_LABELS, MAC_PORT, MODELS, REMOTE_PORT, LANG_NAMES,  # noqa: F401
                     engine_candidates, load_settings, save_config)
from .terms import clean_heard
from .terms import source_forms
from .relisten import Relistener, dropped_letters
from .terms import set_domain as terms_set_domain
from .terms import warm as terms_warm
from .guards import _TERM_SPLIT, length_off, strip_history_echo
from .guards import (echoes_prompt, ends_open, hallucinated_out_of_pair, terms_only,  # noqa: E501
                     looks_truncated, merge_held, pick_segments, stock_phrase, stt_prompt,
                     wrong_language)
from .langdetect import detect_lang, is_echo, norm
from . import inmeeting, notes as notes_mod
from .inmeeting import InMeetingMixin
from . import sysmon
from . import meeting_ai, note_templates  # W1(11/07): 会議に質問・テンプレート・下書き・ライブ要約
from .log import SessionLog
from .meeting_audio import AudioAPI   # W2: 録音・再生・ファイルの取り込み・字幕
from .segmenter import Endpointer, head_text, merge_segments, pick_cut_at
from .speculate import Speculator
from . import stt as _stt_mod
from .stt import STT, RemoteSTT  # noqa: F401
from .uplink import Uplink, server_streams
from . import translate as translate_mod
from .translate import (build_messages, out_lang, pick_model, target_lang,  # noqa: F401
                        translate_stream, translate_with_retry, warm_prompt, warmup)
from .tts import TTS, voicevox_voices
from .pace import PaceController, engine_of, ears_only
from . import library as library_mod
from . import localllm   # 配る版の同梱翻訳エンジン(ユーザー本人の版では何もしない)
from .bridge_library import LibraryApi

WEBUI = Path(__file__).resolve().parent / "webui" / "index.html"

# 試聴用のサンプル文(声色と速さがひと目で/ひと耳で分かる短文)
_PREVIEW_TEXT = {
    "ja": "こんにちは、はじめまして。今日はよろしくお願いします。",
    "en": "Hello, nice to meet you. Shall we begin?",
    "zh": "你好，很高兴认识大家，请多关照。",
    "ko": "안녕하세요, 반갑습니다. 잘 부탁드립니다.",
}

# 起動画面で選べる6言語ペア(index.html の data-pair と必ず一致させる)
_PAIRS = frozenset({"ja-en", "ja-zh", "ja-ko", "en-zh", "en-ko", "zh-ko"})
_DEFAULT_PAIR = "ja-en"
# 使い方(11/06): 通訳だけ / 会議メモだけ(訳さない・読まない) / 通訳+会議メモ
_MODES = ("interpret", "notes", "both")
_MONO = frozenset({"ja", "en", "zh", "ko"})   # 会議メモだけ では1言語も選べる


def _speaker_tracker(s):
    """話者の見分け(speakers.py、声の特徴)。モデルが無ければ None(話者なしで動く)。"""
    try:
        from .speakers import SpeakerTracker, find_model
    except Exception:  # noqa: BLE001
        return None
    try:
        path = find_model(MODELS)
        return SpeakerTracker(path) if path else None
    except Exception as e:  # noqa: BLE001
        print(f"[spk] 話者の見分けを使えません: {e}", flush=True)
        return None


class HearThrottle:
    """「聞こえているか」のメーターを画面へ送る間隔(11/08 hear)。最大 max_hz 回/秒、値が min_delta 以上
    動くか状態が変わった時だけ。同じままでも keep_s 秒に1回は送る(画面の見張りが止まったと思わない)。"""

    def __init__(self, max_hz: float = 16.0, min_delta: float = 0.02, keep_s: float = 1.0):
        self.gap = 1.0 / max_hz
        self.min_delta = min_delta
        self.keep_s = keep_s
        self._t = -1e9
        self._lv = None
        self._st = None

    def due(self, now: float, lv: float, st: str) -> bool:
        if st != self._st:
            ok = now - self._t >= min(self.gap, 0.03)    # 状態の変化はすぐ(続けて替わる時だけ間引く)
        elif now - self._t < self.gap:
            ok = False
        else:
            ok = (self._lv is None or abs(lv - self._lv) >= self.min_delta
                  or now - self._t >= self.keep_s)
        if ok:
            self._t, self._lv, self._st = now, lv, st
        return ok


def hear_state(cap, ui_status, now: float, lost_until: float = 0.0) -> str:
    """画面の「聞こえ」の状態(11/08 hear):
      lost   デバイスが外れた(既定で開き直した直後の数秒)
      nomic  マイクが止まっている / 音が届いていない(完全な 0・届かない・開けない)
      speak  読み上げ中 — マイク休止
      speak_listen 読み上げ中 — 聞いています(読み上げ中も聞き続ける時: capture.tts_active が立ち paused は立たない)
      speech 声を検出(VAD)
      work   聞き取り中 / 翻訳中
      quiet  静か"""
    if now < lost_until:
        return "lost"
    if getattr(cap, "health", "ok") in ("zeros", "stall", "fail"):
        return "nomic"
    pz = getattr(cap, "paused", None)
    if pz is not None and pz.is_set():
        return "speak"
    ta = getattr(cap, "tts_active", None)
    if ta is not None and ta.is_set():
        return "speak_listen"
    if getattr(cap, "speech", False):
        return "speech"
    st = ui_status or ("listen", 0.0)
    if st[0] == "work" and now - st[1] < 15:
        return "work"
    return "quiet"


def voice_breakdown(end, marks, t_deq: float, t_start: float) -> str:
    """読み上げが鳴り始めるまでの内訳(11/08「短い1語の読み上げが +3〜4.3 秒」の切り分け用。ログだけ)。
    区切り = 話し終わり → Endpointer が単位を確定(続き待ち)、訳 = 確定 → 訳ができて行列に入る、
    行列 = 前の読み上げの終わり待ち、鳴り始め = 合成・再生の立ち上がり。"""
    if not end or not marks or marks.get("ready") is None or marks.get("enq") is None:
        return ""
    r, q = float(marks["ready"]), float(marks["enq"])
    return (f"内訳 区切り{max(0.0, r - end):.1f} 訳{max(0.0, q - r):.1f} "
            f"行列{max(0.0, t_deq - q):.1f} 鳴り始め{max(0.0, t_start - t_deq):.1f}")


def latency_marks(end, ready, guess_shown, first_delta, caption, hit) -> "dict | None":
    """応答速度(11/06「応答速度を可視化できるようにしてください」)。すべて話し終わりからの秒。
      heard   = 文が確定した(認識 + 文末の続き待ち)
      first   = 最初に訳が画面に出た(先読みの予想訳 or 確定訳の1文字目の早い方)
      caption = 確定字幕
    声(読み上げの鳴り始め)は後から TSUYAKU.voice({id, lat}) で届く。"""
    if not end:
        return None
    firsts = [t for t in (guess_shown, first_delta) if t]
    r = lambda t: round(max(0.0, t - end), 2)
    return {"heard": r(ready) if ready else None,
            "first": r(min(firsts)) if firsts else r(caption),
            "caption": r(caption), "guess": bool(hit and guess_shown)}


# 訳の長さの外れ(guards.length_off)で空の履歴で訳し直すか(評価の切り替え用)
LENGTH_RETRY = True


def _nl_key(k, user=()):
    """(文字, 言語) の文字を耳の直し(terms.clean_heard)の後の形に(先読みの鍵を確定の文字とそろえる)。"""
    if not k:
        return k
    return (clean_heard(k[0], k[1], user), k[1])


def _resolve_spk_idx(futs) -> "tuple[str | None, list]":
    """1単位の話者 = 重なった区間の見分けの結果を、重なりの長さで多数決。
    区間の番号(assign_indexed)も返す: 会議の後の付け直し(refine)で記録の行を直すのに使う。"""
    if not futs:
        return None, []
    votes: dict = {}
    idxs: list = []
    for w, f in futs:
        try:
            r = f.result(timeout=1.0)
        except Exception:  # noqa: BLE001
            continue
        if isinstance(r, tuple):
            idxs.append(r[0])
            r = r[1]
        if r:
            votes[r] = votes.get(r, 0.0) + max(w, 0.01)
    return (max(votes.items(), key=lambda kv: kv[1])[0] if votes else None), idxs


def _spk_id(r) -> "str | None":
    """assign_indexed の結果 (番号, id) / id → id。"""
    return r[1] if isinstance(r, tuple) else r


def _resolve_spk(futs) -> "str | None":
    return _resolve_spk_idx(futs)[0]


def js_args(args) -> str:
    """Python の値の列を JS の関数呼び出しの引数列にする。"""
    return ",".join(json.dumps(a, ensure_ascii=False) for a in args)


def _norm_pair(pair, mode: str = "interpret") -> str:
    """UIから来たペア文字列を検証。既知のペア以外は既定ペアに落とす。

    None/"" をそのまま流すと `pair.split("-")` でスレッドが無言死する
    (11/05 実障害: 処理場所chipのクリックが TSUYAKU.pair を undefined に
    してしまい、「会話を始める」が永遠に「起動中」のままになった)。"""
    if isinstance(pair, str) and pair in _PAIRS:
        return pair
    if mode == "notes" and isinstance(pair, str) and pair in _MONO:
        return pair
    return _DEFAULT_PAIR


_SAY_CACHE: dict | None = None

_SAY_LOC_RE = None  # 初回にコンパイル

_VV_CACHE: tuple[float, list] | None = None


def _voicevox_list(url: str) -> list:
    """VOICEVOX の音色一覧。エンジン有無が起動後に変わる(手動起動/停止)ので30秒だけ caching。"""
    global _VV_CACHE
    if not url:
        return []
    now = time.monotonic()
    if _VV_CACHE and now - _VV_CACHE[0] < 30.0:
        return _VV_CACHE[1]
    voices = voicevox_voices(url)
    _VV_CACHE = (now, voices)
    return voices


def _reachable(base: str, timeout: float = 1.2) -> bool:
    """翻訳サーバーが実際に応答するか(切れたSSHトンネルは届いても無応答)。"""
    try:
        import requests
        return requests.get(f"{base}/v1/models", timeout=timeout).status_code == 200
    except Exception:
        return False


def _reachable_stt(url: str, timeout: float = 1.5) -> bool:
    """リモートサーバーの耳(server/remote_stt_server.py)が応答するか。"""
    try:
        return requests.get(f"{url}/health", timeout=timeout).status_code == 200
    except Exception:
        return False


def _wake_stt(url: str) -> bool:
    """リモートサーバーの耳が応答しなければ、とんねるスクリプト(冪等: scp・起動・転送)で起こす。

    耳のサーバーは30分アイドルで自分で終わる。起動画面のまま30分以上置いてから
    「会話を始める」と止まっていて、このMacの耳(認識2.3秒)に落ちていた(11/05 ライブ)。
    """
    if _reachable_stt(url):
        return True
    script = Path(__file__).resolve().parents[2] / "scripts" / "remote_tunnel.sh"
    print("[stt-remote] リモートサーバーの耳が止まっているので起こします", flush=True)
    try:
        subprocess.run(["bash", str(script)], timeout=60, check=False)
    except Exception as e:  # noqa: BLE001
        print(f"[stt-remote] 起こせませんでした({type(e).__name__})", flush=True)
    return _reachable_stt(url)


def _warm_ear(stt) -> None:
    """リモートサーバーの耳に1秒の小さな雑音を一度聞かせておく(11/06)。起こした直後の最初の
    認識は GPU の初回準備で3秒かかっていた(12:49 実会話「こんにちは」stt 3.13秒)。"""
    try:
        if getattr(stt, "url", None):
            t0 = time.time()
            x = (np.random.default_rng(0).standard_normal(16000) * 0.003).astype(np.float32)
            stt._remote(x, None)
            print(f"[stt-remote] 耳の準備 {time.time() - t0:.2f}秒", flush=True)
    except Exception:  # noqa: BLE001 - 準備に失敗しても会話は始める
        pass


def _prewarm_engine(s, ear: bool = False) -> None:
    """選んだ処理場所の先頭を裏で起こす(リモートサーバーゲートのモデル載せは1〜2分ある)。
    ear=True なら リモートサーバーの耳(30分使わないと自分で終わる)も起こす(起動画面から、11/08 speed)。"""
    try:
        lab, base, model = engine_candidates(s.engine, s.host, s.port, s.model,
                                             server=s.server_url,
                                             server_model=s.server_model)[0]
    except Exception:
        return
    if ear and lab == ENGINE_LABELS["remote"] and s.stt_url:
        threading.Thread(target=_wake_stt, args=(s.stt_url,), daemon=True).start()
    try:
        localllm.ensure_for(base)
        warmup(base, model)
    except Exception:
        pass


# 読み込みに1〜3秒かかる手元のモデル(読み上げの声・このMacの耳)を会話をまたいで使い回す(11/08 speed)。
# 見つけた問題: 「会話を始める」→ 聞き始めるまで 3.9〜5.1 秒、うち 1.6〜2.6 秒は声(Kokoro)の読み直し。
# アプリを開いたら裏で先に読み(Bridge._preload)、会話はそれを使う。読み込みの途中で会話を始めたら
# 同じ読み込みを待つ(2回読まない)。
_SHARED: dict = {}
_SHARED_LOCKS = {"tts": threading.Lock(), "ear": threading.Lock()}


def _shared(kind: str, key: tuple, make):
    with _SHARED_LOCKS[kind]:
        ent = _SHARED.get(kind)
        if ent is not None and ent[0] == key:
            return ent[1]
        obj = make()
        _SHARED[kind] = (key, obj)
        return obj


def _shared_tts(s):
    """会話の読み上げ(TTS)。声の場所が同じなら前に読んだものを使い、声・速さだけ今の設定にする。"""
    key = (TTS, s.tts_dir_kokoro, s.tts_dir_ko, tuple(sorted((s.tts_sids or {}).items())),
           s.tts_ja_engine)
    t = _shared("tts", key, lambda: TTS(s.tts_dir_kokoro, s.tts_dir_ko, s.tts_sids, s.tts_ja_engine,
                                        voices=s.tts_voice, speeds=s.tts_speed,
                                        voicevox_url=s.voicevox_url))
    t.voices = dict(s.tts_voice or {})
    t.speeds = {k: float(v) for k, v in (s.tts_speed or {}).items()}
    t.voicevox_url = (s.voicevox_url or "").rstrip("/")
    t.volume = getattr(s, "tts_volume", 1.0)
    t.on_start = None
    if hasattr(t, "new_conversation"):
        t.new_conversation()
    return t


def make_local_stt(s):
    """このMacの耳(stt.make_local_stt)。同じモデルなら前に読んだものを使う(11/08 speed)。"""
    key = (_stt_mod.make_local_stt, getattr(s, "mac_ear", ""), getattr(s, "sense_voice_dir", ""),
           getattr(s, "whisper_model", ""))
    return _shared("ear", key, lambda: _stt_mod.make_local_stt(s))


def parse_say_lines(txt: str) -> dict:
    """`say -v '?'` の出力を {言語: [声名,...]} に直す。

    声名は空白を含むことがある(例: 'Eddy (日本語（日本）)')ため、行全体を
    '#' で割り、左側末尾のトークン(ja_JP / en-US など)をローカル、その手前
    全体を声名とみなす。
    """
    global _SAY_LOC_RE
    if _SAY_LOC_RE is None:
        import re
        _SAY_LOC_RE = re.compile(r"^([a-zA-Z]{2,3})(?:[-_][A-Za-z]{2,4})?$")
    out: dict[str, list[str]] = {"ja": [], "en": [], "zh": [], "ko": []}
    for line in txt.splitlines():
        left = line.split("#", 1)[0].rstrip()
        if not left or " " not in left:
            continue
        name, _, loc = left.rpartition(" ")
        name = name.rstrip()
        m = _SAY_LOC_RE.match(loc or "")
        if not name or not m:
            continue
        lang = m.group(1).lower()
        if lang in out and name not in out[lang]:
            out[lang].append(name)
    return out


def _say_voices() -> dict:
    """このMacに入っている macOS 音声を言語別に列挙する(say -v '?')。"""
    global _SAY_CACHE
    if _SAY_CACHE is not None:
        return _SAY_CACHE
    try:
        import subprocess
        txt = subprocess.run(["say", "-v", "?"], capture_output=True,
                             text=True, timeout=10).stdout
        _SAY_CACHE = parse_say_lines(txt)
    except Exception:
        _SAY_CACHE = {"ja": [], "en": [], "zh": [], "ko": []}
    return _SAY_CACHE


def _screen_op(fn):
    """起動画面の操作(声・マイク・まわりの音など)。操作から15分は処理場所を起こしてよい(11/08 speed)。"""
    import functools

    @functools.wraps(fn)
    def wrap(self, *a, **k):
        try:
            self._screen_touched()
        except Exception:  # noqa: BLE001 - 起こせなくても操作は通す
            pass
        return fn(self, *a, **k)
    return wrap


class Bridge(InMeetingMixin, AudioAPI, LibraryApi):
    """JS(pywebview) ⇔ パイプラインの仲介。パイプラインは別スレッドで回る。
    会議中の道具(印・メモ・議題・修正・用語辞書)は inmeeting.InMeetingMixin、録音・再生・取り込み・字幕は meeting_audio.AudioAPI(W2)、
    整理・検索・分析・書き出し・声・カレンダー(W4)は bridge_library.LibraryApi。"""
    _user_terms: tuple = ()     # 用語辞書(W3)。訳の手がかり(terms.py)
    _heard_terms: tuple = ()    # 用語辞書 + テーマの語。耳の文字の聞き違いの直し

    def __init__(self):
        self.s = load_settings()
        self.window = None
        self.ui: "queue.Queue[tuple]" = queue.Queue()
        self.running = threading.Event()
        self.closing = threading.Event()
        self.capture: SpeechCapture | None = None
        self.tts: TTS | None = None
        self._preview_tts: TTS | None = None
        self.stt: "STT | RemoteSTT | None" = None
        self.log: SessionLog | None = None
        self.history: list[dict] = []
        self.tts_q: "queue.Queue" = queue.Queue()  # 読み上げ行列(会話は再作成)
        self.last_spoken = ""
        self.empty_streak = 0
        self._terms_retry = [0, 0.0]   # 手がかりの語だけ → 手がかりなしの聞き直し [回数, 足した秒]
        self.engine_used = ""  # 会話で実際に使った処理場所(ラベル)
        self.model_used = ""   # 会話で実際に使った言語モデル(表示名)
        self.interrupting = threading.Event()
        self.spec: Speculator | None = None   # 先読み翻訳(会話ごとに作る)
        self.uplink: "Uplink | None" = None  # 先送り(会話ごとに作る)
        self._utt_seq = 0                     # 発話の通し番号(応答速度の表示で読み上げと結ぶ)
        self._mic_test: "MicTest | None" = None   # 起動画面の「マイクの確認」(11/08 hear)

    # ---------- JSから呼ばれる ----------

    def start(self, pair: str, topic: str = "", mode: str | None = None,
              source: "str | dict | None" = None, extra=None):
        if isinstance(source, dict):   # 古い画面: start(pair, topic, mode, extra)
            source, extra = None, source
        self._mic_test_stop()   # 起動画面の「マイクの確認」はデバイスを放してから会話で開く(11/08 hear)
        # 記録の meta に足すもの: 議題・テンプレート(画面から)と、用語辞書の写し(W3)
        self._start_extra = inmeeting.meta_extra(extra)
        terms = inmeeting.load_terms()
        if terms:
            self._start_extra["terms"] = terms
        mode = mode if mode in _MODES else self.s.use_mode
        if mode not in _MODES:
            mode = "interpret"
        pair = _norm_pair(pair, mode)  # UIが undefined を送ると None が来る(11/05固まり事故)
        topic = str(topic or "").strip()
        # 聞く音(11/07): "mic" / "mic+system"(オンライン会議)。来なければ前回のまま
        source = source if source in AUDIO_SOURCES else self.s.audio_source
        if (topic != self.s.last_topic or mode != self.s.use_mode
                or source != self.s.audio_source):  # 次も同じで始められるよう残す
            self.s.last_topic = topic
            self.s.use_mode = mode
            self.s.audio_source = source
            save_config(self.s)
        self._conv_th = threading.Thread(target=self._run, args=(pair, topic, mode), daemon=True)
        self._conv_th.start()
        return "ok"

    def open_log(self):
        """今の会話の記録(.txt)を Finder で見せる(11/06 記録レールの「ファイル」)。
        記録がまだ無ければ記録のフォルダを開く。"""
        try:
            log = self.log
            if log is not None and Path(log.path_txt).exists():
                log.flush()
                subprocess.run(["open", "-R", str(log.path_txt)], check=False, timeout=5)
                return "ok"
            d = Path(self.s.log_dir)
            d.mkdir(parents=True, exist_ok=True)
            subprocess.run(["open", str(d)], check=False, timeout=5)
        except Exception:  # noqa: BLE001
            pass
        return "none"

    # ---------- 会議メモ(11/06 設計: docs/superpowers/specs/2026-11-06-meeting-notes-design.md) ----------

    def _meeting_path(self, mid) -> "Path | None":
        """画面の id → 記録の jsonl。"live" は今の会話。ログのフォルダの外は開かない。"""
        if mid in (None, "", "live"):
            log = self.log
            return Path(log.path_jsonl) if log is not None else None
        name = Path(str(mid)).name
        if not name.startswith("session-"):
            return None
        p = Path(self.s.log_dir) / f"{name.removesuffix('.jsonl')}.jsonl"
        return p if p.exists() else None

    def meetings_list(self):
        """会議の記録の一覧(新しい順)。題の変更・タグ・フォルダ・お気に入りつき(W4 library)。"""
        try:
            return library_mod.list_meetings(self.s.log_dir)
        except Exception:  # noqa: BLE001
            return []

    def meeting_get(self, mid, lang=None):
        """1つの会議: 記録の全行(話者つき)・直した名前・メモ(その言語のものがあれば)。"""
        p = self._meeting_path(mid)
        if p is None:
            return None
        if mid == "live" and (lg := getattr(self, "log", None)) is not None:   # 閉じる瞬間に None になっても落ちない
            lg.flush()
        m = notes_mod.load_meeting(p)
        lg = lang if lang in notes_mod.LANGS else None
        return {"id": m["id"], "live": mid == "live", "meta": m["meta"], "lines": m["lines"],
                "speakers": notes_mod.load_speakers(p), "notes": notes_mod.load_notes(p, lg),
                "notes_langs": notes_mod.notes_langs(p),
                **{k: m[k] for k in ("marks", "memos", "agenda", "template", "undo")}}   # W3

    def notes_make(self, mid, lang="ja", template=None):
        """メモを裏で作る。進み具合と結果は TSUYAKU.notes({id, stage, pct, notes?, error?})。
        template(11/07 W1): 会議の種類を変えて作り直す時。無ければ前のメモ → 記録の meta の種類。"""
        p = self._meeting_path(mid)
        if p is None:
            return "none"
        lang = lang if lang in notes_mod.LANGS else "ja"     # 鍵は整えた言語で(None と "ja" を同じ1本に)
        busy = self.__dict__.setdefault("_notes_busy", set())
        key = (str(p), lang)
        if key in busy:
            return "busy"
        busy.add(key)
        tpl = note_templates.normalize(template) if template else None
        threading.Thread(target=self._make_notes, args=(mid, p, lang, key, tpl), daemon=True).start()
        return "started"

    def _make_notes(self, mid, p: Path, lang: str, key, template=None) -> None:
        def prog(stage: str, x: float) -> None:
            self.ui.put(("notes", {"id": mid, "stage": stage, "pct": round(x, 2)}))
        t0 = time.time()
        self._notes_stamp = t0          # ライブ要約(W1)がこれより古い結果で上書きしないように
        try:
            prog("read", 0.0)
            if mid == "live" and (lg := getattr(self, "log", None)) is not None:   # 閉じる瞬間に None になっても落ちない
                lg.flush()
            if mid == "live":
                self._refine_speakers()      # メモの前に話者を付け直す(AMI 卓上 87.7 → 93.8%)
            base, model = self._notes_llm()
            m = notes_mod.load_meeting(p)
            m["terms"] = inmeeting.normalize_terms(m["terms"] + inmeeting.load_terms())   # 綴りの手がかり(W3)
            old = notes_mod.load_notes(p, lang)
            known = notes_mod.load_speakers(p)
            n = notes_mod.make_notes(
                m, lang, lambda msgs: notes_mod.chat(base, model, msgs), prog,
                model=model.rsplit("/", 1)[-1], known=known,
                old=old, template=template or (old or {}).get("template"))
            with notes_mod.LOCK:       # 待つ間のチェック・名前の変更を引き継いでから書く
                notes_mod.carry_user_edits(n, p, known)
                self._save_notes_present(p, n)
            print(f"[notes] {m['id']} {lang} {len(m['lines'])}行 {time.time() - t0:.1f}秒 "
                  f"やること{len(n['todos'])} 決定{len(n['decisions'])}", flush=True)
            self.ui.put(("notes", {"id": mid, "stage": "done", "pct": 1.0, "notes": n}))
            if mid == "live":   # メモで分かった名前を会話中の字幕にも
                names = {q["id"]: q["name"] for q in n.get("participants", []) if q.get("name")}
                if names:
                    self.ui.put(("speakers", names))
        except Exception as e:  # noqa: BLE001
            print(f"[notes] 失敗 {type(e).__name__}: {e}", flush=True)
            self.ui.put(("notes", {"id": mid, "stage": "error",
                                   "error": f"{type(e).__name__}: {str(e)[:200]}"}))
        finally:
            self._notes_busy.discard(key)

    @staticmethod
    def _save_notes_present(p: Path, n: dict) -> None:
        """LLM を待つ(数分)あいだに会議がゴミ箱へ移されていたら書かない。書くとメモのファイルだけが
        ログのフォルダに残り、ゴミ箱から戻す時に同じ名前があるので戻せなくなる。ゴミ箱へ移す側
        (library.trash)と同じ鍵で、あるかの確かめと書き込みの間に移されないようにする。"""
        with library_mod._lock:
            if not Path(p).exists():
                raise FileNotFoundError("会議の記録がもうありません(ゴミ箱へ移されました)")
            notes_mod.save_notes(p, n)

    def _spk_track(self, uid: int, segs: list, every: int = 15) -> None:
        """会話中にも話者を付け直す(refine は2000区間で0.02秒): every 発話ごとに付け直し、
        画面の字幕・記録の印を直す(TSUYAKU.relabel({発話id: 話者}))。一瞬だけ出た
        「幽霊の話者」(話者9 など)も、ここで近い本物の話者に戻る。"""
        if getattr(self, "spk", None) is None or not segs:
            return
        seen = self.__dict__.setdefault("_utt_segs", {})
        seen[uid] = list(segs)
        if len(seen) % every:
            return
        try:
            ids = self.spk.refine()
        except Exception:  # noqa: BLE001
            return
        fix = {}
        for u, ss in seen.items():
            votes: dict = {}
            for i in ss:
                if 0 <= i < len(ids) and ids[i]:
                    votes[ids[i]] = votes.get(ids[i], 0) + 1
            if votes:
                fix[u] = max(votes.items(), key=lambda kv: kv[1])[0]
        if fix:
            self.ui.put(("relabel", fix))
            if self.log:
                self.log.add_refine(ids)
        self._voice_after_refine(ids)        # 覚えた声と照合(W4)

    def _refine_speakers(self) -> None:
        """会議の後の付け直し(speakers.refine): 発言の少ない「幽霊の話者」を消し、各区間を
        近い話者へ付け直す。結果は記録に refine 行で残し、読む側(notes.load_meeting)が直す。"""
        tr, log = getattr(self, "spk", None), self.log
        if tr is None or log is None:
            return
        try:
            ids = tr.refine()
            if ids:
                log.add_refine(ids)
                self._voice_after_refine(ids)    # 話者の声を残し、覚えた声と照合(W4)
        except Exception as e:  # noqa: BLE001
            print(f"[spk] 付け直しに失敗: {e}", flush=True)

    def _notes_llm(self) -> tuple:
        """メモを作る LLM。会話中ならその LLM、そうでなければ処理場所の順に起こして選ぶ。"""
        llm = getattr(self, "llm", None)
        if llm and self.running.is_set():
            return llm
        s = self.s
        last: Exception | None = None
        for _lab, b, pref in engine_candidates(s.engine, s.host, s.port, s.model,
                                     server=s.server_url, server_model=s.server_model):
            try:
                localllm.ensure_for(b)
                warmup(b, pref)
                return (b, pick_model(b, pref))
            except Exception as e:  # noqa: BLE001
                last = e
        raise RuntimeError(f"メモを作る LLM に接続できません({type(last).__name__ if last else '-'})")

    def speaker_rename(self, mid, spk, name):
        p = self._meeting_path(mid)
        if p is None or not spk:
            return "none"
        notes_mod.rename_speaker(p, str(spk), str(name or ""))
        self._voice_named(p, str(spk), str(name or "").strip())   # 名前を付けた声を覚える(W4)
        if mid == "live" or (self.log is not None and Path(self.log.path_jsonl) == p):
            self.ui.put(("speakers", notes_mod.load_speakers(p)))   # 字幕と記録の印を名前に
        return "ok"

    def todo_toggle(self, mid, index, done, lang=None):
        p = self._meeting_path(mid)
        with notes_mod.LOCK:      # メモ作りの書き込み(作り直しの引き継ぎ)と混ざらないように
            n = notes_mod.load_notes(p, lang) if p is not None else None
            try:
                n["todos"][int(index)]["done"] = bool(done)
            except Exception:  # noqa: BLE001
                return "none"
            notes_mod.save_notes(p, n)
        return "ok"

    # ---------- メモの項目を直す・消す・元に戻す(11/08 右クリック) ----------
    def _notes_item(self, mid, lang, work):
        p = self._meeting_path(mid)
        if p is None:
            return None
        with notes_mod.LOCK:      # メモ作りの書き込み(作り直しの引き継ぎ)と混ざらないように
            n = notes_mod.load_notes(p, lang if lang in notes_mod.LANGS else None)
            if not n:
                return None
            try:
                out = work(n)
            except (ValueError, IndexError, TypeError, KeyError):
                return None
            notes_mod.save_notes(p, n)
        return out

    def notes_item_edit(self, mid, kind, index, fields, lang=None):
        """要約・決定・質問・話題の要点の文、やることの文・担当・期限を直す。直したメモを返す。"""
        return self._notes_item(mid, lang, lambda n: (notes_mod.item_edit(n, kind, index, fields), n)[1])

    def notes_item_delete(self, mid, kind, index, lang=None):
        """項目を消す。元に戻すための値 {item, src, user_edits}(notes_item_restore に渡す)と、消した後のメモ notes を返す。"""
        return self._notes_item(mid, lang, lambda n: dict(notes_mod.item_delete(n, kind, index), notes=n))

    def notes_item_restore(self, mid, kind, index, undo, lang=None):
        return self._notes_item(mid, lang, lambda n: (notes_mod.item_restore(n, kind, index, undo), n)[1])

    def notes_markdown(self, mid, lang=None):
        p = self._meeting_path(mid)
        n = notes_mod.load_notes(p, lang) if p is not None else None
        if not n:
            return ""
        return notes_mod.to_markdown(n, notes_mod.load_meeting(p))

    def notes_export(self, mid, lang=None):
        """メモを .md に書き出して Finder で見せる。書いたパスを返す。"""
        md = self.notes_markdown(mid, lang)
        p = self._meeting_path(mid)
        if not md or p is None:
            return ""
        n = notes_mod.load_notes(p, lang) or {}
        out = Path(f"{p.with_suffix('')}.notes.{n.get('lang', 'ja')}.md")
        out.write_text(md, encoding="utf-8")
        try:
            subprocess.run(["open", "-R", str(out)], check=False, timeout=5)
        except Exception:  # noqa: BLE001
            pass
        return str(out)

    # ---------- W1(11/07): 会議に質問・テンプレート・フォローアップの下書き・ライブ要約 ----------
    # 設計: docs/superpowers/specs/2026-11-07-meeting-features-parity.md(担当 W1)。中身は meeting_ai.py。

    def set_template(self, key):
        """起動画面で選んだ会議の種類。次の会話の記録の meta に入る(config にも残す)。"""
        self.s.meeting_template = note_templates.normalize(key)
        save_config(self.s)
        return "ok"

    def get_template(self):
        return note_templates.normalize(getattr(self.s, "meeting_template", "free"))

    def set_theme(self, v):
        """画面の明るさ: "system"(macOS に合わせる)/ "light" / "dark"。config に残す。"""
        v = v if v in ("system", "light", "dark") else "system"
        self.s.ui_theme = v
        save_config(self.s)
        return v

    def get_theme(self):
        v = getattr(self.s, "ui_theme", "system")
        return v if v in ("system", "light", "dark") else "system"

    def _ask_start(self, kind: str, work) -> str:
        """質問を裏で答える。途中の文は TSUYAKU.ask({qid, delta})、終わりは {qid, done, …}。"""
        self.__dict__["_ask_seq"] = self.__dict__.get("_ask_seq", 0) + 1
        qid = f"q{self._ask_seq}"

        def run():
            buf, last = [], [0.0]

            def od(d: str) -> None:       # 1文字ずつ画面へ送ると重いので 0.08 秒ごとにまとめる
                buf.append(d)
                if time.time() - last[0] >= 0.08:
                    last[0] = time.time()
                    self.ui.put(("ask", {"qid": qid, "delta": "".join(buf)}))
                    buf.clear()
            t0 = time.time()
            try:
                base, model = self._notes_llm()
                def st(stage: str, n: int) -> None:   # 途中の段階(長い会議を探す・会議を読む)
                    self.ui.put(("ask", {"qid": qid, "stage": stage, "n": n}))
                r = work(lambda msgs: notes_mod.chat(base, model, msgs),
                         lambda msgs, cb: meeting_ai.chat_stream(base, model, msgs, cb), od, st)
                print(f"[ask] {kind} {time.time() - t0:.1f}秒 根拠{len(r.get('cites', []))}", flush=True)
                self.ui.put(("ask", dict(r, qid=qid, done=True, scope=kind)))
            except Exception as e:  # noqa: BLE001
                print(f"[ask] 失敗 {type(e).__name__}: {e}", flush=True)
                self.ui.put(("ask", {"qid": qid, "done": True, "scope": kind,
                                     "error": f"{type(e).__name__}: {str(e)[:200]}"}))
        threading.Thread(target=run, daemon=True).start()
        return qid

    def meeting_ask(self, mid, question, lang="ja"):
        """会議に質問(根拠の行つき)。返すのは質問の id(答えは TSUYAKU.ask で届く)。"""
        p = self._meeting_path(mid)
        q = str(question or "").strip()
        if p is None or not q:
            return ""
        lang = lang if lang in notes_mod.LANGS else "ja"
        if mid == "live" and (lg := getattr(self, "log", None)) is not None:   # 閉じる瞬間に None になっても落ちない
            lg.flush()

        def work(llm, final, od, st):
            m = notes_mod.load_meeting(p)
            n = notes_mod.load_notes(p, lang) or notes_mod.load_notes(p)
            return meeting_ai.ask(m, n, q, lang, llm, on_delta=od, final=final,
                                  known=notes_mod.load_speakers(p), on_stage=st)
        return self._ask_start("meeting", work)

    def meetings_ask(self, question, lang="ja"):
        """すべての会議に質問。根拠は [M1#12](答えの meetings[0] の12行目)。"""
        q = str(question or "").strip()
        if not q:
            return ""
        lang = lang if lang in notes_mod.LANGS else "ja"
        if (lg := getattr(self, "log", None)) is not None:   # 閉じる瞬間に None になっても落ちない
            lg.flush()

        def work(llm, final, od, st):
            return meeting_ai.ask_all(self.s.log_dir, q, lang, llm, on_delta=od, final=final, on_stage=st)
        return self._ask_start("all", work)

    def ask_suggest(self, mid, lang="ja"):
        """質問の例(決まったこと・〇〇さんの意見・次にやること)。"""
        p = self._meeting_path(mid)
        n = (notes_mod.load_notes(p, lang) or notes_mod.load_notes(p)) if p is not None else None
        return meeting_ai.suggestions(n, lang if lang in notes_mod.LANGS else "ja",
                                      notes_mod.load_speakers(p) if p is not None else None)

    def notes_draft(self, mid, kind, lang=None, remake=True):
        """フォローアップのメール(email)/ 議事録(minutes)の下書き。メモの言語で作り、メモに残す。"""
        p = self._meeting_path(mid)
        n = notes_mod.load_notes(p, lang) if p is not None else None
        if not n:
            return {"kind": kind, "error": "notes"}
        if not remake and (n.get("drafts") or {}).get(kind):
            return {"kind": kind, "text": n["drafts"][kind], "saved": True}
        try:
            base, model = self._notes_llm()
            text = meeting_ai.draft(n, kind, n.get("lang") or lang or "ja",
                                    lambda msgs: notes_mod.chat(base, model, msgs))
        except Exception as e:  # noqa: BLE001
            print(f"[draft] 失敗 {type(e).__name__}: {e}", flush=True)
            return {"kind": kind, "error": f"{type(e).__name__}: {str(e)[:200]}"}
        try:
            with notes_mod.LOCK:
                n2 = notes_mod.load_notes(p, n.get("lang")) or n     # 待つ間のチェックなどを消さない
                n2.setdefault("drafts", {})[kind] = text
                self._save_notes_present(p, n2)
        except FileNotFoundError as e:
            return {"kind": kind, "error": f"{type(e).__name__}: {e}"}
        return {"kind": kind, "text": text}

    def live_notes(self, on=True, lang=None):
        """会話中のライブ要約の入り・切りと、メモの言語(画面から)。"""
        self._live_on = bool(on)
        if lang in notes_mod.LANGS:
            self._live_lang = lang
        return "ok"

    def _live_idle(self) -> bool:
        """ライブ要約を始めてよいか: 会話中で、訳・聞き取りの途中でなく、ほかのメモ作りもしていない。"""
        if not self.running.is_set() or self.__dict__.get("_notes_busy"):
            return False
        st = getattr(self, "_ui_status", ("listen", 0.0))
        if st[0] == "work" and time.time() - st[1] < 15:
            return False
        tq = getattr(self, "_text_q", None)
        return not getattr(self, "_tw_busy", False) and (tq is None or tq.empty())

    def _live_start(self, mode: str, pair: str) -> None:
        """会議メモを使う会話で、数分ごとにメモを作り直す(1本だけ・手すきの時だけ)。
        通訳+会議メモでは翻訳と同じ LLM を使うので間を長めに(8分 / 60行)。"""
        if mode not in ("notes", "both") or self.log is None:
            return
        self._live_sched = (meeting_ai.LiveSchedule(300, 40) if mode == "notes"
                            else meeting_ai.LiveSchedule(480, 60))
        self._live_cache = None
        if self.__dict__.get("_live_lang") not in notes_mod.LANGS:
            self._live_lang = pair.split("-")[0] if pair.split("-")[0] in notes_mod.LANGS else "ja"
        threading.Thread(target=self._live_loop, args=(self.log,), daemon=True).start()

    def _live_loop(self, log) -> None:
        while self.running.is_set() and self.log is log:
            time.sleep(2.0)
            if not self.__dict__.get("_live_on", True):
                continue
            n = getattr(log, "n_rows", 0)
            if not self._live_sched.due(time.time(), n) or not self._live_idle():
                continue
            self._live_sched.mark(time.time(), n)
            try:
                self._live_once(log)
            except Exception as e:  # noqa: BLE001
                print(f"[live] ライブ要約に失敗 {type(e).__name__}: {e}", flush=True)

    def _live_once(self, log) -> None:
        t0 = time.time()
        p = Path(log.path_jsonl)
        log.flush()
        base, model = self._notes_llm()
        if self._live_cache is None or getattr(self._live_cache, "where", None) != (base, model):
            self._live_cache = meeting_ai.CachedLLM(lambda msgs: notes_mod.chat(base, model, msgs),
                                                    key=meeting_ai.live_key)
            self._live_cache.where = (base, model)
        lang = self._live_lang
        m = notes_mod.load_meeting(p)
        old = notes_mod.load_notes(p, lang)
        known = notes_mod.load_speakers(p)
        n = notes_mod.make_notes(m, lang, self._live_cache, model=model.rsplit("/", 1)[-1],
                                 known=known, old=old, workers=1,
                                 template=(old or {}).get("template"))
        if getattr(self, "_notes_stamp", 0) > t0 or self.log is not log:
            return          # 途中で「メモ」が押された / 会話が終わった: そちらを優先
        n["live"] = True
        with notes_mod.LOCK:       # 作っている間のチェック・名前の変更を引き継ぐ
            notes_mod.carry_user_edits(n, p, known)
            notes_mod.save_notes(p, n)
        print(f"[live] ライブ要約 {len(m['lines'])}行 {time.time() - t0:.1f}秒", flush=True)
        self.ui.put(("notes_live", {"id": "live", "notes": n, "lines": len(m["lines"])}))

    def interrupt(self):
        """Space/クリック: 現在の読み上げを止めてすぐマイクを開ける。
        翻訳中に押されたらこの先の読み上げを飛ばす(字幕のみ)。"""
        self.interrupting.set()
        tts = getattr(self, "tts", None)
        if tts is not None:
            try:
                tts.cancel()
            except Exception:
                pass
        return "ok"

    # ---------- 処理場所・まわりの音の設定(起動画面から) ----------

    PREWARM_EVERY_S = 240.0   # 起動画面から処理場所を起こす間隔(ゲートは15分・耳は30分で止まる)
    PREWARM_FOR_S = 15 * 60.0  # 起動画面を開いた・操作した・処理場所を選び直した時から15分だけ起こす

    def _window_hidden(self) -> bool:
        """窓を最小化した・アプリを隠した(⌘H)か。分からなければ False。"""
        if self.__dict__.get("_minimized"):
            return True
        try:
            from AppKit import NSApplication
            app = NSApplication.sharedApplication()
            return bool(app.isHidden())
        except Exception:  # noqa: BLE001
            return False

    def _screen_touched(self, opened: bool = False) -> None:
        """起動画面を開いた(opened)・操作した: ここから PREWARM_FOR_S の間は処理場所を起こしてよい。
        会話を終えて起動画面に戻っただけ(opened で会話の後)は数えない(リーダー 11/08)。"""
        if opened and self.__dict__.get("_after_conv"):
            return
        self._after_conv = False
        self._warm_until = time.monotonic() + self.PREWARM_FOR_S
        self._maybe_prewarm()

    def _back_from_conversation(self) -> None:
        """会話を終えた: 起動画面に戻っても、操作されるまでは起こさない。"""
        self._after_conv = True
        self._warm_until = 0.0

    def _maybe_prewarm(self, force: bool = False) -> None:
        """選んでいる処理場所(リモートサーバー なら耳も)を裏で起こす(11/08 speed): ゲートが止めた llama-server の載せ
        (1〜2分)と耳の起こし(約6秒)を「会話を始める」の前に済ませる。会話中・窓を隠した・最小化した間・
        起こしてよい15分を過ぎた後は起こさない。PREWARM_EVERY_S に1回まで(force = 処理場所を選び直した)。"""
        if self.running.is_set() or self._window_hidden():
            return
        now = time.monotonic()
        if now > self.__dict__.get("_warm_until", 0.0):
            return
        last = self.__dict__.get("_prewarm_at")
        if not force and last is not None and now - last < self.PREWARM_EVERY_S:
            return
        self._prewarm_at = now
        threading.Thread(target=_prewarm_engine, args=(self.s,), kwargs={"ear": True},
                         daemon=True).start()

    def get_engine(self):
        """現在の処理場所と、各サーバーの実応答状態(起動画面のヒント用)。
        起動画面を開いた合図でもある: 処理場所を起こす(_screen_touched、会話を終えた後は除く)。"""
        s = self.s
        self._screen_touched(opened=True)
        port = s.port if s.port not in (REMOTE_PORT, MAC_PORT) else None
        cands = engine_candidates("auto", s.host, port, s.model if port else None,
                                  server=s.server_url, server_model=s.server_model)
        reach = {lab: _reachable(b) for lab, b, _m in cands}
        models = {lab: (m or "").rsplit("/", 1)[-1] for lab, _b, m in cands}
        return {"engine": s.engine if s.engine in ENGINE_LABELS else "auto",
                "labels": dict(ENGINE_LABELS), "reachable": reach, "models": models}

    def get_model(self):
        """会話中に実際に使っている処理場所とモデル名(バッジ表示用)。未接続なら空。"""
        return {"engine": self.engine_used, "model": self.model_used}

    def set_engine(self, engine: str):
        """処理場所を選ぶと、そのサーバーを裏で起こす(会話開始時に詰まらない)。"""
        if engine not in ENGINE_LABELS:
            return "bad"
        self.s.engine = engine
        save_config(self.s)
        self._after_conv = False
        self._warm_until = time.monotonic() + self.PREWARM_FOR_S
        if self.running.is_set():   # 会話中は今までどおり(次の会話の処理場所を起こす)
            threading.Thread(target=_prewarm_engine, args=(self.s,), daemon=True).start()
        else:
            self._maybe_prewarm(force=True)
        return "ok"

    @_screen_op
    def set_noise(self, level: str):
        """まわりの音: noisy=騒がしい会場(VAD感度を下げる)。次の会話から効く。"""
        self.s.vad_threshold = 0.65 if level == "noisy" else 0.5
        save_config(self.s)
        return "ok"

    @_screen_op
    def set_term_domain(self, choice: str):
        """用語の分野(起動画面): "" 自動 / "it" IT・技術 / "general" 一般。次の会話から効く(11/08 terms.py)。"""
        from .terms import TERM_DOMAINS
        if choice not in TERM_DOMAINS:
            return "bad"
        self.s.term_domain = choice
        save_config(self.s)
        return "ok"

    @_screen_op
    def set_voice_on(self, on):
        """読み上げ: True=声も出す / False=字幕だけ(マイクを止めない)。会話中でもすぐ効く。"""
        self.s.voice_on = bool(on)
        save_config(self.s)
        if not self.s.voice_on and self.tts is not None:
            try:
                self.tts.cancel()
            except Exception:
                pass
        if self.running.is_set():
            self.ui.put(("status", "listen", "読み上げを戻しました" if self.s.voice_on
                         else "字幕だけにしました — 続けて話せます"))
        return "ok"

    # ---------- 声・話速の設定(UIから実行中にも変更可) ----------

    def get_tts(self):
        """UIへの現在設定と選択肢。voices={lang:"say:Kyoko"|"kokoro:3"}, speeds={lang:1.0}。"""
        s = self.s
        return {
            "voices": dict(s.tts_voice),
            "speeds": {k: float(v) for k, v in s.tts_speed.items()},
            "speedModes": dict(s.tts_speed_mode),                 # 11/08 話のペースに合わせる
            "listenWhileSpeaking": s.listen_while_speaking,
            "volume": float(getattr(s, "tts_volume", 1.0)),     # 11/10 読み上げの音量
            "speedPriority": s.tts_speed_priority,                # 聞きやすさ優先 / 速さ優先
            # 読み上げの出力が耳元か(True)・部屋に鳴るか(False)・分からないか(None)。起動画面の
            # 「マイクだけ」の案内は False(内蔵スピーカーなど)のときだけ出す
            "outputPrivate": devices.output_is_private(getattr(s, "output_device", "") or ""),
            "kokoro": bool(s.tts_dir_kokoro),
            "sayVoices": _say_voices(),
            "voicevox": _voicevox_list(s.voicevox_url),
            "vadThreshold": float(s.vad_threshold),
            "topic": s.last_topic,
            "voiceOn": bool(s.voice_on),
            "useMode": s.use_mode,
            "audioSource": s.audio_source,
            "termDomain": s.term_domain,            # 11/08 IT 用語
            "edgeGlow": bool(s.edge_glow),          # 11/08 hear
            "showSysmon": bool(getattr(s, "show_sysmon", True)),   # 11/10 使用率の表示
            "inputDevice": s.input_device,
            "outputDevice": s.output_device,
        }

    # ---------- 入力・出力デバイスと「聞こえ」(11/08 hear) ----------

    def audio_devices(self):
        """入力・出力デバイスの今の一覧(メニューを開くたびに呼ぶ。CoreAudio に聞き直す)。"""
        s = self.s
        ins, outs = devices.list_devices(devices.INPUT), devices.list_devices(devices.OUTPUT)
        dflt = lambda ds: next((d["name"] for d in ds if d.get("isDefault")), "")   # noqa: E731
        names = lambda ds: {d["name"] for d in ds}   # noqa: E731
        cap = self.capture
        return {"inputs": [{"name": d["name"], "isDefault": bool(d.get("isDefault"))} for d in ins],
                "outputs": [{"name": d["name"], "isDefault": bool(d.get("isDefault"))} for d in outs],
                "input": s.input_device, "output": s.output_device,
                "defaultInput": dflt(ins), "defaultOutput": dflt(outs),
                "inputMissing": bool(s.input_device) and s.input_device not in names(ins),
                "outputMissing": bool(s.output_device) and s.output_device not in names(outs),
                "liveInput": getattr(cap or self.__dict__.get("_mic_test"), "device_used", "") or ""}

    @_screen_op
    def set_input_device(self, name):
        """マイクを選ぶ(名前。"" = システムの設定に合わせる)。会話中・確認中ならストリームだけ開き直す
        (会議は続く)。そうでなければ次の会話から。"""
        name = str(name or "")
        self.s.input_device = name
        save_config(self.s)
        live = self.capture if self.running.is_set() else None
        live = live or self.__dict__.get("_mic_test")
        if live is not None and hasattr(live, "switch_device"):
            live.switch_device(name)
            return {"ok": True, "applied": "now"}
        return {"ok": True, "applied": "next"}

    @_screen_op
    def set_output_device(self, name):
        """スピーカーを選ぶ。読み上げ(afplay の代わりの AVAudioPlayer / say -a)は次の読み上げから。"""
        name = str(name or "")
        self.s.output_device = name
        save_config(self.s)
        for t in (self.tts, self.__dict__.get("_preview_tts")):
            if t is not None:
                t.output_device = name
        self.ui.put(("audio_dev", {"kind": "output", "event": "switched", "want": name,
                                   "using": name, "applied": "next"}))
        return {"ok": True, "applied": "next"}

    @_screen_op
    def mic_test(self, on=True):
        """起動画面の「マイクの確認」。起動画面が見えている間だけ開く(会話中は会話のマイクを見る)。"""
        if not on:
            self._mic_test_stop()
            return "ok"
        if self.running.is_set():
            return "busy"
        if self.__dict__.get("_mic_test") is None:
            m = MicTest(self.s.input_device)
            m.on_device = lambda ev, want, using: self._on_audio_dev("input", ev, want, using)
            self._mic_test = m
            m.start()
        return "ok"

    def _mic_test_stop(self) -> None:
        m, self._mic_test = self.__dict__.get("_mic_test"), None
        if m is not None:
            try:
                m.stop()
            except Exception:  # noqa: BLE001
                pass

    def _on_audio_dev(self, kind: str, event: str, want: str, using: str) -> None:
        """デバイスの出来事(選んだものが無い・外れた・替えた・開けない)を画面へ。"""
        if kind == "input" and event == "lost":
            self._hear_lost_until = time.time() + 4.0
        applied = "now" if kind == "input" else "next"
        self.ui.put(("audio_dev", {"kind": kind, "event": event, "want": want or "",
                                   "using": using or "", "applied": applied}))

    def set_edge_glow(self, on):
        """声を聞き取っている間の窓の縁の光(画面の設定)。"""
        self.s.edge_glow = bool(on)
        save_config(self.s)
        return "ok"

    def get_sysmon(self):
        """画面の隅の使用率({cpu,mem,gpu} の取れた欄だけ)。設定でオフなら取得せず None(11/10)。"""
        if not getattr(self.s, "show_sysmon", True):
            return None
        sm = getattr(self, "_sysmon", None)
        if sm is None:
            sm = self._sysmon = sysmon.Sysmon()
        try:
            return sm.sample()
        except Exception:
            return None

    def set_sysmon(self, on):
        self.s.show_sysmon = bool(on)
        save_config(self.s)
        return "ok"

    def term_add(self, term):
        """右クリック「用語に登録」: 用語辞書に1語足す(同じ語は足さない)。"""
        t = str(term or "").strip()[:80]
        if not t:
            return {"ok": False}
        cur = inmeeting.load_terms()
        if t.lower() not in {x.lower() for x in cur}:
            cur = inmeeting.save_terms(cur + [t])
        return {"ok": True, "terms": cur}

    def speak_again(self, text, lang):
        """右クリック「もう一度読み上げ」: 会話中だけ、その訳をもう一度読む(マイクは読み上げの間止める)。"""
        tts, text = self.tts, str(text or "").strip()[:600]
        if tts is None or not text:
            return "idle"
        def _go():
            cap = self.capture
            if cap:
                cap.paused.set()
            try:
                tts.say(text, str(lang or "ja"))
            except Exception:
                pass
            finally:
                if cap:
                    try:
                        cap.resume_after_tts(self.s.gate_tail_s)
                    except Exception:
                        pass
        threading.Thread(target=_go, daemon=True).start()
        return "ok"

    def open_mic_privacy(self):
        """システム設定のマイクの許可を開く(音が完全な 0 = 許可が無い時の帯のボタン)。"""
        try:
            subprocess.run(["open", "x-apple.systempreferences:com.apple.preference.security?"
                            "Privacy_Microphone"], check=False, timeout=5)
        except Exception:  # noqa: BLE001
            pass
        return "ok"

    @_screen_op
    def set_tts(self, lang: str, voice: str | None = None, speed: float | None = None):
        s = self.s
        if voice:
            s.tts_voice[lang] = voice
            if self.tts:
                self.tts.set_voice(lang, voice)
        if speed is not None:
            s.tts_speed[lang] = max(0.5, min(2.0, float(speed)))
            if self.tts:
                self.tts.set_speed(lang, s.tts_speed[lang])
        save_config(self.s)
        return "ok"

    def set_volume(self, v):
        """読み上げの音量(0.1〜1.0)。次の読み上げから効く。"""
        from .tts import clamp_volume
        self.s.tts_volume = clamp_volume(v)
        if self.tts:
            self.tts.set_volume(self.s.tts_volume)
        save_config(self.s)
        return self.s.tts_volume

    @_screen_op
    def set_tts_mode(self, lang: str, mode: str):
        """読み上げの速さの決め方(11/08): "auto" = 話のペースに合わせる / "fixed" = つまみの速さのまま。"""
        if lang in self.s.tts_speed_mode and mode in ("auto", "fixed"):
            self.s.tts_speed_mode[lang] = mode
            save_config(self.s)
        return "ok"

    @_screen_op
    def set_tts_priority(self, p: str):
        """自動の速さの上限(11/08): "clear" 聞きやすさ優先(測った上限)/ "fast" 速さ優先。"""
        if p in ("clear", "fast"):
            self.s.tts_speed_priority = p
            save_config(self.s)
        return "ok"

    @_screen_op
    def set_listen(self, mode: str):
        """読み上げ中も聞く(11/08): "auto" / "always" / "off"。次の読み上げから効く。"""
        if mode in ("auto", "always", "off"):
            self.s.listen_while_speaking = mode
            save_config(self.s)
        return "ok"

    @_screen_op
    def preview_tts(self, lang: str, voice: str, speed: float):
        """試聴。会話中最中はそのTTSをそのまま使い(マイク一時停止)、
        起動画面では専用に1つだけ遅延生成して読む。"""
        def _go():
            tts = self.tts
            if tts is None:
                if self._preview_tts is None:
                    # 会話と同じ声(先に読んだもの)を使う(11/08 speed: 試聴のたびに声を読み直さない)
                    self._preview_tts = _shared_tts(self.s)
                tts = self._preview_tts
            tts.output_device = self.s.output_device   # 試聴も選んだスピーカーで(11/08 hear)
            cap = self.capture if self.tts is not None else None
            if cap:
                cap.paused.set()
            try:
                tts.set_voice(lang, str(voice))
                tts.set_speed(lang, float(speed))
                tts.say(str(_PREVIEW_TEXT.get(lang, "こんにちは。")), lang)
            finally:
                if cap:
                    try:
                        cap.resume_after_tts(self.s.gate_tail_s)
                    except Exception:
                        pass
        threading.Thread(target=_go, daemon=True).start()
        return "ok"

    def end_meeting(self, settle_s: float = 2.5, max_s: float = 20.0):
        """「会議を終える」(11/06): アプリは閉じずに会話を終える。録音を止め、最後の発言が
        記録に書き終わるのを待ち(続き待ち1.2秒 + リモートサーバーの耳 + 訳)、話者を付け直して記録を
        閉じる。画面は返った id の会議を開き、会議メモを作る(notes_make)。

        待ち方: 区間と訳の行列が空・訳の担当が手すき・画面の状態が「翻訳中」でない、が
        settle_s 秒続いて記録の行数も変わらなければ終わり。最長 max_s 秒。"""
        if not self.running.is_set():
            return {"id": None}
        log = self.log
        mode = getattr(self, "mode", "interpret")
        self._drain(settle_s, max_s)
        self.running.clear()
        th = getattr(self, "_conv_th", None)
        if th is not None and th is not threading.current_thread():
            th.join(timeout=12)
        if not self.__dict__.get("_refined_end"):
            self._refined_end = True
            self._refine_speakers()
        for closer in (getattr(self, "tts", None), log):
            try:
                if closer is not None:
                    (closer.stop if hasattr(closer, "stop") else closer.close)()
            except Exception:  # noqa: BLE001
                pass
        self.capture = self.tts = self.log = None
        self.spk = None
        n = getattr(log, "n_rows", 0) if log is not None else 0
        mid = Path(log.path_jsonl).stem if log is not None else None
        print(f"[end] 会議を終えました {mid} {n}行 ({mode})", flush=True)
        return {"id": mid, "lines": n, "mode": mode}

    def _drain(self, settle_s: float, max_s: float) -> None:
        """会話を閉じる前の流し切り(「会議を終える」と会話中の「終了」で共通)。読み上げを止め、
        マイクを止め(話し途中の区間は捨てずに渡す)、録音を閉じ、最後の発言が記録に書き終わるのを待つ。

        待ち方: 区間と訳の行列が空・訳の担当が手すき・画面の状態が「翻訳中」でない、が
        settle_s 秒続いて記録の行数も変わらなければ終わり。最長 max_s 秒。"""
        log, cap = self.log, self.capture
        self.interrupt()                  # 読み上げは止める
        tq = getattr(self, "tts_q", None)
        while tq is not None and not tq.empty():
            try:
                tq.get_nowait()
            except queue.Empty:
                break
        if cap is not None:
            cap.paused.set()              # 話し途中の区間は捨てずに渡す(on_chunk が流す)
            time.sleep(0.15)
            try:
                cap.stop()
            except Exception:  # noqa: BLE001
                pass
        self._rec_stop()                  # 録音を閉じて裏で m4a に(W2)
        seg_q, text_q = getattr(self, "_seg_q", None), getattr(self, "_text_q", None)
        t_end = time.time() + max_s
        quiet_since, rows = time.time(), getattr(log, "n_rows", 0)
        while time.time() < t_end:
            n = getattr(log, "n_rows", 0)
            st = getattr(self, "_ui_status", ("listen", 0.0))
            busy = ((seg_q is not None and not seg_q.empty())
                    or (text_q is not None and not text_q.empty())
                    or getattr(self, "_tw_busy", False) or n != rows
                    or (st[0] == "work" and time.time() - st[1] < 15))
            if busy:
                quiet_since, rows = time.time(), n
            elif time.time() - quiet_since >= settle_s:
                break
            time.sleep(0.05)

    def stop(self):
        """「終了」ボタン/Esc。マイクと声を止め、ログを書き出して窓を閉じる。"""
        self._shutdown()
        if self.window:
            try:
                self.window.destroy()
            except Exception:
                pass

    def _on_closing(self):
        """pywebview の closing(窓を閉じる・⌘Q の terminate:・Dock の「終了」)。どれもここを通して後片付けを
        済ませる(11/08: ⌘Q は NSApp がプロセスをそのまま終わらせ、_shutdown を通らなかった)。閉じるのは止めない。"""
        try:
            self._shutdown()
        except Exception as e:  # noqa: BLE001 - 閉じることを優先する
            print(f"[quit] 後片付けに失敗 {type(e).__name__}: {e}", flush=True)

    def menu_state(self, state):
        """画面(JS)から: メニューバーに要る状態(画面の言語・会話中か・会議を開いているか・明るさ…)。"""
        mb = self.__dict__.get("_menubar")
        if mb is not None:
            mb.set_state(state)
        return "ok"

    def _shutdown(self, max_wait_s: float = 5.0):
        """窓以外の後片付け(何度呼んでもよい)。窓が閉じた後の main からも呼ぶ。

        会話中なら「会議を終える」と同じく、訳している途中の最後の発言を記録に書き終えるまで
        待つ(11/07 監査: すぐ記録を閉じていたので最後の1〜2発言が失われた)。閉じるのが
        固まらないよう最長 max_wait_s 秒(続き待ちの断片は1.2秒で出るので、落ち着き1.5秒)。"""
        if self.running.is_set() and getattr(self, "log", None) is not None:
            try:
                self._drain(min(1.5, max_wait_s), max(0.0, max_wait_s - 0.15))
            except Exception as e:  # noqa: BLE001 - 閉じることを優先する
                print(f"[quit] 流し切りに失敗 {type(e).__name__}: {e}", flush=True)
        self.running.clear()
        self.closing.set()
        self._mic_test_stop()
        if not self.__dict__.get("_refined_end"):   # 会話の終わりに話者を付け直して記録に残す
            self._refined_end = True
            self._refine_speakers()
        self._rec_stop()   # 録音を閉じる(m4a への変換が間に合わなければ次の起動で直す)
        self._import_stop()   # 取り込み中なら中止して作りかけを消す(W2)
        for closer in (getattr(self, "capture", None), getattr(self, "tts", None),
                       getattr(self, "log", None)):
            try:
                if closer is None:
                    continue
                (closer.stop if hasattr(closer, "stop") else closer.close)()
            except Exception:
                pass

    # ---------- UIへの送信(メインスレッドへはpywebviewが転送する) ----------

    def _push(self, evt: tuple):
        kind = evt[0]
        try:
            if kind == "status":
                self._ui_status = (evt[1], time.time())   # 「会議を終える」: 聞き取り・訳の途中か
                js = (f"TSUYAKU.status({json.dumps(evt[1])},{json.dumps(evt[2], ensure_ascii=False)}"
                      + (f",{json.dumps(evt[3], ensure_ascii=False)}" if len(evt) > 3 else "") + ")")
            elif kind == "level":
                js = f"TSUYAKU.level({float(evt[1]):.3f})"
            elif kind == "utterance":
                js = "TSUYAKU.utterance(" + json.dumps(evt[1], ensure_ascii=False) + ")"
            elif kind in ("partial", "error"):
                # 引数は1つずつ並べる。以前は配列1個で渡していて、JS 側の
                # partial(orig,trans,lang) に配列がまるごと orig として届いていた(11/06 発見)。
                js = f"TSUYAKU.{kind}(" + js_args(evt[1:]) + ")"
            elif kind == "voice":
                js = "TSUYAKU.voice(" + json.dumps(evt[1], ensure_ascii=False) + ")"
            elif kind == "fix":   # 聞き直しで直した行 / 確認の印を外す(relisten.py)
                js = ("window.TSUYAKU&&TSUYAKU.fix&&TSUYAKU.fix("
                      + json.dumps(evt[1], ensure_ascii=False) + ")")
            elif kind == "relabel":   # 会話中の付け直し: {発話id: 話者}
                js = ("window.TSUYAKU&&TSUYAKU.relabel&&TSUYAKU.relabel("
                      + json.dumps(evt[1], ensure_ascii=False) + ")")
            elif kind == "speakers":   # 話者の名前(会話中に直した/メモで分かった)
                js = ("window.TSUYAKU&&TSUYAKU.speakers&&TSUYAKU.speakers("
                      + json.dumps(evt[1], ensure_ascii=False) + ")")
            elif kind == "notes":   # 会議メモの進み具合・結果
                js = ("window.TSUYAKU&&TSUYAKU.notes&&TSUYAKU.notes("
                      + json.dumps(evt[1], ensure_ascii=False) + ")")
            elif kind == "w2":   # 録音・取り込み・ドロップ(meeting_audio.py)
                js = ("window.TSUYAKU&&TSUYAKU.w2&&TSUYAKU.w2("
                      + json.dumps(evt[1], ensure_ascii=False) + ")")
            elif kind == "ask":   # W1: 会議に質問の答え(途中の文・終わり)
                js = ("window.TSUYAKU&&TSUYAKU.ask&&TSUYAKU.ask("
                      + json.dumps(evt[1], ensure_ascii=False) + ")")
            elif kind == "notes_live":   # W1: 会話中のライブ要約ができた
                js = ("window.TSUYAKU&&TSUYAKU.notesLive&&TSUYAKU.notesLive("
                      + json.dumps(evt[1], ensure_ascii=False) + ")")
            elif kind == "hear":   # 聞こえのメーターと状態(11/08 hear)
                js = ("window.TSUYAKU&&TSUYAKU.hear&&TSUYAKU.hear("
                      + json.dumps(evt[1], ensure_ascii=False) + ")")
            elif kind == "audio_dev":   # デバイスの出来事(11/08 hear)
                js = ("window.TSUYAKU&&TSUYAKU.audioDev&&TSUYAKU.audioDev("
                      + json.dumps(evt[1], ensure_ascii=False) + ")")
            elif kind == "net":   # 回線の混み具合(JS 側に net が無い古い画面でも落ちない)
                js = ("window.TSUYAKU&&TSUYAKU.net&&TSUYAKU.net("
                      + json.dumps(evt[1], ensure_ascii=False) + ")")
            elif kind == "reset":
                js = "TSUYAKU.reset()"
            elif kind == "engine":
                js = "TSUYAKU.engineInfo(" + json.dumps(evt[1], ensure_ascii=False) + ")"
            elif kind == "boot":
                js = "TSUYAKU.boot(" + json.dumps(evt[1], ensure_ascii=False) + ")"
            else:
                return
            self.window.evaluate_js(js)
        except Exception:
            pass

    def _pump(self):
        last_level = 0.0
        while not self.closing.is_set():
            try:
                while True:
                    self._push(self.ui.get_nowait())
            except queue.Empty:
                pass
            now = time.time()
            # 一度だけ読む: 「会議を終える」が別のスレッドで self.capture を None にした瞬間に
            # 当たると AttributeError でこの係が死に、以後の字幕・状態が画面に出なくなる(11/07 監査)
            cap = self.capture
            if cap and now - last_level > 0.09:
                last_level = now
                # 読み上げでマイクが閉いているあいだは計を 0 に落とす。
                # 最後の値が凍結したままだと「聞いているのに応答がない」と
                # 誤解される(11/05「翻訳中は録音されていないのでは」)。
                try:
                    lv = 0.0 if cap.paused.is_set() else cap.level
                    self._push(("level", lv))
                except Exception:  # noqa: BLE001
                    pass
            self._pump_hear(cap, now)
            time.sleep(0.03)

    def _pump_hear(self, cap, now: float) -> None:
        """「聞こえ」のメーターと状態(11/08 hear)。会話のマイク、無ければ起動画面の「マイクの確認」。
        HearThrottle で最大16回/秒・変わった時だけ。どちらも無くなったら静かを1回送って止める。"""
        test = False
        if not cap:
            cap = self.__dict__.get("_mic_test")
            test = cap is not None
        th = self.__dict__.get("_hear_th")
        if th is None:
            th = self._hear_th = HearThrottle()
        try:
            if not cap:
                if self.__dict__.get("_hear_on"):
                    self._hear_on = False
                    self._push(("hear", {"lv": 0.0, "st": "quiet", "test": False, "mic": ""}))
                return
            st = hear_state(cap, getattr(self, "_ui_status", None), now,
                            self.__dict__.get("_hear_lost_until", 0.0))
            peak = self.__dict__.get("_hear_peak", 0.0)
            take = getattr(cap, "take_meter", None)
            peak = max(peak, take() if take else 0.0)
            lv = 0.0 if st in ("speak", "nomic") else round(peak, 3)
            self._hear_peak = peak
            if th.due(now, lv, st):
                self._hear_peak = 0.0
                self._hear_on = True
                self._push(("hear", {"lv": lv, "st": st, "test": test,
                                     "mic": getattr(cap, "device_used", "") or ""}))
        except Exception:  # noqa: BLE001
            pass

    def _started(self, window):
        self.window = window
        threading.Thread(target=self._pump, daemon=True).start()
        threading.Thread(target=self._preload, daemon=True).start()
        try:   # 最小化した・戻した(最小化の間は処理場所を起こさない)
            window.events.minimized += lambda *a: setattr(self, "_minimized", True)
            window.events.restored += lambda *a: setattr(self, "_minimized", False)
        except Exception:  # noqa: BLE001
            pass
        self._screen_touched()   # アプリを開いた
        self._w2_started(window)

    def _preload(self) -> None:
        """アプリを開いたら、会話で使う手元のモデル(このMacの耳・読み上げの声)を裏で先に読む(11/08 speed)。"""
        s = self.s
        t0 = time.time()
        try:
            make_local_stt(s)
            if s.use_mode != "notes":
                _shared_tts(s)
            print(f"[preload] 耳と声を先に読みました {time.time() - t0:.1f}秒", flush=True)
        except Exception as e:  # noqa: BLE001 - 会話を始めたときに読み直す
            print(f"[preload] 先に読めませんでした({type(e).__name__})", flush=True)

    # ---------- パイプライン ----------

    # 会話開始時の読み込みの段(UIの段付きバーと同じ順)。リモートサーバーの初回は 0 段目の
    # モデル載せに1〜2分かかるので、どこで待っているかを見せる。
    BOOT_STEPS = ("翻訳モデル", "音声認識", "読み上げの声", "マイク")

    def _boot(self, step: int, detail: str = "") -> None:
        self.ui.put(("boot", {"step": step, "total": len(self.BOOT_STEPS),
                              "label": self.BOOT_STEPS[step] if step < len(self.BOOT_STEPS)
                              else "", "detail": detail}))

    def _is_conv(self, gen=None) -> bool:
        """gen の会話がまだ今の会話か(None = 確かめない)。"""
        return gen is None or getattr(self, "_conv_gen", 0) == gen

    def _conv_alive(self, gen=None) -> bool:
        """gen の会話が続いているか。running だけを見ると、「会議を終える」→ すぐ次の会話で
        running がまた立つので、前の会話のスレッド(keepalive・詰まっていた訳の担当)が
        次の会話の中でも動き続けた(11/07 監査: 前の処理場所を起こし続ける・前の会議の
        発言を次の会議の記録に書く)。"""
        return self.running.is_set() and self._is_conv(gen)

    def _keepalive(self, base: str, model: str, stt_url: str | None = None, gen=None,
                   prompt: "tuple | None" = None):
        """会話中、翻訳サーバーを時々生かしてモデルを載せっぱなしにする。

        リモートサーバーのゲートは15分無操作だと止まるので、これが無いと会議の途中で
        落ちて、次の一言にモデル載せから1〜2分かかる。480秒(8分)間隔なら
        止まる前に必ず触る。リモートサーバーの耳(30分アイドルで自分で終わる)も同時に触る。
        """
        while self._conv_alive(gen):
            time.sleep(480)
            if not self._conv_alive(gen):
                return
            try:
                if prompt is not None:
                    # 会話の指示文で触る("ping" だとこのMacの LM Studio の読み込み済みの指示文を追い出す)
                    warm_prompt(base, model, *prompt, timeout=30)
                else:
                    warmup(base, model, timeout=30)
            except Exception:
                pass
            if stt_url:
                try:
                    requests.get(f"{stt_url}/health", timeout=3)
                except Exception:
                    pass

    def _run(self, pair: str, topic: str, mode: str = "interpret"):
        try:
            self._conversation(pair, topic, mode)
        except Exception as e:  # noqa: BLE001
            # スレッドが無言で死むと画面は「起動しています…」のまま回りっぱなし
            # になる(11/05 実障害)。「遅い」ではなく「何が起きた」を見せて戻す。
            self.running.clear()
            for closer in (self.capture, self.tts):  # マイク/声は開けたまま戻さない
                try:
                    if closer:
                        closer.stop()
                except Exception:
                    pass
            self.capture = self.tts = None
            self._rec_stop()
            log, self.log = getattr(self, "log", None), None   # 記録のファイルも開けたまま残さない
            if log is not None:
                try:
                    log.close()
                except Exception:  # noqa: BLE001
                    pass
            self.ui.put(("error", "会話を始められませんでした",
                         f"{type(e).__name__}: {e} — もう一度「会話を始める」を押してください"))
            self.ui.put(("reset",))

    def _conversation(self, pair: str, topic: str, mode: str = "interpret"):
        s = self.s
        pair_t = tuple(pair.split("-"))   # 会議メモだけ の1言語なら ("ja",)
        self.mode = mode
        self.history = []
        self.last_spoken = ""
        self.empty_streak = 0
        self._terms_retry = [0, 0.0]   # 手がかりの語だけ → 手がかりなしの聞き直し [回数, 足した秒]
        self._refined_end = False     # 「会議を終える」の後の次の会話でも終わりに付け直す
        self._utt_segs = {}
        self._conv_gen = gen = getattr(self, "_conv_gen", 0) + 1   # この会話の番号
        self.running.set()

        # 翻訳サーバー: 起動画面の『翻訳の処理場所』順に試す(自動は remote→このMac)。
        # リモートサーバーのゲートは止まっていると /v1/models にすら応答しないので、warmupで
        # 先に起こしてからモデルを選ぶ(warmupの1発でモデルがGPUに載るまで1〜2分)。
        self.ui.put(("status", "work", "接続中"))
        base = model = None
        last_err: Exception | None = None
        cands = engine_candidates(s.engine, s.host, s.port, s.model,
                                     server=s.server_url, server_model=s.server_model)
        for lab, b, pref in cands:
            if len(cands) > 1:
                self.ui.put(("status", "work", f"接続中 — {lab}"))
            self._boot(0, f"{lab} — {(pref or '').rsplit('/', 1)[-1]}")
            localllm.ensure_for(b)
            warmup(b, pref)
            try:
                base, model = b, pick_model(b, pref)
                translate_mod.configure_for_model(model)
                break
            except Exception as e:  # noqa: BLE001
                last_err = e
        if not self._conv_alive(gen):   # 読み込み(1〜2分)の間に「会議を終える」/ 次の会話
            self._boot_cancelled()
            return
        if base is None:
            self.ui.put(("error", "翻訳サーバーに接続できません",
                         "起動画面の『翻訳の処理場所』を見直してください(リモートサーバーなら "
                         "scripts/remote_tunnel.sh、このMacなら LM Studio のサーバー起動)。",
                         type(last_err).__name__ if last_err else "unknown"))   # 訳す文と詳しい値は分けて送る
            self.ui.put(("reset",))
            self.running.clear()
            return
        if model != pref:  # ループは成功候補で抜けているので pref はその推奨モデル
            self.ui.put(("status", "work", "モデル読み込み中"))
            warmup(base, model)  # 指定と違うモデルに落ちたときも載せておく
            if not self._conv_alive(gen):
                self._boot_cancelled()
                return
        self.llm = (base, model)   # 会議メモもこの LLM で作る
        self.engine_used = lab  # 実際に繋がった処理場所
        self.model_used = model.rsplit("/", 1)[-1]  # 接頭辞なしのモデル名で表示
        self.ui.put(("engine", {"engine": self.engine_used, "model": self.model_used}))
        # 耳(音声認識): 翻訳を リモートサーバー でやるときは耳も リモートGPU へ(このMac 2.3秒 →
        # リモートサーバー 0.13〜0.26秒、11/05実測)。『このMac』は完全オフラインの約束なので
        # リモートを使わない。ローカル whisper は落ちたときの受け皿として常に載せる。
        stt_url = None
        self._boot(1)
        if lab == ENGINE_LABELS["remote"] and s.stt_url and _wake_stt(s.stt_url):
            stt_url = s.stt_url
        # 指示文を先に読ませる(11/08 speed): 最初の1文の初字が指示文の読み込みを待たない
        warm = (pair_t, topic, tuple((getattr(self, "_start_extra", None) or {}).get("terms", ())))
        if len(pair_t) == 2 and mode != "notes":
            threading.Thread(target=warm_prompt, args=(base, model, *warm), daemon=True).start()
        threading.Thread(target=self._keepalive, args=(base, model, stt_url, gen),
                         kwargs={"prompt": warm if len(pair_t) == 2 and mode != "notes" else None},
                         daemon=True).start()
        try:
            stt = RemoteSTT(stt_url, make_local_stt(s), hedge_s=s.stt_hedge_s,
                            preview_s=s.stt_preview_s)
        except Exception as e:  # noqa: BLE001
            if not self._conv_alive(gen):
                self._boot_cancelled()
                return
            self.ui.put(("error", "音声認識モデルを読み込めません",
                         "models/ggml-large-v3-turbo.bin を確認してください。", str(e)))
            self.ui.put(("reset",))
            self.running.clear()
            return
        if not self._conv_alive(gen):   # 耳を起こす(最長60秒)・読む間に終わった
            self._boot_cancelled()
            return
        self.stt = stt
        self.stt.langs = tuple(pair_t)  # 言語判定を今日のペアに絞る(短い区間の取り違え対策)
        self.stt.local.langs = tuple(pair_t)  # このMacの耳(SenseVoice)もペア外を雑音扱いにする
        threading.Thread(target=_warm_ear, args=(self.stt,), daemon=True).start()
        # テーマメモの用語を耳の手がかりに(11/06: YOLO→「養老」、人名の漢字の取り違え)
        # 用語辞書(W3)もテーマの後ろに足す(語の切れ目で約200字まで)
        # 「語=訳」(訳の指定、11/08)は耳には元の語だけ渡す
        self.stt.prompt = stt_prompt(topic, terms=source_forms(
            (getattr(self, "_start_extra", None) or {}).get("terms", ())))
        # 用語辞書(W3)は訳にも使う: 「語=訳」は訳の指定、英字の語はそのまま(terms.py、11/08)
        self._user_terms = tuple((getattr(self, "_start_extra", None) or {}).get("terms", ()))
        threading.Thread(target=terms_warm, daemon=True).start()   # 用語集を先に読む(最初の訳を待たせない)
        # 用語の分野(起動画面。自動ならテーマ欄に IT の語があれば IT)。訳の手がかりで IT の語を優先する
        terms_set_domain(getattr(s, "term_domain", ""), topic)
        # このMacの耳(SenseVoice)が落とした略語を裏で whisper に聞き直す(11/08 relisten.py)。
        # リモートサーバーの耳では落とさないので使わない
        self._human_edited = set()     # 人が直した行(聞き直しで上書きしない)
        old = self.__dict__.pop("_relisten", None)
        if old is not None:
            old.close()
        if (s.relisten and stt_url is None and getattr(self.stt.local, "fast", False)
                and "ja" in pair_t):
            self._relisten = Relistener(
                make_ear=lambda: STT(s.whisper_model),
                on_fixed=lambda job, new, g=gen: self._relisten_fixed(job, new, g),
                on_done=lambda job: self.ui.put(("fix", {"id": job["uid"], "done": True})),
                busy=lambda: bool(getattr(self, "_tw_busy", False)) or not self._seg_q.empty(),
                clean=lambda t, lang: clean_heard(t, lang, self._heard_terms))
            self._relisten.warm()
            print("[relisten] このMacの耳の略語の聞き直し: あり", flush=True)
        # 耳の文字の聞き違いを直す語: 用語辞書 + テーマの語(「語=訳」は元の語だけ使われる)
        self._heard_terms = self._user_terms + tuple(t.strip() for t in _TERM_SPLIT.split(topic or "")
                                                     if t.strip())
        # 先送り(11/06, uplink.py): 話している間に声を リモートサーバーの耳へ送っておき、聞くときは範囲だけ
        self.uplink = Uplink(stt_url) if (stt_url and s.stt_stream and server_streams(stt_url)) else None
        self.stt.uplink = self.uplink
        local_fast = getattr(self.stt.local, "fast", False)
        ear = "リモートGPU" if stt_url else ("このMac(SenseVoice)" if local_fast else "このMac")
        print(f"[engine] 翻訳={lab} {self.model_used} / 耳={ear}", flush=True)
        self.ui.put(("engine", {"engine": self.engine_used, "model": self.model_used,
                                "ear": ear}))
        self._boot(2)
        # 会議メモだけ は読み上げない: 声のモデルも読まない(起動が速い)
        tts = self.tts = None if mode == "notes" else _shared_tts(s)   # 先に読んだ声(11/08 speed)
        if tts is not None:   # 選んだスピーカー(11/08 hear。無ければ既定で鳴らし1回知らせる)
            tts.output_device = s.output_device
            tts.on_device = lambda ev, want, using: self._on_audio_dev("output", ev, want, using)
        self.pace = PaceController()      # 話のペースに合わせた読み上げの速さ(11/08、会話ごと)
        self._playing = None              # 鳴っている読み上げの (始めた壁時計, 予想の秒数)
        log = self.log = SessionLog(s.log_dir, pair, topic, mode, extra=self._meta_extra(),
                                    audio=self._rec_wanted(mode))
        self._rec_open()   # 録音(W2): 会議メモを使うときは既定で残す(起動画面で切り替え)
        rec = self.__dict__.get("_rec")
        self._live_start(mode, pair)      # W1: 会話中のライブ要約(会議メモを使う時だけ)
        # 話者の見分け(会議メモを使うとき)。通訳の道を待たせないよう別の1本のスレッドで。
        self.spk = _speaker_tracker(s) if mode in ("notes", "both") else None
        self._spk_pool = ThreadPoolExecutor(max_workers=1) if self.spk is not None else None
        if mode in ("notes", "both"):
            print(f"[spk] 話者の見分け: {'あり' if self.spk is not None else 'なし(モデル無し)'}",
                  flush=True)

        seg_q: "queue.Queue" = queue.Queue()
        text_q: "queue.Queue" = queue.Queue()
        self.tts_q = queue.Queue()
        self._seg_q, self._text_q = seg_q, text_q   # 「会議を終える」で流し切りを待つ
        # 先読み翻訳(11/06): 文末の続き待ちの間に「ここで終わる」と予想して訳し始め、
        # 予想訳を先に出す。当たれば待ち終わりと同時に確定訳が出る(speculate.py)。
        self.spec = Speculator(
            run=lambda msgs, od: translate_stream(base, model, msgs, od),
            make_msgs=lambda text, lang: build_messages(pair_t, topic, list(self.history),
                                                        text, lang, terms=self._user_terms),
            show=lambda text, trans, lang, guess: self.ui.put(
                ("partial", text, trans, lang, guess)),
            # 予想訳ができたら読み上げの音も先に作る(合成0.25〜1.0秒を待ちの間に済ませる)
            ready=lambda text, lang, trans: self._prepare_voice(pair_t, lang, trans),
        ) if s.speculate and mode != "notes" else None
        self._boot(3)
        # リモートサーバーの耳(1回0.1〜0.2秒)なら細かく区切って文末で繋ぎ直す。このMacの耳
        # (1回2秒強)で細かく区切ると認識が追いつかないので従来の区切り。
        # このMacの耳も SenseVoice(1回0.15秒)なら リモートサーバー と同じく細かく区切れる(11/06)
        self.fine_seg = stt_url is not None or local_fast
        cap = self.capture = SpeechCapture(s.silero_vad, seg_q, s.sample_rate,
                                           s.min_speech_s, s.max_speech_s, s.min_silence_s,
                                           vad_threshold=s.vad_threshold, fine=self.fine_seg)
        self.capture.uplink = self.uplink
        self._attach_system_audio(self.capture)   # 11/07 オンライン会議: この Mac の音も聞く
        self.capture.tap = self._rec_tap   # マイクの全チャンクを録音へ(読み上げで止めている間も)
        # マイクを開けない・音が届かなくなった(抜いた)を画面に出す(11/07 監査: 今までは無言)
        self.capture.on_error = lambda title, detail: self.ui.put(("error", title, detail))
        # 選んだマイク(11/08 hear): 名前で開く。無い・外れた → システムの既定で続けて画面にそっと知らせる
        self.capture.device = s.input_device
        self.capture.on_device = lambda ev, want, using: self._on_audio_dev("input", ev, want, using)
        ear = self.stt.local if isinstance(self.stt, RemoteSTT) else self.stt
        if s.interim_s > 0 and self.fine_seg and getattr(ear, "fast", False):
            self.capture.interim_s = s.interim_s   # 話している最中の途中字幕(このMacの速い耳)
        if not self._conv_alive(gen):
            # 声・記録を用意している間に終わった: マイクを開けずに、ここで開けた物を閉じる
            # (「会議を終える」はその時点の self.capture / self.log しか閉じられない)
            for closer in (tts, log, self.uplink, getattr(cap, "mix_in", None)):
                try:
                    if closer is not None:
                        (closer.stop if hasattr(closer, "stop") else closer.close)()
                except Exception:  # noqa: BLE001
                    pass
            if rec is not None and self.__dict__.get("_rec") is rec:
                self._rec_stop()
            for name, mine in (("capture", cap), ("tts", tts), ("log", log)):
                if getattr(self, name, None) is mine:
                    setattr(self, name, None)
            self._boot_cancelled()
            return
        self.capture.start()
        self._boot(4)  # 完了: バーを満たして消す
        self.ui.put(("status", "listen", "聞いています"))

        # 3段パイプライン(11/05): 録音 →seg_q→ STTワーカー →text_q→
        # テキストワーカー(翻訳) →tts_q→ 読み上げワーカー。1本のスレッドで
        # 直列にやっていた頃は、読み上げ(3〜6秒)のあいだに積んだ発話が行列し、
        # 翻訳が何秒も止まって見えた(「翻訳中は録音されていない」の実態)。
        # 切り離せば読み上げ中も裏で翻訳が走り、マイクが閉くのは読み上げの
        # 瞬間だけになる。
        w_stt = threading.Thread(target=self._stt_worker,
                                 args=(seg_q, text_q, pair_t, gen), daemon=True)
        w_text = threading.Thread(target=self._text_worker,
                                  args=(text_q, base, model, pair_t, topic, gen),
                                  daemon=True)
        w_tts = threading.Thread(target=self._tts_worker, args=(gen,), daemon=True)
        w_stt.start()
        w_text.start()
        w_tts.start()
        try:
            while self._conv_alive(gen):
                time.sleep(0.2)
        finally:
            self._back_from_conversation()   # 起動画面に戻っても、操作されるまでは処理場所を起こさない
            # この会話のマイクは必ず閉じる(何度止めてもよい)。「会議を終える」が起動の途中で
            # self.capture=None を読んでいると、後から開いたマイクを誰も止めなかった(11/07 監査)
            try:
                cap.stop()
            except Exception:  # noqa: BLE001
                pass
            for w in (w_stt, w_text, w_tts):
                w.join(timeout=10)
            if self.spec is not None:
                print(f"[spec] 先読み 的中{self.spec.hits} / 外れ{self.spec.misses}", flush=True)
                self.spec.cancel_all()
            if self.uplink is not None:
                print(f"[uplink] 範囲で聞けた {self.stt.span_hits} / 送り直し {self.stt.span_miss}",
                      flush=True)
                self.uplink.close()

    def _boot_cancelled(self) -> None:
        """起動(翻訳モデルの読み込み等)の途中で会話が終わった。画面はもう起動画面か次の会話
        なので、ここからは何も送らない(error/reset を送ると次の会話の画面を壊す)。running も
        触らない(次の会話のもの)。"""
        print("[boot] 起動の途中で会話が終わったので、開けずに止めます", flush=True)

    def _stt_worker(self, seg_q: "queue.Queue", text_q: "queue.Queue",
                    pair_t: tuple[str, str], gen=None) -> None:
        """区間 → whisper → 守卫 → Endpointer(文末で区切りを決める) → text_q。

        Endpointer(11/05夜): 区間ごとに文字にして、文が終わっていれば訳へ、途中なら
        続きを待ち、続きが来たら音声ごと繋いで聞き直す(リモートサーバーの耳のときだけ。
        このMacの耳は1回2秒強なので文字を繋ぐ)。言語が変わったら別の人として切る。
        """
        s = self.s

        def preview(text: str, lang: str) -> None:
            """リモートサーバーの耳が回線で遅いとき、このMacの耳の結果で予想訳を先に出す(11/06)。
            確定(字幕の確定・読み上げ)は リモートサーバーの結果で行うので品質は変わらない。"""
            if lang not in pair_t or len(norm(text)) < 2 or is_echo(self.last_spoken, text):
                return
            text = clean_heard(text, lang, self._heard_terms)   # 字名 → 英字の略語・利用者の語(terms.py)
            self.ui.put(("partial", text, "…", lang))
            spec = getattr(self, "spec", None)
            if spec is not None and self._preview_spec():
                spec.update((text, lang), [], preview=True)   # 読み上げの音は作らない

        def interim() -> None:
            """話し続けている最中: ここまでの音声をこのMacの速い耳で聞き、原文と予想訳を出す
            (11/06 英会話で1区間が10〜12.5秒続き、言い終わるまで字幕が空だった)。
            確定はこれまでどおり区間が閉じてから(リモートサーバーの耳)。"""
            cap = self.capture
            it = cap.take_interim() if cap is not None and hasattr(cap, "take_interim") else None
            if it is None:
                return
            ear = self.stt.local if isinstance(self.stt, RemoteSTT) else self.stt
            t1 = time.time()
            toks: list = []
            stamps: list = []
            try:
                if hasattr(ear, "transcribe_stamped"):
                    sg0 = getattr(cap, "segmenter", None)
                    cont = bool(len(it) > 2 and sg0 is not None and hasattr(sg0, "starts_at_cut")
                                and sg0.starts_at_cut(it[2]))
                    text, lang, _p, toks, stamps = ear.transcribe_stamped(it[0], cont=cont)
                else:
                    text, lang, _p = ear.transcribe(it[0])
            except Exception:  # noqa: BLE001 - 途中字幕は出なくても確定で取り返す
                return
            print(f"[interim {time.time() - t1:4.2f}s] {len(it[0]) / s.sample_rate:4.1f}s音声 "
                  f"{lang or '-'} {norm(text)[:30]}", flush=True)
            self._long_cut(it, toks, stamps, lang, pair_t)
            if lang not in pair_t or len(norm(text)) < 2 or is_echo(self.last_spoken, text):
                return
            pend = ep.pending()
            if pend and pend[1] == lang:    # 続き待ちの断片の続き: 繋げて見せる
                text = (pend[0] + text) if lang in ("ja", "zh") else f"{pend[0]} {text}"
            text = clean_heard(text, lang, self._heard_terms)
            self.ui.put(("partial", text, "…", lang))
            spec = getattr(self, "spec", None)
            if spec is not None and self._preview_spec():
                spec.update((text, lang), [], preview=True)   # 予想訳(読み上げの音は作らない)

        def hear(a, pv=None, span=None, prompt=None, cont=False):
            if cont and not isinstance(self.stt, RemoteSTT) and hasattr(self.stt, "transcribe_stamped"):
                return self.stt.transcribe(a, cont=True)   # 区切った後半: BGM の判定でも話の続き
            if isinstance(self.stt, RemoteSTT):
                if prompt is not None:   # この1回だけ手がかりを上書き(self.stt.prompt は書き換えない)
                    return self.stt.transcribe(a, on_preview=pv, span=span, prompt=prompt)
                return self.stt.transcribe(a, on_preview=pv, span=span)
            return self.stt.transcribe(a)

        ep = Endpointer(
            rejoin=((lambda a, span=None: hear(a, preview, span))
                    if getattr(self, "fine_seg", False) else None),
            hold_open_s=s.hold_open_s if s.hold_open_s > 0 else 0.0,
            # 話者の分からない短い区間は、状態を変えない見当で「別の人」なら繋がない(会議メモ、11/07)
            spk_differ=((lambda a, sid: self.spk.different_speaker(a, sid))
                        if getattr(self, "spk", None) is not None
                        and hasattr(self.spk, "different_speaker") else None))
        t_first: "float | None" = None   # 待っている断片の、最初の区間の処理開始時刻
        net_slow = False                 # 画面に「回線が混んでいます」を出しているか
        spk_segs: list = []              # (lo, hi, 話者の見分けの future) 直近の区間

        last_pend: list = [None]        # 先読みに最後に伝えた続き待ちの断片

        def send(units) -> None:
            nonlocal t_first
            for u in units:
                if u.parts > 1:
                    print(f"[seg] {u.parts}区間を1文に: {u.text[:40]}", flush=True)
                # 応答速度の起点 = 話し終わり(マイクの時計を壁時計に直す。区間の終わりには
                # 語尾の余白0.08秒が付いているので引く)
                cap = self.capture
                end = cap.wall_of(u.hi) if cap is not None and hasattr(cap, "wall_of") else None
                tm = {"end": (end - 0.08) if end else None, "ready": time.time(),
                      "span": (u.lo, u.hi),   # 録音のどこか(W2: 記録の行から再生)
                      "spk": [(min(hi2, u.hi) - max(lo2, u.lo), f) for lo2, hi2, f in spk_segs
                              if hi2 > u.lo and lo2 < u.hi],
                      # 話者の分析(W4): 話し始めの壁時計と声の長さを記録に残す
                      "t0": cap.wall_of(u.lo) if end else None, "dur": max(0.0, u.hi - u.lo),
                      "pcm": u.audio}   # この単位の声(このMacの耳の聞き直し relisten.py)
                # 字名の並び(エーアイ)は英字の略語に戻してから字幕・訳へ(11/08 terms.py)。
                # 先読みの鍵(spec)も同じ形にそろえる
                text_q.put((clean_heard(u.text, u.lang, self._heard_terms), u.lang,
                            t_first if t_first is not None else time.time(), tm))
                t_first = None
            if units and ep.pending_text():
                t_first = time.time()
            spec = getattr(self, "spec", None)
            pend = _nl_key(ep.pending(), self._heard_terms)
            # 待ちに入った断片は先に訳し始め、外れた予想は止める。何も変わっていない待ちの間
            # (0.1秒ごとの tick)は呼ばない: 呼ぶと「続き待ちが無い」= 途中字幕・耳の先出しの
            # 予想訳を初字の前に全部止めていた(11/06 AMI 実時間評価: 出た原文の7割に訳が出ない)
            if spec is not None and (units or pend != last_pend[0]):
                spec.update(pend, [_nl_key((u.text, u.lang), self._heard_terms) for u in units])
            last_pend[0] = pend

        while self._conv_alive(gen):
            try:
                item = seg_q.get(timeout=0.1)
            except queue.Empty:
                cap = self.capture
                since = cap.segmenter.speaking_since() if cap is not None else None
                send(ep.tick(time.time(), speaking_since=since))
                interim()
                continue
            seg = item
            pending = [seg]
            while not seg_q.empty():  # 今たまっている分もまとめて数える
                try:
                    pending.append(seg_q.get_nowait())
                except queue.Empty:
                    break
            if len(pending) >= 3 and all(isinstance(x, tuple) for x in pending):
                # 詰まり: 捨てずにまとめて1回で聞く(11/06 英会話 AMI: 回線が混むと3区間たまり、
                # 最新1個だけ残して捨てていた。3分で13〜17区間の声を失った)
                print(f"[backlog] {len(pending)}区間をまとめて聞く", flush=True)
                pending = [merge_segments(pending, s.sample_rate)]
            backlog = pick_segments(pending)  # 詰まっていたら最新1個だけ訳す
            if len(backlog) != len(pending):  # でないと本物の発話が行列する(11/05)
                ep.flush()                    # 詰まっている: 待っていた断片も捨てる
                t_first = None
            for item in backlog:
                if not self._conv_alive(gen):
                    return
                span = None
                if isinstance(item, tuple):
                    seg, lo, hi, pre = item[:4]
                    # マイクの通算サンプルのどこか(先送りがあれば範囲だけ送る)
                    span = (item[4] if len(item) > 4 else
                            [(int(round(lo * s.sample_rate)), int(round(hi * s.sample_rate)))])
                else:  # 区間情報なし(古い呼び出し): 単独の発話として扱う
                    seg, pre = item, 0.0
                    hi = time.time()
                    lo = hi - len(seg) / s.sample_rate
                self.interrupting.clear()  # 新しい発話: 前の中止フラグは捨てる
                t0 = time.time()
                # 部屋の雑音しか乗っていない区間は whisper に回さない(幻覚の短文が出て
                # 読み上げゲートを潰す)。しきいは固定値 s.noise_rms_floor と「今の部屋の
                # 雑音×2」の小さい方: 卓上マイクが遠い/入力が小さいと声の RMS が0.005前後
                # しかなく(AMI 会議)、固定値0.006だけでは本物の発話ごと捨てていた。
                cap = self.capture
                gate = s.noise_rms_floor
                if cap is not None and hasattr(cap, "segmenter"):
                    gate = min(gate, max(2.0 * cap.segmenter.noise, 0.0015))
                if rms(seg) < gate:
                    self.ui.put(("status", "listen", "聞いています"))
                    continue
                self.ui.put(("status", "work", "翻訳中"))
                try:
                    # 続き待ちの断片が無いときだけ予想訳を出す(あれば繋ぎ直しの方で出す)
                    sg = getattr(cap, "segmenter", None)
                    ct = sg.cut_text(hi) if (sg is not None and hasattr(sg, "cut_text")
                                             and not isinstance(self.stt, RemoteSTT)) else None
                    if ct:
                        # 区切った前半(11/08): 途中字幕の字は後ろの文脈まで聞いて出したもの。前半だけ聞き直すと
                        # BGM の上では字の誤りが増えた(eval_long)。このMacの耳のときだけ(リモートサーバーの耳は聞き直す)
                        text, lang, prob = ct[0], ct[1], 1.0
                    else:
                        cont = bool(sg is not None and hasattr(sg, "starts_at_cut") and sg.starts_at_cut(lo))
                        text, lang, prob = hear(seg, preview if ep.pending() is None else None, span,
                                                cont=cont)
                except Exception as e:  # noqa: BLE001
                    self.ui.put(("error", "音声認識に失敗しました", str(e)))
                    self.ui.put(("status", "listen", "聞いています"))
                    continue
                # 手がかり(用語辞書)の語だけが返ったら、中身が消えている(11/07 用語辞書の A/B ja_dev4:
                # 3.7秒が「渡辺さん。」だけ)。長い区間だけ、手がかりなしで1回だけ聞き直して長い方を採る。
                # 手がかりが空なら terms_only は常に False(まったく走らない)。
                if (len(seg) / s.sample_rate >= 2.5 and isinstance(self.stt, RemoteSTT)
                        and terms_only(text, self.stt.prompt)):
                    t_r = time.time()
                    try:
                        # 手がかりは引数で外す(self.stt.prompt を書き換えると、hedge の2本目のように
                        # 別のスレッドで問い合わせを組む呼び出しが "" を読んでしまう)
                        t2, l2, p2 = hear(seg, None, span, prompt="")
                    except Exception:  # noqa: BLE001
                        t2, l2, p2 = "", "", 0.0
                    tr = self._terms_retry
                    tr[0] += 1
                    tr[1] += time.time() - t_r
                    print(f"[terms] 手がかりの語だけ「{text[:16]}」→ 手がかりなしで聞き直し "
                          f"「{t2[:24]}」 {time.time() - t_r:.2f}s", flush=True)
                    if t2 and l2 and len(norm(t2)) > len(norm(text)):
                        text, lang, prob = t2, l2, p2
                cg = bool(getattr(self.stt, "congested", False))
                if cg != net_slow:   # 回線の混み具合が変わったときだけ画面に知らせる
                    net_slow = cg
                    self.ui.put(("net", {"slow": cg, "net_s": round(getattr(self.stt, "net_s", 0.0), 2)}))
                # 段ごとの所要時間を残す(11/05「遅い」の切り分け用。合計だけでは
                # whisper/待ち行列/翻訳のどこが遅いのか分からなかった)。
                print(f"[stt {time.time() - t0:4.2f}s] {len(seg) / s.sample_rate:4.1f}s音声 "
                      f"{lang or '-'} p={prob:.2f}", flush=True)
                if not text or not norm(text) or not lang:
                    # 聞こえなかったらはっきり出す。無音だと壊れたと思う(11/05)。
                    self.empty_streak += 1
                    hint = ("聞き取れませんでした — もう一度" if self.empty_streak == 1
                            else "聞いています")
                    self.ui.put(("status", "listen", hint))
                    continue
                self.empty_streak = 0
                if is_echo(self.last_spoken, text):
                    self.ui.put(("status", "listen", "聞いています"))
                    continue  # 自分の声の返り聞き捨て
                if echoes_prompt(text, self.stt.prompt):
                    print(f"[halluc] 用語の手がかりの復唱を破棄: {text[:24]}", flush=True)
                    self.ui.put(("status", "listen", "聞いています"))
                    continue
                if stock_phrase(text):
                    # 定型句(「ご視聴ありがとうございました」)は whisper の幻覚の定番だが、本物の発言を
                    # そう聞き違えることもある(11/06 擬似会議 ja_dev4「必要な台数を見積もっておきます」
                    # p=0.67)。このMacの速い耳がペア内の言語で聞けていればそちらを採る
                    loc = self.stt.local if isinstance(self.stt, RemoteSTT) else None
                    if loc is not None and getattr(loc, "fast", False):
                        try:
                            t2, l2, p2 = loc.transcribe(seg)
                        except Exception:  # noqa: BLE001
                            t2, l2, p2 = "", "", 0.0
                        # 短すぎる文字は救わない(11/07 擬似会議 v2 ja_booth: 1.3秒の区間の「一たや。」は
                        # 次の文の頭の聞き違いで、記録に雑な行が入った)
                        if l2 in pair_t and len(norm(t2)) >= 4 and not stock_phrase(t2):
                            print(f"[halluc] 定型句({text[:16]})→ このMacの耳の文字: {t2[:24]}", flush=True)
                            text, lang, prob = t2, l2, p2
                if hallucinated_out_of_pair(pair_t, lang, text, prob=prob):
                    # 確率の低い発話 = 雑音が生んだ whisper の幻覚。ペア外は
                    # prob<0.80、ペア内でも prob<0.50 なら捨てる(11/05: 長さだけ
                    # だと長い韓国語幻覚「이게 10분이죠 이게 동거예요」が抜けた)。
                    print(f"[halluc] {lang} prob={prob:.2f} を破棄: {text[:24]}",
                          flush=True)
                    self.ui.put(("status", "listen", "聞いています"))
                    continue
                if t_first is None:
                    t_first = t0
                pool = getattr(self, "_spk_pool", None)
                spk_now = None
                if pool is not None:   # 誰の声か(裏で。訳は待たない)
                    fut = pool.submit(self.spk.assign_indexed, seg)
                    spk_segs.append((lo, hi, fut))
                    del spk_segs[:-12]
                    # 続き待ちの断片と繋ぐか決めるときだけ見る(別の人なら繋がない)。1区間の見分けは
                    # 0.01〜0.03秒なので最大0.1秒だけ待ち、まだなら分からないとして今まで通り
                    spk_now = lambda f=fut: _spk_id(f.result(timeout=0.1))
                sg = getattr(cap, "segmenter", None)
                cut = bool(sg is not None and hasattr(sg, "was_cut") and sg.was_cut(hi))
                send(ep.push(seg, lo, hi, text, lang, now=time.time(), pre=pre, prob=prob, span=span,
                             spk=spk_now, cut=cut))
                if ep.pending_text():  # 文の途中: 続きを聞いている
                    # 聞けた所までの原文を先に出す(11/06 実会話: 2区間の長い文で
                    # 言い終わるまで約7秒、字幕が空のままだった)。訳は文が閉じてから。
                    # 予想訳がもうあれば(このMacの耳の先出しと同じ文字だった等)それを出す
                    spec = getattr(self, "spec", None)
                    pk = _nl_key(ep.pending(), self._heard_terms)
                    have = spec.peek(*pk) if spec is not None and pk else ""
                    shown = clean_heard(ep.pending_text(), lang, self._heard_terms)
                    if have:
                        self.ui.put(("partial", shown, have, lang, True))
                    else:
                        self.ui.put(("partial", shown, "…", lang))
                    self.ui.put(("status", "listen", "聞いています"))

    def _long_cut(self, it, toks: list, stamps: list, lang: str, pair_t) -> "float | None":
        """長い発話を区切る(11/08): 話し続けている区間が long_split_s 秒を超えたら、途中字幕の字の時刻から
        文末(長ければ読点・息つぎ)の所を選び、その前半を確定させる(SpeechCapture の区間を切る)。
        前半はすぐ聞き直されて訳・読み上げへ流れ、後半はそのまま聞き続ける。
        見つけた問題: 切れ目のない独白が 9〜12.5 秒の1単位になり、訳 101 字・読み上げ +3.7 秒、
        次の文は 14.9 秒遅れで読み上げを飛ばした。"""
        s = self.s
        cap = self.capture
        sg = getattr(cap, "segmenter", None)
        if (s.long_split_s <= 0 or sg is None or not hasattr(sg, "request_cut") or not toks
                or lang not in pair_t or it is None or len(it) < 3):
            return None
        dur = len(it[0]) / s.sample_rate
        r = pick_cut_at(toks, stamps, dur, split_s=s.long_split_s, comma_s=s.long_comma_s,
                        breath_s=s.long_breath_s)
        if r is None:
            return None
        t, i = r
        at = it[2] + t
        if sg.request_cut(at, text=head_text(toks, i), lang=lang):
            print(f"[cut] {dur:4.1f}秒話し続けている → {t:4.1f}秒の所で前半を確定", flush=True)
            return at
        return None

    def _text_worker(self, text_q: "queue.Queue", base: str, model: str,
                     pair_t: tuple[str, str], topic: str, gen=None) -> None:
        """1件ずつ翻訳する。読み上げは _tts_worker に渡して自分はすぐ次へ走る
        (読み上げ待ちで翻訳が詰まるのを根治)。文の区切り(続き待ち)は
        _stt_worker の Endpointer が音声の段階で済ませている(11/05夜。以前は
        ここで文字を継ぎ足していたが、0.5秒の無音で2人分が1区間になる問題は
        文字の段階では直せなかった)。"""
        while self._conv_alive(gen):
            try:
                item = text_q.get(timeout=0.2)
            except queue.Empty:
                continue
            pending = [item]
            while not text_q.empty():  # 翻訳を待っている間に積んだ分も数える
                try:
                    pending.append(text_q.get_nowait())
                except queue.Empty:
                    break
            for text, lang, t0, *tm in pick_segments(pending):
                if not self._conv_alive(gen):
                    return
                self._tw_busy = True     # 「会議を終える」は書き終えるまで待つ
                try:
                    self._utterance(base, model, pair_t, topic, text, lang, t0,
                                    tm[0] if tm else None)
                except Exception as e:  # noqa: BLE001
                    # 1件の失敗(記録の書き込み失敗=ディスクがいっぱい 等)でこの担当が無言で死ぬと、
                    # 以後の発言が訳されず「翻訳中」のまま・_tw_busy が True のまま残った(11/07 監査)
                    print(f"[uttr-err] {type(e).__name__}: {e}", flush=True)
                    self.ui.put(("error", "この発言を処理できませんでした", f"{type(e).__name__}: {e}"))
                    self.ui.put(("status", "listen", "聞いています"))
                finally:
                    self._tw_busy = False

    # ---------- 話のペースに合わせた読み上げ(11/08) ----------

    def _auto_speed(self, lang: str) -> bool:
        return (getattr(self, "pace", None) is not None
                and (self.s.tts_speed_mode or {}).get(lang, "fixed") == "auto")

    def _backlog_s(self) -> float:
        """たまっている読み上げの秒数(行列の訳の予想 + 鳴っている読み上げの残り)。"""
        pc = getattr(self, "pace", None)
        if pc is None:
            return 0.0
        tot = 0.0
        for it in list(getattr(self.tts_q, "queue", [])):
            try:
                trans, spoken = it[0], it[1]
                sp = it[4] if len(it) > 4 and it[4] else self.s.tts_speed.get(spoken, 1.0)
                tot += pc.predict(trans, engine_of(self.s.tts_voice.get(spoken, "")), spoken, sp)
            except Exception:  # noqa: BLE001
                continue
        pl = getattr(self, "_playing", None)
        if pl:
            tot += max(0.0, pl[0] + pl[1] - time.time())
        return tot

    def _voice_speed(self, trans: str, spoken: str, src_dur: float,
                     extra_backlog: float = 0.0) -> "float | None":
        """自動ならこの訳を読む速さ。決まった速さのままなら None(TTS のその言語の速さ)。"""
        if not self._auto_speed(spoken):
            return None
        base = float(self.s.tts_speed.get(spoken, 1.0))
        self.pace.priority = getattr(self.s, "tts_speed_priority", "clear")
        return self.pace.choose(trans, engine_of(self.s.tts_voice.get(spoken, "")), spoken,
                                src_dur, self._backlog_s() + extra_backlog, base=base)

    def _listen_how(self) -> str:
        """読み上げ中のマイクの扱い: "keep"(聞き続ける)/ "system"(マイクだけ止め、この Mac の音は聞く)/
        "pause"(止める。今まで)。設定 listen_while_speaking: auto / always / off。"""
        mode = getattr(self.s, "listen_while_speaking", "auto")
        if mode == "off":
            return "pause"
        if mode == "always":
            return "keep"
        name = getattr(self.tts, "output_device", "") or ""      # "" = システムの既定
        private = devices.output_is_private(name)
        if private is None and not name:
            name = devices.default_name(devices.OUTPUT)
        if ears_only(private, name):
            return "keep"            # ヘッドホン・Bluetooth・名前がヘッドセット: 声はマイクに入らない
        cap = self.capture
        if cap is not None and getattr(cap, "mix_in", None) is not None:
            return "system"          # この Mac の音の取り込みは Sokutsu 自身の声を外している
        return "pause"

    def _preview_spec(self) -> bool:
        """途中字幕・耳の先出しからも予想訳を作るか(設定 speculate_preview)。"""
        return bool(getattr(self.s, "speculate_preview", True))

    def _early_voice(self, partial: str, spoken: str, done: set) -> None:
        """訳を流し込んでいる最中に、もう変わらない読点までの声を作り始める(11/08 speed。Kokoro の声だけ。
        読点ごとに作るのは全部をそろえてから作るのと同じ音)。"""
        tts = self.tts
        if tts is None or not self.s.voice_on or not hasattr(tts, "stable_chunks"):
            return
        sp = None
        pc = getattr(self, "pace", None)
        if pc is not None and self._auto_speed(spoken):
            sp = pc.current(engine_of(self.s.tts_voice.get(spoken, "")), spoken,
                            float(self.s.tts_speed.get(spoken, 1.0)))
        try:
            for c in tts.stable_chunks(partial, spoken):
                if c not in done:
                    done.add(c)
                    tts.prepare(c, spoken, sp, early=True)
        except Exception:  # noqa: BLE001 - 先に作れなくても確定で作る
            pass

    def _prepare_voice(self, pair_t: tuple[str, str], lang: str, trans: str) -> None:
        if not self.s.voice_on:
            return   # 字幕だけ: 音を作らない
        spoken = target_lang(pair_t, lang)
        tts = self.tts
        if spoken and tts is not None and not wrong_language("", trans, out_lang(pair_t, lang)):
            # 予想訳の音(guess): 確定訳の音の後に作り、新しい予想が来たら古い予想は作らない(11/08 speed)。
            # 速さが自動なら今の速さで作る(読む時の速さと近ければ作り置きを使う: tts.SPEED_TOL)
            sp = None
            pc = getattr(self, "pace", None)
            if pc is not None and self._auto_speed(spoken):
                sp = pc.current(engine_of(self.s.tts_voice.get(spoken, "")), spoken,
                                float(self.s.tts_speed.get(spoken, 1.0)))
            tts.prepare(trans, spoken, sp, guess=True)

    def _meta_extra(self) -> dict:
        """記録の meta に足す項目(担当ごとに1行ずつ。docs/superpowers/specs/2026-11-07-…)。"""
        extra: dict = dict(getattr(self, "_start_extra", None) or {})   # 議題・テンプレート・用語(W3/W1)
        extra["source"] = self.s.audio_source   # 聞いた音 "mic" | "mic+system"(オンライン会議)
        if self.get_template() != "free":   # 会議の種類(W1。画面から来なければ前回の選択。自由は書かない)
            extra.setdefault("template", self.get_template())
        return extra

    def _attach_system_audio(self, capture) -> None:
        """「マイク+この Mac の音」なら、この Mac で鳴っている音(Zoom 等の相手の声)を
        マイクに足す(sysaudio.py)。聞けなければマイクだけで続け、画面に理由を出す。"""
        if self.s.audio_source != "mic+system":
            return
        from .sysaudio import MicPlusSystem

        def on_status(kind: str, msg: str) -> None:
            print(f"[sysaudio] {kind}: {msg}", flush=True)
            log = self.log   # 状態の行(type=sysaudio。録音ファイルの "audio" と混ぜない)
            if log is not None:
                try:
                    log.add_row({"type": "sysaudio", "state": kind, "detail": msg,
                                 "ts": round(time.time(), 2)})
                except Exception:  # noqa: BLE001
                    pass
            if kind == "running":
                self.ui.put(("status", "listen", "聞いています — マイク+この Mac の音"))
            elif kind == "failed":
                self.ui.put(("error", "この Mac の音を聞けません", msg + " — マイクだけで続けます"))

        mx = MicPlusSystem(self.s.sample_rate, on_status=on_status,
                           gain=float(self.s.system_gain or 1.0))
        capture.mix_in = mx
        mx.start()

    # ---------- このMacの耳の聞き直し(relisten.py、11/08) ----------

    def _relisten_wanted(self, text: str, lang: str, tm) -> bool:
        rl = self.__dict__.get("_relisten")
        if not (rl is not None and tm and tm.get("pcm") is not None):
            return False
        # IT の話の中(分野で IT を選んだ / 今か直前の発話に IT の語)なら、知らない短いカタカナ
        # (「ブキー」= VPC)も落とした跡とみる(11/08 IT 用語)
        from .terms import current_domain, is_it_context
        it = current_domain() == "it" or is_it_context(
            text, lang, context=" ".join(h.get("orig", "") for h in list(self.__dict__.get("history") or [])[-2:]))
        return dropped_letters(text, lang, it=it)

    def _relisten_submit(self, uid: int, text: str, lang: str, tm, extra) -> None:
        log = self.log
        job = {"uid": uid, "orig": text, "lang": lang, "pcm": tm["pcm"], "log": log,
               "i": (getattr(log, "n_rows", 0) - 1) if log is not None else -1}
        job.update(extra or {})
        self._relisten.submit(job)

    def _relisten_human_edited(self, job: dict) -> bool:
        """その行を人が直したか(line_edit で覚えた行、または記録に by の無い edit がある)。"""
        log, i = job.get("log"), job.get("i", -1)
        if i is None or i < 0:
            return False
        if i in self.__dict__.get("_human_edited", ()):
            return True
        if log is None:
            return False
        try:
            log.flush()
            for raw in Path(log.path_jsonl).read_text(encoding="utf-8").splitlines():
                if '"edit"' not in raw:
                    continue
                row = json.loads(raw)
                if row.get("type") == "edit" and row.get("i") == i and not row.get("by"):
                    return True
        except Exception:  # noqa: BLE001
            return False
        return False

    def _relisten_fixed(self, job: dict, new: str, gen=None) -> None:
        """聞き直しで略語が戻った: 訳し直し、画面・記録・(まだ読んでいなければ)読み上げを差し替える。"""
        alive = gen is None or self._is_conv(gen)
        if self._relisten_human_edited(job):
            # 人が直した行(line_edit)は聞き直しで戻さない(aud-ui 11/08)
            if alive:
                self.ui.put(("fix", {"id": job["uid"], "done": True}))
            return
        trans = ""
        if job.get("base") and alive:
            msgs = build_messages(job["pair"], job["topic"],
                                  [h for h in self.history if h is not job["hist"]], new, job["lang"],
                                  terms=getattr(self, "_user_terms", ()))
            trans = translate_with_retry(job["base"], job["model"], msgs)
            if not trans or looks_truncated(new, trans):
                trans = job.get("trans", "")
            job["hist"]["orig"], job["hist"]["trans"] = new, trans
            if job.get("spoken") and job["uid"] not in self.__dict__.setdefault("_voice_started", set()):
                self.__dict__.setdefault("_voice_fix", {})[job["uid"]] = trans   # まだ読んでいない: 直した訳を読む
                if self.tts is not None:
                    self.tts.prepare(trans, job["spoken"])
        row = {"type": "edit", "i": job["i"], "orig": new, "by": "relisten", "ts": round(time.time(), 2)}
        if trans:
            row["trans"] = trans
        if job.get("log") is not None and job["i"] >= 0:
            job["log"].add_row(row)      # 会議を終えた後でも記録は直す(閉じた記録にはファイルへ足す)
        if alive:
            self.ui.put(("fix", {"id": job["uid"], "orig": new, "trans": trans, "done": True}))

    def _tts_worker(self, gen=None) -> None:
        """読み上げ専用。マイクが閉くのはここだけ(翻訳/録音は裏で走り続ける)。

        話のペースに合わせる(11/08): 遅れていれば行列の短い訳をまとめて1回で読み、速さを決め直す。
        飛ばすのは最後の手段(話し終わりから voice_max_lag_s 秒を超えたとき)で、画面に印を出す。
        読み上げ中のマイク: _listen_how()(耳元の機器なら聞き続ける・この Mac の音は聞く・止める)。"""
        s = self.s
        while self._is_conv(gen) and (self.running.is_set() or not self.tts_q.empty()):
            try:
                trans, spoken, *meta = self.tts_q.get(timeout=0.2)
            except queue.Empty:
                continue
            if self.interrupting.is_set():
                continue  # 行列している間にユーザーが中止した: 読まない
            if not s.voice_on:
                continue  # 字幕だけ: 声を出さず、マイクも止めない
            t_deq = time.time()
            uid, end, sp, src_dur, marks = (meta + [None] * 5)[:5]
            trans = self.__dict__.get("_voice_fix", {}).pop(uid, trans)   # 聞き直しで直した訳(まだ読んでいない)
            self.__dict__.setdefault("_voice_started", set()).add(uid)
            lag = (time.time() - end) if end else 0.0
            auto = self._auto_speed(spoken)
            if auto and lag > s.voice_join_after_s:
                # 遅れている: 同じ言語の訳が行列に続いていれば、まとめて1回で読む(鳴り始めの待ちを減らす)
                n = 0
                while n < 2:
                    try:
                        nxt = self.tts_q.queue[0]
                    except IndexError:
                        break
                    if nxt[1] != spoken:
                        break
                    nxt = self.tts_q.get_nowait()
                    nt = self.__dict__.get("_voice_fix", {}).pop(nxt[2], nxt[0]) if len(nxt) > 2 else nxt[0]
                    if len(nxt) > 2:
                        self.__dict__.setdefault("_voice_started", set()).add(nxt[2])
                    trans += ("" if spoken in ("ja", "zh") else " ") + nt
                    src_dur = (src_dur or 0.0) + (nxt[5] if len(nxt) > 5 and nxt[5] else 0.0)
                    if len(nxt) > 2:
                        uid = nxt[2]
                    n += 1
                if n:
                    print(f"[voice join] {n + 1}件をまとめて読む {lag:.1f}秒遅れ", flush=True)
            if end and lag > s.voice_max_lag_s:
                # 読み上げが詰まって順番が来た時にはもう古い: 字幕は出ているので読まない(画面に印)
                print(f"[voice skip] {lag:.1f}秒遅れ {len(trans)}字 {spoken}", flush=True)
                self.ui.put(("voice", {"id": uid, "skipped": True}))
                continue
            if auto:
                sp = self._voice_speed(trans, spoken, src_dur or 1.0, max(0.0, lag - 1.0))
            how = self._listen_how()
            st = ("status", "speak", f"読み上げ中 ({LANG_NAMES.get(spoken, spoken)})")
            if how != "pause":   # 下の帯: 「マイク休止中」ではなく「聞いています」(mic)/「この Mac の音は聞いています」
                st += ({"listen": "mic" if how == "keep" else "system"},)
            self.ui.put(st)
            ta = getattr(self.capture, "tts_active", None)
            if how == "pause":
                self.capture.paused.set()
            else:
                if how == "system" and hasattr(self.capture, "mic_muted"):
                    self.capture.mic_muted.set()
                if ta is not None:
                    ta.set()     # メーター: 「読み上げ中 — 聞いています」(hear_state の speak_listen)
            if end:   # 声が鳴り始めた瞬間を画面の応答速度に足す(ログにも残す: 実会話の切り分け用)
                def _on_start(_u=uid, _e=end, _n=len(trans), _sp=sp, _auto=auto, _m=marks, _dq=t_deq):
                    lat = round(time.time() - _e, 2)
                    print(f"[voice +{lat:.2f}s] {_n}字 {spoken}"
                          + (f" ×{_sp:.2f}" if _sp else "")
                          + " " + voice_breakdown(_e, _m, _dq, time.time()), flush=True)
                    self.ui.put(("voice", {"id": _u, "lat": lat, "speed": _sp, "auto": _auto}))
                self.tts.on_start = _on_start
            eng = engine_of(s.tts_voice.get(spoken, ""))
            pc = getattr(self, "pace", None)
            pred = pc.predict(trans, eng, spoken, sp or s.tts_speed.get(spoken, 1.0)) if pc else 0.0
            self._playing = (time.time(), pred)
            try:
                if sp:
                    self.tts.say(trans, spoken, sp)
                else:
                    self.tts.say(trans, spoken)
            except Exception:
                pass
            self._playing = None
            if pc is not None:   # 実際に鳴った長さで1秒あたりの字数を合わせる
                # 鳴らした速さ: 作り置き(速さが少し違う)を鳴らしたときはその速さ(tts.last_speed)
                played = getattr(self.tts, "last_speed", None)
                pc.observe(trans, eng, spoken,
                           float(played) if isinstance(played, (int, float)) and played > 0
                           else (sp or s.tts_speed.get(spoken, 1.0)),
                           float(getattr(self.tts, "last_play_s", 0.0) or 0.0))
            if how == "pause":
                self.capture.resume_after_tts(s.gate_tail_s)
            else:
                if how == "system" and hasattr(self.capture, "mic_muted"):
                    time.sleep(s.gate_tail_s)
                    self.capture.mic_muted.clear()
                if ta is not None:
                    ta.clear()
            if self.tts_q.empty():   # 読み上げ終わり: 柱と「読み上げ中」を戻す(次が無ければ)
                self.ui.put(("status", "listen", "聞いています"))

    def _utterance(self, base: str, model: str, pair_t: tuple[str, str],
                   topic: str, text: str, lang: str, t0: float,
                   tm: "dict | None" = None) -> None:
        spec = getattr(self, "spec", None)
        if spec is not None:
            spec.mute(True)    # 確定訳を流す間は、次の断片の予想訳で大字幕を上書きしない
        try:
            self._utterance_body(base, model, pair_t, topic, text, lang, t0, tm)
        finally:
            if spec is not None:
                spec.mute(False)

    def _utterance_body(self, base: str, model: str, pair_t: tuple[str, str],
                        topic: str, text: str, lang: str, t0: float,
                        tm: "dict | None" = None) -> None:
        """ガードを通った1件の原文を 翻訳→字幕→読み上げ行列→履歴/ログ。

        テキストワーカーから2つの経路で呼ばれる: ①発話が途中で終わっていた/
        連結完了後 ②待ち時間切れの flush_held。読み上げは tts_q に投げるだけで
        この関数はすぐ次の翻訳に戻れる(11/05まで直列で読み上げのあいだ翻訳が
        止まっていた)。
        """
        s = self.s
        self._rt_ctx = (base, model, pair_t, topic)   # 「訳し直す」用(11/10)
        spk, segs = _resolve_spk_idx((tm or {}).get("spk"))
        tr = getattr(self, "spk", None)
        if tr is not None and spk:
            spk = tr.canonical(spk)          # 途中で同じ人と分かって合わせた話者
            al = tr.aliases
            if al != getattr(self, "_spk_aliases", {}) and self.log:
                self._spk_aliases = al
                self.log.add_alias(al)       # 過去の行は記録を読む側で直す
        if getattr(self, "mode", "interpret") == "notes":
            # 会議メモだけ: 訳さず読まず、聞き取った文字と話者だけを出して残す
            self._utt_seq = getattr(self, "_utt_seq", 0) + 1
            checking = self._relisten_wanted(text, lang, tm)
            self.ui.put(("utterance", {
                "lang": lang, "langName": LANG_NAMES.get(lang, lang), "orig": text,
                "trans": "", "spoken": "", "id": self._utt_seq, "spk": spk or "", "checking": checking}))
            if self.log:
                self.log.add(lang, text, "", None, spk, segs, audio=self._rec_span(tm), timing=tm)
            if checking:
                self._relisten_submit(self._utt_seq, text, lang, tm, None)
            self._spk_track(self._utt_seq, segs)
            print(f"[note] {spk or '-'} {lang} {text[:60]}", flush=True)
            self.ui.put(("status", "listen", "聞いています"))
            return
        msgs = build_messages(pair_t, topic, self.history, text, lang,
                              terms=getattr(self, "_user_terms", ()))
        acc: list[str] = []
        last_flush = [0.0]
        t_tr = time.time()   # 翻訳を投げた時刻(t0からここまで=認識+待ち行列+続き待ち)
        first = [0.0]        # 最初の訳文字が届いた時刻(TTFT)

        early_spoken = target_lang(pair_t, lang)
        early_done: set = set()

        def on_delta(d: str, _acc=acc, _lf=last_flush):
            _acc.append(d)
            now = time.time()
            if not first[0]:
                first[0] = now
            if now - _lf[0] > 0.12:  # 途中字幕の描画間隔(短いほどリアルタイム感)
                _lf[0] = now
                self.ui.put(("partial", text, "".join(_acc), lang))
            if early_spoken:   # 読点までそろった所の声を先に作る(Kokoro の声だけ。字の数十回なので軽い)
                self._early_voice("".join(_acc), early_spoken, early_done)

        # 空訳/一時的失敗は 0.4 秒待って1回だけやり直す(11/05 15:42 の
        # 「こんにちは」→「（翻訳できませんでした）」連発の根治。失敗した6文が
        # 全部あとで正しく訳せた = 混線は待てば救える)。
        trans = None
        spec = getattr(self, "spec", None)
        if spec is not None:
            # 待ちの間に先読みした訳があれば受け取る(同じ文字・同じ言語のときだけ)
            trans = self.spec.take(text, lang, on_delta)
            if trans is None:
                acc.clear()       # 外れ/失敗: 途中まで渡った文字は捨てて普通に訳す
        hit = bool(trans)
        guess_shown = spec.last_first_show if (spec is not None and hit) else 0.0
        if not trans:
            trans = translate_with_retry(base, model, msgs, on_delta)
        if trans:   # 「原文 => 訳」の形で返った(履歴の並べ方を真似た): 訳の部分だけ(11/09)
            trans = strip_history_echo(text, trans)
        tgt = out_lang(pair_t, lang)
        if trans and wrong_language(text, trans, tgt):
            # 原文の写し/別の言語で返った(11/06 FLEURS: zh→ja の写し、en→zh の日本語)。
            # 履歴の写しが次の写しを呼ぶので、空履歴で1回だけやり直す。
            print(f"[lang] 訳の言語が違う(空履歴で再試行): {text[:24]} ⇝ {trans[:24]}",
                  flush=True)
            acc.clear()
            try:
                retry = translate_stream(
                    base, model, build_messages(pair_t, topic, [], text, lang,
                                   terms=getattr(self, "_user_terms", ())), on_delta)
            except Exception:  # noqa: BLE001
                retry = ""
            if retry and not wrong_language(text, retry, tgt):
                trans = retry
        if trans and looks_truncated(text, trans):
            # 履歴を真似て1語だけ返ることがある(11/05実測)。空履歴で
            # 1回だけやり直す。ここで戻さないと次の発話にも汚染が回る。
            print(f"[trunc] 訳が破綻(空履歴で再試行): {text[:24]} ⇝ {trans[:24]}",
                  flush=True)
            acc.clear()  # 途中字幕を破綻訳で汚さない
            try:
                retry = translate_stream(
                    base, model, build_messages(pair_t, topic, [], text, lang,
                                   terms=getattr(self, "_user_terms", ())),
                    on_delta)
            except Exception:  # noqa: BLE001
                retry = ""
            if retry and not looks_truncated(text, retry):
                trans = retry
        if trans and LENGTH_RETRY and length_off(text, trans, lang, tgt):
            # 訳の長さが原文から大きく外れた(断片に続きを創作した・長い文の大半を落とした。11/08 このMac
            # gemma-3-4b)。空の履歴で1回だけ訳し直し、外れていなければそちらを採る
            print(f"[len] 訳の長さが外れた(空履歴で再試行): {text[:24]} ⇝ {trans[:24]}", flush=True)
            acc.clear()
            try:
                retry = translate_stream(
                    base, model, build_messages(pair_t, topic, [], text, lang,
                                                terms=getattr(self, "_user_terms", ())), on_delta)
            except Exception:  # noqa: BLE001
                retry = ""
            if retry and not length_off(text, retry, lang, tgt) and not looks_truncated(text, retry):
                trans = retry
        if not trans or looks_truncated(text, trans):
            # 空/破綻訳は画面で成功のふりをしない。履歴にも積まない —
            # 原文をフォールバックで積むとオウム返しの汚染になる(11/05)。
            self.ui.put(("error", "翻訳が途中で切れました",
                         "もう一度話しかけてください。何度も出る場合は翻訳サーバー"
                         "(リモートサーバーゲート/LM Studio)のモデル設定を確認してください。"))
            self.ui.put(("utterance", {
                "lang": lang, "langName": LANG_NAMES.get(lang, lang),
                "orig": text, "trans": "（翻訳できませんでした）",
                "spoken": ""}))
            if self.log:
                self.log.add(lang, text, "（翻訳できませんでした）", None, spk, segs,
                             audio=self._rec_span(tm), timing=tm)
            self.ui.put(("status", "listen", "聞いています"))
            return

        spoken = target_lang(pair_t, lang)
        self._utt_seq = getattr(self, "_utt_seq", 0) + 1
        uid = self._utt_seq
        end = (tm or {}).get("end")
        lat = latency_marks(end, (tm or {}).get("ready"), guess_shown, first[0],
                            time.time(), hit)
        # 原文エコーの判定材料は訳そのもの。読み上げ完了を待たずに覚えておく
        # (別スレッド読み上げでは完了がいつか分からない)。
        self.last_spoken = trans
        if spoken and not self.interrupting.is_set():
            # 話のペースに合わせる(11/08): 原文を話した秒数と、たまっている読み上げから速さを決める
            src_dur = max(0.5, float((tm or {}).get("dur") or 0.0) - 0.3)
            sp = self._voice_speed(trans, spoken, src_dur)
            if self.tts is not None:
                self.tts.prepare(trans, spoken, sp)  # 前の読み上げ中でも音を作り始める
            marks = {"ready": (tm or {}).get("ready"), "enq": time.time()}   # 鳴り始めまでの内訳(ログ)
            self.tts_q.put((trans, spoken, uid, end, sp, src_dur, marks))  # 読み上げは別スレッド。翻訳は止まらない
        checking = self._relisten_wanted(text, lang, tm)
        self.ui.put(("utterance", {
            "lang": lang, "langName": LANG_NAMES.get(lang, lang),
            "orig": text, "trans": trans, "spoken": spoken or "",
            "id": uid, "lat": lat, "spk": spk or "", "checking": checking}))
        hist_row = {"lang": lang, "orig": text, "trans": trans}
        self.history.append(hist_row)
        if self.log:
            self.log.add(lang, text, trans, spoken, spk, segs, audio=self._rec_span(tm), timing=tm)
        if checking:
            self._relisten_submit(uid, text, lang, tm, {"hist": hist_row, "spoken": spoken,
                                                        "base": base, "model": model, "pair": pair_t,
                                                        "topic": topic, "trans": trans})
        self._spk_track(uid, segs)
        now = time.time()
        dt = now - t0
        ttft = (first[0] - t_tr) if first[0] else -1.0
        print(f"[uttr {dt:5.2f}s] 前段{t_tr - t0:4.2f} 初字{ttft:4.2f} 訳{now - t_tr:4.2f} "
              f"{lang}{'→' + spoken if spoken else '(字幕のみ)'}{' 先読み' if hit else ''} "
              f"{text[:40]} ⇝ {trans[:40]}", flush=True)
        self.ui.put(("status", "listen", "聞いています"))

    def retranslate(self, orig, lang):
        """右クリック/R「訳し直す」: その原文を今の会話の文脈でもう一度訳して返す(失敗は "")。画面の差し替えは JS。"""
        ctx = self.__dict__.get("_rt_ctx")
        orig = str(orig or "").strip()[:2000]
        if not ctx or not orig:
            return ""
        base, model, pair_t, topic = ctx
        try:
            hist = [h for h in list(self.__dict__.get("history") or []) if h.get("orig") != orig]
            msgs = build_messages(pair_t, topic, hist[-6:], orig, str(lang or ""),
                                  terms=getattr(self, "_user_terms", ()))
            trans = translate_with_retry(base, model, msgs)
        except Exception:  # noqa: BLE001
            return ""
        return trans if trans and not looks_truncated(orig, trans) else ""


ICON = Path(__file__).resolve().parent / "assets" / "icon_1024.png"


def _set_dock_icon(path: Path = ICON) -> bool:
    """Dock のアイコンを Sokutsu のロゴにする(既定は Python のロケット)。
    AppKit が無い・画像が読めない等で失敗しても起動は止めない。"""
    try:
        from AppKit import NSApplication, NSImage  # pyobjc(pywebview の cocoa 依存で入る)
        img = NSImage.alloc().initWithContentsOfFile_(str(path))
        if img is None:
            return False
        NSApplication.sharedApplication().setApplicationIconImage_(img)
        return True
    except Exception:  # noqa: BLE001
        return False


def main():
    from . import menubar
    menubar.prepare_app()   # メニューバーの太字の名前を「Sokutsu」に(NSApp が起動を終える前に)
    bridge = Bridge()
    _set_dock_icon()
    window = webview.create_window(
        "Sokutsu", str(WEBUI), js_api=bridge,
        width=1280, height=800, min_size=(900, 600),
        background_color="#0D1216",
    )
    bridge.window = window
    menubar.attach(bridge, window)   # ⌘Q・閉じるで _shutdown を通す。窓が出たらメニューを差し替える
    webview.start(bridge._started, window)
    # 窓が閉じたら確実に終わる。音声入力・whisper・読み上げのスレッドが C 側で
    # 残ると、窓は消えたのにプロセス(とマイク)が居座る。ログは stop() で書き出し済み。
    try:
        bridge._shutdown()
        sys.stdout.flush()
    except Exception:
        pass
    os._exit(0)


if __name__ == "__main__":
    main()
