"""音声・動画ファイルの取り込み(W2, 11/07「他の会議メモアプリにある機能は全て」: Otter/Notta/Rimo/PLAUD の
「ファイルをアップロードして書き起こし」)。

見つけた問題: 即通はマイクの会話しか記録にできず、録ってあった会議・講演の音声や動画からメモを作れなかった。

方式: ライブと同じ道を、音声の時計で実時間より速く回す。
  読み込み   afconvert で 16kHz モノラルへ(m4a/mp3/wav/aac/caf/aiff/flac と、音声が読める mp4/mov)。
             afconvert が読めない動画は avconvert(macOS 標準)で音声だけ取り出してから。ffmpeg は使わない
             (このMacに無く、入れなくても足りる)。
  区切り     ライブと同じ SpeechCapture.on_chunk(自動音量 → silero VAD → StreamSegmenter)。
  聞き取り   区間ができた順に耳へ並べて投げる(リモートサーバーの耳なら3本並列、このMacは1本)。
  文の区切り ライブと同じ Endpointer。待ち時間の判定は「音声の時計」で行い、耳は一瞬で返った
             ものとして扱う(ライブの理想の動きと同じ。繋いだら音声ごと聞き直す)。
  話者       ライブと同じ SpeakerTracker を区間の順に。最後に付け直し(refine)を記録へ。
  記録       新しい会議(mode=notes, source=file)。各行に a0/a1(録音の中の秒)、ts = ts0 + 話し終わり。
  録音       16kHz の音声を AAC(m4a, 32kbps)にして会議の録音にする(画面で行から再生できる)。
"""
from __future__ import annotations

import queue
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from .audio import rms
from .guards import echoes_prompt, hallucinated_out_of_pair
from .langdetect import norm
from .log import SessionLog
from .recorder import BITRATE, m4a_path
from .segmenter import Endpointer

SR = 16000
CHUNK = 512
EXTS = (".m4a", ".mp3", ".wav", ".wave", ".aac", ".caf", ".aif", ".aiff", ".aifc", ".flac",
        ".mp4", ".m4v", ".mov", ".3gp", ".3g2", ".m4b", ".amr")
VIDEO = (".mp4", ".m4v", ".mov", ".3gp", ".3g2")


WORK_PREFIX = ".import-"
STALE_S = 6 * 3600


def cleanup_stale(log_dir: "str | Path", min_age_s: float = STALE_S) -> list:
    """落ちた取り込み(アプリの強制終了・停電)が残した作業ファイルを消す。起動時に呼ぶ。
    作業の WAV は1時間で約115MB。書いている最中のもの(min_age_s 以内に更新)には触らない。"""
    out = []
    now = time.time()
    try:
        cand = list(Path(log_dir).glob(WORK_PREFIX + "*.work.wav*"))
    except OSError:
        return out
    for q in cand:
        try:
            if now - q.stat().st_mtime < min_age_s:
                continue
            q.unlink()
            out.append(q)
        except OSError:
            pass
    return out


class ImportFailed(Exception):
    """読めないファイルなど。メッセージは画面にそのまま出せる日本語。"""


def supported(path: "str | Path") -> bool:
    return Path(str(path)).suffix.lower() in EXTS


def _afconvert(src: Path, dst: Path, timeout: float) -> bool:
    r = subprocess.run(["afconvert", "-f", "WAVE", "-d", "LEI16@16000", "-c", "1", str(src), str(dst)],
                       capture_output=True, timeout=timeout)
    return r.returncode == 0 and dst.exists() and dst.stat().st_size > 44


def _read_wav16(p: Path) -> np.ndarray:
    """afconvert の WAVE(16bit, 16kHz, モノラル)を読む。チャンクの並びは決め打ちしない。"""
    b = p.read_bytes()
    i = 12
    while i + 8 <= len(b):
        cid, n = b[i:i + 4], int.from_bytes(b[i + 4:i + 8], "little")
        if cid == b"data":
            data = b[i + 8:i + 8 + n]
            data = data[:len(data) // 2 * 2]
            return np.frombuffer(data, "<i2").astype(np.float32) / 32768.0
        i += 8 + n + (n & 1)
    raise ImportFailed("音声を読み出せませんでした")


def decode(src: "str | Path", work_wav: "str | Path", timeout: float = 1800) -> np.ndarray:
    """ファイル → 16kHz モノラル float32。work_wav に作業用の WAV を置く(呼び出し側が消す)。"""
    src, work = Path(src), Path(work_wav)
    if not src.is_file():
        raise ImportFailed("ファイルが見つかりません")
    ok = False
    try:
        ok = _afconvert(src, work, timeout)
    except Exception:  # noqa: BLE001
        ok = False
    if not ok:
        # 動画など afconvert が直接読めない入れ物: avconvert で音声だけ m4a に取り出す
        tmp = work.with_name(work.name + ".m4a")
        try:
            r = subprocess.run(["avconvert", "-s", str(src), "-p", "PresetAppleM4A", "-o", str(tmp),
                                "--replace"], capture_output=True, timeout=timeout)
            ok = r.returncode == 0 and tmp.exists() and _afconvert(tmp, work, timeout)
        except Exception:  # noqa: BLE001
            ok = False
        finally:
            tmp.unlink(missing_ok=True)
    if not ok:
        raise ImportFailed("このファイルから音声を読み出せませんでした(音声の無い動画か、対応していない形式です)")
    x = _read_wav16(work)
    if len(x) < SR // 2:
        raise ImportFailed("音声が短すぎます(0.5秒未満)")
    return x


def _vote(segs: list, lo: float, hi: float):
    """1単位の話者 = 重なった区間の見分けの、重なりの長さでの多数決(ライブの _resolve_spk_idx と同じ)。"""
    votes: dict = {}
    idxs: list = []
    for lo2, hi2, fut in segs:
        if hi2 <= lo or lo2 >= hi or fut is None:
            continue
        try:
            r = fut.result(timeout=60)
        except Exception:  # noqa: BLE001
            continue
        if isinstance(r, tuple):
            idxs.append(r[0])
            r = r[1]
        if r:
            votes[r] = votes.get(r, 0.0) + max(min(hi2, hi) - max(lo2, lo), 0.01)
    return (max(votes.items(), key=lambda kv: kv[1])[0] if votes else None), idxs


def _hms(t: float) -> str:
    return time.strftime("%H:%M:%S", time.localtime(t))


def run_import(src: "str | Path", log_dir: "str | Path", pair: str, topic: str = "", *,
               stt, make_capture, spk=None, prog=None, cancel: "threading.Event | None" = None,
               fine: bool = True, workers: int = 1, hold_open_s: float = 1.2,
               noise_floor: float = 0.006, prompt: str = "") -> "dict | None":
    """取り込んで新しい会議にする。戻り値 {id, lines, dur, speakers}。中止したら None(作りかけは消す)。

    stt: transcribe(音声) -> (文, 言語, 確率)。make_capture(行列, 音声) -> on_chunk を持つ区切り役。
    prog(段階, 0..1, 補足): read → listen → speakers → done。"""
    src = Path(src)
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    pair_t = tuple(pair.split("-"))
    say = prog or (lambda *a, **k: None)
    stop = cancel or threading.Event()
    last_pct = [0.0]

    def p(stage, x, info=None):
        x = max(last_pct[0], min(1.0, x))
        last_pct[0] = x
        say(stage, round(x, 3), info)

    p("read", 0.0)
    work = log_dir / f"{WORK_PREFIX}{int(time.time() * 1000)}.work.wav"
    log = None
    pool = spool = None
    finished = False
    try:
        x = decode(src, work)
        dur = len(x) / SR
        p("read", 0.04, {"dur": round(dur, 1)})
        if stop.is_set():
            return None
        log = SessionLog(str(log_dir), pair, topic, "notes", audio=True,
                         extra={"source": "file", "file": src.name, "dur": round(dur, 1)})
        ts0 = log.ts0
        # 録音(画面で行から再生する): 16kHz の作業 WAV を AAC に
        m4a = m4a_path(log.path_jsonl)
        subprocess.run(["afconvert", "-f", "m4af", "-d", "aac", "-b", str(BITRATE), "-c", "1",
                        str(work), str(m4a)], capture_output=True, timeout=1800)
        p("read", 0.06)

        # ---- 区切り(音声の時計で全部流す)。区間ができたら耳と話者の見分けに並べる ----
        seg_q: "queue.Queue" = queue.Queue()
        cap = make_capture(seg_q, x)
        pool = ThreadPoolExecutor(max_workers=max(1, workers))
        spool = ThreadPoolExecutor(max_workers=1) if spk is not None else None
        segs: list = []        # (音声, lo, hi, pre, 耳の future, 話者の future, 雑音の目安)

        def take():
            while True:
                try:
                    it = seg_q.get_nowait()
                except queue.Empty:
                    return
                a, lo, hi, pre = it[:4]
                noise = getattr(getattr(cap, "segmenter", None), "noise", 0.0)
                f = pool.submit(stt.transcribe, a)
                sf = spool.submit(spk.assign_indexed, a) if spool is not None else None
                segs.append((a, lo, hi, pre, f, sf, noise))

        n = len(x)
        tail = np.zeros(int(0.8 * SR), np.float32)      # 最後の区間を閉じる無音
        for i in range(0, n + len(tail), CHUNK):
            if stop.is_set():
                return None
            ch = x[i:i + CHUNK] if i < n else tail[i - n:i - n + CHUNK]
            if len(ch) < CHUNK:
                ch = np.concatenate([ch, np.zeros(CHUNK - len(ch), np.float32)])
            cap.on_chunk(ch)
            take()
            if (i // CHUNK) % 1500 == 0:   # 約48秒ごと
                p("listen", 0.06 + 0.2 * min(1.0, i / max(1, n)))
        sg = getattr(cap, "segmenter", None)
        if sg is not None and hasattr(sg, "flush_current"):
            for it in sg.flush_current():
                seg_q.put(it)
        take()

        # ---- 順に: 守卫 → Endpointer(音声の時計) → 記録 ----
        def hear(a):
            return stt.transcribe(a)

        ep = Endpointer(rejoin=(lambda a: hear(a)) if fine else None,
                        hold_open_s=hold_open_s if hold_open_s > 0 else 0.0)
        spk_segs: list = []
        aliases: dict = {}
        n_lines = 0

        def write(units):
            nonlocal n_lines, aliases
            for u in units:
                who, idxs = _vote(spk_segs, u.lo, u.hi)
                if spk is not None and who:
                    who = spk.canonical(who)
                    al = dict(getattr(spk, "aliases", {}) or {})
                    if al != aliases:
                        aliases = al
                        log.add_alias(al)
                log.add(u.lang, u.text, "", None, who, idxs, audio=(u.lo, u.hi),
                        ts=ts0 + u.hi, t=_hms(ts0 + u.lo))
                n_lines += 1

        def advance(until: float, next_speech: "float | None"):
            """待っている断片の期限が until より前に来たら、ライブと同じく時計を進めて確かめる。"""
            for _ in range(3):
                pd = ep._p
                if pd is None:
                    return
                for t in (pd.deadline, pd.deadline + ep.speak_cap_s):
                    if t >= until:
                        return
                    since = next_speech if (next_speech is not None and next_speech <= t) else None
                    out = ep.tick(t, speaking_since=since)
                    if out:
                        write(out)
                        break
                else:
                    return

        total = max(1, len(segs))
        for k, (a, lo, hi, pre, f, sf, noise) in enumerate(segs):
            if stop.is_set():
                return None
            advance(hi, lo + pre)
            try:
                text, lang, prob = f.result()
            except Exception:  # noqa: BLE001 - 1区間聞けなくても全体は続ける
                text, lang, prob = "", "", 0.0
            p("listen", 0.26 + 0.66 * (k + 1) / total)
            gate = min(noise_floor, max(2.0 * noise, 0.0015))
            if rms(a) < gate or not text or not norm(text) or not lang:
                continue
            if echoes_prompt(text, prompt) or hallucinated_out_of_pair(pair_t, lang, text, prob=prob):
                continue
            spk_segs.append((lo, hi, sf))
            write(ep.push(a, lo, hi, text, lang, now=hi, pre=pre, prob=prob))
        write(ep.flush())

        p("speakers", 0.95)
        if spk is not None:
            try:
                ids = spk.refine()
                if ids:
                    log.add_refine(ids)
            except Exception:  # noqa: BLE001
                pass
        log.close()
        mid = Path(log.path_jsonl).stem
        finished = True     # ここから後に中止が押されても、返す会議は消さない
        p("done", 1.0, {"id": mid, "lines": n_lines})
        return {"id": mid, "lines": n_lines, "dur": round(dur, 1)}
    except BaseException:
        _remove(log)
        raise
    finally:
        # 中止・失敗でも、並べたまま待っている区間の聞き取り・話者の見分けは取り消す
        # (以前は中止の後も全区間を耳に投げ続け、次の取り込みや会話と CPU・リモートサーバーを取り合った)
        for ex in (pool, spool):
            if ex is not None:
                ex.shutdown(wait=False, cancel_futures=True)
        work.unlink(missing_ok=True)
        if stop.is_set() and not finished:
            _remove(log)


def _remove(log) -> None:
    """作りかけの会議を消す(中止・失敗)。"""
    if log is None:
        return
    try:
        log.close()
    except Exception:  # noqa: BLE001
        pass
    stem = Path(log.path_jsonl).with_suffix("")
    for q in stem.parent.glob(stem.name + ".*"):
        try:
            q.unlink()
        except OSError:
            pass
