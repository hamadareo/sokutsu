"""会議の録音(W2, 11/07)。記録の行を押すとその声を聞き直せるように、マイクの音をこの Mac に残す。

形式の選び方(1時間で小さく・落ちても残る):
  録音中   CAF(16kHz・16bit・モノラル)。data の長さを -1(=ファイルの終わりまで)にしておくと、
           アプリが途中で落ちても「書いた所まで」がそのまま正しい音声として読める(ヘッダを直す必要がない)。
           毎秒 flush するので失うのは最後の1秒以内。1時間で 115MB。
  終わったら AAC(m4a, 32kbps)に変換して CAF を消す。1時間で約14.7MB、変換は約2秒(afconvert, 11/07実測)。
           変換に失敗したら CAF を残す(声は消さない)。次の起動で recover() が変換し直す。

時刻の対応(行を押した所から鳴らすため):
  録音の秒 = 壁時計 - ts0。記録を開いた時刻を ts0 とし、最初のチャンクまでの間は無音で埋める。
  入力が1秒以上途切れたら(スリープ・装置の切替)そのぶん無音を足し、壁時計とずれないようにする。
  記録の発言の区間はマイクの通算秒(読み上げ中にマイクを止めている間は進まない)で来るので、
  止めていた区間を覚えておき audio_offset() で録音の秒に直す(読み上げの間も録音は続いている)。
"""
from __future__ import annotations

import bisect
import queue
import re
import struct
import subprocess
import threading
import time
from pathlib import Path

import numpy as np

SR = 16000
BITRATE = 32000


def caf_header(sr: int = SR) -> bytes:
    """LPCM 16bit リトルエンディアン・モノラルの CAF ヘッダ。data の長さは -1(終わりまで)。"""
    desc = struct.pack(">d", float(sr)) + b"lpcm" + struct.pack(">IIIII", 2, 2, 1, 1, 16)
    return (b"caff" + struct.pack(">HH", 1, 0)
            + b"desc" + struct.pack(">q", len(desc)) + desc
            + b"data" + struct.pack(">q", -1) + struct.pack(">I", 0))


def caf_path(jsonl: "str | Path") -> Path:
    p = Path(jsonl)
    return p.with_name(f"{p.stem}.audio.caf")


def m4a_path(jsonl: "str | Path") -> Path:
    p = Path(jsonl)
    return p.with_name(f"{p.stem}.audio.m4a")


def audio_file(jsonl: "str | Path") -> "Path | None":
    """この会議の聞ける録音(変換済みの m4a、まだなら録音中/変換前の CAF)。無ければ None。"""
    for p in (m4a_path(jsonl), caf_path(jsonl)):
        if p.exists() and p.stat().st_size > 100:
            return p
    return None


def should_record(setting, mode: str) -> bool:
    """録音するか。setting: "on" / "off" / "auto"(既定)。
    auto = 会議メモを使うとき(notes / both)だけ残す。通訳だけ は残さない: その場で通じることが目的で、
    相手は通訳に同意していても録音には同意していないことがある(残すときは起動画面で「残す」を選ぶ)。"""
    if setting in ("on", True):
        return True
    if setting in ("off", False):
        return False
    return mode in ("notes", "both")


class Recorder:
    """SpeechCapture.on_chunk から tap(mono, paused) で全チャンクを受け取って書く。

    tap は音声入力のスレッドで呼ばれるので、数えて行列に積むだけ(書くのは別スレッド)。
    書けなくなっても例外は出さない(failed=True。会話は止めない)。"""

    def __init__(self, path: "str | Path", ts0: float, sr: int = SR, clock=time.time,
                 flush_s: float = 1.0, gap_s: float = 1.0):
        self.path = Path(path)
        self.ts0 = float(ts0)
        self.sr = sr
        self._clock = clock
        self.flush_s = flush_s
        self.gap = int(gap_s * sr)
        self.failed = False
        self.closed = False
        self._audio_n = 0        # 録音に入れたサンプル数(無音の埋めも含む)
        self._mic_n = 0          # マイクの通算(読み上げで止めている間は進まない = StreamSegmenter._fed)
        self._bp_mic: list = []  # 対応の折れ目: マイク通算 → 録音の位置
        self._bp_audio: list = []
        self._dirty = True       # 次の「止めていない」チャンクで折れ目を足す
        self._q: "queue.Queue" = queue.Queue()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._f = self.path.open("wb")
        self._f.write(caf_header(sr))
        self._f.flush()
        self._th = threading.Thread(target=self._writer, daemon=True)
        self._th.start()

    # ---------- 音声入力のスレッド ----------

    def tap(self, mono, paused: bool = False) -> None:
        if self.closed:
            return
        if self.failed or not self._th.is_alive():
            # 書き込み係が死んだ/書けなくなった: 積んでも誰も読まず、行列が上限なく溜まっていた
            # (11/07 監査: 1時間で約115MB)。以後は捨てる。止まったことは1回だけ知らせる
            if not getattr(self, "_drop_noted", False):
                self._drop_noted = True
                print(f"[rec] 録音が止まりました(書き込みに失敗)。会話は続けます: {self.path.name}",
                      flush=True)
            return
        try:
            c = np.asarray(mono, dtype=np.float32).reshape(-1)
            n = len(c)
            end = int(round((self._clock() - self.ts0) * self.sr))
            pad = end - (self._audio_n + n)
            # 最初のチャンク(記録を開いてからマイクが動くまで)と、1秒を超える途切れは無音で埋める
            if pad > 0 and (self._audio_n == 0 or pad > self.gap):
                self._q.put(bytes(2 * pad))
                self._audio_n += pad
                self._dirty = True
            if paused:
                self._dirty = True
            else:
                if self._dirty:
                    self._bp_mic.append(self._mic_n)
                    self._bp_audio.append(self._audio_n)
                    self._dirty = False
                self._mic_n += n
            self._q.put((np.clip(c, -1.0, 1.0) * 32767.0).astype("<i2").tobytes())
            self._audio_n += n
        except Exception:  # noqa: BLE001 - 録音のために会話を止めない
            self.failed = True

    # ---------- 書く側 ----------

    def _writer(self) -> None:
        last = time.time()
        while True:
            b = self._q.get()
            if b is None:
                break
            if self.failed:
                continue
            try:
                self._f.write(b)
                if time.time() - last >= self.flush_s:
                    self._f.flush()
                    last = time.time()
            except Exception:  # noqa: BLE001
                self.failed = True
        try:
            if not self._f.closed:
                self._f.flush()
        except Exception:  # noqa: BLE001
            self.failed = True

    def close(self) -> Path:
        if not self.closed:
            self.closed = True
            self._q.put(None)
            self._th.join(timeout=10)
            try:
                self._f.close()
            except Exception:  # noqa: BLE001
                pass
        return self.path

    # ---------- 時刻 ----------

    @property
    def seconds(self) -> float:
        return self._audio_n / self.sr

    def audio_offset(self, mic_s: float) -> "float | None":
        """マイクの通算秒 → 録音の先頭からの秒。まだ何も録っていなければ None。"""
        if not self._bp_mic:
            return None
        m = mic_s * self.sr
        i = max(0, bisect.bisect_right(self._bp_mic, m) - 1)
        return (self._bp_audio[i] + (m - self._bp_mic[i])) / self.sr


# ---------- 変換・後始末 ----------

def duration(path: "str | Path") -> "float | None":
    try:
        out = subprocess.run(["afinfo", str(path)], capture_output=True, text=True,
                             timeout=20).stdout
    except Exception:  # noqa: BLE001
        return None
    m = re.search(r"estimated duration:\s*([0-9.]+)", out)
    return float(m.group(1)) if m else None


def _caf_seconds(p: Path, sr: int = SR) -> float:
    size = p.stat().st_size
    return max(0, size - len(caf_header(sr))) / 2 / sr


def finalize(caf: "str | Path", bitrate: int = BITRATE) -> "Path | None":
    """CAF → AAC(m4a)。長さが合っていれば CAF を消す。失敗したら None(CAF はそのまま)。"""
    caf = Path(caf)
    if not caf.exists():
        return None
    out = caf.with_name(caf.name.removesuffix(".caf") + ".m4a")
    tmp = caf.with_name(caf.name.removesuffix(".caf") + ".tmp.m4a")
    try:
        r = subprocess.run(["afconvert", "-f", "m4af", "-d", "aac", "-b", str(bitrate), "-c", "1",
                            str(caf), str(tmp)], capture_output=True, timeout=600)
        want = _caf_seconds(caf)
        got = duration(tmp) if r.returncode == 0 and tmp.exists() else None
        if got is None or abs(got - want) > max(0.5, want * 0.01):
            tmp.unlink(missing_ok=True)
            return None
        tmp.replace(out)
        caf.unlink(missing_ok=True)
        return out
    except Exception:  # noqa: BLE001
        tmp.unlink(missing_ok=True)
        return None


def recover(log_dir: "str | Path", min_age_s: float = 60.0) -> list:
    """前回落ちて残った録音(CAF)を m4a にする。書き込み中のもの(min_age_s 以内に更新)には触らない。"""
    done = []
    now = time.time()
    for caf in sorted(Path(log_dir).glob("session-*.audio.caf")):
        try:
            if now - caf.stat().st_mtime < min_age_s:
                continue
        except OSError:
            continue
        Path(str(caf).removesuffix(".caf") + ".tmp.m4a").unlink(missing_ok=True)
        out = finalize(caf)
        if out is not None:
            done.append(out)
    return done
