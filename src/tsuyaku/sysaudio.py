"""この Mac で鳴っている音(オンライン会議の相手の声)も聞く(11/07)。

見つけた問題: 録音はこの Mac のマイクだけ。Zoom・Teams・Meet でヘッドホンを使うと、
相手の声はマイクに入らず記録にも字幕にも出ない。スピーカーだと部屋の反響ごと小さく入る。

やり方: 小さな部品 bin/sokutsu-sysaudio(native/sysaudio/sysaudio.c, Core Audio の
プロセスタップ)が、この Mac で鳴っている音をモノラル float32 で標準出力へ流す。
Sokutsu 自身の音(Python とその子の afplay / say = 読み上げ)は部品の側で外している。
ここではそれを 16 kHz にして、マイクの1チャンクごとに足し合わせる(Mixer)。
足した後はマイクだけのときと同じ道(VAD → 区切り → 耳 → 話者の見分け → メモ)を通る。

部品が無い・許可が無い・何度も落ちる → マイクだけで続け、on_status で画面へ知らせる。
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
import threading
import time
from collections import deque
from pathlib import Path
from typing import Callable

import numpy as np

from .distenv import DIST

ROOT = Path(__file__).resolve().parents[2]
HELPER = ROOT / "bin" / "sokutsu-sysaudio"
if DIST:   # 配る版: Sokutsu.app/Contents/MacOS/ の起動役の隣(sys.executable = 起動役)
    HELPER = Path(sys.executable).resolve().parent / "sokutsu-sysaudio"

SOURCES = ("mic", "mic+system")

# 部品の終了コード(sysaudio.c と同じ)
RC_PERMISSION = 3
RC_UNSUPPORTED = 4
RC_OUTPUT_CHANGED = 6

PERMISSION_HINT = ("システム設定 > プライバシーとセキュリティ > 画面とシステムオーディオ録音 の"
                   "「システムオーディオ録音のみ」で Sokutsu を許可してください")


def soft_clip(x: np.ndarray, knee: float = 0.8) -> np.ndarray:
    """|x| <= knee はそのまま、超えた分だけ tanh でなだらかに ±1 の手前へ寄せる。"""
    a = np.abs(x)
    over = a > knee
    if not over.any():
        return x
    y = x.copy()
    room = 1.0 - knee
    y[over] = np.sign(x[over]) * (knee + room * np.tanh((a[over] - knee) / room))
    return y


class Resampler:
    """流れてくる音の周波数を変える(窓付き sinc。scipy なしの numpy だけ)。

    48 kHz → 16 kHz なら 16 kHz の手前で切る(折り返し雑音を入れない)。
    呼ぶたびに続きとして扱う(前の呼び出しの終わりの音を内部に持つ)。
    """

    def __init__(self, sr_in: float, sr_out: float = 16000, half_taps: int = 12):
        self.sr_in = float(sr_in)
        self.sr_out = float(sr_out)
        self.same = abs(self.sr_in - self.sr_out) < 1e-6
        self.step = self.sr_in / self.sr_out
        self.cut = min(1.0, self.sr_out / self.sr_in) * 0.9   # 入力のナイキストに対する比
        self.radius = half_taps / self.cut                     # 入力のサンプル数で
        self._r = int(np.ceil(self.radius))
        self._offs = np.arange(-self._r + 1, self._r + 1)
        self._buf = np.zeros(self._r, dtype=np.float64)        # 頭は無音で埋めておく
        self._t = float(self._r)                               # 次に出す点(_buf の中の位置)

    def __call__(self, x) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        if self.same:
            return x
        buf = np.concatenate([self._buf, x.astype(np.float64)])
        last = len(buf) - self._r - 1        # この位置までなら右側の音がそろっている
        if self._t > last:
            self._buf = buf
            return np.zeros(0, dtype=np.float32)
        n = int(np.floor((last - self._t) / self.step)) + 1
        ts = self._t + self.step * np.arange(n)
        base = np.floor(ts).astype(np.int64)
        d = (ts - base)[:, None] - self._offs[None, :]           # 出す点と元の点の距離
        u = d / self.radius
        h = self.cut * np.sinc(self.cut * d) * np.where(np.abs(u) < 1, 0.5 * (1 + np.cos(np.pi * u)), 0)
        y = np.sum(buf[base[:, None] + self._offs[None, :]] * h, axis=1)
        nxt = ts[-1] + self.step
        drop = max(0, int(np.floor(nxt)) - self._r)
        self._buf = buf[drop:]
        self._t = nxt - drop
        return y.astype(np.float32)


class Mixer:
    """マイクの1チャンクに、この Mac の音を同じ長さだけ足す。

    時計はマイクが親: マイクのチャンクが来たときに、溜まっているこの Mac の音から
    同じ長さを取って足す。溜まりすぎ(= 遅れ)が max_lag_s を超えたら古い分を捨てて
    lag_s まで詰める(二つの時計のずれで遅れが伸び続けない)。足りなければ
    ある分だけ使い、前を無音で埋める。まだ1度も届いていなければマイクのまま。
    """

    def __init__(self, sr: int = 16000, lag_s: float = 0.06, max_lag_s: float = 0.25,
                 gain: float = 1.0):
        self.sr = sr
        self.lag = int(sr * lag_s)
        self.max_lag = int(sr * max_lag_s)
        self.gain = float(gain)
        self._q: deque = deque()
        self._n = 0
        self._lock = threading.Lock()
        self.primed = False
        self.dropped = 0       # 遅れで捨てたサンプル数
        self.underruns = 0     # 足りなかった回数
        self.level = 0.0       # 直近のこの Mac の音の RMS(確認用)
        self.peak = 0.0        # これまでの最大の振れ幅(許可が無いと 0 のままの無音が届く)

    def push(self, x) -> None:
        x = np.asarray(x, dtype=np.float32)
        if not len(x):
            return
        with self._lock:
            self._q.append(x)
            self._n += len(x)
            # マイクのチャンクが来ない間(開く前・抜けた・止まった)も溜まり続けていた(11/07 監査:
            # 1時間で約230MB)。mix() は max_lag(0.25秒)を超えた分を捨てるので、2秒より古い分は使われない
            cap = 2 * self.sr
            while self._n > cap and len(self._q) > 1:
                self._n -= len(self._q.popleft())

    def clear(self) -> None:
        with self._lock:
            self._q.clear()
            self._n = 0
            self.primed = False

    def _take(self, n: int) -> np.ndarray:
        parts, need = [], n
        while need > 0 and self._q:
            a = self._q[0]
            if len(a) <= need:
                parts.append(self._q.popleft())
                need -= len(a)
            else:
                parts.append(a[:need])
                self._q[0] = a[need:]
                need = 0
        self._n -= n - need
        return np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)

    def mix(self, mic) -> np.ndarray:
        mic = np.asarray(mic, dtype=np.float32)
        n = len(mic)
        with self._lock:
            if not self.primed:
                if self._n < n:
                    return mic
                self.primed = True
            if self._n > n + self.max_lag:          # 遅れすぎ: 古い分を捨てる
                drop = self._n - n - self.lag
                self._take(drop)
                self.dropped += drop
            sysa = self._take(min(n, self._n))
        if len(sysa) < n:
            self.underruns += 1
            sysa = np.concatenate([np.zeros(n - len(sysa), dtype=np.float32), sysa])
        if len(sysa):
            self.level = float(np.sqrt(np.mean(np.square(sysa, dtype=np.float64))))
            self.peak = max(self.peak, float(np.max(np.abs(sysa))))
        return soft_clip(mic + self.gain * sysa).astype(np.float32)


_READY = re.compile(r"\brate=(\d+)")


class SystemAudio:
    """部品(bin/sokutsu-sysaudio)を子プロセスで動かし、16 kHz にして sink へ渡す。

    on_status(kind, message): kind は "running"(流れ始めた)/ "failed"(諦めた・マイク
    だけで続ける)/ "restarting"(落ちたので起こし直す)。
    """

    def __init__(self, sink: Callable[[np.ndarray], None], cmd: "list[str] | None" = None,
                 sr: int = 16000, on_status: "Callable[[str, str], None] | None" = None,
                 max_restarts: int = 4, restart_window_s: float = 60.0,
                 backoff_s: float = 1.0):
        self.sink = sink
        self.cmd = list(cmd) if cmd else [str(HELPER)]
        self.sr = sr
        self.on_status = on_status or (lambda k, m: None)
        self.max_restarts = max_restarts
        self.restart_window_s = restart_window_s
        self.backoff_s = backoff_s
        self.state = "idle"
        self.error = ""
        self.rate: "int | None" = None
        self.samples = 0                       # 16 kHz にして渡した数
        self._proc: "subprocess.Popen | None" = None
        self._stop = threading.Event()
        self._thread: "threading.Thread | None" = None
        self._restarts: list = []

    def start(self) -> None:
        self._thread = threading.Thread(target=self._supervise, daemon=True, name="sysaudio")
        self._thread.start()

    def stop(self, timeout: float = 2.0) -> None:
        self._stop.set()
        p = self._proc
        if p is not None and p.poll() is None:
            try:
                if p.stdin:
                    p.stdin.close()        # 部品は標準入力が閉じたら片付けて終わる
                p.wait(timeout=1.0)
            except Exception:
                pass
            if p.poll() is None:
                try:
                    p.terminate()
                    p.wait(timeout=1.0)
                except Exception:
                    p.kill()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
        if self.state not in ("failed",):
            self.state = "stopped"

    def _status(self, kind: str, msg: str) -> None:
        try:
            self.on_status(kind, msg)
        except Exception:
            pass

    def _fail(self, msg: str) -> None:
        self.state = "failed"
        self.error = msg
        self._status("failed", msg)

    def _supervise(self) -> None:
        while not self._stop.is_set():
            rc, err = self._run_once()
            if self._stop.is_set():
                break
            if rc is None:          # 起動できなかった(部品が無い等)
                self._fail(err)
                return
            if rc == RC_PERMISSION:
                self._fail("この Mac の音を聞く許可がありません。" + PERMISSION_HINT)
                return
            if rc == RC_UNSUPPORTED:
                self._fail("この macOS ではこの Mac の音を聞けません(macOS 14.2 以降が必要)")
                return
            now = time.time()
            self._restarts = [t for t in self._restarts if now - t < self.restart_window_s]
            if len(self._restarts) >= self.max_restarts:
                self._fail(f"この Mac の音を聞く部品が何度も止まりました({err or f'終了コード {rc}'})")
                return
            self._restarts.append(now)
            self.state = "restarting"
            self._status("restarting", err or f"終了コード {rc}")
            wait = 0.0 if rc == RC_OUTPUT_CHANGED else self.backoff_s * len(self._restarts)
            if self._stop.wait(wait):
                break

    def _run_once(self) -> "tuple[int | None, str]":
        """部品を1回動かして、終わるまで読む。(終了コード, 最後の ERROR 行)。"""
        self.state = "starting"
        self.rate = None
        cmd = self.cmd + ["--exclude-tree", str(os.getpid())]
        try:
            p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                 stderr=subprocess.PIPE, bufsize=0)
        except (FileNotFoundError, PermissionError, OSError) as e:
            return None, (f"この Mac の音を聞く部品がありません({Path(self.cmd[-1]).name}: "
                          f"{type(e).__name__})。scripts/build_sysaudio.sh で作れます")
        self._proc = p
        if self._stop.is_set():
            # stop() が起動と行き違った(_proc がまだ無かった): ここで止めないと、部品は標準入力が
            # 閉じるまで鳴っている音を取り続け、このスレッドも読み続けていた
            try:
                p.stdin.close()
                p.wait(timeout=1.0)
            except Exception:  # noqa: BLE001
                p.kill()
                p.wait()
        ready = threading.Event()
        last_err = [""]

        def read_err() -> None:
            for raw in iter(p.stderr.readline, b""):
                line = raw.decode("utf-8", "replace").strip()
                if not line:
                    continue
                print(f"[sysaudio] {line}", flush=True)
                if line.startswith("READY"):
                    m = _READY.search(line)
                    self.rate = int(m.group(1)) if m else 48000
                    ready.set()
                elif line.startswith("ERROR"):
                    last_err[0] = line[5:].strip()

        te = threading.Thread(target=read_err, daemon=True)
        te.start()
        rs: "Resampler | None" = None
        rest = b""
        fd = p.stdout.fileno()
        while True:
            try:
                data = os.read(fd, 16384)
            except OSError:
                break
            if not data:
                break
            if rs is None:
                if not ready.wait(2.0):     # READY より先に音が来た(普通は無い)
                    self.rate = self.rate or 48000
                rs = Resampler(self.rate or 48000, self.sr)
                self.state = "running"
                self._status("running", f"{self.rate} Hz")
            data = rest + data
            k = len(data) - len(data) % 4
            rest = data[k:]
            if k:
                y = rs(np.frombuffer(data[:k], dtype=np.float32))
                if len(y):
                    self.samples += len(y)
                    self.sink(y)
        rc = p.wait()
        te.join(timeout=1.0)
        for f in (p.stdin, p.stdout, p.stderr):
            try:
                if f:
                    f.close()
            except Exception:
                pass
        self._proc = None
        return rc, last_err[0]


class MicPlusSystem:
    """SpeechCapture に付ける「マイク + この Mac の音」。mix() をマイクのチャンクごとに呼ぶ。"""

    def __init__(self, sr: int = 16000, cmd: "list[str] | None" = None,
                 on_status: "Callable[[str, str], None] | None" = None, gain: float = 1.0,
                 **kw):
        self.mixer = Mixer(sr, gain=gain)
        self.sys = SystemAudio(self.mixer.push, cmd=cmd, sr=sr, on_status=self._on_status, **kw)
        self._user_status = on_status or (lambda k, m: None)

    def _on_status(self, kind: str, msg: str) -> None:
        if kind != "running":
            self.mixer.clear()     # 止まっている間の古い音を次に混ぜない
        self._user_status(kind, msg)

    @property
    def state(self) -> str:
        return self.sys.state

    def start(self) -> None:
        self.sys.start()

    def mix(self, mic) -> np.ndarray:
        return self.mixer.mix(mic)

    def stop(self) -> None:
        self.sys.stop()
        m = self.mixer
        # 実地の確かめ用: 何か届いたか(peak 0 = 無音だけ = 許可が実は無い/何も鳴っていない)
        print(f"[sysaudio] 状態 {self.sys.state} / 受けた {self.sys.samples / m.sr:.1f}秒 / "
              f"最大 {m.peak:.3f} / 遅れで捨てた {m.dropped / m.sr:.2f}秒 / 不足 {m.underruns}回",
              flush=True)
