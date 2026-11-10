"""読み上げ中も聞き続ける: エコーを消すマイク + 読み上げの再生をひとつの部品で(11/08 試作)。

見つけた問題: 読み上げ中はマイクを止めている(__main__._tts_worker が capture.paused を立てる)。
自分の読み上げを拾って訳し直さないためだが、その間に相手が話した言葉が落ちる。

やり方: 小さな部品 bin/sokutsu-aecio(native/aecio/aecio.c)が macOS の VoiceProcessingIO を開き、
標準入力で受けた読み上げの音を鳴らしながら、マイクの音から「いま鳴らした音」の回り込みを引いて
標準出力へ流す。VPIO が引けるのは自分で鳴らした音だけなので、afplay ではなくこの部品で鳴らす。
ここではマイクの音を 512 サンプルずつ(SpeechCapture.on_chunk と同じ大きさ)callback へ渡し、
play() で1件鳴らして終わるまで待つ。cancel() で止める。

約束(部品の標準入力): 4バイトの uint32 LE = サンプル数 n、続けて float32 16 kHz モノラル n 個。
n = 0 は「止める」。標準エラーの PLAY_START / PLAY_DONE id=… で始まり・終わりが分かる。
id は部品が受けた順に 1, 2, 3 …(ここでも同じ順に数える)。

部品が無い・許可が無い・READY が来ない・落ちた → available が False。呼ぶ側はいつもの道
(SpeechCapture + afplay、読み上げ中はマイクを止める)を使う。
"""
from __future__ import annotations

import os
import re
import struct
import subprocess
import sys
import threading
from pathlib import Path
from typing import Callable

import numpy as np

from .distenv import DIST

ROOT = Path(__file__).resolve().parents[2]
HELPER = ROOT / "bin" / "sokutsu-aecio"
if DIST:   # 配る版: Sokutsu.app/Contents/MacOS/ の起動役の隣
    HELPER = Path(sys.executable).resolve().parent / "sokutsu-aecio"

SR = 16000
CHUNK = 512

# 部品の終了コード(aecio.c と同じ)
RC_PROTOCOL = 2
RC_PERMISSION = 3
RC_UNSUPPORTED = 4
RC_FAILED = 5
RC_DEVICE_CHANGED = 6

FLUSH = struct.pack("<I", 0)
_KV = re.compile(r"(\w+)=(\S+)")


def encode_clip(samples) -> bytes:
    """1件ぶんの約束のバイト列(長さ + float32 LE)。空の音は送れない(0 は「止める」なので)。"""
    x = np.ascontiguousarray(np.clip(np.asarray(samples, dtype=np.float32).ravel(), -1.0, 1.0),
                             dtype="<f4")
    if not len(x):
        raise ValueError("空の音は送れません")
    return struct.pack("<I", len(x)) + x.tobytes()


def _kv(line: str) -> dict:
    return dict(_KV.findall(line))


class Chunker:
    """バイト列を受けて float32 の 512 サンプルずつに切る(半端は次へ持ち越す)。"""

    def __init__(self, chunk: int = CHUNK):
        self.chunk = chunk
        self._bytes = b""
        self._buf = np.zeros(0, dtype=np.float32)

    def feed(self, data: bytes) -> "list[np.ndarray]":
        data = self._bytes + data
        k = len(data) - len(data) % 4
        self._bytes = data[k:]
        if k:
            self._buf = np.concatenate([self._buf, np.frombuffer(data[:k], dtype="<f4")])
        out = []
        n = len(self._buf) // self.chunk
        for i in range(n):
            out.append(self._buf[i * self.chunk:(i + 1) * self.chunk].astype(np.float32))
        self._buf = self._buf[n * self.chunk:]
        return out


class EchoCancelIO:
    """部品を子プロセスで動かす。on_chunk(mono512) にエコーを消したマイクの音が届く。

    on_status(kind, message): "running"(動き始めた)/ "failed"(諦めた)/ "stopped"。
    自動の起こし直しはしない(落ちたら available が False になるので、呼ぶ側がいつもの道へ戻る)。
    """

    def __init__(self, on_chunk: Callable[[np.ndarray], None], cmd: "list[str] | None" = None,
                 on_status: "Callable[[str, str], None] | None" = None,
                 start_timeout: float = 5.0, chunk: int = CHUNK):
        self.on_chunk = on_chunk
        self.cmd = list(cmd) if cmd else [str(HELPER)]
        self.on_status = on_status or (lambda k, m: None)
        self.start_timeout = start_timeout
        self.chunk = chunk
        self.state = "idle"
        self.error = ""
        self.info: dict = {}
        self.returncode: "int | None" = None
        self.samples = 0                    # マイクから受けたサンプル数
        self.last_cancelled = False         # 直前の play() が止められたか
        self._proc: "subprocess.Popen | None" = None
        self._ready = threading.Event()
        self._dead = threading.Event()
        self._stopping = False
        self._wlock = threading.Lock()      # 標準入力へ書く(1件の途中に「止める」を挟まない)
        self._plock = threading.Lock()      # _next_id / _waits
        self._next_id = 1
        self._waits: dict = {}              # id -> [started Event, done Event, cancelled, on_start]
        self._threads: list = []

    # ---------- 起動・終了 ----------
    @property
    def available(self) -> bool:
        p = self._proc
        return (self._ready.is_set() and not self._dead.is_set() and p is not None
                and p.poll() is None)

    def start(self) -> bool:
        """部品を起動して READY を待つ。使えれば True(だめなら False、available も False)。"""
        self.state = "starting"
        try:
            self._proc = subprocess.Popen(self.cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                          stderr=subprocess.PIPE, bufsize=0)
        except (FileNotFoundError, PermissionError, OSError) as e:
            self._fail(f"エコーを消す部品がありません({Path(self.cmd[-1]).name}: "
                       f"{type(e).__name__})。scripts/build_aecio.sh で作れます")
            return False
        for target, name in ((self._read_err, "aecio-err"), (self._read_out, "aecio-out")):
            t = threading.Thread(target=target, daemon=True, name=name)
            t.start()
            self._threads.append(t)
        # READY か、部品が終わる(許可が無い等)まで待つ
        waited = 0.0
        while waited < self.start_timeout and not self._ready.is_set() and not self._dead.is_set():
            self._ready.wait(0.05)
            waited += 0.05
        if self._ready.is_set() and not self._dead.is_set():
            if self.info.get("rate", "16000") != str(SR):
                self._fail(f"部品の周波数が {self.info.get('rate')} Hz(16000 Hz が必要)")
                self.stop()
                return False
            self.state = "running"
            self._status("running", " ".join(f"{k}={v}" for k, v in self.info.items()))
            return True
        timed_out = not self._dead.is_set()
        self.stop()                        # 片付け(管も閉じる)
        if timed_out:
            self._fail(self.error or "エコーを消す部品が動き始めませんでした(時間切れ)")
        else:
            self._fail(self._exit_message())
        return False

    def stop(self, timeout: float = 2.0) -> None:
        self._stopping = True
        p = self._proc
        if p is not None and p.poll() is None:
            try:
                if p.stdin:
                    p.stdin.close()        # 部品は標準入力が閉じたら片付けて終わる(書きかけの play は失敗で返る)
                p.wait(timeout=1.0)
            except Exception:  # noqa: BLE001
                pass
            if p.poll() is None:
                try:
                    p.terminate()
                    p.wait(timeout=1.0)
                except Exception:  # noqa: BLE001
                    p.kill()
        for t in self._threads:
            t.join(timeout=timeout)
        self._release_all()
        if p is not None:
            for f in (p.stdin, p.stdout, p.stderr):
                try:
                    if f:
                        f.close()
                except Exception:  # noqa: BLE001
                    pass
        if self.state != "failed":
            self.state = "stopped"

    # ---------- 再生 ----------
    def play(self, samples, on_start: "Callable[[], None] | None" = None,
             timeout: "float | None" = None) -> bool:
        """16 kHz モノラル float32 を1件鳴らし、部品が「鳴らし終えた」(または止めた)と言うまで待つ。

        True = 終わりの知らせを受けた(止めた場合は last_cancelled が True)。
        False = 使えない・送れない・部品が落ちた・時間切れ(呼ぶ側は afplay で鳴らし直せる)。
        on_start は最初のサンプルを鳴らし始めたとき(PLAY_START)に呼ぶ。
        """
        self.last_cancelled = False
        if not self.available:
            return False
        try:
            data = encode_clip(samples)
        except ValueError:
            return False
        n = (len(data) - 4) // 4
        if timeout is None:
            timeout = n / SR + 5.0
        wait = [threading.Event(), threading.Event(), False, on_start]
        with self._wlock:
            with self._plock:
                cid = self._next_id
                self._next_id += 1
                self._waits[cid] = wait
            if not self._write(data):
                with self._plock:
                    self._waits.pop(cid, None)
                return False
        ok = wait[1].wait(timeout)
        with self._plock:
            self._waits.pop(cid, None)
        if not ok or wait[2] is None:     # 時間切れ / 部品が終わって起こされた
            return False
        self.last_cancelled = bool(wait[2])
        return True

    def cancel(self) -> bool:
        """鳴っている分と待っている分を止める(部品へ n = 0 を送る)。送れたら True。"""
        if not self.available:
            return False
        with self._wlock:
            return self._write(FLUSH)

    def _write(self, data: bytes) -> bool:
        p = self._proc
        if p is None or p.stdin is None:
            return False
        fd = p.stdin.fileno()
        view = memoryview(data)
        try:
            while len(view):
                k = os.write(fd, view)
                view = view[k:]
            return True
        except (OSError, ValueError):
            return False

    # ---------- 読む側 ----------
    def _read_out(self) -> None:
        p = self._proc
        ch = Chunker(self.chunk)
        fd = p.stdout.fileno()
        while True:
            try:
                data = os.read(fd, 16384)
            except OSError:
                break
            if not data:
                break
            for c in ch.feed(data):
                self.samples += len(c)
                try:
                    self.on_chunk(c)
                except Exception as e:  # noqa: BLE001 - 1チャンク捨てても止めない
                    print(f"[aecio] チャンクの処理に失敗 {type(e).__name__}: {e}", flush=True)

    def _read_err(self) -> None:
        p = self._proc
        for raw in iter(p.stderr.readline, b""):
            line = raw.decode("utf-8", "replace").strip()
            if not line:
                continue
            head = line.split(" ", 1)[0]
            if head not in ("PLAY_START", "PLAY_DONE"):
                print(f"[aecio] {line}", flush=True)
            if head == "READY":
                self.info = _kv(line)
                self._ready.set()
            elif head == "ERROR":
                self.error = line[5:].strip()
            elif head in ("PLAY_START", "PLAY_DONE"):
                kv = _kv(line)
                try:
                    cid = int(kv.get("id", "0"))
                except ValueError:
                    continue
                with self._plock:
                    w = self._waits.get(cid)
                if w is None:
                    continue
                if head == "PLAY_START":
                    w[0].set()
                    if w[3] is not None:
                        try:
                            w[3]()
                        except Exception:  # noqa: BLE001
                            pass
                else:
                    w[2] = kv.get("cancelled") == "1"
                    w[1].set()
        # 標準エラーが閉じた = 部品が終わった
        try:
            self.returncode = p.wait(timeout=2.0)
        except Exception:  # noqa: BLE001
            self.returncode = p.poll()
        self._dead.set()
        self._release_all()
        if self.state == "running" and not self._stopping:
            self._fail(self._exit_message())

    def _release_all(self) -> None:
        """待っている play() を起こす(部品が終わった)。"""
        with self._plock:
            for w in self._waits.values():
                if not w[1].is_set():
                    w[2] = None
                    w[1].set()

    def _exit_message(self) -> str:
        rc = self.returncode
        if rc == RC_PERMISSION:
            return ("マイクの許可がありません。システム設定 > プライバシーとセキュリティ > マイク で"
                    "許可してください")
        if rc == RC_UNSUPPORTED:
            return "この Mac ではエコーを消す部品が使えません(" + (self.error or "VPIO が無い") + ")"
        if rc == RC_DEVICE_CHANGED:
            return "入力か出力の機器が変わりました(作り直しが必要)"
        return f"エコーを消す部品が止まりました({self.error or f'終了コード {rc}'})"

    def _status(self, kind: str, msg: str) -> None:
        try:
            self.on_status(kind, msg)
        except Exception:  # noqa: BLE001
            pass

    def _fail(self, msg: str) -> None:
        self.state = "failed"
        self.error = msg
        self._status("failed", msg)
