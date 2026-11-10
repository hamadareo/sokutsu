"""配る版の翻訳エンジン: 同梱の llama-server(llama.cpp 公式の macOS arm64 ビルド、Metal)を
127.0.0.1 の空いたポートで起こし、窓を閉じたら止める。

モデルは初回に modelfetch.py が取った gemma-3-4b-it QAT Q4_0 GGUF(今の Mac の既定の LM Studio の
gemma-3-4b MLX QAT 4bit と同じ QAT の重み)。--alias で今と同じ id "google/gemma-3-4b" を名乗る。
ユーザー本人の版(DIST=False)では ensure_for は何もしない。
"""
from __future__ import annotations

import os
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

from . import distenv
from .distenv import DIST

MODEL_FILE = "gemma-3-4b-it-qat-Q4_0.gguf"
ALIAS = "google/gemma-3-4b"
# 2本(-np 2)で 16k を分け合う(-kvu: 長い会議メモの1本は全体を使える)。
# gemma 3 は SWA(窓のある注意)。既定の小さい KV だと前の依頼の頭を使い回せず、毎回プロンプト
# (約300トークン)を最初から計算していた: 11/07 実測 初字 中央 LM Studio 0.41 → 同梱 1.02 秒、
# 確定字幕の遅れ 中央 2.28 → 5.12 秒。--swa-full で全部残すと、2回目からは新しい 16〜19 トークン
# だけ(0.6 → 0.06〜0.1 秒)。KV は q8_0(フラッシュアテンションが要る)で、16k で約 1.1 GB。
CTX = 16384
PARALLEL = 2       # 先読み翻訳と確定の訳、会議メモの抜き出しを2本並べる


def server_bin() -> Path:
    """同梱の llama-server。Sokutsu.app/Contents/Resources/{python,llama}/(sys.prefix = python)。"""
    env = os.environ.get("SOKUTSU_LLAMA_SERVER")
    if env:
        return Path(env)
    return Path(sys.prefix).resolve().parent / "llama" / "llama-server"


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class LocalLLM:
    def __init__(self, binary: Path, model: Path, port: "int | None" = None,
                 log_path: "Path | None" = None, ctx: int = CTX, parallel: int = PARALLEL,
                 popen=subprocess.Popen, getter=None, sleep=time.sleep, watchdog: bool = True,
                 clock=time.monotonic, spawn=None):
        self.binary, self.model = Path(binary), Path(model)
        self.port = port or free_port()
        self.log_path = log_path
        self.ctx, self.parallel = ctx, parallel
        self.popen, self.sleep, self.watchdog = popen, sleep, watchdog
        if getter is None:
            import requests
            getter = requests.get
        self.getter = getter
        self.proc = None
        self._dog = None
        self._ok = False
        self.error = ""
        self._lock = threading.Lock()
        # 会議の途中で落ちた・応答しない時の立ち直り(note_failure / note_ok)
        self.clock = clock
        self.spawn = spawn or (lambda fn: threading.Thread(target=fn, daemon=True,
                                                           name="llm-recover").start())
        self._flock = threading.Lock()
        self._fail_since: "float | None" = None
        self._recovering = False
        self._recovered = False       # この失敗の続きで、もう起こし直した

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def command(self) -> list[str]:
        return [str(self.binary), "-m", str(self.model), "--host", "127.0.0.1",
                "--port", str(self.port), "--alias", ALIAS, "-ngl", "999",
                "-c", str(self.ctx), "-np", str(self.parallel), "-kvu", "--swa-full",
                "-fa", "on", "-ctk", "q8_0", "-ctv", "q8_0", "--no-webui"]

    def running(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def healthy(self, timeout: float = 1.0) -> bool:
        try:
            return self.getter(f"{self.url}/health", timeout=timeout).status_code == 200
        except Exception:  # noqa: BLE001
            return False

    def start(self) -> None:
        out = subprocess.DEVNULL
        if self.log_path is not None:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            out = open(self.log_path, "ab")   # noqa: SIM115 - 子プロセスが書き続ける
        try:
            self.proc = self.popen(self.command(), stdout=out, stderr=subprocess.STDOUT,
                                   stdin=subprocess.DEVNULL, start_new_session=True)
        finally:
            if out is not subprocess.DEVNULL:
                out.close()            # 子プロセスは自分の写しで書く(起こし直すたびに開きっぱなしにしない)
        print(f"[llm] 同梱の翻訳エンジンを起こしました(port {self.port})", flush=True)
        self._stop_dog()   # 起こし直し: 前の見張りは、もう居ない(使い回されうる)pid を kill する
        if self.watchdog:
            # アプリが落ちても llama-server(数 GB)が残らないよう、アプリの pid を見張る
            sh = (f"while kill -0 {os.getpid()} 2>/dev/null; do sleep 2; done; "
                  f"kill {self.proc.pid} 2>/dev/null")
            try:
                self._dog = subprocess.Popen(["/bin/sh", "-c", sh], start_new_session=True,
                                             stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                             stderr=subprocess.DEVNULL)
            except Exception:  # noqa: BLE001
                self._dog = None

    def ensure(self, timeout: float = 180.0) -> bool:
        """動いていなければ起こし、/health が 200 になるまで待つ(モデルの読み込みは数秒〜)。"""
        with self._lock:
            if self._ok and self.running():
                return True
            if not self.model.is_file():
                self.error = f"翻訳モデルがありません({self.model.name})"
                return False
            if not self.binary.is_file():
                self.error = "翻訳エンジン(llama-server)がありません"
                return False
            if not self.running():
                self.start()
            t_end = time.monotonic() + timeout
            while time.monotonic() < t_end:
                if self.healthy():
                    self._ok = True
                    self.error = ""
                    return True
                if not self.running():
                    self.error = "翻訳エンジンが起動できませんでした(~/Library/Logs/Sokutsu/llama-server.log)"
                    print(f"[llm] {self.error}", flush=True)
                    return False
                self.sleep(0.25)
            self.error = "翻訳エンジンの準備が時間内に終わりませんでした"
            return False

    def stop(self) -> None:
        self._ok = False
        p, self.proc = self.proc, None
        if p is not None and p.poll() is None:
            p.terminate()
            try:
                p.wait(timeout=5)
            except Exception:  # noqa: BLE001
                p.kill()
        self._stop_dog()

    def _stop_dog(self) -> None:
        d, self._dog = self._dog, None
        if d is not None:
            try:
                d.kill()
                d.wait(timeout=2)
            except Exception:  # noqa: BLE001
                pass


    # ---------- 会議の途中の立ち直り ----------

    HUNG_S = 3.0   # 動いているのに応答しない時、この秒数失敗が続いたら起こし直す

    def note_failure(self) -> bool:
        """翻訳の接続が失敗した。落ちていればすぐ、応答しないだけなら HUNG_S 秒続いたら、
        裏で1回だけ起こし直す(会話は止めない。同時に何度も起こさない)。起こし始めたら True。"""
        now = self.clock()
        with self._flock:
            if self._fail_since is None:
                self._fail_since = now
            if self._recovering or self._recovered:
                return False
            if self.running() and now - self._fail_since < self.HUNG_S:
                return False
            self._recovering = self._recovered = True
        self.spawn(self._recover)
        return True

    def note_ok(self) -> None:
        with self._flock:
            self._fail_since = None
            self._recovered = False

    def _recover(self) -> None:
        try:
            with self._lock:
                dead = not self.running()
                if not dead and self.healthy(timeout=2.0):
                    return                      # 一時的だった
                print(f"[llm] 翻訳エンジンが{'止まっていた' if dead else '応答しない'}ので起こし直します",
                      flush=True)
                self.stop()
            self.ensure()
        except Exception as e:  # noqa: BLE001
            print(f"[llm] 起こし直せませんでした({type(e).__name__}: {e})", flush=True)
        finally:
            with self._flock:
                self._recovering = False


_INSTANCE: "LocalLLM | None" = None
_ILOCK = threading.Lock()


def instance() -> LocalLLM:
    global _INSTANCE
    with _ILOCK:
        if _INSTANCE is None:
            data = distenv.DATA or distenv.data_dir()
            _INSTANCE = LocalLLM(server_bin(), data / "models" / MODEL_FILE,
                                 log_path=distenv.app_logs_dir() / "llama-server.log")
        return _INSTANCE


def url() -> str:
    return instance().url


def ensure_for(base: str, timeout: float = 180.0) -> bool:
    """配る版で、base が同梱のエンジンならそれを起こして待つ。それ以外は何もしない。"""
    if not DIST:
        return False
    inst = instance()
    if (base or "").rstrip("/") != inst.url:
        return False
    return inst.ensure(timeout)


def _bundled(base: str) -> "LocalLLM | None":
    if not DIST:
        return None
    inst = instance()
    return inst if (base or "").rstrip("/") == inst.url else None


def note_failure(base: str) -> None:
    """翻訳の接続が失敗した(translate から)。配る版の同梱エンジンの時だけ立ち直りを試す。"""
    inst = _bundled(base)
    if inst is not None:
        inst.note_failure()


def note_ok(base: str) -> None:
    inst = _bundled(base)
    if inst is not None:
        inst.note_ok()


def stop() -> None:
    if _INSTANCE is not None:
        _INSTANCE.stop()
