"""リモートサーバーへの転送(SSH トンネル, scripts/remote_tunnel.sh)を会話の途中で張り直す(11/08 speed)。

見つけた問題: 会話の途中で 127.0.0.1:12345 の転送が切れると(回線の瞬断・SSH の切断)、以後の訳が全部
「翻訳できませんでした」になった。転送を張るのは「会話を始める」の時だけだった。
翻訳の接続が「届かない/切られた」で失敗したら、裏で1回だけ張り直し(冪等なスクリプト。張れていれば何もしない)、
翻訳のやり直しはそれを最大 WAIT_S 秒待つ。応答が遅いだけ(時間切れ)では張り直さない。配る版では使わない。
"""
from __future__ import annotations

import subprocess
import threading
import time
from pathlib import Path

REMOTE_BASES = ("http://127.0.0.1:12345", "http://localhost:12345")
MIN_GAP_S = 20.0      # 張り直しの間隔(切れ続けるときに何度も走らせない)
WAIT_S = 4.0          # 翻訳のやり直しが張り直しを待つ上限

_lock = threading.Lock()
_state: dict = {"ev": None, "at": -1e9}


def _reset() -> None:
    with _lock:
        _state["ev"], _state["at"] = None, -1e9


def _run_script() -> None:
    script = Path(__file__).resolve().parents[2] / "scripts" / "remote_tunnel.sh"
    if script.exists():
        subprocess.run(["bash", str(script)], timeout=60, check=False,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def _is_conn_error(e: BaseException) -> bool:
    try:
        import requests
        if isinstance(e, requests.Timeout):
            return False
        return isinstance(e, requests.ConnectionError)
    except ImportError:
        return isinstance(e, ConnectionError)


def note_failure(base: str, e: BaseException) -> "threading.Event | None":
    """翻訳の接続の失敗。リモートサーバーの転送が届かない/切られたなら張り直しを始め(すでに走っていればそれ)、
    終わったら立つ Event を返す。張り直さないときは None。"""
    from .distenv import DIST
    if DIST or (base or "").rstrip("/") not in REMOTE_BASES or not _is_conn_error(e):
        return None
    with _lock:
        ev = _state["ev"]
        if ev is not None and not ev.is_set():
            return ev
        if time.monotonic() - _state["at"] < MIN_GAP_S:
            return None
        ev = threading.Event()
        _state["ev"], _state["at"] = ev, time.monotonic()

    def go():
        print("[tunnel] リモートサーバーへの転送が切れたので張り直します", flush=True)
        try:
            _run_script()
        except Exception as ex:  # noqa: BLE001
            print(f"[tunnel] 張り直せませんでした({type(ex).__name__})", flush=True)
        finally:
            ev.set()
    threading.Thread(target=go, daemon=True).start()
    return ev


def repairing() -> "threading.Event | None":
    with _lock:
        ev = _state["ev"]
    return ev if ev is not None and not ev.is_set() else None
