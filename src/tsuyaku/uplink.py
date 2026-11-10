"""リモートサーバーの耳へ声を「話している間に」先に送っておく(11/06 先送り)。

見つけた問題(11/06 英会話の実時間評価 scripts/eval_live.py, AMI 会議):
  ふつうの回線では、話し終わってから区間の音声をまとめて送ると リモートサーバーの耳の往復が
  中央1.9秒・p90 3.9秒(4秒の音声=128KB。音声なしの往復は中央0.2秒)。続き待ちの
  繋ぎ直し(rejoin)は前の断片ごと送り直すので15秒=480KBにもなり、4秒の時間切れで
  このMacの耳(誤りが多い)に落ちていた。往復のあいだ STT ワーカーは止まり、
  途中字幕も出せなかった。
方式: マイクの区切り(StreamSegmenter)が「話している区間」に入ったら、その声
  (AutoGain 後、区間の手前の補いも含む)を通算サンプル番号つきで0.25秒ごとに
  送っておく。聞くときは「どこからどこまで」(recipe)と、まだ届いていない
  残りだけを送る。whisper に入る音声は今までと1サンプルも変わらない(品質は同じ)。
  届いていない・サーバーが古い・失敗したら、今までどおり音声ごと送る。
"""
from __future__ import annotations

import threading
import time
import uuid

import numpy as np

SR = 16000


def to_pcm16(x) -> np.ndarray:
    """RemoteSTT._remote と同じ量子化(float32 → int16)。"""
    a = np.asarray(x)
    if a.dtype == np.int16:
        return a.astype("<i2", copy=False)
    return (np.clip(np.asarray(a, dtype=np.float32), -1.0, 1.0) * 32767.0).astype("<i2")


def _merge(iv: list) -> list:
    iv = sorted(iv)
    out: list = []
    for a, b in iv:
        if out and a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return out


def _subtract(a: int, b: int, have: list) -> list:
    """[a,b) から have(整列・併合済み)に含まれる所を除いた範囲。"""
    out, cur = [], a
    for x, y in have:
        if y <= cur:
            continue
        if x >= b:
            break
        if x > cur:
            out.append((cur, min(x, b)))
        cur = max(cur, y)
        if cur >= b:
            break
    if cur < b:
        out.append((cur, b))
    return out


class Uplink:
    def __init__(self, url: str, post=None, batch_s: float = 0.25, keep_s: float = 150.0,
                 sid: str | None = None, start: bool = True):
        self.url = url.rstrip("/")
        self.sid = sid or uuid.uuid4().hex[:16]
        self.batch_s = batch_s
        self.keep = int(keep_s * SR)
        self._post = post
        self._pool: list = []                   # 繋ぎっぱなしの接続
        self._buf = np.zeros(0, dtype="<i2")   # 手元の写し(本文で送り直すとき用)
        self._base = 0                          # _buf[0] の通算サンプル番号
        self._have: list = []                   # 手元にある範囲
        self._acked: list = []                  # サーバーに届いた範囲
        self._todo: list = []                   # 送る範囲
        self._lock = threading.Lock()
        self._cv = threading.Condition(self._lock)
        self._stop = False
        self.fails = 0
        self.sent_bytes = 0
        self.enabled = True
        self._th = None
        if start:
            self._th = threading.Thread(target=self._loop, daemon=True)
            self._th.start()

    # ---------- マイクのスレッドから ----------

    def feed(self, off: int, x) -> None:
        """通算 off サンプル目からの声(区間の中)。手元に写して送る行列に積む。"""
        if not self.enabled:
            return
        pcm = to_pcm16(x)
        n = pcm.size
        if n == 0 or off < 0:
            return
        with self._cv:
            end = off + n
            if self._buf.size == 0:
                self._base = off
            if end > self._base + self._buf.size:
                self._buf = np.concatenate(
                    [self._buf, np.zeros(end - self._base - self._buf.size, dtype="<i2")])
            if off >= self._base:
                self._buf[off - self._base:end - self._base] = pcm
            over = self._buf.size - self.keep
            if over > 0:
                self._buf = self._buf[over:]
                self._base += over
                self._have = [[max(a, self._base), b] for a, b in self._have if b > self._base]
                self._acked = [[max(a, self._base), b] for a, b in self._acked if b > self._base]
                # 送る行列も同じく詰める(11/07 監査: 回線断が keep_s を超えると、手元から消えた
                # 範囲が行列の先頭に残り、戻った後も空の本文を同じ所へ永久に送り続けていた)
                self._todo = [[max(a, self._base), b] for a, b in self._todo if b > self._base]
            self._have = _merge(self._have + [[max(off, self._base), end]])
            self._todo = _merge(self._todo + [[max(off, self._base), end]])
            self._cv.notify()

    # ---------- 耳(RemoteSTT)から ----------

    def plan(self, span) -> "tuple[str, list, bytes] | None":
        """span = [(a, b) か ("z", n)] → (recipe, まだ届いていない範囲, その声の本文)。
        手元に無い範囲がある(古すぎる等)なら None(呼び出し側は音声ごと送る)。"""
        if not self.enabled or not span:
            return None
        parts, need = [], []
        with self._lock:
            for a, b in span:
                if a == "z":
                    parts.append(f"z{int(b)}")
                    continue
                a, b = int(a), int(b)
                if b <= a:
                    continue
                if _subtract(a, b, self._have):
                    return None
                parts.append(f"{a}-{b}")
                need += _subtract(a, b, self._acked)
            need = [list(r) for r in _merge([list(r) for r in need])]
            body = b"".join(self._buf[a - self._base:b - self._base].tobytes() for a, b in need)
        return ",".join(parts), [tuple(r) for r in need], body

    def acked(self, ranges) -> None:
        """本文で送った範囲もサーバーに届いた(先送りで送り直さない)。"""
        with self._lock:
            self._acked = _merge(self._acked + [list(r) for r in ranges])
            for a, b in ranges:
                self._todo = [list(r) for x, y in self._todo for r in _subtract(x, y, [[a, b]])]

    def forget(self, span) -> None:
        """サーバーに無いと言われた(409): その範囲は次から本文で送り直す。"""
        with self._lock:
            for a, b in span:
                if a == "z":
                    continue
                self._acked = [list(r) for x, y in self._acked for r in _subtract(x, y, [[a, b]])]

    def close(self) -> None:
        with self._cv:
            self._stop = True
            self._cv.notify()

    # ---------- 送る ----------

    def post(self, url, data, timeout):
        """繋ぎっぱなしの接続で送る(SSH 転送の通り道を毎回開け直さない)。接続は
        使い回しの束から1本借りて返す(同時に2本使う: 先送りと聞き取り)。"""
        if self._post is not None:
            return self._post(url, data=data, timeout=timeout)
        with self._lock:
            s = self._pool.pop() if self._pool else None
        if s is None:
            import requests
            s = requests.Session()
        try:
            return s.post(url, data=data, timeout=timeout)
        except Exception:
            s.close()      # 壊れた接続は戻さない
            s = None
            raise
        finally:
            if s is not None:
                with self._lock:
                    self._pool.append(s)

    def _loop(self) -> None:
        while True:
            with self._cv:
                while not self._todo and not self._stop:
                    self._cv.wait(1.0)
                if self._stop:
                    return
            time.sleep(self.batch_s)   # 0.25秒分ずつまとめて送る
            with self._lock:
                if not self._todo:
                    continue
                a, b = self._todo[0]
                a = max(a, self._base)
                if b <= a:              # もう手元に無い古い範囲: 送れないので捨てる
                    self._todo.pop(0)
                    continue
                b = min(b, a + 30 * SR)
                body = self._buf[a - self._base:b - self._base].tobytes()
            try:
                r = self.post(f"{self.url}/append?sid={self.sid}&off={a}", data=body, timeout=5.0)
                r.raise_for_status()
                self.sent_bytes += len(body)
                self.acked([(a, b)])
                self.fails = 0
            except Exception:  # noqa: BLE001 - 届かなければ聞くときに本文で送る
                self.fails += 1
                time.sleep(min(2.0, 0.2 * self.fails))


def server_streams(url: str, timeout: float = 2.0) -> bool:
    """リモートサーバーの耳が先送り(/append と範囲での聞き取り)に対応しているか。"""
    try:
        import requests
        r = requests.get(f"{url.rstrip('/')}/health", timeout=timeout)
        return bool(r.ok and r.json().get("stream"))
    except Exception:  # noqa: BLE001
        return False
