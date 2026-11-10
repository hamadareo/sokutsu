"""先読み翻訳(11/06): 文が閉じる前に「ここで終わる」と予想して訳を始めておく。

見つけた問題: 実ログ(11/06 実会話)で、文末で続きを待つ Endpointer の待ち
(「。」なし0.5秒・短い文0.3秒・読点や助詞1.2秒)のあいだ翻訳は止まっていて、
待ち終わってから0.3〜0.6秒かけて訳していた。字幕が出るまで = 待ち + 訳。

方式: Endpointer が断片を待ちに入れた瞬間に、その文字のまま訳を始める。
  - 続きが来ずに同じ文字で確定した(予想が当たった)→ その訳をそのまま使う。
    訳は待ちのあいだに終わっているので、字幕は待ち終わりと同時に出る。
  - 続きが来て文字が変わった(外れた)→ 走っている訳を途中で止め、
    繋がった文字で予想し直す。
  - 予想の訳は「予想訳」として大字幕に先に出す(guess=True)。聞き手は
    待ちの間にもう内容をつかめる。読み上げは確定してからだけ。
リモートサーバーの翻訳サーバーは2本同時でもほぼ遅くならず(実測 +0.05〜0.1秒)、
途中で止めても次の依頼に響かない(止めた直後の依頼の初字0.2秒)。

スレッド: update は STT ワーカー、take はテキストワーカー、訳は1件ごとの
小さなスレッド。状態は _lock、1件の中の文字列と受け手は job.lock で守る。
"""
from __future__ import annotations

import threading
import time
from typing import Callable

Key = tuple  # (text, lang)


class _Cancelled(Exception):
    pass


class _Job:
    def __init__(self, text: str, lang: str):
        self.text = text
        self.lang = lang
        self.acc: list[str] = []
        self.result: str | None = None
        self.done = threading.Event()
        self.cancel = threading.Event()
        self.lock = threading.Lock()
        self.sink: Callable[[str], None] | None = None   # take した人の on_delta
        self.reserved = False     # 確定済み(テキストワーカーが受け取りに来る)
        self.t0 = time.time()
        self.t_done = 0.0
        self.last_show = 0.0
        self.t_first_show = 0.0   # 予想訳を最初に画面に出した時刻
        self.preview = False      # このMacの耳の先出しから始めた予想(読み上げの音は作らない)


class Speculator:
    def __init__(self, run: Callable[[list, Callable[[str], None]], str],
                 make_msgs: Callable[[str, str], list],
                 show: Callable[[str, str, str, bool], None] | None = None,
                 ready: Callable[[str, str, str], None] | None = None,
                 max_reserved: int = 4, max_age_s: float = 30.0,
                 show_every_s: float = 0.12):
        self.run = run
        self.make_msgs = make_msgs
        self.show = show
        self.ready = ready      # 予想訳ができた(text, lang, 訳): 読み上げの音を先に作る
        self.max_reserved = max_reserved
        self.max_age_s = max_age_s
        self.show_every_s = show_every_s
        self._jobs: dict[Key, _Job] = {}
        self._latest: Key | None = None    # 今待っている断片(予想訳を出す対象)
        self._muted = False
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0
        self.last_first_show = 0.0   # 直前の take で受け取った予想訳が最初に出た時刻(0=出ていない)

    # ---------- STT ワーカーから ----------

    def update(self, pending: "Key | None", emitted: list, preview: bool = False) -> None:
        """Endpointer の状態を伝える。pending = 待っている断片 (text, lang) か None、
        emitted = 今確定して訳へ回した単位の (text, lang) の列。"""
        start: _Job | None = None
        with self._lock:
            em = {tuple(e) for e in emitted}
            now = time.time()
            for k, j in list(self._jobs.items()):
                if k in em:
                    j.reserved = True
                elif preview and not j.preview and now - j.t0 <= self.max_age_s:
                    # 途中字幕(このMacの耳)の予想では、リモートサーバーの続き待ちの断片の訳は止めない:
                    # 続きが来なければそのまま確定になり、訳がもう出来ている(11/06 AMI 実時間評価:
                    # 途中字幕が続き待ちの訳を止め、確定で訳し直していた)
                    continue
                elif (not j.reserved and k != pending) or now - j.t0 > self.max_age_s:
                    j.cancel.set()           # 外れた予想(続きが来た/捨てられた)
                    del self._jobs[k]
            if pending is not None and pending[0] and pending not in self._jobs:
                start = _Job(*pending)
                start.preview = preview
                self._jobs[pending] = start
            elif pending is not None and pending in self._jobs and not preview:
                self._jobs[pending].preview = False   # リモートサーバーの結果と同じ文字だった: 本物の予想
            self._latest = pending
            reserved = sorted((j.t0, k) for k, j in self._jobs.items() if j.reserved)
            for _t, k in reserved[: max(0, len(reserved) - self.max_reserved)]:
                self._jobs.pop(k).cancel.set()   # 誰も受け取りに来ない(捨てられた)分
        if start is not None:
            threading.Thread(target=self._work, args=(start,), daemon=True).start()

    # ---------- テキストワーカーから ----------

    def take(self, text: str, lang: str, on_delta: Callable[[str], None] | None,
             timeout: float = 30.0) -> str | None:
        """確定した文の訳を先読みから受け取る。当たりなら訳(途中までの分は
        on_delta にまとめて渡し、残りは届き次第渡す)、外れ/失敗なら None
        (呼び出し側は途中の文字を捨てて普通に訳す)。"""
        with self._lock:
            j = self._jobs.pop((text, lang), None)
            if self._latest == (text, lang):
                self._latest = None
        self.last_first_show = 0.0
        if j is None:
            self.misses += 1
            return None
        self.last_first_show = j.t_first_show
        with j.lock:
            j.sink = on_delta
            replay = "".join(j.acc)
            if on_delta and replay:
                on_delta(replay)
        if not j.done.wait(timeout):
            j.cancel.set()
            self.misses += 1
            return None
        if not j.result:
            self.misses += 1
            return None
        self.hits += 1
        return j.result

    def peek(self, text: str, lang: str) -> str:
        """今その断片について持っている予想訳(途中まででも)。無ければ ""。"""
        with self._lock:
            j = self._jobs.get((text, lang))
        if j is None:
            return ""
        with j.lock:
            return "".join(j.acc)

    def mute(self, on: bool) -> None:
        """確定訳を流している間は予想訳で大字幕を上書きしない。解除したら
        最新の予想を出し直す。"""
        self._muted = on
        if not on:
            with self._lock:
                j = self._jobs.get(self._latest) if self._latest else None
            if j is not None:
                self._show(j, force=True)

    def cancel_all(self) -> None:
        with self._lock:
            for j in self._jobs.values():
                j.cancel.set()
            self._jobs.clear()
            self._latest = None

    # ---------- 中身 ----------

    def _work(self, j: _Job) -> None:
        try:
            msgs = self.make_msgs(j.text, j.lang)
            j.result = self.run(msgs, lambda d: self._delta(j, d)) or None
        except _Cancelled:
            j.result = None
        except Exception as e:  # noqa: BLE001 - 先読みの失敗は本番の訳で取り返す
            print(f"[spec] 先読み失敗({type(e).__name__}): {str(e)[:80]}", flush=True)
            j.result = None
        finally:
            j.t_done = time.time()
            j.done.set()
        if j.result and not j.cancel.is_set():
            self._show(j, force=True)
            if self.ready is not None and not j.preview:
                try:
                    self.ready(j.text, j.lang, j.result)
                except Exception:  # noqa: BLE001
                    pass

    def _delta(self, j: _Job, d: str) -> None:
        if j.cancel.is_set():
            raise _Cancelled()
        with j.lock:
            j.acc.append(d)
            sink = j.sink
            if sink is not None:
                sink(d)
        if sink is None:
            self._show(j)

    def _show(self, j: _Job, force: bool = False) -> None:
        if self.show is None or self._muted or j.sink is not None:
            return
        if self._latest != (j.text, j.lang):
            return          # もう待っている断片ではない(確定 or 外れ)
        now = time.time()
        if not force and now - j.last_show < self.show_every_s:
            return
        with j.lock:
            trans = "".join(j.acc)
        if not trans:
            return
        j.last_show = now
        if not j.t_first_show:
            j.t_first_show = now
        self.show(j.text, trans, j.lang, True)
