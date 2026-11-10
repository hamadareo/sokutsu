"""STT ワーカーを実際に回して、回線が遅いときの予想訳の先出しを確かめる(11/06)。

リモートサーバーの耳が遅い → このMacの耳の結果で原文と予想訳を先に出す → リモートサーバーの結果が
来たらそれを確定として訳へ回す(確定の品質は リモートサーバーのまま)。
"""
import queue
import threading
import time
import unittest

import numpy as np

from tsuyaku.__main__ import Bridge
from tsuyaku.speculate import Speculator
from tsuyaku.stt import RemoteSTT


class FastLocal:
    fast = True
    langs = ()

    def transcribe(self, samples):
        return "こんにちは。", "ja", 1.0


class Resp:
    def __init__(self, obj):
        self.obj = obj

    def raise_for_status(self):
        pass

    def json(self):
        return self.obj


def run_worker(remote_delay, remote_text="こんにちは。"):
    def post(url, data=None, timeout=None):
        time.sleep(remote_delay)
        return Resp({"text": remote_text, "lang": "ja", "prob": 0.95})

    b = Bridge()
    b.stt = RemoteSTT("http://x", FastLocal(), post=post, hedge_s=4.0, preview_s=0.1)
    b.fine_seg = True
    b.capture = None
    b.running.set()
    translated = []
    b.spec = Speculator(run=lambda msgs, od: (od("Hello."), "Hello.")[1],
                        make_msgs=lambda t, l: t,
                        show=lambda text, trans, lang, g: b.ui.put(("partial", text, trans, lang, g)))
    seg_q, text_q = queue.Queue(), queue.Queue()
    th = threading.Thread(target=b._stt_worker, args=(seg_q, text_q, ("ja", "en")), daemon=True)
    th.start()
    rng = np.random.default_rng(0)
    seg = (rng.standard_normal(16000) * 0.1).astype(np.float32)
    seg_q.put((seg, 0.0, 1.0, 0.0))
    got = None
    try:
        got = text_q.get(timeout=5)
    except queue.Empty:
        pass
    time.sleep(0.1)
    b.running.clear()
    th.join(2)
    events = []
    while not b.ui.empty():
        events.append(b.ui.get())
    return got, events, b


class TestPreviewFlow(unittest.TestCase):
    def test_slow_remote_shows_guess_first_and_final_is_remote(self):
        got, events, b = run_worker(0.5, remote_text="こんにちは、皆さん。")
        partials = [e for e in events if e[0] == "partial"]
        self.assertTrue(any(e[1] == "こんにちは。" for e in partials), events)   # 先出しの原文
        self.assertTrue(any(len(e) > 4 and e[4] and e[2] == "Hello." for e in partials), events)
        self.assertEqual(got[0], "こんにちは、皆さん。")                       # 確定は remote
        self.assertEqual(b.stt.last_where, "remote")

    def test_fast_remote_has_no_local_preview(self):
        got, events, b = run_worker(0.0)
        self.assertEqual(b.stt.previewed, 0)
        self.assertEqual(got[0], "こんにちは。")


if __name__ == "__main__":
    unittest.main()
