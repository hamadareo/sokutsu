"""話し続けている最中の途中字幕(11/06 英会話を流して「もっとリアルタイムに」)。

見つけた問題: 途切れずに話し続けると1区間が10〜12.5秒になり(英会話の動画)、
区間が閉じるまで原文も訳も出なかった(話し始めから字幕まで12秒以上)。
変更: 話している最中も1.5秒ごとに「ここまでの音声」をこのMacの速い耳(SenseVoice,
1回0.1〜0.3秒)で聞き、原文と予想訳を出す。確定は今までどおり区間が閉じてから
リモートサーバーの耳で聞いた結果なので、品質は変わらない。
"""
import queue
import threading
import time
import types
import unittest

import numpy as np

from tsuyaku.__main__ import Bridge
from tsuyaku.audio import SpeechCapture, StreamSegmenter
from tsuyaku.speculate import Speculator
from tsuyaku.stt import RemoteSTT


def talking(sg, seconds, rng):
    for _ in range(int(seconds * 16000 / 512)):
        sg.feed((rng.standard_normal(512) * 0.2).astype(np.float32), True)


def fake_capture(interim_s=1.5):
    c = types.SimpleNamespace(sr=16000, segmenter=StreamSegmenter(16000), interim_s=interim_s,
                              interim_first_s=1.0, _interim=None, _interim_at=0,
                              _interim_seg=None)
    c._maybe_interim = lambda closed: SpeechCapture._maybe_interim(c, closed)
    c.take_interim = lambda: SpeechCapture.take_interim(c)
    return c


class TestCaptureInterim(unittest.TestCase):
    def test_interim_every_1_5s_while_talking(self):
        c, rng = fake_capture(), np.random.default_rng(0)
        got = []
        for _ in range(int(5.0 * 16000 / 512)):     # 5秒 途切れずに話す
            items = c.segmenter.feed((rng.standard_normal(512) * 0.2).astype(np.float32), True)
            c._maybe_interim(bool(items))
            it = c.take_interim()
            if it is not None:
                got.append(len(it[0]) / 16000)
        self.assertGreaterEqual(len(got), 3)          # 1秒・2.5秒・4秒あたり
        self.assertLess(got[0], 1.6)
        self.assertTrue(all(b > a for a, b in zip(got, got[1:])))   # ここまでの音声が伸びていく

    def test_off_when_zero(self):
        c, rng = fake_capture(0.0), np.random.default_rng(0)
        talking(c.segmenter, 3.0, rng)
        c._maybe_interim(False)
        self.assertIsNone(c.take_interim())

    def test_closed_segment_clears_interim(self):
        c, rng = fake_capture(), np.random.default_rng(0)
        talking(c.segmenter, 2.0, rng)
        c._maybe_interim(False)
        c._maybe_interim(True)
        self.assertIsNone(c.take_interim())

    def test_current_audio(self):
        sg, rng = StreamSegmenter(16000), np.random.default_rng(0)
        self.assertIsNone(sg.current_audio())
        talking(sg, 2.0, rng)
        self.assertGreater(len(sg.current_audio()), 1.5 * 16000)


class FastLocal:
    fast = True
    langs = ()

    def __init__(self):
        self.n = 0

    def transcribe(self, samples):
        self.n += 1
        return "How's it going? My name is", "en", 1.0


class Cap:
    def __init__(self):
        self.segmenter = types.SimpleNamespace(speaking_since=lambda: 0.0, noise=0.01)
        self.paused = threading.Event()
        self.items = [(np.zeros(16000 * 3, np.float32), 0.0)]

    def take_interim(self):
        return self.items.pop() if self.items else None

    def wall_of(self, s):
        return time.time()


class TestWorkerInterim(unittest.TestCase):
    def test_interim_shows_original_and_guess(self):
        b = Bridge()
        loc = FastLocal()
        b.stt = RemoteSTT("http://x", loc, post=lambda *a, **k: None, hedge_s=4.0)
        b.fine_seg = True
        b.capture = Cap()
        b.running.set()
        b.spec = Speculator(run=lambda msgs, od: (od("調子はどう？"), "調子はどう？")[1],
                            make_msgs=lambda t, l: t,
                            show=lambda text, trans, lang, g: b.ui.put(("partial", text, trans, lang, g)))
        th = threading.Thread(target=b._stt_worker, args=(queue.Queue(), queue.Queue(), ("ja", "en")),
                              daemon=True)
        th.start()
        time.sleep(0.6)
        b.running.clear()
        th.join(2)
        ev = []
        while not b.ui.empty():
            ev.append(b.ui.get())
        parts = [e for e in ev if e[0] == "partial"]
        self.assertTrue(any(e[1].startswith("How's it going") for e in parts), ev)
        self.assertTrue(any(len(e) > 4 and e[4] and e[2] == "調子はどう？" for e in parts), ev)
        self.assertEqual(loc.n, 1)                    # 1回だけ(最新の1つ)


class TestIdleTickKeepsGuess(unittest.TestCase):
    """11/06 AMI 実時間評価: 0.1秒ごとの待ちの tick が毎回「続き待ちなし」を先読みに伝え、
    途中字幕の予想訳を初字の前に止めていた(出た原文の7割に訳が出なかった)。"""

    def test_interim_guess_survives_idle_ticks(self):
        b = Bridge()
        loc = FastLocal()
        b.stt = RemoteSTT("http://x", loc, post=lambda *a, **k: None, hedge_s=4.0)
        b.fine_seg = True
        b.capture = Cap()
        b.running.set()

        def slow_llm(msgs, od):
            time.sleep(0.4)            # 家の回線の リモートサーバーの初字(0.1秒の tick を何回もまたぐ)
            od("調子")
            return "調子はどう？"
        b.spec = Speculator(run=slow_llm, make_msgs=lambda t, l: t,
                            show=lambda text, trans, lang, g: b.ui.put(("partial", text, trans, lang, g)))
        th = threading.Thread(target=b._stt_worker, args=(queue.Queue(), queue.Queue(), ("ja", "en")),
                              daemon=True)
        th.start()
        time.sleep(1.0)
        b.running.clear()
        th.join(2)
        ev = []
        while not b.ui.empty():
            ev.append(b.ui.get())
        self.assertTrue(any(e[0] == "partial" and len(e) > 4 and e[4] and e[2] == "調子" for e in ev), ev)


class TestWiring(unittest.TestCase):
    def test_setting(self):
        from pathlib import Path
        from tsuyaku.config import Settings
        self.assertAlmostEqual(Settings().interim_s, 1.5)
        src = (Path(__file__).resolve().parents[1] / "src/tsuyaku/__main__.py").read_text()
        self.assertIn("self.capture.interim_s = s.interim_s", src)


if __name__ == "__main__":
    unittest.main()


class TestStockPhraseRescue(unittest.TestCase):
    """11/06 擬似会議 v1 ja_dev4: 本物の「必要な台数を見積もっておきます」を リモートサーバーの耳が
    「ご視聴ありがとうございました」(p=0.67)と聞き、定型句の守りで捨てていた。このMacの耳
    (SenseVoice)はペア内の言語で聞けていた → そちらを採る。"""

    def _run(self, local_text, local_lang="ja"):
        b = Bridge()

        class Loc:
            fast = True
            langs = ()

            def transcribe(self, samples):
                return local_text, local_lang, 1.0

        def post(url, data=None, timeout=None):
            class R:
                status_code = 200

                def raise_for_status(self):
                    pass

                def json(self):
                    return {"text": "ご視聴ありがとうございました", "lang": "ja", "prob": 0.67, "sec": 0.1}
            return R()
        b.stt = RemoteSTT("http://x", Loc(), post=post, hedge_s=4.0)
        b.fine_seg = True
        b.capture = None
        b.running.set()
        b.spec = None
        seg_q, text_q = queue.Queue(), queue.Queue()
        seg_q.put((np.ones(16000 * 2, np.float32) * 0.1, 0.0, 2.0, 0.0))
        th = threading.Thread(target=b._stt_worker, args=(seg_q, text_q, ("ja", "en")), daemon=True)
        th.start()
        out = None
        t0 = time.time()
        while time.time() - t0 < 3.0 and out is None:
            try:
                out = text_q.get(timeout=0.1)
            except queue.Empty:
                pass
        b.running.clear()
        th.join(2)
        return out

    def test_local_in_pair_text_is_used(self):
        out = self._run("必要な台数を見積もっておきます。")
        self.assertIsNotNone(out)
        self.assertEqual(out[0], "必要な台数を見積もっておきます。")

    def test_still_dropped_when_local_hears_nothing(self):
        self.assertIsNone(self._run(""))

    def test_still_dropped_when_local_is_stock_too(self):
        self.assertIsNone(self._run("ご視聴ありがとうございました"))


class TestSpeakerJoinWiring(unittest.TestCase):
    def test_worker_passes_speaker_to_endpointer(self):
        from pathlib import Path
        src = (Path(__file__).resolve().parents[1] / "src/tsuyaku/__main__.py").read_text()
        self.assertIn("spk_now = lambda f=fut: _spk_id(f.result(timeout=0.1))", src)
        self.assertIn("spk=spk_now, cut=cut))", src)   # 11/08: 区切った前半の印も渡す
        from tsuyaku.__main__ import _spk_id
        self.assertEqual(_spk_id((3, "S2")), "S2")
        self.assertEqual(_spk_id("S1"), "S1")


class TestStockRescueTooShort(unittest.TestCase):
    """11/07 擬似会議 v2 ja_booth: 1.3秒の区間を リモートサーバー が「ご視聴ありがとうございました」と聞き、
    このMacの耳の雑な「一たや。」が記録に入った。救う文字が短すぎれば入れない。"""

    def test_short_rescue_is_not_used(self):
        self.assertIsNone(TestStockPhraseRescue._run(TestStockPhraseRescue(), "一たや。"))

    def test_long_enough_rescue_is_used(self):
        out = TestStockPhraseRescue._run(TestStockPhraseRescue(), "二台あれば足ります。")
        self.assertEqual(out[0], "二台あれば足ります。")


class TestSpkDifferWiring(unittest.TestCase):
    def test_worker_gives_endpointer_the_guess(self):
        from pathlib import Path
        src = (Path(__file__).resolve().parents[1] / "src/tsuyaku/__main__.py").read_text()
        self.assertIn("spk_differ=((lambda a, sid: self.spk.different_speaker(a, sid))", src)
