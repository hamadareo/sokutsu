"""話し続けても置いていかれない(11/06 18:08 実会話「スピードについていけていない」)。

見つけた問題: 読み上げの間はマイクを止めていて、その間に話した声は捨てていた。
英語の訳の読み上げは1回2〜8秒(122字で約8秒)で、3分半の会話のうち約50秒は
マイクが聞いていなかった。さらに読み上げが始まった瞬間に話し途中だった区間も
捨てていた。読み上げの順番待ちで、7字の訳が話し終わりから11.15秒後に鳴った。
変更: ①「字幕だけ」(声を出さない・マイクを止めない) ②止める直前まで話していた
区間は捨てずに渡す ③順番が来た時にはもう古い(8秒超)読み上げは飛ばす(字幕は出ている)。
"""
import queue
import threading
import time
import unittest
from pathlib import Path

import numpy as np

from tsuyaku.__main__ import Bridge
from tsuyaku.audio import StreamSegmenter

SRC = (Path(__file__).resolve().parents[1] / "src/tsuyaku/audio.py").read_text()


class TestFlushOnPause(unittest.TestCase):
    def test_flush_current_returns_the_unfinished_segment(self):
        sg = StreamSegmenter(16000)
        rng = np.random.default_rng(0)
        for _ in range(40):   # 約1.3秒 話している最中
            out = sg.feed((rng.standard_normal(512) * 0.2).astype(np.float32), True)
            self.assertEqual(out, [])
        got = sg.flush_current()
        self.assertEqual(len(got), 1)
        self.assertGreater(len(got[0][0]), 16000)
        self.assertIsNone(sg.speaking_since())
        self.assertEqual(sg.flush_current(), [])   # 2度は出さない

    def test_flush_when_silent_gives_nothing(self):
        sg = StreamSegmenter(16000)
        sg.feed(np.zeros(512, np.float32), False)
        self.assertEqual(sg.flush_current(), [])

    def test_capture_flushes_instead_of_dropping_when_paused(self):
        i = SRC.index("def on_chunk(")   # cb(indata) は on_chunk を呼ぶだけ(11/07 監査で例外の守りを足した)
        seg = SRC[i:i + 600]
        self.assertIn("self.segmenter.flush_current()", seg)


class FakeTTS:
    def __init__(self):
        self.said = []
        self.on_start = None

    def say(self, text, lang, speed=None):
        self.said.append(text)
        return True

    def cancel(self):
        pass


class FakeCap:
    def __init__(self):
        self.paused = threading.Event()
        self.pauses = 0

    def resume_after_tts(self, tail):
        self.pauses += 1
        self.paused.clear()


def run_tts(items, voice_on=True):
    b = Bridge()
    b.s.voice_on = voice_on
    b.tts = FakeTTS()
    b.capture = FakeCap()
    b.running.set()
    for it in items:
        b.tts_q.put(it)
    th = threading.Thread(target=b._tts_worker, daemon=True)
    th.start()
    t0 = time.time()
    while not b.tts_q.empty() and time.time() - t0 < 3:
        time.sleep(0.02)
    time.sleep(0.1)
    b.running.clear()
    th.join(2)
    return b


class TestTtsWorker(unittest.TestCase):
    def test_captions_only_never_pauses_mic(self):
        b = run_tts([("Hello.", "en", 1, time.time())], voice_on=False)
        self.assertEqual(b.tts.said, [])
        self.assertEqual(b.capture.pauses, 0)

    def test_stale_voice_is_skipped_fresh_is_read(self):
        now = time.time()
        b = run_tts([("old", "ja", 1, now - 11.0), ("new", "ja", 2, now - 1.0)])
        self.assertEqual(b.tts.said, ["new"])

    def test_voice_on_reads(self):
        b = run_tts([("Hello.", "en", 1, time.time())])
        self.assertEqual(b.tts.said, ["Hello."])
        self.assertEqual(b.capture.pauses, 1)


class TestSwitch(unittest.TestCase):
    def test_setting_and_ui(self):
        from tsuyaku.config import Settings
        self.assertTrue(Settings().voice_on)
        self.assertAlmostEqual(Settings().voice_max_lag_s, 8.0)
        html = (Path(__file__).resolve().parents[1] / "src/tsuyaku/webui/index.html").read_text()
        self.assertIn('id="voiceOn"', html)
        self.assertIn("set_voice_on(", html)
        for k in ("字幕だけ", "声も出す", "字幕だけにしました — 続けて話せます", "読み上げを戻しました"):
            self.assertGreaterEqual(html.count(f'"{k}":'), 3, k)   # en/zh/ko の訳がある

    def test_set_voice_on_saves(self):
        from unittest.mock import patch
        b = Bridge()
        with patch("tsuyaku.__main__.save_config") as sc:
            self.assertEqual(b.set_voice_on(False), "ok")
        self.assertFalse(b.s.voice_on)
        sc.assert_called_once()


if __name__ == "__main__":
    unittest.main()
