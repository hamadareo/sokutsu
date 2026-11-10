"""読み上げ中も聞く(エコーを消す部品): 約束のバイト列・512 サンプルずつ・play が終わりを待つ・
cancel が「止める」を送る・部品が無い/許可が無いときは使えない扱い。本物の録音・再生はしない
(偽の部品 fake_aecio.py。本物の部品は機器を使わない --fake のときだけ動かす)。"""
import os
import struct
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path

import numpy as np

from tsuyaku.aecio import (CHUNK, FLUSH, HELPER, Chunker, EchoCancelIO, encode_clip)

FAKE = str(Path(__file__).with_name("fake_aecio.py"))


def fake(*args):
    return [sys.executable, FAKE, *args]


def wait_for(pred, timeout=5.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if pred():
            return True
        time.sleep(0.01)
    return pred()


class Sink:
    def __init__(self):
        self.chunks = []
        self.lock = threading.Lock()

    def __call__(self, x):
        with self.lock:
            self.chunks.append(x)


class TestProtocol(unittest.TestCase):
    def test_encode_clip(self):
        b = encode_clip(np.array([0.5, -0.25, 2.0], dtype=np.float64))
        self.assertEqual(b[:4], struct.pack("<I", 3))
        self.assertEqual(len(b), 4 + 3 * 4)
        np.testing.assert_array_equal(np.frombuffer(b[4:], dtype="<f4"), [0.5, -0.25, 1.0])  # 切り詰め

    def test_flush_is_zero_count(self):
        self.assertEqual(FLUSH, b"\0\0\0\0")

    def test_empty_clip_refused(self):
        with self.assertRaises(ValueError):
            encode_clip(np.zeros(0, dtype=np.float32))

    def test_chunker_splits_odd_bytes(self):
        ch = Chunker(512)
        data = np.arange(1300, dtype="<f4").tobytes()
        got = []
        for i in range(0, len(data), 333):     # float の途中で切れる
            got += ch.feed(data[i:i + 333])
        self.assertEqual(len(got), 2)
        self.assertTrue(all(len(c) == 512 and c.dtype == np.float32 for c in got))
        np.testing.assert_array_equal(np.concatenate(got), np.arange(1024, dtype=np.float32))
        got = ch.feed(np.arange(1300, 1324, dtype="<f4").tobytes())  # 残り 276 + 24 = 300 < 512
        self.assertEqual(got, [])


class TestEchoCancelIO(unittest.TestCase):
    def setUp(self):
        fd, self.log = tempfile.mkstemp(prefix="fake_aecio_", suffix=".log")
        os.close(fd)
        self.io = None

    def tearDown(self):
        if self.io is not None:
            self.io.stop()
        os.unlink(self.log)

    def start(self, *args, **kw):
        self.sink = Sink()
        self.io = EchoCancelIO(self.sink, cmd=fake("--log", self.log, *args), **kw)
        return self.io.start()

    def logged(self):
        with open(self.log) as f:
            return f.read().split("\n")[:-1]

    def test_mic_chunks_are_512(self):
        self.assertTrue(self.start("--mic", "1600"))
        self.assertTrue(self.io.available)
        self.assertTrue(wait_for(lambda: len(self.sink.chunks) == 3))
        time.sleep(0.1)
        self.assertEqual(len(self.sink.chunks), 3)          # 1600 = 512 * 3 + 64(持ち越し)
        self.assertTrue(all(len(c) == CHUNK for c in self.sink.chunks))
        np.testing.assert_array_equal(np.concatenate(self.sink.chunks),
                                      np.arange(1536, dtype=np.float32))
        self.assertEqual(self.io.info.get("rate"), "16000")

    def test_play_waits_for_done(self):
        self.assertTrue(self.start("--play-s", "0.3"))
        started = []
        x = np.linspace(-0.5, 0.5, 4000).astype(np.float32)
        t0 = time.time()
        self.assertTrue(self.io.play(x, on_start=lambda: started.append(time.time() - t0)))
        took = time.time() - t0
        self.assertGreaterEqual(took, 0.28)                  # 「鳴らし終えた」まで返らない
        self.assertFalse(self.io.last_cancelled)
        self.assertEqual(len(started), 1)
        self.assertEqual(self.logged(), ["CLIP 4000 -0.5000 0.5000"])
        # 2件目は id 2(部品と同じ数え方)
        self.assertTrue(self.io.play(np.full(10, 0.1, dtype=np.float32)))
        self.assertEqual(self.logged()[-1], "CLIP 10 0.1000 0.1000")

    def test_cancel_sends_flush_and_play_returns(self):
        self.assertTrue(self.start("--play-s", "5"))
        res = {}

        def run():
            res["ok"] = self.io.play(np.zeros(16000, dtype=np.float32))
        t = threading.Thread(target=run)
        t0 = time.time()
        t.start()
        self.assertTrue(wait_for(lambda: self.logged() == ["CLIP 16000 0.0000 0.0000"]))
        self.assertTrue(self.io.cancel())
        t.join(3.0)
        self.assertFalse(t.is_alive())
        self.assertLess(time.time() - t0, 3.0)
        self.assertTrue(res["ok"])
        self.assertTrue(self.io.last_cancelled)
        self.assertTrue(wait_for(lambda: self.logged()[-1:] == ["FLUSH"]))

    def test_missing_binary_unavailable(self):
        statuses = []
        io = EchoCancelIO(lambda x: None, cmd=["/nonexistent/sokutsu-aecio"],
                          on_status=lambda k, m: statuses.append(k))
        self.assertFalse(io.start())
        self.assertFalse(io.available)
        self.assertFalse(io.play(np.zeros(100, dtype=np.float32)))
        self.assertFalse(io.cancel())
        self.assertEqual(statuses, ["failed"])
        self.assertIn("build_aecio.sh", io.error)

    def test_permission_denied_unavailable(self):
        self.assertFalse(self.start("--mode", "deny"))
        self.assertFalse(self.io.available)
        self.assertEqual(self.io.returncode, 3)
        self.assertIn("マイクの許可", self.io.error)

    def test_unsupported_unavailable(self):
        self.assertFalse(self.start("--mode", "unsup"))
        self.assertEqual(self.io.returncode, 4)
        self.assertIn("VoiceProcessingIO", self.io.error)

    def test_no_ready_times_out(self):
        self.assertFalse(self.start("--mode", "silent", start_timeout=0.3))
        self.assertFalse(self.io.available)
        self.assertEqual(self.io.state, "failed")

    def test_wrong_rate_unavailable(self):
        self.assertFalse(self.start("--mode", "rate"))
        self.assertFalse(self.io.available)

    def test_helper_dies_during_play(self):
        statuses = []
        self.assertTrue(self.start("--mode", "dieplay", on_status=lambda k, m: statuses.append(k)))
        t0 = time.time()
        self.assertFalse(self.io.play(np.zeros(1600, dtype=np.float32)))   # 待ち続けない
        self.assertLess(time.time() - t0, 3.0)
        self.assertTrue(wait_for(lambda: not self.io.available))
        self.assertTrue(wait_for(lambda: statuses[-1:] == ["failed"]))
        self.assertFalse(self.io.play(np.zeros(10, dtype=np.float32)))      # → afplay の道へ


@unittest.skipUnless(HELPER.exists(), "bin/sokutsu-aecio が無い(scripts/build_aecio.sh)")
class TestRealHelperFake(unittest.TestCase):
    """本物の部品を --fake(音声機器を使わない偽の時計)で動かし、約束どおりか確かめる。"""

    def test_play_echo_and_flush(self):
        sink = Sink()
        io = EchoCancelIO(sink, cmd=[str(HELPER), "--fake", "--tail-ms", "10"])
        try:
            self.assertTrue(io.start())
            self.assertEqual(io.info.get("fake"), "1")
            t0 = time.time()
            self.assertTrue(io.play(np.full(3200, 0.8, dtype=np.float32)))   # 0.2 秒
            self.assertGreaterEqual(time.time() - t0, 0.18)
            self.assertFalse(io.last_cancelled)
            res = {}
            t = threading.Thread(target=lambda: res.setdefault("ok", io.play(
                np.full(16000 * 10, 0.4, dtype=np.float32))))
            t.start()
            time.sleep(0.3)
            self.assertTrue(io.cancel())
            t.join(3.0)
            self.assertFalse(t.is_alive())
            self.assertTrue(res["ok"])
            self.assertTrue(io.last_cancelled)
            time.sleep(0.1)
            with sink.lock:
                mic = np.concatenate(sink.chunks)
            self.assertTrue(all(len(c) == CHUNK for c in sink.chunks))
            # 偽の部品のマイク = 鳴らした音 × 0.5: 1件目は全部、2件目は止めるまでの分だけ
            self.assertEqual(int(np.sum(np.abs(mic - 0.4) < 1e-6)), 3200)
            n2 = int(np.sum(np.abs(mic - 0.2) < 1e-6))
            self.assertGreater(n2, 1600)
            self.assertLess(n2, 16000 * 2)
        finally:
            io.stop()
        self.assertEqual(io.state, "stopped")


if __name__ == "__main__":
    unittest.main()
