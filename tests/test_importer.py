"""音声・動画ファイルの取り込み(W2, 11/07): 書き起こし + 話者 + 記録 + 録音を新しい会議として残す。

ライブと同じ道(VAD + StreamSegmenter → 耳 → 守卫 → Endpointer → 話者の見分け → 記録)を、
音声の時計で実時間より速く回す。ここでは耳・区切り・話者を偽物にして道筋を確かめる。
"""
import json
import queue
import shutil
import subprocess
import tempfile
import threading
import unittest
import wave
from pathlib import Path

import numpy as np

from tsuyaku import importer as I
from tsuyaku import notes as N

SR = 16000


def write_wav(p: Path, x: np.ndarray, sr: int, ch: int = 1):
    with wave.open(str(p), "wb") as w:
        w.setnchannels(ch)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes((np.clip(x, -1, 1) * 32767).astype("<i2").tobytes())


class FakeSeg:
    noise = 0.001

    def flush_current(self):
        return []


class FakeCapture:
    """区間を決め打ちで出す。items: [(lo秒, hi秒)]。音声は渡された波形から切る。"""

    def __init__(self, out_q, spans, x):
        self.out = out_q
        self.spans = list(spans)
        self.x = x
        self.fed = 0
        self.segmenter = FakeSeg()

    def on_chunk(self, ch):
        self.fed += len(ch)
        while self.spans and self.spans[0][1] * SR <= self.fed:
            lo, hi = self.spans.pop(0)
            self.out.put((self.x[int(lo * SR):int(hi * SR)], lo, hi, 0.0))


class FakeSTT:
    """区間の長さ(0.1秒単位)で決めた文を返す。聞き直し(繋いだ音声)も長さで引く。"""

    def __init__(self, table):
        self.table = table
        self.calls = 0

    def transcribe(self, a, **_k):
        self.calls += 1
        key = round(len(a) / SR, 1)
        return self.table.get(key, ("", "", 0.0))


class FakeSpk:
    def __init__(self):
        self.n = 0
        self.aliases = {}

    def assign_indexed(self, a):
        i = self.n
        self.n += 1
        return i, ("S1" if i % 2 == 0 else "S2")

    def canonical(self, s):
        return s

    def refine(self):
        return ["S1" if i % 2 == 0 else "S2" for i in range(self.n)]


def rows(p):
    return [json.loads(x) for x in Path(p).read_text(encoding="utf-8").splitlines()]


class TestDecode(unittest.TestCase):
    def test_decode_resamples_to_16k_mono(self):
        with tempfile.TemporaryDirectory() as d:
            src = Path(d) / "talk.wav"
            t = np.arange(44100 * 2) / 44100
            st = np.stack([0.3 * np.sin(2 * np.pi * 300 * t), 0.3 * np.sin(2 * np.pi * 300 * t)], 1)
            write_wav(src, st.reshape(-1), 44100, ch=2)
            x = I.decode(src, Path(d) / "work.wav")
            self.assertEqual(x.dtype, np.float32)
            self.assertAlmostEqual(len(x) / SR, 2.0, delta=0.05)
            self.assertGreater(float(np.abs(x).max()), 0.2)

    @unittest.skipUnless(shutil.which("avconvert"), "avconvert なし")
    def test_decode_movie_container(self):
        with tempfile.TemporaryDirectory() as d:
            w = Path(d) / "a.wav"
            write_wav(w, 0.3 * np.sin(2 * np.pi * 300 * np.arange(SR * 3) / SR), SR)
            m4a = Path(d) / "a.m4a"
            subprocess.run(["afconvert", "-f", "m4af", "-d", "aac", "-b", "32000", str(w), str(m4a)], check=True)
            mov = Path(d) / "clip.mov"
            subprocess.run(["avconvert", "-s", str(m4a), "-p", "PresetPassthrough", "-o", str(mov)],
                           check=True, capture_output=True)
            x = I.decode(mov, Path(d) / "work.wav")
            # mov の AAC は頭の詰め物(priming 2112 サンプル)がそのまま出ることがある。録音も同じ音声から
            # 作るので行と声の対応はずれない
            self.assertAlmostEqual(len(x) / SR, 3.0, delta=0.2)

    def test_decode_rejects_non_audio(self):
        with tempfile.TemporaryDirectory() as d:
            src = Path(d) / "notes.txt"
            src.write_text("hello")
            with self.assertRaises(I.ImportFailed):
                I.decode(src, Path(d) / "work.wav")


class TestImport(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.TemporaryDirectory()
        self.dir = Path(self.d.name)
        x = (0.1 * np.sin(2 * np.pi * 200 * np.arange(SR * 12) / SR)).astype(np.float32)
        self.src = self.dir / "meeting.wav"
        write_wav(self.src, x, SR)
        # 区間: 1.0-2.5 「今日は、」(続きあり) / 2.8-4.0 「晴れです。」 / 6.0-8.0 英語 / 9.0-9.6 雑音の幻覚(韓国語)
        self.spans = [(1.0, 2.5), (2.8, 4.0), (6.0, 8.0), (9.0, 9.6)]
        self.stt = FakeSTT({1.5: ("今日は、", "ja", 0.9), 1.2: ("晴れです。", "ja", 0.9),
                            3.0: ("今日は、晴れです。", "ja", 0.95),   # 繋いで聞き直した 1.5+0.3(間)+1.2
                            2.0: ("Good morning everyone.", "en", 0.9),
                            0.6: ("그.", "ko", 0.2)})
        self.prog = []

    def tearDown(self):
        self.d.cleanup()

    def run_import(self, **kw):
        return I.run_import(
            self.src, self.dir / "logs", "ja-en", "", stt=self.stt,
            make_capture=lambda q, x: FakeCapture(q, self.spans, x), spk=FakeSpk(),
            prog=lambda st, pct, info=None: self.prog.append((st, pct)), fine=True, **kw)

    def test_makes_meeting_with_audio_offsets_speakers_and_m4a(self):
        r = self.run_import()
        p = self.dir / "logs" / f"{r['id']}.jsonl"
        rs = rows(p)
        meta = rs[0]
        self.assertEqual(meta["source"], "file")
        self.assertEqual(meta["file"], "meeting.wav")
        self.assertEqual(meta["mode"], "notes")
        self.assertAlmostEqual(meta["dur"], 12.0, delta=0.1)
        ut = [x for x in rs if x["type"] == "utter"]
        self.assertEqual([u["orig"] for u in ut], ["今日は、晴れです。", "Good morning everyone."])
        self.assertEqual((ut[0]["a0"], ut[0]["a1"]), (1.0, 4.0))
        self.assertEqual((ut[1]["a0"], ut[1]["a1"]), (6.0, 8.0))
        self.assertAlmostEqual(ut[1]["ts"] - meta["ts0"], 8.0, delta=0.01)
        self.assertEqual(ut[0]["segs"], [0, 1])
        self.assertTrue(any(x["type"] == "refine" for x in rs))
        m4a = self.dir / "logs" / meta["audio"]
        self.assertTrue(m4a.exists())
        m = N.load_meeting(p)
        self.assertEqual(len(m["lines"]), 2)
        self.assertEqual(r["lines"], 2)
        self.assertEqual(self.prog[-1][0], "done")
        pcts = [pc for _s, pc in self.prog]
        self.assertEqual(pcts, sorted(pcts))
        self.assertFalse(list((self.dir / "logs").glob("*.work.wav")))   # 作業ファイルは残さない

    def test_cancel_removes_partial_meeting(self):
        ev = threading.Event()
        ev.set()
        r = self.run_import(cancel=ev)
        self.assertIsNone(r)
        self.assertEqual(list((self.dir / "logs").glob("session-*")), [])

    def test_bad_file_reports_error(self):
        bad = self.dir / "x.mp3"
        bad.write_bytes(b"nope")
        with self.assertRaises(I.ImportFailed):
            I.run_import(bad, self.dir / "logs", "ja-en", "", stt=self.stt,
                         make_capture=lambda q, x: FakeCapture(q, [], x), spk=None)
        self.assertEqual(list((self.dir / "logs").glob("session-*")), [])

    def test_supported_extensions(self):
        for ext in ("m4a", "mp3", "wav", "mp4", "mov", "aac", "caf", "aiff", "flac", "m4v"):
            self.assertTrue(I.supported(f"a.{ext}"), ext)
            self.assertTrue(I.supported(f"A.{ext.upper()}"), ext)
        self.assertFalse(I.supported("a.txt"))


if __name__ == "__main__":
    unittest.main()
