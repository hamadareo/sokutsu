"""会議の録音(W2, 11/07「他の会議メモアプリにある機能は全て」: 録音の保存と、行を押すとその声を再生)。

見つけた問題: 記録は文字だけで、声は残っていなかった(聞き直せない・誤認識を確かめられない)。
変更: recorder.Recorder — マイクの全チャンク(読み上げ中に止めている間も)を CAF(LPCM, data の長さ -1)に
書き続け、終わったら AAC(m4a, 32kbps)にする。CAF は途中で落ちてもそこまでが正しい音声のまま読める。
記録の行の a0/a1(録音の先頭からの秒)を出すため、マイクの通算秒(読み上げ中は進まない)→ 録音の秒 を対応づける。
"""
import struct
import subprocess
import tempfile
import time
import unittest
from pathlib import Path

import numpy as np

from tsuyaku import recorder as R

SR = 16000


class Clock:
    def __init__(self, t=1000.0):
        self.t = t

    def __call__(self):
        return self.t


def caf_frames(p: Path) -> int:
    b = p.read_bytes()
    assert b[:4] == b"caff"
    i = b.index(b"data")
    return (len(b) - (i + 12 + 4)) // 2


def feed(rec, clock, n_chunks, paused=False, value=0.1):
    for _ in range(n_chunks):
        clock.t += 512 / SR
        rec.tap(np.full(512, value, np.float32), paused)


class TestRecorder(unittest.TestCase):
    def test_caf_header_is_valid_and_frames_written(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "session-x.audio.caf"
            clk = Clock()
            rec = R.Recorder(p, ts0=clk.t, clock=clk)
            feed(rec, clk, 10)
            rec.close()
            self.assertEqual(caf_frames(p), 10 * 512)
            out = subprocess.run(["afinfo", str(p)], capture_output=True, text=True).stdout
            self.assertIn("16000 Hz", out)
            self.assertIn(f"{10 * 512} valid frames", out)

    def test_survives_crash_without_close(self):
        # 落ちた(close を呼ばない)ときも、書いた所までは読める音声のまま
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "session-x.audio.caf"
            clk = Clock()
            rec = R.Recorder(p, ts0=clk.t, clock=clk, flush_s=0.0)
            feed(rec, clk, 20)
            deadline = time.time() + 3
            while caf_frames(p) < 20 * 512 and time.time() < deadline:
                time.sleep(0.02)
            self.assertEqual(caf_frames(p), 20 * 512)
            out = subprocess.run(["afinfo", str(p)], capture_output=True, text=True).stdout
            self.assertIn(f"{20 * 512} valid frames", out)
            rec.close()

    def test_pads_silence_before_first_chunk(self):
        # 記録を開いた時刻(ts0)からマイクの最初のチャンクまでの間は無音で埋める: 録音の秒 = 壁時計 - ts0
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "a.caf"
            clk = Clock(1000.0)
            rec = R.Recorder(p, ts0=999.5, clock=clk)
            feed(rec, clk, 4)
            rec.close()
            n = caf_frames(p)
            self.assertEqual(n, int(round((clk.t - 999.5) * SR)))
            pcm = np.frombuffer(p.read_bytes()[-n * 2:], "<i2")
            self.assertTrue((pcm[:7000] == 0).all())
            self.assertTrue((pcm[-512:] != 0).all())

    def test_mic_seconds_map_through_pauses(self):
        # 読み上げ中(paused)はマイクの時計が進まないが、録音には入っている
        with tempfile.TemporaryDirectory() as d:
            clk = Clock()
            rec = R.Recorder(Path(d) / "a.caf", ts0=clk.t, clock=clk)
            feed(rec, clk, 10)                 # マイク 0〜10ch, 録音 0〜10ch
            feed(rec, clk, 5, paused=True)     # 録音 10〜15ch(読み上げ)
            feed(rec, clk, 10)                 # マイク 10〜20ch, 録音 15〜25ch
            rec.close()
            ch = 512 / SR
            self.assertAlmostEqual(rec.audio_offset(3 * ch), 3 * ch, places=3)
            self.assertAlmostEqual(rec.audio_offset(12 * ch), 17 * ch, places=3)
            self.assertAlmostEqual(rec.seconds, 25 * ch, places=3)

    def test_gap_in_input_is_filled_to_keep_wall_clock(self):
        # 入力が途切れた(スリープ・装置の切替)ら無音で埋め、壁時計と録音の秒を揃えたままにする
        with tempfile.TemporaryDirectory() as d:
            clk = Clock()
            rec = R.Recorder(Path(d) / "a.caf", ts0=clk.t, clock=clk)
            feed(rec, clk, 10)
            clk.t += 3.0
            feed(rec, clk, 10)
            rec.close()
            self.assertAlmostEqual(rec.seconds, clk.t - rec.ts0, delta=0.05)
            ch = 512 / SR
            self.assertAlmostEqual(rec.audio_offset(12 * ch), 12 * ch + 3.0, delta=0.05)

    def test_tap_never_raises_after_close(self):
        with tempfile.TemporaryDirectory() as d:
            clk = Clock()
            rec = R.Recorder(Path(d) / "a.caf", ts0=clk.t, clock=clk)
            rec.close()
            rec.tap(np.zeros(512, np.float32), False)   # 例外を出さない(会話を止めない)

    def test_write_failure_does_not_break_conversation(self):
        with tempfile.TemporaryDirectory() as d:
            clk = Clock()
            rec = R.Recorder(Path(d) / "a.caf", ts0=clk.t, clock=clk)
            rec._f.close()                      # ディスクが外れた等で書けなくなった
            feed(rec, clk, 30)
            rec.close()
            self.assertTrue(rec.failed)

    def test_finalize_to_m4a_and_remove_caf(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "session-x.audio.caf"
            clk = Clock()
            rec = R.Recorder(p, ts0=clk.t, clock=clk)
            t = np.arange(SR * 3) / SR
            x = (0.2 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
            for i in range(0, len(x), 512):
                clk.t += 512 / SR
                rec.tap(x[i:i + 512], False)
            rec.close()
            m4a = R.finalize(p)
            self.assertEqual(m4a, Path(d) / "session-x.audio.m4a")
            self.assertTrue(m4a.exists())
            self.assertFalse(p.exists())
            self.assertAlmostEqual(R.duration(m4a), 3.0, delta=0.1)
            self.assertLess(m4a.stat().st_size, 3 * 32000 / 8 * 1.6)

    def test_finalize_keeps_caf_when_conversion_fails(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "session-x.audio.caf"
            p.write_bytes(b"not audio")
            self.assertIsNone(R.finalize(p))
            self.assertTrue(p.exists())       # 声は消さない

    def test_recover_converts_stale_caf_only(self):
        with tempfile.TemporaryDirectory() as d:
            clk = Clock()
            old = Path(d) / "session-20261101-100000.audio.caf"
            rec = R.Recorder(old, ts0=clk.t, clock=clk)
            feed(rec, clk, 40)
            rec.close()
            st = time.time() - 600
            import os
            os.utime(old, (st, st))
            fresh = Path(d) / "session-20261107-100000.audio.caf"
            rec2 = R.Recorder(fresh, ts0=clk.t, clock=clk)
            feed(rec2, clk, 40)
            rec2.close()
            done = R.recover(d, min_age_s=60)
            self.assertEqual([p.name for p in done], ["session-20261101-100000.audio.m4a"])
            self.assertTrue(fresh.exists())

    def test_audio_paths(self):
        p = Path("/x/logs/session-20261107-101010.jsonl")
        self.assertEqual(R.caf_path(p).name, "session-20261107-101010.audio.caf")
        self.assertEqual(R.m4a_path(p).name, "session-20261107-101010.audio.m4a")


class TestRecordSetting(unittest.TestCase):
    def test_default_by_mode(self):
        self.assertFalse(R.should_record("auto", "interpret"))
        self.assertTrue(R.should_record("auto", "notes"))
        self.assertTrue(R.should_record("auto", "both"))
        self.assertTrue(R.should_record("on", "interpret"))
        self.assertFalse(R.should_record("off", "notes"))
        self.assertTrue(R.should_record(None, "notes"))


if __name__ == "__main__":
    unittest.main()
