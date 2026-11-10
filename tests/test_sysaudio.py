"""この Mac の音(オンライン会議)をマイクに足す: 部品の読み取り・周波数変換・足し合わせ・
落ちたときにマイクだけで続けること・設定の保存。本物の録音はしない(許可のダイアログを出さない)。"""
import json
import re
import os
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from tsuyaku import config as cfgmod
from tsuyaku.log import SessionLog
from tsuyaku.sysaudio import (HELPER, MicPlusSystem, Mixer, Resampler, SystemAudio,
                              soft_clip)

FAKE = str(Path(__file__).with_name("fake_sysaudio.py"))


def fake(*args):
    return [sys.executable, FAKE, *args]


def peak_hz(x, sr):
    spec = np.abs(np.fft.rfft(x * np.hanning(len(x))))
    return np.argmax(spec) * sr / len(x)


def wait_for(pred, timeout=5.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if pred():
            return True
        time.sleep(0.01)
    return pred()


class TestResampler(unittest.TestCase):
    def test_48k_to_16k_keeps_tone(self):
        sr = 48000
        x = (0.5 * np.sin(2 * np.pi * 440 * np.arange(sr) / sr)).astype(np.float32)
        y = Resampler(sr, 16000)(x)
        self.assertAlmostEqual(len(y), 16000, delta=40)
        mid = y[2000:14000]
        self.assertAlmostEqual(peak_hz(mid, 16000), 440, delta=3)
        self.assertAlmostEqual(float(np.max(np.abs(mid))), 0.5, delta=0.02)

    def test_chunked_equals_whole(self):
        sr = 44100
        rng = np.random.default_rng(0)
        x = rng.standard_normal(sr // 2).astype(np.float32) * 0.1
        whole = Resampler(sr)(x)
        r = Resampler(sr)
        parts, i = [], 0
        for n in [1, 7, 480, 1024, 333, 4096, 9000, 100000]:
            parts.append(r(x[i:i + n]))
            i += n
        chunked = np.concatenate(parts)
        self.assertEqual(len(chunked), len(whole))
        self.assertLess(float(np.max(np.abs(chunked - whole))), 1e-5)

    def test_high_tone_is_filtered(self):
        sr = 48000   # 12 kHz は 16 kHz では表せない: 折り返して聞こえてはいけない
        x = (0.5 * np.sin(2 * np.pi * 12000 * np.arange(sr) / sr)).astype(np.float32)
        y = Resampler(sr, 16000)(x)[1000:-1000]
        self.assertLess(float(np.sqrt(np.mean(y ** 2))), 0.01)

    def test_same_rate_passthrough(self):
        x = np.arange(10, dtype=np.float32)
        np.testing.assert_array_equal(Resampler(16000, 16000)(x), x)


class TestMixer(unittest.TestCase):
    def test_mic_only_until_system_arrives(self):
        m = Mixer()
        mic = np.full(512, 0.1, dtype=np.float32)
        np.testing.assert_allclose(m.mix(mic), mic)
        self.assertFalse(m.primed)

    def test_sums_in_order(self):
        m = Mixer()
        m.push(np.arange(1024, dtype=np.float32) * 1e-4)
        out1 = m.mix(np.zeros(512, dtype=np.float32))
        out2 = m.mix(np.zeros(512, dtype=np.float32))
        np.testing.assert_allclose(np.concatenate([out1, out2]),
                                   np.arange(1024, dtype=np.float32) * 1e-4, atol=1e-7)

    def test_drops_when_too_late(self):
        sr = 16000
        m = Mixer(sr, lag_s=0.06, max_lag_s=0.25)
        m.push(np.zeros(sr, dtype=np.float32))            # 1秒溜まった(遅れ)
        m.push(np.full(512, 0.2, dtype=np.float32))        # 最新
        m.mix(np.zeros(512, dtype=np.float32))
        self.assertGreater(m.dropped, 0)
        self.assertLessEqual(m._n, int(sr * 0.06))         # 遅れは lag_s まで詰まる
        later = [m.mix(np.zeros(512, dtype=np.float32)) for _ in range(3)]
        self.assertTrue(any(np.allclose(c[-100:], 0.2) for c in later))   # 最新の音は残る

    def test_underrun_pads_front(self):
        m = Mixer()
        m.push(np.full(600, 0.1, dtype=np.float32))
        m.mix(np.zeros(512, dtype=np.float32))
        out = m.mix(np.zeros(512, dtype=np.float32))       # 88 しか残っていない
        self.assertEqual(m.underruns, 1)
        np.testing.assert_allclose(out[:424], 0)
        np.testing.assert_allclose(out[424:], 0.1, atol=1e-6)

    def test_soft_clip(self):
        x = np.array([0.0, 0.5, -0.79, 0.95, 1.5, -3.0], dtype=np.float32)
        y = soft_clip(x)
        np.testing.assert_allclose(y[:3], x[:3])
        self.assertTrue(np.all(np.abs(y) <= 1.0))
        self.assertTrue(0.8 < y[3] < y[4] < 1.0)
        m = Mixer()
        m.push(np.full(512, 0.9, dtype=np.float32))
        self.assertLess(float(np.max(m.mix(np.full(512, 0.9, dtype=np.float32)))), 1.0)

    def test_threaded_push(self):
        m = Mixer()
        stop = threading.Event()

        def producer():
            while not stop.is_set():
                m.push(np.full(160, 0.01, dtype=np.float32))
                time.sleep(0.001)

        t = threading.Thread(target=producer)
        t.start()
        try:
            for _ in range(200):
                out = m.mix(np.zeros(512, dtype=np.float32))
                self.assertEqual(len(out), 512)
        finally:
            stop.set()
            t.join()


class TestSystemAudioReader(unittest.TestCase):
    def test_reads_fake_helper_and_resamples(self):
        got = []
        st = []
        sa = SystemAudio(got.append, cmd=fake("--mode", "tone", "--seconds", "1"),
                         on_status=lambda k, m: st.append(k), max_restarts=0)
        sa.start()
        self.assertTrue(wait_for(lambda: sa.state == "failed" or sa.samples >= 15900))
        sa.stop()
        y = np.concatenate(got)
        self.assertEqual(sa.rate, 48000)
        self.assertAlmostEqual(len(y), 16000, delta=60)
        self.assertAlmostEqual(peak_hz(y[1000:15000], 16000), 440, delta=3)
        self.assertIn("running", st)

    def test_audio_before_ready_line(self):
        got = []
        sa = SystemAudio(got.append, cmd=fake("--mode", "early", "--seconds", "0.5"),
                         max_restarts=0)
        sa.start()
        self.assertTrue(wait_for(lambda: sa.samples >= 7900))
        sa.stop()
        self.assertAlmostEqual(peak_hz(np.concatenate(got)[500:7500], 16000), 440, delta=4)

    def test_permission_denied_fails_without_restart(self):
        st = []
        sa = SystemAudio(lambda y: None, cmd=fake("--mode", "deny"),
                         on_status=lambda k, m: st.append((k, m)))
        sa.start()
        self.assertTrue(wait_for(lambda: sa.state == "failed"))
        sa.stop()
        self.assertEqual([k for k, _ in st], ["failed"])
        self.assertIn("許可", st[0][1])
        self.assertIn("システムオーディオ録音", st[0][1])

    def test_missing_helper(self):
        st = []
        sa = SystemAudio(lambda y: None, cmd=["/nonexistent/sokutsu-sysaudio"],
                         on_status=lambda k, m: st.append((k, m)))
        sa.start()
        self.assertTrue(wait_for(lambda: sa.state == "failed"))
        self.assertIn("部品がありません", st[-1][1])

    def test_crash_restarts_then_gives_up(self):
        st = []
        sa = SystemAudio(lambda y: None, cmd=fake("--mode", "crash"), max_restarts=2,
                         backoff_s=0.01, on_status=lambda k, m: st.append(k))
        sa.start()
        self.assertTrue(wait_for(lambda: sa.state == "failed", 10))
        self.assertEqual(st.count("restarting"), 2)
        self.assertEqual(st[-1], "failed")
        self.assertIn("何度も止まりました", sa.error)

    def test_stop_closes_stdin(self):
        sa = SystemAudio(lambda y: None, cmd=fake("--mode", "hold"))
        sa.start()
        self.assertTrue(wait_for(lambda: sa.state == "running"))
        p = sa._proc
        t0 = time.time()
        sa.stop()
        self.assertLess(time.time() - t0, 2.5)
        self.assertIsNotNone(p.poll())
        self.assertEqual(sa.state, "stopped")


class TestMicPlusSystem(unittest.TestCase):
    def test_fallback_is_mic_only(self):
        st = []
        mp = MicPlusSystem(cmd=fake("--mode", "deny"), on_status=lambda k, m: st.append(k))
        mp.start()
        self.assertTrue(wait_for(lambda: mp.state == "failed"))
        mic = np.full(512, 0.05, dtype=np.float32)
        np.testing.assert_allclose(mp.mix(mic), mic)
        mp.stop()
        self.assertEqual(st, ["failed"])

    def test_mixes_system_into_mic(self):
        mp = MicPlusSystem(cmd=fake("--mode", "tonehold", "--seconds", "1"), max_restarts=0)
        mp.start()
        self.assertTrue(wait_for(lambda: mp.sys.samples >= 15900))
        mic = np.zeros(512, dtype=np.float32)
        y = mp.mix(mic)     # 1秒溜まっている = 遅れ: 直近 0.06 秒まで詰めてから足す
        mp.stop()
        self.assertGreater(float(np.sqrt(np.mean(y ** 2))), 0.2)   # 0.5 の正弦 ≒ RMS 0.35
        self.assertGreater(mp.mixer.dropped, 14000)


class TestSpeechCaptureHook(unittest.TestCase):
    def test_mixer_used_by_input_callback(self):
        from tsuyaku import audio

        class Probe:
            def __init__(self):
                self.calls = 0
                self.stopped = False

            def mix(self, mono):
                self.calls += 1
                return mono + 1.0

            def stop(self):
                self.stopped = True

        cap = audio.SpeechCapture.__new__(audio.SpeechCapture)
        cap.sr = 16000
        cap._stop = threading.Event()
        cap._thread = None
        cap.mix_in = Probe()
        seen = []
        cap.on_chunk = lambda m: (seen.append(m), cap._stop.set())

        class FakeStream:
            def __init__(self, callback=None, **kw):
                self.cb = callback

            def __enter__(self):
                self.cb(np.zeros((512, 1), dtype=np.float32), 512, None, None)
                return self

            def __exit__(self, *a):
                return False

        with mock.patch.object(audio.sd, "InputStream", FakeStream):
            cap._run()
        self.assertEqual(cap.mix_in.calls, 1)
        np.testing.assert_allclose(seen[0], 1.0)
        cap.stop()
        self.assertTrue(cap.mix_in.stopped)


class TestStopWithCapture(unittest.TestCase):
    def test_capture_stop_ends_helper_process(self):
        """「会議を終える」(end_meeting)は capture.stop() を呼ぶ: 部品のプロセスも終わること。"""
        from tsuyaku import audio
        cap = audio.SpeechCapture.__new__(audio.SpeechCapture)
        cap._stop = threading.Event()
        cap._thread = None
        cap.mix_in = MicPlusSystem(cmd=fake("--mode", "hold"))
        cap.mix_in.start()
        self.assertTrue(wait_for(lambda: cap.mix_in.state == "running"))
        p = cap.mix_in.sys._proc
        cap.stop()
        self.assertIsNotNone(p.poll())
        self.assertEqual(cap.mix_in.state, "stopped")


class TestConfigAndLog(unittest.TestCase):
    def test_audio_source_persisted(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            with mock.patch.object(cfgmod, "ROOT", root):
                s = cfgmod.Settings()
                self.assertEqual(s.audio_source, "mic")
                s.audio_source = "mic+system"
                cfgmod.save_config(s)
                data = json.loads((root / "config.json").read_text())
                self.assertEqual(data["audio_source"], "mic+system")
                self.assertEqual(cfgmod.load_settings().audio_source, "mic+system")
                with mock.patch.dict(os.environ, {"TSUYAKU_AUDIO_SOURCE": "mic"}):
                    self.assertEqual(cfgmod.load_settings().audio_source, "mic")
                (root / "config.json").write_text(json.dumps({"audio_source": "bogus"}))
                self.assertEqual(cfgmod.load_settings().audio_source, "mic")

    def test_log_meta_has_source(self):
        from tsuyaku import __main__ as app
        b = app.Bridge.__new__(app.Bridge)
        b.s = cfgmod.Settings()
        b.s.audio_source = "mic+system"
        with tempfile.TemporaryDirectory() as d:
            lg = SessionLog(d, "ja-en", "", "both", extra=b._meta_extra())
            lg.close()
            meta = json.loads(Path(lg.path_jsonl).read_text().splitlines()[0])
            self.assertEqual(meta["type"], "meta")
            self.assertEqual(meta["source"], "mic+system")
            self.assertNotIn("audio", meta)   # "audio" は録音ファイル名(W2)の約束


class TestBridgeStart(unittest.TestCase):
    def _bridge(self, saved="mic"):
        from tsuyaku import __main__ as app
        b = app.Bridge.__new__(app.Bridge)
        b.s = cfgmod.Settings()
        b.s.audio_source = saved
        return app, b

    def test_start_saves_source(self):
        app, b = self._bridge("mic")
        with mock.patch.object(app, "save_config") as sv, \
                mock.patch.object(app.threading, "Thread") as th:
            b.start("ja-en", "", "interpret", "mic+system")
        self.assertEqual(b.s.audio_source, "mic+system")
        sv.assert_called_once()
        th.return_value.start.assert_called_once()

    def test_old_call_keeps_saved_source(self):
        app, b = self._bridge("mic+system")
        with mock.patch.object(app, "save_config"), mock.patch.object(app.threading, "Thread"):
            b.start("ja-en", "")
            b.start("ja-en", "", "interpret", "bogus")
        self.assertEqual(b.s.audio_source, "mic+system")

    def test_attach_only_for_system(self):
        app, b = self._bridge("mic")
        cap = mock.Mock()
        cap.mix_in = None
        b._attach_system_audio(cap)
        self.assertIsNone(cap.mix_in)
        b.s.audio_source = "mic+system"
        b.ui = __import__("queue").Queue()
        b.log = None
        with mock.patch("tsuyaku.sysaudio.MicPlusSystem") as mp:
            b._attach_system_audio(cap)
        self.assertIs(cap.mix_in, mp.return_value)
        mp.return_value.start.assert_called_once()
        on_status = mp.call_args.kwargs["on_status"]
        with tempfile.TemporaryDirectory() as d:
            b.log = SessionLog(d, "ja-en", "", "both", extra=b._meta_extra())
            on_status("running", "48000 Hz")
            on_status("failed", "許可がありません")
            b.log.close()
            rows = [json.loads(x) for x in Path(b.log.path_jsonl).read_text().splitlines()]
        st = [r for r in rows if r["type"] == "sysaudio"]
        self.assertEqual([r["state"] for r in st], ["running", "failed"])
        self.assertEqual(st[1]["detail"], "許可がありません")
        self.assertFalse(any(r["type"] == "audio" for r in rows))
        evts = [b.ui.get_nowait() for _ in range(b.ui.qsize())]
        self.assertEqual(evts[-1][0], "error")
        self.assertIn("マイクだけで続けます", evts[-1][2])


class TestStartScreenWarning(unittest.TestCase):
    HTML = (Path(__file__).resolve().parents[1] / "src/tsuyaku/webui/index.html").read_text()

    def test_warning_when_system_audio_with_voice(self):
        h = self.HTML
        self.assertIn('id="srcWarn"', h)
        i = h.index("srcWarnShow(){")
        body = h[i:i + 500]
        for cond in ('this.src==="mic+system"', 'this.mode!=="notes"', 'dataset.voice==="on"'):
            self.assertIn(cond, body)
        self.assertIn("読み上げ中は相手の声も聞けません — オンライン会議では「字幕だけ」がおすすめ", body)
        # 使い方・読み上げ・聞く音のどれを変えても出し直す
        self.assertIn("if(this.srcWarnShow)this.srcWarnShow();", h)
        self.assertIn("this.srcWarnShow();", h[h.index("segNotes(){"):h.index("segNotes(){") + 300])
        self.assertEqual(len(re.findall(r'"読み上げ中は相手の声も聞けません —[^"]*":', h)), 3)   # en/zh/ko の訳


@unittest.skipUnless(HELPER.exists(), "bin/sokutsu-sysaudio が無い(scripts/build_sysaudio.sh)")
class TestRealHelperTone(unittest.TestCase):
    """本物の部品の試験用の音(--tone: 許可も音声機器も使わない)で、読み取り〜変換を通す。"""

    def test_tone_through_real_binary(self):
        got = []
        sa = SystemAudio(got.append, cmd=[str(HELPER), "--tone", "1"], max_restarts=0)
        sa.start()
        self.assertTrue(wait_for(lambda: sa.samples >= 15900 or sa.state == "failed"))
        sa.stop()
        y = np.concatenate(got)
        self.assertAlmostEqual(peak_hz(y[1000:15000], 16000), 440, delta=3)


if __name__ == "__main__":
    unittest.main()


class TestStartScreenFits(unittest.TestCase):
    """背の低い窓(1180×720 など)で、聞く音の欄と注意が出ても左の列が画面からはみ出さない。
    実測は headless Chrome で 1440×900 / 1280×760 / 1180×720(タイトルバー分を引いた高さ)、
    4言語 × 4状態ではみ出しゼロ(11/07)。ここでは省く規則が残っていることだけ見る。"""
    HTML = TestStartScreenWarning.HTML

    def test_short_window_rules(self):
        h = self.HTML
        i = h.index("@media (max-height:940px)")
        self.assertIn("#pairNote,#srcNote{display:none}", h[i:i + 200])
        j = h.index("@media (max-height:760px)")
        self.assertIn("#start .tag{display:none}", h[j:j + 200])
        # 説明を隠しても、聞く音のボタンの title で読める
        self.assertIn('querySelectorAll("#audioSrc .pair").forEach(x=>x.title=', h)
