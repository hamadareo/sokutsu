"""Bridge の W2 の分: 会話中の録音(マイクの全チャンク)・行と録音の対応・再生の URL・字幕・取り込みの知らせ。"""
import json
import queue
import tempfile
import threading
import time
import unittest
import urllib.request
from pathlib import Path
from unittest.mock import patch

import numpy as np

from tsuyaku import importer as I
from tsuyaku.__main__ import Bridge
from tsuyaku.audio import SpeechCapture
from tsuyaku.log import SessionLog

SR = 16000


class UI:
    def __init__(self):
        self.ev = []

    def put(self, e):
        self.ev.append(e)

    def w2(self, typ):
        return [e[1] for e in self.ev if e[0] == "w2" and e[1].get("type") == typ]


def bridge(d):
    b = Bridge()
    b.s.log_dir = d
    b.s.record_audio = "auto"   # 本物の config.json(利用者が「残さない」にしていても)に左右されない
    b.ui = UI()
    return b


class TestLiveRecording(unittest.TestCase):
    def test_conversation_records_and_maps_lines(self):
        with tempfile.TemporaryDirectory() as d:
            b = bridge(d)
            self.assertTrue(b._rec_wanted("notes"))
            self.assertFalse(b._rec_wanted("interpret"))
            b.log = SessionLog(d, "ja-en", "", "notes", audio=b._rec_wanted("notes"))
            b._rec_open()
            self.assertIsNotNone(b._rec)
            self.assertEqual(b.ui.w2("rec")[-1]["on"], True)
            # SpeechCapture.on_chunk が録音にも流す(読み上げで止めている間も)
            cap = SpeechCapture.__new__(SpeechCapture)
            cap.tap = b._rec_tap
            cap.paused, cap._stop = threading.Event(), threading.Event()
            cap._was_paused = True
            cap.paused.set()
            cap.segmenter = type("S", (), {"flush_current": lambda self: []})()
            cap.uplink = None
            for _ in range(20):
                cap.on_chunk(np.full(512, 0.1, np.float32))
            self.assertAlmostEqual(b._rec.seconds, 20 * 512 / SR, delta=0.2)
            self.assertIsNone(b._rec_span({"span": (0.0, 0.0)}))   # マイクの時計はまだ進んでいない
            for _ in range(10):
                b._rec_tap(np.full(512, 0.1, np.float32), False)
            span = b._rec_span({"span": (0.0, 0.2)})
            self.assertAlmostEqual(span[0], 20 * 512 / SR, delta=0.2)   # 読み上げで止めていた分だけ後ろ
            self.assertAlmostEqual(span[1] - span[0], 0.2, places=3)
            b.log.add("ja", "テスト。", "", None, "S1", [0], audio=span)
            b._rec_stop(wait=True)
            stem = Path(b.log.path_jsonl).stem
            b.log.close()
            self.assertTrue((Path(d) / f"{stem}.audio.m4a").exists())
            self.assertFalse((Path(d) / f"{stem}.audio.caf").exists())
            self.assertEqual(b.ui.w2("audio_ready")[-1]["id"], stem)

    def test_no_recording_in_interpret_by_default(self):
        with tempfile.TemporaryDirectory() as d:
            b = bridge(d)
            b.log = SessionLog(d, "ja-en", "", "interpret", audio=b._rec_wanted("interpret"))
            b._rec_open()
            self.assertIsNone(b._rec)
            self.assertIsNone(b._rec_span({"span": (1.0, 2.0)}))
            b._rec_tap(np.zeros(512, np.float32), False)   # 録音なしでも落ちない
            b._rec_stop()
            b.log.close()

    def test_record_setting_saved(self):
        with tempfile.TemporaryDirectory() as d:
            b = bridge(d)
            with patch("tsuyaku.config.save_config") as sv:
                self.assertEqual(b.set_record("on"), "on")
                self.assertEqual(b.set_record("bogus"), "auto")
            self.assertEqual(sv.call_count, 2)
            b.s.record_audio = "on"
            self.assertTrue(b._rec_wanted("interpret"))

    def test_push_w2_event(self):
        b = Bridge()
        js = []
        b.window = type("W", (), {"evaluate_js": lambda self, s: js.append(s)})()
        b._push(("w2", {"type": "rec", "on": True}))
        self.assertEqual(js, ['window.TSUYAKU&&TSUYAKU.w2&&TSUYAKU.w2({"type": "rec", "on": true})'])


class TestPlaybackAndExport(unittest.TestCase):
    def test_audio_info_serves_m4a(self):
        with tempfile.TemporaryDirectory() as d:
            b = bridge(d)
            log = SessionLog(d, "ja-en", "", "notes", audio=True)
            log.add("ja", "こんにちは。", "", None, "S1", [0], audio=(0.5, 1.5))
            log.close()
            stem = Path(log.path_jsonl).stem
            self.assertIsNone(b.audio_info(stem))           # 録音ファイルがまだ無い
            from tsuyaku import recorder as R
            r = R.Recorder(log.audio_caf, ts0=log.ts0)
            for _ in range(60):
                r.tap(np.full(512, 0.1, np.float32), False)
            r.close()
            info = b.audio_info(stem)
            self.assertTrue(info["growing"])                 # 変換前は CAF
            R.finalize(log.audio_caf)
            info = b.audio_info(stem)
            self.assertFalse(info["growing"])
            self.assertEqual(info["ts0"], log.ts0)
            self.assertGreater(info["dur"], 1.5)
            with urllib.request.urlopen(info["url"], timeout=5) as resp:
                self.assertEqual(resp.headers["Content-Type"], "audio/mp4")
            g = b.meeting_get(stem)
            self.assertEqual(g["lines"][0]["a0"], 0.5)
            b._media.close()

    def test_subtitles_text(self):
        with tempfile.TemporaryDirectory() as d:
            b = bridge(d)
            log = SessionLog(d, "ja-en", "", "notes", audio=True)
            log.add("ja", "こんにちは。", "", None, "S1", [0], audio=(0.5, 1.5))
            log.close()
            stem = Path(log.path_jsonl).stem
            out = b.subtitles_text(stem, "vtt", "ja")
            self.assertIn("00:00:00.500 --> 00:00:01.500", out)
            self.assertIn("<v 話者1>こんにちは。", out)
            out = b.subtitles_export(stem, "srt", "en")       # 窓が無ければ記録の隣に
            self.assertTrue(out.endswith(".srt"))
            self.assertIn("Speaker 1: こんにちは。", Path(out).read_text(encoding="utf-8"))
            self.assertEqual(b.subtitles_text("session-nope", "srt"), "")


class TestImportFlow(unittest.TestCase):
    def test_import_start_reports_progress_and_done(self):
        with tempfile.TemporaryDirectory() as d:
            b = bridge(d)
            src = Path(d) / "talk.m4a"
            src.write_bytes(b"x")

            go = threading.Event()

            def fake_run(path, log_dir, pair, topic, **kw):
                kw["prog"]("read", 0.0)
                kw["prog"]("listen", 0.5)
                go.wait(5)            # 2回目の呼び出しの間は取り込み中のまま(終わるのが速すぎると "busy" を見られない)
                return {"id": "session-20261107-120000", "lines": 3, "dur": 60.0}

            fake_ear = (type("E", (), {"langs": (), "prompt": ""})(), type("L", (), {"langs": (), "fast": True})(), None)
            with patch.object(Bridge, "_import_ear", return_value=fake_ear), \
                    patch("tsuyaku.importer.run_import", side_effect=fake_run):
                self.assertEqual(b.import_start(str(src), "ja-en", ""), "started")
                self.assertEqual(b.import_start(str(src), "ja-en", ""), "busy")
                go.set()
                b._import_th.join(5)
            ev = b.ui.w2("import")
            self.assertEqual(ev[-1]["stage"], "done")
            self.assertEqual(ev[-1]["id"], "session-20261107-120000")
            self.assertIn("listen", [e["stage"] for e in ev])

    def test_import_refuses_while_live_or_unsupported(self):
        with tempfile.TemporaryDirectory() as d:
            b = bridge(d)
            self.assertEqual(b.import_start(str(Path(d) / "a.txt"), "ja", ""), "unsupported")
            b.running.set()
            src = Path(d) / "a.wav"
            src.write_bytes(b"x")
            self.assertEqual(b.import_start(str(src), "ja", ""), "live")

    def test_import_error_is_reported(self):
        with tempfile.TemporaryDirectory() as d:
            b = bridge(d)
            src = Path(d) / "a.mp3"
            src.write_bytes(b"x")
            fake_ear = (type("E", (), {"langs": (), "prompt": ""})(), type("L", (), {"langs": (), "fast": True})(), None)
            with patch.object(Bridge, "_import_ear", return_value=fake_ear), \
                    patch("tsuyaku.importer.run_import", side_effect=I.ImportFailed("読めません")):
                b.import_start(str(src), "ja", "")
                b._import_th.join(5)
            self.assertEqual(b.ui.w2("import")[-1], dict(b.ui.w2("import")[-1], stage="error", error="読めません"))


if __name__ == "__main__":
    unittest.main()
