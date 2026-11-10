"""記録と録音の対応(W2): meta の audio/ts0、発言の a0/a1(録音の先頭からの秒)。"""
import json
import tempfile
import time
import unittest
from pathlib import Path

from tsuyaku import notes as N
from tsuyaku.log import SessionLog


def rows(p):
    return [json.loads(x) for x in Path(p).read_text(encoding="utf-8").splitlines()]


class TestLogAudio(unittest.TestCase):
    def test_meta_has_audio_and_ts0_when_recording(self):
        with tempfile.TemporaryDirectory() as d:
            t = time.time()
            log = SessionLog(d, "ja-en", "", "notes", audio=True)
            log.close()
            meta = rows(log.path_jsonl)[0]
            stem = Path(log.path_jsonl).stem
            self.assertEqual(meta["audio"], f"{stem}.audio.m4a")
            self.assertAlmostEqual(meta["ts0"], t, delta=2)
            self.assertEqual(log.ts0, meta["ts0"])
            self.assertEqual(Path(log.audio_caf).name, f"{stem}.audio.caf")

    def test_no_audio_keys_without_recording(self):
        with tempfile.TemporaryDirectory() as d:
            log = SessionLog(d, "ja-en", "", "interpret")
            log.close()
            meta = rows(log.path_jsonl)[0]
            self.assertNotIn("audio", meta)
            self.assertIsNone(log.audio_caf)

    def test_extra_meta(self):
        with tempfile.TemporaryDirectory() as d:
            log = SessionLog(d, "en", "", "notes", extra={"source": "file", "file": "talk.mp4"})
            log.close()
            meta = rows(log.path_jsonl)[0]
            self.assertEqual(meta["source"], "file")
            self.assertEqual(meta["file"], "talk.mp4")

    def test_utter_carries_audio_span_and_loader_passes_it(self):
        with tempfile.TemporaryDirectory() as d:
            log = SessionLog(d, "ja-en", "", "notes", audio=True)
            log.add("ja", "こんにちは。", "", None, "S1", [0], audio=(1.234, 2.5))
            log.add("en", "Hello.", "", None, "S2", [1])            # 録音が無い行
            log.add("ja", "取り込み。", "", None, "S1", [2], audio=(3.0, 4.0),
                    ts=1000.0, t="00:00:03")
            log.close()
            r = rows(log.path_jsonl)
            self.assertEqual((r[1]["a0"], r[1]["a1"]), (1.23, 2.5))
            self.assertNotIn("a0", r[2])
            self.assertEqual((r[3]["ts"], r[3]["t"]), (1000.0, "00:00:03"))
            m = N.load_meeting(log.path_jsonl)
            self.assertEqual((m["lines"][0]["a0"], m["lines"][0]["a1"]), (1.23, 2.5))
            self.assertNotIn("a0", m["lines"][1])


if __name__ == "__main__":
    unittest.main()
