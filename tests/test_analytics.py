"""W4 話者の分析(11/07): 話した時間・割合・発言の回数(ターン)・いちばん長い連続の発言・時間の帯。"""
import json
import tempfile
import time
import unittest
from pathlib import Path

from tsuyaku import analytics as A
from tsuyaku import notes as N
from tsuyaku.log import SessionLog

T0 = time.mktime(time.strptime("2026-11-06 10:00:00", "%Y-%m-%d %H:%M:%S"))


def line(spk, start, dur, orig="あいうえお", lang="ja"):
    return {"spk": spk, "lang": lang, "orig": orig, "t0": T0 + start, "dur": dur,
            "ts": T0 + start + dur + 1.0, "t": time.strftime("%H:%M:%S", time.localtime(T0 + start))}


MEET = {"meta": {"start": "2026-11-06 10:00:00"}, "lines": [
    line("S1", 0, 10), line("S1", 11, 5), line("S2", 17, 4, "質問です。いつですか？"),
    line("S1", 22, 3), line("S2", 26, 20), line("S2", 47, 10)]}


class TestStats(unittest.TestCase):
    def test_talk_time_share_turns_longest(self):
        st = A.speaker_stats(MEET)
        sp = {s["id"]: s for s in st["speakers"]}
        self.assertEqual([s["id"] for s in st["speakers"]], ["S2", "S1"])     # 長く話した順
        self.assertAlmostEqual(sp["S1"]["secs"], 18.0)
        self.assertAlmostEqual(sp["S2"]["secs"], 34.0)
        self.assertAlmostEqual(sp["S1"]["share"] + sp["S2"]["share"], 1.0, places=3)
        self.assertEqual(sp["S1"]["turns"], 2)          # S1 S1 | S2 | S1 | S2 S2
        self.assertEqual(sp["S2"]["turns"], 2)
        self.assertEqual(sp["S1"]["lines"], 3)
        self.assertAlmostEqual(sp["S2"]["longest"], 30.0)
        self.assertEqual(sp["S2"]["longest_i"], 4)      # いちばん長い連続の始まりの行(飛ぶ用)
        self.assertAlmostEqual(sp["S1"]["longest"], 15.0)
        self.assertEqual(sp["S2"]["questions"], 1)
        self.assertFalse(st["estimated"])
        self.assertAlmostEqual(st["total"], 52.0)
        self.assertAlmostEqual(st["span"], 57.0)

    def test_timeline_merges_close_lines_of_the_same_speaker(self):
        st = A.speaker_stats(MEET)
        tl = st["timeline"]
        self.assertEqual(tl[0], ["S1", 0.0, 16.0, 0])   # 0-10 と 11-16 は1つの帯(最後は最初の行)
        self.assertEqual([x[0] for x in tl], ["S1", "S2", "S1", "S2"])
        self.assertEqual(tl[-1], ["S2", 26.0, 57.0, 4])

    def test_old_logs_without_timing_are_estimated_from_text(self):
        rows = [{"spk": "S1", "lang": "ja", "orig": "あ" * 30, "ts": T0 + 6.0, "t": "10:00:06"},
                {"spk": "S2", "lang": "en", "orig": "x" * 70, "ts": T0 + 14.0, "t": "10:00:14"},
                {"spk": "S1", "lang": "ja", "orig": "はい", "t": "10:00:20"}]          # ts も無い
        st = A.speaker_stats({"meta": {"start": "2026-11-06 10:00:00"}, "lines": rows})
        self.assertTrue(st["estimated"])
        sp = {s["id"]: s for s in st["speakers"]}
        self.assertGreater(sp["S1"]["secs"], 3.0)
        self.assertLess(sp["S1"]["secs"], 8.0)
        self.assertGreater(sp["S2"]["secs"], 3.0)
        for _spk, a, b, _i in st["timeline"]:
            self.assertGreaterEqual(a, 0.0)
            self.assertGreater(b, a)

    def test_empty_meeting(self):
        st = A.speaker_stats({"meta": {}, "lines": []})
        self.assertEqual(st["speakers"], [])
        self.assertEqual(st["timeline"], [])
        self.assertEqual(st["total"], 0.0)


class TestLogTiming(unittest.TestCase):
    def test_log_keeps_start_and_duration_and_loader_reads_them(self):
        with tempfile.TemporaryDirectory() as d:
            log = SessionLog(d, "ja-en", mode="notes")
            log.add("ja", "こんにちは", "", None, "S1", [0, 1], timing={"t0": 1000.123, "dur": 2.345})
            log.add("ja", "時刻なし", "", None, "S1")
            log.close()
            row = json.loads(Path(log.path_jsonl).read_text(encoding="utf-8").splitlines()[1])
            self.assertEqual((row["t0"], row["dur"]), (1000.12, 2.35))
            m = N.load_meeting(log.path_jsonl)
            self.assertEqual(m["lines"][0]["dur"], 2.35)
            self.assertEqual(m["lines"][0]["t0"], 1000.12)
            self.assertNotIn("dur", m["lines"][1])


if __name__ == "__main__":
    unittest.main()
