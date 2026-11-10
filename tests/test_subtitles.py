"""字幕(SRT/VTT)とテキストの書き出し(W2)。時刻は録音の中の秒(a0/a1)、話者は名前。"""
import unittest

from tsuyaku import subtitles as S

LINES = [
    {"t": "10:00:05", "lang": "ja", "orig": "山田です。", "trans": "I'm Yamada.", "spk": "S1",
     "ts": 1005.0, "a0": 1.5, "a1": 3.25},
    {"t": "10:00:09", "lang": "en", "orig": "Hi.", "trans": "", "spk": "S2",
     "ts": 1009.0, "a0": 3.0, "a1": 4.0},          # 前の行と重なる
    {"t": "10:01:02", "lang": "ja", "orig": "では --> 決定です。", "trans": "", "spk": "S1",
     "ts": 1062.0, "a0": 3661.2, "a1": 3663.0},
]
NAMES = {"S1": "山田"}


class TestSubtitles(unittest.TestCase):
    def test_srt(self):
        out = S.to_srt(LINES, NAMES, ui="ja")
        blocks = out.strip().split("\n\n")
        self.assertEqual(len(blocks), 3)
        self.assertEqual(blocks[0].splitlines(), [
            "1", "00:00:01,500 --> 00:00:03,000", "山田: 山田です。", "I'm Yamada."])
        self.assertEqual(blocks[1].splitlines()[1], "00:00:03,000 --> 00:00:04,000")
        self.assertEqual(blocks[1].splitlines()[2], "話者2: Hi.")
        self.assertEqual(blocks[2].splitlines()[1], "01:01:01,200 --> 01:01:03,000")
        self.assertNotIn("-->", blocks[2].splitlines()[2])   # 本文の --> は字幕を壊すので置き換える

    def test_srt_without_translation(self):
        out = S.to_srt(LINES, NAMES, ui="ja", with_trans=False)
        self.assertNotIn("I'm Yamada.", out)

    def test_vtt(self):
        out = S.to_vtt(LINES, NAMES, ui="en")
        self.assertTrue(out.startswith("WEBVTT\n\n"))
        self.assertIn("00:00:01.500 --> 00:00:03.000\n<v 山田>山田です。\nI'm Yamada.", out)
        self.assertIn("<v Speaker 2>Hi.", out)
        self.assertIn("<v 山田>では → 決定です。", out)

    def test_vtt_escapes(self):
        out = S.to_vtt([{"orig": "a < b & c", "spk": "S1", "a0": 0, "a1": 1}], {"S1": "A<B>"}, ui="ja")
        self.assertIn("<v A&lt;B&gt;>a &lt; b &amp; c", out)

    def test_text(self):
        out = S.to_text(LINES, NAMES, ui="ja", title="定例")
        ls = out.splitlines()
        self.assertEqual(ls[0], "定例")
        self.assertIn("[00:00:01] 山田: 山田です。", ls)
        self.assertIn("    I'm Yamada.", ls)
        self.assertIn("[01:01:01] 山田: では --> 決定です。", ls)

    def test_without_audio_uses_wall_clock(self):
        # 録音が無い会議: 時刻は会議の始まり(最初の発言)からの秒で近似。並びは崩さない
        ls = [dict(x) for x in LINES]
        for x in ls:
            x.pop("a0")
            x.pop("a1")
        cues = S.cues(ls, {}, ui="ja", t0=1000.0)
        starts = [c[0] for c in cues]
        self.assertEqual(starts, sorted(starts))
        self.assertTrue(all(c[1] > c[0] for c in cues))
        self.assertAlmostEqual(cues[0][1], 5.0, places=2)

    def test_min_length_and_monotonic(self):
        ls = [{"orig": "a", "spk": "", "a0": 5.0, "a1": 5.0}, {"orig": "b", "spk": "", "a0": 4.0, "a1": 6.0}]
        cues = S.cues(ls, {}, ui="ja")
        self.assertGreaterEqual(cues[0][1] - cues[0][0], 0.5)
        self.assertGreaterEqual(cues[1][0], cues[0][0])


if __name__ == "__main__":
    unittest.main()
