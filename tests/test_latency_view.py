"""応答速度の可視化(11/06「応答速度を可視化できるようにしてください」)。

話し終わりを起点に、文の確定・最初の訳(予想訳含む)・確定字幕・声の鳴り始めを
秒で測って画面へ送る。あわせて partial/error の引数の渡し方の不具合を直した。
"""
import json
import time
import unittest

from tsuyaku.__main__ import js_args, latency_marks


class TestLatencyMarks(unittest.TestCase):
    def test_guess_before_confirm(self):
        end = 100.0
        lat = latency_marks(end, 100.9, 100.6, 100.95, 101.0, True)
        self.assertEqual(lat, {"heard": 0.9, "first": 0.6, "caption": 1.0, "guess": True})

    def test_no_speculation(self):
        lat = latency_marks(100.0, 100.3, 0.0, 100.6, 100.8, False)
        self.assertEqual(lat["first"], 0.6)
        self.assertFalse(lat["guess"])

    def test_unknown_end(self):
        self.assertIsNone(latency_marks(None, 1.0, 0.0, 2.0, 3.0, False))


class TestJsArgs(unittest.TestCase):
    def test_positional(self):
        # 以前は TSUYAKU.partial(["原文","訳","ja"]) と配列1個で渡していた
        self.assertEqual(js_args(("原文", "訳", "ja", True)), '"原文","訳","ja",true')

    def test_quotes_escaped(self):
        s = js_args(('a"b</script>',))
        self.assertEqual(json.loads("[" + s + "]"), ['a"b</script>'])


class TestWallClock(unittest.TestCase):
    def test_wall_of(self):
        from tsuyaku.audio import SpeechCapture
        cap = SpeechCapture.__new__(SpeechCapture)
        cap._clock = (10.0, 1000.0)
        self.assertAlmostEqual(cap.wall_of(9.5), 999.5)
        cap._clock = None
        self.assertIsNone(cap.wall_of(9.5))


class TestTtsOnStart(unittest.TestCase):
    def test_on_start_called_once_when_playback_begins(self):
        from unittest.mock import patch
        from tsuyaku.tts import TTS
        t = TTS("", "", {}, voices={"ja": "say:Kyoko"})
        hits = []
        t.on_start = lambda: hits.append(time.time())

        class P:
            def poll(self):
                return 0
        with patch("subprocess.Popen", return_value=P()):
            t.say("はい。", "ja")
            t.say("はい。", "ja")
        self.assertEqual(len(hits), 1)


class TestWiring(unittest.TestCase):
    def test_app_sends_latency(self):
        from pathlib import Path
        src = (Path(__file__).resolve().parents[1] / "src/tsuyaku/__main__.py").read_text()
        self.assertIn('"id": uid, "lat": lat', src)
        self.assertIn('("voice", {"id": _u', src)


if __name__ == "__main__":
    unittest.main()
