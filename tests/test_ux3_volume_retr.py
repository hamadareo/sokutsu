"""ux3(11/10): 読み上げ音量・訳し直しの Python 側。"""
import unittest
from unittest import mock

import tsuyaku.__main__ as main_mod
from tsuyaku import config as cfgmod
from tsuyaku import tts as ttsmod


class TestVolume(unittest.TestCase):
    def test_default_and_clamp(self):
        self.assertEqual(cfgmod.Settings().tts_volume, 1.0)
        self.assertEqual(ttsmod.clamp_volume(5), 1.0)
        self.assertEqual(ttsmod.clamp_volume(0), 0.1)
        self.assertEqual(ttsmod.clamp_volume("x"), 1.0)

    def test_say_text_prefix(self):
        self.assertEqual(ttsmod.say_text("こんにちは", 1.0), "こんにちは")
        self.assertEqual(ttsmod.say_text("hi", 0.5), "[[volm 0.50]] hi")

    def test_afplay_args(self):
        self.assertEqual(ttsmod.afplay_cmd("/a.wav", 1.0), ["afplay", "/a.wav"])
        self.assertEqual(ttsmod.afplay_cmd("/a.wav", 0.4), ["afplay", "-v", "0.40", "/a.wav"])

    def test_bridge_set_get(self):
        from test_hear_devices import _bridge
        b = _bridge()
        with mock.patch.object(main_mod, "save_config", lambda s: None):
            b.set_volume(0.3)
        self.assertAlmostEqual(b.s.tts_volume, 0.3)
        self.assertAlmostEqual(b.get_tts()["volume"], 0.3)

    def test_config_roundtrip_key(self):
        import inspect
        self.assertIn('data["tts_volume"]', inspect.getsource(cfgmod.save_config))


class TestRetranslate(unittest.TestCase):
    def test_no_context_returns_empty(self):
        from test_hear_devices import _bridge
        b = _bridge()
        self.assertEqual(b.retranslate("こんにちは", "ja"), "")

    def test_uses_context(self):
        from test_hear_devices import _bridge
        b = _bridge()
        b._rt_ctx = ("http://x", "m", ("ja", "en"), "topic")
        b.history = []
        with mock.patch.object(main_mod, "translate_with_retry", return_value="Hello") as t:
            self.assertEqual(b.retranslate("こんにちは", "ja"), "Hello")
        self.assertEqual(t.call_args[0][0], "http://x")

    def test_failure_returns_empty(self):
        from test_hear_devices import _bridge
        b = _bridge()
        b._rt_ctx = ("http://x", "m", ("ja", "en"), "")
        b.history = []
        with mock.patch.object(main_mod, "translate_with_retry", side_effect=RuntimeError("x")):
            self.assertEqual(b.retranslate("a", "ja"), "")


if __name__ == "__main__":
    unittest.main()
