"""声・話速・翻訳ペイロード(reasoning_effort)のユニットテスト。

実行: PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -v
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import requests

from tsuyaku.translate import translate_stream
from tsuyaku.tts import SAY_BASE_RATE, TTS, parse_voice, say_cmd


class TestParseVoice(unittest.TestCase):
    def test_say_spec(self):
        self.assertEqual(parse_voice("say:Kyoko"), ("say", "Kyoko"))

    def test_kokoro_spec(self):
        self.assertEqual(parse_voice("kokoro:3"), ("kokoro", 3))

    def test_broken_specs_fall_back(self):
        self.assertEqual(parse_voice(""), ("say", ""))
        self.assertEqual(parse_voice("Kyoko"), ("say", ""))
        self.assertEqual(parse_voice("kokoro:abc"), ("kokoro", 0))
        self.assertEqual(parse_voice("vits:1"), ("say", "1"))


class TestSayCmd(unittest.TestCase):
    def test_default_speed_has_no_rate(self):
        self.assertEqual(say_cmd("Kyoko"), ["say", "-v", "Kyoko"])

    def test_speed_maps_to_rate(self):
        cmd = say_cmd("Alex", 1.5)
        self.assertEqual(cmd[:4], ["say", "-v", "Alex", "-r"])
        self.assertEqual(int(cmd[4]), int(SAY_BASE_RATE * 1.5))

    def test_rate_is_clamped(self):
        self.assertLessEqual(int(say_cmd("", 9.9)[-1]), 520)
        self.assertGreaterEqual(int(say_cmd("", 0.01)[-1]), 80)


class TestTTSLiveSettings(unittest.TestCase):
    def _tts(self):
        return TTS("", "", {"en": 3, "ja": 37, "zh": 45}, "say")

    def test_set_voice_speed(self):
        t = self._tts()
        t.set_voice("en", "say:Alex")
        t.set_speed("en", 1.3)
        self.assertEqual(t.voices["en"], "say:Alex")
        self.assertAlmostEqual(t.speeds["en"], 1.3)

    def test_speed_clamped(self):
        t = self._tts()
        t.set_speed("ja", 9.0)
        self.assertEqual(t.speeds["ja"], 2.0)
        t.set_speed("ja", 0.1)
        self.assertEqual(t.speeds["ja"], 0.5)

    def test_constructor_takes_dicts(self):
        t = TTS("", "", {}, "say", voices={"ja": "say:Kyoko"}, speeds={"ja": 1.2})
        self.assertEqual(t.voices["ja"], "say:Kyoko")
        self.assertAlmostEqual(t.speeds["ja"], 1.2)


_SAY_DUMP = """Albert              en_US    # Hello! My name is Albert.
Eddy (日本語（日本）)  ja_JP    # こんにちは! 私の名前はEddyです。
Eddy (韓国語（韓国）)  ko_KR    # 안녕하세요. 제 이름은 Eddy입니다.
Eddy (英語（アメリカ）)  en_US    # Hello! My name is Eddy.
Kyoko               ja_JP    # こんにちは! 私の名前はKyokoです。
Yuna                ko_KR    # 안녕하세요. 제 이름은 유나입니다.
Zarvox              en_US    # Hello! My name is Zarvox.
壊れた行
# コメントだけ
"""


class TestParseSayLines(unittest.TestCase):
    def test_multiword_names_kept(self):
        from tsuyaku.__main__ import parse_say_lines
        out = parse_say_lines(_SAY_DUMP)
        self.assertEqual(out["ja"], ["Eddy (日本語（日本）)", "Kyoko"])
        self.assertEqual(out["ko"], ["Eddy (韓国語（韓国）)", "Yuna"])
        self.assertEqual(out["en"], ["Albert", "Eddy (英語（アメリカ）)", "Zarvox"])
        self.assertEqual(out["zh"], [])

    def test_real_machine_has_variety(self):
        from tsuyaku.__main__ import _say_voices
        v = _say_voices()
        for lang in ("ja", "en", "zh", "ko"):
            self.assertGreaterEqual(len(v[lang]), 8, f"{lang} の声が少なすぎる(解析ミス?)")
        self.assertIn("Kyoko", v["ja"])


class _FakeResp:
    def __init__(self, status: int, lines: list[bytes]):
        self.status_code = status
        self._lines = lines

    def iter_lines(self):
        return iter(self._lines)

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(str(self.status_code))


_OK_LINES = [b'data: {"choices":[{"delta":{"content":"\xe3\x81\x93\xe3\x82\x93\xe3\x81\xab\xe3\x81\xa1\xe3\x81\xaf"}}]}',
             b"data: [DONE]"]


class TestTranslatePayload(unittest.TestCase):
    def setUp(self):
        import tsuyaku.translate as tr
        self.tr = tr
        self._orig_post = tr.requests.post
        self.payloads: list[dict] = []

    def tearDown(self):
        self.tr.requests.post = self._orig_post

    def _fake(self, *resps):
        it = iter(resps)

        def post(url, json=None, stream=None, timeout=None):
            self.payloads.append(dict(json))
            return next(it)
        self.tr.requests.post = post

    def test_reasoning_off_and_bigger_budget(self):
        self._fake(_FakeResp(200, _OK_LINES))
        out = translate_stream("http://x", "m", [{"role": "user", "content": "hi"}])
        self.assertEqual(out, "こんにちは")
        self.assertEqual(self.payloads[0]["reasoning_effort"], "none")
        self.assertEqual(self.payloads[0]["max_tokens"], 1024)

    def test_retry_once_without_reasoning_on_400(self):
        self._fake(_FakeResp(400, []), _FakeResp(200, _OK_LINES))
        out = translate_stream("http://x", "m", [{"role": "user", "content": "hi"}])
        self.assertEqual(out, "こんにちは")
        self.assertEqual(len(self.payloads), 2)
        self.assertEqual(self.payloads[0]["reasoning_effort"], "none")
        self.assertNotIn("reasoning_effort", self.payloads[1])

    def test_deltas_forwarded(self):
        self._fake(_FakeResp(200, _OK_LINES))
        got: list[str] = []
        translate_stream("http://x", "m", [{"role": "user", "content": "hi"}],
                         on_delta=got.append)
        self.assertEqual("".join(got), "こんにちは")


class TestBridgeVoiceAPI(unittest.TestCase):
    def test_get_tts_shape(self):
        from tsuyaku.__main__ import Bridge
        b = Bridge()
        cfg = b.get_tts()
        for k in ("voices", "speeds", "kokoro", "sayVoices"):
            self.assertIn(k, cfg)
        self.assertIn("ja", cfg["voices"])
        self.assertIn("ja", cfg["speeds"])

    def test_set_tts_persists_and_clamps(self):
        from tsuyaku import config as cfgmod
        from tsuyaku.__main__ import Bridge
        with tempfile.TemporaryDirectory() as d:
            with mock.patch.object(cfgmod, "ROOT", Path(d)):
                b = Bridge()
                b.s.tts_voice = {"ja": "say:Kyoko"}
                b.s.tts_speed = {"ja": 1.0}
                b.set_tts("ja", voice="kokoro:39", speed=3.0)
                cfg = json.loads((Path(d) / "config.json").read_text())
                self.assertEqual(cfg["tts_voice"]["ja"], "kokoro:39")
                self.assertEqual(cfg["tts_speed"]["ja"], 2.0)  # 上限に_clamp
                # 再読込で設定が復活すること
                with mock.patch.object(cfgmod, "ROOT", Path(d)):
                    s2 = cfgmod.load_settings()
                    self.assertEqual(s2.tts_voice["ja"], "kokoro:39")

    def test_preview_spawns_thread_without_speaking(self):
        """preview_tts は別スレッドで読む(テスト中に実際の音を出さないよう
        Thread を差し替えて起動確認のみ行う)。"""
        import tsuyaku.__main__ as m
        from tsuyaku.__main__ import Bridge
        b = Bridge()
        started: list = []

        class FakeThread:
            def __init__(self, target=None, args=(), daemon=None):
                self.target, self.args, self.daemon = target, args, daemon

            def start(self):
                started.append(self.target)

        with mock.patch.object(m.threading, "Thread", FakeThread):
            self.assertEqual(b.preview_tts("ja", "say:Kyoko", 1.0), "ok")
        self.assertEqual(len(started), 1)


class TestUIStruct(unittest.TestCase):
    def test_index_has_voice_ui_wiring(self):
        html = (Path(__file__).resolve().parents[1] / "src" / "tsuyaku"
                / "webui" / "index.html").read_text()
        for token in ("get_tts", "set_tts", "preview_tts", "voiceRows", "vmodal", "vspd"):
            self.assertIn(token, html)


if __name__ == "__main__":
    unittest.main()
