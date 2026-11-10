"""VOICEVOX ENGINE(このMacローカル 127.0.0.1:50021)連携のテスト。

エンジン実物には触れず、requests を偽装して合成2段(/audio_query→/synthesis)と
say へのフォールバック、音色一覧の選別、UI・起動スクリプトの組み込みを確かめる。

実行: PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -v
"""
from __future__ import annotations

import json
import os
import unittest
from pathlib import Path
from unittest.mock import patch

import requests

from tsuyaku.config import ROOT, Settings, load_settings
from tsuyaku.tts import TTS, parse_voice, voicevox_voices

# /speakers の実物形式(抜粋): styles の name でノーマル/アナウンスを選ぶ
_FAKE_SPEAKERS = [
    {"name": "四国めたん", "styles": [
        {"id": 2, "name": "ノーマル"}, {"id": 7, "name": "クイック"}]},
    {"name": "青山龍星", "styles": [{"id": 13, "name": "ノーマル"}]},
    {"name": "No.7", "styles": [{"id": 30, "name": "アナウンス"}]},
]


class FakeResp:
    def __init__(self, payload=None, content: bytes = b"", status: int = 200):
        self._payload = payload
        self.content = content
        self.status_code = status

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(str(self.status_code))

    def json(self):
        return self._payload


class TestParseVoice(unittest.TestCase):
    def test_voicevox_sid(self):
        self.assertEqual(parse_voice("voicevox:13"), ("voicevox", 13))

    def test_voicevox_broken_sid_defaults_ryusei(self):
        self.assertEqual(parse_voice("voicevox:あ"), ("voicevox", 13))

    def test_other_kinds_unchanged(self):
        self.assertEqual(parse_voice("say:Kyoko"), ("say", "Kyoko"))
        self.assertEqual(parse_voice("kokoro:37"), ("kokoro", 37))
        self.assertEqual(parse_voice(""), ("say", ""))

    def test_set_voice_accepts_voicevox(self):
        t = TTS("", "", {}, voicevox_url="http://127.0.0.1:50021")
        t.set_voice("ja", "voicevox:2")
        self.assertEqual(t.voices["ja"], "voicevox:2")


class TestVoicevoxVoices(unittest.TestCase):
    def test_curation_normal_and_announce_only(self):
        with patch("requests.get", return_value=FakeResp(payload=_FAKE_SPEAKERS)):
            vs = voicevox_voices("http://x")
        self.assertEqual([v["id"] for v in vs], [2, 13, 30])  # クイック(7)は出ない
        self.assertEqual(vs[0]["name"], "四国めたん")
        self.assertEqual(vs[2]["name"], "No.7 アナウンス")

    def test_engine_down_returns_empty(self):
        with patch("requests.get", side_effect=requests.ConnectionError):
            self.assertEqual(voicevox_voices("http://x"), [])

    def test_empty_url_returns_empty(self):
        self.assertEqual(voicevox_voices(""), [])


class TestSayVoicevox(unittest.TestCase):
    def _tts(self) -> TTS:
        return TTS("", "", {}, voices={"ja": "voicevox:13"},
                   speeds={"ja": 1.4}, voicevox_url="http://vv.local")

    def test_synth_two_stage_and_speed(self):
        calls = []

        def fake_post(url, **kw):
            calls.append((url, kw))
            if "/audio_query" in url:
                return FakeResp(payload={"speedScale": 1.0, "volumeScale": 1.0})
            return FakeResp(content=b"RIFFfakewav")

        with patch("requests.post", side_effect=fake_post), \
             patch.object(TTS, "_play_wav", return_value=True) as pw:
            ok = self._tts().say("こんにちは。", "ja")
        self.assertTrue(ok)
        self.assertEqual(len(calls), 2)
        self.assertIn("/audio_query?speaker=13", calls[0][0])
        self.assertIn("text=", calls[0][0])
        self.assertIn("/synthesis?speaker=13", calls[1][0])
        self.assertEqual(calls[1][1]["json"]["speedScale"], 1.4)  # 話速が渡る
        pw.assert_called_once_with(b"RIFFfakewav")

    def test_speed_clamped(self):
        def fake_post(url, **kw):
            if "/audio_query" in url:
                return FakeResp(payload={"speedScale": 1.0})
            return FakeResp(content=b"RIFF")

        t = self._tts()
        t.set_speed("ja", 9.0)
        with patch("requests.post", side_effect=fake_post), \
             patch.object(TTS, "_play_wav", return_value=True):
            t.say("テスト。", "ja")
        self.assertLessEqual(t.speeds["ja"], 2.0)  # set_speed 側でも丸めている

    def test_falls_back_to_say_when_engine_down(self):
        t = self._tts()
        with patch("requests.post", side_effect=requests.ConnectionError), \
             patch.object(TTS, "_say_macos", return_value=True) as sm:
            ok = t.say("こんにちは。", "ja")
        self.assertTrue(ok)
        sm.assert_called_once()
        self.assertEqual(sm.call_args[0][1], "Kyoko")  # say 既定声に落ちる

    def test_falls_back_to_say_when_url_unset(self):
        t = TTS("", "", {}, voices={"ja": "voicevox:13"})  # voicevox_url 無し
        with patch.object(TTS, "_say_macos", return_value=True) as sm:
            ok = t.say("こんにちは。", "ja")
        self.assertTrue(ok)
        sm.assert_called_once()


class TestConfig(unittest.TestCase):
    def test_defaults(self):
        s = Settings()
        self.assertEqual(s.voicevox_url, "http://127.0.0.1:50021")
        self.assertEqual(s.tts_voice["ja"], "voicevox:13")

    def test_env_override(self):
        with patch.dict(os.environ, {"TSUYAKU_VOICEVOX_URL": "http://1.2.3.4:1"}):
            self.assertEqual(load_settings().voicevox_url, "http://1.2.3.4:1")

    def test_saved_config_uses_voicevox_ja(self):
        data = json.loads((ROOT / "config.json").read_text())
        self.assertTrue(str(data.get("tts_voice", {}).get("ja", "")).startswith("voicevox:"))


class TestWiring(unittest.TestCase):
    def test_main_exposes_voicevox_list(self):
        src = (Path(__file__).resolve().parents[1] / "src/tsuyaku/__main__.py").read_text()
        self.assertIn('"voicevox": _voicevox_list(s.voicevox_url)', src)
        self.assertIn("voicevox_url=s.voicevox_url", src)
        # 試聴(予備)と会話(本体)はどちらも先に読んだ声 _shared_tts を使う(11/08 speed)。声の場所は1箇所で渡し、
        # 使い回すときも今の設定の voicevox_url に合わせる
        self.assertEqual(src.count("voicevox_url=s.voicevox_url"), 1)
        self.assertIn('t.voicevox_url = (s.voicevox_url or "").rstrip("/")', src)
        self.assertIn("self._preview_tts = _shared_tts(self.s)", src)
        self.assertIn('else _shared_tts(s)', src)

    def test_ui_lists_voicevox_for_ja(self):
        ui = (Path(__file__).resolve().parents[1] / "src/tsuyaku/webui/index.html").read_text()
        self.assertIn('if(lang==="ja")(cfg.voicevox||[]).forEach', ui)
        self.assertIn('"VOICEVOX "', ui)

    def test_run_script_starts_engine(self):
        root = Path(__file__).resolve().parents[1]
        run = (root / "run.sh").read_text()
        self.assertIn("voicevox_local.sh", run)
        self.assertIn("TSUYAKU_NO_VOICEVOX", run)
        sh = (root / "scripts/voicevox_local.sh").read_text()
        self.assertIn("50021", sh)
        self.assertIn("/version", sh)  # 二重起動しない(生存確認してから起動)


if __name__ == "__main__":
    unittest.main()
