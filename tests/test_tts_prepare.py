"""読み上げの音を先に作っておく(TTS.prepare、11/06)。

合成は鳴り始めまで0.25〜1.0秒かかる。予想訳ができた時点(先読み翻訳)や前の
読み上げ中に作っておけば、確定した瞬間に鳴らせる。
"""
import threading
import time
import unittest
from unittest.mock import patch

from tsuyaku.tts import TTS


def tts():
    return TTS("", "", {}, voices={"ja": "voicevox:13", "ko": "say:Yuna"},
               speeds={"ja": 1.0}, voicevox_url="http://vv.local")


class TestPrepare(unittest.TestCase):
    def test_prepared_clip_is_played_without_resynth(self):
        calls = []

        def synth(self_, text, spk, speed):
            calls.append(text)
            time.sleep(0.05)
            return b"RIFF" + text.encode()

        t = tts()
        with patch.object(TTS, "_voicevox_synth", synth), \
             patch.object(TTS, "_play_wav", return_value=True) as pw:
            t.prepare("こんにちは。", "ja")
            time.sleep(0.1)
            t0 = time.time()
            self.assertTrue(t.say("こんにちは。", "ja"))
            self.assertLess(time.time() - t0, 0.03)       # 作り置きをすぐ鳴らす
        self.assertEqual(calls, ["こんにちは。"])
        pw.assert_called_once_with("RIFFこんにちは。".encode())
        self.assertEqual(t.prepared_hits, 1)

    def test_say_waits_for_clip_in_progress(self):
        calls = []

        def synth(self_, text, spk, speed):
            calls.append(text)
            time.sleep(0.1)
            return b"RIFF"

        t = tts()
        with patch.object(TTS, "_voicevox_synth", synth), \
             patch.object(TTS, "_play_wav", return_value=True):
            t.prepare("はい。", "ja")
            t.say("はい。", "ja")
        self.assertEqual(calls, ["はい。"])               # 2度作らない

    def test_different_text_is_synthesized_fresh(self):
        calls = []
        t = tts()
        with patch.object(TTS, "_voicevox_synth", lambda s_, x, a, b: calls.append(x) or b"RIFF"), \
             patch.object(TTS, "_play_wav", return_value=True):
            t.prepare("予想の訳。", "ja")
            time.sleep(0.05)
            t.say("確定した訳。", "ja")
        self.assertEqual(calls, ["予想の訳。", "確定した訳。"])

    def test_voice_change_invalidates(self):
        calls = []
        t = tts()
        with patch.object(TTS, "_voicevox_synth", lambda s_, x, a, b: calls.append(a) or b"RIFF"), \
             patch.object(TTS, "_play_wav", return_value=True):
            t.prepare("はい。", "ja")
            time.sleep(0.05)
            t.set_voice("ja", "voicevox:3")
            t.say("はい。", "ja")
        self.assertEqual(calls, [13, 3])

    def test_say_voice_is_not_prepared(self):
        t = tts()
        with patch.object(TTS, "_synth") as sy:
            t.prepare("안녕하세요", "ko")
        sy.assert_not_called()

    def test_failed_prepare_falls_back_to_say(self):
        import requests
        t = tts()
        with patch("requests.post", side_effect=requests.ConnectionError), \
             patch.object(TTS, "_say_macos", return_value=True) as sm:
            t.prepare("こんにちは。", "ja")
            time.sleep(0.05)
            self.assertTrue(t.say("こんにちは。", "ja"))
        sm.assert_called_once()

    def test_cache_is_bounded(self):
        t = tts()
        with patch.object(TTS, "_voicevox_synth", lambda *a: b"RIFF"):
            for i in range(20):
                t.prepare(f"文{i}。", "ja")
            self.assertLessEqual(len(t._clips), 8)   # 文ごとに作るので8(11/06)
            time.sleep(0.2)   # 作る担当(1本, 11/08)が作り終えるまで差し替えたまま(次の試験に響かせない)

    def test_synth_is_serialized(self):
        active, peak = [0], [0]
        lock = threading.Lock()

        def synth(self_, text, spk, speed):
            with lock:
                active[0] += 1
                peak[0] = max(peak[0], active[0])
            time.sleep(0.03)
            with lock:
                active[0] -= 1
            return b"RIFF"

        t = tts()
        with patch.object(TTS, "_voicevox_synth", synth):
            for i in range(4):
                t.prepare(f"文{i}。", "ja")
            time.sleep(0.3)
        self.assertEqual(peak[0], 1)


class TestAppPrepares(unittest.TestCase):
    def test_app_prepares_audio(self):
        from pathlib import Path
        src = (Path(__file__).resolve().parents[1] / "src/tsuyaku/__main__.py").read_text()
        self.assertIn("self.tts.prepare(trans, spoken, sp)", src)  # 確定訳: 前の読み上げ中に作る(11/08 速さつき)
        self.assertIn("ready=", src)                               # 予想訳: できた時点で作る


if __name__ == "__main__":
    unittest.main()
