"""このMacの速い耳 SenseVoice(11/06「MacBook処理でも即答できるようにしてください」)。"""
import time
import unittest
from pathlib import Path

import numpy as np

from tsuyaku.config import load_settings

S = load_settings()
HAVE = bool(S.sense_voice_dir) and Path(S.sense_voice_dir).exists()


@unittest.skipUnless(HAVE, "SenseVoice のモデルが無い(scripts/get_models.sh)")
class TestSenseVoice(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from tsuyaku.stt import SenseVoiceSTT
        cls.stt = SenseVoiceSTT(S.sense_voice_dir)

    def wav(self, name):
        import wave
        with wave.open(str(Path(S.sense_voice_dir) / "test_wavs" / name)) as w:
            a = np.frombuffer(w.readframes(w.getnframes()), np.int16)
        return a

    def test_japanese_fast_with_punctuation(self):
        self.stt.langs = ("ja", "en")
        t0 = time.time()
        text, lang, prob = self.stt.transcribe(self.wav("ja.wav"))
        self.assertLess(time.time() - t0, 1.0)
        self.assertEqual(lang, "ja")
        self.assertTrue(text.endswith("。"))          # Endpointer が文末を見られる
        self.assertEqual(prob, 1.0)

    def test_noise_gives_nothing(self):
        # 雑音には毎回「그.」を返す → 1文字以下は捨てる(韓国語ペアでも)
        self.stt.langs = ("ja", "ko")
        rng = np.random.default_rng(0)
        text, lang, prob = self.stt.transcribe((rng.standard_normal(32000) * 0.02).astype(np.float32))
        self.assertEqual((text, lang), ("", ""))

    def test_out_of_pair_is_flagged(self):
        from tsuyaku.guards import hallucinated_out_of_pair
        self.stt.langs = ("ja", "en")
        text, lang, prob = self.stt.transcribe(self.wav("ko.wav"))
        self.assertEqual(lang, "ko")
        self.assertEqual(prob, 0.0)
        self.assertTrue(hallucinated_out_of_pair(("ja", "en"), lang, text, prob=prob))


class TestWiring(unittest.TestCase):
    def test_mac_ear_default_and_fine_segmentation(self):
        self.assertEqual(S.mac_ear, "sensevoice")
        src = (Path(__file__).resolve().parents[1] / "src/tsuyaku/__main__.py").read_text()
        self.assertIn("RemoteSTT(stt_url, make_local_stt(s), hedge_s=s.stt_hedge_s,", src)
        self.assertIn("self.fine_seg = stt_url is not None or local_fast", src)

    def test_falls_back_to_whisper_when_missing(self):
        from unittest.mock import patch
        from tsuyaku import stt as m

        class Cfg:
            mac_ear = "sensevoice"
            sense_voice_dir = "/nonexistent"
            whisper_model = "w.bin"
        with patch.object(m, "STT", lambda p: ("whisper", p)):
            self.assertEqual(m.make_local_stt(Cfg()), ("whisper", "w.bin"))



class TestCjkSpaces(unittest.TestCase):
    """11/06 実会話: このMacの耳(SenseVoice)が日本語の語の間に空白を入れていた
    (「パーっていろいろ あり ます から ね。」)。字幕が読みにくく、訳の手がかりも崩れる。"""

    def test_spaces_between_japanese_removed(self):
        from tsuyaku.stt import join_cjk_spaces
        self.assertEqual(join_cjk_spaces("パーっていろいろ あり ます から ね。"),
                         "パーっていろいろありますからね。")
        self.assertEqual(join_cjk_spaces("脳のパーツっていろんな脳があります からね。"),
                         "脳のパーツっていろんな脳がありますからね。")
        self.assertEqual(join_cjk_spaces("我 们 走 吧"), "我们走吧")

    def test_latin_and_korean_spaces_kept(self):
        from tsuyaku.stt import join_cjk_spaces
        self.assertEqual(join_cjk_spaces("IoT AI について"), "IoT AI について")
        self.assertEqual(join_cjk_spaces("안녕하세요 반갑습니다"), "안녕하세요 반갑습니다")
        self.assertEqual(join_cjk_spaces("That does sound interesting."), "That does sound interesting.")

if __name__ == "__main__":
    unittest.main()
