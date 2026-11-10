"""11/06 日中の実会話で出た問題。

1. 漢字だけの中国語(「谢谢」「最棒」「一陣子一陣子」)が文字判定で ja になり、
   ja→zh の素通し訳になった → whisper が音から決めた言語に従う。
2. 「ありがとうございます」→「谢谢」を破綻訳扱いして「翻訳できませんでした」。
3. 読み上げが「キーン」: Kokoro の出力に 2400Hz の倍音の鋭い山。
"""
import unittest
from pathlib import Path

import numpy as np

from tsuyaku.guards import looks_truncated
from tsuyaku.langdetect import resolve_lang
from tsuyaku.tts import remove_tones

ROOT = Path(__file__).resolve().parents[1]


class TestResolveLang(unittest.TestCase):
    def test_han_only_follows_whisper(self):
        for t in ["谢谢", "最棒", "一陣子一陣子", "因人禁咖", "今晚7.5.2"]:
            self.assertEqual(resolve_lang(t, "zh"), "zh", t)
        self.assertEqual(resolve_lang("以上", "ja"), "ja")

    def test_script_wins_when_clear(self):
        self.assertEqual(resolve_lang("私は日本人です", "zh"), "ja")
        self.assertEqual(resolve_lang("hello there", "zh"), "en")
        self.assertEqual(resolve_lang("감사합니다", "ja"), "ko")
        self.assertEqual(resolve_lang("勉強", ""), "ja")   # 耳の情報が無いときは従来どおり
        # 11/08: 簡体字にしか無い字(谢)があれば、耳の情報が無くても中国語(test_lang_fix)
        self.assertEqual(resolve_lang("谢谢", ""), "zh")

    def test_remote_stt_uses_it(self):
        src = (ROOT / "src/tsuyaku/stt.py").read_text()
        self.assertIn('resolve_lang(text, str(obj.get("lang") or ""))', src)


class TestShortChineseTranslation(unittest.TestCase):
    def test_short_han_translation_is_valid(self):
        self.assertFalse(looks_truncated("ありがとうございます", "谢谢"))
        self.assertFalse(looks_truncated("ありがとうございました。", "谢谢大家。"))

    def test_real_truncation_still_caught(self):
        long = "今回素晴らしいシステムを作ったので紹介させてください"
        self.assertTrue(looks_truncated(long, "I"))
        self.assertTrue(looks_truncated(long, "介绍"))


class TestRemoveTones(unittest.TestCase):
    def test_tones_removed_voice_kept(self):
        sr = 24000
        t = np.arange(sr * 2) / sr
        voice = 0.3 * np.sin(2 * np.pi * 220 * t) + 0.1 * np.sin(2 * np.pi * 3100 * t)
        whine = 0.01 * (np.sin(2 * np.pi * 4800 * t) + np.sin(2 * np.pi * 9600 * t))
        y = remove_tones((voice + whine).astype(np.float32), sr)
        Y = np.abs(np.fft.rfft(y)); f = np.fft.rfftfreq(len(y), 1 / sr)
        at = lambda hz: Y[np.argmin(np.abs(f - hz))]
        self.assertLess(at(4800), 1.0)
        self.assertLess(at(9600), 1.0)
        self.assertGreater(at(220), 1000)
        self.assertGreater(at(3100), 1000)
        self.assertLess(np.abs(y - voice).max(), 0.01)

    def test_applied_to_kokoro_output(self):
        src = (ROOT / "src/tsuyaku/tts.py").read_text()
        self.assertIn("samples = remove_tones(audio.samples, audio.sample_rate)", src)


class TestServerSimplifiedPrompt(unittest.TestCase):
    def test_zh_prompt(self):
        src = (ROOT / "server/remote_stt_server.py").read_text()
        self.assertIn('LANG_PROMPT = {"zh": "以下是普通话的句子。"}', src)


if __name__ == "__main__":
    unittest.main()
