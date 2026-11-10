"""ペア外言語の幻覚ガード強化(2026-11-05 ライブ不具合「話していない中国語・韓国語が出る」)。

実ログ logs/session-20261105-131717.txt から:
- ペア外幻覚は全部 norm≤11: 「카우다.」「하다. 하다.」「뭐 할 때?」「물 벗고.」
  「감사합니다.」(5)「에이 또 이홍도 또」(7)「아니, 아까 안 하셨거든요.」(11)「你是我代言」(5)
- 「英語。」「以上」は whisper が日本語音声から出した漢字のみテキストなのに
  kana が無いというだけで zh に誤判定されていた。
→ min_len 5→12 と、中文の機能文字を含まない漢字のみテキストの ja 化。

実行: PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -v
"""
from __future__ import annotations

import unittest

from tsuyaku.guards import hallucinated_out_of_pair
from tsuyaku.langdetect import detect_lang, norm


class TestKanjiOnlyIsJapanese(unittest.TestCase):
    def test_kanji_only_japanese_words(self):
        # 実ログ: 13:21:59 zh 英語。 / 13:35:25 zh 以上
        self.assertEqual(detect_lang("英語"), "ja")
        self.assertEqual(detect_lang("以上"), "ja")
        self.assertEqual(detect_lang("英語。"), "ja")
        self.assertEqual(detect_lang("以上。"), "ja")
        # 漢字だけの日本語らしい断片
        self.assertEqual(detect_lang("実際"), "ja")
        self.assertEqual(detect_lang("勉強"), "ja")

    def test_real_chinese_still_zh(self):
        # 中文の機能文字(了/的/们/…)を含む本物の中国語は zh のまま
        self.assertEqual(detect_lang("我们提出了一个新的框架"), "zh")
        self.assertEqual(detect_lang("这个问题很有意思"), "zh")
        self.assertEqual(detect_lang("我觉得可以"), "zh")

    def test_hangul_keeps_priority(self):
        # ハングルは漢字判定より常に優先(既存の契約)
        self.assertEqual(detect_lang("モデルの 추론 について"), "ko")
        self.assertEqual(detect_lang("추적 모델을 제안합니다"), "ko")

    def test_other_langs_untouched(self):
        self.assertEqual(detect_lang("これは予測モデルの話です"), "ja")
        self.assertEqual(detect_lang("We propose a tracking framework"), "en")
        self.assertEqual(detect_lang("12345 !!!"), "")


class TestOutOfPairLengthGate(unittest.TestCase):
    def test_all_logged_phantoms_dropped(self):
        # ja-en ペアで実ログに出たペア外幻覚がすべて落ちることを1件ずつ確認
        phantoms = ["카우다", "하다. 하다.", "뭐 할 때?", "물 벗고.",
                    "감사합니다", "에이 또 이홍도 또",
                    "아니, 아까 안 하셨거든요", "你是我代言"]
        for t in phantoms:
            with self.subTest(text=t):
                self.assertLessEqual(len(norm(t)), 11)
                self.assertTrue(hallucinated_out_of_pair(("ja", "en"), "ko", t))
        self.assertTrue(hallucinated_out_of_pair(("ja", "en"), "zh", "你是我代言"))

    def test_genuine_third_language_passes(self):
        # 本物の第三言語の一文は通す(norm>11)
        self.assertFalse(hallucinated_out_of_pair(
            ("ja", "en"), "ko", "안녕하세요 여러분 오늘도よろしくお願いします"))
        self.assertFalse(hallucinated_out_of_pair(
            ("ja", "ko"), "en", "Let me quickly add one more point about that."))

    def test_in_pair_never_dropped(self):
        # ペア内の言語は長さに関係なく通す(幻覚ガードはペア外専用)
        self.assertFalse(hallucinated_out_of_pair(("ja", "en"), "ja", "以上"))
        self.assertFalse(hallucinated_out_of_pair(("ja", "en"), "en", "OK"))

    def test_empty_lang_passes(self):
        self.assertFalse(hallucinated_out_of_pair(("ja", "en"), "", "카우다"))


if __name__ == "__main__":
    unittest.main()
