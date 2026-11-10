"""訳の言語の取り違え(11/06 FLEURS 多言語評価)。

日本語の指示文だと zh→ja で中国語をそのまま返し(16/41)、en→zh で日本語を出した
(7/39)。指示文を英語にし、残りは wrong_language で拾って空履歴でやり直す。
"""
import unittest
from pathlib import Path

from tsuyaku.guards import wrong_language
from tsuyaku.translate import build_messages

ROOT = Path(__file__).resolve().parents[1]


class TestWrongLanguage(unittest.TestCase):
    def test_copy_of_source(self):
        self.assertTrue(wrong_language("空中铁匠乐队已经取消了剩余场", "空中铁匠乐队已经取消了剩余场。", "ja"))

    def test_other_language(self):
        self.assertTrue(wrong_language("It is one of the main attractions of South Africa.",
                                       "これは南アフリカの主要な観光名所の一つです。", "zh"))
        self.assertTrue(wrong_language("これは南アフリカです", "这是南非的主要景点之一。", "ja"))
        self.assertTrue(wrong_language("東京大学に行きます", "東京大学に行きます", "en"))
        self.assertTrue(wrong_language("안녕하세요 만나서 반갑습니다", "こんにちは、お会いできて嬉しいです", "ko"))

    def test_valid_translations_pass(self):
        self.assertFalse(wrong_language("谢谢", "ありがとう", "ja"))
        self.assertFalse(wrong_language("Hello my name is Reo", "こんにちは、私の名前はレオです。", "ja"))
        self.assertFalse(wrong_language("東京大学", "東京大学", "ja"))       # 短い固有名詞
        self.assertFalse(wrong_language("YOLOを使いました", "We used YOLO.", "en"))
        self.assertFalse(wrong_language("こんにちは", "안녕하세요", "ko"))
        self.assertFalse(wrong_language("今日は東京大学で発表します", "今天在东京大学发表。", "zh"))


class TestHistoryHygiene(unittest.TestCase):
    def test_copied_translations_are_not_used_as_context(self):
        hist = [{"lang": "zh", "orig": "空中铁匠乐队已经取消了剩余场", "trans": "空中铁匠乐队已经取消了剩余场。"},
                {"lang": "ja", "orig": "そうなんですね", "trans": "原来如此"}]
        msgs = build_messages(("ja", "zh"), "", hist, "我们下次再讨论吧", "zh")
        body = msgs[-1]["content"]
        self.assertNotIn("剩余场", body)
        self.assertIn("原来如此", body)
        self.assertIn("(Simplified Chinese -> Japanese)", body)

    def test_app_retries_on_wrong_language(self):
        src = (ROOT / "src/tsuyaku/__main__.py").read_text()
        self.assertIn("if trans and wrong_language(text, trans, tgt):", src)


if __name__ == "__main__":
    unittest.main()
