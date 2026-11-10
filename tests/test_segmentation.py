"""話の区切り(続き待ちマージ)の回帰テスト。

11/05ライブ(logs/session-20261105-131717.txt)では、0.5秒のポーズで文の途中が
切り離され、断片のまま毎回翻訳に回っていた(「秘密で」「こちらが」「てるから」
「ちょっと」「to the」「like it's a,」「up.」「to get it.」…)。断片ごとの翻訳は
字幕を壊すうえ、1断片ごとに whisper+LLM+読み上げゲートを直列で潰す。
ends_open が「まだ途中」の文末を見分け、主ループが hold_open_s だけ続きを
待って merge_held で1文に戻してから訳す。ここでその判定表を固める。

実行: PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -v
"""
from __future__ import annotations

import os
import unittest
from pathlib import Path
from unittest.mock import patch

from tsuyaku.config import Settings, load_settings
from tsuyaku.guards import ends_open, join_text, merge_held
from tsuyaku.langdetect import norm


class TestEndsOpenJa(unittest.TestCase):
    def test_log_fragments_are_open(self):
        # 13:17ライブログで単独翻訳された日本語断片(そのまま訳すと破綻字幕になった)
        # 末尾が助詞の断片だけ拾える(「せっかく」「中心のゼロ」は語尾が助詞で
        # ないので見逃す — 文末が助詞でない断片は次が来ても訳せるので許容)
        for frag in ["秘密で", "こちらが", "てるから", "ちょっと",
                     "えっと", "たまりに"]:
            self.assertTrue(ends_open(frag, "ja"), frag)

    def test_complete_ja_endings_are_closed(self):
        # これらは「文が終わっている」。待ち時間だけ遅らせるので誤検出は損
        for frag in ["以上", "できた。", "来ていた", "でしょうか", "すごい",
                     "いいよ", "です", "ます", "了解", "そうです。", "英語。"]:
            self.assertFalse(ends_open(frag, "ja"), frag)

    def test_comma_is_always_open(self):
        self.assertTrue(ends_open("そして、", "ja"))
        self.assertTrue(ends_open("えっと、", "ja"))
        self.assertTrue(ends_open("Well,", "en"))


class TestEndsOpenOtherLangs(unittest.TestCase):
    def test_en(self):
        for frag in ["to the", "like it's a", "up", "to get it", "we need to"]:
            self.assertTrue(ends_open(frag, "en"), frag)
        for frag in ["Thank you.", "Okay.", "Yeah.", "It's okay.", "It's up."]:
            self.assertFalse(ends_open(frag, "en"), frag)

    def test_ko_particles(self):
        for frag in ["학교가", "이거를", "내일 우리에게"]:
            self.assertTrue(ends_open(frag, "ko"), frag)
        for frag in ["감사합니다", "네.", "괜찮습니다"]:
            self.assertFalse(ends_open(frag, "ko"), frag)

    def test_zh_particles(self):
        # 的/了/吗/呢/吧 は文末にも来るので開き側では使わない
        for frag in ["这个", "我觉得很", "他还能"]:
            self.assertTrue(ends_open(frag, "zh"), frag)
        for frag in ["好了。", "是的", "谢谢"]:
            self.assertFalse(ends_open(frag, "zh"), frag)

    def test_empty_and_unknown(self):
        self.assertFalse(ends_open("", "ja"))
        self.assertFalse(ends_open("   ", "ja"))
        self.assertFalse(ends_open("anything", "fr"))  # 対象外言語は待たない


class TestJoinAndMerge(unittest.TestCase):
    def test_join_spaces_only_for_en(self):
        self.assertEqual(join_text("ちょっと", "まって", "ja"), "ちょっとまって")
        self.assertEqual(join_text("to the", "next slide", "en"), "to the next slide")
        self.assertEqual(join_text("", "これ", "ja"), "これ")
        self.assertEqual(join_text("これ", "", "ja"), "これ")

    def test_merge_same_lang_rechecks_language(self):
        # ライブログ相当: 「こちらが」+「資料です。」が1文に戻る
        t, lang, n = merge_held(("こちらが", "ja", 0.0, 0), "資料です。", "ja")
        self.assertEqual(t, "こちらが資料です。")
        self.assertEqual(lang, "ja")
        self.assertEqual(n, 1)
        # 英語断片の連結
        t, lang, n = merge_held(("to the", "en", 0.0, 0), "next slide.", "en")
        self.assertEqual(t, "to the next slide.")
        self.assertEqual(lang, "en")

    def test_merge_language_switch_drops_held(self):
        # 言語が違う = 断片側がwhisperのノイズか話題切替。断片を捨てて新規を訳す
        t, lang, n = merge_held(("こちらが", "ja", 0.0, 0), "The file is here.", "en")
        self.assertEqual(t, "The file is here.")
        self.assertEqual(lang, "en")
        self.assertEqual(n, 0)

    def test_merge_unknown_lang_keeps_detection_of_other_side(self):
        t, lang, n = merge_held(("학교가", "ko", 0.0, 0), "내일 만나요", "")
        self.assertEqual(lang, "ko")
        self.assertEqual(n, 1)


class TestHoldSettings(unittest.TestCase):
    def test_defaults(self):
        s = Settings()
        self.assertGreater(s.hold_open_s, 0.5)   # 文中ポーズ(0.5s)より長い
        self.assertLess(s.hold_open_s, 2.0)      # 会話を止めるほどは待たない
        self.assertEqual(s.hold_open_max, 20)
        self.assertEqual(s.hold_max_parts, 2)

    def test_env_override(self):
        with patch.dict(os.environ, {"TSUYAKU_HOLD_S": "0"}):
            self.assertEqual(load_settings().hold_open_s, 0.0)  # 0で無効化できる


class TestWiring(unittest.TestCase):
    """主ループが実際に繋がっているか(呼び出し忘れ・削除戻し防止)。"""

    def setUp(self):
        self.src = (Path(__file__).resolve().parents[1]
                    / "src/tsuyaku/__main__.py").read_text()

    # 11/05夜: 続き待ちは文字の段階(_text_worker の held)から、音声の段階の
    # Endpointer(_stt_worker)へ移した。0.5秒の無音で2人分が1区間になる問題は
    # 文字の段階では直せない(tests/test_endpointer.py, test_stream_segmenter.py)。
    def test_stt_worker_uses_endpointer(self):
        i = self.src.index("def _stt_worker(")
        body = self.src[i:self.src.index("def _text_worker(")]
        self.assertIn("Endpointer(", body)
        self.assertIn("ep.push(seg, lo, hi, text, lang", body)
        self.assertIn("ep.tick(time.time(), speaking_since=", body)
        # 聞き直しは リモートサーバーの耳(細かい区切り)のときだけ
        self.assertIn('getattr(self, "fine_seg", False)', body)

    def test_backlog_abandons_pending_fragment(self):
        i = self.src.index("def _stt_worker(")
        body = self.src[i:self.src.index("def _text_worker(")]
        self.assertIn("ep.flush()                    # 詰まっている", body)

    def test_text_worker_no_longer_holds(self):
        i = self.src.index("def _text_worker(")
        body = self.src[i:self.src.index("def _tts_worker(")]
        self.assertNotIn("held", body)

    def test_utterance_method_defined(self):
        self.assertIn("def _utterance(", self.src)
        self.assertIn("self._utterance(base, model, pair_t, topic, text, lang, t0,",
                      self.src)


if __name__ == "__main__":
    unittest.main()
