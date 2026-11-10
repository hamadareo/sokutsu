"""訳が1語だけになる「履歴汚染ループ」の防御(2026-11-05 ライブ不具合)。

実測: 13:37〜13:47 のログで ja→en 訳が "The"/"This"/"I" など1〜3トークンで
途切れ続けた。同じ発話をクリーンな履歴で叩くと完全な訳が返り、直近4ターンに
短い破綻訳を混ぜると gate(qwen3.8-flash-next) は 'I' の1語だけで返すことを
再現した。→ 破綻訳を履歴に積まない/履歴から除く/一度だけ空履歴でやり直す。

実行: PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -v
"""
from __future__ import annotations

import unittest
from pathlib import Path

from tsuyaku.guards import looks_truncated
from tsuyaku.translate import build_messages

ROOT = Path(__file__).resolve().parent.parent
MAIN = (ROOT / "src/tsuyaku/__main__.py").read_text()


class TestLooksTruncated(unittest.TestCase):
    def test_long_input_one_word_output(self):
        # 実ログそのもの: 0.5の発話 → 'I' だけ
        self.assertTrue(looks_truncated(
            "0.5という数字が 適していると思うので 0.5", "I"))
        self.assertTrue(looks_truncated("綺麗とは言えない", "It"))

    def test_short_input_short_output_is_fine(self):
        # 短い発話への短い訳は正当(はい→Yes / Okay→分かり)
        self.assertFalse(looks_truncated("Okay.", "分かり"))
        self.assertFalse(looks_truncated("はい", "Yes"))
        self.assertFalse(looks_truncated("秘密で", "It's confidential."))

    def test_full_translation_not_flagged(self):
        self.assertFalse(looks_truncated(
            "それそれそれそれそれおいおいおい それだよ",
            "Yes, yes, yes, yes, yes! Hey, hey, hey! That's it!"))

    def test_empty_not_this_guard(self):
        # 空訳は別のガードが処理する(二本立てにしない)
        self.assertFalse(looks_truncated("何か長い発話があります", ""))


class TestHistoryHygiene(unittest.TestCase):
    def test_build_messages_skips_poisoned_pair(self):
        hist = [
            {"lang": "ja", "orig": "その際にしたんですけど", "trans": "I did that at the"},
            {"lang": "ja", "orig": "来たのはいつですか", "trans": "When did it arrive"},
            {"lang": "ja", "orig": "綺麗とは言えない", "trans": "It"},  # 破綻訳
        ]
        msgs = build_messages(("ja", "en"), "", hist, "今度の発話です", "ja")
        joined = "\n".join(m["content"] for m in msgs)
        # 破綻ターンは履歴ブロックから丸ごと除かれ、対になる原文も一緒に消える
        self.assertNotIn("綺麗とは言えない", joined)
        self.assertNotIn("It", joined)
        self.assertIn("When did it arrive", joined)
        self.assertTrue(msgs[-1]["content"].endswith("今度の発話です"))

    def test_history_is_not_fed_as_assistant_turns(self):
        # 11/05実測: マルチターンの assistant ターンに破綻訳が1個でも
        # 混ざるとモデルが真似して短打ちになる('Particles such' 単独でも
        # 汚染、2個だと 'I' の1語に)。閾値ではイタチゴッコなので、履歴を
        # 「参考」と明記した1本のユーザーメッセージに畳んで構造的に免疫化する。
        hist = [{"lang": "ja", "orig": "粒子に土の粒だったりが", "trans": "Particles such"},
                {"lang": "en", "orig": "Okay.", "trans": "分かり"}]
        msgs = build_messages(("ja", "en"), "", hist, "今回の発話です", "ja")
        self.assertNotIn("assistant", [m["role"] for m in msgs])
        joined = "\n".join(m["content"] for m in msgs)
        self.assertIn("context only, do not translate", joined)  # 履歴は「翻訳しない参考」だと分かる形
        self.assertTrue(msgs[-1]["content"].endswith("今回の発話です"))

    def test_system_tells_model_to_finish_sentences(self):
        sys = build_messages(("ja", "en"), "", [], "途中で", "ja")[0]["content"]
        # 11/08: 「Always finish the sentence」は断片に続きを創作させたので、途中で止めない指示に置き換えた
        self.assertIn("do not stop partway", sys)


class TestRunFlow(unittest.TestCase):
    def test_run_retries_once_on_truncated_output(self):
        i = MAIN.index("def _run")
        seg = MAIN[i:]
        self.assertIn("looks_truncated", seg)
        # 再試行経路が2箇所以上あること。11/05午後の再構成で空訳リトライは
        # translate_with_retry ラッパへ移行したので、素呼び出しとラッパを
        # 合わせて数える(ラッパの中身は translate_stream 1発+1リトライ)。
        self.assertGreaterEqual(
            seg.count("translate_stream(") + seg.count("translate_with_retry("), 2)

    def test_run_never_appends_failed_translations_to_history(self):
        # 履歴appendは成功経路の1箇所だけ(空訳の原文フォールバックは汚染になる)
        i = MAIN.index("def _run")
        self.assertEqual(MAIN[i:].count("self.history.append"), 1)


if __name__ == "__main__":
    unittest.main()
