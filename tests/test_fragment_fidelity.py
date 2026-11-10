"""短い断片に内容を足す・長い文の大半を落とす(11/08 このMac gemma-3-4b の日中の会話)。文はすべて作った例。"""
import unittest

from tsuyaku.guards import length_off
from tsuyaku.translate import build_system


class TestRule(unittest.TestCase):
    def test_system_says_do_not_add(self):
        s = build_system(("ja", "zh"))
        self.assertIn("Never add", s)
        self.assertIn("fragment", s)

    def test_system_constant_across_turns(self):
        self.assertEqual(build_system(("ja", "zh"), ""), build_system(("ja", "zh"), ""))


class TestLengthOff(unittest.TestCase):
    def test_made_up_continuation(self):
        # 条件の断片に続きを創作した長い訳
        self.assertTrue(length_off("如果不开灯的话。", "電気をつけないと部屋が暗くて何も見えないので、先にスイッチを探しましょう。", "zh", "ja"))

    def test_long_sentence_collapsed(self):
        self.assertTrue(length_off("明日の午後三時から会議室Bで新しいプロジェクトの予算について山田さんと打ち合わせをします。",
                                   "好的。", "ja", "zh"))

    def test_normal_translations_ok(self):
        for src, tr, a, b in (("周六见。", "土曜日に会いましょう。", "zh", "ja"),
                              ("我饿了。", "お腹が空きました。", "zh", "ja"),
                              ("如果不开灯的话。", "電気をつけないなら。", "zh", "ja"),
                              ("明日の午後三時から会議室Bで打ち合わせをします。", "明天下午三点在B会议室开会。", "ja", "zh"),
                              ("いいですよ。", "好啊。", "ja", "zh"),
                              ("Thank you very much for coming today.", "今日は来てくれてありがとうございます。", "en", "ja")):
            self.assertFalse(length_off(src, tr, a, b), (src, tr))


class TestWiring(unittest.TestCase):
    def test_retry_on_length_off(self):
        from pathlib import Path
        src = (Path(__file__).resolve().parents[1] / "src/tsuyaku/__main__.py").read_text()
        self.assertIn("if trans and LENGTH_RETRY and length_off(text, trans, lang, tgt):", src)


if __name__ == "__main__":
    unittest.main()
