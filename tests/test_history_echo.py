"""訳が「原文 => 訳」の形で返る(履歴の並べ方を真似る)のを直す(11/09)。

見つけた問題: 訳の依頼の履歴は「原文 => 訳」を並べる。gemma-3-4b は今の発話の訳もその形で返すことがある
(qualA の 126 発話で 3、engloop の訳で 5)。字幕・記録・読み上げに原文が二重に出ていた。
"""
import unittest

from tsuyaku.guards import strip_history_echo


class TestStrip(unittest.TestCase):
    def test_source_prefix_is_removed(self):
        self.assertEqual(
            strip_history_echo("娘は砂遊びが大好きなので、それならきっと喜びます。",
                               "娘は砂遊びが大好きなので、それならきっと喜びます。 => 딸이 모래 놀이를 정말 좋아해서, 그렇다면 분명 기뻐할 거예요."),
            "딸이 모래 놀이를 정말 좋아해서, 그렇다면 분명 기뻐할 거예요.")

    def test_normal_translation_is_kept(self):
        self.assertEqual(strip_history_echo("AからBへ。", "From A => B."), "From A => B.")
        self.assertEqual(strip_history_echo("ありがとう。", "Thank you."), "Thank you.")

    def test_empty_right_side_is_kept(self):
        self.assertEqual(strip_history_echo("はい。", "はい。 =>"), "はい。 =>")


class TestWiring(unittest.TestCase):
    def test_utterance_strips_echo(self):
        from pathlib import Path
        src = (Path(__file__).resolve().parents[1] / "src/tsuyaku/__main__.py").read_text()
        body = src[src.index("    def _utterance_body("):]
        self.assertIn("trans = strip_history_echo(text, trans)", body)


if __name__ == "__main__":
    unittest.main()
