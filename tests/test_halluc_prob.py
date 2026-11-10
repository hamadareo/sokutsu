"""whisper のトークン確率で幻覚を捨てる(長さだけの守卫では逃れた11/05実障害)。

11/05 15:46 ライブ: 話していない韓国語「이게 10분이죠 이게 동거예요」が字幕化。
norm=15 ≥ min_len(12) で長さ守卫を通過した。whisper は雑音・ぼやけた音声の
とき low-confidence なデコードになる(本物の発話なら確率が高い)。実測較正:
  実音声(M3, whisper large-v3-turbo, greedy): prob 0.87
  純白雑音の幻覚(「Thank you.」「you」「.」): prob 0.52〜0.76
よって 確率の床を下回る発話は幻覚として捨てる。床は「誤って本物を切る」より
「幻覚を1つ通す」ほうが会話の邪魔が大きい設計方針(第三言語の話者は卓では稀)で、
ペア外は厳しめ 0.80、ペア内は緩め 0.50。
"""
import types
import unittest
from pathlib import Path

from tsuyaku.guards import hallucinated_out_of_pair, seg_probability


def _seg(text, prob):
    s = types.SimpleNamespace()
    s.text = text
    s.probability = prob
    return s


class TestBackwardCompat(unittest.TestCase):
    def test_prob_none_keeps_old_length_behavior(self):
        # prob を渡さなければ従来と完全同型(既存呼び出し・既存テスト互換)。
        # 長い第三言語の一文は長さ守卫を通る(min_len=12)ので False。
        self.assertFalse(hallucinated_out_of_pair(
            ("ja", "en"), "ko", "안녕하세요 여러분 오늘도よろしくお願いします"))
        self.assertTrue(hallucinated_out_of_pair(("ja", "en"), "ko", "합니다"))
        self.assertFalse(hallucinated_out_of_pair(("ja", "en"), "ja", "こんにちは"))
        self.assertFalse(hallucinated_out_of_pair(("ja", "en"), "", "x"))

    def test_langless_still_passes_even_with_low_prob(self):
        # 言語判定不能は従来どおり通す(確率が低くても切らない)
        self.assertFalse(hallucinated_out_of_pair(("ja", "en"), "", "x", prob=0.2))


class TestOutOfPairProbGate(unittest.TestCase):
    def test_long_out_of_pair_low_prob_now_dropped(self):
        # 実障害そのもの: 長い韓国語幻覚を確率で落とす
        self.assertTrue(hallucinated_out_of_pair(
            ("ja", "en"), "ko", "이게 10분이죠 이게 동거예요", prob=0.60))
        self.assertTrue(hallucinated_out_of_pair(
            ("ja", "en"), "ko", "이게 10분이죠 이게 동거예요", prob=0.79))

    def test_confident_third_language_passes(self):
        # 本物の第三言語話者(確率が高い)は通す
        self.assertFalse(hallucinated_out_of_pair(
            ("ja", "en"), "ko", "안녕하세요 여러분 오늘도よろしくお願いします",
            prob=0.92))
        self.assertFalse(hallucinated_out_of_pair(
            ("ja", "en"), "zh", "这个方法的实现复杂度是线性的。", prob=0.88))


class TestInPairProbGate(unittest.TestCase):
    def test_low_prob_casual_speech_in_pair_is_kept(self):
        # 11/05 16:0x ライブ実ログ: ペア内の床0.50が本物の雑談を大量に捨てた。
        # 雑談の確率は 0.22〜0.48 で、純白雑音の幻覚(0.52〜0.76)より「低い」。
        # ペア内は確率で幻覚と本物を分けられない → ペア内に確率の床は置かない
        # (ペア内の雑音対策は rms ゲートが受け持つ)。
        for text, p in [("もっとさ、なんか楽にやった方がいいよ。", 0.48),
                        ("ここら辺に床行ってある 家の最寄りに入ったら 何", 0.46),
                        ("大丈夫だと一応最初に確認して シーアルを送ります", 0.24),
                        ("We'll check it out.", 0.43)]:
            lang = "en" if text.startswith("We") else "ja"
            self.assertFalse(hallucinated_out_of_pair(("ja", "en"), lang, text, prob=p),
                             text)

    def test_real_speech_in_pair_not_dropped(self):
        self.assertFalse(hallucinated_out_of_pair(("ja", "en"), "ja",
                                                  "こんにちは", prob=0.87))
        self.assertFalse(hallucinated_out_of_pair(("ja", "en"), "en", "Thanks.",
                                                  prob=0.55))


class TestSegProbability(unittest.TestCase):
    def test_empty_segments_means_trust_one(self):
        self.assertEqual(seg_probability([]), 1.0)

    def test_nan_probability_ignored(self):
        self.assertEqual(seg_probability([_seg("あ", float("nan"))]), 1.0)

    def test_hallucinated_segment_pulls_down_min(self):
        self.assertAlmostEqual(seg_probability([_seg("本物", 0.9), _seg("카우다", 0.4)]),
                               0.4)

    def test_all_nan_defaults_to_one(self):
        self.assertEqual(seg_probability([_seg("あ", float("nan")),
                                          _seg("い", float("nan"))]), 1.0)


class TestSTTExtractsProbability(unittest.TestCase):
    def test_transcribe_returns_three_tuple(self):
        # (text, lang, prob) に変わった。prob が無いと幻覚守卫が組めない
        s = (Path(__file__).resolve().parents[1] / "src" / "tsuyaku" / "stt.py").read_text()
        self.assertIn("extract_probability=True", s)
        self.assertIn("return text, lang, prob", s)


if __name__ == "__main__":
    unittest.main()
