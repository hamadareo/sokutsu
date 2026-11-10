"""1秒未満の区間(11/06 FLEURS 多言語評価)。

単独で聞かせると幻覚(「ご視聴ありがとうございました」「Cool.」「you」)や
「。」付きの断片(「设施。」「相似。」)になった。junk 13件はすべて 0.37〜0.85秒。
短い区間は続きを待って音声ごと聞き直し、低確率で続きの無いものは捨てる。
"""
import unittest

import numpy as np

from tsuyaku.guards import hallucinated_out_of_pair
from tsuyaku.segmenter import Endpointer

SR = 16000


def seg(sec):
    return np.zeros(int(sec * SR), np.float32)


class TestTiny(unittest.TestCase):
    def mk(self):
        calls = []

        def rejoin(a):
            calls.append(len(a) / SR)
            return ("大多数现代科研望远镜都是巨型设施，位于大气条件优良的地方。", "zh", 0.9)
        return Endpointer(rejoin=rejoin), calls

    def test_tiny_done_fragment_waits_and_rejoins(self):
        ep, calls = self.mk()
        out = ep.push(seg(0.5), 10.0, 10.5, "设施。", "zh", now=100.0)
        self.assertEqual(out, [])                       # 「。」付きでもすぐには出さない
        out = ep.push(seg(2.0), 10.8, 12.8, "位于大气条件优良的地方。", "zh", now=101.0)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0].parts, 2)
        self.assertTrue(calls)                          # 音声ごと聞き直した

    def test_low_prob_tiny_joins_across_language(self):
        # zh 音声の頭 0.4秒が en「Cool.」(低確率)になった → 言語が違っても繋ぐ
        ep, _ = self.mk()
        ep.push(seg(0.4), 10.0, 10.4, "Cool.", "en", now=100.0, prob=0.3)
        out = ep.push(seg(2.0), 10.6, 12.6, "位于大气条件优良的地方。", "zh", now=101.0)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0].lang, "zh")

    def test_isolated_low_prob_tiny_is_dropped(self):
        ep, _ = self.mk()
        ep.push(seg(0.4), 10.0, 10.4, "you", "en", now=100.0, prob=0.3)
        self.assertEqual(ep.tick(102.0), [])
        self.assertIsNone(ep._p)

    def test_isolated_confident_tiny_is_kept(self):
        ep, _ = self.mk()
        ep.push(seg(0.6), 10.0, 10.6, "はい。", "ja", now=100.0, prob=0.9)
        out = ep.tick(100.0 + ep.hold_tiny_s + 0.01)
        self.assertEqual([u.text for u in out], ["はい。"])

    def test_confident_tiny_keeps_language_boundary(self):
        # 話者交代(別言語)を高確率の短い返事で跨がない
        ep, _ = self.mk()
        ep.push(seg(0.6), 10.0, 10.6, "Yes.", "en", now=100.0, prob=0.9)
        out = ep.push(seg(2.0), 10.9, 12.9, "その結果は来週の会議で共有します。", "ja", now=101.0)
        self.assertEqual([u.text for u in out], ["Yes.", "その結果は来週の会議で共有します。"])

    def test_no_rejoin_means_no_tiny_rule(self):
        ep = Endpointer(rejoin=None)
        out = ep.push(seg(0.5), 10.0, 10.5, "その結果は来週共有します。", "ja", now=100.0)
        self.assertEqual(len(out), 1)


class TestStockPhrases(unittest.TestCase):
    def test_youtube_outros_dropped(self):
        for t in ["ご視聴ありがとうございました", "最後にご視聴ありがとうございました",
                  "Thanks for watching!", "请不吝点赞 订阅 转发"]:
            self.assertTrue(hallucinated_out_of_pair(("ja", "zh"), "ja", t, prob=0.9), t)

    def test_conference_closing_kept(self):
        self.assertFalse(hallucinated_out_of_pair(("ja", "en"), "ja", "ご清聴ありがとうございました", prob=0.9))


if __name__ == "__main__":
    unittest.main()
