"""耳が断片の終わりに付けた「。」で、文の途中を確定していた(11/08 品質の切り分け A)。

見つけた問題(作った対面の対話 9 本、このMac = SenseVoice): 発話 126 のうち 61 が文の数より多く割れた。
SenseVoice(と whisper)は息つぎで切った断片にも「。」「？」を付け、Endpointer はそれを文末とみて
すぐ訳していた(「私は土曜に友達と。」「新しくできたラーメン屋に行ったんです。」)。英語には
「.」の直前が機能語なら途中とみる規則があったが、日本語・中国語・韓国語には無かった。
"""
import unittest

from tsuyaku.segmenter import classify


class TestParticleBeforePeriod(unittest.TestCase):
    def test_ja_particle_then_period_is_open(self):
        self.assertEqual(classify("私は土曜に友達と。", "ja"), "open")
        self.assertEqual(classify("新しいバージョンでキャッシュキーの形式が変わったので。", "ja"), "open")
        self.assertEqual(classify("結果は予想より良かったですけど。", "ja"), "open")

    def test_ja_real_sentence_end_stays_done(self):
        for t in ("新しくできたラーメン屋に行ったんです。", "そうですね。", "はい、二つです。", "なるほど。",
                  "金曜日までに教えてください。", "それ、いつまでですか？"):
            self.assertEqual(classify(t, "ja"), "done", t)

    def test_zh_and_ko(self):
        self.assertEqual(classify("我们周末和。", "zh"), "open")
        self.assertEqual(classify("原因找到了吗？", "zh"), "done")
        self.assertEqual(classify("친구와.", "ko"), "open")
        self.assertEqual(classify("벌써 보냈어요.", "ko"), "done")

    def test_en_unchanged(self):
        self.assertEqual(classify("We wanted to.", "en"), "open")
        self.assertEqual(classify("That explains it.", "en"), "done")


if __name__ == "__main__":
    unittest.main()


class TestEarPunctNotTrusted(unittest.TestCase):
    """SenseVoice は区切った断片の終わりに必ず句点を付ける(作った対話: このMac の単位 294 個すべてが
    句点で終わり、126 発話中 61 が文の数より多く割れた)。耳の句点を信じない設定では、句点を外して
    文末の言葉で決め、続きが来れば音声ごと繋いで聞き直す。"""

    def push(self, ep, text, lo, hi, now, lang="ja"):
        import numpy as np
        return ep.push(np.zeros(int((hi - lo) * 16000), np.float32), lo, hi, text, lang, now=now)

    def test_trailing_period_after_fragment_waits_and_joins(self):
        from tsuyaku.segmenter import Endpointer
        ep = Endpointer(rejoin=lambda a, span=None: ("新しくできたラーメン屋に行ったんですけど、1時間並びました。", "ja", 1.0),
                        trust_end_punct=False)
        self.assertEqual(self.push(ep, "新しくできたラーメン屋に行ったんです。", 0.0, 2.0, 10.0), [])
        self.assertIsNotNone(ep.pending())     # 「です」で終わっても、すぐ続きが始まれば繋ぐ(少しだけ待つ)
        out = self.push(ep, "けど、1時間並びました。", 2.25, 4.0, 10.3)
        t = 10.3
        while not out and t < 12:
            t += 0.1
            out = ep.tick(t)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0].parts, 2)
        self.assertIn("けど", out[0].text)

    def test_default_trusts_punctuation(self):
        from tsuyaku.segmenter import Endpointer
        ep = Endpointer(rejoin=None)
        out = self.push(ep, "新しくできたラーメン屋に行ったんですけど、たくさんの人が並んでいました。", 0.0, 4.0, 10.0)
        self.assertEqual(len(out), 1)

    def test_question_and_final_forms_still_close_quickly(self):
        from tsuyaku.segmenter import Endpointer
        ep = Endpointer(rejoin=None, trust_end_punct=False)
        out = self.push(ep, "原因はわかりましたか？", 0.0, 2.0, 10.0)
        t = 10.0
        while not out and t < 13:
            t += 0.1
            out = ep.tick(t)
        self.assertLessEqual(t - 10.0, 0.6)                    # 文末の形(ですか)なら長くは待たない
