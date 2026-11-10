"""間の理解(11/10): 耳が付けた句点付きの助詞止まり(「平日の開店直後なら。」)は、続きを話している最中は待つ。
done の印を言語なしの判定で付けていたため、期限から 1 秒(speak_cap_done_s)で出し、続きと割れていた。"""
import unittest

import numpy as np

from tsuyaku.segmenter import Endpointer

SR = 16000


def tone(sec):
    return np.full(int(SR * sec), 0.1, dtype=np.float32)


class TestOpenWhileNextSpoken(unittest.TestCase):
    def test_open_particle_fragment_keeps_waiting_while_next_is_spoken(self):
        ep = Endpointer(rejoin=None)
        self.assertEqual(ep.push(tone(2.5), 0.0, 2.5, "平日の開店直後なら。", "ja", now=2.6), [])
        # 期限 3.8、続きは 2.6 から話している。期限 + 1.0 = 4.8 を超えても、open は speak_cap_s(3.0)まで待つ
        self.assertEqual(ep.tick(5.0, speaking_since=2.6), [])
        self.assertEqual(len(ep.tick(6.9, speaking_since=2.6)), 1)

    def test_open_fragment_joins_next(self):
        def rej(a, **kw):
            return "平日の開店直後ならそんなに混んでないらしいですよ。", "ja", 0.9

        ep = Endpointer(rejoin=rej)
        self.assertEqual(ep.push(tone(2.5), 0.0, 2.5, "平日の開店直後なら。", "ja", now=2.6), [])
        self.assertEqual(ep.tick(5.0, speaking_since=2.6), [])
        out = ep.push(tone(2.5), 2.5, 5.0, "そんなに混んでないらしいですよ。", "ja", now=5.1)
        self.assertEqual([u.parts for u in out], [2])

    def test_closed_fragment_still_uses_short_cap(self):
        ep = Endpointer(rejoin=None)
        self.assertEqual(ep.push(tone(0.8), 0.0, 0.8, "可以。", "zh", now=1.0), [])
        self.assertEqual(len(ep.tick(2.4, speaking_since=1.0)), 1)


if __name__ == "__main__":
    unittest.main()
