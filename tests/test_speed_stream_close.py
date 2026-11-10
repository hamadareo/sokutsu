"""先読みの訳を途中で止めたら、サーバーへの接続をすぐ閉じる(11/08 speed)。

見つけた問題(このMac、LM Studio の gemma-3-4b は同時に1件しか訳さない parallel=1): 先読み(予想訳)を
途中で止めても接続を開けたままにしていたので、LM Studio は止めた訳を最後まで作り続け、次の確定訳の
初字がその分待たされた。長い訳を 0.3 秒で止めた直後の短い訳の初字: 開けたまま 1.46・1.50 秒、
閉じる 0.70・0.71 秒(閉じると LM Studio が作るのをやめる)。作った日↔英の会話をこのMacで流すと、
訳の初字が 2〜5 秒の文が続いた(途中字幕の予想訳が次々に止められ、止めた訳が GPU を使い続けた)。
"""
import unittest
from unittest import mock

from tsuyaku import translate as tr


class _Resp:
    status_code = 200

    def __init__(self):
        self.closed = False

    def raise_for_status(self):
        pass

    def iter_lines(self):
        for k in range(5):
            yield f'data: {{"choices":[{{"delta":{{"content":"w{k} "}}}}]}}'.encode()
        yield b"data: [DONE]"

    def close(self):
        self.closed = True


class Stop(Exception):
    pass


class TestCloseOnCancel(unittest.TestCase):
    def test_response_is_closed_when_on_delta_raises(self):
        r = _Resp()

        def od(d):
            if d.startswith("w1"):
                raise Stop()
        with mock.patch.object(tr.requests, "post", lambda *a, **k: r):
            with self.assertRaises(Stop):
                tr.translate_stream("http://x", "m", [], od)
        self.assertTrue(r.closed)

    def test_response_is_closed_after_normal_end(self):
        r = _Resp()
        with mock.patch.object(tr.requests, "post", lambda *a, **k: r):
            self.assertEqual(tr.translate_stream("http://x", "m", []), "w0 w1 w2 w3 w4")
        self.assertTrue(r.closed)


if __name__ == "__main__":
    unittest.main()
