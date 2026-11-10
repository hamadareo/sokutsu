"""言語判定を今日の言語ペアに絞る(11/05夜)。細かく区切ると短い区間の言語判定が
外れやすい(合成会話: 0.9秒の「水深は」→ 英語と判定され「Suy-shin-la.」)。
リモートサーバーの耳は langs=ja,en を受け取り、短い区間はペアの中から、長い区間でも
ペア外が確信度高く勝たない限りペアの中から言語を選んで書き起こす。
"""
import importlib.util
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("stt_server", ROOT / "server/remote_stt_server.py")
srv = importlib.util.module_from_spec(spec)
spec.loader.exec_module(srv)


class TestChooseLang(unittest.TestCase):
    P = [("en", 0.55), ("ja", 0.30), ("ko", 0.10), ("zh", 0.05)]

    def test_short_segment_restricted_to_pair(self):
        self.assertEqual(srv.choose_lang([("en", 0.6), ("ja", 0.3)], ["ja", "ko"], 0.9), "ja")

    def test_pair_member_wins_when_top(self):
        self.assertEqual(srv.choose_lang(self.P, ["ja", "en"], 3.0), "en")

    def test_confident_third_language_kept_when_long(self):
        # 本当に第三言語を話している(長く・確信度高い)ならそのまま(字幕のみで出す)
        self.assertEqual(srv.choose_lang([("ko", 0.92), ("ja", 0.05)], ["ja", "en"], 3.0), "ko")

    def test_unconfident_third_language_falls_back_to_pair(self):
        self.assertEqual(srv.choose_lang([("ko", 0.5), ("ja", 0.3)], ["ja", "en"], 3.0), "ja")

    def test_no_langs_means_no_restriction(self):
        self.assertIsNone(srv.choose_lang(self.P, [], 1.0))


class TestClientSendsPair(unittest.TestCase):
    def test_remote_stt_posts_langs(self):
        from tsuyaku.stt import RemoteSTT
        seen = {}

        class R:
            def raise_for_status(self):
                pass

            def json(self):
                return {"text": "水深は", "lang": "ja", "prob": 0.9}

        def post(url, data=None, timeout=None):
            seen["url"] = url
            return R()

        r = RemoteSTT("http://x", local=None, post=post)
        r.langs = ("ja", "en")
        r.transcribe(np.zeros(16000, np.float32))
        self.assertTrue(seen["url"].endswith("/transcribe?langs=ja,en"), seen["url"])


if __name__ == "__main__":
    unittest.main()
