"""会話を始めたら、翻訳の指示文を先にサーバーへ読ませておく(11/08 speed)。

翻訳サーバー(LM Studio・llama-server)は、前の依頼と前から一致する部分の読み込みを使い回す。
今までの warmup は "ping" だけを送るので、最初の1文は指示文の読み込みから始まった。会話中の
keepalive(8分ごと)の "ping" も、このMac(LM Studio は直前の1つだけ覚える)では読み込み済みの
指示文を追い出していた。
"""
import unittest
from unittest import mock

from tsuyaku import translate as tr


class TestWarmPrompt(unittest.TestCase):
    def test_warm_prompt_sends_the_same_prefix_as_the_first_utterance(self):
        sent = []
        with mock.patch.object(tr.requests, "post",
                               lambda url, json=None, timeout=None: sent.append(json)):
            tr.warm_prompt("http://x", "m", ("ja", "zh"), "テーマ", terms=("語",))
        self.assertEqual(len(sent), 1)
        body = sent[0]
        first = tr.build_messages(("ja", "zh"), "テーマ", [], "こんにちは", "ja", terms=("語",))
        self.assertEqual(body["messages"][0], first[0])                       # 指示文が同じ
        head = first[1]["content"].split("こんにちは")[0]
        self.assertTrue(body["messages"][1]["content"].startswith(head))      # 向きの行まで同じ
        self.assertEqual(body["max_tokens"], 1)
        self.assertFalse(body["stream"])

    def test_warm_prompt_never_raises(self):
        import requests
        with mock.patch.object(tr.requests, "post", side_effect=requests.ConnectionError):
            tr.warm_prompt("http://x", "m", ("ja", "en"))

    def test_app_warms_prompt_at_start_and_in_keepalive(self):
        from pathlib import Path
        src = (Path(__file__).resolve().parents[1] / "src/tsuyaku/__main__.py").read_text()
        body = src[src.index("    def _conversation("):src.index("    def _boot_cancelled(")]
        self.assertIn("warm_prompt", body)
        keep = src[src.index("    def _keepalive("):src.index("    def _run(")]
        self.assertIn("warm_prompt(", keep)


if __name__ == "__main__":
    unittest.main()
