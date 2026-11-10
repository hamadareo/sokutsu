"""リモートサーバーの転送(SSH トンネル)が会話の途中で切れたら、裏で張り直して訳し直す(11/08 speed)。

見つけた問題(作った日↔中の会話を リモートサーバー で実時間に流した評価): 会話の途中で 127.0.0.1:12345 の転送が
切れ(Connection reset → refused)、以後の訳が全部「翻訳できませんでした」(25件の接続エラー)。
アプリは「会話を始める」の時にしか転送を張らない。翻訳の失敗では 0.4 秒待って1回やり直すだけで、
転送は戻らなかった。
"""
import threading
import time
import unittest
from unittest import mock

import requests

from tsuyaku import translate as tr
from tsuyaku import tunnel


class TestTunnelRepair(unittest.TestCase):
    def setUp(self):
        tunnel._reset()

    def test_connection_refused_to_remote_starts_one_repair(self):
        runs = []
        with mock.patch.object(tunnel, "_run_script", lambda: runs.append(1) or time.sleep(0.05)):
            ev1 = tunnel.note_failure("http://127.0.0.1:12345", requests.ConnectionError("refused"))
            ev2 = tunnel.note_failure("http://127.0.0.1:12345", requests.ConnectionError("refused"))
            self.assertIsNotNone(ev1)
            self.assertIs(ev1, ev2)          # 張り直しは同時に1回だけ
            self.assertTrue(ev1.wait(1.0))
        self.assertEqual(runs, [1])

    def test_other_servers_and_timeouts_are_ignored(self):
        with mock.patch.object(tunnel, "_run_script", side_effect=AssertionError):
            self.assertIsNone(tunnel.note_failure("http://127.0.0.1:1234", requests.ConnectionError()))
            self.assertIsNone(tunnel.note_failure("http://127.0.0.1:12345", requests.ReadTimeout()))

    def test_repairs_are_spaced(self):
        runs = []
        with mock.patch.object(tunnel, "_run_script", lambda: runs.append(1)):
            tunnel.note_failure("http://127.0.0.1:12345", requests.ConnectionError()).wait(1)
            self.assertIsNone(tunnel.note_failure("http://127.0.0.1:12345", requests.ConnectionError()))
        self.assertEqual(runs, [1])           # 直後にまた切れても MIN_GAP_S は張り直さない

    def test_translate_waits_for_repair_then_succeeds(self):
        state = {"up": False, "n": 0}

        def post(url, json=None, stream=False, timeout=None):
            state["n"] += 1
            if not state["up"]:
                raise requests.ConnectionError("refused")

            class R:
                status_code = 200

                def raise_for_status(self):
                    pass

                def iter_lines(self):
                    yield b'data: {"choices":[{"delta":{"content":"OK"}}]}'
                    yield b"data: [DONE]"
            return R()

        def fix():
            time.sleep(0.8)            # 転送を張り直すのに 0.4 秒より長くかかる
            state["up"] = True
        with mock.patch.object(tr.requests, "post", post), \
             mock.patch.object(tunnel, "_run_script", fix):
            out = tr.translate_with_retry("http://127.0.0.1:12345", "m", [{"role": "user", "content": "x"}])
        self.assertEqual(out, "OK")


if __name__ == "__main__":
    unittest.main()
