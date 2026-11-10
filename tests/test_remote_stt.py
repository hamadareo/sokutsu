"""音声認識を リモートGPU に逃がす(11/05「翻訳が遅くてまともに使えない」対策)。

実測(同じ3.6〜4.0秒の発話): このMac(M3) whisper large-v3-turbo 2.3秒 →
remote faster-whisper 0.13〜0.26秒(往復込み)。リモートサーバーに届かない/失敗したら
その場でこのMacの whisper に落ちる(会話は止めない)。

実行: PYTHONPATH=src .venv/bin/python -m unittest discover -s tests
"""
import unittest
from pathlib import Path

import numpy as np

from tsuyaku.config import ENGINE_MODELS, Settings
from tsuyaku.stt import RemoteSTT

ROOT = Path(__file__).resolve().parents[1]


class FakeLocal:
    def __init__(self):
        self.calls = 0

    def transcribe(self, samples):
        self.calls += 1
        return "ローカルです", "ja", 0.9


class FakeResp:
    def __init__(self, obj, status=200):
        self._obj, self.status_code = obj, status

    def json(self):
        return self._obj

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


def _audio(sec=1.0):
    return (np.sin(np.arange(int(16000 * sec)) / 5) * 0.1).astype(np.float32)


class TestRemoteSTT(unittest.TestCase):
    def test_remote_result_uses_local_char_lang_detection(self):
        # サーバーの言語判定(雑音で en に振れる)ではなく、アプリ共通の文字種判定を使う
        sent = {}

        def post(url, data, timeout):
            sent["url"], sent["n"], sent["timeout"] = url, len(data), timeout
            return FakeResp({"text": "報酬設定を先に詰めるべきだと思います",
                             "lang": "en", "prob": 0.96, "sec": 0.08})
        loc = FakeLocal()
        stt = RemoteSTT("http://127.0.0.1:12346", loc, post=post)
        text, lang, prob = stt.transcribe(_audio())
        self.assertEqual((text, lang), ("報酬設定を先に詰めるべきだと思います", "ja"))
        self.assertAlmostEqual(prob, 0.96)
        self.assertEqual(loc.calls, 0)
        self.assertTrue(sent["url"].endswith("/transcribe"))
        self.assertEqual(sent["n"], 16000 * 2)  # 16bit PCM
        self.assertLessEqual(sent["timeout"], 5)  # 遅いリモートで会話を止めない
        self.assertEqual(stt.last_where, "remote")

    def test_int16_input_sent_as_is(self):
        got = {}

        def post(url, data, timeout):
            got["data"] = data
            return FakeResp({"text": "", "lang": "", "prob": 1.0})
        a = (np.arange(100, dtype=np.int16))
        RemoteSTT("http://x", FakeLocal(), post=post).transcribe(a)
        self.assertEqual(got["data"], a.astype("<i2").tobytes())

    def test_failure_falls_back_to_local_then_cools_down(self):
        n = {"post": 0}
        now = [1000.0]

        def post(url, data, timeout):
            n["post"] += 1
            raise ConnectionError("tunnel down")
        loc = FakeLocal()
        stt = RemoteSTT("http://x", loc, post=post, cooldown=30.0, clock=lambda: now[0])
        self.assertEqual(stt.transcribe(_audio())[0], "ローカルです")
        self.assertEqual(stt.last_where, "mac")
        # 落ちた直後は毎回リモートを待たない(タイムアウト待ちを積まない)
        stt.transcribe(_audio())
        self.assertEqual(n["post"], 1)
        self.assertEqual(loc.calls, 2)
        now[0] += 31  # 冷却後にもう一度リモートを試す
        stt.transcribe(_audio())
        self.assertEqual(n["post"], 2)

    def test_http_error_falls_back(self):
        loc = FakeLocal()
        stt = RemoteSTT("http://x", loc,
                        post=lambda *a, **k: FakeResp({"error": "oom"}, 500))
        self.assertEqual(stt.transcribe(_audio())[0], "ローカルです")

    def test_no_url_is_pure_local(self):
        loc = FakeLocal()
        stt = RemoteSTT(None, loc, post=lambda *a, **k: self.fail("呼ばない"))
        self.assertEqual(stt.transcribe(_audio())[0], "ローカルです")


class TestConfigAndWiring(unittest.TestCase):
    def test_mac_model_is_small(self):
        # gemma-3-12b(8GB)は16GBのMacで whisper と同居するとスワップし、
        # 翻訳1件45〜67秒(11/05ライブ)。qwen3-4b-2507 は 0.74秒(中央値)/2.3GB。
        self.assertEqual(ENGINE_MODELS["mac"], "google/gemma-3-4b")
        self.assertEqual(Settings().model, "google/gemma-3-4b")

    def test_stt_url_setting(self):
        self.assertEqual(Settings().stt_url, "http://127.0.0.1:12346")

    def test_remote_stt_only_when_translation_runs_on_remote(self):
        # 『このMac』は完全オフラインの約束。リモート耳は リモートサーバーで訳すときだけ。
        src = (ROOT / "src/tsuyaku/__main__.py").read_text()
        i = src.index("def _conversation")
        seg = src[i:src.index("def _stt_worker")]
        self.assertIn("RemoteSTT(", seg)
        self.assertIn('ENGINE_LABELS["remote"]', seg)

    def test_tunnel_forwards_stt_port_and_starts_server(self):
        sh = (ROOT / "scripts/remote_tunnel.sh").read_text()
        self.assertIn("9941", sh)
        self.assertIn("12346", sh)
        self.assertIn("remote_stt_server.py", sh)

    def test_keepalive_touches_stt_server(self):
        # リモートサーバーの耳は30分アイドルで自分で終わる → 会話中は keepalive で起こしておく
        src = (ROOT / "src/tsuyaku/__main__.py").read_text()
        i = src.index("def _keepalive")
        seg = src[i:src.index("def _run")]
        self.assertIn("/health", seg)


if __name__ == "__main__":
    unittest.main()
