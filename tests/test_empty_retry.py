"""空訳の根治: SSE の error フレームを握りつぶさない + 空訳の自動再試行。

11/05 15:42 ライブ実障害: 「こんにちは」→「（翻訳できませんでした）」が連発。
原因は2つ重なっていた。
① translate_stream の SSE ループが `data: {"error": ...}` フレームを
   KeyError/IndexError/JSONDecodeError で `continue` して握りつぶし、
   例外なしに空文字を返していた(サーバーは明示的にエラーを返していた)。
② 空訳に対するリトライが無かった。リトライは looks_truncated(非空の1〜3語訳)
   のときだけ発火し、空はそのまま失敗確定していた。
実測で失敗した6文は全部 再実行すると正しく訳せた = サーバーの一時的混線。
1回だけ少し待ってやり直せば救える。
"""
import json
import unittest
from pathlib import Path

from tsuyaku import translate as tr

ROOT = Path(__file__).resolve().parents[1]
MAIN_SRC = (ROOT / "src" / "tsuyaku" / "__main__.py").read_text()


class _Resp:
    status_code = 200

    def __init__(self, lines):
        self._lines = lines

    def raise_for_status(self):
        pass

    def iter_lines(self):
        return iter(self._lines)


def _sse(*payloads: str) -> list[bytes]:
    return [f"data: {p}".encode() for p in payloads]


def _frame(text: str) -> str:
    return json.dumps({"choices": [{"delta": {"content": text}}]})


class TestSSEErrorFrame(unittest.TestCase):
    def test_error_frame_raises_instead_of_silent_empty(self):
        # サーバーが error フレームを返したのに「空訳」として無言で終わるのを許さない
        err = json.dumps({"error": {"message": "Server error",
                                    "type": "internal_server_error"}})
        orig = tr.requests.post
        tr.requests.post = lambda url, **kw: _Resp(
            _sse(err, "[DONE]"))
        try:
            with self.assertRaises(RuntimeError):
                tr.translate_stream("http://x", "m",
                                    [{"role": "user", "content": "こんにちは"}])
        finally:
            tr.requests.post = orig

    def test_error_frame_mid_stream_raises(self):
        err = json.dumps({"error": {"message": "boom"}})
        orig = tr.requests.post
        tr.requests.post = lambda url, **kw: _Resp(
            _sse(_frame("Hallo"), err, "[DONE]"))
        try:
            with self.assertRaises(RuntimeError):
                tr.translate_stream("http://x", "m",
                                    [{"role": "user", "content": "hi"}])
        finally:
            tr.requests.post = orig

    def test_ordinary_unknown_frames_still_skipped(self):
        # error を含まない意味不明フレーム(ハートビート等)は従来どおり無視して続ける
        orig = tr.requests.post
        tr.requests.post = lambda url, **kw: _Resp(
            _sse('{"ping": 1}', _frame("Hello."), "[DONE]"))
        try:
            out = tr.translate_stream("http://x", "m",
                                      [{"role": "user", "content": "hi"}])
        finally:
            tr.requests.post = orig
        self.assertEqual(out, "Hello.")


class TestTranslateWithRetry(unittest.TestCase):
    def _patch(self, results):
        calls = {"n": 0}
        orig = tr.translate_stream

        def fake(base, model, messages, on_delta=None, timeout=30.0):
            r = results[min(calls["n"], len(results) - 1)]
            calls["n"] += 1
            if isinstance(r, Exception):
                raise r
            return r
        tr.translate_stream = fake
        self.addCleanup(setattr, tr, "translate_stream", orig)
        return calls

    def test_success_first_try_no_retry(self):
        calls = self._patch(["Hello."])
        slept = []
        out = tr.translate_with_retry("http://x", "m", [{"role": "user", "content": "hi"}],
                                      sleep=slept.append)
        self.assertEqual(out, "Hello.")
        self.assertEqual(calls["n"], 1)
        self.assertEqual(slept, [])

    def test_empty_then_success_retries_once(self):
        calls = self._patch(["", "こんにちは。"])
        slept = []
        out = tr.translate_with_retry("http://x", "m", [{"role": "user", "content": "hi"}],
                                      sleep=slept.append)
        self.assertEqual(out, "こんにちは。")
        self.assertEqual(calls["n"], 2)
        self.assertEqual(len(slept), 1)

    def test_gives_up_after_tries(self):
        calls = self._patch(["", ""])
        out = tr.translate_with_retry("http://x", "m", [{"role": "user", "content": "hi"}],
                                      sleep=lambda s: None)
        self.assertEqual(out, "")
        self.assertEqual(calls["n"], 2)

    def test_transient_exception_also_retried(self):
        # サーバーの一時的な接続切れも同じ1回リトライで救う
        calls = self._patch([ConnectionError("boom"), "Ok."])
        out = tr.translate_with_retry("http://x", "m", [{"role": "user", "content": "hi"}],
                                      sleep=lambda s: None)
        self.assertEqual(out, "Ok.")
        self.assertEqual(calls["n"], 2)

    def test_all_exceptions_return_empty_not_raise(self):
        self._patch([ConnectionError("boom"), ConnectionError("boom")])
        out = tr.translate_with_retry("http://x", "m", [{"role": "user", "content": "hi"}],
                                      sleep=lambda s: None)
        self.assertEqual(out, "")


class TestMainWiring(unittest.TestCase):
    def test_utterance_uses_retry_wrapper(self):
        # _utterance は素の translate_stream でなく空訳リトライ付きを使う
        i = MAIN_SRC.index("def _utterance(")
        seg = MAIN_SRC[i:]
        self.assertIn("translate_with_retry(base, model, msgs, on_delta)", seg)

    def test_main_imports_retry_wrapper(self):
        self.assertIn("translate_with_retry", MAIN_SRC[:MAIN_SRC.index("class Bridge")])


if __name__ == "__main__":
    unittest.main()
