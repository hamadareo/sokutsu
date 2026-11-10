"""配る版の監査(11/07): 他人の Mac で起きる取得・翻訳エンジン・起動の失敗の再現。"""
import hashlib
import http.server
import io
import subprocess
import tempfile
import threading
import time
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest import mock

from tsuyaku import config, dist_main, localllm
from tsuyaku import modelfetch as mf

DATA = bytes(range(256)) * 64


def _serve(handler_cls):
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler_cls)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


class PortalHandler(http.server.BaseHTTPRequestHandler):
    """ホテル・駅の Wi-Fi のログイン画面: どの URL にも 200 で HTML を返す。"""

    def log_message(self, *a):
        pass

    def do_GET(self):
        body = b"<html><body>Please log in to the Wi-Fi</body></html>"
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class CaptivePortalTest(unittest.TestCase):
    """ログイン画面の HTML を .part に書くと、後で回線が直って続きを取っても頭が HTML のままで、
    2.5 GB を全部受けてから「壊れていました」になっていた。"""

    def setUp(self):
        self.srv = _serve(PortalHandler)
        self.url = f"http://127.0.0.1:{self.srv.server_port}/m.bin"
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        self.srv.shutdown()
        self.srv.server_close()
        self.tmp.cleanup()

    def test_html_is_not_written_into_the_part_file(self):
        it = mf.Item("a", "A", self.url, hashlib.sha256(DATA).hexdigest(), len(DATA), "a.bin")
        f = mf.Fetcher(self.dir, [it], retries=1, sleep=lambda s: None,
                       free_bytes=lambda p: 10 ** 12)
        f.run()
        st = f.state()
        self.assertEqual(st["phase"], "error")
        part = mf.part_path(it, self.dir)
        self.assertFalse(part.exists() and part.stat().st_size > 0, "HTML が .part に残った")
        self.assertEqual(st["errorKind"], "network")


class TwoInstancesTest(unittest.TestCase):
    """Sokutsu が2つ開いている(DMG の上と /Applications)と、同じ .part に2つが書き足して壊していた。"""

    def test_second_fetcher_waits_instead_of_corrupting(self):
        with tempfile.TemporaryDirectory() as d:
            gate = threading.Event()
            it = mf.Item("a", "A", "u:a", hashlib.sha256(DATA).hexdigest(), len(DATA), "a.bin")

            def slow(url, offset):
                def gen():
                    yield DATA[offset:offset + 100]
                    gate.wait(5)
                    yield DATA[offset + 100:]
                return gen(), True, (lambda: None)

            def fast(url, offset):
                return iter([DATA[offset:]]), True, (lambda: None)

            a = mf.Fetcher(Path(d), [it], opener=slow, retries=0, sleep=lambda s: None,
                           free_bytes=lambda p: 10 ** 12)
            b = mf.Fetcher(Path(d), [it], opener=fast, retries=0, sleep=lambda s: None,
                           free_bytes=lambda p: 10 ** 12)
            t = threading.Thread(target=a.run)
            t.start()
            for _ in range(200):
                if a.state()["got"] >= 100:
                    break
                time.sleep(0.01)
            b.run()
            gate.set()
            t.join(10)
            self.assertEqual(b.state()["phase"], "error")
            self.assertEqual(b.state()["errorKind"], "busy")
            self.assertEqual(a.state()["phase"], "done", a.state()["error"])
            self.assertEqual((Path(d) / "a.bin").read_bytes(), DATA)


class WatchdogTest(unittest.TestCase):
    """翻訳エンジンが落ちて起こし直すたびに見張りが増え、古い見張りはアプリの終わりに
    もう居ない(別のプロセスに使い回されたかもしれない)pid を kill していた。"""

    def test_restart_replaces_the_watchdog(self):
        with tempfile.TemporaryDirectory() as d:
            fake = Path(d) / "llama-server"
            fake.write_text("#!/bin/sh\nexec sleep 60\n")
            fake.chmod(0o755)
            model = Path(d) / "m.gguf"
            model.write_bytes(b"x")
            llm = localllm.LocalLLM(fake, model, port=5999, getter=lambda *a, **k: None)
            with redirect_stdout(io.StringIO()):
                llm.start()
                dog1 = llm._dog
                llm.proc.kill()
                llm.proc.wait()
                llm.start()
            try:
                self.assertIsNot(llm._dog, dog1)
                try:
                    dog1.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    pass
                self.assertIsNotNone(dog1.poll(), "古い見張りが残っている")
            finally:
                llm.stop()
                if dog1.poll() is None:
                    dog1.kill()


class BrokenConfigTest(unittest.TestCase):
    """配る版の説明書は config.json を手で書き換える(server_url)よう案内している。
    JSON の書き間違い(末尾のカンマなど)があると、窓も出ずに毎回すぐ終わっていた。"""

    def test_dist_starts_with_defaults_and_keeps_the_broken_file(self):
        with tempfile.TemporaryDirectory() as d:
            data = Path(d)
            bad = '{"server_url": "http://192.0.2.5:8080",}\n'
            (data / "config.json").write_text(bad)
            with mock.patch.object(config, "DIST", True), mock.patch.object(config, "DATA", data), \
                    redirect_stdout(io.StringIO()):
                s = config.load_settings()
            self.assertEqual(s.server_url, "")
            kept = list(data.glob("config.broken*.json"))
            self.assertEqual(len(kept), 1)
            self.assertEqual(kept[0].read_text(), bad)

    def test_dist_non_object_config(self):
        with tempfile.TemporaryDirectory() as d:
            data = Path(d)
            (data / "config.json").write_text("[]")
            with mock.patch.object(config, "DIST", True), mock.patch.object(config, "DATA", data), \
                    redirect_stdout(io.StringIO()):
                config.load_settings()


class StartupFailureTest(unittest.TestCase):
    """起動の途中の失敗(データの場所に書けない等)は、Dock のアイコンが跳ねて消えるだけだった。"""

    def test_fatal_error_is_shown(self):
        with tempfile.TemporaryDirectory() as d:
            blocker = Path(d) / "file"
            blocker.write_text("x")          # ここをフォルダとして作れない
            shown = []
            with mock.patch.object(dist_main.distenv, "DIST", True), \
                    mock.patch.object(config, "DATA", blocker / "Sokutsu"), \
                    mock.patch.object(dist_main, "_alert", lambda *a: shown.append(a)), \
                    redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                rc = dist_main.main()
            self.assertEqual(rc, 1)
            self.assertEqual(len(shown), 1)
            self.assertIn("run.log", shown[0][1])


if __name__ == "__main__":
    unittest.main()


class MidMeetingCrashTest(unittest.TestCase):
    """会議の途中で同梱の llama-server が落ちる・応答しなくなると、その会話の残りの翻訳は
    全部失敗していた(ensure_for は会話の始めとメモの時だけ)。配る版では起こし直す。"""

    def setUp(self):
        self.now = [100.0]
        self.spawned = []
        self.procs = []

        class P:
            def __init__(s, cmd, **kw):
                s.pid, s.code, s.terminated = 4243, None, False
                self.procs.append(s)

            def poll(s):
                return s.code

            def terminate(s):
                s.terminated, s.code = True, -15

            def kill(s):
                s.code = -9

            def wait(s, timeout=None):
                return s.code

        self.health = [200]

        def get(url, timeout=None):
            code = self.health.pop(0) if len(self.health) > 1 else self.health[0]
            if code is None:
                raise OSError("refused")
            return mock.Mock(status_code=code)

        self.llm = localllm.LocalLLM(Path("/app/llama/llama-server"), Path("/x/m.gguf"), port=5555,
                                     popen=P, getter=get, sleep=lambda s: None, watchdog=False,
                                     clock=lambda: self.now[0],
                                     spawn=lambda fn: self.spawned.append(fn))
        self._isf = mock.patch.object(Path, "is_file", return_value=True)
        self._isf.start()
        with redirect_stdout(io.StringIO()):
            self.assertTrue(self.llm.ensure(timeout=5))

    def tearDown(self):
        self._isf.stop()

    def run_spawned(self):
        fns, self.spawned[:] = list(self.spawned), []
        with redirect_stdout(io.StringIO()):
            for fn in fns:
                fn()

    def test_dead_server_is_restarted_once(self):
        self.procs[0].code = 1                     # 落ちた
        for _ in range(5):                          # 先読みと確定の訳が続けて失敗
            self.llm.note_failure()
        self.assertEqual(len(self.spawned), 1)     # 同時に何度も起こさない
        self.run_spawned()
        self.assertEqual(len(self.procs), 2)
        self.assertTrue(self.llm.running())

    def test_hung_server_waits_a_few_seconds_then_restarts_once(self):
        self.health[0] = None                       # 生きているが応答しない
        self.llm.note_failure()
        self.now[0] += 1.0
        self.llm.note_failure()
        self.assertEqual(self.spawned, [])          # 一瞬の失敗では起こし直さない
        self.now[0] += 3.0
        self.llm.note_failure()
        self.llm.note_failure()
        self.assertEqual(len(self.spawned), 1)
        self.health[:] = [None, 200]                # 起こし直しの確かめでも応答せず、新しいものは応答する
        self.run_spawned()
        self.assertTrue(self.procs[0].terminated)  # 応答しないものは止めてから
        self.assertEqual(len(self.procs), 2)
        self.now[0] += 10.0
        self.llm.note_failure()                     # 同じ失敗の続きではもう起こさない
        self.assertEqual(self.spawned, [])

    def test_success_resets_the_streak(self):
        self.procs[0].code = 1
        self.llm.note_failure()
        self.run_spawned()
        self.llm.note_ok()
        self.procs[1].code = 1                      # また落ちた
        self.llm.note_failure()
        self.assertEqual(len(self.spawned), 1)

    def test_translate_reports_to_the_bundled_engine(self):
        import requests
        from tsuyaku import translate
        fake = mock.Mock(url="http://127.0.0.1:5555")
        with mock.patch.object(localllm, "DIST", True), \
                mock.patch.object(localllm, "instance", return_value=fake), \
                mock.patch.object(translate.requests, "post",
                                  side_effect=requests.ConnectionError("refused")), \
                redirect_stdout(io.StringIO()):
            out = translate.translate_with_retry("http://127.0.0.1:5555", "m", [], tries=2,
                                                 sleep=lambda s: None)
            self.assertEqual(out, "")               # 会話は止めない(空訳で返る)
            self.assertEqual(fake.note_failure.call_count, 2)
            translate.translate_with_retry("http://gpu.example:8080", "m", [], tries=1)
            self.assertEqual(fake.note_failure.call_count, 2)   # 自分のサーバーは関係ない
        resp = mock.Mock(status_code=200)
        resp.iter_lines.return_value = [b'data: {"choices":[{"delta":{"content":"ok"}}]}',
                                        b"data: [DONE]"]
        with mock.patch.object(localllm, "DIST", True), \
                mock.patch.object(localllm, "instance", return_value=fake), \
                mock.patch.object(translate.requests, "post", return_value=resp):
            self.assertEqual(translate.translate_stream("http://127.0.0.1:5555", "m", []), "ok")
        fake.note_ok.assert_called_once()

    def test_user_app_is_unchanged(self):
        with mock.patch.object(localllm, "instance") as inst:
            localllm.note_failure("http://127.0.0.1:1234")
            localllm.note_ok("http://127.0.0.1:1234")
            inst.assert_not_called()
