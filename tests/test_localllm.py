"""配る版の同梱翻訳エンジン(llama-server)の起動・待ち・停止。"""
import unittest
from pathlib import Path
from unittest import mock

from tsuyaku import localllm


class FakeProc:
    def __init__(self, cmd, **kw):
        self.cmd, self.kw = cmd, kw
        self.pid = 4242
        self.code = None
        self.terminated = False

    def poll(self):
        return self.code

    def terminate(self):
        self.terminated = True
        self.code = -15

    def kill(self):
        self.code = -9

    def wait(self, timeout=None):
        return self.code


class FakeResp:
    def __init__(self, code):
        self.status_code = code


def make(tmp="/tmp/x", health=(200,), **kw):
    seq = list(health)
    procs = []

    def popen(cmd, **k):
        p = FakeProc(cmd, **k)
        procs.append(p)
        return p

    def get(url, timeout=None):
        code = seq.pop(0) if len(seq) > 1 else seq[0]
        if code is None:
            raise OSError("refused")
        return FakeResp(code)

    llm = localllm.LocalLLM(binary=Path("/app/llama/llama-server"), model=Path(tmp) / "m.gguf",
                            port=5555, log_path=None, popen=popen, getter=get,
                            sleep=lambda s: None, watchdog=False, **kw)
    return llm, procs


class LocalLLMTest(unittest.TestCase):
    def test_command(self):
        llm, _ = make()
        cmd = llm.command()
        self.assertEqual(cmd[0], "/app/llama/llama-server")
        self.assertIn("--host", cmd)
        self.assertEqual(cmd[cmd.index("--host") + 1], "127.0.0.1")   # 外からは繋がらない
        self.assertEqual(cmd[cmd.index("--port") + 1], "5555")
        self.assertEqual(cmd[cmd.index("--alias") + 1], "google/gemma-3-4b")
        self.assertEqual(cmd[cmd.index("-np") + 1], "2")              # 先読み翻訳と会議メモ
        # gemma 3 の SWA: 全部の KV を残さないと毎回プロンプトを最初から計算し直す(11/07 実測
        # 初字 0.41 → 1.02 秒)。q8_0 の KV で大きさを抑え、-kvu で長い会議メモにも全体を使う
        for f in ("--swa-full", "-kvu"):
            self.assertIn(f, cmd)
        self.assertEqual(cmd[cmd.index("-ctk") + 1], "q8_0")
        self.assertEqual(cmd[cmd.index("-ctv") + 1], "q8_0")
        self.assertEqual(cmd[cmd.index("-fa") + 1], "on")
        self.assertEqual(cmd[cmd.index("-c") + 1], "16384")
        self.assertEqual(llm.url, "http://127.0.0.1:5555")

    def test_ensure_waits_for_health(self):
        llm, procs = make(health=(None, 503, 503, 200))
        with mock.patch.object(Path, "is_file", return_value=True):
            self.assertTrue(llm.ensure(timeout=10))
        self.assertEqual(len(procs), 1)
        # 2回目は起こし直さない
        with mock.patch.object(Path, "is_file", return_value=True):
            self.assertTrue(llm.ensure(timeout=10))
        self.assertEqual(len(procs), 1)

    def test_ensure_restarts_dead_server(self):
        llm, procs = make(health=(200,))
        with mock.patch.object(Path, "is_file", return_value=True):
            llm.ensure(timeout=5)
            procs[0].code = 1          # 落ちた
            llm._ok = False
            self.assertTrue(llm.ensure(timeout=5))
        self.assertEqual(len(procs), 2)

    def test_ensure_gives_up_when_process_exits(self):
        llm, procs = make(health=(None,))

        def popen(cmd, **k):
            p = FakeProc(cmd, **k)
            p.code = 1                 # すぐ終わった(モデルが壊れている等)
            procs.append(p)
            return p
        llm.popen = popen
        with mock.patch.object(Path, "is_file", return_value=True):
            self.assertFalse(llm.ensure(timeout=5))

    def test_missing_model_is_not_started(self):
        llm, procs = make()
        self.assertFalse(llm.ensure(timeout=1))
        self.assertEqual(procs, [])
        self.assertIn("モデル", llm.error)

    def test_stop(self):
        llm, procs = make()
        with mock.patch.object(Path, "is_file", return_value=True):
            llm.ensure(timeout=5)
        llm.stop()
        self.assertTrue(procs[0].terminated)
        self.assertFalse(llm.running())


class ModuleTest(unittest.TestCase):
    def test_ensure_for_is_noop_in_user_app(self):
        # ユーザー本人の版(DIST=False)では何も起こさない
        with mock.patch.object(localllm, "instance") as inst:
            self.assertFalse(localllm.ensure_for("http://127.0.0.1:1234"))
            inst.assert_not_called()

    def test_ensure_for_matches_bundled_url(self):
        fake = mock.Mock(url="http://127.0.0.1:5555")
        fake.ensure.return_value = True
        with mock.patch.object(localllm, "DIST", True), \
                mock.patch.object(localllm, "instance", return_value=fake):
            self.assertFalse(localllm.ensure_for("http://gpu.example:8080"))
            self.assertTrue(localllm.ensure_for("http://127.0.0.1:5555"))

    def test_free_port(self):
        p = localllm.free_port()
        self.assertTrue(1024 < p < 65536)


if __name__ == "__main__":
    unittest.main()
