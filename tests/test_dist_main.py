"""配る版の入口: 初回の準備の画面の API(状態・開始・止める・終わり)。"""
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tsuyaku import config, dist_main, modelfetch


class FakeFetcher:
    def __init__(self):
        self.started = self.cancelled = 0

    def state(self):
        return {"phase": "idle", "got": 0, "total": 1, "items": []}

    def start(self):
        self.started += 1

    def cancel(self):
        self.cancelled += 1


class Host(dist_main.SetupMixin):
    def __init__(self, models):
        self.s = config.Settings()
        self._setup_init(models, fetcher=FakeFetcher())


class SetupApiTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.models = Path(self.tmp.name) / "models"

    def tearDown(self):
        self.tmp.cleanup()

    def test_state_has_version(self):
        h = Host(self.models)
        st = h.setup_state()
        self.assertEqual(st["phase"], "idle")
        self.assertIn("version", st)

    def test_start_and_cancel(self):
        h = Host(self.models)
        self.assertEqual(h.setup_start(), "ok")
        self.assertEqual(h.setup_cancel(), "ok")
        self.assertEqual((h._fetch.started, h._fetch.cancelled), (1, 1))

    def test_finish_refuses_until_all_models(self):
        h = Host(self.models)
        called = []
        h._on_ready = lambda: called.append(1)
        self.assertEqual(h.setup_finish(), "missing")
        self.assertEqual(called, [])

    def test_finish_resolves_paths_and_opens_main(self):
        h = Host(self.models)
        called = []
        h._on_ready = lambda: called.append(1)
        with mock.patch.object(modelfetch, "missing", return_value=[]), \
                mock.patch.object(dist_main, "_warm_llm") as warm:
            (self.models / config.DIST_SENSE_VOICE).mkdir(parents=True)
            with mock.patch.object(config, "DATA", Path(self.tmp.name)):
                self.assertEqual(h.setup_finish(), "ok")
        self.assertEqual(called, [1])
        warm.assert_called_once()
        self.assertTrue(h.s.sense_voice_dir.endswith(config.DIST_SENSE_VOICE))


class AutostartTest(unittest.TestCase):
    """試験用(SOKUTSU_AUTOSTART=1): 押さずに取得を始め、そろったら本画面へ。"""

    def test_autostart_starts_and_finishes(self):
        h = Host(Path("/nonexistent"))
        phases = iter(["running", "running", "done"])
        h._fetch.state = lambda: {"phase": next(phases, "done")}
        done = []
        h.setup_finish = lambda: done.append(1) or "ok"
        dist_main._autostart(h, poll=0)
        self.assertEqual(h._fetch.started, 1)
        self.assertEqual(done, [1])

    def test_autostart_stops_on_error(self):
        h = Host(Path("/nonexistent"))
        h._fetch.state = lambda: {"phase": "error"}
        done = []
        h.setup_finish = lambda: done.append(1) or "ok"
        dist_main._autostart(h, poll=0)
        self.assertEqual(done, [])


class SnapshotRequestTest(unittest.TestCase):
    """試験用(SOKUTSU_SNAPSHOT=<dir>): req-<名前> を置くと、その名前で窓の絵を残す。"""

    def test_requests_are_taken_once(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "req-setup1").touch()
            (Path(d) / "other.txt").touch()
            self.assertEqual(dist_main._snap_requests(Path(d)), ["setup1"])
            self.assertEqual(dist_main._snap_requests(Path(d)), [])


class LoopbackFqdnTest(unittest.TestCase):
    """同梱の Python は getfqdn('127.0.0.1') に35秒かかり、pywebview の画面サーバーが立たず
    窓が真っ白のままだった(11/07)。127.0.0.1 などは問い合わせずに localhost と答える。"""

    def test_loopback_is_answered_without_lookup(self):
        import socket
        calls = []
        orig = socket.getfqdn
        try:
            socket.getfqdn = lambda name="": calls.append(name) or "slow.example"
            dist_main._fast_loopback_fqdn()
            for n in ("127.0.0.1", "localhost", "::1", "0.0.0.0"):
                self.assertEqual(socket.getfqdn(n), "localhost")
            self.assertEqual(calls, [])
            self.assertEqual(socket.getfqdn("example.org"), "slow.example")
            self.assertEqual(calls, ["example.org"])
        finally:
            socket.getfqdn = orig


if __name__ == "__main__":
    unittest.main()


class SetupLogTest(unittest.TestCase):
    def test_setup_log_prints_to_run_log(self):
        import contextlib
        import io
        h = Host(Path(tempfile.gettempdir()) / "m")
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            self.assertEqual(h.setup_log("boom"), "ok")
        self.assertIn("[setup.js] boom", buf.getvalue())
