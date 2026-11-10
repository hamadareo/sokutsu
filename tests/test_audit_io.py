"""監査(11/07): 取り込み・録音・再生・設定・この Mac の音の、失敗や中断の道。"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request
from pathlib import Path
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_importer import SR, FakeCapture, FakeSpk, write_wav  # noqa: E402

from tsuyaku import config as cfgmod  # noqa: E402
from tsuyaku import importer as I  # noqa: E402
from tsuyaku.mediaserver import MediaServer  # noqa: E402
from tsuyaku.sysaudio import SystemAudio  # noqa: E402

FAKE = str(Path(__file__).with_name("fake_sysaudio.py"))


class SlowSTT:
    def __init__(self, delay=0.05):
        self.delay = delay
        self.calls = 0

    def transcribe(self, a, **_k):
        self.calls += 1
        time.sleep(self.delay)
        return ("はい。", "ja", 0.9)


class TestImportCancel(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.TemporaryDirectory()
        self.dir = Path(self.d.name)
        x = (0.1 * np.sin(2 * np.pi * 200 * np.arange(SR * 30) / SR)).astype(np.float32)
        self.src = self.dir / "long.wav"
        write_wav(self.src, x, SR)
        self.spans = [(0.5 * i + 0.05, 0.5 * i + 0.4) for i in range(58)]

    def tearDown(self):
        self.d.cleanup()

    def test_cancel_stops_queued_transcriptions(self):
        """中止したら、並べてあった区間の聞き取りも止める(以前は中止の後も全区間を聞き続けた)。"""
        stt = SlowSTT(0.05)
        ev = threading.Event()

        def prog(stage, pct, info=None):
            if stage == "listen" and pct > 0.26:   # 最初の区間が聞き終わった所で中止
                ev.set()

        r = I.run_import(self.src, self.dir / "logs", "ja-en", "", stt=stt,
                         make_capture=lambda q, x: FakeCapture(q, self.spans, x), spk=FakeSpk(),
                         prog=prog, cancel=ev)
        self.assertIsNone(r)
        n = stt.calls
        time.sleep(0.6)
        self.assertLessEqual(stt.calls, n + 1)
        self.assertLess(stt.calls, 10)
        self.assertEqual(list((self.dir / "logs").glob("session-*")), [])

    def test_cancel_after_done_keeps_finished_meeting(self):
        """終わった瞬間に中止が押されても、返した会議を消さない(返した id の記録が無くなっていた)。"""
        stt = SlowSTT(0.0)
        ev = threading.Event()

        def prog(stage, pct, info=None):
            if stage == "done":
                ev.set()

        r = I.run_import(self.src, self.dir / "logs", "ja-en", "", stt=stt,
                         make_capture=lambda q, x: FakeCapture(q, self.spans[:3], x), spk=FakeSpk(),
                         prog=prog, cancel=ev)
        self.assertIsNotNone(r)
        self.assertTrue((self.dir / "logs" / f"{r['id']}.jsonl").exists())


class TestImportStaleWork(unittest.TestCase):
    def test_cleanup_stale_work_files(self):
        """落ちた取り込みの作業ファイル(数百MBになる)を次の起動で消す。新しいものと会議には触らない。"""
        with tempfile.TemporaryDirectory() as d:
            logs = Path(d)
            old = logs / ".import-1.work.wav"
            old_m4a = logs / ".import-1.work.wav.m4a"
            new = logs / ".import-2.work.wav"
            keep = logs / "session-20261107-120000.jsonl"
            for p in (old, old_m4a, new, keep):
                p.write_bytes(b"x" * 10)
            t = time.time() - 7 * 3600
            for p in (old, old_m4a, keep):
                os.utime(p, (t, t))
            removed = I.cleanup_stale(logs)
            self.assertEqual(sorted(p.name for p in removed), sorted([old.name, old_m4a.name]))
            self.assertFalse(old.exists())
            self.assertFalse(old_m4a.exists())
            self.assertTrue(new.exists())
            self.assertTrue(keep.exists())
            self.assertEqual(I.cleanup_stale(logs / "nope"), [])


class TestShutdownDuringImport(unittest.TestCase):
    def test_shutdown_cancels_running_import(self):
        """窓を閉じると os._exit で終わる。取り込み中なら中止して後片付け(作りかけの会議を消す)を待つ。"""
        from tsuyaku.__main__ import Bridge
        with tempfile.TemporaryDirectory() as d:
            b = Bridge()
            b.s.log_dir = d
            b.ui = type("U", (), {"put": lambda self, e: None})()
            src = Path(d) / "a.wav"
            src.write_bytes(b"x")
            seen = {}

            def fake_run(path, log_dir, pair, topic, **kw):
                ev = kw["cancel"]
                for _ in range(500):
                    if ev.is_set():
                        seen["cancelled"] = True
                        return None
                    time.sleep(0.01)
                return {"id": "x", "lines": 0, "dur": 1.0}

            fake_ear = (type("E", (), {"langs": (), "prompt": ""})(),
                        type("L", (), {"langs": (), "fast": True})(), None)
            with mock.patch.object(Bridge, "_import_ear", return_value=fake_ear), \
                    mock.patch("tsuyaku.importer.run_import", side_effect=fake_run):
                self.assertEqual(b.import_start(str(src), "ja", ""), "started")
                time.sleep(0.1)
                b._shutdown()
                self.assertFalse(b._import_th.is_alive())
            self.assertTrue(seen.get("cancelled"))


class TestConfigRobust(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.TemporaryDirectory()
        self.root = Path(self.d.name)
        self.p = mock.patch.object(cfgmod, "ROOT", self.root)
        self.p.start()
        self.env = mock.patch.dict(os.environ, {k: v for k, v in os.environ.items()
                                                if not k.startswith("TSUYAKU_")}, clear=True)
        self.env.start()

    def tearDown(self):
        self.env.stop()
        self.p.stop()
        self.d.cleanup()

    def test_corrupt_config_still_starts_and_is_kept(self):
        (self.root / "config.json").write_text('{"engine": "mac", "vad_thr')
        s = cfgmod.load_settings()
        self.assertEqual(s.engine, cfgmod.Settings().engine)
        # 壊れた中身は消さずに横へ残す(手で直せるように)
        self.assertTrue(list(self.root.glob("config.broken*.json")))

    def test_non_dict_config(self):
        for body in ("[]", "null", "3", '"x"'):
            (self.root / "config.json").write_text(body)
            self.assertEqual(cfgmod.load_settings().engine, "auto")

    def test_wrong_types_are_ignored(self):
        (self.root / "config.json").write_text(json.dumps({
            "port": "abc", "vad_threshold": "high", "tts_voice": "say:Kyoko", "voice_on": "yes",
            "hold_open_s": 0, "engine": "mac", "tts_speed": None}))
        s = cfgmod.load_settings()
        d = cfgmod.Settings()
        self.assertEqual(s.port, d.port)
        self.assertEqual(s.vad_threshold, d.vad_threshold)
        self.assertEqual(s.tts_voice, d.tts_voice)
        self.assertEqual(s.voice_on, d.voice_on)
        self.assertEqual(s.tts_speed, d.tts_speed)
        self.assertEqual(s.hold_open_s, 0)        # float の欄に int は使える
        self.assertEqual(s.engine, "mac")

    def test_save_is_atomic_on_disk_full(self):
        (self.root / "config.json").write_text(json.dumps({"engine": "mac", "last_topic": "古い"}))
        real = Path.write_text

        def half(self_, data, *a, **k):
            real(self_, data[:7], *a, **k)
            raise OSError(28, "No space left on device")

        s = cfgmod.load_settings()
        s.last_topic = "新しい"
        with mock.patch.object(Path, "write_text", half):
            cfgmod.save_config(s)
        self.assertEqual(json.loads((self.root / "config.json").read_text())["last_topic"], "古い")
        self.assertEqual([p.name for p in self.root.iterdir()], ["config.json"])


class TestMediaServerEdges(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.TemporaryDirectory()
        self.base = Path(self.d.name)
        self.logs = self.base / "logs"
        self.logs.mkdir()
        self.srv = MediaServer(self.logs)

    def tearDown(self):
        self.srv.close()
        self.d.cleanup()

    def test_symlink_out_of_log_dir_is_refused(self):
        secret = self.base / "secret.txt"
        secret.write_bytes(b"secret" * 100)
        link = self.logs / "session-20261107-000000.audio.m4a"
        link.symlink_to(secret)
        with self.assertRaises(urllib.error.HTTPError) as cm:
            urllib.request.urlopen(self.srv.url(link), timeout=5)
        self.assertEqual(cm.exception.code, 404)

    def test_non_ascii_token_gets_403_not_a_dropped_connection(self):
        import socket
        with socket.create_connection(("127.0.0.1", self.srv.port), timeout=5) as c:
            c.sendall(b"GET /a/\xc3\xa9\xc3\xa9/session-1.audio.m4a HTTP/1.0\r\n\r\n")
            head = c.recv(64)
        self.assertTrue(head.startswith(b"HTTP/1.0 403"), head)


class TestSysAudioStopRace(unittest.TestCase):
    def test_stop_before_proc_registered_does_not_leave_helper_running(self):
        """stop() が部品の起動と行き違っても、部品を残さない(読み続けるスレッドとタップが残っていた)。"""
        sa = SystemAudio(lambda y: None, cmd=[sys.executable, FAKE, "--mode", "hold"])
        sa._stop.set()          # 起動の直前に stop() が来た(_proc はまだ None だった)
        done = []
        th = threading.Thread(target=lambda: done.append(sa._run_once()), daemon=True)
        th.start()
        th.join(5)
        alive = th.is_alive()
        if alive and sa._proc is not None:
            sa._proc.kill()
        self.assertFalse(alive)



class TestSameSecondSessions(unittest.TestCase):
    """同じ秒に2つ目の記録を開くと、以前は前の記録に追記し(録音の CAF は上書き)、取り込みの失敗・中止の
    後片付け(session-X.*)が前の会議ごと消していた。衝突した時だけ session-X_2, _3 にする。"""

    STAMP = "20261107-120000"

    def setUp(self):
        self.d = tempfile.TemporaryDirectory()
        self.logs = Path(self.d.name) / "logs"
        real = time.strftime
        self.p = mock.patch("tsuyaku.log.time.strftime",
                            side_effect=lambda f, *a: self.STAMP if f == "%Y%m%d-%H%M%S" else real(f, *a))
        self.p.start()

    def tearDown(self):
        self.p.stop()
        self.d.cleanup()

    def test_second_log_in_same_second_gets_its_own_name(self):
        from tsuyaku.log import SessionLog
        a = SessionLog(str(self.logs), "ja-en", "A", "notes", audio=True)
        a.add("ja", "最初の会議", "", None)
        a.close()
        b = SessionLog(str(self.logs), "ja-en", "B", "notes", audio=True)
        c = SessionLog(str(self.logs), "ja-en", "C", "notes", audio=True)
        b.close()
        c.close()
        self.assertEqual(Path(a.path_jsonl).stem, f"session-{self.STAMP}")       # 今の形は変えない
        self.assertEqual(Path(b.path_jsonl).stem, f"session-{self.STAMP}_2")
        self.assertEqual(Path(c.path_jsonl).stem, f"session-{self.STAMP}_3")
        self.assertNotEqual(a.audio_caf, b.audio_caf)
        ra = [json.loads(x) for x in Path(a.path_jsonl).read_text(encoding="utf-8").splitlines()]
        self.assertEqual([r["type"] for r in ra][:1], ["meta"])
        self.assertEqual(ra[0]["topic"], "A")
        self.assertEqual(len([r for r in ra if r["type"] == "meta"]), 1)
        rb = [json.loads(x) for x in Path(b.path_jsonl).read_text(encoding="utf-8").splitlines()]
        self.assertEqual(rb[0]["topic"], "B")
        self.assertEqual(Path(b.audio_caf).name, f"session-{self.STAMP}_2.audio.caf")

    def test_failed_import_in_same_second_keeps_earlier_meeting(self):
        from tsuyaku.log import SessionLog
        a = SessionLog(str(self.logs), "ja-en", "A", "notes")
        a.add("ja", "残すべき会議", "", None)
        a.close()
        src = Path(self.d.name) / "t.wav"
        write_wav(src, (0.1 * np.sin(2 * np.pi * 200 * np.arange(SR * 3) / SR)).astype(np.float32), SR)

        def cap(q, x):
            return FakeCapture(q, [(0.5, 1.5)], x)
        ev = threading.Event()     # 区切りの途中で中止 → 後片付け(_remove の session-X.*)を通す

        def prog(stage, pct, info=None):
            if stage == "listen":
                ev.set()
        r = I.run_import(src, self.logs, "ja-en", "", stt=SlowSTT(0.0), make_capture=cap, prog=prog, cancel=ev)
        self.assertIsNone(r)
        self.assertTrue(Path(a.path_jsonl).exists())
        self.assertIn("残すべき会議", Path(a.path_jsonl).read_text(encoding="utf-8"))
        self.assertEqual(sorted(p.name for p in self.logs.glob("session-*")),
                         sorted([f"session-{self.STAMP}.jsonl", f"session-{self.STAMP}.txt"]))

    def test_readers_accept_suffixed_id(self):
        from tsuyaku import library as L
        from tsuyaku import recorder as R
        from tsuyaku.log import SessionLog
        a = SessionLog(str(self.logs), "ja-en", "A", "notes", audio=True)
        b = SessionLog(str(self.logs), "ja-en", "B", "notes", audio=True)
        for lg in (a, b):
            lg.add("ja", "こんにちは。", "", None)   # 1行も無い記録は一覧に出ない
            lg.close()
        mid = Path(b.path_jsonl).stem
        self.assertTrue(L.valid_id(mid))
        self.assertEqual(R.caf_path(b.path_jsonl).name, f"{mid}.audio.caf")
        self.assertEqual([e["id"] for e in L.list_meetings(self.logs)], [mid, Path(a.path_jsonl).stem])   # 新しい順
        # 前の会議の付属ファイル(session-X.*)に、後の会議(session-X_2.*)は入らない
        for f in (".audio.m4a", ".tags.json"):
            (self.logs / f"session-{self.STAMP}{f}").write_bytes(b"a" * 200)
            (self.logs / f"{mid}{f}").write_bytes(b"b" * 200)
        self.assertTrue(all(p.name.startswith(f"session-{self.STAMP}.") for p in
                            L._files_of(self.logs, f"session-{self.STAMP}")))
        self.assertTrue(L.trash(self.logs, f"session-{self.STAMP}"))
        self.assertTrue(Path(b.path_jsonl).exists())
        self.assertTrue((self.logs / f"{mid}.audio.m4a").exists())
        srv = MediaServer(self.logs)
        try:
            with urllib.request.urlopen(srv.url(self.logs / f"{mid}.audio.m4a"), timeout=5) as r:
                self.assertEqual(r.read(), b"b" * 200)
        finally:
            srv.close()
        from tsuyaku.__main__ import Bridge
        br = Bridge()
        br.s.log_dir = str(self.logs)
        self.assertEqual(br._meeting_path(mid), Path(b.path_jsonl))


if __name__ == "__main__":
    unittest.main()
