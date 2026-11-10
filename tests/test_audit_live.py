"""会話の流れ(マイク → 区切り → 耳 → 訳 → 画面・声)の監査(11/07)で見つけた不具合の再現。

それぞれ「どう壊れるか」を先に試験にしてから直した。
"""
from __future__ import annotations

import queue
import tempfile
import threading
import time
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

import tsuyaku.__main__ as main_mod
import tsuyaku.audio as audio_mod
from tsuyaku.__main__ import Bridge
from tsuyaku.audio import SpeechCapture
from tsuyaku.config import Settings
from tsuyaku.sysaudio import Mixer
from tsuyaku.uplink import Uplink


def _wait(cond, timeout=3.0, step=0.01) -> bool:
    t_end = time.time() + timeout
    while time.time() < t_end:
        if cond():
            return True
        time.sleep(step)
    return cond()


class _Win:
    def __init__(self):
        self.js: list[str] = []

    def evaluate_js(self, js):
        self.js.append(js)


def _bare_bridge() -> Bridge:
    b = Bridge.__new__(Bridge)
    b.s = Settings()
    b.window = _Win()
    b.ui = queue.Queue()
    b.running = threading.Event()
    b.closing = threading.Event()
    b.interrupting = threading.Event()
    b.capture = b.tts = b.log = None
    b.spec = None
    b.history = []
    b.tts_q = queue.Queue()
    return b


# ---------------------------------------------------------------------------
# 1. 画面へ送る係(_pump)が、会議を終えた瞬間の capture=None で死ぬ
# ---------------------------------------------------------------------------
class TestPumpSurvivesCaptureCleared(unittest.TestCase):
    def test_pump_keeps_running_when_capture_goes_away_mid_check(self):
        b = _bare_bridge()

        class _VanishingCap:
            """「if self.capture」を通った直後に会議が終わる(end_meeting が None にする)。"""
            paused = threading.Event()
            level = 0.5

            def __bool__(self):
                b.capture = None
                return True

        b.capture = _VanishingCap()
        th = threading.Thread(target=b._pump, daemon=True)
        th.start()
        time.sleep(0.15)
        b.ui.put(("status", "listen", "聞いています"))
        ok = _wait(lambda: any("TSUYAKU.status" in j for j in b.window.js), 1.0)
        b.closing.set()
        th.join(1.0)
        self.assertTrue(ok, "画面へ送る係が止まり、以後の字幕・状態が画面に出ない")


# ---------------------------------------------------------------------------
# 2. 訳の担当(_text_worker)が1件の例外(記録の書き込み失敗など)で無言で死ぬ
# ---------------------------------------------------------------------------
class TestTextWorkerSurvivesError(unittest.TestCase):
    def test_one_failing_utterance_does_not_stop_translation(self):
        b = _bare_bridge()
        b.running.set()
        done: list = []

        def utt(base, model, pair_t, topic, text, lang, t0, tm=None):
            if text == "壊れる":
                raise OSError(28, "No space left on device")
            done.append(text)

        b._utterance = utt
        q: queue.Queue = queue.Queue()
        th = threading.Thread(target=b._text_worker, args=(q, "b", "m", ("ja", "en"), ""),
                              daemon=True)
        th.start()
        q.put(("壊れる", "ja", time.time(), {}))
        time.sleep(0.3)
        q.put(("次の発言です。", "ja", time.time(), {}))
        ok = _wait(lambda: done == ["次の発言です。"], 2.0)
        b.running.clear()
        th.join(1.0)
        self.assertTrue(ok, "1件の失敗で訳の担当が死に、以後の発言が訳されない")
        self.assertFalse(getattr(b, "_tw_busy", False),
                         "_tw_busy が True のまま(会議を終えるが毎回20秒待つ・ライブ要約が止まる)")
        errs = [e for e in list(b.ui.queue) if e[0] == "error"]
        self.assertTrue(errs, "失敗を画面に知らせていない")


# ---------------------------------------------------------------------------
# 3. 先送り(uplink)が、150秒を超える回線断のあと古い範囲で永久に空回りする
# ---------------------------------------------------------------------------
class TestUplinkAfterLongOutage(unittest.TestCase):
    def test_stale_range_is_dropped_and_new_audio_is_sent(self):
        down = [True]
        posts: list = []

        def post(url, data=None, timeout=None):
            if down[0]:
                raise ConnectionError("tunnel down")
            posts.append((url, len(data)))
            return types.SimpleNamespace(status_code=200, raise_for_status=lambda: None)

        sr = 16000
        up = Uplink("http://x", post=post, batch_s=0.005, keep_s=1.0)
        try:
            up.feed(0, np.full(1000, 0.1, np.float32))           # 回線が落ちている間の声
            time.sleep(0.05)
            up.feed(40000, np.full(1000, 0.1, np.float32))       # 2.5秒後(keep=1秒を超えた)
            down[0] = False                                      # 回線が戻った
            ok = _wait(lambda: any("off=40000" in u for u, _n in posts), 2.0)
            n_before = len(posts)
            time.sleep(0.2)
            n_after = len(posts)
        finally:
            up.close()
        self.assertTrue(ok, f"戻った後も新しい声が送られない: {posts[:5]}")
        self.assertLessEqual(n_after - n_before, 1, "送るものが無いのに送り続けている(空回り)")
        self.assertFalse(any(n == 0 for _u, n in posts), "空の本文を送っている")
        self.assertEqual(up._todo, [])
        self.assertTrue(all(a <= b for a, b in up._acked), f"逆向きの範囲: {up._acked}")
        del sr


# ---------------------------------------------------------------------------
# 4. 会話ごとの裏のスレッドが、次の会話に生き残る
# ---------------------------------------------------------------------------
class _FastTime:
    """__main__ の time だけ差し替える(480秒の間隔を一瞬に)。"""
    def __init__(self):
        self.time = time.time
        self.monotonic = time.monotonic
        self.strftime = time.strftime

    @staticmethod
    def sleep(_s):
        time.sleep(0.01)


class TestPerConversationThreads(unittest.TestCase):
    def test_keepalive_ends_with_its_conversation(self):
        b = _bare_bridge()
        calls: list = []
        b._conv_gen = 1
        b.running.set()
        with mock.patch.object(main_mod, "time", _FastTime()), \
                mock.patch.object(main_mod, "warmup", lambda base, model, timeout=0: calls.append(base)), \
                mock.patch.object(main_mod.requests, "get", lambda *a, **k: None):
            th = threading.Thread(target=b._keepalive, args=("http://old", "m", None, 1),
                                  daemon=True)
            th.start()
            _wait(lambda: calls, 1.0)
            # 「会議を終える」→ すぐ次の会話(処理場所をこの Mac に変えた)
            b.running.clear()
            b._conv_gen = 2
            b.running.set()
            th.join(0.5)
            alive = th.is_alive()
            b.running.clear()
            th.join(1.0)
        self.assertFalse(alive, "前の会話の keepalive が次の会話でも前の処理場所を起こし続ける")

    def test_old_text_worker_does_not_write_into_next_conversation(self):
        b = _bare_bridge()
        b._conv_gen = 1
        b.running.set()
        seen: list = []
        gate = threading.Event()

        def utt(base, model, pair_t, topic, text, lang, t0, tm=None):
            if text == "遅い":
                gate.wait(2.0)        # 訳サーバーが詰まっている
            seen.append(text)

        b._utterance = utt
        old_q: queue.Queue = queue.Queue()
        th = threading.Thread(target=b._text_worker,
                              args=(old_q, "b", "m", ("ja", "en"), "", 1), daemon=True)
        th.start()
        old_q.put(("遅い", "ja", time.time(), {}))
        time.sleep(0.1)
        old_q.put(("前の会議の発言", "ja", time.time(), {}))
        b.running.clear()            # 会議を終える(待ちきれずに次へ)
        b._conv_gen = 2
        b.running.set()              # 次の会話が始まった
        gate.set()
        time.sleep(0.5)
        b.running.clear()
        th.join(1.0)
        self.assertNotIn("前の会議の発言", seen,
                         "前の会話の訳の担当が、次の会話の記録・履歴に前の会議の発言を書いた")


# ---------------------------------------------------------------------------
# 5. 起動中(翻訳モデルの読み込み1〜2分)に「会議を終える」→ 後からマイクが開きっぱなし
# ---------------------------------------------------------------------------
class _FakeCap:
    instances: list = []

    def __init__(self, *a, **k):
        self.paused = threading.Event()
        self.started = False
        self.stopped = False
        self.level = 0.0
        self.uplink = None
        self.tap = None
        self.interim_s = 0.0
        self.segmenter = types.SimpleNamespace(speaking_since=lambda: None, noise=0.01)
        _FakeCap.instances.append(self)

    def start(self):
        self.started = True

    def stop(self):
        self.stopped = True


class _FakeLocal:
    fast = True
    langs = ()


class _FakeSTT:
    def __init__(self, *a, **k):
        self.local = _FakeLocal()
        self.langs = ()
        self.prompt = ""
        self.uplink = None
        self.span_hits = self.span_miss = 0


class _FakeTTS:
    def __init__(self, *a, **k):
        self.stopped = False

    def stop(self):
        self.stopped = True


class TestEndDuringBoot(unittest.TestCase):
    def test_mic_never_left_open(self):
        _FakeCap.instances = []
        b = _bare_bridge()
        d = tempfile.mkdtemp()
        b.s.log_dir = d
        b.s.record_audio = "off"
        loading = threading.Event()
        release = threading.Event()

        def slow_warmup(base, model, timeout=180.0):
            loading.set()
            release.wait(5.0)        # リモートサーバーのモデル載せ(1〜2分)

        patches = [
            mock.patch.object(main_mod, "engine_candidates",
                              lambda *a, **k: [("このMac", "http://mac", "m")]),
            mock.patch.object(main_mod.localllm, "ensure_for", lambda b: None),
            mock.patch.object(main_mod, "warmup", slow_warmup),
            mock.patch.object(main_mod, "pick_model", lambda b, p: "m"),
            mock.patch.object(main_mod, "RemoteSTT", _FakeSTT),
            mock.patch.object(main_mod, "make_local_stt", lambda s: _FakeLocal()),
            mock.patch.object(main_mod, "_warm_ear", lambda stt: None),
            mock.patch.object(main_mod, "TTS", _FakeTTS),
            mock.patch.object(main_mod, "SpeechCapture", _FakeCap),
            mock.patch.object(main_mod, "save_config", lambda s: None),
        ]
        for p in patches:
            p.start()
        try:
            b.start("ja-en", "")
            self.assertTrue(loading.wait(2.0))
            threading.Timer(0.4, release.set).start()   # 会議を終えた後で読み込みが終わる
            r = b.end_meeting(settle_s=0.05, max_s=0.2)
            b._conv_th.join(5.0)
            self.assertFalse(b._conv_th.is_alive())
        finally:
            for p in patches:
                p.stop()
        self.assertEqual(r.get("id"), None)
        opened = [c for c in _FakeCap.instances if c.started and not c.stopped]
        self.assertEqual(opened, [], "会議を終えた後なのにマイクが開いたまま残った")
        self.assertIsNone(b.log, "会議を終えた後なのに新しい記録が開いた")
        self.assertFalse(any(e[0] == "reset" for e in list(b.ui.queue)),
                         "終えた会話のスレッドが画面を起動画面に戻している")


# ---------------------------------------------------------------------------
# 6. マイク: 開けない・コールバックの例外・音が来なくなった(抜いた)が無言
# ---------------------------------------------------------------------------
def _bare_capture() -> SpeechCapture:
    c = SpeechCapture.__new__(SpeechCapture)
    c.sr = 16000
    c._stop = threading.Event()
    c._thread = None
    c.mix_in = None
    c.paused = threading.Event()
    c.errors = []
    c.on_error = lambda *msg: c.errors.append(msg)
    return c


class TestMicFailures(unittest.TestCase):
    def setUp(self):
        self._orig_sd = audio_mod.sd

    def tearDown(self):
        audio_mod.sd = self._orig_sd

    def test_open_failure_is_reported(self):
        class _Err(Exception):
            pass

        def boom(**kw):
            raise _Err("Error querying device -1")

        audio_mod.sd = types.SimpleNamespace(InputStream=boom)
        c = _bare_capture()
        c.start()
        c._thread.join(2.0)
        self.assertTrue(c.errors, "マイクを開けないのに画面には何も出ない(「聞いています」のまま)")

    def test_callback_exception_does_not_reach_portaudio(self):
        holder = {}

        class _Stream:
            def __init__(self, **kw):
                holder["cb"] = kw["callback"]

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

        audio_mod.sd = types.SimpleNamespace(InputStream=_Stream)
        c = _bare_capture()
        n = [0]

        def on_chunk(mono):
            n[0] += 1
            if n[0] == 1:
                raise ValueError("boom")

        c.on_chunk = on_chunk
        c.start()
        self.assertTrue(_wait(lambda: "cb" in holder, 1.0))
        frames = np.zeros((512, 1), np.float32)
        try:
            # sounddevice は callback の例外で paAbort を返し、マイクが以後ずっと止まる
            holder["cb"](frames, 512, None, None)
            raised = False
        except Exception:  # noqa: BLE001
            raised = True
        holder["cb"](frames, 512, None, None)
        c.stop()
        self.assertFalse(raised, "1チャンクの例外が PortAudio へ抜けてマイクが止まる")
        self.assertEqual(n[0], 2)

    def test_silent_stream_is_reported(self):
        holder = {}

        class _Stream:
            def __init__(self, **kw):
                holder["cb"] = kw["callback"]

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

        audio_mod.sd = types.SimpleNamespace(InputStream=_Stream)
        c = _bare_capture()
        c.on_chunk = lambda mono: None
        c.stall_s = 0.3
        c.start()
        self.assertTrue(_wait(lambda: "cb" in holder, 1.0))
        frames = np.zeros((512, 1), np.float32)
        for _ in range(3):
            holder["cb"](frames, 512, None, None)
        # ここでマイクを抜いた: コールバックが来なくなる
        ok = _wait(lambda: c.errors, 2.0)
        c.stop()
        self.assertTrue(ok, "マイクの音が来なくなっても何も知らせない")


class TestMicErrorsReachScreen(unittest.TestCase):
    def test_conversation_wires_capture_errors_to_ui(self):
        src = (Path(main_mod.__file__)).read_text()
        body = src[src.index("def _conversation("):src.index("def _stt_worker(")]
        self.assertIn("self.capture.on_error = lambda title, detail: self.ui.put((\"error\", title, detail))",
                      body)


class TestMixerBounded(unittest.TestCase):
    def test_push_without_mix_does_not_grow_forever(self):
        m = Mixer(16000)
        x = np.zeros(1600, np.float32)
        for _ in range(1000):          # 100秒: マイクが止まっている間もこの Mac の音は届く
            m.push(x)
        self.assertLessEqual(m._n, 16000 * 2, "この Mac の音が際限なく溜まる")
        out = m.mix(np.zeros(512, np.float32))
        self.assertEqual(len(out), 512)


# ---------------------------------------------------------------------------
# 8. 会話中に「終了」: 訳している途中の最後の発言を記録に書き終えてから閉じる(最長5秒)
# ---------------------------------------------------------------------------
def _quit_conversation(b, d: str, slow_last_s: float, stuck: bool = False):
    """会話の骨組み(test_end_meeting と同じ): 最後の発言は text_q にあり、訳の担当が
    slow_last_s 秒かけて記録へ書く。stuck=True なら訳の担当は終わらない(訳サーバーが固まった)。"""
    from tsuyaku.log import SessionLog
    b.log = SessionLog(d, "ja-en", "", "interpret")
    b.log.add("ja", "最初の発言です。", "First.", "en")
    b.capture = _FakeCap()
    b.mode = "interpret"
    b._seg_q, b._text_q = queue.Queue(), queue.Queue()
    b._text_q.put(("最後の発言です。", "ja"))
    b.running.set()
    hang = threading.Event()

    def text_worker():
        while b.running.is_set():
            try:
                text, lang = b._text_q.get(timeout=0.05)
            except queue.Empty:
                continue
            b._tw_busy = True
            if stuck:
                hang.wait(30)
            time.sleep(slow_last_s)
            try:
                b.log.add(lang, text, "Last.", "en")
            except Exception:  # noqa: BLE001 - 閉じた記録に書くと ValueError(=失われた)
                pass
            b._tw_busy = False

    threading.Thread(target=text_worker, daemon=True).start()
    return hang


class TestQuitWaitsForLastLine(unittest.TestCase):
    def test_last_utterance_is_written_before_close(self):
        with tempfile.TemporaryDirectory() as d:
            b = _bare_bridge()
            b.s.log_dir = d
            _quit_conversation(b, d, slow_last_s=0.6)
            path = Path(b.log.path_jsonl)
            b.stop()
            rows = [x for x in path.read_text().splitlines() if '"utter"' in x]
            self.assertEqual(len(rows), 2, "終了で、訳している途中の最後の発言が記録から消えた")
            self.assertTrue(b.capture.stopped)

    def test_quit_never_hangs_more_than_limit(self):
        with tempfile.TemporaryDirectory() as d:
            b = _bare_bridge()
            b.s.log_dir = d
            hang = _quit_conversation(b, d, slow_last_s=0.0, stuck=True)
            t0 = time.time()
            b._shutdown(max_wait_s=0.6)
            dt = time.time() - t0
            hang.set()
            self.assertLess(dt, 1.5, f"訳の担当が固まっていると終了が固まる({dt:.1f}秒)")

    def test_default_limit_is_five_seconds(self):
        import inspect
        self.assertEqual(inspect.signature(Bridge._shutdown).parameters["max_wait_s"].default, 5.0)


# ---------------------------------------------------------------------------
# 9. マイクの許可が無い: macOS は完全な 0 を渡し続ける
# ---------------------------------------------------------------------------
class TestMicPermissionZeros(unittest.TestCase):
    def setUp(self):
        self._orig_sd = audio_mod.sd
        holder = self.holder = {}

        class _Stream:
            def __init__(self, **kw):
                holder["cb"] = kw["callback"]

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

        audio_mod.sd = types.SimpleNamespace(InputStream=_Stream)
        self.c = _bare_capture()
        self.c.on_chunk = lambda mono: None
        self.c.start()
        self.assertTrue(_wait(lambda: "cb" in holder, 1.0))

    def tearDown(self):
        self.c.stop()
        audio_mod.sd = self._orig_sd

    def _feed(self, x, seconds):
        for _ in range(int(seconds * 16000 / 512) + 1):
            self.holder["cb"](x, 512, None, None)

    def test_exact_zeros_for_three_seconds_reported_once(self):
        self._feed(np.zeros((512, 1), np.float32), 3.1)
        self._feed(np.zeros((512, 1), np.float32), 5.0)
        hits = [e for e in self.c.errors if "許可" in e[0]]
        self.assertEqual(len(hits), 1, f"完全な 0 が続くのに知らせない/何度も出す: {self.c.errors}")
        self.assertIn("プライバシーとセキュリティ", hits[0][1])

    def test_shorter_than_three_seconds_not_reported(self):
        self._feed(np.zeros((512, 1), np.float32), 2.5)
        self._feed(np.full((512, 1), 1e-3, np.float32), 0.1)
        self._feed(np.zeros((512, 1), np.float32), 2.5)
        self.assertEqual(self.c.errors, [])

    def test_quiet_real_mic_noise_not_reported(self):
        rng = np.random.default_rng(0)
        for _ in range(int(6 * 16000 / 512)):
            x = (rng.standard_normal((512, 1)) * 1e-6).astype(np.float32)
            self.holder["cb"](x, 512, None, None)
        self.assertEqual(self.c.errors, [], "本物のマイクの小さな雑音を「許可が無い」と誤検知した")


# ---------------------------------------------------------------------------
# 10. 起動失敗の経路で記録を閉じる
# ---------------------------------------------------------------------------
class TestRunFailureClosesLog(unittest.TestCase):
    def test_log_closed_when_conversation_raises(self):
        b = _bare_bridge()
        b._rec_stop = lambda *a, **k: None

        class _Log:
            closed = False

            def close(self):
                self.closed = True

        log = _Log()

        def boom(pair, topic, mode="interpret"):
            b.log = log
            raise RuntimeError("silero_vad.onnx がありません")

        b._conversation = boom
        b._run("ja-en", "")
        self.assertTrue(log.closed, "起動失敗の後も記録のファイルが開いたまま")
        self.assertIsNone(b.log)


# ---------------------------------------------------------------------------
# 11. 録音の書き込み係が死んだら、tap は溜めずに捨てる
# ---------------------------------------------------------------------------
class TestRecorderWriterDead(unittest.TestCase):
    def test_tap_drops_when_writer_is_dead(self):
        import contextlib
        import io
        from tsuyaku.recorder import Recorder
        with tempfile.TemporaryDirectory() as d:
            r = Recorder(Path(d) / "x.audio.caf", ts0=time.time())
            r._q.put(None)                      # 書き込み係が終わった(死んだ)
            r._th.join(2.0)
            self.assertFalse(r._th.is_alive())
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                for _ in range(2000):            # 約64秒分
                    r.tap(np.zeros(512, np.float32))
            n = r._q.qsize()
            r.close()
        self.assertLess(n, 50, f"書き込み係が死んだ後も録音が際限なく溜まる({n}個)")
        self.assertEqual(out.getvalue().count("[rec]"), 1, f"ログは1回だけ: {out.getvalue()!r}")


if __name__ == "__main__":
    unittest.main()
