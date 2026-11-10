"""終了ボタンと、会話開始時の読み込みステータスバー(11/05 ユーザー要望)。

- 終了: これまで Esc しか無く、画面上に終わり方が見えなかった。会話画面の
  ステータス行と起動画面に「終了」を置く。会議中の誤クリックで会話が消えない
  よう、会話中は2回押し(1回目で「もう一度押すと終了」)。終了時はログを閉じて
  確実に書き出す。
- 読み込みバー: 「会話を始める」後の待ち(リモートサーバーの初回モデル載せは1〜2分)が
  文字1行だけで、どこまで進んだか分からなかった。翻訳モデル→耳→声→マイクの
  4段を、Python 側が段ごとに ("boot", …) イベントで UI へ送り、段付きバーで出す。

実行: PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -v
"""
from __future__ import annotations

import json
import queue
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MAIN = (ROOT / "src/tsuyaku/__main__.py").read_text()
HTML = (ROOT / "src/tsuyaku/webui/index.html").read_text()


class _Win:
    def __init__(self):
        self.js: list[str] = []
        self.destroyed = False

    def evaluate_js(self, js):
        self.js.append(js)

    def destroy(self):
        self.destroyed = True


class _Log:
    closed = False

    def close(self):
        self.closed = True


def _bare_bridge():
    from tsuyaku.__main__ import Bridge
    import threading
    b = Bridge.__new__(Bridge)
    b.window = _Win()
    b.ui = queue.Queue()
    b.running = threading.Event()
    b.closing = threading.Event()
    b.capture = b.tts = None
    b.log = None
    return b


class TestBootBar(unittest.TestCase):
    def test_push_boot_event_reaches_ui(self):
        b = _bare_bridge()
        b._push(("boot", {"step": 1, "total": 4, "label": "耳"}))
        self.assertEqual(len(b.window.js), 1)
        js = b.window.js[0]
        self.assertTrue(js.startswith("TSUYAKU.boot("), js)
        payload = json.loads(js[len("TSUYAKU.boot("):-1])
        self.assertEqual(payload["step"], 1)
        self.assertEqual(payload["total"], 4)

    def test_conversation_reports_steps_in_order(self):
        i = MAIN.index("def _conversation(")
        j = MAIN.index("def _stt_worker(")
        body = MAIN[i:j]
        # 翻訳モデル(0) → 耳(1) → 声(2) → マイク(3) → 完了(4) の順で送る
        idx = [body.index(f"self._boot({k}") for k in range(5)]
        self.assertEqual(idx, sorted(idx))
        # それぞれ、実際に時間がかかる処理の直前で送る
        self.assertLess(body.index("self._boot(0"), body.index("warmup(b, pref)"))
        self.assertLess(body.index("self._boot(1"), body.index("make_local_stt(s)"))
        self.assertLess(body.index("self._boot(2"), body.index("self.tts = None if mode == \"notes\" else _shared_tts(s)"))
        self.assertLess(body.index("self._boot(3"), body.index("SpeechCapture("))
        self.assertLess(body.index("self.capture.start()"), body.index("self._boot(4"))

    def test_ui_has_segmented_boot_bar(self):
        self.assertIn('id="boot"', HTML)
        self.assertIn("boot(d)", HTML)
        # 4段のラベルが画面にある
        for lab in ("翻訳モデル", "音声認識", "読み上げの声", "マイク"):
            self.assertIn(lab, HTML)
        # 経過秒を見せる(初回1〜2分の待ちで止まっていないと分かる)
        self.assertIn('id="bootTime"', HTML)
        # 失敗して起動画面に戻るときはバーを消す
        k = HTML.index("reset(){")
        self.assertIn("bootHide", HTML[k:k + 600])


class TestQuitButton(unittest.TestCase):
    def test_quit_buttons_exist(self):
        self.assertIn('id="quit"', HTML)        # 会話画面のステータス行
        self.assertIn('id="quitStart"', HTML)   # 起動画面
        # ステータス行の右端グループの中(声・速度の並び)
        a = HTML.index('id="statusRight"')
        self.assertLess(a, HTML.index('id="quit"'))
        self.assertLess(HTML.index('id="gear"'), HTML.index('id="quit"'))

    def test_quit_in_conversation_needs_second_press(self):
        k = HTML.index("quitPress(")
        self.assertIn("もう一度押すと終了", HTML)
        self.assertIn("api.stop()", HTML[k:])

    def test_stop_closes_log_and_window(self):
        b = _bare_bridge()
        log = _Log()
        b.log = log
        b.stop()
        self.assertTrue(log.closed)
        self.assertTrue(b.window.destroyed)
        self.assertTrue(b.closing.is_set())

    def test_process_really_exits_after_window_closes(self):
        # 窓を閉じても音声/whisper のスレッドが残ってプロセスが居座らないように
        i = MAIN.index("def main(")
        tail = MAIN[i:]
        self.assertLess(tail.index("webview.start("), tail.index("os._exit(0)"))


if __name__ == "__main__":
    unittest.main()


class TestWakeEar(unittest.TestCase):
    """11/05 ライブ: アプリ起動から会話開始まで30分以上空き、リモートサーバーの耳(30分
    アイドルで自分で終了)が止まっていて『耳=このMac』(認識2.3秒)に落ちた。
    音声認識の段で応答が無ければ、とんねるスクリプトでその場で起こす。"""

    def test_wake_runs_tunnel_script_then_rechecks(self):
        import tsuyaku.__main__ as m
        calls = []
        seq = iter([False, True])
        orig_r, orig_run = m._reachable_stt, m.subprocess.run
        try:
            m._reachable_stt = lambda url, timeout=1.5: next(seq)
            m.subprocess.run = lambda cmd, **kw: calls.append(cmd)
            self.assertTrue(m._wake_stt("http://127.0.0.1:12346"))
        finally:
            m._reachable_stt, m.subprocess.run = orig_r, orig_run
        self.assertEqual(len(calls), 1)
        self.assertTrue(str(calls[0][-1]).endswith("scripts/remote_tunnel.sh"))

    def test_wake_skips_script_when_ear_is_up(self):
        import tsuyaku.__main__ as m
        calls = []
        orig_r, orig_run = m._reachable_stt, m.subprocess.run
        try:
            m._reachable_stt = lambda url, timeout=1.5: True
            m.subprocess.run = lambda cmd, **kw: calls.append(cmd)
            self.assertTrue(m._wake_stt("http://127.0.0.1:12346"))
        finally:
            m._reachable_stt, m.subprocess.run = orig_r, orig_run
        self.assertEqual(calls, [])

    def test_conversation_wakes_ear_in_stt_step(self):
        i = MAIN.index("def _conversation(")
        body = MAIN[i:MAIN.index("def _stt_worker(")]
        self.assertLess(body.index("self._boot(1"), body.index("_wake_stt("))
        self.assertLess(body.index("_wake_stt("), body.index("RemoteSTT("))


class TestOpenLog(unittest.TestCase):
    """記録レールの「ファイル」: 今の会話の .txt を Finder で見せる(11/06)。"""

    def test_open_log_reveals_txt(self):
        import tempfile
        from unittest.mock import patch
        from tsuyaku.__main__ import Bridge
        from tsuyaku.log import SessionLog
        with tempfile.TemporaryDirectory() as d:
            b = Bridge()
            b.log = SessionLog(d, "ja-en")
            with patch("tsuyaku.__main__.subprocess.run") as run:
                self.assertEqual(b.open_log(), "ok")
            args = run.call_args[0][0]
            self.assertEqual(args[:2], ["open", "-R"])
            self.assertTrue(args[2].endswith(".txt"))
            b.log.close()

    def test_open_log_without_session_opens_folder(self):
        import tempfile
        from unittest.mock import patch
        from tsuyaku.__main__ import Bridge
        with tempfile.TemporaryDirectory() as d:
            b = Bridge()
            b.log = None
            b.s.log_dir = d
            with patch("tsuyaku.__main__.subprocess.run") as run:
                self.assertEqual(b.open_log(), "none")
            self.assertEqual(run.call_args[0][0], ["open", d])
