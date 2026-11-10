"""「会議を終える」(11/06 ユーザー「ミーティングメモは終了ボタンありますか？」)。

見つけた問題: 会話を終える手段が「終了」(アプリごと閉じる)しかなく、会議メモは次に起動して
「会議の記録」から作るしかなかった。
変更: Bridge.end_meeting() — 録音を止め、聞き取り・訳の途中の最後の発言を記録に書き終えるまで待ち、
話者を付け直して記録を閉じ、その会議の id を返す(画面はそこで起動画面に戻り、メモを作って開く)。
"""
import json
import queue
import tempfile
import threading
import time
import types
import unittest
from pathlib import Path

from tsuyaku.__main__ import Bridge
from tsuyaku.log import SessionLog


class FakeCapture:
    def __init__(self):
        self.paused = threading.Event()
        self.stopped = False
        self.level = 0.0

    def stop(self):
        self.stopped = True


class FakeSpk:
    def refine(self):
        return ["S1", "S1"]


def fake_conversation(b: Bridge, d: str, slow_last_s: float):
    """_conversation の骨組み: 記録・録音・行列を用意し、running の間待つ。
    最後の発言は text_q に入っていて、訳の担当が slow_last_s 秒かけて記録へ書く。"""
    b.log = SessionLog(d, "ja-en", "", "notes")
    b.log.add("ja", "最初の発言です。", "", None, spk="S1", segs=[0])
    b.capture = FakeCapture()
    b.spk = FakeSpk()
    b.mode = "notes"
    b._seg_q, b._text_q = queue.Queue(), queue.Queue()
    b._text_q.put(("最後の発言です。", "ja"))
    b.running.set()

    def text_worker():
        while b.running.is_set():
            try:
                text, lang = b._text_q.get(timeout=0.05)
            except queue.Empty:
                continue
            b._tw_busy = True
            time.sleep(slow_last_s)          # 訳・記録に時間がかかる
            b.log.add(lang, text, "", None, spk="S1", segs=[1])
            b._tw_busy = False

    tw = threading.Thread(target=text_worker, daemon=True)
    tw.start()

    def conv():
        while b.running.is_set():
            time.sleep(0.05)
        tw.join(2)
    b._conv_th = threading.Thread(target=conv, daemon=True)
    b._conv_th.start()


class TestEndMeeting(unittest.TestCase):
    def test_waits_for_last_line_then_closes(self):
        with tempfile.TemporaryDirectory() as d:
            b = Bridge()
            b.s.log_dir = d
            fake_conversation(b, d, slow_last_s=0.8)
            path = Path(b.log.path_jsonl)
            cap = b.capture
            r = b.end_meeting(settle_s=0.3)
            self.assertEqual(r["id"], path.stem)
            self.assertEqual(r["lines"], 2)               # 最後の発言も書き終えてから
            self.assertEqual(r["mode"], "notes")
            self.assertTrue(cap.stopped)
            self.assertFalse(b.running.is_set())
            self.assertIsNone(b.log)                       # 「今の会話」はもう無い
            self.assertIsNone(b.capture)
            rows = [json.loads(x) for x in path.read_text().splitlines()]
            self.assertTrue(any(x.get("type") == "refine" for x in rows))   # 話者の付け直し
            self.assertIsNotNone(b._meeting_path(r["id"]))  # 一覧から開ける

    def test_not_running(self):
        b = Bridge()
        self.assertIsNone(b.end_meeting()["id"])

    def test_next_conversation_refines_again(self):
        """終えた後の次の会話でも、終わりに話者を付け直す(1回きりの旗を戻す)。"""
        src = (Path(__file__).resolve().parents[1] / "src/tsuyaku/__main__.py").read_text()
        body = src[src.index("    def _conversation("):]
        body = body[:body.index("\n    def ", 10)]
        self.assertIn("self._refined_end = False", body)
        self.assertIn("self._seg_q", body)
        self.assertIn("self._text_q", body)


class TestUi(unittest.TestCase):
    def test_button_wired(self):
        html = (Path(__file__).resolve().parents[1] / "src/tsuyaku/webui/index.html").read_text()
        self.assertIn('id="endBtn"', html)
        self.assertIn("end_meeting", html)


if __name__ == "__main__":
    unittest.main()
