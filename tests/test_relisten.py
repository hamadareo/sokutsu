"""このMacの耳(SenseVoice)が落とした略語を裏で聞き直す(11/08、リーダー判断「先に出して、後から直す」)。

見つけた問題: SenseVoice は日本語の中の英字の略語を書かない(作った74文の略語42個が0個)。落ちた語は
文字の直しでは戻らない。このMacの whisper なら戻るが1回2.4〜2.7秒かかるので、先に出して裏で直す。
"""
import json
import queue
import tempfile
import threading
import time
import unittest
from pathlib import Path

import numpy as np

from tsuyaku.relisten import Relistener, dropped_letters, recovered


class TestDropped(unittest.TestCase):
    def test_signs_of_dropped_acronym(self):
        for s in ("のファインチューニングにはが8枚必要です。",        # 文頭が助詞
                  "は画像認識に、は時系列データに向いています。",
                  "このはで結果を返すので、簡単に呼び出せます。",      # この + 助詞
                  "当社はの資金を調達しました。",                     # 助詞が2つ
                  "データは、でゲートウェイに送られます。"):           # 読点の後が助詞
            self.assertTrue(dropped_letters(s, "ja"), s)

    def test_ordinary_sentences(self):
        for s in ("今日はいい天気ですね。", "では、始めましょう。", "とても良いと思います。",
                  "そのとおりです。", "このへんで終わりにします。", "へー、そうなんですか。",
                  "もう少し待ってください。", "はい、わかりました。", "資料はメールで送ります。",
                  "それでは会議を始めます。", "このほかに質問はありますか。"):
            self.assertFalse(dropped_letters(s, "ja"), s)

    def test_only_japanese(self):
        self.assertFalse(dropped_letters("は画像認識に", "zh"))
        self.assertFalse(dropped_letters("", "ja"))

    def test_recovered(self):
        self.assertTrue(recovered("のファインチューニングにはが8枚必要です。",
                                  "LLMのファインチューニングにはGPUが8枚必要です。"))
        self.assertFalse(recovered("のファインチューニングにはが8枚必要です。",
                                   "のファインチューニングにはが8枚必要です。"))     # 戻っていない
        self.assertFalse(recovered("のファインチューニングにはが8枚必要です。",
                                   "Thank you for watching."))                         # 別の文(幻覚)


class FakeEar:
    def __init__(self, out, delay=0.0):
        self.out, self.delay, self.calls = out, delay, 0

    def transcribe(self, pcm):
        self.calls += 1
        time.sleep(self.delay)
        return self.out, "ja", 0.9


def wait(cond, t=3.0):
    t0 = time.time()
    while not cond() and time.time() - t0 < t:
        time.sleep(0.01)
    return cond()


class TestRelistener(unittest.TestCase):
    def job(self, orig="のファインチューニングにはが8枚必要です。"):
        return {"uid": 1, "orig": orig, "lang": "ja", "pcm": np.zeros(16000, np.float32)}

    def test_fixed(self):
        fixed, done = [], []
        r = Relistener(lambda: FakeEar("LLMのファインチューニングにはGPUが8枚必要です。"),
                       lambda j, t: fixed.append((j["uid"], t)), lambda j: done.append(j["uid"]))
        r.submit(self.job())
        self.assertTrue(wait(lambda: fixed))
        self.assertEqual(fixed[0], (1, "LLMのファインチューニングにはGPUが8枚必要です。"))
        self.assertEqual(done, [])
        self.assertEqual(r.stats["fixed"], 1)
        r.close()

    def test_kept_calls_done(self):
        fixed, done = [], []
        r = Relistener(lambda: FakeEar("のファインチューニングにはが8枚必要です。"),
                       lambda j, t: fixed.append(t), lambda j: done.append(j["uid"]))
        r.submit(self.job())
        self.assertTrue(wait(lambda: done))
        self.assertEqual(fixed, [])
        r.close()

    def test_waits_while_busy(self):
        busy = threading.Event()
        busy.set()
        ear = FakeEar("LLMのファインチューニングにはGPUが8枚必要です。")
        fixed = []
        r = Relistener(lambda: ear, lambda j, t: fixed.append(t), lambda j: None, busy=busy.is_set)
        r.submit(self.job())
        time.sleep(0.3)
        self.assertEqual(ear.calls, 0)           # 新しい発言の聞き取り・訳の間は聞かない
        busy.clear()
        self.assertTrue(wait(lambda: fixed))
        r.close()

    def test_stale_job_dropped(self):
        done = []
        ear = FakeEar("LLMのファインチューニングにはGPUが8枚必要です。")
        r = Relistener(lambda: ear, lambda j, t: None, lambda j: done.append(1), max_age_s=0.0)
        j = self.job()
        j["t_submit"] = time.time() - 5
        r.submit(j)
        self.assertTrue(wait(lambda: done))
        self.assertEqual(ear.calls, 0)
        r.close()


class TestRecordEdit(unittest.TestCase):
    def test_edit_row_replaces_orig_and_trans(self):
        from tsuyaku.log import SessionLog
        from tsuyaku.notes import load_meeting
        with tempfile.TemporaryDirectory() as d:
            log = SessionLog(d, "ja-zh", "", "interpret")
            log.add("ja", "最初の文です。", "第一句。", "zh")
            log.add("ja", "のファインチューニングにはが必要です。", "微调需要。", "zh")
            log.add_row({"type": "edit", "i": 1, "orig": "LLMのファインチューニングにはGPUが必要です。",
                         "trans": "LLM的微调需要GPU。", "by": "relisten"})
            log.close()
            m = load_meeting(Path(log.path_jsonl))
            self.assertEqual(m["lines"][1]["orig"], "LLMのファインチューニングにはGPUが必要です。")
            self.assertEqual(m["lines"][1]["trans"], "LLM的微调需要GPU。")
            self.assertEqual(m["lines"][0]["trans"], "第一句。")


class TestHumanEditWins(unittest.TestCase):
    """人の修正(line_edit)は聞き直しより強い(リーダーのレビュー 11/08)。"""

    def bridge_with_log(self, d):
        from tsuyaku.__main__ import Bridge
        from tsuyaku.log import SessionLog
        b = Bridge()
        b.s.log_dir = d
        b.log = SessionLog(d, "ja-zh", "", "notes")
        b.log.add("ja", "のファインチューニングには。", "", None)
        b.log.add("ja", "は画像認識に。", "", None)
        b.ui = queue.Queue()
        return b

    def job(self, b, i):
        return {"uid": i + 1, "i": i, "orig": "x", "lang": "ja", "log": b.log}

    def test_human_edit_in_file_blocks_relisten(self):
        with tempfile.TemporaryDirectory() as d:
            b = self.bridge_with_log(d)
            self.assertFalse(b._relisten_human_edited(self.job(b, 0)))
            b.log.add_row({"type": "edit", "i": 0, "orig": "人が直した文です。"})   # by の無い edit = 人
            self.assertTrue(b._relisten_human_edited(self.job(b, 0)))
            self.assertFalse(b._relisten_human_edited(self.job(b, 1)))
            b.log.close()

    def test_relisten_edit_is_not_human(self):
        with tempfile.TemporaryDirectory() as d:
            b = self.bridge_with_log(d)
            b.log.add_row({"type": "edit", "i": 0, "orig": "LLMの…", "by": "relisten"})
            self.assertFalse(b._relisten_human_edited(self.job(b, 0)))
            b.log.close()

    def test_line_edit_drops_pending_relisten(self):
        """人が直した直後(記録に書く前の聞き直しの結果)も、記録には書かず印だけ外す。"""
        with tempfile.TemporaryDirectory() as d:
            b = self.bridge_with_log(d)
            mid = Path(b.log.path_jsonl).stem
            self.assertIsNotNone(b.line_edit(mid, 1, "人が直した文です。"))
            self.assertIn(1, b.__dict__.get("_human_edited", set()))
            b._relisten_fixed(self.job(b, 1), "CNNは画像認識に。")
            rows = [json.loads(x) for x in Path(b.log.path_jsonl).read_text().splitlines()]
            self.assertEqual([r for r in rows if r.get("by") == "relisten"], [])
            evs = []
            while not b.ui.empty():
                evs.append(b.ui.get())
            self.assertIn(("fix", {"id": 2, "done": True}), evs)
            b.log.close()


class TestUndoSkipsRelisten(unittest.TestCase):
    def test_undo_takes_back_the_human_edit(self):
        from tsuyaku.log import SessionLog
        from tsuyaku.notes import load_meeting
        with tempfile.TemporaryDirectory() as d:
            log = SessionLog(d, "ja-zh", "", "notes")
            log.add("ja", "一行目。", "", None)
            log.add("ja", "は画像認識に。", "", None)
            log.add_row({"type": "edit", "i": 0, "orig": "人が直した一行目。"})
            log.add_row({"type": "edit", "i": 1, "orig": "CNNは画像認識に。", "by": "relisten"})
            log.flush()
            m = load_meeting(Path(log.path_jsonl))
            self.assertEqual(m["undo"], 1)                 # 元に戻せるのは人の修正だけ
            log.add_row({"type": "undo"})
            log.close()
            m = load_meeting(Path(log.path_jsonl))
            self.assertEqual(m["lines"][0]["orig"], "一行目。")             # 人の修正が戻る
            self.assertEqual(m["lines"][1]["orig"], "CNNは画像認識に。")    # 聞き直しは残る
            self.assertEqual(m["undo"], 0)


class TestEditUndoSkipsRelisten(unittest.TestCase):
    def test_edit_undo_after_relisten(self):
        """聞き直しの後に「元に戻す」: 人の修正が取り消され、聞き直しの略語は残る。人の修正が無ければ何もしない。"""
        from tsuyaku.__main__ import Bridge
        from tsuyaku.log import SessionLog
        with tempfile.TemporaryDirectory() as d:
            b = Bridge()
            b.s.log_dir = d
            b.log = SessionLog(d, "ja-zh", "", "notes")
            b.log.add("ja", "一行目。", "", None)
            b.log.add("ja", "は画像認識に。", "", None)
            mid = Path(b.log.path_jsonl).stem
            b.log.add_row({"type": "edit", "i": 1, "orig": "CNNは画像認識に。", "by": "relisten"})
            self.assertIsNone(b.edit_undo(mid))                      # 人の修正が無い: 聞き直しは取り消さない
            b.line_edit(mid, 0, "人が直した一行目。")
            b.log.add_row({"type": "edit", "i": 1, "orig": "CNNは画像認識に、です。", "by": "relisten"})
            r = b.edit_undo(mid)
            self.assertEqual(r["i"], 0)
            self.assertEqual(r["line"]["orig"], "一行目。")
            from tsuyaku.notes import load_meeting
            m = load_meeting(Path(b.log.path_jsonl))
            self.assertEqual(m["lines"][1]["orig"], "CNNは画像認識に、です。")
            self.assertEqual(m["undo"], 0)
            b.log.close()


class TestUiRespectsHumanEdit(unittest.TestCase):
    def test_fix_skips_edited_rows(self):
        html = (Path(__file__).resolve().parents[1] / "src/tsuyaku/webui/index.html").read_text()
        blk = html[html.index("<!-- ===== terms-relisten"):html.index("<!-- ===== /terms-relisten -->")]
        self.assertIn("!el.isContentEditable", blk)       # ダブルクリックで直している最中
        self.assertIn('querySelector(".w3ed")', blk)       # 手で直した印(修正済み)のある行
        self.assertNotIn("outerHTML", blk)                 # 行は作り直さない
        self.assertNotIn(".remove();it", blk)


class TestWiring(unittest.TestCase):
    SRC = (Path(__file__).resolve().parents[1] / "src/tsuyaku/__main__.py").read_text()

    def test_only_with_mac_sensevoice(self):
        self.assertIn("stt_url is None and getattr(self.stt.local, \"fast\", False)", self.SRC)

    def test_unit_audio_passed(self):
        self.assertIn('"pcm": u.audio', self.SRC)

    def test_ui(self):
        html = (Path(__file__).resolve().parents[1] / "src/tsuyaku/webui/index.html").read_text()
        self.assertIn("terms-relisten", html)
        self.assertIn("T.fix=function(d)", html)
        self.assertIn('"checking": checking', self.SRC)


if __name__ == "__main__":
    unittest.main()
