"""W4 の画面向け API(Bridge が LibraryApi を継承): 整理・検索・分析・書き出し・声を覚える・カレンダー。"""
import json
import queue
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from tsuyaku import calendar_mac as C
from tsuyaku import notes as N
from tsuyaku import voicebook as V
from tsuyaku.__main__ import Bridge

RNG = np.random.default_rng(1)


def unit(v):
    v = np.asarray(v, np.float32)
    return v / np.linalg.norm(v)


YAMADA, SMITH = unit(RNG.normal(size=192)), unit(RNG.normal(size=192))


def near(v, k=0.3):
    return unit(v + k * unit(RNG.normal(size=192)))


def write_log(d, stem, start="2026-11-06 10:00:00"):
    p = Path(d) / f"{stem}.jsonl"
    rows = [{"t": "10:00:05", "lang": "ja", "orig": "予算の話をしましょう。", "spk": "S1", "t0": 1.0, "dur": 3.0},
            {"t": "10:00:12", "lang": "en", "orig": "I will send the quotation.", "spk": "S2", "t0": 5.0, "dur": 2.0}]
    with p.open("w", encoding="utf-8") as f:
        f.write(json.dumps({"type": "meta", "pair": "ja-en", "mode": "notes", "start": start}) + "\n")
        for r in rows:
            f.write(json.dumps(dict({"type": "utter"}, **r), ensure_ascii=False) + "\n")
    return p


class Base(unittest.TestCase):
    def setUp(self):
        self.td = tempfile.TemporaryDirectory()
        self.d = Path(self.td.name)
        self.b = Bridge()
        self.b.s.log_dir = str(self.d)
        self.a = write_log(self.d, "session-20261106-100000")
        self.c = write_log(self.d, "session-20261107-100000", start="2026-11-07 10:00:00")

    def tearDown(self):
        self.td.cleanup()


class TestOrganize(Base):
    def test_list_info_trash_restore(self):
        self.b.meeting_info_set("session-20261106-100000", {"title": "定例", "tags": ["予算"], "folder": "チーム",
                                                            "star": True, "evil": 1})
        r = self.b.library_list()
        m = {x["id"]: x for x in r["meetings"]}["session-20261106-100000"]
        self.assertEqual((m["title"], m["tags"], m["folder"], m["star"]), ("定例", ["予算"], "チーム", True))
        self.assertEqual(r["facets"]["folders"], [{"name": "チーム", "n": 1}])
        self.assertEqual(self.b.meeting_trash("session-20261106-100000"), "ok")
        self.assertEqual([x["id"] for x in self.b.trash_list()], ["session-20261106-100000"])
        self.assertEqual(self.b.meeting_restore("session-20261106-100000"), "ok")
        self.assertEqual(self.b.trash_empty("x"), -1)
        self.assertEqual(self.b.meeting_trash("live"), "live")
        self.assertIsNone(self.b.meeting_info_set("../../etc", {"title": "x"}))

    def test_search_adds_titles(self):
        self.b.meeting_info_set("session-20261107-100000", {"title": "新しい定例"})
        r = self.b.search_all("quotation")
        self.assertEqual([m["id"] for m in r["meetings"]], ["session-20261107-100000", "session-20261106-100000"])
        self.assertEqual(r["meetings"][0]["title"], "新しい定例")

    def test_stats(self):
        st = self.b.meeting_stats("session-20261106-100000")
        self.assertEqual({s["id"]: s["secs"] for s in st["speakers"]}, {"S1": 3.0, "S2": 2.0})


class TestExport(Base):
    def test_export_docx_and_pdf_to_the_chosen_path(self):
        N.save_notes(self.a, {"lang": "ja", "title": "定例", "summary": ["要約"], "todos": [{"task": "送る"}],
                              "participants": [{"id": "S1", "name": "山田"}]})
        out_docx, out_pdf = self.d / "out.docx", self.d / "out"
        with patch.object(Bridge, "_save_path", side_effect=[str(out_docx), str(out_pdf)]), \
                patch("subprocess.run") as run:
            run.side_effect = lambda *a, **k: __import__("subprocess").CompletedProcess(a, 0, "", "")
            self.assertEqual(self.b.meeting_export("session-20261106-100000", "ja", "docx"), str(out_docx))
        self.assertTrue(out_docx.read_bytes().startswith(b"PK"))
        with patch.object(Bridge, "_save_path", return_value=str(out_pdf)), patch("subprocess.run"):
            with patch("tsuyaku.bridge_library.export_mod.pdf_subprocess") as pdf:
                got = self.b.meeting_export("session-20261106-100000", "ja", "pdf", True)
        self.assertEqual(got, str(out_pdf) + ".pdf")                       # 拡張子を足す
        doc = pdf.call_args[0][0]
        self.assertEqual(doc["title"], "定例")
        self.assertIn("transcript", [b[0] for b in doc["blocks"]])

    def test_cancelled_dialog_writes_nothing(self):
        with patch.object(Bridge, "_save_path", return_value=None):
            self.assertEqual(self.b.meeting_export("session-20261106-100000", "ja", "pdf"), "")


class TestVoices(Base):
    def test_name_in_one_meeting_then_auto_label_in_the_next_and_undo(self):
        V.save_meeting_voices(self.a, {"S1": (near(YAMADA), 40.0, 12), "S2": (near(SMITH), 30.0, 9)})
        self.b.speaker_rename("session-20261106-100000", "S1", "山田")      # 利用者が名前を付けた → 覚える
        self.assertEqual([p["name"] for p in self.b.voicebook_list()], ["山田"])
        # 次の会議: S2 が山田の声
        V.save_meeting_voices(self.c, {"S1": (near(SMITH), 30.0, 9), "S2": (near(YAMADA), 50.0, 14)})
        r = self.b.voices_meeting("session-20261107-100000")
        self.assertEqual(r["matches"]["S2"]["name"], "山田")
        self.assertEqual(r["matches"]["S2"]["state"], "auto")
        self.assertNotIn("S1", r["matches"])
        self.assertEqual(N.load_speakers(self.c), {"S2": "山田"})
        self.assertEqual(len(self.b.voicebook_list()[0]["meetings"]), 1)    # 自動の名前からは覚えない
        # 取り消す → 名前が消え、開き直しても付かない
        self.assertEqual(self.b.voice_reject("session-20261107-100000", "S2"), "ok")
        self.assertEqual(N.load_speakers(self.c), {})
        self.b.voices_meeting("session-20261107-100000")
        self.assertEqual(N.load_speakers(self.c), {})

    def test_accept_adds_a_sample_and_rename_moves_it(self):
        V.save_meeting_voices(self.a, {"S1": (near(YAMADA), 40.0, 12)})
        self.b.speaker_rename("session-20261106-100000", "S1", "山田")
        V.save_meeting_voices(self.c, {"S2": (near(YAMADA), 50.0, 14)})
        self.b.voices_meeting("session-20261107-100000")
        self.assertEqual(self.b.voice_accept("session-20261107-100000", "S2"), "ok")
        self.assertEqual(self.b.voicebook_list()[0]["samples"], 2)
        self.assertEqual(V.load_matches(self.c)["S2"]["state"], "accepted")
        self.b.speaker_rename("session-20261106-100000", "S1", "")          # 名前を消す → 見本も消える
        self.assertEqual(self.b.voicebook_list()[0]["samples"], 1)

    def test_user_correction_of_an_auto_name_is_learned(self):
        V.save_meeting_voices(self.a, {"S1": (near(YAMADA), 40.0, 12)})
        self.b.speaker_rename("session-20261106-100000", "S1", "山田")
        V.save_meeting_voices(self.c, {"S2": (near(YAMADA), 50.0, 14)})
        self.b.voices_meeting("session-20261107-100000")
        self.b.speaker_rename("session-20261107-100000", "S2", "山田さん")     # 違う人だった
        m = V.load_matches(self.c)["S2"]
        self.assertEqual(m["state"], "rejected")
        self.assertEqual(sorted(p["name"] for p in self.b.voicebook_list()), ["山田", "山田さん"])

    def test_short_voice_is_not_enrolled(self):
        V.save_meeting_voices(self.a, {"S1": (near(YAMADA), V.MIN_ENROLL_S - 2, 2)})
        self.b.speaker_rename("session-20261106-100000", "S1", "山田")
        self.assertEqual(self.b.voicebook_list(), [])
        r = self.b.voices_meeting("session-20261106-100000")
        self.assertFalse(r["voices"]["S1"]["ok"])

    def test_live_refine_saves_voices_and_pushes_names(self):
        V.save_meeting_voices(self.a, {"S1": (near(YAMADA), 40.0, 12)})
        self.b.speaker_rename("session-20261106-100000", "S1", "山田")

        class FakeTracker:
            def centroids(self, ids):
                return {"S1": (near(SMITH), 30.0, 9), "S3": (near(YAMADA), 45.0, 13)}

            def refine(self):
                return ["S1", "S3"]

        class FakeLog:
            path_jsonl = str(self.c)

            def flush(self):
                pass
        self.b.spk, self.b.log, self.b.ui = FakeTracker(), FakeLog(), queue.Queue()
        self.b._voice_after_refine(["S1", "S3"])
        self.assertTrue(V.voices_file(self.c).exists())
        self.assertEqual(N.load_speakers(self.c), {"S3": "山田"})
        self.assertEqual(self.b.ui.get_nowait(), ("speakers", {"S3": "山田"}))

    def test_forget_all_removes_book_and_meeting_voices(self):
        V.save_meeting_voices(self.a, {"S1": (near(YAMADA), 40.0, 12)})
        V.save_meeting_voices(self.c, {"S2": (near(SMITH), 40.0, 12)})
        self.b.speaker_rename("session-20261106-100000", "S1", "山田")
        self.assertEqual(self.b.voicebook_state()["noticed"], False)
        self.b.voicebook_notice_seen()
        self.assertEqual(self.b.voicebook_state(), {"people": 1, "noticed": True})
        self.assertEqual(self.b.voicebook_forget_all(5), -1)              # 画面で見た人数と違う = 消さない
        self.assertEqual(len(self.b.voicebook_list()), 1)
        self.assertEqual(self.b.voicebook_forget_all(1), 1)
        self.assertEqual(self.b.voicebook_list(), [])
        self.assertFalse(V.voices_file(self.a).exists())                 # 会議ごとの声の特徴も消す
        self.assertFalse(V.voices_file(self.c).exists())
        self.assertEqual(N.load_speakers(self.a), {"S1": "山田"})         # 付けた名前は残る


class TestBulkBridge(Base):
    def test_bulk_trash_protects_live_and_deletes_forever(self):
        class FakeLog:
            path_jsonl = str(self.c)

            def flush(self):
                pass
        self.b.log = FakeLog()
        self.assertEqual(self.b.library_busy(), {"live": self.c.stem, "busy": [self.c.stem]})
        Path(self.d, f"{self.a.stem}.audio.m4a").write_bytes(b"x")
        r = self.b.meetings_trash([self.a.stem, self.c.stem])
        self.assertEqual((r["moved"], r["skipped"]), ([self.a.stem], [self.c.stem]))   # 今の会話は選べない
        self.assertTrue(self.c.exists())
        self.assertEqual(self.b.search_all("quotation")["meetings"][0]["id"], self.c.stem)
        self.assertEqual(self.b.meetings_restore([self.a.stem])["restored"], [self.a.stem])
        self.b.meetings_trash([self.a.stem])
        self.assertEqual(self.b.trash_delete([self.a.stem], 2), -1)
        self.assertEqual(self.b.trash_delete([self.a.stem], 1), 1)
        self.assertEqual(self.b.trash_list(), [])
        self.assertFalse(any(Path(self.d).rglob(f"{self.a.stem}.*")))           # 録音も含め何も残らない
        self.assertEqual(self.b.meetings_trash("x"), {"moved": [], "skipped": []})


class TestCalendar(Base):
    def test_calendar_and_reminders_use_the_module(self):
        with patch.object(C, "events_now", return_value={"event": {"title": "定例", "attendees": []}}) as ev:
            self.assertEqual(self.b.calendar_now()["event"]["title"], "定例")
        ev.assert_called_once()
        self.b.meeting_calendar_set("session-20261106-100000", {"title": "予定の題", "attendees": [{"name": "A"}]})
        self.assertEqual(self.b.library_list()["meetings"][1]["title"], "予定の題")
        N.save_notes(self.a, {"lang": "ja", "title": "定例", "participants": [{"id": "S2", "name": "Smith"}],
                              "todos": [{"task": "送る", "owner": "S2"}, {"task": "済んだ", "done": True}]})
        with patch.object(C, "add_reminders", return_value={"added": 1}) as add:
            self.assertEqual(self.b.reminders_add("session-20261106-100000")["added"], 1)
        todos = add.call_args[0][0]
        self.assertEqual([t["task"] for t in todos], ["送る"])              # 済んだものは送らない
        self.assertEqual(add.call_args[1]["names"]["S2"], "Smith")


if __name__ == "__main__":
    unittest.main()
