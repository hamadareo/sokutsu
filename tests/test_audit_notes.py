"""監査(11/07): 会議の記録・メモ・ライブラリの壊れ方(書きかけ・おかしな LLM の答え・同時の操作)。"""
import io
import json
import tempfile
import threading
import time
import unittest
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path
from unittest.mock import patch

from tsuyaku import analytics as A
from tsuyaku import export as E
from tsuyaku import library as L
from tsuyaku import notes as N
from tsuyaku.__main__ import Bridge

from test_notes import GOOD, ROWS, write_log

MID = "session-20261106-100000"


def wait_notes(b, timeout=5.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        e = b.ui.get(timeout=timeout)
        if e[0] == "notes" and e[1]["stage"] in ("done", "error"):
            return e[1]
    return None


class TestTruncatedUtf8(unittest.TestCase):
    """アプリが書く途中で落ちる・ディスクが一杯で、最後の行が日本語の字の途中で切れた記録。"""

    def setUp(self):
        self.td = tempfile.TemporaryDirectory()
        self.d = self.td.name
        self.p = write_log(self.d, ROWS, stem=MID)
        with self.p.open("ab") as f:
            f.write('\n{"type": "utter", "orig": "あ'.encode("utf-8")[:-1])   # 「あ」の3バイトの途中
        write_log(self.d, ROWS, stem="session-20261107-100000")              # 壊れていない別の会議

    def tearDown(self):
        self.td.cleanup()

    def test_meeting_still_loads(self):
        m = N.load_meeting(self.p)
        self.assertEqual(len(m["lines"]), 3)

    def test_library_lists_every_meeting(self):
        ids = [m["id"] for m in L.list_meetings(self.d)]
        self.assertEqual(ids, ["session-20261107-100000", MID])     # 1件のせいで一覧が空にならない


class TestOddLlmShapes(unittest.TestCase):
    """小さい LLM が JSON の形は守っても、中身の型を外す(null・数)。メモ作りは落ちない。"""

    def make(self, raw):
        with tempfile.TemporaryDirectory() as d:
            m = N.load_meeting(write_log(d, ROWS))
            return N.make_notes(m, "ja", lambda msgs: json.dumps(raw))

    def test_participants_null(self):
        n = self.make(dict(GOOD, participants=None))
        self.assertEqual([p["id"] for p in n["participants"]], ["S1", "S2"])
        self.assertEqual(n["title"], "進め方の打ち合わせ")

    def test_lists_given_as_numbers(self):
        n = self.make(dict(GOOD, topics=3, todos=0, participants=1, keywords=2))
        self.assertEqual(n["topics"], [])
        self.assertEqual(n["todos"], [])
        self.assertEqual(n["summary"], ["進め方を決めた"])

    def test_template_sections_as_text(self):
        for v in ("質問はなかった", 5):
            with tempfile.TemporaryDirectory() as d:
                m = N.load_meeting(write_log(d, ROWS))
                n = N.make_notes(m, "ja", lambda msgs, v=v: json.dumps(dict(GOOD, sections=v)), template="regular")
            self.assertTrue(all(sec["items"] == [] for sec in n["sections"]), v)

    def test_translation_items_not_a_list(self):
        """違う言語の項目を訳し直す2回目の LLM が items を数・文字列で返しても、メモは前のまま残る。"""
        g = dict(GOOD, summary=["We decided the plan", "その件は決定"])
        for bad in (5, "x"):   # "x": 項目1つの時、1字の文字列が訳として入っていた
            calls = []

            def llm(msgs, bad=bad):
                calls.append(1)
                return json.dumps(g) if len(calls) == 1 else json.dumps({"items": bad})
            with tempfile.TemporaryDirectory() as d:
                n = N.make_notes(N.load_meeting(write_log(d, ROWS)), "ja", llm)
            self.assertEqual(n["summary"], ["We decided the plan", "その件は決定"], bad)

    def test_points_number(self):
        n = self.make(dict(GOOD, topics=[{"title": "進め方", "points": 5}]))
        self.assertEqual(n["topics"][0]["title"], "進め方")


class TestTrashWhileNotesRun(unittest.TestCase):
    """メモを作っている間(数分)に会議をゴミ箱へ。作り終えたメモがログのフォルダに残ると、
    ゴミ箱から戻せなくなる(同じ名前のファイルがあるので restore が断る)。"""

    def test_restore_still_works(self):
        b = Bridge()
        with tempfile.TemporaryDirectory() as d:
            b.s.log_dir = d
            p = write_log(d, ROWS, stem=MID)
            N.save_notes(p, dict(GOOD, lang="ja"))          # 前に作ったメモがある
            go, asked = threading.Event(), threading.Event()

            def slow(*_a, **_k):
                asked.set()
                go.wait(5)
                return json.dumps(GOOD)
            with patch.object(Bridge, "_notes_llm", return_value=("http://x", "m")), \
                 patch("tsuyaku.notes.chat", side_effect=slow):
                self.assertEqual(b.notes_make(MID, "ja"), "started")
                self.assertTrue(asked.wait(5))
                self.assertEqual(b.meeting_trash(MID), "ok")
                go.set()
                ev = wait_notes(b)
            self.assertEqual(ev["stage"], "error", ev)
            self.assertEqual(sorted(x.name for x in Path(d).glob("session-*")), [])   # 何も残らない
            self.assertEqual(b.meeting_restore(MID), "ok")
            self.assertEqual(N.load_notes(Path(d) / f"{MID}.jsonl")["title"], "進め方の打ち合わせ")

    def test_draft_after_trash_writes_nothing(self):
        b = Bridge()
        with tempfile.TemporaryDirectory() as d:
            b.s.log_dir = d
            p = write_log(d, ROWS, stem=MID)
            N.save_notes(p, dict(GOOD, lang="ja"))

            def trash_then_answer(*_a, **_k):
                b.meeting_trash(MID)
                return "件名: お礼"
            with patch.object(Bridge, "_notes_llm", return_value=("http://x", "m")), \
                 patch("tsuyaku.notes.chat", side_effect=trash_then_answer):
                r = b.notes_draft(MID, "email", "ja")
            self.assertIn("error", r)
            self.assertEqual(sorted(x.name for x in Path(d).glob("session-*")), [])
            self.assertEqual(b.meeting_restore(MID), "ok")


class TestAtomicWrites(unittest.TestCase):
    """メモ・名前のファイルを書く途中で落ちても(ディスクが一杯など)、前のものが残る。"""

    def _failing_write(self):
        real = Path.write_text

        def half(self_, data, *a, **k):
            real(self_, data[: len(data) // 2], *a, **k)
            raise OSError(28, "No space left on device")
        return patch.object(Path, "write_text", half)

    def test_notes_survive_failed_save(self):
        with tempfile.TemporaryDirectory() as d:
            p = write_log(d, ROWS, stem=MID)
            N.save_notes(p, dict(GOOD, lang="ja"))
            with self._failing_write(), self.assertRaises(OSError):
                N.save_notes(p, dict(GOOD, lang="ja", title="新しい題"))
            self.assertEqual(N.load_notes(p)["title"], "進め方の打ち合わせ")
            self.assertEqual(sorted(x.name for x in Path(d).iterdir()),
                             [f"{MID}.jsonl", f"{MID}.notes.json"])        # 書きかけの一時ファイルも残さない

    def test_speaker_names_survive_failed_save(self):
        with tempfile.TemporaryDirectory() as d:
            p = write_log(d, ROWS, stem=MID)
            N.rename_speaker(p, "S1", "山田")
            with self._failing_write(), self.assertRaises(OSError):
                N.rename_speaker(p, "S2", "Smith")
            self.assertEqual(N.load_speakers(p), {"S1": "山田"})


class TestDocxControlChars(unittest.TestCase):
    """記録・題に制御文字(XML 1.0 に書けない字)があっても、Word で開ける docx を作る。"""

    def test_document_xml_is_well_formed(self):
        with tempfile.TemporaryDirectory() as d:
            rows = [dict(ROWS[0], orig="見積もり\x01の件\x1b[0m"), ROWS[1]]
            m = N.load_meeting(write_log(d, rows))
        doc = E.build_doc(dict(GOOD, lang="ja", title="定例\x07"), m, A.speaker_stats(m), {},
                          lang="ja", with_transcript=True)
        z = zipfile.ZipFile(io.BytesIO(E.to_docx(doc)))
        for name in ("word/document.xml", "docProps/core.xml"):
            ET.fromstring(z.read(name))                       # 壊れていれば ParseError
        self.assertIn("見積もりの件", z.read("word/document.xml").decode("utf-8"))


class TestSubtitlesFallbackKeepsLog(unittest.TestCase):
    """保存の窓が出せない時の書き先が記録の隣。テキスト(txt)の書き出しが、記録の人間向けのログ
    session-X.txt を字幕の文で上書きしていた。"""

    def test_txt_export_does_not_overwrite_session_log(self):
        from tsuyaku.log import SessionLog
        b = Bridge()
        with tempfile.TemporaryDirectory() as d:
            b.s.log_dir = d
            log = SessionLog(d, "ja-en", "", "notes")
            log.add("ja", "こんにちは。", "", None, "S1")
            log.close()
            before = Path(log.path_txt).read_text(encoding="utf-8")
            out = b.subtitles_export(Path(log.path_jsonl).stem, "txt", "ja")
            self.assertTrue(out.endswith(".txt"))
            self.assertNotEqual(Path(out), Path(log.path_txt))
            self.assertIn("こんにちは。", Path(out).read_text(encoding="utf-8"))
            self.assertEqual(Path(log.path_txt).read_text(encoding="utf-8"), before)


class TestAskOddFacts(unittest.TestCase):
    """長い会議への質問: 関係する行を挙げる LLM が lines を数1つ・文字列で返しても答えまで進む。"""

    def test_lines_as_number_or_text(self):
        from tsuyaku import meeting_ai as M
        with tempfile.TemporaryDirectory() as d:
            m = N.load_meeting(write_log(d, ROWS * 200))
        for lines in (5, "12"):
            def llm(msgs, lines=lines):
                if "facts" in msgs[0]["content"]:
                    return json.dumps({"facts": [{"text": "a", "lines": lines}]})
                return "スライドを送ります [#5]"
            r = M.ask(m, None, "何を送る?", "ja", llm, budget=2000)
            self.assertEqual(r["cites"], [5], lines)


# ---------------- 2回目(11/07 リードの依頼) ----------------

class TestEditsDuringRemake(unittest.TestCase):
    """(1) メモを作り直している間のやることのチェック・名前の変更は、作り終えたメモに引き継ぐ。"""

    def test_toggle_and_rename_survive(self):
        b = Bridge()
        with tempfile.TemporaryDirectory() as d:
            b.s.log_dir = d
            p = write_log(d, ROWS, stem=MID)
            N.rename_speaker(p, "S1", "山田")
            N.save_notes(p, dict(GOOD, lang="ja", todos=[dict(GOOD["todos"][0], done=False)]))
            go, asked = threading.Event(), threading.Event()

            def slow(*_a, **_k):
                asked.set()
                go.wait(5)
                return json.dumps(dict(GOOD, participants=[{"id": "S2", "name": "Smith", "evidence": "I'm Smith"}]))
            with patch.object(Bridge, "_notes_llm", return_value=("http://x", "m")), \
                 patch("tsuyaku.notes.chat", side_effect=slow):
                self.assertEqual(b.notes_make(MID, "ja"), "started")
                self.assertTrue(asked.wait(5))
                self.assertEqual(b.todo_toggle(MID, 0, True), "ok")          # 待つ間にチェック
                self.assertEqual(b.speaker_rename(MID, "S2", "スミス"), "ok")  # 待つ間に名前を直す
                self.assertEqual(b.speaker_rename(MID, "S1", ""), "ok")       # 待つ間に名前を消す
                go.set()
                ev = wait_notes(b)
            self.assertEqual(ev["stage"], "done", ev)
            for n in (ev["notes"], N.load_notes(p, "ja")):
                self.assertTrue(n["todos"][0]["done"])
                ps = {q["id"]: q for q in n["participants"]}
                self.assertEqual((ps["S2"]["name"], ps["S2"]["fixed"]), ("スミス", True))
                self.assertEqual((ps["S1"]["name"], ps["S1"]["fixed"]), ("", False))

    def test_unchanged_todo_keeps_old_state(self):
        """作り直しの間に触らなかったチェックは、今までどおり前のメモのまま(済みは済み)。"""
        with tempfile.TemporaryDirectory() as d:
            p = write_log(d, ROWS, stem=MID)
            old = dict(GOOD, lang="ja", todos=[dict(GOOD["todos"][0], done=True)])
            N.save_notes(p, old)
            new = N.make_notes(N.load_meeting(p), "ja", lambda m: json.dumps(GOOD), old=N.load_notes(p, "ja"))
            N.carry_user_edits(new, p, known={})
            self.assertTrue(new["todos"][0]["done"])


class TestRenameLock(unittest.TestCase):
    """(2) 名前の変更が2か所から同時に来ても(画面と声の照合)、どちらも残る。"""

    def test_concurrent_renames(self):
        with tempfile.TemporaryDirectory() as d:
            p = write_log(d, ROWS, stem=MID)
            real = N.load_speakers

            def slow_load(*a, **k):
                r = real(*a, **k)
                time.sleep(0.2)
                return r
            with patch.object(N, "load_speakers", slow_load):
                th = [threading.Thread(target=N.rename_speaker, args=(p, s, nm))
                      for s, nm in (("S1", "山田"), ("S2", "Smith"))]
                for x in th:
                    x.start()
                for x in th:
                    x.join()
            self.assertEqual(N.load_speakers(p), {"S1": "山田", "S2": "Smith"})


class TestInfoAfterTrash(unittest.TestCase):
    """(3) ゴミ箱へ移した直後に届いた題・タグの変更は書かない(書くと .tags.json だけが残って戻せない)。"""

    def test_update_info_on_trashed_meeting(self):
        with tempfile.TemporaryDirectory() as d:
            p = write_log(d, ROWS, stem=MID)
            L.update_info(p, title="前の題")
            self.assertTrue(L.trash(d, MID))
            self.assertIsNone(L.update_info(p, title="新しい題"))
            self.assertEqual(sorted(x.name for x in Path(d).glob("session-*")), [])
            self.assertTrue(L.restore(d, MID))
            self.assertEqual(L.load_info(p).get("title"), "前の題")


class TestTrashOneWhileImporting(unittest.TestCase):
    """(4) 1件ずつのゴミ箱も、取り込み中の会議はまとめての時と同じく移さない。"""

    def test_single_trash_skips_importing(self):
        b = Bridge()
        with tempfile.TemporaryDirectory() as d:
            b.s.log_dir = d
            write_log(d, ROWS, stem=MID)
            stop = threading.Event()
            b._import_th = threading.Thread(target=stop.wait, args=(5,), daemon=True)
            b._import_th.start()
            try:
                self.assertEqual(b.meetings_trash([MID])["skipped"], [MID])
                self.assertNotEqual(b.meeting_trash(MID), "ok")
                self.assertTrue((Path(d) / f"{MID}.jsonl").exists())
            finally:
                stop.set()
                b._import_th.join(5)
            self.assertEqual(b.meeting_trash(MID), "ok")      # 取り込みが終われば移せる


class TestBrokenSearchIndex(unittest.TestCase):
    """(5) 検索の索引(キャッシュ)が壊れていたら、消して作り直す。"""

    def test_garbage_index_is_rebuilt(self):
        from tsuyaku import search as S
        with tempfile.TemporaryDirectory() as d:
            write_log(d, ROWS, stem=MID)
            db = Path(d) / ".index" / "search.sqlite3"
            db.parent.mkdir()
            db.write_bytes(b"this is not a sqlite database" * 100)
            r = S.SearchIndex(d).search("進め方")
            self.assertEqual([m["id"] for m in r["meetings"]], [MID])


class TestClosedLogWrites(unittest.TestCase):
    """(6) 会議を終える瞬間(記録を閉じた後、Bridge.log が None になる前)の読み書き。"""

    def test_flush_and_add_row_after_close(self):
        from tsuyaku.log import SessionLog
        with tempfile.TemporaryDirectory() as d:
            log = SessionLog(d, "ja-en", "", "notes")
            log.add("ja", "こんにちは。", "", None, "S1")
            log.close()
            log.flush()                                         # 例外にしない
            log.add_row({"type": "mark", "id": "k1", "i": 0, "kind": "star", "label": "大事"})
            m = N.load_meeting(log.path_jsonl)
            self.assertEqual([x["label"] for x in m["marks"]], ["大事"])   # 印は失わない
            self.assertEqual(len(m["lines"]), 1)

    def test_bridge_calls_while_closing(self):
        from tsuyaku.log import SessionLog
        b = Bridge()
        with tempfile.TemporaryDirectory() as d:
            b.s.log_dir = d
            log = SessionLog(d, "ja-en", "", "notes")
            log.add("ja", "こんにちは。", "", None, "S1")
            log.close()
            b.log = log                       # end_meeting の途中: 閉じたが、まだ None にしていない
            self.assertEqual(len(b.meeting_get("live")["lines"]), 1)
            got = b.mark_set("live", {"i": 0, "label": "大事"})
            self.assertEqual(got["label"], "大事")
            self.assertEqual(len(N.load_meeting(log.path_jsonl)["marks"]), 1)


class TestNotesBusyKey(unittest.TestCase):
    """(7) 同じ会議・同じ言語のメモ作りは、言語の渡し方(None / "ja" / 知らない言語)が違っても1本だけ。"""

    def test_same_language_is_busy(self):
        b = Bridge()
        with tempfile.TemporaryDirectory() as d:
            b.s.log_dir = d
            write_log(d, ROWS, stem=MID)
            go, asked = threading.Event(), threading.Event()

            def slow(*_a, **_k):
                asked.set()
                go.wait(5)
                return json.dumps(GOOD)
            with patch.object(Bridge, "_notes_llm", return_value=("http://x", "m")), \
                 patch("tsuyaku.notes.chat", side_effect=slow):
                self.assertEqual(b.notes_make(MID, None), "started")
                self.assertTrue(asked.wait(5))
                try:
                    self.assertEqual(b.notes_make(MID, "ja"), "busy")
                    self.assertEqual(b.notes_make(MID, "xx"), "busy")
                    self.assertEqual(b.notes_make(MID, "en"), "started")     # 別の言語は別に作れる
                finally:
                    go.set()
                for _ in range(2):
                    self.assertIsNotNone(wait_notes(b))


if __name__ == "__main__":
    unittest.main()
