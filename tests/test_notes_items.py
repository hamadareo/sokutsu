"""会議メモの項目を右クリックから直す・消す・元に戻す(11/08 ユーザー:「右クリックでの操作も入れてください。
例えば、タスクを削除するなど」)。直したもの・消したものは、読み直しても、メモを作り直しても残る
(やることの済み/未済・名前と同じく carry_user_edits で引き継ぐ)。"""
import json
import tempfile
import unittest
from pathlib import Path

from tsuyaku import notes as N
from tsuyaku.__main__ import Bridge

NOTES = {
    "lang": "ja", "title": "定例", "participants": [{"id": "S1", "name": "山田"}, {"id": "S2", "name": ""}],
    "summary": ["予算を見直す", "次回は 11/14"], "summary_src": [[1], [2]],
    "topics": [{"title": "予算", "points": [{"who": "S1", "text": "見積もりを取り直す", "src": [1]},
                                         {"who": "S2", "text": "A 社が安い", "src": [2]}]}],
    "decisions": ["外注はしない", "A 社で進める"], "decisions_src": [[3], [4]],
    "todos": [{"task": "見積もりを送る", "owner": "S2", "due": "金曜", "src": [5], "done": False},
              {"task": "議事録を共有", "owner": "S1", "src": [], "done": True}],
    "questions": ["納期はいつ?"], "questions_src": [[6]], "keywords": ["予算"],
}


class Base(unittest.TestCase):
    def setUp(self):
        self.td = tempfile.TemporaryDirectory()
        d = Path(self.td.name)
        self.p = d / "session-20261106-100000.jsonl"
        rows = [{"type": "meta", "pair": "ja-en", "start": "2026-11-06 10:00:00"}]
        rows += [{"type": "utter", "t": f"10:00:0{i}", "lang": "ja", "orig": f"発言{i}", "spk": "S1"} for i in range(8)]
        self.p.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
        N.save_notes(self.p, json.loads(json.dumps(NOTES)))
        self.b = Bridge()
        self.b.s.log_dir = str(d)
        self.mid = self.p.stem

    def tearDown(self):
        self.td.cleanup()

    def notes(self):
        return N.load_notes(self.p, "ja")


class TestEditDeleteRestore(Base):
    def test_edit_each_kind_and_reload(self):
        b, mid = self.b, self.mid
        self.assertIsNotNone(b.notes_item_edit(mid, "summary", 0, {"text": "予算を半分に"}, "ja"))
        b.notes_item_edit(mid, "decisions", 1, {"text": "B 社で進める"}, "ja")
        b.notes_item_edit(mid, "questions", 0, {"text": "納期は 12 月?"}, "ja")
        b.notes_item_edit(mid, "points", [0, 1], {"text": "B 社が安い"}, "ja")
        b.notes_item_edit(mid, "todos", 0, {"task": "見積もりを今日送る", "owner": "S1", "due": "今日"}, "ja")
        n = self.notes()     # 読み直しても残る
        self.assertEqual(n["summary"][0], "予算を半分に")
        self.assertEqual(n["decisions"][1], "B 社で進める")
        self.assertEqual(n["questions"][0], "納期は 12 月?")
        self.assertEqual(n["topics"][0]["points"][1]["text"], "B 社が安い")
        self.assertEqual((n["todos"][0]["task"], n["todos"][0]["owner"], n["todos"][0]["due"]), ("見積もりを今日送る", "S1", "今日"))
        self.assertEqual(n["todos"][0]["src"], [5])   # 根拠はそのまま

    def test_delete_keeps_src_aligned_and_restore(self):
        b, mid = self.b, self.mid
        undo = b.notes_item_delete(mid, "decisions", 0, "ja")
        self.assertEqual(undo["notes"]["decisions"], ["A 社で進める"])    # 画面は返ったメモで描き直す
        undo = {k: undo[k] for k in ("item", "src", "user_edits")}         # 画面が戻すときに渡すもの
        n = self.notes()
        self.assertEqual(n["decisions"], ["A 社で進める"])
        self.assertEqual(n["decisions_src"], [[4]])
        self.assertTrue(b.notes_item_restore(mid, "decisions", 0, undo, "ja"))
        n = self.notes()
        self.assertEqual(n["decisions"], ["外注はしない", "A 社で進める"])
        self.assertEqual(n["decisions_src"], [[3], [4]])
        self.assertEqual(n.get("user_edits") or [], [])

    def test_delete_todo_point_and_restore(self):
        b, mid = self.b, self.mid
        u1 = b.notes_item_delete(mid, "todos", 1, "ja")
        u2 = b.notes_item_delete(mid, "points", [0, 0], "ja")
        n = self.notes()
        self.assertEqual([t["task"] for t in n["todos"]], ["見積もりを送る"])
        self.assertEqual([p["text"] for p in n["topics"][0]["points"]], ["A 社が安い"])
        b.notes_item_restore(mid, "points", [0, 0], u2, "ja")
        b.notes_item_restore(mid, "todos", 1, u1, "ja")
        n = self.notes()
        self.assertEqual([t["task"] for t in n["todos"]], ["見積もりを送る", "議事録を共有"])
        self.assertTrue(n["todos"][1]["done"])
        self.assertEqual(n["topics"][0]["points"][0]["text"], "見積もりを取り直す")

    def test_bad_requests(self):
        b, mid = self.b, self.mid
        self.assertIsNone(b.notes_item_edit(mid, "nope", 0, {"text": "x"}, "ja"))
        self.assertIsNone(b.notes_item_edit(mid, "summary", 9, {"text": "x"}, "ja"))
        self.assertIsNone(b.notes_item_edit(mid, "summary", 0, {"text": "  "}, "ja"))
        self.assertIsNone(b.notes_item_delete("no-such", "summary", 0, "ja"))
        self.assertIsNone(b.notes_item_edit(mid, "todos", 0, {"owner": "<script>"}, "ja"))   # 話者は S1 / Lja などだけ
        self.assertIsNone(b.notes_item_restore(mid, "summary", 0, {"item": {"x": 1}, "src": []}, "ja"))   # 形の違う値は戻さない
        self.assertIsNone(b.notes_item_restore(mid, "todos", 0, {"item": "文字列", "src": []}, "ja"))
        self.assertEqual(len(self.notes()["summary"]), 2)


class TestCarryOverRemake(Base):
    """メモを作り直しても、直したもの・消したものは残る(LLM が同じ文を出した項目に当てる)。"""

    def test_edits_and_deletes_survive_regeneration(self):
        b, mid = self.b, self.mid
        b.notes_item_delete(mid, "decisions", 0, "ja")
        b.notes_item_edit(mid, "todos", 0, {"task": "見積もりを今日送る", "due": "今日"}, "ja")
        b.notes_item_edit(mid, "points", [0, 1], {"text": "B 社が安い"}, "ja")
        b.notes_item_delete(mid, "summary", 1, "ja")
        new = json.loads(json.dumps(NOTES))          # 作り直し(LLM が同じ項目を出した)
        new["todos"][0]["done"] = False
        out = N.carry_user_edits(new, self.p, {})
        self.assertEqual(out["decisions"], ["A 社で進める"])
        self.assertEqual(out["decisions_src"], [[4]])
        self.assertEqual(out["todos"][0]["task"], "見積もりを今日送る")
        self.assertEqual(out["todos"][0]["due"], "今日")
        self.assertEqual(out["topics"][0]["points"][1]["text"], "B 社が安い")
        self.assertEqual(out["summary"], ["予算を見直す"])
        self.assertTrue(out.get("user_edits"))      # 次の作り直しにも引き継ぐ

    def test_edit_twice_then_delete_is_one_record(self):
        b, mid = self.b, self.mid
        b.notes_item_edit(mid, "questions", 0, {"text": "A"}, "ja")
        b.notes_item_edit(mid, "questions", 0, {"text": "B"}, "ja")
        b.notes_item_delete(mid, "questions", 0, "ja")
        ue = self.notes()["user_edits"]
        self.assertEqual(len(ue), 1)
        self.assertEqual(ue[0]["op"], "del")
        self.assertEqual(ue[0]["key"], "納期はいつ?")
        out = N.carry_user_edits(json.loads(json.dumps(NOTES)), self.p, {})
        self.assertEqual(out["questions"], [])


if __name__ == "__main__":
    unittest.main()
