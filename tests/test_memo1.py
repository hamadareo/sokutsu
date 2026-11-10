"""会議メモの改善(memo1, 11/10): 保存の保険(.bak)・近い重複の整理・用語の綴りの正規化・Markdown/テキスト書き出し。"""
import json
import tempfile
import unittest
from pathlib import Path

from tsuyaku import notes as N
from tsuyaku import export as E


class TestBackup(unittest.TestCase):
    def test_corrupt_notes_fall_back_to_bak(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "s1.jsonl"
            p.write_text("", encoding="utf-8")
            N.save_notes(p, {"lang": "ja", "title": "A"})
            N.save_notes(p, {"lang": "ja", "title": "B"})
            f = N.notes_file(p)
            f.write_text('{"lang": "ja", "ti', encoding="utf-8")     # 書きかけで壊れた
            self.assertEqual(N.load_notes(p)["title"], "A")           # 1つ前に戻る

    def test_corrupt_speakers_fall_back(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "s1.jsonl"
            p.write_text("", encoding="utf-8")
            N._write_json(N.speakers_file(p), {"S1": "山田"})
            N._write_json(N.speakers_file(p), {"S1": "山田", "S2": "鈴木"})
            N.speakers_file(p).write_text("{", encoding="utf-8")
            self.assertEqual(N.load_speakers(p), {"S1": "山田"})

    def test_no_bak_when_missing_and_corrupt_is_none(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "s1.jsonl"
            p.write_text("", encoding="utf-8")
            N.notes_file(p).write_text("{", encoding="utf-8")
            self.assertIsNone(N.load_notes(p))

    def test_bad_current_does_not_overwrite_good_bak(self):
        with tempfile.TemporaryDirectory() as d:
            f = Path(d) / "x.json"
            N._write_json(f, {"v": 1})
            N._write_json(f, {"v": 2})
            f.write_text("{", encoding="utf-8")
            N._write_json(f, {"v": 3})        # 壊れた現行を .bak に写さない
            self.assertEqual(json.loads(f.with_name("x.json.bak").read_text())["v"], 1)


class TestDedupe(unittest.TestCase):
    LINES = [{"spk": "S1", "orig": "x", "t": "", "lang": "ja"}]

    def clean(self, raw):
        return N._clean(raw, self.LINES, {}, None)

    def test_near_duplicate_todos_merged(self):
        out = self.clean({"todos": [
            {"task": "見積もりを送る", "owner": "", "due": ""},
            {"task": "見積もりを送る。", "owner": "S1", "due": "金曜"},
            {"task": "議事録を共有する"}]})
        self.assertEqual([t["task"] for t in out["todos"]], ["見積もりを送る", "議事録を共有する"])
        self.assertEqual(out["todos"][0]["owner"], "S1")       # 空欄は後の項目で埋める
        self.assertEqual(out["todos"][0]["due"], "金曜")

    def test_different_owner_todos_kept(self):
        out = self.clean({"todos": [{"task": "資料を確認する", "owner": "S1"},
                                    {"task": "資料を確認する", "owner": "S2"}]})
        self.assertEqual(len(out["todos"]), 2)

    def test_summary_dupes_keep_src_aligned(self):
        out = self.clean({"summary": ["予算を見直す", "A", "予算を見直す。"],
                          "summary_src": [[1], [2], [3]]})
        self.assertEqual(out["summary"], ["予算を見直す", "A"])
        self.assertEqual(out["summary_src"], [[1], [2]])

    def test_distinct_decisions_kept(self):
        out = self.clean({"decisions": ["外注はしない", "A 社で進める"]})
        self.assertEqual(len(out["decisions"]), 2)

    def test_english_case_and_punct(self):
        out = self.clean({"questions": ["When is the deadline?", "when is the deadline"]})
        self.assertEqual(len(out["questions"]), 1)

    def test_done_still_carried(self):
        old = {"todos": [{"task": "見積もりを送る", "done": True}]}
        out = N._clean({"todos": [{"task": "見積もりを送る"}]}, self.LINES, {}, old)
        self.assertTrue(out["todos"][0]["done"])


class TestTermFix(unittest.TestCase):
    def test_case_and_space_variants(self):
        s = N.fix_terms("yolo と chat GPT と Chat-GPT を使う。gpt-4 も。", ["YOLO", "ChatGPT", "GPT-4"])
        self.assertEqual(s, "YOLO と ChatGPT と ChatGPT を使う。GPT-4 も。")

    def test_common_word_terms_untouched(self):
        self.assertEqual(N.fix_terms("let's go home", ["Go"]), "let's go home")

    def test_inside_word_untouched(self):
        self.assertEqual(N.fix_terms("yolov8 and polyolo", ["YOLO"]), "yolov8 and polyolo")

    def test_apply_to_notes(self):
        out = {"title": "yolo の会", "summary": ["yolo を試す"], "decisions": [], "questions": [],
               "keywords": ["yolo"], "todos": [{"task": "yolo を動かす"}],
               "topics": [{"title": "yolo", "points": [{"text": "yolo は速い"}]}],
               "sections": {"a": [{"text": "yolo"}]}}
        N.fix_terms_in(out, ["YOLO"])
        self.assertEqual(out["title"], "YOLO の会")
        self.assertEqual(out["keywords"], ["YOLO"])
        self.assertEqual(out["todos"][0]["task"], "YOLO を動かす")
        self.assertEqual(out["topics"][0]["points"][0]["text"], "YOLO は速い")
        self.assertEqual(out["sections"]["a"][0]["text"], "YOLO")


class TestTextExport(unittest.TestCase):
    MEETING = {"id": "s1", "meta": {}, "lines": [
        {"t": "10:00:00", "spk": "S1", "lang": "ja", "orig": "こんにちは", "trans": "Hello"}]}
    NOTES = {"lang": "ja", "title": "定例", "participants": [{"id": "S1", "name": ""}],
             "summary": ["要点"], "decisions": ["決定A"],
             "todos": [{"task": "送る", "owner": "S1", "due": "金曜", "done": False}], "questions": []}

    def test_md_uses_renamed_speakers(self):
        md = E.text_export(self.NOTES, self.MEETING, {"S1": "山田"}, "md", True)
        self.assertIn("# 定例", md)
        self.assertIn("山田", md)
        self.assertIn("- [ ] 送る", md)
        self.assertIn("こんにちは", md)

    def test_txt_has_no_markdown_marks(self):
        t = E.text_export(self.NOTES, self.MEETING, {"S1": "山田"}, "txt", False)
        self.assertNotIn("##", t)
        self.assertNotIn("**", t)
        self.assertIn("決定A", t)
        self.assertNotIn("こんにちは", t)

    def test_without_notes_falls_back_to_transcript(self):
        md = E.text_export(None, self.MEETING, {}, "md", False)
        self.assertIn("こんにちは", md)

    def test_file_name(self):
        self.assertTrue(E.file_name({"title": "a/b", "start": "2026-11-10 10:00"}, "md").endswith(".md"))


if __name__ == "__main__":
    unittest.main()


class TestChunkBoundary(unittest.TestCase):
    def lines(self, n, gap_at):
        out, t = [], 0.0
        for k in range(n):
            t += 40.0 if k == gap_at else 2.0
            out.append({"orig": "あ" * 70, "trans": "", "spk": "S1", "ts": t})
        return out

    def test_splits_at_long_pause_near_limit(self):
        ls = self.lines(30, gap_at=20)       # 100字/行 → 5000字で ~30 行目で超える。20行目の前に長い間
        cs = N.chunks(ls, 2500)              # 約 25 行で超える
        self.assertEqual(len(cs[0]), 20)     # 間のところで切る(25 行目ではなく)
        self.assertEqual(sum(len(c) for c in cs), 30)

    def test_no_gap_info_keeps_old_behavior(self):
        ls = [{"orig": "あ" * 70, "trans": "", "spk": "S1"} for _ in range(30)]
        cs = N.chunks(ls, 2500)
        self.assertEqual(sum(len(c) for c in cs), 30)
        self.assertEqual(len(cs[0]), 25)

    def test_pause_too_early_ignored(self):
        ls = self.lines(30, gap_at=5)        # 序盤の間では切らない(区切りが小さくなりすぎる)
        cs = N.chunks(ls, 2500)
        self.assertGreater(len(cs[0]), 15)
