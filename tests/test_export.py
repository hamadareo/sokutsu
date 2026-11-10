"""W4 書き出し(11/07): PDF と Word(.docx)。新しい依存なし(PDF = AppKit の文字組み、docx = OOXML を直接)。"""
import json
import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from xml.dom import minidom

from tsuyaku import export as X

NOTES = {
    "lang": "ja", "title": "チームの定例 <A&B>", "start": "2026-11-06 10:00:00", "minutes": 32,
    "languages": ["ja", "en"], "model": "qwen", "made": "2026-11-06 10:40:00",
    "participants": [{"id": "S1", "name": "山田", "role": "司会"}, {"id": "S2", "name": "", "lang": "en"}],
    "summary": ["予算を見直すことにした", "次回は 11/14"],
    "topics": [{"title": "予算", "points": [{"who": "S1", "text": "見積もりを取り直す"},
                                          {"who": "S2", "text": "Vendor A is cheaper"}]}],
    "decisions": ["外注はしない"],
    "todos": [{"task": "見積もりを送る", "owner": "S2", "due": "金曜", "done": False},
              {"task": "議事録を共有", "owner": "S1", "done": True}],
    "questions": ["納期はいつ?"], "keywords": ["予算", "見積もり"],
}
MEETING = {"id": "session-20261106-100000", "meta": {"start": "2026-11-06 10:00:00", "pair": "ja-en"},
           "lines": [{"t": "10:00:05", "lang": "ja", "orig": "予算の話をします。", "trans": "Let's talk budget.", "spk": "S1"},
                     {"t": "10:00:12", "lang": "en", "orig": "Vendor A is cheaper.", "trans": "A社が安い。", "spk": "S2"}]}
STATS = {"speakers": [{"id": "S1", "secs": 600.0, "share": 0.6, "turns": 10, "lines": 20, "longest": 95.0, "questions": 2},
                      {"id": "S2", "secs": 400.0, "share": 0.4, "turns": 9, "lines": 15, "longest": 40.0, "questions": 0}],
         "total": 1000.0, "span": 1900.0, "estimated": False, "timeline": []}


class TestDocModel(unittest.TestCase):
    def test_doc_has_sections_names_and_optional_transcript(self):
        d = X.build_doc(NOTES, MEETING, STATS, {"tags": ["定例"], "folder": "チーム"}, names={"S2": "Smith"})
        kinds = [b[0] for b in d["blocks"]]
        self.assertEqual(d["title"], "チームの定例 <A&B>")
        for k in ("h1", "bullets", "points", "checks", "todos", "questions", "table"):
            self.assertIn(k, kinds)
        txt = json.dumps(d, ensure_ascii=False)
        self.assertIn("Smith", txt)                    # 画面で直した名前が勝つ
        self.assertIn("山田", txt)
        self.assertIn("10:00", txt)
        self.assertNotIn("Let's talk budget.", txt)    # 既定は全文なし
        d2 = X.build_doc(NOTES, MEETING, STATS, {}, with_transcript=True)
        self.assertIn("Let's talk budget.", json.dumps(d2, ensure_ascii=False))

    def test_doc_without_notes_still_has_transcript_and_stats(self):
        d = X.build_doc(None, MEETING, STATS, {"title": "手の題"}, lang="en", with_transcript=False)
        self.assertEqual(d["title"], "手の題")
        txt = json.dumps(d, ensure_ascii=False)
        self.assertIn("Talk time", txt)
        self.assertIn("Vendor A is cheaper.", txt)     # メモが無ければ記録を入れる

    def test_file_name_is_safe(self):
        self.assertEqual(X.file_name({"title": "a/b:c*?", "start": "2026-11-06 10:00:00"}, "pdf"),
                         "2026-11-06 a-b-c--.pdf")


class TestDocx(unittest.TestCase):
    def test_docx_is_valid_ooxml_and_readable_by_textutil(self):
        d = X.build_doc(NOTES, MEETING, STATS, {"tags": ["定例"]}, with_transcript=True)
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "m.docx"
            p.write_bytes(X.to_docx(d))
            with zipfile.ZipFile(p) as z:
                names = set(z.namelist())
                for n in ("[Content_Types].xml", "_rels/.rels", "word/document.xml", "word/styles.xml",
                          "word/_rels/document.xml.rels", "word/numbering.xml", "docProps/core.xml"):
                    self.assertIn(n, names)
                    minidom.parseString(z.read(n))          # 整形式の XML
                doc = z.read("word/document.xml").decode()
            self.assertIn("チームの定例 &lt;A&amp;B&gt;", doc)
            self.assertIn('w:val="Heading1"', doc)
            self.assertIn("☑", doc)
            out = subprocess.run(["textutil", "-convert", "txt", "-stdout", str(p)],
                                 capture_output=True, text=True, timeout=30)
            self.assertEqual(out.returncode, 0, out.stderr)
            for s in ("チームの定例 <A&B>", "見積もりを送る", "外注はしない", "Vendor A is cheaper", "60%"):
                self.assertIn(s, out.stdout)


class TestPdf(unittest.TestCase):
    def test_pdf_has_pages_and_real_text(self):
        d = X.build_doc(NOTES, MEETING, STATS, {}, with_transcript=True)
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "m.pdf"
            n = X.render_pdf(d, p)
            self.assertGreaterEqual(n, 1)
            self.assertEqual(p.read_bytes()[:5], b"%PDF-")
            import Quartz
            from Foundation import NSURL
            doc = Quartz.PDFDocument.alloc().initWithURL_(NSURL.fileURLWithPath_(str(p)))
            self.assertEqual(doc.pageCount(), n)
            text = doc.string()
            for s in ("チームの定例", "見積もりを送る", "外注はしない", "Vendor A is cheaper", "1 / "):
                self.assertIn(s, text)

    def test_long_meeting_paginates(self):
        lines = [{"t": f"10:{i // 60:02d}:{i % 60:02d}", "lang": "ja", "orig": "長い会議の発言です。" * 3, "spk": "S1"}
                 for i in range(400)]
        d = X.build_doc(NOTES, dict(MEETING, lines=lines), STATS, {}, with_transcript=True)
        with tempfile.TemporaryDirectory() as td:
            self.assertGreater(X.render_pdf(d, Path(td) / "m.pdf"), 5)

    def test_pdf_in_a_child_process(self):
        d = X.build_doc(NOTES, MEETING, STATS, {})
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "c.pdf"
            n = X.pdf_subprocess(d, p, python=sys.executable)
            self.assertGreaterEqual(n, 1)
            self.assertTrue(p.exists())


if __name__ == "__main__":
    unittest.main()
