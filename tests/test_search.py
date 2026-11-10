"""W4 全文検索(11/07): 全会議の書き起こし・訳・メモ・題・タグを横断して探す(sqlite FTS5 trigram)。"""
import json
import tempfile
import time
import unittest
from pathlib import Path

from tsuyaku import library as LB
from tsuyaku import notes as N
from tsuyaku import search as S


def write_log(d, stem, rows, start="2026-11-06 10:00:00"):
    p = Path(d) / f"{stem}.jsonl"
    with p.open("w", encoding="utf-8") as f:
        f.write(json.dumps({"type": "meta", "pair": "ja-en", "start": start}, ensure_ascii=False) + "\n")
        for r in rows:
            f.write(json.dumps(dict({"type": "utter"}, **r), ensure_ascii=False) + "\n")
    return p


A = [
    {"t": "10:00:05", "lang": "ja", "orig": "予算の見直しを来週までにやりましょう。", "trans": "Let's revise the budget by next week.", "spk": "S1"},
    {"t": "10:00:12", "lang": "en", "orig": "I will send the Quotation on Friday.", "trans": "金曜に見積もりを送ります。", "spk": "S2"},
    {"t": "10:00:20", "lang": "ja", "orig": "了解です。", "trans": "", "spk": "S1"},
]
B = [
    {"t": "09:00:05", "lang": "ja", "orig": "新しい予算の案を作りました。", "trans": "", "spk": "S1"},
]


class TestSearch(unittest.TestCase):
    def setUp(self):
        self.td = tempfile.TemporaryDirectory()
        self.d = Path(self.td.name)
        self.a = write_log(self.d, "session-20261106-100000", A)
        self.b = write_log(self.d, "session-20261107-090000", B, start="2026-11-07 09:00:00")
        self.ix = S.SearchIndex(self.d)

    def tearDown(self):
        self.td.cleanup()

    def test_two_char_japanese_word_finds_both_meetings_newest_first(self):
        r = self.ix.search("予算")
        self.assertEqual([m["id"] for m in r["meetings"]], ["session-20261107-090000", "session-20261106-100000"])
        h = r["meetings"][1]["hits"][0]
        self.assertEqual((h["kind"], h["i"], h["spk"], h["t"]), ("line", 0, "S1", "10:00:05"))
        a, b = h["marks"][0]
        self.assertEqual(h["snip"][a:b], "予算")
        self.assertEqual(r["total"], 2)

    def test_case_insensitive_english_and_translation_text(self):
        r = self.ix.search("quotation")
        self.assertEqual(len(r["meetings"]), 1)
        h = r["meetings"][0]["hits"][0]
        self.assertEqual(h["i"], 1)
        a, b = h["marks"][0]
        self.assertEqual(h["snip"][a:b], "Quotation")
        self.assertEqual(self.ix.search("見積もり")["meetings"][0]["hits"][0]["i"], 1)   # 訳でも当たる

    def test_all_terms_must_match_in_the_same_line(self):
        self.assertEqual(self.ix.search("予算　来週")["total"], 1)       # 全角の空白でも区切る
        self.assertEqual(self.ix.search("予算 金曜")["total"], 0)

    def test_notes_title_and_tags_are_searchable(self):
        N.save_notes(self.a, {"lang": "ja", "title": "定例ミーティング", "summary": ["コスト削減の方針を決めた"],
                              "decisions": ["外注はしない"], "todos": [{"task": "資料を共有する", "owner": "S2"}],
                              "topics": [{"title": "体制", "points": [{"who": "S1", "text": "採用を増やす"}]}],
                              "questions": ["納期は?"], "keywords": ["コスト"]})
        LB.update_info(self.a, title="A社との定例", tags=["重要案件"])
        r = self.ix.search("コスト削減")
        h = r["meetings"][0]["hits"][0]
        self.assertEqual((h["kind"], h["sec"]), ("note", "summary"))
        self.assertEqual(self.ix.search("資料")["meetings"][0]["hits"][0]["sec"], "todos")
        self.assertEqual(self.ix.search("採用")["meetings"][0]["hits"][0]["sec"], "topics")
        self.assertEqual(self.ix.search("重要案件")["meetings"][0]["hits"][0]["kind"], "title")
        self.assertEqual(self.ix.search("A社")["meetings"][0]["id"], "session-20261106-100000")

    def test_index_follows_appends_deletes_and_trash(self):
        self.assertEqual(self.ix.search("議事録")["total"], 0)
        time.sleep(0.01)
        with self.a.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"type": "utter", "t": "10:01:00", "lang": "ja", "orig": "議事録を残します",
                                "spk": "S2"}, ensure_ascii=False) + "\n")
        self.assertEqual(self.ix.search("議事録")["meetings"][0]["hits"][0]["i"], 3)
        LB.trash(self.d, self.a.stem)
        self.assertEqual(self.ix.search("議事録")["total"], 0)
        self.assertEqual([m["id"] for m in self.ix.search("予算")["meetings"]], ["session-20261107-090000"])

    def test_like_wildcards_are_literal_and_empty_query(self):
        self.assertEqual(self.ix.search("%")["total"], 0)
        self.assertEqual(self.ix.search("_")["total"], 0)
        self.assertEqual(self.ix.search("   ")["meetings"], [])

    def test_per_meeting_hits_are_capped_but_counted(self):
        rows = [{"t": f"10:{i // 60:02d}:{i % 60:02d}", "lang": "ja", "orig": f"テスト {i}", "spk": "S1"}
                for i in range(30)]
        write_log(self.d, "session-20261108-090000", rows, start="2026-11-08 09:00:00")
        r = self.ix.search("テスト", per_meeting=5)
        m = r["meetings"][0]
        self.assertEqual(len(m["hits"]), 5)
        self.assertEqual(m["n"], 30)

    def test_snippet_is_cut_around_the_hit(self):
        long = "あ" * 200 + "目印" + "い" * 200
        write_log(self.d, "session-20261108-090000", [{"t": "10:00:00", "lang": "ja", "orig": long, "spk": "S1"}],
                  start="2026-11-08 09:00:00")
        h = self.ix.search("目印")["meetings"][0]["hits"][0]
        self.assertLess(len(h["snip"]), 120)
        a, b = h["marks"][0]
        self.assertEqual(h["snip"][a:b], "目印")
        self.assertTrue(h["snip"].startswith("…"))

    def test_w3_memos_marks_agenda_and_edits_are_searchable(self):
        with self.a.open("a", encoding="utf-8") as f:
            for r in ({"type": "memo", "id": "m1", "t": "10:00:15", "text": "あとで価格表を確認", "i": 1},
                      {"type": "mark", "id": "k1", "t": "10:00:16", "kind": "decision", "label": "外注しない方針", "i": 2},
                      {"type": "edit", "i": 0, "orig": "予算の見直しを再来月までにやりましょう。"}):
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        h = self.ix.search("価格表")["meetings"][0]["hits"][0]
        self.assertEqual((h["kind"], h["i"]), ("memo", 1))
        h = self.ix.search("外注しない")["meetings"][0]["hits"][0]
        self.assertEqual((h["kind"], h["i"]), ("mark", 2))
        self.assertEqual(self.ix.search("再来月")["meetings"][0]["hits"][0]["i"], 0)   # 直した後の文
        self.assertEqual(self.ix.search("来週までに")["total"], 0)


if __name__ == "__main__":
    unittest.main()
