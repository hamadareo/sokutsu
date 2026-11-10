"""会議中の道具(W3, 11/07「他の会議メモアプリの機能を全部」):
ブックマーク・自分のメモ・議題・書き起こしの修正(文と話者)・用語辞書。

記録(logs/session-*.jsonl)に足す行と、それを読む notes.load_meeting の約束
(設計メモ(公開版には含めない) の「記録のデータの約束」)を固定する。
"""
import json
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from tsuyaku import inmeeting as IM
from tsuyaku import notes as N
from tsuyaku.__main__ import Bridge
from tsuyaku.guards import echoes_prompt, stt_prompt
from tsuyaku.log import SessionLog

ROOT = Path(__file__).resolve().parents[1]

ROWS = [
    {"t": "10:00:05", "lang": "ja", "orig": "山田です。今日は進め方を決めましょう。", "trans": "I'm Yamada.",
     "spk": "S1", "ts": 1000.0},
    {"t": "10:00:12", "lang": "en", "orig": "Hi, I'm Smith. I will send the slides by Friday.",
     "trans": "スミスです。", "spk": "S2", "ts": 1007.0},
    {"t": "10:01:40", "lang": "ja", "orig": "ではそれで決定です。", "trans": "Then it's decided.",
     "spk": "S1", "ts": 1095.0},
]


def write_log(d, rows, meta=None, stem="session-20261107-100000", extra=()):
    """utter 行(rows)の間に extra の (位置, 行) を差し込んだ記録を書く。位置 k = k 行目の utter の前。"""
    p = Path(d) / f"{stem}.jsonl"
    ins: dict = {}
    for k, r in extra:
        ins.setdefault(k, []).append(r)
    with p.open("w", encoding="utf-8") as f:
        f.write(json.dumps(meta or {"type": "meta", "pair": "ja-en", "topic": "",
                                    "start": "2026-11-07 10:00:00"}, ensure_ascii=False) + "\n")
        for k, r in enumerate(rows):
            for x in ins.get(k, []):
                f.write(json.dumps(x, ensure_ascii=False) + "\n")
            f.write(json.dumps(dict({"type": "utter"}, **r), ensure_ascii=False) + "\n")
        for x in ins.get(len(rows), []):
            f.write(json.dumps(x, ensure_ascii=False) + "\n")
    return p


def append(p, *rows):
    with Path(p).open("a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# ---------------- 記録を読む(notes.load_meeting) ----------------

class TestMarks(unittest.TestCase):
    def test_mark_points_to_last_line_before_it_and_sorts_by_time(self):
        with tempfile.TemporaryDirectory() as d:
            p = write_log(d, ROWS, extra=[
                (2, {"type": "mark", "id": "a", "ts": 1010.0, "t": "10:00:15", "kind": "star", "label": ""}),
                (1, {"type": "mark", "id": "b", "ts": 1001.0, "t": "10:00:06", "kind": "todo", "label": "宿題"}),
            ])
            m = N.load_meeting(p)
        self.assertEqual([x["id"] for x in m["marks"]], ["b", "a"])          # 時刻順
        self.assertEqual([x["i"] for x in m["marks"]], [0, 1])                # その時の最後の行
        self.assertEqual(m["marks"][0]["label"], "宿題")

    def test_explicit_line_index_wins_and_is_clamped(self):
        with tempfile.TemporaryDirectory() as d:
            p = write_log(d, ROWS, extra=[
                (1, {"type": "mark", "id": "a", "ts": 1001.0, "kind": "star", "i": 1}),   # 話している最中 = 次の行
                (3, {"type": "mark", "id": "b", "ts": 1100.0, "kind": "star", "i": 9}),   # 記録より先 → 最後の行
                (0, {"type": "mark", "id": "c", "ts": 999.0, "kind": "star"}),            # 1行目より前 → 0
            ])
            m = N.load_meeting(p)
        got = {x["id"]: x["i"] for x in m["marks"]}
        self.assertEqual(got, {"a": 1, "b": 2, "c": 0})

    def test_later_rows_with_same_id_update_and_del_removes(self):
        with tempfile.TemporaryDirectory() as d:
            p = write_log(d, ROWS, extra=[(2, {"type": "mark", "id": "a", "ts": 1010.0, "kind": "star"}),
                                          (2, {"type": "mark", "id": "b", "ts": 1011.0, "kind": "star"})])
            append(p, {"type": "mark", "id": "a", "kind": "decision", "label": "B案"},
                   {"type": "del", "id": "b"})
            m = N.load_meeting(p)
        self.assertEqual(len(m["marks"]), 1)
        a = m["marks"][0]
        self.assertEqual((a["kind"], a["label"], a["i"], a["ts"]), ("decision", "B案", 1, 1010.0))

    def test_unknown_kind_becomes_star(self):
        with tempfile.TemporaryDirectory() as d:
            p = write_log(d, ROWS, extra=[(1, {"type": "mark", "id": "a", "ts": 1001.0, "kind": "??"})])
            m = N.load_meeting(p)
        self.assertEqual(m["marks"][0]["kind"], "star")

    def test_no_marks_rows_gives_empty_lists(self):
        with tempfile.TemporaryDirectory() as d:
            m = N.load_meeting(write_log(d, ROWS))
        for k in ("marks", "memos", "agenda"):
            self.assertEqual(m[k], [], k)
        self.assertEqual(m["template"], "")
        self.assertEqual(m["terms"], [])


class TestMemos(unittest.TestCase):
    def test_memos_keep_text_position_and_can_be_edited_or_deleted(self):
        with tempfile.TemporaryDirectory() as d:
            p = write_log(d, ROWS, extra=[
                (2, {"type": "memo", "id": "m1", "ts": 1008.0, "t": "10:00:13", "text": "スライドは金曜"}),
                (3, {"type": "memo", "id": "m2", "ts": 1096.0, "t": "10:01:41", "text": "消す"}),
            ])
            append(p, {"type": "memo", "id": "m1", "text": "スライドは金曜まで(Smith)"}, {"type": "del", "id": "m2"},
                   {"type": "memo", "id": "m3", "ts": 1097.0, "text": "   "})     # 空のメモは無視
            m = N.load_meeting(p)
        self.assertEqual([(x["id"], x["text"], x["i"]) for x in m["memos"]],
                         [("m1", "スライドは金曜まで(Smith)", 1)])


class TestAgendaTemplateTerms(unittest.TestCase):
    def test_agenda_text_becomes_items_and_check_rows_mark_done(self):
        meta = {"type": "meta", "pair": "ja", "start": "2026-11-07 10:00:00", "template": "regular",
                "agenda": "1. 先週の振り返り\n- 予算の確認\n\n・ 来週の予定 \n", "terms": ["YOLO", "山田花子"]}
        with tempfile.TemporaryDirectory() as d:
            p = write_log(d, ROWS, meta=meta, extra=[
                (2, {"type": "agenda", "k": 0, "done": True, "ts": 1010.0, "t": "10:00:15"}),
                (3, {"type": "agenda", "k": 1, "done": True, "ts": 1100.0, "t": "10:01:45"}),
                (3, {"type": "agenda", "k": 1, "done": False, "ts": 1101.0, "t": "10:01:46"}),
                (3, {"type": "agenda", "k": 7, "done": True}),      # 無い項目は無視
            ])
            m = N.load_meeting(p)
        self.assertEqual(m["agenda"], [{"text": "先週の振り返り", "done": True, "t": "10:00:15"},
                                       {"text": "予算の確認", "done": False, "t": ""},
                                       {"text": "来週の予定", "done": False, "t": ""}])
        self.assertEqual(m["template"], "regular")
        self.assertEqual(m["terms"], ["YOLO", "山田花子"])

    def test_agenda_items_parser(self):
        self.assertEqual(N.agenda_items("① 開会\n2) 報告\n* まとめ"), ["開会", "報告", "まとめ"])
        self.assertEqual(N.agenda_items(""), [])
        self.assertEqual(N.agenda_items(None), [])


class TestEdits(unittest.TestCase):
    def test_edit_replaces_text_and_keeps_original(self):
        with tempfile.TemporaryDirectory() as d:
            p = write_log(d, ROWS)
            append(p, {"type": "edit", "i": 0, "orig": "山口です。今日は進め方を決めましょう!"},
                   {"type": "edit", "i": 0, "orig": "山口です。今日は進め方を決めましょう。"})
            m = N.load_meeting(p)
        ln = m["lines"][0]
        self.assertEqual(ln["orig"], "山口です。今日は進め方を決めましょう。")
        self.assertEqual(ln["orig0"], "山田です。今日は進め方を決めましょう。")
        self.assertNotIn("orig0", m["lines"][1])

    def test_respk_wins_over_alias_and_refine(self):
        rows = [dict(ROWS[0], segs=[0]), dict(ROWS[1], segs=[1]), dict(ROWS[2], segs=[2])]
        with tempfile.TemporaryDirectory() as d:
            p = write_log(d, rows)
            append(p, {"type": "respk", "i": 2, "spk": "S3"},
                   {"type": "alias", "map": {"S3": "S1"}},
                   {"type": "refine", "ids": ["S1", "S2", "S2"]})
            m = N.load_meeting(p)
        self.assertEqual([ln["spk"] for ln in m["lines"]], ["S1", "S2", "S3"])
        self.assertEqual(m["lines"][2]["spk0"], "S2")      # 付け替える前(付け直しの結果)

    def test_undo_cancels_last_edit_ops_in_order(self):
        with tempfile.TemporaryDirectory() as d:
            p = write_log(d, ROWS)
            append(p, {"type": "edit", "i": 1, "orig": "Hi, I'm Smyth."},
                   {"type": "respk", "i": 1, "spk": "S1"},
                   {"type": "mark", "id": "x", "ts": 1200.0, "kind": "star"},   # 印は元に戻すの対象外
                   {"type": "undo"})
            m = N.load_meeting(p)
            self.assertEqual((m["lines"][1]["orig"], m["lines"][1]["spk"]), ("Hi, I'm Smyth.", "S2"))
            self.assertEqual(len(m["marks"]), 1)
            self.assertEqual(m["undo"], 1)
            append(p, {"type": "undo"}, {"type": "undo"})          # 戻すものが無い undo は何もしない
            m = N.load_meeting(p)
        self.assertEqual(m["lines"][1]["orig"], ROWS[1]["orig"])
        self.assertNotIn("orig0", m["lines"][1])
        self.assertEqual(m["undo"], 0)

    def test_bad_edit_rows_are_ignored(self):
        with tempfile.TemporaryDirectory() as d:
            p = write_log(d, ROWS)
            append(p, {"type": "edit", "i": 9, "orig": "x"}, {"type": "edit", "i": "0", "orig": "x"},
                   {"type": "edit", "i": 0, "orig": "  "}, {"type": "respk", "i": 0, "spk": "<b>"},
                   {"type": "respk", "i": -1, "spk": "S2"})
            m = N.load_meeting(p)
        self.assertEqual([ln["orig"] for ln in m["lines"]], [r["orig"] for r in ROWS])
        self.assertEqual([ln["spk"] for ln in m["lines"]], ["S1", "S2", "S1"])

    def test_notes_use_corrected_text(self):
        seen = []

        def llm(msgs):
            seen.append(msgs[-1]["content"])
            return json.dumps({"title": "t", "participants": [], "summary": [], "topics": [],
                               "decisions": [], "todos": [], "questions": [], "keywords": []})
        with tempfile.TemporaryDirectory() as d:
            p = write_log(d, ROWS)
            append(p, {"type": "edit", "i": 2, "orig": "ではB案で決定です。"})
            N.make_notes(N.load_meeting(p), "ja", llm)
        self.assertIn("ではB案で決定です。", seen[0])
        self.assertNotIn("ではそれで決定です。", seen[0])


# ---------------- 書く側(SessionLog と Bridge) ----------------

class TestSessionLog(unittest.TestCase):
    def test_extra_goes_into_meta(self):
        with tempfile.TemporaryDirectory() as d:
            log = SessionLog(d, "ja", "", "notes", extra={"agenda": "a\nb", "terms": ["YOLO"]})
            log.close()
            meta = json.loads(Path(log.path_jsonl).read_text().splitlines()[0])
        self.assertEqual((meta["agenda"], meta["terms"], meta["mode"]), ("a\nb", ["YOLO"], "notes"))

    def test_add_row_is_safe_from_two_threads(self):
        with tempfile.TemporaryDirectory() as d:
            log = SessionLog(d, "ja-en")

            def utt():
                for k in range(200):
                    log.add("ja", f"文{k}" * 20, "", None)

            def mk():
                for k in range(200):
                    log.add_row({"type": "mark", "id": f"m{k}", "ts": 1.0, "kind": "star", "label": "x" * 50})
            ts = [threading.Thread(target=utt), threading.Thread(target=mk)]
            [t.start() for t in ts]
            [t.join() for t in ts]
            log.close()
            rows = [json.loads(x) for x in Path(log.path_jsonl).read_text().splitlines()]
        self.assertEqual(len(rows), 401)
        self.assertEqual(getattr(log, "n_rows", 0), 200)


class TestBridge(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.d = self.tmp.name
        self.b = Bridge()
        self.b.s.log_dir = self.d

    def tearDown(self):
        if self.b.log is not None:
            self.b.log.close()
        self.tmp.cleanup()

    def live(self, n=3):
        self.b.log = SessionLog(self.d, "ja-en", "", "both")
        for r in ROWS[:n]:
            self.b.log.add(r["lang"], r["orig"], r["trans"], None, r["spk"])
        return self.b.log

    def test_live_mark_kind_label_and_delete(self):
        self.live()
        mk = self.b.mark_set("live", {"i": 2})
        self.assertTrue(mk["id"])
        self.assertEqual((mk["kind"], mk["i"]), ("star", 2))
        self.assertRegex(mk["t"], r"^\d\d:\d\d:\d\d$")
        self.b.mark_set("live", {"id": mk["id"], "kind": "decision", "label": "B案に決定"})
        g = self.b.meeting_get("live")
        self.assertEqual([(x["kind"], x["label"]) for x in g["marks"]], [("decision", "B案に決定")])
        self.assertEqual(self.b.mark_del("live", mk["id"]), "ok")
        self.assertEqual(self.b.meeting_get("live")["marks"], [])

    def test_live_mark_while_someone_is_talking_points_to_next_line(self):
        self.live(2)
        mk = self.b.mark_set("live", {"i": 2})     # 画面: 途中字幕が出ている = 次の行
        self.b.log.add("ja", ROWS[2]["orig"], "", None, "S1")
        g = self.b.meeting_get("live")
        self.assertEqual(g["marks"][0]["i"], 2)
        self.assertEqual(mk["kind"], "star")

    def test_bad_kind_and_long_label_are_cleaned(self):
        self.live()
        mk = self.b.mark_set("live", {"kind": "<script>", "label": "x" * 500})
        self.assertEqual(mk["kind"], "star")
        self.assertLessEqual(len(mk["label"]), 200)

    def test_memo_add_edit_delete(self):
        self.live()
        mm = self.b.memo_set("live", {"text": "  予算は来週  ", "i": 1})
        self.assertEqual(mm["text"], "予算は来週")
        self.assertEqual(self.b.memo_set("live", {"text": "   "}), None)
        self.b.memo_set("live", {"id": mm["id"], "text": "予算は来週に再確認"})
        g = self.b.meeting_get("live")
        self.assertEqual([(x["text"], x["i"]) for x in g["memos"]], [("予算は来週に再確認", 1)])
        self.b.memo_del("live", mm["id"])
        self.assertEqual(self.b.meeting_get("live")["memos"], [])

    def test_agenda_check_live(self):
        self.b._start_extra = IM.meta_extra({"agenda": "報告\n予算"})
        self.b.log = SessionLog(self.d, "ja", "", "notes", extra=self.b._meta_extra())
        self.b.log.add("ja", "始めます", "", None, "S1")
        self.assertEqual(self.b.agenda_check("live", 1, True), "ok")
        self.assertEqual(self.b.agenda_check("live", 5, True), "none")
        g = self.b.meeting_get("live")
        self.assertEqual([a["done"] for a in g["agenda"]], [False, True])
        self.assertRegex(g["agenda"][1]["t"], r"^\d\d:\d\d:\d\d$")

    def test_agenda_check_after_the_meeting_has_no_time(self):
        meta = {"type": "meta", "pair": "ja", "start": "2026-11-07 10:00:00", "agenda": "報告\n予算"}
        p = write_log(self.d, ROWS, meta=meta)
        self.assertEqual(self.b.agenda_check(p.stem, 0, True), "ok")
        self.assertEqual(N.load_meeting(p)["agenda"][0], {"text": "報告", "done": True, "t": ""})

    def test_edit_respeaker_and_undo_on_past_meeting(self):
        p = write_log(self.d, ROWS)
        mid = p.stem
        ln = self.b.line_edit(mid, 2, "ではB案で決定です。")
        self.assertEqual(ln["orig"], "ではB案で決定です。")
        self.assertEqual(self.b.line_speaker(mid, 2, "S3")["spk"], "S3")
        self.assertEqual(self.b.line_edit(mid, 99, "x"), None)
        self.assertEqual(self.b.line_edit(mid, 0, "   "), None)
        self.assertEqual(self.b.line_speaker(mid, 0, "javascript:"), None)
        g = self.b.meeting_get(mid)
        self.assertEqual((g["lines"][2]["orig"], g["lines"][2]["spk"], g["undo"]), ("ではB案で決定です。", "S3", 2))
        u = self.b.edit_undo(mid)
        self.assertEqual(u["i"], 2)
        self.assertEqual(self.b.meeting_get(mid)["lines"][2]["spk"], "S1")
        self.b.edit_undo(mid)
        self.assertEqual(self.b.meeting_get(mid)["lines"][2]["orig"], ROWS[2]["orig"])
        self.assertEqual(self.b.edit_undo(mid), None)

    def test_writing_to_a_crashed_log_starts_on_a_new_line(self):
        p = write_log(self.d, ROWS)
        with p.open("a") as f:
            f.write('{"type": "utter", "t": "10:0')     # 落ちて書きかけ
        self.b.mark_set(p.stem, {"i": 0})
        m = N.load_meeting(p)
        self.assertEqual(len(m["marks"]), 1)
        self.assertEqual(len(m["lines"]), 3)

    def test_past_mark_takes_the_line_time(self):
        p = write_log(self.d, ROWS)
        mk = self.b.mark_set(p.stem, {"i": 1, "kind": "todo"})
        self.assertEqual((mk["ts"], mk["t"], mk["kind"]), (1007.0, "10:00:12", "todo"))

    def test_live_meeting_by_session_id_goes_through_the_open_log(self):
        log = self.live()
        mid = Path(log.path_jsonl).stem
        with patch.object(log, "add_row", wraps=log.add_row) as ar:
            self.b.line_edit(mid, 0, "直した")
        ar.assert_called_once()

    def test_unknown_meeting_and_paths_outside_logs(self):
        self.assertEqual(self.b.mark_set("../../etc/passwd", {"i": 0}), None)
        self.assertEqual(self.b.memo_set("live", {"text": "x"}), None)      # 会話していない
        self.assertEqual(self.b.edit_undo("session-nope"), None)

    def test_start_keeps_agenda_and_dictionary_for_the_meta(self):
        with tempfile.TemporaryDirectory() as d2:
            tp = Path(d2) / "terms.json"
            IM.save_terms(["YOLO", "山田花子"], tp)
            with patch("tsuyaku.__main__.save_config"), patch.object(Bridge, "_run"), \
                    patch.object(IM, "terms_path", return_value=tp):
                self.b.start("ja", "", "notes", {"agenda": "報告\n予算", "evil": "x", "template": "1on1"})
        self.assertEqual(self.b._start_extra, {"agenda": "報告\n予算", "template": "1on1",
                                              "terms": ["YOLO", "山田花子"]})


# ---------------- 用語辞書 ----------------

class TestTerms(unittest.TestCase):
    def test_normalize(self):
        self.assertEqual(IM.normalize_terms(["  YOLO ", "yolo", "", "山田花子", "a、b", None, 3, "x" * 99]),
                         ["YOLO", "山田花子", "a", "b"])

    def test_save_and_load(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "terms.json"
            self.assertEqual(IM.load_terms(p), [])
            IM.save_terms(["Qwen", "RAG"], p)
            self.assertEqual(IM.load_terms(p), ["Qwen", "RAG"])
            p.write_text("{broken")
            self.assertEqual(IM.load_terms(p), [])

    def test_bridge_terms_get_set(self):
        b = Bridge()
        with tempfile.TemporaryDirectory() as d, \
                patch.object(IM, "terms_path", return_value=Path(d) / "terms.json"):
            self.assertEqual(b.terms_set(["ProjectX", "projectx", "ボクセル"]), ["ProjectX", "ボクセル"])
            self.assertEqual(b.terms_get(), ["ProjectX", "ボクセル"])

    def test_terms_file_is_not_committed(self):
        self.assertIn("terms.json", (ROOT / ".gitignore").read_text().split())


class TestSttPromptWithTerms(unittest.TestCase):
    def test_topic_first_then_dictionary_without_duplicates(self):
        self.assertEqual(stt_prompt("YOLO・物体検出", terms=["yolo", "山田花子", "Qwen"]),
                         "YOLO、物体検出、山田花子、Qwen。")
        self.assertEqual(stt_prompt("", terms=["Qwen", "RAG"]), "Qwen、RAG。")
        self.assertEqual(stt_prompt("YOLO・物体検出"), "YOLO、物体検出。")      # 今までどおり

    def test_long_dictionary_is_cut_at_a_term_boundary(self):
        terms = [f"用語{k:03d}" for k in range(100)]
        out = stt_prompt("テーマ", terms=terms, max_len=60)
        self.assertLessEqual(len(out), 61)
        self.assertTrue(out.endswith("。"))
        for it in out.rstrip("。").split("、"):
            self.assertTrue(it == "テーマ" or it in terms, it)     # 語の途中で切らない

    def test_echo_guard_still_works_with_dictionary(self):
        p = stt_prompt("", terms=["Qwen", "RAG", "ProjectX"])
        self.assertTrue(echoes_prompt("Qwen、RAG、ProjetX。", p))
        self.assertFalse(echoes_prompt("Qwenを使いました。", p))

    def test_app_feeds_dictionary_to_the_ear(self):
        src = (ROOT / "src/tsuyaku/__main__.py").read_text()
        self.assertIn("self.stt.prompt = stt_prompt(topic, terms=", src)
        self.assertIn("if echoes_prompt(text, self.stt.prompt):", src)


class TestSpellingHint(unittest.TestCase):
    def test_hint_and_notes_prompt(self):
        m = {"id": "x", "meta": {}, "lines": [dict(ROWS[0])], "terms": ["山田花子", "YOLO"]}
        self.assertIn("山田花子", N.spelling_hint(m))
        self.assertEqual(N.spelling_hint({"terms": []}), "")
        seen = []

        def llm(msgs):
            seen.append(msgs[-1]["content"])
            return json.dumps({"title": "t"})
        N.make_notes(m, "ja", llm)
        self.assertIn("山田花子", seen[0])
        self.assertIn("YOLO", seen[0])


class TestMarkdown(unittest.TestCase):
    def test_marks_memos_agenda_in_markdown(self):
        with tempfile.TemporaryDirectory() as d:
            meta = {"type": "meta", "pair": "ja-en", "start": "2026-11-07 10:00:00", "agenda": "進め方\n日程"}
            p = write_log(d, ROWS, meta=meta, extra=[
                (2, {"type": "mark", "id": "a", "ts": 1010.0, "t": "10:00:15", "kind": "todo", "label": "スライド"}),
                (3, {"type": "memo", "id": "m", "ts": 1096.0, "t": "10:01:41", "text": "B案で進める"}),
                (2, {"type": "agenda", "k": 0, "done": True, "t": "10:00:15"})])
            m = N.load_meeting(p)
        md = N.to_markdown({"lang": "ja", "title": "t", "participants": []}, m)
        self.assertIn("## 議題", md)
        self.assertIn("- [x] 進め方", md)
        self.assertIn("- [ ] 日程", md)
        self.assertIn("## ブックマークと自分のメモ", md)
        self.assertIn("`10:00:15`", md)
        self.assertIn("スライド", md)
        self.assertIn("B案で進める", md)
        en = N.to_markdown({"lang": "en", "title": "t", "participants": []}, m)
        self.assertIn("## Agenda", en)


if __name__ == "__main__":
    unittest.main()


class TestUI(unittest.TestCase):
    """画面(index.html)の配線。動きは headless Chrome と WKWebView でモックの API を使って確かめた(11/07)。"""
    HTML = (ROOT / "src/tsuyaku/webui/index.html").read_text()

    def test_elements_exist(self):
        for i in ('id="markBtn"', 'id="markPop"', 'id="markNav"', 'id="memoPane"', 'id="memoIn"',
                  'id="agendaRail"', 'id="agenda"', 'id="agendaChip"', 'id="termsChip"', 'id="termsModal"'):
            self.assertIn(i, self.HTML, i)

    def test_start_sends_extra_only_when_needed(self):
        h = self.HTML
        self.assertIn("const ex=TSUYAKU.startExtra();", h)
        self.assertIn("window.pywebview.api.start(TSUYAKU.pair,tp,TSUYAKU.mode,src,ex)", h)
        self.assertIn('if(this.mode!=="interpret"){const a=((this.$("#agenda")||{}).value||"").trim(); if(a)ex.agenda=a;}', h)

    def test_hooks_into_meeting_doc(self):
        h = self.HTML
        self.assertIn("secs.push(...this.w3Secs());", h)
        self.assertIn("h+=this.w3AgendaHtml()+this.w3MarksHtml();", h)
        self.assertIn('data-i="\'+i+\'"', h)
        self.assertIn("this.w3LnTools(i)", h)

    def test_keys_do_not_fire_while_typing_and_esc_does_not_quit(self):
        h = self.HTML
        i = h.index('<script id="w3-js">')
        js = h[i:]
        self.assertIn('if(e.metaKey||e.ctrlKey||!T.live||typing(ae)', js)
        self.assertIn('if(e.key==="b"||e.key==="B")', js)
        self.assertIn('if(e.key==="Escape"){e.preventDefault();ag.blur();}', js)
        self.assertIn('},true);', js)      # 札・話者の選択の Esc は先に取る

    def test_w3_strings_have_three_translations(self):
        import re
        i, j = self.HTML.index("/*W3-I18N-BEGIN"), self.HTML.index("/*W3-I18N-END*/")
        block = self.HTML[i:j]
        a = self.HTML.index('<script id="w3-js">')
        js = self.HTML[a:self.HTML.index("</script>", a)]   # W3 のスクリプトだけ(後ろの W2 の塊は含めない)
        keys = set(re.findall(r'this\.L\("([^"]+)"', js))
        keys -= {"保存", "閉じる"}           # 既にある訳
        self.assertGreater(len(keys), 25)
        for k in keys:
            self.assertEqual(block.count('"' + k + '":'), 3, k)


class TestStartSourceAndExtra(unittest.TestCase):
    """マージ(11/07): 聞く音(onlinemeet)と議題など(W3)を一緒に渡しても、古い4引数の呼び方でも受ける。"""

    def test_both_and_legacy(self):
        b = Bridge()
        with patch("tsuyaku.__main__.save_config"), patch.object(Bridge, "_run"), \
                patch.object(IM, "load_terms", return_value=[]):
            b.start("ja", "", "notes", "mic+system", {"agenda": "報告"})
            self.assertEqual(b.s.audio_source, "mic+system")
            self.assertEqual(b._meta_extra(), {"agenda": "報告", "source": "mic+system"})
            b.start("ja", "", "notes", {"agenda": "予算"})          # 古い画面
            self.assertEqual(b._start_extra, {"agenda": "予算"})
            self.assertEqual(b.s.audio_source, "mic+system")      # 来なければ前回のまま


class TestFit(unittest.TestCase):
    """11/07 窓 1100〜1440px × 4言語で会話画面がはみ出さない(headless Chrome と WKWebView で測った)。"""
    HTML = (ROOT / "src/tsuyaku/webui/index.html").read_text()

    def test_rail_head_secondary_actions_are_icons(self):
        h = self.HTML
        self.assertIn('<button id="openLog" class="tb ic"', h)
        self.assertIn('<button id="copyLog" class="tb ic"', h)
        self.assertIn("@container rail (max-width:340px)", h)
        # 「コピーしました」は文字だけ替える(形のボタンを消さない)
        self.assertIn('const lb=btn.querySelector(".lbl")||btn', h)

    def test_mic_note_takes_only_the_space_left(self):
        self.assertIn("#micNote{display:none;align-items:center;gap:9px;min-width:0;flex:1 1 0;overflow:hidden;", self.HTML)

    def test_rail_line_head_keeps_time(self):
        self.assertIn(".item .ihead .badge{min-width:0;overflow:hidden;white-space:nowrap}", self.HTML)
        # 言語名は独立した要素で、そこだけが縮む(話者名は縮めない: WebKit は 1px 未満でも省略記号を出す)
        self.assertIn("""'<span class="bl">'+(notesOnly&&one?"":""", self.HTML)
        self.assertIn(".item .ihead .badge .spk{min-width:0;flex:none}", self.HTML)
        self.assertIn(".item .ihead .badge .bl{min-width:0;overflow:hidden;text-overflow:ellipsis;flex:0 1 auto}", self.HTML)
