"""W1(11/07 「他の会議メモアプリの機能を全部」): 章立て・メモの項目→元の発言・テンプレート・議題・自分のメモ。

見つけた問題: メモの要約・決定・やることがどの発言から来たか分からず、確かめるには全文を探すしかなかった
(tl;dv / Otter / Gemini は項目から元の発言へ飛べる)。会議の種類(定例・1on1・商談…)でメモの重みを
変えられなかった(Notion / tl;dv / Fireflies はテンプレートを選べる)。
変更: メモを今までどおり作った後で、番号つきの記録(#12)と項目を並べて「どの行が支えるか」を LLM に
答えさせ(notes.ground)、*_src / src に入れる。話題の始まりの行 → 章(時刻)を作る。
(はじめは本文に [#12] を書かせたが、AMI 6会議×3回で要約 −6点・やること −13点だったので後段に分けた)
"""
import re
import json
import tempfile
import unittest
from pathlib import Path

from tsuyaku import notes as N
from tsuyaku import note_templates as T


def write_log(d, rows, meta=None, extra_rows=(), stem="session-20261107-100000"):
    p = Path(d) / f"{stem}.jsonl"
    with p.open("w", encoding="utf-8") as f:
        f.write(json.dumps(meta or {"type": "meta", "pair": "ja-en", "topic": "",
                                    "start": "2026-11-07 10:00:00"}, ensure_ascii=False) + "\n")
        for r in rows:
            f.write(json.dumps(dict({"type": "utter"}, **r), ensure_ascii=False) + "\n")
        for r in extra_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return p


ROWS = [
    {"t": "10:00:05", "lang": "ja", "orig": "山田です。今日は進め方を決めましょう。", "trans": "", "spk": "S1", "ts": 1000.0},
    {"t": "10:00:12", "lang": "en", "orig": "Hi, I'm Smith. I will send the slides by Friday.", "trans": "", "spk": "S2", "ts": 1007.0},
    {"t": "10:01:40", "lang": "ja", "orig": "予算は来週もう一度見ます。", "trans": "", "spk": "S1", "ts": 1095.0},
    {"t": "10:02:10", "lang": "ja", "orig": "ではそれで決定です。", "trans": "", "spk": "S1", "ts": 1125.0},
]

REFS = {
    "title": "進め方の打ち合わせ",
    "participants": [{"id": "S1", "name": "山田", "evidence": "山田です"},
                     {"id": "S2", "name": "Smith", "evidence": "I'm Smith"}],
    "summary": ["進め方を決めた", "Smith がスライドを送る", "根拠なしの文"],
    "topics": [{"title": "予算", "points": [{"who": "S1", "text": "来週もう一度見る"}]},
               {"title": "進め方", "points": [{"who": "S2", "text": "金曜までにスライド"},
                                           {"who": "S1", "text": "決定する"}]}],
    "decisions": ["それで決定"],
    "todos": [{"task": "スライドを送る", "owner": "S2", "due": "金曜", "quote": "I will send the slides by Friday."},
              {"task": "予算を見直す", "owner": "S1", "due": "", "quote": "予算は来週もう一度見ます"}],
    "questions": ["日程は?"], "keywords": ["スライド"],
}
# 根拠を当てる段(ground)の答え: 項目の文 → 行番号(99 は記録に無い)
GROUND = {"進め方を決めた": [0, 3], "Smith がスライドを送る": [1], "来週もう一度見る": [2],
          "金曜までにスライド": [1, 99], "決定する": [3], "それで決定": [3], "スライドを送る (said": [1],
          "日程は?": [2, 3], "付箋で集める": [0]}


def fake(gen=None, calls=None, extra=None):
    """メモを作る LLM の代わり。根拠を当てる段には GROUND で答える。"""
    def llm(msgs):
        if calls is not None:
            calls.append(msgs)
        if "link meeting-note items" in msgs[0]["content"]:
            u = msgs[1]["content"]
            refs = {}
            for k, text in re.findall(r"^(\d+)\. (.*)$", u.split("Transcript excerpt:")[0], re.M):
                for key, v in GROUND.items():
                    if text.startswith(key):
                        refs[k] = v
            return json.dumps({"refs": refs})
        if extra:
            r = extra(msgs)
            if r is not None:
                return r
        return json.dumps(gen or REFS, ensure_ascii=False)
    return llm


class TestRefs(unittest.TestCase):
    def test_split_refs(self):
        self.assertEqual(N.split_refs("進め方を決めた [#0, #3]"), ("進め方を決めた", [0, 3]))
        self.assertEqual(N.split_refs("A [#2-#4] B (#7)"), ("A B", [2, 3, 4, 7]))
        self.assertEqual(N.split_refs("no refs"), ("no refs", []))
        self.assertEqual(N.split_refs("x [#5][#5]", n=4), ("x", []))     # 範囲外は捨てる
        self.assertEqual(N.split_refs("price is #1 priority"), ("price is #1 priority", []))  # 括弧なしは本文

    def test_transcript_is_numbered(self):
        lines = [dict(r, i=k) for k, r in enumerate(ROWS)]
        tx = N.transcript(lines)
        self.assertIn("#1 [10:00:12] S2 (en): Hi, I'm Smith", tx)

    def test_make_notes_keeps_sources_and_chapters(self):
        calls = []
        with tempfile.TemporaryDirectory() as d:
            m = N.load_meeting(write_log(d, ROWS))
            n = N.make_notes(m, "ja", fake(calls=calls))
        # 本文を作る指示は今までどおり(番号も [#n] の指示も無い)
        self.assertNotIn("#1 [", calls[0][1]["content"])
        self.assertIn("[10:00:12] S2 (en): Hi, I'm Smith", calls[0][1]["content"])
        self.assertNotIn("[#", calls[0][0]["content"])
        # 後段: 番号つきの記録と項目
        self.assertEqual(len(calls), 2)
        self.assertIn("#1 [10:00:12] S2 (en)", calls[1][1]["content"])
        self.assertIn("1. 進め方を決めた", calls[1][1]["content"])
        self.assertEqual(n["title"], "進め方の打ち合わせ")
        self.assertEqual(n["summary"], ["進め方を決めた", "Smith がスライドを送る", "根拠なしの文"])
        self.assertEqual(n["summary_src"], [[0, 3], [1], []])
        self.assertEqual(n["decisions"], ["それで決定"])
        self.assertEqual(n["decisions_src"], [[3]])
        self.assertEqual(n["questions_src"], [[2, 3]])
        t0, t1 = n["topics"]
        self.assertEqual(t1["title"], "進め方")
        self.assertEqual(t1["points"][0]["src"], [1])                      # 99 は範囲外
        self.assertEqual((t0["i"], t0["t"]), (2, "10:01:40"))
        self.assertEqual((t1["i"], t1["t"]), (1, "10:00:12"))
        # 章は時刻の順
        self.assertEqual([(c["title"], c["t"]) for c in n["chapters"]],
                         [("進め方", "10:00:12"), ("予算", "10:01:40")])
        self.assertEqual(n["todos"][0]["src"], [1])
        self.assertEqual(n["todos"][0]["task"], "スライドを送る")
        self.assertEqual(n["todos"][1]["src"], [2])                         # quote から機械で見つける
        self.assertEqual(n["template"], "free")

    def test_chapter_time_is_from_meeting_start_when_ts(self):
        with tempfile.TemporaryDirectory() as d:
            meta = {"type": "meta", "pair": "ja-en", "start": "2026-11-07 10:00:00", "ts0": 995.0}
            m = N.load_meeting(write_log(d, ROWS, meta=meta))
            n = N.make_notes(m, "ja", fake())
        self.assertEqual([c["at"] for c in n["chapters"]], [12.0, 100.0])  # 録音の先頭からの秒

    def test_long_meeting_prompts_unchanged_and_grounded_per_part(self):
        """長い会議: 区切り・まとめの指示は今までどおり。根拠は約9000字の区切りごとに当て、
        その区切りに無い行の番号は捨てる。"""
        calls = []
        with tempfile.TemporaryDirectory() as d:
            m = N.load_meeting(write_log(d, ROWS))
            m["lines"] = m["lines"] * 60
            n = N.make_notes(m, "ja", fake(calls=calls), chunk_chars=600)
        merge = [c for c in calls if "Merge them" in c[1]["content"]]
        self.assertTrue(merge)
        self.assertNotIn("[#", merge[-1][1]["content"] + merge[-1][0]["content"])
        grounds = [c for c in calls if "link meeting-note items" in c[0]["content"]]
        self.assertEqual(len(grounds), len(N.chunks([dict(x, i=k) for k, x in enumerate(m["lines"])], N.GROUND_CHARS)))
        self.assertGreater(len(grounds), 1)
        self.assertEqual(n["summary_src"][0], [0, 3])          # 2つ目以降の区切りの [0, 3] は捨てる

    def test_ground_failure_keeps_notes(self):
        def llm(msgs):
            if "link meeting-note items" in msgs[0]["content"]:
                return "oops"
            return json.dumps(REFS, ensure_ascii=False)
        with tempfile.TemporaryDirectory() as d:
            n = N.make_notes(N.load_meeting(write_log(d, ROWS)), "ja", llm)
        self.assertEqual(n["summary"], REFS["summary"])
        self.assertEqual(n["summary_src"], [[], [], []])
        self.assertEqual(n["todos"][1]["src"], [2])               # 引用からは見つける

    def test_markdown_has_chapter_times_and_no_refs(self):
        with tempfile.TemporaryDirectory() as d:
            m = N.load_meeting(write_log(d, ROWS))
            n = N.make_notes(m, "ja", fake())
            md = N.to_markdown(n, m)
        self.assertNotIn("[#", md)
        self.assertIn("### 進め方 (10:00:12)", md)

    def test_old_notes_without_src_still_render(self):
        old = {"title": "x", "lang": "ja", "participants": [], "summary": ["a"], "topics": [{"title": "t", "points": []}],
               "decisions": [], "todos": [], "questions": [], "keywords": []}
        self.assertIn("# x", N.to_markdown(old))


class TestTemplates(unittest.TestCase):
    def test_all_templates_have_labels_in_four_languages(self):
        keys = ["free", "regular", "1on1", "sales", "interview", "lecture", "brainstorm", "qa"]
        self.assertEqual(list(T.TEMPLATES), keys)
        for k in keys:
            for lg in ("ja", "en", "zh", "ko"):
                self.assertTrue(T.label(k, lg), (k, lg))
            for sec in T.TEMPLATES[k]["sections"]:
                for lg in ("ja", "en", "zh", "ko"):
                    self.assertTrue(T.section_title(k, sec, lg))

    def test_free_template_adds_nothing(self):
        self.assertEqual(T.context("free", "", [], "ja"), "")
        self.assertEqual(T.context("bogus", "", [], "ja"), "")
        self.assertEqual(T.normalize("bogus"), "free")

    def test_template_changes_prompt_and_sections(self):
        g = dict(REFS, sections={"ideas": ["付箋で集める", "動画にする"], "bogus": ["x"]})
        calls = []
        with tempfile.TemporaryDirectory() as d:
            m = N.load_meeting(write_log(d, ROWS))
            n = N.make_notes(m, "ja", fake(gen=g, calls=calls), template="brainstorm")
        self.assertNotIn("[#", calls[0][1]["content"])
        u = calls[0][1]["content"]
        self.assertIn("brainstorm", u.lower())
        self.assertIn('"sections"', u)
        self.assertEqual(n["template"], "brainstorm")
        secs = {s["key"]: s for s in n["sections"]}
        self.assertEqual(set(secs), set(T.TEMPLATES["brainstorm"]["sections"]))
        self.assertEqual(secs["ideas"]["items"], [{"text": "付箋で集める", "src": [0]}, {"text": "動画にする", "src": []}])
        md = N.to_markdown(n, m)
        self.assertIn("## " + T.section_title("brainstorm", "ideas", "ja"), md)

    def test_template_from_meta(self):
        with tempfile.TemporaryDirectory() as d:
            meta = {"type": "meta", "pair": "ja", "start": "2026-11-07 10:00:00", "template": "lecture"}
            m = N.load_meeting(write_log(d, ROWS, meta=meta))
            n = N.make_notes(m, "ja", lambda msgs: json.dumps(REFS, ensure_ascii=False))
        self.assertEqual(n["template"], "lecture")


class TestAgendaMemosMarks(unittest.TestCase):
    def test_agenda_goes_into_prompt(self):
        calls = []
        with tempfile.TemporaryDirectory() as d:
            meta = {"type": "meta", "pair": "ja", "start": "2026-11-07 10:00:00",
                    "agenda": "1. 進め方\n2. 予算"}
            m = N.load_meeting(write_log(d, ROWS, meta=meta))
            N.make_notes(m, "ja", lambda msgs: calls.append(msgs) or json.dumps(REFS, ensure_ascii=False))
        self.assertIn("agenda", calls[0][1]["content"].lower())
        self.assertIn("2. 予算", calls[0][1]["content"])

    def test_memos_and_marks_are_woven_in(self):
        """W3 の mark / memo 行(load_meeting が marks / memos を返す前でも、dict に入っていれば使う)。"""
        calls = []

        def llm(msgs):
            calls.append(msgs)
            if "bookmarked" in msgs[1]["content"]:
                return json.dumps({"highlights": [{"k": 0, "text": "スライドの期限が決まった"}],
                                   "my_notes": [{"k": 0, "text": "予算は来週もう一度見る(山田)"}]},
                                  ensure_ascii=False)
            return json.dumps(REFS, ensure_ascii=False)
        with tempfile.TemporaryDirectory() as d:
            m = N.load_meeting(write_log(d, ROWS))
            m["marks"] = [{"ts": 1008.0, "t": "10:00:13", "label": "", "kind": "star"}]
            m["memos"] = [{"ts": 1100.0, "t": "10:01:45", "text": "予算 来週"}]
            n = N.make_notes(m, "ja", llm)
        self.assertEqual(len(calls), 3)                              # 作る・根拠を当てる・織り込む
        self.assertIn("予算 来週", calls[0][1]["content"])          # 本体のメモにも手がかりとして
        self.assertEqual(n["highlights"], [{"t": "10:00:13", "i": 1, "label": "", "kind": "star",
                                            "text": "スライドの期限が決まった", "src": [1]}])
        self.assertEqual(n["my_notes"][0]["memo"], "予算 来週")
        self.assertEqual(n["my_notes"][0]["text"], "予算は来週もう一度見る(山田)")
        self.assertEqual(n["my_notes"][0]["src"], [2])
        md = N.to_markdown(n, m)
        self.assertIn("予算 来週", md)

    def test_w3_rows_from_the_log(self):
        """W3 の mark / memo / meta の agenda(load_meeting が読む形)からそのまま使う。"""
        calls = []

        def llm(msgs):
            calls.append(msgs)
            if "bookmarked" in msgs[1]["content"]:
                return json.dumps({"highlights": [{"k": 0, "text": "決定の瞬間"}], "my_notes": []}, ensure_ascii=False)
            return json.dumps(REFS, ensure_ascii=False)
        extra = [{"type": "mark", "id": "m1", "ts": 1126.0, "t": "10:02:11", "kind": "decision", "label": "決定"},
                 {"type": "memo", "id": "n1", "ts": 1096.0, "t": "10:01:41", "text": "予算 来週", "i": 2}]
        meta = {"type": "meta", "pair": "ja", "start": "2026-11-07 10:00:00", "agenda": "進め方\n予算",
                "template": "regular"}
        with tempfile.TemporaryDirectory() as d:
            m = N.load_meeting(write_log(d, ROWS, meta=meta, extra_rows=extra))
            n = N.make_notes(m, "ja", llm)
        u = calls[0][1]["content"]
        self.assertIn("1. 進め方\n2. 予算", u)                     # 議題(W3 は項目の並びで返す)
        self.assertIn("予算 来週", u)
        self.assertEqual(n["template"], "regular")
        self.assertEqual(n["highlights"][0]["i"], 3)
        self.assertEqual(n["highlights"][0]["kind"], "decision")
        self.assertIn("marked as: decision", [c for c in calls if "bookmarked" in c[1]["content"]][0][1]["content"])
        self.assertEqual(n["my_notes"][0]["i"], 2)

    def test_enrich_failure_keeps_notes(self):
        def llm(msgs):
            if "bookmarked" in msgs[1]["content"]:
                raise RuntimeError("down")
            return json.dumps(REFS, ensure_ascii=False)
        with tempfile.TemporaryDirectory() as d:
            m = N.load_meeting(write_log(d, ROWS))
            m["memos"] = [{"ts": 1100.0, "t": "10:01:45", "text": "予算 来週"}]
            n = N.make_notes(m, "ja", llm)
        self.assertEqual(n["my_notes"][0]["text"], "")          # 自分のメモはそのまま残す
        self.assertEqual(n["my_notes"][0]["memo"], "予算 来週")


if __name__ == "__main__":
    unittest.main()
