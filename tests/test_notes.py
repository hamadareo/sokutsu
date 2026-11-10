"""会議メモ(11/06「Notion のミーティングメモのような機能」)と使い方の選択。"""
import json
import queue
import tempfile
import time
import unittest
from concurrent.futures import Future
from pathlib import Path
from unittest.mock import patch

from tsuyaku import notes as N
from tsuyaku.__main__ import Bridge, _norm_pair, _resolve_spk
from tsuyaku.log import SessionLog


def write_log(d, rows, meta=None, stem="session-20261106-100000"):
    p = Path(d) / f"{stem}.jsonl"
    with p.open("w", encoding="utf-8") as f:
        f.write(json.dumps(meta or {"type": "meta", "pair": "ja-en", "topic": "",
                                    "start": "2026-11-06 10:00:00"}, ensure_ascii=False) + "\n")
        for r in rows:
            f.write(json.dumps(dict({"type": "utter"}, **r), ensure_ascii=False) + "\n")
        f.write('{"type": "utter", "t": "10:0')   # 書きかけの最終行
    return p


ROWS = [
    {"t": "10:00:05", "lang": "ja", "orig": "山田です。今日は進め方を決めましょう。", "trans": "I'm Yamada.", "spk": "S1"},
    {"t": "10:00:12", "lang": "en", "orig": "Hi, I'm Smith. I will send the slides by Friday.", "trans": "スミスです。", "spk": "S2"},
    {"t": "10:01:40", "lang": "ja", "orig": "ではそれで決定です。", "trans": "Then it's decided.", "spk": "S1"},
]

GOOD = {
    "title": "進め方の打ち合わせ",
    "participants": [{"id": "S1", "name": "山田", "evidence": "山田です"},
                     {"id": "S2", "name": "Smith", "evidence": "I'm Smith"}],
    "summary": ["進め方を決めた"],
    "topics": [{"title": "進め方", "points": [{"who": "S2", "text": "金曜までにスライドを送る"}]}],
    "decisions": ["それで決定"],
    "todos": [{"task": "スライドを送る", "owner": "S2", "due": "金曜", "quote": "I will send the slides by Friday."}],
    "questions": [], "keywords": ["スライド"],
}


class TestRead(unittest.TestCase):
    def test_load_and_old_logs_use_language_as_speaker(self):
        with tempfile.TemporaryDirectory() as d:
            rows = [dict(r) for r in ROWS]
            rows[0].pop("spk")
            m = N.load_meeting(write_log(d, rows))
        self.assertEqual(len(m["lines"]), 3)               # 書きかけの行は飛ばす
        self.assertEqual(m["lines"][0]["spk"], "Lja")
        self.assertEqual(N.default_name("Lja", "ja"), "日本語の話者")
        self.assertEqual(N.default_name("S2", "en"), "Speaker 2")

    def test_alias_rows_fix_earlier_lines(self):
        with tempfile.TemporaryDirectory() as d:
            p = write_log(d, ROWS)
            with p.open("a") as f:
                f.write("\n" + json.dumps({"type": "alias", "map": {"S2": "S1"}}) + "\n")
            m = N.load_meeting(p)
        self.assertEqual({ln["spk"] for ln in m["lines"]}, {"S1"})

    def test_refine_rows_relabel_lines_by_segment_votes(self):
        rows = [dict(ROWS[0], spk="S1", segs=[0, 1]), dict(ROWS[1], spk="S5", segs=[2]),
                dict(ROWS[2], spk="S1")]
        with tempfile.TemporaryDirectory() as d:
            p = write_log(d, rows)
            with p.open("a") as f:
                f.write("\n" + json.dumps({"type": "refine", "ids": ["S1", "S1", "S2"]}) + "\n")
            m = N.load_meeting(p)
        self.assertEqual([ln["spk"] for ln in m["lines"]], ["S1", "S2", "S1"])   # 幽霊の S5 → S2

    def test_refine_and_shutdown_wiring(self):
        b = Bridge()

        class Tr:
            def refine(self):
                return ["S1", "S2"]

        class Lg:
            rows = []

            def add_refine(self, ids):
                self.rows.append(ids)

            def close(self):
                pass
        b.spk, b.log = Tr(), Lg()
        b._shutdown()
        b._shutdown()
        self.assertEqual(Lg.rows, [["S1", "S2"]])      # 会話の終わりに1回だけ

    def test_live_relabel_every_15(self):
        b = Bridge()

        class Tr:
            def refine(self):
                return ["S1"] * 20 + ["S2"] * 20

        class Lg:
            n = 0

            def add_refine(self, ids):
                Lg.n += 1
        b.spk, b.log = Tr(), Lg()
        for u in range(1, 16):
            b._spk_track(u, [u + 19])          # 区間 20〜34 → S2
        ev = [b.ui.get() for _ in range(b.ui.qsize())]
        rl = [e[1] for e in ev if e[0] == "relabel"]
        self.assertEqual(len(rl), 1)
        self.assertEqual(rl[0][1], "S2")
        self.assertEqual(Lg.n, 1)

    def test_list_meetings(self):
        with tempfile.TemporaryDirectory() as d:
            write_log(d, ROWS)
            write_log(d, [], stem="session-20261106-090000")     # 空は出さない
            got = N.list_meetings(d)
        self.assertEqual([g["id"] for g in got], ["session-20261106-100000"])
        self.assertEqual(got[0]["speakers"], 2)
        self.assertEqual(got[0]["minutes"], 2)
        self.assertFalse(got[0]["has_notes"])
        self.assertEqual(got[0]["preview"], "山田です。今日は進め方を決めましょう。")

    def test_chunks(self):
        lines = [{"orig": "あ" * 100, "trans": "", "spk": "S1"} for _ in range(30)]
        parts = N.chunks(lines, 1000)
        self.assertGreater(len(parts), 3)
        self.assertEqual(sum(len(p) for p in parts), 30)

    def test_parse_json(self):
        self.assertEqual(N.parse_json('<think>x</think>はい\n```json\n{"a": 1}\n```'), {"a": 1})
        with self.assertRaises(ValueError):
            N.parse_json("no json")


class TestMake(unittest.TestCase):
    def meeting(self, d):
        return N.load_meeting(write_log(d, ROWS))

    def test_single_chunk(self):
        calls = []
        with tempfile.TemporaryDirectory() as d:
            n = N.make_notes(self.meeting(d), "ja",
                             lambda msgs: calls.append(msgs) or json.dumps(GOOD, ensure_ascii=False))
        self.assertEqual(len(calls), 2)                            # 作る + 根拠の行を当てる(W1 の後段)
        self.assertIn("link meeting-note items", calls[1][0]["content"])
        self.assertIn("Japanese", calls[0][0]["content"])         # 出力の言語を指定
        self.assertIn("S2 (en): Hi, I'm Smith", calls[0][1]["content"])
        self.assertEqual([p["name"] for p in n["participants"]], ["山田", "Smith"])
        self.assertEqual(n["todos"][0]["owner"], "S2")
        self.assertFalse(n["todos"][0]["done"])
        self.assertEqual(n["languages"], ["ja", "en"])
        self.assertEqual(n["participants"][0]["lines"], 2)
        self.assertFalse(any(p["minor"] for p in n["participants"]))   # 名前のある人は畳まない

    def test_long_meeting_is_split_then_merged(self):
        calls = []
        with tempfile.TemporaryDirectory() as d:
            m = self.meeting(d)
            m["lines"] = m["lines"] * 40
            N.make_notes(m, "en", lambda msgs: calls.append(msgs) or json.dumps(GOOD), chunk_chars=800)
        self.assertGreater(len(calls), 3)
        # 区切りの後にまとめる。GOOD は日本語なので en のメモではまとめの後に訳し直しが1回入る
        # (_fix_language)。その後は根拠の行を当てる段だけ(W1)
        rest = [c for c in calls if "link meeting-note items" not in c[0]["content"]]
        self.assertIn("Merge them", rest[-2][1]["content"])
        self.assertIn("Translate each item into English", rest[-1][0]["content"])
        k = calls.index(rest[-1])
        self.assertTrue(all("link meeting-note items" in c[0]["content"] for c in calls[k + 1:]))

    def test_known_names_and_done_are_kept(self):
        old = dict(GOOD, todos=[dict(GOOD["todos"][0], done=True)])
        with tempfile.TemporaryDirectory() as d:
            n = N.make_notes(self.meeting(d), "ja", lambda m: json.dumps(GOOD),
                             known={"S1": "山田花子"}, old=old)
        self.assertEqual(n["participants"][0]["name"], "山田花子")
        self.assertTrue(n["todos"][0]["done"])

    def test_broken_json_retried_once(self):
        out = iter(["oops", json.dumps(GOOD)])
        with tempfile.TemporaryDirectory() as d:
            n = N.make_notes(self.meeting(d), "ja", lambda m: next(out))
        self.assertEqual(n["title"], "進め方の打ち合わせ")

    def test_markdown(self):
        with tempfile.TemporaryDirectory() as d:
            m = self.meeting(d)
            n = N.make_notes(m, "ja", lambda x: json.dumps(GOOD))
            md = N.to_markdown(n, m)
        for k in ("# 進め方の打ち合わせ", "## やること", "- [ ] スライドを送る (担当: Smith, 期限: 金曜)",
                  "## 決定事項", "**山田**: 山田です。"):
            self.assertIn(k, md)


class TestRobust(unittest.TestCase):
    """11/06 AMI 30分の会議: 10区切りを一度にまとめると出力が切れて JSON が壊れ、メモ作りが止まった。"""

    def test_many_parts_merge_in_groups_and_never_crash(self):
        calls = []

        def llm(msgs):
            calls.append(msgs)
            if "Merge them" in msgs[1]["content"]:
                return '{"title": "broken'        # まとめは毎回壊れる
            return json.dumps(GOOD)
        with tempfile.TemporaryDirectory() as d:
            m = N.load_meeting(write_log(d, ROWS))
            m["lines"] = m["lines"] * 60
            n = N.make_notes(m, "ja", llm, chunk_chars=500)
        merges = [c for c in calls if "Merge them" in c[1]["content"]]
        self.assertTrue(all(c[1]["content"].count("Part ") <= 4 for c in merges))   # 4つずつ
        self.assertEqual(n["todos"][0]["task"], "スライドを送る")                     # 並べた受け皿で残る

    def test_role_is_not_a_name(self):
        bad = dict(GOOD, participants=[{"id": "S1", "name": "Industrial Designer",
                                        "role": "Industrial Designer", "evidence": "x"}])
        with tempfile.TemporaryDirectory() as d:
            n = N.make_notes(N.load_meeting(write_log(d, ROWS)), "en", lambda m: json.dumps(bad))
        self.assertEqual((n["participants"][0]["name"], n["participants"][0]["role"]),
                         ("", "Industrial Designer"))


class TestIds(unittest.TestCase):
    def test_llm_s_ids_mapped_to_language_speakers(self):
        rows = [dict(r) for r in ROWS]
        for r in rows:
            r.pop("spk")                      # 通訳だけの記録: Lja / Len
        with tempfile.TemporaryDirectory() as d:
            n = N.make_notes(N.load_meeting(write_log(d, rows)), "ja", lambda m: json.dumps(GOOD))
        self.assertEqual([(p["id"], p["name"]) for p in n["participants"]],
                         [("Lja", "山田"), ("Len", "Smith")])
        self.assertEqual(n["todos"][0]["owner"], "Len")
        self.assertEqual(n["topics"][0]["points"][0]["who"], "Len")

    def test_ids_in_text_become_names(self):
        g = dict(GOOD, summary=["Ljaが自己紹介し、S2 が質問した"],
                 participants=[{"id": "S2", "name": "Smith", "evidence": "I'm Smith"}])
        with tempfile.TemporaryDirectory() as d:
            n = N.make_notes(N.load_meeting(write_log(d, ROWS)), "ja", lambda m: json.dumps(g))
        self.assertEqual(n["summary"], ["日本語の話者が自己紹介し、Smith が質問した"])


class TestMinor(unittest.TestCase):
    def test_few_lines_unnamed_is_minor(self):
        rows = [dict(ROWS[0], spk="S1") for _ in range(30)] + [dict(ROWS[1], spk="S7")]
        none = dict(GOOD, participants=[])
        with tempfile.TemporaryDirectory() as d:
            n = N.make_notes(N.load_meeting(write_log(d, rows)), "ja", lambda m: json.dumps(none))
        got = {p["id"]: p["minor"] for p in n["participants"]}
        self.assertEqual(got, {"S1": False, "S7": True})


class TestNameCheck(unittest.TestCase):
    """名前は根拠を機械でも確かめる(11/06 AMI: 名前に触れただけの人をその名前にした)。"""
    L = [{"spk": "S1", "orig": "Me and William worked on a prototype."},
         {"spk": "S2", "orig": "Hi, I'm Emily from Melbourne."},
         {"spk": "S3", "orig": "鈴木さん、実験の進み具合はどうですか。"},
         {"spk": "S4", "orig": "はい、8割終わりました。"},
         {"spk": "S5", "orig": "司会の佐藤です。始めます。"}]

    def test_cases(self):
        v = N._verify_name
        self.assertFalse(v("S1", "William", "Me and William worked", self.L))    # 触れただけ
        self.assertTrue(v("S2", "Emily", "I'm Emily", self.L))                  # 名乗った
        self.assertTrue(v("S4", "鈴木", "鈴木さん、実験の進み具合は", self.L))      # 呼ばれて答えた
        self.assertFalse(v("S3", "鈴木", "鈴木さん、実験の進み具合は", self.L))     # 呼んだ人ではない
        self.assertTrue(v("S5", "佐藤", "司会の佐藤です", self.L))
        self.assertFalse(v("S2", "Sarah", "as Project Supervisor", self.L))     # 根拠に名前が無い
        L2 = [{"spk": "Len", "orig": "My name is Onishi Shinnosuke. I'm from Tokyo."}]
        self.assertTrue(v("Len", "大西慎之介", "My name is Onishi Shinnosuke.", L2))   # 書き方違い
        self.assertFalse(v("S1", "William", "Me and William worked", self.L))
        self.assertFalse(v("Len", "東京太郎", "I'm from Tokyo.", L2))              # 名乗りではない

    def test_transliterated_im_name(self):
        """11/06 擬似会議 jaen: 「I'm Emily from the Boston team.」と名乗った人を、日本語の
        メモで LLM が「エミリー」と書き、記録に「エミリー」が無いので名前が消えた。"""
        v = N._verify_name
        L = [{"spk": "S2", "orig": "I'm Emily from the Boston team."},
             {"spk": "S3", "orig": "This is David, I handle the data."},
             {"spk": "S4", "orig": "I'm going to check the schedule."},
             {"spk": "S5", "orig": "I'm Sorry, could you repeat that?"}]
        self.assertTrue(v("S2", "エミリー", "I'm Emily from the Boston team.", L))
        self.assertTrue(v("S3", "デイビッド", "This is David, I handle the data.", L))
        self.assertFalse(v("S4", "ゴーイング", "I'm going to check the schedule.", L))  # 小文字=名前でない
        self.assertFalse(v("S5", "ソーリー", "I'm Sorry, could you repeat that?", L))   # よくある語
        self.assertFalse(v("S3", "エミリー", "I'm Emily from the Boston team.", L))   # 別の人の名乗り


class TestFiles(unittest.TestCase):
    def test_rename_and_langs(self):
        with tempfile.TemporaryDirectory() as d:
            p = write_log(d, ROWS)
            m = N.load_meeting(p)
            N.save_notes(p, N.make_notes(m, "ja", lambda x: json.dumps(GOOD)))
            N.save_notes(p, N.make_notes(m, "en", lambda x: json.dumps(GOOD)))
            self.assertEqual(N.notes_langs(p), ["ja", "en"])
            N.rename_speaker(p, "S2", "John Smith")
            self.assertEqual(N.load_speakers(p), {"S2": "John Smith"})
            for lg in ("ja", "en"):
                self.assertEqual(N.load_notes(p, lg)["participants"][1]["name"], "John Smith")


class TestBridgeApi(unittest.TestCase):
    def test_meeting_path_stays_in_log_dir(self):
        b = Bridge()
        with tempfile.TemporaryDirectory() as d:
            b.s.log_dir = d
            write_log(d, ROWS)
            self.assertIsNotNone(b._meeting_path("session-20261106-100000"))
            self.assertIsNone(b._meeting_path("../../etc/passwd"))
            self.assertIsNone(b._meeting_path("live"))           # 会話していない

    def test_notes_make_runs_and_reports(self):
        b = Bridge()
        with tempfile.TemporaryDirectory() as d:
            b.s.log_dir = d
            write_log(d, ROWS)
            with patch.object(Bridge, "_notes_llm", return_value=("http://x", "m")), \
                 patch("tsuyaku.notes.chat", return_value=json.dumps(GOOD)):
                self.assertEqual(b.notes_make("session-20261106-100000", "ja"), "started")
                ev = None
                t0 = time.time()
                while time.time() - t0 < 3:
                    e = b.ui.get(timeout=3)
                    if e[0] == "notes" and e[1]["stage"] in ("done", "error"):
                        ev = e[1]
                        break
            self.assertEqual(ev["stage"], "done", ev)
            got = b.meeting_get("session-20261106-100000")
            self.assertEqual(got["notes"]["title"], "進め方の打ち合わせ")
            self.assertEqual(b.todo_toggle("session-20261106-100000", 0, True), "ok")
            self.assertIn("- [x] スライドを送る", b.notes_markdown("session-20261106-100000"))
            self.assertEqual(b.meetings_list()[0]["title"], "進め方の打ち合わせ")
            self.assertEqual(b.meetings_list()[0]["names"], ["Smith", "山田"])
            self.assertEqual(b.meetings_list()[0]["keywords"], ["スライド"])


class TestModes(unittest.TestCase):
    def test_mono_pair_only_for_notes(self):
        self.assertEqual(_norm_pair("ja", "notes"), "ja")
        self.assertEqual(_norm_pair("ja", "interpret"), "ja-en")
        self.assertEqual(_norm_pair("ja-zh", "notes"), "ja-zh")

    def test_start_saves_mode(self):
        b = Bridge()
        with patch("tsuyaku.__main__.save_config"), patch.object(Bridge, "_run") as run:
            b.start("ja", "", "notes")
            time.sleep(0.05)
        run.assert_called_once_with("ja", "", "notes")
        self.assertEqual(b.s.use_mode, "notes")

    def test_notes_mode_records_without_translating(self):
        b = Bridge()
        b.mode = "notes"
        with tempfile.TemporaryDirectory() as d:
            b.log = SessionLog(d, "ja", "", "notes")
            f = Future()
            f.set_result("S1")
            with patch("tsuyaku.__main__.translate_with_retry") as tr:
                b._utterance_body("http://x", "m", ("ja",), "", "では始めます。", "ja", time.time(),
                                  {"spk": [(1.0, f)]})
            tr.assert_not_called()
            b.log.close()
            rows = [json.loads(x) for x in Path(b.log.path_jsonl).read_text().splitlines()]
        self.assertEqual(rows[0]["mode"], "notes")
        self.assertEqual((rows[1]["orig"], rows[1]["trans"], rows[1]["spk"]), ("では始めます。", "", "S1"))
        ev = [b.ui.get() for _ in range(b.ui.qsize())]
        utt = [e for e in ev if e[0] == "utterance"][0][1]
        self.assertEqual((utt["trans"], utt["spk"]), ("", "S1"))

    def test_resolve_spk_votes_by_overlap(self):
        def fut(v):
            f = Future()
            f.set_result(v)
            return f
        self.assertEqual(_resolve_spk([(0.5, fut("S1")), (2.0, fut("S2")), (0.4, fut("S1"))]), "S2")
        self.assertIsNone(_resolve_spk([]))
        self.assertIsNone(_resolve_spk([(1.0, fut(None))]))


if __name__ == "__main__":
    unittest.main()


class TestV2Fixes(unittest.TestCase):
    """11/07 擬似会議 v2(eval/meeting/README.md)で残った notes 側の2件。"""

    def test_name_kept_when_llm_picks_wrong_evidence(self):
        """高橋さん: 根拠に別の文(「では森さんは今日中に…」)を選ばれ、名前の確かめで落ちた。
        記録の中に本人の名乗りがあれば、その行を根拠にして名前を残す。"""
        lines = [{"spk": "S1", "orig": "高橋です。今週の開発の進捗を確認します。", "lang": "ja"},
                 {"spk": "S2", "orig": "渡辺です。サーバーを担当しています。", "lang": "ja"},
                 {"spk": "S1", "orig": "では森さんは今日中にデザイン案をみんなに送ってください", "lang": "ja"}]
        raw = {"participants": [{"id": "S1", "name": "高橋", "evidence": "では森さんは今日中にデザイン案をみんなに送ってください"},
                                {"id": "S2", "name": "森", "evidence": "では森さんは今日中に"}]}
        out = N._clean(raw, lines, {}, None)
        p = {x["id"]: x for x in out["participants"]}
        self.assertEqual(p["S1"]["name"], "高橋")
        self.assertTrue(p["S1"]["evidence"].startswith("高橋です"))
        self.assertEqual(p["S2"]["name"], "")            # 呼ばれた次に話したのは S2 ではない

    def test_todo_not_repeated_as_decision(self):
        """ja_lab: やること2件が決定事項にも重ねて入った。"""
        raw = {"decisions": ["佐藤は金曜日までに追加実験の結果を田中に送付する。",
                             "鈴木がGPUの予約管理を担当し、月曜日までに予約表を作成する。",
                             "全員での発表練習を来週木曜日の午後に行う。"],
               "todos": [{"task": "追加実験の結果を田中に送る", "owner": "S3"},
                         {"task": "GPUの予約表を作る", "owner": "S4"},
                         {"task": "スライドを完成させる", "owner": "S2"}]}
        out = N._clean(raw, [{"spk": "S1", "orig": "はい", "lang": "ja"}], {}, None)
        self.assertEqual(out["decisions"], ["全員での発表練習を来週木曜日の午後に行う。"])
        raw = {"decisions": ["We will use a rubber case.", "Mary will send the specs to the team by Friday."],
               "todos": [{"task": "Send the specs to the team", "owner": "S2"}]}
        out = N._clean(raw, [{"spk": "S1", "orig": "ok", "lang": "en"}], {}, None)
        self.assertEqual(out["decisions"], ["We will use a rubber case."])


class TestV3Fixes(unittest.TestCase):
    """11/07 擬似会議 v3(eval/meeting/README.md)の notes 側。"""

    def test_japanese_todo_rephrased_as_decision(self):
        """ja_agenda: 「追加の予算が出るか経理に確認する」がやることと決定の両方に入った(重なり 0.67)。"""
        todos = [{"task": "追加の予算が出るか経理に確認する"}]
        self.assertTrue(N._repeats_todo("追加の予算の有無について、岡田が明日経理に確認することにした。", todos))
        self.assertFalse(N._repeats_todo("検索の改善を最優先にすることにした。", todos))
        # 英語は今まで通り 0.75(AMI の本物の決定を落とさない)
        # (やることの語の 0.6 が決定に入っている: 日本語の 0.65 なら重複、英語の 0.75 では別物)
        self.assertFalse(N._repeats_todo("The team will develop five to six design concepts to choose from.",
                                         [{"task": "Develop five concepts tomorrow quickly"}]))

    def test_items_in_wrong_language_are_translated(self):
        """この Mac の小さい LLM(gemma-3-4b)が、日本語の会議でもやることを英語で書いた。"""
        notes = {"summary": ["検索の改善を最優先にした。"], "decisions": ["We will test next month."],
                 "questions": [], "topics": [{"title": "Budget", "points": [{"who": "S1", "text": "経理に確認する"}]}],
                 "todos": [{"task": "話者3 will create the test script", "owner": "S3"}]}
        seen = []

        def llm(msgs):
            seen.append(msgs[-1]["content"])
            return json.dumps({"items": ["来月テストする。", "予算", "テストの台本を作る"]}, ensure_ascii=False)
        N._fix_language(notes, "ja", llm)
        self.assertEqual(notes["decisions"], ["来月テストする。"])
        self.assertEqual(notes["topics"][0]["title"], "予算")
        self.assertEqual(notes["todos"][0]["task"], "テストの台本を作る")
        self.assertEqual(notes["summary"], ["検索の改善を最優先にした。"])   # 正しい言語のものは触らない
        self.assertEqual(len(seen), 1)                                     # 1回だけ

    def test_wrong_language_check_leaves_good_notes_alone(self):
        notes = {"summary": ["Q3 の KPI を確認した。"], "decisions": [], "questions": [], "topics": [],
                 "todos": [{"task": "GPU の予約表を作る"}]}
        N._fix_language(notes, "ja", lambda m: self.fail("LLM を呼ばない"))
        notes = {"summary": ["We agreed on the budget."], "decisions": [], "questions": [], "topics": [], "todos": []}
        N._fix_language(notes, "en", lambda m: self.fail("LLM を呼ばない"))
