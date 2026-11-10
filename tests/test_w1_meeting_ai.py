"""W1(11/07): 会議に質問(根拠の発言つき)・全会議に質問・フォローアップの下書き・ライブ要約。

見つけた問題: 会議の中身を確かめるには全文を読むしかなかった(Otter / Fireflies / Notion / Granola は
会議に質問でき、答えに根拠の発言が付く)。メールや議事録はメモから手で書き直していた。
会話中のメモは「メモ」ボタンを押した時だけだった(Zoom / Teams / Otter は数分ごとに自動で更新)。
"""
import json
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from tsuyaku import meeting_ai as A
from tsuyaku import notes as N
from tsuyaku.__main__ import Bridge
from tsuyaku.log import SessionLog


def lines_of(n, text="We talked about the budget for item {k}.", lang="en"):
    return [{"t": f"10:{k // 60:02d}:{k % 60:02d}", "lang": lang, "orig": text.format(k=k),
             "trans": "", "spk": "S1" if k % 2 else "S2", "ts": 1000.0 + k} for k in range(n)]


class TestCites(unittest.TestCase):
    def test_parse_cites(self):
        txt, cites = A.parse_cites("Yes [#3]. Also see [#5, #7] and [#99].", n=10)
        self.assertEqual(cites, [3, 5, 7])
        self.assertEqual(txt, "Yes [#3]. Also see [#5] [#7] and.")

    def test_parse_cites_cross_meeting(self):
        txt, cites = A.parse_cites("A [M1#2] B [M2#4, M2#5] C [M9#1]", n={1: 10, 2: 10})
        self.assertEqual(cites, [(1, 2), (2, 4), (2, 5)])
        self.assertIn("[M2#4] [M2#5]", txt)


class TestAsk(unittest.TestCase):
    def test_short_meeting_one_call_with_numbered_transcript(self):
        calls = []

        def llm(msgs):
            calls.append(msgs)
            return "The budget was discussed [#2]."
        m = {"id": "x", "meta": {}, "lines": lines_of(6)}
        notes = {"summary": ["Budget review"], "decisions": ["Approve"], "todos": [], "lang": "en"}
        r = A.ask(m, notes, "What about the budget?", "en", llm)
        self.assertEqual(len(calls), 1)
        u = calls[0][-1]["content"]
        self.assertIn("#2 [10:00:02] S2", u)
        self.assertIn("What about the budget?", u)
        self.assertIn("Budget review", u)                       # メモも手がかりに
        self.assertEqual(r["cites"], [2])
        self.assertEqual(r["answer"], "The budget was discussed [#2].")
        self.assertEqual(r["lines"]["2"]["orig"], "We talked about the budget for item 2.")

    def test_long_meeting_map_then_answer(self):
        calls = []

        def llm(msgs):
            calls.append(msgs)
            u = msgs[-1]["content"]
            if "relevant" in msgs[0]["content"].lower() and "Excerpt" in u:
                return json.dumps({"facts": [{"text": "budget mentioned", "lines": [450]}]})
            return "It was mentioned [#450]."
        m = {"id": "x", "meta": {}, "lines": lines_of(900)}
        st = []
        r = A.ask(m, None, "budget?", "en", llm, budget=6000, on_stage=lambda s, n: st.append(s))
        self.assertEqual(st, ["search"])                          # 画面に「関係する発言を探しています」
        self.assertGreater(len(calls), 3)
        self.assertIn("#450", calls[-1][-1]["content"])          # 抜き出した行を最後の答えに渡す
        self.assertEqual(r["cites"], [450])

    def test_stream_callback(self):
        got = []
        m = {"id": "x", "meta": {}, "lines": lines_of(4)}
        A.ask(m, None, "q", "ja", lambda msgs: "答え [#1]", on_delta=got.append)
        self.assertEqual("".join(got), "答え [#1]")

    def test_answer_language_follows_question(self):
        calls = []
        m = {"id": "x", "meta": {}, "lines": lines_of(4)}
        A.ask(m, None, "決まったことは?", "ja", lambda msgs: calls.append(msgs) or "x")
        self.assertIn("same language as the question", calls[0][0]["content"])

    def test_suggested_questions(self):
        notes = {"participants": [{"id": "S1", "name": "山田"}, {"id": "S2", "name": ""}]}
        qs = A.suggestions(notes, "ja")
        self.assertIn("決まったことは?", qs)
        self.assertIn("次にやることは?", qs)
        self.assertIn("山田さんの意見は?", qs)
        self.assertIn("What did 山田 think?", A.suggestions(notes, "en"))


class TestAskAll(unittest.TestCase):
    def test_pick_then_answer_with_meeting_cites(self):
        with tempfile.TemporaryDirectory() as d:
            for k, (title, word) in enumerate([("予算会議", "budget"), ("採用面接", "hiring"), ("雑談", "lunch")]):
                p = Path(d) / f"session-2026110{k + 1}-100000.jsonl"
                rows = [{"type": "meta", "pair": "en", "start": f"2026-11-0{k + 1} 10:00:00"}]
                rows += [dict({"type": "utter"}, **ln) for ln in lines_of(5, f"We discussed {word} {{k}}.")]
                p.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
                N.save_notes(p, {"lang": "ja", "title": title, "summary": [f"{word} の話"], "participants": [],
                                 "topics": [], "decisions": [], "todos": [], "questions": [], "keywords": [word]})
            calls = []

            def llm(msgs):
                calls.append(msgs)
                if "Which meetings" in msgs[-1]["content"]:
                    import re
                    k = re.search(r"\[(\d+)\][^\n]*予算会議", msgs[-1]["content"]).group(1)
                    return json.dumps({"meetings": [int(k)]})
                return "予算は承認された [M1#3]"
            r = A.ask_all(d, "budget はどうなった?", "ja", llm)
        self.assertIn("予算会議", calls[0][-1]["content"])
        self.assertEqual([m["title"] for m in r["meetings"]], ["予算会議"])
        self.assertEqual(r["meetings"][0]["id"], "session-20261101-100000")
        self.assertEqual(r["cites"], [[1, 3]])
        self.assertIn("We discussed budget 3.", calls[-1][-1]["content"])
        self.assertEqual(r["lines"]["1#3"]["orig"], "We discussed budget 3.")

    def test_no_meetings(self):
        with tempfile.TemporaryDirectory() as d:
            r = A.ask_all(d, "x", "ja", lambda m: "")
        self.assertEqual(r["meetings"], [])


class TestDrafts(unittest.TestCase):
    NOTES = {"lang": "ja", "title": "進め方", "start": "2026-11-07 10:00:00", "minutes": 30,
             "participants": [{"id": "S1", "name": "山田"}], "summary": ["決めた"], "topics": [],
             "decisions": ["Bにする"], "todos": [{"task": "資料", "owner": "S1", "due": "金曜", "done": False}],
             "questions": [], "keywords": []}

    def test_email_and_minutes(self):
        calls = []
        for kind in ("email", "minutes"):
            out = A.draft(self.NOTES, kind, "ja", lambda msgs: calls.append(msgs) or "件名: 本日の打ち合わせ\n\n本文")
            self.assertTrue(out.startswith("件名"))
        self.assertIn("email", calls[0][0]["content"].lower())
        self.assertIn("minutes", calls[1][0]["content"].lower())
        for c in calls:
            self.assertIn("Japanese", c[0]["content"])
            self.assertIn("Bにする", c[1]["content"])
            self.assertIn("山田", c[1]["content"])        # 担当は名前で

    def test_bad_kind(self):
        with self.assertRaises(ValueError):
            A.draft(self.NOTES, "tweet", "ja", lambda m: "")


class TestLive(unittest.TestCase):
    def test_due(self):
        lv = A.LiveSchedule(every_s=300, every_lines=40, min_lines=8)
        self.assertFalse(lv.due(now=100, n_lines=5))          # 少なすぎ
        self.assertFalse(lv.due(now=100, n_lines=20))         # 5分たっていない・40行も無い
        self.assertTrue(lv.due(now=100, n_lines=45))          # 40行増えた
        lv.mark(now=100, n_lines=45)
        self.assertFalse(lv.due(now=200, n_lines=60))
        self.assertTrue(lv.due(now=401, n_lines=50))          # 5分たち、新しい行がある
        lv.mark(now=401, n_lines=50)
        self.assertFalse(lv.due(now=900, n_lines=50))         # 新しい行が無ければ作らない

    def test_cached_llm_reuses_finished_chunks(self):
        calls = []
        llm = A.CachedLLM(lambda msgs: calls.append(1) or "{}")
        m1 = [{"role": "user", "content": "a"}]
        llm(m1)
        llm(m1)
        llm([{"role": "user", "content": "b"}])
        self.assertEqual(len(calls), 2)

    def test_live_key_ignores_speaker_relabel(self):
        """会話中の話者の付け直しで前の区切りの S3 → S1 になっても聞き直さない。"""
        calls = []
        llm = A.CachedLLM(lambda msgs: calls.append(1) or "{}", key=A.live_key)
        a = [{"role": "user", "content": "Speakers: S1 = x; S3 = y\n\nTask\n\nTranscript:\n#0 [10:00:00] S3 (ja): はい"}]
        b = [{"role": "user", "content": "Speakers: S1 = x\n\nTask\n\nTranscript:\n#0 [10:00:00] S1 (ja): はい"}]
        llm(a)
        llm(b)
        self.assertEqual(len(calls), 1)
        llm([{"role": "user", "content": "Speakers: S1 = x\n\nTask\n\nTranscript:\n#0 [10:00:00] S1 (ja): いいえ"}])
        self.assertEqual(len(calls), 2)


class TestBridge(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.TemporaryDirectory()
        self.b = Bridge()
        self.b.s.log_dir = self.d.name
        p = Path(self.d.name) / "session-20261107-100000.jsonl"
        rows = [{"type": "meta", "pair": "en", "start": "2026-11-07 10:00:00"}]
        rows += [dict({"type": "utter"}, **ln) for ln in lines_of(5)]
        p.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
        self.p = p

    def tearDown(self):
        self.d.cleanup()

    def wait(self, kind, pred, timeout=3):
        t0 = time.time()
        while time.time() - t0 < timeout:
            e = self.b.ui.get(timeout=timeout)
            if e[0] == kind and pred(e[1]):
                return e[1]
        return None

    def test_meeting_ask_streams_and_finishes(self):
        def fake_stream(base, model, msgs, on_delta=None, **kw):
            for x in ("The budget ", "[#2]."):
                on_delta(x)
            return "The budget [#2]."
        with patch.object(Bridge, "_notes_llm", return_value=("http://x", "m")), \
             patch("tsuyaku.meeting_ai.chat_stream", side_effect=fake_stream):
            qid = self.b.meeting_ask("session-20261107-100000", "budget?", "en")
            self.assertTrue(qid)
            ev = self.wait("ask", lambda d: d.get("done"))
        self.assertEqual(ev["qid"], qid)
        self.assertEqual(ev["cites"], [2])
        self.assertEqual(ev["answer"], "The budget [#2].")

    def test_meeting_ask_error_is_reported(self):
        with patch.object(Bridge, "_notes_llm", side_effect=RuntimeError("no llm")):
            qid = self.b.meeting_ask("session-20261107-100000", "budget?", "en")
            ev = self.wait("ask", lambda d: d.get("done"))
        self.assertEqual(ev["qid"], qid)
        self.assertIn("no llm", ev["error"])

    def test_notes_make_with_template_saves_it(self):
        good = {"title": "t", "participants": [], "summary": ["s [#1]"], "topics": [], "decisions": [],
                "todos": [], "questions": [], "keywords": [], "sections": {"concepts": ["c [#0]"]}}
        with patch.object(Bridge, "_notes_llm", return_value=("http://x", "m")), \
             patch("tsuyaku.notes.chat", return_value=json.dumps(good)):
            self.assertEqual(self.b.notes_make("session-20261107-100000", "en", "lecture"), "started")
            ev = self.wait("notes", lambda d: d["stage"] in ("done", "error"))
        self.assertEqual(ev["stage"], "done", ev)
        self.assertEqual(ev["notes"]["template"], "lecture")
        self.assertEqual(N.load_notes(self.p, "en")["sections"][0]["key"], "concepts")

    def test_notes_draft_saved_in_notes(self):
        N.save_notes(self.p, dict(TestDrafts.NOTES, lang="en"))
        with patch.object(Bridge, "_notes_llm", return_value=("http://x", "m")), \
             patch("tsuyaku.notes.chat", return_value="Subject: Thanks\n\nHi all"):
            out = self.b.notes_draft("session-20261107-100000", "email", "en")
        self.assertEqual(out["text"], "Subject: Thanks\n\nHi all")
        self.assertEqual(N.load_notes(self.p, "en")["drafts"]["email"], "Subject: Thanks\n\nHi all")
        self.assertEqual(self.b.notes_draft("session-20261107-100000", "email", "en", False)["text"],
                         "Subject: Thanks\n\nHi all")            # 作り直さなければ保存済みを返す

    def test_set_template_goes_into_log_meta(self):
        with patch("tsuyaku.__main__.save_config"):
            self.assertEqual(self.b.set_template("sales"), "ok")
            self.assertEqual(self.b.get_template(), "sales")
            self.b.set_template("nope")
            self.assertEqual(self.b.get_template(), "free")
            self.b.set_template("1on1")
        log = SessionLog(self.d.name, "ja", "", "notes", extra=self.b._meta_extra())
        log.close()
        meta = json.loads(Path(log.path_jsonl).read_text(encoding="utf-8").splitlines()[0])
        self.assertEqual(meta["template"], "1on1")

    def test_live_refresh_only_when_idle(self):
        b = self.b
        b.mode = "both"
        b.running.set()
        b._ui_status = ("work", time.time())
        self.assertFalse(b._live_idle())              # 訳している最中は待つ
        b._ui_status = ("listen", time.time() - 5)
        b._tw_busy = False
        self.assertTrue(b._live_idle())
        b.__dict__.setdefault("_notes_busy", set()).add(("x", "ja"))
        self.assertFalse(b._live_idle())              # ほかのメモ作りと重ねない
        b.running.clear()


if __name__ == "__main__":
    unittest.main()


class TestLiveOnce(unittest.TestCase):
    """会話中のライブ要約: 記録を読んでメモを作り、保存して画面へ。途中で「メモ」が押されたら上書きしない。"""

    def run_once(self, stamp_after=False):
        good = {"title": "live", "participants": [], "summary": ["s [#0]"], "topics": [], "decisions": [],
                "todos": [], "questions": [], "keywords": []}
        b = Bridge()
        d = tempfile.mkdtemp()
        b.s.log_dir = d
        b.log = SessionLog(d, "en", "", "notes", extra={"template": "free"})
        for k in range(10):
            b.log.add("en", f"line {k} about the budget.", "", None, spk="S1")
        b.running.set()
        b._live_lang = "en"
        b._live_cache = None

        def chat(base, model, msgs, **kw):
            if stamp_after:
                b._notes_stamp = time.time() + 5
            return json.dumps(good)
        with patch.object(Bridge, "_notes_llm", return_value=("http://x", "m")), \
             patch("tsuyaku.notes.chat", side_effect=chat):
            b._live_once(b.log)
        b.running.clear()
        evs = []
        while not b.ui.empty():
            evs.append(b.ui.get())
        return b, evs

    def test_saves_and_pushes(self):
        b, evs = self.run_once()
        live = [e[1] for e in evs if e[0] == "notes_live"]
        self.assertEqual(len(live), 1)
        self.assertTrue(live[0]["notes"]["live"])
        self.assertEqual(live[0]["notes"]["summary_src"], [[0]])
        self.assertEqual(N.load_notes(b.log.path_jsonl, "en")["title"], "live")

    def test_manual_make_wins(self):
        b, evs = self.run_once(stamp_after=True)
        self.assertFalse([e for e in evs if e[0] == "notes_live"])
        self.assertIsNone(N.load_notes(b.log.path_jsonl, "en"))
