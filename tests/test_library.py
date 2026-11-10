"""W4 整理(11/07): 題の変更・タグ・フォルダ・お気に入り・ゴミ箱(戻せる)・一覧の付加情報。"""
import json
import tempfile
import time
import unittest
from pathlib import Path

from tsuyaku import library as LB
from tsuyaku import notes as N


def write_log(d, stem="session-20261106-100000", rows=None, start="2026-11-06 10:00:00"):
    p = Path(d) / f"{stem}.jsonl"
    rows = rows if rows is not None else [
        {"t": "10:00:05", "lang": "ja", "orig": "予算の話をしましょう。", "trans": "", "spk": "S1", "ts": 1.0},
        {"t": "10:00:12", "lang": "ja", "orig": "来週までに見積もりを出します。", "trans": "", "spk": "S2", "ts": 2.0},
    ]
    with p.open("w", encoding="utf-8") as f:
        f.write(json.dumps({"type": "meta", "pair": "ja-ja", "mode": "notes", "start": start},
                           ensure_ascii=False) + "\n")
        for r in rows:
            f.write(json.dumps(dict({"type": "utter"}, **r), ensure_ascii=False) + "\n")
    Path(d, f"{stem}.txt").write_text("x", encoding="utf-8")
    return p


class TestSidecar(unittest.TestCase):
    def test_title_tags_folder_star_roundtrip(self):
        with tempfile.TemporaryDirectory() as d:
            p = write_log(d)
            self.assertEqual(LB.load_info(p), {})
            LB.update_info(p, title="  定例  ", tags=["予算", " 予算 ", "", "A社"], folder="研究室", star=True)
            info = LB.load_info(p)
            self.assertEqual(info["title"], "定例")
            self.assertEqual(info["tags"], ["予算", "A社"])        # 空・重複・前後の空白を落とす
            self.assertEqual(info["folder"], "研究室")
            self.assertTrue(info["star"])
            self.assertTrue(Path(d, "session-20261106-100000.tags.json").exists())
            LB.update_info(p, title="")                              # 空 = 自動の題に戻す
            self.assertNotIn("title", LB.load_info(p))
            self.assertEqual(LB.load_info(p)["tags"], ["予算", "A社"])  # ほかは残る

    def test_unknown_fields_are_ignored_and_long_values_cut(self):
        with tempfile.TemporaryDirectory() as d:
            p = write_log(d)
            LB.update_info(p, evil="x", title="あ" * 500, tags=["t" * 200] * 3)
            info = LB.load_info(p)
            self.assertNotIn("evil", info)
            self.assertLessEqual(len(info["title"]), LB.MAX_TITLE)
            self.assertEqual(len(info["tags"]), 1)
            self.assertLessEqual(len(info["tags"][0]), LB.MAX_TAG)

    def test_broken_sidecar_reads_as_empty(self):
        with tempfile.TemporaryDirectory() as d:
            p = write_log(d)
            Path(d, "session-20261106-100000.tags.json").write_text("{oops", encoding="utf-8")
            self.assertEqual(LB.load_info(p), {})


class TestList(unittest.TestCase):
    def test_list_merges_sidecar_and_title_override(self):
        with tempfile.TemporaryDirectory() as d:
            p = write_log(d)
            write_log(d, stem="session-20261107-090000", start="2026-11-07 09:00:00")
            N.save_notes(p, {"lang": "ja", "title": "自動の題", "participants": [], "keywords": ["予算"]})
            LB.update_info(p, title="手で付けた題", tags=["予算"], folder="研究室", star=True)
            lst = LB.list_meetings(d)
            self.assertEqual([m["id"] for m in lst], ["session-20261107-090000", "session-20261106-100000"])
            m = lst[1]
            self.assertEqual(m["title"], "手で付けた題")
            self.assertEqual(m["auto_title"], "自動の題")
            self.assertEqual(m["tags"], ["予算"])
            self.assertEqual(m["folder"], "研究室")
            self.assertTrue(m["star"])
            self.assertFalse(lst[0]["star"])
            self.assertEqual(lst[0]["tags"], [])

    def test_list_is_cached_until_files_change(self):
        with tempfile.TemporaryDirectory() as d:
            p = write_log(d)
            LB.list_meetings(d)
            calls = []
            orig = N.meeting_entry

            def spy(path):
                calls.append(path)
                return orig(path)
            N.meeting_entry, saved = spy, N.meeting_entry
            try:
                LB.list_meetings(d)
                self.assertEqual(calls, [])                 # 何も変わっていない: 読み直さない
                time.sleep(0.01)
                with p.open("a", encoding="utf-8") as f:
                    f.write(json.dumps({"type": "utter", "t": "10:01:00", "lang": "ja", "orig": "追加",
                                        "spk": "S1"}, ensure_ascii=False) + "\n")
                lst = LB.list_meetings(d)
                self.assertEqual(len(calls), 1)
                self.assertEqual(lst[0]["lines"], 3)
            finally:
                N.meeting_entry = saved

    def test_facets_count_tags_and_folders(self):
        with tempfile.TemporaryDirectory() as d:
            a = write_log(d)
            b = write_log(d, stem="session-20261107-090000", start="2026-11-07 09:00:00")
            LB.update_info(a, tags=["予算", "A社"], folder="研究室", star=True)
            LB.update_info(b, tags=["予算"], folder="")
            LB.folder_add(d, "空のフォルダ")
            f = LB.facets(d)
            self.assertEqual(f["tags"], [{"name": "予算", "n": 2}, {"name": "A社", "n": 1}])
            self.assertEqual(f["folders"], [{"name": "研究室", "n": 1}, {"name": "空のフォルダ", "n": 0}])
            self.assertEqual(f["star"], 1)
            self.assertEqual(f["all"], 2)
            self.assertEqual(f["trash"], 0)

    def test_kinds_split_interpret_and_notes(self):
        def mk(d, stem, mode):
            p = write_log(d, stem=stem)
            rows = p.read_text(encoding="utf-8").splitlines()
            meta = json.loads(rows[0])
            if mode is None:
                meta.pop("mode")
            else:
                meta["mode"] = mode
            p.write_text("\n".join([json.dumps(meta, ensure_ascii=False)] + rows[1:]) + "\n", encoding="utf-8")
        with tempfile.TemporaryDirectory() as d:
            mk(d, "session-20261101-000001", "interpret")
            mk(d, "session-20261101-000002", "notes")
            mk(d, "session-20261101-000003", "both")
            mk(d, "session-20261101-000004", None)         # 古い記録(mode 無し)は通訳
            by = {m["id"][-1]: m for m in LB.list_meetings(d)}
            self.assertEqual(by["1"]["kinds"], ["interpret"])
            self.assertEqual(by["2"]["kinds"], ["notes"])
            self.assertEqual(by["3"]["kinds"], ["interpret", "notes"])
            self.assertEqual(by["4"]["kinds"], ["interpret"])
            self.assertEqual(LB.facets(d)["kinds"], {"all": 4, "interpret": 3, "notes": 2})

    def test_folder_rename_and_remove_update_meetings(self):
        with tempfile.TemporaryDirectory() as d:
            a = write_log(d)
            LB.update_info(a, folder="研究室")
            LB.folder_rename(d, "研究室", "ラボ")
            self.assertEqual(LB.load_info(a)["folder"], "ラボ")
            self.assertIn("ラボ", [x["name"] for x in LB.facets(d)["folders"]])
            LB.folder_remove(d, "ラボ")                    # フォルダを消しても会議は消えない
            self.assertNotIn("folder", LB.load_info(a))
            self.assertTrue(a.exists())
            self.assertEqual(LB.facets(d)["folders"], [])

    def test_tag_rename_merges(self):
        with tempfile.TemporaryDirectory() as d:
            a = write_log(d)
            LB.update_info(a, tags=["予算", "budget"])
            LB.tag_rename(d, "budget", "予算")
            self.assertEqual(LB.load_info(a)["tags"], ["予算"])


class TestTrash(unittest.TestCase):
    def test_trash_moves_every_file_of_the_meeting_and_restores(self):
        with tempfile.TemporaryDirectory() as d:
            p = write_log(d)
            other = write_log(d, stem="session-20261106-1000001")  # 名前の頭が同じ別の記録は動かさない
            N.save_notes(p, {"lang": "ja", "title": "t"})
            Path(d, "session-20261106-100000.speakers.json").write_text("{}", encoding="utf-8")
            Path(d, "session-20261106-100000.audio.m4a").write_bytes(b"\0")
            self.assertTrue(LB.trash(d, "session-20261106-100000"))
            self.assertFalse(p.exists())
            self.assertTrue(other.exists())
            names = sorted(x.name for x in Path(d, ".trash").iterdir())
            self.assertIn("session-20261106-100000.jsonl", names)
            self.assertIn("session-20261106-100000.audio.m4a", names)
            self.assertNotIn("session-20261106-1000001.jsonl", names)
            self.assertEqual([m["id"] for m in LB.list_meetings(d)], ["session-20261106-1000001"])
            tr = LB.trash_list(d)
            self.assertEqual([m["id"] for m in tr], ["session-20261106-100000"])
            self.assertEqual(tr[0]["title"], "t")
            self.assertTrue(tr[0]["trashed"])
            self.assertTrue(LB.restore(d, "session-20261106-100000"))
            self.assertTrue(p.exists())
            self.assertEqual(LB.trash_list(d), [])
            self.assertNotIn("trashed", LB.load_info(p))

    def test_trash_refuses_bad_ids_and_live_meeting(self):
        with tempfile.TemporaryDirectory() as d:
            p = write_log(d)
            self.assertFalse(LB.trash(d, "../etc/passwd"))
            self.assertFalse(LB.trash(d, "session-nope"))
            self.assertFalse(LB.trash(d, p.stem, live=p))     # 今の会話は捨てない
            self.assertTrue(p.exists())

    def test_restore_does_not_overwrite(self):
        with tempfile.TemporaryDirectory() as d:
            p = write_log(d)
            LB.trash(d, p.stem)
            write_log(d)                                      # 同じ名前の記録がまたできた
            self.assertFalse(LB.restore(d, p.stem))
            self.assertEqual(len(LB.trash_list(d)), 1)

    def test_empty_trash_needs_the_count(self):
        with tempfile.TemporaryDirectory() as d:
            for i in range(3):
                LB.trash(d, write_log(d, stem=f"session-20261106-10000{i}").stem)
            self.assertEqual(LB.trash_empty(d, expect=2), -1)  # 画面で見た数と違う = 消さない
            self.assertEqual(len(LB.trash_list(d)), 3)
            self.assertEqual(LB.trash_empty(d, expect=3), 3)
            self.assertEqual(LB.trash_list(d), [])
            self.assertEqual(LB.facets(d)["trash"], 0)


class TestBulk(unittest.TestCase):
    def mk(self, d, n=4):
        out = []
        for i in range(n):
            p = write_log(d, stem=f"session-20261106-10000{i}")
            for suf in (".audio.m4a", ".notes.json", ".speakers.json", ".voices.npz", ".voicematch.json"):
                Path(d, p.stem + suf).write_bytes(b"x")
            LB.update_info(p, tags=["t"])
            out.append(p)
        return out

    def test_trash_many_skips_protected_and_unknown(self):
        with tempfile.TemporaryDirectory() as d:
            ps = self.mk(d)
            r = LB.trash_many(d, [p.stem for p in ps[:3]] + ["../x", "session-nope"], protect={ps[1].stem})
            self.assertEqual(r["moved"], [ps[0].stem, ps[2].stem])
            self.assertEqual(sorted(r["skipped"]), sorted([ps[1].stem, "../x", "session-nope"]))
            self.assertTrue(ps[1].exists())
            self.assertEqual(sorted(m["id"] for m in LB.trash_list(d)), [ps[0].stem, ps[2].stem])
            r = LB.restore_many(d, [ps[0].stem, ps[2].stem])
            self.assertEqual(sorted(r["restored"]), [ps[0].stem, ps[2].stem])
            self.assertTrue(Path(d, ps[0].stem + ".audio.m4a").exists())

    def test_delete_forever_removes_every_attached_file_and_needs_the_count(self):
        with tempfile.TemporaryDirectory() as d:
            ps = self.mk(d)
            LB.trash_many(d, [p.stem for p in ps])
            self.assertEqual(LB.delete_forever(d, [ps[0].stem, ps[1].stem], expect=3), -1)   # 数が違う = 消さない
            self.assertEqual(len(LB.trash_list(d)), 4)
            self.assertEqual(LB.delete_forever(d, [ps[0].stem, ps[1].stem, ps[0].stem], expect=2), 2)
            left = sorted(x.name for x in Path(d, ".trash").iterdir())
            self.assertFalse([x for x in left if x.startswith((ps[0].stem + ".", ps[1].stem + "."))])
            self.assertEqual(sorted(m["id"] for m in LB.trash_list(d)), [ps[2].stem, ps[3].stem])
            # ゴミ箱にない(一覧にある)会議は完全に消さない
            q = write_log(d, stem="session-20261107-000000")
            self.assertEqual(LB.delete_forever(d, [q.stem], expect=1), -1)
            self.assertTrue(q.exists())


if __name__ == "__main__":
    unittest.main()
