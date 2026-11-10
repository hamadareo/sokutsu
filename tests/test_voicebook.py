"""W4 声を覚える(11/07): 名前を付けた話者の声(CAM++ の重心)を覚え、次の会議で同じ声に名前を付ける。"""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from tsuyaku import voicebook as V
from tsuyaku.speakers import OnlineClusters, refine_labels  # noqa: F401  (重心の作り方をそろえる)

RNG = np.random.default_rng(0)
DIM = 192


def unit(v):
    v = np.asarray(v, np.float32)
    return v / np.linalg.norm(v)


PEOPLE = {k: unit(RNG.normal(size=DIM)) for k in ("yamada", "smith", "tanaka", "lee")}


def voice_of(who, noise=0.35):
    """同じ人の別の会議の声: 本人の向き + 雑音(類似 ≈ 0.9)。"""
    return unit(PEOPLE[who] + noise * unit(RNG.normal(size=DIM)))


class TestBook(unittest.TestCase):
    def setUp(self):
        self.td = tempfile.TemporaryDirectory()
        self.book = V.VoiceBook(Path(self.td.name) / "voices")

    def tearDown(self):
        self.td.cleanup()

    def test_enroll_persist_and_match_same_person(self):
        self.book.enroll("山田", voice_of("yamada"), 40.0, 12, "session-a:S1")
        self.book.enroll("Smith", voice_of("smith"), 30.0, 9, "session-a:S2")
        book2 = V.VoiceBook(Path(self.td.name) / "voices")       # 保存して読み直す
        self.assertEqual(sorted(p["name"] for p in book2.persons()), ["Smith", "山田"])
        clusters = {"S1": (voice_of("smith"), 25.0, 8), "S2": (voice_of("yamada"), 50.0, 15),
                    "S3": (voice_of("tanaka"), 60.0, 20)}
        m = book2.match(clusters, exclude_mid="session-b")
        self.assertEqual(m["S1"]["name"], "Smith")
        self.assertEqual(m["S2"]["name"], "山田")
        self.assertNotIn("S3", m)                                # 覚えていない人には付けない
        self.assertGreater(m["S1"]["sim"], V.THRESHOLD)

    def test_short_clusters_are_not_matched_or_enrolled(self):
        self.assertIsNone(self.book.enroll("山田", voice_of("yamada"), V.MIN_ENROLL_S - 1, 3, "session-a:S1"))
        self.assertEqual(self.book.persons(), [])
        self.book.enroll("山田", voice_of("yamada"), 40.0, 12, "session-a:S1")
        m = self.book.match({"S1": (voice_of("yamada"), V.MIN_MATCH_S - 1, 3)}, exclude_mid="session-b")
        self.assertEqual(m, {})

    def test_one_name_per_meeting_and_needs_margin(self):
        h = voice_of("yamada")
        self.book.enroll("山田", h, 40.0, 12, "session-a:S1")
        clusters = {"S1": (voice_of("yamada", .2), 40.0, 12), "S4": (voice_of("yamada", .5), 30.0, 10)}
        m = self.book.match(clusters, exclude_mid="session-b")
        self.assertEqual(list(m), ["S1"])                         # 同じ名前を2人に付けない(近い方だけ)
        # 2人の覚えた声がどちらも同じくらい近い(見本がほぼ同じ)= 決められない → 付けない
        twin = unit(h + 0.01 * unit(RNG.normal(size=DIM)))
        self.book.enroll("双子", twin, 40.0, 12, "session-c:S1")
        m = self.book.match({"S1": (voice_of("yamada"), 40.0, 12)}, exclude_mid="session-b")
        self.assertEqual(m, {})

    def test_same_meeting_samples_are_ignored(self):
        self.book.enroll("山田", voice_of("yamada"), 40.0, 12, "session-a:S1")
        self.assertEqual(self.book.match({"S2": (voice_of("yamada"), 40.0, 12)}, exclude_mid="session-a"), {})

    def test_rename_moves_the_sample_and_clear_removes_it(self):
        self.book.enroll("山田", voice_of("yamada"), 40.0, 12, "session-a:S1")
        self.book.enroll("山田", voice_of("yamada"), 30.0, 9, "session-b:S2")
        self.assertEqual(self.book.persons()[0]["samples"], 2)
        self.book.enroll("Smith", voice_of("yamada"), 40.0, 12, "session-a:S1")   # 名前を直した
        ps = {p["name"]: p for p in self.book.persons()}
        self.assertEqual(ps["山田"]["samples"], 1)
        self.assertEqual(ps["Smith"]["samples"], 1)
        self.book.unenroll("session-a:S1")
        self.assertNotIn("Smith", [p["name"] for p in self.book.persons()])     # 空になった人は消える

    def test_forget_and_rename_person(self):
        p = self.book.enroll("山田", voice_of("yamada"), 40.0, 12, "session-a:S1")
        self.book.rename(p["id"], "山田")
        self.assertEqual(self.book.persons()[0]["name"], "山田")
        self.book.forget(p["id"])
        self.assertEqual(self.book.persons(), [])
        self.assertEqual(self.book.match({"S1": (voice_of("yamada"), 40.0, 12)}, exclude_mid="x"), {})

    def test_samples_are_capped(self):
        for i in range(V.MAX_SAMPLES + 3):
            self.book.enroll("山田", voice_of("yamada"), 20.0 + i, 8, f"session-{i}:S1")
        self.assertEqual(self.book.persons()[0]["samples"], V.MAX_SAMPLES)

    def test_forget_all_and_first_time_notice(self):
        self.assertFalse(self.book.noticed())
        self.book.enroll("山田", voice_of("yamada"), 40.0, 12, "session-a:S1")
        self.book.set_noticed()
        self.assertTrue(V.VoiceBook(self.book.root).noticed())          # 保存して読み直しても残る
        self.book.enroll("Smith", voice_of("smith"), 40.0, 12, "session-a:S2")
        self.assertTrue(self.book.noticed())                            # 覚え足しても消えない
        self.assertEqual(self.book.forget_all(), 2)
        self.assertEqual(self.book.persons(), [])
        self.assertFalse(self.book.noticed())                           # 次に初めて覚える時また知らせる
        self.assertEqual(list(self.book.root.glob("book*")), [])        # 声の特徴のファイルも残さない


class TestMeetingVoices(unittest.TestCase):
    def test_save_and_load_cluster_centroids(self):
        with tempfile.TemporaryDirectory() as d:
            jl = Path(d) / "session-20261107-100000.jsonl"
            jl.write_text("", encoding="utf-8")
            V.save_meeting_voices(jl, {"S1": (voice_of("yamada"), 40.0, 12), "S2": (voice_of("smith"), 9.5, 3)})
            got = V.load_meeting_voices(jl)
            self.assertEqual(sorted(got), ["S1", "S2"])
            self.assertAlmostEqual(got["S2"][1], 9.5)
            self.assertEqual(got["S1"][2], 12)
            self.assertAlmostEqual(float(np.linalg.norm(got["S1"][0])), 1.0, places=4)
            self.assertTrue(Path(d, "session-20261107-100000.voices.npz").exists())

    def test_matches_sidecar(self):
        with tempfile.TemporaryDirectory() as d:
            jl = Path(d) / "session-20261107-100000.jsonl"
            V.save_matches(jl, {"S1": {"name": "山田", "sim": 0.81, "state": "auto"}})
            self.assertEqual(V.load_matches(jl)["S1"]["state"], "auto")
            self.assertEqual(json.loads(Path(d, "session-20261107-100000.voicematch.json").read_text())["S1"]["name"], "山田")


class TestCentroids(unittest.TestCase):
    def test_centroids_weight_by_duration_and_follow_labels(self):
        embs = [voice_of("yamada"), voice_of("yamada"), voice_of("smith"), None, voice_of("smith")]
        durs = [3.0, 10.0, 2.0, 5.0, 0.3]
        labels = ["S1", "S1", "S2", "S2", "S2"]
        c = V.centroids(embs, durs, labels, min_s=0.6, max_weight_s=6.0)
        self.assertEqual(sorted(c), ["S1", "S2"])
        self.assertAlmostEqual(c["S1"][1], 13.0)                   # 話した秒(特徴のある区間)
        self.assertEqual(c["S1"][2], 2)
        self.assertEqual(c["S2"][2], 1)                            # 特徴なし・短すぎる区間は数えない
        self.assertGreater(float(c["S1"][0] @ PEOPLE["yamada"]), 0.85)


if __name__ == "__main__":
    unittest.main()
