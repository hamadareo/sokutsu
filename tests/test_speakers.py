"""話者の見分け(speakers.py)のテスト。まとめ分けの規則はモデル無しで、モデルを使う部分は
models/ にモデルがあるときだけ確かめる。

実行: PYTHONPATH=src .venv/bin/python -m unittest tests.test_speakers -v
"""
from __future__ import annotations

import os
import tempfile
import threading
import unittest
from pathlib import Path

import numpy as np

from tsuyaku.speakers import MODEL_NAMES, OnlineClusters, SpeakerTracker, find_model, refine_labels

ROOT = Path(__file__).resolve().parents[1]


def _vec(i: int, dim: int = 32, noise: float = 0.0, seed: int = 0) -> np.ndarray:
    """話者 i の声の特徴の代わり(軸 i 方向 + 少しの揺れ)。"""
    rng = np.random.default_rng(seed)
    v = np.zeros(dim, np.float32)
    v[i] = 1.0
    v += noise * rng.standard_normal(dim).astype(np.float32)
    return v / np.linalg.norm(v)


class TestOnlineClusters(unittest.TestCase):
    def test_same_voice_same_id_new_voice_new_id(self):
        cl = OnlineClusters(threshold=0.5)
        assert cl.assign(_vec(0), 2.0) == "S1"
        assert cl.assign(_vec(0, noise=0.1, seed=1), 2.0) == "S1"
        assert cl.assign(_vec(1), 2.0) == "S2"
        assert cl.assign(_vec(0, noise=0.1, seed=2), 1.0) == "S1"
        assert cl.assign(_vec(1, noise=0.1, seed=3), 3.0) == "S2"
        assert len(cl) == 2


    def test_too_short_is_none(self):
        cl = OnlineClusters(min_s=0.6)
        assert cl.assign(_vec(0), 0.5) is None
        assert len(cl) == 0


    def test_short_unknown_voice_does_not_make_a_speaker(self):
        # 短い区間(new_min_s 未満)からは新しい話者を作らない(幽霊の話者を防ぐ)
        cl = OnlineClusters(new_min_s=1.2)
        assert cl.assign(_vec(0), 0.8) is None
        assert cl.assign(_vec(0), 2.0) == "S1"
        assert cl.assign(_vec(0), 0.8) == "S1"          # 知っている声なら短くても付ける
        assert cl.assign(_vec(1), 0.8) is None          # 知らない短い声は付けない
        assert len(cl) == 1


    def test_max_speakers_forces_nearest(self):
        cl = OnlineClusters(threshold=0.9, max_speakers=2)
        assert cl.assign(_vec(0), 2.0) == "S1"
        assert cl.assign(_vec(1), 2.0) == "S2"
        # 3人目の声: 新しく作れないので一番近い人に付ける(長い区間のとき)
        v = _vec(2) * 0.6 + _vec(1) * 0.8
        assert cl.assign(v, 2.0) == "S2"
        assert cl.assign(_vec(3), 0.8) is None           # 短いものは無理に付けない
        assert len(cl) == 2


    def test_centroid_follows_the_speaker(self):
        # 声が少しずつ変わっても、重心が付いていけば同じ人のまま
        cl = OnlineClusters(threshold=0.7)
        v = _vec(0)
        ids = set()
        for k in range(10):
            w = _vec(0) * np.cos(0.06 * k) + _vec(1) * np.sin(0.06 * k)
            ids.add(cl.assign(w, 3.0))
        assert ids == {"S1"}
        assert np.dot(v, v) > 0


    def test_reset(self):
        cl = OnlineClusters()
        cl.assign(_vec(0), 2.0)
        cl.reset()
        assert len(cl) == 0
        assert cl.assign(_vec(5), 2.0) == "S1"


    def test_confirm_needs_two_similar_segments(self):
        cl = OnlineClusters(confirm=2)
        assert cl.assign(_vec(0), 2.0) is None            # 候補
        assert cl.assign(_vec(0, noise=0.05, seed=4), 2.0) == "S1"   # 2つそろって話者に
        assert cl.assign(_vec(0), 1.0) == "S1"

    def test_update_th_keeps_centroid_clean(self):
        cl = OnlineClusters(threshold=0.3, update_th=0.9)
        assert cl.assign(_vec(0), 2.0) == "S1"
        mixed = (_vec(0) + _vec(1)) / np.sqrt(2)          # 類似 0.71: 付けるが重心は育てない
        assert cl.assign(mixed, 6.0) == "S1"
        c = cl._sums[0] / np.linalg.norm(cl._sums[0])
        assert float(np.dot(c, _vec(0))) > 0.999

    def test_merge_two_ids_of_one_person(self):
        cl = OnlineClusters(threshold=0.9, merge_th=0.8)
        assert cl.assign(_vec(0), 2.0) == "S1"
        v = _vec(0) * 0.85 + _vec(1) * 0.53               # 類似 0.85 < 0.9: 別人として作る
        assert cl.assign(v, 2.0) == "S2"
        # S2 に S1 寄りの声が足されて重心が近づくと、S1 に合わせる
        for k in range(3):
            assert cl.assign(_vec(0) * 0.95 + _vec(1) * 0.31, 6.0) in ("S1", "S2")
        assert cl.aliases == {"S2": "S1"}
        assert cl.canonical("S2") == "S1"
        assert len(cl) == 1
        assert cl.assign(v, 2.0) == "S1"
        assert cl.assign(_vec(5), 2.0) == "S3"            # 番号は使い回さない

    def test_refine_drops_tiny_speaker_and_fills_none(self):
        # 会議の後: 少ししか話していない「幽霊の話者」S3 は消え、近い S1 に付く。None の区間にも付く
        a, b = _vec(0), _vec(1)
        ghost = _vec(0) * 0.8 + _vec(2) * 0.6
        embs = [a, b, a, b, ghost, a, b, None, a * 0.9 + _vec(3) * 0.44]
        durs = [3.0, 3.0, 3.0, 3.0, 0.7, 3.0, 3.0, 0.4, 0.8]
        labs = ["S1", "S2", "S1", "S2", "S3", "S1", "S2", None, None]
        out = refine_labels([e / np.linalg.norm(e) if e is not None else None for e in embs],
                            durs, labs, min_frac=0.05)
        assert out == ["S1", "S2", "S1", "S2", "S1", "S1", "S2", None, "S1"]

    def test_refine_moves_a_mislabeled_segment(self):
        # オンラインでは重心が育つ前に取り違えた区間も、会議の後の重心で付け直す
        a, b = _vec(0), _vec(1)
        embs = [a, b, a * 0.6 + b * 0.8, a, b, a, b]
        embs = [e / np.linalg.norm(e) for e in embs]
        labs = ["S1", "S2", "S1", "S1", "S2", "S1", "S2"]
        out = refine_labels(embs, [3.0] * 7, labs)
        assert out[2] == "S2"
        assert refine_labels([], [], []) == []
        assert refine_labels([None], [0.3], [None]) == [None]

    def test_absorb_small_extra_speaker(self):
        # 2人の会話に、1区間だけ別のまとまり(同じ人の2つ目)ができても、育たなければ合わせる
        cl = OnlineClusters(threshold=0.9, absorb_frac=0.03, absorb_after=15)
        assert cl.assign(_vec(0), 3.0) == "S1"
        assert cl.assign(_vec(1), 3.0) == "S2"
        odd = _vec(0) * 0.8 + _vec(2) * 0.6                   # S1 の声の崩れたもの
        assert cl.assign(odd, 1.5) == "S3"
        for k in range(20):
            cl.assign(_vec(k % 2), 3.0)
        assert cl.aliases == {"S3": "S1"}
        assert len(cl) == 2
        # たくさん話す3人目は合わせない
        cl2 = OnlineClusters(threshold=0.9, absorb_frac=0.03, absorb_after=15)
        for k in range(30):
            cl2.assign(_vec(k % 3), 3.0)
        assert len(cl2) == 3 and cl2.aliases == {}
        # 誰とも似ていない小さい話者(発言の少ない別の人)は absorb_sim 未満なら合わせない
        cl3 = OnlineClusters(threshold=0.9, absorb_frac=0.03, absorb_after=15, absorb_sim=0.2)
        cl3.assign(_vec(0), 3.0)
        cl3.assign(_vec(1), 3.0)
        assert cl3.assign(_vec(2), 1.5) == "S3"
        for k in range(20):
            cl3.assign(_vec(k % 2), 3.0)
        assert len(cl3) == 3 and cl3.aliases == {}

    def test_nearest_does_not_change_state(self):
        cl = OnlineClusters(threshold=0.5)
        assert cl.nearest(_vec(0)) is None
        cl.assign(_vec(0), 3.0)
        cl.assign(_vec(1), 3.0)
        before = [x.copy() for x in cl._sums]
        sid, sim = cl.nearest(_vec(1) * 0.9 + _vec(0) * 0.1)
        assert sid == "S2" and 0.9 < sim <= 1.0
        assert set(cl.sims(_vec(0))) == {"S1", "S2"}
        assert len(cl) == 2 and all((a == b).all() for a, b in zip(before, cl._sums))

    def test_find_model(self):
        with tempfile.TemporaryDirectory() as td:
            tmp_path = Path(td)
            TestOnlineClusters._fm(tmp_path)


    @staticmethod
    def _fm(tmp_path):
        assert find_model(tmp_path) is None
        (tmp_path / "speaker").mkdir()
        p = tmp_path / "speaker" / MODEL_NAMES[0]
        p.write_bytes(b"x")
        assert find_model(tmp_path) == str(p)
        q = tmp_path / MODEL_NAMES[0]
        q.write_bytes(b"x")
        assert find_model(tmp_path) == str(q)            # models/ 直下を先に見る




# TSUYAKU_SPEAKER_MODEL で別のモデルのファイルを指定して確かめられる
MODEL = os.environ.get("TSUYAKU_SPEAKER_MODEL") or find_model(ROOT / "models")


@unittest.skipIf(MODEL is None, "話者モデルが models/ に無い(scripts/get_models.sh)")
class TestTrackerWithModel(unittest.TestCase):
    def test_tracker_with_model_threads_and_shapes(self):
        trk = SpeakerTracker(MODEL)
        rng = np.random.default_rng(0)
        # 声の代わりの音(高さの違う2つの「声」)。同じ音は同じ話者、短い音は None。
        t = np.arange(int(2.5 * 16000)) / 16000

        def voice(f0):
            x = sum(np.sin(2 * np.pi * f0 * k * t) / k for k in range(1, 8))
            x *= 0.5 + 0.5 * np.sin(2 * np.pi * 3 * t)
            return (0.1 * x + 0.003 * rng.standard_normal(len(t))).astype(np.float32)

        a, b = voice(120.0), voice(240.0)
        assert trk.assign(a[:4000]) is None
        sa = trk.assign(a)
        assert sa == "S1"
        assert trk.assign(a.copy()) == "S1"
        assert trk.assign((a * 32767).astype(np.int16)) == "S1"           # int16 も受ける
        a48 = np.interp(np.arange(len(a) * 3) / 48000, t, a).astype(np.float32)
        assert trk.assign(a48, sr=48000) == "S1"                          # 48kHz も受ける
        # スレッド安全: 同時に呼んでも壊れない
        out: list = []

        def work():
            for _ in range(3):
                out.append(trk.assign(b))
        ths = [threading.Thread(target=work) for _ in range(4)]
        [th.start() for th in ths]
        [th.join() for th in ths]
        assert len(out) == 12 and all(o is not None for o in out)
        assert trk.canonical("S1") == "S1" and isinstance(trk.aliases, dict)
        # 状態を変えない見当: 短い区間でも id と類似を返し、話者も記録も増やさない
        n0, h0 = trk.n_speakers, len(trk.refine())
        assert trk.guess(a[:400]) is None                     # 0.25秒未満は見ない
        sid, sim = trk.guess(a[:8000])                         # 0.5秒
        assert sid == "S1" and -1.0 <= sim <= 1.0
        assert trk.n_speakers == n0 and len(trk.refine()) == h0
        assert trk.different_speaker(a[:8000], "S9") is None   # 知らない話者
        i, spk = trk.assign_indexed(a)
        assert spk == "S1" and i == len(trk.refine()) - 1      # 番号は refine() の並びと同じ
        assert trk.refine()[i] == "S1"
        trk.reset()
        assert trk.n_speakers == 0


if __name__ == "__main__":
    unittest.main()
