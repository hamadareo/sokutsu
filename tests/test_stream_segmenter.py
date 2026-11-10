"""StreamSegmenter: VAD の声あり判定 + 音量の落ち込みで区間を切る(11/05夜)。

silero VAD は声が止まってもしばらく「声あり」が残り、0.25〜0.35秒の話者交代を
切れなかった(合成会話 18境目中9つ見逃し)。音量が部屋の雑音近くまで0.18秒
落ちたら、VAD が声ありのままでも区間を閉じる。
"""
import unittest

import numpy as np

from tsuyaku.audio import StreamSegmenter

SR = 16000
CH = 512


def run(sg, spans, total_s, vad_tail_s=0.3, noise=0.002, level=0.1):
    """spans=[(開始秒, 終了秒)] に声(level)、他は noise。VAD は声の間と、その後
    vad_tail_s だけ『声あり』を返す(silero の遅れの再現)。"""
    rng = np.random.default_rng(0)
    n = int(total_s * SR)
    a = rng.normal(0, noise, n).astype(np.float32)
    voiced = np.zeros(n, bool)
    vad = np.zeros(n, bool)
    for lo, hi in spans:
        i, j = int(lo * SR), int(hi * SR)
        a[i:j] += (level * np.sin(np.arange(j - i) * 0.3)).astype(np.float32)
        voiced[i:j] = True
        vad[min(n, i + int(0.25 * SR)):min(n, j + int(vad_tail_s * SR))] = True
    out = []
    for k in range(0, n, CH):
        out += sg.feed(a[k:k + CH], bool(vad[min(n - 1, k + CH - 1)]))
    return out


class TestStreamSegmenter(unittest.TestCase):
    def test_quick_turn_switch_is_cut(self):
        # 0.25秒の交代: VAD は声ありのまま(遅れ0.3秒)だが音量で切れる
        sg = StreamSegmenter()
        out = run(sg, [(0.5, 2.5), (2.75, 4.5)], 6.0)
        self.assertEqual(len(out), 2)
        (a1, lo1, hi1, pre1), (a2, lo2, hi2, pre2) = out
        self.assertAlmostEqual(hi1, 2.55, delta=0.1)
        self.assertAlmostEqual(lo2 + pre2, 2.75, delta=0.12)  # 話し始め

    def test_preroll_never_reaches_into_previous_segment(self):
        sg = StreamSegmenter(pre_s=0.5)
        out = run(sg, [(0.5, 2.5), (2.75, 4.5)], 6.0)
        self.assertGreaterEqual(out[1][1], out[0][2] - 1e-9)

    def test_vad_end_closes_at_last_loud(self):
        sg = StreamSegmenter(cut_s=1e9)          # 音量では切らない(このMacの耳)
        out = run(sg, [(0.5, 2.0)], 4.0, vad_tail_s=0.5)
        self.assertEqual(len(out), 1)
        self.assertAlmostEqual(out[0][2], 2.05, delta=0.1)   # 語尾の余白0.08秒込み
        self.assertGreater(len(out[0][0]), int(1.5 * SR))   # 手前の補い込み

    def test_coarse_mode_keeps_quick_turns_together(self):
        sg = StreamSegmenter(cut_s=1e9)
        out = run(sg, [(0.5, 2.5), (2.75, 4.5)], 6.0, vad_tail_s=0.6)  # VAD は途切れない
        self.assertEqual(len(out), 1)

    def test_max_length_cut(self):
        sg = StreamSegmenter(max_s=3.0)
        out = run(sg, [(0.5, 8.0)], 9.0)
        self.assertGreaterEqual(len(out), 2)
        self.assertTrue(all(o[2] - (o[1] + o[3]) <= 3.05 for o in out))

    def test_quiet_distant_mic_is_kept(self):
        # AMI 会議の卓上マイク1本: 声の RMS 0.005前後・雑音0.0015。絶対値の下限(0.004)で
        # 区間を捨てていた(169区間中56)。判定は雑音との比だけにする
        sg = StreamSegmenter()
        out = run(sg, [(0.5, 2.5), (3.5, 5.0)], 6.0, noise=0.0015, level=0.007)
        self.assertEqual(len(out), 2)

    def test_app_gate_adapts_to_room_noise(self):
        from pathlib import Path
        src = (Path(__file__).resolve().parents[1] / "src/tsuyaku/__main__.py").read_text()
        self.assertIn("gate = min(gate, max(2.0 * cap.segmenter.noise, 0.0015))", src)

    def test_tiny_blip_is_dropped(self):
        sg = StreamSegmenter(min_seg_s=0.3)
        out = run(sg, [(0.5, 0.6)], 2.0)
        self.assertEqual(out, [])

    def test_speaking_since_and_drop_current(self):
        sg = StreamSegmenter()
        n = int(1.0 * SR)
        x = (0.1 * np.sin(np.arange(n) * 0.3)).astype(np.float32)
        for k in range(0, n, CH):
            sg.feed(x[k:k + CH], True)
        self.assertIsNotNone(sg.speaking_since())
        sg.drop_current()
        self.assertIsNone(sg.speaking_since())

    def test_clock_never_goes_back(self):
        sg = StreamSegmenter()
        out1 = run(sg, [(0.5, 1.5)], 2.5)
        sg.drop_current()
        out2 = run(sg, [(0.5, 1.5)], 2.5)
        self.assertGreater(out2[0][1], out1[0][2])


if __name__ == "__main__":
    unittest.main()
