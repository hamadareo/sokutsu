"""話し始めの欠け(プリロール)。11/05夕 E2E 実測:「サンプル数が少ないので」が
VAD の区間開始の遅れで「プル数が…」になった(remote/このMac の両方で同じ = 耳ではなく
区切り側の問題)。区間の手前 0.2 秒で「アンプル」、0.4 秒で「サンプル」に戻る。
"""
import unittest
from pathlib import Path

import numpy as np

from tsuyaku.audio import PreRoll

ROOT = Path(__file__).resolve().parents[1]


class TestPreRoll(unittest.TestCase):
    def test_prefix_is_audio_just_before_start(self):
        pr = PreRoll(sr=10, pre_s=0.3, keep_s=5)
        a = np.arange(40, dtype=np.float32)
        for i in range(0, 40, 7):  # 不揃いなチャンクで流し込む
            pr.feed(a[i:i + 7])
        seg = a[20:30]
        out = pr.extend(20, seg)
        np.testing.assert_array_equal(out, a[17:30])

    def test_start_near_zero_takes_what_exists(self):
        pr = PreRoll(sr=10, pre_s=0.5, keep_s=5)
        a = np.arange(20, dtype=np.float32)
        pr.feed(a)
        np.testing.assert_array_equal(pr.extend(2, a[2:8]), a[0:8])

    def test_old_audio_is_trimmed_but_prefix_still_works(self):
        pr = PreRoll(sr=10, pre_s=0.3, keep_s=2)  # 20サンプルだけ持つ
        a = np.arange(100, dtype=np.float32)
        pr.feed(a)
        self.assertLessEqual(len(pr._buf), 20 + 1)
        np.testing.assert_array_equal(pr.extend(90, a[90:95]), a[87:95])

    def test_reset_restarts_index_like_vad_reset(self):
        # 読み上げ後は VAD を reset する → 区間の start も 0 から数え直しになる
        pr = PreRoll(sr=10, pre_s=0.3, keep_s=5)
        pr.feed(np.arange(30, dtype=np.float32))
        pr.reset()
        b = np.arange(100, 120, dtype=np.float32)
        pr.feed(b)
        np.testing.assert_array_equal(pr.extend(5, b[5:10]), b[2:10])

    def test_capture_segments_with_preroll_and_drops_on_tts(self):
        # 11/05夜: SpeechCapture は StreamSegmenter(手前 pre_s の補いを内蔵)で区切る
        src = (ROOT / "src/tsuyaku/audio.py").read_text()
        i = src.index("def on_chunk", src.index("class SpeechCapture"))   # 開く道具は _MicStream(11/08 hear)
        seg = src[i:src.index("def stop", i)]
        self.assertIn("self.segmenter.feed(", seg)
        j = src.index("def resume_after_tts")
        self.assertIn("self.segmenter.drop_current()", src[j:])


if __name__ == "__main__":
    unittest.main()
