"""入力の自動音量調整(AutoGain)。11/05夜 AMI 会議の卓上マイク1本で、声が小さく
silero VAD が発話の3割強に反応しなかった。自動で持ち上げると単語再現率
34→49% / 33→56%(上限61/68%)、雑音だけの2分では誤検出0件のまま。
"""
import unittest
from pathlib import Path

import numpy as np

from tsuyaku.audio import AutoGain

ROOT = Path(__file__).resolve().parents[1]


def feed(agc, sig, ch=512):
    out = []
    for i in range(0, len(sig), ch):
        out.append(agc(sig[i:i + ch]))
    return np.concatenate(out)


class TestAutoGain(unittest.TestCase):
    def test_quiet_speech_is_raised(self):
        rng = np.random.default_rng(0)
        n = 16000 * 6
        sig = rng.normal(0, 0.0015, n).astype(np.float32)          # 部屋の雑音
        sig[16000:16000 * 5] += (0.007 * np.sin(np.arange(16000 * 4) * 0.3)).astype(np.float32)
        agc = AutoGain()
        feed(agc, sig)
        self.assertGreater(agc.gain, 3.0)
        self.assertLessEqual(agc.gain, 8.0)

    def test_normal_mic_is_left_alone(self):
        sig = (0.1 * np.sin(np.arange(16000 * 4) * 0.3)).astype(np.float32)
        agc = AutoGain()
        out = feed(agc, sig)
        self.assertAlmostEqual(agc.gain, 1.0, delta=0.05)
        np.testing.assert_allclose(out[-512:], sig[-512:], atol=1e-6)

    def test_noise_is_not_pushed_above_cap(self):
        rng = np.random.default_rng(1)
        sig = rng.normal(0, 0.004, 16000 * 8).astype(np.float32)   # 雑音だけ
        agc = AutoGain(noise_cap=0.01)
        feed(agc, sig)
        self.assertLessEqual(agc.gain * agc.noise, 0.0105)

    def test_capture_applies_gain_before_vad(self):
        src = (ROOT / "src/tsuyaku/audio.py").read_text()
        i = src.index("def on_chunk", src.index("class SpeechCapture"))   # 開く道具は _MicStream(11/08 hear)
        body = src[i:src.index("def stop", i)]
        self.assertLess(body.index("self.agc("), body.index("self.vad.accept_waveform("))
        self.assertIn("self.segmenter.feed(g,", body)


if __name__ == "__main__":
    unittest.main()
