"""ライブ経路のエコー抑止の振る舞いテスト(実マイク・実音声再生なし)。

_layers_:
1. ゲート: paused 中のマイクフレームは VAD に届かない。解錠後は届く。
2. resume_after_tts: VADバッファを捨て、tail分入力を遮断し、最後に解錠する。
3. is_echo: 読み上げ文の返り聞き(whisperの言い間違い揺れ込み)は捨て、
   次の本物の発話と、元話者の原文は通す。
4. 順序: エコー判定は翻訳リクエストより前(空振りでLLMを走らせない)。
"""
import queue
import threading
import time
import types
import unittest
from pathlib import Path

import numpy as np

import tsuyaku.audio as audio_mod
from tsuyaku.audio import SpeechCapture
from tsuyaku.langdetect import is_echo

ROOT = Path(__file__).resolve().parents[1]
VAD = str(ROOT / "models" / "silero_vad.onnx")


class _VadRec:
    """本物のVADモデルの代わりに、コール回数を数えるスタブ。"""

    def __init__(self):
        self.n_accept = 0
        self.n_reset = 0

    def accept_waveform(self, x):
        self.n_accept += 1

    def empty(self):
        return True

    def is_speech_detected(self):
        return False

    def reset(self):
        self.n_reset += 1


class _FakeStream:
    """sounddevice.InputStream の偽物。渡された callback をテスト側から叩ける。"""
    cb = None

    def __init__(self, **kw):
        _FakeStream.cb = kw["callback"]

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class TestGate(unittest.TestCase):
    def setUp(self):
        self._orig_sd = audio_mod.sd
        _FakeStream.cb = None
        audio_mod.sd = types.SimpleNamespace(InputStream=_FakeStream)
        self.q: queue.Queue = queue.Queue()
        self.cap = SpeechCapture(VAD, self.q)
        self.cap.vad = _VadRec()
        self.cap.start()
        time.sleep(0.25)  # スレッドが with sd.InputStream(...) を通るまで待つ

    def tearDown(self):
        self.cap.stop()
        audio_mod.sd = self._orig_sd

    def test_paused_frames_never_reach_vad(self):
        cb = _FakeStream.cb
        self.assertIsNotNone(cb, "キャプチャスレッドが InputStream を開いていない")
        frames = np.random.default_rng(1).normal(0, 0.1, (512, 1)).astype(np.float32)
        self.cap.paused.set()
        for _ in range(20):
            cb(frames, 512, None, None)
        self.assertEqual(self.cap.vad.n_accept, 0, "読み上げ中にフレームがVADへ入った")
        self.cap.paused.clear()
        for _ in range(20):
            cb(frames, 512, None, None)
        self.assertEqual(self.cap.vad.n_accept, 20, "解錠後にフレームがVADへ届いていない")
        self.assertGreater(self.cap.level, 0.0, "レベルメーターが動いていない")

    def test_resume_after_tts_resets_vad_and_opens_after_tail(self):
        self.cap.paused.clear()
        t0 = time.time()
        self.cap.resume_after_tts(0.3)
        dt = time.time() - t0
        self.assertGreaterEqual(dt, 0.28, "tail分待たずに解錠した")
        self.assertEqual(self.cap.vad.n_reset, 1, "VADバッファを捨てていない")
        self.assertFalse(self.cap.paused.is_set(), "resume後にマイクが開いていない")
        # 経過中に一時的でも paused だった(=読み上げ直後のテールを捨ている)
        seen = []
        th = threading.Thread(
            target=lambda: (time.sleep(0.1), seen.append(self.cap.paused.is_set())))
        th.start()
        self.cap.resume_after_tts(0.4)
        th.join()
        self.assertTrue(seen and seen[0], "tail中にマイクが閉じていなかった")


class TestEchoFilter(unittest.TestCase):
    JA = "報酬設計を先に詰めるべきだと思います。"
    TR = "I think we should nail down the reward design first."

    def test_exact_repeat_is_echo(self):
        self.assertTrue(is_echo(self.TR, self.TR))

    def test_whisper_mishearing_variants_are_echo(self):
        # whisperが読み上げ英語を少し違った文字に起こしても捨てる
        self.assertTrue(is_echo(self.TR, "I think we should nail down the reward design first"))
        self.assertTrue(is_echo(self.TR, "I think we should nail down our reward design first."))
        self.assertTrue(is_echo(self.TR, "i think we should nail down the reward design 1st"))

    def test_fragment_of_spoken_is_echo(self):
        self.assertTrue(is_echo(self.TR, "nail down the reward design first"))

    def test_different_utterance_passes(self):
        other = "The tracking loss mostly comes from occlusion on the third camera."
        self.assertFalse(is_echo(self.TR, other))

    def test_original_foreign_text_passes(self):
        # 元話者の日本語原文は、英語の読み上げの_echo_ではない
        self.assertFalse(is_echo(self.TR, self.JA))

    def test_empty_or_short_never_echo(self):
        self.assertFalse(is_echo("", "hello"))
        self.assertFalse(is_echo(self.TR, "a"))


class TestBridgeOrdering(unittest.TestCase):
    def setUp(self):
        self.src = (ROOT / "src" / "tsuyaku" / "__main__.py").read_text()

    def test_echo_check_precedes_translation(self):
        self.assertLess(
            self.src.index("is_echo(self.last_spoken, text)"),
            self.src.index("msgs = build_messages(pair_t"),
            "エコー判定が翻訳リクエストより後になっている(LLMを空振りする)")
        # 先読み翻訳(11/06)は Endpointer に入った断片だけを訳す: 入れる前にエコー判定
        self.assertLess(self.src.index("is_echo(self.last_spoken, text)"),
                        self.src.index("send(ep.push("))

    def test_mic_closed_around_say(self):
        # _run: paused.set → tts.say → resume_after_tts の順
        i_pause = self.src.index("self.capture.paused.set()")
        i_say = self.src.index("self.tts.say(trans, spoken)")
        i_resume = self.src.index("self.capture.resume_after_tts(s.gate_tail_s)")
        self.assertLess(i_pause, i_say)
        self.assertLess(i_say, i_resume)
        # last_spoken は読み上げ後に更新(次の発話の判定基準になる)
        self.assertGreater(i_resume, self.src.index("self.last_spoken = \"\""))
        self.assertLess(i_say, self.src.index("self.last_spoken = trans"))


if __name__ == "__main__":
    unittest.main()
