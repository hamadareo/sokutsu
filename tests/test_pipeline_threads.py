"""翻訳と読み上げを別スレッドに分ける(「翻訳が遅い/翻訳中は録音されていない」対策)。

11/05 15:42〜15:48 ライブ実測: 1スレッドで whisper→翻訳→読み上げを直列に
やっていたため、読み上げ(3〜6秒)のあいだに聞こえた発話が seg_q に積まれ、
3個たまると pick_segments が古いものを捨てて「翻訳中」の行列だけが遅くなった。
しかもユーザーには「翻訳中」のあいだマイクが閉じているように見えた。
→ 録音は翻訳中ずっと開いているが、読み上げの間だけ閉じる。読み上げを別
スレッドに逃がせば、読み上げが長くても翻訳は詰まらない。

スレッド構成: 録音(SpeechCapture) → seg_q → STTワーカー(確率守卫まで)
→ text_q → テキストワーカー(heldマージ+翻訳) → tts_q → 読み上げワーカー
(この間だけマイクを閉じる)。
"""
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = (ROOT / "src" / "tsuyaku" / "__main__.py").read_text()


class TestWorkerThreads(unittest.TestCase):
    def test_three_workers_defined(self):
        self.assertIn("def _stt_worker(", SRC)
        self.assertIn("def _text_worker(", SRC)
        self.assertIn("def _tts_worker(", SRC)

    def test_tts_goes_through_queue_not_inline(self):
        # _utterance は say を直接呼ばない。say はワーカーの中だけ。
        i = SRC.index("def _utterance(")
        seg = SRC[i:]
        self.assertIn("self.tts_q.put((trans, spoken, uid, end, sp, src_dur, marks))", seg)
        self.assertNotIn("self.tts.say(trans, spoken)", seg)

    def test_say_happens_in_tts_worker_between_pause_and_resume(self):
        # マイクは読み上げの前後だけで閉じる(従来テストと同じ契約、ワーカー内)
        w = SRC.index("def _tts_worker(")
        self.assertLess(w, SRC.index("self.capture.paused.set()", w))
        self.assertLess(SRC.index("self.capture.paused.set()", w),
                        SRC.index("self.tts.say(trans, spoken)"))
        self.assertLess(SRC.index("self.tts.say(trans, spoken)"),
                        SRC.index("self.capture.resume_after_tts(s.gate_tail_s)", w))

    def test_guards_run_before_translation_queueing(self):
        # エコー/幻覚守卫はテキストワーカー(翻訳)より前の STT ワーカーに居る
        self.assertLess(SRC.index("def _stt_worker("),
                        SRC.index("is_echo(self.last_spoken, text)"))
        self.assertLess(SRC.index("hallucinated_out_of_pair("),
                        SRC.index("def _text_worker("))

    def test_workers_started_in_conversation(self):
        c = SRC.index("def _conversation(")
        seg = SRC[c:SRC.index("def _utterance(")]
        self.assertIn("self._stt_worker", seg)
        self.assertIn("self._text_worker", seg)
        self.assertIn("self._tts_worker", seg)
        self.assertIn(".start()", seg)


class TestMicMeterHonest(unittest.TestCase):
    def test_level_shows_zero_while_paused(self):
        # 読み上げでマイクが閉いているあいだ、レベル計を凍結させない
        # (「録音されていないのに反応だけしている」誤解を防ぐ)
        i = SRC.index("def _pump(")
        seg = SRC[i:SRC.index("def _started(")]
        self.assertIn("cap.paused.is_set()", seg)   # cap = self.capture(一度だけ読む)


if __name__ == "__main__":
    unittest.main()
