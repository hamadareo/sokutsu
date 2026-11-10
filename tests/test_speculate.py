"""先読み翻訳(Speculator): 文がまだ閉じていない間に、今の文字で訳を始めておく(11/06)。

ユーザー「返答内容を予想してさらに速くするのはどうですか？」。実ログでは、
文末で続きを待つ(0.3〜1.2秒)あいだ翻訳は止まっていて、待ち終わってから
0.3〜0.6秒かけて訳していた。待っている間に「ここで文が終わる」と予想して
訳を始め、当たれば(続きが来ずに同じ文字で確定)その訳をそのまま使う。
外れたら(続きが来て文字が変わった)取り消して、新しい文字で予想し直す。
"""
import threading
import time
import unittest

from tsuyaku.speculate import Speculator


class FakeLLM:
    """1文字ずつ遅れて返す翻訳。cancel されたら on_delta が例外を投げて止まる。"""

    def __init__(self, delay=0.01, fail=False):
        self.delay = delay
        self.fail = fail
        self.calls = []
        self.finished = []

    def __call__(self, msgs, on_delta):
        text = msgs
        self.calls.append(text)
        if self.fail:
            raise RuntimeError("server down")
        out = f"<{text}>"
        for ch in out:
            time.sleep(self.delay)
            on_delta(ch)
        self.finished.append(text)
        return out


def make(llm, shown=None):
    return Speculator(run=llm, make_msgs=lambda text, lang: text,
                      show=(lambda *a: shown.append(a)) if shown is not None else None)


class TestSpeculator(unittest.TestCase):
    def test_hit_returns_same_translation_and_replays_deltas(self):
        llm = FakeLLM()
        sp = make(llm)
        sp.update(("今日は", "ja"), [])
        time.sleep(0.02)
        got = []
        out = sp.take("今日は", "ja", got.append)
        self.assertEqual(out, "<今日は>")
        self.assertEqual("".join(got), "<今日は>")   # 途中までの分も含めて全部渡る
        self.assertEqual(len(llm.calls), 1)          # 訳し直していない

    def test_take_after_finished_is_instant(self):
        llm = FakeLLM(delay=0.0)
        sp = make(llm)
        sp.update(("Hello", "en"), [])
        time.sleep(0.05)
        t0 = time.time()
        self.assertEqual(sp.take("Hello", "en", None), "<Hello>")
        self.assertLess(time.time() - t0, 0.02)

    def test_miss_when_text_changed(self):
        llm = FakeLLM()
        sp = make(llm)
        sp.update(("私は英語が", "ja"), [])
        sp.update(("私は英語が苦手です", "ja"), [])   # 続きが来て繋がった
        self.assertIsNone(sp.take("私は英語が", "ja", None))
        self.assertEqual(sp.take("私は英語が苦手です", "ja", None), "<私は英語が苦手です>")

    def test_superseded_guess_is_cancelled(self):
        llm = FakeLLM(delay=0.02)
        sp = make(llm)
        sp.update(("私は英語が", "ja"), [])
        time.sleep(0.03)
        sp.update(("私は英語が苦手です", "ja"), [])
        time.sleep(0.4)
        self.assertNotIn("私は英語が", llm.finished)   # 途中で止めた(サーバーを無駄に使わない)

    def test_emitted_guess_survives_new_pending(self):
        # A が確定して訳待ち(テキストワーカーが前の文を訳している最中)に、
        # 別の人の B が待ちに入っても、A の先読みは取り消さない
        llm = FakeLLM()
        sp = make(llm)
        sp.update(("Good luck", "en"), [])
        sp.update(("そうですか", "ja"), [("Good luck", "en")])
        self.assertEqual(sp.take("Good luck", "en", None), "<Good luck>")
        self.assertEqual(sp.take("そうですか", "ja", None), "<そうですか>")

    def test_failure_falls_back(self):
        sp = make(FakeLLM(fail=True))
        sp.update(("Hello", "en"), [])
        time.sleep(0.02)
        self.assertIsNone(sp.take("Hello", "en", None))   # 呼び出し側が普通に訳す

    def test_lang_must_match(self):
        sp = make(FakeLLM())
        sp.update(("你好", "zh"), [])
        self.assertIsNone(sp.take("你好", "ja", None))

    def test_flush_cancels_everything_pending(self):
        llm = FakeLLM(delay=0.02)
        sp = make(llm)
        sp.update(("長い文の途中", "ja"), [])
        sp.update(None, [])                 # 待っていた断片が捨てられた
        time.sleep(0.4)
        self.assertEqual(llm.finished, [])

    def test_guess_is_shown_while_pending(self):
        shown = []
        sp = make(FakeLLM(), shown)
        sp.update(("Thank you", "en"), [])
        time.sleep(0.3)
        self.assertTrue(shown)
        text, trans, lang, guess = shown[-1]
        self.assertEqual((text, trans, lang, guess), ("Thank you", "<Thank you>", "en", True))

    def test_show_can_be_muted(self):
        # テキストワーカーが確定訳を流している間は、予想訳で大字幕を上書きしない
        shown = []
        sp = make(FakeLLM(), shown)
        sp.mute(True)
        sp.update(("Thank you", "en"), [])
        time.sleep(0.2)
        self.assertEqual(shown, [])
        sp.mute(False)                      # 再開したら最新の予想を出し直す
        self.assertTrue(shown)
        self.assertEqual(shown[-1][1], "<Thank you>")

    def test_take_is_thread_safe_against_streaming(self):
        llm = FakeLLM(delay=0.002)
        sp = make(llm)
        long = "x" * 200
        sp.update((long, "en"), [])
        got = []
        th = threading.Thread(target=lambda: got.append(sp.take(long, "en", None)))
        time.sleep(0.05)
        th.start()
        th.join(5)
        self.assertEqual(got, [f"<{long}>"])

    def test_reserved_jobs_are_bounded(self):
        sp = make(FakeLLM(delay=0.0))
        for i in range(10):
            sp.update(None, [(f"s{i}", "en")])          # 確定のみ(先読み無し)は何も起きない
        for i in range(10):
            sp.update((f"p{i}", "en"), [])
            sp.update(None, [(f"p{i}", "en")])          # 確定したが誰も受け取らない
        self.assertLessEqual(len(sp._jobs), 4)

    def test_ready_called_when_guess_finished(self):
        got = []
        sp = Speculator(run=FakeLLM(delay=0.0), make_msgs=lambda t, l: t,
                        ready=lambda *a: got.append(a))
        sp.update(("Hello", "en"), [])
        time.sleep(0.05)
        self.assertEqual(got, [("Hello", "en", "<Hello>")])

    def test_ready_not_called_for_cancelled_guess(self):
        got = []
        sp = Speculator(run=FakeLLM(delay=0.02), make_msgs=lambda t, l: t,
                        ready=lambda *a: got.append(a))
        sp.update(("Hel", "en"), [])
        time.sleep(0.03)
        sp.update(("Hello there", "en"), [])
        time.sleep(0.5)
        self.assertEqual([g[0] for g in got], ["Hello there"])

    def test_peek_returns_guess_so_far(self):
        sp = make(FakeLLM(delay=0.0))
        self.assertEqual(sp.peek("Hello", "en"), "")
        sp.update(("Hello", "en"), [])
        time.sleep(0.05)
        self.assertEqual(sp.peek("Hello", "en"), "<Hello>")

    def test_preview_guess_does_not_prepare_voice(self):
        # このMacの耳の先出しは外れやすい: 読み上げの音を作って合成を塞がない
        got = []
        sp = Speculator(run=FakeLLM(delay=0.0), make_msgs=lambda t, l: t,
                        ready=lambda *a: got.append(a))
        sp.update(("Hello", "en"), [], preview=True)
        time.sleep(0.05)
        self.assertEqual(got, [])
        self.assertEqual(sp.peek("Hello", "en"), "<Hello>")   # 予想訳としては出す

    def test_stats(self):
        sp = make(FakeLLM(delay=0.0))
        sp.update(("a", "en"), [])
        sp.take("a", "en", None)
        sp.take("b", "en", None)
        self.assertEqual((sp.hits, sp.misses), (1, 1))


class TestEndpointerPending(unittest.TestCase):
    def test_pending_returns_text_and_lang(self):
        import numpy as np
        from tsuyaku.segmenter import Endpointer
        ep = Endpointer()
        self.assertIsNone(ep.pending())
        ep.push(np.zeros(16000, np.float32), 0.0, 1.0, "私は英語が", "ja", now=0.0)
        self.assertEqual(ep.pending(), ("私は英語が", "ja"))


class TestAppWiring(unittest.TestCase):
    def test_app_uses_speculator(self):
        from pathlib import Path
        src = (Path(__file__).resolve().parents[1] / "src/tsuyaku/__main__.py").read_text()
        self.assertIn("self.spec.take(text, lang, on_delta)", src)
        # 鍵は字名を英字に戻した形にそろえる(11/08 terms.py)
        self.assertIn("spec.update(pend, [_nl_key((u.text, u.lang), self._heard_terms) for u in units])", src)


if __name__ == "__main__":
    unittest.main()


class TestInterimKeepsPendingJob(unittest.TestCase):
    def test_interim_preview_does_not_cancel_remote_pending_translation(self):
        """続き待ちの断片(リモートサーバー)の先読みは、途中字幕の予想で止めない。続きが来なければ確定になる
        (11/06 AMI 実時間評価: 途中字幕の予想が続き待ちの訳を止め、確定で訳し直していた)。"""
        llm = FakeLLM(delay=0.01)
        sp = make(llm, [])
        sp.update(("So we wanted,", "en"), [])                       # remote: 続き待ち
        sp.update(("So we wanted, um", "en"), [], preview=True)      # 途中字幕(次の区間)
        time.sleep(0.3)
        self.assertEqual(sp.take("So we wanted,", "en", None), "<So we wanted,>")
        self.assertEqual(sp.hits, 1)

    def test_remote_update_still_cancels_stale_pending(self):
        llm = FakeLLM(delay=0.02)
        sp = make(llm, [])
        sp.update(("So we wanted,", "en"), [])
        sp.update(("So we wanted, um yes.", "en"), [])               # remote: 続きが来て繋がった
        self.assertIsNone(sp.take("So we wanted,", "en", None))
