"""切れ目なく話し続けたときの長い発話を、句読点・息つぎで区切って先に流す(11/08)。

見つけた問題(リーダーが実記録で見た型、このMac = SenseVoice):
  1. 切れ目のない独白が 9〜12.5 秒の1単位になり、訳 101 字・読み上げ +3.7 秒、次の文は 14.9 秒遅れで
     読み上げを飛ばした。区間は音量の谷(0.18秒)が無いと 12 秒の上限まで閉じない(動画の BGM で谷が埋まる)。
  2. 途中字幕が 6.5 秒あたりから伸びなくなった。作った独白に和音の BGM を重ねると、SenseVoice は
     窓が長くなったところで出来事を BGM と判定し(文字は正しく出ている)、SenseVoiceSTT がそれを
     丸ごと捨てていた。音楽・雑音だけのときの文字は「我。」「I.」「.」など2字以下。
"""
import unittest

import numpy as np

from tsuyaku.audio import StreamSegmenter
from tsuyaku.segmenter import Endpointer, pick_cut
from tsuyaku.stt import keep_event

SR = 16000

# 「今天我们来讨论一下新产品的发布计划。首先，市场部已经完成了用户调查」の字と時刻(SenseVoice の形)
TOK = list("今天我们来讨论一下新产品的发布计划。首先，市场部已经完成了用户调查")
TS = [0.06, 0.24, 0.48, 0.66, 0.84, 1.08, 1.32, 1.62, 1.80, 2.04, 2.28, 2.46, 2.70, 2.88, 3.06,
      3.24, 3.42, 3.66, 3.84, 4.08, 4.32, 4.80, 5.04, 5.28, 5.52, 5.64, 5.88, 6.06, 6.30, 6.60,
      6.84, 7.02, 7.20]


class TestKeepEvent(unittest.TestCase):
    """BGM と判定されても、窓の頭が話し声(話している途中から判定が変わった)で字が十分なら残す。
    歌詞のある曲(頭から BGM)・ざわめき(Speech_Noise)・拍手は今まで通り捨てる(11/08 リーダー)。"""

    def test_speech_is_kept(self):
        self.assertTrue(keep_event("Speech", "可以。"))

    def test_bgm_continuing_speech_is_kept(self):
        self.assertTrue(keep_event("BGM", "首先，市场部已经完成了用户调查。", head_speech=True))
        self.assertTrue(keep_event("BGM", "I think we should start now.", head_speech=True))

    def test_bgm_without_speech_head_is_dropped(self):
        self.assertFalse(keep_event("BGM", "我们一起走过春天和夏天", head_speech=False))   # 歌
        self.assertFalse(keep_event("BGM", "我们一起走过春天和夏天"))

    def test_bgm_short_noise_is_dropped(self):
        for t in ("我。", "I.", ".", "그."):
            self.assertFalse(keep_event("BGM", t, head_speech=True), t)

    def test_other_events_are_dropped(self):
        self.assertFalse(keep_event("Laughter", "哈哈哈哈哈哈哈"))
        self.assertFalse(keep_event("Speech_Noise", "今日は会議の資料を", head_speech=True))


class TestPickCut(unittest.TestCase):
    def test_short_utterance_is_not_cut(self):
        self.assertIsNone(pick_cut(TOK[:18], TS[:18], 4.0))

    def test_long_utterance_cuts_after_sentence_end(self):
        t = pick_cut(TOK, TS, 7.4)
        self.assertAlmostEqual(t, (3.66 + 3.84) / 2, places=2)   # 「。」と次の字「首」の間

    def test_sentence_end_too_early_or_too_late_is_not_used(self):
        tok = list("好。") + list("我们来讨论一下新产品的发布计划和市场调查结果")
        ts = [0.1, 0.3] + [0.5 + 0.3 * i for i in range(len(tok) - 2)]
        self.assertIsNone(pick_cut(tok, ts, 7.0))                 # 前半が短すぎる(1字の文)

    def test_comma_when_no_sentence_end_and_very_long(self):
        tok = list("我们来讨论一下新产品的发布计划，") + list("还有市场部已经完成的用户调查结果和预算")
        ts = [0.1 + 0.3 * i for i in range(len(tok))]
        dur = ts[-1] + 0.3
        self.assertIsNone(pick_cut(tok, ts, 7.0))                 # 読点は長くなってから
        t = pick_cut(tok, ts, max(dur, 9.5))
        i = tok.index("，")
        self.assertAlmostEqual(t, (ts[i] + ts[i + 1]) / 2, places=2)

    def test_breath_gap_when_no_punctuation_at_all(self):
        tok = list("我们来讨论一下新产品的发布计划还有市场部已经完成的用户调查结果和预算")
        ts = [0.1 + 0.3 * i for i in range(len(tok))]
        k = 20
        ts = ts[:k] + [x + 0.6 for x in ts[k:]]                  # 20字目の後に 0.9 秒の息つぎ
        t = pick_cut(tok, ts, ts[-1] + 0.4)
        self.assertAlmostEqual(t, (ts[k - 1] + ts[k]) / 2, places=2)

    def test_english_sentence_end(self):
        tok = ["▁We", "▁will", "▁start", "▁the", "▁launch", "▁next", "▁week", ".",
               "▁But", "▁the", "▁budget", "▁is", "▁smaller", "▁this", "▁year"]
        ts = [0.2 + 0.45 * i for i in range(len(tok))]
        t = pick_cut(tok, ts, 7.2)
        self.assertAlmostEqual(t, (ts[7] + ts[8]) / 2, places=2)


def loud(n_s, f=220.0):
    t = np.arange(int(n_s * SR)) / SR
    return (0.2 * np.sin(2 * np.pi * f * t)).astype(np.float32)


class TestSegmenterCut(unittest.TestCase):
    def feed(self, sg, x):
        out = []
        for i in range(0, len(x), 512):
            out += sg.feed(x[i:i + 512], True)
        return out

    def test_cut_closes_head_and_keeps_listening(self):
        sg = StreamSegmenter(SR, cut_s=1e9, max_s=12.0)
        self.assertEqual(self.feed(sg, loud(8.0)), [])
        s0 = sg.speaking_since()
        self.assertTrue(sg.request_cut(s0 + 5.0))
        out = self.feed(sg, loud(0.1))
        self.assertEqual(len(out), 1)
        _a, lo, hi, _pre = out[0]
        self.assertAlmostEqual(hi, s0 + 5.0, places=2)
        self.assertTrue(sg.was_cut(hi))
        self.assertAlmostEqual(sg.speaking_since(), s0 + 5.0, places=2)   # 後半はそのまま続く
        out = self.feed(sg, loud(1.0)) + sg.flush_current()
        _a, lo2, hi2, pre2 = out[0]
        self.assertAlmostEqual(lo2, s0 + 5.0, places=2)                    # 前半と重ねない
        self.assertAlmostEqual(pre2, 0.0, places=3)
        self.assertFalse(sg.was_cut(hi2))

    def test_cut_outside_current_segment_is_ignored(self):
        sg = StreamSegmenter(SR, cut_s=1e9)
        self.assertFalse(sg.request_cut(3.0))                             # 話していない
        self.feed(sg, loud(2.0))
        self.assertFalse(sg.request_cut(sg.speaking_since() + 5.0))       # まだ届いていない所


class TestEndpointerCut(unittest.TestCase):
    def test_cut_head_is_emitted_without_waiting(self):
        ep = Endpointer()
        a = np.zeros(SR * 5, np.float32)
        out = ep.push(a, 0.0, 5.0, "我们来讨论一下新产品的发布计划和", "zh", now=10.0, cut=True)
        self.assertEqual(len(out), 1)                 # 句点が無くても待たない(区切った前半)
        self.assertIsNone(ep.pending())

    def test_without_cut_open_fragment_waits(self):
        ep = Endpointer()
        a = np.zeros(SR * 5, np.float32)
        out = ep.push(a, 0.0, 5.0, "我们来讨论一下新产品的发布计划和", "zh", now=10.0)
        self.assertEqual(out, [])
        self.assertIsNotNone(ep.pending())


class TestBridgeLongCut(unittest.TestCase):
    """STT ワーカーの途中字幕 → 区切りの頼み(Bridge._long_cut)。"""

    def bridge(self, split_s=6.0):
        import types
        from tsuyaku.__main__ import Bridge
        b = Bridge()
        b.s.long_split_s = split_s
        sg = StreamSegmenter(SR, cut_s=1e9)
        x = loud(9.0)
        for i in range(0, len(x), 512):
            sg.feed(x[i:i + 512], True)
        b.capture = types.SimpleNamespace(segmenter=sg)
        return b, sg

    def test_requests_cut_at_sentence_end(self):
        b, sg = self.bridge()
        head = sg.speaking_since() - 0.5
        it = (np.zeros(int(7.4 * SR), np.float32), sg.speaking_since(), head)
        at = b._long_cut(it, TOK, TS, "zh", ("zh", "ja"))
        self.assertAlmostEqual(at, head + 3.75, places=2)
        self.assertEqual(sg._cut_req, int(round(at * SR)))

    def test_off_when_disabled_or_out_of_pair(self):
        b, sg = self.bridge(split_s=0)
        it = (np.zeros(int(7.4 * SR), np.float32), sg.speaking_since(), sg.speaking_since())
        self.assertIsNone(b._long_cut(it, TOK, TS, "zh", ("zh", "ja")))
        b, sg = self.bridge()
        self.assertIsNone(b._long_cut(it, TOK, TS, "ko", ("zh", "ja")))
        self.assertIsNone(sg._cut_req)

    def test_worker_passes_cut_flag_to_endpointer(self):
        from pathlib import Path
        src = (Path(__file__).resolve().parents[1] / "src/tsuyaku/__main__.py").read_text()
        self.assertIn("sg.was_cut(hi)", src)
        self.assertIn("spk=spk_now, cut=cut)", src)
        self.assertIn("self._long_cut(it, toks, stamps, lang, pair_t)", src)


class TestVoiceBreakdown(unittest.TestCase):
    """短い発話の読み上げが +3〜4.3 秒と遅い: どこで待ったかを [voice +x] の行に内訳で残す(11/08)。
    話し終わり → 区切りの確定(Endpointer の続き待ち)→ 訳ができて行列へ → 行列から取る → 鳴り始め。"""

    def test_parts(self):
        from tsuyaku.__main__ import voice_breakdown
        s = voice_breakdown(100.0, {"ready": 100.9, "enq": 101.4}, 103.3, 103.6)
        self.assertEqual(s, "内訳 区切り0.9 訳0.5 行列1.9 鳴り始め0.3")

    def test_missing_marks(self):
        from tsuyaku.__main__ import voice_breakdown
        self.assertEqual(voice_breakdown(100.0, None, 101.0, 101.2), "")
        self.assertEqual(voice_breakdown(None, {"ready": 1, "enq": 2}, 3, 4), "")

    def test_wired(self):
        from pathlib import Path
        src = (Path(__file__).resolve().parents[1] / "src/tsuyaku/__main__.py").read_text()
        self.assertIn("self.tts_q.put((trans, spoken, uid, end, sp, src_dur, marks))", src)
        self.assertIn("voice_breakdown(_e, _m, _dq, time.time())", src)


class TestShortReplyHold(unittest.TestCase):
    """短い1語の発話(「可以。」)の読み上げが +3〜4.3 秒(11/08)。切り分け: 続けて声(動画・相手)が
    0.8 秒以内に始まると、Endpointer は「続きを話している最中」として期限から最大 3 秒待っていた
    (話し終わりから確定まで 3.9 秒)。文末まで言い終わった短い断片は、話し中の延長を 1 秒までにする。"""

    def hold(self, text, since_gap):
        ep = Endpointer(rejoin=lambda a, span=None: ("", None, 0.0))
        a = np.zeros(int(0.6 * SR), np.float32)
        out = ep.push(a, 10.0, 10.6, text, "zh", now=100.0)
        t = 100.0
        while not out and t < 110:
            t += 0.1
            out = ep.tick(t, speaking_since=10.6 + since_gap)
        return round(t - 100.0, 1)

    def test_finished_short_reply_does_not_wait_3s(self):
        self.assertLessEqual(self.hold("可以。", 0.3), 2.0)
        self.assertLessEqual(self.hold("好的。", 0.3), 2.0)

    def test_unfinished_fragment_still_waits_for_continuation(self):
        self.assertGreaterEqual(self.hold("我觉得", 0.3), 3.0)

    def test_quiet_after_reply_is_unchanged(self):
        self.assertLessEqual(self.hold("可以。", 1.5), 1.0)


class TestCutHeadText(unittest.TestCase):
    """区切った前半は、途中字幕の字(後ろの文脈まで聞いた字)をそのまま使う。前半だけを聞き直すと
    BGM の上では文脈が短くなって字の誤りが増えた(eval_long: zh BGM の CER 16.4% → 17.9%)。"""

    def test_pick_cut_at_gives_head_index(self):
        from tsuyaku.segmenter import pick_cut_at, head_text
        t, i = pick_cut_at(TOK, TS, 7.4)
        self.assertAlmostEqual(t, 3.75, places=2)
        self.assertEqual(head_text(TOK, i), "今天我们来讨论一下新产品的发布计划。")
        self.assertEqual(head_text(["▁We", "▁will", "▁go", "."], 3), "We will go.")

    def test_segmenter_keeps_head_text(self):
        sg = StreamSegmenter(SR, cut_s=1e9)
        x = loud(8.0)
        for k in range(0, len(x), 512):
            sg.feed(x[k:k + 512], True)
        s0 = sg.speaking_since()
        self.assertTrue(sg.request_cut(s0 + 5.0, text="前半の字。", lang="ja"))
        out = sg.feed(loud(0.05), True)
        hi = out[0][2]
        self.assertEqual(sg.cut_text(hi), ("前半の字。", "ja"))
        self.assertIsNone(sg.cut_text(hi + 1.0))

    def test_worker_uses_head_text(self):
        from pathlib import Path
        src = (Path(__file__).resolve().parents[1] / "src/tsuyaku/__main__.py").read_text()
        self.assertIn("ct = sg.cut_text(hi)", src)


class _R:
    def __init__(self, text, event, lang="zh"):
        self.text, self.event, self.lang = text, f"<|{event}|>", f"<|{lang}|>"
        self.tokens, self.timestamps = list(text), [0.1 * i for i in range(len(text))]


class _Rec:
    """SenseVoice の代わり: 窓の長さで出来事を決める(長い窓 = BGM、頭3秒だけ = head)。"""
    def __init__(self, long_event, head_event):
        self.long_event, self.head_event = long_event, head_event

    def create_stream(self):
        import types
        st = types.SimpleNamespace()
        st.accept_waveform = lambda sr, x: setattr(st, "n", len(x))
        st.result = None
        st._rec = self
        return st

    def decode_stream(self, st):
        ev = self.head_event if st.n <= 3 * SR else self.long_event
        st.result = _R("首先市场部已经完成了用户调查", ev)


def sv(long_event, head_event, now):
    import threading
    from tsuyaku.stt import SenseVoiceSTT
    e = SenseVoiceSTT.__new__(SenseVoiceSTT)
    e.rec, e.langs, e.prompt, e._lock = _Rec(long_event, head_event), ("zh", "ja"), "", threading.Lock()
    e.clock = lambda: now[0]
    return e


class TestBgmContinuation(unittest.TestCase):
    """区切った後半(話の途中から始まる区間)は頭から BGM と判定されやすく、頭の確かめでは
    捨てられた(eval_long: ja の BGM で後半の1文が丸ごと消えた)。数秒前まで話し声と聞こえて
    いたなら、BGM の判定でも話の続きとして残す。"""

    def test_recent_speech_keeps_bgm_window(self):
        now = [100.0]
        e = sv("BGM", "BGM", now)
        x = np.zeros(SR * 6, np.float32)
        self.assertEqual(e.transcribe(x)[0], "")              # 話し声を聞いていない: 捨てる(歌)
        e.rec.long_event = "Speech"
        self.assertTrue(e.transcribe(np.zeros(SR * 4, np.float32))[0])   # 話し声を聞いた
        e.rec.long_event = "BGM"
        now[0] += 2.0
        self.assertTrue(e.transcribe(x)[0])                   # 2 秒後の BGM の窓: 続きとして残す
        now[0] += 10.0
        self.assertEqual(e.transcribe(x)[0], "")              # ずっと後: 頭の確かめに戻る

    def test_head_speech_still_keeps(self):
        e = sv("BGM", "Speech", [0.0])
        self.assertTrue(e.transcribe(np.zeros(SR * 6, np.float32))[0])


class TestCutTailContinues(unittest.TestCase):
    """区切った後半の区間(始まりが区切った所)は話の続き。4 秒の窓では足りず、後半が長いと
    頭から BGM の後半を丸ごと捨てた(eval_long の ja・BGM 1本で CER 6.3% → 20.4%)。"""

    def test_cont_keeps_bgm_window(self):
        e = sv("BGM", "BGM", [100.0])
        x = np.zeros(SR * 6, np.float32)
        self.assertEqual(e.transcribe(x)[0], "")
        self.assertTrue(e.transcribe(x, cont=True)[0])
        self.assertTrue(e.transcribe_stamped(x, cont=True)[0])

    def test_segmenter_knows_tail_start(self):
        sg = StreamSegmenter(SR, cut_s=1e9)
        x = loud(8.0)
        for k in range(0, len(x), 512):
            sg.feed(x[k:k + 512], True)
        s0 = sg.speaking_since()
        sg.request_cut(s0 + 5.0)
        out = sg.feed(loud(0.05), True)
        out += sg.feed(loud(1.0), True) + sg.flush_current()
        self.assertTrue(sg.starts_at_cut(out[1][1]))      # 後半の区間の頭
        self.assertFalse(sg.starts_at_cut(out[0][1]))

    def test_worker_and_interim_pass_cont(self):
        from pathlib import Path
        src = (Path(__file__).resolve().parents[1] / "src/tsuyaku/__main__.py").read_text()
        self.assertIn("ear.transcribe_stamped(it[0], cont=cont)", src)
        self.assertIn("self.stt.transcribe(a, cont=True)", src)


class TestSongRate(unittest.TestCase):
    """歌の窓は字が遅い(VOICEVOX の歌: BGM と判定された窓の字の速さ p90 2.0 字/秒)。BGM の上の話し声は
    p10 5.0 字/秒。BGM と判定された窓を「話の続き」として残すのは、字の速さが BGM_MIN_RATE 以上のときだけ。"""

    def test_slow_chars_are_dropped_even_when_continuing(self):
        e = sv("BGM", "Speech", [100.0])
        slow = _R("あさのひかりがまどをてらす", "BGM", "ja")
        slow.timestamps = [0.6 * i for i in range(len(slow.tokens))]     # 1.7 字/秒(歌)
        e.rec.decode_stream = lambda st: setattr(st, "result", slow)
        x = np.zeros(SR * 9, np.float32)
        self.assertEqual(e.transcribe(x, cont=True)[0], "")

    def test_speech_rate_is_kept(self):
        e = sv("BGM", "Speech", [100.0])
        self.assertTrue(e.transcribe(np.zeros(SR * 6, np.float32), cont=True)[0])   # 10 字/秒

    def test_rate_helper(self):
        from tsuyaku.stt import char_rate
        self.assertAlmostEqual(char_rate(list("あいうえお"), [0.0, 0.25, 0.5, 0.75, 1.0]), 5.0)
        self.assertAlmostEqual(char_rate(list("あい。うえ"), [0.0, 0.5, 0.6, 1.0, 1.5]), 4 / 1.5, places=2)
        self.assertIsNone(char_rate(["あ"], [0.0]))


if __name__ == "__main__":
    unittest.main()
