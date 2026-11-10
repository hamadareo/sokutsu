"""音声レベルの区切り(Endpointer)。11/05夜「会話の区切りがまだまだ」。

合成会話の評価(正解15ターン)で、現行(無音0.5秒で区切る+文字で続き待ち)は
別人の発言と連結5件・文の分割3件・字幕まで最大9.4秒。0.2〜0.4秒の素早い
交代で2人分が1区間になり、whisper が片方の言語で聞くため英語の返事が丸ごと
消えていた。Endpointer は短い無音で細かく区切り、文末で「終わったか」を判断:
句点等=すぐ訳す / 途中(読点・助詞・but)=続きを待つ / 言語が変わった=切り離す。
続きが来たら音声ごと繋いで聞き直す(リモートサーバーの耳なら0.1〜0.2秒)。
"""
import unittest

import numpy as np

from tsuyaku.segmenter import Endpointer

SR = 16000


def tone(sec):
    return np.full(int(SR * sec), 0.1, dtype=np.float32)


class Rec:
    """聞き直し(transcribe)の呼び出しを記録し、決めた結果を返す。"""

    def __init__(self, result=("繋いだ文です。", "ja", 0.9)):
        self.calls = []
        self.result = result

    def __call__(self, audio):
        self.calls.append(len(audio) / SR)
        return self.result


class TestEndpointer(unittest.TestCase):
    def test_sentence_end_emits_immediately(self):
        ep = Endpointer(rejoin=None)
        out = ep.push(tone(1.0), 0.0, 1.0, "今日は何時に終わりますか、教えてください。", "ja", now=10.0)
        self.assertEqual([u.text for u in out], ["今日は何時に終わりますか、教えてください。"])
        self.assertEqual(out[0].t_ready, 10.0)

    def test_short_punctuated_fragment_waits_briefly(self):
        # 「私も。」+「そう思います。」: 短い断片の句点は whisper の付け足しのことがある
        rec = Rec(("私もそう思います。", "ja", 0.9))
        ep = Endpointer(rejoin=rec, hold_done_s=0.3)
        self.assertEqual(ep.push(tone(0.7), 0.0, 0.7, "私も。", "ja", now=1.0), [])
        out = ep.push(tone(0.9), 0.8, 1.7, "そう思います。", "ja", now=2.0, pre=0.1)
        self.assertEqual([u.text for u in out], ["私もそう思います。"])

    def test_short_reply_still_goes_out_quickly(self):
        ep = Endpointer(rejoin=None, hold_done_s=0.3)
        self.assertEqual(ep.push(tone(0.5), 0.0, 0.5, "Okay.", "en", now=1.0), [])
        self.assertEqual([u.text for u in ep.tick(1.31)], ["Okay."])

    def test_open_fragment_waits_then_merges_by_rejoin(self):
        rec = Rec(("サンプル数が少ないので、有意差はまだ出ていません。", "ja", 0.9))
        ep = Endpointer(rejoin=rec, hold_open_s=1.0)
        self.assertEqual(ep.push(tone(1.5), 0.0, 1.5, "サンプル数が少ないので、", "ja", now=2.0), [])
        self.assertEqual(ep.tick(2.5), [])  # まだ待つ
        # 期限(3.0)を過ぎても、続きを話している最中なら確定しない
        self.assertEqual(ep.tick(3.5, speaking_since=2.0), [])
        out = ep.push(tone(2.0), 2.0, 4.0, "有意差はまだ出ていません。", "ja", now=4.3)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0].text, "サンプル数が少ないので、有意差はまだ出ていません。")
        self.assertEqual(out[0].parts, 2)
        self.assertEqual((out[0].lo, out[0].hi), (0.0, 4.0))
        self.assertEqual(len(rec.calls), 1)  # 繋いだ音声を1回だけ聞き直す
        self.assertAlmostEqual(rec.calls[0], 4.0, delta=0.05)  # 1.5 + 間0.5 + 2.0

    def test_open_fragment_flushes_after_hold(self):
        ep = Endpointer(rejoin=None, hold_open_s=1.0)
        ep.push(tone(1.0), 0.0, 1.0, "この結果は、", "ja", now=1.2)
        self.assertEqual(ep.tick(2.1), [])
        out = ep.tick(2.25)
        self.assertEqual([u.text for u in out], ["この結果は、"])

    def test_language_switch_splits_turns(self):
        # 日本語の途中断片のあと英語が来た = 別の人。繋がない(以前は英語が消えた)
        rec = Rec()
        ep = Endpointer(rejoin=rec, hold_open_s=1.0)
        ep.push(tone(1.0), 0.0, 1.0, "それ全部手作業で書いてるのか", "ja", now=1.2)
        out = ep.push(tone(2.0), 1.25, 3.25, "Yes, I write everything by hand.", "en", now=3.4)
        self.assertEqual([(u.text, u.lang) for u in out],
                         [("それ全部手作業で書いてるのか", "ja"),
                          ("Yes, I write everything by hand.", "en")])
        self.assertEqual(rec.calls, [])

    def test_speech_started_too_late_is_a_new_unit(self):
        # 間が待ち時間より長い = 続きではない(届いた時刻ではなく間の長さで決める)
        rec = Rec()
        ep = Endpointer(rejoin=rec, hold_open_s=1.0)
        ep.push(tone(1), 0.0, 1.0, "この結果は、", "ja", now=1.2)
        out = ep.push(tone(1), 2.6, 3.6, "別の話です。", "ja", now=3.8) + ep.flush()
        self.assertEqual([u.text for u in out], ["この結果は、", "別の話です。"])
        self.assertEqual(out[0].t_ready, 2.2)   # 期限で出たことにする
        self.assertEqual(rec.calls, [])

    def test_speaking_cap_bounds_the_wait(self):
        ep = Endpointer(rejoin=None, hold_open_s=1.0, speak_cap_s=3.0)
        ep.push(tone(1), 0.0, 1.0, "この結果は、", "ja", now=1.0)
        self.assertEqual(ep.tick(4.9, speaking_since=1.5), [])
        self.assertEqual([u.text for u in ep.tick(5.0, speaking_since=1.5)], ["この結果は、"])

    def test_late_speech_does_not_hold_fragment(self):
        # 期限の時点の声が、待ち時間より後に始まった = 別の発話。断片は待たずに出す
        ep = Endpointer(rejoin=None, hold_open_s=1.0)
        ep.push(tone(1), 0.0, 1.0, "この結果は、", "ja", now=1.0)
        self.assertEqual([u.text for u in ep.tick(2.0, speaking_since=2.5)], ["この結果は、"])

    def test_soft_wait_for_unpunctuated_text(self):
        # 句読点なし・途中でもない(whisper は日本語の文末「。」を落としがち)は短く待つ
        ep = Endpointer(rejoin=None, hold_open_s=1.0, hold_soft_s=0.3)
        self.assertEqual(ep.push(tone(1), 0, 1, "センスのカメラ", "ja", now=5.0), [])
        self.assertEqual(ep.tick(5.2), [])
        self.assertEqual([u.text for u in ep.tick(5.31)], ["センスのカメラ"])

    def test_classify_sentence_ends(self):
        from tsuyaku.segmenter import classify
        self.assertEqual(classify("まだ出ていません", "ja"), "done")      # 句点なしの文末
        self.assertEqual(classify("多分5時頃だと思います", "ja"), "done")
        self.assertEqual(classify("水深は", "ja"), "open")
        self.assertEqual(classify("こんにちは", "ja"), "soft")
        self.assertEqual(classify("センスのカメラ", "ja"), "soft")
        self.assertEqual(classify("That is interesting, but...", "en"), "open")
        self.assertEqual(classify("If we collect more data next month", "en"), "mid")
        self.assertEqual(classify("Okay.", "en"), "done")
        # whisper が断片に付けた「.」(AMI 会議)
        self.assertEqual(classify("We wanted to.", "en"), "open")
        self.assertEqual(classify("And this is the.", "en"), "open")
        self.assertEqual(classify("I think so.", "en"), "done")

    def test_text_join_when_no_rejoin(self):
        # このMacの耳(1回2秒強)では聞き直さず文字を繋ぐ
        ep = Endpointer(rejoin=None, hold_open_s=1.0)
        ep.push(tone(1), 0, 1, "If we collect more data,", "en", now=1.1)
        out = ep.push(tone(1), 1.6, 2.6, "we can confirm it.", "en", now=2.7)
        self.assertEqual([u.text for u in out], ["If we collect more data, we can confirm it."])

    def test_overlapping_preroll_is_trimmed_when_rejoining(self):
        # 次の区間の先頭0.4秒(プリロール)が前の区間と重なる: 二重に聞かせない
        rec = Rec()
        ep = Endpointer(rejoin=rec, hold_open_s=1.0)
        ep.push(tone(2.0), 0.0, 2.0, "カメラは二種類使っていて", "ja", now=2.3)
        ep.push(tone(1.5), 1.8, 3.3, "片方は近赤外線にも反応します。", "ja", now=3.5)
        self.assertAlmostEqual(rec.calls[0], 3.3, delta=0.05)

    def test_capture_clock_going_backwards_does_not_merge(self):
        # 読み上げ後に VAD を作り直すと位置が戻る → 繋がない
        rec = Rec()
        ep = Endpointer(rejoin=rec, hold_open_s=5.0)
        ep.push(tone(1), 10.0, 11.0, "こちらが", "ja", now=1.0)
        out = ep.push(tone(1), 0.5, 1.5, "資料です。", "ja", now=1.5) + ep.flush()
        self.assertEqual([u.text for u in out], ["こちらが", "資料です。"])
        self.assertEqual(rec.calls, [])

    def test_merge_capped_by_length(self):
        ep = Endpointer(rejoin=None, hold_open_s=5.0, max_merge_s=6.0)
        ep.push(tone(4), 0, 4, "長い前置きがあって、", "ja", now=4.1)
        out = ep.push(tone(4), 4.2, 8.2, "さらに続きます。", "ja", now=8.3) + ep.flush()
        self.assertEqual([u.text for u in out], ["長い前置きがあって、", "さらに続きます。"])

    def test_rejoin_failure_falls_back_to_text_join(self):
        def boom(_a):
            raise RuntimeError("remote down")
        ep = Endpointer(rejoin=boom, hold_open_s=1.0)
        ep.push(tone(1), 0, 1, "こちらが", "ja", now=1.1)
        out = ep.push(tone(1), 1.3, 2.3, "資料です。", "ja", now=2.4)
        self.assertEqual([u.text for u in out], ["こちらが資料です。"])

    def test_flush_returns_pending(self):
        ep = Endpointer(rejoin=None)
        ep.push(tone(1), 0, 1, "to the", "en", now=1.0)
        self.assertEqual([u.text for u in ep.flush()], ["to the"])
        self.assertEqual(ep.flush(), [])

    def test_pending_text_exposed_for_preview(self):
        ep = Endpointer(rejoin=None)
        ep.push(tone(1), 0, 1, "サンプル数が少ないので、", "ja", now=1.0)
        self.assertEqual(ep.pending_text(), "サンプル数が少ないので、")


if __name__ == "__main__":
    unittest.main()


class TestMeetingSegmentation(unittest.TestCase):
    """11/06 擬似会議 v1(eval/meeting/README.md): 別の人の短い発言が1行に繋がる・
    聞き直しで文が消える。"""

    def test_different_speakers_are_not_joined(self):
        # 「田中です」+「佐藤です」: 同じ言語・間0.3秒でも、話者が違えば繋がない
        rejoined = []
        ep = Endpointer(rejoin=lambda a, span=None: (rejoined.append(1), ("田中です佐藤です", "ja", 0.9))[1],
                        hold_open_s=1.2)
        ep.push(np.zeros(16000, np.float32), 0.0, 1.0, "田中です", "ja", now=1.0, spk=lambda: "S1")
        us = ep.push(np.zeros(16000, np.float32), 1.2, 2.2, "佐藤です", "ja", now=2.5, spk=lambda: "S2")
        self.assertEqual([u.text for u in us], ["田中です"])
        self.assertEqual(ep.pending_text(), "佐藤です")
        self.assertEqual(rejoined, [])

    def test_same_or_unknown_speaker_still_joins(self):
        for a, b in (("S1", "S1"), ("S1", None), (None, "S2")):
            ep = Endpointer(rejoin=lambda x, span=None: ("田中です、佐藤さんと", "ja", 0.9), hold_open_s=1.2)
            ep.push(np.zeros(16000, np.float32), 0.0, 1.0, "田中です、", "ja", now=1.0, spk=lambda a=a: a)
            ep.push(np.zeros(16000, np.float32), 1.2, 2.2, "佐藤さんと", "ja", now=2.5, spk=lambda b=b: b)
            self.assertEqual(ep.pending_text(), "田中です、佐藤さんと", (a, b))

    def test_no_speaker_info_is_unchanged(self):
        ep = Endpointer(rejoin=lambda x, span=None: ("田中です佐藤です", "ja", 0.9), hold_open_s=1.2)
        ep.push(np.zeros(16000, np.float32), 0.0, 1.0, "田中です", "ja", now=1.0)
        us = ep.push(np.zeros(16000, np.float32), 1.2, 2.2, "佐藤です", "ja", now=2.5)
        self.assertEqual([u.text for u in us], ["田中です佐藤です"])   # 話者が分からなければ今まで通り

    def test_rejoin_much_shorter_falls_back_to_text_join(self):
        # 「Okay」+「データは共有ドライブで共有することにします」→ 聞き直しが「OK」だけ
        ep = Endpointer(rejoin=lambda x, span=None: ("OK", "ja", 0.9), hold_open_s=1.2)
        ep.push(np.zeros(16000, np.float32), 0.0, 1.2, "オーケー", "ja", now=1.2)
        us = ep.push(np.zeros(3 * 16000, np.float32), 1.4, 4.7,
                     "データは共有ドライブで共有することにします。", "ja", now=5.0)
        got = " ".join(u.text for u in us) + ep.pending_text()
        self.assertIn("データは共有ドライブで共有することにします", got)

    def test_rejoin_normal_length_is_used(self):
        ep = Endpointer(rejoin=lambda x, span=None: ("データは共有ドライブで共有します。", "ja", 0.9),
                        hold_open_s=1.2)
        ep.push(np.zeros(16000, np.float32), 0.0, 1.0, "データは", "ja", now=1.0)
        us = ep.push(np.zeros(16000, np.float32), 1.2, 2.2, "共有ドライブで共有します。", "ja", now=2.5)
        self.assertEqual([u.text for u in us], ["データは共有ドライブで共有します。"])


class TestShortSegmentSpeaker(unittest.TestCase):
    """11/07 擬似会議 v2: 短い区間(0.6秒未満など)は話者が None で「話者が違えば繋がない」が
    効かず、まだ3件繋がった(「…来週の会議で決めます了解です」)。話者が分からない側は、状態を
    変えない見当(speakers.different_speaker)で「別の人」と分かれば繋がない。"""

    def _ep(self, differ, calls):
        def d(audio, spk):
            calls.append((len(audio), spk))
            return differ
        return Endpointer(rejoin=lambda a, span=None: ("決めます、了解です", "ja", 0.9),
                          hold_open_s=1.2, spk_differ=d)

    def test_short_new_segment_of_other_speaker_is_not_joined(self):
        calls = []
        ep = self._ep(True, calls)
        ep.push(np.zeros(16000 * 2, np.float32), 0.0, 2.0, "来週の会議で決めます、", "ja", now=2.0,
                spk=lambda: "S1")
        us = ep.push(np.ones(8000, np.float32), 2.2, 2.7, "了解です", "ja", now=3.0, spk=lambda: None)
        self.assertEqual([u.text for u in us], ["来週の会議で決めます、"])
        self.assertEqual(calls, [(8000, "S1")])          # 新しい短い区間の声を、前の話者と比べた

    def test_short_pending_of_other_speaker_is_not_joined(self):
        calls = []
        ep = self._ep(True, calls)
        ep.push(np.ones(8000, np.float32), 0.0, 0.5, "はい、", "ja", now=0.5, spk=lambda: None)
        us = ep.push(np.zeros(16000 * 2, np.float32), 0.7, 2.7, "延長コードも必要です。", "ja", now=3.0,
                     spk=lambda: "S2")
        self.assertEqual(us[0].text, "はい、")
        self.assertEqual(calls, [(8000, "S2")])          # 続き待ちの短い区間の声を、新しい話者と比べた

    def test_same_or_unsure_joins(self):
        for differ in (False, None):
            ep = self._ep(differ, [])
            ep.push(np.zeros(16000 * 2, np.float32), 0.0, 2.0, "来週の会議で決めます、", "ja", now=2.0,
                    spk=lambda: "S1")
            ep.push(np.ones(8000, np.float32), 2.2, 2.7, "了解です", "ja", now=3.0, spk=lambda: None)
            self.assertEqual(ep.pending_text(), "", differ)   # 繋いで「決めます、了解です」で確定

    def test_interpret_mode_never_asks(self):
        calls = []
        ep = self._ep(True, calls)
        ep.push(np.zeros(16000 * 2, np.float32), 0.0, 2.0, "来週の会議で決めます、", "ja", now=2.0)
        ep.push(np.ones(8000, np.float32), 2.2, 2.7, "了解です", "ja", now=3.0)
        self.assertEqual(calls, [])
