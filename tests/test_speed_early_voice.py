"""訳を流し込んでいる最中に、読点までできた所の声を作り始める(11/08 speed)。

見つけた問題: 声は訳が全部そろってから作り始めていた。訳の流し込みは長い文で 0.5〜1秒(このMacの
gemma ではもっと長い)かかり、その間 声を作る担当は空いていた。Kokoro の声(英語・中国語)は読点ごとに
別々に作っている(sherpa が読点で分ける。英語は発音記号の列を読点で分ける)ので、読点までの所を先に
作っても、全部をそろえてから作るのと同じ音になる。VOICEVOX(日本語)は読点で分けると聞き取りの
誤りが増えた(1.5% → 16.7%)ので分けない。
"""
import time
import unittest
from unittest.mock import patch

from tsuyaku.tts import TTS, voice_chunks


class TestClauseChunks(unittest.TestCase):
    def test_chinese_clauses(self):
        self.assertEqual(
            voice_chunks("下个月的展会上我们计划介绍三款新产品，所以我想在这周之内决定各自的负责人。", clauses="zh"),
            ["下个月的展会上我们计划介绍三款新产品，", "所以我想在这周之内决定各自的负责人。"])

    def test_english_clauses_keep_short_heads(self):
        self.assertEqual(
            voice_chunks("Yes, I will email you the results by Friday morning, "
                         "because the analysis needs a little more time.", clauses="en"),
            ["Yes, I will email you the results by Friday morning,",
             "because the analysis needs a little more time."])

    def test_short_tail_joins_previous(self):
        self.assertEqual(voice_chunks("We will review the price internally, thanks.", clauses="en"),
                         ["We will review the price internally, thanks."])

    def test_no_clause_split_by_default(self):
        self.assertEqual(len(voice_chunks("所以我想在这周之内，决定各自的负责人。")), 1)


class _FakeKokoro:
    sample_rate = 24000

    def __init__(self, log):
        self.log = log

    def generate(self, text, sid=0, speed=1.0, callback=None):
        import numpy as np
        self.log.append(("start", text, time.time()))
        time.sleep(0.1)
        x = (np.ones(2400) * 0.1).astype(np.float32)
        if callback:
            callback(x, 1.0)

        class A:
            samples, sample_rate = x, 24000
        return A()


class TestStableChunks(unittest.TestCase):
    def setUp(self):
        self.t = TTS("", "", {}, voices={"zh": "kokoro:45", "en": "kokoro:21", "ja": "voicevox:13"},
                     speeds={})

    def test_chunk_is_stable_once_enough_text_follows(self):
        part = "下个月的展会上我们计划介绍三款新产品，所以"
        self.assertEqual(self.t.stable_chunks(part, "zh"), [])          # 後ろがまだ短い
        part = "下个月的展会上我们计划介绍三款新产品，所以我想在这周之内"
        self.assertEqual(self.t.stable_chunks(part, "zh"), ["下个月的展会上我们计划介绍三款新产品，"])

    def test_voicevox_has_no_early_chunks(self):
        self.assertEqual(self.t.stable_chunks("来月の展示会では、新しい製品を三つ紹介する予定なので、", "ja"), [])

    def test_early_prepare_is_used_by_final_say(self):
        log = []
        self.t.engine["kokoro"] = _FakeKokoro(log)
        final = "下个月的展会上我们计划介绍三款新产品，所以我想在这周之内决定各自的负责人。"
        with patch.object(TTS, "_play_wav", return_value=True):
            for c in self.t.stable_chunks("下个月的展会上我们计划介绍三款新产品，所以我想在这周之内", "zh"):
                self.t.prepare(c, "zh", early=True)
            time.sleep(0.15)
            self.t.prepare(final, "zh")
            self.t.say(final, "zh")
        made = [x[1] for x in log]
        self.assertEqual(made, ["下个月的展会上我们计划介绍三款新产品，", "所以我想在这周之内决定各自的负责人。"])


class TestAppEarlyVoice(unittest.TestCase):
    def test_app_prepares_stable_chunks_while_streaming(self):
        from pathlib import Path
        src = (Path(__file__).resolve().parents[1] / "src/tsuyaku/__main__.py").read_text()
        body = src[src.index("    def _utterance_body("):]
        self.assertIn("self._early_voice(", body[:body.index("translate_with_retry(base, model, msgs, on_delta)")])


if __name__ == "__main__":
    unittest.main()
