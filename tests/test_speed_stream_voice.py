"""長い1文の読み上げを、できた所から鳴らす(11/08 speed)。

見つけた問題(作った日↔英・日↔中の会話を リモートサーバーで実時間に流した評価): 訳の確定から声の鳴り始めまで
1〜3秒(長い英文 150字で 3〜6秒)。訳は「、」「,」で続く長い1文が多く、文ごとに分けて鳴らす仕組み
(voice_chunks)では分けられない。Kokoro は1文を読点ごとに作って callback で渡せる(中国語は sherpa が
読点で分ける。英語は発音記号の列を渡すので、こちらで読点ごとの列にする)。最初の読点までの音ができたら
鳴らし始め、続きは鳴らしている間に作る。音そのものは全部を1回で作ったものと同じ切れ目(読点)で作られる。
"""
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from tsuyaku.tts import TTS


class _Audio:
    def __init__(self, x, sr):
        self.samples, self.sample_rate = x, sr


class _FakeKokoro:
    """読点ごとに 0.2 秒かけて作り、できるたびに callback へ渡す。"""
    sample_rate = 24000

    def __init__(self):
        self.calls = 0

    def generate(self, text, sid=0, speed=1.0, callback=None):
        self.calls += 1
        parts = [p for p in text.replace("，", "，|").split("|") if p]
        allx = []
        for p in parts:
            time.sleep(0.2)
            x = (np.ones(2400 * len(p)) * 0.1).astype(np.float32)
            allx.append(x)
            if callback is not None:
                callback(x, len(allx) / len(parts))
        return _Audio(np.concatenate(allx), 24000)


def tts():
    t = TTS("", "", {}, voices={"zh": "kokoro:45"}, speeds={"zh": 1.0})
    t.engine["kokoro"] = _FakeKokoro()
    return t


class TestStreamedVoice(unittest.TestCase):
    def test_first_clause_plays_before_whole_sentence_is_made(self):
        plays = []
        t = tts()
        t0 = time.time()
        with patch.object(TTS, "_play_wav", lambda self_, w: plays.append(time.time() - t0) or True):
            t.say("下个月的展会上我们计划介绍三款新产品，所以我想在这周之内决定各自的负责人。", "zh")
        self.assertEqual(len(plays), 2)          # 読点で2つに分けて鳴らす
        self.assertLess(plays[0], 0.33)          # 1つ目は 0.2 秒で鳴る(全部を作り終える 0.4 秒を待たない)
        # 読点ごとに作る(sherpa が1回の中で読点ごとに作るのと同じ単位。11/08 speed で読点を作る単位にした)
        self.assertEqual(t.engine["kokoro"].calls, 2)

    def test_prepared_streaming_clip_plays_parts_in_order(self):
        got = []
        t = tts()
        with patch.object(TTS, "_play_wav", lambda self_, w: got.append(len(w)) or True):
            t.prepare("我们计划介绍三款新产品，所以我想决定负责人。", "zh")
            time.sleep(0.05)                     # 作っている途中で鳴らしに来る
            t.say("我们计划介绍三款新产品，所以我想决定负责人。", "zh")
        self.assertEqual(len(got), 2)
        self.assertEqual(t.prepared_hits, 1)

    def test_voicevox_still_plays_whole_wav_once(self):
        t = TTS("", "", {}, voices={"ja": "voicevox:13"}, speeds={"ja": 1.0}, voicevox_url="http://vv")
        with patch.object(TTS, "_voicevox_synth", lambda *a: b"RIFFxx"), \
             patch.object(TTS, "_play_wav", return_value=True) as pw:
            t.say("こんにちは、今日はいい天気ですね。", "ja")
        pw.assert_called_once_with(b"RIFFxx")


KOK = Path(__file__).resolve().parents[1] / "models/kokoro-int8-multi-lang-v1_0"


@unittest.skipUnless((KOK / "lexicon-us-en.txt").exists(), "kokoro モデルが無い")
class TestEnglishClauses(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from tsuyaku.tts import _load_g2p
        cls.g = _load_g2p(KOK)

    def test_long_sentence_is_split_at_commas(self):
        s = ("We plan to introduce three new products at next month's exhibition, "
             "so I would like to decide who is in charge of each one by the end of this week.")
        self.assertEqual(len(self.g.sentences(s)), 1)               # 今まで: 1つの列
        cl = self.g.sentences(s, clauses=True)
        self.assertEqual(len(cl), 2)
        self.assertTrue(cl[0].endswith(","))
        self.assertEqual(" ".join(cl), " ".join(self.g.sentences(s)))   # 音の記号は同じ(切れ目だけ)

    def test_short_clause_is_not_split_off(self):
        # 「Yes, ...」「However, ...」のような短い頭は切り離さない(1語で鳴らすと不自然)
        self.assertEqual(len(self.g.sentences("Yes, I will send the results by Friday morning.",
                                              clauses=True)), 1)


if __name__ == "__main__":
    unittest.main()
