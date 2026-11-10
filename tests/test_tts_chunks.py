"""長い訳は文ごとに作って、1文目ができたらすぐ鳴らす(11/06)。

見つけた問題: 読み上げは訳の全文を合成し終えてから鳴っていた。2文の訳で合成
1.7〜2.5秒(VOICEVOX/Kokoro, M3 実測: 74字 2.48秒 → 1文目50字 1.72秒、
英語104字 1.75秒 → 1文目42字 0.82秒)。1文目だけ先に作れば0.5〜0.9秒早く鳴る。
2文目以降は1文目を鳴らしている間に作る。声は文単位で切るので抑揚は変わらない。
"""
import time
import unittest
from unittest.mock import patch

from tsuyaku.tts import TTS, voice_chunks


def tts():
    return TTS("", "", {}, voices={"ja": "voicevox:13", "en": "voicevox:13", "ko": "say:Yuna"},
               speeds={"ja": 1.0}, voicevox_url="http://vv.local")


class TestVoiceChunks(unittest.TestCase):
    def test_split_japanese_sentences(self):
        self.assertEqual(
            voice_chunks("心理学の授業、刑事司法の授業、そして芸術鑑賞の授業です。あ、ダンス鑑賞の授業ですね。"),
            ["心理学の授業、刑事司法の授業、そして芸術鑑賞の授業です。", "あ、ダンス鑑賞の授業ですね。"])

    def test_split_english_sentences(self):
        self.assertEqual(
            voice_chunks("My research focuses on IoT and AI. It is really interesting! Right?"),
            ["My research focuses on IoT and AI.", "It is really interesting! Right?"])

    def test_short_or_single_sentence_not_split(self):
        self.assertEqual(voice_chunks("はい。"), ["はい。"])
        self.assertEqual(voice_chunks("Which parts?"), ["Which parts?"])
        self.assertEqual(voice_chunks("Dr. Smith is here."), ["Dr. Smith is here."])
        self.assertEqual(voice_chunks("It costs 3.5 dollars. OK."), ["It costs 3.5 dollars. OK."])

    def test_tiny_pieces_merge_forward(self):
        # 短すぎる文(「はい。」等)だけを1回の再生にしない: 次の文とまとめる
        self.assertEqual(voice_chunks("はい。それは脳の前頭葉という部分に影響します。"),
                         ["はい。それは脳の前頭葉という部分に影響します。"])

    def test_chinese(self):
        self.assertEqual(voice_chunks("我们正在讨论精神分裂症和抑郁症。精神分裂症影响大脑的哪个部分？"),
                         ["我们正在讨论精神分裂症和抑郁症。", "精神分裂症影响大脑的哪个部分？"])


class TestChunkedSay(unittest.TestCase):
    def test_first_sentence_plays_before_rest_is_synthesized(self):
        events = []

        def synth(self_, text, spk, speed):
            time.sleep(0.15 if len(text) > 20 else 0.05)
            events.append(("synth", text))
            return b"RIFF" + text.encode()

        def play(self_, wav):
            events.append(("play", wav[4:].decode()))
            return True

        t = tts()
        a = "これは一つ目の文で、少し長めに書いてあります。"
        b = "そしてこれが二つ目の文で、やはり長めに書いてあります。"
        with patch.object(TTS, "_voicevox_synth", synth), patch.object(TTS, "_play_wav", play):
            self.assertTrue(t.say(a + b, "ja"))
        self.assertEqual(events.index(("play", a)) < events.index(("synth", b)), True)
        self.assertEqual([e for e in events if e[0] == "play"], [("play", a), ("play", b)])

    def test_prepare_then_say_uses_prepared_chunks(self):
        calls = []

        def synth(self_, text, spk, speed):
            calls.append(text)
            return b"RIFF"

        t = tts()
        a = "これは一つ目の文で、少し長めに書いてあります。"
        b = "そしてこれが二つ目の文で、やはり長めに書いてあります。"
        with patch.object(TTS, "_voicevox_synth", synth), \
             patch.object(TTS, "_play_wav", return_value=True):
            t.prepare(a + b, "ja")
            time.sleep(0.1)
            t.say(a + b, "ja")
        self.assertEqual(calls, [a, b])          # 作り直さない

    def test_cancel_stops_remaining_sentences(self):
        played = []
        t = tts()

        def play(self_, wav):
            played.append(wav)
            t.cancel()                            # 1文目の途中で止められた
            return True

        a = "これは一つ目の文で、少し長めに書いてあります。"
        b = "そしてこれが二つ目の文で、やはり長めに書いてあります。"
        with patch.object(TTS, "_voicevox_synth", lambda s_, x, sp, v: b"RIFF"), \
             patch.object(TTS, "_play_wav", play):
            t.say(a + b, "ja")
        self.assertEqual(len(played), 1)

    def test_on_start_fires_once(self):
        t = tts()
        fired = []
        t.on_start = lambda: fired.append(1)
        a = "これは一つ目の文で、少し長めに書いてあります。"
        b = "そしてこれが二つ目の文で、やはり長めに書いてあります。"

        def play(self_, wav):
            self_._started()
            return True

        with patch.object(TTS, "_voicevox_synth", lambda s_, x, sp, v: b"RIFF"), \
             patch.object(TTS, "_play_wav", play):
            t.say(a + b, "ja")
        self.assertEqual(fired, [1])


if __name__ == "__main__":
    unittest.main()
