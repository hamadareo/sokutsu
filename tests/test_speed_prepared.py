"""作り置きの声を、速さが少し違っても使う(11/08 speed)。

見つけた問題: 読み上げの速さが「自動」(話のペースに合わせる。新しく使う人の既定)のとき、訳を行列に入れる時と
読む直前で速さを2回決める(PaceController.choose は呼ぶたびに少しずつ寄せる)。速さは 0.01 刻みで
作り置きの鍵に入るので、確定訳を入れた時に作った音(×1.11)と読む時の速さ(×1.08)が合わず、
読む時にもう一度合成していた(合成 0.3〜1.9 秒を待つ。作り置きの合成も合成の錠を取り合う)。
予想訳(先読み)の作り置きは速さを持たない(×1.0)ので、自動では当たらなかった。
"""
import time
import unittest
from unittest.mock import patch

from tsuyaku.tts import TTS


def tts():
    return TTS("", "", {}, voices={"ja": "voicevox:13"}, speeds={"ja": 1.0},
               voicevox_url="http://vv.local")


class TestNearSpeed(unittest.TestCase):
    def _run(self, prep_speed, say_speed, text="こんにちは、今日はいい天気ですね。"):
        calls = []

        def synth(self_, t, spk, speed):
            calls.append(round(speed, 2))
            return b"RIFF" + t.encode()

        t = tts()
        with patch.object(TTS, "_voicevox_synth", synth), \
             patch.object(TTS, "_play_wav", return_value=True):
            t.prepare(text, "ja", prep_speed)
            time.sleep(0.1)
            t.say(text, "ja", say_speed)
        return calls, t

    def test_close_speed_uses_prepared_clip(self):
        calls, t = self._run(1.11, 1.08)
        self.assertEqual(calls, [1.11])            # 作り直さない
        self.assertEqual(t.prepared_hits, 1)
        self.assertAlmostEqual(t.last_speed, 1.11)  # 実際に鳴らした速さ(1秒あたりの字数を合わせる用)

    def test_base_speed_prepared_by_guess_is_used_for_auto(self):
        calls, _t = self._run(None, 1.05)          # 予想訳の作り置き(×1.0)
        self.assertEqual(calls, [1.0])

    def test_far_speed_is_resynthesized(self):
        calls, t = self._run(1.0, 1.25)
        self.assertEqual(calls, [1.0, 1.25])
        self.assertAlmostEqual(t.last_speed, 1.25)

    def test_chunked_say_uses_close_clips_without_new_synth(self):
        text = "来月の展示会では新しい製品を三つ紹介します。それぞれの担当者を今週中に決めたいと思います。"
        calls, _t = self._run(1.12, 1.07, text)
        self.assertEqual(calls, [1.12, 1.12])      # 2文とも作り置きのまま(読む時に作り直さない)


if __name__ == "__main__":
    unittest.main()


class TestGuessVoices(unittest.TestCase):
    """予想訳の作り置き(先読み)が確定訳の合成を待たせない(11/08 speed)。

    見つけた問題(作った日↔英の会話を リモートサーバー で実時間に流した評価): 続き待ちの断片の予想訳ができるたびに
    その声を作っていた。長い文では断片ごとに予想訳が替わり(27字 → 70字 → 確定 150字)、外れた予想の
    合成(0.7秒 + 2.2秒)が合成の錠を先に取り、確定訳の声はその後ろで待った(声 +6〜10秒)。
    """

    def _tts(self, log, delay=0.15):
        def synth(self_, t, spk, speed):
            log.append(("start", t))
            time.sleep(delay)
            log.append(("end", t))
            return b"RIFF" + t.encode()
        return synth

    def test_new_guess_drops_queued_older_guess(self):
        log = []
        t = tts()
        with patch.object(TTS, "_voicevox_synth", self._tts(log)):
            t.prepare("一つ目の予想の訳です。", "ja", guess=True)
            time.sleep(0.03)                       # 1つ目は作り始めている
            t.prepare("二つ目の予想の訳です、まだ続きます。", "ja", guess=True)
            t.prepare("三つ目の予想の訳です、もう少し続きます。", "ja", guess=True)
            time.sleep(0.6)
        started = [x for k, x in log if k == "start"]
        self.assertEqual(started, ["一つ目の予想の訳です。", "三つ目の予想の訳です、もう少し続きます。"])

    def test_final_prepare_goes_before_queued_guess(self):
        log = []
        t = tts()
        with patch.object(TTS, "_voicevox_synth", self._tts(log)):
            t.prepare("予想の訳その一です。", "ja", guess=True)
            time.sleep(0.03)                       # 作り始めている(止められない)
            t.prepare("予想の訳その二です。", "ja", guess=True)
            t.prepare("確定した訳です、こちらを先に。", "ja")
            time.sleep(0.7)
        started = [x for k, x in log if k == "start"]
        self.assertEqual(started[:2], ["予想の訳その一です。", "確定した訳です、こちらを先に。"])

    def test_final_say_goes_before_queued_guess(self):
        log = []
        t = tts()
        with patch.object(TTS, "_voicevox_synth", self._tts(log)), \
             patch.object(TTS, "_play_wav", return_value=True):
            t.prepare("長い予想の訳その一です。", "ja", guess=True)
            time.sleep(0.03)
            t.prepare("長い予想の訳その二です。", "ja", guess=True)
            t.say("確定した訳です。", "ja")       # 作り置きが無い: 行列の先頭で作って鳴らす
            time.sleep(0.4)
        started = [x for k, x in log if k == "start"]
        self.assertEqual(started[:2], ["長い予想の訳その一です。", "確定した訳です。"])

    def test_matching_guess_is_still_used(self):
        log = []
        t = tts()
        with patch.object(TTS, "_voicevox_synth", self._tts(log, 0.02)), \
             patch.object(TTS, "_play_wav", return_value=True):
            t.prepare("当たった予想の訳です。", "ja", guess=True)
            time.sleep(0.1)
            t.say("当たった予想の訳です。", "ja")
        self.assertEqual([x for k, x in log if k == "start"], ["当たった予想の訳です。"])
        self.assertEqual(t.prepared_hits, 1)
