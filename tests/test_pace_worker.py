"""読み上げワーカーの「話のペースに合わせる」と「読み上げ中も聞く」(11/08)。

見つけた問題(実記録、中国語の速い話者 → 日本語の読み上げ): 声は話し終わりから4〜7秒後、9〜17秒遅れた
文は黙って飛ばしていた。読み上げ中はマイクを止めていて、話者の言葉が丸ごと失われていた。
"""
import queue
import threading
import time
import unittest
from unittest.mock import patch

from tsuyaku.__main__ import Bridge
from tsuyaku.pace import PaceController


class FakeTTS:
    def __init__(self, cap=None, device=""):
        self.said = []
        self.on_start = None
        self.last_play_s = 0.0
        self.output_device = device
        self.cap = cap
        self.during = []        # 鳴っている間の (paused, mic_muted, tts_active)

    def say(self, text, lang, speed=None):
        self.said.append((text, lang, speed))
        c = self.cap
        if c is not None:
            self.during.append((c.paused.is_set(), c.mic_muted.is_set(), c.tts_active.is_set()))
        cb, self.on_start = self.on_start, None
        if cb:
            cb()
        self.last_play_s = 0.05
        return True

    def prepare(self, *a, **k):
        pass

    def cancel(self):
        pass


class FakeCap:
    def __init__(self, system=False):
        self.paused = threading.Event()
        self.mic_muted = threading.Event()
        self.tts_active = threading.Event()   # 読み上げ中も聞いている(hear の継ぎ目: メーターが speak_listen)
        self.mix_in = object() if system else None
        self.pauses = 0
        self.muted_seen = False

    def resume_after_tts(self, tail):
        self.pauses += 1
        self.paused.clear()


def run(items, mode="auto", listen="off", system=False, private=False, join_after=2.0,
        device="", default_out="", priority="clear"):
    b = Bridge()
    b.s.voice_on = True
    b.s.tts_speed_mode = {k: mode for k in ("ja", "en", "zh", "ko")}
    b.s.tts_speed = {k: 1.0 for k in ("ja", "en", "zh", "ko")}
    b.s.tts_voice = {"ja": "voicevox:13", "en": "kokoro:21", "zh": "kokoro:45", "ko": "say:Yuna"}
    b.s.listen_while_speaking = listen
    b.s.voice_join_after_s = join_after
    b.s.gate_tail_s = 0.0
    b.s.tts_speed_priority = priority
    b.capture = FakeCap(system)
    b.tts = FakeTTS(b.capture, device)
    b.pace = PaceController()
    b.running.set()
    b.ui = queue.Queue()
    for it in items:
        b.tts_q.put(it)
    with patch("tsuyaku.devices.output_is_private", return_value=private), \
            patch("tsuyaku.devices.default_name", return_value=default_out):
        th = threading.Thread(target=b._tts_worker, daemon=True)
        th.start()
        t0 = time.time()
        while not b.tts_q.empty() and time.time() - t0 < 3:
            time.sleep(0.02)
        time.sleep(0.15)
        b.running.clear()
        th.join(2)
    ev = []
    while not b.ui.empty():
        ev.append(b.ui.get())
    return b, ev


class TestAutoSpeed(unittest.TestCase):
    def test_fast_speaker_gets_faster_voice(self):
        now = time.time()
        # 24字の訳、原文は1.5秒 → 速くする(VOICEVOX の上限 ×1.25 まで)
        b, ev = run([("あ" * 24, "ja", 1, now - 0.5, None, 1.5)])
        self.assertEqual(len(b.tts.said), 1)
        self.assertGreater(b.tts.said[0][2], 1.0)
        self.assertLessEqual(b.tts.said[0][2], 1.25)
        shown = [e for e in ev if e[0] == "voice" and "speed" in e[1]]
        self.assertTrue(shown and shown[0][1]["auto"])

    def test_fixed_mode_keeps_user_speed(self):
        b, _ = run([("あ" * 24, "ja", 1, time.time() - 0.5, None, 1.5)], mode="fixed")
        self.assertIsNone(b.tts.said[0][2])

    def test_behind_joins_queued_items(self):
        now = time.time()
        b, ev = run([("一つ目です。", "ja", 1, now - 3.0, None, 1.0),
                     ("二つ目です。", "ja", 2, now - 2.5, None, 1.0),
                     ("三つ目です。", "ja", 3, now - 2.0, None, 1.0)])
        self.assertEqual(len(b.tts.said), 1)
        self.assertEqual(b.tts.said[0][0], "一つ目です。二つ目です。三つ目です。")

    def test_speed_priority_allows_1_5(self):
        """「速さ優先」なら VOICEVOX も ×1.5 まで(既定の「聞きやすさ優先」は ×1.25)。"""
        now = time.time()
        b, _ = run([("あ" * 40, "ja", 1, now - 0.5, None, 1.5)], priority="fast")
        self.assertGreater(b.tts.said[0][2], 1.25)
        self.assertLessEqual(b.tts.said[0][2], 1.5)

    def test_not_behind_reads_separately(self):
        now = time.time()
        b, _ = run([("一つ目です。", "ja", 1, now - 0.5, None, 1.0),
                    ("二つ目です。", "ja", 2, now - 0.4, None, 1.0)])
        self.assertEqual(len(b.tts.said), 2)

    def test_skip_is_shown_not_silent(self):
        now = time.time()
        b, ev = run([("古い訳です。", "ja", 7, now - 20.0, None, 1.0)], mode="fixed")
        self.assertEqual(b.tts.said, [])
        self.assertIn(("voice", {"id": 7, "skipped": True}), ev)


class TestListenWhileSpeaking(unittest.TestCase):
    def item(self):
        return [("こんにちは。", "ja", 1, time.time() - 0.3, None, 1.0)]

    def test_off_pauses_mic(self):
        b, _ = run(self.item(), listen="off")
        self.assertEqual(b.capture.pauses, 1)

    def test_auto_with_headphones_keeps_listening(self):
        b, _ = run(self.item(), listen="auto", private=True)
        self.assertEqual(b.capture.pauses, 0)
        self.assertFalse(b.capture.mic_muted.is_set())

    def test_auto_with_system_audio_mutes_only_mic(self):
        b = None
        seen = []
        orig = FakeTTS.say

        def say(self, text, lang, speed=None, _c=seen):
            _c.append(bb[0].capture.mic_muted.is_set())
            return orig(self, text, lang, speed)
        bb = [None]
        with patch.object(FakeTTS, "say", say):
            # run() を少し崩して、say の最中に mic_muted が立っているかを見る
            bobj = Bridge()
            bb[0] = bobj
            bobj.s.voice_on = True
            bobj.s.tts_speed_mode = {"ja": "fixed"}
            bobj.s.listen_while_speaking = "auto"
            bobj.s.gate_tail_s = 0.0
            bobj.tts = FakeTTS()
            bobj.capture = FakeCap(system=True)
            bobj.pace = PaceController()
            bobj.ui = queue.Queue()
            bobj.running.set()
            bobj.tts_q.put(self.item()[0])
            with patch("tsuyaku.devices.output_is_private", return_value=False):
                th = threading.Thread(target=bobj._tts_worker, daemon=True)
                th.start()
                time.sleep(0.4)
                bobj.running.clear()
                th.join(2)
        self.assertEqual(seen, [True])                     # 読み上げの最中はマイクだけ止める
        self.assertFalse(bobj.capture.mic_muted.is_set())  # 終わったら戻す
        self.assertEqual(bobj.capture.pauses, 0)            # この Mac の音は止めない

    def test_auto_with_speakers_and_mic_only_pauses(self):
        b, _ = run(self.item(), listen="auto", private=False, system=False)
        self.assertEqual(b.capture.pauses, 1)

    def test_always_keeps_listening(self):
        b, _ = run(self.item(), listen="always", private=False)
        self.assertEqual(b.capture.pauses, 0)


    def speak_opts(self, ev):
        return [e[3] if len(e) > 3 else None for e in ev if e[0] == "status" and e[1] == "speak"]

    def test_keep_listening_sets_tts_active_not_paused(self):
        """聞き続ける時は paused ではなく tts_active(メーター「読み上げ中 — 聞いています」)。"""
        b, ev = run(self.item(), listen="always")
        self.assertEqual(b.tts.during, [(False, False, True)])
        self.assertFalse(b.capture.tts_active.is_set())          # 終わったら戻す
        self.assertEqual(self.speak_opts(ev), [{"listen": "mic"}])  # 下の帯: 「マイク休止中」を出さない

    def test_system_audio_mode_marks_listen_system(self):
        b, ev = run(self.item(), listen="auto", system=True, private=False)
        self.assertEqual(b.tts.during, [(False, True, True)])
        self.assertEqual(self.speak_opts(ev), [{"listen": "system"}])

    def test_pause_mode_keeps_old_status(self):
        b, ev = run(self.item(), listen="off")
        self.assertEqual(b.tts.during, [(True, False, False)])
        self.assertEqual(self.speak_opts(ev), [None])

    def test_unknown_output_with_headset_name_keeps_listening(self):
        """USB など分からない出力は止める。ただし名前がヘッドセット・ヘッドホンなら聞き続ける。"""
        b, _ = run(self.item(), listen="auto", private=None, device="Logi USB Headset")
        self.assertEqual(b.capture.pauses, 0)
        b, _ = run(self.item(), listen="auto", private=None, device="", default_out="USB ヘッドセット")
        self.assertEqual(b.capture.pauses, 0)
        b, _ = run(self.item(), listen="auto", private=None, device="USB Audio Speaker")
        self.assertEqual(b.capture.pauses, 1)


class TestSettings(unittest.TestCase):
    def test_existing_user_with_explicit_speed_stays_fixed(self):
        import json
        import tempfile
        from pathlib import Path
        from tsuyaku import config as C
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "config.json"
            p.write_text(json.dumps({"tts_speed": {"ja": 1.2, "en": 1.0, "zh": 1.0, "ko": 1.0}}))
            with patch.object(C, "_config_file", return_value=p):
                s = C.load_settings()
            self.assertEqual(s.tts_speed_mode["ja"], "fixed")
            self.assertEqual(s.tts_speed["ja"], 1.2)
            p.write_text(json.dumps({}))
            with patch.object(C, "_config_file", return_value=p):
                s = C.load_settings()
            self.assertEqual(s.tts_speed_mode["ja"], "auto")    # 新しく使う人は自動
            self.assertEqual(s.listen_while_speaking, "auto")
            self.assertEqual(s.tts_speed_priority, "clear")      # 既定は聞きやすさ優先
            p.write_text(json.dumps({"tts_speed_priority": "nonsense"}))
            with patch.object(C, "_config_file", return_value=p):
                s = C.load_settings()
            self.assertEqual(s.tts_speed_priority, "clear")


if __name__ == "__main__":
    unittest.main()
