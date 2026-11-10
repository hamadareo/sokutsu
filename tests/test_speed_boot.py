"""「会話を始める」から聞き始めるまでを短くする(11/08 speed)。

見つけた問題(作った会話をアプリに流した評価、remote・耳は起きている状態): 「会話を始める」から
マイクを開けるまで 3.9〜5.1 秒。その半分(1.6〜2.6 秒)は読み上げの声(Kokoro)の読み込みで、
会話のたびに読み直していた(このMacの耳 SenseVoice も会話ごとに読み直し)。起動画面にいる間は
処理場所を起こさず(処理場所を選び直したときだけ)、リモートサーバーのゲートが llama-server を止めていると
「会話を始める」を押してからモデル載せ(1〜2分)を待った。
"""
import threading
import time
import unittest
from unittest import mock

import tsuyaku.__main__ as main_mod
from tsuyaku.config import Settings


class _CountTTS:
    made = 0

    def __init__(self, *a, **k):
        type(self).made += 1
        self.voices = dict(k.get("voices") or {})
        self.speeds = dict(k.get("speeds") or {})
        self.voicevox_url = k.get("voicevox_url", "")
        self.on_start = None


class TestSharedModels(unittest.TestCase):
    def setUp(self):
        main_mod._SHARED.clear()
        _CountTTS.made = 0

    def test_tts_is_loaded_once_and_takes_new_voices(self):
        s = Settings()
        with mock.patch.object(main_mod, "TTS", _CountTTS):
            a = main_mod._shared_tts(s)
            s.tts_voice = dict(s.tts_voice, ja="voicevox:3")
            s.tts_speed = dict(s.tts_speed, ja=1.3)
            b = main_mod._shared_tts(s)
        self.assertIs(a, b)
        self.assertEqual(_CountTTS.made, 1)
        self.assertEqual(b.voices["ja"], "voicevox:3")   # 起動画面で替えた声・速さは次の会話で効く
        self.assertEqual(b.speeds["ja"], 1.3)

    def test_tts_reloads_when_model_dirs_change(self):
        s = Settings()
        with mock.patch.object(main_mod, "TTS", _CountTTS):
            a = main_mod._shared_tts(s)
            s.tts_dir_kokoro = "/other"
            b = main_mod._shared_tts(s)
        self.assertIsNot(a, b)

    def test_local_ear_is_loaded_once(self):
        made = []
        s = Settings()
        with mock.patch.object(main_mod._stt_mod, "make_local_stt", lambda s_: made.append(1) or object()):
            a = main_mod.make_local_stt(s)
            b = main_mod.make_local_stt(s)
        self.assertIs(a, b)
        self.assertEqual(len(made), 1)

    def test_concurrent_preload_and_start_load_once(self):
        made = []

        def slow(s_):
            time.sleep(0.1)
            made.append(1)
            return object()
        s = Settings()
        out = []
        with mock.patch.object(main_mod._stt_mod, "make_local_stt", slow):
            ths = [threading.Thread(target=lambda: out.append(main_mod.make_local_stt(s))) for _ in range(3)]
            for t in ths:
                t.start()
            for t in ths:
                t.join()
        self.assertEqual(len(made), 1)
        self.assertTrue(all(o is out[0] for o in out))

    def test_conversation_uses_shared_tts(self):
        from pathlib import Path
        src = (Path(__file__).resolve().parents[1] / "src/tsuyaku/__main__.py").read_text()
        self.assertIn('tts = self.tts = None if mode == "notes" else _shared_tts(s)', src)


class TestStartScreenWarm(unittest.TestCase):
    """起こすのは「起動画面を開いた・画面を操作した・処理場所を選び直した」から15分まで(リーダー 11/08)。
    会話を終えた後と、窓を隠した・最小化した時は起こさない。会話中は keepalive のまま。"""

    def setUp(self):
        self.calls = []
        self.clock = [1000.0]
        self.hidden = [False]
        self.p = [mock.patch.object(main_mod, "_prewarm_engine", lambda s, **k: self.calls.append(s.engine)),
                  mock.patch.object(main_mod, "_reachable", lambda *a, **k: True),
                  mock.patch.object(main_mod, "save_config", lambda s: None),
                  mock.patch.object(main_mod.time, "monotonic", lambda: self.clock[0]),
                  mock.patch.object(main_mod.Bridge, "_window_hidden", lambda b: self.hidden[0])]
        for x in self.p:
            x.start()

    def tearDown(self):
        for x in self.p:
            x.stop()

    def _bridge(self, engine="remote"):
        b = main_mod.Bridge.__new__(main_mod.Bridge)
        b.s = Settings()
        b.s.engine = engine
        b.running = threading.Event()
        return b

    def _n(self):
        time.sleep(0.03)
        return len(self.calls)

    def test_open_warms_once_then_spaced(self):
        b = self._bridge()
        b.get_engine()
        b.get_engine()
        self.assertEqual(self._n(), 1)     # 続けて開いても PREWARM_EVERY_S に1回

    def test_stops_after_15_minutes_without_operation(self):
        b = self._bridge()
        b.get_engine()                      # 起動画面を開いた
        self.clock[0] += 16 * 60
        b._maybe_prewarm()                  # 置いたまま16分: 起こさない
        self.assertEqual(self._n(), 1)
        b.get_engine()                      # 起動画面を開き直した: また15分
        self.assertEqual(self._n(), 2)

    def test_operation_restarts_15_minutes(self):
        b = self._bridge()
        b.get_engine()
        self.clock[0] += 16 * 60
        b.set_noise("quiet")                # 画面を操作した: また15分起こす
        self.assertEqual(self._n(), 2)
        self.clock[0] += 10 * 60
        b.get_engine()
        self.assertEqual(self._n(), 3)

    def test_engine_change_warms_new_engine(self):
        b = self._bridge("mac")
        b.get_engine()
        b.set_engine("remote")                # 選び直した: 間隔によらず起こす
        self.assertEqual(self.calls[-1:], ["remote"])

    def test_no_warm_after_conversation_end(self):
        b = self._bridge()
        b._back_from_conversation()
        b.get_engine()                      # 会話を終えて起動画面に戻っただけでは起こさない
        self.assertEqual(self._n(), 0)
        b.set_noise("quiet")                # 操作したら起こす
        self.assertEqual(self._n(), 1)

    def test_no_warm_while_hidden_or_minimized(self):
        b = self._bridge()
        self.hidden[0] = True
        b.get_engine()
        b.set_noise("quiet")
        self.assertEqual(self._n(), 0)

    def test_no_warm_during_conversation(self):
        b = self._bridge()
        b.running.set()
        b.get_engine()
        b.set_noise("quiet")
        self.assertEqual(self._n(), 0)

    def test_conversation_end_is_called(self):
        from pathlib import Path
        src = (Path(__file__).resolve().parents[1] / "src/tsuyaku/__main__.py").read_text()
        body = src[src.index("    def _conversation("):src.index("    def _boot_cancelled(")]
        self.assertIn("self._back_from_conversation()", body)


class TestPrewarmEngine(unittest.TestCase):
    def test_prewarm_wakes_remote_ear_too(self):
        woke = []
        s = Settings()
        s.engine = "remote"
        with mock.patch.object(main_mod, "warmup", lambda *a, **k: None), \
             mock.patch.object(main_mod.localllm, "ensure_for", lambda b: None), \
             mock.patch.object(main_mod, "_wake_stt", lambda url: woke.append(url) or True):
            main_mod._prewarm_engine(s, ear=True)
        self.assertEqual(woke, [s.stt_url])

    def test_mac_prewarm_does_not_touch_remote_ear(self):
        woke = []
        s = Settings()
        s.engine = "mac"
        with mock.patch.object(main_mod, "warmup", lambda *a, **k: None), \
             mock.patch.object(main_mod.localllm, "ensure_for", lambda b: None), \
             mock.patch.object(main_mod, "_wake_stt", lambda url: woke.append(url) or True):
            main_mod._prewarm_engine(s, ear=True)
        self.assertEqual(woke, [])


if __name__ == "__main__":
    unittest.main()
