"""聞こえの見える化と入出力デバイスの切り替え(11/08 hear)の Python 側。

ユーザー「聞こえているのかUIですぐにわかるようにしてください … ソフト上で入力・出力デバイスを
切り替えられるようにしてください」。
"""
from __future__ import annotations

import json
import queue
import tempfile
import threading
import time
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

import tsuyaku.__main__ as main_mod
import tsuyaku.audio as audio_mod
from tsuyaku import devices, tts as tts_mod
from tsuyaku.__main__ import Bridge, HearThrottle, hear_state
from tsuyaku.audio import MicTest, SpeechCapture, meter_of
from tsuyaku.config import Settings, save_config

ROOT = Path(__file__).resolve().parents[1]
VAD = str(ROOT / "models" / "silero_vad.onnx")

INS = [{"name": "MacBook Air のマイク", "uid": "BuiltInMic", "isDefault": True},
       {"name": "USB マイク", "uid": "USBMIC", "isDefault": False}]
OUTS = [{"name": "MacBook Air のスピーカー", "uid": "BuiltInSpeaker", "isDefault": True},
        {"name": "AirPods", "uid": "AIRPODS-UID", "isDefault": False}]


def fake_list(ins=INS, outs=OUTS):
    return lambda kind, *a, **k: [dict(d) for d in (ins if kind == devices.INPUT else outs)]


def _wait(cond, timeout=2.0):
    t = time.time() + timeout
    while time.time() < t:
        if cond():
            return True
        time.sleep(0.01)
    return cond()


class FakeSD:
    """sounddevice の偽物: 開いたストリーム(device 引数)を覚え、callback を叩ける。"""
    def __init__(self, names):
        self.names = names
        self.opened: list = []
        self.closed = 0
        self.cb = None
        self.reinit = 0
        sd = self

        class Stream:
            def __init__(self, **kw):
                sd.opened.append(kw.get("device"))
                sd.cb = kw["callback"]

            def __enter__(self):
                return self

            def __exit__(self, *a):
                sd.closed += 1
                return False
        self.InputStream = Stream

    def query_devices(self):
        return [{"name": n, "max_input_channels": 1, "max_output_channels": 0} for n in self.names]

    def _terminate(self):
        self.reinit += 1

    def _initialize(self):
        pass


# ---------------------------------------------------------------------------
class TestPersistByName(unittest.TestCase):
    def test_settings_have_device_names_and_glow(self):
        s = Settings()
        self.assertEqual(s.input_device, "")       # "" = システムの設定に合わせる
        self.assertEqual(s.output_device, "")
        self.assertTrue(s.edge_glow)

    def test_saved_by_name_in_config(self):
        with tempfile.TemporaryDirectory() as d:
            with mock.patch("tsuyaku.config.ROOT", Path(d)):
                s = Settings()
                s.input_device = "USB マイク"
                s.output_device = "AirPods"
                save_config(s)
                data = json.loads((Path(d) / "config.json").read_text())
        self.assertEqual(data["input_device"], "USB マイク")   # 番号ではなく名前
        self.assertEqual(data["output_device"], "AirPods")

    def test_bridge_set_devices_saves_names(self):
        b = _bridge()
        saved = []
        with mock.patch.object(main_mod, "save_config", lambda s: saved.append(
                (s.input_device, s.output_device))), \
                mock.patch.object(devices, "list_devices", fake_list()):
            r1 = b.set_input_device("USB マイク")
            r2 = b.set_output_device("AirPods")
        self.assertEqual(saved[-1], ("USB マイク", "AirPods"))
        self.assertEqual(r1["applied"], "next")     # 会話も確認も動いていない: 次から
        self.assertEqual(r2["applied"], "next")     # 読み上げは次の読み上げから

    def test_audio_devices_lists_with_system_first_info(self):
        b = _bridge()
        b.s.input_device = "古いマイク"            # 今はつながっていない
        with mock.patch.object(devices, "list_devices", fake_list()):
            r = b.audio_devices()
        self.assertEqual([d["name"] for d in r["inputs"]], ["MacBook Air のマイク", "USB マイク"])
        self.assertEqual(r["defaultInput"], "MacBook Air のマイク")
        self.assertEqual(r["defaultOutput"], "MacBook Air のスピーカー")
        self.assertTrue(r["inputMissing"])
        self.assertFalse(r["outputMissing"])
        self.assertEqual(r["input"], "古いマイク")


class TestFallbackWhenMissing(unittest.TestCase):
    def test_resolve(self):
        with mock.patch.object(devices, "list_devices", fake_list()):
            self.assertEqual(devices.resolve(devices.INPUT, "USB マイク"), ("USB マイク", False))
            self.assertEqual(devices.resolve(devices.INPUT, "AirPods のマイク"),
                             ("MacBook Air のマイク", True))
            self.assertEqual(devices.resolve(devices.INPUT, ""), ("MacBook Air のマイク", False))

    def test_pa_index_reinits_only_when_no_stream_open(self):
        sd = FakeSD(["MacBook Air のマイク"])
        self.assertIsNone(devices.pa_index(sd, devices.INPUT, "USB マイク"))
        self.assertEqual(sd.reinit, 1)              # 後からつないだ: 開いていなければ初期化し直す
        devices.stream_opened()
        try:
            devices.pa_index(sd, devices.INPUT, "USB マイク")
            self.assertEqual(sd.reinit, 1)          # 開いている最中はやり直さない(壊れる)
        finally:
            devices.stream_closed()
        self.assertEqual(devices.pa_index(sd, devices.INPUT, "MacBook Air のマイク"), 0)

    def test_capture_opens_default_and_notes_missing(self):
        sd = FakeSD(["MacBook Air のマイク", "USB マイク"])
        notes = []
        with mock.patch.object(audio_mod, "sd", sd), \
                mock.patch.object(devices, "list_devices", fake_list()):
            c = SpeechCapture(VAD, queue.Queue())
            c.device = "AirPods のマイク"
            c.on_device = lambda *a: notes.append(a)
            c.start()
            self.assertTrue(_wait(lambda: sd.opened))
            c.stop()
        self.assertEqual(sd.opened[0], 0)           # システムの既定(MacBook Air)を開いた
        self.assertEqual(c.device_used, "MacBook Air のマイク")
        self.assertIn(("missing", "AirPods のマイク", "MacBook Air のマイク"), notes)


class TestSwitchDuringConversation(unittest.TestCase):
    def test_switch_restarts_only_the_stream(self):
        sd = FakeSD(["MacBook Air のマイク", "USB マイク"])
        notes = []
        q: queue.Queue = queue.Queue()
        with mock.patch.object(audio_mod, "sd", sd), \
                mock.patch.object(devices, "list_devices", fake_list()):
            c = SpeechCapture(VAD, q)
            c.on_device = lambda *a: notes.append(a)
            c.start()
            self.assertTrue(_wait(lambda: len(sd.opened) == 1))
            seg, vad = c.segmenter, c.vad
            cb0 = sd.cb
            cb0(np.full((512, 1), 0.01, np.float32), 512, None, None)
            c.switch_device("USB マイク")
            self.assertTrue(_wait(lambda: len(sd.opened) == 2))
            c.stop()
        self.assertEqual(sd.opened, [0, 1])         # 既定 → USB マイク
        self.assertGreaterEqual(sd.closed, 1)       # 前のストリームは閉じた
        self.assertIs(c.segmenter, seg)             # 区切り・VAD・行列はそのまま(会議は続く)
        self.assertIs(c.vad, vad)
        self.assertIs(c.out, q)
        self.assertEqual(c.device_used, "USB マイク")
        self.assertIn(("switched", "USB マイク", "USB マイク"), notes)

    def test_bridge_switches_live_capture_without_ending(self):
        b = _bridge()
        b.running.set()
        calls = []
        b.capture = types.SimpleNamespace(switch_device=lambda n: calls.append(n),
                                          paused=threading.Event())
        with mock.patch.object(main_mod, "save_config", lambda s: None), \
                mock.patch.object(devices, "list_devices", fake_list()):
            r = b.set_input_device("USB マイク")
        self.assertEqual(calls, ["USB マイク"])
        self.assertEqual(r["applied"], "now")
        self.assertTrue(b.running.is_set())         # 会議は終わらない

    def test_output_switch_reaches_running_tts(self):
        b = _bridge()
        b.tts = types.SimpleNamespace(output_device="")
        with mock.patch.object(main_mod, "save_config", lambda s: None):
            b.set_output_device("AirPods")
        self.assertEqual(b.tts.output_device, "AirPods")
        ev = [e for e in list(b.ui.queue) if e[0] == "audio_dev"]
        self.assertEqual(ev[-1][1]["applied"], "next")


class TestLostDevice(unittest.TestCase):
    def test_stall_with_device_gone_falls_back_and_keeps_running(self):
        sd = FakeSD(["MacBook Air のマイク", "USB マイク"])
        notes = []
        present = {"ins": list(INS)}
        with mock.patch.object(audio_mod, "sd", sd), \
                mock.patch.object(devices, "list_devices",
                                  lambda kind, *a, **k: [dict(d) for d in (
                                      present["ins"] if kind == devices.INPUT else OUTS)]):
            c = SpeechCapture(VAD, queue.Queue())
            c.device = "USB マイク"
            c.stall_s = 0.3
            c.on_device = lambda *a: notes.append(a)
            c.start()
            self.assertTrue(_wait(lambda: sd.opened == [1]))
            present["ins"] = [INS[0]]               # USB マイクを抜いた(コールバックが止まる)
            ok = _wait(lambda: len(sd.opened) >= 2, 3.0)
            c.stop()
        self.assertTrue(ok, "外れたマイクから既定へ開き直さない")
        self.assertEqual(sd.opened[1], 0)
        self.assertTrue(any(n[0] == "lost" and n[1] == "USB マイク" for n in notes), notes)


class TestLevelPushes(unittest.TestCase):
    def test_meter_mapping(self):
        self.assertEqual(meter_of(0.0), 0.0)
        self.assertAlmostEqual(meter_of(10 ** (-10 / 20)), 1.0, places=3)   # -10 dBFS = 満
        self.assertGreater(meter_of(0.03), 0.4)                             # 普通の声
        self.assertLess(meter_of(0.001), 0.01)                              # -60 dBFS = 0

    def test_throttle_at_most_16hz(self):
        th = HearThrottle()
        t0 = 1000.0
        n = sum(th.due(t0 + i * 0.005, (i % 10) / 10, "speech") for i in range(200))   # 1秒に200回
        self.assertLessEqual(n, 17)
        self.assertGreaterEqual(n, 14)

    def test_throttle_skips_unchanged_but_keeps_alive(self):
        th = HearThrottle()
        t0 = 2000.0
        n = sum(th.due(t0 + i * 0.05, 0.3, "quiet") for i in range(60))     # 3秒同じ値
        self.assertLessEqual(n, 4)                  # 1秒に1回だけ
        self.assertTrue(th.due(t0 + 3.1, 0.3, "speech"))   # 状態が変われば即

    def test_pump_pushes_hear_events_throttled(self):
        b = _bridge()
        cap = types.SimpleNamespace(paused=threading.Event(), level=0.2, health="ok",
                                    speech=True, device_used="USB マイク",
                                    take_meter=lambda: 0.5)
        b.capture = cap
        th = threading.Thread(target=b._pump, daemon=True)
        th.start()
        time.sleep(0.5)
        b.closing.set()
        th.join(1)
        hears = [j for j in b.window.js if "TSUYAKU.hear(" in j]
        self.assertTrue(hears)
        self.assertLessEqual(len(hears), 3)          # 値も状態も同じ: 0.5秒で何度も送らない
        obj = json.loads(hears[0][hears[0].index("TSUYAKU.hear(") + 13:-1])
        self.assertEqual(obj["st"], "speech")
        self.assertEqual(obj["mic"], "USB マイク")


class TestStates(unittest.TestCase):
    def _cap(self, **kw):
        d = dict(paused=threading.Event(), health="ok", speech=False)
        d.update(kw)
        return types.SimpleNamespace(**d)

    def test_states(self):
        now = 100.0
        self.assertEqual(hear_state(self._cap(), ("listen", 0), now), "quiet")
        self.assertEqual(hear_state(self._cap(speech=True), ("listen", 0), now), "speech")
        self.assertEqual(hear_state(self._cap(), ("work", now - 1), now), "work")
        p = threading.Event()
        p.set()
        self.assertEqual(hear_state(self._cap(paused=p), ("listen", 0), now), "speak")
        for h in ("zeros", "stall", "fail"):
            self.assertEqual(hear_state(self._cap(health=h, speech=True), ("listen", 0), now), "nomic")
        self.assertEqual(hear_state(self._cap(), ("listen", 0), now, lost_until=now + 2), "lost")

    def test_all_zero_input_shows_warning_state(self):
        sd = FakeSD(["MacBook Air のマイク"])
        with mock.patch.object(audio_mod, "sd", sd), \
                mock.patch.object(devices, "list_devices", fake_list()):
            c = SpeechCapture(VAD, queue.Queue())
            c.on_error = lambda *a: None
            c.start()
            self.assertTrue(_wait(lambda: sd.cb is not None))
            for _ in range(int(3.2 * 16000 / 512)):
                sd.cb(np.zeros((512, 1), np.float32), 512, None, None)
            st = hear_state(c, ("listen", 0), time.time())
            sd.cb(np.full((512, 1), 1e-4, np.float32), 512, None, None)   # 音が戻った
            st2 = hear_state(c, ("listen", 0), time.time())
            c.stop()
        self.assertEqual(st, "nomic")
        self.assertEqual(st2, "quiet")


class TestMicTest(unittest.TestCase):
    def test_mic_test_runs_only_until_stopped_and_releases(self):
        sd = FakeSD(["MacBook Air のマイク", "USB マイク"])
        with mock.patch.object(audio_mod, "sd", sd), \
                mock.patch.object(devices, "list_devices", fake_list()):
            m = MicTest("USB マイク")
            m.start()
            self.assertTrue(_wait(lambda: sd.cb is not None))
            sd.cb(np.full((512, 1), 0.05, np.float32), 512, None, None)
            self.assertGreater(m.take_meter(), 0.4)
            m.stop()
        self.assertEqual(sd.opened, [1])
        self.assertEqual(sd.closed, 1)               # 離れたらデバイスを放す

    def test_bridge_mic_test_busy_during_conversation(self):
        b = _bridge()
        b.running.set()
        self.assertEqual(b.mic_test(True), "busy")

    def test_start_stops_mic_test(self):
        b = _bridge()
        stopped = []
        b._mic_test = types.SimpleNamespace(stop=lambda: stopped.append(1))
        with mock.patch.object(main_mod.threading, "Thread"), \
                mock.patch.object(main_mod, "save_config", lambda s: None), \
                mock.patch.object(main_mod.inmeeting, "load_terms", lambda: []):
            b.start("ja-en", "")
        self.assertEqual(stopped, [1])
        self.assertIsNone(b._mic_test)


# ---------------------------------------------------------------------------
class TestOutputPaths(unittest.TestCase):
    def test_say_cmd_device(self):
        self.assertEqual(tts_mod.say_cmd("Kyoko", 1.0, device="AirPods"),
                         ["say", "-v", "Kyoko", "-a", "AirPods"])
        self.assertNotIn("-a", tts_mod.say_cmd("Kyoko", 1.0))

    def _tts(self):
        t = tts_mod.TTS("", "", {}, voicevox_url="")
        t.output_device = "AirPods"
        return t

    def test_say_macos_gets_device(self):
        t = self._tts()
        seen = []

        class P:
            def __init__(self, cmd, **kw):
                seen.append(cmd)

            def poll(self):
                return 0
        with mock.patch.object(tts_mod.subprocess, "Popen", P), \
                mock.patch.object(devices, "list_devices", fake_list()):
            t._say_macos("こんにちは", "Kyoko", 1.0)
        self.assertIn("-a", seen[0])
        self.assertEqual(seen[0][seen[0].index("-a") + 1], "AirPods")

    def test_wav_paths_play_on_device(self):
        """VOICEVOX・Kokoro・ko(vits) の音は全部 _play_wav を通る → 選んだデバイスの UID で鳴らす。"""
        t = self._tts()
        played = []
        t._play_on_device = lambda path, uid: played.append(uid) or True
        with mock.patch.object(devices, "list_devices", fake_list()), \
                mock.patch.object(tts_mod.subprocess, "Popen") as po:
            for kind in ("voicevox:13", "kokoro:3"):
                t.set_voice("ja", kind)
                t._synth = lambda text, lang, speed=None: tts_mod._to_wav(np.zeros(1600, np.float32), 16000)
                t.say("こんにちは。", "ja")
        self.assertEqual(played, ["AIRPODS-UID", "AIRPODS-UID"])
        po.assert_not_called()                       # afplay(既定の出力)は使っていない

    def test_missing_output_falls_back_to_default_and_notes(self):
        t = self._tts()
        notes = []
        t.on_device = lambda *a: notes.append(a)
        t._play_on_device = lambda path, uid: self.fail("無いデバイスで鳴らそうとした")
        seen = []

        class P:
            def __init__(self, cmd, **kw):
                seen.append(cmd)

            def poll(self):
                return 0
        with mock.patch.object(devices, "list_devices", fake_list(outs=OUTS[:1])), \
                mock.patch.object(tts_mod.subprocess, "Popen", P):
            t._play_wav(tts_mod._to_wav(np.zeros(1600, np.float32), 16000))
            t._say_macos("hello", "Alex", 1.0)
            t._play_wav(tts_mod._to_wav(np.zeros(1600, np.float32), 16000))
        self.assertEqual(seen[0][0], "afplay")       # システムの既定で鳴らす
        self.assertNotIn("-a", seen[1])
        self.assertEqual(len(notes), 1)              # 知らせは1回
        self.assertEqual(notes[0][0], "missing")

    def test_device_player_failure_falls_back_to_afplay(self):
        t = self._tts()
        t._play_on_device = lambda path, uid: False   # AVAudioPlayer が鳴らせなかった
        seen = []

        class P:
            def __init__(self, cmd, **kw):
                seen.append(cmd)

            def poll(self):
                return 0
        with mock.patch.object(devices, "list_devices", fake_list()), \
                mock.patch.object(tts_mod.subprocess, "Popen", P):
            self.assertTrue(t._play_wav(tts_mod._to_wav(np.zeros(1600, np.float32), 16000)))
        self.assertEqual(seen[0][0], "afplay")

    def test_system_default_keeps_afplay(self):
        t = self._tts()
        t.output_device = ""
        seen = []

        class P:
            def __init__(self, cmd, **kw):
                seen.append(cmd)

            def poll(self):
                return 0
        t._play_on_device = lambda path, uid: self.fail("既定なのに別の道")
        with mock.patch.object(tts_mod.subprocess, "Popen", P):
            t._play_wav(tts_mod._to_wav(np.zeros(1600, np.float32), 16000))
        self.assertEqual(seen[0][0], "afplay")

    def test_conversation_and_preview_tts_get_output_device(self):
        src = (ROOT / "src/tsuyaku/__main__.py").read_text()
        body = src[src.index("    def _conversation("):src.index("    def _stt_worker(")]
        self.assertIn("output_device", body)
        prev = src[src.index("    def preview_tts("):src.index("    def end_meeting(")]
        self.assertIn("output_device", prev)


class TestMisc(unittest.TestCase):
    def test_glow_setting(self):
        b = _bridge()
        with mock.patch.object(main_mod, "save_config", lambda s: None):
            b.set_edge_glow(False)
        self.assertFalse(b.s.edge_glow)
        with mock.patch.object(main_mod, "_say_voices", lambda: {}), \
                mock.patch.object(main_mod, "_voicevox_list", lambda u: []):
            g = b.get_tts()
        self.assertFalse(g["edgeGlow"])
        self.assertIn("inputDevice", g)
        self.assertIn("outputDevice", g)

    def test_open_mic_privacy(self):
        b = _bridge()
        with mock.patch.object(main_mod.subprocess, "run") as run:
            self.assertEqual(b.open_mic_privacy(), "ok")
        self.assertIn("x-apple.systempreferences:com.apple.preference.security?Privacy_Microphone",
                      run.call_args[0][0])


class TestSystemAudioWithChosenMic(unittest.TestCase):
    def test_mix_in_still_applied_on_non_default_mic(self):
        """「マイク+この Mac の音」: 選んだマイク(既定でない)でも、この Mac の音が足される。"""
        sd = FakeSD(["MacBook Air のマイク", "USB マイク"])
        seen = []
        with mock.patch.object(audio_mod, "sd", sd), \
                mock.patch.object(devices, "list_devices", fake_list()):
            c = SpeechCapture(VAD, queue.Queue())
            c.device = "USB マイク"
            c.mix_in = types.SimpleNamespace(mix=lambda m: m + 0.25, stop=lambda: None)
            c.on_chunk = lambda mono: seen.append(float(mono[0]))
            c.start()
            self.assertTrue(_wait(lambda: sd.cb is not None))
            sd.cb(np.full((512, 1), 0.01, np.float32), 512, None, None)
            c.stop()
        self.assertEqual(sd.opened, [1])
        self.assertAlmostEqual(seen[0], 0.26, places=5)


class TestSeamsForPace(unittest.TestCase):
    """engloop(pace: 読み上げ中も聞く・ヘッドホンの判定・エコー消しの入力)との継ぎ目。"""

    def test_classify_output(self):
        c = devices.classify_output
        self.assertTrue(c("bltn", "hdpn"))        # 内蔵の出力のヘッドホン端子
        self.assertFalse(c("bltn", "ispk"))       # 内蔵スピーカー
        self.assertIsNone(c("bltn", ""))
        self.assertTrue(c("blue", ""))            # Bluetooth(AirPods など)
        self.assertTrue(c("blea", ""))
        for tr in ("hdmi", "dprt", "airp"):
            self.assertFalse(c(tr, ""), tr)       # 部屋に鳴る
        for tr in ("usb ", "virt", "grup", ""):
            self.assertIsNone(c(tr, ""), tr)      # 分からない

    def test_output_is_private_uses_named_or_default_device(self):
        devs = [{"id": 10, "name": "MacBook Air のスピーカー", "uid": "B", "inputs": 0, "outputs": 2},
                {"id": 11, "name": "AirPods", "uid": "A", "inputs": 1, "outputs": 2}]
        props = {(10, "tran"): "bltn", (10, "ssrc"): "ispk", (11, "tran"): "blue"}
        with mock.patch.object(devices, "coreaudio_devices", lambda: devs), \
                mock.patch.object(devices, "_default_id", lambda kind: 10), \
                mock.patch.object(devices, "_prop4", lambda d, sel, scope=None: props.get((d, sel), "")):
            self.assertFalse(devices.output_is_private(""))          # 既定 = 内蔵スピーカー
            self.assertTrue(devices.output_is_private("AirPods"))
            self.assertIsNone(devices.output_is_private("無い機器"))

    def test_input_opener_can_be_replaced(self):
        """エコー消しの入力(VoiceProcessingIO)は _open_input を上書きして同じ cb を呼べばよい。"""
        seen, chunks = [], []

        class VPIO(SpeechCapture):
            def _open_input(self, kw):
                seen.append(kw)

                class S:
                    def __enter__(s):
                        kw["callback"](np.full((512, 1), 0.02, np.float32), 512, None, None)
                        return s

                    def __exit__(s, *a):
                        return False
                return S()
        with mock.patch.object(devices, "list_devices", fake_list()):
            c = VPIO(VAD, queue.Queue())
            c.on_chunk = lambda m: chunks.append(len(m))
            c.start()
            self.assertTrue(_wait(lambda: chunks))
            c.stop()
        self.assertEqual(seen[0]["samplerate"], 16000)
        self.assertEqual(seen[0]["channels"], 1)
        self.assertEqual(chunks[0], 512)

    def test_speak_while_listening_state(self):
        cap = types.SimpleNamespace(paused=threading.Event(), health="ok", speech=True,
                                    tts_active=threading.Event())
        cap.tts_active.set()
        self.assertEqual(hear_state(cap, ("listen", 0), 1.0), "speak_listen")
        cap.paused.set()                      # 今までどおり止めて読むなら「マイク休止」
        self.assertEqual(hear_state(cap, ("listen", 0), 1.0), "speak")

    def test_speak_listen_keeps_level(self):
        b = _bridge()
        ta = threading.Event()
        ta.set()
        b.capture = types.SimpleNamespace(paused=threading.Event(), level=0.2, health="ok",
                                          speech=True, tts_active=ta, device_used="",
                                          take_meter=lambda: 0.6)
        b._pump_hear(b.capture, 100.0)
        js = [j for j in b.window.js if "TSUYAKU.hear(" in j][-1]
        obj = json.loads(js[js.index("TSUYAKU.hear(") + 13:-1])
        self.assertEqual(obj["st"], "speak_listen")
        self.assertGreater(obj["lv"], 0.5)     # 聞いているのでメーターは動く


class TestMenubarDevices(unittest.TestCase):
    """会話 > 入力 / 出力: 開くたびに今のデバイスで作り直し、選んだものに印。押すと Bridge と同じ処理。"""

    def test_submenus_exist_in_four_languages(self):
        from tsuyaku import menubar as M
        ids = [it["id"] for it in M._walk_items(dict((k, v) for k, _t, v in M.MENUS)["conv"])]
        self.assertIn("inputDev", ids)
        self.assertIn("outputDev", ids)
        with mock.patch.object(devices, "list_devices", fake_list()):
            menu, _refs = M.build_menu("ja")
            text = M.dump(menu)
        self.assertIn("入力", text)
        self.assertIn("出力", text)

    def test_fill_lists_devices_and_checks_current(self):
        from AppKit import NSMenu
        from tsuyaku import menubar as M
        b = _bridge()
        b.s.input_device = "USB マイク"
        mb = M.MenuBar(None, "ja")
        mb.bridge = b
        sub = NSMenu.alloc().initWithTitle_("入力")
        with mock.patch.object(devices, "list_devices", fake_list()):
            mb.fill_devices(sub, "input")
            text = M.dump(sub)
        lines = [x.strip() for x in text.splitlines()]
        self.assertTrue(lines[0].startswith("システムの設定に合わせる"), lines)
        self.assertIn("✓ USB マイク", lines)
        self.assertIn("MacBook Air のマイク", lines)

    def test_choosing_calls_bridge(self):
        from tsuyaku import menubar as M
        b = _bridge()
        got = []
        b.set_input_device = lambda n: got.append(("in", n)) or {"ok": True, "applied": "next"}
        b.set_output_device = lambda n: got.append(("out", n)) or {"ok": True, "applied": "next"}
        mb = M.MenuBar(None, "ja")
        mb.bridge = b
        mb.run("dev:input:USB マイク")
        mb.run("dev:output:")
        self.assertTrue(_wait(lambda: len(got) == 2))
        self.assertEqual(sorted(got), [("in", "USB マイク"), ("out", "")])


class _Win:
    def __init__(self):
        self.js: list[str] = []

    def evaluate_js(self, js):
        self.js.append(js)


def _bridge() -> Bridge:
    b = Bridge.__new__(Bridge)
    b.s = Settings()
    b.window = _Win()
    b.ui = queue.Queue()
    b.running = threading.Event()
    b.closing = threading.Event()
    b.interrupting = threading.Event()
    b.capture = b.tts = b.log = None
    b._preview_tts = None
    b._mic_test = None
    return b


if __name__ == "__main__":
    unittest.main()
