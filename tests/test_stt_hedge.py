"""リモートサーバーの耳が回線で遅れたら、このMacの速い耳(SenseVoice)の結果を使う(11/06)。

見つけた問題(11/06 実会話 13:03〜): リモートサーバーの耳はサーバー内では0.13〜0.5秒なのに、
往復は0.3〜7.4秒(この Mac の回線が混んでいて ping が平均0.38秒・最大0.9秒)。
1発話の認識に1.2〜4秒かかり、4秒の ReadTimeout のあと Mac に落ちる発話もあった。
変更: リモートサーバー に投げたまま hedge_s 秒だけ待ち、来なければこのMacの耳で聞く。
回線が良いとき(往復0.14〜0.2秒)は今までどおり リモートサーバーの結果だけを使う(品質は同じ)。
"""
import threading
import time
import unittest

import numpy as np

from tsuyaku.stt import RemoteSTT


class FastLocal:
    fast = True

    def __init__(self):
        self.calls = 0

    def transcribe(self, samples):
        self.calls += 1
        return "ローカル", "ja", 1.0


class Resp:
    def __init__(self, obj):
        self.obj = obj

    def raise_for_status(self):
        pass

    def json(self):
        return self.obj


def slow_post(delay, text="リモート"):
    def post(url, data=None, timeout=None):
        time.sleep(delay)
        return Resp({"text": text, "lang": "ja", "prob": 0.9})
    return post


SAMPLES = np.zeros(16000, np.float32)


class TestHedge(unittest.TestCase):
    def test_fast_remote_is_used_and_local_not_called(self):
        loc = FastLocal()
        st = RemoteSTT("http://x", loc, post=slow_post(0.02), hedge_s=0.3)
        self.assertEqual(st.transcribe(SAMPLES)[0], "リモート")
        self.assertEqual(st.last_where, "remote")
        self.assertEqual(loc.calls, 0)          # 回線が良ければ今までと同じ(品質を落とさない)

    def test_slow_remote_falls_to_local_quickly(self):
        loc = FastLocal()
        st = RemoteSTT("http://x", loc, post=slow_post(1.0), hedge_s=0.2)
        t0 = time.time()
        text, lang, prob = st.transcribe(SAMPLES)
        self.assertLess(time.time() - t0, 0.5)  # 1秒待たない
        self.assertEqual(text, "ローカル")
        self.assertEqual(st.last_where, "mac")
        self.assertEqual(st.hedged, 1)

    def test_slow_remote_does_not_trigger_cooldown(self):
        # 遅いだけ(落ちてはいない): 次の発話も リモートサーバーを試す
        loc = FastLocal()
        delays = [1.0, 0.01]
        def post(url, data=None, timeout=None):
            time.sleep(delays.pop(0))
            return Resp({"text": "リモート", "lang": "ja", "prob": 0.9})
        st = RemoteSTT("http://x", loc, post=post, hedge_s=0.2)
        st.transcribe(SAMPLES)
        self.assertEqual(st.transcribe(SAMPLES)[0], "リモート")

    def test_remote_error_falls_back_and_cools_down(self):
        loc = FastLocal()
        calls = []
        def post(url, data=None, timeout=None):
            calls.append(1)
            raise ConnectionError("down")
        st = RemoteSTT("http://x", loc, post=post, hedge_s=0.5)
        self.assertEqual(st.transcribe(SAMPLES)[0], "ローカル")
        st.transcribe(SAMPLES)
        self.assertEqual(len(calls), 1)          # 落ちている間は投げない(今までどおり)

    def test_slow_local_whisper_is_not_hedged(self):
        # このMacの耳が whisper(1回2秒)なら並べても速くならない: 今までどおり待つ
        class SlowLocal:
            calls = 0
            def transcribe(self, samples):
                SlowLocal.calls += 1
                return "w", "ja", 0.9
        st = RemoteSTT("http://x", SlowLocal(), post=slow_post(0.3), hedge_s=0.1)
        self.assertEqual(st.transcribe(SAMPLES)[0], "リモート")
        self.assertEqual(SlowLocal.calls, 0)

    def test_concurrent_calls_are_safe(self):
        st = RemoteSTT("http://x", FastLocal(), post=slow_post(0.05), hedge_s=0.3)
        out = []
        ths = [threading.Thread(target=lambda: out.append(st.transcribe(SAMPLES)[0]))
               for _ in range(4)]
        [t.start() for t in ths]
        [t.join(2) for t in ths]
        self.assertEqual(out, ["リモート"] * 4)


class TestPreview(unittest.TestCase):
    """確定は リモートサーバーの結果のまま、遅いときはこのMacの耳で予想訳だけ先に出す。
    0.7秒で SenseVoice に切り替える案は、混んだ回線(往復中央0.7〜1.2秒)で字幕 p90 を
    5.3→2.9秒にしたが chrF が4/6会話で2〜3下がった(FLEURS)。品質は落とさない。"""

    def test_slow_remote_previews_local_but_returns_remote(self):
        loc = FastLocal()
        got = []
        st = RemoteSTT("http://x", loc, post=slow_post(0.4), hedge_s=2.0, preview_s=0.1)
        out = st.transcribe(SAMPLES, on_preview=lambda t, l: got.append((t, l)))
        self.assertEqual(got, [("ローカル", "ja")])     # 先に予想訳用
        self.assertEqual(out[0], "リモート")             # 確定は remote
        self.assertEqual(st.last_where, "remote")
        self.assertEqual(st.previewed, 1)

    def test_fast_remote_no_preview(self):
        loc = FastLocal()
        got = []
        st = RemoteSTT("http://x", loc, post=slow_post(0.01), hedge_s=2.0, preview_s=0.2)
        st.transcribe(SAMPLES, on_preview=lambda t, l: got.append(t))
        self.assertEqual(got, [])
        self.assertEqual(loc.calls, 0)

    def test_local_reused_when_cap_reached(self):
        loc = FastLocal()
        st = RemoteSTT("http://x", loc, post=slow_post(1.0), hedge_s=0.3, preview_s=0.1)
        self.assertEqual(st.transcribe(SAMPLES, on_preview=lambda t, l: None)[0], "ローカル")
        self.assertEqual(loc.calls, 1)                  # 2度聞かない


class TestTimeoutIsNotDown(unittest.TestCase):
    def test_timeout_does_not_lock_to_local(self):
        import requests
        loc = FastLocal()
        n = []
        def post(url, data=None, timeout=None):
            n.append(1)
            if len(n) == 1:
                raise requests.ReadTimeout("slow")
            return Resp({"text": "リモート", "lang": "ja", "prob": 0.9})
        st = RemoteSTT("http://x", loc, post=post, hedge_s=0.5)
        self.assertEqual(st.transcribe(SAMPLES)[0], "ローカル")
        self.assertEqual(st.transcribe(SAMPLES)[0], "リモート")   # 30秒ローカル固定にしない


class TestWarmEar(unittest.TestCase):
    def test_warm_ear_calls_remote_once(self):
        from tsuyaku.__main__ import _warm_ear
        n = []
        def post(url, data=None, timeout=None):
            n.append(len(data))
            return Resp({"text": "", "lang": "ja", "prob": 1.0})
        _warm_ear(RemoteSTT("http://x", FastLocal(), post=post))
        self.assertEqual(n, [32000])                 # 1秒(16bit)
        _warm_ear(RemoteSTT(None, FastLocal(), post=post))   # このMacだけ: 何もしない
        self.assertEqual(len(n), 1)

    def test_app_warms_ear_at_start(self):
        from pathlib import Path
        src = (Path(__file__).resolve().parents[1] / "src/tsuyaku/__main__.py").read_text()
        self.assertIn("threading.Thread(target=_warm_ear, args=(self.stt,), daemon=True).start()", src)


class TestCongestion(unittest.TestCase):
    def test_net_overhead_tracked_from_server_sec(self):
        def post(url, data=None, timeout=None):
            time.sleep(0.05)
            return Resp({"text": "x", "lang": "ja", "prob": 0.9, "sec": 0.05})
        st = RemoteSTT("http://x", FastLocal(), post=post)
        st.transcribe(SAMPLES)
        self.assertLess(st.net_s, 0.1)
        self.assertFalse(st.congested)

    def test_slow_round_trips_mark_congested_then_recover(self):
        delay = [0.9]
        def post(url, data=None, timeout=None):
            time.sleep(delay[0])
            return Resp({"text": "x", "lang": "ja", "prob": 0.9, "sec": 0.05})
        st = RemoteSTT("http://x", FastLocal(), post=post)
        for _ in range(3):
            st.transcribe(SAMPLES)
        self.assertTrue(st.congested)
        delay[0] = 0.06
        for _ in range(6):
            st.transcribe(SAMPLES)
        self.assertFalse(st.congested)

    def test_app_emits_net_event_on_change(self):
        from pathlib import Path
        src = (Path(__file__).resolve().parents[1] / "src/tsuyaku/__main__.py").read_text()
        self.assertIn('self.ui.put(("net", {"slow": cg', src)
        self.assertIn("TSUYAKU.net&&TSUYAKU.net(", src)


class TestWiring(unittest.TestCase):
    def test_setting_and_app(self):
        from pathlib import Path
        from tsuyaku.config import Settings
        self.assertAlmostEqual(Settings().stt_hedge_s, 4.0)
        self.assertAlmostEqual(Settings().stt_preview_s, 0.35)
        src = (Path(__file__).resolve().parents[1] / "src/tsuyaku/__main__.py").read_text()
        self.assertIn("hedge_s=s.stt_hedge_s", src)
        self.assertIn("preview_s=s.stt_preview_s", src)
        self.assertIn("hear(seg, preview if ep.pending() is None else None, span,", src)   # 11/08: cont= を足した


if __name__ == "__main__":
    unittest.main()
