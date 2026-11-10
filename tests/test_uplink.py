"""先送り(11/06): 話している間に声を リモートサーバーの耳へ送っておき、聞くときは範囲だけ送る。

見つけた問題: ふつうの回線では話し終わってから区間の音声(4秒=128KB)を送ると往復が
中央2.0秒・p90 3.4秒(音声なしの往復は中央0.2秒)。続き待ちの繋ぎ直しは前の断片ごと
送り直し(15秒=480KB)、4秒の時間切れでこのMacの耳に落ちていた。
守ること: whisper に入る音声は今までと1サンプルも変わらない(品質は同じ)。
届いていない・古いサーバーなら今までどおり音声ごと送る。
"""
import importlib.util
import threading
import unittest
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np

from tsuyaku.audio import SpeechCapture, StreamSegmenter
from tsuyaku.segmenter import Endpointer, _trim_span
from tsuyaku.stt import RemoteSTT
from tsuyaku.uplink import Uplink, _subtract, to_pcm16

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("remote_stt_server", ROOT / "server/remote_stt_server.py")
srv = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(srv)


class Resp:
    def __init__(self, obj, code=200):
        self.obj, self.status_code = obj, code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(self.status_code)

    def json(self):
        return self.obj


class FakeServer:
    """remote_stt_server.py の StreamStore を使う、HTTP 抜きの偽サーバー。"""

    def __init__(self, stream=True):
        self.store = srv.StreamStore()
        self.stream = stream
        self.heard: list = []      # whisper に渡った int16
        self.bodies: list = []     # /transcribe の本文の大きさ
        self.lock = threading.Lock()

    def post(self, url, data=None, timeout=None):
        u = urlparse(url)
        q = parse_qs(u.query)
        sid = q.get("sid", [""])[0]
        if u.path == "/append":
            if not self.stream:
                return Resp({}, 404)
            self.store.append(sid, int(q["off"][0]), np.frombuffer(data, dtype="<i2"))
            return Resp({"ok": True})
        self.bodies.append(len(data))
        if self.stream and sid and q.get("r"):
            raw = np.frombuffer(data, dtype="<i2")
            pos = 0
            for a, b in srv.parse_ranges(q.get("u", [""])[0]):
                self.store.append(sid, a, raw[pos:pos + b - a])
                pos += b - a
            x = self.store.build(sid, q["r"][0])
            if x is None:
                return Resp({"missing": True}, 409)
        else:
            x = np.frombuffer(data, dtype="<i2")
        with self.lock:
            self.heard.append(x.copy())
        out = {"text": f"heard {x.size}", "lang": "en", "prob": 0.9, "sec": 0.1}
        if self.stream and sid and q.get("r"):
            out["span"] = True
        return Resp(out)


def voice(n, seed=0):
    return (np.random.default_rng(seed).standard_normal(n) * 0.1).astype(np.float32)


class TestStore(unittest.TestCase):
    def test_build_ranges_and_zeros(self):
        st = srv.StreamStore()
        x = to_pcm16(voice(32000))
        st.append("s", 1000, x[:16000])
        st.append("s", 17000, x[16000:])
        got = st.build("s", "1500-2500,z80,20000-20100")
        self.assertEqual(got.size, 1000 + 80 + 100)
        np.testing.assert_array_equal(got[:1000], x[500:1500])
        self.assertFalse(got[1000:1080].any())
        np.testing.assert_array_equal(got[1080:], x[19000:19100])

    def test_missing_sample_is_none(self):
        st = srv.StreamStore()
        st.append("s", 0, to_pcm16(voice(1000)))
        self.assertIsNone(st.build("s", "0-1001"))
        self.assertIsNone(st.build("other", "0-10"))

    def test_old_audio_is_dropped(self):
        st = srv.StreamStore(keep_s=1.0)
        st.append("s", 0, to_pcm16(voice(8000)))
        st.append("s", 40000, to_pcm16(voice(8000)))
        self.assertIsNone(st.get("s", 0, 100))
        self.assertIsNotNone(st.get("s", 40000, 48000))


class TestUplinkPlan(unittest.TestCase):
    def test_unsent_audio_goes_in_body_then_only_range(self):
        up = Uplink("http://x", start=False)
        x = voice(16000)
        up.feed(5000, x)
        recipe, need, body = up.plan([(5000, 21000)])
        self.assertEqual(recipe, "5000-21000")
        self.assertEqual(need, [(5000, 21000)])
        self.assertEqual(body, to_pcm16(x).tobytes())
        up.acked(need)
        recipe, need, body = up.plan([(6000, 7000), ("z", 50), (8000, 9000)])
        self.assertEqual(recipe, "6000-7000,z50,8000-9000")
        self.assertEqual((need, body), ([], b""))

    def test_unknown_audio_means_send_whole(self):
        up = Uplink("http://x", start=False)
        up.feed(5000, voice(1000))
        self.assertIsNone(up.plan([(4000, 6000)]))

    def test_subtract(self):
        self.assertEqual(_subtract(0, 10, [[2, 4], [6, 8]]), [(0, 2), (4, 6), (8, 10)])
        self.assertEqual(_subtract(0, 10, [[0, 10]]), [])


class TestRemoteWithUplink(unittest.TestCase):
    def test_same_audio_reaches_whisper_with_small_body(self):
        fs = FakeServer()
        up = Uplink("http://x", post=fs.post, start=False)
        x = voice(48000)
        up.feed(160000, x)
        # 先送りを1回ぶん流す(行列の先頭を送る)
        up._todo = [[160000, 160000 + 48000]]
        a, b = up._todo[0]
        fs.post(f"http://x/append?sid={up.sid}&off={a}", data=to_pcm16(x).tobytes())
        up.acked([(a, b)])
        st = RemoteSTT("http://x", None, post=fs.post)
        st.uplink = up
        text, lang, _p = st.transcribe(x[8000:40000], span=[(168000, 200000)])
        self.assertEqual(text, "heard 32000")
        np.testing.assert_array_equal(fs.heard[-1], to_pcm16(x[8000:40000]))
        self.assertEqual(fs.bodies[-1], 0)          # 範囲だけ(音声は送り直さない)
        self.assertEqual(st.span_hits, 1)

    def test_rejoin_with_zero_gap_matches_old_path(self):
        fs = FakeServer()
        up = Uplink("http://x", post=fs.post, start=False)
        x = voice(64000)
        up.feed(0, x)
        merged = np.concatenate([x[1000:9000], np.zeros(800, np.float32), x[20000:30000]])
        st = RemoteSTT("http://x", None, post=fs.post)
        st.uplink = up
        st.transcribe(merged, span=[(1000, 9000), ("z", 800), (20000, 30000)])
        old = RemoteSTT("http://x", None, post=fs.post)
        old.transcribe(merged)                       # 今までどおり音声ごと
        np.testing.assert_array_equal(fs.heard[0], fs.heard[1])
        # 1回目はまだ届いていなかった分だけ本文で送る
        self.assertEqual(fs.bodies[0], 2 * (8000 + 10000))

    def test_old_server_falls_back_to_whole_audio(self):
        fs = FakeServer(stream=False)
        up = Uplink("http://x", post=fs.post, start=False)
        x = voice(16000)
        up.feed(0, x)
        up.acked([(0, 16000)])     # 届いたつもりでも、サーバーが範囲を知らない
        st = RemoteSTT("http://x", None, post=fs.post)
        st.uplink = up
        text, _l, _p = st.transcribe(x, span=[(0, 16000)])
        self.assertEqual(text, "heard 16000")       # 空の本文を聞いた結果は使わない
        self.assertEqual(st.span_miss, 1)
        fs2 = FakeServer()
        up2 = Uplink("http://x", post=fs2.post, start=False)
        up2.feed(0, x)
        up2.acked([(0, 16000)])    # 届いたつもり(サーバーが再起動した等)
        st2 = RemoteSTT("http://x", None, post=fs2.post)
        st2.uplink = up2
        text, _l, _p = st2.transcribe(x, span=[(0, 16000)])
        self.assertEqual(text, "heard 16000")       # 409 → 音声ごと送り直す
        self.assertEqual(st2.span_miss, 1)
        np.testing.assert_array_equal(fs2.heard[-1], to_pcm16(x))
        self.assertEqual(up2._acked, [])           # 次からその範囲は本文で送る
        st2.transcribe(x, span=[(0, 16000)])
        self.assertEqual(st2.span_hits, 1)

    def test_wrong_length_span_is_ignored(self):
        fs = FakeServer()
        up = Uplink("http://x", post=fs.post, start=False)
        up.feed(0, voice(16000))
        st = RemoteSTT("http://x", None, post=fs.post)
        st.uplink = up
        st.transcribe(voice(8000), span=[(0, 16000)])
        self.assertEqual(fs.bodies[-1], 16000)       # 音声ごと(8000サンプル×2バイト)
        self.assertEqual(st.span_hits, 0)

    def test_background_sender_delivers(self):
        fs = FakeServer()
        up = Uplink("http://x", post=fs.post, batch_s=0.01)
        x = voice(8000)
        up.feed(100, x)
        for _ in range(200):
            if up._acked:
                break
            threading.Event().wait(0.01)
        up.close()
        self.assertEqual(up._acked, [[100, 8100]])
        np.testing.assert_array_equal(fs.store.get(up.sid, 100, 8100), to_pcm16(x))


class TestEndpointerSpan(unittest.TestCase):
    def test_rejoin_gets_span_matching_merged_audio(self):
        got = {}

        def rejoin(a, span=None):
            got["n"], got["span"] = len(a), span
            return "one two three.", "en", 0.9
        ep = Endpointer(rejoin=rejoin, hold_open_s=1.0)
        sr = 16000
        a1 = voice(sr)            # 0.0〜1.0秒
        ep.push(a1, 0.0, 1.0, "one two", "en", now=1.0, span=[(0, sr)])
        a2 = voice(int(1.5 * sr), 1)   # 1.3〜2.8秒(間0.3秒)
        ep.push(a2, 1.3, 2.8, "three.", "en", now=3.0, span=[(int(1.3 * sr), int(2.8 * sr))])
        self.assertEqual(got["span"], [(0, sr), ("z", int(0.3 * sr)), (int(1.3 * sr), int(2.8 * sr))])
        self.assertEqual(sum(b if a == "z" else b - a for a, b in got["span"]), got["n"])

    def test_overlap_trims_span(self):
        got = {}

        def rejoin(a, span=None):
            got["n"], got["span"] = len(a), span
            return "x y.", "en", 0.9
        ep = Endpointer(rejoin=rejoin, hold_open_s=1.0)
        ep.push(voice(16000), 0.0, 1.0, "x", "en", now=1.0, span=[(0, 16000)])
        ep.push(voice(16000), 0.8, 1.8, "y.", "en", now=2.0, pre=0.3, span=[(12800, 28800)])
        self.assertEqual(got["span"], [(0, 16000), (16000, 28800)])
        self.assertEqual(sum(b - a for a, b in got["span"]), got["n"])

    def test_rejoin_without_span_param_still_works(self):
        ep = Endpointer(rejoin=lambda a: ("a b.", "en", 0.9), hold_open_s=1.0)
        ep.push(voice(16000), 0.0, 1.0, "a", "en", now=1.0, span=[(0, 16000)])
        us = ep.push(voice(16000), 1.2, 2.2, "b.", "en", now=2.5, span=[(19200, 35200)])
        self.assertEqual(us[0].text, "a b.")

    def test_trim_span_through_zeros(self):
        self.assertEqual(_trim_span([("z", 10), (100, 200)], 30), [(120, 200)])
        self.assertEqual(_trim_span([(0, 50)], 0), [(0, 50)])


class TestCaptureStreams(unittest.TestCase):
    def test_every_closed_segment_is_already_uploaded(self):
        """区切りが閉じた時点で、その区間(手前の補い込み)の声は全部先送りの行列にある。"""
        cap = SpeechCapture.__new__(SpeechCapture)
        cap.sr = 16000
        cap.segmenter = StreamSegmenter(16000, cut_s=0.18)
        cap.uplink = Uplink("http://x", start=False)
        cap._up = 0
        rng = np.random.default_rng(3)
        segs = []
        # 話す(1.2秒)→ 間(0.6秒)を4回。声は大きい雑音、間は小さい雑音
        pattern = []
        for _ in range(4):
            pattern += [(True, 1.2), (False, 0.6)]
        for sp, dur in pattern:
            for _ in range(int(dur * 16000 / 512)):
                ch = (rng.standard_normal(512) * (0.1 if sp else 0.001)).astype(np.float32)
                items = cap.segmenter.feed(ch, sp)
                cap.stream_up(items)
                segs += items
        self.assertGreaterEqual(len(segs), 3)
        for audio, lo, hi, _pre in segs:
            a, b = int(round(lo * 16000)), int(round(hi * 16000))
            self.assertEqual(b - a, len(audio))
            plan = cap.uplink.plan([(a, b)])
            self.assertIsNotNone(plan)
            self.assertEqual(plan[2], to_pcm16(audio).tobytes())   # 同じ声


class TestWiring(unittest.TestCase):
    def test_app_passes_span_and_creates_uplink(self):
        src = (ROOT / "src/tsuyaku/__main__.py").read_text()
        self.assertIn("Uplink(stt_url) if (stt_url and s.stt_stream and server_streams(stt_url))", src)
        self.assertIn("self.capture.uplink = self.uplink", src)
        self.assertIn("lambda a, span=None: hear(a, preview, span)", src)
        self.assertIn("prob=prob, span=span,", src)

    def test_server_reports_stream(self):
        self.assertGreaterEqual(srv.VER, 2)
        self.assertIn('"stream": True', (ROOT / "server/remote_stt_server.py").read_text())


if __name__ == "__main__":
    unittest.main()


class TestMergeBacklog(unittest.TestCase):
    """11/06 英会話 AMI 実時間評価: 詰まると最新1個だけ残して区間を捨てていた。まとめて聞く。"""

    def test_merge_keeps_all_audio_with_span(self):
        from tsuyaku.segmenter import merge_segments
        sr = 16000
        x = voice(sr * 10)
        items = [(x[0:sr], 0.0, 1.0, 0.0),                    # 0〜1秒
                 (x[int(1.3 * sr):3 * sr], 1.3, 3.0, 0.2),    # 間0.3秒
                 (x[int(2.9 * sr):5 * sr], 2.9, 5.0, 0.1)]    # 0.1秒重なる(手前の補い)
        audio, lo, hi, pre, span = merge_segments(items, sr)
        self.assertEqual((lo, hi, pre), (0.0, 5.0, 0.0))
        self.assertEqual(span, [(0, sr), ("z", int(0.3 * sr)), (int(1.3 * sr), 3 * sr),
                                (3 * sr, 5 * sr)])
        self.assertEqual(sum(b if a == "z" else b - a for a, b in span), len(audio))
        # 範囲の声はマイクの声そのもの
        up = Uplink("http://x", start=False)
        up.feed(0, x)
        plan = up.plan(span)
        self.assertIsNotNone(plan)
        nz = np.concatenate([audio[:sr], audio[sr + int(0.3 * sr):]])   # 無音で埋めた所を除く
        self.assertEqual(plan[2], to_pcm16(nz).tobytes())

    def test_too_long_keeps_newest(self):
        from tsuyaku.segmenter import merge_segments
        sr = 16000
        items = [(voice(sr * 10), 0.0, 10.0, 0.0), (voice(sr * 10), 10.5, 20.5, 0.0),
                 (voice(sr * 10), 21.0, 31.0, 0.0)]
        audio, lo, hi, _pre, span = merge_segments(items, sr, max_s=25.0)
        self.assertEqual(lo, 10.5)
        self.assertEqual(hi, 31.0)

    def test_worker_merges_instead_of_dropping(self):
        src = (ROOT / "src/tsuyaku/__main__.py").read_text()
        self.assertIn("pending = [merge_segments(pending, s.sample_rate)]", src)
        self.assertIn("span = (item[4] if len(item) > 4 else", src)


class TestStoreOverlap(unittest.TestCase):
    def test_shorter_chunk_at_same_offset_keeps_longer_data(self):
        """聞き取りの本文で [a,b) が届いた後に、先送りの [a,c)(c<b)が届いても後ろを消さない
        (11/06 実時間評価: 「サーバーに無い」409 が続いた原因)。"""
        st = srv.StreamStore()
        x = to_pcm16(voice(32000))
        st.append("s", 1000, x)            # 本文で長く
        st.append("s", 1000, x[:8000])     # 先送りで短く(同じ始まり)
        np.testing.assert_array_equal(st.get("s", 1000, 33000), x)
