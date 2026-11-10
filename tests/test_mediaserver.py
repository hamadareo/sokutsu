"""録音を画面(WKWebView)に渡す小さな HTTP サーバー(W2)。<audio> のシークは Range 要求で来る。"""
import tempfile
import unittest
import urllib.error
import urllib.request
from pathlib import Path

from tsuyaku.mediaserver import MediaServer


def get(url, rng=None):
    req = urllib.request.Request(url)
    if rng:
        req.add_header("Range", rng)
    with urllib.request.urlopen(req, timeout=5) as r:
        return r.status, dict(r.headers), r.read()


class TestMediaServer(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.TemporaryDirectory()
        self.dir = Path(self.d.name)
        self.f = self.dir / "session-20261107-101010.audio.m4a"
        self.f.write_bytes(bytes(range(256)) * 40)          # 10240 バイト
        self.srv = MediaServer(self.dir)

    def tearDown(self):
        self.srv.close()
        self.d.cleanup()

    def test_full_and_range(self):
        url = self.srv.url(self.f)
        st, h, b = get(url)
        self.assertEqual(st, 200)
        self.assertEqual(len(b), 10240)
        self.assertEqual(h["Accept-Ranges"], "bytes")
        self.assertEqual(h["Content-Type"], "audio/mp4")
        st, h, b = get(url, "bytes=100-199")
        self.assertEqual(st, 206)
        self.assertEqual(b, (bytes(range(256)) * 40)[100:200])
        self.assertEqual(h["Content-Range"], "bytes 100-199/10240")
        st, h, b = get(url, "bytes=10000-")
        self.assertEqual((st, len(b)), (206, 240))

    def test_only_127_and_token(self):
        url = self.srv.url(self.f)
        self.assertTrue(url.startswith("http://127.0.0.1:"))
        bad = url.replace(self.srv.token, "x" * len(self.srv.token))
        with self.assertRaises(urllib.error.HTTPError) as e:
            get(bad)
        self.assertEqual(e.exception.code, 403)

    def test_only_audio_files_in_the_log_dir(self):
        (self.dir / "session-20261107-101010.jsonl").write_text("{}")
        base = self.srv.url(self.f).rsplit("/", 1)[0]
        for name in ("session-20261107-101010.jsonl", "..%2Fx.audio.m4a", "nope.audio.m4a"):
            with self.assertRaises(urllib.error.HTTPError) as e:
                get(base + "/" + name)
            self.assertEqual(e.exception.code, 404, name)

    def test_bad_range(self):
        with self.assertRaises(urllib.error.HTTPError) as e:
            get(self.srv.url(self.f), "bytes=20000-")
        self.assertEqual(e.exception.code, 416)

    def test_caf_type(self):
        caf = self.dir / "session-20261107-101010.audio.caf"
        caf.write_bytes(b"caff" + bytes(100))
        st, h, _b = get(self.srv.url(caf))
        self.assertEqual(h["Content-Type"], "audio/x-caf")


if __name__ == "__main__":
    unittest.main()
