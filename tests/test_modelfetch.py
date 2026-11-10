"""配る版の初回のモデル取得: 進み具合・途中から再開・SHA256・展開・空き容量・失敗の理由。"""
import hashlib
import io
import tarfile
import tempfile
import unittest
from pathlib import Path

from tsuyaku import modelfetch as mf


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def tar_bz2(files: dict) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:bz2") as tf:
        for name, data in files.items():
            ti = tarfile.TarInfo(name)
            ti.size = len(data)
            tf.addfile(ti, io.BytesIO(data))
    return buf.getvalue()


class Server:
    """URL → bytes。Range(offset)に応じて返す。fail_after: その URL で何バイト返したら切るか。"""

    def __init__(self, blobs: dict, honor_range=True):
        self.blobs = blobs
        self.honor = honor_range
        self.calls = []
        self.fail_after: dict = {}

    def __call__(self, url, offset):
        self.calls.append((url, offset))
        data = self.blobs[url]
        honored = self.honor or offset == 0
        start = offset if honored else 0
        body = data[start:]
        limit = self.fail_after.get(url)

        def gen():
            sent = 0
            for i in range(0, len(body), 7):
                piece = body[i:i + 7]
                if limit is not None and sent + len(piece) > limit:
                    raise ConnectionError("回線が切れた")
                sent += len(piece)
                yield piece
        return gen(), honored, (lambda: None)


def items_for(blobs):
    a = b"weights-" * 20
    t = tar_bz2({"voice-dir/model.onnx": b"m" * 50, "voice-dir/tokens.txt": b"t"})
    blobs["u:a"] = a
    blobs["u:t"] = t
    return [mf.Item("a", "モデルA", "u:a", sha(a), len(a), "a.bin"),
            mf.Item("t", "声", "u:t", sha(t), len(t), "voice-dir", archive=True)]


class FetchTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name) / "models"
        self.blobs = {}
        self.items = items_for(self.blobs)

    def tearDown(self):
        self.tmp.cleanup()

    def fetcher(self, server, **kw):
        kw.setdefault("free_bytes", lambda p: 10**12)
        return mf.Fetcher(self.dir, self.items, opener=server, retries=0, sleep=lambda s: None, **kw)

    def test_downloads_verifies_and_extracts(self):
        f = self.fetcher(Server(self.blobs))
        f.run()
        st = f.state()
        self.assertEqual(st["phase"], "done", st)
        self.assertEqual((self.dir / "a.bin").read_bytes(), self.blobs["u:a"])
        self.assertEqual((self.dir / "voice-dir" / "tokens.txt").read_bytes(), b"t")
        self.assertEqual(mf.missing(self.dir, self.items), [])
        self.assertEqual(st["got"], st["total"])
        self.assertEqual([p.name for p in self.dir.iterdir()
                          if p.name.startswith(".") and p.name != ".fetch.lock"], [])
        self.assertFalse(list(self.dir.glob("*.part")))

    def test_resume_after_interruption(self):
        srv = Server(self.blobs)
        srv.fail_after["u:a"] = 50
        f = self.fetcher(srv)
        f.run()
        st = f.state()
        self.assertEqual(st["phase"], "error")
        self.assertEqual(st["errorKind"], "network")
        part = self.dir / "a.bin.part"
        self.assertTrue(0 < part.stat().st_size <= 50)
        have = part.stat().st_size
        # 次の起動: 続きから
        srv.fail_after.clear()
        f2 = self.fetcher(srv)
        self.assertEqual(f2.state()["got"], have)       # 最初から進み具合に入っている
        f2.run()
        self.assertEqual(f2.state()["phase"], "done")
        self.assertIn(("u:a", have), srv.calls)
        self.assertEqual((self.dir / "a.bin").read_bytes(), self.blobs["u:a"])

    def test_retry_inside_one_run(self):
        srv = Server(self.blobs)
        srv.fail_after["u:a"] = 30
        f = mf.Fetcher(self.dir, self.items, opener=srv, retries=2, sleep=lambda s: None,
                       free_bytes=lambda p: 10**12)

        orig = srv.__call__

        def once(url, offset):   # 1回目だけ切れる
            r = orig(url, offset)
            srv.fail_after.pop("u:a", None)
            return r
        f.opener = once
        f.run()
        self.assertEqual(f.state()["phase"], "done")

    def test_corrupt_download_is_removed(self):
        self.blobs["u:a"] = b"x" * len(self.blobs["u:a"])   # 中身が違う
        f = self.fetcher(Server(self.blobs))
        f.run()
        st = f.state()
        self.assertEqual(st["phase"], "error")
        self.assertEqual(st["errorKind"], "corrupt")
        self.assertFalse((self.dir / "a.bin.part").exists())
        self.assertFalse((self.dir / "a.bin").exists())

    def test_server_ignoring_range_restarts_cleanly(self):
        srv = Server(self.blobs, honor_range=False)
        self.dir.mkdir(parents=True)
        (self.dir / "a.bin.part").write_bytes(self.blobs["u:a"][:40])
        f = self.fetcher(srv)
        f.run()
        self.assertEqual(f.state()["phase"], "done")
        self.assertEqual((self.dir / "a.bin").read_bytes(), self.blobs["u:a"])

    def test_not_enough_disk(self):
        srv = Server(self.blobs)
        f = self.fetcher(srv, free_bytes=lambda p: 10)
        f.run()
        st = f.state()
        self.assertEqual(st["errorKind"], "disk")
        self.assertEqual(srv.calls, [])

    def test_cancel_keeps_partial(self):
        srv = Server(self.blobs)
        f = self.fetcher(srv)
        real = srv.__call__

        def slow(url, offset):
            it, ok, close = real(url, offset)

            def gen():
                for i, piece in enumerate(it):
                    if i == 3:
                        f.cancel()
                    yield piece
            return gen(), ok, close
        f.opener = slow
        f.run()
        self.assertEqual(f.state()["phase"], "cancelled")
        self.assertTrue((self.dir / "a.bin.part").exists())

    def test_unsafe_tar_is_rejected(self):
        bad = tar_bz2({"../evil.txt": b"x"})
        self.blobs["u:t"] = bad
        self.items[1] = mf.Item("t", "声", "u:t", sha(bad), len(bad), "voice-dir", archive=True)
        f = self.fetcher(Server(self.blobs))
        f.run()
        self.assertEqual(f.state()["phase"], "error")
        self.assertFalse((Path(self.tmp.name) / "evil.txt").exists())

    def test_already_done_skips_network(self):
        f = self.fetcher(Server(self.blobs))
        f.run()
        srv = Server(self.blobs)
        f2 = self.fetcher(srv)
        f2.run()
        self.assertEqual(srv.calls, [])
        self.assertEqual(f2.state()["phase"], "done")


class ManifestTest(unittest.TestCase):
    def test_official_sources_only(self):
        for it in mf.MANIFEST:
            self.assertTrue(it.url.startswith(("https://huggingface.co/ggml-org/",
                                               "https://huggingface.co/ggerganov/",
                                               "https://github.com/k2-fsa/sherpa-onnx/releases/")),
                            it.url)
            self.assertEqual(len(it.sha256), 64)
            self.assertGreater(it.size, 0)

    def test_huggingface_is_pinned_to_a_commit(self):
        # resolve/main は公開元の差し替えで SHA256 が合わなくなる(全員が全部受けてから失敗する)
        hf = [it for it in mf.MANIFEST if it.url.startswith("https://huggingface.co/")]
        self.assertTrue(hf)
        for it in hf:
            self.assertRegex(it.url, r"/resolve/[0-9a-f]{40}/", it.url)

    def test_no_duplicates_or_noncommercial(self):
        names = " ".join(i.dest for i in mf.MANIFEST)
        self.assertNotIn("ggml-large-v3-turbo.bin", names)       # fp16 は取らない
        self.assertNotIn("kokoro-multi-lang-v1_0", names)       # fp の kokoro は取らない
        self.assertNotIn("kss", names)                          # 韓国語 vits(CC BY-NC-SA)は取らない

    def test_matches_settings(self):
        from tsuyaku import config, localllm
        dests = {i.dest for i in mf.MANIFEST}
        for d in (config.DIST_WHISPER, config.DIST_SENSE_VOICE, config.DIST_KOKORO,
                  localllm.MODEL_FILE, "silero_vad.onnx"):
            self.assertIn(d, dests)


if __name__ == "__main__":
    unittest.main()
