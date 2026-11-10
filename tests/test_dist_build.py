"""配る版のビルドの最後の確かめ(packaging/sanitize.py check): 私的なアドレスと、環境変数で指定した語が
.app の中のどこか(コメント・I18N・Info.plist・バイナリ)に1つでもあれば失敗する。"""
import importlib.util
import io
import re
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("sanitize", ROOT / "packaging" / "sanitize.py")
sanitize = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sanitize)


def check(text: "str | bytes", name="Info.plist") -> int:
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "Sokutsu.app" / "Contents" / name
        p.parent.mkdir(parents=True)
        p.write_bytes(text if isinstance(text, bytes) else text.encode())
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            return sanitize.check([Path(d) / "Sokutsu.app"])


class CheckTest(unittest.TestCase):
    def test_clean_passes(self):
        self.assertEqual(check("<string>app.sokutsu.Sokutsu</string> 例: 山田太郎"), 0)

    def test_private_addresses_fail(self):
        for w in ("10.1.2.3", "192.168.0.5", "172.16.4.4", "100.100.1.1", "myhost.tail1234.ts.net"):
            self.assertEqual(check(f"<!-- {w} -->"), 1, w)

    def test_public_addresses_pass(self):
        for w in ("127.0.0.1", "8.8.8.8", "1.100.2.3", "version 100.200.1.1"):
            self.assertEqual(check(f"<!-- {w} -->"), 0, w)

    def test_extra_words_from_env_fail(self):
        import os
        old = os.environ.get("SOKUTSU_FORBIDDEN_WORDS")
        os.environ["SOKUTSU_FORBIDDEN_WORDS"] = "Secret-Name, another-word"
        try:
            self.assertEqual(check("<!-- SECRET-NAME -->"), 1)
            self.assertEqual(check("<!-- another-word -->"), 1)
            self.assertEqual(check("<!-- fine -->"), 0)
        finally:
            if old is None:
                os.environ.pop("SOKUTSU_FORBIDDEN_WORDS", None)
            else:
                os.environ["SOKUTSU_FORBIDDEN_WORDS"] = old

    def test_in_binary_too(self):
        self.assertEqual(check(b"\x00\x01192.168.1.1\x00", "MacOS/sokutsu-sysaudio"), 1)


if __name__ == "__main__":
    unittest.main()


class RewriteKeepsTranslationsTest(unittest.TestCase):
    """配る版の書き換えで Python のエラー文と画面の訳の鍵がずれ、英語・中国語・韓国語の画面でも
    「翻訳サーバーに接続できません」の本文が日本語のまま出ていた(1.0.1 のビルドの試験で見つけた)。"""

    def test_server_error_message_is_still_translated(self):
        import json
        import shutil
        with tempfile.TemporaryDirectory() as d:
            src = ROOT / "src" / "tsuyaku"
            for rel in ("__main__.py", "webui/index.html"):
                (Path(d) / rel).parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(src / rel, Path(d) / rel)
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                self.assertEqual(sanitize.rewrite([d]), 0)
            py = (Path(d) / "__main__.py").read_text(encoding="utf-8")
            html = (Path(d) / "webui/index.html").read_text(encoding="utf-8")
            i = py.index('"起動画面の『翻訳の処理場所』を見直してください')
            j = py.index("。\"", i)
            msg = "".join(re.findall(r'"([^"]*)"', py[i:j + 2]))
            self.assertNotIn("server", msg.replace("server_url", ""))
            self.assertEqual(html.count(json.dumps(msg, ensure_ascii=False) + ":"), 3, msg)
            for line in html.splitlines():
                if json.dumps(msg, ensure_ascii=False) + ":" in line:
                    v = line.split(json.dumps(msg, ensure_ascii=False) + ":", 1)[1][:200]
                    self.assertNotIn("tunnel", v)
                    self.assertNotIn("LM Studio", v)
