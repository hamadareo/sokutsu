"""ux3(11/10): 音量・訳し直し・訳せなかった時の案内の画面側。"""
import re
import unittest
from pathlib import Path

from test_audit_ui import run_js

HTML = (Path(__file__).resolve().parents[1] / "src" / "tsuyaku" / "webui" / "index.html").read_text()


def script(sid):
    m = re.search(r'<script id="%s">(.*?)</script>' % sid, HTML, re.S)
    assert m, sid
    return m.group(1)


class TestUx3(unittest.TestCase):
    def test_syntax(self):
        js = script("ux3-js").replace("\\", "\\\\").replace("`", "\\`").replace("$", "\\$")
        self.assertEqual(run_js("try{new Function(`" + js + "`);'ok'}catch(e){'ERR '+e}"), "ok")

    def test_hear_last(self):
        self.assertLess(HTML.index('id="ux3-js"'), HTML.index('<style id="hear-css">'))
        self.assertTrue(HTML.rstrip().rsplit('</script>', 1)[1].strip().startswith('<!-- ===== /hear -->'))

    def test_wires(self):
        js = script("ux3-js")
        for s in ("set_volume", "retranslate", "get_tts", '","', '"."', '"r"', "訳せませんでした"):
            self.assertIn(s, js)

    def test_i18n(self):
        js = script("ux3-js")
        keys = set(re.findall(r'(?<![\w.])L\("([^"]+)"', js)) | set(re.findall(r'<dd data-t>([^<]+)</dd>', js))
        for lang in ("en", "zh", "ko"):
            blk = re.search(r"I18N\.%s,\{(.*?)\}\);" % lang, js, re.S).group(1)
            for k in keys:
                self.assertIn('"%s":' % k, blk, (lang, k))

    def test_no_private_info(self):
        self.assertNotRegex(script("ux3-js"), r"\d+\.\d+\.\d+\.\d+|remote")


if __name__ == "__main__":
    unittest.main()
