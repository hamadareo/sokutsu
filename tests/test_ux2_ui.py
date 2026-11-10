"""ux2(11/10): 履歴をたどる・記録の検索・読み上げ速度・字幕コピー。index.html を文字として読み、script は new Function で構文を確かめる。"""
import re
import unittest
from pathlib import Path

from test_audit_ui import run_js

HTML = (Path(__file__).resolve().parents[1] / "src" / "tsuyaku" / "webui" / "index.html").read_text()


def script(sid: str) -> str:
    m = re.search(r'<script id="%s">(.*?)</script>' % sid, HTML, re.S)
    assert m, sid
    return m.group(1)


class TestUx2(unittest.TestCase):
    def test_syntax(self):
        js = script("ux2-js").replace("\\", "\\\\").replace("`", "\\`").replace("$", "\\$")
        self.assertEqual(run_js("try{new Function(`" + js + "`);'ok'}catch(e){'ERR '+e}"), "ok")

    def test_hear_block_is_last(self):
        self.assertLess(HTML.index('id="ux2-js"'), HTML.index('<style id="hear-css">'))
        self.assertLess(HTML.index('id="ux2-css"'), HTML.index('id="hear-js"'))
        self.assertTrue(HTML.rstrip().rsplit('</script>', 1)[1].strip().startswith('<!-- ===== /hear -->'))

    def test_keys_wired(self):
        js = script("ux2-js")
        for s in ('"/"', '"ArrowUp"', '"End"', '"["', '"]"', 'get_tts', 'set_tts', 'copyText', 'ux2-hide'):
            self.assertIn(s, js)

    def test_find_esc_does_not_quit(self):
        js = script("ux2-js")
        self.assertIn("e.stopPropagation();   // Esc", js)

    def test_i18n_labels(self):
        js = script("ux2-js")
        keys = re.findall(r'(?<![\w.])L\("([^"]+)"', js)
        for lang in ("en", "zh", "ko"):
            m = re.search(r"I18N\.%s,\{(.*?)\}\);" % lang, js, re.S)
            self.assertTrue(m, lang)
            for k in set(keys):
                if k in ("{n}件", "コピーできませんでした"):
                    continue
                self.assertIn('"%s":' % k, m.group(1), (lang, k))

    def test_help_rows(self):
        self.assertIn("操作の早見に追記", script("ux2-js"))

    def test_no_private_info(self):
        self.assertNotRegex(script("ux2-js"), r"\d+\.\d+\.\d+\.\d+|s1" + "09|shu" + "zo")   # 配布ビルドの写しで書き換わらないよう分ける


if __name__ == "__main__":
    unittest.main()
