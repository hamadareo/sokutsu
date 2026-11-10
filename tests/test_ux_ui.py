"""ux1(11/10): 右クリックの追加・使用率の表示。index.html を文字として読み、script は new Function で構文を確かめる。"""
import re
import unittest
from pathlib import Path

from test_audit_ui import run_js

HTML = (Path(__file__).resolve().parents[1] / "src" / "tsuyaku" / "webui" / "index.html").read_text()


def script(sid: str) -> str:
    m = re.search(r'<script id="%s">(.*?)</script>' % sid, HTML, re.S)
    assert m, sid
    return m.group(1)


class TestUx1(unittest.TestCase):
    def test_syntax(self):
        js = script("ux1-js").replace("\\", "\\\\").replace("`", "\\`").replace("$", "\\$")
        self.assertEqual(run_js("try{new Function(`" + js + "`);'ok'}catch(e){'ERR '+e}"), "ok")

    def test_hear_block_is_last(self):
        self.assertLess(HTML.index('id="ux1-js"'), HTML.index('id="hear-js"'))
        self.assertLess(HTML.index('id="ux1-js"'), HTML.index('<style id="hear-css">'))
        self.assertTrue(HTML.rstrip().rsplit('</script>', 1)[1].strip().startswith('<!-- ===== /hear -->'))
        self.assertGreater(HTML.index('id="hear-js"'), HTML.index('id="ux1-js"'))

    def test_wires(self):
        js = script("ux1-js")
        for name in ("get_sysmon", "set_sysmon", "speak_again", "term_add", "sysmonSet", "showSysmon"):
            self.assertIn(name, js)

    def test_no_private_info(self):
        js = script("ux1-js") + script("ux1-js")
        self.assertNotRegex(js, r"\d+\.\d+\.\d+\.\d+|remote|@")

    def test_off_stops_polling(self):
        js = script("ux1-js")
        self.assertIn("if(!sysOn||document.hidden) return;", js)
        self.assertIn("clearInterval(sysT)", js)

    def test_i18n_labels(self):
        js = script("ux1-js")
        for lang in ("en", "zh", "ko"):
            m = re.search(r"I18N\.%s,\{(.*?)\}\);" % lang, js, re.S)
            self.assertTrue(m and "もう一度読み上げ" in m.group(1) and "使用率の表示" in m.group(1), lang)


if __name__ == "__main__":
    unittest.main()
