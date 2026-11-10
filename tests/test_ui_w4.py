"""W4 の画面(index.html の W4 の塊): 4言語の文言がそろっていること・既存の関数を包んでいること・
JS の文法(JavaScriptCore で読めること)。"""
import re
import subprocess
import tempfile
import unittest
from pathlib import Path

HTML = (Path(__file__).resolve().parents[1] / "src" / "tsuyaku" / "webui" / "index.html").read_text()


def w4_block() -> str:
    i, j = HTML.index("/*W4-BEGIN*/"), HTML.index("/*W4-END*/")
    return HTML[i:j]


class TestW4Ui(unittest.TestCase):
    def test_every_w4_string_has_en_zh_ko(self):
        js = w4_block()
        keys = set(re.findall(r'\b(?:this|T)\.L\("((?:[^"\\]|\\.)*)"', js))
        keys |= set(re.findall(r'\bE\("((?:[^"\\]|\\.)*)"\)', js))          # const E=x=>this.esc(this.L(x))
        keys |= set(re.findall(r'<span data-t>([^<]+)</span>', js))
        self.assertGreater(len(keys), 60)
        missing = []
        for k in sorted(keys):
            n = HTML.count(f'"{k}":')
            if n < 3:
                missing.append((k, n))
        self.assertEqual(missing, [], "訳の無い文言")

    def test_wraps_instead_of_rewriting(self):
        js = w4_block()
        for name in ("renderList", "renderDoc", "meetOpen", "meetClose", "rename", "speakers"):
            self.assertIn(f"T.{name}=function", js)
        self.assertRegex(HTML, r'class="tln[^\n]*data-i="\'\+i\+\'"')   # 検索の結果から行へ飛ぶ印

    def test_bulk_select_and_delete(self):
        js = w4_block()
        self.assertIn('id="w4selBtn"', js)                          # 一覧の「選ぶ」
        for k in ("all", "none", "del", "restore", "purge", "kill", "cancel", "exit"):
            self.assertIn(f'data-w4s="{k}"', js)
        self.assertIn('"meetings_trash",ids', js)                    # 削除はまずゴミ箱へ
        self.assertIn('"meetings_restore",moved', js)                # 元に戻す
        self.assertIn('"trash_delete",ids,ids.length', js)           # 完全に削除は件数つき
        self.assertIn("e.shiftKey", js)                               # Shift で範囲
        self.assertIn('"library_busy"', js)                          # 今の会話は選べない
        self.assertIn("w4selToggle(false)", js)                       # Esc で選ぶモードを抜ける
        self.assertIn('id="w4allBtn"', js)                          # 会話中も一覧へ

    def test_calendar_button_only_on_click(self):
        js = w4_block()
        self.assertIn('getElementById("w4calBtn")', js.replace("$(\"#w4calBtn\")", 'getElementById("w4calBtn")'))
        self.assertNotRegex(js, r'DOMContentLoaded[^}]*calendar_now')   # 起動しただけでは予定を読まない

    def test_javascript_parses(self):
        js = "\n".join(re.findall(r"<script(?: [^>]*)?>(.*?)</script>", HTML, re.S))
        with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False) as f:
            f.write(js)
        r = subprocess.run(["osascript", "-l", "JavaScript", "-e",
                            "ObjC.import('Foundation'); var s=$.NSString.stringWithContentsOfFileEncodingError("
                            f"'{f.name}',4,null).js; try{{new Function(s);'ok'}}catch(e){{'ERR '+e}}"],
                           capture_output=True, text=True, timeout=30)
        self.assertEqual(r.stdout.strip(), "ok", r.stdout + r.stderr)


if __name__ == "__main__":
    unittest.main()
