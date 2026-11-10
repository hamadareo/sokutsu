"""W1 の画面(会議に質問・テンプレート・章・元の発言へ飛ぶ・ライブ要約・下書き)。

画面は pywebview の1ファイル(index.html)。W1 が足す CSS・JS・文言はそれぞれ W1 の塊にまとめる
(ほかの担当と差分がぶつからないように)。画面の文言は4言語(ja/en/zh/ko)そろえる。
"""
import re
import unittest
from pathlib import Path

HTML = (Path(__file__).resolve().parents[1] / "src" / "tsuyaku" / "webui" / "index.html").read_text(encoding="utf-8")


def block(a, b):
    i, j = HTML.index(a), HTML.index(b)
    return HTML[i:j]


class TestW1Blocks(unittest.TestCase):
    def test_blocks_exist(self):
        for a, b in (("/*W1-CSS-BEGIN*/", "/*W1-CSS-END*/"), ("/*W1-JS-BEGIN*/", "/*W1-JS-END*/"),
                     ("/*W1-I18N-BEGIN*/", "/*W1-I18N-END*/")):
            self.assertLess(HTML.index(a), HTML.index(b))
        # 文言の塊は I18N の中(既存の「全部訳がある」テストの対象)
        self.assertLess(HTML.index("/*I18N-BEGIN*/"), HTML.index("/*W1-I18N-BEGIN*/"))
        self.assertLess(HTML.index("/*W1-I18N-END*/"), HTML.index("/*I18N-END*/"))

    def test_every_w1_text_is_translated(self):
        js = block("/*W1-JS-BEGIN*/", "/*W1-JS-END*/")
        i18n = block("/*I18N-BEGIN*/", "/*I18N-END*/")
        keys = set(re.findall(r'(?:this|TSUYAKU)\.L\("((?:[^"\\]|\\.)+)"', js))
        keys |= set(re.findall(r'W1T\("((?:[^"\\]|\\.)+)"\)', js))
        self.assertGreater(len(keys), 30)
        missing = [k for k in keys if i18n.count('"' + k + '":') < 3]
        self.assertEqual(missing, [])

    def test_ask_panel(self):
        js = block("/*W1-JS-BEGIN*/", "/*W1-JS-END*/")
        for f in ("askOpen(", "askSend(", "ask(d){", "askRender(", "jumpLine(", "citeHtml("):
            self.assertIn(f, js)
        self.assertIn("meeting_ask", js)
        self.assertIn("meetings_ask", js)            # 全会議に質問
        self.assertIn("isComposing", js)             # 日本語の変換中の Enter で送らない
        self.assertIn('id="askp"', HTML)

    def test_templates_on_start_and_doc(self):
        self.assertIn('id="tplChip"', HTML)      # 起動画面: W3 の「議題」「用語辞書」と並ぶ札
        self.assertIn('id="tplBox"', HTML)
        js = block("/*W1-JS-BEGIN*/", "/*W1-JS-END*/")
        for k in ("free", "regular", "1on1", "sales", "interview", "lecture", "brainstorm", "qa"):
            self.assertIn('"' + k + '"', js)
        self.assertIn("set_template", js)
        # 作り直しはテンプレートを渡す
        self.assertIn("notes_make(this._docId,this._docLang,", HTML)

    def test_drafts_and_live(self):
        js = block("/*W1-JS-BEGIN*/", "/*W1-JS-END*/")
        self.assertIn("notes_draft", js)
        self.assertIn("live_notes", js)
        self.assertIn("notesLive(", js)

    def test_python_pushes_ask_events(self):
        src = (Path(__file__).resolve().parents[1] / "src" / "tsuyaku" / "__main__.py").read_text(encoding="utf-8")
        self.assertIn('kind == "ask"', src)
        self.assertIn("TSUYAKU.ask", src)


if __name__ == "__main__":
    unittest.main()
