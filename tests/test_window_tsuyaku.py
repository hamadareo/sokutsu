"""11/07 W2 が WKWebView で発見: index.html の `const TSUYAKU` は window に載らない。
Python の _push は notes / relabel / speakers / net を "window.TSUYAKU&&TSUYAKU.x&&…" で呼ぶので、
会議メモの進み具合と「できました」・会話中の話者の付け直し・メモで分かった名前・回線混雑の表示が
画面に届かず黙って捨てられていた。"""
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class TestWindowTsuyaku(unittest.TestCase):
    def test_exported_right_after_definition(self):
        h = (ROOT / "src/tsuyaku/webui/index.html").read_text()
        i = h.index("\nconst TSUYAKU={")
        j = h.index("\n};\n", i)
        self.assertTrue(h[j + 4:].startswith("window.TSUYAKU=TSUYAKU;"))

    def test_python_guards_use_window(self):
        src = (ROOT / "src/tsuyaku/__main__.py").read_text()
        self.assertTrue(re.search(r"window\.TSUYAKU&&TSUYAKU\.notes", src))


if __name__ == "__main__":
    unittest.main()
