"""11/08「聞く音を選ぶ項目がありません」: 新しい会話のページで「会議メモだけ」を選ぶと「聞く音」の行が
CSS で隠れていた(5cb85a9)。Zoom などの会議メモは「会議メモだけ」+「マイク+この Mac の音」で使うので、
聞く音はどの使い方でも選べること。"""
import re
import unittest
from pathlib import Path

HTML = (Path(__file__).resolve().parents[1] / "src/tsuyaku/webui/index.html").read_text()


class TestSrcRowInNotesMode(unittest.TestCase):
    def test_not_hidden_in_any_mode(self):
        for m in re.finditer(r"([^{}]*)\{([^{}]*)\}", HTML):
            sel, body = m.group(1), m.group(2)
            if ("#srcRow" in sel or "#srcField" in sel or "#audioSrc" in sel) and "display:none" in body.replace(" ", ""):
                self.fail("聞く音の行を隠す CSS: " + sel.strip())

    def test_row_built(self):
        self.assertIn('row("聞く音"', HTML)


if __name__ == "__main__":
    unittest.main()
