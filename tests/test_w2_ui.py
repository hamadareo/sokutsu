"""W2 の画面(index.html の W2 の塊): Python との約束と、文言の4言語。"""
import re
import unittest
from pathlib import Path

HTML = (Path(__file__).resolve().parents[1] / "src/tsuyaku/webui/index.html").read_text(encoding="utf-8")
I, J = HTML.index("/*W2-I18N-BEGIN*/"), HTML.index("/*W2-I18N-END*/")
W2 = HTML[J:HTML.index("</script>", J)]           # W2 のスクリプト(辞書の後ろ)
DICT = HTML[HTML.index("/*I18N-BEGIN*/"):HTML.index("/*I18N-END*/")] + HTML[I:J]


class TestW2Ui(unittest.TestCase):
    def test_python_events_reach_the_page(self):
        # _push は window.TSUYAKU&&… で呼ぶ。const の TSUYAKU は window に載らないので載せる
        self.assertIn("window.TSUYAKU=TSUYAKU;", HTML)
        self.assertIn("w2(d){", W2)
        for t in ('"rec"', '"audio_ready"', '"import"', '"drop"'):
            self.assertIn("d.type===" + t, W2)

    def test_transcript_lines_carry_index(self):
        self.assertRegex(HTML, r'<div class="tln[^>]*data-i="\'\+i\+\'"')

    def test_line_click_leaves_w3_tools_alone(self):
        # 文を押すと再生位置が動くが、W3 の道具・編集中の行・話者の選択・「修正済み」では動かさない
        self.assertIn('ln.classList.contains("editing")', W2)
        self.assertIn(".w3tools,.w3edit,.w3menu,.w3e", W2)

    def test_uses_bridge_api(self):
        from tsuyaku.__main__ import Bridge
        for name in set(re.findall(r"\ba\.([a-z]+_[a-z_]+)\(", W2)):
            self.assertTrue(callable(getattr(Bridge, name, None)), name)

    def test_every_w2_text_is_translated(self):
        keys = set(re.findall(r'(?:this|T)\.L\("([^"]+)"', W2)) | set(re.findall(r'\bE\("([^"]+)"\)', W2))
        keys |= set(re.findall(r'(?:label|sub):"([^"]+)"', W2))
        keys |= set(re.findall(r'"(この形式は読めません[^"]*|会話中は取り込めません[^"]*|いま別のファイル[^"]*|取り込めませんでした|ファイルの場所を[^"]*)"', W2))
        self.assertGreater(len(keys), 40)
        for k in keys:
            self.assertGreaterEqual(DICT.count('"' + k + '":'), 3, f"訳が足りない: {k}")


if __name__ == "__main__":
    unittest.main()
