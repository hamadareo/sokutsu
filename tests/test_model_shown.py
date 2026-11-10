"""翻訳に使われている言語モデルが見えること(会話中のバッジ + 起動画面のヒント)。

実行: PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -v
"""
from __future__ import annotations

import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MAIN = (ROOT / "src/tsuyaku/__main__.py").read_text()
HTML = (ROOT / "src/tsuyaku/webui/index.html").read_text()


class TestModelShown(unittest.TestCase):
    def test_bridge_records_and_pushes_chosen_model(self):
        self.assertIn("def get_model(", MAIN)
        # モデルが決まったら 処理場所+モデル名 をUIへ送る(pick_model の後)
        self.assertIn('TSUYAKU.engineInfo(', MAIN)
        self.assertLess(MAIN.index("pick_model(b, pref)"),
                        MAIN.index('("engine",'))

    def test_ui_shows_model_badge(self):
        self.assertIn('id="modelTag"', HTML)
        self.assertIn("engineInfo(d)", HTML)
        # バッジは会話中のステータスバーに、gearと同じ列で見える
        self.assertLess(HTML.index('id="modelTag"'), HTML.index('id="gear"'))

    def test_start_hint_includes_model_names(self):
        # 起動画面の get_engine がモデル名も返す(処理場所ごとの表示用)
        i = MAIN.index("def get_engine(")
        j = MAIN.index("def set_engine(")
        self.assertIn('"models"', MAIN[i:j])


if __name__ == "__main__":
    unittest.main()
