"""アプリ名は「Sokutsu」(11/06 ユーザー: 「デカく(即通)じゃなくて Sokutsu とか、もう少しグローバルに」)。
起動画面はロゴ(2色の弧の S)+ワードマーク、「即通」は小さく添えるだけ。ウィンドウ名・タブ名・Dock アイコンを揃える。"""
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HTML = (ROOT / "src/tsuyaku/webui/index.html").read_text()
MAIN = (ROOT / "src/tsuyaku/__main__.py").read_text()


class TestAppName(unittest.TestCase):
    def test_ui_shows_name(self):
        self.assertIn("<title>Sokutsu</title>", HTML)
        # 主役はロゴ+ワードマーク
        self.assertIn('<h1 class="mark" lang="en" aria-label="Sokutsu">', HTML)
        self.assertIn('<svg class="logo"', HTML)
        self.assertIn('<span class="word">Sokutsu</span>', HTML)
        # 「即通」は小さく添える
        self.assertIn('<span class="kanji" lang="ja">即通</span>', HTML)
        # Web フォントは使わない(ローカル完結)
        self.assertNotIn("@font-face", HTML)
        self.assertNotIn("fonts.googleapis", HTML)

    def test_window_title(self):
        self.assertIn('"Sokutsu", str(WEBUI)', MAIN)

    def test_readme_heading(self):
        self.assertTrue((ROOT / "README.md").read_text().startswith("# Sokutsu"))


class TestLogoAssets(unittest.TestCase):
    def test_logo_files(self):
        logo = (ROOT / "src/tsuyaku/webui/logo.svg").read_text()
        self.assertIn('viewBox="0 0 100 100"', logo)
        self.assertIn("#E9A954", logo)   # 上の弧 = 言語A(真鍮)
        self.assertIn("#5CC6C0", logo)   # 下の弧 = 言語B(潮)
        icon = (ROOT / "src/tsuyaku/assets/icon.svg").read_text()
        self.assertIn('width="1024"', icon)
        self.assertIn('rx="185"', icon)  # macOS アイコン格子(本体 824 角・角丸)

    def test_icon_png_is_1024_rgba(self):
        png = (ROOT / "src/tsuyaku/assets/icon_1024.png").read_bytes()
        self.assertEqual(png[:8], b"\x89PNG\r\n\x1a\n")
        w = int.from_bytes(png[16:20], "big")
        h = int.from_bytes(png[20:24], "big")
        self.assertEqual((w, h), (1024, 1024))
        self.assertEqual(png[25], 6)     # color type 6 = RGBA(角の外は透過)

    def test_make_icon_script(self):
        sh = (ROOT / "scripts/make_icon.sh").read_text()
        self.assertIn("--default-background-color=00000000", sh)
        self.assertIn("icon_1024.png", sh)


class TestDockIcon(unittest.TestCase):
    def test_main_sets_dock_icon_before_start(self):
        i = MAIN.index("def main():")
        seg = MAIN[i:]
        self.assertIn("_set_dock_icon()", seg)
        self.assertLess(seg.index("_set_dock_icon()"), seg.index("webview.start("))
        k = MAIN.index("def _set_dock_icon(")
        body = MAIN[k:i]
        self.assertIn("setApplicationIconImage_", body)
        self.assertIn("except Exception", body)   # 失敗しても起動を止めない

    def test_set_dock_icon_never_raises(self):
        from tsuyaku.__main__ import _set_dock_icon
        self.assertFalse(_set_dock_icon(ROOT / "no_such_icon.png"))


if __name__ == "__main__":
    unittest.main()
