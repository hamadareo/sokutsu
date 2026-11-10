"""回帰テスト: 「会話を始める」が pair=None で黙って固まらないこと。

11/05 実障害: 起動画面の言語ペアchip配線が `querySelectorAll(".pair")` と
無指定だったため、クラスを共有する『翻訳の処理場所』『まわりの音』chipにも
同じハンドラが刺さった。「このMac」等を押すと `b.dataset.pair` が undefined
に なり TSUYAKU.pair が消える。そのまま「会話を始める」→ pywebview が
undefined を Python の None に変換 → `_run` が入口の `pair.split("-")` で
即クラッシュ → スレッドが無言で死んで UI は「起動しています…」のまま
(ユーザーには「起動に時間がかかっています」に見えた)。
"""
import unittest
from pathlib import Path

from tsuyaku.__main__ import _DEFAULT_PAIR, _PAIRS, _norm_pair

ROOT = Path(__file__).resolve().parents[1]
MAIN = (ROOT / "src" / "tsuyaku" / "__main__.py").read_text()
HTML = (ROOT / "src" / "tsuyaku" / "webui" / "index.html").read_text()


class TestNormPair(unittest.TestCase):
    def test_valid_pairs_pass_through(self):
        for p in sorted(_PAIRS):
            self.assertEqual(_norm_pair(p), p)
        self.assertEqual(_PAIRS,
                         {"ja-en", "ja-zh", "ja-ko", "en-zh", "en-ko", "zh-ko"})

    def test_none_and_garbage_fall_back(self):
        for bad in (None, "", "JA-EN", "ja-de", "ja", 5, {"pair": "ja-en"}):
            self.assertEqual(_norm_pair(bad), _DEFAULT_PAIR)


class TestWiring(unittest.TestCase):
    def test_pair_chip_selector_is_scoped(self):
        # 無指定の `.pair` 選択が今回の障害の根因(処理場所/音chipまで拾う)
        self.assertNotIn('querySelectorAll(".pair")', HTML)
        self.assertIn('querySelectorAll("#pairs .pair")', HTML)

    def test_start_normalizes_pair(self):
        i = MAIN.index("def start")
        seg = MAIN[i:MAIN.index("def interrupt")]
        self.assertIn("_norm_pair", seg)

    def test_run_reports_errors_instead_of_dying_silently(self):
        # _run は本体を try で包み、例外時は画面にエラーと復帰を出す
        i = MAIN.index("def _run")
        seg = MAIN[i:MAIN.index("def _conversation")]
        self.assertIn("try:", seg)
        self.assertIn("self._conversation(pair, topic, mode)", seg)
        self.assertIn('"error"', seg)
        self.assertIn('"reset"', seg)


if __name__ == "__main__":
    unittest.main()
