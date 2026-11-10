"""日中の言語の取り違えを文字の種類で直す(11/08 リーダー、このMacの耳で試した日中の会話)。

見つけた問題: 中国語の発話(漢字と英字が混じった短い文・広東語の字)が ja と判定され、ja→zh に回って
中国語を中国語に「訳し」ていた。漢字だけの文は耳の判定(heard)に従うので、耳が ja と言えば ja になる。
変更: 簡体字・広東語にしか無い字(们这吗说你佢…)があり仮名が無ければ zh。仮名があれば ja のまま。
文はすべて作った例。
"""
import unittest

from tsuyaku.langdetect import ZH_ONLY, detect_lang, resolve_lang


class TestZhFix(unittest.TestCase):
    def test_zh_judged_ja_becomes_zh(self):
        for t in ("你 just.", "And佢去咗。", "你们好吗", "这个OK吗", "我们明天见。", "谢谢你。", "说一下。"):
            self.assertEqual(resolve_lang(t, "ja"), "zh", t)

    def test_detect_with_latin(self):
        self.assertEqual(detect_lang("你 just."), "zh")
        self.assertEqual(detect_lang("And佢…"), "zh")

    def test_japanese_kanji_only_kept(self):
        for t in ("了解。", "会議室", "大丈夫", "以上", "実際", "勉強", "来週", "会社", "発表", "電話", "東京"):
            self.assertEqual(resolve_lang(t, "ja"), "ja", t)

    def test_kana_means_ja(self):
        for t in ("それは了解です。", "你さん", "これはOKです"):
            self.assertEqual(resolve_lang(t, "zh"), "ja", t)

    def test_no_japanese_char_in_set(self):
        # 日本語で普通に使う字は入れない(来・会・点・度・係・喝・没・那・个・几・吃)
        for c in "来会点度係喝没那个几吃":
            self.assertNotIn(c, ZH_ONLY.pattern, c)


if __name__ == "__main__":
    unittest.main()
