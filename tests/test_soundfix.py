"""耳の文字の用語の崩れを音の近さで直す(soundfix.py、11/08 IT 用語)。"""
import unittest

from tsuyaku import soundfix as sf
from tsuyaku import terms


class TestKeys(unittest.TestCase):
    def test_kana_key(self):
        self.assertEqual(sf.kana_key("クバネティス"), "kubanetisu")
        self.assertEqual(sf.kana_key("クーバネティス"), "kubanetisu")      # 長音は落とす
        self.assertEqual(sf.kana_key("ヴィジュアル"), sf.kana_key("ビジュアル"))

    def test_jamo(self):
        self.assertEqual(len(sf.jamo("쿠버네티스")), 10)
        self.assertEqual(sf.jamo("쿠 버"), sf.jamo("쿠버"))


class TestFixJa(unittest.TestCase):
    def test_broken_product_reading(self):
        self.assertEqual(sf.fix("クーベルネティスで動かしています", "ja"), "Kubernetesで動かしています")
        self.assertEqual(sf.fix("テンセルフローのモデル", "ja"), "TensorFlowのモデル")

    def test_known_reading_kept(self):
        # 用語集にある読み(言ったとおり)はそのまま。訳の手がかりで正しい表記を伝える
        self.assertEqual(sf.fix("クバネティスで動かしています", "ja"), "クバネティスで動かしています")

    def test_general_words_untouched(self):
        for s in ("コンピューターを使う", "スマートフォンを買った", "インターネットで調べる", "プレゼンテーションの準備",
                  "アイスクリームを食べる", "パーティーに出る", "オーストラリアに行く", "オーストリアの首都",
                  "サッカーのワールドカップ", "レストランで食事", "チョコレートケーキ", "アメリカ大統領選挙",
                  "テレビのニュース", "エネルギー問題", "ビタミンが足りない", "ウクライナの情勢"):
            self.assertEqual(sf.fix(s, "ja"), s, s)

    def test_known_suffix(self):
        got = sf.fix("クーベルネティスクラスターを作る", "ja")
        self.assertTrue(got.startswith("Kubernetes"), got)


class TestFixKo(unittest.TestCase):
    def test_broken_hangul_with_particle(self):
        self.assertEqual(sf.fix("쿠베네티스에서 돌립니다", "ko"), "쿠버네티스에서 돌립니다")

    def test_spaced_syllables_joined(self):
        self.assertEqual(sf.fix("쿠 버 네 티 스 에서 돌립니다", "ko"), "쿠버네티스에서 돌립니다")

    def test_general_untouched(self):
        for s in ("오늘 날씨가 좋습니다", "대통령 선거 결과", "아이스크림을 먹었다", "컴퓨터를 켰습니다",
                  "스마트폰을 샀어요", "인터넷으로 찾았다", "오스트레일리아에 갔다"):
            self.assertEqual(sf.fix(s, "ko"), s, s)


class TestCleanHeard(unittest.TestCase):
    def test_clean_heard_uses_soundfix(self):
        self.assertEqual(terms.clean_heard("クーベルネティスで動かす", "ja"), "Kubernetesで動かす")

    def test_other_langs_untouched(self):
        self.assertEqual(sf.fix("Cooper netties", "en"), "Cooper netties")


if __name__ == "__main__":
    unittest.main()


class TestTiers(unittest.TestCase):
    def test_long_vowel_variant_any_loanword(self):
        # 鍵がちょうど同じ(長音・促音の揺れ)ならカタカナ語でも直す
        self.assertEqual(sf.fix("バックアップ中はスルプットが下がります", "ja"), "バックアップ中はスループットが下がります")

    def test_first_sound_only_differs_for_products(self):
        self.assertEqual(sf.fix("ペンソルフローのモデル", "ja"), "TensorFlowのモデル")

    def test_common_words_protected(self):
        for s in ("ステイングの話", "プログラムを書く", "スケジュールを決める", "マネージャーに聞く"):
            self.assertEqual(sf.fix(s, "ja"), s, s)

    def test_loanword_fix_does_not_add_words(self):
        got = sf.fix("ステージイングで確認します", "ja")
        self.assertNotIn("環境", got)
