"""聞き直しの見当を多く当てる(11/08 IT 用語)。

見つけた問題(IT の試験文 220文、このMacの耳 SenseVoice): 略語・製品名を落とした・崩した文 127 のうち、
今の見当(文頭の助詞など)に当たったのは 22。落ちた跡には、ほかの形もある:
文頭や語の頭の長音(「ーにプッシュする」)、カタカナの後の助詞の並び(「メッシュでを」)、
助詞の後の数字だけ(「ルーターで9を」= QoS、「6に移行」= IPv6)。IT の話の中では、
用語集にも一般の語の一覧にも無い短いカタカナ(「ブキー」= VPC、「デー」= DB)。"""
import unittest

from tsuyaku.relisten import dropped_letters


class TestMoreSigns(unittest.TestCase):
    def test_bar_at_start(self):
        self.assertTrue(dropped_letters("ーにプッシュするイメージは、タグを付けてください。", "ja"))

    def test_particle_pair_after_katakana(self):
        self.assertTrue(dropped_letters("通信は、サービスメッシュでを強制しています。", "ja"))

    def test_digit_after_particle(self):
        self.assertTrue(dropped_letters("会議の音声が途切れるので、ルーターで9を設定しました。", "ja"))
        self.assertTrue(dropped_letters("グローバルアドレスが足りないので、6に移行するまでは待ちます。", "ja"))

    def test_short_unknown_katakana_only_in_it_talk(self):
        s = "ブキーのプライベートサブネットにデーを置いています。"
        self.assertTrue(dropped_letters(s, "ja", it=True))
        self.assertFalse(dropped_letters(s, "ja"))

    def test_ordinary_sentences_still_quiet(self):
        for s in ("そのことを話しました。", "2020年に始まりました。", "3%くらい下がりました。",
                  "バスで駅まで行きます。", "ゴマを少し入れてください。", "今日はいい天気ですね。",
                  "おー、美味しいね。", "そのことで相談があります。"):
            self.assertFalse(dropped_letters(s, "ja"), s)
            if s not in ("ゴマを少し入れてください。",):   # IT の話の中の知らない短いカタカナは聞き直す(害は無い)
                self.assertFalse(dropped_letters(s, "ja", it=True), s)


if __name__ == "__main__":
    unittest.main()


class TestBridgeUsesItContext(unittest.TestCase):
    def setUp(self):
        from tsuyaku import __main__ as M
        from tsuyaku import terms
        self.terms = terms
        b = M.Bridge.__new__(M.Bridge)
        b._relisten = object()
        b.history = []
        self.b = b
        self.tm = {"pcm": b"\0\0"}

    def tearDown(self):
        self.terms.set_domain("")

    def test_history_makes_it_context(self):
        s = "ブキーの設定をデーに書いておきます。"
        self.assertFalse(self.b._relisten_wanted(s, "ja", self.tm))
        self.b.history = [{"lang": "ja", "orig": "本番はKubernetesで動かしています。", "trans": "x"}]
        self.assertTrue(self.b._relisten_wanted(s, "ja", self.tm))

    def test_domain_it(self):
        self.terms.set_domain("it")
        self.assertTrue(self.b._relisten_wanted("ブキーの設定を見直します。", "ja", self.tm))
