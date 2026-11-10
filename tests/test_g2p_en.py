"""英語の読み上げで辞書に無い語が消える問題(11/06 実会話)。

Kokoro(sherpa-onnx)は英語を同梱辞書だけで音にし、辞書に無い語を黙って
飛ばしていた: "That's great." →「great」、"Midori" "Aoba" "conversations"
も無音。発音記号にしてから渡す g2p_en で全語が音になることを確かめる。
"""
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KOK = ROOT / "models/kokoro-int8-multi-lang-v1_0"


@unittest.skipUnless((KOK / "lexicon-us-en.txt").exists(), "kokoro モデルが無い")
class TestEnG2P(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from tsuyaku.tts import _load_g2p
        cls.g = _load_g2p(KOK)

    def test_live_session_words_are_all_voiced(self):
        for s in ["That's great.",
                  "I'm Ryo Yamada, a student at Kaihin Institute of Technology.",
                  "I currently live in Midori City, Aoba Prefecture.",
                  "We have refined it so actual conversations can take place."]:
            self.assertEqual(self.g.missing(s), [], s)

    def test_plural_from_lexicon_stem(self):
        self.assertEqual(self.g.word("conversations"), self.g.word("conversation") + "z")
        self.assertTrue(self.g.word("cats").endswith("s"))

    def test_numbers_are_read(self):
        ph = " ".join(self.g.sentences("in 2026."))
        self.assertIn("twˈɛnti", ph)

    def test_function_words_use_weak_forms(self):
        # 辞書の a は「エイ」、the は「ジー」、live は「ライブ」しか無い
        self.assertEqual(self.g.word("a"), "ə")
        self.assertEqual(self.g.word("the"), "ðə")
        self.assertEqual(self.g.word("live"), "lˈɪv")

    def test_sentences_split_and_only_model_symbols(self):
        out = self.g.sentences("Hello, Yamada. How are you? Fine!")
        self.assertEqual(len(out), 3)
        self.assertTrue(out[0].startswith("həlˈO,"))
        self.assertTrue(all(c in self.g.vocab for p in out for c in p))

    def test_tts_uses_phonemes_for_english(self):
        src = (ROOT / "src/tsuyaku/tts.py").read_text()
        # 11/08 speed: 読点ごとにできた音を callback で渡す(できた所から鳴らす)
        self.assertIn("audio = self._kokoro(text, lang, int(val), speed, callback=cb)", src)
        self.assertIn("cfg.phoneme_codepoints", src)


if __name__ == "__main__":
    unittest.main()
