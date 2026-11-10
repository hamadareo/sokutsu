"""専門用語に強くする(11/08 ユーザー「どの言語でも専門用語などに強くなってほしい」)。

見つけた問題: 耳が英字の略語を読みのカタカナで書く(「ジーピーユー」のような字名の並び)と、
訳(LLM)がそれを意味の無い音訳にする。略語のまま(「AI・IoT」)なら正しく訳せる。
グラフ/ラフのような似た語を、前の文の訳から写してしまう。
"""
import unittest

from tsuyaku import terms
from tsuyaku.translate import build_messages, build_system


class TestLetters(unittest.TestCase):
    def n(self, s, lang="ja"):
        return terms.normalize_letters(s, lang)

    def test_ja_spelled_acronyms(self):
        self.assertEqual(self.n("ジーピーユーシーピーユーの比較をしました。"), "GPU・CPUの比較をしました。")
        self.assertEqual(self.n("エピーアイを呼び出します。"), "APIを呼び出します。")          # 短い形「エ」
        self.assertEqual(self.n("社内システムエーピーアイについて調べています。"), "社内システムAPIについて調べています。")
        self.assertEqual(self.n("エーアイスピーカーを買った"), "AIスピーカーを買った")
        self.assertEqual(self.n("ピーシーを使います"), "PCを使います")
        self.assertEqual(self.n("ディーエヌエーを調べる"), "DNAを調べる")
        self.assertEqual(self.n("ジーピーユーで学習する"), "GPUで学習する")
        self.assertEqual(self.n("エーアイオーティーの時代"), "AIoTの時代")
        self.assertEqual(self.n("エムエスの資料"), "MSの資料")      # 知らない略語でも語全体が字名なら

    def test_ja_misheard_letters(self):
        # このMacの耳(SenseVoice)の実際の出力(say の声): 1字の聞き違いは知っている略語へ
        self.assertEqual(self.n("ジービーユーが八枚必要です。"), "GPUが八枚必要です。")
        self.assertEqual(self.n("エスキュエルのクエリが遅いです。"), "SQLのクエリが遅いです。")
        self.assertEqual(self.n("ユーエビーの電源は五ボルトです。"), "USBの電源は五ボルトです。")
        self.assertEqual(self.n("エルエルエのファインチューニング"), "LLMのファインチューニング")
        self.assertEqual(self.n("エイオーティーとは"), "AIoTとは")        # 1字抜け

    def test_ja_real_words_untouched(self):
        for s in ("アイディアを出す", "ハワイに行く", "パーティーに出る", "アイスクリームを食べる",
                  "えーと、エーエーと言った", "スピーシーズの話", "アイがある", "データベースを作る",
                  "コーヒーを飲む", "ワイワイ話す", "シーズンが始まる", "ビーチに行く"):
            self.assertEqual(self.n(s), s, s)

    def test_ko(self):
        self.assertEqual(self.n("에이아이를 공부합니다", "ko"), "AI를 공부합니다")
        self.assertEqual(self.n("지피유가 필요해요", "ko"), "GPU가 필요해요")
        self.assertEqual(self.n("아이오티 기기", "ko"), "IoT 기기")
        for s in ("아이가 웃었다", "오이를 샀다", "비가 온다", "이 모델은 좋다", "아이티에 질병을 전파"):
            self.assertEqual(self.n(s, "ko"), s, s)

    def test_ko_spaced_syllables(self):
        self.assertEqual(self.n("유 에스 비 전 원은 오 볼트", "ko"), "USB 전 원은 오 볼트")
        self.assertEqual(self.n("배터리 로 동작 하기 때문에", "ko"), "배터리 로 동작 하기 때문에")
        self.assertEqual(self.n("이 유 는 간단 합니다", "ko"), "이 유 는 간단 합니다")

    def test_other_langs_untouched(self):
        self.assertEqual(self.n("エーアイ", "en"), "エーアイ")
        self.assertEqual(self.n("我在学习AI。", "zh"), "我在学习AI。")
        self.assertEqual(self.n("", "ja"), "")


class TestUserTermFix(unittest.TestCase):
    def test_near_miss_fixed(self):
        u = ["北山工業大学", "ProjectX3x3"]
        self.assertEqual(terms.fix_user_terms("私は北山工大学の学生です", u), "私は北山工業大学の学生です")
        self.assertEqual(terms.fix_user_terms("we ran ProjectX3x on it", u), "we ran ProjectX3x3 on it")

    def test_other_user_term_kept(self):
        u = ["北山工業大学", "北山工业大学"]
        self.assertEqual(terms.fix_user_terms("在北山工业大学主修", u), "在北山工业大学主修")

    def test_latin_term_does_not_eat_next_char(self):
        u = ["Transformer"]
        self.assertEqual(terms.fix_user_terms("在NLP研究中transformer现在已经成为主流。", u),
                         "在NLP研究中transformer现在已经成为主流。")
        self.assertEqual(terms.fix_user_terms("在研究中Transfomer现在很流行", u), "在研究中Transformer现在很流行")

    def test_short_or_far_untouched(self):
        u = ["山田太郎", "北山工業大学"]
        self.assertEqual(terms.fix_user_terms("山田次郎です", u), "山田次郎です")      # 4字の名前は直さない
        self.assertEqual(terms.fix_user_terms("南山商業大学です", u), "南山商業大学です")  # 2字違い
        self.assertEqual(terms.fix_user_terms("北山工業大学です", u), "北山工業大学です")

    def test_clean_heard(self):
        self.assertEqual(terms.clean_heard("北山工大学でエーアイを学ぶ", "ja", ["北山工業大学"]),
                         "北山工業大学でAIを学ぶ")

    def test_iot_misheard_d(self):
        self.assertEqual(terms.normalize_letters("センサーアイオーディーの設計", "ja"), "センサーIoTの設計")
        self.assertEqual(terms.normalize_letters("アイオティについて", "ja"), "IoTについて")


class TestTermBase(unittest.TestCase):
    def pairs(self, s, src, tgt, **kw):
        return terms.glossary_pairs(s, src, tgt, **kw)

    def test_loads(self):
        tb = terms.termbase()
        self.assertGreater(len(tb.entries), 300)

    def test_match_katakana_boundary(self):
        self.assertIn(("グラフ", "图表"), self.pairs("このグラフの縦軸は時間です。", "ja", "zh"))
        self.assertNotIn("グラフ", [s for s, _ in self.pairs("フォトグラフィーの話", "ja", "zh")])
        self.assertNotIn("ラフ", [s for s, _ in self.pairs("このグラフの縦軸は時間です。", "ja", "zh")])

    def test_han_compound_not_split(self):
        got = dict(self.pairs("这款单片机面向嵌入式应用。", "zh", "ja"))
        self.assertNotEqual(got.get("嵌入"), "埋め込み")
        got = dict(self.pairs("我们用负载均衡器来分散请求。", "zh", "ja"))
        self.assertNotIn("负载均衡", got)

    def test_latin_word_and_case(self):
        srcs = [s for s, _ in self.pairs("I think it works on the GPU.", "en", "ja")]
        self.assertNotIn("it", [x.lower() for x in srcs if x == "it"])
        got = dict(self.pairs("We use machine learning here.", "en", "zh"))
        self.assertEqual(got.get("machine learning"), "机器学习")

    def test_identity_and_keep_skipped(self):
        # 訳しても同じ形(GPU → GPU)は手がかりに載せない(指示を短く)
        self.assertEqual([p for p in self.pairs("GPUで学習します", "ja", "ko") if p[0] == "GPU"], [])

    def test_longest_match_wins(self):
        got = self.pairs("深層学習と機械学習", "ja", "en")
        self.assertIn(("深層学習", "deep learning"), got)
        self.assertIn(("機械学習", "machine learning"), got)

    def test_user_pair_wins(self):
        got = dict(self.pairs("このグラフの縦軸は時間です。", "ja", "zh", user=["グラフ=图形"]))
        self.assertEqual(got["グラフ"], "图形")

    def test_source_forms(self):
        self.assertEqual(terms.source_forms(["グラフ=图表", "YOLO", "ラフ → rough sketch"]),
                         ("グラフ", "YOLO", "ラフ"))

    def test_user_plain_latin_kept(self):
        got = dict(self.pairs("ProjectXの精度が上がった", "ja", "en", user=["ProjectX", "山田"]))
        self.assertEqual(got.get("ProjectX"), "ProjectX")
        self.assertNotIn("山田", got)          # 日本語の名前は訳す側に任せる(ローマ字にする)

    def test_limit(self):
        s = "機械学習と深層学習とニューラルネットワークと強化学習と自然言語処理と音声認識と画像認識と転移学習と過学習"
        self.assertLessEqual(len(self.pairs(s, "ja", "en", limit=5)), 5)

    def test_hint_text(self):
        h = terms.glossary_hint("このグラフの縦軸は時間です。", "ja", "zh")
        self.assertIn("グラフ = 图表", h)
        self.assertEqual(terms.glossary_hint("こんにちは", "ja", "zh"), "")


class TestPrompt(unittest.TestCase):
    def test_hint_goes_after_history_before_utterance(self):
        hist = [{"lang": "ja", "orig": "今日は晴れですね。", "trans": "今天天气很好。"}]
        msgs = build_messages(("ja", "zh"), "", hist, "このグラフの縦軸は時間です。", "ja")
        body = msgs[-1]["content"]
        self.assertIn("图表", body)
        self.assertLess(body.index("Earlier conversation"), body.index("图表"))
        self.assertLess(body.index("图表"), body.index("このグラフの縦軸は時間です。"))
        # 指示文は変えない(サーバーの読み込みの使い回し)
        self.assertEqual(msgs[0]["content"], build_system(("ja", "zh"), ""))

    def test_similar_history_dropped(self):
        # 似た(同じではない)前の文は履歴に載せない: 前の訳を写す(機関 → 期間)のを防ぐ
        hist = [{"lang": "ja", "orig": "この機関について説明します。", "trans": "我来介绍一下这个机构。"},
                {"lang": "ja", "orig": "今日は晴れですね。", "trans": "今天天气很好。"}]
        body = build_messages(("ja", "zh"), "", hist, "この期間について説明します。", "ja")[-1]["content"]
        self.assertNotIn("这个机构", body)
        self.assertIn("今天天气很好", body)
        same = build_messages(("ja", "zh"), "", hist[:1], "この機関について説明します。", "ja")[-1]["content"]
        self.assertIn("这个机构", same)          # 同じ文の繰り返しは今までどおり載せる(訳をそろえる)

    def test_no_hint_no_change(self):
        msgs = build_messages(("ja", "en"), "", [], "こんにちは。", "ja")
        self.assertNotIn("term", msgs[-1]["content"].lower())

    def test_letters_normalized_in_prompt(self):
        msgs = build_messages(("ja", "zh"), "", [], "ジーピーユーシーピーユーの比較をしました。", "ja")
        self.assertIn("GPU・CPU", msgs[-1]["content"])

    def test_user_terms_param(self):
        msgs = build_messages(("ja", "zh"), "", [], "このグラフの縦軸は時間です。", "ja", terms=["グラフ=图形"])
        self.assertIn("图形", msgs[-1]["content"])


class TestWiring(unittest.TestCase):
    def test_heard_text_cleaned_before_caption_and_translation(self):
        from pathlib import Path
        src = (Path(__file__).resolve().parents[1] / "src/tsuyaku/__main__.py").read_text()
        self.assertIn("text_q.put((clean_heard(u.text, u.lang, self._heard_terms)", src)
        self.assertIn("terms=getattr(self, \"_user_terms\", ())", src)


if __name__ == "__main__":
    unittest.main()
