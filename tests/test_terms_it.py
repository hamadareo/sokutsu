"""IT 用語に強くする(11/08 2回目「もっとIT用語や専門用語に強くなって」)。"""
import time
import unittest

from tsuyaku import terms

IT_D = terms.IT_DOMAINS


class TestBigTermBase(unittest.TestCase):
    def pairs(self, s, src, tgt, **kw):
        return dict(terms.glossary_pairs(s, src, tgt, **kw))

    def test_size(self):
        tb = terms.termbase()
        self.assertGreaterEqual(len(tb.entries), 3000)
        self.assertGreaterEqual(sum(e["d"] in IT_D for e in tb.entries), 2500)
        for e in tb.entries:
            for lang in ("ja", "en", "zh", "ko"):
                self.assertTrue(e[lang], e)

    def test_reading_variants(self):
        # 製品名の読みのカタカナ・ハングル・略称 → 正しい表記
        self.assertEqual(self.pairs("クバネティスでポッドを動かします", "ja", "en").get("クバネティス"), "Kubernetes")
        self.assertEqual(self.pairs("ポスグレのインデックスを張り直した", "ja", "en").get("ポスグレ"), "PostgreSQL")
        self.assertEqual(self.pairs("쿠버네티스에서 파드를 띄웠습니다", "ko", "ja").get("쿠버네티스"), "Kubernetes")

    def test_zh_it_terms(self):
        got = self.pairs("我们把单体应用拆成了微服务，前面加了负载均衡。", "zh", "ja")
        self.assertIn("微服务", got)
        self.assertIn("负载均衡", got)

    def test_ambiguous_source_form_excluded(self):
        # 정규화 = normalization / regularization(どちらとも決められない形は手がかりにしない)
        self.assertNotIn("정규화", self.pairs("정규화를 적용했습니다", "ko", "en"))

    def test_mixed_case_is_case_sensitive(self):
        self.assertEqual(self.pairs("LoRaで通信します", "ja", "zh").get("LoRa", "LoRa"), "LoRa")
        got = self.pairs("LoRAで微調整します", "ja", "en")
        self.assertNotEqual(got.get("LoRA", "LoRA"), "LoRa")

    def test_fast_with_big_termbase(self):
        terms.warm()
        s = "本番環境はKubernetesで動かしていて、PostgreSQLはRDSに移しました。レイテンシとスループットも見ます。"
        t0 = time.perf_counter()
        for _ in range(200):
            terms.glossary_hint(s, "ja", "en")
        per = (time.perf_counter() - t0) / 200
        self.assertLess(per, 0.003, f"{per * 1000:.2f} ms")


class TestContextGate(unittest.TestCase):
    """一般の文で意味が割れる語(process = 過程/プロセス)は、IT の話の中だけで手がかりにする。"""

    def test_general_sentence_no_ctx_hint(self):
        got = dict(terms.glossary_pairs("The process of making bread takes hours.", "en", "ja"))
        self.assertNotIn("process", got)

    def test_it_sentence_ctx_hint(self):
        got = dict(terms.glossary_pairs("The Linux kernel kills the process when memory runs out.", "en", "ja"))
        self.assertIn("process", got)

    def test_domain_it_enables_ctx(self):
        got = dict(terms.glossary_pairs("The process takes hours.", "en", "ja", domain="it"))
        self.assertIn("process", got)

    def test_context_from_history(self):
        got = dict(terms.glossary_pairs("The process takes hours.", "en", "ja",
                                        context="We deployed it on Kubernetes yesterday."))
        self.assertIn("process", got)


if __name__ == "__main__":
    unittest.main()


class TestKeepLatinInTarget(unittest.TestCase):
    """原文の形に英字の語(S3・ArF・Pub/Sub)があるのに、訳の第一表記にそれが無い手がかりは出さない
    (リモートサーバーの訳で「S3バケット = 存储桶」が S3 を消した)。英字を残す別の訳があればそれを使う。"""

    def test_latin_token_not_dropped(self):
        for s, src, tgt in (("バックアップはS3バケットに置いています", "ja", "zh"),
                            ("ArF液浸でマルチパターニングします", "ja", "zh"),
                            ("メッセージはPub/Subに流れます", "ja", "zh")):
            for f, t in terms.glossary_pairs(s, src, tgt):
                for tok in ("S3", "ArF", "Pub/Sub"):
                    if tok in f:
                        self.assertIn(tok.lower(), t.lower(), (f, t))


class TestKeepLatinEn(unittest.TestCase):
    def test_en_source_keeps_names_but_translates_words(self):
        got = dict(terms.glossary_pairs("We store backups in an S3 bucket.", "en", "zh"))
        for f, t in got.items():
            if "S3" in f:
                self.assertIn("s3", t.lower())
        self.assertEqual(dict(terms.glossary_pairs("We use machine learning here.", "en", "zh")).get("machine learning"),
                         "机器学习")


class TestGeneralWordsInBase(unittest.TestCase):
    """一般の文で別の意味になる語(FLEURS で見つけた): 和平协议 の 协议 ≠ プロトコル、急勾配 の 勾配 ≠ 梯度、
    香港回归 の 回归 ≠ 回帰。IT の話の中だけで手がかりにする。"""

    def test_general_zh(self):
        for s in ("两国在1940年达成了和平协议。", "香港回归以后变化很大。", "请点赞订阅转发。",
                  "这个法律有一个漏洞。", "这篇文章的摘要很短。"):
            self.assertEqual(terms.glossary_pairs(s, "zh", "ja"), [], s)

    def test_general_ja(self):
        self.assertEqual(terms.glossary_pairs("急勾配の坂に注意してください。", "ja", "zh"), [])

    def test_it_context_still_hints(self):
        got = dict(terms.glossary_pairs("这个API的通信协议改成了gRPC。", "zh", "ja"))
        self.assertEqual(got.get("协议"), "プロトコル")

    def test_hypothesis_zh_target(self):
        self.assertEqual(dict(terms.glossary_pairs("この仮説を検証します。", "ja", "zh")).get("仮説"), "假设")
