"""プロンプト構築・言語判定・ループ抑制・ログ形式のユニットテスト。

実行: PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -v
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from tsuyaku.langdetect import detect_lang, is_echo, norm
from tsuyaku.log import SessionLog
from tsuyaku.translate import build_messages, build_system, out_lang, target_lang
from tsuyaku.tts import TTS


class TestPrompt(unittest.TestCase):
    def test_system_contains_both_langs(self):
        # 11/06: 指示文は英語(日本語の指示文に出力の言語が引っ張られた)
        sys = build_system(("ja", "en"), "トラック・アノテーション")
        self.assertIn("Japanese", sys)
        self.assertIn("English", sys)
        self.assertIn("トラック・アノテーション", sys)
        self.assertIn("Output only the", sys)

    def test_system_without_topic(self):
        sys = build_system(("ja", "ko"))
        self.assertNotIn("Topic", sys)

    def test_messages_shape(self):
        # 履歴は assistant ターンにせず「参考」ブロック1本へ畳む(11/05汚染対策)。
        # 11/06: 直近4〜7件を4件ごとにまとめてずらす(毎回ずらすと先頭が変わり、
        # サーバーの読み込みの使い回しが効かない)。8件なら4〜7番目だけ。
        hist = [{"lang": "ja", "orig": f"{i}番目の発話", "trans": f"utt{i}"} for i in range(8)]
        msgs = build_messages(("ja", "en"), "", hist)
        self.assertEqual(msgs[0]["role"], "system")
        self.assertEqual([m["role"] for m in msgs], ["system", "user"])
        joined = "\n".join(m["content"] for m in msgs)
        for i in (4, 5, 6, 7):
            self.assertIn(f"utt{i}", joined)
            self.assertIn(f"{i}番目の発話", joined)
        for i in (0, 1, 2, 3):
            self.assertNotIn(f"utt{i}", joined)
            self.assertNotIn(f"{i}番目の発話", joined)

    def test_prompt_prefix_is_stable_across_turns(self):
        # 交互に話しても、前の依頼の「指示文+履歴」がそのまま次の依頼の先頭になる
        hist = [{"lang": "ja", "orig": "こんにちは", "trans": "Hello"},
                {"lang": "en", "orig": "Nice to meet you", "trans": "はじめまして"}]
        m1 = build_messages(("ja", "en"), "YOLO", hist, "研究の話です", "ja")
        hist2 = hist + [{"lang": "ja", "orig": "研究の話です", "trans": "It's about research."}]
        m2 = build_messages(("ja", "en"), "YOLO", hist2, "How fast is it?", "en")
        self.assertEqual(m1[0], m2[0])                       # 指示文は向きで変わらない
        shared = m1[1]["content"].split("\n\n")[0]           # 履歴ブロック
        self.assertTrue(m2[1]["content"].startswith(shared))

    def test_messages_end_with_current(self):
        hist = [{"lang": "ja", "orig": "前の発話", "trans": "prev"}]
        msgs = build_messages(("ja", "en"), "", hist, "今の発話です")
        self.assertEqual([m["role"] for m in msgs], ["system", "user"])
        self.assertEqual(msgs[-1]["role"], "user")
        self.assertTrue(msgs[-1]["content"].endswith("今の発話です"))
        self.assertIn("prev", msgs[-1]["content"])  # 参考として履歴も載る

    def test_target_lang(self):
        self.assertEqual(target_lang(("ja", "en"), "ja"), "en")
        self.assertEqual(target_lang(("ja", "en"), "en"), "ja")
        self.assertIsNone(target_lang(("ja", "en"), "zh"))
        self.assertIsNone(target_lang(("ja", "en"), "ko"))

    def test_out_lang(self):
        # ペア内は相手側。ペア外はja(ユーザー側)にフォールバック
        self.assertEqual(out_lang(("ja", "en"), "ja"), "en")
        self.assertEqual(out_lang(("ja", "en"), "zh"), "ja")
        self.assertEqual(out_lang(("en", "zh"), "ko"), "en")
        self.assertEqual(out_lang(("zh", "ko"), "en"), "zh")

    def test_direction_fixed_at_the_end(self):
        # 検出言語から「何語へ訳すか」を固定する(11/05)。11/06から指示文ではなく
        # 今の発話の直前に書く(指示文を会話中ずっと同じにするため)。
        u = build_messages(("ja", "en"), "", [], "Hello", "en")[-1]["content"]
        self.assertIn("This utterance is in English. Translate it into Japanese.", u)
        u2 = build_messages(("ja", "en"), "", [], "你好", "zh")[-1]["content"]
        self.assertIn("This utterance is in Simplified Chinese. Translate it into Japanese.", u2)
        u3 = build_messages(("en", "zh"), "", [], "Hello", "en")[-1]["content"]
        self.assertIn("Translate it into Simplified Chinese.", u3)
        sys_ = build_system(("ja", "en"))
        self.assertIn("Japanese utterances go into English; English utterances go into Japanese", sys_)


class TestDetect(unittest.TestCase):
    def test_langs(self):
        self.assertEqual(detect_lang("これは予測モデルの話です"), "ja")
        self.assertEqual(detect_lang("We propose a tracking framework"), "en")
        self.assertEqual(detect_lang("我们提出了一个新的框架"), "zh")
        self.assertEqual(detect_lang("추적 모델을 제안합니다"), "ko")
        # 日本語文中に英単語が混ざってもja
        self.assertEqual(detect_lang("ReIDの特徴量について"), "ja")
        # ハングルが1文字でもあればko
        self.assertEqual(detect_lang("モデルの 추론 について"), "ko")
        self.assertEqual(detect_lang("12345 !!!"), "")

    def test_english_with_kana_noise(self):
        # whisperが英語音声からかなを混入させたケースは英語として扱う
        noisy = "We propose a robust multi-object tracking from you, 多く that handles occlusion."
        self.assertEqual(detect_lang(noisy), "en")

    def test_japanese_with_latin_terms(self):
        # 英専門用語が多めの学会日本語も日本語として扱う
        self.assertEqual(detect_lang("BERTのmask predictionの精度"), "ja")
        self.assertEqual(detect_lang("GPSのログ"), "ja")


class TestEcho(unittest.TestCase):
    def test_echo_detected(self):
        spoken = "次に、アノテーションの品質について議論しましょう。"
        heard = "次にアノテーションの品質について議論しましょう"  # 句読点違い
        self.assertTrue(is_echo(spoken, heard))

    def test_different_text_not_echo(self):
        self.assertFalse(is_echo("品質について議論しましょう", "完全に別の話題です"))

    def test_short_noise_ignored(self):
        self.assertFalse(is_echo("長い文章の読み上げです", "うん"))

    def test_norm(self):
        self.assertEqual(norm("あ、あ。あ！"), "あああ")


class TestLog(unittest.TestCase):
    def test_write_and_format(self):
        with tempfile.TemporaryDirectory() as d:
            log = SessionLog(d, "ja-en", "テスト")
            log.add("ja", "こんにちは", "Hello.", "en")
            log.close()
            lines = [json.loads(x) for x in Path(log.path_jsonl).read_text().splitlines()]
            self.assertEqual(lines[0]["type"], "meta")
            self.assertEqual(lines[0]["pair"], "ja-en")
            self.assertEqual(lines[1]["type"], "utter")
            self.assertEqual(lines[1]["lang"], "ja")
            self.assertEqual(lines[1]["trans"], "Hello.")
            txt = Path(log.path_txt).read_text()
            self.assertIn("ja→en", txt)
            self.assertIn("Hello.", txt)

    def test_subtitle_only(self):
        with tempfile.TemporaryDirectory() as d:
            log = SessionLog(d, "ja-en")
            log.add("zh", "你好", "こんにちは", None)
            log.close()
            self.assertIn("zh(字幕のみ)", Path(log.path_txt).read_text())


class TestInterrupt(unittest.TestCase):
    def _tts(self):
        # モデルディレクトリを空にすれば sherpa を読まず say のみの構成になる
        return TTS("", "", {"en": 3, "ja": 37, "zh": 45}, "say")

    def test_cancel_kills_say_proc(self):
        t = self._tts()

        class FakeProc:
            def __init__(self):
                self.killed = False

            def poll(self):
                return None  # まだ実行中

            def kill(self):
                self.killed = True

        p = FakeProc()
        t._proc = p
        t.cancel()
        self.assertTrue(t._stop.is_set())
        self.assertTrue(p.killed)
        self.assertFalse(p.poll() is None and not p.killed)

    def test_say_resets_stop_each_turn(self):
        # 一度中止しても次の発話で _stop が戻り、以降の読み上げが生き返る
        t = self._tts()
        t._stop.set()
        self.assertFalse(t.say("", "en"))  # 空テキストは読み上げない
        self.assertFalse(t._stop.is_set())

    def test_bridge_interrupt_without_tts(self):
        from tsuyaku.__main__ import Bridge
        b = Bridge()
        self.assertEqual(b.interrupt(), "ok")  # tts 以前でも落ちない
        self.assertTrue(b.interrupting.is_set())

    def test_main_warms_up_before_picking_model(self):
        """リモートサーバーのゲートはリクエストが来るまで起きない。/v1/models を叩いて
        接続失敗で終わらぬよう、先に warmup で起こす順序になっていること。"""
        import tsuyaku.__main__ as m
        self.assertTrue(callable(m.warmup))
        src = Path(m.__file__).read_text()
        # 処理場所の候補ごとに warmup → pick_model の順(ゲートを起こしてから選ぶ)
        self.assertLess(src.index("warmup(b, pref)"),
                        src.index("pick_model(b, pref)"))


if __name__ == "__main__":
    unittest.main()
