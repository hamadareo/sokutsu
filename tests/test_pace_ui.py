"""画面: 話のペースに合わせる「自動」・読み上げ中も聞く・飛ばした印・自動の速さ(11/08)。"""
import re
import unittest
from pathlib import Path

HTML = (Path(__file__).resolve().parents[1] / "src/tsuyaku/webui/index.html").read_text()
KEYS = ["自動", "話のペースに合わせる", "×{n} 自動", "読み上げを飛ばしました", "読み上げ中も聞く",
        "いつも", "止める", "自動: ヘッドホンなら聞き続けます",
        "聞きやすさ優先", "速さ優先", "自動の速さ",
        "Mac で流している動画や会議の声を訳すときは『マイク+この Mac の音』を選ぶと、読み上げ中も聞き逃しません",
        "読み上げ中も聞いています — 話してかまいません", "マイク休止中 — この Mac の音は聞いています"]


class TestPaceUI(unittest.TestCase):
    def test_controls_call_python(self):
        self.assertIn("a.set_tts_mode(lang,on?\"auto\":\"fixed\")", HTML)
        self.assertIn("a.set_listen(ls.value)", HTML)

    def test_voice_marks_skip_and_auto_speed(self):
        i = HTML.index("  voice(d){")
        seg = HTML[i:i + 1500]
        self.assertIn("d.skipped", seg)
        self.assertIn("読み上げを飛ばしました", seg)
        self.assertIn("×{n} 自動", seg)

    def test_all_four_languages(self):
        for lang in ("en", "zh", "ko"):
            i = HTML.index(f"\n{lang}:{{")
            block = HTML[i:i + 60000]
            for k in KEYS:
                self.assertRegex(block, re.escape(f'"{k}":"'), f"{lang} に {k} の訳が無い")

    def test_python_api(self):
        from tsuyaku.__main__ import Bridge
        b = Bridge()
        self.assertTrue(hasattr(b, "set_tts_mode") and hasattr(b, "set_listen"))
        cfg = b.get_tts()
        self.assertIn("speedModes", cfg)
        self.assertIn("listenWhileSpeaking", cfg)


    def test_speed_priority_control(self):
        self.assertIn("a.set_tts_priority(pr.value)", HTML)
        from tsuyaku.__main__ import Bridge
        b = Bridge()
        self.assertIn(b.get_tts().get("speedPriority"), ("clear", "fast"))
        self.assertIn("outputPrivate", b.get_tts())

    def test_mic_only_guidance(self):
        """「マイクだけ」+ 声あり + 内蔵スピーカー(outputPrivate === false)のときだけ案内を出す。"""
        i = HTML.index("  paceSrcNote(){")
        seg = HTML[i:i + 1200]
        self.assertIn('this.src==="mic"', seg)
        self.assertIn("outputPrivate===false", seg)
        self.assertIn("this.paceSrcNote()", HTML[HTML.index("  srcWarnShow(){"):][:900])

    def test_old_warning_only_when_listening_stops(self):
        """「マイク+この Mac の音」の『読み上げ中は相手の声も聞けません』は、止める設定のときだけ。"""
        i = HTML.index("  srcWarnShow(){")
        self.assertIn('listenWhileSpeaking', HTML[i:i + 900])

    def test_bottom_bar_when_listening(self):
        self.assertIn('classList.toggle("speakListen"', HTML)
        self.assertIn("body.speakListen #micNote", HTML)
        from tsuyaku.__main__ import Bridge
        import types
        b = Bridge()
        got = []
        b.window = types.SimpleNamespace(evaluate_js=got.append)
        b._push(("status", "speak", "読み上げ中 (日本語)", {"listen": "mic"}))
        self.assertTrue(got and got[0].endswith(',{"listen": "mic"})'), got)
        got.clear()
        b._push(("status", "speak", "読み上げ中 (日本語)"))
        self.assertTrue(got[0].endswith('"読み上げ中 (日本語)")'), got)


if __name__ == "__main__":
    unittest.main()
