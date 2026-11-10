"""聞こえの見える化・入出力の切り替え(11/08 hear)の画面: Python との約束(CONTRACT)と、文言の4言語。

ユーザー「聞こえているのかUIですぐにわかるようにしてください。例えばマイクボリュームを視覚的にわかりやすく
表示する、ソフトの周りを赤くするなどなど。あとソフト上で入力・出力デバイスを切り替えられるようにしてください。」
"""
import re
import unittest
from pathlib import Path

HTML = (Path(__file__).resolve().parents[1] / "src/tsuyaku/webui/index.html").read_text(encoding="utf-8")
A = HTML.index("<!-- ===== hear(11/08)")
B = HTML.index("<!-- ===== /hear -->")
BLOCK = HTML[A:B]
CSS = BLOCK[BLOCK.index('<style id="hear-css">'):BLOCK.index("</style>")]
JS = BLOCK[BLOCK.index('<script id="hear-js">'):]
DI, DJ = JS.index("/*HEAR-I18N-BEGIN*/"), JS.index("/*HEAR-I18N-END*/")
DICT = JS[DI:DJ]
CODE = JS[DJ:]

STATES = ("quiet", "speech", "work", "speak", "speak_listen", "nomic", "lost")


class TestHearContract(unittest.TestCase):
    def test_block_is_one_separate_piece_at_the_end(self):
        # aud-ui(右クリックのメニュー)と同時に直すので、塊は1つで </body> の直前
        self.assertEqual(HTML.count("<!-- ===== hear(11/08)"), 1)
        self.assertLess(B, HTML.index("</body>"))
        self.assertEqual(HTML[B:].strip().replace("<!-- ===== /hear -->", "").strip()[:7], "</body>")

    def test_python_events_have_handlers(self):
        # Python は window.TSUYAKU&&TSUYAKU.hear&&TSUYAKU.hear(obj) / TSUYAKU.audioDev(obj) で呼ぶ
        self.assertIn("  hear(d){", CODE)            # Object.assign(T,{hear(d){…}})
        self.assertIn("  audioDev(d){", CODE)
        for st in STATES:
            self.assertIn('"' + st + '"', CODE, st)
        self.assertIn("d.test", CODE)     # 起動画面のマイクの確認

    def test_every_api_call_is_guarded(self):
        names = set(re.findall(r'\ba\.([a-z]+_[a-z_]+)\(', CODE))
        self.assertTrue({"audio_devices", "set_input_device", "set_output_device", "mic_test",
                         "set_edge_glow", "open_mic_privacy"} <= names, names)
        for n in names:
            self.assertTrue(f'typeof a.{n}==="function"' in CODE, f"古い Python で落ちないよう確かめる: {n}")

    def test_device_menu_refreshes_on_open(self):
        i = CODE.index("hdMenu(")
        body = CODE[i:i + 1500]
        self.assertIn("hdDevices(", body)          # 開くたびに一覧を取り直す
        j = CODE.index("hdDevices(){")
        self.assertIn("audio_devices()", CODE[j:j + 400])
        self.assertIn("システムの設定に合わせる", CODE)

    def test_mic_test_only_while_start_page_visible(self):
        i = CODE.index("hearTestSync(")
        body = CODE[i:i + 900]
        for s in ("this.live", "#start", "#meet", "#setp", "document.hidden", "mic_test("):
            self.assertIn(s, body, s)
        self.assertIn('"#go"', CODE)            # 会話を始めたら止める
        self.assertIn("visibilitychange", CODE)

    def test_live_meter_next_to_status_and_opens_popover(self):
        self.assertIn('id="hearMeter"', CODE)
        self.assertIn("#statusText", CODE)
        self.assertIn("hearPop(", CODE)

    def test_output_switch_says_next_reading(self):
        self.assertIn("次の読み上げから", CODE)
        self.assertIn('applied==="next"', CODE)

    def test_glow_setting_wired(self):
        self.assertIn('id="hearEdge"', CODE)
        self.assertIn("hear-noglow", CODE + CSS)
        self.assertIn("edgeGlow", CODE)              # 起動時の値は get_tts().edgeGlow
        self.assertIn("set_edge_glow(", CODE)
        self.assertIn("#sgUi", CODE)                 # 設定のページ(画面)に置く

    def test_no_input_banner_opens_privacy(self):
        self.assertIn("マイクの許可を開く", CODE)
        self.assertIn("open_mic_privacy()", CODE)

    def test_reduced_motion(self):
        self.assertIn("prefers-reduced-motion:reduce", CSS.replace(" ", ""))
        self.assertIn("matchMedia(\"(prefers-reduced-motion: reduce)\")", CODE)

    def test_both_themes_have_colors(self):
        self.assertIn('[data-theme="dark"]', CSS)
        self.assertIn("prefers-color-scheme:dark", CSS.replace(" ", ""))

    def test_every_new_text_is_translated(self):
        keys = set(re.findall(r'(?:this|T)\.L\("([^"]+)"', CODE))
        keys |= set(re.findall(r'data-t(?:="")?>([^<\'"+]+)<', CODE))
        keys |= set(re.findall(r'\brow\("([^"]+)"', CODE))
        for lab, note in re.findall(r'\bsrow\("#\w+","([^"]+)","([^"]*)"', CODE):
            keys |= {lab, note} - {""}
        keys |= set(re.findall(r'\bLB\("([^"]+)"', CODE))
        self.assertGreater(len(keys), 20)
        for k in keys:
            self.assertGreaterEqual(DICT.count('"' + k + '":'), 3, f"訳が足りない(en/zh/ko): {k}")


if __name__ == "__main__":
    unittest.main()
