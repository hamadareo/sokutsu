"""会話中も設定を開ける(11/08 ユーザー:「読み上げの有無をセッション中でも設定から切り替えられるようにして」)。
会話は止めずに、右に設定の板を出す。すぐ効くもの(読み上げ・声と速さ・マイク/スピーカー・縁の光・明るさ・
画面の言語・字幕の大きさ・投影モード)は触れる。会話の最初に決まるもの(翻訳の処理場所・まわりの音・録音)は
「次の会話から変えられます」と添えて押せなくする。声・速度の板のいちばん上にも読み上げの切り替え。"""
import json
import unittest
from pathlib import Path

from tsuyaku import menubar as M

HTML = (Path(__file__).resolve().parents[1] / "src" / "tsuyaku" / "webui" / "index.html").read_text(encoding="utf-8")


def block():
    i = HTML.index('<script id="liveset-js">')
    return HTML[i:HTML.index("</script>", i)]


class TestOpenDuringSession(unittest.TestCase):
    def test_cmd_comma_works_while_live(self):
        i = HTML.index("// 5) キー: ⌘K")
        h = HTML[i:HTML.index("},true);", i)]
        self.assertLess(h.index('if(e.key===","){e.preventDefault();T.setOpen();return;}'), h.index("if(T.live) return;"))

    def test_menubar_settings_enabled_live(self):
        items = {it["id"]: it for it in M.walk()}
        live = {"live": True, "start": False, "doc": False, "layer": False, "speaking": True}
        self.assertTrue(M.enabled(items["settings"], live))
        self.assertTrue(M.enabled(items["settings"], {"live": False}))

    def test_side_panel_does_not_cover_or_stop(self):
        js = block()
        self.assertIn("body.live #setp{", HTML)                       # 右の板(字幕は左に見えたまま)
        self.assertIn('document.body.classList.remove("sb-live")', js)  # 重ねて開いたサイドバーは閉じる
        self.assertIn("会話は続いています", js)
        # 開いている間、会話中の一字のキー(Esc で終える・Space・V・B・M)を設定の外へ流さない。Esc は設定を閉じるだけ
        self.assertIn('window.addEventListener("keydown"', js)
        self.assertIn("e.stopImmediatePropagation(); T.setClose();", js)


class TestLiveAndFixedRows(unittest.TestCase):
    def test_fixed_rows_are_locked_with_note(self):
        js = block()
        for sel in ("#engines", "#noises", "#w2rec"):
            self.assertIn(f'"{sel}"', js, sel)
        self.assertIn("次の会話から変えられます", js)
        for k in ("次の会話から変えられます", "会話は続いています — ここで変えたものはすぐ効きます", "字幕の大きさ"):
            self.assertGreaterEqual(HTML.count(f'"{k}":'), 3, k)

    def test_caption_size_row(self):
        js = block()
        self.assertIn('srow("#sgUi","字幕の大きさ"', js)
        self.assertIn("T.capSet(T.cap-0.1)", js)
        self.assertIn("T.capSet(T.cap+0.1)", js)


class TestOneVoiceSwitch(unittest.TestCase):
    def test_voice_panel_has_switch_on_top(self):
        js = block()
        self.assertIn('id="voiceOnLive"', js)
        self.assertIn('box.insertBefore(vs,box.querySelector("#voiceRows2"))', js)
        self.assertIn("T.voiceSet(x.dataset.voice===\"on\",true)", js)

    def test_settings_switch_uses_voice_set(self):
        i = HTML.index('const vo=document.querySelectorAll("#voiceOn .pair");')
        h = HTML[i:i + 400]
        self.assertIn("this.voiceSet(b.dataset.voice===\"on\",!!this.live)", h)

    def test_all_switches_synced(self):
        self.assertIn("#voiceOnLive .pair", block())


if __name__ == "__main__":
    unittest.main()


class TestPaceRows(unittest.TestCase):
    """pace(11/08)の 自動の速さ・聞きやすさ/速さ優先・読み上げ中も聞く は、Python が読み上げのたびに読む
    (_auto_speed / _voice_speed / _listen_how)ので、会話中の設定でも「すぐ効く」行として出す。"""

    def test_rows_in_settings_not_locked(self):
        js = block()
        self.assertIn('prow("読み上げ中も聞く"', js)
        self.assertIn('prow("自動の速さ"', js)
        self.assertIn('#voiceRows>.vlisten', js)        # 声と速さの欄から自分の行へ移す(声・速度の板はそのまま)
        fixed = js[js.index("const FIXED="):js.index("\n", js.index("const FIXED="))]
        self.assertNotIn("vlisten", fixed)

    def test_python_reads_them_per_utterance(self):
        src = (Path(__file__).resolve().parents[1] / "src" / "tsuyaku" / "__main__.py").read_text(encoding="utf-8")
        self.assertIn('self.pace.priority = getattr(self.s, "tts_speed_priority", "clear")', src)   # _voice_speed の中
        self.assertIn('mode = getattr(self.s, "listen_while_speaking", "auto")', src)              # _listen_how の中


class TestVoiceNote(unittest.TestCase):
    """起動画面の読み上げの説明は、読み上げ中も聞くかどうかで変える(いつも「マイクを止める」と書いていた)。"""

    def test_note_depends_on_listen_state(self):
        self.assertIn("voiceNoteKey(){", HTML)
        i = HTML.index("voiceNoteKey(){")
        f = HTML[i:HTML.index("\n  },", i)]
        for k in ("listenWhileSpeaking", "outputPrivate", 'this.src==="mic+system"'):
            self.assertIn(k, f)
        self.assertIn("this.VOICE_NOTE[v.dataset.voice===\"on\"?this.voiceNoteKey():\"off\"]", HTML)
        self.assertIn("if(this.ttsCfg)this.ttsCfg.listenWhileSpeaking=ls.value; this.segNotes();", HTML)

    def test_new_notes_translated(self):
        import re
        i = HTML.index("VOICE_NOTE:{")
        body = HTML[i:HTML.index("},", i)]
        texts = re.findall(r':"([^"]+)"', body)
        self.assertGreaterEqual(len(texts), 5)
        for t in texts:
            self.assertGreaterEqual(HTML.count(f'"{t}":'), 3, t)
