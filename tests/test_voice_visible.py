"""読み上げが切れている(「字幕だけ」)のが見えるように(11/08 ユーザー:「音声が出ません」。設定 voice_on=false が
11/07 から残っていたが、新しい作りでは設定のページにしか出ず、会話を始めても声が出ない理由が分からなかった)。"""
import re
import unittest
from pathlib import Path

HTML = (Path(__file__).resolve().parents[1] / "src" / "tsuyaku" / "webui" / "index.html").read_text(encoding="utf-8")


def block():
    i = HTML.index('<script id="voice-js">')
    return HTML[i:HTML.index("</script>", i)]


class TestVoiceVisible(unittest.TestCase):
    def test_start_page_has_voice_row(self):
        js = block()
        self.assertIn('row("読み上げ",[vs],"r-voice")', js)            # 新しい会話のプロパティの行
        self.assertIn('data-voice="on"', js)
        self.assertIn('data-voice="off"', js)
        self.assertIn('const b=document.querySelector(\'#voiceOn .pair[data-voice="\'+x.dataset.voice+\'"]\'); if(b)b.click();', js)  # 設定と同じ処理

    def test_one_toggle_for_v_key_menu_and_marker(self):
        js = block()
        self.assertIn("T.voiceSet=function(on,toast){", js)
        self.assertIn("TSUYAKU.voiceSet(on,true)", HTML)              # V キー
        self.assertIn('voice:()=>{const on=!(document.querySelector(\'#voiceOn .pair.sel\')?.dataset.voice==="on"); this.voiceSet(on,true);}', HTML)  # メニュー
        self.assertIn('this.toast2(this.L(on?"声を出します":"字幕だけにしました"))', js)

    def test_live_marker(self):
        js = block()
        self.assertIn("字幕だけ(声なし) — V で声を出す", js)
        self.assertIn("T.voiceSet(true,true)", js)                    # 押すと声を出す
        self.assertIn('t.hidden=!(this.live&&!on&&this.mode!=="notes");', js)

    def test_sync_with_settings(self):
        js = block()
        for n in ("voiceMark", "segNotes", "setMode", "status", "reset"):
            self.assertIn(f'"{n}"', js)

    def test_strings_in_four_languages(self):
        for k in ("字幕だけ(声なし) — V で声を出す", "声を出します", "字幕だけにしました", "読み上げ"):
            self.assertGreaterEqual(HTML.count(f'"{k}":'), 3, k)


if __name__ == "__main__":
    unittest.main()
