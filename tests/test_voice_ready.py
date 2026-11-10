"""回帰テスト: 声一覧は pywebview API の準備完了後に必ず組み直すこと。

11/05 実障害「voicevoxが選択できません」: index.html は `pywebviewloaded` を
聞いていたが、pywebview 6.2.1 が発火するのは `pywebviewready` (js/finish.js
で window.pywebview.api を作った直後に dispatch)。無名のイベントなので一度も
発火せず、唯一の救済だった 400ms 一回限りのタイマーも whisper 等の起動中は
API 注入が間に合わず不発 → ttsCfg が空構成のまま → 声一覧に VOICEVOX 41声
が載らず Kokoro しか出なかった(config.json は voicevox:13 を保持していた)。

教訓: ①イベント名はインストール済み pywebview の実物と突き合わせる、
②一回タイムアウトではなく API が出るまで有界ポーリング、
③ buildVoiceRows は opts が空でも落ちない(1言語行の例外がペア全体を殺す)。
"""
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HTML = (ROOT / "src" / "tsuyaku" / "webui" / "index.html").read_text()


def _installed_finish_js() -> str:
    import webview
    return (Path(webview.__file__).parent / "js" / "finish.js").read_text()


class TestEventName(unittest.TestCase):
    def test_html_does_not_listen_to_bogus_event(self):
        # 存在しないイベント名を聞いても一度も発火しない(今回の根因)。
        # コメントに名前を書くのは許すが、addEventListener は禁止。
        self.assertNotIn('"pywebviewloaded"', HTML)
        self.assertNotIn("'pywebviewloaded'", HTML)

    def test_html_listens_to_event_pywebview_actually_fires(self):
        fired = "pywebviewready"
        self.assertIn(f"'{fired}'", _installed_finish_js())
        self.assertIn(fired, HTML)


class TestPollUntilReady(unittest.TestCase):
    def test_boot_polls_until_api_appears(self):
        # 1発タイムアウトではなく、api() が出るまで有界ポーリングする
        self.assertIn("setTimeout(boot,400)", HTML)
        i = HTML.index("const boot=")
        seg = HTML[i:i + 300]
        self.assertIn("TSUYAKU.api()", seg)
        self.assertIn("ttsRefresh()", seg)
        self.assertIn("<75", seg)  # 無限ループ防止の上限(約30秒)


class TestEmptyOptsGuard(unittest.TestCase):
    def test_build_voice_rows_survives_empty_opts(self):
        # sayVoices も kokoro も無い言語(空構成+ko 等)で opts[0][0] が落ちると
        # 同じ行組み立ての相手言語行まで消える。空ならプレースホルダを1個置く。
        i = HTML.index("buildVoiceRows(sel)")
        seg = HTML[i:HTML.index("setTts(lang,voice,speed)")]
        self.assertIn("if(!opts.length)", seg)
        self.assertIn('cfg.voices[lang]||opts[0][0]', seg)


if __name__ == "__main__":
    unittest.main()
