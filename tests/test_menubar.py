"""macOS のメニューバー(11/08): 名前は Sokutsu、画面の言語、標準の並びとキー、どの項目も画面と同じ処理を呼ぶ。
⌘Q・Dock の「終了」・窓を閉じるは、どれも Bridge._shutdown を通る。窓は出さずに確かめる。"""
import re
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

from tsuyaku import menubar as M

ROOT = Path(__file__).resolve().parents[1]
HTML = (ROOT / "src" / "tsuyaku" / "webui" / "index.html").read_text(encoding="utf-8")
MAIN = (ROOT / "src" / "tsuyaku" / "__main__.py").read_text(encoding="utf-8")
DIST = (ROOT / "src" / "tsuyaku" / "dist_main.py").read_text(encoding="utf-8")


def menudo_block() -> str:
    i = HTML.index("T.menuDo=function(id){")
    return HTML[i:HTML.index("\n};", i)]


class TestSpec(unittest.TestCase):
    def test_every_title_in_four_languages(self):
        for key, title, items in M.MENUS:
            self.assertEqual(set(title), set(M.LANGS), key)
        for it in M.walk():
            self.assertEqual(set(it["t"]), set(M.LANGS), it["id"])
            for lang in M.LANGS:
                self.assertTrue(it["t"][lang].strip(), (it["id"], lang))

    def test_standard_order(self):
        self.assertEqual([k for k, _, _ in M.MENUS], ["app", "file", "edit", "view", "conv", "window", "help"])
        app = [it["id"] for it in M._walk_items(M.MENUS[0][2])]
        self.assertEqual(app, ["about", "settings", "services", "hide", "hideOthers", "showAll", "quit"])

    def test_no_duplicate_shortcuts(self):
        seen = {}
        for it in M.walk():
            if not it["key"]:
                continue
            k = (it["key"].lower(), it["mods"])
            self.assertTrue(k not in seen or seen[k] == it["id"], f"{M.key_label(it)}: {seen.get(k)} / {it['id']}")
            seen[k] = it["id"]

    def test_shortcuts_match_the_web_ui(self):
        want = {"newConv": "⌘N", "settings": "⌘,", "find": "⌘F", "sidebar": "⌘\\", "palette": "⌘K", "start": "⌘↩",
                "quit": "⌘Q", "hide": "⌘H", "hideOthers": "⌥⌘H", "close": "⌘W", "minimize": "⌘M",
                "fullScreen": "⌃⌘F", "undo": "⌘Z", "redo": "⇧⌘Z", "copy": "⌘C", "paste": "⌘V"}
        got = {it["id"]: M.key_label(it) for it in M.walk()}
        for k, v in want.items():
            self.assertEqual(got[k], v, k)
        # 画面のキー(index.html)も同じ: ⌘N / ⌘, / ⌘F / ⌘K / ⌘↩ / ⌘\
        for js in ('if(k==="k")', 'if(k==="n"&&!e.shiftKey)', 'if(e.key===",")', 'if(k==="f"&&!e.shiftKey', 'e.key==="Enter"&&!st.classList.contains("hidden")',
                   '(e.key==="\\\\"||e.code==="Backslash")'):
            self.assertIn(js, HTML)

    def test_single_letter_keys_are_only_hints(self):
        # B・M・V・Space は文字の入力を横取りしないよう、メニューのキーにしない(題に添える)
        for it in M.walk():
            if it["key"]:
                self.assertTrue(it["mods"] & (M.CMD | M.CTRL), it["id"])
        titles = {it["id"]: it["t"]["ja"] for it in M.walk()}
        self.assertIn("(B)", titles["bookmark"])
        self.assertIn("(M)", titles["note"])
        self.assertIn("(V)", titles["voice"])
        self.assertIn("(Space)", titles["stopVoice"])

    def test_every_js_action_is_wired_to_existing_functions(self):
        blk = menudo_block()
        ids = {it["id"] for it in M.walk() if it["kind"] == "js"}
        for i in ids:
            self.assertRegex(blk, r"\b" + re.escape(i) + r":\(\)=>", i)
        # menuDo は既存の関数・ボタンを呼ぶだけ(論理を二重に持たない)
        for fn in re.findall(r"this\.(\w+)\(", blk):
            self.assertRegex(HTML, r"\n  " + fn + r"\(|T\." + fn + r"=function|\b" + fn + r"\([^)]*\)\{", fn)
        for sel in re.findall(r'\$\("(#[\w-]+)', blk):
            self.assertIn('id="' + sel[1:] + '"', HTML.replace("b.id=\"", 'id="').replace("f.id=\"", 'id="'), sel)

    def test_enabled_rules(self):
        items = {it["id"]: it for it in M.walk()}
        idle = {"live": False, "start": True, "doc": False, "layer": False, "speaking": False}
        live = {"live": True, "start": False, "doc": False, "layer": False, "speaking": True}
        self.assertTrue(M.enabled(items["newConv"], idle))
        self.assertFalse(M.enabled(items["newConv"], live))
        self.assertFalse(M.enabled(items["endConv"], idle))
        self.assertTrue(M.enabled(items["endConv"], live))
        self.assertFalse(M.enabled(items["export"], idle))
        self.assertTrue(M.enabled(items["export"], dict(idle, doc=True)))
        self.assertFalse(M.enabled(items["close"], idle))
        self.assertTrue(M.enabled(items["stopVoice"], live))
        self.assertTrue(M.enabled(items["start"], idle))
        self.assertFalse(M.enabled(items["start"], live))
        self.assertTrue(M.checked(items["themeDark"], {"theme": "dark"}))
        self.assertFalse(M.checked(items["themeLight"], {"theme": "dark"}))


class TestBuiltMenu(unittest.TestCase):
    """窓を出さずに NSMenu を作り、中身を書き出して確かめる。"""

    def build(self, lang, state=None):
        mb = M.MenuBar(None, lang)
        mb.state.update(state or {})
        mb.target = M._target_class().alloc().initWithOwner_(mb)
        menu, refs = M.build_menu(lang, mb.target, mb.state)
        return mb, menu, refs

    def test_top_level_titles_per_language(self):
        for lang, tops in {"ja": ["Sokutsu", "ファイル", "編集", "表示", "会話", "ウインドウ", "ヘルプ"],
                           "en": ["Sokutsu", "File", "Edit", "View", "Conversation", "Window", "Help"],
                           "zh": ["Sokutsu", "文件", "编辑", "显示", "对话", "窗口", "帮助"],
                           "ko": ["Sokutsu", "파일", "편집", "보기", "대화", "윈도우", "도움말"]}.items():
            _, menu, refs = self.build(lang)
            self.assertEqual([str(menu.itemAtIndex_(i).title()) for i in range(menu.numberOfItems())], tops)
            self.assertIsNotNone(refs["services"])
            self.assertIsNotNone(refs["window"])

    def test_dump_shows_state(self):
        mb, menu, _ = self.build("en", {"theme": "dark", "live": False, "start": True, "doc": False})
        d = M.dump(menu)
        self.assertIn("About Sokutsu", d)
        self.assertIn("Quit Sokutsu   ⌘Q", d)
        self.assertIn("✓ Dark", d)
        self.assertIn("New Conversation   ⌘N", d)
        self.assertIn("End Conversation  (使えない)", d)
        self.assertIn("Start Conversation   ⌘↩", d)
        self.assertNotIn("Start Conversation   ⌘↩  (使えない)", d)

    def test_language_change_rebuilds(self):
        mb, menu, _ = self.build("ja")
        mb.menu = menu
        calls = []
        with patch.object(M, "_main_thread", lambda fn: calls.append(fn.__name__)):
            mb.set_state({"ui": "en"})
            mb.set_state({"ui": "en", "theme": "light"})
        self.assertEqual(calls, ["install", "refresh"])
        self.assertEqual(mb.lang, "en")

    def test_js_actions_go_through_evaluate_js_off_the_main_thread(self):
        class W:
            def __init__(self):
                self.js, self.thread = [], None

            def evaluate_js(self, js):
                self.js.append(js)
                self.thread = threading.current_thread()
        w = W()
        mb = M.MenuBar(w, "ja")
        mb.run("newConv")
        for t in threading.enumerate():
            if t is not threading.main_thread() and t.daemon:
                t.join(1)
        self.assertEqual(w.js, ['window.TSUYAKU&&TSUYAKU.menuDo&&TSUYAKU.menuDo("newConv")'])
        self.assertIsNot(w.thread, threading.main_thread())


class TestQuitAndStartup(unittest.TestCase):
    def test_quit_and_close_run_the_shutdown_path(self):
        from webview.event import Event

        class Ev:
            pass

        class Win:
            def __init__(self):
                self.events = Ev()
                self.events.closing = Event(self, True)
                self.events.shown = Event(self)

        from tsuyaku.__main__ import Bridge
        b = Bridge.__new__(Bridge)
        called = []
        b._shutdown = lambda *a, **k: called.append(1)
        w = Win()
        M.attach(b, w)
        cancel = w.events.closing.set()    # pywebview: ⌘Q(terminate:)・Dock の終了・窓を閉じる
        self.assertEqual(called, [1])
        self.assertFalse(cancel)            # 閉じるのを止めない

    def test_on_closing_is_the_shutdown(self):
        k = MAIN.index("    def _on_closing(self")
        body = MAIN[k:MAIN.index("\n    def ", k + 10)]
        self.assertIn("self._shutdown()", body)

    def test_name_and_icon_before_start(self):
        for src in (MAIN[MAIN.index("def main():"):], DIST[DIST.index("def _main()"):]):
            self.assertLess(src.index("prepare_app("), src.index("webview.start("))
            self.assertLess(src.index("attach("), src.index("webview.start("))
            self.assertLess(src.index("create_window("), src.index("attach("))

    def test_prepare_app_sets_bundle_name(self):
        from Foundation import NSBundle
        self.assertTrue(M.prepare_app())
        self.assertEqual(NSBundle.mainBundle().infoDictionary()["CFBundleName"], "Sokutsu")

    def test_bridge_menu_state(self):
        from tsuyaku.__main__ import Bridge
        b = Bridge.__new__(Bridge)
        got = []

        class MB:
            def set_state(self, s):
                got.append(s)
        b._menubar = MB()
        self.assertEqual(b.menu_state({"ui": "ko"}), "ok")
        self.assertEqual(got, [{"ui": "ko"}])
        b2 = Bridge.__new__(Bridge)
        self.assertEqual(b2.menu_state({"ui": "ko"}), "ok")   # メニューが無くても落ちない

    def test_web_ui_reports_state(self):
        self.assertIn("a.menu_state(s)", HTML)
        for k in ("ui:", "live:", "start:", "doc:", "speaking:", "layer:", "theme:"):
            self.assertIn(k, HTML[HTML.index("T.menuSync=function(){"):HTML.index("T.menuSync=function(){") + 900])


if __name__ == "__main__":
    unittest.main()
