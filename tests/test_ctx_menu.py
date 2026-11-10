"""右クリックのメニュー(11/08 ユーザー:「右クリックでの操作も入れてください。例えば、タスクを削除するなど」)。
会議メモの項目・話題・章・全文の記録の行・会話中の記録と字幕・サイドバーの会議・ゴミ箱・フォルダ・タグ・参加者の札。
入力欄では Mac のメニューのまま、それ以外では WKWebView の「再読み込み」などを出さない。消すと「元に戻す」。"""
import re
import unittest
from pathlib import Path

HTML = (Path(__file__).resolve().parents[1] / "src" / "tsuyaku" / "webui" / "index.html").read_text(encoding="utf-8")


def block():
    i = HTML.index('<script id="ctx-js">')
    return HTML[i:HTML.index("</script>", i)]


def css():
    i = HTML.index("/*CTX-CSS-BEGIN")
    return HTML[i:HTML.index("/*CTX-CSS-END*/", i)]


class TestTargets(unittest.TestCase):
    def test_every_target_has_a_menu(self):
        js = block()
        for sel in ('#mdoc .doc .todo', '#mdoc .doc .topic .pt', '.nsec[data-k=\\"s-sum\\"] ul.bl>li', '.callout>li', '.qs>li',
                    '#mdoc .doc .topic>h3', '#mdoc .doc .chap>li', '#mdoc .doc .tln[data-i]', '#mdoc .doc .person',
                    '#log .item', '#stage', '#mlist .mi[data-id]', '#mlist .w4tri[data-id]',
                    'data-w4scope=\\"folder\\"', 'data-w4scope=\\"tag\\"'):
            self.assertIn(sel, js, sel)

    def test_items_per_target(self):
        js = block()
        todo = js[js.index("function itemMenu"):js.index("function topicMenu")]
        for k in ("未完了に戻す", "完了にする", "編集", "担当者を変える", "期限", "リマインダーへ", "元の発言へ", "コピー", "削除"):
            self.assertIn(f'"{k}"', todo, k)
        line = js[js.index("function lineMenu"):js.index("function askAbout")]
        for k in ("コピー", "文を直す", "話者を変える", "印を付ける", "ここから再生", "この発言について質問する"):
            self.assertIn(f'"{k}"', line, k)
        mt = js[js.index("function meetingMenu"):js.index("function trashMenu")]
        for k in ("開く", "名前を変える", "お気に入りに入れる", "フォルダへ移動", "タグ", "書き出す", "ゴミ箱へ移す"):
            self.assertIn(f'"{k}"', mt, k)
        tr = js[js.index("function trashMenu"):js.index("function scopeMenu")]
        self.assertIn('"元に戻す"', tr)
        self.assertIn('"完全に削除"', tr)
        sp = js[js.index("function speakerMenu"):js.index("const SEL=")]
        self.assertIn('"名前を変える"', sp)
        self.assertIn('"声を覚える"', sp)

    def test_destructive_items_are_last_and_red(self):
        js = block()
        self.assertIn('"-",del]', js)                      # 区切りの下、いちばん下
        self.assertIn("danger:true", js)
        self.assertIn(".cx.danger", css())
        self.assertIn("var(--err)", css())


class TestBackend(unittest.TestCase):
    def test_notes_items_use_bridge(self):
        js = block()
        for fn in ("notes_item_edit", "notes_item_delete", "notes_item_restore"):
            self.assertIn(f'"{fn}"', js)
        from tsuyaku.__main__ import Bridge
        for fn in ("notes_item_edit", "notes_item_delete", "notes_item_restore", "meeting_trash", "meeting_restore",
                   "trash_delete", "folder_rename", "folder_remove", "tag_rename", "meeting_info_set", "meeting_export",
                   "reminders_add", "speaker_rename"):
            self.assertTrue(callable(getattr(Bridge, fn, None)), fn)

    def test_permanent_delete_confirms_others_undo(self):
        js = block()
        tr = js[js.index("function trashMenu"):js.index("function scopeMenu")]
        self.assertIn("confirm:{", tr)                     # 完全に削除だけ確かめる
        self.assertIn('api("trash_delete",[id],1)', tr)
        self.assertNotIn("confirm(", js.replace("confirm:{", ""))   # window.confirm は使わない
        for undo in ('api("notes_item_restore"', 'api("meeting_restore",id)', 'api("folder_add",v)'):
            self.assertIn(undo, js)
        self.assertIn("T.undoToast=function(msg,undo){", js)
        self.assertIn("6000", js[js.index("T.undoToast"):js.index("// ---------- メニューの箱")])


class TestBehaviour(unittest.TestCase):
    def test_system_menu_kept_in_text_fields(self):
        js = block()
        self.assertIn('document.addEventListener("contextmenu"', js)
        self.assertIn("isContentEditable", js)
        self.assertIn("INPUT|TEXTAREA|SELECT", js)
        h = js[js.index('document.addEventListener("contextmenu"'):]
        self.assertLess(h.index("return;"), h.index("e.preventDefault()"))   # 入力欄なら preventDefault の前に戻る

    def test_does_not_stack_on_other_popovers(self):
        js = block()   # 聞こえのメーター(hear)・…メニューのポップオーバーは閉じてから開く
        self.assertIn("this.hdClose&&this.hdClose(); this.w4popClose&&this.w4popClose();", js)

    def test_templates_are_filled(self):
        self.assertIn("const L=(k,v)=>T.L(k,v);", block())   # 「{t}」を削除しました の {t} を埋める(11/08 撮って見つけた)

    def test_keyboard(self):
        js = block()
        for k in ('"ArrowDown"', '"ArrowUp"', '"Escape"', '"Home"', '"End"', '"ArrowRight"', '"ArrowLeft"', '"F10"', '"ContextMenu"'):
            self.assertIn(k, js, k)
        self.assertIn('role","menu"', js)
        self.assertIn('role="menuitem"', js)
        self.assertIn("r.focus({preventScroll:true})", js)   # 閉じたら元の場所へ

    def test_closes_and_stays_inside_window(self):
        js = block()
        self.assertIn('document.addEventListener("wheel"', js)     # 人のスクロールで閉じる(自動の送りでは閉じない)
        self.assertIn('window.addEventListener("blur"', js)
        self.assertIn("if(x+w>W-8)x=Math.max(8,x-w)", js)    # 端では反対側へ
        self.assertIn("if(y+h>H-8)y=Math.max(8,H-h-8)", js)

    def test_dots_button_on_hover_and_focus(self):
        js = block()
        self.assertIn('d.id="ctxDots"', js)
        self.assertIn('document.addEventListener("mouseover"', js)
        self.assertIn('document.addEventListener("focusin"', js)
        self.assertIn("#ctxDots", css())


class TestStrings(unittest.TestCase):
    def test_four_languages(self):
        js = block()
        keys = set(re.findall(r'L\("([^"]+)"', js))
        for a in re.findall(r'L\(([^()]*\?[^()]*)\)', js):      # L(on?"…":"…")
            keys |= set(re.findall(r'"([^"]+)"', a))
        keys -= {"live", "ok", "denied"}
        self.assertGreater(len(keys), 40)
        for k in sorted(keys):
            self.assertGreaterEqual(HTML.count(f'"{k}":'), 3, f"訳が足りない: {k}")
        for k in ("未完了に戻す", "完了にする", "印を外す", "印を付ける", "お気に入りから外す", "お気に入りに入れる", "たたむ", "開く"):
            self.assertGreaterEqual(HTML.count(f'"{k}":'), 3, k)


if __name__ == "__main__":
    unittest.main()
