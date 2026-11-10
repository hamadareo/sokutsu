"""上のタブバー(11/10 ユーザー「ノーションのように上にタブバーがあるといい」)。
先頭の固定タブ「通訳」+ 開いた会議の記録のタブ + 右端の「＋」。タブの切替で会話は止めない(会議の記録は会話の上に重なるだけ)。
タブの状態(開く・閉じる・複製・並べ替え・保存/復元)は純粋なモデルにして osascript(JXA)で動かして確かめる。"""
import json
import re
import subprocess
import unittest
from pathlib import Path

HTML = (Path(__file__).resolve().parents[1] / "src" / "tsuyaku" / "webui" / "index.html").read_text(encoding="utf-8")


def section(a, b):
    i = HTML.index(a)
    return HTML[i:HTML.index(b, i)]


def model_src():
    return section("/*TABS-MODEL-BEGIN*/", "/*TABS-MODEL-END*/")


def run_js(body):
    code = ("var f=new Function(%s);" % json.dumps(model_src() + "\n" + body)) + "JSON.stringify(f());"
    out = subprocess.run(["osascript", "-l", "JavaScript", "-e", code], capture_output=True, text=True, timeout=30)
    assert out.returncode == 0, out.stderr
    return json.loads(out.stdout.strip())


class TestModel(unittest.TestCase):
    def test_open_focus_existing_and_insert_after_active(self):
        r = run_js("""const m=mkTabs(30); const a=m.open('A','a'), b=m.open('B','b'); m.activate(a); const c=m.open('C','c');
          const again=m.open('B','b2'); return {ids:m.tabs.map(t=>t.id),act:m.active===again, title:m.tabs.find(t=>t.id==='B').title}""")
        self.assertEqual(r["ids"], ["A", "C", "B"])  # C は有効だった A の右に入る
        self.assertTrue(r["act"])
        self.assertEqual(r["title"], "b2")

    def test_close_active_goes_to_neighbour_then_live(self):
        r = run_js("""const m=mkTabs(30); const a=m.open('A'), b=m.open('B'), c=m.open('C'); m.activate(b);
          const n1=m.close(b); const idsA=m.tabs.map(t=>t.id); const n2=m.close(m.tabs[1].k); const n3=m.close(m.tabs[0].k);
          return {n1:n1===c, idsA, n3, left:m.tabs.length, act:m.active}""")
        self.assertEqual(r["idsA"], ["A", "C"])
        self.assertTrue(r["n1"])
        self.assertEqual(r["n3"], "live")
        self.assertEqual(r["left"], 0)
        self.assertEqual(r["act"], "live")

    def test_close_inactive_keeps_active(self):
        r = run_js("""const m=mkTabs(30); const a=m.open('A'), b=m.open('B'); const n=m.close(a); return {same:n===b, act:m.active===b}""")
        self.assertTrue(r["same"] and r["act"])

    def test_close_others_and_right(self):
        r = run_js("""const m=mkTabs(30); const a=m.open('A'), b=m.open('B'), c=m.open('C'), d=m.open('D');
          m.closeRight(b); const x=m.tabs.map(t=>t.id); m.closeOthers(a); const y=m.tabs.map(t=>t.id);
          m.open('E'); m.closeOthers('live'); return {x,y,z:m.tabs.length,act:m.active}""")
        self.assertEqual(r["x"], ["A", "B"])
        self.assertEqual(r["y"], ["A"])
        self.assertEqual(r["z"], 0)
        self.assertEqual(r["act"], "live")

    def test_duplicate_allows_same_doc_twice(self):
        r = run_js("""const m=mkTabs(30); const a=m.open('A','t'); const d=m.dup(a);
          return {ids:m.tabs.map(t=>t.id), distinct:a!==d, act:m.active===d}""")
        self.assertEqual(r["ids"], ["A", "A"])
        self.assertTrue(r["distinct"] and r["act"])

    def test_move(self):
        r = run_js("""const m=mkTabs(30); const a=m.open('A'), b=m.open('B'), c=m.open('C'); m.move(c,0); return m.tabs.map(t=>t.id)""")
        self.assertEqual(r, ["C", "A", "B"])

    def test_step_and_nth(self):
        r = run_js("""const m=mkTabs(30); const a=m.open('A'), b=m.open('B'); m.activate('live');
          const s1=m.step(1)===a, s2=m.step(1)===b, s3=m.step(1)==='live', s4=m.step(-1)===b;
          return {s:[s1,s2,s3,s4], n1:m.nth(1), n2:m.nth(2)===a, n9:m.nth(9)===b, n5:m.nth(5)}""")
        self.assertEqual(r["s"], [True] * 4)
        self.assertEqual(r["n1"], "live")
        self.assertTrue(r["n2"] and r["n9"])
        self.assertIsNone(r["n5"])

    def test_cap_drops_oldest_inactive(self):
        r = run_js("""const m=mkTabs(3); ['A','B','C','D'].forEach(x=>m.open(x)); return {ids:m.tabs.map(t=>t.id), act:m.tabs.find(t=>t.k===m.active).id}""")
        self.assertEqual(r["ids"], ["B", "C", "D"])
        self.assertEqual(r["act"], "D")

    def test_save_restore_skips_live_doc_and_junk(self):
        r = run_js("""const m=mkTabs(30); m.open('A','a'); m.open('live','今の会話'); m.open('B','b');
          const s=m.save(); const m2=mkTabs(30);
          m2.restore(JSON.parse(JSON.stringify(s))); const bad=mkTabs(30); bad.restore({v:1,tabs:[null,{id:5},{id:'X',title:'x'},{}]}); bad.restore('zzz');
          return {ids:m2.tabs.map(t=>t.id), act:m2.active, bad:bad.tabs.map(t=>t.id)}""")
        self.assertEqual(r["ids"], ["A", "B"])
        self.assertEqual(r["act"], "live")  # 再起動すると、いつもの「通訳」から
        self.assertEqual(r["bad"], ["X"])


class TestWiring(unittest.TestCase):
    def js(self):
        return section('<script id="tabbar-js">', "</script>")

    def test_block_is_before_hear_and_has_markup(self):
        self.assertLess(HTML.index('<script id="tabbar-js">'), HTML.index("<!-- ===== hear(11/08)"))
        self.assertLess(HTML.index('<style id="tabbar-css">'), HTML.index("<!-- ===== hear(11/08)"))
        js = self.js()
        for s in ('"tablist"', 'role="tab"', "aria-selected", '"tabbar"', "tbNew", "tbPin"):
            self.assertIn(s, js, s)

    def test_shortcuts_present_and_do_not_clash(self):
        js = self.js()
        for s in ('"t"', '"w"', "Tab", "BracketLeft", "metaKey", "ctrlKey", "auxclick", "contextmenu"):
            self.assertIn(s, js, s)
        # 会話中の素のキー(Space/Esc/R/,/. など)には触れない: すべて ⌘/⌃ つきの時だけ
        h = js[js.index('addEventListener("keydown"'):]
        self.assertIn("if(!(e.metaKey||e.ctrlKey)) return;", h)

    def test_does_not_stop_conversation(self):
        js = self.js()
        for banned in ("T.stop(", "T.setClose(); T.end", ".endConversation", "T.reset("):
            self.assertNotIn(banned, js, banned)

    def test_hides_in_projection_and_fullscreen(self):
        js = self.js()
        self.assertIn('classList.contains("proj")', js)
        self.assertIn("fullscreenElement", js)
        css = section('<style id="tabbar-css">', "</style>")
        for sel in (".pillar", "#start", "#app", "#meet", "#setp", "#appside", "--tbh"):
            self.assertIn(sel, css, sel)
        self.assertIn("overflow-x:auto", css)

    def test_persistence_is_failsafe(self):
        js = self.js()
        self.assertIn("try{", js)
        self.assertIn("tsuyaku.tabs", js)

    def test_theme_tokens_only(self):
        css = section('<style id="tabbar-css">', "</style>")
        self.assertNotRegex(css, r"#[0-9a-fA-F]{6}\b")  # 色は既存のトークン(ダーク/ライト両対応)だけ

    def test_i18n_all_languages(self):
        js = self.js()
        keys = re.findall(r'L\("([^"]+)"\)', js)
        self.assertGreater(len(keys), 8)
        for lang in ("en", "zh", "ko"):
            m = re.search(r"Object\.assign\(I18N\.%s,\{(.*?)\}\);" % lang, js, re.S)
            self.assertIsNotNone(m, lang)
            for k in set(keys):
                if k in ("通訳", "会議の記録"):
                    continue  # 既存の訳を使う
                self.assertIn('"%s":' % k, m.group(1), (lang, k))


if __name__ == "__main__":
    unittest.main()
