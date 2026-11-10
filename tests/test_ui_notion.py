"""会議メモを1枚の読みやすいページに(11/07 notes-notion): 節に包み直して読む順に並べる・見出しでたたむ・
たたんだ中へ飛ぶ時はひらく・紙の見た目は #mdoc の中だけで上書き(元の関数・id・api は変えない)。"""
import re
import unittest
from pathlib import Path

HTML = (Path(__file__).resolve().parents[1] / "src" / "tsuyaku" / "webui" / "index.html").read_text()


def block(tag_id: str, close: str) -> str:
    i = HTML.index(tag_id)
    return HTML[i:HTML.index(close, i)]


JS = block('<script id="ntn-js">', "</script>")
CSS = block('<style id="ntn-css">', "</style>")


class TestNotionPage(unittest.TestCase):
    def test_wraps_render_doc_after_everything_else(self):
        # W2 の wrap("renderDoc") より後ろ(いちばん外側)で包む: 印・書き出し・話した時間を足した後に並べ替える
        self.assertGreater(HTML.index('<script id="ntn-js">'), HTML.index('wrap("renderDoc",function(){this.w2Decorate();});'))
        self.assertIn('wrap("renderDoc",function(){this.ntnDoc();});', JS)
        self.assertIn("a.dataset.ntn", JS)   # 同じ紙を2回包まない

    def test_reading_order(self):
        m = re.search(r"const RANK=\{(.*?)\};", JS, re.S)
        rank = {k.strip('"'): float(v) for k, v in re.findall(r'("?[\w-]+"?):(\d+)', m.group(1))}
        order = ["lead", "s-sum", "s-dec", "s-todo", "s-top", "s-q", "s-chap", "w3ag", "w3mm", "kw", "s-ppl", "s-talk", "s-tx"]
        self.assertEqual(order, sorted(order, key=lambda k: rank[k]))

    def test_folding_rules(self):
        fold = re.search(r"const FOLD=\{(.*?)\};", JS).group(1)
        for k in ("s-dec", "s-todo", "s-top", "s-talk", "kw"):
            self.assertIn(f'"{k}"' if "-" in k else k, fold)
        for k in ("s-ppl", "w3ag", "w3mm", "s-sum"):   # 入力欄・会議中に描き直す節・要約はたたまない
            self.assertNotIn(f'"{k}"', fold)
            self.assertNotIn(f"{k}:", fold)
        self.assertIn('const SHUT0={"s-talk":1};', JS)   # 分析の表だけ(全文の記録はひらいておく)
        self.assertIn('wrap("plState",function(on){if(on){', JS)            # 再生を始めたら全文をひらく
        self.assertIn('localStorage.setItem("tsuyaku.fold"', JS)

    def test_jumps_open_folded_sections(self):
        # 節へ飛ぶ・根拠の札・検索の当たり・記録の行へ飛ぶ操作はどれも scrollIntoView: たたんだ中なら先にひらく
        self.assertIn("Element.prototype.scrollIntoView=function(){", JS)
        self.assertIn('this.closest("#mdoc .nfold.shut")', JS)
        self.assertIn("T.ntnOpenFor(this)", JS)

    def test_ids_and_functions_kept(self):
        # 包み直すだけで、元の節の id と関数はそのまま(ほかの試験・操作が見ている)
        for s in ('id="s-sum"', 'id="s-dec"', 'id="s-todo"', 'id="s-tx"', 'id="todoCnt"', 'id="todoBar"'):
            self.assertIn(s, HTML)
        for f in ("  renderDoc(){", "  docOpen(id,lang,make){", "  txHtml(q,off){", "  jumpLine(i){"):
            self.assertIn(f, HTML)

    def test_css_only_inside_meeting_doc(self):
        sel = [s.strip() for s in re.findall(r"(?:^|})\s*([^{}@/]+)\{", re.sub(r"/\*.*?\*/", "", CSS, flags=re.S))]
        sel = [s for s in sel if s and not s.startswith(("from", "to", "<style"))]
        self.assertGreater(len(sel), 40)
        bad = [s for s in sel if not all(p.strip().startswith(("#mdoc", "#meet.asking #mdoc"))
                                          for p in re.sub(r":is\([^)]*\)", ":is()", s).split(",") if p.strip())]
        self.assertEqual(bad, [], "会議メモの紙の外を変えていないこと")

    def test_toggle_label_translated(self):
        for v in ('"たたむ / ひらく":"Collapse / expand"', '"たたむ / ひらく":"折叠 / 展开"', '"たたむ / ひらく":"접기 / 펼치기"'):
            self.assertIn(v, HTML)

    def test_toolbar_never_scrolls_the_page(self):
        # 狭い窓: 帯は折り返し(節へ飛ぶ所は上の段で横に流す)、紙は窓いっぱい
        self.assertIn("#mdoc .dbar{flex-wrap:wrap;", CSS)
        self.assertIn("#mdoc .dbar .jump{display:flex;flex:1 1 0;min-width:140px;", CSS)
        self.assertIn("overflow-x:auto", CSS)


if __name__ == "__main__":
    unittest.main()


APP = block('<script id="ntn-app-js">', "</script>")
THEME = block('<style id="ntn-theme">', "</style>")
SETUP = (Path(__file__).resolve().parents[1] / "src" / "tsuyaku" / "webui" / "setup.html").read_text()


def contrast(fg: str, bg: str) -> float:
    def lum(h):
        h = h.lstrip("#")
        c = [int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)]
        c = [x / 12.92 if x <= 0.03928 else ((x + 0.055) / 1.055) ** 2.4 for x in c]
        return 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]
    a, b = lum(fg), lum(bg)
    return (max(a, b) + 0.05) / (min(a, b) + 0.05)


def tokens(css: str, selector_start: str) -> dict:
    i = css.index(selector_start)
    body = css[css.index("{", i + len(selector_start) - 1) + 1:css.index("}", i)]
    return dict(re.findall(r"--([\w-]+):(#[0-9A-Fa-f]{6})", body))


class TestAppTheme(unittest.TestCase):
    """アプリ全体の型: 明るい/暗いの色の名前は同じ。システムに合わせる(既定)・明るい・暗いを config に残す。"""

    def test_light_and_dark_tokens_and_contrast(self):
        light = tokens(THEME, ":root{")
        dark = tokens(THEME, ':root[data-theme="dark"]{')
        self.assertEqual(set(light), set(dark))
        self.assertEqual(light["bg"].lower(), "#ffffff")
        self.assertEqual(light["text"].lower(), "#37352f")
        self.assertEqual(dark["bg"].lower(), "#191919")
        for t in (light, dark):   # 字は地に対して AA(4.5:1)以上
            for k in ("text", "text-2", "text-3", "a", "b", "err"):
                self.assertGreaterEqual(contrast(t[k], t["bg"]), 4.5, k)
            self.assertGreaterEqual(contrast(t["text-3"], t["bg-side"]), 4.5)
            self.assertGreaterEqual(contrast("#ffffff", t["accent-fill"]), 4.5)
        # 読み上げ中の帯(話し手の色の地)の上の字(--ink = 地の色)
        self.assertGreaterEqual(contrast(light["bg"], light["a"]), 4.5)
        self.assertGreaterEqual(contrast(dark["bg"], dark["a"]), 4.5)
        self.assertIn('@media (prefers-color-scheme:dark){ :root:not([data-theme="light"])', THEME)

    def test_theme_is_applied_before_paint_and_saved(self):
        self.assertIn('localStorage.getItem("tsuyaku.theme");if(t==="light"||t==="dark")document.documentElement.dataset.theme=t;', HTML)
        self.assertIn("a.set_theme(v)", APP)
        self.assertIn("a.get_theme().then", APP)
        for k in ('data-theme="\'+k+\'"', '["system","システムに合わせる"]', '["light","明るい"]', '["dark","暗い"]'):
            self.assertIn(k, APP)

    def test_sidebar_holds_meeting_list_and_toggles(self):
        self.assertIn('const ms=$("#mside"); if(ms)sb.querySelector(".sbmid").appendChild(ms);', APP)   # 一覧・検索・絞り込みをそのまま
        for k in ('"new","new","新しい会話"', '"meet","meet","会議の記録"', '"set","set","設定"'):
            self.assertIn(k, APP)
        self.assertIn('(e.key==="\\\\"||e.code==="Backslash")', APP)   # ⌘\
        self.assertIn("body.live #appside{display:none}", THEME)     # 会話中は隠す(開ける)
        self.assertIn("body.live.sb-live #appside{display:flex", THEME)

    def test_start_page_and_settings_move_existing_controls(self):
        for k in ('row("使い方",[$("#modes")]);', 'row("会議の種類",[$("#tplChip"),$("#tplBox")],"r-notes");',
                  'row("議題",[$("#agendaChip"),$("#agendaBox")],"r-notes");',
                  'srow("#sgVoice","翻訳の処理場所","",[eg,$("#engineNote")]);', 'srow("#sgUi","画面の言語","",[$("#uiLang")]);'):
            self.assertIn(k, APP)
        self.assertIn("#go{width:auto;background:var(--accent-fill)", THEME)   # 開始は青いボタン

    def test_new_strings_have_all_translations(self):
        for k in ("新しい会話", "設定", "画面の明るさ", "システムに合わせる", "明るい", "暗い", "サイドバーを開く", "サイドバーを閉じる",
                  "投影モード", "翻訳と声"):
            self.assertGreaterEqual(HTML.count(f'"{k}":'), 3, k)

    def test_setup_page_uses_same_tokens(self):
        self.assertIn('@media (prefers-color-scheme:dark){ :root:not([data-theme="light"])', SETUP)
        self.assertIn('localStorage.getItem("tsuyaku.theme")', SETUP)
        self.assertIn("button.main{background:var(--accent-fill)", SETUP)
        for i in ('id="go"', 'id="pause"', 'id="items"', 'id="pct"', 'id="err"', 'id="ui"'):
            self.assertIn(i, SETUP)


UX = block('<script id="ntn-ux-js">', "</script>")
UXCSS = block('<style id="ntn-ux">', "</style>")


class TestUsability(unittest.TestCase):
    """使いやすさ: すぐ始める・いつも同じ所に状態・⌘K・空とエラーの次の一手・押しやすさ・覚える。"""

    def test_start_fast(self):
        self.assertIn('if(e.key==="Enter"&&!st.classList.contains("hidden")', UX)   # ⌘↩ で始める
        self.assertIn('pr.insertAdjacentElement("beforebegin",ns);', UX)            # 始めるボタンは題のすぐ下
        self.assertIn('if(ae&&ae!==document.body) return;', UX)                     # 何も選んでいない Enter だけ

    def test_status_everywhere(self):
        for k in ('"取り込み中"', '"メモ作成中"', '"翻訳サーバー未接続"', '"準備できています"', '"待機中"'):
            self.assertIn(k, UX)
        self.assertIn('["status","notes","impProg","impTick","engineShow","applyUi","reset","docOpen"].forEach(n=>wrap(n,function(){this.gstat();}));', UX)

    def test_command_palette(self):
        for k in ('if(k==="k"){', '"ArrowDown"', '"ArrowUp"', 'e.key==="Escape"', 'role","dialog"', 'aria-activedescendant'):
            self.assertIn(k, UX)
        self.assertIn("「{q}」を記録の中から探す", UX)

    def test_empty_and_error_next_steps(self):
        self.assertIn('if(t!=="まだ会話の記録がありません") return h;', UX)
        self.assertIn('data-ea="new"', UX)
        self.assertIn('data-ea="imp"', UX)
        self.assertIn('b.hidden=!/翻訳サーバー|処理場所|マイク/.test(t)||this.live;', UX)

    def test_targets_and_layout_memory(self):
        self.assertIn("#appside .sbi{min-height:32px}", UXCSS)
        self.assertIn("body.live #foot #markBtn,body.live #stopSpeak,body.live #rail .racts .tb{min-height:44px}", UXCSS)
        self.assertIn('store.set("tsuyaku.sbw"', UX)                     # サイドバーの幅
        self.assertIn('store.set("tsuyaku.side"', APP)                   # 開閉
        self.assertIn('localStorage.setItem("tsuyaku.fold"', JS)         # たたんだ節
        self.assertIn('store.set("tsuyaku.theme",v)', APP)               # 明るさ

    def test_notes_in_ui_language_are_loaded(self):
        # 画面の言語のメモがあるのに、既定で届いた別の言語のメモを捨てて「まだありません」と出していた
        self.assertIn("if(!lang&&nl.includes(this._docLang)){this.docOpen(id,this._docLang,make);return;}", HTML)


class TestStartHintInDistBuild(unittest.TestCase):
    """起動画面の一言(自動: リモートサーバーの GPU…)は開発用。配る版では sanitize が リモートサーバーを書き換え、dist.js が
    「この Mac の翻訳エンジン…で翻訳します」に差し替える。書き換えた画面に自分のサーバー名が残らないこと。"""

    def test_rewritten_webui_has_no_lab_server(self):
        import importlib.util, io, shutil, tempfile
        from contextlib import redirect_stderr, redirect_stdout
        root = Path(__file__).resolve().parents[1]
        spec = importlib.util.spec_from_file_location("sanitize", root / "packaging" / "sanitize.py")
        sz = importlib.util.module_from_spec(spec); spec.loader.exec_module(sz)
        with tempfile.TemporaryDirectory() as d:
            web = Path(d) / "Sokutsu.app" / "Contents" / "Resources" / "webui"
            shutil.copytree(root / "src" / "tsuyaku" / "webui", web)
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                self.assertEqual(sz.rewrite([Path(d) / "Sokutsu.app"]), 0)
                self.assertEqual(sz.check([Path(d) / "Sokutsu.app"]), 0)
            html = (web / "index.html").read_text(encoding="utf-8")
            self.assertNotIn("s1" + "09", html.lower())
        dist = (root / "src" / "tsuyaku" / "webui" / "dist.js").read_text(encoding="utf-8")
        self.assertIn('document.querySelector("#startHint");', dist)
        self.assertIn("この Mac の翻訳エンジン({m})で翻訳します", dist)

    def test_start_hint_is_short(self):
        m = re.search(r'<div class="hint" id="startHint">([^<]*)</div>', HTML)
        self.assertLessEqual(len(m.group(1)), 28, m.group(1))   # 以前は 44 字
