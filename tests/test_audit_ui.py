"""11/07 監査(画面): 属性に入る文字の逃がし・記録の中の検索の印・会議を開く返事の順番・一覧の読み上げ名。
index.html を文字として読み、関数は JavaScriptCore(osascript)で実際に動かして確かめる。"""
import json
import re
import subprocess
import tempfile
import unittest
from pathlib import Path

HTML = (Path(__file__).resolve().parents[1] / "src" / "tsuyaku" / "webui" / "index.html").read_text()


def run_js(src: str) -> str:
    with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False) as f:
        f.write(src)
    r = subprocess.run(["osascript", "-l", "JavaScript", "-e",
                        "ObjC.import('Foundation'); var s=$.NSString.stringWithContentsOfFileEncodingError("
                        f"'{f.name}',4,null).js; var r; try{{r=String(eval(s))}}catch(e){{r='ERR '+e}} r"],
                       capture_output=True, text=True, timeout=30)
    return r.stdout.strip()


def esc_src() -> str:
    m = re.search(r"^\s*(esc\(s\)\{.*\}),$", HTML, re.M)
    assert m, "esc が見つからない"
    return "var T={" + m.group(1) + "};"


def hi_src() -> str:
    m = re.search(r"(const hi=t=>\{.*?return h\+E\(s\.slice\(p\)\);\};)", HTML, re.S)
    assert m, "記録の中の検索の印(hi)が見つからない"
    return m.group(1)


class TestEscapeAttributes(unittest.TestCase):
    """発言・名前・タグ・題に " があると title="…" / value="…" がそこで切れ、続きが属性として読まれていた。
    話者の名前を  Bob" autofocus onfocus="…  にすると、会議の記録を開いただけでスクリプトが動いた。"""

    def test_esc_escapes_quotes_and_markup(self):
        out = run_js(esc_src() + "T.esc('a\"b\\'c<d>&e')")
        self.assertEqual(out, "a&quot;b&#39;c&lt;d&gt;&amp;e")

    def test_esc_null_and_numbers(self):
        out = run_js(esc_src() + "JSON.stringify([T.esc(null),T.esc(undefined),T.esc(0),T.esc(12)])")
        self.assertEqual(json.loads(out), ["", "", "0", "12"])

    def test_attribute_cannot_be_broken_out_of(self):
        out = run_js(esc_src() + "'<input value=\"'+T.esc('Bob\" autofocus onfocus=\"x()')+'\">'")
        self.assertNotIn('" autofocus', out)
        self.assertIn("&quot; autofocus onfocus=&quot;x()", out)


class TestTranscriptSearchMarks(unittest.TestCase):
    """記録の中の検索: 逃がした後の文で探していたので「amp」「lt」「39」が &amp; &lt; &#39; の中に当たり、
    「R&D」が「R&amp;D」と化けていた。"""

    def run_hi(self, q: str, text: str) -> str:
        return run_js(esc_src() + "var E=x=>T.esc(x==null?'':String(x)); var ql=" + json.dumps(q.lower()) + ";"
                      + hi_src() + "hi(" + json.dumps(text) + ")")

    def test_query_inside_entity_does_not_break_text(self):
        self.assertEqual(self.run_hi("amp", "sample & example"),
                         "s<mark>amp</mark>le &amp; ex<mark>amp</mark>le")
        self.assertEqual(self.run_hi("39", "it's 39"), "it&#39;s <mark>39</mark>")

    def test_query_with_markup_chars_is_marked_and_escaped(self):
        self.assertEqual(self.run_hi("<b>", "x <b> y"), "x <mark>&lt;b&gt;</mark> y")
        self.assertEqual(self.run_hi('"hi"', 'He said "Hi"'), "He said <mark>&quot;Hi&quot;</mark>")

    def test_no_query_just_escapes(self):
        self.assertEqual(self.run_hi("", "<img src=x>"), "&lt;img src=x&gt;")


class TestDocOpenLatestWins(unittest.TestCase):
    """長い会議 A を押してすぐ短い B を押すと、後から届いた A の中身が B を選んだまま描かれ、
    名前・題・行の修正が B に書かれていた(pywebview の返事は呼んだ順とは限らない)。"""

    def test_stale_meeting_get_is_ignored(self):
        i = HTML.index("  docOpen(id,lang,make){")
        body = HTML[i:HTML.index("\n  },", i)]
        self.assertIn("const seq=this._docSeq=(this._docSeq||0)+1;", body)
        g = body.index("const got=g=>{")
        self.assertIn("if(seq!==this._docSeq) return;", body[g:g + 80])


class TestMeetingListLabel(unittest.TestCase):
    def test_list_aria_label_is_translated(self):
        self.assertIn('<nav id="mside" aria-label="会議の一覧" data-t-aria>', HTML)
        for v in ('"会議の一覧":"Meeting list"', '"会議の一覧":"会议列表"', '"会議の一覧":"회의 목록"'):
            self.assertIn(v, HTML)


if __name__ == "__main__":
    unittest.main()


def python_ui_messages():
    """Python が画面へ送る決まった文(TSUYAKU.error の題・本文、マイクの _report、取り込みの ImportFailed)。
    隣り合う文字列は1つにつなぎ、f-string の {…} を含むものは「 — 」と「(…)」で切った決まった部分だけを返す。"""
    src = Path(__file__).resolve().parents[1] / "src" / "tsuyaku"
    out = set()
    for f in sorted(src.glob("*.py")):
        s = f.read_text(encoding="utf-8")
        for m in re.finditer(r'\(\s*"error"\s*,|\b_report\(|\bImportFailed\(', s):
            i, depth, j = m.end(), 1, m.end()
            while depth and j < len(s):
                depth += (s[j] == "(") - (s[j] == ")")
                j += 1
            args, groups, last = s[i:j - 1], [], None
            for lit in re.finditer(r'(f?)"((?:[^"\\]|\\.)*)"', args):
                if last is not None and args[last:lit.start()].strip() == "" and groups:
                    groups[-1][0] |= lit.group(1) == "f"
                    groups[-1][1] += lit.group(2)
                else:
                    groups.append([lit.group(1) == "f", lit.group(2)])
                last = lit.end()
            for is_f, t in groups:
                if not re.search(r"[぀-鿿]", t):
                    continue
                for part in t.split(" — "):
                    if is_f and "{" in part:
                        part = re.sub(r"\([^()]*\{[^()]*\)$", "", part)   # 「…。({e})」の値の部分
                    if "{" in part or not re.search(r"[぀-鿿]", part):
                        continue
                    out.add(part)
    return out


class TestPythonMessagesTranslated(unittest.TestCase):
    """aud-live で足したマイクのエラー・発言の処理の失敗・取り込みの失敗などが en/zh/ko に無く、英語の画面でも
    日本語で出ていた。例外の名前が混ざる本文は Lmsg() が「 — 」「(…)」の切れ目ごとに訳す。"""

    def test_every_fixed_message_has_en_zh_ko(self):
        keys = python_ui_messages()
        for k in ("マイクの音が届いていません", "マイクを開けません", "この発言を処理できませんでした",
                  "マイクの接続とシステム設定のマイクの許可を確かめ、会話を始め直してください"):
            self.assertIn(k, keys)
        missing = [k for k in sorted(keys) if HTML.count(json.dumps(k, ensure_ascii=False) + ":") < 3]
        self.assertEqual(missing, [], "訳の無いエラー文")

    def test_lmsg_translates_parts_around_dynamic_values(self):
        m = re.search(r"^\s*(L\(k,v\)\{.*?return t;\},)", HTML, re.M | re.S)
        n = re.search(r"^\s*(Lmsg\(s\)\{.*?\.join\(\" — \"\);\},)", HTML, re.M | re.S)
        self.assertTrue(m and n)
        js = ("var I18N={en:{'マイクを開けません':'Can\\'t open the microphone','続けます':'go on',"
              "'確認してください。':'Check it.'}};"
              "var T={ui:'en'," + m.group(1) + n.group(1) + "};"
              "JSON.stringify([T.Lmsg('マイクを開けません'),T.Lmsg('OSError: x — 続けます'),"
              "T.Lmsg('確認してください。(E: y)'),T.Lmsg('知らない文 — 続けます'),T.Lmsg(null)])")
        self.assertEqual(json.loads(run_js(js)),
                         ["Can't open the microphone", "OSError: x — go on", "Check it.(E: y)",
                          "知らない文 — go on", ""])

    def test_error_and_import_message_use_lmsg(self):
        i = HTML.index("  error(title,body){")
        self.assertIn('this.$("#errTitle").textContent=this.Lmsg(title||"")', HTML[i:i + 400])
        self.assertIn('let eb=this.Lmsg(body||"")', HTML[i:i + 400])
        self.assertIn("""'<p class="w2msg">'+this.esc(this.Lmsg(j.msg))""", HTML)


def method_src(name: str, end_next: str) -> str:
    """オブジェクトの中のメソッド 1 つ(次のメソッド名の手前まで)。"""
    i = HTML.index(f"\n  {name}(")
    j = HTML.index(f"\n  {end_next}(", i)
    body = re.sub(r"(\n\s*//[^\n]*)+\s*$", "", HTML[i:j].rstrip())   # 次のメソッドの前の説明の行は除く
    return body.strip().rstrip(",")


class TestNotesPushTarget(unittest.TestCase):
    """(b) メモの進み具合・結果は _docId だけを見て、今の _doc に書いていた。B を開いて返事を待っている間
    (_doc はまだ A)に B のメモができると、A の記録に B のメモが描かれた。_doc が null なら例外。"""

    def run_notes(self, doc_for, doc, d):
        js = ("var log=[]; var el={hidden:false,scrollTop:0,querySelector:function(){return null;}};"
              "var T={_docId:'B',_docFor:" + json.dumps(doc_for) + ",_doc:" + json.dumps(doc) + ",_docLang:'ja',_mlist:[],"
              "$:function(){return el;},L:function(k){return k;},progStop:function(){log.push('stop');},"
              "renderDoc:function(){log.push('render:'+(this._doc&&this._doc.id));},renderList:function(){},"
              "toast:function(){log.push('toast');},docOpen:function(id,lang){log.push('open:'+id+'/'+lang);},"
              + method_src("notes", "toast") + "};"
              "T.notes(" + json.dumps(d) + "); JSON.stringify({log:log,doc:T._doc})")
        return json.loads(run_js(js))

    def test_done_for_other_doc_reloads_instead_of_drawing(self):
        r = self.run_notes("A", {"id": "A", "notes": None}, {"id": "B", "stage": "done", "notes": {"lang": "ja", "title": "b"}})
        self.assertEqual(r["doc"]["notes"], None)               # A に B のメモを書かない
        self.assertNotIn("render:A", r["log"])
        self.assertIn("open:B/ja", r["log"])                    # B を読み直して描く

    def test_done_with_null_doc_does_not_throw(self):
        r = self.run_notes(None, None, {"id": "B", "stage": "done", "notes": {"lang": "ja"}})
        self.assertIn("open:B/ja", r["log"])

    def test_progress_for_other_doc_is_not_drawn(self):
        r = self.run_notes("A", {"id": "A"}, {"id": "B", "stage": "chunk", "pct": 0.4})
        self.assertNotIn("render:A", r["log"])

    def test_done_for_current_doc_still_draws(self):
        r = self.run_notes("B", {"id": "B", "notes": None}, {"id": "B", "stage": "done", "notes": {"lang": "ja", "title": "b"}})
        self.assertEqual(r["doc"]["notes"]["title"], "b")
        self.assertIn("render:B", r["log"])

    def test_doc_open_remembers_which_meeting_doc_is(self):
        i = HTML.index("  docOpen(id,lang,make){")
        body = HTML[i:HTML.index("\n  },", i)]
        self.assertIn("this._doc=g; this._docFor=id;", body)


class TestVoicesNotDropped(unittest.TestCase):
    """(c) 声の照合は1度に1つだけ呼び、待っている間に開いた会議の分は捨てていた(開き直すまで声の印が出ない)。"""

    def test_reruns_for_latest_doc_after_busy(self):
        body = method_src("w4docVoices", "w4docNames")
        self.assertIn("this._w4vBusy=false;", body)
        self.assertRegex(body, r"if\(this\._docId!==id&&this\._docFor===this\._docId\)this\.w4docVoices\(\);")


class TestMeetingsListLatestWins(unittest.TestCase):
    """(d) 一覧(meetings_list)も返事が呼んだ順とは限らず、古い一覧が後から描かれていた。"""

    def test_every_meetings_list_call_is_numbered(self):
        calls = len(re.findall(r'meetings_list\(\)\.then|w4call\("meetings_list"\)', HTML))
        self.assertEqual(calls, 4)
        self.assertEqual(HTML.count("const ms=this._mlSeq=(this._mlSeq||0)+1;"), calls)
        self.assertEqual(HTML.count("ms===this._mlSeq"), calls)


class TestEscapeInSearch(unittest.TestCase):
    """(e) 一覧の検索・記録の中の検索で Esc を押すと、検索ではなく会議の画面ごと閉じていた。"""

    def test_escape_clears_search_first(self):
        i = HTML.index('    if(e.key==="Escape"){\n      if(!document.getElementById("meet").hidden){')
        blk = HTML[i:i + 700]
        self.assertIn('ae.matches("#mq,#mdoc .txq")&&ae.value', blk)
        self.assertIn('ae.value="";ae.dispatchEvent(new Event("input",{bubbles:true}));', blk)
        self.assertLess(blk.index('ae.matches("#mq,#mdoc .txq")'), blk.index("TSUYAKU.meetClose()"))


class TestErrorDetailSeparate(unittest.TestCase):
    """(f) 「翻訳サーバーに接続できません」「音声認識モデルを読み込めません」の本文は、訳す文と例外の名前を
    1つの文字列につないで送っていたので、訳の鍵にならなかった。文と詳しい値を分けて送り、画面は文だけ訳す。"""

    def test_python_sends_detail_separately(self):
        src = (Path(__file__).resolve().parents[1] / "src" / "tsuyaku" / "__main__.py").read_text(encoding="utf-8")
        # 2つ目は配る版のビルドの試験(packaging/sanitize.py で書き換えた写し)の文
        self.assertRegex(src, r'\("error", "翻訳サーバーに接続できません",\s*"起動画面の『翻訳の処理場所』を見直してください'
                              r'(?:\(リモートサーバーなら "\s*"scripts/remote_tunnel\.sh、このMacなら LM Studio のサーバー起動\)。"'
                              r'|\(自分のサーバーなら config\.json の server_url、"\s*"このMacなら初回の準備が済んでいるか\)。")'
                              r',\s*\n\s*type\(last_err\)')
        self.assertRegex(src, r'\("error", "音声認識モデルを読み込めません",\s*"models/ggml-large-v3-turbo\.bin を確認してください。", str\(e\)\)')

    def test_error_shows_translated_sentence_plus_detail(self):
        m = re.search(r"^\s*(L\(k,v\)\{.*?return t;\},)", HTML, re.M | re.S)
        n = re.search(r"^\s*(Lmsg\(s\)\{.*?\.join\(\" — \"\);\},)", HTML, re.M | re.S)
        js = ("var I18N={en:{'接続できません':'Cannot connect','見直してください。':'Check it.'}}; var els={};"
              "var T={ui:'en',$:function(k){return els[k]||(els[k]={style:{}});}," + m.group(1) + n.group(1)
              + method_src("error", "esc") + "}; var setTimeout=function(){return 1;}; var clearTimeout=function(){};"
              "T.error('接続できません','見直してください。','ConnectError');"
              "var a=els['#errBody'].textContent; T.error(['接続できません','見直してください。']);"
              "JSON.stringify([els['#errTitle'].textContent,a,els['#errBody'].textContent])")
        self.assertEqual(json.loads(run_js(js)), ["Cannot connect", "Check it. (ConnectError)", "Check it."])


class TestNoRawValuesInAttributes(unittest.TestCase):
    """(a) esc() 以外の逃がしは無かった(自前の esc・テンプレート文字列の innerHTML も無し)。残っていたのは
    会話中の記録の行の lang="…" に Python から来た言語コードを逃がさず入れる所。"""

    def test_no_other_escaper_or_template_html(self):
        self.assertEqual(len(re.findall(r"\besc\s*\(s\)\s*\{", HTML)), 1)
        self.assertNotRegex(HTML, r"(innerHTML\s*[+]?=|insertAdjacentHTML\([^,]+,)\s*`")
        self.assertNotRegex(HTML, r"textContent\s*=[^;]*;\s*return\s+\w+\.innerHTML")

    def test_live_item_lang_is_escaped(self):
        self.assertNotRegex(HTML, r"""lang="'\+\(BCP\[""")
