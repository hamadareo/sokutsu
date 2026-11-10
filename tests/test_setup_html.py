"""初回の準備の画面(setup.html)の配線。11/11: head の `var t` が本体の `const t` と衝突し、
本体が丸ごと動かず「取得を始める」が押せなかった。JavaScriptCore(jsc)で head→本体を別々の
script として順に実行し(ブラウザと同じ)、動くことを確かめる。"""
import re
import subprocess
import tempfile
import unittest
from pathlib import Path

HTML = Path(__file__).resolve().parents[1] / "src" / "tsuyaku" / "webui" / "setup.html"
JSC = Path("/System/Library/Frameworks/JavaScriptCore.framework/Versions/A/Helpers/jsc")

STUB = r"""
var __timers=[], __log=[], __listeners={};
function setInterval(f,ms){__timers.push({f:f,ms:ms,on:true});return __timers.length;}
function clearInterval(i){if(__timers[i-1])__timers[i-1].on=false;}
function setTimeout(f,ms){return 0;}
function tick(){__timers.forEach(function(t){if(t.on)t.f();});}
var console={error:function(m){__log.push(String(m));},log:function(){}};
var __els={};
function mkEl(sel){var e={sel:sel,textContent:"",innerHTML:"",style:{},dataset:{},hidden:false,disabled:false,_l:{},
  classList:{_s:{},add:function(c){this._s[c]=1;},remove:function(c){delete this._s[c];},toggle:function(c,v){if(v)this._s[c]=1;else delete this._s[c];}},
  addEventListener:function(n,f){this._l[n]=f;},appendChild:function(){},append:function(){}};return e;}
var document={documentElement:{dataset:{},lang:""},
  querySelector:function(s){return __els[s]||(__els[s]=mkEl(s));},querySelectorAll:function(s){return [];},
  createElement:function(){return mkEl("x");}};
var localStorage={getItem:function(){return null;},setItem:function(){}};
var navigator={language:"ja"};
var window=this;
window.addEventListener=function(n,f){(__listeners[n]=__listeners[n]||[]).push(f);};
function fire(n,ev){(__listeners[n]||[]).forEach(function(f){f(ev||{});});}
function thenable(v){return {then:function(f){var r=f(v);return thenable(r);},catch:function(){return this;}};}
"""

def scripts():
    return re.findall(r"<script>(.*?)</script>", HTML.read_text(encoding="utf-8"), re.S)


def run_jsc(driver):
    with tempfile.TemporaryDirectory() as d:
        files = []
        for i, body in enumerate([STUB] + scripts() + [driver]):
            p = Path(d) / f"{i}.js"
            p.write_text(body, encoding="utf-8")
            files.append(str(p))
        r = subprocess.run([str(JSC), *files], capture_output=True, text=True, timeout=60)
    return r.returncode, r.stdout + r.stderr


class StaticTest(unittest.TestCase):
    def test_no_global_var_in_head_script(self):
        # 本体は const/let を使う。先の script が global に var を作ると衝突して本体が動かない
        for s in scripts()[:-1]:
            code = re.sub(r"/\*.*?\*/", "", s, flags=re.S).strip()
            self.assertTrue(code.startswith("(function(){"), code[:60])

    def test_wiring(self):
        h = HTML.read_text(encoding="utf-8")
        for needle in ['id="go"', "setup_start", "setup_state", "pywebviewready", "apiTimer=setInterval(waitApi", "noapi", "window.addEventListener(\"error\""]:
            self.assertIn(needle, h, needle)
        for lang in ("ja", "en", "zh", "ko"):
            self.assertRegex(h, lang + r':\{[^\n]*agree:')
        self.assertEqual(h.count('noapi:"'), 4)
        self.assertEqual(h.count('jserr:"'), 4)
        for k in ("foundH", "needH", "pick", "nothing", "verifying", "foundNone", "foundN", "linked", "here"):
            self.assertEqual(len(re.findall(r"[,{]" + k + ':"', h)), 4, k)
        self.assertIn("setup_pick_folder", h)


@unittest.skipUnless(JSC.exists(), "jsc が無い")
class JsTest(unittest.TestCase):
    def test_scripts_run_in_sequence_without_error(self):
        rc, out = run_jsc('print("OK-" + (typeof render) + "-" + document.querySelector("#go")._l.click);')
        self.assertEqual(rc, 0, out)
        self.assertIn("OK-function-", out)
        self.assertNotIn("undefined", out)  # click ハンドラが付いている

    def test_click_without_api_shows_message(self):
        rc, out = run_jsc('document.querySelector("#go")._l.click();'
                          'print("ERR=" + document.querySelector("#err").textContent);')
        self.assertEqual(rc, 0, out)
        self.assertIn("内部の準備ができていません", out)

    def test_click_starts_even_if_state_is_null(self):
        rc, out = run_jsc('var n=0; window.pywebview={api:{setup_state:function(){return thenable(null);},'
                          'setup_start:function(){n++;return thenable("ok");},setup_log:function(){return thenable("ok");}}};'
                          'document.querySelector("#go")._l.click(); print("STARTED=" + n);')
        self.assertEqual(rc, 0, out)
        self.assertIn("STARTED=1", out)

    def test_api_arrives_late_without_ready_event(self):
        # pywebviewready を取り逃しても、0.5 秒ごとの再試行で状態を取って描く
        rc, out = run_jsc('var st0={phase:"idle",got:0,total:3670000000,items:[{status:"wait",label:"翻訳",labelEn:"T",size:1000,got:0}],speed:0};'
                          'tick(); window.pywebview={api:{setup_state:function(){return thenable(st0);}}}; tick();'
                          'print("SUM=" + document.querySelector("#sum").textContent);')
        self.assertEqual(rc, 0, out)
        self.assertIn("0 MB / 3.42 GB", out)

    def test_api_never_arrives_shows_message_after_10s(self):
        rc, out = run_jsc('for(var i=0;i<21;i++)tick();'
                          'print("ERR=" + document.querySelector("#err").textContent + "|LOG=" + __log.join(";"));')
        self.assertEqual(rc, 0, out)
        self.assertIn("内部の準備ができていません", out)
        self.assertIn("10 秒", out)

    def test_js_error_is_shown_not_swallowed(self):
        rc, out = run_jsc('fire("error",{message:"boom",lineno:7});'
                          'print("ERR=" + document.querySelector("#err").textContent + "|LOG=" + __log.join(";"));')
        self.assertEqual(rc, 0, out)
        self.assertIn("boom", out)

    def test_found_and_needed_are_separated_and_total_counts_only_missing(self):
        rc, out = run_jsc(
            'var d={phase:"idle",got:0,total:574041195,speed:0,verifyGot:0,verifyTotal:0,'
            'found:[{key:"llm",label:"翻訳",labelEn:"T",size:2526080992,path:"~/x/gemma.gguf",linked:true}],'
            'items:[{status:"wait",label:"音声",labelEn:"W",size:574041195,got:0}]};'
            'window.pywebview={api:{setup_state:function(){return thenable(d);}}}; tick();'
            'print("SUM=" + document.querySelector("#sum").textContent);'
            'print("FH=" + document.querySelector("#foundH").hidden + " NH=" + document.querySelector("#needH").hidden'
            ' + " FL=" + document.querySelector("#found").hidden);')
        self.assertEqual(rc, 0, out)
        self.assertIn("0 MB / 547 MB", out)       # 取得が必要な分だけ
        self.assertIn("FH=false NH=false FL=false", out)

    def test_all_found_shows_ready_state(self):
        rc, out = run_jsc(
            'var d={phase:"done",got:0,total:0,speed:0,verifyGot:0,verifyTotal:0,'
            'found:[{key:"a",label:"a",labelEn:"a",size:5,path:"~/p",linked:false}],items:[]};'
            'window.pywebview={api:{setup_state:function(){return thenable(d);}}}; tick();'
            'print("GO=" + document.querySelector("#go").dataset.t + " PCT=" + document.querySelector("#pct").textContent'
            ' + " NH=" + document.querySelector("#needH").hidden);')
        self.assertEqual(rc, 0, out)
        self.assertIn("GO=open PCT=100% NH=true", out)

    def test_pick_folder_calls_api_and_reports(self):
        rc, out = run_jsc(
            'var n=0; window.pywebview={api:{setup_state:function(){return thenable(null);},'
            'setup_pick_folder:function(){n++;return thenable(0);},setup_log:function(){return thenable("ok");}}};'
            'document.querySelector("#pick")._l.click();'
            'print("N=" + n + " NOTE=" + document.querySelector("#note").textContent);')
        self.assertEqual(rc, 0, out)
        self.assertIn("N=1 NOTE=このフォルダには使えるモデルが見つかりませんでした", out)


if __name__ == "__main__":
    unittest.main()
