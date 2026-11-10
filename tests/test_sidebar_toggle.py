"""サイドバーの開け閉めボタンは、いつも 1 つだけ見える(11/08 ユーザー:「サイドバーを閉じるボタンが2個あって
左側のボタンがロゴと被っている」)。

原因: 閉じている時だけ出す #sbOpen を hidden で隠していたが、.sbopen{display:grid} が [hidden] より強く、
サイドバーを開いていても左上(ロゴの上)に出ていた。会話中に会議メモを開くと逆に 1 つも無かった。
ここでは本物の index.html を画面に出さない WKWebView で描き、状態ごとに「見えていて、押せる」ボタンを数える。"""
import json
import sys
import time
import unittest
from pathlib import Path

HTML = Path(__file__).resolve().parents[1] / "src" / "tsuyaku" / "webui" / "index.html"

try:
    if sys.platform != "darwin":
        raise ImportError("macOS only")
    from AppKit import NSApplication, NSApplicationActivationPolicyProhibited, NSBackingStoreBuffered, NSWindow
    from Foundation import NSDate, NSMakeRect, NSRunLoop, NSURL
    from WebKit import WKWebView
    HAVE_WK = True
except Exception:  # noqa: BLE001
    HAVE_WK = False

PROBE = r"""(()=>{const out=[];
 const logo=document.querySelector('#appside .sblogo'); const side=document.getElementById('appside');
 const sideOn=!!side&&getComputedStyle(side).display!=='none';
 const lr=logo&&logo.getBoundingClientRect();
 document.querySelectorAll('#sbClose,#sbOpen,[data-crumb="side"]').forEach(b=>{
   const r=b.getBoundingClientRect(), cs=getComputedStyle(b);
   if(cs.display==='none'||cs.visibility==='hidden'||r.width<2||r.height<2||r.right<=0||r.left>=innerWidth) return;
   const t=document.elementFromPoint(r.left+r.width/2,r.top+r.height/2); if(!t||!(t===b||b.contains(t))) return;   // 下に隠れたものは数えない
   const over=sideOn&&!!lr&&b.id!=='sbClose'&&!(r.right<=lr.left||r.left>=lr.right||r.bottom<=lr.top||r.top>=lr.bottom);
   out.push({id:b.id||'crumb',over});});
 return JSON.stringify({side:sideOn,t:out});})()"""


@unittest.skipUnless(HAVE_WK, "WKWebView (pyobjc) が無い")
class TestOneSidebarToggle(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        NSApplication.sharedApplication().setActivationPolicy_(NSApplicationActivationPolicyProhibited)
        cls.win = NSWindow.alloc().initWithContentRect_styleMask_backing_defer_(NSMakeRect(0, 0, 1280, 800), 0, NSBackingStoreBuffered, False)
        cls.wv = WKWebView.alloc().initWithFrame_(NSMakeRect(0, 0, 1280, 800))
        cls.win.setContentView_(cls.wv)
        cls.wv.loadFileURL_allowingReadAccessToURL_(NSURL.fileURLWithPath_(str(HTML)), NSURL.fileURLWithPath_(str(HTML.parent)))
        t = time.time()
        while time.time() - t < 15:
            cls.pump(0.05)
            if cls.js("document.readyState==='complete'&&typeof TSUYAKU==='object'&&!!document.getElementById('appside')") is True:
                break
        cls.pump(1.5)    # 起動の後片づけ(reset など)が済んでから
        cls.js("try{localStorage.clear()}catch(e){}")
        cls.js("{const s=document.createElement('style');s.textContent='*,*::before,*::after{transition:none!important;animation:none!important}';document.head.appendChild(s);document.body.classList.remove('opening');}")

    @classmethod
    def pump(cls, sec):
        NSRunLoop.currentRunLoop().runUntilDate_(NSDate.dateWithTimeIntervalSinceNow_(sec))

    @classmethod
    def js(cls, src):
        box = {}
        cls.wv.evaluateJavaScript_completionHandler_(src, lambda r, e: box.update(r=r))
        t = time.time()
        while "r" not in box and time.time() - t < 5:
            cls.pump(0.01)
        return box.get("r")

    def toggles(self, setup):
        self.js("document.body.classList.remove('live','sb-live');document.getElementById('start').classList.remove('hidden');"
                "TSUYAKU.setClose&&TSUYAKU.setClose();"
                "document.getElementById('meet').hidden=true;TSUYAKU.sideSet(true,false);")
        self.js(setup.replace("document.body.classList.add('live'", "document.getElementById('start').classList.add('hidden');document.body.classList.add('live'"))   # 会話を始めた時と同じく起動画面を隠す
        self.pump(0.1)
        return json.loads(self.js(PROBE))

    def check(self, name, setup, side):
        r = self.toggles(setup)
        self.assertEqual(r["side"], side, name)
        self.assertEqual(len(r["t"]), 1, f"{name}: 見えるボタン {r['t']}")
        self.assertFalse(r["t"][0]["over"], f"{name}: ロゴに重なる {r['t']}")
        self.assertEqual(r["t"][0]["id"], "sbClose" if side else r["t"][0]["id"], name)   # 開いている間はサイドバーの頭のボタン

    def test_start_page(self):
        self.check("新しい会話・開", "", True)
        self.check("新しい会話・閉", "TSUYAKU.sideSet(false,false)", False)

    def test_meetings_and_settings(self):
        self.check("会議の記録・開", "TSUYAKU.meetShow(false)", True)
        self.check("会議の記録・閉", "TSUYAKU.meetShow(false);TSUYAKU.sideSet(false,false)", False)
        self.check("設定・開", "TSUYAKU.setOpen()", True)
        self.check("設定・閉", "TSUYAKU.setOpen();TSUYAKU.sideSet(false,false)", False)

    def test_live(self):
        for off in ("", "TSUYAKU.sideSet(false,false);"):      # 会話の前に閉じていても開いていても同じ
            self.check("会話中・閉" + off, off + "document.body.classList.add('live')", False)
            self.check("会話中・開" + off, off + "document.body.classList.add('live','sb-live')", True)
            self.check("会話中に会議メモ・閉" + off, off + "document.body.classList.add('live');TSUYAKU.meetShow(true)", False)
            self.check("会話中に会議メモ・開" + off, off + "document.body.classList.add('live','sb-live');TSUYAKU.meetShow(true)", True)

    def test_narrow_window(self):
        self.wv.setFrame_(NSMakeRect(0, 0, 900, 600))
        try:
            self.check("900・開", "", True)
            self.check("900・閉", "TSUYAKU.sideSet(false,false)", False)
            self.check("900・会話中", "document.body.classList.add('live')", False)
        finally:
            self.wv.setFrame_(NSMakeRect(0, 0, 1280, 800))


if __name__ == "__main__":
    unittest.main()
