"""配る版(他人の Mac 向け Sokutsu.app)の入口。

Sokutsu.app の起動役(native/launcher/launcher.c)が `-m tsuyaku.dist_main` で呼ぶ。
モデルが足りなければ初回の準備の画面(webui/setup.html)を開き、そろったら同じ窓・同じ js_api の
まま本画面(webui/index.html)へ移る。同梱の翻訳エンジン(localllm.py)は窓を閉じたら止める。
ユーザー本人の版(run.sh → python -m tsuyaku)はここを通らない。
"""
from __future__ import annotations

import os
import sys
import threading
from pathlib import Path

from . import config, distenv, localllm, modelfetch

WEBUI_DIR = Path(__file__).resolve().parent / "webui"


def _warm_llm() -> None:
    """翻訳エンジンを裏で起こしておく(「会話を始める」を押した時に待たない)。"""
    threading.Thread(target=localllm.ensure_for, args=(localllm.url(),), daemon=True).start()


class SetupMixin:
    """初回の準備の画面(setup.html)から呼ばれる。本画面の Bridge に混ぜる。"""

    def _setup_init(self, models: Path, fetcher=None) -> None:
        self._models = Path(models)
        self._fetch = fetcher or modelfetch.Fetcher(self._models)
        self._on_ready = None

    def setup_state(self):
        d = self._fetch.state()
        d["version"] = distenv.VERSION
        return d

    def setup_start(self):
        self._fetch.start()
        return "ok"

    def setup_cancel(self):
        self._fetch.cancel()
        return "ok"

    def setup_finish(self):
        if modelfetch.missing(self._models):
            return "missing"
        config.resolve_dist_models(self.s, config.DATA)   # 取り終えたモデルの場所を設定へ
        _warm_llm()
        if self._on_ready:
            self._on_ready()
        return "ok"


def _autostart(bridge, poll: float = 0.5) -> None:
    """試験用(SOKUTSU_AUTOSTART=1): 画面を押さずに取得を始め、そろったら本画面へ移る。
    失敗・一時停止なら画面に任せる(理由と「もう一度」が出る)。"""
    import time
    bridge.setup_start()
    while True:
        ph = bridge._fetch.state().get("phase")
        if ph == "done":
            bridge.setup_finish()
            return
        if ph in ("error", "cancelled"):
            return
        time.sleep(poll)


def _fast_loopback_fqdn() -> None:
    """pywebview の画面サーバー(wsgiref)は起動時に socket.getfqdn(host) を呼ぶ。同梱の Python
    (python-build-standalone)では 127.0.0.1 の逆引きに35秒かかり、窓が真っ白のままだった(11/07)。
    自分の Mac の中の名前だけは問い合わせずに答える。"""
    import socket
    orig = socket.getfqdn
    if getattr(orig, "_sokutsu", False):
        return

    def getfqdn(name: str = "") -> str:
        if name in ("127.0.0.1", "localhost", "::1", "0.0.0.0"):
            return "localhost"
        return orig(name)
    getfqdn._sokutsu = True
    socket.getfqdn = getfqdn


def _snap_requests(d: Path) -> list:
    out = []
    for req in sorted(Path(d).glob("req-*")):
        try:
            req.unlink()
        except OSError:
            continue
        out.append(req.name[4:])
    return out


def _snapshot_loop(window, d: Path) -> None:
    """試験用(SOKUTSU_SNAPSHOT=<dir>): <dir>/req-<名前> を置くと、窓の WebView の絵を
    <dir>/<名前>.png に残す(画面の収録の許可が無い環境でも、自分たちの画面を確かめるため)。"""
    import time
    import AppKit
    import WebKit
    from PyObjCTools import AppHelper

    def find_wk(v):
        if v is None:
            return None
        if v.isKindOfClass_(WebKit.WKWebView):
            return v
        for s in v.subviews() or []:
            r = find_wk(s)
            if r is not None:
                return r
        return None

    def take(name):
        try:
            from webview.platforms.cocoa import BrowserView
            inst = BrowserView.instances.get(window.uid)
            wv = getattr(inst, "webview", None) or find_wk(window.native.contentView())
        except Exception as e:  # noqa: BLE001
            print(f"[snap] {type(e).__name__}: {e}", flush=True)
            wv = None
        if wv is None:
            print(f"[snap] WebView が見つかりません({name})", flush=True)
            return

        def done(img, err):
            if img is None:
                print(f"[snap] 絵を取れませんでした({name}): {err}", flush=True)
                return
            rep = AppKit.NSBitmapImageRep.imageRepWithData_(img.TIFFRepresentation())
            png = rep.representationUsingType_properties_(AppKit.NSBitmapImageFileTypePNG, {})
            png.writeToFile_atomically_(str(Path(d) / f"{name}.png"), True)
            print(f"[snap] {name}.png", flush=True)
        wv.takeSnapshotWithConfiguration_completionHandler_(None, done)

    def dump(name):
        """画面の中身(URL・読み込み・本文の頭)を <name>.txt に。"""
        try:
            r = window.evaluate_js("JSON.stringify({u:location.href,r:document.readyState,"
                                   "bg:getComputedStyle(document.body).backgroundColor,"
                                   "t:(document.body.innerText||'').slice(0,600)})")
        except Exception as e:  # noqa: BLE001
            r = f"{type(e).__name__}: {e}"
        (Path(d) / f"{name}.txt").write_text(str(r), encoding="utf-8")

    while True:
        for name in _snap_requests(d):
            if name.startswith("js_"):
                threading.Thread(target=dump, args=(name,), daemon=True).start()
            else:
                AppHelper.callAfter(take, name)
        time.sleep(0.5)


def _alert(title: str, message: str) -> None:
    """起動できなかった理由を窓で見せる(pywebview・AppKit が読めない時でも出るよう osascript)。"""
    import subprocess
    try:
        subprocess.run(["/usr/bin/osascript", "-e", "on run argv",
                        "-e", "display alert (item 1 of argv) message (item 2 of argv) as critical",
                        "-e", "end run", title, message], timeout=600,
                       stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:  # noqa: BLE001
        pass


def main() -> int:
    if not distenv.DIST:
        print("[dist] 配る版ではありません(python -m tsuyaku で起動してください)", flush=True)
        return 2
    try:
        return _main()
    except Exception as e:  # noqa: BLE001
        # 窓が出る前の失敗(データの場所に書けない等)は、Dock のアイコンが跳ねて消えるだけだった(11/07 監査)
        import traceback
        traceback.print_exc()
        sys.stdout.flush()
        localllm.stop()
        log = distenv.app_logs_dir() / "run.log"
        _alert("Sokutsu を起動できませんでした / Sokutsu could not start",
               f"{type(e).__name__}: {e}\n\n記録 / Log: {log}")
        return 1


def _main() -> int:
    (config.DATA / "logs").mkdir(parents=True, exist_ok=True)
    _fast_loopback_fqdn()
    import webview
    from . import __main__ as app

    class DistBridge(SetupMixin, app.Bridge):
        def __init__(self):
            app.Bridge.__init__(self)
            self._setup_init(config.MODELS)

    print(f"[dist] Sokutsu {distenv.VERSION} / データ {config.DATA}", flush=True)
    from . import menubar
    menubar.prepare_app()
    bridge = DistBridge()
    app._set_dock_icon()
    need = bool(modelfetch.missing(config.MODELS))
    page = WEBUI_DIR / ("setup.html" if need else "index.html")
    window = webview.create_window("Sokutsu", str(page), js_api=bridge, width=1280, height=800,
                                   min_size=(900, 600), background_color="#0D1216")
    bridge.window = window
    menubar.attach(bridge, window)
    booted = threading.Event()

    def boot_main(win):
        if not booted.is_set():
            booted.set()
            bridge._started(win)

    def go_main():
        def loaded():
            boot_main(window)
        window.events.loaded += loaded
        window.load_url(str(WEBUI_DIR / "index.html"))

    bridge._on_ready = go_main

    def started(win):
        bridge.window = win
        if os.environ.get("SOKUTSU_SNAPSHOT"):
            threading.Thread(target=_snapshot_loop, args=(win, Path(os.environ["SOKUTSU_SNAPSHOT"])),
                             daemon=True).start()
        if not need:
            boot_main(win)
            _warm_llm()
        elif os.environ.get("SOKUTSU_AUTOSTART") == "1":
            threading.Thread(target=_autostart, args=(bridge,), daemon=True).start()

    webview.start(started, window)
    try:
        bridge._fetch.cancel()
        bridge._shutdown()
    except Exception:  # noqa: BLE001
        pass
    finally:
        localllm.stop()
        sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    sys.exit(main())
