"""macOS のメニューバー(11/08)。Mac のアプリらしく「Sokutsu」の名前で、画面の言語(ja/en/zh/ko)に合わせる。

見つけた問題: 開発用の起動(/Applications/Sokutsu.app → run.sh → .venv/bin/python -m tsuyaku)では、
メニューバーの太字のアプリ名が「Python」だった(CFBundleName は Python.app のもの)。メニューも pywebview の
既定の「編集」「表示」だけで、⌘Q は NSApp の terminate: が直接プロセスを終わらせ、Bridge._shutdown
(最後の発言の書き出し・録音を閉じる・ログの書き出し)を通らなかった。

作り:
- prepare_app(): NSApplication が起動を終える前に、自分の bundle の情報の CFBundleName と
  プロセス名を「Sokutsu」にする(メニューバーの太字の名前はここから読まれる)。Dock のアイコンも。
- MENUS: メニューの定義(4言語の題・キー・何をするか)。ここが唯一の元。
  ・"sel" … AppKit の標準の動き(隠す・カット・しまう …)。応答の連鎖に任せる
  ・"js"  … 画面のボタンと同じ処理(TSUYAKU.menuDo(id) → 既存の関数)。論理は二重に持たない
  ・"py"  … Python 側(Sokutsu について)
- build_menu(lang, target): NSMenu を作る(窓が無くても作れる。試験で中身を書き出して確かめる)。
- MenuBar: 画面から届く状態(画面の言語・会話中か・会議を開いているか・明るさ…)で、言語が変われば作り直し、
  使えない項目は灰色にする(validateMenuItem_)。押されたらメインの外のスレッドから evaluate_js。
- attach(bridge, window): 窓が閉じる・⌘Q・Dock の「終了」(どれも pywebview の closing を通る)で
  bridge._shutdown() を通す。メニューは窓が出た後にメインスレッドで差し替える。
"""
from __future__ import annotations

import json
import threading
from pathlib import Path

APP_NAME = "Sokutsu"
LANGS = ("ja", "en", "zh", "ko")
ICON = Path(__file__).resolve().parent / "assets" / "icon_1024.png"

# キーの修飾(AppKit の NSEventModifierFlag と同じ値。AppKit が無くても定義を読めるように数で持つ)
SHIFT, CTRL, ALT, CMD = 1 << 17, 1 << 18, 1 << 19, 1 << 20


def T(ja: str, en: str, zh: str, ko: str) -> dict:
    return {"ja": ja, "en": en, "zh": zh, "ko": ko}


def I(id_: str, t: dict, kind: str, key: str = "", mods: int = 0, sel: str = "", when: str = "", check: str = "") -> dict:
    """メニューの1項目。when = 使える時(画面の状態の名前、"!" で否定)。check = 印を付ける時。"""
    return {"id": id_, "t": t, "kind": kind, "key": key, "mods": mods if key else 0, "sel": sel, "when": when, "check": check}


SEP = None
N = APP_NAME
# 会話中の操作の一字のキー(B・M・V・Space)は、文字の入力を横取りしないよう題に添えるだけにする(メニューのキーにしない)
MENUS = [
    ("app", T(N, N, N, N), [
        I("about", T(f"{N} について", f"About {N}", f"关于 {N}", f"{N}에 관하여"), "py"),
        SEP,
        I("settings", T("設定…", "Settings…", "设置…", "설정…"), "js", ",", CMD),   # 会話中も開ける(右の板。会話は止めない)
        SEP,
        I("services", T("サービス", "Services", "服务", "서비스"), "services"),
        SEP,
        I("hide", T(f"{N} を隠す", f"Hide {N}", f"隐藏 {N}", f"{N} 가리기"), "sel", "h", CMD, sel="hide:"),
        I("hideOthers", T("ほかを隠す", "Hide Others", "隐藏其他", "기타 가리기"), "sel", "h", CMD | ALT, sel="hideOtherApplications:"),
        I("showAll", T("すべてを表示", "Show All", "全部显示", "모두 보기"), "sel", sel="unhideAllApplications:"),
        SEP,
        I("quit", T(f"{N} を終了", f"Quit {N}", f"退出 {N}", f"{N} 종료"), "sel", "q", CMD, sel="terminate:"),
    ]),
    ("file", T("ファイル", "File", "文件", "파일"), [
        I("newConv", T("新しい会話", "New Conversation", "新对话", "새 대화"), "js", "n", CMD, when="!live"),
        I("importAudio", T("音声ファイルを取り込む…", "Import Audio File…", "导入音频文件…", "오디오 파일 가져오기…"), "js", when="!live"),
        I("export", T("この会議を書き出す…", "Export This Meeting…", "导出此会议…", "이 회의 내보내기…"), "js", when="doc"),
        SEP,
        I("endConv", T("会話を終える", "End Conversation", "结束对话", "대화 끝내기"), "js", when="live"),
        SEP,
        I("close", T("閉じる", "Close", "关闭", "닫기"), "js", "w", CMD, when="layer"),
    ]),
    ("edit", T("編集", "Edit", "编辑", "편집"), [
        I("undo", T("取り消す", "Undo", "撤销", "실행 취소"), "sel", "z", CMD, sel="undo:"),
        I("redo", T("やり直す", "Redo", "重做", "실행 복귀"), "sel", "z", CMD | SHIFT, sel="redo:"),
        SEP,
        I("cut", T("カット", "Cut", "剪切", "오려두기"), "sel", "x", CMD, sel="cut:"),
        I("copy", T("コピー", "Copy", "拷贝", "복사하기"), "sel", "c", CMD, sel="copy:"),
        I("paste", T("ペースト", "Paste", "粘贴", "붙여넣기"), "sel", "v", CMD, sel="paste:"),
        I("selectAll", T("すべてを選択", "Select All", "全选", "전체 선택"), "sel", "a", CMD, sel="selectAll:"),
        SEP,
        I("find", T("会議を検索…", "Find in Meetings…", "搜索会议…", "회의 검색…"), "js", "f", CMD, when="!live"),
    ]),
    ("view", T("表示", "View", "显示", "보기"), [
        I("sidebar", T("サイドバーの表示・非表示", "Toggle Sidebar", "显示/隐藏侧边栏", "사이드바 보기/가리기"), "js", "\\", CMD),
        I("palette", T("すべてのコマンド…", "Command Palette…", "所有命令…", "모든 명령…"), "js", "k", CMD),
        SEP,
        I("larger", T("字幕を大きく", "Larger Captions", "放大字幕", "자막 크게"), "js", "+", CMD, when="live"),
        I("smaller", T("字幕を小さく", "Smaller Captions", "缩小字幕", "자막 작게"), "js", "-", CMD, when="live"),
        SEP,
        ("appearance", T("画面の明るさ", "Appearance", "外观", "화면 모드"), [
            I("themeSystem", T("システムに合わせる", "Match System", "跟随系统", "시스템에 맞춤"), "js", check="theme=system"),
            I("themeLight", T("明るい", "Light", "浅色", "라이트"), "js", check="theme=light"),
            I("themeDark", T("暗い", "Dark", "深色", "다크"), "js", check="theme=dark"),
        ]),
        SEP,
        I("fullScreen", T("フルスクリーンにする", "Enter Full Screen", "进入全屏幕", "전체 화면 시작"), "sel", "f", CMD | CTRL, sel="toggleFullScreen:"),
    ]),
    ("conv", T("会話", "Conversation", "对话", "대화"), [
        I("start", T("会話を始める", "Start Conversation", "开始对话", "대화 시작"), "js", "\r", CMD, when="start"),
        I("stopVoice", T("読み上げを止める(Space)", "Stop Voice (Space)", "停止朗读(Space)", "읽기 중지(Space)"), "js", when="speaking"),
        I("voice", T("声のあり・なし(V)", "Voice On/Off (V)", "朗读开/关(V)", "음성 켜기/끄기(V)"), "js", when="live"),
        SEP,
        I("bookmark", T("印を付ける(B)", "Bookmark (B)", "标记(B)", "표시하기(B)"), "js", when="live"),
        I("note", T("自分のメモ(M)", "My Note (M)", "我的笔记(M)", "내 메모(M)"), "js", when="live"),
        SEP,
        I("endConv", T("会話を終える", "End Conversation", "结束对话", "대화 끝내기"), "js", when="live"),
        SEP,
        # 入力・出力デバイス(11/08 hear): 開くたびに今のデバイスで作り直す(MenuBar.fill_devices)
        I("inputDev", T("入力", "Input", "输入", "입력"), "devices"),
        I("outputDev", T("出力", "Output", "输出", "출력"), "devices"),
    ]),
    ("window", T("ウインドウ", "Window", "窗口", "윈도우"), [
        I("minimize", T("しまう", "Minimize", "最小化", "최소화"), "sel", "m", CMD, sel="performMiniaturize:"),
        I("zoom", T("拡大/縮小", "Zoom", "缩放", "확대/축소"), "sel", sel="performZoom:"),
        SEP,
        I("front", T("すべてを手前に移動", "Bring All to Front", "前置全部窗口", "모두 앞으로 가져오기"), "sel", sel="arrangeInFront:"),
    ]),
    ("help", T("ヘルプ", "Help", "帮助", "도움말"), [
        I("help", T("操作の早見", "Quick Help", "操作速查", "조작 빠른 보기"), "js"),
        I("keys", T("キーボードショートカット", "Keyboard Shortcuts", "键盘快捷键", "키보드 단축키"), "js"),
        SEP,
        I("terms", T("用語辞書…", "Terms Dictionary…", "术语词典…", "용어 사전…"), "js"),
    ]),
]


def walk(menus=MENUS):
    """すべての項目(区切りを除く)を順に。"""
    for m in menus:
        yield from _walk_items(m[2])


def _walk_items(items):
    for it in items:
        if it is None:
            continue
        if isinstance(it, tuple):
            yield from _walk_items(it[2])
        else:
            yield it


def enabled(item: dict, state: dict) -> bool:
    w = item.get("when") or ""
    if not w:
        return True
    neg = w.startswith("!")
    v = bool(state.get(w.lstrip("!")))
    return not v if neg else v


def checked(item: dict, state: dict) -> bool:
    c = item.get("check") or ""
    if not c:
        return False
    k, _, v = c.partition("=")
    return str(state.get(k, "")) == v


def key_label(item: dict) -> str:
    """試験・書き出し用のキーの書き方(⌃⌥⇧⌘ + 文字)。"""
    if not item.get("key"):
        return ""
    m = item["mods"]
    s = ("⌃" if m & CTRL else "") + ("⌥" if m & ALT else "") + ("⇧" if m & SHIFT else "") + ("⌘" if m & CMD else "")
    k = {"\r": "↩", " ": "Space"}.get(item["key"], item["key"].upper())
    return s + k


def lang_of(ui: str) -> str:
    return ui if ui in LANGS else "en"


# ---------------- AppKit ----------------

def prepare_app(name: str = APP_NAME, icon: Path = ICON) -> bool:
    """NSApplication が起動を終える前に呼ぶ。メニューバーの太字の名前・プロセス名・Dock のアイコン。
    AppKit が無い・読めない時も起動は止めない。"""
    try:
        from Foundation import NSBundle, NSProcessInfo
        b = NSBundle.mainBundle()
        for d in (b.localizedInfoDictionary(), b.infoDictionary()):
            if d is not None:
                d["CFBundleName"] = name
                d["CFBundleDisplayName"] = name
        NSProcessInfo.processInfo().setProcessName_(name)
    except Exception:  # noqa: BLE001
        return False
    try:
        from AppKit import NSApplication, NSImage
        img = NSImage.alloc().initWithContentsOfFile_(str(icon))
        if img is not None:
            NSApplication.sharedApplication().setApplicationIconImage_(img)
    except Exception:  # noqa: BLE001
        pass
    return True


def version() -> str:
    try:
        from . import distenv
        if distenv.DIST:
            return str(distenv.VERSION)
    except Exception:  # noqa: BLE001
        pass
    from . import __version__
    return f"{__version__} (dev)"


_TARGET_CLS = None


def _target_class():
    """押された項目を MenuBar へ渡し、使えるかを答える NSObject(1つのプロセスで1回だけ作る)。"""
    global _TARGET_CLS
    if _TARGET_CLS is not None:
        return _TARGET_CLS
    import objc
    from Foundation import NSObject

    class SokutsuMenuTarget(NSObject):
        def initWithOwner_(self, owner):
            self = objc.super(SokutsuMenuTarget, self).init()
            if self is None:
                return None
            self.owner = owner
            return self

        @objc.IBAction
        def menuAction_(self, sender):
            try:
                self.owner.run(str(sender.representedObject()))
            except Exception as e:  # noqa: BLE001
                print(f"[menu] {type(e).__name__}: {e}", flush=True)

        def menuNeedsUpdate_(self, menu):
            """入力・出力の下の階層を開く直前: 今のデバイスで作り直す(11/08 hear)。"""
            try:
                sup = menu.supermenu()
                i = sup.indexOfItemWithSubmenu_(menu) if sup is not None else -1
                rid = str(sup.itemAtIndex_(i).representedObject() or "") if i >= 0 else ""
                kind = {"inputDev": "input", "outputDev": "output"}.get(rid)
                if kind:
                    self.owner.fill_devices(menu, kind)
            except Exception as e:  # noqa: BLE001
                print(f"[menu] {type(e).__name__}: {e}", flush=True)

        def validateMenuItem_(self, item):
            try:
                return bool(self.owner.can(str(item.representedObject())))
            except Exception:  # noqa: BLE001
                return True

    _TARGET_CLS = SokutsuMenuTarget
    return _TARGET_CLS


def build_menu(lang: str, target=None, state: "dict | None" = None):
    """NSMenu(メニューバー全体)を作る。窓が無くても作れる。"""
    from AppKit import NSMenu, NSMenuItem, NSOnState, NSOffState
    lang = lang_of(lang)
    state = state or {}
    main = NSMenu.alloc().initWithTitle_("MainMenu")
    refs = {"services": None, "window": None, "help": None}

    def add_items(menu, items):
        for it in items:
            if it is None:
                menu.addItem_(NSMenuItem.separatorItem())
                continue
            if isinstance(it, tuple):
                sub = NSMenu.alloc().initWithTitle_(it[1][lang])
                mi = NSMenuItem.alloc().initWithTitle_action_keyEquivalent_(it[1][lang], None, "")
                mi.setSubmenu_(sub)
                menu.addItem_(mi)
                add_items(sub, it[2])
                continue
            title = it["t"][lang]
            if it["kind"] == "devices":
                sub = NSMenu.alloc().initWithTitle_(title)
                mi = NSMenuItem.alloc().initWithTitle_action_keyEquivalent_(title, None, "")
                mi.setRepresentedObject_(it["id"])
                mi.setSubmenu_(sub)
                if target is not None:
                    sub.setDelegate_(target)          # menuNeedsUpdate: で開くたびに作り直す
                menu.addItem_(mi)
                continue
            if it["kind"] == "services":
                sub = NSMenu.alloc().initWithTitle_(title)
                mi = NSMenuItem.alloc().initWithTitle_action_keyEquivalent_(title, None, "")
                mi.setSubmenu_(sub)
                menu.addItem_(mi)
                refs["services"] = sub
                continue
            sel = it["sel"] if it["kind"] == "sel" else "menuAction:"
            mi = NSMenuItem.alloc().initWithTitle_action_keyEquivalent_(title, sel, it["key"])
            if it["key"]:
                mi.setKeyEquivalentModifierMask_(it["mods"])
            mi.setRepresentedObject_(it["id"])
            if it["kind"] != "sel" and target is not None:
                mi.setTarget_(target)
            if it.get("check"):
                mi.setState_(NSOnState if checked(it, state) else NSOffState)
            menu.addItem_(mi)

    for key, title, items in MENUS:
        sub = NSMenu.alloc().initWithTitle_(title[lang])
        top = NSMenuItem.alloc().initWithTitle_action_keyEquivalent_(title[lang], None, "")
        top.setSubmenu_(sub)
        main.addItem_(top)
        add_items(sub, items)
        if key in ("window", "help"):
            refs[key] = sub
    return main, refs


def dump(menu, depth: int = 0) -> str:
    """NSMenu の中身を文字に(試験・報告用)。灰色の項目は (使えない)、印は ✓。"""
    from AppKit import NSOnState
    out = []
    for i in range(menu.numberOfItems()):
        mi = menu.itemAtIndex_(i)
        if mi.isSeparatorItem():
            out.append("  " * depth + "----")
            continue
        k = ""
        if mi.keyEquivalent():
            m = mi.keyEquivalentModifierMask()
            k = ("⌃" if m & CTRL else "") + ("⌥" if m & ALT else "") + ("⇧" if m & SHIFT else "") + ("⌘" if m & CMD else "")
            k += {"\r": "↩", " ": "Space"}.get(mi.keyEquivalent(), mi.keyEquivalent().upper())
        mark = "✓ " if mi.state() == NSOnState else ""
        t = mi.target()
        dis = ""
        if t is not None and hasattr(t, "validateMenuItem_") and not t.validateMenuItem_(mi):
            dis = "  (使えない)"
        out.append("  " * depth + mark + str(mi.title()) + (f"   {k}" if k else "") + dis)
        if mi.hasSubmenu() and mi.submenu() is not None:
            out.append(dump(mi.submenu(), depth + 1))
    return "\n".join(x for x in out if x)


class MenuBar:
    """メニューバーの持ち主。状態は画面(JS)から menu_state で届く。AppKit の操作はメインスレッドだけで。"""

    def __init__(self, window=None, lang: str = "ja"):
        self.window = window
        self.lang = lang_of(lang)
        self.state: dict = {"ui": self.lang, "theme": "system"}
        self.items = {it["id"]: it for it in walk()}
        self.target = None
        self.menu = None

    # どのスレッドからでも
    def set_state(self, st: dict) -> None:
        if not isinstance(st, dict):
            return
        old_lang = self.lang
        self.state.update({k: st[k] for k in st if isinstance(k, str)})
        self.lang = lang_of(str(self.state.get("ui") or self.lang))
        if self.menu is None:
            return
        _main_thread(self.install if self.lang != old_lang else self.refresh)

    def can(self, id_: str) -> bool:
        it = self.items.get(id_)
        return True if it is None else enabled(it, self.state)

    SYS_DEV = T("システムの設定に合わせる", "Match System Settings", "跟随系统设置", "시스템 설정에 맞추기")

    def fill_devices(self, menu, kind: str) -> None:
        """入力・出力の下の階層: 「システムの設定に合わせる」(今の既定の名前つき)+ 今のデバイス。選んだものに印。"""
        from AppKit import NSMenuItem, NSOnState, NSOffState
        from . import devices
        b = getattr(self, "bridge", None)
        cur = getattr(getattr(b, "s", None), "input_device" if kind == "input" else "output_device", "")
        ds = devices.list_devices(devices.INPUT if kind == "input" else devices.OUTPUT)
        dflt = next((d["name"] for d in ds if d.get("isDefault")), "")
        menu.removeAllItems()
        title = self.SYS_DEV[self.lang] + (f"({dflt})" if dflt else "")
        rows = [(title, "")] + [None] + [(d["name"], d["name"]) for d in ds]
        for row in rows:
            if row is None:
                if ds:
                    menu.addItem_(NSMenuItem.separatorItem())
                continue
            t, name = row
            mi = NSMenuItem.alloc().initWithTitle_action_keyEquivalent_(t, "menuAction:", "")
            mi.setRepresentedObject_(f"dev:{kind}:{name}")
            if self.target is not None:
                mi.setTarget_(self.target)
            mi.setState_(NSOnState if name == (cur or "") else NSOffState)
            menu.addItem_(mi)

    def run(self, id_: str) -> None:
        if id_.startswith("dev:"):   # 入力・出力デバイスを選んだ(Bridge と同じ処理。保存と会話中の切り替え)
            _d, kind, name = id_.split(":", 2)
            b = getattr(self, "bridge", None)
            fn = getattr(b, "set_input_device" if kind == "input" else "set_output_device", None)
            if fn is not None:
                threading.Thread(target=self._choose, args=(fn, kind, name), daemon=True).start()
            return
        it = self.items.get(id_)
        if it is None:
            return
        if it["kind"] == "py":
            if id_ == "about":
                self.about()
            return
        if it["kind"] == "js" and self.window is not None:
            js = "window.TSUYAKU&&TSUYAKU.menuDo&&TSUYAKU.menuDo(" + json.dumps(id_) + ")"
            # evaluate_js はメインスレッドの返事を待つので、メインスレッドから直接は呼ばない
            threading.Thread(target=self._eval, args=(js,), daemon=True).start()

    def _choose(self, fn, kind: str, name: str) -> None:
        try:
            r = fn(name) or {}
            b = getattr(self, "bridge", None)
            if kind == "input" and b is not None and r.get("applied") == "next":
                # 会話していない: 画面の起動画面の欄も合わせる(会話中は開き直した後に capture が知らせる)
                b.ui.put(("audio_dev", {"kind": "input", "event": "switched", "want": name,
                                        "using": name, "applied": "next"}))
        except Exception as e:  # noqa: BLE001
            print(f"[menu] {type(e).__name__}: {e}", flush=True)

    def _eval(self, js: str) -> None:
        try:
            self.window.evaluate_js(js)
        except Exception as e:  # noqa: BLE001
            print(f"[menu] {type(e).__name__}: {e}", flush=True)

    # メインスレッドで
    def install(self) -> None:
        from AppKit import NSApplication
        if self.target is None:
            self.target = _target_class().alloc().initWithOwner_(self)
        self.menu, refs = build_menu(self.lang, self.target, self.state)
        app = NSApplication.sharedApplication()
        app.setMainMenu_(self.menu)
        if refs["services"] is not None:
            app.setServicesMenu_(refs["services"])
        if refs["window"] is not None:
            app.setWindowsMenu_(refs["window"])
        if refs["help"] is not None and hasattr(app, "setHelpMenu_"):
            app.setHelpMenu_(refs["help"])

    def refresh(self) -> None:
        """言語はそのまま: 印(明るさ)だけ付け直す。使えるかは開くたびに validateMenuItem_ が答える。"""
        if self.menu is None:
            return
        from AppKit import NSOnState, NSOffState

        def walk_menu(menu):
            for i in range(menu.numberOfItems()):
                mi = menu.itemAtIndex_(i)
                it = self.items.get(str(mi.representedObject() or ""))
                if it and it.get("check"):
                    mi.setState_(NSOnState if checked(it, self.state) else NSOffState)
                if mi.hasSubmenu() and mi.submenu() is not None:
                    walk_menu(mi.submenu())
        walk_menu(self.menu)

    def about(self) -> None:
        from AppKit import NSApplication, NSImage
        opts = {"ApplicationName": APP_NAME, "ApplicationVersion": version(), "Version": ""}
        img = NSImage.alloc().initWithContentsOfFile_(str(ICON))
        if img is not None:
            opts["ApplicationIcon"] = img
        app = NSApplication.sharedApplication()
        app.activateIgnoringOtherApps_(True)
        app.orderFrontStandardAboutPanelWithOptions_(opts)


def _main_thread(fn) -> None:
    try:
        from PyObjCTools import AppHelper
        AppHelper.callAfter(fn)
    except Exception as e:  # noqa: BLE001
        print(f"[menu] {type(e).__name__}: {e}", flush=True)


def system_lang() -> str:
    """画面から言語が届くまでの間の言語: macOS の優先する言語(ja/en/zh/ko 以外は en)。"""
    try:
        from Foundation import NSLocale
        for l in NSLocale.preferredLanguages() or []:
            k = str(l)[:2].lower()
            return k if k in LANGS else "en"
    except Exception:  # noqa: BLE001
        pass
    return "ja"


def attach(bridge, window, lang: "str | None" = None) -> MenuBar:
    """窓を作った後・webview.start の前に呼ぶ。
    1) 窓を閉じる・⌘Q・Dock の「終了」は、どれも pywebview の closing を通るので、そこで Bridge._shutdown を通す
    2) 窓が出たら(pywebview が既定のメニューを置いた後)、メインスレッドでこのメニューに差し替える"""
    mb = MenuBar(window, lang or system_lang())
    mb.bridge = bridge          # 入力・出力デバイスの選択(11/08 hear)
    bridge._menubar = mb
    try:
        window.events.closing += bridge._on_closing
    except Exception as e:  # noqa: BLE001
        print(f"[menu] closing {type(e).__name__}: {e}", flush=True)

    def shown():
        _main_thread(mb.install)
    try:
        window.events.shown += shown
    except Exception as e:  # noqa: BLE001
        print(f"[menu] shown {type(e).__name__}: {e}", flush=True)
    return mb
