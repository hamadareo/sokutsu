"""Bridge の W2 の分(11/07): 録音・再生・ファイルの取り込み・字幕の書き出し。

__main__.Bridge はこの AudioAPI を継ぎ、会話の流れから次の所だけ呼ぶ:
  _conversation  SessionLog(..., audio=self._rec_wanted(mode)) → self._rec_open() → capture.tap = self._rec_tap
  _utterance_body log.add(..., audio=self._rec_span(tm))   (tm["span"] = 区間のマイク通算秒)
  end_meeting / _shutdown / _run の失敗   self._rec_stop()
  _started       self._w2_started(window)   (前回落ちた録音の変換・ドラッグ&ドロップ)
画面へは ui.put(("w2", {...})) → TSUYAKU.w2({...})。
"""
from __future__ import annotations

import re
import sys
import threading
import time
from pathlib import Path

from . import notes as notes_mod
from . import recorder as rec_mod
from . import subtitles as subs_mod
from .recorder import Recorder

LANGS = ("ja", "en", "zh", "ko")


def _dialog(kind: str, fallback: int):
    """pywebview 6 の FileDialog.OPEN/SAVE(古い OPEN_DIALOG/SAVE_DIALOG は非推奨)。"""
    try:
        import webview
        return getattr(webview.FileDialog, kind)
    except Exception:  # noqa: BLE001
        return fallback


class AudioAPI:
    # ---------- 録音(会話中) ----------

    def _rec_wanted(self, mode: str) -> bool:
        return rec_mod.should_record(getattr(self.s, "record_audio", "auto"), mode)

    def _rec_open(self) -> None:
        """記録(SessionLog)が録音つきで開いたら録り始める。失敗しても会話は続ける。"""
        self._rec = None
        log = getattr(self, "log", None)
        if log is None or not getattr(log, "audio_caf", None):
            return
        try:
            self._rec = Recorder(log.audio_caf, ts0=log.ts0)
            self.ui.put(("w2", {"type": "rec", "on": True}))
        except Exception as e:  # noqa: BLE001
            print(f"[rec] 録音を始められません: {e}", flush=True)

    def _rec_tap(self, mono, paused: bool = False) -> None:
        r = getattr(self, "_rec", None)
        if r is not None:
            r.tap(mono, paused)

    def _rec_span(self, tm) -> "tuple | None":
        """発言の区間(マイク通算秒) → 録音の中の (始め, 終わり) 秒。"""
        r = getattr(self, "_rec", None)
        sp = (tm or {}).get("span") if isinstance(tm, dict) else None
        if r is None or not sp:
            return None
        try:
            a0, a1 = r.audio_offset(sp[0]), r.audio_offset(sp[1])
        except Exception:  # noqa: BLE001
            return None
        return (a0, a1) if a0 is not None and a1 is not None else None

    def _rec_stop(self, wait: bool = False) -> None:
        """録音を閉じ、裏で m4a にする(1時間で約2秒)。できたら画面に知らせる(再生中なら差し替え)。"""
        r = self.__dict__.pop("_rec", None)
        if r is None:
            return
        try:
            r.close()
        except Exception:  # noqa: BLE001
            pass
        self.ui.put(("w2", {"type": "rec", "on": False}))
        mid = r.path.name.split(".audio.")[0]

        def fin():
            t0 = time.time()
            out = rec_mod.finalize(r.path)
            if out is not None:
                kb = out.stat().st_size // 1024
                print(f"[rec] {out.name} {r.seconds / 60:.1f}分 {kb}KB 変換{time.time() - t0:.1f}秒",
                      flush=True)
                self.ui.put(("w2", {"type": "audio_ready", "id": mid}))
            else:
                print(f"[rec] m4a にできませんでした。CAF を残します: {r.path.name}", flush=True)
        th = threading.Thread(target=fin, daemon=True)
        th.start()
        if wait:
            th.join(timeout=60)

    # ---------- 起動時 ----------

    def _w2_started(self, window) -> None:
        def boot():
            try:
                done = rec_mod.recover(self.s.log_dir)
                if done:
                    print(f"[rec] 前回の録音を変換しました: {[p.name for p in done]}", flush=True)
            except Exception:  # noqa: BLE001
                pass
            try:
                from .importer import cleanup_stale
                gone = cleanup_stale(self.s.log_dir)
                if gone:
                    print(f"[import] 前回の取り込みの作業ファイルを消しました: {len(gone)}個", flush=True)
            except Exception:  # noqa: BLE001
                pass
        threading.Thread(target=boot, daemon=True).start()
        self._w2_drop(window)

    def _w2_drop(self, window) -> None:
        """窓へのドラッグ&ドロップ: pywebview の DOM の drop でファイルの場所を受け取る
        (WKWebView の JS からはファイルの場所が見えない)。画面には TSUYAKU.w2({type:"drop"})。"""
        if window is None:
            return
        try:
            from webview.dom import DOMEventHandler
        except Exception:  # noqa: BLE001
            return

        def on_drop(e):
            files = ((e or {}).get("dataTransfer") or {}).get("files") or []
            paths = [f.get("pywebviewFullPath") for f in files if f.get("pywebviewFullPath")]
            self.ui.put(("w2", {"type": "drop", "paths": paths,
                                "names": [Path(p).name for p in paths],
                                "ok": [self._import_supported(p) for p in paths]}))

        def bind():
            try:
                window.dom.document.events.drop += DOMEventHandler(on_drop, True, False)
            except Exception as e:  # noqa: BLE001
                print(f"[import] ドラッグ&ドロップを使えません: {e}", flush=True)
        try:
            window.events.loaded += bind
        except Exception:  # noqa: BLE001
            pass

    # ---------- 再生(JS から) ----------

    def _media_server(self):
        srv = getattr(self, "_media", None)
        if srv is None:
            from .mediaserver import MediaServer
            srv = self._media = MediaServer(self.s.log_dir)
        return srv

    def audio_info(self, mid):
        """会議の録音: {url, dur, ts0, growing}。録音が無ければ None。"""
        p = self._meeting_path(mid)
        if p is None:
            return None
        f = rec_mod.audio_file(p)
        if f is None:
            return None
        meta = notes_mod.load_meeting(p)["meta"] if p.exists() else {}
        growing = f.suffix == ".caf"
        dur = rec_mod.duration(f) if not growing else max(0, f.stat().st_size - 68) / 32000
        try:
            url = self._media_server().url(f)
        except Exception as e:  # noqa: BLE001
            print(f"[rec] 再生の用意に失敗: {e}", flush=True)
            return None
        return {"url": url, "dur": dur, "ts0": meta.get("ts0"), "growing": growing,
                "live": getattr(getattr(self, "_rec", None), "path", None) == f}

    # ---------- 字幕・テキストの書き出し ----------

    def subtitles_text(self, mid, fmt="srt", ui="ja", with_trans=True):
        p = self._meeting_path(mid)
        if p is None:
            return ""
        if mid == "live" and (lg := getattr(self, "log", None)) is not None:   # 閉じる瞬間に None になっても落ちない
            lg.flush()
        try:
            return subs_mod.export(p, str(fmt), ui if ui in LANGS else "ja", bool(with_trans))[0]
        except Exception as e:  # noqa: BLE001
            print(f"[subs] 失敗: {e}", flush=True)
            return ""

    def subtitles_export(self, mid, fmt="srt", ui="ja", with_trans=True):
        """保存先を選んで書き出す(既定はダウンロード)。書いたパス、やめたら ""。"""
        p = self._meeting_path(mid)
        if p is None:
            return ""
        if mid == "live" and (lg := getattr(self, "log", None)) is not None:   # 閉じる瞬間に None になっても落ちない
            lg.flush()
        body, ext, title = subs_mod.export(p, str(fmt), ui if ui in LANGS else "ja", bool(with_trans))
        safe = re.sub(r'[\\/:*?"<>|\n\r\t]+', " ", str(title)).strip()[:80] or p.stem
        name = f"{safe}.{ext}"
        out = None
        win = getattr(self, "window", None)
        if win is not None:
            try:
                dl = Path.home() / "Downloads"
                r = win.create_file_dialog(_dialog("SAVE", 30),
                                           directory=str(dl if dl.exists() else Path.home()),
                                           save_filename=name)
                if not r:
                    return ""
                out = Path(r if isinstance(r, str) else r[0])
            except Exception as e:  # noqa: BLE001
                print(f"[subs] 保存の窓を開けません: {e}", flush=True)
        if out is None:   # 記録の隣に(session-X.txt は記録の人間向けのログなので、その名前は使わない)
            out = p.with_name(f"{p.stem}.subtitles.{ext}")
        out.write_text(body, encoding="utf-8")
        return str(out)

    # ---------- ファイルの取り込み ----------

    def _import_supported(self, path) -> bool:
        from .importer import supported
        return bool(path) and supported(path) and Path(path).is_file()

    def import_pick(self):
        """ファイルを選ぶ窓。選んだら {path, name, ok}、やめたら None。"""
        win = getattr(self, "window", None)
        if win is None:
            return None
        try:
            from .importer import EXTS
            pat = ";".join("*" + e for e in EXTS)
            r = win.create_file_dialog(_dialog("OPEN", 10),
                                       directory=str(Path.home()), allow_multiple=False,
                                       file_types=(f"音声・動画 ({pat})", "すべて (*.*)"))
        except Exception as e:  # noqa: BLE001
            print(f"[import] ファイルを選ぶ窓を開けません: {e}", flush=True)
            return None
        if not r:
            return None
        path = r if isinstance(r, str) else r[0]
        return {"path": path, "name": Path(path).name, "ok": self._import_supported(path),
                "size": Path(path).stat().st_size if Path(path).exists() else 0}

    def _app(self):
        """Bridge を定義したモジュール(python -m tsuyaku では __main__)。_wake_stt などを借りる。"""
        return sys.modules[type(self).__module__]

    def _import_ear(self):
        """取り込みの耳: 処理場所が リモートサーバーを使えるなら リモートサーバーの耳(並列3本)、無ければこの Mac。"""
        _wake_stt = self._app()._wake_stt
        from .stt import RemoteSTT, make_local_stt
        s = self.s
        local = make_local_stt(s)
        url = None
        if s.engine != "mac" and s.stt_url and _wake_stt(s.stt_url):
            url = s.stt_url
        stt = RemoteSTT(url, local)
        return stt, local, url

    def import_start(self, path, pair="ja-en", topic=""):
        """取り込みを裏で始める。進み具合は TSUYAKU.w2({type:"import", stage, pct, ...})。"""
        if getattr(self, "_import_th", None) is not None and self._import_th.is_alive():
            return "busy"
        if self.running.is_set():
            return "live"     # 会話中は耳と CPU を会話に使う
        if not self._import_supported(path):
            return "unsupported"
        langs = [x for x in str(pair or "").split("-") if x in LANGS][:2] or ["ja"]
        pair = "-".join(langs)
        self._import_cancel = threading.Event()
        self._import_th = threading.Thread(target=self._import_run,
                                           args=(str(path), pair, str(topic or "")), daemon=True)
        self._import_th.start()
        return "started"

    def import_cancel(self):
        ev = getattr(self, "_import_cancel", None)
        if ev is not None:
            ev.set()
        return "ok"

    def _import_stop(self, timeout: float = 8.0) -> None:
        """終了時: 取り込み中なら中止して、作りかけの会議と作業ファイルを消し終わるのを待つ
        (この後の os._exit では run_import の後片付けが走らず、途中までの会議が一覧に残っていた)。"""
        th = getattr(self, "_import_th", None)
        if th is None or not th.is_alive():
            return
        self.import_cancel()
        th.join(timeout)

    def _import_run(self, path: str, pair: str, topic: str) -> None:
        from .audio import SpeechCapture
        from .guards import stt_prompt
        from .importer import ImportFailed, run_import
        s = self.s
        name = Path(path).name
        t0 = time.time()

        def ev(**k):
            self.ui.put(("w2", dict({"type": "import", "file": name}, **k)))

        def prog(stage, pct, info=None):
            ev(stage=stage, pct=pct, info=info or {}, el=round(time.time() - t0, 1))

        try:
            prog("prepare", 0.0)
            stt, local, url = self._import_ear()
            stt.langs = tuple(pair.split("-"))
            local.langs = stt.langs
            stt.prompt = stt_prompt(topic)
            fine = url is not None or getattr(local, "fast", False)
            ear = "リモートGPU" if url else ("このMac(SenseVoice)" if getattr(local, "fast", False) else "このMac")
            prog("prepare", 0.0, {"ear": ear})
            spk = None
            try:
                spk = self._app()._speaker_tracker(s)
            except Exception:  # noqa: BLE001
                spk = None

            def make_cap(q, _x):
                return SpeechCapture(s.silero_vad, q, s.sample_rate, s.min_speech_s, s.max_speech_s,
                                     s.min_silence_s, vad_threshold=s.vad_threshold, fine=fine)

            r = run_import(path, s.log_dir, pair, topic, stt=stt, make_capture=make_cap, spk=spk,
                           prog=prog, cancel=self._import_cancel, fine=fine,
                           workers=3 if url else 1, hold_open_s=s.hold_open_s,
                           noise_floor=s.noise_rms_floor, prompt=stt.prompt)
            el = time.time() - t0
            if r is None:
                ev(stage="cancel", pct=0)
                print(f"[import] 中止 {name}", flush=True)
                return
            print(f"[import] {name} {r['dur'] / 60:.1f}分 → {r['lines']}行 {el:.1f}秒 "
                  f"(実時間の{r['dur'] / max(el, 0.01):.0f}倍) 耳={ear}", flush=True)
            ev(stage="done", pct=1.0, id=r["id"], lines=r["lines"], dur=r["dur"], el=round(el, 1), ear=ear)
        except ImportFailed as e:
            ev(stage="error", error=str(e))
        except Exception as e:  # noqa: BLE001
            print(f"[import] 失敗 {type(e).__name__}: {e}", flush=True)
            ev(stage="error", error=f"{type(e).__name__}: {str(e)[:160]}")

    # ---------- 録音の設定(起動画面) ----------

    def get_record(self):
        return getattr(self.s, "record_audio", "auto")

    def set_record(self, v):
        v = v if v in ("auto", "on", "off") else "auto"
        self.s.record_audio = v
        from .config import save_config
        save_config(self.s)
        return v
