"""画面(JS)から呼ぶ W4 の機能: 整理・全文検索・話者の分析・書き出し・声を覚える・カレンダー連携。

Bridge(__main__.py)がこのクラスを継承する(Bridge 本体への変更を小さくするため)。使うもの:
self.s.log_dir / self._meeting_path(mid) / self.log / self.spk / self.ui / self.window。
"""
from __future__ import annotations

import threading
import time
from pathlib import Path

from . import analytics as analytics_mod
from . import calendar_mac as cal_mod
from . import export as export_mod
from . import library as lib_mod
from . import notes as notes_mod
from . import search as search_mod
from . import voicebook as vb_mod


class LibraryApi:
    # ---------- 内部 ----------

    def _lib_dir(self) -> Path:
        return Path(self.s.log_dir)

    def _lib_live(self) -> "Path | None":
        pj = getattr(getattr(self, "log", None), "path_jsonl", None)
        return Path(pj) if pj else None

    def _lib_path(self, mid) -> "Path | None":
        return self._meeting_path(mid)

    def _search_index(self) -> search_mod.SearchIndex:
        ix = self.__dict__.get("_w4_ix")
        if ix is None or ix.log_dir != self._lib_dir():
            ix = search_mod.SearchIndex(self._lib_dir())
            self.__dict__["_w4_ix"] = ix
        return ix

    def _voicebook(self) -> vb_mod.VoiceBook:
        root = self._lib_dir() / "voices"
        b = self.__dict__.get("_w4_vb")
        if b is None or b.root != root:
            b = vb_mod.VoiceBook(root)
            self.__dict__["_w4_vb"] = b
        return b

    # ---------- 一覧・整理 ----------

    def library_list(self):
        """会議の一覧(題の変更・タグ・フォルダ・お気に入りつき)と絞り込みの候補。"""
        try:
            d = self._lib_dir()
            return {"meetings": lib_mod.list_meetings(d), "facets": lib_mod.facets(d)}
        except Exception as e:  # noqa: BLE001
            print(f"[lib] 一覧を読めません: {e}", flush=True)
            return {"meetings": [], "facets": {}}

    def library_facets(self):
        try:
            return lib_mod.facets(self._lib_dir())
        except Exception:  # noqa: BLE001
            return {}

    def meeting_info_set(self, mid, fields):
        """題・タグ・フォルダ・お気に入りを変える。fields = {title?, tags?, folder?, star?}。"""
        p = self._lib_path(mid)
        if p is None or not isinstance(fields, dict):
            return None
        kw = {k: fields[k] for k in ("title", "tags", "folder", "star") if k in fields}
        info = lib_mod.update_info(p, **kw)
        if info is not None and kw.get("folder"):
            lib_mod.folder_add(self._lib_dir(), kw["folder"])
        return info

    def folder_add(self, name):
        return lib_mod.folder_add(self._lib_dir(), name)

    def folder_rename(self, old, new):
        return lib_mod.folder_rename(self._lib_dir(), old, new)

    def folder_remove(self, name):
        lib_mod.folder_remove(self._lib_dir(), name)
        return "ok"

    def tag_rename(self, old, new):
        lib_mod.tag_rename(self._lib_dir(), old, new or "")
        return "ok"

    def meeting_trash(self, mid):
        """ゴミ箱へ(今の会話は捨てない)。"""
        if mid in (None, "", "live"):
            return "live"
        if str(mid) in self.library_busy()["busy"]:   # 取り込み中(まとめての meetings_trash と同じ扱い)
            return "busy"
        ok = lib_mod.trash(self._lib_dir(), str(mid), live=self._lib_live())
        return "ok" if ok else "none"

    def meeting_restore(self, mid):
        return "ok" if lib_mod.restore(self._lib_dir(), str(mid)) else "none"

    def trash_list(self):
        try:
            return lib_mod.trash_list(self._lib_dir())
        except Exception:  # noqa: BLE001
            return []

    def library_busy(self):
        """選べない会議: 今の会話(live)と、取り込み中の記録(取り込みが動いていて30秒以内に書かれたもの)。"""
        live = self._lib_live()
        busy = {live.stem} if live is not None else set()
        th = getattr(self, "_import_th", None)
        if th is not None and th.is_alive():
            now = time.time()
            for f in self._lib_dir().glob("session-*.jsonl"):
                try:
                    if now - f.stat().st_mtime < 30:
                        busy.add(f.stem)
                except OSError:
                    pass
        return {"live": live.stem if live is not None else None, "busy": sorted(busy)}

    def meetings_trash(self, ids):
        """選んだ会議をまとめてゴミ箱へ(今の会話・取り込み中は飛ばす)。{moved, skipped}。"""
        if not isinstance(ids, (list, tuple)):
            return {"moved": [], "skipped": []}
        r = lib_mod.trash_many(self._lib_dir(), [str(x) for x in ids], protect=self.library_busy()["busy"])
        if r["moved"]:
            print(f"[lib] ゴミ箱へ {len(r['moved'])}件(飛ばした {len(r['skipped'])}件)", flush=True)
            self.search_warm()          # 索引から外す
        return r

    def meetings_restore(self, ids):
        if not isinstance(ids, (list, tuple)):
            return {"restored": [], "skipped": []}
        r = lib_mod.restore_many(self._lib_dir(), [str(x) for x in ids])
        if r["restored"]:
            self.search_warm()
        return r

    def trash_delete(self, ids, expect):
        """ゴミ箱の中の選んだ会議を完全に削除(元に戻せない)。数が違う・ゴミ箱に無いものがあれば -1。"""
        try:
            n = lib_mod.delete_forever(self._lib_dir(), [str(x) for x in (ids or [])], int(expect))
        except (TypeError, ValueError):
            return -1
        if n > 0:
            print(f"[lib] 完全に削除 {n}件", flush=True)
        return n

    def trash_empty(self, expect):
        """ゴミ箱を空にする(元に戻せない)。画面で見た件数と違えば -1。"""
        try:
            return lib_mod.trash_empty(self._lib_dir(), int(expect))
        except (TypeError, ValueError):
            return -1

    # ---------- 全文検索 ----------

    def search_all(self, q):
        try:
            r = self._search_index().search(str(q or ""))
        except Exception as e:  # noqa: BLE001
            print(f"[search] 失敗: {e}", flush=True)
            return {"q": q, "total": 0, "meetings": [], "error": str(e)[:200]}
        lst = {m["id"]: m for m in lib_mod.list_meetings(self._lib_dir())}
        for m in r["meetings"]:
            e = lst.get(m["id"], {})
            m.update({k: e.get(k) for k in ("title", "preview", "minutes", "star", "folder", "tags", "has_notes")})
            m["names"] = notes_mod.load_speakers(self._lib_dir() / f"{m['id']}.jsonl")
        return r

    def search_warm(self):
        """索引を裏で作っておく(会議の記録を開いた時)。"""
        threading.Thread(target=lambda: self._search_index().refresh(), daemon=True).start()
        return "ok"

    # ---------- 話者の分析 ----------

    def meeting_stats(self, mid):
        p = self._lib_path(mid)
        if p is None:
            return None
        if mid == "live" and (lg := getattr(self, "log", None)) is not None:   # 閉じる瞬間に None になっても落ちない
            lg.flush()
        return analytics_mod.speaker_stats(notes_mod.load_meeting(p))

    # ---------- 書き出し ----------

    def _export_doc(self, p: Path, lang, with_transcript: bool) -> dict:
        lg = lang if lang in notes_mod.LANGS else None
        m = notes_mod.load_meeting(p)
        n = notes_mod.load_notes(p, lg)
        return export_mod.build_doc(n, m, analytics_mod.speaker_stats(m), lib_mod.load_info(p),
                                    lang=(n or {}).get("lang") or lg, with_transcript=bool(with_transcript),
                                    names=notes_mod.load_speakers(p))

    def _save_path(self, name: str, fmt: str) -> "str | None":
        """保存する場所を利用者に選んでもらう(ファイルの保存パネル)。取り消せば None。"""
        import webview
        w = self.window
        if w is None:
            return None
        start = str(Path.home() / "Downloads")
        kind = getattr(getattr(webview, "FileDialog", None), "SAVE", None) or webview.SAVE_DIALOG
        r = w.create_file_dialog(kind, directory=start, save_filename=name,
                                 file_types=(f"{fmt.upper()} (*.{fmt})",))
        if not r:
            return None
        return str(r[0] if isinstance(r, (list, tuple)) else r)

    def meeting_export(self, mid, lang=None, fmt="pdf", with_transcript=False):
        """PDF / Word(docx)に書き出す。保存の場所を聞いてから書き、Finder で見せる。書いたパスを返す。"""
        fmt = fmt if fmt in ("pdf", "docx", "md", "txt") else "pdf"
        p = self._lib_path(mid)
        if p is None:
            return ""
        if mid == "live" and (lg := getattr(self, "log", None)) is not None:   # 閉じる瞬間に None になっても落ちない
            lg.flush()
        doc = self._export_doc(p, lang, bool(with_transcript))
        text = None
        if fmt in ("md", "txt"):
            try:
                lg = lang if lang in notes_mod.LANGS else None
                text = export_mod.text_export(notes_mod.load_notes(p, lg), notes_mod.load_meeting(p),
                                              notes_mod.load_speakers(p), fmt, bool(with_transcript))
            except Exception as e:  # noqa: BLE001
                print(f"[export] 失敗 {type(e).__name__}: {e}", flush=True)
                return ""
        out = self._save_path(export_mod.file_name(doc, fmt), fmt)
        if not out:
            return ""
        if not out.lower().endswith("." + fmt):
            out += "." + fmt
        try:
            if text is not None:
                Path(out).write_text(text, encoding="utf-8")
            elif fmt == "pdf":
                export_mod.pdf_subprocess(doc, out)
            else:
                Path(out).write_bytes(export_mod.to_docx(doc))
        except Exception as e:  # noqa: BLE001
            print(f"[export] 失敗 {type(e).__name__}: {e}", flush=True)
            return ""
        try:
            import subprocess
            subprocess.run(["open", "-R", out], check=False, timeout=5)
        except Exception:  # noqa: BLE001
            pass
        return out

    # ---------- 声を覚える ----------

    def _voice_cents(self, p: Path) -> dict:
        """その会議の話者の重心。今の会話なら見分けの途中の状態から作る。"""
        tr = getattr(self, "spk", None)
        live = self._lib_live()
        if tr is not None and live is not None and live == p:
            try:
                cents = tr.centroids(tr.refine())
                vb_mod.save_meeting_voices(p, cents)
                return cents
            except Exception:  # noqa: BLE001
                pass
        return vb_mod.load_meeting_voices(p)

    def _voice_named(self, p: Path, spk: str, name: str) -> None:
        """利用者が名前を付けた/直した/消した → 声の帳面に反映(Bridge.speaker_rename から呼ぶ)。"""
        try:
            book = self._voicebook()
            src = f"{p.stem}:{spk}"
            m = vb_mod.load_matches(p)
            cur = m.get(spk)
            if not name:
                book.unenroll(src)
                if cur and cur.get("state") == "auto":
                    cur["state"] = "rejected"
                    cur.setdefault("skip", []).append(cur.get("person"))
                vb_mod.save_matches(p, m)
                return
            if cur and cur.get("state") == "auto" and cur.get("name") != name:
                cur["state"] = "rejected"            # 声からの名前を利用者が別の名前に直した
                cur.setdefault("skip", []).append(cur.get("person"))
                vb_mod.save_matches(p, m)
            elif cur and cur.get("state") == "auto" and cur.get("name") == name:
                cur["state"] = "accepted"
                vb_mod.save_matches(p, m)
            c = self._voice_cents(p).get(spk)
            if c is not None:
                book.enroll(name, c[0], c[1], c[2], src)
        except Exception as e:  # noqa: BLE001
            print(f"[voice] 覚えられません: {e}", flush=True)

    def _voice_apply(self, p: Path, cents: dict) -> dict:
        """覚えた声と照合して、名前の無い話者に名前を付ける(声からの印つき)。付けた/今の照合を返す。"""
        book = self._voicebook()
        m = vb_mod.load_matches(p)
        if not cents or not book.persons():
            return m
        names = notes_mod.load_speakers(p)
        user_named = {k: v for k, v in names.items()
                      if v and not (m.get(k, {}).get("state") == "auto" and m[k].get("name") == v)}
        taken = {v.casefold() for v in user_named.values()}
        skip = {k: [x for x in v.get("skip", []) if x] for k, v in m.items()}
        free = {k: c for k, c in cents.items() if k not in user_named and m.get(k, {}).get("state") != "rejected"}
        got = book.match(free, exclude_mid=p.stem, skip=skip)
        changed = False
        for spk, r in got.items():
            if r["name"].casefold() in taken:
                continue                      # その名前はこの会議で利用者が別の話者に付けている
            old = m.get(spk)
            if old and old.get("state") in ("auto", "accepted") and old.get("person") == r["person"]:
                old["sim"] = r["sim"]
                continue
            m[spk] = {"name": r["name"], "person": r["person"], "sim": r["sim"], "state": "auto",
                      "at": time.strftime("%Y-%m-%d %H:%M:%S")}
            notes_mod.rename_speaker(p, spk, r["name"])
            changed = True
        vb_mod.save_matches(p, m)
        if changed:
            print(f"[voice] {p.stem}: 声から {[(k, v['name'], v['sim']) for k, v in got.items()]}", flush=True)
            live = self._lib_live()
            if live is not None and live == p:
                self.ui.put(("speakers", notes_mod.load_speakers(p)))
        return m

    def _voice_after_refine(self, ids) -> None:
        """会話中・会議の後の付け直しのたびに(__main__ から): 重心を残し、覚えた声と照合する。"""
        try:
            tr, p = getattr(self, "spk", None), self._lib_live()
            if tr is None or p is None or not ids or not hasattr(tr, "centroids"):
                return
            cents = tr.centroids(ids)
            vb_mod.save_meeting_voices(p, cents)
            self._voice_apply(p, cents)
        except Exception as e:  # noqa: BLE001
            print(f"[voice] 照合に失敗: {e}", flush=True)

    def voices_meeting(self, mid):
        """会議を開いた時: 声からの名前(照合をし直す)と、覚えられる話者(声の長さ)。"""
        p = self._lib_path(mid)
        if p is None:
            return None
        cents = self._voice_cents(p) if mid == "live" else vb_mod.load_meeting_voices(p)
        m = self._voice_apply(p, cents) if cents else vb_mod.load_matches(p)
        return {"matches": m, "people": len(self._voicebook().persons()),
                "voices": {k: {"secs": round(v[1], 1), "n": v[2], "ok": v[1] >= vb_mod.MIN_ENROLL_S}
                           for k, v in cents.items()},
                "min_enroll_s": vb_mod.MIN_ENROLL_S}

    def voice_reject(self, mid, spk):
        """声からの名前を取り消す(この会議では同じ人をもう付けない)。"""
        p = self._lib_path(mid)
        m = vb_mod.load_matches(p) if p is not None else {}
        cur = m.get(str(spk))
        if not cur:
            return "none"
        if notes_mod.load_speakers(p).get(spk) == cur.get("name"):
            notes_mod.rename_speaker(p, spk, "")
        cur["state"] = "rejected"
        cur.setdefault("skip", []).append(cur.get("person"))
        vb_mod.save_matches(p, m)
        if mid == "live" or self._lib_live() == p:
            self.ui.put(("speakers", {spk: ""}))
        return "ok"

    def voice_accept(self, mid, spk):
        """声からの名前が合っている → 確定し、この会議の声も見本に足す。"""
        p = self._lib_path(mid)
        m = vb_mod.load_matches(p) if p is not None else {}
        cur = m.get(str(spk))
        if not cur:
            return "none"
        self._voice_named(p, spk, cur["name"])
        return "ok"

    def voicebook_list(self):
        return self._voicebook().persons()

    def voicebook_forget(self, pid):
        return "ok" if self._voicebook().forget(str(pid)) else "none"

    def voicebook_state(self):
        """覚えた人数と、初めて覚えた時の知らせを見せたか。"""
        b = self._voicebook()
        return {"people": len(b.persons()), "noticed": b.noticed()}

    def voicebook_notice_seen(self):
        self._voicebook().set_noticed()
        return "ok"

    def voicebook_forget_all(self, expect):
        """覚えた声と、会議ごとの声の特徴(session-….voices.npz、ゴミ箱の中も)を全部消す。元に戻せない。
        画面で見た人数 expect と今の人数が違えば -1(消さない)。付けた名前は残す。"""
        b = self._voicebook()
        try:
            if int(expect) != len(b.persons()):
                return -1
        except (TypeError, ValueError):
            return -1
        n = b.forget_all()
        d = self._lib_dir()
        for f in list(d.glob("session-*.voices.npz")) + list((d / lib_mod.TRASH).glob("session-*.voices.npz")):
            try:
                f.unlink()
            except OSError:
                pass
        print(f"[voice] 覚えた声を全部消しました({n}人)", flush=True)
        return n

    def voicebook_rename(self, pid, name):
        return "ok" if self._voicebook().rename(str(pid), str(name or "")) else "none"

    # ---------- カレンダー・リマインダー ----------

    def calendar_now(self):
        """今の(15分以内に始まる)予定の題と出席者。初回は macOS の許可の確認が出る。"""
        return cal_mod.events_now()

    def meeting_calendar_set(self, mid, event):
        """会話に予定を結びつける(題が無ければ予定の題を使う・出席者を名前の候補に)。"""
        p = self._lib_path(mid)
        if p is None or not isinstance(event, dict):
            return None
        info = lib_mod.load_info(p)
        kw = {"calendar": event}
        if not info.get("title") and event.get("title"):
            kw["title"] = event["title"]
        return lib_mod.update_info(p, **kw)

    def reminders_add(self, mid, lang=None, indexes=None):
        """やること → リマインダー(「Sokutsu」リスト)。indexes を省けば済んでいないもの全部。"""
        p = self._lib_path(mid)
        n = notes_mod.load_notes(p, lang if lang in notes_mod.LANGS else None) if p is not None else None
        if not n:
            return {"added": 0, "error": "nonotes"}
        todos = n.get("todos") or []
        pick = [todos[i] for i in (indexes or []) if isinstance(i, int) and 0 <= i < len(todos)] \
            if indexes else [t for t in todos if not t.get("done")]
        names = {q["id"]: q.get("name") or notes_mod.default_name(q["id"], n.get("lang", "ja"))
                 for q in n.get("participants", [])}
        names.update({k: v for k, v in notes_mod.load_speakers(p).items() if v})
        title = lib_mod.load_info(p).get("title") or n.get("title", "")
        return cal_mod.add_reminders(pick, meeting_title=title, names=names)
