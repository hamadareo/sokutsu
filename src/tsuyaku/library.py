"""会議の記録の整理(W4 11/07): 題の変更・タグ・フォルダ・お気に入り・ゴミ箱(戻せる)。

設計: 設計メモ(公開版には含めない)(W4)
  - 会議ごとの付属ファイル `session-….tags.json` に {title, tags, folder, star, calendar, trashed}
    を置く(記録の jsonl は書き換えない。消せば元どおり)。題は空にすると自動の題(メモの題)に戻る。
  - フォルダは1段だけ(会議は1つのフォルダに入るか、どこにも入らない)。空のフォルダも残せるよう
    フォルダの並びは logs/.library.json に持つ。
  - ゴミ箱 = logs/.trash/ へ、その会議のファイル全部(`session-X.*`)を動かすだけ。戻せる。
    ゴミ箱を空にするのは利用者がはっきり押した時だけで、画面で見た件数(expect)が今の件数と
    違えば消さない(別の画面・別の操作で増えたものを知らずに消さない)。
  - 一覧は notes.meeting_entry をファイルの更新時刻で覚えておく(数百件でも開くたびに全部を
    読み直さない)。
"""
from __future__ import annotations

import json
import re
import shutil
import time
from pathlib import Path

from . import notes as notes_mod

MAX_TITLE = 120
MAX_TAG = 40
MAX_TAGS = 20
MAX_FOLDER = 60
TRASH = ".trash"
LIBRARY = ".library.json"
_ID = re.compile(r"^session-[0-9A-Za-z_-]+$")
_FIELDS = ("title", "tags", "folder", "star", "calendar", "trashed")
_lock = notes_mod.LOCK      # メモ・名前を書く側と同じ鍵(ゴミ箱へ移す間に付属ファイルを書かせない)
_cache: dict = {}          # jsonl のパス → (ファイルの署名, 一覧の1件)


def valid_id(mid) -> bool:
    return isinstance(mid, str) and bool(_ID.match(mid))


def info_file(jsonl: "str | Path") -> Path:
    return Path(f"{Path(jsonl).with_suffix('')}.tags.json")


def _clean_text(v, n: int) -> str:
    return re.sub(r"\s+", " ", str(v or "")).strip()[:n]


def _clean_tags(v) -> list:
    out: list = []
    for t in v or []:
        t = _clean_text(t, MAX_TAG).lstrip("#")
        if t and t not in out:
            out.append(t)
    return out[:MAX_TAGS]


def _clean_calendar(v) -> "dict | None":
    if not isinstance(v, dict):
        return None
    out = {k: _clean_text(v.get(k), 200) for k in ("title", "start", "end", "calendar", "location")
           if v.get(k)}
    att = []
    for a in v.get("attendees") or []:
        if isinstance(a, dict):
            a = {k: _clean_text(a.get(k), 120) for k in ("name", "email") if a.get(k)}
            if a:
                att.append(a)
        elif a:
            att.append({"name": _clean_text(a, 120)})
    if att:
        out["attendees"] = att[:50]
    return out or None


def load_info(jsonl: "str | Path") -> dict:
    f = info_file(jsonl)
    try:
        d = json.loads(f.read_text(encoding="utf-8")) if f.exists() else {}
    except (ValueError, OSError):
        return {}
    return {k: d[k] for k in _FIELDS if k in d} if isinstance(d, dict) else {}


def _save_info(jsonl: "str | Path", info: dict) -> None:
    f = info_file(jsonl)
    if info:
        tmp = f.with_name(f.name + ".tmp")
        tmp.write_text(json.dumps(info, ensure_ascii=False, indent=1), encoding="utf-8")
        tmp.replace(f)
    elif f.exists():
        f.unlink()


def update_info(jsonl: "str | Path", **kw) -> "dict | None":
    """付属情報を部分的に書き換える。空の値はその項目を消す。知らない項目は無視。
    記録がもう無い(画面が開いた後にゴミ箱へ移された)なら書かずに None。書くと .tags.json だけが
    ログのフォルダに残り、ゴミ箱から戻す時に同じ名前があるので戻せなくなる。"""
    with _lock:
        if not Path(jsonl).exists():
            return None
        info = load_info(jsonl)
        for k, v in kw.items():
            if k == "title":
                v = _clean_text(v, MAX_TITLE)
            elif k == "folder":
                v = _clean_text(v, MAX_FOLDER)
            elif k == "tags":
                v = _clean_tags(v)
            elif k == "star":
                v = bool(v)
            elif k == "calendar":
                v = _clean_calendar(v)
            elif k == "trashed":
                v = float(v) if v else None
            else:
                continue
            if v:
                info[k] = v
            else:
                info.pop(k, None)
        _save_info(jsonl, info)
        return info


# ---------------- 一覧 ----------------

def _sig(p: Path) -> tuple:
    out = []
    for f in (p, Path(f"{p.with_suffix('')}.notes.json"), notes_mod.speakers_file(p)):
        try:
            st = f.stat()
            out.append((st.st_mtime_ns, st.st_size))
        except OSError:
            out.append(None)
    return tuple(out)


def _entry(p: Path) -> "dict | None":
    sig = _sig(p)
    with _lock:
        hit = _cache.get(str(p))
    if hit and hit[0] == sig:
        return hit[1]
    e = notes_mod.meeting_entry(p)
    with _lock:
        _cache[str(p)] = (sig, e)
    return e


def _decorate(e: dict, info: dict) -> dict:
    m = dict(e)
    m["auto_title"] = e.get("title", "")
    if info.get("title"):
        m["title"] = info["title"]
    m["tags"] = list(info.get("tags") or [])
    m["folder"] = info.get("folder", "")
    m["star"] = bool(info.get("star"))
    if info.get("calendar"):
        m["calendar"] = info["calendar"]
    return m


def list_meetings(log_dir: "str | Path") -> list[dict]:
    """notes.list_meetings と同じ並び(新しい順)に、題の変更・タグ・フォルダ・お気に入りを足す。"""
    out = []
    for p in sorted(Path(log_dir).glob("session-*.jsonl"), reverse=True):
        e = _entry(p)
        if e is not None:
            out.append(_decorate(e, load_info(p)))
    return out


def _lib(log_dir) -> dict:
    f = Path(log_dir) / LIBRARY
    try:
        d = json.loads(f.read_text(encoding="utf-8")) if f.exists() else {}
    except (ValueError, OSError):
        d = {}
    return d if isinstance(d, dict) else {}


def _lib_save(log_dir, d: dict) -> None:
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    (Path(log_dir) / LIBRARY).write_text(json.dumps(d, ensure_ascii=False, indent=1), encoding="utf-8")


def facets(log_dir: "str | Path") -> dict:
    """絞り込みの候補: タグ(多い順)・フォルダ(名前順、空のフォルダも)・お気に入り・全部・ゴミ箱の数。"""
    lst = list_meetings(log_dir)
    tags: dict = {}
    folders: dict = {f: 0 for f in _lib(log_dir).get("folders", []) if f}
    for m in lst:
        for t in m["tags"]:
            tags[t] = tags.get(t, 0) + 1
        if m["folder"]:
            folders[m["folder"]] = folders.get(m["folder"], 0) + 1
    return {"all": len(lst), "star": sum(1 for m in lst if m["star"]),
            "tags": [{"name": k, "n": v} for k, v in sorted(tags.items(), key=lambda kv: (-kv[1], kv[0]))],
            "folders": [{"name": k, "n": folders[k]} for k in sorted(folders)],
            "trash": len(_trashed(log_dir))}


def _each_meeting(log_dir):
    yield from Path(log_dir).glob("session-*.jsonl")


def folder_add(log_dir, name: str) -> str:
    name = _clean_text(name, MAX_FOLDER)
    if not name:
        return ""
    with _lock:
        d = _lib(log_dir)
        fs = [f for f in d.get("folders", []) if f]
        if name not in fs:
            fs.append(name)
        d["folders"] = fs
        _lib_save(log_dir, d)
    return name


def folder_rename(log_dir, old: str, new: str) -> str:
    new = _clean_text(new, MAX_FOLDER)
    if not new or not old:
        return ""
    with _lock:
        for p in _each_meeting(log_dir):
            if load_info(p).get("folder") == old:
                update_info(p, folder=new)
        d = _lib(log_dir)
        fs = [new if f == old else f for f in d.get("folders", [])]
        d["folders"] = list(dict.fromkeys(f for f in fs if f))
        _lib_save(log_dir, d)
    return new


def folder_remove(log_dir, name: str) -> None:
    """フォルダを消す。中の会議は消さず「フォルダなし」に戻す。"""
    with _lock:
        for p in _each_meeting(log_dir):
            if load_info(p).get("folder") == name:
                update_info(p, folder="")
        d = _lib(log_dir)
        d["folders"] = [f for f in d.get("folders", []) if f and f != name]
        _lib_save(log_dir, d)


def tag_rename(log_dir, old: str, new: str) -> None:
    """タグの名前を変える(new が空ならそのタグを外す)。同じ名前になれば1つにまとめる。"""
    with _lock:
        for p in _each_meeting(log_dir):
            tags = load_info(p).get("tags") or []
            if old in tags:
                update_info(p, tags=[(new if t == old else t) for t in tags if (t != old or new)])


# ---------------- ゴミ箱 ----------------

def _files_of(d: Path, mid: str) -> list[Path]:
    # session-X.jsonl / .txt / .notes.ja.json / .speakers.json / .tags.json / .audio.m4a …
    # glob の "X.*" は「X.」で始まる名前だけ(X1.jsonl のような別の記録は入らない)
    return [p for p in d.glob(f"{mid}.*") if p.is_file()]


def _trashed(log_dir) -> list[Path]:
    return sorted((Path(log_dir) / TRASH).glob("session-*.jsonl"))


def trash(log_dir, mid, live: "str | Path | None" = None) -> bool:
    """会議をゴミ箱へ(ファイルを logs/.trash/ へ動かす)。今の会話・おかしな id は断る。"""
    if not valid_id(mid):
        return False
    d = Path(log_dir)
    src = d / f"{mid}.jsonl"
    if not src.exists():
        return False
    if live is not None and Path(live).resolve() == src.resolve():
        return False
    with _lock:
        update_info(src, trashed=time.time())
        t = d / TRASH
        t.mkdir(exist_ok=True)
        for f in _files_of(d, mid):
            shutil.move(str(f), str(t / f.name))
        _cache.pop(str(src), None)
    return True


def trash_list(log_dir) -> list[dict]:
    """ゴミ箱の中身(捨てた新しい順)。"""
    out = []
    for p in _trashed(log_dir):
        e = notes_mod.meeting_entry(p)
        if e is None:
            e = {"id": p.stem, "start": "", "title": "", "lines": 0, "minutes": 0}
        info = load_info(p)
        m = _decorate(e, info)
        m["trashed"] = info.get("trashed") or p.stat().st_mtime
        out.append(m)
    return sorted(out, key=lambda m: -m["trashed"])


def restore(log_dir, mid) -> bool:
    """ゴミ箱から戻す。同じ名前のファイルが今あれば上書きせずに断る。"""
    if not valid_id(mid):
        return False
    d = Path(log_dir)
    t = d / TRASH
    files = _files_of(t, mid)
    if not any(f.suffix == ".jsonl" for f in files):
        return False
    if any((d / f.name).exists() for f in files):
        return False
    with _lock:
        for f in files:
            shutil.move(str(f), str(d / f.name))
        update_info(d / f"{mid}.jsonl", trashed=None)
    return True


def trash_empty(log_dir, expect: int) -> int:
    """ゴミ箱を空にする(元に戻せない)。画面で見た件数 expect と今の件数が違えば -1(消さない)。"""
    with _lock:
        items = _trashed(log_dir)
        if int(expect) != len(items):
            return -1
        t = Path(log_dir) / TRASH
        for f in list(t.iterdir()) if t.exists() else []:
            if f.is_file() and f.name.startswith("session-"):
                f.unlink()
        return len(items)


# ---------------- まとめて(選んだ会議を)ゴミ箱へ・戻す・完全に削除 ----------------

def trash_many(log_dir, ids, protect=()) -> dict:
    """選んだ会議をまとめてゴミ箱へ。protect(今の会話・取り込み中)とおかしな id は飛ばす。
    {moved: [...], skipped: [...]}(並びは渡した順)。"""
    moved, skipped = [], []
    prot = set(protect or ())
    for mid in dict.fromkeys(ids or []):
        if mid in prot or not trash(log_dir, mid):
            skipped.append(mid)
        else:
            moved.append(mid)
    return {"moved": moved, "skipped": skipped}


def restore_many(log_dir, ids) -> dict:
    restored, skipped = [], []
    for mid in dict.fromkeys(ids or []):
        (restored if restore(log_dir, mid) else skipped).append(mid)
    return {"restored": restored, "skipped": skipped}


def delete_forever(log_dir, ids, expect: int) -> int:
    """ゴミ箱の中の選んだ会議を完全に消す(元に戻せない)。その会議のファイル(記録・録音・メモ・名前・
    付属情報・声の特徴…= session-X.* 全部)を消す。ゴミ箱に無い id が混ざる・数が画面で見た expect と
    違うときは何も消さずに -1。消した会議の数を返す。"""
    if not valid_id_list(ids):
        return -1
    want = list(dict.fromkeys(ids))
    t = Path(log_dir) / TRASH
    with _lock:
        have = {p.stem for p in _trashed(log_dir)}
        if int(expect) != len(want) or any(m not in have for m in want):
            return -1
        for mid in want:
            for f in _files_of(t, mid):
                f.unlink()
    return len(want)


def valid_id_list(ids) -> bool:
    return isinstance(ids, (list, tuple)) and bool(ids) and all(valid_id(x) for x in ids)
