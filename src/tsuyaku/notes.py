"""会議メモ(11/06「Notion のミーティングメモのような機能」)。

会話の記録(logs/session-….jsonl)から、ローカルの LLM(リモートサーバー / このMac)で
タイトル・参加者(名前の理解)・要約・話題ごとの要点(誰が)・決定事項・やること・
残った質問・キーワードを作る。設計: docs/superpowers/specs/2026-11-06-meeting-notes-design.md

作り方:
  - 話者は声の見分け(S1, S2…)。無い記録(通訳だけ・古い記録)は話した言語を話者の
    代わりにする(Lja, Len…: 対面の通訳ではほぼ1言語=1人)。
  - 長い会議は記録を約5000字ごとに区切り、区切りごとに要点を JSON で抜き出してから、
    最後にまとめて1つのメモにする(LLM が一度に読める長さを超えない、抜けを減らす)。
  - 指示文は英語で書き、出力の言語をはっきり指定する(11/06 FLEURS: 指示文の言語に
    出力が引っ張られた)。
  - 名前は会話の中で実際に言われたものだけ(自己紹介・呼びかけ)。根拠の発言も残す。
    画面で直した名前(.speakers.json)は LLM への手がかりにし、上書きしない。
"""
from __future__ import annotations

import json
import re
import shutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable

import requests

LANGS = ("ja", "en", "zh", "ko")
EN = {"ja": "Japanese", "en": "English", "zh": "Simplified Chinese", "ko": "Korean"}
LANG_LABEL = {
    "ja": {"ja": "日本語", "en": "英語", "zh": "中国語", "ko": "韓国語"},
    "en": {"ja": "Japanese", "en": "English", "zh": "Chinese", "ko": "Korean"},
    "zh": {"ja": "日语", "en": "英语", "zh": "中文", "ko": "韩语"},
    "ko": {"ja": "일본어", "en": "영어", "zh": "중국어", "ko": "한국어"},
}
SPEAKER_WORD = {"ja": "話者", "en": "Speaker ", "zh": "说话人", "ko": "화자"}
LANG_SPEAKER = {"ja": "{}の話者", "en": "{} speaker", "zh": "{}说话人", "ko": "{} 화자"}
HEAD = {
    "ja": {"participants": "参加者", "summary": "要約", "topics": "話題ごとの要点",
           "decisions": "決定事項", "todos": "やること", "questions": "残った質問",
           "keywords": "キーワード", "transcript": "記録", "owner": "担当", "due": "期限",
           "minutes": "分", "none": "なし"},
    "en": {"participants": "Participants", "summary": "Summary", "topics": "Topics",
           "decisions": "Decisions", "todos": "Action items", "questions": "Open questions",
           "keywords": "Keywords", "transcript": "Transcript", "owner": "Owner", "due": "Due",
           "minutes": "min", "none": "None"},
    "zh": {"participants": "参与者", "summary": "摘要", "topics": "议题要点",
           "decisions": "决定事项", "todos": "待办事项", "questions": "未解决的问题",
           "keywords": "关键词", "transcript": "记录", "owner": "负责人", "due": "期限",
           "minutes": "分钟", "none": "无"},
    "ko": {"participants": "참석자", "summary": "요약", "topics": "주제별 요점",
           "decisions": "결정 사항", "todos": "할 일", "questions": "남은 질문",
           "keywords": "키워드", "transcript": "기록", "owner": "담당", "due": "기한",
           "minutes": "분", "none": "없음"},
}
CHUNK_CHARS = 5000
# 会議の付属ファイル(メモ・名前・.tags.json)の読み直しと書き込み、ゴミ箱への移動をそろえる鍵。
# 画面・声の照合・メモ作りのスレッドが同じ会議に同時に書いても片方が消えないように(library も使う)
LOCK = threading.RLock()
# W1(11/07)の節の見出し: 自分のメモ・ハイライト(ブックマーク)
HEAD_W1 = {
    "ja": {"my_notes": "自分のメモ", "highlights": "ハイライト"},
    "en": {"my_notes": "My notes", "highlights": "Highlights"},
    "zh": {"my_notes": "我的笔记", "highlights": "重点时刻"},
    "ko": {"my_notes": "내 메모", "highlights": "하이라이트"},
}


# ---------------- 記録を読む ----------------

def speaker_id(row: dict) -> str:
    """その行の話者。声の見分けが無ければ話した言語(L+言語)。"""
    return row.get("spk") or f"L{row.get('lang') or '?'}"


def default_name(spk: str, ui: str = "ja") -> str:
    ui = ui if ui in LANGS else "ja"
    if spk.startswith("S") and spk[1:].isdigit():
        w = SPEAKER_WORD[ui]
        return f"{w}{spk[1:]}"
    if spk.startswith("L"):
        lang = spk[1:]
        return LANG_SPEAKER[ui].format(LANG_LABEL[ui].get(lang, lang))
    return spk


def load_meeting(path: "str | Path") -> dict:
    """jsonl → {id, path, meta, lines, marks, memos, agenda, template, terms, undo}。
    壊れた行は飛ばす(書きかけの最終行など)。

    会議中の道具(W3, 11/07)の行も読む(書く側は inmeeting.py):
      mark / memo  … 同じ id の後の行は上書き(種類・ひとことの変更)、del 行で消す。
                     i(指す行)が無ければ、その行より前に書かれた最後の発言。
      agenda       … meta の agenda(議題の文)の k 番目を済み/未にする。
      edit / respk … 発言 i の文・話者を直す(声の付け直し alias/refine より後に効く)。
      undo         … いちばん新しい edit/respk を取り消す(何回でも)。
    """
    p = Path(path)
    meta: dict = {}
    lines: list[dict] = []
    alias: dict = {}
    refined: list = []
    notes_rows: dict = {"mark": {}, "memo": {}}   # id → 行(書いた順を保つ)
    agenda_rows: list[dict] = []
    ops: list[dict] = []                          # edit / respk(undo で後ろから消す)
    # 書く途中で落ちると最後の行が字の途中で切れることがある(壊れた字は置き換え、その行は下で飛ばす)
    for raw in p.read_bytes().decode("utf-8", "replace").splitlines():
        try:
            row = json.loads(raw)
        except ValueError:
            continue
        if not isinstance(row, dict):
            continue
        typ = row.get("type")
        if typ == "alias":      # 途中で「同じ人」と合わせた話者
            alias.update(row.get("map") or {})
        elif typ == "refine":   # 会議の後の付け直し(最後のものを使う)
            refined = row.get("ids") or []
        elif typ == "meta":
            meta = row
        elif typ == "utter" and (row.get("orig") or "").strip():
            lines.append({"t": row.get("t", ""), "lang": row.get("lang", ""),
                          "orig": row.get("orig", ""), "trans": row.get("trans", ""),
                          "spk": speaker_id(row), "ts": row.get("ts"),
                          "segs": row.get("segs") or []})
            lines[-1].update({k: row[k] for k in ("t0", "dur") if row.get(k) is not None})
            if row.get("a0") is not None:   # 録音の中の秒(W2: 行を押すとそこから鳴らす)
                lines[-1]["a0"], lines[-1]["a1"] = row["a0"], row.get("a1", row["a0"])
        elif typ in ("mark", "memo") and isinstance(row.get("id"), str) and row["id"]:
            store = notes_rows[typ]
            if row["id"] in store:
                store[row["id"]].update({k: v for k, v in row.items() if k not in ("type", "id")})
            else:
                r = dict(row)
                r.setdefault("i", len(lines) - 1)
                store[row["id"]] = r
        elif typ == "del":
            for store in notes_rows.values():
                store.pop(row.get("id"), None)
        elif typ == "agenda":
            agenda_rows.append(row)
        elif typ in ("edit", "respk"):
            ops.append(row)
        elif typ == "undo":
            # 元に戻すのは人の修正だけ(聞き直し by:"relisten" は飛ばす。11/08)
            for k in range(len(ops) - 1, -1, -1):
                if ops[k].get("by") != "relisten":
                    del ops[k]
                    break
    for ln in lines:
        votes: dict = {}
        for i in ln.pop("segs", []):
            if isinstance(i, int) and 0 <= i < len(refined) and refined[i]:
                votes[refined[i]] = votes.get(refined[i], 0) + 1
        if votes:
            ln["spk"] = max(votes.items(), key=lambda kv: kv[1])[0]
            continue
        seen = set()
        while ln["spk"] in alias and ln["spk"] not in seen:
            seen.add(ln["spk"])
            ln["spk"] = alias[ln["spk"]]
    n_ops = _apply_ops(lines, ops)
    return {"id": p.stem, "path": str(p), "meta": meta, "lines": lines,
            "marks": _marks(notes_rows["mark"], len(lines)),
            "memos": _memos(notes_rows["memo"], len(lines)),
            "agenda": _agenda(meta.get("agenda"), agenda_rows),
            "template": str(meta.get("template") or ""),
            "terms": [t for t in (meta.get("terms") or []) if isinstance(t, str) and t.strip()],
            "undo": n_ops}


# ---------------- 会議中の道具の行(W3, 11/07) ----------------

MARK_KINDS = ("star", "decision", "todo", "question")   # 重要 / 決定 / やること / 質問
SPK_RE = re.compile(r"S\d{1,3}|L[a-z]{2}")
_BULLET = re.compile(r"^\s*(?:[-*・•●○◦▪■□–—]|\d{1,2}[.)．、](?!\d)|[①-⑳]|\(\d{1,2}\))?\s*")


def agenda_items(text) -> list[str]:
    """議題の文 → 項目(1行1項目。先頭の「1.」「-」「・」「①」などは取る)。"""
    out = []
    for ln in str(text or "").splitlines():
        it = _BULLET.sub("", ln, count=1).strip()
        if it:
            out.append(it[:200])
    return out[:30]


def _ix(v, n: int) -> int:
    """行の添字を 0..n-1 に収める(数でなければ最後の行)。"""
    if isinstance(v, bool) or not isinstance(v, int):
        v = n - 1
    return max(0, min(v, n - 1)) if n else 0


def _marks(store: dict, n: int) -> list[dict]:
    out = []
    for mid, r in store.items():
        kind = r.get("kind") if r.get("kind") in MARK_KINDS else "star"
        out.append({"id": mid, "ts": r.get("ts"), "t": str(r.get("t") or ""), "kind": kind,
                    "label": str(r.get("label") or ""), "i": _ix(r.get("i"), n)})
    return sorted(out, key=lambda m: (m["ts"] if isinstance(m["ts"], (int, float)) else 0))


def _memos(store: dict, n: int) -> list[dict]:
    out = []
    for mid, r in store.items():
        text = str(r.get("text") or "").strip()
        if text:
            out.append({"id": mid, "ts": r.get("ts"), "t": str(r.get("t") or ""), "text": text,
                        "i": _ix(r.get("i"), n)})
    return sorted(out, key=lambda m: (m["ts"] if isinstance(m["ts"], (int, float)) else 0))


def _agenda(text, rows: list[dict]) -> list[dict]:
    items = [{"text": t, "done": False, "t": ""} for t in agenda_items(text)]
    for r in rows:
        k = r.get("k")
        if isinstance(k, int) and not isinstance(k, bool) and 0 <= k < len(items):
            items[k]["done"] = bool(r.get("done"))
            items[k]["t"] = str(r.get("t") or "") if items[k]["done"] else ""
    return items


def _apply_ops(lines: list[dict], ops: list[dict]) -> int:
    """edit / respk を順に当てる。当たった数(元に戻せる数)を返す。"""
    n = 0
    for op in ops:
        i = op.get("i")
        if isinstance(i, bool) or not isinstance(i, int) or not 0 <= i < len(lines):
            continue
        ln = lines[i]
        human = op.get("by") != "relisten"   # 元に戻せる数には人の修正だけ数える
        if op.get("type") == "edit":
            text = op.get("orig")
            if not isinstance(text, str) or not text.strip():
                continue
            ln.setdefault("orig0", ln["orig"])
            ln["orig"] = text.strip()
            if ln["orig"] == ln["orig0"]:
                ln.pop("orig0")
            tr = op.get("trans")   # 聞き直し(relisten.py 11/08)は訳も差し替える
            if isinstance(tr, str) and tr.strip():
                ln["trans"] = tr.strip()
        else:
            spk = op.get("spk")
            if not isinstance(spk, str) or not SPK_RE.fullmatch(spk):
                continue
            ln.setdefault("spk0", ln["spk"])
            ln["spk"] = spk
            if ln["spk"] == ln["spk0"]:
                ln.pop("spk0")
        n += human
    return n


def _minutes(meta: dict, lines: list[dict]) -> int:
    try:
        t0 = time.mktime(time.strptime(meta.get("start", ""), "%Y-%m-%d %H:%M:%S"))
    except ValueError:
        return 0
    if not lines:
        return 0
    last = lines[-1]
    if last.get("ts"):
        return max(1, round((float(last["ts"]) - t0) / 60))
    try:
        h, m, s = (int(x) for x in last["t"].split(":"))
        day = time.localtime(t0)
        end = time.mktime((day.tm_year, day.tm_mon, day.tm_mday, h, m, s, 0, 0, -1))
        if end < t0:
            end += 86400
        return max(1, round((end - t0) / 60))
    except Exception:  # noqa: BLE001
        return 0


def notes_file(jsonl: "str | Path", lang: str | None = None) -> Path:
    """メモのファイル。既定(最初に作った言語)は .notes.json、ほかの言語は .notes.<lang>.json。"""
    p = Path(jsonl)
    base = p.with_suffix("")
    if lang:
        cand = Path(f"{base}.notes.{lang}.json")
        if cand.exists():
            return cand
        main = Path(f"{base}.notes.json")
        if main.exists():
            try:
                if _read_json(main).get("lang") == lang:
                    return main
            except (ValueError, OSError):
                pass
            return cand
    return Path(f"{base}.notes.json")


def notes_langs(jsonl: "str | Path") -> list[str]:
    base = Path(jsonl).with_suffix("")
    out = []
    main = Path(f"{base}.notes.json")
    if main.exists():
        try:
            out.append(_read_json(main).get("lang", ""))
        except (ValueError, OSError):
            pass
    for lg in LANGS:
        if Path(f"{base}.notes.{lg}.json").exists() and lg not in out:
            out.append(lg)
    return [x for x in out if x]


def load_notes(jsonl: "str | Path", lang: str | None = None) -> "dict | None":
    f = notes_file(jsonl, lang)
    if not f.exists():
        return None
    try:
        return _read_json(f)
    except (ValueError, OSError):
        return None


def save_notes(jsonl: "str | Path", notes: dict) -> Path:
    lang = notes.get("lang")
    f = notes_file(jsonl, lang)
    if not f.exists() and not Path(f"{Path(jsonl).with_suffix('')}.notes.json").exists():
        f = Path(f"{Path(jsonl).with_suffix('')}.notes.json")
    _write_json(f, notes)
    return f


def _keep_backup(f: Path) -> None:
    """置き換える前の、読める状態のファイルを .bak に残す(壊れた現行は写さない。失敗しても保存は続ける)。"""
    try:
        if f.exists():
            json.loads(f.read_text(encoding="utf-8"))
            shutil.copyfile(f, f.with_name(f.name + ".bak"))
    except (OSError, ValueError):
        pass


def _read_json(f: Path):
    """読めなければ .bak(1つ前の保存)から。どちらも読めなければ ValueError / FileNotFoundError。"""
    try:
        return json.loads(f.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        bak = f.with_name(f.name + ".bak")
        if bak.exists():
            return json.loads(bak.read_text(encoding="utf-8"))
        raise


def _write_json(f: Path, obj) -> None:
    """一時ファイルに書いてから置き換える(書く途中で落ちる・ディスクが一杯でも前のファイルが残る)。"""
    tmp = f.with_name(f.name + ".tmp")
    try:
        tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=1), encoding="utf-8")
        _keep_backup(f)
        tmp.replace(f)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def speakers_file(jsonl: "str | Path") -> Path:
    return Path(f"{Path(jsonl).with_suffix('')}.speakers.json")


def load_speakers(jsonl: "str | Path") -> dict:
    f = speakers_file(jsonl)
    try:
        return _read_json(f) if f.exists() else {}
    except (ValueError, OSError):
        return {}


def rename_speaker(jsonl: "str | Path", spk: str, name: str) -> None:
    """画面で直した名前。.speakers.json と、作ってあるメモ全部の参加者に書く。
    画面と声の照合(会話中)から同時に来ても片方が消えないよう、読んで書くまでを LOCK の中で。"""
    name = (name or "").strip()
    with LOCK:
        sp = load_speakers(jsonl)
        if name:
            sp[spk] = name
        else:
            sp.pop(spk, None)
        _write_json(speakers_file(jsonl), sp)
        for lg in notes_langs(jsonl):
            n = load_notes(jsonl, lg)
            if not n:
                continue
            for p in n.get("participants", []):
                if p.get("id") == spk:
                    p["name"] = name
                    p["fixed"] = bool(name)
            save_notes(jsonl, n)


# ---------------- 項目を直す・消す(11/08 右クリック) ----------------
# 利用者が直した・消した項目は notes["user_edits"] に {op: "edit"|"del", kind, key(元の文), to?} で残し、
# メモを作り直した時は carry_user_edits で、同じ文の項目に当て直す(やることの済み/未済と同じ考え)。
ITEM_KINDS = ("summary", "decisions", "questions", "todos", "points")
_TEXT_KEY = {"todos": "task"}
_TODO_FIELDS = ("task", "owner", "due")


def _item_text(kind: str, item) -> str:
    if isinstance(item, dict):
        return str(item.get(_TEXT_KEY.get(kind, "text"), "")).strip()
    return str(item or "").strip()


def _item_list(n: dict, kind: str, idx):
    """(入れ物の list, その中の位置, 根拠の list か None)。points は idx = [話題, 要点]。"""
    if kind == "points":
        ti, pi = (list(idx) + [None, None])[:2] if isinstance(idx, (list, tuple)) else (None, None)
        tps = n.get("topics") or []
        if not isinstance(ti, int) or not 0 <= ti < len(tps):
            raise IndexError(idx)
        return tps[ti].setdefault("points", []), pi, None
    lst = n.get(kind)
    if not isinstance(lst, list):
        raise IndexError(kind)
    src = n.get(f"{kind}_src") if kind in ("summary", "decisions", "questions") else None
    if src is not None and len(src) != len(lst):
        src = None
    return lst, idx, src


def item_edit(n: dict, kind: str, idx, fields: dict) -> dict:
    """項目の文(やることは task・owner・due)を直す。直した項目を返す。おかしな頼みは ValueError/IndexError。"""
    if kind not in ITEM_KINDS or not isinstance(fields, dict):
        raise ValueError(kind)
    lst, i, _src = _item_list(n, kind, idx)
    if not isinstance(i, int) or not 0 <= i < len(lst):
        raise IndexError(idx)
    old = lst[i]
    old_text = _item_text(kind, old)
    if kind == "todos":
        to = {}
        for k in _TODO_FIELDS:
            if k in fields:
                v = str(fields[k] or "").strip()[:500]
                if k == "task" and not v:
                    raise ValueError("empty")
                if k == "owner" and v and not (SPK_RE.fullmatch(v) or v == "all"):
                    raise ValueError("owner")
                to[k] = v
        if not to:
            raise ValueError("nothing")
        new = dict(old, **to)
    else:
        v = str(fields.get("text", "") or "").strip()[:1000]
        if not v:
            raise ValueError("empty")
        to = {"text": v}
        new = dict(old, text=v) if isinstance(old, dict) else v
    lst[i] = new
    ue = n.setdefault("user_edits", [])
    rec = next((r for r in ue if r.get("op") == "edit" and r.get("kind") == kind
                and _item_text(kind, r.get("to", {}).get(_TEXT_KEY.get(kind, "text")) or "") == old_text), None)
    if rec is None:
        ue.append({"op": "edit", "kind": kind, "key": old_text, "to": to})
    else:
        rec["to"] = dict(rec.get("to") or {}, **to)
    return new


def item_delete(n: dict, kind: str, idx) -> dict:
    """項目を消す。元に戻すための {item, src, user_edits} を返す。"""
    if kind not in ITEM_KINDS:
        raise ValueError(kind)
    lst, i, src = _item_list(n, kind, idx)
    if not isinstance(i, int) or not 0 <= i < len(lst):
        raise IndexError(idx)
    undo = {"item": lst[i], "src": src[i] if src is not None else None,
            "user_edits": json.loads(json.dumps(n.get("user_edits") or [], ensure_ascii=False))}
    text = _item_text(kind, lst[i])
    del lst[i]
    if src is not None:
        del src[i]
    ue = n.setdefault("user_edits", [])
    rec = next((r for r in ue if r.get("op") == "edit" and r.get("kind") == kind
                and str((r.get("to") or {}).get(_TEXT_KEY.get(kind, "text"), "")).strip() == text), None)
    if rec is not None:            # 直した後に消した: 元の文を消したことにする(記録は1つ)
        rec.pop("to", None)
        rec["op"] = "del"
    else:
        ue.append({"op": "del", "kind": kind, "key": text})
    return undo


def item_restore(n: dict, kind: str, idx, undo: dict) -> None:
    """item_delete の取り消し: 同じ位置に戻し、user_edits も消す前に戻す。"""
    if kind not in ITEM_KINDS or not isinstance(undo, dict) or "item" not in undo:
        raise ValueError(kind)
    item = undo["item"]     # 画面から戻ってくる値なので形を確かめる(文の項目は文字列、やること・要点は dict)
    if kind in ("todos", "points"):
        if not isinstance(item, dict) or not isinstance(item.get(_TEXT_KEY.get(kind, "text")), str):
            raise TypeError(kind)
    elif not isinstance(item, str):
        raise TypeError(kind)
    lst, i, src = _item_list(n, kind, idx)
    if not isinstance(i, int):
        raise IndexError(idx)
    i = max(0, min(i, len(lst)))
    lst.insert(i, item)
    if kind in ("summary", "decisions", "questions"):
        s = n.get(f"{kind}_src")
        if isinstance(s, list) and len(s) == len(lst) - 1:
            s.insert(i, undo.get("src") or [])
    n["user_edits"] = undo.get("user_edits") or []


def _apply_user_edits(new: dict, edits: list) -> None:
    for r in edits or []:
        kind, key = r.get("kind"), str(r.get("key") or "").strip()
        if kind not in ITEM_KINDS or not key:
            continue
        spots = []
        if kind == "points":
            for ti, t in enumerate(new.get("topics") or []):
                for pi, p in enumerate(t.get("points") or []):
                    if _item_text(kind, p) == key:
                        spots.append([ti, pi])
        else:
            spots = [i for i, x in enumerate(new.get(kind) or []) if _item_text(kind, x) == key]
        for at in reversed(spots):
            try:
                if r.get("op") == "del":
                    lst, i, src = _item_list(new, kind, at)
                    del lst[i]
                    if src is not None:
                        del src[i]
                elif r.get("op") == "edit":
                    lst, i, _ = _item_list(new, kind, at)
                    to = r.get("to") or {}
                    lst[i] = dict(lst[i], **to) if isinstance(lst[i], dict) else str(to.get("text") or lst[i])
            except (IndexError, ValueError, TypeError):
                pass


def carry_user_edits(new: dict, jsonl: "str | Path", known: "dict | None" = None) -> dict:
    """作り直したメモ new に、作っている間(LLM を待つ数分)に利用者がしたことを引き継ぐ。
    書く直前に LOCK の中で呼ぶ。後から利用者が操作したほうを残す:
      - やることの済み/未済: 今のメモ(作り始めの後のチェックも入っている)に同じ文のやることがあれば、その状態。
      - 名前: 今の .speakers.json の名前。作り始めの時(known)にあって今は無い名前は、消したものとして空に。"""
    cur = load_notes(jsonl, new.get("lang")) or {}
    done = {str(x.get("task", "")).strip(): bool(x.get("done"))
            for x in cur.get("todos") or [] if isinstance(x, dict)}
    for td in new.get("todos") or []:
        if td.get("task", "").strip() in done:
            td["done"] = done[td["task"].strip()]
    edits = cur.get("user_edits") or []
    if edits:                     # 直した・消した項目(11/08 右クリック)
        _apply_user_edits(new, edits)
        new["user_edits"] = edits
    latest = load_speakers(jsonl)
    for q in new.get("participants") or []:
        i = q.get("id")
        if latest.get(i):
            q.update(name=latest[i], fixed=True, minor=False)
        elif (known or {}).get(i):
            q.update(name="", fixed=False)
    return new


def meeting_entry(p: "str | Path") -> "dict | None":
    """一覧の1件(会議の記録1つ)。1行も無い・読めない記録は None。"""
    p = Path(p)
    try:
        m = load_meeting(p)
    except OSError:
        return None
    if not m["lines"]:
        return None
    n = load_notes(p)
    meta = m["meta"]
    return {"id": m["id"], "start": meta.get("start", ""),
            "mode": meta.get("mode", "interpret"), "pair": meta.get("pair", ""),
            "topic": meta.get("topic", ""),
            "lines": len(m["lines"]), "minutes": _minutes(meta, m["lines"]),
            "title": (n or {}).get("title", ""), "has_notes": n is not None,
            "preview": m["lines"][0]["orig"][:30],
            "keywords": (n or {}).get("keywords", [])[:10],
            "names": sorted({x for x in list(load_speakers(p).values())
                             + [q.get("name", "") for q in (n or {}).get("participants", [])]
                             if x}),
            "speakers": len({ln["spk"] for ln in m["lines"]}),
            "source": meta.get("source", ""), "file": meta.get("file", ""),
            "audio": bool(meta.get("audio"))}


def list_meetings(log_dir: "str | Path") -> list[dict]:
    """会議の記録の一覧(新しい順)。1行も無い記録は出さない。"""
    out = []
    for p in sorted(Path(log_dir).glob("session-*.jsonl"), reverse=True):
        e = meeting_entry(p)
        if e is not None:
            out.append(e)
    return out


# ---------------- LLM ----------------

def chat(base: str, model: str, messages: list[dict], max_tokens: int = 6000,
         timeout: float = 240.0) -> str:
    """OpenAI 互換の chat(ストリーミングなし)。思考を止める指定が通らない
    サーバーなら1回だけ指定なしでやり直す(translate.translate_stream と同じ)。"""
    payload = {"model": model, "messages": messages, "temperature": 0.2,
               "max_tokens": max_tokens, "reasoning_effort": "none"}
    url = base.rstrip("/") + "/v1/chat/completions"
    try:
        r = requests.post(url, json=payload, timeout=timeout)
    except requests.ConnectionError:
        # 11/06 AMI 3周目: 長い会議の途中でサーバーが接続を切った(一時的)。3秒待って1回だけやり直す
        time.sleep(3)
        r = requests.post(url, json=payload, timeout=timeout)
    if r.status_code == 400:
        payload.pop("reasoning_effort", None)
        r = requests.post(url, json=payload, timeout=timeout)
    r.raise_for_status()
    msg = r.json()["choices"][0]["message"]
    return (msg.get("content") or "").strip()


def parse_json(text: str) -> dict:
    """LLM の返事から JSON を1つ取り出す(```json の囲い・前置きの文があっても)。"""
    t = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S).strip()
    m = re.search(r"```(?:json)?\s*(\{.*\})\s*```", t, re.S)
    if m:
        t = m.group(1)
    i, j = t.find("{"), t.rfind("}")
    if i < 0 or j <= i:
        raise ValueError("JSON がありません")
    return json.loads(t[i:j + 1])


# ---------------- 根拠の行番号(11/07 W1: 章立て・項目から元の発言へ) ----------------

_N1 = r"#\d+(?:\s*[-–~〜]\s*#?\d+)?"
_REFS = re.compile(r"\s*[\[(（【]\s*" + _N1 + r"(?:\s*[,、，;]\s*#?\d+(?:\s*[-–~〜]\s*#?\d+)?)*\s*[\])）】]")


def split_refs(text, n: "int | None" = None) -> tuple[str, list[int]]:
    """「進め方を決めた [#0, #3]」→ ("進め方を決めた", [0, 3])。[#2-#4] は範囲。括弧の無い #1 は本文。
    n があれば 0..n-1 の外の番号は捨てる。"""
    text = str(text or "")
    got: set = set()

    def take(m):
        for a, b in re.findall(r"#?(\d+)(?:\s*[-–~〜]\s*#?(\d+))?", m.group(0)):
            lo = int(a)
            hi = int(b) if b else lo
            if hi < lo or hi - lo > 10:
                hi = lo
            got.update(range(lo, hi + 1))
        return " "
    t = _REFS.sub(take, text)
    t = re.sub(r"\s+([。、.,!?！？)）」])", r"\1", re.sub(r"[ \t]{2,}", " ", t)).strip()
    return t, sorted(x for x in got if n is None or 0 <= x < n)


def _take_refs(raw: dict, n: int) -> dict:
    """LLM の出力の各項目の終わりの [#12] を取り除き、根拠の行番号を src に分ける。
    summary/decisions/questions は文字列のまま(今までの形)で、*_src に同じ並びで番号を置く。"""
    out = dict(raw)

    def items(v):
        res = []
        for x in v if isinstance(v, list) else []:
            if isinstance(x, dict):
                t, s = split_refs(x.get("text") or "", n)
                s = sorted(set(s) | {int(i) for i in (x.get("src") or []) if str(i).isdigit() and int(i) < n})
            else:
                t, s = split_refs(x, n)
            if t:
                res.append((t, s))
        return res
    for k in ("summary", "decisions", "questions"):
        got = items(raw.get(k))
        out[k] = [t for t, _ in got]
        out[f"{k}_src"] = [s for _, s in got]
    out["keywords"] = [split_refs(x)[0] for x in raw.get("keywords") or [] if split_refs(x)[0]]
    out["title"] = split_refs(raw.get("title") or "")[0]
    tps = []
    for t in raw.get("topics") or []:
        if not isinstance(t, dict):
            continue
        title, ts = split_refs(t.get("title") or "", n)
        pts = []
        for p in t.get("points") or []:
            if isinstance(p, dict):
                txt, s = split_refs(p.get("text") or "", n)
                pts.append(dict(p, text=txt, src=s))
        tps.append(dict(t, title=title, points=pts, src=ts))
    out["topics"] = tps
    tds = []
    for x in raw.get("todos") or []:
        if isinstance(x, dict):
            task, s = split_refs(x.get("task") or "", n)
            tds.append(dict(x, task=task, src=s))
    out["todos"] = tds
    secs = raw.get("sections")
    if isinstance(secs, list):        # [{"key", "items"}] の形で来ても読む
        secs = {s.get("key"): s.get("items") for s in secs if isinstance(s, dict)}
    out["sections"] = {k: [{"text": t, "src": s} for t, s in items(v)]
                       for k, v in (secs or {}).items() if isinstance(k, str)}
    return out


def _quote_line(quote: str, lines: list[dict]) -> "int | None":
    """やることの原文の引用 → その行(LLM が番号を書かなかった時の受け皿)。"""
    q = _norm_txt(quote)[:14]
    if len(q) < 4:
        return None
    for i, ln in enumerate(lines):
        if q in _norm_txt(ln.get("orig", "")):
            return i
    return None


def _chapters(out: dict, lines: list[dict], ts0) -> None:
    """話題の始まりの行 → 時刻(章)。各話題に i / t / at(録音の先頭からの秒。meta に ts0 がある時)。"""
    def at(i):
        ts = lines[i].get("ts")
        try:
            return round(float(ts) - float(ts0), 1) if ts0 is not None and ts is not None else None
        except (TypeError, ValueError):
            return None
    ch = []
    for t in out["topics"]:
        src = set(t.pop("src", []) or [])
        for p in t["points"]:
            src |= set(p.get("src") or [])
        if src:
            i = min(src)
            t.update({"i": i, "t": lines[i].get("t", ""), "at": at(i)})
            ch.append({"title": t["title"], "i": i, "t": t["t"], "at": t["at"]})
    out["chapters"] = sorted(ch, key=lambda c: c["i"])


# ---------------- 記録 → LLM への文 ----------------

def transcript(lines: list[dict], with_trans: bool = True) -> str:
    """LLM に渡す記録。1行 = #番号 [時刻] 話者id (言語): 原文 [訳: …]。
    番号(load_meeting の lines の添字。行に "i" があるときだけ)は、メモの各項目の根拠の行を
    LLM に書かせるため(11/07 W1: 項目から元の発言へ飛ぶ)。"""
    out = []
    for ln in lines:
        s = f"[{ln.get('t', '')}] {ln['spk']} ({ln.get('lang', '?')}): {ln['orig']}"
        if isinstance(ln.get("i"), int):
            s = f"#{ln['i']} " + s
        tr = ln.get("trans") or ""
        if with_trans and tr and not tr.startswith("（"):
            s += f"  [translation: {tr}]"
        out.append(s)
    return "\n".join(out)


def _cut_at_pause(cur: list[dict], sizes: list[int], max_chars: int) -> int:
    """区切りの位置: 後ろ 4 割のうち、いちばん長い無音(発言の間 8 秒以上)の直前。無ければ全部(= len(cur))。
    話題の途中で切ると、区切りごとの要約が半端になる(長い会議)。"""
    best, gap, acc = len(cur), 8.0, 0
    for k in range(1, len(cur)):
        acc += sizes[k - 1]
        a, b = cur[k - 1].get("ts"), cur[k].get("ts")
        if acc < 0.6 * max_chars or not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
            continue
        if b - a >= gap:
            best, gap = k, b - a
    return best


def chunks(lines: list[dict], max_chars: int = CHUNK_CHARS) -> list[list[dict]]:
    out: list[list[dict]] = []
    cur: list[dict] = []
    sizes: list[int] = []
    n = 0
    for ln in lines:
        k = len(ln["orig"]) + len(ln.get("trans") or "") + 30
        if cur and n + k > max_chars:
            c = _cut_at_pause(cur, sizes, max_chars)
            out.append(cur[:c])
            cur, sizes = cur[c:], sizes[c:]
            n = sum(sizes)
        cur.append(ln)
        sizes.append(k)
        n += k
    if cur:
        out.append(cur)
    return out


def _speaker_hint(lines: list[dict], known: dict) -> str:
    ids = []
    for ln in lines:
        if ln["spk"] not in ids:
            ids.append(ln["spk"])
    parts = []
    for i in ids:
        if i.startswith("L"):
            d = f"{i} = the person speaking {EN.get(i[1:], i[1:])} (no voice separation)"
        else:
            d = f"{i} = a voice cluster (one person)"
        if known.get(i):
            d += f", known name: {known[i]}"
        parts.append(d)
    return ("; ".join(parts) + ". Use exactly these speaker ids (" + ", ".join(ids)
            + ") in participants, who and owner.")


def _fix_ids(raw: dict, ids: list) -> dict:
    """LLM が記録に無い話者 id(例: Lja/Len の記録なのに S1/S2)を書いたら、
    出てきた順で記録の id に読み替える(11/06: 通訳だけの記録で名前が付かなかった)。"""
    seen: list = []

    def note(x):
        if isinstance(x, str) and re.fullmatch(r"S\d+|L\w+", x) and x not in ids and x not in seen:
            seen.append(x)
    for q in raw.get("participants") or []:
        if isinstance(q, dict):
            note(q.get("id"))
    for t in raw.get("topics") or []:
        for pt in (t.get("points") or []) if isinstance(t, dict) else []:
            if isinstance(pt, dict):
                note(pt.get("who"))
    for td in raw.get("todos") or []:
        if isinstance(td, dict):
            note(td.get("owner"))
    if not seen:
        return raw
    free = [i for i in ids if i not in {q.get("id") for q in raw.get("participants") or []
                                        if isinstance(q, dict)}]
    m = {}
    for x in seen:
        k = int(x[1:]) - 1 if x[1:].isdigit() else -1
        m[x] = ids[k] if 0 <= k < len(ids) else (free.pop(0) if free else x)
    for q in raw.get("participants") or []:
        if isinstance(q, dict) and q.get("id") in m:
            q["id"] = m[q["id"]]
    for t in raw.get("topics") or []:
        for pt in (t.get("points") or []) if isinstance(t, dict) else []:
            if isinstance(pt, dict) and pt.get("who") in m:
                pt["who"] = m[pt["who"]]
    for td in raw.get("todos") or []:
        if isinstance(td, dict) and td.get("owner") in m:
            td["owner"] = m[td["owner"]]
    return raw


_SCHEMA = (
    '{"title": str, '
    '"participants": [{"id": "<speaker id>", "name": str, "role": str, "evidence": str}], '
    '"summary": [str], '
    '"topics": [{"title": str, "points": [{"who": "<speaker id>", "text": str}]}], '
    '"decisions": [str], '
    '"todos": [{"task": str, "owner": "<speaker id>, a role, all, or empty", "due": str, "quote": str}], '
    '"questions": [str], '
    '"keywords": [str]}'
)

_RULES = (
    "Rules:\n"
    "- Use only what is in the transcript. Never invent people, numbers, dates or tasks.\n"
    "- Names: give a participant a name only with clear evidence in the transcript, and put the "
    "exact quote in evidence. Valid evidence: (a) the person introduces themself (\"I'm Tanaka\", "
    "\"山田です\") -> that speaker; (b) someone addresses a person by name and a DIFFERENT speaker "
    "answers right after (\"Tanaka-san, what do you think?\" then S2 answers -> S2 is Tanaka). "
    "A speaker who merely mentions a name (\"as William said\", \"田中さんの資料\") is NOT that "
    "person. If unsure, leave name empty. Keep names in their original script.\n"
    "- role: a job or role the person states or is given (\"I'm the project manager\", "
    "\"司会の佐藤です\", \"as the industrial designer\"). A role is never a name.\n"
    "- who/owner: use the speaker id (S1, Lja, …) of the person. In every other text "
    "(title, summary, points, decisions, tasks, questions) never write speaker ids: use the "
    "person's name, or their role, or a plain description like \"the Japanese speaker\".\n"
    "- todos: concrete things someone said they or someone will do (\"I'll send the slides\", "
    "\"let's check X by Friday\"), including assignments or homework given for the next meeting "
    "(\"you'll look into the user requirements\", \"everyone prepares their part\"). "
    "owner = the speaker id, a role, or \"all\" for the whole group. quote = the original words. "
    "due only if stated.\n"
    "- decisions: choices or conclusions the group settled on, explicitly or by clear "
    "agreement (\"let's go with B\", \"OK, so it'll be yellow\", nobody objecting to a "
    "proposal that the group then follows). A proposal still under discussion, a personal "
    "opinion or a fact is not a decision. questions: questions left unanswered.\n"
    "- Recognition errors may exist; fix obvious ones in your wording but keep proper nouns "
    "as heard if unsure.\n"
    "- Never assume anyone's gender: refer to people by name, or use they/them in English "
    "(no he/she, his/her, 彼/彼女).\n"
    "- Output only JSON, no comments."
)


def _schema(template: str = "free") -> str:
    from . import note_templates as T
    return _SCHEMA[:-1] + T.schema_extra(template) + "}"


def _final_prompt(lang: str, hint: str, body: str, partial: bool, extra: str = "",
                  template: str = "free") -> list[dict]:
    L = EN.get(lang, "Japanese")
    sch = _schema(template)
    if partial:
        task = (f"This is one part of a longer meeting transcript. Extract notes for this part "
                f"only, as JSON with this schema: {sch}")
    else:
        task = (f"Write meeting notes as JSON with this schema: {sch}\n"
                "summary: 3-7 bullet sentences covering the whole meeting. "
                "topics: the main subjects in order, each with who said what.")
    if extra:
        task += "\n\n" + extra
    sys = (f"You are an expert meeting note-taker. Write every text value in {L}, "
           f"even when the meeting was in other languages (quotes in evidence/quote stay in "
           f"their original language).\n{_RULES}")
    user = f"Speakers: {hint}\n\n{task}\n\nTranscript:\n{body}"
    return [{"role": "system", "content": sys}, {"role": "user", "content": user}]


def _merge_prompt(lang: str, hint: str, parts: list[dict], extra: str = "",
                  template: str = "free") -> list[dict]:
    L = EN.get(lang, "Japanese")
    sys = (f"You are an expert meeting note-taker. Write every text value in {L}.\n{_RULES}\n"
           "- Merge duplicates and near-duplicates. Keep every distinct todo and every distinct "
           "decision (merge overlapping ones; at most 12), at most 12 questions, at most 10 keywords.")
    user = (f"Speakers: {hint}\n\nThese are notes extracted from consecutive parts of one "
            f"meeting. Merge them into the final meeting notes as JSON with this schema: "
            f"{_schema(template)}\nsummary: 3-7 bullet sentences covering the whole meeting."
            + (f"\n\n{extra}" if extra else "") + "\n\n"
            + "\n\n".join(f"Part {i + 1}:\n{json.dumps(p, ensure_ascii=False)}"
                          for i, p in enumerate(parts)))
    return [{"role": "system", "content": sys}, {"role": "user", "content": user}]


_LIST_KEYS = ("participants", "summary", "topics", "decisions", "todos", "questions", "keywords")


def _shape(raw: dict) -> dict:
    """JSON の形は守っても中身の型を外した答え(「participants": null、"topics": 3 など。小さい LLM)を
    リストにそろえる(後の処理がリストとして回すので、そのままだと TypeError で全体が止まる)。"""
    for k in _LIST_KEYS:
        v = raw.get(k)
        if isinstance(v, str):
            raw[k] = [v] if v.strip() else []
        elif not isinstance(v, list):
            raw[k] = []
    for t in raw["topics"]:
        if isinstance(t, dict) and not isinstance(t.get("points"), list):
            v = t.get("points")
            t["points"] = [v] if isinstance(v, str) and v.strip() else []
    if not isinstance(raw.get("sections"), (dict, list)):   # テンプレートの節は {key: [..]} か [{key, items}]
        raw.pop("sections", None)
    return raw


def _ask_json(llm: Callable[[list[dict]], str], msgs: list[dict]) -> dict:
    err: Exception | None = None
    for _ in range(2):
        try:
            return _shape(parse_json(llm(msgs)))
        except ValueError as e:     # 壊れた JSON: 1回だけやり直す
            err = e
    raise ValueError(f"メモの形が読めませんでした: {err}")


_SELF = re.compile(
    r"(?:\bI'?m\b|\bI am\b|\bmy name(?:'s| is)\b|\bthis is\b|\bcall me\b|我是|我叫|저는|제 이름은)"
    r"\s*(?P<a>[^\s,.!?、。]+)|(?P<b>[^\s、。]+?)(?:です|と申します|といいます|라고 합니다|입니다)",
    re.IGNORECASE)


# 名前を名乗る言い方だけ(「I'm from Tokyo」の I'm は名乗りではない)
_NAMED = re.compile(r"my name(?:'s| is)|call me|名前は|と申します|といいます|我叫|我的名字|"
                    r"제 이름은|이라고 합니다|라고 합니다", re.IGNORECASE)


# 「I'm X」「This is X」の X が固有名詞らしい(大文字で始まるローマ字で、よくある語でない)
_INTRO_EN = re.compile(r"\b(?i:I'?m|I am|this is)\s+(?P<n>[A-Z][a-zA-Z'-]+)")
_NOT_NAME = {"sorry", "not", "sure", "fine", "good", "great", "glad", "happy", "afraid", "here",
             "just", "so", "the", "a", "an", "ok", "okay", "done", "back", "ready", "also", "still",
             "really", "very", "going", "from", "it", "that", "what", "in", "on", "at", "with",
             "working", "trying", "looking", "thinking", "actually", "basically", "yeah", "yes", "no"}


def _intro_name(text: str) -> bool:
    return any(m.group("n").lower() not in _NOT_NAME for m in _INTRO_EN.finditer(text or ""))


# 1つの行に2人の名乗り(11/07 擬似会議 v3 ja_lab「田中です佐藤です」: 話者交代の間が0.06秒で
# 2人の名乗りが1区間=1行になる。音声の区切りでは直せない = eval/meeting/README.md)
_JA_INTRO = re.compile(r"(?:^|(?<=[。．！？!?\s])|(?<=です))\s*(?P<n>[一-龥々ァ-ヶー]{1,5})です(?![か])")
_NOT_NAME_JA = {"了解", "大丈夫", "本当", "以上", "結構", "問題", "賛成", "同感", "同意", "無理", "確認",
                "最高", "正解", "反対", "必要", "十分", "大切", "大事", "重要", "簡単", "可能", "不明"}


# 時・日付・数・場所によく出る形(「三時です」「月曜です」「五人です」)は名前とみなさない
_NOT_NAME_SHAPE = re.compile(r"[時分日月年曜週円人個回]$|[0-9０-９]")


def _named_elsewhere(name: str, lines: list[dict], skip: int) -> bool:
    """その名前が、ほかの行で呼ばれている(「田中さん」「Tom,」など)。"""
    if re.fullmatch(r"[A-Za-z][A-Za-z'-]*", name):
        pat = re.compile(rf"\b{re.escape(name)}\b")
    else:
        pat = re.compile(re.escape(name) + r"(?:さん|先生|くん|君|様|ちゃん)")
    return any(pat.search(ln["orig"]) for k, ln in enumerate(lines) if k != skip)


def two_intros(text: str) -> "tuple[str, str] | None":
    """行の中の、別々の名前の名乗り2つ (先, 後)。日本語は文の頭の「〇〇です」(漢字・カタカナ1〜5字)、
    英語は I'm / This is + 固有名詞らしい語。ちょうど2つの別の名前のときだけ。"""
    t = text or ""
    names = [m.group("n") for m in _JA_INTRO.finditer(t)
             if m.group("n") not in _NOT_NAME_JA and not _NOT_NAME_SHAPE.search(m.group("n"))]
    names += [m.group("n") for m in _INTRO_EN.finditer(t) if m.group("n").lower() not in _NOT_NAME]
    uniq = list(dict.fromkeys(names))
    return (uniq[0], uniq[1]) if len(uniq) == 2 and len(names) == 2 else None


_CALL_JA = re.compile(r"(?P<n>[一-龥々ァ-ヶー]{1,5})(?:さん|先生|くん|君|様)")
_NOT_CALL = {"皆", "皆様", "皆さま", "みな"}


def called_over_short_intro(parts: list[dict], lines: list[dict], max_chars: int = 6) -> None:
    """短い名乗り1行だけの名前が、会議の中で呼ばれた別の名前と食い違えば、呼ばれた名前を採る。

    11/07 擬似会議 ja_dev4: 森さんの「森です」が「堀です」と聞こえて渡辺さん(S2)の行になり、S2=堀 が
    本人の名乗りとして確かめを通った(作り話の名前)。S2 は「渡辺さんからお願いします」の直後に話している。
    働くのは全部そろうときだけ: 名乗りの行が「〇〇です」程度に短い(記号を除き max_chars 字以下)・その名乗りは
    その話者で1回だけ・その話者が直前の行で別の名前で「〇〇さん/先生…」と呼ばれている(_verify_name の b と
    同じ「呼ばれて2行以内に話す」。「皆さん」は数えない)。呼ばれた名前がちょうど1つで、ほかの人に付いて
    いなければそれにし(根拠は呼んだ行)、そうでなければ名前なし。直した名前(fixed)には触れない。
    擬似会議26回の記録: 日本語の名乗り53件のうち働くのは7件で全部が聞き違い・取り違え、正しい名前を落とすのは0。"""
    used = {p["name"] for p in parts if p.get("name")}
    for p in parts:
        a, spk = p.get("name") or "", p["id"]
        if not a or p.get("fixed"):
            continue
        intro = [ln for ln in lines if ln["spk"] == spk
                 and any(m.group("n") == a for m in _JA_INTRO.finditer(ln["orig"]))]
        if len(intro) != 1 or len(_norm_txt(intro[0]["orig"])) > max_chars:
            continue
        called: dict = {}
        for i, ln in enumerate(lines):
            nxt = [x["spk"] for x in lines[i + 1:i + 3] if x["spk"] != ln["spk"]]
            if not nxt or nxt[0] != spk:
                continue
            for m in _CALL_JA.finditer(ln["orig"]):
                if m.group("n") not in _NOT_CALL:
                    called.setdefault(m.group("n"), ln["orig"])
        if not called or a in called:
            continue
        used.discard(a)
        if len(called) == 1:
            b, ev = next(iter(called.items()))
            if b not in used:
                p["name"], p["evidence"] = b, ev
                used.add(b)
                continue
        p["name"], p["evidence"] = "", ""


def two_intro_names(parts: list[dict], lines: list[dict], llm_names: "set | None" = None) -> None:
    """2つの名乗りがある行から、隣の話者の名前を補う(付いている名前は変えない・同じ名前を2人に付けない)。
      行の話者がもう後の名前なら、先の名前はその行の前に話した別の話者へ
      行の話者がもう先の名前なら、後の名前はその行の後に話す別の話者へ
      行の話者にまだ名前が無ければ、先の名前を行の話者に、後の名前を後に話す別の話者へ。ただし
      どちらかの名前に行の外の根拠(ほかの行で「〇〇さん」と呼ばれている/LLM が名前に挙げた)が
      あるときだけ(「東京です。大阪です。」で2人に名前を作らない)"""
    by = {p["id"]: p for p in parts}
    used = {p["name"] for p in parts if p.get("name")}

    def give(spk, name, ev):
        p = by.get(spk)
        if p is None or p.get("name") or p.get("fixed") or name in used:
            return
        p["name"], p["evidence"], p["minor"] = name, ev, False
        used.add(name)
    for i, ln in enumerate(lines):
        two = two_intros(ln["orig"])
        if not two:
            continue
        a, b = two
        x = ln["spk"]
        prev = next((l["spk"] for l in reversed(lines[:i]) if l["spk"] != x), None)
        nxt = next((l["spk"] for l in lines[i + 1:] if l["spk"] != x), None)
        own = (by.get(x) or {}).get("name", "")
        if own == b:
            if prev:
                give(prev, a, ln["orig"])
        elif own == a:
            if nxt:
                give(nxt, b, ln["orig"])
        elif not own and nxt and any(
                n in (llm_names or set()) or _named_elsewhere(n, lines, i) for n in (a, b)):
            give(x, a, ln["orig"])
            give(nxt, b, ln["orig"])


def _norm_txt(t: str) -> str:
    return re.sub(r"[\s\"'「」『』“”‘’.,!?。、！？]+", "", t or "").lower()


def _verify_name(spk: str, name: str, evidence: str, lines: list[dict]) -> bool:
    """名前に根拠が本当にあるか(LLM の取り違えを機械で落とす。11/06 AMI:
    「Me and William worked on…」と言った人を William にした)。
      a) その人自身の発言で名乗っている(I'm X / Xです …)
      b) 別の人が名前を呼び、すぐ後(2行以内)にその人が話している"""
    name = (name or "").strip()
    if not name:
        return False
    key = _norm_txt(name)
    ev = re.sub(r"^\s*(?:S\d+|L\w+)\s*[:：]\s*", "", evidence or "")
    evn = _norm_txt(ev)[:30]
    for i, ln in enumerate(lines):
        o = _norm_txt(ln["orig"])
        if key not in o or (evn and evn[:15] not in o and o[:15] not in evn):
            continue
        if ln["spk"] == spk:
            for m in _SELF.finditer(ln["orig"]):
                got = _norm_txt(m.group("a") or m.group("b") or "")
                if got and (key.startswith(got) or got.startswith(key) or key in got):
                    return True
        else:
            nxt = [x["spk"] for x in lines[i + 1:i + 3] if x["spk"] != ln["spk"]]
            if nxt and nxt[0] == spk:
                return True
    # 名前が別の文字で書かれた(記録はローマ字の「My name is Onishi Shinnosuke」、メモは
    # 「大西慎之介」): 根拠がその人自身の発言の名乗りなら認める(11/06 実記録)。
    # 「I'm Emily from …」→ メモは「エミリー」も同じ(11/06 擬似会議 jaen)
    if evn and (_NAMED.search(ev) or _intro_name(ev)):
        for ln in lines:
            if ln["spk"] == spk and evn[:15] in _norm_txt(ln["orig"]):
                return True
    return False


def _find_evidence(spk: str, name: str, lines: list[dict]) -> str:
    """名前の根拠になる行を記録から探す(名前を含む行を順に _verify_name に通す)。"""
    key = _norm_txt(name)
    if not key:
        return ""
    for ln in lines:
        if key in _norm_txt(ln["orig"]) and _verify_name(spk, name, ln["orig"], lines):
            return ln["orig"]
    return ""


def _grams(t: str) -> set:
    """重なりを測る単位: ローマ字の多い文は語(3字以上)、それ以外は文字の2つ組。"""
    t = (t or "").lower()
    if len(re.findall(r"[a-z]", t)) > len(t) / 2:
        return {w for w in re.findall(r"[a-z0-9']+", t) if len(w) >= 3}
    c = re.sub(r"[\s、。,.!?！？「」()（）]+", "", t)
    return {c[k:k + 2] for k in range(len(c) - 1)}


def _latin(t: str) -> bool:
    t = t or ""
    return len(re.findall(r"[A-Za-z]", t)) > len(t) / 2


def _repeats_todo(decision: str, todos: list[dict], th: "float | None" = None) -> bool:
    """決定事項が、やることの言い換え(誰が・何を・いつまでに)か(11/07 擬似会議 v2 ja_lab:
    やること2件が決定事項にも重ねて入った)。やることの単位の th 以上が決定の中にあれば重複。
    th は英語(語)0.75・それ以外(文字の2つ組)0.65。日本語は言い換えで助詞が変わりやすく
    (v3 ja_agenda「予算が出るか経理に確認する」→「予算の有無について…経理に確認することにした」0.67)、
    英語は 0.65 だと AMI の本物の決定を2件落とした。"""
    d = _grams(decision)
    th = th if th is not None else (0.75 if _latin(decision) else 0.65)
    for td in todos:
        g = _grams(td.get("task", ""))
        if len(g) >= 3 and len(g & d) / len(g) >= th:
            return True
    return False


def _wrong_lang(text: str, lang: str) -> bool:
    """メモの言語と違う文か(ja/zh/ko なのに字がほぼローマ字、en なのにローマ字がほぼ無い)。
    「Q3 の KPI」のような混ざりは正しいとみる。"""
    t = re.sub(r"[\s\d\W_]+", "", text or "")
    if len(t) < 4:
        return False
    lat = len(re.findall(r"[A-Za-z]", t)) / len(t)
    return lat > 0.7 if lang in ("ja", "zh", "ko") else (lat < 0.2 if lang == "en" else False)


def _fix_language(notes: dict, lang: str, llm: Callable[[list[dict]], str]) -> None:
    """違う言語で書かれた項目だけを1回の LLM で訳し直す(11/07 擬似会議 v3: この Mac の
    gemma-3-4b が日本語の会議でもやることを「話者3 will create the test script…」と英語で書いた)。"""
    refs: list = []          # (入れ物, 鍵) — 訳した文を戻す場所
    for k in ("summary", "decisions", "questions"):
        for i, x in enumerate(notes.get(k) or []):
            if isinstance(x, str) and _wrong_lang(x, lang):
                refs.append((notes[k], i))
    for t in notes.get("topics") or []:
        if _wrong_lang(t.get("title", ""), lang):
            refs.append((t, "title"))
        for pt in t.get("points") or []:
            if _wrong_lang(pt.get("text", ""), lang):
                refs.append((pt, "text"))
    for td in notes.get("todos") or []:
        if _wrong_lang(td.get("task", ""), lang):
            refs.append((td, "task"))
    if not refs:
        return
    src = [c[k] for c, k in refs]
    msgs = [{"role": "system", "content":
             f"Translate each item into {EN.get(lang, lang)}. Keep names, numbers and technical terms. "
             'Output only JSON: {"items": [...]} with the same number of items in the same order.'},
            {"role": "user", "content": json.dumps({"items": src}, ensure_ascii=False)}]
    try:
        got = parse_json(llm(msgs)).get("items") or []
    except Exception:  # noqa: BLE001 - 訳せなければそのまま(メモは捨てない)
        return
    if not isinstance(got, list) or len(got) != len(src):   # 数・文字列(1字ずつ戻さない)は使わない
        return
    for (c, k), new in zip(refs, got):
        if isinstance(new, str) and new.strip():
            c[k] = new.strip()


def _dup_key(t: str) -> str:
    return re.sub(r"[\s、。,.!?！？「」『』()（）:：;；・\-]+", "", str(t or "").lower())


def _is_dup(a: str, b: str) -> bool:
    """同じ文(大小・句読点・空白の違いだけ)か、ほぼ同じ文(語/2文字組の重なり 0.85 以上)。"""
    ka, kb = _dup_key(a), _dup_key(b)
    if not ka or not kb:
        return False
    if ka == kb:
        return True
    ga, gb = _grams(a), _grams(b)
    return len(ga) >= 4 and len(gb) >= 4 and len(ga & gb) / len(ga | gb) >= 0.85


def _dedupe_idx(texts: list[str]) -> list[int]:
    """残す項目の添字(先に出たものを残す)。"""
    keep: list[int] = []
    for k, t in enumerate(texts):
        if not any(_is_dup(t, texts[j]) for j in keep):
            keep.append(k)
    return keep


def _clean(notes: dict, lines: list[dict], known: dict, old: "dict | None") -> dict:
    """LLM の出力を整える: 型をそろえ、記録に出てくる話者を全員入れ、直した名前・
    チェック済みのやることを引き継ぐ。"""
    def strs(v):
        return [str(x).strip() for x in (v or []) if str(x).strip()] if isinstance(v, list) else []

    ids = []
    count: dict[str, int] = {}
    langs: dict[str, dict] = {}
    for ln in lines:
        if ln["spk"] not in ids:
            ids.append(ln["spk"])
        count[ln["spk"]] = count.get(ln["spk"], 0) + 1
        langs.setdefault(ln["spk"], {}).setdefault(ln.get("lang", ""), 0)
        langs[ln["spk"]][ln.get("lang", "")] += 1
    got = {p.get("id"): p for p in notes.get("participants", []) if isinstance(p, dict)}
    oldp = {p.get("id"): p for p in (old or {}).get("participants", [])}
    parts = []
    for i in ids:
        p = got.get(i, {})
        name = str(p.get("name") or "").strip()
        ev = str(p.get("evidence") or "").strip()
        if known.get(i):
            name, fixed = known[i], True
        elif oldp.get(i, {}).get("fixed"):
            name, fixed = oldp[i].get("name", ""), True
        else:
            fixed = False
        lg = max(langs.get(i, {"": 1}).items(), key=lambda kv: kv[1])[0]
        if name and not fixed and not _verify_name(i, name, ev, lines):
            # LLM が根拠に別の文を選ぶことがある(11/07 擬似会議 v2: 高橋さんの根拠が「では森さんは…」)。
            # 記録の中に本人の名乗り/呼ばれて答えた行があれば、その行を根拠にする
            ev = _find_evidence(i, name, lines)
            if not ev:
                name = ""                # 根拠のない名前は付けない(役割は残す)
        role = str(p.get("role") or "").strip()
        if role and name and role.lower() == name.lower():
            name = "" if not fixed else name     # 役職を名前にしない(11/06 AMI: "Industrial Designer")
        # 発言が少ない話者(11/06 AMI: 4人の会議で声の見分けが5〜8人を出した。相づちや
        # 小さい声の切れ端)は minor: 画面では畳む。名前が付いた人・直した人は畳まない。
        minor = (not name and not fixed and count.get(i, 0) < max(2, 0.05 * len(lines)))
        parts.append({"id": i, "name": name, "role": role, "lang": lg, "evidence": ev,
                      "lines": count.get(i, 0), "fixed": fixed, "minor": minor})
    called_over_short_intro(parts, lines)
    two_intro_names(parts, lines, {str(p.get("name") or "").strip() for p in got.values()} - {""})
    topics = []
    for t in notes.get("topics") or []:
        if not isinstance(t, dict):
            continue
        pts = [{"who": str(x.get("who") or ""), "text": str(x.get("text") or "").strip(),
                "src": list(x.get("src") or [])}
               for x in (t.get("points") or []) if isinstance(x, dict) and x.get("text")]
        if t.get("title") or pts:
            topics.append({"title": str(t.get("title") or "").strip(), "points": pts,
                           "src": list(t.get("src") or [])})
    done_before = {str(x.get("task", "")).strip(): bool(x.get("done"))
                   for x in (old or {}).get("todos", [])}
    todos = []
    for x in notes.get("todos") or []:
        if not isinstance(x, dict) or not str(x.get("task") or "").strip():
            continue
        task = str(x["task"]).strip()
        todos.append({"task": task, "owner": str(x.get("owner") or "").strip(),
                      "due": str(x.get("due") or "").strip(),
                      "quote": str(x.get("quote") or "").strip(),
                      "done": done_before.get(task, False), "src": list(x.get("src") or [])})
    out = {"title": str(notes.get("title") or "").strip(), "participants": parts,
           "summary": strs(notes.get("summary")), "topics": topics,
           "decisions": strs(notes.get("decisions")), "todos": todos,
           "questions": strs(notes.get("questions")), "keywords": strs(notes.get("keywords"))}
    for k in ("summary", "decisions", "questions"):   # 根拠の行(_take_refs が同じ並びで作る)
        src = notes.get(f"{k}_src")
        out[f"{k}_src"] = src if isinstance(src, list) and len(src) == len(out[k]) else [[] for _ in out[k]]
    # 区切りごとの抜き出しを統合した時に残る言い換え・同じ文の重複を整理(根拠 *_src も同じ並びで)
    for k in ("summary", "decisions", "questions"):
        ix = _dedupe_idx(out[k])
        if len(ix) != len(out[k]):
            out[k] = [out[k][j] for j in ix]
            out[f"{k}_src"] = [out[f"{k}_src"][j] for j in ix]
    merged: list[dict] = []
    for td in todos:
        for m in merged:
            if _is_dup(td["task"], m["task"]) and (not td["owner"] or not m["owner"] or td["owner"] == m["owner"]):
                for f in ("owner", "due", "quote"):
                    m[f] = m[f] or td[f]
                m["src"] = m["src"] or td["src"]
                break
        else:
            merged.append(td)
    todos = out["todos"] = merged
    keep = [k for k, d in enumerate(out["decisions"]) if not _repeats_todo(d, todos)]   # やることの言い換えは重ねない
    out["decisions"] = [out["decisions"][k] for k in keep]
    out["decisions_src"] = [out["decisions_src"][k] for k in keep]
    if isinstance(notes.get("sections"), dict):
        out["sections"] = notes["sections"]
    return out


def _mechanical_merge(parts: list[dict]) -> dict:
    """LLM でまとめられなかった時の受け皿: 区切りごとの抜き出しをそのまま並べる(重複は落とす)。"""
    out: dict = {"title": "", "participants": [], "summary": [], "topics": [], "decisions": [],
                 "todos": [], "questions": [], "keywords": []}
    seen: dict = {k: set() for k in out}
    for p in parts:
        if not out["title"] and p.get("title"):
            out["title"] = p["title"]
        for k in ("summary", "decisions", "questions", "keywords"):
            for x in p.get(k) or []:
                if str(x) not in seen[k]:
                    seen[k].add(str(x))
                    out[k].append(x)
        out["topics"] += [t for t in p.get("topics") or [] if isinstance(t, dict)]
        for t in p.get("todos") or []:
            if isinstance(t, dict) and str(t.get("task")) not in seen["todos"]:
                seen["todos"].add(str(t.get("task")))
                out["todos"].append(t)
        for q in p.get("participants") or []:
            if isinstance(q, dict) and q.get("name") and q.get("id") not in seen["participants"]:
                seen["participants"].add(q.get("id"))
                out["participants"].append(q)
        if isinstance(p.get("sections"), dict):      # テンプレートの節(11/07 W1)
            sec = out.setdefault("sections", {})
            for k, v in p["sections"].items():
                for x in v if isinstance(v, list) else []:
                    if x not in sec.setdefault(k, []):
                        sec[k].append(x)
    out["summary"] = out["summary"][:10]
    return out


def _merge_all(llm, lang: str, hint: str, parts: list[dict], group: int = 4, extra: str = "",
               template: str = "free") -> dict:
    """区切りごとの抜き出しを1つのメモに。多いときは4つずつまとめてから、さらにまとめる
    (11/06 AMI 30分の会議: 10区切りを一度にまとめると出力が長すぎて JSON が途中で切れた)。"""
    if not parts:
        raise ValueError("どの区切りも読めませんでした")
    if len(parts) == 1:
        return parts[0]
    while len(parts) > group:
        nxt = []
        for i in range(0, len(parts), group):
            g = parts[i:i + group]
            if len(g) == 1:
                nxt.append(g[0])
                continue
            try:
                nxt.append(_ask_json(llm, _merge_prompt(lang, hint, g, extra, template)))
            except ValueError:
                nxt.append(_mechanical_merge(g))
        parts = nxt
    try:
        return _ask_json(llm, _merge_prompt(lang, hint, parts, extra, template))
    except ValueError:
        return _mechanical_merge(parts)


_ID_RE = re.compile(r"(?<![A-Za-z0-9])(S\d+|L(?:ja|en|zh|ko))(?![A-Za-z0-9])")


def _humanize(out: dict, ui: str) -> dict:
    """本文に残った話者 id(「Ljaが…」「S2 said…」)を名前か既定の呼び名に置き換える。"""
    names = {p["id"]: (p.get("name") or default_name(p["id"], ui)) for p in out["participants"]}

    def fix(t: str) -> str:
        return _ID_RE.sub(lambda m: names.get(m.group(1), default_name(m.group(1), ui)), t)
    out["title"] = fix(out["title"])
    for k in ("summary", "decisions", "questions"):
        out[k] = [fix(x) for x in out[k]]
    for t in out["topics"]:
        t["title"] = fix(t["title"])
        for pt in t["points"]:
            pt["text"] = fix(pt["text"])
    for td in out["todos"]:
        td["task"] = fix(td["task"])
    for items in (out.get("sections") or {}).values():
        for x in items:
            x["text"] = fix(x["text"])
    return out


def spelling_hint(meeting: dict, max_terms: int = 80, max_chars: int = 800) -> str:
    """用語辞書(会議の開始時の写し)→ LLM への綴りの手がかり。無ければ ""。
    聞き取りの誤り(「養老」← YOLO)を、メモでは正しい綴りに直してもらう。"""
    out: list[str] = []
    n = 0
    for t in meeting.get("terms") or []:
        t = str(t).strip()
        if not t or t in out:
            continue
        if len(out) >= max_terms or n + len(t) > max_chars:
            break
        out.append(t)
        n += len(t) + 2
    if not out:
        return ""
    return ("\nCorrect spellings of names and terms that may appear (speech recognition may have "
            "misheard them; when a word in the transcript sounds like one of these, write it this "
            "way): " + ", ".join(out) + ".")


def _term_re(term: str):
    """英数字だけの用語の、大小・区切り(空白/ハイフン)違いに当たる正規表現。直せない用語は None。
    Go のような普通の語(先頭だけ大文字)は、本物の語を壊すので対象にしない。"""
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9 .\-]{1,38}[A-Za-z0-9]", term) or len(term) < 3:
        return None
    if not (re.search(r"[A-Z]", term[1:]) or re.search(r"\d", term)):
        return None
    parts = []
    for k, ch in enumerate(term):
        if k:
            a = term[k - 1]
            if ch in " -" or a in " -":
                pass
            elif (a.islower() and ch.isupper()) or (a.isalpha() != ch.isalpha()):
                parts.append("[ -]?")
        parts.append(r"[ -]?" if ch in " -" else re.escape(ch))
    return re.compile(r"(?<![A-Za-z0-9])" + "".join(parts) + r"(?![A-Za-z0-9])", re.I)


def fix_terms(text: str, terms: list) -> str:
    """用語辞書の綴りに、大小・空白の違い(yolo / chat GPT)だけ直す。意味の言い換えはしない。"""
    for t in terms or []:
        rx = _term_re(str(t).strip())
        if rx:
            text = rx.sub(str(t).strip(), text)
    return text


def fix_terms_in(out: dict, terms: list) -> dict:
    """できたメモ(dict)の文に fix_terms をかける(根拠の番号などは触らない)。"""
    terms = [t for t in terms or [] if _term_re(str(t).strip())]
    if not terms:
        return out
    f = lambda x: fix_terms(x, terms)  # noqa: E731
    out["title"] = f(out.get("title", ""))
    for k in ("summary", "decisions", "questions", "keywords"):
        out[k] = [f(x) for x in out.get(k, [])]
    for t in out.get("topics", []):
        t["title"] = f(t.get("title", ""))
        for pt in t.get("points", []):
            pt["text"] = f(pt.get("text", ""))
    for td in out.get("todos", []):
        td["task"] = f(td.get("task", ""))
    secs = out.get("sections")
    for items in (secs.values() if isinstance(secs, dict) else []):
        for x in items:
            x["text"] = f(x.get("text", ""))
    return out


GROUND_CHARS = 9000


def _ground_items(out: dict) -> list:
    """根拠を当てる項目: (入れ物, 鍵, LLM に見せる文)。"""
    items = []
    for k in ("summary", "decisions", "questions"):
        src = out.setdefault(f"{k}_src", [[] for _ in out.get(k, [])])
        for i, x in enumerate(out.get(k) or []):
            items.append((src, i, x))
    for t in out.get("topics") or []:
        for p in t.get("points") or []:
            items.append((p, "src", p.get("text", "")))
    for td in out.get("todos") or []:
        txt = td.get("task", "") + (f" (said: \"{td['quote']}\")" if td.get("quote") else "")
        items.append((td, "src", txt))
    for sec in (out.get("sections") or []):
        for x in sec.get("items") or []:
            items.append((x, "src", x.get("text", "")))
    return items


def ground(out: dict, lines: list[dict], llm: Callable[[list[dict]], str], workers: int = 2,
           chunk_chars: int = GROUND_CHARS) -> dict:
    """メモの各項目に根拠の行(記録の添字)を当てる(11/07 W1)。

    はじめは本文を作る指示に「各項目の終わりに [#12]」と書かせていたが、AMI 6会議×3回で要約 −6点・
    やること −13点(今までの版と同じ日に比べた)。本文の指示は今までのまま戻し、できたメモの項目を
    番号つきの記録(約9000字ずつ)と並べて「どの行が支えるか」だけを答えさせる(返事は番号だけで短い)。
    区切りの外の番号・記録に無い番号は捨てる。失敗した区切りは飛ばす(メモは止めない)。"""
    items = _ground_items(out)
    if not items or not lines:
        return out
    numbered_lines = [ln if "i" in ln else dict(ln, i=k) for k, ln in enumerate(lines)]
    listing = "\n".join(f"{k + 1}. {str(t)[:300]}" for k, (_c, _key, t) in enumerate(items))
    parts = chunks(numbered_lines, chunk_chars)
    found: list = [[] for _ in items]
    lock = threading.Lock()
    sys = ("You link meeting-note items to the transcript lines that support them. Each transcript "
           "line starts with its number (#12). For each item that is supported by lines in this "
           "excerpt, give the 1-3 most relevant line numbers. Skip items this excerpt does not "
           'support. Output only JSON: {"refs": {"<item number>": [line numbers]}}.')

    def one(part):
        ok = {ln["i"] for ln in part}
        msgs = [{"role": "system", "content": sys},
                {"role": "user", "content": f"Items:\n{listing}\n\nTranscript excerpt:\n{transcript(part)}"}]
        try:
            refs = parse_json(llm(msgs)).get("refs") or {}
        except Exception:  # noqa: BLE001 - 1区切りが読めなくても全体は止めない
            return
        if not isinstance(refs, dict):
            return
        with lock:
            for key, nums in refs.items():
                try:
                    k = int(str(key).strip().rstrip(".")) - 1
                except ValueError:
                    continue
                if not 0 <= k < len(items) or not isinstance(nums, list):
                    continue
                for x in nums[:3]:
                    try:
                        x = int(str(x).lstrip("#"))
                    except ValueError:
                        continue
                    if x in ok and x not in found[k]:
                        found[k].append(x)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        list(ex.map(one, parts))
    for (c, key, _t), got in zip(items, found):
        cur = c[key] if isinstance(c, list) else c.get(key)
        merged = sorted(set(cur or []) | set(got))[:4]
        if isinstance(c, list):
            c[key] = merged
        else:
            c[key] = merged
    return out


def make_notes(meeting: dict, lang: str, llm: Callable[[list[dict]], str],
               progress: Callable[[str, float], None] | None = None,
               model: str = "", known: dict | None = None,
               old: "dict | None" = None, chunk_chars: int = CHUNK_CHARS,
               workers: int = 2, template: "str | None" = None) -> dict:
    """会議の記録 → メモ(dict)。llm(messages)->str。progress(stage, 0..1)。

    11/07 W1: できたメモの各項目に根拠の行(*_src / src)を後から当て(ground)、章(chapters)を作る。
    template(会議の種類)は 引数 → 記録の template → meta の template → 自由 の順。議題(agenda)・
    会議中の自分のメモ(memos)・ブックマーク(marks)は記録(load_meeting)にあれば使う。"""
    from . import note_templates as T
    lang = lang if lang in LANGS else "ja"
    meta = meeting.get("meta", {}) or {}
    plain = meeting["lines"]                                   # 作る時は今までどおり番号なし
    lines = [dict(ln, i=k) for k, ln in enumerate(plain)]      # 根拠を当てる時だけ番号つき
    known = known or {}
    prog = progress or (lambda st, x: None)
    if not lines:
        raise ValueError("記録が空です")
    tpl = T.normalize(template or meeting.get("template") or meta.get("template"))
    ag = meeting.get("agenda") or meta.get("agenda") or ""
    if isinstance(ag, list):      # W3 の load_meeting: [{text, done, t}]
        ag = "\n".join(f"{k + 1}. {a.get('text', '')}" for k, a in enumerate(ag)
                       if isinstance(a, dict) and a.get("text"))
    agenda = str(ag)
    memos = [m for m in (meeting.get("memos") or []) if isinstance(m, dict)]
    marks = [m for m in (meeting.get("marks") or []) if isinstance(m, dict)]
    ctx = {st: T.context(tpl, agenda, memos, lang, st) for st in ("full", "part", "merge")}
    hint = _speaker_hint(lines, known) + spelling_hint(meeting)   # 用語辞書の綴り(W3)
    parts = chunks(plain, chunk_chars)
    prog("read", 0.05)
    if len(parts) == 1:
        raw = _ask_json(llm, _final_prompt(lang, hint, transcript(plain), partial=False,
                                           extra=ctx["full"], template=tpl))
        prog("merge", 0.9)
    else:
        # 区切りごとの抜き出しは2本ずつ並べる(リモートサーバーの LLM は2本同時でもほぼ遅くならない)
        got: list = [None] * len(parts)
        done = [0]
        lock = threading.Lock()

        def one(k: int) -> None:
            try:
                got[k] = _ask_json(llm, _final_prompt(lang, hint, transcript(parts[k]), partial=True,
                                                      extra=ctx["part"], template=tpl))
            except ValueError:
                got[k] = {}          # 1区切りが読めなくても全体は止めない
            with lock:
                done[0] += 1
                prog("chunk", 0.05 + 0.75 * done[0] / len(parts))
        with ThreadPoolExecutor(max_workers=workers) as ex:
            list(ex.map(one, range(len(parts))))
        raw = _merge_all(llm, lang, hint, [g for g in got if g], extra=ctx["merge"], template=tpl)
        prog("merge", 0.95)
    ids: list = []
    for ln in lines:
        if ln["spk"] not in ids:
            ids.append(ln["spk"])
    raw = _take_refs(_fix_ids(raw, ids), len(lines))
    out = _humanize(_clean(raw, lines, known, old), lang)
    fix_terms_in(out, meeting.get("terms") or [])      # 用語辞書の綴り(大小・空白の違い)
    secs = out.pop("sections", None) or {}
    if T.section_keys(tpl):
        out["sections"] = [{"key": k, "title": T.section_title(tpl, k, lang), "items": secs.get(k, [])}
                           for k in T.section_keys(tpl)]
    langs_used = []
    for ln in lines:
        if ln.get("lang") and ln["lang"] not in langs_used:
            langs_used.append(ln["lang"])
    out.update({"session": meeting["id"], "made": time.strftime("%Y-%m-%d %H:%M:%S"),
                "model": model, "lang": lang, "date": (meta.get("start") or "")[:10],
                "start": meta.get("start", ""), "minutes": _minutes(meta, lines),
                "languages": langs_used, "mode": meta.get("mode", "interpret"), "template": tpl})
    if agenda:
        out["agenda"] = agenda
    _fix_language(out, lang, llm)   # 違う言語で書かれた項目だけ訳し直す(小さい LLM 対策)
    prog("merge", 0.97)
    ground(out, lines, llm, workers=workers)   # 根拠の行(11/07 W1: 本文を作った後で当てる)
    for td in out["todos"]:       # 根拠が見つからなかったやることは、原文の引用から行を探す
        if not td["src"] and td.get("quote"):
            k = _quote_line(td["quote"], lines)
            td["src"] = [k] if k is not None else []
    _chapters(out, lines, meta.get("ts0"))
    if memos or marks:            # 自分のメモ・ブックマークをメモに織り込む(1回だけ LLM)
        from . import meeting_ai
        meeting_ai.enrich(out, lines, memos, marks, lang, llm)
    return out


# ---------------- Markdown ----------------

def who_name(notes: dict, who: str, ui: str) -> str:
    for p in notes.get("participants", []):
        if p.get("id") == who:
            return p.get("name") or default_name(who, ui)
    return who if who and not re.fullmatch(r"(S\d+|L\w+)", who) else (default_name(who, ui) if who else "")


def to_markdown(notes: dict, meeting: "dict | None" = None, with_transcript: bool = True) -> str:
    ui = notes.get("lang", "ja")
    H = HEAD.get(ui, HEAD["ja"])
    L = LANG_LABEL.get(ui, LANG_LABEL["ja"])
    out = [f"# {notes.get('title') or notes.get('session', '')}", ""]
    langs = " / ".join(L.get(x, x) for x in notes.get("languages", []))
    out.append(f"{notes.get('start', '')}  ·  {notes.get('minutes', 0)}{H['minutes']}"
               + (f"  ·  {langs}" if langs else ""))
    out.append("")
    out.append(f"## {H['participants']}")
    for p in notes.get("participants", []):
        nm = p.get("name") or default_name(p["id"], ui)
        extra = [x for x in (p.get("role"), L.get(p.get("lang"), p.get("lang"))) if x]
        out.append(f"- {nm}" + (f" ({', '.join(extra)})" if extra else ""))
    W = HEAD_W1.get(ui, HEAD_W1["ja"])
    if notes.get("my_notes"):        # 会議中の自分のメモ(11/07 W1: Granola のように、自分の言葉が先)
        out += ["", f"## {W['my_notes']}"]
        for x in notes["my_notes"]:
            out.append(f"- **{x.get('memo', '')}**" + (f" — {x['text']}" if x.get("text") else "")
                       + (f" ({x['t']})" if x.get("t") else ""))
    for key in ("summary",):
        out += ["", f"## {H[key]}"] + [f"- {x}" for x in notes.get(key, [])]
    if notes.get("topics"):
        out += ["", f"## {H['topics']}"]
        for t in notes["topics"]:
            out.append(f"### {t.get('title', '')}" + (f" ({t['t']})" if t.get("t") else ""))
            for pt in t.get("points", []):
                nm = who_name(notes, pt.get("who", ""), ui)
                out.append(f"- " + (f"**{nm}**: " if nm else "") + pt.get("text", ""))
    for sec in notes.get("sections") or []:     # テンプレートの節
        if sec.get("items"):
            out += ["", f"## {sec.get('title') or sec.get('key')}"] + [f"- {x.get('text', '')}" for x in sec["items"]]
    out += ["", f"## {H['decisions']}"] + ([f"- {x}" for x in notes.get("decisions", [])] or [f"- {H['none']}"])
    out += ["", f"## {H['todos']}"]
    for x in notes.get("todos", []) or []:
        extra = []
        if x.get("owner"):
            extra.append(f"{H['owner']}: {who_name(notes, x['owner'], ui)}")
        if x.get("due"):
            extra.append(f"{H['due']}: {x['due']}")
        out.append(f"- [{'x' if x.get('done') else ' '}] {x['task']}"
                   + (f" ({', '.join(extra)})" if extra else ""))
    if not notes.get("todos"):
        out.append(f"- {H['none']}")
    if notes.get("questions"):
        out += ["", f"## {H['questions']}"] + [f"- {x}" for x in notes["questions"]]
    if notes.get("highlights") and not (meeting or {}).get("marks"):   # 印の場面(W3 の一覧が無い時だけ)
        out += ["", f"## {W['highlights']}"]
        out += [f"- `{x.get('t', '')}` " + (f"**{x['label']}** " if x.get("label") else "") + (x.get("text") or "")
                for x in notes["highlights"]]
    if notes.get("keywords"):
        out += ["", f"## {H['keywords']}", ", ".join(notes["keywords"])]
    if meeting:
        out += _w3_markdown(meeting, ui)   # 議題・ブックマーク・自分のメモ(W3)
    if with_transcript and meeting:
        out += ["", f"## {H['transcript']}"]
        for ln in meeting["lines"]:
            nm = who_name(notes, ln["spk"], ui)
            row = f"- `{ln.get('t', '')}` **{nm}**: {ln['orig']}"
            if ln.get("trans") and not ln["trans"].startswith("（"):
                row += f"  \n  → {ln['trans']}"
            out.append(row)
    return "\n".join(out) + "\n"


W3_HEAD = {
    "ja": {"agenda": "議題", "marks": "ブックマークと自分のメモ", "memo": "メモ",
           "star": "重要", "decision": "決定", "todo": "やること", "question": "質問"},
    "en": {"agenda": "Agenda", "marks": "Bookmarks and my notes", "memo": "Note",
           "star": "Important", "decision": "Decision", "todo": "To do", "question": "Question"},
    "zh": {"agenda": "议程", "marks": "书签和我的笔记", "memo": "笔记",
           "star": "重要", "decision": "决定", "todo": "待办", "question": "问题"},
    "ko": {"agenda": "안건", "marks": "북마크와 내 메모", "memo": "메모",
           "star": "중요", "decision": "결정", "todo": "할 일", "question": "질문"},
}


def _w3_markdown(meeting: dict, ui: str) -> list[str]:
    """議題(済みに印)と、ブックマーク・自分のメモを時刻順に。どちらも無ければ何も足さない。"""
    W = W3_HEAD.get(ui, W3_HEAD["ja"])
    out: list[str] = []
    if meeting.get("agenda"):
        out += ["", f"## {W['agenda']}"]
        out += [f"- [{'x' if a.get('done') else ' '}] {a['text']}" for a in meeting["agenda"]]
    lines = meeting.get("lines") or []
    items = [(m.get("ts") or 0, 0, m) for m in meeting.get("marks") or []]
    items += [(m.get("ts") or 0, 1, m) for m in meeting.get("memos") or []]
    if items:
        out += ["", f"## {W['marks']}"]
        for _ts, is_memo, m in sorted(items, key=lambda x: (x[0], x[1])):
            t = f"`{m['t']}` " if m.get("t") else ""
            if is_memo:
                out.append(f"- {t}✎ {m['text']}")
                continue
            q = ""
            if 0 <= m.get("i", -1) < len(lines):
                o = lines[m["i"]]["orig"]
                q = f" — 「{o[:80]}{'…' if len(o) > 80 else ''}」"
            lab = f": {m['label']}" if m.get("label") else ""
            out.append(f"- {t}★ {W.get(m.get('kind'), W['star'])}{lab}{q}")
    return out
