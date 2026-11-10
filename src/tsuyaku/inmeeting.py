"""会議中の道具(W3, 11/07「他の会議メモアプリの機能は全て」)。

  - ブックマーク(印): 会議中にボタンか B キーで「ここ大事」。種類(重要/決定/やること/質問)と
    ひとことを後から付けられる。                         → 記録に mark 行
  - 自分のメモ: 会議中に打った短いメモ(Granola のように AI のメモに織り込む元)。 → memo 行
  - 議題: 起動画面で入れた議題(meta の agenda)を、会議中にチェックしていく。     → agenda 行
  - 書き起こしの修正: 会議の記録で、行の文を直す・話者を付け替える・元に戻す。
                                                        → edit / respk / undo 行
  - 用語辞書: 人名・専門用語を覚えておき、毎回の聞き取りの手がかり(whisper の
    initial_prompt。テーマ欄と合わせる)と、会議メモの綴りの手がかりにする。 → terms.json

記録の行の形は docs/superpowers/specs/2026-11-07-meeting-features-parity.md の
「記録のデータの約束」。読む側は notes.load_meeting(同じ id の後の行で上書き・del で消す・
undo で直近の修正を取り消す)。記録は追記だけで、元の行は書き換えない(壊れても戻せる)。

Bridge はこの InMeetingMixin を継ぎ、画面(JS)から pywebview 経由で呼ばれる。
"""
from __future__ import annotations

import json
import time
import uuid
from pathlib import Path

from . import notes as notes_mod
from .config import DATA

META_KEYS = ("agenda", "template")   # 画面から meta に入れてよいもの(template は W1)
MAX_TERMS = 300
MAX_TERM_LEN = 40
MAX_LABEL = 200
MAX_MEMO = 2000
MAX_AGENDA = 3000
_TERM_SEP = ("、", ",", "，", "\n", ";", "；", "・", "/")


# ---------------- 用語辞書 ----------------

def terms_path() -> Path:
    """config.json の隣(config.json と同じく git に入れない。配る版は Application Support)。"""
    return DATA / "terms.json"


def normalize_terms(items) -> list[str]:
    """前後の空白を取り、区切り(、・, 改行…)で分け、同じ語(大文字小文字を問わない)は最初だけ。
    長すぎる語(貼り付けの事故)は捨てる。"""
    out: list[str] = []
    seen: set = set()
    for x in items or []:
        if not isinstance(x, str):
            continue
        parts = [x]
        for sep in _TERM_SEP:
            parts = [q for p in parts for q in p.split(sep)]
        for t in parts:
            t = t.strip(" 　。.\t")
            if not t or len(t) > MAX_TERM_LEN or t.lower() in seen:
                continue
            seen.add(t.lower())
            out.append(t)
    return out[:MAX_TERMS]


def load_terms(path: "Path | None" = None) -> list[str]:
    p = Path(path) if path else terms_path()
    try:
        data = json.loads(p.read_text(encoding="utf-8")) if p.exists() else []
    except (ValueError, OSError):
        return []
    if isinstance(data, dict):
        data = data.get("terms") or []
    return normalize_terms(data if isinstance(data, list) else [])


def save_terms(items, path: "Path | None" = None) -> list[str]:
    p = Path(path) if path else terms_path()
    terms = normalize_terms(items)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps({"terms": terms}, ensure_ascii=False, indent=1), encoding="utf-8")
    tmp.replace(p)   # 書きかけで落ちても辞書が空にならない
    return terms


# ---------------- 会議の開始時の meta ----------------

def meta_extra(extra) -> dict:
    """画面から来た dict → meta に入れる分(許したキーだけ・文字列だけ・長さを抑える)。"""
    out: dict = {}
    if not isinstance(extra, dict):
        return out
    for k in META_KEYS:
        v = extra.get(k)
        if isinstance(v, str) and v.strip():
            out[k] = v.strip()[:MAX_AGENDA]
    return out


# ---------------- 記録に行を足す ----------------

def new_id() -> str:
    return uuid.uuid4().hex[:10]


def append_row(path: Path, row: dict) -> None:
    """会議中でない記録に1行足す。落ちて書きかけの最終行があれば、改行してから書く。"""
    p = Path(path)
    with p.open("rb") as f:
        try:
            f.seek(-1, 2)
            need_nl = f.read(1) != b"\n"
        except OSError:      # 空のファイル
            need_nl = False
    with p.open("a", encoding="utf-8") as f:
        f.write(("\n" if need_nl else "") + json.dumps(row, ensure_ascii=False) + "\n")


def _clip(s, n: int) -> str:
    """1行に(改行・連続の空白を1つに)して n 字まで。"""
    return " ".join(str(s or "").split())[:n]


class InMeetingMixin:
    """Bridge に混ぜる: 画面から呼ばれる会議中の道具。mid は "live"(今の会話)か記録の id。"""

    # ---------- 共通 ----------

    def _w3_target(self, mid) -> "tuple[Path, object] | None":
        """(記録のパス, 開いている SessionLog か None)。今の会話の記録は開いている log から書く
        (同じファイルを2つの口から書くと行が混ざる)。"""
        p = self._meeting_path(mid)
        if p is None:
            return None
        log = getattr(self, "log", None)
        if log is not None and Path(log.path_jsonl) == p:
            return p, log
        return p, None

    def _w3_write(self, target, row: dict) -> None:
        p, log = target
        if log is not None:
            log.add_row(row)
        else:
            append_row(p, row)

    @staticmethod
    def _w3_index(i, n: int) -> "int | None":
        if isinstance(i, bool):
            return None
        try:
            i = int(i)
        except (TypeError, ValueError):
            return None
        return i if 0 <= i < n else None

    def _w3_when(self, target, meeting: "dict | None", i) -> tuple:
        """印・メモの時刻: 会話中は今。終わった会議は、指した行の時刻(無ければ記録の最後)。"""
        if target[1] is not None:
            return round(time.time(), 2), time.strftime("%H:%M:%S")
        lines = (meeting or {}).get("lines") or []
        if lines:
            ln = lines[i if isinstance(i, int) and 0 <= i < len(lines) else -1]
            return ln.get("ts"), ln.get("t", "")
        return None, ""

    # ---------- ブックマーク ----------

    def mark_set(self, mid, mark=None):
        """印を付ける(id 無し)か、種類・ひとことを変える(id あり)。付けた/変えた印を返す。
        mark = {id?, i?, kind?, label?}。i は画面で指した行(途中字幕の間は次の行)。"""
        tg = self._w3_target(mid)
        if tg is None or not isinstance(mark, dict):
            return None
        m = notes_mod.load_meeting(tg[0])
        n = len(m["lines"])
        row: dict = {"type": "mark"}
        mid_ = mark.get("id")
        if isinstance(mid_, str) and mid_:
            if mid_ not in {x["id"] for x in m["marks"]}:
                return None
            row["id"] = mid_
        else:
            row["id"] = new_id()
            i = mark.get("i")
            i = i if isinstance(i, int) and not isinstance(i, bool) else n - 1
            # 会話中は「今話している最中の次の行」(= n)まで指せる。終わった会議は今ある行だけ
            row["i"] = max(0, min(i, n if tg[1] is not None else n - 1))
            row["ts"], row["t"] = self._w3_when(tg, m, row["i"])
            row["kind"] = "star"
            row["label"] = ""
        if "kind" in mark:
            row["kind"] = mark["kind"] if mark["kind"] in notes_mod.MARK_KINDS else "star"
        if "label" in mark:
            row["label"] = _clip(mark["label"], MAX_LABEL)
        self._w3_write(tg, row)
        got = [x for x in notes_mod.load_meeting(tg[0])["marks"] if x["id"] == row["id"]]
        if got:
            return got[0]
        # 会話中で、指した行(次の発言)がまだ記録に無い: 書いたとおりに返す
        return {"id": row["id"], "ts": row.get("ts"), "t": row.get("t", ""), "kind": row.get("kind", "star"),
                "label": row.get("label", ""), "i": row.get("i", 0)}

    def mark_del(self, mid, mark_id):
        tg = self._w3_target(mid)
        if tg is None or not isinstance(mark_id, str) or not mark_id:
            return "none"
        self._w3_write(tg, {"type": "del", "id": mark_id})
        return "ok"

    # ---------- 自分のメモ ----------

    def memo_set(self, mid, memo=None):
        """メモを足す(id 無し)か、直す(id あり)。空のメモは足さない(消すのは memo_del)。"""
        tg = self._w3_target(mid)
        if tg is None or not isinstance(memo, dict):
            return None
        text = str(memo.get("text") or "").strip()[:MAX_MEMO]
        if not text:
            return None
        m = notes_mod.load_meeting(tg[0])
        row: dict = {"type": "memo", "text": text}
        mid_ = memo.get("id")
        if isinstance(mid_, str) and mid_:
            if mid_ not in {x["id"] for x in m["memos"]}:
                return None
            row["id"] = mid_
        else:
            row["id"] = new_id()
            n = len(m["lines"])
            i = memo.get("i")
            row["i"] = max(0, min(i if isinstance(i, int) and not isinstance(i, bool) else n - 1, max(n - 1, 0)))
            row["ts"], row["t"] = self._w3_when(tg, m, row["i"])
        self._w3_write(tg, row)
        got = [x for x in notes_mod.load_meeting(tg[0])["memos"] if x["id"] == row["id"]]
        return got[0] if got else None

    def memo_del(self, mid, memo_id):
        return self.mark_del(mid, memo_id)

    # ---------- 議題 ----------

    def agenda_check(self, mid, k, done):
        tg = self._w3_target(mid)
        if tg is None:
            return "none"
        m = notes_mod.load_meeting(tg[0])
        k = self._w3_index(k, len(m["agenda"]))
        if k is None:
            return "none"
        row = {"type": "agenda", "k": k, "done": bool(done)}
        if tg[1] is not None:   # 会議中に済ませた時刻(章の区切りになる)。終わってからの印は時刻なし
            row["ts"], row["t"] = round(time.time(), 2), time.strftime("%H:%M:%S")
        self._w3_write(tg, row)
        return "ok"

    # ---------- 書き起こしの修正 ----------

    def line_edit(self, mid, i, text):
        """発言 i の文を直す。直した行を返す(直せなければ None)。"""
        tg = self._w3_target(mid)
        if tg is None:
            return None
        m = notes_mod.load_meeting(tg[0])
        i = self._w3_index(i, len(m["lines"]))
        text = str(text or "").strip()[:MAX_MEMO]
        if i is None or not text:
            return None
        if text == m["lines"][i]["orig"]:
            return m["lines"][i]
        self._w3_write(tg, {"type": "edit", "i": i, "orig": text, "ts": round(time.time(), 2)})
        if tg[1] is not None:   # 会話中: 裏の聞き直し(relisten.py)の結果でこの行を上書きしない
            self.__dict__.setdefault("_human_edited", set()).add(i)
        return notes_mod.load_meeting(tg[0])["lines"][i]

    def line_speaker(self, mid, i, spk):
        """発言 i の話者を付け替える(既にいる話者か、新しい話者 S<n>)。"""
        tg = self._w3_target(mid)
        if tg is None or not isinstance(spk, str) or not notes_mod.SPK_RE.fullmatch(spk):
            return None
        m = notes_mod.load_meeting(tg[0])
        i = self._w3_index(i, len(m["lines"]))
        if i is None:
            return None
        if spk == m["lines"][i]["spk"]:
            return m["lines"][i]
        self._w3_write(tg, {"type": "respk", "i": i, "spk": spk, "ts": round(time.time(), 2)})
        return notes_mod.load_meeting(tg[0])["lines"][i]

    def edit_undo(self, mid):
        """いちばん新しい修正(文・話者)を取り消す。取り消した行 {i, line} か None。"""
        tg = self._w3_target(mid)
        if tg is None:
            return None
        before = notes_mod.load_meeting(tg[0])
        if not before["undo"]:
            return None
        self._w3_write(tg, {"type": "undo", "ts": round(time.time(), 2)})
        after = notes_mod.load_meeting(tg[0])
        for k, (a, b) in enumerate(zip(before["lines"], after["lines"])):
            if (a["orig"], a["spk"]) != (b["orig"], b["spk"]):
                return {"i": k, "line": b}
        return {"i": -1, "line": None}

    # ---------- 用語辞書 ----------

    def terms_get(self):
        return load_terms(terms_path())

    def terms_set(self, items):
        if isinstance(items, str):
            items = [items]
        return save_terms(items if isinstance(items, list) else [], terms_path())
