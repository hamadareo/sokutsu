"""全会議の横断検索(W4 11/07): 書き起こし(原文・訳)・会議メモ・題・タグ。

作り方:
  - logs/.index/search.sqlite3 に sqlite の FTS5(tokenize=trigram)の索引を置く。trigram は
    3文字ずつの切れ端で引くので、日本語・中国語・韓国語のように単語の区切りが無い文でも
    そのまま当たる(分かち書きの辞書が要らない)。英字は大文字小文字を区別しない。
  - 3文字以上の語は MATCH '"語"'(索引を引く。20万行で数ミリ秒)。2文字以下の語(「予算」など
    日本語ではよくある)は trigram の索引では引けない(何も当たらない)ので instr で全件を順に見る
    (20万行で約50ミリ秒、英字は大文字小文字をそろえるので約200ミリ秒。長い語があればその結果の中だけ)。
    LIKE は ESCAPE を付けると索引を使わず、付けないと % _ が特別な文字になるので使わない。
  - 索引は検索のたびに、会議ごとのファイルの署名(更新時刻と大きさ)を比べて、変わった会議だけ
    入れ直す(会話中の記録・メモの作り直し・題の変更・ゴミ箱もそのまま追いつく)。
  - 複数の語(空白区切り、全角の空白も)は「同じ1行に全部ある」で絞る。
  - ヒットの抜粋と、語の位置(marks)を返す。画面はそれを使って当たった語に印を付け、
    その会議のその行へ飛ぶ。
"""
from __future__ import annotations

import json
import os
import re
import sqlite3
import threading
import time
from pathlib import Path

from . import library as lib_mod
from . import notes as notes_mod

SNIP = 90           # 抜粋のおおよその長さ(字)
BEFORE = 28         # 当たった語の前に残す字数
_SECS = ("title", "agenda", "summary", "topics", "decisions", "todos", "questions", "keywords")
_locks: dict = {}
_glock = threading.Lock()


def _lock_for(path: str) -> threading.Lock:
    with _glock:
        return _locks.setdefault(path, threading.Lock())


def _phrase(term: str) -> str:
    return '"' + term.replace('"', '""') + '"'


def terms_of(q: str) -> list[str]:
    return [t for t in re.split(r"[\s　]+", str(q or "").strip()) if t][:8]


def _snippet(text: str, terms: list[str]) -> tuple[str, list]:
    low = text.lower()
    first = min((p for p in (low.find(t.lower()) for t in terms) if p >= 0), default=0)
    a = max(0, first - BEFORE)
    b = min(len(text), a + SNIP)
    if b - a < SNIP:
        a = max(0, b - SNIP)
    snip = text[a:b].replace("\n", " / ")
    pre = "…" if a > 0 else ""
    snip = pre + snip + ("…" if b < len(text) else "")
    marks = []
    sl = snip.lower()
    for t in terms:
        tl = t.lower()
        k = sl.find(tl)
        while k >= 0 and tl:
            marks.append([k, k + len(t)])
            k = sl.find(tl, k + len(tl))
    marks.sort()
    merged: list = []
    for m in marks:
        if merged and m[0] <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], m[1])
        else:
            merged.append(m)
    return snip, merged


class SearchIndex:
    """log_dir の会議を索引する。どのスレッドからでも呼べる(呼ぶたびに接続を開く)。"""

    def __init__(self, log_dir: "str | Path", db_path: "str | Path | None" = None):
        self.log_dir = Path(log_dir)
        self.db_path = Path(db_path) if db_path else self.log_dir / ".index" / "search.sqlite3"
        self._lock = _lock_for(str(self.db_path))

    # ---------- 索引 ----------

    def _connect(self) -> sqlite3.Connection:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        c = sqlite3.connect(str(self.db_path), timeout=10)
        c.execute("PRAGMA journal_mode=WAL")
        c.execute("CREATE TABLE IF NOT EXISTS files(mid TEXT PRIMARY KEY, sig TEXT, start TEXT)")
        c.execute("CREATE VIRTUAL TABLE IF NOT EXISTS docs USING fts5(text, mid UNINDEXED, kind UNINDEXED,"
                  " sec UNINDEXED, i UNINDEXED, t UNINDEXED, spk UNINDEXED, tokenize='trigram')")
        return c

    def _scan(self) -> dict:
        """ログのフォルダを1回だけ見て、会議ごとの署名(関係するファイルの名前・更新時刻・大きさ)。"""
        files: dict = {}
        has_log: set = set()
        try:
            it = list(os.scandir(self.log_dir))
        except OSError:
            return {}
        for e in it:
            n = e.name
            if not n.startswith("session-") or not e.is_file():
                continue
            stem, _, rest = n.partition(".")
            if rest == "jsonl" or rest == "tags.json" or (rest.startswith("notes") and rest.endswith(".json")):
                st = e.stat()
                if rest == "jsonl":
                    has_log.add(stem)
                files.setdefault(stem, []).append(f"{n}:{st.st_mtime_ns}:{st.st_size}")
        return {stem: "|".join(sorted(v)) for stem, v in files.items() if stem in has_log}

    def _rows(self, p: Path) -> tuple[str, list]:
        m = notes_mod.load_meeting(p)
        mid = m["id"]
        rows = []
        for i, ln in enumerate(m["lines"]):
            txt = ln.get("orig", "")
            tr = ln.get("trans") or ""
            if tr and not tr.startswith("（"):
                txt += "\n" + tr
            rows.append((txt, mid, "line", "", i, ln.get("t", ""), ln.get("spk", "")))
        # 会議中の道具(W3): 自分のメモ・印のひとこと(指す行へ飛ぶ)・議題
        for mm in m.get("memos") or []:
            rows.append((mm.get("text", ""), mid, "memo", "memos", mm.get("i") or 0, mm.get("t", ""), ""))
        for mk in m.get("marks") or []:
            if mk.get("label"):
                rows.append((mk["label"], mid, "mark", mk.get("kind", ""), mk.get("i") or 0, mk.get("t", ""), ""))
        for k, ag in enumerate(m.get("agenda") or []):
            if ag.get("text"):
                rows.append((ag["text"], mid, "note", "agenda", k, "", ""))
        info = lib_mod.load_info(p)
        head = " ".join([info.get("title", "")] + ["#" + t for t in info.get("tags", [])]).strip()
        if head:
            rows.append((head, mid, "title", "title", 0, "", ""))
        seen = set()
        for lg in notes_mod.notes_langs(p) or [None]:
            n = notes_mod.load_notes(p, lg)
            if not n:
                continue

            def add(sec, i, text, who=""):
                text = str(text or "").strip()
                if text and (sec, text) not in seen:
                    seen.add((sec, text))
                    rows.append((text, mid, "note", sec, i, "", who))
            add("title", 0, n.get("title"))
            for k in ("summary", "decisions", "questions"):
                for i, x in enumerate(n.get(k) or []):
                    add(k, i, x)
            for i, t in enumerate(n.get("topics") or []):
                add("topics", i, t.get("title"))
                for pt in t.get("points") or []:
                    add("topics", i, pt.get("text"), pt.get("who", ""))
            for i, t in enumerate(n.get("todos") or []):
                add("todos", i, " ".join(x for x in (t.get("task"), t.get("due")) if x), t.get("owner", ""))
            if n.get("keywords"):
                add("keywords", 0, " · ".join(str(k) for k in n["keywords"]))
        return m["meta"].get("start", ""), rows

    def refresh(self) -> dict:
        """変わった会議だけ入れ直す。{added, updated, removed, took_ms}。
        索引のファイルが壊れていたら(sqlite の「データベースでない」「壊れた」)、消して1から作り直す
        (索引は記録とメモから作れるキャッシュ。そのままだと検索がずっと失敗する)。ロック中などの
        一時的な失敗(OperationalError)は消さずにそのまま上げる。"""
        try:
            return self._refresh()
        except sqlite3.DatabaseError as e:
            if isinstance(e, sqlite3.OperationalError):
                raise
            print(f"[search] 索引が壊れていたので作り直します: {e}", flush=True)
            with self._lock:
                for suf in ("", "-wal", "-shm", "-journal"):
                    Path(f"{self.db_path}{suf}").unlink(missing_ok=True)
            return self._refresh()

    def _refresh(self) -> dict:
        t0 = time.perf_counter()
        st = {"added": 0, "updated": 0, "removed": 0}
        with self._lock:
            c = self._connect()
            try:
                have = dict(c.execute("SELECT mid, sig FROM files"))
                now = self._scan()
                for mid in set(have) - set(now):
                    c.execute("DELETE FROM docs WHERE mid=?", (mid,))
                    c.execute("DELETE FROM files WHERE mid=?", (mid,))
                    st["removed"] += 1
                for mid, sig in now.items():
                    if have.get(mid) == sig:
                        continue
                    p = self.log_dir / f"{mid}.jsonl"
                    try:
                        start, rows = self._rows(p)
                    except (OSError, ValueError):
                        continue
                    if mid in have:
                        c.execute("DELETE FROM docs WHERE mid=?", (mid,))
                        st["updated"] += 1
                    else:
                        st["added"] += 1
                    c.executemany("INSERT INTO docs(text, mid, kind, sec, i, t, spk) VALUES (?,?,?,?,?,?,?)", rows)
                    c.execute("INSERT OR REPLACE INTO files(mid, sig, start) VALUES (?,?,?)", (mid, sig, start))
                c.commit()
            finally:
                c.close()
        st["took_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        return st

    # ---------- 検索 ----------

    def search(self, q: str, max_meetings: int = 60, per_meeting: int = 5, max_rows: int = 5000) -> dict:
        """q の語を全部含む行・メモの項目を、会議ごとに(新しい会議から)返す。

        返り値: {q, terms, took_ms, total, meetings: [{id, start, n, hits: [{kind, sec, i, t, spk, snip, marks}]}]}
        kind: "line"(記録の i 行目) / "memo"・"mark"(会議中の自分のメモ・印、i = 指す行) /
              "note"(メモの sec 節の i 番目、議題は sec="agenda") / "title"(手で付けた題・タグ)。
        """
        t0 = time.perf_counter()
        terms = terms_of(q)
        out = {"q": q, "terms": terms, "total": 0, "meetings": []}
        if not terms:
            out["took_ms"] = 0.0
            return out
        self.refresh()
        t1 = time.perf_counter()
        longs = [t for t in terms if len(t) >= 3]
        conds, args = [], []
        if longs:
            conds.append("docs MATCH ?")
            args.append(" AND ".join(_phrase(t) for t in longs))
        for t in terms:
            if len(t) < 3:
                if t.isascii():
                    conds.append("instr(lower(text), ?) > 0")
                    args.append(t.lower())
                else:
                    conds.append("instr(text, ?) > 0")
                    args.append(t)
        with self._lock:
            c = self._connect()
            try:
                starts = dict(c.execute("SELECT mid, start FROM files"))
                rows = c.execute(f"SELECT text, mid, kind, sec, i, t, spk FROM docs WHERE {' AND '.join(conds)}"
                                 " LIMIT ?", [*args, max_rows]).fetchall()
            except sqlite3.OperationalError:
                rows = []
            finally:
                c.close()
        order = {"title": 0, "note": 1, "memo": 2, "mark": 2, "line": 3}
        by: dict = {}
        for text, mid, kind, sec, i, t, spk in rows:
            by.setdefault(mid, []).append((order.get(kind, 3), _SECS.index(sec) if sec in _SECS else 9,
                                           int(i), text, kind, sec, t, spk))
        mids = sorted(by, key=lambda m: (starts.get(m, ""), m), reverse=True)
        for mid in mids[:max_meetings]:
            hs = sorted(by[mid])
            hits = []
            for _o, _s, i, text, kind, sec, t, spk in hs[:per_meeting]:
                snip, marks = _snippet(text, terms)
                hits.append({"kind": kind, "sec": sec, "i": i, "t": t, "spk": spk, "snip": snip, "marks": marks})
            out["meetings"].append({"id": mid, "start": starts.get(mid, ""), "n": len(hs), "hits": hits})
        out["total"] = sum(len(v) for v in by.values())
        out["more"] = len(mids) > max_meetings or len(rows) >= max_rows
        out["took_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        out["query_ms"] = round((time.perf_counter() - t1) * 1000, 1)
        return out


def _bench(n_meetings: int = 500, n_lines: int = 400, out_dir: "str | None" = None) -> dict:
    """数百件の会議で速さを測る(python -m tsuyaku.search で実行)。"""
    import random
    import tempfile
    rnd = random.Random(0)
    words_ja = ["予算", "見積もり", "来週", "会議", "資料", "共有", "確認", "設計", "評価", "実験", "締め切り",
                "担当", "顧客", "提案", "議題", "スライド", "データ", "モデル", "結果", "発表"]
    words_en = ["budget", "quotation", "deadline", "design", "review", "customer", "proposal", "slides",
                "experiment", "result", "schedule", "meeting"]
    d = Path(out_dir or tempfile.mkdtemp(prefix="tsuyaku-bench-"))
    for k in range(n_meetings):
        stem = f"session-2026{(k % 12) + 1:02d}{(k % 28) + 1:02d}-{k:06d}"
        with (d / f"{stem}.jsonl").open("w", encoding="utf-8") as f:
            f.write(json.dumps({"type": "meta", "pair": "ja-en", "start": f"2026-01-01 00:00:{k % 60:02d}"}) + "\n")
            for i in range(n_lines):
                ja = "".join(rnd.choice(words_ja) + rnd.choice(["を", "の", "は", "で"]) for _ in range(6)) + "。"
                en = " ".join(rnd.choice(words_en) for _ in range(10)) + "."
                f.write(json.dumps({"type": "utter", "t": "10:00:00", "lang": "ja", "orig": ja, "trans": en,
                                    "spk": f"S{i % 4 + 1}"}, ensure_ascii=False) + "\n")
        if k == n_meetings // 2:
            with (d / f"{stem}.jsonl").open("a", encoding="utf-8") as f:
                f.write(json.dumps({"type": "utter", "t": "10:00:00", "lang": "ja", "orig": "珍しい合言葉ポンカン",
                                    "spk": "S1"}, ensure_ascii=False) + "\n")
    ix = SearchIndex(d)
    res = {"meetings": n_meetings, "lines": n_meetings * n_lines}
    t = time.perf_counter()
    res["build"] = ix.refresh()
    res["build_s"] = round(time.perf_counter() - t, 2)
    res["noop_refresh_ms"] = ix.refresh()["took_ms"]
    for q in ("ポンカン", "合言葉ポンカン", "予算", "budget deadline", "見積もり 来週", "存在しない語句"):
        ts = []
        for _ in range(3):
            r = ix.search(q)
            ts.append(r["took_ms"])
        res[f"q:{q}"] = {"ms_median": sorted(ts)[1], "total": r["total"], "meetings": len(r["meetings"])}
    with (d / sorted(p.name for p in d.glob("session-*.jsonl"))[0]).open("a", encoding="utf-8") as f:
        f.write(json.dumps({"type": "utter", "t": "10:00:00", "lang": "ja", "orig": "追記", "spk": "S1"}) + "\n")
    res["one_meeting_changed_ms"] = ix.refresh()["took_ms"]
    res["db_mb"] = round(ix.db_path.stat().st_size / 1e6, 1)
    return res


if __name__ == "__main__":
    import sys
    print(json.dumps(_bench(*(int(x) for x in sys.argv[1:3])), ensure_ascii=False, indent=1))
