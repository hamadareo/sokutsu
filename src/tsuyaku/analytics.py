"""話者の分析(W4 11/07): 話した時間・割合・ターン(発言の回数)・いちばん長い連続の発言・質問の数・
時間の帯(誰がいつ話したか)。

時間の取り方:
  - 11/07 以降の記録は各発言に t0(話し始めの壁時計)と dur(声の区間の長さ、秒)がある
    (__main__ の send が区切りの lo/hi から入れる)。これをそのまま使う。
  - それより前の記録には無いので、文字数から長さを見積もる(言語ごとの話す速さ: ja 7.5 / zh 4.5 /
    ko 6.5 / en 14 字/秒。FLEURS の読み上げのおおよその値)。終わりは記録した時刻(ts)の
    LAG_S 秒前(聞き取り・訳のあいだ記録が遅れる)とする。この時は estimated=True を返し、画面は
    「約」を付ける。
"""
from __future__ import annotations

import re
import time

RATE = {"ja": 7.5, "zh": 4.5, "ko": 6.5, "en": 14.0}
LAG_S = 1.0
MERGE_GAP_S = 1.5        # 同じ人の発言のあいだがこれ未満なら時間の帯では1本にする
_Q = re.compile(r"[?？]\s*$|(吗|呢|까|니|ですか|ますか)[。．.!！]?\s*$")


def _start_of(meta: dict) -> "float | None":
    if meta.get("ts0"):
        try:
            return float(meta["ts0"])
        except (TypeError, ValueError):
            pass
    try:
        return time.mktime(time.strptime(meta.get("start", ""), "%Y-%m-%d %H:%M:%S"))
    except (ValueError, TypeError):
        return None


def _wall_of_t(t: str, t0: "float | None") -> "float | None":
    if not t or t0 is None:
        return None
    try:
        h, m, s = (int(x) for x in str(t).split(":")[:3])
    except ValueError:
        return None
    day = time.localtime(t0)
    w = time.mktime((day.tm_year, day.tm_mon, day.tm_mday, h, m, s, 0, 0, -1))
    return w + 86400 if w < t0 - 60 else w


def est_dur(text: str, lang: str) -> float:
    n = len(re.sub(r"\s+", " ", str(text or "")).strip())
    return max(0.6, min(30.0, n / RATE.get(lang, 8.0)))


def timed_lines(meeting: dict) -> tuple[list, bool, "float | None"]:
    """各発言の (話者, 始まり, 終わり)(会議の始まりからの秒)。見積もったか、会議の始まり(壁時計)。"""
    meta, lines = meeting.get("meta") or {}, meeting.get("lines") or []
    t0 = _start_of(meta)
    raw = []
    est = False
    for i, ln in enumerate(lines):
        if ln.get("t0") is not None and ln.get("dur") is not None:
            a = float(ln["t0"])
            b = a + max(0.0, float(ln["dur"]))
        else:
            est = True
            end = float(ln["ts"]) if ln.get("ts") else _wall_of_t(ln.get("t", ""), t0)
            d = est_dur(ln.get("orig", ""), ln.get("lang", ""))
            if end is None:
                end = (raw[-1][2] + d + 0.5) if raw else ((t0 or 0.0) + d)
            else:
                end -= LAG_S
            a, b = end - d, end
        raw.append((ln.get("spk") or "", a, b, i))
    if not raw:
        return [], est, t0
    base = t0 if t0 is not None else min(a for _s, a, _b, _i in raw)
    base = min(base, min(a for _s, a, _b, _i in raw))
    return [(s, round(a - base, 2), round(b - base, 2), i) for s, a, b, i in raw], est, base


def speaker_stats(meeting: dict) -> dict:
    """{speakers: [{id, secs, share, turns, lines, longest, longest_i, longest_at, questions}],
    total, span, estimated, timeline: [[話者, 始まり, 終わり, 最初の行], …]}。speakers は長く話した順。"""
    tl, est, _base = timed_lines(meeting)
    lines = meeting.get("lines") or []
    sp: dict = {}
    runs: list = []          # [話者, 合計秒, 始まりの行, 始まりの秒]
    for spk, a, b, i in tl:
        d = max(0.0, b - a)
        s = sp.setdefault(spk, {"id": spk, "secs": 0.0, "turns": 0, "lines": 0, "longest": 0.0,
                                "longest_i": i, "longest_at": a, "questions": 0})
        s["secs"] += d
        s["lines"] += 1
        if _Q.search(str(lines[i].get("orig", ""))):
            s["questions"] += 1
        if runs and runs[-1][0] == spk:
            runs[-1][1] += d
        else:
            runs.append([spk, d, i, a])
            s["turns"] += 1
    for spk, d, i, a in runs:
        s = sp[spk]
        if d > s["longest"]:
            s["longest"], s["longest_i"], s["longest_at"] = d, i, a
    total = sum(s["secs"] for s in sp.values())
    for s in sp.values():
        s["share"] = round(s["secs"] / total, 4) if total else 0.0
        s["secs"] = round(s["secs"], 1)
        s["longest"] = round(s["longest"], 1)
    band: list = []
    for spk, a, b, i in tl:
        if band and band[-1][0] == spk and a - band[-1][2] < MERGE_GAP_S:
            band[-1][2] = max(band[-1][2], b)
        else:
            band.append([spk, a, b, i])
    return {"speakers": sorted(sp.values(), key=lambda s: -s["secs"]),
            "total": round(total, 1), "span": round(max((b for _s, _a, b, _i in tl), default=0.0), 1),
            "estimated": est, "timeline": [[s, round(a, 1), round(b, 1), i] for s, a, b, i in band]}
