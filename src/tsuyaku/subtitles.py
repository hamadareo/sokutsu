"""字幕(SRT / WebVTT)と時刻つきテキストの書き出し(W2, 11/07)。

時刻は録音の中の秒(記録の a0/a1)。録音が無い会議は、壁時計(ts = その発言を書いた時刻)から
会議の始まりを引いて近似する(発言の長さは字数から見積もる)。話者は名前(直した名前 → メモで
分かった名前 → 「話者2」)。動画編集ソフト・YouTube・VLC にそのまま読ませられる形にする。
"""
from __future__ import annotations

from . import notes as notes_mod

MIN_CUE_S = 0.5


def speaker_label(spk: str, names: dict, ui: str = "ja") -> str:
    if not spk:
        return ""
    return names.get(spk) or notes_mod.default_name(spk, ui)


def _est_len(text: str) -> float:
    """録音が無いときの発言の長さの見積もり(日本語・中国語は1秒7字、英語などは1秒15字)。"""
    t = text or ""
    cjk = sum(1 for ch in t if ord(ch) > 0x2E80)
    return max(1.0, min(12.0, cjk / 7.0 + (len(t) - cjk) / 15.0))


def cues(lines: list, names: dict, ui: str = "ja", t0: "float | None" = None,
         with_trans: bool = True) -> list:
    """[(始め秒, 終わり秒, 話者名, 原文, 訳)]。始めは前の字幕より前にならず、重なった前の字幕は縮める。"""
    out: list = []
    if t0 is None:
        tss = [ln.get("ts") for ln in lines if ln.get("ts")]
        t0 = min(tss) - _est_len(lines[0].get("orig", "")) if tss else 0.0
    prev_end = 0.0
    for ln in lines:
        text = (ln.get("orig") or "").strip()
        if not text:
            continue
        if ln.get("a0") is not None:
            a0, a1 = float(ln["a0"]), float(ln.get("a1") or ln["a0"])
        elif ln.get("ts"):
            a1 = max(0.0, float(ln["ts"]) - t0)
            a0 = max(prev_end, a1 - _est_len(text))
        else:
            a0 = prev_end
            a1 = a0 + _est_len(text)
        if out and a0 < out[-1][0]:
            a0 = out[-1][0]
        if out and a0 < out[-1][1]:   # 重なり: 前の字幕をこの字幕の始めで終える(短くしすぎない)
            p = out[-1]
            end = max(a0, p[0] + MIN_CUE_S)
            out[-1] = (p[0], end, *p[2:])
            a0 = max(a0, end)
        a1 = max(a1, a0 + MIN_CUE_S)
        trans = (ln.get("trans") or "").strip() if with_trans else ""
        if trans.startswith("（翻訳できませんでした）"):
            trans = ""
        out.append((a0, a1, speaker_label(ln.get("spk") or "", names, ui), text, trans))
        prev_end = a1
    return out


def _ts(sec: float, sep: str) -> str:
    ms = int(round(max(0.0, sec) * 1000))
    h, ms = divmod(ms, 3600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d}{sep}{ms:03d}"


def _clean(t: str) -> str:
    """字幕の書式を壊す文字を避ける(--> と空行)。"""
    return " ".join((t or "").replace("-->", "→").split())


def to_srt(lines: list, names: dict, ui: str = "ja", with_trans: bool = True,
           t0: "float | None" = None) -> str:
    blocks = []
    for i, (a0, a1, who, text, trans) in enumerate(cues(lines, names, ui, t0, with_trans), 1):
        body = (f"{who}: " if who else "") + _clean(text)
        if trans:
            body += "\n" + _clean(trans)
        blocks.append(f"{i}\n{_ts(a0, ',')} --> {_ts(a1, ',')}\n{body}")
    return "\n\n".join(blocks) + "\n"


def _esc(t: str) -> str:
    return t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def to_vtt(lines: list, names: dict, ui: str = "ja", with_trans: bool = True,
           t0: "float | None" = None) -> str:
    out = ["WEBVTT", ""]
    for a0, a1, who, text, trans in cues(lines, names, ui, t0, with_trans):
        body = (f"<v {_esc(who)}>" if who else "") + _esc(_clean(text))
        if trans:
            body += "\n" + _esc(_clean(trans))
        out += [f"{_ts(a0, '.')} --> {_ts(a1, '.')}", body, ""]
    return "\n".join(out)


def to_text(lines: list, names: dict, ui: str = "ja", with_trans: bool = True,
            t0: "float | None" = None, title: str = "") -> str:
    out = [title, ""] if title else []
    for a0, _a1, who, text, trans in cues(lines, names, ui, t0, with_trans):
        out.append(f"[{_ts(a0, ',')[:8]}] " + (f"{who}: " if who else "") + text)
        if trans:
            out.append("    " + trans)
    return "\n".join(out) + "\n"


def meeting_names(jsonl, ui: str = "ja") -> dict:
    """話者の名前: メモで分かった名前(どの言語のメモでも)に、画面で直した名前を上書き。"""
    names: dict = {}
    for lg in notes_mod.notes_langs(jsonl) or [None]:
        n = notes_mod.load_notes(jsonl, lg) or {}
        for q in n.get("participants", []):
            if q.get("name") and q.get("id"):
                names.setdefault(q["id"], q["name"])
    for k, v in notes_mod.load_speakers(jsonl).items():
        if v:
            names[k] = v
    return names


def export(jsonl, fmt: str, ui: str = "ja", with_trans: bool = True) -> tuple:
    """(本文, 拡張子, 題)。fmt: srt / vtt / txt。"""
    m = notes_mod.load_meeting(jsonl)
    meta = m["meta"]
    names = meeting_names(jsonl, ui)
    n = None
    for lg in [ui] + list(notes_mod.LANGS):
        n = notes_mod.load_notes(jsonl, lg)
        if n:
            break
    title = (n or {}).get("title") or meta.get("file") or m["id"]
    t0 = meta.get("ts0")
    if fmt == "srt":
        return to_srt(m["lines"], names, ui, with_trans, t0), "srt", title
    if fmt == "vtt":
        return to_vtt(m["lines"], names, ui, with_trans, t0), "vtt", title
    head = title + ("  " + meta.get("start", "") if meta.get("start") else "")
    return to_text(m["lines"], names, ui, with_trans, t0, title=head), "txt", title
