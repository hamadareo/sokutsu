"""会議メモの書き出し(W4 11/07): PDF と Word(.docx)。新しい依存は入れない。

  - 中身は build_doc で1つの「文書の形」(題・プロパティ・ブロックの列)にしてから、PDF と docx の
    それぞれへ組む(どちらも同じ並び・同じ言葉になる)。
  - PDF: macOS の AppKit の文字組み(NSLayoutManager)でページに流し、CGPDFContext に描く。文字は
    本物の文字として入る(検索・コピーできる)。足りない字形(日本語の書体に無い簡体字・ハングル)は
    AppKit が自動で別の書体に切り替える。A4、ページ番号つき。AppKit は GUI のスレッドを嫌うので、
    アプリからは子プロセス(python -m tsuyaku.export)で作る(pdf_subprocess)。
    比べた他の道: WKWebView の createPDF は1枚の長いページになる、印刷パネルは利用者の操作が要る、
    textutil は PDF を作れない、cupsfilter は HTML を読めない、reportlab は新しい依存で日本語の書体の
    埋め込みも要る。
  - docx: OOXML を直接書く(python-docx は入っていない)。Word の見出しスタイル(Heading1/2)・
    箇条書き(numbering.xml)・表(プロパティと話した時間)。textutil で読み戻せることをテストで確かめる。
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
import time
import zipfile
from io import BytesIO
from pathlib import Path
from xml.sax.saxutils import escape

from . import notes as notes_mod

SPK_PAPER = ["#4453C9", "#C23F75", "#4F7A12", "#7444C2", "#56606B", "#9C3A93", "#2F6FB7", "#7A7420"]
EXTRA = {
    "ja": {"kicker": "会議メモ", "date": "日時", "length": "長さ", "langs": "言語", "tags": "タグ",
           "folder": "フォルダ", "made": "メモの作成", "talk": "話した時間", "speaker": "話者",
           "time": "時間", "share": "割合", "turns": "ターン", "longest": "最長の連続", "approx": "約",
           "lines": "{n}件の発言", "nonotes": "この会議のメモはまだありません(記録のみ)。",
           "page": "{i} / {n}", "estimated": "時間は文字数からの見積もりです。", "sec": "秒", "min": "分"},
    "en": {"kicker": "MEETING NOTES", "date": "Date", "length": "Length", "langs": "Languages", "tags": "Tags",
           "folder": "Folder", "made": "Notes made", "talk": "Talk time", "speaker": "Speaker",
           "time": "Time", "share": "Share", "turns": "Turns", "longest": "Longest", "approx": "~",
           "lines": "{n} utterances", "nonotes": "No notes yet for this meeting (transcript only).",
           "page": "{i} / {n}", "estimated": "Times are estimated from text length.", "sec": "s", "min": "min"},
    "zh": {"kicker": "会议纪要", "date": "时间", "length": "时长", "langs": "语言", "tags": "标签",
           "folder": "文件夹", "made": "纪要生成", "talk": "发言时间", "speaker": "说话人",
           "time": "时间", "share": "占比", "turns": "轮次", "longest": "最长连续", "approx": "约",
           "lines": "{n}条发言", "nonotes": "此会议还没有纪要(仅记录)。",
           "page": "{i} / {n}", "estimated": "时间根据字数估算。", "sec": "秒", "min": "分钟"},
    "ko": {"kicker": "회의록", "date": "일시", "length": "길이", "langs": "언어", "tags": "태그",
           "folder": "폴더", "made": "회의록 작성", "talk": "발언 시간", "speaker": "화자",
           "time": "시간", "share": "비율", "turns": "턴", "longest": "최장 연속", "approx": "약",
           "lines": "발언 {n}건", "nonotes": "이 회의의 회의록이 아직 없습니다(기록만).",
           "page": "{i} / {n}", "estimated": "시간은 글자 수로 추정했습니다.", "sec": "초", "min": "분"},
}


def _dur(s: float, X: dict) -> str:
    s = int(round(s or 0))
    if s < 60:
        return f"{s}{X['sec']}" if X["sec"] != "s" else f"{s}s"
    m, r = divmod(s, 60)
    return f"{m}:{r:02d}"


def file_name(doc: dict, ext: str) -> str:
    t = re.sub(r'[/\\:*?"<>|\x00-\x1f]', "-", str(doc.get("title") or "meeting")).strip(" .") or "meeting"
    day = str(doc.get("start") or "")[:10]
    return (f"{day} " if day else "") + t[:80] + f".{ext}"


def text_export(notes: "dict | None", meeting: dict, names: "dict | None", fmt: str = "md",
                with_transcript: bool = False) -> str:
    """Markdown(md)かテキスト(txt)の書き出し。画面で直した話者名(names)をメモの参加者より優先する。
    メモが無い会議は、記録(発言)だけを書く。"""
    import copy
    lines = meeting.get("lines") or []
    n = copy.deepcopy(notes) if notes else {}
    n.setdefault("lang", "ja")
    n.setdefault("title", (lines[0]["orig"][:40] if lines else "") or meeting.get("id", ""))
    n.setdefault("start", (meeting.get("meta") or {}).get("start", ""))
    parts = n.setdefault("participants", [])
    have = {p.get("id") for p in parts}
    for spk in dict.fromkeys(ln.get("spk") for ln in lines if ln.get("spk")):
        if spk not in have:
            parts.append({"id": spk, "name": ""})
    for p in parts:
        if (names or {}).get(p.get("id")):
            p["name"] = names[p["id"]]
    md = notes_mod.to_markdown(n, meeting, with_transcript=with_transcript or not notes)
    if fmt != "txt":
        return md
    out = []
    for ln in md.splitlines():
        ln = re.sub(r"^#{1,6}\s+", "", ln)
        ln = ln.replace("**", "").replace("`", "")
        out.append(ln)
    return "\n".join(out) + "\n"


def build_doc(notes: "dict | None", meeting: dict, stats: "dict | None" = None, info: "dict | None" = None,
              lang: "str | None" = None, with_transcript: bool = False, names: "dict | None" = None) -> dict:
    """メモ(無くてもよい)・記録・話者の分析・付属情報 → 書き出しの文書の形。

    names: 画面で直した名前(.speakers.json)。メモの参加者の名前より優先する。"""
    n = notes or {}
    info = info or {}
    ui = lang or n.get("lang") or "ja"
    ui = ui if ui in notes_mod.LANGS else "ja"
    H, X = notes_mod.HEAD[ui], EXTRA[ui]
    L = notes_mod.LANG_LABEL[ui]
    meta = meeting.get("meta") or {}
    lines = meeting.get("lines") or []
    nm: dict = {p["id"]: p.get("name") for p in n.get("participants", []) if p.get("name")}
    nm.update({k: v for k, v in (names or {}).items() if v})

    def who(w: str) -> str:
        if not w:
            return ""
        if w == "all":
            return {"ja": "全員", "en": "Everyone", "zh": "全体", "ko": "전원"}[ui]
        if re.fullmatch(r"S\d+|L\w+", w):
            return nm.get(w) or notes_mod.default_name(w, ui)
        return w

    title = info.get("title") or n.get("title") or (lines[0]["orig"][:40] if lines else "") or meeting.get("id", "")
    start = n.get("start") or meta.get("start", "")
    langs = n.get("languages") or [x for x in str(meta.get("pair", "")).split("-") if x]
    langs = list(dict.fromkeys(langs))
    mins = n.get("minutes")
    pm = [[X["date"], start[:16]]]
    if mins is not None:
        pm.append([X["length"], f"{mins} {H['minutes']}  ·  " + X["lines"].format(n=len(lines))])
    else:
        pm.append([X["length"], X["lines"].format(n=len(lines))])
    if langs:
        pm.append([X["langs"], " / ".join(L.get(x, x) for x in langs)])
    if info.get("tags"):
        pm.append([X["tags"], "  ".join("#" + t for t in info["tags"])])
    if info.get("folder"):
        pm.append([X["folder"], info["folder"]])
    if n.get("made"):
        pm.append([X["made"], str(n["made"])[:16] + (f"  ({n['model']})" if n.get("model") else "")])
    B: list = []
    parts = n.get("participants") or [{"id": k} for k in dict.fromkeys(ln["spk"] for ln in lines if ln.get("spk"))]
    if parts:
        B.append(["h1", H["participants"]])
        ppl = []
        for p in parts:
            extra = [x for x in (p.get("role"), L.get(p.get("lang"), p.get("lang") or "")) if x]
            ppl.append([who(p["id"]), ", ".join(extra)])
        B.append(["people", ppl])
    if n:
        if n.get("summary"):
            B += [["h1", H["summary"]], ["bullets", list(n["summary"])]]
        if n.get("topics"):
            B.append(["h1", H["topics"]])
            for t in n["topics"]:
                B.append(["h2", t.get("title", "")])
                B.append(["points", [[who(pt.get("who", "")), pt.get("text", "")] for pt in t.get("points", [])]])
        B += [["h1", H["decisions"]], ["checks", list(n.get("decisions") or [])] if n.get("decisions")
              else ["para", H["none"]]]
        B.append(["h1", H["todos"]])
        if n.get("todos"):
            rows = []
            for t in n["todos"]:
                m = []
                if t.get("owner"):
                    m.append(f"{H['owner']}: {who(t['owner'])}")
                if t.get("due"):
                    m.append(f"{H['due']}: {t['due']}")
                rows.append([bool(t.get("done")), t.get("task", ""), "  ·  ".join(m)])
            B.append(["todos", rows])
        else:
            B.append(["para", H["none"]])
        if n.get("questions"):
            B += [["h1", H["questions"]], ["questions", list(n["questions"])]]
        if n.get("keywords"):
            B += [["h1", H["keywords"]], ["para", "  ·  ".join(n["keywords"])]]
    else:
        B.append(["note", X["nonotes"]])
    sp = (stats or {}).get("speakers") or []
    if sp:
        B.append(["h1", X["talk"]])
        ap = X["approx"] if stats.get("estimated") else ""
        B.append(["table", {"head": [X["speaker"], X["time"], X["share"], X["turns"], X["longest"]],
                            "rows": [[who(s["id"]), ap + _dur(s["secs"], X), f"{round(s['share'] * 100)}%",
                                      str(s["turns"]), ap + _dur(s["longest"], X)] for s in sp],
                            "bars": [s["share"] for s in sp],
                            "colors": [_color(s["id"]) for s in sp]}])
        if stats.get("estimated"):
            B.append(["note", X["estimated"]])
    if lines and (with_transcript or not n):
        B.append(["h1", H["transcript"]])
        B.append(["transcript", [[str(ln.get("t", ""))[:5], who(ln.get("spk", "")), ln.get("orig", ""),
                                  ln.get("trans", "") if not str(ln.get("trans", "")).startswith("（") else ""]
                                 for ln in lines]])
    return {"title": title, "lang": ui, "kicker": X["kicker"], "start": start, "meta": pm, "blocks": B,
            "page": X["page"]}


def _color(spk: str) -> str:
    m = re.search(r"\d+", spk or "")
    k = int(m.group()) if m else 1
    return SPK_PAPER[(k - 1) % len(SPK_PAPER)]


# ====================================================================== docx

_W = 'xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"'
_BAD_XML = re.compile("[\x00-\x08\x0b\x0c\x0e-\x1f\ufffe\uffff]")


def _x(text) -> str:
    """XML の本文へ: & < > を逃がし、XML 1.0 に書けない制御文字は落とす(1字でもあると Word が開けない)。"""
    return escape(_BAD_XML.sub("", str(text)))


def _run(text: str, bold=False, color=None, size=None, italic=False) -> str:
    rp = ""
    if bold:
        rp += "<w:b/>"
    if italic:
        rp += "<w:i/>"
    if color:
        rp += f'<w:color w:val="{color.lstrip("#")}"/>'
    if size:
        rp += f'<w:sz w:val="{int(size * 2)}"/><w:szCs w:val="{int(size * 2)}"/>'
    out = []
    for i, part in enumerate(str(text).split("\n")):
        if i:
            out.append("<w:br/>")
        out.append(f'<w:t xml:space="preserve">{_x(part)}</w:t>')
    return f"<w:r>{('<w:rPr>' + rp + '</w:rPr>') if rp else ''}{''.join(out)}</w:r>"


def _p(runs: str, style: "str | None" = None, num: bool = False, extra_ppr: str = "") -> str:
    ppr = ""
    if style:
        ppr += f'<w:pStyle w:val="{style}"/>'
    if num:
        ppr += '<w:numPr><w:ilvl w:val="0"/><w:numId w:val="1"/></w:numPr>'
    ppr += extra_ppr
    return f"<w:p>{('<w:pPr>' + ppr + '</w:pPr>') if ppr else ''}{runs}</w:p>"


def _cell(content: str, w: int, shade: "str | None" = None) -> str:
    sh = f'<w:shd w:val="clear" w:color="auto" w:fill="{shade}"/>' if shade else ""
    return f'<w:tc><w:tcPr><w:tcW w:w="{w}" w:type="dxa"/>{sh}</w:tcPr>{content}</w:tc>'


def _table(rows: list, widths: list, head: bool = False) -> str:
    border = ('<w:tblBorders>' + "".join(f'<w:{k} w:val="single" w:sz="4" w:space="0" w:color="DCD6CA"/>'
                                         for k in ("top", "bottom", "insideH")) + "</w:tblBorders>")
    out = [f'<w:tbl><w:tblPr><w:tblW w:w="{sum(widths)}" w:type="dxa"/>{border}<w:tblLayout w:type="fixed"/>'
           '<w:tblCellMar><w:top w:w="60" w:type="dxa"/><w:bottom w:w="60" w:type="dxa"/></w:tblCellMar>'
           '</w:tblPr><w:tblGrid>' + "".join(f'<w:gridCol w:w="{w}"/>' for w in widths) + "</w:tblGrid>"]
    for r, row in enumerate(rows):
        first = head and r == 0
        out.append("<w:tr>" + "".join(_cell(_p(c if c.startswith("<w:r") else _run(c, bold=first, size=9.5 if first else None,
                                                                                       color="636D77" if first else None),
                                              style="Compact"), w, "EFEAE0" if first else None)
                                      for c, w in zip(row, widths)) + "</w:tr>")
    out.append("</w:tbl>")
    return "".join(out)


def _docx_body(doc: dict) -> str:
    b = [_p(_run(doc["kicker"], bold=True, color="8D959D", size=9), "Kicker"), _p(_run(doc["title"]), "Title")]
    b.append(_table([[_run(k, color="8D959D"), _run(v)] for k, v in doc["meta"]], [1900, 7100]))
    for blk in doc["blocks"]:
        k, v = blk[0], blk[1]
        if k == "h1":
            b.append(_p(_run(v), "Heading1"))
        elif k == "h2":
            b.append(_p(_run(v), "Heading2"))
        elif k in ("bullets",):
            b += [_p(_run(x), "ListParagraph", num=True) for x in v]
        elif k == "questions":
            b += [_p(_run("?  ", bold=True, color="636D77") + _run(x), "ListParagraph") for x in v]
        elif k == "checks":
            b += [_p(_run("✓  ", bold=True, color="2F7A4C") + _run(x), "ListParagraph") for x in v]
        elif k == "points":
            for w, t in v:
                b.append(_p((_run(w + ":  ", bold=True) if w else "") + _run(t), "ListParagraph", num=True))
        elif k == "people":
            b += [_p(_run(nm, bold=True) + (_run("   " + d, color="636D77") if d else ""), "ListParagraph", num=True)
                  for nm, d in v]
        elif k == "todos":
            for done, task, m in v:
                b.append(_p(_run("☑  " if done else "☐  ", color="2F7A4C" if done else "636D77")
                            + _run(task, bold=not done, color="8D959D" if done else None)
                            + (_run("\n" + m, color="636D77", size=9.5) if m else ""), "Todo"))
        elif k == "para":
            b.append(_p(_run(v)))
        elif k == "note":
            b.append(_p(_run(v, color="8D959D", italic=True, size=9.5)))
        elif k == "table":
            rows = [v["head"]] + [[_run("■ ", color=c.lstrip("#")) + _run(r[0], bold=True)] + r[1:]
                                  for r, c in zip(v["rows"], v["colors"])]
            b.append(_table(rows, [3000, 1500, 1300, 1300, 1900], head=True))
        elif k == "transcript":
            for t, w, o, tr in v:
                b.append(_p(_run(t + "  ", color="8D959D", size=9) + _run(w + "  ", bold=True, size=10)
                            + _run(o, size=10) + (_run("\n" + tr, color="636D77", size=9.5) if tr else ""), "Line"))
    return "".join(b)


_EAST = {"ja": "Yu Gothic", "zh": "Microsoft YaHei", "ko": "Malgun Gothic", "en": "Yu Gothic"}


def _styles(lang: str = "ja") -> str:
    def st(sid, name, ppr="", rpr="", based="Normal", typ="paragraph"):
        return (f'<w:style w:type="{typ}" w:styleId="{sid}"><w:name w:val="{name}"/>'
                + (f'<w:basedOn w:val="{based}"/>' if based else "") + '<w:qFormat/>'
                + (f"<w:pPr>{ppr}</w:pPr>" if ppr else "") + (f"<w:rPr>{rpr}</w:rPr>" if rpr else "") + "</w:style>")
    # Word は Mac にも Windows にも游ゴシック・YaHei・Malgun を持つ(Hiragino は Windows に無い)
    fonts = f'<w:rFonts w:ascii="Arial" w:hAnsi="Arial" w:eastAsia="{_EAST.get(lang, "Yu Gothic")}" w:cs="Arial"/>'
    return (f'<?xml version="1.0" encoding="UTF-8" standalone="yes"?><w:styles {_W}>'
            f'<w:docDefaults><w:rPrDefault><w:rPr>{fonts}<w:color w:val="1C2127"/><w:sz w:val="21"/><w:szCs w:val="21"/>'
            f'<w:lang w:val="en-US" w:eastAsia="{ {"zh": "zh-CN", "ko": "ko-KR"}.get(lang, "ja-JP") }"/></w:rPr></w:rPrDefault>'
            '<w:pPrDefault><w:pPr><w:spacing w:after="80" w:line="300" w:lineRule="auto"/></w:pPr></w:pPrDefault></w:docDefaults>'
            + st("Normal", "Normal", based=None)
            + st("Kicker", "Kicker", '<w:spacing w:after="40"/>', '<w:caps/><w:spacing w:val="20"/>')
            + st("Title", "Title", '<w:spacing w:after="240"/>', '<w:b/><w:sz w:val="48"/><w:szCs w:val="48"/><w:color w:val="14181D"/>')
            + st("Heading1", "heading 1", '<w:keepNext/><w:spacing w:before="420" w:after="120"/><w:outlineLvl w:val="0"/>'
                 '<w:pBdr><w:bottom w:val="single" w:sz="4" w:space="4" w:color="DCD6CA"/></w:pBdr>',
                 '<w:b/><w:sz w:val="30"/><w:szCs w:val="30"/><w:color w:val="14181D"/>')
            + st("Heading2", "heading 2", '<w:keepNext/><w:spacing w:before="200" w:after="60"/><w:outlineLvl w:val="1"/>',
                 '<w:b/><w:sz w:val="24"/><w:szCs w:val="24"/>')
            + st("ListParagraph", "List Paragraph", '<w:ind w:left="360" w:hanging="240"/><w:spacing w:after="60"/>')
            + st("Todo", "To-do", '<w:ind w:left="360" w:hanging="360"/><w:spacing w:after="100"/>'
                 '<w:pBdr><w:bottom w:val="single" w:sz="2" w:space="4" w:color="E9E4DA"/></w:pBdr>')
            + st("Line", "Transcript line", '<w:spacing w:after="60"/><w:ind w:left="600" w:hanging="600"/>')
            + st("Compact", "Compact", '<w:spacing w:after="0"/>')
            + "</w:styles>")


def _numbering() -> str:
    return (f'<?xml version="1.0" encoding="UTF-8" standalone="yes"?><w:numbering {_W}>'
            '<w:abstractNum w:abstractNumId="0"><w:multiLevelType w:val="singleLevel"/>'
            '<w:lvl w:ilvl="0"><w:start w:val="1"/><w:numFmt w:val="bullet"/><w:lvlText w:val="•"/>'
            '<w:lvlJc w:val="left"/><w:pPr><w:ind w:left="360" w:hanging="240"/></w:pPr>'
            '<w:rPr><w:color w:val="8D959D"/></w:rPr></w:lvl></w:abstractNum>'
            '<w:num w:numId="1"><w:abstractNumId w:val="0"/></w:num></w:numbering>')


def to_docx(doc: dict) -> bytes:
    body = _docx_body(doc)
    sect = ('<w:sectPr><w:pgSz w:w="11906" w:h="16838"/>'
            '<w:pgMar w:top="1300" w:right="1300" w:bottom="1300" w:left="1300" w:header="600" w:footer="600" w:gutter="0"/>'
            '</w:sectPr>')
    document = (f'<?xml version="1.0" encoding="UTF-8" standalone="yes"?><w:document {_W}><w:body>'
                + body + sect + "</w:body></w:document>")
    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    lang = {"ja": "ja-JP", "en": "en-US", "zh": "zh-CN", "ko": "ko-KR"}.get(doc.get("lang"), "en-US")
    files = {
        "[Content_Types].xml": (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
            '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
            '<Default Extension="xml" ContentType="application/xml"/>'
            '<Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>'
            '<Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/>'
            '<Override PartName="/word/numbering.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.numbering+xml"/>'
            '<Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>'
            '<Override PartName="/docProps/app.xml" ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/>'
            "</Types>"),
        "_rels/.rels": (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>'
            '<Relationship Id="rId2" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" Target="docProps/core.xml"/>'
            '<Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties" Target="docProps/app.xml"/>'
            "</Relationships>"),
        "word/_rels/document.xml.rels": (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>'
            '<Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/numbering" Target="numbering.xml"/>'
            "</Relationships>"),
        "word/document.xml": document,
        "word/styles.xml": _styles(doc.get("lang", "ja")),
        "word/numbering.xml": _numbering(),
        "docProps/core.xml": (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" '
            'xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:dcterms="http://purl.org/dc/terms/" '
            'xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">'
            f"<dc:title>{_x(doc['title'])}</dc:title><dc:creator>Sokutsu</dc:creator><dc:language>{lang}</dc:language>"
            f'<dcterms:created xsi:type="dcterms:W3CDTF">{now}</dcterms:created></cp:coreProperties>'),
        "docProps/app.xml": (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties">'
            "<Application>Sokutsu</Application></Properties>"),
    }
    bio = BytesIO()
    with zipfile.ZipFile(bio, "w", zipfile.ZIP_DEFLATED) as z:
        for name, data in files.items():
            z.writestr(name, data)
    return bio.getvalue()


# ====================================================================== PDF

_FONTS = {"ja": ("HiraginoSans-W3", "HiraginoSans-W6"), "zh": ("PingFangSC-Regular", "PingFangSC-Semibold"),
          "ko": ("AppleSDGothicNeo-Regular", "AppleSDGothicNeo-Bold"), "en": ("HelveticaNeue", "HelveticaNeue-Bold")}
PAGE_W, PAGE_H, MARGIN, FOOT = 595.28, 841.89, 56.0, 30.0


def _rgb(h: str):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) / 255 for i in (0, 2, 4))


def render_pdf(doc: dict, path: "str | Path") -> int:
    """文書の形 → A4 の PDF。ページ数を返す。AppKit を使う(メインスレッドか子プロセスで呼ぶ)。"""
    import AppKit
    import Foundation
    import Quartz

    reg, bold = _FONTS.get(doc.get("lang"), _FONTS["ja"])
    S = AppKit.NSMutableAttributedString.alloc().init()

    def font(size, b=False):
        return (AppKit.NSFont.fontWithName_size_(bold if b else reg, size)
                or AppKit.NSFont.systemFontOfSize_weight_(size, 0.4 if b else 0.0))

    def para(before=0.0, after=4.0, indent=0.0, first=None, tabs=None, line=1.3, keep=False):
        ps = AppKit.NSMutableParagraphStyle.alloc().init()
        ps.setParagraphSpacingBefore_(before)
        ps.setParagraphSpacing_(after)
        ps.setLineHeightMultiple_(line)
        ps.setHeadIndent_(indent)
        ps.setFirstLineHeadIndent_(indent if first is None else first)
        if tabs is not None:
            ps.setTabStops_([AppKit.NSTextTab.alloc().initWithTextAlignment_location_options_(al, x, {})
                             for x, al in tabs])
        return ps

    def add(text, size=10.5, b=False, color="#1C2127", ps=None, end="\n", kern=0.0):
        attrs = {AppKit.NSFontAttributeName: font(size, b),
                 AppKit.NSForegroundColorAttributeName: AppKit.NSColor.colorWithSRGBRed_green_blue_alpha_(*_rgb(color), 1.0),
                 AppKit.NSParagraphStyleAttributeName: ps or para()}
        if kern:
            attrs[AppKit.NSKernAttributeName] = kern
        S.appendAttributedString_(AppKit.NSAttributedString.alloc().initWithString_attributes_(str(text) + end, attrs))

    W = PAGE_W - 2 * MARGIN
    INK, MUT, FAINT, GREEN = "#1C2127", "#636D77", "#8D959D", "#2F7A4C"
    add(doc.get("kicker", ""), 8.5, True, FAINT, para(after=6), kern=1.2)
    add(doc["title"], 21, True, "#14181D", para(after=14, line=1.2))
    kv = para(after=3, tabs=[(80.0, AppKit.NSTextAlignmentLeft)], indent=80.0, first=0.0)
    for k, v in doc["meta"]:
        add(k + "\t", 9.5, False, FAINT, kv, end="")
        add(v, 9.5, False, INK, kv)
    add("", 4, ps=para(after=6))
    h1 = para(before=16, after=6, line=1.15)
    h2 = para(before=8, after=3, line=1.2)
    bl = para(after=3.5, indent=14.0, first=0.0, tabs=[(14.0, AppKit.NSTextAlignmentLeft)])
    for blk in doc["blocks"]:
        k, v = blk[0], blk[1]
        if k == "h1":
            add(v, 14, True, "#14181D", h1)
        elif k == "h2":
            add(v, 11.5, True, "#14181D", h2)
        elif k == "bullets":
            for x in v:
                add("•\t", 10.5, False, FAINT, bl, end="")
                add(x, 10.5, False, INK, bl)
        elif k in ("checks", "questions"):
            mk, col = ("✓", GREEN) if k == "checks" else ("?", MUT)
            for x in v:
                add(mk + "\t", 10.5, True, col, bl, end="")
                add(x, 10.5, False, INK, bl)
        elif k == "points":
            for w, t in v:
                add("•\t", 10.5, False, FAINT, bl, end="")
                if w:
                    add(w + "  ", 10.5, True, INK, bl, end="")
                add(t, 10.5, False, INK, bl)
        elif k == "people":
            for nm, d in v:
                add("•\t", 10.5, False, FAINT, bl, end="")
                add(nm, 10.5, True, INK, bl, end="")
                add(("   " + d) if d else "", 9.5, False, MUT, bl)
        elif k == "todos":
            tp = para(after=1, indent=18.0, first=0.0, tabs=[(18.0, AppKit.NSTextAlignmentLeft)])
            tm = para(after=7, indent=18.0)
            for done, task, m in v:
                add(("☑" if done else "☐") + "\t", 11, False, GREEN if done else MUT, tp, end="")
                add(task, 10.5, not done, FAINT if done else INK, tp if m else para(after=7, indent=18.0, first=0.0,
                                                                                  tabs=[(18.0, AppKit.NSTextAlignmentLeft)]))
                if m:
                    add(m, 9, False, MUT, tm)
        elif k == "para":
            add(v, 10.5, False, INK, para(after=4))
        elif k == "note":
            add(v, 9, False, FAINT, para(after=4))
        elif k == "table":
            # 本物の表(NSTextTable)。右揃えのタブは Hiragino で行がずれることがあった(11/07 目で確認)
            tbl = AppKit.NSTextTable.alloc().init()
            tbl.setNumberOfColumns_(5)
            tbl.setCollapsesBorders_(True)
            widths = (40.0, 15.0, 13.0, 12.0, 20.0)

            def cell(r, c, text, size, b, color, right, bar=None):
                blk = AppKit.NSTextTableBlock.alloc().initWithTable_startingRow_rowSpan_startingColumn_columnSpan_(
                    tbl, r, 1, c, 1)
                blk.setValue_type_forDimension_(widths[c], AppKit.NSTextBlockPercentageValueType, AppKit.NSTextBlockWidth)
                blk.setWidth_type_forLayer_(5.0, AppKit.NSTextBlockAbsoluteValueType, AppKit.NSTextBlockPadding)
                blk.setWidth_type_forLayer_edge_(0.6 if r else 0.0, AppKit.NSTextBlockAbsoluteValueType,
                                                 AppKit.NSTextBlockBorder, AppKit.NSMinYEdge)
                blk.setBorderColor_(AppKit.NSColor.colorWithSRGBRed_green_blue_alpha_(*_rgb("#DCD6CA"), 1.0))
                ps = para(after=0, line=1.15)
                ps.setTextBlocks_([blk])
                ps.setAlignment_(AppKit.NSTextAlignmentRight if right else AppKit.NSTextAlignmentLeft)
                if bar is not None:
                    add("■ ", size - 1, False, bar[1], ps, end="")
                add(text, size, b, color, ps, end="\n" if bar is None else "")
                if bar is not None:
                    add("\n" + "━" * max(1, int(round(bar[0] * 30))), 5, False, bar[1], ps)

            for c, hd in enumerate(v["head"]):
                cell(0, c, hd, 8.5, True, FAINT, c > 0)
            for r, (row, share, col) in enumerate(zip(v["rows"], v["bars"], v["colors"]), start=1):
                cell(r, 0, row[0], 10, True, INK, False, bar=(share, col))
                for c in range(1, 5):
                    cell(r, c, row[c], 10, False, INK, True)
            add("", 4, ps=para(after=4))
        elif k == "transcript":
            tl = para(after=3, indent=40.0, first=0.0, tabs=[(40.0, AppKit.NSTextAlignmentLeft)], line=1.25)
            for t, w, o, tr in v:
                add(t + "\t", 8.5, False, FAINT, tl, end="")
                add(w + "  ", 9.5, True, INK, tl, end="")
                add(o, 9.5, False, INK, tl)
                if tr:
                    add(tr, 9, False, MUT, para(after=4, indent=40.0, line=1.2))

    storage = AppKit.NSTextStorage.alloc().initWithAttributedString_(S)
    lm = AppKit.NSLayoutManager.alloc().init()
    storage.addLayoutManager_(lm)
    body_h = PAGE_H - 2 * MARGIN - FOOT
    ranges = []
    total = lm.numberOfGlyphs()
    while True:
        tc = AppKit.NSTextContainer.alloc().initWithSize_((W, body_h))
        tc.setLineFragmentPadding_(0.0)
        lm.addTextContainer_(tc)
        gr = lm.glyphRangeForTextContainer_(tc)
        if gr.length == 0:
            break
        ranges.append(gr)
        if gr.location + gr.length >= total or len(ranges) > 2000:
            break
    url = Foundation.NSURL.fileURLWithPath_(str(path))
    info = {Quartz.kCGPDFContextTitle: doc["title"], Quartz.kCGPDFContextCreator: "Sokutsu"}
    ctx = Quartz.CGPDFContextCreateWithURL(url, Quartz.CGRectMake(0, 0, PAGE_W, PAGE_H), info)
    n = len(ranges)
    foot_attrs = {AppKit.NSFontAttributeName: font(8),
                  AppKit.NSForegroundColorAttributeName: AppKit.NSColor.colorWithSRGBRed_green_blue_alpha_(*_rgb(FAINT), 1.0)}
    for i, gr in enumerate(ranges):
        Quartz.CGContextBeginPage(ctx, None)
        Quartz.CGContextSaveGState(ctx)
        Quartz.CGContextTranslateCTM(ctx, 0, PAGE_H)
        Quartz.CGContextScaleCTM(ctx, 1, -1)
        g = AppKit.NSGraphicsContext.graphicsContextWithCGContext_flipped_(ctx, True)
        AppKit.NSGraphicsContext.saveGraphicsState()
        AppKit.NSGraphicsContext.setCurrentContext_(g)
        lm.drawBackgroundForGlyphRange_atPoint_(gr, (MARGIN, MARGIN))
        lm.drawGlyphsForGlyphRange_atPoint_(gr, (MARGIN, MARGIN))
        foot = AppKit.NSAttributedString.alloc().initWithString_attributes_(
            doc.get("page", "{i} / {n}").format(i=i + 1, n=n), foot_attrs)
        fw = foot.size().width
        foot.drawAtPoint_((PAGE_W - MARGIN - fw, PAGE_H - MARGIN + 6))
        left = AppKit.NSAttributedString.alloc().initWithString_attributes_(
            (doc["title"][:60] + ("  ·  " + doc.get("start", "")[:10] if doc.get("start") else "")), foot_attrs)
        left.drawAtPoint_((MARGIN, PAGE_H - MARGIN + 6))
        AppKit.NSGraphicsContext.restoreGraphicsState()
        Quartz.CGContextRestoreGState(ctx)
        Quartz.CGContextEndPage(ctx)
    Quartz.CGPDFContextClose(ctx)
    del ctx
    return n


def pdf_subprocess(doc: dict, path: "str | Path", python: "str | None" = None, timeout: float = 60.0) -> int:
    """子プロセスで PDF を作る(アプリの GUI スレッドの外で AppKit を安全に使うため)。ページ数を返す。"""
    src = Path(__file__).resolve().parents[1]
    env = dict(__import__("os").environ)
    env["PYTHONPATH"] = str(src) + (":" + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    r = subprocess.run([python or sys.executable, "-m", "tsuyaku.export", "pdf", str(path)],
                       input=json.dumps(doc, ensure_ascii=False), capture_output=True, text=True,
                       timeout=timeout, env=env)
    if r.returncode != 0:
        raise RuntimeError((r.stderr or r.stdout or "pdf").strip()[-300:])
    return int((r.stdout or "0").strip().splitlines()[-1])


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "pdf":
        print(render_pdf(json.loads(sys.stdin.read()), sys.argv[2]))
