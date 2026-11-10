"""会議に質問・全会議に質問・フォローアップの下書き・ライブ要約・自分のメモの織り込み(11/07 W1)。

他の会議メモアプリ(Otter / Fireflies / Notion / Granola / Zoom / Teams)にあって即通に無かった機能。
すべてローカルの LLM(remote / このMac)で動く。設計: 設計メモ(公開版には含めない)

- 会議に質問 ask(): 記録の各行に番号(#12)を付けて渡し、答えの各文の終わりに根拠の行 [#12] を
  書かせる。画面はその番号を押すと記録のその行へ飛ぶ。長い会議(ASK_BUDGET 字を超える)は、
  区切りごとに「質問に関係する行」を抜き出してから(2本ずつ並べる)、その行と前後だけで答える。
- 全会議に質問 ask_all(): 会議の一覧(題・日付・名前・キーワード・要約の頭)から LLM が関係する
  会議を最大3つ選び、それぞれの記録で答える。根拠は [M1#12](1つ目の会議の12行目)。
- 下書き draft(): メモ → フォローアップのメール / 正式な議事録(メモの言語で)。
- 自分のメモ・ブックマーク enrich(): 会議中のメモ(W3 の memo 行)をその時の発言で補って
  完全な文に、ブックマーク(mark 行)にその時に話されていたことを1文で。
- ライブ要約 LiveSchedule / CachedLLM: 数分ごと(または40行ごと)に会話中のメモを作り直す。
  区切りごとの抜き出しは、記録が伸びても前の区切りの中身は変わらないので覚えておき、
  新しい区切りとまとめだけ LLM に聞く(リモートサーバーの負荷を小さく)。
"""
from __future__ import annotations

import json
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable

import requests

from . import notes as N

EN = N.EN
ASK_BUDGET = 20000      # 1回の答えに渡す記録の字数(リモートサーバーは1本あたり約3.2万トークン)
MAP_CHARS = 9000        # 長い会議で関係する行を探す区切り
CATALOG_MAX = 60        # 全会議に質問: LLM に見せる会議の数


# ---------------- LLM(ストリーミング) ----------------

def chat_stream(base: str, model: str, messages: list[dict],
                on_delta: Callable[[str], None] | None = None,
                max_tokens: int = 1500, timeout: float = 120.0) -> str:
    """OpenAI 互換の chat をストリーミングで(答えを書きながら画面に出す)。"""
    payload = {"model": model, "messages": messages, "temperature": 0.2, "stream": True,
               "max_tokens": max_tokens, "reasoning_effort": "none"}
    url = base.rstrip("/") + "/v1/chat/completions"
    r = requests.post(url, json=payload, stream=True, timeout=timeout)
    if r.status_code == 400:
        payload.pop("reasoning_effort", None)
        r = requests.post(url, json=payload, stream=True, timeout=timeout)
    r.raise_for_status()
    out: list[str] = []
    for line in r.iter_lines():
        if not line:
            continue
        line = line.decode("utf-8", "replace")
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if data == "[DONE]":
            break
        try:
            obj = json.loads(data)
        except ValueError:
            continue
        if isinstance(obj, dict) and "error" in obj:
            raise RuntimeError(str(obj.get("error"))[:200])
        try:
            d = obj["choices"][0]["delta"].get("content")
        except (KeyError, IndexError, TypeError, AttributeError):
            continue
        if d:
            out.append(d)
            if on_delta:
                on_delta(d)
    return "".join(out).strip()


def _strip_think(t: str) -> str:
    return re.sub(r"<think>.*?</think>", "", t or "", flags=re.S).strip()


# ---------------- 根拠の番号 ----------------

_ONE = re.compile(r"\[\s*#\s*\d+(?:\s*[-–~]\s*#?\s*\d+)?(?:\s*[,、，]\s*#?\s*\d+(?:\s*[-–~]\s*#?\s*\d+)?)*\s*\]")
_MANY = re.compile(r"\[\s*M\s*\d+\s*#\s*\d+(?:\s*[,、，]\s*(?:M\s*\d+\s*)?#?\s*\d+)*\s*\]")


def parse_cites(text: str, n) -> tuple[str, list]:
    """答えの [#3] / [#5, #7] / [#2-#4](1つの会議。n = 行数)、[M1#2] / [M2#4, M2#5](全会議。
    n = {会議の番号: 行数})を読み、1つずつの [#5] [#7] にそろえる。範囲外は消す。
    返す: (整えた文, 根拠の並び(出てきた順・重複なし))。"""
    cites: list = []

    def add(c):
        if c not in cites:
            cites.append(c)
    if isinstance(n, dict):
        def rep(m):
            toks, cur = [], None
            for mm, ii in re.findall(r"(?:M\s*(\d+)\s*)?#?\s*(\d+)", m.group(0)[1:-1]):
                cur = int(mm) if mm else cur
                if cur is not None and cur in n and 0 <= int(ii) < n[cur]:
                    add((cur, int(ii)))
                    toks.append(f"[M{cur}#{int(ii)}]")
            return " ".join(toks)
        t = _MANY.sub(rep, text or "")
    else:
        def rep(m):
            toks = []
            for a, b in re.findall(r"#?\s*(\d+)(?:\s*[-–~]\s*#?\s*(\d+))?", m.group(0)[1:-1]):
                lo = int(a)
                hi = int(b) if b and 0 <= int(b) - lo <= 10 else lo
                for i in range(lo, hi + 1):
                    if 0 <= i < n:
                        add(i)
                        toks.append(f"[#{i}]")
            return " ".join(toks)
        t = _ONE.sub(rep, text or "")
    t = re.sub(r"[ \t]{2,}", " ", t)
    t = re.sub(r"[ \t]+([。、.,!?！？)）])", r"\1", t)
    return t.strip(), cites


# ---------------- 記録・メモ → LLM への文 ----------------

def numbered(lines: list[dict]) -> list[dict]:
    return [dict(ln, i=k) for k, ln in enumerate(lines)]


def _names(notes: "dict | None", known: "dict | None" = None) -> dict:
    m = {p.get("id"): p.get("name") for p in (notes or {}).get("participants", []) or []
         if isinstance(p, dict) and p.get("name")}
    m.update({k: v for k, v in (known or {}).items() if v})
    return m


def _speakers(lines: list[dict], names: dict) -> str:
    ids: list = []
    for ln in lines:
        if ln["spk"] not in ids:
            ids.append(ln["spk"])
    out = []
    for i in ids:
        if names.get(i):
            out.append(f"{i} = {names[i]}")
        elif i.startswith("L"):
            out.append(f"{i} = the {EN.get(i[1:], i[1:])} speaker (name unknown)")
        else:
            out.append(f"{i} = {N.default_name(i, 'en').strip()} (name unknown)")
    return "; ".join(out)


def _brief(notes: "dict | None", limit: int = 3000) -> str:
    """メモの要点(根拠の行番号つき)。答えの手がかりに。"""
    if not notes:
        return "(no notes yet)"

    def refs(src):
        return f" [{', '.join('#' + str(x) for x in src[:4])}]" if src else ""
    out = []
    if notes.get("title"):
        out.append(f"Title: {notes['title']}")
    for key, lab in (("summary", "Summary"), ("decisions", "Decisions"), ("questions", "Open questions")):
        items = notes.get(key) or []
        src = notes.get(f"{key}_src") or []
        if items:
            out.append(f"{lab}:")
            out += [f"- {x}{refs(src[k] if k < len(src) else [])}" for k, x in enumerate(items)]
    if notes.get("todos"):
        out.append("Action items:")
        for t in notes["todos"]:
            who = N.who_name(notes, t.get("owner", ""), notes.get("lang", "en")) if t.get("owner") else ""
            out.append(f"- {t.get('task', '')}" + (f" (owner: {who})" if who else "")
                       + (f" (due: {t['due']})" if t.get("due") else "") + refs(t.get("src") or []))
    s = "\n".join(out)
    return s[:limit]


def _line_brief(ln: dict, names: dict, lang: str) -> dict:
    return {"t": ln.get("t", ""), "spk": ln.get("spk", ""), "lang": ln.get("lang", ""),
            "name": names.get(ln.get("spk", "")) or "", "orig": ln.get("orig", ""),
            "trans": ln.get("trans", "")}


def _fix_ids(text: str, names: dict, lang: str) -> str:
    """答えに残った話者 id(S2 / Lja)を名前か「話者2」に。"""
    return N._ID_RE.sub(lambda m: names.get(m.group(1)) or N.default_name(m.group(1), lang), text)


def _run_final(msgs, llm, on_delta, final) -> str:
    if final is not None:
        return _strip_think(final(msgs, on_delta))
    t = _strip_think(llm(msgs))
    if on_delta and t:
        on_delta(t)
    return t


# ---------------- 長い会議: 質問に関係する行を探す ----------------

def _map_msgs(question: str, body: str) -> list[dict]:
    sys = ("You find the parts of a meeting transcript that are relevant to a question. Each line "
           "starts with its number (#12). List short facts from this excerpt that help answer the "
           "question, each with the numbers of the lines it comes from. Output only JSON: "
           '{"facts": [{"text": str, "lines": [int]}]}; use {"facts": []} if nothing is relevant.')
    user = f"Question: {question}\n\nExcerpt of the transcript:\n{body}"
    return [{"role": "system", "content": sys}, {"role": "user", "content": user}]


def _relevant(lines: list[dict], question: str, llm, budget: int, workers: int = 2) -> tuple[str, list]:
    """区切りごとに関係する行を LLM に挙げさせ、その行と前後を時刻順に並べた記録を返す。"""
    parts = N.chunks(lines, min(MAP_CHARS, max(2000, budget)))
    facts: list = [None] * len(parts)

    def one(k):
        try:
            got = N.parse_json(llm(_map_msgs(question, N.transcript(parts[k], with_trans=False))))
            facts[k] = [f for f in got.get("facts") or [] if isinstance(f, dict)]
        except Exception:  # noqa: BLE001
            facts[k] = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        list(ex.map(one, range(len(parts))))
    flat = [f for fs in facts for f in fs or []]
    hit: set = set()
    for f in flat:
        ls = f.get("lines")
        for x in ls if isinstance(ls, list) else [ls]:   # 小さい LLM は "lines": 5 / "12" とも書く
            try:
                if 0 <= int(x) < len(lines):
                    hit.add(int(x))
            except (TypeError, ValueError):
                continue
    for pad in (2, 1, 0):
        keep = sorted({j for i in hit for j in range(i - pad, i + pad + 1) if 0 <= j < len(lines)})
        body = _join_gaps([lines[i] for i in keep])
        if len(body) <= budget:
            break
    else:
        body = body[:budget]
    return body, flat


def _join_gaps(sel: list[dict]) -> str:
    out, prev = [], None
    for ln in sel:
        if prev is not None and ln["i"] != prev + 1:
            out.append("…")
        out.append(N.transcript([ln]))
        prev = ln["i"]
    return "\n".join(out)


# ---------------- 会議に質問 ----------------

_ASK_RULES = (
    "Answer in the same language as the question (if unclear, in {L}). Answer only what was asked "
    "(no unrelated details). Be concise: a few sentences, "
    "or a short list with \"- \" when listing several things. After each statement, cite the "
    "transcript lines it is based on, like {ex}. Cite only line numbers shown in the transcript. "
    "If the transcript does not answer the question, say so plainly in one sentence. Never invent "
    "names, numbers, dates or decisions. Refer to people by name when known (see Speakers), "
    "otherwise as \"Speaker 1\" and so on; never write speaker ids like S1. Never assume anyone's "
    "gender. Plain text only (no headings, no bold)."
)


def ask(meeting: dict, notes: "dict | None", question: str, lang: str, llm,
        on_delta: Callable[[str], None] | None = None, final=None, budget: int = ASK_BUDGET,
        known: "dict | None" = None, workers: int = 2, on_stage=None) -> dict:
    """1つの会議に質問。返す: {answer, cites: [行番号], lines: {"12": {t, spk, name, orig, trans}}}。
    llm(msgs)->str。final(msgs, on_delta)->str があれば答えはそれで(ストリーミング)。"""
    lines = numbered(meeting.get("lines") or [])
    names = _names(notes, known)
    if not lines:
        return {"answer": "", "cites": [], "lines": {}, "empty": True}
    body = N.transcript(lines)
    partial = len(body) > budget
    if partial:
        if on_stage:
            on_stage("search", 1)     # 長い会議: 関係する行を探してから答える
        body, _facts = _relevant(lines, question, llm, budget, workers)
    L = EN.get(lang, "Japanese")
    sys = ("You answer questions about a meeting, using only the meeting transcript and notes below. "
           + _ASK_RULES.format(L=L, ex="[#12] or [#12, #15]"))
    user = (f"Speakers: {_speakers(lines, names)}\n\nMeeting notes (may be incomplete):\n{_brief(notes)}\n\n"
            + ("Transcript (only the parts that matter for the question; … = lines skipped):\n" if partial
               else "Transcript:\n")
            + (body or "(no part of the transcript is about this)") + f"\n\nQuestion: {question}")
    text = _run_final([{"role": "system", "content": sys}, {"role": "user", "content": user}],
                      llm, on_delta, final)
    clean, cites = parse_cites(text, len(lines))
    clean = _fix_ids(clean, names, lang)
    return {"answer": clean, "cites": cites,
            "lines": {str(i): _line_brief(lines[i], names, lang) for i in cites}}


_SUGGEST = {
    "ja": (["決まったことは?", "次にやることは?", "残っている課題は?"], "{n}さんの意見は?"),
    "en": (["What was decided?", "What are the next steps?", "What is still open?"], "What did {n} think?"),
    "zh": (["决定了什么?", "接下来要做什么?", "还有哪些未解决的问题?"], "{n}的意见是什么?"),
    "ko": (["무엇이 결정됐나요?", "다음에 할 일은?", "남은 과제는?"], "{n} 님의 의견은?"),
}


def suggestions(notes: "dict | None", lang: str, known: "dict | None" = None) -> list[str]:
    """質問の例: 決まったこと・次にやること・〇〇さんの意見(名前が分かった人、最大2人)。"""
    base, per = _SUGGEST.get(lang, _SUGGEST["ja"])
    names = [v for v in _names(notes, known).values() if v][:2]
    return base[:1] + [per.format(n=x) for x in names] + base[1:]


# ---------------- 全会議に質問 ----------------

def _terms(q: str) -> set:
    q = q.lower()
    out = {w for w in re.findall(r"[a-z0-9][a-z0-9'-]{2,}", q)}
    cjk = re.sub(r"[^぀-ヿ㐀-鿿가-힯]", " ", q)
    for seg in cjk.split():
        out |= {seg[i:i + 2] for i in range(len(seg) - 1)}
    return out - {"the", "and", "what", "was", "did", "how", "who", "about"}


def _catalog(log_dir) -> list[dict]:
    out = []
    for it in N.list_meetings(log_dir):
        p = Path(log_dir) / f"{it['id']}.jsonl"
        n = N.load_notes(p) or {}
        out.append(dict(it, path=str(p), summary=(n.get("summary") or [])[:2],
                        title=it.get("title") or n.get("title") or ""))
    return out


def ask_all(log_dir, question: str, lang: str, llm, on_delta=None, final=None,
            budget: int = ASK_BUDGET, max_meetings: int = 3, on_stage=None) -> dict:
    """すべての会議に質問。関係する会議を選んで(最大3)、それぞれの記録で答える。
    返す: {answer, cites: [[会議k, 行]], meetings: [{k, id, title, start}], lines: {"k#i": {...}}}。"""
    cat = _catalog(log_dir)
    empty = {"answer": "", "cites": [], "meetings": [], "lines": {}}
    if not cat:
        return empty
    terms = _terms(question)

    def score(c):
        text = " ".join([c.get("title", ""), c.get("preview", ""), " ".join(c.get("keywords") or []),
                         " ".join(c.get("names") or []), " ".join(c.get("summary") or [])]).lower()
        return sum(1 for t in terms if t in text)
    if len(cat) > CATALOG_MAX:      # 多すぎる時は言葉の重なりで絞ってから見せる(同点は新しい順)
        cat = sorted(cat, key=lambda c: -score(c))[:CATALOG_MAX]
        cat.sort(key=lambda c: c.get("start", ""), reverse=True)
    rows = []
    for k, c in enumerate(cat, 1):
        rows.append(f"[{k}] {c.get('start', '')[:16]} {c.get('title') or c.get('preview', '')}"
                    + (f" | people: {', '.join(c['names'])}" if c.get("names") else "")
                    + (f" | keywords: {', '.join(c['keywords'][:8])}" if c.get("keywords") else "")
                    + (f" | {' '.join(c['summary'])}" if c.get("summary") else ""))
    pick_msgs = [{"role": "system", "content": "You pick which past meetings can answer a question."},
                 {"role": "user", "content": f"Question: {question}\n\nMeetings (newest first):\n"
                  + "\n".join(rows) + "\n\nWhich meetings are needed to answer the question? Output only "
                  'JSON {"meetings": [numbers]} with at most ' + str(max_meetings)
                  + ", most relevant first; [] if none fits."}]
    sel: list = []
    try:
        for x in N.parse_json(llm(pick_msgs)).get("meetings") or []:
            if isinstance(x, int) and 1 <= x <= len(cat) and x not in sel:
                sel.append(x)
    except Exception:  # noqa: BLE001
        sel = []
    if not sel:     # 選べなかった: 言葉の重なりがある会議、無ければいちばん新しい会議
        ranked = sorted(range(len(cat)), key=lambda k: -score(cat[k]))
        sel = [k + 1 for k in ranked if score(cat[k]) > 0][:max_meetings] or [1]
    sel = sel[:max_meetings]
    if on_stage:
        on_stage("read", len(sel))
    per = max(4000, budget // len(sel))
    blocks, nlines, metas, by = [], {}, [], {}
    for j, k in enumerate(sel, 1):
        c = cat[k - 1]
        m = N.load_meeting(c["path"])
        notes = N.load_notes(c["path"])
        names = _names(notes, N.load_speakers(c["path"]))
        lines = numbered(m["lines"])
        body = N.transcript(lines)
        if len(body) > per:
            body, _f = _relevant(lines, question, llm, per)
        nlines[j] = len(lines)
        by[j] = (lines, names)
        metas.append({"k": j, "id": m["id"], "title": c.get("title") or c.get("preview", ""),
                      "start": c.get("start", "")})
        blocks.append(f"=== Meeting M{j}: {c.get('start', '')[:16]} {c.get('title') or ''} ===\n"
                      f"Speakers: {_speakers(lines, names)}\nNotes:\n{_brief(notes, 1500)}\n"
                      f"Transcript:\n{body}")
    if on_stage:
        on_stage("answer", len(sel))
    L = EN.get(lang, "Japanese")
    sys = ("You answer questions about past meetings, using only the meetings below. "
           + _ASK_RULES.format(L=L, ex="[M1#12] (meeting M1, line 12) or [M1#12, M2#3]")
           + " Refer to a meeting by its date or title in the text (never as \"M1\" outside the "
           "brackets of a citation).")
    user = "\n\n".join(blocks) + f"\n\nQuestion: {question}"
    text = _run_final([{"role": "system", "content": sys}, {"role": "user", "content": user}],
                      llm, on_delta, final)
    clean, cites = parse_cites(text, nlines)
    allnames = {}
    for _lines, nm in by.values():
        allnames.update(nm)
    clean = _fix_ids(clean, allnames, lang)
    return {"answer": clean, "cites": [[a, b] for a, b in cites], "meetings": metas,
            "lines": {f"{a}#{b}": dict(_line_brief(by[a][0][b], by[a][1], lang), id=metas[a - 1]["id"])
                      for a, b in cites}}


# ---------------- フォローアップの下書き ----------------

_DRAFT = {
    "email": ("You write the follow-up email to send to the participants after a meeting, in {L}. "
              "Polite, concise business tone{jp}. Start with the subject line ({subj}), then a greeting, "
              "a short thank-you, a brief summary, the decisions, the action items with owner and due "
              "date, open questions if any, and a closing. Use only the notes below; never invent facts, "
              "dates or people. Leave the sender's signature as a placeholder. Output plain text only "
              "(no markdown symbols such as ** or #)."),
    "minutes": ("You write formal meeting minutes in {L}{jpm}, from the notes below. Sections in this "
                "order: title, date and time, participants, agenda (the topics), discussion (a few "
                "sentences per topic, who said what when it matters), decisions, action items (task, "
                "owner, due), open issues, next meeting if mentioned. Formal, neutral wording. Use only "
                "the notes; never invent facts. Plain text with simple headings (e.g. 【決定事項】 in "
                "Japanese) and \"・\" or \"- \" bullets; no markdown symbols such as ** or #."),
}
_SUBJ = {"ja": "「件名: …」", "en": "\"Subject: …\"", "zh": "「主题: …」", "ko": "\"제목: …\""}


def draft(notes: dict, kind: str, lang: str, llm) -> str:
    """メモ → メールの下書き(email)/ 議事録(minutes)。メモの言語で。"""
    if kind not in _DRAFT:
        raise ValueError(f"kind: {kind}")
    lang = lang if lang in N.LANGS else (notes.get("lang") or "ja")
    sys = _DRAFT[kind].format(L=EN.get(lang, "Japanese"), subj=_SUBJ.get(lang, _SUBJ["en"]),
                              jp=" (Japanese business email: です・ます)" if lang == "ja" else "",
                              jpm=" (Japanese 議事録, である調)" if lang == "ja" else "")
    md = N.to_markdown(notes, None, with_transcript=False)
    text = _strip_think(llm([{"role": "system", "content": sys},
                             {"role": "user", "content": f"Meeting notes:\n{md}"}]))
    text = re.sub(r"^```\w*\n|\n```\s*$", "", text.strip())
    return text.replace("**", "").strip()


# ---------------- 自分のメモ・ブックマークを織り込む ----------------

def line_at(lines: list[dict], ts=None, t: str = "") -> "int | None":
    """その時刻に(直前まで)話されていた行。ts(壁時計の秒)が無ければ "HH:MM:SS" で。"""
    best = None
    for i, ln in enumerate(lines):
        if ts is not None and ln.get("ts") is not None:
            try:
                if float(ln["ts"]) <= float(ts) + 0.5:
                    best = i
                continue
            except (TypeError, ValueError):
                pass
        if t and ln.get("t") and ln["t"] <= t:
            best = i
    return best if best is not None else (0 if lines else None)


def enrich(out: dict, lines: list[dict], memos: list, marks: list, lang: str, llm) -> dict:
    """会議中の自分のメモ(memos)・ブックマーク(marks)をメモに入れる(1回の LLM)。
    out["my_notes"] = [{t, i, memo, text, src}](memo = 自分の言葉そのまま、text = 補った文)、
    out["highlights"] = [{t, i, label, text, src}]。LLM が失敗しても自分の言葉は残す。"""
    lines = numbered(lines) if lines and "i" not in lines[0] else lines
    hl, my = [], []
    def where(m):     # W3 の load_meeting が指す行(i)を付けていればそれ、無ければ時刻から
        i = m.get("i")
        if isinstance(i, int) and not isinstance(i, bool) and 0 <= i < len(lines):
            return i
        return line_at(lines, m.get("ts"), m.get("t", ""))
    for m in marks:
        i = where(m)
        if i is not None:
            hl.append({"t": m.get("t") or lines[i].get("t", ""), "i": i, "label": str(m.get("label") or ""),
                       "kind": str(m.get("kind") or "star"), "text": "", "src": [i]})
    for m in memos:
        txt = str(m.get("text") or "").strip()
        if not txt:
            continue
        i = where(m)
        my.append({"t": m.get("t", ""), "i": i, "memo": txt, "text": "", "src": [i] if i is not None else []})
    if not hl and not my:
        return out
    L = EN.get(lang, "Japanese")
    names = _names(out)
    parts = []
    if hl:
        parts.append("Bookmarks:")
        for k, h in enumerate(hl):
            ctx = lines[max(0, h["i"] - 3):h["i"] + 2]
            parts.append(f"[k={k}] at line #{h['i']} ({h['t']})" + (f" label: {h['label']}" if h["label"] else "")
                         + (f" (marked as: {h['kind']})" if h["kind"] != "star" else "")
                         + "\n" + N.transcript(ctx, with_trans=False))
    if my:
        parts.append("User notes:")
        for k, x in enumerate(my):
            i = x["i"] if x["i"] is not None else 0
            ctx = lines[max(0, i - 6):i + 2]
            parts.append(f"[k={k}] ({x['t']}) \"{x['memo']}\"\n" + N.transcript(ctx, with_trans=False))
    msgs = [{"role": "system", "content": f"You help write meeting notes. Write every text in {L}. "
             "Use only the transcript; never invent facts. Never write speaker ids like S1; use names "
             f"(Speakers: {_speakers(lines, names)}) or \"a participant\". Never assume gender."},
            {"role": "user", "content":
             f"During the meeting the user bookmarked moments and wrote their own notes. For each "
             f"bookmark, write one sentence in {L} about what was being said at that moment. For each "
             f"user note, rewrite it as a complete, clear note (1-2 sentences in {L}) using the "
             "transcript around it; keep the user's point and wording where possible. Output only "
             'JSON: {"highlights": [{"k": number, "text": str}], "my_notes": [{"k": number, "text": str}]}'
             "\n\n" + "\n\n".join(parts)}]
    try:
        got = N.parse_json(llm(msgs))
        for key, arr in (("highlights", hl), ("my_notes", my)):
            for x in got.get(key) or []:
                k = x.get("k") if isinstance(x, dict) else None
                if isinstance(k, int) and 0 <= k < len(arr):
                    arr[k]["text"] = _fix_ids(N.split_refs(str(x.get("text") or ""))[0], names, lang)
    except Exception as e:  # noqa: BLE001
        print(f"[notes] 自分のメモの織り込みに失敗: {type(e).__name__}: {e}", flush=True)
    if hl:
        out["highlights"] = [{k: h[k] for k in ("t", "i", "label", "kind", "text", "src")} for h in hl]
    if my:
        out["my_notes"] = my
    return out


# ---------------- ライブ要約 ----------------

class LiveSchedule:
    """会話中のメモの作り直しの間隔: every_s 秒たって新しい行がある、または every_lines 行増えた。"""

    def __init__(self, every_s: float = 300.0, every_lines: int = 40, min_lines: int = 8):
        self.every_s, self.every_lines, self.min_lines = every_s, every_lines, min_lines
        self.start: "float | None" = None
        self.last_t: "float | None" = None
        self.last_n = 0

    def due(self, now: float, n_lines: int) -> bool:
        if self.start is None:
            self.start = now
        if n_lines < self.min_lines or n_lines <= self.last_n:
            return False
        if n_lines - self.last_n >= self.every_lines:
            return True
        ref = self.last_t if self.last_t is not None else self.start
        return now - ref >= self.every_s

    def mark(self, now: float, n_lines: int) -> None:
        self.last_t, self.last_n = now, n_lines


def live_key(msgs) -> str:
    """ライブ要約の覚え方の鍵: 話者の付け直し(会話中に15発話ごと)で前の区切りの話者 id が少し
    変わっても、同じ区切りとみなす(変わるたびに全部の区切りを聞き直すと リモートサーバーが重くなる)。
    話者の違いは会話の後の「メモ」(付け直しの後で全部作る)で正しくなる。"""
    out = []
    for m in msgs:
        c = re.sub(r"^Speakers: .*?\n\n", "", str(m.get("content", "")), flags=re.S)
        out.append([m.get("role"), re.sub(r"\] (?:S\d+|L\w+) \(", "] ? (", c)])
    return json.dumps(out, ensure_ascii=False)


class CachedLLM:
    """同じ問い(区切りの中身が同じ)には前の答えを返す。ライブ要約で前の区切りを聞き直さない。"""

    def __init__(self, llm, max_items: int = 300, key=None):
        self.llm, self.max_items = llm, max_items
        self.key = key or (lambda msgs: json.dumps(msgs, ensure_ascii=False, sort_keys=True))
        self._c: dict = {}
        self._lock = threading.Lock()

    def __call__(self, msgs):
        key = self.key(msgs)
        with self._lock:
            if key in self._c:
                return self._c[key]
        out = self.llm(msgs)
        try:
            N.parse_json(out)           # 壊れた答えは覚えない(やり直しが効くように)
        except ValueError:
            return out
        with self._lock:
            if len(self._c) >= self.max_items:
                self._c.pop(next(iter(self._c)))
            self._c[key] = out
        return out

