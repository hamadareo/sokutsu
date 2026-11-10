"""会議メモの評価(AMI 会議コーパス: 英語の実会議、CC BY 4.0)。

AMI には人が書いた要約(abstract)・決定事項(decisions)・やること(actions)・問題(problems)がある
(ami_public_manual の abstractive/*.abssumm.xml)。手作業の書き起こし(words/*.words.xml、話者 A〜D と
単語の時刻)から会議の記録を組み立て、アプリと同じ notes.make_notes でメモを作り、
人の要約の各文がメモに入っているか(再現率)を LLM に判定させる。

  .venv/bin/python -I scripts/eval_notes.py --zip AMI/ami_manual.zip --meetings ES2004a,ES2004b --out DIR
指標(会議ごと・合計):
  abstract/actions/decisions の再現率(人の要約の文のうちメモが扱っている割合。NA は数えない)
  メモの件数(要約・やること・決定・質問)、名前の付いた参加者と根拠が記録に実在するか(作り話の名前)、
  かかった秒数・区切りの数。
LLM は remote(SSH 転送 127.0.0.1:12345)。判定役も同じ LLM(偏りはあるが A/B の差を見るには足りる)。
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tsuyaku import notes as N  # noqa: E402
from tsuyaku.translate import pick_model, warmup  # noqa: E402

NS = "{http://nite.sourceforge.net/}"


def words(z: zipfile.ZipFile, mid: str) -> list[tuple[float, float, str, str]]:
    out = []
    for name in z.namelist():
        m = re.fullmatch(rf"words/{mid}\.([A-Z])\.words\.xml", name)
        if not m:
            continue
        root = ET.fromstring(z.read(name))
        for w in root.iter("w"):
            if not w.text:
                continue
            try:
                st, en = float(w.get("starttime")), float(w.get("endtime"))
            except (TypeError, ValueError):
                continue
            out.append((st, en, m.group(1), w.text, w.get("punc") == "true"))
    return sorted(out)


def utterances(ws, gap: float = 1.0) -> list[dict]:
    """話者ごとに、間が gap 秒を超えるか話者が替わったら切る。"""
    by: dict[str, list] = {}
    for st, en, spk, tx, punc in ws:
        by.setdefault(spk, []).append((st, en, tx, punc))
    utts = []
    for spk, lst in by.items():
        cur, last = [], None
        for st, en, tx, punc in lst:
            if cur and last is not None and st - last > gap:
                utts.append((cur[0][0], spk, cur))
                cur = []
            cur.append((st, en, tx, punc))
            last = en
        if cur:
            utts.append((cur[0][0], spk, cur))
    lines = []
    for st, spk, cur in sorted(utts):
        text = ""
        for _s, _e, tx, punc in cur:
            text += tx if (punc or not text) else " " + tx
        if len(text.split()) < 2 and text.lower().strip(" .") in ("hmm", "mm", "uh", "um", "yeah", "okay", "mm-hmm"):
            continue      # 相槌だけの行は記録でもほぼ意味がない(アプリでも幻覚守卫で多くが落ちる)
        m, s = divmod(int(st), 60)
        lines.append({"t": f"{m // 60:02d}:{m % 60:02d}:{s:02d}", "lang": "en", "orig": text.strip(),
                      "trans": "", "spk": f"S{ord(spk) - 64}", "ts": None})
    return lines


def gold(z: zipfile.ZipFile, mid: str) -> dict:
    name = f"abstractive/{mid}.abssumm.xml"
    if name not in z.namelist():
        return {}
    root = ET.fromstring(z.read(name))
    out = {}
    for tag in ("abstract", "actions", "decisions", "problems"):
        sents = []
        for el in root.iter(tag):
            for s in el.iter("sentence"):
                t = (s.text or "").strip()
                if t and t.upper().strip(".") != "NA":
                    sents.append(t)
        out[tag] = sents
    return out


def judge(chat, notes_md: str, items: list[str]) -> list[bool]:
    if not items:
        return []
    msgs = [{"role": "system", "content":
             "You compare meeting notes against reference points written by a human annotator. "
             "For each reference point, answer true if the notes cover the same content (wording may "
             "differ; partial but clearly matching coverage counts), else false. Output only JSON: "
             '{"covered": [true, false, ...]} in the same order.'},
            {"role": "user", "content": f"Meeting notes:\n{notes_md}\n\nReference points:\n"
             + "\n".join(f"{i + 1}. {t}" for i, t in enumerate(items))}]
    for _ in range(2):
        try:
            got = N.parse_json(chat(msgs)).get("covered", [])
            got = [bool(x) for x in got][:len(items)]
            return got + [False] * (len(items) - len(got))
        except ValueError:
            continue
    return [False] * len(items)


def cite_check(chat, n: dict, lines: list[dict]) -> tuple[int, int, int]:
    """11/07 W1: 根拠の行の評価。返す: (項目の数, 番号の付いた数, 付いた番号が中身を支えている数)。
    支えているかは LLM が判定(要約・決定・やること・話題の要点)。"""
    items = []
    for k in ("summary", "decisions"):
        for t, s in zip(n.get(k, []), n.get(f"{k}_src", [])):
            items.append((t, s))
    items += [(t["task"], t.get("src", [])) for t in n.get("todos", [])]
    items += [(p["text"], p.get("src", [])) for tp in n.get("topics", []) for p in tp["points"]]
    with_src = [(t, s) for t, s in items if s]
    if not with_src:
        return len(items), 0, 0
    rows = []
    for k, (t, s) in enumerate(with_src):
        ctx = " / ".join(f"#{i}: {lines[i]['orig']}" for i in s[:4] if 0 <= i < len(lines))
        rows.append(f"{k + 1}. ITEM: {t}\n   LINES: {ctx}")
    msgs = [{"role": "system", "content":
             "For each meeting-note item, judge whether the quoted transcript lines support it "
             "(they state or clearly lead to the item; partial support of the main point counts). "
             'Output only JSON: {"supported": [true, false, ...]} in the same order.'},
            {"role": "user", "content": "\n".join(rows)}]
    for _ in range(2):
        try:
            got = [bool(x) for x in N.parse_json(chat(msgs)).get("supported", [])][:len(with_src)]
            return len(items), len(with_src), sum(got)
        except ValueError:
            continue
    return len(items), len(with_src), 0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", required=True)
    ap.add_argument("--meetings", default="ES2004a,ES2004b,ES2004c,ES2004d,IS1009b,TS3003c")
    ap.add_argument("--out", required=True)
    ap.add_argument("--lang", default="en")
    ap.add_argument("--base", default="http://127.0.0.1:12345")
    ap.add_argument("--model", default="qwen3.8-flash-next")
    ap.add_argument("--chunk", type=int, default=N.CHUNK_CHARS)
    ap.add_argument("--template", default="free", help="会議の種類(11/07 W1)")
    ap.add_argument("--cites", action="store_true", help="根拠の行の正しさも測る(11/07 W1)")
    args = ap.parse_args()
    warmup(args.base, args.model)
    model = pick_model(args.base, args.model)
    chat = lambda msgs: N.chat(args.base, model, msgs)  # noqa: E731
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    z = zipfile.ZipFile(args.zip)
    tot: dict[str, list[int]] = {}
    rows = []
    for mid in args.meetings.split(","):
        g = gold(z, mid)
        lines = utterances(words(z, mid))
        if not lines or not g:
            print(f"{mid}: データなし", flush=True)
            continue
        meeting = {"id": mid, "meta": {"start": "2026-01-01 10:00:00", "mode": "notes"}, "lines": lines}
        t0 = time.time()
        n = N.make_notes(meeting, args.lang, chat, model=model, chunk_chars=args.chunk,
                         template=args.template)
        dt = time.time() - t0
        md = N.to_markdown(n, None, with_transcript=False)
        (out / f"{mid}.notes.json").write_text(json.dumps(n, ensure_ascii=False, indent=1))
        (out / f"{mid}.md").write_text(md)
        alltext = " ".join(ln["orig"].lower() for ln in lines)
        named = [p for p in n["participants"] if p.get("name")]
        fake = [p for p in named if not p.get("evidence") or p["evidence"].lower()[:20] not in alltext]
        r = {"meeting": mid, "lines": len(lines), "chars": sum(len(x["orig"]) for x in lines),
             "chunks": len(N.chunks(lines, args.chunk)), "sec": round(dt, 1),
             "n_summary": len(n["summary"]), "n_todos": len(n["todos"]),
             "n_decisions": len(n["decisions"]), "n_questions": len(n["questions"]),
             "named": len(named), "names_without_evidence": len(fake)}
        for tag in ("abstract", "actions", "decisions", "problems"):
            cov = judge(chat, md, g.get(tag, []))
            r[f"{tag}_recall"] = f"{sum(cov)}/{len(cov)}"
            tot.setdefault(tag, [0, 0])
            tot[tag][0] += sum(cov)
            tot[tag][1] += len(cov)
        if args.cites:
            a, b, c = cite_check(chat, n, lines)
            r["cites"] = f"{b}/{a} 付き, {c}/{b} 正しい"
            for k, v in (("cite_items", a), ("cite_with", b), ("cite_ok", c)):
                tot.setdefault(k, [0, 0])
                tot[k][0] += v
                tot[k][1] += 1
        rows.append(r)
        print(json.dumps(r, ensure_ascii=False), flush=True)
    ci = {k: tot.pop(k)[0] for k in ("cite_items", "cite_with", "cite_ok") if k in tot}
    summ = {k: f"{a}/{b} = {a / b:.0%}" if b else "-" for k, (a, b) in tot.items()}
    if ci:
        summ["cites_attached"] = f"{ci['cite_with']}/{ci['cite_items']}"
        summ["cites_supported"] = f"{ci['cite_ok']}/{ci['cite_with']}"
    print("合計", json.dumps(summ, ensure_ascii=False), flush=True)
    (out / "summary.json").write_text(json.dumps({"rows": rows, "total": summ}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
