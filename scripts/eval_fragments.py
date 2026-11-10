"""日中の短文・断片の訳の「足し・抜け」の評価(11/08)。eval/fragments/fragments.json(作った文 140)を、
アプリと同じ訳の依頼(translate.build_messages)で訳し、足した・抜いたを数える。

  run   : 訳す。--hist none(空の履歴)/ prev(同じ向きの直前4件を、そのモデルの訳つきで履歴に)
          --src-root で読む tsuyaku を切り替え(A/B)、--patch で設定を切り替え
  judge : リモートサーバーの LLM に「原文に無い内容を足したか・原文の内容を落としたか」を判定させる
  score : 表

指標: 抜け = must(訳に出るべき語)のどれかが無い項目の割合 / 足し = must_not(よくある足し)が出た項目の数 /
長さの比(訳 ÷ 参照訳)/ chrF(参照訳と)/ 判定(judge)の足し・抜け。
リモートサーバー と このMac の LLM は、ユーザーのアプリ(-m tsuyaku)が動いていたら使わない。
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import eval_conv as EC  # noqa: E402
from eval_terms import _llm, apply_patches, git_rev, guard_app, has_term, use_src  # noqa: E402

SET = HERE.parent / "eval" / "fragments" / "fragments.json"


def load():
    return json.loads(SET.read_text(encoding="utf-8"))["items"]


_FIRST = threading.local()


def _tr(base, model, msgs):
    from tsuyaku.translate import translate_stream
    t0 = time.time()
    first = [0.0]

    def od(_d):
        if not first[0]:
            first[0] = time.time()
    try:
        out = translate_stream(base, model, msgs, od, timeout=60)
    except Exception as e:  # noqa: BLE001
        print(f"[err] {e}", flush=True)
        out = ""
    _FIRST.ttft = (first[0] - t0) if first[0] else -1.0
    return out, time.time() - t0


def run(a) -> None:
    guard_app()
    use_src(a.src_root)
    apply_patches(a.patch)
    from tsuyaku.translate import build_messages
    base, model = _llm(a)
    items = load()
    rows, lock = [], threading.Lock()

    def direction(seq):
        hist = []
        for it in seq:
            pair = (it["src"], it["tgt"])
            msgs = build_messages(pair, "", hist[-4:] if a.hist == "prev" else [], it["text"], it["src"])
            out, sec = _tr(base, model, msgs)
            ttft = getattr(_FIRST, "ttft", -1.0)
            retried = False
            if a.guard:
                try:
                    from tsuyaku.guards import length_off
                except ImportError:
                    length_off = None
                if length_off and out and length_off(it["text"], out, it["src"], it["tgt"]):
                    out2, s2 = _tr(base, model, build_messages(pair, "", [], it["text"], it["src"]))
                    sec += s2
                    retried = True
                    if out2 and not length_off(it["text"], out2, it["src"], it["tgt"]):
                        out = out2
            if out:
                hist.append({"lang": it["src"], "orig": it["text"], "trans": out})
            with lock:
                rows.append({"id": it["id"], "src": it["src"], "kind": it["kind"], "text": it["text"],
                             "ref": it["ref"], "must": it["must"], "must_not": it["must_not"], "trans": out,
                             "sec": round(sec, 2), "ttft": round(ttft, 3), "retried": retried})

    seqs = [[i for i in items if i["src"] == s] for s in ("zh", "ja")]
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        list(ex.map(direction, seqs))
    meta = {"engine": a.engine, "model": model, "git": git_rev(a.src_root), "hist": a.hist, "patch": a.patch,
            "guard": a.guard}
    a.out.write_text(json.dumps({"meta": meta, "rows": rows}, ensure_ascii=False, indent=0), encoding="utf-8")
    print(json.dumps(score(rows), ensure_ascii=False))


JUDGE = ("You check translations. Source ({s}): {src}\nTranslation ({t}): {tr}\n"
         "Answer with JSON only: {{\"added\": true/false, \"omitted\": true/false}}. "
         "added = the translation states information that is not in the source (extra time, place, reason, "
         "a completed clause the source left unfinished). omitted = a piece of information in the source is "
         "missing from the translation. Ignore politeness level and small wording differences.")
NAMES = {"ja": "Japanese", "zh": "Chinese"}


def judge(a) -> None:
    from tsuyaku.translate import translate_stream
    base, model = a.base or "http://127.0.0.1:12345", a.model or "qwen3.8-flash-next"
    if not a.base:
        guard_app()
    d = json.loads(a.file.read_text(encoding="utf-8"))

    def one(r):
        if "judge" in r or not r["trans"]:
            return
        p = JUDGE.format(s=NAMES[r["src"]], t=NAMES["ja" if r["src"] == "zh" else "zh"], src=r["text"], tr=r["trans"])
        try:
            out = translate_stream(base, model, [{"role": "user", "content": p}], None, timeout=60)
            j = json.loads(out[out.index("{"):out.rindex("}") + 1])
            r["judge"] = {"added": bool(j.get("added")), "omitted": bool(j.get("omitted"))}
        except Exception as e:  # noqa: BLE001
            r["judge"] = {"error": str(e)[:80]}
    with ThreadPoolExecutor(max_workers=2) as ex:
        list(ex.map(one, d["rows"]))
    a.file.write_text(json.dumps(d, ensure_ascii=False, indent=0), encoding="utf-8")
    print(json.dumps(score(d["rows"]), ensure_ascii=False))


def score(rows, by_kind: bool = False) -> dict:
    def agg(rs):
        st = None
        miss = add = j_add = j_om = jn = 0
        ratio = []
        for r in rs:
            st = EC.add_stats(st, EC.chrf_stats(r["trans"], r["ref"]))
            miss += not all(has_term(r["trans"], m) for m in r["must"])
            add += any(has_term(r["trans"], m) for m in r["must_not"])
            ratio.append(len(EC._strip(r["trans"])) / max(1, len(EC._strip(r["ref"]))))
            j = r.get("judge") or {}
            if "added" in j:
                jn += 1
                j_add += j["added"]
                j_om += j["omitted"]
        import numpy as np
        return {"n": len(rs), "chrF": round(EC.chrf_from_stats(st), 2) if st else 0, "missing_must": miss,
                "must_not_hit": add, "ratio_med": round(float(np.median(ratio)), 2),
                "ratio_gt1.8": sum(x > 1.8 for x in ratio), "ratio_lt0.5": sum(x < 0.5 for x in ratio),
                "judge_added": j_add, "judge_omitted": j_om, "judged": jn,
                "retried": sum(1 for r in rs if r.get("retried")),
                "ttft_med": round(float(np.median([r["ttft"] for r in rs if r.get("ttft", -1) > 0])), 3)
                if any(r.get("ttft", -1) > 0 for r in rs) else None,
                "sec_med": round(float(np.median([r["sec"] for r in rs])), 3)}
    out = {"all": agg(rows)}
    if by_kind:
        for s in ("zh", "ja"):
            for k in ("short", "fragment", "long", "time"):
                rs = [r for r in rows if r["src"] == s and r["kind"] == k]
                if rs:
                    out[f"{s}-{k}"] = agg(rs)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--out", type=Path, required=True)
    r.add_argument("--engine", choices=["remote", "mac"], default="mac")
    r.add_argument("--src-root", default=None)
    r.add_argument("--patch", nargs="*", default=None)
    r.add_argument("--hist", choices=["none", "prev"], default="prev")
    r.add_argument("--guard", action="store_true", help="長さの比が外れたら空の履歴で訳し直す(アプリと同じ)")
    r.add_argument("--model", default=None, help="LLM のモデル名(このMacの別のモデルを試す)")
    r.add_argument("--base", default=None, help="LLM の URL(自分で立てた llama-server)")
    r.add_argument("--workers", type=int, default=2, help="1 = 1本ずつ(このMacで速さを測るとき)")
    j = sub.add_parser("judge")
    j.add_argument("file", type=Path)
    j.add_argument("--base", default=None, help="判定に使う LLM の URL(既定: アプリの リモートサーバーの訳 = トンネル 12345)")
    j.add_argument("--model", default=None)
    s = sub.add_parser("score")
    s.add_argument("files", nargs="+", type=Path)
    s.add_argument("--by-kind", action="store_true")
    a = ap.parse_args()
    if a.cmd == "run":
        run(a)
    elif a.cmd == "judge":
        judge(a)
    else:
        for f in a.files:
            d = json.loads(f.read_text(encoding="utf-8"))
            print(f.name, d.get("meta"))
            for k, v in score(d["rows"], a.by_kind).items():
                print("  ", k, v)


if __name__ == "__main__":
    main()
