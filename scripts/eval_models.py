"""リモートサーバーの訳のモデルの選び直し(11/08 品質の柱 B「品質の高い翻訳をリアルタイムで」)。

同じ材料を、候補のモデル(自分で立てた llama-server)と今の 翻訳ゲートウェイ に通し、品質と速さを比べる。

材料(どれも作った文):
  fleurs : FLEURS の並行文 N 文 × 12方向(一般の文。アプリと同じ履歴つき)
  qual   : engloop の qualA の対話 9本・126発話(文脈が要る 56)。text 条件 = 正しい原文と前の文脈 → 訳
  it     : itterms の IT 用語の試験文から M 項目 × 12方向(用語の当たり)
  frag   : 日中の断片 140(足し・抜け。判定は eval_fragments.py judge)
プロンプト:
  app    : アプリと同じ依頼(translate.build_messages、指示文 + 直前の履歴 + 用語の手がかり)
  native : 翻訳専用モデルの勧める1文の依頼(履歴なし。Hunyuan-MT / Seed-X / Tower)
速さ: 初字(最初の文字が届くまで)と1文の秒。--workers 2(アプリは確定訳と先読みの2本を並べる)。

  run   : 訳す → results/<name>.json
  comet : COMET(Unbabel/wmt22-comet-da)を足す(COMET の環境の python で動かす)
  score : 表
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
sys.path.insert(0, str(HERE.parent / "src"))

LANGS = ("ja", "en", "zh", "ko")
NAMES = {"ja": "Japanese", "en": "English", "zh": "Chinese", "ko": "Korean"}
ZH_NAMES = {"ja": "日语", "en": "英语", "zh": "中文", "ko": "韩语"}
SEEDX = {"ja": "<ja>", "en": "<en>", "zh": "<zh>", "ko": "<ko>"}


# ---------------------------------------------------------------- 材料

def load_sets(a) -> list:
    """[(set, key, src, tgt, text, ref, extra, conv_key)]。conv_key が同じものは順に(履歴つき)訳す。"""
    import eval_conv as EC
    out = []
    if "fleurs" in a.sets:
        tsv = {l: {r["sid"]: r["text"] for r in EC.load_tsv(Path(a.fleurs), l)} for l in LANGS}
        ids = sorted(set.intersection(*[set(v) for v in tsv.values()]))[: a.n_fleurs]
        for s in LANGS:
            for t in LANGS:
                if s != t:
                    for i in ids:
                        out.append(("fleurs", f"{i}", s, t, tsv[s][i], tsv[t][i], {}, f"fleurs:{s}-{t}"))
    if "qual" in a.sets:
        for p in sorted(Path(a.qual).glob("*.json")):
            d = json.loads(p.read_text(encoding="utf-8"))
            for k, u in enumerate(d["turns"]):
                pair = d["pair"].split("-")
                tgt = pair[1] if u["lang"] == pair[0] else pair[0]
                out.append(("qual", f"{p.stem}:{k}", u["lang"], tgt, u["text"], u["ref"],
                            {"ctx": u.get("ctx", False), "kind": u.get("kind", "")}, f"qual:{p.stem}"))
    if "it" in a.sets:
        items = json.loads((HERE.parent / "eval/terms/it_testset.json").read_text(encoding="utf-8"))
        items = items[:: max(1, len(items) // a.n_it)][: a.n_it]
        for s in LANGS:
            for t in LANGS:
                if s != t:
                    for it in items:
                        out.append(("it", it["id"], s, t, it["texts"][s], it["texts"][t],
                                    {"terms": [x[t] for x in it["terms"]], "kinds": [x["kind"] for x in it["terms"]]},
                                    f"it:{s}-{t}"))
    if "frag" in a.sets:
        fr = json.loads((HERE.parent / "eval/fragments/fragments.json").read_text(encoding="utf-8"))["items"]
        for it in fr:
            out.append(("frag", it["id"], it["src"], it["tgt"], it["text"], it["ref"],
                        {"must": it["must"], "must_not": it["must_not"], "kind": it["kind"]}, f"frag:{it['src']}"))
    return out


# ---------------------------------------------------------------- 依頼

def native_messages(family: str, src: str, tgt: str, text: str) -> "tuple[list | None, str | None]":
    """翻訳専用モデルの勧める依頼。(chat の messages, completion の prompt) のどちらか。"""
    if family == "hunyuan":
        if "zh" in (src, tgt):
            p = f"把下面的文本翻译成{ZH_NAMES[tgt]}，不要额外解释。\n\n{text}"
        else:
            p = f"Translate the following segment into {NAMES[tgt]}, without additional explanation.\n\n{text}"
        return [{"role": "user", "content": p}], None
    if family == "seedx":
        return None, f"Translate the following {NAMES[src]} sentence into {NAMES[tgt]}:\n{text} {SEEDX[tgt]}"
    if family == "tower":
        p = (f"Translate the following {NAMES[src]} source text to {NAMES[tgt]}:\n"
             f"{NAMES[src]}: {text}\n{NAMES[tgt]}: ")
        return [{"role": "user", "content": p}], None
    raise ValueError(family)


def _stream(base: str, model: str, msgs, prompt, timeout=90):
    """(訳, 初字の秒, 全体の秒)。chat は translate_stream(アプリと同じ)、completion は /v1/completions。"""
    import requests
    t0 = time.time()
    first = [0.0]
    if msgs is not None:
        from tsuyaku.translate import translate_stream

        def od(_d):
            if not first[0]:
                first[0] = time.time()
        try:
            out = translate_stream(base, model, msgs, od, timeout=timeout)
        except Exception as e:  # noqa: BLE001
            print(f"[err] {str(e)[:120]}", flush=True)
            out = ""
    else:
        acc = []
        try:
            r = requests.post(f"{base}/v1/completions", json={"model": model, "prompt": prompt, "stream": True,
                                                              "temperature": 0.0, "max_tokens": 512},
                              stream=True, timeout=timeout)
            for line in r.iter_lines():
                if not line or not line.startswith(b"data:"):
                    continue
                pl = line[5:].strip()
                if pl == b"[DONE]":
                    break
                try:
                    tx = json.loads(pl)["choices"][0].get("text") or ""
                except (ValueError, KeyError, IndexError):
                    continue
                if tx:
                    if not first[0]:
                        first[0] = time.time()
                    acc.append(tx)
        except Exception as e:  # noqa: BLE001
            print(f"[err] {str(e)[:120]}", flush=True)
        out = "".join(acc).strip().split("\n")[0].strip()
    return out, (first[0] - t0) if first[0] else -1.0, time.time() - t0


def run(a) -> None:
    from tsuyaku.translate import build_messages, configure_for_model
    configure_for_model(a.model)   # アプリと同じ: gemma なら断片の例を足す(TSU_SHOTTURNS で上書き)
    work = load_sets(a)
    convs: dict = {}
    for w in work:
        convs.setdefault(w[7], []).append(w)
    rows, lock = [], threading.Lock()

    def conv(seq):
        hist = []
        for st, key, s, t, text, ref, extra, _ck in seq:
            if a.prompt == "app":
                pair = (s, t)
                msgs = build_messages(pair, "", hist, text, s)
                out, ttft, tot = _stream(a.base, a.model, msgs, None)
                # アプリ(と eval_qual.run_translate)と同じ: 言語違い・破綻なら空の文脈で1回だけやり直す
                from tsuyaku.guards import looks_truncated, wrong_language
                if not out or wrong_language(text, out, t) or looks_truncated(text, out):
                    out2, _t2, tot2 = _stream(a.base, a.model, build_messages(pair, "", [], text, s), None)
                    tot += tot2
                    if out2 and not wrong_language(text, out2, t) and not looks_truncated(text, out2):
                        out = out2
                # アプリの LENGTH_RETRY と同じ(長さが外れたら空履歴で1回): TSU_LENRETRY=0 で切る
                import os
                from tsuyaku.guards import length_off
                if out and os.environ.get("TSU_LENRETRY", "1") == "1" and length_off(text, out, s, t):
                    out2, _t2, tot2 = _stream(a.base, a.model, build_messages(pair, "", [], text, s), None)
                    tot += tot2
                    if out2 and not length_off(text, out2, s, t) and not looks_truncated(text, out2):
                        out = out2
            else:
                m, p = native_messages(a.family, s, t, text)
                out, ttft, tot = _stream(a.base, a.model, m, p)
            if out:
                hist.append({"lang": s, "orig": text, "trans": out})
            with lock:
                rows.append({"set": st, "key": key, "src": s, "tgt": t, "text": text, "ref": ref, "trans": out,
                             "ttft": round(ttft, 3), "total": round(tot, 3), **extra})

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        list(ex.map(conv, convs.values()))
    meta = {"name": a.name, "base": a.base, "model": a.model, "prompt": a.prompt, "family": a.family,
            "workers": a.workers, "sets": a.sets, "sec": round(time.time() - t0, 1), "n": len(rows)}
    out = Path(a.out_dir) / f"{a.name}.json"
    out.write_text(json.dumps({"meta": meta, "rows": rows}, ensure_ascii=False, indent=0), encoding="utf-8")
    print(json.dumps(score_file(out), ensure_ascii=False), flush=True)


# ---------------------------------------------------------------- 物差し

def comet(a) -> None:
    """COMET の環境の python で動かす(eval_conv などは読まない)。発話ごとに row["comet"] を足す。"""
    from comet import download_model, load_from_checkpoint
    model = load_from_checkpoint(download_model(a.comet_model))
    for f in a.files:
        d = json.loads(Path(f).read_text(encoding="utf-8"))
        todo = [r for r in d["rows"] if "comet" not in r]
        if todo:
            data = [{"src": r["text"], "mt": r["trans"] or "", "ref": r["ref"]} for r in todo]
            sc = model.predict(data, batch_size=64, gpus=int(__import__("os").environ.get("COMET_GPUS", "1")), progress_bar=False).scores
            for r, s in zip(todo, sc):
                r["comet"] = round(float(s), 4)
            Path(f).write_text(json.dumps(d, ensure_ascii=False, indent=0), encoding="utf-8")
        print(f, "comet", round(100 * sum(r["comet"] for r in d["rows"]) / max(1, len(d["rows"])), 2), flush=True)


def score_file(path: Path, by: str = "set") -> dict:
    import numpy as np
    import eval_conv as EC
    from eval_terms import has_term
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = d["rows"]
    out = {}
    groups: dict = {}
    for r in rows:
        k = r["set"] if by == "set" else f"{r['set']}:{r['src']}-{r['tgt']}"
        groups.setdefault(k, []).append(r)
        if r["set"] == "qual" and r.get("ctx"):
            groups.setdefault("qual_ctx" if by == "set" else f"qual_ctx:{r['src']}-{r['tgt']}", []).append(r)
    for k, rs in sorted(groups.items()):
        st = None
        for r in rs:
            st = EC.add_stats(st, EC.chrf_stats(r["trans"], r["ref"]))
        g = {"n": len(rs), "chrF": round(EC.chrf_from_stats(st), 2), "empty": sum(1 for r in rs if not r["trans"])}
        cs = [r["comet"] for r in rs if "comet" in r]
        if cs:
            g["comet"] = round(100 * float(np.mean(cs)), 2)
        if rs[0]["set"] == "it":
            hit = tot = 0
            for r in rs:
                for spec in r.get("terms", []):
                    hit += has_term(r["trans"], spec)
                    tot += 1
            g["term"] = round(100 * hit / max(tot, 1), 1)
        if rs[0]["set"] == "frag":
            g["missing_must"] = sum(1 for r in rs if not all(has_term(r["trans"], m) for m in r["must"]))
            js = [r["judge"] for r in rs if isinstance(r.get("judge"), dict) and "added" in r["judge"]]
            if js:
                g["judge_added"] = sum(j["added"] for j in js)
                g["judge_omitted"] = sum(j["omitted"] for j in js)
        tt = [r["ttft"] for r in rs if r["ttft"] > 0]
        if tt:
            g["ttft_med"] = round(float(np.median(tt)), 3)
            g["ttft_p90"] = round(float(np.percentile(tt, 90)), 3)
            g["total_med"] = round(float(np.median([r["total"] for r in rs])), 3)
        out[k] = g
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--name", required=True)
    r.add_argument("--base", required=True)
    r.add_argument("--model", default="model")
    r.add_argument("--prompt", choices=["app", "native"], default="app")
    r.add_argument("--family", choices=["hunyuan", "seedx", "tower", ""], default="")
    r.add_argument("--sets", default="fleurs,qual,it,frag")
    r.add_argument("--fleurs", default="")
    r.add_argument("--qual", default="")
    r.add_argument("--n-fleurs", type=int, default=30)
    r.add_argument("--n-it", type=int, default=40)
    r.add_argument("--workers", type=int, default=2)
    r.add_argument("--out-dir", required=True)
    c = sub.add_parser("comet")
    c.add_argument("files", nargs="+")
    c.add_argument("--comet-model", default="Unbabel/wmt22-comet-da")
    s = sub.add_parser("score")
    s.add_argument("files", nargs="+")
    s.add_argument("--by", choices=["set", "dir"], default="set")
    a = ap.parse_args()
    if a.cmd == "run":
        a.sets = a.sets.split(",")
        run(a)
    elif a.cmd == "comet":
        comet(a)
    else:
        for f in a.files:
            print(Path(f).name, json.dumps(json.loads(Path(f).read_text())["meta"], ensure_ascii=False)[:200])
            for k, v in score_file(Path(f), a.by).items():
                print("  ", k, v)


if __name__ == "__main__":
    main()
