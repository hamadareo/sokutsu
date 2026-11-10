"""専門用語・略語の評価(11/08)。eval/terms/testset.json(74項目 × ja/en/zh/ko = 12方向)と
eval/terms/confusables.json(似た2文の続けての訳 = 前の訳の写し)を、アプリと同じ訳の依頼
(translate.build_messages)と耳(リモートサーバーの faster-whisper / このMacの SenseVoice)で通す。

  tts   : 項目ごと・言語ごとの音声を作る(ja = VOICEVOX 127.0.0.1:50021、en/zh/ko = say)
  stt   : 音声を耳に通し、聞き取りの CER/WER と用語の聞き取り率(字名を英字に戻す前/後)
  mt    : 12方向の訳(方向ごとに項目を順に、アプリと同じ履歴つき)。原文は正解の文か stt の結果
  cf    : 似た2文(A のあと B)の B の訳に、A の訳の語が写るか
  fleurs: FLEURS の並行文で、一般の文の chrF が下がらないか(12方向、履歴つき)
  score : mt / cf / fleurs の結果 → 表

A/B は --src-root で読む tsuyaku を切り替える(例: 変更前の worktree の src)。
リモートサーバーを使う前に、ユーザーのアプリが動いていないことを確かめる(ps に「-m tsuyaku」が無い)。
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import threading
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import eval_conv as EC  # noqa: E402  (chrF・CER・wav の読み書き)

import numpy as np  # noqa: E402

LANGS = ("ja", "en", "zh", "ko")
SAY = {"en": "Samantha", "zh": "Tingting", "ko": "Yuna"}
VV = (13, 2, 8, 11)          # VOICEVOX の声(項目ごとに回す)


def use_src(root: str | None) -> None:
    """読む tsuyaku を切り替える(eval_conv が入れた ROOT/src より前へ)。"""
    if root:
        sys.path.insert(0, str(Path(root).resolve()))
    for m in [m for m in sys.modules if m == "tsuyaku" or m.startswith("tsuyaku.")]:
        del sys.modules[m]


def apply_patches(patches) -> None:
    """--patch tsuyaku.translate:SIMILAR_HIST=drop(評価の切り替え。値は JSON か文字列)。"""
    import importlib
    for p in patches or ():
        mod, kv = p.split(":", 1)
        k, v = kv.split("=", 1)
        try:
            v = json.loads(v)
        except ValueError:
            pass
        setattr(importlib.import_module(mod), k, v)


def app_running() -> bool:
    out = subprocess.run(["ps", "ax", "-o", "command"], capture_output=True, text=True).stdout
    # python の命令で -m tsuyaku のものだけ(ほかの担当のシェルの命令文に含まれる文字に当てない。itterms 11/08)
    pat = re.compile(r"^\S*[Pp]ython[\d.]*\s+(?:-\S+\s+)*-m\s+tsuyaku(?:\s|$)")
    return any(pat.search(line.strip()) for line in out.splitlines())


def guard_app() -> None:
    if app_running():
        sys.exit("ユーザーのアプリ(-m tsuyaku)が動いています。remote/LM Studio の評価はしません。")


def alts(s: str) -> list:
    return [a for a in (s or "").split("|") if a]


def has_term(text: str, spec: str) -> bool:
    t = unicodedata.normalize("NFKC", text or "").lower()
    t2 = re.sub(r"\s+", "", t)
    for a in alts(spec):
        a = unicodedata.normalize("NFKC", a).lower()
        if a in t or re.sub(r"\s+", "", a) in t2:
            return True
    return False


def load_set(p: Path) -> list:
    d = json.loads(p.read_text(encoding="utf-8"))
    return d["items"] if isinstance(d, dict) else d


# ====================================================================== tts

def tts(args) -> None:
    import requests
    items = load_set(args.set)
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    for k, it in enumerate(items):
        for lang in LANGS:
            p = out / f"{it['id']}.{lang}.wav"
            if p.exists():
                continue
            text = it["texts"][lang]
            if args.langs and lang not in args.langs.split(","):
                continue
            if lang == "ja" and not args.ja_say:
                sp = VV[k % len(VV)]
                q = requests.post(f"{args.voicevox}/audio_query", params={"text": text, "speaker": sp},
                                  timeout=30).json()
                wav = requests.post(f"{args.voicevox}/synthesis", params={"speaker": sp}, json=q,
                                    timeout=60).content
                x = EC.wav_bytes_to_16k(wav)
            else:
                with tempfile.TemporaryDirectory() as d:
                    a = Path(d) / "a.aiff"
                    w = Path(d) / "a.wav"
                    voice = args.ja_say if lang == "ja" else SAY[lang]
                    subprocess.run(["say", "-v", voice, "-o", str(a), text], check=True)
                    subprocess.run(["afconvert", "-f", "WAVE", "-d", "LEI16@16000", "-c", "1",
                                    str(a), str(w)], check=True, capture_output=True)
                    x = EC.read_wav(w)
            x = np.concatenate([np.zeros(4000, np.float32), x.astype(np.float32), np.zeros(4000, np.float32)])
            EC.write_wav(p, x)
        print(f"[tts] {it['id']}", flush=True)


# ====================================================================== stt

def stt(args) -> None:
    use_src(args.src_root)
    from tsuyaku.terms import normalize_letters
    items = load_set(args.set)
    if args.ear == "remote":
        guard_app()
        from tsuyaku.stt import RemoteSTT
        ear = RemoteSTT(args.stt_url, None, timeout=15.0)
    else:
        from tsuyaku.config import load_settings
        from tsuyaku.stt import SenseVoiceSTT
        ear = SenseVoiceSTT(load_settings().sense_voice_dir)
    prompt = args.prompt or ""
    if args.prompt_mode == "bias":
        from tsuyaku.terms import stt_bias
        prompt = stt_bias("", ())
    rows = []
    for it in items:
        for lang in LANGS:
            if args.langs and lang not in args.langs.split(","):
                continue
            x = EC.read_wav(args.audio / f"{it['id']}.{lang}.wav")
            ear.langs = (lang,)
            if hasattr(ear, "prompt"):
                ear.prompt = prompt
            t0 = time.time()
            text, got, prob = ear.transcribe(x)
            sec = time.time() - t0
            rows.append({"id": it["id"], "lang": lang, "heard": text, "norm": normalize_letters(text, lang),
                         "ref": it["texts"][lang], "sec": round(sec, 3), "prob": prob,
                         "terms": [t[lang] for t in it["terms"]],
                         "kinds": [t["kind"] for t in it["terms"]]})
        print(f"[stt] {it['id']} {rows[-1]['norm'][:40]}", flush=True)
    args.out.write_text(json.dumps({"ear": args.ear, "prompt": prompt, "rows": rows},
                                   ensure_ascii=False, indent=0), encoding="utf-8")


def score_stt(d: dict) -> dict:
    out: dict = {}
    for lang in LANGS:
        rs = [r for r in d["rows"] if r["lang"] == lang]
        if not rs:
            continue
        for key in ("heard", "norm"):
            e = n = 0
            hit = tot = ahit = atot = 0
            for r in rs:
                ref = EC.asr_tokens(r["ref"], lang)
                hyp = EC.asr_tokens(r[key], lang)
                e += EC.edit_distance(ref, hyp)
                n += len(ref)
                for spec, kind in zip(r["terms"], r["kinds"]):
                    h = has_term(r[key], spec)
                    hit += h
                    tot += 1
                    if kind == "acronym":
                        ahit += h
                        atot += 1
            out[(lang, key)] = {"err": 100 * e / max(n, 1), "term": 100 * hit / max(tot, 1),
                                "acr": 100 * ahit / max(atot, 1), "n_term": tot, "n_acr": atot,
                                "sec": float(np.mean([r["sec"] for r in rs]))}
    return out


# ====================================================================== mt

def _llm(args):
    if getattr(args, "base", None):          # 自分で立てたサーバー(リモートサーバーの llama-server など)
        return args.base, getattr(args, "model", None) or "model"
    if args.engine == "remote":
        return "http://127.0.0.1:12345", getattr(args, "model", None) or "qwen3.8-flash-next"
    return "http://127.0.0.1:1234", getattr(args, "model", None) or "google/gemma-3-4b"


def _translate(base, model, msgs):
    from tsuyaku.translate import translate_stream
    t0 = time.time()
    first = [0.0]

    def od(_d):
        if not first[0]:
            first[0] = time.time()
    try:
        out = translate_stream(base, model, msgs, od, timeout=60)
    except Exception as e:  # noqa: BLE001
        out = ""
        print(f"[mt-err] {e}", flush=True)
    return out, (first[0] - t0) if first[0] else -1.0, time.time() - t0


def _direction(args, base, model, seq, src, tgt, lock, rows, terms=()):
    """1方向: 項目を順に、アプリと同じ履歴(直近4〜7件)つきで訳す。"""
    from tsuyaku.translate import build_messages
    hist: list = []
    for key, text, extra in seq:
        t_b = time.time()
        msgs = build_messages((src, tgt), args.topic, hist if args.hist else [], text, src,
                              **({"terms": terms} if terms else {}))
        t_build = time.time() - t_b
        trans, ttft, total = _translate(base, model, msgs)
        if trans:
            hist.append({"lang": src, "orig": text, "trans": trans})
        r = {"key": key, "src": src, "tgt": tgt, "text": text, "trans": trans, "ttft": round(ttft, 3),
             "total": round(total, 3), "build_ms": round(1000 * t_build, 2),
             "prompt_chars": sum(len(m["content"]) for m in msgs), **extra}
        with lock:
            rows.append(r)


def mt(args) -> None:
    guard_app()
    use_src(args.src_root)
    apply_patches(args.patch)
    base, model = _llm(args)
    items = load_set(args.set)
    heard = {}
    if args.heard:
        for f in args.heard:
            for r in json.loads(Path(f).read_text(encoding="utf-8"))["rows"]:
                heard[(r["id"], r["lang"])] = r["heard"]     # 耳の生の文字(直しはアプリ側の仕事)
    rows: list = []
    lock = threading.Lock()
    jobs = []
    for src in LANGS:
        for tgt in LANGS:
            if src == tgt or (args.dirs and f"{src}-{tgt}" not in args.dirs.split(",")):
                continue
            seq = []
            for it in items:
                text = heard.get((it["id"], src), it["texts"][src]) if args.heard else it["texts"][src]
                if not text:
                    continue
                seq.append((it["id"], text, {"ref": it["texts"][tgt],
                                             "terms": [t[tgt] for t in it["terms"]],
                                             "kinds": [t["kind"] for t in it["terms"]]}))
            jobs.append((seq, src, tgt))
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for f in [ex.submit(_direction, args, base, model, seq, s, t, lock, rows) for seq, s, t in jobs]:
            f.result()
    meta = {"engine": args.engine, "model": model, "src_root": args.src_root or "this", "hist": args.hist,
            "patch": args.patch,
            "heard": bool(args.heard), "sec": round(time.time() - t0, 1), "git": git_rev(args.src_root)}
    args.out.write_text(json.dumps({"meta": meta, "rows": rows}, ensure_ascii=False, indent=0), encoding="utf-8")
    print(json.dumps(score_mt({"rows": rows}), ensure_ascii=False))


def git_rev(root) -> str:
    d = Path(root).resolve().parent if root else HERE.parent
    try:
        r = subprocess.run(["git", "-C", str(d), "rev-parse", "--short", "HEAD"], capture_output=True, text=True)
        dirty = subprocess.run(["git", "-C", str(d), "status", "--porcelain", "src"], capture_output=True,
                               text=True).stdout.strip()
        return r.stdout.strip() + ("+dirty" if dirty else "")
    except OSError:
        return "?"


def score_mt(d: dict, by_dir: bool = False) -> dict:
    rows = d["rows"]
    out: dict = {}

    def agg(rs):
        st = None
        hit = tot = ahit = atot = 0
        for r in rs:
            st = EC.add_stats(st, EC.chrf_stats(r["trans"], r["ref"]))
            for spec, kind in zip(r.get("terms", []), r.get("kinds", [])):
                h = has_term(r["trans"], spec)
                hit += h
                tot += 1
                if kind == "acronym":
                    ahit += h
                    atot += 1
        tt = [r["ttft"] for r in rs if r["ttft"] > 0]
        return {"n": len(rs), "chrF": round(EC.chrf_from_stats(st), 2) if st else 0.0,
                "term": round(100 * hit / max(tot, 1), 1), "acr": round(100 * ahit / max(atot, 1), 1),
                "n_term": tot, "n_acr": atot, "empty": sum(1 for r in rs if not r["trans"]),
                "ttft_med": round(float(np.median(tt)), 3) if tt else -1,
                "total_med": round(float(np.median([r["total"] for r in rs])), 3),
                "build_ms_max": max((r.get("build_ms", 0) for r in rs), default=0),
                "prompt_chars": round(float(np.mean([r.get("prompt_chars", 0) for r in rs])), 0)}
    out["all"] = agg(rows)
    if by_dir:
        for src in LANGS:
            for tgt in LANGS:
                rs = [r for r in rows if r["src"] == src and r["tgt"] == tgt]
                if rs:
                    out[f"{src}-{tgt}"] = agg(rs)
    return out


# ====================================================================== cf

def cf(args) -> None:
    guard_app()
    use_src(args.src_root)
    apply_patches(args.patch)
    from tsuyaku.translate import build_messages
    base, model = _llm(args)
    pairs = load_set(args.set)
    rows: list = []
    lock = threading.Lock()

    def one(p, tgt, order):
        first, second = (p["a"], p["b"]) if order == "ab" else (p["b"], p["a"])
        src = p["src"]
        t1, *_ = _translate(base, model, build_messages((src, tgt), "", [], first["text"], src))
        hist = [{"lang": src, "orig": first["text"], "trans": t1}] if t1 else []
        t2, ttft, _ = _translate(base, model, build_messages((src, tgt), "", hist, second["text"], src))
        r = {"id": p["id"], "src": src, "tgt": tgt, "order": order, "first": first["text"], "t1": t1,
             "second": second["text"], "t2": t2, "ok": has_term(t2, second["key"][tgt]),
             "leak": has_term(t2, first["key"][tgt]) and not has_term(second["ref"][tgt], first["key"][tgt]),
             "first_ok": has_term(t1, first["key"][tgt]), "ttft": ttft}
        with lock:
            rows.append(r)

    jobs = [(p, tgt, o) for p in pairs for tgt in LANGS if tgt != p["src"] for o in ("ab", "ba")]
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        list(ex.map(lambda j: one(*j), jobs))
    args.out.write_text(json.dumps({"meta": {"engine": args.engine, "git": git_rev(args.src_root),
                                             "patch": args.patch},
                                    "rows": rows}, ensure_ascii=False, indent=0), encoding="utf-8")
    print(json.dumps(score_cf({"rows": rows}), ensure_ascii=False))


def score_cf(d: dict) -> dict:
    rs = d["rows"]
    n = len(rs)
    return {"n": n, "second_ok": round(100 * sum(r["ok"] for r in rs) / max(n, 1), 1),
            "leak": sum(r["leak"] for r in rs),
            "first_ok": round(100 * sum(r["first_ok"] for r in rs) / max(n, 1), 1)}


# ====================================================================== fleurs

def fleurs(args) -> None:
    guard_app()
    use_src(args.src_root)
    apply_patches(args.patch)
    base, model = _llm(args)
    tsv = {lang: {r["sid"]: r for r in EC.load_tsv(args.fleurs, lang)} for lang in LANGS}
    ids = sorted(set.intersection(*[set(v) for v in tsv.values()]))[: args.n]
    rows: list = []
    lock = threading.Lock()
    jobs = []
    for src in LANGS:
        for tgt in LANGS:
            if src != tgt:
                seq = [(i, tsv[src][i]["text"], {"ref": tsv[tgt][i]["text"]})
                       for i in ids]
                jobs.append((seq, src, tgt))
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for f in [ex.submit(_direction, args, base, model, seq, s, t, lock, rows) for seq, s, t in jobs]:
            f.result()
    args.out.write_text(json.dumps({"meta": {"engine": args.engine, "git": git_rev(args.src_root), "n": len(ids),
                                             "patch": args.patch},
                                    "rows": rows}, ensure_ascii=False, indent=0), encoding="utf-8")
    print(json.dumps(score_mt({"rows": rows}), ensure_ascii=False))


# ====================================================================== score

def score(args) -> None:
    for p in args.files:
        d = json.loads(Path(p).read_text(encoding="utf-8"))
        rows = d.get("rows", [])
        if rows and "second" in rows[0]:
            print(p, d.get("meta"), json.dumps(score_cf(d), ensure_ascii=False))
        elif rows and "heard" in rows[0]:
            for (lang, key), v in score_stt(d).items():
                print(p, lang, key, {k: round(x, 2) if isinstance(x, float) else x for k, x in v.items()})
        else:
            sc = score_mt(d, by_dir=args.by_dir)
            print(p, d.get("meta"))
            for k, v in sc.items():
                print(f"  {k}: {v}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("tts")
    t.add_argument("--set", type=Path, default=HERE.parent / "eval/terms/testset.json")
    t.add_argument("--out", type=Path, required=True)
    t.add_argument("--voicevox", default="http://127.0.0.1:50021")
    t.add_argument("--ja-say", default="", help="日本語も say の声で(例 Kyoko)")
    t.add_argument("--langs", default="")
    s = sub.add_parser("stt")
    s.add_argument("--set", type=Path, default=HERE.parent / "eval/terms/testset.json")
    s.add_argument("--audio", type=Path, required=True)
    s.add_argument("--out", type=Path, required=True)
    s.add_argument("--ear", choices=["remote", "mac"], default="remote")
    s.add_argument("--stt-url", default="http://127.0.0.1:12346")
    s.add_argument("--prompt", default="")
    s.add_argument("--prompt-mode", choices=["fixed", "bias"], default="fixed")
    s.add_argument("--langs", default="")
    s.add_argument("--src-root", default=None)
    for name in ("mt", "cf", "fleurs"):
        m = sub.add_parser(name)
        m.add_argument("--out", type=Path, required=True)
        m.add_argument("--engine", choices=["remote", "mac"], default="remote")
        m.add_argument("--src-root", default=None)
        m.add_argument("--workers", type=int, default=2)
        m.add_argument("--topic", default="")
        m.add_argument("--hist", action="store_true", default=True)
        m.add_argument("--no-hist", dest="hist", action="store_false")
        m.add_argument("--dirs", default="")
        m.add_argument("--model", default=None, help="LLM のモデル名(既定: リモートサーバー qwen3.8-flash-next / mac gemma-3-4b)")
        m.add_argument("--base", default=None, help="LLM の URL(例 http://127.0.0.1:9961。自分で立てた llama-server)")
        m.add_argument("--patch", nargs="*", default=None, help="mod:ATTR=値(例 tsuyaku.translate:SIMILAR_HIST=drop)")
        if name == "mt":
            m.add_argument("--set", type=Path, default=HERE.parent / "eval/terms/testset.json")
            m.add_argument("--heard", nargs="*", default=None, help="stt の結果(原文を耳の文字にする)")
        if name == "cf":
            m.add_argument("--set", type=Path, default=HERE.parent / "eval/terms/confusables.json")
        if name == "fleurs":
            m.add_argument("--fleurs", type=Path, required=True)
            m.add_argument("--n", type=int, default=40)
    sc = sub.add_parser("score")
    sc.add_argument("files", nargs="+")
    sc.add_argument("--by-dir", action="store_true")
    a = ap.parse_args()
    {"tts": tts, "stt": stt, "mt": mt, "cf": cf, "fleurs": fleurs, "score": score}[a.cmd](a)


if __name__ == "__main__":
    main()
