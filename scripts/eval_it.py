"""IT 用語の評価(11/08 2回目「もっとIT用語や専門用語に強くなって」)。

eval/terms/it_testset.json(220項目 × ja/en/zh/ko、クラウド・コンテナ・ネットワーク・セキュリティ・AI/ML・
データベース・プログラミング・半導体/ハード・IoT・開発の進め方)を、eval_terms.py と同じ訳の依頼・耳に通す。
用語は種類(acronym = 字名の略語 / product = 製品名 / loan = カタカナ語 / term = 漢字などの用語)ごとに数える。

  tts   : 音声(ja = VOICEVOX・say Kyoko/Eddy/Flo、en = say Samantha・Kokoro、zh = say Tingting/Reed/Flo、
          ko = say Yuna/Reed/Flo。Kokoro の ja/zh は英字の語を読み飛ばすので使わない)。項目ごとに声を回す。"speak" があればその読み(製品名のカタカナ読み)を読ませる
  stt   : 耳(remote whisper / このMac SenseVoice)→ 聞き取りの CER/WER と用語の当たり(耳の直しの前/後)
  mt    : eval_terms.mt と同じ(--set を it_testset に)
  score : 種類ごとの表

ユーザーの発言は使わない(文はすべて作った例)。音声・結果は git に入れない。
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import eval_conv as EC  # noqa: E402
import eval_terms as ET  # noqa: E402

import numpy as np  # noqa: E402

LANGS = ET.LANGS


def app_running() -> bool:
    """ユーザーのアプリ(python -m tsuyaku)が動いているか。シェルの命令文に「-m tsuyaku」という字が
    入っているだけのもの(ps を grep する他の担当のシェルなど)は数えない。"""
    out = subprocess.run(["ps", "ax", "-o", "command"], capture_output=True, text=True).stdout
    pat = re.compile(r"^\S*[Pp]ython[\d.]*\s+(?:-\S+\s+)*-m\s+tsuyaku(?:\s|$)")
    return any(pat.search(line) for line in out.splitlines())


ET.app_running = app_running
SET = HERE.parent / "eval/terms/it_testset.json"
KINDS = ("acronym", "product", "loan", "term")
VV = (13, 2, 8, 11, 3, 14)
# Kokoro は en だけ(ja/zh は辞書に無い字・英字の語を読み飛ばす = 製品名が音から消える)
KOKORO_SID = {"en": (3, 6, 15, 20)}
# 同じ名前の声が言語ごとにある(Flo・Reed・Eddy)。名前だけだと英語の声になり、英字の語しか読まない。
# 言語つきの名前はシステムの言語(このMacは日本語)で書く
_LOC = {"ja": "日本語（日本）", "zh": "中国語（中国本土）", "ko": "韓国語（韓国）"}
SAY2 = {"ja": ("Kyoko", f"Eddy ({_LOC['ja']})", f"Flo ({_LOC['ja']})"), "en": ("Samantha",),
        "zh": ("Tingting", f"Reed ({_LOC['zh']})", f"Flo ({_LOC['zh']})"),
        "ko": ("Yuna", f"Reed ({_LOC['ko']})", f"Flo ({_LOC['ko']})")}


def voice_for(lang: str, k: int) -> tuple:
    """項目 k・言語 lang の声。ja は半分 VOICEVOX、残りを say で。en は say と Kokoro。"""
    if lang == "ja":
        if k % 2 == 0:
            return ("voicevox", VV[(k // 2) % len(VV)])
        return ("say", SAY2["ja"][(k // 2) % len(SAY2["ja"])])
    if lang in ("ko", "zh"):
        return ("say", SAY2[lang][k % len(SAY2[lang])])
    if k % 2 == 0:
        return ("say", SAY2[lang][0])
    sids = KOKORO_SID[lang]
    return ("kokoro", sids[(k // 2) % len(sids)])


def _say_wav(text: str, voice: str) -> np.ndarray:
    with tempfile.TemporaryDirectory() as d:
        a, w = Path(d) / "a.aiff", Path(d) / "a.wav"
        r = subprocess.run(["say", "-v", voice, "-o", str(a), text], capture_output=True)
        if r.returncode != 0 or not a.exists():
            raise RuntimeError(f"say -v {voice} failed")
        subprocess.run(["afconvert", "-f", "WAVE", "-d", "LEI16@16000", "-c", "1", str(a), str(w)],
                       check=True, capture_output=True)
        return EC.read_wav(w)


def tts(args) -> None:
    import requests
    items = ET.load_set(args.set)
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    kok = None
    voices = {}
    for k, it in enumerate(items):
        for lang in LANGS:
            if args.langs and lang not in args.langs.split(","):
                continue
            p = out / f"{it['id']}.{lang}.wav"
            if p.exists():
                continue
            text = (it.get("speak") or {}).get(lang) or it["texts"][lang]
            kind, val = voice_for(lang, k)
            if kind == "voicevox":
                q = requests.post(f"{args.voicevox}/audio_query", params={"text": text, "speaker": val},
                                  timeout=30).json()
                wav = requests.post(f"{args.voicevox}/synthesis", params={"speaker": val}, json=q,
                                    timeout=60).content
                x = EC.wav_bytes_to_16k(wav)
            elif kind == "kokoro":
                if kok is None:
                    from tsuyaku.config import load_settings
                    from tsuyaku.tts import TTS
                    kok = TTS(load_settings().tts_dir_kokoro, "", {})
                kok.voices[lang] = f"kokoro:{val}"
                wav = kok._synth(text, lang, 1.0)
                if not wav:
                    kind, val = "say", SAY2[lang][0]
                    x = _say_wav(text, val)
                else:
                    x = EC.wav_bytes_to_16k(wav)
            else:
                x = _say_wav(text, val)
            voices[f"{it['id']}.{lang}"] = f"{kind}:{val}"
            x = np.concatenate([np.zeros(4000, np.float32), x.astype(np.float32), np.zeros(4000, np.float32)])
            EC.write_wav(p, x)
        print(f"[tts] {it['id']}", flush=True)
    vf = out / "voices.json"
    old = json.loads(vf.read_text()) if vf.exists() else {}
    old.update(voices)
    vf.write_text(json.dumps(old, ensure_ascii=False, indent=0))


def stt(args) -> None:
    ET.use_src(args.src_root)
    from tsuyaku.terms import clean_heard
    items = ET.load_set(args.set)
    ET.guard_app()          # ユーザーのアプリが動いていたら測らない(remote / このMac の取り合い)
    if args.ear == "remote":
        from tsuyaku.stt import RemoteSTT
        ear = RemoteSTT(args.stt_url, None, timeout=20.0)
    elif args.ear == "macwhisper":     # このMacの whisper(聞き直し relisten.py の耳)
        from tsuyaku.config import load_settings
        from tsuyaku.stt import STT
        ear = STT(load_settings().whisper_model)
    else:
        from tsuyaku.config import load_settings
        from tsuyaku.stt import SenseVoiceSTT
        ear = SenseVoiceSTT(load_settings().sense_voice_dir)
    rows = []
    part = args.out.with_suffix(".partial.jsonl")     # 途中でアプリが起きたら止め、次はここから続ける
    done = {}
    if part.exists():
        for line in part.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            done[(r["id"], r["lang"])] = r
    pf = part.open("a", encoding="utf-8")
    for it in items:
        if ET.app_running():
            pf.close()
            sys.exit("ユーザーのアプリが起きたので止めました(続きは同じコマンドで)。")
        for lang in LANGS:
            if args.langs and lang not in args.langs.split(","):
                continue
            if (it["id"], lang) in done:
                rows.append(done[(it["id"], lang)])
                continue
            x = EC.read_wav(args.audio / f"{it['id']}.{lang}.wav")
            ear.langs = (lang,)
            if hasattr(ear, "prompt"):
                ear.prompt = args.prompt_by_lang.get(lang, "") if args.prompt_by_lang else ""
            t0 = time.time()
            for k in range(4):        # リモートサーバーの耳がたまに応答しない(ローカルの代わりが無いと落ちる)
                try:
                    text, got, prob = ear.transcribe(x)
                    break
                except AttributeError:
                    if k == 3:
                        raise
                    time.sleep(3)
            sec = time.time() - t0
            rows.append({"id": it["id"], "lang": lang, "heard": text, "norm": clean_heard(text, lang),
                         "ref": it["texts"][lang], "sec": round(sec, 3),
                         "terms": [t[lang] for t in it["terms"]], "kinds": [t["kind"] for t in it["terms"]]})
            pf.write(json.dumps(rows[-1], ensure_ascii=False) + "\n")
            pf.flush()
        print(f"[stt] {it['id']} {rows[-1]['norm'][:40]}", flush=True)
    pf.close()
    args.out.write_text(json.dumps({"ear": args.ear, "rows": rows}, ensure_ascii=False, indent=0),
                        encoding="utf-8")
    part.unlink(missing_ok=True)
    print_stt(json.loads(args.out.read_text()))


def renorm(args) -> None:
    """stt の結果の耳の文字(heard)を、--src-root の tsuyaku の耳の直し(clean_heard)で直し直す
    (耳は同じ・直しだけ前後で比べる)。"""
    ET.use_src(args.src_root)
    ET.apply_patches(args.patch)
    from tsuyaku.terms import clean_heard
    d = json.loads(Path(args.inp).read_text(encoding="utf-8"))
    for r in d["rows"]:
        r["norm"] = clean_heard(r["heard"], r["lang"])
    d["renorm"] = ET.git_rev(args.src_root)
    Path(args.out).write_text(json.dumps(d, ensure_ascii=False, indent=0), encoding="utf-8")
    print_stt(d)


def load(args) -> None:
    """聞き直しの負荷: ja の文を実時間の間隔で流し、言い終わりごとにこのMacの訳(LM Studio)を頼む。
    見当が当たった文は裏で このMacの whisper に聞き直させる(relisten.Relistener そのもの、訳の最中は待つ)。
    訳の初字の時間(中央/p90)を、見当なし / 前の見当 / 新しい見当 で比べる(GPU の取り合い)。"""
    import threading as th
    ET.guard_app()
    from tsuyaku import relisten as R
    from tsuyaku import terms
    from tsuyaku.config import load_settings
    from tsuyaku.stt import STT
    from tsuyaku.translate import build_messages
    sv = [r for r in json.loads(Path(args.heard).read_text())["rows"] if r["lang"] == "ja"][: args.n]

    def old_cue(t, ctx):
        t = t.strip()
        return bool(R._START.match(t) or R._CLAUSE.search(t) or R._KONO.search(t) or R._PAIR.search(t))

    cues = {"none": lambda t, c: False, "old": old_cue,
            "new": lambda t, c: R.dropped_letters(t, "ja", it=terms.is_it_context(t, "ja", context=c))}
    whisper = STT(load_settings().whisper_model)
    out = {}
    for ci, name in enumerate(args.conds.split(",")):
        cue = cues[name]
        busy = th.Event()
        rl = R.Relistener(make_ear=lambda: whisper, on_fixed=lambda j, n: None, on_done=lambda j: None,
                          busy=busy.is_set)
        hist: list = []
        ttft: list = []
        trig = 0
        t0 = time.time()
        clock = 0.0
        prev = ""
        for r in sv:
            x = EC.read_wav(args.audio / f"{r['id']}.ja.wav")
            clock += len(x) / 16000 + 0.4
            wait = clock - (time.time() - t0)
            if wait > 0:
                time.sleep(wait)          # 言い終わりまで待つ(実時間)
            if cue(r["norm"], prev):
                trig += 1
                rl.submit({"pcm": x, "lang": "ja", "orig": r["norm"], "uid": r["id"]})
            prev = r["norm"]
            busy.set()
            msgs = build_messages(("ja", "en"), "", hist, r["norm"], "ja")
            tr, first, _total = ET._translate("http://127.0.0.1:1234", "google/gemma-3-4b", msgs)
            busy.clear()
            ttft.append(first)
            if tr:
                hist.append({"lang": "ja", "orig": r["norm"], "trans": tr})
        time.sleep(3)
        rl.close()
        tt = sorted(v for v in ttft if v > 0)
        out[f"{name}#{ci}"] = {"n": len(sv), "relisten": trig, "ttft_med": round(float(np.median(tt)), 3),
                     "ttft_p90": round(float(np.percentile(tt, 90)), 3), "relisten_sec": rl.stats["sec"]}
        print(name, {k: v for k, v in out[f"{name}#{ci}"].items() if k != "relisten_sec"},
              "聞き直しの秒 中央", round(float(np.median(rl.stats["sec"])), 2) if rl.stats["sec"] else "-", flush=True)
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=1))


def score_stt(d: dict, key: str = "norm") -> dict:
    out: dict = {}
    for lang in LANGS:
        rs = [r for r in d["rows"] if r["lang"] == lang]
        if not rs:
            continue
        e = n = 0
        hit = {k: [0, 0] for k in KINDS + ("all",)}
        for r in rs:
            ref = EC.asr_tokens(r["ref"], lang)
            e += EC.edit_distance(ref, EC.asr_tokens(r.get(key) or "", lang))
            n += len(ref)
            for spec, kind in zip(r["terms"], r["kinds"]):
                h = ET.has_term(r.get(key) or "", spec)
                for kk in (kind, "all"):
                    hit[kk][0] += h
                    hit[kk][1] += 1
        out[lang] = {"err": round(100 * e / max(n, 1), 2),
                     **{k: round(100 * a / b, 1) if b else None for k, (a, b) in hit.items()},
                     "n": {k: b for k, (a, b) in hit.items()}}
    return out


def print_stt(d: dict) -> None:
    for key in ("heard", "norm"):
        for lang, v in score_stt(d, key).items():
            print(f"  {key:5s} {lang} err {v['err']:5.2f}  all {v['all']}  "
                  + "  ".join(f"{k} {v[k]}" for k in KINDS), flush=True)


def score_mt(d: dict, by_dir: bool = False) -> dict:
    def agg(rs):
        st = None
        hit = {k: [0, 0] for k in KINDS + ("all",)}
        for r in rs:
            st = EC.add_stats(st, EC.chrf_stats(r["trans"], r["ref"]))
            for spec, kind in zip(r.get("terms", []), r.get("kinds", [])):
                h = ET.has_term(r["trans"], spec)
                for kk in (kind, "all"):
                    hit[kk][0] += h
                    hit[kk][1] += 1
        tt = [r["ttft"] for r in rs if r["ttft"] > 0]
        return {"n": len(rs), "chrF": round(EC.chrf_from_stats(st), 2) if st else 0.0,
                **{k: round(100 * a / b, 1) if b else None for k, (a, b) in hit.items()},
                "empty": sum(1 for r in rs if not r["trans"]),
                "ttft_med": round(float(np.median(tt)), 3) if tt else -1,
                "build_ms_max": max((r.get("build_ms", 0) for r in rs), default=0),
                "build_ms_med": round(float(np.median([r.get("build_ms", 0) for r in rs])), 2)}
    rows = d["rows"]
    out = {"all": agg(rows)}
    if by_dir:
        for s in LANGS:
            for t in LANGS:
                rs = [r for r in rows if r["src"] == s and r["tgt"] == t]
                if rs:
                    out[f"{s}-{t}"] = agg(rs)
        for s in LANGS:
            rs = [r for r in rows if r["src"] == s]
            if rs:
                out[f"{s}->"] = agg(rs)
    return out


def score(args) -> None:
    for p in args.files:
        d = json.loads(Path(p).read_text(encoding="utf-8"))
        rows = d.get("rows", [])
        if rows and "heard" in rows[0]:
            print(p, d.get("ear"))
            print_stt(d)
        else:
            print(p, d.get("meta"))
            for k, v in score_mt(d, by_dir=args.by_dir).items():
                print(f"  {k}: {v}")


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "et":
        # eval_terms.py の mt / cf / fleurs を、こちらのアプリの見張り(app_running)で流す
        sys.argv = ["eval_terms.py"] + sys.argv[2:]
        # 訳のサーバーを差し替える(11/08 方針: 品質の評価は リモートサーバー。Mac のモデルも リモートサーバーの自分の llama-server で)
        # TSUYAKU_EVAL_LLM="http://127.0.0.1:9981|gemma-3-4b"
        if os.environ.get("TSUYAKU_EVAL_LLM"):
            url, _, model = os.environ["TSUYAKU_EVAL_LLM"].partition("|")
            ET._llm = lambda a: (url, model or "local")
        ET.main()
        return
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("tts")
    t.add_argument("--set", type=Path, default=SET)
    t.add_argument("--out", type=Path, required=True)
    t.add_argument("--voicevox", default="http://127.0.0.1:50021")
    t.add_argument("--langs", default="")
    s = sub.add_parser("stt")
    s.add_argument("--set", type=Path, default=SET)
    s.add_argument("--audio", type=Path, required=True)
    s.add_argument("--out", type=Path, required=True)
    s.add_argument("--ear", choices=["remote", "mac", "macwhisper"], default="remote")
    s.add_argument("--stt-url", default="http://127.0.0.1:12346")
    s.add_argument("--prompt-json", default="", help='{"ja": "手がかり", ...}(言語ごとの耳の手がかり)')
    s.add_argument("--langs", default="")
    s.add_argument("--src-root", default=None)
    ld = sub.add_parser("load")
    ld.add_argument("--heard", required=True)
    ld.add_argument("--audio", type=Path, required=True)
    ld.add_argument("--out", required=True)
    ld.add_argument("--n", type=int, default=40)
    ld.add_argument("--conds", default="none,old,new,none,old,new")
    rn = sub.add_parser("renorm")
    rn.add_argument("inp")
    rn.add_argument("out")
    rn.add_argument("--src-root", default=None)
    rn.add_argument("--patch", nargs="*", default=None)
    sc = sub.add_parser("score")
    sc.add_argument("files", nargs="+")
    sc.add_argument("--by-dir", action="store_true")
    a = ap.parse_args()
    if a.cmd == "stt":
        a.prompt_by_lang = json.loads(a.prompt_json) if a.prompt_json else {}
    {"tts": tts, "stt": stt, "score": score, "renorm": renorm, "load": load}[a.cmd](a)


if __name__ == "__main__":
    main()
