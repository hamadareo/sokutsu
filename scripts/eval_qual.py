"""通訳の品質がどこで落ちるかを4つに分けて測る(11/08「品質の高い翻訳をリアルタイムで」の土台 A)。

試験セット(eval/qual/dialogs/*.json): 作った対面の対話 9 本(ja↔zh・ja↔en・ja↔ko × 雑談・学会の質疑・IT)。
各発話に、文脈を解いた参照訳と「文脈がないと訳せないか(ctx)」の印がある。

比べる条件(同じ対話・同じ訳のモデルで):
  full    本番と同じ: 対話の音声を流し、耳 → 区切り(StreamSegmenter + Endpointer)→ 訳(前の文脈つき)
  seg     正しい区切り: 発話ごとの音声を耳で聞き → 訳(前の文脈つき)
  text    正しい原文: 正解の原文 → 訳(前の文脈つき。文脈は正解の原文 + その時の訳)
  noctx   正しい原文・文脈なし
  fullctx 正しい原文・前と後ろの文脈あり(リアルタイムでは使えない上限: 後ろの1発話も見せる)
品質の落ち方:
  耳の誤り   = text − seg
  区切りの誤り = seg − full
  文脈不足   = fullctx − text(前後が見えれば直る分)。noctx − text は今の文脈の効き
  訳の誤り   = 強いモデルの text − 今のモデルの text(B と合わせて)。単独では text の点そのもの
物差し: COMET(Unbabel/wmt22-comet-da、Apache-2.0、原文・訳・参照訳)と chrF。発話ごとに比べる
(full は単位の時刻から発話に割り当てて繋ぐ)。

実行:
  build  (Mac。say の声): .venv/bin/python -I scripts/eval_qual.py build --out eval/qual/audio
  以降は計算機(node / remote)で。eval_qual.py run / score(このあと足す)。
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import wave
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SR = 16000
DIALOGS = ROOT / "eval" / "qual" / "dialogs"
# 話し手 A(日本語)と B の声。対話ごとに入れ替える(声による耳の差を均す)
VOICES = {"ja": ["Kyoko", "Eddy (日本語（日本）)", "Flo (日本語（日本）)"],
          "zh": ["Tingting", "Eddy (中国語（中国本土）)", "Flo (中国語（中国本土）)"],
          "en": ["Samantha", "Daniel", "Eddy (英語（アメリカ）)"],
          "ko": ["Yuna", "Eddy (韓国語（韓国）)", "Flo (韓国語（韓国）)"]}
DOMAINS = ["chat", "qa", "it"]


def load_dialogs() -> list[dict]:
    out = []
    for p in sorted(DIALOGS.glob("*.json")):
        d = json.loads(p.read_text(encoding="utf-8"))
        d["name"] = p.stem
        out.append(d)
    return out


def _say(text: str, voice: str, path: str) -> np.ndarray:
    subprocess.run(["say", "-v", voice, "-o", path, "--data-format=LEI16@16000", text], check=True)
    with wave.open(path) as w:
        a = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768
    nz = np.where(np.abs(a) > 0.01)[0]
    return a[max(0, nz[0] - 800): nz[-1] + 800] if len(nz) else a


def _clauses(text: str) -> list[str]:
    """読点・文末で区切った節(息つぎの間を入れる所)。"""
    parts = re.split(r"(?<=[、，,。．.！？!?])\s*", text)
    return [x for x in parts if x.strip()]


def build(out: Path, breath: bool = False, breath_lo: float = 0.35, breath_hi: float = 0.9,
          gap_lo: float = 0.3, gap_hi: float = 0.9) -> None:
    """対話ごとに1本の WAV(発話の間 gap_lo〜gap_hi 秒)と、発話ごとの時刻を書いた JSON。
    breath=True: 発話の中の読点・文末で breath_lo〜breath_hi 秒の息つぎの間を入れる(節ごとに声にして繋ぐ。
    11/10 間の評価: 耳が息つぎの断片に句点を付ける場面を作る)。"""
    out.mkdir(parents=True, exist_ok=True)
    (out / ".gitignore").write_text("*\n")
    rng = np.random.default_rng(7)
    with tempfile.TemporaryDirectory() as td:
        for d in load_dialogs():
            k = DOMAINS.index(d["domain"]) if d["domain"] in DOMAINS else 0
            other = d["pair"].split("-")[1]
            voice = {"ja": VOICES["ja"][k % 3], other: VOICES[other][(k + 1) % 3]}
            parts, times, t = [np.zeros(int(0.5 * SR), np.float32)], [], 0.5
            for i, turn in enumerate(d["turns"]):
                if breath:
                    cl = []
                    for j, c in enumerate(_clauses(turn["text"])):
                        if j:
                            cl.append(np.zeros(int(float(rng.uniform(breath_lo, breath_hi)) * SR), np.float32))
                        cl.append(_say(c, voice[turn["lang"]], os.path.join(td, f"t{i}_{j}.wav")))
                    a = np.concatenate(cl)
                else:
                    a = _say(turn["text"], voice[turn["lang"]], os.path.join(td, f"t{i}.wav"))
                times.append({"start": round(t, 3), "end": round(t + len(a) / SR, 3)})
                gap = float(rng.uniform(gap_lo, gap_hi))
                parts += [a, np.zeros(int(gap * SR), np.float32)]
                t += len(a) / SR + gap
            x = np.concatenate(parts + [np.zeros(int(1.5 * SR), np.float32)])
            with wave.open(str(out / f"{d['name']}.wav"), "wb") as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(SR)
                w.writeframes((np.clip(x, -1, 1) * 32767).astype(np.int16).tobytes())
            (out / f"{d['name']}.json").write_text(json.dumps(
                {"name": d["name"], "voices": voice, "times": times, "dur": len(x) / SR},
                ensure_ascii=False, indent=1), encoding="utf-8")
            print(d["name"], round(len(x) / SR, 1), voice, flush=True)


# ---------------------------------------------------------------- 計算機なしで確かめる部分

def tgt_of(pair: str, lang: str) -> str:
    a, b = pair.split("-")
    return b if lang == a else a


def assign_lists(units: list, times: list) -> list[list[str]]:
    """出た単位(lo・hi・訳 tr)を、真ん中の時刻が入る発話に割り当てる。発話 i の範囲は
    [i の始まり − 0.3, 次の発話の始まり)。2つの発話にまたがって繋がった単位は片方に入り、
    もう片方は空になる(区切りの誤りとして点が下がる)。"""
    out: list = [[] for _ in times]
    for u in units:
        m = (u["lo"] + u["hi"]) / 2
        for i, t in enumerate(times):
            nxt = times[i + 1]["start"] if i + 1 < len(times) else 1e9
            if t["start"] - 0.3 <= m < nxt:
                if u.get("tr"):
                    out[i].append(u["tr"])
                break
    return out


def join_tr(parts: list, tgt: str) -> str:
    return ("" if tgt in ("ja", "zh") else " ").join(parts)


def assign(units: list, times: list, tgt: str) -> list[str]:
    return [join_tr(o, tgt) for o in assign_lists(units, times)]


def messages_with_next(pair: tuple, history: list, text: str, lang: str, nxt: str) -> list:
    """アプリと同じ訳の指示に、次の発話を「文脈だけ」として足す(fullctx: 前後の文脈の上限)。"""
    sys.path.insert(0, str(ROOT / "src"))
    from tsuyaku.translate import build_messages
    msgs = build_messages(pair, "", history, text, lang)
    if not nxt:
        return msgs
    c = msgs[-1]["content"]
    k = c.find("This utterance is in")
    if k < 0:
        k = c.rfind(text) if text in c else len(c)
    add = f"Next utterance (context only, do not translate): {nxt}\n\n"
    msgs[-1] = dict(msgs[-1], content=c[:k] + add + c[k:])
    return msgs


def breakdown(s: dict) -> dict:
    """条件ごとの点から、品質の落ち方。"""
    return {"ear": s["text"] - s["seg"], "seg": s["seg"] - s["full"],
            "ctx_now": s["text"] - s["noctx"], "ctx_more": s["fullctx"] - s["text"]}


# ---------------------------------------------------------------- 計算機で(耳・区切り)

def _sv(model_dir: Path):
    sys.path.insert(0, str(ROOT / "src"))
    from tsuyaku.stt import SenseVoiceSTT
    return SenseVoiceSTT(str(model_dir), num_threads=4)


def make_ear(engine: str, stt_url: str, pair: tuple):
    """mac = SenseVoice(このMac の耳)/ auto = faster-whisper のサーバー(リモートサーバーの耳と同じ server/remote_stt_server.py)
    + 先出し用の SenseVoice。アプリと同じ RemoteSTT を通す。"""
    sys.path.insert(0, str(ROOT / "src"))
    from tsuyaku.guards import stt_prompt
    from tsuyaku.stt import RemoteSTT
    sv = _sv(next((ROOT / "models").glob("sherpa-onnx-sense-voice-*")))
    sv.langs = pair
    if engine == "mac":
        return sv
    ear = RemoteSTT(stt_url, sv)
    ear.langs = pair
    ear.prompt = stt_prompt("")
    return ear


def replay_units(ear, wav: Path, pair: tuple, trust_punct: bool = True) -> list:
    """対話の音声をアプリの区切り(SpeechCapture fine → StreamSegmenter → Endpointer、繋ぎ直しは同じ耳)
    で音声の時計に流し、確定した単位(文字・言語・lo・hi)を返す。守り(雑音の床・ペア外の幻覚)もアプリと同じ。"""
    import queue as _q
    sys.path.insert(0, str(ROOT / "src"))
    sys.path.insert(0, str(ROOT / "scripts"))
    from eval_long import _no_audio_device_ok
    _no_audio_device_ok()
    from tsuyaku.audio import SpeechCapture, rms
    from tsuyaku.config import Settings
    from tsuyaku.guards import hallucinated_out_of_pair
    from tsuyaku.segmenter import Endpointer, merge_segments
    s = Settings()
    with wave.open(str(wav)) as w:
        x = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768
    q: "_q.Queue" = _q.Queue()
    cap = SpeechCapture(str(ROOT / "models" / "silero_vad.onnx"), q, SR, s.min_speech_s, s.max_speech_s,
                        s.min_silence_s, vad_threshold=s.vad_threshold, fine=True)
    clock = [0.0]

    def hear(a, span=None):
        return ear.transcribe(a)

    ep = Endpointer(rejoin=hear, hold_open_s=s.hold_open_s if s.hold_open_s > 0 else 0.0,
                    trust_end_punct=trust_punct)
    units: list = []

    def emit(us):
        for u in us:
            units.append({"text": u.text, "lang": u.lang, "lo": u.lo, "hi": u.hi,
                          "wait": round(clock[0] - u.hi, 2)})   # 話し終わり → 確定までの待ち(音声の時計)

    next_tick = 0.0
    for i in range(0, len(x), 512):
        cap.on_chunk(x[i:i + 512])
        clock[0] = (i + 512) / SR
        if not q.empty():
            pend = []
            while not q.empty():
                pend.append(q.get_nowait())
            if len(pend) >= 3:
                pend = [merge_segments(pend, SR)]
            for item in pend:
                seg, lo, hi, pre = item[:4]
                if rms(seg) < min(s.noise_rms_floor, max(2.0 * cap.segmenter.noise, 0.0015)):
                    continue
                text, lang, prob = ear.transcribe(seg)
                if not text or not lang or hallucinated_out_of_pair(pair, lang, text, prob=prob):
                    continue
                emit(ep.push(seg, lo, hi, text, lang, now=clock[0], pre=pre, prob=prob,
                             cut=cap.segmenter.was_cut(hi)))
        if clock[0] >= next_tick:
            next_tick = clock[0] + 0.1
            emit(ep.tick(clock[0], speaking_since=cap.segmenter.speaking_since()))
    emit(ep.flush())
    return units


def run_hear(engine: str, stt_url: str, audio: Path, out: Path, trust_punct: bool = True) -> None:
    """耳と区切り: full(対話を流した単位)と seg(発話ごとに聞いた文字)を書く。"""
    res = {}
    for d in load_dialogs():
        pair = tuple(d["pair"].split("-"))
        meta = json.loads((audio / f"{d['name']}.json").read_text(encoding="utf-8"))
        ear = make_ear(engine, stt_url, pair)
        units = replay_units(ear, audio / f"{d['name']}.wav", pair, trust_punct)
        with wave.open(str(audio / f"{d['name']}.wav")) as w:
            x = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768
        seg = []
        for t in meta["times"]:
            a = x[int(max(0, t["start"] - 0.2) * SR): int((t["end"] + 0.2) * SR)]
            text, lang, _p = ear.transcribe(a)
            seg.append({"text": text, "lang": lang})
        res[d["name"]] = {"units": units, "seg": seg, "times": meta["times"]}
        print(d["name"], engine, len(units), "単位", flush=True)
    out.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")


# ---------------------------------------------------------------- 計算機で(訳)

def app_translate(base: str, model: str, pair: tuple, history: list, text: str, lang: str,
                  nxt: str = "") -> str:
    """アプリの訳と同じ: 指示文+文脈 → 訳(空なら1回やり直し)→ 言語違い・破綻なら空の文脈で1回やり直す。"""
    sys.path.insert(0, str(ROOT / "src"))
    from tsuyaku.guards import looks_truncated, wrong_language
    from tsuyaku.translate import build_messages, out_lang, translate_stream, translate_with_retry
    msgs = messages_with_next(pair, history, text, lang, nxt)
    trans = translate_with_retry(base, model, msgs)
    tgt = out_lang(pair, lang)
    for bad in (lambda t: wrong_language(text, t, tgt), lambda t: looks_truncated(text, t)):
        if trans and bad(trans):
            try:
                r = translate_stream(base, model, build_messages(pair, "", [], text, lang))
            except Exception:  # noqa: BLE001
                r = ""
            if r and not bad(r):
                trans = r
    return trans or ""


def run_translate(hear_file: Path, base: str, model: str, out: Path, conds: list) -> None:
    hres = json.loads(hear_file.read_text(encoding="utf-8")) if hear_file else {}
    res = json.loads(out.read_text(encoding="utf-8")) if out.exists() else {}
    for d in load_dialogs():
        pair = tuple(d["pair"].split("-"))
        turns = d["turns"]
        r = res.setdefault(d["name"], {})
        if "text" in conds or "noctx" in conds or "fullctx" in conds:
            hist: list = []
            outs = {"text": [], "noctx": [], "fullctx": []}
            for i, t in enumerate(turns):
                nxt = turns[i + 1]["text"] if i + 1 < len(turns) else ""
                if "text" in conds:
                    outs["text"].append(app_translate(base, model, pair, hist, t["text"], t["lang"]))
                if "noctx" in conds:
                    outs["noctx"].append(app_translate(base, model, pair, [], t["text"], t["lang"]))
                if "fullctx" in conds:
                    outs["fullctx"].append(app_translate(base, model, pair, hist, t["text"], t["lang"], nxt))
                tr = outs["text"][-1] if outs["text"] else outs["fullctx"][-1]
                hist.append({"lang": t["lang"], "orig": t["text"], "trans": tr})
            for k, v in outs.items():
                if v:
                    r[k] = v
        h = hres.get(d["name"])
        if h and "seg" in conds:
            hist, o = [], []
            for sg in h["seg"]:
                tr = app_translate(base, model, pair, hist, sg["text"], sg["lang"]) if sg["text"] else ""
                o.append(tr)
                if sg["text"] and tr:
                    hist.append({"lang": sg["lang"], "orig": sg["text"], "trans": tr})
            r["seg"] = o
            r["seg_src"] = [sg["text"] for sg in h["seg"]]
        if h and "full" in conds:
            hist, units = [], [dict(u) for u in h["units"]]
            for u in units:
                u["tr"] = app_translate(base, model, pair, hist, u["text"], u["lang"])
                if u["tr"]:
                    hist.append({"lang": u["lang"], "orig": u["text"], "trans": u["tr"]})
            r["full_units"] = units
            r["times"] = h["times"]
        print(d["name"], "訳", flush=True)
        out.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")


# ---------------------------------------------------------------- 計算機で(点)

def run_score(tr_files: list, save: "Path | None", comet_model: str = "Unbabel/wmt22-comet-da") -> None:
    """発話ごとに COMET と chrF。条件ごとの平均と落ち方(全体・文脈が要る発話・長さ別)。"""
    from comet import download_model, load_from_checkpoint
    from sacrebleu.metrics import CHRF
    chrf = CHRF()
    model = load_from_checkpoint(download_model(comet_model))
    dialogs = {d["name"]: d for d in load_dialogs()}
    lines = []
    for f in tr_files:
        res = json.loads(Path(f).read_text(encoding="utf-8"))
        rows = []                  # (条件, 対話, 発話, src, mt, ref)
        for name, r in res.items():
            d = dialogs[name]
            for cond in ("seg", "text", "noctx", "fullctx"):
                if cond in r:
                    for i, t in enumerate(d["turns"]):
                        rows.append((cond, name, i, t["text"], r[cond][i] or "", t["ref"]))
            if "full_units" in r and r.get("times"):
                parts = assign_lists(r["full_units"], r["times"])
                for i, t in enumerate(d["turns"]):
                    rows.append(("full", name, i, t["text"], join_tr(parts[i], tgt_of(d["pair"], t["lang"])),
                                 t["ref"]))
        data = [{"src": s, "mt": m, "ref": rf} for _c, _n, _i, s, m, rf in rows]
        sc = model.predict(data, batch_size=32, gpus=int(os.environ.get("QUAL_COMET_GPUS", "1")), progress_bar=False).scores
        if save:   # 発話ごとの点(切り分けの分析用)
            Path(save).with_name(Path(save).stem + f".{Path(f).stem}.rows.json").write_text(json.dumps(
                [{"cond": c, "dialog": n, "i": i, "mt": m, "comet": float(v)}
                 for (c, n, i, _s, m, _rf), v in zip(rows, sc)], ensure_ascii=False), encoding="utf-8")
        per: dict = {}
        for (c, n, i, s, m, rf), v in zip(rows, sc):
            t = dialogs[n]["turns"][i]
            ch = chrf.sentence_score(m, [rf]).score
            for key in ("all", "ctx" if t["ctx"] else "noctx_turn", t["kind"], dialogs[n]["pair"]):
                per.setdefault((c, key), []).append((v, ch))
        lines.append(f"### {Path(f).stem}")
        lines.append("| 区分 | " + " | ".join(("full", "seg", "text", "noctx", "fullctx")) + " |")
        lines.append("|---|---|---|---|---|---|")
        for key in ("all", "ctx", "noctx_turn", "short", "mid", "long", "ja-zh", "ja-en", "ja-ko"):
            cells = []
            for c in ("full", "seg", "text", "noctx", "fullctx"):
                v = per.get((c, key))
                cells.append(f"{100 * np.mean([a for a, b in v]):.1f} / {np.mean([b for a, b in v]):.1f}"
                             if v else "—")
            lines.append(f"| {key} | " + " | ".join(cells) + " |")
        means = {c: float(np.mean([a for a, b in per[(c, "all")]])) for c in
                 ("full", "seg", "text", "noctx", "fullctx") if (c, "all") in per}
        if len(means) == 5:
            b = breakdown(means)
            lines.append("")
            lines.append("落ち方(COMET×100): 耳 %.1f / 区切り %.1f / 今の文脈の効き %.1f / 後ろの文脈で直る分 %.1f"
                         % (100 * b["ear"], 100 * b["seg"], 100 * b["ctx_now"], 100 * b["ctx_more"]))
        lines.append("")
    text = "\n".join(lines)
    print(text)
    if save:
        save.write_text(text + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    b = sp.add_parser("build")
    b.add_argument("--out", type=Path, default=ROOT / "eval" / "qual" / "audio")
    b.add_argument("--breath", action="store_true", help="発話の中の読点・文末に息つぎの間を入れる")
    b.add_argument("--breath-gap", type=float, nargs=2, default=[0.35, 0.9])
    b.add_argument("--turn-gap", type=float, nargs=2, default=[0.3, 0.9])
    h = sp.add_parser("hear")
    h.add_argument("--engine", choices=["mac", "auto"], required=True)
    h.add_argument("--stt-url", default="")
    h.add_argument("--audio", type=Path, default=ROOT / "eval" / "qual" / "audio")
    h.add_argument("--out", type=Path, required=True)
    h.add_argument("--no-trust-punct", action="store_true", help="耳の文末の句読点を信じない(Endpointer)")
    t = sp.add_parser("translate")
    t.add_argument("--hear", type=Path)
    t.add_argument("--base", required=True)
    t.add_argument("--model", required=True)
    t.add_argument("--out", type=Path, required=True)
    t.add_argument("--conds", default="full,seg,text,noctx,fullctx")
    c = sp.add_parser("score")
    c.add_argument("files", nargs="+")
    c.add_argument("--save", type=Path)
    a = ap.parse_args()
    if a.cmd == "build":
        build(a.out, a.breath, *a.breath_gap, *a.turn_gap)
    elif a.cmd == "hear":
        run_hear(a.engine, a.stt_url, a.audio, a.out, not a.no_trust_punct)
    elif a.cmd == "translate":
        run_translate(a.hear, a.base, a.model, a.out, a.conds.split(","))
    else:
        run_score(a.files, a.save)


if __name__ == "__main__":
    main()
