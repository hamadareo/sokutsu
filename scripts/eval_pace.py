"""読み上げが話のペースについていけるかの再生シミュレーション(11/08、pace.py の評価)。

話者の発話(FLEURS の実際の長さ・文。CC BY 4.0)を、遅い/ふつう/速いペースで並べ、
  話し終わり + 前段(認識・訳)= 読み上げの行列に入る時刻
  読み上げの長さ = 訳の量 ÷ 1秒あたりの量(このMacで声ごと・速さごとに測った表 eval/pace/tts_speed.json)+ 鳴り始めの遅れ
で、アプリの _tts_worker と同じ規則(1本ずつ読む・遅れすぎたら飛ばす・読み上げ中はマイクを止める)を時計の上で回す。
訳の量は同じ文 id の別の言語の FLEURS の文(人の訳)。

比べる方式:
  now        今: 速さ固定(ユーザーの ×1.0)、行列で8秒遅れたら飛ばす、読み上げ中は聞かない
  auto       話のペースに合わせた速さ(pace.PaceController)+ 遅れたら行列の短い訳をまとめて読む
  auto+listen  auto + 読み上げ中も聞く(声を捨てない)
  now+listen   今の速さ + 読み上げ中も聞く
指標: 読み上げ中に聞けなかった原文の割合(秒)、声の遅れ(話し終わり→鳴り始め)p50/p90、飛ばした数、
  使った速さの平均/最大。

実行: .venv/bin/python -I scripts/eval_pace.py [--n 40] [--save eval/pace/sim.md]
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tsuyaku.pace import PaceController, units  # noqa: E402

FLEURS = ROOT / "eval" / "fleurs"
DL = {"ja": "dl_ja_jp", "zh": "dl_cmn_hans_cn", "en": "dl_en_us", "ko": "dl_ko_kr"}
VOICE = {"ja": ("voicevox", "ja"), "zh": ("kokoro", "zh"), "en": ("kokoro", "en"), "ko": ("say", "ko")}
DIRS = [("zh", "ja"), ("ja", "zh"), ("en", "ja"), ("ja", "en"), ("ko", "ja")]
# 話者のペース: FLEURS の読み上げの長さに掛ける数と、発話の間(秒)
PACES = {"slow": (1.25, 0.8), "normal": (1.0, 0.5), "fast": (0.7, 0.25)}
PREP_S = 1.2        # 話し終わり → 訳が行列に入るまで(実記録の 前段+訳 の中ほど)
ONSET_S = 0.3       # 行列から取ってから鳴り始めるまで(合成。作り置きが当たれば短い)
GATE_TAIL_S = 0.25  # 読み上げの後、マイクを開くまで
MAX_LAG_S = 8.0     # voice_max_lag_s
csv.field_size_limit(10_000_000)


def load_rows(lang: str) -> dict:
    out = {}
    with open(FLEURS / DL[lang] / "test.tsv", encoding="utf-8", newline="") as f:
        for r in csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE):
            if len(r) < 6 or not r[0].isdigit() or not r[5].isdigit():
                continue
            out.setdefault(int(r[0]), (r[2].strip(), int(r[5]) / 16000.0))
    return out


def speed_table(path: Path) -> dict:
    """{(声, 言語): [(速さ, 1秒あたりの量)]}"""
    t: dict = {}
    for r in json.loads(path.read_text(encoding="utf-8")):
        k = (r["voice"].split(":")[0], r["lang"])
        t.setdefault(k, []).append((r["speed"], r["units_per_s"]))
    return {k: sorted(v) for k, v in t.items()}


def true_cps(table: dict, key, speed: float) -> float:
    pts = table[key]
    if speed <= pts[0][0]:
        return pts[0][1] * speed / pts[0][0]
    for (a, ca), (b, cb) in zip(pts, pts[1:]):
        if a <= speed <= b:
            return ca + (cb - ca) * (speed - a) / (b - a)
    a, ca = pts[-1]
    return ca * speed / a


def build(src: str, tgt: str, pace: str, n: int, rows: dict) -> list[dict]:
    f, gap = PACES[pace]
    ids = sorted(set(rows[src]) & set(rows[tgt]))
    t = 0.0
    out = []
    for i in ids[:n]:
        stext, sdur = rows[src][i]
        dur = max(0.8, sdur * 0.85 * f)     # FLEURS の前後の無音(1割強)を除く
        out.append({"start": t, "end": t + dur, "dur": dur, "src_units": units(stext, src),
                    "tr": rows[tgt][i][0]})
        t += dur + gap
    return out


def simulate(utts: list[dict], tgt: str, table: dict, policy: str, caps: dict,
             join_after_s: float = 2.0, max_join: int = 3) -> dict:
    key = VOICE[tgt]
    listen = policy.endswith("+listen")
    auto = policy.startswith("auto")
    pc = PaceController(caps=caps) if auto else None
    play: list[tuple[float, float]] = []      # 鳴っていた区間
    q: list[dict] = []
    lost_s = 0.0
    lags, speeds, skips, joined = [], [], 0, 0
    pending = sorted(utts, key=lambda u: u["end"])
    # 読み上げ中に話された所は聞けない(listen でなければ)。行列に入るかは、話した区間のうち
    # 鳴っていた時間が半分未満なら入る(残りの声で訳せる)、とする
    def overlap(u):
        o = 0.0
        for a, b in play:
            o += max(0.0, min(u["end"], b + GATE_TAIL_S) - max(u["start"], a))
        return o
    clock = 0.0
    # 訳が行列に入る時刻: 既定は 話し終わり + PREP_S。u["ready"] があればそれ(eval_long.py: 区切り方で変わる)
    rdy = lambda u: u.get("ready", u["end"] + PREP_S)
    pending.sort(key=rdy)
    while pending or q:
        if not q:                                  # 何も読んでいない: 次の訳が届くまで進む
            clock = max(clock, rdy(pending[0]))
        # 届いた分を行列へ(読み上げ中に話された所は、listen でなければ聞けない)
        while pending and rdy(pending[0]) <= clock + 1e-9:
            u = pending.pop(0)
            o = 0.0 if listen else overlap(u)
            lost_s += min(o, u["dur"])
            if o < 0.5 * u["dur"]:
                q.append(u)
        if not q:
            continue
        u = q.pop(0)
        lag = clock - u["end"]
        if lag > MAX_LAG_S:
            skips += 1
            u["skipped"] = True
            continue
        group = [u]
        text, src_dur = u["tr"], u["dur"]
        if auto and q and lag > join_after_s:       # 遅れている: 行列の訳をまとめて読む
            k = 0
            while q and k < max_join - 1:
                v = q.pop(0)
                group.append(v)
                text += ("" if tgt in ("ja", "zh") else " ") + v["tr"]
                src_dur += v["dur"]
                k += 1
            joined += k
        if auto:
            backlog = sum(pc.predict(v["tr"], *key, pc.current(*key)) for v in q)
            sp = pc.choose(text, key[0], key[1], src_dur, backlog + max(0.0, lag - PREP_S))
        else:
            sp = 1.0
        dur = units(text, tgt) / true_cps(table, key, sp)
        start = clock + ONSET_S
        play.append((start, start + dur))
        for v in group:                             # どの訳がいつ鳴り始めたか(eval_long.py の文ごとの遅れ)
            v["voice_at"] = start
        if auto:
            pc.observe(text, key[0], key[1], sp, dur)
        lags.append(start - u["end"])
        speeds.append(sp)
        clock = start + dur + GATE_TAIL_S
    total = sum(u["dur"] for u in utts)
    lags.sort()
    pct = lambda p: round(lags[min(len(lags) - 1, int(math.ceil(p * len(lags))) - 1)], 1) if lags else None
    return {"lost": round(100 * lost_s / total, 1), "lag50": pct(0.5), "lag90": pct(0.9),
            "skips": skips, "joined": joined, "spoken": len(lags),
            "sp_avg": round(sum(speeds) / len(speeds), 2) if speeds else None,
            "sp_max": round(max(speeds), 2) if speeds else None}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--table", type=Path, default=ROOT / "eval" / "pace" / "tts_speed.json")
    ap.add_argument("--caps", type=Path, default=ROOT / "eval" / "pace" / "caps.json")
    ap.add_argument("--save", type=Path)
    a = ap.parse_args()
    table = speed_table(a.table)
    caps = {tuple(k.split("|")): v for k, v in json.loads(a.caps.read_text()).items()} if a.caps.exists() else {}
    rows = {L: load_rows(L) for L in ("ja", "zh", "en", "ko")}
    policies = ["now", "now+listen", "auto", "auto+listen"]
    lines = ["| 方向 | 話者のペース | 原文の量/秒 | 方式 | 聞けなかった原文 | 声の遅れ p50/p90 | 飛ばした | まとめた | 速さ 平均/最大 |",
             "|---|---|---|---|---|---|---|---|---|"]
    for src, tgt in DIRS:
        for pace in PACES:
            utts = build(src, tgt, pace, a.n, rows)
            rate = sum(u["src_units"] for u in utts) / sum(u["dur"] for u in utts)
            for pol in policies:
                r = simulate(utts, tgt, table, pol, caps)
                lines.append(f"| {src}→{tgt} | {pace} | {rate:.1f} | {pol} | {r['lost']}% | {r['lag50']}/{r['lag90']}秒 "
                             f"| {r['skips']}/{len(utts)} | {r['joined']} | ×{r['sp_avg']}/×{r['sp_max']} |")
    txt = "\n".join(lines)
    print(txt)
    if a.save:
        a.save.write_text(txt + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
