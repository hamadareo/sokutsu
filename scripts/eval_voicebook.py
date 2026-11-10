"""声を覚える(src/tsuyaku/voicebook.py)の評価: 覚えた声と別の会議の話者を照合して、
間違った名前を付ける割合(いちばん避けたい)と、正しく付く割合を測り、しきい値を決める。

使うデータ(どれもこの Mac の中だけ。リモートサーバー に音声は送らない):
  bank : 話者の声の束(scripts/eval_speakers.py prep で作った 28人、ja 4 / en 8 / zh 8 / ko 8、
         静かな読み上げ)。1人のクリップを「覚える用(約30秒)」と「試す用(約20秒・40秒の束)」に
         分け(同じクリップは使わない)、同じ人/別の人の類似を出す。
  conv : 合成会議 56本(bank の声で作った 2〜4人の会話、静か/会場の雑音)。アプリと同じ区切り
         (AutoGain → silero VAD → StreamSegmenter)・同じ話者の見分け(OnlineClusters + refine)で
         会議の話者(まとまり)を作り、その重心を照合する。帳面は bank のクリップのうち、その会議で
         使っていないものから作る(同じ録音で照合しない)。
           known : 会議の人も帳面にいる(28人全員が帳面)→ 正しい/違う名前/付かない
           stranger: 会議の人を帳面から抜く → 付いたら全部「違う名前」
  ami  : 本物の4人の英語会議 AMI 4本(16人)。会議を前半/後半に分け、前半(口元マイク混ぜ)で覚え、
         後半(卓上マイク1本 = アプリの使い方に近い)で照合する(時間もマイクも違う)。stranger は
         その会議の4人を帳面から抜く。

実行(リポジトリ直下。CPU を使うのでアプリの邪魔をしないよう nice で):
  nice -n 19 .venv/bin/python -I scripts/eval_voicebook.py --bank BANK --conv CONV --ami AMI_DIR \
      [--ami-extra DIR2] --out OUT
結果は OUT/summary.md と OUT/rows.json(eval/voicebook/ に写す)。
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "src"))

from tsuyaku import voicebook as V  # noqa: E402
from tsuyaku.speakers import OnlineClusters, find_model, refine_labels  # noqa: E402

SR = 16000
APP = dict(threshold=0.3, max_speakers=8, min_s=0.6, new_min_s=1.2, max_weight_s=6.0, confirm=1,
           update_th=0.5, merge_th=0.7, absorb_frac=0.03, absorb_after=30, absorb_sim=0.2)


def _extractor():
    from tsuyaku.speakers import SpeakerTracker
    root = HERE.parent
    m = find_model(root / "models")
    if not m:
        raise SystemExit("話者モデルが models/ にありません(scripts/get_models.sh)")
    return SpeakerTracker(m, num_threads=1)


class Emb:
    """特徴のキャッシュ(npz)。キーごとに1回だけ計算する。"""

    def __init__(self, path: Path, extra: "list[Path]" = ()):
        self.path = path
        self.c = dict(np.load(path)) if path.exists() else {}
        for p in extra:
            if p.exists():
                for k, v in np.load(p).items():
                    self.c.setdefault(k, v)
        self._trk = None
        self.new = 0

    def get(self, key: str, x) -> "np.ndarray | None":
        if key not in self.c:
            if self._trk is None:
                self._trk = _extractor()
            e = self._trk.embed(np.asarray(x(), np.float32))
            self.c[key] = e if e is not None else np.zeros(1, np.float32)
            self.new += 1
        e = self.c[key]
        return e if e.size > 1 else None

    def save(self):
        if self.new:
            np.savez(self.path, **self.c)
            self.new = 0


def cluster(embs: list, durs: list) -> list:
    """アプリと同じ: オンラインのまとめ分け → 会議の後の付け直し。"""
    cl = OnlineClusters(**APP)
    pred = [cl.assign(e, d) if e is not None else None for e, d in zip(embs, durs)]
    return refine_labels(embs, durs, [cl.canonical(p) for p in pred])


def clusters_with_gold(embs, durs, labels, golds) -> dict:
    """{話者: (重心, 秒, 区間数, 主な正解の人, 純度)}。"""
    cents = V.centroids(embs, durs, labels)
    out = {}
    for spk, (v, secs, n) in cents.items():
        w: dict = {}
        for l, g, d, e in zip(labels, golds, durs, embs):
            if l == spk and e is not None and g not in (None, "?"):
                w[g] = w.get(g, 0.0) + d
        if not w:
            continue
        g, gw = max(w.items(), key=lambda kv: kv[1])
        out[spk] = (v, secs, n, g, gw / sum(w.values()))
    return out


def book_from(samples: dict, root: Path) -> V.VoiceBook:
    """{人: [(重心, 秒, 区間数, src)]} → 帳面(一時フォルダ)。"""
    b = V.VoiceBook(root / "_none_")          # 書き出さない(評価では何千回も作る)
    b._save = lambda: None
    b._fresh = lambda: None
    for who, ss in samples.items():
        for v, secs, n, src in ss:
            b.enroll(who, v, max(secs, V.MIN_ENROLL_S), n, src)
    return b


def judge(book: V.VoiceBook, cl: dict, mid: str, known: set) -> dict:
    """1会議の照合の結果を数える。"""
    m = book.match({k: (v, s, n) for k, (v, s, n, g, p) in cl.items()}, exclude_mid=mid)
    r = {"eligible": 0, "eligible_known": 0, "correct": 0, "wrong": 0, "missed": 0, "wrong_stranger": 0,
         "sims_correct": [], "sims_wrong": [], "stranger_top": [], "known_true": []}
    for spk, (v, secs, n, g, pur) in cl.items():
        if secs < V.MIN_MATCH_S or n < V.MIN_MATCH_N:
            continue
        r["eligible"] += 1
        isk = g in known
        sc = book.scores(v, mid)
        if not isk and sc:
            r["stranger_top"].append(sc[0][0])          # 帳面にいない人の、いちばん近い別人との類似
        if isk:
            r["known_true"] += [s for s, p in sc if p["name"] == g][:1]   # 本人の見本との類似
        r["eligible_known"] += int(isk)
        if spk in m:
            if m[spk]["name"] == g:
                r["correct"] += 1
                r["sims_correct"].append(m[spk]["sim"])
            else:
                r["wrong"] += 1
                r["wrong_stranger"] += int(not isk)
                r["sims_wrong"].append(m[spk]["sim"])
        elif isk:
            r["missed"] += 1
    return r


def add(a: dict, b: dict) -> dict:
    for k, v in b.items():
        a[k] = a.get(k, [] if isinstance(v, list) else 0) + v
    return a


# ====================================================================== bank

def load_bank(bank: Path, emb: Emb):
    from eval_conv import read_wav
    items = json.loads((bank / "bank.json").read_text(encoding="utf-8"))
    out: dict = {}
    for b in items:
        e = emb.get(f"bank:{b['path']}", lambda b=b: read_wav(bank / b["path"]))
        out.setdefault(b["spk"], []).append((b["path"], e, float(b["dur"]), b.get("gender", ""), b["lang"]))
    emb.save()
    return out


def groups(clips: list, secs: float) -> list:
    """クリップを順に束ねて、secs 秒以上の束の列。"""
    out, cur, t = [], [], 0.0
    for c in clips:
        cur.append(c)
        t += c[2]
        if t >= secs:
            out.append(cur)
            cur, t = [], 0.0
    return out


def cent_of(clips: list):
    c = V.centroids([x[1] for x in clips], [x[2] for x in clips], ["x"] * len(clips))
    return c.get("x")


def eval_bank(bank_items: dict, enroll_s=30.0, test_s=(20.0, 40.0)) -> dict:
    res = {"enroll_s": enroll_s, "people": len(bank_items), "tests": {}}
    enr, tests = {}, {t: {} for t in test_s}
    for spk, clips in bank_items.items():
        clips = sorted(clips)
        g = groups(clips, enroll_s)
        if not g:
            continue
        enr[spk] = cent_of(g[0])
        used = len(g[0])
        for t in test_s:
            tests[t][spk] = [cent_of(x) for x in groups(clips[used:], t)]
    meta = {spk: (c[0][3], c[0][4]) for spk, c in bank_items.items()}
    for t in test_s:
        same, diff, diff_sl = [], [], []
        for a, ea in enr.items():
            for b, tb in tests[t].items():
                for c in tb:
                    s = float(ea[0] @ c[0])
                    if a == b:
                        same.append(s)
                    else:
                        diff.append(s)
                        if meta[a] == meta[b]:
                            diff_sl.append(s)     # 同じ言語・同じ性別の別の人(いちばん紛らわしい)
        res["tests"][t] = {"same": _dist(same), "diff": _dist(diff), "diff_same_lang_gender": _dist(diff_sl),
                           "n_same": len(same), "n_diff": len(diff)}
        res["tests"][t]["_raw"] = (same, diff)
    return res


def _dist(x: list) -> dict:
    if not x:
        return {}
    a = np.asarray(x)
    return {"min": round(float(a.min()), 3), "p1": round(float(np.percentile(a, 1)), 3),
            "p50": round(float(np.median(a)), 3), "p99": round(float(np.percentile(a, 99)), 3),
            "max": round(float(a.max()), 3)}


# ====================================================================== conv / ami(アプリの話者の見分けを通す)

def conv_sessions(conv_dir: Path, emb: Emb) -> list:
    from eval_conv import read_wav
    from eval_speakers import _gold, segments_like_app
    out = []
    for jp in sorted(conv_dir.glob("*.json")):
        meta = json.loads(jp.read_text(encoding="utf-8"))
        audio = read_wav(conv_dir / meta["wav"])
        segs = segments_like_app(audio)
        embs, durs, golds = [], [], []
        for k, (seg, lo, hi, pre) in enumerate(segs):
            embs.append(emb.get(f"conv:{jp.stem}:{k}:{len(seg)}", lambda seg=seg: seg))
            durs.append(len(seg) / SR)
            golds.append(_gold(lo + pre, hi, meta["turns"]))
        labels = cluster(embs, durs)
        out.append({"id": jp.stem, "cond": meta["cond"], "speakers": meta["speakers"],
                    "srcs": {t["src"] for t in meta["turns"]},
                    "clusters": clusters_with_gold(embs, durs, labels, golds)})
        print(f"[conv] {jp.stem}: 区間 {len(segs)} まとまり {len(out[-1]['clusters'])}", flush=True)
    emb.save()
    return out


def eval_conv(sessions: list, bank_items: dict, root: Path, enroll_s: float = 30.0) -> dict:
    tot_k, tot_s = {}, {}
    for s in sessions:
        # 帳面: その会議で使っていない bank のクリップから約30秒(同じ録音で照合しない)
        samples = {}
        for spk, clips in bank_items.items():
            free = [c for c in sorted(clips) if c[0] not in s["srcs"]]
            g = groups(free, enroll_s)
            if g:
                v, secs, n = cent_of(g[0])
                samples[spk] = [(v, secs, n, f"bank:{spk}")]
        bk = book_from(samples, root / "book")
        add(tot_k, judge(bk, s["clusters"], s["id"], set(samples)))
        strangers = {k: v for k, v in samples.items() if k not in s["speakers"]}
        bs = book_from(strangers, root / "book")
        add(tot_s, judge(bs, s["clusters"], s["id"], set(strangers)))
    return {"known": tot_k, "stranger": tot_s}


def ami_sessions(ami_dirs: list, emb: Emb, meetings=("ES2004a", "TS3003a", "IS1009a", "EN2002a"),
                 mics=("Mix-Headset", "Array1-01")) -> list:
    from eval_conv import read_wav
    from eval_speakers import _gold, ami_turns, segments_like_app
    out = []
    for mt in meetings:
        ann = next((d / "ann" for d in ami_dirs if (d / "ann").exists()), ami_dirs[0] / "ann")
        turns = ami_turns(ann, mt)
        for mic in mics:
            wav = next((d / f"{mt}.{mic}.wav" for d in ami_dirs if (d / f"{mt}.{mic}.wav").is_file()), None)
            if wav is None or not turns:
                print(f"[ami] {mt} {mic}: 無い → 飛ばす", flush=True)
                continue
            audio = read_wav(wav)
            half = len(audio) / SR / 2
            segs = segments_like_app(audio)
            t0 = time.time()
            rows = []
            for k, (seg, lo, hi, pre) in enumerate(segs):
                e = emb.get(f"ami:{mt}:{mic}:{k}:{len(seg)}", lambda seg=seg: seg)
                rows.append((e, len(seg) / SR, _gold(lo + pre, hi, turns), lo))
            emb.save()
            for part, keep in (("a", lambda lo: lo < half), ("b", lambda lo: lo >= half)):
                rr = [r for r in rows if keep(r[3])]
                embs, durs, golds = [r[0] for r in rr], [r[1] for r in rr], [r[2] for r in rr]
                labels = cluster(embs, durs)
                out.append({"id": f"{mt}-{part}-{mic}", "meeting": mt, "part": part, "mic": mic,
                            "speakers": sorted({t["spk"] for t in turns}),
                            "clusters": clusters_with_gold(embs, durs, labels, golds)})
            print(f"[ami] {mt} {mic}: {len(audio) / SR / 60:.1f}分 区間 {len(segs)} ({time.time() - t0:.0f}s)",
                  flush=True)
    return out


def eval_ami(sessions: list, root: Path, enroll_mic="Mix-Headset", test_mic="Array1-01") -> dict:
    by = {s["id"]: s for s in sessions}
    meetings = sorted({s["meeting"] for s in sessions})
    # 前半(enroll_mic)のまとまりを、主な正解の人の名前で覚える(利用者が名前を付けるのと同じ)
    samples: dict = {}
    for mt in meetings:
        s = by.get(f"{mt}-a-{enroll_mic}")
        if not s:
            continue
        for spk, (v, secs, n, g, pur) in s["clusters"].items():
            if secs >= V.MIN_ENROLL_S and pur >= 0.6:
                samples.setdefault(g, []).append((v, secs, n, f"{s['id']}:{spk}"))
    tot_k, tot_s = {}, {}
    for mt in meetings:
        s = by.get(f"{mt}-b-{test_mic}")
        if not s:
            continue
        bk = book_from(samples, root / "book")
        add(tot_k, judge(bk, s["clusters"], s["id"], set(samples)))
        strangers = {k: v for k, v in samples.items() if not k.startswith(mt + ":")}
        bs = book_from(strangers, root / "book")
        add(tot_s, judge(bs, s["clusters"], s["id"], set(strangers)))
    return {"known": tot_k, "stranger": tot_s, "enrolled_people": len(samples)}


# ====================================================================== まとめ

def rates(r: dict) -> dict:
    lab = r.get("correct", 0) + r.get("wrong", 0)
    return {"stranger_top": _dist(r.get("stranger_top", [])), "known_true": _dist(r.get("known_true", [])),
            "eligible": r.get("eligible", 0), "correct": r.get("correct", 0), "wrong": r.get("wrong", 0),
            "missed": r.get("missed", 0),
            "precision": round(100 * r.get("correct", 0) / lab, 1) if lab else None,
            "recall": round(100 * r.get("correct", 0) / r["eligible_known"], 1) if r.get("eligible_known") else None,
            "false_rate": round(100 * r.get("wrong", 0) / r["eligible"], 2) if r.get("eligible") else None}


def sweep(conv_s, ami_s, bank_items, root: Path, grid) -> list:
    rows = []
    for th, mg, ms in grid:
        V.THRESHOLD, V.MARGIN, V.MIN_MATCH_S = th, mg, ms
        row = {"threshold": th, "margin": mg, "min_match_s": ms}
        if conv_s:
            c = eval_conv(conv_s, bank_items, root)
            row["conv_known"], row["conv_stranger"] = rates(c["known"]), rates(c["stranger"])
        if ami_s:
            a = eval_ami(ami_s, root)
            row["ami_known"], row["ami_stranger"] = rates(a["known"]), rates(a["stranger"])
            row["ami_enrolled"] = a["enrolled_people"]
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", type=Path, required=True)
    ap.add_argument("--conv", type=Path)
    ap.add_argument("--ami", type=Path, action="append", default=[])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--cache-extra", type=Path, action="append", default=[])
    ap.add_argument("--th", default="", help="照合のしきい値の候補(カンマ区切り)")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    emb = Emb(a.out / "emb.npz", a.cache_extra)
    t0 = time.time()
    bank_items = load_bank(a.bank, emb)
    b = eval_bank(bank_items)
    print(f"[bank] {len(bank_items)}人 {time.time() - t0:.0f}s", flush=True)
    conv_s = conv_sessions(a.conv, emb) if a.conv else []
    ami_s = ami_sessions(a.ami, emb) if a.ami else []
    ths = [float(x) for x in a.th.split(",")] if a.th else (0.45, 0.5, 0.55, 0.6, 0.65, 0.7)
    grid = list(itertools.product(ths, (0.0, 0.05, 0.1), (10.0, 20.0, 30.0)))
    rows = sweep(conv_s, ami_s, bank_items, a.out, grid)
    # 同じ人/別の人の類似の分布(まとまりの重心どうし)
    dist = {}
    for name, ss in (("conv", conv_s), ("ami", ami_s)):
        same, diff = [], []
        for x, y in itertools.combinations(ss, 2):
            for (vx, sx, nx, gx, px), (vy, sy, ny, gy, py) in itertools.product(x["clusters"].values(),
                                                                                    y["clusters"].values()):
                if min(sx, sy) < 20 or min(px, py) < 0.8:
                    continue
                (same if gx == gy else diff).append(float(vx @ vy))
        dist[name] = {"same": _dist(same), "diff": _dist(diff), "n_same": len(same), "n_diff": len(diff)}
    for t, v in b["tests"].items():
        v.pop("_raw", None)
    out = {"bank": b, "cluster_dist": dist, "sweep": rows}
    (a.out / "rows.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({"bank": b, "cluster_dist": dist}, ensure_ascii=False, indent=1))
    print(f"[done] {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
