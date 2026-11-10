"""話者の見分け(src/tsuyaku/speakers.py)の評価: 話者の分かる合成会議で、区間ごとの話者の
正しさと話者数の当たりを測り、モデルとしきい値を決める。

FLEURS は話者 id が無い(同じ「話者」役でも文ごとに別人)ので話者の正解に使えない。
話者 id の付いた無料コーパスから言語ごとに話者の声の束(bank)を作る:
  ja: JVNV(litagin/jvnv_corpus_v1_no_nv, CC BY-SA 4.0, 4人 F1/F2/M1/M2)
  en: LibriSpeech test-clean(HF openslr/librispeech_asr の parquet, CC BY 4.0)
  zh: AISHELL-1(HF AISHELL/AISHELL-1 の話者別 tar のうち10人、Apache-2.0 / 学術利用)
  ko: Zeroth-Korean test(kresnik/zeroth_korean, CC BY 4.0)
会場の雑音(venue)のざわめきだけは FLEURS の発話(eval_conv.py の clips)を使う。

手順(リポジトリ直下。ダウンロードは信頼しないデータなので -I で読む):
  1) prep  : 各コーパス → bank/<lang>/<spk>/NN.wav(16kHz mono、無音を落とした2〜12秒)+ bank.json
             pyarrow と soundfile が要る(アプリの venv には無いので別の venv で):
             <venv>/bin/python -I scripts/eval_speakers.py prep --src DIR --out BANK
             DIR に jvnv/(F1..M2/*.wav)、libri/test.clean.parquet(HF openslr/librispeech_asr)、
             aishell/S00xx.tar.gz、zeroth/test.parquet を置く。
  2) verify: モデルごとに bank の全クリップの特徴を取り、同じ人/別の人の EER(言語別・全体)と速さ
             .venv/bin/python -I scripts/eval_speakers.py verify --bank BANK --models M1.onnx,M2.onnx
  3) build : 合成会議(quiet/venue × 2人・3〜4人 × 言語の組み)を作る
             .venv/bin/python -I scripts/eval_speakers.py build --bank BANK --fleurs-clips CLIPS --out CONV
  4) run   : アプリと同じ区切り(AutoGain → silero VAD → StreamSegmenter)の区間ごとに
             SpeakerTracker で話者を付け、しきい値を振って正解率を出す
             .venv/bin/python -I scripts/eval_speakers.py run --conv CONV --models M.onnx --out RES

結果(11/06、合成会議56本 = 14場面 × 静か/会場 × seed 2): 声の照合の EER は 3D-Speaker CAM++ zh_en
advanced 1.3% / ERes2Net-base 0.8%(2倍遅い)/ TitaNet-small 2.0% / CN-Celeb ResNet34 2.8% /
VoxCeleb の CAM++ 2種は 39〜45%(使えない)。CAM++ zh_en を採用し、threshold 0.3・update_th 0.5・
merge_th 0.7 で 区間の正解率 静か2人 95.4 / 静か3〜4人 90.0 / 会場2人 90.8 / 会場3〜4人 80.1%。
本物の会議(5: ami、AMI 4人の英語会議4本): 口元マイク混ぜ 91.5% / 卓上マイク1本 87.7%。
  .venv/bin/python -I scripts/eval_speakers.py ami --ami AMI_DIR --models M.onnx --out RES
  (AMI_DIR に <id>.Mix-Headset.wav / <id>.Array1-01.wav と ann/words/ か ann/ami_manual.zip)

正解率: 区間(正解の発話と0.2秒以上重なり、主な話者が決まるもの)ごとに、付けた話者 id と
正解の話者を1対1で対応づけ(全通りから一致の秒数が最大になる対応)、一致した秒数 / 全区間の秒数。
None(判断せず)は不正解に数える。あわせて None の割合、付けた区間だけの正解率、
付けた話者の人数 − 正解の人数、ざわめきだけの区間に話者を付けた数。
"""
from __future__ import annotations

import argparse
import io
import itertools
import json
import math
import os
import random
import re
import sys
import tarfile
import time
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402

SR = 16000
LANGS = ("ja", "en", "zh", "ko")


# ====================================================================== 1) prep

def _to16k_mono(x: np.ndarray, sr: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.ndim > 1:
        x = x.mean(axis=1)
    if sr != SR:
        n_out = int(round(len(x) * SR / sr))
        X = np.fft.rfft(x)
        keep = min(len(X), n_out // 2 + 1)
        Y = np.zeros(n_out // 2 + 1, dtype=complex)
        Y[:keep] = X[:keep]
        x = (np.fft.irfft(Y, n_out) * (n_out / len(x))).astype(np.float32)
    return x


def _decode(b: bytes) -> tuple[np.ndarray, int]:
    import soundfile as sf
    x, sr = sf.read(io.BytesIO(b), dtype="float32", always_2d=False)
    return x, sr


def prep(src: Path, out: Path, per_lang: int = 8, per_spk: int = 25, seed: int = 0) -> None:
    from eval_conv import trim_silence, write_wav
    rng = random.Random(seed)
    out.mkdir(parents=True, exist_ok=True)
    bank: list[dict] = []

    def keep(lang, spk, gender, items):
        """items: [(bytes or ndarray, sr)] から 2〜12秒のものを per_spk 個まで書く。"""
        n = 0
        d = out / lang / spk
        d.mkdir(parents=True, exist_ok=True)
        for it in items:
            if n >= per_spk:
                break
            try:
                x, sr = it if isinstance(it, tuple) else _decode(it)
            except Exception:  # noqa: BLE001
                continue
            x = trim_silence(_to16k_mono(x, sr))
            dur = len(x) / SR
            if not (2.0 <= dur <= 12.0):
                continue
            p = d / f"{n:02d}.wav"
            write_wav(p, np.clip(x / max(1e-6, float(np.max(np.abs(x)))) * 0.5, -1, 1))
            bank.append({"lang": lang, "spk": f"{lang}:{spk}", "gender": gender,
                         "path": str(p.relative_to(out)), "dur": round(dur, 2)})
            n += 1
        print(f"[prep] {lang} {spk} ({gender}) {n}本", flush=True)

    def pick(spks: dict, genders: dict) -> list[str]:
        """男女をなるべく半々にして per_lang 人選ぶ。"""
        f = sorted(s for s in spks if genders.get(s) == "F")
        m = sorted(s for s in spks if genders.get(s) == "M")
        u = sorted(s for s in spks if genders.get(s) not in ("F", "M"))
        rng.shuffle(f), rng.shuffle(m), rng.shuffle(u)
        out_ = []
        while len(out_) < per_lang and (f or m or u):
            for pool in (f, m, u):
                if pool and len(out_) < per_lang:
                    out_.append(pool.pop())
        return out_

    # ja: JVNV(48kHz wav、F1/F2/M1/M2)
    jd = src / "jvnv"
    if jd.is_dir():
        import soundfile as sf
        for spk in sorted(p.name for p in jd.iterdir() if p.is_dir()):
            files = sorted((jd / spk).glob("*.wav"))
            rng.shuffle(files)
            items = []
            for f in files:
                x, sr = sf.read(str(f), dtype="float32", always_2d=False)
                items.append((x, sr))
            keep("ja", spk, spk[0], items)

    # en: LibriSpeech test-clean(HF openslr/librispeech_asr の parquet: audio{bytes=flac}, speaker_id)
    lp = src / "libri" / "test.clean.parquet"
    if lp.is_file():
        import pyarrow.parquet as pq
        t = pq.read_table(lp, columns=["audio", "speaker_id"])
        files: dict = {}
        for a_, s_ in zip(t.column("audio").to_pylist(), t.column("speaker_id").to_pylist()):
            if a_ and a_.get("bytes") and len(files.setdefault(str(s_), [])) < 60:
                files[str(s_)].append(a_["bytes"])
        for spk in pick(files, {}):
            rng.shuffle(files[spk])
            keep("en", spk, "?", files[spk])

    # zh: AISHELL-1(HF AISHELL/AISHELL-1 の話者ごとの S0xxx.tar.gz、16kHz wav)
    ad = src / "aishell"
    if ad.is_dir():
        files = {}
        for tp in sorted(ad.glob("S*.tar.gz")):
            with tarfile.open(tp, "r:gz") as tf:
                for m in tf:
                    mm = re.search(r"(S\d{4})/[\w]+\.wav$", m.name)
                    if m.isreg() and mm and m.size < 2_000_000 and len(files.setdefault(mm.group(1), [])) < 80:
                        files[mm.group(1)].append(tf.extractfile(m).read())
        for spk in pick(files, {}):
            rng.shuffle(files[spk])
            keep("zh", spk, "?", files[spk])

    # ko: Zeroth-Korean test(parquet: audio{bytes}, speaker_id)
    zp = src / "zeroth" / "test.parquet"
    if zp.is_file():
        import pyarrow.parquet as pq
        t = pq.read_table(zp)
        cols = t.column_names
        sid_col = next(c for c in ("speaker_id", "speaker") if c in cols)
        files = {}
        for a, s in zip(t.column("audio").to_pylist(), t.column(sid_col).to_pylist()):
            if a and a.get("bytes"):
                files.setdefault(str(s), []).append(a["bytes"])
        for spk in pick(files, {}):
            rng.shuffle(files[spk])
            keep("ko", spk, "?", files[spk])

    (out / "bank.json").write_text(json.dumps(bank, ensure_ascii=False, indent=0), encoding="utf-8")
    print(f"[prep] {len(bank)}本 → {out / 'bank.json'}")


# ====================================================================== 2) verify

def _extractor(model: str, threads: int = 2):
    import sherpa_onnx
    cfg = sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=model, num_threads=threads,
                                                      debug=False, provider="cpu")
    assert cfg.validate(), model
    return sherpa_onnx.SpeakerEmbeddingExtractor(cfg)


def _embed(ext, x: np.ndarray) -> np.ndarray:
    st = ext.create_stream()
    st.accept_waveform(sample_rate=SR, waveform=np.ascontiguousarray(x, dtype=np.float32))
    st.input_finished()
    e = np.asarray(ext.compute(st), dtype=np.float32)
    return e / (np.linalg.norm(e) + 1e-9)


def _eer(same: np.ndarray, diff: np.ndarray) -> tuple[float, float]:
    """EER(%) と、そのときのしきい値。"""
    ths = np.unique(np.concatenate([same, diff]))
    best = (1.0, 0.0, 0.5)
    for t in ths[:: max(1, len(ths) // 2000)]:
        frr = float(np.mean(same < t))
        far = float(np.mean(diff >= t))
        if abs(frr - far) < best[0]:
            best = (abs(frr - far), t, (frr + far) / 2)
    return 100 * best[2], float(best[1])


def verify(bank_dir: Path, models: list[str], out: Path | None = None) -> None:
    from eval_conv import read_wav
    bank = json.loads((bank_dir / "bank.json").read_text(encoding="utf-8"))
    xs = [read_wav(bank_dir / b["path"]) for b in bank]
    lines = [f"bank: {len(bank)}本 / 話者 {len({b['spk'] for b in bank})}人"]
    for mp in models:
        ext = _extractor(mp)
        t0 = time.time()
        E = np.stack([_embed(ext, x) for x in xs])
        dt = (time.time() - t0) / len(xs)
        # 3秒の区間1本の時間(1スレッド/2スレッド)
        x3 = xs[0][: 3 * SR]
        tt = {}
        for th in (1, 2):
            e2 = _extractor(mp, th)
            _embed(e2, x3)
            t1 = time.time()
            for _ in range(10):
                _embed(e2, x3)
            tt[th] = (time.time() - t1) / 10
        spk = np.array([b["spk"] for b in bank])
        lang = np.array([b["lang"] for b in bank])
        S = E @ E.T
        iu = np.triu_indices(len(bank), 1)
        same_m = (spk[:, None] == spk[None, :])[iu]
        res = []
        for L in LANGS + ("all",):
            m = np.ones(len(iu[0]), bool) if L == "all" else (lang[iu[0]] == L) & (lang[iu[1]] == L)
            if not m.any():
                continue
            eer, th = _eer(S[iu][m & same_m], S[iu][m & ~same_m])
            res.append(f"{L} EER {eer:.1f}%(th {th:.2f})")
        lines.append(f"{Path(mp).name}: {' / '.join(res)} | 1本平均 {dt * 1000:.0f}ms(平均 "
                     f"{np.mean([b['dur'] for b in bank]):.1f}秒) / 3秒 {tt[1] * 1000:.0f}ms(1スレッド) "
                     f"{tt[2] * 1000:.0f}ms(2スレッド) / 次元 {E.shape[1]}")
        print(lines[-1], flush=True)
    if out:
        out.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ====================================================================== 3) build

SCENES = [  # (名前, 言語の並び = 話者ごとの言語)
    ("ja2", ("ja", "ja")), ("en2", ("en", "en")), ("zh2", ("zh", "zh")), ("ko2", ("ko", "ko")),
    ("ja-en", ("ja", "en")), ("ja-zh", ("ja", "zh")), ("ja-ko", ("ja", "ko")), ("en-zh", ("en", "zh")),
    ("ja3", ("ja", "ja", "ja")), ("ja4", ("ja", "ja", "ja", "ja")), ("en4", ("en", "en", "en", "en")),
    ("ja2en1", ("ja", "ja", "en")), ("ja2en2", ("ja", "ja", "en", "en")), ("zh2ko2", ("zh", "zh", "ko", "ko")),
]


def build(bank_dir: Path, clips_dir: Path, out: Path, seeds=(1,), dur_s: float = 150.0) -> None:
    from eval_conv import active_rms, lowpass, pink_noise, read_wav, trim_silence, write_wav
    bank = json.loads((bank_dir / "bank.json").read_text(encoding="utf-8"))
    by_spk: dict[str, list[dict]] = {}
    for b in bank:
        by_spk.setdefault(b["spk"], []).append(b)
    babble_files = sorted(p for p in clips_dir.rglob("*.wav"))
    out.mkdir(parents=True, exist_ok=True)
    for seed in seeds:
        for scene, langs in SCENES:
            for cond in ("quiet", "venue"):
                rng = random.Random(f"{seed}-{scene}-{cond}")
                nrng = np.random.default_rng(zlib.crc32(f"{seed}-{scene}-{cond}".encode()))
                # 話者: 言語ごとに別人を選ぶ(同じ言語なら男女を混ぜる)
                spks: list[str] = []
                for L in langs:
                    cand = [s for s in by_spk if s.startswith(L + ":") and s not in spks]
                    g_used = [by_spk[s][0]["gender"] for s in spks if s.startswith(L + ":")]
                    pref = [s for s in cand if by_spk[s][0]["gender"] not in g_used] or cand
                    spks.append(rng.choice(pref))
                lvl = {s: (-26.0 + (rng.uniform(-6, 6) if cond == "venue" else rng.uniform(-2, 2)))
                       for s in spks}
                used = {s: 0 for s in spks}
                turns = []
                t = 1.0
                cur = rng.randrange(len(spks))
                pieces = []
                while t < dur_s:
                    s = spks[cur]
                    items = by_spk[s]
                    b = items[used[s] % len(items)]
                    used[s] += 1
                    x = trim_silence(read_wav(bank_dir / b["path"]))
                    g = lvl[s] + rng.uniform(-2, 2)
                    x = x * (10 ** (g / 20) / max(active_rms(x), 1e-6))
                    if turns:
                        if turns[-1]["spk"] == s:
                            gap, kind = rng.uniform(0.4, 1.2), "same"
                        else:
                            u = rng.random()
                            if cond == "venue" and u < 0.25 and turns[-1]["end"] - turns[-1]["start"] > 1.5:
                                gap, kind = -rng.uniform(0.2, 0.4), "overlap"
                            elif u < 0.6:
                                gap, kind = rng.uniform(0.2, 0.4), "quick"
                            else:
                                gap, kind = rng.uniform(0.6, 1.5), "pause"
                        t += gap
                    else:
                        kind = "first"
                    turns.append({"idx": len(turns), "spk": s, "lang": s.split(":")[0],
                                  "start": round(t, 3), "end": round(t + len(x) / SR, 3),
                                  "gap_kind": kind, "src": b["path"]})
                    pieces.append((int(round(t * SR)), x))
                    t += len(x) / SR
                    if not (rng.random() < 0.15 and (len(turns) < 2 or turns[-2]["spk"] != s)):
                        cur = rng.choice([i for i in range(len(spks)) if i != cur])
                total = int((t + 2.0) * SR)
                mix = np.zeros(total, np.float32)
                for a, x in pieces:
                    mix[a:a + len(x)] += x[: max(0, min(len(x), total - a))]
                snr = None
                if cond == "venue" and babble_files:
                    babble = np.zeros(total, np.float32)
                    for k in range(rng.randint(6, 10)):
                        pos = -int(rng.uniform(0, 8) * SR)
                        files = rng.sample(babble_files, 3)
                        j = 0
                        lv = 10 ** (rng.uniform(-3, 3) / 20)
                        while pos < total:
                            y = trim_silence(read_wav(files[j % 3]))
                            y = lowpass(y) * (lv * 0.05 / max(active_rms(y), 1e-6))
                            a, bb = max(0, pos), min(total, pos + len(y))
                            if bb > a:
                                babble[a:bb] += y[a - pos:bb - pos]
                            pos += len(y) + int(rng.uniform(0.2, 0.8) * SR)
                            j += 1
                    fg_db = 10 * math.log10(np.mean([10 ** (lvl[s] / 10) for s in spks]))
                    snr = rng.uniform(5, 10)
                    babble *= 10 ** ((fg_db - snr) / 20) / (np.sqrt(np.mean(babble ** 2)) + 1e-12)
                    mix += babble
                ndb = -45 + rng.uniform(-2, 2)
                noise = 0.8 * pink_noise(total, nrng) + 0.2 * nrng.standard_normal(total).astype(np.float32)
                mix += noise * (10 ** (ndb / 20) / (np.sqrt(np.mean(noise ** 2)) + 1e-12))
                pk = float(np.max(np.abs(mix)))
                if pk > 0.98:
                    mix *= 0.98 / pk
                stem = f"{scene}_{cond}_s{seed}"
                write_wav(out / f"{stem}.wav", mix)
                (out / f"{stem}.json").write_text(json.dumps({
                    "scene": scene, "cond": cond, "seed": seed, "speakers": spks,
                    "levels": {s: round(v, 1) for s, v in lvl.items()}, "snr_db": snr,
                    "wav": f"{stem}.wav", "duration": round(total / SR, 2), "turns": turns},
                    ensure_ascii=False, indent=0), encoding="utf-8")
                print(f"[build] {stem}: {total / SR:.0f}s / {len(spks)}人 / {len(turns)}ターン", flush=True)


# ====================================================================== 4) run

def segments_like_app(audio: np.ndarray, vad_threshold: float | None = None) -> list[tuple]:
    """アプリ(リモートサーバーの耳 = fine)と同じ区切り: AutoGain → silero VAD → StreamSegmenter。"""
    import sherpa_onnx
    from tsuyaku.audio import AutoGain, StreamSegmenter
    from tsuyaku.config import load_settings
    s = load_settings()
    conf = sherpa_onnx.VadModelConfig()
    conf.silero_vad.model = s.silero_vad
    conf.silero_vad.threshold = s.vad_threshold if vad_threshold is None else vad_threshold
    conf.silero_vad.min_silence_duration = 0.2
    conf.silero_vad.min_speech_duration = s.min_speech_s
    conf.silero_vad.max_speech_duration = max(30.0, s.max_speech_s)
    conf.sample_rate = SR
    vad = sherpa_onnx.VoiceActivityDetector(conf, buffer_size_in_seconds=60)
    agc = AutoGain(SR)
    sg = StreamSegmenter(SR, min_speech_s=s.min_speech_s, min_silence_s=0.2, cut_s=0.18,
                         max_s=s.max_speech_s)
    out = []
    for i in range(0, len(audio) + SR, 512):
        ch = audio[i:i + 512] if i < len(audio) else np.zeros(512, np.float32)
        g = agc(ch)
        vad.accept_waveform(g)
        while not vad.empty():
            vad.pop()
        for seg, lo, hi, pre in sg.feed(g, vad.is_speech_detected()):
            out.append((seg, lo, hi, pre))
    return out


def _gold(seg_lo: float, seg_hi: float, turns: list[dict]) -> str | None:
    """区間の主な話者(0.2秒以上重なり、重なりの7割以上がその人)。無ければ None。"""
    ov: dict[str, float] = {}
    for t in turns:
        o = min(seg_hi, t["end"]) - max(seg_lo, t["start"])
        if o > 0:
            ov[t["spk"]] = ov.get(t["spk"], 0.0) + o
    if not ov:
        return None
    s, o = max(ov.items(), key=lambda kv: kv[1])
    if o < 0.2 or o < 0.7 * sum(ov.values()):
        return "?"          # 重なりが小さい/2人が混ざる: 正解が決まらない(数えない)
    return s


def score(pred: list, gold: list, durs: list) -> dict:
    """1対1の対応(一致秒数が最大)を全通りで探して正解率など。"""
    P = sorted({p for p in pred if p is not None})
    G = sorted({g for g in gold})
    W = {(p, g): 0.0 for p in P for g in G}
    for p, g, d in zip(pred, gold, durs):
        if p is not None:
            W[(p, g)] += d
    best, best_map = -1.0, {}
    small, big = (P, G) if len(P) <= len(G) else (G, P)
    for perm in itertools.permutations(big, len(small)):
        pairs = list(zip(small, perm)) if small is P else [(b, a) for a, b in zip(small, perm)]
        v = sum(W[pg] for pg in pairs)
        if v > best:
            best, best_map = v, dict(pairs)
        if len(big) > 8:
            break
    tot = sum(durs)
    assigned = sum(d for p, d in zip(pred, durs) if p is not None)
    # 話した時間が全体の5%以上の話者だけを数えた人数(数区間だけの「幽霊の話者」を除く)
    per = {p: sum(d for q, d in zip(pred, durs) if q == p) for p in P}
    n_major = sum(1 for p in P if per[p] >= 0.05 * max(assigned, 1e-9))
    return {"acc": round(100 * max(best, 0) / max(tot, 1e-9), 1),
            "coverage": round(100 * assigned / max(tot, 1e-9), 1),
            "acc_assigned": round(100 * max(best, 0) / max(assigned, 1e-9), 1),
            "n_pred": len(P), "n_major": n_major, "n_gold": len(G), "map": best_map}


def run(conv_dir: Path, models: list[str], out: Path, thresholds: list[float],
        min_s: float = 0.6, new_min_s=(1.2,), only: str = "", confirms=(1,),
        update_ths=(None,), merge_ths=(None,)) -> None:
    from eval_conv import read_wav
    from tsuyaku.speakers import SpeakerTracker, refine_labels
    out.mkdir(parents=True, exist_ok=True)
    convs = sorted(p for p in conv_dir.glob("*.json") if only in p.name)
    rows = []
    for mp in models:
        mname = Path(mp).stem
        trk = SpeakerTracker(mp, max_speakers=8, min_s=min_s)
        ext = trk._ext
        cache_p = out / f"emb_{mname}.npz"
        cache = dict(np.load(cache_p)) if cache_p.exists() else {}
        times = []
        for cp in convs:
            meta = json.loads(cp.read_text(encoding="utf-8"))
            audio = read_wav(conv_dir / meta["wav"])
            key = cp.stem
            segs = segments_like_app(audio)
            info = []
            embs = []
            for k, (seg, lo, hi, pre) in enumerate(segs):
                g = _gold(lo + pre, hi, meta["turns"])
                d = len(seg) / SR
                ck = f"{key}:{k}"
                if ck in cache:
                    e = cache[ck]
                else:
                    t0 = time.time()
                    e = trk.embed(np.asarray(seg, np.float32))
                    times.append((d, time.time() - t0))
                    cache[ck] = e
                embs.append(e)
                info.append((g, d))
            for th, nm, cf, ut, mt in itertools.product(thresholds, new_min_s, confirms,
                                                        update_ths, merge_ths):
                if ut is not None and ut < th:
                    continue
                trk.threshold = th
                trk._cl.new_min_s = nm
                trk._cl.confirm = cf
                trk._cl.update_th = ut
                trk._cl.merge_th = mt
                trk.reset()
                pred = [trk._cl.assign(e, d) if e is not None else None
                        for e, (g, d) in zip(embs, info)]
                keep = [(p, g, d) for p, (g, d) in zip(pred, info) if g not in (None, "?")]
                sc = score([p for p, _, _ in keep], [g for _, g, _ in keep], [d for _, _, d in keep])
                # 会議の後でメモを作るときは、合わせた話者を合わせ先に直して表示できる
                fin = score([trk._cl.canonical(p) for p, _, _ in keep], [g for _, g, _ in keep],
                            [d for _, _, d in keep])
                # 会議の後の付け直し(speakers.refine_labels)
                ref = refine_labels(embs, [d for _, d in info], [trk._cl.canonical(p) for p in pred])
                kr = [(p, g, d) for p, (g, d) in zip(ref, info) if g not in (None, "?")]
                rsc = score([p for p, _, _ in kr], [g for _, g, _ in kr], [d for _, _, d in kr])
                ghost = sum(1 for p, (g, d) in zip(pred, info) if g is None and p is not None)
                rows.append({"model": mname, "conv": key, "scene": meta["scene"], "cond": meta["cond"],
                             "n_spk": len(meta["speakers"]), "th": th, "new_min_s": nm, "confirm": cf,
                             "update_th": ut, "merge_th": mt, "acc_final": fin["acc"],
                             "n_pred_final": fin["n_pred"], "n_major_final": fin["n_major"],
                             "acc_refined": rsc["acc"], "n_major_refined": rsc["n_major"],
                             "segs": len(keep),
                             "ghost_assigned": ghost, **{k: v for k, v in sc.items() if k != "map"}})
        np.savez(cache_p, **cache)
        if times:
            ds = np.array(times)
            print(f"[run] {mname}: 区間 {len(ds)}本 特徴 平均 {ds[:, 1].mean() * 1000:.0f}ms "
                  f"最大 {ds[:, 1].max() * 1000:.0f}ms(区間 平均 {ds[:, 0].mean():.1f}秒 最大 {ds[:, 0].max():.1f}秒)",
                  flush=True)
    (out / "rows.json").write_text(json.dumps(rows, ensure_ascii=False, indent=0), encoding="utf-8")
    summarize(rows, out / "summary.md")


# ====================================================================== 5) ami(本物の会議)

def ami_turns(ann: Path, meeting: str, gap_s: float = 0.5) -> list[dict]:
    """AMI の手作業の書き起こし(words/<id>.<A-D>.words.xml、ann_dir か ami_manual.zip)→
    話者ごとの発話区間(語の間が gap_s 秒以内なら繋ぐ)。XML は正規表現で <w> の時刻だけ読む。"""
    import zipfile
    rx = re.compile(r'<w [^>]*?starttime="([\d.]+)"[^>]*?endtime="([\d.]+)"(?![^>]*punc="true")[^>]*>')
    texts: dict[str, str] = {}
    d = ann / "words"
    for ch in "ABCDE":
        f = d / f"{meeting}.{ch}.words.xml"
        if f.is_file():
            texts[ch] = f.read_text(encoding="latin-1")
    if not texts and (ann / "ami_manual.zip").is_file():
        with zipfile.ZipFile(ann / "ami_manual.zip") as z:
            for ch in "ABCDE":
                n = f"words/{meeting}.{ch}.words.xml"
                if n in z.namelist():
                    texts[ch] = z.read(n).decode("latin-1")
    turns = []
    for ch, t in texts.items():
        cur = None
        for a, b in ((float(x), float(y)) for x, y in rx.findall(t)):
            if b <= a:
                continue
            if cur and a - cur[1] <= gap_s:
                cur[1] = max(cur[1], b)
            else:
                if cur:
                    turns.append({"spk": f"{meeting}:{ch}", "start": cur[0], "end": cur[1]})
                cur = [a, b]
        if cur:
            turns.append({"spk": f"{meeting}:{ch}", "start": cur[0], "end": cur[1]})
    return sorted(turns, key=lambda t: t["start"])


def ami(ami_dir: Path, meetings: list[str], mics: list[str], models: list[str], out: Path,
        thresholds: list[float], update_ths=(0.5,), merge_ths=(0.7,), audio_dirs=()) -> None:
    """本物の4人会議(AMI)で、アプリと同じ区切りの区間ごとに話者を付けて正解率を出す。
    マイク: Mix-Headset(各人の口元マイクを混ぜたもの)/ Array1-01(卓上マイク1本 = このアプリの使い方に近い)。"""
    from eval_conv import read_wav
    from tsuyaku.speakers import SpeakerTracker, refine_labels
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for mp in models:
        mname = Path(mp).stem
        trk = SpeakerTracker(mp)
        cache_p = out / f"emb_{mname}.npz"
        cache = dict(np.load(cache_p)) if cache_p.exists() else {}
        for mt in meetings:
            turns = ami_turns(ami_dir / "ann", mt)
            for mic in mics:
                # 音声は ami_dir か audio_dirs(他の担当が落とした置き場を読むだけ)から探す
                wav = next((d / f"{mt}.{mic}.wav" for d in (ami_dir, *audio_dirs)
                            if (d / f"{mt}.{mic}.wav").is_file()), ami_dir / f"{mt}.{mic}.wav")
                if not wav.is_file() or not turns:
                    print(f"[ami] {mt} {mic}: 音声か書き起こしが無い → 飛ばす", flush=True)
                    continue
                audio = read_wav(wav)
                segs = segments_like_app(audio)
                embs, info = [], []
                t0 = time.time()
                for k, (seg, lo, hi, pre) in enumerate(segs):
                    ck = f"{mt}:{mic}:{k}"
                    if ck not in cache:
                        e = trk.embed(np.asarray(seg, np.float32))
                        cache[ck] = e if e is not None else np.zeros(1, np.float32)
                    e = cache[ck]
                    embs.append(e if e.size > 1 else None)
                    info.append((_gold(lo + pre, hi, turns), len(seg) / SR))
                print(f"[ami] {mt} {mic}: {len(audio) / SR / 60:.1f}分 区間 {len(segs)} "
                      f"(特徴 {time.time() - t0:.0f}s)", flush=True)
                for th, ut, mg in itertools.product(thresholds, update_ths, merge_ths):
                    if ut is not None and ut < th:
                        continue
                    trk.threshold = th
                    trk._cl.update_th = ut
                    trk._cl.merge_th = mg
                    trk.reset()
                    pred = [trk._cl.assign(e, d) if e is not None else None for e, (g, d) in zip(embs, info)]
                    keep = [(p, g, d) for p, (g, d) in zip(pred, info) if g not in (None, "?")]
                    sc = score([p for p, _, _ in keep], [g for _, g, _ in keep], [d for _, _, d in keep])
                    fin = score([trk._cl.canonical(p) for p, _, _ in keep], [g for _, g, _ in keep],
                                [d for _, _, d in keep])
                    ref = refine_labels(embs, [d for _, d in info], [trk._cl.canonical(p) for p in pred])
                    kr = [(p, g, d) for p, (g, d) in zip(ref, info) if g not in (None, "?")]
                    rsc = score([p for p, _, _ in kr], [g for _, g, _ in kr], [d for _, _, d in kr])
                    rows.append({"model": mname, "meeting": mt, "mic": mic, "th": th, "update_th": ut,
                                 "merge_th": mg, "segs": len(segs), "scored": len(keep),
                                 "mixed": sum(1 for g, _ in info if g == "?"),
                                 "nogold": sum(1 for g, _ in info if g is None),
                                 "ghost_assigned": sum(1 for p, (g, _) in zip(pred, info) if g is None and p),
                                 **{k: v for k, v in sc.items() if k != "map"},
                                 "acc_final": fin["acc"], "n_major_final": fin["n_major"],
                                 "acc_refined": rsc["acc"], "n_major_refined": rsc["n_major"]})
        np.savez(cache_p, **cache)
    (out / "rows.json").write_text(json.dumps(rows, ensure_ascii=False, indent=0), encoding="utf-8")
    lines = ["| モデル | マイク | th/update/merge | 会議 | 正解率 | 会議後(合わせ) | 会議後(付け直し) | 判断した割合 | 人数(全部/5%以上/正解) | 区間(採点/混ざり/声なし) |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['model'][:24]} | {r['mic']} | {r['th']}/{r['update_th']}/{r['merge_th']} | {r['meeting']} "
                     f"| {r['acc']} | {r['acc_final']} | {r['acc_refined']} | {r['coverage']} | {r['n_pred']}/{r['n_major_final']}/{r['n_gold']} "
                     f"| {r['scored']}/{r['mixed']}/{r['nogold']} |")
    txt = "\n".join(lines)
    (out / "summary.md").write_text(txt + "\n", encoding="utf-8")
    print(txt)


# ====================================================================== 6) guess(短い区間の見当)

def guess_eval(ami_dir: Path | None, audio_dirs, meetings_dir: Path | None, model: str,
               out: Path | None = None, gap_s: float = 1.2) -> str:
    """SpeakerTracker.guess の使い道(Endpointer の「話者が違えば繋がない」)の確かめ。

    アプリと同じ区切りの区間を順に assign していき、1.2秒未満の区間が話者の付いた直前の区間から
    gap_s 秒以内に始まるたびに、その時点で different_speaker(区間, 直前の話者) を取る。
    正解(書き起こしの話者)で、直前と同じ人か別の人かが分かっている組だけを数える。
      別の人を別と言えた割合(高いほど繋ぎ違いが減る)/ 同じ人を別と言ってしまった割合(低いほど
      1つの文が割れない)を margin ごとに出す(判定は SpeakerTracker.different_speaker)。"""
    from eval_conv import read_wav
    from tsuyaku.speakers import SpeakerTracker
    trk = SpeakerTracker(model)
    sets = []
    if ami_dir is not None:
        for mt in ["ES2004a", "TS3003a", "IS1009a", "EN2002a"]:
            turns = ami_turns(ami_dir / "ann", mt)
            for mic in ["Mix-Headset", "Array1-01"]:
                wav = next((d / f"{mt}.{mic}.wav" for d in (ami_dir, *audio_dirs)
                            if (d / f"{mt}.{mic}.wav").is_file()), None)
                if wav is not None and turns:
                    sets.append((f"AMI {mic}", mt, read_wav(wav), turns))
    if meetings_dir is not None:
        for jp in sorted(meetings_dir.glob("*.json")):
            meta = json.loads(jp.read_text(encoding="utf-8"))
            if "gold" not in meta:
                continue
            turns = [{"spk": g["spk"], "start": g["start"], "end": g["end"]} for g in meta["gold"]]
            sets.append(("擬似会議", jp.stem, read_wav(jp.parent / meta["wav"]), turns))
    margins = [0.0, 0.1, 0.2]
    res: dict = {}
    for grp, name, audio, turns in sets:
        segs = segments_like_app(audio)
        trk.reset()
        prev = None   # (話者 id, 正解) 直前の話者の付いた区間
        prev_hi = None
        for seg, lo, hi, pre in segs:
            x = np.asarray(seg, np.float32)
            d = len(x) / SR
            g = _gold(lo + pre, hi, turns)
            # 1.2秒未満の区間が、話者の付いた直前の区間から gap_s 秒以内に続いた組(判定は assign の前)
            if d < 1.2 and prev is not None and prev_hi is not None and (lo + pre) - prev_hi <= gap_s \
                    and g not in (None, "?") and prev[1] not in (None, "?"):
                diff = g != prev[1]
                r = res.setdefault(grp, {"n_diff": 0, "n_same": 0, "none": 0,
                                         **{m: [0, 0] for m in margins}})
                r["n_diff" if diff else "n_same"] += 1
                for m in margins:
                    v = trk.different_speaker(x, prev[0], margin=m)
                    if v is None and m == margins[0]:
                        r["none"] += 1
                    r[m][0 if diff else 1] += int(bool(v))
            spk = trk.assign(x) if d >= trk._cl.min_s else None
            if spk is not None:
                prev = (spk, g)
            prev_hi = hi
    lines = ["| 組 | 別の人の組 | 同じ人の組 | 判断できない | " +
             " | ".join(f"margin {m}: 別を別 / 同じを別" for m in margins) + " |",
             "|---|---|---|---|" + "---|" * len(margins)]
    tot = {"n_diff": 0, "n_same": 0, "none": 0, **{m: [0, 0] for m in margins}}
    for r in res.values():
        for k in ("n_diff", "n_same", "none"):
            tot[k] += r[k]
        for m in margins:
            tot[m][0] += r[m][0]
            tot[m][1] += r[m][1]
    for grp, r in list(res.items()) + [("全部", tot)]:
        cells = [f"{100 * r[m][0] / max(1, r['n_diff']):.0f}% / {100 * r[m][1] / max(1, r['n_same']):.0f}%"
                 for m in margins]
        lines.append(f"| {grp} | {r['n_diff']} | {r['n_same']} | {r['none']} | " + " | ".join(cells) + " |")
    txt = "\n".join(lines)
    print(txt)
    if out:
        out.write_text(txt + "\n", encoding="utf-8")
    return txt


def summarize(rows: list[dict], path: Path | None = None) -> str:
    """モデル×しきい値ごと、条件(quiet/venue)×人数(2/3〜4)で平均。"""
    lines = ["| モデル | th/new_min_s/confirm/update_th/merge_th | 条件 | 人数 | 会話 | 正解率 | 判断した割合 | 判断した区間の正解率 | 人数の差(平均/当たり) | ざわめきに付けた |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    keyf = lambda r: (r["model"], f'{r["th"]}/{r.get("new_min_s", "")}/{r.get("confirm", "")}'
                      f'/{r.get("update_th")}/{r.get("merge_th")}', r["cond"],
                      "2" if r["n_spk"] == 2 else "3-4")
    groups: dict = {}
    for r in rows:
        groups.setdefault(keyf(r), []).append(r)
    for k in sorted(groups):
        g = groups[k]
        m = lambda f: sum(r[f] for r in g) / len(g)
        dn = [r["n_pred"] - r["n_gold"] for r in g]
        lines.append(f"| {k[0]} | {k[1]} | {k[2]} | {k[3]} | {len(g)} | {m('acc'):.1f} | {m('coverage'):.1f} "
                     f"| {m('acc_assigned'):.1f} | {sum(dn) / len(dn):+.2f} / {sum(1 for x in dn if x == 0)}/{len(dn)} "
                     f"| {sum(r['ghost_assigned'] for r in g)} |")
    txt = "\n".join(lines)
    if path:
        path.write_text(txt + "\n", encoding="utf-8")
    print(txt)
    return txt


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prep")
    p.add_argument("--src", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--per-lang", type=int, default=8)
    p.add_argument("--per-spk", type=int, default=25)
    v = sub.add_parser("verify")
    v.add_argument("--bank", type=Path, required=True)
    v.add_argument("--models", required=True)
    v.add_argument("--save", type=Path)
    b = sub.add_parser("build")
    b.add_argument("--bank", type=Path, required=True)
    b.add_argument("--fleurs-clips", type=Path, required=True)
    b.add_argument("--out", type=Path, required=True)
    b.add_argument("--seeds", default="1")
    b.add_argument("--dur", type=float, default=150.0)
    r = sub.add_parser("run")
    r.add_argument("--conv", type=Path, required=True)
    r.add_argument("--models", required=True)
    r.add_argument("--out", type=Path, required=True)
    r.add_argument("--th", default="0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65")
    r.add_argument("--min-s", type=float, default=0.6)
    r.add_argument("--new-min-s", default="1.2")
    r.add_argument("--confirm", default="1")
    r.add_argument("--update-th", default="none")
    r.add_argument("--merge-th", default="none")
    r.add_argument("--only", default="")
    m = sub.add_parser("ami", help="本物の4人会議(AMI)で評価")
    m.add_argument("--ami", type=Path, required=True, help="<id>.<mic>.wav と ann/(words/ か ami_manual.zip)")
    m.add_argument("--meetings", default="ES2004a,TS3003a,IS1009a,EN2002a")
    m.add_argument("--mics", default="Mix-Headset,Array1-01")
    m.add_argument("--models", required=True)
    m.add_argument("--out", type=Path, required=True)
    m.add_argument("--th", default="0.3,0.35,0.4,0.45,0.5")
    m.add_argument("--update-th", default="0.5")
    m.add_argument("--merge-th", default="0.7")
    m.add_argument("--audio", default="", help="音声を探す別の置き場(カンマ区切り)")
    gq = sub.add_parser("guess", help="短い区間の話者の見当(guess)が別の人を言い当てるか")
    gq.add_argument("--ami", type=Path, default=None)
    gq.add_argument("--audio", default="")
    gq.add_argument("--meetings", type=Path, default=None, help="eval_meeting.py build の出力")
    gq.add_argument("--model", required=True)
    gq.add_argument("--save", type=Path, default=None)
    a = ap.parse_args()
    if a.cmd == "prep":
        prep(a.src, a.out, a.per_lang, a.per_spk)
    elif a.cmd == "verify":
        verify(a.bank, a.models.split(","), a.save)
    elif a.cmd == "build":
        build(a.bank, a.fleurs_clips, a.out, tuple(int(x) for x in a.seeds.split(",")), a.dur)
    elif a.cmd == "guess":
        guess_eval(a.ami, [Path(x) for x in a.audio.split(",") if x], a.meetings, a.model, a.save)
    elif a.cmd == "ami":
        nn = lambda v: None if v == "none" else float(v)
        ami(a.ami, a.meetings.split(","), a.mics.split(","), a.models.split(","), a.out,
            [float(x) for x in a.th.split(",")], [nn(x) for x in a.update_th.split(",")],
            [nn(x) for x in a.merge_th.split(",")],
            [Path(x) for x in a.audio.split(",") if x])
    elif a.cmd == "run":
        run(a.conv, a.models.split(","), a.out, [float(x) for x in a.th.split(",")],
            a.min_s, [float(x) for x in a.new_min_s.split(",")], a.only,
            [int(x) for x in a.confirm.split(",")],
            [None if x == "none" else float(x) for x in a.update_th.split(",")],
            [None if x == "none" else float(x) for x in a.merge_th.split(",")])


if __name__ == "__main__":
    main()
