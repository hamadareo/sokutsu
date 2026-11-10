"""擬似会議の通しテスト(会議メモ): 台本の分かっている会議を合成音声で作り、アプリそのもの
(Bridge._conversation)にマイクの代わりに実時間で流して、認識 → 話者の見分け → 記録 →
live のメモ作り(notes_make("live") の道、付け直し _refine_speakers も通る)まで通す。

なぜ: 会議メモは部品ごと(話者の見分け・メモの LLM)には測っているが、本物の道を通したときに
どこで崩れるか(認識の誤りが名前ややることに響く、区切りと話者がずれる、付け直しと
メモの話者の対応など)を、正解の分かる会議で見たい。

台本(SCRIPTS): 話者・名前・自己紹介/呼びかけ・決定事項・やること(担当・期限)・質問が分かっている。
  日本語の3〜4人会議 3本(チームの定例 / 学会のブース準備(会場の雑音)/ 同性2人ずつの4人(会場))、
  日英の会議 1本(通訳+会議メモ)。声: 日本語 = この Mac の VOICEVOX(127.0.0.1:50021、話者ごとに
  違う声、男女混ぜる)、英語 = macOS say(Samantha / Daniel)。
  声の選び方(11/06): VOICEVOX の声は同じ合成エンジンなので、話者の特徴(CAM++)が人の声より
  ずっと似ている(四国めたん−春日部つむぎ 0.71、春日部つむぎ−冥鳴ひまり 0.70。人の別人は AMI で
  おおむね 0.3 未満、同じ声の2文は 0.79〜0.87)。1回目(v1)はこの似た声で、女性2声が1人にまとまった
  (話者の見分けの合わせ merge_th 0.7 を超える)。台本の通しテストとして意味があるよう、互いの類似が
  0.3 前後以下の声に選び直した: 女性 波音リツ9 / No.7 29 / 小夜46、男性 青山龍星13 / 玄野武宏11 /
  白上虎太郎12(リツ−No.7 0.21、龍星−虎太郎 0.16、龍星−武宏 0.23)。間・相づち(前の発言の終わりに
  少し重なる)・話者交代の時々の重なり・部屋の雑音(静か: -50dBFS / 会場: FLEURS のざわめき SNR 12dB)。

手順(リポジトリ直下):
  build: .venv/bin/python scripts/eval_meeting.py build --out DIR [--only ja_lab]
         → DIR/<id>.wav と DIR/<id>.json(台本・各行の時刻と話者)。声は DIR/tts/ にキャッシュ。
  run:   .venv/bin/python scripts/eval_meeting.py run --meeting DIR/<id>.json --out RES
         実時間で流す(長さ + メモ作りの時間)。ユーザーがアプリを使っている間は待つ。
  eval:  .venv/bin/python scripts/eval_meeting.py eval --meeting DIR/<id>.json --out RES [--save eval/meeting]
指標: 認識(台本との CER、言語別)、話者(行ごとの正解率 = 字数の重み・1対1の対応、人数)、
  名前(メモの参加者の名前が台本と合うか、根拠つきか)、決定事項/やること(担当・期限)/質問の再現
  (台本の各項目がメモにあるかを リモートサーバーの LLM で判定)、作り話(台本に無い名前・やること)、
  メモができるまでの秒数、記録とメモの話者の対応(やることの担当の id が正しい人か)。
生成物(音声・記録)は git に入れない。eval/meeting/ には md と小さい JSON だけ。
"""
from __future__ import annotations

import argparse
import difflib
import hashlib
import itertools
import json
import math
import os
import random
import re
import subprocess
import sys
import tempfile
import threading
import time
import wave
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
SR = 16000

# ====================================================================== 台本

SCRIPTS = [
    {
        "id": "ja_lab", "title": "チームの定例", "mode": "notes", "pair": "ja", "cond": "quiet",
        "people": {
            "P1": {"name": "山田", "lang": "ja", "voice": "vv:13", "gender": "M"},
            "P2": {"name": "田中", "lang": "ja", "voice": "vv:9", "gender": "F"},
            "P3": {"name": "佐藤", "lang": "ja", "voice": "vv:11", "gender": "M"},
            "P4": {"name": "鈴木", "lang": "ja", "voice": "vv:29", "gender": "F"},
        },
        "lines": [
            ["P1", "それでは定例ミーティングを始めます。司会の山田です。よろしくお願いします。"],
            ["P2", "よろしくお願いします。田中です。"],
            ["P3", "佐藤です。よろしくお願いします。"],
            ["P4", "鈴木です。お願いします。"],
            ["P1", "今日の議題は、来月の学会発表の準備と、実験用のGPUの使い方の二つです。まず田中さん、発表の準備はどうですか。"],
            ["P2", "スライドの下書きは半分くらいできています。ただ、実験の結果のグラフがまだ足りません。"],
            ["P1", "なるほど。", "bc"],
            ["P2", "佐藤さんの追加実験の結果が出たら、グラフに入れたいと思っています。"],
            ["P3", "追加実験は昨日始めました。結果は今週の金曜日に出る予定です。"],
            ["P4", "はい。", "bc"],
            ["P1", "では、佐藤さんは金曜日までに追加実験の結果を田中さんに送ってください。"],
            ["P3", "わかりました。金曜日までに送ります。"],
            ["P4", "一つ質問してもいいですか。GPUの予約は、今は誰が管理していますか。"],
            ["P1", "今は特に決まっていないですね。では、GPUの予約の管理は鈴木さんにお願いしてもいいですか。"],
            ["P4", "はい、私がやります。来週の月曜日までに予約表を作ります。"],
            ["P2", "発表の練習はいつにしますか。"],
            ["P1", "来週の木曜日の午後に、全員で発表練習をすることにしましょう。"],
            ["P3", "そうですね。", "bc"],
            ["P2", "わかりました。それまでにスライドを完成させます。"],
            ["P3", "ところで、論文の締め切りが延びるという話を聞きましたが、本当ですか。"],
            ["P1", "それはまだ分かりません。私が学会の事務局に確認しておきます。"],
            ["P1", "最後に確認します。佐藤さんは金曜日までに実験結果を送る。鈴木さんは月曜日までにGPUの予約表を作る。田中さんは木曜日までにスライドを完成させる。発表練習は木曜日の午後です。"],
            ["P1", "今日はここまでにします。お疲れさまでした。"],
            ["P2", "お疲れさまでした。"],
        ],
        "decisions": ["来週の木曜日の午後に全員で発表練習をする",
                      "GPUの予約の管理は鈴木さんが担当する"],
        "todos": [{"task": "追加実験の結果を田中さんに送る", "owner": "P3", "due": "金曜日"},
                  {"task": "GPUの予約表を作る", "owner": "P4", "due": "来週の月曜日"},
                  {"task": "スライドを完成させる", "owner": "P2", "due": "木曜日"},
                  {"task": "論文の締め切りが延びるか学会の事務局に確認する", "owner": "P1", "due": ""}],
        "questions": ["論文の締め切りは延びるのか(未確認)"],
    },
    {
        "id": "ja_booth", "title": "学会のブース準備", "mode": "notes", "pair": "ja", "cond": "venue",
        "terms": ["A0", "山本", "伊藤", "中村", "懇親会"],
        "people": {
            "P1": {"name": "山本", "lang": "ja", "voice": "vv:46", "gender": "F"},
            "P2": {"name": "伊藤", "lang": "ja", "voice": "vv:12", "gender": "M"},
            "P3": {"name": "中村", "lang": "ja", "voice": "vv:9", "gender": "F"},
        },
        "lines": [
            ["P1", "みなさんお疲れさまです。山本です。今日は学会の展示ブースの準備について決めたいと思います。"],
            ["P3", "中村です。よろしくお願いします。"],
            ["P1", "伊藤さん、ポスターのサイズはどうしましょうか。"],
            ["P2", "会場の案内を見ると、A0サイズまで貼れるそうです。A0で印刷するのがいいと思います。"],
            ["P3", "うん、うん。", "bc"],
            ["P1", "では、ポスターはA0サイズで印刷することにします。印刷は伊藤さんにお願いできますか。"],
            ["P2", "はい。十一月二十日までに業者に頼んでおきます。"],
            ["P3", "デモはどうしますか。ノートパソコンは何台持っていきますか。"],
            ["P1", "二台あれば足りると思います。デモは二台のノートパソコンで行いましょう。"],
            ["P2", "そうですね。", "bc"],
            ["P1", "中村さん、デモの動画を作ってもらえますか。"],
            ["P3", "わかりました。来週の水曜日までに作ります。"],
            ["P2", "会場に電源はあるんでしょうか。延長コードも必要かもしれません。"],
            ["P1", "それは私が明日、運営に電話して確認します。"],
            ["P3", "あと、懇親会の参加費は研究費から出せるんですか。"],
            ["P1", "うーん、それは事務の方に聞かないと分からないですね。"],
            ["P2", "なるほど。", "bc"],
            ["P1", "では、今日決めたことをまとめます。ポスターはA0、印刷は伊藤さんが二十日まで、デモの動画は中村さんが水曜日まで、電源の確認は私が明日やります。"],
            ["P3", "はい、了解です。"],
            ["P2", "お疲れさまでした。"],
        ],
        "decisions": ["ポスターはA0サイズで印刷する", "デモは二台のノートパソコンで行う"],
        "todos": [{"task": "ポスターの印刷を業者に頼む", "owner": "P2", "due": "十一月二十日"},
                  {"task": "デモの動画を作る", "owner": "P3", "due": "来週の水曜日"},
                  {"task": "会場の電源を運営に確認する", "owner": "P1", "due": "明日"}],
        "questions": ["懇親会の参加費を研究費から出せるか(未確認)"],
    },
    {
        "id": "ja_dev4", "title": "アプリ開発の進捗(同性2人ずつ)", "mode": "notes", "pair": "ja",
        "terms": ["高橋", "渡辺", "小川", "森"],
        "cond": "venue",
        "people": {
            "P1": {"name": "高橋", "lang": "ja", "voice": "vv:13", "gender": "M"},
            "P2": {"name": "渡辺", "lang": "ja", "voice": "vv:12", "gender": "M"},
            "P3": {"name": "小川", "lang": "ja", "voice": "vv:29", "gender": "F"},
            "P4": {"name": "森", "lang": "ja", "voice": "vv:9", "gender": "F"},
        },
        "lines": [
            ["P1", "高橋です。今週の開発の進捗を確認します。渡辺さんからお願いします。"],
            ["P2", "渡辺です。ログイン画面は完成しました。テストもほぼ終わっています。"],
            ["P1", "ありがとうございます。小川さんはどうですか。"],
            ["P3", "小川です。通知の機能でバグが一つ残っています。明日までに直す予定です。"],
            ["P4", "森です。デザインの修正案を作りました。後でみなさんに送ります。"],
            ["P1", "はい。", "bc"],
            ["P2", "リリースの日はいつにしますか。"],
            ["P1", "十二月一日にリリースすることに決めましょう。"],
            ["P3", "わかりました。", "bc"],
            ["P4", "デザイン案は今日中に送ります。"],
            ["P1", "では森さんは今日中にデザイン案をみんなに送ってください。小川さんは明日までに通知のバグを直してください。"],
            ["P3", "はい、明日までに直します。"],
            ["P2", "テストの環境はどうしますか。サーバーが足りない気がします。"],
            ["P1", "サーバーを一台追加するかどうかは、来週の会議で決めます。"],
            ["P2", "了解です。それまでに必要な台数を見積もっておきます。"],
            ["P1", "渡辺さん、よろしくお願いします。今日はここまでにします。"],
        ],
        "decisions": ["十二月一日にリリースする"],
        "todos": [{"task": "デザイン案をみんなに送る", "owner": "P4", "due": "今日中"},
                  {"task": "通知のバグを直す", "owner": "P3", "due": "明日"},
                  {"task": "サーバーの必要な台数を見積もる", "owner": "P2", "due": "来週の会議まで"}],
        "questions": ["サーバーを一台追加するか(来週の会議で決める)"],
    },
    {
        "id": "jaen_collab", "title": "日英の共同研究の打ち合わせ", "mode": "both", "pair": "ja-en",
        "cond": "quiet",
        "people": {
            "P1": {"name": "山田", "lang": "ja", "voice": "vv:13", "gender": "M"},
            "P2": {"name": "小林", "lang": "ja", "voice": "vv:9", "gender": "F"},
            "P3": {"name": "Emily", "lang": "en", "voice": "say:Samantha", "gender": "F",
                   "aka": ["エミリー"]},
            "P4": {"name": "David", "lang": "en", "voice": "say:Daniel", "gender": "M",
                   "aka": ["デビッド", "デイビッド", "デヴィッド", "デービッド"]},
        },
        "lines": [
            ["P1", "こんにちは、山田です。今日は共同研究の打ち合わせをしましょう。"],
            ["P3", "Hi everyone, I'm Emily from the Boston team. Thank you for joining."],
            ["P4", "And I'm David. Nice to meet you all."],
            ["P2", "小林です。よろしくお願いします。"],
            ["P1", "まず、データの共有の方法を決めたいです。Emilyさん、どう思いますか。"],
            ["P3", "I think we should share the dataset on a shared drive. It is the easiest way."],
            ["P1", "わかりました。データは共有ドライブで共有することにします。"],
            ["P4", "Okay.", "bc"],
            ["P2", "データの整理は私がやります。金曜日までに共有します。"],
            ["P3", "Great. I will send the annotation guideline by Wednesday."],
            ["P4", "I can set up the evaluation script. I will finish it by next Monday."],
            ["P1", "ありがとうございます。", "bc"],
            ["P2", "一つ質問です。このデータセットは論文と一緒に公開してもいいですか。"],
            ["P3", "That is a good question. We need to ask our legal team first."],
            ["P1", "では、次の打ち合わせは来週の火曜日にしましょう。"],
            ["P4", "Tuesday works for me. See you then."],
            ["P2", "お疲れさまでした。"],
        ],
        "decisions": ["データは共有ドライブで共有する", "次の打ち合わせは来週の火曜日"],
        "todos": [{"task": "データを整理して共有する", "owner": "P2", "due": "金曜日"},
                  {"task": "アノテーションのガイドラインを送る", "owner": "P3", "due": "水曜日"},
                  {"task": "評価スクリプトを用意する", "owner": "P4", "due": "来週の月曜日"}],
        "questions": ["データセットを論文と一緒に公開してよいか(法務に確認が必要)"],
    },
    {
        # v3(11/07): 議題つき(meta agenda)・会議中に印(mark)と自分のメモ(memo)を入れ、議題をチェックする。
        # events: 台本の行 at の話し終わりから after 秒で、画面のボタンと同じ Bridge の口を呼ぶ。
        "id": "ja_agenda", "title": "製品企画の定例(議題・印・メモつき)", "mode": "notes", "pair": "ja",
        "terms": ["岡田", "西村", "藤井", "経理"],
        "cond": "quiet",
        "agenda": "1. 新機能の優先順位\n2. 来月のユーザーテスト\n3. 予算の確認",
        "people": {
            "P1": {"name": "岡田", "lang": "ja", "voice": "vv:13", "gender": "M"},
            "P2": {"name": "西村", "lang": "ja", "voice": "vv:9", "gender": "F"},
            "P3": {"name": "藤井", "lang": "ja", "voice": "vv:12", "gender": "M"},
        },
        "lines": [
            ["P1", "岡田です。今日は議題が三つあります。まず新機能の優先順位から決めましょう。"],
            ["P2", "西村です。ユーザーの要望が一番多いのは、検索の改善です。"],
            ["P3", "藤井です。私も検索の改善を最初にやるのがいいと思います。"],
            ["P1", "では、検索の改善を最優先にすることに決めます。"],
            ["P2", "はい。", "bc"],
            ["P1", "次に、来月のユーザーテストについてです。西村さん、参加者は集まりそうですか。"],
            ["P2", "今のところ五人決まっています。あと三人、来週の金曜日までに集めます。"],
            ["P3", "テストの台本は私が作ります。水曜日までに共有します。"],
            ["P1", "ありがとうございます。最後に予算ですが、追加の予算が出るかは、まだ分かりません。"],
            ["P3", "経理に確認したほうがいいですか。"],
            ["P1", "はい、私が明日、経理に確認します。今日はここまでです。"],
        ],
        "events": [
            {"at": 3, "after": 1.0, "do": "mark", "kind": "decision", "label": "検索を最優先"},
            {"at": 3, "after": 1.5, "do": "agenda", "k": 0, "done": True},
            {"at": 6, "after": 1.0, "do": "memo", "text": "参加者あと3人(金曜まで)"},
            {"at": 7, "after": 1.0, "do": "mark", "kind": "todo", "label": "台本は藤井さん"},
            {"at": 7, "after": 1.5, "do": "agenda", "k": 1, "done": True},
            {"at": 8, "after": 1.0, "do": "mark", "kind": "question", "label": "追加予算は未定"},
        ],
        "decisions": ["検索の改善を最優先にする"],
        "todos": [{"task": "ユーザーテストの参加者をあと三人集める", "owner": "P2", "due": "来週の金曜日"},
                  {"task": "テストの台本を作って共有する", "owner": "P3", "due": "水曜日"},
                  {"task": "追加の予算が出るか経理に確認する", "owner": "P1", "due": "明日"}],
        "questions": ["追加の予算が出るか(まだ分からない)"],
    },
]


# ====================================================================== 声と音

def _resample(x: np.ndarray, sr: int) -> np.ndarray:
    if sr == SR:
        return x.astype(np.float32)
    n_out = int(round(len(x) * SR / sr))
    X = np.fft.rfft(x)
    keep = min(len(X), n_out // 2 + 1)
    Y = np.zeros(n_out // 2 + 1, dtype=complex)
    Y[:keep] = X[:keep]
    return (np.fft.irfft(Y, n_out) * (n_out / len(x))).astype(np.float32)


def _read_wav_any(b: bytes) -> np.ndarray:
    import io
    with wave.open(io.BytesIO(b), "rb") as w:
        sr, ch, sw = w.getframerate(), w.getnchannels(), w.getsampwidth()
        raw = w.readframes(w.getnframes())
    assert sw == 2
    x = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    if ch > 1:
        x = x.reshape(-1, ch).mean(axis=1)
    return _resample(x, sr)


def tts(voice: str, text: str, cache: Path) -> np.ndarray:
    """vv:<id> = VOICEVOX(この Mac)、say:<声> = macOS say。16kHz mono。キャッシュする。"""
    cache.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha1(f"{voice}|{text}".encode()).hexdigest()[:16]
    p = cache / f"{key}.wav"
    if not p.exists():
        if voice.startswith("vv:"):
            import requests
            sid = voice[3:]
            url = os.environ.get("TSUYAKU_VOICEVOX_URL", "http://127.0.0.1:50021")
            q = requests.post(f"{url}/audio_query", params={"text": text, "speaker": sid}, timeout=60)
            q.raise_for_status()
            r = requests.post(f"{url}/synthesis", params={"speaker": sid}, json=q.json(), timeout=120)
            r.raise_for_status()
            x = _read_wav_any(r.content)
        elif voice.startswith("say:"):
            with tempfile.TemporaryDirectory() as td:
                f = Path(td) / "a.wav"
                subprocess.run(["say", "-v", voice[4:], "-o", str(f), "--file-format=WAVE",
                                "--data-format=LEI16@16000", text], check=True, timeout=120)
                x = _read_wav_any(f.read_bytes())
        else:
            raise ValueError(voice)
        from eval_conv import write_wav
        write_wav(p, x)
    from eval_conv import read_wav
    return read_wav(p)


def build(out: Path, only: str = "", seed: int = 1, babble_dir: Path | None = None) -> list[Path]:
    from eval_conv import active_rms, lowpass, pink_noise, read_wav, trim_silence, write_wav
    out.mkdir(parents=True, exist_ok=True)
    made = []
    for sc in SCRIPTS:
        if only and sc["id"] != only:
            continue
        rng = random.Random(f"{seed}-{sc['id']}")
        nrng = np.random.default_rng(seed)
        lvl = {p: -24.0 + rng.uniform(-4, 4) for p in sc["people"]}
        t = 1.0
        pieces, gold = [], []
        prev_end, prev_spk = None, None
        for k, ln in enumerate(sc["lines"]):
            spk, text = ln[0], ln[1]
            bc = len(ln) > 2 and ln[2] == "bc"
            pp = sc["people"][spk]
            x = trim_silence(tts(pp["voice"], text, out / "tts"))
            g = lvl[spk] + rng.uniform(-2, 2) - (4.0 if bc else 0.0)
            x = x * (10 ** (g / 20) / max(active_rms(x), 1e-6))
            if prev_end is None:
                start = t
            elif bc:
                start = prev_end - rng.uniform(0.2, 0.5)          # 相づちは前の発言の終わりに重なる
            elif spk == prev_spk:
                start = prev_end + rng.uniform(0.5, 1.0)
            else:
                u = rng.random()
                start = prev_end + (-rng.uniform(0.15, 0.35) if u < 0.15 else
                                    rng.uniform(0.25, 0.5) if u < 0.6 else rng.uniform(0.6, 1.2))
            start = max(start, 0.5)
            end = start + len(x) / SR
            gold.append({"i": k, "spk": spk, "name": pp["name"], "lang": pp["lang"], "text": text,
                         "bc": bc, "start": round(start, 3), "end": round(end, 3)})
            pieces.append((int(round(start * SR)), x))
            if not bc:
                prev_end, prev_spk = end, spk
            else:
                prev_end = max(prev_end, end)
        total = int((max(g["end"] for g in gold) + 2.0) * SR)
        mix = np.zeros(total, np.float32)
        for a, x in pieces:
            mix[a:a + len(x)] += x[: total - a]
        if sc["cond"] == "venue" and babble_dir and babble_dir.is_dir():
            files = sorted(babble_dir.rglob("*.wav"))
            babble = np.zeros(total, np.float32)
            for _ in range(8):
                pos = -int(rng.uniform(0, 8) * SR)
                fs = rng.sample(files, 3)
                j = 0
                while pos < total:
                    y = lowpass(trim_silence(read_wav(fs[j % 3])))
                    y *= 0.05 / max(active_rms(y), 1e-6)
                    a, b = max(0, pos), min(total, pos + len(y))
                    if b > a:
                        babble[a:b] += y[a - pos:b - pos]
                    pos += len(y) + int(rng.uniform(0.2, 0.8) * SR)
                    j += 1
            fg = 10 * math.log10(np.mean([10 ** (v / 10) for v in lvl.values()]))
            babble *= 10 ** ((fg - 12.0) / 20) / (np.sqrt(np.mean(babble ** 2)) + 1e-12)
            mix += babble
        ndb = -50.0 if sc["cond"] == "quiet" else -45.0
        noise = 0.8 * pink_noise(total, nrng) + 0.2 * nrng.standard_normal(total).astype(np.float32)
        mix += noise * (10 ** (ndb / 20) / (np.sqrt(np.mean(noise ** 2)) + 1e-12))
        pk = float(np.max(np.abs(mix)))
        if pk > 0.98:
            mix *= 0.98 / pk
        write_wav(out / f"{sc['id']}.wav", mix)
        meta = {k: v for k, v in sc.items() if k != "lines"}
        meta.update({"wav": f"{sc['id']}.wav", "duration": round(total / SR, 2), "gold": gold,
                     "levels": {k: round(v, 1) for k, v in lvl.items()}})
        jp = out / f"{sc['id']}.json"
        jp.write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"[build] {sc['id']}: {total / SR:.0f}秒 / {len(sc['people'])}人 / {len(gold)}行", flush=True)
        made.append(jp)
    return made


# ====================================================================== run(アプリそのもの)

def run(meeting_json: Path, out_root: Path, wait: bool = True, notes_lang: str = "ja",
        tail_s: float = 6.0, end: bool = False, end_after_s: float = 1.0,
        record: "str | None" = None, terms: str = "app") -> Path:
    """end=True: 「会議を終える」(Bridge.end_meeting)の道。話し終わって end_after_s 秒で
    end_meeting() を呼び(最後の発言が記録に残るかを見る)、返った id で notes_make。
    end=False: 会話を続けたまま notes_make("live")。"""
    from eval_live import app_in_use
    if wait:
        while app_in_use():
            print("[run] アプリが使われている: 1分待つ", flush=True)
            time.sleep(60)
    sys.path.insert(0, str(ROOT / "src"))
    import tsuyaku.__main__ as app
    from tsuyaku import audio as audio_mod
    from eval_conv import read_wav

    meta = json.loads(meeting_json.read_text(encoding="utf-8"))
    x = read_wav(meeting_json.parent / meta["wav"])
    out = out_root / meeting_json.stem
    out.mkdir(parents=True, exist_ok=True)
    rec: list = []
    lock = threading.Lock()

    def note(kind, **kw):
        with lock:
            rec.append((time.time(), kind, kw))

    feed_done = threading.Event()
    body_done = threading.Event()      # 台本の音声を流し終えた(この後は部屋の小さな雑音)
    clock: dict = {}

    class FileCapture(audio_mod.SpeechCapture):
        """マイクの代わりに WAV を実時間で流す(1チャンクの処理はアプリと同じ on_chunk)。
        区間を切る段(StreamSegmenter)の feed を包み、出た区間ごとに 開始・終わり・切った理由
        (silence = VAD の無音 / dip = 声ありのまま音量の落ち込み cut_s / max = 長さの上限 /
        flush = 読み上げ・終わりで閉じた)を記録する。アプリのコードは変えない。"""

        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            sg = self.segmenter
            feed0, flush0 = sg.feed, sg.flush_current

            def feed(chunk, speech):
                out = feed0(chunk, speech)
                for item in out:
                    why = "silence" if not speech else ("dip" if getattr(sg, "_cut_wait", False) else "max")
                    note("seg", lo=round(item[1], 3), hi=round(item[2], 3), pre=round(item[3], 3), why=why)
                return out

            def flush_current():
                out = flush0()
                for item in out:
                    note("seg", lo=round(item[1], 3), hi=round(item[2], 3), pre=round(item[3], 3), why="flush")
                return out
            sg.feed, sg.flush_current = feed, flush_current

        def start(self):
            self._thread = threading.Thread(target=self._run_file, daemon=True)
            self._thread.start()

        def _run_file(self):
            rng = np.random.default_rng(0)
            n, tail = len(x), int(tail_s * SR)
            t0 = time.time()
            clock["t0"] = t0
            note("feed_start")
            i = 0
            if end:
                tail = int(120 * SR)       # 「会議を終える」まで雑音を流し続ける(止めるのは end_meeting)
            while i < n + tail and not self._stop.is_set():
                d = t0 + (i + 512) / SR - time.time()
                if d > 0:
                    time.sleep(d)
                if i < n:
                    ch = x[i:i + 512]
                    if len(ch) < 512:
                        ch = np.concatenate([ch, np.zeros(512 - len(ch), np.float32)])
                else:
                    if not body_done.is_set():
                        note("audio_body_end")
                        body_done.set()
                    ch = (rng.standard_normal(512) * 3e-4).astype(np.float32)
                self.on_chunk(ch)
                i += 512
            note("audio_end")
            feed_done.set()

    class _NoTTS:
        def __init__(self, *a, **k):
            self.on_start = None

        def prepare(self, *a, **k):
            pass

        def say(self, *a, **k):
            pass

        def stop(self):
            pass

    notes_done = threading.Event()

    class UIRec:
        def put(self, evt):
            note("ui", evt=list(evt))
            if evt and evt[0] == "status":       # アプリの画面の行列(_pump)と同じく状態を覚える
                b._ui_status = (evt[1], time.time())
            if evt and evt[0] == "notes" and isinstance(evt[1], dict) and \
                    evt[1].get("stage") in ("done", "error"):
                notes_done.set()

    class RecEndpointer(app.Endpointer):
        """訳す単位(Endpointer が文末を見て繋いだもの)を記録する: 範囲・繋いだ区間の数・文・出し方。"""

        def push(self, *a, **k):
            us = super().push(*a, **k)
            self._rec(us, "push")
            return us

        def tick(self, *a, **k):
            us = super().tick(*a, **k)
            self._rec(us, "tick")
            return us

        def flush(self):
            us = super().flush()
            self._rec(us, "flush")
            return us

        def _rec(self, us, how):
            for u in us:
                note("unit", how=how, text=u.text, lang=u.lang, lo=round(u.lo, 3), hi=round(u.hi, 3),
                     speech_lo=round(getattr(u, "speech_lo", u.lo), 3), parts=u.parts)
    app.Endpointer = RecEndpointer
    app.SpeechCapture = FileCapture
    app.TTS = _NoTTS
    b = app.Bridge()
    b.s.log_dir = str(out / "applog")
    if record is not None:
        # 録音(W2)の設定をこの実行の中だけ上書き(config.json は書かない。ユーザーの設定は "off" のことがある)
        b.s.record_audio = record
    b.ui = UIRec()
    logf = (out / "app.log").open("w", encoding="utf-8")
    old_out = sys.stdout
    from eval_live import _Tee
    sys.stdout = _Tee(logf, old_out)
    # 議題(W3): 画面の起動と同じく meta に入れる(start() の extra と同じ道)
    try:
        from tsuyaku import inmeeting
        b._start_extra = inmeeting.meta_extra({"agenda": meta.get("agenda", "")})
        # 用語辞書(W3): app = アプリと同じ(保存された辞書)/ none = 無し / script = 台本の固有名詞
        tl = (inmeeting.load_terms() if terms == "app" else
              inmeeting.normalize_terms(meta.get("terms") or []) if terms == "script" else [])
        if tl:
            b._start_extra["terms"] = tl
    except Exception:  # noqa: BLE001  古い版(W3 前)
        pass
    th = threading.Thread(target=b._conversation, args=(meta["pair"], "", meta["mode"]), daemon=True)
    b._conv_th = th
    th.start()
    t_notes = t_done = None
    end_ret = None
    log_path = None
    ev_done: list = []

    def fire_events():
        """台本の events を、その行の話し終わり + after 秒(音声の時計)に、画面のボタンと同じ口で実行。"""
        gold = {g["i"]: g for g in meta["gold"]}
        evs = sorted(meta.get("events") or [], key=lambda e: gold[e["at"]]["end"] + e.get("after", 1.0))
        while "t0" not in clock:
            time.sleep(0.05)
        for e in evs:
            at = clock["t0"] + gold[e["at"]]["end"] + e.get("after", 1.0)
            while time.time() < at:
                time.sleep(0.02)
            try:
                if e["do"] == "mark":
                    r = b.mark_set("live", {"kind": e.get("kind", "star"), "label": e.get("label", "")})
                elif e["do"] == "memo":
                    r = b.memo_set("live", {"text": e["text"]})
                elif e["do"] == "agenda":
                    r = b.agenda_check("live", e["k"], e.get("done", True))
                else:
                    r = None
            except Exception as ex:  # noqa: BLE001
                r = f"error {type(ex).__name__}: {ex}"
            ev_done.append({**e, "t_audio": round(time.time() - clock["t0"], 2), "ret": r})
            note("event", ev=dict(e), ret=r)
    if meta.get("events"):
        threading.Thread(target=fire_events, daemon=True).start()
    ts0 = None
    try:
        if end:
            while not body_done.wait(1.0):
                if not th.is_alive():
                    raise RuntimeError("会話が始まらなかった(app.log を見る)")
            time.sleep(end_after_s)
            log_path = str(Path(b.log.path_jsonl)) if getattr(b, "log", None) else None
            ts0 = getattr(b.log, "ts0", None) if getattr(b, "log", None) else None
            t_end = time.time()
            end_ret = b.end_meeting()             # 画面の「会議を終える」と同じ
            note("end_meeting", ret=end_ret, took=round(time.time() - t_end, 2))
            t_notes = time.time()
            r = b.notes_make(end_ret.get("id"), notes_lang)
            note("notes_make", ret=r)
            notes_done.wait(600)
            t_done = time.time()
            _after_notes(b, end_ret.get("id"), log_path, out, notes_lang)
        else:
            while not feed_done.wait(1.0):
                if not th.is_alive():
                    raise RuntimeError("会話が始まらなかった(app.log を見る)")
            deadline = time.time() + 30
            while time.time() < deadline:      # 最後の訳・記録が出きるまで
                with lock:
                    last = max((t for t, k, kw in rec if k == "ui" and kw["evt"][0] in
                                ("utterance", "partial")), default=0)
                if time.time() - last > 4.0:
                    break
                time.sleep(0.5)
            log_path = str(Path(b.log.path_jsonl)) if getattr(b, "log", None) else None
            ts0 = getattr(b.log, "ts0", None) if getattr(b, "log", None) else None
            t_notes = time.time()
            r = b.notes_make("live", notes_lang)       # アプリのメモボタンと同じ道(付け直しも通る)
            note("notes_make", ret=r)
            notes_done.wait(600)
            t_done = time.time()
    finally:
        b.running.clear()
        th.join(timeout=20)
        try:
            b._shutdown()
        except Exception:  # noqa: BLE001
            pass
        sys.stdout = old_out
        logf.close()
    t0 = clock.get("t0", time.time())
    events = [{"t": round(t - t0, 3), "kind": k, **kw} for t, k, kw in rec]
    run_meta = {"meeting": str(meeting_json), "git": _git(), "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                "t0": t0, "notes_s": round(t_done - t_notes, 1) if t_notes and t_done else None,
                "model": getattr(b, "model_used", ""), "engine": getattr(b, "engine_used", ""),
                "log": log_path, "end": end, "end_ret": end_ret, "ts0": ts0, "events": ev_done,
                "terms": (getattr(b, "_start_extra", None) or {}).get("terms", []),
                "stt_prompt": getattr(getattr(b, "stt", None), "prompt", None)}
    (out / "events.json").write_text(json.dumps({"meta": run_meta, "events": events},
                                                ensure_ascii=False, indent=0, default=str),
                                     encoding="utf-8")
    return out


def _after_notes(b, mid, log_path, out: Path, lang: str) -> None:
    """メモの Markdown(書き出しと同じ notes_markdown)を残し、録音の m4a ができるのを待つ(最大60秒)。"""
    try:
        md = b.notes_markdown(mid, lang)
        (out / "notes.md").write_text(md if isinstance(md, str) else json.dumps(md, ensure_ascii=False),
                                      encoding="utf-8")
    except Exception as e:  # noqa: BLE001
        (out / "notes.md").write_text(f"(notes_markdown 失敗: {type(e).__name__}: {e})", encoding="utf-8")
    if not log_path:
        return
    m4a = Path(log_path).with_name(Path(log_path).stem + ".audio.m4a")
    caf = Path(log_path).with_name(Path(log_path).stem + ".audio.caf")
    t_end = time.time() + 60
    while time.time() < t_end and not m4a.exists() and caf.exists():
        time.sleep(0.5)


def _git() -> str:
    try:
        h = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"],
                           capture_output=True, text=True, timeout=5).stdout.strip()
        d = subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain", "src"],
                           capture_output=True, text=True, timeout=5).stdout.split()
        return h + ("+dirty" if d else "")
    except Exception:  # noqa: BLE001
        return "?"


# ====================================================================== eval

_P = re.compile(r"[\s\W_]+", re.UNICODE)


# 名前の異体字(音から字は決まらない: 渡辺/渡邊)。評価の判定でだけ同じ字とみる
_ITAI = str.maketrans({"濱": "浜", "邊": "辺", "邉": "辺", "髙": "高", "﨑": "崎", "齊": "斉", "齋": "斎",
                       "澤": "沢", "嶋": "島", "廣": "広", "國": "国", "櫻": "桜", "斎": "斉", "籐": "藤",
                       "槇": "槙", "瀨": "瀬", "萬": "万", "惠": "恵", "眞": "真", "德": "徳"})


def _itai(name: str) -> str:
    return (name or "").translate(_ITAI)


def _norm(t: str) -> str:
    import unicodedata
    return _P.sub("", unicodedata.normalize("NFKC", t or "").lower())


def _toks(t: str, lang: str) -> list:
    if lang == "en":
        return re.findall(r"[a-z0-9']+", (t or "").lower())
    return list(_norm(t))


def _ed(a: list, b: list) -> int:
    sm = difflib.SequenceMatcher(None, a, b, autojunk=False)
    return sum(max(i2 - i1, j2 - j1) for op, i1, i2, j1, j2 in sm.get_opcodes() if op != "equal")


def _best_map(W: dict, P: list, G: list) -> dict:
    best, bm = -1.0, {}
    small, big = (P, G) if len(P) <= len(G) else (G, P)
    for perm in itertools.permutations(big, len(small)):
        pairs = list(zip(small, perm)) if small is P else [(b, a) for a, b in zip(small, perm)]
        v = sum(W.get(pg, 0.0) for pg in pairs)
        if v > best:
            best, bm = v, dict(pairs)
    return bm


def _judge(meta: dict, notes: dict, map_spk: dict) -> dict:
    """台本の決定事項・やること・質問がメモにあるかを リモートサーバーの LLM で判定(JSON)。"""
    from tsuyaku import notes as notes_mod
    from tsuyaku.config import engine_candidates, load_settings
    from tsuyaku.translate import pick_model
    s = load_settings()
    _l, base, pref = engine_candidates("remote", s.host, None, None)[0]
    model = pick_model(base, pref)
    names = {p: v["name"] for p, v in meta["people"].items()}

    def who(x):
        if not x:
            return ""
        g = map_spk.get(x)
        nm = next((q.get("name") for q in notes.get("participants", []) if q.get("id") == x), "")
        return f"{x}(記録上の人={names.get(g, '不明')}, メモの名前={nm or 'なし'})"
    gold = {
        "decisions": meta["decisions"],
        "todos": [f"{t['task']} / 担当={names[t['owner']]} / 期限={t['due'] or 'なし'}" for t in meta["todos"]],
        "questions": meta["questions"],
    }
    got = {
        "decisions": notes.get("decisions", []),
        "todos": [f"{t.get('task')} / 担当={who(t.get('owner'))} / 期限={t.get('due') or 'なし'}"
                  for t in notes.get("todos", [])],
        "questions": notes.get("questions", []),
    }
    prompt = (
        "会議の正解の項目と、自動で作った会議メモの項目を比べて判定してください。JSON だけを返す。\n"
        "正解: " + json.dumps(gold, ensure_ascii=False) + "\n"
        "メモ: " + json.dumps(got, ensure_ascii=False) + "\n"
        "返す形: {\"decisions\": [正解の各項目がメモにあれば true],"
        " \"todos\": [{\"found\": 正解の各やることがメモにあるか, \"owner_ok\": 担当(記録上の人)が正しいか,"
        " \"due_ok\": 期限が合うか(正解が「なし」なら true)}],"
        " \"questions\": [正解の各質問がメモにあれば true],"
        " \"extra_decisions\": [正解のどれにも当たらないメモの決定事項], \"extra_todos\": [正解のどれにも"
        "当たらないメモのやること]}。言い換え・要約は同じ項目とみなす。")
    raw = notes_mod.chat(base, model, [{"role": "user", "content": prompt}], max_tokens=3000)
    try:
        return notes_mod.parse_json(raw)
    except Exception:  # noqa: BLE001
        return {"error": raw[:500]}


def evaluate(meeting_json: Path, out_root: Path, save: Path | None = None, judge: bool = True) -> dict:
    sys.path.insert(0, str(ROOT / "src"))
    from tsuyaku import notes as notes_mod
    meta = json.loads(meeting_json.read_text(encoding="utf-8"))
    out = out_root / meeting_json.stem
    ev = json.loads((out / "events.json").read_text(encoding="utf-8"))
    rm = ev["meta"]
    log = Path(rm["log"]) if rm.get("log") else next((out / "applog").glob("session-*.jsonl"))
    m = notes_mod.load_meeting(log)                 # 付け直し(refine 行)を当てた後の行
    notes = notes_mod.load_notes(log, "ja") or {}
    lines = m["lines"]
    gold = meta["gold"]
    t0 = rm["t0"]
    # ---- 記録の行 → 台本の行(時刻の近い候補から文字の重なりが一番多いもの) ----
    rows = []
    for ln in lines:
        ts = (ln.get("ts") or t0) - t0
        # 記録の行が確定した時刻 ts より前に話し始め、ts の30秒前より後に話し終えた台本の行が候補
        # (長い発言の前半だけが先に確定することがあるので、終わりが ts より後でもよい)。
        # 文字の重なり(3字以上の一致の合計)が一番多いもの。同じなら時刻の近いもの。
        cands = [g for g in gold if g["start"] - 1 <= ts and g["end"] >= ts - 30] or gold
        lt = _norm(ln["orig"])

        def ov(g):
            gt = _norm(g["text"])
            sm = difflib.SequenceMatcher(None, lt, gt, autojunk=False)
            return (sum(b.size for b in sm.get_matching_blocks() if b.size >= 3), -abs(ts - g["end"]))
        best = max(cands, key=ov)
        rows.append({"spk": ln["spk"], "lang": ln.get("lang"), "orig": ln["orig"], "trans": ln.get("trans", ""),
                     "ts": round(ts, 1), "gold_i": best["i"], "gold_spk": best["spk"],
                     "chars": max(1, len(lt)), "a0": ln.get("a0"), "a1": ln.get("a1")})
    # ---- 認識: 言語ごとに全体をつないで CER/WER ----
    asr = {}
    for L in sorted({g["lang"] for g in gold}):
        ref = [t for g in gold if g["lang"] == L for t in _toks(g["text"], L)]
        hyp = [t for r in rows if (r["lang"] or L) == L for t in _toks(r["orig"], L)]
        asr[L] = {"metric": "WER" if L == "en" else "CER", "err": round(100 * _ed(hyp, ref) / max(1, len(ref)), 1),
                  "ref": len(ref)}
    # ---- 話者: 行ごと(字数の重み)、1対1 ----
    P = sorted({r["spk"] for r in rows})
    G = sorted(meta["people"])
    W: dict = {}
    for r in rows:
        W[(r["spk"], r["gold_spk"])] = W.get((r["spk"], r["gold_spk"]), 0) + r["chars"]
    mp = _best_map(W, P, G)
    tot = sum(r["chars"] for r in rows)
    spk_acc = round(100 * sum(W.get((p, g), 0) for p, g in mp.items()) / max(1, tot), 1)
    # 相づちを除いた行の当たり
    spk_rows_ok = sum(1 for r in rows if mp.get(r["spk"]) == r["gold_spk"])
    # ---- 名前 ----
    gold_names = {p: v["name"] for p, v in meta["people"].items()}
    # 名前の書き方の違い(日本語のメモで英語の名前をカタカナで書く: Emily → エミリー)も当たり
    spell = {p: [v["name"], *v.get("aka", [])] for p, v in meta["people"].items()}

    def raw_same(nm, p):
        return any(x in nm or nm in x for x in spell[p])

    def same(nm, p):
        # 異体字(濱/浜 など)は音から決まらないので同じとみる。字の違いは別に数える(variant)
        a = _itai(nm)
        return any(_itai(x) in a or a in _itai(x) for x in spell[p])
    parts = notes.get("participants", [])
    name_rows = []
    for q in parts:
        g = mp.get(q.get("id"))
        nm = q.get("name") or ""
        ok = bool(g) and bool(nm) and same(nm, g)
        name_rows.append({"id": q.get("id"), "name": nm, "gold": gold_names.get(g, "(対応なし)"),
                          "ok": ok, "variant": ok and not raw_same(nm, g),
                          "evidence": q.get("evidence", "")})
    fab_names = [q["name"] for q in name_rows if q["name"] and not any(
        same(q["name"], p) for p in spell)]
    # 最後の発言が記録に残ったか(「会議を終える」で切れていないか): 台本の最後の2行の文字が記録にあるか
    allrec = _norm("".join(r["orig"] for r in rows))
    tail = []
    for g in [g for g in gold if not g["bc"]][-2:]:
        gt = _norm(g["text"])
        sm_ = difflib.SequenceMatcher(None, allrec, gt, autojunk=False)
        got = sum(b_.size for b_ in sm_.get_matching_blocks() if b_.size >= 3)
        tail.append({"spk": g["spk"], "text": g["text"], "kept": round(got / max(1, len(gt)), 2)})
    w2 = _check_audio(log, m, rows, gold, rm)
    segrep = seg_split_report(ev.get("events") or [], gold)
    tm = _check_terms(meta, rows, gold, rm)
    w3 = _check_w3(m, rows, meta, out, rm)
    jd = _judge(meta, notes, mp) if judge and notes else {}
    sm = {
        "id": meta["id"], "title": meta["title"], "mode": meta["mode"], "pair": meta["pair"],
        "cond": meta["cond"], "duration": meta["duration"], "people": len(G),
        "git": rm.get("git"), "model": rm.get("model"), "notes_s": rm.get("notes_s"),
        "lines_gold": len(gold), "lines_rec": len(rows), "asr": asr,
        "spk_acc": spk_acc, "spk_rows_ok": f"{spk_rows_ok}/{len(rows)}", "spk_n": len(P),
        "spk_map": mp, "names": name_rows, "names_ok": sum(1 for q in name_rows if q["ok"]),
        "fab_names": fab_names, "names_variant": sum(1 for q in name_rows if q.get("variant")),
        "judge": jd, "tail": tail, "end": rm.get("end", False),
        "w2": w2, "w3": w3, "terms": tm, "segs": segrep,
        "notes_counts": {k: len(notes.get(k, [])) for k in ("decisions", "todos", "questions", "summary")},
    }
    (out / "summary.json").write_text(json.dumps(sm, ensure_ascii=False, indent=1), encoding="utf-8")
    md = fmt(sm, rows, notes, meta)
    (out / "report.md").write_text(md, encoding="utf-8")
    if save:
        save.mkdir(parents=True, exist_ok=True)
        (save / f"{meta['id']}.md").write_text(md, encoding="utf-8")
        (save / f"{meta['id']}.json").write_text(json.dumps(sm, ensure_ascii=False, indent=1),
                                                encoding="utf-8")
    print(md.split("\n## 記録")[0])
    return sm


def _gold_line(lo: float, hi: float, gold: list) -> "int | None":
    """区間(音声の秒)にいちばん重なる台本の行の番号(相づちは除く)。0.2秒未満なら None。"""
    best, bo = None, 0.2
    for g in gold:
        if g.get("bc"):
            continue
        o = min(hi, g["end"]) - max(lo, g["start"])
        if o > bo:
            best, bo = g["i"], o
    return best


def seg_split_report(events: list, gold: list, gap_s: float = 1.2) -> dict:
    """区間の記録から「同じ人の1文が、確定のせいで2行に割れた」所を数える(11/07 ja_booth の
    「印刷をします。」+「することにします。」)。

    隣り合う区間 k, k+1 が別の単位に入り(k が単位の最後、k+1 が次の単位の最初)、
    間が gap_s 秒以内で、どちらも台本の同じ行にいちばん重なる → 「割れ」。
    割れごとに k の切った理由(silence/dip/max/flush)と、k の単位の文が句点で終わっていたかを数える。"""
    segs = [e for e in events if e.get("kind") == "seg"]
    units = [e for e in events if e.get("kind") == "unit"]
    if not segs or not units:
        return {}
    segs = sorted(segs, key=lambda e: e["lo"])

    def unit_of(sg):
        a, b = sg["lo"] + sg.get("pre", 0.0), sg["hi"]
        for k, u in enumerate(units):
            if u["lo"] - 0.05 <= a and b <= u["hi"] + 0.05:
                return k
        return None
    why_n: dict = {}
    for sg in segs:
        why_n[sg.get("why", "?")] = why_n.get(sg.get("why", "?"), 0) + 1
    splits = []
    for a, b in zip(segs, segs[1:]):
        ua, ub = unit_of(a), unit_of(b)
        if ua is None or ub is None or ua == ub:
            continue
        gap = (b["lo"] + b.get("pre", 0.0)) - a["hi"]
        if gap > gap_s:
            continue
        ga = _gold_line(a["lo"] + a.get("pre", 0.0), a["hi"], gold)
        gb = _gold_line(b["lo"] + b.get("pre", 0.0), b["hi"], gold)
        if ga is None or ga != gb:
            continue
        txt = units[ua]["text"]
        gtext = next(g["text"] for g in gold if g["i"] == ga)
        splits.append({"why": a.get("why", "?"), "gap": round(gap, 2), "gold_i": ga,
                       "period": txt.rstrip().endswith(("。", ".", "?", "？", "!", "！")),
                       "mid": _mid_sentence(txt, units[ub]["text"], gtext),
                       "text_a": txt[-20:], "text_b": units[ub]["text"][:20]})
    by: dict = {}
    for sp in splits:
        if sp["mid"] is False:
            continue        # 台本でもそこで文が終わっている(正しい区切り)
        k = f"{sp['why']}{'+句点' if sp['period'] else ''}{'' if sp['mid'] else '(不明)'}"
        by[k] = by.get(k, 0) + 1
    mids = [sp for sp in splits if sp["mid"] is not False]
    return {"segments": len(segs), "why": why_n, "units": len(units), "splits": len(mids),
            "splits_at_sentence_end": len(splits) - len(mids), "splits_by": by, "examples": mids[:8]}


_SENT_END = "。．.！!？?"


def _mid_sentence(text_a: str, text_b: str, gold_text: str) -> "bool | None":
    """割れ目が台本の文の途中か。text_b の頭(4字)を台本の中で探し、その直前が文末の記号なら
    False(文の切れ目)、そうでなければ True(文の途中)。見つからなければ text_a の終わり(4字)で探す。
    どちらも見つからなければ None(認識の誤りで判定できない)。"""
    g = gold_text or ""
    # 台本の文字(句読点・空白を除く)と、その元の位置
    keep = [(i, ch) for i, ch in enumerate(g) if _norm(ch)]
    gn = "".join(_norm(ch) for _, ch in keep)

    def prev_is_end(k: int) -> bool:   # 台本の正規化 k 文字目の直前(元の文字列で)に文末の記号があるか
        if k <= 0:
            return True
        a, b = keep[k - 1][0], keep[k][0]
        return any(c in _SENT_END for c in g[a + 1:b])
    hb = _norm(text_b)[:4]
    if len(hb) >= 3:
        k = gn.find(hb)
        if k >= 0:
            return not prev_is_end(k)
    ta = _norm(text_a)[-4:]
    if len(ta) >= 3:
        k = gn.find(ta)
        if k >= 0:
            return not prev_is_end(k + len(ta)) if k + len(ta) < len(gn) else False
    return None


def _check_terms(meta: dict, rows: list, gold: list, rm: dict) -> dict:
    """台本の固有名詞(meta terms)ごとに: 台本で言った回数、記録で正しく出た回数(その語を含む台本の行に
    対応する記録の行に出た)、幻覚(その語を含まない台本の行に対応する記録の行に出た)。"""
    terms = meta.get("terms") or []
    if not terms:
        return {}
    by_g: dict = {}
    for r in rows:
        by_g.setdefault(r["gold_i"], []).append(_norm(r["orig"]))
    out = {"used": rm.get("terms") or [], "per": {}}
    # 手がかりなしの聞き直し(11/07 守り2)が走った回数と足した秒(app.log の [terms] 行)
    lg = Path(rm.get("log") or "").parent.parent / "app.log" if rm.get("log") else None
    if lg is not None and lg.exists():
        sec = [float(x) for x in re.findall(r"^\[terms\].* ([\d.]+)s$", lg.read_text(encoding="utf-8"), re.M)]
        out["retry"] = {"n": len(sec), "sec": round(sum(sec), 2)}
        out["echo_dropped"] = len(re.findall(r"^\[halluc\] 用語の手がかりの復唱を破棄", lg.read_text(encoding="utf-8"), re.M))
    tot = [0, 0, 0]
    for t in terms:
        tn = _norm(t)
        said = [g for g in gold if tn in _norm(g["text"])]
        hit = sum(1 for g in said if any(tn in x for x in by_g.get(g["i"], [])))
        fake = sum(1 for gi, xs in by_g.items() if tn not in _norm(gold[gi]["text"])
                   for x in xs if tn in x)
        out["per"][t] = {"said": len(said), "heard": hit, "fake": fake}
        tot[0] += len(said)
        tot[1] += hit
        tot[2] += fake
    out["said"], out["heard"], out["fake"] = tot
    return out


def _check_audio(log: Path, m: dict, rows: list, gold: list, rm: dict) -> dict:
    """録音(W2): m4a があるか・長さ、各行の a0/a1 があるか、a0/a1 が正解の発言の時刻と合うか。
    録音の秒 = 壁時計 - ts0。流し始め(t0)の音声の秒 g は、録音の g + (t0 - ts0) 秒。"""
    meta = m.get("meta", {})
    out: dict = {"audio_meta": meta.get("audio")}
    m4a = log.with_name(log.stem + ".audio.m4a")
    caf = log.with_name(log.stem + ".audio.caf")
    out["m4a"] = m4a.exists()
    out["caf_left"] = caf.exists()
    if m4a.exists():
        try:
            info = subprocess.run(["afinfo", str(m4a)], capture_output=True, text=True, timeout=20).stdout
            mm = re.search(r"estimated duration:\s*([\d.]+)", info)
            out["m4a_sec"] = round(float(mm.group(1)), 1) if mm else None
            out["m4a_kb"] = m4a.stat().st_size // 1024
        except Exception:  # noqa: BLE001
            pass
    ts0 = rm.get("ts0") or meta.get("ts0")
    t0 = rm.get("t0")
    out["lines_with_a0"] = f"{sum(1 for r in rows if r.get('a0') is not None)}/{len(rows)}"
    if ts0 is None or t0 is None:
        return out
    off = t0 - ts0
    out["offset"] = round(off, 2)
    g = {x["i"]: x for x in gold}
    ok, n = 0, 0
    first: dict = {}      # 台本の行ごとの、記録の最初の切れ端の a0(1つの発言が何行かに割れることがある)
    for r in rows:
        if r.get("a0") is None:
            continue
        n += 1
        gl = g[r["gold_i"]]
        s, e = gl["start"] + off, gl["end"] + off
        if min(r["a1"], e) - max(r["a0"], s) > 0:      # 録音の区間が正解の発言と重なる
            ok += 1
        first[r["gold_i"]] = min(first.get(r["gold_i"], 1e9), r["a0"])
    if n:
        d0 = sorted(a0 - (g[i]["start"] + off) for i, a0 in first.items())
        out["overlap_gold"] = f"{ok}/{n}"
        # 発言の始まり(台本の行の最初の切れ端の a0)− 正解の話し始め。区間の手前の補い(0.5秒)の分は負になる
        out["start_err_med"] = round(d0[len(d0) // 2], 2)
        out["start_err_range"] = [round(d0[0], 2), round(d0[-1], 2)]
    return out


def _check_w3(m: dict, rows: list, meta: dict, out: Path, rm: dict) -> dict:
    """議題・印・自分のメモ(W3): 台本の events が記録に残り、指した行が話していた台本の行か、Markdown に出るか。"""
    evs = meta.get("events") or []
    if not evs and not meta.get("agenda"):
        return {}
    res: dict = {"agenda": [{"text": a.get("text"), "done": a.get("done"), "t": a.get("t")}
                            for a in m.get("agenda") or []]}
    exp_done = {e["k"] for e in evs if e["do"] == "agenda" and e.get("done", True)}
    res["agenda_ok"] = (len(res["agenda"]) == len(re.findall(r"^\s*\d", meta.get("agenda", ""), re.M))
                        and all(bool(a["done"]) == (k in exp_done) for k, a in enumerate(res["agenda"])))
    got_marks, got_memos = m.get("marks") or [], m.get("memos") or []
    checks = []
    for e in evs:
        if e["do"] == "mark":
            hit = next((x for x in got_marks if x.get("label") == e.get("label") and x.get("kind") == e.get("kind")), None)
        elif e["do"] == "memo":
            hit = next((x for x in got_memos if x.get("text") == e.get("text")), None)
        else:
            continue
        i = hit.get("i") if hit else None
        gi = rows[i]["gold_i"] if hit is not None and isinstance(i, int) and 0 <= i < len(rows) else None
        checks.append({"do": e["do"], "label": e.get("label") or e.get("text"), "found": hit is not None,
                       "at": e["at"], "line_gold": gi,
                       "near": gi is not None and e["at"] - 1 <= gi <= e["at"] + 1})
    res["items"] = checks
    md_p = out / "notes.md"
    md = md_p.read_text(encoding="utf-8") if md_p.exists() else ""
    res["md_has"] = {
        "agenda": all(a["text"] in md for a in res["agenda"]) if res["agenda"] else None,
        "marks": all((c["label"] or "") in md for c in checks if c["do"] == "mark"),
        "memos": all((c["label"] or "") in md for c in checks if c["do"] == "memo"),
    }
    return res


def fmt(sm: dict, rows: list, notes: dict, meta: dict) -> str:
    jd = sm.get("judge") or {}

    def cnt(k):
        v = jd.get(k)
        if not isinstance(v, list):
            return "?"
        if k == "todos":
            return (f"{sum(1 for x in v if isinstance(x, dict) and x.get('found'))}/{len(meta['todos'])}"
                    f"(担当 {sum(1 for x in v if isinstance(x, dict) and x.get('found') and x.get('owner_ok'))}・"
                    f"期限 {sum(1 for x in v if isinstance(x, dict) and x.get('found') and x.get('due_ok'))})")
        return f"{sum(1 for x in v if x is True)}/{len(meta[k])}"
    asr = " / ".join(f"{L} {a['metric']} {a['err']}%" for L, a in sm["asr"].items())
    out = [f"# {sm['id']}: {sm['title']}({sm['mode']} {sm['pair']}、{sm['cond']}、{sm['people']}人、"
           f"{sm['duration']:.0f}秒、app {sm['git']}、{sm['model']})", "",
           f"- 認識: {asr}",
           f"- 話者: 行の正解率(字数の重み) {sm['spk_acc']}% / 行 {sm['spk_rows_ok']} / "
           f"人数 {sm['spk_n']}(正解 {sm['people']})/ 対応 {sm['spk_map']}",
           f"- 名前: {sm['names_ok']}/{sm['people']} 人当たり(うち字の違い {sm.get('names_variant', 0)})"
           f" / 作り話の名前 {sm['fab_names'] or 'なし'}",
           f"- メモ: 決定事項 {cnt('decisions')} / やること {cnt('todos')} / 質問 {cnt('questions')} / "
           f"台本に無い決定 {len(jd.get('extra_decisions') or [])} / 台本に無いやること {len(jd.get('extra_todos') or [])}"
           f" / メモまで {sm['notes_s']}秒",
           f"- 最後の発言が記録に残ったか({'会議を終える' if sm.get('end') else 'live のメモ'}): "
           + " / ".join(f"{t['spk']} {t['kept']:.0%}「{t['text'][:20]}」" for t in sm.get("tail", [])),
           f"- 録音(W2): {json.dumps(sm.get('w2') or {}, ensure_ascii=False)}",
           *([f"- 区間(切った理由 {sm['segs'].get('why')}): 同じ人の1文の途中で2行に割れた {sm['segs'].get('splits')} "
              f"(台本でも文の切れ目だった所 {sm['segs'].get('splits_at_sentence_end')} は除く) "
              f"{sm['segs'].get('splits_by')} 例: "
              + " / ".join(f"[{x['why']}{'+。' if x['period'] else ''} 間{x['gap']}秒]「…{x['text_a']}」|「{x['text_b']}…」"
                           for x in sm['segs'].get('examples', [])[:4])] if sm.get("segs") else []),
           *([f"- 用語(W3 辞書 {sm['terms'].get('used') or 'なし'}): 台本で {sm['terms']['said']}回 → 記録に正しく "
              f"{sm['terms']['heard']} / 言っていないのに出た {sm['terms']['fake']} / "
              f"聞き直し {sm['terms'].get('retry', {}).get('n', '-')}回 {sm['terms'].get('retry', {}).get('sec', '-')}秒 / "
              f"復唱を捨てた {sm['terms'].get('echo_dropped', '-')} / "
              + " ".join(f"{k}:{v['heard']}/{v['said']}" + (f"(幻覚{v['fake']})" if v['fake'] else "")
                         for k, v in sm['terms']['per'].items())] if sm.get("terms") else []),
           *( [f"- 議題・印・メモ(W3): 議題 {'○' if sm['w3'].get('agenda_ok') else '×'} "
               f"{[(a['text'], a['done']) for a in sm['w3'].get('agenda', [])]} / "
               + " / ".join(f"{c['do']}「{c['label']}」{'あり' if c['found'] else 'なし'}"
                            f"(行→台本{c['line_gold']}、予定{c['at']}{'○' if c['near'] else '×'})"
                            for c in sm['w3'].get('items', []))
               + f" / Markdown {sm['w3'].get('md_has')}"] if sm.get("w3") else []),
           "", "## 名前", "| id | メモの名前 | 正解 | 当たり | 根拠 |", "|---|---|---|---|---|"]
    for q in sm["names"]:
        out.append(f"| {q['id']} | {q['name']} | {q['gold']} | {'○' if q['ok'] else '×'} | {q['evidence'][:40]} |")
    out += ["", "## メモ(抜粋)",
            "- 決定事項: " + " / ".join(notes.get("decisions", [])),
            "- やること: " + " / ".join(f"{t.get('task')}[{t.get('owner')}・{t.get('due') or '-'}]"
                                    for t in notes.get("todos", [])),
            "- 質問: " + " / ".join(notes.get("questions", [])),
            "- 台本に無い決定/やること(判定): " + " / ".join(map(str, (jd.get("extra_decisions") or [])
                                                          + (jd.get("extra_todos") or []))),
            "", "## 記録(付け直し後)", "| 秒 | 話者(正解) | 原文 |", "|---|---|---|"]
    for r in rows:
        mark = "" if sm["spk_map"].get(r["spk"]) == r["gold_spk"] else " ×"
        out.append(f"| {r['ts']} | {r['spk']}({r['gold_spk']}){mark} | {r['orig'][:60]} |")
    return "\n".join(out) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--out", type=Path, required=True)
    b.add_argument("--only", default="")
    b.add_argument("--babble", type=Path, default=None, help="会場のざわめきの wav(FLEURS clips)")
    r = sub.add_parser("run")
    r.add_argument("--meeting", type=Path, required=True)
    r.add_argument("--out", type=Path, required=True)
    r.add_argument("--no-wait", action="store_true")
    r.add_argument("--end", action="store_true", help="「会議を終える」(end_meeting)の道でメモを作る")
    r.add_argument("--terms", default="app", choices=["app", "none", "script"],
                   help="用語辞書(W3): app = 保存された辞書 / none = 無し / script = 台本の固有名詞")
    r.add_argument("--record", default=None, choices=["on", "off", "auto"],
                   help="録音(W2)の設定をこの実行の中だけ上書き(config.json は変えない)")
    e = sub.add_parser("eval")
    e.add_argument("--meeting", type=Path, required=True)
    e.add_argument("--out", type=Path, required=True)
    e.add_argument("--save", type=Path, default=None)
    e.add_argument("--no-judge", action="store_true")
    a = ap.parse_args()
    if a.cmd == "build":
        build(a.out, a.only, babble_dir=a.babble)
    elif a.cmd == "run":
        run(a.meeting, a.out, wait=not a.no_wait, end=a.end, record=a.record, terms=a.terms)
    elif a.cmd == "eval":
        evaluate(a.meeting, a.out, a.save, judge=not a.no_judge)


if __name__ == "__main__":
    main()
