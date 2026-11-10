"""セッションログ。同じstemで .jsonl(機械向け) と .txt(人間が読む用) を1本ずつ残す。"""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path


class SessionLog:
    def __init__(self, log_dir: str, pair: str, topic: str = "", mode: str = "interpret",
                 extra: dict | None = None, audio: bool = False):
        """audio=True: 録音する(W2)。meta に録音のファイル名と ts0(録音の先頭の壁時計)を書く。
        録音の秒 = 壁時計 - ts0(recorder.Recorder が最初のチャンクまでを無音で埋める)。"""
        base = Path(log_dir)
        base.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime('%Y%m%d-%H%M%S')
        stem = base / f"session-{stamp}"
        # 同じ秒に2つ目(取り込みの直後の会話など): 前の記録に追記・録音の上書きをせず _2, _3 … にする。
        # "-" でなく "_" なのは名前の並び(= 新しい順)を保つため: "X_2.jsonl" > "X.jsonl" > "X-2.jsonl"。
        # id の形は session-[0-9A-Za-z_-]+ のままなので読み手(一覧・再生・ゴミ箱)はそのまま通る。
        k = 1
        while any(base.glob(f"{stem.name}.*")):
            k += 1
            stem = base / f"session-{stamp}_{k}"
        self.path_jsonl = stem.with_suffix(".jsonl")
        self.path_txt = stem.with_suffix(".txt")
        fresh = not self.path_jsonl.exists()
        # 会議中の道具(印・メモ・修正)は画面のスレッドから書くので、発言の行と混ざらないよう鍵を掛ける
        self._lock = threading.Lock()
        self.ts0 = round(time.time(), 3) if audio else None
        self.audio_caf = str(stem.with_name(f"{stem.name}.audio.caf")) if audio else None
        self._j = self.path_jsonl.open("a", encoding="utf-8")
        self._t = self.path_txt.open("a", encoding="utf-8")
        if fresh:
            meta = {"type": "meta", "pair": pair, "topic": topic, "mode": mode,
                    "start": time.strftime("%Y-%m-%d %H:%M:%S")}
            if audio:   # 録音(W2)
                meta["audio"] = f"{stem.name}.audio.m4a"
                meta["ts0"] = self.ts0
            meta.update(extra or {})   # 担当ごとに meta へ足す項目(Bridge._meta_extra: 議題・テンプレート・用語・聞く音)
            self._j.write(json.dumps(meta, ensure_ascii=False) + "\n")
            self._t.write(f"# 通訳 {meta['start']}  ペア {pair}"
                          + (f"  テーマ: {topic}" if topic else "") + "\n\n")
            self.flush()

    def add(self, lang: str, orig: str, trans: str, spoken: str | None,
            spk: str | None = None, segs: list | None = None,
            audio: "tuple | None" = None, ts: "float | None" = None, t: "str | None" = None,
            timing: dict | None = None) -> None:
        """audio=(a0, a1): この発言の録音の中の秒(W2: 行を押すとそこから鳴らす)。
        ts/t: 時刻を指定する(ファイルの取り込みは、その発言が録音のどこかで決める)。"""
        tstr = t or time.strftime("%H:%M:%S")
        row = {"type": "utter", "t": tstr, "lang": lang, "orig": orig,
               "trans": trans, "spoken": spoken or "",
               "ts": round(time.time() if ts is None else ts, 2)}
        if audio and audio[0] is not None and audio[1] is not None:
            row["a0"], row["a1"] = round(float(audio[0]), 2), round(float(audio[1]), 2)
        for k in ("t0", "dur"):   # 話し始めの壁時計と声の長さ(話者の分析 W4 11/07)
            if timing and timing.get(k) is not None:
                row[k] = round(float(timing[k]), 2)
        if spk:
            row["spk"] = spk   # 声で見分けた話者(会議メモ 11/06)
        if segs:
            row["segs"] = list(segs)   # 話者の見分けの区間の番号(付け直しで使う)
        with self._lock:
            self._write_j(json.dumps(row, ensure_ascii=False) + "\n")
            self.n_rows = getattr(self, "n_rows", 0) + 1   # 会議を終える時の「書き終えたか」の目安
            if self._t.closed:
                return
            arrow = f"{lang}→{spoken}" if spoken else f"{lang}(字幕のみ)"
            who = f" {spk}" if spk else ""
            if trans:
                self._t.write(f"[{tstr}]{who} {arrow}\n    原文: {orig}\n    翻訳: {trans}\n\n")
            else:   # 会議メモだけ(訳さない)
                self._t.write(f"[{tstr}]{who} {lang}\n    {orig}\n\n")
        self.flush()

    def add_alias(self, aliases: dict) -> None:
        """声の見分けで「S3 は S1 と同じ人」と分かった(会議の途中で合わせた)。読む側が過去の行を直す。"""
        self.add_row({"type": "alias", "map": dict(aliases)})

    def add_refine(self, ids: list) -> None:
        """会議の後の付け直し: 区間の番号順の話者 id。読む側が各行の segs の多数決で話者を直す。"""
        self.add_row({"type": "refine", "ids": list(ids)})

    def add_row(self, row: dict) -> None:
        """機械向けの行を1つ(印・メモ・議題・修正など。.txt には書かない)。どのスレッドからでも。"""
        line = json.dumps(row, ensure_ascii=False) + "\n"
        with self._lock:
            self._write_j(line)

    def _write_j(self, line: str) -> None:
        """jsonl に1行(鍵の中で)。会議を終える瞬間(close の後、Bridge.log を None にする前)に画面から
        来た印・メモは、閉じたファイルの口でなくファイルへ直接足す(例外にせず、行も失わない)。"""
        if self._j.closed:
            with self.path_jsonl.open("a", encoding="utf-8") as f:
                f.write(line)
            return
        self._j.write(line)
        self._j.flush()

    def flush(self) -> None:
        with self._lock:
            if not self._j.closed:
                self._j.flush()
            if not self._t.closed:
                self._t.flush()

    def close(self) -> None:
        with self._lock:
            try:
                self._j.close()
            finally:
                self._t.close()
