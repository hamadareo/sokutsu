"""エコー消し(bin/sokutsu-aecio)が読み上げの回り込みをどれだけ減らすかを、実機で数秒だけ測る。

人が手で動かす試験(自動の試験からは呼ばない)。スピーカーから短い読み上げが3回鳴る。
測っている間は静かにしてください(話すと「回り込み」に数えられます)。
音はファイルに残さない(読み上げの元の音だけ一時ファイルに作り、終わったら消す)。

測るもの(どれもマイクの音の RMS、dBFS):
  A. 部品(VPIO、エコー消しあり)で鳴らしながら、部品のマイクで聞く  … これからの道
  B. 部品で鳴らすがエコー消しを素通し(--bypass)                    … 同じ道で消す分だけ比べる
  C. afplay で鳴らしながら、いつものマイク(sounddevice)で聞く        … いまの道
  それぞれ鳴らす前の静かな 0.5 秒(床の雑音)と、鳴っている間の値を出し、
  「C − A」「B − A」を減った dB として出す(大きいほどよく消えている)。
  VPIO は雑音も減らすので、床の雑音との差(鳴っている間 − 床)も並べる。

実行: PYTHONPATH=src .venv/bin/python scripts/aec_echo_test.py [--text "…"] [--voice Kyoko]
      [--wav 16kHz の WAV] [--repeat 2] [--skip-plain] [--skip-bypass]
  --repeat: 同じ音を続けて何回鳴らすか(エコー消しは鳴り始めに慣れる時間が要るので、最後の1回を使う)
"""
from __future__ import annotations

import argparse
import math
import os
import subprocess
import sys
import tempfile
import threading
import time
import wave
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402

from tsuyaku.aecio import HELPER, SR, EchoCancelIO  # noqa: E402

TEXT = "これはエコーを消す試験です。読み上げの声がマイクにどれだけ戻るかを測ります。"


def db(x: np.ndarray) -> float:
    if not len(x):
        return float("nan")
    r = float(np.sqrt(np.mean(np.square(x, dtype=np.float64))))
    return 20 * math.log10(max(r, 1e-10))


def read_wav16k(path: str) -> np.ndarray:
    with wave.open(path, "rb") as w:
        if w.getframerate() != SR or w.getsampwidth() != 2:
            raise SystemExit(f"{path}: 16 kHz・16 ビットの WAV にしてください")
        x = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(np.float32) / 32768
        if w.getnchannels() > 1:
            x = x.reshape(-1, w.getnchannels()).mean(axis=1)
    return x


def make_clip(text: str, voice: str, tmp: str) -> str:
    path = os.path.join(tmp, "clip.wav")
    cmd = ["say", "-o", path, "--file-format=WAVE", "--data-format=LEI16@16000"]
    if voice:
        cmd += ["-v", voice]
    subprocess.run(cmd + [text], check=True)
    return path


class Recorder:
    """届いたマイクの音をメモリにためる(ファイルには書かない)。"""

    def __init__(self):
        self.parts: list = []
        self.n = 0
        self.lock = threading.Lock()

    def __call__(self, x):
        with self.lock:
            self.parts.append(np.asarray(x, dtype=np.float32).copy())
            self.n += len(x)

    def mark(self) -> int:
        with self.lock:
            return self.n

    def span(self, a: int, b: int) -> np.ndarray:
        with self.lock:
            x = np.concatenate(self.parts) if self.parts else np.zeros(0, np.float32)
        return x[a:b]


def measure_helper(clip: np.ndarray, repeat: int, bypass: bool) -> "dict | None":
    rec = Recorder()
    cmd = [str(HELPER)] + (["--bypass"] if bypass else [])
    io = EchoCancelIO(rec, cmd=cmd)
    if not io.start():
        print(f"  部品が使えません: {io.error}")
        return None
    try:
        time.sleep(1.0)                      # 動き始めの揺れを避ける
        a = rec.mark()
        time.sleep(0.5)
        floor = rec.span(a, rec.mark())
        span = (0, 0)
        for _ in range(repeat):
            st = [rec.mark()]
            ok = io.play(clip, on_start=lambda: st.__setitem__(0, rec.mark()))
            span = (st[0], rec.mark())
            if not ok:
                print("  再生に失敗しました")
                return None
        time.sleep(0.2)
        echo = rec.span(*span)
        return {"floor": db(floor), "echo": db(echo), "secs": len(echo) / SR, "info": io.info}
    finally:
        io.stop()


def measure_plain(path: str, repeat: int) -> "dict | None":
    import sounddevice as sd
    rec = Recorder()
    with sd.InputStream(samplerate=SR, channels=1, dtype="float32", blocksize=512,
                        callback=lambda indata, f, t, s: rec(indata[:, 0])):
        time.sleep(1.0)
        a = rec.mark()
        time.sleep(0.5)
        floor = rec.span(a, rec.mark())
        span = (0, 0)
        for _ in range(repeat):
            s0 = rec.mark()
            subprocess.run(["afplay", path], check=True)
            span = (s0, rec.mark())
        echo = rec.span(*span)
    return {"floor": db(floor), "echo": db(echo), "secs": len(echo) / SR}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--text", default=TEXT)
    ap.add_argument("--voice", default="")
    ap.add_argument("--wav", default="")
    ap.add_argument("--repeat", type=int, default=2)
    ap.add_argument("--skip-plain", action="store_true")
    ap.add_argument("--skip-bypass", action="store_true")
    a = ap.parse_args()
    if not HELPER.exists():
        raise SystemExit(f"{HELPER} がありません。scripts/build_aecio.sh で作ってください")
    with tempfile.TemporaryDirectory(prefix="aec_echo_") as tmp:
        path = a.wav or make_clip(a.text, a.voice, tmp)
        clip = read_wav16k(path)
        print(f"読み上げ {len(clip) / SR:.1f} 秒 × {a.repeat} 回(最後の1回で測ります)。静かにしてください。")
        res = {}
        print("A. 部品(エコー消しあり)…")
        res["A"] = measure_helper(clip, a.repeat, bypass=False)
        if not a.skip_bypass:
            print("B. 部品(エコー消しを素通し)…")
            res["B"] = measure_helper(clip, a.repeat, bypass=True)
        if not a.skip_plain:
            print("C. afplay + いつものマイク…")
            res["C"] = measure_plain(path, a.repeat)
    names = {"A": "部品・消す", "B": "部品・素通し", "C": "afplay+マイク"}
    print()
    print(f"{'':14}{'床の雑音':>10}{'鳴っている間':>14}{'差':>8}")
    for k, r in res.items():
        if r:
            print(f"{names[k]:14}{r['floor']:>9.1f}dB{r['echo']:>12.1f}dB{r['echo'] - r['floor']:>7.1f}dB"
                  f"  ({r['secs']:.1f}秒)")
    if res.get("A"):
        info = res["A"]["info"]
        print(f"\n機器: in={info.get('in')} out={info.get('out')} "
              f"in_hw={info.get('in_hw')} out_hw={info.get('out_hw')}")
        for k in ("C", "B"):
            if res.get(k):
                print(f"{names[k]} → 部品・消す: 回り込みが {res[k]['echo'] - res['A']['echo']:.1f} dB 減った")
        print("目安: 20 dB 以上減り、「鳴っている間 − 床」が数 dB 以内なら、読み上げ中もマイクを開けておける")


if __name__ == "__main__":
    main()
