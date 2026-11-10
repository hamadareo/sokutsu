"""試験用の偽の bin/sokutsu-sysaudio(本物と同じ約束で標準出力に float32 を流す)。

  --mode tone      READY を出して 440 Hz を --seconds 秒ぶん流して終わる(終了コード 0)
  --mode deny      ERROR PERMISSION_DENIED を出して終了コード 3
  --mode crash     少し流してから終了コード 5(毎回)
  --mode hold      READY を出して標準入力が閉じるまで無音を流し続ける
  --mode tonehold  tone のあと、標準入力が閉じるまで待つ(本物と同じく自分では終わらない)
  --mode early     READY より先に音を出す(順番が前後しても読めるか)
"""
import argparse
import math
import os
import struct
import sys
import time

ap = argparse.ArgumentParser()
ap.add_argument("--mode", default="tone")
ap.add_argument("--rate", type=int, default=48000)
ap.add_argument("--seconds", type=float, default=1.0)
ap.add_argument("--exclude-tree", action="append")
a = ap.parse_args()


def err(s):
    sys.stderr.write(s + "\n")
    sys.stderr.flush()


def tone(n, t0=0):
    return struct.pack(f"<{n}f", *(0.5 * math.sin(2 * math.pi * 440 * (t0 + i) / a.rate)
                                   for i in range(n)))


out = sys.stdout.buffer
if a.mode == "deny":
    err("ERROR PERMISSION_DENIED システムオーディオ録音が許可されていません")
    sys.exit(3)
if a.mode == "early":
    out.write(tone(480))
    out.flush()
    time.sleep(0.05)
err(f"READY rate={a.rate} excluded={len(a.exclude_tree or [])}")
if a.mode in ("tone", "early", "tonehold"):
    total = int(a.seconds * a.rate)
    t = 480 if a.mode == "early" else 0
    while t < total:
        n = min(1000, total - t)
        out.write(tone(n, t))
        t += n
    out.flush()
    if a.mode == "tonehold":
        sys.stdin.buffer.read()
    sys.exit(0)
if a.mode == "crash":
    out.write(tone(4800))
    out.flush()
    err("ERROR 集約デバイスを作れません (OSStatus -1)")
    sys.exit(5)
if a.mode == "hold":
    os.set_blocking(0, False)
    while True:
        try:
            if sys.stdin.buffer.read(1) == b"":
                sys.exit(0)
        except (BlockingIOError, TypeError):
            pass
        out.write(b"\0" * 4 * 480)
        out.flush()
        time.sleep(0.01)
