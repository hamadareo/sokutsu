"""試験用の偽の bin/sokutsu-aecio(本物と同じ約束。音声機器は使わない)。

  --mode ok       READY を出し、マイクの音として 0,1,2,… の float32 を --mic 個、半端なバイト数に
                  分けて流す。標準入力の約束を読み、受けた命令を --log へ1行ずつ書く
                  (CLIP n first last / FLUSH)。1件は --play-s 秒「鳴らす」(止められたら cancelled=1)
  --mode deny     ERROR PERMISSION_DENIED を出して終了コード 3
  --mode unsup    ERROR を出して終了コード 4
  --mode silent   READY を出さずに標準入力が閉じるまで待つ(時間切れの試験)
  --mode dieplay  READY のあと、1件目を受けたら終了コード 5
  --mode rate     READY rate=48000(周波数が違う)
"""
import argparse
import os
import struct
import sys
import threading
import time

ap = argparse.ArgumentParser()
ap.add_argument("--mode", default="ok")
ap.add_argument("--mic", type=int, default=1600)
ap.add_argument("--play-s", type=float, default=0.1)
ap.add_argument("--log", default="")
a = ap.parse_args()


def err(s):
    sys.stderr.write(s + "\n")
    sys.stderr.flush()


def log(s):
    if a.log:
        with open(a.log, "a") as f:
            f.write(s + "\n")


if a.mode == "deny":
    err("ERROR PERMISSION_DENIED マイクが許可されていません")
    sys.exit(3)
if a.mode == "unsup":
    err("ERROR この Mac には VoiceProcessingIO がありません")
    sys.exit(4)
if a.mode == "silent":
    sys.stdin.buffer.read()
    sys.exit(0)

err(f"READY rate={48000 if a.mode == 'rate' else 16000} in=\"fake\" out=\"fake\" fake=1")

# マイクの音: 0,1,2,… を 1001 バイトずつ(float の途中で切れる)
out = sys.stdout.buffer
data = struct.pack(f"<{a.mic}f", *range(a.mic))
for i in range(0, len(data), 1001):
    out.write(data[i:i + 1001])
    out.flush()

cancel = threading.Event()
lock = threading.Lock()
queue = []
wake = threading.Event()


def player():
    while True:
        wake.wait()
        with lock:
            if not queue:
                wake.clear()
                continue
            cid, n = queue.pop(0)
        err(f"PLAY_START id={cid} n={n}")
        t0 = time.time()
        stopped = False
        while time.time() - t0 < a.play_s:
            if cancel.is_set():
                stopped = True
                break
            time.sleep(0.005)
        err(f"PLAY_DONE id={cid} n={n} cancelled={int(stopped)}")


threading.Thread(target=player, daemon=True).start()
inp = sys.stdin.buffer
cid = 0


def read_full(n):
    b = b""
    while len(b) < n:
        c = inp.read(n - len(b))
        if not c:
            return None
        b += c
    return b


while True:
    h = read_full(4)
    if h is None:
        sys.exit(0)
    (n,) = struct.unpack("<I", h)
    if n == 0:
        with lock:
            pending = len(queue)
            for q in queue:
                err(f"PLAY_DONE id={q[0]} n={q[1]} cancelled=1")
            queue.clear()
        cancel.set()
        log("FLUSH")
        err(f"FLUSH pending={pending}")
        time.sleep(0.02)
        cancel.clear()
        continue
    body = read_full(4 * n)
    if body is None:
        sys.exit(0)
    x = struct.unpack(f"<{n}f", body)
    cid += 1
    log(f"CLIP {n} {x[0]:.4f} {x[-1]:.4f}")
    if a.mode == "dieplay":
        err("ERROR 試験: 落ちます")
        os._exit(5)
    with lock:
        queue.append((cid, n))
    wake.set()
