#!/usr/bin/env python3
"""通訳アプリの「耳」を リモートサーバーの GPU で動かす小さな HTTP サービス (faster-whisper)。

なぜ: このMac(M3)の whisper large-v3-turbo は 4秒の発話に約2.3秒かかり、話し終わり
→字幕の最大の待ちになっていた(11/05実測)。リモートサーバーの GPU なら同じモデルで0.2秒台。

  POST /transcribe  本文 = 16kHz 16bit mono PCM (raw, little endian)
                    → {"text": "...", "lang": "ja", "prob": 0.87, "sec": 0.21}
  GET  /health      → {"ok": true, "model": ...}

prob は各セグメントの exp(avg_logprob)(=トークン確率の幾何平均)の最小値。
Mac 側 pywhispercpp の probability と同じ定義なので、幻覚守卫の床がそのまま使える。

127.0.0.1 でだけ待ち受ける(Mac からは SSH 転送で届く)。idle_exit 秒リクエストが
無ければ自分で終わって GPU を空ける。起動は Mac の scripts/remote_tunnel.sh が冪等に行う。

  ~/tsuyaku_stt/venv/bin/python ~/tsuyaku_stt/remote_stt_server.py --port 9941
"""
import argparse
import json
import math
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import numpy as np


# 版(11/06 先送り): scripts/remote_tunnel.sh が /health の ver と比べ、古ければ置き換えて起動し直す
VER = 3


class StreamStore:
    """話している間に先に届いた声(/append)を、会話(sid)ごとに通算サンプル番号で持つ。

    11/06 英会話の実時間評価: ふつうの回線では、話し終わってから区間の音声(4秒=128KB)を
    送ると往復が中央2.0秒・p90 3.4秒(音声なしの往復は中央0.2秒)。続き待ちの繋ぎ直しは
    前の断片ごと送り直すので15秒=480KBにもなり、4秒の時間切れでこのMacの耳に落ちていた。
    声は話している間に少しずつ送っておき、聞くときは「どこからどこまで」だけを送る。
    """

    def __init__(self, keep_s: float = 180.0, sr: int = 16000, max_sids: int = 8):
        self.keep = int(keep_s * sr)
        self.max_sids = max_sids
        self.sids: dict = {}            # sid -> {"chunks": {off: int16 配列}, "end": 最大, "t": 最終}
        self.lock = threading.Lock()

    def append(self, sid: str, off: int, x: np.ndarray) -> None:
        if not sid or off < 0 or x.size == 0:
            return
        with self.lock:
            d = self.sids.get(sid)
            if d is None:
                if len(self.sids) >= self.max_sids:   # 古い会話から捨てる
                    old = min(self.sids, key=lambda k: self.sids[k]["t"])
                    del self.sids[old]
                d = self.sids[sid] = {"chunks": {}, "end": 0, "t": 0.0}
            x = x.astype("<i2", copy=False)
            prev = d["chunks"].get(int(off))
            if prev is not None and prev.size > x.size:
                # 同じ所から始まる長い塊が先に届いていた(聞き取りの本文で送った分): 短い方で
                # 上書きすると後ろが消える(11/06 実時間評価で 409 が続いた原因)
                prev = prev.copy()
                prev[:x.size] = x
                x = prev
            d["chunks"][int(off)] = x
            d["end"] = max(d["end"], int(off) + x.size)
            d["t"] = time.time()
            lim = d["end"] - self.keep
            for k in [k for k, v in d["chunks"].items() if k + v.size < lim]:
                del d["chunks"][k]

    def get(self, sid: str, a: int, b: int):
        """[a, b) の int16。1サンプルでも欠けていれば None。"""
        if b <= a:
            return np.zeros(0, dtype="<i2")
        with self.lock:
            d = self.sids.get(sid)
            if d is None:
                return None
            out = np.zeros(b - a, dtype="<i2")
            have = np.zeros(b - a, dtype=bool)
            for k, v in d["chunks"].items():
                lo, hi = max(a, k), min(b, k + v.size)
                if lo < hi:
                    out[lo - a:hi - a] = v[lo - k:hi - k]
                    have[lo - a:hi - a] = True
        return out if have.all() else None

    def build(self, sid: str, recipe: str):
        """recipe = "a-b,zN,a-b"(通算サンプル番号の範囲と、N サンプルの無音)→ int16 か None。"""
        parts = []
        for tok in (recipe or "").split(","):
            tok = tok.strip()
            if not tok:
                continue
            if tok[0] == "z":
                parts.append(np.zeros(max(0, int(tok[1:])), dtype="<i2"))
                continue
            a, b = (int(v) for v in tok.split("-"))
            x = self.get(sid, a, b)
            if x is None:
                return None
            parts.append(x)
        return np.concatenate(parts) if parts else None


def parse_ranges(spec: str) -> list:
    """"a-b,c-d" → [(a, b), (c, d)]。"""
    out = []
    for tok in (spec or "").split(","):
        if "-" in tok:
            a, b = tok.split("-", 1)
            out.append((int(a), int(b)))
    return out


# 言語ごとの書き方の手がかり(11/06): whisper は中国語を繁体字で書きがち
# (実会話「千萬年」「一切有貨的啊」)。簡体字の文を前に置くと簡体字で書く。
LANG_PROMPT = {"zh": "以下是普通话的句子。"}


def choose_lang(probs, langs, dur: float, short_s: float = 1.5, keep_p: float = 0.7):
    """言語ペア(langs)に絞って言語を選ぶ。langs 空なら None(whisper 任せ)。

    短い区間(< short_s 秒)は必ずペアの中から。長い区間でペア外が keep_p 以上で
    勝つときだけ、本当に第三言語を話しているとみてそのまま返す。
    """
    if not langs:
        return None
    probs = sorted(probs, key=lambda kv: -kv[1])
    if probs and probs[0][0] not in langs and dur >= short_s and probs[0][1] >= keep_p:
        return probs[0][0]
    inpair = [kv for kv in probs if kv[0] in langs]
    return inpair[0][0] if inpair else langs[0]


def cuda_lib_paths():
    """pip の nvidia-cublas / nvidia-cudnn の .so の場所(ctranslate2 用に LD_LIBRARY_PATH へ)。"""
    try:
        import nvidia.cublas.lib
        import nvidia.cudnn.lib
        return [os.path.dirname(nvidia.cublas.lib.__file__),
                os.path.dirname(nvidia.cudnn.lib.__file__)]
    except Exception:  # noqa: BLE001
        return []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=9941)
    ap.add_argument("--model", default="large-v3-turbo")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--compute-type", default="float16")
    ap.add_argument("--idle-exit", type=float, default=1800.0)
    args = ap.parse_args()
    libs = cuda_lib_paths()
    if libs and not os.environ.get("_STT_REEXEC"):
        os.environ["LD_LIBRARY_PATH"] = ":".join(libs + [os.environ.get("LD_LIBRARY_PATH", "")])
        os.environ["_STT_REEXEC"] = "1"
        os.execv(sys.executable, [sys.executable] + sys.argv)
    from faster_whisper import WhisperModel  # noqa: PLC0415
    t = time.time()
    model = WhisperModel(args.model, device=args.device, compute_type=args.compute_type)
    print(f"[tsuyaku-stt] {args.model} ({args.device}/{args.compute_type}) "
          f"読み込み {time.time() - t:.1f} 秒", flush=True)
    lock = threading.Lock()
    last = [time.time()]

    def transcribe(pcm: bytes, langs: list, prompt: str = "", ctx_s: float = 0.0,
                   lang_prompt: bool = True, beam: int = 1) -> dict:
        """ctx_s>0: 先頭 ctx_s 秒は「文脈として聞かせるだけ」の音声。言語判定は本体だけで行い、
        書き起こしは全体で行って、単語の時刻が ctx_s より前の語を捨てる(短い区間でも
        前後の音の手がかりで聞き取れるようにする)。"""
        x = np.frombuffer(pcm[:len(pcm) // 2 * 2], dtype="<i2").astype(np.float32) / 32768.0
        if x.size < 1600:  # 0.1秒未満
            return {"text": "", "lang": "", "prob": 1.0}
        with lock:
            lang = None
            c = int(max(0.0, ctx_s) * 16000)
            body = x[c:] if 0 < c < x.size else x
            if langs:  # 今日の言語ペアに絞る(短い区間の言語の取り違え対策)
                _l, _p, allp = model.detect_language(body)
                lang = choose_lang(allp, langs, body.size / 16000.0)
            if lang_prompt and lang in LANG_PROMPT:
                prompt = LANG_PROMPT[lang] + prompt
            segs, info = model.transcribe(
                x, language=lang, beam_size=max(1, min(5, beam)), vad_filter=False,
                initial_prompt=(prompt or None), word_timestamps=c > 0,
                condition_on_previous_text=False, without_timestamps=True)
            segs = list(segs)
        if c > 0:   # 文脈部分(先頭 ctx_s 秒)に入る語を捨てる
            keep = []
            for sg in segs:
                for w in (sg.words or []):
                    if (w.start + w.end) / 2 >= ctx_s:
                        keep.append(w.word)
            text = "".join(keep).strip()
        else:
            text = " ".join(s.text.strip() for s in segs if s.text.strip()).strip()
        ps = [math.exp(s.avg_logprob) for s in segs if s.avg_logprob == s.avg_logprob]
        return {"text": text, "lang": info.language or "",
                "prob": round(min(ps), 4) if ps else 1.0}

    store = StreamStore()

    class H(BaseHTTPRequestHandler):
        # 繋ぎっぱなし(keep-alive): 1回ごとに SSH の通り道を開け直す往復を省く
        protocol_version = "HTTP/1.1"

        def log_message(self, *a):
            pass

        def _send(self, code, obj):
            b = json.dumps(obj, ensure_ascii=False).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(b)))
            self.end_headers()
            self.wfile.write(b)

        def do_GET(self):
            last[0] = time.time()
            self._send(200, {"ok": True, "model": args.model, "ver": VER, "stream": True})

        def do_POST(self):
            last[0] = time.time()
            n = int(self.headers.get("Content-Length") or 0)
            pcm = self.rfile.read(n)
            u = urlparse(self.path)
            q = parse_qs(u.query)
            sid = (q.get("sid", [""])[0] or "")[:64]
            if u.path == "/append":   # 話している間の先送り: 通算 off サンプル目から
                try:
                    store.append(sid, int(q.get("off", ["-1"])[0]),
                                 np.frombuffer(pcm[:len(pcm) // 2 * 2], dtype="<i2"))
                except ValueError:
                    return self._send(400, {"error": "off"})
                return self._send(200, {"ok": True})
            recipe = q.get("r", [""])[0] or ""
            if sid and recipe:
                # 本文は先送りがまだ届いていない分だけ(u= の範囲を順に並べたもの)
                try:
                    pos = 0
                    raw = np.frombuffer(pcm[:len(pcm) // 2 * 2], dtype="<i2")
                    for a, b in parse_ranges(q.get("u", [""])[0] or ""):
                        store.append(sid, a, raw[pos:pos + (b - a)])
                        pos += b - a
                    x = store.build(sid, recipe)
                except ValueError:
                    x = None
                if x is None:
                    return self._send(409, {"missing": True})
                pcm = x.tobytes()
            langs = [x for x in (q.get("langs", [""])[0] or "").split(",") if x]
            prompt = (q.get("prompt", [""])[0] or "")[-300:]  # 前の文・テーマ(用語の手がかり)
            try:
                ctx_s = float(q.get("ctx", ["0"])[0] or 0)
            except ValueError:
                ctx_s = 0.0
            t0 = time.time()
            try:
                out = transcribe(pcm, langs, prompt, ctx_s,
                                 lang_prompt=(q.get("lp", ["1"])[0] != "0"),
                                 beam=int(q.get("beam", ["1"])[0] or 1))
            except Exception as e:  # noqa: BLE001
                return self._send(500, {"error": str(e)[:200]})
            out["sec"] = round(time.time() - t0, 3)
            if sid and recipe:
                out["span"] = True   # 範囲で聞いた(古いサーバーは付けない → 手元で音声ごと送り直す)
            last[0] = time.time()
            self._send(200, out)

    srv = ThreadingHTTPServer(("127.0.0.1", args.port), H)

    def idle_watch():
        while True:
            time.sleep(30)
            if time.time() - last[0] > args.idle_exit:
                print(f"[tsuyaku-stt] {args.idle_exit:.0f}秒アイドル → 終了して GPU を空ける",
                      flush=True)
                os._exit(0)

    threading.Thread(target=idle_watch, daemon=True).start()
    print(f"[tsuyaku-stt] http://127.0.0.1:{args.port}/transcribe", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
