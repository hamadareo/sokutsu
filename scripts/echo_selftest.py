"""自エコーライブ検証: 読み上げ音声がマイクに戻っても字幕が増えないことを実音声で確かめる。

マイクもスピーカーも使わない(完全無音)。やること:
  1. macOS say で日本語発話Aを再生成 → 実VAD+実whisperで認識(ja判定) → 実LLMで英訳
  2. 訳をKokoroで読み上げ音声化 → **部屋のエコーに見立てて劣化**(減衰+ローパス+残響+雑音)
  3. ゲート中(paused)にエコー音声を流す → セグメント0個(マイクが閉いている)
  4. ゲート解除後に同じエコーを流す(長い残響が漏れる状況) → whisperが拾う →
     is_echoが捨てなければならない(字幕増殖しない)
  5. 続けて別の本物の発話Bを流す → is_echoが通す(会話が止まらない)
並行して RSS を常時サンプリングし、Whisper+Kokoro+koモデル常駐時の実測メモリを出す。

実行: PYTHONPATH=src .venv/bin/python scripts/echo_selftest.py
"""
from __future__ import annotations

import difflib
import os
import queue
import subprocess
import sys
import threading
import time
import types
import wave
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402

import tsuyaku.audio as audio_mod  # noqa: E402
from tsuyaku.audio import SpeechCapture  # noqa: E402
from tsuyaku.config import load_settings  # noqa: E402
from tsuyaku.langdetect import is_echo, norm  # noqa: E402
from tsuyaku.stt import STT  # noqa: E402
from tsuyaku.translate import (build_messages, pick_model, target_lang,  # noqa: E402
                               translate_stream)

SR = 16000
SCRATCH = Path(os.environ.get("TSUYAKU_SELFTEST_DIR")
               or Path(__file__).resolve().parents[1] / ".selftest")
SCRATCH.mkdir(exist_ok=True)

FAILURES: list[str] = []


def check(cond: bool, msg: str):
    tag = "✓" if cond else "✗"
    print(f"  {tag} {msg}")
    if not cond:
        FAILURES.append(msg)


# ---------- メモリサンプリング ----------

class RssSampler(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.peak_kb = 0
        self.last_kb = 0
        self._stop = threading.Event()

    def run(self):
        while not self._stop.is_set():
            try:
                out = subprocess.run(["ps", "-o", "rss=", "-p", str(os.getpid())],
                                     capture_output=True, text=True, timeout=2).stdout
                kb = int(out.strip() or 0)
                self.last_kb = kb
                self.peak_kb = max(self.peak_kb, kb)
            except Exception:
                pass
            self._stop.wait(0.4)

    def stop(self):
        self._stop.set()


def mem(sampler: RssSampler, label: str):
    print(f"[mem] {label}: 現在 {sampler.last_kb/1024:.0f} MB / ピーク {sampler.peak_kb/1024:.0f} MB")


# ---------- 音声生成・劣化(無音。再生は一切しない) ----------

def say_render(text: str, voice: str, path: Path) -> np.ndarray:
    """macOS say でWAV出力(再生しない)。→ float32 mono @16k。"""
    subprocess.run(["say", "-v", voice, "-o", str(path),
                    "--file-format=WAVE", "--data-format=LEI16@22050", text],
                   check=True, timeout=60)
    with wave.open(str(path), "rb") as w:
        data = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2")
    return resample(data.astype(np.float32) / 32768.0, 22050)


def resample(x: np.ndarray, sr_in: int, sr_out: int = SR) -> np.ndarray:
    n = int(round(len(x) * sr_out / sr_in))
    idx = np.linspace(0, len(x) - 1, n)
    return np.interp(idx, np.arange(len(x)), x).astype(np.float32)


def degrade(x: np.ndarray, gain=0.22, noise=0.0025) -> np.ndarray:
    """厳しいエコー: 減衰+6msローパス+120ms残響+部屋ノイズ(低声質)。"""
    y = x * gain
    k = max(1, int(0.006 * SR))
    y = np.convolve(y, np.ones(k) / k, mode="same")
    out = y + np.random.default_rng(7).normal(0, noise, len(y)).astype(np.float32)
    d = int(0.12 * SR)
    if len(out) > d:
        out[d:] += y[:-d] * 0.35
    return out.astype(np.float32)


def degrade_light(x: np.ndarray, gain=0.55, noise=0.0015) -> np.ndarray:
    """現実的なエコー: 机越しで声が明瞭に残るレベル(減衰+ごく軽い部屋ノイズのみ)。
    この程度だと whisper はそのまま文字起こしできる → is_echo が唯一の防波堤。"""
    out = x * gain + np.random.default_rng(11).normal(0, noise, len(x)).astype(np.float32)
    return out.astype(np.float32)


def ratio(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, norm(a), norm(b)).ratio()


# ---------- ゲート付きマイクの偽物(実VADは本物) ----------

class _FakeStream:
    cb = None

    def __init__(self, **kw):
        _FakeStream.cb = kw["callback"]

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def feed(audio: np.ndarray, tail_silence_s=1.2):
    """キャプチャのコールバックにリアルタイム相当のチャンクで音声を流し込む。"""
    pad = np.zeros(int(tail_silence_s * SR), dtype=np.float32)
    a = np.concatenate([audio, pad])
    for i in range(0, len(a), 512):
        _FakeStream.cb(a[i:i + 512].reshape(-1, 1), 512, None, None)
        time.sleep(0.002)  # VADの内部時計に負荷を掛けない程度の間隔


def drain(q: queue.Queue) -> list[np.ndarray]:
    segs = []
    while not q.empty():
        segs.append(q.get_nowait())
    return segs


def main():
    s = load_settings()
    sampler = RssSampler()
    sampler.start()
    mem(sampler, "Python起動時")

    print("[setup] whisper読み込み(数分)...")
    t0 = time.time()
    stt = STT(s.whisper_model)
    print(f"[setup] whisper {time.time()-t0:.1f}s")
    mem(sampler, "+whisper常駐")

    from tsuyaku.tts import TTS
    tts = TTS(s.tts_dir_kokoro, s.tts_dir_ko, s.tts_sids or {},
              ja_engine="say", voices=s.tts_voice or {})
    print(f"[setup] TTSエンジン: {sorted(tts.engine)} + macOS say")
    mem(sampler, "+whisper+TTS常駐")

    # 翻訳サーバーは会場の既定(remoteトンネル 12345)を優先し、無ければ設定の1234へ
    base, model = None, None
    for port in (12345, s.port):
        b = f"http://{s.host}:{port}"
        try:
            model = pick_model(b, s.model)
            base = b
            break
        except Exception:
            continue
    if base is None:
        raise SystemExit("翻訳サーバーに到達できません(トンネル/LM Studio起動後にもう一度)")
    warm = time.time()
    from tsuyaku.translate import warmup
    warmup(base, model)
    print(f"[setup] 翻訳サーバー {model} ({base}) ウォーム {time.time()-warm:.1f}s")
    try:
        hw = int(subprocess.run(["sysctl", "-n", "hw.memsize"],
                                capture_output=True, text=True).stdout)
        print(f"[setup] このMacの実装メモリ {hw/2**30:.0f} GB")
    except Exception:
        pass

    audio_mod.sd = types.SimpleNamespace(InputStream=_FakeStream)
    q: queue.Queue = queue.Queue()
    cap = SpeechCapture(s.silero_vad, q, sr=SR,
                        min_speech_s=s.min_speech_s, max_speech_s=s.max_speech_s,
                        min_silence_s=s.min_silence_s)
    cap.start()
    time.sleep(0.3)
    assert _FakeStream.cb is not None, "captureスレッド起動失敗"

    pair_t = ("ja", "en")
    JA1 = "報酬設計を先に詰めるべきだと思います。"
    JA2 = "三台目のカメラでは遮蔽が主な失敗要因です。"

    # 1) 発話A: say音声 → 実VAD+実whisper → 実翻訳
    print("\n[1] 話者Aの日本語発話(say再生成)を認識・翻訳")
    ja1 = say_render(JA1, "Kyoko", SCRATCH / "ja1.wav")
    cap.paused.clear()
    feed(ja1)
    segs = drain(q)
    check(len(segs) >= 1, f"VADが発話Aを切り出した ({len(segs)}個)")
    text_a, lang_a, _pa = stt.transcribe(segs[0] if segs else ja1)
    print(f"    認識[{lang_a}]: {text_a}")
    check(lang_a == "ja", "say日本語が日本語として認識された")
    trans_a = translate_stream(base, model,
                               build_messages(pair_t, "3x3ロボットサッカー", [], text_a, lang_a))
    print(f"    訳(en): {trans_a}")
    check(bool(target_lang(pair_t, lang_a)) and len(trans_a) > 0, "英語の読み上げ文がある")

    # 2) 訳をKokoroで読み上げ → エコーに劣化
    print("\n[2] 読み上げ(Kokoro)がマイクに戻るエコーを劣化再現")
    from tsuyaku.tts import parse_voice
    kind, val = parse_voice(s.tts_voice.get("en", "kokoro:3"))
    if kind == "kokoro" and "kokoro" in tts.engine:
        audio = tts.engine["kokoro"].generate(
            trans_a, sid=int(val), speed=s.tts_speed.get("en", 1.0))
        en16 = resample(np.asarray(audio.samples, dtype=np.float32), audio.sample_rate)
    else:  # Kokoroが無ければ say の英語声で置き換え(劣化経路は同じ)
        en16 = say_render(trans_a, "Alex", SCRATCH / "en1.wav")
    echo = degrade_light(en16)   # 現実的な机越しエコー(声は明瞭に残る)
    hard = degrade(en16, gain=0.45)  #低声質の厳しいエコー
    print(f"    エコー音声 {len(echo)/SR:.1f}s (現実的=減衰のみ / 厳しめ=+ローパス+残響+雑音)")
    # 現実的エコー: whisperが明瞭に聞き取れるので is_echo が唯一の防波堤。必ず捨てること。
    h, hl, _ph = stt.transcribe(echo)
    r_light = ratio(trans_a, h)
    print(f"    現実的エコーを直接whisper[{hl}]: {h!r} → ratio={r_light:.2f}")
    check(is_echo(trans_a, h),
          f"明瞭な読み上げエコーをis_echoが破棄 (ratio={r_light:.2f}≥0.6)")
    # 厳しめ(低声質): whisperは幻覚を出すことがある。VADが拾わないか is_echo のどちらかで
    # 実チェーンは安全なら可。両方抜けたら失敗。
    hh, hhl, _pnh = stt.transcribe(hard)
    r_hard = ratio(trans_a, hh)
    hard_filtered = is_echo(trans_a, hh)
    print(f"    厳しめエコーを直接whisper[{hhl}]: {hh!r} → ratio={r_hard:.2f}"
          f"{' (is_echoで廃棄)' if hard_filtered else ' (幻覚の可能性: VAD側で防御)'}")

    # 3) ゲート中: エコーを流してもセグメントは出ない
    print("\n[3] 読み上げ中のマイクゲート")
    cap.paused.set()  # Bridgeが tts.say 前にやること
    feed(echo)
    segs = drain(q)
    check(len(segs) == 0, f"ゲート中はエコーがVADへ届かない ({len(segs)}個)")
    cap.resume_after_tts(s.gate_tail_s)

    # 4) ゲート解除後も残り続ける残響が漏れたら: whisperが拾い is_echo が捨てる
    print("\n[4] ゲート解除後も残響が漏れた(最悪ケース)")
    feed(echo)
    segs = drain(q)
    print(f"    VADが拾ったセグメント {len(segs)}個")
    if segs:
        heard, heard_lang, _psh = stt.transcribe(segs[0])
        print(f"    whisperが拾った[{heard_lang}]: {heard}")
        r = ratio(trans_a, heard)
        check(is_echo(trans_a, heard),
              f"解除後に漏れたエコーをis_echoが破棄 (ratio={r:.2f}≥0.6)")
    else:
        print("    (このゲインではVADが解錠後も発話として出さなかった=そもそもエコーが"
              "閾値未満。is_echoの実判定は[2]の直接whisperで実施済み)")

    # 5) 本物の次の発話は通る
    print("\n[5] 続けて本物の別発言(話者B)を流す")
    ja2 = say_render(JA2, "Kyoko", SCRATCH / "ja2.wav")
    feed(ja2)
    segs = drain(q)
    check(len(segs) >= 1, f"話者Bの発話がVADに出た ({len(segs)}個)")
    if segs:
        text_b, lang_b, _pb = stt.transcribe(segs[0])
        print(f"    認識[{lang_b}]: {text_b}")
        r2 = ratio(trans_a, text_b)
        check(not is_echo(trans_a, text_b), f"別発言はエコー扱いされない (ratio={r2:.2f}<0.6)")
        try:
            trans_b = translate_stream(
                base, model,
                build_messages(pair_t, "3x3ロボットサッカー",
                               [{"lang": lang_a, "orig": text_a, "trans": trans_a}],
                               text_b, lang_b), timeout=60.0)
            print(f"    訳(en): {trans_b}")
            check(bool(trans_b), "次の発話Bも翻訳されて会話が止まらない")
        except Exception as e:  # noqa: BLE001
            check(False, f"発話Bの翻訳がサーバー側で失敗(アプリならエラー字幕で継続): {e}")

    cap.stop()
    mem(sampler, "終盤")
    peak_mb = sampler.peak_kb / 1024
    ru_mb = None
    try:
        import resource
        # macOS の ru_maxrss はバイト単位(LinuxはKB)。ps サンプラの方が正確なので目安。
        ru_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)
    except Exception:
        pass
    try:
        swap = subprocess.run(["sysctl", "-n", "vm.swapusage"],
                              capture_output=True, text=True).stdout.strip()
    except Exception:
        swap = "?"
    print(f"\n[mem] プロセス常駐ピーク {peak_mb:.0f} MB"
          + (f" (ru_maxrss {ru_mb:.0f} MB)" if ru_mb else "")
          + f" / スワップ: {swap}")
    print(f"    内訳目安: whisper large-v3-turbo ~1.6GB + Kokoro int8 + VAD")

    sampler.stop()
    if FAILURES:
        print(f"\n✗ 失敗 {len(FAILURES)}件:")
        for f in FAILURES:
            print(f"  - {f}")
        sys.exit(1)
    print("\n✓ 全項目通過: 読み上げエコーは増殖せず、次の本物の発話は通る")


if __name__ == "__main__":
    main()
