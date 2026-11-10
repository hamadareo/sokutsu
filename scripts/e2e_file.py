"""E2E: マイクの代わりにWAVをVADに通して、翻訳パイプライン全体を測る。

実行: PYTHONPATH=src .venv/bin/python scripts/e2e_file.py <wav> [--pair ja-en] [--topic テーマ]
"""
from __future__ import annotations

import sys
import time
import wave
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402

from tsuyaku.audio import AutoGain, StreamSegmenter  # noqa: E402
from tsuyaku.config import engine_candidates, load_settings  # noqa: E402
from tsuyaku.segmenter import Endpointer  # noqa: E402
from tsuyaku.log import SessionLog  # noqa: E402
from tsuyaku.stt import STT, RemoteSTT  # noqa: E402
from tsuyaku.translate import (build_messages, out_lang, pick_model,  # noqa: E402
                                target_lang, translate_stream, warmup)


def read_wav(path: str) -> np.ndarray:
    with wave.open(path, "rb") as w:
        assert w.getframerate() == 16000, f"16kHzが必要: {w.getframerate()}"
        assert w.getnchannels() == 1
        data = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2")
    return data.astype(np.float32) / 32768.0


def main():
    wav = sys.argv[1]
    pair = "ja-en"
    topic = ""
    engine = ""  # remote / mac(アプリと同じ組み合わせ: remote なら耳も リモートサーバー)。空=従来どおり
    args = sys.argv[2:]
    for i, a in enumerate(args):
        if a == "--pair":
            pair = args[i + 1]
        elif a == "--topic":
            topic = args[i + 1]
        elif a == "--engine":
            engine = args[i + 1]
    pair_t = tuple(pair.split("-"))

    s = load_settings()
    import sherpa_onnx
    conf = sherpa_onnx.VadModelConfig()
    conf.silero_vad.model = s.silero_vad
    conf.silero_vad.threshold = 0.5
    conf.silero_vad.min_silence_duration = s.min_silence_s
    conf.silero_vad.min_speech_duration = s.min_speech_s
    conf.sample_rate = 16000

    audio = read_wav(wav)
    t_load0 = time.time()
    stt = STT(s.whisper_model)
    if engine == "remote":
        stt = RemoteSTT(s.stt_url, stt)
    print(f"[e2e] STT読み込み {time.time()-t_load0:.1f}s / 入力 {len(audio)/16000:.1f}s")

    # アプリ(SpeechCapture)と同じ区切り: VAD は声の有無、区間は StreamSegmenter
    # (リモートサーバーの耳なら音量の落ち込みでも細かく切る)。文の区切りは後段の Endpointer。
    fine = engine == "remote"
    if engine:
        stt.langs = pair_t
    vad_sil = 0.2 if fine else s.min_silence_s
    conf.silero_vad.min_silence_duration = vad_sil
    conf.silero_vad.max_speech_duration = 30.0
    vad = sherpa_onnx.VoiceActivityDetector(conf, buffer_size_in_seconds=60)
    sg = StreamSegmenter(16000, min_speech_s=s.min_speech_s, min_silence_s=vad_sil,
                         cut_s=0.18 if fine else 1e9, max_s=s.max_speech_s)
    chunk = 512
    segs = []  # (検出した音声時刻, 音声, 開始秒, 終了秒, 補い秒)
    agc = AutoGain()  # アプリと同じ自動音量調整
    for i in range(0, len(audio), chunk):
        ch = agc(audio[i:i + chunk])
        vad.accept_waveform(ch)
        while not vad.empty():
            vad.pop()
        for seg, lo, hi, pre in sg.feed(ch, vad.is_speech_detected()):
            segs.append(((i + chunk) / 16000.0, seg, lo, hi, pre))
    for seg, lo, hi, pre in sg.feed(np.zeros(16000, np.float32), False):
        segs.append((hi + 0.3, seg, lo, hi, pre))
    print(f"[e2e] 区間 {len(segs)}個")

    if engine:
        _lab, base, pref = engine_candidates(engine, s.host, None, None)[0]
        model = pick_model(base, pref)
    else:
        base = f"http://{s.host}:{s.port}"
        model = pick_model(base, s.model)
    t_w0 = time.time()
    warmup(base, model)
    print(f"[e2e] LLMウォームアップ {time.time()-t_w0:.1f}s ({model})")
    log = SessionLog(s.log_dir, pair, topic + " [e2e]")
    history: list[dict] = []

    ok = 0
    n_units = 0
    ep = Endpointer(rejoin=(lambda a: stt.transcribe(a)) if fine else None,
                    hold_open_s=s.hold_open_s)

    def emit(u, t_stt: float) -> None:
        nonlocal ok, n_units
        n_units += 1
        text, lang = u.text, u.lang
        target = target_lang(pair_t, lang)
        out = out_lang(pair_t, lang)
        msgs = build_messages(pair_t, topic, history, text, lang)
        t1 = time.time()
        try:
            trans = translate_stream(base, model, msgs)
        except Exception as e:  # noqa: BLE001
            print(f"[unit{n_units}] 翻訳エラー: {e}")
            return
        t_tr = time.time() - t1
        latency = t_stt + t_tr
        icon = "✓" if latency <= 5.0 else "✗"
        if latency <= 5.0:
            ok += 1
        merge = f" ({u.parts}区間を1文に)" if u.parts > 1 else ""
        where = getattr(stt, "last_where", "") or "mac"
        print(f"[unit{n_units}] {icon} 話終わり→字幕 {latency:.2f}s "
              f"(区切り+STT[{where}] {t_stt:.2f} + 翻訳 {t_tr:.2f}){merge}")
        print(f"       {lang}→{out}{' (読み上げ)' if target else ' (字幕のみ)'} | {text}")
        print(f"       ⇝ {trans}")
        history.append({"lang": lang, "orig": text, "trans": trans})
        log.add(lang, text, trans, target)

    # 音声の時計で再生: 区間が検出された時刻 + STT の実時間 = 区切りを決められる時刻。
    # 待ち期限の判定は push が「話し始めまでの間」で行う(アプリは別途0.1秒ごとに
    # tick するが、ここでは次の区間が来た時点でまとめて決まる)。
    for t_det, seg, lo, hi, pre in segs:
        t0 = time.time()
        text, lang, _prob = stt.transcribe(seg)
        dt = time.time() - t0
        if not text or not lang:
            continue
        t1 = time.time()
        units = ep.push(seg, lo, hi, text, lang, now=t_det + dt, pre=pre)
        dt2 = time.time() - t1
        for u in units:  # t_ready = 区切りを決めた時刻(前の断片は自分の待ち期限で出る)
            emit(u, max(0.0, u.t_ready - u.hi + (dt2 if u.parts > 1 else 0)))
    for u in ep.flush():
        emit(u, max(0.0, u.t_ready - u.hi))
    log.close()
    print(f"[e2e] 5秒以内: {ok}/{n_units}")


if __name__ == "__main__":
    main()
