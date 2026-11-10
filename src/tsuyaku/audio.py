"""マイク入力 + silero-VAD による発話区間検出。読み上げ中は入力を捨てる(半二重)。"""
from __future__ import annotations

import queue
import threading
import time

import numpy as np
import sounddevice as sd

from . import devices


def rms(x) -> float:
    """セグメントの実効レベル(0..1 相当)。雑音ゲート(noise_rms_floor)との比較に使う。"""
    a = np.asarray(x, dtype=np.float64)
    if a.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(a))))


class PreRoll:
    """VAD 区間の手前 pre_s 秒を足して返す(話し始めの欠け対策)。

    11/05夕 E2E: 「サンプル数が少ないので」が「プル数が…」になった。息の音
    (サ・ハ・s/f)で始まる語は VAD の区間開始が遅れて頭が欠ける。手前0.2秒で
    「アンプル」、0.4秒で「サンプル」に戻る(remote/このMac の耳とも同じ結果)。
    feed() した通算サンプル番号は VAD の front.start と同じ数え方(reset で0へ)。
    """

    def __init__(self, sr: int = 16000, pre_s: float = 0.4, keep_s: float = 15.0):
        self.pre = int(sr * pre_s)
        self.keep = int(sr * keep_s)
        self.reset()

    def reset(self) -> None:
        self._buf = np.zeros(0, dtype=np.float32)
        self._buf_start = 0   # _buf[0] の通算サンプル番号
        self._fed = 0

    def feed(self, chunk) -> None:
        c = np.asarray(chunk, dtype=np.float32)
        self._buf = np.concatenate([self._buf, c])
        self._fed += len(c)
        over = len(self._buf) - self.keep
        if over > 0:
            self._buf = self._buf[over:]
            self._buf_start += over

    def extend(self, start: int, seg) -> np.ndarray:
        seg = np.asarray(seg, dtype=np.float32)
        lo = max(start - self.pre, self._buf_start, 0)
        if lo >= start:
            return seg
        return np.concatenate([self._buf[lo - self._buf_start:start - self._buf_start], seg])


class AutoGain:
    """入力の自動音量調整(VAD と区切りの前段)。

    11/05夜 AMI 会議の卓上マイク1本: 声の RMS が0.005前後と小さく、silero VAD が
    発話の3割強に反応しなかった(単語が区間に入った率 64〜68%)。入力を4〜8倍に
    すると 79〜85% に上がる(VAD は音量に敏感)。直近 win_s 秒の大きめの音(上位5%)が
    target になるよう持ち上げる。ただし部屋の雑音が noise_cap を超えない範囲で、
    倍率は 1〜max_gain(普通の近いマイクでは 1 のまま=何もしない)。
    """

    def __init__(self, sr: int = 16000, target: float = 0.05, noise_cap: float = 0.01,
                 max_gain: float = 8.0, win_s: float = 10.0, chunk: int = 512):
        self.target = target
        self.noise_cap = noise_cap
        self.max_gain = max_gain
        self._n = max(8, int(win_s * sr / chunk))
        self._hist: list[float] = []
        self.noise = 0.01
        self.gain = 1.0

    def __call__(self, chunk) -> np.ndarray:
        c = np.asarray(chunk, dtype=np.float32)
        r = float(np.sqrt(np.mean(np.square(c, dtype=np.float64)))) if len(c) else 0.0
        self._hist.append(r)
        if len(self._hist) > self._n:
            del self._hist[0]
        if r < self.noise:
            self.noise = 0.9 * self.noise + 0.1 * r
        else:
            self.noise = 0.998 * self.noise + 0.002 * r
        if len(self._hist) >= 8:
            loud = float(np.percentile(self._hist, 95))
            g = min(self.target / max(loud, 1e-6), self.noise_cap / max(self.noise, 1e-6))
            g = max(1.0, min(self.max_gain, g))
            self.gain = 0.9 * self.gain + 0.1 * g   # 急に変えない
        if self.gain <= 1.01:
            return c
        return np.clip(c * self.gain, -1.0, 1.0)


class StreamSegmenter:
    """マイク入力を「話の区間」に切る(VAD の声あり判定 + 音量の落ち込み)。

    11/05夜: silero VAD は声が止まってもしばらく「声あり」が残り、無音判定を
    0.1秒まで縮めても 0.25〜0.35秒の話者交代をほとんど切れなかった(合成会話で
    18境目中9つ)。2人分が1区間になると whisper が片方の言語で聞き、英語の返事が
    丸ごと消えた。VAD は「今声が出ているか」だけに使い、区間の端は自前で決める:
      始まり: VAD が声ありになった(min_speech の手前 + pre_s も含める)
      終わり: VAD が声なしになった / 声ありのまま音量が部屋の雑音近くまで
              cut_s 続いた(話者交代・文の間)/ max_s に達した
    切りすぎは後段の Endpointer が文末を見て繋ぎ直すので害が小さい。
    feed() はチャンクごとに (音声, 開始秒, 終了秒, 手前の補い秒) のリストを返す。
    秒は feed した通算(読み上げ中に止めても巻き戻らない)。
    """

    def __init__(self, sr: int = 16000, pre_s: float = 0.5, min_speech_s: float = 0.25,
                 min_silence_s: float = 0.2, cut_s: float = 0.18, ratio: float = 0.1,
                 max_s: float = 12.0, min_seg_s: float = 0.3, keep_s: float = 20.0,
                 loud_ratio: float = 2.0, min_loud_s: float = 0.15, quiet_mul: float = 1.5):
        self.sr = sr
        self.pre = int(sr * pre_s)
        self.min_speech = int(sr * min_speech_s)
        self.min_silence = int(sr * min_silence_s)
        self.cut = int(sr * cut_s)
        self.ratio = ratio
        self.max_len = int(sr * max_s)
        self.min_seg = int(sr * min_seg_s)
        self.keep = int(sr * keep_s)
        self.post = int(sr * 0.08)       # 語尾の余白(息の抜けを切らない)
        self.loud_ratio = loud_ratio
        self.quiet_mul = quiet_mul
        self.min_loud = int(sr * min_loud_s)
        self._buf = np.zeros(0, dtype=np.float32)
        self._buf_start = 0
        self._fed = 0
        self.noise = 0.01      # 部屋の雑音(ゆっくり上がり、すぐ下がる)
        self._peak = 0.0
        self._start: int | None = None   # 今の区間の開始(通算サンプル)
        self._quiet = 0                  # 区間中の静かなサンプル数(連続)
        self._last_loud = 0              # 最後に声の大きさがあった位置
        self._cut_wait = False           # 音量で切った後、VADはまだ声あり: 次の声を待つ
        self._prev_hi = 0                # 直前の区間の終わり(手前の補いはここより前に戻さない)
        self._cut_req: "int | None" = None   # 話し続けている区間をここ(通算サンプル)で区切る(11/08)
        self._cut_his: list = []         # 区切って閉じた区間の (終わりの秒, 前半の字)。Endpointer に「待たない」と伝える
        self._cut_head = None            # 頼まれた区切りの前半の字 (文字, 言語)

    def current_audio(self) -> "np.ndarray | None":
        """今話している区間の、ここまでの音声(手前の補い込み)。区間外なら None。"""
        if self._start is None:
            return None
        lo = max(self._start - self.pre, self._prev_hi, self._buf_start, 0)
        return self._slice(lo, self._fed)

    def flush_current(self) -> list:
        """読み上げでマイクを止める直前: 話し途中の区間をそこまでで閉じて渡す
        (11/06 実会話: 読み上げが始まった瞬間に話し途中だった声を捨てていた)。
        止める前の音なので読み上げの声は混ざっていない。"""
        out: list = []
        if self._start is not None:
            end = self._last_loud if self._last_loud > self._start else self._fed
            self._close(min(self._fed, end + self.post), out)
        self._cut_wait = False
        return out

    def drop_current(self) -> None:
        """読み上げでマイクを止めた: 話し途中の区間は捨てる(通算の時計は進めない)。"""
        self._start = None
        self._quiet = 0
        self._peak = 0.0
        self._cut_wait = False

    def request_cut(self, at_s: float, text: str = "", lang: str = "") -> bool:
        """話し続けている区間を at_s 秒(通算)で区切る(途中字幕の句読点・息つぎ。segmenter.pick_cut)。
        前半は次の feed で閉じた区間として返し、後半はそのまま今の区間として続ける。別のスレッド
        (STT ワーカー)から呼ぶので、ここでは頼みを置くだけ。今の区間の中でなければ False。"""
        pos = int(round(at_s * self.sr))
        st = self._start
        if st is None or pos - st < self.min_seg or pos > self._fed:
            return False
        self._cut_req = pos
        self._cut_head = (text, lang) if text else None
        return True

    def cut_text(self, hi_s: float) -> "tuple[str, str] | None":
        """区切った前半の途中字幕の字 (文字, 言語)。後ろの文脈まで聞いた字なので、前半だけ聞き直すより正確。"""
        for h, t in self._cut_his:
            if abs(hi_s - h) < 1e-3:
                return t
        return None

    def starts_at_cut(self, lo_s: float) -> bool:
        """その頭(秒)の区間は区切った後半(話の続き)か。後半の頭 = 前半の終わり。"""
        return self.was_cut(lo_s)

    def was_cut(self, hi_s: float) -> bool:
        """その終わり(秒)の区間は request_cut で区切った前半か。"""
        return any(abs(hi_s - h) < 1e-3 for h, _t in self._cut_his)

    def _do_cut(self, out: list) -> None:
        pos, self._cut_req = self._cut_req, None
        head, self._cut_head = getattr(self, "_cut_head", None), None
        st = self._start
        if pos is None or st is None or pos - st < self.min_seg or pos > self._fed:
            return
        n0 = len(out)
        self._close(pos, out)
        if len(out) > n0:
            self._cut_his = (self._cut_his + [(pos / self.sr, head)])[-16:]
        self._start = pos                # 後半は続けて同じ話の区間(手前の補いは前半と重ねない)
        self._prev_hi = max(self._prev_hi, pos)
        self._last_loud = max(self._last_loud, pos)

    def speaking_since(self) -> "float | None":
        """今話している区間の話し始め(通算秒)。区間外なら None。"""
        return None if self._start is None else self._start / self.sr

    def _slice(self, lo: int, hi: int) -> np.ndarray:
        lo = max(lo, self._buf_start)
        return self._buf[lo - self._buf_start:max(lo, hi) - self._buf_start].copy()

    def _close(self, hi: int, out: list) -> None:
        lo = self._start
        self._start = None
        self._quiet = 0
        self._peak = 0.0
        if lo is None or hi - lo < self.min_seg:
            return
        # 実際に声の大きさがあった時間が0.15秒未満 = 物音・咳: 区間にしない
        body = self._slice(lo, hi)
        f = len(body) // 512
        if f:
            fr = np.sqrt(np.mean(np.square(body[:f * 512].reshape(f, 512), dtype=np.float64),
                                 axis=1))
            # 判定は部屋の雑音との比だけ(絶対値の下限は置かない): AMI 会議の卓上マイク
            # 1本では声の RMS が0.005前後しかなく、下限0.004で169区間中56を捨てていた
            if np.sum(fr > self.noise * self.loud_ratio) * 512 < self.min_loud:
                return
        # 手前の補いは直前の区間の終わりまで(前の人の語尾を一緒に聞かせない)
        a_lo = max(lo - self.pre, self._prev_hi, self._buf_start, 0)
        self._prev_hi = hi
        out.append((self._slice(a_lo, hi), a_lo / self.sr, hi / self.sr, (lo - a_lo) / self.sr))

    def feed(self, chunk, speech: bool) -> list:
        c = np.asarray(chunk, dtype=np.float32)
        n = len(c)
        self._buf = np.concatenate([self._buf, c])
        pos0 = self._fed
        self._fed += n
        over = len(self._buf) - self.keep
        if over > 0:
            self._buf = self._buf[over:]
            self._buf_start += over
        r = float(np.sqrt(np.mean(np.square(c, dtype=np.float64)))) if n else 0.0
        if r < self.noise:
            self.noise = 0.9 * self.noise + 0.1 * r
        else:
            self.noise = 0.998 * self.noise + 0.002 * r
        out: list = []
        if self._cut_req is not None:
            self._do_cut(out)
        if not speech:
            self._cut_wait = False
            if self._start is not None:
                # VAD は min_silence 分遅れて「声なし」になる: 最後に声のあった所で閉じる
                end = (self._last_loud if self._last_loud > self._start
                       else self._fed - self.min_silence)
                self._close(min(self._fed, end + self.post), out)
            return out
        quiet = r < max(self.noise * self.quiet_mul, self._peak * self.ratio)
        if self._start is None:
            if self._cut_wait:
                if quiet:
                    return out
                self._cut_wait = False
                self._start = pos0
            else:
                self._start = max(0, pos0 + n - self.min_speech)
            self._last_loud = self._fed
        if not quiet:
            self._peak = max(r, self._peak * 0.995)
            self._quiet = 0
            self._last_loud = self._fed
        else:
            self._quiet += n
            if self._quiet >= self.cut and self._last_loud - self._start >= self.min_seg:
                self._close(min(self._fed, self._last_loud + self.post), out)
                self._cut_wait = True
                return out
        if self._fed - self._start >= self.max_len:
            self._close(self._fed, out)
        return out


def meter_of(r: float) -> float:
    """RMS → メーターの値 0..1(-60 dBFS = 0、-10 dBFS = 1。声の大きさの感じ方に合わせて dB で)。"""
    if r <= 0:
        return 0.0
    db = 20.0 * float(np.log10(max(r, 1e-9)))
    return float(min(1.0, max(0.0, (db + 60.0) / 50.0)))


class _MicStream:
    """マイクのストリームを開いて保つ道具(会話の SpeechCapture と、起動画面の「マイクの確認」MicTest)。

    11/07 監査: 開けない・1チャンクの例外・抜けた、を無言にしない。
    11/08 hear: デバイスを名前で選ぶ(devices.py)。switch_device() はストリームだけを開き直す
    (区切り・VAD・行列・会話はそのまま)。使っているデバイスが外れたらシステムの既定で開き直して続ける。
      device       選んだ名前("" = システムの設定に合わせる)
      device_used  実際に開いた名前
      health       "ok" / "zeros"(完全な 0 = 許可が無い等)/ "stall"(届かない)/ "fail"(開けない)
      speech       今 声がある(VAD。MicTest は音量で見当)
      on_device(event, want, using)  "missing" / "lost" / "switched" / "failed"
    """

    # クラスの既定(__new__ で作った試験の偽物・古い呼び方でも動く)
    device = ""
    device_used = ""
    health = "ok"
    speech = False
    on_device = None
    _meter = 0.0
    _switching = False
    _zero_noted = False
    _reopen = None
    # 読み上げ中も聞き続ける(pace: エコー消し・ヘッドホン)ときは paused の代わりにこれを立てる。
    # 画面は「読み上げ中」と「聞いています」を同時に出す(hear_state の speak_listen)
    tts_active = None

    def _reopen_ev(self) -> threading.Event:
        if self._reopen is None:
            self._reopen = threading.Event()
        return self._reopen

    def _stream_init(self) -> None:
        self.device = ""
        self.device_used = ""
        self.health = "ok"
        self.speech = False
        self.on_device = None
        self._meter = 0.0
        self._reopen = threading.Event()
        self._switching = False
        self._zero_noted = False

    def start(self):
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def switch_device(self, name: str) -> None:
        """会話を続けたまま、マイクだけ別のデバイスで開き直す。"""
        self.device = name or ""
        self._reopen_ev().set()

    def take_meter(self) -> float:
        """前に読んでからの一番大きいメーターの値(読むと 0 に戻す)。"""
        m, self._meter = self._meter, 0.0
        return m

    def _report(self, title: str, detail: str) -> None:
        print(f"[mic] {title}: {detail}", flush=True)
        cb = getattr(self, "on_error", None)
        if cb is not None:
            try:
                cb(title, detail)
            except Exception:  # noqa: BLE001
                pass

    def _note(self, event: str, want: str, using: str) -> None:
        print(f"[mic] デバイス {event}: {want or 'システム'} → {using or 'システムの既定'}", flush=True)
        cb = getattr(self, "on_device", None)
        if cb is not None:
            try:
                cb(event, want, using)
            except Exception:  # noqa: BLE001
                pass

    def _target(self) -> "tuple[int | None, str, bool]":
        """開くデバイス (PortAudio の番号, 名前, 選んだものが無かったか)。分からなければ (None, "", False)
        = PortAudio の既定(試験の偽物の sounddevice など)。"""
        try:
            name, missing = devices.resolve(devices.INPUT, getattr(self, "device", ""))
        except Exception:  # noqa: BLE001
            return None, "", False
        return devices.pa_index(sd, devices.INPUT, name), name, missing

    def _run(self):
        force_default = False
        while not self._stop.is_set():
            self._reopen_ev().clear()
            want = getattr(self, "device", "")
            idx, name, missing = (None, "", False) if force_default else self._target()
            r = self._run_stream(idx, name, missing, want)
            if r == "fail" and idx is not None:
                # 選んだ・既定のデバイスを番号で開けない(使用中など): PortAudio の既定で開き直す
                if want:
                    self._note("failed", want, "")
                force_default = True
                continue
            if r == "lost":
                time.sleep(0.3)
                self.health = "ok"
                new = ""
                try:
                    new = devices.resolve(devices.INPUT, getattr(self, "device", ""))[0]
                except Exception:  # noqa: BLE001
                    pass
                self._note("lost", name, new)
                continue
            if r == "reopen":
                force_default = False
                continue
            return

    def _run_stream(self, idx, name: str, missing: bool, want: str) -> str:
        """1本のストリームを開いて保つ。返す: "stop" / "reopen" / "lost" / "fail"。"""
        last = [time.monotonic()]
        errs = [0]
        # マイクの許可が無いと macOS は完全な 0 のサンプルを渡し続ける(例外も出ない)。本物のマイクの
        # 無音には必ず小さな雑音が乗るので、ちょうど 0.0 だけが zero_s 秒続いたら知らせる(1回だけ)
        zeros = [0]
        zero_n = int(getattr(self, "zero_s", 3.0) * self.sr)

        def cb(indata, frames, t, status):
            last[0] = time.monotonic()
            try:
                if indata.any():
                    zeros[0] = 0
                    if self.health == "zeros":
                        self.health = "ok"
                else:
                    zeros[0] += len(indata)
                    if zeros[0] >= zero_n and self.health != "zeros":
                        self.health = "zeros"
                        if not self._zero_noted:
                            self._zero_noted = True
                            self._report("マイクの音が届いていません(許可が無い可能性)",
                                         "システム設定 > プライバシーとセキュリティ > マイク で Sokutsu"
                                         "(または使っているターミナル/Python)を許可してください")
            except Exception:  # noqa: BLE001
                pass
            try:
                self.on_chunk(self._mixed(indata[:, 0]))
            except Exception as e:  # noqa: BLE001 - 1チャンク捨てても、マイクは止めない
                errs[0] += 1
                if errs[0] == 1 or errs[0] % 1000 == 0:
                    print(f"[mic] チャンクの処理に失敗({errs[0]}回目) {type(e).__name__}: {e}",
                          flush=True)

        kw = {"samplerate": self.sr, "channels": 1, "dtype": "float32", "callback": cb}
        if idx is not None:
            kw["device"] = idx
        stall_s = getattr(self, "stall_s", 3.0)
        devices.stream_opened()
        try:
            with self._open_input(kw):
                self.device_used = name
                if self.health in ("fail", "stall"):
                    self.health = "ok"
                if missing:
                    self._note("missing", want, name)
                elif self._switching:
                    self._note("switched", want, name)
                self._switching = False
                last[0] = time.monotonic()
                stalled = False
                while not self._stop.is_set():
                    if self._reopen_ev().wait(0.1):
                        self._switching = True
                        return "reopen"
                    quiet = time.monotonic() - last[0]
                    if ((quiet > stall_s or self.health == "zeros") and name
                            and devices.find(devices.INPUT, name) is None):
                        return "lost"          # 使っていたデバイスが外れた: 既定で開き直す
                    if not stalled and quiet > stall_s:
                        stalled = True
                        self.health = "stall"
                        self._report("マイクの音が届いていません",
                                     "マイクが外れた・切り替わった可能性があります。接続を確かめ、"
                                     "会話を終えてから始め直してください")
                    elif stalled and quiet < 0.5:
                        stalled = False
                        self.health = "ok"
                        print("[mic] マイクの音がまた届き始めました", flush=True)
                return "stop"
        except Exception as e:  # noqa: BLE001
            if self._stop.is_set():
                return "stop"
            if idx is not None:
                return "fail"
            self.health = "fail"
            self._report("マイクを開けません",
                         f"{type(e).__name__}: {e} — マイクの接続とシステム設定の"
                         "マイクの許可を確かめ、会話を始め直してください")
            return "fail"
        finally:
            devices.stream_closed()
            if not self._stop.is_set():
                try:
                    self._after_stream()
                except Exception:  # noqa: BLE001
                    pass

    def _open_input(self, kw: dict):
        """入力の入口(with で使えるストリーム)。既定は sounddevice.InputStream(**kw)。
        kw = samplerate・channels=1・dtype="float32"・callback(indata[frames,1], frames, time, status)
        (+ 選んだ機器があれば device = PortAudio の番号。名前は self.device_used、UID は devices.find)。
        エコー消しの入力(macOS VoiceProcessingIO, pace)はここを上書きして、同じ callback を呼べばよい:
        止まった・外れた・完全な 0 の見張りとデバイスの選び直しはそのまま効く。"""
        return sd.InputStream(**kw)

    def _after_stream(self) -> None:
        """ストリームを閉じた後・開き直す前(既定は何もしない)。"""

    def _mixed(self, mono):
        return mono

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)


class SpeechCapture(_MicStream):
    """マイクを16kHzで読み、区切った発話を out_queue へ積む。

    積むのは (音声 float32 mono, 開始秒, 終了秒, 手前の補い秒)。秒はマイク入力の通算。
    fine=True(リモートサーバーの耳): VAD の無音判定0.2秒 + 音量の落ち込み0.18秒で細かく切る
    (後段の Endpointer が文末を見て繋ぎ直す)。fine=False(このMacの耳、1回2秒強):
    細かく切ると認識が追いつかないので従来どおり min_silence_s の無音で切る。
    """

    def __init__(self, vad_model: str, out_queue: "queue.Queue",
                 sr: int = 16000, min_speech_s=0.35, max_speech_s=12.0, min_silence_s=0.6,
                 vad_threshold: float = 0.5, fine: bool = False):
        self.sr = sr
        self.out = out_queue
        self.paused = threading.Event()
        self.level = 0.0  # 直近のRMS(0..1 相当、UIでスケール)
        # 途中字幕(11/06): 話し続けている最中も interim_s 秒ごとに「ここまでの音声」を置く。
        # 0 なら作らない。STT ワーカーが take_interim() で最新の1つだけ取る。
        self.interim_s = 0.0
        self.interim_first_s = 1.0
        self._interim = None          # (音声, 区間の話し始め 通算秒)
        self._interim_at = 0          # 最後に置いたときの区間の長さ(サンプル)
        self._interim_seg = None      # その区間の話し始め(区間が替わったら数え直す)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._clock: "tuple[float, float] | None" = None   # (マイク通算秒, 壁時計) の最新
        # 先送り(11/06, uplink.py): 話している区間の声を リモートサーバーの耳へ先に送っておく
        self.uplink = None
        self._up = 0                  # ここ(通算サンプル)まで送る行列に積んだ
        # 11/07 オンライン会議: マイクの各チャンクに足す「この Mac の音」(sysaudio.MicPlusSystem)
        self.mix_in = None
        self.mic_muted = threading.Event()   # 読み上げ中にマイクだけ止める(11/08)
        self.tts_active = threading.Event()  # 読み上げ中も聞いている(メーター「読み上げ中 — 聞いています」)
        self._was_paused = False
        # 録音(W2, recorder.Recorder.tap): 全チャンクを (音声, 止めているか) で渡す
        self.tap = None
        # マイクの異常(開けない・音が届かなくなった)を画面へ: on_error(見出し, 詳しく)
        self.on_error = None
        self.stall_s = 3.0            # これだけマイクのチャンクが来なければ「届いていない」
        self._stream_init()           # デバイスの選び方・メーター・状態(11/08 hear, _MicStream)

        import sherpa_onnx
        conf = sherpa_onnx.VadModelConfig()
        conf.silero_vad.model = vad_model
        conf.silero_vad.threshold = vad_threshold  # 騒がしい会場では 0.65 へ(起動画面で選択)
        vad_sil = 0.2 if fine else min_silence_s
        conf.silero_vad.min_silence_duration = vad_sil
        conf.silero_vad.min_speech_duration = min_speech_s
        # 区間の長さは StreamSegmenter が決める(VAD は声の有無だけ)
        conf.silero_vad.max_speech_duration = max(30.0, max_speech_s)
        conf.sample_rate = self.sr
        self.vad = sherpa_onnx.VoiceActivityDetector(conf, buffer_size_in_seconds=30)
        self.agc = AutoGain(self.sr)
        self.segmenter = StreamSegmenter(
            self.sr, min_speech_s=min_speech_s, min_silence_s=vad_sil,
            cut_s=0.18 if fine else 1e9, max_s=max_speech_s)

    def on_chunk(self, mono) -> None:
        """マイクの1チャンク(512サンプル)。評価(scripts/eval_live.py)も WAV をここへ流す。"""
        if self.paused.is_set() or self._stop.is_set():
            if not self._was_paused:
                self._was_paused = True
                items = self.segmenter.flush_current()
                self.stream_up(items)
                for item in items:
                    self.out.put(item)   # 止める直前まで話していた分は捨てずに渡す
            self._tap(mono, True)   # 録音(W2): 読み上げでマイクを止めている間も録る
            return
        self._was_paused = False
        self._tap(mono, False)
        r = float(np.sqrt(np.mean(np.square(mono, dtype=np.float64))))
        self.level = min(1.0, r * 3.0)
        self._meter = max(self._meter, meter_of(r))   # 画面のメーター(11/08 hear)
        g = self.agc(mono)  # 小さい声を持ち上げてから VAD へ(遠い卓上マイク対策)
        self.vad.accept_waveform(g)
        while not self.vad.empty():  # VAD の区間は使わない(声の有無だけ見る)
            self.vad.pop()
        self.speech = speech = bool(self.vad.is_speech_detected())
        items = self.segmenter.feed(g, speech)
        self._clock = (self.segmenter._fed / self.sr, time.time())
        self.stream_up(items)
        for item in items:
            self.out.put(item)
        self._maybe_interim(bool(items))

    def _after_stream(self) -> None:
        """デバイスを替える・外れたので開き直す前(ストリームは閉じていてコールバックは来ない):
        話し途中の区間は捨てずに渡し、前のデバイスの VAD の状態は捨てる。"""
        items = self.segmenter.flush_current()
        self.stream_up(items)
        for item in items:
            self.out.put(item)
        self.speech = False
        try:
            self.vad.reset()
        except Exception:  # noqa: BLE001
            pass

    def _tap(self, mono, paused: bool) -> None:
        """録音(W2, recorder.Recorder.tap)へ全チャンクを渡す。壁時計とずらさないため止めている間も。"""
        tap = getattr(self, "tap", None)
        if tap is not None:
            tap(mono, paused)

    def stream_up(self, items) -> None:
        """話している区間(手前の補いも含む)の、まだ積んでいない声を先送りの行列へ。
        閉じた区間は終わり(hi)まで。区間の外の無音は送らない。"""
        up = self.uplink
        if up is None:
            return
        sg = self.segmenter
        rngs = [(int(round(lo * self.sr)), int(round(hi * self.sr))) for _a, lo, hi, _p in items]
        if sg._start is not None:
            rngs.append((max(sg._start - sg.pre, sg._prev_hi, 0), sg._fed))
        for a, b in rngs:
            a = max(a, self._up, sg._buf_start)
            if b > a:
                up.feed(a, sg._slice(a, b))
                self._up = b

    def _maybe_interim(self, closed: bool) -> None:
        sg = self.segmenter
        if self.interim_s <= 0:
            return
        if closed or sg._start is None:
            self._interim = None        # 区間が閉じた: 確定の方で聞く
            self._interim_at = 0
            return
        n = sg._fed - sg._start
        if sg._start != self._interim_seg:   # 新しい区間(読み上げで止めて閉じた後なども)
            self._interim_seg = sg._start
            self._interim_at = 0
        if n < self.interim_first_s * self.sr or n - self._interim_at < self.interim_s * self.sr:
            return
        a = sg.current_audio()
        if a is not None:
            self._interim_at = n
            # (音声, 話し始め秒, 音声の頭の秒): 頭の秒は途中字幕の字の時刻を通算に直す(長い発話を区切る)
            self._interim = (a, sg._start / self.sr, (sg._fed - len(a)) / self.sr)

    def take_interim(self):
        """最新の途中音声 (音声, 話し始め秒, 音声の頭の秒) を1つ取る(無ければ None)。"""
        it, self._interim = self._interim, None
        return it

    def wall_of(self, mic_s: float) -> "float | None":
        """マイク通算秒 → 壁時計(time.time())。話し終わりの時刻を応答速度の起点にする。"""
        c = self._clock
        if c is None:
            return None
        return c[1] - (c[0] - mic_s)

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)
        mx = getattr(self, "mix_in", None)
        if mx is not None:
            try:
                mx.stop()
            except Exception:
                pass

    def _mixed(self, mono):
        """マイク + この Mac の音(11/07, sysaudio.MicPlusSystem)。mix_in が無ければマイクだけ。"""
        mm = getattr(self, "mic_muted", None)
        if mm is not None and mm.is_set():
            # 読み上げ中(11/08): マイクだけ止めて、この Mac の音(Sokutsu 自身の声は入っていない)は聞く
            mono = np.zeros_like(mono)
        mx = self.mix_in
        if mx is None:
            return mono
        try:
            return mx.mix(mono)
        except Exception:
            return mono

    def resume_after_tts(self, tail_s: float):
        """TTS読み上げ後、バッファを捨ててテール分だけ入力を無視してから再開。"""
        self.paused.set()
        try:
            self.vad.reset()
        except Exception:
            pass
        self.segmenter.drop_current()  # 話し途中の区間は捨てる
        time.sleep(tail_s)
        self.paused.clear()


class MicTest(_MicStream):
    """起動画面の「マイクの確認」: 選んだデバイスを開いてメーターだけ動かす(VAD も区切りもしない)。
    起動画面が見えている間だけ開き、離れたら stop() でデバイスを放す。"""

    def __init__(self, device: str = "", sr: int = 16000):
        self.sr = sr
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.paused = threading.Event()   # 読み上げは無い(状態の計算で同じ形にするため)
        self.on_error = None
        self.stall_s = 3.0
        self._stream_init()
        self.device = device or ""

    def on_chunk(self, mono) -> None:
        r = float(np.sqrt(np.mean(np.square(mono, dtype=np.float64)))) if len(mono) else 0.0
        m = meter_of(r)
        self._meter = max(self._meter, m)
        self.speech = m > 0.45
