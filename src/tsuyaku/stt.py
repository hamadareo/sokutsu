"""whisper.cpp (pywhispercpp) による多言語音声認識。言語は文字種ベースで後判定する。"""
from __future__ import annotations

import numpy as np

from .langdetect import detect_lang, resolve_lang


class STT:
    def __init__(self, model_path: str, n_threads: int | None = None):
        from pywhispercpp.model import Model
        kw = {}
        if n_threads:
            kw["n_threads"] = n_threads
        # params_sampling_strategy=0 + best_of=1 = greedy デコード。既定の
        # beam search(beam=5) 比で通訳1文の復号が体感2倍速になる(11/05実測)。
        # 通訳用途ではビームの微妙な精度差より遅延が効くのでgreedy採用。
        kw.setdefault("params_sampling_strategy", 0)
        self.model = Model(model_path, **kw)

    def transcribe(self, samples: np.ndarray) -> tuple[str, str, float]:
        """16kHz mono (float32 or int16) を渡す → (text, lang, prob)。

        prob はセグメントの最低トークン確率(0〜1、取れれば)。幻覚守卫
        (guards.hallucinated_out_of_pair)がこれを見て、雑音が生んだ低確率の
        幻覚を長さに関係なく捨てる(11/05の長い韓国語幻覚の抜けどまり対策)。
        """
        from .guards import seg_probability
        if samples.dtype == np.int16:
            samples = samples.astype(np.float32) / 32768.0
        samples = np.ascontiguousarray(samples, dtype=np.float32)
        # language="auto" を明示しないと英固定になり、日本語音声が英訳で返る
        # greedy: 1トークン=1候補。suppress_nst: 翻訳モード時の "*Mario plays*"
        # 的な音楽/効果音アーティファクトを抑制(11/05実音声で出現)。
        # extract_probability: トークン確率を返させる(幻覚守卫が使う)。
        segs = self.model.transcribe(samples, language="auto", greedy={"best_of": 1},
                                     suppress_nst=True, print_progress=False,
                                     print_timestamps=False, extract_probability=True)
        text = " ".join(s.text.strip() for s in segs if s.text.strip()).strip()
        lang = detect_lang(text)
        prob = seg_probability(segs)
        return text, lang, prob


_CJK = "\u3000-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\uff01-\uff0f\uff1a-\uff20\uff5e-\uff65"


def join_cjk_spaces(text: str) -> str:
    """日本語・中国語の字と字の間の空白を消す(SenseVoice は語ごとに空白を入れる)。
    欧文どうし・ハングルの分かち書きの空白は残す。"""
    import re
    return re.sub(f"(?<=[{_CJK}])\\s+(?=[{_CJK}])", "", text)


class SenseVoiceSTT:
    """このMacの速い耳(11/06「MacBook処理でも即答できるようにしてください」)。

    見つけた問題: このMacの whisper large-v3-turbo(Metal)は1回2.1〜2.5秒(FLEURS 48文、
    平均10秒の音声)。話し終わり→字幕の待ちの大半がこれで、細かく区切って文末で
    繋ぎ直す方式(Endpointer)も使えなかった。
    SenseVoice-Small(sherpa-onnx, int8, CPU 4スレッド)は同じ48文で平均0.15秒
    (約15倍速)。誤り率は ja 11.9% / en 7.1% / zh 12.6% / ko 6.3%(whisper は
    6.1 / 3.8 / 11.2 / 4.1)、言語判定は48/48。非自己回帰なので whisper のような
    長い幻覚は出ないが、雑音には毎回「그.」(韓国語)を返し、トークン確率は出ない。
    → 1文字以下は捨てる、ペア外の言語は雑音扱い(prob=0 で幻覚守卫が捨てる)。
    """

    fast = True
    # BGM と判定された長い窓の文字を残すか(11/08): "head" = 頭の HEAD_S 秒が Speech なら残す(話の続き)/
    # "loose" = 6字以上なら残す / "off" = 捨てる(今まで)。評価で比べる
    bgm_keep = "head"

    def __init__(self, model_dir: str, num_threads: int = 4):
        import threading
        from pathlib import Path

        import sherpa_onnx
        d = Path(model_dir)
        model = d / "model.int8.onnx"
        if not model.exists():
            model = d / "model.onnx"
        self.rec = sherpa_onnx.OfflineRecognizer.from_sense_voice(
            model=str(model), tokens=str(d / "tokens.txt"), num_threads=num_threads,
            use_itn=True, language="auto")   # use_itn: 句読点・数字(Endpointer が文末を見る)
        self.langs: "tuple[str, ...]" = ()
        self.prompt = ""
        self._lock = threading.Lock()

    clock = staticmethod(__import__("time").time)   # 評価(eval_long)は音声の時計に差し替える
    _last_speech_t = -1e9                            # 最後に話し声(Speech)と聞こえた時刻

    def transcribe(self, samples: np.ndarray, ctx=None, cont: bool = False) -> tuple[str, str, float]:
        return self.transcribe_stamped(samples, cont=cont)[:3]

    def _head_is_speech(self, x: np.ndarray) -> bool:
        """窓の頭 HEAD_S 秒だけを聞き直し、話し声(Speech)で字があるか。"""
        from .langdetect import norm
        with self._lock:
            s = self.rec.create_stream()
            s.accept_waveform(16000, np.ascontiguousarray(x[:int(HEAD_S * 16000)]))
            self.rec.decode_stream(s)
            r = s.result
        ev = (r.event or "").strip("<|>")
        return ev == "Speech" and len(norm(r.text or "")) >= 3

    def transcribe_stamped(self, samples: np.ndarray, cont: bool = False):
        """transcribe と同じ (文字, 言語, 確率) に、字(トークン)とその時刻(音声の頭からの秒)を足して返す。
        長い発話を句読点・息つぎで区切る(segmenter.pick_cut)ための時刻。捨てたときは字も時刻も空。
        cont=True: 話の続きと分かっている音声(区切った後半)。BGM と判定されても字が十分なら残す。"""
        from .langdetect import norm
        if samples.dtype == np.int16:
            samples = samples.astype(np.float32) / 32768.0
        x = np.ascontiguousarray(samples, dtype=np.float32)
        with self._lock:
            s = self.rec.create_stream()
            s.accept_waveform(16000, x)
            self.rec.decode_stream(s)
            r = s.result
        text = join_cjk_spaces((r.text or "").strip())
        heard = (r.lang or "").strip("<|>")
        heard = {"yue": "zh"}.get(heard, heard)
        event = (r.event or "").strip("<|>")
        head = None
        now = self.clock()
        if event == "Speech" and len(norm(text)) >= 3:
            self._last_speech_t = now
        if (event == "BGM" and self.bgm_keep == "head" and len(norm(text)) >= EVENT_MIN_CHARS):
            if cont or now - self._last_speech_t <= CONT_S:
                # cont = 区切った後半(話の続きと分かっている区間。StreamSegmenter.starts_at_cut)
                head = True      # 数秒前まで話し声: 話の続き(区切った後半は頭から BGM と判定されやすい)
            elif len(x) > int(HEAD_S * 16000) + 16000:
                head = self._head_is_speech(x)
        if head and self.bgm_keep == "head":
            # 歌の窓は字が遅い(1音に長く伸ばす)。話し声の速さでなければ BGM の上の声とみなさない
            rate = char_rate(list(r.tokens), [float(t) for t in r.timestamps])
            if rate is not None and rate < BGM_MIN_RATE:
                head = False
        if not keep_event(event, text, head if self.bgm_keep == "head" else
                          (True if self.bgm_keep == "loose" else False)):
            return "", "", 0.0, [], []       # 拍手・笑い・BGM だけ(歌の歌詞も)
        if len(norm(text)) <= 1:             # 雑音の「그.」「..」
            return "", "", 0.0, [], []
        lang = resolve_lang(text, heard)
        prob = 1.0
        if self.langs and lang and lang not in self.langs:
            prob = 0.0
        try:
            toks, ts = list(r.tokens), [float(t) for t in r.timestamps]
        except Exception:  # noqa: BLE001 - 時刻を出さない版: 区切らないだけ
            toks, ts = [], []
        if len(toks) != len(ts):
            toks, ts = [], []
        return text, lang, prob, toks, ts


# SenseVoice の出来事が Speech でなくても、これだけの字があれば本物の発話(BGM の上の話し声)。
# 11/08: 作った独白に和音の BGM を重ねると、窓が 9 秒ほどになったところで出来事が BGM になり、
# 正しく聞けている文字ごと捨てて途中字幕が止まっていた。音楽・雑音・拍手だけのときの文字は
# 「我。」「I.」「.」「그.」で、記号を除くと2字以下(scratchpad の probe3)。
# ただし歌詞のある曲も BGM の上の声なので、6字以上だけでは字幕に乗る(作った歌で 12〜68 字)。
# 窓の頭(HEAD_S 秒)が Speech と聞こえた = 話している途中から BGM と判定が変わった、のときだけ残す。
# BGM 以外の出来事(Speech_Noise = ざわめき、笑い、拍手)は今まで通り捨てる。
# 区切った後半(話の途中から始まる区間)は頭から BGM と判定されやすく、頭の確かめで捨てた
# (eval_long の ja・BGM で1文が丸ごと消えた)。CONT_S 秒前までに話し声と聞こえていれば続きとして残す。
EVENT_MIN_CHARS = 6
HEAD_S = 3.0
CONT_S = 4.0
# 歌の見分け(11/08): BGM と判定された窓の字の速さ(句読点を除く字の数 ÷ 最初の字から最後の字まで)。
# VOICEVOX の歌(自作の旋律・歌詞、84/120 BPM、伴奏 0〜2 倍)は p10/p50/p90 = 1.0/1.4/2.0 字/秒、
# BGM の上の独白(zh・ja)は 5.0/5.6/7.0 字/秒(scratchpad の rate_probe、リモートサーバー)。
BGM_MIN_RATE = 3.0
_PUNCT_TOK = set("。．！？!?.，、,；;：: ")


def char_rate(tokens: list, stamps: list) -> "float | None":
    """字(句読点を除く)の速さ 字/秒。字が2つ未満なら None。"""
    ts = [t for tok, t in zip(tokens, stamps) if str(tok).strip() and str(tok).strip() not in _PUNCT_TOK]
    if len(ts) < 2:
        return None
    return len(ts) / max(0.5, ts[-1] - ts[0])


def keep_event(event: str, text: str, head_speech: "bool | None" = None) -> bool:
    """SenseVoice の出来事(Speech / BGM / Laughter / Applause …)と文字から、発話として残すか。
    head_speech = 窓の頭だけ聞き直して話し声だったか(BGM のときだけ見る)。"""
    from .langdetect import norm
    if not event or event == "Speech":
        return True
    if event != "BGM" or not head_speech:
        return False
    return len(norm(text)) >= EVENT_MIN_CHARS


def make_local_stt(s):
    """このMacの耳。既定は SenseVoice(速い)、無い/失敗なら whisper。"""
    if getattr(s, "mac_ear", "sensevoice") == "sensevoice" and getattr(s, "sense_voice_dir", ""):
        try:
            return SenseVoiceSTT(s.sense_voice_dir)
        except Exception as e:  # noqa: BLE001
            print(f"[stt] SenseVoice を読めないため whisper を使います: {e}", flush=True)
    return STT(s.whisper_model)


def _is_timeout(e: Exception) -> bool:
    try:
        import requests
        if isinstance(e, requests.Timeout):
            return True
    except ImportError:
        pass
    return isinstance(e, TimeoutError)


class RemoteSTT:
    """リモートGPU の耳(server/remote_stt_server.py)に投げ、ダメならこのMacの whisper。

    11/05実測(同じ3.6〜4.0秒の発話): このMac 2.3秒 → remote 0.13〜0.26秒(往復込み)。
    話し終わり→字幕の待ちの最大要因だった。リモートが落ちたら即ローカルに落ち、
    cooldown 秒はリモートを試さない(毎発話でタイムアウトを待たないため)。
    言語は応答の lang ではなく文字種判定(detect_lang)で決める: アプリ全体で同じ
    基準にする(サーバーの言語判定は雑音で en に振れる)。prob はサーバーが
    exp(avg_logprob) の最小値で返す = pywhispercpp の probability と同じ定義。
    """

    def __init__(self, url: str | None, local, post=None, timeout: float = 4.0,
                 cooldown: float = 30.0, clock=None, hedge_s: float = 0.0,
                 preview_s: float = 0.0):
        import time as _time
        # hedge_s(11/06): リモートサーバーの返事をこの秒数だけ待ち、来なければこのMacの速い耳
        # (SenseVoice, fast=True)で聞く。回線が混むと往復が0.3〜7秒に揺れるため。
        # 0.7秒/2.5秒で切り替えると混んだ回線では SenseVoice が増えて chrF が2〜4下がった
        # (FLEURS 6会話)ので、確定は今までの待ち(4秒)のまま。速さは preview(予想訳)で出す。
        # 0 = 待ち切る(今まで)。このMacの耳が遅い(whisper)ときは並べても速くならないので使わない。
        self.hedge_s = hedge_s
        self.hedged = 0
        self.preview_s = preview_s   # これを過ぎたら このMacの耳で予想訳だけ先に出す
        self.previewed = 0
        # 回線の混み具合(11/06): 往復からサーバー内の処理時間を引いた分の移動平均。
        # 画面に「回線が混んでいます」を出すため(遅さの原因が回線だと分かるように)。
        self.net_s = 0.0
        self.url = url.rstrip("/") if url else None
        self.local = local
        if post is None:
            import requests
            post = requests.post
        self._post = post
        self.timeout = timeout
        self.cooldown = cooldown
        self._clock = clock or _time.time
        self._down_at: float | None = None
        self.last_where = ""
        self.langs: "tuple[str, ...]" = ()  # 今日の言語ペア: リモートサーバーの言語判定をこの中に絞る
        self.prompt = ""  # 前の文・テーマメモ(whisper の initial_prompt。短い断片の手がかり)
        # 先送り(11/06, uplink.py): 話している間に声を リモートサーバーへ送っておき、聞くときは範囲だけ送る
        self.uplink = None
        self.span_hits = 0      # 範囲だけで聞けた回数
        self.span_miss = 0      # 届いていなくて音声ごと送り直した回数

    def transcribe(self, samples: np.ndarray, ctx=None,
                   on_preview=None, span=None, prompt=None) -> tuple[str, str, float]:
        """ctx: 直前の音声(文脈として聞かせるだけ。リモートサーバーの耳でのみ使い、出力からは除く)。
        on_preview(text, lang): リモートサーバーが preview_s 秒で返らないとき、このMacの耳の結果を
        先に渡す(予想訳を出すため。確定には使わない)。
        span: samples がマイクの通算サンプル番号のどこか([(a, b) か ("z", 無音の長さ)])。
        先送り(uplink)があれば範囲だけ送る。
        prompt: この1回だけ手がかり(whisper の initial_prompt)を上書きする(None = self.prompt、
        "" = 手がかりなし)。self.prompt を書き換えないので、hedge の2本目のように別のスレッドで
        問い合わせを組んでも混ざらない(11/07 用語辞書の守り: 手がかりなしの聞き直し)。"""
        if span is not None and sum((b if a == "z" else b - a) for a, b in span) != len(samples):
            span = None    # 長さが合わない: 範囲は使わない
        if self.url and (self._down_at is None
                         or self._clock() - self._down_at >= self.cooldown):
            if self.hedge_s > 0 and getattr(self.local, "fast", False):
                return self._hedged(samples, ctx, on_preview, span, prompt)
            try:
                out = self._remote(samples, ctx, span, prompt)
                self.last_where = "remote"
                return out
            except Exception as e:  # noqa: BLE001 — 種別を問わずローカルへ
                self._fail(e)
        self.last_where = "mac"
        return self.local.transcribe(samples)

    def _note_net(self, net: float) -> None:
        self.net_s = max(0.0, 0.6 * self.net_s + 0.4 * max(0.0, net))

    @property
    def congested(self) -> bool:
        """回線が混んでいる(耳の往復の上乗せが平均0.5秒超)。"""
        return bool(self.url) and self.net_s > 0.5

    def _fail(self, e: Exception) -> None:
        print(f"[stt-remote] リモートサーバーの耳に失敗({type(e).__name__}) → このMacで認識",
              flush=True)
        self._down_at = self._clock()

    def _hedged(self, samples: np.ndarray, ctx, on_preview=None,
                span=None, prompt=None) -> tuple[str, str, float]:
        import threading
        import time as _t
        t0 = _t.time()
        box: dict = {}
        done = threading.Event()

        def run():
            try:
                box["out"] = self._remote(samples, ctx, span, prompt)
            except Exception as e:  # noqa: BLE001
                box["err"] = e
                if _is_timeout(e):
                    # 遅いだけ(落ちてはいない): 次の発話も リモートサーバーを試す。以前は1回の
                    # タイムアウトで30秒このMacの耳に固定され、混んだ回線では精度が落ちていた
                    print("[stt-remote] リモートサーバーの耳が時間切れ(回線) → この発話はこのMacで認識",
                          flush=True)
                else:
                    self._fail(e)      # 落ちている: 以後しばらくローカル
            finally:
                done.set()

        threading.Thread(target=run, daemon=True).start()
        local = None
        if (on_preview is not None and 0 < self.preview_s < self.hedge_s
                and not done.wait(self.preview_s)):
            self._note_net(self.preview_s)     # 少なくともこれだけ遅れている
            # リモートサーバーが遅い: このMacの耳で先に聞いて予想訳を出す(確定は リモートサーバーの結果)
            local = self.local.transcribe(samples)
            self.previewed += 1
            if local[0] and local[1]:
                try:
                    on_preview(local[0], local[1])
                except Exception:  # noqa: BLE001
                    pass
        if done.wait(max(0.0, self.hedge_s - (_t.time() - t0))) and "out" in box:
            self.last_where = "remote"
            return box["out"]
        if not done.is_set():
            self._note_net(self.hedge_s)       # 返らなかった = 混んでいる
            self.hedged += 1
            print(f"[stt-remote] リモートサーバーの耳が{self.hedge_s:.1f}秒で返らない(回線) → このMacで認識",
                  flush=True)
        self.last_where = "mac"
        return local if local is not None else self.local.transcribe(samples)

    def _remote(self, samples: np.ndarray, ctx, span=None, prompt=None) -> tuple[str, str, float]:
        up = self.uplink
        if span is not None and up is not None and ctx is None:
            plan = up.plan(span)
            why = "手元に無い"
            if plan is not None:
                recipe, need, body = plan
                params = {"sid": up.sid, "r": recipe}
                if need:
                    params["u"] = ",".join(f"{a}-{b}" for a, b in need)
                r, obj = self._ask(body, params, post=up.post, prompt=prompt)
                why = "サーバーに無い"
                if getattr(r, "status_code", 200) == 409:
                    up.forget(span)          # 次からその範囲は本文で送る
                else:
                    r.raise_for_status()
                    if obj.get("span"):      # 古いサーバーは範囲を知らず空の本文を聞く
                        up.acked(need)
                        self.span_hits += 1
                        return self._result(obj)
                    why = "古いサーバー"
            self.span_miss += 1   # 届いていない/古いサーバー: 今までどおり音声ごと
            print(f"[uplink] 範囲で聞けない({why}) {span[:3]}… → 音声ごと送る", flush=True)
        ctx_s = 0.0
        if ctx is not None and len(ctx):
            c = np.asarray(ctx, dtype=np.float32)
            if samples.dtype == np.int16:
                c = (c * 32767.0).astype(np.int16)
            ctx_s = len(c) / 16000.0
            full = np.concatenate([c, samples.astype(c.dtype)])
        else:
            full = samples
        if full.dtype == np.int16:
            pcm = full.astype("<i2").tobytes()
        else:
            x = np.clip(np.asarray(full, dtype=np.float32), -1.0, 1.0)
            pcm = (x * 32767.0).astype("<i2").tobytes()
        params = {}
        if ctx_s > 0:
            params["ctx"] = f"{ctx_s:.3f}"
        r, obj = self._ask(pcm, params, prompt=prompt)
        r.raise_for_status()
        return self._result(obj)

    def _ask(self, body: bytes, params: dict, post=None, prompt=None):
        from urllib.parse import urlencode
        params = dict(params)
        if self.langs:
            params["langs"] = ",".join(self.langs)
        pr = self.prompt if prompt is None else prompt
        if pr:
            params["prompt"] = pr[-300:]
        q = ("?" + urlencode(params, safe=",-")) if params else ""
        import time as _t
        t0 = _t.time()
        r = (post or self._post)(f"{self.url}/transcribe{q}", data=body, timeout=self.timeout)
        if getattr(r, "status_code", 200) >= 400:
            return r, {}
        obj = r.json()
        try:
            self._note_net(_t.time() - t0 - float(obj.get("sec", 0.0) or 0.0))
        except (TypeError, ValueError):
            pass
        return r, obj

    def _result(self, obj: dict) -> tuple[str, str, float]:
        text = str(obj.get("text") or "").strip()
        prob = float(obj.get("prob", 1.0))
        self._down_at = None
        return text, resolve_lang(text, str(obj.get("lang") or "")), prob
