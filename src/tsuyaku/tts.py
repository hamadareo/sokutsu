"""sherpa-onnx オフラインTTS + VOICEVOX ENGINE + macOS say。言語ごとに「声(エンジン+音色)」と「速さ」を変えられる。

声の指定は "kokoro:3"(Kokoroのsid3) / "say:Kyoko"(macOS音声) / "voicevox:13"
(VOICEVOXのstyleId13=青山龍星) の文字列。UIから実行中に変更でき、設定は config.json に保存される。
VOICEVOX はこのMacのローカルエンジン (127.0.0.1:50021、~/voicevox)。
エンジンが無い/失敗しても音声は止まらず macOS say へフォールバックする。
再生は macOS の afplay を使う(PortAudioと違いマイクやGPUの負荷でぷつぷつ切れない)。
"""
from __future__ import annotations

import os
import subprocess
import threading
import time
from pathlib import Path

import numpy as np


def _find(d: Path, *patterns: str) -> str | None:
    for p in patterns:
        hits = sorted(d.glob(p))
        if hits:
            return str(hits[0])
    return None


def remove_tones(samples: np.ndarray, sr: int, base: float = 2400.0,
                 width: float = 25.0) -> np.ndarray:
    """Kokoro の出力に乗る 2400Hz の倍音(4800/7200/9600Hz)の「キーン」を消す。

    11/06 日中の実会話「声がキーンって聞こえる」。全声で 4800/9600Hz に周りの
    30〜130倍の鋭い山があった(int8・全精度の両方、声の出ている間だけ)。
    声の出口(iSTFT)の周期の名残なので、その周波数だけを ±width Hz 削る。
    声の倍音は毎瞬高さが変わるので、固定周波数の細い溝では削れない。
    """
    x = np.asarray(samples, dtype=np.float32)
    if len(x) < sr // 10:
        return x
    X = np.fft.rfft(x)
    f = np.fft.rfftfreq(len(x), 1.0 / sr)
    k = base
    while k < sr / 2:
        X[np.abs(f - k) <= width] = 0
        k += base
    return np.fft.irfft(X, n=len(x)).astype(np.float32)


def _load_g2p(d: Path):
    """Kokoro 同梱の米語辞書 + espeak-ng で英語を発音記号にする道具(無ければ None)。"""
    try:
        from .g2p_en import EnG2P
        lex = d / "lexicon-us-en.txt"
        if not lex.exists():
            return None
        vocab = set()
        for line in (d / "tokens.txt").read_text(encoding="utf-8").split("\n"):
            if line:
                vocab.add(" " if line.startswith(" ") else line.split(" ")[0])
        return EnG2P(lex, vocab=vocab)
    except Exception as e:  # noqa: BLE001
        print(f"[tts] 英語の発音記号化を使えません(辞書の単語だけ読みます): {e}")
        return None


# Kyoko/Tingting/Yuna/Alex はいずれもmacOS標準で実在確認済み (11/05)
SAY_VOICE_DEFAULTS = {"ja": "Kyoko", "zh": "Tingting", "ko": "Yuna", "en": "Alex"}
# kokoro-multi-lang-v1_0 の公式sid表: en=0-27, ja=37-41, zh=45-52
KOKORO_SIDS = {"ja": range(37, 42), "zh": range(45, 53), "en": range(0, 28)}
# macOS say の既定speed=1.0に相当する -r (話速) 値。実測: 既定≈185, -r200で約1.12倍速
SAY_BASE_RATE = 185


def parse_voice(spec: str) -> tuple[str, object]:
    """'say:Kyoko' -> ('say','Kyoko') / 'kokoro:3' -> ('kokoro',3)。壊れていたら say へ。"""
    if not spec or ":" not in str(spec):
        return ("say", "")
    kind, _, rest = str(spec).partition(":")
    if kind == "kokoro":
        try:
            return ("kokoro", int(rest))
        except ValueError:
            return ("kokoro", 0)
    if kind == "voicevox":
        try:
            return ("voicevox", int(rest))
        except ValueError:
            return ("voicevox", 13)  # 数値でなければ青山龍星
    return ("say", rest)


def voicevox_voices(url: str, timeout: float = 1.5) -> list[dict]:
    """VOICEVOX ENGINE の音色一覧 [{id, name}]。エンジンが無ければ空。

    43体×複数スタイルあるので、実用向けの「ノーマル」と「アナウンス」だけに
    絞って一覧を出す(クイック/怒った/泣く等は翻訳字幕の読み上げに向かない)。
    """
    if not url:
        return []
    try:
        import requests
        r = requests.get(f"{url}/speakers", timeout=timeout)
        r.raise_for_status()
        out = []
        for sp in r.json():
            for st in sp.get("styles", []):
                sname = str(st.get("name", ""))
                if "ノーマル" in sname:
                    out.append({"id": int(st["id"]), "name": str(sp["name"])})
                elif "アナウンス" in sname:
                    out.append({"id": int(st["id"]), "name": f"{sp['name']} {sname}"})
        return out
    except Exception as e:  # noqa: BLE001 - 不在でも起動画面は出る
        print(f"[tts] voicevox /speakers 取得失敗: {e}")
        return []


def clamp_volume(v) -> float:
    try:
        return max(0.1, min(1.0, float(v)))
    except (TypeError, ValueError):
        return 1.0


def say_text(text: str, volume: float = 1.0) -> str:
    """say の音量は文頭の埋め込みコマンド [[volm x]] で下げる(1.0 ならそのまま)。"""
    v = clamp_volume(volume)
    return text if v > 0.99 else f"[[volm {v:.2f}]] {text}"


def afplay_cmd(path: str, volume: float = 1.0) -> list[str]:
    v = clamp_volume(volume)
    return ["afplay", path] if v > 0.99 else ["afplay", "-v", f"{v:.2f}", path]


def say_cmd(name: str, speed: float = 1.0, device: "str | None" = None) -> list[str]:
    """device: 鳴らす出力デバイスの名前(say -a。None/"" = システムの既定)。"""
    cmd = ["say"]
    if name:
        cmd += ["-v", name]
    if device:
        cmd += ["-a", device]
    try:
        sp = float(speed)
    except (TypeError, ValueError):
        sp = 1.0
    if abs(sp - 1.0) > 0.01:
        cmd += ["-r", str(max(80, min(520, int(SAY_BASE_RATE * sp))))]
    return cmd


_ABBR = ("Mr.", "Mrs.", "Ms.", "Dr.", "St.", "Prof.", "e.g.", "i.e.", "vs.", "etc.", "No.")


# 読点で分けて作る声(11/08 speed): Kokoro は読点ごとに別々に作る(中国語は sherpa が読点で分ける、英語は
# g2p_en が読点で分ける)ので、読点で分けても全部を1回で作るのと同じ音になる。短い頭・しっぽは分けない。
_CLAUSE_RE = {"zh": r"[，；：]", "en": r"[,;:](?=\s)"}
_CLAUSE_MIN = {"zh": 6, "en": 4}       # zh は字数、en は語数


def _clause_short(x: str, lang: str) -> bool:
    return (len(x.split()) if lang == "en" else len(x.strip())) < _CLAUSE_MIN[lang]


def _clause_split(sent: str, lang: str) -> list[str]:
    import re
    pieces, start = [], 0
    for m in re.finditer(_CLAUSE_RE[lang], sent):
        pieces.append(sent[start:m.end()].strip())
        start = m.end()
    if start < len(sent):
        pieces.append(sent[start:].strip())
    pieces = [p for p in pieces if p]
    out: list[str] = []
    for p in pieces:
        if out and _clause_short(out[-1], lang):      # 短い頭(Yes, / 然后，)は次とまとめる
            out[-1] = f"{out[-1]} {p}" if lang == "en" else out[-1] + p
        else:
            out.append(p)
    if len(out) > 1 and _clause_short(out[-1], lang):   # 短いしっぽ(, thanks.)は前とまとめる
        last = out.pop()
        out[-1] = f"{out[-1]} {last}" if lang == "en" else out[-1] + last
    return out


def voice_chunks(text: str, min_chars: int = 8, clauses: "str | None" = None) -> list[str]:
    """読み上げを文ごとに分ける(1文目ができたらすぐ鳴らすため、11/06)。
    文末(。！？!? と、後ろに空白が続く .)で切る。Mr./Dr. などの略語では切らない。
    min_chars 未満の短い文(「はい。」「OK.」)は単独で鳴らさず隣とまとめる。
    clauses="zh"/"en"(Kokoro の声): 長い文は読点でも分ける(11/08 speed)。"""
    import re
    text = (text or "").strip()
    if not text:
        return []
    parts: list[str] = []
    start = 0
    for m in re.finditer(r"[。！？!?]+[」』）)\"']*\s*|\.(?:[\"')]*)\s+", text):
        end = m.end()
        piece = text[start:end].strip()
        if m.group(0).startswith(".") and any(piece.endswith(x) for x in _ABBR):
            continue
        parts.append(piece)
        start = end
    if start < len(text):
        parts.append(text[start:].strip())
    parts = [x for x in parts if x]
    if clauses in _CLAUSE_RE:
        parts = [c for x in parts for c in _clause_split(x, clauses)]
    out: list[str] = []
    carry = ""
    for x in parts:
        x = (carry + (" " if carry and _latin_end(carry) else "") + x) if carry else x
        carry = ""
        if len(x) < min_chars:
            carry = x
            continue
        out.append(x)
    if carry:
        if out:
            out[-1] = out[-1] + (" " if _latin_end(out[-1]) else "") + carry
        else:
            out.append(carry)
    return out


def _latin_end(s: str) -> bool:
    return bool(s) and (s[-1].isascii())


_AV_PLAYER = []


def _av_player_class():
    """AVFoundation の AVAudioPlayer(pyobjc-core だけで読む。Foundation の包みは要らない)。"""
    if not _AV_PLAYER:
        try:
            import objc
            objc.loadBundle("AVFoundation", {},
                            bundle_path="/System/Library/Frameworks/AVFoundation.framework")
            _AV_PLAYER.append(objc.lookUpClass("AVAudioPlayer"))
        except Exception:  # noqa: BLE001
            _AV_PLAYER.append(None)
    return _AV_PLAYER[0]


class _Clip:
    def __init__(self, guess: bool = False):
        self.wav: bytes | None = None
        self.parts: list = []       # できた所から鳴らす音(Kokoro は読点ごと。VOICEVOX は全体1つ)
        self.cv = threading.Condition()
        self.done = threading.Event()
        self.guess = guess          # 予想訳(先読み)の音: 確定訳の音より後に作る
        self.dropped = False        # もう要らない(新しい予想に替わった・捨てた): 作らない
        self.started = False


def _to_wav(samples: np.ndarray, rate: int) -> bytes:
    import io
    import wave
    pcm = (np.clip(np.asarray(samples, dtype=np.float32), -1.0, 1.0) * 32767.0).astype("<i2")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm.tobytes())
    return buf.getvalue()


# 作り置きの声を使ってよい速さの差(11/08 speed)。×0.1 の違いは聞き分けにくく、作り直しの
# 合成(0.3〜1.9秒)を待つより早く鳴る。
SPEED_TOL = 0.1


class TTS:
    """say() は発話完了までブロックする。エンジンが無い/失敗した音声は macOS say へ。"""

    def __init__(self, tts_dir_kokoro: str, tts_dir_ko: str, sids: dict,
                 ja_engine: str = "say", voices: dict | None = None,
                 speeds: dict | None = None, voicevox_url: str = ""):
        self.voicevox_url = (voicevox_url or "").rstrip("/")
        self.sids = dict(sids)
        self.voices: dict[str, str] = dict(voices or {})
        self.speeds: dict[str, float] = {k: float(v) for k, v in (speeds or {}).items()}
        self.engine: dict[str, object] = {}
        self.rate: dict[str, int] = {}
        self._stop = threading.Event()
        self._proc: subprocess.Popen | None = None  # 実行中の afplay / say
        self._clips: "dict[tuple, _Clip]" = {}      # 作り置きの音(prepare)
        self._clip_lock = threading.Lock()
        self._jobs: list = []                       # 作る順の (clip, 文, 言語, 速さ)
        self._jobs_cv = threading.Condition(self._clip_lock)
        self._worker: "threading.Thread | None" = None
        self._synth_lock = threading.Lock()
        self.prepared_hits = 0
        self.volume = 1.0       # 読み上げの音量(0.1〜1.0)
        self.last_speed = 1.0   # 直前に鳴らした速さ(作り置きの速さを使ったときはその速さ)
        self.on_start = None    # 音が鳴り始めた瞬間に呼ぶ(応答速度の表示用)
        # 鳴らす出力デバイスの名前(11/08 hear。"" = システムの既定 = 今までどおり afplay / say)。
        # 会話中に替えると次の読み上げから効く。無ければ既定で鳴らし on_device("missing", 名前, "") で1回知らせる
        self.output_device = ""
        self.on_device = None
        self._player = None     # 再生中の AVAudioPlayer(選んだデバイスで鳴らしているとき)
        self._missing_noted = ""

        kokoro = None
        if tts_dir_kokoro and Path(tts_dir_kokoro).exists():
            try:
                import sherpa_onnx
                d = Path(tts_dir_kokoro).resolve()
                conf = sherpa_onnx.OfflineTtsConfig(
                    model=sherpa_onnx.OfflineTtsModelConfig(
                        kokoro=sherpa_onnx.OfflineTtsKokoroModelConfig(
                            model=_find(d, "model.int8.onnx", "model.onnx", "*.onnx"),
                            voices=_find(d, "voices.bin"),
                            tokens=str(d / "tokens.txt"),
                            lexicon=",".join(str(p) for p in sorted(d.glob("lexicon-*.txt"))),
                            data_dir=str(d / "espeak-ng-data") if (d / "espeak-ng-data").exists() else "",
                            dict_dir=str(d / "dict") if (d / "dict").exists() else "",
                            length_scale=1.0,
                        ),
                        num_threads=4,
                        provider="cpu",
                    ),
                )
                kokoro = sherpa_onnx.OfflineTts(conf)
            except Exception as e:  # noqa: BLE001 - 起動は落とさない
                print(f"[tts] kokoro 読み込み失敗: {e}")
                kokoro = None
        self.g2p_en = None
        if kokoro is not None:
            self.engine["kokoro"] = kokoro
            self.g2p_en = _load_g2p(Path(tts_dir_kokoro).resolve())
            if ja_engine == "kokoro":
                self.rate["ja"] = kokoro.sample_rate
        if tts_dir_ko and Path(tts_dir_ko).exists():
            try:
                import sherpa_onnx
                d = Path(tts_dir_ko).resolve()
                conf = sherpa_onnx.OfflineTtsConfig(
                    model=sherpa_onnx.OfflineTtsModelConfig(
                        vits=sherpa_onnx.OfflineTtsVitsModelConfig(
                            model=_find(d, "*.onnx"),
                            tokens=str(d / "tokens.txt"),
                            data_dir=str(d / "espeak-ng-data") if (d / "espeak-ng-data").exists() else "",
                        ),
                        num_threads=2,
                        provider="cpu",
                    ),
                )
                eng = sherpa_onnx.OfflineTts(conf)
                # sherpa 2.0.x は piper 系モデルの音素化を持っておらず、生成が
                # 空になることがある。実際に発話できていないなら say(Yuna) に委ねる。
                probe = eng.generate("테스트", sid=0, speed=1.0)
                if len(probe.samples) > 1600:
                    self.engine["vits_ko"] = eng
                    self.rate["ko"] = eng.sample_rate
                else:
                    print("[tts] ko モデルは音素化不可のため macOS say(Yuna) を使用")
            except Exception as e:  # noqa: BLE001
                print(f"[tts] ko 読み込み失敗: {e}")

    # ---------- UIから実行中の変更 ----------

    def new_conversation(self) -> None:
        """会話をまたいで使い回すとき(11/08 speed): 前の会話の止め印・作り置き・出力の知らせを捨てる。"""
        self._stop.clear()
        with self._clip_lock:
            for clip in list(self._clips.values()):
                self._drop(clip)
            self._clips.clear()
        self.on_device = None
        self._missing_noted = ""

    def set_voice(self, lang: str, spec: str):
        if parse_voice(spec)[0] in ("say", "kokoro", "voicevox"):
            self.voices[lang] = spec

    def set_volume(self, v: float):
        self.volume = clamp_volume(v)

    def set_speed(self, lang: str, speed: float):
        self.speeds[lang] = max(0.5, min(2.0, float(speed)))

    def has_kokoro(self) -> bool:
        return "kokoro" in self.engine

    # ---------- 発話 ----------

    def say(self, text: str, lang: str, speed: "float | None" = None) -> bool:
        """阻塞して発話完了まで待つ。読み上げたならTrue。cancel()で中断できる。

        prepare() で先に作ってあればその音をすぐ鳴らす(作っている最中なら待つ)。
        speed: この1回だけの速さ(話のペースに合わせる 11/08)。None ならその言語の速さ。
        鳴っていた秒数は self.last_play_s に残る(1秒あたりの字数を合わせるため)。"""
        text = text.strip()
        self._stop.clear()  # 新しい発話なので前回の中止フラグを戻す
        self.last_play_s = 0.0
        if not text:
            return False
        chunks = self._chunks(text, lang)
        if len(chunks) > 1 and parse_voice(self.voices.get(lang, ""))[0] in ("voicevox", "kokoro"):
            return self._say_chunks(chunks, text, lang, speed)
        self.last_speed = self._speed(lang, speed)
        clip = self._take_clip(text, lang, speed)
        hit = clip is not None
        if clip is None and parse_voice(self.voices.get(lang, ""))[0] in ("voicevox", "kokoro"):
            # 作り置きが無い: 作る担当の行列の先頭に入れて待つ(予想訳の音より先に作る)
            self.prepare(text, lang, speed)
            clip = self._take_clip(text, lang, speed)
        if clip is not None and self._play_clip(clip):
            if hit:
                self.prepared_hits += 1
            return True
        self.last_speed = self._speed(lang, speed)
        wav = self._synth(text, lang, speed)
        if wav:
            return self._play_wav(wav)
        sp = self._speed(lang, speed)
        kind, val = parse_voice(self.voices.get(lang, ""))
        name = val if kind == "say" and val else SAY_VOICE_DEFAULTS.get(lang, "")
        return self._say_macos(text, name, sp)

    def _speed(self, lang: str, speed: "float | None") -> float:
        return float(speed) if speed else self.speeds.get(lang, 1.0)

    def _say_chunks(self, chunks: list[str], text: str, lang: str,
                    speed: "float | None" = None) -> bool:
        """文ごとに作って順に鳴らす。作るのは1本のスレッドが文の順に(prepare)、
        鳴らすのはここ。1文目ができた瞬間に鳴り始め、その間に2文目を作る。"""
        with self._clip_lock:
            ready = self._near_key(chunks[0], lang, speed) is not None
        if ready:
            self.prepared_hits += 1
        self.last_speed = self._speed(lang, speed)
        self.prepare(text, lang, speed)
        for c in chunks:
            if self._stop.is_set():
                break
            clip = self._take_clip(c, lang, speed)
            if clip is not None and self._play_clip(clip):
                continue
            wav = self._synth(c, lang, speed)
            if self._stop.is_set():
                break
            if wav:
                self._play_wav(wav)
            else:
                self._say_macos(c, SAY_VOICE_DEFAULTS.get(lang, ""), self._speed(lang, speed))
        return True

    # ---------- 先に音を作っておく(11/06) ----------
    # 合成は鳴り始めまでに 0.25〜1.0秒かかる(M3 実測: Kokoro 英語2.8秒の文で0.7秒、
    # 中国語3.7秒で1.0秒、VOICEVOX 2.8秒で0.7秒)。予想訳ができた時点/前の読み上げ中に
    # 作っておけば、確定した瞬間に鳴らせる。macOS say の声は作り置きしない(その場で読む)。

    def _chunks(self, text: str, lang: str) -> list[str]:
        """声を作る単位: 文ごと。Kokoro の英語・中国語は読点ごとも(作る音は同じ。11/08 speed)。"""
        kokoro = parse_voice(self.voices.get(lang, ""))[0] == "kokoro" and lang in _CLAUSE_RE
        return voice_chunks(text, clauses=lang if kokoro else None)

    def stable_chunks(self, partial: str, lang: str) -> list[str]:
        """流し込み途中の訳のうち、もう変わらない単位(後ろに十分な字が続いている読点・文まで)。
        Kokoro の声だけ(VOICEVOX は読点で分けない)。11/08 speed: 訳がそろう前に声を作り始める。"""
        if parse_voice(self.voices.get(lang, ""))[0] != "kokoro" or lang not in _CLAUSE_RE:
            return []
        chunks = self._chunks(partial, lang)
        out = []
        pos = 0
        text = (partial or "").strip()
        for c in chunks[:-1]:
            i = text.find(c, pos)
            if i < 0:
                break
            pos = i + len(c)
            rest = text[pos:].strip()
            if len(rest) < 8 or _clause_short(rest, lang):
                break      # 後ろが短い: まだ前とまとめられるかもしれない
            out.append(c)
        return out

    def _clip_key(self, text: str, lang: str, speed: "float | None" = None) -> tuple:
        return (text.strip(), lang, self.voices.get(lang, ""), round(self._speed(lang, speed), 2))

    def _near_key(self, text: str, lang: str, speed: "float | None" = None) -> "tuple | None":
        """作り置きの中で、同じ文・同じ声で速さが SPEED_TOL 以内のもの(いちばん近いもの)の鍵。
        _clip_lock を持って呼ぶ。
        11/08 speed: 速さが「自動」だと行列に入れる時と読む直前で速さを決め直し(0.01 刻み)、
        作り置き(×1.11)と読む時(×1.08)が合わずに読む時に作り直していた(合成 0.3〜1.9 秒)。"""
        key = self._clip_key(text, lang, speed)
        if key in self._clips:
            return key
        best = None
        for k in self._clips:
            if k[:3] == key[:3] and abs(k[3] - key[3]) <= SPEED_TOL + 1e-9:
                if best is None or abs(k[3] - key[3]) < abs(best[3] - key[3]):
                    best = k
        return best

    def _take_clip(self, text: str, lang: str, speed: "float | None" = None) -> "_Clip | None":
        """作り置きを取り出す(速さが近ければそれ)。鳴らす速さは self.last_speed に残す。"""
        with self._clip_lock:
            k = self._near_key(text, lang, speed)
            clip = self._clips.pop(k, None) if k is not None else None
            if clip is not None and clip.guess:   # 今鳴らす: 予想の音でも捨てさせず先に作る
                clip.guess = False
                self._promote(clip)
        if clip is not None:
            self.last_speed = k[3]
        return clip

    def prepare(self, text: str, lang: str, speed: "float | None" = None,
                guess: bool = False, early: bool = False) -> None:
        """text の音を裏で作っておく(同じ文・同じ声なら say がそれを鳴らす)。

        作るのは1本の担当(_synth_loop)が順に。guess=True は予想訳(先読み)の音で、確定訳の音の後に回し、
        新しい予想が来たら まだ作り始めていない古い予想は作らない(11/08 speed: 外れた予想の合成 0.7+2.2秒が
        確定訳の声を待たせていた)。"""
        text = (text or "").strip()
        if not text or parse_voice(self.voices.get(lang, ""))[0] not in ("voicevox", "kokoro"):
            return
        with self._clip_lock:
            if guess:   # 新しい予想: まだ作っていない古い予想は捨てる
                for j in list(self._jobs):
                    if j[0].guess and not j[0].started:
                        self._drop(j[0])
            for c in self._chunks(text, lang):   # 長い訳は文ごと(Kokoro は読点ごとも。1つ目から順に作る)
                key = self._clip_key(c, lang, speed)
                near = self._near_key(c, lang, speed)
                if near is not None:
                    # 同じ文の作り置きがある(速さが近い: それを鳴らすので作らない)。確定訳なら先に作る
                    old = self._clips[near]
                    if not guess and old.guess:
                        old.guess = False
                        self._promote(old)
                    continue
                clip = _Clip(guess)
                self._clips[key] = clip
                self._enqueue((clip, c, lang, speed))
            while len(self._clips) > 8:          # 使われなかった作り置きは古い順に捨てる
                self._drop(self._clips.pop(next(iter(self._clips))))
            self._jobs_cv.notify_all()
            if self._worker is None or not self._worker.is_alive():
                self._worker = threading.Thread(target=self._synth_loop, daemon=True)
                self._worker.start()

    def _enqueue(self, job: tuple) -> None:
        """確定訳の音は予想訳の音より前へ(_clip_lock を持って呼ぶ)。"""
        if job[0].guess:
            self._jobs.append(job)
            return
        i = next((k for k, j in enumerate(self._jobs) if j[0].guess), len(self._jobs))
        self._jobs.insert(i, job)

    def _promote(self, clip: "_Clip") -> None:
        for k, j in enumerate(self._jobs):
            if j[0] is clip:
                self._enqueue(self._jobs.pop(k))
                return

    def _drop(self, clip: "_Clip") -> None:
        """もう要らない作り置き(作り始めていなければ作らない)。"""
        clip.dropped = True
        for k, j in enumerate(self._jobs):
            if j[0] is clip:
                self._jobs.pop(k)
                clip.done.set()
                break
        for k, v in list(self._clips.items()):
            if v is clip:
                del self._clips[k]

    def _synth_loop(self) -> None:
        """作り置きの担当(1本)。行列が空なら少し待って終わる(次の prepare で起き直す)。"""
        while True:
            with self._clip_lock:
                if not self._jobs:
                    self._jobs_cv.wait(5.0)
                if not self._jobs:
                    self._worker = None
                    return
                clip, c, lang, speed = self._jobs.pop(0)
                if clip.dropped:
                    clip.done.set()
                    continue
                clip.started = True
            def on_part(w, _c=clip):
                with _c.cv:
                    _c.parts.append(w)
                    _c.cv.notify_all()
            try:
                clip.wav = self._synth(c, lang, speed, on_part=on_part)
            except Exception:  # noqa: BLE001
                clip.wav = None
            finally:
                with clip.cv:
                    if not clip.parts and clip.wav:
                        clip.parts.append(clip.wav)
                    clip.done.set()
                    clip.cv.notify_all()

    def _play_clip(self, clip: "_Clip", timeout: float = 30.0) -> bool:
        """作り置きの音を、できた所から順に鳴らす(11/08 speed)。1つも鳴らせなければ False。"""
        i = 0
        t_end = time.time() + timeout
        while not self._stop.is_set():
            with clip.cv:
                while len(clip.parts) <= i and not clip.done.is_set() and time.time() < t_end:
                    clip.cv.wait(0.05)
                if len(clip.parts) <= i:
                    break
                part = clip.parts[i]
            self._play_wav(part)
            i += 1
        return i > 0

    def _synth(self, text: str, lang: str, speed: "float | None" = None,
               on_part=None) -> bytes | None:
        """1文を WAV バイト列にする。None = macOS say で読む(声の指定/エンジン失敗)。
        エンジンは同時に1本だけ(sherpa の生成を2スレッドで同時に呼ばない)。"""
        speed = self._speed(lang, speed)
        kind, val = parse_voice(self.voices.get(lang, ""))
        with self._synth_lock:
            if kind == "voicevox" and self.voicevox_url:
                try:
                    wav = self._voicevox_synth(text, int(val), speed)
                    if wav:
                        return wav
                except Exception as e:  # noqa: BLE001 - 不在/不調でも会話を止めない
                    print(f"[tts] voicevox 生成失敗、say にフォールバック: {e}")
            elif kind == "voicevox":
                print("[tts] voicevox_url 未設定、say を使用")
            if kind == "kokoro" and "kokoro" in self.engine:
                try:
                    cb = None
                    if on_part is not None:   # 読点ごとにできた音をすぐ渡す(できた所から鳴らす)
                        sr = int(self.engine["kokoro"].sample_rate)

                        def cb(x, _p, _sr=sr):
                            y = remove_tones(np.asarray(x, dtype=np.float32), _sr)
                            if len(y):
                                on_part(_to_wav(y, _sr))
                            return 1
                    audio = self._kokoro(text, lang, int(val), speed, callback=cb)
                    samples = remove_tones(audio.samples, audio.sample_rate)
                    if len(samples) > 0:
                        return _to_wav(samples, audio.sample_rate)
                except Exception as e:  # noqa: BLE001
                    print(f"[tts] {lang} kokoro 生成失敗: {e}")
            elif kind == "kokoro" and "vits_ko" in self.engine and lang == "ko":
                try:
                    audio = self.engine["vits_ko"].generate(text, sid=int(val), speed=speed)
                    samples = np.asarray(audio.samples, dtype=np.float32)
                    if len(samples) > 0:
                        return _to_wav(samples, audio.sample_rate)
                except Exception as e:  # noqa: BLE001
                    print(f"[tts] {lang} vits 生成失敗: {e}")
        return None

    def _kokoro(self, text: str, lang: str, sid: int, speed: float, callback=None):
        """英語は発音記号にしてから渡す(辞書に無い語の読み飛ばし対策、g2p_en.py)。
        失敗したら従来の文字入力に戻す。"""
        eng = self.engine["kokoro"]
        if lang == "en" and getattr(self, "g2p_en", None) is not None:
            try:
                import sherpa_onnx
                phs = self.g2p_en.sentences(text, clauses=True)
                if phs:
                    cfg = sherpa_onnx.GenerationConfig()
                    cfg.sid, cfg.speed = sid, speed
                    cfg.phoneme_codepoints = [[ord(c) for c in p] for p in phs]
                    audio = eng.generate("", cfg, callback) if callback else eng.generate("", cfg)
                    if len(audio.samples) > 0:
                        return audio
            except Exception as e:  # noqa: BLE001
                print(f"[tts] 発音記号での生成に失敗、文字入力で読みます: {e}")
        if callback is not None:
            return eng.generate(text, sid=sid, speed=speed, callback=callback)
        return eng.generate(text, sid=sid, speed=speed)

    def _play(self, samples: np.ndarray, rate: int) -> bool:
        """PCMをWAVにして afplay。sayと同じ再生経路なのでCPU/GPUが忙しくても切れない。"""
        try:
            return self._play_wav(_to_wav(samples, rate))
        except Exception:
            return False

    def _out_target(self) -> "tuple[str, str]":
        """鳴らす出力 (名前, CoreAudio の UID)。システムの既定・選んだものが無いときは ("", "")。"""
        name = getattr(self, "output_device", "") or ""
        if not name:
            return "", ""
        from . import devices
        try:
            d = devices.find(devices.OUTPUT, name, max_age=3.0)
        except Exception:  # noqa: BLE001
            d = None
        if d is None:
            if self._missing_noted != name:   # 同じ名前は1回だけ知らせる
                self._missing_noted = name
                print(f"[tts] 出力デバイス「{name}」が見つからないので、システムの既定で鳴らします", flush=True)
                cb = getattr(self, "on_device", None)
                if cb is not None:
                    try:
                        cb("missing", name, "")
                    except Exception:  # noqa: BLE001
                        pass
            return "", ""
        self._missing_noted = ""
        return name, d.get("uid") or ""

    def _play_on_device(self, path: str, uid: str) -> bool:
        """選んだ出力デバイスで WAV を鳴らす(AVAudioPlayer.currentDevice。afplay は出力を選べない)。
        鳴らせなければ False(呼ぶ側が afplay = システムの既定で鳴らす)。この Mac の音の取り込み
        (sysaudio)は Sokutsu の process ごと外すので、ここで鳴らした声も入らない。"""
        try:
            import objc
            from Foundation import NSURL
            cls = _av_player_class()
            if cls is None:
                return False
            with objc.autorelease_pool():
                r = cls.alloc().initWithContentsOfURL_error_(NSURL.fileURLWithPath_(path), None)
                p = r[0] if isinstance(r, tuple) else r
                if p is None:
                    return False
                p.setCurrentDevice_(uid)
                try:
                    p.setVolume_(self.volume)
                except Exception:
                    pass
                if not p.play():
                    return False
                self._player = p
                self._started()
                t0 = time.time()
                while p.isPlaying():
                    if self._stop.is_set():
                        p.stop()
                        break
                    time.sleep(0.03)
                self.last_play_s = getattr(self, "last_play_s", 0.0) + (time.time() - t0)
                return True
        except Exception as e:  # noqa: BLE001
            print(f"[tts] 選んだ出力で鳴らせません({type(e).__name__}: {e})、既定で鳴らします", flush=True)
            return False
        finally:
            self._player = None

    def _play_wav(self, data: bytes) -> bool:
        """WAVバイト列を一時ファイルに書いて鳴らす(選んだ出力デバイスがあればそこへ、無ければ afplay)。
        再生中は cancel() で止められる。"""
        import tempfile
        path = None
        try:
            fd, path = tempfile.mkstemp(suffix=".wav", prefix="tsuyaku_")
            with os.fdopen(fd, "wb") as f:
                f.write(data)
            _name, uid = self._out_target()
            if uid and self._play_on_device(path, uid):
                return True
            proc = subprocess.Popen(afplay_cmd(path, self.volume),
                                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            self._proc = proc
            self._started()
            t0 = time.time()
            while proc.poll() is None:
                if self._stop.is_set():
                    try:
                        proc.kill()
                    except Exception:
                        pass
                    break
                time.sleep(0.05)
            self.last_play_s = getattr(self, "last_play_s", 0.0) + (time.time() - t0)
            return True
        except Exception:
            return False
        finally:
            self._proc = None
            if path:
                try:
                    os.unlink(path)
                except OSError:
                    pass

    def _voicevox_synth(self, text: str, speaker: int, speed: float) -> bytes:
        """VOICEVOX ENGINE で1文合成してWAVバイト列を返す(失敗は例外)。

        /audio_query(読み上げパラメータ生成) → /synthesis(WAV) の2段。
        話速 speedScale は UI スライダーと同样 0.5〜2.0 に丸める。
        """
        import requests
        from urllib.parse import quote
        base = self.voicevox_url
        r = requests.post(f"{base}/audio_query?speaker={speaker}&text={quote(text)}",
                          headers={"Content-Type": "text/plain"}, data=b"",
                          timeout=(1.5, 30))
        r.raise_for_status()
        query = r.json()
        query["speedScale"] = max(0.5, min(2.0, float(speed)))
        r2 = requests.post(f"{base}/synthesis?speaker={speaker}", json=query,
                           headers={"Content-Type": "application/json"},
                           timeout=(1.5, 60))
        r2.raise_for_status()
        return r2.content

    def _say_macos(self, text: str, name: str, speed: float = 1.0) -> bool:
        proc = None
        try:
            dev, _uid = self._out_target()
            proc = subprocess.Popen(say_cmd(name, speed, device=dev or None) + [say_text(text, self.volume)],
                                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            self._proc = proc
            self._started()
            t0 = time.time()
            while proc.poll() is None:
                if self._stop.is_set():
                    proc.kill()
                    break
                time.sleep(0.05)
            if dev and getattr(proc, "returncode", 0) not in (0, None) and not self._stop.is_set():
                # say が名前のデバイスを見つけられない(「Found no Audio Output Device」で 1): 既定で読む
                print(f"[tts] say が出力「{dev}」を使えないので、システムの既定で読みます", flush=True)
                proc = subprocess.Popen(say_cmd(name, speed) + [say_text(text, self.volume)],
                                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                self._proc = proc
                while proc.poll() is None:
                    if self._stop.is_set():
                        proc.kill()
                        break
                    time.sleep(0.05)
            self.last_play_s = getattr(self, "last_play_s", 0.0) + (time.time() - t0)
            return True
        except Exception:
            return False
        finally:
            self._proc = None

    def _started(self) -> None:
        cb, self.on_start = self.on_start, None
        if cb is not None:
            try:
                cb()
            except Exception:  # noqa: BLE001
                pass

    def cancel(self):
        """現在の読み上げだけ中止する(エンジン自体はそのまま使える)。"""
        self._stop.set()
        pl = getattr(self, "_player", None)
        if pl is not None:
            try:
                pl.stop()
            except Exception:  # noqa: BLE001
                pass
        p = self._proc
        if p is not None and p.poll() is None:
            try:
                p.kill()
            except Exception:
                pass

    def stop(self):
        self._stop.set()
        self.cancel()
