"""英語の読み上げ用の発音記号化(11/06)。

問題: sherpa-onnx の Kokoro は英語を同梱の辞書(lexicon-us-en.txt)だけで
音にし、辞書に無い単語を黙って読み飛ばす。実際の会話で "That's" "conversations"
人名・地名(辞書に無い固有名詞)が消えた(「That's great.」が「great」になる)。
data_dir に espeak-ng-data を渡しても英語では使われない(実測)。

設計: 文章をこちらで発音記号(Kokoro の記号体系)にして渡す。
  1. 辞書にあれば辞書の発音(いちばん自然)
  2. 複数形・所有格の -s / 's は語幹の辞書発音 + z/s
  3. それ以外(固有名詞・数字・新語)は espeak-ng で発音記号にして、
     Kokoro の記号へ置き換える(misaki の espeak 代替と同じ対応表)
espeak-ng は venv の espeakng_loader 同梱のライブラリを ctypes で呼ぶ
(外部コマンド不要)。使えなければ 1・2 だけで動き、それでも無理な語は従来どおり飛ばす。
"""
from __future__ import annotations

import ctypes
import re
import threading
from pathlib import Path

_PUNCT = set(';:,.!?—…"()“”')
_TOKEN_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)*|\d+(?:[.,]\d+)*|[^\sA-Za-z\d]")
_SENT_END = set(".!?;")
_CLAUSE = set(",:")          # 長い文の読点(sentences(clauses=True))
CLAUSE_MIN_WORDS = 4

# espeak-ng(IPA, 結合記号 ^)→ Kokoro/misaki の記号。長いものから置き換える。
_E2M = sorted({
    "ʔˌn̩": "tᵊn", "ʔn̩": "tᵊn", "ʔn": "tᵊn", "ʔ": "t",
    "a^ɪ": "I", "a^ʊ": "W", "d^ʒ": "ʤ", "e^ɪ": "A", "e": "A", "t^ʃ": "ʧ",
    "o^ʊ": "O", "ɔ^ɪ": "Y", "ə^l": "ᵊl", "ɚ": "əɹ", "r": "ɹ", "x": "k",
    "ç": "k", "ɐ": "ə", "ɬ": "l", "ʲ": "", "̃": "", "̩": "",
}.items(), key=lambda kv: -len(kv[0]))

# 辞書の機能語は単独で読んだ形(a=エイ, the=ジー)なので、文中の弱い形に直す。
# live は形容詞(ライブ)の発音しか無く「I live in ...」がライブになる。
_OVERRIDE = {"a": "ə", "an": "ən", "the": "ðə", "live": "lˈɪv", "lives": "lˈɪvz"}

_VOICELESS_END = set("ptkfθ")
_SIBILANT_END = ("s", "z", "ʃ", "ʒ", "ʧ", "ʤ")


def _plural(ph: str) -> str:
    """語幹の発音に複数形/所有格の s を付ける(英語の規則どおり z/s/ᵻz)。"""
    if ph.endswith(_SIBILANT_END):
        return ph + "ᵻz"
    if ph and ph[-1] in _VOICELESS_END:
        return ph + "s"
    return ph + "z"


class _Espeak:
    """libespeak-ng を直接呼ぶ最小の音素化(1語ずつ、スレッドは錠で直列)。"""

    def __init__(self):
        import espeakng_loader as el
        lib = ctypes.CDLL(el.get_library_path())
        lib.espeak_Initialize.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_char_p, ctypes.c_int]
        lib.espeak_Initialize.restype = ctypes.c_int
        lib.espeak_SetVoiceByName.argtypes = [ctypes.c_char_p]
        lib.espeak_SetVoiceByName.restype = ctypes.c_int
        lib.espeak_TextToPhonemes.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_int, ctypes.c_int]
        lib.espeak_TextToPhonemes.restype = ctypes.c_char_p
        data = str(Path(el.get_data_path()).parent)
        if lib.espeak_Initialize(1, 0, data.encode(), 0) <= 0:  # 1=RETRIEVAL(音は出さない)
            raise RuntimeError("espeak_Initialize failed")
        if lib.espeak_SetVoiceByName(b"en-us") != 0:
            raise RuntimeError("espeak en-us voice missing")
        self.lib = lib
        self.lock = threading.Lock()

    def ipa(self, text: str) -> str:
        buf = ctypes.create_string_buffer(text.encode("utf-8"))
        ptr = ctypes.c_void_p(ctypes.addressof(buf))
        mode = 0x02 | 0x80 | (ord("^") << 8)   # IPA + 複数文字の音素を ^ で結ぶ
        out = []
        with self.lock:
            while ptr.value:
                r = self.lib.espeak_TextToPhonemes(ctypes.byref(ptr), 1, mode)
                if r:
                    out.append(r.decode("utf-8", "ignore"))
        return " ".join(s.strip() for s in out if s.strip())


def espeak_to_kokoro(ipa: str) -> str:
    ps = ipa
    for a, b in _E2M:
        ps = ps.replace(a, b)
    ps = ps.replace("ɜːɹ", "ɜɹ").replace("ɜː", "ɜɹ").replace("ɪə", "iə")
    return ps.replace("ː", "").replace("^", "")


class EnG2P:
    def __init__(self, lexicon_path: str | Path, vocab: set[str] | None = None,
                 use_espeak: bool = True):
        self.lex: dict[str, str] = {}
        with open(lexicon_path, encoding="utf-8") as f:
            for line in f:
                parts = line.rstrip("\n").split(" ")
                if len(parts) >= 2 and parts[0] and parts[0].lower() not in self.lex:
                    self.lex[parts[0].lower()] = "".join(parts[1:])
        self.vocab = vocab
        self.espeak = None
        if use_espeak:
            try:
                self.espeak = _Espeak()
            except Exception as e:  # noqa: BLE001 - 無ければ辞書だけで動く
                print(f"[tts] espeak-ng を使えないため辞書の単語だけ読みます: {e}")
        self._cache: dict[str, str | None] = {}

    def word(self, w: str) -> str | None:
        key = w.lower()
        if key in self._cache:
            return self._cache[key]
        ph = _OVERRIDE.get(key) or self.lex.get(key)
        if ph is None:
            for suf in ("'s", "s'", "s"):
                stem = key[: -len(suf)]
                if key.endswith(suf) and len(stem) >= 2 and stem in self.lex:
                    ph = _plural(self.lex[stem])
                    break
        if ph is None and self.espeak is not None:
            try:
                ph = espeak_to_kokoro(self.espeak.ipa(w)) or None
            except Exception:  # noqa: BLE001
                ph = None
        if ph is not None and self.vocab is not None:
            ph = "".join(c for c in ph if c in self.vocab or c == " ") or None
        self._cache[key] = ph
        return ph

    def missing(self, text: str) -> list[str]:
        """読めない語(テスト・ログ用)。"""
        return [t for t in _TOKEN_RE.findall(text.replace("’", "'"))
                if t[0].isalnum() and self.word(t) is None]

    def sentences(self, text: str, max_len: int = 400, clauses: bool = False) -> list[str]:
        """文ごとの発音記号列。語の間は空白、句読点は前の語にくっつける。

        clauses=True(11/08 speed): 長い文(CLAUSE_MIN_WORDS*3 語以上)は「,」「:」の所でも分ける
        (どちらの側も CLAUSE_MIN_WORDS 語以上のときだけ)。Kokoro は列ごとに作って callback で渡すので、
        最初の読点までの音ができた所で鳴らし始められる。文字で渡したとき(中国語)も sherpa は読点で分けて
        作るので、作り方は同じになる。"""
        text = text.replace("’", "'").replace("‘", "'")
        out: list[str] = []
        sent: list[str] = []      # 今の文の、読点ごとの列
        cur = ""

        def close_sentence():
            nonlocal sent
            if not sent:
                return
            if not clauses or sum(len(x.split()) for x in sent) < CLAUSE_MIN_WORDS * 3:
                out.append(" ".join(sent))
            else:
                merged: list[str] = []
                for x in sent:   # 短い頭(Yes, / However,)・短いしっぽは隣とまとめる
                    if merged and (len(merged[-1].split()) < CLAUSE_MIN_WORDS
                                   or len(x.split()) < CLAUSE_MIN_WORDS):
                        merged[-1] = f"{merged[-1]} {x}"
                    else:
                        merged.append(x)
                out.extend(merged)
            sent = []

        for tok in _TOKEN_RE.findall(text):
            if tok[0].isalnum():
                ph = self.word(tok)
                if ph:
                    cur = f"{cur} {ph}" if cur else ph
            elif tok in _PUNCT:
                if cur:
                    cur += tok
                if tok in _SENT_END or len(cur) > max_len:
                    if cur.strip():
                        sent.append(cur.strip())
                    cur = ""
                    close_sentence()
                elif tok in _CLAUSE and cur.strip():
                    sent.append(cur.strip())
                    cur = ""
        if cur.strip():
            sent.append(cur.strip())
        close_sentence()
        return out
