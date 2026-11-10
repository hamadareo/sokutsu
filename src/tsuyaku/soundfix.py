"""耳の文字の、用語の崩れを音の近さで直す(11/08 IT 用語 2回目)。

見つけた問題(eval/terms/it_testset.json、作った220文): 耳は日本語の中の製品名・カタカナ語を、
人によって揺れる読みのまま、または1〜2音崩して書く(クバネティス・クーベネティス、ロードバランス)。
韓国語は音節ごとに空けて書く(라우터펌미웨 어를)。用語集にちょうど同じ形が無いと、訳の手がかりも
付かず、訳は崩れた音のまま音訳する。

設計: 用語集のカタカナ読み(ja)・ハングル読み(ko)と、耳の文字の語の「音の距離」を比べ、
十分近く・ほかの候補と紛れないときだけ、その語の正しい表記に戻す。
  - ja: カタカナの並び(4字以上)をローマ字の音に直し(長音・促音は落とす。ヴ→ブ)、編集距離で比べる。
    頭の音が同じ・長さが近い・距離が 1(長い語は 2)以下・2番目の候補より近い。
    耳の文字がもう用語集の形そのもの(クバネティス)なら変えない(言った読みのまま)。
  - ko: ハングルを字母(初声・中声・終声)に分けて比べる。助詞を外した語幹で。音節ごとの空白は繋いで試す。
  - 一般の語を壊さないよう、ja は用語集のカタカナの形だけ(4字以下の短い形・文脈の要る形は候補にしない)、
    ko は3音節以上の形だけ。一般の文(FLEURS の参照文・耳の文字)で変わる文の数を見張る(scripts/audit_terms.py)。
"""
from __future__ import annotations

import re
import threading

# ---------------------------------------------------------------- カタカナ → 音

_K2 = {
    "キャ": "kya", "キュ": "kyu", "キョ": "kyo", "ギャ": "gya", "ギュ": "gyu", "ギョ": "gyo",
    "シャ": "sha", "シュ": "shu", "ショ": "sho", "シェ": "she", "ジャ": "ja", "ジュ": "ju", "ジョ": "jo",
    "ジェ": "je", "チャ": "cha", "チュ": "chu", "チョ": "cho", "チェ": "che", "ニャ": "nya", "ニュ": "nyu",
    "ニョ": "nyo", "ヒャ": "hya", "ヒュ": "hyu", "ヒョ": "hyo", "ビャ": "bya", "ビュ": "byu", "ビョ": "byo",
    "ピャ": "pya", "ピュ": "pyu", "ピョ": "pyo", "ミャ": "mya", "ミュ": "myu", "ミョ": "myo",
    "リャ": "rya", "リュ": "ryu", "リョ": "ryo", "ティ": "ti", "ディ": "di", "トゥ": "tu", "ドゥ": "du",
    "テュ": "tyu", "デュ": "dyu", "ファ": "fa", "フィ": "fi", "フェ": "fe", "フォ": "fo", "フュ": "fyu",
    "ウィ": "wi", "ウェ": "we", "ウォ": "wo", "ヴァ": "ba", "ヴィ": "bi", "ヴェ": "be", "ヴォ": "bo",
    "ツァ": "tsa", "ツィ": "tsi", "ツェ": "tse", "ツォ": "tso", "クァ": "kwa", "クィ": "kwi", "クェ": "kwe",
    "クォ": "kwo", "グァ": "gwa", "イェ": "ye", "スィ": "si", "ズィ": "zi",
}
_K1 = dict(zip(
    "アイウエオカキクケコガギグゲゴサシスセソザジズゼゾタチツテトダヂヅデドナニヌネノハヒフヘホバビブベボ"
    "パピプペポマミムメモヤユヨラリルレロワヲンヴァィゥェォャュョヮ",
    ["a", "i", "u", "e", "o", "ka", "ki", "ku", "ke", "ko", "ga", "gi", "gu", "ge", "go",
     "sa", "shi", "su", "se", "so", "za", "ji", "zu", "ze", "zo", "ta", "chi", "tsu", "te", "to",
     "da", "ji", "zu", "de", "do", "na", "ni", "nu", "ne", "no", "ha", "hi", "fu", "he", "ho",
     "ba", "bi", "bu", "be", "bo", "pa", "pi", "pu", "pe", "po", "ma", "mi", "mu", "me", "mo",
     "ya", "yu", "yo", "ra", "ri", "ru", "re", "ro", "wa", "o", "n", "bu",
     "a", "i", "u", "e", "o", "ya", "yu", "yo", "wa"]))


def kana_key(s: str) -> str:
    """カタカナ → 音の鍵(ローマ字。長音・促音・中黒・空白は落とす)。"""
    out = []
    i = 0
    while i < len(s):
        two = s[i:i + 2]
        if two in _K2:
            out.append(_K2[two])
            i += 2
            continue
        ch = s[i]
        if ch in _K1:
            out.append(_K1[ch])
        i += 1
    return "".join(out)


# ---------------------------------------------------------------- ハングル → 字母

def jamo(s: str) -> tuple:
    """ハングルの音節を (初声, 中声, 終声) の字母の列に(空白は落とす)。"""
    out = []
    for ch in s:
        c = ord(ch) - 0xAC00
        if 0 <= c < 11172:
            out += [c // 588, 19 + (c % 588) // 28]
            if c % 28:
                out.append(40 + c % 28)
    return tuple(out)


def _lev(a, b, cap: int) -> int:
    """編集距離(cap を超えたら cap+1)。"""
    if abs(len(a) - len(b)) > cap:
        return cap + 1
    prev = list(range(len(b) + 1))
    for i in range(1, len(a) + 1):
        cur = [i] + [0] * len(b)
        lo = cur[0]
        ai = a[i - 1]
        for j in range(1, len(b) + 1):
            v = prev[j - 1] + (ai != b[j - 1])
            if prev[j] + 1 < v:
                v = prev[j] + 1
            if cur[j - 1] + 1 < v:
                v = cur[j - 1] + 1
            cur[j] = v
            if v < lo:
                lo = v
        if lo > cap:
            return cap + 1
        prev = cur
    return prev[-1]


def _cap(n: int) -> int:
    """鍵の長さ n で許す距離(ja はローマ字の字数、ko は字母の数)。"""
    if n < 7:
        return 0
    if n < 13:
        return 1
    return 2


# ---------------------------------------------------------------- 索引

_LATIN = re.compile(r"[A-Za-z]")

def _common() -> tuple:
    """一般の語の一覧(data/common_words.json、作った一覧)。ここにある語は書き換えない。"""
    import json
    from pathlib import Path
    try:
        d = json.loads((Path(__file__).resolve().parent / "data" / "common_words.json").read_text(encoding="utf-8"))
        return set(d.get("ja", ())), set(d.get("ko", ()))
    except (OSError, ValueError):
        return set(), set()


class SoundIndex:
    """3段で寄せる。
    1) 音の鍵がちょうど同じ(長音・促音・ヴ の揺れだけ違う: スルプット = スループット)→ どの語でも
    2) 鍵の距離が 1〜2: 製品名(英字で書く語)と 6字以上のカタカナ語 / 4音節以上のハングル語
    3) 頭の音だけ違う(ペンソルフロー = テンソルフロー): 製品名で鍵が9字以上
    一般の語の一覧(common_words.json)と用語集の形そのものは書き換えない。"""

    def __init__(self, entries: list):
        self.common_ja, self.common_ko = _common()
        self.ja_known: set = set()        # 用語集のカタカナの形(そのままなら変えない)
        self.ja_exact: dict = {}          # 鍵 → {正しい表記}
        self.ja: dict = {}                # 頭の字 → [(鍵, 形, 正しい表記)]
        self.ja_prod: dict = {}           # 鍵の長さ → [(鍵, 形, 正しい表記)](頭の音だけ違う形を探す)
        self.ko_known: set = set()
        self.ko_exact: dict = {}
        self.ko: dict = {}
        for e in entries:
            ctx_all = bool(e.get("ctx"))
            skip_ja = set((e.get("x") or {}).get("ja", ())) | set((e.get("c") or {}).get("ja", ()))
            skip_ko = set((e.get("x") or {}).get("ko", ())) | set((e.get("c") or {}).get("ko", ()))
            ja_forms = e.get("ja") or []
            ko_forms = e.get("ko") or []
            # 音の距離で寄せる先は製品名(英字で書く語)と長い語だけ。短い普通のカタカナ語・外来語は、
            # 耳が崩すより、近い別の語(プログラム と プログラマ、여행 と 이행)である方が多い
            product = bool(e.get("keep")) or bool(_LATIN.search((e.get("en") or [""])[0])
                                                  and _LATIN.search(ja_forms[0] if ja_forms else ""))
            for f in ja_forms:
                if not re.fullmatch(r"[ァ-ヺー]+", f):
                    continue
                self.ja_known.add(f)
                if ctx_all or f in skip_ja or len(f) < 4:
                    continue
                k = kana_key(f)
                # 直す先: 製品名は正しい表記(英字)、カタカナ語は近かった形そのもの(第一表記に
                # 「ステージング環境」のような長い形があっても語を足さない)
                to = ja_forms[0] if product else f
                self.ja_exact.setdefault(k, set()).add(to)
                if len(f) >= 5 and len(k) >= 7 and (product or len(f) >= 6):
                    self.ja.setdefault(k[0], []).append((k, f, to, product))
                    if product and len(k) >= 9:
                        self.ja_prod.setdefault(len(k), []).append((k, f, to))
            product_ko = bool(e.get("keep")) or bool(_LATIN.search(ko_forms[0] if ko_forms else ""))
            for f in ko_forms:
                g = f.replace(" ", "")
                if not re.fullmatch(r"[가-힣]+", g):
                    continue
                self.ko_known.add(g)
                if ctx_all or f in skip_ko or len(g) < 3:
                    continue
                k = jamo(g)
                to = ko_forms[0] if product_ko else f
                self.ko_exact.setdefault(k, set()).add(to)
                if len(k) >= 7 and (product_ko or len(g) >= 4):
                    self.ko.setdefault(k[0], []).append((k, f, to, product_ko))

    @staticmethod
    def _best(key, cands: list, loan_ok: bool = True) -> tuple | None:
        """一番近い候補(距離が許す範囲で、2番目と紛れない)→ (距離, 形, 正しい表記)。
        loan_ok = False なら製品名だけに寄せる(耳の語が短いとき)。"""
        cap = _cap(len(key))
        if cap == 0:
            return None
        best: list = []
        for k, f, canon, product in cands:
            if not (loan_ok or product):
                continue
            if abs(len(k) - len(key)) > cap:
                continue
            d = _lev(key, k, cap)
            if d <= cap:
                best.append((d, f, canon))
        if not best:
            return None
        best.sort(key=lambda x: x[0])
        top = best[0]
        # 2番目の候補が同じ距離で別の語なら紛れる
        if any(b[0] == top[0] and b[2] != top[2] for b in best[1:]):
            return None
        return top

    def fix_ja_word(self, w: str) -> str | None:
        if len(w) < 4 or w in self.ja_known or w in self.common_ja:
            return None
        k = kana_key(w)
        ex = self.ja_exact.get(k)
        if ex:
            return next(iter(ex)) if len(ex) == 1 else None
        if len(w) < 5 or len(k) < 7:
            return None
        # カタカナ語に寄せるのは耳の語も6字以上のとき(ステイング → ステージング のような取り違えを避ける)
        got = self._best(k, self.ja.get(k[0], []), loan_ok=len(w) >= 6)
        if got:
            return got[2]
        if len(k) >= 9:
            hits = {c for kk, _f, c in self.ja_prod.get(len(k), ()) if kk[1:] == k[1:]}
            if len(hits) == 1:
                return next(iter(hits))
        return None

    def fix_ko_stem(self, w: str) -> str | None:
        if len(w) < 3 or w in self.ko_known or w in self.common_ko:
            return None
        k = jamo(w)
        ex = self.ko_exact.get(k)
        if ex:
            return next(iter(ex)) if len(ex) == 1 else None
        if len(k) < 7:
            return None
        got = self._best(k, self.ko.get(k[0], []))
        return got[2] if got else None


_IDX: SoundIndex | None = None
_LOCK = threading.Lock()


def index() -> SoundIndex:
    global _IDX
    with _LOCK:
        if _IDX is None:
            from .terms import termbase
            _IDX = SoundIndex(termbase().entries)
        return _IDX


_KATA_RUN = re.compile(r"[ァ-ヺー]{4,}")


def fix_ja(text: str) -> str:
    """日本語の文の中の崩れたカタカナの語 → 用語集の正しい表記。"""
    if not text or not _KATA_RUN.search(text):
        return text
    idx = index()

    def rep(m: re.Match) -> str:
        w = m.group(0)
        got = idx.fix_ja_word(w)
        if got:
            return got
        # 語の後ろか前が用語集の形(クバネティスクラスタ = 崩れた語 + 知っている語)
        for j in range(len(w) - 2, 4, -1):
            if w[j:] in idx.ja_known:
                got = idx.fix_ja_word(w[:j])
                if got:
                    return got + w[j:]
        for j in range(3, len(w) - 4):
            if w[:j] in idx.ja_known:
                got = idx.fix_ja_word(w[j:])
                if got:
                    return w[:j] + got
        return w
    return _KATA_RUN.sub(rep, text)


_KO_PART = ("", "을", "를", "이", "가", "은", "는", "와", "과", "의", "에", "에서", "로", "으로", "도", "만",
            "에는", "에도", "으로는", "로는", "이나", "나", "랑", "이랑", "하고", "까지", "부터", "처럼", "보다",
            "라는", "이라는", "란", "인", "이다", "입니다", "이고", "고", "에게", "한테", "의")
_KO_TOK = re.compile(r"[가-힣]+")


def _ko_try(word: str, idx: SoundIndex, joined: bool = False) -> str | None:
    """1語(助詞つき)を直す。直せなければ None。joined = 空白で切れた音節を繋いだ語:
    繋いだ語幹が用語集の形そのものなら繋ぐ(音の近さで寄せるのは1語のときだけ。繋いで寄せると
    「여행 계획」→「이행 계획」のように別の語の並びを取り違える)。"""
    for p in sorted(set(_KO_PART), key=len, reverse=True):
        if p and not word.endswith(p):
            continue
        stem = word[:len(word) - len(p)] if p else word
        if len(stem) < 3:
            continue
        if stem in idx.ko_known:
            return word if joined else None
        if joined:
            continue
        got = idx.fix_ko_stem(stem)
        if got:
            return got + p
    return None


def fix_ko(text: str) -> str:
    """韓国語の文の中の崩れたハングルの語 → 用語集の正しい表記。空白で切れた音節は繋いで試す。"""
    if not text or not _KO_TOK.search(text):
        return text
    idx = index()
    toks = text.split(" ")
    out: list = []
    i = 0
    while i < len(toks):
        done = False
        # 長い繋ぎから(6語 → 1語)。繋ぐのは、繋いだ語幹が用語集の語に近いときだけ
        for n in range(min(6, len(toks) - i), 0, -1):
            seg = toks[i:i + n]
            if not all(_KO_TOK.fullmatch(t) for t in seg):
                continue
            # 繋ぐのは音節ごとに切れた並び(最後の語(助詞つき)の前は2音節まで)。「컴퓨터 과학」は繋がない
            if n > 1 and any(len(t) > 2 for t in seg[:-1]):
                continue
            joined = "".join(seg)
            got = _ko_try(joined, idx, joined=n > 1)
            if got:
                out.append(got)
                i += n
                done = True
                break
        if not done:
            out.append(toks[i])
            i += 1
    return " ".join(out)


def fix(text: str, lang: str) -> str:
    """耳の文字の用語の崩れを直す(ja / ko だけ。ほかはそのまま)。"""
    try:
        if lang == "ja":
            return fix_ja(text)
        if lang == "ko":
            return fix_ko(text)
    except Exception:  # noqa: BLE001 - 直しに失敗しても耳の文字のまま
        return text
    return text
