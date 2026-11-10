"""専門用語・略語に強くする(11/08)。

見つけた問題(11/08 ユーザーの日中の会話のログ): 耳が英字の略語を字名のカタカナで書き
(「ジーピーユー」のような並び)、訳(LLM)がそれを意味の無い音訳にした。略語のまま書けた行は
正しく訳せていた。1字違いの似た語は、前の文の訳を写すことがあった。

1. normalize_letters: 字名(エー・ビー・シー… / 에이・비・씨…)の並びを英字の略語に戻す。
   本物の語(アイ=愛、ハワイ、アイディア、아이=子ども)を壊さないよう、
   - 日本語: カタカナの語の全体が字名だけで2字以上 → 英字(同じ字の繰り返し「エーエー」は除く)。
     語の一部(前か後ろがカタカナの語に続く)なら、知っている略語に全部分けられるときだけ。
     「エ」「ティ」「ディ」のような短い形は、3字以上で知っている略語に全部分けられるときだけ。
   - 韓国語: 知っている略語に全部分けられ、後ろが無いか助詞のときだけ。
2. glossary_pairs / glossary_hint: 内蔵の用語集(data/termbase.json、ja/en/zh/ko)と利用者の
   用語辞書から、今の発話に出てくる語だけを「この訳を使う」として訳の依頼に足す(短く)。
   利用者の「語=訳」が内蔵より優先。英字の語(製品名など)はそのまま残すように伝える。
"""
from __future__ import annotations

import json
import re
import threading
from pathlib import Path

DATA = Path(__file__).resolve().parent / "data" / "termbase.json"

# ---------------------------------------------------------------- 字名 → 英字

_JA_STRONG = {
    "A": ("エー", "エイ"), "B": ("ビー",), "C": ("シー",), "D": ("ディー",), "E": ("イー",),
    "F": ("エフ",), "G": ("ジー",), "H": ("エイチ", "エッチ"), "I": ("アイ",),
    "J": ("ジェー", "ジェイ"), "K": ("ケー", "ケイ"), "L": ("エル",), "M": ("エム",),
    "N": ("エヌ",), "O": ("オー",), "P": ("ピー",), "Q": ("キュー",), "R": ("アール",),
    "S": ("エス",), "T": ("ティー",), "U": ("ユー",), "V": ("ブイ", "ヴィー"),
    "W": ("ダブリュー", "ダブル"), "X": ("エックス",), "Y": ("ワイ",), "Z": ("ゼット", "ゼッド", "ズィー"),
}
# 弱い形(短い・聞き違えやすい形。知っている略語に全部分けられる3字以上のときだけ)。
# ティ/ディ は耳がよく取り違える(T を「ディー」と書く)ので T にも D にも読む
_JA_WEAK = {"A": ("エ",), "O": ("オ",), "D": ("ディ",), "T": ("ティ", "ディ", "ディー"), "P": ("ピ",), "B": ("ビ",),
            "C": ("シ",), "Q": ("キュ",)}
_KO_STRONG = {
    "A": ("에이",), "B": ("비",), "C": ("씨",), "D": ("디",), "E": ("이",), "F": ("에프",),
    "G": ("지",), "H": ("에이치",), "I": ("아이",), "J": ("제이",), "K": ("케이",), "L": ("엘",),
    "M": ("엠",), "N": ("엔",), "O": ("오",), "P": ("피",), "Q": ("큐",), "R": ("알",),
    "S": ("에스",), "T": ("티",), "U": ("유",), "V": ("브이",), "W": ("더블유",), "X": ("엑스",),
    "Y": ("와이",), "Z": ("제트", "지"),
}
# 字名と同じ綴りの韓国語の語(아이티 = ハイチ。FLEURS ko 270文で2件の誤り)
_KO_STOP = {"아이티"}
_KO_PARTICLES = ("", "을", "를", "이", "가", "은", "는", "와", "과", "의", "에", "에서", "로", "으로",
                 "도", "만", "랑", "이랑", "하고", "에게", "까지", "부터", "처럼", "보다", "라는", "란",
                 "이나", "나", "인", "이다", "입니다", "예요", "이에요")

# 字名で読まれる略語(内蔵の用語集の英字の語にも足す)。大文字小文字は表記どおりに戻す。
_ACRONYMS = (
    "AI", "IoT", "AIoT", "IT", "ICT", "PC", "CPU", "GPU", "NPU", "TPU", "API", "SDK", "OS", "DB",
    "SQL", "LLM", "NLP", "CNN", "RNN", "GAN", "ML", "DX", "UI", "UX", "AR", "VR", "XR", "MR",
    "IP", "TCP", "UDP", "HTTP", "HTTPS", "URL", "DNS", "VPN", "LAN", "WAN", "SSD", "HDD", "USB",
    "LED", "PCB", "IC", "CAD", "CAE", "PWM", "ADC", "DAC", "MCU", "FPGA", "ASIC", "ROS", "SLAM",
    "GPS", "IMU", "KPI", "ROI", "OKR", "CRM", "ERP", "B2B", "B2C", "CEO", "CTO", "CFO", "HR", "PR",
    "PM", "QA", "FAQ", "PDF", "DNA", "RNA", "PCR", "MRI", "CT", "ICU", "ECG", "EV", "SNS", "TV",
    "CM", "ID", "PDCA", "OJT", "DIY", "AWS", "GCP", "CI", "CD", "QR", "OCR", "RPA", "SaaS", "PaaS",
    "IaaS", "BLE", "NFC", "RFID", "OK", "NG", "AGI", "RAG", "GPT", "BERT", "HPC", "SoC", "IEEE",
)


_CORE = frozenset(a.upper() for a in _ACRONYMS)


def _known_acronyms() -> dict:
    """大文字 → 表記(IOT → IoT)。内蔵の用語集の英字の語も含める。"""
    out = {a.upper(): a for a in _ACRONYMS}
    try:
        for e in termbase().entries:
            for f in e.get("en", []) + e.get("ja", []):
                if re.fullmatch(r"[A-Za-z][A-Za-z0-9]{1,5}", f) and sum(c.isupper() for c in f) >= 2:
                    out.setdefault(f.upper(), f)
    except Exception:  # noqa: BLE001 - 用語集が読めなくても字名の直しは動く
        pass
    return out


_KNOWN: dict | None = None


def _known() -> dict:
    global _KNOWN
    if _KNOWN is None:
        _KNOWN = _known_acronyms()
    return _KNOWN


def _names(table: dict) -> list:
    """(読み, 字) を長い読みから(エイチ が エイ より先)。"""
    return sorted(((r, ch) for ch, rs in table.items() for r in rs), key=lambda x: -len(x[0]))


_JA_S, _JA_W = _names(_JA_STRONG), _names(_JA_WEAK)
_KO_S = _names(_KO_STRONG)


def _parses(s: str, strong: list, weak: list) -> list:
    """s 全体を字名に分ける分け方(字の列, 弱い形を使ったか)をすべて返す(短い語なので総当たり)。"""
    out: list = []

    def rec(i: int, acc: str, used_weak: bool) -> None:
        if len(out) > 32:
            return
        if i == len(s):
            out.append((acc, used_weak))
            return
        for table, w in ((strong, False), (weak, True)):
            for r, ch in table:
                if s.startswith(r, i):
                    rec(i + len(r), acc + ch, used_weak or w)
    rec(0, "", False)
    return out


def _split_known(letters: str) -> list | None:
    """字の列を知っている略語に全部分ける(少ない数で)。分けられなければ None。"""
    known = _known()
    best: dict = {0: []}
    for i in range(len(letters)):
        if i not in best:
            continue
        for j in range(i + 2, min(len(letters), i + 6) + 1):
            k = letters[i:j]
            if k in known and (j not in best or len(best[j]) > len(best[i]) + 1):
                best[j] = best[i] + [known[k]]
    return best.get(len(letters))


def _near_known(letters: str) -> str | None:
    """1字だけ違う(置き換え・抜け・余分)知っている略語(3字以上・頭の字が同じ・候補が1つだけ)。
    耳の聞き違い(ジービーユー = GBU → GPU、ユーエビー = UAB → USB、エイオーティー = AOT → AIoT)。"""
    if len(letters) < 3:
        return None
    def one_edit(k: str) -> bool:
        if len(k) == len(letters):
            return sum(a != b for a, b in zip(k, letters)) == 1
        if abs(len(k) - len(letters)) != 1:
            return False
        short, long_ = (k, letters) if len(k) < len(letters) else (letters, k)
        return any(long_[:i] + long_[i + 1:] == short for i in range(len(long_)))
    hits = [v for k, v in _known().items() if len(k) >= 3 and k[0] == letters[0] and one_edit(k)]
    if len(hits) > 1:
        # 用語集が大きくなると候補が増える(GBU → GPU・GRU・GNU)。よく字名で読まれる略語が1つだけならそれ
        core = [v for v in hits if v.upper() in _CORE]
        hits = core if len(core) == 1 else hits
    return hits[0] if len(hits) == 1 else None


def _convert(sub: str, strong: list, weak: list, whole: bool, sep: str) -> str | None:
    """字名の並び sub → 英字。whole = カタカナの語の全体か。だめなら None。"""
    parses = sorted(_parses(sub, strong, weak), key=lambda p: p[1])
    if whole:
        # 語の全体が字名で、知っている略語と1字違い(耳の聞き違い)なら、その略語に
        for letters, used_weak in parses:
            if len(letters) >= 3 and not _split_known(letters):
                near = _near_known(letters)
                if near:
                    return near
    for letters, used_weak in parses:
        if len(letters) < 2:
            continue
        parts = _split_known(letters)
        if parts:
            if used_weak and len(letters) < 3:
                continue
            return sep.join(parts)
        if used_weak or not whole:
            continue
        if len(set(letters)) == 1:        # 「エーエー」「ワイワイ」= 言いよどみ・擬音
            continue
        return letters
    return None


_KATA = re.compile(r"[ァ-ヺー]+")


def _ja(text: str) -> str:
    def fix(m: re.Match) -> str:
        w = m.group(0)
        if len(w) < 3:
            return w
        got = _convert(w, _JA_S, _JA_W, True, "・")
        if got:
            return got
        # 語の後ろ(ロボット|エーアイ)/ 前(エーアイ|スピーカー): 知っている略語だけ
        for i in range(1, len(w) - 2):
            got = _convert(w[i:], _JA_S, _JA_W, False, "・")
            if got:
                return w[:i] + got
        for j in range(len(w) - 1, 2, -1):
            got = _convert(w[:j], _JA_S, (), False, "・")
            if got:
                return got + w[j:]
        return w
    return _KATA.sub(fix, text)


_HANGUL = re.compile(r"[가-힣]+")


def _ko(text: str) -> str:
    def fix(m: re.Match) -> str:
        w = m.group(0)
        start = m.start()
        if start > 0 and text[start - 1].isalnum():
            return w
        for j in range(len(w), 1, -1):
            if w[j:] not in _KO_PARTICLES or w[:j] in _KO_STOP:
                continue
            for letters, _ in _parses(w[:j], _KO_S, ()):
                if len(letters) >= 2:
                    parts = _split_known(letters)
                    if parts:
                        return " ".join(parts) + w[j:]
        return w
    return _HANGUL.sub(fix, text)


_KO_SPACED = re.compile(r"(?<![가-힣A-Za-z0-9])((?:[가-힣]{1,3} ){1,4}[가-힣]{1,4})")


def _ko_spaced(text: str) -> str:
    """このMacの耳(SenseVoice)は韓国語を音節ごとに空けて書く(「유 에스 비」)。空白で分かれた
    字名を繋いで、知っている略語(3字以上)になるときだけ英字にする。"""
    def fix(m: re.Match) -> str:
        toks = m.group(1).split(" ")
        for a in range(len(toks)):
            for b in range(len(toks), a + 1, -1):
                joined = "".join(toks[a:b])
                for j in range(len(joined), 1, -1):
                    if joined[j:] not in _KO_PARTICLES or joined[:j] in _KO_STOP:
                        continue
                    for letters, _ in _parses(joined[:j], _KO_S, ()):
                        if len(letters) >= 3:
                            parts = _split_known(letters)
                            if parts:
                                out = toks[:a] + [" ".join(parts) + joined[j:]] + toks[b:]
                                return " ".join(out)
        return m.group(1)
    return _KO_SPACED.sub(fix, text)


def normalize_letters(text: str, lang: str) -> str:
    """字名の並びを英字の略語に戻す(ja / ko だけ。ほかの言語はそのまま)。"""
    if not text:
        return text
    if lang == "ja":
        return _ja(text)
    if lang == "ko":
        return _ko(_ko_spaced(text))
    return text


# ---------------------------------------------------------------- 用語集

# IT の分野(用語集の "d")。IT の話かどうかの見当と、テーマで「IT」を選んだときの優先に使う
IT_DOMAINS = frozenset({
    "ai", "cs", "iot", "robotics", "electronics", "net", "security", "cloud", "general_tech",
    "container", "devops", "network", "data", "database", "programming", "web", "os", "itbiz",
    "semiconductor", "hardware", "devprocess",
})


def _case_sensitive(form: str) -> bool:
    """英字の形を大文字小文字どおりに探すか。短い大文字の略語(IT, US)と、2字目以降に大文字がある形
    (LoRA と LoRa、IoC)は区別する。ほか(machine learning、Kubernetes)は区別しない。"""
    if not _LATIN.search(form):
        return False
    if form.isupper() and len(form) <= 4:
        return True
    return bool(re.search(r".[A-Z]", form)) and not form.isupper()


class TermBase:
    """用語集の索引。語が増えても(3800語 × 4言語)訳の依頼の準備が数 ms で済むよう、
    形の頭の2字(英字は小文字)で引く。1回の発話は数十字なので、字の位置ごとに引けば速い。

    entries の1語: {"d": 分野, "ja"/"en"/"zh"/"ko": [第一表記(訳す先), 表記ゆれ・読み・略称…],
      "keep": 英字のまま書く語, "x": {言語: [原文の形としては使わない形]}(意味が割れる形),
      "c": {言語: [IT の話の中だけで使う形]}(一般の文で別の意味になりやすい短い形・普通の英単語)、
      "ctx": すべての形を IT の話の中だけで使う(process = 過程 / プロセス、server、memory)、
      "noev": 一般のニュースにも出る IT の語(AI、website)。IT の話の証拠には数えない}"""

    def __init__(self, entries: list):
        self.entries = entries
        self.index: dict = {}
        for lang in ("ja", "en", "zh", "ko"):
            idx: dict = {}
            for i, e in enumerate(entries):
                skip = set((e.get("x") or {}).get(lang, ()))
                ctx = set(e.get(lang, ())) if e.get("ctx") else set((e.get("c") or {}).get(lang, ()))
                it = e.get("d") in IT_DOMAINS and not e.get("noev")   # noev = 一般のニュースにも出る(AI、website)
                for f in e.get(lang, []):
                    if not f or len(f) < 2 or f in skip:
                        continue
                    cs = _case_sensitive(f)
                    idx.setdefault(f[:2].lower(), []).append((f, i, cs, f in ctx, it))
            for v in idx.values():
                v.sort(key=lambda x: -len(x[0]))
            self.index[lang] = idx

    def matches(self, text: str, lang: str) -> list:
        """text の中の用語集の形 [(始め, 終わり, 形, 何番目の語, 文脈が要るか, IT か)](重なりあり)。"""
        idx = self.index.get(lang) or {}
        if not text or not idx:
            return []
        low = text.lower()
        if len(low) != len(text):
            low = "".join(ch.lower() if len(ch.lower()) == 1 else ch for ch in text)
        out = []
        for i in range(len(text) - 1):
            cands = idx.get(low[i:i + 2])
            if not cands:
                continue
            for f, k, cs, ctx, it in cands:
                j = i + len(f)
                if j > len(text):
                    continue
                if (text[i:j] != f) if cs else (low[i:j] != f.lower()):
                    continue
                if _boundary_ok(text, f, i, j):
                    out.append((i, j, f, k, ctx, it))
        return out


_TB: TermBase | None = None
_TB_LOCK = threading.Lock()


def termbase() -> TermBase:
    global _TB
    with _TB_LOCK:
        if _TB is None:
            try:
                data = json.loads(DATA.read_text(encoding="utf-8"))
                _TB = TermBase(list(data.get("entries", [])))
            except (OSError, ValueError):
                _TB = TermBase([])
        return _TB


_LATIN = re.compile(r"[A-Za-z]")
_HAN = re.compile(r"[一-鿿]")
# 漢字の語の後ろに付いて別の語にする字(用語集に長い方の語が無いときは手がかりを出さない)
_HAN_SUFFIX = set("器式性化机機者率量法型")
_KATA_CH = re.compile(r"[ァ-ヺー]")


def _boundary_ok(text: str, form: str, i: int, j: int) -> bool:
    """text[i:j] の form が語の途中でないか(英字の語の切れ目、カタカナの語の続き、漢字の後ろに付く字)。"""
    if _LATIN.search(form):
        if not ((i == 0 or not text[i - 1].isalnum() or not text[i - 1].isascii())
                and (j == len(text) or not text[j].isalnum() or not text[j].isascii())):
            return False
    if _KATA_CH.match(form[0]) and i > 0 and _KATA_CH.match(text[i - 1]):
        return False         # フォトグラフ の グラフ
    if _KATA_CH.match(form[-1]) and j < len(text) and _KATA_CH.match(text[j]):
        return False         # グラフィック の グラフ
    if _HAN.match(form[-1]) and j < len(text) and text[j] in _HAN_SUFFIX:
        return False         # 嵌入式 の 嵌入、负载均衡器 の 负载均衡(後ろに付く字で別の語になる)
    return True


def _find(text: str, form: str, taken: list) -> int:
    """text の中の form の位置(重なり・語の途中は飛ばす)。無ければ -1。"""
    latin = bool(_LATIN.search(form))
    # 英字: 短い大文字の略語(IT, US)は大文字小文字を区別、ほかは区別しない
    hay, needle = text, form
    if latin and not (form.isupper() and len(form) <= 4):
        hay, needle = text.lower(), form.lower()
    i = hay.find(needle)
    while i >= 0:
        j = i + len(needle)
        if not any(a < j and i < b for a, b in taken) and _boundary_ok(text, form, i, j):
            return i
        i = hay.find(needle, i + 1)
    return -1


def _user_pairs(user) -> list:
    """利用者の用語辞書・テーマ → [(元の語, 訳 or None)]。「語=訳」「語→訳」は訳つき。"""
    out = []
    for t in user or ():
        t = str(t or "").strip()
        if not t:
            continue
        m = re.split(r"\s*(?:=|→|->|＝)\s*", t, maxsplit=1)
        if len(m) == 2 and m[0] and m[1]:
            out.append((m[0], m[1]))
        else:
            out.append((t, None))
    return out


def source_forms(user) -> tuple:
    """用語辞書の項目の元の語だけ(「グラフ=图表」→「グラフ」)。耳の手がかり用。"""
    return tuple(s for s, _t in _user_pairs(user))


def _fits(t: str, tgt: str) -> bool:
    """利用者の訳 t が訳す先の言語 tgt 向けか(文字の種類で見る)。"""
    if re.search(r"[ぁ-ゖァ-ヺ]", t):
        return tgt == "ja"
    if re.search(r"[가-힣]", t):
        return tgt == "ko"
    if re.search(r"[一-鿿]", t):
        return tgt in ("ja", "zh")
    if re.fullmatch(r"[A-Z0-9][A-Za-z0-9 .+\-_/]*", t) and not re.search(r"[a-z]{3}", t):
        return True                # 略語・型番はどの言語でもそのまま
    return tgt == "en"


_LATIN_TOK = re.compile(r"[A-Za-z][A-Za-z0-9]*(?:[/.][A-Za-z0-9]+)*")


def _keep_latin(form: str, cands: list, en: bool = False) -> str | None:
    """訳の表から使う訳を選ぶ(英語以外の原文)。原文の形に英字の語(2字以上: S3・ArF・Pub/Sub)があれば、それを全部残す
    訳だけ(11/08 リモートサーバーの訳で「S3バケット = 存储桶」「ArF液浸 = 浸没式光刻」が英字を消した)。無ければ None。"""
    toks = [t.lower() for t in _LATIN_TOK.findall(form) if len(t) >= 2
            and (not en or re.search(r"[A-Z0-9]", t))]   # 英語の原文では大文字・数字のある語(S3・ArF)だけ
    if not toks:
        return cands[0]
    for c in cands:
        low = c.lower()
        if all(t in low for t in toks):
            return c
    return None


def is_it_context(text: str, src: str, domain: str | None = None, context: str = "",
                  found: list | None = None) -> bool:
    """IT の話か: テーマで IT を選んだ、または今の発話か直前の発話(context)に、一般の文では出ない
    IT の語(製品名・略語・長い用語 = 文脈の要らない形)がある。"""
    if domain == "it":
        return True
    tb = termbase()
    if any(m[5] and not m[4] for m in (found if found is not None else tb.matches(text, src))):
        return True
    if context:
        for lang in ((src,) if src else ()) + tuple(x for x in ("ja", "en", "zh", "ko") if x != src):
            if any(m[5] and not m[4] for m in tb.matches(context, lang)):
                return True
    return False


def glossary_pairs(text: str, src: str, tgt: str, user=(), limit: int = 8, domain: str | None = None,
                   context: str = "") -> list:
    """今の発話に出てくる語の [(元の形, 使う訳)]。利用者の語が先・長い形が先・重ならない。
    domain = "it"(テーマで IT を選んだ)なら IT の語を先に、文脈の要る形も使う。
    context = 直前の発話(IT の話かどうかの見当に使う)。"""
    if not text or not src or not tgt or src == tgt:
        return []
    taken: list = []
    out: list = []
    seen: set = set()
    for s, t in sorted(_user_pairs(user), key=lambda p: -len(p[0])):
        if len(out) >= limit:
            break
        i = _find(text, s, taken)
        if i < 0:
            continue
        if t is None:
            # 訳の無い語: 英字の語(製品名・略語)だけ「そのまま」と伝える
            if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9 .+\-_/]*", s):
                continue
            t = s
        elif not _fits(t, tgt):
            continue           # 別の言語向けの訳
        taken.append((i, i + len(s)))
        if s.lower() not in seen:
            seen.add(s.lower())
            out.append((s, t))
    tb = termbase()
    found = tb.matches(text, src)
    if any(m[4] for m in found) and not is_it_context(text, src, domain, context, found):
        found = [m for m in found if not m[4]]      # 一般の文で意味が割れる形は IT の話の中だけ
    prefer_it = domain == "it"
    found.sort(key=lambda m: (not (prefer_it and m[5]), -(m[1] - m[0]), m[0]))
    for i, j, form, k, _ctx, _it in found:
        if len(out) >= limit:
            break
        if any(a < j and i < b for a, b in taken):
            continue
        e = tb.entries[k]
        cands = e.get(tgt) or []
        if not cands:
            continue
        taken.append((i, j))
        t = _keep_latin(form, cands, en=src == "en")
        if t is None:
            continue           # 原文の英字の語(S3・ArF)を消す訳しか無い: 手がかりにしない
        if t.lower() == form.lower() or form.lower() in seen:
            continue
        if e.get("keep") and _LATIN.search(form):
            continue           # 英字のまま書く語(製品名)は、英字で聞こえたなら手がかり不要
        seen.add(form.lower())
        out.append((form, t))
    out.sort(key=lambda p: text.find(p[0]) if text.find(p[0]) >= 0 else 1 << 20)
    return out


# ---------------------------------------------------------------- 分野(起動画面で選ぶ)

TERM_DOMAINS = ("", "it", "general")      # "" = 自動(テーマ欄に IT の語があれば IT)/ IT・技術 / 一般
_DOMAIN: str | None = None


def set_domain(choice: str, topic: str = "") -> None:
    """会話の始めに呼ぶ。choice = 起動画面の分野("" 自動 / "it" / "general")。"""
    global _DOMAIN
    if choice == "it":
        _DOMAIN = "it"
    elif choice == "general":
        _DOMAIN = None
    else:
        t = str(topic or "")
        _DOMAIN = "it" if t and (re.search(r"(?<![A-Za-z])IT(?![A-Za-z])", t)
                                  or any(is_it_context(t, lang) for lang in ("ja", "en", "zh", "ko"))) else None


def current_domain() -> str | None:
    return _DOMAIN


HINT_PREFIX = "Term translations to use: "


def glossary_hint(text: str, src: str, tgt: str, user=(), limit: int = 8, domain: str | None = None,
                  context: str = "") -> str:
    """訳の依頼に足す1行(無ければ "")。"""
    pairs = glossary_pairs(text, src, tgt, user=user, limit=limit, domain=domain, context=context)
    if not pairs:
        return ""
    return HINT_PREFIX + "; ".join(f"{s} = {t}" for s, t in pairs)


# ---------------------------------------------------------------- 耳の手がかり

# 字名で読まれやすい略語(耳の手がかり用に短く)。whisper の手がかりに英字の略語があると、
# 「エーアイ」ではなく「AI」と書きやすくなる。
STT_ACRONYMS = ("AI", "IoT", "AIoT", "GPU", "CPU", "API", "LLM", "DX", "PC", "KPI", "DNA", "SNS")


def stt_bias(topic: str, terms=(), max_len: int = 200) -> str:
    """耳(whisper)の手がかり: テーマ・用語辞書(今までどおり)の後ろに、よく字名で読まれる略語を足す。"""
    from .guards import stt_prompt
    return stt_prompt(topic, max_len=max_len, terms=tuple(terms or ()) + STT_ACRONYMS)


# ---------------------------------------------------------------- 履歴の写し

def similar(a: str, b: str) -> float:
    """2つの文の似かた(0〜1、句読点・空白を除いた文字で)。"""
    import difflib
    na = re.sub(r"[\s\W_]+", "", a or "").lower()
    nb = re.sub(r"[\s\W_]+", "", b or "").lower()
    if not na or not nb:
        return 0.0
    return difflib.SequenceMatcher(None, na, nb).ratio()


# ---------------------------------------------------------------- 利用者の語の聞き違い

def _lev(a: str, b: str, cap: int) -> int:
    """編集距離(cap を超えたら cap+1)。"""
    if abs(len(a) - len(b)) > cap:
        return cap + 1
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        lo = cap + 1
        for j, cb in enumerate(b, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb))
            lo = min(lo, cur[j])
        if lo > cap:
            return cap + 1
        prev = cur
    return prev[-1]


def fix_user_terms(text: str, user=()) -> str:
    """耳の文字の中の、利用者の語(用語辞書・テーマ)の1〜2字の聞き違いを直す
    (「北山工大学」→「北山工業大学」)。別の語を壊さないよう、5字以上(英字は6字以上)の語だけ、
    頭の字が同じで、違いが1字(10字以上は2字)まで。SenseVoice のように手がかりを渡せない耳でも効く。"""
    if not text or not user:
        return text
    pairs = _user_pairs(user)
    forms = {t.lower() for t, _tr in pairs}
    for t, _tr in pairs:
        latin = bool(_LATIN.search(t))
        if len(t) < (6 if latin else 5) or (t.lower() in text.lower() if latin else t in text):
            continue
        hay = text.lower() if latin else text
        nd = t.lower() if latin else t
        if len(set(nd) & set(hay)) < 0.6 * len(set(nd)):
            continue                      # 字がほとんど無い(速く飛ばす)
        cap = 2 if len(t) >= 10 else 1
        best = None
        for i, ch in enumerate(hay):
            if ch != nd[0]:
                continue
            for L in (len(nd) - 1, len(nd) + 1, len(nd)):
                w = hay[i:i + L]
                if len(w) != L or w == nd:
                    continue
                if w.lower() in forms:
                    continue           # 別の利用者の語そのもの(北山工业大学 と 北山工業大学)
                if latin and not w.isascii():
                    continue           # 英字の語は英字の並びの中だけで比べる(後ろの漢字を食べない)
                if latin and ((i > 0 and hay[i - 1].isalnum() and hay[i - 1].isascii())
                              or (i + L < len(hay) and hay[i + L].isalnum() and hay[i + L].isascii())):
                    continue
                d = _lev(w, nd, cap)
                if d <= cap and (best is None or d < best[0]):
                    best = (d, i, L)
        if best:
            _d, i, L = best
            text = text[:i] + t + text[i + L:]
    return text


# 音の近さで用語の崩れを直すか(soundfix.py、11/08 IT 用語。評価の切り替え用)
USE_SOUNDFIX = True


def clean_heard(text: str, lang: str, user=()) -> str:
    """耳の文字の直し(字幕・訳の前): 字名の並び → 英字の略語、崩れた用語の読み → 正しい表記
    (soundfix.py)、利用者の語の聞き違い。"""
    text = normalize_letters(text, lang)
    if USE_SOUNDFIX and text:
        from .soundfix import fix
        text = fix(text, lang)
    return fix_user_terms(text, user)


def warm() -> None:
    """用語集・略語の表を先に読んでおく(最初の訳の依頼で 20〜30ms 待たないように。会話の始めに裏で呼ぶ)。"""
    try:
        normalize_letters("エーアイ", "ja")
        glossary_hint("グラフ", "ja", "en")
        from .soundfix import index
        index()
        similar("a", "b")
    except Exception:  # noqa: BLE001
        pass
