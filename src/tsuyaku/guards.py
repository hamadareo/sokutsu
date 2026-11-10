"""詰まり防止の防波堤: 詰め込み解消と、雑音から生まれる幻覚の除去。

ライブ不具合(2026-11-05): 会場の雑音で VAD が蹴ると whisper が幻覚の短文を
出す(「카우다.」「하다. 하다.」「Calder. Calder. Hold on.」)。1つにつき
whisper+LLM+読み上げゲート(約2〜3秒)が直列に潰れ、本物の発話が行列して
「ずっと翻訳してくれない」状態になった。ここでそれを止める。
"""
from __future__ import annotations

import re

import numpy as np

from .langdetect import HAN, detect_lang, norm


def pick_segments(pending: list[np.ndarray]) -> list[np.ndarray]:
    """キューに詰まったセグメントのうち、実際に whisper へ流すものを選ぶ。

    3個以上たまっていたら=もう詰まっている。古い順に訳していても追いつけないし、
    古い発話の訳は今さら会話を止められるものでもない。最新1個だけに絞って
    即座に現在へ追いつかせる(1〜2個なら会話のテンポなので全部訳す)。
    """
    if len(pending) >= 3:
        return [pending[-1]]
    return list(pending)


# トークン確率の床(11/05 M3 + whisper large-v3-turbo greedy で較正):
#   実音声: 日本語 0.87 / 英語 0.87
#   純白雑音の幻覚: 「Thank you.」0.52 /「.」0.69 / 90Hzうなり→「you」0.76
# 方針は「幻覚を1つ通すほうが、本物を1つ切るより会話の邪魔が小さい」。
# ペア外(第三言語の話者は卓では稀)だけ 0.80 の床を置く。
# ペア内には床を置かない(11/05 16時台ライブ実測): 早口の雑談は 0.22〜0.48 と
# 雑音幻覚(0.52〜0.76)より「低く」出るので、確率では本物と幻覚を分けられない。
# 0.50 の床で「もっとさ、なんか楽にやった方がいいよ。」(0.48)等のペア内を27件
# 捨てていた。ペア内の雑音対策は rms ゲート(noise_rms_floor)が受け持つ。
_PROB_OOP_FLOOR = 0.80

# whisper が無音・雑音・短い区間で出す動画の締めの定型句(11/06 FLEURS 評価で
# 「ご視聴ありがとうございました」3件、ライブでも 11/05 に既出)。会議で本当に
# 言うのは「ご清聴ありがとうございました」(清聴)なので、そちらは捨てない。
_STOCK = re.compile(r"ご視聴ありがとうございま|チャンネル登録|高評価(を|お願い)|"
                    r"[Tt]hanks? (you )?for watching|[Ss]ubscribe to|"
                    r"请不吝点赞|订阅|字幕(by|提供|由)|시청(해 ?주셔서|해주셔서) 감사|구독")


def seg_probability(segs) -> float:
    """whisper セグメント群の最低トークン確率(幻覚守卫に渡す)。

    1つでも怪しいセグメントがあれば文ごと怪しいので min を取る。
    確率が取れない(NaN・属性なし)セグメントだけなら 1.0 = 信頼する
    (守卫で本物を切らない方向に倒す)。
    """
    ps: list[float] = []
    for sg in segs or []:
        p = getattr(sg, "probability", None)
        if p is None:
            continue
        try:
            p = float(p)
        except (TypeError, ValueError):
            continue
        if p != p:  # NaN
            continue
        ps.append(p)
    return min(ps) if ps else 1.0


def stock_phrase(text: str) -> bool:
    """動画の締めの定型句(「ご視聴ありがとうございました」等、whisper の幻覚の定番)か。"""
    return bool(_STOCK.search(text or ""))


def hallucinated_out_of_pair(pair: tuple[str, ...], lang: str, text: str,
                             min_len: int = 12,
                             prob: float | None = None) -> bool:
    """whisper の幻覚なら True。

    幻覚は毎回違う言語に振れる(カウダ→Calder→하다)うえ、たいて数文字。
    本物の発話がペア外の言語になることは学会の卓では稀なので、
    「ペア外×短文」で捨てても失うものは小さい。ペア内の言語、
    判定不能(空言語)、ペア外の言語の長い発話(=本当に第三言語の話者)は通す。

    11/05 13:17〜14:07 のライブで出たペア外幻覚は全部 norm≤11
    (「감사합니다.」=5、「에이 또 이홍도 또」=7、「아니, 아까 안 하셨거든요.」=11)。
    min_len を 5→12 に上げるとそれらが全部落ち、本物の第三言語の一文
    (「안녕하세요 여러분 오늘도よろしくお願いします」等)は通る。

    prob(whisperのトークン確率)を渡すと長さだけでなく確率でも捨てる。
    11/05 15:46 に話していない韓国語「이게 10분이죠 이게 동거예요」(norm=15)が
    長さ守卫を抜けた実障害がこれ。ペア外は prob<0.80 で捨てる。ペア内は確率を
    見ない(本物の雑談のほうが確率が低いことがある)。prob=None は従来の長さだけ判定。
    """
    if _STOCK.search(text or ""):
        return True   # 動画の締めの定型句(whisper の学習データ由来の幻覚)。言語を問わない
    if not lang:
        return False
    if lang in pair:
        return False
    if prob is not None and prob < _PROB_OOP_FLOOR:
        return True
    return len(norm(text)) < min_len


# 話の区切り(11/05ライブ): 0.5秒のポーズで文の途中で切れた断片がそのまま
# 翻訳に回っていた(「秘密で」「こちらが」「てるから」「ちょっと」「to the」
# 「like it's a,」「up.」「to get it.」…)。文末が助詞/読点で終わる断片は
# 「まだ途中」なので、主ループが hold_open_s だけ続きを待って merge_held で
# 1文に戻してから訳す。
_OPEN_JA = set("をがのにとでもやへばてけれらど")  # よ/か/す(です・ます)は文末なので不可
_EN_OPEN = {
    "a", "an", "the", "and", "but", "or", "nor", "to", "of", "for", "with",
    "in", "on", "at", "from", "as", "that", "this", "these", "those", "so",
    "because", "if", "when", "while", "my", "your", "their", "our", "its",
    "is", "are", "was", "were", "be", "been", "have", "has", "had", "can",
    "could", "would", "should", "will", "just", "like", "more", "very",
    "about", "not", "i", "we", "you", "he", "she", "they", "it", "me", "us",
    "up", "down", "out", "off", "back",  # 句読点付きの "up." "It's up." は文末扱い
}
_OPEN_ZH = set("和与而或把被让给在很再还就要能会个之从对")  # 了/吗/呢/吧是文末なので不可
_OPEN_KO = ("을", "를", "이", "가", "은", "는", "과", "와", "도", "만", "에", "의",
            "로", "으로", "에서", "에게", "부터", "까지", "이나", "께", "뿐", "한테")
_SENT_END = "。．.！!？?"
_COMMA = ",，、"
_EN_WORD = re.compile(r"[A-Za-z']+")


def ends_open(text: str, lang: str) -> bool:
    """読んだ文末が「まだ途中」なら True。続き待ちマージの判定に使う。

    句点・読点・question mark が付いていればそれがwhisperの判断なので最優先:
    「。」等なら終わり、「,」等なら途中。句読点が無いときは助詞/機能語で止まる
    断片だけ途中とみなす(「以上」「でしょうか」「来ていた」は終わりとみる)。
    """
    t = (text or "").strip()
    if not t:
        return False
    if t[-1] in _COMMA:
        return True
    if t[-1] in _SENT_END:
        return False
    if lang == "ja":
        return t[-1] in _OPEN_JA
    if lang == "en":
        w = _EN_WORD.findall(t)
        return bool(w) and w[-1].lower() in _EN_OPEN
    if lang == "zh":
        return t[-1] in _OPEN_ZH
    if lang == "ko":
        return any(t.endswith(p) for p in _OPEN_KO)
    return False


def join_text(prev: str, nxt: str, lang: str) -> str:
    """断片と続きを1文につなぐ。英語だけ間に空白を入れる。"""
    a, b = (prev or "").strip(), (nxt or "").strip()
    if not a:
        return b
    if not b:
        return a
    return f"{a} {b}" if lang == "en" else a + b


def merge_held(held: tuple, text: str, lang: str) -> tuple[str, str, int]:
    """待っていた断片(held)に今回の聞き取りを続けて1文にする。

    held = (text, lang, t_last, n_parts)(主ループが持つ)。同じ言語なら連結して
    連結後でもう一度言語判定し直す。言語が違う = 断片側がwhisperのノイズか
    話の切り替えなので、断片は捨てて今回の発話をそのまま返す。
    戻り値 (text, lang, 連結回数)。
    """
    h_text, h_lang, _, h_n = held
    if not text or (lang and h_lang and lang != h_lang):
        return text, lang, 0
    if not h_text:
        return text, lang, 0
    merged = join_text(h_text, text, lang or h_lang)
    return merged, detect_lang(merged) or lang or h_lang, h_n + 1


def looks_truncated(text: str, trans: str) -> bool:
    """本文のある発話に対し訳が1〜3文字しか無い = 生成が最初で止まっている。

    11/05実測: 直近ターンに破綻訳が混ざると qwen がそれを真似て 'I' のような
    1語だけ返し、それがまた履歴に入って…と自己強化した(13:37〜13:47ログ)。
    空訳は別のガードが拾うのでここでは対象外。短い発話への短い訳は正当なので
    発話側が本文(6文字以上)のときだけ疑う。
    """
    if not trans:
        return False
    n_src, n_tr = len(norm(text)), len(norm(trans))
    if n_src < 6 or n_tr > 3:
        return False
    # 中国語の訳は漢字1字の情報量が大きく短くて正しい(11/06 日中の実会話:
    # 「ありがとうございます」→「谢谢」を破綻扱いして「翻訳できませんでした」)。
    # 漢字だけの訳は、原文の1/6より短いときだけ疑う。
    if re.fullmatch(f"(?:{HAN.pattern})+", norm(trans)) and n_tr * 6 >= n_src:
        return False
    return True


# ---- 訳の長さの外れ(11/08 このMac gemma-3-4b の日中の会話) ----
# 条件の断片(「如果…的话。」)に続きを創作して別の文にする・長い文の大半を落として「はい、そうですね。」と
# 返す。訳の字数 ÷ 原文の字数(句読点・空白を除く)の中央を FLEURS の並行文(12方向)で測った値。
_LEN_MED = {("ja", "en"): 2.04, ("ja", "zh"): 0.69, ("ja", "ko"): 0.92, ("en", "ja"): 0.49, ("en", "zh"): 0.34,
            ("en", "ko"): 0.44, ("zh", "ja"): 1.46, ("zh", "en"): 2.98, ("zh", "ko"): 1.31, ("ko", "ja"): 1.08,
            ("ko", "en"): 2.29, ("ko", "zh"): 0.77}


def _nchars(s: str) -> int:
    import unicodedata
    t = unicodedata.normalize("NFKC", s or "")
    return sum(1 for c in t if unicodedata.category(c)[0] not in "PSZC")


def length_off(src: str, trans: str, src_lang: str, tgt_lang: str) -> bool:
    """訳の長さが原文から大きく外れた(足した / 落とした)。短い文は字数の揺れが大きいので幅を広く取る:
    長すぎ = 見込み × 1.8 + 6字 を超える、短すぎ = 原文12字以上で 見込み × 0.35 未満。
    FLEURS の並行文では、見込みの p1〜p99 が 0.6〜1.7倍に収まる。"""
    med = _LEN_MED.get((src_lang, tgt_lang))
    if not med or not trans:
        return False
    n_src, n_tr = _nchars(src), _nchars(trans)
    if n_src == 0:
        return False
    exp = med * n_src
    if n_tr > exp * 1.8 + 6:
        return True
    return n_src >= 12 and n_tr < exp * 0.35


# ---- 用語の手がかり(11/06 実会話: YOLO→「養老」、人名の漢字の取り違え) ----
# テーマメモを whisper の initial_prompt に渡すと固有名詞が正しく出る。
# VOICEVOX 2声×12文の実測: 手がかり無し 9/24 → 「、」区切り+「。」で 22/24。
# 「・」区切りは 19/24、「用語: …」の文にすると「用語モデル」と混ざる。
# 「、…。」の形は文末の句点も 18/24 → 24/24 に増やす(区切り判定が安定する)。
# 英語の発話は壊れない(10文×2声で単語の取りこぼし増加なし、Qwen/RAG は正しく)。
_TERM_SPLIT = re.compile(r"[・,、，/;；\n]+")


def stt_prompt(topic: str, max_len: int = 200, terms=()) -> str:
    """テーマメモ(+用語辞書)→ whisper の手がかり文字列(「YOLO、山田太郎、…。」)。

    用語辞書(W3, 11/07)はテーマの後ろに、同じ語(大文字小文字を問わない)を除いて足す。
    長いときは語の切れ目で止める(語の途中で切ると、崩れた語を手がかりにしてしまう)。
    テーマだけで max_len を超える1語は、今までどおりそこで切る。
    """
    items = [t.strip(" 　。.") for t in _TERM_SPLIT.split(topic or "")]
    items = [t for t in items if t]
    if not terms:
        if not items:
            return ""
        return "、".join(items)[:max_len] + "。"
    seen = {t.lower() for t in items}
    for t in terms:
        t = str(t or "").strip(" 　。.")
        if t and t.lower() not in seen:
            seen.add(t.lower())
            items.append(t)
    out = ""
    for t in items:
        nxt = f"{out}、{t}" if out else t
        if len(nxt) > max_len:
            if not out:
                out = t[:max_len]
            break
        out = nxt
    return out + "。" if out else ""


def echoes_prompt(text: str, prompt: str) -> bool:
    """雑音に対して whisper が手がかりを復唱しただけの出力なら True。

    雑音22区間で1回「Qwen、RAG、Bス、TLACKID。」(p=0.18)が出た。手がかりの語を
    「、」で並べただけの出力を捨てる。用語を使った本物の文
    (「YOLO、Whisper、Qwenを使いました」)は最後の項目が長いので残る。
    """
    import difflib
    terms = [t.lower() for t in _TERM_SPLIT.split(prompt.rstrip("。")) if t.strip()]
    if not terms:
        return False
    items = [t.strip().lower() for t in re.split(r"[、,・\s。.]+", text) if t.strip()]
    if len(items) < 2:
        return False

    def is_term(it: str) -> bool:
        # 綴りの崩れ(TLACKID)は拾い、語尾の付いた本物(「YOLOです」)は拾わない
        return any(it == t or (abs(len(it) - len(t)) <= 1
                               and difflib.SequenceMatcher(None, it, t).ratio() >= 0.75)
                   for t in terms)
    flags = [is_term(it) for it in items]
    # 辞書の語が3つ以上並んでいれば、ほかの項目に関わらず復唱(11/07 用語辞書の A/B: 雑音の区間が
    # 「A0、山本、伊藤、中村、 nord、山跳森也。」になり、辞書に無い長めの項目で下の条件をすり抜けた)
    run = best = 0
    for f in flags:
        run = run + 1 if f else 0
        best = max(best, run)
    if best >= 3:
        return True
    other = [it for it, f in zip(items, flags) if not f]
    return (len(items) - len(other) >= 2 and len(other) <= 1
            and all(len(it) <= 4 for it in other))


# 名前に付く呼びかけ(「渡辺さん」)。terms_only で語の後ろから外す
_HONORIFIC = re.compile(r"(?:さん|さま|様|くん|君|ちゃん|先生|氏|-?san|-?sama)$", re.I)


def terms_only(text: str, prompt: str) -> bool:
    """聞き取りが手がかり(用語辞書・テーマ)の語(+さん/様など)だけなら True。

    11/07 用語辞書の A/B(ja_dev4): 3.7秒の区間(「それまでに必要な台数を見積もっておきます」+
    「渡辺さん」)が手がかりの語「渡辺さん。」だけになり、中身が消えた。呼ぶ側はこれを見て、
    長い区間なら手がかりなしで1回だけ聞き直す。手がかりが空なら常に False。"""
    import unicodedata
    terms = {unicodedata.normalize("NFKC", t).strip().lower()
             for t in _TERM_SPLIT.split((prompt or "").rstrip("。")) if t.strip()}
    if not terms:
        return False
    items = [unicodedata.normalize("NFKC", t).strip().lower()
             for t in re.split(r"[、,・\s。.!?！？]+", text or "") if t.strip()]
    if not items:
        return False
    return all(_HONORIFIC.sub("", it) in terms for it in items)


# ---- 訳の言語の取り違え(11/06 FLEURS 多言語評価) ----
# zh→ja で中国語をそのまま返す(16/41)、en→zh で日本語を出す(7/39)。英語の指示文で
# 大半は消えるが、残りを拾って空履歴でやり直すための判定。短い訳(4字未満)は
# 固有名詞や相づちで正当に同じ形になりうるので見ない。
def wrong_language(orig: str, trans: str, tgt: str) -> bool:
    """訳が届け先の言語(tgt)になっていない(原文の写し・別の言語)なら True。"""
    import difflib
    from .langdetect import _LATIN_WORD, _ZH_FUNC, HANGUL, KANA
    if not trans or not tgt:
        return False
    nt, no = norm(trans), norm(orig)
    if len(nt) < 4:
        return False
    if no and len(no) >= 6 and (nt == no or difflib.SequenceMatcher(None, nt, no).ratio() >= 0.8):
        return True   # 原文の写し
    kana = len(KANA.findall(trans))
    hangul = len(HANGUL.findall(trans))
    han = len(HAN.findall(trans))
    latin = len(_LATIN_WORD.findall(trans))
    if tgt == "ja":
        return hangul > 0 or (kana == 0 and han >= 4 and bool(_ZH_FUNC.search(trans))) \
            or (kana == 0 and han == 0 and latin >= 3)
    if tgt == "zh":
        return kana >= 2 or hangul > 0 or (han == 0 and latin >= 3)
    if tgt == "ko":
        return hangul == 0 and (kana > 0 or han >= 4 or latin >= 3)
    if tgt == "en":
        return (kana + hangul + han) >= 4 and latin <= 2
    return False


def strip_history_echo(orig: str, trans: str) -> str:
    """訳が「原文 => 訳」の形で返った(依頼の履歴の並べ方を真似た)とき、訳の部分だけにする(11/09)。
    => の左が原文とよく似ている時だけ(訳の中の矢印はそのまま)。"""
    import difflib
    t = trans or ""
    if "=>" not in t:
        return t
    left, right = t.split("=>", 1)
    if not right.strip():
        return t
    if difflib.SequenceMatcher(None, left.strip(), (orig or "").strip(), autojunk=False).ratio() < 0.6:
        return t
    return right.strip()
