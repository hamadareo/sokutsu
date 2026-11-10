"""音声レベルの区切り(Endpointer): 「どこで1文を訳すか」を決める。

11/05夜「会話の区切りがまだまだ」。合成会話(正解15ターン)の評価で、現行の
「無音0.5秒で区切る + 文字で続き待ち」は 別人と連結5件・文の分割3件・字幕まで
最大9.4秒だった。

- 対面の会話の交代は0.2〜0.4秒しか空かない。0.5秒の無音待ちでは2人分が
  1区間になり、whisper は1区間を1言語で聞くので英語の返事が丸ごと消えた。
- 文の途中の間(0.7〜1.2秒)では切れ、断片がそのまま訳された。
- 長い独り話は12秒の上限まで1塊で、字幕まで9秒待たされた。

方式: VAD は短い無音(0.2秒)で細かく区切り、区間ごとに文字にしてから文末で決める。
  句点・？などで終わる     → すぐ訳す
  読点・助詞・but で止まる → 続きを hold_open_s 待つ
  句読点なし(途中でもない) → hold_soft_s だけ待つ(whisper は日本語の「。」を落としがち)
  言語が変わった           → 別の人として切り離す
続きが来たら音声ごと繋いで聞き直す(rejoin)。断片の文字を継ぎ足すより正確で、
言語判定も1文全体で行える。rejoin が無い(このMacの耳)か失敗したら文字を繋ぐ。

時刻は2種類: lo/hi はマイク入力の通算秒(間の長さと重なりの計算用)、now は
壁時計(待ち時間の期限用)。どちらも呼び出し側が渡すので、評価スクリプトでは
音声の時計でそのまま再生できる。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from .guards import _COMMA, _SENT_END, ends_open, join_text
from .langdetect import detect_lang, norm

SR = 16000


@dataclass
class Unit:
    """訳す1単位。lo/hi はマイク入力の通算秒、t_ready は訳してよくなった壁時計。"""
    text: str
    lang: str
    lo: float
    hi: float
    t_ready: float
    parts: int = 1
    speech_lo: float = -1.0   # 話し始め(手前の補いを除く)。-1 = lo と同じ
    audio: "np.ndarray | None" = field(default=None, repr=False, compare=False)   # この単位の声(聞き直し用)

    def __post_init__(self):
        if self.speech_lo < 0:
            self.speech_lo = self.lo


@dataclass
class _Pend:
    audio: np.ndarray
    text: str
    lang: str
    lo: float
    hi: float
    deadline: float
    wait: float
    parts: int
    speech_lo: float = 0.0
    wild: bool = False     # 1秒未満・低確率: 言語も中身も当てにならない断片
    span: "list | None" = None   # audio がマイクの通算サンプルのどこか(先送りの範囲, uplink.py)
    spk: "Callable[[], str | None] | None" = None   # 最後の区間の話者(分かっていれば。会議メモ)
    last: "np.ndarray | None" = None   # 最後の区間の声(話者が分からないとき見当に使う)
    done: bool = False     # 文末まで言い終わっている(短いので念のため続きを待っているだけ)


# 英語でこの語の直後に「.」が来たら whisper の付け足し(文はまだ途中)
_EN_DANGLING = {"a", "an", "the", "and", "but", "or", "to", "of", "for", "with", "because",
                "if", "when", "while", "my", "your", "their", "our", "its", "at", "from",
                "as", "than", "into", "about", "are", "is", "was", "were", "be", "we", "i",
                "you", "they", "he", "she", "um", "uh"}

# 句点が無くても文の終わりとみてよい日本語の文末(whisper は「。」を落としがち)
_JA_FINAL = ("ます", "です", "ました", "でした", "ません", "でしょう", "ましょう",
             "ください", "ますね", "ですね", "ますよ", "ですよ", "ますか", "ですか")


# 助詞と同じ字で終わるが、言い切りの形(句点の直前が「ど」でも途中ではない)
_JA_NOT_OPEN = ("なるほど", "ほど")


def finished(text: str) -> bool:
    return classify(text, "") == "done"


def classify(text: str, lang: str) -> str:
    """文末の状態: done=すぐ訳す / open=続きを長く待つ / mid=中くらい / soft=少し待つ。"""
    t = (text or "").rstrip()
    if not t:
        return "soft"
    if t.endswith("...") or t.endswith("…"):
        return "open"          # 言いよどみ(「but...」)は途中
    if t[-1] in _SENT_END:
        if lang == "en":
            # whisper は短い断片にも「.」を付ける(AMI 会議:「We wanted to.」「to be
            # original…」に割れた)。「.」の直前が機能語なら文の途中とみる
            w = re.findall(r"[A-Za-z']+", t)
            if w and w[-1].lower() in _EN_DANGLING:
                return "open"
        elif lang in ("ja", "zh", "ko"):
            # 耳は息つぎで切った断片にも「。」「？」を付ける(11/08 作った対話 9 本: このMac で 126 発話中 61 が
            # 文の数より多く割れ、「私は土曜に友達と。」を訳していた)。句点の直前が助詞なら文の途中とみる
            core = t.rstrip(_SENT_END + " ")
            if core and ends_open(core, lang) and not core.endswith(_JA_NOT_OPEN):
                return "open"
        return "done"
    if ends_open(t, lang):
        return "open"
    if lang == "ja" and t.endswith("は") and not t.endswith(("こんにちは", "こんばんは")):
        return "open"          # 「水深は」= 主題だけ言って間を置いた
    if lang == "ja" and t.endswith(_JA_FINAL):
        return "done"
    if lang == "en":
        return "mid"           # 英語は whisper がほぼ句読点を付ける: 無い = 途中寄り
    return "soft"


def _trim_span(span: "list | None", k: int) -> "list | None":
    """範囲の先頭 k サンプルを落とす(音声の頭を切ったのに合わせる)。"""
    if span is None or k <= 0:
        return span
    out = []
    for a, b in span:
        n = b if a == "z" else b - a
        if k >= n:
            k -= n
            continue
        if k > 0:
            a, b = ("z", n - k) if a == "z" else (a + k, b)
            k = 0
        out.append((a, b))
    return out


def merge_segments(items: list, sr: int = SR, max_gap_fill_s: float = 0.5,
                   max_s: float = 28.0) -> tuple:
    """詰まってたまった区間 [(音声, lo, hi, pre)](時刻順)を1つにまとめる
    → (音声, lo, hi, pre, span)。間は無音で埋め(最大 max_gap_fill_s 秒)、前の区間と
    重なる手前の補いは落とす(Endpointer._join と同じ)。長すぎるときは新しい方から max_s 秒。

    11/06 英会話 AMI 実時間評価: 回線が混んで耳の往復が2〜4秒になると区間が3つたまり、
    最新1個だけ残して捨てていた(3分で13〜17区間の声を失った)。まとめて1回で聞けば
    捨てずに追いつける(先送りがあれば送るのは範囲だけ)。"""
    keep: list = []
    total = 0.0
    for it in reversed(items):
        total += it[2] - it[1]
        if keep and total > max_s:
            break
        keep.append(it)
    keep.reverse()
    parts, span = [], []
    audio0, lo, hi, pre = keep[0][:4]
    parts.append(np.asarray(audio0, dtype=np.float32))
    span.append((int(round(lo * sr)), int(round(hi * sr))))
    for audio, lo2, hi2, _pre2 in (it[:4] for it in keep[1:]):
        a = np.asarray(audio, dtype=np.float32)
        b0 = int(round(lo2 * sr))
        gap = lo2 - hi
        if gap < 0:
            k = min(len(a), int(round(-gap * sr)))
            a = a[k:]
            b0 += k
        else:
            n = int(round(min(gap, max_gap_fill_s) * sr))
            if n:
                parts.append(np.zeros(n, np.float32))
                span.append(("z", n))
        if len(a):
            parts.append(a)
            span.append((b0, b0 + len(a)))
        hi = max(hi, hi2)
    return np.concatenate(parts), lo, hi, pre, span


# ---------- 長い発話を区切る(11/08「切れ目なく話し続けると 9〜12.5 秒の1単位になり、読み上げを飛ばした」) ----------
_CUT_SENT = set("。．！？!?.")
_CUT_COMMA = set("，、,；;：:")


def pick_cut_at(tokens: list, stamps: list, dur_s: float, *, split_s: float = 6.0,
             comma_s: float = 9.0, breath_s: float = 9.0, min_head_s: float = 1.5,
             min_tail_s: float = 0.3, min_breath_gap_s: float = 0.45) -> "tuple[float, int] | None":
    """話し続けている区間(今 dur_s 秒)の途中字幕の字と時刻から、どこで前半を確定するか(音声の頭からの秒)。
    区切らないなら None。

      split_s 秒を超えたら: いちばん後ろの文末(。！？ .)の後ろ
      comma_s 秒を超えても文末が無ければ: いちばん後ろの読点(，、,)の後ろ
      breath_s 秒を超えても句読点が無ければ: 字と字の間がいちばん空いた所(息つぎ。min_breath_gap_s 以上)
    切る時刻は句読点と次の字の時刻の間(次の字がまだ無ければ句読点の少し後)。前半が min_head_s 秒より
    短い・後半が min_tail_s 秒より短い所では切らない(1字の文や、言い終わりの句点で切らない)。"""
    n = min(len(tokens), len(stamps))
    if n < 2 or dur_s < split_s:
        return None

    def at(i: int) -> float:
        nxt = stamps[i + 1] if i + 1 < n else min(dur_s, stamps[i] + 0.24)
        return (stamps[i] + nxt) / 2

    def ok(t: float) -> bool:
        return t >= min_head_s and dur_s - t >= min_tail_s

    def last(marks: set) -> "tuple[float, int] | None":
        for i in range(n - 1, -1, -1):
            if str(tokens[i]).strip() in marks and ok(at(i)):
                return at(i), i
        return None

    t = last(_CUT_SENT)
    if t is None and dur_s >= comma_s:
        t = last(_CUT_COMMA)
    if t is None and dur_s >= breath_s:
        best, gap = None, min_breath_gap_s
        for i in range(n - 1):
            g = stamps[i + 1] - stamps[i]
            if g >= gap and ok(at(i)):
                best, gap = (at(i), i), g
        t = best
    return t



def pick_cut(tokens: list, stamps: list, dur_s: float, **kw) -> "float | None":
    """pick_cut_at の時刻だけ。"""
    r = pick_cut_at(tokens, stamps, dur_s, **kw)
    return None if r is None else r[0]


def head_text(tokens: list, i: int) -> str:
    """区切った前半の字(トークン 0〜i)。SenseVoice の英語のトークンは語頭が「▁」。"""
    t = "".join(str(x) for x in tokens[:i + 1]).replace("▁", " ").strip()
    t = re.sub(r"\s+([.,!?;:])", r"\1", t)
    from .stt import join_cjk_spaces
    return join_cjk_spaces(t)


def _spk_now(f) -> "str | None":
    try:
        return f()
    except Exception:  # noqa: BLE001 - 話者が分からないだけ(今まで通り繋ぐ)
        return None


class Endpointer:
    def __init__(self, rejoin: "Callable[[np.ndarray], tuple] | None" = None, *,
                 hold_open_s: float = 1.2, hold_soft_s: float = 0.5, hold_mid_s: float = 1.0,
                 max_merge_s: float = 15.0, max_gap_fill_s: float = 0.5,
                 speak_cap_s: float = 3.0, speak_cap_done_s: float = 1.0,
                 hold_done_s: float = 0.3, short_done: int = 8,
                 tiny_s: float = 0.9, hold_tiny_s: float = 0.8, tiny_prob: float = 0.5,
                 trust_end_punct: bool = True,
                 spk_differ: "Callable[[np.ndarray, str], bool | None] | None" = None,
                 sr: int = SR):
        self.rejoin = rejoin
        self.hold_open_s = hold_open_s
        self.hold_soft_s = hold_soft_s
        self.hold_mid_s = hold_mid_s
        self.max_merge_s = max_merge_s
        self.max_gap_fill_s = max_gap_fill_s
        self.speak_cap_s = speak_cap_s
        # 耳の文末の句読点を信じるか(11/08 品質 A)。SenseVoice は区切った断片の終わりに必ず句点を付ける
        # (作った対話: 単位 294 個すべて)。信じないときは句点を外して文末の言葉で決め、言い切りの形でも
        # hold_done_s だけ続きを待つ(「行ったんです。」+「けど、…」を繋ぐ)
        self.trust_end_punct = trust_end_punct
        # 文末まで言い終わった短い断片(「可以。」)は、続けて声が始まっても延長はここまで(11/08: 動画の音や
        # 相手の声が続くと 3 秒待ち、話し終わりから確定まで 3.9 秒、読み上げ +3〜4.3 秒になっていた)
        self.speak_cap_done_s = speak_cap_done_s
        self.hold_done_s = hold_done_s
        self.short_done = short_done
        # 1秒未満の区間(11/06 FLEURS 多言語評価): 単独で whisper に聞かせると幻覚
        # (「ご視聴ありがとうございました」「Cool.」「you」)や言語の取り違え、
        # 「。」付きの断片(「设施。」「相似。」「例如」)になる。junk 13件はすべて
        # 0.37〜0.85秒。続きを hold_tiny_s 待って音声ごと聞き直す(rejoin)。
        # 低確率(tiny_prob 未満)の断片は言語を当てにせず次と繋ぎ、続きが無ければ捨てる。
        self.tiny_s = tiny_s
        self.hold_tiny_s = hold_tiny_s
        self.tiny_prob = tiny_prob
        # 話者の分からない短い区間の見当(11/07 擬似会議 v2): (声, 相手の話者) → 別の人なら True
        self.spk_differ = spk_differ
        self.sr = sr
        self._p: _Pend | None = None
        # 聞き直しに音声の範囲も渡せるか(先送り 11/06: 範囲だけ送れば送り直しが要らない)
        self._rejoin_span = False
        if rejoin is not None:
            try:
                import inspect
                self._rejoin_span = "span" in inspect.signature(rejoin).parameters
            except (TypeError, ValueError):
                pass

    # ---------- 外から ----------

    def pending_text(self) -> str:
        return self._p.text if self._p else ""

    def pending(self) -> "tuple[str, str] | None":
        """待っている断片の (文字, 言語)。先読み翻訳(speculate.py)の鍵。"""
        return (self._p.text, self._p.lang) if self._p else None

    def push(self, audio, lo: float, hi: float, text: str, lang: str,
             now: float, pre: float = 0.0, prob: float = 1.0,
             span: "list | None" = None,
             spk: "Callable[[], str | None] | None" = None,
             cut: bool = False) -> list[Unit]:
        """新しい区間(既に文字にしたもの)を入れ、訳してよくなった単位を返す。
        audio は lo〜hi 秒(手前の補い pre 秒を含む。話し始めは lo+pre)。

        区間が届くのは話し終えた後なので、続きかどうかは届いた時刻ではなく
        「前の区間の終わりから話し始めるまでの間」が待ち時間以内かで決める。
        """
        out: list[Unit] = []
        audio = np.asarray(audio, dtype=np.float32)
        tiny = self.rejoin is not None and (hi - (lo + pre)) < self.tiny_s
        wild = tiny and prob < self.tiny_prob
        p = self._p
        seg_audio = audio
        if p is not None and not self._joinable(p, lo, hi, lang, lo + pre, wild, spk, seg_audio):
            out += self._emit_list(min(now, p.deadline))
            p = None
        speech_lo = lo + pre
        if p is not None:
            audio, text, lang, span = self._join(p, audio, lo, text, lang, span)
            lo, parts, speech_lo = p.lo, p.parts + 1, p.speech_lo
            self._p = None
        else:
            parts = 1
        if self.trust_end_punct:
            state = classify(text, lang)
        else:
            state = classify(text.rstrip("。．.！!？? "), lang)
            if state == "done":
                state = "short"
        if state == "done" and parts == 1 and len(norm(text)) <= self.short_done:
            # whisper は短い断片に単独で聞かせると「。」を付けがち(「私も。」「そう思い
            # ます。」に割れた): 短いものだけ hold_done_s 続きを待つ
            state = "short"
        if parts == 1 and tiny and state != "open":
            state = "tiny"
        if cut:
            # 話し続けている区間を途中で区切った前半(pick_cut。句読点・息つぎの所): 続きを待たずに訳す。
            # 待つと続きの区間と繋ぎ直して、また長い1単位に戻る
            state = "done"
        if state == "done":
            out.append(Unit(text, lang, lo, hi, now, parts, speech_lo, audio=audio))
            return out
        wait = {"open": self.hold_open_s, "mid": self.hold_mid_s, "tiny": self.hold_tiny_s,
                "short": self.hold_done_s}.get(state, self.hold_soft_s)
        self._p = _Pend(audio, text, lang, lo, hi, now + wait, wait, parts, speech_lo,
                        wild=(wild and parts == 1), span=span,
                        spk=spk if spk is not None else (p.spk if p is not None else None),
                        last=seg_audio,
                        # 文末まで言い終わっている印は、その言語で見た状態から(句点付きの助詞止まり「…なら。」は
                        # 言い終わりではない。11/10: finished() は言語なしで見て「done」とし、続きを話している最中でも
                        # 期限の 1 秒後に出していた)
                        done=(classify(text, lang) == "done"))
        return out

    def tick(self, now: float, speaking_since: "float | None" = None) -> list[Unit]:
        """待ち期限が過ぎた断片を、続きなしとしてそのまま返す。

        speaking_since = 今話している区間の話し始め(マイク入力の通算秒)。それが
        断片の終わりから待ち時間以内なら続きを話している最中なので確定を待つ
        (区間が届くのは話し終えた後)。ただし期限から speak_cap_s を超えたら出す。
        待ち時間より後に始まった声は別の発話なので、待たずに断片を出す。
        """
        p = self._p
        if p is None or now < p.deadline:
            return []
        cap = self.speak_cap_done_s if p.done else self.speak_cap_s
        if (speaking_since is not None and speaking_since - p.hi <= p.wait
                and now < p.deadline + cap):
            return []
        return self._emit_list(now)

    def flush(self) -> list[Unit]:
        if self._p is None:
            return []
        return self._emit_list(self._p.deadline)

    # ---------- 中身 ----------

    def _emit(self, t_ready: float) -> Unit:
        p = self._p
        self._p = None
        return Unit(p.text, p.lang, p.lo, p.hi, t_ready, p.parts, p.speech_lo, audio=p.audio)

    def _emit_list(self, t_ready: float) -> list[Unit]:
        """待っていた断片を出す。続きの来なかった低確率の短い断片は雑音の幻覚として捨てる。"""
        wild = self._p is not None and self._p.wild
        u = self._emit(t_ready)
        return [] if wild else [u]

    def _joinable(self, p: _Pend, lo: float, hi: float, lang: str, speech_lo: float,
                  wild: bool = False, spk=None, audio=None) -> bool:
        if lo < p.lo:                       # 入力の時計が戻った(読み上げ後の作り直し)
            return False
        if speech_lo - p.hi > p.wait:       # 待ち時間より後に話し始めた = 続きではない
            return False
        if lang and p.lang and lang != p.lang and not (wild or p.wild):
            return False                    # 別の言語 = 別の人(当てにならない断片は除く)
        if hi - p.lo > self.max_merge_s:    # 長すぎる塊は作らない(字幕が遅れる)
            return False
        if spk is not None and p.spk is not None:
            # 会議メモ(11/06 擬似会議 v1): 同じ言語の別の人の短い発言を繋いでいた(「田中です佐藤です」
            # 107行中5行)。両方の話者が分かっていて違えば別の人
            a, b = _spk_now(p.spk), _spk_now(spk)
            if a is not None and b is not None and a != b:
                return False
            # 片方だけ分からない(短い区間は話者を付けない。11/07 v2 で3件繋がった): その区間の声を
            # 分かっている方の話者と比べ、別の人と分かれば繋がない(状態を変えない見当)
            if self.spk_differ is not None and (a is None) != (b is None):
                x, other = (audio, a) if b is None else (p.last, b)
                if x is not None and len(x):
                    try:
                        if self.spk_differ(x, other) is True:
                            return False
                    except Exception:  # noqa: BLE001 - 分からなければ今まで通り繋ぐ
                        pass
        return True

    def _join(self, p: _Pend, audio: np.ndarray, lo: float, text: str, lang: str,
              span: "list | None" = None):
        gap = lo - p.hi
        if gap < 0:   # 次の区間の先頭(プリロール)が前の区間と重なる分を落とす
            k = min(len(audio), int(round(-gap * self.sr)))
            audio = audio[k:]
            mid = np.zeros(0, dtype=np.float32)
            span = _trim_span(span, k)
        else:
            mid = np.zeros(int(round(min(gap, self.max_gap_fill_s) * self.sr)), np.float32)
        merged = np.concatenate([p.audio, mid, audio])
        mspan = None
        if p.span is not None and span is not None:
            mspan = list(p.span) + ([("z", len(mid))] if len(mid) else []) + list(span)
        joined = join_text(p.text, text, p.lang or lang)
        if self.rejoin is not None:
            try:
                if self._rejoin_span and mspan is not None:
                    t2, l2, _prob = self.rejoin(merged, span=mspan)
                else:
                    t2, l2, _prob = self.rejoin(merged)
                if t2 and t2.strip() and len(norm(t2)) * 2 >= len(norm(p.text)) + len(norm(text)):
                    return merged, t2, (l2 or detect_lang(t2) or p.lang), mspan
                # 聞き直しが繋ぐ前の文字の半分にも満たない = 文が消えた(11/06 擬似会議 v1:
                # 「Okay」+「データは共有ドライブで共有することにします」→「OK」だけ)。文字で繋ぐ
            except Exception:  # noqa: BLE001  聞き直しに失敗 → 文字で繋ぐ
                pass
        return merged, joined, (p.lang or lang), mspan
