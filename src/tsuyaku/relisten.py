"""このMacの耳(SenseVoice)が落とした英字の略語を、裏で whisper に聞き直して直す(11/08)。

見つけた問題(eval/terms): SenseVoice は日本語の中の英字の略語を書かない(「LLMのファインチューニングに
はGPUが8枚」→「のファインチューニングにはが8枚」。作った74文の略語42個が聞き取りに0個)。落ちた語は
文字の直し(terms.normalize_letters)では戻らない。このMacの whisper なら戻る(34/42・22/42)が、1回
2.4〜2.7秒かかる。

設計(リーダー判断 11/08「先に出して、後から直す」): 落とした見当(dropped_letters)が当たった文は
今までどおりすぐ字幕・訳を出して「確認中」の印を付け、裏で1本ずつ whisper で聞き直す。略語が戻ったら
その行の原文と訳を差し替え、記録に edit 行を残す(Bridge 側)。リモートサーバーの耳のときは使わない。

見当: 文が助詞で始まる・読点の後が助詞・「この」の直後が助詞・助詞が2つ並ぶ(「はの」「はと」…)。
当たり: 略語を含む文の 14/29(VOICEVOX)・9/29(say Kyoko)、略語を含まない文 0/45、FLEURS ja 1/143。
"""
from __future__ import annotations

import queue
import re
import threading
import time
from typing import Callable

_P = "はがをにでとのへも"
_NOT_KANA = r"(?![ぁ-ゖー])"
_START = re.compile(rf"^[{_P}]{_NOT_KANA}")
_CLAUSE = re.compile(rf"[、。,，]\s*[{_P}]{_NOT_KANA}")
_KONO = re.compile(rf"(?:この|その|あの|どの)[{_P}]{_NOT_KANA}")
_PAIR = re.compile(rf"(?:はの|はと|がの|をの|はを|がを|をが|はが|がは|をは|はで|がで){_NOT_KANA}")
_LATIN2 = re.compile(r"[A-Za-z]{2,}")


# 11/08 IT 用語(IT の試験文 220文で、落とした・崩した文 127 のうち今までの見当は 22 → 35。一般の文
# (FLEURS の参照文 650・耳の文字 325)で当たるのは 1+1 → 2+5):
# 文頭や仮名でない字の後の長音(「ーにプッシュ」)、仮名でない字の後の助詞の並び(「メッシュでを」)、
# 文頭・助詞・読点の後の1〜2桁の数字だけ + 助詞(「ルーターで9を」= QoS、「6に移行」= IPv6)
_BAR = re.compile(r"(?:^|[^ぁ-ゖァ-ヺー])ー")
_PAIR2 = re.compile(r"(?<![ぁ-ゖ])(?:でを|とを|でと|をで|とで|にを|をに|がで|にで|がを|をと)(?![ぁ-ゖ])")
_DIGIT = re.compile(r"(?:^|[はがをにで、])\d{1,2}(?:[をにでがはと]|以上|未満)")
_KATA = re.compile(r"[ァ-ヺー]+")
_LEX: set | None = None


def _lexicon() -> set:
    """知っているカタカナの語(用語集の形 + 一般の語の一覧)。"""
    global _LEX
    if _LEX is None:
        try:
            from .soundfix import index
            idx = index()
            _LEX = set(idx.ja_known) | set(idx.common_ja)
        except Exception:  # noqa: BLE001
            _LEX = set()
    return _LEX


def _short_unknown_katakana(t: str) -> bool:
    """用語集にも一般の語の一覧にも無い短いカタカナ(「ブキー」= VPC、「デー」= DB)。
    IT の話の中だけで見る(一般の会話では地名・人名・言いよどみでも出る)。"""
    lex = _lexicon()
    return any(1 <= len(k.rstrip("ー")) <= 3 and k not in lex for k in _KATA.findall(t))


def dropped_letters(text: str, lang: str, it: bool = False) -> bool:
    """SenseVoice が日本語の文の中の英字の略語を落とした・崩した見当(ja だけ)。
    it = IT の話の中(分野で IT を選んだ、または今か直前の発話に IT の語がある)。"""
    if lang != "ja" or not text:
        return False
    t = text.strip()
    if (_START.match(t) or _CLAUSE.search(t) or _KONO.search(t) or _PAIR.search(t)
            or _BAR.search(t) or _PAIR2.search(t) or _DIGIT.search(t)):
        return True
    return bool(it and _short_unknown_katakana(t))


def recovered(old: str, new: str) -> bool:
    """聞き直しで英字の語が戻ったか(前に無かった英字2字以上の語が増え、残りの文はおおむね同じ)。"""
    import difflib
    if not new:
        return False
    olds = {w.lower() for w in _LATIN2.findall(old or "")}
    gained = [w for w in _LATIN2.findall(new) if w.lower() not in olds]
    if not gained:
        return False
    a = re.sub(r"[\sA-Za-z0-9、。,.!?！？]+", "", old or "")
    b = re.sub(r"[\sA-Za-z0-9、。,.!?！？]+", "", new)
    return difflib.SequenceMatcher(None, a, b).ratio() >= 0.6


class Relistener:
    """1本の裏の担当。submit した区間を順に whisper で聞き、戻ったら on_fixed(job, 新しい文)。
    戻らなければ on_done(job)。新しい発言の邪魔をしないよう、busy() が真の間は待ってから聞く。"""

    def __init__(self, make_ear: Callable[[], object], on_fixed: Callable[[dict, str], None],
                 on_done: Callable[[dict], None], busy: Callable[[], bool] = lambda: False,
                 clean: Callable[[str, str], str] = lambda t, lang: t,
                 max_age_s: float = 30.0, max_wait_busy_s: float = 3.0):
        self.make_ear = make_ear
        self.on_fixed = on_fixed
        self.on_done = on_done
        self.busy = busy
        self.clean = clean
        self.max_age_s = max_age_s
        self.max_wait_busy_s = max_wait_busy_s
        self.q: "queue.Queue[dict]" = queue.Queue()
        self.stop = threading.Event()
        self.ear = None
        self.stats = {"asked": 0, "fixed": 0, "kept": 0, "stale": 0, "sec": []}
        self._th = threading.Thread(target=self._run, daemon=True)
        self._th.start()

    def warm(self) -> None:
        """耳を先に読む(最初の聞き直しで読み込みを待たない)。裏のスレッドで呼ぶ。"""
        self.q.put({"_warm": True})

    def submit(self, job: dict) -> None:
        job.setdefault("t_submit", time.time())
        self.stats["asked"] += 1
        self.q.put(job)

    def close(self) -> None:
        self.stop.set()

    def _ear(self):
        if self.ear is None:
            self.ear = self.make_ear()
        return self.ear

    def _run(self) -> None:
        while not self.stop.is_set():
            try:
                job = self.q.get(timeout=0.2)
            except queue.Empty:
                continue
            if job.get("_warm"):
                try:
                    self._ear()
                except Exception as e:  # noqa: BLE001
                    print(f"[relisten] whisper を読めません: {e}", flush=True)
                continue
            t_wait = time.time()
            while self.busy() and time.time() - t_wait < self.max_wait_busy_s and not self.stop.is_set():
                time.sleep(0.05)      # 新しい発言を聞いている・訳している間は待つ
            if time.time() - job["t_submit"] > self.max_age_s:
                self.stats["stale"] += 1
                self._done(job)
                continue
            t0 = time.time()
            try:
                text, lang, _p = self._ear().transcribe(job["pcm"])
            except Exception as e:  # noqa: BLE001 - 聞き直しの失敗は元の文のまま
                print(f"[relisten] 失敗 {type(e).__name__}: {e}", flush=True)
                text, lang = "", ""
            sec = time.time() - t0
            self.stats["sec"].append(round(sec, 2))
            new = self.clean(text, job["lang"]) if text else ""
            if lang == job["lang"] and recovered(job["orig"], new):
                self.stats["fixed"] += 1
                print(f"[relisten {sec:.2f}s] 略語が戻った: {job['orig'][:24]} → {new[:32]}", flush=True)
                try:
                    self.on_fixed(job, new)
                except Exception as e:  # noqa: BLE001
                    print(f"[relisten] 差し替えに失敗 {type(e).__name__}: {e}", flush=True)
                    self._done(job)
            else:
                self.stats["kept"] += 1
                print(f"[relisten {sec:.2f}s] そのまま: {job['orig'][:24]}", flush=True)
                self._done(job)

    def _done(self, job: dict) -> None:
        try:
            self.on_done(job)
        except Exception:  # noqa: BLE001
            pass
