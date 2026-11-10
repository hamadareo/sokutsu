"""話のペースに合わせて読み上げの速さを決める(11/08「話のペースに合わせて話すスピードを自動で
変えるモード」「中文の翻訳で話のペースに全然ついていけていない」)。

見つけた問題(実記録 run.log、中国語の速い話者 → 日本語の読み上げ):
  訳は0.3〜0.6秒で出ているのに、声は話し終わりから4〜7秒後に鳴り始め、9〜17秒遅れた文は丸ごと
  飛ばしていた。日本語の訳は中国語の原文の約2倍の字数なので、決まった速さでは速い話者に追いつけない。

方式: 1つ読むごとに、
  使える時間 = その原文を話すのにかかった秒数(話者のペース) − 今たまっている読み上げの秒数
  読むのにかかる秒数(×1.0)= 字数 ÷ 1秒あたりの字数(声と言語ごと。鳴った長さで少しずつ合わせる)
  必要な速さ = 読むのにかかる秒数 ÷ 使える時間
を求め、ユーザーの速さ(下限)〜聞き取れる上限(声ごとに測った値)の中に収める。速さは急に変えず、
上げるときは速く・下げるときはゆっくり寄せる。
"""
from __future__ import annotations

import re
import threading

# 速さ×1.0 での1秒あたりの字数(英語・韓国語は語数)。11/08 このMacで測った値(scratchpad の
# tts_speed.py: FLEURS の文を各声で合成して長さを測る)。鳴った長さで少しずつ合わせる。
CPS0 = {
    ("voicevox", "ja"): 6.55, ("say", "ja"): 5.61, ("kokoro", "ja"): 6.0,
    ("kokoro", "zh"): 4.3, ("say", "zh"): 4.63,
    ("kokoro", "en"): 2.86, ("say", "en"): 2.66,
    ("say", "ko"): 5.75, ("kokoro", "ko"): 5.75,
}
# 聞き取れる上限: このMacの SenseVoice で聞き直した誤り(字・語の誤り率)が ×1.0 より5ポイント以上
# 増えない最も速い速さ(11/08 測定、FLEURS の文8つずつ、eval/pace/tts_speed.json)。
#   VOICEVOX 13 ja ×1.0 8.1% → ×1.25 10.8% → ×1.5 16.5%、Kokoro zh ×1.25 6.6% → ×1.5 22.0%、
#   Kokoro en ×1.25 7.3% → ×1.5 23.4%、say Kyoko ×2.0 11.5%(×1.0 6.5%)、say Tingting ×2.5 4.4%、
#   say Samantha ×2.0 9.7%(5.6%)、say Yuna ×1.75 12.8%(9.3%)
CAPS = {("voicevox", "ja"): 1.25, ("say", "ja"): 2.0, ("kokoro", "zh"): 1.25, ("say", "zh"): 2.5,
        ("kokoro", "en"): 1.25, ("say", "en"): 2.0, ("say", "ko"): 1.75}
DEFAULT_CAP = 1.25
# 「速さ優先」(11/08 リーダー): VOICEVOX・Kokoro は ×1.5 まで許す(機械の聞き取りの誤りは VOICEVOX で
# 8% → 17% に増えるが、速い話者の飛ばしが減る)。say の声は ×1.5 より上まで測った上限があるのでそのまま。
FAST_CAP = 1.5
FAST_ENGINES = ("voicevox", "kokoro")
PRIORITIES = ("clear", "fast")      # 聞きやすさ優先(既定)/ 速さ優先

# 分からない出力(USB・仮想の機器)でも、名前が耳に付けるものなら聞き手だけに届く
_EARS = re.compile(r"head ?(set|phone)|ear ?(phone|bud)|airpods|\bbuds\b|"
                   r"ヘッドセット|ヘッドホン|ヘッドフォン|イヤホン|イヤフォン", re.I)


def ears_only(private: "bool | None", name: str = "") -> bool:
    """読み上げの声が聞き手の耳だけに届く(マイクに回り込まない)か。private は
    devices.output_is_private の答え(True/False はそのまま)。None(USB・仮想など分からない)は
    USB スピーカーもあるので止める側に倒し、名前がヘッドセット・ヘッドホン・イヤホンのときだけ True。"""
    if private is not None:
        return bool(private)
    return bool(_EARS.search(name or ""))


def units(text: str, lang: str) -> int:
    """読み上げの量: 英語は語数、日本語・中国語・韓国語は字数(記号と空白を除く。韓国語は音節)。"""
    t = text or ""
    if lang == "en":
        return len(re.findall(r"[\w']+", t))
    return len(re.sub(r"[\s\W_]+", "", t))


def engine_of(voice_spec: str) -> str:
    kind = (voice_spec or "").split(":", 1)[0]
    return kind if kind in ("voicevox", "kokoro", "say") else "say"


class PaceController:
    """スレッド安全。choose() は読み上げワーカーから、observe() は鳴り終わったときに呼ぶ。"""

    def __init__(self, caps: "dict | None" = None, cps0: "dict | None" = None,
                 rise: float = 0.6, fall: float = 0.25, min_budget_s: float = 0.6,
                 keep: float = 1.0, learn: float = 0.2, priority: str = "clear"):
        self.caps = dict(CAPS)
        self.caps.update(caps or {})
        self.cps = dict(CPS0)
        self.cps.update(cps0 or {})
        self.rise = rise            # 上げるときの寄せ方(0〜1。1 = すぐその速さ)
        self.fall = fall            # 下げるときの寄せ方
        self.min_budget_s = min_budget_s
        self.keep = keep            # 原文の時間の何倍までで読み終えたいか
        self.learn = learn          # 鳴った長さで1秒あたりの字数を合わせる強さ
        self.priority = priority    # "clear" 聞きやすさ優先 / "fast" 速さ優先
        self.cur: dict = {}         # (声, 言語) → 今の速さ
        self._lock = threading.Lock()

    def cap(self, engine: str, lang: str) -> float:
        c = float(self.caps.get((engine, lang), DEFAULT_CAP))
        if self.priority == "fast" and engine in FAST_ENGINES:
            c = max(c, FAST_CAP)
        return c

    def predict(self, text: str, engine: str, lang: str, speed: float = 1.0) -> float:
        """その速さで読むと何秒か。"""
        cps = self.cps.get((engine, lang)) or CPS0.get(("say", lang), 4.0)
        return units(text, lang) / max(0.1, cps * max(0.1, speed))

    def observe(self, text: str, engine: str, lang: str, speed: float, dur_s: float) -> None:
        """実際に鳴った長さで、1秒あたりの字数を少しずつ合わせる。"""
        n = units(text, lang)
        if n < 3 or dur_s <= 0.3 or speed <= 0:
            return
        got = n / (dur_s * speed)
        with self._lock:
            k = (engine, lang)
            old = self.cps.get(k, got)
            self.cps[k] = old + self.learn * (got - old)

    def choose(self, text: str, engine: str, lang: str, src_dur_s: float, backlog_s: float,
               base: float = 1.0) -> float:
        """この文を読む速さ。base(ユーザーの速さ)より遅くはしない。"""
        need_t = self.predict(text, engine, lang, 1.0)
        budget = max(self.min_budget_s, self.keep * max(0.0, src_dur_s) - max(0.0, backlog_s))
        target = max(base, need_t / budget)
        cap = max(base, self.cap(engine, lang))
        target = min(target, cap)
        with self._lock:
            k = (engine, lang)
            cur = self.cur.get(k, base)
            a = self.rise if target > cur else self.fall
            cur = cur + a * (target - cur)
            cur = max(base, min(cap, cur))
            self.cur[k] = cur
        return round(cur, 2)

    def current(self, engine: str, lang: str, base: float = 1.0) -> float:
        with self._lock:
            return round(self.cur.get((engine, lang), base), 2)
