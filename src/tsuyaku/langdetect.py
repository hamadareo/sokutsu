"""文字種から見た目で言語を判定する(whisperの出力に対する後処理)。ja/en/zh/ko専用。"""
from __future__ import annotations

import difflib
import re

HANGUL = re.compile(r"[가-힯ᄀ-ᇿ〮-〯]")
KANA = re.compile(r"[ぁ-ゖァ-ヺ]")
HAN = re.compile(r"[一-鿿豈-﫿]")
LATIN = re.compile(r"[A-Za-z]")
_PUNCT = re.compile(r"[\W_]+", re.UNICODE)


_LATIN_WORD = re.compile(r"[A-Za-z]+(?:['’-][A-Za-z]+)*")

# 中文の機能文字。中国語の自然な文はほぼ必ずこのどれかを含む。
# 11/05ライブログで「英語。」「以上」が kana 無しを理由に zh と誤判定された
# (漢字だけの日本語は zh との文字種の区別がつかない)。これらの字が1つもない
# 漢字のみテキストは日本語の読み上げ候補とみなす(「以上」「実際」「勉強」等)。
# 日本語でも使われやすい 的/以 は入れていない(「以上の通り」等が zh に化ける)。
_ZH_FUNC = re.compile(r"[了吗呢吧啊呀么们这那什怎谁被把让很就还过在有是我不没儿样他她它时]")

# 簡体字・広東語にしか無い字(日本語の文には出ない)。11/08 このMacの耳で試した日中の会話: 中国語の
# 発話「你 just.」のように漢字と英字が混じる短い文が ja になり、中国語を中国語へ訳していた。
# 日本語の文(FLEURS ja 321文・SenseVoice の ja 143文・擬似会議 1199行・試験文 ほか 2206行)に
# 1度も出ない字だけ。来・会・点・度・係・喝 は日本語に出たので入れない。没・那・个・几・吃 も日本語で使うので入れない。
ZH_ONLY = re.compile(
    r"[们这么吗呢吧啊呀哦嗯嘛呗哪说你您谁怎让给对还过时别该为啥咱俩话语请谢问题欢见现东车门长开关发"
    r"间进边运动电脑网页买卖钱样难应佢嘅咗冇唔喺哋嘢啲乜咁嚟啦咩噉睇揾畀攞嗰]")


def detect_lang(text: str) -> str:
    """ハングル→ko、かな主体→ja、漢字のみ→zh、ラテン→en。判定不能なら空文字。

    かながあってもラテン語の方が多い場合は英語寄りとみなす(whisperが英語音声に
    かなを1-2字混入させる誤認対策)。逆に英専門用語が多い日本語でも、かな+漢字が
    英単語数を上回れば日本語とる。
    漢字のみでも中文の機能文字(了/吗/这等 _ZH_FUNC)が1つも無ければ日本語とみなす。
    """
    if HANGUL.search(text):
        return "ko"
    kana = len(KANA.findall(text))
    han = len(HAN.findall(text))
    latin_words = len(_LATIN_WORD.findall(text))
    if kana:
        # 中国語の文にかなは現れないので、かながある時点でja/enのどちらか。
        # 割合でjaを取り、日本語が薄い(=whisperノイズ)なら英語とみなす。
        return "ja" if (kana + han) >= latin_words + 1 else "en"
    if han:
        # 11/05実測: 「英語。」「以上」は whisper が日本語音声から出した漢字のみ
        # テキスト。中文の機能文字を1字も含まないなら日本語として扱う。
        # 英字が混じっても、簡体字・広東語にしか無い字があれば中国語(「你 just.」)
        if ZH_ONLY.search(text):
            return "zh"
        return "zh" if _ZH_FUNC.search(text) else "ja"
    if latin_words:
        return "en"
    return ""


def resolve_lang(text: str, heard: str = "") -> str:
    """文字で言語を決め、漢字だけの文は whisper が音から決めた言語(heard)に従う。

    11/06 日中の実会話: 中国語の発話「谢谢」「最棒」「一陣子一陣子」「因人禁咖」が
    中文の機能文字を含まないため文字判定で ja になり、ja→zh(中国語→中国語)の
    素通し訳になっていた。漢字だけでは日中を文字から決められない。リモートサーバーの耳は
    音声から言語を選んでいる(今日のペアに絞り済み)ので、そちらが正しい。
    """
    d = detect_lang(text)
    if d == "zh" and ZH_ONLY.search(text) and not KANA.search(text):
        return "zh"     # 簡体字・広東語にしか無い字: 耳が ja と言っても中国語(11/08)
    if (heard in ("ja", "zh") and d in ("ja", "zh") and HAN.search(text)
            and not KANA.search(text) and not _LATIN_WORD.search(text)):
        return heard
    return d


def norm(text: str) -> str:
    return _PUNCT.sub("", text or "")


def is_echo(prev_spoken: str, new_text: str, threshold: float = 0.6) -> bool:
    """TTSで読み上げた直後、マイクが自分の声を拾って同じような文を返したか。"""
    a, b = norm(prev_spoken), norm(new_text)
    if not a or len(b) < 2:
        return False
    return difflib.SequenceMatcher(None, a, b).ratio() >= threshold
