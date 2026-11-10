"""LM Studio (OpenAI互換 /v1/chat/completions) のストリーミング翻訳クライアント。"""
from __future__ import annotations

import json
import os
import time
from typing import Callable, Iterable

import requests

LANG_NAMES = {"ja": "日本語", "en": "英語", "zh": "中国語", "ko": "韓国語"}


def out_lang(pair: tuple[str, str], lang: str) -> str:
    """その言語の発話を何語へ訳すか。ペア内なら相手側、ペア外はja、無ければen、さらに無ければpair[0]。"""
    a, b = pair
    if lang == a:
        return b
    if lang == b:
        return a
    if "ja" in pair:
        return "ja"
    if "en" in pair:
        return "en"
    return pair[0]


# 指示文は英語で書く(11/06 FLEURS 多言語評価)。日本語の指示文だと
#   zh→ja で中国語をそのまま返す(16/41)・en→zh で日本語を出す(7/39、空履歴だと27/41)。
#   指示文の言語に出力が引っ張られる。英語の指示文で同じ単位を訳し直すと 0件 と 0〜1件。
EN_NAMES = {"ja": "Japanese", "en": "English", "zh": "Simplified Chinese", "ko": "Korean"}


def build_system(pair: tuple[str, str], topic: str = "", src: str = "") -> str:
    """会話のあいだ1字も変わらない指示文(11/06)。

    見つけた問題: このMac(LM Studio, qwen3-4b)で交互に話す会話を訳すと、1回1.06秒の
    うち初字まで0.92秒。同じ依頼の2回目は初字0.14〜0.29秒 = 指示文の読み込み(prefill)
    待ち。以前は指示文の先頭近くに「この発話は X 語、Y 語へ」を入れていたので、
    話し手が替わるたびに先頭から読み直しになっていた(サーバーの使い回し=前方一致)。
    向き・今の発話は末尾(build_messages)へ移し、指示文には両方向の対応だけを書く。
    src は互換のため受けるが使わない。
    """
    a, b = pair
    A, B = EN_NAMES[a], EN_NAMES[b]
    other = EN_NAMES[out_lang(pair, "_")]
    topic_line = f"\nTopic, names and technical terms: {topic}" if topic else ""
    return (
        f"You are a professional simultaneous interpreter at a face-to-face meeting "
        f"between {A} and {B} speakers.{topic_line}\n"
        "Rules:\n"
        f"- Translate only the final utterance (after \"Utterance to translate\"). "
        f"{A} utterances go into {B}; {B} utterances go into {A}; "
        f"utterances in any other language go into {other}.\n"
        "- Output only the translation in the target language. Never copy the source text, "
        "and never answer questions, add notes, preambles or quotation marks.\n"
        "- If the utterance is short or unfinished, render it naturally as a phrase.\n"
        + (NO_ADD_LINE if NO_ADD_RULE else "- Always finish the sentence. ") +
        "A one-word output is wrong unless the utterance is one word.\n"
        "- Match the speaker's formality. Keep technical terms consistent across the conversation."
        + (shots_block(pair) if FEWSHOT else "")
    )


# 専門用語の手がかり(terms.py)を訳の依頼に足すか(評価の切り替え用)
USE_GLOSSARY = True
# 今の発話とよく似た(同じではない)前の発話を履歴にどう載せるか(11/08 1字違いの似た語の訳の写し):
# "keep" = 今までどおり / "drop" = 載せない / "warn" = 載せて「似ているが別の文」と添える。
# 似た2文の続けての訳(eval/terms/confusables.json、264組)で、2文目が正しい割合
# remote 94.3 → 97.7%(写し 6 → 1)、このMac gemma-3-4b 83.0 → 86.7%(写し 15 → 2)(用語の手がかりと合わせて)
SIMILAR_HIST = "drop"
SIMILAR_TH = 0.75     # 似た2文の組(confusables)は 0.80 以上、FLEURS の前後7文どうしは 0 組が 0.6 以上


# 足さない・落とさない(11/08 このMac gemma-3-4b: 「周六见。」に「昼」を足す、条件の断片に続きを創作する)
NO_ADD_RULE = True
# 前の「Always finish the sentence.」は断片に続きを作らせる向きに働くので、この行に置き換える
NO_ADD_LINE = ("- Translate a fragment as the same fragment (an unfinished \"if ...\" or \"because ...\" clause stays "
               "unfinished). Never add days, times, places, reasons or other details that are not in the utterance, "
               "and never drop information that is in it. Translate the whole utterance; do not stop partway. ")


def _flag(name: str, default: str) -> str:
    """評価の切り替え用(環境変数 TSU_*)。アプリでは既定のまま。"""
    return os.environ.get(name, default)


# 断片・短文の例(11/10 mac1): 指示文の末尾に固定で置く(会話の間は変わらないので先頭一致の使い回しは効く)。
# 「断片は断片のまま」「足さない」を言葉だけでなく例で見せる。評価の断片(eval/fragments)と文は重ねない。
FEWSHOT = _flag("TSU_FEWSHOT", "0") == "1"      # 系: 指示文の末尾に例
SHOTTURNS = 0                                   # 今の会話で使う例の件数(configure_for_model が決める)
SHOTS_SMALL = 4                                 # 小さいモデル(gemma 系)の既定の件数
_SHOTS_ENV = os.environ.get("TSU_SHOTTURNS")    # 評価用の上書き(0 で無し)


def configure_for_model(model: str) -> None:
    """会話で使う訳モデルに合わせる(11/10 mac1)。gemma-3-4b(このMac・配る版の同梱)は、断片・短文の例を
    user/assistant の往復で指示文の直後に置くと足し・抜けが減る。大きいモデル(リモートサーバーの Qwen)には足さない。"""
    global SHOTTURNS
    if _SHOTS_ENV is not None:
        SHOTTURNS = int(_SHOTS_ENV)
    else:
        SHOTTURNS = SHOTS_SMALL if "gemma" in (model or "").lower() else 0
_SHOTS = [
    {"en": "If the budget gets approved,", "ja": "予算が承認されたら、", "zh": "如果预算获批的话，", "ko": "예산이 승인되면,"},
    {"en": "See you Friday.", "ja": "金曜日にね。", "zh": "周五见。", "ko": "금요일에 봐요."},
    {"en": "Because the server was down,", "ja": "サーバーが落ちていたので、", "zh": "因为服务器宕机了，", "ko": "서버가 다운돼서,"},
    {"en": "Thanks, that helps.", "ja": "ありがとう、助かります。", "zh": "谢谢，帮大忙了。", "ko": "고마워요, 도움이 됐어요."},
    {"en": "By the time I get home,", "ja": "家に着くころには、", "zh": "等我到家的时候，", "ko": "집에 도착할 때쯤에는,"},
    {"en": "The new office opens on March 3, and about forty people will move in.",
     "ja": "新しいオフィスは3月3日にオープンし、約40人が入居する予定です。",
     "zh": "新办公室3月3日启用，大约有四十人入驻。", "ko": "새 사무실은 3월 3일에 문을 열고 약 40명이 입주할 예정입니다."},
    {"en": "Okay, sounds good.", "ja": "はい、いいですね。", "zh": "好的，可以。", "ko": "네, 좋아요."},
    {"en": "I'll look into it tomorrow morning.", "ja": "明日の朝に調べてみます。", "zh": "我明天早上查一下。", "ko": "내일 아침에 알아볼게요."},
]


def shot_turns(pair: tuple[str, str], n: int) -> list[dict]:
    """例を user/assistant の往復にして並べる(指示文の直後・履歴の前。会話の間は変わらない)。"""
    a, b = pair
    out = []
    for sh in _SHOTS[:n]:
        for x, y in ((a, b), (b, a)):
            out.append({"role": "user", "content": f"This utterance is in {EN_NAMES[x]}. Translate it into {EN_NAMES[y]}.\n"
                        f"Utterance to translate ({EN_NAMES[x]} -> {EN_NAMES[y]}):\n{sh[x]}"})
            out.append({"role": "assistant", "content": sh[y]})
    return out


def shots_block(pair: tuple[str, str]) -> str:
    a, b = pair
    lines = []
    for sh in _SHOTS:
        lines.append(f"{sh[a]} => {sh[b]}")
        lines.append(f"{sh[b]} => {sh[a]}")
    return "\nExamples of faithful translation (fragments stay fragments, nothing added):\n" + "\n".join(lines)


HKEEP = int(_flag("TSU_HKEEP", "4"))
# 文末が閉じていない短い発話に「途中かもしれない。言われた分だけ訳す」と添える(11/10 mac1)
FRAGHINT = _flag("TSU_FRAGHINT", "0") == "1"


def history_window(n: int, keep: int = 0) -> int:
    """履歴の何番目から載せるか。直近 keep〜2keep-1 件を、keep 件ごとにまとめて
    ずらす(1件ごとにずらすと毎回先頭が変わり、読み込みの使い回しが効かない)。"""
    keep = keep or HKEEP
    return max(0, n - keep) // keep * keep


def build_messages(pair: tuple[str, str], topic: str, history: list[dict],
                   current: str = "", src: str = "", terms=()) -> list[dict]:
    """history: [{"lang":"en","orig":"...","trans":"..."}, ...] 古い→新しい。
    current: 今回の発話(末尾の user メッセージになる)。src: その検出言語。

    履歴は user/assistant のマルチターンに「しない」。11/05実測: assistant
    ターンに破綻訳が1個でも混ざると qwen がそれを真似て訳を短打ちする
    (「Particles such」単独でも汚染、2個で 'I' の1語に)。履歴は「参考」と
    明記した1本のユーザーメッセージへ畳み、今回の発話をその末尾に置く。
    原文の写しになった訳(11/06 zh→ja)も参考に載せない: 載せると次の写しを呼ぶ。
    並びは「変わらない指示文 → 増えていく履歴 → 向きと今の発話」(11/06): 前から
    順に一致する部分はサーバーが読み込み済みのものを使える。

    専門用語(11/08、terms.py): 今の発話の字名の並び(エーアイ)を英字に戻し、内蔵の用語集と
    利用者の用語辞書(terms)・テーマに出てくる語だけを「この訳を使う」1行で今の発話の直前に足す
    (指示文と履歴は変えないので読み込みの使い回しは効いたまま)。
    """
    from .guards import _TERM_SPLIT, looks_truncated, wrong_language  #循環防止: 関数内import
    from .terms import current_domain, glossary_hint, normalize_letters, similar
    if current and src:
        current = normalize_letters(current, src)
    near = False
    msgs = [{"role": "system", "content": build_system(pair, topic)}]
    if SHOTTURNS:
        msgs += shot_turns(pair, SHOTTURNS)
    hist: list[str] = []
    for h in history[history_window(len(history)):]:
        orig, trans = h.get("orig", ""), h.get("trans", "")
        if not orig or not trans or looks_truncated(orig, trans):
            continue  # 明らかな破綻ターン(1〜3語訳)は参考にも載せない
        if wrong_language(orig, trans, out_lang(pair, h.get("lang", "")) if h.get("lang") else ""):
            continue
        if (current and SIMILAR_HIST != "keep" and orig.strip() != current.strip()
                and similar(orig, current) >= SIMILAR_TH):
            near = True
            if SIMILAR_HIST == "drop":
                continue
        hist.append(f"{orig} => {trans}")
    head = ""
    if src and src in EN_NAMES:
        X, Y = EN_NAMES[src], EN_NAMES[out_lang(pair, src)]
        head = (f"This utterance is in {X}. Translate it into {Y}.\n"
                f"Utterance to translate ({X} -> {Y}):\n")
    hint = ""
    if current and src in EN_NAMES and USE_GLOSSARY:
        user = list(terms or ()) + [t for t in _TERM_SPLIT.split(topic or "") if t.strip()]
        # IT の話かの見当に直前の2発話も使う(一般の文で意味が割れる形 = process などは IT の話の中だけ)
        ctx = " ".join(h.get("orig", "") for h in history[-2:])
        hint = glossary_hint(current, src, out_lang(pair, src), user=user, domain=current_domain(),
                             context=ctx)
    if FRAGHINT and current and src in EN_NAMES and current.strip()[-1:] not in "。．.！!？?」』)）":
        head = ("Note: this utterance may be unfinished. Translate exactly what was said; "
                "do not complete it or add anything.\n") + head
    if near and SIMILAR_HIST == "warn":
        head = ("Note: an earlier utterance looks similar but is a different sentence; "
                "translate this one exactly as written.\n") + head
    if hint:
        head = hint + "\n" + head
    if hist:
        body = "Earlier conversation (context only, do not translate):\n" + " / ".join(hist)
        if current:
            body += "\n\n" + (head or "Utterance to translate:\n") + current
        msgs.append({"role": "user", "content": body})
    elif current:
        msgs.append({"role": "user", "content": f"{head}{current}"})
    return msgs


def target_lang(pair: tuple[str, str], lang: str) -> str | None:
    """その言語の発話をどの言語へ読むべきか。ペア外ならNone(字幕のみ)。"""
    a, b = pair
    if lang == a:
        return b
    if lang == b:
        return a
    return None


def translate_stream(base_url: str, model: str, messages: list[dict],
                     on_delta: Callable[[str], None] | None = None,
                     timeout: float = 30.0) -> str:
    """ストリーミング翻訳。最終文言を返す。on_delta に逐次テキストを渡す。

    reasoning_effort="none": 推論系モデル(qwen3.8-flash-next)が思考トークンを
    垂れ流して max_tokens を食い尽くし、本文が空になる不具合対策(11/05実測、
    リモートサーバーではこれで思考が止まり本文が即ストリーム開始)。非対応サーバーは
    400 を返すので、その場合だけ一度だけ思考ありでリトライする。
    """
    payload = {
        "model": model,
        "messages": messages,
        "stream": True,
        "temperature": float(_flag("TSU_TEMP", "0.2")),
        "max_tokens": 1024,
        "reasoning_effort": "none",
    }
    from . import localllm   # 配る版: 同梱エンジンが会議の途中で落ちたら起こし直す(本人の版では何もしない)
    try:
        r = requests.post(f"{base_url}/v1/chat/completions", json=payload,
                          stream=True, timeout=timeout)
        if r.status_code == 400:
            payload.pop("reasoning_effort", None)  # 古いサーバーは未知フィールドで400
            r = requests.post(f"{base_url}/v1/chat/completions", json=payload,
                              stream=True, timeout=timeout)
        r.raise_for_status()
    except requests.RequestException as e:
        localllm.note_failure(base_url)
        from . import tunnel   # リモートサーバーの転送が切れていたら裏で張り直す(11/08 speed)
        tunnel.note_failure(base_url, e)
        raise
    localllm.note_ok(base_url)
    out: list[str] = []
    try:
        _read_stream(r, out, on_delta)
    finally:
        # 途中で止めた(先読みの外れ = on_delta が例外)ときも接続をすぐ閉じる。開けたままだと
        # LM Studio(同時に1件)は止めた訳を最後まで作り続け、次の訳を待たせた(11/08 speed)
        try:
            r.close()
        except Exception:  # noqa: BLE001
            pass
    return "".join(out).strip()


def _read_stream(r, out: list, on_delta) -> None:
    for line in r.iter_lines():
        if not line:
            continue
        line = line.decode("utf-8", "replace")
        if not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if payload == "[DONE]":
            break
        try:
            obj = json.loads(payload)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and "error" in obj:
            # サーバーが明示的にエラーを返しているのに「空訳」として無言で
            # 終わるのを許さない(11/05 15:42 の「こんにちは」失敗はこれ)。
            raise RuntimeError(str(obj.get("error"))[:200])
        try:
            delta = obj["choices"][0]["delta"].get("content")
        except (KeyError, IndexError, TypeError):
            continue
        if delta:
            out.append(delta)
            if on_delta:
                on_delta(delta)


def translate_with_retry(base_url: str, model: str, messages: list[dict],
                         on_delta: Callable[[str], None] | None = None,
                         tries: int = 2, delay: float = 0.4,
                         sleep=time.sleep) -> str:
    """空訳・一時的失敗を少し待って1回だけやり直す。全部ダメなら "" を返す。

    11/05 15:42 ライブ実障害: 「こんにちは」等6文が「（翻訳できませんでした）」
    になったが、同じ発話を後で叩くと全部正しく訳せた = サーバーの一時的混線。
    空訳/例外のとき 0.4 秒だけ待って1回だけ再試行すれば救える。無限リトライは
    会話を待たせるので禁止。全部空なら "" を返し、呼び出し側が正直に失敗を出す。
    """
    trans = ""
    for i in range(max(1, tries)):
        try:
            trans = translate_stream(base_url, model, messages, on_delta)
        except Exception as e:  # noqa: BLE001 — 混ぜるな危険のリトライ、種別は問わない
            print(f"[tr-err] 翻訳失敗({type(e).__name__}): {str(e)[:120]}", flush=True)
            trans = ""
        if trans:
            return trans
        if i + 1 < tries and delay:
            sleep(delay)
            from . import tunnel
            ev = tunnel.repairing()
            if ev is not None:      # リモートサーバーの転送を張り直している: 終わるのを待ってからやり直す
                ev.wait(tunnel.WAIT_S)
    return trans


def warmup(base_url: str, model: str, timeout: float = 180.0) -> None:
    """初回リクエストでモデルをロードさせるための軽い ping。"""
    try:
        requests.post(
            f"{base_url}/v1/chat/completions",
            json={"model": model, "messages": [{"role": "user", "content": "ping"}],
                  "max_tokens": 4, "stream": False},
            timeout=timeout,
        )
    except requests.RequestException:
        pass


def warm_prompt(base_url: str, model: str, pair: tuple[str, str], topic: str = "", terms=(),
                timeout: float = 60.0) -> None:
    """会話の指示文を先にサーバーへ読ませる(11/08 speed)。最初の1文の依頼と前から一致する所
    (指示文 + 向きの行)まで同じにして、1字だけ答えさせる。サーバーは前方一致の読み込みを使い回すので、
    最初の訳の初字が指示文の読み込みを待たない。失敗しても何もしない。"""
    msgs = build_messages(pair, topic, [], "OK", pair[0], terms=terms)
    try:
        requests.post(f"{base_url}/v1/chat/completions",
                      json={"model": model, "messages": msgs, "max_tokens": 1, "stream": False,
                            "temperature": 0.2, "reasoning_effort": "none"},
                      timeout=timeout)
    except requests.RequestException:
        pass


def server_models(base_url: str, timeout: float = 3.0) -> list[str]:
    r = requests.get(f"{base_url}/v1/models", timeout=timeout)
    r.raise_for_status()
    return [m["id"] for m in r.json().get("data", [])]


def pick_model(base_url: str, preferred: str) -> str:
    """preferred があればそれ、無ければサーバーの最初のモデル。"""
    ids = server_models(base_url)
    for i in ids:
        if i == preferred or i.startswith(preferred):
            return i
    if not ids:
        raise RuntimeError("LM Studio にモデルがありません")
    return ids[0]
