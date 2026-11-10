"""用語辞書(W3)の守り2つ(11/07 用語辞書の A/B で見つけた崩れ)。

1. 復唱を捨てる: 雑音の区間を リモートサーバーの耳が「A0、山本、伊藤、中村、 nord、山跳森也。」と聞き、
   辞書に無い長めの項目が混ざったので echoes_prompt をすり抜けた → 辞書の語が3つ以上並んだら捨てる。
2. 中身が消えたら聞き直す: 3.7秒の区間が手がかりの語「渡辺さん。」だけになり、中身(「それまでに必要な
   台数を見積もっておきます」)が消えた → 2.5秒以上の区間で結果が辞書の語(+さん/様など)だけなら、
   手がかりなしで1回だけ聞き直し、長い方を採る。

実行: PYTHONPATH=src .venv/bin/python -m unittest tests.test_terms_guards -v
"""
import queue
import threading
import time
import unittest

import numpy as np

from tsuyaku.guards import echoes_prompt, stt_prompt, terms_only

PROMPT = stt_prompt("", terms=["A0", "山本", "伊藤", "中村", "懇親会"])
PROMPT4 = stt_prompt("", terms=["高橋", "渡辺", "小川", "森"])


class TestEchoesPrompt(unittest.TestCase):
    def test_three_terms_in_a_row_with_junk_is_dropped(self):
        # 11/07 ja_booth でそのまま記録に入った行
        assert echoes_prompt("A0、山本、伊藤、中村、 nord、山跳森也。", PROMPT)
        assert echoes_prompt("山本、伊藤、中村、懇親会、それから資料。", PROMPT)

    def test_old_cases_still_dropped(self):
        assert echoes_prompt("山本、伊藤、中村、懇親会。", PROMPT)
        assert echoes_prompt("伊藤、中村、懇親会。", PROMPT)

    def test_real_sentences_are_kept(self):
        # 語が文の中に散っている
        assert not echoes_prompt("山本さんと伊藤さんと中村さんで決めました。", PROMPT)
        # 辞書の語を2つ含む普通の文
        assert not echoes_prompt("ポスターはA0、印刷は伊藤さんが20日まで。", PROMPT)
        assert not echoes_prompt("山本です。伊藤さん、ポスターのサイズはどうしましょうか。", PROMPT)
        # 2語 + 長い本文(今までどおり残す)
        assert not echoes_prompt("A0、山本さんの案で印刷することにします。", PROMPT)
        assert not echoes_prompt("YOLO、Whisper、Qwenを使いました",
                                 stt_prompt("", terms=["YOLO", "Whisper", "Qwen"]))

    def test_no_terms_never_drops(self):
        assert not echoes_prompt("山本、伊藤、中村、懇親会。", "")


class TestTermsOnly(unittest.TestCase):
    def test_name_with_honorific_only(self):
        assert terms_only("渡辺さん。", PROMPT4)
        assert terms_only("渡辺様", PROMPT4)
        assert terms_only("高橋、渡辺。", PROMPT4)
        assert terms_only("森さん!", PROMPT4)

    def test_real_content_is_not_terms_only(self):
        assert not terms_only("渡辺さん、よろしくお願いします。", PROMPT4)
        assert not terms_only("それまでに必要な台数を見積もっておきます。", PROMPT4)
        assert not terms_only("渡辺です。", PROMPT4)

    def test_empty_prompt_or_text(self):
        assert not terms_only("渡辺さん。", "")
        assert not terms_only("", PROMPT4)
        assert not terms_only("。", PROMPT4)


# ---------------- STT ワーカー(呼ぶ側)の聞き直し ----------------

class Resp:
    def __init__(self, obj):
        self.obj = obj

    def raise_for_status(self):
        pass

    def json(self):
        return self.obj


class NoLocal:
    fast = False
    langs = ()

    def transcribe(self, samples):
        return "", "", 0.0


class FastLocal:
    fast = True
    langs = ()

    def transcribe(self, samples):
        return "", "", 0.0


def run_worker(seg_s: float, terms: list, with_prompt: str, without_prompt: str, hedge: bool = False):
    """リモートサーバーの耳の代わり: 手がかり(prompt)を付けたら with_prompt、付けなければ without_prompt を返す。"""
    from tsuyaku.__main__ import Bridge
    from tsuyaku.stt import RemoteSTT
    calls = []

    def post(url, data=None, timeout=None):
        has_prompt = "prompt=" in url
        calls.append(has_prompt)
        return Resp({"text": with_prompt if has_prompt else without_prompt, "lang": "ja", "prob": 0.9})

    b = Bridge()
    b.stt = (RemoteSTT("http://x", FastLocal(), post=post, hedge_s=4.0) if hedge
             else RemoteSTT("http://x", NoLocal(), post=post))
    b.stt.prompt = stt_prompt("", terms=terms)
    b.fine_seg = True
    b.capture = None
    b.running.set()
    seg_q, text_q = queue.Queue(), queue.Queue()
    th = threading.Thread(target=b._stt_worker, args=(seg_q, text_q, ("ja", "en")), daemon=True)
    th.start()
    rng = np.random.default_rng(0)
    seg = (rng.standard_normal(int(seg_s * 16000)) * 0.1).astype(np.float32)
    seg_q.put((seg, 0.0, seg_s, 0.0))
    got = None
    try:
        got = text_q.get(timeout=5)
    except queue.Empty:
        pass
    time.sleep(0.1)
    b.running.clear()
    th.join(2)
    return got, calls, b


class TestRetryWithoutPrompt(unittest.TestCase):
    FULL = "それまでに必要な台数を見積もっておきます。渡辺さん。"

    def test_long_segment_terms_only_is_reheard_without_prompt(self):
        got, calls, b = run_worker(3.7, ["高橋", "渡辺", "小川", "森"], "渡辺さん。", self.FULL)
        assert got is not None and got[0] == self.FULL, got
        assert calls == [True, False]                       # 手がかりあり → なしで1回だけ
        assert b._terms_retry[0] == 1 and b._terms_retry[1] >= 0.0

    def test_hedged_path_also_reheard_without_prompt_and_prompt_kept(self):
        # hedge(別のスレッドで問い合わせを組む)でも、聞き直しだけ手がかりなし。self.stt.prompt は書き換えない
        got, calls, b = run_worker(3.7, ["高橋", "渡辺", "小川", "森"], "渡辺さん。", self.FULL, hedge=True)
        assert got is not None and got[0] == self.FULL, got
        assert calls == [True, False]
        assert b.stt.prompt == stt_prompt("", terms=["高橋", "渡辺", "小川", "森"])

    def test_keeps_longer_result(self):
        # 聞き直しの方が短い(空)なら最初の結果のまま
        got, calls, b = run_worker(3.7, ["高橋", "渡辺"], "渡辺さん。", "")
        assert got is not None and got[0] == "渡辺さん。", got
        assert calls == [True, False]

    def test_short_segment_is_not_reheard(self):
        got, calls, b = run_worker(1.5, ["高橋", "渡辺"], "渡辺さん。", self.FULL)
        assert got is not None and got[0] == "渡辺さん。"
        assert calls == [True]
        assert b._terms_retry[0] == 0

    def test_no_terms_never_reheard(self):
        got, calls, b = run_worker(3.7, [], "渡辺さん。", self.FULL)
        assert calls == [False]                             # 手がかりなし: 1回だけ・聞き直さない
        assert b._terms_retry[0] == 0

    def test_normal_result_not_reheard(self):
        got, calls, b = run_worker(3.7, ["高橋", "渡辺"], self.FULL, "x")
        assert got is not None and got[0] == self.FULL
        assert calls == [True]


if __name__ == "__main__":
    unittest.main()
