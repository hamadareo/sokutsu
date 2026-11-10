"""擬似会議の通しテスト(scripts/eval_meeting.py)の、音声や リモートサーバーを使わない部分のテスト。

実行: PYTHONPATH=src .venv/bin/python -m unittest tests.test_eval_meeting -v
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import eval_meeting as EM  # noqa: E402


class TestScripts(unittest.TestCase):
    def test_scripts_are_consistent(self):
        ids = set()
        for sc in EM.SCRIPTS:
            assert sc["id"] not in ids
            ids.add(sc["id"])
            assert sc["mode"] in ("notes", "both")
            people = sc["people"]
            assert 3 <= len(people) <= 4
            for p in people.values():
                assert p["voice"].startswith(("vv:", "say:"))
                assert p["lang"] in sc["pair"].split("-")
            for ln in sc["lines"]:
                assert ln[0] in people, ln
                assert ln[1]
                assert len(ln) == 2 or ln[2] == "bc"
            for t in sc["todos"]:
                assert t["owner"] in people
            # events(v3): 行の番号が台本の中、やることは画面のボタンと同じ3種類
            for e in sc.get("events", []):
                assert 0 <= e["at"] < len(sc["lines"])
                assert e["do"] in ("mark", "memo", "agenda")
                if e["do"] == "agenda":
                    assert sc.get("agenda") and e["k"] < len(sc["agenda"].splitlines())
            # 台本の全員が少なくとも1回は話す(話者の数の正解になる)
            assert {ln[0] for ln in sc["lines"]} == set(people)


class TestMetrics(unittest.TestCase):
    def test_norm_and_tokens(self):
        assert EM._norm("「Ａ０サイズ」で、印刷！") == "a0サイズで印刷"
        assert EM._toks("I'm Emily, from Boston.", "en") == ["i'm", "emily", "from", "boston"]
        assert EM._toks("はい、了解です。", "ja") == list("はい了解です")

    def test_edit_distance(self):
        assert EM._ed(list("abc"), list("abc")) == 0
        assert EM._ed(list("abc"), list("abd")) == 1
        assert EM._ed([], list("ab")) == 2
        assert EM._ed(list("abcd"), list("ab")) == 2

    def test_best_map_one_to_one(self):
        W = {("S1", "P1"): 10, ("S1", "P2"): 8, ("S2", "P2"): 5, ("S3", "P1"): 1}
        mp = EM._best_map(W, ["S1", "S2", "S3"], ["P1", "P2"])
        assert mp == {"S1": "P1", "S2": "P2"}
        mp2 = EM._best_map({("S1", "P1"): 3, ("S1", "P2"): 7}, ["S1"], ["P1", "P2", "P3"])
        assert mp2 == {"S1": "P2"}


class TestSegSplitReport(unittest.TestCase):
    GOLD = [{"i": 0, "spk": "P1", "start": 1.0, "end": 6.0, "bc": False, "text": "では印刷することにします。"},
            {"i": 1, "spk": "P2", "start": 6.5, "end": 9.0, "bc": False, "text": "はい、了解です。"}]

    def test_dip_cut_then_period_split_is_counted(self):
        ev = [{"kind": "seg", "lo": 0.5, "hi": 4.0, "pre": 0.5, "why": "dip"},
              {"kind": "seg", "lo": 4.1, "hi": 6.1, "pre": 0.1, "why": "silence"},
              {"kind": "seg", "lo": 6.2, "hi": 9.1, "pre": 0.3, "why": "silence"},
              {"kind": "unit", "lo": 0.5, "hi": 4.0, "parts": 1, "text": "では印刷をします。"},
              {"kind": "unit", "lo": 4.1, "hi": 6.1, "parts": 1, "text": "することにします。"},
              {"kind": "unit", "lo": 6.2, "hi": 9.1, "parts": 1, "text": "はい、了解です。"}]
        r = EM.seg_split_report(ev, self.GOLD)
        assert r["splits"] == 1 and r["splits_by"] == {"dip+句点": 1}
        assert r["why"] == {"dip": 1, "silence": 2}
        assert r["examples"][0]["text_b"].startswith("することに")

    def test_joined_or_other_speaker_is_not_a_split(self):
        ev = [{"kind": "seg", "lo": 0.5, "hi": 4.0, "pre": 0.5, "why": "dip"},
              {"kind": "seg", "lo": 4.1, "hi": 6.1, "pre": 0.1, "why": "silence"},
              {"kind": "seg", "lo": 6.2, "hi": 9.1, "pre": 0.3, "why": "silence"},
              {"kind": "unit", "lo": 0.5, "hi": 6.1, "parts": 2, "text": "では印刷することにします。"},
              {"kind": "unit", "lo": 6.2, "hi": 9.1, "parts": 1, "text": "はい、了解です。"}]
        r = EM.seg_split_report(ev, self.GOLD)
        assert r["splits"] == 0          # 同じ行は1単位に繋がった / 次は別の人の行

    def test_split_at_real_sentence_end_is_not_counted(self):
        gold = [{"i": 0, "spk": "P1", "start": 1.0, "end": 6.0, "bc": False, "text": "山本です。今日は準備について決めます。"}]
        ev = [{"kind": "seg", "lo": 0.5, "hi": 2.0, "pre": 0.5, "why": "dip"},
              {"kind": "seg", "lo": 2.2, "hi": 6.0, "pre": 0.1, "why": "silence"},
              {"kind": "unit", "lo": 0.5, "hi": 2.0, "parts": 1, "text": "山本です。"},
              {"kind": "unit", "lo": 2.2, "hi": 6.0, "parts": 1, "text": "今日は準備について決めます。"}]
        r = EM.seg_split_report(ev, gold)
        assert r["splits"] == 0 and r["splits_at_sentence_end"] == 1

    def test_mid_sentence(self):
        assert EM._mid_sentence("では印刷をします。", "することにします。", "では印刷することにします。") is True
        assert EM._mid_sentence("山本です。", "今日は準備", "山本です。今日は準備について。") is False
        assert EM._mid_sentence("xyz", "abc", "全然ちがう文") is None

    def test_no_records(self):
        assert EM.seg_split_report([], self.GOLD) == {}


if __name__ == "__main__":
    unittest.main()


class TestItaiNames(unittest.TestCase):
    def test_variant_kanji_are_the_same_name(self):
        self.assertEqual(EM._itai("渡邊"), EM._itai("渡辺"))
        self.assertEqual(EM._itai("渡邊"), "渡辺")
        self.assertNotEqual(EM._itai("堀"), EM._itai("森"))
