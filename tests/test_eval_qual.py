"""品質の切り分け評価(scripts/eval_qual.py)の、計算機なしで確かめられる部分。"""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import eval_qual as Q  # noqa: E402

TIMES = [{"start": 0.5, "end": 2.0}, {"start": 2.6, "end": 6.0}, {"start": 6.5, "end": 7.0}]


class TestAssign(unittest.TestCase):
    def test_units_go_to_the_turn_of_their_middle(self):
        units = [{"lo": 0.4, "hi": 2.1, "tr": "a"}, {"lo": 2.5, "hi": 4.0, "tr": "b1"},
                 {"lo": 4.0, "hi": 6.1, "tr": "b2"}, {"lo": 6.4, "hi": 7.1, "tr": "c"}]
        self.assertEqual(Q.assign(units, TIMES, "en"), ["a", "b1 b2", "c"])
        self.assertEqual(Q.assign(units, TIMES, "ja"), ["a", "b1b2", "c"])

    def test_merged_across_turns_leaves_the_other_turn_empty(self):
        units = [{"lo": 0.4, "hi": 6.1, "tr": "ab"}, {"lo": 6.4, "hi": 7.1, "tr": "c"}]
        self.assertEqual(Q.assign(units, TIMES, "en"), ["", "ab", "c"])

    def test_target_language_per_turn(self):
        turns = [{"lang": "ja"}, {"lang": "en"}]
        self.assertEqual([Q.tgt_of("ja-en", t["lang"]) for t in turns], ["en", "ja"])


class TestFullContext(unittest.TestCase):
    def test_next_utterance_is_added_as_context_only(self):
        msgs = Q.messages_with_next(("ja", "en"), [], "それ、いつまでですか", "ja",
                                    "The deadline is Friday.")
        body = msgs[-1]["content"]
        self.assertIn("それ、いつまでですか", body)
        self.assertIn("The deadline is Friday.", body)
        self.assertLess(body.index("The deadline is Friday."), body.index("それ、いつまでですか"))

    def test_no_next_is_the_app_message(self):
        from tsuyaku.translate import build_messages
        self.assertEqual(Q.messages_with_next(("ja", "en"), [], "はい", "ja", ""),
                         build_messages(("ja", "en"), "", [], "はい", "ja"))


class TestLosses(unittest.TestCase):
    def test_breakdown(self):
        s = {"full": 0.70, "seg": 0.78, "text": 0.84, "noctx": 0.80, "fullctx": 0.86}
        b = Q.breakdown(s)
        self.assertAlmostEqual(b["ear"], 0.06)
        self.assertAlmostEqual(b["seg"], 0.08)
        self.assertAlmostEqual(b["ctx_more"], 0.02)
        self.assertAlmostEqual(b["ctx_now"], 0.04)


if __name__ == "__main__":
    unittest.main()
