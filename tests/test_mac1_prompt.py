import unittest

from tsuyaku import translate as T


class Mac1PromptTest(unittest.TestCase):
    def test_defaults_keep_prompt_unchanged(self):
        sysmsg = T.build_system(("ja", "zh"))
        self.assertNotIn("Examples of faithful", sysmsg)
        msgs = T.build_messages(("ja", "zh"), "", [], "如果下雨的话", "zh")
        self.assertNotIn("unfinished. Translate exactly", msgs[-1]["content"])

    def test_shots_cover_both_directions(self):
        b = T.shots_block(("ja", "zh"))
        self.assertIn("周五见。 => 金曜日にね。", b)
        self.assertIn("金曜日にね。 => 周五见。", b)

    def test_fraghint_only_on_open_ending(self):
        old = T.FRAGHINT
        T.FRAGHINT = True
        try:
            open_ = T.build_messages(("ja", "zh"), "", [], "如果下雨的话", "zh")[-1]["content"]
            closed = T.build_messages(("ja", "zh"), "", [], "我饿了。", "zh")[-1]["content"]
        finally:
            T.FRAGHINT = old
        self.assertIn("may be unfinished", open_)
        self.assertNotIn("may be unfinished", closed)

    def test_configure_for_model_only_small_gemma(self):
        old = T.SHOTTURNS
        try:
            T.configure_for_model("google/gemma-3-4b")
            self.assertEqual(T.SHOTTURNS, T.SHOTS_SMALL)
            msgs = T.build_messages(("ja", "zh"), "", [], "如果下雨的话", "zh")
            self.assertEqual([m["role"] for m in msgs[:3]], ["system", "user", "assistant"])
            self.assertEqual(msgs[-1]["role"], "user")
            T.configure_for_model("qwen3.8-flash-next")
            self.assertEqual(T.SHOTTURNS, 0)
            msgs = T.build_messages(("ja", "zh"), "", [], "如果下雨的话", "zh")
            self.assertEqual(len(msgs), 2)
        finally:
            T.SHOTTURNS = old

    def test_shot_prefix_is_static(self):
        a = T.shot_turns(("en", "ja"), 4)
        self.assertEqual(a, T.shot_turns(("en", "ja"), 4))
        self.assertEqual(len(a), 16)

    def test_history_window_keep(self):
        self.assertEqual(T.history_window(9, 4), 4)
        self.assertEqual(T.history_window(9, 2), 6)


if __name__ == "__main__":
    unittest.main()
