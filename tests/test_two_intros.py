"""1つの行に別々の名前の名乗りが2つある(11/07 擬似会議 v3 ja_lab「田中です佐藤です」: 話者交代の間が
0.06秒で、2人の名乗りが1区間=1行になる)。音声の区切りは変えず、名前の付け方だけで救う。"""
import unittest

from tsuyaku import notes as N


def L(spk, orig):
    return {"spk": spk, "orig": orig, "lang": "ja"}


def P(i, name=""):
    return {"id": i, "name": name, "evidence": "", "fixed": False}


class TestTwoIntros(unittest.TestCase):
    def test_detect(self):
        self.assertEqual(N.two_intros("田中です佐藤です"), ("田中", "佐藤"))
        self.assertEqual(N.two_intros("高橋です。渡辺です。"), ("高橋", "渡辺"))
        self.assertEqual(N.two_intros("I'm Tom, and I'm Lisa."), ("Tom", "Lisa"))
        self.assertIsNone(N.two_intros("田中です。よろしくお願いします。"))
        self.assertIsNone(N.two_intros("田中です。田中です。"))
        self.assertIsNone(N.two_intros("それはまだわかりませんです。大丈夫です。"))

    def test_ja_lab_second_name_owner_gives_first_name_to_previous(self):
        # 実際の v3: 行は佐藤さん(S3)の声が多く S3。S3 は「佐藤」、前に話した S2 はまだ名前が無い
        lines = [L("S1", "司会の山田です"), L("S2", "よろしくお願いします。"), L("S3", "田中です佐藤です"),
                 L("S3", "よろしくお願いします"), L("S4", "鈴木です。お願いします。")]
        parts = [P("S1", "山田"), P("S2"), P("S3", "佐藤"), P("S4", "鈴木")]
        N.two_intro_names(parts, lines)
        self.assertEqual({p["id"]: p["name"] for p in parts},
                         {"S1": "山田", "S2": "田中", "S3": "佐藤", "S4": "鈴木"})

    def test_ja_dev4_first_name_owner_gives_second_name_to_next(self):
        lines = [L("S1", "高橋です。渡辺です。"), L("S2", "ログイン画面は完成しました"), L("S1", "小川さんは?")]
        parts = [P("S1", "高橋"), P("S2")]
        N.two_intro_names(parts, lines)
        self.assertEqual(parts[1]["name"], "渡辺")
        self.assertIn("渡辺です", parts[1]["evidence"])

    def test_unnamed_line_speaker_follows_order_with_outside_evidence(self):
        lines = [L("S1", "I'm Tom, and I'm Lisa."), L("S2", "Let's start with the budget."),
                 L("S1", "Thanks, Lisa.")]
        parts = [P("S1"), P("S2")]
        N.two_intro_names(parts, lines)
        self.assertEqual((parts[0]["name"], parts[1]["name"]), ("Tom", "Lisa"))
        parts = [P("S1"), P("S2")]                        # LLM が名前に挙げた
        N.two_intro_names(parts, lines[:2], {"Tom"})
        self.assertEqual((parts[0]["name"], parts[1]["name"]), ("Tom", "Lisa"))

    def test_unnamed_without_outside_evidence_does_nothing(self):
        lines = [L("S1", "I'm Tom, and I'm Lisa."), L("S2", "Let's start with the budget.")]
        parts = [P("S1"), P("S2")]
        N.two_intro_names(parts, lines)
        self.assertEqual((parts[0]["name"], parts[1]["name"]), ("", ""))

    def test_times_and_numbers_are_not_names(self):
        for t in ("月曜です。三時です。", "五人です。三回です。", "10時です。山田です。"):
            self.assertIsNone(N.two_intros(t), t)
            lines = [L("S1", t), L("S2", "はい。"), L("S1", "月曜さん、山田さん")]
            parts = [P("S1"), P("S2")]
            N.two_intro_names(parts, lines, {"月曜", "山田"})
            self.assertEqual((parts[0]["name"], parts[1]["name"]), ("", ""), t)

    def test_tokyo_osaka_without_evidence_does_nothing(self):
        lines = [L("S1", "東京です。大阪です。"), L("S2", "そうですか。")]
        parts = [P("S1"), P("S2")]
        N.two_intro_names(parts, lines)
        self.assertEqual((parts[0]["name"], parts[1]["name"]), ("", ""))

    def test_single_name_line_does_nothing(self):
        lines = [L("S1", "田中です。よろしくお願いします。"), L("S2", "はい。")]
        parts = [P("S1"), P("S2")]
        N.two_intro_names(parts, lines)
        self.assertEqual((parts[0]["name"], parts[1]["name"]), ("", ""))

    def test_never_overwrites_or_duplicates(self):
        lines = [L("S1", "高橋です。渡辺です。"), L("S2", "ログイン画面は完成しました")]
        parts = [P("S1", "高橋"), P("S2", "森")]          # 次の話者に既に名前
        N.two_intro_names(parts, lines)
        self.assertEqual(parts[1]["name"], "森")
        parts = [P("S1", "高橋"), P("S2"), P("S3", "渡辺")]  # 名前が既に別の人に付いている
        N.two_intro_names(parts, lines + [L("S3", "はい")])
        self.assertEqual(parts[1]["name"], "")

    def test_no_neighbor_does_nothing(self):
        lines = [L("S1", "高橋です。渡辺です。"), L("S1", "続けます")]
        parts = [P("S1", "高橋")]
        N.two_intro_names(parts, lines)
        self.assertEqual([p["name"] for p in parts], ["高橋"])

    def test_clean_calls_it(self):
        raw = {"participants": [{"id": "S1", "name": "高橋", "evidence": "高橋です。渡辺です。"}]}
        lines = [L("S1", "高橋です。渡辺です。"), L("S2", "ログイン画面は完成しました")]
        out = N._clean(raw, lines, {}, None)
        self.assertEqual({p["id"]: p["name"] for p in out["participants"]}, {"S1": "高橋", "S2": "渡辺"})
