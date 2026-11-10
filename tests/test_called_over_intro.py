"""短い名乗りの聞き違い・取り違えより、呼ばれた名前を採る(11/07 擬似会議 ja_dev4: 森さんの「森です」が
「堀です」と聞こえ渡辺さん S2 の行になり、S2=堀 が本人の名乗りとして確かめを通った。S2 は会議の中で
「渡辺さんからお願いします」の直後に話している)。"""
import unittest

from tsuyaku import notes as N


def L(spk, orig):
    return {"spk": spk, "orig": orig, "lang": "ja"}


def P(i, name="", fixed=False, ev=""):
    return {"id": i, "name": name, "evidence": ev, "fixed": fixed, "minor": False}


DEV4 = [L("S1", "高橋です。今週の開発の進捗を確認します。"), L("S1", "渡辺さんからお願いします。"),
        L("S2", "ログイン画面は完成しました"), L("S1", "小川さんはどうですか"),
        L("S3", "小川です。通知の機能でバグが一つ残っています。"), L("S2", "堀です。"),
        L("S2", "デザインの修正案を作りました")]


class TestCalledOverShortIntro(unittest.TestCase):
    def test_hori_becomes_watanabe(self):
        parts = [P("S1", "高橋"), P("S2", "堀", ev="堀です。"), P("S3", "小川")]
        N.called_over_short_intro(parts, DEV4)
        self.assertEqual(parts[1]["name"], "渡辺")
        self.assertEqual(parts[1]["evidence"], "渡辺さんからお願いします。")

    def test_two_called_names_means_no_name(self):
        lines = [L("S1", "小川さん、どうですか"), L("S3", "はい"), L("S1", "森さんは?"),
                 L("S3", "小田和です。")]
        parts = [P("S1", "高橋"), P("S3", "小田和")]
        N.called_over_short_intro(parts, lines)
        self.assertEqual(parts[1]["name"], "")

    def test_minasan_is_not_a_call(self):
        lines = [L("S1", "皆さん、始めます"), L("S2", "堀です。"), L("S2", "よろしく")]
        parts = [P("S1"), P("S2", "堀")]
        N.called_over_short_intro(parts, lines)
        self.assertEqual(parts[1]["name"], "堀")

    def test_two_intros_by_same_speaker_keeps_name(self):
        lines = DEV4 + [L("S2", "堀です。もう一度言います。")]
        parts = [P("S1", "高橋"), P("S2", "堀")]
        N.called_over_short_intro(parts, lines)
        self.assertEqual(parts[1]["name"], "堀")

    def test_long_intro_line_keeps_name(self):
        lines = [L("S1", "渡辺さんからお願いします。"), L("S2", "森です、よろしくお願いします、設計担当です")]
        parts = [P("S1"), P("S2", "森")]
        N.called_over_short_intro(parts, lines)
        self.assertEqual(parts[1]["name"], "森")

    def test_called_name_already_used_means_no_name(self):
        parts = [P("S1", "高橋"), P("S2", "堀"), P("S3", "小川"), P("S4", "渡辺")]
        N.called_over_short_intro(parts, DEV4)
        self.assertEqual(parts[1]["name"], "")

    def test_fixed_name_is_never_touched(self):
        parts = [P("S1", "高橋"), P("S2", "堀", fixed=True)]
        N.called_over_short_intro(parts, DEV4)
        self.assertEqual(parts[1]["name"], "堀")

    def test_called_same_name_keeps(self):
        lines = [L("S1", "森さん、どうぞ"), L("S2", "森です。")]
        parts = [P("S1"), P("S2", "森")]
        N.called_over_short_intro(parts, lines)
        self.assertEqual(parts[1]["name"], "森")

    def test_clean_applies_it(self):
        raw = {"participants": [{"id": "S1", "name": "高橋", "evidence": "高橋です。"},
                                {"id": "S2", "name": "堀", "evidence": "堀です。"},
                                {"id": "S3", "name": "小川", "evidence": "小川です。"}]}
        out = N._clean(raw, DEV4, {}, None)
        self.assertEqual({p["id"]: p["name"] for p in out["participants"]},
                         {"S1": "高橋", "S2": "渡辺", "S3": "小川"})
