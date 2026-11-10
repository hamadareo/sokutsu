"""話のペースに合わせた読み上げの速さ(pace.py、11/08)。"""
import unittest

from tsuyaku.pace import PaceController, engine_of, units


class TestUnits(unittest.TestCase):
    def test_units(self):
        self.assertEqual(units("今日は、晴れです。", "ja"), 7)
        self.assertEqual(units("I think so, too.", "en"), 4)
        self.assertEqual(units("안녕하세요, 반갑습니다.", "ko"), 10)
        self.assertEqual(engine_of("voicevox:13"), "voicevox")
        self.assertEqual(engine_of("say:Kyoko"), "say")
        self.assertEqual(engine_of(""), "say")


class TestChoose(unittest.TestCase):
    def pc(self, **kw):
        return PaceController(caps={("voicevox", "ja"): 1.8}, cps0={("voicevox", "ja"): 6.0},
                              rise=1.0, fall=1.0, **kw)

    def test_enough_time_keeps_user_speed(self):
        # 12字 = ×1.0 で2秒。話者は4秒かけて話した → 速める必要なし
        self.assertEqual(self.pc().choose("あ" * 12, "voicevox", "ja", 4.0, 0.0, base=1.0), 1.0)

    def test_fast_speaker_speeds_up(self):
        # 24字 = 4秒。話者は2.5秒 → ×1.6
        self.assertAlmostEqual(self.pc().choose("あ" * 24, "voicevox", "ja", 2.5, 0.0), 1.6, places=2)

    def test_backlog_speeds_up_more_but_capped(self):
        sp = self.pc().choose("あ" * 24, "voicevox", "ja", 2.5, 1.0)
        self.assertEqual(sp, 1.8)                       # 聞き取れる上限で止める

    def test_never_slower_than_user_speed(self):
        self.assertEqual(self.pc().choose("あ" * 6, "voicevox", "ja", 5.0, 0.0, base=1.3), 1.3)

    def test_unknown_voice_uses_default_cap(self):
        from tsuyaku.pace import DEFAULT_CAP
        sp = PaceController(rise=1.0).choose("가" * 80, "kokoro", "ko", 1.0, 5.0)
        self.assertEqual(sp, DEFAULT_CAP)

    def test_smoothing_rises_fast_and_falls_slowly(self):
        pc = PaceController(caps={("voicevox", "ja"): 2.0}, cps0={("voicevox", "ja"): 6.0},
                            rise=0.6, fall=0.25)
        up = pc.choose("あ" * 24, "voicevox", "ja", 2.0, 0.0)     # 目標 2.0
        self.assertAlmostEqual(up, 1.6, places=2)                   # 1.0 + 0.6*(2.0-1.0)
        down = pc.choose("あ" * 6, "voicevox", "ja", 10.0, 0.0)     # 目標 1.0
        self.assertAlmostEqual(down, 1.45, places=2)                # 1.6 + 0.25*(1.0-1.6)
        self.assertEqual(pc.current("voicevox", "ja"), 1.45)


class TestObserve(unittest.TestCase):
    def test_learns_chars_per_second(self):
        pc = PaceController(cps0={("say", "ja"): 6.0}, learn=0.5)
        pc.observe("あ" * 20, "say", "ja", 1.0, 4.0)               # 実際は5字/秒
        self.assertAlmostEqual(pc.cps[("say", "ja")], 5.5, places=2)
        pc.observe("あ" * 20, "say", "ja", 2.0, 2.0)               # ×2.0 で2秒 = ×1.0 で5字/秒
        self.assertAlmostEqual(pc.cps[("say", "ja")], 5.25, places=2)

    def test_ignores_tiny_samples(self):
        pc = PaceController(cps0={("say", "ja"): 6.0})
        pc.observe("はい", "say", "ja", 1.0, 0.2)
        self.assertEqual(pc.cps[("say", "ja")], 6.0)


class TestPriority(unittest.TestCase):
    """自動の中の選択(11/08 リーダー): 「聞きやすさ優先」= 測った上限(既定)/「速さ優先」=
    VOICEVOX・Kokoro は ×1.5 まで(say の声は測った上限のまま)。"""

    def test_clear_is_default_and_uses_measured_caps(self):
        pc = PaceController()
        self.assertEqual(pc.priority, "clear")
        self.assertEqual(pc.cap("voicevox", "ja"), 1.25)
        self.assertEqual(pc.cap("kokoro", "zh"), 1.25)

    def test_fast_allows_1_5_for_voicevox_and_kokoro_only(self):
        pc = PaceController(priority="fast")
        self.assertEqual(pc.cap("voicevox", "ja"), 1.5)
        self.assertEqual(pc.cap("kokoro", "en"), 1.5)
        self.assertEqual(pc.cap("say", "ja"), 2.0)       # say は測った上限のまま
        self.assertEqual(pc.cap("say", "ko"), 1.75)
        pc.priority = "clear"
        self.assertEqual(pc.cap("voicevox", "ja"), 1.25)


class TestEarsOnly(unittest.TestCase):
    """読み上げの声が聞き手の耳だけに届くか(devices.output_is_private の答え + 分からない時は名前)。"""

    def test_kinds(self):
        from tsuyaku.pace import ears_only
        self.assertTrue(ears_only(True, ""))
        self.assertFalse(ears_only(False, "AirPods Pro"))          # 判定が付いたらそれに従う
        self.assertTrue(ears_only(None, "Logi USB Headset"))       # USB でも名前がヘッドセット
        self.assertTrue(ears_only(None, "USB ヘッドホン"))
        self.assertTrue(ears_only(None, "Jabra Evolve2 Headphones"))
        self.assertFalse(ears_only(None, "USB Audio Speaker"))     # USB スピーカーは部屋に鳴る
        self.assertFalse(ears_only(None, "Microsoft Teams Audio"))
        self.assertFalse(ears_only(None, ""))


if __name__ == "__main__":
    unittest.main()
