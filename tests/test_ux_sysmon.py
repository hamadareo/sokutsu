"""使用率表示(sysmon.py)と設定 show_sysmon の試験。値の読み取りは文字列の解釈だけを試し、取得失敗は None。"""
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tsuyaku import sysmon
from tsuyaku import config as cfgmod

VM = """Mach Virtual Memory Statistics: (page size of 16384 bytes)
Pages free:                               10000.
Pages active:                            200000.
Pages inactive:                          190000.
Pages wired down:                         80000.
Pages occupied by compressor:             20000.
"""
IOREG = '''+-o AGXAcceleratorG14X  <class AGXAcceleratorG14X>
    {
      "PerformanceStatistics" = {"Device Utilization %"=37,"Renderer Utilization %"=30,"Tiler Utilization %"=12}
    }'''


class TestParse(unittest.TestCase):
    def test_mem(self):
        # 使用 = active + wired + compressor = 300000 ページ × 16384 / 総量 8GiB
        pct = sysmon.parse_mem(VM, 8 * 1024 ** 3)
        self.assertAlmostEqual(pct, 300000 * 16384 / (8 * 1024 ** 3) * 100, places=1)

    def test_mem_bad(self):
        self.assertIsNone(sysmon.parse_mem("", 8 * 1024 ** 3))
        self.assertIsNone(sysmon.parse_mem(VM, 0))

    def test_gpu(self):
        self.assertEqual(sysmon.parse_gpu(IOREG), 37)
        self.assertIsNone(sysmon.parse_gpu("nothing"))
        self.assertEqual(sysmon.parse_gpu('"Device Utilization %"=250'), 100)

    def test_cpu_delta(self):
        self.assertEqual(sysmon.cpu_pct((10, 10, 70, 10), (30, 20, 140, 10)), 30)
        self.assertIsNone(sysmon.cpu_pct(None, (1, 1, 1, 1)))
        self.assertIsNone(sysmon.cpu_pct((1, 1, 1, 1), (1, 1, 1, 1)))


class TestSample(unittest.TestCase):
    def test_failure_is_silent(self):
        m = sysmon.Sysmon()
        with mock.patch.object(sysmon, "_run", side_effect=OSError("x")), \
                mock.patch.object(sysmon, "_cpu_ticks", return_value=None):
            d = m.sample()
        self.assertEqual(d, {})

    def test_cache(self):
        m = sysmon.Sysmon()
        with mock.patch.object(sysmon, "_run", return_value="") as r, \
                mock.patch.object(sysmon, "_cpu_ticks", return_value=None):
            m.sample(); m.sample()
        self.assertEqual(r.call_count, 3)   # 初回は sysctl + vm_stat + ioreg、2回目は温存


class TestSetting(unittest.TestCase):
    def test_default_on_and_saved(self):
        self.assertTrue(cfgmod.Settings().show_sysmon)
        with tempfile.TemporaryDirectory() as d:
            with mock.patch.object(cfgmod, "_config_file", lambda: Path(d) / "config.json"):
                s = cfgmod.Settings(); s.show_sysmon = False
                cfgmod.save_config(s)
                self.assertFalse(cfgmod.load_settings().show_sysmon)


if __name__ == "__main__":
    unittest.main()


class TestBridgeApi(unittest.TestCase):
    def _b(self):
        from test_hear_devices import _bridge
        return _bridge()

    def test_off_means_no_sampling(self):
        import tsuyaku.__main__ as main_mod
        b = self._b()
        with mock.patch.object(main_mod, "save_config", lambda s: None):
            b.set_sysmon(False)
        self.assertFalse(b.s.show_sysmon)
        with mock.patch.object(sysmon.Sysmon, "sample", side_effect=AssertionError("取得してはいけない")):
            self.assertIsNone(b.get_sysmon())

    def test_on_returns_sample(self):
        b = self._b()
        with mock.patch.object(sysmon.Sysmon, "sample", return_value={"cpu": 5}):
            self.assertEqual(b.get_sysmon(), {"cpu": 5})

    def test_get_tts_has_flag(self):
        import tsuyaku.__main__ as main_mod
        b = self._b()
        with mock.patch.object(main_mod, "_say_voices", lambda: {}), \
                mock.patch.object(main_mod, "_voicevox_list", lambda u: []):
            self.assertTrue(b.get_tts()["showSysmon"])

    def test_term_add_dedup(self):
        import tsuyaku.__main__ as main_mod
        b = self._b()
        store = {"v": ["Kubernetes"]}
        from tsuyaku import inmeeting
        with mock.patch.object(inmeeting, "load_terms", lambda *a: list(store["v"])), \
                mock.patch.object(inmeeting, "save_terms", lambda items, *a: store.update(v=list(items)) or items):
            r = b.term_add("  GPU  ")
            b.term_add("gpu")
            b.term_add("")
        self.assertEqual(store["v"], ["Kubernetes", "GPU"])
        self.assertTrue(r["ok"])

    def test_speak_again_idle(self):
        b = self._b()
        self.assertEqual(b.speak_again("こんにちは", "ja"), "idle")   # 会話していない時は何もしない
