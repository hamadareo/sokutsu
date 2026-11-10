"""画面の明るさ(11/07 notes-notion): システムに合わせる / 明るい / 暗い。config.json に残す。"""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tsuyaku import config as C
from tsuyaku.__main__ import Bridge


class TestThemeSetting(unittest.TestCase):
    def test_default_is_system(self):
        self.assertEqual(C.Settings().ui_theme, "system")

    def test_set_and_get_saves_config(self):
        with tempfile.TemporaryDirectory() as d:
            cfg = Path(d) / "config.json"
            cfg.write_text(json.dumps({"engine": "auto", "other": 1}), encoding="utf-8")
            with patch.object(C, "_config_file", return_value=cfg):
                b = Bridge()
                self.assertEqual(b.set_theme("dark"), "dark")
                self.assertEqual(b.get_theme(), "dark")
                data = json.loads(cfg.read_text(encoding="utf-8"))
                self.assertEqual(data["ui_theme"], "dark")
                self.assertEqual(data["other"], 1)            # ほかの設定は残す
                self.assertEqual(b.set_theme("purple"), "system")   # 知らない値はシステムに合わせる
                self.assertEqual(json.loads(cfg.read_text(encoding="utf-8"))["ui_theme"], "system")


if __name__ == "__main__":
    unittest.main()
