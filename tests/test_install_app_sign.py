"""11/08「音声が入っていない」: /Applications/Sokutsu.app(起動役)に署名が無く、macOS がマイクの
許可を聞かないまま無音を渡していた。install_app.sh は入れる前にアドホック署名をする。"""
import unittest
from pathlib import Path


class TestInstallAppSigns(unittest.TestCase):
    def test_codesign_before_install(self):
        s = (Path(__file__).resolve().parents[1] / "scripts/install_app.sh").read_text()
        self.assertIn('codesign --force --sign - "$TMP/Sokutsu.app"', s)
        # 署名は Info.plist と起動役を書いた後、/Applications へ動かす前
        self.assertLess(s.index('chmod +x "$B/MacOS/Sokutsu"'), s.index("codesign --force"))

    def test_launcher_spawns_not_exec(self):
        """Python へ exec すると実体が Python.app になりマイクの許可が聞かれない。C の起動役が子として起こす。"""
        root = Path(__file__).resolve().parents[1]
        s = (root / "scripts/install_app.sh").read_text()
        self.assertIn("native/devlauncher/devlauncher.c", s)
        self.assertNotIn("exec /usr/bin/arch", s)
        c = (root / "native/devlauncher/devlauncher.c").read_text()
        self.assertIn("posix_spawn(", c)
        self.assertIn("waitpid(", c)
        self.assertNotIn("execv", c)
        self.assertLess(s.index("codesign --force"), s.index('mv "$TMP/Sokutsu.app" "$APP"'))


if __name__ == "__main__":
    unittest.main()
