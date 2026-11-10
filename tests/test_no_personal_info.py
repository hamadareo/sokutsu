"""配る版に入るコード(src/)に、私的なアドレスやメールアドレスを入れない。

ビルドする人が自分の名前・所属・ホスト名を確かめたいときは、環境変数 SOKUTSU_FORBIDDEN_WORDS
(コンマ区切り)に入れて実行する。リポジトリには個人の語を置かない。
"""
import os
import re
import unittest
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
EXTRA = [w.strip() for w in os.environ.get("SOKUTSU_FORBIDDEN_WORDS", "").split(",") if w.strip()]
BLOCK = re.compile("|".join(
    [r"(?<![0-9.])(?:10|192\.168|172\.(?:1[6-9]|2[0-9]|3[01]))\.[0-9]{1,3}\.[0-9]{1,3}(?:\.[0-9]{1,3})?(?![0-9])",
     r"\.ts\.net",
     r"[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+\.(?:com|jp|net|org|ac\.jp|co\.jp)"]
    + [re.escape(w) for w in EXTRA]), re.I)


class TestNoPersonalInfo(unittest.TestCase):
    def test_src_has_no_private_address_or_email(self):
        hits = []
        for p in SRC.rglob("*"):
            if p.suffix not in {".py", ".html", ".js", ".css", ".json", ".txt", ".md"} or "__pycache__" in p.parts:
                continue
            for i, line in enumerate(p.read_text(errors="ignore").splitlines(), 1):
                if BLOCK.search(line):
                    hits.append(f"{p.relative_to(SRC)}:{i}: {line.strip()[:80]}")
        self.assertEqual(hits, [], "\n".join(hits))


if __name__ == "__main__":
    unittest.main()
