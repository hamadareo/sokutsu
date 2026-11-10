"""分野を選ぶ(11/08 IT 用語): 起動画面で「IT・技術」を選ぶと、IT の用語集を優先し、
一般の文では使わない形(process = プロセス など)も使う。"""
import json
import queue
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

from tsuyaku import __main__ as main_mod
from tsuyaku import terms
from tsuyaku.config import Settings, save_config
from tsuyaku.translate import build_messages


def _bridge():
    b = main_mod.Bridge.__new__(main_mod.Bridge)
    b.s = Settings()
    b.ui = queue.Queue()
    b.running = threading.Event()
    return b


class TestDomain(unittest.TestCase):
    def tearDown(self):
        terms.set_domain("")

    def test_set_domain_it(self):
        terms.set_domain("it")
        self.assertEqual(terms.current_domain(), "it")
        got = dict(terms.glossary_pairs("The process takes hours.", "en", "ja", domain=terms.current_domain()))
        self.assertIn("process", got)

    def test_auto_from_topic(self):
        terms.set_domain("", topic="Kubernetes 移行の打ち合わせ")
        self.assertEqual(terms.current_domain(), "it")
        terms.set_domain("", topic="家族旅行の相談")
        self.assertIsNone(terms.current_domain())
        terms.set_domain("general", topic="Kubernetes 移行の打ち合わせ")   # 一般を選んだら一般
        self.assertIsNone(terms.current_domain())

    def test_prompt_uses_domain(self):
        terms.set_domain("it")
        msgs = build_messages(("en", "ja"), "", [], "The process takes hours.", "en")
        self.assertIn("process = プロセス", msgs[-1]["content"])
        terms.set_domain("")
        msgs = build_messages(("en", "ja"), "", [], "The process takes hours.", "en")
        self.assertNotIn("process = プロセス", msgs[-1]["content"])

    def test_prompt_uses_history_as_context(self):
        hist = [{"lang": "en", "orig": "We moved the API to Kubernetes last week.",
                 "trans": "先週 API を Kubernetes に移しました。"}]
        msgs = build_messages(("en", "ja"), "", hist, "The process restarts every night.", "en")
        self.assertIn("process = プロセス", msgs[-1]["content"])


class TestConfigAndBridge(unittest.TestCase):
    def test_default_and_saved(self):
        self.assertEqual(Settings().term_domain, "")
        with tempfile.TemporaryDirectory() as d:
            with mock.patch("tsuyaku.config.ROOT", Path(d)):
                s = Settings()
                s.term_domain = "it"
                save_config(s)
                data = json.loads((Path(d) / "config.json").read_text())
        self.assertEqual(data["term_domain"], "it")

    def test_bridge_set_term_domain(self):
        b = _bridge()
        with mock.patch.object(main_mod, "save_config", lambda s: None):
            self.assertEqual(b.set_term_domain("it"), "ok")
            self.assertEqual(b.s.term_domain, "it")
            self.assertEqual(b.set_term_domain("bogus"), "bad")
            self.assertEqual(b.s.term_domain, "it")
            self.assertEqual(b.set_term_domain("general"), "ok")
        self.assertEqual(b.s.term_domain, "general")


if __name__ == "__main__":
    unittest.main()
