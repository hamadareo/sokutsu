"""テーマメモの用語を耳(whisper)の手がかりにする(11/06 実会話)。

「YOLOモデル」が「養老モデル」、「山田花子」が「山田良子」と書き起こされた。
VOICEVOX 2声×12文で、手がかり無し 9/24 → 「、」区切り+「。」の手がかりで 22/24。
"""
import unittest
from pathlib import Path

from tsuyaku.guards import echoes_prompt, stt_prompt

ROOT = Path(__file__).resolve().parents[1]


class TestSttPrompt(unittest.TestCase):
    def test_list_becomes_touten_list_with_kuten(self):
        self.assertEqual(stt_prompt("YOLO・物体検出・山田花子"), "YOLO、物体検出、山田花子。")
        self.assertEqual(stt_prompt("YOLO, Whisper ,Qwen"), "YOLO、Whisper、Qwen。")
        self.assertEqual(stt_prompt("YOLOを使った物体検出の研究。"), "YOLOを使った物体検出の研究。")

    def test_empty(self):
        self.assertEqual(stt_prompt(""), "")
        self.assertEqual(stt_prompt(" ・ 、"), "")


class TestEchoesPrompt(unittest.TestCase):
    P = "YOLO、山田花子、Whisper、Qwen、RAG、ボクセル、DSLC、TrackID。"

    def test_noise_echo_is_dropped(self):
        # 雑音22区間で実際に出た復唱(p=0.18)
        self.assertTrue(echoes_prompt("Qwen、RAG、Bス、TLACKID。", self.P))
        self.assertTrue(echoes_prompt("YOLO、山田花子、Whisper。", self.P))

    def test_real_sentences_with_terms_are_kept(self):
        for t in ["YOLOモデルについて話したいです。", "YOLO、Whisper、Qwenを使いました。",
                  "私の名前は山田花子です。", "We use YOLO and Whisper.", "はい、YOLOです。"]:
            self.assertFalse(echoes_prompt(t, self.P), t)

    def test_no_prompt_never_drops(self):
        self.assertFalse(echoes_prompt("Qwen、RAG、TrackID。", ""))


class TestWiring(unittest.TestCase):
    def test_app_passes_topic_to_ear_and_guards_echo(self):
        src = (ROOT / "src/tsuyaku/__main__.py").read_text()
        self.assertIn("self.stt.prompt = stt_prompt(topic, terms=", src)   # 11/07 用語辞書も(W3)
        self.assertIn("if echoes_prompt(text, self.stt.prompt):", src)

    def test_topic_is_remembered(self):
        src = (ROOT / "src/tsuyaku/__main__.py").read_text()
        self.assertIn("self.s.last_topic = topic", src)
        self.assertIn('"topic": s.last_topic', src)
        cfg = (ROOT / "src/tsuyaku/config.py").read_text()
        self.assertIn('data["last_topic"] = s.last_topic', cfg)
        ui = (ROOT / "src/tsuyaku/webui/index.html").read_text()
        self.assertIn("cfg.topic", ui)


if __name__ == "__main__":
    unittest.main()


class TestPendingCaption(unittest.TestCase):
    def test_held_fragment_is_shown_while_waiting(self):
        src = (ROOT / "src/tsuyaku/__main__.py").read_text()
        self.assertIn('shown = clean_heard(ep.pending_text(), lang, self._heard_terms)', src)
        self.assertIn('self.ui.put(("partial", shown, "…", lang))', src)
