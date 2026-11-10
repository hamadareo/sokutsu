"""詰まり防止の防波堤テスト(ライブ不具合 2026-11-05: 雑音でずっと翻訳しなくなる)。

原因: 雑音でVADが蹴ると whisper が幻覚の短文(カウダ。/하다. 하다.等)を出し、
その1つずつが whisper+LLM+読み上げゲートを直列に潰して本物の発話が行列する。
_layers_:
1. rms: 部屋ノイズは noise_rms_floor 未満、普通の声は十分に上。
2. pick_segments: 詰まったら最新1個だけ訳す。少しの遅れはターンを落とさない。
3. hallucinated_out_of_pair: ペア外言語の幻覚短文は捨て、長い本物とペア内は通す。
4. config: engine 既定"auto"/TSUYAKU_ENGINE上書き/engine_candidates の優先順/save_config。
5. __main__ の順序: ゲートは whisper より前、幻覚チェックはエコーの後・翻訳の前、keepalive。
6. UI: 処理場所(自動/リモートGPU/このMac)と雑音環境の選択、動的ヒント。
"""
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from tsuyaku.audio import rms
from tsuyaku.config import (ENGINE_LABELS, MAC_PORT, REMOTE_PORT, Settings,
                            engine_candidates, load_settings, save_config)
from tsuyaku.guards import hallucinated_out_of_pair, pick_segments

ROOT = Path(__file__).resolve().parents[1]


class TestRms(unittest.TestCase):
    def test_quiet_noise_below_normal_speech(self):
        rng = np.random.default_rng(3)
        noise = rng.normal(0, 0.002, 16000).astype(np.float32)   # 遠くの部屋ノイズ
        speech = rng.normal(0, 0.05, 16000).astype(np.float32)    # 普通の声
        s = Settings()
        self.assertLess(rms(noise), s.noise_rms_floor)
        self.assertGreater(rms(speech), s.noise_rms_floor * 3)

    def test_empty_is_zero(self):
        self.assertEqual(rms(np.zeros(0, dtype=np.float32)), 0.0)


class TestBacklog(unittest.TestCase):
    def test_one_pending_kept(self):
        a = np.zeros(10, dtype=np.float32)
        self.assertEqual(pick_segments([a]), [a])

    def test_two_pending_kept(self):
        a, b = np.zeros(4, np.float32), np.ones(4, np.float32)
        self.assertEqual(len(pick_segments([a, b])), 2)

    def test_backlog_collapses_to_newest(self):
        segs = [np.full(2, float(i), np.float32) for i in range(5)]
        out = pick_segments(segs)
        self.assertEqual(len(out), 1)
        self.assertEqual(float(out[0][0]), 4.0)


class TestOutOfPair(unittest.TestCase):
    def test_short_out_of_pair_dropped(self):
        # 実セッションでノイズから出た幻覚そのもの(ja⇔en の会議)
        self.assertTrue(hallucinated_out_of_pair(("ja", "en"), "ko", "카우다."))
        self.assertTrue(hallucinated_out_of_pair(("ja", "en"), "ko", "하다. 하다."))
        self.assertTrue(hallucinated_out_of_pair(("ja", "ko"), "en", "yes."))

    def test_in_pair_or_long_or_unknown_kept(self):
        self.assertFalse(hallucinated_out_of_pair(("ja", "en"), "ja", "카우다 풍"))
        # 本物の第三言語の一文は通す(min_len 5→12 に上げた11/05以降の契約では、
        # ペア外は norm≥12 を「本物の発話」とみなす。字幕のみ表示の言語なので
        # 短い本物を捨てても実害は小さい)
        self.assertFalse(hallucinated_out_of_pair(
            ("ja", "en"), "ko", "오늘 회의 정말 좋았습니다. 감사합니다"))
        self.assertFalse(hallucinated_out_of_pair(("ja", "en"), "en", "yes indeed"))
        self.assertFalse(hallucinated_out_of_pair(("ja", "en"), "", "...."))


class TestEngineConfig(unittest.TestCase):
    def test_defaults(self):
        s = Settings()
        self.assertEqual(s.engine, "auto")
        self.assertEqual(s.vad_threshold, 0.5)
        self.assertGreater(s.noise_rms_floor, 0.0)

    def test_auto_prefers_remote_then_mac(self):
        c = engine_candidates("auto")
        self.assertEqual([lab for lab, _b, _m in c],
                         [ENGINE_LABELS["remote"], ENGINE_LABELS["mac"]])
        self.assertIn(f":{REMOTE_PORT}", c[0][1])
        self.assertIn(f":{MAC_PORT}", c[1][1])
        self.assertEqual(c[0][2], "qwen3.8-flash-next")
        # 11/05夕: gemma-3-12b は16GBのMacでスワップ→翻訳45〜67秒。4Bへ。
        self.assertEqual(c[1][2], "google/gemma-3-4b")

    def test_forced_modes(self):
        (lab, base, model), = engine_candidates("remote")
        self.assertEqual(lab, ENGINE_LABELS["remote"])
        self.assertIn(f":{REMOTE_PORT}", base)
        (lab, base, model), = engine_candidates("mac")
        self.assertEqual(lab, ENGINE_LABELS["mac"])
        self.assertIn(f":{MAC_PORT}", base)

    def test_custom_port_wins(self):
        c = engine_candidates("auto", "127.0.0.1", 9999, "mymodel")
        self.assertEqual(len(c), 1)
        self.assertIn(":9999", c[0][1])
        self.assertEqual(c[0][2], "mymodel")

    def test_env_overrides_engine(self):
        with mock.patch.dict(os.environ, {"TSUYAKU_ENGINE": "mac"}):
            self.assertEqual(load_settings().engine, "mac")


class TestSaveConfig(unittest.TestCase):
    def test_save_config_merges(self):
        with tempfile.TemporaryDirectory() as d:
            cfg = Path(d) / "config.json"
            cfg.write_text(json.dumps({"log_dir": "keep"}))
            with mock.patch("tsuyaku.config.ROOT", Path(d)):
                s = Settings()
                s.engine = "remote"
                s.vad_threshold = 0.65
                save_config(s)
                data = json.loads(cfg.read_text())
            self.assertEqual(data["log_dir"], "keep")
            self.assertEqual(data["engine"], "remote")
            self.assertEqual(data["vad_threshold"], 0.65)
            self.assertIn("tts_voice", data)
            self.assertIn("tts_speed", data)


class TestMainGuards(unittest.TestCase):
    def setUp(self):
        self.src = (ROOT / "src" / "tsuyaku" / "__main__.py").read_text()

    def test_energy_gate_before_whisper(self):
        self.assertLess(self.src.index("s.noise_rms_floor"),
                        self.src.index("text, lang, prob = hear(seg,"),
                        "雑音ゲートが whisper より後では空振りを防げない")

    def test_backlog_before_whisper(self):
        # 11/05 続き待ちマージ導入で pending → pick_segments → backlog にリネーム済み
        self.assertLess(self.src.index("pick_segments(pending)"),
                        self.src.index("text, lang, prob = hear(seg,"),
                        "詰め込み解消が whisper より後では詰まりが続く")

    def test_hallucination_gate_after_echo_before_translation(self):
        self.assertLess(self.src.index("is_echo(self.last_spoken, text)"),
                        self.src.index("hallucinated_out_of_pair("))
        self.assertLess(self.src.index("hallucinated_out_of_pair("),
                        self.src.index("msgs = build_messages(pair_t"))
        # 先読み翻訳(11/06)も幻覚判定を通った断片だけ(Endpointer に入れる前に判定)
        self.assertLess(self.src.index("hallucinated_out_of_pair("),
                        self.src.index("send(ep.push("))

    def test_keepalive_keeps_engine_hot(self):
        self.assertIn("_keepalive", self.src)
        self.assertIn("480", self.src, "翻訳サーバーの生存確認間隔(480s)がない")

    def test_vad_threshold_passed_to_capture(self):
        self.assertIn("vad_threshold=s.vad_threshold", self.src)


class TestEngineUI(unittest.TestCase):
    def setUp(self):
        self.html = (ROOT / "src" / "tsuyaku" / "webui" / "index.html").read_text()

    def test_engine_buttons(self):
        for e in ("auto", "remote", "mac"):
            self.assertIn(f'data-engine="{e}"', self.html)

    def test_noise_buttons(self):
        for n in ("normal", "noisy"):
            self.assertIn(f'data-noise="{n}"', self.html)

    def test_engine_wiring(self):
        self.assertIn("a.get_engine()", self.html)
        self.assertIn("set_engine(", self.html)
        self.assertIn("set_noise(", self.html)
        self.assertIn("#startHint", self.html)


if __name__ == "__main__":
    unittest.main()
