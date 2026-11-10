"""配る版(他人の Mac 向けの Sokutsu.app)の切り替え・データの場所・処理場所の候補。"""
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tsuyaku import config, distenv


class MarkerTest(unittest.TestCase):
    def test_no_marker_means_user_app(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertIsNone(distenv.load_marker(Path(d) / "_dist.json"))

    def test_marker_is_read(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "_dist.json"
            p.write_text(json.dumps({"version": "1.0.0"}))
            self.assertEqual(distenv.load_marker(p)["version"], "1.0.0")

    def test_broken_marker_still_counts_as_dist(self):
        # 壊れていても「配る版」のまま(リポジトリ直下に書き始めない)
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "_dist.json"
            p.write_text("{")
            self.assertEqual(distenv.load_marker(p), {})

    def test_repo_has_no_marker(self):
        # ユーザー本人の版(run.sh の道)は今のまま
        self.assertFalse(distenv.DIST)
        self.assertEqual(config.DATA, config.ROOT)
        self.assertEqual(config.MODELS, config.ROOT / "models")


class DataDirTest(unittest.TestCase):
    def test_application_support(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("SOKUTSU_DATA_DIR", None)
            self.assertEqual(distenv.data_dir(home="/h"),
                             Path("/h/Library/Application Support/Sokutsu"))

    def test_env_override(self):
        with mock.patch.dict(os.environ, {"SOKUTSU_DATA_DIR": "/x/y"}):
            self.assertEqual(distenv.data_dir(home="/h"), Path("/x/y"))

    def test_logs_dir(self):
        self.assertEqual(distenv.app_logs_dir(home="/h"), Path("/h/Library/Logs/Sokutsu"))


class DistDefaultsTest(unittest.TestCase):
    def _models(self, d: Path) -> Path:
        m = d / "models"
        (m / "sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17").mkdir(parents=True)
        (m / "kokoro-int8-multi-lang-v1_0").mkdir()
        return m

    def test_paths_and_voices(self):
        with tempfile.TemporaryDirectory() as d:
            data = Path(d)
            m = self._models(data)
            s = config.Settings()
            config.apply_dist(s, data, cfg={})
            self.assertEqual(s.log_dir, str(data / "logs"))
            self.assertEqual(s.whisper_model, str(m / "ggml-large-v3-turbo-q5_0.bin"))
            self.assertEqual(s.silero_vad, str(m / "silero_vad.onnx"))
            self.assertTrue(s.sense_voice_dir.endswith("int8-2024-07-17"))
            self.assertTrue(s.tts_dir_kokoro.endswith("kokoro-int8-multi-lang-v1_0"))
            self.assertEqual(s.tts_dir_ko, "")          # 韓国語は say(KSS は非商用なので取らない)
            self.assertEqual(s.stt_url, "")             # リモートの耳は使わない
            self.assertEqual(s.tts_voice["ja"], "say:Kyoko")   # VOICEVOX は入れない
            self.assertEqual(s.model, config.DIST_MODEL)

    def test_saved_choice_wins(self):
        with tempfile.TemporaryDirectory() as d:
            data = Path(d)
            self._models(data)
            s = config.Settings()
            s.tts_voice = {"ja": "voicevox:3", "en": "kokoro:3", "zh": "kokoro:45", "ko": "say:Yuna"}
            config.apply_dist(s, data, cfg={"tts_voice": s.tts_voice})
            self.assertEqual(s.tts_voice["ja"], "voicevox:3")

    def test_models_missing_gives_empty_dirs(self):
        with tempfile.TemporaryDirectory() as d:
            s = config.Settings()
            config.apply_dist(s, Path(d), cfg={})
            self.assertEqual(s.sense_voice_dir, "")
            self.assertEqual(s.tts_dir_kokoro, "")


class DistCandidatesTest(unittest.TestCase):
    LOCAL = "http://127.0.0.1:5555"

    def cands(self, engine, server=""):
        with mock.patch("tsuyaku.localllm.url", return_value=self.LOCAL):
            return config.engine_candidates(engine, "127.0.0.1", 1234, config.DIST_MODEL,
                                            server=server, dist=True)

    def test_mac_is_bundled_server(self):
        self.assertEqual(self.cands("mac"), [("このMac", self.LOCAL, config.DIST_MODEL)])

    def test_auto_without_server_is_this_mac_only(self):
        self.assertEqual([c[1] for c in self.cands("auto")], [self.LOCAL])

    def test_auto_with_server(self):
        c = self.cands("auto", server="http://gpu.example:8080/v1/")
        self.assertEqual([x[1] for x in c], ["http://gpu.example:8080", self.LOCAL])
        self.assertEqual(c[0][0], config.DIST_ENGINE_LABELS["remote"])
        self.assertEqual(c[0][2], "")     # サーバーの最初のモデル

    def test_server_engine_without_url_falls_back(self):
        self.assertEqual([c[1] for c in self.cands("remote")], [self.LOCAL])

    def test_user_app_unchanged(self):
        c = config.engine_candidates("auto", "127.0.0.1", 1234, "google/gemma-3-4b",
                                     server="http://x:1", dist=False)
        self.assertEqual([x[1] for x in c], ["http://127.0.0.1:12345", "http://127.0.0.1:1234"])


if __name__ == "__main__":
    unittest.main()


class BridgeHooksTest(unittest.TestCase):
    """__main__ は処理場所の候補を試す直前に localllm.ensure_for を呼ぶ(配る版の同梱エンジン)。"""

    def test_notes_llm_wakes_bundled_engine_first(self):
        from tsuyaku import __main__ as M
        b = M.Bridge.__new__(M.Bridge)
        b.s = config.Settings()
        b.s.server_url, b.s.server_model = "http://gpu.example:8080", "my-model"
        import threading
        b.running = threading.Event()
        order = []
        with mock.patch.object(M, "engine_candidates",
                               return_value=[("このMac", "http://127.0.0.1:5555", "g")]) as ec, \
                mock.patch.object(M.localllm, "ensure_for",
                                  side_effect=lambda base, *a, **k: order.append(("ensure", base))), \
                mock.patch.object(M, "warmup", side_effect=lambda b_, m: order.append(("warm", b_))), \
                mock.patch.object(M, "pick_model", return_value="g"):
            self.assertEqual(b._notes_llm(), ("http://127.0.0.1:5555", "g"))
        self.assertEqual(order[0], ("ensure", "http://127.0.0.1:5555"))
        self.assertEqual(ec.call_args.kwargs.get("server"), "http://gpu.example:8080")
        self.assertEqual(ec.call_args.kwargs.get("server_model"), "my-model")
