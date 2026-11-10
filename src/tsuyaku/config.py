"""通訳の設定。環境変数とconfig.jsonで上書きできる。"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

from . import distenv
from .distenv import DIST

ROOT = Path(__file__).resolve().parents[2]
# 設定・記録・用語・モデルの場所。ユーザー本人の版はリポジトリ直下(今のまま)、
# 配る版は ~/Library/Application Support/Sokutsu/(distenv.py)。
DATA = distenv.DATA if DIST else ROOT
MODELS = DATA / "models"


def _config_file() -> Path:
    """config.json の場所(呼ぶたびに決める: 試験は ROOT を一時フォルダに差し替える)。"""
    return (DATA if DIST else ROOT) / "config.json"

LANG_NAMES = {"ja": "日本語", "en": "英語", "zh": "中国語", "ko": "韓国語"}

# 処理場所(翻訳LLMをどこで走らせるか)。リモートサーバー は 翻訳ゲートウェイ へのSSH転送ポート。
ENGINE_LABELS = {"auto": "自動", "remote": "リモートGPU", "mac": "このMac"}
# Mac は qwen3-4b-2507(2.3GB)。gemma-3-12b(8GB)は16GBのMacで whisper/VOICEVOX と
# 同居するとスワップを使い切り、翻訳1件45〜67秒・タイムアウト連発(11/05ライブ)。
# 同じ8文の実測: gemma-3-12b 中央値6.12秒 → qwen3-4b-2507 0.74秒、品質は同等。
# 11/06: gemma-3-4b(MLX QAT 4bit)へ。FLEURS 合成会話4ペア(耳=SenseVoice)で qwen3-4b より
# 字幕が速く(日中 1.51→1.14s・日韓 1.92→1.64s・日英 1.04→0.87s)、chrF は同等以上
# (英中だけ -2)。人名・校名も正しく訳した(聞き違えた人名・校名を、テーマの綴りに直して訳した)。
ENGINE_MODELS = {"remote": "qwen3.8-flash-next", "mac": "google/gemma-3-4b"}
REMOTE_PORT = 12345
AUDIO_SOURCES = ("mic", "mic+system")
MAC_PORT = 1234

# 配る版(11/07): 処理場所の札は同じ3つ。真ん中は「自分のサーバー」(config.json の server_url、
# OpenAI 互換。既定は空 = 札を隠す)。「このMac」は同梱の llama-server(localllm.py)。
DIST_MODEL = "google/gemma-3-4b"   # llama-server の --alias(今の Mac の既定と同じ id)
DIST_ENGINE_LABELS = {"auto": "自動", "remote": "自分のサーバー", "mac": "このMac"}
if DIST:
    ENGINE_LABELS = dict(DIST_ENGINE_LABELS)


def _server_base(url: str) -> str:
    """http://host:port/v1/ → http://host:port(translate は base + /v1/... で呼ぶ)。"""
    u = (url or "").strip().rstrip("/")
    return u[:-3].rstrip("/") if u.endswith("/v1") else u


def _dist_candidates(engine: str, server: str, server_model: str) -> list[tuple[str, str, str]]:
    from . import localllm
    local = (DIST_ENGINE_LABELS["mac"], localllm.url(), DIST_MODEL)
    srv = ((DIST_ENGINE_LABELS["remote"], _server_base(server), server_model or "")
           if _server_base(server) else None)
    if engine == "mac" or srv is None:
        return [local]
    if engine == "remote":
        return [srv]
    return [srv, local]


def engine_candidates(engine: str = "auto", host: str = "127.0.0.1",
                      port: int | None = None,
                      model: str | None = None, *, server: str = "",
                      server_model: str = "",
                      dist: "bool | None" = None) -> list[tuple[str, str, str]]:
    """処理場所 → 試す順の [(ラベル, base_url, 推奨モデル)]。

    auto: リモートGPU を先に試し、届かなければこの Mac に落ちる。
    リモートサーバー / mac: その1つだけ。TSUYAKU_PORT でカスタムポートを明示した場合は
    それを従う(処理場所よりカスタム指定を優先。以前の起動方法との互換)。
    配る版(dist): このMac = 同梱の llama-server、自分のサーバー = server(空なら使わない)。
    """
    if DIST if dist is None else dist:
        return _dist_candidates(engine, server, server_model)
    if port is not None and port not in (REMOTE_PORT, MAC_PORT):
        return [("カスタム", f"http://{host}:{port}", model or ENGINE_MODELS["mac"])]
    if engine == "remote":
        return [(ENGINE_LABELS["remote"], f"http://{host}:{REMOTE_PORT}",
                 model or ENGINE_MODELS["remote"])]
    if engine == "mac":
        return [(ENGINE_LABELS["mac"], f"http://{host}:{MAC_PORT}",
                 model or ENGINE_MODELS["mac"])]
    return [(ENGINE_LABELS["remote"], f"http://{host}:{REMOTE_PORT}",
             ENGINE_MODELS["remote"]),
            (ENGINE_LABELS["mac"], f"http://{host}:{MAC_PORT}",
             ENGINE_MODELS["mac"])]


@dataclass
class Settings:
    host: str = "127.0.0.1"                # LM Studio サーバー(リモート可: リモートサーバートンネル先)
    port: int = 1234                       # LM Studio サーバー
    model: str = "google/gemma-3-4b"      # /v1/models の id。見つからなければ最初のLLMへ
    whisper_model: str = str(MODELS / "ggml-large-v3-turbo.bin")
    # このMacの耳(11/06): "sensevoice"(既定, 1回0.15秒) / "whisper"(turbo, 1回2.4秒・誤りは約半分)
    mac_ear: str = "sensevoice"
    # このMacの耳(SenseVoice)が日本語の中の英字の略語を落とした見当の文を、裏で whisper に聞き直して
    # 直す(11/08 relisten.py)。リモートサーバーの耳のときは使わない
    relisten: bool = True
    sense_voice_dir: str = ""
    # リモートGPU の耳(SSH転送 127.0.0.1:12346 → remote:9941)。翻訳を リモートサーバー でやるときだけ使う。
    stt_url: str = "http://127.0.0.1:12346"
    silero_vad: str = str(MODELS / "silero_vad.onnx")
    tts_dir_kokoro: str = ""               # models/kokoro-int8-multi-lang-v1_0
    tts_dir_ko: str = ""                   # models/vits-mimic3-ko_KO-kss_low
    # kokoro-multi-lang-v1_0 の公式sid表: en=0-27, ja=37-41, zh=45-52
    tts_sids: dict = field(default_factory=lambda: {"ja": 37, "zh": 45, "en": 3, "ko": 0})
    tts_ja_engine: str = "say"             # 日本語読み上げ: "say"(既定, Kyoko) or "kokoro"(実験的)
    # VOICEVOX ENGINE(このMacローカル, CPU)。起動は run.sh → scripts/voicevox_local.sh。
    # 届かなくても音声は止まらず macOS say にフォールバックする。
    voicevox_url: str = "http://127.0.0.1:50021"
    # 言語ごとの声: "say:音色名"(macOS音声) / "kokoro:sid"(Kokoro) / "voicevox:sid"(VOICEVOX)。
    # UIから変えて保存される。日本語は VOICEVOX 青山龍星(13)を既定、エンジン不在なら say。
    tts_voice: dict = field(default_factory=lambda: {
        "ja": "voicevox:13", "en": "kokoro:3", "zh": "kokoro:45", "ko": "say:Yuna"})
    tts_speed: dict = field(default_factory=lambda: {
        "ja": 1.0, "en": 1.0, "zh": 1.0, "ko": 1.0})
    # 読み上げの速さの決め方(11/08): "auto" = 話のペースに合わせる(上の速さは下限)/ "fixed" = 上の速さのまま。
    # 新しく使う人は auto。前から速さを決めていた人(config.json に tts_speed があって これが無い)は fixed。
    tts_speed_mode: dict = field(default_factory=lambda: {
        "ja": "auto", "en": "auto", "zh": "auto", "ko": "auto"})
    # 読み上げ中も聞く(11/08): "auto" = 耳元の機器(ヘッドホン・Bluetooth)なら聞き続け、「この Mac の音」も
    # 聞いているならマイクだけ止めてそちらは聞く / "always" = いつも聞く / "off" = 止める(今まで)
    listen_while_speaking: str = "auto"
    # 自動の速さの上限(11/08): "clear" 聞きやすさ優先(測った上限)/ "fast" 速さ優先(VOICEVOX・Kokoro ×1.5)
    tts_speed_priority: str = "clear"
    # 読み上げの音量(0.1〜1.0。システムの音量に対する割合。画面の , . キー 11/10)
    tts_volume: float = 1.0
    context_turns: int = 4                 # 翻訳promptに添える直前ターン数
    gate_tail_s: float = 0.25              # 読み上げが終わってからマイクを開けるまでの余裕
    sample_rate: int = 16000
    min_speech_s: float = 0.35
    max_speech_s: float = 12.0
    min_silence_s: float = 0.5             # 区切り待ち0.6→0.5秒(11/05: スムーズな会話のため)
    log_dir: str = str(DATA / "logs")
    # 処理場所: "auto"(remote→このMacの順に自動) / "remote" / "mac"。UIから変えて保存される。
    engine: str = "auto"
    # VAD感度: 0.5=静かな部屋(既定) / 0.65=騒がしい会場。再起動なしで次の会話から効く。
    vad_threshold: float = 0.5
    # このRMS未満のセグメントは部屋ノイズとして whisper を走らせない(11/05詰まり対策)
    noise_rms_floor: float = 0.006
    # 続き待ちマージ(11/05): 文末が助詞/読点の短い断片は hold_open_s だけ次の
    # 聞き取りを待ち、同じ言語なら連結して1文として訳す。続きが来なければそのまま訳す。
    # 連結は hold_max_parts 回まで(=断片 hold_max_parts+1 個まで)。0で無効化。
    hold_open_s: float = 1.2
    hold_open_max: int = 20    # normがこの字数以上の断片は待たない(長い文は先に訳す)
    hold_max_parts: int = 2
    # 先読み翻訳(11/06): 文末の続き待ちの間に、今の文字で訳を始めておく(speculate.py)。
    speculate: bool = True
    # 途中字幕(このMacの耳の途中の聞き取り)からも予想訳を作るか(11/08 speed の評価用)
    speculate_preview: bool = True
    # リモートサーバーの耳を待つ上限(11/06): 回線が混むと往復が0.3〜7秒に揺れる。これを過ぎたら
    # このMacの速い耳(SenseVoice 0.05〜0.15秒)の結果を使う。0 = 待ち切る。
    # 確定の品質は リモートサーバーのまま: 0.7秒/2.5秒で切ると混んだ回線で chrF -2〜4 だったので、
    # 今までの時間切れ(4秒)と同じにする。速さは下の予想訳の先出しで出す。
    stt_hedge_s: float = 4.0
    # これを過ぎても リモートサーバー が返らなければ、このMacの耳の結果で予想訳だけ先に出す。
    stt_preview_s: float = 0.35
    # 先送り(11/06, uplink.py): 話している間に声を リモートサーバーの耳へ送っておき、聞くときは範囲だけ
    # 送る(家の回線で耳の往復 中央1.9秒 → 範囲だけなら音声の大きさに関係なく1往復)。
    stt_stream: bool = True
    # 前回のテーマメモ(人名・専門用語)。次の起動で入力欄に戻す(11/06)。
    last_topic: str = ""
    # 読み上げ(11/06): False = 字幕だけ。声を出さないのでマイクを止めず、話し続けても全部聞く。
    voice_on: bool = True
    # 読み上げの順番が来たとき、話し終わりからこれ以上たっていたら読まない(字幕は出ている)。
    voice_max_lag_s: float = 8.0
    # 話のペースに合わせる(11/08): 話し終わりからこれ以上遅れていたら、行列の同じ言語の訳をまとめて読む
    voice_join_after_s: float = 2.0
    # 途中字幕(11/06 英会話を流して「もっとリアルタイムに」): 話し続けている最中も
    # この秒数ごとに、ここまでの音声をこのMacの速い耳(SenseVoice)で聞いて原文と予想訳を出す。
    # 確定は今までどおり区間が閉じてから リモートサーバーの耳で聞いた結果(品質は同じ)。0 で無効。
    interim_s: float = 1.5
    # 長い発話を区切る(11/08): 話し続けている区間がこの秒数を超えたら、途中字幕の文末の所で前半を確定
    # して先に訳・読み上げへ流す(読点は long_comma_s、句読点の無い息つぎは long_breath_s 秒から)。0 で切らない。
    # 既定は切らない: 作った独白 32 本(eval/long)で zh・BGM の遅れ p90 は 8.3 → 7.0 秒・単位の最長 8.2 → 5.9 秒に
    # なったが、訳の chrF が4つの組すべてで 0.1〜1.7 下がった(速さのために品質を落とさない約束)
    long_split_s: float = 0.0
    long_comma_s: float = 9.0
    long_breath_s: float = 9.0
    # 使い方(11/06): interpret=通訳だけ / notes=会議メモだけ(訳さない・読まない) / both=通訳+会議メモ
    use_mode: str = "interpret"
    # 聞く音(11/07 オンライン会議): "mic"=このMacのマイクだけ / "mic+system"=マイク + この Mac で
    # 鳴っている音(Zoom・Teams・Meet の相手の声。bin/sokutsu-sysaudio、Sokutsu 自身の読み上げは除く)
    audio_source: str = "mic"
    # 用語の分野(11/08 IT 用語): "" = 自動(テーマ欄に IT の語があれば IT)/ "it" = IT・技術(IT の用語集を
    # 優先し、一般の文で意味が割れる形 = process・server なども使う)/ "general" = 一般
    term_domain: str = ""
    # この Mac の音を足すときの倍率(会議アプリの音はマイクの声より大きいことが多い)
    system_gain: float = 1.0
    # 録音(W2, 11/07): "auto" = 会議メモを使うとき(notes/both)だけ logs/ に残す / "on" / "off"
    record_audio: str = "auto"
    # 会議の種類(11/07 W1, note_templates.py): free=自由(今までのメモ)/ regular / 1on1 / sales /
    # interview / lecture / brainstorm / qa。起動画面で選び、記録の meta の template に入る。
    meeting_template: str = "free"
    # 画面の明るさ(11/07): "system"=macOS に合わせる / "light" / "dark"
    ui_theme: str = "system"
    # 入力・出力デバイス(11/08 hear): 名前で残す(番号は抜き差しで変わる)。"" = システムの設定に合わせる。
    # 無ければシステムの既定で続け、画面にそっと知らせる(devices.py)
    input_device: str = ""
    output_device: str = ""
    # 声を聞き取っている間、窓の縁を光らせる(11/08 hear。画面の設定で切れる)
    edge_glow: bool = True
    # 画面の隅に小さく出す、この Mac の使用率(CPU・メモリ・GPU)。設定で切れる。切ると取得もしない(11/10)
    show_sysmon: bool = True
    # 画面の隅に小さく出す、この Mac の使用率(CPU・メモリ・GPU)。設定で切れる。切ると取得もしない(11/10)
    show_sysmon: bool = True
    # 配る版(11/07): 自分の翻訳サーバー(OpenAI 互換の URL、例 http://192.0.2.5:8080)。既定は空 =
    # この Mac の同梱エンジンだけ。server_model が空ならサーバーの最初のモデル。
    server_url: str = ""
    server_model: str = ""


# 配る版のモデル(初回に modelfetch.py が取る)。重複(fp16 の whisper・fp の kokoro)は取らない。
DIST_WHISPER = "ggml-large-v3-turbo-q5_0.bin"
DIST_SENSE_VOICE = "sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17"
DIST_KOKORO = "kokoro-int8-multi-lang-v1_0"


def resolve_dist_models(s: "Settings", data: Path) -> None:
    """配る版のモデルの場所(初回の取得が終わった後にも呼び直す)。"""
    m = Path(data) / "models"
    s.whisper_model = str(m / DIST_WHISPER)
    s.silero_vad = str(m / "silero_vad.onnx")
    s.sense_voice_dir = str(m / DIST_SENSE_VOICE) if (m / DIST_SENSE_VOICE).is_dir() else ""
    s.tts_dir_kokoro = str(m / DIST_KOKORO) if (m / DIST_KOKORO).is_dir() else ""
    s.tts_dir_ko = ""                      # 韓国語は macOS say(vits の KSS は非商用なので取らない)


def apply_dist(s: "Settings", data: Path, cfg: dict) -> None:
    """配る版の既定。cfg(config.json の中身)に書いてある値はそのまま(利用者の選択が勝つ)。"""
    s.log_dir = str(Path(data) / "logs")
    resolve_dist_models(s, data)
    s.stt_url = ""                         # リモートの耳(研究室のサーバー)は使わない
    if "model" not in cfg:
        s.model = DIST_MODEL
    if "tts_voice" not in cfg:
        s.tts_voice = dict(s.tts_voice, ja="say:Kyoko")   # VOICEVOX は入れない
    if s.engine not in DIST_ENGINE_LABELS:
        s.engine = "auto"


def _same_kind(default, v) -> bool:
    """config.json の値が既定と同じ種類か(float の欄に int は可。bool と数は別)。"""
    if isinstance(default, bool) or isinstance(v, bool):
        return isinstance(default, bool) and isinstance(v, bool)
    if isinstance(default, float):
        return isinstance(v, (int, float))
    return isinstance(v, type(default))


def _read_config(cfg: Path) -> dict:
    """config.json を読む。壊れている(書きかけで落ちた・配る版の説明書どおり手で server_url を足した時の
    書き間違いなど)ときは既定で起動し、壊れたものは config.broken-<時刻>.json に名前を変えて残す
    (以前は JSONDecodeError で窓も出ずに毎回すぐ終わっていた。11/07 監査)。"""
    if not cfg.exists():
        return {}
    try:
        saved = json.loads(cfg.read_text(encoding="utf-8"))
        if isinstance(saved, dict):
            return saved
    except (OSError, ValueError):
        pass
    import time as _t
    keep = cfg.with_name(f"config.broken-{_t.strftime('%Y%m%d-%H%M%S')}.json")
    try:
        cfg.replace(keep)
        print(f"[config] config.json が読めないので既定で始めます(元は {keep.name})", flush=True)
    except OSError:
        print("[config] config.json が読めないので既定で始めます", flush=True)
    return {}


def load_settings() -> Settings:
    s = Settings()
    cfg = _config_file()
    saved = _read_config(cfg)
    for k, v in saved.items():
        if hasattr(s, k) and not k.startswith("_"):
            if _same_kind(getattr(s, k), v):
                setattr(s, k, v)
            else:   # 古い形・手で書き違えた値: 既定のまま(後で int() や .get() で落ちない)
                print(f"[config] {k} の値 {v!r} は使えないので既定 {getattr(s, k)!r} にします", flush=True)
    s.tts_speed_mode = {k: (v if v in ("auto", "fixed") else "fixed")
                        for k, v in {**Settings().tts_speed_mode, **s.tts_speed_mode}.items()}
    if s.listen_while_speaking not in ("auto", "always", "off"):
        s.listen_while_speaking = "auto"
    if s.tts_speed_priority not in ("clear", "fast"):
        s.tts_speed_priority = "clear"
    if "tts_speed" in saved and "tts_speed_mode" not in saved:
        # 前から速さを自分で決めていた人: その速さのまま(自動は画面で選べる)
        s.tts_speed_mode = {k: "fixed" for k in s.tts_speed_mode}
    if DIST:
        apply_dist(s, DATA, saved)
    s.host = os.environ.get("TSUYAKU_HOST", s.host)
    s.port = int(os.environ.get("TSUYAKU_PORT", s.port))
    s.model = os.environ.get("TSUYAKU_MODEL", s.model)
    s.engine = os.environ.get("TSUYAKU_ENGINE", s.engine)
    s.voicevox_url = os.environ.get("TSUYAKU_VOICEVOX_URL", s.voicevox_url)
    s.stt_url = os.environ.get("TSUYAKU_STT_URL", s.stt_url)
    s.hold_open_s = float(os.environ.get("TSUYAKU_HOLD_S", s.hold_open_s))
    if os.environ.get("TSUYAKU_INTERIM"):
        s.interim_s = float(os.environ["TSUYAKU_INTERIM"])
    if os.environ.get("TSUYAKU_LONG_SPLIT"):   # 評価の A/B 用(0 で長い発話を区切らない)
        s.long_split_s = float(os.environ["TSUYAKU_LONG_SPLIT"])
    if os.environ.get("TSUYAKU_STT_HEDGE"):
        s.stt_hedge_s = float(os.environ["TSUYAKU_STT_HEDGE"])
    if os.environ.get("TSUYAKU_STT_STREAM"):
        s.stt_stream = os.environ["TSUYAKU_STT_STREAM"] not in ("0", "false", "off")
    s.audio_source = os.environ.get("TSUYAKU_AUDIO_SOURCE", s.audio_source)
    if s.audio_source not in AUDIO_SOURCES:
        s.audio_source = "mic"
    if os.environ.get("TSUYAKU_SPECULATE"):
        s.speculate = os.environ["TSUYAKU_SPECULATE"] not in ("0", "false", "off")
    if not s.tts_dir_kokoro and not DIST:
        cand = sorted(MODELS.glob("kokoro*"))
        full = [c for c in cand if "int8" not in c.name]  # 全精度版があればそちらを優先
        s.tts_dir_kokoro = str((full or cand)[0]) if cand else ""
    if not s.sense_voice_dir and not DIST:
        cand = sorted(MODELS.glob("sherpa-onnx-sense-voice*"))
        s.sense_voice_dir = str(cand[0]) if cand else ""
    s.mac_ear = os.environ.get("TSUYAKU_MAC_EAR", s.mac_ear)
    if os.environ.get("TSUYAKU_RELISTEN"):
        s.relisten = os.environ["TSUYAKU_RELISTEN"] not in ("0", "false", "off")
    if not s.tts_dir_ko and not DIST:
        cand = sorted(MODELS.glob("*ko*"))
        s.tts_dir_ko = str(cand[0]) if cand else ""
    return s


def save_config(s: Settings) -> None:
    """UIから変えられる設定だけを config.json に書き戻す(他設定は温存)。"""
    cfg = _config_file()
    data: dict = {}
    try:
        if cfg.exists():
            data = json.loads(cfg.read_text(encoding="utf-8"))
    except Exception:
        data = {}
    if not isinstance(data, dict):
        data = {}
    data["engine"] = s.engine
    data["vad_threshold"] = s.vad_threshold
    data["noise_rms_floor"] = s.noise_rms_floor
    data["tts_voice"] = s.tts_voice
    data["tts_speed"] = s.tts_speed
    data["tts_speed_mode"] = s.tts_speed_mode
    data["listen_while_speaking"] = s.listen_while_speaking
    data["tts_speed_priority"] = s.tts_speed_priority
    data["tts_volume"] = s.tts_volume
    data["last_topic"] = s.last_topic
    data["voice_on"] = s.voice_on
    data["use_mode"] = s.use_mode
    data["audio_source"] = s.audio_source
    data["term_domain"] = s.term_domain
    data["record_audio"] = s.record_audio
    data["meeting_template"] = s.meeting_template
    data["ui_theme"] = s.ui_theme
    data["input_device"] = s.input_device      # 名前(11/08 hear)
    data["output_device"] = s.output_device
    data["edge_glow"] = s.edge_glow
    data["show_sysmon"] = s.show_sysmon
    data["show_sysmon"] = s.show_sysmon
    # 隣に書いてから差し替える(書いている途中でディスクが一杯・落ちると、以前は config.json が
    # 途中で切れ、次の起動で読めなかった)
    tmp = cfg.with_name(cfg.name + ".tmp")
    try:
        cfg.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp, cfg)
    except Exception:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
