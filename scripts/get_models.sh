#!/usr/bin/env bash
# 実行に必要なモデルを models/ にダウンロードする(初回のみ・途中再開可)。
# 合計 約1.8GB。LLM(gemma等)はLM Studio側で別途ダウンロードする。
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p models

dl() { # dl <保存先> <url>
  if [[ -e "$1" ]]; then echo "skip: $1"; return; fi
  echo "down: $2"
  curl -L -C - --speed-time 45 --speed-limit 200000 --retry 3 -o "$1.part" "$2"
  mv "$1.part" "$1"
}

dl models/ggml-large-v3-turbo.bin \
  "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3-turbo.bin"
dl models/silero_vad.onnx \
  "https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad.onnx"

# 全精度版(11/06〜): M3 の CPU では int8 より2倍速く(10秒の声を2.5秒で)、雑音の山も小さい
if [[ ! -d models/kokoro-multi-lang-v1_0 ]]; then
  dl models/kokoro.tar.bz2 \
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/kokoro-multi-lang-v1_0.tar.bz2"
  tar -xjf models/kokoro.tar.bz2 -C models && rm models/kokoro.tar.bz2
fi
if [[ ! -d models/vits-mimic3-ko_KO-kss_low ]]; then
  dl models/kss.tar.bz2 \
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-mimic3-ko_KO-kss_low.tar.bz2"
  tar -xjf models/kss.tar.bz2 -C models && rm models/kss.tar.bz2
fi
# このMacの速い耳(11/06): SenseVoice-Small(ja/en/zh/ko/yue, 1回0.15秒)
if [[ ! -d models/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17 ]]; then
  dl models/sv.tar.bz2 \
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17.tar.bz2"
  tar -xjf models/sv.tar.bz2 -C models && rm models/sv.tar.bz2
fi
# 話者の見分け(会議メモ 11/06): 3D-Speaker CAM++(zh+en 学習、日英中韓の混ざる会議で比べて採用、28MB)
dl models/3dspeaker_speech_campplus_sv_zh_en_16k-common_advanced.onnx \
  "https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/3dspeaker_speech_campplus_sv_zh_en_16k-common_advanced.onnx"
echo "OK: models/ 準備完了"
