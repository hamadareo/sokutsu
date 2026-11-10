#!/usr/bin/env bash
# Sokutsu(即通)の起動。
# 翻訳の処理場所はアプリ内の『翻訳の処理場所』で選ぶ(既定: 自動 = リモートGPU →
# この Mac の LM Studio)。ここは リモートサーバー へのSSHトンネルを試すだけ。
# 強制指定: TSUYAKU_HOST/TSUYAKU_PORT/TSUYAKU_MODEL/TSUYAKU_ENGINE を自分で
# 設定すればそのまま使う。
set -uo pipefail
cd "$(dirname "$0")"
export PYTHONPATH="$PWD/src"

if [[ -z "${TSUYAKU_HOST:-}" && -z "${TSUYAKU_PORT:-}" ]]; then
  if bash scripts/remote_tunnel.sh; then
    echo "[run] リモートサーバー とんねる接続済み — 既定は『自動』で リモートGPU を使います(起動画面で変更可)"
  else
    echo "[run] リモートサーバー に届きませんでした —『自動』ならこの Mac の LM Studio(ポート1234)で進みます"
  fi
fi

# 日本語の声の VOICEVOX エンジン(このMac, 127.0.0.1:50021)。立っていなければ起こす。
# 無効化: TSUYAKU_NO_VOICEVOX=1 ./run.sh
if [[ -z "${TSUYAKU_NO_VOICEVOX:-}" ]]; then
  bash scripts/voicevox_local.sh || true
fi

# この Mac の音(オンライン会議の相手の声)を聞く部品 bin/sokutsu-sysaudio。
# 無いか古いときだけ作る(数秒)。作れなくてもアプリはマイクだけで動く。
bash scripts/build_sysaudio.sh || echo "[run] bin/sokutsu-sysaudio を作れませんでした —『マイク+この Mac の音』はマイクだけで動きます"

exec .venv/bin/python -u -m tsuyaku
