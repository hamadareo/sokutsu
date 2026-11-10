#!/usr/bin/env bash
# VOICEVOX ENGINE(このMacローカル, 127.0.0.1:50021)を起動しておく。
# すでに立っていれば何もしない(二重起動しない)。本体は ~/voicevox
# (VOICEVOX ENGINE 0.25.2, CPU版。公式の配布物を展開したもの)。
# 起動したくないときは TSUYAKU_NO_VOICEVOX=1 を付けて run.sh を実行すると
# 日本語の声は Mac 標準 say になる。
set -uo pipefail
VV_HOME="${VOICEVOX_HOME:-$HOME/voicevox}"
HOST="127.0.0.1"
PORT="50021"
LOG="${VOICEVOX_LOG:-$HOME/voicevox.log}"

if curl -sf -m 1 "http://$HOST:$PORT/version" >/dev/null 2>&1; then
  echo "[voicevox] 起動済み (http://$HOST:$PORT)"
  exit 0
fi
if [[ ! -x "$VV_HOME/run" ]]; then
  echo "[voicevox] $VV_HOME/run が見つからないためスキップします(日本語の声は Mac 標準 say)"
  exit 0
fi
mkdir -p "$(dirname "$LOG")"
(cd "$VV_HOME" && nohup "$VV_HOME/run" --host "$HOST" --port "$PORT" >>"$LOG" 2>&1 &)
for _ in $(seq 1 30); do
  if curl -sf -m 1 "http://$HOST:$PORT/version" >/dev/null 2>&1; then
    echo "[voicevox] 起動しました (http://$HOST:$PORT)"
    exit 0
  fi
  sleep 0.5
done
echo "[voicevox] 起動を15秒待てませんでした(日本語の声は Mac 標準 say を使います)"
exit 0
