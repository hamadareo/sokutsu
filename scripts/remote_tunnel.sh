#!/usr/bin/env bash
# 自分のリモート推論サーバー(任意)への SSH トンネル。
# リモートの OpenAI 互換 LLM サーバー(llama-server など)を、この Mac の 127.0.0.1:12345 へ転送し、
# 任意で音声認識サーバー(server/remote_stt_server.py)を 127.0.0.1:12346 へ転送する。
# すでに転送中なら何もしない(冪等)。届かなければ exit 1(アプリは「このMac」で進む)。
#
# 必須: TSUYAKU_SSH=user@host (ssh の鍵認証で入れること。未設定なら何もせず exit 1)
# 変更できるもの(環境変数):
#   TSUYAKU_SSH_PORT(22)  TSUYAKU_LOCAL_PORT(12345)  TSUYAKU_REMOTE_PORT(9931: リモートの LLM サーバーのポート)
#   TSUYAKU_STT_LOCAL_PORT(12346)  TSUYAKU_STT_REMOTE_PORT(9941)
#   TSUYAKU_STT_VENV(リモート側の faster-whisper 入り venv。既定 $HOME/tsuyaku_stt/venv)
#
# 使い方: TSUYAKU_SSH=user@host bash scripts/remote_tunnel.sh
set -uo pipefail

SSH_HOST="${TSUYAKU_SSH:-}"
if [[ -z "$SSH_HOST" ]]; then
  echo "[tunnel] TSUYAKU_SSH(user@host)が未設定です。リモートサーバーは使わず、このMacで進みます"
  exit 1
fi
SSH_PORT="${TSUYAKU_SSH_PORT:-22}"
LOCAL_PORT="${TSUYAKU_LOCAL_PORT:-12345}"
REMOTE_PORT="${TSUYAKU_REMOTE_PORT:-9931}"        # リモートの LLM サーバー(または翻訳ゲートウェイ)

probe() { curl -sf -m 2 "http://127.0.0.1:$LOCAL_PORT/v1/models" >/dev/null 2>&1; }

# ---- リモートサーバーの耳(音声認識)。11/05: このMacの whisper 2.3秒 → リモートGPU 0.13〜0.26秒 ----
# server/remote_stt_server.py を リモートサーバーの ~/tsuyaku_stt へ置き、止まっていれば起こし
# (30分アイドルで自分で終わる)、127.0.0.1:12346 → remote:9941 を転送する。
# 失敗しても翻訳の転送は続ける(アプリはこのMacの whisper で認識する)。
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
STT_LOCAL="${TSUYAKU_STT_LOCAL_PORT:-12346}"
STT_REMOTE="${TSUYAKU_STT_REMOTE_PORT:-9941}"
# faster-whisper 入りの venv。$HOME は remote 側で展開させる。
STT_VENV="${TSUYAKU_STT_VENV:-}"
[[ -z "$STT_VENV" ]] && STT_VENV='$HOME/tsuyaku_stt/venv'
stt_probe() { curl -sf -m 2 "http://127.0.0.1:$STT_LOCAL/health" >/dev/null 2>&1; }
# 耳のサーバーの版(server/remote_stt_server.py の VER)。動いている版が古ければ置き換えて起動し直す
# (11/06 先送り: /append と範囲での聞き取りは VER 2 から)。
STT_VER="$(sed -n 's/^VER = \([0-9]*\).*/\1/p' "$ROOT/server/remote_stt_server.py" | head -1)"
stt_ver() { curl -sf -m 2 "http://127.0.0.1:$STT_LOCAL/health" 2>/dev/null | sed -n 's/.*"ver": *\([0-9]*\).*/\1/p'; }
ensure_stt() {
  local restart=""
  if stt_probe; then
    if [[ "$(stt_ver)" == "$STT_VER" ]]; then
      echo "[tunnel] 耳: すでに 127.0.0.1:$STT_LOCAL で転送中"; return 0
    fi
    echo "[tunnel] 耳: リモートサーバーの耳の版が古い($(stt_ver) → $STT_VER)。置き換えて起動し直す"
    restart=1
  fi
  ssh -o BatchMode=yes -o ConnectTimeout=5 "$SSH_HOST" 'mkdir -p ~/tsuyaku_stt' </dev/null \
    && scp -q -o BatchMode=yes "$ROOT/server/remote_stt_server.py" "$SSH_HOST:tsuyaku_stt/" \
    || { echo "[tunnel] 耳: リモートサーバーへ配置できない → このMacで認識"; return 1; }
  if [[ -n "$restart" ]]; then
    # 止めるのはこの通訳アプリの耳(remote_stt_server.py --port $STT_REMOTE)だけ
    ssh -o BatchMode=yes -o ConnectTimeout=5 "$SSH_HOST" \
      "pkill -f 'remote_stt_server.py --port $STT_REMOTE' ; sleep 1" </dev/null || true
  fi
  # setsid -f で完全に切り離す(nohup & だと ssh がサーバー終了まで戻らなかった)
  ssh -o BatchMode=yes -o ConnectTimeout=5 "$SSH_HOST" \
    "curl -sf -m 2 http://127.0.0.1:$STT_REMOTE/health >/dev/null 2>&1 || (cd ~/tsuyaku_stt && setsid -f $STT_VENV/bin/python remote_stt_server.py --port $STT_REMOTE > stt.log 2>&1 < /dev/null)" </dev/null
  if ! lsof -nP -iTCP:"$STT_LOCAL" -sTCP:LISTEN >/dev/null 2>&1; then
    ssh -f -N -o BatchMode=yes -o ExitOnForwardFailure=yes \
      -o ServerAliveInterval=20 -o ServerAliveCountMax=3 \
      -L "127.0.0.1:$STT_LOCAL:127.0.0.1:$STT_REMOTE" "$SSH_HOST" </dev/null
  fi
  for _ in $(seq 1 30); do stt_probe && break; sleep 1; done   # モデル読み込み 4秒前後
  if stt_probe; then
    echo "[tunnel] 耳: 127.0.0.1:$STT_LOCAL → remote:$STT_REMOTE (faster-whisper GPU) 準備完了"
  else
    echo "[tunnel] 耳: リモートサーバーが応答しない → このMacで認識"
  fi
}

# LISTEN していても SSH チャンネルが死んでいることがある(11/05: 古い転送が
# LISTEN だけ残し、アプリの warmup が 180秒ハングした)。HTTPで実応答を見る。
if lsof -nP -iTCP:"$LOCAL_PORT" -sTCP:LISTEN >/dev/null 2>&1; then
  if probe; then
    echo "[tunnel] すでに 127.0.0.1:$LOCAL_PORT で転送中"
    ensure_stt
    exit 0
  fi
  echo "[tunnel] $LOCAL_PORT は生きた転送ではない(古い転送?)。捨てて張り直す"
  stale=$(lsof -nP -tiTCP:"$LOCAL_PORT" -sTCP:LISTEN 2>/dev/null || true)
  if [[ -n "$stale" ]]; then kill $stale 2>/dev/null || true; sleep 0.5; fi
fi

SSH_HOSTNAME="${SSH_HOST#*@}"
if ! nc -z -G 3 "$SSH_HOSTNAME" "$SSH_PORT" >/dev/null 2>&1; then
  echo "[tunnel] $SSH_HOSTNAME に届かない(リモートサーバーが offline?)。Mac ローカルの LM Studio で進む"
  exit 1
fi

if ssh -f -N -o BatchMode=yes -o ExitOnForwardFailure=yes \
     -o ServerAliveInterval=20 -o ServerAliveCountMax=3 \
     -L "127.0.0.1:$LOCAL_PORT:127.0.0.1:$REMOTE_PORT" "$SSH_HOST"; then
  for _ in 1 2 3 4 5; do probe && break; sleep 1; done
  if probe; then
    echo "[tunnel] 127.0.0.1:$LOCAL_PORT → remote:$REMOTE_PORT (翻訳ゲートウェイ) 開始"
    ensure_stt
    exit 0
  fi
  echo "[tunnel] 転送を貼ったがゲートが応答しない(remote側 翻訳ゲートウェイ 確認)"
  exit 1
fi
echo "[tunnel] SSH 転送に失敗。Mac ローカルの LM Studio で進む"
exit 1
