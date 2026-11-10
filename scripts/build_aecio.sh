#!/bin/bash
# 読み上げを鳴らしながらマイクを聞く部品(エコー消し)bin/sokutsu-aecio を作る(Xcode 不要、clang だけ)。
# ソースが新しくなったときだけ作り直す。失敗してもアプリはいつもの道(マイク + afplay)で動く。
#   使い方: scripts/build_aecio.sh [--force]
set -euo pipefail
cd "$(dirname "$0")/.."
SRC=native/aecio/aecio.c
PLIST=native/aecio/Info.plist
OUT=bin/sokutsu-aecio
if [[ "${1:-}" != "--force" && -x "$OUT" && "$OUT" -nt "$SRC" && "$OUT" -nt "$PLIST" ]]; then
  exit 0
fi
if ! command -v clang >/dev/null 2>&1; then
  echo "[aecio] clang がありません(xcode-select --install)— いつもの道で動きます" >&2
  exit 1
fi
mkdir -p bin
TMP="$OUT.tmp.$$"
trap 'rm -f "$TMP"' EXIT
# Info.plist を実行ファイルに埋める(許可の説明文 NSMicrophoneUsageDescription)
clang -O2 -Wall -Wextra -fblocks -mmacosx-version-min=13.0 \
  -framework AudioToolbox -framework CoreAudio -framework CoreFoundation \
  -Wl,-sectcreate,__TEXT,__info_plist,"$PLIST" \
  -o "$TMP" "$SRC"
codesign -s - -f "$TMP" >/dev/null 2>&1 || true
mv -f "$TMP" "$OUT"
echo "[aecio] 作りました: $OUT"
