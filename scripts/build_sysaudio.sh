#!/bin/bash
# この Mac の音(Zoom など)を聞く部品 bin/sokutsu-sysaudio を作る(Xcode 不要、clang だけ)。
# ソースが新しくなったときだけ作り直す。失敗してもアプリはマイクだけで動く。
#   使い方: scripts/build_sysaudio.sh [--force]
set -euo pipefail
cd "$(dirname "$0")/.."
SRC=native/sysaudio/sysaudio.c
PLIST=native/sysaudio/Info.plist
OUT=bin/sokutsu-sysaudio
if [[ "${1:-}" != "--force" && -x "$OUT" && "$OUT" -nt "$SRC" && "$OUT" -nt "$PLIST" ]]; then
  exit 0
fi
if ! command -v clang >/dev/null 2>&1; then
  echo "[sysaudio] clang がありません(xcode-select --install)— マイクだけで動きます" >&2
  exit 1
fi
mkdir -p bin
TMP="$OUT.tmp.$$"
trap 'rm -f "$TMP"' EXIT
# Info.plist を実行ファイルに埋める(許可の説明文 NSAudioCaptureUsageDescription)
clang -O2 -Wall -Wextra -fblocks -mmacosx-version-min=13.0 \
  -framework CoreAudio -framework CoreFoundation -lobjc \
  -Wl,-sectcreate,__TEXT,__info_plist,"$PLIST" \
  -o "$TMP" "$SRC"
codesign -s - -f "$TMP" >/dev/null 2>&1 || true
mv -f "$TMP" "$OUT"
echo "[sysaudio] 作りました: $OUT"
