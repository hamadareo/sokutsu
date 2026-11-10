#!/bin/bash
# Sokutsu の Dock アイコン PNG(1024 角・透過)を icon.svg から作る。
# 依存: Google Chrome(headless)。背景を透明にしてスクショするだけなので、ほかの道具は要らない。
#   使い方: scripts/make_icon.sh   → src/tsuyaku/assets/icon_1024.png
set -euo pipefail
cd "$(dirname "$0")/.."
CHROME="${CHROME:-/Applications/Google Chrome.app/Contents/MacOS/Google Chrome}"
SRC="$PWD/src/tsuyaku/assets/icon.svg"
OUT="$PWD/src/tsuyaku/assets/icon_1024.png"
TMP="$(mktemp -d)"
# SVG を余白なしの HTML に包む(直接開くと既定の余白や白背景が入ることがある)
cat > "$TMP/icon.html" <<HTML
<!doctype html><html><head><style>html,body{margin:0;background:transparent}img{display:block;width:1024px;height:1024px}</style></head>
<body><img src="file://$SRC"></body></html>
HTML
"$CHROME" --headless=new --disable-gpu --hide-scrollbars --force-device-scale-factor=1 \
  --default-background-color=00000000 --window-size=1024,1024 \
  --screenshot="$OUT" "file://$TMP/icon.html" >/dev/null 2>&1
rm -rf "$TMP"
echo "$OUT"
