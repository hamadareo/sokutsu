#!/bin/bash
# 他人の Mac に配る Sokutsu.app と DMG を作る(Apple シリコン専用、macOS 13.3 以上)。
#   使い方: scripts/build_dist.sh            → dist/Sokutsu-<版>.dmg
#           SOKUTSU_VERSION=1.0.1 scripts/build_dist.sh
#           SKIP_TESTS=1 scripts/build_dist.sh   (書き換えた写しでの単体試験を飛ばす)
# 中身: 同梱の Python 3.12(python-build-standalone)+ 依存 + tsuyaku(.pyc だけ)、llama.cpp の llama-server、
# 起動役(C)、この Mac の音の部品。モデルは入れない(初回起動の画面で公開元から取る)。
# 自分のリモートサーバー・本人の名前・本人のデータは入れない(最後に grep で確かめ、あれば失敗)。
# 設計: 設計メモ(公開版には含めない)
set -euo pipefail
cd "$(dirname "$0")/.."
REPO="$PWD"
VER="${SOKUTSU_VERSION:-1.0.1}"
REV="$(git rev-parse --short HEAD 2>/dev/null || echo dev)"
BUILD_PY="${BUILD_PY:-python3}"           # wheel を落として広げるだけ(ビルドする Mac の Python 3.9+)
CACHE="$REPO/dist/cache"
BUILD="$REPO/dist/build"
APP="$BUILD/Sokutsu.app"
C="$APP/Contents"
RES="$C/Resources"
PYH="$RES/python"
SP="$PYH/lib/python3.12/site-packages"
OUT="$REPO/dist/Sokutsu-$VER.dmg"
MINOS=13.3   # llama.cpp 公式の macOS arm64 ビルドの最小版
# 配る版の bundle id は作者の名前を含まないもの。ユーザー本人の /Applications/Sokutsu.app
# (install_app.sh、app.sokutsu.dev)は許可がやり直しになるので変えない。
BUNDLE_ID=app.sokutsu.Sokutsu
USER_ID_PREFIX=app.sokutsu.dev

# 取得物(公式の配布元・版と SHA256 を固定)
PY_URL="https://github.com/astral-sh/python-build-standalone/releases/download/20261026/cpython-3.12.16%2B20261026-aarch64-apple-darwin-install_only_stripped.tar.gz"
PY_SHA=353d5f1b4fd2715e7053f12f068435d5bbad16b91e057006b5d01e0925709a7e
LLAMA_TAG=b12180
LLAMA_URL="https://github.com/ggml-org/llama.cpp/releases/download/$LLAMA_TAG/llama-$LLAMA_TAG-bin-macos-arm64.tar.gz"
LLAMA_SHA=0883467d96aa2ec28b3992e06796de66623b037f4b5722da52c39ca53941ed5c
PROXY_URL="https://files.pythonhosted.org/packages/f2/cf/77d3e19b7fabd03895caca7857ef51e4c409e0ca6b37ee6e9f7daa50b642/proxy_tools-0.1.0.tar.gz"
PROXY_SHA=ccb3751f529c047e2d8a58440d86b205303cf0fe8146f784d1cbcd94f0a28010
# sherpa-onnx の wheel に入っている ONNX Runtime(1.28.2)の通知とライセンスの全文(wheel には入っていない)
ORT_TAG=v1.28.2
ORT_NOTICES_URL="https://raw.githubusercontent.com/microsoft/onnxruntime/$ORT_TAG/ThirdPartyNotices.txt"
ORT_NOTICES_SHA=0e07b95f3a8d6230037707c5c4a2b554d12c4cb67369669ac255635528ffcee2
ORT_LICENSE_URL="https://raw.githubusercontent.com/microsoft/onnxruntime/$ORT_TAG/LICENSE"
ORT_LICENSE_SHA=2f07c72751aed99790b8a4869cf2311df85a860b22ded05fa22803587a48922c
VAD_URL="https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx"
VAD_SHA=9e2449e1087496d8d4caba907f23e0bd3f78d91fa552479bb9c23ac09cbb1fd6

say() { printf '\n== %s\n' "$*"; }
die() { echo "build_dist: $*" >&2; exit 1; }
sha_ok() { [[ -f "$1" && "$(shasum -a 256 "$1" | cut -d' ' -f1)" == "$2" ]]; }
fetch() { # fetch <url> <sha256> <保存先>
  sha_ok "$3" "$2" && return 0
  echo "取得: $1"
  curl -fL --retry 3 -o "$3.part" "$1"
  sha_ok "$3.part" "$2" || { rm -f "$3.part"; die "SHA256 が合いません: $1"; }
  mv "$3.part" "$3"
}

[[ "$(uname -m)" == arm64 ]] || die "Apple シリコンの Mac でビルドしてください"
command -v clang >/dev/null || die "clang がありません(xcode-select --install)"

say "1/9 取得(キャッシュ: dist/cache)"
mkdir -p "$CACHE/wheels"
fetch "$PY_URL" "$PY_SHA" "$CACHE/python.tar.gz"
fetch "$LLAMA_URL" "$LLAMA_SHA" "$CACHE/llama-$LLAMA_TAG.tar.gz"
fetch "$PROXY_URL" "$PROXY_SHA" "$CACHE/proxy_tools-0.1.0.tar.gz"
fetch "$VAD_URL" "$VAD_SHA" "$CACHE/silero_vad.onnx"
fetch "$ORT_NOTICES_URL" "$ORT_NOTICES_SHA" "$CACHE/onnxruntime-$ORT_TAG-ThirdPartyNotices.txt"
fetch "$ORT_LICENSE_URL" "$ORT_LICENSE_SHA" "$CACHE/onnxruntime-$ORT_TAG-LICENSE.txt"
PIPX=(--disable-pip-version-check -q --no-deps --only-binary=:all: --implementation cp
      --python-version 3.12 --platform macosx_11_0_arm64 --platform macosx_12_0_arm64
      --platform macosx_13_0_arm64 --platform macosx_10_13_universal2 --platform macosx_10_6_universal2
      --platform macosx_10_6_x86_64.macosx_10_6_universal2)
"$BUILD_PY" -m pip download "${PIPX[@]}" -d "$CACHE/wheels" -r packaging/requirements-dist.txt

say "2/9 骨組みと Python"
rm -rf "$BUILD"
mkdir -p "$C/MacOS" "$RES" "$BUILD/x"
tar -xzf "$CACHE/python.tar.gz" -C "$RES"          # → Resources/python
# 使わないもの(Tk・IDLE・pip・ヘッダ・静的ライブラリ)を除く
rm -rf "$PYH/include" "$PYH/share" "$PYH/lib/pkgconfig" "$PYH"/lib/tcl* "$PYH"/lib/tk* \
       "$PYH"/lib/libtcl* "$PYH"/lib/itcl* "$PYH"/lib/thread* \
       "$PYH/lib/python3.12/idlelib" "$PYH/lib/python3.12/tkinter" "$PYH/lib/python3.12/turtledemo" \
       "$PYH/lib/python3.12/ensurepip" "$PYH/lib/python3.12/test" "$PYH/lib/python3.12/config-3.12-darwin" \
       "$SP"/pip "$SP"/pip-*.dist-info
find "$PYH/lib/python3.12/lib-dynload" -name "_tkinter*" -delete
find "$PYH/bin" -mindepth 1 ! -name python3.12 -exec rm -f {} +
PYBIN="$PYH/bin/python3.12"     # ビルド中の試験と .pyc 作りに使う(最後に除く)

say "3/9 依存(wheel を広げるだけ。setup.py は実行しない)"
"$BUILD_PY" -m pip install "${PIPX[@]}" --no-compile --no-index --find-links "$CACHE/wheels" \
  --target "$SP" -r packaging/requirements-dist.txt
rm -rf "$SP/bin"
find "$SP" -name "*.dll" -delete   # sounddevice の Windows 用(ASIO 版を含む)。macOS では使わない
# proxy_tools(pywebview の依存、wheel が無い): sdist からモジュールのファイルだけ写す
mkdir -p "$BUILD/x/proxy" && tar -xzf "$CACHE/proxy_tools-0.1.0.tar.gz" -C "$BUILD/x/proxy"
cp -R "$BUILD/x/proxy/proxy_tools-0.1.0/proxy_tools" "$SP/"
mkdir -p "$SP/proxy_tools-0.1.0.dist-info"
printf 'Metadata-Version: 2.1\nName: proxy_tools\nVersion: 0.1.0\nLicense: BSD\n' > "$SP/proxy_tools-0.1.0.dist-info/METADATA"
[[ -f "$BUILD/x/proxy/proxy_tools-0.1.0/LICENSE" ]] && cp "$BUILD/x/proxy/proxy_tools-0.1.0/LICENSE" "$SP/proxy_tools-0.1.0.dist-info/"
"$PYBIN" -I -B -c "import webview, sherpa_onnx, pywhispercpp.model, sounddevice, numpy, requests, objc, EventKit, Quartz; print('import OK')"

say "4/9 Sokutsu 本体(書き換えた写し)"
rsync -a --exclude __pycache__ --exclude '*.pyc' "$REPO/src/tsuyaku/" "$SP/tsuyaku/"
"$BUILD_PY" packaging/sanitize.py rewrite "$SP/tsuyaku"
# 配る版の画面の差分(処理場所の札)を index.html の最後に足す
grep -q '<script src="dist.js"></script>' "$SP/tsuyaku/webui/index.html" || \
  sed -i '' 's#</body>#<script src="dist.js"></script>\n</body>#' "$SP/tsuyaku/webui/index.html"
grep -q 'src="dist.js"' "$SP/tsuyaku/webui/index.html" || die "index.html に dist.js を足せませんでした"
printf '{"version": "%s", "rev": "%s", "built": "%s"}\n' "$VER" "$REV" "$(date +%Y-%m-%d)" > "$SP/tsuyaku/_dist.json"

if [[ -z "${SKIP_TESTS:-}" ]]; then
  say "5/9 書き換えた写しで単体試験(同梱の Python)"
  T="$BUILD/testrepo"
  mkdir -p "$T"
  rsync -a --exclude .git --exclude dist --exclude models --exclude logs --exclude bin --exclude .venv \
        --exclude config.json --exclude terms.json --exclude __pycache__ --exclude '*.pyc' "$REPO/" "$T/"
  "$BUILD_PY" packaging/sanitize.py rewrite "$T/src" >/dev/null
  mkdir -p "$T/models" && cp "$CACHE/silero_vad.onnx" "$T/models/"
  printf '{"tts_voice": {"ja": "voicevox:13"}}\n' > "$T/config.json"   # test_voicevox が読む(試験用の写し)
  (cd "$T" && PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$T/src" "$PYBIN" -m unittest discover -s tests > "$BUILD/tests.txt" 2>&1) || true
  tail -3 "$BUILD/tests.txt"
  if ! grep -q '^OK' "$BUILD/tests.txt"; then
    # 時間に左右される試験(索引の書き込みと後片付けの競り合いなど)は、落ちたものだけもう一度
    failed="$(sed -nE 's/^(ERROR|FAIL): [^ ]+ \(([^)]+)\)$/\2/p' "$BUILD/tests.txt" | sort -u | tr '\n' ' ')"
    [[ -n "$failed" ]] || die "書き換えた写しの単体試験が通りません(dist/build/tests.txt)"
    echo "もう一度: $failed"
    (cd "$T/tests" && PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$T/src:$T/tests" "$PYBIN" -m unittest $failed > "$BUILD/tests-retry.txt" 2>&1) || true
    tail -3 "$BUILD/tests-retry.txt"
    grep -q '^OK' "$BUILD/tests-retry.txt" || die "書き換えた写しの単体試験が通りません(dist/build/tests-retry.txt)"
  fi
  rm -rf "$T"
fi

say "6/9 .pyc(実行中は .app に書かない)・arm64 だけに薄くする"
# tsuyaku はソースを入れず .pyc だけ(コメントの実会話の例を配らない)
"$PYBIN" -I -B -m compileall -f -q -b -j0 --invalidation-mode unchecked-hash -s "$SP" -p "" "$SP/tsuyaku" >/dev/null
find "$SP/tsuyaku" -name "*.py" -delete
find "$SP/tsuyaku" -name __pycache__ -type d -exec rm -rf {} +
"$PYBIN" -I -B -m compileall -f -q -j0 --invalidation-mode unchecked-hash -s "$PYH" -p "" \
  -x '/tsuyaku/' "$PYH/lib/python3.12" >/dev/null || true
while IFS= read -r f; do
  if lipo -archs "$f" 2>/dev/null | grep -q x86_64; then lipo -thin arm64 "$f" -output "$f.thin" && mv "$f.thin" "$f"; fi
done < <(find "$PYH" -type f \( -name "*.so" -o -name "*.dylib" \))

say "7/9 起動役・この Mac の音の部品・llama-server"
clang -O2 -Wall -arch arm64 -mmacosx-version-min=$MINOS native/launcher/launcher.c \
  -L"$PYH/lib" -lpython3.12 -Wl,-rpath,@executable_path/../Resources/python/lib -o "$C/MacOS/Sokutsu"
# この Mac の音の部品: 識別子(Info.plist・集約デバイスの UID)を配る版の id にした写しから作る
# (リポジトリの sysaudio.c を変えると本人の部品が作り直され、許可がやり直しになる)
mkdir -p "$BUILD/x/sysaudio"
sed "s/$USER_ID_PREFIX/$BUNDLE_ID/g" native/sysaudio/sysaudio.c > "$BUILD/x/sysaudio/sysaudio.c"
sed "s/$USER_ID_PREFIX/$BUNDLE_ID/g" native/sysaudio/Info.plist > "$BUILD/x/sysaudio/Info.plist"
grep -q "$BUNDLE_ID" "$BUILD/x/sysaudio/Info.plist" || die "sysaudio の Info.plist の id を替えられませんでした"
clang -O2 -Wall -Wextra -fblocks -arch arm64 -mmacosx-version-min=$MINOS \
  -framework CoreAudio -framework CoreFoundation -lobjc \
  -Wl,-sectcreate,__TEXT,__info_plist,"$BUILD/x/sysaudio/Info.plist" \
  -o "$C/MacOS/sokutsu-sysaudio" "$BUILD/x/sysaudio/sysaudio.c"
rm -rf "$PYH/bin"
mkdir -p "$BUILD/x/llama" "$RES/llama"
tar -xzf "$CACHE/llama-$LLAMA_TAG.tar.gz" -C "$BUILD/x/llama"
L="$BUILD/x/llama/llama-$LLAMA_TAG"
cp "$L/llama-server" "$L/LICENSE" "$RES/llama/"
# llama-server が要る dylib を辿って写す(@rpath = 同じフォルダ)
need=("$RES/llama/llama-server")
while ((${#need[@]})); do
  f="${need[0]}"; need=("${need[@]:1}")
  for d in $(otool -L "$f" | awk '/@rpath\//{print $1}' | sed 's#@rpath/##'); do
    if [[ ! -e "$RES/llama/$d" ]]; then
      cp -L "$L/$d" "$RES/llama/$d" || die "llama の部品がありません: $d"
      need+=("$RES/llama/$d")
    fi
  done
done
[[ "$(otool -l "$RES/llama/llama-server" | grep -A2 LC_RPATH | awk '/path/{print $2}')" == "@loader_path" ]] \
  || die "llama-server の rpath が想定と違います"

say "8/9 Info.plist・アイコン・ライセンス・署名"
SET="$BUILD/x/Sokutsu.iconset"; mkdir -p "$SET"
PNG="$REPO/src/tsuyaku/assets/icon_1024.png"
for s in 16 32 128 256 512; do
  sips -z $s $s "$PNG" --out "$SET/icon_${s}x${s}.png" >/dev/null
  sips -z $((s * 2)) $((s * 2)) "$PNG" --out "$SET/icon_${s}x${s}@2x.png" >/dev/null
done
iconutil -c icns "$SET" -o "$RES/Sokutsu.icns"
# 許可の文言は scripts/install_app.sh と同じ
cat > "$C/Info.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>CFBundleName</key><string>Sokutsu</string>
  <key>CFBundleDisplayName</key><string>Sokutsu</string>
  <key>CFBundleIdentifier</key><string>$BUNDLE_ID</string>
  <key>CFBundleExecutable</key><string>Sokutsu</string>
  <key>CFBundleIconFile</key><string>Sokutsu</string>
  <key>CFBundlePackageType</key><string>APPL</string>
  <key>CFBundleShortVersionString</key><string>$VER</string>
  <key>CFBundleVersion</key><string>$REV</string>
  <key>LSMinimumSystemVersion</key><string>$MINOS</string>
  <key>NSHighResolutionCapable</key><true/>
  <key>LSArchitecturePriority</key><array><string>arm64</string></array>
  <key>LSRequiresNativeExecution</key><true/>
  <key>NSMicrophoneUsageDescription</key><string>話した声を聞き取って翻訳するためにマイクを使います。</string>
  <key>NSAudioCaptureUsageDescription</key><string>オンライン会議の相手の声(この Mac で鳴っている音)を聞き取って翻訳・会議メモにするために使います。</string>
  <key>NSAppleEventsUsageDescription</key><string>「カレンダーの予定から」「リマインダーへ」を押した時に、カレンダーとリマインダーのアプリに予定を読む・やることを入れるよう頼みます。</string>
  <key>NSCalendarsUsageDescription</key><string>「カレンダーの予定から」を押した時に、今の予定の題と出席者を会議の題・名前の候補にします。</string>
  <key>NSCalendarsFullAccessUsageDescription</key><string>「カレンダーの予定から」を押した時に、今の予定の題と出席者を会議の題・名前の候補にします。</string>
  <key>NSRemindersUsageDescription</key><string>「リマインダーへ」を押した時に、会議メモのやることをリマインダーに入れます。</string>
  <key>NSRemindersFullAccessUsageDescription</key><string>「リマインダーへ」を押した時に、会議メモのやることをリマインダーに入れます。</string>
</dict></plist>
PLIST
plutil -lint "$C/Info.plist" >/dev/null
# 許可の文言が install_app.sh と同じか
for k in NSMicrophoneUsageDescription NSAudioCaptureUsageDescription NSCalendarsUsageDescription NSRemindersUsageDescription; do
  a="$(grep "<key>$k</key>" scripts/install_app.sh | sed 's/^ *//')"; b="$(grep "<key>$k</key>" "$C/Info.plist" | sed 's/^ *//')"
  [[ -n "$a" && "$a" == "$b" ]] || die "許可の文言が install_app.sh と違います: $k"
done
strings "$SP/sherpa_onnx/lib/libonnxruntime.dylib" | grep -x "${ORT_TAG#v}" >/dev/null \
  || die "同梱の ONNX Runtime の版が ${ORT_TAG#v} ではありません(ORT_TAG と通知を更新してください)"
N="$RES/THIRD_PARTY_NOTICES.txt"
{
  cat packaging/notices_header.txt
  printf '\n\n==== llama.cpp (%s) LICENSE ====\n\n' "$LLAMA_TAG"; cat "$RES/llama/LICENSE"
  printf '\n\n==== CPython 3.12 LICENSE ====\n\n'; cat "$PYH/lib/python3.12/LICENSE.txt"
  printf '\n\n==== ONNX Runtime %s (sherpa-onnx 内の libonnxruntime.dylib) LICENSE ====\n\n' "$ORT_TAG"
  cat "$CACHE/onnxruntime-$ORT_TAG-LICENSE.txt"
  printf '\n\n==== ONNX Runtime %s ThirdPartyNotices.txt ====\n\n' "$ORT_TAG"
  cat "$CACHE/onnxruntime-$ORT_TAG-ThirdPartyNotices.txt"
  for d in "$SP"/*.dist-info; do
    for f in $(find "$d" -type f \( -iname 'LICENSE*' -o -iname 'COPYING*' -o -iname 'NOTICE*' -o -iname 'AUTHORS*' \) | sort); do
      printf '\n\n==== %s: %s ====\n\n' "$(basename "$d" .dist-info)" "${f#$d/}"; cat "$f"
    done
  done
} > "$N"
# 署名: 中の Mach-O を全部(内側から)→ .app。ad-hoc、Hardened Runtime なし
n=0
while IFS= read -r f; do
  if file -b "$f" | grep -q Mach-O; then codesign -s - -f --timestamp=none "$f" 2>/dev/null; n=$((n + 1)); fi
done < <(find "$C" -type f \( -name "*.so" -o -name "*.dylib" -o -perm -u+x \) ! -path "$C/MacOS/Sokutsu")
codesign -s - -f --timestamp=none --identifier "$BUNDLE_ID.sysaudio" "$C/MacOS/sokutsu-sysaudio"
codesign -s - -f --timestamp=none --identifier "$BUNDLE_ID" "$APP"
codesign --verify --deep --strict "$APP" || die "署名の確かめに失敗しました"
echo "署名: Mach-O $n 個 + .app"

say "9/9 確かめ(リモートサーバー・個人の情報・本人のデータ)と DMG"
"$BUILD_PY" packaging/sanitize.py check "$APP"
# 本人のデータ(設定・用語・記録・録音・声の帳面・索引)とモデルが入っていないこと
leak="$(find "$APP" \( -name config.json -o -name terms.json -o -name "session-*" -o -name "*.jsonl" \
        -o -name "*.caf" -o -name "*.wav" -o -name "book.json" -o -name "book.npz" -o -name "*.voices.npz" \
        -o -name "*.voicematch.json" -o -name "*.sqlite3" -o -name "*.gguf" -o -name "ggml-*.bin" -o -name "*.onnx" \
        -o -type d \( -name logs -o -name models -o -name voices \) -o -name "remote*" \) -print)"
[[ -z "$leak" ]] || die "配ってはいけないファイルがあります: $leak"
D="$BUILD/dmg"; mkdir -p "$D"
cp -R "$APP" "$D/"
ln -s /Applications "$D/Applications"
cp packaging/dmg/*.txt "$D/"
rm -f "$OUT"
hdiutil create -quiet -volname "Sokutsu $VER" -srcfolder "$D" -fs HFS+ -format ULFO -ov "$OUT"
MNT="$(mktemp -d)"
hdiutil attach -quiet -readonly -nobrowse -mountpoint "$MNT" "$OUT"
trap 'hdiutil detach -quiet "$MNT" 2>/dev/null || true' EXIT
"$BUILD_PY" packaging/sanitize.py check "$MNT"
codesign --verify --deep --strict "$MNT/Sokutsu.app"
hdiutil detach -quiet "$MNT"; trap - EXIT
echo
echo "できました: $OUT ($(du -h "$OUT" | cut -f1))  .app $(du -sh "$APP" | cut -f1)  版 $VER ($REV)"
