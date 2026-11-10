#!/bin/bash
# Sokutsu を Mac のアプリとして入れる(/Applications/Sokutsu.app)。
# Launchpad・Dock・Spotlight から起動できる。中身はこのリポジトリの run.sh を呼ぶだけなので、
# git pull で更新すれば入れ直さなくても次の起動から新しい版になる。
#   使い方: scripts/install_app.sh            → /Applications/Sokutsu.app
#           APP_DIR=~/Applications scripts/install_app.sh
# ログ: ~/Library/Logs/Sokutsu/run.log
set -euo pipefail
cd "$(dirname "$0")/.."
REPO="${SOKUTSU_REPO:-$PWD}"   # 起動役が呼ぶ run.sh のあるリポジトリ(既定はこのスクリプトのある所)
APP_DIR="${APP_DIR:-/Applications}"
APP="$APP_DIR/Sokutsu.app"
PNG="$REPO/src/tsuyaku/assets/icon_1024.png"
VER="$(git -C "$REPO" rev-parse --short HEAD 2>/dev/null || echo dev)"

# この Mac の音(オンライン会議)を聞く部品。ここで作っておく(run.sh も無ければ作る)
bash "$REPO/scripts/build_sysaudio.sh" || echo "bin/sokutsu-sysaudio を作れませんでした(マイクだけで動きます)"

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
B="$TMP/Sokutsu.app/Contents"
mkdir -p "$B/MacOS" "$B/Resources"

# アイコン(.icns): 1024 角の PNG から各サイズを作る
SET="$TMP/Sokutsu.iconset"
mkdir -p "$SET"
for s in 16 32 128 256 512; do
  sips -z $s $s "$PNG" --out "$SET/icon_${s}x${s}.png" >/dev/null
  sips -z $((s * 2)) $((s * 2)) "$PNG" --out "$SET/icon_${s}x${s}@2x.png" >/dev/null
done
iconutil -c icns "$SET" -o "$B/Resources/Sokutsu.icns"

cat > "$B/Info.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>CFBundleName</key><string>Sokutsu</string>
  <key>CFBundleDisplayName</key><string>Sokutsu</string>
  <key>CFBundleIdentifier</key><string>app.sokutsu.dev</string>
  <key>CFBundleExecutable</key><string>Sokutsu</string>
  <key>CFBundleIconFile</key><string>Sokutsu</string>
  <key>CFBundlePackageType</key><string>APPL</string>
  <key>CFBundleShortVersionString</key><string>1.0</string>
  <key>CFBundleVersion</key><string>$VER</string>
  <key>LSMinimumSystemVersion</key><string>12.0</string>
  <key>NSHighResolutionCapable</key><true/>
  <key>LSArchitecturePriority</key><array><string>arm64</string></array>
  <key>LSRequiresNativeExecution</key><true/>
  <key>LSUIElement</key><true/>
  <key>NSMicrophoneUsageDescription</key><string>話した声を聞き取って翻訳するためにマイクを使います。</string>
  <key>NSAudioCaptureUsageDescription</key><string>オンライン会議の相手の声(この Mac で鳴っている音)を聞き取って翻訳・会議メモにするために使います。</string>
  <key>NSAppleEventsUsageDescription</key><string>「カレンダーの予定から」「リマインダーへ」を押した時に、カレンダーとリマインダーのアプリに予定を読む・やることを入れるよう頼みます。</string>
  <key>NSCalendarsUsageDescription</key><string>「カレンダーの予定から」を押した時に、今の予定の題と出席者を会議の題・名前の候補にします。</string>
  <key>NSCalendarsFullAccessUsageDescription</key><string>「カレンダーの予定から」を押した時に、今の予定の題と出席者を会議の題・名前の候補にします。</string>
  <key>NSRemindersUsageDescription</key><string>「リマインダーへ」を押した時に、会議メモのやることをリマインダーに入れます。</string>
  <key>NSRemindersFullAccessUsageDescription</key><string>「リマインダーへ」を押した時に、会議メモのやることをリマインダーに入れます。</string>
</dict></plist>
PLIST

# 起動: Finder から開くと PATH が最小なので足してから run.sh へ(exec で同じプロセスのまま)。
# 起動役(Contents/MacOS/Sokutsu)は C の小さな実行ファイル(native/devlauncher)。run.sh を子として
# 起こして待つ。シェルスクリプトから Python へ exec すると実体が Python.app になり、マイクの許可を
# 聞く窓が出ずに無音になっていた(11/08)。arch -arm64 で Apple シリコンのまま走らせる(11/06)。
# 起動役そのものは画面を持たないので Dock には出さない(LSUIElement)。Dock には Python の窓が出る。
clang -O2 -Wall -arch arm64 -mmacosx-version-min=12.0 -DRUN_SH="\"$REPO/run.sh\"" \
  "$PWD/native/devlauncher/devlauncher.c" -o "$B/MacOS/Sokutsu"
chmod +x "$B/MacOS/Sokutsu"

# 署名(アドホック)。署名の無いアプリだと macOS がマイクの許可を聞く窓を出さず、一覧にも載らないまま
# 無音を渡していた(11/08「音声が入っていない」、システム設定のマイクに Sokutsu が無かった)。
# 署名があれば初回にマイクの許可を聞き、システム設定の一覧にも Sokutsu として載る。
codesign --force --sign - "$TMP/Sokutsu.app" >/dev/null 2>&1 \
  || echo "署名できませんでした(codesign)。マイクの許可が出ない時は、ターミナルから ./run.sh で起動してください" >&2

if [ -d "$APP" ]; then rm -rf "$APP"; fi
mv "$TMP/Sokutsu.app" "$APP"
touch "$APP"
/System/Library/Frameworks/CoreServices.framework/Frameworks/LaunchServices.framework/Support/lsregister -f "$APP" >/dev/null 2>&1 || true
echo "入れました: $APP ($VER)"
