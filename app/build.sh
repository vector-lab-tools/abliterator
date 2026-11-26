#!/bin/bash
# Build dist/Abliterator.app and dist/Abliterator-<version>.dmg (Apple silicon, macOS 14 or later).
# The app holds the shell, uv, the setup page and a copy of the code (the files git tracks or would track in
# backend/, web/, app/ and data/, plus Tune's clips if this tree has them). It is signed ad hoc for now.
#   app/build.sh [--no-dmg]
set -e
cd "$(dirname "$0")/.."
V=$(cat VERSION)
APP=dist/Abliterator.app
C=$APP/Contents
rm -rf "$APP"
mkdir -p "$C/MacOS" "$C/Resources/abliterator"

swiftc -O -target arm64-apple-macos14 -framework Cocoa -framework WebKit app/macos/main.swift -o "$C/MacOS/Abliterator"
sed "s/@VERSION@/$V/g" app/macos/Info.plist > "$C/Info.plist"
cp app/macos/setup.html "$C/Resources/"
sed "s/@VERSION@/$V/g" app/macos/Credits.html > "$C/Resources/Credits.html"
cp "$(command -v uv)" "$C/Resources/uv"

git ls-files -co --exclude-standard -- backend web app data VERSION LICENSE README.md CITATION.cff \
  | grep -v -e '/__pycache__/' -e '\.pyc$' -e '^app/macos/' \
  | rsync -a --files-from=- . "$C/Resources/abliterator/"
[ -d data/calibration/clips ] && rsync -a data/calibration/clips "$C/Resources/abliterator/data/calibration/"

# The icon, from the page's mark.
T=$(mktemp -d); trap 'rm -rf "$T"' EXIT
qlmanage -t -s 1024 -o "$T" web/mark.svg >/dev/null 2>&1
if [ -f "$T/mark.svg.png" ]; then
  mkdir "$T/a.iconset"
  for s in 16 32 128 256 512; do
    sips -z $s $s "$T/mark.svg.png" --out "$T/a.iconset/icon_${s}x${s}.png" >/dev/null
    sips -z $((s*2)) $((s*2)) "$T/mark.svg.png" --out "$T/a.iconset/icon_${s}x${s}@2x.png" >/dev/null
  done
  iconutil -c icns "$T/a.iconset" -o "$C/Resources/Abliterator.icns"
fi

codesign --force --deep --sign - "$APP"
echo "Built $APP ($(du -sh "$APP" | cut -f1))"
if [ "$1" != "--no-dmg" ]; then
  rm -f "dist/Abliterator-$V.dmg"
  mkdir "$T/dmg" && cp -R "$APP" "$T/dmg/" && ln -s /Applications "$T/dmg/Applications"
  hdiutil create -quiet -volname "Abliterator $V" -srcfolder "$T/dmg" -ov -format UDZO "dist/Abliterator-$V.dmg"
  echo "Built dist/Abliterator-$V.dmg ($(du -sh "dist/Abliterator-$V.dmg" | cut -f1))"
fi
