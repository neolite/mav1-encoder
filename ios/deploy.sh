#!/bin/sh
# Rebuild libmav1 + mav1cam, install and launch on the connected iPhone.
# Usage: ios/deploy.sh [device-name]        (default: first connected device)
#        ios/deploy.sh --sim "iPhone 18 Pro" (simulator, unsigned)
set -e
cd "$(dirname "$0")/.."
TEAM=F4JHBRD585   # Rafkat Galiullin (Personal Team)

echo "== libmav1 =="
make ios -j8

cd ios/mav1cam
xcodegen generate >/dev/null

if [ "$1" = "--sim" ]; then
  SIM="${2:-iPhone 18 Pro}"
  echo "== build (simulator: $SIM) =="
  xcodebuild -project mav1cam.xcodeproj -scheme mav1cam \
    -destination "platform=iOS Simulator,name=$SIM" \
    -configuration Debug build CODE_SIGNING_ALLOWED=NO
  APP=$(find ~/Library/Developer/Xcode/DerivedData/mav1cam-*/Build/Products/Debug-iphonesimulator/mav1cam.app -maxdepth 0 | head -1)
  xcrun simctl boot "$SIM" 2>/dev/null || true
  xcrun simctl install booted "$APP"
  xcrun simctl launch booted com.mav1.mav1cam
  exit 0
fi

DEV="${1:-iPhoneRf}"
echo "== build (device: $DEV) =="
xcodebuild -project mav1cam.xcodeproj -scheme mav1cam \
  -destination "platform=iOS,name=$DEV" \
  -configuration Release build -allowProvisioningUpdates DEVELOPMENT_TEAM=$TEAM

APP=$(find ~/Library/Developer/Xcode/DerivedData/mav1cam-*/Build/Products/Release-iphoneos/mav1cam.app -maxdepth 0 | head -1)
echo "== install =="
xcrun devicectl device install app --device "$DEV" "$APP"
echo "== launch =="
xcrun devicectl device process launch --device "$DEV" com.mav1.mav1cam
echo "== done =="
