#!/usr/bin/env bash
# Builds ios/Frameworks/llama.xcframework (iPhone + simulator) from a pinned llama.cpp tag.
# The upstream release zip ships iPhone and macOS only, no simulator, so we build it ourselves
# with llama.cpp's own build-xcframework.sh. Needs Xcode and CMake.
set -euo pipefail

TAG=b11146
COMMIT=7fe450e19305b828c199d602c23a8337aaa1f03b

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

echo "Downloading llama.cpp $TAG…"
curl -fsSL -o "$WORK/src.tar.gz" "https://github.com/ggml-org/llama.cpp/archive/refs/tags/$TAG.tar.gz"

# GitHub writes the commit into the archive header; refuse anything else.
GOT="$(gunzip -c "$WORK/src.tar.gz" | git get-tar-commit-id)"
if [[ "$GOT" != "$COMMIT" ]]; then
  echo "Commit mismatch: expected $COMMIT, got $GOT" >&2
  exit 1
fi

tar -xzf "$WORK/src.tar.gz" -C "$WORK"
cd "$WORK/llama.cpp-$TAG"
./build-xcframework.sh ios-sim ios-device

OUT="$ROOT/ios/Frameworks/llama.xcframework"
rm -rf "$OUT"
mkdir -p "$ROOT/ios/Frameworks" "$ROOT/dist"
cp -R build-apple/llama.xcframework "$OUT"
echo "$TAG $COMMIT" > "$ROOT/ios/Frameworks/LLAMA_VERSION"
# MIT asks for the notice in every copy: the license travels with the binary.
cp LICENSE "$ROOT/ios/Frameworks/LICENSE-llama.cpp"

# Debug symbols are ~170 of ~195 MB. They stay out of the npm package and go to
# dist/ as a separate zip for crash symbolication (attached to the GitHub release).
rm -f "$ROOT/dist/llama-$TAG-dSYMs.zip"
(cd "$OUT" && zip -qry "$ROOT/dist/llama-$TAG-dSYMs.zip" ./*/dSYMs)
rm -rf "$OUT"/*/dSYMs
LIBRARIES=$(plutil -extract AvailableLibraries raw "$OUT/Info.plist")
for ((i = 0; i < LIBRARIES; i++)); do
  plutil -remove "AvailableLibraries.$i.DebugSymbolsPath" "$OUT/Info.plist"
done

echo "Done: ios/Frameworks/llama.xcframework ($TAG), $(du -sh "$OUT" | cut -f1) without debug symbols"
