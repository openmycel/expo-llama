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

# LLAMA_SRC: a GitHub archive of the tag already on disk, checked the same way below.
if [[ -n "${LLAMA_SRC:-}" ]]; then
  cp "$LLAMA_SRC" "$WORK/src.tar.gz"
else
  echo "Downloading llama.cpp $TAG…"
  curl -fsSL -o "$WORK/src.tar.gz" "https://github.com/ggml-org/llama.cpp/archive/refs/tags/$TAG.tar.gz"
fi

# GitHub writes the commit into the archive header; refuse anything else.
# git reads only the header and closes the pipe: with pipefail gunzip's SIGPIPE would end
# the script without a word, so the archive comes through a process substitution.
GOT="$(git get-tar-commit-id < <(gunzip -c "$WORK/src.tar.gz"))"
if [[ "$GOT" != "$COMMIT" ]]; then
  echo "Commit mismatch: expected $COMMIT, got $GOT" >&2
  exit 1
fi

tar -xzf "$WORK/src.tar.gz" -C "$WORK"
cd "$WORK/llama.cpp-$TAG"
# The compiler writes the path of every source file into the binary (__FILE__ in ggml's
# asserts, debug info). Without this the build folder of the Mac it was built on travels with
# the package: 0.1.1–0.3.0 carried a temporary folder's full path. The paths start at
# llama.cpp-$TAG/ instead. build-xcframework.sh passes its own C and C++ flags to CMake;
# Objective-C (ggml-metal) takes OBJCFLAGS from the environment.
MAP="-ffile-prefix-map=$PWD=llama.cpp-$TAG"
sed -i '' -e "s#^COMMON_C_FLAGS=\"#COMMON_C_FLAGS=\"$MAP #" -e "s#^COMMON_CXX_FLAGS=\"#COMMON_CXX_FLAGS=\"$MAP #" build-xcframework.sh
grep -q -- "$MAP" build-xcframework.sh || { echo "Could not add $MAP to build-xcframework.sh" >&2; exit 1; }
export OBJCFLAGS="$MAP" OBJCXXFLAGS="$MAP"
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

# Nothing of the build Mac may stay in the binary.
for BIN in "$OUT"/*/llama.framework/llama; do
  if strings -a "$BIN" | grep -E "$WORK|/Users/|/private/|/var/folders/" >/dev/null; then
    echo "Build paths left in $BIN" >&2
    exit 1
  fi
done

echo "Done: ios/Frameworks/llama.xcframework ($TAG), $(du -sh "$OUT" | cut -f1) without debug symbols"
