#!/usr/bin/env bash
#
# Cross-build llm-visuals for macOS, Windows and Linux on x86-64 and ARM64,
# from any one of those hosts.
#
#   ./scripts/cross-build.sh              # every target this host can link
#   ./scripts/cross-build.sh linux-arm64  # just one (see TARGETS below)
#   PROFILE=debug ./scripts/cross-build.sh
#
# Finished archives land in dist/.
#
# Needs: rustup, and cargo-zigbuild + zig for anything but the host target
# (see "Cross-compiling" in README.md for the one-time setup). macOS targets
# additionally need Apple's SDK, which cannot be redistributed: build those on
# a Mac, or point SDKROOT at a MacOSX.sdk you already have.

set -euo pipefail

cd "$(dirname "$0")/.."

# Friendly name -> rustc target triple.
#
# Windows uses the *-gnullvm triples rather than *-msvc because they are the
# ones zig can link. Both produce ordinary .exe files with no runtime
# dependency on a toolchain; CI builds the msvc ones natively instead.
declare -A TARGETS=(
  [linux-x86_64]=x86_64-unknown-linux-gnu
  [linux-arm64]=aarch64-unknown-linux-gnu
  [macos-x86_64]=x86_64-apple-darwin
  [macos-arm64]=aarch64-apple-darwin
  [windows-x86_64]=x86_64-pc-windows-gnullvm
  [windows-arm64]=aarch64-pc-windows-gnullvm
)
ORDER=(linux-x86_64 linux-arm64 macos-x86_64 macos-arm64 windows-x86_64 windows-arm64)

PROFILE="${PROFILE:-release}"
# Link Linux binaries against an old glibc so they run on distributions older
# than this build host. Set GLIBC= (empty) to link against the host's instead.
GLIBC="${GLIBC-2.28}"
HOST="$(rustc -vV | sed -n 's/^host: //p')"
DIST="$(pwd)/dist"
VERSION="$(sed -n 's/^version = "\(.*\)"/\1/p' Cargo.toml | head -1)"

want=("${ORDER[@]}")
if [ $# -gt 0 ]; then
  want=("$@")
fi

# cargo-zigbuild drives zig as the C compiler and linker; plain cargo build is
# enough when we are not actually crossing.
zigbuild_ok=false
if command -v cargo-zigbuild >/dev/null 2>&1 && (command -v zig >/dev/null 2>&1 || python3 -m ziglang version >/dev/null 2>&1); then
  zigbuild_ok=true
fi

mkdir -p "$DIST"
built=() skipped=()

for name in "${want[@]}"; do
  triple="${TARGETS[$name]:-}"
  if [ -z "$triple" ]; then
    echo "unknown target '$name'; known: ${ORDER[*]}" >&2
    exit 2
  fi

  # macOS links against frameworks that only ship in Apple's SDK.
  if [[ "$triple" == *apple-darwin ]] && [[ "$HOST" != *apple-darwin ]] && [ -z "${SDKROOT:-}" ]; then
    skipped+=("$name (needs a Mac, or SDKROOT=/path/to/MacOSX.sdk)")
    continue
  fi

  # zig is what makes an old glibc reachable, so a native Linux build goes
  # through it too: linking against the host's glibc would produce a binary
  # that only runs on distributions as new as this one.
  build_triple="$triple"
  pin_glibc=false
  if [[ "$triple" == *-linux-gnu ]] && [ -n "$GLIBC" ]; then
    pin_glibc=true
  fi

  if [ "$zigbuild_ok" = true ]; then
    cmd=(cargo zigbuild)
    [ "$pin_glibc" = true ] && build_triple="$triple.$GLIBC"
  elif [ "$triple" = "$HOST" ]; then
    cmd=(cargo build)
    [ "$pin_glibc" = true ] &&
      echo "    note: no zig, so linking against this host's glibc; the binary" \
           "will not run on older distributions" >&2
  else
    skipped+=("$name (install cargo-zigbuild + zig to cross-compile)")
    continue
  fi

  echo "==> $name ($build_triple)"
  rustup target add "$triple" >/dev/null 2>&1 || true
  "${cmd[@]}" --profile "$PROFILE" --target "$build_triple"

  # --profile dev still writes to target/<triple>/debug.
  out_dir="target/$triple/$PROFILE"
  [ "$PROFILE" = dev ] && out_dir="target/$triple/debug"

  stem="llm-visuals-$VERSION-$name"
  if [[ "$triple" == *windows* ]]; then
    (cd "$out_dir" && zip -q "$DIST/$stem.zip" llm-visuals.exe)
    built+=("dist/$stem.zip")
  else
    tar -C "$out_dir" -czf "$DIST/$stem.tar.gz" llm-visuals
    built+=("dist/$stem.tar.gz")
  fi
done

echo
for b in "${built[@]}"; do echo "built    $b"; done
for s in "${skipped[@]}"; do echo "skipped  $s"; done
[ ${#built[@]} -gt 0 ]
