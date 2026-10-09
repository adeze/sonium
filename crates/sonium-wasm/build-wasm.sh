#!/bin/bash
# Build WASM module with parallel execution support (wasm-bindgen-rayon)
#
# Requirements:
# - Rust nightly toolchain
# - wasm-pack: cargo install wasm-pack
# - wasm32-unknown-unknown target: rustup target add wasm32-unknown-unknown
#
# The built WASM will support multi-threading via Web Workers + SharedArrayBuffer.
# To use parallel execution in the browser, serve with these headers:
#   Cross-Origin-Opener-Policy: same-origin
#   Cross-Origin-Embedder-Policy: require-corp

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
WORKSPACE_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
FRONTEND_DIR="$SCRIPT_DIR/frontend"
cd "$WORKSPACE_ROOT"

echo "Building WASM with parallel execution support..."
echo "Using nightly toolchain with atomics and bulk-memory features"

# Build with threading support
# -C target-feature=+atomics,+bulk-memory enables WebAssembly threads
# Shared memory must be imported with a declared maximum for Web Workers.
# -Z build-std rebuilds the standard library with thread support
RUSTFLAGS='-C target-feature=+atomics,+bulk-memory,+mutable-globals -C link-arg=--import-memory -C link-arg=--shared-memory -C link-arg=--max-memory=2147483648 -C link-arg=--export=__wasm_init_tls -C link-arg=--export=__tls_size -C link-arg=--export=__tls_align -C link-arg=--export=__tls_base' \
cargo +nightly build \
    -p sonium-roomsim \
    --features meshlang \
    --target wasm32-unknown-unknown \
    --release \
    -Z build-std=panic_abort,std

# Run wasm-bindgen to generate JS bindings
# The WASM output is in the workspace root's target directory
echo "Generating JS bindings..."
mkdir -p "$FRONTEND_DIR/pkg"
wasm-bindgen \
  --target web \
  --out-dir "$FRONTEND_DIR/pkg" \
  "$WORKSPACE_ROOT/target/wasm32-unknown-unknown/release/sonium_roomsim.wasm"

echo ""
echo "Build complete! Output in $FRONTEND_DIR/pkg/"
echo ""
echo "To use parallel execution, serve with Cross-Origin Isolation headers:"
echo "  Cross-Origin-Opener-Policy: same-origin"
echo "  Cross-Origin-Embedder-Policy: require-corp"
echo ""
echo "For local testing, you can use:"
echo "  npx serve $SCRIPT_DIR --cors"
