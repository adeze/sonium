#!/usr/bin/env bash
# Native SM120 SR2 breadth packet: resident multi-source/frequency room FEM,
# source-amplitude derivative oracle, and launch-failure recovery.
# Keep all Cargo, temporary, runtime, and evidence paths repository-local.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
target_dir="$repo_root/.target-sr2-breadth-fresh"
artifact_dir="$target_dir/artifact"
tmp_dir="$target_dir/tmp"
runtime_dir="$target_dir/runtime"
artifact="$artifact_dir/cuda-breadth.json"
check_output="$artifact_dir/breadth-check.json"

test "$(uname -s)" = Linux || { echo "SR2 breadth requires Linux" >&2; exit 2; }
test "$(uname -m)" = x86_64 || { echo "SR2 breadth requires x86_64" >&2; exit 2; }
mkdir -p "$artifact_dir" "$tmp_dir" "$runtime_dir"
chmod 700 "$runtime_dir"
rm -f "$artifact" "$artifact_dir/breadth.sha256" "$artifact_dir/checksums.sha256" "$check_output"
export CARGO_TARGET_DIR="$target_dir/cargo"
export TMPDIR="$tmp_dir"
export TMP="$tmp_dir"
export TEMP="$tmp_dir"
export XDG_RUNTIME_DIR="$runtime_dir"

SONIUM_SIM_ROOM_REQUIRE_CUDA=1 \
    SONIUM_SIM_ROOM_SR2_BREADTH_ARTIFACT="$artifact" \
    cargo test --release -p sonium-sim-room --features cuda \
    --test backend_parity_sr2_breadth cuda_sr2_breadth_resident_derivative_and_recovery -- \
    --nocapture --test-threads=1

python3 "$repo_root/crates/sonium-sim-room/benchmarks/check_sr2_breadth_cuda.py" "$artifact" > "$check_output"

(
    cd "$artifact_dir"
    sha256sum "$(basename "$artifact")" > breadth.sha256
    sha256sum "$(basename "$artifact")" breadth.sha256 "$(basename "$check_output")" > checksums.sha256
    sha256sum -c checksums.sha256
)
python3 "$repo_root/crates/sonium-sim-room/benchmarks/check_sr2_breadth_cuda.py" "$artifact" > "$check_output"
(cd "$artifact_dir" && sha256sum -c checksums.sha256)
