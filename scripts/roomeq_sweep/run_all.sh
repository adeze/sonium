#!/usr/bin/env bash
# Orchestrator: build binaries, probe the machine, run CPU and/or GPU sweeps.
#
# Usage:
#   run_all.sh [--backend cpu|gpu|both] [--out-root DIR] [passthrough...]
#
# Pipeline:
#   1. cargo build (release) the CPU binary; with --features cuda into
#      target/cuda/ for the GPU binary (separate target dir, no clobber).
#   2. Record machine state (nvidia-smi, nproc, RAM, git rev).
#   3. Run run_cpu.sh and/or run_gpu.sh; outputs land in
#      <out-root>/sweep_cpu/ and <out-root>/sweep_gpu/.
#
# Strategy recap: one full 20-2000 Hz / 200-point sweep per scenario.
# Parameters differ per MODEL CLASS (small/medium/heavy/large tiers in
# common.sh), while per-FREQUENCY adaptation is the binary's own job
# (GCRO-DR recycling across adjacent freqs + --rebuild-precond-every).
# Band-splitting a scenario was rejected: it cold-restarts the recycler at
# every boundary and each band would be HF-extended independently, so bands
# cannot be concatenated into one RoomEQ dataset.

set -u
SRC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "$SRC_DIR/common.sh"

BACKEND="both"
OUT_ROOT="data_tests/roomeq"
EXTRA=()

while [ $# -gt 0 ]; do
  case "$1" in
    --backend) BACKEND="$2"; shift 2 ;;
    --out-root) OUT_ROOT="$2"; shift 2 ;;
    *) EXTRA+=("$1"); shift ;;
  esac
done

ROOT="$(repo_root)"
cd "$ROOT"
mkdir -p "$OUT_ROOT"

echo "=== build CPU binary ==="
cargo build --release -p sonium-sim-room --bin generate-roomeq-data
CPU_BIN="$ROOT/target/release/generate-roomeq-data"

if [ "$BACKEND" = "gpu" ] || [ "$BACKEND" = "both" ]; then
  echo "=== build CUDA binary ==="
  CARGO_TARGET_DIR="$ROOT/target/cuda" cargo build --release \
    -p sonium-sim-room --features cuda --bin generate-roomeq-data
fi

echo "=== machine probe ==="
{
  echo "date: $(date -u +%FT%TZ)"
  echo "git: $(git_rev)"
  echo "nproc: $(nproc)"
  free -g || true
  nvidia-smi --query-gpu=name,memory.total,memory.free,driver_version,compute_cap \
    --format=csv || echo "no GPU"
  "$CPU_BIN" --execution-contract --max-concurrent 1 2>/dev/null || true
} | tee "$OUT_ROOT/probe.txt"

case "$BACKEND" in
  cpu) "$SRC_DIR/run_cpu.sh" --out "$OUT_ROOT/sweep_cpu" --bin "$CPU_BIN" "${EXTRA[@]}" ;;
  gpu) "$SRC_DIR/run_gpu.sh" --out "$OUT_ROOT/sweep_gpu" \
         --bin "$ROOT/target/cuda/release/generate-roomeq-data" "${EXTRA[@]}" ;;
  both)
    "$SRC_DIR/run_cpu.sh" --out "$OUT_ROOT/sweep_cpu" --bin "$CPU_BIN" "${EXTRA[@]}"
    "$SRC_DIR/run_gpu.sh" --out "$OUT_ROOT/sweep_gpu" \
      --bin "$ROOT/target/cuda/release/generate-roomeq-data" "${EXTRA[@]}"
    ;;
  *) echo "unknown --backend: $BACKEND" >&2; exit 2 ;;
esac
