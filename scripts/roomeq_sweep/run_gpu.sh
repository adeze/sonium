#!/usr/bin/env bash
# GPU sweep: all 20 scenarios on the RTX PRO 6000 Blackwell via the CUDA FEM
# path. Serial over scenarios (one GPU), 32 host threads for assembly/IO.
#
# Usage:
#   run_gpu.sh [--out DIR] [--bin PATH] [--only SCENARIO]... [--no-fallback]
#
# GPU settings rationale (from the CLI help + fem_runner consts):
#   --gpu-precision f64-outer : Carson-Higham IR (Complex64 outer FGMRES +
#                           Complex32 inner). Pure f32 stalls above the
#                           promoted residual gate (~1e-4) on the full
#                           20-2000 Hz grid; f64-outer is the proven
#                           full-grid choice on RTX cards.
#   --gpu-precond auto    : resolves to CUDA AMG. Never use point-jacobi for
#                           production: it stalls around residual 1e-2 on
#                           indefinite Helmholtz.
# The CUDA FEM runner is still labelled a prototype in-tree, so every
# scenario is validated (CSVs + metadata present, no failure in log) and any
# failure is recorded and, unless --no-fallback, re-run on the CPU backend
# into <out>/fallback_cpu/ so the sweep still yields a complete dataset.

set -u
SRC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "$SRC_DIR/common.sh"

OUT="data_tests/roomeq/sweep_gpu"
BIN="$(repo_root)/target/cuda/release/generate-roomeq-data"
FALLBACK=1
GPU_PRECISION="f64-outer"
GPU_PRECOND="auto"
ONLY=()

while [ $# -gt 0 ]; do
  case "$1" in
    --out) OUT="$2"; shift 2 ;;
    --bin) BIN="$2"; shift 2 ;;
    --only) ONLY+=("$2"); shift 2 ;;
    --no-fallback) FALLBACK=0; shift ;;
    # f32 is fastest but stalls above the promoted gate, so it is
    # opt-in only; f64-outer (IR) is the default full-grid choice.
    --gpu-precision) GPU_PRECISION="$2"; shift 2 ;;
    --gpu-precond) GPU_PRECOND="$2"; shift 2 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

if ! command -v nvidia-smi >/dev/null; then
  echo "nvidia-smi not found; no NVIDIA GPU visible" >&2; exit 1
fi
nvidia-smi --query-gpu=name,memory.free --format=csv
if [ ! -x "$BIN" ]; then
  echo "CUDA binary not found at $BIN" >&2
  echo "Build it first: CARGO_TARGET_DIR=target/cuda cargo build --release \\" >&2
  echo "  -p sonium-sim-room --features cuda --bin generate-roomeq-data" >&2
  exit 1
fi
LOG_DIR="$OUT/logs"
mkdir -p "$OUT" "$LOG_DIR"
: >"$OUT/manifest.jsonl"
: >"$OUT/failed.txt"

TARGETS=("${ALL_SCENARIOS[@]}")
[ "${#ONLY[@]}" -gt 0 ] && TARGETS=("${ONLY[@]}")

HOST_THREADS=32
GPU_FLAGS=(--backend cuda --gpu-device "$GPU_DEVICE"
  --gpu-precision "$GPU_PRECISION" --gpu-precond "$GPU_PRECOND")

for s in "${TARGETS[@]}"; do
  echo "=== gpu: $s ==="
  if ! run_one "$BIN" "$OUT" "$LOG_DIR" "$s" "$HOST_THREADS" \
      "${COMMON_FEM_FLAGS[@]}" "${GPU_FLAGS[@]}"; then
    echo "$s" >>"$OUT/failed.txt"
  fi
  if ! check_outputs "$OUT" "$s"; then
    echo "$s (incomplete outputs)" >>"$OUT/failed.txt"
  fi
  # Flag solver non-convergence traces for review (exit may still be 0 on
  # retry-ladder success; the log tells which).
  grep -il "non-convergen\|did not converge\|true-residual gate" "$LOG_DIR/$s.log" \
    && echo "note: $s log mentions convergence retries (see log)" || true
done

if [ -s "$OUT/failed.txt" ] && [ "$FALLBACK" -eq 1 ]; then
  CPU_BIN="$(repo_root)/target/release/generate-roomeq-data"
  if [ ! -x "$CPU_BIN" ]; then
    echo "CPU fallback binary missing at $CPU_BIN; skipping fallback" >&2
  else
    echo "--- CPU fallback for failed GPU scenarios ---"
    # Scenario names contain no spaces: strip " (incomplete outputs)" markers,
    # then dedupe so each scenario is retried exactly once.
    sed 's/ .*//' "$OUT/failed.txt" | sort -u >"$OUT/failed.uniq"
    while read -r s; do
      [ -z "$s" ] && continue
      echo "=== cpu-fallback: $s ==="
      # run_one logs to $LOG_DIR/$s.log, which currently holds the failed
      # GPU run; move it aside first so the fallback does not overwrite it.
      [ -f "$LOG_DIR/$s.log" ] && mv "$LOG_DIR/$s.log" "$LOG_DIR/gpu-$s.log"
      if run_one "$CPU_BIN" "$OUT/fallback_cpu" "$LOG_DIR" "$s" "$TOTAL_THREADS" \
          "${COMMON_FEM_FLAGS[@]}" --backend cpu; then
        mv "$LOG_DIR/$s.log" "$LOG_DIR/cpu-fallback-$s.log"
      else
        mv "$LOG_DIR/$s.log" "$LOG_DIR/cpu-fallback-$s.log"
        echo "$s (fallback failed)" >>"$OUT/failed.txt"
      fi
    done <"$OUT/failed.uniq"
    echo "--- fallback output check ---"
    while read -r s; do
      [ -z "$s" ] && continue
      check_outputs "$OUT/fallback_cpu" "$s" || echo "$s (fallback incomplete)" >>"$OUT/failed.txt"
    done <"$OUT/failed.uniq"
  fi
fi

echo "--- summary ---"
sort -u -o "$OUT/failed.txt" "$OUT/failed.txt"
if [ -s "$OUT/failed.txt" ]; then
  echo "FAILED/INCOMPLETE (see logs + fallback_cpu/ if enabled):"
  cat "$OUT/failed.txt"
  exit 1
fi
echo "GPU sweep complete: $OUT"
