#!/usr/bin/env bash
# CPU sweep: all 20 scenarios, full 20-2000 Hz grid, tiered parameters.
#
# Usage:
#   run_cpu.sh [--out DIR] [--bin PATH] [--tiers small,medium,heavy,large]
#              [--only SCENARIO]...
#
# Concurrency (jobs x threads <= 60 cores, <= ~160 GB):
#   small  4x15 | medium 2x30 | heavy 2x30 | large 1x60
# Each job also gets --memory-limit-gb 160 so the binary's own estimator
# never prompts on stdin; real OOM protection comes from the jobs x threads
# cap plus graded meshes (uniform grading 1.0 is the memory worst case and
# is deliberately NOT used here).

set -u
SRC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "$SRC_DIR/common.sh"

OUT="data_tests/roomeq/sweep_cpu"
BIN="$(repo_root)/target/release/generate-roomeq-data"
TIERS="small,medium,heavy,large"
ONLY=()

while [ $# -gt 0 ]; do
  case "$1" in
    --out) OUT="$2"; shift 2 ;;
    --bin) BIN="$2"; shift 2 ;;
    --tiers) TIERS="$2"; shift 2 ;;
    --only) ONLY+=("$2"); shift 2 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

if [ ! -x "$BIN" ]; then
  echo "CPU binary not found at $BIN" >&2
  echo "Build it first: cargo build --release -p sonium-sim-room --bin generate-roomeq-data" >&2
  exit 1
fi

LOG_DIR="$OUT/logs"
mkdir -p "$OUT" "$LOG_DIR"
: >"$OUT/manifest.jsonl"
: >"$OUT/failed.txt"

tier_var() {  # tier_var <tier> <JOBS|THREADS>
  case "$1,$2" in
    small,JOBS) echo 4 ;; small,THREADS) echo 15 ;;
    medium,JOBS) echo 2 ;; medium,THREADS) echo 30 ;;
    heavy,JOBS) echo 2 ;; heavy,THREADS) echo 30 ;;
    large,JOBS) echo 1 ;; large,THREADS) echo 60 ;;
  esac
}

run_tier() {  # run_tier <tier> <scenario...>
  local tier="$1"; shift
  local jobs threads
  jobs="$(tier_var "$tier" JOBS)"; threads="$(tier_var "$tier" THREADS)"
  local extra=()
  [ "$tier" = "heavy" ] && extra=(--gmres-restart 60)
  [ "$tier" = "large" ] && extra=(--rebuild-precond-every 8)
  echo "=== tier $tier: $* (jobs=$jobs threads=$threads ${extra[*]:-defaults}) ==="
  local running=0 s
  for s in "$@"; do
    # shellcheck disable=SC2086
    run_one "$BIN" "$OUT" "$LOG_DIR" "$s" "$threads" \
      "${COMMON_FEM_FLAGS[@]}" ${extra[@]+"${extra[@]}"} || echo "$s" >>"$OUT/failed.txt" &
    running=$((running + 1))
    if [ "$running" -ge "$jobs" ]; then
      wait -n
      running=$((running - 1))
    fi
  done
  wait
}

pick() {  # pick <tier-name>; prints tier scenarios filtered by --only
  local t="$1" s o
  case "$t" in
    small) set -- "${SCENARIOS_SMALL[@]}" ;;
    medium) set -- "${SCENARIOS_MEDIUM[@]}" ;;
    heavy) set -- "${SCENARIOS_MEDIUM_HEAVY[@]}" ;;
    large) set -- "${SCENARIOS_LARGE[@]}" ;;
    *) echo "unknown tier: $t" >&2; exit 2 ;;
  esac
  for s in "$@"; do
    if [ "${#ONLY[@]}" -eq 0 ]; then
      printf '%s\n' "$s"
    else
      for o in "${ONLY[@]}"; do
        [ "$o" = "$s" ] && printf '%s\n' "$s"
      done
    fi
  done
}

IFS=',' read -ra WANT <<<"$TIERS"
for t in "${WANT[@]}"; do
  mapfile -t L < <(pick "$t")
  [ "${#L[@]}" -eq 0 ] && continue
  run_tier "$t" "${L[@]}"
done

echo "--- output check ---"
if [ "${#ONLY[@]}" -eq 0 ]; then
  check_outputs "$OUT" "${ALL_SCENARIOS[@]}" || true
else
  check_outputs "$OUT" "${ONLY[@]}" || true
fi
if [ -s "$OUT/failed.txt" ]; then
  echo "FAILED scenarios:"; sort -u "$OUT/failed.txt"
  exit 1
fi
echo "CPU sweep complete: $OUT"
