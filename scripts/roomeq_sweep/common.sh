#!/usr/bin/env bash
# Shared definitions for the RoomEQ 20 Hz - 2 kHz full-sweep scripts.
#
# Design (see run_all.sh header for rationale):
#   * One full 20-2000 Hz / 200-point log sweep per scenario per backend.
#     The binary solves frequencies sequentially *by design* (warm-start from
#     the previous frequency; GCRO-DR subspace recycling measured harmful,
#     see the flag note below), solves the sources of one frequency in
#     parallel, and runs scenarios sequentially so each one owns
#     the whole rayon pool. Scale-out therefore happens at THIS level:
#     several single-scenario binary processes, each with its own
#     --max-concurrent slice of the 60-core budget.
#   * Parameters are tiered by room class (small / medium / large+surround),
#     not by frequency band: splitting one scenario into freq bands would
#     cold-restart the warm-start chain at every boundary and each band would
#     be HF-extended independently, so bands are not concatenable.
#     Per-frequency adaptation inside a sweep is already handled by the binary
#     (--rebuild-precond-every; subspace recycling is disabled, see below).
#   * NEVER pass --auto-reduce-freq in production: it silently shrinks the
#     contracted 20-2000 Hz grid. Always pass an explicit --memory-limit-gb
#     large enough to pass the estimator so the binary never prompts on
#     stdin (prompt_yes_no hangs under non-interactive runners).

set -euo pipefail

# --------------------------------------------------------------------------
# Machine budget (RTX PRO 6000 Blackwell / 128 CPUs / ~256 GB RAM machine;
# user-capped sweep budget: 60 cores, 160 GB RAM, one 96 GB GPU).
# --------------------------------------------------------------------------
TOTAL_THREADS=60
TOTAL_MEM_GB=160
GPU_DEVICE=0

# --------------------------------------------------------------------------
# Scenario catalogue (20 built-ins, from scenarios.rs `name: "..."`).
# --------------------------------------------------------------------------
SCENARIOS_SMALL=(
  small_stereo_2_0
  small_stereo_2_1
  small_stereo_2_2_mso
  small_stereo_2_2_cardioid
  small_stereo_2_2_group
)
SCENARIOS_MEDIUM=(
  medium_stereo_2_0
  medium_stereo_2_1
  medium_multi_sub_4
  medium_multi_seat
  medium_multi_sub_multi_seat
  medium_surround_5_0
  medium_surround_5_1
)
SCENARIOS_MEDIUM_HEAVY=(
  medium_surround_5_1_4
  medium_surround_5_2_4_multi_seat
)
SCENARIOS_LARGE=(
  large_stereo_2_0
  large_stereo_2_1
  large_multi_sub_4
  large_multi_seat_2_1
  large_surround_5_1
  large_surround_5_1_4
)
ALL_SCENARIOS=(
  "${SCENARIOS_SMALL[@]}"
  "${SCENARIOS_MEDIUM[@]}"
  "${SCENARIOS_MEDIUM_HEAVY[@]}"
  "${SCENARIOS_LARGE[@]}"
)

# --------------------------------------------------------------------------
# Common FEM flags: identical for every tier unless a tier overrides them.
# k2 is the production shifted-Laplacian preset (legacy is the default but
# not the production choice; csl is experimental). oras is the sweep CPU
# preconditioner: matched A/B shows 1.5-2.5x over block-jacobi at equal
# iteration counts (cheaper setup + cheaper apply) with parity-level outputs
# (max 0.0002 dB / 0.0031 deg over 1020 rows); see the flag note below.
# (Global ilut apply is mostly serial and is correctly NOT the default.)
# Grading 4.0 x 0.6 m keeps the wavelength-resolved mesh only near
# sources/listeners; uniform (1.0) is the memory worst case.
# --------------------------------------------------------------------------
# Frequency grid override for smoke tests, e.g.
#   MIN_FREQ=20 MAX_FREQ=30 NUM_POINTS=3 run_cpu.sh --only small_stereo_2_0 ...
# Defaults are the production 20-2000 Hz / 200-point grid.
MIN_FREQ="${MIN_FREQ:-20}"
MAX_FREQ="${MAX_FREQ:-800}"
NUM_POINTS="${NUM_POINTS:-100}"
# shellcheck disable=SC2034
COMMON_FEM_FLAGS=(
  --solver fem
  --min-freq "$MIN_FREQ" --max-freq "$MAX_FREQ" --num-points "$NUM_POINTS"
  --mesh-grading 4.0 --mesh-fine-radius 0.6
  --shifted-laplacian k2
  --source-model point
  # NOTE: oras over block-jacobi by measurement 2026-09-24. Same iteration
  # counts at high k, but setup is ms instead of seconds-to-minutes (70 ms vs
  # 2.4 s on 24k meshes; 0.26 s vs 4.7 s at 283k/2000 Hz) and each apply is
  # cheaper, for 1.5-2.5x wall-clock (16-18 s vs 40-44 s/mode high band;
  # 221/279 s vs 328/412 s per mode at 2000 Hz). Full-domain path also
  # healthy (3000-3300 iters/source, 92 ms setup, gate met). Outputs at
  # parity (same 1e-6 gate + defect correction). HABC measures identically
  # on this path. Revisit only with contrary production evidence.
  --preconditioner oras
  # NOTE: recycling is OFF (0) by measurement, not by theory. GCRO-DR(30)
  # across adjacent frequencies slowed the high band ~2-3x vs warm-start-only
  # (700-800 Hz: 2.8x wall-clock, plus 13.6k/26k-iteration blowups) because
  # the deflated subspace goes stale across %-level k-steps while adding
  # per-iteration augmentation overhead. Outputs are unaffected (same 1e-6
  # gate enforced by true-residual verification + defect correction; max
  # 0.0006 dB / 0.0054 deg drift over 1020 rows). Revisit only with
  # production-step (2.3%) A/B evidence; see PARALLELISM_FIX_PLAN.md.
  --recycler gcro-dr --recycle-subspace-size 0
  --rebuild-precond-every 5
  --gmres-tolerance 1e-6 --gmres-restart 50
)

# Per-tier overrides (appended after COMMON_FEM_FLAGS):
#  small  - cheap meshes; 4 jobs x 15 threads = 60 cores
#  medium - mid meshes, more sources; 2 jobs x 30 threads
#  heavy  - 10-16 sources/seats; 2 jobs x 30 threads, wider restart window
#  large  - peak memory lives here; 1 job x 60 threads (safest under 160 GB)
TIER_JOBS_small=4;         TIER_THREADS_small=15
TIER_JOBS_medium=2;        TIER_THREADS_medium=30
TIER_JOBS_heavy=2;         TIER_THREADS_heavy=30
TIER_JOBS_large=1;         TIER_THREADS_large=60
# shellcheck disable=SC2034
TIER_FLAGS_heavy=(--gmres-restart 60)
# shellcheck disable=SC2034
TIER_FLAGS_large=(--rebuild-precond-every 8)

# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
repo_root() {
  # scripts/roomeq_sweep -> repo root
  local d
  d="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
  printf '%s' "$d"
}

git_rev() {
  git -C "$(repo_root)" rev-parse --short HEAD 2>/dev/null || echo "unknown"
}

# run_one <binary> <out_root> <log_dir> <scenario> <threads> [extra flags...]
# Runs a single scenario sweep, logs to <log_dir>/<scenario>.log, appends a
# JSON manifest line to <out_root>/manifest.jsonl. Returns nonzero on failure
# without killing the caller's sweep loop.
run_one() {
  local bin="$1" out_root="$2" log_dir="$3" scenario="$4" threads="$5"
  shift 5
  local log="$log_dir/$scenario.log"
  local start end rc=0
  start="$(date -u +%FT%TZ)"
  # shellcheck disable=SC2086
  "$bin" --scenario "$scenario" --output-dir "$out_root" \
    --max-concurrent "$threads" --memory-limit-gb "$TOTAL_MEM_GB" \
    "$@" >"$log" 2>&1 || rc=$?
  end="$(date -u +%FT%TZ)"
  printf '{"scenario":"%s","threads":%s,"exit":%d,"start":"%s","end":"%s","git":"%s"}\n' \
    "$scenario" "$threads" "$rc" "$start" "$end" "$(git_rev)" >>"$out_root/manifest.jsonl"
  if [ "$rc" -ne 0 ]; then
    echo "FAIL $scenario (see $log)" >&2
  else
    echo "OK   $scenario"
  fi
  return "$rc"
}

# check_outputs <out_root> <scenario...> : verify expected exports exist.
check_outputs() {
  local out_root="$1"
  shift
  local missing=0 s
  for s in "$@"; do
    for f in config.json simulation_metadata.json; do
      if [ ! -f "$out_root/fem/$s/$f" ]; then
        echo "MISSING $out_root/fem/$s/$f" >&2
        missing=1
      fi
    done
    if ! compgen -G "$out_root/fem/$s/*.csv" >/dev/null; then
      echo "MISSING CSVs in $out_root/fem/$s/" >&2
      missing=1
    fi
  done
  return "$missing"
}
