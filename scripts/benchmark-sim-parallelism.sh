#!/usr/bin/env bash
set -euo pipefail

workspace_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
evidence_dir=${SONIUM_SCALING_OUT:-"$workspace_dir/target/parallel-evidence"}
report_path="$evidence_dir/structural-release-scaling.csv"
motor_report_path="$evidence_dir/motor-extraction-release-scaling.csv"
mkdir -p "$evidence_dir"

cd "$workspace_dir"
cargo build --release -p sonium-sim-structural-mechanics --bin structural-bracket-demo
cargo build --release -p sonium-sim-speaker-modeling --bin motor-extract

printf 'size,threads,elapsed_seconds,user_seconds,system_seconds,maxrss_kib\n' >"$report_path"
for size in small medium large; do
    for threads in 1 2 4 8 16 32 64; do
        /usr/bin/time -a -o "$report_path" \
            -f "$size,$threads,%e,%U,%S,%M" \
            target/release/structural-bracket-demo \
            --scaling-size "$size" \
            --threads "$threads" \
            "$evidence_dir/$size-t$threads" >/dev/null
    done
done

printf 'threads,elapsed_seconds,user_seconds,system_seconds,maxrss_kib\n' >"$motor_report_path"
for threads in 1 2 4 8 16 32 64; do
    /usr/bin/time -a -o "$motor_report_path" \
        -f "$threads,%e,%U,%S,%M" \
        target/release/motor-extract \
        --geometry crates/sonium-sim-speaker-modeling/examples/motor_extraction_reference.json \
        --threads "$threads" \
        --output "$evidence_dir/motor-t$threads.json" >/dev/null
done
for threads in 2 4 8 16 32 64; do
    cmp "$evidence_dir/motor-t1.json" "$evidence_dir/motor-t$threads.json"
done
sha256sum "$evidence_dir/motor-t1.json" >"$evidence_dir/motor-extraction.sha256"

printf 'wrote %s\n' "$report_path"
printf 'wrote %s\n' "$motor_report_path"
printf 'wrote %s\n' "$evidence_dir/motor-extraction.sha256"
printf '%s\n' 'Run each production binary with --execution-contract to record its resolved automatic pool and observed concurrency.'
