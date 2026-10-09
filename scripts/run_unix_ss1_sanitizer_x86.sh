#!/usr/bin/env bash
# Reproducible native Linux/x86 SS1 sparse AddressSanitizer evidence.
# The resulting bundle is self-contained and explicitly marked unaccepted;
# it is evidence for CI review, not an accepted CI release artifact.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
target_dir="${CARGO_TARGET_DIR:-${repo_root}/.target-unix-ss1-sanitizer}"
output_root="${SONIUM_OUTPUT_DIR:-${repo_root}/target/sonium-unix-ss1-sanitizer}"
case "$target_dir" in /*) ;; *) target_dir="$repo_root/$target_dir" ;; esac
case "$output_root" in /*) ;; *) output_root="$repo_root/$output_root" ;; esac
run_kind="${SONIUM_SPARSE_SANITIZER_RUN_KIND:-local}"

test "$(uname -s)" = Linux || { echo "SS1 sanitizer requires Linux" >&2; exit 2; }
test "$(uname -m)" = x86_64 || { echo "SS1 sanitizer requires x86_64" >&2; exit 2; }
grep -qw avx2 /proc/cpuinfo || { echo "SS1 sanitizer requires AVX2" >&2; exit 2; }
grep -qw fma /proc/cpuinfo || { echo "SS1 sanitizer requires FMA" >&2; exit 2; }
test "$run_kind" = local || test "$run_kind" = ci || {
  echo "SS1 sanitizer run kind must be local or ci" >&2
  exit 2
}
test "$(rustc +nightly -vV | sed -n 's/^host: //p')" = x86_64-unknown-linux-gnu

mkdir -p "$output_root"
artifact="$output_root/native.json"
manifest="$output_root/manifest.json"
rm -f "$artifact" "$artifact.sha256" "$manifest" "$manifest.sha256" \
  "$output_root/native-sanitizer.log" "$output_root/rustc-nightly-vv.txt" \
  "$output_root/cargo-nightly-version.txt" "$output_root/lscpu.txt" \
  "$output_root/environment.txt" "$output_root/checksums.sha256"

rustc +nightly -Vv > "$output_root/rustc-nightly-vv.txt"
cargo +nightly -V > "$output_root/cargo-nightly-version.txt"
lscpu > "$output_root/lscpu.txt"
{
  printf 'run_kind=%s\n' "$run_kind"
  printf 'sanitizer=address\n'
  printf 'toolchain=nightly\n'
  printf 'target_arch=x86_64\n'
  printf 'dispatch=runtime-native-avx2-fma\n'
  printf 'force_scalar=0\n'
  printf 'disable_fma=0\n'
  printf 'asan_options=detect_leaks=0\n'
  printf 'lsan_options=detect_leaks=0\n'
  printf 'target_dir=%s\n' "$target_dir"
  printf 'output_root=%s\n' "$output_root"
} > "$output_root/environment.txt"

env \
  ASAN_OPTIONS=detect_leaks=0 \
  LSAN_OPTIONS=detect_leaks=0 \
  RUSTFLAGS=-Zsanitizer=address \
  RUSTC_WRAPPER="$repo_root/scripts/sonium-sanitizer-rustc-wrapper" \
  SONIUM_SPARSE_FORCE_SCALAR=0 \
  SONIUM_SPARSE_DISABLE_FMA=0 \
  SONIUM_SPARSE_SANITIZER=address \
  SONIUM_SPARSE_SANITIZER_RUN_KIND="$run_kind" \
  SONIUM_SPARSE_EVIDENCE_OUT="$artifact" \
  CARGO_TARGET_DIR="$target_dir" \
  cargo +nightly test --release -p sonium-solvers-sparse --test native_x86_sparse_evidence -- \
  --nocapture --test-threads=1 > "$output_root/native-sanitizer.log" 2>&1

(cd "$output_root" && sha256sum native.json > native.json.sha256 && sha256sum -c native.json.sha256)

python3 - "$output_root" "$manifest" "$run_kind" <<'PY'
import hashlib
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1]).resolve()
manifest = pathlib.Path(sys.argv[2]).resolve()
run_kind = sys.argv[3]
payload = [
    "native.json",
    "native.json.sha256",
    "native-sanitizer.log",
    "rustc-nightly-vv.txt",
    "cargo-nightly-version.txt",
    "lscpu.txt",
    "environment.txt",
]
digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
artifact = json.loads((root / "native.json").read_text(encoding="utf-8"))
report = {
    "schema": "sonium-solvers-sparse.ss1.native-x86-sanitizer/v1",
    "artifact_kind": "sonium-solvers-sparse.ss1.native-x86-sanitizer-bundle",
    "run_kind": run_kind,
    "acceptance_status": "not_accepted",
    "sanitizer": "address",
    "toolchain": "nightly",
    "target_arch": "x86_64",
    "evidence": {
        "path": "native.json",
        "sha256": digest(root / "native.json"),
        "schema": artifact["schema"],
        "artifact_kind": artifact["artifact_kind"],
        "corpus_sha256": artifact["corpus_sha256"],
    },
    "files": [
        {"path": name, "sha256": digest(root / name), "bytes": (root / name).stat().st_size}
        for name in payload
    ],
    "checksums": {"path": "checksums.sha256", "entries": len(payload) + 2},
}
manifest.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
PY

# Regenerate both sidecars and the aggregate list after the manifest is final.
(cd "$output_root" && sha256sum manifest.json > manifest.json.sha256 && sha256sum \
  native.json native.json.sha256 native-sanitizer.log rustc-nightly-vv.txt \
  cargo-nightly-version.txt lscpu.txt environment.txt manifest.json manifest.json.sha256 \
  > checksums.sha256)

python3 "$repo_root/scripts/check_unix_ss1_sanitizer_x86.py" "$manifest" --negative-probes
echo "SS1 native x86 sanitizer evidence: passed; unaccepted bundle=$output_root"
