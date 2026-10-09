#!/usr/bin/env bash
set -euo pipefail

# Produce the host-applicable PP4 shard fan-in.  The aggregate is deliberately
# gated: it is not the unavailable PP2/PP3 external/full-fidelity release join.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET="$ROOT/.target-pp4-fanin-fresh"
EVIDENCE="$TARGET/evidence"
CARGO_TARGET_DIR="$TARGET/cargo"
export ROOT TARGET EVIDENCE CARGO_TARGET_DIR
export TMPDIR="$TARGET/tmp"
export SONIUM_PP4_FAULTS_DIR="$EVIDENCE/release/pp4-faults"
export RAYON_NUM_THREADS=1
export SONIUM_QA_DETERMINISTIC_MODE=1
export SONIUM_QA_THREAD_POLICY=RAYON_NUM_THREADS=1
mkdir -p "$TMPDIR" "$EVIDENCE" "$CARGO_TARGET_DIR"

test "$(uname -s)" = Linux || { echo "PP4 fan-in requires native Linux" >&2; exit 2; }
test "$(uname -m)" = x86_64 || { echo "PP4 fan-in requires native x86_64" >&2; exit 2; }

QA="$EVIDENCE/qa"
RELEASE="$EVIDENCE/release"
GALLERY="$EVIDENCE/gallery-replay"
rm -rf "$QA" "$RELEASE" "$GALLERY" "$EVIDENCE/shards"
mkdir -p "$QA" "$RELEASE" "$GALLERY"

SONIUM_PP4_QA_DIR="$QA" python3 "$ROOT/crates/sonium-sim-propellers/benchmarks/run_pp4_qa.py" \
  | tee "$EVIDENCE/qa-run.log"
python3 "$ROOT/crates/sonium-sim-propellers/benchmarks/check_pp4_qa.py" \
  "$QA/qa-propeller-fan-in.json" | tee "$EVIDENCE/qa-check.json"

SONIUM_OUTPUT_DIR="$RELEASE" \
  cargo test --release -p sonium-sim-propellers --test pp4_release_evidence -- \
  --nocapture | tee "$EVIDENCE/release-run.log"
python3 "$ROOT/crates/sonium-sim-propellers/benchmarks/check_pp4_release.py" \
  "$RELEASE/native-x86-pp4-release-v1.json" | tee "$EVIDENCE/release-check.json"

SONIUM_OUTPUT_DIR="$GALLERY" \
  cargo test --release -p sonium-sim-propellers --test pp4_gallery_replay_evidence -- \
  --nocapture | tee "$EVIDENCE/gallery-run.log"
printf '%s  %s\n' "$(sha256sum "$GALLERY/native-x86-pp4-gallery-replay-v1.json" | cut -d' ' -f1)" \
  "native-x86-pp4-gallery-replay-v1.json" > "$GALLERY/checksums.sha256"
python3 "$ROOT/crates/sonium-sim-propellers/benchmarks/check_pp4_gallery_replay.py" \
  "$GALLERY/native-x86-pp4-gallery-replay-v1.json" | tee "$EVIDENCE/gallery-check.json"

python3 "$ROOT/crates/sonium-sim-propellers/benchmarks/build_pp4_fanin.py" \
  "$EVIDENCE" --qa "$QA" --release "$RELEASE" --gallery "$GALLERY"

CHECKER="$EVIDENCE/pp4-fanin-checker.json"
python3 "$ROOT/crates/sonium-sim-propellers/benchmarks/check_pp4_fanin.py" \
  "$EVIDENCE/pp4-release-fanin-v1.json" \
  > "$TARGET/pp4-fanin-checker-output.json"
python3 - "$TARGET/pp4-fanin-checker-output.json" "$CHECKER" \
  "$EVIDENCE/pp4-release-fanin-v1.json" <<'PY'
import hashlib
import json
import pathlib
import sys

output, summary, artifact = map(pathlib.Path, sys.argv[1:])
result = json.loads(output.read_text())
summary.write_text(json.dumps({
    "schema": "sonium.propellers.pp4-release-fanin-checker.v1",
    "status": "pass",
    "checker": "crates/sonium-sim-propellers/benchmarks/check_pp4_fanin.py",
    "artifact": artifact.name,
    "artifact_sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
    "result": result,
}, indent=2) + "\n")
PY

# Rebuild only the aggregate manifest so the finalized checker result is also
# checksum-bound, then replay the complete strict checker against that bundle.
python3 "$ROOT/crates/sonium-sim-propellers/benchmarks/build_pp4_fanin.py" \
  "$EVIDENCE" --qa "$QA" --release "$RELEASE" --gallery "$GALLERY" \
  --checker-summary "$CHECKER"
python3 "$ROOT/crates/sonium-sim-propellers/benchmarks/check_pp4_fanin.py" \
  "$EVIDENCE/pp4-release-fanin-v1.json"

(cd "$EVIDENCE" && sha256sum -c checksums.sha256)
sha256sum "$EVIDENCE/checksums.sha256"
echo "PP4 host-applicable fan-in complete: gated on PP2/PP3 external/full-fidelity lanes"
