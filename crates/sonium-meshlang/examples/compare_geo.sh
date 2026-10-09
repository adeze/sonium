#!/usr/bin/env bash
# Compare original .geo files against MeshLang-compiled .geo output.
#
# For each pair, this script:
#   1. Compiles the .mlg to a .geo file
#   2. Runs gmsh -3 on both the original and compiled .geo
#   3. Prints mesh statistics side-by-side (nodes, elements, physical groups)
#
# Usage:
#   cd crates/sonium-meshlang/examples
#   bash compare_geo.sh
#
# Prerequisites: gmsh in PATH, cargo build done

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

TMPDIR="${TMPDIR:-/tmp}/meshlang_compare_$$"
mkdir -p "$TMPDIR"
trap 'rm -rf "$TMPDIR"' EXIT

# Build the meshlang compiler
echo "=== Building meshlang compiler ==="
cargo build -p meshlang-compiler --release --quiet 2>/dev/null
MESHLANG="$(cd "$SCRIPT_DIR/../../.." && pwd)/target/release/meshlang"

if [ ! -x "$MESHLANG" ]; then
    echo "ERROR: meshlang binary not found at $MESHLANG"
    exit 1
fi

# Extract mesh stats from gmsh output
# Prints: nodes elements physical_groups
mesh_stats() {
    local geo_file="$1"
    local log="$TMPDIR/gmsh_$$.log"

    # Gmsh may exit non-zero on warnings; capture all output and check for mesh file
    gmsh "$geo_file" -3 -format msh4 -o "$TMPDIR/out_$$.msh" -v 3 >"$log" 2>&1
    local rc=$?

    if [ ! -f "$TMPDIR/out_$$.msh" ]; then
        echo "MESH_FAILED"
        grep -i 'error' "$log" >&2
        return 1
    fi

    local nodes elements physicals
    # Gmsh prints "Info    : N nodes M elements" — match that pattern
    nodes=$(grep -Eo '[0-9]+ nodes [0-9]+ elements' "$log" | tail -1 | sed 's/ nodes.*//' || echo "0")
    elements=$(grep -Eo '[0-9]+ nodes [0-9]+ elements' "$log" | tail -1 | sed 's/.* nodes //' | sed 's/ elements//' || echo "0")
    physicals=$(grep -c 'Physical' "$geo_file" 2>/dev/null || echo "0")

    echo "$nodes $elements $physicals"
    rm -f "$TMPDIR/out_$$.msh" "$log"
}

# Print a comparison row
compare_row() {
    local label="$1" orig_val="$2" mlg_val="$3"
    local diff=""
    if [ "$orig_val" != "MESH_FAILED" ] && [ "$mlg_val" != "MESH_FAILED" ] && [ "$orig_val" -gt 0 ] 2>/dev/null; then
        local pct=$(( (mlg_val - orig_val) * 100 / orig_val ))
        if [ "$pct" -eq 0 ]; then
            diff="(same)"
        elif [ "$pct" -gt 0 ]; then
            diff="(+${pct}%)"
        else
            diff="(${pct}%)"
        fi
    fi
    printf "  %-20s %10s  %10s  %s\n" "$label" "$orig_val" "$mlg_val" "$diff"
}

# Test pairs: original .geo → .mlg
declare -a PAIRS=(
    "originals/t16.geo:gmsh_t16_csg.mlg"
    "originals/t19.geo:gmsh_t19_loft_pipe.mlg"
)
# t5.geo uses the built-in kernel (not OCC), so geometry differs fundamentally.
# We still compile and mesh the .mlg but don't compare stats with t5.geo.

PASS=0
FAIL=0

echo ""
for pair in "${PAIRS[@]}"; do
    orig="${pair%%:*}"
    mlg="${pair##*:}"
    compiled="$TMPDIR/$(basename "$mlg" .mlg)_compiled.geo"

    echo "=== $(basename "$orig") vs $(basename "$mlg") ==="

    # Compile .mlg → .geo
    if ! "$MESHLANG" compile "$mlg" -o "$compiled" 2>/dev/null; then
        echo "  FAIL: meshlang compilation failed for $mlg"
        FAIL=$((FAIL + 1))
        echo ""
        continue
    fi

    # Mesh original
    echo "  Meshing original: $orig"
    orig_stats=$(mesh_stats "$orig") || { FAIL=$((FAIL + 1)); echo ""; continue; }
    read -r orig_nodes orig_elems orig_phys <<< "$orig_stats"

    # Mesh compiled
    echo "  Meshing compiled: $compiled"
    mlg_stats=$(mesh_stats "$compiled") || { FAIL=$((FAIL + 1)); echo ""; continue; }
    read -r mlg_nodes mlg_elems mlg_phys <<< "$mlg_stats"

    # Print comparison
    printf "\n  %-20s %10s  %10s\n" "" "ORIGINAL" "MESHLANG"
    printf "  %-20s %10s  %10s\n" "" "--------" "--------"
    compare_row "Nodes" "$orig_nodes" "$mlg_nodes"
    compare_row "Elements" "$orig_elems" "$mlg_elems"
    compare_row "Physical groups" "$orig_phys" "$mlg_phys"

    PASS=$((PASS + 1))
    echo ""
done

# Also compile-only the t5 and helmholtz examples (no original .geo to compare)
echo "=== Compile-only checks ==="
for mlg in gmsh_t5_cheese.mlg gmsh_helmholtz_resonator.mlg; do
    compiled="$TMPDIR/$(basename "$mlg" .mlg)_compiled.geo"
    if "$MESHLANG" compile "$mlg" -o "$compiled" 2>/dev/null; then
        if mesh_stats "$compiled" >/dev/null 2>&1; then
            echo "  PASS: $mlg → compile + mesh OK"
            PASS=$((PASS + 1))
        else
            echo "  FAIL: $mlg → compiled but gmsh mesh failed"
            FAIL=$((FAIL + 1))
        fi
    else
        echo "  FAIL: $mlg → meshlang compilation failed"
        FAIL=$((FAIL + 1))
    fi
done

echo ""
echo "=== Summary: $PASS passed, $FAIL failed ==="
exit $FAIL
