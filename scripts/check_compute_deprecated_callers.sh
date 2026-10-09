#!/usr/bin/env bash
# Fail if any non-allowlisted Rust code calls the deprecated panicking CSR
# constructors from sonium-compute (`from_host*`, `from_csr_host_parts*`,
# `copy_values_from_host`). Production paths must use the `try_*` variants;
# the allowlist below covers the deprecated definitions themselves plus
# test/bench fixtures that intentionally use them under `#[allow(deprecated)]`.
#
# The host-tape `record_input` family is intentionally NOT covered: the name
# is shared with the checked device-tape method, so grep cannot separate
# them. Compiler `deprecated` warnings remain the backstop for that family.
# `BackendAmgPreconditioner::from_host_amg` is likewise excluded: it was
# never deprecated (the AMG setup family keeps infallible wrappers by
# design), so it is filtered below rather than flagged.
set -eu

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if ! command -v rg >/dev/null 2>&1; then
    echo "error: ripgrep (rg) is required to run this gate." >&2
    exit 2
fi

ALLOWLIST=(
    "crates/sonium-compute/src/operator.rs"
    "crates/sonium-compute/src/operator_tape.rs"
    "crates/sonium-compute/src/preconditioner.rs"
    "crates/sonium-compute/src/preconditioner/saddle_point.rs"
    "crates/sonium-compute/src/preconditioner/chebyshev.rs"
    "crates/sonium-compute/src/block_operator.rs"
    "crates/sonium-compute/src/block_krylov.rs"
    "crates/sonium-compute/src/gmres.rs"
    "crates/sonium-compute/src/gmres_ir.rs"
    "crates/sonium-compute/tests/device_preconditioner_cuda.rs"
    "crates/sonium-solvers/tests/backend_krylov.rs"
    "crates/sonium-solvers/benches/backend_gmres_bench.rs"
    "crates/sonium-solvers/crates/sonium-solvers-iterative/src/backend.rs"
    "crates/sonium-solvers/crates/sonium-solvers-iterative/src/host_operator.rs"
)

# Calls are `Type::from_host(...)` associated functions. The boundary class
# `[^a-zA-Z0-9_]` excludes the checked `try_*` spellings (`try_from_host`
# has `_` before `from_host`) and `*_from_*` test names; the optional
# turbofish covers `from_host_cast::<T>(...)`.
PATTERN='(^|[^a-zA-Z0-9_])(from_host[a-z0-9_]*(::<[A-Za-z0-9_ ,]+>)?|from_csr_host_parts[a-z0-9_]*|copy_values_from_host)\('

matches="$(rg --no-heading -n -g '*.rs' "$PATTERN" crates || true)"
failures=""
while IFS= read -r line; do
    [ -n "$line" ] || continue
    file="${line%%:*}"
    rest="${line#*:}"
    code="${rest#*:}"
    # Skip definitions (`pub fn from_host...`, `fn from_host_..._test...`),
    # the never-deprecated `from_host_amg` family, and comment-only lines
    # (doc links included).
    case "$code" in
        *pub\ fn*|*from_host_amg*) continue ;;
    esac
    case "$code" in
        *fn\ from_host*|*fn\ backend_from_csr*|*fn\ copy_values*) continue ;;
    esac
    trimmed="$(printf '%s' "$code" | sed -e 's/^[[:space:]]*//')"
    case "$trimmed" in
        '//'*) continue ;;
    esac
    allowlisted=0
    for allowed in "${ALLOWLIST[@]}"; do
        if [ "$file" = "$allowed" ]; then
            allowlisted=1
            break
        fi
    done
    if [ "$allowlisted" -eq 0 ]; then
        failures="${failures}${line}"$'\n'
    fi
done <<< "$matches"

if [ -n "$failures" ]; then
    echo "error: deprecated panicking CSR constructor calls outside the allowlist:" >&2
    printf '%s' "$failures" >&2
    echo "Migrate to the try_* variants or extend the allowlist with justification." >&2
    exit 1
fi
echo "OK: no non-allowlisted deprecated CSR constructor calls."
