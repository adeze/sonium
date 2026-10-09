# sonium-qa (lib: `sonium_qa`, version: 0.1.5)

QA harness for sonium: 214+ standalone validation binaries plus shared library helpers.

## Layout

```
src/
  lib.rs            Norms (relative_error, residual_norm, relative_residual_norm),
                    matrix helpers (spd_tridiagonal, check_symmetry, check_spd),
                    mesh point generators, analytical Laplacian eigenvalues
  fixtures/         Shared test fixtures (room geometries, speaker rigs, wavenumbers)
  validation/       Reusable validation routines (convergence_rate, MMS helpers)
bin/                214+ standalone validation binaries
```

## Binary Naming Convention

- `bem_*.rs` → `qa-bem-*` (BEM accuracy / convergence / FMM / SOTA)
- `fem_*.rs` → `qa-fem-*` (FEM physics modules and convergence)
- `solvers_*.rs` → `qa-solvers-*` (iterative, nonlinear, optimization, preconditioners)
- `room_*.rs`, `speaker_*.rs`, `pipeline_*.rs` → application-level validation

Every binary must:

1. Exit with code 0 on success.
2. Print `SUCCESS` on completion.
3. Never print `FAILED` on success.

Manifest-listed release cases should additionally emit one validated
`QA_RESULT:` JSON line using the shared `QaResult` API. Keep the human markers
for compatibility, but put numerical metrics, thresholds, replay seed, and
backend provenance in the structured record so the release runner can audit
the claim without scraping prose.

The Just recipes enforce all three.

## QA Tiers (Justfile)

| Tier | Purpose | Cost |
|------|---------|------|
| `qa-smoke` | Pre-merge physics sanity checks | seconds |
| `qa-deep` | Convergence + analytical scattering + WaveHoltz + Schwarz-PML | minutes |
| `qa-expensive` | Room/speaker application pipeline (real BEM solves) | tens of minutes |
| `qa-full` | All standalone QA binaries | hours |

## Tolerances

```
REL_TOL      = 1e-2     // Relative error against analytical
ABS_TOL      = 1e-8
RESIDUAL_TOL = 1e-8
GRADIENT_TOL = 1e-8
KKT_TOL      = 1e-6
```

Binaries that need looser tolerances (e.g. coarse-mesh BEM) document them inline.

## Library Helpers (used by binaries)

- `relative_error(a, b)` — handles zero-norm reference deterministically (0 if both zero, ∞ if reference zero and a non-zero, else `||a-b|| / ||b||`).
- `relative_residual_norm(A, x, b)` — `||Ax - b|| / ||b||` with the same zero-rhs handling.
- `convergence_rate(errors, element_sizes)` — `ln(e_i/e_{i+1}) / ln(h_i/h_{i+1})`. Use the variant that takes mesh sizes, not the historical sample-count form.
- `check_spd` — necessary screening only; not a substitute for Cholesky.
- `rank::spearman_rho` / `rank::kendall_tau_b` / `rank::top_k_set` — rank-correlation helpers used by `qa-speaker-rank-preservation` (ties get average ranks; top-k is k smallest, i.e. pass negated values for maximization objectives).

## Testing

```bash
cargo test -p sonium-qa --release
cargo run -p sonium-qa --release --bin <qa-binary>
cargo check -p sonium-qa && cargo clippy -p sonium-qa
```

## Notes

- `qa-wave-1d-analytical-consistency` and
  `qa-wave-cylinder-2d-analytical-consistency` emit structured
  analytical-reference records, not production BEM solver evidence. They must
  remain `reference-only` and must not be counted as BEM accuracy proofs.
  `qa-bem-3d-sphere` is the production Mie-sphere solver validation.
- New physics binaries should land in the right tier — keeping `qa-smoke` under a minute is a hard rule.
