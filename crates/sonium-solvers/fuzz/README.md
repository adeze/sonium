# sonium-solvers fuzzing

Install `cargo-fuzz`, then run from this directory:

```bash
cargo fuzz run checked_fgmres -- -max_len=256
cargo fuzz run checked_csr -- -max_len=256
cargo fuzz run checked_preconditioners -- -max_len=256
```

The target intentionally accepts numerical-error results; it asserts only
that checked FGMRES does not panic for bounded generated operators and RHS
vectors. Minimized regressions belong under `corpus/checked_fgmres/`,
`corpus/checked_csr/`, or `corpus/checked_preconditioners/`.
