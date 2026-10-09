# Parser fuzzing

Install `cargo-fuzz`, then run from this directory:

```bash
cargo fuzz run parse_recovering -- -max_len=1048576
# Invariant target: deterministic JSON/span/recovery contract.
cargo fuzz run parse_recovering_invariants -- -max_len=4096
```

The checked-in corpus contains every minimized historical crash/recovery shape.
When a crash or misparse is fixed, add its minimized input under
`corpus/parse_recovering/` before closing the report. CI smoke runs should use a
fixed time budget; long coverage-guided campaigns retain their corpus artifact
and compare corpus additions explicitly in review.
