# Compiler fuzzing

Install `cargo-fuzz`, then run from this directory:

```bash
cargo fuzz run compiler_pipeline -- -max_len=1048576
```

The target treats arbitrary bytes as lossy UTF-8, applies bounded strict
parsing, validates import paths without loading them, then crosses name
resolution, type checking, canonical lowering, and both code generators when
the previous stage accepts the input. The seed corpus represents valid and
invalid inputs at each boundary.

Keep minimized regressions in `corpus/compiler_pipeline/`. A CI smoke campaign
should use a fixed time budget and preserve corpus/crash artifacts.
