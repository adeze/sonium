# MeshLang compiler performance

## Benchmark suite

Run:

```bash
cargo bench -p meshlang-compiler --bench compiler_bench
```

The Criterion suite uses generated, deterministic inputs and records four
growth dimensions:

- `compiler_large_fragment_array`: 64 and 256 volume operands passed through a
  fragment array.
- `compiler_nested_csg`: 32 and 128 left-associated boolean operands.
- `compiler_import_chain_64`: cold and warm compilation of a 64-module chain.
- `compiler_candidate_recompile_64_variant_working_set`: repeated evaluation
  of a bounded design-candidate set with a shared imported module.

Criterion HTML reports are enabled by the workspace. Compare distributions and
regressions rather than copying machine-specific timings into source control.

## Incremental cache contract

`IncrementalCompiler` owns two independent caches:

1. `ImportResolver` reads imported sources and validates cached ASTs by a
   content hash. File modification times are not trusted.
2. `CanonicalIrCache` keys lowered IR by the fully flattened AST structural
   hash. A changed root parameter or changed imported declaration therefore
   produces a miss; an unchanged candidate reuses the exact `Arc`.

`stats()` exposes entries, hits, and misses for both layers. `invalidate_module`
supports watcher-driven eviction, `clear_caches` bounds memory at tool-defined
epochs, and `reset_stats` starts a new observation window without discarding
entries. File IDs survive cache clears so source provenance remains stable.

The implementation deliberately caches whole parsed modules and whole
canonical variants. It does not yet do incremental subtree parsing, dependency-
exact sensitivity invalidation, or unaffected-root variant splicing. Add those
only after profiles show that hashing/whole-module work dominates real editor or
optimizer workloads.

## Fuzz and resource bounds

The `fuzz/compiler_pipeline` target caps input at 1 MiB and parser nesting at
64. It crosses import validation, name resolution, type checking, canonical IR,
and both code generators. Run coverage-guided campaigns separately from normal
Cargo tests; the checked-in corpus keeps boundary regressions reproducible even
when `cargo-fuzz` is unavailable.
