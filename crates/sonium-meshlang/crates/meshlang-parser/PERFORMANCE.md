# Parser performance protocol

`benches/parser_bench.rs` measures lexing and strict parsing at 1,000 and
10,000 generated declarations. The generated programs are intentionally flat,
so a doubling sweep exposes accidental quadratic tokenization or recovery.

```bash
cargo bench -p meshlang-parser --bench parser_bench
```

Criterion records throughput distributions. Peak resident memory is measured
around the same benchmark executable with the platform-native sampler:

```bash
# macOS
/usr/bin/time -l cargo bench -p meshlang-parser --bench parser_bench

# Linux
/usr/bin/time -v cargo bench -p meshlang-parser --bench parser_bench
```

Record the declaration count, elapsed result, and maximum resident set size in
CI artifacts. Compare accepted corpus/throughput changes against the preceding
main-branch artifact. `ParseCache` is the editor baseline: it caches complete
files by stable content hash and parse options. It also supports bounded
borrowed source ranges keyed by original-file hash, range, and options; this is
syntax-only subtree parsing, not dependency-aware incremental compilation. A
future editor trace should decide whether finer-grained dependency invalidation
is worth its complexity.

## 2026-08-01 baseline

Optimized Criterion run on the aarch64 macOS development worker:

| Operation | 1,000 declarations | 10,000 declarations | 10x scaling |
|---|---:|---:|---:|
| Lex | 192.22 µs | 1.8447 ms | 9.60x |
| Parse | 572.20 µs | 6.2794 ms | 10.97x |

The generated 100-file import graph (100 declarations per file) measured
11.465 ms for strict parsing and 5.2431 ms through the warm content-hash file
cache (central estimates from the short Criterion profile).

The short memory-profile run reported a maximum resident set size of
481,804,288 bytes for the complete Cargo/Criterion process (57,164,352-byte
peak-memory-footprint counter). The CI linearity gate is at most 12x elapsed
time when declarations increase 10x; both lexer and parser satisfy it. The RSS
number is a harness-level regression baseline rather than a claim that one AST
retains that amount.
