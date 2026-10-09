# `sme2/` — ARM SME2 dense BLAS micro-kernels

Single-threaded dense BLAS-2 / BLAS-3 kernels for Apple silicon and ARM
servers with `FEAT_SME2`. Production path for `sonium_compute::cpu::blas2`
and `sonium_compute::cpu::blas3` on aarch64; falls back to scalar
reference code elsewhere or when SME2 is unavailable.

The kernels follow Deng et al., *"Demystifying ARM SME to Optimize General
Matrix Multiplications"* ([arxiv 2512.21473v1][paper]). The notable Apple
M4 quirk around `BFMOPA` widening is documented in `gemm_bf16.rs`.

[paper]: https://arxiv.org/abs/2512.21473

## What's here

| File | Purpose |
|------|---------|
| `mod.rs`           | Re-exports + integration tests (21 cases covering panel/Kc/Nc tails, grouped loads, non-tight strides, bf16). |
| `detect.rs`        | Runtime detection of SME2, `FEAT_SME_F64F64`, streaming vector length, and L2 budget. |
| `gemm.rs`          | f32 / f64 GEMM driver: paneling, Kc/Nc cache blocking, A/B packing, thread-local workspaces. |
| `gemm_bf16.rs`     | bf16 × bf16 → f32 GEMM (widening at pack time — see "BFMOPA on Apple silicon" below). |
| `gemv.rs`          | f32 / f64 GEMV wrappers. |
| `blas1.rs`         | f32 / f64 streaming-SVE BLAS-1 wrappers (axpy, scale, dot, nrm2). Dispatched from `cpu::blas1` above ~4096 elements; below threshold NEON wins. |
| `sme2_kernels.S`   | All assembly: 2× GEMV, 4× GEMM single/4-tile (f32+f64), 1× DGEMM 8-tile, 6× streaming-SVE BLAS-1 (axpy/scale/dot f32+f64), 1× SVL reader, 1× preserved-but-unused BFMOPA kernel. |

## Pipeline

```
┌─ Rust driver (gemm.rs / gemm_bf16.rs / gemv.rs) ──────────────────┐
│                                                                   │
│  for j_c in 0..N step Nc:        ← outer Nc blocking (adaptive)   │
│    for p_c in 0..K step Kc:      ← outer Kc blocking (adaptive)   │
│      pack B[p_c..,j_c..] → packed_B   (Nr-wide panels, contiguous)│
│      for i_p in 0..M step Mr:                                     │
│        pack A[i_p..,p_c..] → packed_A (col-major within row-panel)│
│        for j_r in 0..nc_full step Nr:                             │
│          ┌─► sonium_sgemm_panel4_sme2     (4-tile fast path)      │
│        for j_r in nc_full..nc step SVL_T:                         │
│          └─► sonium_sgemm_panel_sme2      (single-tile col tail)  │
└───────────────────────────────────────────────────────────────────┘
```

| Layer | What it does | Where |
|-------|--------------|-------|
| **L0 — single-tile FMOPA** | `SVL_T × SVL_T` outer product per inner-k step into one ZA tile. At SVL=512: 256 f32 FMAs or 64 f64 FMAs per instruction. | `sme2_kernels.S::sonium_{s,d}gemm_panel_sme2` |
| **L1 — multi-tile (4-tile) FMOPA** | One A column drives **four** FMOPAs into ZA0..ZA3; B side uses a grouped multi-vector `LD1W {Zn-Zn+3}` (paper's ~900 GB/s path vs ~230 GB/s for four single-Z loads). 1024 f32 FMAs / 256 f64 FMAs per inner step. | `sme2_kernels.S::sonium_{s,d}gemm_panel4_sme2` |
| **L1b — DGEMM 8-tile FMOPA** | f64 has 8 ZA.D tiles (vs 4 ZA.S for f32). One A column drives **eight** FMOPAs into ZA0..ZA7.D; B side uses two grouped 4-vector loads to cover 8×SVL_D = 64 contiguous f64 lanes. 512 f64 FMAs per inner step — 2× the 4-tile path. dgemm tries 8-tile first, then 4-tile for any 4-aligned-but-not-8-aligned remainder, then single-tile for the column tail. | `sme2_kernels.S::sonium_dgemm_panel8_sme2` |
| **L2 — A panel + B panel packing** | A panel is packed column-major within each row-panel so the inner-k loop reads contiguous A columns with one `LD1W`. B is packed into Nr-wide column panels so the 4-tile kernel's `mul vl` offsets land on contiguous rows. Both packs reuse thread-local growth buffers — no per-call allocation. | `gemm.rs::pack_a_panel_kc` / `pack_b_panels` |
| **L3 — Kc / Nc cache blocking** | Adaptive: each is engaged only when the unblocked B-band working set would overflow the L2 budget. When blocked, the first Kc pass uses caller's `beta`; subsequent passes use `beta=1` so partial sums accumulate into C. | `gemm.rs::choose_kc` / `choose_nc` |
| **L4 — GEMV grouped 4-Z load + 4-acc unroll** | Inner dot product reads 4×SVL_T elements per iteration via `LD1{W,D} {Zn-Zn+3}, pn8/z, [...]` into four independent FMA accumulators (z24..z27). Breaks the dep chain that throttles a single-accumulator dot to one FMA per latency. | `sme2_kernels.S::sonium_{s,d}gemv_sme2` |

## Public API

```rust
use sonium_compute::sme2::{gemm, gemm_bf16, gemv, detect};

// f32 / f64 GEMM
gemm::sgemm(m, n, k, alpha, &a, lda, &b, ldb, beta, &mut c, ldc);
gemm::dgemm(m, n, k, alpha, &a, lda, &b, ldb, beta, &mut c, ldc);

// bf16 inputs, f32 output / accumulator
gemm_bf16::gemm_bf16(m, n, k, alpha, &a, lda, &b, ldb, beta, &mut c, ldc);

// f32 / f64 GEMV
gemv::sgemv(m, n, alpha, &a, lda, &x, beta, &mut y);
gemv::dgemv(m, n, alpha, &a, lda, &x, beta, &mut y);

// Inspect what was detected
let caps = detect::Sme2Caps::detect();    // has_sme2, has_f64, svl_bits
let l2   = detect::l2_budget_bytes();     // cache-blocking budget
```

## Tuning knobs

```bash
# Override the auto-detected L2 working-set budget (bytes).
# Below this size choose_kc / choose_nc skip blocking entirely.
SONIUM_L2_BUDGET_BYTES=4194304 cargo run ...
```

Compile-time constants (rare to touch — defaults match Apple M4 P-cluster):

| Constant | Default | Meaning |
|----------|---------|---------|
| `TILES_PER_PANEL`  | 4    | Number of ZA tiles the multi-tile kernel uses. f32: 16×64 output, f64: 8×32. |
| `SGEMM_KC_CAP`     | 512  | Cap on `Kc` once outer-K blocking engages. |
| `DGEMM_KC_CAP`     | 256  | Same for f64. |
| `SGEMM_NC_CAP`     | 2048 | Cap on `Nc` once outer-N blocking engages. |
| `DGEMM_NC_CAP`     | 2048 | Same for f64. |

L2 budget detection (`detect::l2_budget_bytes`):

1. `SONIUM_L2_BUDGET_BYTES` env var (must parse as `usize`, ≥ 256 KB).
2. macOS: `sysctl hw.perflevel0.l2cachesize / 2` (P-cluster L2 with sharing
   headroom). Falls back to `hw.l2cachesize / 2` on older Apple silicon
   that doesn't expose perflevel0.
3. Linux: `sysconf(_SC_LEVEL2_CACHE_SIZE) / 2`.
4. Fallback: 8 MB.

The `/ 2` margin reserves room for `packed_A`, the C row band, and the
still-cold A/B rows being streamed, since other P-cores in the same
cluster may be doing useful work.

## Performance (Apple M4 Pro, SVL=512, single-threaded)

GEMM, vs Apple Accelerate (lower is better):

| Shape | Type | Sonium | Accelerate | Speedup |
|-------|------|--------|------------|---------|
| 256³  | f32  | 87 µs   | 319 µs   | **3.66×** |
| 1024³ | f32  | 2.08 ms | 19.55 ms | **9.40×** |
| 2048·128·2048 | f32 | 2.14 ms | 10.32 ms | **4.82×** |
| 256³  | f64  | 172 µs  | 594 µs   | **3.45×** |
| 1024³ | f64  | 6.63 ms | 36.56 ms | **5.52×** |

GEMV:

| Shape | Type | Sonium | Accelerate | Speedup |
|-------|------|--------|------------|---------|
| 1024×1024 | f32 | 76.9 µs | 953 µs  | **12.4×** |
| 4096×1024 | f32 | 313 µs  | 8.60 ms | **27.5×** |
| 1024×1024 | f64 | 145 µs  | 1.41 ms | **9.7×**  |

bf16 GEMM (widens to f32 at pack time, routes through `sgemm`):

| Shape | bf16 path | f32 baseline | Widening overhead |
|-------|-----------|--------------|-------------------|
| 256³  | 91 µs   | 89 µs   | ≈ 2% |
| 1024³ | 2.81 ms | 2.49 ms | ≈ 13% |

## Reproducing benchmarks

```bash
cargo bench -p sonium-compute --bench blas3_compare -- gemm_f32
cargo bench -p sonium-compute --bench blas2_compare -- gemv_f32
cargo bench -p sonium-compute --bench blas3_compare -- gemm_bf16
```

Each bench prints `sonium / accelerate` rows side-by-side and a
Gelem/s throughput. Sizes live in `SIZES` at the top of each bench file.

## Streaming-mode discipline

Every kernel that touches the ZA tile follows the same prologue/epilogue:

```text
stp d8-d15                  ; save callee-saved SIMD (smstart clobbers these)
fmov w/x, s/d args          ; stash float args in GPRs (also clobbered)
smstart sm                  ; enter streaming SVE
smstart za                  ; enable ZA tile storage
... kernel body ...
smstop za
smstop sm
ldp d8-d15                  ; restore
ret
```

**Note**: bare `smstart` (no qualifier) appears to silently *not* enable
ZA on Apple M4 — we hit this during development. Always use the explicit
`smstart sm` + `smstart za` sequence. Same on the way out.

The Rust glue never sees streaming mode directly; transitions are 100 %
asm-side. `fmov` to/from GPRs is used to stash float arguments across
the SMSTART since SIMD register state is undefined post-transition.

## BFMOPA on Apple silicon

The widening BFMOPA kernel (`sonium_bfgemm_panel_sme2`) assembles
correctly with `bfmopa za0.s, p0/m, p0/m, z0.h, z1.h`, but Apple M4 Pro
executes it as a **single-lane outer product** — only the first lane of
each H-pair contributes to the output. Two empirical tests confirmed
this (we computed `0.5625` where the spec's pair-dot gives `0.3125`,
and verified the difference is exactly the missing odd-lane terms).

The kernel is preserved in `sme2_kernels.S` with a header noting this;
the Rust dispatch in `gemm_bf16.rs` widens bf16 → f32 at pack time and
routes through the f32 single-tile FMOPA kernel instead. If a future
silicon revision fixes pair-widening, only the dispatch in `gemm_bf16.rs`
needs to change.

## Testing

```bash
cargo test -p sonium-compute --release --lib sme2
cargo test -p sonium-compute --release --lib test_detect_caps -- --nocapture
```

21 SME2-specific tests cover:

- Basic / identity / alpha / accumulate (single-tile path) × {f32, f64}
- Multi-panel with tail (M/N panel iteration + tail predicates)
- 4-tile with mixed tail (multi-tile fast path + single-tile column tail)
- Kc multi-pass (partial-sum accumulation across blocked K)
- Nc multi-block (outer N blocking)
- Non-tight leading dimensions (lda > k, ldb > n, ldc > n)
- GEMV grouped 4-Z load + tail × {f32, f64}
- bf16 basic + odd-K with row/col tails

## Future work

- DGEMM 8-tile FMOPA kernel — ✅ done (`sonium_dgemm_panel8_sme2`). 8×64 micro-kernel using ZA0..ZA7.D, 512 f64 FMAs per inner step (2× the 4-tile path).
- PRFM prefetch hints in the inner-k loop could hide L2 latency on
  large GEMMs.
- True hardware bf16 acceleration once Apple silicon implements
  BFMOPA pair-widening per the ARM ARM (preserved kernel ready to re-enable).
