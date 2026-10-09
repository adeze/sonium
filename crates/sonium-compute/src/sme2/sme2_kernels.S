// ARM SME2 GEMV and GEMM kernels for sonium-compute.
//
// Compiled via cc::Build in build.rs with -march=armv9-a+sme2+sme-f64f64.
// All functions use the AAPCS64 calling convention.
//
// SME2 tile geometry (SVL=512 bits, Apple M4):
//   f32: SVL_S = 16 elements/register, ZA = 16x16 f32 tile
//   f64: SVL_D =  8 elements/register, ZA =  8x8  f64 tile
//
// SMSTART enters streaming mode and enables ZA (zeroes all NEON/SVE state).
// SMSTOP exits streaming mode and disables ZA (zeroes NEON/SVE state again).
// We must save/restore callee-saved NEON registers (d8-d15) around transitions.
//
// Symbol naming: Mach-O (macOS) prepends an underscore to C symbol names;
// ELF (Linux) does not. The `.S` extension means clang runs the C
// preprocessor on this file, so we use a CPP macro to spell each global the
// way the active object format expects. Rust externs always write the bare
// name (e.g. `sonium_sgemm_panel_sme2`); the toolchain attaches the leading
// underscore on Mach-O targets when linking.

// On ARM64 GAS `;` introduces a comment (it is NOT a statement separator
// the way it is on x86 GAS), and CPP line continuation collapses multiple
// physical lines into one logical line. That rules out the usual
// "single-macro emits three directives" trick. Instead we just expose a
// token-pasting `SYM(name)` macro and write the three directives (`.globl`,
// `.p2align`, and the label) explicitly at each function — three short
// lines that the assembler sees as three distinct statements.

#if defined(__APPLE__) || defined(__MACH__)
#  define SYM(name) _##name
#else
#  define SYM(name) name
#endif

.text

// ============================================================================
// SVL query: returns Streaming Vector Length in bytes.
// ============================================================================
//
// size_t sonium_read_svl_bytes(void)
//
// MUST only be called after confirming SME2 is present (via sysctl/hwcap).
// Enters streaming mode, reads RDSVL, exits streaming mode.

.globl SYM(sonium_read_svl_bytes)
.p2align 4
SYM(sonium_read_svl_bytes):
    stp     d8, d9, [sp, #-64]!
    stp     d10, d11, [sp, #16]
    stp     d12, d13, [sp, #32]
    stp     d14, d15, [sp, #48]

    smstart sm
    rdsvl   x0, #1              // x0 = SVL in bytes
    smstop  sm

    ldp     d14, d15, [sp, #48]
    ldp     d12, d13, [sp, #32]
    ldp     d10, d11, [sp, #16]
    ldp     d8, d9, [sp], #64
    ret


// ============================================================================
// SGEMV: y = alpha * A * x + beta * y
// ============================================================================
//
// void sonium_sgemv_sme2(size_t m, size_t n, float alpha,
//                         const float* a, size_t lda,
//                         const float* x, float beta,
//                         float* y)
//
// ABI:  m=x0, n=x1, alpha=s0, a=x2, lda=x3, x_vec=x4, beta=s1, y=x5
//
// Algorithm: For each row i of A, compute dot(A[i,:], x) using streaming
// SVE, then y[i] = alpha * dot + beta * y[i].

.globl SYM(sonium_sgemv_sme2)
.p2align 4
SYM(sonium_sgemv_sme2):
    // Save callee-saved registers (SMSTART will zero NEON state)
    stp     d8, d9, [sp, #-64]!
    stp     d10, d11, [sp, #16]
    stp     d12, d13, [sp, #32]
    stp     d14, d15, [sp, #48]
    stp     x19, x20, [sp, #-32]!
    stp     x21, x22, [sp, #16]

    // Stash float params to GPRs before SMSTART zeros NEON
    fmov    w19, s0             // w19 = alpha (as bits)
    fmov    w20, s1             // w20 = beta (as bits)

    // Save remaining args to callee-saved regs before SMSTART
    mov     x21, x2             // a
    mov     x22, x5             // y

    // Enter streaming SVE mode (no ZA needed for GEMV)
    smstart sm

    // x0=m, x1=n, x3=lda, x4=x_vec still live (SMSTART touches only SIMD).
    lsl     x3, x3, #2         // lda_bytes = lda * 4

    // Predicate setup. SME2 predicates pn0..pn15 alias the SVE predicate
    // registers p0..p15 — we use a distinct PN index (8) and SVE index (0)
    // only to keep the disassembly readable. Both names refer to *different*
    // physical predicate registers.
    //   pn8.s : count-form predicate for the grouped 4-Z multi-vector LD1W
    //           (paper §3: ~900 GB/s vs ~230 GB/s for four single-Z loads).
    //   p0.s  : full SVE predicate used by the tail single-Z loads and by
    //           the final FADDV horizontal reduction.
    ptrue   pn8.s
    ptrue   p0.s

    // Precompute SVL_S elements and 4*SVL_S byte / element stride.
    rdsvl   x13, #1                // SVL bytes      (= 64 at SVL=512)
    lsr     x14, x13, #2           // SVL_S elements (= 16)
    lsl     x15, x13, #2           // 4*SVL bytes    (= 256)
    lsl     x16, x14, #2           // 4*SVL_S elems  (= 64)

    // Row loop: x8 = row index i
    mov     x8, #0
.Lsgemv_row:
    cmp     x8, x0
    b.ge    .Lsgemv_done

    madd    x9, x8, x3, x21       // &A[i, 0]

    // Four independent accumulators break the FMA dep chain that
    // throttles a single-accumulator dot to one FMA per latency.
    mov     z24.s, #0
    mov     z25.s, #0
    mov     z26.s, #0
    mov     z27.s, #0

    mov     x10, #0                // j
    mov     x11, x4                // &x[0]
    mov     x12, x9                // &A[i, 0]

    // Main loop: stride of 4*SVL_S elements, grouped 4-Z load on both
    // A row and x. Each iteration consumes 4*SVL_S inputs (=64 f32 at
    // SVL=512) and produces 4 FMAs into independent accumulators.
.Lsgemv_4x:
    add     x17, x10, x16          // j + 4*SVL_S
    cmp     x17, x1
    b.gt    .Lsgemv_tail

    ld1w    {z0.s - z3.s}, pn8/z, [x12]
    ld1w    {z4.s - z7.s}, pn8/z, [x11]

    fmla    z24.s, p0/m, z0.s, z4.s
    fmla    z25.s, p0/m, z1.s, z5.s
    fmla    z26.s, p0/m, z2.s, z6.s
    fmla    z27.s, p0/m, z3.s, z7.s

    add     x12, x12, x15          // advance by 4*SVL bytes
    add     x11, x11, x15
    add     x10, x10, x16          // j += 4*SVL_S
    b       .Lsgemv_4x

.Lsgemv_tail:
    // Fold the 4 accumulators down to z29 once, then handle 1..3 single-Z
    // tail iterations into z29.
    fadd    z24.s, z24.s, z25.s
    fadd    z26.s, z26.s, z27.s
    fadd    z29.s, z24.s, z26.s

.Lsgemv_col:
    cmp     x10, x1
    b.ge    .Lsgemv_col_done

    whilelt p1.s, x10, x1
    ld1w    {z0.s}, p1/z, [x12]
    ld1w    {z1.s}, p1/z, [x11]
    fmla    z29.s, p1/m, z0.s, z1.s

    add     x12, x12, x13          // SVL bytes
    add     x11, x11, x13
    add     x10, x10, x14          // SVL_S elements
    b       .Lsgemv_col

.Lsgemv_col_done:
    // Horizontal reduction of z29
    faddv   s2, p0, z29.s         // s2 = sum of all z29 lanes

    // result = alpha * dot + beta * y[i]
    // s2 = dot product result
    // s30 lane 0 = alpha, s31 lane 0 = beta
    fmov    s3, s2              // s3 = dot
    fmov    s4, w19             // s4 = alpha (reload from GPR to be safe)
    fmul    s3, s3, s4          // s3 = alpha * dot

    // Use x6 (an unused argument slot — GEMV only takes 6 integer args, so
    // x6/x7 are free scratch) for the y[i] byte offset. x13 must NOT be
    // clobbered here — the inner tail loop on subsequent rows reads it as
    // "SVL bytes". And x18 is platform-reserved on Apple AAPCS64, so it's
    // off-limits as scratch despite being unused by the calling code.
    lsl     x6, x8, #2         // byte offset = i * 4
    ldr     s5, [x22, x6]      // s5 = y[i]
    fmov    s6, w20             // s6 = beta
    fmadd   s3, s6, s5, s3     // s3 = alpha*dot + beta*y[i]

    str     s3, [x22, x6]      // y[i] = result

    add     x8, x8, #1
    b       .Lsgemv_row

.Lsgemv_done:
    smstop sm

    // Restore callee-saved registers
    ldp     x21, x22, [sp, #16]
    ldp     x19, x20, [sp], #32
    ldp     d14, d15, [sp, #48]
    ldp     d12, d13, [sp, #32]
    ldp     d10, d11, [sp, #16]
    ldp     d8, d9, [sp], #64
    ret


// ============================================================================
// DGEMV: y = alpha * A * x + beta * y
// ============================================================================
//
// void sonium_dgemv_sme2(size_t m, size_t n, double alpha,
//                         const double* a, size_t lda,
//                         const double* x, double beta,
//                         double* y)
//
// ABI:  m=x0, n=x1, alpha=d0, a=x2, lda=x3, x_vec=x4, beta=d1, y=x5

.globl SYM(sonium_dgemv_sme2)
.p2align 4
SYM(sonium_dgemv_sme2):
    stp     d8, d9, [sp, #-64]!
    stp     d10, d11, [sp, #16]
    stp     d12, d13, [sp, #32]
    stp     d14, d15, [sp, #48]
    stp     x19, x20, [sp, #-32]!
    stp     x21, x22, [sp, #16]

    // Stash double params to GPRs before SMSTART zeros NEON
    fmov    x19, d0             // x19 = alpha bits
    fmov    x20, d1             // x20 = beta bits

    mov     x21, x2             // a
    mov     x22, x5             // y

    smstart sm

    // x0=m, x1=n, x3=lda, x4=x_vec, x21=a, x22=y still live.
    lsl     x3, x3, #3                 // lda_bytes = lda * 8

    // pn8 and p0 alias distinct physical predicate registers (see SGEMV
    // for context). pn8 in count form drives the grouped multi-vector
    // LD1D; p0 is the all-true SVE predicate used by the tail and FADDV.
    ptrue   pn8.d
    ptrue   p0.d

    rdsvl   x13, #1                    // SVL bytes        (=64)
    lsr     x14, x13, #3               // SVL_D elements   (=8)
    lsl     x15, x13, #2               // 4*SVL bytes      (=256)
    lsl     x16, x14, #2               // 4*SVL_D elements (=32)

    mov     x8, #0
.Ldgemv_row:
    cmp     x8, x0
    b.ge    .Ldgemv_done

    madd    x9, x8, x3, x21            // &A[i,0]

    mov     z24.d, #0
    mov     z25.d, #0
    mov     z26.d, #0
    mov     z27.d, #0

    mov     x10, #0
    mov     x11, x4
    mov     x12, x9

.Ldgemv_4x:
    add     x17, x10, x16
    cmp     x17, x1
    b.gt    .Ldgemv_tail

    ld1d    {z0.d - z3.d}, pn8/z, [x12]
    ld1d    {z4.d - z7.d}, pn8/z, [x11]

    fmla    z24.d, p0/m, z0.d, z4.d
    fmla    z25.d, p0/m, z1.d, z5.d
    fmla    z26.d, p0/m, z2.d, z6.d
    fmla    z27.d, p0/m, z3.d, z7.d

    add     x12, x12, x15
    add     x11, x11, x15
    add     x10, x10, x16
    b       .Ldgemv_4x

.Ldgemv_tail:
    fadd    z24.d, z24.d, z25.d
    fadd    z26.d, z26.d, z27.d
    fadd    z29.d, z24.d, z26.d

.Ldgemv_col:
    cmp     x10, x1
    b.ge    .Ldgemv_col_done

    whilelt p1.d, x10, x1
    ld1d    {z0.d}, p1/z, [x12]
    ld1d    {z1.d}, p1/z, [x11]
    fmla    z29.d, p1/m, z0.d, z1.d

    add     x12, x12, x13
    add     x11, x11, x13
    add     x10, x10, x14
    b       .Ldgemv_col

.Ldgemv_col_done:
    faddv   d2, p0, z29.d              // d2 = sum of all lanes

    // result = alpha * dot + beta * y[i]
    fmov    d3, d2
    fmov    d4, x19             // alpha
    fmul    d3, d3, d4          // alpha * dot

    // x6 is a free scratch slot (GEMV has no 7th integer arg). x18 is
    // platform-reserved on Apple AAPCS64 so we avoid it even though it's
    // unused by callers; x13 must not be clobbered (SVL bytes for the tail).
    lsl     x6, x8, #3         // byte offset = i * 8
    ldr     d5, [x22, x6]      // y[i]
    fmov    d6, x20             // beta
    fmadd   d3, d6, d5, d3     // alpha*dot + beta*y[i]

    str     d3, [x22, x6]

    add     x8, x8, #1
    b       .Ldgemv_row

.Ldgemv_done:
    smstop sm

    ldp     x21, x22, [sp, #16]
    ldp     x19, x20, [sp], #32
    ldp     d14, d15, [sp, #48]
    ldp     d12, d13, [sp, #32]
    ldp     d10, d11, [sp, #16]
    ldp     d8, d9, [sp], #64
    ret


// ============================================================================
// SGEMM panel micro-kernel: FMOPA outer-product accumulation into ZA0.S
// ============================================================================
//
// void sonium_sgemm_panel_sme2(
//     size_t mb,                  // 1..=SVL_S actual rows in this panel
//     size_t nb,                  // 1..=SVL_S actual cols
//     size_t k,                   // inner dim (>=1)
//     float alpha,                // s0
//     const float* packed_a,      // SVL_S * k floats, col p at &packed_a[p*SVL_S]
//                                 // (zero-padded on the row tail so rows mb..SVL_S = 0)
//     const float* b,             // &B[0, j0]
//     size_t ldb,                 // elements
//     float beta,                 // s1
//     float* c,                   // &C[i0, j0]
//     size_t ldc                  // elements (passed on stack)
// );
//
// AAPCS64: mb=x0, nb=x1, k=x2, alpha=s0, packed_a=x3, b=x4, ldb=x5,
//          beta=s1, c=x6, ldc=x7
//
// Algorithm: a true SME2 outer-product kernel. For each inner step p:
//   z0.s = packed_a[p*SVL_S .. p*SVL_S + SVL_S]   // column of A panel
//   z1.s = b[p*ldb + 0 .. p*ldb + SVL_S]          // row of B (col-tile)
//   ZA0.S += outer_product(z0, z1)                // FMOPA — SVL_S^2 FMAs / inst
// Epilogue: extract ZA0 rows 0..mb, scale by alpha, fuse with beta*C, store.
//
// References:
//   "Demystifying ARM SME to Optimize General Matrix Multiplications",
//   Deng et al., arxiv 2512.21473 — Apple M4 Pro has SVL=512b, so this is a
//   16×16 micro-kernel doing 256 FMAs per FMOPA versus 16 per SVE FMLA.
//   Used for the column-tail (nb < 4*SVL_S) of the multi-tile pipeline;
//   the full-width fast path is `sonium_sgemm_panel4_sme2` below.

.globl SYM(sonium_sgemm_panel_sme2)
.p2align 4
SYM(sonium_sgemm_panel_sme2):
    stp     d8, d9, [sp, #-64]!
    stp     d10, d11, [sp, #16]
    stp     d12, d13, [sp, #32]
    stp     d14, d15, [sp, #48]
    stp     x19, x20, [sp, #-32]!
    stp     x21, x22, [sp, #16]

    // Stash float params before SMSTART zeros NEON/SVE state.
    fmov    w19, s0                 // alpha bits in GPR (survives smstart)
    fmov    w20, s1                 // beta bits in GPR
    mov     x21, x6                 // c
    mov     x22, x4                 // b

    // Convert ldb / ldc to bytes (x5 = ldb, x7 = ldc).
    lsl     x5, x5, #2              // ldb_bytes
    lsl     x7, x7, #2              // ldc_bytes

    // Enter streaming mode AND enable ZA explicitly.
    smstart sm
    smstart za

    // Predicates:
    //   p0 = ptrue (rows; pack is row-padded so all SVL_S lanes are valid)
    //   p1 = whilelt(0, nb) (column tail predicate)
    ptrue   p0.s
    mov     x10, #0
    whilelt p1.s, x10, x1           // p1 covers nb columns

    // Zero the entire ZA storage (safest mask).
    zero    {za}

    // k-loop driver:
    //   x11 = packed_a cursor, advances by SVL bytes per p
    //   x12 = b cursor, advances by ldb_bytes per p
    //   x10 = p counter
    mov     x10, #0
    mov     x11, x3                 // packed_a
    mov     x12, x22                // b
    rdsvl   x13, #1                 // SVL in bytes (constant within streaming)
.Lsg_k:
    cmp     x10, x2
    b.ge    .Lsg_kdone

    // A column (already packed contiguously, row-padded to SVL_S).
    ld1w    {z0.s}, p0/z, [x11]
    // B row tile, column-tail predicated (zero-fills unused lanes).
    ld1w    {z1.s}, p1/z, [x12]

    // ZA0.S += z0(col) * z1(row)  — predicated by p0 (rows) and p1 (cols).
    fmopa   za0.s, p0/m, p1/m, z0.s, z1.s

    add     x11, x11, x13           // packed_a += SVL bytes
    add     x12, x12, x5            // b += ldb_bytes
    add     x10, x10, #1
    b       .Lsg_k

.Lsg_kdone:
    // Broadcast alpha / beta for the epilogue FMA (via GPR — avoids any
    // post-smstart SIMD-register-aliasing edge cases).
    dup     z30.s, w19              // bcast alpha bits
    dup     z31.s, w20              // bcast beta bits

    // Epilogue: for each row r in 0..mb store C[r] = alpha*ZA0[r] + beta*C[r].
    mov     w14, #0                 // r (used as ZA index in w12 form below)
    mov     x15, x21                // c cursor (advances by ldc_bytes per row)
.Lsg_epi:
    cmp     w14, w0                 // r < mb ?
    b.ge    .Lsg_epi_done

    // Extract ZA0 horizontal slice r. The slice-selector reg must be w12-w15.
    mov     w12, w14
    mova    z2.s, p0/m, za0h.s[w12, 0]

    // z2 = alpha * z2 (predicated by p1 so col-tail lanes stay 0).
    fmul    z2.s, p1/m, z2.s, z30.s

    // Load C row, accumulate beta * C, store.
    ld1w    {z3.s}, p1/z, [x15]
    fmla    z2.s, p1/m, z3.s, z31.s
    st1w    {z2.s}, p1, [x15]

    add     w14, w14, #1
    add     x15, x15, x7            // c += ldc_bytes
    b       .Lsg_epi

.Lsg_epi_done:
    smstop za
    smstop sm

    ldp     x21, x22, [sp, #16]
    ldp     x19, x20, [sp], #32
    ldp     d14, d15, [sp, #48]
    ldp     d12, d13, [sp, #32]
    ldp     d10, d11, [sp, #16]
    ldp     d8, d9, [sp], #64
    ret


// ============================================================================
// SGEMM 4-tile micro-kernel: 16×64 outer product (ZA0..ZA3, SVL=512)
// ============================================================================
//
// void sonium_sgemm_panel4_sme2(
//     size_t mb, size_t nb,      // nb MUST be exactly 4 * SVL_S (=64 here)
//     size_t k,
//     float alpha,               // s0
//     const float* packed_a,     // SVL_S * k floats, col p at &packed_a[p*SVL_S]
//     const float* b,            // &B[0, j0]
//     size_t ldb,
//     float beta,                // s1
//     float* c,                  // &C[i0, j0]
//     size_t ldc
// );
//
// AAPCS64 register layout matches the single-tile kernel:
//   mb=x0, nb=x1 (unused, assumed 4*SVL), k=x2, alpha=s0, packed_a=x3,
//   b=x4, ldb=x5, beta=s1, c=x6, ldc=x7
//
// Per inner-k iteration we issue:
//   z0   = packed_A[:, p]                       // one contiguous load
//   z4   = B[p, j0..j0+16]                       // 4 contiguous B tiles via
//   z5   = B[p, j0+16..j0+32]                    //   mul-vl offsets
//   z6   = B[p, j0+32..j0+48]
//   z7   = B[p, j0+48..j0+64]
//   ZA0  += z0 ⊗ z4
//   ZA1  += z0 ⊗ z5
//   ZA2  += z0 ⊗ z6
//   ZA3  += z0 ⊗ z7
// One A load drives FOUR FMOPAs (1024 FMAs / inner step at SVL=512) versus
// 256 for the single-tile kernel. Rust only calls this when nb == 4*SVL_S,
// so all predicates are `ptrue` — no col-tail handling needed inside.

.globl SYM(sonium_sgemm_panel4_sme2)
.p2align 4
SYM(sonium_sgemm_panel4_sme2):
    stp     d8, d9, [sp, #-64]!
    stp     d10, d11, [sp, #16]
    stp     d12, d13, [sp, #32]
    stp     d14, d15, [sp, #48]
    stp     x19, x20, [sp, #-32]!
    stp     x21, x22, [sp, #16]

    fmov    w19, s0                 // alpha bits in GPR
    fmov    w20, s1                 // beta bits
    mov     x21, x6                 // c
    mov     x22, x4                 // b

    lsl     x5, x5, #2              // ldb_bytes
    lsl     x7, x7, #2              // ldc_bytes

    smstart sm
    smstart za

    // p0 = ptrue.s — used as row predicate AND col predicate for every tile.
    ptrue   p0.s

    // Set up a predicate-as-counter spanning 4×SVL_S lanes (= the full
    // 64-col panel), used by the SME2 multi-vector load below. PTRUE (pn,
    // count) generates a counter pn8 with the requested element count.
    ptrue   pn8.s

    // Zero all of ZA (single insn, safest for cross-tile state).
    zero    {za}

    // k-loop:
    mov     x10, #0                 // p
    mov     x11, x3                 // packed_a cursor
    mov     x12, x22                // b cursor (row of B at offset j0)
    rdsvl   x13, #1                 // SVL bytes
.Lsg4_k:
    cmp     x10, x2
    b.ge    .Lsg4_kdone

    // Prefetch A column and B panel two iterations ahead.
    // At SVL=512 each A column is 64 bytes (x13); each packed B panel row
    // is 4*SVL = 256 bytes (x5).  Two-iteration look-ahead keeps the M4's
    // out-of-order window filled without over-saturating the L1 MSHR count.
    add     x8, x11, x13, lsl #1   // packed_a + 2*SVL_bytes
    prfm    pldl1keep, [x8]
    add     x9, x12, x5, lsl #1    // b + 2*ldb_bytes
    prfm    pldl1keep, [x9]

    // A column (contiguous from the packed panel).
    ld1w    {z0.s}, p0/z, [x11]

    // Four contiguous B tiles via SME2 grouped multi-vector load.
    // Per ARM SME2: LD1W (multi, consecutive) reads 4×SVL_S contiguous
    // f32 lanes from [x12 .. x12+4*SVL] into the consecutive register
    // group z4.s..z7.s. Paper measures ~900 GB/s for this form versus
    // ~230 GB/s for four separate single-Z loads at mul-vl offsets.
    ld1w    {z4.s-z7.s}, pn8/z, [x12]

    // 4× FMOPA, all reading the SAME A column z0.
    fmopa   za0.s, p0/m, p0/m, z0.s, z4.s
    fmopa   za1.s, p0/m, p0/m, z0.s, z5.s
    fmopa   za2.s, p0/m, p0/m, z0.s, z6.s
    fmopa   za3.s, p0/m, p0/m, z0.s, z7.s

    add     x11, x11, x13           // packed_a += SVL bytes
    add     x12, x12, x5            // b += ldb_bytes
    add     x10, x10, #1
    b       .Lsg4_k

.Lsg4_kdone:
    // Broadcast alpha / beta into Z regs (via stashed GPR bit patterns).
    dup     z30.s, w19
    dup     z31.s, w20

    // Epilogue: 4 tile stores per row.
    mov     w14, #0                 // r
    mov     x15, x21                // c cursor
.Lsg4_epi:
    cmp     w14, w0
    b.ge    .Lsg4_epi_done

    mov     w12, w14                // slice selector

    // Tile 0 — cols j0..j0+16
    mova    z2.s, p0/m, za0h.s[w12, 0]
    fmul    z2.s, p0/m, z2.s, z30.s
    ld1w    {z3.s}, p0/z, [x15]
    fmla    z2.s, p0/m, z3.s, z31.s
    st1w    {z2.s}, p0, [x15]

    // Tile 1 — cols j0+16..j0+32
    mova    z2.s, p0/m, za1h.s[w12, 0]
    fmul    z2.s, p0/m, z2.s, z30.s
    ld1w    {z3.s}, p0/z, [x15, #1, mul vl]
    fmla    z2.s, p0/m, z3.s, z31.s
    st1w    {z2.s}, p0, [x15, #1, mul vl]

    // Tile 2 — cols j0+32..j0+48
    mova    z2.s, p0/m, za2h.s[w12, 0]
    fmul    z2.s, p0/m, z2.s, z30.s
    ld1w    {z3.s}, p0/z, [x15, #2, mul vl]
    fmla    z2.s, p0/m, z3.s, z31.s
    st1w    {z2.s}, p0, [x15, #2, mul vl]

    // Tile 3 — cols j0+48..j0+64
    mova    z2.s, p0/m, za3h.s[w12, 0]
    fmul    z2.s, p0/m, z2.s, z30.s
    ld1w    {z3.s}, p0/z, [x15, #3, mul vl]
    fmla    z2.s, p0/m, z3.s, z31.s
    st1w    {z2.s}, p0, [x15, #3, mul vl]

    add     w14, w14, #1
    add     x15, x15, x7
    b       .Lsg4_epi

.Lsg4_epi_done:
    smstop  za
    smstop  sm

    ldp     x21, x22, [sp, #16]
    ldp     x19, x20, [sp], #32
    ldp     d14, d15, [sp, #48]
    ldp     d12, d13, [sp, #32]
    ldp     d10, d11, [sp, #16]
    ldp     d8, d9, [sp], #64
    ret


// ============================================================================
// DGEMM panel micro-kernel: FMOPA into ZA0.D (8x8 tile at SVL=512)
// ============================================================================
//
// void sonium_dgemm_panel_sme2(
//     size_t mb, size_t nb, size_t k,
//     double alpha,                // d0
//     const double* packed_a,      // SVL_D * k doubles, col p at &packed_a[p*SVL_D]
//     const double* b,             // &B[0, j0]
//     size_t ldb,
//     double beta,                 // d1
//     double* c,                   // &C[i0, j0]
//     size_t ldc
// );
//
// Same structure as the f32 panel kernel above; .d element type, ZA0.D tile.

.globl SYM(sonium_dgemm_panel_sme2)
.p2align 4
SYM(sonium_dgemm_panel_sme2):
    stp     d8, d9, [sp, #-64]!
    stp     d10, d11, [sp, #16]
    stp     d12, d13, [sp, #32]
    stp     d14, d15, [sp, #48]
    stp     x19, x20, [sp, #-32]!
    stp     x21, x22, [sp, #16]

    fmov    x19, d0                 // alpha bits
    fmov    x20, d1                 // beta bits
    mov     x21, x6                 // c
    mov     x22, x4                 // b

    lsl     x5, x5, #3              // ldb_bytes
    lsl     x7, x7, #3              // ldc_bytes

    smstart sm
    smstart za

    ptrue   p0.d
    mov     x10, #0
    whilelt p1.d, x10, x1           // p1 covers nb columns

    zero    {za}

    mov     x10, #0
    mov     x11, x3
    mov     x12, x22
    rdsvl   x13, #1                 // SVL in bytes
.Ldg_k:
    cmp     x10, x2
    b.ge    .Ldg_kdone

    ld1d    {z0.d}, p0/z, [x11]
    ld1d    {z1.d}, p1/z, [x12]

    fmopa   za0.d, p0/m, p1/m, z0.d, z1.d

    add     x11, x11, x13           // packed_a += SVL bytes (= 8*f64)
    add     x12, x12, x5            // b += ldb_bytes
    add     x10, x10, #1
    b       .Ldg_k

.Ldg_kdone:
    dup     z30.d, x19              // bcast alpha bits
    dup     z31.d, x20              // bcast beta bits

    mov     w14, #0
    mov     x15, x21
.Ldg_epi:
    cmp     w14, w0
    b.ge    .Ldg_epi_done

    mov     w12, w14
    mova    z2.d, p0/m, za0h.d[w12, 0]

    fmul    z2.d, p1/m, z2.d, z30.d
    ld1d    {z3.d}, p1/z, [x15]
    fmla    z2.d, p1/m, z3.d, z31.d
    st1d    {z2.d}, p1, [x15]

    add     w14, w14, #1
    add     x15, x15, x7
    b       .Ldg_epi

.Ldg_epi_done:
    smstop za
    smstop sm

    ldp     x21, x22, [sp, #16]
    ldp     x19, x20, [sp], #32
    ldp     d14, d15, [sp, #48]
    ldp     d12, d13, [sp, #32]
    ldp     d10, d11, [sp, #16]
    ldp     d8, d9, [sp], #64
    ret


// ============================================================================
// DGEMM 4-tile micro-kernel: 8×32 outer product (ZA0.D..ZA3.D, SVL=512)
// ============================================================================
//
// void sonium_dgemm_panel4_sme2(
//     size_t mb, size_t nb,      // nb MUST be exactly 4 * SVL_D (=32 here)
//     size_t k,
//     double alpha,              // d0
//     const double* packed_a,
//     const double* b,
//     size_t ldb,
//     double beta,               // d1
//     double* c,
//     size_t ldc
// );
//
// Same operand layout as the f32 4-tile kernel — one A column reused across
// four FMOPAs (256 FMAs per inner-k step at SVL=512). Rust only invokes
// this when nb == 4*SVL_D.

.globl SYM(sonium_dgemm_panel4_sme2)
.p2align 4
SYM(sonium_dgemm_panel4_sme2):
    stp     d8, d9, [sp, #-64]!
    stp     d10, d11, [sp, #16]
    stp     d12, d13, [sp, #32]
    stp     d14, d15, [sp, #48]
    stp     x19, x20, [sp, #-32]!
    stp     x21, x22, [sp, #16]

    fmov    x19, d0                 // alpha bits in GPR
    fmov    x20, d1                 // beta bits
    mov     x21, x6                 // c
    mov     x22, x4                 // b

    lsl     x5, x5, #3              // ldb_bytes
    lsl     x7, x7, #3              // ldc_bytes

    smstart sm
    smstart za

    ptrue   p0.d
    // PN counter for the 4-Z grouped load (4×SVL_D = 32 lanes total).
    ptrue   pn8.d

    zero    {za}

    mov     x10, #0
    mov     x11, x3
    mov     x12, x22
    rdsvl   x13, #1
.Ldg4_k:
    cmp     x10, x2
    b.ge    .Ldg4_kdone

    // Prefetch A column and B panel two iterations ahead.
    // Each A column is SVL bytes (x13 = 64 at SVL=512, all f64); each
    // packed B panel row is ldb_bytes = 4*SVL*2 = 256 bytes (x5).
    add     x8, x11, x13, lsl #1   // packed_a + 2*SVL_bytes
    prfm    pldl1keep, [x8]
    add     x9, x12, x5, lsl #1    // b + 2*ldb_bytes
    prfm    pldl1keep, [x9]

    ld1d    {z0.d}, p0/z, [x11]

    // Grouped 4-Z multi-vector load (SME2): 32 contiguous f64 lanes.
    ld1d    {z4.d-z7.d}, pn8/z, [x12]

    fmopa   za0.d, p0/m, p0/m, z0.d, z4.d
    fmopa   za1.d, p0/m, p0/m, z0.d, z5.d
    fmopa   za2.d, p0/m, p0/m, z0.d, z6.d
    fmopa   za3.d, p0/m, p0/m, z0.d, z7.d

    add     x11, x11, x13
    add     x12, x12, x5
    add     x10, x10, #1
    b       .Ldg4_k

.Ldg4_kdone:
    dup     z30.d, x19              // bcast alpha
    dup     z31.d, x20              // bcast beta

    mov     w14, #0
    mov     x15, x21
.Ldg4_epi:
    cmp     w14, w0
    b.ge    .Ldg4_epi_done

    mov     w12, w14

    mova    z2.d, p0/m, za0h.d[w12, 0]
    fmul    z2.d, p0/m, z2.d, z30.d
    ld1d    {z3.d}, p0/z, [x15]
    fmla    z2.d, p0/m, z3.d, z31.d
    st1d    {z2.d}, p0, [x15]

    mova    z2.d, p0/m, za1h.d[w12, 0]
    fmul    z2.d, p0/m, z2.d, z30.d
    ld1d    {z3.d}, p0/z, [x15, #1, mul vl]
    fmla    z2.d, p0/m, z3.d, z31.d
    st1d    {z2.d}, p0, [x15, #1, mul vl]

    mova    z2.d, p0/m, za2h.d[w12, 0]
    fmul    z2.d, p0/m, z2.d, z30.d
    ld1d    {z3.d}, p0/z, [x15, #2, mul vl]
    fmla    z2.d, p0/m, z3.d, z31.d
    st1d    {z2.d}, p0, [x15, #2, mul vl]

    mova    z2.d, p0/m, za3h.d[w12, 0]
    fmul    z2.d, p0/m, z2.d, z30.d
    ld1d    {z3.d}, p0/z, [x15, #3, mul vl]
    fmla    z2.d, p0/m, z3.d, z31.d
    st1d    {z2.d}, p0, [x15, #3, mul vl]

    add     w14, w14, #1
    add     x15, x15, x7
    b       .Ldg4_epi

.Ldg4_epi_done:
    smstop  za
    smstop  sm

    ldp     x21, x22, [sp, #16]
    ldp     x19, x20, [sp], #32
    ldp     d14, d15, [sp, #48]
    ldp     d12, d13, [sp, #32]
    ldp     d10, d11, [sp, #16]
    ldp     d8, d9, [sp], #64
    ret


// ============================================================================
// DGEMM 8-tile micro-kernel: FMOPA into ZA0.D..ZA7.D
// (8x64 f64 micro-kernel at SVL=512)
// ============================================================================
//
// One A column reused across EIGHT FMOPAs into all available f64 ZA tiles.
// Two grouped 4-vector loads on the B side stream 8*SVL_D = 64 f64 lanes
// per inner-k step. At SVL=512 that's 512 f64 FMAs per inner-k step (2x
// the 4-tile path's 256). Front-end pressure is comparable — still one A
// LD, one inner-step branch — so the wall-clock win is the doubled FMOPA
// density alone.
//
// void sonium_dgemm_panel8_sme2(
//     size_t mb, size_t nb,      // nb MUST be exactly 8 * SVL_D (=64 here)
//     size_t k,
//     double alpha,              // d0
//     const double* packed_a,
//     const double* b,
//     size_t ldb,                // bytes/row of B == 8 * SVL_D elements
//     double beta,               // d1
//     double* c,
//     size_t ldc
// );
//
// Caller (gemm.rs::dgemm) only invokes this when nb == 8 * SVL_D.

.globl SYM(sonium_dgemm_panel8_sme2)
.p2align 4
SYM(sonium_dgemm_panel8_sme2):
    stp     d8, d9, [sp, #-64]!
    stp     d10, d11, [sp, #16]
    stp     d12, d13, [sp, #32]
    stp     d14, d15, [sp, #48]
    stp     x19, x20, [sp, #-32]!
    stp     x21, x22, [sp, #16]

    fmov    x19, d0                 // alpha bits in GPR
    fmov    x20, d1                 // beta bits
    mov     x21, x6                 // c
    mov     x22, x4                 // b

    lsl     x5, x5, #3              // ldb_bytes = ldb * 8
    lsl     x7, x7, #3              // ldc_bytes = ldc * 8

    smstart sm
    smstart za

    ptrue   p0.d
    // PN counter for the 4-Z grouped multi-vector load (32 f64 lanes per group).
    ptrue   pn8.d

    zero    {za}

    mov     x10, #0
    mov     x11, x3                 // packed_a cursor
    mov     x12, x22                // b cursor
    rdsvl   x13, #1                 // SVL in bytes (= 64 at SVL=512)
.Ldg8_k:
    cmp     x10, x2
    b.ge    .Ldg8_kdone

    // Prefetch A column and BOTH halves of the B panel two iterations ahead.
    // At SVL=512: A column = 64 bytes (x13); B row = 8*SVL = 512 bytes (x5).
    add     x8, x11, x13, lsl #1    // packed_a + 2*SVL bytes
    prfm    pldl1keep, [x8]
    add     x9, x12, x5, lsl #1     // b + 2*ldb_bytes (first 256 B)
    prfm    pldl1keep, [x9]
    add     x9, x9, #256            // + 256 bytes (second half of same row)
    prfm    pldl1keep, [x9]

    // Single A column shared across all 8 FMOPAs.
    ld1d    {z0.d}, p0/z, [x11]

    // Two grouped 4-vector loads cover 8*SVL_D = 64 contiguous f64 lanes.
    ld1d    {z4.d-z7.d}, pn8/z, [x12]
    ld1d    {z8.d-z11.d}, pn8/z, [x12, #4, mul vl]

    fmopa   za0.d, p0/m, p0/m, z0.d, z4.d
    fmopa   za1.d, p0/m, p0/m, z0.d, z5.d
    fmopa   za2.d, p0/m, p0/m, z0.d, z6.d
    fmopa   za3.d, p0/m, p0/m, z0.d, z7.d
    fmopa   za4.d, p0/m, p0/m, z0.d, z8.d
    fmopa   za5.d, p0/m, p0/m, z0.d, z9.d
    fmopa   za6.d, p0/m, p0/m, z0.d, z10.d
    fmopa   za7.d, p0/m, p0/m, z0.d, z11.d

    add     x11, x11, x13           // packed_a += SVL bytes
    add     x12, x12, x5            // b += ldb_bytes
    add     x10, x10, #1
    b       .Ldg8_k

.Ldg8_kdone:
    dup     z30.d, x19              // bcast alpha across SVL_D lanes
    dup     z31.d, x20              // bcast beta

    mov     w14, #0
    mov     x15, x21
.Ldg8_epi:
    cmp     w14, w0
    b.ge    .Ldg8_epi_done

    mov     w12, w14

    mova    z2.d, p0/m, za0h.d[w12, 0]
    fmul    z2.d, p0/m, z2.d, z30.d
    ld1d    {z3.d}, p0/z, [x15]
    fmla    z2.d, p0/m, z3.d, z31.d
    st1d    {z2.d}, p0, [x15]

    mova    z2.d, p0/m, za1h.d[w12, 0]
    fmul    z2.d, p0/m, z2.d, z30.d
    ld1d    {z3.d}, p0/z, [x15, #1, mul vl]
    fmla    z2.d, p0/m, z3.d, z31.d
    st1d    {z2.d}, p0, [x15, #1, mul vl]

    mova    z2.d, p0/m, za2h.d[w12, 0]
    fmul    z2.d, p0/m, z2.d, z30.d
    ld1d    {z3.d}, p0/z, [x15, #2, mul vl]
    fmla    z2.d, p0/m, z3.d, z31.d
    st1d    {z2.d}, p0, [x15, #2, mul vl]

    mova    z2.d, p0/m, za3h.d[w12, 0]
    fmul    z2.d, p0/m, z2.d, z30.d
    ld1d    {z3.d}, p0/z, [x15, #3, mul vl]
    fmla    z2.d, p0/m, z3.d, z31.d
    st1d    {z2.d}, p0, [x15, #3, mul vl]

    mova    z2.d, p0/m, za4h.d[w12, 0]
    fmul    z2.d, p0/m, z2.d, z30.d
    ld1d    {z3.d}, p0/z, [x15, #4, mul vl]
    fmla    z2.d, p0/m, z3.d, z31.d
    st1d    {z2.d}, p0, [x15, #4, mul vl]

    mova    z2.d, p0/m, za5h.d[w12, 0]
    fmul    z2.d, p0/m, z2.d, z30.d
    ld1d    {z3.d}, p0/z, [x15, #5, mul vl]
    fmla    z2.d, p0/m, z3.d, z31.d
    st1d    {z2.d}, p0, [x15, #5, mul vl]

    mova    z2.d, p0/m, za6h.d[w12, 0]
    fmul    z2.d, p0/m, z2.d, z30.d
    ld1d    {z3.d}, p0/z, [x15, #6, mul vl]
    fmla    z2.d, p0/m, z3.d, z31.d
    st1d    {z2.d}, p0, [x15, #6, mul vl]

    mova    z2.d, p0/m, za7h.d[w12, 0]
    fmul    z2.d, p0/m, z2.d, z30.d
    ld1d    {z3.d}, p0/z, [x15, #7, mul vl]
    fmla    z2.d, p0/m, z3.d, z31.d
    st1d    {z2.d}, p0, [x15, #7, mul vl]

    add     w14, w14, #1
    add     x15, x15, x7
    b       .Ldg8_epi

.Ldg8_epi_done:
    smstop  za
    smstop  sm

    ldp     x21, x22, [sp, #16]
    ldp     x19, x20, [sp], #32
    ldp     d14, d15, [sp, #48]
    ldp     d12, d13, [sp, #32]
    ldp     d10, d11, [sp, #16]
    ldp     d8, d9, [sp], #64
    ret


// ============================================================================
// BFGEMM single-tile micro-kernel: BFMOPA into ZA0.S (16x16 f32 tile at SVL=512)
// ============================================================================
//
// *** Currently unused. ***  Apple M4 Pro emits the BFMOPA opcode but
// executes it as a single-lane outer product (the K-second element of each
// pair is ignored), so this kernel produces incorrect results on that CPU.
// We instead widen bf16 → f32 at pack time and dispatch through the f32
// pipeline — see `gemm_bf16.rs`. This kernel is preserved so the Rust
// dispatch can switch back over once Apple silicon supports pair-widening.
//
// void sonium_bfgemm_panel_sme2(
//     size_t mb, size_t nb,         // 1..=SVL_S each
//     size_t k_pairs,               // K/2 (caller pads K to even with a zero pair)
//     float alpha,                  // s0
//     const __bf16* packed_a,       // (SVL_S*2) * k_pairs bf16 elements, row-pair layout:
//                                   //   packed_a[pair*MR*2 + i*2 + 0] = A[i0+i, 2*pair+0]
//                                   //   packed_a[pair*MR*2 + i*2 + 1] = A[i0+i, 2*pair+1]
//                                   // Tail rows i in mb..SVL_S are zero-padded.
//     const __bf16* packed_b,       // (SVL_S*2) * k_pairs, col-pair layout:
//                                   //   packed_b[pair*NR*2 + j*2 + 0] = B[2*pair+0, j0+j]
//                                   //   packed_b[pair*NR*2 + j*2 + 1] = B[2*pair+1, j0+j]
//                                   // Tail cols j in nb..SVL_S are zero-padded.
//     float beta,                   // s1
//     float* c,                     // &C[i0, j0]  (f32 output)
//     size_t ldc                    // elements
// );
//
// AAPCS64: mb=x0, nb=x1, k_pairs=x2, alpha=s0, packed_a=x3, packed_b=x4,
//          beta=s1, c=x5, ldc=x6
//
// Per inner k_pairs step:
//   z0.h = packed_a[pair*64..(pair+1)*64]  (32 bf16 lanes = 16 row-pairs)
//   z1.h = packed_b[pair*64..(pair+1)*64]  (32 bf16 lanes = 16 col-pairs)
//   ZA0.S += <bf16 pair-dot outer product>
// That is, ZA0.S[i,j] += A[i,2p]*B[2p,j] + A[i,2p+1]*B[2p+1,j].  ONE BFMOPA
// = 256 f32 outputs × 2 (pair-dot) = 512 effective FMAs / instruction.
//
// Both A and B are pre-padded by Rust so we use ptrue throughout the inner
// loop; mb/nb masking applies only in the C epilogue (via `cmp w14, w0`).

.globl SYM(sonium_bfgemm_panel_sme2)
.p2align 4
SYM(sonium_bfgemm_panel_sme2):
    stp     d8, d9, [sp, #-64]!
    stp     d10, d11, [sp, #16]
    stp     d12, d13, [sp, #32]
    stp     d14, d15, [sp, #48]
    stp     x19, x20, [sp, #-32]!
    stp     x21, x22, [sp, #16]

    fmov    w19, s0                 // alpha bits (f32)
    fmov    w20, s1                 // beta bits  (f32)
    mov     x21, x5                 // c
    mov     x22, x4                 // packed_b

    lsl     x6, x6, #2              // ldc_bytes (f32)

    smstart sm
    smstart za

    ptrue   p0.s                    // ZA tile row predicate (f32 element type)
    ptrue   p1.h                    // .H predicate for the BFMOPA source loads

    zero    {za}

    // k_pairs loop. Stride = SVL bytes (= 32 bf16 = 16 row/col pairs).
    mov     x10, #0                 // pair index
    mov     x11, x3                 // packed_a cursor
    mov     x12, x22                // packed_b cursor
    rdsvl   x13, #1                 // SVL bytes = 64
.Lbf_k:
    cmp     x10, x2
    b.ge    .Lbf_kdone

    ld1h    {z0.h}, p1/z, [x11]
    ld1h    {z1.h}, p1/z, [x12]

    bfmopa  za0.s, p0/m, p0/m, z0.h, z1.h

    add     x11, x11, x13
    add     x12, x12, x13
    add     x10, x10, #1
    b       .Lbf_k

.Lbf_kdone:
    // Broadcast alpha / beta (f32).
    dup     z30.s, w19
    dup     z31.s, w20

    // Set up an nb-bounded col predicate for the epilogue store.
    mov     x10, #0
    whilelt p2.s, x10, x1

    mov     w14, #0                 // r (epilogue row counter)
    mov     x15, x21                // c cursor
.Lbf_epi:
    cmp     w14, w0                 // r < mb ?
    b.ge    .Lbf_epi_done

    mov     w12, w14
    mova    z2.s, p0/m, za0h.s[w12, 0]

    fmul    z2.s, p2/m, z2.s, z30.s
    ld1w    {z3.s}, p2/z, [x15]
    fmla    z2.s, p2/m, z3.s, z31.s
    st1w    {z2.s}, p2, [x15]

    add     w14, w14, #1
    add     x15, x15, x6            // ldc_bytes
    b       .Lbf_epi

.Lbf_epi_done:
    smstop  za
    smstop  sm

    ldp     x21, x22, [sp, #16]
    ldp     x19, x20, [sp], #32
    ldp     d14, d15, [sp, #48]
    ldp     d12, d13, [sp, #32]
    ldp     d10, d11, [sp, #16]
    ldp     d8, d9, [sp], #64
    ret


// SVL is read once at startup via `sonium_read_svl_bytes` (above) and
// stored in `Sme2Caps::svl_bits`. No other SVL readers are needed.


// ============================================================================
// Streaming-SVE BLAS-1 kernels (axpy, scale, dot)
// ============================================================================
//
// SVE-512 in streaming mode gives 4× the per-instruction throughput of
// NEON-128. The wall-clock win on bandwidth-bound BLAS-1 is bounded by
// memory bandwidth (~50 GB/s on M4 P-cluster), so effectively ~1.5-2× over
// NEON when the working set spills L2. Below threshold (~4096 elements)
// the per-call SMSTART/SMSTOP overhead (~50 ns) dominates and NEON wins;
// the Rust dispatcher picks the right path.
//
// Pattern follows the SGEMV streaming-SVE kernel: 4-Z grouped multi-vector
// LD1{W,D} (paper's ~900 GB/s path) + 4 independent FMA accumulators to
// break the dep chain.


// ----------------------------------------------------------------------------
// AXPY: y[i] += alpha * x[i]
// ----------------------------------------------------------------------------
//
// void sonium_axpy_f32_sme2(float alpha, const float* x, float* y, size_t n)
//
// ABI: alpha=s0, x=x0, y=x1, n=x2.

.globl SYM(sonium_axpy_f32_sme2)
.p2align 4
SYM(sonium_axpy_f32_sme2):
    stp     d8, d9,   [sp, #-64]!
    stp     d10, d11, [sp, #16]
    stp     d12, d13, [sp, #32]
    stp     d14, d15, [sp, #48]
    stp     x19, x20, [sp, #-16]!

    // Stash alpha bits in callee-saved GPR before SMSTART zeros NEON.
    fmov    w19, s0

    smstart sm

    ptrue   pn8.s
    ptrue   p0.s

    rdsvl   x9, #1             // SVL bytes
    lsr     x10, x9, #2        // SVL_S elements
    lsl     x11, x9, #2        // 4*SVL bytes
    lsl     x12, x10, #2       // 4*SVL_S elements

    dup     z31.s, w19         // broadcast alpha

    mov     x3, #0
.Laxpy_s_4x:
    add     x4, x3, x12
    cmp     x4, x2
    b.gt    .Laxpy_s_tail

    ld1w    {z0.s - z3.s}, pn8/z, [x0]
    ld1w    {z4.s - z7.s}, pn8/z, [x1]
    fmla    z4.s, p0/m, z0.s, z31.s
    fmla    z5.s, p0/m, z1.s, z31.s
    fmla    z6.s, p0/m, z2.s, z31.s
    fmla    z7.s, p0/m, z3.s, z31.s
    st1w    {z4.s - z7.s}, pn8, [x1]

    add     x0, x0, x11
    add     x1, x1, x11
    add     x3, x3, x12
    b       .Laxpy_s_4x

.Laxpy_s_tail:
    cmp     x3, x2
    b.ge    .Laxpy_s_done

    whilelt p1.s, x3, x2
    ld1w    {z0.s}, p1/z, [x0]
    ld1w    {z4.s}, p1/z, [x1]
    fmla    z4.s, p1/m, z0.s, z31.s
    st1w    {z4.s}, p1, [x1]

    add     x0, x0, x9
    add     x1, x1, x9
    add     x3, x3, x10
    b       .Laxpy_s_tail

.Laxpy_s_done:
    smstop  sm

    ldp     x19, x20, [sp], #16
    ldp     d14, d15, [sp, #48]
    ldp     d12, d13, [sp, #32]
    ldp     d10, d11, [sp, #16]
    ldp     d8, d9,   [sp], #64
    ret


// void sonium_axpy_f64_sme2(double alpha, const double* x, double* y, size_t n)
//
// ABI: alpha=d0, x=x0, y=x1, n=x2.

.globl SYM(sonium_axpy_f64_sme2)
.p2align 4
SYM(sonium_axpy_f64_sme2):
    stp     d8, d9,   [sp, #-64]!
    stp     d10, d11, [sp, #16]
    stp     d12, d13, [sp, #32]
    stp     d14, d15, [sp, #48]
    stp     x19, x20, [sp, #-16]!

    fmov    x19, d0

    smstart sm

    ptrue   pn8.d
    ptrue   p0.d

    rdsvl   x9, #1             // SVL bytes
    lsr     x10, x9, #3        // SVL_D elements
    lsl     x11, x9, #2        // 4*SVL bytes
    lsl     x12, x10, #2       // 4*SVL_D elements

    dup     z31.d, x19         // broadcast alpha (f64 bits)

    mov     x3, #0
.Laxpy_d_4x:
    add     x4, x3, x12
    cmp     x4, x2
    b.gt    .Laxpy_d_tail

    ld1d    {z0.d - z3.d}, pn8/z, [x0]
    ld1d    {z4.d - z7.d}, pn8/z, [x1]
    fmla    z4.d, p0/m, z0.d, z31.d
    fmla    z5.d, p0/m, z1.d, z31.d
    fmla    z6.d, p0/m, z2.d, z31.d
    fmla    z7.d, p0/m, z3.d, z31.d
    st1d    {z4.d - z7.d}, pn8, [x1]

    add     x0, x0, x11
    add     x1, x1, x11
    add     x3, x3, x12
    b       .Laxpy_d_4x

.Laxpy_d_tail:
    cmp     x3, x2
    b.ge    .Laxpy_d_done

    whilelt p1.d, x3, x2
    ld1d    {z0.d}, p1/z, [x0]
    ld1d    {z4.d}, p1/z, [x1]
    fmla    z4.d, p1/m, z0.d, z31.d
    st1d    {z4.d}, p1, [x1]

    add     x0, x0, x9
    add     x1, x1, x9
    add     x3, x3, x10
    b       .Laxpy_d_tail

.Laxpy_d_done:
    smstop  sm

    ldp     x19, x20, [sp], #16
    ldp     d14, d15, [sp, #48]
    ldp     d12, d13, [sp, #32]
    ldp     d10, d11, [sp, #16]
    ldp     d8, d9,   [sp], #64
    ret


// ----------------------------------------------------------------------------
// SCALE: x[i] *= alpha
// ----------------------------------------------------------------------------
//
// void sonium_scal_f32_sme2(float alpha, float* x, size_t n)
//
// ABI: alpha=s0, x=x0, n=x1.

.globl SYM(sonium_scal_f32_sme2)
.p2align 4
SYM(sonium_scal_f32_sme2):
    stp     d8, d9,   [sp, #-64]!
    stp     d10, d11, [sp, #16]
    stp     d12, d13, [sp, #32]
    stp     d14, d15, [sp, #48]
    stp     x19, x20, [sp, #-16]!

    fmov    w19, s0

    smstart sm

    ptrue   pn8.s
    ptrue   p0.s

    rdsvl   x9, #1
    lsr     x10, x9, #2
    lsl     x11, x9, #2
    lsl     x12, x10, #2

    dup     z31.s, w19

    mov     x3, #0
.Lscal_s_4x:
    add     x4, x3, x12
    cmp     x4, x1
    b.gt    .Lscal_s_tail

    ld1w    {z4.s - z7.s}, pn8/z, [x0]
    fmul    z4.s, p0/m, z4.s, z31.s
    fmul    z5.s, p0/m, z5.s, z31.s
    fmul    z6.s, p0/m, z6.s, z31.s
    fmul    z7.s, p0/m, z7.s, z31.s
    st1w    {z4.s - z7.s}, pn8, [x0]

    add     x0, x0, x11
    add     x3, x3, x12
    b       .Lscal_s_4x

.Lscal_s_tail:
    cmp     x3, x1
    b.ge    .Lscal_s_done

    whilelt p1.s, x3, x1
    ld1w    {z4.s}, p1/z, [x0]
    fmul    z4.s, p1/m, z4.s, z31.s
    st1w    {z4.s}, p1, [x0]

    add     x0, x0, x9
    add     x3, x3, x10
    b       .Lscal_s_tail

.Lscal_s_done:
    smstop  sm

    ldp     x19, x20, [sp], #16
    ldp     d14, d15, [sp, #48]
    ldp     d12, d13, [sp, #32]
    ldp     d10, d11, [sp, #16]
    ldp     d8, d9,   [sp], #64
    ret


// void sonium_scal_f64_sme2(double alpha, double* x, size_t n)
//
// ABI: alpha=d0, x=x0, n=x1.

.globl SYM(sonium_scal_f64_sme2)
.p2align 4
SYM(sonium_scal_f64_sme2):
    stp     d8, d9,   [sp, #-64]!
    stp     d10, d11, [sp, #16]
    stp     d12, d13, [sp, #32]
    stp     d14, d15, [sp, #48]
    stp     x19, x20, [sp, #-16]!

    fmov    x19, d0

    smstart sm

    ptrue   pn8.d
    ptrue   p0.d

    rdsvl   x9, #1
    lsr     x10, x9, #3
    lsl     x11, x9, #2
    lsl     x12, x10, #2

    dup     z31.d, x19

    mov     x3, #0
.Lscal_d_4x:
    add     x4, x3, x12
    cmp     x4, x1
    b.gt    .Lscal_d_tail

    ld1d    {z4.d - z7.d}, pn8/z, [x0]
    fmul    z4.d, p0/m, z4.d, z31.d
    fmul    z5.d, p0/m, z5.d, z31.d
    fmul    z6.d, p0/m, z6.d, z31.d
    fmul    z7.d, p0/m, z7.d, z31.d
    st1d    {z4.d - z7.d}, pn8, [x0]

    add     x0, x0, x11
    add     x3, x3, x12
    b       .Lscal_d_4x

.Lscal_d_tail:
    cmp     x3, x1
    b.ge    .Lscal_d_done

    whilelt p1.d, x3, x1
    ld1d    {z4.d}, p1/z, [x0]
    fmul    z4.d, p1/m, z4.d, z31.d
    st1d    {z4.d}, p1, [x0]

    add     x0, x0, x9
    add     x3, x3, x10
    b       .Lscal_d_tail

.Lscal_d_done:
    smstop  sm

    ldp     x19, x20, [sp], #16
    ldp     d14, d15, [sp, #48]
    ldp     d12, d13, [sp, #32]
    ldp     d10, d11, [sp, #16]
    ldp     d8, d9,   [sp], #64
    ret


// ----------------------------------------------------------------------------
// DOT: result = sum_i x[i] * y[i]
// ----------------------------------------------------------------------------
//
// float sonium_dot_f32_sme2(const float* x, const float* y, size_t n)
//
// ABI: x=x0, y=x1, n=x2, result in s0.

.globl SYM(sonium_dot_f32_sme2)
.p2align 4
SYM(sonium_dot_f32_sme2):
    stp     d8, d9,   [sp, #-64]!
    stp     d10, d11, [sp, #16]
    stp     d12, d13, [sp, #32]
    stp     d14, d15, [sp, #48]

    smstart sm

    ptrue   pn8.s
    ptrue   p0.s

    rdsvl   x9, #1
    lsr     x10, x9, #2
    lsl     x11, x9, #2
    lsl     x12, x10, #2

    // 4 independent accumulators break the FMA dep chain.
    mov     z24.s, #0
    mov     z25.s, #0
    mov     z26.s, #0
    mov     z27.s, #0

    mov     x3, #0
.Ldot_s_4x:
    add     x4, x3, x12
    cmp     x4, x2
    b.gt    .Ldot_s_tail

    ld1w    {z0.s - z3.s}, pn8/z, [x0]
    ld1w    {z4.s - z7.s}, pn8/z, [x1]
    fmla    z24.s, p0/m, z0.s, z4.s
    fmla    z25.s, p0/m, z1.s, z5.s
    fmla    z26.s, p0/m, z2.s, z6.s
    fmla    z27.s, p0/m, z3.s, z7.s

    add     x0, x0, x11
    add     x1, x1, x11
    add     x3, x3, x12
    b       .Ldot_s_4x

.Ldot_s_tail:
    // Fold 4 accumulators down to z29 once.
    fadd    z24.s, z24.s, z25.s
    fadd    z26.s, z26.s, z27.s
    fadd    z29.s, z24.s, z26.s

.Ldot_s_tail_loop:
    cmp     x3, x2
    b.ge    .Ldot_s_done

    whilelt p1.s, x3, x2
    ld1w    {z0.s}, p1/z, [x0]
    ld1w    {z1.s}, p1/z, [x1]
    fmla    z29.s, p1/m, z0.s, z1.s

    add     x0, x0, x9
    add     x1, x1, x9
    add     x3, x3, x10
    b       .Ldot_s_tail_loop

.Ldot_s_done:
    // Stash the scalar result through a GPR before SMSTOP — SMSTOP SM
    // zeroes all SVE/NEON registers including s0 (which aliases Z0[0..31]).
    faddv   s0, p0, z29.s
    fmov    w0, s0
    smstop  sm
    fmov    s0, w0

    ldp     d14, d15, [sp, #48]
    ldp     d12, d13, [sp, #32]
    ldp     d10, d11, [sp, #16]
    ldp     d8, d9,   [sp], #64
    ret


// double sonium_dot_f64_sme2(const double* x, const double* y, size_t n)
//
// ABI: x=x0, y=x1, n=x2, result in d0.

.globl SYM(sonium_dot_f64_sme2)
.p2align 4
SYM(sonium_dot_f64_sme2):
    stp     d8, d9,   [sp, #-64]!
    stp     d10, d11, [sp, #16]
    stp     d12, d13, [sp, #32]
    stp     d14, d15, [sp, #48]

    smstart sm

    ptrue   pn8.d
    ptrue   p0.d

    rdsvl   x9, #1
    lsr     x10, x9, #3
    lsl     x11, x9, #2
    lsl     x12, x10, #2

    mov     z24.d, #0
    mov     z25.d, #0
    mov     z26.d, #0
    mov     z27.d, #0

    mov     x3, #0
.Ldot_d_4x:
    add     x4, x3, x12
    cmp     x4, x2
    b.gt    .Ldot_d_tail

    ld1d    {z0.d - z3.d}, pn8/z, [x0]
    ld1d    {z4.d - z7.d}, pn8/z, [x1]
    fmla    z24.d, p0/m, z0.d, z4.d
    fmla    z25.d, p0/m, z1.d, z5.d
    fmla    z26.d, p0/m, z2.d, z6.d
    fmla    z27.d, p0/m, z3.d, z7.d

    add     x0, x0, x11
    add     x1, x1, x11
    add     x3, x3, x12
    b       .Ldot_d_4x

.Ldot_d_tail:
    fadd    z24.d, z24.d, z25.d
    fadd    z26.d, z26.d, z27.d
    fadd    z29.d, z24.d, z26.d

.Ldot_d_tail_loop:
    cmp     x3, x2
    b.ge    .Ldot_d_done

    whilelt p1.d, x3, x2
    ld1d    {z0.d}, p1/z, [x0]
    ld1d    {z1.d}, p1/z, [x1]
    fmla    z29.d, p1/m, z0.d, z1.d

    add     x0, x0, x9
    add     x1, x1, x9
    add     x3, x3, x10
    b       .Ldot_d_tail_loop

.Ldot_d_done:
    // Stash through GPR before SMSTOP (see f32 sibling for rationale).
    faddv   d0, p0, z29.d
    fmov    x0, d0
    smstop  sm
    fmov    d0, x0

    ldp     d14, d15, [sp, #48]
    ldp     d12, d13, [sp, #32]
    ldp     d10, d11, [sp, #16]
    ldp     d8, d9,   [sp], #64
    ret
