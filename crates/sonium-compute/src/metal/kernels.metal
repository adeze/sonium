// MSL kernels for sonium-compute Metal backend.
//
// f32 / Complex32 only — Apple Silicon Metal compiler rejects `double`.
// The sonium-solvers GMRES-IR path uses these as the inner Krylov:
// f32 path for `f64`-outer Helmholtz/elasticity; Complex32 path for
// `Complex64`-outer Helmholtz BEM. The outer (f64 / Complex64) loop
// runs host-staged.
//
// Complex32 is packed as MSL `float2` with `.x = re`, `.y = im`:
//
//   (a + bi)(c + di) = (ac − bd) + (ad + bc)i        — multiplication
//   conj(a + bi)(c + di) = (ac + bd) + (ad − bc)i    — Hermitian dot
//   |a + bi|² = a² + b²                              — squared modulus
//
// Memory layout: `num_complex::Complex<f32>` is `#[repr(C)]` with
// `re: f32, im: f32`, so an array of `Complex32` is binary-identical
// to an array of `float2` (both 8-byte stride, re-then-im).
//
// Naming: `sonium_*` prefix to avoid colliding with anything else
// loaded in the same Metal library namespace.

#include <metal_stdlib>
using namespace metal;

// Complex multiplication helpers.
inline float2 cmul(float2 a, float2 b) {
    // (a.re + i·a.im) · (b.re + i·b.im)
    return float2(a.x * b.x - a.y * b.y,
                  a.x * b.y + a.y * b.x);
}

inline float2 cmul_conj_a(float2 a, float2 b) {
    // conj(a) · b = (a.re − i·a.im)·(b.re + i·b.im)
    return float2(a.x * b.x + a.y * b.y,
                  a.x * b.y - a.y * b.x);
}

inline float cnorm_sqr(float2 a) {
    return a.x * a.x + a.y * a.y;
}

// One radix-2 Stockham stage for a contiguous batch of power-of-two FFTs.
// Each stage reads the two inputs at a fixed N/2 stride and writes contiguous
// output groups. The Rust dispatcher ping-pongs src/dst, so no bit-reversal
// permutation or host round-trip is required. `m` is the current butterfly
// width; `sign` is -1 for forward and +1 for inverse.
kernel void sonium_fft_stockham_radix2_stage_c32(
    device const float2* src       [[buffer(0)]],
    device float2*       dst       [[buffer(1)]],
    constant uint&       fft_len   [[buffer(2)]],
    constant uint&       batch     [[buffer(3)]],
    constant uint&       m         [[buffer(4)]],
    constant float&      sign      [[buffer(5)]],
    uint                  gid      [[thread_position_in_grid]]
) {
    uint pair_count = (fft_len * batch) / 2;
    if (gid >= pair_count) { return; }

    uint butterflies_per_batch = fft_len / 2;
    uint batch_index = gid / butterflies_per_batch;
    uint batch_base = batch_index * fft_len;
    uint pair = gid - batch_index * butterflies_per_batch;
    uint group = pair / m;
    uint lane = pair - group * m;
    uint base_in = batch_base + group * m + lane;
    uint base_out = batch_base + group * (m * 2) + lane;
    uint stride = fft_len / 2;
    float angle = sign * 6.2831853071795864769f
        * float(lane) / float(m * 2);
    float2 twiddle = float2(cos(angle), sin(angle));
    float2 a = src[base_in];
    float2 b = cmul(src[base_in + stride], twiddle);
    dst[base_out] = a + b;
    dst[base_out + m] = a - b;
}

// One radix-4 Stockham stage for a contiguous batch of power-of-two FFTs.
// The four inputs are separated by N/4 in the stage input layout and the
// four outputs are packed contiguously in the stage output layout. `m` is the
// current sub-FFT width; the stage twiddles are W_N^(k*l) for k = 1, 2, 3.
kernel void sonium_fft_stockham_radix4_stage_c32(
    device const float2* src       [[buffer(0)]],
    device float2*       dst       [[buffer(1)]],
    constant uint&       fft_len   [[buffer(2)]],
    constant uint&       batch     [[buffer(3)]],
    constant uint&       m         [[buffer(4)]],
    constant float&      sign      [[buffer(5)]],
    uint                  gid       [[thread_position_in_grid]]
) {
    uint butterfly_count = (fft_len * batch) / 4;
    if (gid >= butterfly_count) { return; }

    uint butterflies_per_batch = fft_len / 4;
    uint batch_index = gid / butterflies_per_batch;
    uint batch_base = batch_index * fft_len;
    uint butterfly = gid - batch_index * butterflies_per_batch;
    uint group = butterfly / m;
    uint lane = butterfly - group * m;
    uint base_in = batch_base + group * m + lane;
    uint base_out = batch_base + group * (m * 4) + lane;
    uint stride = fft_len / 4;

    float base_angle = sign * 6.2831853071795864769f
        * float(lane) / float(m * 4);
    float2 tw1 = float2(cos(base_angle), sin(base_angle));
    float2 tw2 = float2(cos(2.0f * base_angle), sin(2.0f * base_angle));
    float2 tw3 = float2(cos(3.0f * base_angle), sin(3.0f * base_angle));

    float2 x0 = src[base_in];
    float2 x1 = cmul(src[base_in + stride], tw1);
    float2 x2 = cmul(src[base_in + 2 * stride], tw2);
    float2 x3 = cmul(src[base_in + 3 * stride], tw3);

    float2 y0 = x0 + x1 + x2 + x3;
    float2 y2 = x0 - x1 + x2 - x3;
    float2 a = x0 - x2;
    float2 b = x1 - x3;
    // sign·i·b: forward (sign = -1) uses -i, inverse uses +i.
    float2 sign_i_b = float2(-sign * b.y, sign * b.x);
    float2 y1 = a + sign_i_b;
    float2 y3 = a - sign_i_b;

    dst[base_out] = y0;
    dst[base_out + m] = y1;
    dst[base_out + 2 * m] = y2;
    dst[base_out + 3 * m] = y3;
}

// ---------------------------------------------------------------------------
// BLAS-1 — element-wise kernels
// ---------------------------------------------------------------------------

// y[i] = alpha * x[i] + y[i]
kernel void sonium_axpy_f32(
    device const float*  x      [[buffer(0)]],
    device float*        y      [[buffer(1)]],
    constant float&      alpha  [[buffer(2)]],
    constant uint&       n      [[buffer(3)]],
    uint                 gid    [[thread_position_in_grid]]
) {
    if (gid >= n) { return; }
    y[gid] = fma(alpha, x[gid], y[gid]);
}

// y[i] += sum_j alphas[j] * x_basis[j][i], where x_basis is packed as
// contiguous row-major vectors: x_basis[j * n + i].
kernel void sonium_axpy_many_f32(
    constant uint& basis_offset [[buffer(5)]],
    constant uint& basis_row_stride [[buffer(6)]],
    constant uint& basis_col_stride [[buffer(7)]],
    constant uint& y_offset [[buffer(8)]],
    constant uint& y_stride [[buffer(9)]],
    device const float*  x_basis [[buffer(0)]],
    device const float*  alphas  [[buffer(1)]],
    device float*        y       [[buffer(2)]],
    constant uint&       n       [[buffer(3)]],
    constant uint&       k       [[buffer(4)]],
    uint                 gid     [[thread_position_in_grid]]
) {
    if (gid >= n) { return; }
    float acc = y[y_offset + gid * y_stride];
    for (uint basis = 0; basis < k; basis++) {
        acc = fma(
            alphas[basis],
            x_basis[basis_offset + basis * basis_row_stride + gid * basis_col_stride],
            acc);
    }
    y[y_offset + gid * y_stride] = acc;
}

// x[i] = alpha * x[i]
kernel void sonium_scale_f32(
    device float*        x      [[buffer(0)]],
    constant float&      alpha  [[buffer(1)]],
    constant uint&       n      [[buffer(2)]],
    uint                 gid    [[thread_position_in_grid]]
) {
    if (gid >= n) { return; }
    x[gid] *= alpha;
}

// ---------------------------------------------------------------------------
// Reductions — produce one partial per threadgroup. Host sums the partials.
// Threadgroup shared memory at index 0: float[REDUCTION_THREADGROUP_SIZE].
// ---------------------------------------------------------------------------

// partials[tg] = sum over thread group of x[i] * y[i]
kernel void sonium_dot_f32(
    device const float*  x        [[buffer(0)]],
    device const float*  y        [[buffer(1)]],
    device float*        partials [[buffer(2)]],
    constant uint&       n        [[buffer(3)]],
    threadgroup float*   shared   [[threadgroup(0)]],
    uint                 gid      [[thread_position_in_grid]],
    uint                 lid      [[thread_position_in_threadgroup]],
    uint                 tg_id    [[threadgroup_position_in_grid]],
    uint                 tg_size  [[threads_per_threadgroup]]
) {
    float v = (gid < n) ? (x[gid] * y[gid]) : 0.0f;
    shared[lid] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Tree reduction in shared memory.
    for (uint stride = tg_size / 2; stride > 0; stride >>= 1) {
        if (lid < stride) {
            shared[lid] += shared[lid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (lid == 0) {
        partials[tg_id] = shared[0];
    }
}

// partials[tg] = sum over thread group of x[i] * x[i]
//
// Argument layout matches `sonium_dot_f32` for binding-site uniformity
// (same Rust dispatcher binds slots 2–3); slot 1 is unused but must be
// validly bound. Caller passes the same buffer at slots 0 and 1 so the
// kernel's `x[i] * y[i]` would equal `x[i]²`. We compute that directly
// to keep the per-thread cost identical and avoid relying on aliasing.
kernel void sonium_nrm2_sq_f32(
    device const float*  x         [[buffer(0)]],
    device const float*  x_alias   [[buffer(1)]],
    device float*        partials  [[buffer(2)]],
    constant uint&       n         [[buffer(3)]],
    threadgroup float*   shared    [[threadgroup(0)]],
    uint                 gid       [[thread_position_in_grid]],
    uint                 lid       [[thread_position_in_threadgroup]],
    uint                 tg_id     [[threadgroup_position_in_grid]],
    uint                 tg_size   [[threads_per_threadgroup]]
) {
    (void)x_alias;
    float v = (gid < n) ? (x[gid] * x[gid]) : 0.0f;
    shared[lid] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = tg_size / 2; stride > 0; stride >>= 1) {
        if (lid < stride) {
            shared[lid] += shared[lid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (lid == 0) {
        partials[tg_id] = shared[0];
    }
}

// Fused multi-dot: `partials[b * num_chunks + chunk] = sum over chunk of
// basis_concat[b * n + i] * y[i]`. Caller packs the basis vectors into
// a contiguous buffer of length `num_basis * n` (one row per basis
// vector) and dispatches a 2-D grid `(num_chunks, num_basis)` with
// threadgroup `(THREADGROUP_SIZE, 1)`.
//
// Used by the Backend's `multi_dot` to collapse `j+1` GMRES Gram-Schmidt
// dot products into a single dispatch with a single host-sync. Saves
// `j × ~100 µs` per inner-iter on Apple Silicon (no per-dot wait_until_completed).
kernel void sonium_multi_dot_f32(
    // descriptor offsets and strides are bound after the logical dimensions
    constant uint& basis_offset [[buffer(5)]],
    constant uint& basis_row_stride [[buffer(6)]],
    constant uint& basis_col_stride [[buffer(7)]],
    constant uint& y_offset [[buffer(8)]],
    constant uint& y_stride [[buffer(9)]],
    device const float*  basis_concat  [[buffer(0)]],
    device const float*  y             [[buffer(1)]],
    device float*        partials      [[buffer(2)]],
    constant uint&       n             [[buffer(3)]],
    constant uint&       num_chunks    [[buffer(4)]],
    threadgroup float*   shared        [[threadgroup(0)]],
    uint2                gid           [[thread_position_in_grid]],
    uint2                lid           [[thread_position_in_threadgroup]],
    uint2                tg_id         [[threadgroup_position_in_grid]],
    uint2                tg_size       [[threads_per_threadgroup]]
) {
    uint i = gid.x;          // index in n
    uint b = gid.y;          // basis index
    uint chunk = tg_id.x;    // chunk along n
    uint local = lid.x;      // 1-D thread index inside the threadgroup
    uint group = tg_size.x;  // threads per threadgroup along n

    float v = (i < n)
        ? (basis_concat[basis_offset + b * basis_row_stride + i * basis_col_stride]
            * y[y_offset + i * y_stride])
        : 0.0f;
    shared[local] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = group / 2; stride > 0; stride >>= 1) {
        if (local < stride) {
            shared[local] += shared[local + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (local == 0) {
        partials[b * num_chunks + chunk] = shared[0];
    }
}

// ---------------------------------------------------------------------------
// Final reduction kernels — collapse partials buffer to a single scalar
// in a single threadgroup. Saves a host sync vs the original "host sums
// partials" path; the host then only reads one 4-byte / 8-byte scalar.
// ---------------------------------------------------------------------------

// out[0] = sum_{i=0..count} partials[i]
//
// Dispatched as a single threadgroup of REDUCTION_THREADGROUP_SIZE threads.
// Each thread strides through `partials` summing into a private accumulator,
// then the threadgroup tree-reduces in shared memory. Handles count >
// threadgroup size via the stride loop. count == 0 writes 0.0f.
kernel void sonium_reduce_sum_f32(
    device const float*  partials  [[buffer(0)]],
    device float*        out       [[buffer(1)]],
    constant uint&       count     [[buffer(2)]],
    threadgroup float*   shared    [[threadgroup(0)]],
    uint                 lid       [[thread_position_in_threadgroup]],
    uint                 tg_size   [[threads_per_threadgroup]]
) {
    float acc = 0.0f;
    for (uint i = lid; i < count; i += tg_size) {
        acc += partials[i];
    }
    shared[lid] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = tg_size / 2; stride > 0; stride >>= 1) {
        if (lid < stride) {
            shared[lid] += shared[lid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (lid == 0) {
        out[0] = shared[0];
    }
}

// Same shape but `partials` is `float2` (Complex32). The output is also
// `float2`.
kernel void sonium_reduce_sum_c32(
    device const float2* partials  [[buffer(0)]],
    device float2*       out       [[buffer(1)]],
    constant uint&       count     [[buffer(2)]],
    threadgroup float2*  shared    [[threadgroup(0)]],
    uint                 lid       [[thread_position_in_threadgroup]],
    uint                 tg_size   [[threads_per_threadgroup]]
) {
    float2 acc = float2(0.0f, 0.0f);
    for (uint i = lid; i < count; i += tg_size) {
        acc += partials[i];
    }
    shared[lid] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = tg_size / 2; stride > 0; stride >>= 1) {
        if (lid < stride) {
            shared[lid] += shared[lid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (lid == 0) {
        out[0] = shared[0];
    }
}

// Multi-dot final reduction: collapse the `(num_basis × num_chunks)` partials
// matrix down to `num_basis` scalars by summing along chunks. One threadgroup
// per basis vector, threads stride through `num_chunks`.
kernel void sonium_reduce_multi_sum_f32(
    device const float*  partials    [[buffer(0)]],
    device float*        out         [[buffer(1)]],
    constant uint&       num_chunks  [[buffer(2)]],
    threadgroup float*   shared      [[threadgroup(0)]],
    uint                 lid         [[thread_position_in_threadgroup]],
    uint                 tg_id       [[threadgroup_position_in_grid]],
    uint                 tg_size     [[threads_per_threadgroup]]
) {
    uint b = tg_id;
    float acc = 0.0f;
    for (uint i = lid; i < num_chunks; i += tg_size) {
        acc += partials[b * num_chunks + i];
    }
    shared[lid] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = tg_size / 2; stride > 0; stride >>= 1) {
        if (lid < stride) {
            shared[lid] += shared[lid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (lid == 0) {
        out[b] = shared[0];
    }
}

kernel void sonium_reduce_multi_sum_c32(
    device const float2* partials    [[buffer(0)]],
    device float2*       out         [[buffer(1)]],
    constant uint&       num_chunks  [[buffer(2)]],
    threadgroup float2*  shared      [[threadgroup(0)]],
    uint                 lid         [[thread_position_in_threadgroup]],
    uint                 tg_id       [[threadgroup_position_in_grid]],
    uint                 tg_size     [[threads_per_threadgroup]]
) {
    uint b = tg_id;
    float2 acc = float2(0.0f, 0.0f);
    for (uint i = lid; i < num_chunks; i += tg_size) {
        acc += partials[b * num_chunks + i];
    }
    shared[lid] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = tg_size / 2; stride > 0; stride >>= 1) {
        if (lid < stride) {
            shared[lid] += shared[lid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (lid == 0) {
        out[b] = shared[0];
    }
}

// ---------------------------------------------------------------------------
// Element/boundary assembly helpers
// ---------------------------------------------------------------------------

// One thread owns each segment. The host-side checked API validates the
// offsets before dispatch, so this kernel can stay branch-light in the hot
// path while preserving the input-order accumulation contract.
kernel void sonium_segmented_sum_f32(
    device const float* values [[buffer(0)]],
    device const int* offsets [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& segments [[buffer(3)]],
    uint gid [[thread_position_in_grid]]) {
    if (gid >= segments) {
        return;
    }
    const int start = offsets[gid];
    const int end = offsets[gid + 1];
    float sum = 0.0f;
    for (int i = start; i < end; ++i) {
        sum += values[i];
    }
    output[gid] = sum;
}

kernel void sonium_segmented_sum_c32(
    device const float2* values [[buffer(0)]],
    device const int* offsets [[buffer(1)]],
    device float2* output [[buffer(2)]],
    constant uint& segments [[buffer(3)]],
    uint gid [[thread_position_in_grid]]) {
    if (gid >= segments) {
        return;
    }
    const int start = offsets[gid];
    const int end = offsets[gid + 1];
    float2 sum = float2(0.0f, 0.0f);
    for (int i = start; i < end; ++i) {
        sum += values[i];
    }
    output[gid] = sum;
}

kernel void sonium_batched_local_contract_f32(
    device const float* lhs [[buffer(0)]],
    device const float* rhs [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& batches [[buffer(3)]],
    constant uint& rows [[buffer(4)]],
    constant uint& inner [[buffer(5)]],
    constant uint& cols [[buffer(6)]],
    uint gid [[thread_position_in_grid]]) {
    const uint total = batches * rows * cols;
    if (gid >= total) {
        return;
    }
    const uint col = gid % cols;
    const uint row = (gid / cols) % rows;
    const uint batch = gid / (rows * cols);
    const uint lhs_base = (batch * rows + row) * inner;
    const uint rhs_base = batch * inner * cols + col;
    float sum = 0.0f;
    for (uint k = 0; k < inner; ++k) {
        sum += lhs[lhs_base + k] * rhs[rhs_base + k * cols];
    }
    output[gid] = sum;
}

kernel void sonium_batched_local_contract_c32(
    device const float2* lhs [[buffer(0)]],
    device const float2* rhs [[buffer(1)]],
    device float2* output [[buffer(2)]],
    constant uint& batches [[buffer(3)]],
    constant uint& rows [[buffer(4)]],
    constant uint& inner [[buffer(5)]],
    constant uint& cols [[buffer(6)]],
    uint gid [[thread_position_in_grid]]) {
    const uint total = batches * rows * cols;
    if (gid >= total) {
        return;
    }
    const uint col = gid % cols;
    const uint row = (gid / cols) % rows;
    const uint batch = gid / (rows * cols);
    const uint lhs_base = (batch * rows + row) * inner;
    const uint rhs_base = batch * inner * cols + col;
    float2 sum = float2(0.0f, 0.0f);
    for (uint k = 0; k < inner; ++k) {
        const float2 a = lhs[lhs_base + k];
        const float2 b = rhs[rhs_base + k * cols];
        sum += float2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
    }
    output[gid] = sum;
}

// Multi-RHS CSR product. Each thread owns one `(rhs, row)` output and keeps
// the column-major RHS layout used by the backend contract, so no temporary
// transpose or host staging is required.
kernel void sonium_spmm_csr_f32(
    device const int* row_ptr [[buffer(0)]],
    device const int* col_idx [[buffer(1)]],
    device const float* values [[buffer(2)]],
    device const float* x [[buffer(3)]],
    device float* y [[buffer(4)]],
    constant uint& rows [[buffer(5)]],
    constant uint& nrhs [[buffer(6)]],
    constant uint& ldx [[buffer(7)]],
    constant uint& ldy [[buffer(8)]],
    uint gid [[thread_position_in_grid]]) {
    const uint total = rows * nrhs;
    if (gid >= total) {
        return;
    }
    const uint row = gid % rows;
    const uint rhs = gid / rows;
    float sum = 0.0f;
    for (int k = row_ptr[row]; k < row_ptr[row + 1]; ++k) {
        sum += values[k] * x[rhs * ldx + uint(col_idx[k])];
    }
    y[rhs * ldy + row] = sum;
}

kernel void sonium_spmm_csr_c32(
    device const int* row_ptr [[buffer(0)]],
    device const int* col_idx [[buffer(1)]],
    device const float2* values [[buffer(2)]],
    device const float2* x [[buffer(3)]],
    device float2* y [[buffer(4)]],
    constant uint& rows [[buffer(5)]],
    constant uint& nrhs [[buffer(6)]],
    constant uint& ldx [[buffer(7)]],
    constant uint& ldy [[buffer(8)]],
    uint gid [[thread_position_in_grid]]) {
    const uint total = rows * nrhs;
    if (gid >= total) {
        return;
    }
    const uint row = gid % rows;
    const uint rhs = gid / rows;
    float2 sum = float2(0.0f, 0.0f);
    for (int k = row_ptr[row]; k < row_ptr[row + 1]; ++k) {
        const float2 a = values[k];
        const float2 b = x[rhs * ldx + uint(col_idx[k])];
        sum += float2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
    }
    y[rhs * ldy + row] = sum;
}

// ---------------------------------------------------------------------------
// BLAS-2 — GEMV (NoTrans only, f32 / Complex32)
// ---------------------------------------------------------------------------
//
// y = alpha * A * x + beta * y   (NoTrans)
//
// Dispatch: one simdgroup (32 threads) per output row. Each thread strides
// `lane..n` step `32`, accumulating into a private scalar; `simd_sum` reduces
// across the simdgroup; lane 0 writes `alpha*sum + beta*y[row]`. This wins
// big over thread-per-row when `n` is large (BEM gemv with n=8192 ≈ 26 µs
// vs ≈ 90 µs for thread-per-row on M4). For small n it's still acceptable.
// A is column-major with leading dimension `lda` (matches `cpu_blas2::gemv_f32`).

kernel void sonium_gemv_f32_notrans(
    device const float*  a         [[buffer(0)]],
    device const float*  x         [[buffer(1)]],
    device float*        y         [[buffer(2)]],
    constant float&      alpha     [[buffer(3)]],
    constant float&      beta      [[buffer(4)]],
    constant uint&       m         [[buffer(5)]],
    constant uint&       n         [[buffer(6)]],
    constant uint&       lda       [[buffer(7)]],
    uint                 tg_id     [[threadgroup_position_in_grid]],
    uint                 sg_lane   [[thread_index_in_simdgroup]]
) {
    uint row = tg_id;
    if (row >= m) { return; }
    float acc = 0.0f;
    // Row-major: A(row, col) = a[row*lda + col]. Matches the Sonium
    // Backend::gemv contract and the CPU `cpu_blas2::gemv_f32` NoTrans path
    // (where rows are contiguous of length `n` with stride `lda >= n`).
    for (uint col = sg_lane; col < n; col += 32) {
        acc = fma(a[row * lda + col], x[col], acc);
    }
    float sum = simd_sum(acc);
    if (sg_lane == 0) {
        float yv = beta == 0.0f ? 0.0f : beta * y[row];
        y[row] = alpha * sum + yv;
    }
}

kernel void sonium_gemv_c32_notrans(
    device const float2* a         [[buffer(0)]],
    device const float2* x         [[buffer(1)]],
    device float2*       y         [[buffer(2)]],
    constant float2&     alpha     [[buffer(3)]],
    constant float2&     beta      [[buffer(4)]],
    constant uint&       m         [[buffer(5)]],
    constant uint&       n         [[buffer(6)]],
    constant uint&       lda       [[buffer(7)]],
    uint                 tg_id     [[threadgroup_position_in_grid]],
    uint                 sg_lane   [[thread_index_in_simdgroup]]
) {
    uint row = tg_id;
    if (row >= m) { return; }
    float2 acc = float2(0.0f, 0.0f);
    // Row-major: A(row, col) = a[row*lda + col].
    for (uint col = sg_lane; col < n; col += 32) {
        acc += cmul(a[row * lda + col], x[col]);
    }
    // No simd_sum for float2 in MSL stdlib; reduce components separately.
    float acc_re = simd_sum(acc.x);
    float acc_im = simd_sum(acc.y);
    if (sg_lane == 0) {
        float2 sum = float2(acc_re, acc_im);
        float2 yv = (beta.x == 0.0f && beta.y == 0.0f)
            ? float2(0.0f, 0.0f)
            : cmul(beta, y[row]);
        y[row] = cmul(alpha, sum) + yv;
    }
}

// ---------------------------------------------------------------------------
// BLAS-2 — GEMV (Trans / ConjTrans, f32 / Complex32)
// ---------------------------------------------------------------------------
//
// Trans semantics (matches Sonium's row-major BLAS contract):
//   stored A is `n × m` row-major with `lda >= m`,
//   y = alpha * A^T * x + beta * y, where A^T is `m × n` (applied).
//   So `y[row] = sum_{j=0..n} stored_A[j, row] * x[j]`
//            = sum_{j=0..n} a[j * lda + row] * x[j].
//
// One simdgroup (32 threads) per *output* row (m rows). Each thread strides
// through `n` source rows, accumulating; `simd_sum` reduces. Memory pattern:
// each lane reads `a[j*lda + row]` strided by `lda * sizeof` between j and
// j+1 — column-strided. For typical Sonium row-major storage this is a
// strided gather; perf is bandwidth-bound but still wins over the host-
// bounce CPU path on M4 above n ≈ 256.

kernel void sonium_gemv_f32_trans(
    device const float*  a         [[buffer(0)]],
    device const float*  x         [[buffer(1)]],
    device float*        y         [[buffer(2)]],
    constant float&      alpha     [[buffer(3)]],
    constant float&      beta      [[buffer(4)]],
    constant uint&       m         [[buffer(5)]],
    constant uint&       n         [[buffer(6)]],
    constant uint&       lda       [[buffer(7)]],
    uint                 tg_id     [[threadgroup_position_in_grid]],
    uint                 sg_lane   [[thread_index_in_simdgroup]]
) {
    uint row = tg_id;
    if (row >= m) { return; }
    float acc = 0.0f;
    // A^T[row, j] = a[j*lda + row]: column-strided access by `lda`.
    for (uint j = sg_lane; j < n; j += 32) {
        acc = fma(a[j * lda + row], x[j], acc);
    }
    float sum = simd_sum(acc);
    if (sg_lane == 0) {
        float yv = beta == 0.0f ? 0.0f : beta * y[row];
        y[row] = alpha * sum + yv;
    }
}

// Complex32 Trans. Caller selects Trans vs ConjTrans by passing
// `conjugate_a = 1` (0.0 disables) — when nonzero, the loaded a element
// is conjugated before the multiply. This handles BEM Burton-Miller's
// `A^H * x` (ConjTrans) and the rarer plain `A^T * x` (Trans) with one
// kernel.
kernel void sonium_gemv_c32_trans(
    device const float2* a            [[buffer(0)]],
    device const float2* x            [[buffer(1)]],
    device float2*       y            [[buffer(2)]],
    constant float2&     alpha        [[buffer(3)]],
    constant float2&     beta         [[buffer(4)]],
    constant uint&       m            [[buffer(5)]],
    constant uint&       n            [[buffer(6)]],
    constant uint&       lda          [[buffer(7)]],
    constant uint&       conjugate_a  [[buffer(8)]],
    uint                 tg_id        [[threadgroup_position_in_grid]],
    uint                 sg_lane      [[thread_index_in_simdgroup]]
) {
    uint row = tg_id;
    if (row >= m) { return; }
    float2 acc = float2(0.0f, 0.0f);
    if (conjugate_a != 0) {
        for (uint j = sg_lane; j < n; j += 32) {
            float2 aij = a[j * lda + row];
            // conj(aij) = (re, -im)
            float2 conj = float2(aij.x, -aij.y);
            acc += cmul(conj, x[j]);
        }
    } else {
        for (uint j = sg_lane; j < n; j += 32) {
            acc += cmul(a[j * lda + row], x[j]);
        }
    }
    float acc_re = simd_sum(acc.x);
    float acc_im = simd_sum(acc.y);
    if (sg_lane == 0) {
        float2 sum = float2(acc_re, acc_im);
        float2 yv = (beta.x == 0.0f && beta.y == 0.0f)
            ? float2(0.0f, 0.0f)
            : cmul(beta, y[row]);
        y[row] = cmul(alpha, sum) + yv;
    }
}

// ---------------------------------------------------------------------------
// FMM helpers — Complex32, device-resident row/vector kernels
// ---------------------------------------------------------------------------
//
// These mirror `sonium_compute::fmm_cpu`'s Complex64 helper shapes for the
// Metal-friendly mixed-precision path. Direction index buffers are signed
// because the Backend gather helpers already use i32. Negative or out-of-range
// entries are ignored. Scatter-add kernels assume each dispatch has unique
// destination indices; that is true for directional cone active-direction maps.

kernel void sonium_fmm_phase_accum_c32(
    device float2*       dst    [[buffer(0)]],
    device const float2* phase  [[buffer(1)]],
    device const float2* src    [[buffer(2)]],
    constant uint&       n      [[buffer(3)]],
    uint                 gid    [[thread_position_in_grid]]
) {
    if (gid >= n) { return; }
    dst[gid] += cmul(phase[gid], src[gid]);
}

kernel void sonium_fmm_compact_gather_c32(
    device const float2* src      [[buffer(0)]],
    device const int*    indices  [[buffer(1)]],
    device float2*       dst      [[buffer(2)]],
    constant uint&       src_len  [[buffer(3)]],
    constant uint&       n        [[buffer(4)]],
    uint                 gid      [[thread_position_in_grid]]
) {
    if (gid >= n) { return; }
    int idx = indices[gid];
    if (idx >= 0 && uint(idx) < src_len) {
        dst[gid] = src[uint(idx)];
    }
}

kernel void sonium_fmm_compact_scatter_accum_c32(
    device const float2* src      [[buffer(0)]],
    device const int*    indices  [[buffer(1)]],
    device float2*       dst      [[buffer(2)]],
    constant uint&       dst_len  [[buffer(3)]],
    constant uint&       n        [[buffer(4)]],
    uint                 gid      [[thread_position_in_grid]]
) {
    if (gid >= n) { return; }
    int idx = indices[gid];
    if (idx >= 0 && uint(idx) < dst_len) {
        dst[uint(idx)] += src[gid];
    }
}

kernel void sonium_fmm_indexed_dense_matvec_accum_c32(
    device const float2* matrix       [[buffer(0)]],
    device const float2* src          [[buffer(1)]],
    device const int*    src_indices  [[buffer(2)]],
    device float2*       dst          [[buffer(3)]],
    device const int*    dst_indices  [[buffer(4)]],
    constant uint&       rows         [[buffer(5)]],
    constant uint&       cols         [[buffer(6)]],
    constant uint&       lda          [[buffer(7)]],
    constant uint&       src_len      [[buffer(8)]],
    constant uint&       dst_len      [[buffer(9)]],
    uint                 tg_id        [[threadgroup_position_in_grid]],
    uint                 sg_lane      [[thread_index_in_simdgroup]]
) {
    uint row = tg_id;
    if (row >= rows) { return; }

    float2 acc = float2(0.0f, 0.0f);
    for (uint col = sg_lane; col < cols; col += 32) {
        int src_idx = src_indices[col];
        if (src_idx >= 0 && uint(src_idx) < src_len) {
            acc += cmul(matrix[row * lda + col], src[uint(src_idx)]);
        }
    }

    float acc_re = simd_sum(acc.x);
    float acc_im = simd_sum(acc.y);
    if (sg_lane == 0) {
        int dst_idx = dst_indices[row];
        if (dst_idx >= 0 && uint(dst_idx) < dst_len) {
            dst[uint(dst_idx)] += float2(acc_re, acc_im);
        }
    }
}

// ---------------------------------------------------------------------------
// Sparse — CSR SpMV, one thread per row
// ---------------------------------------------------------------------------

// y[i] = sum_{k=row_ptr[i]..row_ptr[i+1]} values[k] * x[col_idx[k]]
kernel void sonium_spmv_csr_f32(
    device const int*    row_ptr  [[buffer(0)]],
    device const int*    col_idx  [[buffer(1)]],
    device const float*  values   [[buffer(2)]],
    device const float*  x        [[buffer(3)]],
    device float*        y        [[buffer(4)]],
    constant uint&       m        [[buffer(5)]],
    uint                 gid      [[thread_position_in_grid]]
) {
    if (gid >= m) { return; }
    int lo = row_ptr[gid];
    int hi = row_ptr[gid + 1];
    float acc = 0.0f;
    for (int k = lo; k < hi; ++k) {
        acc = fma(values[k], x[col_idx[k]], acc);
    }
    y[gid] = acc;
}

// ---------------------------------------------------------------------------
// CSR-Adaptive SpMV — three kernels for short / medium / long rows.
// The host (`prebake_csr`) buckets rows by nnz once at prebake time and
// dispatches each kernel over its own row index list. Empty buckets are
// skipped. Variable-nnz BEM compressed operators (where some rows have
// 1-2 entries and others have hundreds) benefit from this stratification:
//   - thread-per-row for nnz < 8        — minimal launch overhead
//   - simdgroup-per-row for nnz < 64    — 32-way parallel reduction, simd_sum
//   - threadgroup-per-row for nnz ≥ 64  — 256-way parallel reduction
// `row_indices[k]` selects which row this k-th invocation works on.
// ---------------------------------------------------------------------------

// Short rows: one thread per row, identical to `spmv_csr_f32` but with
// `row_indices` indirection so we only process bucketed rows.
kernel void sonium_spmv_csr_short_f32(
    device const int*    row_ptr      [[buffer(0)]],
    device const int*    col_idx      [[buffer(1)]],
    device const float*  values       [[buffer(2)]],
    device const float*  x            [[buffer(3)]],
    device float*        y            [[buffer(4)]],
    device const int*    row_indices  [[buffer(5)]],
    constant uint&       num_rows     [[buffer(6)]],
    uint                 gid          [[thread_position_in_grid]]
) {
    if (gid >= num_rows) { return; }
    int row = row_indices[gid];
    int lo = row_ptr[row];
    int hi = row_ptr[row + 1];
    float acc = 0.0f;
    for (int k = lo; k < hi; ++k) {
        acc = fma(values[k], x[col_idx[k]], acc);
    }
    y[row] = acc;
}

// Medium rows: one simdgroup (32 threads) per row, simd_sum reduction.
kernel void sonium_spmv_csr_simd_f32(
    device const int*    row_ptr      [[buffer(0)]],
    device const int*    col_idx      [[buffer(1)]],
    device const float*  values       [[buffer(2)]],
    device const float*  x            [[buffer(3)]],
    device float*        y            [[buffer(4)]],
    device const int*    row_indices  [[buffer(5)]],
    constant uint&       num_rows     [[buffer(6)]],
    uint                 tg_id        [[threadgroup_position_in_grid]],
    uint                 sg_lane      [[thread_index_in_simdgroup]]
) {
    if (tg_id >= num_rows) { return; }
    int row = row_indices[tg_id];
    int lo = row_ptr[row];
    int hi = row_ptr[row + 1];
    float acc = 0.0f;
    for (int k = lo + int(sg_lane); k < hi; k += 32) {
        acc = fma(values[k], x[col_idx[k]], acc);
    }
    float sum = simd_sum(acc);
    if (sg_lane == 0) {
        y[row] = sum;
    }
}

// Long rows: one threadgroup (256 threads) per row, full tree reduction.
kernel void sonium_spmv_csr_cta_f32(
    device const int*    row_ptr      [[buffer(0)]],
    device const int*    col_idx      [[buffer(1)]],
    device const float*  values       [[buffer(2)]],
    device const float*  x            [[buffer(3)]],
    device float*        y            [[buffer(4)]],
    device const int*    row_indices  [[buffer(5)]],
    constant uint&       num_rows     [[buffer(6)]],
    threadgroup float*   shared       [[threadgroup(0)]],
    uint                 tg_id        [[threadgroup_position_in_grid]],
    uint                 lid          [[thread_position_in_threadgroup]],
    uint                 tg_size      [[threads_per_threadgroup]]
) {
    if (tg_id >= num_rows) { return; }
    int row = row_indices[tg_id];
    int lo = row_ptr[row];
    int hi = row_ptr[row + 1];
    float acc = 0.0f;
    for (int k = lo + int(lid); k < hi; k += int(tg_size)) {
        acc = fma(values[k], x[col_idx[k]], acc);
    }
    shared[lid] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = tg_size / 2; stride > 0; stride >>= 1) {
        if (lid < stride) {
            shared[lid] += shared[lid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lid == 0) {
        y[row] = shared[0];
    }
}

// Same trio for Complex32. The simd_sum path reduces re/im components
// separately (no float2 reduction primitive in MSL).
kernel void sonium_spmv_csr_short_c32(
    device const int*    row_ptr      [[buffer(0)]],
    device const int*    col_idx      [[buffer(1)]],
    device const float2* values       [[buffer(2)]],
    device const float2* x            [[buffer(3)]],
    device float2*       y            [[buffer(4)]],
    device const int*    row_indices  [[buffer(5)]],
    constant uint&       num_rows     [[buffer(6)]],
    uint                 gid          [[thread_position_in_grid]]
) {
    if (gid >= num_rows) { return; }
    int row = row_indices[gid];
    int lo = row_ptr[row];
    int hi = row_ptr[row + 1];
    float2 acc = float2(0.0f, 0.0f);
    for (int k = lo; k < hi; ++k) {
        acc += cmul(values[k], x[col_idx[k]]);
    }
    y[row] = acc;
}

kernel void sonium_spmv_csr_simd_c32(
    device const int*    row_ptr      [[buffer(0)]],
    device const int*    col_idx      [[buffer(1)]],
    device const float2* values       [[buffer(2)]],
    device const float2* x            [[buffer(3)]],
    device float2*       y            [[buffer(4)]],
    device const int*    row_indices  [[buffer(5)]],
    constant uint&       num_rows     [[buffer(6)]],
    uint                 tg_id        [[threadgroup_position_in_grid]],
    uint                 sg_lane      [[thread_index_in_simdgroup]]
) {
    if (tg_id >= num_rows) { return; }
    int row = row_indices[tg_id];
    int lo = row_ptr[row];
    int hi = row_ptr[row + 1];
    float2 acc = float2(0.0f, 0.0f);
    for (int k = lo + int(sg_lane); k < hi; k += 32) {
        acc += cmul(values[k], x[col_idx[k]]);
    }
    float acc_re = simd_sum(acc.x);
    float acc_im = simd_sum(acc.y);
    if (sg_lane == 0) {
        y[row] = float2(acc_re, acc_im);
    }
}

kernel void sonium_spmv_csr_cta_c32(
    device const int*    row_ptr      [[buffer(0)]],
    device const int*    col_idx      [[buffer(1)]],
    device const float2* values       [[buffer(2)]],
    device const float2* x            [[buffer(3)]],
    device float2*       y            [[buffer(4)]],
    device const int*    row_indices  [[buffer(5)]],
    constant uint&       num_rows     [[buffer(6)]],
    threadgroup float2*  shared       [[threadgroup(0)]],
    uint                 tg_id        [[threadgroup_position_in_grid]],
    uint                 lid          [[thread_position_in_threadgroup]],
    uint                 tg_size      [[threads_per_threadgroup]]
) {
    if (tg_id >= num_rows) { return; }
    int row = row_indices[tg_id];
    int lo = row_ptr[row];
    int hi = row_ptr[row + 1];
    float2 acc = float2(0.0f, 0.0f);
    for (int k = lo + int(lid); k < hi; k += int(tg_size)) {
        acc += cmul(values[k], x[col_idx[k]]);
    }
    shared[lid] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = tg_size / 2; stride > 0; stride >>= 1) {
        if (lid < stride) {
            shared[lid] += shared[lid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lid == 0) {
        y[row] = shared[0];
    }
}

// ===========================================================================
// Complex32 kernels — Complex<f32> packed as MSL `float2` (re, im)
// ===========================================================================
//
// Memory layout is binary-identical to `num_complex::Complex<f32>` (8-byte
// stride, re-then-im), so these kernels can be dispatched directly against
// the same shared-storage Metal buffers used for f32 ops, just reinterpreted
// as `float2` arrays of half the element count.

// ---------------------------------------------------------------------------
// BLAS-1 — element-wise
// ---------------------------------------------------------------------------

// y[i] = alpha * x[i] + y[i]
kernel void sonium_axpy_c32(
    device const float2* x      [[buffer(0)]],
    device float2*       y      [[buffer(1)]],
    constant float2&     alpha  [[buffer(2)]],
    constant uint&       n      [[buffer(3)]],
    uint                 gid    [[thread_position_in_grid]]
) {
    if (gid >= n) { return; }
    y[gid] = cmul(alpha, x[gid]) + y[gid];
}

// y[i] += sum_j alphas[j] * x_basis[j][i], packed as contiguous vectors.
kernel void sonium_axpy_many_c32(
    constant uint& basis_offset [[buffer(5)]],
    constant uint& basis_row_stride [[buffer(6)]],
    constant uint& basis_col_stride [[buffer(7)]],
    constant uint& y_offset [[buffer(8)]],
    constant uint& y_stride [[buffer(9)]],
    device const float2* x_basis [[buffer(0)]],
    device const float2* alphas  [[buffer(1)]],
    device float2*       y       [[buffer(2)]],
    constant uint&       n       [[buffer(3)]],
    constant uint&       k       [[buffer(4)]],
    uint                 gid     [[thread_position_in_grid]]
) {
    if (gid >= n) { return; }
    float2 acc = y[y_offset + gid * y_stride];
    for (uint basis = 0; basis < k; basis++) {
        acc += cmul(
            alphas[basis],
            x_basis[basis_offset + basis * basis_row_stride + gid * basis_col_stride]);
    }
    y[y_offset + gid * y_stride] = acc;
}

// x[i] = alpha * x[i]
kernel void sonium_scale_c32(
    device float2*       x      [[buffer(0)]],
    constant float2&     alpha  [[buffer(1)]],
    constant uint&       n      [[buffer(2)]],
    uint                 gid    [[thread_position_in_grid]]
) {
    if (gid >= n) { return; }
    x[gid] = cmul(alpha, x[gid]);
}

// ---------------------------------------------------------------------------
// Reductions — per-threadgroup partials, host sums
// ---------------------------------------------------------------------------

// partials[tg] = sum over thread group of conj(x[i]) * y[i]
//
// This is the Hermitian inner product ⟨x, y⟩ used by GMRES / CG over a
// complex Hilbert space. Reduces to the real dot product when x, y are
// real (im components = 0).
kernel void sonium_dot_c32(
    device const float2* x        [[buffer(0)]],
    device const float2* y        [[buffer(1)]],
    device float2*       partials [[buffer(2)]],
    constant uint&       n        [[buffer(3)]],
    threadgroup float2*  shared   [[threadgroup(0)]],
    uint                 gid      [[thread_position_in_grid]],
    uint                 lid      [[thread_position_in_threadgroup]],
    uint                 tg_id    [[threadgroup_position_in_grid]],
    uint                 tg_size  [[threads_per_threadgroup]]
) {
    float2 v = (gid < n) ? cmul_conj_a(x[gid], y[gid]) : float2(0.0f, 0.0f);
    shared[lid] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = tg_size / 2; stride > 0; stride >>= 1) {
        if (lid < stride) {
            shared[lid] += shared[lid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (lid == 0) {
        partials[tg_id] = shared[0];
    }
}

// partials[tg] = sum over thread group of |x[i]|² (real-valued)
//
// Slot 1 is unused; layout matches `sonium_dot_c32` so the Rust dispatcher
// can bind both kernels with the same parameter pattern.
kernel void sonium_nrm2_sq_c32(
    device const float2* x         [[buffer(0)]],
    device const float2* x_alias   [[buffer(1)]],
    device float*        partials  [[buffer(2)]],
    constant uint&       n         [[buffer(3)]],
    threadgroup float*   shared    [[threadgroup(0)]],
    uint                 gid       [[thread_position_in_grid]],
    uint                 lid       [[thread_position_in_threadgroup]],
    uint                 tg_id     [[threadgroup_position_in_grid]],
    uint                 tg_size   [[threads_per_threadgroup]]
) {
    (void)x_alias;
    float v = (gid < n) ? cnorm_sqr(x[gid]) : 0.0f;
    shared[lid] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = tg_size / 2; stride > 0; stride >>= 1) {
        if (lid < stride) {
            shared[lid] += shared[lid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (lid == 0) {
        partials[tg_id] = shared[0];
    }
}

// Fused Hermitian multi-dot: partials[b * num_chunks + chunk] =
//   sum over chunk of conj(basis_concat[b * n + i]) * y[i].
//
// Layout matches `sonium_multi_dot_f32`. Used by Metal-backed GMRES CGS2 to
// collapse `j+1` Gram-Schmidt syncs into a single dispatch.
kernel void sonium_multi_dot_c32(
    constant uint& basis_offset [[buffer(5)]],
    constant uint& basis_row_stride [[buffer(6)]],
    constant uint& basis_col_stride [[buffer(7)]],
    constant uint& y_offset [[buffer(8)]],
    constant uint& y_stride [[buffer(9)]],
    device const float2* basis_concat  [[buffer(0)]],
    device const float2* y             [[buffer(1)]],
    device float2*       partials      [[buffer(2)]],
    constant uint&       n             [[buffer(3)]],
    constant uint&       num_chunks    [[buffer(4)]],
    threadgroup float2*  shared        [[threadgroup(0)]],
    uint2                gid           [[thread_position_in_grid]],
    uint2                lid           [[thread_position_in_threadgroup]],
    uint2                tg_id         [[threadgroup_position_in_grid]],
    uint2                tg_size       [[threads_per_threadgroup]]
) {
    uint i = gid.x;
    uint b = gid.y;
    uint chunk = tg_id.x;
    uint local = lid.x;
    uint group = tg_size.x;

    float2 v = (i < n)
        ? cmul_conj_a(
            basis_concat[basis_offset + b * basis_row_stride + i * basis_col_stride],
            y[y_offset + i * y_stride])
        : float2(0.0f, 0.0f);
    shared[local] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = group / 2; stride > 0; stride >>= 1) {
        if (local < stride) {
            shared[local] += shared[local + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (local == 0) {
        partials[b * num_chunks + chunk] = shared[0];
    }
}

// ---------------------------------------------------------------------------
// Sparse — Complex32 CSR SpMV
// ---------------------------------------------------------------------------

// y[i] = sum_{k=row_ptr[i]..row_ptr[i+1]} values[k] * x[col_idx[k]]
//
// No `fma` for float2 in MSL stdlib; use direct cmul + add. Apple's compiler
// folds the multiply-add into FMAs at the scalar level when possible.
kernel void sonium_spmv_csr_c32(
    device const int*    row_ptr  [[buffer(0)]],
    device const int*    col_idx  [[buffer(1)]],
    device const float2* values   [[buffer(2)]],
    device const float2* x        [[buffer(3)]],
    device float2*       y        [[buffer(4)]],
    constant uint&       m        [[buffer(5)]],
    uint                 gid      [[thread_position_in_grid]]
) {
    if (gid >= m) { return; }
    int lo = row_ptr[gid];
    int hi = row_ptr[gid + 1];
    float2 acc = float2(0.0f, 0.0f);
    for (int k = lo; k < hi; ++k) {
        acc = acc + cmul(values[k], x[col_idx[k]]);
    }
    y[gid] = acc;
}

// ---------------------------------------------------------------------------
// Gather4 — weighted four-point gather for interpolation / sampling
// ---------------------------------------------------------------------------

kernel void sonium_gather4_f32(
    device const float*  values   [[buffer(0)]],
    device const int*    indices  [[buffer(1)]],
    device const float*  weights  [[buffer(2)]],
    device float*        out      [[buffer(3)]],
    constant uint&       n        [[buffer(4)]],
    uint                 gid      [[thread_position_in_grid]]
) {
    if (gid >= n) { return; }
    float acc = 0.0f;
    uint base = gid * 4;
    for (uint j = 0; j < 4; ++j) {
        uint idx = indices[base + j];
        acc = fma(weights[base + j], values[idx], acc);
    }
    out[gid] = acc;
}

kernel void sonium_gather4_c32(
    device const float2* values   [[buffer(0)]],
    device const int*    indices  [[buffer(1)]],
    device const float*  weights  [[buffer(2)]],
    device float2*       out      [[buffer(3)]],
    constant uint&       n        [[buffer(4)]],
    uint                 gid      [[thread_position_in_grid]]
) {
    if (gid >= n) { return; }
    float2 acc = float2(0.0f, 0.0f);
    uint base = gid * 4;
    for (uint j = 0; j < 4; ++j) {
        uint idx = indices[base + j];
        acc = acc + weights[base + j] * values[idx];
    }
    out[gid] = acc;
}

// ---------------------------------------------------------------------------
// BEM regular-pair Helmholtz kernel — Phase 1.3 of `bem-step1.md`.
//
// Per-pair Helmholtz kernel evaluation on a triangle field element using
// fixed-order Gauss quadrature, no adaptive subdivision. The host pre-
// screens each (source, field) pair with `regular::optimal_quadrature_order`
// and routes only the geometrically far-field pairs (ratio ≥ 3.0,
// 4-point-quadrature regime) through this kernel — closer pairs and
// self-element pairs stay on CPU under their existing adaptive paths.
//
// One thread = one pair. Each thread runs the entire 4-point quadrature
// loop and writes one `PairOut` (four Complex32 kernel integrals).
// The host then upcasts to Complex64 and folds in the BC and Burton-
// Miller weighting on the CPU side, matching the existing tbem assembly
// shape exactly.
//
// Mirrors `regular_integration` from
// crates/sonium-bem/src/core/integration/regular.rs lines 75-225 in the
// `iforie == true` branch (single subelement covering the whole field
// triangle, factor = 1.0). The four output kernels:
//   G   = ∫ exp(ikr)/(4πr) dS_y
//   H   = ∫ ∂G/∂n_y dS_y
//   H^T = ∫ ∂G/∂n_x dS_y
//   E   = ∫ ∂²G/∂n_x∂n_y dS_y
// match the field names of `IntegrationResult`. f32 only — Apple Silicon
// Metal does not support `double`. Block-relative error vs the f64 CPU
// path is bounded by f32 ULP × condition number of the integrand, which
// for kr ≤ 50 stays inside 1e-6 in spot-checks; the host-side integration
// shim asserts this.

constant float SONIUM_BEM_PI = 3.14159265358979323846f;

// 4-point triangle quadrature (GAUCORWEI_TR4 from gauss.rs, weights
// already pre-scaled by AREA_SCALE = 0.5 so they sum to 0.5 — the
// reference triangle area). xi, eta, w.
constant float3 SONIUM_BEM_TRI4_QUAD[4] = {
    float3(0.333333333333333f, 0.333333333333333f, -0.281250f),
    float3(0.6f,               0.2f,                0.260416666666667f),
    float3(0.2f,               0.6f,                0.260416666666667f),
    float3(0.2f,               0.2f,                0.260416666666667f),
};

// Per-pair input descriptor. Layout matches the Rust `PairIn` struct.
// 16-aligned to avoid MSL packing surprises across float3 fields.
struct SoniumBemRegularPairIn {
    packed_float3 source_point;   //  0..12
    float         _pad0;          // 12..16  (alignment)
    packed_float3 source_normal;  // 16..28
    float         _pad1;          // 28..32
    packed_float3 v0;             // 32..44
    float         _pad2;          // 44..48
    packed_float3 v1;             // 48..60
    float         _pad3;          // 60..64
    packed_float3 v2;             // 64..76
    float         _pad4;          // 76..80
};

// Per-pair output: four Complex32 kernels packed contiguously.
struct SoniumBemRegularPairOut {
    float2 g;
    float2 h;
    float2 ht;
    float2 e;
};

// Constant params shared by every pair in a batch.
struct SoniumBemPhysicsParams {
    float k;                // wave number (rad/m)
    float harmonic_factor;  // ±1, sign convention of imaginary part
    uint  n_pairs;
    uint  _pad;
};

kernel void sonium_bem_regular_pair_c32(
    device const SoniumBemRegularPairIn*  pairs    [[buffer(0)]],
    device       SoniumBemRegularPairOut* out      [[buffer(1)]],
    constant     SoniumBemPhysicsParams&  params   [[buffer(2)]],
    uint                                  gid      [[thread_position_in_grid]]
) {
    if (gid >= params.n_pairs) { return; }

    const SoniumBemRegularPairIn p = pairs[gid];
    const float3 src    = float3(p.source_point);
    const float3 n_x    = float3(p.source_normal);
    const float3 v0     = float3(p.v0);
    const float3 v1     = float3(p.v1);
    const float3 v2     = float3(p.v2);

    const float k         = params.k;
    const float k2        = k * k;
    const float wavruim   = params.harmonic_factor * k;

    // Element flat normal = (v1-v0) × (v2-v0); Jacobian = ‖normal‖
    // (== 2 × triangle area). Tri3 normals and Jacobian are constant
    // across the element, so compute once outside the quadrature loop.
    const float3 e1     = v1 - v0;
    const float3 e2     = v2 - v0;
    const float3 raw_n  = cross(e1, e2);
    const float  jac    = length(raw_n);
    const float3 n_y    = (jac > 1e-15f) ? (raw_n / jac) : float3(0.0f);

    // Accumulators (Complex32). Per-component summation is f32-only;
    // the host upcasts to Complex64 before storing in the system matrix.
    float2 acc_g  = float2(0.0f);
    float2 acc_h  = float2(0.0f);
    float2 acc_ht = float2(0.0f);
    float2 acc_e  = float2(0.0f);

    // 4-point Gauss-tri quadrature, single subelement covering the whole
    // field triangle (no adaptive subdivision — caller has pre-screened).
    for (uint q = 0; q < 4; ++q) {
        const float3 qp  = SONIUM_BEM_TRI4_QUAD[q];
        const float  xi  = qp.x;
        const float  eta = qp.y;
        const float  w   = qp.z;

        // Tri3 shape functions: N0 = 1-ξ-η, N1 = ξ, N2 = η.
        // crd_poi = N0·v0 + N1·v1 + N2·v2.
        const float  n0  = 1.0f - xi - eta;
        const float3 crd = n0 * v0 + xi * v1 + eta * v2;

        const float wga = w * jac;

        // Distance vector and unit normal.
        const float3 diff = crd - src;
        const float  r    = length(diff);
        if (r < 1e-15f) { continue; }
        const float3 ur   = diff / r;

        // G = exp(ikr) · wga / (4πr) — written as cos+i·sin to avoid
        // the f32 cexp pitfall.
        const float  re1  = wavruim * r;
        const float  re2  = wga / (4.0f * SONIUM_BEM_PI * r);
        const float2 g    = float2(precise::cos(re1) * re2,
                                   precise::sin(re1) * re2);

        // z1 = -1/r + i·k (sign-bearing).
        const float2 z1   = float2(-1.0f / r, wavruim);
        const float2 hb   = cmul(g, z1);

        const float  r1h  = dot(ur,  n_y);     //  (y-x)·n_y / r
        const float  r2h  = -dot(ur, n_x);     // -(y-x)·n_x / r

        const float2 h_  = float2(hb.x * r1h, hb.y * r1h);  // hb · r1h
        const float2 ht_ = float2(hb.x * r2h, hb.y * r2h);  // hb · r2h

        // E = G · [(3/r² - k²)·rq + nx·ny/r²,  -(k/r)·(3·rq + nx·ny)]
        const float rq      = r1h * r2h;
        const float nx_ny   = dot(n_x, n_y);
        const float dq      = r * r;
        const float2 e_fac  = float2((3.0f / dq - k2) * rq + nx_ny / dq,
                                     -wavruim / r * (3.0f * rq + nx_ny));
        const float2 e_     = cmul(g, e_fac);

        acc_g  = acc_g  + g;
        acc_h  = acc_h  + h_;
        acc_ht = acc_ht + ht_;
        acc_e  = acc_e  + e_;
    }

    SoniumBemRegularPairOut r;
    r.g  = acc_g;
    r.h  = acc_h;
    r.ht = acc_ht;
    r.e  = acc_e;
    out[gid] = r;
}
