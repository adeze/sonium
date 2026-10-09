// Device-resident Complex64 primitives used by hierarchical BEM operators.
//
// These kernels intentionally use one deterministic accumulation lane per
// output row.  FMM translations are normally small dense rows; keeping the
// reduction inside the owning thread avoids atomics when several cone or leaf
// maps are reused on the same stream.  Index maps are signed so stale entries
// can be ignored without an out-of-bounds device access.

struct Complex64 {
    double re;
    double im;
};

__device__ __forceinline__ Complex64 c64_mul(Complex64 a, Complex64 b) {
    return Complex64{a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re};
}

__device__ __forceinline__ Complex64 c64_add(Complex64 a, Complex64 b) {
    return Complex64{a.re + b.re, a.im + b.im};
}

extern "C" __global__ void sonium_fmm_phase_accum_c64(
    Complex64* dst,
    const Complex64* phase,
    const Complex64* src,
    unsigned n
) {
    const unsigned gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= n) return;
    dst[gid] = c64_add(dst[gid], c64_mul(phase[gid], src[gid]));
}

extern "C" __global__ void sonium_fmm_compact_gather_c64(
    const Complex64* src,
    const int* indices,
    Complex64* dst,
    unsigned src_len,
    unsigned n
) {
    const unsigned gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= n) return;
    const int index = indices[gid];
    if (index >= 0 && static_cast<unsigned>(index) < src_len) {
        dst[gid] = src[index];
    }
}

extern "C" __global__ void sonium_fmm_compact_scatter_accum_c64(
    const Complex64* src,
    const int* indices,
    Complex64* dst,
    unsigned dst_len,
    unsigned n
) {
    const unsigned gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= n) return;
    const int index = indices[gid];
    if (index >= 0 && static_cast<unsigned>(index) < dst_len) {
        dst[index] = c64_add(dst[index], src[gid]);
    }
}

extern "C" __global__ void sonium_fmm_indexed_dense_matvec_accum_c64(
    const Complex64* matrix,
    const Complex64* src,
    const int* src_indices,
    Complex64* dst,
    const int* dst_indices,
    unsigned rows,
    unsigned cols,
    unsigned lda,
    unsigned src_len,
    unsigned dst_len
) {
    const unsigned row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) return;

    Complex64 acc = {0.0, 0.0};
    for (unsigned col = 0; col < cols; ++col) {
        const int src_index = src_indices[col];
        if (src_index >= 0 && static_cast<unsigned>(src_index) < src_len) {
            acc = c64_add(acc, c64_mul(matrix[row * lda + col], src[src_index]));
        }
    }

    const int dst_index = dst_indices[row];
    if (dst_index >= 0 && static_cast<unsigned>(dst_index) < dst_len) {
        dst[dst_index] = c64_add(dst[dst_index], acc);
    }
}

extern "C" __global__ void sonium_fmm_dense_matvec_scatter_accum_c64(
    const Complex64* matrix,
    const Complex64* src,
    Complex64* dst,
    const int* dst_indices,
    unsigned rows,
    unsigned cols,
    unsigned lda,
    unsigned dst_len
) {
    const unsigned row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) return;

    Complex64 acc = {0.0, 0.0};
    for (unsigned col = 0; col < cols; ++col) {
        acc = c64_add(acc, c64_mul(matrix[row * lda + col], src[col]));
    }

    const int dst_index = dst_indices[row];
    if (dst_index >= 0 && static_cast<unsigned>(dst_index) < dst_len) {
        dst[dst_index] = c64_add(dst[dst_index], acc);
    }
}

// Strided variants bind the same specialised operations to caller-owned
// views.  Offsets and strides are expressed in elements, matching
// `sonium_compute::BufferView`; index values are relative to their logical
// value view rather than the backing allocation.  Each operation keeps the
// original whole-buffer entry point above so existing callers retain ABI
// compatibility while new callers get checked offset/stride coverage.
extern "C" __global__ void sonium_fmm_phase_accum_strided_c64(
    Complex64* dst,
    const Complex64* phase,
    const Complex64* src,
    unsigned long long n,
    unsigned long long dst_offset,
    unsigned long long dst_stride,
    unsigned long long phase_offset,
    unsigned long long phase_stride,
    unsigned long long src_offset,
    unsigned long long src_stride
) {
    const unsigned long long gid =
        static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (gid >= n) return;
    const unsigned long long dst_index = dst_offset + gid * dst_stride;
    const unsigned long long phase_index = phase_offset + gid * phase_stride;
    const unsigned long long src_index = src_offset + gid * src_stride;
    dst[dst_index] = c64_add(dst[dst_index], c64_mul(phase[phase_index], src[src_index]));
}

extern "C" __global__ void sonium_fmm_zero_strided_c64(
    Complex64* dst,
    unsigned long long n,
    unsigned long long dst_offset,
    unsigned long long dst_stride
) {
    const unsigned long long gid =
        static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (gid >= n) return;
    dst[dst_offset + gid * dst_stride] = Complex64{0.0, 0.0};
}

extern "C" __global__ void sonium_fmm_compact_gather_strided_c64(
    const Complex64* src,
    const int* indices,
    Complex64* dst,
    unsigned long long src_len,
    unsigned long long n,
    unsigned long long src_offset,
    unsigned long long src_stride,
    unsigned long long indices_offset,
    unsigned long long indices_stride,
    unsigned long long dst_offset,
    unsigned long long dst_stride
) {
    const unsigned long long gid =
        static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (gid >= n) return;
    const int index = indices[indices_offset + gid * indices_stride];
    if (index >= 0 && static_cast<unsigned long long>(index) < src_len) {
        dst[dst_offset + gid * dst_stride] = src[src_offset + static_cast<unsigned long long>(index) * src_stride];
    }
}

extern "C" __global__ void sonium_fmm_compact_scatter_accum_strided_c64(
    const Complex64* src,
    const int* indices,
    Complex64* dst,
    unsigned long long dst_len,
    unsigned long long n,
    unsigned long long src_offset,
    unsigned long long src_stride,
    unsigned long long indices_offset,
    unsigned long long indices_stride,
    unsigned long long dst_offset,
    unsigned long long dst_stride
) {
    const unsigned long long gid =
        static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (gid >= n) return;
    const int index = indices[indices_offset + gid * indices_stride];
    if (index >= 0 && static_cast<unsigned long long>(index) < dst_len) {
        const unsigned long long dst_index = dst_offset + static_cast<unsigned long long>(index) * dst_stride;
        const unsigned long long src_index = src_offset + gid * src_stride;
        dst[dst_index] = c64_add(dst[dst_index], src[src_index]);
    }
}

extern "C" __global__ void sonium_fmm_indexed_dense_matvec_accum_strided_c64(
    const Complex64* matrix,
    const Complex64* src,
    const int* src_indices,
    Complex64* dst,
    const int* dst_indices,
    unsigned long long rows,
    unsigned long long cols,
    unsigned long long lda,
    unsigned long long src_len,
    unsigned long long dst_len,
    unsigned long long matrix_offset,
    unsigned long long src_offset,
    unsigned long long src_stride,
    unsigned long long src_indices_offset,
    unsigned long long src_indices_stride,
    unsigned long long dst_offset,
    unsigned long long dst_stride,
    unsigned long long dst_indices_offset,
    unsigned long long dst_indices_stride
) {
    const unsigned long long row =
        static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (row >= rows) return;

    Complex64 acc = {0.0, 0.0};
    for (unsigned long long col = 0; col < cols; ++col) {
        const int src_index = src_indices[src_indices_offset + col * src_indices_stride];
        if (src_index >= 0 && static_cast<unsigned long long>(src_index) < src_len) {
            acc = c64_add(
                acc,
                c64_mul(
                    matrix[matrix_offset + row * lda + col],
                    src[src_offset + static_cast<unsigned long long>(src_index) * src_stride]
                )
            );
        }
    }

    const int dst_index = dst_indices[dst_indices_offset + row * dst_indices_stride];
    if (dst_index >= 0 && static_cast<unsigned long long>(dst_index) < dst_len) {
        const unsigned long long target = dst_offset + static_cast<unsigned long long>(dst_index) * dst_stride;
        dst[target] = c64_add(dst[target], acc);
    }
}

extern "C" __global__ void sonium_fmm_dense_matvec_scatter_accum_strided_c64(
    const Complex64* matrix,
    const Complex64* src,
    Complex64* dst,
    const int* dst_indices,
    unsigned long long rows,
    unsigned long long cols,
    unsigned long long lda,
    unsigned long long dst_len,
    unsigned long long matrix_offset,
    unsigned long long src_offset,
    unsigned long long src_stride,
    unsigned long long dst_offset,
    unsigned long long dst_stride,
    unsigned long long dst_indices_offset,
    unsigned long long dst_indices_stride
) {
    const unsigned long long row =
        static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (row >= rows) return;

    Complex64 acc = {0.0, 0.0};
    for (unsigned long long col = 0; col < cols; ++col) {
        acc = c64_add(
            acc,
            c64_mul(matrix[matrix_offset + row * lda + col], src[src_offset + col * src_stride])
        );
    }

    const int dst_index = dst_indices[dst_indices_offset + row * dst_indices_stride];
    if (dst_index >= 0 && static_cast<unsigned long long>(dst_index) < dst_len) {
        const unsigned long long target = dst_offset + static_cast<unsigned long long>(dst_index) * dst_stride;
        dst[target] = c64_add(dst[target], acc);
    }
}
