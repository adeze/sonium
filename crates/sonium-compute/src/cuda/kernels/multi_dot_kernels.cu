extern "C" __global__ void sonium_multi_dot_f32(
    const unsigned long long* x_ptrs,
    const float* y,
    float* out,
    unsigned long long n
) {
    extern __shared__ float sh_f32[];
    const unsigned int tid = threadIdx.x;
    const unsigned int basis = blockIdx.x;
    const float* x = (const float*)x_ptrs[basis];
    float acc = 0.0f;
    for (unsigned long long i = tid; i < n; i += blockDim.x) {
        acc += x[i] * y[i];
    }
    sh_f32[tid] = acc;
    __syncthreads();

    for (unsigned int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sh_f32[tid] += sh_f32[tid + stride];
        }
        __syncthreads();
    }
    if (tid == 0) {
        out[basis] = sh_f32[0];
    }
}

extern "C" __global__ void sonium_axpy_many_f32(
    const unsigned long long* x_ptrs,
    const float* alphas,
    float* y,
    unsigned long long n,
    unsigned long long k
) {
    const unsigned long long i =
        static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }
    float acc = y[i];
    for (unsigned long long basis = 0; basis < k; basis++) {
        const float* x = (const float*)x_ptrs[basis];
        acc = fmaf(alphas[basis], x[i], acc);
    }
    y[i] = acc;
}

extern "C" __global__ void sonium_multi_dot_f64(
    const unsigned long long* x_ptrs,
    const double* y,
    double* out,
    unsigned long long n
) {
    extern __shared__ double sh_f64[];
    const unsigned int tid = threadIdx.x;
    const unsigned int basis = blockIdx.x;
    const double* x = (const double*)x_ptrs[basis];
    double acc = 0.0;
    for (unsigned long long i = tid; i < n; i += blockDim.x) {
        acc += x[i] * y[i];
    }
    sh_f64[tid] = acc;
    __syncthreads();

    for (unsigned int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sh_f64[tid] += sh_f64[tid + stride];
        }
        __syncthreads();
    }
    if (tid == 0) {
        out[basis] = sh_f64[0];
    }
}

extern "C" __global__ void sonium_axpy_many_f64(
    const unsigned long long* x_ptrs,
    const double* alphas,
    double* y,
    unsigned long long n,
    unsigned long long k
) {
    const unsigned long long i =
        static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }
    double acc = y[i];
    for (unsigned long long basis = 0; basis < k; basis++) {
        const double* x = (const double*)x_ptrs[basis];
        acc = alphas[basis] * x[i] + acc;
    }
    y[i] = acc;
}

// Arch-independent double-precision atomic add: atomicAdd(double) needs
// sm_60+ at NVRTC compile time, while the CAS loop below compiles for every
// architecture. One atomic per block, so the retry cost is negligible.
__device__ inline double sonium_atomic_add_f64(double* addr, double value) {
    unsigned long long* addr_u64 = (unsigned long long*)addr;
    unsigned long long old = *addr_u64;
    unsigned long long assumed;
    do {
        assumed = old;
        old = atomicCAS(addr_u64, assumed,
                        __double_as_longlong(value + __longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
}

// Tiled pointer-array multi-dot variants: same occupancy rationale as the
// strided tiled kernels, but each basis vector is fetched through the device
// pointer table. gridDim.x = basis_count * tiles; out[] MUST be zeroed before
// launch; blockDim.x must be a power of two.
extern "C" __global__ void sonium_multi_dot_ptr_tiled_f32(
    const unsigned long long* x_ptrs,
    const float* y,
    float* out,
    unsigned long long n,
    unsigned long long tiles) {
    extern __shared__ float sh_ptr_tiled_f32[];
    const unsigned int tid = threadIdx.x;
    const unsigned long long tile = blockIdx.x % tiles;
    const unsigned int basis = (unsigned int)(blockIdx.x / tiles);
    const float* x = (const float*)x_ptrs[basis];
    const unsigned long long lane_count = (unsigned long long)blockDim.x * tiles;
    float acc = 0.0f;
    for (unsigned long long i = tile * (unsigned long long)blockDim.x + tid; i < n;
         i += lane_count) {
        acc += x[i] * y[i];
    }
    sh_ptr_tiled_f32[tid] = acc;
    __syncthreads();
    for (unsigned int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) sh_ptr_tiled_f32[tid] += sh_ptr_tiled_f32[tid + stride];
        __syncthreads();
    }
    if (tid == 0) atomicAdd(out + basis, sh_ptr_tiled_f32[0]);
}

extern "C" __global__ void sonium_multi_dot_ptr_tiled_f64(
    const unsigned long long* x_ptrs,
    const double* y,
    double* out,
    unsigned long long n,
    unsigned long long tiles) {
    extern __shared__ double sh_ptr_tiled_f64[];
    const unsigned int tid = threadIdx.x;
    const unsigned long long tile = blockIdx.x % tiles;
    const unsigned int basis = (unsigned int)(blockIdx.x / tiles);
    const double* x = (const double*)x_ptrs[basis];
    const unsigned long long lane_count = (unsigned long long)blockDim.x * tiles;
    double acc = 0.0;
    for (unsigned long long i = tile * (unsigned long long)blockDim.x + tid; i < n;
         i += lane_count) {
        acc += x[i] * y[i];
    }
    sh_ptr_tiled_f64[tid] = acc;
    __syncthreads();
    for (unsigned int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) sh_ptr_tiled_f64[tid] += sh_ptr_tiled_f64[tid + stride];
        __syncthreads();
    }
    if (tid == 0) sonium_atomic_add_f64(out + basis, sh_ptr_tiled_f64[0]);
}

struct Complex32Kernel {
    float re;
    float im;
};

extern "C" __global__ void sonium_multi_dot_c32(
    const unsigned long long* x_ptrs,
    const Complex32Kernel* y,
    Complex32Kernel* out,
    unsigned long long n
) {
    extern __shared__ float sh_c32[];
    float* sh_re = sh_c32;
    float* sh_im = sh_c32 + blockDim.x;
    const unsigned int tid = threadIdx.x;
    const unsigned int basis = blockIdx.x;
    const Complex32Kernel* x = (const Complex32Kernel*)x_ptrs[basis];
    float acc_re = 0.0f;
    float acc_im = 0.0f;
    for (unsigned long long i = tid; i < n; i += blockDim.x) {
        const float xr = x[i].re;
        const float xi = x[i].im;
        const float yr = y[i].re;
        const float yi = y[i].im;
        acc_re += xr * yr + xi * yi;
        acc_im += xr * yi - xi * yr;
    }
    sh_re[tid] = acc_re;
    sh_im[tid] = acc_im;
    __syncthreads();

    for (unsigned int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sh_re[tid] += sh_re[tid + stride];
            sh_im[tid] += sh_im[tid + stride];
        }
        __syncthreads();
    }
    if (tid == 0) {
        out[basis].re = sh_re[0];
        out[basis].im = sh_im[0];
    }
}

extern "C" __global__ void sonium_axpy_many_c32(
    const unsigned long long* x_ptrs,
    const Complex32Kernel* alphas,
    Complex32Kernel* y,
    unsigned long long n,
    unsigned long long k
) {
    const unsigned long long i =
        static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }
    float acc_re = y[i].re;
    float acc_im = y[i].im;
    for (unsigned long long basis = 0; basis < k; basis++) {
        const Complex32Kernel alpha = alphas[basis];
        const Complex32Kernel* x = (const Complex32Kernel*)x_ptrs[basis];
        const float xr = x[i].re;
        const float xi = x[i].im;
        acc_re += alpha.re * xr - alpha.im * xi;
        acc_im += alpha.re * xi + alpha.im * xr;
    }
    y[i].re = acc_re;
    y[i].im = acc_im;
}

struct Complex64Kernel {
    double re;
    double im;
};

extern "C" __global__ void sonium_multi_dot_c64(
    const unsigned long long* x_ptrs,
    const Complex64Kernel* y,
    Complex64Kernel* out,
    unsigned long long n
) {
    extern __shared__ double sh_c64[];
    double* sh_re = sh_c64;
    double* sh_im = sh_c64 + blockDim.x;
    const unsigned int tid = threadIdx.x;
    const unsigned int basis = blockIdx.x;
    const Complex64Kernel* x = (const Complex64Kernel*)x_ptrs[basis];
    double acc_re = 0.0;
    double acc_im = 0.0;
    for (unsigned long long i = tid; i < n; i += blockDim.x) {
        const double xr = x[i].re;
        const double xi = x[i].im;
        const double yr = y[i].re;
        const double yi = y[i].im;
        acc_re += xr * yr + xi * yi;
        acc_im += xr * yi - xi * yr;
    }
    sh_re[tid] = acc_re;
    sh_im[tid] = acc_im;
    __syncthreads();

    for (unsigned int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sh_re[tid] += sh_re[tid + stride];
            sh_im[tid] += sh_im[tid + stride];
        }
        __syncthreads();
    }
    if (tid == 0) {
        out[basis].re = sh_re[0];
        out[basis].im = sh_im[0];
    }
}

extern "C" __global__ void sonium_axpy_many_c64(
    const unsigned long long* x_ptrs,
    const Complex64Kernel* alphas,
    Complex64Kernel* y,
    unsigned long long n,
    unsigned long long k
) {
    const unsigned long long i =
        static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }
    double acc_re = y[i].re;
    double acc_im = y[i].im;
    for (unsigned long long basis = 0; basis < k; basis++) {
        const Complex64Kernel alpha = alphas[basis];
        const Complex64Kernel* x = (const Complex64Kernel*)x_ptrs[basis];
        const double xr = x[i].re;
        const double xi = x[i].im;
        acc_re += alpha.re * xr - alpha.im * xi;
        acc_im += alpha.re * xi + alpha.im * xr;
    }
    y[i].re = acc_re;
    y[i].im = acc_im;
}

// Flat column-major basis variants avoid the host pointer-table upload used
// by the pointer-array multi_dot kernels. The basis layout is
// `basis[basis_index * vector_len + element]`.
extern "C" __global__ void sonium_multi_dot_strided_f32(
    const float* basis,
    const float* y,
    float* out,
    unsigned long long vector_len,
    unsigned long long basis_count,
    unsigned long long basis_offset,
    unsigned long long basis_row_stride,
    unsigned long long basis_col_stride,
    unsigned long long y_offset,
    unsigned long long y_stride) {
    extern __shared__ float sh_strided_f32[];
    const unsigned int tid = threadIdx.x;
    const unsigned long long basis_index = blockIdx.x;
    if (basis_index >= basis_count) return;
    const float* x = basis + basis_offset + basis_index * basis_row_stride;
    float acc = 0.0f;
    for (unsigned long long i = tid; i < vector_len; i += blockDim.x) {
        acc += x[i * basis_col_stride] * y[y_offset + i * y_stride];
    }
    sh_strided_f32[tid] = acc;
    __syncthreads();
    for (unsigned int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) sh_strided_f32[tid] += sh_strided_f32[tid + stride];
        __syncthreads();
    }
    if (tid == 0) out[basis_index] = sh_strided_f32[0];
}

extern "C" __global__ void sonium_multi_dot_strided_f64(
    const double* basis,
    const double* y,
    double* out,
    unsigned long long vector_len,
    unsigned long long basis_count,
    unsigned long long basis_offset,
    unsigned long long basis_row_stride,
    unsigned long long basis_col_stride,
    unsigned long long y_offset,
    unsigned long long y_stride) {
    extern __shared__ double sh_strided_f64[];
    const unsigned int tid = threadIdx.x;
    const unsigned long long basis_index = blockIdx.x;
    if (basis_index >= basis_count) return;
    const double* x = basis + basis_offset + basis_index * basis_row_stride;
    double acc = 0.0;
    for (unsigned long long i = tid; i < vector_len; i += blockDim.x) {
        acc += x[i * basis_col_stride] * y[y_offset + i * y_stride];
    }
    sh_strided_f64[tid] = acc;
    __syncthreads();
    for (unsigned int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) sh_strided_f64[tid] += sh_strided_f64[tid + stride];
        __syncthreads();
    }
    if (tid == 0) out[basis_index] = sh_strided_f64[0];
}

extern "C" __global__ void sonium_multi_dot_strided_c32(
    const Complex32Kernel* basis,
    const Complex32Kernel* y,
    Complex32Kernel* out,
    unsigned long long vector_len,
    unsigned long long basis_count,
    unsigned long long basis_offset,
    unsigned long long basis_row_stride,
    unsigned long long basis_col_stride,
    unsigned long long y_offset,
    unsigned long long y_stride) {
    extern __shared__ float sh_strided_c32[];
    float* shared_re = sh_strided_c32;
    float* shared_im = sh_strided_c32 + blockDim.x;
    const unsigned int tid = threadIdx.x;
    const unsigned long long basis_index = blockIdx.x;
    if (basis_index >= basis_count) return;
    const Complex32Kernel* x = basis + basis_offset + basis_index * basis_row_stride;
    float acc_re = 0.0f;
    float acc_im = 0.0f;
    for (unsigned long long i = tid; i < vector_len; i += blockDim.x) {
        const Complex32Kernel value = x[i * basis_col_stride];
        const Complex32Kernel target = y[y_offset + i * y_stride];
        acc_re += value.re * target.re + value.im * target.im;
        acc_im += value.re * target.im - value.im * target.re;
    }
    shared_re[tid] = acc_re;
    shared_im[tid] = acc_im;
    __syncthreads();
    for (unsigned int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_re[tid] += shared_re[tid + stride];
            shared_im[tid] += shared_im[tid + stride];
        }
        __syncthreads();
    }
    if (tid == 0) {
        out[basis_index].re = shared_re[0];
        out[basis_index].im = shared_im[0];
    }
}

extern "C" __global__ void sonium_multi_dot_strided_c64(
    const Complex64Kernel* basis,
    const Complex64Kernel* y,
    Complex64Kernel* out,
    unsigned long long vector_len,
    unsigned long long basis_count,
    unsigned long long basis_offset,
    unsigned long long basis_row_stride,
    unsigned long long basis_col_stride,
    unsigned long long y_offset,
    unsigned long long y_stride) {
    extern __shared__ double sh_strided_c64[];
    double* shared_re = sh_strided_c64;
    double* shared_im = sh_strided_c64 + blockDim.x;
    const unsigned int tid = threadIdx.x;
    const unsigned long long basis_index = blockIdx.x;
    if (basis_index >= basis_count) return;
    const Complex64Kernel* x = basis + basis_offset + basis_index * basis_row_stride;
    double acc_re = 0.0;
    double acc_im = 0.0;
    for (unsigned long long i = tid; i < vector_len; i += blockDim.x) {
        const Complex64Kernel value = x[i * basis_col_stride];
        const Complex64Kernel target = y[y_offset + i * y_stride];
        acc_re += value.re * target.re + value.im * target.im;
        acc_im += value.re * target.im - value.im * target.re;
    }
    shared_re[tid] = acc_re;
    shared_im[tid] = acc_im;
    __syncthreads();
    for (unsigned int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_re[tid] += shared_re[tid + stride];
            shared_im[tid] += shared_im[tid + stride];
        }
        __syncthreads();
    }
    if (tid == 0) {
        out[basis_index].re = shared_re[0];
        out[basis_index].im = shared_im[0];
    }
}

extern "C" __global__ void sonium_basis_axpy_strided_f32(
    const float* basis,
    const float* alphas,
    float* y,
    unsigned long long vector_len,
    unsigned long long basis_count,
    unsigned long long basis_offset,
    unsigned long long basis_row_stride,
    unsigned long long basis_col_stride,
    unsigned long long y_offset,
    unsigned long long y_stride) {
    const unsigned long long i =
        static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= vector_len) {
        return;
    }
    float acc = y[y_offset + i * y_stride];
    for (unsigned long long basis_index = 0; basis_index < basis_count; basis_index++) {
        acc = fmaf(
            alphas[basis_index],
            basis[basis_offset + basis_index * basis_row_stride + i * basis_col_stride],
            acc);
    }
    y[y_offset + i * y_stride] = acc;
}

extern "C" __global__ void sonium_basis_axpy_strided_f64(
    const double* basis,
    const double* alphas,
    double* y,
    unsigned long long vector_len,
    unsigned long long basis_count,
    unsigned long long basis_offset,
    unsigned long long basis_row_stride,
    unsigned long long basis_col_stride,
    unsigned long long y_offset,
    unsigned long long y_stride) {
    const unsigned long long i =
        static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= vector_len) {
        return;
    }
    double acc = y[y_offset + i * y_stride];
    for (unsigned long long basis_index = 0; basis_index < basis_count; basis_index++) {
        acc += alphas[basis_index]
            * basis[basis_offset + basis_index * basis_row_stride + i * basis_col_stride];
    }
    y[y_offset + i * y_stride] = acc;
}

extern "C" __global__ void sonium_basis_axpy_strided_c32(
    const Complex32Kernel* basis,
    const Complex32Kernel* alphas,
    Complex32Kernel* y,
    unsigned long long vector_len,
    unsigned long long basis_count,
    unsigned long long basis_offset,
    unsigned long long basis_row_stride,
    unsigned long long basis_col_stride,
    unsigned long long y_offset,
    unsigned long long y_stride) {
    const unsigned long long i =
        static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= vector_len) {
        return;
    }
    Complex32Kernel target = y[y_offset + i * y_stride];
    float acc_re = target.re;
    float acc_im = target.im;
    for (unsigned long long basis_index = 0; basis_index < basis_count; basis_index++) {
        const Complex32Kernel value =
            basis[basis_offset + basis_index * basis_row_stride + i * basis_col_stride];
        const float scaled_re = alphas[basis_index].re;
        const float scaled_im = alphas[basis_index].im;
        acc_re += scaled_re * value.re - scaled_im * value.im;
        acc_im += scaled_re * value.im + scaled_im * value.re;
    }
    y[y_offset + i * y_stride].re = acc_re;
    y[y_offset + i * y_stride].im = acc_im;
}

extern "C" __global__ void sonium_basis_axpy_strided_c64(
    const Complex64Kernel* basis,
    const Complex64Kernel* alphas,
    Complex64Kernel* y,
    unsigned long long vector_len,
    unsigned long long basis_count,
    unsigned long long basis_offset,
    unsigned long long basis_row_stride,
    unsigned long long basis_col_stride,
    unsigned long long y_offset,
    unsigned long long y_stride) {
    const unsigned long long i =
        static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= vector_len) {
        return;
    }
    Complex64Kernel target = y[y_offset + i * y_stride];
    double acc_re = target.re;
    double acc_im = target.im;
    for (unsigned long long basis_index = 0; basis_index < basis_count; basis_index++) {
        const Complex64Kernel value =
            basis[basis_offset + basis_index * basis_row_stride + i * basis_col_stride];
        const double scaled_re = alphas[basis_index].re;
        const double scaled_im = alphas[basis_index].im;
        acc_re += scaled_re * value.re - scaled_im * value.im;
        acc_im += scaled_re * value.im + scaled_im * value.re;
    }
    y[y_offset + i * y_stride].re = acc_re;
    y[y_offset + i * y_stride].im = acc_im;
}

// Tiled multi-dot variants raise occupancy when basis_count is small (GMRES
// restart sizes): gridDim.x = basis_count * tiles, so tiles thread blocks
// cooperate on each dot. Each block reduces one tile with the same
// shared-memory tree as the strided kernels, then atomically accumulates
// into out[basis_index]. The caller MUST zero the first basis_count outputs
// before launch because the blocks accumulate with atomics. blockDim.x must
// be a power of two (256 in the Rust launch path).
// (Uses the file-top sonium_atomic_add_f64 helper for f64/c64.)
extern "C" __global__ void sonium_multi_dot_tiled_f32(
    const float* basis,
    const float* y,
    float* out,
    unsigned long long vector_len,
    unsigned long long basis_count,
    unsigned long long basis_offset,
    unsigned long long basis_row_stride,
    unsigned long long basis_col_stride,
    unsigned long long y_offset,
    unsigned long long y_stride,
    unsigned long long tiles) {
    extern __shared__ float sh_tiled_f32[];
    const unsigned int tid = threadIdx.x;
    const unsigned long long tile = blockIdx.x % tiles;
    const unsigned long long basis_index = blockIdx.x / tiles;
    if (basis_index >= basis_count) return;
    const float* x = basis + basis_offset + basis_index * basis_row_stride;
    const unsigned long long lane_count = (unsigned long long)blockDim.x * tiles;
    float acc = 0.0f;
    for (unsigned long long i = tile * (unsigned long long)blockDim.x + tid; i < vector_len;
         i += lane_count) {
        acc += x[i * basis_col_stride] * y[y_offset + i * y_stride];
    }
    sh_tiled_f32[tid] = acc;
    __syncthreads();
    for (unsigned int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) sh_tiled_f32[tid] += sh_tiled_f32[tid + stride];
        __syncthreads();
    }
    if (tid == 0) atomicAdd(out + basis_index, sh_tiled_f32[0]);
}

extern "C" __global__ void sonium_multi_dot_tiled_f64(
    const double* basis,
    const double* y,
    double* out,
    unsigned long long vector_len,
    unsigned long long basis_count,
    unsigned long long basis_offset,
    unsigned long long basis_row_stride,
    unsigned long long basis_col_stride,
    unsigned long long y_offset,
    unsigned long long y_stride,
    unsigned long long tiles) {
    extern __shared__ double sh_tiled_f64[];
    const unsigned int tid = threadIdx.x;
    const unsigned long long tile = blockIdx.x % tiles;
    const unsigned long long basis_index = blockIdx.x / tiles;
    if (basis_index >= basis_count) return;
    const double* x = basis + basis_offset + basis_index * basis_row_stride;
    const unsigned long long lane_count = (unsigned long long)blockDim.x * tiles;
    double acc = 0.0;
    for (unsigned long long i = tile * (unsigned long long)blockDim.x + tid; i < vector_len;
         i += lane_count) {
        acc += x[i * basis_col_stride] * y[y_offset + i * y_stride];
    }
    sh_tiled_f64[tid] = acc;
    __syncthreads();
    for (unsigned int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) sh_tiled_f64[tid] += sh_tiled_f64[tid + stride];
        __syncthreads();
    }
    if (tid == 0) sonium_atomic_add_f64(out + basis_index, sh_tiled_f64[0]);
}

extern "C" __global__ void sonium_multi_dot_tiled_c32(
    const Complex32Kernel* basis,
    const Complex32Kernel* y,
    Complex32Kernel* out,
    unsigned long long vector_len,
    unsigned long long basis_count,
    unsigned long long basis_offset,
    unsigned long long basis_row_stride,
    unsigned long long basis_col_stride,
    unsigned long long y_offset,
    unsigned long long y_stride,
    unsigned long long tiles) {
    extern __shared__ float sh_tiled_c32[];
    float* shared_re = sh_tiled_c32;
    float* shared_im = sh_tiled_c32 + blockDim.x;
    const unsigned int tid = threadIdx.x;
    const unsigned long long tile = blockIdx.x % tiles;
    const unsigned long long basis_index = blockIdx.x / tiles;
    if (basis_index >= basis_count) return;
    const Complex32Kernel* x = basis + basis_offset + basis_index * basis_row_stride;
    const unsigned long long lane_count = (unsigned long long)blockDim.x * tiles;
    float acc_re = 0.0f;
    float acc_im = 0.0f;
    for (unsigned long long i = tile * (unsigned long long)blockDim.x + tid; i < vector_len;
         i += lane_count) {
        const Complex32Kernel value = x[i * basis_col_stride];
        const Complex32Kernel target = y[y_offset + i * y_stride];
        acc_re += value.re * target.re + value.im * target.im;
        acc_im += value.re * target.im - value.im * target.re;
    }
    shared_re[tid] = acc_re;
    shared_im[tid] = acc_im;
    __syncthreads();
    for (unsigned int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_re[tid] += shared_re[tid + stride];
            shared_im[tid] += shared_im[tid + stride];
        }
        __syncthreads();
    }
    if (tid == 0) {
        atomicAdd(&(out[basis_index].re), shared_re[0]);
        atomicAdd(&(out[basis_index].im), shared_im[0]);
    }
}

extern "C" __global__ void sonium_multi_dot_tiled_c64(
    const Complex64Kernel* basis,
    const Complex64Kernel* y,
    Complex64Kernel* out,
    unsigned long long vector_len,
    unsigned long long basis_count,
    unsigned long long basis_offset,
    unsigned long long basis_row_stride,
    unsigned long long basis_col_stride,
    unsigned long long y_offset,
    unsigned long long y_stride,
    unsigned long long tiles) {
    extern __shared__ double sh_tiled_c64[];
    double* shared_re = sh_tiled_c64;
    double* shared_im = sh_tiled_c64 + blockDim.x;
    const unsigned int tid = threadIdx.x;
    const unsigned long long tile = blockIdx.x % tiles;
    const unsigned long long basis_index = blockIdx.x / tiles;
    if (basis_index >= basis_count) return;
    const Complex64Kernel* x = basis + basis_offset + basis_index * basis_row_stride;
    const unsigned long long lane_count = (unsigned long long)blockDim.x * tiles;
    double acc_re = 0.0;
    double acc_im = 0.0;
    for (unsigned long long i = tile * (unsigned long long)blockDim.x + tid; i < vector_len;
         i += lane_count) {
        const Complex64Kernel value = x[i * basis_col_stride];
        const Complex64Kernel target = y[y_offset + i * y_stride];
        acc_re += value.re * target.re + value.im * target.im;
        acc_im += value.re * target.im - value.im * target.re;
    }
    shared_re[tid] = acc_re;
    shared_im[tid] = acc_im;
    __syncthreads();
    for (unsigned int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_re[tid] += shared_re[tid + stride];
            shared_im[tid] += shared_im[tid + stride];
        }
        __syncthreads();
    }
    if (tid == 0) {
        sonium_atomic_add_f64(&(out[basis_index].re), shared_re[0]);
        sonium_atomic_add_f64(&(out[basis_index].im), shared_im[0]);
    }
}

extern "C" __global__ void sonium_multi_dot_ptr_tiled_c32(
    const unsigned long long* x_ptrs,
    const Complex32Kernel* y,
    Complex32Kernel* out,
    unsigned long long n,
    unsigned long long tiles) {
    extern __shared__ float sh_ptr_tiled_c32[];
    float* shared_re = sh_ptr_tiled_c32;
    float* shared_im = sh_ptr_tiled_c32 + blockDim.x;
    const unsigned int tid = threadIdx.x;
    const unsigned long long tile = blockIdx.x % tiles;
    const unsigned int basis = (unsigned int)(blockIdx.x / tiles);
    const Complex32Kernel* x = (const Complex32Kernel*)x_ptrs[basis];
    const unsigned long long lane_count = (unsigned long long)blockDim.x * tiles;
    float acc_re = 0.0f;
    float acc_im = 0.0f;
    for (unsigned long long i = tile * (unsigned long long)blockDim.x + tid; i < n;
         i += lane_count) {
        const float xr = x[i].re;
        const float xi = x[i].im;
        const float yr = y[i].re;
        const float yi = y[i].im;
        acc_re += xr * yr + xi * yi;
        acc_im += xr * yi - xi * yr;
    }
    shared_re[tid] = acc_re;
    shared_im[tid] = acc_im;
    __syncthreads();
    for (unsigned int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_re[tid] += shared_re[tid + stride];
            shared_im[tid] += shared_im[tid + stride];
        }
        __syncthreads();
    }
    if (tid == 0) {
        atomicAdd(&(out[basis].re), shared_re[0]);
        atomicAdd(&(out[basis].im), shared_im[0]);
    }
}

extern "C" __global__ void sonium_multi_dot_ptr_tiled_c64(
    const unsigned long long* x_ptrs,
    const Complex64Kernel* y,
    Complex64Kernel* out,
    unsigned long long n,
    unsigned long long tiles) {
    extern __shared__ double sh_ptr_tiled_c64[];
    double* shared_re = sh_ptr_tiled_c64;
    double* shared_im = sh_ptr_tiled_c64 + blockDim.x;
    const unsigned int tid = threadIdx.x;
    const unsigned long long tile = blockIdx.x % tiles;
    const unsigned int basis = (unsigned int)(blockIdx.x / tiles);
    const Complex64Kernel* x = (const Complex64Kernel*)x_ptrs[basis];
    const unsigned long long lane_count = (unsigned long long)blockDim.x * tiles;
    double acc_re = 0.0;
    double acc_im = 0.0;
    for (unsigned long long i = tile * (unsigned long long)blockDim.x + tid; i < n;
         i += lane_count) {
        const double xr = x[i].re;
        const double xi = x[i].im;
        const double yr = y[i].re;
        const double yi = y[i].im;
        acc_re += xr * yr + xi * yi;
        acc_im += xr * yi - xi * yr;
    }
    shared_re[tid] = acc_re;
    shared_im[tid] = acc_im;
    __syncthreads();
    for (unsigned int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_re[tid] += shared_re[tid + stride];
            shared_im[tid] += shared_im[tid + stride];
        }
        __syncthreads();
    }
    if (tid == 0) {
        sonium_atomic_add_f64(&(out[basis].re), shared_re[0]);
        sonium_atomic_add_f64(&(out[basis].im), shared_im[0]);
    }
}
