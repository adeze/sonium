// NVRTC-compiled f64 explicit-dynamics kernels (Phase 1 lane 1E).
//
// Determinism contract: each kernel assigns one output lane per thread and
// accumulates that lane sequentially, reproducing the serial CPU kernels in
// `explicit_scatter.rs` / `explicit_dt.rs` bit-exactly. Host code validates
// index bounds, value ranges, and u32 size limits before upload; violating
// those contracts is a host-side programming error, never silently clamped
// here.

// Packed gather: out[slot * width + c] = field[indices[slot] * width + c].
// A pure copy, so device and CPU results agree bit-exactly by construction.
extern "C" __global__ void sonium_explicit_gather_f64(
    const double* field, const int* indices, double* out,
    unsigned width, unsigned count) {
    unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned total = count * width;
    if (idx >= total) return;
    unsigned slot = idx / width;
    unsigned c = idx % width;
    unsigned node = (unsigned)indices[slot];
    out[idx] = field[node * width + c];
}

// Deterministic segmented scatter-add: thread (node, c) owns output lane
// target[node * width + c] and folds that node's slots in ascending slot
// order onto the lane's prior value, matching the serial CPU accumulation
// order exactly (bit-exact). `offsets`/`slots` form host-built CSR over the
// contribution list: node n owns slots[offsets[n] .. offsets[n + 1]].
extern "C" __global__ void sonium_explicit_scatter_add_f64(
    const double* values, const int* slots, const int* offsets,
    double* target, unsigned width, unsigned nodes) {
    unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned total = nodes * width;
    if (idx >= total) return;
    unsigned node = idx / width;
    unsigned c = idx % width;
    int start = offsets[node];
    int end = offsets[node + 1];
    double acc = target[node * width + c];
    for (int i = start; i < end; ++i) {
        unsigned slot = (unsigned)slots[i];
        acc += values[slot * width + c];
    }
    target[node * width + c] = acc;
}

// Block-partial minima of length/speed quotients for critical-dt. Emits one
// partial per block; the host folds the partials with f64::min. Out-of-range
// threads reuse element n - 1 (harmless for a minimum), so the kernel needs
// no infinity literal and every partial holds a real quotient. Requires a
// power-of-two block size; the host always launches 256 threads per block.
extern "C" __global__ void sonium_explicit_min_dt_block_f64(
    const double* lengths, const double* speeds, double* partials, unsigned n) {
    extern __shared__ double tile[];
    unsigned tid = threadIdx.x;
    unsigned i = blockIdx.x * blockDim.x + tid;
    if (i >= n) i = n - 1;
    tile[tid] = lengths[i] / speeds[i];
    __syncthreads();
    for (unsigned stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            double other = tile[tid + stride];
            if (other < tile[tid]) tile[tid] = other;
        }
        __syncthreads();
    }
    if (tid == 0) partials[blockIdx.x] = tile[0];
}

// Batched HEXA8 hourglass forces: thread (e, i) owns element e's i-th force
// component lane and evaluates f[i, I] = coeff * sum_a Gamma[a, I] *
// sum_J Gamma[a, J] * rates[i, J] with the exact serial CPU accumulation
// order (modes outer, nodes inner, then modes inner per node), so device
// and CPU agree bit-exactly. Gamma is mode-major (32 doubles/element),
// rates and forces node-major (24 doubles/element).
extern "C" __global__ void sonium_explicit_hourglass_f64(
    const double* gamma, const double* rates, double* forces,
    double coeff, unsigned elements) {
    unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned total = elements * 3u;
    if (idx >= total) return;
    unsigned e = idx / 3u;
    unsigned i = idx % 3u;
    const double* g = gamma + (unsigned long long)e * 32u;
    const double* r = rates + (unsigned long long)e * 24u;
    double* f = forces + (unsigned long long)e * 24u;
    double modal[4];
    for (unsigned a = 0; a < 4; ++a) {
        double sum = 0.0;
        for (unsigned node = 0; node < 8; ++node)
            sum += g[a * 8u + node] * r[3u * node + i];
        modal[a] = sum;
    }
    for (unsigned node = 0; node < 8; ++node) {
        double force = 0.0;
        for (unsigned a = 0; a < 4; ++a)
            force += g[a * 8u + node] * modal[a];
        f[3u * node + i] = coeff * force;
    }
}
