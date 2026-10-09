// Native CUDA kernels for element-local tensor contractions and segmented
// assembly reductions. Each output owns one deterministic accumulation lane.

extern "C" __global__ void sonium_segmented_sum_f32(
    const float* values, const int* offsets, float* output, unsigned segments) {
    unsigned segment = blockIdx.x * blockDim.x + threadIdx.x;
    if (segment >= segments) return;
    int start = offsets[segment];
    int end = offsets[segment + 1];
    float sum = 0.0f;
    for (int i = start; i < end; ++i) sum += values[i];
    output[segment] = sum;
}

extern "C" __global__ void sonium_segmented_sum_f64(
    const double* values, const int* offsets, double* output, unsigned segments) {
    unsigned segment = blockIdx.x * blockDim.x + threadIdx.x;
    if (segment >= segments) return;
    int start = offsets[segment];
    int end = offsets[segment + 1];
    double sum = 0.0;
    for (int i = start; i < end; ++i) sum += values[i];
    output[segment] = sum;
}

extern "C" __global__ void sonium_segmented_sum_c32(
    const float2* values, const int* offsets, float2* output, unsigned segments) {
    unsigned segment = blockIdx.x * blockDim.x + threadIdx.x;
    if (segment >= segments) return;
    int start = offsets[segment];
    int end = offsets[segment + 1];
    float2 sum = make_float2(0.0f, 0.0f);
    for (int i = start; i < end; ++i) {
        sum.x += values[i].x;
        sum.y += values[i].y;
    }
    output[segment] = sum;
}

extern "C" __global__ void sonium_segmented_sum_c64(
    const double2* values, const int* offsets, double2* output, unsigned segments) {
    unsigned segment = blockIdx.x * blockDim.x + threadIdx.x;
    if (segment >= segments) return;
    int start = offsets[segment];
    int end = offsets[segment + 1];
    double2 sum = make_double2(0.0, 0.0);
    for (int i = start; i < end; ++i) {
        sum.x += values[i].x;
        sum.y += values[i].y;
    }
    output[segment] = sum;
}

extern "C" __global__ void sonium_batched_local_contract_f32(
    const float* lhs, const float* rhs, float* output,
    unsigned batches, unsigned rows, unsigned inner, unsigned cols) {
    unsigned index = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned total = batches * rows * cols;
    if (index >= total) return;
    unsigned col = index % cols;
    unsigned row = (index / cols) % rows;
    unsigned batch = index / (rows * cols);
    unsigned lhs_base = (batch * rows + row) * inner;
    unsigned rhs_base = batch * inner * cols + col;
    float sum = 0.0f;
    for (unsigned k = 0; k < inner; ++k)
        sum += lhs[lhs_base + k] * rhs[rhs_base + k * cols];
    output[index] = sum;
}

extern "C" __global__ void sonium_batched_local_contract_f64(
    const double* lhs, const double* rhs, double* output,
    unsigned batches, unsigned rows, unsigned inner, unsigned cols) {
    unsigned index = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned total = batches * rows * cols;
    if (index >= total) return;
    unsigned col = index % cols;
    unsigned row = (index / cols) % rows;
    unsigned batch = index / (rows * cols);
    unsigned lhs_base = (batch * rows + row) * inner;
    unsigned rhs_base = batch * inner * cols + col;
    double sum = 0.0;
    for (unsigned k = 0; k < inner; ++k)
        sum += lhs[lhs_base + k] * rhs[rhs_base + k * cols];
    output[index] = sum;
}

extern "C" __global__ void sonium_batched_local_contract_c32(
    const float2* lhs, const float2* rhs, float2* output,
    unsigned batches, unsigned rows, unsigned inner, unsigned cols) {
    unsigned index = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned total = batches * rows * cols;
    if (index >= total) return;
    unsigned col = index % cols;
    unsigned row = (index / cols) % rows;
    unsigned batch = index / (rows * cols);
    unsigned lhs_base = (batch * rows + row) * inner;
    unsigned rhs_base = batch * inner * cols + col;
    float2 sum = make_float2(0.0f, 0.0f);
    for (unsigned k = 0; k < inner; ++k) {
        float2 a = lhs[lhs_base + k];
        float2 b = rhs[rhs_base + k * cols];
        sum.x += a.x * b.x - a.y * b.y;
        sum.y += a.x * b.y + a.y * b.x;
    }
    output[index] = sum;
}

// One Gauss-Seidel SOR sweep over a single color subset: thread `t` owns
// row `i = color_rows[t]` (same-color rows share no edges, so updates are
// independent). Mirrors the host gs_color_sweep_host semantics: missing
// diagonal defaults to 1, updates skipped when |a_ii| <= 1e-15.
extern "C" __global__ void sonium_gs_color_sweep_f32(
    const int* row_ptr, const int* col_idx, const float* values,
    const int* color_rows, const float* b, float* x, float omega, unsigned count) {
    unsigned t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= count) return;
    int i = color_rows[t];
    float diag = 1.0f;
    float sum = b[i];
    for (int k = row_ptr[i]; k < row_ptr[i + 1]; ++k) {
        int j = col_idx[k];
        if (j == i) diag = values[k];
        else sum -= values[k] * x[j];
    }
    if (diag > 1e-15f || diag < -1e-15f) {
        float new_val = sum / diag;
        x[i] = (1.0f - omega) * x[i] + omega * new_val;
    }
}

extern "C" __global__ void sonium_gs_color_sweep_f64(
    const int* row_ptr, const int* col_idx, const double* values,
    const int* color_rows, const double* b, double* x, double omega, unsigned count) {
    unsigned t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= count) return;
    int i = color_rows[t];
    double diag = 1.0;
    double sum = b[i];
    for (int k = row_ptr[i]; k < row_ptr[i + 1]; ++k) {
        int j = col_idx[k];
        if (j == i) diag = values[k];
        else sum -= values[k] * x[j];
    }
    if (diag > 1e-15 || diag < -1e-15) {
        double new_val = sum / diag;
        x[i] = (1.0 - omega) * x[i] + omega * new_val;
    }
}

extern "C" __global__ void sonium_gs_color_sweep_c32(
    const int* row_ptr, const int* col_idx, const float2* values,
    const int* color_rows, const float2* b, float2* x, float omega, unsigned count) {
    unsigned t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= count) return;
    int i = color_rows[t];
    float2 diag = make_float2(1.0f, 0.0f);
    float2 sum = b[i];
    for (int k = row_ptr[i]; k < row_ptr[i + 1]; ++k) {
        int j = col_idx[k];
        if (j == i) {
            diag = values[k];
        } else {
            float2 a = values[k];
            float2 xj = x[j];
            sum.x -= a.x * xj.x - a.y * xj.y;
            sum.y -= a.x * xj.y + a.y * xj.x;
        }
    }
    float mag2 = diag.x * diag.x + diag.y * diag.y;
    if (mag2 > 1e-30f) {
        float2 n;
        n.x = (sum.x * diag.x + sum.y * diag.y) / mag2;
        n.y = (sum.y * diag.x - sum.x * diag.y) / mag2;
        float2 old = x[i];
        x[i].x = (1.0f - omega) * old.x + omega * n.x;
        x[i].y = (1.0f - omega) * old.y + omega * n.y;
    }
}

extern "C" __global__ void sonium_gs_color_sweep_c64(
    const int* row_ptr, const int* col_idx, const double2* values,
    const int* color_rows, const double2* b, double2* x, double omega, unsigned count) {
    unsigned t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= count) return;
    int i = color_rows[t];
    double2 diag = make_double2(1.0, 0.0);
    double2 sum = b[i];
    for (int k = row_ptr[i]; k < row_ptr[i + 1]; ++k) {
        int j = col_idx[k];
        if (j == i) {
            diag = values[k];
        } else {
            double2 a = values[k];
            double2 xj = x[j];
            sum.x -= a.x * xj.x - a.y * xj.y;
            sum.y -= a.x * xj.y + a.y * xj.x;
        }
    }
    double mag2 = diag.x * diag.x + diag.y * diag.y;
    if (mag2 > 1e-30) {
        double2 n;
        n.x = (sum.x * diag.x + sum.y * diag.y) / mag2;
        n.y = (sum.y * diag.x - sum.x * diag.y) / mag2;
        double2 old = x[i];
        x[i].x = (1.0 - omega) * old.x + omega * n.x;
        x[i].y = (1.0 - omega) * old.y + omega * n.y;
    }
}

extern "C" __global__ void sonium_batched_local_contract_c64(
    const double2* lhs, const double2* rhs, double2* output,
    unsigned batches, unsigned rows, unsigned inner, unsigned cols) {
    unsigned index = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned total = batches * rows * cols;
    if (index >= total) return;
    unsigned col = index % cols;
    unsigned row = (index / cols) % rows;
    unsigned batch = index / (rows * cols);
    unsigned lhs_base = (batch * rows + row) * inner;
    unsigned rhs_base = batch * inner * cols + col;
    double2 sum = make_double2(0.0, 0.0);
    for (unsigned k = 0; k < inner; ++k) {
        double2 a = lhs[lhs_base + k];
        double2 b = rhs[rhs_base + k * cols];
        sum.x += a.x * b.x - a.y * b.y;
        sum.y += a.x * b.y + a.y * b.x;
    }
    output[index] = sum;
}
