// CUDA pre/post-processing kernels for real FFT derived transforms.
//
// The FFT itself is delegated to cuFFT.  These kernels build the even/odd
// length-2N extension and fold the phase rotation into the extraction pass,
// leaving only two custom launches around one cached cuFFT plan.  `kind`
// follows sonium_compute::RealTransformKind:
//   0 DCT-II, 1 DCT-III, 2 DCT-II adjoint, 3 DCT-III adjoint,
//   4 DST-II, 5 DST-III, 6 DST-II adjoint, 7 DST-III adjoint.

extern "C" __global__ void sonium_real_transform_pack_f32(
    const float* input, float2* work, unsigned n, unsigned batch, unsigned kind) {
    unsigned index = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned width = 2u * n;
    unsigned total = width * batch;
    if (index >= total) return;
    unsigned q = index % width;
    unsigned b = index / width;
    unsigned input_base = b * n;

    // DCT-II, DST-II, and their adjoints use an even/odd extension of the
    // real input.  The adjoint extraction applies the normalization later.
    if (kind == 0u || kind == 3u || kind == 4u || kind == 7u) {
        unsigned source = q < n ? q : width - 1u - q;
        float value = input[input_base + source];
        if ((kind == 4u || kind == 7u) && q >= n) value = -value;
        work[index] = make_float2(value, 0.0f);
        return;
    }

    // DCT-III and DST-III are inverse length-2N FFTs of phase-shifted,
    // conjugate-symmetric spectra.  The adjoint variants alter only the
    // coefficient weighting before the same inverse FFT.
    const float pi = 3.14159265358979323846f;
    if (kind == 1u || kind == 2u) {
        if (q == 0u) {
            work[index] = make_float2(input[input_base], 0.0f);
        } else if (q < n) {
            float scale = kind == 2u ? 0.5f : 1.0f;
            float theta = pi * (float)q / (2.0f * (float)n);
            float s, c;
            sincosf(theta, &s, &c);
            float value = input[input_base + q] * scale;
            work[index] = make_float2(value * c, value * s);
        } else if (q == n) {
            work[index] = make_float2(0.0f, 0.0f);
        } else {
            unsigned mirror = width - q;
            float scale = kind == 2u ? 0.5f : 1.0f;
            float theta = pi * (float)mirror / (2.0f * (float)n);
            float s, c;
            sincosf(theta, &s, &c);
            float value = input[input_base + mirror] * scale;
            work[index] = make_float2(value * c, -value * s);
        }
        return;
    }

    // DST-III has frequency q=1..N.  The Nyquist term is self-conjugate and
    // is placed on the imaginary axis so the extraction pass obtains its
    // alternating real sequence.  DST-II adjoint preweights the rows by the
    // inverse of the DST-III normalization.
    if (q == 0u) {
        work[index] = make_float2(0.0f, 0.0f);
    } else if (q < n) {
        float scale = kind == 6u ? 0.5f * (float)n : 1.0f;
        float theta = pi * (float)q / (2.0f * (float)n);
        float s, c;
        sincosf(theta, &s, &c);
        float value = input[input_base + q - 1u] * scale;
        work[index] = make_float2(value * c, value * s);
    } else if (q == n) {
        float scale = kind == 6u ? (float)n : 1.0f;
        work[index] = make_float2(0.0f, input[input_base + n - 1u] * scale);
    } else {
        unsigned mirror = width - q;
        float scale = kind == 6u ? 0.5f * (float)n : 1.0f;
        float theta = pi * (float)mirror / (2.0f * (float)n);
        float s, c;
        sincosf(theta, &s, &c);
        float value = input[input_base + mirror - 1u] * scale;
        work[index] = make_float2(-value * c, value * s);
    }
}

extern "C" __global__ void sonium_real_transform_extract_f32(
    const float2* work, float* output, unsigned n, unsigned batch, unsigned kind) {
    unsigned index = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned total = n * batch;
    if (index >= total) return;
    unsigned k = index % n;
    unsigned b = index / n;
    unsigned work_index = (kind == 4u || kind == 7u) ? k + 1u : k;
    const float2 value = work[b * (2u * n) + work_index];
    const float pi = 3.14159265358979323846f;
    float result;

    if (kind == 0u || kind == 3u) {
        float theta = pi * (float)k / (2.0f * (float)n);
        float s, c;
        sincosf(theta, &s, &c);
        result = 0.5f * (value.x * c + value.y * s);
        if (kind == 3u) result *= k == 0u ? 1.0f / (float)n : 2.0f / (float)n;
    } else if (kind == 4u || kind == 7u) {
        unsigned q = k + 1u;
        float theta = pi * (float)q / (2.0f * (float)n);
        float s, c;
        sincosf(theta, &s, &c);
        result = 0.5f * (value.x * s - value.y * c);
        if (kind == 7u) result *= q == n ? 1.0f / (float)n : 2.0f / (float)n;
    } else if (kind == 1u || kind == 2u) {
        result = value.x / (kind == 1u ? (float)n : 1.0f);
    } else {
        result = value.y / (float)n;
    }
    output[index] = result;
}

extern "C" __global__ void sonium_real_transform_pack_f64(
    const double* input, double2* work, unsigned n, unsigned batch, unsigned kind) {
    unsigned index = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned width = 2u * n;
    unsigned total = width * batch;
    if (index >= total) return;
    unsigned q = index % width;
    unsigned b = index / width;
    unsigned input_base = b * n;

    if (kind == 0u || kind == 3u || kind == 4u || kind == 7u) {
        unsigned source = q < n ? q : width - 1u - q;
        double value = input[input_base + source];
        if ((kind == 4u || kind == 7u) && q >= n) value = -value;
        work[index] = make_double2(value, 0.0);
        return;
    }

    const double pi = 3.141592653589793238462643383279502884;
    if (kind == 1u || kind == 2u) {
        if (q == 0u) {
            work[index] = make_double2(input[input_base], 0.0);
        } else if (q < n) {
            double scale = kind == 2u ? 0.5 : 1.0;
            double theta = pi * (double)q / (2.0 * (double)n);
            double s, c;
            sincos(theta, &s, &c);
            double value = input[input_base + q] * scale;
            work[index] = make_double2(value * c, value * s);
        } else if (q == n) {
            work[index] = make_double2(0.0, 0.0);
        } else {
            unsigned mirror = width - q;
            double scale = kind == 2u ? 0.5 : 1.0;
            double theta = pi * (double)mirror / (2.0 * (double)n);
            double s, c;
            sincos(theta, &s, &c);
            double value = input[input_base + mirror] * scale;
            work[index] = make_double2(value * c, -value * s);
        }
        return;
    }

    if (q == 0u) {
        work[index] = make_double2(0.0, 0.0);
    } else if (q < n) {
        double scale = kind == 6u ? 0.5 * (double)n : 1.0;
        double theta = pi * (double)q / (2.0 * (double)n);
        double s, c;
        sincos(theta, &s, &c);
        double value = input[input_base + q - 1u] * scale;
        work[index] = make_double2(value * c, value * s);
    } else if (q == n) {
        double scale = kind == 6u ? (double)n : 1.0;
        work[index] = make_double2(0.0, input[input_base + n - 1u] * scale);
    } else {
        unsigned mirror = width - q;
        double scale = kind == 6u ? 0.5 * (double)n : 1.0;
        double theta = pi * (double)mirror / (2.0 * (double)n);
        double s, c;
        sincos(theta, &s, &c);
        double value = input[input_base + mirror - 1u] * scale;
        work[index] = make_double2(-value * c, value * s);
    }
}

extern "C" __global__ void sonium_real_transform_extract_f64(
    const double2* work, double* output, unsigned n, unsigned batch, unsigned kind) {
    unsigned index = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned total = n * batch;
    if (index >= total) return;
    unsigned k = index % n;
    unsigned b = index / n;
    unsigned work_index = (kind == 4u || kind == 7u) ? k + 1u : k;
    const double2 value = work[b * (2u * n) + work_index];
    const double pi = 3.141592653589793238462643383279502884;
    double result;

    if (kind == 0u || kind == 3u) {
        double theta = pi * (double)k / (2.0 * (double)n);
        double s, c;
        sincos(theta, &s, &c);
        result = 0.5 * (value.x * c + value.y * s);
        if (kind == 3u) result *= k == 0u ? 1.0 / (double)n : 2.0 / (double)n;
    } else if (kind == 4u || kind == 7u) {
        unsigned q = k + 1u;
        double theta = pi * (double)q / (2.0 * (double)n);
        double s, c;
        sincos(theta, &s, &c);
        result = 0.5 * (value.x * s - value.y * c);
        if (kind == 7u) result *= q == n ? 1.0 / (double)n : 2.0 / (double)n;
    } else if (kind == 1u || kind == 2u) {
        result = value.x / (kind == 1u ? (double)n : 1.0);
    } else {
        result = value.y / (double)n;
    }
    output[index] = result;
}
