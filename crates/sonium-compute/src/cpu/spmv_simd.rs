//! CPU CSR row reducers with explicit SIMD where the ISA supports gathers.

#[inline]
pub fn row_f32(col_idx: &[i32], values: &[f32], x: &[f32], start: usize, end: usize) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if !crate::cpu::force_scalar_dispatch() && is_x86_feature_detected!("avx2") {
            return unsafe { x86_64::row_f32_avx2(col_idx, values, x, start, end) };
        }
    }
    row_f32_scalar(col_idx, values, x, start, end)
}

#[inline]
pub fn row_f64(col_idx: &[i32], values: &[f64], x: &[f64], start: usize, end: usize) -> f64 {
    #[cfg(target_arch = "x86_64")]
    {
        if !crate::cpu::force_scalar_dispatch() && is_x86_feature_detected!("avx2") {
            return unsafe { x86_64::row_f64_avx2(col_idx, values, x, start, end) };
        }
    }
    row_f64_scalar(col_idx, values, x, start, end)
}

#[inline]
fn row_f32_scalar(col_idx: &[i32], values: &[f32], x: &[f32], start: usize, end: usize) -> f32 {
    let mut sum0 = 0.0_f32;
    let mut sum1 = 0.0_f32;
    let mut sum2 = 0.0_f32;
    let mut sum3 = 0.0_f32;
    let mut k = start;
    while k + 4 <= end {
        sum0 += values[k] * x[col_idx[k] as usize];
        sum1 += values[k + 1] * x[col_idx[k + 1] as usize];
        sum2 += values[k + 2] * x[col_idx[k + 2] as usize];
        sum3 += values[k + 3] * x[col_idx[k + 3] as usize];
        k += 4;
    }
    let mut acc = (sum0 + sum1) + (sum2 + sum3);
    while k < end {
        acc += values[k] * x[col_idx[k] as usize];
        k += 1;
    }
    acc
}

/// Scalar reference for [`row_f32`], exposed for cross-ISA parity tests.
#[inline]
pub fn row_f32_scalar_reference(
    col_idx: &[i32],
    values: &[f32],
    x: &[f32],
    start: usize,
    end: usize,
) -> f32 {
    row_f32_scalar(col_idx, values, x, start, end)
}

#[inline]
fn row_f64_scalar(col_idx: &[i32], values: &[f64], x: &[f64], start: usize, end: usize) -> f64 {
    let mut sum0 = 0.0_f64;
    let mut sum1 = 0.0_f64;
    let mut sum2 = 0.0_f64;
    let mut sum3 = 0.0_f64;
    let mut k = start;
    while k + 4 <= end {
        sum0 += values[k] * x[col_idx[k] as usize];
        sum1 += values[k + 1] * x[col_idx[k + 1] as usize];
        sum2 += values[k + 2] * x[col_idx[k + 2] as usize];
        sum3 += values[k + 3] * x[col_idx[k + 3] as usize];
        k += 4;
    }
    let mut acc = (sum0 + sum1) + (sum2 + sum3);
    while k < end {
        acc += values[k] * x[col_idx[k] as usize];
        k += 1;
    }
    acc
}

/// Scalar reference for [`row_f64`], exposed for cross-ISA parity tests.
#[inline]
pub fn row_f64_scalar_reference(
    col_idx: &[i32],
    values: &[f64],
    x: &[f64],
    start: usize,
    end: usize,
) -> f64 {
    row_f64_scalar(col_idx, values, x, start, end)
}

#[cfg(target_arch = "x86_64")]
mod x86_64 {
    use std::arch::x86_64::*;

    #[target_feature(enable = "avx2")]
    pub unsafe fn row_f32_avx2(
        col_idx: &[i32],
        values: &[f32],
        x: &[f32],
        start: usize,
        end: usize,
    ) -> f32 {
        let mut acc = _mm256_setzero_ps();
        let mut k = start;
        while k + 8 <= end {
            let idx = unsafe { _mm256_loadu_si256(col_idx.as_ptr().add(k).cast::<__m256i>()) };
            let xv = unsafe { _mm256_i32gather_ps(x.as_ptr(), idx, 4) };
            let av = unsafe { _mm256_loadu_ps(values.as_ptr().add(k)) };
            acc = _mm256_add_ps(acc, _mm256_mul_ps(av, xv));
            k += 8;
        }

        let mut lanes = [0.0_f32; 8];
        unsafe { _mm256_storeu_ps(lanes.as_mut_ptr(), acc) };
        let mut sum = lanes.iter().sum::<f32>();
        while k < end {
            sum += values[k] * x[col_idx[k] as usize];
            k += 1;
        }
        sum
    }

    #[target_feature(enable = "avx2")]
    pub unsafe fn row_f64_avx2(
        col_idx: &[i32],
        values: &[f64],
        x: &[f64],
        start: usize,
        end: usize,
    ) -> f64 {
        let mut acc = _mm256_setzero_pd();
        let mut k = start;
        while k + 4 <= end {
            let idx = unsafe { _mm_loadu_si128(col_idx.as_ptr().add(k).cast::<__m128i>()) };
            let xv = unsafe { _mm256_i32gather_pd(x.as_ptr(), idx, 8) };
            let av = unsafe { _mm256_loadu_pd(values.as_ptr().add(k)) };
            acc = _mm256_add_pd(acc, _mm256_mul_pd(av, xv));
            k += 4;
        }

        let mut lanes = [0.0_f64; 4];
        unsafe { _mm256_storeu_pd(lanes.as_mut_ptr(), acc) };
        let mut sum = lanes.iter().sum::<f64>();
        while k < end {
            sum += values[k] * x[col_idx[k] as usize];
            k += 1;
        }
        sum
    }
}
