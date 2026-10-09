//! Dense nonsymmetric eigenvalues via the Francis-QR algorithm.
//!
//! [`nonsymmetric_eigenvalues`] (and its `f32` twin
//! [`nonsymmetric_eigenvalues_f32`]) returns all eigenvalues of a small real
//! square matrix, including complex-conjugate pairs. The pipeline is the
//! classical EISPACK sequence: diagonal balancing (exact power-of-two
//! similarities), Householder reduction to upper Hessenberg form, then
//! Francis double-shift QR sweeps with deflation. Only eigenvalues are
//! computed; eigenvectors would need the accumulated transformations.
//!
//! Values come back unordered, with each complex pair stored as adjacent
//! conjugates `(re + im·i, re - im·i)`. Intended for reference-sized
//! problems such as ESPRIT rotation operators (`p ≈ 2·tones`); large
//! sparse spectra belong to the Arnoldi/Lanczos paths.

use num_complex::{Complex, Complex32, Complex64};
use num_traits::{Float, NumAssign};
use sonium_solvers_core::dense::DenseMatrix;
use sonium_solvers_core::traits::{SolverError, SolverStatus};

/// Maximum Francis-QR sweeps spent on one eigenvalue.
///
/// EISPACK `hqr` allows 30 iterations per root before reporting failure;
/// well-scaled small matrices converge in a handful.
const MAX_SWEEPS_PER_ROOT: usize = 30;

/// Sweep counts that trigger an ad-hoc exceptional shift.
///
/// After 10 (then 20) sweeps without deflation, EISPACK `hqr` shifts the
/// active block by its trailing diagonal and continues with a Rayleigh
/// quotient shift built from the trailing subdiagonals. This breaks the
/// rare non-converging cycles of the plain Francis shift.
const EXCEPTIONAL_SHIFT_FIRST: usize = 10;
/// Second exceptional-shift trigger; see [`EXCEPTIONAL_SHIFT_FIRST`].
const EXCEPTIONAL_SHIFT_SECOND: usize = 20;

/// Maximum full balancing passes over the matrix rows.
///
/// Scale-only balancing always preserves the spectrum exactly (power-of-two
/// similarities with flushed-to-zero rejections), so the cap only bounds
/// work, never correctness.
const MAX_BALANCE_PASSES: usize = 8;

/// Largest binary exponent used by one balancing scale factor.
///
/// `2⁶⁰` rebalances row/column norm ratios up to `~10³⁶` while keeping the
/// tentative scale itself far from overflow; entries that would still go
/// non-finite or flush to zero reject the scale.
const MAX_BALANCE_EXPONENT: i32 = 60;

/// Compute all eigenvalues of a real square matrix.
///
/// Runs the balancing/Hessenberg/Francis-QR pipeline in `f64` and returns
/// `n` eigenvalues for an `n x n` input. Real eigenvalues have zero
/// imaginary part; complex ones appear as adjacent conjugate pairs.
///
/// # Examples
///
/// A plane rotation has the conjugate pair `e^(±iθ)`:
///
/// ```rust
/// use sonium_solvers_core::dense::DenseMatrix;
/// use sonium_solvers_eigen::nonsymmetric_eigenvalues;
///
/// let theta = 0.7_f64;
/// let rotation = DenseMatrix::from_row_major_vec(
///     2,
///     2,
///     vec![theta.cos(), -theta.sin(), theta.sin(), theta.cos()],
/// );
/// let values = nonsymmetric_eigenvalues(&rotation).unwrap();
/// assert!((values[0].re - theta.cos()).abs() < 1.0e-13);
/// assert!((values[0].im.abs() - theta.sin()).abs() < 1.0e-13);
/// assert_eq!(values[1], values[0].conj());
/// ```
///
/// # Errors
///
/// Returns [`SolverError::DimensionMismatch`] for non-square input,
/// [`SolverError::OperationFailed`] for non-finite entries or an internal
/// non-finite sweep, and [`SolverError::ConvergenceError`] when the sweep
/// budget ([`MAX_SWEEPS_PER_ROOT`]) per root is exhausted.
pub fn nonsymmetric_eigenvalues(a: &DenseMatrix<f64>) -> Result<Vec<Complex64>, SolverError> {
    francis_qr_values(a)
}

/// Compute all eigenvalues of a real square matrix in `f32`.
///
/// Same contract as [`nonsymmetric_eigenvalues`], evaluated natively in
/// single precision.
///
/// # Errors
///
/// Same variants as [`nonsymmetric_eigenvalues`].
pub fn nonsymmetric_eigenvalues_f32(a: &DenseMatrix<f32>) -> Result<Vec<Complex32>, SolverError> {
    francis_qr_values(a)
}

/// Balancing/Hessenberg/Francis-QR pipeline over a real float type.
fn francis_qr_values<T: Float + NumAssign>(
    a: &DenseMatrix<T>,
) -> Result<Vec<Complex<T>>, SolverError> {
    let n = a.nrows();
    if a.ncols() != n {
        return Err(SolverError::DimensionMismatch {
            expected: n,
            got: a.ncols(),
        });
    }
    if a.as_slice().iter().any(|value| !value.is_finite()) {
        return Err(SolverError::OperationFailed {
            operation: "nonsymmetric eigensolver input",
            reason: "input contains NaN or infinity".to_string(),
        });
    }
    let mut h = a.as_slice().to_vec();
    balance(&mut h, n);
    to_hessenberg(&mut h, n)?;
    francis_sweep(&mut h, n)
}

/// Rebalance row/column norms by exact power-of-two similarities.
///
/// For each row `i` with off-diagonal row norm `r` and column norm `c`,
/// scales row `i` by `2^-k` and column `i` by `2^k` with `k` near
/// `½·log₂(r/c)`, the exponent that equalizes the pair. A scale applies
/// only when it strictly reduces `r + c` and keeps every touched entry
/// finite and unflushed, so balancing never perturbs the spectrum.
fn balance<T: Float + NumAssign>(h: &mut [T], n: usize) {
    let one = T::one();
    let two = one + one;
    let mut scaled_row = vec![T::zero(); n];
    let mut scaled_column = vec![T::zero(); n];
    for _ in 0..MAX_BALANCE_PASSES {
        let mut changed = false;
        for i in 0..n {
            let mut row_norm = T::zero();
            let mut column_norm = T::zero();
            for j in 0..n {
                if j != i {
                    row_norm += h[i * n + j].abs();
                    column_norm += h[j * n + i].abs();
                }
            }
            if row_norm == T::zero() || column_norm == T::zero() {
                continue;
            }
            let sum = row_norm + column_norm;
            if !sum.is_finite() {
                continue;
            }
            let exponent = ((row_norm.log2() - column_norm.log2()) / two)
                .round()
                .to_i32()
                .unwrap_or(0)
                .clamp(-MAX_BALANCE_EXPONENT, MAX_BALANCE_EXPONENT);
            if exponent == 0 {
                continue;
            }
            let factor = two.powi(exponent);
            let inverse = one / factor;
            let mut applicable = true;
            for j in 0..n {
                scaled_row[j] = h[i * n + j] * inverse;
                scaled_column[j] = h[j * n + i] * factor;
                if j != i {
                    // Reject flushed entries: an exact similarity must not
                    // turn a nonzero entry into zero.
                    if !scaled_row[j].is_finite() || !scaled_column[j].is_finite() {
                        applicable = false;
                        break;
                    }
                    if scaled_row[j] == T::zero() && h[i * n + j] != T::zero() {
                        applicable = false;
                        break;
                    }
                    if scaled_column[j] == T::zero() && h[j * n + i] != T::zero() {
                        applicable = false;
                        break;
                    }
                }
            }
            if !applicable {
                continue;
            }
            let mut scaled_sum = T::zero();
            for j in 0..n {
                if j != i {
                    scaled_sum += scaled_row[j].abs() + scaled_column[j].abs();
                }
            }
            if scaled_sum < sum {
                let diagonal = h[i * n + i];
                for j in 0..n {
                    h[i * n + j] = scaled_row[j];
                    h[j * n + i] = scaled_column[j];
                }
                // Row and column are inverse similarities, so the pivot is
                // unchanged; restore it (it was written twice above).
                h[i * n + i] = diagonal;
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }
}

/// Reduce a square matrix to upper Hessenberg form in place.
///
/// Householder reflectors normalized by the column norm keep every
/// intermediate within a small factor of the input scale: no squared norms
/// appear, so huge-but-finite inputs cannot overflow the reduction.
fn to_hessenberg<T: Float + NumAssign>(h: &mut [T], n: usize) -> Result<(), SolverError> {
    if n <= 2 {
        return Ok(());
    }
    let one = T::one();
    let mut reflector = vec![T::zero(); n];
    let mut work = vec![T::zero(); n];
    for k in 0..n - 2 {
        let mut norm = T::zero();
        for i in k + 1..n {
            norm = norm.hypot(h[i * n + k]);
        }
        if norm == T::zero() {
            continue;
        }
        let alpha = h[(k + 1) * n + k];
        // Reflect onto -sign(alpha)·norm·e₁: v₀ = alpha + sign·norm.
        let sign = if alpha >= T::zero() { one } else { -one };
        let first = alpha / norm + sign;
        if first == T::zero() {
            // Column already aligned with e₁; nothing to eliminate.
            continue;
        }
        for i in k + 1..n {
            reflector[i] = h[i * n + k] / norm;
        }
        reflector[k + 1] = first;
        let mut squared = T::zero();
        for value in &reflector[k + 1..n] {
            squared += *value * *value;
        }
        let tau = (one + one) / squared;
        // Apply from the left: rows k+1..n, columns k..n.
        for j in k..n {
            let mut dot = T::zero();
            for i in k + 1..n {
                dot += reflector[i] * h[i * n + j];
            }
            work[j] = dot * tau;
        }
        for j in k..n {
            let update = work[j];
            for i in k + 1..n {
                h[i * n + j] -= reflector[i] * update;
            }
        }
        // Apply from the right: all rows, columns k+1..n.
        for i in 0..n {
            let mut dot = T::zero();
            for j in k + 1..n {
                dot += h[i * n + j] * reflector[j];
            }
            work[i] = dot * tau;
        }
        for i in 0..n {
            let update = work[i];
            for j in k + 1..n {
                h[i * n + j] -= update * reflector[j];
            }
        }
        h[(k + 1) * n + k] = -sign * norm;
        for i in k + 2..n {
            h[i * n + k] = T::zero();
        }
    }
    if h.iter().any(|value| !value.is_finite()) {
        return Err(SolverError::Breakdown {
            operation: "Hessenberg reduction",
            reason: "reduction generated NaN or infinity".to_string(),
        });
    }
    Ok(())
}

/// Francis double-shift QR sweeps on an upper Hessenberg matrix.
///
/// Follows EISPACK `hqr`: each inner iteration scans for a negligible
/// subdiagonal (deflating 1×1 real or 2×2 real/complex blocks) or performs
/// one implicit double-shift bulge chase. Complex pairs land in adjacent
/// slots as conjugates. Indices below are 0-based; `en` is the active
/// block size (rows/columns `0..en`).
fn francis_sweep<T: Float + NumAssign>(
    h: &mut [T],
    n: usize,
) -> Result<Vec<Complex<T>>, SolverError> {
    // Trapezoidal norm of the Hessenberg matrix: the deflation yardstick.
    let mut norm = T::zero();
    for i in 0..n {
        let start = i.saturating_sub(1);
        for j in start..n {
            norm += h[i * n + j].abs();
        }
    }
    let mut real = vec![T::zero(); n];
    let mut imag = vec![T::zero(); n];
    let mut shift = T::zero();
    let mut en = n;
    while en > 0 {
        let mut sweeps = 0;
        loop {
            if !(0..en).all(|i| (0..en).all(|j| h[i * n + j].is_finite())) {
                return Err(SolverError::Breakdown {
                    operation: "Francis-QR sweep",
                    reason: "sweep generated NaN or infinity".to_string(),
                });
            }
            let last = en - 1;
            // Scan for a negligible subdiagonal from the bottom up.
            let mut l = 0;
            for candidate in (0..en).rev() {
                if candidate == 0 {
                    l = 0;
                    break;
                }
                let scale = h[(candidate - 1) * n + candidate - 1].abs()
                    + h[candidate * n + candidate].abs();
                let scale = if scale == T::zero() { norm } else { scale };
                if scale + h[candidate * n + candidate - 1].abs() == scale {
                    l = candidate;
                    h[candidate * n + candidate - 1] = T::zero();
                    break;
                }
            }
            let mut x = h[last * n + last];
            if l == last {
                real[last] = x + shift;
                en -= 1;
                break;
            }
            // Reaching past the 1×1 branch with en ≤ 2 always deflates, so
            // `na` and the exceptional-shift `na - 1` below are in range.
            let na = last - 1;
            let mut y = h[na * n + na];
            let mut w = h[last * n + na] * h[na * n + last];
            if l == na {
                // Trailing 2×2 block: real pair or complex pair from the
                // characteristic equation, cancellation-free in the real case.
                let p = (y - x) / (T::one() + T::one());
                let q = p * p + w;
                let mut z = q.abs().sqrt();
                x += shift;
                if q < T::zero() {
                    real[na] = x + p;
                    real[last] = x + p;
                    imag[na] = z;
                    imag[last] = -z;
                } else {
                    if p < T::zero() {
                        z = -z;
                    }
                    z += p;
                    real[na] = x + z;
                    real[last] = x + z;
                    if z != T::zero() {
                        real[last] = x - w / z;
                    }
                }
                en -= 2;
                break;
            }
            if sweeps == MAX_SWEEPS_PER_ROOT {
                return Err(SolverError::ConvergenceError {
                    status: SolverStatus::MaxIterationsReached,
                    iterations: sweeps,
                    residual: h[last * n + na].abs().to_f64().unwrap_or(f64::NAN),
                });
            }
            if sweeps == EXCEPTIONAL_SHIFT_FIRST || sweeps == EXCEPTIONAL_SHIFT_SECOND {
                // Ad-hoc shift: recenter the block, then chase with shifts
                // built from the trailing subdiagonals (exact dyadic
                // rationals 3/4 and 7/16).
                shift += x;
                for i in 0..en {
                    h[i * n + i] -= x;
                }
                let one = T::one();
                let two = one + one;
                let three = two + one;
                let four = two + two;
                let seven = four + two + one;
                let sixteen = four * four;
                let s = h[last * n + na].abs() + h[na * n + na - 1].abs();
                x = s * three / four;
                y = x;
                w = -(s * s) * seven / sixteen;
            }
            sweeps += 1;
            double_shift_sweep(h, n, l, en, x, y, w);
        }
    }
    Ok(real
        .into_iter()
        .zip(imag)
        .map(|(re, im)| Complex::new(re, im))
        .collect())
}

/// One implicit double-shift bulge chase over rows/columns `l..en`.
///
/// Forms the shift vector from the trailing-shift polynomial at the first
/// non-negligible subdiagonal `m`, then chases the resulting 3×3 bulge down
/// the subdiagonal with Householder reflectors applied as similarities,
/// preserving Hessenberg form. Requires `en ≥ 3` with all subdiagonals in
/// `(l, en)` non-negligible, which the deflation scan guarantees.
fn double_shift_sweep<T: Float + NumAssign>(
    h: &mut [T],
    n: usize,
    l: usize,
    en: usize,
    x: T,
    y: T,
    w: T,
) {
    // Shift-vector formation: scan m from en-3 down to l.
    let mut m = l;
    let (mut p, mut q, mut r) = (T::zero(), T::zero(), T::zero());
    for candidate in (l..=(en - 3)).rev() {
        m = candidate;
        let diagonal = h[m * n + m];
        let row = x - diagonal;
        let column = y - diagonal;
        p = (row * column - w) / h[(m + 1) * n + m] + h[m * n + m + 1];
        q = h[(m + 1) * n + m + 1] - diagonal - row - column;
        r = h[(m + 2) * n + m + 1];
        let scale = p.abs() + q.abs() + r.abs();
        if scale == T::zero() {
            // Degenerate shift vector (measure-zero): any nonzero seed keeps
            // the chase a well-defined orthogonal similarity.
            p = T::one();
        } else {
            p /= scale;
            q /= scale;
            r /= scale;
        }
        if m == l {
            break;
        }
        let test = p.abs()
            * (h[(m - 1) * n + m - 1].abs() + diagonal.abs() + h[(m + 1) * n + m + 1].abs());
        if test + h[m * n + m - 1].abs() * (q.abs() + r.abs()) == test {
            break;
        }
    }
    for i in m + 2..en {
        h[i * n + i - 2] = T::zero();
        if i != m + 2 {
            h[i * n + i - 3] = T::zero();
        }
    }
    // Bulge chase: reflector k acts on rows/columns k..k+2 (last step k..k+1).
    let na = en - 2;
    for k in m..=na {
        let not_last = k != na;
        let mut scale = T::one();
        if k != m {
            p = h[k * n + k - 1];
            q = h[(k + 1) * n + k - 1];
            r = if not_last {
                h[(k + 2) * n + k - 1]
            } else {
                T::zero()
            };
            scale = p.abs() + q.abs() + r.abs();
            if scale == T::zero() {
                continue;
            }
            p /= scale;
            q /= scale;
            r /= scale;
        }
        let mut s = (p * p + q * q + r * r).sqrt();
        if p < T::zero() {
            s = -s;
        }
        if k != m {
            h[k * n + k - 1] = -s * scale;
        } else if l != m {
            h[k * n + k - 1] = -h[k * n + k - 1];
        }
        p += s;
        let reflector_x = p / s;
        let reflector_y = q / s;
        let reflector_z = r / s;
        q /= p;
        r /= p;
        // Apply from the left: rows k..k+2, columns k..en.
        for j in k..en {
            let mut dot = h[k * n + j] + q * h[(k + 1) * n + j];
            if not_last {
                dot += r * h[(k + 2) * n + j];
                h[(k + 2) * n + j] -= dot * reflector_z;
            }
            h[(k + 1) * n + j] -= dot * reflector_y;
            h[k * n + j] -= dot * reflector_x;
        }
        // Apply from the right: rows l.., columns k..k+2.
        let row_end = en.min(k + 4);
        for i in l..row_end {
            let mut dot = reflector_x * h[i * n + k] + reflector_y * h[i * n + k + 1];
            if not_last {
                dot += reflector_z * h[i * n + k + 2];
                h[i * n + k + 2] -= dot * r;
            }
            h[i * n + k + 1] -= dot * q;
            h[i * n + k] -= dot;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use sonium_solvers_direct::dense_svd;

    fn sorted_by_real_then_imag(mut values: Vec<Complex64>) -> Vec<Complex64> {
        values.sort_by(|left, right| {
            left.re
                .total_cmp(&right.re)
                .then(left.im.total_cmp(&right.im))
        });
        values
    }

    #[test]
    fn rotation_block_gives_conjugate_pair() {
        let theta = 0.7_f64;
        let rotation = DenseMatrix::from_row_major_vec(
            2,
            2,
            vec![theta.cos(), -theta.sin(), theta.sin(), theta.cos()],
        );
        let values = nonsymmetric_eigenvalues(&rotation).unwrap();
        assert_eq!(values.len(), 2);
        assert!((values[0].re - theta.cos()).abs() < 1.0e-13);
        assert!((values[0].im.abs() - theta.sin()).abs() < 1.0e-13);
        // Complex pairs are adjacent conjugates.
        assert_eq!(values[1], values[0].conj());
    }

    #[test]
    fn upper_triangular_returns_diagonal() {
        let triangular = DenseMatrix::from_row_major_vec(
            4,
            4,
            vec![
                3.0, 1.0, -2.0, 0.5, //
                0.0, -1.0, 4.0, 1.5, //
                0.0, 0.0, 2.0, -0.5, //
                0.0, 0.0, 0.0, 2.0,
            ],
        );
        let values = sorted_by_real_then_imag(nonsymmetric_eigenvalues(&triangular).unwrap());
        let expected = [-1.0, 2.0, 2.0, 3.0];
        for (value, want) in values.iter().zip(expected) {
            assert!(
                (value.re - want).abs() < 1.0e-12,
                "value={value} want={want}"
            );
            assert!(value.im.abs() < 1.0e-12);
        }
    }

    #[test]
    fn companion_matrix_recovers_known_roots() {
        // Companion of (x - 1)(x - 2)(x + 0.5) = x³ - 2.5x² + 0.5x + 1.
        let companion = DenseMatrix::from_row_major_vec(
            3,
            3,
            vec![0.0, 0.0, -1.0, 1.0, 0.0, -0.5, 0.0, 1.0, 2.5],
        );
        let values = sorted_by_real_then_imag(nonsymmetric_eigenvalues(&companion).unwrap());
        for (value, want) in values.iter().zip([-0.5, 1.0, 2.0]) {
            assert!(
                (value.re - want).abs() < 1.0e-12,
                "value={value} want={want}"
            );
            assert!(value.im.abs() < 1.0e-12);
        }
        // Product of eigenvalues equals the determinant (-1 here).
        let product: Complex64 = values.iter().copied().product();
        assert!((product.re + 1.0).abs() < 1.0e-11);
        assert!(product.im.abs() < 1.0e-11);
        // Independent residual oracle: each real root singularizes A - λI.
        for root in [-0.5, 1.0, 2.0] {
            let shifted = DenseMatrix::from_fn(3, 3, |i, j| {
                companion[[i, j]] - if i == j { root } else { 0.0 }
            });
            let svd = dense_svd(&shifted).unwrap();
            let smallest = svd.singular_values.last().copied().unwrap_or(f64::NAN);
            assert!(smallest < 1.0e-9, "root={root} sigma_min={smallest}");
        }
    }

    #[test]
    fn symmetric_input_matches_symmetric_eigen() {
        let symmetric = DenseMatrix::from_row_major_vec(
            5,
            5,
            vec![
                4.0, 1.0, -0.5, 0.25, 0.0, //
                1.0, 3.0, 1.5, -0.25, 0.5, //
                -0.5, 1.5, 2.0, 1.0, -0.5, //
                0.25, -0.25, 1.0, 5.0, 1.25, //
                0.0, 0.5, -0.5, 1.25, 3.5,
            ],
        );
        let (mut reference, _) = crate::lobpcg::symmetric_eigen(&symmetric, 5).unwrap();
        reference.sort_by(|left, right| left.total_cmp(right));
        let values = sorted_by_real_then_imag(nonsymmetric_eigenvalues(&symmetric).unwrap());
        for (value, want) in values.iter().zip(reference) {
            assert!(
                (value.re - want).abs() < 1.0e-9,
                "value={value} want={want}"
            );
            assert!(value.im.abs() < 1.0e-9);
        }
    }

    #[test]
    fn fixed_nonsymmetric_preserves_trace_and_pairs() {
        // Deterministic pseudo-random 6x6 with no special structure.
        let matrix = DenseMatrix::from_fn(6, 6, |i, j| {
            let k = (i * 6 + j) as u64;
            let h = k
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407)
                >> 11;
            (h % 2000) as f64 / 1000.0 - 1.0
        });
        let trace: f64 = (0..6).map(|i| matrix[[i, i]]).sum();
        let values = nonsymmetric_eigenvalues(&matrix).unwrap();
        assert_eq!(values.len(), 6);
        // Trace identity: Σλ = trace(A).
        let sum: Complex64 = values.iter().copied().sum();
        assert!((sum.re - trace).abs() < 1.0e-8, "sum={sum} trace={trace}");
        assert!(sum.im.abs() < 1.0e-8);
        // Non-real eigenvalues come in conjugate pairs.
        for value in &values {
            if value.im.abs() > 1.0e-9 {
                assert!(
                    values
                        .iter()
                        .any(|other| (other.re - value.re).abs() < 1.0e-9
                            && (other.im + value.im).abs() < 1.0e-9),
                    "unpaired {value}"
                );
            }
        }
    }

    #[test]
    fn jordan_block_returns_zero_spectrum() {
        let jordan = DenseMatrix::from_row_major_vec(
            4,
            4,
            vec![
                0.0, 1.0, 0.0, 0.0, //
                0.0, 0.0, 1.0, 0.0, //
                0.0, 0.0, 0.0, 1.0, //
                0.0, 0.0, 0.0, 0.0,
            ],
        );
        let values = nonsymmetric_eigenvalues(&jordan).unwrap();
        for value in &values {
            assert!(value.norm() < 1.0e-12, "value={value}");
        }
    }

    #[test]
    fn scaled_similarity_keeps_spectrum() {
        // diag(R(0.7), 2) under D·A·D⁻¹ with D = diag(1e3, 1, 1e-3): entry
        // magnitudes spread over 1e6, exercising the balancing stage.
        let theta = 0.7_f64;
        let cos = theta.cos();
        let sin = theta.sin();
        let scales = [1.0e3, 1.0, 1.0e-3];
        let base = [cos, -sin, 0.0, sin, cos, 0.0, 0.0, 0.0, 2.0];
        let matrix = DenseMatrix::from_fn(3, 3, |i, j| base[i * 3 + j] * scales[i] / scales[j]);
        let values = nonsymmetric_eigenvalues(&matrix).unwrap();
        assert_eq!(values.len(), 3);
        let pair: Vec<_> = values.iter().filter(|v| v.im.abs() > 1.0e-9).collect();
        let real: Vec<_> = values.iter().filter(|v| v.im.abs() <= 1.0e-9).collect();
        assert_eq!(pair.len(), 2);
        assert_eq!(real.len(), 1);
        for value in pair {
            assert!((value.re - cos).abs() < 1.0e-9, "value={value}");
            assert!((value.im.abs() - sin).abs() < 1.0e-9, "value={value}");
        }
        assert!((real[0].re - 2.0).abs() < 1.0e-9);
    }

    #[test]
    fn rejects_non_square_and_non_finite() {
        let wide = DenseMatrix::from_row_major_vec(2, 3, vec![0.0; 6]);
        assert!(matches!(
            nonsymmetric_eigenvalues(&wide),
            Err(SolverError::DimensionMismatch { .. })
        ));
        let nan = DenseMatrix::from_row_major_vec(2, 2, vec![0.0, f64::NAN, 0.0, 0.0]);
        assert!(matches!(
            nonsymmetric_eigenvalues(&nan),
            Err(SolverError::OperationFailed { .. })
        ));
    }

    #[test]
    fn scalar_and_empty_cases() {
        let one = DenseMatrix::from_row_major_vec(1, 1, vec![2.5_f64]);
        let values = nonsymmetric_eigenvalues(&one).unwrap();
        assert_eq!(values, vec![Complex64::new(2.5, 0.0)]);
        let empty = DenseMatrix::from_row_major_vec(0, 0, vec![]);
        assert!(nonsymmetric_eigenvalues(&empty).unwrap().is_empty());
    }

    #[test]
    fn f32_rotation_matches_single_precision() {
        let theta = 0.7_f32;
        let rotation = DenseMatrix::from_row_major_vec(
            2,
            2,
            vec![theta.cos(), -theta.sin(), theta.sin(), theta.cos()],
        );
        let values = nonsymmetric_eigenvalues_f32(&rotation).unwrap();
        assert_eq!(values.len(), 2);
        assert!((values[0].re - theta.cos()).abs() < 1.0e-5);
        assert!((values[0].im.abs() - theta.sin()).abs() < 1.0e-5);
        assert_eq!(values[1], values[0].conj());
    }
}
