//! `HelmholtzDomain` — owns the scaled scattering potential, Laplace kernels,
//! propagator, and operator scale used by the Richardson iteration.
//!
//! Single-domain (no decomposition). Mirrors `wavesim_py.helmholtzdomain.Helmholtz`.

use num_complex::Complex64;
use rayon::prelude::*;
use sonium_fft::rustfft_compat::FftPlanner;
use std::fmt;

use crate::array::{Array1, Array2, Array3, array1_from_elem, array1_zeros};
use crate::parallelism::{POINTWISE_CELLS_PER_THREAD, use_parallel};

use super::boundary::add_absorbing_boundary;
use super::engine::FftPlanCache;

/// Pointwise complex multiply, serially or with rayon over contiguous
/// output. Both paths visit elements in the same order with identical
/// operations; only the thread assignment differs.
fn pointwise_multiply(data: &mut [Complex64], rhs: &[Complex64], parallel: bool) {
    debug_assert_eq!(data.len(), rhs.len());
    if parallel {
        data.par_iter_mut()
            .zip(rhs.par_iter())
            .for_each(|(o, b)| *o *= *b);
    } else {
        for (o, b) in data.iter_mut().zip(rhs.iter()) {
            *o *= *b;
        }
    }
}

/// Pointwise propagator divide `data[off] /= kx[i] + ky[j] + kz[k]`,
/// serially or with rayon over contiguous `i` slabs.
///
/// The non-degeneracy contract lives with [`HelmholtzDomain::try_new`];
/// debug builds re-assert it per mode on both paths.
fn pointwise_divide(
    data: &mut [Complex64],
    kernel_x: &[Complex64],
    kernel_y: &[Complex64],
    kernel_z: &[Complex64],
    shape: (usize, usize, usize),
    parallel: bool,
) {
    let (nx, ny, nz) = shape;
    debug_assert_eq!(data.len(), nx * ny * nz);
    if parallel {
        data.par_chunks_mut(ny * nz)
            .enumerate()
            .for_each(|(i, slab)| {
                divide_slab(slab, kernel_x[i], kernel_y, kernel_z, i, ny, nz);
            });
    } else {
        for (i, slab) in data.chunks_mut(ny * nz).enumerate() {
            divide_slab(slab, kernel_x[i], kernel_y, kernel_z, i, ny, nz);
        }
    }
}

fn divide_slab(
    slab: &mut [Complex64],
    kx: Complex64,
    kernel_y: &[Complex64],
    kernel_z: &[Complex64],
    i: usize,
    ny: usize,
    nz: usize,
) {
    for j in 0..ny {
        let ky = kernel_y[j];
        for k in 0..nz {
            let denom = kx + ky + kernel_z[k];
            debug_assert!(
                denom.norm() > 0.0,
                "WaveSim propagator mode ({i},{j},{k}) is degenerate (kx+ky+kz == 0)"
            );
            slab[j * nz + k] /= denom;
        }
    }
}

/// Invalid input to [`HelmholtzDomain::try_new`].
///
/// WaveSim is a numerical preview engine rather than a normative analytical
/// reference.  Its checked constructor nevertheless rejects malformed media
/// before they can become non-finite FFT operators or panic while padding.
#[derive(Debug, Clone, PartialEq)]
pub enum HelmholtzDomainError {
    /// A scalar parameter is non-finite or outside the stated model domain.
    InvalidParameter {
        /// Parameter name.
        field: &'static str,
        /// Required domain or invariant.
        requirement: &'static str,
    },
    /// A physical grid axis has zero extent.
    EmptyAxis {
        /// Zero-length axis index.
        axis: usize,
    },
    /// Padding one axis would overflow an allocation dimension.
    PaddingOverflow {
        /// Axis whose padded length overflowed.
        axis: usize,
    },
    /// The padded grid volume cannot be represented by `usize`.
    GridVolumeOverflow,
    /// A complex permittivity entry contains NaN or infinity.
    NonFinitePermittivity {
        /// Flat C-order entry index.
        index: usize,
    },
    /// The medium has gain, which the modified Born-series operator excludes.
    GainPermittivity {
        /// Flat C-order entry index.
        index: usize,
    },
    /// The scattering and wrapping norms vanish, so no operator scale exists.
    DegenerateOperator,
    /// An intermediate operator coefficient is not finite.
    NonFiniteOperator,
}

impl fmt::Display for HelmholtzDomainError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidParameter { field, requirement } => {
                write!(f, "WaveSim parameter {field} must be {requirement}")
            }
            Self::EmptyAxis { axis } => write!(f, "WaveSim physical grid axis {axis} is empty"),
            Self::PaddingOverflow { axis } => {
                write!(f, "WaveSim boundary padding overflows axis {axis}")
            }
            Self::GridVolumeOverflow => write!(f, "WaveSim padded grid volume overflows usize"),
            Self::NonFinitePermittivity { index } => {
                write!(f, "WaveSim permittivity entry {index} is non-finite")
            }
            Self::GainPermittivity { index } => write!(
                f,
                "WaveSim permittivity entry {index} has negative imaginary part (gain)"
            ),
            Self::DegenerateOperator => write!(
                f,
                "WaveSim scattering and wrapping norms vanish; no operator scale exists"
            ),
            Self::NonFiniteOperator => write!(f, "WaveSim derived operator is non-finite"),
        }
    }
}

impl std::error::Error for HelmholtzDomainError {}

/// Pre-built Helmholtz operator on a regular grid.
pub struct HelmholtzDomain {
    /// Scattering operator B = 1 − scale·(V_raw − shift), pointwise multiplier.
    pub b_scat: Array3<Complex64>,
    /// Per-axis Fourier-space Laplace kernel scaled by `scale`, with shift+1
    /// added to axis 0 only. The sum along all three axes equals (L+1) in
    /// k-space.
    pub kernel_x: Array1<Complex64>,
    pub kernel_y: Array1<Complex64>,
    pub kernel_z: Array1<Complex64>,
    /// Operator scale `c = -0.95i / (‖V_scat‖ + ‖V_wrap‖)`.
    pub scale: Complex64,
    /// Pre-shift constant `c·shift_old + 1` (diagnostics only).
    pub shift_const: Complex64,
    /// Padded shape = physical shape + boundary pad.
    padded_shape: (usize, usize, usize),
    /// FFT plan cache.
    pub fft: FftPlanCache,
}

impl HelmholtzDomain {
    pub fn padded_shape(&self) -> (usize, usize, usize) {
        self.padded_shape
    }

    /// Build the domain from physical (un-padded) permittivity and parameters.
    ///
    /// # Panics
    ///
    /// Panics on invalid input (non-sub-Nyquist sampling, gain media,
    /// degenerate operator, …). Prefer [`try_new`](Self::try_new), which
    /// returns a [`HelmholtzDomainError`] instead, where inputs are
    /// user-controlled.
    pub fn new(
        permittivity: &Array3<Complex64>,
        wavelength: f64,
        pixel_size: f64,
        boundary_pad: [(usize, usize); 3],
        periodic: [bool; 3],
    ) -> Self {
        Self::try_new(permittivity, wavelength, pixel_size, boundary_pad, periodic)
            .expect("invalid WaveSim Helmholtz-domain input")
    }

    /// Build a domain after validating the physical grid and modified
    /// Born-series assumptions.
    pub fn try_new(
        permittivity: &Array3<Complex64>,
        wavelength: f64,
        pixel_size: f64,
        boundary_pad: [(usize, usize); 3],
        periodic: [bool; 3],
    ) -> Result<Self, HelmholtzDomainError> {
        if !wavelength.is_finite() || wavelength <= 0.0 {
            return Err(HelmholtzDomainError::InvalidParameter {
                field: "wavelength",
                requirement: "finite and positive",
            });
        }
        if !pixel_size.is_finite() || pixel_size <= 0.0 {
            return Err(HelmholtzDomainError::InvalidParameter {
                field: "pixel_size",
                requirement: "finite and positive",
            });
        }
        if pixel_size >= wavelength / 2.0 {
            return Err(HelmholtzDomainError::InvalidParameter {
                field: "pixel_size",
                requirement: "strictly below wavelength / 2 (Nyquist)",
            });
        }
        let k0 = std::f64::consts::TAU / wavelength;
        let k02 = k0 * k0;
        if !k0.is_finite() || !k02.is_finite() {
            return Err(HelmholtzDomainError::InvalidParameter {
                field: "wavelength",
                requirement: "large enough to yield a finite wave number squared",
            });
        }

        let physical_shape = permittivity.shape();
        let physical_axes = [physical_shape.0, physical_shape.1, physical_shape.2];
        let mut padded_axes = [0usize; 3];
        for axis in 0..3 {
            if physical_axes[axis] == 0 {
                return Err(HelmholtzDomainError::EmptyAxis { axis });
            }
            padded_axes[axis] = physical_axes[axis]
                .checked_add(boundary_pad[axis].0)
                .and_then(|length| length.checked_add(boundary_pad[axis].1))
                .ok_or(HelmholtzDomainError::PaddingOverflow { axis })?;
        }
        padded_axes[0]
            .checked_mul(padded_axes[1])
            .and_then(|length| length.checked_mul(padded_axes[2]))
            .ok_or(HelmholtzDomainError::GridVolumeOverflow)?;
        for (index, value) in permittivity.as_slice().iter().enumerate() {
            if !value.re.is_finite() || !value.im.is_finite() {
                return Err(HelmholtzDomainError::NonFinitePermittivity { index });
            }
            if value.im < -1.0e-9 {
                return Err(HelmholtzDomainError::GainPermittivity { index });
            }
        }

        // 1. Pad permittivity with absorbing boundary.
        let mut eps = add_absorbing_boundary(permittivity, boundary_pad);
        let padded_shape = eps.shape();

        // 2. V_raw = k0² · ε.
        for v in eps.as_mut_slice().iter_mut() {
            *v *= k02;
        }

        // 3. Optimal shift (enclosing-circle center) and v_scat_norm (radius).
        let (shift_old, v_scat_norm) = enclosing_circle(&eps);

        // 4. Raw Laplace kernels per axis (one shared planner).
        let mut planner = FftPlanner::<f64>::new();
        let mut k_x =
            laplace_kernel_axis_with_planner(padded_shape.0, periodic[0], pixel_size, &mut planner);
        let mut k_y =
            laplace_kernel_axis_with_planner(padded_shape.1, periodic[1], pixel_size, &mut planner);
        let mut k_z =
            laplace_kernel_axis_with_planner(padded_shape.2, periodic[2], pixel_size, &mut planner);

        // 5. V_wrap norm (single-domain): correction count = number of non-periodic axes.
        let mut correction_count: usize = 0;
        let mut max_pad = 0;
        for axis in 0..3 {
            if !periodic[axis] {
                correction_count += 1;
                max_pad = max_pad.max(boundary_pad[axis].0).max(boundary_pad[axis].1);
            }
        }
        let v_wrap_norm = if max_pad == 0 || correction_count == 0 {
            0.0
        } else {
            let v_wrap = make_wrap_matrix(pixel_size, max_pad);
            spectral_norm(&v_wrap) * correction_count as f64
        };

        // 6. Scale.
        let operator_norm = v_scat_norm + v_wrap_norm;
        if !operator_norm.is_finite() {
            return Err(HelmholtzDomainError::NonFiniteOperator);
        }
        if operator_norm <= 0.0 {
            return Err(HelmholtzDomainError::DegenerateOperator);
        }
        let scale = Complex64::new(0.0, -0.95) / Complex64::new(operator_norm, 0.0);
        if !scale.re.is_finite() || !scale.im.is_finite() {
            return Err(HelmholtzDomainError::NonFiniteOperator);
        }

        // 7. shift_new = scale·shift_old + 1.
        let shift_const = scale * shift_old + Complex64::new(1.0, 0.0);
        if !shift_const.re.is_finite() || !shift_const.im.is_finite() {
            return Err(HelmholtzDomainError::NonFiniteOperator);
        }

        // 8. B_scat = -scale·V_raw + shift_const = 1 - scale·(V_raw − shift_old).
        let mut b_scat = eps.clone();
        for v in b_scat.as_mut_slice().iter_mut() {
            *v = -scale * *v + shift_const;
        }

        // 9. Scale and shift kernels: k_d → scale·k_d (+ shift_const on axis 0 only).
        for v in k_x.iter_mut() {
            *v = scale * *v + shift_const;
        }
        for v in k_y.iter_mut() {
            *v *= scale;
        }
        for v in k_z.iter_mut() {
            *v *= scale;
        }

        if b_scat
            .as_slice()
            .iter()
            .chain(k_x.iter())
            .chain(k_y.iter())
            .chain(k_z.iter())
            .any(|value| !value.re.is_finite() || !value.im.is_finite())
        {
            return Err(HelmholtzDomainError::NonFiniteOperator);
        }

        Ok(Self {
            b_scat,
            kernel_x: k_x,
            kernel_y: k_y,
            kernel_z: k_z,
            scale,
            shift_const,
            padded_shape,
            fft: FftPlanCache::new(),
        })
    }

    /// Apply the medium operator: `out = B_scat ⊙ x` (pointwise multiply).
    pub fn medium(&self, x: &Array3<Complex64>) -> Array3<Complex64> {
        let mut out = x.clone();
        let parallel = use_parallel(out.len(), POINTWISE_CELLS_PER_THREAD);
        pointwise_multiply(out.as_mut_slice(), self.b_scat.as_slice(), parallel);
        out
    }

    /// In-place medium application: `x ← B_scat ⊙ x`.
    pub fn medium_inplace(&self, x: &mut Array3<Complex64>) {
        let parallel = use_parallel(x.len(), POINTWISE_CELLS_PER_THREAD);
        pointwise_multiply(x.as_mut_slice(), self.b_scat.as_slice(), parallel);
    }

    /// Apply the propagator `(L+1)⁻¹ x` via FFT.
    pub fn propagator(&self, x: &Array3<Complex64>) -> Array3<Complex64> {
        let mut buf = x.clone();
        self.propagator_inplace(&mut buf);
        buf
    }

    /// Apply the propagator `(L+1)⁻¹` in place via FFT, pointwise divide, IFFT.
    ///
    /// Every mode is divided by `denom = kx + ky + kz`. [`try_new`](Self::try_new)
    /// constructs the shift so this sum stays nonzero (it rejects degenerate
    /// operators with [`HelmholtzDomainError::DegenerateOperator`]); a zero
    /// here would silently blow up that mode, so debug builds assert
    /// non-degeneracy per mode instead of dividing blindly. Note `kernel_*`
    /// are public: mutating them after construction voids that guarantee.
    pub fn propagator_inplace(&self, buf: &mut Array3<Complex64>) {
        self.fft.fftn(buf);
        let (nx, ny, nz) = self.padded_shape;
        let parallel = use_parallel(buf.len(), POINTWISE_CELLS_PER_THREAD);
        pointwise_divide(
            buf.as_mut_slice(),
            self.kernel_x.as_slice(),
            self.kernel_y.as_slice(),
            self.kernel_z.as_slice(),
            (nx, ny, nz),
            parallel,
        );
        self.fft.ifftn(buf);
    }
}

/// Build the unscaled Laplace kernel along a single axis in Fourier space.
pub fn laplace_kernel_axis(length: usize, periodic: bool, pixel_size: f64) -> Array1<Complex64> {
    let mut planner = FftPlanner::<f64>::new();
    laplace_kernel_axis_with_planner(length, periodic, pixel_size, &mut planner)
}

/// [`laplace_kernel_axis`] with a caller-provided FFT planner.
///
/// Domain construction needs three axis kernels; sharing one planner across
/// them (and across repeated constructions) avoids rebuilding the cached
/// twiddle tables per axis.
pub fn laplace_kernel_axis_with_planner(
    length: usize,
    periodic: bool,
    pixel_size: f64,
    planner: &mut FftPlanner<f64>,
) -> Array1<Complex64> {
    if length == 1 {
        return array1_zeros(1);
    }

    if periodic {
        // -(2π f / Δx)² on the FFT lattice (no factor of L because fftfreq with d=Δx/(2π) returns rad/length).
        let mut out: Array1<Complex64> = array1_zeros(length);
        for (i, slot) in out.iter_mut().enumerate() {
            let f_idx = if i <= length / 2 {
                i as f64
            } else {
                i as f64 - length as f64
            };
            let omega = f_idx * std::f64::consts::TAU / (pixel_size * length as f64);
            *slot = Complex64::new(-omega * omega, 0.0);
        }
        out
    } else {
        let real_kernel = laplace_kernel_1d_real_space(pixel_size, length);
        let mut buf: Vec<Complex64> = real_kernel
            .iter()
            .map(|&r| Complex64::new(r, 0.0))
            .collect();
        let fft = planner.plan_fft_forward(length);
        fft.process(&mut buf);
        // Real part only; zero DC.
        let mut out: Array1<Complex64> = array1_zeros(length);
        for (i, c) in buf.iter().enumerate() {
            out[i] = Complex64::new(c.re, 0.0);
        }
        out[0] = Complex64::new(0.0, 0.0);
        out
    }
}

/// Real-space truncated sinc-derivative Laplace kernel.
///
/// Direct port of `wavesim_py.utilities.laplace_kernel_1d`.
pub fn laplace_kernel_1d_real_space(pixel_size: f64, length: usize) -> Array1<f64> {
    if length == 1 {
        return array1_zeros(1);
    }

    // x = π · fftfreq(L) · L = π · {0, 1, …, L/2-1, -L/2, …, -1}
    let mut x: Array1<f64> = array1_zeros(length);
    for (i, slot) in x.iter_mut().enumerate() {
        let k = if i < length.div_ceil(2) {
            i as f64
        } else {
            i as f64 - length as f64
        };
        *slot = std::f64::consts::PI * k;
    }

    let mut kernel: Array1<f64> = array1_zeros(length);
    for (xi, slot) in x.iter().zip(kernel.iter_mut()) {
        *slot = if *xi == 0.0 {
            1.0 / 3.0
        } else {
            2.0 * xi.cos() / (xi * xi)
        };
    }
    let factor = -std::f64::consts::PI * std::f64::consts::PI / (pixel_size * pixel_size);
    for v in kernel.iter_mut() {
        *v *= factor;
    }

    let total: f64 = kernel.iter().sum();
    if length.is_multiple_of(2) {
        let mid = length / 2;
        kernel[mid] -= total;
    } else {
        let lo = length / 2;
        kernel[lo] -= 0.5 * total;
        kernel[lo + 1] -= 0.5 * total;
    }
    kernel
}

/// Build the n×n wrap-correction matrix used to estimate ‖V_wrap‖.
pub fn make_wrap_matrix(pixel_size: f64, n_boundary: usize) -> Array2<f64> {
    if n_boundary == 0 {
        return Array2::<f64>::zeros(0, 0);
    }
    let kernel = laplace_kernel_1d_real_space(pixel_size, 6 * n_boundary + 1);
    let mut m = Array2::<f64>::zeros(n_boundary, n_boundary);
    for r in 0..n_boundary {
        for c in 0..n_boundary {
            m[[r, c]] = kernel[n_boundary - r + c];
        }
    }
    m
}

/// `A·v` for a row-major `HostMatrix<f64>` and `&[f64]` vector.
fn matvec(a: &Array2<f64>, v: &[f64]) -> Array1<f64> {
    let (rows, cols) = a.shape();
    assert_eq!(cols, v.len());
    let mut out: Array1<f64> = array1_zeros(rows);
    let data = a.as_slice();
    for (r, slot) in out.iter_mut().enumerate() {
        let base = r * cols;
        let mut s = 0.0;
        for c in 0..cols {
            s += data[base + c] * v[c];
        }
        *slot = s;
    }
    out
}

/// `Aᵀ·v` for a row-major matrix.
fn matvec_transposed(a: &Array2<f64>, v: &[f64]) -> Array1<f64> {
    let (rows, cols) = a.shape();
    assert_eq!(rows, v.len());
    let mut out: Array1<f64> = array1_zeros(cols);
    let data = a.as_slice();
    for (r, &vr) in v.iter().enumerate() {
        let base = r * cols;
        for c in 0..cols {
            out[c] += data[base + c] * vr;
        }
    }
    out
}

#[inline]
fn dot(a: &[f64], b: &[f64]) -> f64 {
    debug_assert_eq!(a.len(), b.len());
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// Spectral norm via power iteration on `Aᵀ A`. Adequate for our small matrices.
fn spectral_norm(a: &Array2<f64>) -> f64 {
    let n = a.ncols();
    if n == 0 {
        return 0.0;
    }
    let mut v: Array1<f64> = array1_from_elem(n, 1.0 / (n as f64).sqrt());
    let mut sigma = 0.0;
    for _ in 0..400 {
        let av = matvec(a, &v);
        let ata_v = matvec_transposed(a, &av);
        let nrm = dot(&ata_v, &ata_v).sqrt();
        if nrm == 0.0 {
            return 0.0;
        }
        for (dst, src) in v.iter_mut().zip(ata_v.iter()) {
            *dst = *src / nrm;
        }
        let new_sigma = nrm.sqrt();
        if (new_sigma - sigma).abs() < 1e-12 * new_sigma.max(1e-30) {
            return new_sigma;
        }
        sigma = new_sigma;
    }
    sigma
}

/// Smallest enclosing disc of a set of complex values (Welzl).
fn enclosing_circle(values: &Array3<Complex64>) -> (Complex64, f64) {
    let mut pts: Vec<Complex64> = Vec::new();
    for v in values.as_slice().iter() {
        if !pts.iter().any(|p| (p - v).norm() < 1e-12) {
            pts.push(*v);
        }
    }
    if pts.is_empty() {
        return (Complex64::new(0.0, 0.0), 0.0);
    }
    if pts.len() == 1 {
        return (pts[0], 0.0);
    }
    pts.sort_by(|a, b| {
        a.re.partial_cmp(&b.re)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.im.partial_cmp(&b.im).unwrap_or(std::cmp::Ordering::Equal))
    });
    let mut r: [Complex64; 3] = [Complex64::new(0.0, 0.0); 3];
    welzl(&mut pts, &mut r, 0)
}

fn welzl(p: &mut [Complex64], r: &mut [Complex64; 3], rn: usize) -> (Complex64, f64) {
    if p.is_empty() || rn == 3 {
        return trivial(&r[..rn]);
    }
    let last = p.len() - 1;
    let pt = p[last];
    let head = &mut p[..last];
    let (c, rad) = welzl(head, r, rn);
    if (pt - c).norm() <= rad + 1e-12 {
        return (c, rad);
    }
    r[rn] = pt;
    welzl(head, r, rn + 1)
}

fn trivial(r: &[Complex64]) -> (Complex64, f64) {
    match r.len() {
        0 => (Complex64::new(0.0, 0.0), 0.0),
        1 => (r[0], 0.0),
        2 => {
            let c = (r[0] + r[1]) * 0.5;
            (c, (r[0] - c).norm())
        }
        3 => circumscribe(r[0], r[1], r[2]),
        _ => unreachable!(),
    }
}

fn circumscribe(a: Complex64, b: Complex64, c: Complex64) -> (Complex64, f64) {
    let bx = b.re - a.re;
    let by = b.im - a.im;
    let cx = c.re - a.re;
    let cy = c.im - a.im;
    let d = 2.0 * (bx * cy - by * cx);
    if d.abs() < 1e-18 {
        let mut best = (a, 0.0);
        for (p, q) in [(a, b), (a, c), (b, c)] {
            let cc = (p + q) * 0.5;
            let rr = (p - cc).norm();
            if rr > best.1 {
                best = (cc, rr);
            }
        }
        return best;
    }
    let ux = (cy * (bx * bx + by * by) - by * (cx * cx + cy * cy)) / d;
    let uy = (bx * (cx * cx + cy * cy) - cx * (bx * bx + by * by)) / d;
    let center = a + Complex64::new(ux, uy);
    let radius = (center - a).norm();
    (center, radius)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn enclosing_circle_constant() {
        let arr = Array3::<Complex64>::from_elem((4, 1, 1), Complex64::new(2.0, -1.0));
        let (c, r) = enclosing_circle(&arr);
        assert!((c - Complex64::new(2.0, -1.0)).norm() < 1e-12);
        assert!(r < 1e-12);
    }

    #[test]
    fn laplace_kernel_real_space_sums_to_zero() {
        let k = laplace_kernel_1d_real_space(0.05, 16);
        let s: f64 = k.iter().sum();
        assert!(s.abs() < 1e-10, "sum={s}");
    }

    #[test]
    fn singleton_axis_kernel_is_zero() {
        let k = laplace_kernel_axis(1, true, 0.05);
        assert_eq!(k.len(), 1);
        assert_eq!(k[0], Complex64::new(0.0, 0.0));
    }

    #[test]
    fn wrap_matrix_shape() {
        let m = make_wrap_matrix(0.05, 5);
        assert_eq!(m.shape(), (5, 5));
        let s = spectral_norm(&m);
        assert!(s > 0.0);
    }

    fn valid_permittivity() -> Array3<Complex64> {
        let mut permittivity = Array3::from_elem((2, 1, 1), Complex64::new(1.0, 0.0));
        permittivity[[1, 0, 0]] = Complex64::new(1.1, 0.0);
        permittivity
    }

    #[test]
    fn checked_domain_constructor_matches_legacy_for_valid_input() {
        let permittivity = valid_permittivity();
        let checked = HelmholtzDomain::try_new(
            &permittivity,
            1.0,
            0.1,
            [(1, 1), (0, 0), (0, 0)],
            [false, true, true],
        )
        .unwrap();
        let legacy = HelmholtzDomain::new(
            &permittivity,
            1.0,
            0.1,
            [(1, 1), (0, 0), (0, 0)],
            [false, true, true],
        );

        assert_eq!(checked.padded_shape(), legacy.padded_shape());
        assert_eq!(checked.b_scat.as_slice(), legacy.b_scat.as_slice());
        assert_eq!(checked.scale, legacy.scale);
    }

    #[test]
    fn propagator_denominators_are_nonzero_and_output_is_finite() {
        // Guards the unguarded `data /= denom` in `propagator_inplace`: a
        // `try_new`-constructed domain must have no degenerate mode, and the
        // propagator of a finite field must stay finite.
        let permittivity = valid_permittivity();
        let domain = HelmholtzDomain::try_new(
            &permittivity,
            1.0,
            0.1,
            [(1, 1), (0, 0), (0, 0)],
            [false, true, true],
        )
        .unwrap();
        for &kx in domain.kernel_x.iter() {
            for &ky in domain.kernel_y.iter() {
                for &kz in domain.kernel_z.iter() {
                    assert!((kx + ky + kz).norm() > 0.0);
                }
            }
        }
        let input = Array3::from_elem(domain.padded_shape(), Complex64::new(1.0, 0.0));
        let output = domain.propagator(&input);
        assert!(
            output
                .as_slice()
                .iter()
                .all(|v| v.re.is_finite() && v.im.is_finite())
        );
    }

    #[test]
    fn checked_domain_constructor_rejects_invalid_physical_admission() {
        let valid = valid_permittivity();
        assert!(matches!(
            HelmholtzDomain::try_new(&valid, 0.0, 0.1, [(0, 0); 3], [true; 3]),
            Err(HelmholtzDomainError::InvalidParameter {
                field: "wavelength",
                ..
            })
        ));
        assert!(matches!(
            HelmholtzDomain::try_new(&valid, 1.0, 0.5, [(0, 0); 3], [true; 3]),
            Err(HelmholtzDomainError::InvalidParameter {
                field: "pixel_size",
                ..
            })
        ));
        assert!(matches!(
            HelmholtzDomain::try_new(&Array3::zeros((0, 1, 1)), 1.0, 0.1, [(0, 0); 3], [true; 3]),
            Err(HelmholtzDomainError::EmptyAxis { axis: 0 })
        ));
        assert!(matches!(
            HelmholtzDomain::try_new(
                &valid,
                1.0,
                0.1,
                [(usize::MAX, 1), (0, 0), (0, 0)],
                [true; 3]
            ),
            Err(HelmholtzDomainError::PaddingOverflow { axis: 0 })
        ));
    }

    #[test]
    fn checked_domain_constructor_rejects_nonphysical_media_and_degeneracy() {
        let mut nonfinite = valid_permittivity();
        nonfinite[[0, 0, 0]] = Complex64::new(f64::NAN, 0.0);
        assert!(matches!(
            HelmholtzDomain::try_new(&nonfinite, 1.0, 0.1, [(0, 0); 3], [true; 3]),
            Err(HelmholtzDomainError::NonFinitePermittivity { index: 0 })
        ));

        let mut gain = valid_permittivity();
        gain[[0, 0, 0]] = Complex64::new(1.0, -0.1);
        assert!(matches!(
            HelmholtzDomain::try_new(&gain, 1.0, 0.1, [(0, 0); 3], [true; 3]),
            Err(HelmholtzDomainError::GainPermittivity { index: 0 })
        ));

        let constant = Array3::from_elem((2, 1, 1), Complex64::new(1.0, 0.0));
        assert!(matches!(
            HelmholtzDomain::try_new(&constant, 1.0, 0.1, [(0, 0); 3], [true; 3]),
            Err(HelmholtzDomainError::DegenerateOperator)
        ));
    }

    #[test]
    fn pointwise_kernels_match_on_both_paths() {
        // Serial and parallel pointwise passes must agree bitwise: same
        // operations in the same element order on both paths.
        let n = 1024;
        let rhs: Vec<Complex64> = (0..n)
            .map(|i| Complex64::new(i as f64 * 0.01, 1.0 - i as f64 * 0.001))
            .collect();
        for parallel in [false, true] {
            let mut serial = rhs.clone();
            let mut threaded = rhs.clone();
            pointwise_multiply(&mut serial, &rhs, false);
            pointwise_multiply(&mut threaded, &rhs, parallel);
            assert_eq!(serial, threaded, "multiply parallel={parallel}");
        }
        let (nx, ny, nz) = (8, 8, 16);
        let kernels: Vec<Complex64> = (0..16)
            .map(|i| Complex64::new(0.5 + i as f64 * 0.1, 0.01 * i as f64))
            .collect();
        let seed: Vec<Complex64> = (0..nx * ny * nz)
            .map(|i| Complex64::new(i as f64 * 0.02, 1.0))
            .collect();
        for parallel in [false, true] {
            let mut serial = seed.clone();
            let mut threaded = seed.clone();
            pointwise_divide(
                &mut serial,
                &kernels[..nx],
                &kernels[..ny],
                &kernels[..nz],
                (nx, ny, nz),
                false,
            );
            pointwise_divide(
                &mut threaded,
                &kernels[..nx],
                &kernels[..ny],
                &kernels[..nz],
                (nx, ny, nz),
                parallel,
            );
            assert_eq!(serial, threaded, "divide parallel={parallel}");
        }
    }
}
