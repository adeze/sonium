//! Hyperelastic material models
//!
//! Each model defines a strain energy density function `W(F)` where `F` is the
//! deformation gradient, plus the first Piola-Kirchhoff stress `P = dW/dF` and
//! the material tangent `A = d²W/dF²`.
//!
//! All models work in the total Lagrangian framework: quantities are measured
//! relative to the reference (undeformed) configuration.

/// Trait for hyperelastic material models.
///
/// The deformation gradient `F = I + grad(u)` maps reference to current configuration.
/// For 2D plane strain, `F` is 2x2; for 3D, `F` is 3x3.
pub trait HyperelasticMaterial: Send + Sync {
    /// Spatial dimension (2 or 3).
    fn dim(&self) -> usize;

    /// Strain energy density `W(F)`.
    fn energy(&self, f: &[f64]) -> f64;

    /// First Piola-Kirchhoff stress `P = dW/dF`.
    /// Returns P as a flat array in row-major order: `[P_00, P_01, ..., P_{d-1,d-1}]`.
    fn stress(&self, f: &[f64]) -> Vec<f64>;

    /// First Piola-Kirchhoff stress into caller storage.
    ///
    /// Writes the same row-major `P = dW/dF` that [`stress`](Self::stress)
    /// returns into the first `dim*dim` entries of `out`, so hot paths
    /// (explicit element loops) avoid one heap allocation per quadrature
    /// point. The default implementation allocates through
    /// [`stress`](Self::stress) and copies; models on hot paths override
    /// it with stack arithmetic. Overrides must produce bit-identical
    /// values to [`stress`](Self::stress).
    ///
    /// # Examples
    ///
    /// ```rust
    /// use sonium_fem::physics::solid::nonlinear_solid::{HyperelasticMaterial, NeoHookean};
    ///
    /// let mat = NeoHookean::from_young(200.0e9, 0.3, 3);
    /// let f = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
    /// let mut p = [0.0; 9];
    /// mat.stress_into(&f, &mut p);
    /// assert_eq!(p, [0.0; 9]);
    /// ```
    ///
    /// # Panics
    ///
    /// Panics when `out` holds fewer than `dim*dim` entries.
    fn stress_into(&self, f: &[f64], out: &mut [f64]) {
        let p = self.stress(f);
        out[..p.len()].copy_from_slice(&p);
    }

    /// Material tangent `A_{iJkL} = d²W / dF_{iJ} dF_{kL}`.
    /// Returns as flat array of size `(dim*dim)^2` in row-major order,
    /// indexed as `A[(i*dim+J) * dim*dim + (k*dim+L)]`.
    fn tangent(&self, f: &[f64]) -> Vec<f64>;
}

/// Compressible Neo-Hookean material.
///
/// Strain energy:
/// ```text
/// W(F) = (mu/2)(I_1 - dim) - mu ln(J) + (lambda/2)(ln J)^2
/// ```
/// where `I_1 = tr(F^T F)`, `J = det(F)`, and `mu`, `lambda` are Lamé parameters.
#[derive(Debug, Clone, Copy)]
pub struct NeoHookean {
    /// Shear modulus (second Lamé parameter)
    pub mu: f64,
    /// First Lamé parameter
    pub lambda: f64,
    /// Spatial dimension
    dim: usize,
}

impl NeoHookean {
    /// Create from Lamé parameters.
    pub fn new(mu: f64, lambda: f64, dim: usize) -> Self {
        assert!(dim == 2 || dim == 3);
        Self { mu, lambda, dim }
    }

    /// Create from Young's modulus and Poisson's ratio.
    pub fn from_young(e: f64, nu: f64, dim: usize) -> Self {
        let lambda = e * nu / ((1.0 + nu) * (1.0 - 2.0 * nu));
        let mu = e / (2.0 * (1.0 + nu));
        Self::new(mu, lambda, dim)
    }
}

impl HyperelasticMaterial for NeoHookean {
    fn dim(&self) -> usize {
        self.dim
    }

    fn energy(&self, f: &[f64]) -> f64 {
        let d = self.dim;
        let j = det(f, d);
        let i1 = trace_ftf(f, d);
        let ln_j = j.ln();

        0.5 * self.mu * (i1 - d as f64) - self.mu * ln_j + 0.5 * self.lambda * ln_j * ln_j
    }

    fn stress(&self, f: &[f64]) -> Vec<f64> {
        let mut p = vec![0.0; self.dim * self.dim];
        self.stress_into(f, &mut p);
        p
    }

    fn stress_into(&self, f: &[f64], out: &mut [f64]) {
        let d = self.dim;
        let j = det(f, d);
        let mut f_inv_t = [0.0; 9];
        inv_transpose_into(f, d, &mut f_inv_t[..d * d]);
        let ln_j = j.ln();

        // P = mu * F + (lambda * ln(J) - mu) * F^{-T}
        let coeff = self.lambda * ln_j - self.mu;
        for i in 0..d * d {
            out[i] = self.mu * f[i] + coeff * f_inv_t[i];
        }
    }

    fn tangent(&self, f: &[f64]) -> Vec<f64> {
        let d = self.dim;
        let j = det(f, d);
        let f_inv_t = inv_transpose(f, d);
        let ln_j = j.ln();
        let n = d * d;

        // A_{iJkL} = mu * delta_{ik} delta_{JL}
        //          + lambda * F^{-T}_{iJ} F^{-T}_{kL}
        //          + (mu - lambda * ln J) * F^{-1}_{Ji} * F^{-1}_{Lk}  (via F^{-T})
        //
        // Using: d(F^{-T}_{iJ})/dF_{kL} = -F^{-1}_{Jk} F^{-1}_{Li}
        //        = -F^{-T}_{kJ} F^{-T}_{iL}  (in terms of F^{-T})

        let coeff2 = self.mu - self.lambda * ln_j;
        let mut a = vec![0.0; n * n];

        for i in 0..d {
            for jj in 0..d {
                let ij = i * d + jj;
                for k in 0..d {
                    for l in 0..d {
                        let kl = k * d + l;
                        let idx = ij * n + kl;

                        // mu * delta_{ik} delta_{JL}
                        if i == k && jj == l {
                            a[idx] += self.mu;
                        }

                        // lambda * F^{-T}_{iJ} F^{-T}_{kL}
                        a[idx] += self.lambda * f_inv_t[ij] * f_inv_t[kl];

                        // (mu - lambda*lnJ) * F^{-T}_{kJ} * F^{-T}_{iL}
                        // Note: F^{-T}_{kJ} = f_inv_t[k*d+jj], F^{-T}_{iL} = f_inv_t[i*d+l]
                        a[idx] += coeff2 * f_inv_t[k * d + jj] * f_inv_t[i * d + l];
                    }
                }
            }
        }

        a
    }
}

/// Mooney-Rivlin material (3D only).
///
/// Strain energy:
/// ```text
/// W(F) = c1 * (I1_bar - 3) + c2 * (I2_bar - 3) + (K/2) * (J - 1)^2
/// ```
/// where `I1_bar = J^{-2/3} * I1`, `I2_bar = J^{-4/3} * I2`, and K is the bulk modulus.
#[derive(Debug, Clone, Copy)]
pub struct MooneyRivlin {
    /// First Mooney-Rivlin parameter
    pub c1: f64,
    /// Second Mooney-Rivlin parameter
    pub c2: f64,
    /// Bulk modulus
    pub bulk: f64,
}

impl MooneyRivlin {
    pub fn new(c1: f64, c2: f64, bulk: f64) -> Self {
        Self { c1, c2, bulk }
    }

    /// Create from shear modulus and bulk modulus (c2 = 0 reduces to Neo-Hookean).
    pub fn from_moduli(mu: f64, bulk: f64) -> Self {
        Self {
            c1: mu / 2.0,
            c2: 0.0,
            bulk,
        }
    }
}

impl HyperelasticMaterial for MooneyRivlin {
    fn dim(&self) -> usize {
        3
    }

    fn energy(&self, f: &[f64]) -> f64 {
        let j = det(f, 3);
        let i1 = trace_ftf(f, 3);
        let i2 = second_invariant_ftf(f, 3);

        let j23 = j.powf(-2.0 / 3.0);
        let j43 = j23 * j23;

        let i1_bar = j23 * i1;
        let i2_bar = j43 * i2;

        self.c1 * (i1_bar - 3.0) + self.c2 * (i2_bar - 3.0) + 0.5 * self.bulk * (j - 1.0).powi(2)
    }

    fn stress(&self, f: &[f64]) -> Vec<f64> {
        // Numerical differentiation for Mooney-Rivlin stress
        // (analytical form is complex; numerical is fine for validation)
        numerical_stress(self, f, 3)
    }

    fn tangent(&self, f: &[f64]) -> Vec<f64> {
        numerical_tangent(self, f, 3)
    }
}

/// Ogden hyperelastic material (3D only).
///
/// Strain energy in principal stretches (isochoric-volumetric split):
/// ```text
/// W = sum_i (mu_i / alpha_i) (l1b^ai + l2b^ai + l3b^ai - 3) + (K/2) (J - 1)^2
/// ```
/// where `lib = J^{-1/3} li` are the isochoric principal stretches.
/// Recovers the Neo-Hookean response for one term with `alpha = 2`.
/// Stress and tangent use numerical differentiation of `energy`, matching
/// the Mooney-Rivlin treatment in this module.
///
/// # Examples
///
/// ```rust
/// use sonium_fem::physics::solid::nonlinear_solid::{HyperelasticMaterial, OgdenMaterial};
///
/// let rubber = OgdenMaterial::treloar();
/// let f = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
/// assert!(rubber.energy(&f).abs() < 1e-12);
/// ```
#[derive(Debug, Clone)]
pub struct OgdenMaterial {
    /// Ogden shear moduli `mu_i` in Pa, one per term.
    pub mu: Vec<f64>,
    /// Ogden exponents `alpha_i`, one per term.
    pub alpha: Vec<f64>,
    /// Bulk modulus in Pa.
    pub bulk: f64,
}

impl OgdenMaterial {
    /// Creates an Ogden material from term data and bulk modulus.
    ///
    /// # Panics
    ///
    /// Panics when `mu` and `alpha` differ in length or are empty.
    pub fn new(mu: Vec<f64>, alpha: Vec<f64>, bulk: f64) -> Self {
        assert!(
            !mu.is_empty() && mu.len() == alpha.len(),
            "ogden terms need matching non-empty mu and alpha lists"
        );
        Self { mu, alpha, bulk }
    }

    /// Treloar's vulcanized rubber fit (three terms).
    pub fn treloar() -> Self {
        Self::new(
            vec![0.618e6, 0.00118e6, -0.00981e6],
            vec![1.3, 5.0, -2.0],
            100.0e6,
        )
    }

    /// Single-term Neo-Hookean-equivalent Ogden fit.
    pub fn neo_hookean(mu: f64, bulk: f64) -> Self {
        Self::new(vec![mu], vec![2.0], bulk)
    }

    /// Isochoric principal stretches of a deformation gradient.
    ///
    /// Returns `(J, [l1b, l2b, l3b])`: the Jacobian plus the eigenvalues
    /// of the isochoric right stretch, via cyclic Jacobi diagonalization.
    fn isochoric_stretches(&self, f: &[f64]) -> (f64, [f64; 3]) {
        let j = det(f, 3).max(1e-30);
        let scale = j.powf(-1.0 / 3.0);
        // Isochoric right Cauchy-Green Cb = J^{-2/3} F^T F.
        let mut cb = [0.0; 9];
        for i in 0..3 {
            for jj in 0..3 {
                let mut sum = 0.0;
                for k in 0..3 {
                    sum += f[k * 3 + i] * f[k * 3 + jj];
                }
                cb[i * 3 + jj] = scale * scale * sum;
            }
        }
        let (values, _) = sym_eigen_3x3(&cb);
        (
            j,
            [
                values[0].max(1e-30).sqrt(),
                values[1].max(1e-30).sqrt(),
                values[2].max(1e-30).sqrt(),
            ],
        )
    }
}

impl HyperelasticMaterial for OgdenMaterial {
    fn dim(&self) -> usize {
        3
    }

    fn energy(&self, f: &[f64]) -> f64 {
        let (j, stretches) = self.isochoric_stretches(f);
        let mut w = 0.0;
        for (mu, alpha) in self.mu.iter().zip(self.alpha.iter()) {
            // expm1 form: exact at identity and cancellation-free nearby,
            // which the numerical stress and tangent differentiate cleanly.
            let sum: f64 = stretches.iter().map(|l| (alpha * l.ln()).exp_m1()).sum();
            w += mu / alpha * sum;
        }
        w + 0.5 * self.bulk * (j - 1.0).powi(2)
    }

    fn stress(&self, f: &[f64]) -> Vec<f64> {
        numerical_stress(self, f, 3)
    }

    fn tangent(&self, f: &[f64]) -> Vec<f64> {
        numerical_tangent(self, f, 3)
    }
}

/// Incompressible uniaxial Cauchy stress of an Ogden solid.
///
/// Closed form `sigma = sum_i mu_i (l^ai - l^{-ai/2})` for the axial
/// stretch `l`, used by the C2 golden and the rubber QA leg.
///
/// # Examples
///
/// ```rust
/// use sonium_fem::physics::solid::nonlinear_solid::ogden_uniaxial_cauchy;
///
/// let sigma = ogden_uniaxial_cauchy(&[0.618e6], &[1.3], 1.5);
/// assert!(sigma > 0.0);
/// ```
pub fn ogden_uniaxial_cauchy(mu: &[f64], alpha: &[f64], stretch: f64) -> f64 {
    mu.iter()
        .zip(alpha.iter())
        .map(|(m, a)| m * (stretch.powf(*a) - stretch.powf(-a / 2.0)))
        .sum()
}

/// Yeoh hyperelastic material (3D only).
///
/// Strain energy in the first isochoric invariant:
/// ```text
/// W = c1 (I1b - 3) + c2 (I1b - 3)^2 + c3 (I1b - 3)^3 + (K/2) (J - 1)^2
/// ```
/// Captures the stiffening of filled rubbers at large stretch with a
/// single invariant. Stress and tangent use numerical differentiation.
///
/// # Examples
///
/// ```rust
/// use sonium_fem::physics::solid::nonlinear_solid::{HyperelasticMaterial, YeohMaterial};
///
/// let rubber = YeohMaterial::filled_rubber();
/// let f = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
/// assert!(rubber.energy(&f).abs() < 1e-12);
/// ```
#[derive(Debug, Clone, Copy)]
pub struct YeohMaterial {
    /// First Yeoh coefficient in Pa.
    pub c1: f64,
    /// Second Yeoh coefficient in Pa.
    pub c2: f64,
    /// Third Yeoh coefficient in Pa.
    pub c3: f64,
    /// Bulk modulus in Pa.
    pub bulk: f64,
}

impl YeohMaterial {
    /// Creates a Yeoh material from three coefficients and bulk modulus.
    pub fn new(c1: f64, c2: f64, c3: f64, bulk: f64) -> Self {
        Self { c1, c2, c3, bulk }
    }

    /// Carbon-black filled natural rubber fit.
    pub fn filled_rubber() -> Self {
        Self::new(0.40e6, -0.022e6, 0.0032e6, 100.0e6)
    }
}

impl HyperelasticMaterial for YeohMaterial {
    fn dim(&self) -> usize {
        3
    }

    fn energy(&self, f: &[f64]) -> f64 {
        let j = det(f, 3);
        let i1 = trace_ftf(f, 3);
        let i1_bar = j.powf(-2.0 / 3.0) * i1;
        let d = i1_bar - 3.0;
        self.c1 * d + self.c2 * d * d + self.c3 * d * d * d + 0.5 * self.bulk * (j - 1.0).powi(2)
    }

    fn stress(&self, f: &[f64]) -> Vec<f64> {
        numerical_stress(self, f, 3)
    }

    fn tangent(&self, f: &[f64]) -> Vec<f64> {
        numerical_tangent(self, f, 3)
    }
}

/// Eigendecomposition of a symmetric 3x3 via cyclic Jacobi rotations.
///
/// Returns eigenvalues and the column eigenvector matrix, so that
/// `a = V diag(values) V^T`. Shared by the Ogden principal stretches.
fn sym_eigen_3x3(a: &[f64; 9]) -> ([f64; 3], [f64; 9]) {
    let mut d = *a;
    let mut v = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
    for _ in 0..12 {
        let mut off = 0.0;
        for (p, q) in [(0, 1), (0, 2), (1, 2)] {
            off += d[p * 3 + q] * d[p * 3 + q];
        }
        if off < 1e-30 {
            break;
        }
        for (p, q) in [(0, 1), (0, 2), (1, 2)] {
            let apq = d[p * 3 + q];
            if apq * apq < 1e-32 {
                continue;
            }
            let app = d[p * 3 + p];
            let aqq = d[q * 3 + q];
            let theta = (aqq - app) / (2.0 * apq);
            let t = theta.signum() / (theta.abs() + (1.0 + theta * theta).sqrt());
            let c = 1.0 / (1.0 + t * t).sqrt();
            let s = t * c;
            for k in 0..3 {
                let dpk = d[p * 3 + k];
                let dqk = d[q * 3 + k];
                d[p * 3 + k] = c * dpk - s * dqk;
                d[q * 3 + k] = s * dpk + c * dqk;
            }
            for k in 0..3 {
                let dkp = d[k * 3 + p];
                let dkq = d[k * 3 + q];
                d[k * 3 + p] = c * dkp - s * dkq;
                d[k * 3 + q] = s * dkp + c * dkq;
            }
            for k in 0..3 {
                let vkp = v[k * 3 + p];
                let vkq = v[k * 3 + q];
                v[k * 3 + p] = c * vkp - s * vkq;
                v[k * 3 + q] = s * vkp + c * vkq;
            }
        }
    }
    ([d[0], d[4], d[8]], v)
}

// === Helper functions ===

/// Determinant of a dim x dim matrix stored row-major.
fn det(f: &[f64], dim: usize) -> f64 {
    match dim {
        2 => f[0] * f[3] - f[1] * f[2],
        3 => {
            f[0] * (f[4] * f[8] - f[5] * f[7]) - f[1] * (f[3] * f[8] - f[5] * f[6])
                + f[2] * (f[3] * f[7] - f[4] * f[6])
        }
        _ => unreachable!(),
    }
}

/// tr(F^T F) = sum of squared entries of F.
fn trace_ftf(f: &[f64], dim: usize) -> f64 {
    f[..dim * dim].iter().map(|&x| x * x).sum()
}

/// Second invariant of F^T F (3D only): I2 = 0.5 * (I1^2 - tr((F^T F)^2))
fn second_invariant_ftf(f: &[f64], dim: usize) -> f64 {
    assert_eq!(dim, 3);
    let i1 = trace_ftf(f, 3);

    // Compute (F^T F)^2 trace
    let mut ftf = [0.0; 9];
    for i in 0..3 {
        for j in 0..3 {
            for k in 0..3 {
                ftf[i * 3 + j] += f[k * 3 + i] * f[k * 3 + j];
            }
        }
    }
    let mut tr_ftf2 = 0.0;
    for i in 0..3 {
        for k in 0..3 {
            tr_ftf2 += ftf[i * 3 + k] * ftf[k * 3 + i];
        }
    }

    0.5 * (i1 * i1 - tr_ftf2)
}

/// Inverse transpose of a dim x dim matrix.
fn inv_transpose(f: &[f64], dim: usize) -> Vec<f64> {
    let mut out = vec![0.0; dim * dim];
    inv_transpose_into(f, dim, &mut out);
    out
}

/// Inverse transpose of a dim x dim matrix into caller storage.
///
/// Writes the same entries [`inv_transpose`] returns into the first
/// `dim*dim` slots of `out`, so hot paths avoid the per-call heap
/// allocation. Operation order matches [`inv_transpose`] exactly, keeping
/// results bit-identical between the two spellings.
fn inv_transpose_into(f: &[f64], dim: usize, out: &mut [f64]) {
    let d = det(f, dim);
    let inv_d = 1.0 / d;
    match dim {
        2 => {
            // inv(F) = (1/det) * [[f[3], -f[1]], [-f[2], f[0]]]
            // inv(F)^T = (1/det) * [[f[3], -f[2]], [-f[1], f[0]]]
            out[0] = f[3] * inv_d;
            out[1] = -f[2] * inv_d;
            out[2] = -f[1] * inv_d;
            out[3] = f[0] * inv_d;
        }
        3 => {
            // Cofactor matrix (transposed) divided by det
            let c00 = f[4] * f[8] - f[5] * f[7];
            let c01 = -(f[3] * f[8] - f[5] * f[6]);
            let c02 = f[3] * f[7] - f[4] * f[6];
            let c10 = -(f[1] * f[8] - f[2] * f[7]);
            let c11 = f[0] * f[8] - f[2] * f[6];
            let c12 = -(f[0] * f[7] - f[1] * f[6]);
            let c20 = f[1] * f[5] - f[2] * f[4];
            let c21 = -(f[0] * f[5] - f[2] * f[3]);
            let c22 = f[0] * f[4] - f[1] * f[3];

            // F^{-T} = cofactor(F) / det(F)
            out[0] = c00 * inv_d;
            out[1] = c01 * inv_d;
            out[2] = c02 * inv_d;
            out[3] = c10 * inv_d;
            out[4] = c11 * inv_d;
            out[5] = c12 * inv_d;
            out[6] = c20 * inv_d;
            out[7] = c21 * inv_d;
            out[8] = c22 * inv_d;
        }
        _ => unreachable!(),
    }
}

/// Numerical stress via central differences: P_{iJ} = dW/dF_{iJ}.
fn numerical_stress(mat: &dyn HyperelasticMaterial, f: &[f64], dim: usize) -> Vec<f64> {
    let n = dim * dim;
    let eps = 1e-7;
    let mut p = vec![0.0; n];
    let mut f_plus = f.to_vec();
    let mut f_minus = f.to_vec();

    for ij in 0..n {
        f_plus[ij] = f[ij] + eps;
        f_minus[ij] = f[ij] - eps;
        p[ij] = (mat.energy(&f_plus) - mat.energy(&f_minus)) / (2.0 * eps);
        f_plus[ij] = f[ij];
        f_minus[ij] = f[ij];
    }
    p
}

/// Numerical tangent via central differences: A_{iJkL} = dP_{iJ}/dF_{kL}.
fn numerical_tangent(mat: &dyn HyperelasticMaterial, f: &[f64], dim: usize) -> Vec<f64> {
    let n = dim * dim;
    let eps = 1e-7;
    let mut a = vec![0.0; n * n];
    let mut f_plus = f.to_vec();
    let mut f_minus = f.to_vec();

    for kl in 0..n {
        f_plus[kl] = f[kl] + eps;
        f_minus[kl] = f[kl] - eps;
        let p_plus = mat.stress(&f_plus);
        let p_minus = mat.stress(&f_minus);
        for ij in 0..n {
            a[ij * n + kl] = (p_plus[ij] - p_minus[ij]) / (2.0 * eps);
        }
        f_plus[kl] = f[kl];
        f_minus[kl] = f[kl];
    }
    a
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_neohookean_identity_stress_is_zero() {
        let mat = NeoHookean::from_young(1.0, 0.3, 3);
        // F = I: no deformation, stress should be zero
        let f = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        let p = mat.stress(&f);
        for (i, &pi) in p.iter().enumerate() {
            assert!(pi.abs() < 1e-14, "P[{i}] = {pi}, expected 0");
        }
    }

    #[test]
    fn test_neohookean_uniaxial_stretch() {
        let mu = 1.0;
        let lambda = 1.0;
        let mat = NeoHookean::new(mu, lambda, 3);

        // Uniaxial stretch: F = diag(stretch, 1, 1)
        let stretch = 1.1;
        let f = [stretch, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        let p = mat.stress(&f);

        // P_11 should be positive (tension)
        assert!(p[0] > 0.0, "P_11 = {} should be positive for tension", p[0]);
        // Off-diagonal should be zero (uniaxial)
        assert!(p[1].abs() < 1e-14);
        assert!(p[2].abs() < 1e-14);
    }

    #[test]
    fn test_neohookean_stress_vs_numerical() {
        let mat = NeoHookean::from_young(10.0, 0.3, 3);
        let f = [1.05, 0.02, 0.0, -0.01, 1.03, 0.01, 0.0, 0.0, 0.98];

        let p_analytical = mat.stress(&f);
        let p_numerical = numerical_stress(&mat, &f, 3);

        for i in 0..9 {
            assert!(
                (p_analytical[i] - p_numerical[i]).abs() < 1e-5,
                "P[{i}]: analytical={:.6e}, numerical={:.6e}",
                p_analytical[i],
                p_numerical[i]
            );
        }
    }

    #[test]
    fn test_neohookean_tangent_vs_numerical() {
        let mat = NeoHookean::from_young(10.0, 0.3, 3);
        let f = [1.02, 0.01, 0.0, -0.01, 1.01, 0.005, 0.0, 0.0, 0.99];

        let a_analytical = mat.tangent(&f);
        let a_numerical = numerical_tangent(&mat, &f, 3);

        for i in 0..81 {
            assert!(
                (a_analytical[i] - a_numerical[i]).abs() < 1e-3,
                "A[{i}]: analytical={:.6e}, numerical={:.6e}",
                a_analytical[i],
                a_numerical[i]
            );
        }
    }

    #[test]
    fn test_neohookean_tangent_major_symmetry() {
        let mat = NeoHookean::from_young(1.0, 0.3, 3);
        let f = [1.05, 0.02, 0.0, -0.01, 1.03, 0.01, 0.0, 0.0, 0.98];
        let a = mat.tangent(&f);
        let n = 9;
        // Major symmetry: A_{iJkL} = A_{kLiJ}
        for ij in 0..n {
            for kl in ij + 1..n {
                let diff = (a[ij * n + kl] - a[kl * n + ij]).abs();
                assert!(diff < 1e-12, "Major symmetry: A[{ij},{kl}] vs A[{kl},{ij}]");
            }
        }
    }

    #[test]
    fn test_neohookean_2d() {
        let mat = NeoHookean::from_young(1.0, 0.3, 2);
        let f = [1.0, 0.0, 0.0, 1.0]; // identity
        let p = mat.stress(&f);
        for &pi in &p {
            assert!(pi.abs() < 1e-14);
        }
    }

    #[test]
    fn test_ogden_identity_is_stress_free() {
        let mat = OgdenMaterial::treloar();
        let f = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        assert!(mat.energy(&f).abs() < 1e-12);
        // Numerical differentiation of the spectral energy bottoms out at
        // mu * eig_err / eps ~ 1e-3 Pa; the bound below is 1e-8 of mu_1.
        for pi in mat.stress(&f) {
            assert!(pi.abs() < 5.0e-3, "P = {pi}, expected 0");
        }
    }

    #[test]
    fn test_ogden_uniaxial_energy_closed_form() {
        let mu = 0.5e6;
        let alpha = 1.3;
        let mat = OgdenMaterial::new(vec![mu], vec![alpha], 1.0e9);
        let l = 1.5f64;
        let lt = 1.0 / l.sqrt();
        let f = [l, 0.0, 0.0, 0.0, lt, 0.0, 0.0, 0.0, lt];
        let expected = mu / alpha * (l.powf(alpha) + 2.0 * lt.powf(alpha) - 3.0);
        assert!((mat.energy(&f) - expected).abs() / expected < 1e-12);
    }

    #[test]
    fn test_ogden_alpha2_matches_neo_hookean_isochoric() {
        let mu = 0.8e6;
        let ogden = OgdenMaterial::neo_hookean(mu, 1.0e9);
        // Isochoric shear: J = 1, so only the (mu/2)(I1 - 3) part acts.
        let f = [1.0, 0.3, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        let i1 = trace_ftf(&f, 3);
        let expected = 0.5 * mu * (i1 - 3.0);
        assert!((ogden.energy(&f) - expected).abs() / expected < 1e-9);
    }

    #[test]
    fn test_ogden_uniaxial_cauchy_closed_form() {
        let sigma = ogden_uniaxial_cauchy(&[0.618e6], &[1.3], 1.5);
        let expected = 0.618e6 * (1.5f64.powf(1.3) - 1.5f64.powf(-0.65));
        assert!((sigma - expected).abs() / expected < 1e-15);
        assert!(ogden_uniaxial_cauchy(&[0.618e6], &[1.3], 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_yeoh_uniaxial_energy_closed_form() {
        let mat = YeohMaterial::filled_rubber();
        let l = 2.0f64;
        let lt = 1.0 / l.sqrt();
        let f = [l, 0.0, 0.0, 0.0, lt, 0.0, 0.0, 0.0, lt];
        let d = (l * l + 2.0 * lt * lt) - 3.0;
        let expected = mat.c1 * d + mat.c2 * d * d + mat.c3 * d * d * d;
        assert!((mat.energy(&f) - expected).abs() / expected < 1e-12);
    }

    #[test]
    fn test_yeoh_initial_modulus_is_2c1() {
        let mat = YeohMaterial::new(0.40e6, -0.022e6, 0.0032e6, 100.0e6);
        let f = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        let a = mat.tangent(&f);
        // Isotropic tangent at identity with mu = 2 c1.
        let mu = 2.0 * mat.c1;
        let lambda = mat.bulk - 2.0 * mu / 3.0;
        let expected_0000 = lambda + 2.0 * mu;
        assert!(
            (a[0] - expected_0000).abs() / expected_0000 < 0.01,
            "A0000 = {}, want {expected_0000}",
            a[0]
        );
    }

    #[test]
    fn stress_into_matches_stress_bit_exact() {
        // Sheared stretch with all entries exercised, plus identity.
        let cases = [
            [1.05, 0.02, -0.01, 0.03, 0.97, 0.04, -0.02, 0.01, 1.08],
            [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        ];
        let neo = NeoHookean::from_young(200.0e9, 0.3, 3);
        // Mooney-Rivlin keeps the default provided method (heap + copy).
        let mooney = MooneyRivlin::new(0.3e6, 0.1e6, 100.0e6);
        for f in cases {
            for mat in [
                &neo as &dyn HyperelasticMaterial,
                &mooney as &dyn HyperelasticMaterial,
            ] {
                let heap = mat.stress(&f);
                let mut stack = [0.0; 9];
                mat.stress_into(&f, &mut stack);
                for i in 0..9 {
                    assert_eq!(
                        stack[i].to_bits(),
                        heap[i].to_bits(),
                        "entry {i} differs: stack {}, heap {}",
                        stack[i],
                        heap[i]
                    );
                }
            }
        }
        // 2D override path writes exactly four entries.
        let neo2 = NeoHookean::from_young(70.0e9, 0.33, 2);
        let f2 = [1.04, 0.03, -0.02, 0.96];
        let heap2 = neo2.stress(&f2);
        let mut stack2 = [f64::NAN; 4];
        neo2.stress_into(&f2, &mut stack2);
        for i in 0..4 {
            assert_eq!(stack2[i].to_bits(), heap2[i].to_bits());
        }
    }

    #[test]
    fn inv_transpose_into_matches_heap_spelling() {
        let f = [1.05, 0.02, -0.01, 0.03, 0.97, 0.04, -0.02, 0.01, 1.08];
        let heap = inv_transpose(&f, 3);
        let mut stack = [0.0; 9];
        inv_transpose_into(&f, 3, &mut stack);
        for i in 0..9 {
            assert_eq!(stack[i].to_bits(), heap[i].to_bits());
        }
    }
}
