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
        let d = self.dim;
        let j = det(f, d);
        let f_inv_t = inv_transpose(f, d);
        let ln_j = j.ln();

        // P = mu * F + (lambda * ln(J) - mu) * F^{-T}
        let coeff = self.lambda * ln_j - self.mu;
        let mut p = vec![0.0; d * d];
        for i in 0..d * d {
            p[i] = self.mu * f[i] + coeff * f_inv_t[i];
        }
        p
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
    let d = det(f, dim);
    let inv_d = 1.0 / d;
    match dim {
        2 => {
            // inv(F) = (1/det) * [[f[3], -f[1]], [-f[2], f[0]]]
            // inv(F)^T = (1/det) * [[f[3], -f[2]], [-f[1], f[0]]]
            vec![f[3] * inv_d, -f[2] * inv_d, -f[1] * inv_d, f[0] * inv_d]
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
            vec![
                c00 * inv_d,
                c01 * inv_d,
                c02 * inv_d,
                c10 * inv_d,
                c11 * inv_d,
                c12 * inv_d,
                c20 * inv_d,
                c21 * inv_d,
                c22 * inv_d,
            ]
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
}
