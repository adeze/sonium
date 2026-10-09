//! Eddy current bilinear forms
//!
//! Frequency-domain A-formulation:
//! `curl(1/mu * curl(A)) + j*omega*sigma*A = J_s`
//!
//! Weak form:
//! `(1/mu) integral curl(A).curl(v) dOmega + j*omega*sigma integral A.v dOmega = integral J_s.v dOmega`
//!
//! Uses H(curl) Nédélec elements (same as Maxwell).

use crate::forms::{BilinearForm, LinearForm, LocalMatrix, LocalVector};
use crate::spaces::BasisTabulation;
use num_complex::Complex64;

use super::materials::ConductorMaterial;

/// Eddy current bilinear form (frequency domain).
///
/// `a(A, v) = (1/mu) integral curl(A).curl(v) dOmega + j*omega*sigma integral A.v dOmega`
///
/// The curl-curl term gives the magnetic stiffness; the j*omega*sigma term
/// represents eddy current losses (resistive dissipation in conductors).
#[derive(Debug, Clone)]
pub struct EddyCurrentForm {
    /// 1/mu for the curl-curl term.
    pub inv_mu: f64,
    /// omega * sigma for the eddy current term.
    pub omega_sigma: f64,
    /// Angular frequency.
    pub omega: f64,
}

impl EddyCurrentForm {
    /// Create from material properties and frequency.
    pub fn new(material: &ConductorMaterial, frequency: f64) -> Self {
        let omega = 2.0 * std::f64::consts::PI * frequency;
        Self {
            inv_mu: 1.0 / material.permeability(),
            omega_sigma: omega * material.conductivity,
            omega,
        }
    }

    /// For non-conducting regions (air, permanent magnets): only curl-curl, no eddy currents.
    pub fn non_conducting(mu_r: f64) -> Self {
        let mu = mu_r * 4.0 * std::f64::consts::PI * 1e-7;
        Self {
            inv_mu: 1.0 / mu,
            omega_sigma: 0.0,
            omega: 0.0,
        }
    }
}

impl BilinearForm<Complex64> for EddyCurrentForm {
    fn integrate_point(
        &self,
        test: &BasisTabulation,
        trial: &BasisTabulation,
        weight: f64,
        local_matrix: &mut LocalMatrix<Complex64>,
    ) {
        let n = test.n_dofs;
        let dim = test.spatial_dim;
        let w = weight * test.det_j;

        match dim {
            2 => {
                // 2D: scalar curl, 2D vector values
                for i in 0..n {
                    let curl_i = test.derivatives[i];
                    for j in 0..n {
                        let curl_j = trial.derivatives[j];

                        // Curl-curl: (1/mu) * curl_i * curl_j
                        let cc = self.inv_mu * curl_i * curl_j;

                        // Eddy current: j*omega*sigma * N_i . N_j
                        let dot: f64 = (0..2)
                            .map(|d| test.values[i * 2 + d] * trial.values[j * 2 + d])
                            .sum();

                        let val = Complex64::new(cc * w, self.omega_sigma * dot * w);
                        local_matrix.add(i, j, val);
                    }
                }
            }
            3 => {
                // 3D: vector curl, 3D vector values
                for i in 0..n {
                    for j in 0..n {
                        let curl_dot: f64 = (0..3)
                            .map(|d| test.derivatives[i * 3 + d] * trial.derivatives[j * 3 + d])
                            .sum();
                        let cc = self.inv_mu * curl_dot;

                        let dot: f64 = (0..3)
                            .map(|d| test.values[i * 3 + d] * trial.values[j * 3 + d])
                            .sum();

                        let val = Complex64::new(cc * w, self.omega_sigma * dot * w);
                        local_matrix.add(i, j, val);
                    }
                }
            }
            _ => unreachable!(),
        }
    }
}

/// Conductivity mass form: `a(A, v) = sigma * integral A.v dOmega`
///
/// Delegates to [`WeightedEdgeMassForm`](crate::forms::WeightedEdgeMassForm) with `coefficient = sigma`.
#[derive(Debug, Clone, Copy)]
pub struct ConductivityMassForm(crate::forms::WeightedEdgeMassForm);

impl ConductivityMassForm {
    pub fn new(material: &ConductorMaterial) -> Self {
        Self(crate::forms::WeightedEdgeMassForm {
            coefficient: material.conductivity,
        })
    }
}

impl BilinearForm<f64> for ConductivityMassForm {
    #[inline]
    fn integrate_point(
        &self,
        test: &BasisTabulation,
        trial: &BasisTabulation,
        weight: f64,
        local_matrix: &mut LocalMatrix<f64>,
    ) {
        self.0.integrate_point(test, trial, weight, local_matrix);
    }
}

/// Source current linear form: `l(v) = integral J_s . v dOmega`
pub struct CurrentSourceForm {
    /// Source current density [Jx, Jy, Jz].
    pub current_density: [f64; 3],
}

impl CurrentSourceForm {
    pub fn new(j: [f64; 3]) -> Self {
        Self { current_density: j }
    }
}

impl LinearForm<Complex64> for CurrentSourceForm {
    fn integrate_point(
        &self,
        test: &BasisTabulation,
        weight: f64,
        local_vector: &mut LocalVector<Complex64>,
    ) {
        let n = test.n_dofs;
        let dim = test.spatial_dim;
        let w = weight * test.det_j;

        for i in 0..n {
            let dot: f64 = (0..dim)
                .map(|d| test.values[i * dim + d] * self.current_density[d])
                .sum();
            local_vector.add(i, Complex64::new(dot * w, 0.0));
        }
    }
}

/// Compute complex impedance from eddy current solution.
///
/// The impedance of a coil in the presence of eddy currents is:
/// `Z = j*omega * L + R_dc + Z_eddy`
///
/// where `Z_eddy` is extracted from the A-field solution via the
/// power balance: `P + jQ = integral (j*omega*sigma*|A|^2 + (1/mu)|curl A|^2) dOmega`.
pub fn extract_impedance(
    _omega: f64,
    power_loss: f64,     // real power dissipated by eddy currents
    reactive_power: f64, // reactive power stored in magnetic field
    current_rms: f64,
) -> (f64, f64) {
    let i_sq = current_rms * current_rms;
    let r_eff = power_loss / i_sq; // effective resistance (includes eddy current loss)
    let x_eff = reactive_power / i_sq; // effective reactance
    (r_eff, x_eff)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spaces::{FunctionSpace, HcurlSpace};

    #[test]
    fn test_eddy_current_form_symmetry_2d() {
        let mesh = crate::mesh::unit_square_triangles(2);
        let space = HcurlSpace::new();
        let tab = space.tabulate(&mesh, 0, 1.0 / 3.0, 1.0 / 3.0, 0.0);
        let n = tab.n_dofs;

        let mat = ConductorMaterial::steel_pole_piece();
        let form = EddyCurrentForm::new(&mat, 1000.0);
        let mut k = LocalMatrix::zeros(n);
        form.integrate_point(&tab, &tab, 1.0, &mut k);

        // Real part should be symmetric (curl-curl is symmetric)
        for i in 0..n {
            for j in i + 1..n {
                let diff = (k.get(i, j).re - k.get(j, i).re).abs();
                assert!(diff < 1e-10, "Real part not symmetric at ({i},{j})");
            }
        }

        // Imaginary part should be symmetric (mass-like form is symmetric)
        for i in 0..n {
            for j in i + 1..n {
                let diff = (k.get(i, j).im - k.get(j, i).im).abs();
                assert!(diff < 1e-10, "Imag part not symmetric at ({i},{j})");
            }
        }
    }

    #[test]
    fn test_eddy_current_non_conducting_is_real() {
        let mesh = crate::mesh::unit_square_triangles(2);
        let space = HcurlSpace::new();
        let tab = space.tabulate(&mesh, 0, 1.0 / 3.0, 1.0 / 3.0, 0.0);
        let n = tab.n_dofs;

        let form = EddyCurrentForm::non_conducting(1.0);
        let mut k = LocalMatrix::zeros(n);
        form.integrate_point(&tab, &tab, 1.0, &mut k);

        // No conductivity: imaginary part should be zero
        for i in 0..n {
            for j in 0..n {
                assert!(
                    k.get(i, j).im.abs() < 1e-14,
                    "Non-conducting should have zero imaginary part at ({i},{j})"
                );
            }
        }
    }

    #[test]
    fn test_impedance_extraction() {
        let (r, x) = extract_impedance(
            2.0 * std::f64::consts::PI * 1000.0,
            10.0, // 10 W dissipated
            50.0, // 50 VAR reactive
            1.0,  // 1 A rms
        );
        assert!((r - 10.0).abs() < 1e-10);
        assert!((x - 50.0).abs() < 1e-10);
    }
}
