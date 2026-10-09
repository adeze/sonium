//! Thermal bilinear and linear forms
//!
//! All forms use scalar H1 function space (temperature is a scalar field).

use crate::forms::{
    BilinearForm, LinearForm, LocalMatrix, LocalVector, WeightedLaplacianForm, WeightedMassForm,
    require_closure_components,
};
use crate::spaces::BasisTabulation;

use super::materials::{ConvectionCoeff, STEFAN_BOLTZMANN, ThermalMaterial};

/// Heat conduction form: `a(T, v) = integral k * grad(T) . grad(v) dOmega`
///
/// Delegates to [`WeightedLaplacianForm`] with `coefficient = k`.
#[derive(Debug, Clone, Copy)]
pub struct ConductionForm(WeightedLaplacianForm);

impl ConductionForm {
    pub fn new(material: &ThermalMaterial) -> Self {
        Self(WeightedLaplacianForm {
            coefficient: material.conductivity,
        })
    }

    pub fn from_conductivity(k: f64) -> Self {
        Self(WeightedLaplacianForm { coefficient: k })
    }
}

impl BilinearForm<f64> for ConductionForm {
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

/// Heat capacity form: `m(T, v) = integral rho * c * T * v dOmega`
///
/// Used for transient analysis: `M * dT/dt + K * T = f`.
/// Delegates to [`WeightedMassForm`] with `coefficient = rho * c`.
#[derive(Debug, Clone, Copy)]
pub struct HeatCapacityForm(WeightedMassForm);

impl HeatCapacityForm {
    pub fn new(material: &ThermalMaterial) -> Self {
        Self(WeightedMassForm {
            coefficient: material.volumetric_heat_capacity(),
        })
    }
}

impl BilinearForm<f64> for HeatCapacityForm {
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

/// Advection form: `a(T, v) = integral rho * c * u . grad(T) * v dOmega`
///
/// For forced convection / advection-diffusion problems.
/// The velocity field `u` is given as a function of position.
///
/// The closure must return at least `dim` components on a `dim`-dimensional
/// mesh (conventionally `[u_x, u_y, u_z]`); extra components are ignored.
/// A short return panics instead of silently truncating the dot product.
pub struct AdvectionForm<F: Fn(f64, f64, f64) -> Vec<f64> + Send + Sync> {
    /// Volumetric heat capacity rho * c.
    pub rho_c: f64,
    /// Velocity field u(x, y, z) -> [u_x, u_y, u_z].
    pub velocity_fn: F,
}

impl<F: Fn(f64, f64, f64) -> Vec<f64> + Send + Sync> BilinearForm<f64> for AdvectionForm<F> {
    fn integrate_point(
        &self,
        test: &BasisTabulation,
        trial: &BasisTabulation,
        weight: f64,
        local_matrix: &mut LocalMatrix<f64>,
    ) {
        let n = test.n_dofs;
        let dim = test.spatial_dim;
        let [x, y, z] = test.physical_coords;
        let u = (self.velocity_fn)(x, y, z);
        let w = weight * test.det_j * self.rho_c;

        // A short closure return would silently truncate the dot product and
        // assemble wrong physics with no error. Fail loudly in all builds;
        // extra components beyond `dim` are ignored.
        require_closure_components("AdvectionForm velocity", &u, dim);
        for i in 0..n {
            for j in 0..n {
                // u . grad(N_j) * N_i
                let u_dot_grad: f64 = (0..dim).map(|d| u[d] * trial.grad(j, d)).sum();
                local_matrix.add(i, j, test.value(i) * u_dot_grad * w);
            }
        }
    }
}

/// Volumetric heat source: `l(v) = integral Q * v dOmega`
pub struct HeatSourceForm<F: Fn(f64, f64, f64) -> f64 + Send + Sync> {
    pub source_fn: F,
}

impl<F: Fn(f64, f64, f64) -> f64 + Send + Sync> LinearForm<f64> for HeatSourceForm<F> {
    fn integrate_point(
        &self,
        test: &BasisTabulation,
        weight: f64,
        local_vector: &mut LocalVector<f64>,
    ) {
        let [x, y, z] = test.physical_coords;
        let q = (self.source_fn)(x, y, z);
        let w = weight * test.det_j;

        for i in 0..test.n_dofs {
            local_vector.add(i, test.value(i) * q * w);
        }
    }
}

/// Convection boundary Robin form (matrix part): `a(T, v) = integral h * T * v dGamma`
///
/// The convection BC `q = h * (T - T_inf)` splits into:
/// - Matrix contribution: `h * integral T * v dGamma` (this form)
/// - RHS contribution: `h * T_inf * integral v dGamma` (use `ConvectionRhsForm`)
///
/// Delegates to [`WeightedMassForm`] with `coefficient = h`.
#[derive(Debug, Clone, Copy)]
pub struct ConvectionMatrixForm(WeightedMassForm);

impl ConvectionMatrixForm {
    pub fn new(conv: &ConvectionCoeff) -> Self {
        Self(WeightedMassForm {
            coefficient: conv.h,
        })
    }
}

impl BilinearForm<f64> for ConvectionMatrixForm {
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

/// Convection boundary RHS form: `l(v) = integral h * T_inf * v dGamma`
#[derive(Debug, Clone, Copy)]
pub struct ConvectionRhsForm {
    pub h: f64,
    pub t_ambient: f64,
}

impl ConvectionRhsForm {
    pub fn new(conv: &ConvectionCoeff) -> Self {
        Self {
            h: conv.h,
            t_ambient: conv.t_ambient,
        }
    }
}

impl LinearForm<f64> for ConvectionRhsForm {
    fn integrate_point(
        &self,
        test: &BasisTabulation,
        weight: f64,
        local_vector: &mut LocalVector<f64>,
    ) {
        let w = weight * test.det_j * self.h * self.t_ambient;
        for i in 0..test.n_dofs {
            local_vector.add(i, test.value(i) * w);
        }
    }
}

/// Radiation boundary nonlinear residual: `R = integral eps * sigma * (T^4 - T_inf^4) * v dGamma`
///
/// For use with Newton-Raphson. Implements the nonlinear radiation term
/// `q = eps * sigma * (T^4 - T_inf^4)` on boundary faces.
///
/// The linearized tangent (for Newton) is: `dR/dT = integral 4 * eps * sigma * T^3 * v * N dGamma`
pub struct RadiationResidual {
    /// Surface emissivity (0-1).
    pub emissivity: f64,
    /// Ambient radiation temperature (K).
    pub t_ambient: f64,
}

impl RadiationResidual {
    pub fn new(emissivity: f64, t_ambient: f64) -> Self {
        Self {
            emissivity,
            t_ambient,
        }
    }

    /// Evaluate the radiation heat flux at a given surface temperature.
    pub fn heat_flux(&self, t_surface: f64) -> f64 {
        self.emissivity * STEFAN_BOLTZMANN * (t_surface.powi(4) - self.t_ambient.powi(4))
    }

    /// Linearized radiation coefficient at a given temperature: `4 * eps * sigma * T^3`.
    pub fn linearized_h(&self, t_surface: f64) -> f64 {
        4.0 * self.emissivity * STEFAN_BOLTZMANN * t_surface.powi(3)
    }
}

/// Joule heating source for voice coil: `Q = I^2 * R / Volume`
///
/// The heat generated by current flowing through a resistive coil.
#[derive(Debug, Clone, Copy)]
pub struct JouleHeatingSource {
    /// Electrical resistance (ohms).
    pub resistance: f64,
    /// RMS current (amperes).
    pub current_rms: f64,
    /// Coil volume (m^3).
    pub volume: f64,
}

impl JouleHeatingSource {
    pub fn new(resistance: f64, current_rms: f64, volume: f64) -> Self {
        Self {
            resistance,
            current_rms,
            volume,
        }
    }

    /// Volumetric heat generation rate Q (W/m^3).
    pub fn heat_generation_rate(&self) -> f64 {
        self.resistance * self.current_rms.powi(2) / self.volume
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::basis::PolynomialDegree;
    use crate::mesh::unit_square_triangles;
    use crate::spaces::{FunctionSpace, H1Space};

    #[test]
    fn test_conduction_form_symmetry() {
        let mesh = unit_square_triangles(2);
        let space = H1Space::new(PolynomialDegree::P1);
        let tab = space.tabulate(&mesh, 0, 1.0 / 3.0, 1.0 / 3.0, 0.0);
        let n = tab.n_dofs;

        let mat = ThermalMaterial::copper();
        let form = ConductionForm::new(&mat);
        let mut k = LocalMatrix::zeros(n);
        form.integrate_point(&tab, &tab, 1.0, &mut k);

        for i in 0..n {
            for j in i + 1..n {
                assert!(
                    (k.get(i, j) - k.get(j, i)).abs() < 1e-10,
                    "Not symmetric at ({i},{j})"
                );
            }
        }
    }

    #[test]
    fn test_heat_capacity_positive_diagonal() {
        let mesh = unit_square_triangles(2);
        let space = H1Space::new(PolynomialDegree::P1);
        let tab = space.tabulate(&mesh, 0, 1.0 / 3.0, 1.0 / 3.0, 0.0);
        let n = tab.n_dofs;

        let mat = ThermalMaterial::steel();
        let form = HeatCapacityForm::new(&mat);
        let mut m = LocalMatrix::zeros(n);
        form.integrate_point(&tab, &tab, 1.0, &mut m);

        for i in 0..n {
            assert!(m.get(i, i) > 0.0);
        }
    }

    #[test]
    fn test_radiation_heat_flux() {
        let rad = RadiationResidual::new(0.9, 300.0);
        // At T = T_inf: no radiation
        assert!(rad.heat_flux(300.0).abs() < 1e-10);
        // At T > T_inf: positive flux (heat leaving surface)
        assert!(rad.heat_flux(400.0) > 0.0);
        // At T < T_inf: negative flux (heat entering surface)
        assert!(rad.heat_flux(200.0) < 0.0);
    }

    #[test]
    fn test_radiation_linearized_h() {
        let rad = RadiationResidual::new(1.0, 0.0);
        let h = rad.linearized_h(300.0);
        // h = 4 * sigma * T^3 at emissivity=1
        let expected = 4.0 * STEFAN_BOLTZMANN * 300.0_f64.powi(3);
        assert!((h - expected).abs() < 1e-6);
    }

    #[test]
    fn test_joule_heating() {
        let source = JouleHeatingSource::new(8.0, 1.0, 1e-6);
        // Q = R * I^2 / V = 8 * 1 / 1e-6 = 8e6 W/m^3
        assert!((source.heat_generation_rate() - 8e6).abs() < 1.0);
    }

    #[test]
    fn test_advection_form_uniform_flow_is_finite_and_nonsymmetric() {
        // Uniform flow u = (1, 0): advection is skew-ish, never symmetric.
        let mesh = unit_square_triangles(2);
        let space = H1Space::new(PolynomialDegree::P1);
        let tab = space.tabulate(&mesh, 0, 1.0 / 3.0, 1.0 / 3.0, 0.0);
        let n = tab.n_dofs;

        let form = AdvectionForm {
            rho_c: 1.0,
            velocity_fn: |_, _, _| vec![1.0, 0.0, 0.0],
        };
        let mut k = LocalMatrix::zeros(n);
        form.integrate_point(&tab, &tab, 1.0, &mut k);

        let mut off_diag = 0.0;
        let mut asymmetry = 0.0;
        for i in 0..n {
            for j in 0..n {
                assert!(k.get(i, j).is_finite());
                if i != j {
                    off_diag += k.get(i, j).abs();
                }
                asymmetry += (k.get(i, j) - k.get(j, i)).abs();
            }
        }
        assert!(off_diag > 0.0, "uniform flow must couple distinct DOFs");
        assert!(asymmetry > 0.0, "advection matrix must not be symmetric");
    }

    #[test]
    #[should_panic(expected = "AdvectionForm velocity closure must return at least")]
    fn test_advection_form_short_velocity_panics_in_debug() {
        // A closure returning fewer components than the mesh dimension must
        // fail loudly instead of silently truncating the dot product.
        let mesh = unit_square_triangles(2);
        let space = H1Space::new(PolynomialDegree::P1);
        let tab = space.tabulate(&mesh, 0, 1.0 / 3.0, 1.0 / 3.0, 0.0);
        let n = tab.n_dofs;

        let form = AdvectionForm {
            rho_c: 1.0,
            velocity_fn: |_, _, _| vec![1.0],
        };
        let mut k = LocalMatrix::zeros(n);
        form.integrate_point(&tab, &tab, 1.0, &mut k);
    }
}
