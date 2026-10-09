//! Weak forms for the incompressible Reynolds equation.

use crate::forms::{BilinearForm, LinearForm, LocalMatrix, LocalVector};
use crate::spaces::BasisTabulation;

use super::Lubricant;

/// Reynolds diffusion form
/// `integral h^3/(12 mu) grad(p).grad(q) dA`.
pub struct ReynoldsPressureForm<H>
where
    H: Fn(f64, f64, f64) -> f64 + Send + Sync,
{
    pub lubricant: Lubricant,
    pub film_thickness: H,
}

impl<H> BilinearForm<f64> for ReynoldsPressureForm<H>
where
    H: Fn(f64, f64, f64) -> f64 + Send + Sync,
{
    fn integrate_point(
        &self,
        test: &BasisTabulation,
        trial: &BasisTabulation,
        weight: f64,
        local_matrix: &mut LocalMatrix<f64>,
    ) {
        let [x, y, z] = test.physical_coords;
        let coefficient = self.lubricant.mobility((self.film_thickness)(x, y, z));
        let w = weight * test.det_j * coefficient;
        for i in 0..test.n_dofs {
            for j in 0..trial.n_dofs {
                let grad_dot = (0..test.spatial_dim)
                    .map(|d| test.grad(i, d) * trial.grad(j, d))
                    .sum::<f64>();
                local_matrix.add(i, j, grad_dot * w);
            }
        }
    }
}

/// Couette/wedge forcing
/// `integral 0.5 h (U1 + U2).grad(q) dA`.
pub struct ReynoldsWedgeSourceForm<H, U>
where
    H: Fn(f64, f64, f64) -> f64 + Send + Sync,
    U: Fn(f64, f64, f64) -> [f64; 3] + Send + Sync,
{
    pub film_thickness: H,
    /// Sum of the two bounding-surface velocities, `U1 + U2`.
    pub surface_velocity_sum: U,
}

impl<H, U> LinearForm<f64> for ReynoldsWedgeSourceForm<H, U>
where
    H: Fn(f64, f64, f64) -> f64 + Send + Sync,
    U: Fn(f64, f64, f64) -> [f64; 3] + Send + Sync,
{
    fn integrate_point(
        &self,
        test: &BasisTabulation,
        weight: f64,
        local_vector: &mut LocalVector<f64>,
    ) {
        let [x, y, z] = test.physical_coords;
        let h = (self.film_thickness)(x, y, z);
        let velocity = (self.surface_velocity_sum)(x, y, z);
        let w = 0.5 * h * weight * test.det_j;
        for i in 0..test.n_dofs {
            let projection = (0..test.spatial_dim)
                .map(|d| velocity[d] * test.grad(i, d))
                .sum::<f64>();
            local_vector.add(i, w * projection);
        }
    }
}

/// Squeeze forcing `-integral dh/dt q dA`.
///
/// The sign follows the weak form used by [`ReynoldsPressureForm`], so a
/// closing film (`dh/dt < 0`) creates a positive right-hand side.
pub struct ReynoldsSqueezeSourceForm<R>
where
    R: Fn(f64, f64, f64) -> f64 + Send + Sync,
{
    pub film_rate: R,
}

impl<R> LinearForm<f64> for ReynoldsSqueezeSourceForm<R>
where
    R: Fn(f64, f64, f64) -> f64 + Send + Sync,
{
    fn integrate_point(
        &self,
        test: &BasisTabulation,
        weight: f64,
        local_vector: &mut LocalVector<f64>,
    ) {
        let [x, y, z] = test.physical_coords;
        let source = -(self.film_rate)(x, y, z);
        let w = weight * test.det_j;
        for i in 0..test.n_dofs {
            local_vector.add(i, source * test.value(i) * w);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tabulation() -> BasisTabulation {
        BasisTabulation {
            n_dofs: 2,
            spatial_dim: 2,
            values: vec![0.25, 0.75],
            derivatives: vec![1.0, -2.0, 3.0, 4.0],
            hessians: vec![],
            det_j: 2.0,
            physical_coords: [0.5, 0.25, 0.0],
        }
    }

    fn lubricant() -> Lubricant {
        Lubricant {
            dynamic_viscosity: 2.0,
            density: 850.0,
            ambient_pressure: 0.0,
            minimum_film_thickness: 1.0e-9,
        }
    }

    #[test]
    fn pressure_form_uses_cubic_film_mobility() {
        let tab = tabulation();
        let form = ReynoldsPressureForm {
            lubricant: lubricant(),
            film_thickness: |_, _, _| 2.0,
        };
        let mut matrix = LocalMatrix::zeros(2);
        form.integrate_point(&tab, &tab, 0.5, &mut matrix);
        let mobility = 8.0 / 24.0;
        assert!((matrix.get(0, 1) - mobility * (1.0 * 3.0 + -2.0 * 4.0)).abs() < 1e-12);
    }

    #[test]
    fn closing_film_has_positive_squeeze_rhs() {
        let tab = tabulation();
        let form = ReynoldsSqueezeSourceForm {
            film_rate: |_, _, _| -0.5,
        };
        let mut rhs = LocalVector::zeros(2);
        form.integrate_point(&tab, 1.0, &mut rhs);
        assert_eq!(rhs.values, vec![0.25, 0.75]);
    }

    #[test]
    fn zero_surface_velocity_is_exact_wedge_null_state() {
        let tab = tabulation();
        let form = ReynoldsWedgeSourceForm {
            film_thickness: |_, _, _| 1.0,
            surface_velocity_sum: |_, _, _| [0.0; 3],
        };
        let mut rhs = LocalVector::zeros(2);
        form.integrate_point(&tab, 1.0, &mut rhs);
        assert_eq!(rhs.values, vec![0.0, 0.0]);
    }
}
