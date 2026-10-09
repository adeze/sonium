//! Region-local conjugate heat-transfer forms.

use crate::forms::{BilinearForm, LocalMatrix};
use crate::spaces::BasisTabulation;

use super::{ConjugateRegion, ConjugateRegionKind};

#[derive(Debug, Clone, Copy)]
pub struct ChtConductionForm {
    pub thermal_conductivity: f64,
}

impl From<ConjugateRegion> for ChtConductionForm {
    fn from(region: ConjugateRegion) -> Self {
        Self {
            thermal_conductivity: region.thermal_conductivity,
        }
    }
}

impl BilinearForm<f64> for ChtConductionForm {
    fn integrate_point(
        &self,
        test: &BasisTabulation,
        trial: &BasisTabulation,
        weight: f64,
        local_matrix: &mut LocalMatrix<f64>,
    ) {
        let w = weight * test.det_j * self.thermal_conductivity;
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

#[derive(Debug, Clone, Copy)]
pub struct ChtCapacityForm {
    pub volumetric_heat_capacity: f64,
}

impl From<ConjugateRegion> for ChtCapacityForm {
    fn from(region: ConjugateRegion) -> Self {
        Self {
            volumetric_heat_capacity: region.volumetric_heat_capacity(),
        }
    }
}

impl BilinearForm<f64> for ChtCapacityForm {
    fn integrate_point(
        &self,
        test: &BasisTabulation,
        trial: &BasisTabulation,
        weight: f64,
        local_matrix: &mut LocalMatrix<f64>,
    ) {
        let w = weight * test.det_j * self.volumetric_heat_capacity;
        for i in 0..test.n_dofs {
            for j in 0..trial.n_dofs {
                local_matrix.add(i, j, test.value(i) * trial.value(j) * w);
            }
        }
    }
}

/// Fluid-only advective heat transport `integral rho cp (u.grad T) q`.
#[derive(Debug, Clone, Copy)]
pub struct ChtAdvectionForm {
    pub volumetric_heat_capacity: f64,
    pub velocity: [f64; 3],
}

impl From<ConjugateRegion> for ChtAdvectionForm {
    fn from(region: ConjugateRegion) -> Self {
        Self {
            volumetric_heat_capacity: if region.kind == ConjugateRegionKind::Fluid {
                region.volumetric_heat_capacity()
            } else {
                0.0
            },
            velocity: region.velocity,
        }
    }
}

impl BilinearForm<f64> for ChtAdvectionForm {
    fn integrate_point(
        &self,
        test: &BasisTabulation,
        trial: &BasisTabulation,
        weight: f64,
        local_matrix: &mut LocalMatrix<f64>,
    ) {
        let w = weight * test.det_j * self.volumetric_heat_capacity;
        for i in 0..test.n_dofs {
            for j in 0..trial.n_dofs {
                let directional_gradient = (0..test.spatial_dim)
                    .map(|d| self.velocity[d] * trial.grad(j, d))
                    .sum::<f64>();
                local_matrix.add(i, j, test.value(i) * directional_gradient * w);
            }
        }
    }
}

/// Fluid advection with a spatially varying velocity evaluated at each
/// quadrature point.
///
/// This is the coefficient-field counterpart of [`ChtAdvectionForm`] and
/// assembles `rho*c_p * v(x)·grad(T)` without freezing a flow profile at an
/// element centroid.
#[derive(Clone, Copy)]
pub struct ChtAdvectionFieldForm<F> {
    /// Volumetric heat capacity `rho*c_p` in J/(m³ K).
    pub volumetric_heat_capacity: f64,
    /// Velocity field in m/s as a function of physical coordinates.
    pub velocity: F,
}

impl<F> BilinearForm<f64> for ChtAdvectionFieldForm<F>
where
    F: Fn([f64; 3]) -> [f64; 3] + Send + Sync,
{
    fn integrate_point(
        &self,
        test: &BasisTabulation,
        trial: &BasisTabulation,
        weight: f64,
        local_matrix: &mut LocalMatrix<f64>,
    ) {
        let velocity = (self.velocity)(test.physical_coords);
        let w = weight * test.det_j * self.volumetric_heat_capacity;
        for i in 0..test.n_dofs {
            for j in 0..trial.n_dofs {
                let v_dot_grad = (0..test.spatial_dim)
                    .map(|dimension| velocity[dimension] * trial.grad(j, dimension))
                    .sum::<f64>();
                local_matrix.add(i, j, test.value(i) * v_dot_grad * w);
            }
        }
    }
}

/// Report equal-and-opposite flux conservation at a conforming interface.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct InterfaceHeatBalance {
    pub solid_outward_heat_rate: f64,
    pub fluid_outward_heat_rate: f64,
    pub imbalance: f64,
    pub relative_imbalance: f64,
}

impl InterfaceHeatBalance {
    pub fn new(solid_outward_heat_rate: f64, fluid_outward_heat_rate: f64) -> Self {
        let imbalance = solid_outward_heat_rate + fluid_outward_heat_rate;
        let scale = solid_outward_heat_rate
            .abs()
            .max(fluid_outward_heat_rate.abs())
            .max(f64::MIN_POSITIVE);
        Self {
            solid_outward_heat_rate,
            fluid_outward_heat_rate,
            imbalance,
            relative_imbalance: imbalance.abs() / scale,
        }
    }

    pub fn is_conservative(&self, tolerance: f64) -> bool {
        tolerance.is_finite() && tolerance >= 0.0 && self.relative_imbalance <= tolerance
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn solid_region_has_exact_zero_advection() {
        let solid = ConjugateRegion::solid(10.0, 1000.0, 500.0);
        let form = ChtAdvectionForm::from(solid);
        assert_eq!(form.volumetric_heat_capacity, 0.0);
    }

    #[test]
    fn interface_balance_uses_opposing_normals() {
        let balance = InterfaceHeatBalance::new(100.0, -100.0);
        assert_eq!(balance.imbalance, 0.0);
        assert!(balance.is_conservative(1e-12));
    }

    #[test]
    fn spatial_advection_matches_constant_form_for_constant_field() {
        let tabulation = BasisTabulation {
            n_dofs: 2,
            spatial_dim: 2,
            values: vec![0.25, 0.75],
            derivatives: vec![1.0, -2.0, -1.0, 2.0],
            hessians: Vec::new(),
            det_j: 0.5,
            physical_coords: [0.3, 0.7, 0.0],
        };
        let constant = ChtAdvectionForm {
            volumetric_heat_capacity: 12.0,
            velocity: [2.0, -1.0, 0.0],
        };
        let field = ChtAdvectionFieldForm {
            volumetric_heat_capacity: 12.0,
            velocity: |_| [2.0, -1.0, 0.0],
        };
        let mut constant_matrix = LocalMatrix::zeros(2);
        let mut field_matrix = LocalMatrix::zeros(2);
        constant.integrate_point(&tabulation, &tabulation, 1.0, &mut constant_matrix);
        field.integrate_point(&tabulation, &tabulation, 1.0, &mut field_matrix);
        for row in 0..2 {
            for column in 0..2 {
                assert_eq!(
                    constant_matrix.get(row, column),
                    field_matrix.get(row, column)
                );
            }
        }
    }
}
