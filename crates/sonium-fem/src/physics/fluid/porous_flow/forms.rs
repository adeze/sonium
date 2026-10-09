//! Weak forms for porous-media flow.

use crate::forms::{BilinearForm, LinearForm, LocalMatrix, LocalVector};
use crate::spaces::BasisTabulation;

use super::materials::{BiotMaterial, DarcyMaterial, RichardsMaterial};

/// Darcy pressure form: `a(p,q) = integral (k/mu grad p) . grad q`.
#[derive(Debug, Clone, Copy)]
pub struct DarcyPressureForm {
    pub mobility: [[f64; 3]; 3],
}

impl DarcyPressureForm {
    pub fn new(material: &DarcyMaterial) -> Self {
        Self {
            mobility: material.hydraulic_mobility(),
        }
    }
}

impl BilinearForm<f64> for DarcyPressureForm {
    fn integrate_point(
        &self,
        test: &BasisTabulation,
        trial: &BasisTabulation,
        weight: f64,
        local_matrix: &mut LocalMatrix<f64>,
    ) {
        let w = weight * test.det_j;
        for i in 0..test.n_dofs {
            for j in 0..trial.n_dofs {
                let mut value = 0.0;
                for a in 0..test.spatial_dim {
                    for b in 0..test.spatial_dim {
                        value += test.grad(i, a) * self.mobility[a][b] * trial.grad(j, b);
                    }
                }
                local_matrix.add(i, j, value * w);
            }
        }
    }
}

/// Storage/capacity form: `m(p,q) = integral S_s p q`.
#[derive(Debug, Clone, Copy)]
pub struct DarcyStorageForm {
    pub specific_storage: f64,
}

impl DarcyStorageForm {
    pub fn new(material: &DarcyMaterial) -> Self {
        Self {
            specific_storage: material.specific_storage,
        }
    }
}

impl BilinearForm<f64> for DarcyStorageForm {
    fn integrate_point(
        &self,
        test: &BasisTabulation,
        trial: &BasisTabulation,
        weight: f64,
        local_matrix: &mut LocalMatrix<f64>,
    ) {
        let w = weight * test.det_j * self.specific_storage;
        for i in 0..test.n_dofs {
            for j in 0..trial.n_dofs {
                local_matrix.add(i, j, test.value(i) * trial.value(j) * w);
            }
        }
    }
}

/// Richards pressure form with saturation-dependent mobility.
#[derive(Debug, Clone, Copy)]
pub struct RichardsFlowForm {
    pub mobility: [[f64; 3]; 3],
}

impl RichardsFlowForm {
    pub fn new(material: &RichardsMaterial, saturation: f64) -> Self {
        Self {
            mobility: material.hydraulic_mobility(saturation),
        }
    }
}

impl BilinearForm<f64> for RichardsFlowForm {
    fn integrate_point(
        &self,
        test: &BasisTabulation,
        trial: &BasisTabulation,
        weight: f64,
        local_matrix: &mut LocalMatrix<f64>,
    ) {
        let w = weight * test.det_j;
        for i in 0..test.n_dofs {
            for j in 0..trial.n_dofs {
                let mut value = 0.0;
                for a in 0..test.spatial_dim {
                    for b in 0..test.spatial_dim {
                        value += test.grad(i, a) * self.mobility[a][b] * trial.grad(j, b);
                    }
                }
                local_matrix.add(i, j, value * w);
            }
        }
    }
}

/// Richards moisture-storage (capacitance) form `m(c)`.
#[derive(Debug, Clone, Copy)]
pub struct RichardsStorageForm {
    pub moisture_capacity: f64,
}

impl RichardsStorageForm {
    pub fn new(material: &RichardsMaterial) -> Self {
        Self {
            moisture_capacity: material.moisture_capacity,
        }
    }
}

impl BilinearForm<f64> for RichardsStorageForm {
    fn integrate_point(
        &self,
        test: &BasisTabulation,
        trial: &BasisTabulation,
        weight: f64,
        local_matrix: &mut LocalMatrix<f64>,
    ) {
        let w = weight * test.det_j * self.moisture_capacity;
        for i in 0..test.n_dofs {
            for j in 0..trial.n_dofs {
                local_matrix.add(i, j, test.value(i) * trial.value(j) * w);
            }
        }
    }
}

/// Volumetric source/sink form: `l(q) = integral s q`.
pub struct DarcySourceForm<F: Fn(f64, f64, f64) -> f64 + Send + Sync> {
    pub source_fn: F,
}

impl<F: Fn(f64, f64, f64) -> f64 + Send + Sync> LinearForm<f64> for DarcySourceForm<F> {
    fn integrate_point(
        &self,
        test: &BasisTabulation,
        weight: f64,
        local_vector: &mut LocalVector<f64>,
    ) {
        let [x, y, z] = test.physical_coords;
        let s = (self.source_fn)(x, y, z);
        let w = weight * test.det_j;
        for i in 0..test.n_dofs {
            local_vector.add(i, test.value(i) * s * w);
        }
    }
}

/// Biot pressure-displacement coupling contribution
/// `b(u,q) = integral alpha div(u) q`.
#[derive(Debug, Clone, Copy)]
pub struct BiotDivergenceCouplingForm {
    pub biot_alpha: f64,
}

impl BiotDivergenceCouplingForm {
    pub fn new(material: &BiotMaterial) -> Self {
        Self {
            biot_alpha: material.biot_alpha,
        }
    }
}

impl BilinearForm<f64> for BiotDivergenceCouplingForm {
    fn integrate_point(
        &self,
        test: &BasisTabulation,
        trial: &BasisTabulation,
        weight: f64,
        local_matrix: &mut LocalMatrix<f64>,
    ) {
        let dim = test.spatial_dim;
        let w = weight * test.det_j * self.biot_alpha;
        for pressure_dof in 0..test.n_dofs {
            for u_dof in 0..trial.n_dofs {
                let comp = u_dof % dim;
                let grad = if dim > 0 && comp < dim {
                    trial.grad(u_dof, comp)
                } else {
                    0.0
                };
                local_matrix.add(pressure_dof, u_dof, test.value(pressure_dof) * grad * w);
            }
        }
    }
}

/// Biot coupling transpose from pressure pressure to displacement residual:
/// `a(u,q) += -alpha q div(v)`.
#[derive(Debug, Clone, Copy)]
pub struct PressureToDisplacementCouplingForm {
    pub biot_alpha: f64,
}

impl PressureToDisplacementCouplingForm {
    pub fn new(material: &BiotMaterial) -> Self {
        Self {
            biot_alpha: material.biot_alpha,
        }
    }
}

impl BilinearForm<f64> for PressureToDisplacementCouplingForm {
    fn integrate_point(
        &self,
        test: &BasisTabulation,
        trial: &BasisTabulation,
        weight: f64,
        local_matrix: &mut LocalMatrix<f64>,
    ) {
        let dim = test.spatial_dim;
        let w = -weight * test.det_j * self.biot_alpha;
        for u_dof in 0..test.n_dofs {
            let comp = u_dof % dim;
            let div_phi = test.grad(u_dof, comp);
            for p in 0..trial.n_dofs {
                local_matrix.add(u_dof, p, div_phi * trial.value(p) * w);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tab() -> BasisTabulation {
        BasisTabulation {
            n_dofs: 2,
            spatial_dim: 2,
            values: vec![0.25, 0.75],
            derivatives: vec![1.0, 2.0, -1.0, 0.5],
            hessians: vec![],
            det_j: 2.0,
            physical_coords: [1.0, 2.0, 0.0],
        }
    }

    #[test]
    fn darcy_tensor_form_matches_manual_contraction() {
        let form = DarcyPressureForm {
            mobility: [[2.0, 0.5, 0.0], [0.25, 3.0, 0.0], [0.0; 3]],
        };
        let t = tab();
        let mut local = LocalMatrix::zeros(2);
        form.integrate_point(&t, &t, 0.5, &mut local);

        let expected =
            1.0 * (1.0 * 2.0 * -1.0 + 1.0 * 0.5 * 0.5 + 2.0 * 0.25 * -1.0 + 2.0 * 3.0 * 0.5);
        assert!((local.get(0, 1) - expected).abs() < 1.0e-12);
    }

    #[test]
    fn storage_and_source_include_jacobian_and_weight() {
        let t = tab();
        let mut m = LocalMatrix::zeros(2);
        DarcyStorageForm {
            specific_storage: 4.0,
        }
        .integrate_point(&t, &t, 0.5, &mut m);
        assert!((m.get(0, 1) - 0.25 * 0.75 * 4.0).abs() < 1.0e-12);

        let mut v = LocalVector::zeros(2);
        DarcySourceForm {
            source_fn: |x, y, _| x + y,
        }
        .integrate_point(&t, 0.5, &mut v);
        assert!((v.values[1] - 0.75 * 3.0).abs() < 1.0e-12);
    }

    #[test]
    fn biot_material_recovers_undrained_bulk_modulus() {
        let material = BiotMaterial {
            biot_alpha: 0.8,
            storage: 0.02,
            drained_bulk_modulus: 10.0,
        };
        assert!((material.undrained_bulk_modulus() - 42.0).abs() < 1.0e-12);
    }

    #[test]
    fn richards_form_uses_saturation_dependent_mobility() {
        let material = RichardsMaterial {
            saturated_permeability: [[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 2.0]],
            viscosity: 1.0,
            residual_saturation: 0.05,
            saturation_exponent: 2.0,
            moisture_capacity: 1.0,
            retention: None,
        };
        let t = BasisTabulation {
            n_dofs: 2,
            spatial_dim: 2,
            values: vec![0.25, 0.75],
            derivatives: vec![1.0, 0.0, -1.0, 0.0],
            hessians: vec![],
            det_j: 2.0,
            physical_coords: [0.0, 0.0, 0.0],
        };
        let mut local_half = LocalMatrix::zeros(2);
        RichardsFlowForm::new(&material, 0.5).integrate_point(&t, &t, 0.5, &mut local_half);
        let mut local_sat = LocalMatrix::zeros(2);
        RichardsFlowForm::new(&material, 1.0).integrate_point(&t, &t, 0.5, &mut local_sat);
        let kr05 = ((0.5 - 0.05) / 0.95f64).powf(2.0);
        let expected = local_sat.get(0, 0) * kr05;
        assert!((local_half.get(0, 0) - expected).abs() < 1.0e-12);
    }

    #[test]
    fn pressure_to_displacement_coupling_maps_pressure_shape_to_divergence() {
        let material = BiotMaterial {
            biot_alpha: 2.0,
            storage: 1.0,
            drained_bulk_modulus: 1.0,
        };
        let t_disp = BasisTabulation {
            n_dofs: 4,
            spatial_dim: 2,
            values: vec![1.0; 4],
            derivatives: vec![1.0, 0.0, 0.0, 1.0, 2.0, 1.0, 1.0, 0.0],
            hessians: vec![],
            det_j: 1.0,
            physical_coords: [0.0; 3],
        };
        let t_p = BasisTabulation {
            n_dofs: 2,
            spatial_dim: 2,
            values: vec![0.3, 0.7],
            derivatives: vec![0.0; 4],
            hessians: vec![],
            det_j: 1.0,
            physical_coords: [0.0; 3],
        };
        let mut local = LocalMatrix::zeros(4);
        PressureToDisplacementCouplingForm {
            biot_alpha: material.biot_alpha,
        }
        .integrate_point(&t_disp, &t_p, 1.0, &mut local);
        assert!((local.get(0, 0) + 2.0 * t_p.values[0]).abs() < 1.0e-12);
    }
}
