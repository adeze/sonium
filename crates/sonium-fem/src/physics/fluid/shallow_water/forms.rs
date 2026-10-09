//! Linearized shallow-water weak forms.

use crate::forms::{BilinearForm, LinearForm, LocalMatrix, LocalVector};
use crate::spaces::BasisTabulation;

use super::materials::ShallowWaterMaterial;

/// Gravity-wave stiffness `integral g h grad eta.grad v`.
#[derive(Debug, Clone, Copy)]
pub struct ShallowWaterGravityWaveForm {
    pub material: ShallowWaterMaterial,
}

impl BilinearForm<f64> for ShallowWaterGravityWaveForm {
    fn integrate_point(
        &self,
        test: &BasisTabulation,
        trial: &BasisTabulation,
        weight: f64,
        local_matrix: &mut LocalMatrix<f64>,
    ) {
        let w = weight * test.det_j * self.material.gravity * self.material.mean_depth;
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

/// Free-surface mass `integral eta v`.
#[derive(Debug, Clone, Copy)]
pub struct FreeSurfaceMassForm;

impl BilinearForm<f64> for FreeSurfaceMassForm {
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
                local_matrix.add(i, j, test.value(i) * trial.value(j) * w);
            }
        }
    }
}

/// Bed-slope forcing `-integral g h grad(z_b).grad(v)`.
///
/// This weak-gradient form is discretely well balanced with
/// [`ShallowWaterGravityWaveForm`]: for a lake at rest,
/// `eta + z_b = constant`, the assembled vectors satisfy `K eta = f_bed`
/// to roundoff when `eta` and `z_b` are represented in the same H1 space.
#[derive(Debug, Clone, Copy)]
pub struct BedSlopeSourceForm {
    pub material: ShallowWaterMaterial,
    pub bed_slope: [f64; 3],
    pub flow_direction: Option<[f64; 3]>,
}

impl BedSlopeSourceForm {
    pub fn new(material: ShallowWaterMaterial, bed_slope: [f64; 3]) -> Self {
        Self {
            material,
            bed_slope,
            flow_direction: None,
        }
    }

    pub fn with_flow_direction(
        material: ShallowWaterMaterial,
        bed_slope: [f64; 3],
        flow_direction: [f64; 3],
    ) -> Self {
        Self {
            material,
            bed_slope,
            flow_direction: Some(flow_direction),
        }
    }

    #[inline]
    fn projected_bed_slope(&self, spatial_dim: usize) -> [f64; 3] {
        let mut projected = [0.0; 3];
        if let Some(direction) = self.flow_direction {
            let norm2 = direction
                .iter()
                .take(spatial_dim)
                .map(|d| d * d)
                .sum::<f64>();
            if norm2 > 0.0 {
                let inv_len = 1.0 / norm2.sqrt();
                let projection = self
                    .bed_slope
                    .iter()
                    .take(spatial_dim)
                    .zip(direction.iter())
                    .map(|(s, d)| s * d * inv_len)
                    .sum::<f64>();
                for d in 0..spatial_dim {
                    projected[d] = projection * direction[d] * inv_len;
                }
            } else {
                projected[..spatial_dim].copy_from_slice(&self.bed_slope[..spatial_dim]);
            }
        } else {
            projected[..spatial_dim].copy_from_slice(&self.bed_slope[..spatial_dim]);
        }
        projected
    }
}

impl LinearForm<f64> for BedSlopeSourceForm {
    fn integrate_point(
        &self,
        test: &BasisTabulation,
        weight: f64,
        local_vector: &mut LocalVector<f64>,
    ) {
        let bed_slope = self.projected_bed_slope(test.spatial_dim);
        let coefficient = -weight * test.det_j * self.material.gravity * self.material.mean_depth;
        for i in 0..test.n_dofs {
            let gradient_dot_slope = (0..test.spatial_dim)
                .map(|d| test.grad(i, d) * bed_slope[d])
                .sum::<f64>();
            local_vector.add(i, coefficient * gradient_dot_slope);
        }
    }
}

/// Linear friction term for a scalar momentum component:
/// `a(h, v) = ∫ beta(h, |u|) h v d\Omega`.
#[derive(Debug, Clone, Copy)]
pub struct ManningFrictionForm {
    pub coefficient: f64,
}

impl ManningFrictionForm {
    pub fn new(material: &ShallowWaterMaterial, velocity_magnitude: f64, water_depth: f64) -> Self {
        let depth = water_depth.max(1e-12);
        Self {
            coefficient: material.manning_friction_slope(velocity_magnitude) / depth,
        }
    }
}

impl BilinearForm<f64> for ManningFrictionForm {
    fn integrate_point(
        &self,
        test: &BasisTabulation,
        trial: &BasisTabulation,
        weight: f64,
        local_matrix: &mut LocalMatrix<f64>,
    ) {
        let w = weight * test.det_j * self.coefficient;
        for i in 0..test.n_dofs {
            for j in 0..trial.n_dofs {
                local_matrix.add(i, j, test.value(i) * trial.value(j) * w);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wave_speed_matches_linear_shallow_water() {
        let material = ShallowWaterMaterial {
            gravity: 9.81,
            mean_depth: 4.0,
            manning_n: 0.03,
        };
        assert!((material.wave_speed() - (39.24_f64).sqrt()).abs() < 1.0e-12);
        assert!(material.manning_friction_slope(2.0) > 0.0);
    }

    #[test]
    fn gravity_wave_form_uses_gh_laplacian() {
        let material = ShallowWaterMaterial {
            gravity: 10.0,
            mean_depth: 2.0,
            manning_n: 0.0,
        };
        let tab = BasisTabulation {
            n_dofs: 2,
            spatial_dim: 2,
            values: vec![0.5, 0.5],
            derivatives: vec![1.0, 2.0, 3.0, 4.0],
            hessians: vec![],
            det_j: 0.5,
            physical_coords: [0.0; 3],
        };
        let mut local = LocalMatrix::zeros(2);
        ShallowWaterGravityWaveForm { material }.integrate_point(&tab, &tab, 2.0, &mut local);
        assert!((local.get(0, 1) - 20.0 * (1.0 * 3.0 + 2.0 * 4.0)).abs() < 1.0e-12);
    }

    #[test]
    fn bed_slope_source_uses_signed_projection() {
        let material = ShallowWaterMaterial {
            gravity: 10.0,
            mean_depth: 2.0,
            manning_n: 0.0,
        };
        let tab = BasisTabulation {
            n_dofs: 1,
            spatial_dim: 2,
            values: vec![0.5],
            derivatives: vec![0.25, -0.5],
            hessians: vec![],
            det_j: 1.0,
            physical_coords: [0.0; 3],
        };
        let mut rhs = LocalVector::zeros(1);
        BedSlopeSourceForm::new(material, [3.0, -4.0, 99.0]).integrate_point(&tab, 1.0, &mut rhs);
        assert!((rhs.values[0] - -55.0).abs() < 1.0e-12);
    }

    #[test]
    fn bed_slope_source_uses_flow_projection_when_provided() {
        let material = ShallowWaterMaterial {
            gravity: 10.0,
            mean_depth: 2.0,
            manning_n: 0.0,
        };
        let tab = BasisTabulation {
            n_dofs: 1,
            spatial_dim: 2,
            values: vec![0.5],
            derivatives: vec![2.0, 4.0],
            hessians: vec![],
            det_j: 1.0,
            physical_coords: [0.0; 3],
        };
        let mut rhs = LocalVector::zeros(1);
        BedSlopeSourceForm::with_flow_direction(material, [3.0, -4.0, 99.0], [1.0, 1.0, 0.0])
            .integrate_point(&tab, 1.0, &mut rhs);
        // Projection of (3,-4) onto the (1,1) direction is (-0.5,-0.5).
        // Its dot product with grad(v)=(2,4) is -3.
        assert!((rhs.values[0] - 60.0).abs() < 1.0e-10);
    }

    #[test]
    fn manning_form_uses_friction_as_mass_like_term() {
        let material = ShallowWaterMaterial {
            gravity: 10.0,
            mean_depth: 2.0,
            manning_n: 0.5,
        };
        let tab = BasisTabulation {
            n_dofs: 2,
            spatial_dim: 2,
            values: vec![0.2, 0.8],
            derivatives: vec![0.0; 4],
            hessians: vec![],
            det_j: 2.0,
            physical_coords: [0.0; 3],
        };
        let mut local = LocalMatrix::zeros(2);
        ManningFrictionForm::new(&material, 3.0, material.mean_depth)
            .integrate_point(&tab, &tab, 1.0, &mut local);
        let coeff = material.manning_friction_slope(3.0) / material.mean_depth;
        assert!((local.get(0, 1) - 0.2 * 0.8 * 1.0 * 2.0 * coeff).abs() < 1.0e-12);
    }
}
