//! Composite laminate membrane form.

use crate::forms::{BilinearForm, LocalMatrix};
use crate::spaces::BasisTabulation;

use super::materials::Laminate;

/// Plane-stress membrane stiffness using laminate `A` matrix.
#[derive(Debug, Clone)]
pub struct LaminateMembraneForm {
    pub a_matrix: [[f64; 3]; 3],
}

/// Kirchhoff plate bending stiffness using laminate `D` matrix.
#[derive(Debug, Clone)]
pub struct LaminateBendingForm {
    pub d_matrix: [[f64; 3]; 3],
}

impl LaminateBendingForm {
    pub fn new(laminate: &Laminate) -> Self {
        Self {
            d_matrix: laminate.bending_stiffness(),
        }
    }
}

impl BilinearForm<f64> for LaminateBendingForm {
    fn integrate_point(
        &self,
        test: &BasisTabulation,
        trial: &BasisTabulation,
        weight: f64,
        local_matrix: &mut LocalMatrix<f64>,
    ) {
        if test.spatial_dim != 2 || !test.has_hessians() || !trial.has_hessians() {
            return;
        }
        let w = weight * test.det_j;
        for a in 0..test.n_dofs {
            let ka = curvature_col(test, a);
            for b in 0..trial.n_dofs {
                let kb = curvature_col(trial, b);
                let mut value = 0.0;
                for (i, &kai) in ka.iter().enumerate() {
                    for (j, &kbj) in kb.iter().enumerate() {
                        value += kai * self.d_matrix[i][j] * kbj;
                    }
                }
                local_matrix.add(a, b, value * w);
            }
        }
    }
}

impl LaminateMembraneForm {
    pub fn new(laminate: &Laminate) -> Self {
        Self {
            a_matrix: laminate.membrane_stiffness(),
        }
    }
}

impl BilinearForm<f64> for LaminateMembraneForm {
    fn integrate_point(
        &self,
        test: &BasisTabulation,
        trial: &BasisTabulation,
        weight: f64,
        local_matrix: &mut LocalMatrix<f64>,
    ) {
        assert_eq!(test.spatial_dim, 2, "laminate membrane form is 2D");
        let w = weight * test.det_j;
        for a in 0..test.n_dofs {
            let ba = strain_col(a % 2, test.grad(a, 0), test.grad(a, 1));
            for b in 0..trial.n_dofs {
                let bb = strain_col(b % 2, trial.grad(b, 0), trial.grad(b, 1));
                let mut value = 0.0;
                for (i, &bai) in ba.iter().enumerate() {
                    for (j, &bbj) in bb.iter().enumerate() {
                        value += bai * self.a_matrix[i][j] * bbj;
                    }
                }
                local_matrix.add(a, b, value * w);
            }
        }
    }
}

#[inline]
fn strain_col(component: usize, dphi_dx: f64, dphi_dy: f64) -> [f64; 3] {
    match component {
        0 => [dphi_dx, 0.0, dphi_dy],
        1 => [0.0, dphi_dy, dphi_dx],
        _ => unreachable!(),
    }
}

#[inline]
fn curvature_col(tab: &BasisTabulation, dof: usize) -> [f64; 3] {
    [
        tab.hessian(dof, 0, 0),
        tab.hessian(dof, 1, 1),
        2.0 * tab.hessian(dof, 0, 1),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::physics::solid::composites::{Laminate, OrthotropicPly};

    #[test]
    fn reduced_stiffness_is_symmetric_and_positive() {
        let ply = OrthotropicPly {
            e1: 140.0,
            e2: 10.0,
            g12: 5.0,
            nu12: 0.25,
            thickness: 0.001,
            angle_rad: 0.0,
        };
        let q = ply.reduced_stiffness();
        assert!((q[0][1] - q[1][0]).abs() < 1.0e-12);
        assert!(q[0][0] > q[1][1]);
    }

    #[test]
    fn ninety_degree_ply_swaps_material_axes() {
        let ply0 = OrthotropicPly {
            e1: 140.0,
            e2: 10.0,
            g12: 5.0,
            nu12: 0.25,
            thickness: 1.0,
            angle_rad: 0.0,
        };
        let ply90 = OrthotropicPly {
            angle_rad: std::f64::consts::FRAC_PI_2,
            ..ply0
        };
        let q0 = ply0.transformed_stiffness();
        let q90 = ply90.transformed_stiffness();
        assert!((q0[0][0] - q90[1][1]).abs() < 1.0e-10);
        assert!((q0[1][1] - q90[0][0]).abs() < 1.0e-10);
    }

    #[test]
    fn laminate_form_uses_membrane_a_matrix() {
        let ply = OrthotropicPly {
            e1: 10.0,
            e2: 10.0,
            g12: 4.0,
            nu12: 0.25,
            thickness: 2.0,
            angle_rad: 0.0,
        };
        let form = LaminateMembraneForm::new(&Laminate { plies: vec![ply] });
        let tab = BasisTabulation {
            n_dofs: 2,
            spatial_dim: 2,
            values: vec![0.0; 4],
            derivatives: vec![1.0, 0.0, 0.0, 2.0],
            hessians: vec![],
            det_j: 1.0,
            physical_coords: [0.0; 3],
        };
        let mut local = LocalMatrix::zeros(2);
        form.integrate_point(&tab, &tab, 0.5, &mut local);
        assert!(local.get(0, 0) > 0.0);
        assert!(local.get(1, 1) > 0.0);
    }

    #[test]
    fn laminate_bending_form_uses_d_matrix_and_hessians() {
        let ply = OrthotropicPly {
            e1: 10.0,
            e2: 10.0,
            g12: 4.0,
            nu12: 0.25,
            thickness: 2.0,
            angle_rad: 0.0,
        };
        let form = LaminateBendingForm::new(&Laminate { plies: vec![ply] });
        let tab = BasisTabulation {
            n_dofs: 1,
            spatial_dim: 2,
            values: vec![0.0],
            derivatives: vec![0.0; 2],
            hessians: vec![1.0, 0.5, 0.5, 2.0],
            det_j: 1.0,
            physical_coords: [0.0; 3],
        };
        let mut local = LocalMatrix::zeros(1);
        form.integrate_point(&tab, &tab, 1.0, &mut local);
        assert!(local.get(0, 0) > 0.0);
    }
}
