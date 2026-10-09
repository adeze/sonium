//! Composite laminate assembler helpers.

use std::collections::HashMap;

use crate::assembly::TripletMatrix;
use crate::assembly::prebaked::PrebakedAssembler;
use crate::basis::PolynomialDegree;
use crate::mesh::Mesh;
use crate::quadrature;
use crate::spaces::{H1Space, VectorH1Space};
use sonium_solvers::CsrMatrix;

use super::{Laminate, LaminateBendingForm, LaminateMembraneForm};

/// Single component:
/// 0 = laminate membrane stiffness.
pub type CompositesAssembler = PrebakedAssembler<1>;

pub type CompositeBendingAssembler = PrebakedAssembler<1>;

/// Build laminate membrane components without boundary contributions.
pub fn build_composites_assembler(
    mesh: &Mesh,
    degree: PolynomialDegree,
    laminate: &Laminate,
) -> CompositesAssembler {
    let space = VectorH1Space::new(degree, 2);
    let quad_stiffness = quadrature::required_order_for_stiffness(degree.degree());
    let membrane = crate::assembly::assemble_bilinear(
        &LaminateMembraneForm::new(laminate),
        &space,
        &space,
        mesh,
        quad_stiffness,
    );

    PrebakedAssembler::from_triplets([membrane], &[])
}

pub fn build_composite_bending_assembler(
    mesh: &Mesh,
    degree: PolynomialDegree,
    laminate: &Laminate,
) -> CompositeBendingAssembler {
    let space = H1Space::new(degree);
    let quad_order = quadrature::required_order_for_stiffness(degree.degree() + 1);
    let bending = crate::assembly::assemble_bilinear(
        &LaminateBendingForm::new(laminate),
        &space,
        &space,
        mesh,
        quad_order,
    );
    PrebakedAssembler::from_triplets([bending], &[])
}

/// Build laminate membrane components with optional boundary mass terms.
pub fn build_composites_assembler_with_boundaries(
    mesh: &Mesh,
    degree: PolynomialDegree,
    boundary_tags: &[i32],
    laminate: &Laminate,
) -> CompositesAssembler {
    let space = VectorH1Space::new(degree, 2);
    let quad_stiffness = quadrature::required_order_for_stiffness(degree.degree());
    let membrane = crate::assembly::assemble_bilinear(
        &LaminateMembraneForm::new(laminate),
        &space,
        &space,
        mesh,
        quad_stiffness,
    );

    let mut boundary_triplets: Vec<(usize, TripletMatrix<f64>)> = Vec::new();
    for &tag in boundary_tags {
        let scalar_mass = crate::assembly::assemble_boundary_mass_generic(mesh, tag);
        if scalar_mass.nnz() > 0 {
            boundary_triplets.push((tag as usize, lift_scalar_boundary_mass(&scalar_mass, 2)));
        }
    }

    PrebakedAssembler::from_triplets([membrane], &boundary_triplets)
}

fn lift_scalar_boundary_mass(scalar: &TripletMatrix<f64>, components: usize) -> TripletMatrix<f64> {
    let mut vector = TripletMatrix::with_capacity(
        scalar.num_rows * components,
        scalar.num_cols * components,
        scalar.nnz() * components,
    );
    for k in 0..scalar.nnz() {
        for c in 0..components {
            vector.push(
                scalar.rows[k] * components + c,
                scalar.cols[k] * components + c,
                scalar.values[k],
            );
        }
    }
    vector
}

/// Extension methods for composite membrane matrix assembly.
pub trait CompositesAssemblerExt {
    /// Assemble scaled membrane operator.
    fn assemble_membrane_operator(
        &self,
        scale: f64,
        boundary_coeffs: &HashMap<usize, f64>,
    ) -> CsrMatrix<f64>;
}

impl CompositesAssemblerExt for CompositesAssembler {
    fn assemble_membrane_operator(
        &self,
        scale: f64,
        boundary_coeffs: &HashMap<usize, f64>,
    ) -> CsrMatrix<f64> {
        let mut system = self.assemble_real(&[scale]);
        for (&tag, &coeff) in boundary_coeffs {
            if let Some(boundary_values) = self.boundary_components.get(&tag) {
                for (a, &b) in system.values.iter_mut().zip(boundary_values.iter()) {
                    *a += coeff * b;
                }
            }
        }
        system
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh::unit_square_triangles;

    #[test]
    fn build_composites_assembler_smoke() {
        let mesh = unit_square_triangles(2);
        let laminate = Laminate {
            plies: vec![
                super::super::OrthotropicPly {
                    e1: 140.0,
                    e2: 10.0,
                    g12: 5.0,
                    nu12: 0.25,
                    thickness: 0.001,
                    angle_rad: 0.0,
                },
                super::super::OrthotropicPly {
                    e1: 140.0,
                    e2: 10.0,
                    g12: 5.0,
                    nu12: 0.25,
                    thickness: 0.001,
                    angle_rad: std::f64::consts::FRAC_PI_2,
                },
            ],
        };
        let assembler = build_composites_assembler(&mesh, PolynomialDegree::P1, &laminate);
        assert!(assembler.nnz() > 0);
        assert_eq!(assembler.num_rows, 2 * mesh.num_nodes());
    }

    #[test]
    fn boundary_builder_stores_vector_boundary_components() {
        let mut mesh = unit_square_triangles(2);
        mesh.detect_boundaries();
        let laminate = Laminate {
            plies: vec![super::super::OrthotropicPly {
                e1: 10.0,
                e2: 10.0,
                g12: 4.0,
                nu12: 0.25,
                thickness: 0.1,
                angle_rad: 0.0,
            }],
        };

        let assembler = build_composites_assembler_with_boundaries(
            &mesh,
            PolynomialDegree::P1,
            &[0],
            &laminate,
        );
        assert!(assembler.boundary_components.contains_key(&0));
        assert_eq!(assembler.num_rows, 2 * mesh.num_nodes());

        let base = assembler.assemble_membrane_operator(1.0, &HashMap::new());
        let with_boundary =
            assembler.assemble_membrane_operator(1.0, &HashMap::from([(0usize, 0.5)]));
        let base_sum = base.values.iter().sum::<f64>();
        let boundary_sum = with_boundary.values.iter().sum::<f64>();
        assert!((boundary_sum - base_sum).abs() > 1.0e-10);
    }
}
