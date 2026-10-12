//! Fracture assembler helpers.

use crate::assembly::prebaked::PrebakedAssembler;
use crate::basis::PolynomialDegree;
use crate::mesh::Mesh;
use crate::quadrature;
use crate::spaces::H1Space;
use sonium_solvers::CsrMatrix;

use super::{
    DamageHistoryRhsForm, DamageHistoryTangentForm, PhaseFieldDamageForm,
    PhaseFieldFractureMaterial,
};

/// Single component:
/// 0 = phase-field damage operator.
pub type FractureAssembler = PrebakedAssembler<1>;

/// Build a phase-field damage assembler.
pub fn build_fracture_assembler(
    mesh: &Mesh,
    degree: PolynomialDegree,
    material: PhaseFieldFractureMaterial,
) -> FractureAssembler {
    let space = H1Space::new(degree);
    // The regularization contains both gradient and value products.
    let quad_damage = quadrature::required_order_for_stiffness(degree.degree())
        .max(quadrature::required_order_for_mass(degree.degree()));
    let damage = crate::assembly::assemble_bilinear(
        &PhaseFieldDamageForm { material },
        &space,
        &space,
        mesh,
        quad_damage,
    );
    PrebakedAssembler::from_triplets([damage], &[])
}

/// Assemble the consistent staggered AT2 damage operator.
///
/// Base regularization plus the per-element `2 * H` mass term from
/// `history` (element order): `(Gc/l + 2H) d - Gc l Δd = 2H`.
/// Without the history term the update linearizes to the unbounded
/// `d = 2Hl/Gc`; with it the bounded law holds and the operator must
/// be rebuilt (and refactorized) as history evolves.
pub fn assemble_damage_operator_with_history(
    mesh: &Mesh,
    degree: PolynomialDegree,
    material: PhaseFieldFractureMaterial,
    history: &[f64],
) -> sonium_solvers::TripletMatrix<f64> {
    debug_assert_eq!(
        history.len(),
        mesh.num_elements(),
        "history must cover every element"
    );
    let space = H1Space::new(degree);
    let quad_damage = quadrature::required_order_for_stiffness(degree.degree())
        .max(quadrature::required_order_for_mass(degree.degree()));
    let mut triplets = crate::assembly::assemble_bilinear(
        &PhaseFieldDamageForm { material },
        &space,
        &space,
        mesh,
        quad_damage,
    );
    let quad_mass = quadrature::required_order_for_mass(degree.degree());
    let tangent_forms: Vec<DamageHistoryTangentForm> = history
        .iter()
        .map(|&energy| DamageHistoryTangentForm {
            coefficient: 2.0 * energy,
        })
        .collect();
    let tangent = crate::assembly::assemble_bilinear_per_element(
        &tangent_forms,
        &space,
        &space,
        mesh,
        quad_mass,
    );
    triplets.rows.extend(tangent.rows.iter().copied());
    triplets.cols.extend(tangent.cols.iter().copied());
    triplets.values.extend(tangent.values.iter().copied());
    triplets
}

/// Assemble history-field RHS for the damage equation.
pub fn assemble_damage_history_rhs(
    mesh: &Mesh,
    degree: PolynomialDegree,
    history_energy_density: f64,
) -> Vec<f64> {
    let space = H1Space::new(degree);
    let quad_mass = quadrature::required_order_for_mass(degree.degree());
    crate::assembly::assemble_linear(
        &DamageHistoryRhsForm {
            history_energy_density,
        },
        &space,
        mesh,
        quad_mass,
    )
}

/// Extension methods for simple fracture operator assembly.
pub trait FractureAssemblerExt {
    /// Assemble scaled phase-field damage operator.
    fn assemble_damage_operator(&self, scale: f64) -> CsrMatrix<f64>;
}

impl FractureAssemblerExt for FractureAssembler {
    fn assemble_damage_operator(&self, scale: f64) -> CsrMatrix<f64> {
        self.assemble_real(&[scale])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh::{unit_cube_tetrahedra, unit_square_triangles};
    use crate::spaces::FunctionSpace;

    #[test]
    fn damage_operators_match_independent_cube_polynomial_integrals() {
        let mesh = unit_cube_tetrahedra(1);
        let material = PhaseFieldFractureMaterial {
            critical_energy_release_rate: 2.0,
            length_scale: 0.25,
            residual_stiffness: 1e-6,
        };
        let history = vec![2.0; mesh.num_elements()];
        // P = x^p + y^p + z^p. Integrate P^2 and |grad(P)|^2
        // independently over [0,1]^3; no assembled matrix defines the reference.
        for (degree, p, mass, stiffness) in [
            (PolynomialDegree::P1, 1, 5.0 / 2.0, 3.0),
            (PolynomialDegree::P2, 2, 19.0 / 15.0, 4.0),
            (PolynomialDegree::P3, 3, 45.0 / 56.0, 27.0 / 5.0),
        ] {
            let values: Vec<_> = H1Space::new(degree)
                .dof_coordinates(&mesh)
                .unwrap()
                .iter()
                .map(|point| point.iter().map(|x| x.powi(p)).sum::<f64>())
                .collect();
            let base =
                build_fracture_assembler(&mesh, degree, material).assemble_damage_operator(1.0);
            let driven =
                assemble_damage_operator_with_history(&mesh, degree, material, &history).to_csr();
            for (matrix, mass_coefficient) in [(&base, 8.0), (&driven, 12.0)] {
                let actual: f64 = values
                    .iter()
                    .zip(matrix.matvec(&values))
                    .map(|(a, b)| a * b)
                    .sum();
                let expected = 0.5 * stiffness + mass_coefficient * mass;
                assert!(
                    (actual - expected).abs() < 3e-12 * expected,
                    "{degree:?}: actual={actual:e}, expected={expected:e}"
                );
            }
            let rhs = assemble_damage_history_rhs(&mesh, degree, 2.0);
            let actual: f64 = values.iter().zip(rhs).map(|(a, b)| a * b).sum();
            let expected = 12.0 / f64::from(p + 1);
            assert!((actual - expected).abs() < 3e-12 * expected);
        }
    }

    #[test]
    fn build_fracture_assembler_smoke() {
        let mesh = unit_square_triangles(2);
        let material = PhaseFieldFractureMaterial {
            critical_energy_release_rate: 2.0,
            length_scale: 0.1,
            residual_stiffness: 1.0e-6,
        };
        let assembler = build_fracture_assembler(&mesh, PolynomialDegree::P1, material);
        assert!(assembler.nnz() > 0);
        assert_eq!(assembler.num_rows, mesh.num_nodes());

        let rhs = assemble_damage_history_rhs(&mesh, PolynomialDegree::P1, 1.0);
        assert_eq!(rhs.len(), mesh.num_nodes());
    }
}
