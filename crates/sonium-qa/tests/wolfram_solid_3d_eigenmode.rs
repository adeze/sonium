//! Literal Sonium reproduction of Wolfram verification case
//! `SolidMechanics-FEM-Eigenmode-3D-0001`.
//!
//! The official case fixes the x=0 face of a 0.5 x 0.05 x 0.02 m steel
//! cantilever and checks its first natural frequency against 67 Hz. Sonium's
//! quadratic tetrahedral solve is checked against the published frequency.

use sonium_fem::{
    basis::PolynomialDegree,
    mesh::box_mesh_tetrahedra,
    physics::{
        solid::eigenmodes::{StructuralModalConfig, build_structural_modal_problem},
        solid::elasticity::IsotropicMaterial,
    },
};
use sonium_solvers::ShiftInvertConfig;

const LENGTH_M: f64 = 0.5;
const WIDTH_M: f64 = 0.05;
const HEIGHT_M: f64 = 0.02;
const YOUNG_MODULUS_PA: f64 = 210.0e9;
const POISSON_RATIO: f64 = 0.28;
const DENSITY_KG_M3: f64 = 7800.0;
const REFERENCE_FREQUENCY_HZ: f64 = 67.0;

fn first_frequency(degree: PolynomialDegree, nx: usize, ny: usize, nz: usize) -> f64 {
    let mut mesh = box_mesh_tetrahedra(0.0, LENGTH_M, 0.0, WIDTH_M, 0.0, HEIGHT_M, nx, ny, nz);
    for boundary in &mut mesh.boundaries {
        boundary.marker = if boundary
            .nodes
            .iter()
            .all(|&node| mesh.nodes[node].x.abs() < 1.0e-12)
        {
            1
        } else {
            2
        };
    }

    let problem = build_structural_modal_problem(
        &mesh,
        IsotropicMaterial::new(YOUNG_MODULUS_PA, POISSON_RATIO),
        &StructuralModalConfig {
            degree,
            components: 3,
            density: DENSITY_KG_M3,
            plane: None,
            fixed_boundary_markers: vec![1],
        },
    );
    let result = problem
        .solve_near(&ShiftInvertConfig {
            target: 0.0,
            num_eigenvalues: 3,
            krylov_dimension: 30,
            tolerance: 5.0e-6,
            breakdown_tolerance: 1.0e-14,
        })
        .expect("3D structural modal solve");
    assert!(
        result.converged,
        "3D {degree:?} shift-invert residuals {:?}",
        result.residual_norms
    );
    assert!(
        result
            .residual_norms
            .iter()
            .all(|&residual| residual < 5.0e-6)
    );
    let mut eigenvalues = result.eigenvalues;
    eigenvalues.sort_by(f64::total_cmp);
    problem.eigenvalues_to_frequencies_hz(&eigenvalues)[0]
}

#[test]
fn sonium_matches_wolfram_first_3d_cantilever_mode() {
    let linear_frequency = first_frequency(PolynomialDegree::P1, 12, 3, 2);
    let frequency = first_frequency(PolynomialDegree::P2, 12, 3, 2);
    let linear_error = ((linear_frequency - REFERENCE_FREQUENCY_HZ) / REFERENCE_FREQUENCY_HZ).abs();
    let relative_error = ((frequency - REFERENCE_FREQUENCY_HZ) / REFERENCE_FREQUENCY_HZ).abs();
    eprintln!("frequency={frequency:.9} Hz relative_error={relative_error:.9}");
    assert!(
        relative_error < 0.02,
        "P2 Sonium {frequency:.9} Hz versus Wolfram {REFERENCE_FREQUENCY_HZ:.9} Hz, relative error {relative_error:.3e}"
    );
    assert!(
        relative_error < linear_error,
        "P2 order refinement must improve the published-reference error: P1 {linear_error:.3e}, P2 {relative_error:.3e}"
    );
}
