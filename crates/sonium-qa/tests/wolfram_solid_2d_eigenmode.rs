//! Literal Sonium reproduction of Wolfram verification case
//! `SolidMechanics-FEM-Eigenmode-2D-0001` (whose published test IDs contain
//! the upstream `Eigennode` spelling).
//!
//! The quadratic triangle solve checks the published solid-element frequency;
//! the independent Sonium beam model provides a second analytical-scale check.

use sonium_fem::{
    basis::PolynomialDegree,
    mesh::generators::rectangular_mesh_triangles,
    physics::{
        solid::eigenmodes::{StructuralModalConfig, build_structural_modal_problem},
        solid::elasticity::IsotropicMaterial,
    },
};
use sonium_sim_structural_mechanics::{
    StructuralMaterial,
    beam::{Beam2DGeometry, Beam2DStudy, run_beam2d_modal},
};
use sonium_solvers::ShiftInvertConfig;

const LENGTH_M: f64 = 10.0;
const HEIGHT_M: f64 = 0.7;
const YOUNG_MODULUS_PA: f64 = 210.0e9;
const POISSON_RATIO: f64 = 0.3;
const DENSITY_KG_M3: f64 = 7850.0;
const REFERENCE_FREQUENCY_HZ: f64 = 5.85289;

fn beam_frequency() -> f64 {
    run_beam2d_modal(&Beam2DStudy {
        name: "wolfram-2d-eigenmode-0001".into(),
        geometry: Beam2DGeometry {
            length_m: LENGTH_M,
            width_m: 1.0,
            height_m: HEIGHT_M,
            elements: 40,
        },
        material: StructuralMaterial {
            youngs_modulus_pa: YOUNG_MODULUS_PA,
            poisson_ratio: POISSON_RATIO,
            density_kg_m3: DENSITY_KG_M3,
            damping_ratio: 0.0,
        },
        vertical_tip_force_n: -1.0,
        timoshenko: false,
        modal_modes: 1,
    })
    .expect("matching Euler-Bernoulli beam modal solve")
    .frequencies_hz[0]
}

fn solid_frequency(degree: PolynomialDegree, nx: usize, ny: usize) -> f64 {
    let mut mesh = rectangular_mesh_triangles(0.0, LENGTH_M, 0.0, HEIGHT_M, nx, ny);
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
            components: 2,
            density: DENSITY_KG_M3,
            plane: Some(sonium_fem::physics::solid::elasticity::PlaneAssumption::PlaneStress),
            fixed_boundary_markers: vec![1],
        },
    );
    let result = problem
        .solve_near(&ShiftInvertConfig {
            target: 0.0,
            num_eigenvalues: 1,
            krylov_dimension: 20,
            tolerance: 1.0e-8,
            breakdown_tolerance: 1.0e-14,
        })
        .expect("2D structural modal solve");
    assert!(result.converged, "2D shift-invert eigenpairs must converge");
    assert!(
        result
            .residual_norms
            .iter()
            .all(|&residual| residual < 1.0e-8)
    );
    let mut eigenvalues = result.eigenvalues;
    eigenvalues.sort_by(f64::total_cmp);
    problem.eigenvalues_to_frequencies_hz(&eigenvalues)[0]
}

#[test]
fn sonium_matches_wolfram_first_2d_cantilever_mode() {
    let linear_frequency = solid_frequency(PolynomialDegree::P1, 30, 3);
    let frequency = solid_frequency(PolynomialDegree::P2, 30, 3);
    let linear_error = ((linear_frequency - REFERENCE_FREQUENCY_HZ) / REFERENCE_FREQUENCY_HZ).abs();
    let relative_error = ((frequency - REFERENCE_FREQUENCY_HZ) / REFERENCE_FREQUENCY_HZ).abs();
    assert!(
        relative_error < 0.01,
        "P2 Sonium {frequency:.9} Hz versus Wolfram {REFERENCE_FREQUENCY_HZ:.9} Hz, relative error {relative_error:.3e}"
    );
    assert!(
        relative_error < linear_error,
        "P2 order refinement must improve the published-reference error: P1 {linear_error:.3e}, P2 {relative_error:.3e}"
    );

    let beam = beam_frequency();
    let beam_relative_error = ((beam - REFERENCE_FREQUENCY_HZ) / REFERENCE_FREQUENCY_HZ).abs();
    assert!(
        beam_relative_error < 0.005,
        "matching Sonium beam frequency {beam:.9} Hz versus Wolfram {REFERENCE_FREQUENCY_HZ:.9} Hz, relative error {beam_relative_error:.3e}"
    );
}
