//! Sonium reproduction of Wolfram's 1D Gaussian-pulse absorbing-boundary
//! example from `AcousticsTimeDomain`.
//!
//! Sonium has no 1D FEM element, so the plane wave is represented by a thin
//! rigid-walled 2D strip. The right boundary uses the exact normal-incidence
//! first-order ABC from the Wolfram example.

use sonium_fem::{
    assembly::{PrebakedAssembler, assemble_bilinear, assemble_boundary_mass_generic},
    basis::PolynomialDegree,
    forms::{LaplacianForm, MassForm},
    mesh::{BoundaryType, generators::rectangular_mesh_triangles},
    spaces::H1Space,
};
use sonium_solvers::SparseLuFactorization;

const LENGTH_M: f64 = 0.5;
const STRIP_HEIGHT_M: f64 = 0.01;
const SOUND_SPEED_M_S: f64 = 343.2;
const PULSE_WIDTH_M: f64 = 0.1;
const END_TIME_S: f64 = 0.002;
const TIME_STEP_S: f64 = 2.0e-6;
const ABSORBING_MARKER: i32 = 7;

#[test]
fn sonium_reproduces_wolfram_normal_incidence_absorbing_boundary() {
    let mut mesh = rectangular_mesh_triangles(0.0, LENGTH_M, 0.0, STRIP_HEIGHT_M, 200, 1);
    mesh.set_boundary_condition(BoundaryType::Robin, ABSORBING_MARKER, |points| {
        points
            .iter()
            .all(|point| (point.x - LENGTH_M).abs() < 1.0e-12)
    });

    let space = H1Space::new(PolynomialDegree::P1);
    let stiffness_triplet = assemble_bilinear(&LaplacianForm, &space, &space, &mesh, 2);
    let mass_triplet = assemble_bilinear(&MassForm, &space, &space, &mesh, 2);
    let boundary_triplet = assemble_boundary_mass_generic(&mesh, ABSORBING_MARKER);
    let assembler = PrebakedAssembler::<3>::from_triplets(
        [stiffness_triplet, mass_triplet, boundary_triplet],
        &[],
    );
    let stiffness = assembler.assemble_real(&[1.0, 0.0, 0.0]);
    let mass = assembler.assemble_real(&[0.0, 1.0 / SOUND_SPEED_M_S.powi(2), 0.0]);
    let damping = assembler.assemble_real(&[0.0, 0.0, 1.0 / SOUND_SPEED_M_S]);

    let newmark_beta = 0.25;
    let newmark_gamma = 0.5;
    let effective = assembler.assemble_real(&[
        newmark_beta * TIME_STEP_S.powi(2),
        1.0 / SOUND_SPEED_M_S.powi(2),
        newmark_gamma * TIME_STEP_S / SOUND_SPEED_M_S,
    ]);
    let effective_factor =
        SparseLuFactorization::factorize(&effective).expect("ABC Newmark factorization");
    let mass_factor = SparseLuFactorization::factorize(&mass).expect("ABC mass factorization");

    let ndof = mesh.num_nodes();
    let mut pressure = mesh
        .nodes
        .iter()
        .map(|point| 2.0 * (-(point.x / PULSE_WIDTH_M).powi(2)).exp())
        .collect::<Vec<_>>();
    let mut velocity = vec![0.0; ndof];
    let initial_rhs = stiffness
        .matvec(&pressure)
        .into_iter()
        .map(|value| -value)
        .collect::<Vec<_>>();
    let mut acceleration = mass_factor
        .solve(&initial_rhs)
        .expect("ABC initial acceleration");

    let steps = (END_TIME_S / TIME_STEP_S).round() as usize;
    for _ in 0..steps {
        let pressure_predictor = (0..ndof)
            .map(|index| {
                pressure[index]
                    + TIME_STEP_S * velocity[index]
                    + (0.5 - newmark_beta) * TIME_STEP_S.powi(2) * acceleration[index]
            })
            .collect::<Vec<_>>();
        let velocity_predictor = (0..ndof)
            .map(|index| {
                velocity[index] + (1.0 - newmark_gamma) * TIME_STEP_S * acceleration[index]
            })
            .collect::<Vec<_>>();
        let stiffness_pressure = stiffness.matvec(&pressure_predictor);
        let damping_velocity = damping.matvec(&velocity_predictor);
        let rhs = (0..ndof)
            .map(|index| -stiffness_pressure[index] - damping_velocity[index])
            .collect::<Vec<_>>();
        let acceleration_new = effective_factor.solve(&rhs).expect("ABC Newmark step");
        for index in 0..ndof {
            pressure[index] = pressure_predictor[index]
                + newmark_beta * TIME_STEP_S.powi(2) * acceleration_new[index];
            velocity[index] =
                velocity_predictor[index] + newmark_gamma * TIME_STEP_S * acceleration_new[index];
        }
        acceleration = acceleration_new;
    }

    let mut maximum_error = 0.0_f64;
    let mut interior_reflection_error = 0.0_f64;
    for (point, &numerical) in mesh.nodes.iter().zip(&pressure) {
        if point.y.abs() >= 1.0e-12 {
            continue;
        }
        let analytical =
            (-(SOUND_SPEED_M_S * END_TIME_S - point.x).powi(2) / PULSE_WIDTH_M.powi(2)).exp()
                + (-(SOUND_SPEED_M_S * END_TIME_S + point.x).powi(2) / PULSE_WIDTH_M.powi(2)).exp();
        let error = (numerical - analytical).abs();
        maximum_error = maximum_error.max(error);
        if point.x < 0.3 {
            interior_reflection_error = interior_reflection_error.max(error);
        }
    }
    assert!(
        maximum_error < 0.02,
        "maximum Wolfram analytical-pulse error {maximum_error:.3e}"
    );
    assert!(
        interior_reflection_error < 0.006,
        "interior ABC reflection/error {interior_reflection_error:.3e}"
    );
}
