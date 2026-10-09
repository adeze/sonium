//! Sonium reproduction of Wolfram's periodic time-domain acoustic tube.
//!
//! The official example unwraps a circular tube into a 0.5 m interval and
//! sends a right-travelling Gaussian pulse through a periodic boundary. This
//! test uses Sonium's public multipoint-constraint condensation on a thin 2D
//! strip because the FEM mesh/space API currently has no 1D H1 element.
//! The condensed lossless ring is the conservative equivalent of Wolfram's
//! periodic-left plus absorbing-right boundary construction for this pulse.

use num_complex::Complex64;
use sonium_fem::{
    assembly::{PrebakedAssembler, assemble_bilinear},
    basis::PolynomialDegree,
    boundary::periodic::{MpcSystem, PeriodicDofConstraint, periodic_residuals},
    forms::{LaplacianForm, MassForm},
    mesh::generators::rectangular_mesh_triangles,
    spaces::H1Space,
};
use sonium_solvers::SparseLuFactorization;
use std::f64::consts::PI;

const LENGTH_M: f64 = 0.5;
const STRIP_HEIGHT_M: f64 = 0.01;
const SOUND_SPEED_M_S: f64 = 343.2;
const PULSE_CENTER_M: f64 = 0.25;
const PULSE_WIDTH_M: f64 = 0.075;
const TIME_STEP_S: f64 = 2.0e-6;
const END_TIME_S: f64 = 0.002;

fn initial_pressure(x: f64) -> f64 {
    0.25 / (PI.sqrt() * PULSE_WIDTH_M) * (-((x - PULSE_CENTER_M) / PULSE_WIDTH_M).powi(2)).exp()
}

fn initial_pressure_gradient(x: f64) -> f64 {
    -2.0 * (x - PULSE_CENTER_M) / PULSE_WIDTH_M.powi(2) * initial_pressure(x)
}

fn quadratic_form(matrix: &sonium_solvers::CsrMatrix<f64>, vector: &[f64]) -> f64 {
    vector
        .iter()
        .zip(matrix.matvec(vector))
        .map(|(left, right)| left * right)
        .sum()
}

#[test]
fn sonium_reproduces_wolfram_periodic_travelling_pulse() {
    let mesh = rectangular_mesh_triangles(0.0, LENGTH_M, 0.0, STRIP_HEIGHT_M, 200, 1);
    let space = H1Space::new(PolynomialDegree::P1);
    let laplacian = assemble_bilinear(&LaplacianForm, &space, &space, &mesh, 2);
    let mass = assemble_bilinear(&MassForm, &space, &space, &mesh, 2);
    let assembler = PrebakedAssembler::<2>::from_triplets([laplacian, mass], &[]);

    let mass_full = assembler.assemble_real(&[0.0, 1.0 / SOUND_SPEED_M_S.powi(2)]);
    let stiffness_full = assembler.assemble_real(&[1.0, 0.0]);
    let effective_full =
        assembler.assemble_real(&[0.25 * TIME_STEP_S.powi(2), 1.0 / SOUND_SPEED_M_S.powi(2)]);

    let mut constraints = Vec::new();
    for (left, left_point) in mesh.nodes.iter().enumerate() {
        if left_point.x.abs() >= 1.0e-12 {
            continue;
        }
        let right = mesh
            .nodes
            .iter()
            .position(|point| {
                (point.x - LENGTH_M).abs() < 1.0e-12 && (point.y - left_point.y).abs() < 1.0e-12
            })
            .expect("each left strip node must have a periodic right partner");
        constraints.push(PeriodicDofConstraint {
            master: right,
            slave: left,
            phase: Complex64::new(1.0, 0.0),
        });
    }
    assert_eq!(
        constraints.len(),
        2,
        "thin strip has two periodic node pairs"
    );

    let mpc = MpcSystem::from_periodic(mesh.num_nodes(), &constraints)
        .expect("periodic acoustic MPC must be valid");
    let mass_matrix = mpc
        .condense_real_matrix(&mass_full)
        .expect("condense periodic mass matrix");
    let stiffness_matrix = mpc
        .condense_real_matrix(&stiffness_full)
        .expect("condense periodic stiffness matrix");
    let effective_matrix = mpc
        .condense_real_matrix(&effective_full)
        .expect("condense periodic Newmark matrix");
    let mass_factorization =
        SparseLuFactorization::factorize(&mass_matrix).expect("periodic mass factorization");
    let effective_factorization = SparseLuFactorization::factorize(&effective_matrix)
        .expect("periodic Newmark factorization");

    let representatives = mpc.reduced_full_dofs();
    let mut pressure = representatives
        .iter()
        .map(|&node| initial_pressure(mesh.nodes[node].x))
        .collect::<Vec<_>>();
    let mut velocity = representatives
        .iter()
        .map(|&node| -SOUND_SPEED_M_S * initial_pressure_gradient(mesh.nodes[node].x))
        .collect::<Vec<_>>();
    let initial_rhs = stiffness_matrix
        .matvec(&pressure)
        .into_iter()
        .map(|value| -value)
        .collect::<Vec<_>>();
    let mut acceleration = mass_factorization
        .solve(&initial_rhs)
        .expect("periodic initial acceleration");

    let initial_peak = pressure.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let initial_energy = 0.5 * quadratic_form(&mass_matrix, &velocity)
        + 0.5 * quadratic_form(&stiffness_matrix, &pressure);

    let beta = 0.25;
    let gamma = 0.5;
    let steps = (END_TIME_S / TIME_STEP_S).round() as usize;
    for _ in 0..steps {
        let pressure_predictor = pressure
            .iter()
            .zip(&velocity)
            .zip(&acceleration)
            .map(|((&p, &v), &a)| p + TIME_STEP_S * v + (0.5 - beta) * TIME_STEP_S.powi(2) * a)
            .collect::<Vec<_>>();
        let velocity_predictor = velocity
            .iter()
            .zip(&acceleration)
            .map(|(&v, &a)| v + (1.0 - gamma) * TIME_STEP_S * a)
            .collect::<Vec<_>>();
        let rhs = stiffness_matrix
            .matvec(&pressure_predictor)
            .into_iter()
            .map(|value| -value)
            .collect::<Vec<_>>();
        let acceleration_new = effective_factorization
            .solve(&rhs)
            .expect("periodic Newmark step");

        pressure = pressure_predictor
            .iter()
            .zip(&acceleration_new)
            .map(|(&predictor, &a)| predictor + beta * TIME_STEP_S.powi(2) * a)
            .collect();
        velocity = velocity_predictor
            .iter()
            .zip(&acceleration_new)
            .map(|(&predictor, &a)| predictor + gamma * TIME_STEP_S * a)
            .collect();
        acceleration = acceleration_new;
    }

    let expected = representatives
        .iter()
        .map(|&node| {
            let translated_x =
                (mesh.nodes[node].x - SOUND_SPEED_M_S * END_TIME_S).rem_euclid(LENGTH_M);
            initial_pressure(translated_x)
        })
        .collect::<Vec<_>>();
    let max_error = pressure
        .iter()
        .zip(&expected)
        .map(|(&computed, &reference)| (computed - reference).abs())
        .fold(0.0_f64, f64::max);
    assert!(
        max_error < 0.02,
        "periodic translated-pulse maximum error {max_error:.6} Pa"
    );

    let final_peak = pressure.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    assert!(
        (final_peak - initial_peak).abs() < 0.01,
        "periodic wrap must preserve pulse peak: initial {initial_peak:.6}, final {final_peak:.6}"
    );
    let peak_node = pressure
        .iter()
        .enumerate()
        .max_by(|(_, left), (_, right)| left.total_cmp(right))
        .map(|(index, _)| representatives[index])
        .expect("periodic pressure vector is nonempty");
    let expected_peak_x = (PULSE_CENTER_M + SOUND_SPEED_M_S * END_TIME_S).rem_euclid(LENGTH_M);
    assert!(
        (mesh.nodes[peak_node].x - expected_peak_x).abs() < 0.005,
        "wrapped pulse peak at {:.6} m, expected {expected_peak_x:.6} m",
        mesh.nodes[peak_node].x
    );

    let final_energy = 0.5 * quadratic_form(&mass_matrix, &velocity)
        + 0.5 * quadratic_form(&stiffness_matrix, &pressure);
    let relative_energy_drift = ((final_energy - initial_energy) / initial_energy).abs();
    assert!(
        relative_energy_drift < 2.0e-8,
        "lossless periodic Newmark energy drift {relative_energy_drift:.3e}"
    );

    let full_pressure = mpc
        .recover(
            &pressure
                .iter()
                .map(|&value| Complex64::new(value, 0.0))
                .collect::<Vec<_>>(),
        )
        .expect("recover periodic full pressure");
    let maximum_periodic_residual = periodic_residuals(&full_pressure, &constraints)
        .into_iter()
        .map(|residual| residual.norm())
        .fold(0.0_f64, f64::max);
    assert!(maximum_periodic_residual < 1.0e-12);
}
