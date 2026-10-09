//! Derived Wolfram/closed-form check of Sonium's 3D BEM impedance boundary.
//!
//! Wolfram's PDEModels catalog contains no literal 3D BEM sphere. This test
//! therefore adapts the plane-wave sphere used by the rigid-sphere audit and
//! independently evaluates its Robin partial-wave series in Wolfram.

use num_complex::Complex64;
use serde::Deserialize;
use sonium_bem::{
    analytical::impedance_sphere_scattering_3d,
    core::{
        AssemblyMethod, BemProblem, BemSolver, SolverMethod, bem_solver::BoundaryConditionType,
    },
};
use sonium_qa::subprocess::WolframCommandExt;
use std::{env, f64::consts::PI, path::PathBuf, process::Command};

const SOUND_SPEED_M_S: f64 = 343.0;
const DENSITY_KG_M3: f64 = 1.21;
const RADIUS_M: f64 = 1.0;
const KA: f64 = 1.0;
const NORMALIZED_ADMITTANCE: Complex64 = Complex64::new(1.0, 0.0);
const RECEIVER_RADIUS_M: f64 = 2.0;
const RECEIVER_ANGLES_RAD: [f64; 5] = [0.0, PI / 4.0, PI / 2.0, 3.0 * PI / 4.0, PI];

#[derive(Debug, Deserialize)]
struct WolframReference {
    schema_version: u32,
    provenance: String,
    radius_m: f64,
    ka: f64,
    normalized_admittance: [f64; 2],
    receiver_radius_m: f64,
    receiver_angles_rad: [f64; 5],
    total_pressure: [[f64; 2]; 5],
    soft_total_pressure: [[f64; 2]; 5],
}

#[test]
fn sonium_direct_and_iterative_reproduce_pressure_release_sphere_series() {
    let exact = exact_soft();
    let direct = solve_soft(SolverMethod::Direct);
    let iterative = solve_soft(SolverMethod::BiCgStab);
    let direct_error = relative_l2_error(&direct, &exact);
    let iterative_error = relative_l2_error(&iterative, &exact);
    let solver_parity = relative_l2_error(&iterative, &direct);
    eprintln!(
        "soft sphere: direct/exact={direct_error:.3e}, iterative/exact={iterative_error:.3e}, iterative/direct={solver_parity:.3e}"
    );
    assert!(
        direct_error < 0.05,
        "direct soft-sphere error {direct_error:.3e}"
    );
    assert!(
        iterative_error < 0.05,
        "iterative soft-sphere error {iterative_error:.3e}"
    );
    assert!(
        solver_parity < 1.0e-8,
        "soft-sphere solver mismatch {solver_parity:.3e}"
    );

    if let Some(reference) = wolfram_reference() {
        let wolfram = reference
            .soft_total_pressure
            .map(|pair| Complex64::new(pair[0], pair[1]));
        assert!(relative_l2_error(&wolfram, &exact) < 1.0e-12);
        assert!(relative_l2_error(&direct, &wolfram) < 0.05);
        assert!(relative_l2_error(&iterative, &wolfram) < 0.05);
    }
}

fn receiver(theta: f64) -> [f64; 3] {
    [
        RECEIVER_RADIUS_M * theta.sin(),
        0.0,
        RECEIVER_RADIUS_M * theta.cos(),
    ]
}

fn solve(assembly_method: AssemblyMethod) -> Vec<Complex64> {
    let frequency_hz = KA * SOUND_SPEED_M_S / (2.0 * PI * RADIUS_M);
    let problem =
        BemProblem::rigid_sphere_scattering(RADIUS_M, frequency_hz, SOUND_SPEED_M_S, DENSITY_KG_M3)
            .with_boundary_condition(BoundaryConditionType::Impedance);
    let solution = BemSolver::new()
        .with_solver_method(SolverMethod::BiCgStab)
        .with_assembly_method(assembly_method)
        .with_tolerance(1.0e-9)
        .solve(&problem)
        .expect("solve matched-impedance sphere");
    RECEIVER_ANGLES_RAD
        .iter()
        .map(|&theta| solution.evaluate_pressure(&receiver(theta)))
        .collect()
}

fn solve_soft(solver_method: SolverMethod) -> Vec<Complex64> {
    let frequency_hz = KA * SOUND_SPEED_M_S / (2.0 * PI * RADIUS_M);
    let problem =
        BemProblem::rigid_sphere_scattering(RADIUS_M, frequency_hz, SOUND_SPEED_M_S, DENSITY_KG_M3)
            .with_boundary_condition(BoundaryConditionType::Soft);
    let solution = BemSolver::new()
        .with_solver_method(solver_method)
        .with_assembly_method(AssemblyMethod::Tbem)
        .with_tolerance(1.0e-10)
        .solve(&problem)
        .expect("solve pressure-release sphere");
    RECEIVER_ANGLES_RAD
        .iter()
        .map(|&theta| solution.evaluate_pressure(&receiver(theta)))
        .collect()
}

fn exact() -> Vec<Complex64> {
    impedance_sphere_scattering_3d(
        KA / RADIUS_M,
        RADIUS_M,
        30,
        NORMALIZED_ADMITTANCE,
        vec![RECEIVER_RADIUS_M],
        RECEIVER_ANGLES_RAD.to_vec(),
    )
    .pressure
}

fn exact_soft() -> Vec<Complex64> {
    // The sign of beta is immaterial in the infinite-admittance limit.
    impedance_sphere_scattering_3d(
        KA / RADIUS_M,
        RADIUS_M,
        30,
        Complex64::new(1.0e12, 0.0),
        vec![RECEIVER_RADIUS_M],
        RECEIVER_ANGLES_RAD.to_vec(),
    )
    .pressure
}

fn relative_l2_error(actual: &[Complex64], expected: &[Complex64]) -> f64 {
    assert_eq!(actual.len(), expected.len());
    let numerator = actual
        .iter()
        .zip(expected)
        .map(|(actual, expected)| (*actual - *expected).norm_sqr())
        .sum::<f64>();
    let denominator = expected.iter().map(Complex64::norm_sqr).sum::<f64>();
    (numerator / denominator).sqrt()
}

fn wolfram_reference() -> Option<WolframReference> {
    let executable = env::var_os("WOLFRAMSCRIPT")?;
    let script = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("wolfram/bem_impedance_sphere.wls");
    let output = match Command::new(&executable)
        .arg("-file")
        .arg(script)
        .wolfram_output()
    {
        Ok(output) => output,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return None,
        Err(error) => panic!("failed to launch {executable:?}: {error}"),
    };
    assert!(
        output.status.success(),
        "Wolfram impedance-sphere reference failed:\n{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    let json = stdout
        .lines()
        .rev()
        .find(|line| line.trim_start().starts_with('{'))
        .unwrap_or(&stdout);
    Some(serde_json::from_str(json).unwrap_or_else(|error| {
        panic!("Wolfram impedance-sphere output was not JSON: {error}\nstdout:\n{stdout}")
    }))
}

#[test]
fn sonium_dense_and_hmatrix_reproduce_impedance_sphere_series() {
    let exact = exact();
    let dense = solve(AssemblyMethod::Tbem);
    let hmatrix = solve(AssemblyMethod::Hmatrix);
    let dense_error = relative_l2_error(&dense, &exact);
    let hmatrix_error = relative_l2_error(&hmatrix, &exact);
    let compressed_parity = relative_l2_error(&hmatrix, &dense);
    eprintln!(
        "impedance sphere: dense/exact={dense_error:.3e}, H-matrix/exact={hmatrix_error:.3e}, H-matrix/dense={compressed_parity:.3e}"
    );
    assert!(
        dense_error < 0.05,
        "dense impedance-sphere error {dense_error:.3e}"
    );
    assert!(
        hmatrix_error < 0.05,
        "H-matrix impedance-sphere error {hmatrix_error:.3e}"
    );
    assert!(
        compressed_parity < 0.01,
        "dense/H-matrix mismatch {compressed_parity:.3e}"
    );
    if let Some(reference) = wolfram_reference() {
        assert_eq!(reference.schema_version, 1);
        assert!(
            reference
                .provenance
                .contains("not a literal PDEModels geometry")
        );
        assert_eq!(reference.radius_m, RADIUS_M);
        assert_eq!(reference.ka, KA);
        assert_eq!(reference.normalized_admittance, [1.0, 0.0]);
        assert_eq!(reference.receiver_radius_m, RECEIVER_RADIUS_M);
        assert_eq!(reference.receiver_angles_rad, RECEIVER_ANGLES_RAD);
        let wolfram = reference
            .total_pressure
            .map(|pair| Complex64::new(pair[0], pair[1]));
        assert!(relative_l2_error(&wolfram, &exact) < 1.0e-12);
        assert!(relative_l2_error(&dense, &wolfram) < 0.05);
        assert!(relative_l2_error(&hmatrix, &wolfram) < 0.05);
    }
}
