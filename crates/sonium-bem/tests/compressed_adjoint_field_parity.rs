//! Compressed-backend Hermitian-adjoint field parity contract.

use num_complex::Complex64;
use sonium_bem::core::{AssemblyMethod, BemProblem, BemSolver, SolverMethod};

const FREQUENCIES: [f64; 2] = [100.0, 120.0];
const FIELD_POINTS: [[f64; 3]; 3] = [[0.0, 0.0, 0.24], [0.08, 0.03, 0.21], [0.14, -0.02, 0.17]];

#[derive(Debug, Clone, Copy)]
pub struct CompressedAdjointFieldMetrics {
    pub backend_count: usize,
    pub matrix_dimension: usize,
    pub maximum_residual: f64,
    pub maximum_exact_reference_residual: f64,
    pub maximum_field_error: f64,
    pub invalid_point_rejections: usize,
}

fn problem() -> BemProblem {
    BemProblem::rigid_sphere_scattering_custom(0.1, FREQUENCIES[0], 343.0, 1.21, 4, 8)
}

fn rhs(dimension: usize) -> Vec<Complex64> {
    (0..dimension)
        .map(|row| Complex64::new(0.25 + row as f64 / 37.0, -((row % 5) as f64) / 41.0))
        .collect()
}

fn relative_error(actual: &[Complex64], reference: &[Complex64]) -> f64 {
    actual
        .iter()
        .zip(reference)
        .map(|(value, expected)| (*value - *expected).norm_sqr())
        .sum::<f64>()
        .sqrt()
        / reference
            .iter()
            .map(Complex64::norm_sqr)
            .sum::<f64>()
            .sqrt()
            .max(1.0e-30)
}

pub fn run_compressed_adjoint_field_parity() -> CompressedAdjointFieldMetrics {
    let problem = problem();
    let rhs_batches = FREQUENCIES
        .iter()
        .map(|_| vec![rhs(48)])
        .collect::<Vec<_>>();
    let dense = BemSolver::new()
        .with_assembly_method(AssemblyMethod::Tbem)
        .with_solver_method(SolverMethod::Direct)
        .solve_adjoint_frequency_sweep_many(&problem, &FREQUENCIES, &rhs_batches)
        .expect("dense adjoint reference");
    let reference = dense
        .iter()
        .map(|batch| {
            batch
                .first()
                .expect("dense adjoint solution")
                .evaluate_scattered_pressure_field_checked(&FIELD_POINTS)
                .expect("finite dense adjoint field")
        })
        .collect::<Vec<_>>();

    let backends = [
        AssemblyMethod::Hmatrix,
        AssemblyMethod::Slfmm,
        AssemblyMethod::Mlfmm,
        AssemblyMethod::DirectionalMlfmm,
    ];
    let mut maximum_residual: f64 = 0.0;
    let mut maximum_exact_reference_residual: f64 = 0.0;
    let mut maximum_field_error: f64 = 0.0;
    let mut invalid_point_rejections = 0;
    for assembly_method in backends {
        let solutions = BemSolver::new()
            .with_assembly_method(assembly_method)
            .with_solver_method(SolverMethod::Gmres)
            .with_max_iterations(500)
            .with_tolerance(1.0e-7)
            .with_compressed_exact_residual_certification(64, 1.0e-4)
            .with_compressed_retries(0)
            .solve_adjoint_frequency_sweep_many(&problem, &FREQUENCIES, &rhs_batches)
            .unwrap_or_else(|error| panic!("{assembly_method:?} adjoint solve: {error}"));
        for (frequency_index, batch) in solutions.iter().enumerate() {
            let solution = batch.first().expect("compressed adjoint solution");
            maximum_residual = maximum_residual.max(solution.relative_residual());
            let exact_reference_residual = solution
                .exact_reference_residual()
                .expect("compressed exact-reference residual certificate");
            assert!(
                exact_reference_residual.is_finite() && exact_reference_residual <= 1.0e-4,
                "{assembly_method:?} exact-reference residual {exact_reference_residual:e}"
            );
            maximum_exact_reference_residual =
                maximum_exact_reference_residual.max(exact_reference_residual);
            let field = solution
                .evaluate_scattered_pressure_field_checked(&FIELD_POINTS)
                .unwrap_or_else(|error| panic!("{assembly_method:?} adjoint field: {error}"));
            assert!(
                field
                    .iter()
                    .all(|value| value.re.is_finite() && value.im.is_finite())
            );
            maximum_field_error =
                maximum_field_error.max(relative_error(&field, &reference[frequency_index]));
            if solution
                .evaluate_scattered_pressure_field_checked(&[[f64::NAN, 0.0, 0.24]])
                .is_err()
            {
                invalid_point_rejections += 1;
            }
        }
    }

    assert!(maximum_residual <= 1.0e-6);
    assert!(maximum_field_error <= 1.0e-6);
    assert_eq!(invalid_point_rejections, backends.len() * FREQUENCIES.len());
    CompressedAdjointFieldMetrics {
        backend_count: backends.len(),
        matrix_dimension: dense
            .first()
            .and_then(|batch| batch.first())
            .expect("dense adjoint solution")
            .num_dofs(),
        maximum_residual,
        maximum_exact_reference_residual,
        maximum_field_error,
        invalid_point_rejections,
    }
}

#[test]
fn compressed_adjoint_fields_match_dense_reference() {
    let metrics = run_compressed_adjoint_field_parity();
    assert_eq!(metrics.backend_count, 4);
    assert_eq!(metrics.matrix_dimension, 48);
    assert!(metrics.maximum_residual <= 1.0e-6);
    assert!(metrics.maximum_exact_reference_residual <= 1.0e-4);
    assert!(metrics.maximum_field_error <= 1.0e-6);
    assert_eq!(metrics.invalid_point_rejections, 8);
}
