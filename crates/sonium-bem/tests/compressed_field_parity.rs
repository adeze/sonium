//! Common rigid-sphere field-evaluation parity across compressed backends.

use num_complex::Complex64;
use sonium_bem::core::{AssemblyMethod, BemProblem, BemSolver, SolverMethod};

fn relative_l2(actual: &[Complex64], reference: &[Complex64]) -> f64 {
    let numerator = actual
        .iter()
        .zip(reference)
        .map(|(value, expected)| (*value - *expected).norm_sqr())
        .sum::<f64>()
        .sqrt();
    let denominator = reference
        .iter()
        .map(|value| value.norm_sqr())
        .sum::<f64>()
        .sqrt()
        .max(1.0e-12);
    numerator / denominator
}

#[test]
fn compressed_backends_match_dense_field_evaluation() {
    let problem = BemProblem::rigid_sphere_scattering_custom(0.1, 100.0, 343.0, 1.21, 4, 8);
    let points = [
        [0.0, 0.0, 0.24],
        [0.12, 0.04, 0.22],
        [-0.16, 0.05, 0.28],
        [0.08, -0.14, 0.26],
    ];
    let dense = BemSolver::new()
        .with_assembly_method(AssemblyMethod::Tbem)
        .with_solver_method(SolverMethod::Direct)
        .solve(&problem)
        .expect("dense rigid-sphere reference solve");
    let dense_field = points
        .iter()
        .map(|point| dense.evaluate_pressure(point))
        .collect::<Vec<_>>();

    for assembly in [
        AssemblyMethod::Hmatrix,
        AssemblyMethod::Slfmm,
        AssemblyMethod::Mlfmm,
        AssemblyMethod::DirectionalMlfmm,
    ] {
        let solver = BemSolver::new()
            .with_assembly_method(assembly)
            .with_solver_method(SolverMethod::Gmres)
            .with_tolerance(1.0e-7)
            .with_compressed_retries(0);
        let solution = solver
            .solve(&problem)
            .unwrap_or_else(|error| panic!("{assembly:?} compressed solve: {error}"));
        let field = points
            .iter()
            .map(|point| solution.evaluate_pressure(point))
            .collect::<Vec<_>>();
        let checked = solution
            .evaluate_pressure_field_checked(&points)
            .unwrap_or_else(|error| panic!("{assembly:?} checked field evaluation: {error}"));
        assert_eq!(checked.len(), points.len());
        assert!(
            checked
                .iter()
                .all(|point| point.p_total.re.is_finite() && point.p_total.im.is_finite())
        );
        assert!(
            solution
                .evaluate_pressure_field_checked(&[[f64::NAN, 0.0, 0.24]])
                .is_err()
        );
        let error = relative_l2(&field, &dense_field);
        assert!(
            error < 5.0e-2,
            "{assembly:?} field parity error {error:.3e} exceeds 5%"
        );
    }
}
