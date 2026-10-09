use std::f64::consts::PI;

use num_complex::Complex64;
use sonium_bem::{
    AxisymmetricBemError, AxisymmetricPenetrableProfile, AxisymmetricPenetrableSphere,
    PhysicsParams,
};

const RADIUS: f64 = 0.2;
const RATIO: f64 = 0.65;
const POINTS: usize = 8;
const AZIMUTHAL_SAMPLES: usize = 32;

fn media() -> (PhysicsParams, PhysicsParams) {
    (
        PhysicsParams::new(420.0, 343.0, 1.21, false),
        PhysicsParams::new(420.0, 240.0, 2.4, true),
    )
}

fn sphere_profile(count: usize) -> (Vec<[f64; 2]>, Vec<[f64; 2]>) {
    (0..count)
        .map(|index| {
            let theta = PI * (index as f64 + 0.5) / count as f64;
            (
                [RADIUS * theta.sin(), RADIUS * theta.cos()],
                [theta.sin(), theta.cos()],
            )
        })
        .unzip()
}

#[test]
fn profile_sphere_matches_concentric_sphere_operator() {
    let (profile, normals) = sphere_profile(POINTS);
    let (exterior, interior) = media();
    let profile_problem = AxisymmetricPenetrableProfile::try_new(
        profile,
        normals,
        RADIUS * (1.0 - RATIO),
        RADIUS * (1.0 / RATIO - 1.0),
        AZIMUTHAL_SAMPLES,
        exterior.clone(),
        interior.clone(),
    )
    .expect("profile transmission problem");
    let sphere_problem = AxisymmetricPenetrableSphere::try_new(
        RADIUS,
        POINTS,
        RATIO,
        AZIMUTHAL_SAMPLES,
        exterior,
        interior,
    )
    .expect("sphere transmission problem");

    let (profile_matrix, profile_rhs) = profile_problem
        .assemble_system(Complex64::new(1.0, 0.0))
        .expect("profile assembly");
    let (sphere_matrix, sphere_rhs) = sphere_problem
        .assemble_system(Complex64::new(1.0, 0.0))
        .expect("sphere assembly");
    let matrix_error = profile_matrix
        .as_slice()
        .iter()
        .zip(sphere_matrix.as_slice())
        .map(|(actual, reference)| (*actual - *reference).norm())
        .fold(0.0, f64::max);
    let rhs_error = profile_rhs
        .iter()
        .zip(sphere_rhs.iter())
        .map(|(actual, reference)| (*actual - *reference).norm())
        .fold(0.0, f64::max);
    assert!(
        matrix_error <= 1.0e-12,
        "sphere/profile matrix error {matrix_error:e}"
    );
    assert!(
        rhs_error <= 1.0e-12,
        "sphere/profile RHS error {rhs_error:e}"
    );

    let profile_solution = profile_problem
        .solve(Complex64::new(1.0, 0.0), 1.0e-10)
        .expect("profile solve");
    let sphere_solution = sphere_problem
        .solve(Complex64::new(1.0, 0.0), 1.0e-10)
        .expect("sphere solve");
    assert!(profile_solution.relative_residual() <= 1.0e-10);
    assert_eq!(
        profile_solution.exterior_coefficients().len(),
        sphere_solution.exterior_coefficients().len()
    );
    let coefficient_error = profile_solution
        .exterior_coefficients()
        .iter()
        .chain(profile_solution.interior_coefficients())
        .zip(
            sphere_solution
                .exterior_coefficients()
                .iter()
                .chain(sphere_solution.interior_coefficients()),
        )
        .map(|(actual, reference)| (*actual - *reference).norm())
        .fold(0.0, f64::max);
    assert!(
        coefficient_error <= 1.0e-10,
        "coefficient error {coefficient_error:e}"
    );
    assert!(profile_solution.pressure_continuity_error().unwrap() <= 1.0e-10);
    assert!(profile_solution.velocity_continuity_error().unwrap() <= 1.0e-10);
}

#[test]
fn profile_forward_many_reuses_operator_and_matches_scalar_solves() {
    let (profile, normals) = sphere_profile(POINTS);
    let (exterior, interior) = media();
    let problem = AxisymmetricPenetrableProfile::try_new(
        profile,
        normals,
        RADIUS * (1.0 - RATIO),
        RADIUS * (1.0 / RATIO - 1.0),
        AZIMUTHAL_SAMPLES,
        exterior,
        interior,
    )
    .expect("profile transmission problem");
    let amplitudes = [Complex64::new(1.0, 0.0), Complex64::new(-0.35, 0.2)];
    let batch = problem
        .solve_many(&amplitudes, 1.0e-9)
        .expect("profile forward batch");
    assert_eq!(batch.len(), amplitudes.len());
    for (index, &amplitude) in amplitudes.iter().enumerate() {
        let scalar = problem
            .solve(amplitude, 1.0e-9)
            .expect("profile scalar solve");
        assert!(batch[index].relative_residual() <= 1.0e-9);
        let exterior_error = batch[index]
            .exterior_coefficients()
            .iter()
            .zip(scalar.exterior_coefficients())
            .map(|(actual, expected)| (*actual - *expected).norm())
            .fold(0.0, f64::max);
        let interior_error = batch[index]
            .interior_coefficients()
            .iter()
            .zip(scalar.interior_coefficients())
            .map(|(actual, expected)| (*actual - *expected).norm())
            .fold(0.0, f64::max);
        assert!(
            exterior_error <= 1.0e-10,
            "exterior batch error {exterior_error:e}"
        );
        assert!(
            interior_error <= 1.0e-10,
            "interior batch error {interior_error:e}"
        );
    }
    assert!(problem.solve_many(&[], 1.0e-9).unwrap().is_empty());
    assert!(matches!(
        problem.solve_many(&[Complex64::new(f64::NAN, 0.0)], 1.0e-9),
        Err(AxisymmetricBemError::NonFinite("incident amplitude"))
    ));
}

#[test]
fn deformed_profile_forward_and_adjoint_contracts_are_checked() {
    let count = 12;
    let (profile, normals): (Vec<_>, Vec<_>) = (0..count)
        .map(|index| {
            let theta = PI * (index as f64 + 0.5) / count as f64;
            let radial = RADIUS * (1.0 + 0.12 * (2.0 * theta).cos());
            let axial = RADIUS * theta.cos();
            let radial_derivative = -0.24 * RADIUS * (2.0 * theta).sin();
            let axial_derivative = -RADIUS * theta.sin();
            ([radial, axial], [-axial_derivative, radial_derivative])
        })
        .unzip();
    let (exterior, interior) = media();
    let problem = AxisymmetricPenetrableProfile::try_new(
        profile,
        normals,
        0.025,
        0.035,
        AZIMUTHAL_SAMPLES,
        exterior,
        interior,
    )
    .expect("deformed profile problem");
    let solution = problem
        .solve(Complex64::new(1.0, 0.0), 1.0e-9)
        .expect("deformed profile solve");
    assert!(solution.relative_residual() <= 1.0e-9);
    assert!(solution.pressure_continuity_error().unwrap() <= 1.0e-9);
    assert!(solution.velocity_continuity_error().unwrap() <= 1.0e-9);
    let exterior_field = solution
        .evaluate_exterior_pressure(&[[0.0, 0.8], [0.2, 0.7]])
        .expect("exterior field");
    assert!(
        exterior_field
            .iter()
            .all(|value| value.re.is_finite() && value.im.is_finite())
    );
    assert!(
        solution
            .evaluate_exterior_pressure(&[[f64::NAN, 0.8]])
            .is_err()
    );

    let rhs = (0..problem.num_dofs())
        .map(|index| Complex64::new(0.2 + index as f64 / 31.0, -0.1 + index as f64 / 47.0))
        .collect::<Vec<_>>();
    let adjoint = problem
        .solve_adjoint(&rhs, 1.0e-9)
        .expect("deformed profile adjoint");
    assert!(adjoint.relative_residual() <= 1.0e-9);
    let second_rhs = rhs
        .iter()
        .enumerate()
        .map(|(index, value)| *value + Complex64::new(0.07 * index as f64, -0.03))
        .collect::<Vec<_>>();
    let adjoint_batch = problem
        .solve_adjoint_many(&[rhs.clone(), second_rhs], 1.0e-9)
        .expect("deformed profile adjoint batch");
    assert_eq!(adjoint_batch.len(), 2);
    let batch_error = adjoint_batch[0]
        .coefficients()
        .iter()
        .zip(adjoint.coefficients())
        .map(|(actual, reference)| (*actual - *reference).norm())
        .fold(0.0, f64::max);
    assert!(batch_error <= 1.0e-10);
    assert!(
        adjoint_batch
            .iter()
            .all(|solution| solution.relative_residual() <= 1.0e-9)
    );
    assert!(
        problem
            .solve_adjoint_many(&[vec![Complex64::default()]], 1.0e-9)
            .is_err()
    );
    assert!(
        adjoint
            .evaluate_exterior_adjoint_pressure(&[[0.0, 0.8]])
            .unwrap()[0]
            .norm()
            .is_finite()
    );
    assert!(
        adjoint
            .evaluate_interior_adjoint_pressure(&[[0.0, 0.0]])
            .unwrap()[0]
            .norm()
            .is_finite()
    );
    assert!(
        adjoint
            .evaluate_exterior_adjoint_pressure(&[[f64::NAN, 0.8]])
            .is_err()
    );
    assert!(
        adjoint
            .evaluate_interior_adjoint_pressure(&[[f64::NAN, 0.0]])
            .is_err()
    );
}

#[test]
fn deformed_profile_compressed_action_matches_dense_operator() {
    let count = 12;
    let (profile, normals): (Vec<_>, Vec<_>) = (0..count)
        .map(|index| {
            let theta = PI * (index as f64 + 0.5) / count as f64;
            let radial = RADIUS * (1.0 + 0.12 * (2.0 * theta).cos());
            let axial = RADIUS * theta.cos();
            let radial_derivative = -0.24 * RADIUS * (2.0 * theta).sin();
            let axial_derivative = -RADIUS * theta.sin();
            ([radial, axial], [-axial_derivative, radial_derivative])
        })
        .unzip();
    let (exterior, interior) = media();
    let problem = AxisymmetricPenetrableProfile::try_new(
        profile,
        normals,
        0.025,
        0.035,
        AZIMUTHAL_SAMPLES,
        exterior,
        interior,
    )
    .expect("deformed profile problem");
    let (dense, _) = problem
        .assemble_system(Complex64::new(1.0, 0.0))
        .expect("dense profile assembly");
    let compressed = problem
        .compress(4, 1.0e-2, 24)
        .expect("compressed profile operator");
    assert_eq!(compressed.interface_dofs(), count);
    assert_eq!(compressed.total_dofs(), 2 * count);
    assert!(compressed.compressed_block_count() > 0);

    let trial = (0..2 * count)
        .map(|index| Complex64::new(0.13 + index as f64 / 19.0, -0.07 + index as f64 / 31.0))
        .collect::<Vec<_>>();
    let dense_action = (0..2 * count)
        .map(|row| {
            (0..2 * count)
                .map(|column| dense[[row, column]] * trial[column])
                .sum::<Complex64>()
        })
        .collect::<Vec<_>>();
    let compressed_action = sonium_solvers::traits::LinearOperator::apply(&compressed, &trial);
    let action_error = compressed_action
        .iter()
        .zip(&dense_action)
        .map(|(actual, reference)| (*actual - *reference).norm_sqr())
        .sum::<f64>()
        .sqrt();
    let action_scale = dense_action
        .iter()
        .map(|value| value.norm_sqr())
        .sum::<f64>()
        .sqrt()
        .max(1.0e-15);
    assert!(action_error / action_scale <= 2.0e-2);
    assert!(
        compressed_action
            .iter()
            .all(|value| value.re.is_finite() && value.im.is_finite())
    );

    assert!(matches!(
        problem.compress(0, 1.0e-2, 24),
        Err(sonium_bem::core::assembly::TransmissionCompressionError::InvalidBlockSize)
    ));
}

#[test]
fn profile_geometry_and_media_admission_fails_closed() {
    let (profile, normals) = sphere_profile(POINTS);
    let (exterior, interior) = media();
    assert!(matches!(
        AxisymmetricPenetrableProfile::try_new(
            profile.clone(),
            normals[..normals.len() - 1].to_vec(),
            0.02,
            0.03,
            AZIMUTHAL_SAMPLES,
            exterior.clone(),
            interior.clone(),
        ),
        Err(AxisymmetricBemError::ProfileLengthMismatch { .. })
    ));
    let mut bad_normals = normals;
    bad_normals[2] = [f64::NAN, 0.0];
    assert!(matches!(
        AxisymmetricPenetrableProfile::try_new(
            profile,
            bad_normals,
            0.02,
            0.03,
            AZIMUTHAL_SAMPLES,
            exterior,
            interior,
        ),
        Err(AxisymmetricBemError::InvalidProfileNormal { index: 2 })
    ));

    let invalid_source_profile = vec![[0.02, -0.10], [0.03, -0.03], [0.04, 0.04], [0.03, 0.10]];
    let invalid_source_normals = vec![[1.0, 0.0]; 4];
    let (exterior, interior) = media();
    assert!(matches!(
        AxisymmetricPenetrableProfile::try_new(
            invalid_source_profile,
            invalid_source_normals,
            0.05,
            0.01,
            AZIMUTHAL_SAMPLES,
            exterior,
            interior,
        ),
        Err(AxisymmetricBemError::InvalidParameter {
            name: "exterior_source_radius",
            ..
        })
    ));
}
