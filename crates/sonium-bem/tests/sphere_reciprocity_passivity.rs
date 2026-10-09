//! Bounded physical/formulation evidence for the production rigid-sphere path.
//!
//! This is deliberately narrower than the full QA matrix: it solves one
//! sphere twice with reciprocal plane-wave/receiver directions, then checks
//! the outgoing scattered power against the independent Mie series.  The
//! two solves exercise the high-level production problem/solver/field APIs;
//! the Mie comparison supplies the independent physical reference.

use num_complex::Complex64;
use sonium_bem::core::{
    AssemblyMethod, BemProblem, BemSolution, BemSolver, IncidentField, SolverMethod,
};
use sonium_wave::analytical::sphere_scattering_3d;
use std::f64::consts::PI;

const CASE_ID: &str = "bem.sphere-reciprocity-passivity.v1";
const RADIUS_M: f64 = 0.2;
const SPEED_OF_SOUND_M_PER_S: f64 = 343.0;
const DENSITY_KG_PER_M3: f64 = 1.21;
const KA: f64 = 0.5;
const MAX_RECIPROCITY_RELATIVE_ERROR: f64 = 2.0e-2;
const MAX_SCATTERED_POWER_RELATIVE_ERROR: f64 = 8.0e-2;
const MAX_FIELD_PATTERN_L2_RELATIVE_ERROR: f64 = 1.5e-1;
const MAX_FIELD_PATTERN_POINTWISE_RELATIVE_ERROR: f64 = 2.5e-1;
const MAX_HMATRIX_BACKSCATTER_RELATIVE_ERROR: f64 = 2.0e-5;
const MAX_MATRIX_BACKEND_PARITY_RELATIVE_ERROR: f64 = 1.0e-4;

fn frequency_hz() -> f64 {
    frequency_hz_for_ka(KA)
}

fn frequency_hz_for_ka(ka: f64) -> f64 {
    ka * SPEED_OF_SOUND_M_PER_S / (2.0 * PI * RADIUS_M)
}

fn solve_for_direction(direction: [f64; 3]) -> Result<BemSolution, String> {
    solve_for_direction_with_backend_and_formulation(
        direction,
        AssemblyMethod::Tbem,
        SolverMethod::BiCgStab,
        true,
    )
}

fn solve_for_direction_with_backend(
    direction: [f64; 3],
    assembly_method: AssemblyMethod,
    solver_method: SolverMethod,
) -> Result<BemSolution, String> {
    solve_for_direction_with_backend_and_formulation(
        direction,
        assembly_method,
        solver_method,
        true,
    )
}

fn solve_for_direction_with_backend_and_formulation(
    direction: [f64; 3],
    assembly_method: AssemblyMethod,
    solver_method: SolverMethod,
    use_burton_miller: bool,
) -> Result<BemSolution, String> {
    let problem = BemProblem::try_rigid_sphere_scattering(
        RADIUS_M,
        frequency_hz(),
        SPEED_OF_SOUND_M_PER_S,
        DENSITY_KG_PER_M3,
    )
    .map_err(|error| format!("sphere problem admission failed: {error:?}"))?
    .with_incident_field(IncidentField::plane_wave(direction, 1.0))
    .with_burton_miller(use_burton_miller);

    BemSolver::new()
        .with_assembly_method(assembly_method)
        .with_solver_method(solver_method)
        .with_max_iterations(500)
        .with_tolerance(1.0e-8)
        .solve(&problem)
        .map_err(|error| format!("sphere solve failed for direction {direction:?}: {error:?}"))
}

fn scattered_pressure(solution: &BemSolution, point: [f64; 3]) -> Complex64 {
    solution.evaluate_pressure(&point)
        - solution
            .incident_field
            .evaluate_pressure(&[point], &solution.physics)[0]
}

fn relative_complex_error(actual: Complex64, reference: Complex64) -> f64 {
    (actual - reference).norm() / actual.norm().max(reference.norm()).max(f64::MIN_POSITIVE)
}

fn scattered_power_against_mie(solution: &BemSolution) -> Result<(f64, f64), String> {
    let eval_radius = (12.0 * RADIUS_M).max(RADIUS_M + 1.0);
    let samples = 64usize;
    let mut theta = Vec::with_capacity(samples);
    let mut points = Vec::with_capacity(samples);
    for sample in 0..samples {
        let mu = -1.0 + 2.0 * (sample as f64 + 0.5) / samples as f64;
        let sin_theta = (1.0 - mu * mu).max(0.0).sqrt();
        theta.push(mu.acos());
        points.push([eval_radius * sin_theta, 0.0, eval_radius * mu]);
    }

    let bem_scattered: Vec<Complex64> = solution
        .evaluate_pressure_field(&points)
        .into_iter()
        .map(|field| field.p_scattered)
        .collect();
    if bem_scattered
        .iter()
        .any(|value| !value.re.is_finite() || !value.im.is_finite())
    {
        return Err("BEM scattered field contains a non-finite value".into());
    }

    let mie = sphere_scattering_3d(
        solution.physics.wave_number,
        RADIUS_M,
        40,
        vec![eval_radius],
        theta,
    );
    let incident_reference: Vec<Complex64> = points
        .iter()
        .map(|point| {
            let phase = solution.physics.wave_number * point[2];
            Complex64::new(phase.cos(), phase.sin())
        })
        .collect();
    let mie_scattered: Vec<Complex64> = mie
        .pressure
        .iter()
        .zip(incident_reference.iter())
        .map(|(total, incident)| *total - *incident)
        .collect();

    let angular_weight = 2.0 / samples as f64;
    let bem_integral = bem_scattered
        .iter()
        .map(|value| value.norm_sqr())
        .sum::<f64>()
        * angular_weight
        * 2.0
        * PI;
    let mie_integral = mie_scattered
        .iter()
        .map(|value| value.norm_sqr())
        .sum::<f64>()
        * angular_weight
        * 2.0
        * PI;
    let factor = eval_radius.powi(2) / (2.0 * DENSITY_KG_PER_M3 * SPEED_OF_SOUND_M_PER_S);
    let bem_power = factor * bem_integral;
    let mie_power = factor * mie_integral;
    if !bem_power.is_finite() || !mie_power.is_finite() || mie_power <= f64::MIN_POSITIVE {
        return Err(format!(
            "scattered power is non-finite or non-positive: BEM={bem_power:.6e}, Mie={mie_power:.6e}"
        ));
    }

    Ok(((bem_power - mie_power).abs() / mie_power, bem_power))
}

fn scattered_field_pattern_against_mie(solution: &BemSolution) -> Result<(f64, f64), String> {
    let eval_radius = (12.0 * RADIUS_M).max(RADIUS_M + 1.0);
    let theta: [f64; 7] = [0.2, 0.65, 1.1, 1.55, 2.0, 2.45, 2.9];
    let points: Vec<[f64; 3]> = theta
        .iter()
        .map(|&angle| [eval_radius * angle.sin(), 0.0, eval_radius * angle.cos()])
        .collect();
    let bem_scattered: Vec<Complex64> = solution
        .evaluate_pressure_field(&points)
        .into_iter()
        .map(|field| field.p_scattered)
        .collect();
    let mie = sphere_scattering_3d(
        solution.physics.wave_number,
        RADIUS_M,
        40,
        vec![eval_radius],
        theta.to_vec(),
    );
    let reference_scattered: Vec<Complex64> = mie
        .pressure
        .iter()
        .zip(points.iter())
        .map(|(total, point)| {
            let phase = solution.physics.wave_number * point[2];
            *total - Complex64::from_polar(1.0, phase)
        })
        .collect();
    if bem_scattered.len() != reference_scattered.len()
        || bem_scattered
            .iter()
            .chain(reference_scattered.iter())
            .any(|value| !value.re.is_finite() || !value.im.is_finite())
    {
        return Err("BEM/Mie field pattern contains a non-finite value".into());
    }

    let reference_norm = reference_scattered
        .iter()
        .map(|value| value.norm_sqr())
        .sum::<f64>()
        .sqrt();
    if reference_norm <= f64::MIN_POSITIVE {
        return Err("Mie field pattern has no measurable scattered field".into());
    }
    let l2_error = bem_scattered
        .iter()
        .zip(reference_scattered.iter())
        .map(|(actual, reference)| (*actual - *reference).norm_sqr())
        .sum::<f64>()
        .sqrt()
        / reference_norm;
    let pointwise_error = bem_scattered
        .iter()
        .zip(reference_scattered.iter())
        .map(|(actual, reference)| relative_complex_error(*actual, *reference))
        .fold(0.0, f64::max);
    Ok((l2_error, pointwise_error))
}

#[derive(Debug, Clone, Copy)]
pub struct ReleaseMetrics {
    pub reciprocity_relative_error: f64,
    pub scattered_power_w: f64,
    pub scattered_power_relative_error: f64,
    pub field_pattern_l2_relative_error: f64,
    pub field_pattern_pointwise_relative_error: f64,
}

pub fn release_metrics() -> Result<ReleaseMetrics, String> {
    let forward = solve_for_direction([0.0, 0.0, 1.0])?;
    let reverse = solve_for_direction([0.0, 0.0, -1.0])?;
    if forward.num_dofs() != reverse.num_dofs() {
        return Err(format!(
            "forward/reverse DOF mismatch: {} != {}",
            forward.num_dofs(),
            reverse.num_dofs()
        ));
    }

    let eval_radius = (12.0 * RADIUS_M).max(RADIUS_M + 1.0);
    let reciprocity_relative_error = relative_complex_error(
        scattered_pressure(&forward, [0.0, 0.0, -eval_radius]),
        scattered_pressure(&reverse, [0.0, 0.0, eval_radius]),
    );
    if !reciprocity_relative_error.is_finite()
        || reciprocity_relative_error > MAX_RECIPROCITY_RELATIVE_ERROR
    {
        return Err(format!(
            "reciprocity error {reciprocity_relative_error:.6e} exceeds {MAX_RECIPROCITY_RELATIVE_ERROR:.6e}"
        ));
    }

    let (scattered_power_relative_error, scattered_power_w) =
        scattered_power_against_mie(&forward)?;
    if scattered_power_relative_error > MAX_SCATTERED_POWER_RELATIVE_ERROR {
        return Err(format!(
            "scattered-power error {scattered_power_relative_error:.6e} exceeds {MAX_SCATTERED_POWER_RELATIVE_ERROR:.6e}"
        ));
    }

    let (field_pattern_l2_relative_error, field_pattern_pointwise_relative_error) =
        scattered_field_pattern_against_mie(&forward)?;
    if field_pattern_l2_relative_error > MAX_FIELD_PATTERN_L2_RELATIVE_ERROR
        || field_pattern_pointwise_relative_error > MAX_FIELD_PATTERN_POINTWISE_RELATIVE_ERROR
    {
        return Err(format!(
            "field-pattern errors exceed limits: L2={field_pattern_l2_relative_error:.6e} pointwise={field_pattern_pointwise_relative_error:.6e}"
        ));
    }

    Ok(ReleaseMetrics {
        reciprocity_relative_error,
        scattered_power_w,
        scattered_power_relative_error,
        field_pattern_l2_relative_error,
        field_pattern_pointwise_relative_error,
    })
}

#[cfg_attr(test, test)]
fn bem_sphere_reciprocity_passivity_v1() {
    let forward = solve_for_direction([0.0, 0.0, 1.0]).expect("forward production solve");
    let reverse = solve_for_direction([0.0, 0.0, -1.0]).expect("reverse production solve");
    assert_eq!(forward.num_dofs(), reverse.num_dofs());

    // Reciprocity compares the same source/receiver pair after exchanging
    // source and receiver: (+z incident, -z receiver) versus (-z incident,
    // +z receiver). The sphere and medium are unchanged.
    let eval_radius = (12.0 * RADIUS_M).max(RADIUS_M + 1.0);
    let forward_backscatter = scattered_pressure(&forward, [0.0, 0.0, -eval_radius]);
    let reverse_backscatter = scattered_pressure(&reverse, [0.0, 0.0, eval_radius]);
    let reciprocity_error = relative_complex_error(forward_backscatter, reverse_backscatter);
    assert!(
        reciprocity_error.is_finite() && reciprocity_error <= MAX_RECIPROCITY_RELATIVE_ERROR,
        "{CASE_ID}: reciprocal back-scattering error {reciprocity_error:.3e} exceeds {MAX_RECIPROCITY_RELATIVE_ERROR:.3e}"
    );

    // A rigid sphere is lossless: its scattered far-field power must be
    // positive and must agree with the independent Mie power reference.
    let (power_error, scattered_power_w) =
        scattered_power_against_mie(&forward).expect("finite positive scattered power");
    assert!(
        power_error <= MAX_SCATTERED_POWER_RELATIVE_ERROR,
        "{CASE_ID}: scattered-power error {power_error:.3e} exceeds {MAX_SCATTERED_POWER_RELATIVE_ERROR:.3e}"
    );

    println!(
        "QA_RESULT: {{\"case_id\":\"{CASE_ID}\",\"status\":\"passed\",\"ka\":{KA:.6},\"reciprocity_relative_error\":{reciprocity_error:.6e},\"scattered_power_w\":{scattered_power_w:.6e},\"scattered_power_relative_error\":{power_error:.6e},\"max_reciprocity_relative_error\":{MAX_RECIPROCITY_RELATIVE_ERROR:.6e},\"max_scattered_power_relative_error\":{MAX_SCATTERED_POWER_RELATIVE_ERROR:.6e}}}"
    );
}

#[cfg_attr(test, test)]
fn bem_sphere_field_pattern_matches_independent_mie_v1() {
    let solution = solve_for_direction([0.0, 0.0, 1.0]).expect("production field solve");
    let (l2_error, pointwise_error) = scattered_field_pattern_against_mie(&solution)
        .expect("finite production and independent reference field pattern");

    assert!(
        l2_error.is_finite() && l2_error <= MAX_FIELD_PATTERN_L2_RELATIVE_ERROR,
        "{CASE_ID}: normalized field-pattern L2 error {l2_error:.3e} exceeds {MAX_FIELD_PATTERN_L2_RELATIVE_ERROR:.3e}"
    );
    assert!(
        pointwise_error.is_finite()
            && pointwise_error <= MAX_FIELD_PATTERN_POINTWISE_RELATIVE_ERROR,
        "{CASE_ID}: pointwise field-pattern error {pointwise_error:.3e} exceeds {MAX_FIELD_PATTERN_POINTWISE_RELATIVE_ERROR:.3e}"
    );

    println!(
        "BEM_EVIDENCE: {{\"case_id\":\"bem.sphere-field-pattern-mie.v1\",\"status\":\"passed\",\"l2_relative_error\":{l2_error:.6e},\"max_pointwise_relative_error\":{pointwise_error:.6e},\"max_l2_relative_error\":{MAX_FIELD_PATTERN_L2_RELATIVE_ERROR:.6e},\"max_pointwise_relative_error\":{MAX_FIELD_PATTERN_POINTWISE_RELATIVE_ERROR:.6e}}}"
    );
}

#[derive(Debug, Clone, Copy)]
pub struct HMatrixReleaseMetrics {
    pub dense_backscatter_relative_error: f64,
    pub reciprocity_relative_error: f64,
    pub scattered_power_w: f64,
    pub scattered_power_relative_error: f64,
}

pub fn hmatrix_release_metrics() -> Result<HMatrixReleaseMetrics, String> {
    let dense = solve_for_direction_with_backend_and_formulation(
        [0.0, 0.0, 1.0],
        AssemblyMethod::Tbem,
        SolverMethod::Direct,
        true,
    )?;
    let hmatrix_forward = solve_for_direction_with_backend_and_formulation(
        [0.0, 0.0, 1.0],
        AssemblyMethod::Hmatrix,
        SolverMethod::Gmres,
        true,
    )?;
    let hmatrix_reverse = solve_for_direction_with_backend_and_formulation(
        [0.0, 0.0, -1.0],
        AssemblyMethod::Hmatrix,
        SolverMethod::Gmres,
        true,
    )?;
    if dense.num_dofs() != hmatrix_forward.num_dofs()
        || hmatrix_forward.num_dofs() != hmatrix_reverse.num_dofs()
    {
        return Err("dense/H-matrix forward/reverse DOF mismatch".into());
    }

    let eval_radius = (12.0 * RADIUS_M).max(RADIUS_M + 1.0);
    let dense_backscatter = scattered_pressure(&dense, [0.0, 0.0, -eval_radius]);
    let hmatrix_backscatter = scattered_pressure(&hmatrix_forward, [0.0, 0.0, -eval_radius]);
    let dense_backscatter_relative_error =
        relative_complex_error(hmatrix_backscatter, dense_backscatter);
    if !dense_backscatter_relative_error.is_finite()
        || dense_backscatter_relative_error > MAX_HMATRIX_BACKSCATTER_RELATIVE_ERROR
    {
        return Err(format!(
            "H-matrix backscatter error {dense_backscatter_relative_error:.6e} exceeds {MAX_HMATRIX_BACKSCATTER_RELATIVE_ERROR:.6e}"
        ));
    }

    let (scattered_power_relative_error, scattered_power_w) =
        scattered_power_against_mie(&hmatrix_forward)?;
    if scattered_power_relative_error > MAX_SCATTERED_POWER_RELATIVE_ERROR {
        return Err(format!(
            "H-matrix scattered-power error {scattered_power_relative_error:.6e} exceeds {MAX_SCATTERED_POWER_RELATIVE_ERROR:.6e}"
        ));
    }
    let reciprocity_relative_error = relative_complex_error(
        hmatrix_backscatter,
        scattered_pressure(&hmatrix_reverse, [0.0, 0.0, eval_radius]),
    );
    if !reciprocity_relative_error.is_finite()
        || reciprocity_relative_error > MAX_RECIPROCITY_RELATIVE_ERROR
    {
        return Err(format!(
            "H-matrix reciprocity error {reciprocity_relative_error:.6e} exceeds {MAX_RECIPROCITY_RELATIVE_ERROR:.6e}"
        ));
    }

    Ok(HMatrixReleaseMetrics {
        dense_backscatter_relative_error,
        reciprocity_relative_error,
        scattered_power_w,
        scattered_power_relative_error,
    })
}

#[cfg_attr(test, test)]
fn bem_sphere_hmatrix_backend_preserves_reciprocity_and_passivity_v1() {
    let dense = solve_for_direction_with_backend(
        [0.0, 0.0, 1.0],
        AssemblyMethod::Tbem,
        SolverMethod::Direct,
    )
    .expect("dense TBEM reference solve");
    let hmatrix_forward = solve_for_direction_with_backend(
        [0.0, 0.0, 1.0],
        AssemblyMethod::Hmatrix,
        SolverMethod::Gmres,
    )
    .expect("H-matrix forward solve");
    let hmatrix_reverse = solve_for_direction_with_backend(
        [0.0, 0.0, -1.0],
        AssemblyMethod::Hmatrix,
        SolverMethod::Gmres,
    )
    .expect("H-matrix reverse solve");

    assert_eq!(dense.num_dofs(), hmatrix_forward.num_dofs());
    assert_eq!(hmatrix_forward.num_dofs(), hmatrix_reverse.num_dofs());

    // The compressed backend must preserve the physical observable, not just
    // its internal matrix-vector product: compare a far-field backscatter
    // sample with the dense TBEM reference.
    let eval_radius = (12.0 * RADIUS_M).max(RADIUS_M + 1.0);
    let dense_backscatter = scattered_pressure(&dense, [0.0, 0.0, -eval_radius]);
    let hmatrix_backscatter = scattered_pressure(&hmatrix_forward, [0.0, 0.0, -eval_radius]);
    let backend_error = relative_complex_error(hmatrix_backscatter, dense_backscatter);
    assert!(
        backend_error.is_finite() && backend_error <= MAX_HMATRIX_BACKSCATTER_RELATIVE_ERROR,
        "H-matrix backscatter must match dense TBEM: rel={backend_error:.3e}, max={MAX_HMATRIX_BACKSCATTER_RELATIVE_ERROR:.3e}"
    );

    // The independent Mie comparison carries the passivity/lossless-sphere
    // evidence through the alternative production backend as well.
    let (power_error, scattered_power_w) = scattered_power_against_mie(&hmatrix_forward)
        .expect("H-matrix scattered power must be finite and positive");
    assert!(
        power_error <= MAX_SCATTERED_POWER_RELATIVE_ERROR,
        "H-matrix scattered-power error {power_error:.3e} exceeds {MAX_SCATTERED_POWER_RELATIVE_ERROR:.3e}"
    );

    // Exchanging the plane-wave and receiver directions must remain
    // reciprocal after compression.
    let forward_backscatter = scattered_pressure(&hmatrix_forward, [0.0, 0.0, -eval_radius]);
    let reverse_backscatter = scattered_pressure(&hmatrix_reverse, [0.0, 0.0, eval_radius]);
    let reciprocity_error = relative_complex_error(forward_backscatter, reverse_backscatter);
    assert!(
        reciprocity_error.is_finite() && reciprocity_error <= MAX_RECIPROCITY_RELATIVE_ERROR,
        "H-matrix reciprocal back-scattering error {reciprocity_error:.3e} exceeds {MAX_RECIPROCITY_RELATIVE_ERROR:.3e}"
    );

    println!(
        "BEM_EVIDENCE: {{\"case_id\":\"bem.sphere-hmatrix-backend-parity.v1\",\"status\":\"passed\",\"backend\":\"hmatrix\",\"dense_backscatter_relative_error\":{backend_error:.6e},\"reciprocity_relative_error\":{reciprocity_error:.6e},\"scattered_power_w\":{scattered_power_w:.6e},\"scattered_power_relative_error\":{power_error:.6e}}}"
    );
}

#[cfg_attr(test, test)]
pub fn bem_sphere_backend_formulation_matrix_v1() {
    let rows = [
        (
            "tbem-regular",
            AssemblyMethod::Tbem,
            SolverMethod::Direct,
            false,
        ),
        (
            "tbem-burton-miller",
            AssemblyMethod::Tbem,
            SolverMethod::Direct,
            true,
        ),
        (
            "hmatrix-regular",
            AssemblyMethod::Hmatrix,
            SolverMethod::Gmres,
            false,
        ),
        (
            "hmatrix-burton-miller",
            AssemblyMethod::Hmatrix,
            SolverMethod::Gmres,
            true,
        ),
        (
            "slfmm-regular",
            AssemblyMethod::Slfmm,
            SolverMethod::Gmres,
            false,
        ),
        (
            "slfmm-burton-miller",
            AssemblyMethod::Slfmm,
            SolverMethod::Gmres,
            true,
        ),
        (
            "mlfmm-regular",
            AssemblyMethod::Mlfmm,
            SolverMethod::Gmres,
            false,
        ),
        (
            "mlfmm-burton-miller",
            AssemblyMethod::Mlfmm,
            SolverMethod::Gmres,
            true,
        ),
    ];

    let dense_regular = solve_for_direction_with_backend_and_formulation(
        [0.0, 0.0, 1.0],
        AssemblyMethod::Tbem,
        SolverMethod::Direct,
        false,
    )
    .expect("dense regular reference solve");
    let dense_burton_miller = solve_for_direction_with_backend_and_formulation(
        [0.0, 0.0, 1.0],
        AssemblyMethod::Tbem,
        SolverMethod::Direct,
        true,
    )
    .expect("dense Burton-Miller reference solve");
    let eval_radius = (12.0 * RADIUS_M).max(RADIUS_M + 1.0);
    for (name, assembly_method, solver_method, use_burton_miller) in rows {
        let forward = solve_for_direction_with_backend_and_formulation(
            [0.0, 0.0, 1.0],
            assembly_method,
            solver_method,
            use_burton_miller,
        )
        .unwrap_or_else(|error| panic!("{name} forward solve failed: {error}"));
        let reverse = solve_for_direction_with_backend_and_formulation(
            [0.0, 0.0, -1.0],
            assembly_method,
            solver_method,
            use_burton_miller,
        )
        .unwrap_or_else(|error| panic!("{name} reverse solve failed: {error}"));
        assert_eq!(
            forward.num_dofs(),
            reverse.num_dofs(),
            "{name} changed the active sphere discretization"
        );
        let (power_error, power_w) = scattered_power_against_mie(&forward)
            .unwrap_or_else(|error| panic!("{name} physical check failed: {error}"));
        let reciprocity_error = relative_complex_error(
            scattered_pressure(&forward, [0.0, 0.0, -eval_radius]),
            scattered_pressure(&reverse, [0.0, 0.0, eval_radius]),
        );
        let reference = if use_burton_miller {
            &dense_burton_miller
        } else {
            &dense_regular
        };
        let backend_error = relative_complex_error(
            scattered_pressure(&forward, [0.0, 0.0, -eval_radius]),
            scattered_pressure(reference, [0.0, 0.0, -eval_radius]),
        );
        let formulation = if use_burton_miller {
            "burton_miller"
        } else {
            "regular"
        };
        assert!(
            backend_error.is_finite() && backend_error <= MAX_MATRIX_BACKEND_PARITY_RELATIVE_ERROR,
            "{name} {formulation} backend parity error {backend_error:.3e} exceeds {MAX_MATRIX_BACKEND_PARITY_RELATIVE_ERROR:.3e}"
        );
        assert!(
            reciprocity_error.is_finite() && reciprocity_error <= MAX_RECIPROCITY_RELATIVE_ERROR,
            "{name} {formulation} reciprocity error {reciprocity_error:.3e} exceeds {MAX_RECIPROCITY_RELATIVE_ERROR:.3e}"
        );
        assert!(
            power_error.is_finite() && power_error <= MAX_SCATTERED_POWER_RELATIVE_ERROR,
            "{name} {formulation} Mie scattered-power error {power_error:.3e} exceeds {MAX_SCATTERED_POWER_RELATIVE_ERROR:.3e}"
        );
        println!(
            "BEM_EVIDENCE: {{\"case_id\":\"bem.sphere-backend-formulation-matrix.v1\",\"status\":\"passed\",\"backend_formulation\":\"{name}\",\"formulation\":\"{formulation}\",\"backend_parity_relative_error\":{backend_error:.6e},\"reciprocity_relative_error\":{reciprocity_error:.6e},\"scattered_power_w\":{power_w:.6e},\"scattered_power_relative_error\":{power_error:.6e}}}"
        );
    }
}

/// Validate the high-level production frequency × load path over a bounded
/// moderate-ka grid for all four production assembly backends (dense TBEM,
/// H-matrix, SLFMM, MLFMM). The Mie series is independent of the BEM
/// operators, so each backend carries its own analytic check rather than
/// only parity against dense; this is a sweep contract, not a claim about
/// transmission or complex-pole resonance locations.
#[cfg_attr(test, test)]
fn bem_sphere_frequency_sweep_matches_mie_v1() {
    const SWEEP_KA: [f64; 5] = [0.25, 0.375, 0.5, 0.625, 0.75];
    const MAX_SWEEP_POWER_RELATIVE_ERROR: f64 = 1.2e-1;
    const MAX_SWEEP_BACKEND_RELATIVE_ERROR: f64 = 5.0e-4;
    const MAX_SWEEP_FMM_BACKEND_RELATIVE_ERROR: f64 = 5.0e-4;

    let frequencies: Vec<f64> = SWEEP_KA.into_iter().map(frequency_hz_for_ka).collect();
    let incident_field = IncidentField::plane_wave([0.0, 0.0, 1.0], 1.0);
    let problem = BemProblem::try_rigid_sphere_scattering(
        RADIUS_M,
        frequencies[0],
        SPEED_OF_SOUND_M_PER_S,
        DENSITY_KG_PER_M3,
    )
    .expect("frequency-sweep sphere admission")
    .with_incident_field(incident_field.clone())
    .with_burton_miller(true);

    let dense = BemSolver::new()
        .with_assembly_method(AssemblyMethod::Tbem)
        .with_solver_method(SolverMethod::Direct)
        .solve_frequency_sweep_many(
            &problem,
            &frequencies,
            std::slice::from_ref(&incident_field),
        )
        .expect("dense frequency sweep");
    let hmatrix = BemSolver::new()
        .with_assembly_method(AssemblyMethod::Hmatrix)
        .with_solver_method(SolverMethod::Gmres)
        .with_max_iterations(500)
        .with_tolerance(1.0e-8)
        .solve_frequency_sweep_many(
            &problem,
            &frequencies,
            std::slice::from_ref(&incident_field),
        )
        .expect("H-matrix frequency sweep");
    let slfmm = BemSolver::new()
        .with_assembly_method(AssemblyMethod::Slfmm)
        .with_solver_method(SolverMethod::Gmres)
        .with_max_iterations(500)
        .with_tolerance(1.0e-8)
        .solve_frequency_sweep_many(
            &problem,
            &frequencies,
            std::slice::from_ref(&incident_field),
        )
        .expect("SLFMM frequency sweep");
    let mlfmm = BemSolver::new()
        .with_assembly_method(AssemblyMethod::Mlfmm)
        .with_solver_method(SolverMethod::Gmres)
        .with_max_iterations(500)
        .with_tolerance(1.0e-8)
        .solve_frequency_sweep_many(
            &problem,
            &frequencies,
            std::slice::from_ref(&incident_field),
        )
        .expect("MLFMM frequency sweep");

    assert_eq!(dense.len(), SWEEP_KA.len());
    assert_eq!(hmatrix.len(), SWEEP_KA.len());
    assert_eq!(slfmm.len(), SWEEP_KA.len());
    assert_eq!(mlfmm.len(), SWEEP_KA.len());
    let eval_radius = (12.0 * RADIUS_M).max(RADIUS_M + 1.0);
    for (index, expected_ka) in SWEEP_KA.into_iter().enumerate() {
        let dense_solution = dense[index]
            .first()
            .expect("dense sweep must return one incident-field solution");
        let hmatrix_solution = hmatrix[index]
            .first()
            .expect("H-matrix sweep must return one incident-field solution");
        let slfmm_solution = slfmm[index]
            .first()
            .expect("SLFMM sweep must return one incident-field solution");
        let mlfmm_solution = mlfmm[index]
            .first()
            .expect("MLFMM sweep must return one incident-field solution");
        let expected_frequency = frequencies[index];
        for (backend, solution) in [
            ("tbem", dense_solution),
            ("hmatrix", hmatrix_solution),
            ("slfmm", slfmm_solution),
            ("mlfmm", mlfmm_solution),
        ] {
            let frequency_error =
                (solution.physics.frequency - expected_frequency).abs() / expected_frequency;
            let ka_error =
                (solution.physics.wave_number * RADIUS_M - expected_ka).abs() / expected_ka;
            assert!(
                frequency_error <= 1.0e-12 && ka_error <= 1.0e-12,
                "{backend} sweep point {index} changed the requested frequency: expected f={expected_frequency:.6e}, ka={expected_ka:.6e}, got f={:.6e}, ka={:.6e}",
                solution.physics.frequency,
                solution.physics.wave_number * RADIUS_M
            );
            assert_eq!(
                solution.num_dofs(),
                problem.mesh.elements.len(),
                "{backend} sweep point {index} changed the active discretization"
            );
        }

        let (dense_power_error, dense_power_w) =
            scattered_power_against_mie(dense_solution).expect("dense sweep Mie power check");
        let (hmatrix_power_error, hmatrix_power_w) =
            scattered_power_against_mie(hmatrix_solution).expect("H-matrix sweep Mie power check");
        let (slfmm_power_error, slfmm_power_w) =
            scattered_power_against_mie(slfmm_solution).expect("SLFMM sweep Mie power check");
        let (mlfmm_power_error, mlfmm_power_w) =
            scattered_power_against_mie(mlfmm_solution).expect("MLFMM sweep Mie power check");
        let hmatrix_backend_error = relative_complex_error(
            scattered_pressure(hmatrix_solution, [0.0, 0.0, -eval_radius]),
            scattered_pressure(dense_solution, [0.0, 0.0, -eval_radius]),
        );
        let slfmm_backend_error = relative_complex_error(
            scattered_pressure(slfmm_solution, [0.0, 0.0, -eval_radius]),
            scattered_pressure(dense_solution, [0.0, 0.0, -eval_radius]),
        );
        let mlfmm_backend_error = relative_complex_error(
            scattered_pressure(mlfmm_solution, [0.0, 0.0, -eval_radius]),
            scattered_pressure(dense_solution, [0.0, 0.0, -eval_radius]),
        );
        assert!(
            dense_power_error <= MAX_SWEEP_POWER_RELATIVE_ERROR,
            "TBEM sweep point {index} Mie power error {dense_power_error:.3e} exceeds {MAX_SWEEP_POWER_RELATIVE_ERROR:.3e}"
        );
        assert!(
            hmatrix_power_error <= MAX_SWEEP_POWER_RELATIVE_ERROR,
            "H-matrix sweep point {index} Mie power error {hmatrix_power_error:.3e} exceeds {MAX_SWEEP_POWER_RELATIVE_ERROR:.3e}"
        );
        assert!(
            slfmm_power_error <= MAX_SWEEP_POWER_RELATIVE_ERROR,
            "SLFMM sweep point {index} Mie power error {slfmm_power_error:.3e} exceeds {MAX_SWEEP_POWER_RELATIVE_ERROR:.3e}"
        );
        assert!(
            mlfmm_power_error <= MAX_SWEEP_POWER_RELATIVE_ERROR,
            "MLFMM sweep point {index} Mie power error {mlfmm_power_error:.3e} exceeds {MAX_SWEEP_POWER_RELATIVE_ERROR:.3e}"
        );
        assert!(
            hmatrix_backend_error.is_finite()
                && hmatrix_backend_error <= MAX_SWEEP_BACKEND_RELATIVE_ERROR,
            "H-matrix sweep point {index} backscatter parity error {hmatrix_backend_error:.3e} exceeds {MAX_SWEEP_BACKEND_RELATIVE_ERROR:.3e}"
        );
        assert!(
            slfmm_backend_error.is_finite()
                && slfmm_backend_error <= MAX_SWEEP_FMM_BACKEND_RELATIVE_ERROR,
            "SLFMM sweep point {index} backscatter parity error {slfmm_backend_error:.3e} exceeds {MAX_SWEEP_FMM_BACKEND_RELATIVE_ERROR:.3e}"
        );
        assert!(
            mlfmm_backend_error.is_finite()
                && mlfmm_backend_error <= MAX_SWEEP_FMM_BACKEND_RELATIVE_ERROR,
            "MLFMM sweep point {index} backscatter parity error {mlfmm_backend_error:.3e} exceeds {MAX_SWEEP_FMM_BACKEND_RELATIVE_ERROR:.3e}"
        );
        println!(
            "BEM_EVIDENCE: {{\"case_id\":\"bem.sphere-frequency-sweep-mie.v1\",\"status\":\"passed\",\"index\":{index},\"ka\":{expected_ka:.6e},\"tbem_scattered_power_w\":{dense_power_w:.6e},\"tbem_power_relative_error\":{dense_power_error:.6e},\"hmatrix_scattered_power_w\":{hmatrix_power_w:.6e},\"hmatrix_power_relative_error\":{hmatrix_power_error:.6e},\"hmatrix_backscatter_relative_error\":{hmatrix_backend_error:.6e},\"slfmm_scattered_power_w\":{slfmm_power_w:.6e},\"slfmm_power_relative_error\":{slfmm_power_error:.6e},\"slfmm_backscatter_relative_error\":{slfmm_backend_error:.6e},\"mlfmm_scattered_power_w\":{mlfmm_power_w:.6e},\"mlfmm_power_relative_error\":{mlfmm_power_error:.6e},\"mlfmm_backscatter_relative_error\":{mlfmm_backend_error:.6e}}}"
        );
    }
}
