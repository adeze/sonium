//! Checked multilayer porous-surface impedance validation.
//!
//! This is a deterministic CPU material-model gate.  It does not launch a
//! BEM solve or any simulator workload.

use num_complex::Complex64;
use sonium_bem::core::{
    AssemblyMethod, BemProblem, BemSolver, BoundaryCondition, BoundaryConditionType, SolverMethod,
};
use sonium_bem::materials::{
    AmbientMedium, LayeredPorousSurface, MaterialError, PorousLayer, PorousModel,
};
use sonium_qa::QaResult;

const CASE_ID: &str = "bem.layered-porous-surface.v1";

struct Metrics {
    single_layer_error: f64,
    transfer_matrix_error: f64,
    derivative_error: f64,
    dimensional_derivative_error: f64,
    absorption: f64,
    solver_admittance_error: f64,
    solver_rejection_count: usize,
    passivity_rejection_count: usize,
    rejection_count: usize,
}

fn characteristic(
    layer: PorousLayer,
    frequency_hz: f64,
    medium: AmbientMedium,
) -> (Complex64, Complex64) {
    let x = (frequency_hz / layer.flow_resistivity).max(1.0e-12);
    let (zc_norm, gamma_over_k) = match layer.model {
        PorousModel::DelanyBazley => (
            Complex64::new(1.0 + 0.0571 * x.powf(-0.754), -0.087 * x.powf(-0.732)),
            Complex64::new(0.0978 * x.powf(-0.700), 1.0 + 0.189 * x.powf(-0.595)),
        ),
        PorousModel::Miki => (
            Complex64::new(1.0 + 0.070 * x.powf(-0.632), -0.107 * x.powf(-0.632)),
            Complex64::new(0.160 * x.powf(-0.618), 1.0 + 0.109 * x.powf(-0.618)),
        ),
    };
    let k0 = 2.0 * std::f64::consts::PI * frequency_hz / medium.speed_of_sound;
    (
        zc_norm * medium.characteristic_impedance(),
        gamma_over_k * k0,
    )
}

fn run() -> Result<Metrics, String> {
    let medium = AmbientMedium::default();
    let frequency = 800.0;
    let first = PorousLayer::miki(0.035, 12_000.0);
    let second = PorousLayer::delany_bazley(0.055, 18_000.0);
    let one_layer = LayeredPorousSurface::try_new(vec![first]).map_err(|e| e.to_string())?;
    let stack = LayeredPorousSurface::try_new(vec![first, second]).map_err(|e| e.to_string())?;

    let single = first
        .try_surface_impedance(frequency, medium)
        .map_err(|e| e.to_string())?;
    let stacked_single = one_layer
        .surface_impedance(frequency, medium)
        .map_err(|e| e.to_string())?;
    let single_layer_error = (single - stacked_single).norm();

    let (second_zc, second_gamma) = characteristic(second, frequency, medium);
    let second_load = second_zc / (second_gamma * second.thickness).tanh();
    let (first_zc, first_gamma) = characteristic(first, frequency, medium);
    let first_tangent = (first_gamma * first.thickness).tanh();
    let expected = first_zc * (second_load + first_zc * first_tangent)
        / (first_zc + second_load * first_tangent);
    let actual = stack
        .surface_impedance(frequency, medium)
        .map_err(|e| e.to_string())?;
    let transfer_matrix_error = (actual - expected).norm() / expected.norm().max(1.0);

    let derivative_stack = LayeredPorousSurface::try_new(vec![
        PorousLayer::miki(0.03, 10_000.0).with_air_gap(0.02),
        PorousLayer::delany_bazley(0.045, 20_000.0),
    ])
    .map_err(|e| e.to_string())?;
    let step = 0.25;
    let analytic = derivative_stack
        .surface_impedance_frequency_derivative(1_200.0, medium)
        .map_err(|e| e.to_string())?;
    let plus = derivative_stack
        .surface_impedance(1_200.0 + step, medium)
        .map_err(|e| e.to_string())?;
    let minus = derivative_stack
        .surface_impedance(1_200.0 - step, medium)
        .map_err(|e| e.to_string())?;
    let centered = (plus - minus) / (2.0 * step);
    let derivative_error =
        (analytic - centered).norm() / analytic.norm().max(centered.norm()).max(1.0);
    let dimensional_analytic = derivative_stack
        .admittance_frequency_derivative(1_200.0, medium)
        .map_err(|e| e.to_string())?;
    let dimensional_plus = derivative_stack
        .try_admittance(1_200.0 + step, medium)
        .map_err(|e| e.to_string())?;
    let dimensional_minus = derivative_stack
        .try_admittance(1_200.0 - step, medium)
        .map_err(|e| e.to_string())?;
    let dimensional_centered = (dimensional_plus - dimensional_minus) / (2.0 * step);
    let dimensional_derivative_error = (dimensional_analytic - dimensional_centered).norm()
        / dimensional_analytic
            .norm()
            .max(dimensional_centered.norm())
            .max(1.0);
    let absorption = stack
        .try_passive_absorption(frequency, medium, 1.0e-12)
        .map_err(|e| e.to_string())?;
    let passivity_rejection_count = usize::from(matches!(
        stack.try_passive_absorption(frequency, medium, -1.0),
        Err(MaterialError::InvalidTolerance(-1.0))
    ));

    let solver_problem = BemProblem::rigid_sphere_scattering_custom(
        0.1,
        frequency,
        medium.speed_of_sound,
        medium.density,
        3,
        4,
    )
    .with_boundary_condition(BoundaryConditionType::Impedance);
    let expected_solver_admittance = stack
        .try_admittance(frequency, medium)
        .map_err(|e| e.to_string())?;
    let solver_solution = BemSolver::new()
        .with_solver_method(SolverMethod::Direct)
        .with_assembly_method(AssemblyMethod::Tbem)
        .with_layered_porous_surface(stack.clone())
        .solve(&solver_problem)
        .map_err(|e| format!("layered porous BEM solve: {e}"))?;
    let mut solver_admittance_error: f64 = 0.0;
    for element in &solver_solution.elements {
        match &element.boundary_condition {
            BoundaryCondition::VelocityWithAdmittance { admittance, .. } => {
                solver_admittance_error =
                    solver_admittance_error.max((*admittance - expected_solver_admittance).norm());
            }
            other => {
                return Err(format!(
                    "layered porous solver prepared unexpected boundary condition: {other:?}"
                ));
            }
        }
    }
    let solver_rejection_count = usize::from(
        BemSolver::new()
            .with_layered_porous_surface(stack.clone())
            .solve(&BemProblem::rigid_sphere_scattering_custom(
                0.1,
                frequency,
                medium.speed_of_sound,
                medium.density,
                3,
                4,
            ))
            .is_err(),
    );

    let mut rejection_count = 0;
    if matches!(
        LayeredPorousSurface::try_new(Vec::new()),
        Err(MaterialError::InvalidLayer(_))
    ) {
        rejection_count += 1;
    }
    let invalid = LayeredPorousSurface {
        layers: vec![PorousLayer::miki(-0.01, 12_000.0)],
    };
    if matches!(
        invalid.surface_impedance(frequency, medium),
        Err(MaterialError::InvalidLayer(_))
    ) {
        rejection_count += 1;
    }
    if matches!(
        stack.surface_impedance(f64::NAN, medium),
        Err(MaterialError::InvalidFrequency(_))
    ) {
        rejection_count += 1;
    }
    if matches!(
        stack.surface_impedance(frequency, AmbientMedium::new(0.0, 343.0)),
        Err(MaterialError::InvalidMedium { .. })
    ) {
        rejection_count += 1;
    }

    if single_layer_error > 1.0e-12 {
        return Err(format!(
            "single-layer equivalence error {single_layer_error:e}"
        ));
    }
    if transfer_matrix_error > 1.0e-12 {
        return Err(format!("transfer-matrix error {transfer_matrix_error:e}"));
    }
    if derivative_error > 2.0e-5 {
        return Err(format!("frequency derivative error {derivative_error:e}"));
    }
    if dimensional_derivative_error > 2.0e-5 {
        return Err(format!(
            "dimensional frequency derivative error {dimensional_derivative_error:e}"
        ));
    }
    if !absorption.is_finite() || !(0.0..=1.0).contains(&absorption) {
        return Err(format!("invalid absorption {absorption}"));
    }
    if !solver_admittance_error.is_finite() || solver_admittance_error > 1.0e-12 {
        return Err(format!(
            "layered porous solver admittance error {solver_admittance_error:e}"
        ));
    }
    if solver_rejection_count != 1 {
        return Err(format!(
            "expected one non-impedance solver rejection, got {solver_rejection_count}"
        ));
    }
    if passivity_rejection_count != 1 {
        return Err(format!(
            "expected one passivity-tolerance rejection, got {passivity_rejection_count}"
        ));
    }
    if rejection_count != 4 {
        return Err(format!(
            "expected four typed rejections, got {rejection_count}"
        ));
    }
    Ok(Metrics {
        single_layer_error,
        transfer_matrix_error,
        derivative_error,
        dimensional_derivative_error,
        absorption,
        solver_admittance_error,
        solver_rejection_count,
        passivity_rejection_count,
        rejection_count,
    })
}

fn main() {
    match run() {
        Ok(metrics) => {
            let result = QaResult::passed(CASE_ID)
                .metric("single_layer_error", metrics.single_layer_error)
                .metric("transfer_matrix_error", metrics.transfer_matrix_error)
                .metric("frequency_derivative_error", metrics.derivative_error)
                .metric(
                    "dimensional_frequency_derivative_error",
                    metrics.dimensional_derivative_error,
                )
                .metric("absorption", metrics.absorption)
                .metric("solver_admittance_error", metrics.solver_admittance_error)
                .metric(
                    "solver_typed_rejections",
                    metrics.solver_rejection_count as f64,
                )
                .metric(
                    "passivity_typed_rejections",
                    metrics.passivity_rejection_count as f64,
                )
                .metric("typed_rejections", metrics.rejection_count as f64)
                .requested_backend("cpu")
                .actual_backend("cpu")
                .reference("independent equivalent-fluid transfer-matrix recursion")
                .dimension("normal-incidence two-layer Delany-Bazley/Miki porous stack")
                .formulation("checked multilayer porous surface impedance")
                .seed("bem-layered-porous-surface:cpu:v1");
            println!(
                "QA_RESULT: {}",
                result.to_json_line().expect("serialize QA result")
            );
            println!("SUCCESS");
        }
        Err(error) => {
            let result = QaResult::failed(CASE_ID, error.clone())
                .requested_backend("cpu")
                .actual_backend("cpu")
                .reference("checked multilayer porous surface impedance")
                .dimension("normal-incidence two-layer porous stack")
                .formulation("checked multilayer porous surface impedance")
                .seed("bem-layered-porous-surface:cpu:v1");
            println!(
                "QA_RESULT: {}",
                result.to_json_line().expect("serialize QA result")
            );
            eprintln!("Layered porous validation failed: {error}");
            println!("FAILED");
            std::process::exit(1);
        }
    }
}
