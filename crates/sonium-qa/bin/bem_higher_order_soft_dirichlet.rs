//! Release-backed higher-order indirect soft-Dirichlet BEM contract.

use sonium_bem::core::{
    BasisContinuity, BasisOrder, BemProblem, BemSolver, BoundaryConditionType, SolverMethod,
};
use sonium_qa::{QaResult, QaStatus};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::time::Instant;

const CASE_ID: &str = "bem.higher-order-soft-dirichlet.v1";

#[derive(Debug, Clone, Copy)]
struct Metrics {
    configuration_count: usize,
    max_dofs: usize,
    max_surface_pressure: f64,
    finite_field_configurations: usize,
}

fn run() -> Result<Metrics, String> {
    let problem = BemProblem::rigid_sphere_scattering_custom(0.1, 100.0, 343.0, 1.21, 3, 4)
        .with_boundary_condition(BoundaryConditionType::Soft);
    let points = [[0.0, 0.0, 0.14], [0.08, 0.02, 0.12], [-0.04, 0.07, 0.13]];
    let configurations = [
        (BasisOrder::Linear, BasisContinuity::Discontinuous),
        (BasisOrder::Quadratic, BasisContinuity::Discontinuous),
        (BasisOrder::Quadratic, BasisContinuity::Continuous),
    ];
    let mut max_dofs = 0;
    let mut max_surface_pressure = 0.0_f64;
    let mut finite_field_configurations = 0;
    for (basis, continuity) in configurations {
        let solver = BemSolver::new()
            .with_basis_order(basis)
            .with_basis_continuity(continuity)
            .with_solver_method(SolverMethod::Direct);
        let solution = solver
            .solve(&problem)
            .map_err(|error| format!("{basis:?}/{continuity:?} soft solve: {error}"))?;
        if solution.num_dofs() == 0
            || !solution
                .surface_pressure
                .iter()
                .all(|value| value.re.is_finite() && value.im.is_finite())
        {
            return Err(format!(
                "{basis:?}/{continuity:?} produced invalid higher-order soft coefficients"
            ));
        }
        let fields = solution
            .evaluate_pressure_field_checked(&points)
            .map_err(|error| format!("{basis:?}/{continuity:?} field evaluation: {error}"))?;
        if fields
            .iter()
            .any(|field| !field.p_total.re.is_finite() || !field.p_total.im.is_finite())
        {
            return Err(format!(
                "{basis:?}/{continuity:?} produced non-finite total field"
            ));
        }
        finite_field_configurations += 1;
        max_dofs = max_dofs.max(solution.num_dofs());
        max_surface_pressure = max_surface_pressure.max(solution.max_surface_pressure());
    }
    Ok(Metrics {
        configuration_count: configurations.len(),
        max_dofs,
        max_surface_pressure,
        finite_field_configurations,
    })
}

fn main() {
    let started = Instant::now();
    let result = match catch_unwind(AssertUnwindSafe(run)) {
        Ok(Ok(metrics)) => QaResult::passed(CASE_ID)
            .metric("configuration_count", metrics.configuration_count as f64)
            .metric("max_dofs", metrics.max_dofs as f64)
            .metric("max_surface_pressure", metrics.max_surface_pressure)
            .metric(
                "finite_field_configurations",
                metrics.finite_field_configurations as f64,
            )
            .minimum("configuration_count", 3.0)
            .minimum("finite_field_configurations", 3.0)
            .minimum("max_dofs", 1.0)
            .input_checksum("analytic/deterministic-fixture:bem-higher-order-soft-dirichlet-v1")
            .reference_checksum("analytic/deterministic-fixture:bem-higher-order-soft-dirichlet-v1")
            .formulation(
                "Galerkin single-layer indirect Dirichlet with projected incident pressure",
            )
            .dimension("3-D rigid sphere; discontinuous P1/P2 and continuous P2 traces")
            .seed("bem-higher-order-soft-dirichlet:cpu:v1")
            .elapsed_ms(started.elapsed().as_millis().try_into().unwrap_or(u64::MAX)),
        Ok(Err(error)) => QaResult::failed(CASE_ID, error)
            .elapsed_ms(started.elapsed().as_millis().try_into().unwrap_or(u64::MAX)),
        Err(_) => QaResult::failed(
            CASE_ID,
            "panic during higher-order soft-Dirichlet validation",
        )
        .elapsed_ms(started.elapsed().as_millis().try_into().unwrap_or(u64::MAX)),
    };
    println!(
        "QA_RESULT: {}",
        result.to_json_line().expect("serialize QA result")
    );
    if !matches!(&result.status, QaStatus::Passed) {
        std::process::exit(1);
    }
}
