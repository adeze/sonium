//! Fail-closed admission sentinel for public BEM incident-field variants.
//!
//! This deliberately performs no solve: malformed source data must be rejected
//! by `BemProblem::validate` before mesh assembly or a numerical kernel starts.

use num_complex::Complex64;
use sonium_bem::core::{BemError, BemProblem, BemSolver, IncidentField};
use sonium_qa::QaResult;
use std::time::Instant;

const CASE_ID: &str = "bem.incident-field-admission.v1";

fn run() -> Result<(usize, usize), String> {
    let valid = [
        IncidentField::plane_wave_z(),
        IncidentField::point_source([0.2, -0.1, 0.3], 0.7),
        IncidentField::MultiplePlaneWaves(vec![([1.0, 0.0, 0.0], Complex64::new(0.4, -0.2))]),
        IncidentField::MultiplePointSources(vec![([-0.2, 0.1, 0.4], Complex64::new(-0.3, 0.6))]),
    ];
    let valid_count = valid.len();
    for field in &valid {
        field
            .validate()
            .map_err(|error| format!("finite incident field rejected: {error}"))?;
    }

    let invalid = [
        IncidentField::PlaneWave {
            direction: [f64::NAN, 0.0, 1.0],
            amplitude: Complex64::new(1.0, 0.0),
        },
        IncidentField::PointSource {
            position: [0.0, f64::INFINITY, 0.0],
            strength: Complex64::new(1.0, 0.0),
        },
        IncidentField::MultiplePlaneWaves(vec![([0.0, 1.0, 0.0], Complex64::new(f64::NAN, 0.0))]),
        IncidentField::MultiplePointSources(vec![(
            [0.0, 0.0, 1.0],
            Complex64::new(0.0, f64::INFINITY),
        )]),
    ];

    let solver = BemSolver::new();
    let mut rejected = 0;
    for field in invalid {
        let mut problem = BemProblem::rigid_sphere_scattering_custom(0.1, 100.0, 343.0, 1.21, 3, 6);
        problem.incident_field = field;
        let result = solver.solve(&problem);
        if matches!(result, Err(BemError::InvalidParameters(ref message)) if message.contains("incident field"))
        {
            rejected += 1;
        } else {
            return Err(format!(
                "malformed incident field was not rejected: {result:?}"
            ));
        }
    }
    Ok((valid_count, rejected))
}

fn main() {
    let started = Instant::now();
    let result = std::panic::catch_unwind(run);
    let qa = match result {
        Ok(Ok((valid, rejected))) if valid == 4 && rejected == 4 => QaResult::passed(CASE_ID)
            .requested_backend("cpu")
            .actual_backend("cpu")
            .reference("typed IncidentField finite-value admission")
            .reference_checksum("analytic/deterministic-fixture:bem-incident-field-admission-v1")
            .input_checksum("synthetic/deterministic-fixture:bem-incident-field-admission-v1")
            .dimension("four incident families; four malformed source cases")
            .formulation("BemProblem::validate before assembly")
            .metric("finite_cases_accepted", valid as f64)
            .metric("malformed_cases_rejected", rejected as f64)
            .minimum("finite_cases_accepted", 4.0)
            .minimum("malformed_cases_rejected", 4.0)
            .tolerance("all malformed values rejected before numerical assembly")
            .elapsed_ms(started.elapsed().as_millis() as u64),
        Ok(Ok((valid, rejected))) => QaResult::failed(
            CASE_ID,
            format!("unexpected admission counts: finite={valid}, rejected={rejected}"),
        )
        .elapsed_ms(started.elapsed().as_millis() as u64),
        Ok(Err(error)) => {
            QaResult::failed(CASE_ID, error).elapsed_ms(started.elapsed().as_millis() as u64)
        }
        Err(payload) => QaResult::failed(CASE_ID, format!("panic: {payload:?}"))
            .elapsed_ms(started.elapsed().as_millis() as u64),
    };
    println!("{}", qa.to_json_line().expect("serialize QA result"));
    if !matches!(qa.status, sonium_qa::QaStatus::Passed) {
        std::process::exit(1);
    }
}
