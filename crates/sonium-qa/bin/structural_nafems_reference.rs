//! Bounded structural validation catalogue QA.
//!
//! These four cases exercise the production static, modal, thermal-stress, and
//! nonlinear validation axes. They are analytical surrogates and deliberately
//! cannot be reported as official NAFEMS parity without caller-supplied
//! published-reference data.

use sonium_qa::QaResult;
use sonium_sim_structural_mechanics::{
    NafemsBenchmarkKind, StructuralBackend, StructuralExecutionMode, StructuralPrecisionPolicy,
    ValidationReferenceKind, run_nafems_benchmark_suite,
};

const CASE_ID: &str = "structural.nafems-reference-catalogue.v1";
const EXPECTED_CASES: usize = 4;
const MAX_RELATIVE_ERROR: f64 = 5.0e-2;

#[derive(Debug, Clone, Copy)]
struct Metrics {
    cases: usize,
    passed_cases: usize,
    maximum_relative_error: f64,
    maximum_solver_residual: f64,
    maximum_energy_error: f64,
}

fn run() -> Result<Metrics, String> {
    let suite = run_nafems_benchmark_suite().map_err(|error| error.to_string())?;
    if suite.schema_version != 1 {
        return Err(format!(
            "unexpected structural validation schema {}",
            suite.schema_version
        ));
    }
    if suite.cases.len() != EXPECTED_CASES {
        return Err(format!(
            "expected {EXPECTED_CASES} validation axes, got {}",
            suite.cases.len()
        ));
    }
    if suite.official_reference_data_loaded || suite.official_parity_passed() {
        return Err("analytical surrogate catalogue was promoted as official parity".into());
    }
    if !suite.execution.capability_verified
        || suite.execution.backend != StructuralBackend::Cpu
        || suite.execution.execution_mode != StructuralExecutionMode::CpuReference
        || suite.execution.precision_policy != StructuralPrecisionPolicy::F64
    {
        return Err(format!(
            "unexpected execution provenance: {:?}",
            suite.execution
        ));
    }

    let mut kind_counts = [0usize; 4];
    let mut maximum_relative_error = 0.0_f64;
    let mut maximum_solver_residual: f64 = 0.0;
    let mut maximum_energy_error: f64 = 0.0;
    for case in &suite.cases {
        if case.reference_kind != ValidationReferenceKind::AnalyticalSurrogate
            || case.reference_source.trim().is_empty()
            || case.applicability.trim().is_empty()
        {
            return Err(format!("{} is missing surrogate provenance", case.id));
        }
        if !case.passed
            || !case.computed_value.is_finite()
            || !case.reference_value.is_finite()
            || !case.relative_error.is_finite()
            || !case.tolerance.is_finite()
            || case.relative_error > case.tolerance
        {
            return Err(format!(
                "{} failed its validation gate: passed={} error={} tolerance={}",
                case.id, case.passed, case.relative_error, case.tolerance
            ));
        }
        if case.relative_error > MAX_RELATIVE_ERROR {
            return Err(format!(
                "{} exceeds catalogue error bound: {} > {}",
                case.id, case.relative_error, MAX_RELATIVE_ERROR
            ));
        }
        let index = match case.kind {
            NafemsBenchmarkKind::Static => 0,
            NafemsBenchmarkKind::Modal => 1,
            NafemsBenchmarkKind::ThermalStress => 2,
            NafemsBenchmarkKind::Nonlinear => 3,
        };
        kind_counts[index] += 1;
        maximum_relative_error = maximum_relative_error.max(case.relative_error);
        if let Some(residual) = case.solver_residual {
            if !residual.is_finite() {
                return Err(format!("{} has a non-finite solver residual", case.id));
            }
            maximum_solver_residual = maximum_solver_residual.max(residual);
        }
        if let Some(error) = case.energy_error {
            if !error.is_finite() {
                return Err(format!("{} has a non-finite energy error", case.id));
            }
            maximum_energy_error = maximum_energy_error.max(error);
        }
    }
    if kind_counts != [1, 1, 1, 1] {
        return Err(format!(
            "validation axes are not represented exactly once: {kind_counts:?}"
        ));
    }

    Ok(Metrics {
        cases: suite.cases.len(),
        passed_cases: suite.cases.iter().filter(|case| case.passed).count(),
        maximum_relative_error,
        maximum_solver_residual,
        maximum_energy_error,
    })
}

fn main() {
    let started = std::time::Instant::now();
    match run() {
        Ok(metrics) => {
            let record = QaResult::passed(CASE_ID)
                .requested_backend("cpu")
                .actual_backend("cpu")
                .metric("cases", metrics.cases as f64)
                .metric("passed_cases", metrics.passed_cases as f64)
                .metric("maximum_relative_error", metrics.maximum_relative_error)
                .metric("maximum_solver_residual", metrics.maximum_solver_residual)
                .metric("maximum_energy_error", metrics.maximum_energy_error)
                .minimum("cases", EXPECTED_CASES as f64)
                .minimum("passed_cases", EXPECTED_CASES as f64)
                .maximum("maximum_relative_error", MAX_RELATIVE_ERROR)
                .reference("native static/modal/thermal-stress/nonlinear analytical surrogate catalogue")
                .reference_checksum("analytic/deterministic-fixture:structural-nafems-reference-catalogue-v1")
                .dimension("four NAFEMS validation axes: static, modal, thermal-stress, nonlinear")
                .formulation("production structural studies with case-specific analytical comparison gates")
                .tolerance("all four cases pass their case-specific tolerance; official reference data remains unloaded")
                .seed("structural-nafems-reference-catalogue:cpu")
                .elapsed_ms(started.elapsed().as_millis().try_into().unwrap_or(u64::MAX));
            println!(
                "QA_RESULT: {}",
                record.to_json_line().expect("serialize QA result")
            );
            println!("SUCCESS");
        }
        Err(error) => {
            let record = QaResult::failed(CASE_ID, error.clone())
                .requested_backend("cpu")
                .actual_backend("cpu")
                .seed("structural-nafems-reference-catalogue:cpu")
                .elapsed_ms(started.elapsed().as_millis().try_into().unwrap_or(u64::MAX));
            println!(
                "QA_RESULT: {}",
                record.to_json_line().expect("serialize QA result")
            );
            eprintln!("STRUCTURAL NAFEMS REFERENCE QA FAILED: {error}");
            println!("FAILED");
            std::process::exit(1);
        }
    }
}
