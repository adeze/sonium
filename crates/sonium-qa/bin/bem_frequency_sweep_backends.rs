//! Structured QA for production SLFMM/MLFMM frequency-sweep dispatch.

use num_complex::Complex64;
use sonium_bem::core::bem_solver::{AssemblyMethod, BemError};
use sonium_bem::core::solver::frequency_sweep::{FrequencySweepConfig, frequency_sweep};
use sonium_bem::core::types::{BoundaryCondition, Element, ElementProperty, ElementType};
use sonium_qa::QaResult;
use std::time::Instant;

const CASE_ID: &str = "bem.frequency-sweep-backends.v1";
const MAX_RESIDUAL: f64 = 2.0e-5;
const MAX_OPERATOR_ERROR: f64 = 1.0e-3;

struct BackendSummary {
    max_residual: f64,
    max_operator_error: f64,
    total_iterations: f64,
}

fn fixture() -> (Vec<Element>, Vec<[f64; 3]>) {
    let nodes = vec![
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.5, 1.0, 0.0],
        [1.5, 1.0, 0.0],
    ];
    let zero_velocity = || BoundaryCondition::Velocity(vec![Complex64::new(0.0, 0.0)]);
    let elements = vec![
        Element {
            connectivity: vec![0, 1, 2],
            element_type: ElementType::Tri3,
            property: ElementProperty::Surface,
            normal: [0.0, 0.0, 1.0],
            node_normals: vec![[0.0; 3]; 3],
            center: [0.5, 0.33, 0.0],
            area: 0.5,
            boundary_condition: zero_velocity(),
            group: 0,
            dof_addresses: vec![0],
        },
        Element {
            connectivity: vec![1, 3, 2],
            element_type: ElementType::Tri3,
            property: ElementProperty::Surface,
            normal: [0.0, 0.0, 1.0],
            node_normals: vec![[0.0; 3]; 3],
            center: [1.0, 0.67, 0.0],
            area: 0.5,
            boundary_condition: zero_velocity(),
            group: 0,
            dof_addresses: vec![1],
        },
    ];
    (elements, nodes)
}

fn run_backend(method: AssemblyMethod) -> Result<BackendSummary, BemError> {
    let (elements, nodes) = fixture();
    let config = FrequencySweepConfig {
        frequencies: vec![500.0, 550.0, 600.0],
        tolerance: 1.0e-6,
        max_iterations: 100,
        recycle_solution: true,
        gmres_restart: 20,
        beta_scale: 4.0,
        recycle_subspace_size: 2,
        max_recycle_relative_frequency_delta: 0.1,
    };
    let result = frequency_sweep(
        &elements,
        &nodes,
        &sonium_bem::core::incident::IncidentField::plane_wave_z(),
        &config,
        method,
        343.0,
        1.21,
    )?;
    result.validate()?;
    if !result.converged.iter().all(|&value| value) {
        return Err(BemError::SolverFailed(format!(
            "{method:?} frequency sweep did not converge: {:?}",
            result.converged
        )));
    }
    Ok(BackendSummary {
        max_residual: result
            .exact_relative_residuals
            .iter()
            .copied()
            .fold(0.0, f64::max),
        max_operator_error: result
            .operator_accuracy_errors
            .iter()
            .copied()
            .fold(0.0, f64::max),
        total_iterations: result.total_iterations() as f64,
    })
}

fn main() {
    let started = Instant::now();
    let result = (|| -> Result<QaResult, String> {
        let slfmm = run_backend(AssemblyMethod::Slfmm).map_err(|error| error.to_string())?;
        let mlfmm = run_backend(AssemblyMethod::Mlfmm).map_err(|error| error.to_string())?;
        let max_residual = slfmm.max_residual.max(mlfmm.max_residual);
        let max_operator_error = slfmm.max_operator_error.max(mlfmm.max_operator_error);
        let total_iterations = slfmm.total_iterations + mlfmm.total_iterations;
        if max_residual > MAX_RESIDUAL || max_operator_error > MAX_OPERATOR_ERROR {
            return Err(format!(
                "FMM sweep evidence exceeded bounds: residual={max_residual:.3e}, operator_error={max_operator_error:.3e}"
            ));
        }
        Ok(QaResult::passed(CASE_ID)
            .requested_backend("cpu")
            .actual_backend("cpu")
            .metric("max_exact_relative_residual", max_residual)
            .metric("max_operator_accuracy_error", max_operator_error)
            .metric("total_iterations", total_iterations)
            .maximum("max_exact_relative_residual", MAX_RESIDUAL)
            .maximum("max_operator_accuracy_error", MAX_OPERATOR_ERROR)
            .input_checksum("analytic/deterministic-fixture:bem-frequency-sweep-v1")
            .reference("independent matrix-free stabilized TBEM action")
            .reference_checksum("analytic/deterministic-fixture")
            .dimension("two-panel 2D fixture, three frequencies, SLFMM and MLFMM")
            .formulation(
                "SLFMM and MLFMM frequency sweep against matrix-free stabilized TBEM action",
            )
            .tolerance("exact residual <= 2e-5; operator action error <= 1e-3")
            .seed("bem-frequency-sweep-backends:two-panel:500-600hz:v1")
            .timing_ms("total", started.elapsed().as_millis() as u64)
            .elapsed_ms(started.elapsed().as_millis() as u64))
    })();

    match result {
        Ok(record) => {
            println!(
                "QA_RESULT: {}",
                record.to_json_line().expect("BEM sweep result serializes")
            );
            println!("SUCCESS");
        }
        Err(error) => {
            let record =
                QaResult::failed(CASE_ID, error).elapsed_ms(started.elapsed().as_millis() as u64);
            println!(
                "QA_RESULT: {}",
                record.to_json_line().expect("BEM sweep failure serializes")
            );
            println!("FAILED");
            std::process::exit(1);
        }
    }
}
