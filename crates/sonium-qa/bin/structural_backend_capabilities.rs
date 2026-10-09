//! Compile-only/reference QA for structural backend capability declarations.

use sonium_qa::QaResult;
use sonium_sim_structural_mechanics::{
    STRUCTURAL_CAPABILITY_SCHEMA_VERSION, StructuralAnalysisFamily, StructuralBackend,
    StructuralExecutionMode, StructuralPrecisionPolicy, structural_backend_capabilities,
};

const CASE_ID: &str = "structural.backend-capability-catalogue.v1";

fn run() -> Result<(usize, usize), String> {
    let catalogue = structural_backend_capabilities();
    if catalogue.len() != 4
        || catalogue
            .iter()
            .any(|capability| capability.schema_version != STRUCTURAL_CAPABILITY_SCHEMA_VERSION)
    {
        return Err("structural capability schema/catalogue shape failed".into());
    }
    let cpu = catalogue
        .iter()
        .find(|capability| capability.backend == StructuralBackend::Cpu)
        .ok_or_else(|| "CPU capability is missing".to_string())?;
    if !cpu.integrated
        || cpu.execution_mode != StructuralExecutionMode::CpuReference
        || !cpu.supports(
            StructuralAnalysisFamily::TopologyOptimization,
            StructuralPrecisionPolicy::F64,
        )
    {
        return Err("CPU reference capability gate failed".into());
    }

    let unsupported = catalogue
        .iter()
        .filter(|capability| capability.execution_mode == StructuralExecutionMode::Unsupported)
        .count();
    if unsupported != 3
        || catalogue.iter().any(|capability| {
            capability.execution_mode == StructuralExecutionMode::Unsupported
                && (capability.integrated
                    || !capability.analysis_families.is_empty()
                    || !capability.precision_policies.is_empty())
        })
    {
        return Err("unwired backend fail-closed gate failed".into());
    }
    Ok((catalogue.len(), unsupported))
}

fn main() {
    let started = std::time::Instant::now();
    match run() {
        Ok((capabilities, unsupported)) => {
            let record = QaResult::passed(CASE_ID)
                .requested_backend("cpu")
                .actual_backend("cpu")
                .metric("capabilities", capabilities as f64)
                .metric("unsupported_backends", unsupported as f64)
                .minimum("capabilities", 4.0)
                .minimum("unsupported_backends", 3.0)
                .reference("structural CPU-reference capability catalogue with fail-closed accelerator declarations")
                .reference_checksum("analytic/deterministic-fixture:structural-backend-capability-catalogue-v1")
                .dimension("CPU, CUDA, Metal, and WASM structural backend declarations")
                .formulation("schema/version, execution-mode, integration, precision, and family-support gates")
                .tolerance("one integrated CPU/f64 reference; three explicitly unsupported backends")
                .seed("structural-backend-capabilities:catalogue:cpu")
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
                .seed("structural-backend-capabilities:catalogue:cpu")
                .elapsed_ms(started.elapsed().as_millis().try_into().unwrap_or(u64::MAX));
            if let Ok(json) = record.to_json_line() {
                println!("QA_RESULT: {json}");
            }
            eprintln!("STRUCTURAL BACKEND CAPABILITY QA FAILED: {error}");
            println!("FAILED");
            std::process::exit(1);
        }
    }
}
