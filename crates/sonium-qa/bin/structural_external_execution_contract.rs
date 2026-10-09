//! QA for the opt-in external-process execution and evidence contract.
//!
//! The child mode is a deterministic fixture executable. It exercises the
//! same process, deck, marker, comparison, and provenance stages as a real
//! solver without pretending to be CalculiX, Code_Aster, or FEniCSx evidence.

use sonium_qa::QaResult;
use sonium_sim_structural_mechanics::{
    ExternalBenchmarkDefinition, ExternalRunOptions, ExternalSolverKind, discover_external_solvers,
    parse_external_displacement_for_parity, run_external_benchmark,
};
use std::fs;
use std::path::{Path, PathBuf};
use std::process;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

const CASE_ID: &str = "structural.external-execution-contract.v1";
const CHILD_MODE: &str = "SONIUM_EXTERNAL_EXECUTION_FIXTURE_CHILD";
const OPT_IN_EXTERNAL_MODE: &str = "SONIUM_RUN_EXTERNAL_INTEGRATION";
const RELATIVE_TOLERANCE: f64 = 1.0e-10;

#[derive(Debug, Clone, Copy)]
struct Metrics {
    solvers: usize,
    completed_stages: usize,
    maximum_relative_error: f64,
    rejected_conflicting_markers: usize,
}

fn fixture_child() -> bool {
    if std::env::var_os(CHILD_MODE).is_none() {
        return false;
    }
    let definition = ExternalBenchmarkDefinition::default();
    println!(
        "fixture solver output displacement_m={:.17e}",
        definition.expected_displacement_m()
    );
    true
}

fn temporary_directory() -> Result<PathBuf, String> {
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|error| format!("system clock before UNIX epoch: {error}"))?
        .as_nanos();
    let path = std::env::temp_dir().join(format!(
        "sonium-structural-external-execution-contract-{}-{nonce}",
        process::id()
    ));
    fs::create_dir(&path)
        .map_err(|error| format!("failed to create {}: {error}", path.display()))?;
    Ok(path)
}

struct TemporaryDirectory(PathBuf);

impl TemporaryDirectory {
    fn new() -> Result<Self, String> {
        temporary_directory().map(Self)
    }

    fn path(&self) -> &Path {
        &self.0
    }
}

impl Drop for TemporaryDirectory {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

fn run() -> Result<Metrics, String> {
    let definition = ExternalBenchmarkDefinition::default();
    definition
        .validate()
        .map_err(|error| format!("fixture definition failed validation: {error}"))?;
    let executable = std::env::current_exe()
        .map_err(|error| format!("failed to resolve fixture executable: {error}"))?;
    // The external runner intentionally owns command construction, so the
    // fixture is selected through inherited environment just like an opt-in
    // solver wrapper would be selected by its launcher configuration.
    unsafe { std::env::set_var(CHILD_MODE, "1") };
    let temporary = TemporaryDirectory::new()?;
    let solvers = [
        ExternalSolverKind::Calculix,
        ExternalSolverKind::CodeAster,
        ExternalSolverKind::Fenicsx,
    ];
    let mut completed_stages = 0;
    let mut maximum_relative_error: f64 = 0.0;
    for solver in solvers {
        let workdir = temporary
            .path()
            .join(solver.display_name().replace(' ', "-"));
        let options = ExternalRunOptions {
            workdir,
            executable: Some(executable.clone()),
            tolerance: RELATIVE_TOLERANCE,
            timeout: Duration::from_secs(10),
        };
        let result = run_external_benchmark(&definition, solver, &options).map_err(|error| {
            format!(
                "{} fixture execution failed: {error}",
                solver.display_name()
            )
        })?;
        if !result.comparison.parity_passed()
            || !result.evidence_sufficient
            || result.provenance.completed_stages.len() != 6
        {
            return Err(format!(
                "{} did not complete the evidence contract: parity={} evidence={} stages={}",
                solver.display_name(),
                result.comparison.parity_passed(),
                result.evidence_sufficient,
                result.provenance.completed_stages.len()
            ));
        }
        completed_stages += result.provenance.completed_stages.len();
        maximum_relative_error = maximum_relative_error.max(result.comparison.relative_error);
    }

    let conflicting = "displacement_m=1.0e-6\ndisplacement_m=2.0e-6";
    let rejected_conflicting_markers = solvers
        .iter()
        .filter(|&&solver| parse_external_displacement_for_parity(solver, conflicting).is_err())
        .count();
    if rejected_conflicting_markers != solvers.len() {
        return Err(format!(
            "conflicting explicit markers were not rejected for every solver: {rejected_conflicting_markers}/{}",
            solvers.len()
        ));
    }

    if std::env::var_os(OPT_IN_EXTERNAL_MODE).is_some() {
        let discovered = discover_external_solvers()
            .into_iter()
            .filter_map(|availability| {
                availability
                    .executable
                    .map(|executable| (availability.solver, executable))
            })
            .collect::<Vec<_>>();
        if discovered.is_empty() {
            return Err(format!(
                "{OPT_IN_EXTERNAL_MODE} is set, but no external solver executable was discovered"
            ));
        }

        for (solver, executable) in discovered {
            let workdir = temporary.path().join(format!(
                "actual-{}",
                solver.display_name().replace(' ', "-")
            ));
            let options = ExternalRunOptions {
                workdir,
                executable: Some(executable),
                tolerance: RELATIVE_TOLERANCE,
                timeout: Duration::from_secs(120),
            };
            let result =
                run_external_benchmark(&definition, solver, &options).map_err(|error| {
                    format!("{} actual execution failed: {error}", solver.display_name())
                })?;
            if !result.parity_passed() {
                return Err(format!(
                    "{} actual execution did not satisfy parity evidence: source={:?}, stages={:?}, relative_error={}",
                    solver.display_name(),
                    result.observation_source,
                    result.provenance.completed_stages,
                    result.comparison.relative_error
                ));
            }
        }
    }

    Ok(Metrics {
        solvers: solvers.len(),
        completed_stages,
        maximum_relative_error,
        rejected_conflicting_markers,
    })
}

fn main() {
    if fixture_child() {
        return;
    }
    let started = std::time::Instant::now();
    match run() {
        Ok(metrics) => {
            let record = QaResult::passed(CASE_ID)
                .requested_backend("cpu")
                .actual_backend("cpu")
                .metric("solvers", metrics.solvers as f64)
                .metric("completed_stages", metrics.completed_stages as f64)
                .metric("maximum_relative_error", metrics.maximum_relative_error)
                .metric(
                    "rejected_conflicting_markers",
                    metrics.rejected_conflicting_markers as f64,
                )
                .minimum("solvers", 3.0)
                .minimum("completed_stages", 18.0)
                .minimum("rejected_conflicting_markers", 3.0)
                .maximum("maximum_relative_error", RELATIVE_TOLERANCE)
                .reference("deterministic caller-supplied fixture process; validates external adapter execution/provenance only")
                .reference_checksum("synthetic/deterministic-fixture:structural-external-execution-contract-v1")
                .dimension("CalculiX, Code_Aster, and FEniCSx adapter command contracts")
                .formulation("versioned deck render/write, process completion, explicit marker parse, and numeric comparison")
                .tolerance("relative displacement error <= 1e-10; conflicting markers fail closed")
                .seed("structural-external-execution-contract:fixture-child:cpu")
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
                .seed("structural-external-execution-contract:fixture-child:cpu")
                .elapsed_ms(started.elapsed().as_millis().try_into().unwrap_or(u64::MAX));
            println!(
                "QA_RESULT: {}",
                record.to_json_line().expect("serialize QA result")
            );
            eprintln!("STRUCTURAL EXTERNAL EXECUTION CONTRACT QA FAILED: {error}");
            println!("FAILED");
            process::exit(1);
        }
    }
}
