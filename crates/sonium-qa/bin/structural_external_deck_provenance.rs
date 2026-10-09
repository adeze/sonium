//! Bounded CPU QA for versioned structural external-deck provenance.
//!
//! This packet validates deck generation and filesystem safety only. It never
//! launches CalculiX, Code_Aster, FEniCSx, or any other external process.

use sonium_qa::QaResult;
use sonium_sim_structural_mechanics::{
    EXTERNAL_BENCHMARK_SCHEMA_VERSION, EXTERNAL_BENCHMARK_STATUS_SCHEMA_VERSION,
    ExternalBenchmarkDeck, ExternalBenchmarkDefinition, ExternalComparisonStatus,
    ExternalSolverKind, external_benchmark_status, render_external_decks,
};
use std::{
    collections::BTreeSet,
    fs,
    path::{Component, Path, PathBuf},
    time::{SystemTime, UNIX_EPOCH},
};

const CASE_ID: &str = "structural.external-deck-provenance.v1";

#[derive(Debug, Clone, Copy)]
struct Metrics {
    expected_displacement_m: f64,
    definition_json_bytes: usize,
    deck_count: usize,
    input_file_count: usize,
    written_file_count: usize,
    available_solver_count: usize,
}

fn main() {
    match run() {
        Ok(metrics) => {
            let record = QaResult::passed(CASE_ID)
                .requested_backend("cpu")
                .actual_backend("cpu")
                .metric("expected_displacement_m", metrics.expected_displacement_m)
                .metric(
                    "definition_json_bytes",
                    metrics.definition_json_bytes as f64,
                )
                .metric("deck_count", metrics.deck_count as f64)
                .metric("input_file_count", metrics.input_file_count as f64)
                .metric("written_file_count", metrics.written_file_count as f64)
                .metric(
                    "available_solver_count",
                    metrics.available_solver_count as f64,
                )
                .minimum("deck_count", 3.0)
                .minimum("input_file_count", 4.0)
                .minimum("written_file_count", 4.0)
                .reference("closed-form axial displacement and independently checked deck/file provenance")
                .dimension("3D axial solid external-deck adapter")
                .formulation("CPU-only validation of CalculiX, Code_Aster, and FEniCSx deck declarations, JSON roundtrip, safe writes, and NotRun external execution status")
                .seed("structural-external-deck-provenance:default:cpu");
            if let Ok(json) = record.to_json_line() {
                println!("QA_RESULT: {json}");
            }
            println!("SUCCESS");
        }
        Err(error) => {
            let record = QaResult::failed(CASE_ID, error.clone())
                .requested_backend("cpu")
                .actual_backend("cpu")
                .seed("structural-external-deck-provenance:default:cpu");
            if let Ok(json) = record.to_json_line() {
                println!("QA_RESULT: {json}");
            }
            eprintln!("STRUCTURAL EXTERNAL DECK PROVENANCE FAILED: {error}");
            println!("FAILED");
            std::process::exit(1);
        }
    }
}

fn run() -> Result<Metrics, String> {
    let definition = ExternalBenchmarkDefinition::default();
    definition
        .validate()
        .map_err(|error| format!("default external definition failed validation: {error}"))?;

    let expected_displacement_m = definition.expected_displacement_m();
    if !expected_displacement_m.is_finite() || expected_displacement_m <= 0.0 {
        return Err(format!(
            "default closed-form displacement is invalid: {expected_displacement_m}"
        ));
    }

    let definition_json = serde_json::to_string(&definition)
        .map_err(|error| format!("failed to serialize default definition: {error}"))?;
    let round_tripped: ExternalBenchmarkDefinition = serde_json::from_str(&definition_json)
        .map_err(|error| format!("failed to deserialize default definition: {error}"))?;
    if round_tripped != definition {
        return Err("default definition changed during serde JSON round-trip".into());
    }
    round_tripped
        .validate()
        .map_err(|error| format!("round-tripped definition failed validation: {error}"))?;

    let status = external_benchmark_status();
    if status.schema_version != EXTERNAL_BENCHMARK_STATUS_SCHEMA_VERSION {
        return Err(format!(
            "unexpected external benchmark status schema version: {}",
            status.schema_version
        ));
    }
    if status.status != ExternalComparisonStatus::NotRun {
        return Err(format!(
            "default external benchmark status was {:?}, expected NotRun",
            status.status
        ));
    }
    if status.evidence_sufficient || status.parity_passed || status.parity_passed() {
        return Err(
            "default external benchmark status incorrectly claims sufficient evidence".into(),
        );
    }

    let decks = render_external_decks(&definition)
        .map_err(|error| format!("failed to render external decks: {error}"))?;
    let expected_solvers = [
        ExternalSolverKind::Calculix,
        ExternalSolverKind::CodeAster,
        ExternalSolverKind::Fenicsx,
    ];
    let actual_solvers: Vec<_> = decks.iter().map(|deck| deck.solver).collect();
    if actual_solvers != expected_solvers.to_vec() {
        return Err(format!(
            "rendered solver order/kinds were {:?}, expected {:?}",
            actual_solvers, expected_solvers
        ));
    }

    let temporary_directory = TemporaryDirectory::new()?;
    let mut input_file_count = 0usize;
    let mut written_file_count = 0usize;
    for (index, (deck, solver)) in decks.iter().zip(expected_solvers).enumerate() {
        verify_deck_declarations(&definition, deck, solver)?;
        input_file_count += deck.files.len();

        let solver_directory = temporary_directory
            .path()
            .join(format!("{index}-{}", solver.display_name()));
        let written = deck
            .write_to(&solver_directory)
            .map_err(|error| format!("{} deck write failed: {error}", solver.display_name()))?;
        if written.len() != deck.files.len() {
            return Err(format!(
                "{} wrote {} files but declared {} input files",
                solver.display_name(),
                written.len(),
                deck.files.len()
            ));
        }

        for (input, path) in deck.files.iter().zip(written.iter()) {
            let relative = safe_relative_path(&input.relative_path)?;
            let expected_path = solver_directory.join(relative);
            if path != &expected_path || !path.starts_with(&solver_directory) {
                return Err(format!(
                    "{} wrote input outside its deck directory: {}",
                    solver.display_name(),
                    path.display()
                ));
            }
            let metadata = fs::metadata(path).map_err(|error| {
                format!(
                    "{} written input {} is not readable: {error}",
                    solver.display_name(),
                    path.display()
                )
            })?;
            if !metadata.is_file() || metadata.len() == 0 {
                return Err(format!(
                    "{} written input {} is not a non-empty file",
                    solver.display_name(),
                    path.display()
                ));
            }
            if fs::read(path)
                .map_err(|error| format!("failed to read {}: {error}", path.display()))?
                .is_empty()
            {
                return Err(format!(
                    "{} written input {} has empty contents",
                    solver.display_name(),
                    path.display()
                ));
            }
            written_file_count += 1;
        }
    }

    let mut traversal_probe = decks[0].clone();
    traversal_probe.files[0].relative_path = "../escape.out".to_string();
    let traversal_directory = temporary_directory.path().join("traversal-probe");
    if traversal_probe.write_to(&traversal_directory).is_ok() {
        return Err("deck writer accepted a parent-directory traversal path".into());
    }
    if temporary_directory.path().join("escape.out").exists() {
        return Err("deck writer created a file outside its destination directory".into());
    }

    Ok(Metrics {
        expected_displacement_m,
        definition_json_bytes: definition_json.len(),
        deck_count: decks.len(),
        input_file_count,
        written_file_count,
        available_solver_count: status.available_solvers.len(),
    })
}

fn verify_deck_declarations(
    definition: &ExternalBenchmarkDefinition,
    deck: &ExternalBenchmarkDeck,
    solver: ExternalSolverKind,
) -> Result<(), String> {
    if deck.schema_version != EXTERNAL_BENCHMARK_SCHEMA_VERSION {
        return Err(format!(
            "{} deck has schema version {}, expected {}",
            solver.display_name(),
            deck.schema_version,
            EXTERNAL_BENCHMARK_SCHEMA_VERSION
        ));
    }
    if deck.benchmark_name != definition.name {
        return Err(format!(
            "{} deck benchmark name {:?} does not match {:?}",
            solver.display_name(),
            deck.benchmark_name,
            definition.name
        ));
    }

    let case = &definition.name;
    let (expected_primary_file, expected_output_files): (String, Vec<String>) = match solver {
        ExternalSolverKind::Calculix => (
            format!("{case}.inp"),
            vec![format!("{case}.dat"), format!("{case}.frd")],
        ),
        ExternalSolverKind::CodeAster => (format!("{case}.comm"), vec![format!("{case}.resu")]),
        ExternalSolverKind::Fenicsx => (format!("{case}.py"), Vec::new()),
    };
    if deck.primary_file != expected_primary_file {
        return Err(format!(
            "{} primary file {:?} does not match {:?}",
            solver.display_name(),
            deck.primary_file,
            expected_primary_file
        ));
    }
    if deck.expected_output_files != expected_output_files {
        return Err(format!(
            "{} output declarations {:?} do not match {:?}",
            solver.display_name(),
            deck.expected_output_files,
            expected_output_files
        ));
    }

    safe_relative_path(&deck.primary_file)?;
    for output in &deck.expected_output_files {
        safe_relative_path(output)?;
    }

    let mut input_paths = BTreeSet::new();
    for input in &deck.files {
        safe_relative_path(&input.relative_path)?;
        if !input_paths.insert(input.relative_path.as_str()) {
            return Err(format!(
                "{} deck repeats input path {:?}",
                solver.display_name(),
                input.relative_path
            ));
        }
        if input.contents.is_empty() {
            return Err(format!(
                "{} deck input {:?} has empty contents",
                solver.display_name(),
                input.relative_path
            ));
        }
    }
    if !input_paths.contains(expected_primary_file.as_str()) {
        return Err(format!(
            "{} primary file {:?} is not among input files",
            solver.display_name(),
            deck.primary_file
        ));
    }

    Ok(())
}

fn safe_relative_path(value: &str) -> Result<&Path, String> {
    if value.is_empty() || value.contains('\0') || value.contains('\\') {
        return Err(format!("unsafe relative path {:?}", value));
    }
    let path = Path::new(value);
    if path.is_absolute()
        || path.components().any(|component| {
            matches!(
                component,
                Component::Prefix(_)
                    | Component::RootDir
                    | Component::ParentDir
                    | Component::CurDir
            )
        })
    {
        return Err(format!("unsafe relative path {:?}", value));
    }
    if !path
        .components()
        .any(|component| matches!(component, Component::Normal(_)))
    {
        return Err(format!(
            "relative path {:?} has no filename component",
            value
        ));
    }
    Ok(path)
}

struct TemporaryDirectory {
    path: PathBuf,
}

impl TemporaryDirectory {
    fn new() -> Result<Self, String> {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_err(|error| format!("system clock is before UNIX epoch: {error}"))?
            .as_nanos();
        let path = std::env::temp_dir().join(format!(
            "sonium-structural-external-deck-provenance-{}-{nonce}",
            std::process::id()
        ));
        fs::create_dir(&path).map_err(|error| {
            format!(
                "failed to create unique temporary directory {}: {error}",
                path.display()
            )
        })?;
        Ok(Self { path })
    }

    fn path(&self) -> &Path {
        &self.path
    }
}

impl Drop for TemporaryDirectory {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.path);
    }
}
