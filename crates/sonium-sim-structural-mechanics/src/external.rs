//! Versioned external-solver benchmark adapters.
//!
//! This module keeps external parity evidence separate from the native FEM
//! implementation. It owns a small, reproducible 3D axial-solid benchmark,
//! renders inputs for CalculiX, Code_Aster, and FEniCSx, discovers configured
//! executables, and parses a common displacement result marker. External
//! execution is opt-in: the native benchmark suite records adapter/deck
//! availability but never launches an untrusted or unexpectedly expensive
//! process by default.

use serde::{Deserialize, Serialize};
use std::path::{Component, Path, PathBuf};
use std::process::{Command, Stdio};
use std::thread;
use std::time::{Duration, Instant};

use crate::{Solid3DStudy, StructuralMechanicsError};

/// Schema version for generated external benchmark inputs and comparisons.
pub const EXTERNAL_BENCHMARK_SCHEMA_VERSION: u32 = 1;

/// External structural solver supported by the adapter layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ExternalSolverKind {
    Calculix,
    CodeAster,
    Fenicsx,
}

impl ExternalSolverKind {
    pub fn display_name(self) -> &'static str {
        match self {
            Self::Calculix => "CalculiX",
            Self::CodeAster => "Code_Aster",
            Self::Fenicsx => "FEniCSx",
        }
    }

    fn executable_env_var(self) -> &'static str {
        match self {
            Self::Calculix => "SONIUM_CALCULIX",
            Self::CodeAster => "SONIUM_CODE_ASTER",
            Self::Fenicsx => "SONIUM_FENICSX_PYTHON",
        }
    }

    fn default_executable(self) -> &'static str {
        match self {
            Self::Calculix => "ccx",
            Self::CodeAster => "as_run",
            Self::Fenicsx => "python3",
        }
    }
}

impl std::fmt::Display for ExternalSolverKind {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(self.display_name())
    }
}

/// Reproducible 3D axial-solid benchmark definition.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ExternalBenchmarkDefinition {
    pub schema_version: u32,
    pub name: String,
    pub length_m: f64,
    pub width_m: f64,
    pub height_m: f64,
    pub youngs_modulus_pa: f64,
    pub poisson_ratio: f64,
    pub density_kg_m3: f64,
    pub force_n: f64,
}

impl Default for ExternalBenchmarkDefinition {
    fn default() -> Self {
        Self {
            schema_version: EXTERNAL_BENCHMARK_SCHEMA_VERSION,
            name: "solid3d-axial-external-v1".to_string(),
            length_m: 0.12,
            width_m: 0.04,
            height_m: 0.04,
            youngs_modulus_pa: 70.0e9,
            poisson_ratio: 0.33,
            density_kg_m3: 2700.0,
            force_n: 100.0,
        }
    }
}

impl From<&Solid3DStudy> for ExternalBenchmarkDefinition {
    fn from(study: &Solid3DStudy) -> Self {
        Self {
            schema_version: EXTERNAL_BENCHMARK_SCHEMA_VERSION,
            name: format!("{}-external-v1", study.name),
            length_m: study.geometry.length_m,
            width_m: study.geometry.width_m,
            height_m: study.geometry.height_m,
            youngs_modulus_pa: study.material.youngs_modulus_pa,
            poisson_ratio: study.material.poisson_ratio,
            density_kg_m3: study.material.density_kg_m3,
            force_n: study.loads.force_n[0],
        }
    }
}

impl ExternalBenchmarkDefinition {
    pub fn validate(&self) -> Result<(), StructuralMechanicsError> {
        if self.schema_version != EXTERNAL_BENCHMARK_SCHEMA_VERSION {
            return Err(StructuralMechanicsError::ExternalBenchmarkInvalid {
                field: "schema_version",
                value: self.schema_version.to_string(),
            });
        }
        for (field, value) in [
            ("length_m", self.length_m),
            ("width_m", self.width_m),
            ("height_m", self.height_m),
            ("youngs_modulus_pa", self.youngs_modulus_pa),
            ("density_kg_m3", self.density_kg_m3),
            ("force_n", self.force_n),
        ] {
            if !value.is_finite() || value <= 0.0 {
                return Err(StructuralMechanicsError::ExternalBenchmarkInvalid {
                    field,
                    value: value.to_string(),
                });
            }
        }
        if !self.poisson_ratio.is_finite()
            || self.poisson_ratio <= -1.0
            || self.poisson_ratio >= 0.5
        {
            return Err(StructuralMechanicsError::ExternalBenchmarkInvalid {
                field: "poisson_ratio",
                value: self.poisson_ratio.to_string(),
            });
        }
        if self.name.is_empty() || self.name.contains('/') || self.name.contains('\\') {
            return Err(StructuralMechanicsError::ExternalBenchmarkInvalid {
                field: "name",
                value: self.name.clone(),
            });
        }
        Ok(())
    }

    /// Closed-form axial displacement used as the cross-solver reference.
    pub fn expected_displacement_m(&self) -> f64 {
        self.force_n * self.length_m / (self.youngs_modulus_pa * self.width_m * self.height_m)
    }
}

/// One named input file belonging to an external solver deck.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExternalInputFile {
    pub relative_path: String,
    pub contents: String,
}

/// A complete versioned deck/script bundle.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExternalBenchmarkDeck {
    pub solver: ExternalSolverKind,
    pub schema_version: u32,
    pub benchmark_name: String,
    pub primary_file: String,
    pub expected_output_files: Vec<String>,
    pub files: Vec<ExternalInputFile>,
}

impl ExternalBenchmarkDeck {
    /// Write this deck below an explicit caller-owned directory.
    pub fn write_to(&self, directory: &Path) -> Result<Vec<PathBuf>, StructuralMechanicsError> {
        if self.schema_version != EXTERNAL_BENCHMARK_SCHEMA_VERSION {
            return Err(StructuralMechanicsError::ExternalBenchmarkInvalid {
                field: "deck.schema_version",
                value: self.schema_version.to_string(),
            });
        }
        std::fs::create_dir_all(directory).map_err(|error| {
            StructuralMechanicsError::ExternalExecutionFailed {
                solver: self.solver.to_string(),
                status: None,
                message: format!(
                    "cannot create deck directory {}: {error}",
                    directory.display()
                ),
            }
        })?;

        let mut written = Vec::with_capacity(self.files.len());
        for file in &self.files {
            let relative = Path::new(&file.relative_path);
            if relative.is_absolute()
                || relative
                    .components()
                    .any(|component| matches!(component, Component::ParentDir))
            {
                return Err(StructuralMechanicsError::ExternalBenchmarkInvalid {
                    field: "deck.relative_path",
                    value: file.relative_path.clone(),
                });
            }
            let path = directory.join(relative);
            if let Some(parent) = path.parent() {
                std::fs::create_dir_all(parent).map_err(|error| {
                    StructuralMechanicsError::ExternalExecutionFailed {
                        solver: self.solver.to_string(),
                        status: None,
                        message: format!("cannot create deck parent {}: {error}", parent.display()),
                    }
                })?;
            }
            std::fs::write(&path, &file.contents).map_err(|error| {
                StructuralMechanicsError::ExternalExecutionFailed {
                    solver: self.solver.to_string(),
                    status: None,
                    message: format!("cannot write deck file {}: {error}", path.display()),
                }
            })?;
            written.push(path);
        }
        Ok(written)
    }
}

/// Executable discovery result. For FEniCSx this finds Python; the `dolfinx`
/// import is checked only when the script is actually run.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExternalSolverAvailability {
    pub solver: ExternalSolverKind,
    pub executable: Option<PathBuf>,
    pub detail: String,
}

/// Schema version for [`ExternalBenchmarkStatus`].
pub const EXTERNAL_BENCHMARK_STATUS_SCHEMA_VERSION: u32 = 1;

/// Versioned status of the external-solver evidence represented by a report.
///
/// Deck generation and executable discovery are capability evidence only. A
/// parity result is `Passed` only after [`run_external_benchmark`] completes a
/// numeric comparison with sufficient observation provenance.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum ExternalComparisonStatus {
    #[default]
    NotRun,
    Passed,
    Failed,
}

/// Machine-readable external comparison status for benchmark aggregation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ExternalBenchmarkStatus {
    pub schema_version: u32,
    pub status: ExternalComparisonStatus,
    pub adapters_registered: Vec<ExternalSolverKind>,
    pub available_solvers: Vec<ExternalSolverKind>,
    pub parity_passed: bool,
    pub evidence_sufficient: bool,
}

impl Default for ExternalBenchmarkStatus {
    fn default() -> Self {
        Self {
            schema_version: EXTERNAL_BENCHMARK_STATUS_SCHEMA_VERSION,
            status: ExternalComparisonStatus::NotRun,
            adapters_registered: Vec::new(),
            available_solvers: Vec::new(),
            parity_passed: false,
            evidence_sufficient: false,
        }
    }
}

impl ExternalBenchmarkStatus {
    pub fn parity_passed(&self) -> bool {
        self.status == ExternalComparisonStatus::Passed
            && self.parity_passed
            && self.evidence_sufficient
    }
}

pub fn discover_external_solver(solver: ExternalSolverKind) -> ExternalSolverAvailability {
    let configured = std::env::var_os(solver.executable_env_var()).map(PathBuf::from);
    let executable = configured.or_else(|| find_on_path(solver.default_executable()));
    let detail = match &executable {
        Some(_path) => match solver {
            ExternalSolverKind::Fenicsx => {
                "Python found; dolfinx import is checked at execution time".to_string()
            }
            _ => "executable found".to_string(),
        },
        None => format!(
            "not found; set {} or install {}",
            solver.executable_env_var(),
            solver.default_executable()
        ),
    };
    ExternalSolverAvailability {
        solver,
        executable,
        detail,
    }
}

pub fn discover_external_solvers() -> Vec<ExternalSolverAvailability> {
    [
        ExternalSolverKind::Calculix,
        ExternalSolverKind::CodeAster,
        ExternalSolverKind::Fenicsx,
    ]
    .into_iter()
    .map(discover_external_solver)
    .collect()
}

/// Describe external adapters and discovered executables without launching
/// any solver. The returned status is intentionally `NotRun`.
pub fn external_benchmark_status() -> ExternalBenchmarkStatus {
    let adapters_registered = vec![
        ExternalSolverKind::Calculix,
        ExternalSolverKind::CodeAster,
        ExternalSolverKind::Fenicsx,
    ];
    let available_solvers = discover_external_solvers()
        .into_iter()
        .filter_map(|availability| availability.executable.map(|_| availability.solver))
        .collect();
    ExternalBenchmarkStatus {
        schema_version: EXTERNAL_BENCHMARK_STATUS_SCHEMA_VERSION,
        status: ExternalComparisonStatus::NotRun,
        adapters_registered,
        available_solvers,
        parity_passed: false,
        evidence_sufficient: false,
    }
}

/// Human-readable status used by the native benchmark suite.
pub fn external_solver_status_summary() -> String {
    let statuses = discover_external_solvers()
        .into_iter()
        .map(|status| {
            format!(
                "{}={}",
                status.solver,
                status
                    .executable
                    .map(|path| path.display().to_string())
                    .unwrap_or_else(|| status.detail.clone())
            )
        })
        .collect::<Vec<_>>();
    format!(
        "Versioned adapters present; external comparison not run by default. {}",
        statuses.join(", ")
    )
}

/// Render all three external inputs for one benchmark definition.
pub fn render_external_decks(
    definition: &ExternalBenchmarkDefinition,
) -> Result<Vec<ExternalBenchmarkDeck>, StructuralMechanicsError> {
    definition.validate()?;
    [
        ExternalSolverKind::Calculix,
        ExternalSolverKind::CodeAster,
        ExternalSolverKind::Fenicsx,
    ]
    .into_iter()
    .map(|solver| render_external_deck(definition, solver))
    .collect()
}

pub fn render_external_deck(
    definition: &ExternalBenchmarkDefinition,
    solver: ExternalSolverKind,
) -> Result<ExternalBenchmarkDeck, StructuralMechanicsError> {
    definition.validate()?;
    let case = definition.name.clone();
    let (primary_file, expected_output_files, files) = match solver {
        ExternalSolverKind::Calculix => (
            format!("{case}.inp"),
            vec![format!("{case}.dat"), format!("{case}.frd")],
            vec![ExternalInputFile {
                relative_path: format!("{case}.inp"),
                contents: calculix_input(definition),
            }],
        ),
        ExternalSolverKind::CodeAster => (
            format!("{case}.comm"),
            vec![format!("{case}.resu")],
            vec![
                ExternalInputFile {
                    relative_path: format!("{case}.mail"),
                    contents: code_aster_mesh(definition),
                },
                ExternalInputFile {
                    relative_path: format!("{case}.comm"),
                    contents: code_aster_command_file(definition),
                },
            ],
        ),
        ExternalSolverKind::Fenicsx => (
            format!("{case}.py"),
            Vec::new(),
            vec![ExternalInputFile {
                relative_path: format!("{case}.py"),
                contents: fenicsx_script(definition),
            }],
        ),
    };
    Ok(ExternalBenchmarkDeck {
        solver,
        schema_version: EXTERNAL_BENCHMARK_SCHEMA_VERSION,
        benchmark_name: definition.name.clone(),
        primary_file,
        expected_output_files,
        files,
    })
}

/// Comparison result against the benchmark's closed-form displacement.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ExternalDisplacementSource {
    /// Explicit `displacement_m=`, `DISPLACEMENT_M=`, or `U1=` marker.
    CommonMarker,
    /// Numeric table fallback; retained for legacy post-processing but not
    /// sufficient evidence of a solver-specific result by itself.
    SolverTableHeuristic,
    /// Value supplied directly to the comparison API rather than parsed.
    CallerSupplied,
}

/// A parsed external displacement together with its evidence source.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ExternalDisplacementObservation {
    pub displacement_m: f64,
    pub source: ExternalDisplacementSource,
}

/// Comparison result against the benchmark's closed-form displacement.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ExternalComparisonResult {
    pub solver: ExternalSolverKind,
    pub benchmark_name: String,
    pub schema_version: u32,
    pub expected_displacement_m: f64,
    pub measured_displacement_m: f64,
    pub relative_error: f64,
    pub tolerance: f64,
    pub observation_source: ExternalDisplacementSource,
    /// Numeric agreement passed; callers should also require this flag for
    /// solver-parity evidence instead of accepting heuristic table parsing.
    pub evidence_sufficient: bool,
    pub passed: bool,
}

impl ExternalComparisonResult {
    /// True only when numeric agreement came from sufficient solver-output
    /// evidence, not merely from a heuristic table parse.
    pub fn parity_passed(&self) -> bool {
        self.passed && self.evidence_sufficient
    }
}

/// Schema version for external execution-stage provenance.
pub const EXTERNAL_EXECUTION_PROVENANCE_SCHEMA_VERSION: u32 = 1;

/// How the executable used for an external run was selected.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum ExternalExecutableSource {
    CallerSupplied,
    Discovered,
    #[default]
    Unknown,
}

/// Stages that must be completed before an external run can be parity
/// evidence. Deck rendering and numeric comparison are intentionally separate
/// from process execution and output observation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ExternalExecutionStage {
    DeckRendered,
    ExecutableResolved,
    DeckWritten,
    ProcessCompleted,
    OutputObserved,
    NumericCompared,
}

/// Provenance for a successfully completed external solver invocation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ExternalExecutionProvenance {
    pub schema_version: u32,
    pub executable_source: ExternalExecutableSource,
    pub completed_stages: Vec<ExternalExecutionStage>,
    pub observation_source: Option<ExternalDisplacementSource>,
    pub numeric_comparison_passed: bool,
    pub evidence_sufficient: bool,
}

impl Default for ExternalExecutionProvenance {
    fn default() -> Self {
        Self {
            schema_version: 0,
            executable_source: ExternalExecutableSource::Unknown,
            completed_stages: Vec::new(),
            observation_source: None,
            numeric_comparison_passed: false,
            evidence_sufficient: false,
        }
    }
}

impl ExternalExecutionProvenance {
    fn completed(
        executable_source: ExternalExecutableSource,
        observation: ExternalDisplacementObservation,
        comparison: &ExternalComparisonResult,
    ) -> Self {
        Self {
            schema_version: EXTERNAL_EXECUTION_PROVENANCE_SCHEMA_VERSION,
            executable_source,
            completed_stages: vec![
                ExternalExecutionStage::DeckRendered,
                ExternalExecutionStage::ExecutableResolved,
                ExternalExecutionStage::DeckWritten,
                ExternalExecutionStage::ProcessCompleted,
                ExternalExecutionStage::OutputObserved,
                ExternalExecutionStage::NumericCompared,
            ],
            observation_source: Some(observation.source),
            numeric_comparison_passed: comparison.passed,
            evidence_sufficient: comparison.evidence_sufficient,
        }
    }

    pub fn has_stage(&self, stage: ExternalExecutionStage) -> bool {
        self.completed_stages.contains(&stage)
    }

    pub fn parity_passed(&self) -> bool {
        self.has_stage(ExternalExecutionStage::NumericCompared)
            && self.numeric_comparison_passed
            && self.evidence_sufficient
    }
}

pub fn compare_external_displacement(
    solver: ExternalSolverKind,
    definition: &ExternalBenchmarkDefinition,
    measured_displacement_m: f64,
    tolerance: f64,
) -> Result<ExternalComparisonResult, StructuralMechanicsError> {
    compare_external_observation(
        solver,
        definition,
        ExternalDisplacementObservation {
            displacement_m: measured_displacement_m,
            source: ExternalDisplacementSource::CallerSupplied,
        },
        tolerance,
    )
}

/// Compare a parsed displacement while retaining how the value was obtained.
pub fn compare_external_observation(
    solver: ExternalSolverKind,
    definition: &ExternalBenchmarkDefinition,
    observation: ExternalDisplacementObservation,
    tolerance: f64,
) -> Result<ExternalComparisonResult, StructuralMechanicsError> {
    definition.validate()?;
    if !observation.displacement_m.is_finite() || observation.displacement_m <= 0.0 {
        return Err(StructuralMechanicsError::ExternalOutputParseFailed {
            solver: solver.to_string(),
            message: format!("invalid displacement {}", observation.displacement_m),
        });
    }
    if !tolerance.is_finite() || tolerance < 0.0 {
        return Err(StructuralMechanicsError::ExternalBenchmarkInvalid {
            field: "comparison.tolerance",
            value: tolerance.to_string(),
        });
    }
    let expected = definition.expected_displacement_m();
    let relative_error = (observation.displacement_m - expected).abs() / expected.abs().max(1e-30);
    Ok(ExternalComparisonResult {
        solver,
        benchmark_name: definition.name.clone(),
        schema_version: definition.schema_version,
        expected_displacement_m: expected,
        measured_displacement_m: observation.displacement_m,
        relative_error,
        tolerance,
        observation_source: observation.source,
        evidence_sufficient: observation.source == ExternalDisplacementSource::CommonMarker,
        passed: relative_error <= tolerance,
    })
}

/// Parse the common marker emitted by the FEniCSx script, or a simple `U1`
/// table emitted by CalculiX/Code_Aster post-processing.
pub fn parse_external_displacement(
    solver: ExternalSolverKind,
    output: &str,
) -> Result<f64, StructuralMechanicsError> {
    Ok(parse_external_displacement_observation(solver, output)?.displacement_m)
}

/// Parse an external displacement and retain whether it came from an explicit
/// marker or a heuristic solver-table fallback.
pub fn parse_external_displacement_observation(
    solver: ExternalSolverKind,
    output: &str,
) -> Result<ExternalDisplacementObservation, StructuralMechanicsError> {
    let mut explicit_values = Vec::new();
    for marker in ["displacement_m=", "DISPLACEMENT_M=", "U1="] {
        for value in explicit_marker_values(output, marker).map_err(|message| {
            StructuralMechanicsError::ExternalOutputParseFailed {
                solver: solver.to_string(),
                message,
            }
        })? {
            explicit_values.push(value);
        }
    }
    if let Some(&first) = explicit_values.first() {
        if explicit_values
            .iter()
            .copied()
            .skip(1)
            .any(|value| !equivalent_marker_value(first, value))
        {
            return Err(StructuralMechanicsError::ExternalOutputParseFailed {
                solver: solver.to_string(),
                message: "conflicting explicit displacement markers in solver output".to_string(),
            });
        }
        return Ok(ExternalDisplacementObservation {
            displacement_m: first,
            source: ExternalDisplacementSource::CommonMarker,
        });
    }

    if solver == ExternalSolverKind::Calculix || solver == ExternalSolverKind::CodeAster {
        for line in output.lines() {
            let trimmed = line.trim_start();
            if trimmed
                .chars()
                .next()
                .is_some_and(|character| character.is_ascii_digit())
            {
                let numbers = trimmed
                    .split(|character: char| character.is_ascii_whitespace() || character == ',')
                    .filter_map(|token| token.parse::<f64>().ok())
                    .collect::<Vec<_>>();
                if numbers.len() >= 2 {
                    let value = numbers[1].abs();
                    if value.is_finite() && value > 0.0 {
                        return Ok(ExternalDisplacementObservation {
                            displacement_m: value,
                            source: ExternalDisplacementSource::SolverTableHeuristic,
                        });
                    }
                }
            }
        }
    }

    Err(StructuralMechanicsError::ExternalOutputParseFailed {
        solver: solver.to_string(),
        message: "no positive displacement_m=, U1=, or tabular displacement value found"
            .to_string(),
    })
}

/// Parse only evidence that is sufficient for an external-solver parity claim.
///
/// Legacy table parsing remains available through
/// [`parse_external_displacement_observation`], but callers promoting an
/// external result must use this checked boundary so a positional numeric table
/// cannot accidentally become solver-parity evidence.
pub fn parse_external_displacement_for_parity(
    solver: ExternalSolverKind,
    output: &str,
) -> Result<f64, StructuralMechanicsError> {
    let observation = parse_external_displacement_observation(solver, output)?;
    if observation.source != ExternalDisplacementSource::CommonMarker {
        return Err(StructuralMechanicsError::ExternalOutputParseFailed {
            solver: solver.to_string(),
            message: "external parity evidence requires one explicit displacement marker"
                .to_string(),
        });
    }
    Ok(observation.displacement_m)
}

fn explicit_marker_values(output: &str, marker: &str) -> Result<Vec<f64>, String> {
    let mut values = Vec::new();
    let mut offset = 0;
    while let Some(relative_start) = output[offset..].find(marker) {
        let start = offset + relative_start;
        offset = start + marker.len();
        if output[..start]
            .chars()
            .next_back()
            .is_some_and(|character| character.is_ascii_alphanumeric() || character == '_')
        {
            continue;
        }
        let token = output[offset..]
            .split_whitespace()
            .next()
            .ok_or_else(|| format!("explicit {marker:?} marker has no value"))?
            .trim_matches(|character: char| ",;".contains(character));
        let value = token
            .parse::<f64>()
            .map_err(|_| format!("explicit {marker:?} marker has invalid value {token:?}"))?;
        if !value.is_finite() || value <= 0.0 {
            return Err(format!(
                "explicit {marker:?} marker must contain a finite positive displacement"
            ));
        }
        values.push(value);
    }
    Ok(values)
}

fn equivalent_marker_value(left: f64, right: f64) -> bool {
    (left - right).abs() <= 1e-12 * left.abs().max(right.abs()).max(1.0)
}

/// Options for an explicit external execution.
#[derive(Debug, Clone, PartialEq)]
pub struct ExternalRunOptions {
    pub workdir: PathBuf,
    pub executable: Option<PathBuf>,
    pub tolerance: f64,
    /// Hard wall-clock limit for one external solver process.
    pub timeout: Duration,
}

impl ExternalRunOptions {
    pub fn new(workdir: impl Into<PathBuf>) -> Self {
        Self {
            workdir: workdir.into(),
            executable: None,
            tolerance: 1e-2,
            timeout: Duration::from_secs(120),
        }
    }

    pub fn tolerance_f64(&self) -> Result<f64, StructuralMechanicsError> {
        if !self.tolerance.is_finite() || self.tolerance < 0.0 {
            Err(StructuralMechanicsError::ExternalBenchmarkInvalid {
                field: "run.tolerance",
                value: self.tolerance.to_string(),
            })
        } else {
            Ok(self.tolerance)
        }
    }

    pub fn timeout(&self) -> Result<Duration, StructuralMechanicsError> {
        if self.timeout.is_zero() {
            Err(StructuralMechanicsError::ExternalBenchmarkInvalid {
                field: "run.timeout",
                value: "zero duration".to_string(),
            })
        } else {
            Ok(self.timeout)
        }
    }
}

/// Output and comparison from one explicitly requested external run.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ExternalRunResult {
    pub solver: ExternalSolverKind,
    pub benchmark_name: String,
    pub executable: PathBuf,
    pub exit_code: Option<i32>,
    pub stdout: String,
    pub stderr: String,
    pub measured_displacement_m: f64,
    pub observation_source: ExternalDisplacementSource,
    pub evidence_sufficient: bool,
    pub comparison: ExternalComparisonResult,
    #[serde(default)]
    pub provenance: ExternalExecutionProvenance,
}

impl ExternalRunResult {
    pub fn parity_passed(&self) -> bool {
        self.comparison.parity_passed() && self.provenance.parity_passed()
    }

    /// Convert one completed run into the same machine-readable status used
    /// by benchmark aggregation.
    pub fn benchmark_status(&self) -> ExternalBenchmarkStatus {
        ExternalBenchmarkStatus {
            schema_version: EXTERNAL_BENCHMARK_STATUS_SCHEMA_VERSION,
            status: if self.parity_passed() {
                ExternalComparisonStatus::Passed
            } else {
                ExternalComparisonStatus::Failed
            },
            adapters_registered: vec![self.solver],
            available_solvers: vec![self.solver],
            parity_passed: self.parity_passed(),
            evidence_sufficient: self.provenance.evidence_sufficient,
        }
    }
}

pub fn run_external_benchmark(
    definition: &ExternalBenchmarkDefinition,
    solver: ExternalSolverKind,
    options: &ExternalRunOptions,
) -> Result<ExternalRunResult, StructuralMechanicsError> {
    let deck = render_external_deck(definition, solver)?;
    let (executable, executable_source) = if let Some(executable) = options.executable.clone() {
        (executable, ExternalExecutableSource::CallerSupplied)
    } else if let Some(executable) = discover_external_solver(solver).executable {
        (executable, ExternalExecutableSource::Discovered)
    } else {
        return Err(StructuralMechanicsError::ExternalSolverUnavailable {
            solver: solver.to_string(),
            executable: solver.default_executable().to_string(),
        });
    };
    deck.write_to(&options.workdir)?;

    let mut command = Command::new(&executable);
    command.current_dir(&options.workdir);
    match solver {
        ExternalSolverKind::Calculix => {
            let stem = Path::new(&deck.primary_file)
                .file_stem()
                .and_then(|value| value.to_str())
                .ok_or_else(|| StructuralMechanicsError::ExternalBenchmarkInvalid {
                    field: "deck.primary_file",
                    value: deck.primary_file.clone(),
                })?;
            command.arg(stem);
        }
        ExternalSolverKind::CodeAster => {
            command.arg(&deck.primary_file);
        }
        ExternalSolverKind::Fenicsx => {
            command.arg(options.workdir.join(&deck.primary_file));
        }
    }

    command.stdout(Stdio::piped()).stderr(Stdio::piped());
    let timeout = options.timeout()?;
    let mut child =
        command
            .spawn()
            .map_err(|error| StructuralMechanicsError::ExternalExecutionFailed {
                solver: solver.to_string(),
                status: None,
                message: format!("failed to launch {}: {error}", executable.display()),
            })?;
    let started = Instant::now();
    let output = loop {
        match child.try_wait().map_err(|error| {
            StructuralMechanicsError::ExternalExecutionFailed {
                solver: solver.to_string(),
                status: None,
                message: format!("failed while waiting for {}: {error}", executable.display()),
            }
        })? {
            Some(_) => {
                break child.wait_with_output().map_err(|error| {
                    StructuralMechanicsError::ExternalExecutionFailed {
                        solver: solver.to_string(),
                        status: None,
                        message: format!(
                            "failed to collect output from {}: {error}",
                            executable.display()
                        ),
                    }
                })?;
            }
            None if started.elapsed() >= timeout => {
                let _ = child.kill();
                let output = child.wait_with_output().map_err(|error| {
                    StructuralMechanicsError::ExternalExecutionFailed {
                        solver: solver.to_string(),
                        status: None,
                        message: format!(
                            "timed out after {:?}; failed to collect output from {}: {error}",
                            timeout,
                            executable.display()
                        ),
                    }
                })?;
                let stdout = String::from_utf8_lossy(&output.stdout);
                let stderr = String::from_utf8_lossy(&output.stderr);
                return Err(StructuralMechanicsError::ExternalExecutionFailed {
                    solver: solver.to_string(),
                    status: output.status.code(),
                    message: format!(
                        "timed out after {:?}\nstdout={stdout}\nstderr={stderr}",
                        timeout
                    ),
                });
            }
            None => {
                thread::sleep(Duration::from_millis(10));
                continue;
            }
        }
    };
    let stdout = String::from_utf8_lossy(&output.stdout).into_owned();
    let stderr = String::from_utf8_lossy(&output.stderr).into_owned();
    if !output.status.success() {
        return Err(StructuralMechanicsError::ExternalExecutionFailed {
            solver: solver.to_string(),
            status: output.status.code(),
            message: format!("stdout={stdout}\nstderr={stderr}"),
        });
    }

    let mut result_text = format!("{stdout}\n{stderr}");
    for filename in &deck.expected_output_files {
        let path = options.workdir.join(filename);
        if let Ok(contents) = std::fs::read_to_string(path) {
            result_text.push('\n');
            result_text.push_str(&contents);
        }
    }
    let observation = parse_external_displacement_observation(solver, &result_text)?;
    let comparison =
        compare_external_observation(solver, definition, observation, options.tolerance_f64()?)?;
    let provenance =
        ExternalExecutionProvenance::completed(executable_source, observation, &comparison);
    Ok(ExternalRunResult {
        solver,
        benchmark_name: definition.name.clone(),
        executable,
        exit_code: output.status.code(),
        stdout,
        stderr,
        measured_displacement_m: observation.displacement_m,
        observation_source: observation.source,
        evidence_sufficient: comparison.evidence_sufficient,
        comparison,
        provenance,
    })
}

fn find_on_path(executable: &str) -> Option<PathBuf> {
    let candidate = Path::new(executable);
    if candidate.components().count() > 1 {
        return candidate.is_file().then(|| candidate.to_path_buf());
    }
    let path = std::env::var_os("PATH")?;
    for directory in std::env::split_paths(&path) {
        let direct = directory.join(executable);
        if direct.is_file() {
            return Some(direct);
        }
        #[cfg(windows)]
        for extension in [".exe", ".cmd", ".bat"] {
            let candidate = directory.join(format!("{executable}{extension}"));
            if candidate.is_file() {
                return Some(candidate);
            }
        }
    }
    None
}

fn calculix_input(definition: &ExternalBenchmarkDefinition) -> String {
    let quarter_force = definition.force_n / 4.0;
    format!(
        r#"*HEADING
** SONIUM external benchmark schema {schema}
** {name}
*NODE
1, 0.0, 0.0, 0.0
2, {length:.17e}, 0.0, 0.0
3, {length:.17e}, {width:.17e}, 0.0
4, 0.0, {width:.17e}, 0.0
5, 0.0, 0.0, {height:.17e}
6, {length:.17e}, 0.0, {height:.17e}
7, {length:.17e}, {width:.17e}, {height:.17e}
8, 0.0, {width:.17e}, {height:.17e}
*ELEMENT, TYPE=C3D8, ELSET=SOLID
1, 1, 2, 3, 4, 5, 6, 7, 8
*NSET, NSET=FIXED
1, 4, 5, 8
*NSET, NSET=TIP
2, 3, 6, 7
*MATERIAL, NAME=SONIUM_MAT
*ELASTIC
{youngs:.17e}, {poisson:.17e}
*DENSITY
{density:.17e}
*SOLID SECTION, ELSET=SOLID, MATERIAL=SONIUM_MAT
,
*STEP
*STATIC
1.0, 1.0
*BOUNDARY
FIXED, 1, 3
*CLOAD
2, 1, {quarter_force:.17e}
3, 1, {quarter_force:.17e}
6, 1, {quarter_force:.17e}
7, 1, {quarter_force:.17e}
*NODE PRINT, NSET=TIP
U
*EL PRINT, ELSET=SOLID
S
*END STEP
"#,
        schema = definition.schema_version,
        name = definition.name,
        length = definition.length_m,
        width = definition.width_m,
        height = definition.height_m,
        youngs = definition.youngs_modulus_pa,
        poisson = definition.poisson_ratio,
        density = definition.density_kg_m3,
        quarter_force = quarter_force,
    )
}

fn code_aster_mesh(definition: &ExternalBenchmarkDefinition) -> String {
    format!(
        r#"TITRE
SONIUM external benchmark schema {schema}
COOR_3D
1 0.0 0.0 0.0
2 {length:.17e} 0.0 0.0
3 {length:.17e} {width:.17e} 0.0
4 0.0 {width:.17e} 0.0
5 0.0 0.0 {height:.17e}
6 {length:.17e} 0.0 {height:.17e}
7 {length:.17e} {width:.17e} {height:.17e}
8 0.0 {width:.17e} {height:.17e}
FINSF
HEXA8
1 1 2 3 4 5 6 7 8
FINSF
GROUP_NO
FIXED_NODES
1 4 5 8
TIP_NODES
2 3 6 7
FINSF
"#,
        schema = definition.schema_version,
        length = definition.length_m,
        width = definition.width_m,
        height = definition.height_m,
    )
}

fn code_aster_command_file(definition: &ExternalBenchmarkDefinition) -> String {
    let quarter_force = definition.force_n / 4.0;
    format!(
        r#"# SONIUM external benchmark schema {schema}
DEBUT(PAR_LOT='NON');
mesh = LIRE_MAILLAGE(FORMAT='ASTER', UNITE=20);
model = AFFE_MODELE(MAILLAGE=mesh, AFFE=_F(TOUT='OUI', PHENOMENE='MECANIQUE', MODELISATION='3D'));
material = DEFI_MATERIAU(ELAS=_F(E={youngs:.17e}, NU={poisson:.17e}, RHO={density:.17e}));
field = AFFE_MATERIAU(MAILLAGE=mesh, AFFE=_F(TOUT='OUI', MATER=material));
load = AFFE_CHAR_MECA(
    MODELE=model,
    DDL_IMPO=_F(GROUP_NO='FIXED_NODES', DX=0.0, DY=0.0, DZ=0.0),
    FORCE_NODALE=_F(GROUP_NO='TIP_NODES', FX={quarter_force:.17e}),
);
result = MECA_STATIQUE(MODELE=model, CHAM_MATER=field, EXCIT=_F(CHARGE=load));
IMPR_RESU(RESU=_F(RESULTAT=result, NOM_CHAM='DEPL'), FORMAT='RESULTAT', UNITE=80);
FIN();
"#,
        schema = definition.schema_version,
        youngs = definition.youngs_modulus_pa,
        poisson = definition.poisson_ratio,
        density = definition.density_kg_m3,
        quarter_force = quarter_force,
    )
}

fn fenicsx_script(definition: &ExternalBenchmarkDefinition) -> String {
    let traction = definition.force_n / (definition.width_m * definition.height_m);
    format!(
        r#"""SONIUM external benchmark schema {schema}: {name}"""
from mpi4py import MPI
import numpy as np
import ufl
from dolfinx import fem, mesh
from dolfinx.fem.petsc import LinearProblem

L = {length:.17e}
W = {width:.17e}
H = {height:.17e}
E = {youngs:.17e}
NU = {poisson:.17e}
TRACTION = {traction:.17e}
domain = mesh.create_box(
    MPI.COMM_WORLD,
    [np.array([0.0, 0.0, 0.0]), np.array([L, W, H])],
    [2, 1, 1],
    cell_type=mesh.CellType.tetrahedron,
)
V = fem.functionspace(domain, ("Lagrange", 1, (domain.geometry.dim,)))
u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)
mu = E / (2.0 * (1.0 + NU))
lam = E * NU / ((1.0 + NU) * (1.0 - 2.0 * NU))
def eps(w):
    return ufl.sym(ufl.grad(w))
def sigma(w):
    return 2.0 * mu * eps(w) + lam * ufl.tr(eps(w)) * ufl.Identity(3)

left = mesh.locate_entities_boundary(domain, 2, lambda x: np.isclose(x[0], 0.0))
right = mesh.locate_entities_boundary(domain, 2, lambda x: np.isclose(x[0], L))
left_dofs = fem.locate_dofs_topological(V, 2, left)
zero = fem.Constant(domain, np.zeros(3, dtype=np.float64))
bc = fem.dirichletbc(zero, left_dofs, V)
fdim = domain.topology.dim - 1
tags = mesh.meshtags(domain, fdim, np.concatenate([left, right]),
                     np.concatenate([np.full(len(left), 1), np.full(len(right), 2)]))
ds = ufl.Measure("ds", domain=domain, subdomain_data=tags)
traction = fem.Constant(domain, np.array([TRACTION, 0.0, 0.0], dtype=np.float64))
a = ufl.inner(sigma(u), eps(v)) * ufl.dx
Lform = ufl.inner(traction, v) * ds(2)
problem = LinearProblem(a, Lform, bcs=[bc], petsc_options={{"ksp_type": "preonly", "pc_type": "lu"}})
solution = problem.solve()
solution.x.scatter_forward()
tip_dofs = fem.locate_dofs_topological(V, 2, right)
tip_value = float(np.max(np.abs(solution.x.array[tip_dofs])))
if domain.comm.rank == 0:
    print(f"SONIUM_EXTERNAL_RESULT displacement_m={{tip_value:.17e}}")
"#,
        schema = definition.schema_version,
        name = definition.name,
        length = definition.length_m,
        width = definition.width_m,
        height = definition.height_m,
        youngs = definition.youngs_modulus_pa,
        poisson = definition.poisson_ratio,
        traction = traction,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_definition_has_a_versioned_closed_form_reference() {
        let definition = ExternalBenchmarkDefinition::default();
        definition.validate().unwrap();
        assert_eq!(definition.schema_version, EXTERNAL_BENCHMARK_SCHEMA_VERSION);
        assert!((definition.expected_displacement_m() - 1.0714285714285714e-7).abs() < 1e-20);
    }

    #[test]
    fn all_decks_are_versioned_and_contain_the_same_benchmark() {
        let definition = ExternalBenchmarkDefinition::default();
        let decks = render_external_decks(&definition).unwrap();
        assert_eq!(decks.len(), 3);
        for deck in decks {
            assert_eq!(deck.schema_version, EXTERNAL_BENCHMARK_SCHEMA_VERSION);
            assert_eq!(deck.benchmark_name, definition.name);
            assert!(deck.files.iter().all(|file| !file.contents.is_empty()));
            assert!(
                deck.files
                    .iter()
                    .any(|file| { file.contents.contains("SONIUM external benchmark schema 1") })
            );
        }
    }

    #[test]
    fn parser_and_comparison_share_the_reference_contract() {
        let definition = ExternalBenchmarkDefinition::default();
        for solver in [
            ExternalSolverKind::Calculix,
            ExternalSolverKind::CodeAster,
            ExternalSolverKind::Fenicsx,
        ] {
            let output = format!(
                "SONIUM_EXTERNAL_RESULT displacement_m={:.17e}",
                definition.expected_displacement_m()
            );
            let observation = parse_external_displacement_observation(solver, &output).unwrap();
            assert_eq!(observation.source, ExternalDisplacementSource::CommonMarker);
            let measured = parse_external_displacement(solver, &output).unwrap();
            let result =
                compare_external_displacement(solver, &definition, measured, 1e-12).unwrap();
            assert!(result.passed);
            assert!(!result.evidence_sufficient);
            let result =
                compare_external_observation(solver, &definition, observation, 1e-12).unwrap();
            assert!(result.passed);
            assert!(result.evidence_sufficient);
        }
    }

    #[test]
    fn explicit_markers_are_unambiguous_and_required_for_parity() {
        let solver = ExternalSolverKind::Calculix;
        let expected = ExternalBenchmarkDefinition::default().expected_displacement_m();
        let same_marker = format!("displacement_m={expected:.17e}\nU1={expected:.17e};");
        assert_eq!(
            parse_external_displacement_for_parity(solver, &same_marker).unwrap(),
            expected
        );

        let conflicting = format!("displacement_m={expected:.17e}\nU1={:.17e}", expected * 1.1);
        assert!(matches!(
            parse_external_displacement_observation(solver, &conflicting),
            Err(StructuralMechanicsError::ExternalOutputParseFailed { message, .. })
                if message.contains("conflicting explicit displacement markers")
        ));

        assert!(matches!(
            parse_external_displacement_for_parity(solver, &format!("1 {expected:.17e} 0.0")),
            Err(StructuralMechanicsError::ExternalOutputParseFailed { message, .. })
                if message.contains("requires one explicit displacement marker")
        ));
        assert!(matches!(
            parse_external_displacement_observation(solver, "nondisplacement_m=1.0"),
            Err(StructuralMechanicsError::ExternalOutputParseFailed { .. })
        ));
    }

    #[test]
    fn parity_provenance_exposes_every_required_execution_stage() {
        let definition = ExternalBenchmarkDefinition::default();
        let observation = ExternalDisplacementObservation {
            displacement_m: definition.expected_displacement_m(),
            source: ExternalDisplacementSource::CommonMarker,
        };
        let comparison = compare_external_observation(
            ExternalSolverKind::Fenicsx,
            &definition,
            observation,
            1e-12,
        )
        .unwrap();
        assert!(comparison.parity_passed());

        let provenance = ExternalExecutionProvenance::completed(
            ExternalExecutableSource::CallerSupplied,
            observation,
            &comparison,
        );
        assert_eq!(
            provenance.schema_version,
            EXTERNAL_EXECUTION_PROVENANCE_SCHEMA_VERSION
        );
        for stage in [
            ExternalExecutionStage::DeckRendered,
            ExternalExecutionStage::ExecutableResolved,
            ExternalExecutionStage::DeckWritten,
            ExternalExecutionStage::ProcessCompleted,
            ExternalExecutionStage::OutputObserved,
            ExternalExecutionStage::NumericCompared,
        ] {
            assert!(provenance.has_stage(stage));
        }
        assert!(provenance.parity_passed());

        let run_result = ExternalRunResult {
            solver: ExternalSolverKind::Fenicsx,
            benchmark_name: definition.name.clone(),
            executable: PathBuf::from("fenicsx"),
            exit_code: Some(0),
            stdout: String::new(),
            stderr: String::new(),
            measured_displacement_m: observation.displacement_m,
            observation_source: observation.source,
            evidence_sufficient: true,
            comparison: comparison.clone(),
            provenance: provenance.clone(),
        };
        let status = run_result.benchmark_status();
        assert_eq!(status.status, ExternalComparisonStatus::Passed);
        assert!(status.parity_passed());

        let heuristic = ExternalDisplacementObservation {
            displacement_m: definition.expected_displacement_m(),
            source: ExternalDisplacementSource::SolverTableHeuristic,
        };
        let heuristic_comparison = compare_external_observation(
            ExternalSolverKind::Calculix,
            &definition,
            heuristic,
            1e-12,
        )
        .unwrap();
        let heuristic_provenance = ExternalExecutionProvenance::completed(
            ExternalExecutableSource::Discovered,
            heuristic,
            &heuristic_comparison,
        );
        assert!(!heuristic_comparison.parity_passed());
        assert!(!heuristic_provenance.parity_passed());
    }

    #[test]
    fn table_fallback_is_reported_as_insufficient_parity_evidence() {
        let definition = ExternalBenchmarkDefinition::default();
        let observation = parse_external_displacement_observation(
            ExternalSolverKind::Calculix,
            &format!("1 {:.17e} 0.0", definition.expected_displacement_m()),
        )
        .unwrap();
        assert_eq!(
            observation.source,
            ExternalDisplacementSource::SolverTableHeuristic
        );
        let comparison = compare_external_observation(
            ExternalSolverKind::Calculix,
            &definition,
            observation,
            1e-12,
        )
        .unwrap();
        assert!(comparison.passed);
        assert!(!comparison.evidence_sufficient);
    }

    #[test]
    fn external_solver_integration_is_opt_in_and_capability_gated() {
        if std::env::var_os("SONIUM_RUN_EXTERNAL_INTEGRATION").is_none() {
            return;
        }

        let definition = ExternalBenchmarkDefinition::default();
        let mut attempted = false;
        for availability in discover_external_solvers() {
            let Some(executable) = availability.executable else {
                continue;
            };
            attempted = true;
            let directory = std::env::temp_dir().join(format!(
                "sonium-external-integration-{}",
                availability.solver
            ));
            let _ = std::fs::remove_dir_all(&directory);
            std::fs::create_dir_all(&directory).unwrap();
            let mut options = ExternalRunOptions::new(&directory);
            options.executable = Some(executable);
            options.tolerance = 1e-5;
            let result = run_external_benchmark(&definition, availability.solver, &options)
                .unwrap_or_else(|error| {
                    panic!("{} integration failed: {error}", availability.solver)
                });
            assert!(result.parity_passed());
            assert!(result.provenance.parity_passed());
            let _ = std::fs::remove_dir_all(&directory);
        }

        // A capability-gated test is allowed to have no executable available;
        // the normal test suite must never launch an external solver.
        assert!(!attempted || std::env::var_os("SONIUM_RUN_EXTERNAL_INTEGRATION").is_some());
    }

    #[test]
    #[cfg(unix)]
    fn caller_supplied_fixture_executes_and_records_all_stages() {
        use std::os::unix::fs::PermissionsExt;

        let definition = ExternalBenchmarkDefinition::default();
        let directory = std::env::temp_dir().join(format!(
            "sonium-external-fixture-test-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&directory);
        std::fs::create_dir_all(&directory).unwrap();
        let executable = directory.join("fixture-solver.sh");
        std::fs::write(
            &executable,
            format!(
                "#!/bin/sh\nprintf 'displacement_m={:.17e}\\n'\n",
                definition.expected_displacement_m()
            ),
        )
        .unwrap();
        let mut permissions = std::fs::metadata(&executable).unwrap().permissions();
        permissions.set_mode(0o755);
        std::fs::set_permissions(&executable, permissions).unwrap();

        for solver in [
            ExternalSolverKind::Calculix,
            ExternalSolverKind::CodeAster,
            ExternalSolverKind::Fenicsx,
        ] {
            let workdir = directory.join(solver.to_string());
            let mut options = ExternalRunOptions::new(&workdir);
            options.executable = Some(executable.clone());
            options.tolerance = 1.0e-12;
            let result = run_external_benchmark(&definition, solver, &options).unwrap();
            assert!(result.parity_passed());
            assert!(result.provenance.parity_passed());
            assert_eq!(result.provenance.completed_stages.len(), 6);
            assert_eq!(
                result.provenance.observation_source,
                Some(ExternalDisplacementSource::CommonMarker)
            );
        }
        let _ = std::fs::remove_dir_all(&directory);
    }

    #[test]
    #[cfg(unix)]
    fn external_execution_timeout_kills_hanging_process_and_reports_typed_failure() {
        use std::os::unix::fs::PermissionsExt;
        use std::time::Instant;

        let definition = ExternalBenchmarkDefinition::default();
        let directory = std::env::temp_dir().join(format!(
            "sonium-external-timeout-test-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&directory);
        std::fs::create_dir_all(&directory).unwrap();
        let executable = directory.join("hanging-solver.sh");
        std::fs::write(&executable, "#!/bin/sh\nexec sleep 2\n").unwrap();
        let mut permissions = std::fs::metadata(&executable).unwrap().permissions();
        permissions.set_mode(0o755);
        std::fs::set_permissions(&executable, permissions).unwrap();

        let mut options = ExternalRunOptions::new(directory.join("run"));
        options.executable = Some(executable);
        options.timeout = Duration::from_millis(50);
        let started = Instant::now();
        let error = run_external_benchmark(&definition, ExternalSolverKind::Calculix, &options)
            .expect_err("hanging external process must time out");
        assert!(started.elapsed() < Duration::from_secs(1));
        match error {
            StructuralMechanicsError::ExternalExecutionFailed { message, .. } => {
                assert!(message.contains("timed out"), "{message}");
            }
            other => panic!("unexpected timeout error: {other:?}"),
        }

        options.timeout = Duration::ZERO;
        assert!(matches!(
            run_external_benchmark(&definition, ExternalSolverKind::Calculix, &options),
            Err(StructuralMechanicsError::ExternalBenchmarkInvalid {
                field: "run.timeout",
                ..
            })
        ));
        let _ = std::fs::remove_dir_all(&directory);
    }

    #[test]
    fn deck_writer_rejects_path_traversal() {
        let mut deck = render_external_deck(
            &ExternalBenchmarkDefinition::default(),
            ExternalSolverKind::Calculix,
        )
        .unwrap();
        deck.files[0].relative_path = "../escape.inp".to_string();
        let directory =
            std::env::temp_dir().join(format!("sonium-external-deck-test-{}", std::process::id()));
        let error = deck.write_to(&directory).unwrap_err();
        assert!(matches!(
            error,
            StructuralMechanicsError::ExternalBenchmarkInvalid {
                field: "deck.relative_path",
                ..
            }
        ));
    }

    #[test]
    fn availability_summary_is_explicit_about_default_non_execution() {
        let summary = external_solver_status_summary();
        assert!(summary.contains("Versioned adapters present"));
        assert!(summary.contains("not run by default"));

        let status = external_benchmark_status();
        assert_eq!(
            status.schema_version,
            EXTERNAL_BENCHMARK_STATUS_SCHEMA_VERSION
        );
        assert_eq!(status.status, ExternalComparisonStatus::NotRun);
        assert_eq!(status.adapters_registered.len(), 3);
        assert!(!status.parity_passed());
    }
}
