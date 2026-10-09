//! Deterministic problem exchange for independently solved room references.
//!
//! This is an input-side contract for a later FreeFEM/FEniCS (or other
//! external) solve.  It intentionally exports only isotropic source models
//! whose frequency-dependent crossover amplitudes can be represented exactly
//! by the manifest.  Directional and complex directivity must use a richer
//! contract rather than being silently downgraded.

use crate::gallery::{GalleryReference, load_reference_response_json};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use sonium_xem::{BoundaryConfig, Directivity, Point3D, RoomGeometry, RoomSimulation};
use std::collections::HashSet;
use std::ffi::OsString;
use std::io::Read;
use std::path::Path;
use std::process::{Command, Stdio};
use std::thread;
use std::time::{Duration, Instant};

/// Versioned external-reference problem manifest.
pub const EXTERNAL_REFERENCE_PROBLEM_SCHEMA_V1: &str = "roomsim.external-reference-problem.v1";
/// Versioned external-reference manifest carrying directional source data.
pub const EXTERNAL_REFERENCE_PROBLEM_SCHEMA_V2: &str = "roomsim.external-reference-problem.v2";
/// Transfer quantity represented by the exported problem.
pub const EXTERNAL_REFERENCE_CONTRACT_V1: &str = "complex-pressure-transfer.v1";

/// Default wall-clock limit for one external reference driver invocation.
pub const DEFAULT_EXTERNAL_REFERENCE_TIMEOUT: Duration = Duration::from_secs(300);

/// Source record with exact declared-frequency crossover amplitudes.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExternalReferenceSource {
    pub name: String,
    pub position: Point3D,
    pub orientation: Point3D,
    pub amplitudes: Vec<f64>,
    /// Directional source pattern. Omitted by v1 isotropic manifests.
    #[serde(
        default,
        skip_serializing_if = "ExternalReferenceDirectivity::is_isotropic"
    )]
    pub directivity: ExternalReferenceDirectivity,
}

/// Directivity data exchanged with an independent reference solver.
///
/// The external driver is responsible for evaluating this source-local
/// pattern. Keeping the samples in the manifest prevents a directional source
/// from being silently downgraded to an isotropic amplitude-only source.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
#[derive(Default)]
pub enum ExternalReferenceDirectivity {
    #[default]
    Isotropic,
    Classical {
        horizontal_beamwidth_deg: f64,
        vertical_beamwidth_deg: f64,
    },
    Grid {
        horizontal_angles_deg: Vec<f64>,
        vertical_angles_deg: Vec<f64>,
        magnitude: Vec<Vec<f64>>,
    },
    ComplexGrid {
        horizontal_angles_deg: Vec<f64>,
        vertical_angles_deg: Vec<f64>,
        frequencies_hz: Vec<f64>,
        patterns: Vec<Vec<Vec<ExternalReferenceComplexSample>>>,
    },
}

impl ExternalReferenceDirectivity {
    fn is_isotropic(&self) -> bool {
        matches!(self, Self::Isotropic)
    }

    fn from_source(source: &sonium_xem::Source) -> Self {
        if source.is_isotropic() {
            return Self::Isotropic;
        }
        match &source.directivity {
            Directivity::Grid(grid) => Self::Grid {
                horizontal_angles_deg: grid.horizontal_angles().to_vec(),
                vertical_angles_deg: grid.vertical_angles().to_vec(),
                magnitude: (0..grid.magnitude().nrows())
                    .map(|row| grid.magnitude().row(row).to_vec())
                    .collect(),
            },
            Directivity::Classical { h_angle, v_angle } => Self::Classical {
                horizontal_beamwidth_deg: *h_angle,
                vertical_beamwidth_deg: *v_angle,
            },
            Directivity::ComplexGrid(grid) => Self::ComplexGrid {
                horizontal_angles_deg: grid.horizontal_angles().to_vec(),
                vertical_angles_deg: grid.vertical_angles().to_vec(),
                frequencies_hz: grid.frequencies().to_vec(),
                patterns: grid
                    .patterns()
                    .iter()
                    .map(|pattern| {
                        (0..pattern.nrows())
                            .map(|row| {
                                pattern
                                    .row(row)
                                    .iter()
                                    .map(|value| ExternalReferenceComplexSample {
                                        real: value.re,
                                        imag: value.im,
                                    })
                                    .collect()
                            })
                            .collect()
                    })
                    .collect(),
            },
        }
    }

    fn validate(&self) -> Result<(), String> {
        match self {
            Self::Isotropic => Ok(()),
            Self::Classical {
                horizontal_beamwidth_deg,
                vertical_beamwidth_deg,
            } => {
                for (name, value) in [
                    ("horizontal beamwidth", horizontal_beamwidth_deg),
                    ("vertical beamwidth", vertical_beamwidth_deg),
                ] {
                    if !value.is_finite() || !(0.0..=360.0).contains(value) || *value == 0.0 {
                        return Err(format!(
                            "external-reference {name} must be finite in (0, 360] degrees"
                        ));
                    }
                }
                Ok(())
            }
            Self::Grid {
                horizontal_angles_deg,
                vertical_angles_deg,
                magnitude,
            } => {
                validate_directivity_axes(horizontal_angles_deg, vertical_angles_deg)?;
                if magnitude.len() != vertical_angles_deg.len()
                    || magnitude
                        .iter()
                        .any(|row| row.len() != horizontal_angles_deg.len())
                    || magnitude.iter().flatten().any(|value| !value.is_finite())
                {
                    return Err(
                        "external-reference directivity magnitude shape or values are invalid"
                            .into(),
                    );
                }
                Ok(())
            }
            Self::ComplexGrid {
                horizontal_angles_deg,
                vertical_angles_deg,
                frequencies_hz,
                patterns,
            } => {
                validate_directivity_axes(horizontal_angles_deg, vertical_angles_deg)?;
                if frequencies_hz.is_empty()
                    || frequencies_hz
                        .iter()
                        .any(|frequency| !frequency.is_finite() || *frequency <= 0.0)
                    || frequencies_hz.windows(2).any(|pair| pair[1] <= pair[0])
                {
                    return Err(
                        "external-reference directivity frequencies must be finite, positive, and strictly increasing"
                            .into(),
                    );
                }
                if patterns.len() != frequencies_hz.len()
                    || patterns.iter().any(|pattern| {
                        pattern.len() != vertical_angles_deg.len()
                            || pattern
                                .iter()
                                .any(|row| row.len() != horizontal_angles_deg.len())
                    })
                    || patterns
                        .iter()
                        .flatten()
                        .flatten()
                        .any(|value| !value.real.is_finite() || !value.imag.is_finite())
                {
                    return Err(
                        "external-reference complex directivity shape or values are invalid".into(),
                    );
                }
                Ok(())
            }
        }
    }
}

/// One complex sample in an exchanged directivity pattern.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExternalReferenceComplexSample {
    pub real: f64,
    pub imag: f64,
}

fn validate_directivity_axes(horizontal: &[f64], vertical: &[f64]) -> Result<(), String> {
    if horizontal.len() < 2
        || horizontal
            .iter()
            .any(|angle| !angle.is_finite() || *angle < 0.0 || *angle >= 360.0)
        || horizontal.windows(2).any(|pair| pair[1] <= pair[0])
    {
        return Err(
            "external-reference directivity horizontal angles must be finite, strictly increasing, and in [0, 360)"
                .into(),
        );
    }
    if vertical.len() < 2
        || vertical
            .iter()
            .any(|angle| !angle.is_finite() || *angle < 0.0 || *angle > 180.0)
        || vertical.windows(2).any(|pair| pair[1] <= pair[0])
    {
        return Err(
            "external-reference directivity vertical angles must be finite, strictly increasing, and in [0, 180]"
                .into(),
        );
    }
    Ok(())
}

/// A validated, digest-bound external-solver input manifest.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExternalReferenceManifest {
    pub schema_version: String,
    pub contract: String,
    pub backend_label: String,
    pub scenario_id: String,
    pub geometry: RoomGeometry,
    pub boundaries: BoundaryConfig,
    pub sound_speed_m_per_s: f64,
    pub frequencies_hz: Vec<f64>,
    pub sources: Vec<ExternalReferenceSource>,
    pub listeners: Vec<Point3D>,
    pub content_sha256: String,
}

/// Explicit process command for an external reference driver.
///
/// The process receives absolute paths through
/// `ROOMSIM_EXTERNAL_REFERENCE_MANIFEST`, `ROOMSIM_EXTERNAL_REFERENCE_DATA`,
/// and `ROOMSIM_EXTERNAL_REFERENCE_OUTPUT`. The output must be a validated
/// `sonium-sim-room-reference-response/v1` or `/v2` JSON envelope.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExternalReferenceCommand {
    pub program: OsString,
    pub args: Vec<OsString>,
}

impl ExternalReferenceCommand {
    pub fn new(program: impl Into<OsString>) -> Self {
        Self {
            program: program.into(),
            args: Vec::new(),
        }
    }

    pub fn with_args<I, S>(mut self, args: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<OsString>,
    {
        self.args = args.into_iter().map(Into::into).collect();
        self
    }
}

/// Validated result of one external reference process invocation.
#[derive(Debug, Clone)]
pub struct ExternalReferenceExecution {
    pub manifest_path: std::path::PathBuf,
    pub data_path: std::path::PathBuf,
    pub response_path: std::path::PathBuf,
    pub status_code: Option<i32>,
    pub stdout: String,
    pub stderr: String,
    pub reference: GalleryReference,
}

fn valid_label(value: &str, field: &str) -> Result<(), String> {
    if value.trim().is_empty()
        || value == "."
        || value == ".."
        || value.contains('/')
        || value.contains('\\')
        || value.chars().any(|character| character.is_control())
    {
        return Err(format!("{field} must be a portable non-path label"));
    }
    Ok(())
}

fn valid_point(point: Point3D, field: &str) -> Result<(), String> {
    if [point.x, point.y, point.z].into_iter().all(f64::is_finite) {
        Ok(())
    } else {
        Err(format!("{field} must contain finite coordinates"))
    }
}

fn update_len(hasher: &mut Sha256, value: usize) {
    hasher.update((value as u64).to_le_bytes());
}

fn update_str(hasher: &mut Sha256, value: &str) {
    update_len(hasher, value.len());
    hasher.update(value.as_bytes());
}

fn update_f64(hasher: &mut Sha256, value: f64) {
    hasher.update(value.to_bits().to_le_bytes());
}

fn update_point(hasher: &mut Sha256, point: Point3D) {
    update_f64(hasher, point.x);
    update_f64(hasher, point.y);
    update_f64(hasher, point.z);
}

fn canonical_content_sha256(manifest: &ExternalReferenceManifest) -> String {
    let mut hasher = Sha256::new();
    update_str(&mut hasher, &manifest.schema_version);
    update_str(&mut hasher, &manifest.contract);
    update_str(&mut hasher, &manifest.backend_label);
    update_str(&mut hasher, &manifest.scenario_id);
    let geometry = serde_json::to_vec(&manifest.geometry).expect("geometry is serializable");
    update_len(&mut hasher, geometry.len());
    hasher.update(geometry);
    let boundaries = serde_json::to_vec(&manifest.boundaries).expect("boundaries are serializable");
    update_len(&mut hasher, boundaries.len());
    hasher.update(boundaries);
    update_f64(&mut hasher, manifest.sound_speed_m_per_s);
    update_len(&mut hasher, manifest.frequencies_hz.len());
    for frequency in &manifest.frequencies_hz {
        update_f64(&mut hasher, *frequency);
    }
    update_len(&mut hasher, manifest.sources.len());
    for source in &manifest.sources {
        update_str(&mut hasher, &source.name);
        update_point(&mut hasher, source.position);
        update_point(&mut hasher, source.orientation);
        update_len(&mut hasher, source.amplitudes.len());
        for amplitude in &source.amplitudes {
            update_f64(&mut hasher, *amplitude);
        }
        if manifest.schema_version == EXTERNAL_REFERENCE_PROBLEM_SCHEMA_V2 {
            let directivity =
                serde_json::to_vec(&source.directivity).expect("directivity is serializable");
            update_len(&mut hasher, directivity.len());
            hasher.update(directivity);
        }
    }
    update_len(&mut hasher, manifest.listeners.len());
    for listener in &manifest.listeners {
        update_point(&mut hasher, *listener);
    }
    hasher
        .finalize()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

impl ExternalReferenceManifest {
    /// Build a manifest from an existing RoomSim problem.
    pub fn from_simulation(
        scenario_id: impl Into<String>,
        backend_label: impl Into<String>,
        simulation: &RoomSimulation,
    ) -> Result<Self, String> {
        let mut manifest = Self {
            schema_version: EXTERNAL_REFERENCE_PROBLEM_SCHEMA_V1.into(),
            contract: EXTERNAL_REFERENCE_CONTRACT_V1.into(),
            backend_label: backend_label.into(),
            scenario_id: scenario_id.into(),
            geometry: simulation.room.clone(),
            boundaries: simulation.boundaries.clone(),
            sound_speed_m_per_s: simulation.speed_of_sound,
            frequencies_hz: simulation.frequencies.clone(),
            sources: Vec::new(),
            listeners: simulation.listening_positions.clone(),
            content_sha256: String::new(),
        };
        let mut has_directional_source = false;
        for source in &simulation.sources {
            let directivity = ExternalReferenceDirectivity::from_source(source);
            has_directional_source |= !directivity.is_isotropic();
            manifest.sources.push(ExternalReferenceSource {
                name: source.name.clone(),
                position: source.position,
                orientation: source.orientation,
                amplitudes: simulation
                    .frequencies
                    .iter()
                    .map(|frequency| source.isotropic_amplitude(*frequency))
                    .collect(),
                directivity,
            });
        }
        if has_directional_source {
            manifest.schema_version = EXTERNAL_REFERENCE_PROBLEM_SCHEMA_V2.into();
        }
        manifest.content_sha256 = canonical_content_sha256(&manifest);
        manifest.validate()?;
        Ok(manifest)
    }

    /// Validate schema, topology, finite physical inputs, source model, and digest.
    pub fn validate(&self) -> Result<(), String> {
        if self.schema_version != EXTERNAL_REFERENCE_PROBLEM_SCHEMA_V1
            && self.schema_version != EXTERNAL_REFERENCE_PROBLEM_SCHEMA_V2
        {
            return Err(format!(
                "unsupported external-reference schema {:?}",
                self.schema_version
            ));
        }
        if self.contract != EXTERNAL_REFERENCE_CONTRACT_V1 {
            return Err(format!(
                "unsupported external-reference contract {:?}",
                self.contract
            ));
        }
        valid_label(&self.backend_label, "backend label")?;
        valid_label(&self.scenario_id, "scenario id")?;
        let (width, depth, height) = self.geometry.dimensions();
        if [width, depth, height]
            .into_iter()
            .any(|value| !value.is_finite() || value <= 0.0)
        {
            return Err(
                "external-reference geometry dimensions must be finite and positive".into(),
            );
        }
        if !self.sound_speed_m_per_s.is_finite() || self.sound_speed_m_per_s <= 0.0 {
            return Err("external-reference sound speed must be finite and positive".into());
        }
        if self.frequencies_hz.is_empty()
            || self
                .frequencies_hz
                .iter()
                .any(|frequency| !frequency.is_finite() || *frequency <= 0.0)
            || self
                .frequencies_hz
                .windows(2)
                .any(|window| window[1] <= window[0])
        {
            return Err(
                "external-reference frequencies must be finite, positive, and strictly increasing"
                    .into(),
            );
        }
        if self.sources.is_empty() || self.listeners.is_empty() {
            return Err("external-reference problem requires sources and listeners".into());
        }
        let mut source_names = HashSet::with_capacity(self.sources.len());
        for (source_index, source) in self.sources.iter().enumerate() {
            valid_label(&source.name, &format!("source {source_index} name"))?;
            if !source_names.insert(source.name.to_ascii_lowercase()) {
                return Err("external-reference source names must be unique".into());
            }
            valid_point(source.position, &format!("source {source_index} position"))?;
            valid_point(
                source.orientation,
                &format!("source {source_index} orientation"),
            )?;
            if source.amplitudes.len() != self.frequencies_hz.len()
                || source
                    .amplitudes
                    .iter()
                    .any(|amplitude| !amplitude.is_finite())
            {
                return Err(format!(
                    "external-reference source {source_index} amplitudes do not match frequency grid"
                ));
            }
            source.directivity.validate()?;
            if self.schema_version == EXTERNAL_REFERENCE_PROBLEM_SCHEMA_V1
                && !source.directivity.is_isotropic()
            {
                return Err(
                    "external-reference v1 cannot carry directional source data; use v2".into(),
                );
            }
        }
        for (listener_index, listener) in self.listeners.iter().enumerate() {
            valid_point(*listener, &format!("listener {listener_index}"))?;
        }
        if self.content_sha256.len() != 64
            || !self
                .content_sha256
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
        {
            return Err("external-reference content_sha256 must be lowercase SHA-256".into());
        }
        let expected = canonical_content_sha256(self);
        if self.content_sha256 != expected {
            return Err(format!(
                "external-reference content_sha256 mismatch: expected {expected}"
            ));
        }
        Ok(())
    }

    /// Serialize a validated deterministic JSON manifest.
    pub fn to_json(&self) -> Result<String, String> {
        self.validate()?;
        serde_json::to_string_pretty(self)
            .map_err(|error| format!("failed to serialize external reference: {error}"))
    }

    /// Parse and validate a JSON manifest exported by another process.
    pub fn from_json(document: &str) -> Result<Self, String> {
        let manifest: Self = serde_json::from_str(document)
            .map_err(|error| format!("invalid external-reference JSON: {error}"))?;
        manifest.validate()?;
        Ok(manifest)
    }

    /// Write the checked JSON manifest to disk.
    pub fn write_json(&self, path: impl AsRef<Path>) -> Result<(), String> {
        std::fs::write(path.as_ref(), self.to_json()?)
            .map_err(|error| format!("failed to write external-reference manifest: {error}"))
    }

    /// Render a simple line-oriented data file suitable for a FreeFEM driver.
    ///
    /// This is an input adapter, not a FreeFEM solver script.  Names use JSON
    /// string quoting so a driver can parse them without path-like ambiguity.
    pub fn to_freefem_data(&self) -> Result<String, String> {
        self.validate()?;
        let mut data = String::new();
        data.push_str(&format!(
            "# RoomSim external reference data {}\n",
            self.schema_version
        ));
        data.push_str(&format!("schema {}\n", self.schema_version));
        data.push_str(&format!("contract {}\n", self.contract));
        data.push_str(&format!("backend {}\n", self.backend_label));
        data.push_str(&format!("scenario {}\n", self.scenario_id));
        let (width, depth, height) = self.geometry.dimensions();
        data.push_str(&format!(
            "room_dimensions {:.17e} {:.17e} {:.17e}\n",
            width, depth, height
        ));
        data.push_str(&format!(
            "sound_speed_m_per_s {:.17e}\n",
            self.sound_speed_m_per_s
        ));
        data.push_str(&format!("frequency_count {}\n", self.frequencies_hz.len()));
        for (index, frequency) in self.frequencies_hz.iter().enumerate() {
            data.push_str(&format!("frequency {} {:.17e}\n", index, frequency));
        }
        data.push_str(&format!("source_count {}\n", self.sources.len()));
        for (source_index, source) in self.sources.iter().enumerate() {
            let name = serde_json::to_string(&source.name).expect("source name is serializable");
            data.push_str(&format!(
                "source {} {} {:.17e} {:.17e} {:.17e} {:.17e} {:.17e} {:.17e}\n",
                source_index,
                name,
                source.position.x,
                source.position.y,
                source.position.z,
                source.orientation.x,
                source.orientation.y,
                source.orientation.z
            ));
            for (frequency_index, amplitude) in source.amplitudes.iter().enumerate() {
                data.push_str(&format!(
                    "source_amplitude {} {} {:.17e}\n",
                    source_index, frequency_index, amplitude
                ));
            }
            if self.schema_version == EXTERNAL_REFERENCE_PROBLEM_SCHEMA_V2 {
                let directivity = serde_json::to_string(&source.directivity)
                    .expect("directivity is serializable");
                data.push_str(&format!(
                    "source_directivity {} {}\n",
                    source_index, directivity
                ));
            }
        }
        data.push_str(&format!("listener_count {}\n", self.listeners.len()));
        for (listener_index, listener) in self.listeners.iter().enumerate() {
            data.push_str(&format!(
                "listener {} {:.17e} {:.17e} {:.17e}\n",
                listener_index, listener.x, listener.y, listener.z
            ));
        }
        data.push_str(&format!("content_sha256 {}\n", self.content_sha256));
        Ok(data)
    }
}

fn validate_reference_against_manifest(
    manifest: &ExternalReferenceManifest,
    reference: &GalleryReference,
) -> Result<(), String> {
    if reference.backend != manifest.backend_label {
        return Err(format!(
            "external-reference backend mismatch: expected {}, got {}",
            manifest.backend_label, reference.backend
        ));
    }
    if reference.problem_sha256.as_deref() != Some(manifest.content_sha256.as_str()) {
        return Err(
            "external-reference response problem_sha256 does not match the input manifest".into(),
        );
    }
    let output = &reference.output;
    output.validate_layout()?;
    if output.frequencies.len() != manifest.frequencies_hz.len()
        || output
            .frequencies
            .iter()
            .zip(&manifest.frequencies_hz)
            .any(|(actual, expected)| actual.to_bits() != expected.to_bits())
    {
        return Err("external-reference response frequency topology mismatch".into());
    }
    let expected_source_names = manifest
        .sources
        .iter()
        .map(|source| source.name.clone())
        .collect::<Vec<_>>();
    if output.source_names != expected_source_names {
        return Err("external-reference response source topology mismatch".into());
    }
    if output.pressures.len() != manifest.sources.len()
        || output
            .pressures
            .iter()
            .any(|source| source.len() != manifest.listeners.len())
    {
        return Err("external-reference response listener topology mismatch".into());
    }
    Ok(())
}

/// Execute an external reference driver and validate its solved response.
pub fn execute_external_reference(
    manifest: &ExternalReferenceManifest,
    working_directory: impl AsRef<Path>,
    command: &ExternalReferenceCommand,
) -> Result<ExternalReferenceExecution, String> {
    execute_external_reference_with_timeout(
        manifest,
        working_directory,
        command,
        DEFAULT_EXTERNAL_REFERENCE_TIMEOUT,
    )
}

/// Execute an external reference driver with an explicit wall-clock limit.
///
/// The child is spawned with separately drained stdout/stderr pipes so a
/// verbose driver cannot deadlock on a full pipe. On timeout the child is
/// killed, waited, and its captured diagnostics are included in the returned
/// error.
pub fn execute_external_reference_with_timeout(
    manifest: &ExternalReferenceManifest,
    working_directory: impl AsRef<Path>,
    command: &ExternalReferenceCommand,
    timeout: Duration,
) -> Result<ExternalReferenceExecution, String> {
    manifest.validate()?;
    if timeout.is_zero() {
        return Err("external-reference timeout must be positive".into());
    }
    let working_directory = working_directory.as_ref();
    if !working_directory.is_dir() {
        return Err(format!(
            "external-reference working directory is not a directory: {}",
            working_directory.display()
        ));
    }

    let manifest_path = working_directory.join("roomsim-external-reference-manifest.v1.json");
    let data_path = working_directory.join("roomsim-external-reference-data.v1.txt");
    let response_path = working_directory.join("roomsim-external-reference-response.json");
    manifest.write_json(&manifest_path)?;
    std::fs::write(&data_path, manifest.to_freefem_data()?)
        .map_err(|error| format!("failed write external-reference data: {error}"))?;
    if response_path.exists() {
        std::fs::remove_file(&response_path)
            .map_err(|error| format!("failed remove stale external-reference response: {error}"))?;
    }

    let deadline = Instant::now()
        .checked_add(timeout)
        .ok_or_else(|| "external-reference timeout deadline overflowed".to_string())?;

    let mut process = Command::new(&command.program)
        .args(&command.args)
        .current_dir(working_directory)
        .env("ROOMSIM_EXTERNAL_REFERENCE_MANIFEST", &manifest_path)
        .env("ROOMSIM_EXTERNAL_REFERENCE_DATA", &data_path)
        .env("ROOMSIM_EXTERNAL_REFERENCE_OUTPUT", &response_path)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|error| format!("external-reference process launch failed: {error}"))?;
    let mut stdout_pipe = match process.stdout.take() {
        Some(pipe) => pipe,
        None => {
            let _ = process.kill();
            let _ = process.wait();
            return Err("external-reference stdout pipe was unavailable".into());
        }
    };
    let mut stderr_pipe = match process.stderr.take() {
        Some(pipe) => pipe,
        None => {
            let _ = process.kill();
            let _ = process.wait();
            return Err("external-reference stderr pipe was unavailable".into());
        }
    };
    let stdout_reader = thread::spawn(move || {
        let mut bytes = Vec::new();
        stdout_pipe
            .read_to_end(&mut bytes)
            .map(|_| bytes)
            .map_err(|error| error.to_string())
    });
    let stderr_reader = thread::spawn(move || {
        let mut bytes = Vec::new();
        stderr_pipe
            .read_to_end(&mut bytes)
            .map(|_| bytes)
            .map_err(|error| error.to_string())
    });
    let mut timed_out = false;
    let status = loop {
        match process.try_wait() {
            Ok(Some(status)) => break status,
            Ok(None) if Instant::now() >= deadline => {
                timed_out = true;
                process.kill().map_err(|error| {
                    format!("failed to kill timed-out external-reference process: {error}")
                })?;
                break process.wait().map_err(|error| {
                    format!("failed waiting for killed external-reference process: {error}")
                })?;
            }
            Ok(None) => thread::sleep(Duration::from_millis(5)),
            Err(error) => {
                let message = format!("failed waiting for external-reference process: {error}");
                if process.kill().is_ok() {
                    let _ = process.wait();
                    let _ = stdout_reader.join();
                    let _ = stderr_reader.join();
                }
                return Err(message);
            }
        }
    };
    let stdout_bytes = stdout_reader
        .join()
        .map_err(|_| "external-reference stdout reader panicked".to_string())?
        .map_err(|error| format!("failed reading external-reference stdout: {error}"))?;
    let stderr_bytes = stderr_reader
        .join()
        .map_err(|_| "external-reference stderr reader panicked".to_string())?
        .map_err(|error| format!("failed reading external-reference stderr: {error}"))?;
    let stdout = String::from_utf8_lossy(&stdout_bytes).into_owned();
    let stderr = String::from_utf8_lossy(&stderr_bytes).into_owned();
    if timed_out {
        return Err(format!(
            "external-reference process timed out after {} ms: {}",
            timeout.as_millis(),
            stderr.trim()
        ));
    }
    if !status.success() {
        return Err(format!(
            "external-reference process failed with {}: {}",
            status,
            stderr.trim()
        ));
    }
    if !response_path.is_file() {
        return Err("external-reference process did not produce a response file".into());
    }

    let reference = load_reference_response_json(&response_path)?;
    validate_reference_against_manifest(manifest, &reference)?;
    Ok(ExternalReferenceExecution {
        manifest_path,
        data_path,
        response_path,
        status_code: status.code(),
        stdout,
        stderr,
        reference,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use sonium_xem::{Directivity, Point3D, RectangularRoom, RoomGeometry, Source};

    fn simulation(source: Source) -> RoomSimulation {
        RoomSimulation {
            room: RoomGeometry::Rectangular(RectangularRoom::new(5.0, 4.0, 3.0)),
            sources: vec![source],
            listening_positions: vec![Point3D::new(2.0, 2.0, 1.2)],
            frequencies: vec![100.0, 200.0],
            boundaries: BoundaryConfig::default(),
            speed_of_sound: 343.0,
        }
    }

    #[test]
    fn isotropic_manifest_is_digest_bound_deterministic_and_freefem_ready() {
        let source =
            Source::omnidirectional(Point3D::new(1.0, 1.0, 1.0), 1.0).with_name("left".into());
        let manifest = ExternalReferenceManifest::from_simulation(
            "fixture-room",
            "freefem",
            &simulation(source),
        )
        .unwrap();
        let first = manifest.to_json().unwrap();
        let second = manifest.to_json().unwrap();
        assert_eq!(first, second);
        let decoded = ExternalReferenceManifest::from_json(&first).unwrap();
        assert_eq!(decoded.content_sha256, manifest.content_sha256);
        assert_eq!(decoded.sources, manifest.sources);
        assert_eq!(decoded.listeners, manifest.listeners);
        let data = manifest.to_freefem_data().unwrap();
        assert!(data.contains("source_amplitude 0 1"));
        assert!(data.contains("listener 0"));
        assert!(data.contains(&manifest.content_sha256));
    }

    #[test]
    fn directional_source_is_exported_without_downgrading_to_isotropic() {
        let source = Source::new(
            Point3D::new(1.0, 1.0, 1.0),
            Directivity::Classical {
                h_angle: 90.0,
                v_angle: 90.0,
            },
            1.0,
        )
        .with_name("directional".into());
        let manifest = ExternalReferenceManifest::from_simulation(
            "fixture-room",
            "freefem",
            &simulation(source),
        )
        .unwrap();
        assert_eq!(
            manifest.schema_version,
            EXTERNAL_REFERENCE_PROBLEM_SCHEMA_V2
        );
        assert!(matches!(
            manifest.sources[0].directivity,
            ExternalReferenceDirectivity::Classical { .. }
        ));
        let decoded = ExternalReferenceManifest::from_json(&manifest.to_json().unwrap()).unwrap();
        assert_eq!(decoded.sources, manifest.sources);
        let data = manifest.to_freefem_data().unwrap();
        assert!(data.contains("source_directivity 0"));
        assert!(data.contains("horizontal_beamwidth_deg"));
    }

    #[test]
    fn sampled_directivity_round_trips_and_invalid_axes_fail_closed() {
        let source = Source::new(
            Point3D::new(1.0, 1.0, 1.0),
            Directivity::Grid(sonium_xem::DirectivityGrid::cardioid()),
            1.0,
        )
        .with_name("grid".into());
        let manifest = ExternalReferenceManifest::from_simulation(
            "fixture-room",
            "freefem",
            &simulation(source),
        )
        .unwrap();
        assert!(matches!(
            manifest.sources[0].directivity,
            ExternalReferenceDirectivity::Grid { .. }
        ));
        manifest.validate().unwrap();

        let invalid = ExternalReferenceDirectivity::Grid {
            horizontal_angles_deg: vec![0.0, 360.0],
            vertical_angles_deg: vec![0.0, 180.0],
            magnitude: vec![vec![1.0, 1.0], vec![1.0, 1.0]],
        };
        assert!(invalid.validate().is_err());
    }

    #[test]
    fn complex_directivity_round_trips_with_frequency_patterns() {
        let pattern =
            sonium_compute::HostMatrix::from_elem(2, 2, num_complex::Complex64::new(0.5, 0.25));
        let directivity = sonium_xem::ComplexDirectivityGrid::try_new(
            vec![0.0, 180.0],
            vec![0.0, 180.0],
            vec![100.0],
            vec![pattern],
        )
        .unwrap();
        let source = Source::new(
            Point3D::new(1.0, 1.0, 1.0),
            Directivity::ComplexGrid(directivity),
            1.0,
        )
        .with_name("complex".into());
        let manifest = ExternalReferenceManifest::from_simulation(
            "fixture-room",
            "freefem",
            &simulation(source),
        )
        .unwrap();
        assert!(matches!(
            manifest.sources[0].directivity,
            ExternalReferenceDirectivity::ComplexGrid { .. }
        ));
        let decoded = ExternalReferenceManifest::from_json(&manifest.to_json().unwrap()).unwrap();
        assert_eq!(decoded.sources, manifest.sources);
    }

    #[test]
    fn tampering_or_unsafe_labels_fail_closed() {
        let source =
            Source::omnidirectional(Point3D::new(1.0, 1.0, 1.0), 1.0).with_name("left".into());
        let mut manifest = ExternalReferenceManifest::from_simulation(
            "fixture-room",
            "freefem",
            &simulation(source),
        )
        .unwrap();
        manifest.frequencies_hz[0] = 150.0;
        assert!(manifest.validate().is_err());
        manifest.frequencies_hz[0] = 100.0;
        manifest.scenario_id = "../escape".into();
        assert!(manifest.validate().is_err());
    }

    fn response_document(manifest: &ExternalReferenceManifest, backend: &str) -> String {
        serde_json::json!({
            "schema": crate::gallery::REFERENCE_RESPONSE_SCHEMA_V1,
            "backend": backend,
            "problem_sha256": manifest.content_sha256,
            "frequencies": manifest.frequencies_hz,
            "source_names": manifest.sources.iter().map(|source| source.name.clone()).collect::<Vec<_>>(),
            "pressures": [[[ {
                "real_pa": 1.0,
                "imag_pa": 0.0
            }, {
                "real_pa": 0.5,
                "imag_pa": 0.25
            } ]]],
            "spatial_field": null
        })
        .to_string()
    }

    #[test]
    fn external_command_round_trips_inputs_and_validates_response_topology() {
        let source =
            Source::omnidirectional(Point3D::new(1.0, 1.0, 1.0), 1.0).with_name("left".into());
        let manifest = ExternalReferenceManifest::from_simulation(
            "fixture-room",
            "freefem",
            &simulation(source),
        )
        .unwrap();
        let directory = tempfile::tempdir().unwrap();
        std::fs::write(
            directory.path().join("fixture-response.json"),
            response_document(&manifest, "freefem"),
        )
        .unwrap();
        let command = ExternalReferenceCommand::new("sh").with_args([
            "-c",
            "cp fixture-response.json \"$ROOMSIM_EXTERNAL_REFERENCE_OUTPUT\"",
        ]);

        let execution = execute_external_reference(&manifest, directory.path(), &command).unwrap();
        assert_eq!(execution.status_code, Some(0));
        assert!(execution.manifest_path.is_file());
        assert!(execution.data_path.is_file());
        assert!(execution.response_path.is_file());
        assert_eq!(execution.reference.backend, "freefem");
        assert_eq!(
            execution.reference.output.frequencies,
            manifest.frequencies_hz
        );
    }

    #[test]
    fn external_command_rejects_failed_or_missing_driver_output() {
        let source =
            Source::omnidirectional(Point3D::new(1.0, 1.0, 1.0), 1.0).with_name("left".into());
        let manifest = ExternalReferenceManifest::from_simulation(
            "fixture-room",
            "freefem",
            &simulation(source),
        )
        .unwrap();
        let directory = tempfile::tempdir().unwrap();

        let failed =
            ExternalReferenceCommand::new("sh").with_args(["-c", "echo failed >&2; exit 9"]);
        let error = execute_external_reference(&manifest, directory.path(), &failed).unwrap_err();
        assert!(error.contains("process failed"), "{error}");
        assert!(error.contains("failed"), "{error}");

        let missing = ExternalReferenceCommand::new("sh").with_args(["-c", "exit 0"]);
        let error = execute_external_reference(&manifest, directory.path(), &missing).unwrap_err();
        assert!(error.contains("did not produce a response file"), "{error}");
    }

    #[test]
    fn external_command_rejects_backend_and_frequency_mismatches() {
        let source =
            Source::omnidirectional(Point3D::new(1.0, 1.0, 1.0), 1.0).with_name("left".into());
        let manifest = ExternalReferenceManifest::from_simulation(
            "fixture-room",
            "freefem",
            &simulation(source),
        )
        .unwrap();
        let directory = tempfile::tempdir().unwrap();
        let response_path = directory.path().join("fixture-response.json");
        let command = ExternalReferenceCommand::new("sh").with_args([
            "-c",
            "cp fixture-response.json \"$ROOMSIM_EXTERNAL_REFERENCE_OUTPUT\"",
        ]);

        std::fs::write(
            &response_path,
            response_document(&manifest, "other-backend"),
        )
        .unwrap();
        let error = execute_external_reference(&manifest, directory.path(), &command).unwrap_err();
        assert!(error.contains("backend mismatch"), "{error}");

        let mut changed = manifest.clone();
        changed.frequencies_hz[0] = 150.0;
        changed.content_sha256 = canonical_content_sha256(&changed);
        std::fs::write(&response_path, response_document(&manifest, "freefem")).unwrap();
        let error = execute_external_reference(&changed, directory.path(), &command).unwrap_err();
        assert!(
            error.contains("frequency topology mismatch")
                || error.contains("content_sha256")
                || error.contains("problem_sha256"),
            "{error}"
        );
    }

    #[test]
    fn external_command_timeout_kills_hanging_driver() {
        let source =
            Source::omnidirectional(Point3D::new(1.0, 1.0, 1.0), 1.0).with_name("left".into());
        let manifest = ExternalReferenceManifest::from_simulation(
            "fixture-room",
            "freefem",
            &simulation(source),
        )
        .unwrap();
        let directory = tempfile::tempdir().unwrap();
        let command =
            ExternalReferenceCommand::new("sh").with_args(["-c", "while true; do :; done"]);

        let error = execute_external_reference_with_timeout(
            &manifest,
            directory.path(),
            &command,
            Duration::from_millis(25),
        )
        .unwrap_err();
        assert!(error.contains("timed out"), "{error}");
    }

    #[test]
    fn external_command_rejects_zero_timeout_before_launch() {
        let source =
            Source::omnidirectional(Point3D::new(1.0, 1.0, 1.0), 1.0).with_name("left".into());
        let manifest = ExternalReferenceManifest::from_simulation(
            "fixture-room",
            "freefem",
            &simulation(source),
        )
        .unwrap();
        let directory = tempfile::tempdir().unwrap();
        let command = ExternalReferenceCommand::new("sh")
            .with_args(["-c", "touch roomsim-external-reference-launched"]);

        let error = execute_external_reference_with_timeout(
            &manifest,
            directory.path(),
            &command,
            Duration::ZERO,
        )
        .unwrap_err();

        assert!(error.contains("timeout must be positive"), "{error}");
        assert!(
            !directory
                .path()
                .join("roomsim-external-reference-launched")
                .exists()
        );
    }
}
