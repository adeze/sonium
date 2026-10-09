//! Deterministic, static gallery artifacts for room-simulation outputs.
//!
//! The plots deliberately retain their numerical CSV/WAV siblings as the
//! authoritative data.  They are presentation assets, not an alternative
//! simulation format.

use crate::SimulationOutput;
use crate::fast_room::sample_fast_hybrid_field;
use crate::fem_runner::FemNodalField;
use crate::room_metrics::SpatialMetricSeries;
use num_complex::Complex64;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use sonium_xem::io::GalleryRenderer;
use sonium_xem::{Point3D, RoomSimulation};
use std::path::Path;

/// Versioned JSON envelope for an independently solved gallery reference.
pub const REFERENCE_RESPONSE_SCHEMA_V1: &str = "sonium-sim-room-reference-response/v1";
/// Digest-bound envelope for independently solved gallery references.
pub const REFERENCE_RESPONSE_SCHEMA_V2: &str = "sonium-sim-room-reference-response/v2";
/// Versioned payload for an independently solved planar field.
pub const REFERENCE_SPATIAL_FIELD_SCHEMA_V1: &str = "sonium-sim-room-reference-field/v1";

/// Default XY slice sampling grid (32x32 keeps smoke galleries fast).
pub const DEFAULT_SLICE_GRID: usize = 32;
/// Bounds for caller-supplied slice grids; larger grids sample the same
/// field evaluator, so cost grows quadratically with the side length.
pub const MIN_SLICE_GRID: usize = 8;
pub const MAX_SLICE_GRID: usize = 1024;

/// Presentation options for gallery writers. Numerical CSV exports are
/// unaffected; only slice sampling density changes.
#[derive(Debug, Clone, Copy)]
pub struct GalleryOptions {
    /// Side length of the square XY slice sampling grid.
    pub slice_grid: usize,
}

impl Default for GalleryOptions {
    fn default() -> Self {
        Self {
            slice_grid: DEFAULT_SLICE_GRID,
        }
    }
}

impl GalleryOptions {
    fn validated_slice_grid(&self) -> Result<usize, String> {
        if !(MIN_SLICE_GRID..=MAX_SLICE_GRID).contains(&self.slice_grid) {
            return Err(format!(
                "gallery slice grid must be within {MIN_SLICE_GRID}..={MAX_SLICE_GRID}"
            ));
        }
        Ok(self.slice_grid)
    }
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct ReferencePressure {
    real_pa: f64,
    imag_pa: f64,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct ReferenceSpatialFieldDocument {
    schema: String,
    source_index: usize,
    source_name: String,
    plane_axis: String,
    plane_coordinate_m: f64,
    grid_rows: usize,
    grid_columns: usize,
    positions: Vec<ReferencePoint>,
    frequencies: Vec<f64>,
    pressures: Vec<Vec<ReferencePressure>>,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct ReferencePoint {
    x_m: f64,
    y_m: f64,
    z_m: f64,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct ReferenceResponseDocument {
    schema: String,
    backend: String,
    frequencies: Vec<f64>,
    source_names: Vec<String>,
    pressures: Vec<Vec<Vec<ReferencePressure>>>,
    spatial_field: Option<ReferenceSpatialFieldDocument>,
    content_sha256: Option<String>,
    problem_sha256: Option<String>,
}

#[derive(Debug, Serialize)]
struct ReferenceResponseDigestPayload<'a> {
    schema: &'a str,
    backend: &'a str,
    frequencies: &'a [f64],
    source_names: &'a [String],
    pressures: &'a [Vec<Vec<ReferencePressure>>],
    spatial_field: &'a Option<ReferenceSpatialFieldDocument>,
}

fn canonical_reference_response_sha256(
    document: &ReferenceResponseDocument,
) -> Result<String, String> {
    let payload = ReferenceResponseDigestPayload {
        schema: &document.schema,
        backend: &document.backend,
        frequencies: &document.frequencies,
        source_names: &document.source_names,
        pressures: &document.pressures,
        spatial_field: &document.spatial_field,
    };
    let bytes = serde_json::to_vec(&payload)
        .map_err(|error| format!("reference response digest serialization failed: {error}"))?;
    Ok(Sha256::digest(bytes)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect())
}

fn validate_reference_response_digest(document: &ReferenceResponseDocument) -> Result<(), String> {
    if let Some(problem_sha256) = document.problem_sha256.as_deref()
        && (problem_sha256.len() != 64
            || !problem_sha256
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte)))
    {
        return Err("reference response problem_sha256 must be a lowercase SHA-256 digest".into());
    }
    match document.schema.as_str() {
        REFERENCE_RESPONSE_SCHEMA_V1 if document.content_sha256.is_none() => Ok(()),
        REFERENCE_RESPONSE_SCHEMA_V1 => {
            Err("reference response content_sha256 requires schema v2".into())
        }
        REFERENCE_RESPONSE_SCHEMA_V2 => {
            let declared = document
                .content_sha256
                .as_deref()
                .ok_or_else(|| "reference response v2 requires content_sha256".to_string())?;
            if declared.len() != 64 || !declared.bytes().all(|byte| byte.is_ascii_hexdigit()) {
                return Err("reference response content_sha256 must be 64 hex characters".into());
            }
            let expected = canonical_reference_response_sha256(document)?;
            if declared != expected {
                return Err(format!(
                    "reference response content_sha256 mismatch: declared {declared}, expected {expected}"
                ));
            }
            Ok(())
        }
        _ => Err(format!(
            "unsupported reference response schema: {}",
            document.schema
        )),
    }
}

/// A checked planar complex-pressure field supplied by an independent
/// reference solver. Positions are row-major over the declared grid.
#[derive(Debug, Clone)]
pub struct GalleryReferenceField {
    pub source_index: usize,
    pub source_name: String,
    pub plane_axis: String,
    pub plane_coordinate_m: f64,
    pub grid_rows: usize,
    pub grid_columns: usize,
    pub positions: Vec<Point3D>,
    pub frequencies: Vec<f64>,
    pub pressures: Vec<Vec<Complex64>>,
}

impl GalleryReferenceField {
    fn from_document(document: ReferenceSpatialFieldDocument) -> Result<Self, String> {
        if document.schema != REFERENCE_SPATIAL_FIELD_SCHEMA_V1 {
            return Err(format!(
                "unsupported reference spatial field schema: {}",
                document.schema
            ));
        }
        let field = Self {
            source_index: document.source_index,
            source_name: document.source_name,
            plane_axis: document.plane_axis,
            plane_coordinate_m: document.plane_coordinate_m,
            grid_rows: document.grid_rows,
            grid_columns: document.grid_columns,
            positions: document
                .positions
                .into_iter()
                .map(|point| Point3D::new(point.x_m, point.y_m, point.z_m))
                .collect(),
            frequencies: document.frequencies,
            pressures: document
                .pressures
                .into_iter()
                .map(|row| {
                    row.into_iter()
                        .map(|pressure| Complex64::new(pressure.real_pa, pressure.imag_pa))
                        .collect()
                })
                .collect(),
        };
        field.validate()?;
        Ok(field)
    }

    fn validate(&self) -> Result<(), String> {
        if self.source_name.trim().is_empty() {
            return Err("reference spatial field source name is empty".into());
        }
        if self.plane_axis != "z" {
            return Err("reference spatial field must declare a z plane".into());
        }
        if !self.plane_coordinate_m.is_finite() || self.grid_rows < 2 || self.grid_columns < 2 {
            return Err("reference spatial field has invalid plane or grid metadata".into());
        }
        let expected_samples = self
            .grid_rows
            .checked_mul(self.grid_columns)
            .ok_or_else(|| "reference spatial field grid dimensions overflow".to_string())?;
        if self.positions.len() != expected_samples {
            return Err("reference spatial field position/grid shape mismatch".into());
        }
        if self.frequencies.is_empty()
            || self
                .frequencies
                .iter()
                .any(|frequency| !frequency.is_finite() || *frequency <= 0.0)
            || self
                .frequencies
                .windows(2)
                .any(|window| window[1] <= window[0])
        {
            return Err(
                "reference spatial field frequencies must be finite, positive, and increasing"
                    .into(),
            );
        }
        if self.pressures.len() != self.frequencies.len()
            || self
                .pressures
                .iter()
                .any(|row| row.len() != self.positions.len())
        {
            return Err("reference spatial field pressure/frequency shape mismatch".into());
        }
        let plane_tolerance = 1.0e-8 * self.plane_coordinate_m.abs().max(1.0);
        for row in 0..self.grid_rows {
            let row_start = row * self.grid_columns;
            let row_points = &self.positions[row_start..row_start + self.grid_columns];
            for window in row_points.windows(2) {
                if window[1].x <= window[0].x || (window[1].y - window[0].y).abs() > plane_tolerance
                {
                    return Err("reference spatial field grid columns are not ordered".into());
                }
            }
        }
        for column in 0..self.grid_columns {
            for row in 1..self.grid_rows {
                let previous = self.positions[(row - 1) * self.grid_columns + column];
                let current = self.positions[row * self.grid_columns + column];
                if current.y <= previous.y || (current.x - previous.x).abs() > plane_tolerance {
                    return Err("reference spatial field grid rows are not ordered".into());
                }
            }
        }
        if self.positions.iter().any(|position| {
            !position.x.is_finite()
                || !position.y.is_finite()
                || !position.z.is_finite()
                || (position.z - self.plane_coordinate_m).abs() > plane_tolerance
        }) || self
            .pressures
            .iter()
            .flatten()
            .any(|pressure| !pressure.re.is_finite() || !pressure.im.is_finite())
        {
            return Err("reference spatial field contains non-finite or off-plane samples".into());
        }
        Ok(())
    }
}

/// A validated external solved response ready for gallery comparison.
#[derive(Debug, Clone)]
pub struct GalleryReference {
    /// Reference envelope schema used to validate the loaded document.
    pub schema: String,
    /// Backend label displayed in comparison captions and JSON metadata.
    pub backend: String,
    /// Declared content digest for schema v2; absent for compatibility v1.
    pub content_sha256: Option<String>,
    /// Optional digest of the external-reference problem manifest.
    pub problem_sha256: Option<String>,
    /// All samples are marked [`crate::FrequencyProvenance::Solved`].
    pub output: SimulationOutput,
    /// Optional independently solved planar field aligned to the same source.
    pub spatial_field: Option<GalleryReferenceField>,
}

/// Load a versioned solved reference response for gallery comparison.
pub fn load_reference_response_json(path: impl AsRef<Path>) -> Result<GalleryReference, String> {
    let document: ReferenceResponseDocument = serde_json::from_slice(
        &std::fs::read(path).map_err(|error| format!("reference response read failed: {error}"))?,
    )
    .map_err(|error| format!("reference response JSON is invalid: {error}"))?;
    validate_reference_response_digest(&document)?;
    let schema = document.schema.clone();
    let content_sha256 = document.content_sha256.clone();
    let problem_sha256 = document.problem_sha256.clone();
    if document.backend.trim().is_empty() {
        return Err("reference response backend label is empty".into());
    }
    let pressures = document
        .pressures
        .into_iter()
        .map(|source| {
            source
                .into_iter()
                .map(|listener| {
                    listener
                        .into_iter()
                        .map(|pressure| Complex64::new(pressure.real_pa, pressure.imag_pa))
                        .collect()
                })
                .collect()
        })
        .collect();
    let output = SimulationOutput::solved(document.frequencies, pressures, document.source_names);
    output.validate_layout()?;
    let spatial_field = document
        .spatial_field
        .map(GalleryReferenceField::from_document)
        .transpose()?;
    if let Some(field) = spatial_field.as_ref() {
        let response_source_name =
            output.source_names.get(field.source_index).ok_or_else(|| {
                "reference spatial field source index is absent from response".to_string()
            })?;
        if response_source_name != &field.source_name {
            return Err(
                "reference spatial field source disagrees with response source axis".into(),
            );
        }
    }
    Ok(GalleryReference {
        schema,
        backend: document.backend,
        content_sha256,
        problem_sha256,
        output,
        spatial_field,
    })
}

pub fn write_gallery(
    output: &SimulationOutput,
    directory: &Path,
    scenario: &str,
    solver: &str,
    simulation: &RoomSimulation,
    options: &GalleryOptions,
) -> Result<(), String> {
    write_gallery_impl(
        output, directory, scenario, solver, simulation, None, None, None, None, options,
    )
}

/// Write a gallery with a genuine physical-domain FEM nodal-field panel.
///
/// The field must come from the same simulation family, but may contain only
/// the solved FEM band when the response also has dense or synthetic samples.
/// The panel records the field frequency and source explicitly in its sidecar.
pub fn write_gallery_with_fem_field(
    output: &SimulationOutput,
    directory: &Path,
    scenario: &str,
    solver: &str,
    simulation: &RoomSimulation,
    fem_field: &FemNodalField,
    options: &GalleryOptions,
) -> Result<(), String> {
    write_gallery_impl(
        output,
        directory,
        scenario,
        solver,
        simulation,
        Some(fem_field),
        None,
        None,
        None,
        options,
    )
}

/// Write a gallery with a solved external-reference comparison.
///
/// `reference_backend` is deliberately caller-supplied so the same checked
/// panel can represent a FreeFEM, BEM, or other independent solved reference.
/// The reference must match the candidate's solved frequency prefix and
/// source/listener topology. A non-solved tail in the gallery output is
/// allowed and is intentionally excluded from the external-solver comparison.
#[expect(
    clippy::too_many_arguments,
    reason = "gallery writer threading output, paths, scenario, reference and options inputs; grouping would churn the datagen call sites"
)]
pub fn write_gallery_with_reference(
    output: &SimulationOutput,
    directory: &Path,
    scenario: &str,
    solver: &str,
    simulation: &RoomSimulation,
    reference_backend: &str,
    reference: &SimulationOutput,
    frequency_tolerance_hz: f64,
    relative_floor_pa: f64,
    options: &GalleryOptions,
) -> Result<(), String> {
    write_gallery_impl(
        output,
        directory,
        scenario,
        solver,
        simulation,
        None,
        Some((
            reference_backend,
            reference,
            frequency_tolerance_hz,
            relative_floor_pa,
        )),
        None,
        None,
        options,
    )
}

/// Write a gallery containing both a physical FEM slice and an external
/// solved-reference comparison.
#[expect(
    clippy::too_many_arguments,
    reason = "gallery writer threading output, paths, scenario and reference inputs; grouping would churn the datagen call sites"
)]
pub fn write_gallery_with_fem_field_and_reference(
    output: &SimulationOutput,
    directory: &Path,
    scenario: &str,
    solver: &str,
    simulation: &RoomSimulation,
    fem_field: &FemNodalField,
    reference_backend: &str,
    reference: &SimulationOutput,
    frequency_tolerance_hz: f64,
    relative_floor_pa: f64,
    reference_field: Option<&GalleryReferenceField>,
    options: &GalleryOptions,
) -> Result<(), String> {
    write_gallery_impl(
        output,
        directory,
        scenario,
        solver,
        simulation,
        Some(fem_field),
        Some((
            reference_backend,
            reference,
            frequency_tolerance_hz,
            relative_floor_pa,
        )),
        reference_field,
        None,
        options,
    )
}

/// Write a gallery using a loaded reference envelope, preserving its
/// schema and verified content digest in comparison metadata.
pub fn write_gallery_with_loaded_reference(
    output: &SimulationOutput,
    directory: &Path,
    scenario: &str,
    solver: &str,
    simulation: &RoomSimulation,
    reference: &GalleryReference,
    frequency_tolerance_hz: f64,
    relative_floor_pa: f64,
    options: &GalleryOptions,
) -> Result<(), String> {
    write_gallery_impl(
        output,
        directory,
        scenario,
        solver,
        simulation,
        None,
        Some((
            &reference.backend,
            &reference.output,
            frequency_tolerance_hz,
            relative_floor_pa,
        )),
        None,
        Some((&reference.schema, reference.content_sha256.as_deref())),
        options,
    )
}

/// Write a gallery with a retained FEM field using a loaded reference
/// envelope, preserving its schema and verified content digest.
#[expect(
    clippy::too_many_arguments,
    reason = "gallery writer threading output, paths, scenario, reference and options inputs; grouping would churn the datagen call sites"
)]
pub fn write_gallery_with_fem_field_and_loaded_reference(
    output: &SimulationOutput,
    directory: &Path,
    scenario: &str,
    solver: &str,
    simulation: &RoomSimulation,
    fem_field: &FemNodalField,
    reference: &GalleryReference,
    frequency_tolerance_hz: f64,
    relative_floor_pa: f64,
    options: &GalleryOptions,
) -> Result<(), String> {
    write_gallery_impl(
        output,
        directory,
        scenario,
        solver,
        simulation,
        Some(fem_field),
        Some((
            &reference.backend,
            &reference.output,
            frequency_tolerance_hz,
            relative_floor_pa,
        )),
        reference.spatial_field.as_ref(),
        Some((&reference.schema, reference.content_sha256.as_deref())),
        options,
    )
}

#[expect(
    clippy::too_many_arguments,
    reason = "single internal gallery implementation shared by six public writers; options ride along with the existing reference tuple"
)]
fn write_gallery_impl(
    output: &SimulationOutput,
    directory: &Path,
    scenario: &str,
    solver: &str,
    simulation: &RoomSimulation,
    fem_field: Option<&FemNodalField>,
    reference: Option<(&str, &SimulationOutput, f64, f64)>,
    reference_field: Option<&GalleryReferenceField>,
    reference_identity: Option<(&str, Option<&str>)>,
    options: &GalleryOptions,
) -> Result<(), String> {
    output.validate_layout()?;
    let slice_grid = options.validated_slice_grid()?;
    if reference_field.is_some() && fem_field.is_none() {
        return Err(
            "reference spatial field comparison requires a retained FEM nodal field".to_string(),
        );
    }
    if reference_field.is_some() && reference.is_none() {
        return Err(
            "reference spatial field comparison requires a solved response reference".to_string(),
        );
    }
    if let Some((reference_backend, reference_output, frequency_tolerance_hz, relative_floor_pa)) =
        reference
    {
        let comparison_candidate = solved_reference_candidate(output)?;
        crate::backend_parity::compare_backend_outputs(
            reference_backend,
            reference_output,
            solver,
            &comparison_candidate,
            frequency_tolerance_hz,
            relative_floor_pa,
        )?;
    }
    std::fs::create_dir_all(directory).map_err(|e| e.to_string())?;
    let wav_files = crate::wav_export::export_wavs(output, directory).map_err(|e| e.to_string())?;
    let response_csv = directory.join("frequency_response.csv");
    let mut csv = String::from("frequency_hz,spl_db,provenance\n");
    for (index, frequency) in output.frequencies.iter().enumerate() {
        let pressure = output.pressures[0][0][index];
        let spl = 20.0 * pressure.norm().max(20.0e-6).log10();
        csv.push_str(&format!(
            "{frequency:.9},{spl:.9},{}\n",
            output.provenance[index].as_str()
        ));
    }
    std::fs::write(&response_csv, csv).map_err(|e| e.to_string())?;
    let all_response_csv = directory.join("source_listener_response.csv");
    let mut all_csv = String::from("source_index,listener_index,frequency_hz,spl_db,provenance\n");
    for (source_index, source) in output.pressures.iter().enumerate() {
        for (listener_index, listener) in source.iter().enumerate() {
            for (frequency_index, (&frequency, &pressure)) in
                output.frequencies.iter().zip(listener).enumerate()
            {
                let spl = 20.0 * pressure.norm().max(20.0e-6).log10();
                all_csv.push_str(&format!(
                    "{source_index},{listener_index},{frequency:.9},{spl:.9},{}\n",
                    output.provenance[frequency_index].as_str()
                ));
            }
        }
    }
    std::fs::write(&all_response_csv, all_csv).map_err(|e| e.to_string())?;
    let geometry_csv = directory.join("room_geometry.csv");
    write_room_geometry_csv(&geometry_csv, simulation)?;
    write_room_layout_svg(&directory.join("room_layout.svg"), simulation, scenario)?;
    let response_db: Vec<f64> = output.pressures[0][0]
        .iter()
        .map(|pressure| 20.0 * pressure.norm().max(20.0e-6).log10())
        .collect();
    let (ir_time, ir_samples) =
        crate::wav_export::impulse_response(output, 0, 0).map_err(|error| error.to_string())?;
    let ir_time: Vec<f64> = ir_time.into_iter().map(f64::from).collect();
    let ir_samples: Vec<f64> = ir_samples.into_iter().map(f64::from).collect();
    let energy_decay = schroeder_energy_decay_db(&ir_samples)?;
    let all_response_db: Vec<f64> = output
        .pressures
        .iter()
        .flat_map(|source| source.iter())
        .flat_map(|listener| listener.iter())
        .map(|pressure| 20.0 * pressure.norm().max(20.0e-6).log10())
        .collect();
    let spatial_metrics = SpatialMetricSeries::for_source(output, 0)?;
    let spatial_metrics_csv = directory.join("spatial_metrics.csv");
    write_spatial_metrics_csv(&spatial_metrics_csv, &spatial_metrics)?;
    let waterfall_db: Vec<f64> = output.pressures[0]
        .iter()
        .flat_map(|listener| listener.iter())
        .map(|pressure| 20.0 * pressure.norm().max(20.0e-6).log10())
        .collect();
    let rt60_seconds = estimate_rt60_seconds(&ir_time, &energy_decay);
    let metrics_json = serde_json::json!({
        "schema": "sonium-sim-room-metrics/v1",
        "source_index": 0,
        "listener_count": spatial_metrics.listener_count,
        "frequency_count": spatial_metrics.frequencies.len(),
        "rt60_seconds": rt60_seconds,
        "rt60_status": if rt60_seconds.is_some() { "estimated" } else { "insufficient_decay" },
        "provenance": output.provenance.iter().map(|value| value.as_str()).collect::<Vec<_>>(),
    });
    std::fs::write(
        directory.join("room_metrics.json"),
        serde_json::to_vec_pretty(&metrics_json).map_err(|error| error.to_string())?,
    )
    .map_err(|error| error.to_string())?;
    let provenance_code: Vec<f64> = output
        .provenance
        .iter()
        .map(|provenance| match provenance {
            crate::FrequencyProvenance::Solved => 0.0,
            crate::FrequencyProvenance::ReducedOrder => 1.0,
            crate::FrequencyProvenance::Interpolated => 2.0,
            crate::FrequencyProvenance::Geometric => 3.0,
            crate::FrequencyProvenance::Synthetic => 4.0,
        })
        .collect();
    write_gallery_shell(directory, scenario)?;
    let mut gallery = GalleryRenderer::new(directory).map_err(|e| e.to_string())?;
    gallery
        .existing_figure(
            "room_layout.svg",
            "svg",
            "room_geometry.csv",
            "Fixed-camera room plan with dimensions, source positions, and listening positions from the scenario geometry",
        )
        .map_err(|e| e.to_string())?;
    gallery
        .heatmap_png(
            "frequency_response_heatmap.png",
            &response_db,
            "frequency_response.csv",
            "Deterministic false-colour rendering of the solved frequency-response samples; units dB re 20 uPa",
        )
        .map_err(|e| e.to_string())?;
    gallery
        .grid_heatmap_png(
            "listener_frequency_waterfall.png",
            &waterfall_db,
            output.frequencies.len(),
            "source_listener_response.csv",
            "Listener-by-frequency SPL waterfall for source 0; rows follow listener index and columns frequency index",
        )
        .map_err(|e| e.to_string())?;
    gallery
        .heatmap_png(
            "source_listener_response_heatmap.png",
            &all_response_db,
            "source_listener_response.csv",
            "All source/listener/frequency SPL samples (dB re 20 uPa), with source/listener/frequency indices retained in the authoritative CSV",
        )
        .map_err(|e| e.to_string())?;
    gallery
        .line_svg(
            "frequency_response_plot.svg",
            &output.frequencies,
            &response_db,
            "Room frequency response",
            "frequency_response.csv",
            "Frequency (Hz) versus SPL (dB re 20 uPa); provenance is recorded per sample in CSV",
        )
        .map_err(|e| e.to_string())?;
    gallery
        .line_svg(
            "spatial_rms_response_plot.svg",
            &spatial_metrics.frequencies,
            &spatial_metrics
                .samples
                .iter()
                .map(|sample| sample.rms_spl_db)
                .collect::<Vec<_>>(),
            "Spatial RMS response across listeners",
            "spatial_metrics.csv",
            "Listener-order-independent RMS SPL; numerical metrics remain authoritative in spatial_metrics.csv",
        )
        .map_err(|e| e.to_string())?;
    gallery
        .line_svg(
            "impulse_response_plot.svg",
            &ir_time,
            &ir_samples,
            "Room impulse response",
            &wav_files[0],
            "Time-domain impulse response reconstructed from the same frequency samples as source_lp0.wav",
        )
        .map_err(|e| e.to_string())?;
    gallery
        .line_svg(
            "energy_decay_plot.svg",
            &ir_time,
            &energy_decay,
            "Schroeder energy-decay curve",
            &wav_files[0],
            "Remaining cumulative IR energy in dB relative to the total energy; derived from source_lp0.wav samples",
        )
        .map_err(|e| e.to_string())?;
    gallery
        .line_svg(
            "provenance_plot.svg",
            &output.frequencies,
            &provenance_code,
            "Frequency provenance (0 solved, 1 reduced-order, 2 interpolated, 3 geometric, 4 synthetic)",
            "frequency_response.csv",
            "The numeric provenance code is intentionally plotted from the exact per-frequency CSV labels",
        )
        .map_err(|e| e.to_string())?;
    if let Some((reference_backend, reference_output, frequency_tolerance_hz, relative_floor_pa)) =
        reference
    {
        write_reference_comparison(
            &mut gallery,
            directory,
            output,
            solver,
            reference_backend,
            reference_output,
            frequency_tolerance_hz,
            relative_floor_pa,
            reference_identity,
        )?;
    }
    if let Some(reference_field) = reference_field {
        let fem_field = fem_field.ok_or_else(|| {
            "reference spatial field comparison requires a retained FEM nodal field".to_string()
        })?;
        let (reference_backend, _, frequency_tolerance_hz, relative_floor_pa) = reference
            .ok_or_else(|| {
                "reference spatial field comparison requires a solved response reference"
                    .to_string()
            })?;
        write_reference_field_comparison(
            &mut gallery,
            directory,
            fem_field,
            solver,
            reference_backend,
            reference_field,
            frequency_tolerance_hz,
            relative_floor_pa,
            reference_identity,
        )?;
    }
    if let Some(fem_field) = fem_field {
        write_fem_nodal_slice(
            &mut gallery,
            directory,
            fem_field,
            simulation,
            reference_field.map_or(0, |reference| reference.source_index),
            slice_grid,
        )?;
    } else if solver.to_ascii_lowercase().contains("fast") {
        write_fast_hybrid_slice(
            &mut gallery,
            directory,
            simulation,
            &output.frequencies,
            slice_grid,
        )?;
    }
    let manifest_path = gallery
        .finish(
            format!("Room simulation gallery: {scenario}"),
            vec![
                format!("solver={solver}; scenario={scenario}"),
                "fixed 1920x1080 viewport and deterministic colour mapping".into(),
                "CSV/WAV data remain authoritative; plots are presentation only".into(),
            ],
        )
        .map_err(|e| e.to_string())?;
    for entry in std::fs::read_dir(directory).map_err(|error| error.to_string())? {
        let entry = entry.map_err(|error| error.to_string())?;
        if entry
            .path()
            .extension()
            .and_then(|extension| extension.to_str())
            == Some("png")
        {
            sonium_xem::io::validate_png_nonblank(entry.path())
                .map_err(|error| format!("{}: {error}", entry.path().display()))?;
        }
    }
    sonium_xem::io::validate_gallery_manifest(&manifest_path).map_err(|e| e.to_string())?;
    let checksums = std::fs::read_dir(directory)
        .map_err(|e| e.to_string())?
        .filter_map(Result::ok)
        .filter_map(|entry| entry.file_name().into_string().ok())
        .map(|name| {
            let data = std::fs::read(directory.join(&name)).unwrap_or_default();
            (
                name,
                serde_json::json!(format!("fnv1a64:{:016x}", fnv1a64(&data))),
            )
        })
        .collect::<serde_json::Map<String, serde_json::Value>>();
    let manifest = serde_json::json!({
        "schema": "sonium-sim-room-gallery/v1",
        "scenario": scenario,
        "solver": solver,
        "git_sha": git_sha(),
        "viewport": {"width": 1920, "height": 1080},
        "units": {"frequency": "Hz", "sound_pressure_level": "dB re 20 uPa"},
        "provenance": "frequency_response.csv is the authoritative plotted data; per-bin provenance is explicit",
        "audio": wav_files,
        "checksums": checksums,
    });
    std::fs::write(
        directory.join("gallery-provenance.json"),
        serde_json::to_vec_pretty(&manifest).unwrap(),
    )
    .map_err(|e| e.to_string())
}

/// Emit deterministic comparison panels for an independent solved response.
///
/// The CSV is the authoritative per-sample record. The heatmap retains the
/// complete source/listener/frequency ordering, while the line plot summarizes
/// RMS relative complex-pressure error by frequency. This keeps an external
/// reference visible without turning a scalar pass/fail into a substitute for
/// the underlying transfer-function data. If the gallery response has an
/// explicit reduced/geometric/synthetic tail, only its solved prefix is
/// compared.
fn write_reference_comparison(
    gallery: &mut GalleryRenderer,
    directory: &Path,
    candidate: &SimulationOutput,
    candidate_backend: &str,
    reference_backend: &str,
    reference: &SimulationOutput,
    frequency_tolerance_hz: f64,
    relative_floor_pa: f64,
    reference_identity: Option<(&str, Option<&str>)>,
) -> Result<(), String> {
    let comparison_candidate = solved_reference_candidate(candidate)?;
    let report = crate::backend_parity::compare_backend_outputs(
        reference_backend,
        reference,
        candidate_backend,
        &comparison_candidate,
        frequency_tolerance_hz,
        relative_floor_pa,
    )?;

    let frequency_count = comparison_candidate.frequencies.len();
    let mut per_frequency_sum_squared = vec![0.0_f64; frequency_count];
    let mut per_frequency_counts = vec![0_usize; frequency_count];
    let mut per_frequency_max = vec![0.0_f64; frequency_count];
    let mut all_errors = Vec::with_capacity(report.sample_count);
    let mut csv = String::from(
        "source_index,listener_index,frequency_index,frequency_hz,reference_frequency_hz,reference_spl_db,candidate_spl_db,relative_complex_error,reference_backend,candidate_backend,provenance\n",
    );

    for (source_index, (candidate_source, reference_source)) in comparison_candidate
        .pressures
        .iter()
        .zip(&reference.pressures)
        .enumerate()
    {
        for (listener_index, (candidate_listener, reference_listener)) in
            candidate_source.iter().zip(reference_source).enumerate()
        {
            for frequency_index in 0..frequency_count {
                let candidate_pressure = candidate_listener[frequency_index];
                let reference_pressure = reference_listener[frequency_index];
                let error = (candidate_pressure - reference_pressure).norm()
                    / reference_pressure.norm().max(relative_floor_pa);
                if !error.is_finite() {
                    return Err("reference comparison contains non-finite error".into());
                }
                all_errors.push(error);
                per_frequency_sum_squared[frequency_index] += error * error;
                per_frequency_counts[frequency_index] += 1;
                per_frequency_max[frequency_index] = per_frequency_max[frequency_index].max(error);
                csv.push_str(&format!(
                    "{source_index},{listener_index},{frequency_index},{:.9},{:.9},{:.9},{:.9},{:.12},{},{},{}\n",
                    comparison_candidate.frequencies[frequency_index],
                    reference.frequencies[frequency_index],
                    crate::room_metrics::peak_pressure_to_spl(reference_pressure),
                    crate::room_metrics::peak_pressure_to_spl(candidate_pressure),
                    error,
                    csv_escape(reference_backend),
                    csv_escape(candidate_backend),
                    comparison_candidate.provenance[frequency_index].as_str(),
                ));
            }
        }
    }
    let comparison_csv_name = "reference_comparison.csv";
    std::fs::write(directory.join(comparison_csv_name), csv).map_err(|error| error.to_string())?;

    let rms_by_frequency: Vec<f64> = per_frequency_sum_squared
        .iter()
        .zip(&per_frequency_counts)
        .map(|(sum_squared, count)| (sum_squared / *count as f64).sqrt())
        .collect();
    let per_frequency = comparison_candidate
        .frequencies
        .iter()
        .enumerate()
        .map(|(frequency_index, &frequency_hz)| {
            serde_json::json!({
                "frequency_index": frequency_index,
                "frequency_hz": frequency_hz,
                "rms_relative_error": rms_by_frequency[frequency_index],
                "max_relative_error": per_frequency_max[frequency_index],
                "sample_count": per_frequency_counts[frequency_index],
                "provenance": comparison_candidate.provenance[frequency_index].as_str(),
            })
        })
        .collect::<Vec<_>>();
    let comparison_json = serde_json::json!({
        "schema": "sonium-sim-room-reference-comparison/v1",
        "reference_backend": reference_backend,
        "reference_schema": reference_identity.map(|(schema, _)| schema),
        "reference_content_sha256": reference_identity
            .and_then(|(_, content_sha256)| content_sha256),
        "candidate_backend": candidate_backend,
        "frequency_tolerance_hz": frequency_tolerance_hz,
        "relative_floor_pa": relative_floor_pa,
        "report": {
            "sample_count": report.sample_count,
            "rms_relative_error": report.rms_relative_error,
            "max_relative_error": report.max_relative_error,
        },
        "per_frequency": per_frequency,
        "csv": comparison_csv_name,
    });
    let comparison_json_name = "reference_comparison.json";
    std::fs::write(
        directory.join(comparison_json_name),
        serde_json::to_vec_pretty(&comparison_json).map_err(|error| error.to_string())?,
    )
    .map_err(|error| error.to_string())?;

    let caption = format!(
        "Relative complex-pressure error against {reference_backend}; CSV retains every source/listener/frequency sample"
    );
    gallery
        .heatmap_png(
            "reference_comparison_heatmap.png",
            &all_errors,
            comparison_csv_name,
            &caption,
        )
        .map_err(|error| error.to_string())?;
    gallery
        .line_svg(
            "reference_comparison_error_plot.svg",
            &comparison_candidate.frequencies,
            &rms_by_frequency,
            &format!("RMS error versus {reference_backend}"),
            comparison_json_name,
            "RMS relative complex-pressure error by frequency; see reference_comparison.csv for the full tensor",
        )
        .map_err(|error| error.to_string())?;
    Ok(())
}

/// Compare a retained FEM nodal field against an independently solved planar
/// reference field and emit authoritative per-sample and per-frequency
/// artifacts. The listener-response comparison remains a separate contract;
/// this path compares complex pressure at the physical slice positions.
fn write_reference_field_comparison(
    gallery: &mut GalleryRenderer,
    directory: &Path,
    candidate_field: &FemNodalField,
    candidate_backend: &str,
    reference_backend: &str,
    reference_field: &GalleryReferenceField,
    frequency_tolerance_hz: f64,
    relative_floor_pa: f64,
    reference_identity: Option<(&str, Option<&str>)>,
) -> Result<(), String> {
    candidate_field.validate()?;
    reference_field.validate()?;
    if !frequency_tolerance_hz.is_finite() || frequency_tolerance_hz < 0.0 {
        return Err("reference field frequency tolerance must be finite and non-negative".into());
    }
    if !relative_floor_pa.is_finite() || relative_floor_pa <= 0.0 {
        return Err("reference field relative floor must be finite and positive".into());
    }
    let candidate_source_name = candidate_field
        .source_names
        .get(reference_field.source_index)
        .ok_or_else(|| "reference field source index is absent from FEM field".to_string())?;
    if candidate_source_name != &reference_field.source_name {
        return Err(format!(
            "reference field source mismatch: reference {:?}, FEM {:?}",
            reference_field.source_name, candidate_source_name
        ));
    }
    if candidate_field.frequencies.len() != reference_field.frequencies.len() {
        return Err("reference field frequency counts do not match FEM field".into());
    }
    for (candidate_frequency, reference_frequency) in candidate_field
        .frequencies
        .iter()
        .zip(&reference_field.frequencies)
    {
        if (candidate_frequency - reference_frequency).abs() > frequency_tolerance_hz {
            return Err(format!(
                "reference field frequency mismatch: FEM {candidate_frequency:.9} Hz, reference {reference_frequency:.9} Hz"
            ));
        }
    }

    let mut csv = String::from(
        "frequency_index,frequency_hz,x_m,y_m,z_m,reference_spl_db,candidate_spl_db,relative_complex_error,reference_backend,candidate_backend\n",
    );
    let mut per_frequency = Vec::with_capacity(reference_field.frequencies.len());
    let mut all_errors = Vec::with_capacity(
        reference_field
            .frequencies
            .len()
            .saturating_mul(reference_field.positions.len()),
    );
    let mut reference_values_by_frequency = Vec::with_capacity(reference_field.frequencies.len());
    let mut candidate_values_by_frequency = Vec::with_capacity(reference_field.frequencies.len());

    for (frequency_index, (&frequency_hz, reference_values)) in reference_field
        .frequencies
        .iter()
        .zip(&reference_field.pressures)
        .enumerate()
    {
        let candidate_values = candidate_field.sample_p1(
            reference_field.source_index,
            frequency_index,
            &reference_field.positions,
        )?;
        let mut sum_squared = 0.0;
        let mut max_error = 0.0_f64;
        let mut reference_spl = Vec::with_capacity(reference_values.len());
        let mut candidate_spl = Vec::with_capacity(candidate_values.len());
        for (position, (reference_pressure, candidate_pressure)) in reference_field
            .positions
            .iter()
            .zip(reference_values.iter().zip(&candidate_values))
        {
            let error = (candidate_pressure - reference_pressure).norm()
                / reference_pressure.norm().max(relative_floor_pa);
            if !error.is_finite() {
                return Err("reference field comparison contains non-finite error".into());
            }
            let reference_db = crate::room_metrics::peak_pressure_to_spl(*reference_pressure);
            let candidate_db = crate::room_metrics::peak_pressure_to_spl(*candidate_pressure);
            if !reference_db.is_finite() || !candidate_db.is_finite() {
                return Err("reference field comparison contains non-finite SPL".into());
            }
            csv.push_str(&format!(
                "{frequency_index},{frequency_hz:.9},{:.9},{:.9},{:.9},{reference_db:.9},{candidate_db:.9},{error:.12},{},{}\n",
                position.x,
                position.y,
                position.z,
                csv_escape(reference_backend),
                csv_escape(candidate_backend),
            ));
            reference_spl.push(reference_db);
            candidate_spl.push(candidate_db);
            sum_squared += error * error;
            max_error = max_error.max(error);
            all_errors.push(error);
        }
        let sample_count = reference_values.len();
        per_frequency.push(serde_json::json!({
            "frequency_index": frequency_index,
            "frequency_hz": frequency_hz,
            "sample_count": sample_count,
            "rms_relative_error": (sum_squared / sample_count as f64).sqrt(),
            "max_relative_error": max_error,
        }));
        reference_values_by_frequency.push(reference_spl);
        candidate_values_by_frequency.push(candidate_spl);
    }

    let sample_count = all_errors.len();
    if sample_count == 0 {
        return Err("reference field comparison has no samples".into());
    }
    let rms_relative_error =
        (all_errors.iter().map(|error| error * error).sum::<f64>() / sample_count as f64).sqrt();
    let max_relative_error = all_errors.iter().copied().fold(0.0, f64::max);
    let csv_name = "fem_reference_field_comparison.csv";
    std::fs::write(directory.join(csv_name), csv).map_err(|error| error.to_string())?;

    let mut panels = Vec::with_capacity(reference_field.frequencies.len());
    for (frequency_index, &frequency_hz) in reference_field.frequencies.iter().enumerate() {
        let token = spatial_frequency_token(frequency_index, frequency_hz);
        let reference_png = format!("reference_field_slice_{token}.png");
        let candidate_png = format!("candidate_field_slice_{token}.png");
        let error_png = format!("reference_field_error_{token}.png");
        let caption_prefix =
            format!("{frequency_hz:.3} Hz field comparison against {reference_backend}");
        let reference_caption = format!("Reference-solver FEM field slice; {caption_prefix}");
        let candidate_caption =
            format!("Sonium FEM field slice sampled on the reference grid; {caption_prefix}");
        gallery
            .grid_heatmap_png(
                &reference_png,
                &reference_values_by_frequency[frequency_index],
                reference_field.grid_columns,
                csv_name,
                &reference_caption,
            )
            .map_err(|error| format!("{reference_png}: {error}"))?;
        gallery
            .grid_heatmap_png(
                &candidate_png,
                &candidate_values_by_frequency[frequency_index],
                reference_field.grid_columns,
                csv_name,
                &candidate_caption,
            )
            .map_err(|error| format!("{candidate_png}: {error}"))?;
        let reference_values = &reference_field.pressures[frequency_index];
        let candidate_values = candidate_field.sample_p1(
            reference_field.source_index,
            frequency_index,
            &reference_field.positions,
        )?;
        let errors: Vec<f64> = reference_values
            .iter()
            .zip(&candidate_values)
            .map(|(reference_pressure, candidate_pressure)| {
                (candidate_pressure - reference_pressure).norm()
                    / reference_pressure.norm().max(relative_floor_pa)
            })
            .collect();
        gallery
            .grid_heatmap_png(
                &error_png,
                &errors,
                reference_field.grid_columns,
                csv_name,
                format!("Relative complex-pressure error; {caption_prefix}"),
            )
            .map_err(|error| format!("{error_png}: {error}"))?;
        panels.push(serde_json::json!({
            "frequency_index": frequency_index,
            "frequency_hz": frequency_hz,
            "reference_png": reference_png,
            "candidate_png": candidate_png,
            "error_png": error_png,
            "sample_count": reference_field.positions.len(),
        }));
    }

    let manifest = serde_json::json!({
        "schema": "sonium-sim-room-reference-field-comparison/v1",
        "reference_backend": reference_backend,
        "reference_schema": reference_identity.map(|(schema, _)| schema),
        "reference_content_sha256": reference_identity
            .and_then(|(_, content_sha256)| content_sha256),
        "candidate_backend": candidate_backend,
        "source_index": reference_field.source_index,
        "source_name": reference_field.source_name,
        "plane": {
            "axis": reference_field.plane_axis,
            "coordinate_m": reference_field.plane_coordinate_m,
        },
        "grid": {
            "rows": reference_field.grid_rows,
            "columns": reference_field.grid_columns,
        },
        "frequency_tolerance_hz": frequency_tolerance_hz,
        "relative_floor_pa": relative_floor_pa,
        "report": {
            "sample_count": sample_count,
            "rms_relative_error": rms_relative_error,
            "max_relative_error": max_relative_error,
        },
        "per_frequency": per_frequency,
        "csv": csv_name,
        "panels": panels,
    });
    std::fs::write(
        directory.join("fem_reference_field_comparison.json"),
        serde_json::to_vec_pretty(&manifest).map_err(|error| error.to_string())?,
    )
    .map_err(|error| error.to_string())
}

fn solved_reference_candidate(candidate: &SimulationOutput) -> Result<SimulationOutput, String> {
    let first_non_solved = candidate
        .provenance
        .iter()
        .position(|provenance| *provenance != crate::FrequencyProvenance::Solved);
    let Some(prefix_len) = first_non_solved else {
        return Ok(candidate.clone());
    };
    if prefix_len == 0 {
        return Err("reference comparison requires at least one solved frequency".into());
    }
    if candidate.provenance[prefix_len..].contains(&crate::FrequencyProvenance::Solved) {
        return Err("reference comparison requires non-solved provenance to be a tail".into());
    }
    Ok(SimulationOutput {
        frequencies: candidate.frequencies[..prefix_len].to_vec(),
        pressures: candidate
            .pressures
            .iter()
            .map(|source| {
                source
                    .iter()
                    .map(|listener| listener[..prefix_len].to_vec())
                    .collect()
            })
            .collect(),
        source_names: candidate.source_names.clone(),
        provenance: vec![crate::FrequencyProvenance::Solved; prefix_len],
    })
}

/// Emit one labelled XY SPL slice per retained FEM frequency.
///
/// CSV is authoritative; PNG is a deterministic view of the same samples.
/// Each panel is sampled from the physical-domain P1 nodal field, and the
/// first panel is copied to the historical unlabelled filenames.
fn write_fem_nodal_slice(
    gallery: &mut GalleryRenderer,
    directory: &Path,
    field: &FemNodalField,
    simulation: &RoomSimulation,
    source_index: usize,
    grid_size: usize,
) -> Result<(), String> {
    field.validate()?;
    validate_spatial_slice_frequencies(&field.frequencies)?;

    let (width, depth, _height) = simulation.room.dimensions();
    let z = simulation
        .listening_positions
        .first()
        .ok_or_else(|| "FEM nodal slice requires listener position".to_string())?
        .z;
    let positions = xy_slice_positions(width, depth, z, grid_size);
    let mut panels = Vec::with_capacity(field.frequencies.len());

    for (frequency_index, &frequency_hz) in field.frequencies.iter().enumerate() {
        let token = spatial_frequency_token(frequency_index, frequency_hz);
        let csv_name = format!("fem_pressure_slice_xy_{token}.csv");
        let png_name = format!("fem_pressure_slice_xy_{token}.png");
        let provenance_name = format!("fem_pressure_slice_xy_{token}_provenance.json");
        let vtu_name = format!("fem_pressure_nodal_field_{token}.vtu");
        let pressures = field.sample_p1(source_index, frequency_index, &positions)?;
        let values: Vec<f64> = pressures
            .iter()
            .map(|pressure| crate::room_metrics::peak_pressure_to_spl(*pressure))
            .collect();

        let mut csv = String::from("x_m,y_m,z_m,frequency_hz,spl_db,provenance\n");
        for ((position, spl), pressure) in positions.iter().zip(&values).zip(&pressures) {
            if !spl.is_finite() || !pressure.re.is_finite() || !pressure.im.is_finite() {
                return Err("FEM nodal slice contains non-finite pressure".into());
            }
            csv.push_str(&format!(
                "{:.9},{:.9},{:.9},{frequency_hz:.9},{spl:.9},solved\n",
                position.x, position.y, position.z
            ));
        }
        std::fs::write(directory.join(&csv_name), csv).map_err(|error| error.to_string())?;

        field.write_vtu(directory.join(&vtu_name), source_index, frequency_index)?;

        let provenance = serde_json::json!({
            "schema": "sonium-sim-room-spatial-slice/v2",
            "solver": "fem",
            "frequency_index": frequency_index,
            "frequency_hz": frequency_hz,
            "label": format!("{frequency_hz:.3} Hz"),
            "source_index": source_index,
            "plane": {"axis": "z", "coordinate_m": z},
            "grid": {"rows": grid_size, "columns": grid_size},
            "provenance": "solved",
            "measured": false,
            "fem_nodal_field": true,
            "sampling": "P1 barycentric interpolation from physical-domain FEM nodes",
        });
        std::fs::write(
            directory.join(&provenance_name),
            serde_json::to_vec_pretty(&provenance).map_err(|error| error.to_string())?,
        )
        .map_err(|error| error.to_string())?;

        let caption = format!(
            "Solved FEM physical-domain nodal-pressure XY SPL slice at {frequency_hz:.3} Hz; values are P1-interpolated from the retained complex nodal field"
        );
        gallery
            .smooth_grid_heatmap_png(&png_name, &values, grid_size, &csv_name, &caption)
            .map_err(|error| error.to_string())?;

        panels.push(serde_json::json!({
            "frequency_index": frequency_index,
            "frequency_hz": frequency_hz,
            "label": format!("{frequency_hz:.3} Hz"),
            "csv": csv_name,
            "png": png_name,
            "provenance": provenance_name,
            "vtu": vtu_name,
        }));

        if frequency_index == 0 {
            copy_spatial_slice_alias(
                directory,
                &format!("fem_pressure_slice_xy_{token}.csv"),
                "fem_pressure_slice_xy.csv",
            )?;
            copy_spatial_slice_alias(
                directory,
                &format!("fem_pressure_slice_xy_{token}.png"),
                "fem_pressure_slice_xy.png",
            )?;
            copy_spatial_slice_alias(
                directory,
                &format!("fem_pressure_slice_xy_{token}_provenance.json"),
                "fem_pressure_slice_xy_provenance.json",
            )?;
            copy_spatial_slice_alias(
                directory,
                &format!("fem_pressure_nodal_field_{token}.vtu"),
                "fem_pressure_nodal_field.vtu",
            )?;
        }
    }

    write_spatial_slice_manifest(
        &directory.join("fem_pressure_slice_xy_manifest.json"),
        "fem",
        z,
        grid_size,
        panels,
    )
}
/// Return a Schroeder energy-decay curve in dB relative to its first sample.
///
/// Each point includes the energy at its own time sample, so a non-empty curve
/// begins at 0 dB.  This is the convention used by room-acoustic decay plots
/// and avoids an off-by-one drop when the IR starts with a direct arrival.
fn schroeder_energy_decay_db(samples: &[f64]) -> Result<Vec<f64>, String> {
    if samples.iter().any(|sample| !sample.is_finite()) {
        return Err("gallery impulse response contains non-finite samples".into());
    }
    let total_energy: f64 = samples.iter().map(|sample| sample * sample).sum();
    if !(total_energy.is_finite() && total_energy > 0.0) {
        return Err("gallery impulse response has no finite energy".into());
    }

    let mut remaining_energy = 0.0;
    let mut decay = Vec::with_capacity(samples.len());
    for &sample in samples.iter().rev() {
        remaining_energy += sample * sample;
        decay.push(10.0 * (remaining_energy / total_energy).max(1.0e-12).log10());
    }
    decay.reverse();
    Ok(decay)
}

fn write_spatial_metrics_csv(path: &Path, metrics: &SpatialMetricSeries) -> Result<(), String> {
    let mut csv = String::from(
        "frequency_hz,listener_count,rms_pressure_pa,rms_spl_db,mean_listener_spl_db,min_listener_spl_db,max_listener_spl_db,listener_spread_db\n",
    );
    for (frequency, sample) in metrics.frequencies.iter().zip(&metrics.samples) {
        csv.push_str(&format!(
            "{frequency:.9},{},{:.12},{:.9},{:.9},{:.9},{:.9},{:.9}\n",
            metrics.listener_count,
            sample.rms_pressure_pa,
            sample.rms_spl_db,
            sample.mean_listener_spl_db,
            sample.min_listener_spl_db,
            sample.max_listener_spl_db,
            sample.listener_spread_db,
        ));
    }
    std::fs::write(path, csv).map_err(|error| error.to_string())
}

/// Emit one labelled reduced-order XY pressure slice per requested frequency.
///
/// FEM galleries deliberately do not call this path: a true FEM slice needs
/// a solved nodal field, not a raster inferred from listener responses.
fn write_fast_hybrid_slice(
    gallery: &mut GalleryRenderer,
    directory: &Path,
    simulation: &RoomSimulation,
    frequencies: &[f64],
    grid_size: usize,
) -> Result<(), String> {
    validate_spatial_slice_frequencies(frequencies)?;

    let (width, depth, _height) = simulation.room.dimensions();
    let z = simulation
        .listening_positions
        .first()
        .ok_or_else(|| "fast-hybrid slice requires listener position".to_string())?
        .z;
    let positions = xy_slice_positions(width, depth, z, grid_size);
    let mut panels = Vec::with_capacity(frequencies.len());

    for (frequency_index, &frequency_hz) in frequencies.iter().enumerate() {
        let token = spatial_frequency_token(frequency_index, frequency_hz);
        let csv_name = format!("pressure_slice_xy_{token}.csv");
        let png_name = format!("pressure_slice_xy_{token}.png");
        let provenance_name = format!("pressure_slice_xy_{token}_provenance.json");
        let pressures = sample_fast_hybrid_field(simulation, 0, frequency_hz, &positions)
            .map_err(|error| error.to_string())?;
        let values: Vec<f64> = pressures
            .iter()
            .map(|pressure| crate::room_metrics::peak_pressure_to_spl(*pressure))
            .collect();

        let mut csv = String::from("x_m,y_m,z_m,frequency_hz,spl_db,provenance\n");
        for ((position, spl), pressure) in positions.iter().zip(&values).zip(&pressures) {
            if !spl.is_finite() || !pressure.re.is_finite() || !pressure.im.is_finite() {
                return Err("fast-hybrid non-finite pressure".into());
            }
            csv.push_str(&format!(
                "{:.9},{:.9},{:.9},{frequency_hz:.9},{spl:.9},reduced-order\n",
                position.x, position.y, position.z
            ));
        }
        std::fs::write(directory.join(&csv_name), csv).map_err(|error| error.to_string())?;

        let provenance = serde_json::json!({
            "schema": "sonium-sim-room-spatial-slice/v2",
            "solver": "fast-hybrid",
            "frequency_index": frequency_index,
            "frequency_hz": frequency_hz,
            "label": format!("{frequency_hz:.3} Hz"),
            "source_index": 0,
            "plane": {"axis": "z", "coordinate_m": z},
            "grid": {"rows": grid_size, "columns": grid_size},
            "provenance": "reduced-order",
            "measured": false,
            "fem_nodal_field": false,
            "sampling": "reduced-order fast-hybrid field evaluator",
        });
        std::fs::write(
            directory.join(&provenance_name),
            serde_json::to_vec_pretty(&provenance).map_err(|error| error.to_string())?,
        )
        .map_err(|error| error.to_string())?;

        let caption = format!(
            "Fast-hybrid reduced-order XY SPL slice at {frequency_hz:.3} Hz; not a measured or FEM field"
        );
        gallery
            .smooth_grid_heatmap_png(&png_name, &values, grid_size, &csv_name, &caption)
            .map_err(|error| error.to_string())?;

        panels.push(serde_json::json!({
            "frequency_index": frequency_index,
            "frequency_hz": frequency_hz,
            "label": format!("{frequency_hz:.3} Hz"),
            "csv": csv_name,
            "png": png_name,
            "provenance": provenance_name,
        }));

        if frequency_index == 0 {
            copy_spatial_slice_alias(
                directory,
                &format!("pressure_slice_xy_{token}.csv"),
                "pressure_slice_xy.csv",
            )?;
            copy_spatial_slice_alias(
                directory,
                &format!("pressure_slice_xy_{token}.png"),
                "pressure_slice_xy.png",
            )?;
            copy_spatial_slice_alias(
                directory,
                &format!("pressure_slice_xy_{token}_provenance.json"),
                "pressure_slice_xy_provenance.json",
            )?;
        }
    }

    write_spatial_slice_manifest(
        &directory.join("pressure_slice_xy_manifest.json"),
        "fast-hybrid",
        z,
        grid_size,
        panels,
    )
}

fn validate_spatial_slice_frequencies(frequencies: &[f64]) -> Result<(), String> {
    if frequencies.is_empty()
        || frequencies
            .iter()
            .any(|frequency| !frequency.is_finite() || *frequency <= 0.0)
        || frequencies.windows(2).any(|window| window[1] <= window[0])
    {
        return Err("spatial-slice frequency grid must be finite, positive, and increasing".into());
    }
    Ok(())
}

fn spatial_frequency_token(frequency_index: usize, frequency_hz: f64) -> String {
    format!("f{frequency_index:03}_{frequency_hz:.3}hz").replace('.', "p")
}

fn xy_slice_positions(
    width: f64,
    depth: f64,
    z: f64,
    grid_size: usize,
) -> Vec<sonium_xem::Point3D> {
    let mut positions = Vec::with_capacity(grid_size * grid_size);
    for row in 0..grid_size {
        let y = depth * row as f64 / (grid_size - 1) as f64;
        for column in 0..grid_size {
            let x = width * column as f64 / (grid_size - 1) as f64;
            positions.push(sonium_xem::Point3D::new(x, y, z));
        }
    }
    positions
}

fn copy_spatial_slice_alias(
    directory: &Path,
    source_name: &str,
    target_name: &str,
) -> Result<(), String> {
    if source_name == target_name {
        return Ok(());
    }
    std::fs::copy(directory.join(source_name), directory.join(target_name))
        .map_err(|error| error.to_string())?;
    Ok(())
}

fn write_spatial_slice_manifest(
    path: &Path,
    solver: &str,
    z: f64,
    grid_size: usize,
    panels: Vec<serde_json::Value>,
) -> Result<(), String> {
    let manifest = serde_json::json!({
        "schema": "sonium-sim-room-spatial-slice/v2",
        "solver": solver,
        "plane": {"axis": "z", "coordinate_m": z},
        "grid": {"rows": grid_size, "columns": grid_size},
        "panels": panels,
    });
    std::fs::write(
        path,
        serde_json::to_vec_pretty(&manifest).map_err(|error| error.to_string())?,
    )
    .map_err(|error| error.to_string())
}
/// Estimate RT60 from the standard -5 to -35 dB Schroeder interval.
///
/// A short or lossless synthetic impulse response is valid gallery input but
/// cannot support a decay-time claim; `None` is emitted in its metrics JSON.
fn estimate_rt60_seconds(times: &[f64], decay_db: &[f64]) -> Option<f64> {
    if times.len() != decay_db.len() || times.len() < 2 {
        return None;
    }
    let mut start = None;
    let mut end = None;
    for (index, &decay) in decay_db.iter().enumerate().skip(1) {
        if start.is_none() && decay <= -5.0 {
            start = Some(index);
        }
        if end.is_none() && decay <= -35.0 {
            end = Some(index);
            break;
        }
    }
    let (start, end) = (start?, end?);
    if end <= start || !times[start].is_finite() || !times[end].is_finite() {
        return None;
    }
    let duration = times[end] - times[start];
    if duration.is_finite() && duration > 0.0 {
        Some(duration * 2.0)
    } else {
        None
    }
}

fn write_room_geometry_csv(path: &Path, simulation: &RoomSimulation) -> Result<(), String> {
    let (width, depth, height) = simulation.room.dimensions();
    let mut csv = format!(
        "kind,index,name,x_m,y_m,z_m,width_m,depth_m,height_m\nroom,0,room,,,,{width:.9},{depth:.9},{height:.9}\n"
    );
    for (index, source) in simulation.sources.iter().enumerate() {
        csv.push_str(&format!(
            "source,{index},{},{:.9},{:.9},{:.9},,,\n",
            csv_escape(&source.name),
            source.position.x,
            source.position.y,
            source.position.z
        ));
    }
    for (index, listener) in simulation.listening_positions.iter().enumerate() {
        csv.push_str(&format!(
            "listener,{index},listener-{index},{:.9},{:.9},{:.9},,,\n",
            listener.x, listener.y, listener.z
        ));
    }
    std::fs::write(path, csv).map_err(|error| error.to_string())
}

fn write_gallery_shell(directory: &Path, scenario: &str) -> Result<(), String> {
    let title = html_escape(scenario);
    let shell = include_str!("../assets/room_gallery.html").replace("{{SCENARIO}}", &title);
    std::fs::write(directory.join("index.html"), shell).map_err(|error| error.to_string())
}

fn html_escape(value: &str) -> String {
    value
        .replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&#39;")
}

/// Escape one RFC-4180 CSV field. Geometry/source names originate in user
/// configuration and may contain commas, quotes, or line breaks; leaving them
/// raw would make the provenance file ambiguous to external readers.
fn csv_escape(value: &str) -> String {
    if value
        .bytes()
        .any(|byte| matches!(byte, b',' | b'"' | b'\n' | b'\r'))
    {
        format!("\"{}\"", value.replace('"', "\"\""))
    } else {
        value.to_owned()
    }
}

fn write_room_layout_svg(
    path: &Path,
    simulation: &RoomSimulation,
    scenario: &str,
) -> Result<(), String> {
    let (width, depth, height) = simulation.room.dimensions();
    if !(width > 0.0 && depth > 0.0 && height > 0.0) {
        return Err("gallery room dimensions must be positive".into());
    }
    let sx = |x: f64| 180.0 + 1560.0 * (x / width).clamp(0.0, 1.0);
    let sy = |y: f64| 900.0 - 720.0 * (y / depth).clamp(0.0, 1.0);
    let sources = simulation
        .sources
        .iter()
        .enumerate()
        .map(|(index, source)| format!(
            "<circle cx=\"{:.2}\" cy=\"{:.2}\" r=\"18\" fill=\"#f9c74f\"/><text x=\"{:.2}\" y=\"{:.2}\" fill=\"#101820\" font-size=\"18\">S{index}</text>",
            sx(source.position.x), sy(source.position.y), sx(source.position.x) - 7.0, sy(source.position.y) + 6.0
        ))
        .collect::<String>();
    let listeners = simulation
        .listening_positions
        .iter()
        .enumerate()
        .map(|(index, listener)| format!(
            "<circle cx=\"{:.2}\" cy=\"{:.2}\" r=\"18\" fill=\"#f94144\"/><text x=\"{:.2}\" y=\"{:.2}\" fill=\"white\" font-size=\"18\">L{index}</text>",
            sx(listener.x), sy(listener.y), sx(listener.x) - 7.0, sy(listener.y) + 6.0
        ))
        .collect::<String>();
    let svg = format!(
        "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"1920\" height=\"1080\" viewBox=\"0 0 1920 1080\"><rect width=\"100%\" height=\"100%\" fill=\"#101820\"/><text x=\"120\" y=\"100\" fill=\"white\" font-size=\"44\" font-family=\"sans-serif\">Room layout: {scenario}</text><rect x=\"180\" y=\"180\" width=\"1560\" height=\"720\" fill=\"#25445a\" stroke=\"#dfe7ef\" stroke-width=\"8\"/>{sources}{listeners}<text x=\"120\" y=\"1010\" fill=\"#dfe7ef\" font-size=\"28\" font-family=\"sans-serif\">yellow S = source · red L = listener · dimensions {width:.2} × {depth:.2} × {height:.2} m</text></svg>"
    );
    std::fs::write(path, svg).map_err(|error| error.to_string())
}

fn git_sha() -> String {
    std::process::Command::new("git")
        .args(["rev-parse", "HEAD"])
        .output()
        .ok()
        .filter(|out| out.status.success())
        .and_then(|out| String::from_utf8(out.stdout).ok())
        .map(|sha| sha.trim().to_owned())
        .unwrap_or_else(|| "unknown".to_owned())
}

fn fnv1a64(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf29ce484222325_u64, |hash, byte| {
        (hash ^ u64::from(*byte)).wrapping_mul(0x100000001b3)
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{FrequencyProvenance, SimulationOutput, scenarios};
    use num_complex::Complex64;
    use sonium_xem::io::{
        gallery::{GALLERY_HEIGHT, GALLERY_WIDTH},
        validate_png_nonblank,
    };

    fn assert_slice_manifest(
        directory: &std::path::Path,
        manifest_name: &str,
        expected_frequencies: &[f64],
        require_vtu: bool,
    ) {
        let manifest: serde_json::Value =
            serde_json::from_slice(&std::fs::read(directory.join(manifest_name)).unwrap()).unwrap();
        assert_eq!(manifest["schema"], "sonium-sim-room-spatial-slice/v2");
        let panels = manifest["panels"].as_array().unwrap();
        assert_eq!(panels.len(), expected_frequencies.len());

        for (frequency_index, (&frequency_hz, panel)) in
            expected_frequencies.iter().zip(panels).enumerate()
        {
            assert_eq!(
                panel["frequency_index"].as_u64(),
                Some(frequency_index as u64)
            );
            assert_eq!(panel["frequency_hz"].as_f64(), Some(frequency_hz));
            assert_eq!(panel["label"], format!("{frequency_hz:.3} Hz"));

            let csv_name = panel["csv"].as_str().unwrap();
            let png_name = panel["png"].as_str().unwrap();
            let provenance_name = panel["provenance"].as_str().unwrap();
            assert!(directory.join(csv_name).is_file());
            assert!(directory.join(png_name).is_file());
            assert!(directory.join(provenance_name).is_file());
            let gallery_provenance: serde_json::Value = serde_json::from_slice(
                &std::fs::read(directory.join("gallery-provenance.json")).unwrap(),
            )
            .unwrap();
            let checksums = &gallery_provenance["checksums"];
            for artifact_name in [csv_name, png_name, provenance_name] {
                assert!(
                    checksums[artifact_name]
                        .as_str()
                        .unwrap()
                        .starts_with("fnv1a64:")
                );
            }
            if require_vtu {
                let vtu_name = panel["vtu"].as_str().unwrap();
                assert!(directory.join(vtu_name).is_file());
                assert!(
                    checksums[vtu_name]
                        .as_str()
                        .unwrap()
                        .starts_with("fnv1a64:")
                );
            }

            let sidecar: serde_json::Value =
                serde_json::from_slice(&std::fs::read(directory.join(provenance_name)).unwrap())
                    .unwrap();
            assert_eq!(
                sidecar["frequency_index"].as_u64(),
                Some(frequency_index as u64)
            );
            assert_eq!(sidecar["frequency_hz"].as_f64(), Some(frequency_hz));
            assert_eq!(sidecar["label"], format!("{frequency_hz:.3} Hz"));
            assert_eq!(sidecar["fem_nodal_field"], require_vtu);
        }
    }

    #[test]
    fn gallery_writes_manifest_and_explicit_provenance() {
        let output = SimulationOutput::with_uniform_provenance(
            vec![100.0, 200.0],
            vec![vec![vec![
                Complex64::new(1.0, 0.0),
                Complex64::new(0.5, 0.0),
            ]]],
            vec!["source".into()],
            FrequencyProvenance::Solved,
        );
        let directory = tempfile::tempdir().unwrap();
        let simulation = scenarios::scenario_by_name("small_stereo_2_0")
            .unwrap()
            .simulation;
        write_gallery(
            &output,
            directory.path(),
            "fixture",
            "fem",
            &simulation,
            &GalleryOptions::default(),
        )
        .unwrap();
        assert!(directory.path().join("gallery.json").is_file());
        assert_eq!(
            validate_png_nonblank(directory.path().join("frequency_response_heatmap.png")).unwrap(),
            (GALLERY_WIDTH, GALLERY_HEIGHT)
        );
        let manifest =
            sonium_xem::io::validate_gallery_manifest(directory.path().join("gallery.json"))
                .unwrap();
        assert_eq!(manifest.width_px, GALLERY_WIDTH);
        assert_eq!(manifest.height_px, GALLERY_HEIGHT);
        let manifest_json: serde_json::Value =
            serde_json::from_slice(&std::fs::read(directory.path().join("gallery.json")).unwrap())
                .unwrap();
        let figures = manifest_json["figures"].as_array().unwrap();
        assert!(figures.len() >= 9, "gallery must retain every native view");
        let mut rasters = 0;
        let mut vectors = 0;
        for figure in figures {
            let filename = figure["filename"].as_str().unwrap();
            assert!(
                figure["source_checksum_fnv1a64"].is_string(),
                "figure {filename} must retain source provenance"
            );
            let artifact = directory.path().join(filename);
            assert!(artifact.is_file(), "missing gallery figure {filename}");
            match figure["kind"].as_str().unwrap() {
                "png" => {
                    assert_eq!(
                        validate_png_nonblank(&artifact).unwrap(),
                        (GALLERY_WIDTH, GALLERY_HEIGHT),
                        "raster figure {filename} must be nonblank and fixed-size"
                    );
                    rasters += 1;
                }
                "svg" => {
                    let svg = std::fs::read_to_string(&artifact).unwrap();
                    assert!(
                        svg.contains("<svg"),
                        "SVG figure {filename} is not structured"
                    );
                    assert!(
                        svg.contains("width=\"1920\""),
                        "SVG figure {filename} is not fixed-size"
                    );
                    vectors += 1;
                }
                other => panic!("unexpected gallery figure kind {other}"),
            }
        }
        assert!(rasters >= 3, "expected response and waterfall rasters");
        assert!(
            vectors >= 6,
            "expected room, response, and spatial SVG views"
        );
        assert!(
            std::fs::read_to_string(directory.path().join("frequency_response.csv"))
                .unwrap()
                .contains("solved")
        );
        assert!(
            std::fs::read_to_string(directory.path().join("source_listener_response.csv"))
                .unwrap()
                .contains("source_index,listener_index,frequency_hz,spl_db,provenance")
        );
        assert!(directory.path().join("source_lp0.wav").is_file());
        assert!(directory.path().join("room_layout.svg").is_file());
        assert!(
            directory
                .path()
                .join("listener_frequency_waterfall.png")
                .is_file()
        );
        assert!(directory.path().join("spatial_metrics.csv").is_file());
        assert!(directory.path().join("room_metrics.json").is_file());
        let shell = std::fs::read_to_string(directory.path().join("index.html")).unwrap();
        assert!(shell.contains("gallery.json"));
        assert!(shell.contains("frequency_response.csv"));
        assert!(shell.contains("source_listener_response.csv"));
        assert!(shell.contains("room_metrics.json"));
        assert!(shell.contains("figure.filename"));
        assert!(shell.contains("figure.caption"));
        assert!(shell.contains("figure.source"));
        assert!(shell.contains("provenance.audio"));
        assert!(shell.contains("id=\"provenance\""));
        assert!(shell.contains("id=\"frequency-provenance\""));
        assert!(shell.contains("Frequency provenance by bin"));
        assert!(shell.contains("provenance.provenance"));
        assert!(!shell.contains("figure.path"));
        assert!(!shell.contains("figure.name"));
        assert!(!shell.contains("figure.description"));
        assert!(shell.contains("fixture"));
        assert!(
            std::fs::read_to_string(directory.path().join("room_geometry.csv"))
                .unwrap()
                .contains("listener")
        );
    }

    #[test]
    fn gallery_shell_contains_fail_visible_artifact_integrity_verifier() {
        let shell = include_str!("../assets/room_gallery.html");
        for marker in [
            "id=\"integrity\"",
            "verifyArtifactChecksums",
            "fnv1a64:",
            "data-status",
            "Artifact integrity failed",
            "Manifest withheld until artifact integrity passes",
            "await verifyArtifactChecksums",
        ] {
            assert!(shell.contains(marker), "gallery shell missing {marker}");
        }
    }

    #[test]
    fn reference_loader_requires_versioned_schema_and_validates_tensor() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("reference.json");
        std::fs::write(
            &path,
            serde_json::json!({
                "schema": REFERENCE_RESPONSE_SCHEMA_V1,
                "backend": "freefem",
                "frequencies": [100.0, 200.0],
                "source_names": ["source"],
                "pressures": [[[
                    {"real_pa": 1.0, "imag_pa": 0.0},
                    {"real_pa": 0.5, "imag_pa": 0.0}
                ]]]
            })
            .to_string(),
        )
        .unwrap();

        let reference = load_reference_response_json(&path).unwrap();
        assert_eq!(reference.backend, "freefem");
        assert_eq!(reference.output.frequencies, vec![100.0, 200.0]);
        assert_eq!(
            reference.output.pressures[0][0][1],
            Complex64::new(0.5, 0.0)
        );
        assert!(
            reference
                .output
                .provenance
                .iter()
                .all(|provenance| *provenance == FrequencyProvenance::Solved)
        );

        let invalid_path = directory.path().join("invalid.json");
        std::fs::write(
            &invalid_path,
            serde_json::json!({
                "schema": "wrong",
                "backend": "freefem",
                "frequencies": [100.0],
                "source_names": ["source"],
                "pressures": [[[{"real_pa": 1.0, "imag_pa": 0.0}]]]
            })
            .to_string(),
        )
        .unwrap();
        assert!(
            load_reference_response_json(invalid_path)
                .unwrap_err()
                .contains("unsupported reference response schema")
        );
    }

    #[test]
    fn reference_loader_preserves_checked_spatial_field() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("reference-with-field.json");
        std::fs::write(
            &path,
            serde_json::json!({
                "schema": REFERENCE_RESPONSE_SCHEMA_V1,
                "backend": "freefem",
                "frequencies": [100.0],
                "source_names": ["source"],
                "pressures": [[[{"real_pa": 1.0, "imag_pa": 0.0}]]],
                "spatial_field": {
                    "schema": REFERENCE_SPATIAL_FIELD_SCHEMA_V1,
                    "source_index": 0,
                    "source_name": "source",
                    "plane_axis": "z",
                    "plane_coordinate_m": 0.5,
                    "grid_rows": 2,
                    "grid_columns": 2,
                    "positions": [
                        {"x_m": 0.0, "y_m": 0.0, "z_m": 0.5},
                        {"x_m": 1.0, "y_m": 0.0, "z_m": 0.5},
                        {"x_m": 0.0, "y_m": 1.0, "z_m": 0.5},
                        {"x_m": 1.0, "y_m": 1.0, "z_m": 0.5}
                    ],
                    "frequencies": [100.0],
                    "pressures": [[
                        {"real_pa": 1.0, "imag_pa": 0.0},
                        {"real_pa": 2.0, "imag_pa": 0.0},
                        {"real_pa": 3.0, "imag_pa": 0.0},
                        {"real_pa": 4.0, "imag_pa": 0.0}
                    ]]
                }
            })
            .to_string(),
        )
        .unwrap();

        let reference = load_reference_response_json(path).unwrap();
        let field = reference.spatial_field.as_ref().unwrap();
        assert_eq!(field.source_name, "source");
        assert_eq!(field.grid_rows, 2);
        assert_eq!(field.grid_columns, 2);
        assert_eq!(field.frequencies, vec![100.0]);
        assert_eq!(field.pressures[0][3], Complex64::new(4.0, 0.0));

        let invalid_path = directory.path().join("reference-with-invalid-field.json");
        let mut invalid_document: serde_json::Value = serde_json::from_slice(
            &std::fs::read(directory.path().join("reference-with-field.json")).unwrap(),
        )
        .unwrap();
        invalid_document["spatial_field"]["schema"] = serde_json::Value::String("wrong".into());
        std::fs::write(
            &invalid_path,
            serde_json::to_vec(&invalid_document).unwrap(),
        )
        .unwrap();
        assert!(
            load_reference_response_json(invalid_path)
                .unwrap_err()
                .contains("unsupported reference spatial field schema")
        );

        invalid_document["spatial_field"]["schema"] =
            serde_json::Value::String(REFERENCE_SPATIAL_FIELD_SCHEMA_V1.into());
        invalid_document["source_names"][0] = serde_json::Value::String("other".into());
        let mismatched_path = directory
            .path()
            .join("reference-with-mismatched-field.json");
        std::fs::write(
            &mismatched_path,
            serde_json::to_vec(&invalid_document).unwrap(),
        )
        .unwrap();
        assert!(
            load_reference_response_json(mismatched_path)
                .unwrap_err()
                .contains("disagrees with response source axis")
        );
    }

    #[test]
    fn digest_bound_reference_v2_accepts_valid_payload_and_rejects_tampering() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("reference-v2.json");
        let mut document = ReferenceResponseDocument {
            schema: REFERENCE_RESPONSE_SCHEMA_V2.into(),
            backend: "FreeFEM".into(),
            frequencies: vec![100.0],
            source_names: vec!["source".into()],
            pressures: vec![vec![vec![ReferencePressure {
                real_pa: 1.0,
                imag_pa: -0.25,
            }]]],
            spatial_field: Some(ReferenceSpatialFieldDocument {
                schema: REFERENCE_SPATIAL_FIELD_SCHEMA_V1.into(),
                source_index: 0,
                source_name: "source".into(),
                plane_axis: "z".into(),
                plane_coordinate_m: 0.5,
                grid_rows: 2,
                grid_columns: 2,
                positions: vec![
                    ReferencePoint {
                        x_m: 0.0,
                        y_m: 0.0,
                        z_m: 0.5,
                    },
                    ReferencePoint {
                        x_m: 1.0,
                        y_m: 0.0,
                        z_m: 0.5,
                    },
                    ReferencePoint {
                        x_m: 0.0,
                        y_m: 1.0,
                        z_m: 0.5,
                    },
                    ReferencePoint {
                        x_m: 1.0,
                        y_m: 1.0,
                        z_m: 0.5,
                    },
                ],
                frequencies: vec![100.0],
                pressures: vec![vec![
                    ReferencePressure {
                        real_pa: 1.0,
                        imag_pa: 0.0,
                    },
                    ReferencePressure {
                        real_pa: 1.1,
                        imag_pa: 0.0,
                    },
                    ReferencePressure {
                        real_pa: 1.2,
                        imag_pa: 0.0,
                    },
                    ReferencePressure {
                        real_pa: 1.3,
                        imag_pa: 0.0,
                    },
                ]],
            }),
            content_sha256: None,
            problem_sha256: None,
        };
        document.content_sha256 = Some(canonical_reference_response_sha256(&document).unwrap());
        document.problem_sha256 = Some("NOT-A-SHA256".into());
        std::fs::write(&path, serde_json::to_vec_pretty(&document).unwrap()).unwrap();
        assert!(
            load_reference_response_json(&path)
                .unwrap_err()
                .contains("problem_sha256 must be a lowercase SHA-256 digest")
        );
        document.problem_sha256 = None;
        std::fs::write(&path, serde_json::to_vec_pretty(&document).unwrap()).unwrap();

        let loaded = load_reference_response_json(&path).unwrap();
        assert_eq!(loaded.backend, "FreeFEM");

        document.content_sha256 = None;
        std::fs::write(&path, serde_json::to_vec_pretty(&document).unwrap()).unwrap();
        assert!(
            load_reference_response_json(&path)
                .unwrap_err()
                .contains("v2 requires content_sha256")
        );
        document.content_sha256 = Some(canonical_reference_response_sha256(&document).unwrap());
        std::fs::write(&path, serde_json::to_vec_pretty(&document).unwrap()).unwrap();

        let mut tampered: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
        tampered["spatial_field"]["pressures"][0][0]["real_pa"] = serde_json::json!(2.0);
        std::fs::write(&path, serde_json::to_vec_pretty(&tampered).unwrap()).unwrap();
        let error = load_reference_response_json(&path).unwrap_err();
        assert!(error.contains("content_sha256 mismatch"));
    }

    #[test]
    fn loaded_reference_gallery_preserves_verified_identity() {
        let candidate = SimulationOutput::solved(
            vec![100.0, 200.0],
            vec![vec![vec![
                Complex64::new(1.01, 0.0),
                Complex64::new(0.81, 0.0),
            ]]],
            vec!["source".into()],
        );
        let mut document = ReferenceResponseDocument {
            schema: REFERENCE_RESPONSE_SCHEMA_V2.into(),
            backend: "FreeFEM".into(),
            frequencies: vec![100.0, 200.0],
            source_names: vec!["source".into()],
            pressures: vec![vec![vec![
                ReferencePressure {
                    real_pa: 1.0,
                    imag_pa: 0.0,
                },
                ReferencePressure {
                    real_pa: 0.8,
                    imag_pa: 0.0,
                },
            ]]],
            spatial_field: None,
            content_sha256: None,
            problem_sha256: None,
        };
        document.content_sha256 = Some(canonical_reference_response_sha256(&document).unwrap());

        let directory = tempfile::tempdir().unwrap();
        let reference_path = directory.path().join("reference-v2.json");
        std::fs::write(
            &reference_path,
            serde_json::to_vec_pretty(&document).unwrap(),
        )
        .unwrap();
        let loaded = load_reference_response_json(&reference_path).unwrap();
        let gallery_directory = directory.path().join("gallery");
        let simulation = scenarios::scenario_by_name("small_stereo_2_0")
            .unwrap()
            .simulation;

        write_gallery_with_loaded_reference(
            &candidate,
            &gallery_directory,
            "fixture",
            "fem",
            &simulation,
            &loaded,
            0.0,
            20.0e-6,
            &GalleryOptions::default(),
        )
        .unwrap();

        let comparison: serde_json::Value = serde_json::from_slice(
            &std::fs::read(gallery_directory.join("reference_comparison.json")).unwrap(),
        )
        .unwrap();
        assert_eq!(comparison["reference_schema"], REFERENCE_RESPONSE_SCHEMA_V2);
        assert_eq!(
            comparison["reference_content_sha256"],
            loaded.content_sha256.as_deref().unwrap()
        );
    }

    #[test]
    fn reference_comparison_rejects_unsolved_output_before_writing_artifacts() {
        let candidate = SimulationOutput::with_uniform_provenance(
            vec![100.0],
            vec![vec![vec![Complex64::new(1.0, 0.0)]]],
            vec!["source".into()],
            FrequencyProvenance::ReducedOrder,
        );
        let reference = SimulationOutput::solved(
            vec![100.0],
            vec![vec![vec![Complex64::new(1.0, 0.0)]]],
            vec!["source".into()],
        );
        let directory = tempfile::tempdir().unwrap().path().join("gallery");
        let simulation = scenarios::scenario_by_name("small_stereo_2_0")
            .unwrap()
            .simulation;

        let error = write_gallery_with_reference(
            &candidate,
            &directory,
            "fixture",
            "fast-hybrid",
            &simulation,
            "freefem",
            &reference,
            0.0,
            20.0e-6,
            &GalleryOptions::default(),
        )
        .unwrap_err();
        assert!(error.contains("at least one solved frequency"));
        assert!(!directory.exists());
    }

    #[test]
    fn gallery_writes_external_reference_comparison_panels() {
        let candidate = SimulationOutput {
            frequencies: vec![100.0, 200.0, 400.0],
            pressures: vec![vec![vec![
                Complex64::new(1.0, 0.0),
                Complex64::new(0.5, 0.0),
                Complex64::new(0.25, 0.0),
            ]]],
            source_names: vec!["source".into()],
            provenance: vec![
                FrequencyProvenance::Solved,
                FrequencyProvenance::Solved,
                FrequencyProvenance::Synthetic,
            ],
        };
        let reference = SimulationOutput::solved(
            vec![100.0, 200.0],
            vec![vec![vec![
                Complex64::new(1.01, 0.0),
                Complex64::new(0.49, 0.0),
            ]]],
            vec!["source".into()],
        );
        let directory = tempfile::tempdir().unwrap();
        let simulation = scenarios::scenario_by_name("small_stereo_2_0")
            .unwrap()
            .simulation;

        write_gallery_with_reference(
            &candidate,
            directory.path(),
            "fixture",
            "fem",
            &simulation,
            "freefem",
            &reference,
            0.0,
            20.0e-6,
            &GalleryOptions::default(),
        )
        .unwrap();

        let comparison: serde_json::Value = serde_json::from_slice(
            &std::fs::read(directory.path().join("reference_comparison.json")).unwrap(),
        )
        .unwrap();
        assert_eq!(
            comparison["schema"],
            "sonium-sim-room-reference-comparison/v1"
        );
        assert_eq!(comparison["reference_backend"], "freefem");
        assert_eq!(comparison["candidate_backend"], "fem");
        assert_eq!(comparison["report"]["sample_count"], 2);
        assert_eq!(comparison["per_frequency"].as_array().unwrap().len(), 2);
        assert!(directory.path().join("reference_comparison.csv").is_file());
        assert!(
            directory
                .path()
                .join("reference_comparison_error_plot.svg")
                .is_file()
        );
        assert_eq!(
            validate_png_nonblank(directory.path().join("reference_comparison_heatmap.png"))
                .unwrap(),
            (GALLERY_WIDTH, GALLERY_HEIGHT)
        );
        assert!(
            std::fs::read_to_string(directory.path().join("reference_comparison.csv"))
                .unwrap()
                .contains("reference_backend")
        );
        let manifest = serde_json::from_slice::<serde_json::Value>(
            &std::fs::read(directory.path().join("gallery.json")).unwrap(),
        )
        .unwrap();
        let figures = manifest["figures"].as_array().unwrap();
        assert!(
            figures
                .iter()
                .any(|figure| { figure["filename"] == "reference_comparison_heatmap.png" })
        );
        assert!(
            figures
                .iter()
                .any(|figure| { figure["filename"] == "reference_comparison_error_plot.svg" })
        );
    }

    #[test]
    fn gallery_artifacts_are_byte_deterministic_for_same_output() {
        let output = SimulationOutput::with_uniform_provenance(
            vec![100.0, 200.0],
            vec![vec![vec![
                Complex64::new(1.0, 0.0),
                Complex64::new(0.5, 0.0),
            ]]],
            vec!["source".into()],
            FrequencyProvenance::Solved,
        );
        let simulation = scenarios::scenario_by_name("small_stereo_2_0")
            .unwrap()
            .simulation;
        let root = tempfile::tempdir().unwrap();
        let first = root.path().join("first");
        let second = root.path().join("second");
        let options = GalleryOptions::default();
        write_gallery(&output, &first, "fixture", "fem", &simulation, &options).unwrap();
        write_gallery(&output, &second, "fixture", "fem", &simulation, &options).unwrap();

        let collect = |directory: &Path| {
            std::fs::read_dir(directory)
                .unwrap()
                .map(|entry| {
                    let entry = entry.unwrap();
                    (entry.file_name(), std::fs::read(entry.path()).unwrap())
                })
                .collect::<std::collections::BTreeMap<_, _>>()
        };
        assert_eq!(collect(&first), collect(&second));
    }

    #[test]
    fn csv_escape_quotes_source_names() {
        assert_eq!(csv_escape("plain"), "plain");
        assert_eq!(
            csv_escape(r#"front, "A""#),
            String::from("\"front, \"\"A\"\"\"")
        );
        assert_eq!(csv_escape("line\nfeed"), "\"line\nfeed\"");
    }

    #[test]
    fn fast_gallery_emits_explicit_reduced_order_slice_provenance() {
        let output = SimulationOutput::with_uniform_provenance(
            vec![100.0, 200.0],
            vec![vec![vec![
                Complex64::new(1.0, 0.0),
                Complex64::new(0.5, 0.0),
            ]]],
            vec!["source".into()],
            FrequencyProvenance::ReducedOrder,
        );
        let directory = tempfile::tempdir().unwrap();
        let simulation = scenarios::scenario_by_name("small_stereo_2_0")
            .unwrap()
            .simulation;

        write_gallery(
            &output,
            directory.path(),
            "fast-fixture",
            "fast-hybrid",
            &simulation,
            &GalleryOptions::default(),
        )
        .unwrap();

        assert!(
            std::fs::read_to_string(directory.path().join("pressure_slice_xy.csv"))
                .unwrap()
                .contains("reduced-order")
        );
        let provenance: serde_json::Value = serde_json::from_slice(
            &std::fs::read(directory.path().join("pressure_slice_xy_provenance.json")).unwrap(),
        )
        .unwrap();
        assert_eq!(provenance["provenance"], "reduced-order");
        assert_eq!(provenance["measured"], false);
        assert_eq!(provenance["fem_nodal_field"], false);
        assert_slice_manifest(
            directory.path(),
            "pressure_slice_xy_manifest.json",
            &[100.0, 200.0],
            false,
        );
        assert_eq!(
            validate_png_nonblank(directory.path().join("pressure_slice_xy.png")).unwrap(),
            (GALLERY_WIDTH, GALLERY_HEIGHT)
        );
    }

    #[test]
    fn fem_gallery_emits_nodal_slice_and_vtu_provenance() {
        let mut simulation = scenarios::scenario_by_name("small_stereo_2_0")
            .unwrap()
            .simulation;
        simulation.frequencies.truncate(2);
        let mesh = crate::fem_runner::build_full_domain_room_mesh(
            &simulation,
            &crate::fem_runner::FemMeshConfig::default(),
        )
        .unwrap();
        let first_values: Vec<Complex64> = mesh
            .nodes
            .iter()
            .map(|point| Complex64::new(1.0 + point.x + 0.5 * point.y, 0.0))
            .collect();
        let second_values: Vec<Complex64> = first_values.iter().map(|value| *value * 0.8).collect();
        let field = FemNodalField {
            pressures: vec![vec![first_values, second_values]],
            mesh,
            frequencies: simulation.frequencies.clone(),
            source_names: vec!["source".into()],
            listener_positions: simulation.listening_positions.clone(),
        };
        let output = SimulationOutput::with_uniform_provenance(
            simulation.frequencies.clone(),
            vec![vec![vec![
                Complex64::new(1.0, 0.0),
                Complex64::new(0.8, 0.0),
            ]]],
            vec!["source".into()],
            FrequencyProvenance::Solved,
        );
        let (width, depth, _) = simulation.room.dimensions();
        let z = simulation.listening_positions[0].z;
        let reference_positions = xy_slice_positions(width, depth, z, 32);
        let reference_pressures = (0..field.frequencies.len())
            .map(|frequency_index| {
                field
                    .sample_p1(0, frequency_index, &reference_positions)
                    .unwrap()
                    .into_iter()
                    .enumerate()
                    .map(|(sample_index, pressure)| {
                        pressure * if sample_index % 2 == 0 { 0.8 } else { 1.2 }
                    })
                    .collect::<Vec<_>>()
            })
            .collect();
        let reference_field = GalleryReferenceField {
            source_index: 0,
            source_name: "source".into(),
            plane_axis: "z".into(),
            plane_coordinate_m: z,
            grid_rows: 32,
            grid_columns: 32,
            positions: reference_positions,
            frequencies: field.frequencies.clone(),
            pressures: reference_pressures,
        };
        let reference_output = SimulationOutput::solved(
            simulation.frequencies.clone(),
            vec![vec![vec![
                Complex64::new(1.01, 0.0),
                Complex64::new(0.79, 0.0),
            ]]],
            vec!["source".into()],
        );
        let directory = tempfile::tempdir().unwrap();
        write_gallery_with_fem_field_and_reference(
            &output,
            directory.path(),
            "fem-fixture",
            "fem",
            &simulation,
            &field,
            "freefem",
            &reference_output,
            0.0,
            20.0e-6,
            Some(&reference_field),
            &GalleryOptions::default(),
        )
        .unwrap();

        assert!(directory.path().join("fem_pressure_slice_xy.csv").is_file());
        assert!(
            directory
                .path()
                .join("fem_pressure_nodal_field.vtu")
                .is_file()
        );
        sonium_xem::io::vtu::validate_vtu_file(
            directory.path().join("fem_pressure_nodal_field.vtu"),
        )
        .unwrap();
        assert_eq!(
            validate_png_nonblank(directory.path().join("fem_pressure_slice_xy.png")).unwrap(),
            (GALLERY_WIDTH, GALLERY_HEIGHT)
        );
        assert!(
            directory
                .path()
                .join("fem_reference_field_comparison.csv")
                .is_file()
        );
        let comparison: serde_json::Value = serde_json::from_slice(
            &std::fs::read(directory.path().join("fem_reference_field_comparison.json")).unwrap(),
        )
        .unwrap();
        assert_eq!(
            comparison["schema"],
            "sonium-sim-room-reference-field-comparison/v1"
        );
        assert_eq!(comparison["report"]["sample_count"], 2 * 32 * 32);
        assert!(comparison["report"]["rms_relative_error"].as_f64().unwrap() > 0.0);
        assert!(comparison["report"]["max_relative_error"].as_f64().unwrap() > 0.0);
        for frequency_index in 0..2 {
            let token =
                spatial_frequency_token(frequency_index, field.frequencies[frequency_index]);
            assert_eq!(
                validate_png_nonblank(
                    directory
                        .path()
                        .join(format!("reference_field_error_{token}.png"))
                )
                .unwrap(),
                (GALLERY_WIDTH, GALLERY_HEIGHT)
            );
            assert!(
                directory
                    .path()
                    .join(format!("reference_field_slice_{token}.png"))
                    .is_file()
            );
            assert!(
                directory
                    .path()
                    .join(format!("candidate_field_slice_{token}.png"))
                    .is_file()
            );
        }
        let provenance: serde_json::Value = serde_json::from_slice(
            &std::fs::read(
                directory
                    .path()
                    .join("fem_pressure_slice_xy_provenance.json"),
            )
            .unwrap(),
        )
        .unwrap();
        assert_eq!(provenance["provenance"], "solved");
        assert_eq!(provenance["measured"], false);
        assert_eq!(provenance["fem_nodal_field"], true);
        assert_slice_manifest(
            directory.path(),
            "fem_pressure_slice_xy_manifest.json",
            &simulation.frequencies,
            true,
        );
    }

    #[test]
    fn schroeder_decay_starts_at_zero_and_never_increases() {
        let decay = schroeder_energy_decay_db(&[2.0, -1.0, 0.5]).unwrap();

        assert_eq!(decay.len(), 3);
        assert!(decay[0].abs() < 1.0e-12);
        assert!(decay.windows(2).all(|window| window[1] <= window[0]));
        assert!(decay.iter().all(|value| value.is_finite()));
    }

    #[test]
    fn schroeder_decay_rejects_nonfinite_or_silent_impulses() {
        assert!(schroeder_energy_decay_db(&[0.0, f64::NAN]).is_err());
        assert!(schroeder_energy_decay_db(&[0.0, 0.0]).is_err());
    }

    #[test]
    fn rt60_estimate_requires_a_standard_decay_interval() {
        let times: Vec<f64> = (0..8).map(|index| index as f64 * 0.01).collect();
        let decay = vec![0.0, -3.0, -6.0, -12.0, -20.0, -30.0, -36.0, -40.0];
        let rt60 = estimate_rt60_seconds(&times, &decay).unwrap();
        assert!((rt60 - 0.08).abs() < 1.0e-12);
        assert_eq!(estimate_rt60_seconds(&times[..4], &decay[..4]), None);
        assert_eq!(estimate_rt60_seconds(&times, &decay[..7]), None);
    }
}
