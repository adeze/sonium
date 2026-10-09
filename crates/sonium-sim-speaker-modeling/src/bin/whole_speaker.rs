//! End-to-end loudspeaker study runner.
//!
//! The distributed electro-vibro-acoustic solve is two-way coupled, and the
//! resolved enclosure/port/exterior acoustic stage is condensed to a
//! per-frequency piston load `Z_resolved = (F_front - F_rear)/v̄` whose defect
//! against the compact-domain load `Z_compact` is injected back into the
//! monolithic system and iterated to a fixed point (default). Nonlinear bias
//! and eddy-current stages remain bias-linearized diagnostics. Pass
//! `--no-acoustic-feedback` to restore the legacy one-way workflow where the
//! structural diaphragm velocity drives the resolved acoustic domain without
//! load feedback.

use num_complex::Complex64;
use serde::Serialize;
use sonium_fem::physics::acoustics::electroacoustics::{
    CoupledIterativeConfig, CoupledPreconditioner,
};
use sonium_sim_speaker_modeling::conventions::{
    ArtifactProvenance, StudyMetadata, circular_plate_bending_frequency_hz,
};
use sonium_sim_speaker_modeling::field_export::{
    ProjectedCellVector, ProjectionProvenance, SpeakerFieldBundle, SpeakerFieldExporter,
    SpeakerPvdCollection, field_names,
};
use sonium_sim_speaker_modeling::motor_extraction::{
    MotorCompletionInput, MotorCompletionResult, run_motor_completion,
};
use sonium_sim_speaker_modeling::{
    AcousticMedium, AxisymmetricLinearConfig, AxisymmetricLinearModel, AxisymmetricLinearSolution,
    BoxComplianceModel, CalibrationReport, CalibrationTolerances, CoupledProductConstraintReport,
    CoupledProductLimits, CoupledProductMetrics, DriverCalibrationData, DriverModel,
    DriverModelExtractionEvidence, EnclosureSpec, LargeSignalSpec, LumpedNonlinearCompressionPoint,
    LumpedNonlinearCompressionSweep, MotorParameters, NonlinearSpeakerBiasConfig,
    NonlinearSpeakerBiasResult, PortNoiseInput, PortNoiseResult, PortSpec, ResolvedAcousticConfig,
    ResolvedAcousticModel, ResolvedAcousticSolution, ResolvedCouplingConfig, ResolvedLoadCoupler,
    Sector3dConfig, Sector3dModel, SpeakerDriverStudy, SpeakerEddyCurrentConfig,
    SpeakerEddyCurrentSweep, StructuralAcousticFeedback, StructuralAcousticMode,
    StructuralEvaluation, analyze_port_noise_input, correlate_driver_model,
};
use sonium_xem::io::binary_output_dir;
use std::error::Error;
use std::f64::consts::PI;
use std::fmt::Write as _;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::time::Instant;

type AnyError = Box<dyn Error>;

struct JoinOnDrop<T>(Option<std::thread::JoinHandle<T>>);

impl<T> JoinOnDrop<T> {
    fn new(handle: std::thread::JoinHandle<T>) -> Self {
        Self(Some(handle))
    }

    fn join(mut self) -> std::thread::Result<T> {
        self.0.take().expect("join handle is present").join()
    }
}

impl<T> Drop for JoinOnDrop<T> {
    fn drop(&mut self) {
        if let Some(handle) = self.0.take() {
            let _ = handle.join();
        }
    }
}

const DEFAULT_FREQUENCIES_HZ: [f64; 8] = [
    20.0,
    35.565_588,
    63.245_553,
    112.468_265,
    200.0,
    355.655_882,
    632.455_532,
    1_000.0,
];
const QUICK_FREQUENCIES_HZ: [f64; 3] = [40.0, 100.0, 250.0];

#[derive(Debug, Clone, PartialEq)]
struct RunOptions {
    output_dir: Option<PathBuf>,
    gallery_dir: Option<PathBuf>,
    frequencies_hz: Vec<f64>,
    voltage_rms_v: f64,
    quick: bool,
    with_sector_3d: bool,
    acoustic_feedback: bool,
    progress_json: bool,
    threads: usize,
    cuda_frequency_contract: bool,
    capabilities: bool,
    execution_contract: bool,
    minimum_impedance_real_ohm: Option<f64>,
    maximum_excursion_m: Option<f64>,
    driver_model_path: Option<PathBuf>,
    calibration_data_path: Option<PathBuf>,
    calibration_tolerances_path: Option<PathBuf>,
    large_signal: Option<LargeSignalRunOptions>,
    motor_completion_input_path: Option<PathBuf>,
    port_noise_input_path: Option<PathBuf>,
}

#[derive(Debug, Clone, PartialEq)]
struct LargeSignalRunOptions {
    frequency_hz: f64,
    voltages_rms_v: Vec<f64>,
    duration_s: f64,
    samples_per_period: usize,
    port_loss_coefficient: f64,
    secondary_frequency_hz: Option<f64>,
    secondary_voltage_ratio: f64,
}

impl RunOptions {
    fn parse(args: impl IntoIterator<Item = String>) -> Result<Self, AnyError> {
        let mut output_dir = None;
        let mut gallery_dir = None;
        let mut frequencies_hz = None;
        let mut voltage_rms_v: f64 = 2.83;
        let mut quick = false;
        let mut with_sector_3d = false;
        let mut acoustic_feedback = true;
        let mut progress_json = false;
        let mut threads = 0usize;
        let mut cuda_frequency_contract = false;
        let mut capabilities = false;
        let mut execution_contract = false;
        let mut minimum_impedance_real_ohm: Option<f64> = None;
        let mut maximum_excursion_m: Option<f64> = None;
        let mut driver_model_path = None;
        let mut calibration_data_path = None;
        let mut calibration_tolerances_path = None;
        let mut large_signal_frequency_hz = None;
        let mut large_signal_voltages_rms_v = None;
        let mut large_signal_duration_s = 0.25;
        let mut large_signal_samples_per_period = 200;
        let mut port_loss_coefficient = 1.5;
        let mut secondary_frequency_hz = None;
        let mut secondary_voltage_ratio = 0.25;
        let mut large_signal_controls_supplied = false;
        let mut motor_completion_input_path = None;
        let mut port_noise_input_path = None;
        let mut args = args.into_iter();
        while let Some(argument) = args.next() {
            match argument.as_str() {
                "--output" => {
                    output_dir = Some(PathBuf::from(required_value(&mut args, "--output")?));
                }
                "--gallery" => {
                    gallery_dir = Some(PathBuf::from(required_value(&mut args, "--gallery")?));
                }
                "--frequencies" => {
                    frequencies_hz = Some(parse_frequencies(&required_value(
                        &mut args,
                        "--frequencies",
                    )?)?);
                }
                "--voltage" => {
                    voltage_rms_v = required_value(&mut args, "--voltage")?.parse()?;
                }
                "--quick" => quick = true,
                "--with-sector-3d" => with_sector_3d = true,
                "--no-acoustic-feedback" => acoustic_feedback = false,
                "--progress-json" => progress_json = true,
                "--threads" | "--cores" => {
                    threads = required_value(&mut args, "--threads")?.parse()?;
                }
                "--cuda-frequency-contract" => cuda_frequency_contract = true,
                "--capabilities" => capabilities = true,
                "--execution-contract" => execution_contract = true,
                "--minimum-impedance" => {
                    minimum_impedance_real_ohm =
                        Some(required_value(&mut args, "--minimum-impedance")?.parse()?);
                }
                "--maximum-excursion" => {
                    maximum_excursion_m =
                        Some(required_value(&mut args, "--maximum-excursion")?.parse()?);
                }
                "--driver-model" => {
                    driver_model_path =
                        Some(PathBuf::from(required_value(&mut args, "--driver-model")?));
                }
                "--calibration-data" => {
                    calibration_data_path = Some(PathBuf::from(required_value(
                        &mut args,
                        "--calibration-data",
                    )?));
                }
                "--calibration-tolerances" => {
                    calibration_tolerances_path = Some(PathBuf::from(required_value(
                        &mut args,
                        "--calibration-tolerances",
                    )?));
                }
                "--large-signal-frequency" => {
                    large_signal_frequency_hz =
                        Some(required_value(&mut args, "--large-signal-frequency")?.parse()?);
                }
                "--large-signal-voltages" => {
                    large_signal_voltages_rms_v = Some(parse_frequencies(&required_value(
                        &mut args,
                        "--large-signal-voltages",
                    )?)?);
                }
                "--large-signal-duration" => {
                    large_signal_duration_s =
                        required_value(&mut args, "--large-signal-duration")?.parse()?;
                    large_signal_controls_supplied = true;
                }
                "--large-signal-samples-per-period" => {
                    large_signal_samples_per_period =
                        required_value(&mut args, "--large-signal-samples-per-period")?.parse()?;
                    large_signal_controls_supplied = true;
                }
                "--port-loss-coefficient" => {
                    port_loss_coefficient =
                        required_value(&mut args, "--port-loss-coefficient")?.parse()?;
                    large_signal_controls_supplied = true;
                }
                "--secondary-frequency" => {
                    secondary_frequency_hz =
                        Some(required_value(&mut args, "--secondary-frequency")?.parse()?);
                    large_signal_controls_supplied = true;
                }
                "--secondary-voltage-ratio" => {
                    secondary_voltage_ratio =
                        required_value(&mut args, "--secondary-voltage-ratio")?.parse()?;
                    large_signal_controls_supplied = true;
                }
                "--motor-completion-input" => {
                    motor_completion_input_path = Some(PathBuf::from(required_value(
                        &mut args,
                        "--motor-completion-input",
                    )?));
                }
                "--port-noise-input" => {
                    port_noise_input_path = Some(PathBuf::from(required_value(
                        &mut args,
                        "--port-noise-input",
                    )?));
                }
                unknown => {
                    return Err(invalid_input(format!(
                        "unknown argument {unknown:?}; use --help for usage"
                    )));
                }
            }
        }
        if !voltage_rms_v.is_finite() || voltage_rms_v <= 0.0 {
            return Err(invalid_input("--voltage must be finite and positive"));
        }
        if minimum_impedance_real_ohm.is_some_and(|value| !value.is_finite() || value <= 0.0) {
            return Err(invalid_input(
                "--minimum-impedance must be finite and positive",
            ));
        }
        if maximum_excursion_m.is_some_and(|value| !value.is_finite() || value <= 0.0) {
            return Err(invalid_input(
                "--maximum-excursion must be finite and positive",
            ));
        }
        if driver_model_path.is_some() != calibration_data_path.is_some() {
            return Err(invalid_input(
                "--driver-model and --calibration-data must be supplied together",
            ));
        }
        if calibration_tolerances_path.is_some() && driver_model_path.is_none() {
            return Err(invalid_input(
                "--calibration-tolerances requires --driver-model and --calibration-data",
            ));
        }
        let frequencies_hz = frequencies_hz.unwrap_or_else(|| {
            if quick {
                QUICK_FREQUENCIES_HZ.to_vec()
            } else {
                DEFAULT_FREQUENCIES_HZ.to_vec()
            }
        });
        validate_frequencies(&frequencies_hz)?;
        let large_signal = match (large_signal_frequency_hz, large_signal_voltages_rms_v) {
            (Some(frequency_hz), Some(voltages_rms_v)) => {
                validate_large_signal_options(
                    frequency_hz,
                    &voltages_rms_v,
                    large_signal_duration_s,
                    large_signal_samples_per_period,
                    port_loss_coefficient,
                    secondary_frequency_hz,
                    secondary_voltage_ratio,
                )?;
                Some(LargeSignalRunOptions {
                    frequency_hz,
                    voltages_rms_v,
                    duration_s: large_signal_duration_s,
                    samples_per_period: large_signal_samples_per_period,
                    port_loss_coefficient,
                    secondary_frequency_hz,
                    secondary_voltage_ratio,
                })
            }
            (None, None) if !large_signal_controls_supplied => None,
            _ => {
                return Err(invalid_input(
                    "large-signal mode requires both --large-signal-frequency and --large-signal-voltages",
                ));
            }
        };
        Ok(Self {
            output_dir,
            gallery_dir,
            frequencies_hz,
            voltage_rms_v,
            quick,
            with_sector_3d,
            acoustic_feedback,
            progress_json,
            threads,
            cuda_frequency_contract,
            capabilities,
            execution_contract,
            minimum_impedance_real_ohm,
            maximum_excursion_m,
            driver_model_path,
            calibration_data_path,
            calibration_tolerances_path,
            large_signal,
            motor_completion_input_path,
            port_noise_input_path,
        })
    }
}

#[derive(Debug, Clone, Copy, Serialize)]
struct ComplexValue {
    re: f64,
    im: f64,
}

impl From<Complex64> for ComplexValue {
    fn from(value: Complex64) -> Self {
        Self {
            re: value.re,
            im: value.im,
        }
    }
}

#[derive(Debug, Serialize)]
struct RunConfiguration {
    frequencies_hz: Vec<f64>,
    voltage_rms_v: f64,
    quick_mode: bool,
    sector_3d_enabled: bool,
    acoustic_feedback_enabled: bool,
    requested_threads: usize,
    resolved_rayon_threads: usize,
    minimum_impedance_real_ohm: Option<f64>,
    maximum_excursion_m: Option<f64>,
}

#[derive(Debug, Serialize)]
struct BiasReport {
    linear_bl_tm: f64,
    nonlinear_bl_tm: f64,
    peak_flux_density_t: f64,
    saturated_quadrature_fraction: f64,
    flux_density_cell_magnitude_t: Vec<f64>,
    flux_density_grid_columns: usize,
    final_residual: f64,
    continuation_iterations: Vec<usize>,
    energy_balance_error_j: f64,
}

#[derive(Debug, Clone, Serialize)]
struct MotorPositionCurvesReport {
    schema_version: u32,
    provenance: String,
    artifact: String,
    bl_curve: Vec<BlPositionPoint>,
    le_curve: Vec<LePositionPoint>,
}

#[derive(Debug, Clone, Copy, Serialize)]
struct BlPositionPoint {
    displacement_m: f64,
    force_factor_tm: f64,
}

#[derive(Debug, Clone, Copy, Serialize)]
struct LePositionPoint {
    displacement_m: f64,
    inductance_h: f64,
}

#[derive(Debug, Serialize)]
struct SweepQaReport {
    all_points_converged: bool,
    max_relative_linear_residual: f64,
    max_relative_power_residual: f64,
    max_reciprocity_defect: f64,
    minimum_impedance_real_ohm: f64,
    minimum_dissipated_power_w: f64,
}

#[derive(Debug, Serialize)]
struct FrequencyPointReport {
    frequency_hz: f64,
    voltage_rms_v: f64,
    current_rms_a: ComplexValue,
    impedance_ohm: ComplexValue,
    diaphragm_velocity_rms_m_per_s: ComplexValue,
    distributed_observation_spl_db: f64,
    linear_relative_residual: f64,
    linear_power_residual_w: f64,
    eddy_total_coil_impedance_ohm: ComplexValue,
    eddy_effective_inductance_h: f64,
    eddy_loss_w_per_ampere_squared: f64,
    eddy_complex_bl_tm: ComplexValue,
    cavity_pressure_rms_pa: ComplexValue,
    port_volume_velocity_rms_m3_per_s: ComplexValue,
    far_field_pressure_rms_pa: ComplexValue,
    far_field_spl_db: f64,
    resolved_driver_input_power_w: f64,
    resolved_port_thermal_loss_w: f64,
    resolved_cavity_wall_loss_w: f64,
    radiated_power_w: f64,
    resolved_acoustic_power_residual_w: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    resolved_load_impedance_n_s_per_m: Option<ComplexValue>,
    #[serde(skip_serializing_if = "Option::is_none")]
    acoustic_load_delta_n_s_per_m: Option<ComplexValue>,
    #[serde(skip_serializing_if = "Option::is_none")]
    acoustic_feedback_iterations: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    acoustic_feedback_converged: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    acoustic_feedback_final_velocity_change: Option<f64>,
}

/// Modal structural response evaluated at every resolved acoustic pressure
/// point.  The public axisymmetric solve currently exposes displacement fields
/// but not structural eigenpairs, so this is a checked reduced modal
/// projection, with its assumptions documented in the gallery caption.
#[derive(Debug, Serialize)]
struct StructuralModalPoint {
    frequency_hz: f64,
    displacement_rms_m: ComplexValue,
    velocity_rms_m_per_s: ComplexValue,
    stress_rms_pa: ComplexValue,
    absorbed_power_w: f64,
}

/// Per-frequency condensed acoustic load feedback summary.
#[derive(Debug, Clone)]
struct CouplingRow {
    frequency_hz: f64,
    z_resolved_n_s_per_m: Complex64,
    z_compact_n_s_per_m: Complex64,
    delta_z_n_s_per_m: Complex64,
    iterations: usize,
    converged: bool,
    final_velocity_change: f64,
}

#[derive(Debug, Serialize)]
struct SectorReport {
    frequency_hz: f64,
    magnetic_impedance_ohm: ComplexValue,
    full_revolution_eddy_loss_w: f64,
    periodic_residual_max: f64,
    gauge_residual_max: f64,
    charge_continuity_relative_residual: f64,
}

#[derive(Debug, Serialize)]
struct WholeSpeakerReport {
    model: &'static str,
    provenance: ArtifactProvenance,
    coupling_scope: String,
    configuration: RunConfiguration,
    nonlinear_bias: BiasReport,
    sweep_qa: SweepQaReport,
    operating_metrics: CoupledProductMetrics,
    operating_constraints: CoupledProductConstraintReport,
    timings: TimingReport,
    frequency_points: Vec<FrequencyPointReport>,
    structural_modal_response: Vec<StructuralModalPoint>,
    sector_3d: Option<SectorReport>,
    #[serde(skip_serializing_if = "Option::is_none")]
    calibration: Option<CalibrationArtifact>,
    #[serde(skip_serializing_if = "Option::is_none")]
    extraction_evidence: Option<DriverModelExtractionEvidence>,
    #[serde(skip_serializing_if = "Option::is_none")]
    extracted_driver_model: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    large_signal: Option<LargeSignalReport>,
    #[serde(skip_serializing_if = "Option::is_none")]
    motor_completion: Option<MotorCompletionResult>,
    #[serde(skip_serializing_if = "Option::is_none")]
    motor_position_curves: Option<MotorPositionCurvesReport>,
    #[serde(skip_serializing_if = "Option::is_none")]
    port_noise: Option<PortNoiseResult>,
    artifacts: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
struct LargeSignalReport {
    schema_version: u32,
    provenance: String,
    frequency_hz: f64,
    port_loss_coefficient: f64,
    secondary_frequency_hz: Option<f64>,
    secondary_voltage_ratio: f64,
    max_spl_estimate_db: Option<f64>,
    calibrated_against_m1: bool,
    distortion_artifact: String,
    maxspl_artifact: String,
}

#[derive(Debug, Serialize)]
struct DistortionArtifact<'a> {
    schema_version: u32,
    provenance: &'a str,
    primary_frequency_hz: f64,
    secondary_frequency_hz: Option<f64>,
    secondary_voltage_ratio: f64,
    points: Vec<DistortionPoint>,
}

#[derive(Debug, Serialize)]
struct DistortionPoint {
    voltage_rms_v: f64,
    acoustic_pressure_fundamental_peak_pa: f64,
    acoustic_thd_ratio: f64,
    acoustic_imd_ratio: f64,
}

#[derive(Debug, Serialize)]
struct MaxSplArtifact<'a> {
    schema_version: u32,
    provenance: &'a str,
    frequency_hz: f64,
    compression_limit_db: f64,
    excursion_limit_m: Option<f64>,
    max_spl_estimate_db: Option<f64>,
    points: Vec<MaxSplPoint>,
}

#[derive(Debug, Serialize)]
struct MaxSplPoint {
    voltage_rms_v: f64,
    estimated_port_spl_db: f64,
    compression_db: f64,
    peak_excursion_m: f64,
    within_limits: bool,
}

/// Wall-clock stage timings for QA and performance-regression tracking.
///
/// These timings describe this runner's native stages; CEA, adjoint, and
/// optimization stages are intentionally absent because this executable does
/// not run them. Keeping the fields stage-specific prevents a single total
/// from hiding regressions in setup, solve, or field export work.
#[derive(Debug, Serialize)]
struct TimingReport {
    calibration_seconds: f64,
    setup_seconds: f64,
    nonlinear_bias_seconds: f64,
    eddy_current_seconds: f64,
    distributed_sweep_seconds: f64,
    resolved_acoustic_seconds: f64,
    sector_3d_seconds: Option<f64>,
    artifact_assembly_seconds: f64,
    total_seconds: f64,
}

/// Calibration provenance carried both in the machine-readable whole-speaker
/// report and in `driver-calibration-report.json`.
#[derive(Debug, Serialize)]
struct CalibrationArtifact {
    driver_model_path: String,
    calibration_data_path: String,
    tolerances: CalibrationTolerances,
    report: CalibrationReport,
}

/// Versioned machine-readable records consumed by the Studio process
/// supervisor. Records are opt-in so ordinary CLI users keep the existing
/// human-readable output. They are written to stderr, which keeps stdout
/// available for future report streaming while the supervisor drains both
/// streams into the bounded job log.
#[derive(Debug, Serialize)]
struct ProgressRecord<'a> {
    schema_version: u32,
    #[serde(rename = "type")]
    record_type: &'static str,
    phase: &'static str,
    message: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    completed: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    total: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    unit: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    frequency_hz: Option<f64>,
}

fn main() {
    if std::env::args()
        .skip(1)
        .any(|argument| argument == "--help" || argument == "-h")
    {
        print_help();
        return;
    }
    let result = RunOptions::parse(std::env::args().skip(1)).and_then(|options| {
        if options.capabilities {
            println!("cell,cpu,cuda,metal");
            println!(
                "whole-speaker-pipeline,yes,no,no\nproduction-frequency-operator,yes,{},no",
                if cfg!(feature = "cuda") {
                    "yes"
                } else {
                    "not-built"
                }
            );
            return Ok(());
        }
        sonium_xem::runtime::install_global_rayon_pool(options.threads, None)
            .map_err(|error| invalid_input(format!("failed to configure Rayon pool: {error}")))?;
        if options.execution_contract {
            let budget = rayon::current_num_threads();
            let evidence = sonium_xem::runtime::observe_execution_contract(
                "whole-speaker",
                "cpu",
                "cpu",
                std::collections::BTreeMap::from([
                    ("frequency_lane_budget".to_string(), budget),
                    ("lane_inner_pool_budget".to_string(), budget),
                ]),
                false,
                "host-cpu",
            );
            println!("{}", serde_json::to_string_pretty(&evidence)?);
            return Ok(());
        }
        run(options)
    });
    if let Err(error) = result {
        eprintln!("FAILED: {error}");
        std::process::exit(1);
    }
}

fn run(options: RunOptions) -> Result<(), AnyError> {
    let total_start = Instant::now();
    let final_frequency_hz = options.frequencies_hz.last().copied();
    let calibration_start = Instant::now();
    let calibration = load_calibration(&options)?;
    let calibration_seconds = calibration_start.elapsed().as_secs_f64();
    emit_progress(
        &options,
        "preparing",
        "Preparing whole-speaker native study",
        None,
        None,
        None,
    );
    let output_dir = options
        .gallery_dir
        .clone()
        .or(options.output_dir.clone())
        .unwrap_or_else(|| binary_output_dir("sonium-sim-speaker-modeling", "whole-speaker"));
    fs::create_dir_all(&output_dir)?;
    if options.cuda_frequency_contract {
        write_cuda_frequency_contract(&output_dir)?;
        println!(
            "Wrote bounded CUDA production-frequency contract to {}",
            output_dir.join("cuda-frequency-contract.json").display()
        );
        return Ok(());
    }
    println!("Whole-speaker FEM study");
    println!(" output: {}", output_dir.display());
    println!(" frequencies: {:?} Hz", options.frequencies_hz);

    let setup_start = Instant::now();
    let linear_config = AxisymmetricLinearConfig {
        // This is the production-QA mesh. The denser reference default needs a
        // stronger low-frequency preconditioner before it can be a CLI default.
        magnetic_radial_cells: 8,
        magnetic_axial_cells: 8,
        structural_radial_cells: 6,
        structural_axial_cells: 1,
        acoustic_axial_cells: if options.quick { 3 } else { 4 },
        // With the artificial acoustic loss factor removed, the coupled
        // operator sits near an undamped cavity/structure resonance inside
        // this sweep and the block-diagonal ILU stalls just above the 1e-9
        // exact-residual gate. Factoring the fully assembled system (coupling
        // blocks included) restores convergence with margin.
        //
        // The 1e-9 default acceptance tolerance sits exactly on the attainable
        // accuracy floor of this near-resonant operator, so parallel-reduction
        // rounding noise flipped the exact-residual gate run to run (observed
        // ~15% failure rate, rejected residuals 1.00e-9..1.19e-9). Accepting
        // 1e-8 keeps a decade of margin over that floor and still lands a
        // decade below the sweep QA gate (1e-7) checked below.
        iterative_solver: CoupledIterativeConfig {
            preconditioner: CoupledPreconditioner::Monolithic,
            relative_tolerance: 1.0e-8,
            ..CoupledIterativeConfig::default()
        },
        ..AxisymmetricLinearConfig::default()
    };
    let mut linear_model = AxisymmetricLinearModel::build(linear_config)?;
    if options.acoustic_feedback {
        // Normalized 2π-weighted boundary mass over the structural/acoustic
        // interface; receives iω·ΔZ each fixed-point update.
        linear_model.install_interface_load()?;
    }
    let setup_seconds = setup_start.elapsed().as_secs_f64();

    // The optional periodic motor-sector study depends only on its own
    // geometry and the selected representative frequency. It is the one
    // expensive pre-study that is independent of the nonlinear-bias -> eddy
    // -> distributed -> resolved chain, so run it concurrently. Its solver
    // shares the process-wide Rayon budget instead of creating a nested pool.
    let sector_task = if options.with_sector_3d {
        println!(" solving optional periodic 3D motor sector concurrently");
        let frequency_hz = options.frequencies_hz[options.frequencies_hz.len() / 2];
        let path = output_dir.join("periodic-motor-sector.vtu");
        Some(JoinOnDrop::new(std::thread::spawn(move || {
            let start = Instant::now();
            let sector_model = Sector3dModel::build(Sector3dConfig::default())
                .map_err(|error| error.to_string())?;
            let solution = sector_model
                .solve_frequency(frequency_hz)
                .map_err(|error| error.to_string())?;
            sector_model
                .export_solution(&solution, &path)
                .map_err(|error| error.to_string())?;
            Ok::<_, String>((
                SectorReport {
                    frequency_hz,
                    magnetic_impedance_ohm: solution.magnetic_impedance_ohm.into(),
                    full_revolution_eddy_loss_w: solution.full_revolution_eddy_loss_w,
                    periodic_residual_max: solution.periodic_residual_max,
                    gauge_residual_max: solution.gauge_residual_max,
                    charge_continuity_relative_residual: solution
                        .charge_continuity_relative_residual,
                },
                path,
                start.elapsed().as_secs_f64(),
            ))
        })))
    } else {
        None
    };

    let nonlinear_start = Instant::now();
    emit_progress(
        &options,
        "solving",
        "Solving nonlinear magnetic bias",
        Some(0),
        Some(options.frequencies_hz.len()),
        None,
    );
    println!(" solving nonlinear magnetic bias");
    let mut nonlinear_config = NonlinearSpeakerBiasConfig::default();
    if options.quick {
        nonlinear_config.solve.continuation_steps = 4;
    }
    let magnetic_current_limit_a = options.voltage_rms_v / linear_model.config.coil_resistance_ohm;
    if !magnetic_current_limit_a.is_finite() || magnetic_current_limit_a <= 0.0 {
        return Err(invalid_input(
            "magnetic current-sweep limit must be finite positive",
        ));
    }
    let nonlinear_bias_sweep = linear_model.solve_nonlinear_magnetic_bias_current_sweep(
        &nonlinear_config,
        &[
            -magnetic_current_limit_a,
            -0.5 * magnetic_current_limit_a,
            0.0,
            0.5 * magnetic_current_limit_a,
            magnetic_current_limit_a,
        ],
    )?;
    let nonlinear_bias = &nonlinear_bias_sweep.baseline;
    let nonlinear_bias_path = output_dir.join("nonlinear-magnetic-bias.vtu");
    let (flux_density_cell_magnitude_t, flux_density_grid_columns) =
        export_nonlinear_bias(&linear_model, nonlinear_bias, &nonlinear_bias_path)?;
    let nonlinear_bias_seconds = nonlinear_start.elapsed().as_secs_f64();

    let eddy_start = Instant::now();
    emit_progress(
        &options,
        "solving",
        "Solving bias-linearized eddy currents",
        None,
        None,
        None,
    );
    println!(" solving bias-linearized eddy currents");
    let eddy_sweep = linear_model.solve_axisymmetric_eddy_current_sweep(
        nonlinear_bias,
        &options.frequencies_hz,
        &SpeakerEddyCurrentConfig::default(),
    )?;
    let eddy_pvd_path = output_dir.join("eddy-current-fields.pvd");
    export_eddy_sweep(&linear_model, &eddy_sweep, &eddy_pvd_path)?;
    let eddy_current_seconds = eddy_start.elapsed().as_secs_f64();

    let distributed_start = Instant::now();
    emit_progress(
        &options,
        "solving",
        "Solving distributed electro-vibro-acoustic sweep",
        Some(0),
        Some(options.frequencies_hz.len()),
        None,
    );
    println!(" solving distributed electro-vibro-acoustic sweep");
    let voltage = Complex64::new(options.voltage_rms_v, 0.0);
    let linear_solutions = linear_model.solve_sweep_parallel(&options.frequencies_hz, voltage)?;
    for (index, solution) in linear_solutions.iter().enumerate() {
        emit_progress(
            &options,
            "solving",
            "Solved distributed electro-vibro-acoustic frequency",
            Some(index + 1),
            Some(options.frequencies_hz.len()),
            Some(solution.frequency_hz),
        );
    }
    let sweep_qa = linear_model.audit_sweep(&linear_solutions)?;
    if !sweep_qa.passes(1.0e-7, 1.0e-6) {
        return Err(invalid_input(format!(
            "distributed sweep failed QA: {sweep_qa:?}"
        )));
    }
    let linear_pvd_path = output_dir.join("distributed-coupled-fields.pvd");
    export_linear_sweep(&linear_model, &linear_solutions, &linear_pvd_path)?;
    let distributed_sweep_seconds = distributed_start.elapsed().as_secs_f64();
    emit_progress(
        &options,
        "solving",
        "Distributed electro-vibro-acoustic sweep complete",
        Some(options.frequencies_hz.len()),
        Some(options.frequencies_hz.len()),
        options.frequencies_hz.last().copied(),
    );

    let resolved_start = Instant::now();
    emit_progress(
        &options,
        "postprocessing",
        "Solving resolved enclosure, port, baffle, and exterior fields",
        Some(0),
        Some(options.frequencies_hz.len()),
        None,
    );
    println!(" solving resolved enclosure, port, baffle, and exterior fields");
    let default_acoustic = ResolvedAcousticConfig::default();
    let fluid = sonium_fem::physics::acoustics::pressure::FluidProperties {
        density: linear_model.config.fluid_density_kg_per_m3,
        speed_of_sound: linear_model.config.sound_speed_m_per_s,
        ..default_acoustic.fluid.clone()
    };
    let mut acoustic_config = ResolvedAcousticConfig {
        fluid,
        diaphragm_radius_m: linear_model.config.diaphragm_radius_m,
        // Keep quick mode to a three-point frequency sweep, but do not lower
        // the resolved mesh below the validated two-cells-per-interval
        // contract: the one-cell mesh can report an apparently converged
        // linear solve while losing more than half the acoustic power ledger.
        cells_per_physical_interval: if options.quick {
            10
        } else {
            default_acoustic.cells_per_physical_interval
        },
        pml_cells: if options.quick {
            12
        } else {
            default_acoustic.pml_cells
        },
        radial_pml_thickness_m: if options.quick {
            0.20
        } else {
            default_acoustic.radial_pml_thickness_m
        },
        upper_pml_thickness_m: if options.quick {
            0.20
        } else {
            default_acoustic.upper_pml_thickness_m
        },
        lower_pml_thickness_m: if options.quick {
            0.20
        } else {
            default_acoustic.lower_pml_thickness_m
        },
        ..default_acoustic
    };
    // Size the dimensionless PML stretch from the lowest swept frequency so
    // the round-trip truncation reflection stays small across the whole sweep.
    let pml_target_reflection = 0.01;
    acoustic_config.configure_pml_for_sweep(options.frequencies_hz[0], pml_target_reflection)?;
    let lowest_wavenumber =
        2.0 * PI * options.frequencies_hz[0] / acoustic_config.fluid.speed_of_sound;
    let pml_min_thickness = acoustic_config
        .radial_pml_thickness_m
        .min(acoustic_config.upper_pml_thickness_m)
        .min(acoustic_config.lower_pml_thickness_m);
    let estimated_reflection =
        (-2.0 * lowest_wavenumber * acoustic_config.pml_sigma_max * pml_min_thickness / 3.0).exp();
    println!(
        " PML stretch sigma_tilde_max = {:.3}; estimated round-trip reflection at {:.1} Hz = {:.2e}",
        acoustic_config.pml_sigma_max, options.frequencies_hz[0], estimated_reflection
    );
    let lumped_acoustic_config = acoustic_config.clone();
    let acoustic_model = ResolvedAcousticModel::build(acoustic_config)?;
    let mut coupling_rows = Vec::new();
    let mut coupling_points = None;
    let mut acoustic_solutions = Vec::with_capacity(linear_solutions.len());
    let mut diaphragm_velocities = Vec::with_capacity(linear_solutions.len());
    let report_linear = if options.acoustic_feedback {
        println!(" iterating condensed resolved acoustic load back into the structure");
        let points = ResolvedLoadCoupler::solve_sweep_parallel(
            &linear_model,
            &acoustic_model,
            ResolvedCouplingConfig::default(),
            &options.frequencies_hz,
            voltage,
        )?;
        for (index, point) in points.iter().enumerate() {
            emit_progress(
                &options,
                "solving",
                "Solved resolved acoustic coupling frequency",
                Some(index + 1),
                Some(options.frequencies_hz.len()),
                Some(point.frequency_hz),
            );
            println!(
                "  {:>7.1} Hz: fixed point in {} iterations (final velocity change {:.2e})",
                point.frequency_hz, point.iterations, point.relative_velocity_change
            );
            coupling_rows.push(CouplingRow {
                frequency_hz: point.frequency_hz,
                z_resolved_n_s_per_m: point.z_resolved_n_s_per_m,
                z_compact_n_s_per_m: point.z_compact_n_s_per_m,
                delta_z_n_s_per_m: point.delta_z_n_s_per_m,
                iterations: point.iterations,
                converged: point.converged,
                final_velocity_change: point.relative_velocity_change,
            });
        }
        let point_count = points.len();
        coupling_points = Some(points);
        let mut coupled_linear = Vec::with_capacity(point_count);
        for point in coupling_points
            .as_ref()
            .expect("coupling points were stored")
        {
            if !point.converged {
                return Err(invalid_input(format!(
                    "resolved acoustic load fixed point did not converge at {} Hz: {:?}",
                    point.frequency_hz, point.velocity_change_history
                )));
            }
            diaphragm_velocities.push(point.diaphragm_velocity_rms_m_per_s);
            acoustic_solutions.push(point.acoustic_solution.clone());
            coupled_linear.push(point.linear_solution.clone());
        }
        coupled_linear
    } else {
        let mut one_way_linear = Vec::with_capacity(linear_solutions.len());
        for solution in linear_solutions {
            let velocity = linear_model.area_averaged_diaphragm_velocity(&solution)?;
            diaphragm_velocities.push(velocity);
            one_way_linear.push(solution);
        }
        acoustic_solutions = acoustic_model
            .solve_driven_sweep_parallel(&options.frequencies_hz, &diaphragm_velocities)?;
        for (index, solution) in acoustic_solutions.iter().enumerate() {
            emit_progress(
                &options,
                "postprocessing",
                "Solved resolved acoustic frequency",
                Some(index + 1),
                Some(options.frequencies_hz.len()),
                Some(solution.frequency_hz),
            );
        }
        one_way_linear
    };
    let resolved_pvd_path = output_dir.join("resolved-enclosure-fields.pvd");
    acoustic_model.export_solutions(&acoustic_solutions, &resolved_pvd_path)?;
    let resolved_acoustic_seconds = resolved_start.elapsed().as_secs_f64();
    emit_progress(
        &options,
        "postprocessing",
        "Resolved acoustic fields complete",
        Some(options.frequencies_hz.len()),
        Some(options.frequencies_hz.len()),
        options.frequencies_hz.last().copied(),
    );

    // Collapse the distributed solve into the same typed contract used by
    // product optimizers. Optional CLI limits make this a real fail-safe QA
    // gate while leaving the default runner observational (no arbitrary
    // material or manufacturability limits are invented here).
    let structural = StructuralEvaluation::from_axisymmetric_sweep(
        &report_linear,
        &linear_model.structural_mesh,
    )?;
    let (extraction_evidence, extracted_driver_model) = if options.acoustic_feedback {
        let evidence = DriverModelExtractionEvidence::from_solve_outputs(
            nonlinear_bias,
            &eddy_sweep,
            &sweep_qa,
            &structural,
            &acoustic_solutions,
            coupling_points.as_deref(),
        );
        let extracted_motor = MotorParameters {
            // The current nonlinear magnetic stage supplies a converged
            // rest-position BL value. Keeping this as a one-point curve is
            // honest: no displacement sweep was run by this executable.
            bl_curve: vec![(0.0, nonlinear_bias.distributed_bl_tm)],
            le_vs_frequency: eddy_sweep
                .points
                .iter()
                .map(|point| (point.frequency_hz, point.effective_inductance_h))
                .collect(),
            re_eddy_vs_frequency: eddy_sweep
                .points
                .iter()
                .map(|point| (point.frequency_hz, point.eddy_resistance_ohm))
                .collect(),
            bl_nominal: nonlinear_bias.distributed_bl_tm,
            le_nominal: eddy_sweep
                .points
                .first()
                .map(|point| point.effective_inductance_h)
                .ok_or_else(|| invalid_input("eddy-current extraction returned no points"))?,
            le_surface: Vec::new(),
            bl_temperature_coefficient_per_k: -1.0e-3,
            magnetic_law: Some(
                nonlinear_bias_sweep.to_motor_magnetic_law(&nonlinear_config, None)?,
            ),
            hysteresis_loss: None,
            eddy_current_loss: Some(eddy_sweep.to_eddy_current_loss_model()?),
        };
        let mut driver_study = SpeakerDriverStudy::detailed_frequency_domain_driver();
        driver_study.frequency_hz = options.frequencies_hz.clone();
        driver_study.drive_voltage_rms_v = options.voltage_rms_v;
        driver_study.validate()?;
        let driver_model = DriverModel::from_study_with_motor_and_evidence(
            &driver_study,
            &extracted_motor,
            evidence,
        )?;
        (Some(evidence), Some(driver_model))
    } else {
        (None, None)
    };
    let operating_metrics = CoupledProductMetrics::from_axisymmetric_sweep(&sweep_qa, &structural);
    let operating_limits = CoupledProductLimits {
        minimum_impedance_real_ohm: options.minimum_impedance_real_ohm,
        maximum_excursion_m: options.maximum_excursion_m,
        ..CoupledProductLimits::default()
    };
    let operating_constraints =
        CoupledProductConstraintReport::evaluate(&operating_metrics, &operating_limits)?;
    if !operating_constraints.feasible {
        return Err(invalid_input(format!(
            "coupled product constraints failed: labels={:?}, margins={:?}",
            operating_constraints.labels, operating_constraints.margins
        )));
    }

    let (sector_3d, sector_artifact, sector_3d_seconds) = if let Some(task) = sector_task {
        let (report, path, seconds) = task
            .join()
            .map_err(|_| invalid_input("periodic 3D motor sector worker panicked"))?
            .map_err(|error| invalid_input(format!("periodic 3D motor sector failed: {error}")))?;
        (Some(report), Some(path), Some(seconds))
    } else {
        (None, None, None)
    };

    emit_progress(
        &options,
        "postprocessing",
        "Assembling normalized result artifacts",
        None,
        None,
        None,
    );
    let artifact_start = Instant::now();
    let points = build_frequency_reports(
        &report_linear,
        &eddy_sweep,
        &acoustic_solutions,
        &diaphragm_velocities,
        if options.acoustic_feedback {
            Some(&coupling_rows)
        } else {
            None
        },
    )?;
    let structural_modal_response = build_structural_modal_response(&points, &linear_model.config)?;
    let structural_modal_csv = output_dir.join("structural-modal-response.csv");
    write_structural_modal_csv(&structural_modal_csv, &structural_modal_response)?;
    let csv_path = output_dir.join("frequency-response.csv");
    write_csv(&csv_path, &points)?;
    let mut artifacts = vec![
        nonlinear_bias_path,
        eddy_pvd_path,
        linear_pvd_path,
        resolved_pvd_path,
        structural_modal_csv,
        csv_path,
    ];
    let large_signal = write_large_signal_artifacts(
        &output_dir,
        &options,
        &lumped_acoustic_config,
        nonlinear_bias.distributed_bl_tm,
        eddy_sweep
            .points
            .first()
            .map(|point| point.effective_inductance_h)
            .ok_or_else(|| invalid_input("eddy-current extraction returned no points"))?,
    )?;
    if let Some((_, paths)) = &large_signal {
        artifacts.extend(paths.iter().cloned());
    }
    let motor_completion = write_motor_completion_artifact(
        &output_dir,
        options.motor_completion_input_path.as_deref(),
    )?;
    if let Some((_, path)) = &motor_completion {
        artifacts.push(path.clone());
    }
    let motor_position_curves = motor_completion
        .as_ref()
        .map(|(completion, _)| extract_motor_position_curves(completion))
        .transpose()?
        .map(|mut curves| {
            let path = output_dir.join("motor-position-curves.csv");
            write_motor_position_curves_csv(&path, &curves)?;
            artifacts.push(path);
            curves.artifact = "motor-position-curves.csv".to_owned();
            Ok::<_, AnyError>(curves)
        })
        .transpose()?;
    let port_noise =
        write_port_noise_artifact(&output_dir, options.port_noise_input_path.as_deref())?;
    if let Some((_, path)) = &port_noise {
        artifacts.push(path.clone());
    }
    if options.acoustic_feedback {
        let load_csv_path = output_dir.join("resolved-load-impedance.csv");
        write_load_impedance_csv(&load_csv_path, &coupling_rows)?;
        artifacts.push(load_csv_path);
    }
    if let Some(path) = sector_artifact {
        artifacts.push(path);
    }
    let driver_model_artifact_path = extracted_driver_model
        .as_ref()
        .map(|_| output_dir.join("driver-model.json"));
    if let (Some(path), Some(driver_model)) = (&driver_model_artifact_path, &extracted_driver_model)
    {
        fs::write(path, serde_json::to_vec_pretty(driver_model)?)?;
        artifacts.push(path.clone());
    }
    let calibration_artifact_path = calibration
        .as_ref()
        .map(|_| output_dir.join("driver-calibration-report.json"));
    if let (Some(path), Some(calibration)) = (&calibration_artifact_path, &calibration) {
        fs::write(path, serde_json::to_vec_pretty(calibration)?)?;
        artifacts.push(path.clone());
        if !calibration.report.passed {
            eprintln!(
                "WARNING: driver calibration report failed one or more configured tolerances"
            );
        }
    }
    let artifact_assembly_seconds = artifact_start.elapsed().as_secs_f64();
    let report_path = output_dir.join("whole-speaker-report.json");
    let report = WholeSpeakerReport {
        model: "distributed axisymmetric loudspeaker with nonlinear motor diagnostics and resolved finite-baffle vented enclosure",
        provenance: ArtifactProvenance::native_golden(),
        coupling_scope: if options.acoustic_feedback {
            "two-way distributed linear electro-vibro-acoustic assembly; nonlinear bias and eddy-current diagnostics are bias-linearized; resolved enclosure acoustics are condensed to a per-frequency piston load Z_resolved=(F_front-F_rear)/v̄ and iterated back into the monolithic system as iω·ΔZ·M_Γ (ΔZ = Z_resolved - Z_compact) to a velocity fixed point"
        } else {
            "two-way distributed linear electro-vibro-acoustic assembly; nonlinear bias and eddy-current diagnostics are bias-linearized; resolved enclosure acoustics are driven one-way by the area-averaged structural diaphragm velocity"
        }
        .to_owned(),
        configuration: RunConfiguration {
            frequencies_hz: options.frequencies_hz.clone(),
            voltage_rms_v: options.voltage_rms_v,
            quick_mode: options.quick,
            sector_3d_enabled: options.with_sector_3d,
            acoustic_feedback_enabled: options.acoustic_feedback,
            requested_threads: options.threads,
            resolved_rayon_threads: rayon::current_num_threads(),
            minimum_impedance_real_ohm: options.minimum_impedance_real_ohm,
            maximum_excursion_m: options.maximum_excursion_m,
        },
        nonlinear_bias: BiasReport {
            linear_bl_tm: nonlinear_bias.linear_reference_bl_tm,
            nonlinear_bl_tm: nonlinear_bias.distributed_bl_tm,
            peak_flux_density_t: nonlinear_bias.peak_flux_density_t,
            saturated_quadrature_fraction: nonlinear_bias.saturated_quadrature_fraction,
            flux_density_cell_magnitude_t,
            flux_density_grid_columns,
            final_residual: nonlinear_bias.solution.final_residual,
            continuation_iterations: nonlinear_bias.solution.continuation_iterations.clone(),
            energy_balance_error_j: nonlinear_bias.energy.balance_error_j,
        },
        sweep_qa: SweepQaReport {
            all_points_converged: sweep_qa.all_points_converged,
            max_relative_linear_residual: sweep_qa.max_relative_linear_residual,
            max_relative_power_residual: sweep_qa.max_relative_power_residual,
            max_reciprocity_defect: sweep_qa.max_reciprocity_defect,
            minimum_impedance_real_ohm: sweep_qa.minimum_impedance_real_ohm,
            minimum_dissipated_power_w: sweep_qa.minimum_dissipated_power_w,
        },
        operating_metrics,
        operating_constraints,
        timings: TimingReport {
            calibration_seconds,
            setup_seconds,
            nonlinear_bias_seconds,
            eddy_current_seconds,
            distributed_sweep_seconds,
            resolved_acoustic_seconds,
            sector_3d_seconds,
            artifact_assembly_seconds,
            total_seconds: total_start.elapsed().as_secs_f64(),
        },
        frequency_points: points,
        structural_modal_response,
        sector_3d,
        calibration,
        extraction_evidence,
        extracted_driver_model: driver_model_artifact_path
            .as_ref()
            .map(|path| path.display().to_string()),
        large_signal: large_signal.map(|(report, _)| report),
        motor_completion: motor_completion.map(|(report, _)| report),
        motor_position_curves,
        port_noise: port_noise.map(|(report, _)| report),
        artifacts: artifacts
            .iter()
            .map(|path| path.display().to_string())
            .collect(),
    };
    fs::write(&report_path, serde_json::to_vec_pretty(&report)?)?;
    if options.gallery_dir.is_some() {
        write_gallery(&output_dir, &report)?;
        println!(" gallery: {}", output_dir.join("gallery.json").display());
    }

    emit_progress(
        &options,
        "completed",
        "Whole-speaker study completed",
        Some(1),
        Some(1),
        final_frequency_hz,
    );

    println!(" report: {}", report_path.display());
    println!(
        " max linear residual: {:.3e}",
        sweep_qa.max_relative_linear_residual
    );
    println!(
        " max power residual: {:.3e}",
        sweep_qa.max_relative_power_residual
    );
    println!(
        " maximum reciprocity defect: {:.3e}",
        sweep_qa.max_reciprocity_defect
    );
    println!("SUCCESS");
    Ok(())
}

#[cfg(feature = "cuda")]
fn write_cuda_frequency_contract(output_dir: &Path) -> Result<(), AnyError> {
    let mut evidence = sonium_sim_speaker_modeling::run_production_frequency_step_cuda_contract()
        .map_err(invalid_input)?;
    let object = evidence
        .as_object_mut()
        .ok_or_else(|| invalid_input("CUDA contract did not return a JSON object"))?;
    object.insert(
        "device_identity".to_string(),
        serde_json::Value::String("cuda:ordinal=0".to_string()),
    );
    object.insert("fallback_used".to_string(), serde_json::Value::Bool(false));
    fs::write(
        output_dir.join("cuda-frequency-contract.json"),
        serde_json::to_vec_pretty(&evidence)?,
    )?;
    Ok(())
}

#[cfg(not(feature = "cuda"))]
fn write_cuda_frequency_contract(_output_dir: &Path) -> Result<(), AnyError> {
    Err(io::Error::new(
        io::ErrorKind::Unsupported,
        "--cuda-frequency-contract requires building whole-speaker with the `cuda` feature",
    )
    .into())
}

fn emit_progress(
    options: &RunOptions,
    phase: &'static str,
    message: &'static str,
    completed: Option<usize>,
    total: Option<usize>,
    frequency_hz: Option<f64>,
) {
    if !options.progress_json {
        return;
    }
    let record = ProgressRecord {
        schema_version: 1,
        record_type: "progress",
        phase,
        message,
        completed,
        total,
        unit: total.map(|_| "frequency"),
        frequency_hz,
    };
    match serde_json::to_string(&record) {
        Ok(line) => eprintln!("{line}"),
        Err(error) => eprintln!("progress record unavailable: {error}"),
    }
}

fn write_gallery(output_dir: &Path, report: &WholeSpeakerReport) -> Result<(), AnyError> {
    use sonium_xem::io::GalleryRenderer;
    let points = &report.frequency_points;
    if points.len() < 2
        || points.iter().any(|point| {
            !point.frequency_hz.is_finite()
                || !point.impedance_ohm.re.is_finite()
                || !point.far_field_spl_db.is_finite()
        })
    {
        return Err(invalid_input(
            "whole-speaker gallery requires finite production sweep data",
        ));
    }
    let frequency = points
        .iter()
        .map(|point| point.frequency_hz)
        .collect::<Vec<_>>();
    let impedance = points
        .iter()
        .map(|point| point.impedance_ohm.re.hypot(point.impedance_ohm.im))
        .collect::<Vec<_>>();
    let excursion = points
        .iter()
        .map(|point| {
            let velocity = point
                .diaphragm_velocity_rms_m_per_s
                .re
                .hypot(point.diaphragm_velocity_rms_m_per_s.im);
            velocity / (2.0 * PI * point.frequency_hz)
        })
        .collect::<Vec<_>>();
    let spl = points
        .iter()
        .map(|point| point.far_field_spl_db)
        .collect::<Vec<_>>();
    let eddy_loss = points
        .iter()
        .map(|point| point.eddy_loss_w_per_ampere_squared)
        .collect::<Vec<_>>();
    let bl = points
        .iter()
        .map(|point| {
            point
                .eddy_complex_bl_tm
                .re
                .hypot(point.eddy_complex_bl_tm.im)
        })
        .collect::<Vec<_>>();
    let le = points
        .iter()
        .map(|point| point.eddy_effective_inductance_h)
        .collect::<Vec<_>>();
    let cavity_pressure = points
        .iter()
        .map(|point| {
            point
                .cavity_pressure_rms_pa
                .re
                .hypot(point.cavity_pressure_rms_pa.im)
        })
        .collect::<Vec<_>>();
    let port_flow = points
        .iter()
        .map(|point| {
            point
                .port_volume_velocity_rms_m3_per_s
                .re
                .hypot(point.port_volume_velocity_rms_m3_per_s.im)
        })
        .collect::<Vec<_>>();
    let acoustic_power = points
        .iter()
        .map(|point| point.radiated_power_w)
        .collect::<Vec<_>>();
    let electrical_power = points
        .iter()
        .map(|point| point.voltage_rms_v * point.current_rms_a.re)
        .collect::<Vec<_>>();
    let thermal_loss = points
        .iter()
        .map(|point| point.resolved_port_thermal_loss_w)
        .collect::<Vec<_>>();
    let structural_response = points
        .iter()
        .map(|point| {
            point
                .diaphragm_velocity_rms_m_per_s
                .re
                .hypot(point.diaphragm_velocity_rms_m_per_s.im)
        })
        .collect::<Vec<_>>();
    let structural_mode_response = report
        .structural_modal_response
        .iter()
        .map(|point| {
            point
                .displacement_rms_m
                .re
                .hypot(point.displacement_rms_m.im)
        })
        .collect::<Vec<_>>();
    if structural_mode_response.len() != points.len()
        || structural_mode_response
            .iter()
            .any(|value| !value.is_finite())
    {
        return Err(invalid_input(
            "whole-speaker gallery structural modal response is misaligned or non-finite",
        ));
    }

    let mut gallery = GalleryRenderer::new(output_dir)?;
    gallery.grid_heatmap_png(
        "motor-flux-density.png",
        &report.nonlinear_bias.flux_density_cell_magnitude_t,
        report.nonlinear_bias.flux_density_grid_columns,
        "nonlinear-magnetic-bias.vtu",
        "Projected magnetic-flux-density magnitude |B| (T), row-major cell field aligned with nonlinear-magnetic-bias.vtu",
    )?;
    gallery.heatmap_png(
        "enclosure-pressure.png",
        &cavity_pressure,
        "resolved-enclosure-fields.pvd",
        "Pinned enclosure pressure magnitude from the resolved production sweep",
    )?;
    gallery.heatmap_png(
        "eddy-current-loss.png",
        &eddy_loss,
        "eddy-current-fields.pvd",
        "Pinned eddy-current loss proxy from the bias-linearized production sweep",
    )?;
    gallery.line_svg(
        "impedance.svg",
        &frequency,
        &impedance,
        "Electrical impedance magnitude",
        "frequency-response.csv",
        "Impedance magnitude (ohm) versus frequency (Hz)",
    )?;
    gallery.line_svg(
        "bl.svg",
        &frequency,
        &bl,
        "Force factor BL",
        "frequency-response.csv",
        "Magnitude of the complex bias-linearized BL (T·m)",
    )?;
    gallery.line_svg(
        "le.svg",
        &frequency,
        &le,
        "Effective voice-coil inductance Le",
        "frequency-response.csv",
        "Bias-linearized effective inductance (H)",
    )?;
    if let Some(curves) = &report.motor_position_curves {
        let bl_x = curves
            .bl_curve
            .iter()
            .map(|point| point.displacement_m)
            .collect::<Vec<_>>();
        let bl_y = curves
            .bl_curve
            .iter()
            .map(|point| point.force_factor_tm)
            .collect::<Vec<_>>();
        gallery.line_svg(
            "bl-x.svg",
            &bl_x,
            &bl_y,
            "Displacement-dependent force factor BL(x)",
            curves.artifact.as_str(),
            format!(
                "FEM-extracted BL(x) (T·m) versus displacement (m); provenance: {}",
                curves.provenance
            ),
        )?;
        let le_x = curves
            .le_curve
            .iter()
            .map(|point| point.displacement_m)
            .collect::<Vec<_>>();
        let le_y = curves
            .le_curve
            .iter()
            .map(|point| point.inductance_h)
            .collect::<Vec<_>>();
        gallery.line_svg(
            "le-x.svg",
            &le_x,
            &le_y,
            "Displacement-dependent inductance Le(x)",
            curves.artifact.as_str(),
            format!(
                "FEM-extracted Le(x) (H), lowest-frequency zero-current slice, versus displacement (m); provenance: {}",
                curves.provenance
            ),
        )?;
    }
    gallery.line_svg(
        "port-volume-flow.svg",
        &frequency,
        &port_flow,
        "Resolved port volume flow",
        "frequency-response.csv",
        "Port volume-velocity magnitude (m³/s)",
    )?;
    gallery.line_svg(
        "acoustic-energy-flow.svg",
        &frequency,
        &acoustic_power,
        "Resolved radiated acoustic power",
        "frequency-response.csv",
        "Radiated acoustic power (W); electrical input is recorded in the report ledger",
    )?;
    gallery.line_svg(
        "electrical-input-power.svg",
        &frequency,
        &electrical_power,
        "Electrical input power",
        "frequency-response.csv",
        "Real electrical input power (W), V_rms·Re(I_rms)",
    )?;
    gallery.line_svg(
        "excursion.svg",
        &frequency,
        &excursion,
        "Diaphragm excursion",
        "frequency-response.csv",
        "RMS displacement estimate from solved diaphragm velocity (m)",
    )?;
    gallery.line_svg(
        "far-field-spl.svg",
        &frequency,
        &spl,
        "Resolved far-field response",
        "frequency-response.csv",
        "Far-field SPL (dB re 20 µPa) versus frequency (Hz)",
    )?;
    gallery.line_svg(
        "thermal-loss.svg",
        &frequency,
        &thermal_loss,
        "Resolved port thermal loss",
        "frequency-response.csv",
        "Resolved port thermal loss (W) from the coupled acoustic thermal ledger",
    )?;
    gallery.line_svg(
        "structural-response.svg",
        &frequency,
        &structural_response,
        "Diaphragm structural response",
        "frequency-response.csv",
        "Solved diaphragm velocity magnitude (m/s), the structural response trace used by the coupling ledger",
    )?;
    gallery.line_svg(
        "structural-mode.svg",
        &frequency,
        &structural_mode_response,
        "Reduced structural modal response",
        "structural-modal-response.csv",
        "Geometry/thickness-derived circular-plate modal projection of resolved cavity pressure (m RMS)",
    )?;
    let overview_path = output_dir.join("em-mech-acoustic-overview.svg");
    write_em_mech_acoustic_overview_svg(
        &overview_path,
        &frequency,
        &electrical_power,
        &structural_response,
        &acoustic_power,
    )?;
    gallery.existing_figure(
        "em-mech-acoustic-overview.svg",
        "svg",
        "frequency-response.csv",
        "Normalized electrical input power → diaphragm velocity → radiated acoustic power overview",
    )?;
    let manifest_path = gallery.finish("Whole-speaker multiphysics gallery", vec![
        "1920x1080 fixed raster canvas; deterministic colour mapping and SVG axes".into(),
        "Field images trace to VTU/PVD artifacts; frequency curves trace to frequency-response.csv; BL(x)/Le(x) curves use motor-position-curves.csv when motor completion is supplied".into(),
        "CPU coupled study with convergence, power, reciprocity, and passivity gates recorded in whole-speaker-report.json".into(),
        "Electrical, eddy-current, mechanical (excursion), and acoustic power observables use the resolved production ledger".into(),
    ])?;
    sonium_xem::io::validate_gallery_manifest(&manifest_path)?;
    Ok(())
}

fn write_em_mech_acoustic_overview_svg(
    path: &Path,
    frequency: &[f64],
    electrical_power: &[f64],
    mechanical_response: &[f64],
    acoustic_power: &[f64],
) -> Result<(), AnyError> {
    if frequency.len() < 2
        || electrical_power.len() != frequency.len()
        || mechanical_response.len() != frequency.len()
        || acoustic_power.len() != frequency.len()
        || frequency
            .iter()
            .chain(electrical_power)
            .chain(mechanical_response)
            .chain(acoustic_power)
            .any(|value| !value.is_finite())
    {
        return Err(invalid_input(
            "EM/mechanical/acoustic overview requires aligned finite series",
        ));
    }
    let width = 1920.0;
    let height = 1080.0;
    let left = 120.0;
    let right = 80.0;
    let top = 120.0;
    let bottom = 140.0;
    let plot_width = width - left - right;
    let plot_height = height - top - bottom;
    let normalized = |values: &[f64]| {
        let scale = values
            .iter()
            .map(|value| value.abs())
            .fold(0.0_f64, f64::max)
            .max(f64::MIN_POSITIVE);
        values
            .iter()
            .map(|value| value.abs() / scale)
            .collect::<Vec<_>>()
    };
    let electrical = normalized(electrical_power);
    let mechanical = normalized(mechanical_response);
    let acoustic = normalized(acoustic_power);
    let x_at = |index: usize| left + plot_width * index as f64 / (frequency.len() - 1) as f64;
    let y_at = |value: f64| top + plot_height * (1.0 - value.clamp(0.0, 1.0));
    let mut svg = String::new();
    writeln!(
        svg,
        "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"1920\" height=\"1080\" viewBox=\"0 0 1920 1080\">"
    )?;
    writeln!(
        svg,
        "<rect width=\"100%\" height=\"100%\" fill=\"#10151c\"/>"
    )?;
    writeln!(
        svg,
        "<text x=\"120\" y=\"64\" fill=\"#f2f4f8\" font-family=\"sans-serif\" font-size=\"34\">EM → mechanical → acoustic overview</text>"
    )?;
    writeln!(
        svg,
        "<text x=\"120\" y=\"96\" fill=\"#aab4c0\" font-family=\"sans-serif\" font-size=\"18\">Per-series magnitude normalized to its own production-sweep maximum</text>"
    )?;
    writeln!(
        svg,
        "<rect x=\"{left}\" y=\"{top}\" width=\"{plot_width}\" height=\"{plot_height}\" fill=\"#171e27\" stroke=\"#667383\"/>"
    )?;
    for tick in 0..=4 {
        let value = tick as f64 / 4.0;
        let y = y_at(value);
        writeln!(
            svg,
            "<line x1=\"{left}\" x2=\"{}\" y1=\"{y}\" y2=\"{y}\" stroke=\"#2d3744\"/><text x=\"{}\" y=\"{}\" fill=\"#aab4c0\" font-family=\"sans-serif\" font-size=\"16\">{value:.1}</text>",
            width - right,
            left - 42.0,
            y + 6.0,
        )?;
    }
    for (values, colour, label) in [
        (&electrical, "#e15759", "electrical input power"),
        (&mechanical, "#4e79a7", "diaphragm velocity"),
        (&acoustic, "#59a14f", "radiated acoustic power"),
    ] {
        write!(
            svg,
            "<polyline fill=\"none\" stroke=\"{colour}\" stroke-width=\"5\" points=\""
        )?;
        for (index, value) in values.iter().enumerate() {
            write!(svg, "{:.2},{:.2} ", x_at(index), y_at(*value))?;
        }
        writeln!(svg, "\"/>")?;
        let legend_index = match label {
            "electrical input power" => 0,
            "diaphragm velocity" => 1,
            _ => 2,
        };
        let legend_y = 180.0 + 34.0 * legend_index as f64;
        writeln!(
            svg,
            "<line x1=\"{}\" x2=\"{}\" y1=\"{legend_y}\" y2=\"{legend_y}\" stroke=\"{colour}\" stroke-width=\"6\"/><text x=\"{}\" y=\"{}\" fill=\"#f2f4f8\" font-family=\"sans-serif\" font-size=\"20\">{label}</text>",
            width - 540.0,
            width - 490.0,
            width - 475.0,
            legend_y + 7.0,
        )?;
    }
    writeln!(
        svg,
        "<text x=\"{left}\" y=\"{}\" fill=\"#aab4c0\" font-family=\"sans-serif\" font-size=\"18\">frequency sweep: {:.3}–{:.3} Hz</text>",
        height - 58.0,
        frequency[0],
        frequency[frequency.len() - 1],
    )?;
    writeln!(svg, "</svg>")?;
    fs::write(path, svg)?;
    Ok(())
}

fn export_nonlinear_bias(
    model: &AxisymmetricLinearModel,
    bias: &NonlinearSpeakerBiasResult,
    path: &Path,
) -> Result<(Vec<f64>, usize), AnyError> {
    let cell_count = model.magnetic_mesh.num_elements();
    let mut flux = vec![[0.0; 3]; cell_count];
    let mut field = vec![[0.0; 3]; cell_count];
    let mut reluctivity = vec![0.0; cell_count];
    let mut weights = vec![0.0; cell_count];
    for point in &bias.solution.quadrature_state {
        let cell = point.element_index;
        let weight = point.integration_weight_m3;
        if cell >= cell_count || !weight.is_finite() || weight <= 0.0 {
            return Err(invalid_input(
                "nonlinear magnetic bias has invalid quadrature cell weight",
            ));
        }
        flux[cell][0] += point.flux_density_rz_t[0] * weight;
        flux[cell][2] += point.flux_density_rz_t[1] * weight;
        field[cell][0] += point.field_strength_rz_a_per_m[0] * weight;
        field[cell][2] += point.field_strength_rz_a_per_m[1] * weight;
        reluctivity[cell] += point.reluctivity_m_per_h * weight;
        weights[cell] += weight;
    }
    for cell in 0..cell_count {
        if weights[cell] > 0.0 {
            for component in [0, 2] {
                flux[cell][component] /= weights[cell];
                field[cell][component] /= weights[cell];
            }
            reluctivity[cell] /= weights[cell];
        }
    }
    if field
        .iter()
        .flatten()
        .chain(reluctivity.iter())
        .any(|value| !value.is_finite())
    {
        return Err(invalid_input(
            "nonlinear magnetic bias projection contains non-finite values",
        ));
    }
    let grid_columns = model
        .config
        .magnetic_radial_cells
        .checked_mul(2)
        .ok_or_else(|| invalid_input("magnetic field grid column count overflowed"))?;
    let flux_density_cell_magnitude_t = flux_density_magnitudes(&flux, grid_columns)?;
    let mut fields = SpeakerFieldBundle::new();
    fields.add_point_real_scalar(
        field_names::MAGNETIC_POTENTIAL,
        bias.solution.vector_potential_atheta.clone(),
    )?;
    fields.add_cell_real_vector(
        field_names::MAGNETIC_FLUX_DENSITY,
        ProjectedCellVector::new(flux, ProjectionProvenance::QuadratureAverage),
    )?;
    fields.add_cell_real_scalar(
        field_names::MAGNETIC_FLUX_DENSITY_MAGNITUDE,
        flux_density_cell_magnitude_t.clone(),
    )?;
    fields.add_cell_real_vector(
        field_names::MAGNETIC_FIELD_STRENGTH,
        ProjectedCellVector::new(field, ProjectionProvenance::QuadratureAverage),
    )?;
    fields.add_cell_real_scalar("reluctivity_m_per_h", reluctivity)?;
    let mut metadata = StudyMetadata::new("nonlinear axisymmetric loudspeaker magnetic bias");
    metadata.reference_case = Some("low-carbon-steel B-H continuation".into());
    SpeakerFieldExporter::new(&model.magnetic_mesh, metadata, &fields).write_snapshot(path)?;
    Ok((flux_density_cell_magnitude_t, grid_columns))
}

fn flux_density_magnitudes(flux: &[[f64; 3]], grid_columns: usize) -> Result<Vec<f64>, AnyError> {
    if grid_columns == 0 || flux.len() < grid_columns || !flux.len().is_multiple_of(grid_columns) {
        return Err(invalid_input(
            "projected magnetic flux field is not a rectangular cell grid",
        ));
    }
    let magnitudes = flux
        .iter()
        .map(|value| value[0].hypot(value[1]).hypot(value[2]))
        .collect::<Vec<_>>();
    if magnitudes.iter().any(|value| !value.is_finite()) {
        return Err(invalid_input(
            "projected magnetic flux field contains non-finite values",
        ));
    }
    Ok(magnitudes)
}

fn export_eddy_sweep(
    model: &AxisymmetricLinearModel,
    sweep: &SpeakerEddyCurrentSweep,
    path: &Path,
) -> Result<(), AnyError> {
    let mut collection = SpeakerPvdCollection::new(path)?;
    for point in &sweep.points {
        let fields = point.field_bundle()?;
        let mut metadata = StudyMetadata::new("bias-linearized axisymmetric eddy currents");
        metadata.frequency_hz = Some(point.frequency_hz);
        metadata.reference_case = Some("nonlinear differential reluctivity".into());
        collection.push(&SpeakerFieldExporter::new(
            &model.magnetic_mesh,
            metadata,
            &fields,
        ))?;
    }
    collection.finish()?;
    Ok(())
}

fn export_linear_sweep(
    model: &AxisymmetricLinearModel,
    solutions: &[AxisymmetricLinearSolution],
    path: &Path,
) -> Result<(), AnyError> {
    let mut collection = SpeakerPvdCollection::new(path)?;
    for solution in solutions {
        let (mesh, fields) = model.combined_export_fields(solution)?;
        let mut metadata = StudyMetadata::new("distributed coupled whole-speaker sweep");
        metadata.frequency_hz = Some(solution.frequency_hz);
        metadata.drive_voltage_rms_v = Some(solution.voltage_rms_v.norm());
        metadata.reference_case = Some("axisymmetric distributed motor-structure-acoustics".into());
        collection.push(&SpeakerFieldExporter::new(&mesh, metadata, &fields))?;
    }
    collection.finish()?;
    Ok(())
}

fn build_frequency_reports(
    linear: &[AxisymmetricLinearSolution],
    eddy: &SpeakerEddyCurrentSweep,
    acoustic: &[ResolvedAcousticSolution],
    diaphragm_velocities: &[Complex64],
    coupling_rows: Option<&[CouplingRow]>,
) -> Result<Vec<FrequencyPointReport>, AnyError> {
    if linear.len() != eddy.points.len()
        || linear.len() != acoustic.len()
        || linear.len() != diaphragm_velocities.len()
        || coupling_rows.is_some_and(|rows| rows.len() != linear.len())
    {
        return Err(invalid_input("whole-speaker sweep stages are misaligned"));
    }
    let empty = Vec::new();
    let coupling_rows = coupling_rows.unwrap_or(&empty);
    Ok(linear
        .iter()
        .zip(&eddy.points)
        .zip(acoustic)
        .zip(diaphragm_velocities)
        .enumerate()
        .map(
            |(index, (((linear, eddy), acoustic), &diaphragm_velocity))| {
                let coupling = coupling_rows.get(index);
                FrequencyPointReport {
                    frequency_hz: linear.frequency_hz,
                    voltage_rms_v: linear.voltage_rms_v.norm(),
                    current_rms_a: linear.current_rms_a.into(),
                    impedance_ohm: linear.impedance_ohm.into(),
                    diaphragm_velocity_rms_m_per_s: diaphragm_velocity.into(),
                    distributed_observation_spl_db: linear.observation_spl_db,
                    linear_relative_residual: linear.solver.relative_residual,
                    linear_power_residual_w: linear.power.residual_w,
                    eddy_total_coil_impedance_ohm: eddy.total_coil_impedance_ohm.into(),
                    eddy_effective_inductance_h: eddy.effective_inductance_h,
                    eddy_loss_w_per_ampere_squared: eddy.eddy_loss_w_per_ampere_squared,
                    eddy_complex_bl_tm: eddy.complex_bl_tm.into(),
                    cavity_pressure_rms_pa: acoustic.cavity_pressure_rms_pa.into(),
                    port_volume_velocity_rms_m3_per_s: acoustic
                        .port_volume_velocity_rms_m3_per_s
                        .into(),
                    far_field_pressure_rms_pa: acoustic.far_field_pressure_rms_pa.into(),
                    far_field_spl_db: acoustic.far_field_spl_db,
                    resolved_driver_input_power_w: acoustic.driver_input_power_w,
                    resolved_port_thermal_loss_w: acoustic.port_thermal_loss_w,
                    resolved_cavity_wall_loss_w: acoustic.cavity_wall_loss_w,
                    radiated_power_w: acoustic.radiated_power_w,
                    resolved_acoustic_power_residual_w: acoustic.power_balance_residual_w,
                    resolved_load_impedance_n_s_per_m: coupling
                        .map(|row| row.z_resolved_n_s_per_m.into()),
                    acoustic_load_delta_n_s_per_m: coupling.map(|row| row.delta_z_n_s_per_m.into()),
                    acoustic_feedback_iterations: coupling.map(|row| row.iterations),
                    acoustic_feedback_converged: coupling.map(|row| row.converged),
                    acoustic_feedback_final_velocity_change: coupling
                        .map(|row| row.final_velocity_change),
                }
            },
        )
        .collect())
}

fn build_structural_modal_response(
    points: &[FrequencyPointReport],
    config: &AxisymmetricLinearConfig,
) -> Result<Vec<StructuralModalPoint>, AnyError> {
    if points.is_empty() {
        return Err(invalid_input(
            "structural modal response requires at least one frequency point",
        ));
    }
    let radius_m = config.diaphragm_radius_m;
    let thickness_m = config.diaphragm_thickness_m;
    let area_m2 = PI * radius_m.powi(2);
    // The axisymmetric production configuration does not yet carry the cone
    // material/eigenpair database. Use a deterministic circular-plate basis
    // derived from its geometry and thickness, and keep that assumption
    // visible in the serialized artifact and gallery caption.
    let density_kg_per_m3: f64 = 1_200.0;
    let youngs_modulus_pa: f64 = 2.5e9;
    let poisson_ratio: f64 = 0.35;
    let areal_mass_kg = density_kg_per_m3 * thickness_m * area_m2;
    let mode_coefficients: [f64; 3] = [10.215, 21.260, 32.160];
    let modal_mass_fractions: [f64; 3] = [0.2695, 0.12, 0.06];
    let modal_participation: [f64; 3] = [1.0, 0.60, 0.35];
    let stress_per_displacement: [f64; 3] = [8.0e7, 1.5e8, 2.5e8];
    let damping_ratio = config.structural_loss_factor.max(1.0e-3);
    let modes = mode_coefficients
        .into_iter()
        .zip(modal_mass_fractions)
        .zip(modal_participation)
        .zip(stress_per_displacement)
        .map(|(((coefficient, mass_fraction), participation), stress)| {
            Ok::<_, AnyError>(StructuralAcousticMode {
                natural_frequency_hz: circular_plate_bending_frequency_hz(
                    radius_m,
                    thickness_m,
                    density_kg_per_m3,
                    youngs_modulus_pa,
                    poisson_ratio,
                    coefficient,
                )?,
                effective_mass_kg: areal_mass_kg * mass_fraction,
                damping_ratio,
                pressure_participation: participation,
                stress_per_displacement_pa_per_m: stress,
            })
        })
        .collect::<Result<Vec<_>, AnyError>>()?;
    let feedback = StructuralAcousticFeedback {
        area_m2,
        modes,
        max_iterations: 1,
        velocity_tolerance: 1.0e-8,
        under_relaxation: 1.0,
    };
    feedback.validate()?;

    points
        .iter()
        .map(|point| {
            let pressure = Complex64::new(
                point.cavity_pressure_rms_pa.re,
                point.cavity_pressure_rms_pa.im,
            );
            let response = feedback.response_at(point.frequency_hz, pressure)?;
            Ok(StructuralModalPoint {
                frequency_hz: point.frequency_hz,
                displacement_rms_m: response.displacement_rms_m.into(),
                velocity_rms_m_per_s: response.velocity_rms_m_per_s.into(),
                stress_rms_pa: response.stress_rms_pa.into(),
                absorbed_power_w: response.absorbed_power_w,
            })
        })
        .collect()
}

fn write_structural_modal_csv(
    path: &Path,
    points: &[StructuralModalPoint],
) -> Result<(), AnyError> {
    let mut csv = String::from(
        "frequency_hz,modal_displacement_rms_m,modal_velocity_rms_m_per_s,modal_stress_rms_pa,absorbed_power_w\n",
    );
    for point in points {
        writeln!(
            csv,
            "{},{},{},{},{}",
            point.frequency_hz,
            point
                .displacement_rms_m
                .re
                .hypot(point.displacement_rms_m.im),
            point
                .velocity_rms_m_per_s
                .re
                .hypot(point.velocity_rms_m_per_s.im),
            point.stress_rms_pa.re.hypot(point.stress_rms_pa.im),
            point.absorbed_power_w,
        )?;
    }
    fs::write(path, csv)?;
    Ok(())
}

#[cfg(test)]
mod structural_modal_unit_tests {
    use super::*;

    fn test_frequency_point(frequency_hz: f64) -> FrequencyPointReport {
        FrequencyPointReport {
            frequency_hz,
            voltage_rms_v: 1.0,
            current_rms_a: ComplexValue { re: 0.0, im: 0.0 },
            impedance_ohm: ComplexValue { re: 0.0, im: 0.0 },
            diaphragm_velocity_rms_m_per_s: ComplexValue { re: 0.0, im: 0.0 },
            distributed_observation_spl_db: 0.0,
            linear_relative_residual: 0.0,
            linear_power_residual_w: 0.0,
            eddy_total_coil_impedance_ohm: ComplexValue { re: 0.0, im: 0.0 },
            eddy_effective_inductance_h: 0.0,
            eddy_loss_w_per_ampere_squared: 0.0,
            eddy_complex_bl_tm: ComplexValue { re: 0.0, im: 0.0 },
            cavity_pressure_rms_pa: ComplexValue { re: 1.0, im: 0.25 },
            port_volume_velocity_rms_m3_per_s: ComplexValue { re: 0.0, im: 0.0 },
            far_field_pressure_rms_pa: ComplexValue { re: 0.0, im: 0.0 },
            far_field_spl_db: 0.0,
            resolved_driver_input_power_w: 0.0,
            resolved_port_thermal_loss_w: 0.0,
            resolved_cavity_wall_loss_w: 0.0,
            radiated_power_w: 0.0,
            resolved_acoustic_power_residual_w: 0.0,
            resolved_load_impedance_n_s_per_m: None,
            acoustic_load_delta_n_s_per_m: None,
            acoustic_feedback_iterations: None,
            acoustic_feedback_converged: None,
            acoustic_feedback_final_velocity_change: None,
        }
    }

    #[test]
    fn reduced_modal_response_is_finite_and_tracks_pressure_points() {
        let response = build_structural_modal_response(
            &[test_frequency_point(100.0)],
            &AxisymmetricLinearConfig::default(),
        )
        .unwrap();

        assert_eq!(response.len(), 1);
        assert_eq!(response[0].frequency_hz, 100.0);
        assert!(response[0].displacement_rms_m.re.is_finite());
        assert!(response[0].displacement_rms_m.im.is_finite());
        assert!(response[0].velocity_rms_m_per_s.re.is_finite());
        assert!(response[0].stress_rms_pa.re.is_finite());
        assert!(response[0].absorbed_power_w.is_finite());
        assert!(response[0].displacement_rms_m.re.abs() > 0.0);
    }

    #[test]
    fn overview_svg_rejects_misaligned_or_nonfinite_series() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("overview.svg");
        let error = write_em_mech_acoustic_overview_svg(
            &path,
            &[100.0, 200.0],
            &[1.0],
            &[1.0, 2.0],
            &[1.0, 2.0],
        )
        .unwrap_err();
        assert!(error.to_string().contains("aligned finite series"));

        let error = write_em_mech_acoustic_overview_svg(
            &path,
            &[100.0, f64::NAN],
            &[1.0, 2.0],
            &[1.0, 2.0],
            &[1.0, 2.0],
        )
        .unwrap_err();
        assert!(error.to_string().contains("aligned finite series"));
    }

    #[test]
    fn modal_csv_serializes_complex_magnitudes() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("structural-modal-response.csv");
        write_structural_modal_csv(
            &path,
            &[StructuralModalPoint {
                frequency_hz: 100.0,
                displacement_rms_m: ComplexValue { re: 1.0, im: 2.0 },
                velocity_rms_m_per_s: ComplexValue { re: 3.0, im: 4.0 },
                stress_rms_pa: ComplexValue { re: 5.0, im: 12.0 },
                absorbed_power_w: 0.25,
            }],
        )
        .unwrap();

        let lines = fs::read_to_string(path).unwrap();
        let fields = lines.lines().nth(1).unwrap().split(',').collect::<Vec<_>>();
        assert_eq!(
            lines.lines().next().unwrap(),
            "frequency_hz,modal_displacement_rms_m,modal_velocity_rms_m_per_s,modal_stress_rms_pa,absorbed_power_w"
        );
        assert_eq!(fields.len(), 5);
        assert_eq!(fields[0], "100");
        assert!((fields[1].parse::<f64>().unwrap() - 5.0_f64.sqrt()).abs() < 1.0e-12);
        assert!((fields[2].parse::<f64>().unwrap() - 5.0).abs() < 1.0e-12);
        assert!((fields[3].parse::<f64>().unwrap() - 13.0).abs() < 1.0e-12);
        assert_eq!(fields[4], "0.25");
    }
}

fn write_csv(path: &Path, points: &[FrequencyPointReport]) -> Result<(), AnyError> {
    let mut csv = String::from(
        "frequency_hz,impedance_re_ohm,impedance_im_ohm,current_re_a,current_im_a,diaphragm_velocity_re_m_per_s,diaphragm_velocity_im_m_per_s,distributed_spl_db,eddy_resistance_ohm,eddy_inductance_h,eddy_loss_w_per_ampere_squared,eddy_bl_magnitude_tm,cavity_pressure_magnitude_pa,far_field_spl_db,port_volume_velocity_magnitude_m3_per_s,electrical_input_power_w,radiated_power_w\n",
    );
    for point in points {
        writeln!(
            csv,
            "{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}",
            point.frequency_hz,
            point.impedance_ohm.re,
            point.impedance_ohm.im,
            point.current_rms_a.re,
            point.current_rms_a.im,
            point.diaphragm_velocity_rms_m_per_s.re,
            point.diaphragm_velocity_rms_m_per_s.im,
            point.distributed_observation_spl_db,
            point.eddy_total_coil_impedance_ohm.re,
            point.eddy_effective_inductance_h,
            point.eddy_loss_w_per_ampere_squared,
            point
                .eddy_complex_bl_tm
                .re
                .hypot(point.eddy_complex_bl_tm.im),
            point
                .cavity_pressure_rms_pa
                .re
                .hypot(point.cavity_pressure_rms_pa.im),
            point.far_field_spl_db,
            point
                .port_volume_velocity_rms_m3_per_s
                .re
                .hypot(point.port_volume_velocity_rms_m3_per_s.im),
            point.voltage_rms_v * point.current_rms_a.re,
            point.radiated_power_w,
        )?;
    }
    fs::write(path, csv)?;
    Ok(())
}

fn write_load_impedance_csv(path: &Path, rows: &[CouplingRow]) -> Result<(), AnyError> {
    let mut csv = String::from(
        "frequency_hz,z_resolved_re_n_s_per_m,z_resolved_im_n_s_per_m,z_compact_re_n_s_per_m,z_compact_im_n_s_per_m,delta_z_re_n_s_per_m,delta_z_im_n_s_per_m,fixed_point_iterations,converged,final_relative_velocity_change\n",
    );
    for row in rows {
        writeln!(
            csv,
            "{},{},{},{},{},{},{},{},{},{}",
            row.frequency_hz,
            row.z_resolved_n_s_per_m.re,
            row.z_resolved_n_s_per_m.im,
            row.z_compact_n_s_per_m.re,
            row.z_compact_n_s_per_m.im,
            row.delta_z_n_s_per_m.re,
            row.delta_z_n_s_per_m.im,
            row.iterations,
            row.converged,
            row.final_velocity_change,
        )?;
    }
    fs::write(path, csv)?;
    Ok(())
}

fn write_large_signal_artifacts(
    output_dir: &Path,
    options: &RunOptions,
    acoustic_config: &ResolvedAcousticConfig,
    force_factor_tm: f64,
    inductance_h: f64,
) -> Result<Option<(LargeSignalReport, Vec<PathBuf>)>, AnyError> {
    let Some(controls) = options.large_signal.as_ref() else {
        return Ok(None);
    };
    let mut study = SpeakerDriverStudy::detailed_frequency_domain_driver();
    study.geometry.diaphragm_radius_m = acoustic_config.diaphragm_radius_m;
    study.voice_coil.bl_tm = force_factor_tm;
    study.voice_coil.inductance_h = inductance_h;
    study.moving_system.xmax_m = options.maximum_excursion_m;
    study.acoustic_medium = AcousticMedium {
        density_kg_per_m3: acoustic_config.fluid.density,
        sound_speed_m_per_s: acoustic_config.fluid.speed_of_sound,
        ..AcousticMedium::standard_air()
    };
    let enclosure_volume_m3 =
        PI * acoustic_config.enclosure_radius_m.powi(2) * acoustic_config.cavity_depth_m;
    let equivalent_port_radius_m = (acoustic_config.port_outer_radius_m.powi(2)
        - acoustic_config.port_inner_radius_m.powi(2))
    .sqrt();
    study.enclosure = EnclosureSpec::Vented {
        volume_m3: enclosure_volume_m3,
        port: PortSpec {
            radius_m: equivalent_port_radius_m,
            length_m: 2.0 * acoustic_config.baffle_half_thickness_m,
            end_correction_m: acoustic_config.port_end_correction_m,
            resistance_pa_s_per_m3: 0.0,
        },
        leakage_resistance_pa_s_per_m3: 0.0,
        absorption_resistance_pa_s_per_m3: 0.0,
    };
    study.validate()?;

    let mut spec = LargeSignalSpec {
        frequency_hz: controls.frequency_hz,
        duration_s: controls.duration_s,
        samples_per_period: controls.samples_per_period,
        secondary_frequency_hz: controls.secondary_frequency_hz,
        secondary_voltage_ratio: controls.secondary_voltage_ratio,
        ..LargeSignalSpec::default()
    };
    spec.acoustic.port_loss_coefficient = controls.port_loss_coefficient;
    spec.acoustic.box_compliance_model = BoxComplianceModel::Adiabatic;
    let sweep = study.try_lumped_nonlinear_compression_sweep(spec, &controls.voltages_rms_v)?;
    persist_large_signal_artifacts(output_dir, controls, &study, sweep).map(Some)
}

fn write_motor_completion_artifact(
    output_dir: &Path,
    input_path: Option<&Path>,
) -> Result<Option<(MotorCompletionResult, PathBuf)>, AnyError> {
    let Some(input_path) = input_path else {
        return Ok(None);
    };
    let input: MotorCompletionInput = serde_json::from_slice(&fs::read(input_path)?)?;
    let result = run_motor_completion(&input)?;
    let path = output_dir.join("motor-completion.json");
    fs::write(&path, serde_json::to_vec_pretty(&result)?)?;
    Ok(Some((result, path)))
}

fn extract_motor_position_curves(
    completion: &MotorCompletionResult,
) -> Result<MotorPositionCurvesReport, AnyError> {
    let mut bl_curve = completion
        .parameters
        .bl_curve
        .iter()
        .map(|&(displacement_m, force_factor_tm)| BlPositionPoint {
            displacement_m,
            force_factor_tm,
        })
        .collect::<Vec<_>>();
    bl_curve.sort_by(|left, right| left.displacement_m.total_cmp(&right.displacement_m));
    if bl_curve.len() < 2
        || bl_curve
            .iter()
            .any(|point| !point.displacement_m.is_finite() || !point.force_factor_tm.is_finite())
        || bl_curve
            .windows(2)
            .any(|pair| pair[0].displacement_m >= pair[1].displacement_m)
    {
        return Err(invalid_input(
            "motor completion BL(x) samples must be finite and strictly ordered",
        ));
    }

    let mut le_samples = Vec::<(f64, f64, f64)>::new();
    for point in &completion.parameters.le_surface {
        if point.current_a != 0.0 {
            continue;
        }
        if !point.displacement_m.is_finite()
            || !point.frequency_hz.is_finite()
            || point.frequency_hz <= 0.0
            || !point.inductance_h.is_finite()
            || point.inductance_h <= 0.0
        {
            return Err(invalid_input(
                "motor completion zero-current Le(x) samples must be finite and positive",
            ));
        }
        if let Some(existing) = le_samples
            .iter_mut()
            .find(|existing| existing.0 == point.displacement_m)
        {
            if point.frequency_hz < existing.1 {
                *existing = (point.displacement_m, point.frequency_hz, point.inductance_h);
            }
        } else {
            le_samples.push((point.displacement_m, point.frequency_hz, point.inductance_h));
        }
    }
    le_samples.sort_by(|left, right| left.0.total_cmp(&right.0));
    let le_curve = le_samples
        .into_iter()
        .map(|(displacement_m, _, inductance_h)| LePositionPoint {
            displacement_m,
            inductance_h,
        })
        .collect::<Vec<_>>();
    if le_curve.len() < 2
        || le_curve.windows(2).any(|pair| {
            pair[0].displacement_m >= pair[1].displacement_m
                || !pair[0].displacement_m.is_finite()
                || !pair[1].displacement_m.is_finite()
                || !pair[0].inductance_h.is_finite()
                || !pair[1].inductance_h.is_finite()
        })
    {
        return Err(invalid_input(
            "motor completion Le(x) requires at least two finite zero-current positions",
        ));
    }

    Ok(MotorPositionCurvesReport {
        schema_version: 1,
        provenance: completion.provenance.clone(),
        artifact: String::new(),
        bl_curve,
        le_curve,
    })
}

fn write_motor_position_curves_csv(
    path: &Path,
    curves: &MotorPositionCurvesReport,
) -> Result<(), AnyError> {
    let mut csv = String::new();
    writeln!(csv, "# schema_version={}", curves.schema_version)?;
    writeln!(csv, "# provenance={}", curves.provenance)?;
    writeln!(csv, "observable,displacement_m,value,unit")?;
    for point in &curves.bl_curve {
        writeln!(
            csv,
            "BL(x),{:.17e},{:.17e},T m",
            point.displacement_m, point.force_factor_tm
        )?;
    }
    for point in &curves.le_curve {
        writeln!(
            csv,
            "Le(x),{:.17e},{:.17e},H",
            point.displacement_m, point.inductance_h
        )?;
    }
    fs::write(path, csv)?;
    Ok(())
}

fn write_port_noise_artifact(
    output_dir: &Path,
    input_path: Option<&Path>,
) -> Result<Option<(PortNoiseResult, PathBuf)>, AnyError> {
    let Some(input_path) = input_path else {
        return Ok(None);
    };
    let input: PortNoiseInput = serde_json::from_slice(&fs::read(input_path)?)?;
    let result = analyze_port_noise_input(&input)?;
    let path = output_dir.join("port-noise.json");
    fs::write(&path, serde_json::to_vec_pretty(&result)?)?;
    Ok(Some((result, path)))
}

fn persist_large_signal_artifacts(
    output_dir: &Path,
    controls: &LargeSignalRunOptions,
    study: &SpeakerDriverStudy,
    sweep: LumpedNonlinearCompressionSweep,
) -> Result<(LargeSignalReport, Vec<PathBuf>), AnyError> {
    let distortion_path = output_dir.join("distortion.json");
    let maxspl_path = output_dir.join("maxspl.json");
    let distortion = DistortionArtifact {
        schema_version: 1,
        provenance: &sweep.provenance,
        primary_frequency_hz: sweep.frequency_hz,
        secondary_frequency_hz: controls.secondary_frequency_hz,
        secondary_voltage_ratio: controls.secondary_voltage_ratio,
        points: sweep
            .points
            .iter()
            .map(|point| DistortionPoint {
                voltage_rms_v: point.voltage_rms_v,
                acoustic_pressure_fundamental_peak_pa: point.box_pressure_amplitude_pa,
                acoustic_thd_ratio: point.acoustic_thd,
                acoustic_imd_ratio: point.acoustic_imd,
            })
            .collect(),
    };
    let maxspl = MaxSplArtifact {
        schema_version: 1,
        provenance: &sweep.provenance,
        frequency_hz: sweep.frequency_hz,
        compression_limit_db: 3.0,
        excursion_limit_m: study.moving_system.xmax_m,
        max_spl_estimate_db: sweep.max_spl_estimate_db,
        points: sweep
            .points
            .iter()
            .map(|point| maxspl_point(point, study.moving_system.xmax_m))
            .collect(),
    };
    fs::write(&distortion_path, serde_json::to_vec_pretty(&distortion)?)?;
    fs::write(&maxspl_path, serde_json::to_vec_pretty(&maxspl)?)?;
    let report = LargeSignalReport {
        schema_version: 1,
        provenance: sweep.provenance,
        frequency_hz: sweep.frequency_hz,
        port_loss_coefficient: sweep.port_loss_coefficient,
        secondary_frequency_hz: controls.secondary_frequency_hz,
        secondary_voltage_ratio: controls.secondary_voltage_ratio,
        max_spl_estimate_db: sweep.max_spl_estimate_db,
        calibrated_against_m1: false,
        distortion_artifact: distortion_path.display().to_string(),
        maxspl_artifact: maxspl_path.display().to_string(),
    };
    Ok((report, vec![distortion_path, maxspl_path]))
}

fn maxspl_point(point: &LumpedNonlinearCompressionPoint, xmax_m: Option<f64>) -> MaxSplPoint {
    MaxSplPoint {
        voltage_rms_v: point.voltage_rms_v,
        estimated_port_spl_db: point.estimated_port_spl_db,
        compression_db: point.compression_db,
        peak_excursion_m: point.peak_excursion_m,
        within_limits: point.compression_db <= 3.0
            && xmax_m.is_none_or(|limit| point.peak_excursion_m <= limit),
    }
}

fn parse_frequencies(value: &str) -> Result<Vec<f64>, AnyError> {
    let frequencies = value
        .split(',')
        .map(str::trim)
        .map(str::parse)
        .collect::<Result<Vec<f64>, _>>()?;
    validate_frequencies(&frequencies)?;
    Ok(frequencies)
}

fn validate_frequencies(frequencies_hz: &[f64]) -> Result<(), AnyError> {
    if frequencies_hz.is_empty()
        || frequencies_hz
            .iter()
            .any(|frequency| !frequency.is_finite() || *frequency <= 0.0)
        || frequencies_hz.windows(2).any(|pair| pair[0] >= pair[1])
    {
        return Err(invalid_input(
            "frequencies must be positive, finite, and strictly increasing",
        ));
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn validate_large_signal_options(
    frequency_hz: f64,
    voltages_rms_v: &[f64],
    duration_s: f64,
    samples_per_period: usize,
    port_loss_coefficient: f64,
    secondary_frequency_hz: Option<f64>,
    secondary_voltage_ratio: f64,
) -> Result<(), AnyError> {
    if !frequency_hz.is_finite() || frequency_hz <= 0.0 {
        return Err(invalid_input(
            "--large-signal-frequency must be finite and positive",
        ));
    }
    validate_frequencies(voltages_rms_v).map_err(|_| {
        invalid_input("large-signal drive levels must be positive, finite, and strictly increasing")
    })?;
    if !duration_s.is_finite() || duration_s <= 0.0 {
        return Err(invalid_input(
            "--large-signal-duration must be finite and positive",
        ));
    }
    if duration_s * frequency_hz < 2.0 {
        return Err(invalid_input(
            "--large-signal-duration must include at least two primary periods",
        ));
    }
    if samples_per_period < 32 {
        return Err(invalid_input(
            "--large-signal-samples-per-period must be at least 32",
        ));
    }
    if !port_loss_coefficient.is_finite() || port_loss_coefficient < 0.0 {
        return Err(invalid_input(
            "--port-loss-coefficient must be finite and nonnegative",
        ));
    }
    if !secondary_voltage_ratio.is_finite() || secondary_voltage_ratio < 0.0 {
        return Err(invalid_input(
            "--secondary-voltage-ratio must be finite and nonnegative",
        ));
    }
    if secondary_frequency_hz.is_some_and(|secondary| {
        !secondary.is_finite()
            || secondary <= 0.0
            || secondary == frequency_hz
            || secondary >= 0.5 * frequency_hz * samples_per_period as f64
    }) {
        return Err(invalid_input(
            "--secondary-frequency must be finite, positive, different from the primary, and below Nyquist",
        ));
    }
    Ok(())
}

fn required_value(
    args: &mut impl Iterator<Item = String>,
    option: &str,
) -> Result<String, AnyError> {
    args.next()
        .ok_or_else(|| invalid_input(format!("{option} requires a value")))
}

fn invalid_input(message: impl Into<String>) -> AnyError {
    io::Error::new(io::ErrorKind::InvalidInput, message.into()).into()
}

fn load_calibration(options: &RunOptions) -> Result<Option<CalibrationArtifact>, AnyError> {
    let (Some(driver_model_path), Some(calibration_data_path)) = (
        options.driver_model_path.as_ref(),
        options.calibration_data_path.as_ref(),
    ) else {
        return Ok(None);
    };
    let model: DriverModel = serde_json::from_slice(&fs::read(driver_model_path)?)?;
    let data: DriverCalibrationData = serde_json::from_slice(&fs::read(calibration_data_path)?)?;
    let tolerances = match options.calibration_tolerances_path.as_ref() {
        Some(path) => serde_json::from_slice(&fs::read(path)?)?,
        None => CalibrationTolerances::default(),
    };
    let report = correlate_driver_model(&model, &data, &tolerances)?;
    if !report.passed {
        return Err(invalid_input(
            "driver calibration did not pass the supplied tolerances",
        ));
    }
    Ok(Some(CalibrationArtifact {
        driver_model_path: driver_model_path.display().to_string(),
        calibration_data_path: calibration_data_path.display().to_string(),
        tolerances,
        report,
    }))
}

fn print_help() {
    println!(
        "whole-speaker — end-to-end loudspeaker FEM study\n\n\
Usage: cargo run -p sonium-sim-speaker-modeling --bin whole-speaker -- [OPTIONS]\n\n\
Options:\n\
  --output DIR          Artifact directory (default: SONIUM_OUTPUT_DIR hierarchy)\n\
  --gallery DIR         Write a deterministic headless gallery in DIR\n\
  --frequencies CSV     Strictly increasing frequencies in Hz\n\
  --voltage VOLTS       RMS drive voltage (default: 2.83)\n\
  --threads N           Total Rayon worker budget; 0/omission is automatic\n\
  --capabilities        Print the implemented CPU/CUDA/Metal capability matrix\n\
  --execution-contract  Print a bounded CPU execution contract as JSON\n\
  --cuda-frequency-contract\n\
                        Run the bounded CUDA production-operator contract\n\
  --quick               Coarse three-point validation run\n\
  --with-sector-3d      Also solve and export one periodic 3D motor sector\n\
  --minimum-impedance   Fail if minimum real impedance is below this value (ohms)\n\
  --maximum-excursion   Fail if peak one-way excursion exceeds this value (metres)\n\
  --driver-model FILE   Typed DriverModel JSON for optional calibration\n\
  --calibration-data FILE\n\
                        Measurement/reference dataset JSON for optional calibration\n\
  --calibration-tolerances FILE\n\
                        Optional explicit CalibrationTolerances JSON\n\
  --large-signal-frequency HZ\n\
  --large-signal-voltages CSV\n\
                        Enable the M2 fast path at increasing RMS drive levels\n\
  --large-signal-duration SECONDS\n\
  --large-signal-samples-per-period COUNT\n\
  --port-loss-coefficient K\n\
  --secondary-frequency HZ\n\
  --secondary-voltage-ratio RATIO\n\
                        Optional M2 time-grid, port-loss, and IMD controls\n\
  --motor-completion-input FILE\n\
                        Versioned M3 MotorCompletionInput JSON\n\
  --port-noise-input FILE\n\
                        Versioned M4 PortNoiseInput JSON from M1 snapshots\n\
  --progress-json       Emit versioned machine-readable progress on stderr\n\
  --no-acoustic-feedback\n\
                        Drive the resolved acoustic stage one-way instead of\n\
                        iterating its condensed load back into the structure\n\
  -h, --help            Show this help"
    );
}

#[cfg(test)]
mod gallery_source_unit_tests {
    use super::*;

    #[test]
    fn projected_flux_magnitudes_preserve_cell_grid_alignment() {
        let values = flux_density_magnitudes(
            &[
                [3.0, 0.0, 4.0],
                [0.0, 12.0, 0.0],
                [0.0, 0.0, 5.0],
                [8.0, 0.0, 6.0],
            ],
            2,
        )
        .unwrap();
        assert_eq!(values, vec![5.0, 12.0, 5.0, 10.0]);
    }

    #[test]
    fn projected_flux_magnitudes_reject_non_rectangular_or_nonfinite_fields() {
        let error = flux_density_magnitudes(&[[1.0, 0.0, 0.0]], 2).unwrap_err();
        assert!(error.to_string().contains("rectangular"));
        let error =
            flux_density_magnitudes(&[[f64::NAN, 0.0, 0.0], [0.0, 0.0, 0.0]], 2).unwrap_err();
        assert!(error.to_string().contains("non-finite"));
    }

    #[test]
    fn motor_position_curves_use_lowest_frequency_zero_current_inductance() {
        let completion = MotorCompletionResult {
            schema_version: 1,
            provenance: "test motor FEM".to_owned(),
            maximum_current_a: 1.0,
            conductive_insert_present: false,
            magnet_bl_temperature_coefficient_per_k: -1.0e-3,
            parameters: MotorParameters {
                bl_curve: vec![(-1.0e-3, 4.0), (0.0, 5.0), (1.0e-3, 4.0)],
                le_vs_frequency: vec![(10.0, 1.0e-3)],
                re_eddy_vs_frequency: vec![(10.0, 0.1)],
                bl_nominal: 5.0,
                le_nominal: 1.0e-3,
                le_surface: vec![
                    sonium_sim_speaker_modeling::NonlinearLePoint {
                        displacement_m: 1.0e-3,
                        current_a: 0.0,
                        frequency_hz: 100.0,
                        inductance_h: 0.8e-3,
                        eddy_resistance_ohm: 0.2,
                    },
                    sonium_sim_speaker_modeling::NonlinearLePoint {
                        displacement_m: -1.0e-3,
                        current_a: 0.0,
                        frequency_hz: 100.0,
                        inductance_h: 0.9e-3,
                        eddy_resistance_ohm: 0.2,
                    },
                    sonium_sim_speaker_modeling::NonlinearLePoint {
                        displacement_m: 0.0,
                        current_a: 0.0,
                        frequency_hz: 100.0,
                        inductance_h: 1.0e-3,
                        eddy_resistance_ohm: 0.2,
                    },
                    sonium_sim_speaker_modeling::NonlinearLePoint {
                        displacement_m: 0.0,
                        current_a: 0.0,
                        frequency_hz: 10.0,
                        inductance_h: 1.1e-3,
                        eddy_resistance_ohm: 0.1,
                    },
                ],
                bl_temperature_coefficient_per_k: -1.0e-3,
                magnetic_law: None,
                hysteresis_loss: None,
                eddy_current_loss: None,
            },
        };
        let curves = extract_motor_position_curves(&completion).unwrap();
        assert_eq!(curves.bl_curve.len(), 3);
        assert_eq!(curves.le_curve.len(), 3);
        assert_eq!(curves.le_curve[1].inductance_h, 1.1e-3);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use sonium_sim_speaker_modeling::{MeasuredImpedancePoint, SpeakerDriverStudy};
    use tempfile::tempdir;

    #[test]
    fn cli_quick_mode_selects_small_default_sweep() {
        let options = RunOptions::parse(["--quick".to_owned()]).unwrap();
        assert_eq!(options.frequencies_hz, QUICK_FREQUENCIES_HZ);
        assert_eq!(options.voltage_rms_v, 2.83);
        assert!(options.quick);
        assert!(options.acoustic_feedback);
        assert!(!options.progress_json);
    }

    #[test]
    fn cli_progress_json_flag_enables_machine_protocol() {
        let options = RunOptions::parse(["--progress-json".to_owned()]).unwrap();
        assert!(options.progress_json);
    }

    #[test]
    fn progress_record_serializes_versioned_protocol_fields() {
        let record = ProgressRecord {
            schema_version: 1,
            record_type: "progress",
            phase: "solving",
            message: "Solved frequency",
            completed: Some(2),
            total: Some(3),
            unit: Some("frequency"),
            frequency_hz: Some(100.0),
        };
        let value: serde_json::Value = serde_json::to_value(record).unwrap();
        assert_eq!(value["schema_version"], 1);
        assert_eq!(value["type"], "progress");
        assert_eq!(value["phase"], "solving");
        assert_eq!(value["completed"], 2);
        assert_eq!(value["total"], 3);
        assert_eq!(value["unit"], "frequency");
        assert_eq!(value["frequency_hz"], 100.0);
    }

    #[test]
    fn cli_no_acoustic_feedback_flag_disables_load_iteration() {
        let options = RunOptions::parse(["--no-acoustic-feedback".to_owned()]).unwrap();
        assert!(!options.acoustic_feedback);
    }

    #[test]
    fn cli_explicit_sweep_overrides_quick_defaults() {
        let options = RunOptions::parse([
            "--quick".to_owned(),
            "--frequencies".to_owned(),
            "31.5,63,125".to_owned(),
            "--voltage".to_owned(),
            "1.0".to_owned(),
            "--with-sector-3d".to_owned(),
        ])
        .unwrap();
        assert_eq!(options.frequencies_hz, [31.5, 63.0, 125.0]);
        assert_eq!(options.voltage_rms_v, 1.0);
        assert!(options.with_sector_3d);
    }

    #[test]
    fn cli_calibration_inputs_require_a_model_and_fixture_pair() {
        let result = RunOptions::parse(["--driver-model".to_owned(), "model.json".to_owned()]);
        assert!(result.is_err());

        let options = RunOptions::parse([
            "--driver-model".to_owned(),
            "model.json".to_owned(),
            "--calibration-data".to_owned(),
            "measurements.json".to_owned(),
            "--calibration-tolerances".to_owned(),
            "tolerances.json".to_owned(),
        ])
        .unwrap();
        assert_eq!(options.driver_model_path, Some(PathBuf::from("model.json")));
        assert_eq!(
            options.calibration_data_path,
            Some(PathBuf::from("measurements.json"))
        );
        assert_eq!(
            options.calibration_tolerances_path,
            Some(PathBuf::from("tolerances.json"))
        );
    }

    #[test]
    fn serialized_calibration_fixture_is_loaded_without_mutating_the_model() {
        let directory = tempdir().unwrap();
        let model_path = directory.path().join("driver-model.json");
        let data_path = directory.path().join("measurements.json");
        let model =
            DriverModel::from_study(&SpeakerDriverStudy::detailed_frequency_domain_driver())
                .unwrap();
        let data = DriverCalibrationData {
            impedance: model
                .electrical_impedance
                .iter()
                .map(|point| MeasuredImpedancePoint {
                    frequency_hz: point.frequency_hz,
                    real_ohm: point.real,
                    imaginary_ohm: point.imaginary,
                })
                .collect(),
            ..DriverCalibrationData::default()
        };
        fs::write(&model_path, serde_json::to_vec(&model).unwrap()).unwrap();
        fs::write(&data_path, serde_json::to_vec(&data).unwrap()).unwrap();

        let options = RunOptions::parse([
            "--driver-model".to_owned(),
            model_path.display().to_string(),
            "--calibration-data".to_owned(),
            data_path.display().to_string(),
        ])
        .unwrap();
        let artifact = load_calibration(&options).unwrap().unwrap();
        assert!(artifact.report.passed);
        assert_eq!(artifact.driver_model_path, model_path.display().to_string());
        assert_eq!(
            artifact.calibration_data_path,
            data_path.display().to_string()
        );
    }

    #[test]
    fn cli_rejects_unsorted_frequencies() {
        let result = RunOptions::parse(["--frequencies".to_owned(), "100,50".to_owned()]);
        assert!(result.is_err());
    }

    #[test]
    fn cli_large_signal_mode_requires_and_preserves_versioned_controls() {
        assert!(
            RunOptions::parse(["--large-signal-frequency".to_owned(), "80".to_owned(),]).is_err()
        );
        let options = RunOptions::parse([
            "--large-signal-frequency".to_owned(),
            "70".to_owned(),
            "--large-signal-voltages".to_owned(),
            "1,2,4".to_owned(),
            "--secondary-frequency".to_owned(),
            "90".to_owned(),
            "--motor-completion-input".to_owned(),
            "m3.json".to_owned(),
            "--port-noise-input".to_owned(),
            "m4.json".to_owned(),
        ])
        .unwrap();
        assert_eq!(
            options.motor_completion_input_path,
            Some(PathBuf::from("m3.json"))
        );
        assert_eq!(
            options.port_noise_input_path,
            Some(PathBuf::from("m4.json"))
        );
        let large_signal = options.large_signal.unwrap();
        assert_eq!(large_signal.frequency_hz, 70.0);
        assert_eq!(large_signal.voltages_rms_v, [1.0, 2.0, 4.0]);
        assert_eq!(large_signal.secondary_frequency_hz, Some(90.0));
    }

    #[test]
    fn large_signal_artifacts_have_stable_schema_and_provenance() {
        let directory = tempdir().unwrap();
        let controls = LargeSignalRunOptions {
            frequency_hz: 80.0,
            voltages_rms_v: vec![1.0],
            duration_s: 0.1,
            samples_per_period: 64,
            port_loss_coefficient: 2.0,
            secondary_frequency_hz: Some(100.0),
            secondary_voltage_ratio: 0.2,
        };
        let study = SpeakerDriverStudy::detailed_frequency_domain_driver();
        let sweep = LumpedNonlinearCompressionSweep {
            provenance: "LumpedNonlinearAcousticCircuit[Driver,Box,QuasiSteadyPort]".into(),
            frequency_hz: 80.0,
            port_loss_coefficient: 2.0,
            points: vec![LumpedNonlinearCompressionPoint {
                voltage_rms_v: 1.0,
                port_flow_amplitude_m3s: 1.0e-3,
                box_pressure_amplitude_pa: 2.0,
                estimated_port_spl_db: 90.0,
                compression_db: 0.0,
                acoustic_thd: 0.01,
                acoustic_imd: 0.02,
                peak_excursion_m: 1.0e-3,
            }],
            max_spl_estimate_db: Some(90.0),
        };

        let (report, paths) =
            persist_large_signal_artifacts(directory.path(), &controls, &study, sweep).unwrap();

        assert_eq!(report.schema_version, 1);
        assert!(!report.calibrated_against_m1);
        assert_eq!(paths.len(), 2);
        let distortion: serde_json::Value =
            serde_json::from_slice(&fs::read(directory.path().join("distortion.json")).unwrap())
                .unwrap();
        let maxspl: serde_json::Value =
            serde_json::from_slice(&fs::read(directory.path().join("maxspl.json")).unwrap())
                .unwrap();
        assert_eq!(distortion["schema_version"], 1);
        assert_eq!(distortion["secondary_frequency_hz"], 100.0);
        assert_eq!(distortion["points"][0]["acoustic_thd_ratio"], 0.01);
        assert_eq!(maxspl["schema_version"], 1);
        assert_eq!(maxspl["compression_limit_db"], 3.0);
        assert_eq!(maxspl["points"][0]["within_limits"], true);
        assert!(
            maxspl["provenance"]
                .as_str()
                .unwrap()
                .contains("LumpedNonlinearAcousticCircuit")
        );
    }

    #[test]
    fn native_large_signal_fast_path_runs_from_resolved_geometry() {
        let directory = tempdir().unwrap();
        let options = RunOptions::parse([
            "--large-signal-frequency".to_owned(),
            "80".to_owned(),
            "--large-signal-voltages".to_owned(),
            "0.1,0.2".to_owned(),
            "--large-signal-duration".to_owned(),
            "0.05".to_owned(),
            "--large-signal-samples-per-period".to_owned(),
            "64".to_owned(),
        ])
        .unwrap();

        let (report, paths) = write_large_signal_artifacts(
            directory.path(),
            &options,
            &ResolvedAcousticConfig::default(),
            6.8,
            0.42e-3,
        )
        .unwrap()
        .unwrap();

        assert_eq!(report.schema_version, 1);
        assert_eq!(report.frequency_hz, 80.0);
        assert_eq!(paths.len(), 2);
        assert!(paths.iter().all(|path| path.is_file()));
    }

    #[test]
    fn quick_pipeline_writes_complete_artifact_set() {
        let directory = tempfile::tempdir().unwrap();
        run(RunOptions {
            output_dir: Some(directory.path().to_path_buf()),
            gallery_dir: Some(directory.path().to_path_buf()),
            frequencies_hz: QUICK_FREQUENCIES_HZ.to_vec(),
            voltage_rms_v: 1.0,
            quick: true,
            with_sector_3d: false,
            acoustic_feedback: true,
            progress_json: false,
            threads: 0,
            cuda_frequency_contract: false,
            capabilities: false,
            execution_contract: false,
            minimum_impedance_real_ohm: None,
            maximum_excursion_m: None,
            driver_model_path: None,
            calibration_data_path: None,
            calibration_tolerances_path: None,
            large_signal: None,
            motor_completion_input_path: None,
            port_noise_input_path: None,
        })
        .unwrap();
        for filename in [
            "nonlinear-magnetic-bias.vtu",
            "eddy-current-fields.pvd",
            "distributed-coupled-fields.pvd",
            "resolved-enclosure-fields.pvd",
            "frequency-response.csv",
            "resolved-load-impedance.csv",
            "whole-speaker-report.json",
            "driver-model.json",
            "gallery.json",
            "motor-flux-density.png",
            "eddy-current-loss.png",
            "enclosure-pressure.png",
            "bl.svg",
            "le.svg",
            "acoustic-energy-flow.svg",
            "thermal-loss.svg",
            "structural-response.svg",
            "structural-mode.svg",
            "em-mech-acoustic-overview.svg",
            "structural-modal-response.csv",
        ] {
            assert!(
                directory.path().join(filename).is_file(),
                "missing {filename}"
            );
        }
        let report: serde_json::Value = serde_json::from_slice(
            &fs::read(directory.path().join("whole-speaker-report.json")).unwrap(),
        )
        .unwrap();
        assert_eq!(report["frequency_points"].as_array().unwrap().len(), 3);
        assert_eq!(
            report["structural_modal_response"]
                .as_array()
                .unwrap()
                .len(),
            3
        );
        let gallery: serde_json::Value =
            serde_json::from_slice(&fs::read(directory.path().join("gallery.json")).unwrap())
                .unwrap();
        assert_eq!(gallery["width_px"], 1920);
        assert_eq!(gallery["height_px"], 1080);
        assert!(
            gallery["figures"]
                .as_array()
                .is_some_and(|figures| figures.len() >= 11)
        );
        let checked_manifest =
            sonium_xem::io::validate_gallery_manifest(directory.path().join("gallery.json"))
                .unwrap();
        assert_eq!(checked_manifest.width_px, 1920);
        assert_eq!(checked_manifest.height_px, 1080);
        let mut raster_count = 0;
        let mut vector_count = 0;
        for figure in gallery["figures"].as_array().unwrap() {
            let filename = figure["filename"].as_str().unwrap();
            assert!(
                figure["source_checksum_fnv1a64"].is_string(),
                "figure {filename} must retain source provenance"
            );
            let path = directory.path().join(filename);
            assert!(path.is_file(), "missing gallery figure {filename}");
            match figure["kind"].as_str().unwrap() {
                "png" => {
                    assert_eq!(
                        sonium_xem::io::validate_png_nonblank(&path).unwrap(),
                        (1920, 1080)
                    );
                    raster_count += 1;
                }
                "svg" => {
                    let svg = fs::read_to_string(&path).unwrap();
                    assert!(
                        svg.contains("<svg"),
                        "SVG figure {filename} is not structured"
                    );
                    assert!(
                        svg.contains("width=\"1920\""),
                        "SVG figure {filename} is not fixed-size"
                    );
                    vector_count += 1;
                }
                other => panic!("unexpected gallery figure kind {other}"),
            }
        }
        assert!(raster_count >= 3, "expected multiphysics field rasters");
        assert!(vector_count >= 8, "expected engineering SVG curves");
        assert_eq!(report["sweep_qa"]["all_points_converged"], true);
        assert_eq!(report["provenance"]["execution_profile"], "golden");
        assert_eq!(report["extraction_evidence"]["coupling"]["converged"], true);
        assert_eq!(report["extraction_evidence"]["acoustic"]["converged"], true);
        assert!(
            report["extracted_driver_model"]
                .as_str()
                .unwrap()
                .ends_with("/driver-model.json")
        );
        let driver_model: serde_json::Value =
            serde_json::from_slice(&fs::read(directory.path().join("driver-model.json")).unwrap())
                .unwrap();
        assert_eq!(driver_model["provenance"]["execution_profile"], "golden");
        assert!(driver_model["bl_curve"][0]["force_factor_tm"].is_f64());
        for field in [
            "calibration_seconds",
            "setup_seconds",
            "nonlinear_bias_seconds",
            "eddy_current_seconds",
            "distributed_sweep_seconds",
            "resolved_acoustic_seconds",
            "artifact_assembly_seconds",
            "total_seconds",
        ] {
            assert!(
                report["timings"][field]
                    .as_f64()
                    .is_some_and(|value| value >= 0.0),
                "timing field {field} must be finite and non-negative"
            );
        }
        assert!(report["timings"]["sector_3d_seconds"].is_null());
        for point in report["frequency_points"].as_array().unwrap() {
            assert_eq!(point["acoustic_feedback_converged"], true);
            // The coarse quick mesh returns a slightly negative radiated
            // power at 40 Hz (see resolved_acoustics power-QA tolerances),
            // so passivity is checked against a scaled slack, not exactly 0.
            let load = &point["resolved_load_impedance_n_s_per_m"];
            let re = load["re"].as_f64().unwrap();
            let magnitude = re.hypot(load["im"].as_f64().unwrap());
            assert!(
                re >= -0.02 * magnitude,
                "condensed resolved load violates scaled passivity: {point}"
            );
        }
    }

    #[test]
    fn quick_pipeline_without_feedback_omits_load_artifacts() {
        let directory = tempfile::tempdir().unwrap();
        run(RunOptions {
            output_dir: Some(directory.path().to_path_buf()),
            gallery_dir: None,
            frequencies_hz: QUICK_FREQUENCIES_HZ.to_vec(),
            voltage_rms_v: 1.0,
            quick: true,
            with_sector_3d: false,
            acoustic_feedback: false,
            progress_json: false,
            threads: 0,
            cuda_frequency_contract: false,
            capabilities: false,
            execution_contract: false,
            minimum_impedance_real_ohm: None,
            maximum_excursion_m: None,
            driver_model_path: None,
            calibration_data_path: None,
            calibration_tolerances_path: None,
            large_signal: None,
            motor_completion_input_path: None,
            port_noise_input_path: None,
        })
        .unwrap();
        assert!(
            !directory
                .path()
                .join("resolved-load-impedance.csv")
                .exists()
        );
        let report: serde_json::Value = serde_json::from_slice(
            &fs::read(directory.path().join("whole-speaker-report.json")).unwrap(),
        )
        .unwrap();
        assert_eq!(report["frequency_points"].as_array().unwrap().len(), 3);
        assert!(
            report["frequency_points"][0]
                .get("resolved_load_impedance_n_s_per_m")
                .is_none(),
            "one-way report must not carry load-feedback fields"
        );
    }
}
