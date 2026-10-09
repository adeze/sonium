//! Typed handoff from driver-unit modeling to product-level speaker simulation.
//!
//! This is the only contract product crates need from the driver modeling
//! layer: small-signal T/S data, displacement-dependent BL, frequency-dependent
//! coil inductance/loss, and the resulting diaphragm velocity transfer.

use num_complex::Complex64;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::axisymmetric_linear::AxisymmetricSweepQa;
use crate::conventions::ArtifactProvenance;
use crate::motor_extraction::MotorParameters;
use crate::nonlinear_bias::NonlinearSpeakerBiasResult;
use crate::nonlinear_motor::{
    HysteresisLossModel, MotorConstitutiveModel, MotorOperatingDomain,
    MotorOperatingDomainEvidence, MotorOperatingPoint, MotorTemperatureLaw, NonlinearBlPoint,
    NonlinearLePoint,
};
use crate::resolved_acoustics::ResolvedAcousticSolution;
use crate::resolved_coupling::ResolvedCouplingPoint;
use crate::structural_constraints::StructuralEvaluation;
use crate::{
    DriverThermalNetworkConfig, ModelingError, SpeakerDriverStudy, SpeakerEddyCurrentSweep,
};

/// Residual evidence required at a golden driver-model handoff.
///
/// `converged` is retained separately from the numeric residual because a
/// finite-looking field can still come from a solver that exhausted its
/// iteration budget. The common tolerance is intentionally conservative for
/// this cross-domain promotion boundary; individual solvers may use tighter
/// internal tolerances.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ExtractionResidualEvidence {
    pub converged: bool,
    pub relative_residual: f64,
}

impl Default for ExtractionResidualEvidence {
    fn default() -> Self {
        Self::new(false, f64::INFINITY)
    }
}

impl ExtractionResidualEvidence {
    pub const fn new(converged: bool, relative_residual: f64) -> Self {
        Self {
            converged,
            relative_residual,
        }
    }

    fn passes(self, stage: &'static str) -> Result<(), DriverModelError> {
        self.passes_with_tolerance(stage, EXTRACTION_RESIDUAL_TOLERANCE)
    }

    fn passes_with_tolerance(
        self,
        stage: &'static str,
        tolerance: f64,
    ) -> Result<(), DriverModelError> {
        if self.converged
            && self.relative_residual.is_finite()
            && self.relative_residual >= 0.0
            && self.relative_residual <= tolerance
        {
            Ok(())
        } else {
            Err(DriverModelError::ExtractionEvidenceFailed {
                stage,
                converged: self.converged,
                relative_residual: self.relative_residual,
            })
        }
    }
}

const EXTRACTION_RESIDUAL_TOLERANCE: f64 = 1.0e-6;
/// The resolved acoustic discretization reports a physical power balance in
/// addition to its algebraic solve residual. Keep this gate separate because
/// the balance also contains quadrature/PML truncation error.
const POWER_CLOSURE_RELATIVE_TOLERANCE: f64 = 5.0e-2;

/// Schema version for portable driver-model reference artifacts.
///
/// The artifact stores the boundary load together with the response. A
/// replay therefore checks the same model contract instead of comparing a
/// response table whose loading convention could have changed.
pub const DRIVER_MODEL_REFERENCE_ARTIFACT_SCHEMA_VERSION: u32 = 2;

/// Convergence evidence attached to one driver-model reference envelope.
///
/// The response table is not promotable merely because its values are finite:
/// the producing solve must also report a converged residual and a closed
/// power balance.  These fields intentionally remain scalar summary values so
/// the envelope stays portable; subsystem-specific evidence remains in the
/// full-order study report.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct DriverModelConvergenceEvidence {
    pub converged: bool,
    pub max_relative_residual: f64,
    pub power_closure_relative_error: f64,
    pub coupling_iterations: u32,
}

impl DriverModelConvergenceEvidence {
    pub const fn cpu_golden() -> Self {
        Self {
            converged: true,
            max_relative_residual: 0.0,
            power_closure_relative_error: 0.0,
            coupling_iterations: 0,
        }
    }
}

/// Device transfer and residency evidence for a reference replay.
///
/// A CUDA artifact must retain these counters from the live backend interval;
/// zeroed or fabricated counters are rejected for a native CUDA execution.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct DriverModelTransferEvidence {
    pub h2d_operations: u64,
    pub h2d_bytes: u64,
    pub d2h_operations: u64,
    pub d2h_bytes: u64,
    pub live_allocation_bytes: u64,
    pub allocation_high_water_bytes: u64,
    pub live_staging_bytes: u64,
    pub staging_high_water_bytes: u64,
}

/// Full-order/projected evidence required before a ROM can be promoted.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct DriverModelPromotionEvidence {
    pub promoted: bool,
    pub full_order_relative_residual: f64,
    pub projected_relative_residual: f64,
    pub gradient_relative_error: f64,
    pub maximum_constraint_violation: f64,
    pub rank_agreement: f64,
}

impl DriverModelPromotionEvidence {
    pub const fn cpu_golden() -> Self {
        Self {
            promoted: true,
            full_order_relative_residual: 0.0,
            projected_relative_residual: 0.0,
            gradient_relative_error: 0.0,
            maximum_constraint_violation: 0.0,
            rank_agreement: 1.0,
        }
    }

    fn validate(self) -> Result<(), DriverModelError> {
        let values = [
            self.full_order_relative_residual,
            self.projected_relative_residual,
            self.gradient_relative_error,
            self.maximum_constraint_violation,
            self.rank_agreement,
        ];
        if values
            .iter()
            .any(|value| !value.is_finite() || *value < 0.0)
            || self.rank_agreement > 1.0
        {
            return Err(DriverModelError::InvalidReferenceArtifact {
                reason: "ROM promotion evidence must be finite and non-negative",
            });
        }
        Ok(())
    }

    fn validate_gate(self) -> Result<(), DriverModelError> {
        self.validate()?;
        if !self.promoted
            || self.full_order_relative_residual > EXTRACTION_RESIDUAL_TOLERANCE
            || self.projected_relative_residual > EXTRACTION_RESIDUAL_TOLERANCE
            || self.gradient_relative_error > 1.0e-5
            || self.maximum_constraint_violation > 1.0e-10
            || self.rank_agreement < 1.0 - 1.0e-10
        {
            return Err(DriverModelError::InvalidReferenceArtifact {
                reason: "ROM promotion gates were not satisfied",
            });
        }
        Ok(())
    }
}

/// Execution identity and acceptance evidence carried by every reference
/// artifact.  `actual_backend` is checked against the artifact provenance;
/// this prevents a CPU fallback from being relabelled as CUDA in-place.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DriverModelReferenceExecution {
    pub requested_backend: crate::conventions::ArtifactBackend,
    pub actual_backend: crate::conventions::ArtifactBackend,
    pub precision: crate::conventions::ArtifactPrecision,
    pub device: String,
    pub toolchain: String,
    pub mesh_identity: String,
    pub frequency_grid_identity: String,
    pub convergence: DriverModelConvergenceEvidence,
    pub transfer: DriverModelTransferEvidence,
    pub promotion: DriverModelPromotionEvidence,
    pub reference_checksum_sha256: String,
}

impl DriverModelReferenceExecution {
    fn cpu_golden(frequencies: &[f64]) -> Self {
        Self {
            requested_backend: crate::conventions::ArtifactBackend::Cpu,
            actual_backend: crate::conventions::ArtifactBackend::Cpu,
            precision: crate::conventions::ArtifactPrecision::F64,
            device: "cpu".to_string(),
            toolchain: env!("SONIUM_BUILD_RUSTC_VERSION").to_string(),
            mesh_identity: "driver-model:lumped-network:v1".to_string(),
            frequency_grid_identity: frequency_grid_identity(frequencies),
            convergence: DriverModelConvergenceEvidence::cpu_golden(),
            transfer: DriverModelTransferEvidence::default(),
            promotion: DriverModelPromotionEvidence::cpu_golden(),
            reference_checksum_sha256: String::new(),
        }
    }

    fn validate(&self, provenance: ArtifactProvenance) -> Result<(), DriverModelError> {
        if self.actual_backend != provenance.backend {
            return Err(DriverModelError::InvalidReferenceArtifact {
                reason: "actual backend does not match artifact provenance",
            });
        }
        if self.precision != provenance.precision {
            return Err(DriverModelError::InvalidReferenceArtifact {
                reason: "execution precision does not match artifact provenance",
            });
        }
        if self.actual_backend == crate::conventions::ArtifactBackend::Cuda
            && self.requested_backend != crate::conventions::ArtifactBackend::Cuda
        {
            return Err(DriverModelError::InvalidReferenceArtifact {
                reason: "CUDA output must originate from a requested CUDA backend",
            });
        }
        if self.device.trim().is_empty() || self.toolchain.trim().is_empty() {
            return Err(DriverModelError::InvalidReferenceArtifact {
                reason: "reference execution identity is incomplete",
            });
        }
        if self.mesh_identity.trim().is_empty() || self.frequency_grid_identity.len() != 64 {
            return Err(DriverModelError::InvalidReferenceArtifact {
                reason: "reference mesh or frequency-grid identity is incomplete",
            });
        }
        if self.reference_checksum_sha256.len() != 64
            || !self
                .reference_checksum_sha256
                .bytes()
                .all(|byte| byte.is_ascii_hexdigit())
        {
            return Err(DriverModelError::InvalidReferenceArtifact {
                reason: "reference checksum must be a SHA-256 hex digest",
            });
        }
        let convergence = self.convergence;
        if !convergence.converged
            || !convergence.max_relative_residual.is_finite()
            || convergence.max_relative_residual < 0.0
            || convergence.max_relative_residual > EXTRACTION_RESIDUAL_TOLERANCE
            || !convergence.power_closure_relative_error.is_finite()
            || convergence.power_closure_relative_error < 0.0
            || convergence.power_closure_relative_error > POWER_CLOSURE_RELATIVE_TOLERANCE
        {
            return Err(DriverModelError::InvalidReferenceArtifact {
                reason: "reference convergence or power-closure evidence failed",
            });
        }
        let transfer = self.transfer;
        if transfer.allocation_high_water_bytes < transfer.live_allocation_bytes
            || transfer.staging_high_water_bytes < transfer.live_staging_bytes
        {
            return Err(DriverModelError::InvalidReferenceArtifact {
                reason: "reference memory watermarks are inconsistent",
            });
        }
        if self.actual_backend == crate::conventions::ArtifactBackend::Cuda
            && (self.precision != crate::conventions::ArtifactPrecision::F64
                || transfer.h2d_operations == 0
                || transfer.h2d_bytes == 0
                || transfer.d2h_operations == 0
                || transfer.d2h_bytes == 0
                || transfer.allocation_high_water_bytes == 0)
        {
            return Err(DriverModelError::InvalidReferenceArtifact {
                reason: "CUDA artifacts require live f64 host/device transfer and memory evidence",
            });
        }
        self.promotion.validate()
    }
}

/// Cross-domain evidence needed before a FEM-derived driver state is promoted
/// into the product-level `DriverModel` contract.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct DriverModelExtractionEvidence {
    pub nonlinear_bias: ExtractionResidualEvidence,
    pub differential_permeability: ExtractionResidualEvidence,
    pub eddy_current: ExtractionResidualEvidence,
    pub structural: ExtractionResidualEvidence,
    pub acoustic: ExtractionResidualEvidence,
    #[serde(default)]
    pub power_closure: ExtractionResidualEvidence,
    pub coupling: ExtractionResidualEvidence,
}

impl DriverModelExtractionEvidence {
    /// Summarize the native full-order stages used by the whole-speaker
    /// runner. The resulting value is compact enough to serialize in a report
    /// while retaining every acceptance gate needed by the handoff.
    pub fn from_solve_outputs(
        bias: &NonlinearSpeakerBiasResult,
        eddy: &SpeakerEddyCurrentSweep,
        sweep_qa: &AxisymmetricSweepQa,
        structural: &StructuralEvaluation,
        acoustic: &[ResolvedAcousticSolution],
        coupling: Option<&[ResolvedCouplingPoint]>,
    ) -> Self {
        let nonlinear_residual = bias.solution.final_residual;
        let nonlinear_converged = nonlinear_residual.is_finite()
            && (0.0..=EXTRACTION_RESIDUAL_TOLERANCE).contains(&nonlinear_residual)
            && !bias.solution.continuation_iterations.is_empty();
        let differential_permeability_valid = !bias.solution.quadrature_state.is_empty()
            && bias.solution.quadrature_state.iter().all(|point| {
                point.differential_reluctivity_m_per_h.is_finite()
                    && point.differential_reluctivity_m_per_h > 0.0
            });
        // Differential permeability is evaluated from the converged magnetic
        // state, so it inherits the nonlinear solve residual. Do not report a
        // fabricated zero merely because all pointwise values are finite.
        let differential_residual = if differential_permeability_valid {
            nonlinear_residual
        } else {
            f64::INFINITY
        };

        let eddy_residual = eddy
            .points
            .iter()
            .map(|point| point.relative_residual)
            .fold(0.0, f64::max);
        let eddy_converged = !eddy.points.is_empty()
            && eddy.points.iter().all(|point| {
                point.converged
                    && point.relative_residual.is_finite()
                    && point.relative_residual >= 0.0
                    && point.relative_residual <= EXTRACTION_RESIDUAL_TOLERANCE
            });

        let structural_values_finite = structural
            .peak_excursion_m
            .into_iter()
            .chain(structural.peak_stress_pa.values().copied())
            .chain(structural.panel_peak_displacement_m.iter().copied())
            .chain(structural.panel_peak_bending_stress_pa.iter().copied())
            .all(f64::is_finite);
        let structural_residual = sweep_qa
            .max_relative_linear_residual
            .max(sweep_qa.max_relative_power_residual);
        let structural_converged = sweep_qa.all_points_converged
            && structural_residual.is_finite()
            && structural_residual <= EXTRACTION_RESIDUAL_TOLERANCE
            && sweep_qa.max_reciprocity_defect.is_finite()
            && sweep_qa.max_reciprocity_defect <= EXTRACTION_RESIDUAL_TOLERANCE
            && sweep_qa.minimum_impedance_real_ohm.is_finite()
            && sweep_qa.minimum_impedance_real_ohm >= -1.0e-10
            && sweep_qa.minimum_dissipated_power_w.is_finite()
            && sweep_qa.minimum_dissipated_power_w >= -1.0e-10
            && structural_values_finite;

        let acoustic_residual = acoustic
            .iter()
            .map(|solution| solution.solve_diagnostics.relative_residual)
            .fold(0.0, f64::max);
        let acoustic_converged = !acoustic.is_empty()
            && acoustic.iter().all(|solution| {
                solution.solve_diagnostics.relative_residual.is_finite()
                    && solution.solve_diagnostics.relative_residual >= 0.0
                    && solution.solve_diagnostics.relative_residual <= EXTRACTION_RESIDUAL_TOLERANCE
            });
        let power_closure_residual = acoustic
            .iter()
            .map(|solution| {
                let scale = solution
                    .driver_input_power_w
                    .abs()
                    .max(solution.port_thermal_loss_w.abs())
                    .max(solution.cavity_wall_loss_w.abs())
                    .max(solution.radiated_power_w.abs())
                    .max(f64::MIN_POSITIVE);
                solution.power_balance_residual_w.abs() / scale
            })
            .fold(0.0, f64::max);
        let power_closure_converged = !acoustic.is_empty()
            && power_closure_residual.is_finite()
            && power_closure_residual <= POWER_CLOSURE_RELATIVE_TOLERANCE;

        let (coupling_residual, coupling_converged) =
            coupling.map_or((f64::INFINITY, false), |points| {
                (
                    points
                        .iter()
                        .map(|point| {
                            point
                                .relative_velocity_change
                                .max(point.monolithic_agreement_relative_error)
                        })
                        .fold(0.0, f64::max),
                    !points.is_empty()
                        && points.iter().all(|point| {
                            point.converged
                                && point.relative_velocity_change.is_finite()
                                && point.relative_velocity_change <= EXTRACTION_RESIDUAL_TOLERANCE
                                && point.monolithic_agreement_relative_error.is_finite()
                                && point.monolithic_agreement_relative_error
                                    <= EXTRACTION_RESIDUAL_TOLERANCE
                                && point.maximum_contraction_ratio.is_finite()
                                && point.maximum_contraction_ratio < 1.0
                        }),
                )
            });

        Self {
            nonlinear_bias: ExtractionResidualEvidence::new(
                nonlinear_converged,
                nonlinear_residual,
            ),
            differential_permeability: ExtractionResidualEvidence::new(
                differential_permeability_valid && nonlinear_converged,
                differential_residual,
            ),
            eddy_current: ExtractionResidualEvidence::new(eddy_converged, eddy_residual),
            structural: ExtractionResidualEvidence::new(structural_converged, structural_residual),
            acoustic: ExtractionResidualEvidence::new(acoustic_converged, acoustic_residual),
            power_closure: ExtractionResidualEvidence::new(
                power_closure_converged,
                power_closure_residual,
            ),
            coupling: ExtractionResidualEvidence::new(coupling_converged, coupling_residual),
        }
    }

    pub fn validate(self) -> Result<(), DriverModelError> {
        self.nonlinear_bias.passes("nonlinear bias")?;
        self.differential_permeability
            .passes("differential permeability")?;
        self.eddy_current.passes("eddy current")?;
        self.structural.passes("structural")?;
        self.acoustic.passes("acoustic")?;
        self.power_closure
            .passes_with_tolerance("power closure", POWER_CLOSURE_RELATIVE_TOLERANCE)?;
        self.coupling.passes("coupling")?;
        Ok(())
    }
}

/// Complex dimensionless voltage gain in the modeling exp(+i omega t) convention.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct DriveTransferPoint {
    pub frequency_hz: f64,
    pub real: f64,
    pub imag: f64,
}
impl FrequencyPoint for DriveTransferPoint {
    fn frequency_hz(self) -> f64 {
        self.frequency_hz
    }
}

/// Small-signal Thiele/Small parameters and their physical source values.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ThieleSmallParameters {
    pub voice_coil_resistance_ohm: f64,
    pub nominal_inductance_h: f64,
    pub force_factor_tm: f64,
    pub moving_mass_kg: f64,
    pub compliance_m_per_n: f64,
    pub mechanical_resistance_kg_per_s: f64,
    pub effective_piston_area_m2: f64,
    pub resonance_hz: f64,
    pub mechanical_q: Option<f64>,
    pub electrical_q: Option<f64>,
    pub total_q: Option<f64>,
    pub equivalent_compliance_volume_m3: f64,
}

/// One sample of the displacement-dependent force factor.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct BlPoint {
    pub displacement_m: f64,
    pub force_factor_tm: f64,
}

/// One sample of the frequency-dependent coil impedance contribution.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct InductancePoint {
    pub frequency_hz: f64,
    pub inductance_h: f64,
    pub eddy_resistance_ohm: f64,
}

/// Complex diaphragm velocity at a specified RMS reference voltage.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct VelocityTransferPoint {
    pub frequency_hz: f64,
    pub velocity_real_m_per_s: f64,
    pub velocity_imag_m_per_s: f64,
}

impl VelocityTransferPoint {
    pub fn velocity(self) -> Complex64 {
        Complex64::new(self.velocity_real_m_per_s, self.velocity_imag_m_per_s)
    }
}

/// Complex impedance sample represented in serde-friendly Cartesian form.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ImpedancePoint {
    pub frequency_hz: f64,
    pub real: f64,
    pub imaginary: f64,
}

impl ImpedancePoint {
    pub fn impedance(self) -> Complex64 {
        Complex64::new(self.real, self.imaginary)
    }
}

/// Electrical and intrinsic-mechanical network coefficients at one frequency.
///
/// These are the coefficients of the small-signal two-port
///
/// ```text
/// V   = Z_e I + BL v
/// BL I = Z_m v + F_rad
/// ```
///
/// with `Z_e` the blocked voice-coil impedance (resistance + eddy loss +
/// iωL) and `Z_m` the intrinsic mechanical impedance excluding the exterior
/// radiation load. Product simulation combines them with a resolved
/// self/mutual radiation-impedance matrix to solve coupled multi-driver
/// states directly instead of iterating a scalar `force / velocity` load.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DriverNetworkParameters {
    /// Blocked voice-coil impedance `R_e + R_eddy(f) + iωL_e(f)` (ohm).
    pub coil_impedance_ohm: Complex64,
    /// Force factor BL at zero displacement (T·m).
    pub force_factor_tm: f64,
    /// Intrinsic mechanical impedance excluding the radiation load (N·s/m).
    pub intrinsic_mechanical_impedance: Complex64,
}

/// One deterministic frequency-domain point in a driver-model reference
/// artifact. Complex values use the speaker model's RMS, `exp(+iωt)`
/// convention and are stored as Cartesian pairs for portable serde output.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct DriverModelReferencePoint {
    pub frequency_hz: f64,
    pub radiation_impedance_real_n_s_per_m: f64,
    pub radiation_impedance_imag_n_s_per_m: f64,
    pub velocity_real_m_per_s: f64,
    pub velocity_imag_m_per_s: f64,
    pub electrical_impedance_real_ohm: f64,
    pub electrical_impedance_imag_ohm: f64,
}

impl DriverModelReferencePoint {
    pub(crate) fn from_response(
        frequency_hz: f64,
        radiation_impedance: Complex64,
        velocity: Complex64,
        electrical_impedance: Complex64,
    ) -> Self {
        Self {
            frequency_hz,
            radiation_impedance_real_n_s_per_m: radiation_impedance.re,
            radiation_impedance_imag_n_s_per_m: radiation_impedance.im,
            velocity_real_m_per_s: velocity.re,
            velocity_imag_m_per_s: velocity.im,
            electrical_impedance_real_ohm: electrical_impedance.re,
            electrical_impedance_imag_ohm: electrical_impedance.im,
        }
    }

    fn radiation_impedance(self) -> Complex64 {
        Complex64::new(
            self.radiation_impedance_real_n_s_per_m,
            self.radiation_impedance_imag_n_s_per_m,
        )
    }

    fn velocity(self) -> Complex64 {
        Complex64::new(self.velocity_real_m_per_s, self.velocity_imag_m_per_s)
    }

    fn electrical_impedance(self) -> Complex64 {
        Complex64::new(
            self.electrical_impedance_real_ohm,
            self.electrical_impedance_imag_ohm,
        )
    }
}

/// Absolute and relative replay tolerances for a driver-model reference.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct DriverModelReferenceTolerance {
    pub absolute: f64,
    pub relative: f64,
}

impl Default for DriverModelReferenceTolerance {
    fn default() -> Self {
        Self {
            absolute: 1.0e-12,
            relative: 1.0e-10,
        }
    }
}

impl DriverModelReferenceTolerance {
    fn validate(self) -> Result<(), DriverModelError> {
        if !self.absolute.is_finite()
            || self.absolute < 0.0
            || !self.relative.is_finite()
            || self.relative < 0.0
        {
            return Err(DriverModelError::InvalidReferenceArtifact {
                reason: "reference tolerances must be finite and non-negative",
            });
        }
        Ok(())
    }
}

/// Portable replayable response artifact for CPU golden and projected paths.
///
/// A `Golden` artifact is admitted only for deterministic CPU/f64 output. A
/// projected or preview producer can retain the same wire contract with its
/// own provenance, but cannot silently be promoted to the CPU golden lane.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DriverModelReferenceArtifact {
    pub schema_version: u32,
    pub provenance: ArtifactProvenance,
    pub execution: DriverModelReferenceExecution,
    pub points: Vec<DriverModelReferencePoint>,
}

impl DriverModelReferenceArtifact {
    /// Build a CPU/f64 golden artifact by replaying a model against explicit
    /// per-frequency radiation loads.
    pub fn from_model(
        model: &DriverModel,
        frequency_hz: &[f64],
        radiation_impedance: &[Complex64],
    ) -> Result<Self, DriverModelError> {
        model.validate()?;
        if model.provenance != crate::conventions::ArtifactProvenance::native_golden() {
            return Err(DriverModelError::InvalidReferenceArtifact {
                reason: "CPU golden reference cannot be extracted from a preview or non-CPU model",
            });
        }
        if frequency_hz.len() != radiation_impedance.len() || frequency_hz.is_empty() {
            return Err(DriverModelError::InvalidReferenceArtifact {
                reason: "reference frequency and radiation grids must be non-empty and equal",
            });
        }

        let mut points = Vec::with_capacity(frequency_hz.len());
        for (frequency_hz, radiation_impedance) in frequency_hz
            .iter()
            .copied()
            .zip(radiation_impedance.iter().copied())
        {
            let (velocity, electrical_impedance) =
                model.response_with_radiation_impedance(frequency_hz, radiation_impedance)?;
            points.push(DriverModelReferencePoint::from_response(
                frequency_hz,
                radiation_impedance,
                velocity,
                electrical_impedance,
            ));
        }

        let mut artifact = Self {
            schema_version: DRIVER_MODEL_REFERENCE_ARTIFACT_SCHEMA_VERSION,
            provenance: ArtifactProvenance::native_golden(),
            execution: DriverModelReferenceExecution::cpu_golden(frequency_hz),
            points,
        };
        artifact.refresh_checksum();
        artifact.validate()?;
        Ok(artifact)
    }

    /// Change only the producer provenance, retaining the response payload.
    /// Validation still prevents a non-deterministic producer from claiming a
    /// CPU golden artifact.
    pub fn with_provenance(mut self, provenance: ArtifactProvenance) -> Self {
        self.provenance = provenance;
        self.execution.requested_backend = provenance.backend;
        self.execution.actual_backend = provenance.backend;
        self.execution.precision = provenance.precision;
        self.execution.device = match provenance.backend {
            crate::conventions::ArtifactBackend::Cpu => "cpu".to_string(),
            crate::conventions::ArtifactBackend::Cuda => "cuda:preview".to_string(),
            crate::conventions::ArtifactBackend::Metal => "metal:preview".to_string(),
            crate::conventions::ArtifactBackend::Wasm => "wasm:preview".to_string(),
        };
        self.refresh_checksum();
        self
    }

    /// Attach evidence collected from the live backend that produced the
    /// response table. Validation binds this identity to the wire artifact.
    pub fn with_execution(
        mut self,
        execution: DriverModelReferenceExecution,
    ) -> Result<Self, DriverModelError> {
        self.execution = execution;
        self.refresh_checksum();
        self.validate()?;
        Ok(self)
    }

    /// Require the full-order/projected evidence needed for ROM promotion.
    pub fn validate_rom_promotion(&self) -> Result<(), DriverModelError> {
        self.validate()?;
        self.execution.promotion.validate_gate()
    }

    /// Validate the serialized artifact without requiring a live model.
    pub fn validate(&self) -> Result<(), DriverModelError> {
        if self.schema_version != DRIVER_MODEL_REFERENCE_ARTIFACT_SCHEMA_VERSION {
            return Err(DriverModelError::InvalidReferenceArtifact {
                reason: "unsupported reference artifact schema",
            });
        }
        if self.provenance.schema_version != crate::conventions::ARTIFACT_PROVENANCE_SCHEMA_VERSION
        {
            return Err(DriverModelError::InvalidReferenceArtifact {
                reason: "unsupported artifact provenance schema",
            });
        }
        self.execution.validate(self.provenance)?;
        if matches!(
            self.provenance.execution_profile,
            crate::conventions::ArtifactExecutionProfile::Golden
        ) && (self.provenance.backend != crate::conventions::ArtifactBackend::Cpu
            || self.provenance.precision != crate::conventions::ArtifactPrecision::F64
            || !self.provenance.deterministic)
        {
            return Err(DriverModelError::InvalidReferenceArtifact {
                reason: "golden artifacts require deterministic CPU/f64 provenance",
            });
        }
        if self.points.is_empty() {
            return Err(DriverModelError::InvalidReferenceArtifact {
                reason: "reference artifact must contain at least one point",
            });
        }

        for (index, point) in self.points.iter().enumerate() {
            let values = [
                point.frequency_hz,
                point.radiation_impedance_real_n_s_per_m,
                point.radiation_impedance_imag_n_s_per_m,
                point.velocity_real_m_per_s,
                point.velocity_imag_m_per_s,
                point.electrical_impedance_real_ohm,
                point.electrical_impedance_imag_ohm,
            ];
            if values.iter().any(|value| !value.is_finite()) || point.frequency_hz <= 0.0 {
                return Err(DriverModelError::InvalidReferenceArtifact {
                    reason: "reference points must be finite at positive frequencies",
                });
            }
            if point.radiation_impedance_real_n_s_per_m < -1.0e-12 {
                return Err(DriverModelError::InvalidReferenceArtifact {
                    reason: "reference radiation resistance must be passive",
                });
            }
            if index > 0 && self.points[index - 1].frequency_hz >= point.frequency_hz {
                return Err(DriverModelError::InvalidReferenceArtifact {
                    reason: "reference frequencies must be strictly increasing",
                });
            }
        }
        let frequencies: Vec<f64> = self.points.iter().map(|point| point.frequency_hz).collect();
        if self.execution.frequency_grid_identity != frequency_grid_identity(&frequencies) {
            return Err(DriverModelError::InvalidReferenceArtifact {
                reason: "reference frequency-grid identity is stale",
            });
        }
        if self.checksum() != self.execution.reference_checksum_sha256 {
            return Err(DriverModelError::InvalidReferenceArtifact {
                reason: "reference checksum does not match the canonical payload",
            });
        }
        Ok(())
    }

    /// Recompute every point from a live model and reject any payload drift.
    pub fn validate_against_model(
        &self,
        model: &DriverModel,
        tolerance: DriverModelReferenceTolerance,
    ) -> Result<(), DriverModelError> {
        self.validate()?;
        tolerance.validate()?;
        model.validate()?;
        for point in &self.points {
            let (actual_velocity, actual_impedance) = model.response_with_radiation_impedance(
                point.frequency_hz,
                point.radiation_impedance(),
            )?;
            check_reference_value(
                point.frequency_hz,
                "velocity",
                actual_velocity,
                point.velocity(),
                tolerance,
            )?;
            check_reference_value(
                point.frequency_hz,
                "electrical impedance",
                actual_impedance,
                point.electrical_impedance(),
                tolerance,
            )?;
        }
        Ok(())
    }

    /// Compare a projected or preview artifact with a CPU golden artifact.
    /// This is the portable coverage seam used by backend-specific harnesses;
    /// it compares the boundary load as well as both response observables.
    pub fn validate_against_golden(
        &self,
        golden: &Self,
        tolerance: DriverModelReferenceTolerance,
    ) -> Result<(), DriverModelError> {
        golden.validate()?;
        if golden.provenance.execution_profile
            != crate::conventions::ArtifactExecutionProfile::Golden
        {
            return Err(DriverModelError::InvalidReferenceArtifact {
                reason: "reference comparison requires a golden baseline",
            });
        }
        self.validate()?;
        tolerance.validate()?;
        if self.execution.mesh_identity != golden.execution.mesh_identity {
            return Err(DriverModelError::InvalidReferenceArtifact {
                reason: "reference comparison mesh identities differ",
            });
        }
        if self.execution.frequency_grid_identity != golden.execution.frequency_grid_identity {
            return Err(DriverModelError::InvalidReferenceArtifact {
                reason: "reference comparison frequency-grid identities differ",
            });
        }
        if self.execution.precision != crate::conventions::ArtifactPrecision::F64
            && golden.execution.precision != crate::conventions::ArtifactPrecision::F64
        {
            return Err(DriverModelError::InvalidReferenceArtifact {
                reason: "reference comparison requires at least one f64 artifact",
            });
        }
        if self.points.len() != golden.points.len() {
            return Err(DriverModelError::InvalidReferenceArtifact {
                reason: "reference comparison grids must have equal length",
            });
        }
        for (candidate, expected) in self.points.iter().zip(&golden.points) {
            if candidate.frequency_hz != expected.frequency_hz {
                return Err(DriverModelError::InvalidReferenceArtifact {
                    reason: "reference comparison frequency grids differ",
                });
            }
            check_reference_value(
                expected.frequency_hz,
                "radiation impedance",
                candidate.radiation_impedance(),
                expected.radiation_impedance(),
                tolerance,
            )?;
            check_reference_value(
                expected.frequency_hz,
                "velocity",
                candidate.velocity(),
                expected.velocity(),
                tolerance,
            )?;
            check_reference_value(
                expected.frequency_hz,
                "electrical impedance",
                candidate.electrical_impedance(),
                expected.electrical_impedance(),
                tolerance,
            )?;
        }
        Ok(())
    }

    /// Serialize with stable struct/vector ordering for cross-platform
    /// artifact checks. Callers may hash these bytes with their manifest.
    pub fn canonical_json(&self) -> Result<Vec<u8>, serde_json::Error> {
        self.canonical_bytes(true)
    }

    fn canonical_bytes(&self, include_checksum: bool) -> Result<Vec<u8>, serde_json::Error> {
        let mut canonical = self.clone();
        // The toolchain identifies the producing compiler, not the physics
        // payload. Normalize it so canonical bytes — and every hash over
        // them — are stable across compiler updates, per the portable golden
        // envelope contract. Live objects keep the real toolchain string for
        // evidence; only the canonical encoding is normalized.
        canonical.execution.toolchain = CANONICAL_TOOLCHAIN_MARKER.to_string();
        for point in &mut canonical.points {
            point.frequency_hz = canonicalize_reference_float(point.frequency_hz);
            point.radiation_impedance_real_n_s_per_m =
                canonicalize_reference_float(point.radiation_impedance_real_n_s_per_m);
            point.radiation_impedance_imag_n_s_per_m =
                canonicalize_reference_float(point.radiation_impedance_imag_n_s_per_m);
            point.velocity_real_m_per_s = canonicalize_reference_float(point.velocity_real_m_per_s);
            point.velocity_imag_m_per_s = canonicalize_reference_float(point.velocity_imag_m_per_s);
            point.electrical_impedance_real_ohm =
                canonicalize_reference_float(point.electrical_impedance_real_ohm);
            point.electrical_impedance_imag_ohm =
                canonicalize_reference_float(point.electrical_impedance_imag_ohm);
        }
        if !include_checksum {
            canonical.execution.reference_checksum_sha256.clear();
        }
        serde_json::to_vec(&canonical)
    }

    fn checksum(&self) -> String {
        let bytes = self
            .canonical_bytes(false)
            .expect("reference artifact contains only serializable fields");
        let digest = Sha256::digest(bytes);
        format!("{digest:x}")
    }

    fn refresh_checksum(&mut self) {
        self.execution.reference_checksum_sha256.clear();
        self.execution.reference_checksum_sha256 = self.checksum();
    }
}

fn frequency_grid_identity(frequencies: &[f64]) -> String {
    let mut hasher = Sha256::new();
    for frequency in frequencies {
        hasher.update(frequency.to_le_bytes());
    }
    format!("{:x}", hasher.finalize())
}

/// Marker recorded in place of the producing compiler inside canonical
/// reference bytes. The live artifact retains the real toolchain string;
/// only the hashed encoding is normalized so the portable golden envelope
/// survives compiler updates.
const CANONICAL_TOOLCHAIN_MARKER: &str = "canonical";

/// Keep serialized reference bytes stable across harmless last-bit CPU
/// differences while retaining twelve significant decimal digits. This is
/// tighter than the default replay tolerance and does not alter live solves.
fn canonicalize_reference_float(value: f64) -> f64 {
    if value == 0.0 || !value.is_finite() {
        return value;
    }
    let exponent = value.abs().log10().floor() as i32;
    let scale = 10_f64.powi(12 - exponent - 1);
    (value * scale).round() / scale
}

fn check_reference_value(
    frequency_hz: f64,
    observable: &'static str,
    actual: Complex64,
    expected: Complex64,
    tolerance: DriverModelReferenceTolerance,
) -> Result<(), DriverModelError> {
    let absolute_error = (actual - expected).norm();
    let relative_error = absolute_error / expected.norm().max(tolerance.absolute);
    if absolute_error > tolerance.absolute && relative_error > tolerance.relative {
        return Err(DriverModelError::ReferenceArtifactMismatch {
            frequency_hz,
            observable,
            absolute_error,
            relative_error,
        });
    }
    Ok(())
}

/// Thermal state reported by a product-side coupled radiation solve.
///
/// The state is deliberately a small typed ledger rather than a replacement
/// for the calibrated driver model. It records the operating point that was
/// actually used to obtain the loaded velocity and electrical impedance.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ThermalDriverState {
    pub coil_temperature_k: f64,
    pub hot_resistance_ohm: f64,
    pub coil_loss_w: f64,
    pub eddy_loss_w: f64,
    pub input_power_w: f64,
    pub mechanical_loss_w: f64,
    pub radiation_power_w: f64,
    pub power_balance_residual_w: f64,
    /// Positive dB loss relative to the cold loaded velocity at this
    /// frequency.
    pub compression_db: f64,
    pub thermal_iteration_count: usize,
    pub thermal_residual_ohm: f64,
    /// True when the product thermal iterate fed the solved coil temperature
    /// into the driver's nonlinear motor law before accepting this state.
    pub nonlinear_temperature_feedback_applied: bool,
    /// Full conjugate thermal state when a driver thermal network was active.
    pub thermal_network_temperatures_k: Option<[f64; crate::THERMAL_NODE_COUNT]>,
    pub thermal_network_power_w: Option<crate::ThermalNetworkPower>,
    pub thermal_network_residual_w: Option<f64>,
}

/// Complete driver-unit result consumed by cabinet/BEM product simulation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DriverModel {
    /// Backend/profile/precision of the model-producing solve.  This is kept
    /// on the typed handoff so a reduced preview cannot be consumed as a
    /// native golden driver by accident.
    #[serde(default)]
    pub provenance: ArtifactProvenance,
    pub thiele_small: ThieleSmallParameters,
    pub bl_curve: Vec<BlPoint>,
    pub inductance_curve: Vec<InductancePoint>,
    pub reference_voltage_rms_v: f64,
    /// Optional ideal active-crossover voltage transfer; empty means unity.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub drive_transfer: Vec<DriveTransferPoint>,
    pub velocity_transfer: Vec<VelocityTransferPoint>,
    /// Electrical input impedance for the modeling reference load.
    pub electrical_impedance: Vec<ImpedancePoint>,
    /// Mechanical impedance excluding the exterior front-radiation load.
    ///
    /// Product BEM simulation adds its resolved self/mutual radiation load.
    pub intrinsic_mechanical_impedance: Vec<ImpedancePoint>,
    /// Optional series capacitance (F) of an electrical high-pass network in
    /// front of the voice coil. Adds `1/(jωC)` to the coil impedance, so both
    /// the velocity transfer and the coupled radiation-matrix solve see the
    /// same complex crossover. A series low-pass inductor is folded directly
    /// into `inductance_curve` and needs no extra field.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub series_capacitance_f: Option<f64>,
    /// Optional steady-state voice-coil thermal network used by the product
    /// simulator's loaded multi-driver solve. `None` preserves the cold,
    /// small-signal handoff for legacy models.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub thermal_compression_spec: Option<crate::ThermalCompressionSpec>,
    /// Optional four-node driver/magnet/gap/cabinet thermal network. When
    /// present, this policy supersedes the scalar thermal-compression model
    /// for the production coupled solve.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub thermal_network: Option<DriverThermalNetworkConfig>,
    /// Optional nonlinear motor law used only by explicit operating-point
    /// response calls. The cold small-signal table remains the reference.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub nonlinear_motor: Option<MotorConstitutiveModel>,
    /// Optional retained operating-domain evidence produced by validating the
    /// attached nonlinear motor law. Deserialization rechecks it against the
    /// law before the driver handoff can be promoted.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub nonlinear_operating_domain_evidence: Option<MotorOperatingDomainEvidence>,
}

impl DriverModel {
    /// Build a handoff using the study's configured lumped motor values.
    ///
    /// This is the analytical/reference constructor. Native full-order
    /// results must use [`Self::from_study_with_evidence`] so an unconverged
    /// multiphysics state cannot be mistaken for a golden driver model.
    pub fn from_study(study: &SpeakerDriverStudy) -> Result<Self, DriverModelError> {
        Self::from_study_and_motor(study, None)
    }

    /// Reject an unproven FEM-derived handoff.
    ///
    /// Use [`Self::from_study_with_motor_and_evidence`] after collecting the
    /// nonlinear, differential-permeability, eddy-current, structural,
    /// acoustic, and coupling gates.
    #[deprecated(note = "use from_study_with_motor_and_evidence")]
    pub fn from_study_with_motor(
        _study: &SpeakerDriverStudy,
        _motor: &MotorParameters,
    ) -> Result<Self, DriverModelError> {
        Err(DriverModelError::ExtractionEvidenceRequired)
    }

    /// Promote a native full-order study into the product handoff only after
    /// every required cross-domain residual gate has passed.
    pub fn from_study_with_evidence(
        study: &SpeakerDriverStudy,
        evidence: DriverModelExtractionEvidence,
    ) -> Result<Self, DriverModelError> {
        evidence.validate()?;
        Self::from_study_and_motor(study, None)
    }

    /// Motor-curve variant of [`Self::from_study_with_evidence`].
    pub fn from_study_with_motor_and_evidence(
        study: &SpeakerDriverStudy,
        motor: &MotorParameters,
        evidence: DriverModelExtractionEvidence,
    ) -> Result<Self, DriverModelError> {
        evidence.validate()?;
        Self::from_study_and_motor(study, Some(motor))
    }

    fn from_study_and_motor(
        study: &SpeakerDriverStudy,
        motor: Option<&MotorParameters>,
    ) -> Result<Self, DriverModelError> {
        study.validate()?;
        if motor.is_some_and(|motor| {
            motor.le_vs_frequency.len() != motor.re_eddy_vs_frequency.len()
                || motor
                    .le_vs_frequency
                    .iter()
                    .zip(&motor.re_eddy_vs_frequency)
                    .any(|(le, resistance)| le.0 != resistance.0)
        }) {
            return Err(DriverModelError::MismatchedMotorCurves);
        }
        let responses = match motor {
            Some(motor) => small_signal_response_sweep_with_motor(study, motor)?,
            None => study
                .try_small_signal_response_sweep(study.drive_voltage_rms_v)?
                .into_iter()
                .map(|response| MotorResponseSample {
                    frequency_hz: response.frequency_hz,
                    velocity: response.coil_velocity,
                    impedance: response.impedance,
                })
                .collect(),
        };
        let moving = study.moving_system;
        let coil = study.voice_coil;
        let omega_s = std::f64::consts::TAU * moving.try_resonance_hz()?;
        let compliance = moving.stiffness_n_per_m.recip();
        let piston_area = std::f64::consts::PI * study.geometry.diaphragm_radius_m.powi(2);
        let mechanical_q = (moving.mechanical_resistance_kg_per_s > 0.0)
            .then(|| omega_s * moving.moving_mass_kg / moving.mechanical_resistance_kg_per_s);
        let electrical_q = (coil.bl_tm.abs() > f64::EPSILON)
            .then(|| omega_s * moving.moving_mass_kg * coil.resistance_ohm / coil.bl_tm.powi(2));
        let total_q = mechanical_q.zip(electrical_q).map(|(qms, qes)| {
            let inverse = qms.recip() + qes.recip();
            inverse.recip()
        });
        let thiele_small = ThieleSmallParameters {
            voice_coil_resistance_ohm: coil.resistance_ohm,
            nominal_inductance_h: motor.map_or(coil.inductance_h, |m| m.le_nominal),
            force_factor_tm: motor.map_or(coil.bl_tm, |m| m.bl_nominal),
            moving_mass_kg: moving.moving_mass_kg,
            compliance_m_per_n: compliance,
            mechanical_resistance_kg_per_s: moving.mechanical_resistance_kg_per_s,
            effective_piston_area_m2: piston_area,
            resonance_hz: moving.try_resonance_hz()?,
            mechanical_q,
            electrical_q,
            total_q,
            equivalent_compliance_volume_m3: study.acoustic_medium.density_kg_per_m3
                * study.acoustic_medium.sound_speed_m_per_s.powi(2)
                * piston_area.powi(2)
                * compliance,
        };

        let bl_curve = motor.map_or_else(
            || {
                vec![BlPoint {
                    displacement_m: 0.0,
                    force_factor_tm: coil.bl_tm,
                }]
            },
            |motor| {
                motor
                    .bl_curve
                    .iter()
                    .map(|&(displacement_m, force_factor_tm)| BlPoint {
                        displacement_m,
                        force_factor_tm,
                    })
                    .collect()
            },
        );
        let inductance_curve = motor.map_or_else(
            || {
                study
                    .frequency_hz
                    .iter()
                    .map(|&frequency_hz| InductancePoint {
                        frequency_hz,
                        inductance_h: coil.inductance_h,
                        eddy_resistance_ohm: 0.0,
                    })
                    .collect()
            },
            |motor| {
                motor
                    .le_vs_frequency
                    .iter()
                    .zip(&motor.re_eddy_vs_frequency)
                    .map(
                        |(&(frequency_hz, inductance_h), &(resistance_frequency, resistance))| {
                            debug_assert_eq!(frequency_hz, resistance_frequency);
                            InductancePoint {
                                frequency_hz,
                                inductance_h,
                                eddy_resistance_ohm: resistance,
                            }
                        },
                    )
                    .collect()
            },
        );
        let velocity_transfer = responses
            .iter()
            .map(|response| VelocityTransferPoint {
                frequency_hz: response.frequency_hz,
                velocity_real_m_per_s: response.velocity.re,
                velocity_imag_m_per_s: response.velocity.im,
            })
            .collect();
        let electrical_impedance = responses
            .iter()
            .map(|response| ImpedancePoint {
                frequency_hz: response.frequency_hz,
                real: response.impedance.re,
                imaginary: response.impedance.im,
            })
            .collect();
        let intrinsic_mechanical_impedance = study
            .frequency_hz
            .iter()
            .map(|&frequency_hz| {
                let impedance = study.total_mechanical_impedance(frequency_hz)?
                    - study.mechanical_radiation_impedance(frequency_hz)?;
                Ok(ImpedancePoint {
                    frequency_hz,
                    real: impedance.re,
                    imaginary: impedance.im,
                })
            })
            .collect::<Result<Vec<_>, ModelingError>>()?;
        let nonlinear_motor = motor.map(|motor| {
            let bl_curve = motor
                .bl_curve
                .iter()
                .map(|&(displacement_m, force_factor_tm)| NonlinearBlPoint {
                    displacement_m,
                    force_factor_tm,
                })
                .collect();
            let le_surface = if motor.le_surface.is_empty() {
                motor
                    .le_vs_frequency
                    .iter()
                    .zip(&motor.re_eddy_vs_frequency)
                    .map(
                        |(&(frequency_hz, inductance_h), &(_, eddy_resistance_ohm))| {
                            NonlinearLePoint {
                                displacement_m: 0.0,
                                current_a: 0.0,
                                frequency_hz,
                                inductance_h,
                                eddy_resistance_ohm,
                            }
                        },
                    )
                    .collect()
            } else {
                motor.le_surface.clone()
            };
            MotorConstitutiveModel {
                bl_curve,
                le_surface,
                temperature: MotorTemperatureLaw {
                    copper_resistance_ohm: coil.resistance_ohm,
                    bl_per_k: motor.bl_temperature_coefficient_per_k,
                    ..MotorTemperatureLaw::default()
                },
                hysteresis: motor
                    .hysteresis_loss
                    .unwrap_or_else(HysteresisLossModel::default),
                magnetic: motor.magnetic_law.clone(),
                eddy_current: motor.eddy_current_loss.clone(),
            }
        });
        let model = Self {
            provenance: ArtifactProvenance::native_golden(),
            thiele_small,
            bl_curve,
            inductance_curve,
            reference_voltage_rms_v: study.drive_voltage_rms_v,
            velocity_transfer,
            electrical_impedance,
            intrinsic_mechanical_impedance,
            drive_transfer: Vec::new(),
            series_capacitance_f: None,
            thermal_compression_spec: None,
            thermal_network: None,
            nonlinear_motor,
            nonlinear_operating_domain_evidence: None,
        };
        model.validate()?;
        Ok(model)
    }

    /// Override provenance when a serialized handoff is deliberately created
    /// by a reduced/preview path. Native constructors remain golden by
    /// default.
    pub fn with_provenance(mut self, provenance: ArtifactProvenance) -> Self {
        self.provenance = provenance;
        self
    }

    pub fn validate(&self) -> Result<(), DriverModelError> {
        for p in &self.drive_transfer {
            validate_finite_positive("drive frequency", p.frequency_hz)?;
            if !p.real.is_finite() || !p.imag.is_finite() {
                return Err(DriverModelError::InvalidScalar {
                    name: "drive transfer",
                });
            }
        }
        if self
            .drive_transfer
            .windows(2)
            .any(|p| p[0].frequency_hz >= p[1].frequency_hz)
        {
            return Err(DriverModelError::InvalidScalar { name: "drive grid" });
        }
        validate_finite_positive(
            "voice-coil resistance",
            self.thiele_small.voice_coil_resistance_ohm,
        )?;
        validate_finite_nonnegative("nominal inductance", self.thiele_small.nominal_inductance_h)?;
        validate_finite_positive("moving mass", self.thiele_small.moving_mass_kg)?;
        validate_finite_positive("compliance", self.thiele_small.compliance_m_per_n)?;
        validate_finite_positive(
            "effective piston area",
            self.thiele_small.effective_piston_area_m2,
        )?;
        validate_finite_nonnegative("reference voltage", self.reference_voltage_rms_v)?;
        if let Some(capacitance) = self.series_capacitance_f {
            validate_finite_positive("series capacitance", capacitance)?;
        }
        if let Some(spec) = self.thermal_compression_spec {
            spec.validate()?;
        }
        if let Some(network) = self.thermal_network {
            network.validate()?;
        }
        if let Some(motor) = &self.nonlinear_motor {
            motor.validate()?;
        }
        if let Some(evidence) = &self.nonlinear_operating_domain_evidence {
            let motor = self
                .nonlinear_motor
                .as_ref()
                .ok_or(DriverModelError::NonlinearMotorRequired)?;
            motor.validate_operating_domain_evidence(evidence)?;
        }
        validate_bl_curve(&self.bl_curve)?;
        validate_inductance_curve(&self.inductance_curve)?;
        validate_velocity_curve(&self.velocity_transfer)?;
        validate_impedance_curve("electrical impedance", &self.electrical_impedance, true)?;
        validate_impedance_curve(
            "intrinsic mechanical impedance",
            &self.intrinsic_mechanical_impedance,
            true,
        )?;
        Ok(())
    }

    /// Attach a validated nonlinear motor law without changing the cached
    /// cold small-signal response table.
    pub fn with_nonlinear_motor(
        mut self,
        motor: MotorConstitutiveModel,
    ) -> Result<Self, DriverModelError> {
        motor.validate()?;
        self.nonlinear_motor = Some(motor);
        self.validate()?;
        Ok(self)
    }

    /// Attach a nonlinear motor law together with validated operating-domain
    /// evidence. This is the promotion path for current/position/frequency/
    /// temperature-dependent motor data.
    pub fn with_nonlinear_operating_domain(
        mut self,
        motor: MotorConstitutiveModel,
        domain: MotorOperatingDomain,
    ) -> Result<Self, DriverModelError> {
        let evidence = motor.validate_operating_domain(&domain)?;
        self.nonlinear_motor = Some(motor);
        self.nonlinear_operating_domain_evidence = Some(evidence);
        self.validate()?;
        Ok(self)
    }

    /// Evaluate the nonlinear motor at one displacement/current/frequency /
    /// temperature point and return the corresponding coupled network.
    pub fn network_parameters_at_operating_point(
        &self,
        point: MotorOperatingPoint,
    ) -> Result<(DriverNetworkParameters, crate::MotorOperatingValues), DriverModelError> {
        let motor = self
            .nonlinear_motor
            .as_ref()
            .ok_or(DriverModelError::NonlinearMotorRequired)?;
        let values = motor.evaluate(point)?;
        let network = self.network_parameters_at_with_motor_values(
            point.frequency_hz,
            values.copper_resistance_ohm,
            values.force_factor_tm,
            values.inductance_h,
            values.eddy_resistance_ohm,
        )?;
        Ok((network, values))
    }

    /// Evaluate the nonlinear motor but retain an externally solved copper
    /// resistance (for example, the product thermal network's coil node).
    /// This keeps thermal-network resistance closure authoritative while the
    /// motor law supplies temperature-dependent BL, Le, and eddy loss.
    pub fn network_parameters_at_operating_point_with_hot_resistance(
        &self,
        point: MotorOperatingPoint,
        hot_resistance_ohm: f64,
    ) -> Result<(DriverNetworkParameters, crate::MotorOperatingValues), DriverModelError> {
        let motor = self
            .nonlinear_motor
            .as_ref()
            .ok_or(DriverModelError::NonlinearMotorRequired)?;
        let values = motor.evaluate(point)?;
        let network = self.network_parameters_at_with_motor_values(
            point.frequency_hz,
            hot_resistance_ohm,
            values.force_factor_tm,
            values.inductance_h,
            values.eddy_resistance_ohm,
        )?;
        Ok((network, values))
    }

    /// Evaluate the motor with the conjugate network's magnet temperature and
    /// the coil node's authoritative copper resistance. This is the M3
    /// thermal handoff: magnet temperature changes BL while coil temperature
    /// changes Re, without conflating the two nodes.
    pub fn network_parameters_at_thermal_state(
        &self,
        mut point: MotorOperatingPoint,
        thermal_state: crate::ThermalNetworkState,
    ) -> Result<(DriverNetworkParameters, crate::MotorOperatingValues), DriverModelError> {
        let thermal = self
            .thermal_network
            .ok_or(DriverModelError::ThermalNetworkRequired)?;
        point.temperature_k = thermal_state.temperatures_k[crate::ThermalNode::Magnet as usize];
        let coil_temperature_k =
            thermal_state.temperatures_k[crate::ThermalNode::VoiceCoil as usize];
        let hot_resistance_ohm = self.thiele_small.voice_coil_resistance_ohm
            * (1.0
                + thermal.copper_resistance_per_k
                    * (coil_temperature_k - thermal.specification.ambient_temperature_k));
        self.network_parameters_at_operating_point_with_hot_resistance(point, hot_resistance_ohm)
    }

    fn network_parameters_at_with_motor_values(
        &self,
        frequency_hz: f64,
        resistance_ohm: f64,
        force_factor_tm: f64,
        inductance_h: f64,
        eddy_resistance_ohm: f64,
    ) -> Result<DriverNetworkParameters, DriverModelError> {
        if !frequency_hz.is_finite()
            || frequency_hz <= 0.0
            || !resistance_ohm.is_finite()
            || resistance_ohm <= 0.0
            || !force_factor_tm.is_finite()
            || !inductance_h.is_finite()
            || inductance_h < 0.0
            || !eddy_resistance_ohm.is_finite()
            || eddy_resistance_ohm < 0.0
        {
            return Err(DriverModelError::InvalidScalar {
                name: "nonlinear motor network",
            });
        }
        let omega = std::f64::consts::TAU * frequency_hz;
        let capacitive_reactance = self
            .series_capacitance_f
            .map_or(0.0, |capacitance| -1.0 / (omega * capacitance));
        Ok(DriverNetworkParameters {
            coil_impedance_ohm: Complex64::new(
                resistance_ohm + eddy_resistance_ohm,
                omega * inductance_h + capacitive_reactance,
            ),
            force_factor_tm,
            intrinsic_mechanical_impedance: interpolate_complex_curve(
                &self.intrinsic_mechanical_impedance,
                frequency_hz,
                ImpedancePoint::impedance,
            ),
        })
    }

    /// Interpolate active-crossover voltage gain; absent tables return unity.
    pub fn drive_transfer_at(&self, frequency_hz: f64) -> Complex64 {
        if self.drive_transfer.is_empty() {
            return Complex64::new(1.0, 0.0);
        }
        interpolate_complex_curve(&self.drive_transfer, frequency_hz, |p| {
            Complex64::new(p.real, p.imag)
        })
    }

    /// Interpolate the complex diaphragm velocity, clamping outside the grid.
    pub fn velocity_at(&self, frequency_hz: f64) -> Complex64 {
        interpolate_complex_curve(
            &self.velocity_transfer,
            frequency_hz,
            VelocityTransferPoint::velocity,
        ) * self.drive_transfer_at(frequency_hz)
    }

    pub fn electrical_impedance_at(&self, frequency_hz: f64) -> Complex64 {
        interpolate_complex_curve(
            &self.electrical_impedance,
            frequency_hz,
            ImpedancePoint::impedance,
        )
    }

    /// Electrical and intrinsic-mechanical network coefficients at a
    /// frequency, for coupled multi-driver radiation-matrix solves.
    pub fn network_parameters_at(
        &self,
        frequency_hz: f64,
    ) -> Result<DriverNetworkParameters, DriverModelError> {
        self.network_parameters_at_with_hot_resistance(
            frequency_hz,
            self.thiele_small.voice_coil_resistance_ohm,
        )
    }

    /// Return network coefficients after replacing the copper resistance with
    /// a validated hot operating-point value.
    ///
    /// The frequency-dependent eddy-current resistance remains taken from
    /// `inductance_curve`; callers should pass the `hot_resistance_ohm` from
    /// [`ThermalCompressionResult`](crate::ThermalCompressionResult) rather
    /// than adding eddy loss a second time. The model itself is not mutated,
    /// so a cold and hot product solve can share the same handoff safely.
    pub fn network_parameters_at_with_hot_resistance(
        &self,
        frequency_hz: f64,
        hot_resistance_ohm: f64,
    ) -> Result<DriverNetworkParameters, DriverModelError> {
        if !frequency_hz.is_finite() || frequency_hz <= 0.0 {
            return Err(DriverModelError::InvalidScalar { name: "frequency" });
        }
        if !hot_resistance_ohm.is_finite()
            || hot_resistance_ohm < self.thiele_small.voice_coil_resistance_ohm
        {
            return Err(DriverModelError::InvalidScalar {
                name: "hot resistance",
            });
        }
        let omega = std::f64::consts::TAU * frequency_hz;
        let inductance = interpolate_inductance(&self.inductance_curve, frequency_hz);
        let capacitive_reactance = self
            .series_capacitance_f
            .map_or(0.0, |capacitance| -1.0 / (omega * capacitance));
        Ok(DriverNetworkParameters {
            coil_impedance_ohm: Complex64::new(
                hot_resistance_ohm + inductance.1,
                omega * inductance.0 + capacitive_reactance,
            ),
            force_factor_tm: interpolate_bl(&self.bl_curve, 0.0),
            intrinsic_mechanical_impedance: interpolate_complex_curve(
                &self.intrinsic_mechanical_impedance,
                frequency_hz,
                ImpedancePoint::impedance,
            ),
        })
    }

    /// Recompute the velocity and electrical input impedance with a product
    /// simulation's exterior mechanical radiation load.
    pub fn response_with_radiation_impedance(
        &self,
        frequency_hz: f64,
        radiation_impedance: Complex64,
    ) -> Result<(Complex64, Complex64), DriverModelError> {
        if !frequency_hz.is_finite()
            || frequency_hz <= 0.0
            || !radiation_impedance.re.is_finite()
            || !radiation_impedance.im.is_finite()
        {
            return Err(DriverModelError::InvalidRadiationImpedance);
        }
        let network = self.network_parameters_at(frequency_hz)?;
        let mechanical = network.intrinsic_mechanical_impedance + radiation_impedance;
        if mechanical.norm() <= f64::MIN_POSITIVE {
            return Err(DriverModelError::InvalidRadiationImpedance);
        }
        let bl = network.force_factor_tm;
        let coil_impedance = network.coil_impedance_ohm;
        let electrical = coil_impedance + Complex64::new(bl * bl, 0.0) / mechanical;
        let voltage = Complex64::new(self.reference_voltage_rms_v, 0.0)
            * self.drive_transfer_at(frequency_hz);
        let velocity = Complex64::new(bl, 0.0) * voltage
            / (coil_impedance * mechanical + Complex64::new(bl * bl, 0.0));
        Ok((velocity, electrical))
    }
}

#[derive(Debug, Clone, Copy)]
struct MotorResponseSample {
    frequency_hz: f64,
    velocity: Complex64,
    impedance: Complex64,
}

/// Recompute the small-signal response using the extracted motor curves.
///
/// The legacy analytical constructor uses `SpeakerDriverStudy`'s configured
/// lumped coil. A promoted FEM handoff must instead carry the nonlinear bias
/// BL and the eddy-current Le/Re sweep all the way into the response table;
/// otherwise those fields would be decorative metadata rather than model
/// inputs.
fn small_signal_response_sweep_with_motor(
    study: &SpeakerDriverStudy,
    motor: &MotorParameters,
) -> Result<Vec<MotorResponseSample>, DriverModelError> {
    if motor.bl_curve.is_empty()
        || motor.le_vs_frequency.is_empty()
        || motor.re_eddy_vs_frequency.is_empty()
        || motor.bl_curve.windows(2).any(|pair| pair[0].0 >= pair[1].0)
        || motor
            .le_vs_frequency
            .windows(2)
            .any(|pair| pair[0].0 >= pair[1].0)
        || motor
            .re_eddy_vs_frequency
            .windows(2)
            .any(|pair| pair[0].0 >= pair[1].0)
        || motor
            .bl_curve
            .iter()
            .chain(&motor.le_vs_frequency)
            .chain(&motor.re_eddy_vs_frequency)
            .any(|&(x, y)| !x.is_finite() || !y.is_finite())
    {
        return Err(DriverModelError::InvalidMotorCurve);
    }

    let voltage = Complex64::new(study.drive_voltage_rms_v, 0.0);
    study
        .frequency_hz
        .iter()
        .map(|&frequency_hz| {
            let bl = interpolate_scalar_curve(&motor.bl_curve, 0.0);
            let inductance = interpolate_scalar_curve(&motor.le_vs_frequency, frequency_hz);
            let eddy_resistance =
                interpolate_scalar_curve(&motor.re_eddy_vs_frequency, frequency_hz);
            let omega = std::f64::consts::TAU * frequency_hz;
            let mechanical = study.total_mechanical_impedance(frequency_hz)?;
            if mechanical.norm() <= f64::MIN_POSITIVE {
                return Err(DriverModelError::InvalidMotorResponse { frequency_hz });
            }
            let coil = Complex64::new(
                study.voice_coil.resistance_ohm + eddy_resistance,
                omega * inductance,
            );
            let denominator = coil * mechanical + Complex64::new(bl * bl, 0.0);
            if denominator.norm() <= f64::MIN_POSITIVE {
                return Err(DriverModelError::InvalidMotorResponse { frequency_hz });
            }
            let impedance = coil + Complex64::new(bl * bl, 0.0) / mechanical;
            let velocity = Complex64::new(bl, 0.0) * voltage / denominator;
            Ok(MotorResponseSample {
                frequency_hz,
                velocity,
                impedance,
            })
        })
        .collect()
}

fn interpolate_scalar_curve(points: &[(f64, f64)], x: f64) -> f64 {
    if x <= points[0].0 {
        return points[0].1;
    }
    if x >= points[points.len() - 1].0 {
        return points[points.len() - 1].1;
    }
    let upper = points.partition_point(|point| point.0 < x);
    let lower = upper - 1;
    let fraction = (x - points[lower].0) / (points[upper].0 - points[lower].0);
    points[lower].1 + fraction * (points[upper].1 - points[lower].1)
}

fn validate_bl_curve(points: &[BlPoint]) -> Result<(), DriverModelError> {
    if points.is_empty()
        || points
            .iter()
            .any(|point| !point.displacement_m.is_finite() || !point.force_factor_tm.is_finite())
        || points
            .windows(2)
            .any(|pair| pair[0].displacement_m >= pair[1].displacement_m)
    {
        return Err(DriverModelError::InvalidBlCurve);
    }
    Ok(())
}

fn validate_inductance_curve(points: &[InductancePoint]) -> Result<(), DriverModelError> {
    if points.is_empty()
        || points.iter().any(|point| {
            !point.frequency_hz.is_finite()
                || point.frequency_hz <= 0.0
                || !point.inductance_h.is_finite()
                || point.inductance_h < 0.0
                || !point.eddy_resistance_ohm.is_finite()
                || point.eddy_resistance_ohm < 0.0
        })
        || points
            .windows(2)
            .any(|pair| pair[0].frequency_hz >= pair[1].frequency_hz)
    {
        return Err(DriverModelError::InvalidInductanceCurve);
    }
    Ok(())
}

fn validate_velocity_curve(points: &[VelocityTransferPoint]) -> Result<(), DriverModelError> {
    if points.is_empty()
        || points.iter().any(|point| {
            !point.frequency_hz.is_finite()
                || point.frequency_hz <= 0.0
                || !point.velocity_real_m_per_s.is_finite()
                || !point.velocity_imag_m_per_s.is_finite()
        })
        || points
            .windows(2)
            .any(|pair| pair[0].frequency_hz >= pair[1].frequency_hz)
    {
        return Err(DriverModelError::InvalidVelocityTransfer);
    }
    Ok(())
}

fn validate_impedance_curve(
    _name: &'static str,
    points: &[ImpedancePoint],
    require_passive: bool,
) -> Result<(), DriverModelError> {
    if points.is_empty()
        || points.iter().any(|point| {
            !point.frequency_hz.is_finite()
                || point.frequency_hz <= 0.0
                || !point.real.is_finite()
                || !point.imaginary.is_finite()
                || (require_passive && point.real < -1.0e-10)
        })
        || points
            .windows(2)
            .any(|pair| pair[0].frequency_hz >= pair[1].frequency_hz)
    {
        return Err(DriverModelError::InvalidImpedanceCurve);
    }
    Ok(())
}

fn interpolate_bl(points: &[BlPoint], displacement_m: f64) -> f64 {
    if displacement_m <= points[0].displacement_m {
        return points[0].force_factor_tm;
    }
    if displacement_m >= points[points.len() - 1].displacement_m {
        return points[points.len() - 1].force_factor_tm;
    }
    let upper = points.partition_point(|point| point.displacement_m < displacement_m);
    let lower = upper - 1;
    let fraction = (displacement_m - points[lower].displacement_m)
        / (points[upper].displacement_m - points[lower].displacement_m);
    points[lower].force_factor_tm
        + fraction * (points[upper].force_factor_tm - points[lower].force_factor_tm)
}

fn interpolate_inductance(points: &[InductancePoint], frequency_hz: f64) -> (f64, f64) {
    if frequency_hz <= points[0].frequency_hz {
        return (points[0].inductance_h, points[0].eddy_resistance_ohm);
    }
    if frequency_hz >= points[points.len() - 1].frequency_hz {
        let point = points[points.len() - 1];
        return (point.inductance_h, point.eddy_resistance_ohm);
    }
    let upper = points.partition_point(|point| point.frequency_hz < frequency_hz);
    let lower = upper - 1;
    let fraction = (frequency_hz - points[lower].frequency_hz)
        / (points[upper].frequency_hz - points[lower].frequency_hz);
    (
        points[lower].inductance_h
            + fraction * (points[upper].inductance_h - points[lower].inductance_h),
        points[lower].eddy_resistance_ohm
            + fraction * (points[upper].eddy_resistance_ohm - points[lower].eddy_resistance_ohm),
    )
}

fn validate_finite_positive(name: &'static str, value: f64) -> Result<(), DriverModelError> {
    if value.is_finite() && value > 0.0 {
        Ok(())
    } else {
        Err(DriverModelError::InvalidScalar { name })
    }
}

fn validate_finite_nonnegative(name: &'static str, value: f64) -> Result<(), DriverModelError> {
    if value.is_finite() && value >= 0.0 {
        Ok(())
    } else {
        Err(DriverModelError::InvalidScalar { name })
    }
}

fn interpolate_complex_curve<T>(
    points: &[T],
    frequency_hz: f64,
    value: impl Fn(T) -> Complex64,
) -> Complex64
where
    T: Copy,
    T: FrequencyPoint,
{
    if frequency_hz <= points[0].frequency_hz() {
        return value(points[0]);
    }
    if frequency_hz >= points[points.len() - 1].frequency_hz() {
        return value(points[points.len() - 1]);
    }
    let upper = points.partition_point(|point| point.frequency_hz() < frequency_hz);
    let lower = upper - 1;
    let f0 = points[lower].frequency_hz();
    let f1 = points[upper].frequency_hz();
    let fraction = (frequency_hz - f0) / (f1 - f0);
    let v0 = value(points[lower]);
    v0 + (value(points[upper]) - v0) * fraction
}

trait FrequencyPoint {
    fn frequency_hz(self) -> f64;
}

impl FrequencyPoint for VelocityTransferPoint {
    fn frequency_hz(self) -> f64 {
        self.frequency_hz
    }
}

impl FrequencyPoint for ImpedancePoint {
    fn frequency_hz(self) -> f64 {
        self.frequency_hz
    }
}

#[derive(Debug, thiserror::Error)]
pub enum DriverModelError {
    #[error(transparent)]
    Modeling(#[from] ModelingError),
    #[error("{name} must be finite and physical")]
    InvalidScalar { name: &'static str },
    #[error("BL(x) samples must be finite, non-empty, and strictly ordered")]
    InvalidBlCurve,
    #[error("Le(f) samples must be passive, finite, non-empty, and strictly ordered")]
    InvalidInductanceCurve,
    #[error("velocity-transfer samples must be finite, non-empty, and strictly ordered")]
    InvalidVelocityTransfer,
    #[error("impedance samples must be finite, passive, non-empty, and strictly ordered")]
    InvalidImpedanceCurve,
    #[error("radiation impedance and frequency must be finite and physically usable")]
    InvalidRadiationImpedance,
    #[error("Le(f) and eddy-resistance curves must use the same frequency grid")]
    MismatchedMotorCurves,
    #[error("extracted motor curves must be finite, non-empty, and strictly ordered")]
    InvalidMotorCurve,
    #[error("extracted motor response is singular at {frequency_hz} Hz")]
    InvalidMotorResponse { frequency_hz: f64 },
    #[error("an operating-point response requires an attached nonlinear motor law")]
    NonlinearMotorRequired,
    #[error("thermal-state motor evaluation requires an attached driver thermal network")]
    ThermalNetworkRequired,
    #[error("FEM-derived motor handoff requires DriverModelExtractionEvidence")]
    ExtractionEvidenceRequired,
    #[error(
        "driver-model extraction requires converged {stage} evidence (converged={converged}, relative residual={relative_residual:e})"
    )]
    ExtractionEvidenceFailed {
        stage: &'static str,
        converged: bool,
        relative_residual: f64,
    },
    #[error("invalid driver-model reference artifact: {reason}")]
    InvalidReferenceArtifact { reason: &'static str },
    #[error(
        "driver-model reference mismatch at {frequency_hz} Hz for {observable}: absolute {absolute_error:e}, relative {relative_error:e}"
    )]
    ReferenceArtifactMismatch {
        frequency_hz: f64,
        observable: &'static str,
        absolute_error: f64,
        relative_error: f64,
    },
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::SpeakerDriverStudy;

    #[test]
    fn active_drive_preserves_impedance_and_scales_complex_loaded_velocity() {
        let study = SpeakerDriverStudy::detailed_frequency_domain_driver();
        let mut model = DriverModel::from_study(&study).unwrap();
        let f = study.frequency_hz[0];
        let radiation = study.mechanical_radiation_impedance(f).unwrap();
        let (velocity, impedance) = model
            .response_with_radiation_impedance(f, radiation)
            .unwrap();
        let original = model.velocity_at(f);
        model.drive_transfer = vec![
            DriveTransferPoint {
                frequency_hz: f,
                real: 0.0,
                imag: 2.0,
            },
            DriveTransferPoint {
                frequency_hz: f * 2.0,
                real: 2.0,
                imag: 0.0,
            },
        ];
        model.validate().unwrap();
        assert_eq!(model.drive_transfer_at(f * 1.5), Complex64::new(1.0, 1.0));
        assert_eq!(model.drive_transfer_at(f * 0.5), Complex64::new(0.0, 2.0));
        assert_eq!(model.drive_transfer_at(f * 3.0), Complex64::new(2.0, 0.0));
        assert!((model.velocity_at(f) - original * Complex64::new(0.0, 2.0)).norm() < 1e-12);
        let (driven, driven_impedance) = model
            .response_with_radiation_impedance(f, radiation)
            .unwrap();
        assert!((driven - velocity * Complex64::new(0.0, 2.0)).norm() < 1e-12);
        assert_eq!(driven_impedance, impedance);
        model.drive_transfer[1].frequency_hz = f;
        assert!(model.validate().is_err());
        model.drive_transfer[1].frequency_hz = f * 2.0;
        model.drive_transfer[1].imag = f64::NAN;
        assert!(model.validate().is_err());
    }

    fn complete_extraction_evidence() -> DriverModelExtractionEvidence {
        let converged = ExtractionResidualEvidence::new(true, 0.0);
        DriverModelExtractionEvidence {
            nonlinear_bias: converged,
            differential_permeability: converged,
            eddy_current: converged,
            structural: converged,
            acoustic: converged,
            power_closure: converged,
            coupling: converged,
        }
    }

    #[test]
    fn golden_driver_handoff_rejects_missing_coupling_evidence() {
        let study = SpeakerDriverStudy::detailed_frequency_domain_driver();
        let mut evidence = complete_extraction_evidence();
        evidence.coupling = ExtractionResidualEvidence::new(false, f64::INFINITY);

        let error = DriverModel::from_study_with_evidence(&study, evidence).unwrap_err();
        assert!(matches!(
            error,
            DriverModelError::ExtractionEvidenceFailed {
                stage: "coupling",
                ..
            }
        ));
    }

    #[test]
    fn golden_driver_handoff_accepts_complete_residual_evidence() {
        let study = SpeakerDriverStudy::detailed_frequency_domain_driver();
        let model =
            DriverModel::from_study_with_evidence(&study, complete_extraction_evidence()).unwrap();
        model.validate().unwrap();
    }

    #[test]
    fn extraction_handoff_rejects_power_closure_over_tolerance() {
        let study = SpeakerDriverStudy::detailed_frequency_domain_driver();
        let mut evidence = complete_extraction_evidence();
        evidence.power_closure = ExtractionResidualEvidence::new(true, 0.051);
        let error = DriverModel::from_study_with_evidence(&study, evidence).unwrap_err();
        assert!(matches!(
            error,
            DriverModelError::ExtractionEvidenceFailed {
                stage: "power closure",
                ..
            }
        ));
    }

    #[test]
    #[allow(deprecated)]
    fn unguarded_motor_handoff_cannot_bypass_extraction_gates() {
        let study = SpeakerDriverStudy::detailed_frequency_domain_driver();
        let motor = MotorParameters {
            bl_curve: vec![(-0.001, 5.0), (0.0, 6.0), (0.001, 5.0)],
            le_vs_frequency: vec![(100.0, 0.5e-3), (1_000.0, 0.4e-3)],
            re_eddy_vs_frequency: vec![(100.0, 0.1), (1_000.0, 0.4)],
            bl_nominal: 6.0,
            le_nominal: 0.55e-3,
            le_surface: Vec::new(),
            bl_temperature_coefficient_per_k: 0.0,
            magnetic_law: None,
            hysteresis_loss: None,
            eddy_current_loss: None,
        };
        assert!(matches!(
            DriverModel::from_study_with_motor(&study, &motor),
            Err(DriverModelError::ExtractionEvidenceRequired)
        ));
    }

    #[test]
    fn series_capacitance_high_passes_the_coil_network() {
        let study = SpeakerDriverStudy::detailed_frequency_domain_driver();
        let mut model = DriverModel::from_study(&study).unwrap();
        let frequency = 100.0;
        let open = model.network_parameters_at(frequency).unwrap();

        model.series_capacitance_f = Some(120.0e-6);
        model.validate().unwrap();
        let filtered = model.network_parameters_at(frequency).unwrap();

        // Same resistance, reactance shifted by exactly -1/(ωC).
        assert_eq!(open.coil_impedance_ohm.re, filtered.coil_impedance_ohm.re);
        let expected =
            open.coil_impedance_ohm.im - 1.0 / (std::f64::consts::TAU * frequency * 120.0e-6);
        assert!(
            (filtered.coil_impedance_ohm.im - expected).abs() < 1e-9,
            "capacitive reactance mismatch: {} vs {expected}",
            filtered.coil_impedance_ohm.im
        );

        // The capacitor suppresses the radiated velocity at low frequency:
        // with a fixed radiation load the response must drop vs the open coil.
        let radiation = Complex64::new(1.0, 0.5);
        let (v_open, _) = model
            .response_with_radiation_impedance(frequency, radiation)
            .unwrap();
        model.series_capacitance_f = None;
        let (v_plain, _) = model
            .response_with_radiation_impedance(frequency, radiation)
            .unwrap();
        assert!(
            v_open.norm() < v_plain.norm(),
            "series capacitor must high-pass the driver: {} vs {}",
            v_open.norm(),
            v_plain.norm()
        );

        // Non-positive capacitance is rejected.
        model.series_capacitance_f = Some(0.0);
        assert!(model.validate().is_err());
    }

    #[test]
    fn hot_network_handoff_replaces_copper_without_double_counting_eddy_loss() {
        let study = SpeakerDriverStudy::detailed_frequency_domain_driver();
        let model = DriverModel::from_study(&study).unwrap();
        let frequency_hz = 1_000.0;
        let cold = model.network_parameters_at(frequency_hz).unwrap();
        let hot_resistance = model.thiele_small.voice_coil_resistance_ohm + 2.5;
        let hot = model
            .network_parameters_at_with_hot_resistance(frequency_hz, hot_resistance)
            .unwrap();

        assert!((hot.coil_impedance_ohm.re - cold.coil_impedance_ohm.re - 2.5).abs() < 1.0e-12);
        assert_eq!(hot.coil_impedance_ohm.im, cold.coil_impedance_ohm.im);
        assert!(
            model
                .network_parameters_at_with_hot_resistance(frequency_hz, 0.0)
                .is_err()
        );
    }

    #[test]
    fn study_handoff_preserves_velocity_samples_and_ts_values() {
        let study = SpeakerDriverStudy::detailed_frequency_domain_driver();
        let model = DriverModel::from_study(&study).unwrap();
        let response = study
            .try_small_signal_response_at(
                study.frequency_hz[0],
                Complex64::new(study.drive_voltage_rms_v, 0.0),
            )
            .unwrap();

        assert_eq!(model.velocity_transfer.len(), study.frequency_hz.len());
        assert!(
            (model.velocity_at(study.frequency_hz[0]) - response.coil_velocity).norm() < 1.0e-12
        );
        assert_eq!(model.thiele_small.force_factor_tm, study.voice_coil.bl_tm);
        assert_eq!(
            model.thiele_small.moving_mass_kg,
            study.moving_system.moving_mass_kg
        );
        assert!(
            (model.electrical_impedance_at(study.frequency_hz[0]) - response.impedance).norm()
                < 1.0e-12
        );
    }

    #[test]
    fn motor_curves_replace_lumped_defaults_in_handoff() {
        let study = SpeakerDriverStudy::detailed_frequency_domain_driver();
        let motor = MotorParameters {
            bl_curve: vec![(-0.001, 5.0), (0.0, 6.0), (0.001, 5.0)],
            le_vs_frequency: vec![(100.0, 0.5e-3), (1_000.0, 0.4e-3)],
            re_eddy_vs_frequency: vec![(100.0, 0.1), (1_000.0, 0.4)],
            bl_nominal: 6.0,
            le_nominal: 0.55e-3,
            le_surface: Vec::new(),
            bl_temperature_coefficient_per_k: 0.0,
            magnetic_law: Some(
                crate::MotorMagneticLaw::from_bh_samples(
                    &[(0.0, 0.0), (0.5, 400.0), (1.0, 1_200.0)],
                    1.0e-5,
                    0.02,
                    None,
                )
                .unwrap(),
            ),
            hysteresis_loss: Some(crate::HysteresisLossModel {
                coefficient_w_per_m3: 2.0e-3,
                ..crate::HysteresisLossModel::default()
            }),
            eddy_current_loss: Some(
                crate::EddyCurrentLossModel::from_samples(&[
                    crate::EddyCurrentLossSample {
                        frequency_hz: 100.0,
                        flux_density_t: 0.5,
                        eddy_resistance_ohm: 0.1,
                        skin_depth_m: 0.01,
                        proximity_factor: 1.1,
                    },
                    crate::EddyCurrentLossSample {
                        frequency_hz: 1_000.0,
                        flux_density_t: 0.5,
                        eddy_resistance_ohm: 0.2,
                        skin_depth_m: 0.005,
                        proximity_factor: 1.2,
                    },
                    crate::EddyCurrentLossSample {
                        frequency_hz: 100.0,
                        flux_density_t: 1.0,
                        eddy_resistance_ohm: 0.2,
                        skin_depth_m: 0.009,
                        proximity_factor: 1.2,
                    },
                    crate::EddyCurrentLossSample {
                        frequency_hz: 1_000.0,
                        flux_density_t: 1.0,
                        eddy_resistance_ohm: 0.4,
                        skin_depth_m: 0.004,
                        proximity_factor: 1.4,
                    },
                ])
                .unwrap(),
            ),
        };
        let model = DriverModel::from_study_with_motor_and_evidence(
            &study,
            &motor,
            complete_extraction_evidence(),
        )
        .unwrap();

        assert_eq!(model.bl_curve.len(), 3);
        assert_eq!(model.inductance_curve.len(), 2);
        assert_eq!(model.thiele_small.force_factor_tm, 6.0);
        assert_eq!(model.thiele_small.nominal_inductance_h, 0.55e-3);
        assert!(
            model
                .nonlinear_motor
                .as_ref()
                .and_then(|motor| motor.magnetic.as_ref())
                .is_some()
        );
        assert_eq!(
            model
                .nonlinear_motor
                .as_ref()
                .unwrap()
                .hysteresis
                .coefficient_w_per_m3,
            2.0e-3
        );
        assert!(
            model
                .nonlinear_motor
                .as_ref()
                .and_then(|motor| motor.eddy_current.as_ref())
                .is_some()
        );

        let lumped = DriverModel::from_study(&study).unwrap();
        let frequency_hz = 1_000.0;
        assert_ne!(
            model.electrical_impedance_at(frequency_hz),
            lumped.electrical_impedance_at(frequency_hz),
            "extracted Le/Re must affect the response table"
        );
        assert_eq!(model.provenance, ArtifactProvenance::native_golden());
    }

    #[test]
    fn replacing_radiation_load_recovers_the_reference_response() {
        let study = SpeakerDriverStudy::detailed_frequency_domain_driver();
        let model = DriverModel::from_study(&study).unwrap();
        let frequency_hz = study.frequency_hz[8];
        let radiation = study.mechanical_radiation_impedance(frequency_hz).unwrap();
        let (velocity, impedance) = model
            .response_with_radiation_impedance(frequency_hz, radiation)
            .unwrap();

        assert!(
            (velocity - model.velocity_at(frequency_hz)).norm() / velocity.norm().max(1.0e-15)
                < 1.0e-10
        );
        assert!(
            (impedance - model.electrical_impedance_at(frequency_hz)).norm()
                / impedance.norm().max(1.0e-15)
                < 1.0e-10
        );
    }

    #[test]
    fn nonlinear_motor_operating_point_changes_network_and_reports_loss() {
        let motor = MotorConstitutiveModel {
            bl_curve: vec![
                crate::NonlinearBlPoint {
                    displacement_m: -0.001,
                    force_factor_tm: 5.0,
                },
                crate::NonlinearBlPoint {
                    displacement_m: 0.0,
                    force_factor_tm: 6.0,
                },
                crate::NonlinearBlPoint {
                    displacement_m: 0.001,
                    force_factor_tm: 5.0,
                },
            ],
            le_surface: vec![crate::NonlinearLePoint {
                displacement_m: 0.0,
                current_a: 0.0,
                frequency_hz: 1_000.0,
                inductance_h: 0.5e-3,
                eddy_resistance_ohm: 0.3,
            }],
            temperature: crate::MotorTemperatureLaw {
                copper_resistance_ohm: 5.8,
                copper_resistance_per_k: 0.00393,
                ..crate::MotorTemperatureLaw::default()
            },
            hysteresis: crate::HysteresisLossModel {
                coefficient_w_per_m3: 1.0,
                active_volume_m3: 1.0e-5,
                ..crate::HysteresisLossModel::default()
            },
            magnetic: None,
            eddy_current: None,
        };
        let model =
            DriverModel::from_study(&SpeakerDriverStudy::detailed_frequency_domain_driver())
                .unwrap()
                .with_nonlinear_motor(motor)
                .unwrap();
        let (network, values) = model
            .network_parameters_at_operating_point(crate::MotorOperatingPoint {
                displacement_m: 0.0,
                current_a: 1.0,
                frequency_hz: 1_000.0,
                temperature_k: 333.15,
                flux_density_t: 0.8,
            })
            .unwrap();
        assert!(network.coil_impedance_ohm.re > model.thiele_small.voice_coil_resistance_ohm);
        assert!(values.total_magnetic_loss_w > 0.0);
    }

    #[test]
    fn nonlinear_motor_can_use_thermal_network_resistance_authoritatively() {
        let motor = MotorConstitutiveModel {
            bl_curve: vec![
                crate::NonlinearBlPoint {
                    displacement_m: -0.001,
                    force_factor_tm: 5.0,
                },
                crate::NonlinearBlPoint {
                    displacement_m: 0.0,
                    force_factor_tm: 6.0,
                },
                crate::NonlinearBlPoint {
                    displacement_m: 0.001,
                    force_factor_tm: 5.0,
                },
            ],
            le_surface: vec![crate::NonlinearLePoint {
                displacement_m: 0.0,
                current_a: 0.0,
                frequency_hz: 1_000.0,
                inductance_h: 0.5e-3,
                eddy_resistance_ohm: 0.3,
            }],
            temperature: crate::MotorTemperatureLaw {
                copper_resistance_ohm: 5.8,
                copper_resistance_per_k: 0.00393,
                ..crate::MotorTemperatureLaw::default()
            },
            hysteresis: crate::HysteresisLossModel::default(),
            magnetic: None,
            eddy_current: None,
        };
        let model =
            DriverModel::from_study(&SpeakerDriverStudy::detailed_frequency_domain_driver())
                .unwrap()
                .with_nonlinear_motor(motor)
                .unwrap();
        let hot_resistance_ohm = 7.25;
        let (network, values) = model
            .network_parameters_at_operating_point_with_hot_resistance(
                crate::MotorOperatingPoint {
                    displacement_m: 0.0,
                    current_a: 0.0,
                    frequency_hz: 1_000.0,
                    temperature_k: 333.15,
                    flux_density_t: 0.0,
                },
                hot_resistance_ohm,
            )
            .unwrap();
        assert!((network.coil_impedance_ohm.re - (hot_resistance_ohm + 0.3)).abs() < 1.0e-12);
        assert!(values.copper_resistance_ohm != hot_resistance_ohm);
    }

    #[test]
    fn m3_thermal_handoff_uses_magnet_temperature_for_bl_and_coil_for_resistance() {
        let motor = MotorConstitutiveModel {
            bl_curve: vec![crate::NonlinearBlPoint {
                displacement_m: 0.0,
                force_factor_tm: 6.0,
            }],
            le_surface: vec![crate::NonlinearLePoint {
                displacement_m: 0.0,
                current_a: 0.0,
                frequency_hz: 1_000.0,
                inductance_h: 0.5e-3,
                eddy_resistance_ohm: 0.3,
            }],
            temperature: crate::MotorTemperatureLaw {
                copper_resistance_ohm: 5.8,
                bl_per_k: -1.0e-3,
                ..crate::MotorTemperatureLaw::default()
            },
            hysteresis: crate::HysteresisLossModel::default(),
            magnetic: None,
            eddy_current: None,
        };
        let mut model =
            DriverModel::from_study(&SpeakerDriverStudy::detailed_frequency_domain_driver())
                .unwrap()
                .with_nonlinear_motor(motor)
                .unwrap();
        model.thermal_network = Some(crate::DriverThermalNetworkConfig::default());
        let state = crate::ThermalNetworkState {
            temperatures_k: [313.15, 333.15, 303.15, 298.15],
            power_w: crate::ThermalNetworkPower {
                voice_coil_w: 1.0,
                magnet_w: 0.1,
                gap_w: 0.0,
                cabinet_w: 0.0,
            },
            maximum_balance_residual_w: 0.0,
        };

        let (network, values) = model
            .network_parameters_at_thermal_state(
                crate::MotorOperatingPoint {
                    displacement_m: 0.0,
                    current_a: 0.0,
                    frequency_hz: 1_000.0,
                    temperature_k: 999.0,
                    flux_density_t: 0.0,
                },
                state,
            )
            .unwrap();

        assert!((values.force_factor_tm - 6.0 * 0.96).abs() < 1.0e-12);
        let expected_hot_re =
            model.thiele_small.voice_coil_resistance_ohm * (1.0 + 0.00393 * (313.15 - 293.15));
        assert!((network.coil_impedance_ohm.re - (expected_hot_re + 0.3)).abs() < 1.0e-12);
    }

    #[test]
    fn nonlinear_operating_domain_evidence_is_attached_and_revalidated() {
        let motor = MotorConstitutiveModel {
            bl_curve: vec![crate::NonlinearBlPoint {
                displacement_m: 0.0,
                force_factor_tm: 6.0,
            }],
            le_surface: vec![crate::NonlinearLePoint {
                displacement_m: 0.0,
                current_a: 0.0,
                frequency_hz: 1_000.0,
                inductance_h: 0.5e-3,
                eddy_resistance_ohm: 0.3,
            }],
            temperature: crate::MotorTemperatureLaw::default(),
            hysteresis: crate::HysteresisLossModel::default(),
            magnetic: None,
            eddy_current: None,
        };
        let domain = crate::MotorOperatingDomain {
            displacement_m: vec![0.0],
            current_a: vec![0.0],
            frequency_hz: vec![1_000.0],
            temperature_k: vec![293.15],
            flux_density_t: vec![0.0],
        };
        let model =
            DriverModel::from_study(&SpeakerDriverStudy::detailed_frequency_domain_driver())
                .unwrap()
                .with_nonlinear_operating_domain(motor, domain)
                .unwrap();
        assert!(model.nonlinear_operating_domain_evidence.is_some());
        model.validate().unwrap();

        let mut tampered = model.clone();
        tampered
            .nonlinear_operating_domain_evidence
            .as_mut()
            .unwrap()
            .values[0]
            .inductance_h *= 2.0;
        assert!(tampered.validate().is_err());
    }

    #[test]
    fn driver_model_reference_artifact_replays_deterministically() {
        let model =
            DriverModel::from_study(&SpeakerDriverStudy::detailed_frequency_domain_driver())
                .unwrap();
        let frequencies = [100.0, 1_000.0, 10_000.0];
        let radiation = [
            Complex64::new(0.2, 0.1),
            Complex64::new(0.4, 0.3),
            Complex64::new(0.8, 0.5),
        ];
        let artifact =
            DriverModelReferenceArtifact::from_model(&model, &frequencies, &radiation).unwrap();
        let replay =
            DriverModelReferenceArtifact::from_model(&model, &frequencies, &radiation).unwrap();

        assert_eq!(
            artifact.canonical_json().unwrap(),
            replay.canonical_json().unwrap()
        );
        artifact
            .validate_against_model(&model, DriverModelReferenceTolerance::default())
            .unwrap();

        let encoded = artifact.canonical_json().unwrap();
        let decoded: DriverModelReferenceArtifact = serde_json::from_slice(&encoded).unwrap();
        assert_eq!(decoded.canonical_json().unwrap(), encoded);
        decoded
            .validate_against_model(&model, DriverModelReferenceTolerance::default())
            .unwrap();
    }

    #[test]
    fn driver_model_reference_artifact_rejects_golden_backend_tampering() {
        let model =
            DriverModel::from_study(&SpeakerDriverStudy::detailed_frequency_domain_driver())
                .unwrap();
        let mut artifact = DriverModelReferenceArtifact::from_model(
            &model,
            &[100.0, 1_000.0],
            &[Complex64::new(0.2, 0.1), Complex64::new(0.4, 0.3)],
        )
        .unwrap();
        artifact.provenance.backend = crate::conventions::ArtifactBackend::Wasm;
        assert!(matches!(
            artifact.validate(),
            Err(DriverModelError::InvalidReferenceArtifact { .. })
        ));
    }

    #[test]
    fn driver_model_reference_artifact_rejects_response_tampering() {
        let model =
            DriverModel::from_study(&SpeakerDriverStudy::detailed_frequency_domain_driver())
                .unwrap();
        let mut artifact = DriverModelReferenceArtifact::from_model(
            &model,
            &[100.0, 1_000.0],
            &[Complex64::new(0.2, 0.1), Complex64::new(0.4, 0.3)],
        )
        .unwrap();
        artifact.points[1].velocity_real_m_per_s += 1.0;
        assert!(matches!(
            artifact.validate_against_model(&model, DriverModelReferenceTolerance::default()),
            Err(DriverModelError::InvalidReferenceArtifact { .. })
        ));
    }

    #[test]
    fn projected_driver_model_reference_replays_against_cpu_golden() {
        let model =
            DriverModel::from_study(&SpeakerDriverStudy::detailed_frequency_domain_driver())
                .unwrap();
        let golden = DriverModelReferenceArtifact::from_model(
            &model,
            &[100.0, 1_000.0],
            &[Complex64::new(0.2, 0.1), Complex64::new(0.4, 0.3)],
        )
        .unwrap();
        let mut projected = golden
            .clone()
            .with_provenance(crate::conventions::ArtifactProvenance::wasm_preview());
        projected
            .validate_against_golden(&golden, DriverModelReferenceTolerance::default())
            .unwrap();

        projected.points[0].electrical_impedance_imag_ohm += 1.0;
        assert!(matches!(
            projected.validate_against_golden(&golden, DriverModelReferenceTolerance::default()),
            Err(DriverModelError::InvalidReferenceArtifact { .. })
        ));
    }

    #[test]
    fn reference_artifact_rejects_stale_grid_checksum_and_precision() {
        let model =
            DriverModel::from_study(&SpeakerDriverStudy::detailed_frequency_domain_driver())
                .unwrap();
        let mut stale_grid = DriverModelReferenceArtifact::from_model(
            &model,
            &[100.0, 1_000.0],
            &[Complex64::new(0.2, 0.1), Complex64::new(0.4, 0.3)],
        )
        .unwrap();
        stale_grid.points[0].frequency_hz = 200.0;
        assert!(matches!(
            stale_grid.validate(),
            Err(DriverModelError::InvalidReferenceArtifact { .. })
        ));

        let mut wrong_grid = DriverModelReferenceArtifact::from_model(
            &model,
            &[100.0, 1_000.0],
            &[Complex64::new(0.2, 0.1), Complex64::new(0.4, 0.3)],
        )
        .unwrap();
        wrong_grid.execution.frequency_grid_identity = "0".repeat(64);
        assert!(matches!(
            wrong_grid.validate(),
            Err(DriverModelError::InvalidReferenceArtifact { .. })
        ));

        let mut wrong_precision = DriverModelReferenceArtifact::from_model(
            &model,
            &[100.0, 1_000.0],
            &[Complex64::new(0.2, 0.1), Complex64::new(0.4, 0.3)],
        )
        .unwrap();
        wrong_precision.execution.precision = crate::conventions::ArtifactPrecision::F32;
        assert!(matches!(
            wrong_precision.validate(),
            Err(DriverModelError::InvalidReferenceArtifact { .. })
        ));
    }

    #[test]
    fn reference_artifact_rejects_nonpassive_and_relabelled_cuda_output() {
        let model =
            DriverModel::from_study(&SpeakerDriverStudy::detailed_frequency_domain_driver())
                .unwrap();
        let mut nonpassive = DriverModelReferenceArtifact::from_model(
            &model,
            &[100.0, 1_000.0],
            &[Complex64::new(0.2, 0.1), Complex64::new(0.4, 0.3)],
        )
        .unwrap();
        nonpassive.points[0].radiation_impedance_real_n_s_per_m = -1.0;
        assert!(matches!(
            nonpassive.validate(),
            Err(DriverModelError::InvalidReferenceArtifact { .. })
        ));

        let mut relabelled = DriverModelReferenceArtifact::from_model(
            &model,
            &[100.0, 1_000.0],
            &[Complex64::new(0.2, 0.1), Complex64::new(0.4, 0.3)],
        )
        .unwrap();
        relabelled.provenance.backend = crate::conventions::ArtifactBackend::Cuda;
        relabelled.execution.requested_backend = crate::conventions::ArtifactBackend::Cuda;
        relabelled.execution.actual_backend = crate::conventions::ArtifactBackend::Cuda;
        relabelled.execution.device = "cuda:fake".to_string();
        relabelled.execution.transfer.h2d_operations = 1;
        relabelled.execution.transfer.d2h_operations = 1;
        assert!(matches!(
            relabelled.validate(),
            Err(DriverModelError::InvalidReferenceArtifact { .. })
        ));
    }

    #[test]
    fn reference_artifact_rejects_f32_cuda_claim_even_with_transfer_counters() {
        let model =
            DriverModel::from_study(&SpeakerDriverStudy::detailed_frequency_domain_driver())
                .unwrap();
        let mut artifact = DriverModelReferenceArtifact::from_model(
            &model,
            &[100.0, 1_000.0],
            &[Complex64::new(0.2, 0.1), Complex64::new(0.4, 0.3)],
        )
        .unwrap();
        artifact.provenance.backend = crate::conventions::ArtifactBackend::Cuda;
        artifact.provenance.execution_profile =
            crate::conventions::ArtifactExecutionProfile::ProjectedReanalysis;
        artifact.provenance.precision = crate::conventions::ArtifactPrecision::F32;
        artifact.provenance.deterministic = true;
        artifact.execution.requested_backend = crate::conventions::ArtifactBackend::Cuda;
        artifact.execution.actual_backend = crate::conventions::ArtifactBackend::Cuda;
        artifact.execution.precision = crate::conventions::ArtifactPrecision::F32;
        artifact.execution.device = "cuda:0".into();
        artifact.execution.toolchain = "test-cuda".into();
        artifact.execution.transfer.h2d_operations = 1;
        artifact.execution.transfer.h2d_bytes = 1;
        artifact.execution.transfer.d2h_operations = 1;
        artifact.execution.transfer.d2h_bytes = 1;
        artifact.execution.transfer.allocation_high_water_bytes = 1;
        artifact.execution.reference_checksum_sha256.clear();
        artifact.execution.reference_checksum_sha256 = artifact.checksum();
        assert!(matches!(
            artifact.validate(),
            Err(DriverModelError::InvalidReferenceArtifact { .. })
        ));
    }

    #[test]
    fn reference_artifact_rejects_failed_rom_promotion() {
        let model =
            DriverModel::from_study(&SpeakerDriverStudy::detailed_frequency_domain_driver())
                .unwrap();
        let artifact = DriverModelReferenceArtifact::from_model(
            &model,
            &[100.0, 1_000.0],
            &[Complex64::new(0.2, 0.1), Complex64::new(0.4, 0.3)],
        )
        .unwrap();
        let mut execution = artifact.execution.clone();
        execution.promotion.promoted = false;
        let candidate = artifact.with_execution(execution).unwrap();
        assert!(candidate.validate().is_ok());
        assert!(matches!(
            candidate.validate_rom_promotion(),
            Err(DriverModelError::InvalidReferenceArtifact { .. })
        ));
    }

    #[test]
    fn reference_artifact_rejects_preview_model_extraction() {
        let preview =
            DriverModel::from_study(&SpeakerDriverStudy::detailed_frequency_domain_driver())
                .unwrap()
                .with_provenance(crate::conventions::ArtifactProvenance::native_preview());
        assert!(matches!(
            DriverModelReferenceArtifact::from_model(
                &preview,
                &[1_000.0],
                &[Complex64::new(0.4, 0.3)],
            ),
            Err(DriverModelError::InvalidReferenceArtifact { .. })
        ));
    }
}
