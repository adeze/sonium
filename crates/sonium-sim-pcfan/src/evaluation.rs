use crate::cfd::solve_pc_fan_cfd_mean_flow_for_geometry;
use crate::model::*;
use rayon::prelude::*;
use sonium_fem::physics::acoustics::aeroacoustics::PropellerSource;
use sonium_fem::physics::acoustics::pressure::sound_metrics;
use std::fmt;

#[cfg(test)]
mod envelope_regression_tests {
    use super::*;

    #[test]
    fn default_envelopes_are_finite_and_feasible() {
        let mut config = PcFanConfig::bench_120mm();
        // This is an operating-envelope reference, not a target-flow
        // optimization case. Use a broad acceptance band so the baseline's
        // reduced-order prior does not turn the trend check into an
        // unrelated feasibility gate.
        config.weights.airflow_band_fraction = 1.0;
        let design = config.default_design_variables();
        let evaluation = evaluate_pc_fan(&config, &design);
        assert!(
            evaluation.is_feasible(&config),
            "airflow={} collision={:?} margins={:?} thermal={:?} structural={:?} manufacturing={:?}",
            evaluation.airflow_cfm,
            evaluation.collision_status,
            evaluation.constraint_margins,
            evaluation.thermal,
            evaluation.structural,
            evaluation.manufacturing
        );
    }

    #[test]
    fn thermal_structural_coupling_adds_temperature_dependent_stress_and_growth() {
        let config = PcFanConfig::bench_120mm();
        let design = config.default_design_variables();
        let flow = PcFanFlowSolution::zero();
        let cold = thermal_solution(&config, &flow);
        let mut hot = cold;
        hot.winding_temperature_c += 20.0;

        let cold_structural = structural_solution(&config, &design, &flow, &cold);
        let hot_structural = structural_solution(&config, &design, &flow, &hot);

        assert_eq!(cold_structural.thermal_stress_pa, 0.0);
        assert_eq!(cold_structural.thermal_expansion_m, 0.0);
        assert!(hot_structural.thermal_stress_pa > 0.0);
        assert!(hot_structural.thermal_expansion_m > 0.0);
        assert!(hot_structural.maximum_stress_pa > cold_structural.maximum_stress_pa);
        assert_eq!(cold_structural.fatigue_alternating_stress_pa, 0.0);
        assert!(cold_structural.fatigue_margin_pa > 0.0);
        assert!(cold_structural.fatigue_allowable_stress_pa > 0.0);
    }

    #[test]
    fn checked_evaluation_rejects_invalid_thermal_structural_coupling() {
        let design = PcFanConfig::bench_120mm().default_design_variables();
        let mut config = PcFanConfig::bench_120mm();
        config.structural.thermal_constraint_factor = 1.1;
        assert!(matches!(
            try_evaluate_pc_fan(&config, &design),
            Err(PcFanCfdEvaluationError::InvalidEnvelopeParameter {
                envelope: "structural",
                parameter: "thermal_constraint_factor",
                ..
            })
        ));

        config.structural.thermal_expansion_coefficient_per_k = -1.0;
        assert!(matches!(
            try_evaluate_pc_fan(&config, &design),
            Err(PcFanCfdEvaluationError::InvalidEnvelopeParameter {
                envelope: "structural",
                parameter: "thermal_expansion_coefficient_per_k",
                ..
            })
        ));

        config.structural = PcFanStructuralConfig::default();
        config.structural.fatigue_reference_stress_pa = 0.0;
        assert!(matches!(
            try_evaluate_pc_fan(&config, &design),
            Err(PcFanCfdEvaluationError::InvalidEnvelopeParameter {
                envelope: "structural",
                parameter: "fatigue_reference_stress_pa",
                ..
            })
        ));

        config.structural = PcFanStructuralConfig::default();
        config.structural.fatigue_design_cycles = 1.0e5;
        assert!(matches!(
            try_evaluate_pc_fan(&config, &design),
            Err(PcFanCfdEvaluationError::InvalidEnvelopeParameter {
                envelope: "structural",
                parameter: "fatigue_design_cycles",
                ..
            })
        ));
    }

    #[test]
    fn fatigue_allowable_decreases_with_design_life() {
        let mut config = PcFanConfig::bench_120mm();
        let design = config.default_design_variables();
        let flow = PcFanFlowSolution {
            pressure_rise_pa: 80.0,
            ..PcFanFlowSolution::zero()
        };
        let thermal = thermal_solution(&config, &flow);
        let long_life = structural_solution(&config, &design, &flow, &thermal);

        config.structural.fatigue_design_cycles = config.structural.fatigue_reference_cycles;
        let reference_life = structural_solution(&config, &design, &flow, &thermal);

        assert!(long_life.fatigue_allowable_stress_pa < reference_life.fatigue_allowable_stress_pa);
        assert!(long_life.fatigue_alternating_stress_pa > 0.0);
    }

    #[test]
    fn fatigue_margin_is_a_hard_feasibility_constraint() {
        let mut config = PcFanConfig::bench_120mm();
        config.weights.airflow_band_fraction = 1.0;
        let design = config.default_design_variables();
        let mut evaluation = evaluate_pc_fan(&config, &design);
        assert!(evaluation.is_feasible(&config));

        evaluation.structural.fatigue_margin_pa = -1.0;
        assert!(!evaluation.is_feasible(&config));
        assert!(evaluation.feasibility_violation(&config) > 0.0);
    }

    #[test]
    fn checked_evaluation_rejects_invalid_manufacturing_parameter() {
        let mut config = PcFanConfig::bench_120mm();
        config.manufacturing.min_blade_chord_m = f64::NAN;
        let design = config.default_design_variables();

        assert!(matches!(
            try_evaluate_pc_fan(&config, &design),
            Err(PcFanCfdEvaluationError::InvalidEnvelopeParameter {
                envelope: "manufacturing",
                parameter: "min_blade_chord_m",
                ..
            })
        ));
    }

    #[test]
    fn checked_evaluation_rejects_invalid_manufacturing_tolerance() {
        let mut config = PcFanConfig::bench_120mm();
        config.manufacturing.tolerance.blade_thickness_fraction = 1.0;
        let design = config.default_design_variables();
        assert!(matches!(
            try_evaluate_pc_fan(&config, &design),
            Err(PcFanCfdEvaluationError::InvalidEnvelopeParameter {
                envelope: "manufacturing",
                parameter: "blade_thickness_fraction",
                ..
            })
        ));
    }

    #[test]
    fn checked_evaluation_rejects_invalid_conservation_tolerance() {
        let mut config = PcFanConfig::bench_120mm();
        config.cfd.max_relative_momentum_imbalance = f64::NAN;
        let design = config.default_design_variables();

        assert!(matches!(
            try_evaluate_pc_fan(&config, &design),
            Err(PcFanCfdEvaluationError::InvalidConservationTolerance {
                metric: "momentum imbalance",
                ..
            })
        ));
    }

    fn small_cfd_fixture() -> (PcFanConfig, PcFanDesignVariables) {
        let config = PcFanConfig {
            mesh: PcFanMeshResolution {
                radial_segments: 2,
                chord_segments: 2,
                circumferential_segments: 8,
                harmonic_count: 2,
            },
            cfd: PcFanCfdConfig {
                flow_model: PcFanFlowModel::IncompressibleCfd,
                cells_x: 2,
                cells_y: 2,
                cells_z: 2,
                max_picard_iterations: 1,
                nonlinear_tolerance: 1.0e-4,
                ..PcFanCfdConfig::default()
            },
            ..PcFanConfig::bench_120mm()
        };
        let design = config.default_design_variables();
        (config, design)
    }

    #[test]
    fn cfd_conservation_ledger_is_finite() {
        let (config, design) = small_cfd_fixture();
        let cfd = crate::solve_pc_fan_cfd_mean_flow(&config, &design);
        let ledger = cfd.conservation;
        let values = [
            ledger.inlet_mass_flow_kg_s,
            ledger.outlet_mass_flow_kg_s,
            ledger.relative_mass_imbalance,
            ledger.inlet_axial_momentum_flux_n,
            ledger.outlet_axial_momentum_flux_n,
            ledger.pressure_force_n,
            ledger.body_force_n,
            ledger.axial_momentum_residual_n,
            ledger.relative_momentum_imbalance,
            ledger.pressure_force_from_loading_n,
            ledger.relative_pressure_closure,
            ledger.pressure_power_w,
            ledger.shaft_power_to_fluid_w,
            ledger.relative_energy_imbalance,
        ];
        assert!(values.iter().all(|value| value.is_finite()));
        assert!(
            (ledger.relative_mass_imbalance
                - cfd.mass_balance_error_kg_s.abs()
                    / ledger
                        .inlet_mass_flow_kg_s
                        .abs()
                        .max(ledger.outlet_mass_flow_kg_s.abs()))
            .abs()
                < 1.0e-12
        );
    }

    #[test]
    fn checked_admission_rejects_malformed_cfd_fields() {
        let (config, design) = small_cfd_fixture();
        let mut cfd = crate::solve_pc_fan_cfd_mean_flow(&config, &design);
        cfd.pressure_pa.pop();
        assert!(matches!(
            try_evaluate_pc_fan_from_cfd_solution(&config, &design, &cfd),
            Err(PcFanCfdEvaluationError::InvalidFieldShape {
                field: "pressure",
                ..
            })
        ));

        let mut cfd = crate::solve_pc_fan_cfd_mean_flow(&config, &design);
        cfd.velocity_m_s[0][1] = f64::NAN;
        assert!(matches!(
            try_evaluate_pc_fan_from_cfd_solution(&config, &design, &cfd),
            Err(PcFanCfdEvaluationError::NonFiniteField {
                field: "velocity",
                index: 0,
            })
        ));

        let mut cfd = crate::solve_pc_fan_cfd_mean_flow(&config, &design);
        cfd.airflow_cfm = f64::INFINITY;
        assert!(matches!(
            try_evaluate_pc_fan_from_cfd_solution(&config, &design, &cfd),
            Err(PcFanCfdEvaluationError::NonFiniteObservable {
                field: "airflow_cfm",
            })
        ));
    }

    #[test]
    fn checked_admission_rejects_stalled_or_reversed_operating_point() {
        let (config, design) = small_cfd_fixture();
        let mut cfd = crate::solve_pc_fan_cfd_mean_flow(&config, &design);
        // The 2x2x2 fixture never reaches Picard convergence, so declare it
        // (same pattern as the backend-admission test below): this test
        // isolates the operating-point check, while the convergence gate has
        // its own tests.
        cfd.converged = true;
        cfd.residual = 0.0;
        // Force a stalled operating point: no forward pressure gain.
        cfd.pressure_drop_pa = cfd.pressure_drop_pa.abs().max(1.0e-6);
        let error = try_evaluate_pc_fan_from_cfd_solution(&config, &design, &cfd)
            .expect_err("stalled flow must not be admitted");
        assert!(
            matches!(
                error,
                PcFanCfdEvaluationError::ReversedOperatingPoint { .. }
            ),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn checked_admission_rejects_unadmitted_backend_execution() {
        let (config, design) = small_cfd_fixture();
        let mut cfd = crate::solve_pc_fan_cfd_mean_flow(&config, &design);
        cfd.converged = true;
        cfd.residual = 0.0;

        let mut strict_backend_config = config.clone();
        strict_backend_config.cfd.backend = PcFanCfdBackend::Cuda;
        strict_backend_config.cfd.allow_host_fallback = false;
        cfd.backend_provenance =
            PcFanCfdBackendProvenance::from_dispatch(PcFanCfdBackend::Cuda, PcFanCfdBackend::Cpu);
        cfd.actual_backend = PcFanCfdBackend::Cpu;
        cfd.execution_environment = PcFanCfdExecutionEnvironment::capture(
            &strict_backend_config.cfd,
            cfd.backend_provenance,
        );
        assert!(matches!(
            try_evaluate_pc_fan_from_cfd_solution(&strict_backend_config, &design, &cfd),
            Err(PcFanCfdEvaluationError::BackendFallback {
                requested: PcFanCfdBackend::Cuda,
                actual: PcFanCfdBackend::Cpu,
                ..
            })
        ));

        let mut resident_config = config;
        resident_config.cfd.backend = PcFanCfdBackend::Cpu;
        resident_config.cfd.require_device_resident = true;
        cfd.backend_provenance =
            PcFanCfdBackendProvenance::from_dispatch(PcFanCfdBackend::Cpu, PcFanCfdBackend::Cpu);
        cfd.execution_environment =
            PcFanCfdExecutionEnvironment::capture(&resident_config.cfd, cfd.backend_provenance);
        assert!(matches!(
            try_evaluate_pc_fan_from_cfd_solution(&resident_config, &design, &cfd),
            Err(PcFanCfdEvaluationError::BackendResidencyUnavailable {
                requested: PcFanCfdBackend::Cpu,
                actual: PcFanCfdBackend::Cpu,
                mode: PcFanCfdExecutionMode::Cpu,
            })
        ));
    }

    #[test]
    fn checked_admission_rejects_injected_conservation_failures() {
        let (config, design) = small_cfd_fixture();
        let mut cfd = crate::solve_pc_fan_cfd_mean_flow(&config, &design);
        cfd.converged = true;
        cfd.residual = 0.0;

        cfd.conservation.relative_mass_imbalance = config.cfd.max_relative_mass_imbalance + 1.0;
        assert!(matches!(
            try_evaluate_pc_fan_from_cfd_solution(&config, &design, &cfd),
            Err(PcFanCfdEvaluationError::MassImbalance { .. })
        ));
        cfd.conservation.relative_mass_imbalance = 0.0;

        cfd.conservation.relative_momentum_imbalance =
            config.cfd.max_relative_momentum_imbalance + 1.0;
        assert!(matches!(
            try_evaluate_pc_fan_from_cfd_solution(&config, &design, &cfd),
            Err(PcFanCfdEvaluationError::MomentumImbalance { .. })
        ));

        cfd.conservation.relative_momentum_imbalance = 0.0;
        cfd.conservation.relative_pressure_closure = config.cfd.max_relative_pressure_closure + 1.0;
        assert!(matches!(
            try_evaluate_pc_fan_from_cfd_solution(&config, &design, &cfd),
            Err(PcFanCfdEvaluationError::PressureClosure { .. })
        ));

        cfd.conservation.relative_pressure_closure = 0.0;
        cfd.conservation.relative_energy_imbalance = config.cfd.max_relative_energy_imbalance + 1.0;
        assert!(matches!(
            try_evaluate_pc_fan_from_cfd_solution(&config, &design, &cfd),
            Err(PcFanCfdEvaluationError::EnergyImbalance { .. })
        ));

        cfd.conservation.relative_energy_imbalance = 0.0;

        let original_loading = cfd
            .actuator_loading
            .expect("actuator fixture should publish a loading ledger");
        let mut corrupted_loading = original_loading;
        corrupted_loading.shaft_power_to_fluid_w *= 1.1;
        cfd.actuator_loading = Some(corrupted_loading);
        assert!(matches!(
            try_evaluate_pc_fan_from_cfd_solution(&config, &design, &cfd),
            Err(PcFanCfdEvaluationError::InvalidActuatorLoading)
        ));
        cfd.actuator_loading = Some(original_loading);

        cfd.conservation.shaft_power_to_fluid_w *= 1.1;
        assert!(matches!(
            try_evaluate_pc_fan_from_cfd_solution(&config, &design, &cfd),
            Err(PcFanCfdEvaluationError::InvalidActuatorLoading)
        ));
        cfd.conservation.shaft_power_to_fluid_w = original_loading.shaft_power_to_fluid_w;

        cfd.conservation.body_force_n += 1.0;
        assert!(matches!(
            try_evaluate_pc_fan_from_cfd_solution(&config, &design, &cfd),
            Err(PcFanCfdEvaluationError::InvalidActuatorLoading)
        ));
        cfd.conservation.body_force_n = original_loading.force_on_fluid_n[2];

        cfd.actuator_loading = None;
        assert!(matches!(
            try_evaluate_pc_fan_from_cfd_solution(&config, &design, &cfd),
            Err(PcFanCfdEvaluationError::MissingPowerLedger)
        ));
    }

    #[test]
    fn checked_admission_rejects_stale_conservation_residuals() {
        let (config, design) = small_cfd_fixture();
        let mut cfd = crate::solve_pc_fan_cfd_mean_flow(&config, &design);
        cfd.converged = true;
        cfd.residual = 0.0;
        cfd.conservation.inlet_mass_flow_kg_s *= 1.25;
        cfd.conservation.relative_mass_imbalance = 0.0;

        let result = try_evaluate_pc_fan_from_cfd_solution(&config, &design, &cfd);
        assert!(matches!(
            result,
            Err(PcFanCfdEvaluationError::InvalidConservationLedger)
        ));

        let mut stale_momentum = cfd.clone();
        stale_momentum.conservation.inlet_axial_momentum_flux_n *= 1.25;
        stale_momentum.conservation.relative_momentum_imbalance = 0.0;
        assert!(matches!(
            try_evaluate_pc_fan_from_cfd_solution(&config, &design, &stale_momentum),
            Err(PcFanCfdEvaluationError::InvalidConservationLedger)
        ));

        let mut stale_pressure = cfd.clone();
        stale_pressure.conservation.pressure_force_n *= 1.25;
        stale_pressure.conservation.relative_pressure_closure = 0.0;
        assert!(matches!(
            try_evaluate_pc_fan_from_cfd_solution(&config, &design, &stale_pressure),
            Err(PcFanCfdEvaluationError::InvalidConservationLedger)
        ));

        let mut stale_energy = cfd;
        stale_energy.conservation.pressure_power_w *= 1.25;
        stale_energy.conservation.relative_energy_imbalance = 0.0;
        assert!(matches!(
            try_evaluate_pc_fan_from_cfd_solution(&config, &design, &stale_energy),
            Err(PcFanCfdEvaluationError::InvalidConservationLedger)
        ));
    }

    #[test]
    fn rotating_wall_cfd_publishes_loading_ledger() {
        let (mut config, design) = small_cfd_fixture();
        config.cfd.flow_model = PcFanFlowModel::IncompressibleCfdFull;
        let cfd = crate::solve_pc_fan_cfd_mean_flow(&config, &design);
        let loading = cfd
            .rotating_wall_loading
            .expect("rotating-wall CFD must publish immersed-wall loading");
        assert!(
            loading
                .force_on_fluid_n
                .iter()
                .all(|value| value.is_finite())
        );
        assert!(loading.torque_on_fluid_nm.is_finite());
        assert!(loading.shaft_power_to_fluid_w.is_finite());
    }
}

const M3S_TO_CFM: f64 = 2_118.880_003_289_3;
const PRESSURE_REFERENCE_PA: f64 = 20.0e-6;
const SOUND_POWER_REFERENCE_W: f64 = 1.0e-12;
/// Stall floor for the inlet-minus-outlet pressure drop [Pa]: drops above
/// this are stalled/reversed flow and rejected; drops at or below it are
/// numerical noise around a zero-gain operating point, not a stall.
const STALLED_FLOW_DROP_TOLERANCE_PA: f64 = 1e-9;
/// Relative round-off budget for algebraic CFD ledger identities. These
/// identities are evaluated from published primitive fields, so a violation
/// indicates a stale or corrupted diagnostic rather than a physical modelling
/// discrepancy.
const CFD_ALGEBRAIC_IDENTITY_RELATIVE_TOLERANCE: f64 = 1.0e-9;

/// Error returned when a caller requests a shape gradient from a policy that
/// deliberately routes the objective through a derivative-free optimizer.
#[derive(Debug, Clone, PartialEq)]
pub enum PcFanDerivativePolicyError {
    /// No derivative is promised by the selected policy.
    ShapeGradientUnavailable { policy: PcFanDerivativePolicy },
    /// The reduced-order adjoint cannot be applied to a resolved CFD state.
    DiscreteAdjointUnavailable { policy: PcFanDerivativePolicy },
    /// The legacy full-objective finite-difference path is not admitted for
    /// resolved CFD because it does not carry the checked CFD convergence and
    /// conservation contract through each perturbed evaluation.
    FiniteDifferenceUnavailable { policy: PcFanDerivativePolicy },
    /// A checked resolved-CFD base or perturbed solve failed admission.
    FiniteDifferenceEvaluationFailed {
        policy: PcFanDerivativePolicy,
        source: PcFanCfdEvaluationError,
    },
    /// The reduced acoustic pressure state is at the diagnostic dB floor.
    ///
    /// Physical pressure may be zero, but the clipped logarithmic diagnostic
    /// is not differentiable there. Do not publish a reduced adjoint gradient
    /// for that state.
    AcousticStateAtDiagnosticFloor { harmonic: usize },
}

impl fmt::Display for PcFanDerivativePolicyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ShapeGradientUnavailable { policy } => write!(
                f,
                "PC-fan shape gradient is unavailable under derivative policy {policy:?}; use the derivative-free optimizer"
            ),
            Self::DiscreteAdjointUnavailable { policy } => write!(
                f,
                "PC-fan reduced discrete adjoint unavailable under policy {policy:?} for this flow model"
            ),
            Self::FiniteDifferenceUnavailable { policy } => write!(
                f,
                "PC-fan full-objective finite-difference gradient unavailable under policy {policy:?} for resolved CFD; use a checked resolved-flow derivative or derivative-free optimizer"
            ),
            Self::FiniteDifferenceEvaluationFailed { policy, source } => write!(
                f,
                "PC-fan checked resolved-CFD finite-difference evaluation failed under policy {policy:?}: {source}"
            ),
            Self::AcousticStateAtDiagnosticFloor { harmonic } => write!(
                f,
                "PC-fan reduced acoustic state harmonic {harmonic} is at the diagnostic dB floor; gradient is not admitted"
            ),
        }
    }
}

/// Why a CFD flow field cannot be admitted to objective evaluation.
#[derive(Debug, Clone, PartialEq)]
pub enum PcFanCfdEvaluationError {
    /// The configured nonlinear tolerance is not a usable positive finite bound.
    InvalidNonlinearTolerance {
        /// Requested tolerance.
        tolerance: f64,
    },
    /// The CFD solver did not declare nonlinear convergence.
    NonConverged {
        /// Picard iterations performed.
        iterations: usize,
        /// Final nonlinear residual.
        residual: f64,
        /// Configured nonlinear tolerance.
        tolerance: f64,
    },
    /// The requested accelerator was substituted by another backend and the
    /// configuration did not explicitly admit host fallback.
    BackendFallback {
        requested: PcFanCfdBackend,
        actual: PcFanCfdBackend,
        reason: PcFanCfdFallbackReason,
    },
    /// The solution was produced for a different backend request than the
    /// checked evaluation configuration.
    BackendRequestMismatch {
        configured: PcFanCfdBackend,
        recorded: PcFanCfdBackend,
    },
    /// The current FEM bridge is host-assembled and therefore cannot satisfy
    /// a complete device-resident execution request.
    BackendResidencyUnavailable {
        requested: PcFanCfdBackend,
        actual: PcFanCfdBackend,
        mode: PcFanCfdExecutionMode,
    },
    /// The duplicated compatibility backend field disagrees with the
    /// structured provenance record.
    BackendProvenanceMismatch,
    /// The CFD solver produced a non-finite residual.
    NonFiniteResidual {
        /// Reported residual.
        residual: f64,
    },
    /// A CFD field does not have one value per mesh node.
    InvalidFieldShape {
        field: &'static str,
        expected: usize,
        actual: usize,
    },
    /// A mesh or solution field contains a non-finite value.
    NonFiniteField {
        field: &'static str,
        index: usize,
    },
    /// A scalar CFD observable used by the objective is non-finite.
    NonFiniteObservable {
        field: &'static str,
    },
    /// The configured relative mass-imbalance bound is not usable.
    InvalidMassBalanceTolerance {
        /// Requested relative bound.
        tolerance: f64,
    },
    /// A configured conservation tolerance is not a finite non-negative
    /// relative bound.
    InvalidConservationTolerance {
        /// Name of the rejected conservation metric.
        metric: &'static str,
        /// Requested relative bound.
        tolerance: f64,
    },
    /// A reduced-order thermal, structural, or manufacturing limit is not
    /// physically usable.
    InvalidEnvelopeParameter {
        /// Envelope family containing the rejected parameter.
        envelope: &'static str,
        /// Name of the rejected parameter.
        parameter: &'static str,
        /// Rejected value.
        value: f64,
    },
    /// The inlet/outlet mass-flow closure is not finite or exceeds its
    /// configured relative bound.
    MassImbalance {
        /// Relative imbalance between inlet and outlet mass flow.
        relative_error: f64,
        /// Configured relative bound.
        tolerance: f64,
    },
    /// A CFD control-volume momentum ledger exceeded its configured bound.
    MomentumImbalance {
        relative_error: f64,
        tolerance: f64,
    },
    /// The pressure force and recovered source loading do not close.
    PressureClosure {
        relative_error: f64,
        tolerance: f64,
    },
    /// Mechanical source power and pressure power do not close.
    EnergyImbalance {
        relative_error: f64,
        tolerance: f64,
    },
    /// A CFD path did not publish a source-power ledger.
    /// A published conservation residual does not match the primitive ledger
    /// fields from which it was computed.
    InvalidConservationLedger,
    MissingPowerLedger,
    /// The actuator-force solver published a nonphysical or non-finite
    /// loading ledger. Rotating-wall CFD has no actuator ledger and is not
    /// covered by this variant until immersed-wall tractions are recovered.
    InvalidActuatorLoading,
    /// The solved operating point shows no forward pressure gain
    /// (inlet-minus-outlet drop above [`STALLED_FLOW_DROP_TOLERANCE_PA`]):
    /// stalled or reversed flow. Rejected so the magnitude-only
    /// `pressure_rise_pa` cannot report a healthy-looking positive rise for
    /// a failed operating point.
    ReversedOperatingPoint {
        /// Reported inlet-minus-outlet pressure difference in Pa.
        pressure_drop_pa: f64,
    },
}

impl fmt::Display for PcFanCfdEvaluationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidNonlinearTolerance { tolerance } => write!(
                f,
                "PC-fan CFD nonlinear tolerance must be finite and positive, got {tolerance}"
            ),
            Self::NonConverged {
                iterations,
                residual,
                tolerance,
            } => write!(
                f,
                "PC-fan CFD did not converge after {iterations} Picard iterations: residual {residual:.3e} exceeds tolerance {tolerance:.3e} or convergence was not declared"
            ),
            Self::BackendFallback {
                requested,
                actual,
                reason,
            } => write!(
                f,
                "PC-fan CFD requested {requested:?}, ran on {actual:?}, and was not admitted: {reason:?}"
            ),
            Self::BackendRequestMismatch {
                configured,
                recorded,
            } => write!(
                f,
                "PC-fan CFD was produced for backend request {recorded:?}, but evaluation configured {configured:?}"
            ),
            Self::BackendResidencyUnavailable {
                requested,
                actual,
                mode,
            } => write!(
                f,
                "PC-fan CFD requested device residency for {requested:?}, but ran as {mode:?} on {actual:?}"
            ),
            Self::BackendProvenanceMismatch => write!(
                f,
                "PC-fan CFD backend provenance disagrees with the compatibility backend field"
            ),
            Self::NonFiniteResidual { residual } => write!(
                f,
                "PC-fan CFD reported a non-finite nonlinear residual {residual}"
            ),
            Self::InvalidFieldShape {
                field,
                expected,
                actual,
            } => write!(
                f,
                "PC-fan CFD {field} field has {actual} values; expected {expected}"
            ),
            Self::NonFiniteField { field, index } => write!(
                f,
                "PC-fan CFD {field} field contains a non-finite value at index {index}"
            ),
            Self::NonFiniteObservable { field } => {
                write!(f, "PC-fan CFD observable {field} is non-finite")
            }
            Self::InvalidMassBalanceTolerance { tolerance } => write!(
                f,
                "PC-fan CFD mass-imbalance tolerance must be finite and non-negative, got {tolerance}"
            ),
            Self::InvalidConservationTolerance { metric, tolerance } => write!(
                f,
                "PC-fan CFD {metric} conservation tolerance must be finite and non-negative, got {tolerance}"
            ),
            Self::InvalidEnvelopeParameter {
                envelope,
                parameter,
                value,
            } => write!(
                f,
                "PC-fan {envelope} parameter {parameter} is physically invalid: {value}"
            ),
            Self::MassImbalance {
                relative_error,
                tolerance,
            } => write!(
                f,
                "PC-fan CFD inlet/outlet mass imbalance {relative_error:.3e} exceeds tolerance {tolerance:.3e}"
            ),
            Self::MomentumImbalance {
                relative_error,
                tolerance,
            } => write!(
                f,
                "PC-fan CFD axial momentum imbalance {relative_error:.3e} exceeds tolerance {tolerance:.3e}"
            ),
            Self::PressureClosure {
                relative_error,
                tolerance,
            } => write!(
                f,
                "PC-fan CFD pressure-force closure {relative_error:.3e} exceeds tolerance {tolerance:.3e}"
            ),
            Self::EnergyImbalance {
                relative_error,
                tolerance,
            } => write!(
                f,
                "PC-fan CFD mechanical-power imbalance {relative_error:.3e} exceeds tolerance {tolerance:.3e}"
            ),
            Self::InvalidConservationLedger => write!(
                f,
                "PC-fan CFD conservation ledger contains stale or inconsistent derived residuals"
            ),
            Self::MissingPowerLedger => write!(
                f,
                "PC-fan CFD did not publish actuator or rotating-wall power ledger"
            ),
            Self::InvalidActuatorLoading => write!(
                f,
                "PC-fan actuator-force CFD published a nonphysical actuator loading ledger"
            ),
            Self::ReversedOperatingPoint { pressure_drop_pa } => write!(
                f,
                "PC-fan CFD operating point is stalled or reversed: inlet-minus-outlet pressure drop is {pressure_drop_pa:.3e} Pa (must be negative for forward flow)"
            ),
        }
    }
}

impl std::error::Error for PcFanCfdEvaluationError {}

/// Input/admission errors for the reduced-order fan-curve evaluator.
#[derive(Debug, Clone, PartialEq)]
pub enum PcFanFanCurveError {
    /// Fan curves currently use the closed-form blade-element model. A CFD
    /// curve needs a pressure-boundary/load sweep and must not silently fall
    /// back to the reduced model.
    UnsupportedFlowModel(PcFanFlowModel),
    InvalidRpm(f64),
    NotEnoughSamples,
    InvalidFlowFraction {
        index: usize,
        value: f64,
    },
    NonIncreasingFlowFractions {
        index: usize,
        previous: f64,
        value: f64,
    },
    TargetLengthMismatch {
        field: &'static str,
        expected: usize,
        actual: usize,
    },
    InvalidObjectiveWeight {
        field: &'static str,
        value: f64,
    },
    InvalidObjectiveTarget {
        field: &'static str,
        index: usize,
        value: f64,
    },
    OptimizerFailure(String),
    InvalidGeometry(String),
    NonFiniteModelParameter(&'static str),
}

impl fmt::Display for PcFanFanCurveError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnsupportedFlowModel(model) => write!(
                f,
                "PC-fan reduced fan curve is unavailable for flow model {model:?}"
            ),
            Self::InvalidRpm(rpm) => write!(
                f,
                "PC-fan fan-curve RPM must be finite and positive, got {rpm}"
            ),
            Self::NotEnoughSamples => {
                write!(f, "PC-fan fan curve requires at least two flow samples")
            }
            Self::InvalidFlowFraction { index, value } => write!(
                f,
                "PC-fan fan-curve flow fraction {value} at index {index} must be finite and in [0, 1]"
            ),
            Self::NonIncreasingFlowFractions {
                index,
                previous,
                value,
            } => write!(
                f,
                "PC-fan fan-curve flow fractions must increase: index {index} has {value} after {previous}"
            ),
            Self::TargetLengthMismatch {
                field,
                expected,
                actual,
            } => write!(
                f,
                "PC-fan fan-curve target `{field}` has {actual} samples; expected {expected}"
            ),
            Self::InvalidObjectiveWeight { field, value } => write!(
                f,
                "PC-fan fan-curve objective weight `{field}` must be finite and non-negative, got {value}"
            ),
            Self::InvalidObjectiveTarget {
                field,
                index,
                value,
            } => write!(
                f,
                "PC-fan fan-curve objective target `{field}` at index {index} is non-finite: {value}"
            ),
            Self::OptimizerFailure(error) => {
                write!(f, "PC-fan fan-curve optimizer failed: {error}")
            }
            Self::InvalidGeometry(error) => write!(f, "invalid PC-fan fan-curve geometry: {error}"),
            Self::NonFiniteModelParameter(parameter) => {
                write!(
                    f,
                    "PC-fan fan-curve model parameter `{parameter}` is non-finite"
                )
            }
        }
    }
}

impl std::error::Error for PcFanFanCurveError {}

/// Admission errors for the reduced-order manufacturing-tolerance ensemble.
#[derive(Debug, Clone, PartialEq)]
pub enum PcFanManufacturingEnsembleError {
    /// The ensemble intentionally does not launch a resolved-flow solve for
    /// every tolerance corner.
    UnsupportedFlowModel(PcFanFlowModel),
    InvalidTolerance {
        parameter: &'static str,
        value: f64,
    },
    InvalidGeometry(String),
    NonFiniteEvaluation {
        index: usize,
    },
}

impl fmt::Display for PcFanManufacturingEnsembleError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnsupportedFlowModel(model) => write!(
                f,
                "PC-fan manufacturing ensemble supports reduced-order flow only, got {model:?}"
            ),
            Self::InvalidTolerance { parameter, value } => write!(
                f,
                "PC-fan manufacturing tolerance `{parameter}` is invalid: {value}"
            ),
            Self::InvalidGeometry(error) => {
                write!(
                    f,
                    "invalid PC-fan manufacturing realization geometry: {error}"
                )
            }
            Self::NonFiniteEvaluation { index } => write!(
                f,
                "PC-fan manufacturing realization {index} produced a non-finite evaluation"
            ),
        }
    }
}

impl std::error::Error for PcFanManufacturingEnsembleError {}

/// Admission errors for the reduced-order manufacturing/operating-envelope
/// cross-product evaluator.
#[derive(Debug, Clone, PartialEq)]
pub enum PcFanManufacturingOperatingEnvelopeError {
    /// The cross-product intentionally does not launch a resolved-flow solve
    /// for every tolerance corner and RPM.
    UnsupportedFlowModel(PcFanFlowModel),
    InvalidTolerance {
        parameter: &'static str,
        value: f64,
    },
    InvalidRpm {
        index: usize,
        value: f64,
    },
    InvalidGeometry(String),
    NonFiniteEvaluation {
        variation_index: usize,
        rpm_index: usize,
    },
}

impl fmt::Display for PcFanManufacturingOperatingEnvelopeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnsupportedFlowModel(model) => write!(
                f,
                "PC-fan manufacturing operating envelope supports reduced-order flow only, got {model:?}"
            ),
            Self::InvalidTolerance { parameter, value } => write!(
                f,
                "PC-fan manufacturing tolerance `{parameter}` is invalid: {value}"
            ),
            Self::InvalidRpm { index, value } => write!(
                f,
                "PC-fan operating-envelope RPM at index {index} is invalid: {value}"
            ),
            Self::InvalidGeometry(error) => write!(
                f,
                "invalid PC-fan manufacturing operating-envelope geometry: {error}"
            ),
            Self::NonFiniteEvaluation {
                variation_index,
                rpm_index,
            } => write!(
                f,
                "PC-fan manufacturing realization {variation_index} at RPM index {rpm_index} produced a non-finite evaluation"
            ),
        }
    }
}

impl std::error::Error for PcFanManufacturingOperatingEnvelopeError {}

/// Smooth band penalty for the airflow constraint.
///
/// Returns zero inside `[target − band, target + band]`, and grows
/// quadratically with `weight` outside. The band half-width is
/// `weights.airflow_band_fraction · |target_airflow_cfm|`.
///
/// Formula:
/// ```text
///   margin = airflow_cfm − target
///   excess = max(|margin| − band, 0)
///   penalty = weight · excess²
/// ```
fn airflow_band_penalty(
    weights: &PcFanOptimizationWeights,
    target_cfm: f64,
    airflow_cfm: f64,
) -> f64 {
    let margin = airflow_cfm - target_cfm;
    let band = weights.airflow_band_fraction.max(0.0) * target_cfm.abs();
    let excess = (margin.abs() - band).max(0.0);
    weights.airflow_penalty_weight * excess * excess
}

/// Derivative of `airflow_band_penalty` with respect to `airflow_cfm`.
/// Zero inside the band, `2·weight·excess·sign(margin)` outside.
#[allow(dead_code)]
fn airflow_band_penalty_gradient(
    weights: &PcFanOptimizationWeights,
    target_cfm: f64,
    airflow_cfm: f64,
) -> f64 {
    let margin = airflow_cfm - target_cfm;
    let band = weights.airflow_band_fraction.max(0.0) * target_cfm.abs();
    let abs_margin = margin.abs();
    if abs_margin <= band {
        0.0
    } else {
        2.0 * weights.airflow_penalty_weight * (abs_margin - band) * margin.signum()
    }
}

impl PcFanFlowSolution {
    fn zero() -> Self {
        Self {
            airflow_cfm: 0.0,
            axial_velocity_m_s: 0.0,
            pressure_rise_pa: 0.0,
            reynolds_number: 0.0,
            mach_number: 0.0,
            disk_solidity: 0.0,
            fixed_hardware: PcFanHardwareFlowEffects::none(),
        }
    }
}

impl PcFanHardwareFlowEffects {
    fn none() -> Self {
        Self {
            inlet_open_area_m2: 0.0,
            blockage_area_m2: 0.0,
            blockage_ratio: 0.0,
            motor_blockage_ratio: 0.0,
            strut_blockage_ratio: 0.0,
            frame_lip_loss_factor: 1.0,
            support_wake_loss_factor: 1.0,
            total_loss_factor: 1.0,
        }
    }
}

impl PcFanAcousticSolution {
    fn quiet() -> Self {
        Self {
            blade_passing_frequency_hz: 0.0,
            overall_spl_db: 0.0,
            a_weighted_overall_spl_db: 0.0,
            overall_swl_db: 0.0,
            fixed_hardware_noise: PcFanHardwareNoise::quiet(),
            harmonics: Vec::new(),
            tip_leakage_spl_db: 0.0,
            trailing_edge_spl_db: 0.0,
            leading_edge_inflow_spl_db: 0.0,
        }
    }
}

impl PcFanHardwareNoise {
    fn quiet() -> Self {
        Self {
            support_wake_spl_db: 0.0,
            motor_mount_spl_db: 0.0,
            combined_spl_db: 0.0,
            support_wake_broadband_spl_db: 0.0,
            motor_mount_broadband_spl_db: 0.0,
        }
    }
}

fn thermal_solution(config: &PcFanConfig, flow: &PcFanFlowSolution) -> PcFanThermalSolution {
    let q_m3_s = flow.airflow_cfm.abs() / M3S_TO_CFM;
    let fluid_power_w = flow.pressure_rise_pa.max(0.0) * q_m3_s;
    let motor_efficiency = config.thermal.motor_efficiency.clamp(1.0e-3, 1.0);
    let shaft_power_w = fluid_power_w / 0.60;
    let electrical_power_w = shaft_power_w / motor_efficiency;
    let motor_loss_w = (electrical_power_w - shaft_power_w).max(0.0);
    let winding_temperature_c = config.thermal.ambient_temperature_c
        + motor_loss_w * config.thermal.motor_thermal_resistance_k_per_w.max(0.0);
    PcFanThermalSolution {
        fluid_power_w,
        shaft_power_w,
        electrical_power_w,
        motor_loss_w,
        winding_temperature_c,
        temperature_margin_c: config.thermal.max_winding_temperature_c - winding_temperature_c,
        power_margin_w: config.thermal.max_electrical_power_w - electrical_power_w,
    }
}

fn structural_solution(
    config: &PcFanConfig,
    design: &PcFanDesignVariables,
    flow: &PcFanFlowSolution,
    thermal: &PcFanThermalSolution,
) -> PcFanStructuralSolution {
    let hub = design.effective_hub_radius_m(config).max(1.0e-6);
    let tip = design.effective_tip_radius_m(config).max(hub + 1.0e-6);
    let span = tip - hub;
    let omega = std::f64::consts::TAU * config.operating_point.rpm.abs() / 60.0;
    let density = config.structural.material_density_kg_m3.max(1.0);
    let mean_chord = design
        .blade_sections
        .iter()
        .map(|section| section.chord_m.max(1.0e-6))
        .sum::<f64>()
        / design.blade_sections.len().max(1) as f64;
    let mean_thickness = design
        .blade_sections
        .iter()
        .map(|section| section.thickness_m.max(1.0e-6))
        .sum::<f64>()
        / design.blade_sections.len().max(1) as f64;
    let centrifugal_stress = density * omega * omega * (tip * tip - hub * hub) * 0.5;
    let bending_stress = flow.pressure_rise_pa.abs() * span * span
        / (6.0 * mean_thickness * mean_thickness).max(1.0e-12);
    let temperature_delta_c =
        (thermal.winding_temperature_c - config.thermal.ambient_temperature_c).max(0.0);
    let thermal_strain =
        config.structural.thermal_expansion_coefficient_per_k * temperature_delta_c;
    let thermal_stress_pa = config.structural.youngs_modulus_pa
        * thermal_strain
        * config.structural.thermal_constraint_factor;
    let thermal_expansion_m = thermal_strain * span;
    let maximum_stress_pa = centrifugal_stress + bending_stress + thermal_stress_pa;
    let area = (mean_chord * mean_thickness).max(1.0e-12);
    let inertia = mean_chord * mean_thickness.powi(3) / 12.0;
    let line_mass = density * area;
    let first_mode_hz = 1.875_f64.powi(2) / (2.0 * std::f64::consts::PI)
        * (config.structural.youngs_modulus_pa.max(1.0) * inertia
            / (line_mass * span.powi(4)).max(1.0e-18))
        .sqrt();
    let bpf = config.blade_count.max(1) as f64 * config.operating_point.rpm.abs() / 60.0;
    let nearest_bpf_separation_hz = (1..=config.mesh.harmonic_count.max(1))
        .map(|harmonic| (first_mode_hz - harmonic as f64 * bpf).abs())
        .fold(f64::INFINITY, f64::min);
    let blade_mass = density * area * span;
    let imbalance_g_mm = design
        .blade_path
        .iter()
        .map(|path| path.tangential_offset_m.abs())
        .sum::<f64>()
        / design.blade_path.len().max(1) as f64
        * blade_mass
        * config.blade_count.max(1) as f64
        * 1.0e6;
    let allowable_stress_pa =
        config.structural.allowable_stress_pa / config.structural.safety_factor.max(1.0);
    let fatigue_alternating_stress_pa = bending_stress;
    let fatigue_allowable_stress_pa = config.structural.fatigue_reference_stress_pa
        * (config.structural.fatigue_design_cycles / config.structural.fatigue_reference_cycles)
            .powf(config.structural.fatigue_exponent)
        / config.structural.safety_factor.max(1.0);
    PcFanStructuralSolution {
        maximum_stress_pa,
        thermal_stress_pa,
        thermal_expansion_m,
        allowable_stress_pa,
        first_mode_hz,
        nearest_bpf_separation_hz,
        stress_margin_pa: allowable_stress_pa - maximum_stress_pa,
        modal_margin_hz: nearest_bpf_separation_hz - config.structural.min_modal_separation_hz,
        imbalance_g_mm,
        imbalance_margin_g_mm: config.structural.max_imbalance_g_mm - imbalance_g_mm,
        fatigue_alternating_stress_pa,
        fatigue_allowable_stress_pa,
        fatigue_margin_pa: fatigue_allowable_stress_pa - fatigue_alternating_stress_pa,
    }
}

fn manufacturing_status(
    config: &PcFanConfig,
    design: &PcFanDesignVariables,
) -> PcFanManufacturingStatus {
    let minimum_thickness_m = design
        .blade_sections
        .iter()
        .map(|section| section.thickness_m)
        .fold(f64::INFINITY, f64::min);
    let minimum_chord_m = design
        .blade_sections
        .iter()
        .map(|section| section.chord_m)
        .fold(f64::INFINITY, f64::min);
    let maximum_twist_gradient_deg = design
        .blade_sections
        .windows(2)
        .map(|sections| (sections[1].twist_deg - sections[0].twist_deg).abs() * 4.0)
        .fold(0.0, f64::max);
    let maximum_shroud_offset_m = design
        .shroud_radius_offsets_m
        .iter()
        .map(|offset| offset.abs())
        .fold(0.0, f64::max);
    let root = design.blade_sections.first().copied().unwrap_or_default();
    let minimum_root_thickness_ratio = root.thickness_m / root.chord_m.max(1.0e-12);
    let limits = config.manufacturing;
    let margins = [
        minimum_thickness_m - limits.min_blade_thickness_m,
        minimum_chord_m - limits.min_blade_chord_m,
        limits.max_twist_gradient_deg - maximum_twist_gradient_deg,
        limits.max_shroud_offset_m - maximum_shroud_offset_m,
        minimum_root_thickness_ratio - limits.min_root_thickness_ratio,
    ];
    let minimum_margin = margins.iter().copied().fold(f64::INFINITY, f64::min);
    PcFanManufacturingStatus {
        valid: margins
            .iter()
            .all(|margin| margin.is_finite() && *margin >= 0.0),
        minimum_thickness_m,
        minimum_chord_m,
        maximum_twist_gradient_deg,
        maximum_shroud_offset_m,
        minimum_root_thickness_ratio,
        minimum_margin,
    }
}

fn normalized_violation(value: f64, scale: f64) -> f64 {
    if !value.is_finite() {
        1.0e6
    } else {
        (-value / scale.abs().max(1.0e-12)).max(0.0)
    }
}

impl PcFanEvaluation {
    pub fn from_design(
        config: &PcFanConfig,
        design: &PcFanDesignVariables,
        geometry: &PcFanGeometry,
    ) -> Self {
        let flow = solve_mean_flow(config, design, geometry);
        Self::from_flow_solution(config, design, geometry, flow)
    }

    pub fn from_flow_solution(
        config: &PcFanConfig,
        design: &PcFanDesignVariables,
        geometry: &PcFanGeometry,
        flow: PcFanFlowSolution,
    ) -> Self {
        let acoustics = evaluate_acoustics(config, design, geometry, &flow);
        Self::from_flow_solution_with_acoustics(config, design, geometry, flow, acoustics)
    }

    fn from_flow_solution_with_acoustics(
        config: &PcFanConfig,
        design: &PcFanDesignVariables,
        geometry: &PcFanGeometry,
        flow: PcFanFlowSolution,
        acoustics: PcFanAcousticSolution,
    ) -> Self {
        let collision_status = geometry.validate_blade_intersections();
        let constraint_margins = constraint_margins(config, design, geometry, flow.airflow_cfm);
        let airflow_penalty = airflow_band_penalty(
            &config.weights,
            config.operating_point.target_airflow_cfm,
            flow.airflow_cfm,
        );
        let collision_penalty = if collision_status.is_valid {
            0.0
        } else {
            config.weights.collision_penalty_weight
                * collision_status.invalid_intersection_count as f64
        };
        let acoustic_level_db = acoustics.overall_spl_db;
        let a_weighted_acoustic_level_db = acoustics.a_weighted_overall_spl_db;
        let sound_power_level_db = acoustics.overall_swl_db;
        let spectral_flatness = harmonic_spectral_flatness(&acoustics.harmonics);
        let spectral_spread_penalty =
            config.weights.spectral_spread_weight * (1.0 - spectral_flatness);
        let observer_spls_db = compute_directivity_array(config, &acoustics);
        let directivity_spread_db = std_dev_db(&observer_spls_db);
        let directivity_spread_penalty =
            config.weights.directivity_spread_weight * directivity_spread_db;
        let shroud_clearance_penalty = compute_shroud_clearance_penalty(config, geometry);
        let frame_containment_penalty = compute_frame_containment_penalty(config, geometry);
        let hub_attachment_penalty = compute_hub_attachment_penalty(config, design);
        let thermal = thermal_solution(config, &flow);
        let structural = structural_solution(config, design, &flow, &thermal);
        let manufacturing = manufacturing_status(config, design);
        let thermal_penalty = config.weights.thermal_penalty_weight.max(0.0)
            * (normalized_violation(thermal.temperature_margin_c, 10.0).powi(2)
                + normalized_violation(thermal.power_margin_w, 1.0).powi(2));
        let structural_penalty = config.weights.structural_penalty_weight.max(0.0)
            * (normalized_violation(structural.stress_margin_pa, 1.0e6).powi(2)
                + normalized_violation(structural.modal_margin_hz, 10.0).powi(2)
                + normalized_violation(structural.imbalance_margin_g_mm, 0.5).powi(2)
                + normalized_violation(structural.fatigue_margin_pa, 1.0e6).powi(2));
        let manufacturing_penalty = config.weights.manufacturing_penalty_weight.max(0.0)
            * normalized_violation(manufacturing.minimum_margin, 0.001).powi(2);

        let perceptual_powers =
            perceptual_power_spectrum(config.weights.a_weighting_enabled, &acoustics.harmonics);
        let tonality_penalty_db = sound_metrics::tonality_penalty(
            &perceptual_powers,
            config.weights.tonality_threshold_db,
        );
        let tonality_penalty =
            config.weights.tonality_penalty_weight.max(0.0) * tonality_penalty_db;

        // Objective term: A-weighted level when enabled (default), unweighted
        // otherwise. The optimiser is therefore minimising perceived loudness
        // by default; the unweighted level remains available on
        // `acoustic_level_db` for diagnostics.
        let objective_acoustic_level_db = if config.weights.a_weighting_enabled {
            a_weighted_acoustic_level_db
        } else {
            acoustic_level_db
        };

        Self {
            airflow_cfm: flow.airflow_cfm,
            acoustic_level_db,
            a_weighted_acoustic_level_db,
            tonality_penalty_db,
            tonality_penalty,
            sound_power_level_db,
            flow,
            acoustics,
            airflow_penalty,
            collision_penalty,
            collision_status,
            objective: config.weights.acoustic_weight * objective_acoustic_level_db
                + tonality_penalty
                + spectral_spread_penalty
                + directivity_spread_penalty
                + airflow_penalty
                + collision_penalty
                + shroud_clearance_penalty
                + frame_containment_penalty
                + hub_attachment_penalty
                + thermal_penalty
                + structural_penalty
                + manufacturing_penalty,
            constraint_margins,
            spectral_flatness,
            spectral_spread_penalty,
            observer_spls_db,
            directivity_spread_db,
            directivity_spread_penalty,
            // Operating-point-spread fields are populated by
            // `attach_operating_point_spread`, which the optimiser calls
            // once per design (not per RPM). The centre-RPM solve here
            // leaves them empty.
            operating_point_spls_db: Vec::new(),
            operating_point_spread_db: 0.0,
            operating_point_spread_penalty: 0.0,
            shroud_clearance_penalty,
            frame_containment_penalty,
            hub_attachment_penalty,
            thermal,
            structural,
            manufacturing,
            thermal_penalty,
            structural_penalty,
            manufacturing_penalty,
        }
    }

    pub fn geometry_only(config: &PcFanConfig, geometry: &PcFanGeometry) -> Self {
        let collision_status = geometry.validate_blade_intersections();
        let constraint_margins =
            constraint_margins(config, &PcFanDesignVariables::default(), geometry, 0.0);
        let airflow_penalty = airflow_band_penalty(
            &config.weights,
            config.operating_point.target_airflow_cfm,
            0.0,
        );
        let collision_penalty = if collision_status.is_valid {
            0.0
        } else {
            config.weights.collision_penalty_weight
                * collision_status.invalid_intersection_count as f64
        };

        let observer_count = config.observers.observer_count().max(1);
        Self {
            airflow_cfm: 0.0,
            acoustic_level_db: 0.0,
            a_weighted_acoustic_level_db: 0.0,
            tonality_penalty_db: 0.0,
            tonality_penalty: 0.0,
            sound_power_level_db: 0.0,
            flow: PcFanFlowSolution::zero(),
            acoustics: PcFanAcousticSolution::quiet(),
            airflow_penalty,
            collision_penalty,
            collision_status,
            objective: airflow_penalty + collision_penalty,
            constraint_margins,
            spectral_flatness: 1.0,
            spectral_spread_penalty: 0.0,
            observer_spls_db: vec![0.0; observer_count],
            directivity_spread_db: 0.0,
            directivity_spread_penalty: 0.0,
            operating_point_spls_db: Vec::new(),
            operating_point_spread_db: 0.0,
            operating_point_spread_penalty: 0.0,
            shroud_clearance_penalty: 0.0,
            frame_containment_penalty: 0.0,
            hub_attachment_penalty: 0.0,
            thermal: thermal_solution(config, &PcFanFlowSolution::zero()),
            structural: structural_solution(
                config,
                &PcFanDesignVariables::default(),
                &PcFanFlowSolution::zero(),
                &thermal_solution(config, &PcFanFlowSolution::zero()),
            ),
            manufacturing: manufacturing_status(config, &PcFanDesignVariables::default()),
            thermal_penalty: 0.0,
            structural_penalty: 0.0,
            manufacturing_penalty: 0.0,
        }
    }

    /// True only when the hard collision, tip-clearance, and operating-point
    /// constraints are satisfied. This contract is deliberately independent
    /// of objective penalty weights, so disabling a penalty cannot promote an
    /// invalid fan. Frame containment remains a soft enclosure-design term:
    /// the current reference geometry intentionally extends beyond its
    /// simplified inner-box proxy.
    pub fn is_feasible(&self, config: &PcFanConfig) -> bool {
        let margins = self.constraint_margins;
        let airflow_band_fraction = config.weights.airflow_band_fraction;
        let shroud_clearance_min_m = config.weights.shroud_clearance_min_m;
        airflow_band_fraction.is_finite()
            && airflow_band_fraction >= 0.0
            && shroud_clearance_min_m.is_finite()
            && shroud_clearance_min_m >= 0.0
            && self.airflow_cfm.is_finite()
            && self.objective.is_finite()
            && self.collision_status.is_valid
            && margins.invalid_intersection_count == 0
            && margins.airflow_cfm >= 0.0
            && margins.minimum_shroud_clearance_m >= shroud_clearance_min_m
            && self.thermal.temperature_margin_c.is_finite()
            && self.thermal.power_margin_w.is_finite()
            && self.thermal.temperature_margin_c >= 0.0
            && self.thermal.power_margin_w >= 0.0
            && self.structural.stress_margin_pa.is_finite()
            && self.structural.modal_margin_hz.is_finite()
            && self.structural.imbalance_margin_g_mm.is_finite()
            && self.structural.fatigue_alternating_stress_pa.is_finite()
            && self.structural.fatigue_allowable_stress_pa.is_finite()
            && self.structural.fatigue_margin_pa.is_finite()
            && self.structural.stress_margin_pa >= 0.0
            && self.structural.modal_margin_hz >= 0.0
            && self.structural.imbalance_margin_g_mm >= 0.0
            && self.structural.fatigue_margin_pa >= 0.0
            && self.manufacturing.valid
    }

    /// Non-negative aggregate violation used only to choose the most useful
    /// diagnostic result when an optimization run finds no feasible iterate.
    pub fn feasibility_violation(&self, config: &PcFanConfig) -> f64 {
        let margins = self.constraint_margins;
        let invalid = margins.invalid_intersection_count as f64;
        invalid
            + (-margins.airflow_cfm).max(0.0)
            + (config.weights.shroud_clearance_min_m.max(0.0) - margins.minimum_shroud_clearance_m)
                .max(0.0)
            + (-self.thermal.temperature_margin_c).max(0.0) / 10.0
            + (-self.thermal.power_margin_w).max(0.0)
            + (-self.structural.stress_margin_pa).max(0.0) / 1.0e6
            + (-self.structural.modal_margin_hz).max(0.0)
            + (-self.structural.imbalance_margin_g_mm).max(0.0)
            + (-self.structural.fatigue_margin_pa).max(0.0) / 1.0e6
            + if self.manufacturing.valid { 0.0 } else { 1.0 }
    }
}

fn constraint_margins(
    config: &PcFanConfig,
    design: &PcFanDesignVariables,
    geometry: &PcFanGeometry,
    airflow_cfm: f64,
) -> PcFanConstraintMargins {
    let target = config.operating_point.target_airflow_cfm;
    let airflow_band = config.weights.airflow_band_fraction.max(0.0) * target.abs();
    let mut minimum_shroud_clearance_m = f64::INFINITY;
    let mut minimum_frame_clearance_m = f64::INFINITY;
    let frame_half_side = 0.5 * geometry.frame_box.inner_side_m;
    let frame_half_depth = 0.5 * geometry.frame_box.depth_m;
    let shroud_radius = geometry.shroud.minimum_radius_m();
    let mut vertex_count = 0usize;

    for blade in &geometry.blades {
        for panel in &blade.panels {
            for vertex in &panel.vertices {
                vertex_count += 1;
                let radius = vertex.x.hypot(vertex.y);
                minimum_shroud_clearance_m = minimum_shroud_clearance_m.min(shroud_radius - radius);
                minimum_frame_clearance_m = minimum_frame_clearance_m.min(
                    (frame_half_side - vertex.x.abs())
                        .min(frame_half_side - vertex.y.abs())
                        .min(frame_half_depth - vertex.z.abs()),
                );
            }
        }
    }
    if vertex_count == 0 {
        minimum_shroud_clearance_m = f64::NEG_INFINITY;
        minimum_frame_clearance_m = f64::NEG_INFINITY;
    }

    PcFanConstraintMargins {
        airflow_cfm: airflow_band - (airflow_cfm - target).abs(),
        minimum_shroud_clearance_m,
        minimum_frame_clearance_m,
        root_tangential_attachment_error_m: design
            .blade_path
            .first()
            .map_or(f64::INFINITY, |root| root.tangential_offset_m.abs()),
        invalid_intersection_count: geometry
            .validate_blade_intersections()
            .invalid_intersection_count,
    }
}

/// Test-only re-export of `compute_shroud_clearance_penalty` so the
/// optimizer's unit tests can exercise the helper without depending on
/// the full evaluation pipeline.
#[cfg(test)]
pub(crate) fn shroud_clearance_penalty_for_test(
    config: &PcFanConfig,
    geometry: &PcFanGeometry,
) -> f64 {
    compute_shroud_clearance_penalty(config, geometry)
}

/// Quadratic penalty for blades that violate the minimum tip-shroud gap.
/// Walks every blade panel vertex, computes the radial distance from the
/// rotation axis, and penalises any vertex whose radius exceeds
/// `shroud.minimum_radius_m() − weights.shroud_clearance_min_m`.
fn compute_shroud_clearance_penalty(config: &PcFanConfig, geometry: &PcFanGeometry) -> f64 {
    let weight = config.weights.shroud_clearance_penalty_weight;
    if weight <= 0.0 {
        return 0.0;
    }
    let min_gap = config.weights.shroud_clearance_min_m.max(0.0);
    let allowed_radius = (geometry.shroud.minimum_radius_m() - min_gap).max(0.0);
    let mut total_violation_sq = 0.0;
    for blade in &geometry.blades {
        for panel in &blade.panels {
            for v in &panel.vertices {
                let radius = (v.x * v.x + v.y * v.y).sqrt();
                if radius > allowed_radius {
                    let excess = radius - allowed_radius;
                    total_violation_sq += excess * excess;
                }
            }
        }
    }
    weight * total_violation_sq
}

/// Quadratic penalty for blades that exit the inner-frame cavity.
/// Each blade vertex must satisfy `|x|, |y| ≤ frame_inner_half` and
/// `|z| ≤ depth/2`. Any axis where a vertex pokes through the inner
/// cavity wall contributes the squared overshoot to the penalty.
fn compute_frame_containment_penalty(config: &PcFanConfig, geometry: &PcFanGeometry) -> f64 {
    let weight = config.weights.frame_containment_penalty_weight;
    if weight <= 0.0 {
        return 0.0;
    }
    let frame_inner_half = 0.5 * geometry.frame_box.inner_side_m;
    let depth_half = 0.5 * geometry.frame_box.depth_m;
    let mut total_violation_sq = 0.0;
    for blade in &geometry.blades {
        for panel in &blade.panels {
            for v in &panel.vertices {
                let dx = (v.x.abs() - frame_inner_half).max(0.0);
                let dy = (v.y.abs() - frame_inner_half).max(0.0);
                let dz = (v.z.abs() - depth_half).max(0.0);
                total_violation_sq += dx * dx + dy * dy + dz * dz;
            }
        }
    }
    weight * total_violation_sq
}

/// Quadratic penalty for the hub-attached blade-path control point
/// having any in-plane tangential lean. Operates directly on the
/// design vector (`blade_path[0].tangential_offset_m`) so:
///
/// - The penalty is exactly zero on the baseline design (which has
///   `blade_path[0].tangential_offset_m == 0`), regardless of all
///   other design variables.
/// - The gradient w.r.t. that single design variable is closed-form
///   (linear in the offset), giving the optimiser the cleanest
///   possible signal: "leave the root attached to the hub."
///
/// Only `tangential_offset_m` is penalised because only it causes the
/// detachment bug — `polar_point` adds it as a Cartesian-orthogonal
/// term to the radial vector, so a non-zero root tangential lean
/// turns `R_hub` into `√(R_hub² + tang²) > R_hub`, visually
/// disconnecting the blade root from the hub. `axial_offset_m` is a
/// pure z-shift that leaves the radial extent untouched, and the
/// other path control points (i ≥ 1) are legitimate sweep degrees of
/// freedom.
///
/// This penalty is the safety net: the primary enforcement is the
/// `(0, 0)` bound pin in `optimizer::design_bounds` plus the hard
/// reset in `vector_to_design`. Both together guarantee the root
/// stays welded to the hub even under accumulated FP drift or future
/// design-vector additions that might bypass the bound.
fn compute_hub_attachment_penalty(config: &PcFanConfig, design: &PcFanDesignVariables) -> f64 {
    let weight = config.weights.hub_attachment_penalty_weight;
    if weight <= 0.0 || design.blade_path.is_empty() {
        return 0.0;
    }
    let tang = design.blade_path[0].tangential_offset_m;
    weight * tang * tang
}

/// Re-evaluate the design at each off-design RPM, then attach the std-dev
/// (in dB) of overall SPL across operating points to the supplied
/// `evaluation` and update `objective` accordingly. Skips the work and
/// returns `evaluation` unchanged when the spread weight is zero or the
/// RPM list has fewer than two distinct entries.
///
/// This is split out from `PcFanEvaluation::from_flow_solution` because
/// each RPM is a full nested evaluation (CFD + acoustics + directivity),
/// and we want the centre-RPM fast path to remain inexpensive.
pub fn attach_operating_point_spread(
    config: &PcFanConfig,
    design: &PcFanDesignVariables,
    geometry: &PcFanGeometry,
    mut evaluation: PcFanEvaluation,
) -> PcFanEvaluation {
    let weight = config.weights.operating_spread_weight;
    let rpms = &config.operating_point_spread.rpms;
    if weight <= 0.0 || rpms.len() < 2 {
        return evaluation;
    }
    let mut spls = Vec::with_capacity(rpms.len());
    for &rpm in rpms {
        let mut sub_config = config.clone();
        sub_config.operating_point.rpm = rpm;
        // Disable nested operating-point solves to prevent recursion.
        sub_config.weights.operating_spread_weight = 0.0;
        let sub_eval = PcFanEvaluation::from_design(&sub_config, design, geometry);
        spls.push(sub_eval.acoustic_level_db);
    }
    let spread_db = std_dev_db(&spls);
    let penalty = weight * spread_db;
    evaluation.operating_point_spls_db = spls;
    evaluation.operating_point_spread_db = spread_db;
    evaluation.operating_point_spread_penalty = penalty;
    evaluation.objective += penalty;
    evaluation
}

/// Evaluate a fixed design at every configured operating-envelope RPM.
///
/// Samples are sorted by RPM and the nested spread objective is disabled for
/// each point. This makes the result suitable for independent fan-curve and
/// feasibility gates rather than reporting only a weighted SPL spread.
pub fn evaluate_pc_fan_operating_envelope(
    config: &PcFanConfig,
    design: &PcFanDesignVariables,
) -> PcFanOperatingEnvelope {
    let mut rpms = config.operating_point_spread.rpms.clone();
    if rpms.is_empty() {
        rpms.push(config.operating_point.rpm);
    }
    rpms.sort_by(|a, b| a.total_cmp(b));
    rpms.dedup_by(|a, b| *a == *b);

    let geometry = config.build_geometry(design);
    let mut points = Vec::with_capacity(rpms.len());
    let mut finite = true;
    for rpm in rpms {
        let mut point_config = config.clone();
        point_config.operating_point.rpm = rpm;
        point_config.weights.operating_spread_weight = 0.0;
        let evaluation = PcFanEvaluation::from_design(&point_config, design, &geometry);
        let point = PcFanOperatingEnvelopePoint {
            rpm,
            airflow_cfm: evaluation.airflow_cfm,
            pressure_rise_pa: evaluation.flow.pressure_rise_pa,
            acoustic_level_db: evaluation.acoustic_level_db,
            sound_power_level_db: evaluation.sound_power_level_db,
            temperature_margin_c: evaluation.thermal.temperature_margin_c,
            stress_margin_pa: evaluation.structural.stress_margin_pa,
            modal_margin_hz: evaluation.structural.modal_margin_hz,
            feasible: evaluation.is_feasible(&point_config),
        };
        finite &= rpm.is_finite()
            && point.airflow_cfm.is_finite()
            && point.pressure_rise_pa.is_finite()
            && point.acoustic_level_db.is_finite()
            && point.sound_power_level_db.is_finite()
            && point.temperature_margin_c.is_finite()
            && point.stress_margin_pa.is_finite()
            && point.modal_margin_hz.is_finite();
        points.push(point);
    }

    let airflow_strictly_increases_with_rpm = points
        .windows(2)
        .all(|pair| pair[1].rpm > pair[0].rpm && pair[1].airflow_cfm > pair[0].airflow_cfm);
    let pressure_strictly_increases_with_rpm = points.windows(2).all(|pair| {
        pair[1].rpm > pair[0].rpm && pair[1].pressure_rise_pa > pair[0].pressure_rise_pa
    });
    let all_points_feasible = !points.is_empty() && points.iter().all(|point| point.feasible);

    PcFanOperatingEnvelope {
        points,
        finite,
        airflow_strictly_increases_with_rpm,
        pressure_strictly_increases_with_rpm,
        all_points_feasible,
    }
}

/// Evaluate every deterministic manufacturing corner at every configured
/// operating-envelope RPM.
///
/// The nominal realization is first, followed by the same deterministic
/// tolerance-corner ordering used by [`evaluate_pc_fan_manufacturing_ensemble`].
/// Within each realization, points are sorted by ascending RPM.  The
/// evaluator is intentionally reduced-order and fail-closed for CFD configs;
/// it must not turn a tolerance sweep into unvalidated resolved-flow evidence.
pub fn evaluate_pc_fan_manufacturing_operating_envelope(
    config: &PcFanConfig,
    design: &PcFanDesignVariables,
) -> Result<PcFanManufacturingOperatingEnvelope, PcFanManufacturingOperatingEnvelopeError> {
    if config.cfd.flow_model != PcFanFlowModel::ReducedOrder {
        return Err(
            PcFanManufacturingOperatingEnvelopeError::UnsupportedFlowModel(config.cfd.flow_model),
        );
    }
    if let Some((parameter, value)) =
        invalid_manufacturing_tolerance(config.manufacturing.tolerance)
    {
        return Err(PcFanManufacturingOperatingEnvelopeError::InvalidTolerance {
            parameter,
            value,
        });
    }
    config.validate_geometry_config().map_err(|error| {
        PcFanManufacturingOperatingEnvelopeError::InvalidGeometry(error.to_string())
    })?;

    let mut rpms = config.operating_point_spread.rpms.clone();
    if rpms.is_empty() {
        rpms.push(config.operating_point.rpm);
    }
    for (index, rpm) in rpms.iter().copied().enumerate() {
        if !rpm.is_finite() || rpm <= 0.0 {
            return Err(PcFanManufacturingOperatingEnvelopeError::InvalidRpm { index, value: rpm });
        }
    }
    rpms.sort_by(|a, b| a.total_cmp(b));
    rpms.dedup_by(|a, b| *a == *b);

    let variations = manufacturing_corner_variations(config.manufacturing.tolerance);
    let variation_count = variations.len();
    let rpm_count = rpms.len();
    let mut points = Vec::with_capacity(variation_count * rpm_count);
    let mut finite = true;
    let mut all_points_feasible = true;
    let mut all_variations_monotonic = true;
    let mut objective_sum = 0.0;
    let mut worst_objective = f64::NEG_INFINITY;

    for (variation_index, variation) in variations.into_iter().enumerate() {
        let realization = variation.apply(design);
        let geometry = config.try_build_geometry(&realization).map_err(|error| {
            PcFanManufacturingOperatingEnvelopeError::InvalidGeometry(error.to_string())
        })?;
        let mut local_points = Vec::with_capacity(rpm_count);

        for (rpm_index, rpm) in rpms.iter().copied().enumerate() {
            let mut point_config = config.clone();
            point_config.operating_point.rpm = rpm;
            point_config.weights.operating_spread_weight = 0.0;
            let evaluation = PcFanEvaluation::from_design(&point_config, &realization, &geometry);
            let point = PcFanOperatingEnvelopePoint {
                rpm,
                airflow_cfm: evaluation.airflow_cfm,
                pressure_rise_pa: evaluation.flow.pressure_rise_pa,
                acoustic_level_db: evaluation.acoustic_level_db,
                sound_power_level_db: evaluation.sound_power_level_db,
                temperature_margin_c: evaluation.thermal.temperature_margin_c,
                stress_margin_pa: evaluation.structural.stress_margin_pa,
                modal_margin_hz: evaluation.structural.modal_margin_hz,
                feasible: evaluation.is_feasible(&point_config),
            };
            let values = [
                evaluation.objective,
                point.airflow_cfm,
                point.pressure_rise_pa,
                point.acoustic_level_db,
                point.sound_power_level_db,
                point.temperature_margin_c,
                point.stress_margin_pa,
                point.modal_margin_hz,
            ];
            if values.iter().any(|value| !value.is_finite()) {
                return Err(
                    PcFanManufacturingOperatingEnvelopeError::NonFiniteEvaluation {
                        variation_index,
                        rpm_index,
                    },
                );
            }
            finite &= evaluation.objective.is_finite();
            all_points_feasible &= point.feasible;
            objective_sum += evaluation.objective;
            worst_objective = worst_objective.max(evaluation.objective);
            local_points.push((point, evaluation.objective));
        }

        all_variations_monotonic &= local_points.windows(2).all(|pair| {
            pair[1].0.rpm > pair[0].0.rpm
                && pair[1].0.airflow_cfm > pair[0].0.airflow_cfm
                && pair[1].0.pressure_rise_pa > pair[0].0.pressure_rise_pa
        });
        for (point, objective) in local_points {
            points.push(PcFanManufacturingOperatingEnvelopePoint {
                variation,
                rpm: point.rpm,
                design_variables: realization.clone(),
                point,
                objective,
            });
        }
    }

    let point_count = points.len();
    Ok(PcFanManufacturingOperatingEnvelope {
        points,
        variation_count,
        rpm_count,
        finite,
        all_points_feasible,
        all_variations_monotonic,
        worst_objective,
        mean_objective: objective_sum / point_count as f64,
    })
}

fn reduced_fan_curve_operating_point(
    config: &PcFanConfig,
    geometry: &PcFanGeometry,
    fixed_hardware: PcFanHardwareFlowEffects,
    free_airflow_cfm: f64,
    shutoff_pressure_pa: f64,
) -> Option<PcFanFanCurveOperatingPoint> {
    let residual_at = |flow_fraction: f64| {
        let airflow_cfm = free_airflow_cfm * flow_fraction;
        let fan_pressure_rise_pa =
            (shutoff_pressure_pa * (1.0 - flow_fraction * flow_fraction)).max(0.0);
        let system_pressure_rise_pa = installed_system_pressure_loss_pa(
            config,
            geometry,
            airflow_cfm / M3S_TO_CFM,
            fixed_hardware,
        );
        (
            airflow_cfm,
            fan_pressure_rise_pa,
            system_pressure_rise_pa,
            fan_pressure_rise_pa - system_pressure_rise_pa,
        )
    };

    let pressure_tolerance = 1.0e-9 * shutoff_pressure_pa.abs().max(1.0);
    let mut lower = 0.0;
    let mut upper = 1.0;
    let mut lower_values = residual_at(lower);
    let upper_values = residual_at(upper);
    if !lower_values.3.is_finite() || !upper_values.3.is_finite() {
        return None;
    }
    if lower_values.3.abs() <= pressure_tolerance {
        return Some(PcFanFanCurveOperatingPoint {
            flow_fraction: lower,
            airflow_cfm: lower_values.0,
            fan_pressure_rise_pa: lower_values.1,
            system_pressure_rise_pa: lower_values.2,
            residual_pa: lower_values.3,
            iterations: 0,
            converged: true,
        });
    }
    if upper_values.3.abs() <= pressure_tolerance {
        return Some(PcFanFanCurveOperatingPoint {
            flow_fraction: upper,
            airflow_cfm: upper_values.0,
            fan_pressure_rise_pa: upper_values.1,
            system_pressure_rise_pa: upper_values.2,
            residual_pa: upper_values.3,
            iterations: 0,
            converged: true,
        });
    }
    if !((lower_values.3 > 0.0 && upper_values.3 < 0.0)
        || (lower_values.3 < 0.0 && upper_values.3 > 0.0))
    {
        return None;
    }

    for iteration in 1..=80 {
        let midpoint = 0.5 * (lower + upper);
        let midpoint_values = residual_at(midpoint);
        if !midpoint_values.3.is_finite() {
            return None;
        }
        if midpoint_values.3.abs() <= pressure_tolerance || (upper - lower) <= 1.0e-12 {
            return Some(PcFanFanCurveOperatingPoint {
                flow_fraction: midpoint,
                airflow_cfm: midpoint_values.0,
                fan_pressure_rise_pa: midpoint_values.1,
                system_pressure_rise_pa: midpoint_values.2,
                residual_pa: midpoint_values.3,
                iterations: iteration,
                converged: midpoint_values.3.abs() <= pressure_tolerance,
            });
        }
        if (lower_values.3 > 0.0 && midpoint_values.3 > 0.0)
            || (lower_values.3 < 0.0 && midpoint_values.3 < 0.0)
        {
            lower = midpoint;
            lower_values = midpoint_values;
        } else {
            upper = midpoint;
        }
    }
    None
}

/// Evaluate a reduced-order static fan curve at one RPM.
///
/// The curve uses the model's blade-element free-airflow and shutoff-pressure
/// endpoints, with the conventional quadratic pressure interpolation
/// `Δp_fan = Δp_shutoff (1 - (Q/Q_free)^2)`. Installed enclosure loss is
/// reported separately. This is a deterministic reduced-order curve contract;
/// resolved CFD requires a pressure/load sweep and is rejected explicitly.
pub fn evaluate_pc_fan_fan_curve(
    config: &PcFanConfig,
    design: &PcFanDesignVariables,
    rpm: f64,
    flow_fractions: &[f64],
) -> Result<PcFanFanCurve, PcFanFanCurveError> {
    if config.cfd.flow_model != PcFanFlowModel::ReducedOrder {
        return Err(PcFanFanCurveError::UnsupportedFlowModel(
            config.cfd.flow_model,
        ));
    }
    if !rpm.is_finite() || rpm <= 0.0 {
        return Err(PcFanFanCurveError::InvalidRpm(rpm));
    }
    if flow_fractions.len() < 2 {
        return Err(PcFanFanCurveError::NotEnoughSamples);
    }
    for (index, &fraction) in flow_fractions.iter().enumerate() {
        if !fraction.is_finite() || !(0.0..=1.0).contains(&fraction) {
            return Err(PcFanFanCurveError::InvalidFlowFraction {
                index,
                value: fraction,
            });
        }
        if let Some(&previous) = index
            .checked_sub(1)
            .and_then(|previous| flow_fractions.get(previous))
            && fraction <= previous
        {
            return Err(PcFanFanCurveError::NonIncreasingFlowFractions {
                index,
                previous,
                value: fraction,
            });
        }
    }

    let geometry = config
        .try_build_geometry(design)
        .map_err(|error| PcFanFanCurveError::InvalidGeometry(error.to_string()))?;
    let mut curve_config = config.clone();
    curve_config.operating_point.rpm = rpm;
    let summary = blade_element_flow_summary(&curve_config, design, &geometry);
    let free_airflow_cfm = summary.free_airflow_m3_s * M3S_TO_CFM;
    if !free_airflow_cfm.is_finite() || free_airflow_cfm <= 0.0 {
        return Err(PcFanFanCurveError::NonFiniteModelParameter(
            "free_airflow_cfm",
        ));
    }
    if !summary.shutoff_pressure_pa.is_finite() || summary.shutoff_pressure_pa < 0.0 {
        return Err(PcFanFanCurveError::NonFiniteModelParameter(
            "shutoff_pressure_pa",
        ));
    }

    let fixed_hardware = fixed_hardware_flow_effects(&curve_config, &geometry);
    let operating_point = reduced_fan_curve_operating_point(
        &curve_config,
        &geometry,
        fixed_hardware,
        free_airflow_cfm,
        summary.shutoff_pressure_pa,
    );

    let mut points = Vec::with_capacity(flow_fractions.len());
    for &flow_fraction in flow_fractions {
        let airflow_cfm = free_airflow_cfm * flow_fraction;
        let system_pressure_rise_pa = installed_system_pressure_loss_pa(
            &curve_config,
            &geometry,
            airflow_cfm / M3S_TO_CFM,
            fixed_hardware,
        );
        let fan_pressure_rise_pa =
            (summary.shutoff_pressure_pa * (1.0 - flow_fraction * flow_fraction)).max(0.0);
        let mut flow = flow_from_airflow_cfm(&curve_config, design, &geometry, airflow_cfm);
        flow.pressure_rise_pa = fan_pressure_rise_pa;
        let mut point_config = curve_config.clone();
        point_config.operating_point.target_airflow_cfm = airflow_cfm;
        let evaluation =
            PcFanEvaluation::from_flow_solution(&point_config, design, &geometry, flow);
        let thermal = evaluation.thermal;
        let efficiency = if thermal.electrical_power_w > 0.0 {
            thermal.fluid_power_w / thermal.electrical_power_w
        } else {
            0.0
        };
        let feasible = evaluation.is_feasible(&point_config);
        points.push(PcFanFanCurvePoint {
            rpm,
            flow_fraction,
            airflow_cfm,
            fan_pressure_rise_pa,
            system_pressure_rise_pa,
            acoustic_level_db: evaluation.acoustic_level_db,
            sound_power_level_db: evaluation.sound_power_level_db,
            fluid_power_w: thermal.fluid_power_w,
            electrical_power_w: thermal.electrical_power_w,
            efficiency,
            feasible,
        });
    }

    let finite = points.iter().all(|point| {
        [
            point.rpm,
            point.flow_fraction,
            point.airflow_cfm,
            point.fan_pressure_rise_pa,
            point.system_pressure_rise_pa,
            point.acoustic_level_db,
            point.sound_power_level_db,
            point.fluid_power_w,
            point.electrical_power_w,
            point.efficiency,
        ]
        .into_iter()
        .all(f64::is_finite)
    });
    let airflow_strictly_increases = points
        .windows(2)
        .all(|pair| pair[1].airflow_cfm > pair[0].airflow_cfm);
    let fan_pressure_strictly_decreases = points
        .windows(2)
        .all(|pair| pair[1].fan_pressure_rise_pa < pair[0].fan_pressure_rise_pa);
    let efficiencies_bounded = points
        .iter()
        .all(|point| point.efficiency >= -1.0e-12 && point.efficiency <= 1.0 + 1.0e-12);
    let all_points_feasible = !points.is_empty() && points.iter().all(|point| point.feasible);

    Ok(PcFanFanCurve {
        points,
        free_airflow_cfm,
        shutoff_pressure_pa: summary.shutoff_pressure_pa,
        operating_point,
        finite,
        airflow_strictly_increases,
        fan_pressure_strictly_decreases,
        efficiencies_bounded,
        all_points_feasible,
    })
}

/// Score a reduced-order fan curve against explicit flow/pressure targets.
///
/// The normalized squared error is averaged over all active target terms.
/// Zero weights disable a term; at least one weight must be positive. A curve
/// that is finite but violates its ordering or hard point-feasibility contract
/// receives an infinite score and `feasible = false` so optimizers cannot
/// promote it by accident.
pub fn evaluate_pc_fan_fan_curve_objective(
    config: &PcFanConfig,
    design: &PcFanDesignVariables,
    objective_config: &PcFanFanCurveObjectiveConfig,
) -> Result<PcFanFanCurveObjective, PcFanFanCurveError> {
    let sample_count = objective_config.flow_fractions.len();
    if objective_config.target_airflow_cfm.len() != sample_count {
        return Err(PcFanFanCurveError::TargetLengthMismatch {
            field: "target_airflow_cfm",
            expected: sample_count,
            actual: objective_config.target_airflow_cfm.len(),
        });
    }
    if objective_config.target_pressure_rise_pa.len() != sample_count {
        return Err(PcFanFanCurveError::TargetLengthMismatch {
            field: "target_pressure_rise_pa",
            expected: sample_count,
            actual: objective_config.target_pressure_rise_pa.len(),
        });
    }
    for (field, values) in [
        (
            "target_airflow_cfm",
            objective_config.target_airflow_cfm.as_slice(),
        ),
        (
            "target_pressure_rise_pa",
            objective_config.target_pressure_rise_pa.as_slice(),
        ),
    ] {
        for (index, &value) in values.iter().enumerate() {
            if !value.is_finite() {
                return Err(PcFanFanCurveError::InvalidObjectiveTarget {
                    field,
                    index,
                    value,
                });
            }
        }
    }
    for (field, values) in [
        (
            "target_acoustic_level_db",
            objective_config.target_acoustic_level_db.as_deref(),
        ),
        (
            "target_efficiency",
            objective_config.target_efficiency.as_deref(),
        ),
    ] {
        if let Some(values) = values {
            if values.len() != sample_count {
                return Err(PcFanFanCurveError::TargetLengthMismatch {
                    field,
                    expected: sample_count,
                    actual: values.len(),
                });
            }
            for (index, &value) in values.iter().enumerate() {
                if !value.is_finite() {
                    return Err(PcFanFanCurveError::InvalidObjectiveTarget {
                        field,
                        index,
                        value,
                    });
                }
            }
        }
    }

    let weights = [
        ("airflow_weight", objective_config.airflow_weight),
        ("pressure_weight", objective_config.pressure_weight),
        ("acoustic_weight", objective_config.acoustic_weight),
        ("efficiency_weight", objective_config.efficiency_weight),
    ];
    for (field, value) in weights {
        if !value.is_finite() || value < 0.0 {
            return Err(PcFanFanCurveError::InvalidObjectiveWeight { field, value });
        }
    }
    if weights.iter().all(|(_, weight)| *weight == 0.0) {
        return Err(PcFanFanCurveError::InvalidObjectiveWeight {
            field: "all",
            value: 0.0,
        });
    }

    let curve = evaluate_pc_fan_fan_curve(
        config,
        design,
        objective_config.rpm,
        &objective_config.flow_fractions,
    )?;
    let airflow_scale = objective_config
        .target_airflow_cfm
        .iter()
        .copied()
        .map(f64::abs)
        .fold(curve.free_airflow_cfm.abs().max(1.0), f64::max);
    let pressure_scale = objective_config
        .target_pressure_rise_pa
        .iter()
        .copied()
        .map(f64::abs)
        .fold(curve.shutoff_pressure_pa.abs().max(1.0), f64::max);
    let airflow_rmse_cfm = rmse(
        curve
            .points
            .iter()
            .map(|point| point.airflow_cfm)
            .zip(objective_config.target_airflow_cfm.iter().copied())
            .map(|(actual, target)| (actual - target) / airflow_scale),
    ) * airflow_scale;
    let pressure_rmse_pa = rmse(
        curve
            .points
            .iter()
            .map(|point| point.fan_pressure_rise_pa)
            .zip(objective_config.target_pressure_rise_pa.iter().copied())
            .map(|(actual, target)| (actual - target) / pressure_scale),
    ) * pressure_scale;
    let acoustic_rmse_db = objective_config
        .target_acoustic_level_db
        .as_ref()
        .map(|target| {
            rmse(
                curve
                    .points
                    .iter()
                    .map(|point| point.acoustic_level_db)
                    .zip(target.iter().copied())
                    .map(|(actual, target)| actual - target),
            )
        });
    let efficiency_rmse = objective_config.target_efficiency.as_ref().map(|target| {
        rmse(
            curve
                .points
                .iter()
                .map(|point| point.efficiency)
                .zip(target.iter().copied())
                .map(|(actual, target)| actual - target),
        )
    });

    let score = objective_config.airflow_weight * (airflow_rmse_cfm / airflow_scale).powi(2)
        + objective_config.pressure_weight * (pressure_rmse_pa / pressure_scale).powi(2)
        + objective_config.acoustic_weight * acoustic_rmse_db.unwrap_or(0.0).powi(2)
        + objective_config.efficiency_weight * efficiency_rmse.unwrap_or(0.0).powi(2);
    let feasible = curve.is_valid() && score.is_finite();
    Ok(PcFanFanCurveObjective {
        score: if feasible { score } else { f64::INFINITY },
        airflow_rmse_cfm,
        pressure_rmse_pa,
        acoustic_rmse_db,
        efficiency_rmse,
        curve,
        feasible,
    })
}

pub(crate) fn invalid_manufacturing_tolerance(
    tolerance: PcFanManufacturingTolerance,
) -> Option<(&'static str, f64)> {
    [
        (
            "blade_thickness_fraction",
            tolerance.blade_thickness_fraction,
        ),
        ("blade_chord_fraction", tolerance.blade_chord_fraction),
        ("twist_deg", tolerance.twist_deg),
        ("shroud_offset_m", tolerance.shroud_offset_m),
    ]
    .into_iter()
    .find_map(|(parameter, value)| {
        let valid = match parameter {
            "blade_thickness_fraction" | "blade_chord_fraction" => {
                value.is_finite() && (0.0..1.0).contains(&value)
            }
            "twist_deg" | "shroud_offset_m" => value.is_finite() && value >= 0.0,
            _ => false,
        };
        (!valid).then_some((parameter, value))
    })
}

pub(crate) fn manufacturing_corner_variations(
    tolerance: PcFanManufacturingTolerance,
) -> Vec<PcFanManufacturingVariation> {
    let nominal = PcFanManufacturingVariation::nominal();
    let axes = [
        (0usize, tolerance.blade_thickness_fraction),
        (1usize, tolerance.blade_chord_fraction),
        (2usize, tolerance.twist_deg),
        (3usize, tolerance.shroud_offset_m),
    ];
    let active = axes
        .into_iter()
        .filter(|(_, magnitude)| *magnitude > 0.0)
        .collect::<Vec<_>>();
    if active.is_empty() {
        return vec![nominal];
    }

    let mut variations = Vec::with_capacity(1usize + (1usize << active.len()));
    variations.push(nominal);
    for mask in 0..(1usize << active.len()) {
        let mut variation = nominal;
        for (bit, (axis, magnitude)) in active.iter().enumerate() {
            let value = if mask & (1usize << bit) == 0 {
                -*magnitude
            } else {
                *magnitude
            };
            match axis {
                0 => variation.blade_thickness_fraction = value,
                1 => variation.blade_chord_fraction = value,
                2 => variation.twist_deg = value,
                3 => variation.shroud_offset_m = value,
                _ => unreachable!("manufacturing axis index is bounded"),
            }
        }
        variations.push(variation);
    }
    variations
}

/// Evaluate a design at the nominal point and deterministic manufacturing
/// tolerance corners.
///
/// This is deliberately a reduced-order admission contract. It reports
/// whether every requested corner is finite and feasible, but it does not
/// claim resolved-flow, structural-FEM, or measured-product validation.
pub fn evaluate_pc_fan_manufacturing_ensemble(
    config: &PcFanConfig,
    design: &PcFanDesignVariables,
) -> Result<PcFanManufacturingEnsemble, PcFanManufacturingEnsembleError> {
    if config.cfd.flow_model != PcFanFlowModel::ReducedOrder {
        return Err(PcFanManufacturingEnsembleError::UnsupportedFlowModel(
            config.cfd.flow_model,
        ));
    }
    if let Some((parameter, value)) =
        invalid_manufacturing_tolerance(config.manufacturing.tolerance)
    {
        return Err(PcFanManufacturingEnsembleError::InvalidTolerance { parameter, value });
    }
    config
        .validate_geometry_config()
        .map_err(|error| PcFanManufacturingEnsembleError::InvalidGeometry(error.to_string()))?;

    let variations = manufacturing_corner_variations(config.manufacturing.tolerance);
    let mut points = Vec::with_capacity(variations.len());
    let mut worst_objective = f64::NEG_INFINITY;
    let mut objective_sum = 0.0;
    let mut worst_feasibility_violation = 0.0_f64;

    for (index, variation) in variations.into_iter().enumerate() {
        let realization = variation.apply(design);
        let geometry = config
            .try_build_geometry(&realization)
            .map_err(|error| PcFanManufacturingEnsembleError::InvalidGeometry(error.to_string()))?;
        let evaluation = PcFanEvaluation::from_design(config, &realization, &geometry);
        let finite_values = [
            evaluation.objective,
            evaluation.airflow_cfm,
            evaluation.flow.pressure_rise_pa,
            evaluation.thermal.temperature_margin_c,
            evaluation.thermal.power_margin_w,
            evaluation.structural.stress_margin_pa,
            evaluation.structural.modal_margin_hz,
            evaluation.structural.imbalance_margin_g_mm,
            evaluation.feasibility_violation(config),
        ];
        if finite_values.iter().any(|value| !value.is_finite()) {
            return Err(PcFanManufacturingEnsembleError::NonFiniteEvaluation { index });
        }
        worst_objective = worst_objective.max(evaluation.objective);
        objective_sum += evaluation.objective;
        worst_feasibility_violation =
            worst_feasibility_violation.max(evaluation.feasibility_violation(config));
        points.push(PcFanManufacturingEnsemblePoint {
            variation,
            design_variables: realization,
            evaluation,
        });
    }

    let finite = points.iter().all(|point| {
        point.evaluation.objective.is_finite()
            && point.evaluation.feasibility_violation(config).is_finite()
    });
    let all_points_feasible = points
        .iter()
        .all(|point| point.evaluation.is_feasible(config));
    Ok(PcFanManufacturingEnsemble {
        nominal_index: 0,
        mean_objective: objective_sum / points.len() as f64,
        worst_objective,
        worst_feasibility_violation,
        finite,
        all_points_feasible,
        points,
    })
}

fn rmse(values: impl Iterator<Item = f64>) -> f64 {
    let values = values.collect::<Vec<_>>();
    if values.is_empty() {
        return 0.0;
    }
    (values.iter().map(|value| value * value).sum::<f64>() / values.len() as f64).sqrt()
}

/// Spectral flatness of the per-harmonic acoustic powers.
/// Returns 1.0 for an empty harmonic vector (no spectral structure to assess);
/// otherwise delegates to `sound_metrics::spectral_flatness` over the squared
/// pressures (proportional to power).
fn harmonic_spectral_flatness(harmonics: &[PcFanAcousticHarmonic]) -> f64 {
    if harmonics.is_empty() {
        return 1.0;
    }
    let powers: Vec<f64> = harmonics
        .iter()
        .map(|h| h.pressure_pa * h.pressure_pa)
        .collect();
    sound_metrics::spectral_flatness(&powers)
}

/// Per-harmonic axial-vs-tangential dipole weighting. Returns the
/// `(axial_weight, tangential_weight)` pair (each in [0,1], summing to 1).
///
/// Tonal blade-passing radiation is dominated by loading noise (compact
/// dipole perpendicular to the rotation axis) at low harmonics and shifts
/// toward thickness/quadrupole noise (more axial) at high harmonics. This
/// gives a smooth blend used in the analytical directivity factor.
fn dipole_weights_for_harmonic(harmonic: usize) -> (f64, f64) {
    // h=1 → mostly tangential (axial=0.30, tang=0.70).
    // h→∞ → 50/50 split (axial=0.50, tang=0.50).
    let axial = 0.50 - 0.20 * (-((harmonic as f64 - 1.0) / 4.0)).exp();
    let tangential = 1.0 - axial;
    (axial.clamp(0.0, 1.0), tangential.clamp(0.0, 1.0))
}

/// Compute SPL at one observer given the harmonic content.
///
/// Geometry: the rotation axis is `+Z`; `theta_polar_rad` is the angle from
/// the axis (0 = on-axis, π/2 = in fan plane); `phi_azimuth_rad` is azimuth
/// around the axis. Strut-induced azimuthal modulation breaks axisymmetry,
/// and a downstream-bias factor breaks fore/aft symmetry — together these
/// turn the otherwise-symmetric `cos²θ` / `sin²θ` model into a polar
/// pattern that responds to twist, lean and sweep.
///
/// Per-harmonic `pressure_pa` is RMS pressure at
/// `enclosure.reference_distance_m`. Free-field/duct observers receive the
/// `r_ref/r_observer` pressure-spreading factor here. The directivity pattern
/// is normalized to unit spherical mean so it preserves radiated power.
fn observer_spl_db(
    config: &PcFanConfig,
    acoustics: &PcFanAcousticSolution,
    theta_polar_rad: f64,
    phi_azimuth_rad: f64,
) -> f64 {
    if acoustics.harmonics.is_empty() {
        return 0.0;
    }
    let cos_theta = theta_polar_rad.cos();

    let reference_distance_m = acoustic_reference_distance_m(config);
    let distance_gain = if config.enclosure.mode.is_rigid_cavity() {
        // Rigid-cavity harmonics are already evaluated at the configured
        // listener position, not on a free-field observer sphere.
        1.0
    } else {
        reference_distance_m / config.observers.distance_m.max(1.0e-6)
    };

    let mut sum_pressure_sq = 0.0;
    for h in &acoustics.harmonics {
        let directivity_sq = directivity_squared_gain(h.harmonic, cos_theta, phi_azimuth_rad);
        let observer_pressure = h.pressure_pa * distance_gain;
        sum_pressure_sq += observer_pressure * observer_pressure * directivity_sq;
    }
    if sum_pressure_sq <= 0.0 {
        return 0.0;
    }
    pressure_level_db(sum_pressure_sq.sqrt())
}

/// Return the squared directivity gain used by the production observer path.
///
/// The gain is normalized against the spherical mean of the unmodulated
/// dipole mixture.  The separate [`directivity_power_ratio`] reference below
/// integrates this exact production gain over an independent angular
/// quadrature and is therefore useful for checking that the observer path
/// redistributes, rather than creates or destroys, acoustic power.
fn directivity_squared_gain(harmonic: usize, cos_theta: f64, phi_azimuth_rad: f64) -> f64 {
    let sin_theta_sq = (1.0 - cos_theta * cos_theta).max(0.0);
    let (axial_w, tang_w) = dipole_weights_for_harmonic(harmonic);
    let strut_count = SUPPORT_STRUT_COUNT as f64 * 2.0;
    let azimuth_modulation = 1.0 + 0.15 * (strut_count * phi_azimuth_rad).cos();
    let downstream_bias = 1.0 + 0.3 * cos_theta;
    let raw_directivity_sq = (axial_w * cos_theta * cos_theta
        + tang_w * sin_theta_sq * azimuth_modulation)
        * downstream_bias.max(0.0);
    let spherical_mean = (axial_w + 2.0 * tang_w) / 3.0;
    raw_directivity_sq.max(0.0) / spherical_mean.max(f64::EPSILON)
}

/// Independently integrate the production directivity pattern over a sphere.
///
/// The returned ratio is the directional pressure-squared power at the
/// configured reference radius divided by the isotropic reference power
/// represented by `acoustics.harmonics`.  A value of one is the expected
/// conservation identity.  The eight-point Gauss-Legendre rule in `cos(θ)`
/// and a uniform azimuth rule are independent of the observer grid, which is
/// intentionally midpoint-sampled for presentation/optimization purposes.
///
/// `None` means that the acoustic solution has no finite, non-zero harmonic
/// power to normalize.
pub fn directivity_power_ratio(acoustics: &PcFanAcousticSolution) -> Option<f64> {
    const GAUSS_COS_THETA: [f64; 4] = [
        0.960_289_856_497_536_3,
        0.796_666_477_413_626_7,
        0.525_532_409_916_329,
        0.183_434_642_495_649_8,
    ];
    const GAUSS_WEIGHTS: [f64; 4] = [
        0.101_228_536_290_376_3,
        0.222_381_034_453_374_5,
        0.313_706_645_877_887_3,
        0.362_683_783_378_362,
    ];
    const AZIMUTH_SAMPLES: usize = 32;

    let reference_power: f64 = acoustics
        .harmonics
        .iter()
        .map(|harmonic| {
            if harmonic.pressure_pa.is_finite() && harmonic.pressure_pa >= 0.0 {
                harmonic.pressure_pa * harmonic.pressure_pa
            } else {
                f64::NAN
            }
        })
        .sum();
    if !reference_power.is_finite() || reference_power <= 0.0 {
        return None;
    }

    let mut directional_power = 0.0;
    for harmonic in &acoustics.harmonics {
        let mut spherical_mean_gain = 0.0;
        for (positive_cos_theta, weight) in GAUSS_COS_THETA
            .iter()
            .copied()
            .zip(GAUSS_WEIGHTS.iter().copied())
        {
            for cos_theta in [positive_cos_theta, -positive_cos_theta] {
                let azimuth_mean = (0..AZIMUTH_SAMPLES)
                    .map(|azimuth| {
                        let phi = std::f64::consts::TAU * azimuth as f64 / AZIMUTH_SAMPLES as f64;
                        directivity_squared_gain(harmonic.harmonic, cos_theta, phi)
                    })
                    .sum::<f64>()
                    / AZIMUTH_SAMPLES as f64;
                spherical_mean_gain += 0.5 * weight * azimuth_mean;
            }
        }
        directional_power += harmonic.pressure_pa * harmonic.pressure_pa * spherical_mean_gain;
    }

    Some(directional_power / reference_power)
}

/// Compute SPL on the configured observer grid.
fn compute_directivity_array(config: &PcFanConfig, acoustics: &PcFanAcousticSolution) -> Vec<f64> {
    let az = config.observers.azimuth_count.max(1);
    let el = config.observers.elevation_count.max(1);
    let mut spls = Vec::with_capacity(az * el);
    for ie in 0..el {
        // Polar angle θ ∈ (0, π) sampled with offset to avoid the poles
        // (where cosθ=±1 makes the in-plane contribution vanish exactly).
        let theta = std::f64::consts::PI * (ie as f64 + 0.5) / el as f64;
        for ia in 0..az {
            let phi = std::f64::consts::TAU * ia as f64 / az as f64;
            spls.push(observer_spl_db(config, acoustics, theta, phi));
        }
    }
    spls
}

/// Standard deviation of a slice of dB values (zero for empty/single).
fn std_dev_db(values: &[f64]) -> f64 {
    if values.len() < 2 {
        return 0.0;
    }
    let n = values.len() as f64;
    let mean = values.iter().sum::<f64>() / n;
    let variance = values
        .iter()
        .map(|v| {
            let d = v - mean;
            d * d
        })
        .sum::<f64>()
        / n;
    variance.sqrt()
}

pub fn evaluate_pc_fan(config: &PcFanConfig, design: &PcFanDesignVariables) -> PcFanEvaluation {
    let geometry = config.build_geometry(design);
    let evaluation = PcFanEvaluation::from_design(config, design, &geometry);
    attach_operating_point_spread(config, design, &geometry, evaluation)
}

pub fn evaluate_pc_fan_from_cfd_solution(
    config: &PcFanConfig,
    design: &PcFanDesignVariables,
    cfd: &PcFanCfdFlowSolution,
) -> PcFanEvaluation {
    let geometry = config.build_geometry(design);
    let flow = flow_from_cfd_solution(config, design, &geometry, cfd);
    let evaluation = PcFanEvaluation::from_flow_solution(config, design, &geometry, flow);
    attach_operating_point_spread(config, design, &geometry, evaluation)
}

/// Evaluate a CFD field only after its nonlinear solve has closed.
///
/// This is the production admission boundary for CLI, gallery, and any caller
/// that intends to promote a CFD-backed objective. The legacy
/// [`evaluate_pc_fan_from_cfd_solution`] remains available for diagnostics and
/// compatibility fixtures that deliberately inspect a non-converged iterate.
fn validate_checked_config(config: &PcFanConfig) -> Result<(), PcFanCfdEvaluationError> {
    let conservation_tolerances = [
        (
            "momentum imbalance",
            config.cfd.max_relative_momentum_imbalance,
        ),
        ("pressure closure", config.cfd.max_relative_pressure_closure),
        ("energy imbalance", config.cfd.max_relative_energy_imbalance),
    ];
    for (metric, tolerance) in conservation_tolerances {
        if !tolerance.is_finite() || tolerance < 0.0 {
            return Err(PcFanCfdEvaluationError::InvalidConservationTolerance {
                metric,
                tolerance,
            });
        }
    }

    macro_rules! require_envelope {
        ($condition:expr, $envelope:literal, $parameter:literal, $value:expr) => {
            if !$condition {
                return Err(PcFanCfdEvaluationError::InvalidEnvelopeParameter {
                    envelope: $envelope,
                    parameter: $parameter,
                    value: $value,
                });
            }
        };
    }

    let thermal = config.thermal;
    require_envelope!(
        thermal.ambient_temperature_c.is_finite(),
        "thermal",
        "ambient_temperature_c",
        thermal.ambient_temperature_c
    );
    require_envelope!(
        thermal.motor_thermal_resistance_k_per_w.is_finite()
            && thermal.motor_thermal_resistance_k_per_w >= 0.0,
        "thermal",
        "motor_thermal_resistance_k_per_w",
        thermal.motor_thermal_resistance_k_per_w
    );
    require_envelope!(
        thermal.motor_efficiency.is_finite()
            && thermal.motor_efficiency > 0.0
            && thermal.motor_efficiency <= 1.0,
        "thermal",
        "motor_efficiency",
        thermal.motor_efficiency
    );
    require_envelope!(
        thermal.max_winding_temperature_c.is_finite(),
        "thermal",
        "max_winding_temperature_c",
        thermal.max_winding_temperature_c
    );
    require_envelope!(
        thermal.max_electrical_power_w.is_finite() && thermal.max_electrical_power_w > 0.0,
        "thermal",
        "max_electrical_power_w",
        thermal.max_electrical_power_w
    );

    let structural = config.structural;
    require_envelope!(
        structural.material_density_kg_m3.is_finite() && structural.material_density_kg_m3 > 0.0,
        "structural",
        "material_density_kg_m3",
        structural.material_density_kg_m3
    );
    require_envelope!(
        structural.youngs_modulus_pa.is_finite() && structural.youngs_modulus_pa > 0.0,
        "structural",
        "youngs_modulus_pa",
        structural.youngs_modulus_pa
    );
    require_envelope!(
        structural.thermal_expansion_coefficient_per_k.is_finite()
            && structural.thermal_expansion_coefficient_per_k >= 0.0,
        "structural",
        "thermal_expansion_coefficient_per_k",
        structural.thermal_expansion_coefficient_per_k
    );
    require_envelope!(
        structural.thermal_constraint_factor.is_finite()
            && (0.0..=1.0).contains(&structural.thermal_constraint_factor),
        "structural",
        "thermal_constraint_factor",
        structural.thermal_constraint_factor
    );
    require_envelope!(
        structural.allowable_stress_pa.is_finite() && structural.allowable_stress_pa > 0.0,
        "structural",
        "allowable_stress_pa",
        structural.allowable_stress_pa
    );
    require_envelope!(
        structural.safety_factor.is_finite() && structural.safety_factor >= 1.0,
        "structural",
        "safety_factor",
        structural.safety_factor
    );
    require_envelope!(
        structural.fatigue_reference_stress_pa.is_finite()
            && structural.fatigue_reference_stress_pa > 0.0,
        "structural",
        "fatigue_reference_stress_pa",
        structural.fatigue_reference_stress_pa
    );
    require_envelope!(
        structural.fatigue_reference_cycles.is_finite()
            && structural.fatigue_reference_cycles > 0.0,
        "structural",
        "fatigue_reference_cycles",
        structural.fatigue_reference_cycles
    );
    require_envelope!(
        structural.fatigue_design_cycles.is_finite()
            && structural.fatigue_design_cycles >= structural.fatigue_reference_cycles,
        "structural",
        "fatigue_design_cycles",
        structural.fatigue_design_cycles
    );
    require_envelope!(
        structural.fatigue_exponent.is_finite() && structural.fatigue_exponent < 0.0,
        "structural",
        "fatigue_exponent",
        structural.fatigue_exponent
    );
    require_envelope!(
        structural.min_modal_separation_hz.is_finite() && structural.min_modal_separation_hz >= 0.0,
        "structural",
        "min_modal_separation_hz",
        structural.min_modal_separation_hz
    );
    require_envelope!(
        structural.max_imbalance_g_mm.is_finite() && structural.max_imbalance_g_mm >= 0.0,
        "structural",
        "max_imbalance_g_mm",
        structural.max_imbalance_g_mm
    );

    let manufacturing = config.manufacturing;
    require_envelope!(
        manufacturing.min_blade_thickness_m.is_finite()
            && manufacturing.min_blade_thickness_m > 0.0,
        "manufacturing",
        "min_blade_thickness_m",
        manufacturing.min_blade_thickness_m
    );
    require_envelope!(
        manufacturing.min_blade_chord_m.is_finite() && manufacturing.min_blade_chord_m > 0.0,
        "manufacturing",
        "min_blade_chord_m",
        manufacturing.min_blade_chord_m
    );
    require_envelope!(
        manufacturing.max_twist_gradient_deg.is_finite()
            && manufacturing.max_twist_gradient_deg >= 0.0,
        "manufacturing",
        "max_twist_gradient_deg",
        manufacturing.max_twist_gradient_deg
    );
    require_envelope!(
        manufacturing.max_shroud_offset_m.is_finite() && manufacturing.max_shroud_offset_m >= 0.0,
        "manufacturing",
        "max_shroud_offset_m",
        manufacturing.max_shroud_offset_m
    );
    require_envelope!(
        manufacturing.min_root_thickness_ratio.is_finite()
            && manufacturing.min_root_thickness_ratio > 0.0
            && manufacturing.min_root_thickness_ratio <= 1.0,
        "manufacturing",
        "min_root_thickness_ratio",
        manufacturing.min_root_thickness_ratio
    );
    if let Some((parameter, value)) = invalid_manufacturing_tolerance(manufacturing.tolerance) {
        return Err(PcFanCfdEvaluationError::InvalidEnvelopeParameter {
            envelope: "manufacturing",
            parameter,
            value,
        });
    }

    Ok(())
}

fn validate_cfd_solution_fields(cfd: &PcFanCfdFlowSolution) -> Result<(), PcFanCfdEvaluationError> {
    let node_count = cfd.nodes.len();
    for (field, actual) in [
        ("velocity", cfd.velocity_m_s.len()),
        ("pressure", cfd.pressure_pa.len()),
    ] {
        if actual != node_count {
            return Err(PcFanCfdEvaluationError::InvalidFieldShape {
                field,
                expected: node_count,
                actual,
            });
        }
    }
    for (index, node) in cfd.nodes.iter().enumerate() {
        if [node.x, node.y, node.z]
            .iter()
            .any(|value| !value.is_finite())
        {
            return Err(PcFanCfdEvaluationError::NonFiniteField {
                field: "nodes",
                index,
            });
        }
    }
    for (index, velocity) in cfd.velocity_m_s.iter().enumerate() {
        if velocity.iter().any(|value| !value.is_finite()) {
            return Err(PcFanCfdEvaluationError::NonFiniteField {
                field: "velocity",
                index,
            });
        }
    }
    for (index, pressure) in cfd.pressure_pa.iter().enumerate() {
        if !pressure.is_finite() {
            return Err(PcFanCfdEvaluationError::NonFiniteField {
                field: "pressure",
                index,
            });
        }
    }
    for (field, value) in [
        ("airflow_cfm", cfd.airflow_cfm),
        ("inlet_airflow_cfm", cfd.inlet_airflow_cfm),
        ("mass_balance_error_kg_s", cfd.mass_balance_error_kg_s),
        ("pressure_drop_pa", cfd.pressure_drop_pa),
        ("max_velocity_m_s", cfd.max_velocity_m_s),
        ("reynolds_number", cfd.reynolds_number),
    ] {
        if !value.is_finite() {
            return Err(PcFanCfdEvaluationError::NonFiniteObservable { field });
        }
    }
    Ok(())
}

fn conservation_ledger_is_algebraically_consistent(conservation: &PcFanCfdConservation) -> bool {
    let primitive_values = [
        conservation.inlet_mass_flow_kg_s,
        conservation.outlet_mass_flow_kg_s,
        conservation.inlet_axial_momentum_flux_n,
        conservation.outlet_axial_momentum_flux_n,
        conservation.pressure_force_n,
        conservation.body_force_n,
        conservation.axial_momentum_residual_n,
        conservation.pressure_force_from_loading_n,
        conservation.pressure_power_w,
        conservation.shaft_power_to_fluid_w,
    ];
    if primitive_values.iter().any(|value| !value.is_finite()) {
        return false;
    }

    let relative_error = |reported: f64, expected: f64| {
        (reported - expected).abs() / reported.abs().max(expected.abs()).max(f64::MIN_POSITIVE)
    };
    let mass_scale = conservation
        .inlet_mass_flow_kg_s
        .abs()
        .max(conservation.outlet_mass_flow_kg_s.abs())
        .max(f64::MIN_POSITIVE);
    let expected_mass =
        (conservation.outlet_mass_flow_kg_s - conservation.inlet_mass_flow_kg_s).abs() / mass_scale;

    let momentum_scale = (conservation.outlet_axial_momentum_flux_n
        - conservation.inlet_axial_momentum_flux_n)
        .abs()
        .max(conservation.pressure_force_n.abs())
        .max(conservation.body_force_n.abs())
        .max(f64::MIN_POSITIVE);
    let expected_momentum = conservation.axial_momentum_residual_n.abs() / momentum_scale;

    let pressure_scale = conservation
        .pressure_force_n
        .abs()
        .max(conservation.pressure_force_from_loading_n.abs())
        .max(f64::MIN_POSITIVE);
    let expected_pressure =
        (conservation.pressure_force_n + conservation.pressure_force_from_loading_n).abs()
            / pressure_scale;

    let energy_scale = conservation
        .pressure_power_w
        .abs()
        .max(conservation.shaft_power_to_fluid_w.abs())
        .max(f64::MIN_POSITIVE);
    let expected_energy =
        (conservation.pressure_power_w - conservation.shaft_power_to_fluid_w).abs() / energy_scale;

    [
        relative_error(conservation.relative_mass_imbalance, expected_mass),
        relative_error(conservation.relative_momentum_imbalance, expected_momentum),
        relative_error(conservation.relative_pressure_closure, expected_pressure),
        relative_error(conservation.relative_energy_imbalance, expected_energy),
    ]
    .iter()
    .all(|error| error.is_finite() && *error <= CFD_ALGEBRAIC_IDENTITY_RELATIVE_TOLERANCE)
}

pub fn try_evaluate_pc_fan_from_cfd_solution(
    config: &PcFanConfig,
    design: &PcFanDesignVariables,
    cfd: &PcFanCfdFlowSolution,
) -> Result<PcFanEvaluation, PcFanCfdEvaluationError> {
    validate_checked_config(config)?;
    validate_cfd_solution_fields(cfd)?;
    if cfd.backend_provenance.actual_backend != cfd.actual_backend {
        return Err(PcFanCfdEvaluationError::BackendProvenanceMismatch);
    }
    if cfd.backend_provenance.requested_backend != config.cfd.backend {
        return Err(PcFanCfdEvaluationError::BackendRequestMismatch {
            configured: config.cfd.backend,
            recorded: cfd.backend_provenance.requested_backend,
        });
    }
    if !cfd
        .execution_environment
        .matches_provenance(cfd.backend_provenance)
    {
        return Err(PcFanCfdEvaluationError::BackendProvenanceMismatch);
    }
    if !cfd.backend_provenance.is_admitted(
        config.cfd.allow_host_fallback,
        config.cfd.require_device_resident,
    ) {
        if cfd.backend_provenance.is_fallback() && !config.cfd.allow_host_fallback {
            return Err(PcFanCfdEvaluationError::BackendFallback {
                requested: cfd.backend_provenance.requested_backend,
                actual: cfd.backend_provenance.actual_backend,
                reason: cfd
                    .backend_provenance
                    .fallback_reason
                    .expect("is_fallback guarantees a reason"),
            });
        }
        return Err(PcFanCfdEvaluationError::BackendResidencyUnavailable {
            requested: cfd.backend_provenance.requested_backend,
            actual: cfd.backend_provenance.actual_backend,
            mode: cfd.backend_provenance.execution_mode,
        });
    }
    let tolerance = config.cfd.nonlinear_tolerance;
    if !tolerance.is_finite() || tolerance <= 0.0 {
        return Err(PcFanCfdEvaluationError::InvalidNonlinearTolerance { tolerance });
    }
    if !cfd.residual.is_finite() {
        return Err(PcFanCfdEvaluationError::NonFiniteResidual {
            residual: cfd.residual,
        });
    }
    if !cfd.converged || cfd.residual > tolerance {
        return Err(PcFanCfdEvaluationError::NonConverged {
            iterations: cfd.iterations,
            residual: cfd.residual,
            tolerance,
        });
    }
    // A healthy fan raises outlet pressure above inlet, so the
    // inlet-minus-outlet drop is clearly negative. A drop at or above the
    // stall floor means stalled or reversed flow; `pressure_rise_pa` is a
    // magnitude and would otherwise report a healthy-looking positive rise
    // for it. The floor is +1e-9 Pa (not exactly zero) so converged
    // coarse-grid solutions sitting at numerical noise (~1e-19) are not
    // misreported as reversed: a sub-nanopascal "stall" is noise, not
    // physics. (`pressure_drop_pa` finiteness is already enforced by
    // `validate_cfd_solution_fields` above.)
    if cfd.pressure_drop_pa > STALLED_FLOW_DROP_TOLERANCE_PA {
        return Err(PcFanCfdEvaluationError::ReversedOperatingPoint {
            pressure_drop_pa: cfd.pressure_drop_pa,
        });
    }
    let mass_tolerance = config.cfd.max_relative_mass_imbalance;
    if !mass_tolerance.is_finite() || mass_tolerance < 0.0 {
        return Err(PcFanCfdEvaluationError::InvalidMassBalanceTolerance {
            tolerance: mass_tolerance,
        });
    }
    let density = config.operating_point.air_density_kg_m3;
    let inlet_mass_flow = density * cfd.inlet_airflow_cfm / M3S_TO_CFM;
    let outlet_mass_flow = density * cfd.airflow_cfm / M3S_TO_CFM;
    let relative_mass_imbalance = cfd.mass_balance_error_kg_s.abs()
        / inlet_mass_flow
            .abs()
            .max(outlet_mass_flow.abs())
            .max(f64::MIN_POSITIVE);
    if !relative_mass_imbalance.is_finite() || relative_mass_imbalance > mass_tolerance {
        return Err(PcFanCfdEvaluationError::MassImbalance {
            relative_error: relative_mass_imbalance,
            tolerance: mass_tolerance,
        });
    }
    let momentum_tolerance = config.cfd.max_relative_momentum_imbalance;
    let pressure_tolerance = config.cfd.max_relative_pressure_closure;
    let energy_tolerance = config.cfd.max_relative_energy_imbalance;
    let conservation = cfd.conservation;
    let conservation_values = [
        conservation.relative_mass_imbalance,
        conservation.relative_momentum_imbalance,
        conservation.relative_pressure_closure,
        conservation.relative_energy_imbalance,
        conservation.pressure_power_w,
        conservation.shaft_power_to_fluid_w,
    ];
    if conservation_values.iter().any(|value| !value.is_finite()) {
        return Err(PcFanCfdEvaluationError::NonFiniteResidual {
            residual: conservation.relative_momentum_imbalance,
        });
    }
    if conservation.relative_mass_imbalance > mass_tolerance {
        return Err(PcFanCfdEvaluationError::MassImbalance {
            relative_error: conservation.relative_mass_imbalance,
            tolerance: mass_tolerance,
        });
    }
    if conservation.relative_momentum_imbalance > momentum_tolerance {
        return Err(PcFanCfdEvaluationError::MomentumImbalance {
            relative_error: conservation.relative_momentum_imbalance,
            tolerance: momentum_tolerance,
        });
    }
    if conservation.relative_pressure_closure > pressure_tolerance {
        return Err(PcFanCfdEvaluationError::PressureClosure {
            relative_error: conservation.relative_pressure_closure,
            tolerance: pressure_tolerance,
        });
    }
    if conservation.relative_energy_imbalance > energy_tolerance {
        return Err(PcFanCfdEvaluationError::EnergyImbalance {
            relative_error: conservation.relative_energy_imbalance,
            tolerance: energy_tolerance,
        });
    }
    let Some(loading) = cfd.actuator_loading.or(cfd.rotating_wall_loading) else {
        return Err(PcFanCfdEvaluationError::MissingPowerLedger);
    };
    let finite = loading.force_on_fluid_n.into_iter().all(f64::is_finite)
        && loading.torque_on_fluid_nm.is_finite()
        && loading.angular_speed_rad_s.is_finite()
        && loading.shaft_power_to_fluid_w.is_finite();
    if !finite || loading.angular_speed_rad_s <= 0.0 || loading.shaft_power_to_fluid_w <= 0.0 {
        return Err(PcFanCfdEvaluationError::InvalidActuatorLoading);
    }

    // The loading object is the source of truth for the mechanical ledger.
    // Recheck its algebraic identity here instead of trusting a copied
    // `shaft_power_to_fluid_w` or conservation residual supplied by a
    // caller. Without this check, a stale positive power value could pass
    // the positivity and relative-residual gates while disagreeing with the
    // torque actually integrated from the CFD force field.
    let expected_shaft_power_w = loading.torque_on_fluid_nm * loading.angular_speed_rad_s;
    let power_identity_scale = loading
        .shaft_power_to_fluid_w
        .abs()
        .max(expected_shaft_power_w.abs())
        .max(f64::MIN_POSITIVE);
    let power_identity_error =
        (loading.shaft_power_to_fluid_w - expected_shaft_power_w).abs() / power_identity_scale;
    let loading_axial_force_n = loading.force_on_fluid_n[2];
    let force_identity_scale = conservation
        .body_force_n
        .abs()
        .max(conservation.pressure_force_from_loading_n.abs())
        .max(loading_axial_force_n.abs())
        .max(f64::MIN_POSITIVE);
    let force_identity_error = (conservation.body_force_n - loading_axial_force_n)
        .abs()
        .max((conservation.pressure_force_from_loading_n - loading_axial_force_n).abs())
        / force_identity_scale;
    let conservation_power_scale = conservation
        .shaft_power_to_fluid_w
        .abs()
        .max(loading.shaft_power_to_fluid_w.abs())
        .max(f64::MIN_POSITIVE);
    let conservation_power_error =
        (conservation.shaft_power_to_fluid_w - loading.shaft_power_to_fluid_w).abs()
            / conservation_power_scale;
    if !power_identity_error.is_finite()
        || !force_identity_error.is_finite()
        || !conservation_power_error.is_finite()
        || power_identity_error > CFD_ALGEBRAIC_IDENTITY_RELATIVE_TOLERANCE
        || force_identity_error > CFD_ALGEBRAIC_IDENTITY_RELATIVE_TOLERANCE
        || conservation_power_error > CFD_ALGEBRAIC_IDENTITY_RELATIVE_TOLERANCE
    {
        return Err(PcFanCfdEvaluationError::InvalidActuatorLoading);
    }
    if !conservation_ledger_is_algebraically_consistent(&conservation) {
        return Err(PcFanCfdEvaluationError::InvalidConservationLedger);
    }
    Ok(evaluate_pc_fan_from_cfd_solution(config, design, cfd))
}

/// Evaluate a design through the checked CFD admission path when CFD is active.
pub fn try_evaluate_pc_fan(
    config: &PcFanConfig,
    design: &PcFanDesignVariables,
) -> Result<PcFanEvaluation, PcFanCfdEvaluationError> {
    validate_checked_config(config)?;
    let geometry = config.build_geometry(design);
    if config.cfd.flow_model.is_cfd() {
        let cfd = solve_pc_fan_cfd_mean_flow_for_geometry(config, design, &geometry);
        return try_evaluate_pc_fan_from_cfd_solution(config, design, &cfd);
    }
    let evaluation = PcFanEvaluation::from_design(config, design, &geometry);
    Ok(attach_operating_point_spread(
        config, design, &geometry, evaluation,
    ))
}

pub fn solve_pc_fan_mean_flow(
    config: &PcFanConfig,
    design: &PcFanDesignVariables,
) -> PcFanFlowSolution {
    let geometry = config.build_geometry(design);
    solve_mean_flow(config, design, &geometry)
}

pub fn solve_pc_fan_acoustics(
    config: &PcFanConfig,
    design: &PcFanDesignVariables,
) -> PcFanAcousticSolution {
    let geometry = config.build_geometry(design);
    let flow = solve_mean_flow(config, design, &geometry);
    evaluate_acoustics(config, design, &geometry, &flow)
}

/// Evaluate the legacy shape-gradient entry point.
///
/// Reduced-order configurations use the checked discrete-adjoint path so the
/// returned state multipliers and gradient are genuine adjoint quantities.
/// Resolved-CFD configurations retain the historical infallible finite-
/// difference compatibility behavior; callers that need a checked resolved
/// derivative must use one of the policy-aware or resolved-adjoint APIs.
pub fn solve_pc_fan_shape_gradient(
    config: &PcFanConfig,
    design: &PcFanDesignVariables,
) -> PcFanShapeGradient {
    if !config.cfd.flow_model.is_cfd()
        && let Ok(gradient) = solve_pc_fan_shape_gradient_reduced_adjoint(config, design)
    {
        return gradient;
    }
    solve_pc_fan_shape_gradient_finite_difference(config, design)
}

fn solve_pc_fan_shape_gradient_finite_difference(
    config: &PcFanConfig,
    design: &PcFanDesignVariables,
) -> PcFanShapeGradient {
    let geometry = config.build_geometry(design);
    let flow = solve_mean_flow(config, design, &geometry);
    let pressure_rhs = acoustic_pressure_rhs(config, design, &geometry, flow.airflow_cfm);
    let residual = primal_residual(config, design, &geometry, &flow, &pressure_rhs);
    let objective_gradient = finite_difference_shape_objective_gradient(config, design);

    PcFanShapeGradient {
        objective_gradient,
        adjoint: PcFanAdjointState {
            airflow_multiplier: 0.0,
            acoustic_pressure_multipliers: vec![0.0; pressure_rhs.len()],
        },
        residual,
        adjoint_is_finite_difference: true,
    }
}

/// Evaluate a resolved-CFD finite-difference gradient only through the
/// checked objective-admission boundary.
///
/// The legacy [`solve_pc_fan_shape_gradient`] helper remains an infallible
/// compatibility entry point: reduced-order callers use its checked adjoint,
/// while resolved-CFD callers retain finite-difference compatibility behavior.
/// This checked path has a stronger contract: the base state and every
/// plus/minus perturbation must independently satisfy convergence,
/// field-shape, conservation, loading, and objective-admission checks before
/// its objective can enter the gradient.
fn solve_pc_fan_shape_gradient_checked_cfd(
    config: &PcFanConfig,
    design: &PcFanDesignVariables,
) -> Result<PcFanShapeGradient, PcFanCfdEvaluationError> {
    let geometry = config.build_geometry(design);
    let base_cfd = solve_pc_fan_cfd_mean_flow_for_geometry(config, design, &geometry);
    try_evaluate_pc_fan_from_cfd_solution(config, design, &base_cfd)?;

    let base_flow = flow_from_cfd_solution(config, design, &geometry, &base_cfd);
    let pressure_rhs = acoustic_pressure_rhs(config, design, &geometry, base_flow.airflow_cfm);
    let residual = primal_residual(config, design, &geometry, &base_flow, &pressure_rhs);
    let mut objective_gradient = [0.0; DESIGN_VARIABLE_COUNT];

    for (index, value) in design.to_vector().into_iter().enumerate() {
        let step = design_variable_step(config, index, value);
        let plus_design = perturbed_design(design, index, step);
        let minus_design = perturbed_design(design, index, -step);
        let plus_geometry = config.build_geometry(&plus_design);
        let minus_geometry = config.build_geometry(&minus_design);

        let plus_cfd =
            solve_pc_fan_cfd_mean_flow_for_geometry(config, &plus_design, &plus_geometry);
        let minus_cfd =
            solve_pc_fan_cfd_mean_flow_for_geometry(config, &minus_design, &minus_geometry);
        let plus =
            try_evaluate_pc_fan_from_cfd_solution(config, &plus_design, &plus_cfd)?.objective;
        let minus =
            try_evaluate_pc_fan_from_cfd_solution(config, &minus_design, &minus_cfd)?.objective;

        objective_gradient[index] = (plus - minus) / (2.0 * step);
    }

    Ok(PcFanShapeGradient {
        objective_gradient,
        adjoint: PcFanAdjointState {
            airflow_multiplier: 0.0,
            acoustic_pressure_multipliers: vec![0.0; pressure_rhs.len()],
        },
        residual,
        adjoint_is_finite_difference: true,
    })
}

/// Solve the discrete adjoint of the reduced-order scalar airflow state.
///
/// The reduced primal equation is
/// `R(q, z) = q - q_rhs(z) = 0`, where all tonal, broadband, thermal, and
/// structural terms are evaluated from `q`. The acoustic algebraic state is
/// therefore eliminated exactly from this reduced model and its tonal
/// sensitivity enters `dJ/dq`; it is not represented as a fabricated zero
/// pressure multiplier. The Jacobian entries are centered numerical
/// derivatives of the declared reduced residual/objective, while the
/// transposed solve and Lagrangian assembly are explicit.
#[allow(dead_code)]
fn solve_pc_fan_shape_gradient_reduced_scalar_adjoint(
    config: &PcFanConfig,
    design: &PcFanDesignVariables,
) -> Result<PcFanShapeGradient, PcFanDerivativePolicyError> {
    if config.cfd.flow_model.is_cfd() {
        return Err(PcFanDerivativePolicyError::DiscreteAdjointUnavailable {
            policy: PcFanDerivativePolicy::ReducedOrderDiscreteAdjoint,
        });
    }

    let geometry = config.build_geometry(design);
    let airflow_cfm = mean_flow_airflow_rhs(config, design, &geometry);
    let flow = flow_from_airflow_cfm(config, design, &geometry, airflow_cfm);
    let pressure_rhs = acoustic_pressure_rhs(config, design, &geometry, airflow_cfm);
    let residual = primal_residual(config, design, &geometry, &flow, &pressure_rhs);

    let airflow_step = (config.optimizer.finite_difference_step.abs().max(1.0e-7)
        * airflow_cfm.abs().max(1.0))
    .max(1.0e-6);
    let objective_plus =
        reduced_objective_at_airflow(config, design, &geometry, airflow_cfm + airflow_step);
    let objective_minus =
        reduced_objective_at_airflow(config, design, &geometry, airflow_cfm - airflow_step);
    let objective_airflow_derivative = (objective_plus - objective_minus) / (2.0 * airflow_step);

    // R_q = 1. The sign follows J(z) + λᵀR(z), hence λ = -J_q.
    let airflow_multiplier = -objective_airflow_derivative;
    let mut objective_gradient = [0.0; DESIGN_VARIABLE_COUNT];
    for (index, value) in design.to_vector().into_iter().enumerate() {
        let step = design_variable_step(config, index, value);
        let plus_design = perturbed_design(design, index, step);
        let minus_design = perturbed_design(design, index, -step);
        let plus_geometry = config.build_geometry(&plus_design);
        let minus_geometry = config.build_geometry(&minus_design);

        // Partial objective derivative at fixed primal state q.
        let direct_plus =
            reduced_objective_at_airflow(config, &plus_design, &plus_geometry, airflow_cfm);
        let direct_minus =
            reduced_objective_at_airflow(config, &minus_design, &minus_geometry, airflow_cfm);
        let objective_design_derivative = (direct_plus - direct_minus) / (2.0 * step);

        // R_z = -q_rhs,z. The adjoint contribution is λᵀR_z.
        let plus_rhs = mean_flow_airflow_rhs(config, &plus_design, &plus_geometry);
        let minus_rhs = mean_flow_airflow_rhs(config, &minus_design, &minus_geometry);
        let residual_design_derivative = -(plus_rhs - minus_rhs) / (2.0 * step);
        objective_gradient[index] =
            objective_design_derivative + airflow_multiplier * residual_design_derivative;
    }

    Ok(PcFanShapeGradient {
        objective_gradient,
        adjoint: PcFanAdjointState {
            airflow_multiplier,
            // Acoustic pressure is an explicit eliminated output of the
            // reduced primal model, so no independent pressure multiplier is
            // claimed here.
            acoustic_pressure_multipliers: vec![0.0; pressure_rhs.len()],
        },
        residual,
        adjoint_is_finite_difference: false,
    })
}

/// Apply an explicit derivative policy to shape-gradient requests.
///
/// `FiniteDifferenceFullObjective` returns a centered finite-difference
/// gradient for reduced-order flow and bounded resolved-CFD cases whose base
/// and every perturbation pass checked convergence/conservation admission. It
/// remains a finite-difference fallback, not an exact resolved-flow discrete
/// adjoint. `DerivativeFree` also fails closed instead of returning a
/// misleading gradient; pair it with
/// [`crate::optimize_pc_fan_with_derivative_policy`] for DE optimization.
/// Solve the full reduced-order adjoint for airflow and harmonic pressure
/// states.
///
/// The reduced primal equations are `R_q = q - q_rhs(z)` and
/// `R_p = p - p_rhs(q, z)`. The previous scalar implementation eliminated
/// `p` inside `dJ/dq`, which left the pressure multipliers at zero. This
/// implementation solves the triangular transposed state system explicitly
/// and retains the harmonic multipliers in the public diagnostic.
pub fn solve_pc_fan_shape_gradient_reduced_adjoint(
    config: &PcFanConfig,
    design: &PcFanDesignVariables,
) -> Result<PcFanShapeGradient, PcFanDerivativePolicyError> {
    if config.cfd.flow_model.is_cfd() {
        return Err(PcFanDerivativePolicyError::DiscreteAdjointUnavailable {
            policy: PcFanDerivativePolicy::ReducedOrderDiscreteAdjoint,
        });
    }

    let geometry = config.build_geometry(design);
    let airflow_cfm = mean_flow_airflow_rhs(config, design, &geometry);
    let flow = flow_from_airflow_cfm(config, design, &geometry, airflow_cfm);
    let pressure_rhs = acoustic_pressure_rhs(config, design, &geometry, airflow_cfm);
    let residual = primal_residual(config, design, &geometry, &flow, &pressure_rhs);
    let airflow_step = (config.optimizer.finite_difference_step.abs().max(1.0e-7)
        * airflow_cfm.abs().max(1.0))
    .max(1.0e-6);

    let objective_plus = reduced_objective_at_states(
        config,
        design,
        &geometry,
        airflow_cfm + airflow_step,
        &pressure_rhs,
    );
    let objective_minus = reduced_objective_at_states(
        config,
        design,
        &geometry,
        airflow_cfm - airflow_step,
        &pressure_rhs,
    );
    let objective_airflow_derivative = (objective_plus - objective_minus) / (2.0 * airflow_step);

    let mut acoustic_pressure_multipliers = Vec::with_capacity(pressure_rhs.len());
    for (index, &pressure) in pressure_rhs.iter().enumerate() {
        let pressure_step = (config.optimizer.finite_difference_step.abs().max(1.0e-7)
            * pressure.abs().max(1.0e-6))
        .max(1.0e-10);
        if !pressure.is_finite() || pressure <= 1.0e-12 || pressure - pressure_step <= 1.0e-12 {
            return Err(PcFanDerivativePolicyError::AcousticStateAtDiagnosticFloor {
                harmonic: index + 1,
            });
        }
        let mut pressure_plus = pressure_rhs.clone();
        let mut pressure_minus = pressure_rhs.clone();
        pressure_plus[index] += pressure_step;
        pressure_minus[index] -= pressure_step;
        let objective_plus =
            reduced_objective_at_states(config, design, &geometry, airflow_cfm, &pressure_plus);
        let objective_minus =
            reduced_objective_at_states(config, design, &geometry, airflow_cfm, &pressure_minus);
        let derivative =
            (objective_plus - objective_minus) / (pressure_plus[index] - pressure_minus[index]);
        acoustic_pressure_multipliers.push(-derivative);
    }

    // R_p = p - p_rhs(q, z), so λ_p = -J_p and
    // λ_q = -J_q + (∂p_rhs/∂q)^T λ_p.
    let pressure_plus =
        acoustic_pressure_rhs(config, design, &geometry, airflow_cfm + airflow_step);
    let pressure_minus =
        acoustic_pressure_rhs(config, design, &geometry, airflow_cfm - airflow_step);
    let airflow_multiplier = -objective_airflow_derivative
        + pressure_plus
            .iter()
            .zip(pressure_minus.iter())
            .zip(acoustic_pressure_multipliers.iter())
            .map(|((plus, minus), multiplier)| multiplier * (plus - minus) / (2.0 * airflow_step))
            .sum::<f64>();

    let mut objective_gradient = [0.0; DESIGN_VARIABLE_COUNT];
    for (index, value) in design.to_vector().into_iter().enumerate() {
        let step = design_variable_step(config, index, value);
        let plus_design = perturbed_design(design, index, step);
        let minus_design = perturbed_design(design, index, -step);
        let plus_geometry = config.build_geometry(&plus_design);
        let minus_geometry = config.build_geometry(&minus_design);

        let direct_plus = reduced_objective_at_states(
            config,
            &plus_design,
            &plus_geometry,
            airflow_cfm,
            &pressure_rhs,
        );
        let direct_minus = reduced_objective_at_states(
            config,
            &minus_design,
            &minus_geometry,
            airflow_cfm,
            &pressure_rhs,
        );
        let objective_design_derivative = (direct_plus - direct_minus) / (2.0 * step);

        let plus_flow_rhs = mean_flow_airflow_rhs(config, &plus_design, &plus_geometry);
        let minus_flow_rhs = mean_flow_airflow_rhs(config, &minus_design, &minus_geometry);
        let airflow_residual_derivative = -(plus_flow_rhs - minus_flow_rhs) / (2.0 * step);

        let plus_pressure_rhs =
            acoustic_pressure_rhs(config, &plus_design, &plus_geometry, airflow_cfm);
        let minus_pressure_rhs =
            acoustic_pressure_rhs(config, &minus_design, &minus_geometry, airflow_cfm);
        let pressure_residual_contribution = plus_pressure_rhs
            .iter()
            .zip(minus_pressure_rhs.iter())
            .zip(acoustic_pressure_multipliers.iter())
            .map(|((plus, minus), multiplier)| multiplier * (-(plus - minus) / (2.0 * step)))
            .sum::<f64>();

        objective_gradient[index] = objective_design_derivative
            + airflow_multiplier * airflow_residual_derivative
            + pressure_residual_contribution;
    }

    Ok(PcFanShapeGradient {
        objective_gradient,
        adjoint: PcFanAdjointState {
            airflow_multiplier,
            acoustic_pressure_multipliers,
        },
        residual,
        adjoint_is_finite_difference: false,
    })
}

pub fn solve_pc_fan_shape_gradient_with_policy(
    config: &PcFanConfig,
    design: &PcFanDesignVariables,
    policy: PcFanDerivativePolicy,
) -> Result<PcFanShapeGradient, PcFanDerivativePolicyError> {
    match policy {
        PcFanDerivativePolicy::FiniteDifferenceFullObjective => {
            if config.cfd.flow_model.is_cfd() {
                solve_pc_fan_shape_gradient_checked_cfd(config, design).map_err(|source| {
                    PcFanDerivativePolicyError::FiniteDifferenceEvaluationFailed { policy, source }
                })
            } else {
                Ok(solve_pc_fan_shape_gradient_finite_difference(
                    config, design,
                ))
            }
        }
        PcFanDerivativePolicy::ReducedOrderDiscreteAdjoint => {
            solve_pc_fan_shape_gradient_reduced_adjoint(config, design)
        }
        PcFanDerivativePolicy::DerivativeFree => {
            Err(PcFanDerivativePolicyError::ShapeGradientUnavailable { policy })
        }
    }
}

fn solve_mean_flow(
    config: &PcFanConfig,
    design: &PcFanDesignVariables,
    geometry: &PcFanGeometry,
) -> PcFanFlowSolution {
    if config.cfd.flow_model.is_cfd() {
        return flow_from_cfd_solution(
            config,
            design,
            geometry,
            &solve_pc_fan_cfd_mean_flow_for_geometry(config, design, geometry),
        );
    }

    let airflow_cfm = mean_flow_airflow_rhs(config, design, geometry);
    flow_from_airflow_cfm(config, design, geometry, airflow_cfm)
}

fn flow_from_cfd_solution(
    config: &PcFanConfig,
    design: &PcFanDesignVariables,
    geometry: &PcFanGeometry,
    cfd: &PcFanCfdFlowSolution,
) -> PcFanFlowSolution {
    let hub_radius_m = design.effective_hub_radius_m(config);
    let tip_radius_m = design.effective_tip_radius_m(config);
    let disk_area_m2 = std::f64::consts::PI * (tip_radius_m.powi(2) - hub_radius_m.powi(2));
    // `try_build_geometry` rejects tip <= hub before any flow helper runs, so
    // a non-positive annulus here means unvalidated input: fail loudly
    // instead of dividing by EPSILON and reporting an astronomic velocity.
    assert!(
        disk_area_m2 > 0.0,
        "degenerate fan annulus: tip radius {tip_radius_m:.6} m must exceed hub radius {hub_radius_m:.6} m"
    );
    let airflow_cfm = cfd.airflow_cfm;
    let flow_m3_s = airflow_cfm / M3S_TO_CFM;
    let axial_velocity_m_s = flow_m3_s / disk_area_m2;
    // Magnitude only: a stalled/reversed point (non-negative drop) would read
    // as a healthy positive rise here, so the checked admission boundary
    // (`try_evaluate_pc_fan_from_cfd_solution`) rejects such points before
    // this value can enter an objective.
    let pressure_rise_pa = cfd.pressure_drop_pa.abs();
    let mach_number = axial_velocity_m_s.abs() / config.operating_point.speed_of_sound_m_s.max(1.0);

    PcFanFlowSolution {
        airflow_cfm,
        axial_velocity_m_s,
        pressure_rise_pa,
        reynolds_number: cfd.reynolds_number,
        mach_number,
        disk_solidity: disk_solidity(geometry),
        fixed_hardware: fixed_hardware_flow_effects(config, geometry),
    }
}

fn mean_flow_airflow_rhs(
    config: &PcFanConfig,
    design: &PcFanDesignVariables,
    geometry: &PcFanGeometry,
) -> f64 {
    installed_operating_airflow_m3_s(config, design, geometry) * M3S_TO_CFM
}

fn flow_from_airflow_cfm(
    config: &PcFanConfig,
    design: &PcFanDesignVariables,
    geometry: &PcFanGeometry,
    airflow_cfm: f64,
) -> PcFanFlowSolution {
    let hub_radius_m = design.effective_hub_radius_m(config);
    let tip_radius_m = design.effective_tip_radius_m(config);
    let disk_area_m2 = std::f64::consts::PI * (tip_radius_m.powi(2) - hub_radius_m.powi(2));
    // See `flow_from_cfd_solution`: the geometry constructor already rejects
    // a collapsed annulus; fail loudly here rather than hiding it behind an
    // EPSILON floor.
    assert!(
        disk_area_m2 > 0.0,
        "degenerate fan annulus: tip radius {tip_radius_m:.6} m must exceed hub radius {hub_radius_m:.6} m"
    );
    let tip_speed_m_s = std::f64::consts::TAU * config.operating_point.rpm / 60.0 * tip_radius_m;
    let solidity = disk_solidity(geometry);
    let flow_m3_s = airflow_cfm / M3S_TO_CFM;
    let axial_velocity_m_s = flow_m3_s / disk_area_m2;
    let mean_chord_m = mean_blade_chord_m(design);
    let fixed_hardware = fixed_hardware_flow_effects(config, geometry);
    let pressure_rise_pa =
        installed_system_pressure_loss_pa(config, geometry, flow_m3_s, fixed_hardware);
    let reynolds_number = config.operating_point.air_density_kg_m3 * tip_speed_m_s * mean_chord_m
        / config
            .operating_point
            .dynamic_viscosity_pa_s
            .max(f64::EPSILON);

    PcFanFlowSolution {
        airflow_cfm,
        axial_velocity_m_s,
        pressure_rise_pa,
        reynolds_number,
        mach_number: axial_velocity_m_s.abs() / config.operating_point.speed_of_sound_m_s.max(1.0),
        disk_solidity: solidity,
        fixed_hardware,
    }
}

#[derive(Debug, Clone, Copy)]
struct BladeElementFlowSummary {
    free_airflow_m3_s: f64,
    shutoff_pressure_pa: f64,
}

fn reduced_objective_at_airflow(
    config: &PcFanConfig,
    design: &PcFanDesignVariables,
    geometry: &PcFanGeometry,
    airflow_cfm: f64,
) -> f64 {
    let flow = flow_from_airflow_cfm(config, design, geometry, airflow_cfm);
    let evaluation = PcFanEvaluation::from_flow_solution(config, design, geometry, flow);
    attach_operating_point_spread(config, design, geometry, evaluation).objective
}

/// Evaluate the reduced objective with explicit primal airflow and harmonic
/// pressure states. At the converged state this is identical to
/// `reduced_objective_at_airflow`; off-state values are used only for partial
/// derivatives in the full reduced-order adjoint.
fn reduced_objective_at_states(
    config: &PcFanConfig,
    design: &PcFanDesignVariables,
    geometry: &PcFanGeometry,
    airflow_cfm: f64,
    pressures: &[f64],
) -> f64 {
    let flow = flow_from_airflow_cfm(config, design, geometry, airflow_cfm);
    let acoustics = evaluate_acoustics(config, design, geometry, &flow);
    let acoustics = acoustic_solution_with_pressure_state(acoustics, pressures);
    let evaluation = PcFanEvaluation::from_flow_solution_with_acoustics(
        config, design, geometry, flow, acoustics,
    );
    attach_operating_point_spread(config, design, geometry, evaluation).objective
}

fn installed_operating_airflow_m3_s(
    config: &PcFanConfig,
    design: &PcFanDesignVariables,
    geometry: &PcFanGeometry,
) -> f64 {
    let summary = blade_element_flow_summary(config, design, geometry);
    if summary.free_airflow_m3_s <= 0.0 || summary.shutoff_pressure_pa <= 0.0 {
        return 0.0;
    }

    let fixed_hardware = fixed_hardware_flow_effects(config, geometry);
    let system_k = installed_system_pressure_loss_coefficient(config, geometry, fixed_hardware);
    let fan_k = summary.shutoff_pressure_pa / summary.free_airflow_m3_s.powi(2);
    (summary.shutoff_pressure_pa / (fan_k + system_k).max(f64::EPSILON)).sqrt()
}

fn blade_element_flow_summary(
    config: &PcFanConfig,
    design: &PcFanDesignVariables,
    geometry: &PcFanGeometry,
) -> BladeElementFlowSummary {
    let rho = config.operating_point.air_density_kg_m3.max(1.0e-9);
    let omega = std::f64::consts::TAU * config.operating_point.rpm.max(0.0) / 60.0;
    let r_hub = design.effective_hub_radius_m(config).max(1.0e-6);
    let r_tip = design.effective_tip_radius_m(config).max(r_hub + 1.0e-6);
    let span_m = r_tip - r_hub;
    let disk_area_m2 = std::f64::consts::PI * (r_tip * r_tip - r_hub * r_hub);
    let tip_speed_m_s = omega * r_tip;
    if tip_speed_m_s <= 0.0 || disk_area_m2 <= 0.0 {
        return BladeElementFlowSummary {
            free_airflow_m3_s: 0.0,
            shutoff_pressure_pa: 0.0,
        };
    }

    let samples = 21;
    let mut pressure_loading = 0.0;
    let mut flow_coefficient = 0.0;
    let mut weight_sum = 0.0;

    for i in 0..samples {
        let eta = i as f64 / (samples - 1) as f64;
        let r = r_hub + eta * span_m;
        let section = design.blade_section_at(eta);
        let path = design.blade_path_at(eta);
        let chord_m = section.chord_m.max(1.0e-6);
        let local_speed = omega * r;
        let radius_weight = r / r_tip;
        let span_weight = if i == 0 || i + 1 == samples { 0.5 } else { 1.0 };
        let station_weight = span_weight * radius_weight;
        let local_solidity = (config.blade_count as f64 * chord_m
            / (std::f64::consts::TAU * r).max(1.0e-9))
        .clamp(0.0, 1.6);
        let camber_ratio = (section.camber_m / chord_m).clamp(-0.20, 0.20);
        let thickness_ratio = (section.thickness_m / chord_m).clamp(0.01, 0.25);
        let pitch = section.twist_deg.to_radians();
        let useful_pitch = pitch.sin().max(0.0);
        let sweep_relief =
            (1.0 - 0.22 * (path.tangential_offset_m / span_m).abs()).clamp(0.72, 1.0);

        let lift_proxy = (0.55 * useful_pitch + 1.55 * camber_ratio.max(0.0)).clamp(0.0, 1.25);
        let profile_drag = 0.018 + 0.18 * thickness_ratio.powi(2) + 0.035 * camber_ratio.abs();
        let section_efficiency = (lift_proxy - profile_drag).max(0.0);

        pressure_loading += station_weight
            * local_solidity
            * section_efficiency
            * sweep_relief
            * (local_speed / tip_speed_m_s).powi(2);
        flow_coefficient += station_weight
            * local_solidity
            * (0.20 + 0.85 * useful_pitch + 0.55 * camber_ratio.max(0.0))
            * sweep_relief;
        weight_sum += station_weight;
    }

    let loading = if weight_sum > 0.0 {
        pressure_loading / weight_sum
    } else {
        0.0
    };
    let flow_coeff = if weight_sum > 0.0 {
        flow_coefficient / weight_sum
    } else {
        0.0
    };
    let clearance_m = geometry.shroud.minimum_radius_m() - r_tip;
    let clearance_loss = tip_clearance_flow_factor(clearance_m, r_tip);
    let fixed_hardware = fixed_hardware_flow_effects(config, geometry);
    let area_ratio = shroud_effective_open_area_m2(geometry) / disk_area_m2.max(f64::EPSILON);

    let shutoff_pressure_pa = 0.5
        * rho
        * tip_speed_m_s
        * tip_speed_m_s
        * (2.6 * loading)
        * clearance_loss
        * fixed_hardware.total_loss_factor
        * area_ratio.clamp(0.65, 1.20);
    let free_airflow_m3_s = disk_area_m2
        * tip_speed_m_s
        * (0.08 + 0.42 * flow_coeff).clamp(0.04, 0.42)
        * clearance_loss
        * fixed_hardware.total_loss_factor
        * area_ratio.clamp(0.70, 1.15);

    BladeElementFlowSummary {
        free_airflow_m3_s: free_airflow_m3_s.max(0.0),
        shutoff_pressure_pa: shutoff_pressure_pa.max(0.0),
    }
}

fn installed_system_pressure_loss_pa(
    config: &PcFanConfig,
    geometry: &PcFanGeometry,
    flow_m3_s: f64,
    fixed_hardware: PcFanHardwareFlowEffects,
) -> f64 {
    installed_system_pressure_loss_coefficient(config, geometry, fixed_hardware)
        * flow_m3_s.max(0.0).powi(2)
}

fn installed_system_pressure_loss_coefficient(
    config: &PcFanConfig,
    geometry: &PcFanGeometry,
    fixed_hardware: PcFanHardwareFlowEffects,
) -> f64 {
    let rho = config.operating_point.air_density_kg_m3.max(1.0e-9);
    let area_m2 = shroud_effective_open_area_m2(geometry).max(1.0e-9);
    let depth_ratio =
        (geometry.shroud.depth_m / geometry.shroud.inner_radius_m.max(1.0e-6)).clamp(0.05, 1.0);
    let blockage = fixed_hardware.blockage_ratio.clamp(0.0, 0.85);
    let contraction = (1.0 / (1.0 - blockage).max(0.15).powi(2) - 1.0).max(0.0);
    let wall = (geometry.frame_box.wall_thickness_m / geometry.frame_box.depth_m.max(1.0e-6))
        .clamp(0.0, 0.35);
    let shroud_loss = 0.18 + 0.38 * depth_ratio;
    let frame_loss = 0.10 + 0.80 * wall;
    let support_loss = 0.65 * contraction + 0.20 * fixed_hardware.strut_blockage_ratio;
    let loss_coefficient = shroud_loss + frame_loss + support_loss;

    0.5 * rho * loss_coefficient / (area_m2 * area_m2)
}

fn shroud_effective_open_area_m2(geometry: &PcFanGeometry) -> f64 {
    let radius = geometry.shroud.minimum_radius_m().max(0.0);
    let circular_area = std::f64::consts::PI * radius * radius;
    let frame_area = geometry.frame_box.inner_side_m * geometry.frame_box.inner_side_m;
    circular_area.min(frame_area).max(0.0)
}

fn tip_clearance_flow_factor(clearance_m: f64, r_tip_m: f64) -> f64 {
    if clearance_m <= 0.0 {
        return 0.20;
    }
    let nominal_clearance_m = (0.055 * r_tip_m).max(1.0e-6);
    (nominal_clearance_m / clearance_m).clamp(0.30, 1.10)
}

fn evaluate_acoustics(
    config: &PcFanConfig,
    design: &PcFanDesignVariables,
    geometry: &PcFanGeometry,
    flow: &PcFanFlowSolution,
) -> PcFanAcousticSolution {
    let source = propeller_source_for(config, design, flow);
    let blade_pressures = blade_acoustic_pressures(config, design, geometry, flow);
    let fixed_hardware_pressures =
        fixed_hardware_noise_pressures(config, design, geometry, flow, blade_pressures.len());
    let fixed_hardware_noise = PcFanHardwareNoise {
        support_wake_spl_db: combine_pressures_db(
            fixed_hardware_pressures
                .iter()
                .map(|p| p.support_wake_pressure_pa),
        ),
        motor_mount_spl_db: combine_pressures_db(
            fixed_hardware_pressures
                .iter()
                .map(|p| p.motor_mount_pressure_pa),
        ),
        combined_spl_db: combine_pressures_db(
            fixed_hardware_pressures
                .iter()
                .map(|p| p.total_pressure_pa()),
        ),
        support_wake_broadband_spl_db: combine_pressures_db(
            fixed_hardware_pressures
                .iter()
                .map(|p| p.support_wake_broadband_pa),
        ),
        motor_mount_broadband_spl_db: combine_pressures_db(
            fixed_hardware_pressures
                .iter()
                .map(|p| p.motor_mount_broadband_pa),
        ),
    };
    let reference_distance_m = acoustic_reference_distance_m(config);
    // With RMS far-field pressure and the unit-mean directivity convention,
    // integrating intensity over the observer sphere gives source power.
    let radiation_area_m2 = 4.0 * std::f64::consts::PI * reference_distance_m.powi(2);
    let tip_leakage_pressures =
        tip_leakage_pressures_per_harmonic(config, design, geometry, blade_pressures.len());
    let trailing_edge_pressures =
        trailing_edge_pressures_per_harmonic(config, design, flow, blade_pressures.len());
    let leading_edge_pressures =
        leading_edge_inflow_pressures_per_harmonic(config, design, flow, blade_pressures.len());
    let mut harmonics = Vec::with_capacity(blade_pressures.len());

    for (i, blade_pressure_pa) in blade_pressures.into_iter().enumerate() {
        let harmonic = i + 1;
        let frequency_hz = source.harmonic_frequency(harmonic);
        let hardware_pressure_pa = fixed_hardware_pressures[i].total_pressure_pa();
        let tip_leakage_pa = tip_leakage_pressures[i];
        let trailing_edge_pa = trailing_edge_pressures[i];
        let leading_edge_pa = leading_edge_pressures[i];
        // Keep the physical harmonic state at zero when every source is
        // silent. Floors belong only to logarithmic diagnostics below; a
        // pressure floor here would create a fictitious acoustic source and
        // a spurious adjoint sensitivity at zero load.
        let source_pressure_pa = (blade_pressure_pa * blade_pressure_pa
            + hardware_pressure_pa * hardware_pressure_pa
            + tip_leakage_pa * tip_leakage_pa
            + trailing_edge_pa * trailing_edge_pa
            + leading_edge_pa * leading_edge_pa)
            .sqrt();
        let acoustic_intensity_w_m2 = source_pressure_pa * source_pressure_pa
            / (config.operating_point.air_density_kg_m3
                * config.operating_point.speed_of_sound_m_s);
        let sound_power_w = (acoustic_intensity_w_m2 * radiation_area_m2).max(1.0e-18);
        let sound_power_level_db = 10.0 * (sound_power_w / SOUND_POWER_REFERENCE_W).log10();
        // Apply the configured propagation correction (free-field, duct
        // cut-off, or coherent rigid-rectangular-cavity modal coloration) —
        // see `PcFanPropagationMode`.
        let pressure_pa =
            apply_propagation_filter(config, geometry, frequency_hz, source_pressure_pa);
        let sound_pressure_level_db = pressure_level_db(pressure_pa);
        harmonics.push(PcFanAcousticHarmonic {
            harmonic,
            frequency_hz,
            pressure_pa,
            sound_pressure_level_db,
            sound_power_level_db,
        });
    }

    let tip_leakage_spl_db = combine_pressures_db(tip_leakage_pressures.iter().copied());
    let trailing_edge_spl_db = combine_pressures_db(trailing_edge_pressures.iter().copied());
    let leading_edge_inflow_spl_db = combine_pressures_db(leading_edge_pressures.iter().copied());
    let a_weighted_overall_spl_db = a_weighted_overall_db_from_harmonics(&harmonics);

    PcFanAcousticSolution {
        blade_passing_frequency_hz: source.blade_passing_frequency(),
        overall_spl_db: combine_decibels(harmonics.iter().map(|h| h.sound_pressure_level_db)),
        a_weighted_overall_spl_db,
        overall_swl_db: combine_decibels(harmonics.iter().map(|h| h.sound_power_level_db)),
        fixed_hardware_noise,
        harmonics,
        tip_leakage_spl_db,
        trailing_edge_spl_db,
        leading_edge_inflow_spl_db,
    }
}

/// Replace the reduced-order harmonic pressure state while preserving the
/// geometry/flow-derived source bookkeeping used by the objective.
///
/// The acoustic state equation is `p - p_rhs(q, z) = 0`; keeping this helper
/// separate from `evaluate_acoustics` lets the adjoint take partial
/// derivatives with respect to `p` instead of silently folding those terms
/// into a full-objective finite difference.
fn acoustic_solution_with_pressure_state(
    mut acoustics: PcFanAcousticSolution,
    pressures: &[f64],
) -> PcFanAcousticSolution {
    if pressures.len() != acoustics.harmonics.len() {
        return acoustics;
    }

    for (harmonic, &pressure) in acoustics.harmonics.iter_mut().zip(pressures) {
        let pressure = pressure.max(0.0);
        let old_pressure = harmonic.pressure_pa.max(1.0e-12);
        harmonic.pressure_pa = pressure;
        harmonic.sound_pressure_level_db = pressure_level_db(pressure);
        // Keep the diagnostic finite while retaining the physical zero in
        // `harmonic.pressure_pa`.
        harmonic.sound_power_level_db += 20.0 * (pressure.max(1.0e-12) / old_pressure).log10();
    }
    acoustics.overall_spl_db = combine_decibels(
        acoustics
            .harmonics
            .iter()
            .map(|harmonic| harmonic.sound_pressure_level_db),
    );
    acoustics.a_weighted_overall_spl_db =
        a_weighted_overall_db_from_harmonics(&acoustics.harmonics);
    acoustics.overall_swl_db = combine_decibels(
        acoustics
            .harmonics
            .iter()
            .map(|harmonic| harmonic.sound_power_level_db),
    );
    acoustics
}

fn acoustic_reference_distance_m(config: &PcFanConfig) -> f64 {
    config.enclosure.reference_distance_m.max(1.0e-6)
}

/// A-weighted overall SPL from per-harmonic pressures and frequencies.
/// `p_A² = Σ_h α(f_h)² · p_h²` with `α = sound_metrics::a_weighting`. The
/// result is `pressure_level_db(√p_A²)`.
fn a_weighted_overall_db_from_harmonics(harmonics: &[PcFanAcousticHarmonic]) -> f64 {
    let weighted_sq: f64 = harmonics
        .iter()
        .map(|h| {
            let alpha = sound_metrics::a_weighting(h.frequency_hz.max(1.0));
            alpha * alpha * h.pressure_pa * h.pressure_pa
        })
        .sum();
    pressure_level_db(weighted_sq.max(0.0).sqrt())
}

/// Per-harmonic powers (Pa²) used by the tonality penalty. When
/// A-weighting is enabled the powers are scaled by `α(f_h)²` so the
/// tonality metric reflects perceived prominence rather than raw SPL.
fn perceptual_power_spectrum(
    a_weighting_enabled: bool,
    harmonics: &[PcFanAcousticHarmonic],
) -> Vec<f64> {
    harmonics
        .iter()
        .map(|h| {
            let alpha = if a_weighting_enabled {
                sound_metrics::a_weighting(h.frequency_hz.max(1.0))
            } else {
                1.0
            };
            alpha * alpha * h.pressure_pa * h.pressure_pa
        })
        .collect()
}

/// Tip-leakage vortex broadband noise — Fukano-style empirical model.
///
/// Acoustic power scales with `(tip_clearance / r_tip)² · tip_mach^5 ·
/// tip_loading²`. The total power is distributed across harmonics with a
/// peak near `peak_harmonic` (≈ 10·BPF for typical PC fans) and a smooth
/// decay on either side, approximating the broadband hump observed in
/// shrouded-fan measurements.
///
/// Returns one pressure (Pa) per harmonic. Call sites add this in
/// quadrature to the existing tonal + hardware-wake pressures.
fn tip_leakage_pressures_per_harmonic(
    config: &PcFanConfig,
    design: &PcFanDesignVariables,
    geometry: &PcFanGeometry,
    harmonic_count: usize,
) -> Vec<f64> {
    if harmonic_count == 0 || !config.broadband.tip_leakage_enabled {
        return vec![0.0; harmonic_count];
    }
    let tip_radius_m = design.effective_tip_radius_m(config);
    let tip_clearance_m = (geometry.shroud.minimum_radius_m() - tip_radius_m).max(0.0);
    if tip_clearance_m <= 0.0 {
        return vec![0.0; harmonic_count];
    }

    let r_tip = tip_radius_m.max(1.0e-6);
    let omega = std::f64::consts::TAU * config.operating_point.rpm.abs() / 60.0;
    let tip_speed_m_s = omega * r_tip;
    let c_air = config.operating_point.speed_of_sound_m_s;
    let tip_mach = (tip_speed_m_s / c_air).abs();
    if tip_mach <= 0.0 {
        return vec![0.0; harmonic_count];
    }

    // Tip-section loading proxy: chord, twist and camber at the tip station
    // (span_fraction = 1.0). A heavily-loaded tip leaks more, so loading
    // appears squared in the power scaling.
    let tip_section = design.blade_section_at(1.0);
    let chord_ratio = (tip_section.chord_m / r_tip).clamp(0.0, 1.0);
    let twist_factor = (tip_section.twist_deg.to_radians().sin().abs()).clamp(0.0, 1.0);
    let camber_ratio = (tip_section.camber_m / tip_section.chord_m.max(1.0e-6)).clamp(0.0, 0.30);
    let tip_loading =
        (0.20 + 0.55 * twist_factor + 0.50 * camber_ratio + 0.40 * chord_ratio).clamp(0.0, 1.6);

    // Calibration constant. Picked so that a 1% chord clearance at
    // typical tip Mach gives ~30 dB tip-leakage SPL — broadly consistent
    // with PC-fan tip-leakage literature.
    const TIP_LEAKAGE_CONSTANT: f64 = 1.5e-3;
    let clearance_ratio = tip_clearance_m / r_tip;
    let total_power_w = TIP_LEAKAGE_CONSTANT
        * config.operating_point.air_density_kg_m3
        * c_air.powi(3)
        * clearance_ratio.powi(2)
        * tip_mach.powi(5)
        * tip_loading
        * tip_loading
        * std::f64::consts::PI
        * r_tip
        * r_tip;

    // Spectral shape: Gaussian-on-log-frequency centred on `peak_harmonic`
    // with a width of ~one decade. Falls off as ~ −6 dB/oct away from peak.
    let peak_harmonic = 10.0_f64;
    let log_sigma = 1.5_f64; // ≈ one decade FWHM
    let mut weights = Vec::with_capacity(harmonic_count);
    let mut weight_sum = 0.0;
    for h in 1..=harmonic_count {
        let log_h = (h as f64).ln() - peak_harmonic.ln();
        let w = (-(log_h * log_h) / (2.0 * log_sigma * log_sigma)).exp();
        weights.push(w);
        weight_sum += w;
    }
    if weight_sum <= 0.0 {
        return vec![0.0; harmonic_count];
    }

    let reference_distance_m = acoustic_reference_distance_m(config);
    let radiation_area_m2 = 4.0 * std::f64::consts::PI * reference_distance_m.powi(2);
    weights
        .into_iter()
        .map(|w| {
            let band_power_w = total_power_w * (w / weight_sum);
            let intensity_w_m2 = band_power_w / radiation_area_m2.max(1.0e-12);
            let pressure_sq = intensity_w_m2 * config.operating_point.air_density_kg_m3 * c_air;
            pressure_sq.max(0.0).sqrt()
        })
        .collect()
}

/// Brooks–Pope–Marcolini turbulent-boundary-layer / trailing-edge
/// (TBL-TE) self-noise, integrated across the blade span.
///
/// Physics: the turbulent boundary layer over a blade scatters acoustic
/// energy off the trailing edge as a compact dipole. Per section, total
/// acoustic power scales as `ρ · c³ · M⁵ · (δ*/c_chord) · L_span` (BPM
/// dimensional form, NASA RP-1218 1989, Eq. (10) for δ* and Eq. (26)
/// for SPL). The boundary-layer displacement thickness δ* uses the
/// turbulent flat-plate `δ*/c = 0.0337 · Re_c^(−1/5)` correlation —
/// dimensionally identical to BPM's empirical fit and adequate for a
/// reduced-order model.
///
/// The total radiated power is binned across the existing per-harmonic
/// vector by a log-normal spectrum centred on `f_peak = St_peak · U / δ*`
/// with `St_peak = 0.02 · M^(−0.6)` (BPM Fig. 11), producing the same
/// `Vec<f64>` shape the tip-leakage and hardware sources use so the
/// downstream quadrature, directivity and spectral-flatness paths stay
/// unchanged.
///
/// Returns one characteristic pressure (Pa) per harmonic. Call sites
/// add this in quadrature to the existing tonal + tip-leakage +
/// hardware pressures. Returns zeros when
/// `config.broadband.trailing_edge_enabled` is false.
fn trailing_edge_pressures_per_harmonic(
    config: &PcFanConfig,
    design: &PcFanDesignVariables,
    flow: &PcFanFlowSolution,
    harmonic_count: usize,
) -> Vec<f64> {
    if harmonic_count == 0 || !config.broadband.trailing_edge_enabled {
        return vec![0.0; harmonic_count];
    }

    let r_hub = design.effective_hub_radius_m(config).max(1.0e-6);
    let r_tip = design.effective_tip_radius_m(config).max(r_hub + 1.0e-4);
    let omega = std::f64::consts::TAU * config.operating_point.rpm.abs() / 60.0;
    let c_air = config.operating_point.speed_of_sound_m_s.max(1.0);
    let rho = config.operating_point.air_density_kg_m3.max(1.0e-6);
    let mu = config.operating_point.dynamic_viscosity_pa_s.max(1.0e-9);
    let n_blades = config.blade_count.max(1) as f64;
    let v_axial = flow.axial_velocity_m_s.abs();
    let bpf = n_blades * omega / std::f64::consts::TAU;
    if bpf <= 0.0 {
        return vec![0.0; harmonic_count];
    }

    // Span integration: 17 stations matches the resolution used by
    // `mean_section_noise_factor` for consistency.
    let n_stations: usize = 17;
    let span_m = (r_tip - r_hub).max(1.0e-6);

    // BPM amplitude constant. From Curle's equation applied to a TE
    // dipole, p² ∝ ρ²·c⁴·M⁵·(δ*·L/r²). RP-1218's K_1 ≈ -4.31 dB and
    // angular-averaged directivity D̄ ≈ 0.5 fold into a single
    // dimensionless calibration factor tuned so the default 120 mm fan
    // at 2000 rpm produces ≈ 30 dB TE-broadband SPL — consistent with
    // anechoic measurements of low-tip-Mach shrouded axial fans
    // (Brooks–Pope–Marcolini RP-1218; Sharland 1964; Fukano–Jang 2004).
    // At 600 rpm (M_tip ≈ 0.01) the M⁵ scaling drops TE noise toward
    // the floor, realistic for very-quiet operation.
    const TE_AMPLITUDE_K: f64 = 4.0e-3;
    let r_ref_sq = acoustic_reference_distance_m(config).powi(2);

    // Per-station acoustic source pressure², plus per-station log-normal
    // spectral weights centred on f_peak. Accumulate per-harmonic pressure².
    let mut p_sq_per_h = vec![0.0_f64; harmonic_count];
    let log_sigma = 1.2_f64; // FWHM ≈ one decade in frequency

    for i in 0..n_stations {
        let span_fraction = i as f64 / (n_stations - 1) as f64;
        let r = r_hub + span_fraction * span_m;
        let section = design.blade_section_at(span_fraction);
        let chord = section.chord_m.max(1.0e-6);

        // Trapezoidal-rule span weights.
        let edge = i == 0 || i == n_stations - 1;
        let dr = span_m / (n_stations - 1) as f64 * if edge { 0.5 } else { 1.0 };

        // Local relative velocity (rotation + axial).
        let u_rot = omega * r;
        let u_local = (u_rot * u_rot + v_axial * v_axial).sqrt();
        if u_local <= 0.0 {
            continue;
        }
        let m_local = (u_local / c_air).max(1.0e-6);

        // Turbulent flat-plate displacement thickness (BPM-equivalent).
        // δ* ∝ c · Re_c^(-1/5) so doubling chord increases δ* by 2^0.8.
        let re_c = (rho * u_local * chord / mu).max(1.0);
        let delta_star = 0.0337 * chord / re_c.powf(0.2);

        // BPM Fig. 11 peak Strouhal:  St_peak ≈ 0.02 · M^(-0.6).
        let st_peak = 0.02 * m_local.powf(-0.6);
        let f_peak_hz = (st_peak * u_local / delta_star.max(1.0e-12)).max(1.0);

        // BPM RP-1218 source-pressure² (per blade, before observer-
        // distance scaling), Curle's-equation form:
        //
        //     p² = K · ρ² · c⁴ · M⁵ · (δ* · L_span / r_ref²)
        //
        // `δ* · L / r_ref²` is dimensionless; `ρ²·c⁴` carries Pa². Chord
        // dependence enters through δ* ∝ c^0.8 (monotone increasing in
        // chord at fixed local velocity), giving the optimiser correct
        // pressure to trim wide blades.
        let geometric = (delta_star * dr / r_ref_sq).max(0.0);
        let mut total_pressure_sq =
            TE_AMPLITUDE_K * rho * rho * c_air.powi(4) * m_local.powi(5) * geometric;
        // All blades radiate incoherently → multiply by n_blades.
        total_pressure_sq *= n_blades;
        if total_pressure_sq <= 0.0 {
            continue;
        }

        // Log-normal binning around f_peak across harmonic frequencies.
        let log_peak = f_peak_hz.ln();
        let mut weights = Vec::with_capacity(harmonic_count);
        let mut weight_sum = 0.0;
        for h in 1..=harmonic_count {
            let f_h = (h as f64 * bpf).max(1.0);
            let lh = f_h.ln() - log_peak;
            let w = (-(lh * lh) / (2.0 * log_sigma * log_sigma)).exp();
            weights.push(w);
            weight_sum += w;
        }
        if weight_sum <= 0.0 {
            continue;
        }
        for (h_idx, w) in weights.into_iter().enumerate() {
            p_sq_per_h[h_idx] += total_pressure_sq * (w / weight_sum);
        }
    }

    p_sq_per_h
        .into_iter()
        .map(|p_sq| p_sq.max(0.0).sqrt())
        .collect()
}

/// Amiet (1975) leading-edge inflow-turbulence broadband noise.
///
/// Each blade leading edge scatters ingested turbulent eddies into the
/// far-field. Per-section radiated source pressure² follows the Amiet
/// dimensional form (low-Mach asymptote, von Kármán-distributed
/// over harmonics):
///
/// `p²_section(f_h) ∝ TI² · ρ² · c⁴ · M³ · b · L_span · Φ_vK(f_h / f_c)`
///
/// where `b = c_chord/2` is the half-chord, `TI = u'_rms / U_local` is
/// the inflow turbulence intensity, `Φ_vK(x) = (1 + x²)^(−7/6)` is the
/// von Kármán longitudinal-component spectrum shape, and
/// `f_c = U_local / (2π · Λ)` is the cutoff frequency set by the
/// integral length scale `Λ`.
///
/// Returns one source pressure (Pa) per harmonic, on the same
/// "characteristic radiated pressure" footing as the other broadband
/// terms. Returns zeros when `TI == 0`, when the term is disabled, or
/// at zero RPM. Calibration is *gradient-quality-first*: the absolute
/// SPL may differ from anechoic-chamber measurements, but the
/// monotonicity in TI, chord, U_local and Λ is exact, so the
/// optimiser sees correct sensitivity directions.
fn leading_edge_inflow_pressures_per_harmonic(
    config: &PcFanConfig,
    design: &PcFanDesignVariables,
    flow: &PcFanFlowSolution,
    harmonic_count: usize,
) -> Vec<f64> {
    if harmonic_count == 0 || !config.broadband.leading_edge_inflow_enabled {
        return vec![0.0; harmonic_count];
    }
    let ti = config.operating_point.inflow_turbulence_intensity.max(0.0);
    if ti <= 0.0 {
        return vec![0.0; harmonic_count];
    }
    let lambda = config
        .operating_point
        .inflow_integral_length_scale_m
        .max(1.0e-6);

    let r_hub = design.effective_hub_radius_m(config).max(1.0e-6);
    let r_tip = design.effective_tip_radius_m(config).max(r_hub + 1.0e-4);
    let omega = std::f64::consts::TAU * config.operating_point.rpm.abs() / 60.0;
    if omega <= 0.0 {
        return vec![0.0; harmonic_count];
    }
    let c_air = config.operating_point.speed_of_sound_m_s.max(1.0);
    let rho = config.operating_point.air_density_kg_m3.max(1.0e-6);
    let n_blades = config.blade_count.max(1) as f64;
    let v_axial = flow.axial_velocity_m_s.abs();
    let bpf = n_blades * omega / std::f64::consts::TAU;
    if bpf <= 0.0 {
        return vec![0.0; harmonic_count];
    }

    // Calibration anchor. Same "characteristic source pressure"
    // convention as TE / tip-leakage / RSI; constant tuned so that the
    // default 120 mm fan at 600 rpm with 10 % TI produces a few-tens-
    // of-dB LE-noise level. Absolute level is empirical; relative
    // scaling is the priority (gradient quality).
    // Dimensionless calibration. The dimensional pressure scale below is
    // rho²*c⁴; the previous rho²*c² expression was not pressure squared.
    // 8.5e-6 preserves the default-air calibration after restoring c².
    const LE_AMPLITUDE_K: f64 = 8.5e-6;

    let n_stations: usize = 17;
    let span_m = (r_tip - r_hub).max(1.0e-6);
    let r_ref_sq = acoustic_reference_distance_m(config).powi(2);

    let mut p_sq_per_h = vec![0.0_f64; harmonic_count];

    for i in 0..n_stations {
        let span_fraction = i as f64 / (n_stations - 1) as f64;
        let r = r_hub + span_fraction * span_m;
        let section = design.blade_section_at(span_fraction);
        let chord = section.chord_m.max(1.0e-6);
        let half_chord = 0.5 * chord;

        let edge = i == 0 || i == n_stations - 1;
        let dr = span_m / (n_stations - 1) as f64 * if edge { 0.5 } else { 1.0 };

        let u_rot = omega * r;
        let u_local = (u_rot * u_rot + v_axial * v_axial).sqrt();
        if u_local <= 0.0 {
            continue;
        }
        let m_local = (u_local / c_air).max(1.0e-6);

        // Per-section integrated amplitude (Amiet dimensional form):
        //   p²_section ∝ TI² · ρ² · c⁴ · M³ · b · L_span / r²
        let p_sq_total = LE_AMPLITUDE_K
            * ti
            * ti
            * rho
            * rho
            * c_air
            * c_air
            * c_air
            * c_air
            * m_local.powi(3)
            * half_chord
            * dr
            * n_blades
            / r_ref_sq;
        if p_sq_total <= 0.0 {
            continue;
        }

        // von Kármán longitudinal spectrum cutoff frequency.
        let f_cutoff = (u_local / (std::f64::consts::TAU * lambda)).max(1.0);

        // Distribute total over BPF harmonics by Φ_vK(f/f_c) =
        // (1 + (f/f_c)²)^(−7/6).
        let mut weights = Vec::with_capacity(harmonic_count);
        let mut weight_sum = 0.0;
        for h in 1..=harmonic_count {
            let f_h = h as f64 * bpf;
            let ratio = f_h / f_cutoff;
            let w = (1.0 + ratio * ratio).powf(-7.0 / 6.0);
            weights.push(w);
            weight_sum += w;
        }
        if weight_sum <= 0.0 {
            continue;
        }
        for (h_idx, w) in weights.into_iter().enumerate() {
            p_sq_per_h[h_idx] += p_sq_total * (w / weight_sum);
        }
    }

    p_sq_per_h
        .into_iter()
        .map(|p_sq| p_sq.max(0.0).sqrt())
        .collect()
}

/// Closed-form aerodynamic-response helpers for unsteady-loading
/// noise (Sears 1941 gust response, Theodorsen 1935 oscillating-airfoil
/// response). Used by the rotor–stator interaction model below; will be
/// reused for Amiet leading-edge inflow-turbulence noise (Phase 4).
mod aerodynamic_response {
    /// Bessel function `J_0(x)` — Abramowitz & Stegun 9.4.1 (|x|≤3) and
    /// 9.4.2 asymptotic (|x|>3). Accurate to ≲ 1e-7 over the range
    /// relevant to acoustic reduced frequencies (typically `k ≤ 10`).
    pub fn bessel_j0(x: f64) -> f64 {
        let ax = x.abs();
        if ax < 3.0 {
            let y = (x / 3.0).powi(2);
            1.0 - 2.2499997 * y + 1.2656208 * y.powi(2) - 0.3163866 * y.powi(3)
                + 0.0444479 * y.powi(4)
                - 0.0039444 * y.powi(5)
                + 0.0002100 * y.powi(6)
        } else {
            let z = 3.0 / ax;
            let f = 0.79788456 - 0.00000077 * z - 0.00552740 * z.powi(2) - 0.00009512 * z.powi(3)
                + 0.00137237 * z.powi(4)
                - 0.00072805 * z.powi(5)
                + 0.00014476 * z.powi(6);
            let theta = ax - std::f64::consts::FRAC_PI_4 - 0.04166397 * z - 0.00003954 * z.powi(2)
                + 0.00262573 * z.powi(3)
                - 0.00054125 * z.powi(4)
                - 0.00029333 * z.powi(5)
                + 0.00013558 * z.powi(6);
            f * theta.cos() / ax.sqrt()
        }
    }

    /// Bessel function `J_1(x)` — Abramowitz & Stegun 9.4.4/9.4.6.
    pub fn bessel_j1(x: f64) -> f64 {
        let ax = x.abs();
        let s = if x < 0.0 { -1.0 } else { 1.0 };
        if ax < 3.0 {
            let y = (x / 3.0).powi(2);
            x * (0.5 - 0.56249985 * y + 0.21093573 * y.powi(2) - 0.03954289 * y.powi(3)
                + 0.00443319 * y.powi(4)
                - 0.00031761 * y.powi(5)
                + 0.00001109 * y.powi(6))
        } else {
            let z = 3.0 / ax;
            let f = 0.79788456 + 0.00000156 * z + 0.01659667 * z.powi(2) + 0.00017105 * z.powi(3)
                - 0.00249511 * z.powi(4)
                + 0.00113653 * z.powi(5)
                - 0.00020033 * z.powi(6);
            let theta =
                ax - 3.0 * std::f64::consts::FRAC_PI_4 + 0.12499612 * z + 0.00005650 * z.powi(2)
                    - 0.00637879 * z.powi(3)
                    + 0.00074348 * z.powi(4)
                    + 0.00079824 * z.powi(5)
                    - 0.00029166 * z.powi(6);
            s * f * theta.cos() / ax.sqrt()
        }
    }

    /// Theodorsen oscillating-airfoil function `C(k)` (Hassig 1971
    /// 4-pole approximation, accurate to ≲ 0.5 % for `k ∈ [0, 5]`):
    ///
    /// `C(k) = 1 − 0.165/(1 − 0.0455 i/k) − 0.335/(1 − 0.3 i/k)`
    ///
    /// Returns `(real, imag)`. At `k → 0`, `C → 1`; at `k → ∞`,
    /// `C → 0.5`.
    pub fn theodorsen(k: f64) -> (f64, f64) {
        if k < 1.0e-9 {
            return (1.0, 0.0);
        }
        // Compute a / (1 − b i / k) = a · (1 + b i/k) / (1 + (b/k)²).
        let term = |a: f64, b: f64| -> (f64, f64) {
            let bk = b / k;
            let denom = 1.0 + bk * bk;
            (a / denom, a * bk / denom)
        };
        let (re1, im1) = term(0.165, 0.0455);
        let (re2, im2) = term(0.335, 0.3);
        (1.0 - re1 - re2, -im1 - im2)
    }

    /// Magnitude of the Sears function `S(k)` (Sears 1941). Describes
    /// the unsteady-lift response of a thin airfoil to a sinusoidal
    /// upwash gust at reduced frequency `k = ω·c/(2U)`.
    ///
    /// `S(k) = [J_0(k) − i·J_1(k)] · C(k) + i·J_1(k)`
    ///
    /// At `k → 0`, `|S| → 1` (quasi-steady limit). At `k → ∞`,
    /// `|S| → 1/√(2πk)` (high-frequency asymptote — strong rolloff).
    pub fn sears_magnitude(k: f64) -> f64 {
        if k < 1.0e-9 {
            return 1.0;
        }
        let j0 = bessel_j0(k);
        let j1 = bessel_j1(k);
        let (cre, cim) = theodorsen(k);
        // Expand S = (J0 - iJ1)·(Cre + iCim) + iJ1
        //         = J0·Cre + J1·Cim + i·(J0·Cim - J1·Cre + J1)
        let sre = j0 * cre + j1 * cim;
        let sim = j0 * cim - j1 * cre + j1;
        (sre * sre + sim * sim).sqrt()
    }
}

/// Rotor–stator interaction (Sears) tonal noise.
///
/// Each blade traverses `S` strut wakes per revolution, encountering a
/// periodic upwash of fundamental frequency `S·Ω`. The unsteady lift
/// per unit span follows Sears' gust response:
///
/// `L̃(ω_h) = π · ρ · U · c · Δv_n · |S(k_red)|`
///
/// where `n` is the matched (Tyler–Sofrin) wake harmonic
/// `n_m = round(m · B / S)` for the m-th BPF tone, `Δv_n` is the
/// n-th azimuthal Fourier coefficient of the strut-wake velocity
/// deficit, and `k_red = ω_h · c / (2U)` is the reduced frequency.
///
/// The tonal pressure for the m-th BPF harmonic is the compact-dipole
/// far-field of `L̃`, integrated coherently across the compact span and
/// across blades at a BPF harmonic:
///
/// `p_m ∝ B · Σ_stations L̃_m · ω_m / (c_air · r_ref)`
///
/// The strut-wake amplitude `Δv_0` decays with rotor-strut axial gap
/// as `Δv = Δv_0 / √(1 + gap/c_strut)` (mixing-length similarity), so
/// increasing the gap is the dominant lever for suppressing BPF tones.
///
/// Returns one source pressure (Pa) per harmonic, on the same
/// "characteristic radiated pressure" footing as
/// `tip_leakage_pressures_per_harmonic` and
/// `trailing_edge_pressures_per_harmonic`.
fn rotor_stator_interaction_pressures_per_harmonic(
    config: &PcFanConfig,
    design: &PcFanDesignVariables,
    flow: &PcFanFlowSolution,
    harmonic_count: usize,
) -> Vec<f64> {
    if harmonic_count == 0 {
        return vec![0.0; harmonic_count];
    }

    let b_blades = config.blade_count.max(1) as f64;
    // 2 crossed bars × 2 wake events per bar pass = 4 wakes per revolution
    // for the default X-support (matches the existing strut_count_eff
    // convention used by `fixed_hardware_noise_pressures` broadband path
    // and `observer_spl_db` directivity).
    let s_struts = (SUPPORT_STRUT_COUNT * 2) as f64;
    let omega = std::f64::consts::TAU * config.operating_point.rpm.abs() / 60.0;
    if omega <= 0.0 {
        return vec![0.0; harmonic_count];
    }
    let c_air = config.operating_point.speed_of_sound_m_s.max(1.0);
    let rho = config.operating_point.air_density_kg_m3.max(1.0e-6);
    let r_hub = design.effective_hub_radius_m(config).max(1.0e-6);
    let r_tip = design.effective_tip_radius_m(config).max(r_hub + 1.0e-4);

    let strut_chord = config.dimensions.support_strut_width_m.max(1.0e-4);
    let axial_gap = config.dimensions.rotor_strut_axial_gap_m.max(0.0);
    let wake_deficit_0 = config
        .dimensions
        .strut_wake_centerline_deficit
        .clamp(0.0, 1.0);
    let wake_half_width = config.dimensions.strut_wake_half_width_m.max(1.0e-4);
    // Wake decay between rotor and strut (mixing-length similarity).
    let gap_decay = (1.0 + axial_gap / strut_chord).sqrt().recip();
    let wake_deficit = wake_deficit_0 * gap_decay;
    if wake_deficit <= 0.0 {
        return vec![0.0; harmonic_count];
    }

    // Calibration anchor: tuned so the default 120 mm fan at 600 rpm
    // produces a per-blade BPF tonal level comparable to the legacy
    // empirical formula it replaces (a few dB above the broadband floor),
    // matching Goldstein-style RSI predictions for the default
    // 8-blade / 4-strut arrangement.
    const RSI_CALIBRATION: f64 = 7.35e-5;

    let n_stations: usize = 17;
    let span_m = (r_tip - r_hub).max(1.0e-6);
    let r_ref = acoustic_reference_distance_m(config);
    let v_axial = flow.axial_velocity_m_s.abs();

    let mut p_per_h = vec![0.0_f64; harmonic_count];

    for i in 0..n_stations {
        let span_fraction = i as f64 / (n_stations - 1) as f64;
        let r = r_hub + span_fraction * span_m;
        let section = design.blade_section_at(span_fraction);
        let chord = section.chord_m.max(1.0e-6);

        let edge = i == 0 || i == n_stations - 1;
        let dr = span_m / (n_stations - 1) as f64 * if edge { 0.5 } else { 1.0 };

        let u_rot = omega * r;
        let u_local = (u_rot * u_rot + v_axial * v_axial).sqrt();
        if u_local <= 0.0 {
            continue;
        }

        for h in 1..=harmonic_count {
            let h_f = h as f64;
            // Tyler–Sofrin matched wake harmonic.
            let n_wake = ((h_f * b_blades) / s_struts).round().max(0.0);
            if n_wake <= 0.0 {
                continue;
            }
            // Gaussian-wake n-th azimuthal Fourier coefficient (relative
            // to centerline deficit) — derived from periodic Gaussian
            // expansion with period 2π/S and half-width b at radius r.
            let bb = wake_half_width;
            let arg = n_wake * s_struts * bb / (2.0 * r);
            let kernel = (s_struts * bb / (r * std::f64::consts::PI.sqrt())) * (-arg * arg).exp();
            let dv = wake_deficit * kernel * u_local;
            if dv <= 0.0 {
                continue;
            }

            // Sears reduced frequency at the m-th BPF harmonic.
            let omega_h = h_f * b_blades * omega;
            let k_red = omega_h * chord / (2.0 * u_local);
            let sears_mag = aerodynamic_response::sears_magnitude(k_red);

            // Unsteady lift per unit span × span element width.
            let l_segment = std::f64::consts::PI * rho * u_local * chord * dv * sears_mag * dr;

            // Compact-dipole far-field pressure from this span segment.
            // A deterministic rotor-stator tone is phase locked. Summing
            // segment powers would make the result shrink as 1/N when the
            // span quadrature is refined.
            p_per_h[h - 1] += l_segment * omega_h / (c_air * r_ref);
        }
    }

    p_per_h
        .into_iter()
        .map(|p| (p * b_blades * RSI_CALIBRATION.sqrt()).abs())
        .collect()
}

/// First non-trivial radial-mode eigenvalue of a hard-walled circular
/// duct: first zero of `J_1'(α)` (NB: derivative, since hard walls
/// require `∂J_m/∂r = 0` at the wall). Tabulated value α_10 = 1.8412.
const DUCT_MODE_10_EIGENVALUE: f64 = 1.8412;

/// Plane-wave-vs-(1,0) modal split assumed by the first-order duct
/// cut-off filter. A real source is decomposable into all `(m, n)`
/// modes, but for the dominant rotor-strut interaction tone of a
/// PC fan the next-strongest mode is (1,0). 0.5 is the simplest
/// neutral choice; tunable later if measurements warrant it.
const DUCT_CUTOFF_MODAL_FRACTION: f64 = 0.5;

/// Hard-walled circular-duct cut-off attenuation factor (linear, i.e.
/// applied to pressure not pressure²) for the (1,0) mode at the given
/// frequency.
///
/// Above the cut-on frequency `f_10 = α_10 · c / (2π · R)`, the mode
/// propagates and the factor is 1.0. Below cut-on, the axial
/// wavenumber is imaginary and the mode decays as
/// `exp(−|k_z|·L_shroud)` with `|k_z| = √((α_10/R)² − (ω/c)²)`.
///
/// Applies only to the (1,0) modal *component* of the source — the
/// plane-wave (0,0) component is unaffected. The caller blends
/// per-harmonic pressures using `DUCT_CUTOFF_MODAL_FRACTION`.
fn duct_cutoff_attenuation(
    frequency_hz: f64,
    shroud_inner_radius_m: f64,
    shroud_axial_length_m: f64,
    c_air: f64,
) -> f64 {
    if frequency_hz <= 0.0
        || shroud_inner_radius_m <= 0.0
        || shroud_axial_length_m <= 0.0
        || c_air <= 0.0
    {
        return 1.0;
    }
    let omega = std::f64::consts::TAU * frequency_hz;
    let k = omega / c_air;
    let kr = DUCT_MODE_10_EIGENVALUE / shroud_inner_radius_m;
    if k >= kr {
        return 1.0;
    }
    let kz = (kr * kr - k * k).sqrt();
    (-kz * shroud_axial_length_m).exp()
}

/// Apply the first-order duct cut-off filter to a source pressure (Pa)
/// at the given harmonic frequency. Returns the filtered pressure.
///
/// `p_filtered² = p_source² · [(1 − f) + f · A²]`
///
/// where `A` is `duct_cutoff_attenuation` and `f` is
/// `DUCT_CUTOFF_MODAL_FRACTION`.
#[cfg(test)]
fn apply_duct_cutoff_filter(config: &PcFanConfig, frequency_hz: f64, pressure_pa: f64) -> f64 {
    apply_duct_cutoff_filter_for_dimensions(
        config,
        frequency_hz,
        config.dimensions.shroud_inner_radius_m,
        config.dimensions.depth_m,
        pressure_pa,
    )
}

fn apply_duct_cutoff_filter_for_geometry(
    config: &PcFanConfig,
    geometry: &PcFanGeometry,
    frequency_hz: f64,
    pressure_pa: f64,
) -> f64 {
    apply_duct_cutoff_filter_for_dimensions(
        config,
        frequency_hz,
        geometry.shroud.minimum_radius_m(),
        geometry.shroud.depth_m,
        pressure_pa,
    )
}

fn apply_duct_cutoff_filter_for_dimensions(
    config: &PcFanConfig,
    frequency_hz: f64,
    shroud_inner_radius_m: f64,
    shroud_axial_length_m: f64,
    pressure_pa: f64,
) -> f64 {
    if !config.broadband.duct_cutoff_enabled {
        return pressure_pa;
    }
    let attn = duct_cutoff_attenuation(
        frequency_hz,
        shroud_inner_radius_m,
        shroud_axial_length_m,
        config.operating_point.speed_of_sound_m_s,
    );
    let f = DUCT_CUTOFF_MODAL_FRACTION;
    let factor = (1.0 - f) + f * attn * attn;
    pressure_pa * factor.max(0.0).sqrt()
}

/// Linear (pressure) propagation gain at `frequency_hz` for the
/// configured `enclosure.mode`. Multiply per-harmonic source pressure
/// by this number to get the listener-position pressure.
///
/// - `FreeField`: 1.0 unconditionally.
/// - `DuctCutoff`: legacy `apply_duct_cutoff_filter` factor (i.e. the
///   ratio applied to `pressure_pa = 1`).
/// - `RigidCavity`: coherent rigid-rectangular-cavity modal coloration.
#[cfg(test)]
fn propagation_gain(config: &PcFanConfig, frequency_hz: f64) -> f64 {
    match config.enclosure.mode {
        PcFanPropagationMode::FreeField => 1.0,
        PcFanPropagationMode::DuctCutoff => apply_duct_cutoff_filter(config, frequency_hz, 1.0),
        PcFanPropagationMode::RigidCavity => rigid_cavity_pressure_gain(config, frequency_hz),
    }
}

fn propagation_gain_for_geometry(
    config: &PcFanConfig,
    geometry: &PcFanGeometry,
    frequency_hz: f64,
) -> f64 {
    match config.enclosure.mode {
        PcFanPropagationMode::FreeField => 1.0,
        PcFanPropagationMode::DuctCutoff => {
            apply_duct_cutoff_filter_for_geometry(config, geometry, frequency_hz, 1.0)
        }
        PcFanPropagationMode::RigidCavity => rigid_cavity_pressure_gain(config, frequency_hz),
    }
}

/// Apply the configured propagation correction to a source pressure.
/// Generalisation of `apply_duct_cutoff_filter` over the propagation
/// modes defined in `PcFanPropagationMode`.
fn apply_propagation_filter(
    config: &PcFanConfig,
    geometry: &PcFanGeometry,
    frequency_hz: f64,
    pressure_pa: f64,
) -> f64 {
    pressure_pa * propagation_gain_for_geometry(config, geometry, frequency_hz)
}

/// Rigid-rectangular-cavity pressure gain at `frequency_hz`.
///
/// Engineering modal-coloration model: starts from a free-field baseline
/// and coherently adds the complex response from each cavity mode whose
/// resonance frequency `ω_{mnp} = π·c·√((m/Lx)² + (n/Ly)² + (p/Lz)²)`
/// is excited by the source. The gain at frequency `ω` is
///
/// `H(ω) = 1 + Σ C_mnp · [iηr / (1 − r² + iηr)] / √η`, where
/// `r = ω/ω_mnp`. The signed source/listener coupling `C_mnp` retains
/// modal phase, so modes can amplify or cancel. The previous sum of
/// squared couplings forced every mode to add energy and could never
/// represent an antiresonance.
///
/// This is the Phase-3 closed-form chassis correction: not a full BEM
/// solve (Phase 5 territory) but enough for the optimiser to see modal
/// coloration and place blade-passing harmonics away from chassis
/// resonances. The DC (0,0,0) mode is excluded — for a closed cavity it
/// would contribute a static-stiffness divergence at ω→0 that has no
/// physical meaning for a real chassis with vents.
fn rigid_cavity_pressure_gain(config: &PcFanConfig, frequency_hz: f64) -> f64 {
    if frequency_hz <= 0.0 {
        return 1.0;
    }
    let cfg = &config.enclosure;
    let [lx, ly, lz] = cfg.chassis_dimensions_m;
    if lx <= 0.0 || ly <= 0.0 || lz <= 0.0 {
        return 1.0;
    }
    let c = config.operating_point.speed_of_sound_m_s.max(1.0);
    let omega = std::f64::consts::TAU * frequency_hz;
    let eta = cfg.modal_damping.max(1.0e-4);
    let max_idx = cfg.max_mode_index;
    let [xs, ys, zs] = cfg.source_position_m;
    let [xl, yl, zl] = cfg.listener_position_m;

    let mut transfer_re = 1.0_f64;
    let mut transfer_im = 0.0_f64;
    for m in 0..=max_idx {
        let kx = m as f64 * std::f64::consts::PI / lx;
        let psi_s_x = (kx * xs).cos();
        let psi_l_x = (kx * xl).cos();
        let eps_x = if m == 0 { 1.0 } else { 2.0 };
        for n in 0..=max_idx {
            let ky = n as f64 * std::f64::consts::PI / ly;
            let psi_s_y = (ky * ys).cos();
            let psi_l_y = (ky * yl).cos();
            let eps_y = if n == 0 { 1.0 } else { 2.0 };
            for p in 0..=max_idx {
                if m == 0 && n == 0 && p == 0 {
                    continue;
                }
                let kz = p as f64 * std::f64::consts::PI / lz;
                let psi_s_z = (kz * zs).cos();
                let psi_l_z = (kz * zl).cos();
                let eps_z = if p == 0 { 1.0 } else { 2.0 };
                let k_mnp_sq = kx * kx + ky * ky + kz * kz;
                let omega_mnp = c * k_mnp_sq.sqrt();
                if omega_mnp <= 0.0 {
                    continue;
                }
                // Signed, normalized source-to-listener coupling. Keeping
                // the cosine signs is essential for modal interference.
                let eps_total = eps_x * eps_y * eps_z;
                let coupling =
                    (eps_total / 8.0) * psi_s_x * psi_s_y * psi_s_z * psi_l_x * psi_l_y * psi_l_z;
                let r = omega / omega_mnp;
                let damp = eta * r;
                let detune = 1.0 - r * r;
                let denom = detune * detune + damp * damp;
                if denom <= f64::EPSILON {
                    continue;
                }
                let modal_re = damp * damp / denom;
                let modal_im = damp * detune / denom;
                let scale = coupling / eta.sqrt();
                transfer_re += scale * modal_re;
                transfer_im += scale * modal_im;
            }
        }
    }
    transfer_re.hypot(transfer_im)
}

fn acoustic_pressure_rhs(
    config: &PcFanConfig,
    design: &PcFanDesignVariables,
    geometry: &PcFanGeometry,
    airflow_cfm: f64,
) -> Vec<f64> {
    let flow = flow_from_airflow_cfm(config, design, geometry, airflow_cfm);
    let blade_pressures = blade_acoustic_pressures(config, design, geometry, &flow);
    let fixed_hardware_pressures =
        fixed_hardware_noise_pressures(config, design, geometry, &flow, blade_pressures.len());
    let tip_leakage_pressures =
        tip_leakage_pressures_per_harmonic(config, design, geometry, blade_pressures.len());
    // BPM trailing-edge broadband. Included so the adjoint shape-FD loop
    // (`lagrangian_shape_gradient`) and the flow-derivative FD
    // (`acoustic_pressure_rhs_flow_derivative`) pick up TE-noise
    // sensitivities automatically — without this term, the analytic
    // adjoint computes a gradient that disagrees with FD on
    // `evaluate_pc_fan`'s objective whenever TE is enabled.
    let trailing_edge_pressures =
        trailing_edge_pressures_per_harmonic(config, design, &flow, blade_pressures.len());
    // Amiet leading-edge inflow-turbulence broadband. Mirrors the
    // primal `evaluate_acoustics` so the adjoint shape-FD loop picks up
    // chord/twist sensitivities of the LE term automatically. Returns
    // zeros at the default operating point (TI = 0) so this is a no-op
    // until the user opts in.
    let leading_edge_pressures =
        leading_edge_inflow_pressures_per_harmonic(config, design, &flow, blade_pressures.len());
    // Source for harmonic frequency lookup so the duct cut-off filter
    // sees the same `frequency_hz = h · BPF` as `evaluate_acoustics`.
    let source = propeller_source_for(config, design, &flow);

    (0..blade_pressures.len())
        .map(|i| {
            let hardware_pressure = fixed_hardware_pressures[i].total_pressure_pa();
            let pressure = (blade_pressures[i] * blade_pressures[i]
                + hardware_pressure * hardware_pressure
                + tip_leakage_pressures[i] * tip_leakage_pressures[i]
                + trailing_edge_pressures[i] * trailing_edge_pressures[i]
                + leading_edge_pressures[i] * leading_edge_pressures[i])
                .sqrt();
            let frequency_hz = source.harmonic_frequency(i + 1);
            apply_propagation_filter(config, geometry, frequency_hz, pressure)
        })
        .collect()
}

fn blade_acoustic_pressures(
    config: &PcFanConfig,
    design: &PcFanDesignVariables,
    geometry: &PcFanGeometry,
    flow: &PcFanFlowSolution,
) -> Vec<f64> {
    let source = propeller_source_for(config, design, flow);
    let harmonic_count = config.mesh.harmonic_count.max(1);
    let amplitudes = source
        .harmonic_amplitudes_at_distance(harmonic_count, acoustic_reference_distance_m(config));
    let shape_factor = acoustic_shape_factor(config, design, geometry, flow);

    amplitudes
        .into_iter()
        // PropellerSource returns peak phasors; every other source term in
        // this module is RMS pressure, and SPL is defined from RMS.
        .map(|amplitude| (amplitude.abs() * shape_factor / 2.0_f64.sqrt()).max(0.0))
        .collect()
}

#[derive(Debug, Clone, Copy)]
struct FixedHardwarePressure {
    support_wake_pressure_pa: f64,
    motor_mount_pressure_pa: f64,
    support_wake_broadband_pa: f64,
    motor_mount_broadband_pa: f64,
}

impl FixedHardwarePressure {
    fn total_pressure_pa(self) -> f64 {
        (self.support_wake_pressure_pa * self.support_wake_pressure_pa
            + self.motor_mount_pressure_pa * self.motor_mount_pressure_pa
            + self.support_wake_broadband_pa * self.support_wake_broadband_pa
            + self.motor_mount_broadband_pa * self.motor_mount_broadband_pa)
            .sqrt()
    }
}

fn fixed_hardware_noise_pressures(
    config: &PcFanConfig,
    design: &PcFanDesignVariables,
    geometry: &PcFanGeometry,
    flow: &PcFanFlowSolution,
    harmonic_count: usize,
) -> Vec<FixedHardwarePressure> {
    let effects = flow.fixed_hardware;
    let tip_speed_m_s = std::f64::consts::TAU * config.operating_point.rpm / 60.0
        * design.effective_tip_radius_m(config);
    let dynamic_pressure_pa =
        0.5 * config.operating_point.air_density_kg_m3 * tip_speed_m_s * tip_speed_m_s;
    let strut_angle_factor = mean_support_strut_angle_factor(geometry);
    let empirical_reference_gain = acoustic_reference_distance_m(config).recip();

    // Sears RSI tonal — physics-based replacement for the empirical
    // `0.45/√h` decay below. Scales with axial gap, wake deficit, blade
    // chord per radius, and the Tyler–Sofrin matched wake harmonic.
    let rsi_pressures =
        rotor_stator_interaction_pressures_per_harmonic(config, design, flow, harmonic_count);
    // Modulate by the existing strut-angle factor so radial/azimuthal
    // strut alignment continues to influence the tonal level
    // monotonically (preserves the existing optimisation incentives the
    // empirical model exposed).
    let strut_angle_modifier = 1.0 + 0.35 * strut_angle_factor;

    // Broadband spectral shape: log-normal in harmonic index, peaked at the
    // strut-wake passing harmonic (≈ 4 wakes per revolution → harmonic 4 of
    // BPF/blade for 8-blade fan), with a one-decade FWHM.
    let broadband_enabled = config.broadband.strut_motor_enabled;
    let strut_count_eff = (SUPPORT_STRUT_COUNT * 2) as f64; // 2 crossing bars → 4 wakes
    let blade_count_f = config.blade_count as f64;
    let support_peak_h = (strut_count_eff / blade_count_f).max(1.0);
    let motor_peak_h = 1.0_f64; // motor-mount buffeting concentrates near BPF
    let log_sigma = 1.2_f64;

    let support_weights: Vec<f64> = (1..=harmonic_count)
        .map(|h| {
            let lh = (h as f64).ln() - support_peak_h.ln();
            (-(lh * lh) / (2.0 * log_sigma * log_sigma)).exp()
        })
        .collect();
    let motor_weights: Vec<f64> = (1..=harmonic_count)
        .map(|h| {
            let lh = (h as f64).ln() - motor_peak_h.ln();
            (-(lh * lh) / (2.0 * log_sigma * log_sigma)).exp()
        })
        .collect();
    let support_weight_sum: f64 = support_weights.iter().sum();
    let motor_weight_sum: f64 = motor_weights.iter().sum();

    // Total broadband pressure squared, scaled like the tonal source but
    // distributed over all harmonics. Calibration constants tuned so the
    // broadband adds roughly 3–5 dB to the existing tonal hardware noise
    // at typical operating points.
    let support_total_pressure_sq = if broadband_enabled {
        let p = dynamic_pressure_pa
            * effects.strut_blockage_ratio
            * (1.0 + 0.35 * strut_angle_factor)
            * 6.0e-4;
        p * p
    } else {
        0.0
    };
    let motor_total_pressure_sq = if broadband_enabled {
        let p = dynamic_pressure_pa
            * effects.motor_blockage_ratio
            * effects.support_wake_loss_factor.recip().min(1.6)
            * 2.5e-4;
        p * p
    } else {
        0.0
    };

    (1..=harmonic_count)
        .map(|harmonic| {
            let harmonic_f = harmonic as f64;
            let support_wake_pressure_pa = rsi_pressures[harmonic - 1] * strut_angle_modifier;
            let motor_mount_pressure_pa = dynamic_pressure_pa
                * effects.motor_blockage_ratio
                * effects.support_wake_loss_factor.recip().min(1.6)
                * 1.4e-4
                / harmonic_f.sqrt()
                * empirical_reference_gain;

            // Broadband: distribute total power across harmonics by the
            // log-normal weights, then convert per-harmonic power → pressure.
            let support_band_w = if support_weight_sum > 0.0 {
                support_weights[harmonic - 1] / support_weight_sum
            } else {
                0.0
            };
            let motor_band_w = if motor_weight_sum > 0.0 {
                motor_weights[harmonic - 1] / motor_weight_sum
            } else {
                0.0
            };
            let support_wake_broadband_pa =
                (support_total_pressure_sq * support_band_w).max(0.0).sqrt()
                    * empirical_reference_gain;
            let motor_mount_broadband_pa =
                (motor_total_pressure_sq * motor_band_w).max(0.0).sqrt() * empirical_reference_gain;

            FixedHardwarePressure {
                support_wake_pressure_pa,
                motor_mount_pressure_pa,
                support_wake_broadband_pa,
                motor_mount_broadband_pa,
            }
        })
        .collect()
}

pub(crate) fn propeller_source_for(
    config: &PcFanConfig,
    design: &PcFanDesignVariables,
    flow: &PcFanFlowSolution,
) -> PropellerSource {
    let hub_radius_m = design.effective_hub_radius_m(config);
    let tip_radius_m = design.effective_tip_radius_m(config);
    let disk_area_m2 = std::f64::consts::PI * (tip_radius_m.powi(2) - hub_radius_m.powi(2));
    // Zero aerodynamic loading must remain zero. A positive floor here
    // fabricated tonal pressure at free-flow/zero-speed states and made the
    // reduced acoustic state equation nonsmooth for no physical reason.
    let thrust = (flow.pressure_rise_pa * disk_area_m2).max(0.0);
    let torque = (thrust * tip_radius_m * 0.08).max(0.0);

    PropellerSource {
        n_blades: config.blade_count,
        rpm: config.operating_point.rpm,
        thrust,
        torque,
        tip_radius: tip_radius_m,
        hub_radius: hub_radius_m,
        speed_of_sound: config.operating_point.speed_of_sound_m_s,
        density: config.operating_point.air_density_kg_m3,
    }
}

fn primal_residual(
    config: &PcFanConfig,
    design: &PcFanDesignVariables,
    geometry: &PcFanGeometry,
    flow: &PcFanFlowSolution,
    pressures: &[f64],
) -> PcFanPrimalResidual {
    let airflow_residual = flow.airflow_cfm - mean_flow_airflow_rhs(config, design, geometry);
    let pressure_rhs = acoustic_pressure_rhs(config, design, geometry, flow.airflow_cfm);
    let acoustic_pressure_pa = pressures
        .iter()
        .zip(pressure_rhs.iter())
        .map(|(state, rhs)| state - rhs)
        .collect::<Vec<_>>();
    let max_abs = acoustic_pressure_pa
        .iter()
        .fold(airflow_residual.abs(), |max_abs, residual| {
            max_abs.max(residual.abs())
        });

    PcFanPrimalResidual {
        airflow_cfm: airflow_residual,
        acoustic_pressure_pa,
        max_abs,
    }
}

fn finite_difference_shape_objective_gradient(
    config: &PcFanConfig,
    design: &PcFanDesignVariables,
) -> PcFanDesignVector {
    let mut gradient = [0.0; DESIGN_VARIABLE_COUNT];
    let values = design.to_vector();
    let evaluate_component = |index: usize| {
        let step = design_variable_step(config, index, values[index]);
        let plus_design = perturbed_design(design, index, step);
        let minus_design = perturbed_design(design, index, -step);
        let (plus, minus) = rayon::join(
            || evaluate_pc_fan(config, &plus_design).objective,
            || evaluate_pc_fan(config, &minus_design).objective,
        );
        (plus - minus) / (2.0 * step)
    };
    let components: Vec<f64> = if crate::parallelism::use_pcfan_parallel(
        DESIGN_VARIABLE_COUNT,
        crate::parallelism::PCFAN_FD_COMPONENTS_PER_THREAD,
    ) {
        (0..DESIGN_VARIABLE_COUNT)
            .into_par_iter()
            .map(evaluate_component)
            .collect()
    } else {
        (0..DESIGN_VARIABLE_COUNT).map(evaluate_component).collect()
    };
    gradient.copy_from_slice(&components);

    gradient
}

fn perturbed_design(
    design: &PcFanDesignVariables,
    index: usize,
    delta: f64,
) -> PcFanDesignVariables {
    let mut values = design.to_vector();
    values[index] += delta;
    PcFanDesignVariables::from_vector(values)
}

fn design_variable_step(config: &PcFanConfig, index: usize, value: f64) -> f64 {
    config.optimizer.finite_difference_step.max(1.0e-7)
        * value.abs().max(design_variable_scale(config, index))
}

fn design_variable_scale(config: &PcFanConfig, index: usize) -> f64 {
    if index < BLADE_PATH_CONTROL_COUNT * 2 {
        return 0.01;
    }

    let section_start = BLADE_PATH_CONTROL_COUNT * 2;
    let section_end = section_start + BLADE_SECTION_STATION_COUNT * 4;
    if index < section_end {
        return match (index - section_start) % 4 {
            0 => config.dimensions.depth_m.max(1.0e-6),
            1 => 0.004,
            2 => 0.001,
            _ => 10.0,
        };
    }

    0.003
}

fn disk_solidity(geometry: &PcFanGeometry) -> f64 {
    let disk_area_m2 = std::f64::consts::PI
        * (geometry.blade_tip_radius_m.powi(2) - geometry.hub.radius_m.powi(2));
    let blade_area_m2 = geometry
        .blades
        .iter()
        .flat_map(|blade| blade.panels.iter())
        .map(|panel| panel.area)
        .sum::<f64>();

    (blade_area_m2 / disk_area_m2.max(f64::EPSILON)).clamp(0.05, 1.5)
}

fn fixed_hardware_flow_effects(
    _config: &PcFanConfig,
    geometry: &PcFanGeometry,
) -> PcFanHardwareFlowEffects {
    let tip_radius_m = geometry.blade_tip_radius_m;
    // Hub area is already excluded from the blade annulus, so additional
    // motor/strut blockage must be normalized by that same open annulus.
    let disk_area_m2 =
        std::f64::consts::PI * (tip_radius_m.powi(2) - geometry.hub.radius_m.powi(2)).max(0.0);
    let inlet_open_area_m2 = geometry.frame_box.inner_side_m * geometry.frame_box.inner_side_m;
    let motor_extra_area_m2 = std::f64::consts::PI
        * (geometry.motor_mount.radius_m.powi(2) - geometry.hub.radius_m.powi(2)).max(0.0);
    let strut_area_m2 = geometry
        .support_struts
        .iter()
        .map(|strut| support_strut_projected_area_m2(strut, tip_radius_m))
        .sum::<f64>();
    let motor_blockage_ratio =
        (motor_extra_area_m2 / disk_area_m2.max(f64::EPSILON)).clamp(0.0, 0.35);
    let strut_blockage_ratio = (strut_area_m2 / disk_area_m2.max(f64::EPSILON)).clamp(0.0, 0.35);
    let blockage_ratio = (motor_blockage_ratio + strut_blockage_ratio).clamp(0.0, 0.55);
    let wall_to_depth = (geometry.frame_box.wall_thickness_m
        / geometry.frame_box.depth_m.max(1.0e-6))
    .clamp(0.0, 0.25);
    let frame_lip_loss_factor = (1.0 - 0.28 * wall_to_depth).clamp(0.90, 1.0);
    let support_wake_loss_factor =
        (1.0 - 0.62 * strut_blockage_ratio - 0.35 * motor_blockage_ratio).clamp(0.65, 1.0);
    let total_loss_factor = frame_lip_loss_factor * support_wake_loss_factor;

    PcFanHardwareFlowEffects {
        inlet_open_area_m2,
        blockage_area_m2: motor_extra_area_m2 + strut_area_m2,
        blockage_ratio,
        motor_blockage_ratio,
        strut_blockage_ratio,
        frame_lip_loss_factor,
        support_wake_loss_factor,
        total_loss_factor,
    }
}

fn support_strut_projected_area_m2(strut: &PcFanSupportStrut, radius_m: f64) -> f64 {
    let dx = strut.end.x - strut.start.x;
    let dy = strut.end.y - strut.start.y;
    let length_m = dx.hypot(dy);
    let radial_start = (strut.start.x * strut.start.x + strut.start.y * strut.start.y).sqrt();
    let radial_end = (strut.end.x * strut.end.x + strut.end.y * strut.end.y).sqrt();
    let active_fraction = if radial_start <= f64::EPSILON && radial_end <= f64::EPSILON {
        0.0
    } else {
        (radius_m / radial_start.max(radial_end).max(f64::EPSILON)).clamp(0.0, 1.0)
    };

    length_m * active_fraction * strut.width_m
}

fn mean_support_strut_angle_factor(geometry: &PcFanGeometry) -> f64 {
    let mean_abs_sin = geometry
        .support_struts
        .iter()
        .map(|strut| {
            let dx = strut.end.x - strut.start.x;
            let dy = strut.end.y - strut.start.y;
            let length = dx.hypot(dy).max(f64::EPSILON);
            (dy / length).abs()
        })
        .sum::<f64>()
        / geometry.support_struts.len().max(1) as f64;

    mean_abs_sin.clamp(0.0, 1.0)
}

fn acoustic_shape_factor(
    config: &PcFanConfig,
    design: &PcFanDesignVariables,
    geometry: &PcFanGeometry,
    flow: &PcFanFlowSolution,
) -> f64 {
    let clearance_m = geometry.shroud.minimum_radius_m() - design.effective_tip_radius_m(config);
    let clearance_factor = if clearance_m <= 0.0 {
        2.4
    } else {
        (0.003 / clearance_m).clamp(0.55, 2.4)
    };
    let loading_factor = (flow.disk_solidity * (1.0 + flow.mach_number * 4.0)).clamp(0.2, 2.5);
    let wake_factor = (1.0 + 1.7 * flow.fixed_hardware.blockage_ratio).clamp(1.0, 2.2);
    let section_factor = mean_section_noise_factor(design);

    clearance_factor * loading_factor * wake_factor * section_factor
}

fn mean_section_noise_factor(design: &PcFanDesignVariables) -> f64 {
    let samples = 17;
    let sum = (0..samples)
        .map(|i| {
            let span_fraction = i as f64 / (samples - 1) as f64;
            let section = design.blade_section_at(span_fraction);
            let chord_m = section.chord_m.max(1.0e-6);
            let camber_ratio = (section.camber_m.abs() / chord_m).min(0.35);
            let thickness_ratio = (section.thickness_m / chord_m).clamp(0.01, 0.25);
            let twist_factor = (section.twist_deg.abs() / 45.0).min(1.0);
            1.0 + 1.8 * camber_ratio + 0.8 * thickness_ratio + 0.35 * twist_factor
        })
        .sum::<f64>();
    sum / samples as f64
}

fn pressure_level_db(pressure_pa: f64) -> f64 {
    20.0 * (pressure_pa.max(1.0e-12) / PRESSURE_REFERENCE_PA).log10()
}

fn combine_decibels(levels: impl Iterator<Item = f64>) -> f64 {
    let sum = levels.map(|level| 10.0_f64.powf(level / 10.0)).sum::<f64>();
    if sum <= 0.0 { 0.0 } else { 10.0 * sum.log10() }
}

fn combine_pressures_db(pressures: impl Iterator<Item = f64>) -> f64 {
    let sum_pressure_sq = pressures
        .map(|pressure| pressure.max(0.0) * pressure.max(0.0))
        .sum::<f64>();
    pressure_level_db(sum_pressure_sq.sqrt())
}

fn mean_blade_chord_m(design: &PcFanDesignVariables) -> f64 {
    let samples = 17;
    let sum = (0..samples)
        .map(|i| {
            let span_fraction = i as f64 / (samples - 1) as f64;
            design.blade_section_at(span_fraction).chord_m.max(1.0e-6)
        })
        .sum::<f64>();
    sum / samples as f64
}

#[cfg(test)]
mod airflow_band_tests {
    use super::*;

    fn weights(band_fraction: f64, weight: f64) -> PcFanOptimizationWeights {
        PcFanOptimizationWeights {
            airflow_band_fraction: band_fraction,
            airflow_penalty_weight: weight,
            ..PcFanOptimizationWeights::default()
        }
    }

    #[test]
    fn band_penalty_is_zero_inside_band() {
        let w = weights(0.10, 0.05);
        let target = 55.0;
        // ±5.5 CFM band — every value strictly inside should give zero.
        for &flow in &[55.0, 56.0, 53.0, 60.4, 49.6, 50.0] {
            assert_eq!(airflow_band_penalty(&w, target, flow), 0.0, "flow {flow}");
            assert_eq!(
                airflow_band_penalty_gradient(&w, target, flow),
                0.0,
                "gradient at flow {flow}"
            );
        }
    }

    #[test]
    fn band_penalty_is_positive_outside_band() {
        let w = weights(0.10, 0.05);
        let target = 55.0;
        let above = airflow_band_penalty(&w, target, 70.0);
        let below = airflow_band_penalty(&w, target, 40.0);
        assert!(above > 0.0);
        assert!(below > 0.0);
        // Symmetric: same |excess| over and under should give same penalty.
        let band = 5.5;
        let symmetric_above = airflow_band_penalty(&w, target, target + band + 4.0);
        let symmetric_below = airflow_band_penalty(&w, target, target - band - 4.0);
        let diff = (symmetric_above - symmetric_below).abs();
        assert!(diff < 1.0e-9, "symmetric penalty mismatch: {diff}");
    }

    #[test]
    fn band_penalty_grows_quadratically_outside() {
        let w = weights(0.10, 1.0);
        let target = 55.0;
        // Step 1 CFM beyond the band edge.
        let edge = target * (1.0 + 0.10);
        let p1 = airflow_band_penalty(&w, target, edge + 1.0); // excess = 1
        let p2 = airflow_band_penalty(&w, target, edge + 2.0); // excess = 2
        // p2 / p1 should be 4 within rounding.
        let ratio = p2 / p1;
        assert!(
            (ratio - 4.0).abs() < 1.0e-6,
            "expected p2/p1 ≈ 4 (quadratic), got {ratio}"
        );
    }

    #[test]
    fn band_penalty_gradient_signs_with_margin() {
        let w = weights(0.10, 0.05);
        let target = 55.0;
        let band = 5.5;
        let g_above = airflow_band_penalty_gradient(&w, target, target + band + 1.0);
        let g_below = airflow_band_penalty_gradient(&w, target, target - band - 1.0);
        assert!(g_above > 0.0, "above-band gradient should be positive");
        assert!(g_below < 0.0, "below-band gradient should be negative");
    }

    #[test]
    fn band_penalty_gradient_matches_finite_difference() {
        let w = weights(0.10, 0.05);
        let target = 55.0;
        let flow = 70.0; // outside the band
        let h = 1.0e-4;
        let plus = airflow_band_penalty(&w, target, flow + h);
        let minus = airflow_band_penalty(&w, target, flow - h);
        let fd = (plus - minus) / (2.0 * h);
        let analytic = airflow_band_penalty_gradient(&w, target, flow);
        let err = (fd - analytic).abs();
        assert!(err < 1.0e-6, "fd {fd} vs analytic {analytic}");
    }

    #[test]
    fn zero_band_falls_back_to_squared_penalty() {
        // band_fraction = 0 → penalty = w · margin² (the old behaviour).
        let w = weights(0.0, 0.05);
        let target = 55.0;
        let flow = 60.0;
        let expected = 0.05 * 5.0 * 5.0;
        let actual = airflow_band_penalty(&w, target, flow);
        assert!((actual - expected).abs() < 1.0e-12, "got {actual}");
    }
}

#[cfg(test)]
mod spectral_spread_tests {
    use super::*;

    fn harmonic(idx: usize, pressure_pa: f64) -> PcFanAcousticHarmonic {
        PcFanAcousticHarmonic {
            harmonic: idx + 1,
            frequency_hz: 100.0 * (idx + 1) as f64,
            pressure_pa,
            sound_pressure_level_db: 0.0,
            sound_power_level_db: 0.0,
        }
    }

    #[test]
    fn empty_harmonics_give_unit_flatness() {
        let f = harmonic_spectral_flatness(&[]);
        assert!((f - 1.0).abs() < 1.0e-12);
    }

    #[test]
    fn flat_spectrum_is_close_to_one() {
        let h: Vec<_> = (0..6).map(|i| harmonic(i, 1.0)).collect();
        let f = harmonic_spectral_flatness(&h);
        assert!(f > 0.99, "flat spectrum flatness {f}");
    }

    #[test]
    fn tonal_spectrum_is_close_to_zero() {
        // One big peak, rest tiny.
        let mut h = vec![harmonic(0, 100.0)];
        for i in 1..6 {
            h.push(harmonic(i, 1.0e-6));
        }
        let f = harmonic_spectral_flatness(&h);
        assert!(f < 0.05, "tonal spectrum flatness {f}");
    }

    #[test]
    fn flatter_spectrum_yields_lower_penalty() {
        // Compare a tonal vs a flatter spectrum at the same total power —
        // higher flatness must give lower (1 − flatness) penalty.
        let tonal = vec![
            harmonic(0, 10.0),
            harmonic(1, 0.1),
            harmonic(2, 0.1),
            harmonic(3, 0.1),
        ];
        let flat = vec![
            harmonic(0, 5.0),
            harmonic(1, 5.0),
            harmonic(2, 5.0),
            harmonic(3, 5.0),
        ];
        let f_tonal = harmonic_spectral_flatness(&tonal);
        let f_flat = harmonic_spectral_flatness(&flat);
        assert!(f_flat > f_tonal, "flat {f_flat} vs tonal {f_tonal}");
    }
}

#[cfg(test)]
mod directivity_tests {
    use super::*;

    #[test]
    fn production_directivity_preserves_spherical_power() {
        let config = PcFanConfig::default_120mm();
        let design = config.default_design_variables();
        let evaluation = evaluate_pc_fan(&config, &design);

        let ratio = directivity_power_ratio(&evaluation.acoustics)
            .expect("production harmonics must contain finite non-zero power");
        assert!(
            (ratio - 1.0).abs() < 1.0e-12,
            "directivity spherical power ratio should be one, got {ratio:.16}"
        );
    }

    #[test]
    fn observer_array_has_expected_size() {
        let config = PcFanConfig::default_120mm();
        let design = config.default_design_variables();
        let evaluation = evaluate_pc_fan(&config, &design);
        let expected = config.observers.observer_count();
        assert_eq!(evaluation.observer_spls_db.len(), expected);
        assert!(expected >= 1);
    }

    #[test]
    fn observer_spls_are_finite() {
        let config = PcFanConfig::default_120mm();
        let design = config.default_design_variables();
        let evaluation = evaluate_pc_fan(&config, &design);
        for spl in &evaluation.observer_spls_db {
            assert!(spl.is_finite(), "non-finite observer SPL {spl}");
        }
        assert!(evaluation.directivity_spread_db.is_finite());
        assert!(evaluation.directivity_spread_db >= 0.0);
    }

    #[test]
    fn directivity_spread_zero_for_single_observer() {
        let mut config = PcFanConfig::default_120mm();
        config.observers = PcFanObserverArrayConfig {
            azimuth_count: 1,
            elevation_count: 1,
            distance_m: 1.0,
        };
        let design = config.default_design_variables();
        let evaluation = evaluate_pc_fan(&config, &design);
        assert_eq!(evaluation.observer_spls_db.len(), 1);
        assert_eq!(evaluation.directivity_spread_db, 0.0);
        assert_eq!(evaluation.directivity_spread_penalty, 0.0);
    }

    #[test]
    fn directivity_spread_weight_changes_objective() {
        // Two configs that differ only in directivity_spread_weight should
        // produce two different objective values (assuming the spread is
        // non-zero on the default geometry).
        let mut a = PcFanConfig::default_120mm();
        a.weights.acoustic_weight = 0.0; // isolate the spread term
        a.weights.airflow_penalty_weight = 0.0;
        a.weights.directivity_spread_weight = 0.0;
        let design = a.default_design_variables();
        let no_weight = evaluate_pc_fan(&a, &design);

        let mut b = a.clone();
        b.weights.directivity_spread_weight = 1.0;
        let with_weight = evaluate_pc_fan(&b, &design);

        // Spread term must contribute to the objective at the second config.
        assert!(with_weight.directivity_spread_db > 0.0);
        assert!(
            (with_weight.objective - no_weight.objective - with_weight.directivity_spread_penalty)
                .abs()
                < 1.0e-9
        );
    }

    #[test]
    fn dipole_weights_sum_to_one() {
        for h in 1..=10 {
            let (a, t) = dipole_weights_for_harmonic(h);
            assert!((a + t - 1.0).abs() < 1.0e-12, "h {h}: a={a} t={t}");
        }
    }
}

#[cfg(test)]
mod tip_leakage_tests {
    use super::*;

    #[test]
    fn tip_leakage_zero_when_clearance_is_zero() {
        // Force tip clearance to exactly zero by raising the blade tip
        // radius up to the shroud's clipped inner radius. (`shroud_inner_radius_m`
        // is upper-bounded at `0.5·frame_side − wall thicknesses` inside
        // `PcFanShroud::from_design`, so changing that field alone does
        // not always shrink the clearance below the cap.)
        let mut config = PcFanConfig::default_120mm();
        let max_shroud_inner = 0.5 * config.dimensions.frame_side_m
            - config.dimensions.frame_wall_thickness_m
            - config.dimensions.shroud_wall_thickness_m;
        config.dimensions.blade_tip_radius_m = max_shroud_inner;
        let design = config.default_design_variables();
        let geometry = config.build_geometry(&design);
        let pressures = tip_leakage_pressures_per_harmonic(
            &config,
            &design,
            &geometry,
            config.mesh.harmonic_count,
        );
        // Every per-harmonic tip-leakage pressure must be exactly zero.
        for p in &pressures {
            assert_eq!(*p, 0.0, "non-zero tip-leakage pressure with zero clearance");
        }
    }

    #[test]
    fn tip_leakage_grows_with_clearance() {
        // Smaller blade tip radius → larger clearance → louder tip-leakage.
        // We vary `blade_tip_radius_m` rather than `shroud_inner_radius_m`
        // because the latter is upper-bounded by the frame geometry and
        // can be clipped silently.
        let mut tight_config = PcFanConfig::default_120mm();
        tight_config.dimensions.blade_tip_radius_m = 0.055; // ≈ 2 mm clearance
        let design = tight_config.default_design_variables();
        let tight = evaluate_pc_fan(&tight_config, &design);

        let mut wide_config = tight_config.clone();
        wide_config.dimensions.blade_tip_radius_m = 0.050; // ≈ 7 mm clearance
        let wide = evaluate_pc_fan(&wide_config, &design);

        assert!(
            wide.acoustics.tip_leakage_spl_db > tight.acoustics.tip_leakage_spl_db,
            "wide clearance {} should exceed tight clearance {}",
            wide.acoustics.tip_leakage_spl_db,
            tight.acoustics.tip_leakage_spl_db
        );
    }

    #[test]
    fn tip_leakage_grows_with_rpm() {
        // tip Mach^5 dependence — doubling RPM should sharply increase
        // tip-leakage SPL.
        let mut low_config = PcFanConfig::default_120mm();
        low_config.dimensions.shroud_inner_radius_m += 0.005; // ensure non-zero clearance
        low_config.dimensions.frame_side_m += 0.012; // preserve the fixed-envelope fit
        low_config.operating_point.rpm = 600.0;
        let design = low_config.default_design_variables();
        let low = evaluate_pc_fan(&low_config, &design);

        let mut high_config = low_config.clone();
        high_config.operating_point.rpm = 1200.0;
        let high = evaluate_pc_fan(&high_config, &design);

        // 5th-power scaling means doubling RPM should add ~ 5·3 = 15 dB.
        let delta = high.acoustics.tip_leakage_spl_db - low.acoustics.tip_leakage_spl_db;
        assert!(delta > 10.0, "expected >10 dB increase, got {delta:.2}");
    }

    #[test]
    fn tip_leakage_pressures_have_correct_length() {
        let config = PcFanConfig::default_120mm();
        let design = config.default_design_variables();
        let evaluation = evaluate_pc_fan(&config, &design);
        assert_eq!(
            evaluation.acoustics.harmonics.len(),
            config.mesh.harmonic_count
        );
        assert!(evaluation.acoustics.tip_leakage_spl_db.is_finite());
    }
}

#[cfg(test)]
mod trailing_edge_tests {
    use super::*;

    #[test]
    fn trailing_edge_zero_when_disabled() {
        let mut config = PcFanConfig::default_120mm();
        config.broadband = PcFanBroadbandConfig::off();
        let design = config.default_design_variables();
        let eval = evaluate_pc_fan(&config, &design);
        // dB on zero pressure clamps to the floor (≈ -146 dB), not 0 dB.
        assert!(
            eval.acoustics.trailing_edge_spl_db < -100.0,
            "expected near-floor SPL, got {}",
            eval.acoustics.trailing_edge_spl_db
        );
    }

    #[test]
    fn trailing_edge_grows_with_chord() {
        // BPM TE-noise power scales with `δ*/c · L_span` where δ* ∝ c/Re_c^0.2
        // and Re_c ∝ c, so per-section power ∝ c^0.8 — strictly monotone in
        // chord. Doubling every section's chord must raise the integrated
        // TE SPL.
        let config = PcFanConfig::default_120mm();
        let small_design = config.default_design_variables();
        let mut large_design = small_design.clone();
        for s in &mut large_design.blade_sections {
            s.chord_m *= 2.0;
        }
        let small = evaluate_pc_fan(&config, &small_design);
        let large = evaluate_pc_fan(&config, &large_design);
        assert!(
            large.acoustics.trailing_edge_spl_db > small.acoustics.trailing_edge_spl_db,
            "TE SPL should increase with chord: small={}, large={}",
            small.acoustics.trailing_edge_spl_db,
            large.acoustics.trailing_edge_spl_db
        );
    }

    #[test]
    fn trailing_edge_grows_with_rpm() {
        // BPM scales as M⁵: doubling RPM should add ≈ 5·log10(2)·10 ≈ 15 dB.
        let mut low = PcFanConfig::default_120mm();
        low.operating_point.rpm = 600.0;
        let design = low.default_design_variables();
        let low_eval = evaluate_pc_fan(&low, &design);

        let mut high = low.clone();
        high.operating_point.rpm = 1200.0;
        let high_eval = evaluate_pc_fan(&high, &design);

        let delta =
            high_eval.acoustics.trailing_edge_spl_db - low_eval.acoustics.trailing_edge_spl_db;
        assert!(
            delta > 10.0,
            "expected >10 dB increase from RPM doubling, got {delta:.2}"
        );
    }

    #[test]
    fn trailing_edge_present_in_overall_spl() {
        // Turning TE on must raise the overall SPL versus the same config
        // with TE disabled (other sources held fixed).
        let design = PcFanConfig::default_120mm().default_design_variables();
        let mut off = PcFanConfig::default_120mm();
        off.broadband.trailing_edge_enabled = false;
        let mut on = PcFanConfig::default_120mm();
        on.broadband.trailing_edge_enabled = true;

        let off_eval = evaluate_pc_fan(&off, &design);
        let on_eval = evaluate_pc_fan(&on, &design);

        assert!(
            on_eval.acoustics.overall_spl_db >= off_eval.acoustics.overall_spl_db,
            "overall SPL should not decrease when TE is enabled: on={}, off={}",
            on_eval.acoustics.overall_spl_db,
            off_eval.acoustics.overall_spl_db
        );
        // Sanity check: TE diagnostic itself must be finite when enabled.
        assert!(
            on_eval.acoustics.trailing_edge_spl_db.is_finite(),
            "TE SPL must be finite, got {}",
            on_eval.acoustics.trailing_edge_spl_db
        );
    }

    #[test]
    fn trailing_edge_spectral_peak_inside_resolved_harmonics() {
        // The per-harmonic TE pressure vector must place its maximum inside
        // [1, harmonic_count]; if the log-normal binning falls entirely off
        // the resolved spectrum the term silently becomes zero, which would
        // be a regression.
        let mut config = PcFanConfig::default_120mm();
        // Run at a higher RPM where TE noise is well above the floor so the
        // peak harmonic is meaningful.
        config.operating_point.rpm = 2000.0;
        let design = config.default_design_variables();
        let geometry = config.build_geometry(&design);
        let flow = solve_pc_fan_mean_flow(&config, &design);
        let pressures = trailing_edge_pressures_per_harmonic(
            &config,
            &design,
            &flow,
            config.mesh.harmonic_count,
        );
        assert_eq!(pressures.len(), config.mesh.harmonic_count);
        let peak_pressure = pressures.iter().copied().fold(0.0_f64, f64::max);
        assert!(
            peak_pressure > 0.0,
            "TE per-harmonic vector is all zero — spectrum likely fell outside resolved band"
        );
        // Drop unused geometry to silence the linter without mutating the
        // function signature.
        let _ = geometry;
    }
}

#[cfg(test)]
mod strut_motor_broadband_tests {
    use super::*;

    #[test]
    fn broadband_zero_when_disabled() {
        let mut config = PcFanConfig::default_120mm();
        config.broadband.strut_motor_enabled = false;
        let design = config.default_design_variables();
        let eval = evaluate_pc_fan(&config, &design);
        // dB on zero pressure clamps to the floor (≈ -146 dB), not 0 dB.
        // Use a finite-but-very-low threshold instead.
        assert!(
            eval.acoustics
                .fixed_hardware_noise
                .support_wake_broadband_spl_db
                < -100.0,
            "expected near-floor SPL, got {}",
            eval.acoustics
                .fixed_hardware_noise
                .support_wake_broadband_spl_db
        );
        assert!(
            eval.acoustics
                .fixed_hardware_noise
                .motor_mount_broadband_spl_db
                < -100.0
        );
    }

    #[test]
    fn broadband_positive_when_enabled() {
        let mut config = PcFanConfig::default_120mm();
        config.broadband.strut_motor_enabled = true;
        let design = config.default_design_variables();
        let eval = evaluate_pc_fan(&config, &design);
        let strut = eval
            .acoustics
            .fixed_hardware_noise
            .support_wake_broadband_spl_db;
        let motor = eval
            .acoustics
            .fixed_hardware_noise
            .motor_mount_broadband_spl_db;
        assert!(strut.is_finite() && strut > 0.0, "strut broadband {strut}");
        // The motor sits inside the hub by default, so motor-wake broadband
        // collapses to the dB floor; we only require the value stays finite.
        assert!(motor.is_finite(), "motor broadband {motor}");
    }

    #[test]
    fn broadband_increases_combined_hardware_spl() {
        // Same operating point, broadband off vs on — combined hardware
        // SPL should be strictly higher when broadband is enabled.
        let design = PcFanConfig::default_120mm().default_design_variables();
        let mut off = PcFanConfig::default_120mm();
        off.broadband.strut_motor_enabled = false;
        let mut on = PcFanConfig::default_120mm();
        on.broadband.strut_motor_enabled = true;
        let off_eval = evaluate_pc_fan(&off, &design);
        let on_eval = evaluate_pc_fan(&on, &design);
        assert!(
            on_eval.acoustics.fixed_hardware_noise.combined_spl_db
                > off_eval.acoustics.fixed_hardware_noise.combined_spl_db,
            "on={} off={}",
            on_eval.acoustics.fixed_hardware_noise.combined_spl_db,
            off_eval.acoustics.fixed_hardware_noise.combined_spl_db
        );
    }
}

#[cfg(test)]
mod rsi_tests {
    use super::*;

    fn baseline_config() -> PcFanConfig {
        let mut config = PcFanConfig::default_120mm();
        // Disable broadband paths so the support_wake_spl_db field
        // reflects only the new Sears tonal contribution.
        config.broadband = PcFanBroadbandConfig::off();
        config
    }

    #[test]
    fn rsi_tone_grows_with_strut_wake_deficit() {
        let mut weak = baseline_config();
        weak.dimensions.strut_wake_centerline_deficit = 0.10;
        let mut strong = baseline_config();
        strong.dimensions.strut_wake_centerline_deficit = 0.40;

        let design = weak.default_design_variables();
        let weak_eval = evaluate_pc_fan(&weak, &design);
        let strong_eval = evaluate_pc_fan(&strong, &design);

        // |Δv| ∝ wake_deficit, lift ∝ Δv, p² ∝ Δv² → +12 dB for ×4.
        let delta = strong_eval
            .acoustics
            .fixed_hardware_noise
            .support_wake_spl_db
            - weak_eval.acoustics.fixed_hardware_noise.support_wake_spl_db;
        assert!(
            delta > 8.0,
            "expected ≳ 8 dB increase from 4× wake deficit, got {delta:.2} dB"
        );
    }

    #[test]
    fn rsi_tone_falls_with_axial_gap() {
        let mut tight = baseline_config();
        tight.dimensions.rotor_strut_axial_gap_m = 0.001; // ~0 mm gap
        let mut loose = baseline_config();
        loose.dimensions.rotor_strut_axial_gap_m = 0.030; // 30 mm gap

        let design = tight.default_design_variables();
        let tight_eval = evaluate_pc_fan(&tight, &design);
        let loose_eval = evaluate_pc_fan(&loose, &design);

        let delta = tight_eval
            .acoustics
            .fixed_hardware_noise
            .support_wake_spl_db
            - loose_eval
                .acoustics
                .fixed_hardware_noise
                .support_wake_spl_db;
        assert!(
            delta > 3.0,
            "expected ≳ 3 dB drop from increased axial gap, got Δ = {delta:.2} dB"
        );
    }

    #[test]
    fn rsi_tone_concentrates_in_low_harmonics() {
        // For B = 8 blades, S = 4·2 = 8 wakes/rev → matched n_wake = m
        // for the m-th BPF. The 1st BPF (n_wake = 1) has the strongest
        // Gaussian-wake Fourier content; high BPF harmonics decay.
        // Verify the per-harmonic vector is monotonically falling at the
        // highest harmonic versus the first.
        let config = baseline_config();
        let design = config.default_design_variables();
        let flow = solve_pc_fan_mean_flow(&config, &design);
        let pressures = rotor_stator_interaction_pressures_per_harmonic(&config, &design, &flow, 6);
        assert_eq!(pressures.len(), 6);
        assert!(
            pressures[0] > pressures[5],
            "1st BPF tone should exceed 6th BPF: {} vs {}",
            pressures[0],
            pressures[5]
        );
    }

    #[test]
    fn sears_function_low_and_high_k_limits() {
        use super::aerodynamic_response::sears_magnitude;

        // Low-k quasi-steady limit: |S(0+)| → 1.
        let low = sears_magnitude(1.0e-4);
        assert!(
            (low - 1.0).abs() < 1.0e-3,
            "low-k Sears should approach 1, got {low:.4}"
        );

        // Sears is monotone-decreasing in |S(k)| for moderate k.
        // Hassig's Theodorsen approximation in the conventional sign
        // convention gives |S(1)| ≈ 0.38, |S(5)| ≈ 0.18.
        let mid = sears_magnitude(1.0);
        let high = sears_magnitude(5.0);
        assert!(
            (0.30..=0.50).contains(&mid),
            "|S(1)| should sit in [0.30, 0.50], got {mid:.4}"
        );
        assert!(
            (0.12..=0.25).contains(&high),
            "|S(5)| should sit in [0.12, 0.25], got {high:.4}"
        );
        assert!(
            high < mid,
            "|S(k)| must be monotone decreasing for k ∈ [1, 5]: |S(1)|={mid}, |S(5)|={high}"
        );

        // High-k asymptote |S(k)| → 1/√(2πk).
        let asymptote_k = 10.0;
        let asymptote = sears_magnitude(asymptote_k);
        let expected = 1.0 / (std::f64::consts::TAU * asymptote_k).sqrt();
        let rel = (asymptote - expected).abs() / expected;
        assert!(
            rel < 0.05,
            "|S(10)| should approach 1/√(20π) ≈ {expected:.4}, got {asymptote:.4} (rel {rel:.3})"
        );
    }
}

#[cfg(test)]
mod duct_cutoff_tests {
    use super::*;

    #[test]
    fn cutoff_passes_above_cut_on() {
        // Default 120 mm fan: R = 57 mm, c = 343 m/s →
        //   f_10 = 1.8412 · 343 / (2π · 0.057) ≈ 1763 Hz.
        // At 5 kHz (well above cut-on), no attenuation.
        let attn = duct_cutoff_attenuation(5000.0, 0.057, 0.025, 343.0);
        assert!(
            (attn - 1.0).abs() < 1.0e-12,
            "above cut-on attenuation should be exactly 1.0, got {attn}"
        );
    }

    #[test]
    fn cutoff_attenuates_below_cut_on() {
        // At 100 Hz (well below f_10 ≈ 1763 Hz), the (1,0) mode is
        // strongly evanescent across a 25 mm shroud.
        let attn = duct_cutoff_attenuation(100.0, 0.057, 0.025, 343.0);
        assert!(
            attn < 1.0,
            "below cut-on attenuation should be < 1, got {attn}"
        );
        // Sanity bound: |k_z|·L = √((1.8412/0.057)² − (200π/343)²)·0.025
        //              ≈ 32.3 · 0.025 ≈ 0.81  → attn ≈ 0.45.
        assert!(
            (0.4..=0.50).contains(&attn),
            "expected ≈ 0.45 attenuation at 100 Hz, got {attn:.4}"
        );
    }

    #[test]
    fn cutoff_grows_with_shroud_length() {
        // Doubling the axial length doubles the |kz|·L exponent, so
        // attn_2L = attn_L².
        let attn_l = duct_cutoff_attenuation(100.0, 0.057, 0.025, 343.0);
        let attn_2l = duct_cutoff_attenuation(100.0, 0.057, 0.050, 343.0);
        let expected = attn_l * attn_l;
        let rel = (attn_2l - expected).abs() / expected;
        assert!(
            rel < 1.0e-9,
            "attn(2L) = attn(L)²: got attn_l={attn_l}, attn_2l={attn_2l}, expected {expected}"
        );
    }

    #[test]
    fn cutoff_lowers_overall_spl_below_cut_on() {
        // At 600 rpm, the BPF and all resolved harmonics sit below the
        // (1,0) cut-on. Enabling the duct filter must reduce the overall
        // SPL versus a config where it's disabled.
        let mut on = PcFanConfig::default_120mm();
        on.broadband.duct_cutoff_enabled = true;
        let mut off = PcFanConfig::default_120mm();
        off.broadband.duct_cutoff_enabled = false;
        let design = on.default_design_variables();
        let on_eval = evaluate_pc_fan(&on, &design);
        let off_eval = evaluate_pc_fan(&off, &design);
        assert!(
            on_eval.acoustics.overall_spl_db < off_eval.acoustics.overall_spl_db,
            "duct cut-off should lower SPL below cut-on: on={}, off={}",
            on_eval.acoustics.overall_spl_db,
            off_eval.acoustics.overall_spl_db
        );
    }

    #[test]
    fn cutoff_disabled_is_passthrough() {
        // With `duct_cutoff_enabled = false`, `apply_duct_cutoff_filter`
        // must return its input pressure unchanged.
        let mut config = PcFanConfig::default_120mm();
        config.broadband.duct_cutoff_enabled = false;
        let p_in = 1.234e-3;
        let p_out = apply_duct_cutoff_filter(&config, 100.0, p_in);
        assert_eq!(p_in, p_out, "passthrough with cutoff disabled");
    }
}

#[cfg(test)]
mod leading_edge_inflow_tests {
    use super::*;

    fn baseline_config_with_ti(ti: f64) -> PcFanConfig {
        let mut config = PcFanConfig::default_120mm();
        // Disable other broadband / propagation paths so the
        // leading_edge_inflow_spl_db field reflects only the LE term.
        config.broadband = PcFanBroadbandConfig {
            tip_leakage_enabled: false,
            strut_motor_enabled: false,
            trailing_edge_enabled: false,
            duct_cutoff_enabled: false,
            leading_edge_inflow_enabled: true,
        };
        config.operating_point.inflow_turbulence_intensity = ti;
        config
    }

    #[test]
    fn leading_edge_inflow_zero_when_ti_zero() {
        let config = baseline_config_with_ti(0.0);
        let design = config.default_design_variables();
        let eval = evaluate_pc_fan(&config, &design);
        // dB on zero pressure clamps to the floor, not 0 dB.
        assert!(
            eval.acoustics.leading_edge_inflow_spl_db < -100.0,
            "expected near-floor LE SPL at TI=0, got {}",
            eval.acoustics.leading_edge_inflow_spl_db
        );
    }

    #[test]
    fn leading_edge_inflow_zero_when_disabled() {
        let mut config = baseline_config_with_ti(0.10);
        config.broadband.leading_edge_inflow_enabled = false;
        let design = config.default_design_variables();
        let eval = evaluate_pc_fan(&config, &design);
        assert!(
            eval.acoustics.leading_edge_inflow_spl_db < -100.0,
            "expected near-floor LE SPL when disabled, got {}",
            eval.acoustics.leading_edge_inflow_spl_db
        );
    }

    #[test]
    fn leading_edge_inflow_grows_with_ti() {
        // p² ∝ TI² → 4× TI → +12 dB.
        let weak = baseline_config_with_ti(0.05);
        let strong = baseline_config_with_ti(0.20);
        let design = weak.default_design_variables();
        let weak_eval = evaluate_pc_fan(&weak, &design);
        let strong_eval = evaluate_pc_fan(&strong, &design);
        let delta = strong_eval.acoustics.leading_edge_inflow_spl_db
            - weak_eval.acoustics.leading_edge_inflow_spl_db;
        // 4× TI → expected +12 dB; allow 0.5 dB slack for binning artefacts.
        assert!(
            (11.5..=12.5).contains(&delta),
            "expected ≈ +12 dB from 4× TI, got {delta:.2}"
        );
    }

    #[test]
    fn leading_edge_inflow_grows_with_chord() {
        // BPM-style scaling: per-section p² ∝ b = c/2 × spectrum_shape.
        // Doubling chord must raise the integrated LE SPL.
        let config = baseline_config_with_ti(0.10);
        let small_design = config.default_design_variables();
        let mut large_design = small_design.clone();
        for s in &mut large_design.blade_sections {
            s.chord_m *= 2.0;
        }
        let small_eval = evaluate_pc_fan(&config, &small_design);
        let large_eval = evaluate_pc_fan(&config, &large_design);
        assert!(
            large_eval.acoustics.leading_edge_inflow_spl_db
                > small_eval.acoustics.leading_edge_inflow_spl_db,
            "LE SPL should increase with chord: small={}, large={}",
            small_eval.acoustics.leading_edge_inflow_spl_db,
            large_eval.acoustics.leading_edge_inflow_spl_db
        );
    }

    #[test]
    fn leading_edge_inflow_present_in_overall_spl() {
        // Turning the LE term on (via TI > 0) must raise overall SPL.
        let mut off = PcFanConfig::default_120mm();
        off.operating_point.inflow_turbulence_intensity = 0.0;
        let mut on = PcFanConfig::default_120mm();
        on.operating_point.inflow_turbulence_intensity = 0.10;
        let design = off.default_design_variables();
        let off_eval = evaluate_pc_fan(&off, &design);
        let on_eval = evaluate_pc_fan(&on, &design);
        assert!(
            on_eval.acoustics.overall_spl_db >= off_eval.acoustics.overall_spl_db,
            "overall SPL must not decrease when LE inflow is enabled: on={}, off={}",
            on_eval.acoustics.overall_spl_db,
            off_eval.acoustics.overall_spl_db
        );
        assert!(on_eval.acoustics.leading_edge_inflow_spl_db.is_finite());
    }

    #[test]
    fn leading_edge_inflow_length_scale_shifts_spectrum() {
        // Smaller integral length scale Λ → higher von Kármán cutoff
        // f_c = U / (2π·Λ) → spectrum bias toward higher harmonics.
        // Verify the ratio p²(h_high) / p²(h_low) increases with smaller Λ.
        let mut large_lambda = baseline_config_with_ti(0.10);
        large_lambda.operating_point.inflow_integral_length_scale_m = 0.020;
        let mut small_lambda = baseline_config_with_ti(0.10);
        small_lambda.operating_point.inflow_integral_length_scale_m = 0.001;
        let design = large_lambda.default_design_variables();
        let flow = solve_pc_fan_mean_flow(&large_lambda, &design);
        let p_large = leading_edge_inflow_pressures_per_harmonic(&large_lambda, &design, &flow, 6);
        let p_small = leading_edge_inflow_pressures_per_harmonic(&small_lambda, &design, &flow, 6);
        let ratio_large = p_large[5] / p_large[0].max(1.0e-30);
        let ratio_small = p_small[5] / p_small[0].max(1.0e-30);
        assert!(
            ratio_small > ratio_large,
            "smaller Λ should raise high-harmonic share: small Λ ratio={}, large Λ ratio={}",
            ratio_small,
            ratio_large
        );
    }
}

#[cfg(test)]
mod hub_attachment_tests {
    use super::*;

    #[test]
    fn hub_attachment_penalty_zero_at_baseline() {
        // Baseline blade_path is all zero → penalty identically zero
        // regardless of twist/camber/chord (which only spread vertices
        // tangentially around the panel center, not via the path).
        let config = PcFanConfig::default_120mm();
        let design = config.default_design_variables();
        let penalty = compute_hub_attachment_penalty(&config, &design);
        assert_eq!(
            penalty, 0.0,
            "baseline hub attachment penalty must be zero, got {penalty}"
        );
    }

    #[test]
    fn hub_attachment_penalty_positive_when_root_swept() {
        // Force a non-zero root tangential offset (bypassing the
        // optimizer's bound pin) and verify the penalty fires.
        let config = PcFanConfig::default_120mm();
        let mut design = config.default_design_variables();
        design.blade_path[0].tangential_offset_m = 0.020;
        let penalty = compute_hub_attachment_penalty(&config, &design);
        assert!(
            penalty > 0.0,
            "swept root must produce positive hub attachment penalty, got {penalty}"
        );
    }

    #[test]
    fn hub_attachment_penalty_zero_when_weight_zero() {
        // Belt-and-braces: even with a swept root, weight=0 → no penalty.
        let mut config = PcFanConfig::default_120mm();
        config.weights.hub_attachment_penalty_weight = 0.0;
        let mut design = config.default_design_variables();
        design.blade_path[0].tangential_offset_m = 0.020;
        let penalty = compute_hub_attachment_penalty(&config, &design);
        assert_eq!(penalty, 0.0);
    }

    #[test]
    fn optimiser_keeps_blade_inside_shroud() {
        // End-to-end smoke: with the bumped default penalties (1e6) and
        // the hub-attachment penalty active, a tiny DE run must produce
        // a design whose maximum blade-vertex radius stays inside
        // `shroud_inner_radius_m`. This catches regressions where any
        // of the three patches (bound-pin, weight-bump, hub-penalty)
        // gets weakened or removed.
        let config = PcFanConfig {
            mesh: PcFanMeshResolution {
                radial_segments: 4,
                chord_segments: 2,
                circumferential_segments: 8,
                harmonic_count: 3,
            },
            ..PcFanConfig::default_120mm()
        };
        let result = crate::optimizer::optimize_pc_fan_de(config.clone(), 4, 6, 7);
        let mut max_radius = 0.0_f64;
        for blade in &result.geometry.blades {
            for panel in &blade.panels {
                for v in &panel.vertices {
                    let r = (v.x * v.x + v.y * v.y).sqrt();
                    if r > max_radius {
                        max_radius = r;
                    }
                }
            }
        }
        // Allow a tiny floating-point tolerance — the penalty isn't a
        // hard constraint, just a very stiff one.
        let limit = config.dimensions.shroud_inner_radius_m + 1.0e-4;
        assert!(
            max_radius < limit,
            "optimised blade radius {} exceeds shroud {} (limit {})",
            max_radius,
            config.dimensions.shroud_inner_radius_m,
            limit
        );
    }
}

#[cfg(test)]
mod operating_point_tests {
    use super::*;

    #[test]
    fn operating_envelope_reports_sorted_monotonic_reference_points() {
        let mut config = PcFanConfig::bench_120mm();
        config.weights.airflow_band_fraction = 1.0;
        config.operating_point_spread.rpms = vec![900.0, 400.0, 1200.0, 600.0];
        let envelope =
            evaluate_pc_fan_operating_envelope(&config, &config.default_design_variables());

        assert_eq!(
            envelope
                .points
                .iter()
                .map(|point| point.rpm)
                .collect::<Vec<_>>(),
            vec![400.0, 600.0, 900.0, 1200.0]
        );
        assert!(envelope.finite);
        assert!(envelope.airflow_strictly_increases_with_rpm);
        assert!(envelope.pressure_strictly_increases_with_rpm);
        assert!(envelope.all_points_feasible);
        assert!(envelope.is_valid());
    }

    #[test]
    fn operating_envelope_rejects_nonfinite_rpm_and_deduplicates_samples() {
        let mut config = PcFanConfig::default_120mm();
        config.operating_point_spread.rpms = vec![600.0, f64::NAN, 600.0];
        let envelope =
            evaluate_pc_fan_operating_envelope(&config, &config.default_design_variables());

        assert_eq!(envelope.points.len(), 2);
        assert!(!envelope.finite);
        assert!(!envelope.is_valid());
    }

    #[test]
    fn manufacturing_ensemble_is_deterministic_and_covers_tolerance_corners() {
        let mut config = PcFanConfig::bench_120mm();
        config.weights.airflow_band_fraction = 1.0;
        config.structural.min_modal_separation_hz = 0.0;
        config.structural.max_imbalance_g_mm = 100.0;
        config.manufacturing.tolerance = PcFanManufacturingTolerance {
            blade_thickness_fraction: 0.05,
            blade_chord_fraction: 0.03,
            twist_deg: 2.0,
            shroud_offset_m: 0.0005,
        };
        let design = config.default_design_variables();
        let first = evaluate_pc_fan_manufacturing_ensemble(&config, &design)
            .expect("valid reduced-order manufacturing ensemble");
        let second = evaluate_pc_fan_manufacturing_ensemble(&config, &design)
            .expect("deterministic manufacturing ensemble");

        assert_eq!(first, second);
        assert_eq!(first.nominal_index, 0);
        assert_eq!(first.points.len(), 17);
        assert!(first.finite);
        assert!(first.has_tolerance_coverage());
        assert!(first.is_robust(), "ensemble was not admitted: {first:?}");
        assert_eq!(
            first.points[0].variation,
            PcFanManufacturingVariation::nominal()
        );
        assert_eq!(first.points[0].design_variables, design);

        let optimized = OptimizedPcFan {
            config: config.clone(),
            design_variables: design.clone(),
            geometry: config.build_geometry(&design),
            evaluation: first.points[0].evaluation.clone(),
            optimization: PcFanOptimizationSummary {
                baseline_objective: first.points[0].evaluation.objective,
                best_objective: first.points[0].evaluation.objective,
                iterations: 0,
                function_evaluations: 1,
                gradient_evaluations: 0,
                feasible: true,
                fan_curve_objective: None,
                fan_curve_feasible: true,
                converged: true,
                status: sonium_solvers_optimization::OptimStatus::Converged,
            },
        };
        assert!(
            optimized
                .is_manufacturing_robustly_feasible()
                .expect("valid manufacturing ensemble")
        );
        assert!(
            optimized
                .is_manufacturing_envelope_robustly_feasible()
                .expect("valid manufacturing operating envelope")
        );
    }

    #[test]
    fn manufacturing_ensemble_rejects_cfd_and_invalid_tolerance() {
        let mut config = PcFanConfig::bench_120mm();
        let design = config.default_design_variables();
        config.manufacturing.tolerance.blade_chord_fraction = 1.0;
        assert!(matches!(
            evaluate_pc_fan_manufacturing_ensemble(&config, &design),
            Err(PcFanManufacturingEnsembleError::InvalidTolerance {
                parameter: "blade_chord_fraction",
                ..
            })
        ));

        config.manufacturing.tolerance = PcFanManufacturingTolerance::default();
        config.cfd.flow_model = PcFanFlowModel::IncompressibleCfd;
        assert!(matches!(
            evaluate_pc_fan_manufacturing_ensemble(&config, &design),
            Err(PcFanManufacturingEnsembleError::UnsupportedFlowModel(
                PcFanFlowModel::IncompressibleCfd
            ))
        ));
    }

    #[test]
    fn manufacturing_ensemble_fails_robust_admission_at_thickness_corner() {
        let mut config = PcFanConfig::bench_120mm();
        config.weights.airflow_band_fraction = 1.0;
        config.manufacturing.min_blade_thickness_m = 0.00119;
        config.manufacturing.tolerance.blade_thickness_fraction = 0.02;
        let design = config.default_design_variables();
        let ensemble = evaluate_pc_fan_manufacturing_ensemble(&config, &design)
            .expect("valid tolerance configuration");

        assert!(ensemble.finite);
        assert!(!ensemble.all_points_feasible);
        assert!(!ensemble.is_robust());
        assert!(
            ensemble
                .points
                .iter()
                .any(|point| !point.evaluation.manufacturing.valid)
        );
    }

    #[test]
    fn manufacturing_operating_envelope_covers_each_corner_and_rpm() {
        let mut config = PcFanConfig::bench_120mm();
        config.weights.airflow_band_fraction = 2.0;
        config.structural.min_modal_separation_hz = 0.0;
        config.structural.max_imbalance_g_mm = 100.0;
        config.operating_point_spread.rpms = vec![900.0, 400.0, 1200.0, 600.0];
        config.manufacturing.tolerance = PcFanManufacturingTolerance {
            blade_thickness_fraction: 0.02,
            blade_chord_fraction: 0.02,
            twist_deg: 1.0,
            shroud_offset_m: 0.00025,
        };
        let design = config.default_design_variables();

        let first = evaluate_pc_fan_manufacturing_operating_envelope(&config, &design)
            .expect("valid manufacturing operating envelope");
        let second = evaluate_pc_fan_manufacturing_operating_envelope(&config, &design)
            .expect("deterministic manufacturing operating envelope");

        assert_eq!(first, second);
        assert_eq!(first.variation_count, 17);
        assert_eq!(first.rpm_count, 4);
        assert_eq!(first.points.len(), 68);
        assert!(first.finite);
        assert!(
            first.all_points_feasible,
            "envelope was not feasible: {first:?}"
        );
        assert!(first.all_variations_monotonic);
        assert!(first.has_tolerance_coverage());
        assert!(first.is_robust());

        for chunk in first.points.chunks_exact(first.rpm_count) {
            assert_eq!(chunk[0].variation, chunk[1].variation);
            assert_eq!(
                chunk.iter().map(|point| point.rpm).collect::<Vec<_>>(),
                vec![400.0, 600.0, 900.0, 1200.0]
            );
            assert!(chunk.iter().all(|point| point.objective.is_finite()));
        }
    }

    #[test]
    fn manufacturing_operating_envelope_rejects_invalid_rpm_and_cfd() {
        let mut config = PcFanConfig::bench_120mm();
        config.operating_point_spread.rpms = vec![600.0, f64::NAN];
        let design = config.default_design_variables();
        assert!(matches!(
            evaluate_pc_fan_manufacturing_operating_envelope(&config, &design),
            Err(PcFanManufacturingOperatingEnvelopeError::InvalidRpm { index: 1, .. })
        ));

        config.operating_point_spread.rpms = vec![600.0, 900.0];
        config.cfd.flow_model = PcFanFlowModel::IncompressibleCfd;
        assert!(matches!(
            evaluate_pc_fan_manufacturing_operating_envelope(&config, &design),
            Err(
                PcFanManufacturingOperatingEnvelopeError::UnsupportedFlowModel(
                    PcFanFlowModel::IncompressibleCfd
                )
            )
        ));
    }

    #[test]
    fn manufacturing_operating_envelope_requires_nonzero_tolerance_and_two_rpms_for_robustness() {
        let mut config = PcFanConfig::bench_120mm();
        config.weights.airflow_band_fraction = 2.0;
        config.structural.min_modal_separation_hz = 0.0;
        config.structural.max_imbalance_g_mm = 100.0;
        config.operating_point_spread.rpms = vec![600.0];
        let design = config.default_design_variables();

        let envelope = evaluate_pc_fan_manufacturing_operating_envelope(&config, &design)
            .expect("nominal single-point envelope is still reportable");
        assert!(envelope.is_valid());
        assert!(!envelope.has_tolerance_coverage());
        assert!(!envelope.is_robust());
    }

    #[test]
    fn reduced_fan_curve_has_ordered_pressure_and_flow_endpoints() {
        let mut config = PcFanConfig::bench_120mm();
        config.mesh = PcFanMeshResolution {
            radial_segments: 4,
            chord_segments: 2,
            circumferential_segments: 8,
            harmonic_count: 2,
        };
        config.weights.airflow_band_fraction = 1.0;
        let curve = evaluate_pc_fan_fan_curve(
            &config,
            &config.default_design_variables(),
            900.0,
            &[0.0, 0.5, 1.0],
        )
        .expect("valid reduced-order fan curve");

        assert!(curve.is_valid(), "invalid fan curve: {curve:?}");
        assert!(curve.free_airflow_cfm > 0.0);
        assert!(curve.shutoff_pressure_pa > 0.0);
        assert_eq!(curve.points[0].airflow_cfm, 0.0);
        assert_eq!(
            curve.points[0].fan_pressure_rise_pa,
            curve.shutoff_pressure_pa
        );
        assert_eq!(curve.points[2].airflow_cfm, curve.free_airflow_cfm);
        assert!(curve.points[2].fan_pressure_rise_pa.abs() < 1.0e-12);
        assert!(
            curve
                .points
                .iter()
                .all(|point| point.system_pressure_rise_pa.is_finite())
        );
        let operating_point = curve
            .operating_point
            .expect("fan and system curves should intersect");
        assert!(operating_point.converged);
        assert!((0.0..=1.0).contains(&operating_point.flow_fraction));
        assert!(operating_point.residual_pa.abs() < 1.0e-6);
        assert!(
            (operating_point.fan_pressure_rise_pa - operating_point.system_pressure_rise_pa).abs()
                < 1.0e-6
        );
    }

    #[test]
    fn reduced_fan_curve_rejects_invalid_sampling_contracts() {
        let config = PcFanConfig::bench_120mm();
        let design = config.default_design_variables();

        assert!(matches!(
            evaluate_pc_fan_fan_curve(&config, &design, 0.0, &[0.0, 1.0]),
            Err(PcFanFanCurveError::InvalidRpm(0.0))
        ));
        assert!(matches!(
            evaluate_pc_fan_fan_curve(&config, &design, 600.0, &[0.0]),
            Err(PcFanFanCurveError::NotEnoughSamples)
        ));
        assert!(matches!(
            evaluate_pc_fan_fan_curve(&config, &design, 600.0, &[0.0, f64::NAN]),
            Err(PcFanFanCurveError::InvalidFlowFraction { index: 1, .. })
        ));
        assert!(matches!(
            evaluate_pc_fan_fan_curve(&config, &design, 600.0, &[0.0, 0.0, 1.0]),
            Err(PcFanFanCurveError::NonIncreasingFlowFractions { index: 1, .. })
        ));
    }

    #[test]
    fn reduced_fan_curve_fails_closed_for_resolved_flow() {
        let mut config = PcFanConfig::bench_120mm();
        config.cfd.flow_model = PcFanFlowModel::IncompressibleCfd;
        let design = config.default_design_variables();

        assert!(matches!(
            evaluate_pc_fan_fan_curve(&config, &design, 600.0, &[0.0, 1.0]),
            Err(PcFanFanCurveError::UnsupportedFlowModel(
                PcFanFlowModel::IncompressibleCfd
            ))
        ));
    }

    #[test]
    fn fan_curve_objective_is_zero_on_its_reference_curve() {
        let config = PcFanConfig::bench_120mm();
        let design = config.default_design_variables();
        let fractions = vec![0.0, 0.5, 1.0];
        let curve = evaluate_pc_fan_fan_curve(&config, &design, 600.0, &fractions).unwrap();
        let objective_config = PcFanFanCurveObjectiveConfig {
            rpm: 600.0,
            flow_fractions: fractions,
            target_airflow_cfm: curve.points.iter().map(|point| point.airflow_cfm).collect(),
            target_pressure_rise_pa: curve
                .points
                .iter()
                .map(|point| point.fan_pressure_rise_pa)
                .collect(),
            target_acoustic_level_db: Some(
                curve
                    .points
                    .iter()
                    .map(|point| point.acoustic_level_db)
                    .collect(),
            ),
            target_efficiency: Some(curve.points.iter().map(|point| point.efficiency).collect()),
            airflow_weight: 1.0,
            pressure_weight: 1.0,
            acoustic_weight: 0.1,
            efficiency_weight: 0.1,
        };
        let scored = evaluate_pc_fan_fan_curve_objective(&config, &design, &objective_config)
            .expect("reference curve scores");

        assert!(scored.feasible);
        assert!(scored.score < 1.0e-24, "score={}", scored.score);
        assert!(scored.airflow_rmse_cfm < 1.0e-12);
        assert!(scored.pressure_rmse_pa < 1.0e-12);
        assert!(scored.acoustic_rmse_db.unwrap() < 1.0e-12);
        assert!(scored.efficiency_rmse.unwrap() < 1.0e-12);
    }

    #[test]
    fn fan_curve_objective_rejects_malformed_targets() {
        let config = PcFanConfig::bench_120mm();
        let design = config.default_design_variables();
        let base = PcFanFanCurveObjectiveConfig {
            rpm: 600.0,
            flow_fractions: vec![0.0, 1.0],
            target_airflow_cfm: vec![0.0, 1.0],
            target_pressure_rise_pa: vec![1.0, 0.0],
            target_acoustic_level_db: None,
            target_efficiency: None,
            airflow_weight: 1.0,
            pressure_weight: 1.0,
            acoustic_weight: 0.0,
            efficiency_weight: 0.0,
        };
        let mut wrong_length = base.clone();
        wrong_length.target_airflow_cfm.pop();
        assert!(matches!(
            evaluate_pc_fan_fan_curve_objective(&config, &design, &wrong_length),
            Err(PcFanFanCurveError::TargetLengthMismatch {
                field: "target_airflow_cfm",
                ..
            })
        ));

        let mut bad_weight = base.clone();
        bad_weight.pressure_weight = f64::NAN;
        assert!(matches!(
            evaluate_pc_fan_fan_curve_objective(&config, &design, &bad_weight),
            Err(PcFanFanCurveError::InvalidObjectiveWeight {
                field: "pressure_weight",
                ..
            })
        ));
    }

    #[test]
    fn fast_path_skips_when_weight_zero() {
        let config = PcFanConfig::default_120mm();
        let design = config.default_design_variables();
        let geometry = config.build_geometry(&design);
        let base = PcFanEvaluation::from_design(&config, &design, &geometry);
        assert!(base.operating_point_spls_db.is_empty());
        assert_eq!(base.operating_point_spread_db, 0.0);
        assert_eq!(base.operating_point_spread_penalty, 0.0);

        // attach with zero weight should be a no-op.
        let updated = attach_operating_point_spread(&config, &design, &geometry, base.clone());
        assert!(updated.operating_point_spls_db.is_empty());
        assert_eq!(updated.objective, base.objective);
    }

    #[test]
    fn populates_spread_when_weight_nonzero() {
        let mut config = PcFanConfig::default_120mm();
        config.weights.operating_spread_weight = 1.0;
        config.operating_point_spread.rpms = vec![400.0, 600.0, 800.0];
        let design = config.default_design_variables();
        let geometry = config.build_geometry(&design);
        let base = PcFanEvaluation::from_design(&config, &design, &geometry);
        let updated = attach_operating_point_spread(&config, &design, &geometry, base.clone());
        assert_eq!(updated.operating_point_spls_db.len(), 3);
        assert!(updated.operating_point_spread_db > 0.0);
        assert!(updated.operating_point_spread_penalty > 0.0);
        assert!(updated.objective > base.objective);
    }

    #[test]
    fn single_rpm_gives_zero_spread() {
        let mut config = PcFanConfig::default_120mm();
        config.weights.operating_spread_weight = 1.0;
        config.operating_point_spread.rpms = vec![600.0]; // only one entry
        let design = config.default_design_variables();
        let geometry = config.build_geometry(&design);
        let base = PcFanEvaluation::from_design(&config, &design, &geometry);
        let updated = attach_operating_point_spread(&config, &design, &geometry, base.clone());
        // Helper requires at least 2 entries; falls through unchanged.
        assert!(updated.operating_point_spls_db.is_empty());
        assert_eq!(updated.operating_point_spread_db, 0.0);
    }
}

#[cfg(test)]
mod turbulence_integration_tests {
    use super::*;

    fn small_cfd_config() -> PcFanConfig {
        PcFanConfig {
            cfd: PcFanCfdConfig {
                flow_model: PcFanFlowModel::IncompressibleCfd,
                cells_x: 2,
                cells_y: 2,
                cells_z: 2,
                max_picard_iterations: 4,
                nonlinear_tolerance: 1.0e-3,
                ..PcFanCfdConfig::default()
            },
            mesh: PcFanMeshResolution {
                radial_segments: 2,
                chord_segments: 2,
                circumferential_segments: 8,
                harmonic_count: 2,
            },
            ..PcFanConfig::default_120mm()
        }
    }

    #[test]
    fn cfd_runs_with_turbulence_enabled() {
        let mut config = small_cfd_config();
        config.cfd.turbulence = PcFanTurbulenceModel::KOmegaSstAlgebraic;
        let design = config.default_design_variables();
        let evaluation = evaluate_pc_fan(&config, &design);
        // Both airflow and SPL must remain finite under the algebraic SST.
        assert!(
            evaluation.airflow_cfm.is_finite() && evaluation.airflow_cfm > 0.0,
            "airflow {}",
            evaluation.airflow_cfm
        );
        assert!(evaluation.acoustic_level_db.is_finite());
    }

    #[test]
    fn turbulent_solve_differs_from_laminar() {
        // Same geometry, two flow models — the converged airflow should
        // differ because the algebraic eddy viscosity inflates μ_eff and
        // reshapes the velocity field.
        let mut laminar = small_cfd_config();
        laminar.cfd.turbulence = PcFanTurbulenceModel::Laminar;
        let mut turbulent = small_cfd_config();
        turbulent.cfd.turbulence = PcFanTurbulenceModel::KOmegaSstAlgebraic;

        let design = laminar.default_design_variables();
        let lam = evaluate_pc_fan(&laminar, &design);
        let turb = evaluate_pc_fan(&turbulent, &design);

        // Both must be finite and positive.
        assert!(lam.airflow_cfm > 0.0);
        assert!(turb.airflow_cfm > 0.0);
        // The two airflows must be different by a meaningful amount.
        let rel_diff = (turb.airflow_cfm - lam.airflow_cfm).abs() / lam.airflow_cfm.max(1.0e-6);
        assert!(
            rel_diff > 0.01,
            "expected turbulent airflow to differ from laminar by >1%, got {:.2}% (lam={:.3}, turb={:.3})",
            rel_diff * 100.0,
            lam.airflow_cfm,
            turb.airflow_cfm
        );
    }
}

#[cfg(test)]
mod a_weighting_tests {
    use super::*;

    /// At the default 600 rpm operating point the BPF (80 Hz) and its
    /// low harmonics fall deep in the A-weighting attenuation band. The
    /// A-weighted overall must be strictly below the unweighted overall
    /// for any non-trivial harmonic content.
    #[test]
    fn a_weighted_overall_is_below_unweighted_at_low_bpf() {
        let config = PcFanConfig::default_120mm();
        let design = config.default_design_variables();
        let eval = evaluate_pc_fan(&config, &design);
        let lin = eval.acoustics.overall_spl_db;
        let aw = eval.acoustics.a_weighted_overall_spl_db;
        assert!(
            lin > 0.0 && aw.is_finite(),
            "expected finite SPLs (lin {lin}, aw {aw})"
        );
        assert!(
            aw < lin - 5.0,
            "A-weighted overall {aw:.2} dB(A) should be well below unweighted {lin:.2} dB at BPF ≈ {:.0} Hz",
            eval.acoustics.blade_passing_frequency_hz
        );
    }

    /// The optimiser objective must equal the A-weighted level (times
    /// `acoustic_weight`) plus the non-acoustic penalties when
    /// `a_weighting_enabled = true` (the default).
    #[test]
    fn objective_uses_a_weighted_level_by_default() {
        let mut config = PcFanConfig::default_120mm();
        // Isolate the acoustic term by zeroing every other contribution.
        config.weights.airflow_penalty_weight = 0.0;
        config.weights.airflow_band_fraction = 0.0;
        config.weights.spectral_spread_weight = 0.0;
        config.weights.directivity_spread_weight = 0.0;
        config.weights.shroud_clearance_penalty_weight = 0.0;
        config.weights.frame_containment_penalty_weight = 0.0;
        config.weights.hub_attachment_penalty_weight = 0.0;
        config.weights.tonality_penalty_weight = 0.0;
        let design = config.default_design_variables();
        let eval = evaluate_pc_fan(&config, &design);
        let expected = config.weights.acoustic_weight * eval.acoustics.a_weighted_overall_spl_db;
        let diff = (eval.objective - expected).abs();
        assert!(
            diff < 1.0e-9,
            "objective {} differs from expected {} (A-weighted) by {}",
            eval.objective,
            expected,
            diff
        );
    }

    /// Disabling A-weighting reverts the objective to the unweighted
    /// level. Used as a regression check that the toggle takes effect.
    #[test]
    fn disabling_a_weighting_reverts_to_unweighted_level() {
        let mut config = PcFanConfig::default_120mm();
        config.weights.airflow_penalty_weight = 0.0;
        config.weights.airflow_band_fraction = 0.0;
        config.weights.spectral_spread_weight = 0.0;
        config.weights.directivity_spread_weight = 0.0;
        config.weights.shroud_clearance_penalty_weight = 0.0;
        config.weights.frame_containment_penalty_weight = 0.0;
        config.weights.hub_attachment_penalty_weight = 0.0;
        config.weights.tonality_penalty_weight = 0.0;
        config.weights.a_weighting_enabled = false;
        let design = config.default_design_variables();
        let eval = evaluate_pc_fan(&config, &design);
        let expected = config.weights.acoustic_weight * eval.acoustics.overall_spl_db;
        let diff = (eval.objective - expected).abs();
        assert!(
            diff < 1.0e-9,
            "objective {} differs from expected {} (unweighted) by {}",
            eval.objective,
            expected,
            diff
        );
    }

    /// The A-weighted overall must respect IEC 61672-1: at fixed
    /// per-harmonic pressure, shifting BPF from 80 Hz to ~1 kHz must
    /// raise the A-weighted level (because the A-weighting factor
    /// approaches unity near 1 kHz) — even though the unweighted level
    /// is unchanged. We synthesise a harmonic vector and call the
    /// helper directly.
    #[test]
    fn a_weighted_level_rises_as_content_moves_toward_1khz() {
        let mut low = Vec::with_capacity(5);
        let mut high = Vec::with_capacity(5);
        for h in 1..=5_usize {
            let pressure_pa = 1.0e-3;
            low.push(PcFanAcousticHarmonic {
                harmonic: h,
                frequency_hz: 80.0 * h as f64,
                pressure_pa,
                sound_pressure_level_db: pressure_level_db(pressure_pa),
                sound_power_level_db: 0.0,
            });
            high.push(PcFanAcousticHarmonic {
                harmonic: h,
                frequency_hz: 1000.0 * h as f64,
                pressure_pa,
                sound_pressure_level_db: pressure_level_db(pressure_pa),
                sound_power_level_db: 0.0,
            });
        }
        let aw_low = a_weighted_overall_db_from_harmonics(&low);
        let aw_high = a_weighted_overall_db_from_harmonics(&high);
        // Difference is dominated by the lowest harmonic (α(80 Hz) ≈ 0.063
        // vs α(1 kHz) = 1.0 → ~24 dB on that bin), but higher bins of the
        // low band partially close the gap (α(400 Hz) ≈ 0.71). Empirically
        // the gap on this 5-harmonic synthesised spectrum lands ~9 dB.
        assert!(
            aw_high > aw_low + 5.0,
            "1 kHz-band A-weighted level {aw_high:.2} should exceed 80 Hz-band {aw_low:.2} by >5 dB"
        );
    }

    /// Tonality penalty is zero for a flat spectrum and positive when a
    /// single interior bin dominates. Exercises the
    /// `perceptual_power_spectrum` helper directly because
    /// `sound_metrics::tonality_penalty` excludes the spectrum endpoints
    /// — the per-harmonic vector exposed by the public API is short
    /// enough that the dominant BPF bin lives at index 0 and never
    /// triggers the metric on a vanilla design. The point of this test
    /// is to verify the wiring: weighting → power vector → penalty.
    #[test]
    fn tonality_penalty_fires_on_synthetic_tone() {
        let mut harmonics = Vec::with_capacity(7);
        for h in 1..=7_usize {
            harmonics.push(PcFanAcousticHarmonic {
                harmonic: h,
                frequency_hz: 1000.0 * h as f64, // park in the A-weight flat region
                pressure_pa: 1.0e-3,
                sound_pressure_level_db: 0.0,
                sound_power_level_db: 0.0,
            });
        }
        // Inject a 30 dB tone at an interior bin (index 3 → 4th
        // harmonic), so the tonality metric's neighbour-floor sees a
        // strong prominence.
        harmonics[3].pressure_pa *= 10.0_f64.powf(30.0 / 20.0);

        let flat_powers = perceptual_power_spectrum(true, &harmonics[..3]); // too short → 0
        assert_eq!(sound_metrics::tonality_penalty(&flat_powers, 3.0), 0.0);

        let tonal_powers = perceptual_power_spectrum(true, &harmonics);
        let penalty = sound_metrics::tonality_penalty(&tonal_powers, 3.0);
        assert!(penalty > 10.0, "expected >10 dB tonality, got {penalty}");
    }

    /// Default config keeps tonality off so the new term is opt-in.
    #[test]
    fn tonality_penalty_default_weight_is_zero() {
        let config = PcFanConfig::default_120mm();
        assert_eq!(config.weights.tonality_penalty_weight, 0.0);
        let design = config.default_design_variables();
        let eval = evaluate_pc_fan(&config, &design);
        assert_eq!(eval.tonality_penalty, 0.0);
    }
}

#[cfg(test)]
mod chassis_propagation_tests {
    use super::*;

    /// Default propagation mode is `DuctCutoff` so existing studies
    /// (that pre-date Phase 3) continue to see the legacy bench-style
    /// model. Switching to `RigidCavity` is opt-in.
    #[test]
    fn default_mode_is_duct_cutoff() {
        let config = PcFanConfig::default_120mm();
        assert_eq!(
            config.enclosure.mode,
            PcFanPropagationMode::DuctCutoff,
            "default propagation mode regression"
        );
    }

    /// Free-field mode is unit gain at every frequency.
    #[test]
    fn free_field_is_unit_gain() {
        let mut config = PcFanConfig::default_120mm();
        config.enclosure = PcFanEnclosureConfig::free_field();
        for &f in &[10.0_f64, 100.0, 800.0, 1000.0, 5000.0] {
            let g = propagation_gain(&config, f);
            assert!(
                (g - 1.0).abs() < 1.0e-12,
                "free-field gain at {f} Hz = {g}, expected 1.0"
            );
        }
    }

    /// Rigid-cavity gain peaks near the analytical first-mode frequency
    /// of a 0.40 × 0.40 × 0.20 m box: f_100 = c / (2·L_x) ≈ 428 Hz.
    /// Place source and listener on the same (1,0,0) anti-node so the
    /// resonant contribution has the same phase at both points.
    #[test]
    fn rigid_cavity_resonance_amplifies_at_first_mode() {
        let mut config = PcFanConfig::default_120mm();
        let lx = 0.40_f64;
        config.enclosure = PcFanEnclosureConfig {
            mode: PcFanPropagationMode::RigidCavity,
            chassis_dimensions_m: [lx, 0.40, 0.20],
            source_position_m: [0.0, 0.20, 0.10],
            listener_position_m: [0.0, 0.20, 0.10],
            modal_damping: 0.05,
            max_mode_index: 4,
            reference_distance_m: 1.0,
        };
        let c = config.operating_point.speed_of_sound_m_s;
        let f_first_mode = c / (2.0 * lx);
        let on_resonance = propagation_gain(&config, f_first_mode);
        let off_resonance_low = propagation_gain(&config, 200.0);
        assert!(
            on_resonance > off_resonance_low * 1.5,
            "expected resonance amplification: on {on_resonance:.3} vs off {off_resonance_low:.3}"
        );
        let expected_f100 = 343.0 / (2.0 * 0.40);
        assert!(
            (f_first_mode - expected_f100).abs() < 5.0,
            "first-mode prediction off: {f_first_mode} vs analytical {expected_f100}"
        );
    }

    #[test]
    fn opposite_cavity_antinodes_can_cancel() {
        let mut config = PcFanConfig::default_120mm();
        let lx = 0.40_f64;
        config.enclosure = PcFanEnclosureConfig {
            mode: PcFanPropagationMode::RigidCavity,
            chassis_dimensions_m: [lx, 0.40, 0.20],
            source_position_m: [0.0, 0.20, 0.10],
            listener_position_m: [lx, 0.20, 0.10],
            modal_damping: 0.05,
            max_mode_index: 1,
            reference_distance_m: 1.0,
        };
        let f_first_mode = config.operating_point.speed_of_sound_m_s / (2.0 * lx);
        assert!(rigid_cavity_pressure_gain(&config, f_first_mode) < 1.0);
    }

    /// Rigid-cavity damping reduces the resonance peak — exercises the
    /// `modal_damping` parameter.
    #[test]
    fn rigid_cavity_damping_reduces_peak_amplification() {
        let lx = 0.40_f64;
        let make_config = |eta: f64| {
            let mut config = PcFanConfig::default_120mm();
            config.enclosure = PcFanEnclosureConfig {
                mode: PcFanPropagationMode::RigidCavity,
                chassis_dimensions_m: [lx, 0.40, 0.20],
                source_position_m: [0.0, 0.20, 0.10],
                listener_position_m: [0.0, 0.20, 0.10],
                modal_damping: eta,
                max_mode_index: 4,
                reference_distance_m: 1.0,
            };
            config
        };
        let f_first_mode = 343.0 / (2.0 * lx);
        let g_low_damping = propagation_gain(&make_config(0.01), f_first_mode);
        let g_high_damping = propagation_gain(&make_config(0.20), f_first_mode);
        assert!(
            g_low_damping > g_high_damping * 1.1,
            "expected lower damping → higher peak: low {g_low_damping:.3} vs high {g_high_damping:.3}"
        );
    }

    /// `chassis_installed_120mm` activates the Amiet ingestion-noise
    /// term and the rigid-cavity propagation, both of which raise the
    /// installed level above the bench level — the user's "ingestion
    /// often wins" point made measurable.
    #[test]
    fn chassis_installed_preset_is_louder_than_bench_preset() {
        let bench = PcFanConfig::bench_120mm();
        let chassis = PcFanConfig::chassis_installed_120mm();
        let bench_design = bench.default_design_variables();
        let chassis_design = chassis.default_design_variables();
        let bench_eval = evaluate_pc_fan(&bench, &bench_design);
        let chassis_eval = evaluate_pc_fan(&chassis, &chassis_design);
        assert!(
            chassis_eval.acoustics.overall_spl_db > bench_eval.acoustics.overall_spl_db,
            "chassis-installed preset should be louder than bench: {:.2} vs {:.2}",
            chassis_eval.acoustics.overall_spl_db,
            bench_eval.acoustics.overall_spl_db
        );
        // Ingestion path active in chassis preset.
        assert!(
            chassis_eval.acoustics.leading_edge_inflow_spl_db
                > bench_eval.acoustics.leading_edge_inflow_spl_db + 20.0,
            "Amiet LE-noise should fire in chassis preset (TI > 0): \
             chassis {} dB, bench {} dB",
            chassis_eval.acoustics.leading_edge_inflow_spl_db,
            bench_eval.acoustics.leading_edge_inflow_spl_db,
        );
    }

    /// End-to-end: the chassis-coloration path actually changes the
    /// optimisation objective. Switching from free-field to rigid-cavity
    /// must move the per-harmonic SPLs (the user's plan claims 5–15 dB
    /// of coloration; this test only requires non-trivial movement).
    #[test]
    fn rigid_cavity_changes_overall_spl_vs_free_field() {
        let mut config = PcFanConfig::default_120mm();
        let design = config.default_design_variables();

        config.enclosure = PcFanEnclosureConfig::free_field();
        let free = evaluate_pc_fan(&config, &design);

        config.enclosure = PcFanEnclosureConfig::mid_tower();
        let cavity = evaluate_pc_fan(&config, &design);

        let delta = (cavity.acoustics.overall_spl_db - free.acoustics.overall_spl_db).abs();
        assert!(
            delta > 0.5,
            "expected non-trivial chassis coloration, got Δ={delta:.2} dB \
             (free {:.2} → cavity {:.2})",
            free.acoustics.overall_spl_db,
            cavity.acoustics.overall_spl_db
        );
    }
}

#[cfg(test)]
mod physical_scaling_tests {
    use super::*;

    #[test]
    fn observer_pressure_obeys_inverse_distance_spreading() {
        let mut config = PcFanConfig::bench_120mm();
        config.broadband = PcFanBroadbandConfig::off();
        config.observers.distance_m = 1.0;
        let design = config.default_design_variables();
        let acoustics = solve_pc_fan_acoustics(&config, &design);
        let at_one_metre = observer_spl_db(&config, &acoustics, 0.7, 0.3);
        config.observers.distance_m = 2.0;
        let at_two_metres = observer_spl_db(&config, &acoustics, 0.7, 0.3);
        let expected_drop_db = 20.0 * 2.0_f64.log10();
        assert!((at_one_metre - at_two_metres - expected_drop_db).abs() < 1.0e-10);
    }

    #[test]
    fn free_field_swl_matches_spherical_rms_intensity() {
        let mut config = PcFanConfig::bench_120mm();
        config.broadband = PcFanBroadbandConfig::off();
        config.blade_count = 2;
        config.operating_point.rpm = 18_000.0;
        let design = config.default_design_variables();
        let acoustics = solve_pc_fan_acoustics(&config, &design);
        let r = acoustic_reference_distance_m(&config);
        let expected_delta_db = 10.0
            * (4.0 * std::f64::consts::PI * r * r * PRESSURE_REFERENCE_PA.powi(2)
                / (config.operating_point.air_density_kg_m3
                    * config.operating_point.speed_of_sound_m_s
                    * SOUND_POWER_REFERENCE_W))
                .log10();
        assert!(
            (acoustics.overall_swl_db - acoustics.overall_spl_db - expected_delta_db).abs()
                < 1.0e-10
        );
    }

    #[test]
    fn reduced_flow_mach_uses_configured_sound_speed() {
        let mut config = PcFanConfig::default_120mm();
        config.operating_point.speed_of_sound_m_s = 500.0;
        let design = config.default_design_variables();
        let geometry = config.build_geometry(&design);
        let flow = flow_from_airflow_cfm(&config, &design, &geometry, 10.0);
        assert!((flow.mach_number - flow.axial_velocity_m_s.abs() / 500.0).abs() < 1.0e-12);
    }

    #[test]
    fn fixed_hardware_blockage_uses_open_annulus() {
        let config = PcFanConfig::default_120mm();
        let design = config.default_design_variables();
        let geometry = config.build_geometry(&design);
        let effects = fixed_hardware_flow_effects(&config, &geometry);
        let annulus = std::f64::consts::PI
            * (geometry.blade_tip_radius_m.powi(2) - geometry.hub.radius_m.powi(2));
        assert!((effects.blockage_ratio - effects.blockage_area_m2 / annulus).abs() < 1.0e-12);
    }

    #[test]
    fn duct_cutoff_uses_deformed_shroud_radius() {
        let mut config = PcFanConfig::default_120mm();
        config.enclosure.mode = PcFanPropagationMode::DuctCutoff;
        config.broadband.duct_cutoff_enabled = true;
        let baseline_design = config.default_design_variables();
        let mut tighter_design = baseline_design.clone();
        tighter_design.shroud_radius_offsets_m = [-0.001; SHROUD_CONTROL_COUNT];
        let baseline = config.build_geometry(&baseline_design);
        let tighter = config.build_geometry(&tighter_design);
        let baseline_pressure =
            apply_duct_cutoff_filter_for_geometry(&config, &baseline, 100.0, 1.0);
        let tighter_pressure = apply_duct_cutoff_filter_for_geometry(&config, &tighter, 100.0, 1.0);
        assert!(tighter_pressure < baseline_pressure);
    }

    #[test]
    fn leading_edge_pressure_has_dimensionally_correct_sound_speed_scaling() {
        let mut base = PcFanConfig::default_120mm();
        base.operating_point.inflow_turbulence_intensity = 0.1;
        base.operating_point.speed_of_sound_m_s = 343.0;
        base.operating_point.rpm = 600.0;
        let design = base.default_design_variables();
        let flow = PcFanFlowSolution::zero();
        let p1 = leading_edge_inflow_pressures_per_harmonic(&base, &design, &flow, 6);

        let mut scaled = base.clone();
        scaled.operating_point.speed_of_sound_m_s *= 2.0;
        scaled.operating_point.rpm *= 2.0;
        let p2 = leading_edge_inflow_pressures_per_harmonic(&scaled, &design, &flow, 6);
        let rms1 = p1.iter().map(|p| p * p).sum::<f64>().sqrt();
        let rms2 = p2.iter().map(|p| p * p).sum::<f64>().sqrt();
        assert!((rms2 / rms1 - 4.0).abs() < 1.0e-10);
    }

    #[test]
    fn tonal_peak_phasor_is_converted_to_rms() {
        let mut config = PcFanConfig::bench_120mm();
        config.blade_count = 2;
        config.operating_point.rpm = 18_000.0;
        config.mesh.harmonic_count = 1;
        let design = config.default_design_variables();
        let geometry = config.build_geometry(&design);
        let flow = solve_mean_flow(&config, &design, &geometry);
        let source = propeller_source_for(&config, &design, &flow);
        let peak = source.harmonic_amplitudes(1)[0];
        let shape = acoustic_shape_factor(&config, &design, &geometry, &flow);
        let rms = blade_acoustic_pressures(&config, &design, &geometry, &flow)[0];
        assert!((rms / (peak * shape) - 1.0 / 2.0_f64.sqrt()).abs() < 1.0e-12);
    }

    #[test]
    fn acoustic_rhs_includes_tip_leakage() {
        let mut off = PcFanConfig::bench_120mm();
        off.broadband = PcFanBroadbandConfig::off();
        let design = off.default_design_variables();
        let geometry = off.build_geometry(&design);
        let airflow = mean_flow_airflow_rhs(&off, &design, &geometry);
        let rhs_off = acoustic_pressure_rhs(&off, &design, &geometry, airflow);
        let mut on = off.clone();
        on.broadband.tip_leakage_enabled = true;
        let rhs_on = acoustic_pressure_rhs(&on, &design, &geometry, airflow);
        assert!(rhs_on.iter().zip(rhs_off).any(|(on, off)| on > &off));
    }

    #[test]
    fn zero_load_acoustics_preserve_zero_harmonic_pressure() {
        let mut config = PcFanConfig::bench_120mm();
        config.mesh.harmonic_count = 3;
        config.operating_point.rpm = 0.0;
        config.broadband = PcFanBroadbandConfig::off();
        let design = config.default_design_variables();
        let geometry = config.build_geometry(&design);
        let flow = PcFanFlowSolution::zero();

        let source = propeller_source_for(&config, &design, &flow);
        assert!(
            source
                .harmonic_amplitudes(config.mesh.harmonic_count)
                .iter()
                .all(|pressure| *pressure == 0.0)
        );

        let acoustics = evaluate_acoustics(&config, &design, &geometry, &flow);
        assert!(
            acoustics
                .harmonics
                .iter()
                .all(|harmonic| harmonic.pressure_pa == 0.0)
        );
        assert!(acoustics.overall_spl_db.is_finite());
        assert!(acoustics.overall_swl_db.is_finite());
        let zero_state = acoustic_solution_with_pressure_state(
            acoustics.clone(),
            &vec![0.0; config.mesh.harmonic_count],
        );
        assert!(
            zero_state
                .harmonics
                .iter()
                .all(|harmonic| harmonic.pressure_pa == 0.0)
        );
        assert!(zero_state.overall_spl_db.is_finite());
        assert!(
            acoustic_pressure_rhs(&config, &design, &geometry, 0.0)
                .iter()
                .all(|pressure| *pressure == 0.0)
        );
    }

    #[test]
    fn reduced_adjoint_rejects_zero_acoustic_state_at_diagnostic_floor() {
        let mut config = PcFanConfig::bench_120mm();
        config.operating_point.rpm = 0.0;
        config.broadband = PcFanBroadbandConfig::off();
        let design = config.default_design_variables();

        assert!(matches!(
            solve_pc_fan_shape_gradient_reduced_adjoint(&config, &design),
            Err(PcFanDerivativePolicyError::AcousticStateAtDiagnosticFloor { harmonic: 1 })
        ));
    }

    #[test]
    fn feasibility_rejects_nonfinite_or_negative_acceptance_configuration() {
        let config = PcFanConfig::default_120mm();
        let design = config.default_design_variables();
        let geometry = config.build_geometry(&design);
        let mut evaluation = PcFanEvaluation::geometry_only(&config, &geometry);
        evaluation.airflow_cfm = config.operating_point.target_airflow_cfm;
        evaluation.objective = 0.0;
        evaluation.constraint_margins.airflow_cfm = 0.0;
        evaluation.constraint_margins.minimum_shroud_clearance_m =
            config.weights.shroud_clearance_min_m;
        assert!(evaluation.is_feasible(&config));

        let mut nonfinite_band = config.clone();
        nonfinite_band.weights.airflow_band_fraction = f64::NAN;
        assert!(!evaluation.is_feasible(&nonfinite_band));

        let mut negative_clearance = config;
        negative_clearance.weights.shroud_clearance_min_m = -1.0e-3;
        assert!(!evaluation.is_feasible(&negative_clearance));
    }
}
#[cfg(test)]
mod derivative_policy_tests {
    use super::*;

    #[test]
    fn finite_difference_policy_stays_finite_difference_for_reduced_order() {
        let config = PcFanConfig::default_120mm();
        let design = config.default_design_variables();
        let expected = finite_difference_shape_objective_gradient(&config, &design);
        let actual = solve_pc_fan_shape_gradient_with_policy(
            &config,
            &design,
            PcFanDerivativePolicy::FiniteDifferenceFullObjective,
        )
        .expect("reduced-order finite-difference policy should be admitted");

        assert_eq!(actual.objective_gradient, expected);
        assert!(
            actual.adjoint_is_finite_difference,
            "FD policy must tag its placeholder-zero adjoint"
        );
        assert_eq!(actual.adjoint.airflow_multiplier, 0.0);
        assert!(
            actual
                .adjoint
                .acoustic_pressure_multipliers
                .iter()
                .all(|value| *value == 0.0)
        );
    }

    #[test]
    fn finite_difference_shape_gradient_supports_bounded_resolved_cfd() {
        let mut config = PcFanConfig::default_120mm();
        config.cfd.flow_model = PcFanFlowModel::IncompressibleCfdFull;
        config.cfd.backend = PcFanCfdBackend::Cpu;
        config.cfd.cells_x = 2;
        config.cfd.cells_y = 2;
        config.cfd.cells_z = 2;
        config.cfd.max_picard_iterations = 1;
        config.cfd.max_relative_mass_imbalance = 10.0;
        config.cfd.max_relative_momentum_imbalance = 10.0;
        config.cfd.max_relative_pressure_closure = 10.0;
        config.cfd.max_relative_energy_imbalance = 10.0;
        config.cfd.nonlinear_tolerance = 2.0;
        config.cfd.linear_solver_max_iterations = 64;
        config.optimizer.finite_difference_step = 1.0e-4;
        let design = config.default_design_variables();

        let gradient = solve_pc_fan_shape_gradient_with_policy(
            &config,
            &design,
            PcFanDerivativePolicy::FiniteDifferenceFullObjective,
        )
        .expect("bounded resolved-CFD finite-difference gradient should be available");
        assert_eq!(gradient.objective_gradient.len(), DESIGN_VARIABLE_COUNT);
        assert!(
            gradient
                .objective_gradient
                .iter()
                .all(|value| value.is_finite())
        );
        assert!(
            gradient
                .objective_gradient
                .iter()
                .any(|value| value.abs() > f64::EPSILON)
        );
        assert!(gradient.residual.airflow_cfm.is_finite());
        assert!(gradient.residual.max_abs.is_finite());
    }

    #[test]
    fn checked_resolved_finite_difference_rejects_nonconverged_base() {
        let mut config = PcFanConfig::default_120mm();
        config.cfd.flow_model = PcFanFlowModel::IncompressibleCfdFull;
        config.cfd.backend = PcFanCfdBackend::Cpu;
        config.cfd.cells_x = 2;
        config.cfd.cells_y = 2;
        config.cfd.cells_z = 2;
        config.cfd.max_picard_iterations = 1;
        config.cfd.nonlinear_tolerance = 1.0e-12;
        let design = config.default_design_variables();

        assert!(matches!(
            solve_pc_fan_shape_gradient_with_policy(
                &config,
                &design,
                PcFanDerivativePolicy::FiniteDifferenceFullObjective,
            ),
            Err(
                PcFanDerivativePolicyError::FiniteDifferenceEvaluationFailed {
                    policy: PcFanDerivativePolicy::FiniteDifferenceFullObjective,
                    source: PcFanCfdEvaluationError::NonConverged { .. },
                }
            )
        ));
    }

    #[test]
    fn checked_resolved_finite_difference_propagates_conservation_rejection() {
        let mut config = PcFanConfig::default_120mm();
        config.cfd.flow_model = PcFanFlowModel::IncompressibleCfdFull;
        config.cfd.backend = PcFanCfdBackend::Cpu;
        config.cfd.cells_x = 2;
        config.cfd.cells_y = 2;
        config.cfd.cells_z = 2;
        config.cfd.max_picard_iterations = 1;
        config.cfd.nonlinear_tolerance = 2.0;
        config.cfd.max_relative_energy_imbalance = f64::NAN;
        let design = config.default_design_variables();

        assert!(matches!(
            solve_pc_fan_shape_gradient_with_policy(
                &config,
                &design,
                PcFanDerivativePolicy::FiniteDifferenceFullObjective,
            ),
            Err(
                PcFanDerivativePolicyError::FiniteDifferenceEvaluationFailed {
                    policy: PcFanDerivativePolicy::FiniteDifferenceFullObjective,
                    source: PcFanCfdEvaluationError::InvalidConservationTolerance {
                        metric: "energy imbalance",
                        ..
                    },
                },
            )
        ));
    }
}
