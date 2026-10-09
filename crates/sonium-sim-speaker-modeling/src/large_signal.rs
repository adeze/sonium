use num_complex::Complex64;
use serde::{Deserialize, Serialize};
use sonium_fem::physics::acoustics::electroacoustics::speaker::SpeakerResponse;
use sonium_fem::physics::acoustics::electroacoustics::thermal_coupling::VoiceCoilThermalModel;
use sonium_fem::physics::coupled::lumped_circuit::VoiceCoilParams;

use super::{
    LumpedNonlinearAcousticSpec, LumpedNonlinearDiagnostics, ModelingError, SpeakerDriverStudy,
    require_finite, require_nonnegative, require_positive,
};
use crate::lumped_nonlinear::{LumpedNonlinearRunConfig, run_lumped_nonlinear};
use crate::port_flow::analysis::{dft_amplitude, imd_fraction, port_spl_db, thd_fraction};

const MAX_TIME_STEPS: usize = 1_000_000;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LargeSignalProvenance {
    /// Lumped nonlinear electromechanical Newmark solve without an acoustic load;
    /// not a distributed FEM field solve.
    LumpedNonlinearElectromechanicalWithoutAcousticLoad,
    /// Nonlinear lumped solve with a passive frequency-domain exterior load
    /// mapped to equivalent damping and added mass at the drive frequency.
    /// This is a bounded two-way load adapter, not a distributed acoustic
    /// field solve.
    LumpedNonlinearElectromechanicalWithHarmonicAcousticLoad,
    /// M2 time-domain driver, sealed/vented box, and nonlinear port circuit.
    LumpedNonlinearElectromechanicalWithCoupledAcousticCircuit,
}

/// Passive acoustic load supplied by an enclosure/port/exterior solver.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct LargeSignalAcousticLoad {
    pub frequency_hz: f64,
    pub radiation_impedance_n_s_per_m: Complex64,
}

/// Amplitude-dependent viscoelastic damping surrogate for the suspension.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct MovingAssemblyViscoelasticSpec {
    /// Force coefficient in `N·s/m²`; force is `c * |x| * v`.
    pub damping_n_s_per_m2: f64,
}

impl MovingAssemblyViscoelasticSpec {
    fn validate(self) -> Result<(), ModelingError> {
        require_nonnegative(
            "moving_assembly.viscoelastic.damping_n_s_per_m2",
            self.damping_n_s_per_m2,
        )
    }
}

/// One-sided radial contact/rub surrogate after a specified clearance.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct MovingAssemblyContactSpec {
    pub clearance_m: f64,
    pub stiffness_n_per_m: f64,
    pub damping_n_s_per_m: f64,
}

impl MovingAssemblyContactSpec {
    fn validate(self) -> Result<(), ModelingError> {
        require_positive("moving_assembly.contact.clearance_m", self.clearance_m)?;
        require_nonnegative(
            "moving_assembly.contact.stiffness_n_per_m",
            self.stiffness_n_per_m,
        )?;
        require_nonnegative(
            "moving_assembly.contact.damping_n_s_per_m",
            self.damping_n_s_per_m,
        )?;
        if self.stiffness_n_per_m == 0.0 && self.damping_n_s_per_m == 0.0 {
            return Err(ModelingError::OutOfRange {
                field: "moving_assembly.contact",
                constraint: "positive stiffness or damping",
            });
        }
        Ok(())
    }
}

/// Breakup onset and bounded suspension-softening surrogate.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct MovingAssemblyBreakupSpec {
    pub onset_excursion_m: f64,
    pub softening_stiffness_n_per_m: f64,
}

impl MovingAssemblyBreakupSpec {
    fn validate(self) -> Result<(), ModelingError> {
        require_positive(
            "moving_assembly.breakup.onset_excursion_m",
            self.onset_excursion_m,
        )?;
        require_nonnegative(
            "moving_assembly.breakup.softening_stiffness_n_per_m",
            self.softening_stiffness_n_per_m,
        )
    }
}

/// Miner-style fatigue evidence for the reduced moving assembly.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct MovingAssemblyFatigueSpec {
    pub reference_excursion_m: f64,
    pub cycles_to_failure: f64,
    #[serde(default = "default_fatigue_damage_exponent")]
    pub damage_exponent: f64,
}

const fn default_fatigue_damage_exponent() -> f64 {
    3.0
}

impl MovingAssemblyFatigueSpec {
    fn validate(self) -> Result<(), ModelingError> {
        require_positive(
            "moving_assembly.fatigue.reference_excursion_m",
            self.reference_excursion_m,
        )?;
        require_positive(
            "moving_assembly.fatigue.cycles_to_failure",
            self.cycles_to_failure,
        )?;
        require_positive(
            "moving_assembly.fatigue.damage_exponent",
            self.damage_exponent,
        )
    }
}

/// Geometrically nonlinear moving-assembly reduction.
///
/// The quadratic term represents suspension asymmetry; the optional hard stop
/// is a bounded contact/rub surrogate and is intentionally reported as part
/// of the large-signal provenance rather than being mistaken for a linear
/// stiffness.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct MovingAssemblyNonlinearity {
    pub suspension_quadratic_n_per_m2: f64,
    pub suspension_cubic_n_per_m3: f64,
    pub hard_stop_excursion_m: Option<f64>,
    pub hard_stop_stiffness_n_per_m: f64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub viscoelastic: Option<MovingAssemblyViscoelasticSpec>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub contact: Option<MovingAssemblyContactSpec>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub breakup: Option<MovingAssemblyBreakupSpec>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fatigue: Option<MovingAssemblyFatigueSpec>,
}

impl MovingAssemblyNonlinearity {
    pub fn validate(self) -> Result<(), ModelingError> {
        require_finite(
            "moving_assembly.suspension_quadratic_n_per_m2",
            self.suspension_quadratic_n_per_m2,
        )?;
        require_nonnegative(
            "moving_assembly.suspension_cubic_n_per_m3",
            self.suspension_cubic_n_per_m3,
        )?;
        require_nonnegative(
            "moving_assembly.hard_stop_stiffness_n_per_m",
            self.hard_stop_stiffness_n_per_m,
        )?;
        if let Some(viscoelastic) = self.viscoelastic {
            viscoelastic.validate()?;
        }
        if let Some(contact) = self.contact {
            contact.validate()?;
        }
        if let Some(breakup) = self.breakup {
            breakup.validate()?;
        }
        if let Some(fatigue) = self.fatigue {
            fatigue.validate()?;
        }
        if let Some(excursion) = self.hard_stop_excursion_m {
            require_positive("moving_assembly.hard_stop_excursion_m", excursion)?;
            if self.hard_stop_stiffness_n_per_m == 0.0 {
                return Err(ModelingError::OutOfRange {
                    field: "moving_assembly.hard_stop_stiffness_n_per_m",
                    constraint: "positive when a hard stop is configured",
                });
            }
        }
        Ok(())
    }

    pub fn from_large_signal_spec(spec: LargeSignalSpec) -> Self {
        spec.moving_assembly.unwrap_or(Self {
            suspension_quadratic_n_per_m2: 0.0,
            suspension_cubic_n_per_m3: spec.suspension_cubic_n_per_m3,
            hard_stop_excursion_m: None,
            hard_stop_stiffness_n_per_m: 0.0,
            viscoelastic: None,
            contact: None,
            breakup: None,
            fatigue: None,
        })
    }

    pub fn nonlinear_force(self, displacement_m: f64) -> f64 {
        let mut force = self.suspension_quadratic_n_per_m2 * displacement_m.powi(2)
            + self.suspension_cubic_n_per_m3 * displacement_m.powi(3);
        if let Some(stop) = self.hard_stop_excursion_m {
            let penetration = displacement_m.abs() - stop;
            if penetration > 0.0 {
                force += displacement_m.signum()
                    * self.hard_stop_stiffness_n_per_m
                    * penetration.powi(2);
            }
        }
        if let Some(breakup) = self.breakup {
            let penetration = displacement_m.abs() - breakup.onset_excursion_m;
            if penetration > 0.0 {
                force -=
                    displacement_m.signum() * breakup.softening_stiffness_n_per_m * penetration;
            }
        }
        if let Some(contact) = self.contact {
            let penetration = displacement_m.abs() - contact.clearance_m;
            if penetration > 0.0 {
                force += displacement_m.signum() * contact.stiffness_n_per_m * penetration;
            }
        }
        force
    }

    /// Velocity-dependent viscoelastic and contact/rub force.
    pub fn nonlinear_damping_force(self, displacement_m: f64, velocity_m_per_s: f64) -> f64 {
        let mut force = self
            .viscoelastic
            .map(|viscoelastic| {
                viscoelastic.damping_n_s_per_m2 * displacement_m.abs() * velocity_m_per_s
            })
            .unwrap_or(0.0);
        if let Some(contact) = self.contact
            && displacement_m.abs() > contact.clearance_m
        {
            force += contact.damping_n_s_per_m * velocity_m_per_s;
        }
        force
    }

    pub fn contact_penetration_m(self, displacement_m: f64) -> f64 {
        self.contact
            .map(|contact| (displacement_m.abs() - contact.clearance_m).max(0.0))
            .unwrap_or(0.0)
    }

    pub fn breakup_active(self, displacement_m: f64) -> bool {
        self.breakup
            .is_some_and(|breakup| displacement_m.abs() > breakup.onset_excursion_m)
    }

    pub fn fatigue_damage(self, peak_excursion_m: f64, cycles: f64) -> f64 {
        self.fatigue
            .map(|fatigue| {
                cycles.max(0.0)
                    * (peak_excursion_m.abs() / fatigue.reference_excursion_m)
                        .powf(fatigue.damage_exponent)
                    / fatigue.cycles_to_failure
            })
            .unwrap_or(0.0)
    }

    pub fn potential_energy(self, displacement_m: f64) -> f64 {
        let mut energy = self.suspension_quadratic_n_per_m2 * displacement_m.powi(3) / 3.0
            + self.suspension_cubic_n_per_m3 * displacement_m.powi(4) / 4.0;
        if let Some(stop) = self.hard_stop_excursion_m {
            let penetration = displacement_m.abs() - stop;
            if penetration > 0.0 {
                energy += self.hard_stop_stiffness_n_per_m * penetration.powi(3) / 3.0;
            }
        }
        if let Some(breakup) = self.breakup {
            let penetration = displacement_m.abs() - breakup.onset_excursion_m;
            if penetration > 0.0 {
                energy -= breakup.softening_stiffness_n_per_m * penetration.powi(2) / 2.0;
            }
        }
        if let Some(contact) = self.contact {
            let penetration = displacement_m.abs() - contact.clearance_m;
            if penetration > 0.0 {
                energy += contact.stiffness_n_per_m * penetration.powi(2) / 2.0;
            }
        }
        energy
    }
}

impl LargeSignalAcousticLoad {
    pub fn validate(self) -> Result<(), ModelingError> {
        require_positive("large_signal.acoustic_load.frequency_hz", self.frequency_hz)?;
        if !self.radiation_impedance_n_s_per_m.re.is_finite()
            || !self.radiation_impedance_n_s_per_m.im.is_finite()
            || self.radiation_impedance_n_s_per_m.re < 0.0
        {
            return Err(ModelingError::InvalidRadiationImpedance);
        }
        Ok(())
    }
}

/// Polynomial large-signal curves and transient sampling controls.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct LargeSignalSpec {
    pub frequency_hz: f64,
    pub voltage_rms_v: f64,
    pub duration_s: f64,
    pub samples_per_period: usize,
    /// Excursion at which the configured quadratic BL drop is reached.
    pub reference_excursion_m: f64,
    /// Fractional BL reduction at `reference_excursion_m`, in `[0, 0.95]`.
    pub bl_drop_fraction: f64,
    /// Linear inductance gradient dLe/dx in H/m.
    pub inductance_gradient_h_per_m: f64,
    /// Cubic suspension coefficient in N/m³; positive values are hardening.
    pub suspension_cubic_n_per_m3: f64,
    /// Optional second tone for acoustic IMD reporting.
    #[serde(default)]
    pub secondary_frequency_hz: Option<f64>,
    /// RMS secondary-tone voltage relative to `voltage_rms_v`.
    #[serde(default = "default_secondary_voltage_ratio")]
    pub secondary_voltage_ratio: f64,
    /// M2 sealed/vented acoustic-circuit controls.
    #[serde(default)]
    pub acoustic: LumpedNonlinearAcousticSpec,
    /// Optional explicit moving-assembly contact, breakup, viscoelastic,
    /// and fatigue laws. Legacy artifacts retain polynomial suspension
    /// behavior when this field is absent.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub moving_assembly: Option<MovingAssemblyNonlinearity>,
}

const fn default_secondary_voltage_ratio() -> f64 {
    0.25
}

impl LargeSignalSpec {
    pub fn validate(self) -> Result<(), ModelingError> {
        require_positive("large_signal.frequency_hz", self.frequency_hz)?;
        require_nonnegative("large_signal.voltage_rms_v", self.voltage_rms_v)?;
        require_positive("large_signal.duration_s", self.duration_s)?;
        let periods = self.duration_s * self.frequency_hz;
        require_finite("large_signal.periods", periods)?;
        if periods < 2.0 {
            return Err(ModelingError::OutOfRange {
                field: "large_signal.duration_s",
                constraint: "at least two excitation periods",
            });
        }
        if self.samples_per_period < 32 {
            return Err(ModelingError::OutOfRange {
                field: "large_signal.samples_per_period",
                constraint: "at least 32",
            });
        }
        require_positive(
            "large_signal.reference_excursion_m",
            self.reference_excursion_m,
        )?;
        require_nonnegative("large_signal.bl_drop_fraction", self.bl_drop_fraction)?;
        if self.bl_drop_fraction > 0.95 {
            return Err(ModelingError::OutOfRange {
                field: "large_signal.bl_drop_fraction",
                constraint: "no greater than 0.95",
            });
        }
        require_finite(
            "large_signal.inductance_gradient_h_per_m",
            self.inductance_gradient_h_per_m,
        )?;
        require_nonnegative(
            "large_signal.suspension_cubic_n_per_m3",
            self.suspension_cubic_n_per_m3,
        )?;
        if let Some(moving_assembly) = self.moving_assembly {
            moving_assembly.validate()?;
        }
        require_nonnegative(
            "large_signal.secondary_voltage_ratio",
            self.secondary_voltage_ratio,
        )?;
        if let Some(frequency) = self.secondary_frequency_hz {
            require_positive("large_signal.secondary_frequency_hz", frequency)?;
            if (frequency - self.frequency_hz).abs()
                <= 1.0e-9 * frequency.max(self.frequency_hz).max(1.0)
            {
                return Err(ModelingError::OutOfRange {
                    field: "large_signal.secondary_frequency_hz",
                    constraint: "different from the primary frequency",
                });
            }
            let nyquist = 0.5 / self.time_step_s()?;
            if frequency >= nyquist {
                return Err(ModelingError::OutOfRange {
                    field: "large_signal.secondary_frequency_hz",
                    constraint: "below Nyquist",
                });
            }
        }
        self.acoustic.validate()?;
        let steps = self.num_steps()?;
        if steps == 0 || steps > MAX_TIME_STEPS {
            return Err(ModelingError::OutOfRange {
                field: "large_signal.time_steps",
                constraint: "between 1 and 1,000,000",
            });
        }
        Ok(())
    }

    pub fn time_step_s(self) -> Result<f64, ModelingError> {
        require_positive("large_signal.frequency_hz", self.frequency_hz)?;
        if self.samples_per_period == 0 {
            return Err(ModelingError::OutOfRange {
                field: "large_signal.samples_per_period",
                constraint: "greater than zero",
            });
        }
        let dt = 1.0 / (self.frequency_hz * self.samples_per_period as f64);
        require_positive("large_signal.time_step_s", dt)?;
        Ok(dt)
    }

    pub fn num_steps(self) -> Result<usize, ModelingError> {
        let dt = self.time_step_s()?;
        let count = (self.duration_s / dt).ceil();
        if !count.is_finite() || count > usize::MAX as f64 {
            return Err(ModelingError::OutOfRange {
                field: "large_signal.time_steps",
                constraint: "representable as usize",
            });
        }
        Ok(count as usize)
    }
}

impl Default for LargeSignalSpec {
    fn default() -> Self {
        Self {
            frequency_hz: 80.0,
            voltage_rms_v: 8.0,
            duration_s: 0.25,
            samples_per_period: 200,
            reference_excursion_m: 5.0e-3,
            bl_drop_fraction: 0.25,
            inductance_gradient_h_per_m: -0.02,
            suspension_cubic_n_per_m3: 2.0e8,
            secondary_frequency_hz: None,
            secondary_voltage_ratio: default_secondary_voltage_ratio(),
            acoustic: LumpedNonlinearAcousticSpec::default(),
            moving_assembly: None,
        }
    }
}

/// Serializable nonlinear transient result with basic distortion diagnostics.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LargeSignalResult {
    pub provenance: LargeSignalProvenance,
    pub time_s: Vec<f64>,
    pub displacement_m: Vec<f64>,
    pub velocity_m_per_s: Vec<f64>,
    pub current_a: Vec<f64>,
    pub force_n: Vec<f64>,
    pub peak_displacement_m: f64,
    #[serde(default)]
    pub peak_contact_penetration_m: f64,
    #[serde(default)]
    pub breakup_exceeded: bool,
    #[serde(default)]
    pub fatigue_damage: f64,
    #[serde(default)]
    pub fatigue_exceeded: bool,
    pub peak_current_a: f64,
    pub displacement_fundamental_peak_m: f64,
    pub displacement_thd: f64,
    #[serde(default)]
    pub box_pressure_pa: Vec<f64>,
    #[serde(default)]
    pub port_flow_m3s: Vec<f64>,
    #[serde(default)]
    pub diaphragm_flow_m3s: Vec<f64>,
    #[serde(default)]
    pub acoustic_pressure_fundamental_peak_pa: f64,
    #[serde(default)]
    pub acoustic_pressure_thd: f64,
    #[serde(default)]
    pub acoustic_pressure_imd: f64,
    #[serde(default)]
    pub port_flow_fundamental_peak_m3s: f64,
    #[serde(default)]
    pub estimated_port_spl_db: Option<f64>,
    #[serde(default)]
    pub acoustic_diagnostics: Option<LumpedNonlinearDiagnostics>,
}

/// Diagnostics for the electrothermal operating point.
///
/// Keeping the acoustic load and power ledger with the result prevents a
/// caller from reporting a hot response without retaining its coupling
/// provenance and closure evidence.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ThermalCompressionDiagnostics {
    pub radiation_impedance_n_s_per_m: Complex64,
    pub input_power_w: f64,
    pub mechanical_loss_w: f64,
    pub radiation_power_w: f64,
    pub power_balance_residual_w: f64,
    pub thermal_iteration_count: usize,
    pub thermal_residual_ohm: f64,
    pub thermal_time_constant_s: f64,
}

/// Steady-state electrothermal compression result using a self-consistent complex load.
#[derive(Debug, Clone, PartialEq)]
pub struct ThermalCompressionResult {
    pub response: SpeakerResponse,
    pub coil_temperature_k: f64,
    pub hot_resistance_ohm: f64,
    pub coil_loss_w: f64,
    /// Positive when heating reduces diaphragm velocity.
    pub compression_db: f64,
    pub diagnostics: ThermalCompressionDiagnostics,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ThermalCompressionSpec {
    pub ambient_temperature_k: f64,
    pub thermal_resistance_k_per_w: f64,
    pub coil_thermal_capacitance_j_per_k: f64,
}

impl ThermalCompressionSpec {
    pub fn validate(self) -> Result<(), ModelingError> {
        require_positive("thermal.ambient_temperature_k", self.ambient_temperature_k)?;
        require_positive(
            "thermal.thermal_resistance_k_per_w",
            self.thermal_resistance_k_per_w,
        )?;
        require_positive(
            "thermal.coil_thermal_capacitance_j_per_k",
            self.coil_thermal_capacitance_j_per_k,
        )
    }
}

impl Default for ThermalCompressionSpec {
    fn default() -> Self {
        Self {
            ambient_temperature_k: 293.15,
            thermal_resistance_k_per_w: 3.0,
            coil_thermal_capacitance_j_per_k: 12.0,
        }
    }
}

impl SpeakerDriverStudy {
    /// Execute the FEM crate's nonlinear lumped Newmark predictor-corrector path.
    pub fn try_large_signal_response(
        &self,
        spec: LargeSignalSpec,
    ) -> Result<LargeSignalResult, ModelingError> {
        self.try_large_signal_response_with_acoustic_load_and_assembly(
            spec,
            None,
            MovingAssemblyNonlinearity::from_large_signal_spec(spec),
        )
    }

    /// Execute the nonlinear transient adapter with a passive exterior load.
    /// The complex load is linearized at the excitation frequency as added
    /// damping `Re(Z)` and added mass `Im(Z)/ω`, providing a bounded feedback
    /// path for cabinet/port/BEM loads while retaining explicit provenance.
    pub fn try_large_signal_response_with_acoustic_load(
        &self,
        spec: LargeSignalSpec,
        acoustic_load: Option<LargeSignalAcousticLoad>,
    ) -> Result<LargeSignalResult, ModelingError> {
        self.try_large_signal_response_with_acoustic_load_and_assembly(
            spec,
            acoustic_load,
            MovingAssemblyNonlinearity::from_large_signal_spec(spec),
        )
    }

    /// Full reduced large-signal entry point with explicit moving-assembly
    /// nonlinearity and a passive exterior load.
    pub fn try_large_signal_response_with_acoustic_load_and_assembly(
        &self,
        spec: LargeSignalSpec,
        acoustic_load: Option<LargeSignalAcousticLoad>,
        assembly: MovingAssemblyNonlinearity,
    ) -> Result<LargeSignalResult, ModelingError> {
        self.validate()?;
        spec.validate()?;
        assembly.validate()?;
        if let Some(load) = acoustic_load {
            load.validate()?;
            if (load.frequency_hz - spec.frequency_hz).abs() > 1.0e-9 * spec.frequency_hz.max(1.0) {
                return Err(ModelingError::OutOfRange {
                    field: "large_signal.acoustic_load.frequency_hz",
                    constraint: "must match the excitation frequency",
                });
            }
        }
        let dt = spec.time_step_s()?;
        let num_steps = spec.num_steps()?;
        let bl0 = self.voice_coil.bl_tm;
        let le0 = self.voice_coil.inductance_h;
        let x_ref = spec.reference_excursion_m;
        let bl_drop = spec.bl_drop_fraction;
        let le_gradient = spec.inductance_gradient_h_per_m;
        let min_le = (le0 * 0.05).max(f64::MIN_POSITIVE);
        let voice_coil = VoiceCoilParams::nonlinear(
            self.voice_coil.resistance_ohm,
            move |x| bl0 * (1.0 - bl_drop * (x / x_ref).powi(2)).max(0.05),
            move |x| (le0 + le_gradient * x).max(min_le),
            move |x| {
                if le0 + le_gradient * x > min_le {
                    le_gradient
                } else {
                    0.0
                }
            },
        );
        let (acoustic_rms, acoustic_mass) = acoustic_load.map_or((0.0, 0.0), |load| {
            (
                load.radiation_impedance_n_s_per_m.re,
                load.radiation_impedance_n_s_per_m.im / (std::f64::consts::TAU * spec.frequency_hz),
            )
        });
        if !(self.moving_system.moving_mass_kg + acoustic_mass).is_finite()
            || self.moving_system.moving_mass_kg + acoustic_mass <= 0.0
        {
            return Err(ModelingError::OutOfRange {
                field: "large_signal.acoustic_load",
                constraint: "must preserve positive effective moving mass",
            });
        }
        let omega = std::f64::consts::TAU * spec.frequency_hz;
        let voltage_peak = std::f64::consts::SQRT_2 * spec.voltage_rms_v;
        let secondary_omega = spec
            .secondary_frequency_hz
            .map(|frequency| std::f64::consts::TAU * frequency);
        let secondary_voltage_peak = voltage_peak * spec.secondary_voltage_ratio;
        let config = LumpedNonlinearRunConfig {
            mms_kg: self.moving_system.moving_mass_kg,
            rms_kg_per_s: self.moving_system.mechanical_resistance_kg_per_s,
            kms_n_per_m: self.moving_system.stiffness_n_per_m,
            sd_m2: std::f64::consts::PI * self.geometry.diaphragm_radius_m.powi(2),
            voice_coil,
            assembly,
            enclosure: self.enclosure.clone(),
            medium: self.acoustic_medium,
            acoustic: spec.acoustic,
            harmonic_front_damping_n_s_per_m: acoustic_rms,
            harmonic_front_mass_kg: acoustic_mass,
            voltage_fn: Box::new(move |time| {
                voltage_peak * (omega * time).sin()
                    + secondary_omega
                        .map(|secondary| secondary_voltage_peak * (secondary * time).sin())
                        .unwrap_or(0.0)
            }),
            dt,
            num_steps,
        };
        let result = run_lumped_nonlinear(config, spec.frequency_hz)?;
        let peak_displacement_m = max_abs(&result.displacement_m);
        let peak_contact_penetration_m = result
            .displacement_m
            .iter()
            .map(|&displacement_m| assembly.contact_penetration_m(displacement_m))
            .fold(0.0, f64::max);
        let breakup_exceeded = result
            .displacement_m
            .iter()
            .any(|&displacement_m| assembly.breakup_active(displacement_m));
        let fatigue_damage =
            assembly.fatigue_damage(peak_displacement_m, spec.duration_s * spec.frequency_hz);
        let fatigue_exceeded = fatigue_damage >= 1.0;
        let peak_current_a = max_abs(&result.current_a);
        let periods_available = (spec.duration_s * spec.frequency_hz).floor() as usize;
        let analysis_periods = periods_available.clamp(1, 8);
        let analysis_samples =
            (analysis_periods * spec.samples_per_period).min(result.displacement_m.len());
        let signal = &result.displacement_m[result.displacement_m.len() - analysis_samples..];
        let harmonics = harmonic_peaks(signal, dt, spec.frequency_hz, 5);
        let fundamental = harmonics[0];
        let thd = if fundamental > 0.0 {
            harmonics[1..]
                .iter()
                .map(|amplitude| amplitude * amplitude)
                .sum::<f64>()
                .sqrt()
                / fundamental
        } else {
            0.0
        };
        require_finite("large_signal.displacement_thd", thd)?;
        let pressure_tail =
            &result.box_pressure_pa[result.box_pressure_pa.len() - analysis_samples..];
        let flow_tail = &result.port_flow_m3s[result.port_flow_m3s.len() - analysis_samples..];
        let acoustic_pressure_fundamental_peak_pa =
            dft_amplitude(pressure_tail, dt, spec.frequency_hz);
        let acoustic_pressure_thd = thd_fraction(pressure_tail, dt, spec.frequency_hz, 5);
        let acoustic_pressure_imd = spec.secondary_frequency_hz.map_or(0.0, |secondary| {
            imd_fraction(
                pressure_tail,
                dt,
                spec.frequency_hz.min(secondary),
                spec.frequency_hz.max(secondary),
            )
        });
        let port_flow_fundamental_peak_m3s = dft_amplitude(flow_tail, dt, spec.frequency_hz);
        let estimated_port_spl_db = (port_flow_fundamental_peak_m3s > 0.0).then(|| {
            port_spl_db(
                port_flow_fundamental_peak_m3s,
                spec.frequency_hz,
                self.acoustic_medium.density_kg_per_m3,
                1.0,
            )
        });
        let sd_m2 = std::f64::consts::PI * self.geometry.diaphragm_radius_m.powi(2);
        let diaphragm_flow_m3s = result
            .velocity_m_per_s
            .iter()
            .map(|velocity| sd_m2 * velocity)
            .collect();
        Ok(LargeSignalResult {
            provenance: if self.enclosure != super::EnclosureSpec::FreeAir {
                LargeSignalProvenance::LumpedNonlinearElectromechanicalWithCoupledAcousticCircuit
            } else if acoustic_load.is_some() {
                LargeSignalProvenance::LumpedNonlinearElectromechanicalWithHarmonicAcousticLoad
            } else {
                LargeSignalProvenance::LumpedNonlinearElectromechanicalWithoutAcousticLoad
            },
            time_s: result.time_s,
            displacement_m: result.displacement_m,
            velocity_m_per_s: result.velocity_m_per_s,
            current_a: result.current_a,
            force_n: result.force_n,
            peak_displacement_m,
            peak_contact_penetration_m,
            breakup_exceeded,
            fatigue_damage,
            fatigue_exceeded,
            peak_current_a,
            displacement_fundamental_peak_m: fundamental,
            displacement_thd: thd,
            box_pressure_pa: result.box_pressure_pa,
            port_flow_m3s: result.port_flow_m3s,
            diaphragm_flow_m3s,
            acoustic_pressure_fundamental_peak_pa,
            acoustic_pressure_thd,
            acoustic_pressure_imd,
            port_flow_fundamental_peak_m3s,
            estimated_port_spl_db,
            acoustic_diagnostics: (self.enclosure != super::EnclosureSpec::FreeAir)
                .then_some(result.diagnostics),
        })
    }

    /// Solve the hot steady-state response with copper resistance iterated to thermal balance.
    pub fn try_thermal_compressed_response_at(
        &self,
        frequency_hz: f64,
        voltage_rms_v: f64,
        thermal_spec: ThermalCompressionSpec,
    ) -> Result<ThermalCompressionResult, ModelingError> {
        let radiation_impedance = self.mechanical_radiation_impedance(frequency_hz)?;
        self.try_thermal_compressed_response_with_radiation_load_at(
            frequency_hz,
            voltage_rms_v,
            thermal_spec,
            radiation_impedance,
        )
    }

    /// Solve the hot response with a caller-supplied front acoustic load.
    ///
    /// The supplied impedance replaces the analytical baffled-piston load;
    /// the configured rear enclosure remains part of the mechanical system.
    /// This is the typed bridge for resolved BEM/FEM product simulations: the
    /// same passive loaded state is used for cold-reference velocity, copper
    /// heating, hot resistance, and the returned power ledger.
    pub fn try_thermal_compressed_response_with_radiation_load_at(
        &self,
        frequency_hz: f64,
        voltage_rms_v: f64,
        thermal_spec: ThermalCompressionSpec,
        radiation_impedance: Complex64,
    ) -> Result<ThermalCompressionResult, ModelingError> {
        self.validate()?;
        require_positive("frequency_hz", frequency_hz)?;
        require_nonnegative("voltage_rms_v", voltage_rms_v)?;
        thermal_spec.validate()?;
        if !radiation_impedance.re.is_finite()
            || !radiation_impedance.im.is_finite()
            || radiation_impedance.re < -1.0e-10 * radiation_impedance.norm().max(f64::MIN_POSITIVE)
        {
            return Err(ModelingError::InvalidRadiationImpedance);
        }
        let voltage = Complex64::new(voltage_rms_v, 0.0);
        let omega = std::f64::consts::TAU * frequency_hz;
        let mechanical_impedance = self.moving_system.try_mechanical_impedance(frequency_hz)?
            + self.enclosure_mechanical_impedance(frequency_hz)?
            + radiation_impedance;
        let response_for_resistance = |resistance: f64| {
            let coupling =
                sonium_fem::physics::acoustics::electroacoustics::transducer::ElectrodynamicCoupling {
                    resistance,
                    inductance: self.voice_coil.inductance_h,
                    bl: self.voice_coil.bl_tm,
                    coil_dofs: vec![(0, 1.0)],
                };
            let impedance = coupling.total_impedance(omega, mechanical_impedance);
            let (current, velocity, displacement) =
                coupling.solve_frequency(voltage, omega, mechanical_impedance);
            SpeakerResponse {
                frequency_hz,
                voltage,
                current,
                impedance,
                bl_force: Complex64::new(self.voice_coil.bl_tm, 0.0) * current,
                coil_velocity: velocity,
                diaphragm_displacement: displacement,
            }
        };
        let cold = response_for_resistance(self.voice_coil.resistance_ohm);
        let thermal = VoiceCoilThermalModel::simple(
            self.voice_coil.resistance_ohm,
            thermal_spec.ambient_temperature_k,
            thermal_spec.thermal_resistance_k_per_w,
            thermal_spec.coil_thermal_capacitance_j_per_k,
        );
        let mut resistance = self.voice_coil.resistance_ohm;
        let mut converged = false;
        let mut thermal_iteration_count = 0;
        let mut thermal_residual_ohm = f64::INFINITY;
        for iteration in 1..=100 {
            let response = response_for_resistance(resistance);
            let current = response.current;
            let coil_loss = resistance * current.norm_sqr();
            let temperature = thermal_spec.ambient_temperature_k
                + coil_loss * thermal_spec.thermal_resistance_k_per_w;
            let next_resistance = thermal.resistance_at(temperature);
            for value in [
                response.impedance.re,
                response.impedance.im,
                next_resistance,
                temperature,
                coil_loss,
            ] {
                require_finite("thermal.response", value)?;
            }
            thermal_residual_ohm = (next_resistance - resistance).abs();
            thermal_iteration_count = iteration;
            if thermal_residual_ohm <= 1.0e-10 * self.voice_coil.resistance_ohm.max(1.0) {
                resistance = next_resistance;
                converged = true;
                break;
            }
            resistance = 0.5 * (resistance + next_resistance);
        }
        if !converged {
            return Err(ModelingError::ThermalIterationDidNotConverge);
        }
        let final_response = response_for_resistance(resistance);
        let coil_loss = resistance * final_response.current.norm_sqr();
        let coil_temperature_k = thermal_spec.ambient_temperature_k
            + coil_loss * thermal_spec.thermal_resistance_k_per_w;
        let compression_db = if final_response.coil_velocity.norm() > 0.0 {
            20.0 * (cold.coil_velocity.norm() / final_response.coil_velocity.norm()).log10()
        } else {
            0.0
        };
        require_finite("thermal.compression_db", compression_db)?;
        let input_power = (voltage * final_response.current.conj()).re;
        let mechanical_loss = (mechanical_impedance.re - radiation_impedance.re)
            * final_response.coil_velocity.norm_sqr();
        let radiation_power = radiation_impedance.re * final_response.coil_velocity.norm_sqr();
        let power_balance_residual = input_power - coil_loss - mechanical_loss - radiation_power;
        let power_scale = input_power
            .abs()
            .max(coil_loss.abs())
            .max(mechanical_loss.abs())
            .max(radiation_power.abs())
            .max(1.0e-12);
        if !power_balance_residual.is_finite()
            || power_balance_residual.abs() > 1.0e-8 * power_scale
        {
            return Err(ModelingError::ThermalPowerBalanceDidNotClose);
        }
        Ok(ThermalCompressionResult {
            response: final_response,
            coil_temperature_k,
            hot_resistance_ohm: resistance,
            coil_loss_w: coil_loss,
            compression_db,
            diagnostics: ThermalCompressionDiagnostics {
                radiation_impedance_n_s_per_m: radiation_impedance,
                input_power_w: input_power,
                mechanical_loss_w: mechanical_loss,
                radiation_power_w: radiation_power,
                power_balance_residual_w: power_balance_residual,
                thermal_iteration_count,
                thermal_residual_ohm,
                thermal_time_constant_s: thermal_spec.coil_thermal_capacitance_j_per_k
                    * thermal_spec.thermal_resistance_k_per_w,
            },
        })
    }
}

fn max_abs(values: &[f64]) -> f64 {
    values.iter().map(|value| value.abs()).fold(0.0, f64::max)
}

fn harmonic_peaks(signal: &[f64], dt: f64, fundamental_hz: f64, count: usize) -> Vec<f64> {
    let scale = 2.0 / signal.len() as f64;
    (1..=count)
        .map(|harmonic| {
            let omega = std::f64::consts::TAU * fundamental_hz * harmonic as f64;
            let (real, imag) =
                signal
                    .iter()
                    .enumerate()
                    .fold((0.0, 0.0), |(real, imag), (index, &sample)| {
                        let phase = omega * index as f64 * dt;
                        (real + sample * phase.cos(), imag - sample * phase.sin())
                    });
            scale * real.hypot(imag)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::SpeakerDriverStudy;

    #[test]
    fn passive_acoustic_load_is_consumed_by_nonlinear_transient_adapter() {
        let study = SpeakerDriverStudy::transient_large_signal_driver();
        let spec = LargeSignalSpec {
            duration_s: 0.06,
            samples_per_period: 64,
            ..LargeSignalSpec::default()
        };
        let result = study
            .try_large_signal_response_with_acoustic_load(
                spec,
                Some(LargeSignalAcousticLoad {
                    frequency_hz: spec.frequency_hz,
                    radiation_impedance_n_s_per_m: Complex64::new(0.8, 0.04),
                }),
            )
            .unwrap();
        assert_eq!(
            result.provenance,
            LargeSignalProvenance::LumpedNonlinearElectromechanicalWithHarmonicAcousticLoad
        );
        assert!(result.displacement_m.iter().all(|value| value.is_finite()));
    }

    #[test]
    fn active_or_mismatched_acoustic_load_is_rejected() {
        let study = SpeakerDriverStudy::transient_large_signal_driver();
        let spec = LargeSignalSpec::default();
        assert!(
            study
                .try_large_signal_response_with_acoustic_load(
                    spec,
                    Some(LargeSignalAcousticLoad {
                        frequency_hz: spec.frequency_hz,
                        radiation_impedance_n_s_per_m: Complex64::new(-0.1, 0.0),
                    }),
                )
                .is_err()
        );
        assert!(
            study
                .try_large_signal_response_with_acoustic_load(
                    spec,
                    Some(LargeSignalAcousticLoad {
                        frequency_hz: spec.frequency_hz * 2.0,
                        radiation_impedance_n_s_per_m: Complex64::new(0.1, 0.0),
                    }),
                )
                .is_err()
        );
    }

    #[test]
    fn moving_assembly_model_carries_asymmetry_and_hard_stop() {
        let assembly = MovingAssemblyNonlinearity {
            suspension_quadratic_n_per_m2: 1.0e6,
            suspension_cubic_n_per_m3: 2.0e8,
            hard_stop_excursion_m: Some(1.0e-3),
            hard_stop_stiffness_n_per_m: 1.0e5,
            viscoelastic: None,
            contact: None,
            breakup: None,
            fatigue: None,
        };
        assembly.validate().unwrap();
        assert_ne!(
            assembly.nonlinear_force(2.0e-3),
            -assembly.nonlinear_force(-2.0e-3)
        );
        assert!(assembly.nonlinear_force(2.0e-3).abs() > assembly.nonlinear_force(0.5e-3).abs());
    }

    #[test]
    fn moving_assembly_extended_laws_validate_and_round_trip() {
        let assembly = MovingAssemblyNonlinearity {
            suspension_quadratic_n_per_m2: 0.0,
            suspension_cubic_n_per_m3: 1.0e8,
            hard_stop_excursion_m: None,
            hard_stop_stiffness_n_per_m: 0.0,
            viscoelastic: Some(MovingAssemblyViscoelasticSpec {
                damping_n_s_per_m2: 2.0e3,
            }),
            contact: Some(MovingAssemblyContactSpec {
                clearance_m: 1.0e-3,
                stiffness_n_per_m: 2.0e4,
                damping_n_s_per_m: 3.0,
            }),
            breakup: Some(MovingAssemblyBreakupSpec {
                onset_excursion_m: 0.8e-3,
                softening_stiffness_n_per_m: 100.0,
            }),
            fatigue: Some(MovingAssemblyFatigueSpec {
                reference_excursion_m: 1.0e-3,
                cycles_to_failure: 1.0e6,
                damage_exponent: 3.0,
            }),
        };
        assembly.validate().unwrap();
        assert!(assembly.nonlinear_damping_force(2.0e-3, 0.5) > 0.0);
        assert!(assembly.contact_penetration_m(2.0e-3) > 0.0);
        assert!(assembly.breakup_active(2.0e-3));
        assert!(assembly.fatigue_damage(2.0e-3, 100.0) > 0.0);
        assert!(assembly.potential_energy(2.0e-3).is_finite());

        let spec = LargeSignalSpec {
            moving_assembly: Some(assembly),
            ..LargeSignalSpec::default()
        };
        spec.validate().unwrap();
        let decoded: LargeSignalSpec =
            serde_json::from_slice(&serde_json::to_vec(&spec).unwrap()).unwrap();
        assert_eq!(decoded, spec);

        let mut legacy = serde_json::to_value(spec).unwrap();
        legacy.as_object_mut().unwrap().remove("moving_assembly");
        let decoded_legacy: LargeSignalSpec = serde_json::from_value(legacy).unwrap();
        assert!(decoded_legacy.moving_assembly.is_none());
    }

    #[test]
    fn moving_assembly_extended_laws_reject_invalid_parameters() {
        let mut assembly = MovingAssemblyNonlinearity {
            suspension_quadratic_n_per_m2: 0.0,
            suspension_cubic_n_per_m3: 0.0,
            hard_stop_excursion_m: None,
            hard_stop_stiffness_n_per_m: 0.0,
            viscoelastic: Some(MovingAssemblyViscoelasticSpec {
                damping_n_s_per_m2: -1.0,
            }),
            contact: None,
            breakup: None,
            fatigue: None,
        };
        assert!(assembly.validate().is_err());

        assembly.viscoelastic = None;
        assembly.contact = Some(MovingAssemblyContactSpec {
            clearance_m: 1.0e-3,
            stiffness_n_per_m: 0.0,
            damping_n_s_per_m: 0.0,
        });
        assert!(assembly.validate().is_err());

        assembly.contact = None;
        assembly.fatigue = Some(MovingAssemblyFatigueSpec {
            reference_excursion_m: 1.0e-3,
            cycles_to_failure: 0.0,
            damage_exponent: 3.0,
        });
        assert!(assembly.validate().is_err());
    }
}
