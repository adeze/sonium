//! Bounded S-N/Miner fatigue screening reference.
//!
//! This module admits a caller-supplied stress-cycle spectrum through a
//! Goodman mean-stress correction and a single-slope Basquin S-N curve. It is
//! a transparent fatigue evidence contract, not a load-generation,
//! aeroelastic, crack-growth, or manufactured-section fatigue solver.

use std::fmt;

/// One constant-amplitude stress block in a fatigue spectrum.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FatigueCycle {
    /// Mean normal stress [Pa]. Compressive mean stress may be negative.
    pub mean_stress_pa: f64,
    /// Alternating normal-stress amplitude [Pa].
    pub alternating_stress_pa: f64,
    /// Number of repetitions in the supplied spectrum block [-].
    pub cycles: f64,
}

/// Options for the bounded Goodman-corrected S-N screen.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FatigueOptions {
    /// Ultimate tensile strength used by the Goodman correction [Pa].
    pub ultimate_strength_pa: f64,
    /// Stress at the reference S-N life [Pa].
    pub reference_stress_pa: f64,
    /// Reference life for `reference_stress_pa` [cycles].
    pub reference_cycles: f64,
    /// Positive Basquin slope in `S = S_ref * (N / N_ref)^(-slope)` [-].
    pub sn_slope: f64,
    /// Alternating-stress endurance limit before the design safety factor [Pa].
    pub endurance_limit_pa: f64,
    /// Multiplicative stress safety factor applied before the S-N lookup [-].
    pub safety_factor: f64,
}

impl Default for FatigueOptions {
    fn default() -> Self {
        Self {
            ultimate_strength_pa: 300.0e6,
            reference_stress_pa: 100.0e6,
            reference_cycles: 1.0e6,
            sn_slope: 0.12,
            endurance_limit_pa: 30.0e6,
            safety_factor: 1.5,
        }
    }
}

/// Evaluated damage contribution from one supplied stress block.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FatigueCycleResult {
    /// Original stress block.
    pub cycle: FatigueCycle,
    /// Goodman-corrected alternating stress before the safety factor [Pa].
    pub corrected_alternating_stress_pa: f64,
    /// Allowable repetitions, or `None` when below the endurance limit.
    pub allowable_cycles: Option<f64>,
    /// Miner damage contributed by this block [-].
    pub damage_fraction: f64,
}

/// Result of the bounded stress-cycle fatigue screen.
#[derive(Debug, Clone, PartialEq)]
pub struct FatigueResult {
    /// Per-block corrected stress and damage evidence.
    pub cycles: Vec<FatigueCycleResult>,
    /// Sum of Miner damage for the supplied spectrum block [-].
    pub total_damage_fraction: f64,
    /// Total repetitions in the supplied spectrum block [cycles].
    pub total_applied_cycles: f64,
    /// Number of repeated spectrum blocks to reach unit damage, or `None` for
    /// an endurance-limited spectrum with zero computed damage.
    pub predicted_life_blocks: Option<f64>,
}

impl FatigueResult {
    /// Whether all computed values satisfy the finite result contract.
    pub fn is_finite(&self) -> bool {
        self.total_damage_fraction.is_finite()
            && self.total_damage_fraction >= 0.0
            && self.total_applied_cycles.is_finite()
            && self.total_applied_cycles > 0.0
            && self
                .predicted_life_blocks
                .is_none_or(|value| value.is_finite() && value > 0.0)
            && self.cycles.iter().all(|cycle| {
                cycle.cycle.mean_stress_pa.is_finite()
                    && cycle.cycle.alternating_stress_pa.is_finite()
                    && cycle.cycle.alternating_stress_pa >= 0.0
                    && cycle.cycle.cycles.is_finite()
                    && cycle.cycle.cycles > 0.0
                    && cycle.corrected_alternating_stress_pa.is_finite()
                    && cycle.corrected_alternating_stress_pa >= 0.0
                    && cycle
                        .allowable_cycles
                        .is_none_or(|value| value.is_finite() && value > 0.0)
                    && cycle.damage_fraction.is_finite()
                    && cycle.damage_fraction >= 0.0
            })
    }

    /// Whether the supplied spectrum remains below unit Miner damage.
    pub fn is_feasible(&self) -> bool {
        self.is_finite() && self.total_damage_fraction <= 1.0
    }
}

/// Failure-closed errors for the fatigue reference.
#[derive(Debug, Clone, PartialEq)]
pub enum FatigueError {
    /// A model option is outside the declared domain.
    InvalidOptions {
        /// Option field name.
        field: &'static str,
        /// Required domain.
        requirement: &'static str,
    },
    /// No stress blocks were supplied.
    EmptySpectrum,
    /// A supplied block is malformed.
    InvalidCycle {
        /// Zero-based block index.
        index: usize,
        /// Invalid field name.
        field: &'static str,
        /// Required domain.
        requirement: &'static str,
    },
    /// Mean stress reaches or exceeds the Goodman ultimate-strength limit.
    GoodmanSingularity {
        /// Zero-based block index.
        index: usize,
    },
    /// A derived quantity is non-finite.
    NonFiniteResult {
        /// Derived field name.
        field: &'static str,
    },
}

impl fmt::Display for FatigueError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidOptions { field, requirement } => {
                write!(formatter, "invalid fatigue option {field}: {requirement}")
            }
            Self::EmptySpectrum => write!(formatter, "fatigue spectrum must not be empty"),
            Self::InvalidCycle {
                index,
                field,
                requirement,
            } => write!(
                formatter,
                "invalid fatigue cycle {index} field {field}: {requirement}"
            ),
            Self::GoodmanSingularity { index } => write!(
                formatter,
                "fatigue cycle {index} mean stress reaches ultimate strength"
            ),
            Self::NonFiniteResult { field } => {
                write!(formatter, "non-finite fatigue result field {field}")
            }
        }
    }
}

impl std::error::Error for FatigueError {}

/// Evaluate a caller-supplied spectrum with Goodman correction and Miner sum.
pub fn evaluate_s_n_fatigue_reference(
    spectrum: &[FatigueCycle],
    options: FatigueOptions,
) -> Result<FatigueResult, FatigueError> {
    validate_options(options)?;
    if spectrum.is_empty() {
        return Err(FatigueError::EmptySpectrum);
    }

    let mut results = Vec::with_capacity(spectrum.len());
    let mut total_damage_fraction = 0.0;
    let mut total_applied_cycles = 0.0;

    for (index, &cycle) in spectrum.iter().enumerate() {
        validate_cycle(index, cycle)?;
        let denominator = 1.0 - cycle.mean_stress_pa / options.ultimate_strength_pa;
        if !denominator.is_finite() || denominator <= 0.0 {
            return Err(FatigueError::GoodmanSingularity { index });
        }

        let corrected_alternating_stress_pa = cycle.alternating_stress_pa / denominator;
        let design_stress_pa = corrected_alternating_stress_pa * options.safety_factor;
        if !corrected_alternating_stress_pa.is_finite() || !design_stress_pa.is_finite() {
            return Err(FatigueError::NonFiniteResult {
                field: "corrected_alternating_stress_pa",
            });
        }

        let (allowable_cycles, damage_fraction) = if design_stress_pa <= options.endurance_limit_pa
        {
            (None, 0.0)
        } else {
            let life = options.reference_cycles
                * (options.reference_stress_pa / design_stress_pa).powf(1.0 / options.sn_slope);
            if !life.is_finite() || life <= 0.0 {
                return Err(FatigueError::NonFiniteResult {
                    field: "allowable_cycles",
                });
            }
            let damage = cycle.cycles / life;
            if !damage.is_finite() || damage < 0.0 {
                return Err(FatigueError::NonFiniteResult {
                    field: "damage_fraction",
                });
            }
            (Some(life), damage)
        };

        total_damage_fraction += damage_fraction;
        total_applied_cycles += cycle.cycles;
        if !total_damage_fraction.is_finite() || !total_applied_cycles.is_finite() {
            return Err(FatigueError::NonFiniteResult {
                field: "spectrum_totals",
            });
        }
        results.push(FatigueCycleResult {
            cycle,
            corrected_alternating_stress_pa,
            allowable_cycles,
            damage_fraction,
        });
    }

    let predicted_life_blocks = if total_damage_fraction > 0.0 {
        let life_blocks = 1.0 / total_damage_fraction;
        if !life_blocks.is_finite() || life_blocks <= 0.0 {
            return Err(FatigueError::NonFiniteResult {
                field: "predicted_life_blocks",
            });
        }
        Some(life_blocks)
    } else {
        None
    };

    let result = FatigueResult {
        cycles: results,
        total_damage_fraction,
        total_applied_cycles,
        predicted_life_blocks,
    };
    if result.is_finite() {
        Ok(result)
    } else {
        Err(FatigueError::NonFiniteResult {
            field: "fatigue_result",
        })
    }
}

fn validate_options(options: FatigueOptions) -> Result<(), FatigueError> {
    for (field, value, requirement) in [
        (
            "ultimate_strength_pa",
            options.ultimate_strength_pa,
            "finite and positive",
        ),
        (
            "reference_stress_pa",
            options.reference_stress_pa,
            "finite and positive",
        ),
        (
            "reference_cycles",
            options.reference_cycles,
            "finite and positive",
        ),
        ("sn_slope", options.sn_slope, "finite and positive"),
        (
            "endurance_limit_pa",
            options.endurance_limit_pa,
            "finite and non-negative",
        ),
        (
            "safety_factor",
            options.safety_factor,
            "finite and positive",
        ),
    ] {
        if !value.is_finite()
            || (field == "endurance_limit_pa" && value < 0.0)
            || (field != "endurance_limit_pa" && value <= 0.0)
        {
            return Err(FatigueError::InvalidOptions { field, requirement });
        }
    }
    Ok(())
}

fn validate_cycle(index: usize, cycle: FatigueCycle) -> Result<(), FatigueError> {
    if !cycle.mean_stress_pa.is_finite() {
        return Err(FatigueError::InvalidCycle {
            index,
            field: "mean_stress_pa",
            requirement: "finite",
        });
    }
    if !cycle.alternating_stress_pa.is_finite() || cycle.alternating_stress_pa < 0.0 {
        return Err(FatigueError::InvalidCycle {
            index,
            field: "alternating_stress_pa",
            requirement: "finite and non-negative",
        });
    }
    if !cycle.cycles.is_finite() || cycle.cycles <= 0.0 {
        return Err(FatigueError::InvalidCycle {
            index,
            field: "cycles",
            requirement: "finite and positive",
        });
    }
    Ok(())
}
