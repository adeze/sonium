//! Bounded nonlinear single-mode aeroelastic response reference.
//!
//! The model solves the first-harmonic amplitude equation for a damped Duffing
//! mode,
//!
//! `m q̈ + c q̇ + k q + α q³ = F cos(ω t)`,
//!
//! using a deterministic amplitude bracket scan followed by bisection. It is
//! intended to screen nonlinear stiffness sensitivity and stress/deflection
//! admission around a caller-declared mode. It is not a transient,
//! centrifugal-stiffened, multi-mode, flutter, contact, or measured fatigue
//! solver.

use std::fmt;

/// Caller-declared single-mode Duffing response and admission settings.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NonlinearAeroelasticModalOptions {
    /// Effective modal mass [kg].
    pub effective_mass_kg: f64,
    /// Undamped linear natural frequency [Hz].
    pub natural_frequency_hz: f64,
    /// Modal viscous damping ratio [-].
    pub damping_ratio: f64,
    /// Cubic stiffness coefficient α [N/m³]. Negative values model softening.
    pub cubic_stiffness_n_m3: f64,
    /// First-harmonic modal forcing amplitude [N].
    pub forcing_amplitude_n: f64,
    /// Harmonic forcing frequency [Hz].
    pub forcing_frequency_hz: f64,
    /// Caller-declared modal stress transfer coefficient [Pa/m].
    pub modal_stress_per_m_pa: f64,
    /// Maximum admitted modal amplitude [m].
    pub maximum_amplitude_m: f64,
    /// Maximum admitted alternating stress [Pa].
    pub maximum_stress_pa: f64,
    /// Number of deterministic amplitude intervals used to find the first root.
    pub bracket_samples: usize,
    /// Maximum bisection iterations after a sign-changing bracket is found.
    pub max_iterations: usize,
    /// Force-amplitude residual tolerance [N].
    pub residual_tolerance_n: f64,
}

impl Default for NonlinearAeroelasticModalOptions {
    fn default() -> Self {
        Self {
            effective_mass_kg: 0.2,
            natural_frequency_hz: 100.0,
            damping_ratio: 0.02,
            cubic_stiffness_n_m3: 0.0,
            forcing_amplitude_n: 1.0,
            forcing_frequency_hz: 100.0,
            modal_stress_per_m_pa: 1_000.0,
            maximum_amplitude_m: 0.1,
            maximum_stress_pa: 500.0,
            bracket_samples: 64,
            max_iterations: 100,
            residual_tolerance_n: 1.0e-10,
        }
    }
}

/// Result of the checked nonlinear single-mode response reference.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NonlinearAeroelasticModalResult {
    /// Linear modal stiffness `m (2π f₀)²` [N/m].
    pub linear_stiffness_n_m: f64,
    /// Viscous modal damping coefficient [N s/m].
    pub damping_n_s_m: f64,
    /// Forcing angular frequency [rad/s].
    pub forcing_angular_frequency_rad_s: f64,
    /// Selected first positive amplitude bracket [m].
    pub selected_bracket_m: [f64; 2],
    /// Nonlinear first-harmonic displacement amplitude [m].
    pub amplitude_m: f64,
    /// Displacement phase relative to the forcing [rad].
    pub phase_rad: f64,
    /// Absolute force-amplitude residual at the selected root [N].
    pub nonlinear_residual_n: f64,
    /// Linear-response amplitude using α = 0 [m].
    pub linear_amplitude_m: f64,
    /// Nonlinear-to-linear amplitude ratio [-].
    pub nonlinear_to_linear_amplitude_ratio: f64,
    /// Estimated alternating stress [Pa].
    pub alternating_stress_pa: f64,
    /// Bounded amplitude and stress admission result.
    pub feasible: bool,
    /// Bisection iterations used after bracketing.
    pub iterations: usize,
}

impl NonlinearAeroelasticModalResult {
    /// Returns whether all result values are finite and internally admissible.
    pub fn is_finite(&self) -> bool {
        self.linear_stiffness_n_m.is_finite()
            && self.linear_stiffness_n_m > 0.0
            && self.damping_n_s_m.is_finite()
            && self.damping_n_s_m > 0.0
            && self.forcing_angular_frequency_rad_s.is_finite()
            && self.forcing_angular_frequency_rad_s >= 0.0
            && self
                .selected_bracket_m
                .iter()
                .all(|value| value.is_finite())
            && self.selected_bracket_m[0] >= 0.0
            && self.selected_bracket_m[1] >= self.selected_bracket_m[0]
            && self.amplitude_m.is_finite()
            && self.amplitude_m >= 0.0
            && self.phase_rad.is_finite()
            && self.nonlinear_residual_n.is_finite()
            && self.nonlinear_residual_n >= 0.0
            && self.linear_amplitude_m.is_finite()
            && self.linear_amplitude_m >= 0.0
            && self.nonlinear_to_linear_amplitude_ratio.is_finite()
            && self.nonlinear_to_linear_amplitude_ratio >= 0.0
            && self.alternating_stress_pa.is_finite()
            && self.alternating_stress_pa >= 0.0
            && self.iterations > 0
    }

    /// Whether the computed response satisfies both caller-declared gates.
    pub fn is_feasible(&self, options: &NonlinearAeroelasticModalOptions) -> bool {
        self.is_finite()
            && self.amplitude_m <= options.maximum_amplitude_m
            && self.alternating_stress_pa <= options.maximum_stress_pa
    }
}

/// Failure-closed errors for the nonlinear modal reference.
#[derive(Debug, Clone, PartialEq)]
pub enum NonlinearAeroelasticModalError {
    /// An option is outside its admitted domain.
    InvalidOptions {
        /// Option field name.
        field: &'static str,
        /// Required domain.
        requirement: &'static str,
    },
    /// No finite positive response root exists within the amplitude bound.
    NoRootWithinAmplitude {
        /// Maximum scanned amplitude [m].
        maximum_amplitude_m: f64,
        /// Force residual at the maximum amplitude [N].
        residual_at_limit_n: f64,
    },
    /// Bisection did not meet the force residual tolerance.
    NonConverged {
        /// Iterations performed.
        iterations: usize,
        /// Final residual [N].
        residual_n: f64,
        /// Requested residual tolerance [N].
        tolerance_n: f64,
    },
}

impl fmt::Display for NonlinearAeroelasticModalError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidOptions { field, requirement } => write!(
                formatter,
                "invalid nonlinear modal option {field}: {requirement}"
            ),
            Self::NoRootWithinAmplitude {
                maximum_amplitude_m,
                residual_at_limit_n,
            } => write!(
                formatter,
                "no nonlinear response root through {maximum_amplitude_m} m; residual at limit {residual_at_limit_n} N"
            ),
            Self::NonConverged {
                iterations,
                residual_n,
                tolerance_n,
            } => write!(
                formatter,
                "nonlinear modal response did not converge in {iterations} iterations: residual {residual_n} N > {tolerance_n} N"
            ),
        }
    }
}

impl std::error::Error for NonlinearAeroelasticModalError {}

/// Evaluate the bounded first-harmonic nonlinear modal response.
pub fn evaluate_nonlinear_aeroelastic_modal(
    options: NonlinearAeroelasticModalOptions,
) -> Result<NonlinearAeroelasticModalResult, NonlinearAeroelasticModalError> {
    validate_options(&options)?;
    let natural_angular_frequency_rad_s = 2.0 * std::f64::consts::PI * options.natural_frequency_hz;
    let forcing_angular_frequency_rad_s = 2.0 * std::f64::consts::PI * options.forcing_frequency_hz;
    let linear_stiffness_n_m = options.effective_mass_kg * natural_angular_frequency_rad_s.powi(2);
    let damping_n_s_m =
        2.0 * options.damping_ratio * options.effective_mass_kg * natural_angular_frequency_rad_s;
    let linear_dynamic_stiffness_n_m =
        linear_stiffness_n_m - options.effective_mass_kg * forcing_angular_frequency_rad_s.powi(2);
    let damping_dynamic_stiffness_n_m = damping_n_s_m * forcing_angular_frequency_rad_s;
    let linear_denominator =
        (linear_dynamic_stiffness_n_m.powi(2) + damping_dynamic_stiffness_n_m.powi(2)).sqrt();
    if !linear_denominator.is_finite() || linear_denominator <= 0.0 {
        return Err(NonlinearAeroelasticModalError::InvalidOptions {
            field: "damping_ratio/forcing_frequency_hz",
            requirement: "must produce a finite nonzero linear dynamic stiffness",
        });
    }
    let linear_amplitude_m = options.forcing_amplitude_n / linear_denominator;

    let residual = |amplitude_m: f64| {
        let effective_stiffness_n_m = linear_dynamic_stiffness_n_m
            + 0.75 * options.cubic_stiffness_n_m3 * amplitude_m.powi(2);
        amplitude_m
            * (effective_stiffness_n_m.powi(2) + damping_dynamic_stiffness_n_m.powi(2)).sqrt()
            - options.forcing_amplitude_n
    };
    let (mut lower_m, mut upper_m) = (0.0, 0.0);
    let mut lower_residual = -options.forcing_amplitude_n;
    let mut upper_residual = lower_residual;
    let mut found_bracket = false;
    for index in 1..=options.bracket_samples {
        let amplitude_m =
            options.maximum_amplitude_m * index as f64 / options.bracket_samples as f64;
        let amplitude_residual = residual(amplitude_m);
        if amplitude_residual.is_finite()
            && (amplitude_residual >= 0.0 || index == options.bracket_samples)
        {
            if amplitude_residual >= 0.0 {
                lower_m = options.maximum_amplitude_m * (index - 1) as f64
                    / options.bracket_samples as f64;
                upper_m = amplitude_m;
                lower_residual = if index == 1 {
                    -options.forcing_amplitude_n
                } else {
                    residual(lower_m)
                };
                upper_residual = amplitude_residual;
                found_bracket = true;
                break;
            }
            upper_residual = amplitude_residual;
        }
    }
    if !found_bracket {
        return Err(NonlinearAeroelasticModalError::NoRootWithinAmplitude {
            maximum_amplitude_m: options.maximum_amplitude_m,
            residual_at_limit_n: upper_residual,
        });
    }

    let mut amplitude_m = if upper_residual.abs() < lower_residual.abs() {
        upper_m
    } else {
        lower_m
    };
    let mut iterations = 0;
    let mut residual_n = residual(amplitude_m).abs();
    for iteration in 1..=options.max_iterations {
        iterations = iteration;
        let midpoint_m = 0.5 * (lower_m + upper_m);
        let midpoint_residual = residual(midpoint_m);
        amplitude_m = midpoint_m;
        residual_n = midpoint_residual.abs();
        if residual_n <= options.residual_tolerance_n {
            break;
        }
        if midpoint_residual >= 0.0 {
            upper_m = midpoint_m;
        } else {
            lower_m = midpoint_m;
        }
    }
    if residual_n > options.residual_tolerance_n {
        return Err(NonlinearAeroelasticModalError::NonConverged {
            iterations,
            residual_n,
            tolerance_n: options.residual_tolerance_n,
        });
    }

    let effective_stiffness_n_m =
        linear_dynamic_stiffness_n_m + 0.75 * options.cubic_stiffness_n_m3 * amplitude_m.powi(2);
    let phase_rad = damping_dynamic_stiffness_n_m.atan2(effective_stiffness_n_m);
    let alternating_stress_pa = options.modal_stress_per_m_pa * amplitude_m;
    let result = NonlinearAeroelasticModalResult {
        linear_stiffness_n_m,
        damping_n_s_m,
        forcing_angular_frequency_rad_s,
        selected_bracket_m: [lower_m, upper_m],
        amplitude_m,
        phase_rad,
        nonlinear_residual_n: residual_n,
        linear_amplitude_m,
        nonlinear_to_linear_amplitude_ratio: amplitude_m / linear_amplitude_m,
        alternating_stress_pa,
        feasible: amplitude_m <= options.maximum_amplitude_m
            && alternating_stress_pa <= options.maximum_stress_pa,
        iterations,
    };
    debug_assert!(result.is_finite());
    Ok(result)
}

fn validate_options(
    options: &NonlinearAeroelasticModalOptions,
) -> Result<(), NonlinearAeroelasticModalError> {
    for (field, value) in [
        ("effective_mass_kg", options.effective_mass_kg),
        ("natural_frequency_hz", options.natural_frequency_hz),
        ("damping_ratio", options.damping_ratio),
        ("forcing_amplitude_n", options.forcing_amplitude_n),
        ("forcing_frequency_hz", options.forcing_frequency_hz),
        ("modal_stress_per_m_pa", options.modal_stress_per_m_pa),
        ("maximum_amplitude_m", options.maximum_amplitude_m),
        ("maximum_stress_pa", options.maximum_stress_pa),
        ("residual_tolerance_n", options.residual_tolerance_n),
    ] {
        if !value.is_finite() {
            return Err(NonlinearAeroelasticModalError::InvalidOptions {
                field,
                requirement: "finite",
            });
        }
    }
    if options.effective_mass_kg <= 0.0 {
        return Err(NonlinearAeroelasticModalError::InvalidOptions {
            field: "effective_mass_kg",
            requirement: "> 0",
        });
    }
    if options.natural_frequency_hz <= 0.0 {
        return Err(NonlinearAeroelasticModalError::InvalidOptions {
            field: "natural_frequency_hz",
            requirement: "> 0",
        });
    }
    if options.damping_ratio <= 0.0 {
        return Err(NonlinearAeroelasticModalError::InvalidOptions {
            field: "damping_ratio",
            requirement: "> 0 for a finite linear comparison",
        });
    }
    if !options.cubic_stiffness_n_m3.is_finite() {
        return Err(NonlinearAeroelasticModalError::InvalidOptions {
            field: "cubic_stiffness_n_m3",
            requirement: "finite",
        });
    }
    if options.forcing_amplitude_n <= 0.0 {
        return Err(NonlinearAeroelasticModalError::InvalidOptions {
            field: "forcing_amplitude_n",
            requirement: "> 0",
        });
    }
    if options.forcing_frequency_hz < 0.0 {
        return Err(NonlinearAeroelasticModalError::InvalidOptions {
            field: "forcing_frequency_hz",
            requirement: ">= 0",
        });
    }
    if options.modal_stress_per_m_pa <= 0.0 {
        return Err(NonlinearAeroelasticModalError::InvalidOptions {
            field: "modal_stress_per_m_pa",
            requirement: "> 0",
        });
    }
    if options.maximum_amplitude_m <= 0.0 {
        return Err(NonlinearAeroelasticModalError::InvalidOptions {
            field: "maximum_amplitude_m",
            requirement: "> 0",
        });
    }
    if options.maximum_stress_pa <= 0.0 {
        return Err(NonlinearAeroelasticModalError::InvalidOptions {
            field: "maximum_stress_pa",
            requirement: "> 0",
        });
    }
    if options.bracket_samples < 2 {
        return Err(NonlinearAeroelasticModalError::InvalidOptions {
            field: "bracket_samples",
            requirement: ">= 2",
        });
    }
    if options.max_iterations == 0 {
        return Err(NonlinearAeroelasticModalError::InvalidOptions {
            field: "max_iterations",
            requirement: "> 0",
        });
    }
    if options.residual_tolerance_n <= 0.0 {
        return Err(NonlinearAeroelasticModalError::InvalidOptions {
            field: "residual_tolerance_n",
            requirement: "> 0",
        });
    }
    Ok(())
}
