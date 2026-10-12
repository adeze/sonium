//! One-point crash shell elements: Q4 and T3 explicit paths.
//!
//! Under-integrated 4-node quadrilateral shells (Theory Manual "Shell
//! Elements", pp. 106-151; "One-Point Quadrature Shell Element",
//! p. 142) evaluate membrane, bending, and transverse-shear strain
//! rates at the element center and stabilize the rank deficiency with
//! Flanagan-Belytschko hourglass control ("Hourglass Modes", p. 117;
//! "Hourglass Viscous Forces", p. 118; "Hourglass Viscous Moments",
//! p. 119; "Physical Hourglass Forces", p. 122). Three-node shells
//! (Theory Manual "Three Node Shell Shape Functions", p. 151) reuse
//! the constant-strain membrane
//! ([`ShellElement::cst_b_matrix`](super::element::ShellElement::cst_b_matrix))
//! and the DKT curvature operator
//! ([`dkt_b_matrix_at`](super::dkt::dkt_b_matrix_at)) at the centroid,
//! fully integrated, so no hourglass control enters.
//!
//! Stress at each through-thickness point comes from any frozen C1
//! [`StressUpdateModel`](super::super::plasticity::update::StressUpdateModel)
//! through a plane-stress projection (thickness strain iterated so
//! `szz = 0`). `/DEF_SHELL` formulation flags (Reference Guide
//! pp. 55-58) select the hourglass variant, strain, thickness,
//! plasticity, triangle, and drilling behavior; the fully integrated
//! QBAT (Ishell 12) and QEPH (Ishell 24) formulations need different
//! kernels and return [`CrashError`] instead of silently degrading,
//! with the dated reason on [`ShellFormulation::native_status`].
//! The DKT18 (Ish3n 30) and DKT_S3 (Ish3n 31) triangles likewise fail
//! loudly with the reason on [`TriShellFormulation::native_status`].
//! [`ThickShellHook`] carries transverse-shear resultants from thin
//! crash shells; the solid-shell (TSHELL) build/defer verdict lives on
//! [`THICK_SHELL_VERDICT`] (deferred 2026-10-12, no target).
//!
//! Geometry reuses the seeds: [`QuadShellElement`](super::mitc4::QuadShellElement)
//! frames the Q4 path, [`ShellElement`](super::element::ShellElement)
//! frames the T3 path. Local DOF order per node is `[u, v, w, tx, ty,
//! tz]`, matching both seeds. All quantities use SI units.

// Rust guideline compliant 2026-02-21

use super::super::plasticity::update::{StressUpdate, StressUpdateModel};
use super::dkt::dkt_b_matrix_at;
use super::element::ShellElement;
use super::mitc4::QuadShellElement;

/// Error of the crash-shell explicit path.
#[derive(Debug, Clone)]
pub struct CrashError {
    /// Human-readable failure description.
    pub message: String,
}

impl CrashError {
    /// Builds an error from a message.
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

impl std::fmt::Display for CrashError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "crash shell error: {}", self.message)
    }
}

impl std::error::Error for CrashError {}

/// Native-kernel availability of a deck formulation flag.
///
/// Every `/DEF_SHELL` formulation reports either its implementing
/// kernel or a dated deferral reason; dispatch sites embed the reason
/// in their [`CrashError`] so unsupported flags fail loudly with a
/// pointer instead of a bare rejection.
///
/// # Examples
///
/// ```rust
/// use sonium_fem::physics::solid::shell::crash::{
///     NativeStatus, ShellFormulation, TriShellFormulation,
/// };
///
/// assert!(ShellFormulation::BelytschkoTsay.native_status().is_implemented());
/// assert!(matches!(
///     TriShellFormulation::Dkt18.native_status(),
///     NativeStatus::Deferred { .. }
/// ));
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NativeStatus {
    /// Implemented by the named native kernel.
    Implemented(&'static str),
    /// Deferred with the verdict date and reason.
    Deferred {
        /// Verdict date (`YYYY-MM-DD`).
        since: &'static str,
        /// Why no native kernel exists.
        reason: &'static str,
    },
}

impl NativeStatus {
    /// Reports whether a native kernel implements the flag.
    #[must_use]
    pub fn is_implemented(&self) -> bool {
        matches!(self, Self::Implemented(_))
    }
}

impl std::fmt::Display for NativeStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Implemented(kernel) => write!(f, "implemented ({kernel})"),
            Self::Deferred { since, reason } => write!(f, "deferred since {since}: {reason}"),
        }
    }
}

/// `/DEF_SHELL Ishell` quadrilateral formulation flag (Reference Guide p. 55).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShellFormulation {
    /// Q4 Belytschko orthogonal visco-elastic hourglass (explicit default).
    BelytschkoTsay = 1,
    /// Q4 Hallquist non-orthogonal visco-elastic hourglass.
    Hallquist = 2,
    /// Q4 elasto-plastic hourglass with orthogonality.
    ElastoPlasticHg = 3,
    /// Q4 improved type 1 with warped-element orthogonalization.
    WarpOrthogonal = 4,
    /// Fully integrated QBAT shell (needs another kernel).
    Qbat = 12,
    /// QEPH shell (needs another kernel).
    Qeph = 24,
}

impl ShellFormulation {
    /// Parses the deck flag (0 selects the explicit default 1).
    ///
    /// # Errors
    ///
    /// Returns [`CrashError`] for flags outside `{0, 1, 2, 3, 4, 12, 24}`.
    pub fn from_flag(flag: i32) -> Result<Self, CrashError> {
        match flag {
            0 | 1 => Ok(Self::BelytschkoTsay),
            2 => Ok(Self::Hallquist),
            3 => Ok(Self::ElastoPlasticHg),
            4 => Ok(Self::WarpOrthogonal),
            12 => Ok(Self::Qbat),
            24 => Ok(Self::Qeph),
            other => Err(CrashError::new(format!("Ishell flag {other} is unknown"))),
        }
    }

    /// Reports whether the one-point kernel implements the formulation.
    #[must_use]
    pub fn is_one_point(&self) -> bool {
        matches!(
            self,
            Self::BelytschkoTsay | Self::Hallquist | Self::ElastoPlasticHg | Self::WarpOrthogonal
        )
    }

    /// Native-kernel availability of the formulation.
    ///
    /// Flags 1-4 run the one-point kernel ([`q4_crash_forces`]); QBAT
    /// (12) and QEPH (24) carry their dated deferral reasons.
    #[must_use]
    pub fn native_status(&self) -> NativeStatus {
        match self {
            Self::BelytschkoTsay
            | Self::Hallquist
            | Self::ElastoPlasticHg
            | Self::WarpOrthogonal => NativeStatus::Implemented("one-point Q4 crash kernel"),
            Self::Qbat => NativeStatus::Deferred {
                since: "2026-10-12",
                reason: "QBAT needs a fully integrated 2x2 Bathe-Dvorkin kernel \
                     with membrane-shear locking control; no such kernel exists \
                     on the MITC4 seed",
            },
            Self::Qeph => NativeStatus::Deferred {
                since: "2026-10-12",
                reason: "QEPH needs physical (Belytschko-Leviathan) stabilization \
                     projections; no such projector exists on the MITC4 seed",
            },
        }
    }
}

/// `/DEF_SHELL Ismstr` small-strain flag (Reference Guide pp. 55-56).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShellStrainFlag {
    /// Small strain from time = 0.
    SmallStrain = 1,
    /// Full geometric nonlinearities (explicit default).
    FullNonlinear = 2,
    /// Old small strain (only with Ishell = 2).
    OldSmallStrain = 3,
    /// Full nonlinearities, engine switch disabled.
    FullNoSwitch = 4,
    /// Automatic best value.
    Auto = -1,
    /// Automatic best value, overwriting property cards.
    AutoForce = -2,
}

impl ShellStrainFlag {
    /// Parses the deck flag (0 selects the default 2).
    ///
    /// # Errors
    ///
    /// Returns [`CrashError`] for flags outside `{-2, -1, 0, 1, 2, 3, 4}`.
    pub fn from_flag(flag: i32) -> Result<Self, CrashError> {
        match flag {
            0 | 2 => Ok(Self::FullNonlinear),
            1 => Ok(Self::SmallStrain),
            3 => Ok(Self::OldSmallStrain),
            4 => Ok(Self::FullNoSwitch),
            -1 => Ok(Self::Auto),
            -2 => Ok(Self::AutoForce),
            other => Err(CrashError::new(format!("Ismstr flag {other} is unknown"))),
        }
    }
}

/// `/DEF_SHELL Ithick` thickness-update flag (Reference Guide pp. 56-57).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShellThicknessFlag {
    /// Thickness change is taken into account.
    Update = 1,
    /// Thickness stays constant (default).
    Constant = 2,
    /// Automatic best value.
    Auto = -1,
    /// Automatic best value, overwriting property cards.
    AutoForce = -2,
}

impl ShellThicknessFlag {
    /// Parses the deck flag (0 selects the default 2).
    ///
    /// # Errors
    ///
    /// Returns [`CrashError`] for flags outside `{-2, -1, 0, 1, 2}`.
    pub fn from_flag(flag: i32) -> Result<Self, CrashError> {
        match flag {
            0 | 2 => Ok(Self::Constant),
            1 => Ok(Self::Update),
            -1 => Ok(Self::Auto),
            -2 => Ok(Self::AutoForce),
            other => Err(CrashError::new(format!("Ithick flag {other} is unknown"))),
        }
    }
}

/// `/DEF_SHELL Iplas` plane-stress plasticity flag (Reference Guide p. 57).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShellPlasticityFlag {
    /// Iterative projection with three Newton iterations.
    Iterative = 1,
    /// Radial return (explicit default).
    RadialReturn = 2,
    /// Automatic best value.
    Auto = -1,
    /// Automatic best value, overwriting property cards.
    AutoForce = -2,
}

impl ShellPlasticityFlag {
    /// Parses the deck flag (0 selects the explicit default 2).
    ///
    /// # Errors
    ///
    /// Returns [`CrashError`] for flags outside `{-2, -1, 0, 1, 2}`.
    pub fn from_flag(flag: i32) -> Result<Self, CrashError> {
        match flag {
            0 | 2 => Ok(Self::RadialReturn),
            1 => Ok(Self::Iterative),
            -1 => Ok(Self::Auto),
            -2 => Ok(Self::AutoForce),
            other => Err(CrashError::new(format!("Iplas flag {other} is unknown"))),
        }
    }
}

/// `/DEF_SHELL Ish3n` 3-node shell flag (Reference Guide pp. 57-58).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TriShellFormulation {
    /// Standard C0 triangle.
    C0 = 1,
    /// Standard C0 triangle with large-rotation modification (default).
    C0LargeRotation = 2,
    /// DKT18 (needs membrane-drilling coupling).
    Dkt18 = 30,
    /// DKT_S3 after Batoz (needs membrane-drilling coupling).
    DktS3 = 31,
}

impl TriShellFormulation {
    /// Parses the deck flag (0 selects the default 2).
    ///
    /// # Errors
    ///
    /// Returns [`CrashError`] for flags outside `{0, 1, 2, 30, 31}`.
    pub fn from_flag(flag: i32) -> Result<Self, CrashError> {
        match flag {
            0 | 2 => Ok(Self::C0LargeRotation),
            1 => Ok(Self::C0),
            30 => Ok(Self::Dkt18),
            31 => Ok(Self::DktS3),
            other => Err(CrashError::new(format!("Ish3n flag {other} is unknown"))),
        }
    }

    /// Reports whether the centroid kernel implements the formulation.
    #[must_use]
    pub fn is_centroid(&self) -> bool {
        matches!(self, Self::C0 | Self::C0LargeRotation)
    }

    /// Native-kernel availability of the formulation.
    ///
    /// Flags 1-2 run the centroid kernel ([`t3_crash_forces`]); DKT18
    /// (30) and DKT_S3 (31) carry their dated deferral reasons.
    #[must_use]
    pub fn native_status(&self) -> NativeStatus {
        match self {
            Self::C0 | Self::C0LargeRotation => {
                NativeStatus::Implemented("centroid T3 crash kernel")
            }
            Self::Dkt18 => NativeStatus::Deferred {
                since: "2026-10-12",
                reason: "DKT18 needs membrane-drilling coupling on the DKT plate \
                     plus a non-conforming membrane; the centroid kernel carries \
                     no drilling coupling",
            },
            Self::DktS3 => NativeStatus::Deferred {
                since: "2026-10-12",
                reason: "DKT_S3 after Batoz needs the discrete-shear triangle \
                     with drilling coupling; the CST/DKT centroid seed has neither",
            },
        }
    }
}

/// `/DEF_SHELL Idril` drilling-stiffness flag (Reference Guide p. 58).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DrillingFlag {
    /// Drilling stiffness active.
    Yes = 1,
    /// No drilling stiffness (explicit default).
    No = 2,
}

impl DrillingFlag {
    /// Parses the deck flag (0 selects the explicit default 2).
    ///
    /// # Errors
    ///
    /// Returns [`CrashError`] for flags outside `{0, 1, 2}`.
    pub fn from_flag(flag: i32) -> Result<Self, CrashError> {
        match flag {
            0 | 2 => Ok(Self::No),
            1 => Ok(Self::Yes),
            other => Err(CrashError::new(format!("Idril flag {other} is unknown"))),
        }
    }
}

/// `/DEF_SHELL` default set for crash shells (Reference Guide pp. 55-58).
///
/// `Istrain` (post-processing strain output) is driver-side and not
/// modeled here.
///
/// # Examples
///
/// ```rust
/// use sonium_fem::physics::solid::shell::crash::{DefShell, DrillingFlag};
///
/// let def = DefShell::explicit_defaults();
/// assert_eq!(def.drilling, DrillingFlag::No);
/// assert!(def.validate().is_ok());
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DefShell {
    /// Quadrilateral formulation flag.
    pub formulation: ShellFormulation,
    /// Small-strain flag.
    pub strain: ShellStrainFlag,
    /// Thickness-update flag.
    pub thickness: ShellThicknessFlag,
    /// Plane-stress plasticity flag.
    pub plasticity: ShellPlasticityFlag,
    /// Triangle formulation flag.
    pub tri: TriShellFormulation,
    /// Drilling-stiffness flag.
    pub drilling: DrillingFlag,
}

impl DefShell {
    /// Explicit-engine defaults (no `/IMPLICIT` in deck).
    #[must_use]
    pub fn explicit_defaults() -> Self {
        Self {
            formulation: ShellFormulation::BelytschkoTsay,
            strain: ShellStrainFlag::FullNonlinear,
            thickness: ShellThicknessFlag::Constant,
            plasticity: ShellPlasticityFlag::RadialReturn,
            tri: TriShellFormulation::C0LargeRotation,
            drilling: DrillingFlag::No,
        }
    }

    /// Parses six raw deck flags in card order.
    ///
    /// # Errors
    ///
    /// Returns [`CrashError`] when any flag value is unknown.
    pub fn from_flags(
        ishell: i32,
        ismstr: i32,
        ithick: i32,
        iplas: i32,
        ish3n: i32,
        idril: i32,
    ) -> Result<Self, CrashError> {
        Ok(Self {
            formulation: ShellFormulation::from_flag(ishell)?,
            strain: ShellStrainFlag::from_flag(ismstr)?,
            thickness: ShellThicknessFlag::from_flag(ithick)?,
            plasticity: ShellPlasticityFlag::from_flag(iplas)?,
            tri: TriShellFormulation::from_flag(ish3n)?,
            drilling: DrillingFlag::from_flag(idril)?,
        })
    }

    /// Rejects incompatible flag combinations.
    ///
    /// The old small-strain formulation only runs with Ishell = 2
    /// (Reference Guide p. 56); every other combination passes here and
    /// dispatches in the force kernels.
    ///
    /// # Errors
    ///
    /// Returns [`CrashError`] for `Ismstr = 3` with `Ishell != 2`.
    pub fn validate(&self) -> Result<(), CrashError> {
        if self.strain == ShellStrainFlag::OldSmallStrain
            && self.formulation != ShellFormulation::Hallquist
        {
            return Err(CrashError::new(
                "Ismstr = 3 (old small strain) needs Ishell = 2",
            ));
        }
        Ok(())
    }
}

/// Through-thickness quadrature point.
#[derive(Debug, Clone, Copy)]
pub struct ThicknessPoint {
    /// Offset from the mid-surface in m.
    pub z: f64,
    /// Quadrature weight in m (weights sum to the thickness).
    pub weight: f64,
    /// Weight for the stress moment integral in m².
    ///
    /// Equals `weight * z` for Gauss integration. Piecewise-linear
    /// stress interpolation has a distinct first-moment weight.
    pub moment_weight: f64,
}

/// Gauss-Legendre thickness quadrature with `n` points.
///
/// Supports 1 to 5 points. This is a Gauss integration rule; the legacy
/// surface-sampled shell rule is [`linear_thickness_points`].
/// Nodes and weights are the standard Gauss-Legendre
/// tables scaled from `[-1, 1]` to `[-t/2, t/2]`.
///
/// # Examples
///
/// ```rust
/// use sonium_fem::physics::solid::shell::crash::thickness_points;
///
/// let points = thickness_points(5, 0.002).unwrap();
/// let total: f64 = points.iter().map(|p| p.weight).sum();
/// assert!((total - 0.002).abs() < 1e-15);
/// assert!(points.iter().all(|p| p.z.abs() <= 0.001));
/// ```
///
/// # Errors
///
/// Returns [`CrashError`] for point counts outside 1-5 or non-positive
/// thickness.
pub fn thickness_points(n: usize, thickness: f64) -> Result<Vec<ThicknessPoint>, CrashError> {
    if !thickness.is_finite() || thickness <= 0.0 {
        return Err(CrashError::new("shell thickness must be positive finite"));
    }
    // Standard Gauss-Legendre nodes/weights on [-1, 1]; 17 digits keep
    // full f64 fidelity for the weight sums tests check.
    let table: &[(f64, f64)] = match n {
        1 => &[(0.0, 2.0)],
        2 => &[(-0.5773502691896258, 1.0), (0.5773502691896258, 1.0)],
        3 => &[
            (-0.7745966692414834, 5.0 / 9.0),
            (0.0, 8.0 / 9.0),
            (0.7745966692414834, 5.0 / 9.0),
        ],
        4 => &[
            (-0.8611363115940526, 0.3478548451374538),
            (-0.3399810435848563, 0.6521451548625461),
            (0.3399810435848563, 0.6521451548625461),
            (0.8611363115940526, 0.3478548451374538),
        ],
        5 => &[
            (-0.906179845938664, 0.2369268850561891),
            (-0.5384693101056831, 0.4786286704993665),
            (0.0, 0.5688888888888889),
            (0.5384693101056831, 0.4786286704993665),
            (0.906179845938664, 0.2369268850561891),
        ],
        _ => {
            return Err(CrashError::new(format!(
                "thickness quadrature needs 1-5 points, got {n}"
            )));
        }
    };
    Ok(table
        .iter()
        .map(|(xi, w)| ThicknessPoint {
            z: xi * thickness / 2.0,
            weight: w * thickness / 2.0,
            moment_weight: w * xi * thickness.powi(2) / 4.0,
        })
        .collect())
}

/// Integrates piecewise-linear stresses sampled through the thickness.
///
/// Uniformly spaced samples include both shell surfaces. Membrane weights
/// integrate the linear nodal hat functions; moment weights integrate those
/// functions multiplied by z. Separate moment weights reproduce the exact
/// elastic bending stiffness rather than the trapezoidal z² approximation.
///
/// # Errors
///
/// Returns [`CrashError`] for counts outside 2-11 or non-positive thickness.
pub fn linear_thickness_points(
    n: usize,
    thickness: f64,
) -> Result<Vec<ThicknessPoint>, CrashError> {
    if !(2..=11).contains(&n) || !thickness.is_finite() || thickness <= 0. {
        return Err(CrashError::new(
            "linear thickness integration needs 2-11 points and positive finite thickness",
        ));
    }
    let spacing = thickness / (n - 1) as f64;
    Ok((0..n)
        .map(|i| {
            let z = -0.5 * thickness + i as f64 * spacing;
            let endpoint = i == 0 || i + 1 == n;
            let weight = if endpoint { 0.5 * spacing } else { spacing };
            let moment_weight = z * weight
                + if i == 0 {
                    spacing.powi(2) / 6.
                } else if i + 1 == n {
                    -spacing.powi(2) / 6.
                } else {
                    0.
                };
            ThicknessPoint {
                z,
                weight,
                moment_weight,
            }
        })
        .collect())
}

/// Per-layer material states of a crash shell.
#[derive(Debug, Clone)]
pub struct LayerStates<S> {
    /// Material states, one per thickness point.
    pub states: Vec<S>,
    /// Thickness quadrature.
    pub points: Vec<ThicknessPoint>,
    /// Current thickness scale (1.0 unstrained).
    pub thickness_scale: f64,
    /// Accumulated local engineering transverse shear `[gamma_xz, gamma_yz]`.
    /// The elastic shear hook consumes this total, including held increments.
    pub transverse_shear: [f64; 2],
}

impl<S: Clone> LayerStates<S> {
    /// Builds surface-sampled states with piecewise-linear stress integration.
    ///
    /// # Errors
    ///
    /// Returns [`CrashError`] for counts outside 2-11 or non-positive thickness.
    pub fn new_piecewise_linear(
        point: S,
        n_points: usize,
        thickness: f64,
    ) -> Result<Self, CrashError> {
        let points = linear_thickness_points(n_points, thickness)?;
        Ok(Self {
            states: vec![point; n_points],
            points,
            thickness_scale: 1.,
            transverse_shear: [0.; 2],
        })
    }

    /// Builds layer states from one virgin point state.
    ///
    /// # Errors
    ///
    /// Returns [`CrashError`] for point counts outside 1-5 or
    /// non-positive thickness.
    pub fn new(point: S, n_points: usize, thickness: f64) -> Result<Self, CrashError> {
        let points = thickness_points(n_points, thickness)?;
        Ok(Self {
            states: vec![point; n_points],
            points,
            thickness_scale: 1.0,
            transverse_shear: [0.0; 2],
        })
    }
}

/// Transverse-shear hook for thin crash shells.
///
/// Thin crash shells select their transverse-shear behavior through
/// this hook: [`NoThickShell`] for Kirchhoff (no shear energy),
/// [`ElasticShearHook`] for shear-stiff response. A solid-shell
/// (TSHELL) element is a different element, not a hook setting; its
/// build/defer verdict lives on [`THICK_SHELL_VERDICT`].
pub trait ThickShellHook: std::fmt::Debug {
    /// Shear forces `[Qx, Qy]` in N/m from shear strains.
    fn shear_forces(&self, shear_strain: &[f64; 2], thickness: f64) -> [f64; 2];
}

/// Thick-shell (TSHELL) build/defer verdict (Phase 7 lane 7B).
///
/// DEFER: a solid-shell element needs 8-node kinematics with
/// through-thickness stretch plus a dedicated explicit kernel (Theory
/// Manual solid-shell theory, p. 164); no current seed covers either,
/// and no harvested thick-shell golden exists to prove a native solve
/// against. Thin crash shells keep [`NoThickShell`] as the deliberate
/// Kirchhoff default and [`ElasticShearHook`] as the shear-stiff path.
pub const THICK_SHELL_VERDICT: NativeStatus = NativeStatus::Deferred {
    since: "2026-10-12",
    reason: "solid-shell TSHELL needs 8-node through-thickness-stretch \
         kinematics with a dedicated explicit kernel; no seed covers it, \
         and no harvested thick-shell golden exists to prove a native solve",
};

/// Thin-shell default: no transverse-shear resultants.
///
/// This is the deliberate Kirchhoff selection (transverse shear
/// carries no energy in thin-shell theory), not a placeholder: the
/// solid-shell build question is settled on [`THICK_SHELL_VERDICT`].
#[derive(Debug, Clone, Copy, Default)]
pub struct NoThickShell;

impl ThickShellHook for NoThickShell {
    fn shear_forces(&self, _shear_strain: &[f64; 2], _thickness: f64) -> [f64; 2] {
        [0.0, 0.0]
    }
}

/// Isotropic elastic transverse shear: `Q = kappa G t gamma`.
///
/// Uses the Reissner [`SHEAR_CORRECTION`](super::section::SHEAR_CORRECTION),
/// the same value as the MITC4 seed, for drivers that need shear-stiff
/// thin shells without a solid-shell element.
#[derive(Debug, Clone, Copy)]
pub struct ElasticShearHook {
    /// Shear modulus in Pa.
    pub shear_modulus: f64,
}

impl ElasticShearHook {
    /// Builds the hook from a shear modulus in Pa.
    ///
    /// # Panics
    ///
    /// Panics for non-positive moduli.
    #[must_use]
    pub fn new(shear_modulus: f64) -> Self {
        assert!(
            shear_modulus.is_finite() && shear_modulus > 0.0,
            "shear modulus must be positive finite"
        );
        Self { shear_modulus }
    }
}

impl ThickShellHook for ElasticShearHook {
    fn shear_forces(&self, shear_strain: &[f64; 2], thickness: f64) -> [f64; 2] {
        // Single source: `section::SHEAR_CORRECTION` documents the value.
        let factor = super::section::SHEAR_CORRECTION * self.shear_modulus * thickness;
        [factor * shear_strain[0], factor * shear_strain[1]]
    }
}

/// Hourglass stabilization coefficients of the Q4 path (SI).
///
/// Viscous coefficients damp the hourglass velocity projection
/// `gamma . v`; stiffness coefficients resist the hourglass
/// displacement projection `gamma . u`. All scale with the `1/4`
/// normalization of `gamma . gamma = 4` on the square.
#[derive(Debug, Clone, Copy, Default)]
pub struct HourglassParams {
    /// Membrane viscous coefficient in N s/m.
    pub membrane_viscous: f64,
    /// Membrane stiffness coefficient in N/m.
    pub membrane_stiffness: f64,
    /// Bending viscous coefficient in N m s.
    pub bending_viscous: f64,
    /// Bending stiffness coefficient in N m.
    pub bending_stiffness: f64,
    /// Transverse-displacement viscous coefficient in N s/m.
    ///
    /// Controls the checkerboard normal-velocity mode separately from
    /// the rotational modes governed by `bending_viscous`.
    pub transverse_viscous: f64,
    /// Transverse-displacement stiffness coefficient in N/m.
    pub transverse_stiffness: f64,
    /// Drilling stiffness in N m/rad (0 disables).
    pub drilling: f64,
}

impl HourglassParams {
    /// Derives coefficients from material and element scales.
    ///
    /// Takes density, sound speed, a shear modulus (elastic `G`, or the
    /// current tangent for `ElastoPlasticHg`), area, thickness, and
    /// characteristic length. The formulation selects the active terms:
    /// BelytschkoTsay and WarpOrthogonal use viscous terms only,
    /// Hallquist likewise with the raw (non-orthogonal) hourglass
    /// vector handled in the kernel, ElastoPlasticHg adds stiffness
    /// terms scaled by the passed modulus, and QBAT/QEPH fail. The
    /// viscous (`0.05`) and stiffness (`0.05`) scales sit in the
    /// literature 0.01-0.10 band; drilling defaults to the
    /// `1e-3`-of-`G t A` seed scale when `drilling` is true.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use sonium_fem::physics::solid::shell::crash::{HourglassParams, ShellFormulation};
    ///
    /// let hg = HourglassParams::for_shell(
    ///     ShellFormulation::BelytschkoTsay,
    ///     7800.0,
    ///     4500.0,
    ///     80.0e9,
    ///     1.0,
    ///     0.002,
    ///     1.0,
    ///     false,
    /// )
    /// .unwrap();
    /// assert!(hg.membrane_viscous > 0.0);
    /// assert_eq!(hg.membrane_stiffness, 0.0);
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`CrashError`] for QBAT/QEPH or non-positive scales.
    pub fn for_shell(
        formulation: ShellFormulation,
        density: f64,
        sound_speed: f64,
        shear_modulus: f64,
        area: f64,
        thickness: f64,
        char_length: f64,
        drilling: bool,
    ) -> Result<Self, CrashError> {
        if !formulation.is_one_point() {
            return Err(CrashError::new(format!(
                "one-point hourglass has no {formulation:?} variant ({})",
                formulation.native_status()
            )));
        }
        for (name, value) in [
            ("density", density),
            ("sound speed", sound_speed),
            ("shear modulus", shear_modulus),
            ("area", area),
            ("thickness", thickness),
            ("char length", char_length),
        ] {
            if !value.is_finite() || value <= 0.0 {
                return Err(CrashError::new(format!("{name} must be positive finite")));
            }
        }
        // Damped-spurious-mode band 0.01-0.10; 0.05 damps hourglass
        // motion within a few steps without touching physical waves.
        const VISCOUS_SCALE: f64 = 0.05;
        const STIFFNESS_SCALE: f64 = 0.05;
        // Drilling seed scale: 1e-3 of the in-plane shear stiffness,
        // matching the MITC4 stabilization family.
        const DRILL_SCALE: f64 = 1e-3;
        let membrane_viscous =
            VISCOUS_SCALE * density * sound_speed * thickness * char_length / 4.0;
        let bending_viscous =
            VISCOUS_SCALE * density * sound_speed * thickness.powi(3) * char_length / 48.0;
        let elastoplastic = formulation == ShellFormulation::ElastoPlasticHg;
        let membrane_stiffness = if elastoplastic {
            STIFFNESS_SCALE * shear_modulus * thickness / 4.0
        } else {
            0.0
        };
        let bending_stiffness = if elastoplastic {
            STIFFNESS_SCALE * shear_modulus * thickness.powi(3) / 48.0
        } else {
            0.0
        };
        // Transverse stabilization uses the shear wave speed and the
        // thickness scale; its modal damping has units N s/m.
        let transverse_viscous =
            VISCOUS_SCALE * (density * shear_modulus).sqrt() * thickness.powi(2) / 4.0;
        // A normal-displacement mode has curvature of order w / L^2;
        // convert the rotational bending coefficient with L^2 ~ area.
        let transverse_stiffness = bending_stiffness / area;
        Ok(Self {
            transverse_viscous,
            transverse_stiffness,
            membrane_viscous,
            membrane_stiffness,
            bending_viscous,
            bending_stiffness,
            drilling: if drilling {
                DRILL_SCALE * shear_modulus * thickness * area / 4.0
            } else {
                0.0
            },
        })
    }
}

/// Plane-stress projection of a C1 material step.
///
/// Iterates the thickness strain so the C1 update returns `szz = 0`
/// for the in-plane increment `d_mem + z d_curv`. `Iterative` runs at
/// most three Newton corrections (Reference Guide p. 57);
/// `RadialReturn` iterates to a tight tolerance. Uses the pure `update`
/// (or `update_with_rate` for rate-dependent materials) so callers
/// commit exactly once. Returns the accepted update plus `dezz`.
///
/// # Errors
///
/// Returns [`CrashError`] for non-positive `dt` or a non-positive
/// `C22` tangent (iteration impossible).
pub fn plane_stress_step<M: StressUpdateModel<StrainIncrement = [f64; 6]>>(
    material: &M,
    state: &M::State,
    d_membrane: &[f64; 3],
    d_curvature: &[f64; 3],
    z: f64,
    dt: f64,
    flag: ShellPlasticityFlag,
) -> Result<(StressUpdate, f64), CrashError> {
    if dt.is_nan() || dt <= 0.0 {
        return Err(CrashError::new("explicit step dt must be positive"));
    }
    let d_inplane = [
        d_membrane[0] + z * d_curvature[0],
        d_membrane[1] + z * d_curvature[1],
        d_membrane[2] + z * d_curvature[2],
    ];
    let step = |dezz: f64| {
        let d_eps = [d_inplane[0], d_inplane[1], dezz, d_inplane[2], 0.0, 0.0];
        if material.is_rate_dependent() {
            material.update_with_rate(state, &d_eps, dt)
        } else {
            material.update(state, &d_eps)
        }
    };
    let max_iter = match flag {
        // Three Newton iterations per the manual; Auto resolves to the
        // explicit radial-return default.
        ShellPlasticityFlag::Iterative => 3,
        _ => 25,
    };
    let mut dezz = 0.0;
    let mut update = step(dezz);
    for _ in 0..max_iter {
        let inplane = update.stress[0].abs() + update.stress[1].abs();
        // Relative tolerance 1e-9 on the in-plane stress with a 1 Pa
        // floor so stress-free steps converge immediately.
        if update.stress[2].abs() <= 1e-9 * inplane.max(1.0) {
            break;
        }
        let c22 = update.tangent[2 * 6 + 2];
        if c22.is_nan() || c22 <= 0.0 {
            return Err(CrashError::new("plane-stress iteration needs C22 > 0"));
        }
        dezz -= update.stress[2] / c22;
        update = step(dezz);
    }
    Ok((update, dezz))
}

/// Membrane, bending, and shear resultants of a shell point.
#[derive(Debug, Clone, Copy, Default)]
pub struct ShellResultants {
    /// Membrane forces `[Nxx, Nyy, Nxy]` in N/m.
    pub membrane: [f64; 3],
    /// Bending moments `[Mxx, Myy, Mxy]` in N.
    pub bending: [f64; 3],
    /// Transverse shear forces `[Qx, Qy]` in N/m.
    pub shear: [f64; 2],
}

/// One-point Q4 crash-shell geometry in its local frame.
#[derive(Debug, Clone)]
pub struct Q4CrashGeom {
    /// Element area in m^2.
    pub area: f64,
    /// Shell thickness in m.
    pub thickness: f64,
    /// Global origin of the local frame.
    pub origin: [f64; 3],
    /// Local x axis in global coordinates.
    pub ex: [f64; 3],
    /// Local y axis in global coordinates.
    pub ey: [f64; 3],
    /// Local z axis (normal) in global coordinates.
    pub ez: [f64; 3],
    /// Corner coordinates in the local plane.
    pub local: [(f64, f64); 4],
    /// Corner out-of-plane warp offsets in m.
    pub warp: [f64; 4],
    /// Center membrane B, 3x8 row-major over `[u, v]` per node.
    pub membrane_b: [f64; 24],
    /// Center bending B, 3x12 row-major over `[w, tx, ty]` per node.
    pub bending_b: [f64; 36],
    /// Center shear B, 2x12 row-major over `[w, tx, ty]` per node.
    pub shear_b: [f64; 24],
    /// Orthogonalized hourglass vector (4 nodes).
    ///
    /// Ishell 4 (warp orthogonalization) uses this same vector: the
    /// center warp slopes vanish identically for any normal-consistent
    /// frame (the center tangents are perpendicular to the normal by
    /// construction), so a surface-metric correction reduces to the
    /// projected vector exactly.
    pub gamma: [f64; 4],
    /// Characteristic length `A / L_max` in m.
    pub char_length: f64,
}

/// Q4 natural-coordinate signs in counter-clockwise order.
const QUAD_SIGNS: [(f64, f64); 4] = [(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)];

/// Raw Q4 hourglass pattern.
const QUAD_HG: [f64; 4] = [1.0, -1.0, 1.0, -1.0];

impl Q4CrashGeom {
    /// Builds crash geometry from a MITC4 seed element.
    ///
    /// Reuses the seed frame, local coordinates, warp, area, and
    /// thickness, then derives the center operators and hourglass
    /// vectors.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use sonium_fem::mesh::Point;
    /// use sonium_fem::physics::solid::shell::QuadShellElement;
    /// use sonium_fem::physics::solid::shell::ShellSection;
    /// use sonium_fem::physics::solid::shell::crash::Q4CrashGeom;
    ///
    /// let section = ShellSection::new(200.0e9, 0.3, 0.002, 7800.0);
    /// let quad = QuadShellElement::from_points_with_section(
    ///     [0, 1, 2, 3],
    ///     &Point::new_3d(0.0, 0.0, 0.0),
    ///     &Point::new_3d(1.0, 0.0, 0.0),
    ///     &Point::new_3d(1.0, 1.0, 0.0),
    ///     &Point::new_3d(0.0, 1.0, 0.0),
    ///     section,
    /// );
    /// let geom = Q4CrashGeom::from_quad(&quad).unwrap();
    /// assert!((geom.area - 1.0).abs() < 1e-12);
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`CrashError`] for degenerate (zero-area) quads.
    pub fn from_quad(quad: &QuadShellElement) -> Result<Self, CrashError> {
        Self::from_frame(
            quad.origin,
            [
                [quad.rotation[0], quad.rotation[3], quad.rotation[6]],
                [quad.rotation[1], quad.rotation[4], quad.rotation[7]],
                [quad.rotation[2], quad.rotation[5], quad.rotation[8]],
            ],
            quad.local_coords,
            quad.warp,
            quad.area,
            quad.section.thickness,
        )
    }

    /// Builds crash geometry from four counter-clockwise corners.
    ///
    /// Frames the element with local x along edge 0-1 projected into
    /// the diagonal-cross plane, z normal to that plane, and y completing
    /// the orthonormal basis. Warped edges need this projection to preserve
    /// lengths and force/velocity work under local/global transformations.
    ///
    /// # Errors
    ///
    /// Returns [`CrashError`] for degenerate (zero-area) quads.
    pub fn from_corners(corners: [[f64; 3]; 4], thickness: f64) -> Result<Self, CrashError> {
        if !thickness.is_finite() || thickness <= 0.0 {
            return Err(CrashError::new("shell thickness must be positive finite"));
        }
        let e01 = sub3(&corners[1], &corners[0]);
        let len = norm3(&e01);
        if len.is_nan() || len <= 0.0 {
            return Err(CrashError::new("quad edge 0-1 is degenerate"));
        }
        let d02 = sub3(&corners[2], &corners[0]);
        let d13 = sub3(&corners[3], &corners[1]);
        let normal = cross3(&d02, &d13);
        let nlen = norm3(&normal);
        if nlen.is_nan() || nlen <= 0.0 {
            return Err(CrashError::new("quad normal is degenerate"));
        }
        let ez = [normal[0] / nlen, normal[1] / nlen, normal[2] / nlen];
        // A warped edge has a normal component. Using it directly as
        // local x makes the frame nonorthogonal and changes vector work.
        let normal_component = dot3(&e01, &ez);
        let projected = std::array::from_fn(|i| e01[i] - normal_component * ez[i]);
        let projected_length = norm3(&projected);
        if !projected_length.is_finite() || projected_length <= 0.0 {
            return Err(CrashError::new("quad projected edge 0-1 is degenerate"));
        }
        let ex = projected.map(|component| component / projected_length);
        let ey = cross3(&ez, &ex);
        let mut local = [(0.0, 0.0); 4];
        let mut warp = [0.0; 4];
        for (i, corner) in corners.iter().enumerate() {
            let d = sub3(corner, &corners[0]);
            local[i] = (dot3(&d, &ex), dot3(&d, &ey));
            warp[i] = dot3(&d, &ez);
        }
        let area = 0.5
            * ((local[2].0 - local[0].0) * (local[3].1 - local[1].1)
                - (local[2].1 - local[0].1) * (local[3].0 - local[1].0))
                .abs();
        Self::from_frame(corners[0], [ex, ey, ez], local, warp, area, thickness)
    }

    /// Assembles geometry from a local frame and plane coordinates.
    fn from_frame(
        origin: [f64; 3],
        axes: [[f64; 3]; 3],
        local: [(f64, f64); 4],
        warp: [f64; 4],
        area: f64,
        thickness: f64,
    ) -> Result<Self, CrashError> {
        if !area.is_finite() || area <= 0.0 {
            return Err(CrashError::new("quad area must be positive finite"));
        }
        if !thickness.is_finite() || thickness <= 0.0 {
            return Err(CrashError::new("shell thickness must be positive finite"));
        }
        // Center Jacobian of the bilinear map.
        let mut j = [0.0; 4];
        for (i, (sx, sy)) in QUAD_SIGNS.iter().enumerate() {
            j[0] += local[i].0 * sx / 4.0;
            j[1] += local[i].0 * sy / 4.0;
            j[2] += local[i].1 * sx / 4.0;
            j[3] += local[i].1 * sy / 4.0;
        }
        let det = j[0] * j[3] - j[1] * j[2];
        if det.is_nan() || det <= 0.0 {
            return Err(CrashError::new("quad center Jacobian is degenerate"));
        }
        let inv = [j[3] / det, -j[1] / det, -j[2] / det, j[0] / det];
        // Physical gradients use the inverse transpose: dN/dx is
        // column 0 of J^-1 dotted with the natural pattern (matching
        // the hex seed convention).
        let mut grad = [[0.0; 2]; 4];
        for (i, (sx, sy)) in QUAD_SIGNS.iter().enumerate() {
            let natural = [sx / 4.0, sy / 4.0];
            grad[i][0] = inv[0] * natural[0] + inv[2] * natural[1];
            grad[i][1] = inv[1] * natural[0] + inv[3] * natural[1];
        }
        let mut membrane_b = [0.0; 24];
        let mut bending_b = [0.0; 36];
        let mut shear_b = [0.0; 24];
        for (i, g) in grad.iter().enumerate() {
            membrane_b[i * 2] = g[0];
            membrane_b[8 + i * 2 + 1] = g[1];
            membrane_b[16 + i * 2] = g[1];
            membrane_b[16 + i * 2 + 1] = g[0];
            // Curvatures kxx = d(ty)/dx, kyy = -d(tx)/dy,
            // kxy = d(ty)/dy - d(tx)/dx over [w, tx, ty].
            bending_b[i * 3 + 2] = g[0];
            bending_b[12 + i * 3 + 1] = -g[1];
            bending_b[24 + i * 3 + 1] = -g[0];
            bending_b[24 + i * 3 + 2] = g[1];
            // Transverse shear gxz = dw/dx + ty, gyz = dw/dy - tx;
            // rotations interpolate with N = 1/4 at the center.
            shear_b[i * 3] = g[0];
            shear_b[i * 3 + 2] = 0.25;
            shear_b[12 + i * 3] = g[1];
            shear_b[12 + i * 3 + 1] = -0.25;
        }
        let bx: [f64; 4] = std::array::from_fn(|i| grad[i][0]);
        let by: [f64; 4] = std::array::from_fn(|i| grad[i][1]);
        let gamma = orthogonalize_hg(&QUAD_HG, &bx, &by, &local);
        // No separate warp-aware vector: the center warp slopes
        // vanish identically for normal-consistent frames, so a
        // surface-metric correction would reproduce `gamma` exactly.
        let mut longest: f64 = 0.0;
        for i in 0..4 {
            let a = local[i];
            let b = local[(i + 1) % 4];
            let edge = ((b.0 - a.0).powi(2) + (b.1 - a.1).powi(2)).sqrt();
            longest = longest.max(edge);
        }
        Ok(Self {
            area,
            thickness,
            origin,
            ex: axes[0],
            ey: axes[1],
            ez: axes[2],
            local,
            warp,
            membrane_b,
            bending_b,
            shear_b,
            gamma,
            char_length: area / longest,
        })
    }

    /// Hourglass vector selected by the formulation flag.
    ///
    /// Hallquist uses the raw (non-orthogonal) pattern; every other
    /// one-point formulation, including WarpOrthogonal, uses the
    /// orthogonalized vector (see [`Q4CrashGeom::gamma`]).
    #[must_use]
    pub fn hg_vector(&self, formulation: ShellFormulation) -> [f64; 4] {
        match formulation {
            ShellFormulation::Hallquist => QUAD_HG,
            _ => self.gamma,
        }
    }

    /// Lumped nodal masses and rotary inertias (24-vector).
    ///
    /// Translational mass `rho A t / 4` on `[u, v, w]`, rotary inertia
    /// `m t^2 / 12` on `[tx, ty, tz]` (Theory Manual "Mass and Inertia",
    /// pp. 115-116).
    #[must_use]
    pub fn lumped_masses(&self, density: f64) -> [f64; 24] {
        let nodal = density * self.area * self.thickness / 4.0;
        let rotary = nodal * self.thickness.powi(2) / 12.0;
        let mut masses = [0.0; 24];
        for node in 0..4 {
            masses[node * 6] = nodal;
            masses[node * 6 + 1] = nodal;
            masses[node * 6 + 2] = nodal;
            masses[node * 6 + 3] = rotary;
            masses[node * 6 + 4] = rotary;
            masses[node * 6 + 5] = rotary;
        }
        masses
    }

    /// Critical explicit step `L / c` in s ("Stability Time Step", p. 110).
    #[must_use]
    pub fn critical_step(&self, sound_speed: f64) -> f64 {
        self.char_length / sound_speed
    }
}

/// Flanagan-Belytschko orthogonalization of the hourglass pattern.
///
/// Returns `h - (h.x) bx - (h.y) by`, orthogonal to constant and linear
/// fields on the local coordinates.
fn orthogonalize_hg(
    h: &[f64; 4],
    bx: &[f64; 4],
    by: &[f64; 4],
    local: &[(f64, f64); 4],
) -> [f64; 4] {
    let mut px = 0.0;
    let mut py = 0.0;
    for i in 0..4 {
        px += h[i] * local[i].0;
        py += h[i] * local[i].1;
    }
    std::array::from_fn(|i| h[i] - px * bx[i] - py * by[i])
}

/// Internal-force evaluation of one Q4 crash shell.
#[derive(Debug, Clone, Copy)]
pub struct Q4ForceOut {
    /// Nodal forces/moments in N and N m (local, 24-vector).
    pub forces: [f64; 24],
    /// Resultants at the center point.
    pub resultants: ShellResultants,
    /// Hourglass power in W (non-negative for viscous terms).
    pub hg_power: f64,
    /// Plastic work rate density integral in W.
    pub plastic_work_rate: f64,
}

/// Explicit internal forces of one Q4 crash shell.
///
/// Advances every layer C1 material by its plane-stress increment from
/// the center membrane/curvature rates over `dt`, commits each layer,
/// and returns `A (Bm^T N + Bb^T M + Bs^T Q)` plus hourglass and
/// drilling forces. `u`/`v` are local 24-vectors per node `[u, v, w,
/// tx, ty, tz]`. With `Ithick = Update` the layer thickness scale
/// evolves by the mean thickness strain; `Constant` freezes it. The
/// thick-shell `hook` supplies transverse-shear resultants.
///
/// # Errors
///
/// Returns [`CrashError`] for QBAT/QEPH formulations, incompatible
/// flags, non-positive `dt`, non-finite input, or layer/material count
/// mismatch.
pub fn q4_crash_forces<M: StressUpdateModel<StrainIncrement = [f64; 6]>, H: ThickShellHook>(
    geom: &Q4CrashGeom,
    layers: &mut LayerStates<M::State>,
    material: &M,
    def_shell: &DefShell,
    u: &[f64; 24],
    v: &[f64; 24],
    dt: f64,
    hg: &HourglassParams,
    hook: &H,
) -> Result<Q4ForceOut, CrashError> {
    def_shell.validate()?;
    if !def_shell.formulation.is_one_point() {
        return Err(CrashError::new(format!(
            "one-point kernel has no {:?} variant ({})",
            def_shell.formulation,
            def_shell.formulation.native_status()
        )));
    }
    if dt.is_nan() || dt <= 0.0 {
        return Err(CrashError::new("explicit step dt must be positive"));
    }
    if !u.iter().chain(v.iter()).all(|x| x.is_finite()) {
        return Err(CrashError::new("shell kinematics must be finite"));
    }
    if layers.states.len() != layers.points.len() {
        return Err(CrashError::new("layer states must match thickness points"));
    }
    // Membrane and curvature increments from the center operators.
    let mut d_membrane = [0.0; 3];
    let mut d_curvature = [0.0; 3];
    let mut shear_rate = [0.0; 2];
    for node in 0..4 {
        let disp = [
            v[node * 6] * dt,
            v[node * 6 + 1] * dt,
            v[node * 6 + 2] * dt,
            v[node * 6 + 3] * dt,
            v[node * 6 + 4] * dt,
            v[node * 6 + 5] * dt,
        ];
        for row in 0..3 {
            d_membrane[row] += geom.membrane_b[row * 8 + node * 2] * disp[0]
                + geom.membrane_b[row * 8 + node * 2 + 1] * disp[1];
            d_curvature[row] += geom.bending_b[row * 12 + node * 3] * disp[2]
                + geom.bending_b[row * 12 + node * 3 + 1] * disp[3]
                + geom.bending_b[row * 12 + node * 3 + 2] * disp[4];
        }
        for (slot, brow) in shear_rate.iter_mut().zip(geom.shear_b.chunks(12)) {
            *slot += brow[node * 3] * disp[2] / dt
                + brow[node * 3 + 1] * disp[3] / dt
                + brow[node * 3 + 2] * disp[4] / dt;
        }
    }
    let mut resultants = ShellResultants::default();
    let mut dezz_sum = 0.0;
    let mut plastic_work_rate = 0.0;
    for (slot, point) in layers.points.iter().enumerate() {
        let z = point.z * layers.thickness_scale;
        let (update, dezz) = plane_stress_step(
            material,
            &layers.states[slot],
            &d_membrane,
            &d_curvature,
            z,
            dt,
            def_shell.plasticity,
        )?;
        material.commit(&mut layers.states[slot], &update);
        let w = point.weight * layers.thickness_scale;
        resultants.membrane[0] += w * update.stress[0];
        resultants.membrane[1] += w * update.stress[1];
        resultants.membrane[2] += w * update.stress[3];
        let moment_weight = point.moment_weight * layers.thickness_scale.powi(2);
        resultants.bending[0] += moment_weight * update.stress[0];
        resultants.bending[1] += moment_weight * update.stress[1];
        resultants.bending[2] += moment_weight * update.stress[3];
        dezz_sum += dezz;
        // Plastic dissipation density `q dep/dt` integrated through
        // the thickness; exact for the committed J2 increment.
        let q = von_mises_plane(&update.stress);
        plastic_work_rate += w * q * update.delta_equiv_plastic_strain / dt;
    }
    // Shear resultants from the accumulated shear strain increment.
    for (total, rate) in layers.transverse_shear.iter_mut().zip(shear_rate) {
        *total += rate * dt;
    }
    let shear_q = hook.shear_forces(
        &layers.transverse_shear,
        geom.thickness * layers.thickness_scale,
    );
    resultants.shear = shear_q;
    if def_shell.thickness == ShellThicknessFlag::Update {
        let mean = dezz_sum / layers.points.len() as f64;
        layers.thickness_scale *= 1.0 + mean;
    }
    let mut forces = [0.0; 24];
    for node in 0..4 {
        for row in 0..3 {
            forces[node * 6] +=
                geom.area * geom.membrane_b[row * 8 + node * 2] * resultants.membrane[row];
            forces[node * 6 + 1] +=
                geom.area * geom.membrane_b[row * 8 + node * 2 + 1] * resultants.membrane[row];
            forces[node * 6 + 2] +=
                geom.area * geom.bending_b[row * 12 + node * 3] * resultants.bending[row];
            forces[node * 6 + 3] +=
                geom.area * geom.bending_b[row * 12 + node * 3 + 1] * resultants.bending[row];
            forces[node * 6 + 4] +=
                geom.area * geom.bending_b[row * 12 + node * 3 + 2] * resultants.bending[row];
        }
        for (brow, q) in geom.shear_b.chunks(12).zip(shear_q.iter()) {
            forces[node * 6 + 2] += geom.area * brow[node * 3] * q;
            forces[node * 6 + 3] += geom.area * brow[node * 3 + 1] * q;
            forces[node * 6 + 4] += geom.area * brow[node * 3 + 2] * q;
        }
    }
    // Hourglass forces on the formulation-selected vector.
    let gamma = geom.hg_vector(def_shell.formulation);
    let mut hg_power = 0.0;
    for component in 0..2 {
        let mut modal_v = 0.0;
        let mut modal_u = 0.0;
        for node in 0..4 {
            modal_v += gamma[node] * v[node * 6 + component];
            modal_u += gamma[node] * u[node * 6 + component];
        }
        for node in 0..4 {
            let f = gamma[node] * (hg.membrane_viscous * modal_v + hg.membrane_stiffness * modal_u);
            forces[node * 6 + component] += f;
            hg_power += f * v[node * 6 + component];
        }
    }
    // The normal-displacement checkerboard is also invisible to
    // center membrane, curvature and shear operators. Stabilizing only
    // in-plane translations and nodal rotations leaves this mode free.
    let modal_v: f64 = (0..4).map(|node| gamma[node] * v[node * 6 + 2]).sum();
    let modal_u: f64 = (0..4).map(|node| gamma[node] * u[node * 6 + 2]).sum();
    let modal_force = hg.transverse_viscous * modal_v + hg.transverse_stiffness * modal_u;
    for node in 0..4 {
        let f = gamma[node] * modal_force;
        forces[node * 6 + 2] += f;
        hg_power += f * v[node * 6 + 2];
    }
    for component in 0..2 {
        let mut modal_v = 0.0;
        let mut modal_u = 0.0;
        for node in 0..4 {
            modal_v += gamma[node] * v[node * 6 + 3 + component];
            modal_u += gamma[node] * u[node * 6 + 3 + component];
        }
        for node in 0..4 {
            let m = gamma[node] * (hg.bending_viscous * modal_v + hg.bending_stiffness * modal_u);
            forces[node * 6 + 3 + component] += m;
            hg_power += m * v[node * 6 + 3 + component];
        }
    }
    if hg.drilling > 0.0 {
        for node in 0..4 {
            let m = hg.drilling * u[node * 6 + 5];
            forces[node * 6 + 5] += m;
            hg_power += m * v[node * 6 + 5];
        }
    }
    plastic_work_rate *= geom.area;
    Ok(Q4ForceOut {
        forces,
        resultants,
        hg_power,
        plastic_work_rate,
    })
}

/// Von Mises stress of a plane-stress C1 tensor.
fn von_mises_plane(stress: &[f64; 6]) -> f64 {
    let s0 = stress[0];
    let s1 = stress[1];
    let t = stress[3];
    (s0 * s0 - s0 * s1 + s1 * s1 + 3.0 * t * t).sqrt()
}

/// Centroid T3 crash-shell geometry in its local frame.
#[derive(Debug, Clone)]
pub struct T3CrashGeom {
    /// Element area in m^2.
    pub area: f64,
    /// Shell thickness in m.
    pub thickness: f64,
    /// Global origin of the local frame.
    pub origin: [f64; 3],
    /// Local x axis in global coordinates.
    pub ex: [f64; 3],
    /// Local y axis in global coordinates.
    pub ey: [f64; 3],
    /// Local z axis (normal) in global coordinates.
    pub ez: [f64; 3],
    /// Node coordinates in the local plane.
    pub local: [(f64, f64); 3],
    /// Constant-strain membrane B, 3x6 row-major over `[u, v]` per node.
    pub membrane_b: [f64; 18],
    /// Centroid DKT bending B, 3x9 row-major over `[w, tx, ty]` per node.
    pub bending_b: [f64; 27],
    /// Characteristic length `A / L_max` in m.
    pub char_length: f64,
}

impl T3CrashGeom {
    /// Builds crash geometry from a triangle seed element.
    ///
    /// Reuses the seed frame, coordinates, area, thickness, membrane
    /// operator, and the DKT curvature operator at the centroid
    /// (barycentric `1/3, 1/3`).
    ///
    /// # Examples
    ///
    /// ```rust
    /// use sonium_fem::mesh::Point;
    /// use sonium_fem::physics::solid::shell::ShellElement;
    /// use sonium_fem::physics::solid::shell::ShellSection;
    /// use sonium_fem::physics::solid::shell::crash::T3CrashGeom;
    ///
    /// let section = ShellSection::new(200.0e9, 0.3, 0.002, 7800.0);
    /// let shell = ShellElement::from_points(
    ///     [0, 1, 2],
    ///     &Point::new_3d(0.0, 0.0, 0.0),
    ///     &Point::new_3d(1.0, 0.0, 0.0),
    ///     &Point::new_3d(0.0, 1.0, 0.0),
    ///     section,
    /// );
    /// let geom = T3CrashGeom::from_shell(&shell).unwrap();
    /// assert!((geom.area - 0.5).abs() < 1e-12);
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`CrashError`] for degenerate (zero-area) triangles.
    pub fn from_shell(shell: &ShellElement) -> Result<Self, CrashError> {
        if !shell.area.is_finite() || shell.area <= 0.0 {
            return Err(CrashError::new("triangle area must be positive finite"));
        }
        let third = 1.0 / 3.0;
        let bending_b = dkt_b_matrix_at(&shell.local_coords, shell.area, third, third);
        let mut longest: f64 = 0.0;
        for i in 0..3 {
            let a = shell.local_coords[i];
            let b = shell.local_coords[(i + 1) % 3];
            let edge = ((b.0 - a.0).powi(2) + (b.1 - a.1).powi(2)).sqrt();
            longest = longest.max(edge);
        }
        Ok(Self {
            area: shell.area,
            thickness: shell.section.thickness,
            origin: shell.origin,
            ex: [shell.rotation[0], shell.rotation[3], shell.rotation[6]],
            ey: [shell.rotation[1], shell.rotation[4], shell.rotation[7]],
            ez: [shell.rotation[2], shell.rotation[5], shell.rotation[8]],
            local: shell.local_coords,
            membrane_b: shell.cst_b_matrix(),
            bending_b,
            char_length: shell.area / longest,
        })
    }

    /// Lumped nodal masses and rotary inertias (18-vector).
    #[must_use]
    pub fn lumped_masses(&self, density: f64) -> [f64; 18] {
        let nodal = density * self.area * self.thickness / 3.0;
        let rotary = nodal * self.thickness.powi(2) / 12.0;
        let mut masses = [0.0; 18];
        for node in 0..3 {
            masses[node * 6] = nodal;
            masses[node * 6 + 1] = nodal;
            masses[node * 6 + 2] = nodal;
            masses[node * 6 + 3] = rotary;
            masses[node * 6 + 4] = rotary;
            masses[node * 6 + 5] = rotary;
        }
        masses
    }

    /// Critical explicit step `L / c` in s.
    #[must_use]
    pub fn critical_step(&self, sound_speed: f64) -> f64 {
        self.char_length / sound_speed
    }
}

/// Internal-force evaluation of one T3 crash shell.
#[derive(Debug, Clone, Copy)]
pub struct T3ForceOut {
    /// Nodal forces/moments in N and N m (local, 18-vector).
    pub forces: [f64; 18],
    /// Resultants at the centroid point.
    pub resultants: ShellResultants,
    /// Plastic work rate in W.
    pub plastic_work_rate: f64,
}

/// Explicit internal forces of one T3 crash shell.
///
/// Advances every layer C1 material by its plane-stress increment from
/// the centroid membrane/curvature rates over `dt`, commits each
/// layer, and returns `A (Bm^T N + Bb^T M)`. Fully integrated: no
/// hourglass terms. `u`/`v` are local 18-vectors per node `[u, v, w,
/// tx, ty, tz]`. DKT18/DKT_S3 triangle flags fail (they need
/// membrane-drilling coupling outside the centroid kernel); both C0
/// flags run this kernel, with large rotations handled by the
/// caller's corotational frame update.
///
/// # Errors
///
/// Returns [`CrashError`] for DKT18/DKT_S3 flags, non-positive `dt`,
/// non-finite input, or layer/material count mismatch.
pub fn t3_crash_forces<M: StressUpdateModel<StrainIncrement = [f64; 6]>>(
    geom: &T3CrashGeom,
    layers: &mut LayerStates<M::State>,
    material: &M,
    def_shell: &DefShell,
    u: &[f64; 18],
    v: &[f64; 18],
    dt: f64,
    drill_stiffness: f64,
) -> Result<T3ForceOut, CrashError> {
    def_shell.validate()?;
    if !def_shell.tri.is_centroid() {
        return Err(CrashError::new(format!(
            "centroid kernel has no {:?} variant ({})",
            def_shell.tri,
            def_shell.tri.native_status()
        )));
    }
    if dt.is_nan() || dt <= 0.0 {
        return Err(CrashError::new("explicit step dt must be positive"));
    }
    if !u.iter().chain(v.iter()).all(|x| x.is_finite()) {
        return Err(CrashError::new("shell kinematics must be finite"));
    }
    if layers.states.len() != layers.points.len() {
        return Err(CrashError::new("layer states must match thickness points"));
    }
    let mut d_membrane = [0.0; 3];
    let mut d_curvature = [0.0; 3];
    for node in 0..3 {
        let disp = [
            v[node * 6] * dt,
            v[node * 6 + 1] * dt,
            v[node * 6 + 2] * dt,
            v[node * 6 + 3] * dt,
            v[node * 6 + 4] * dt,
        ];
        for row in 0..3 {
            d_membrane[row] += geom.membrane_b[row * 6 + node * 2] * disp[0]
                + geom.membrane_b[row * 6 + node * 2 + 1] * disp[1];
            d_curvature[row] += geom.bending_b[row * 9 + node * 3] * disp[2]
                + geom.bending_b[row * 9 + node * 3 + 1] * disp[3]
                + geom.bending_b[row * 9 + node * 3 + 2] * disp[4];
        }
    }
    let mut resultants = ShellResultants::default();
    let mut dezz_sum = 0.0;
    let mut plastic_work_rate = 0.0;
    for (slot, point) in layers.points.iter().enumerate() {
        let z = point.z * layers.thickness_scale;
        let (update, dezz) = plane_stress_step(
            material,
            &layers.states[slot],
            &d_membrane,
            &d_curvature,
            z,
            dt,
            def_shell.plasticity,
        )?;
        material.commit(&mut layers.states[slot], &update);
        let w = point.weight * layers.thickness_scale;
        resultants.membrane[0] += w * update.stress[0];
        resultants.membrane[1] += w * update.stress[1];
        resultants.membrane[2] += w * update.stress[3];
        let moment_weight = point.moment_weight * layers.thickness_scale.powi(2);
        resultants.bending[0] += moment_weight * update.stress[0];
        resultants.bending[1] += moment_weight * update.stress[1];
        resultants.bending[2] += moment_weight * update.stress[3];
        dezz_sum += dezz;
        let q = von_mises_plane(&update.stress);
        plastic_work_rate += w * q * update.delta_equiv_plastic_strain / dt;
    }
    if def_shell.thickness == ShellThicknessFlag::Update {
        let mean = dezz_sum / layers.points.len() as f64;
        layers.thickness_scale *= 1.0 + mean;
    }
    let mut forces = [0.0; 18];
    for node in 0..3 {
        for row in 0..3 {
            forces[node * 6] +=
                geom.area * geom.membrane_b[row * 6 + node * 2] * resultants.membrane[row];
            forces[node * 6 + 1] +=
                geom.area * geom.membrane_b[row * 6 + node * 2 + 1] * resultants.membrane[row];
            forces[node * 6 + 2] +=
                geom.area * geom.bending_b[row * 9 + node * 3] * resultants.bending[row];
            forces[node * 6 + 3] +=
                geom.area * geom.bending_b[row * 9 + node * 3 + 1] * resultants.bending[row];
            forces[node * 6 + 4] +=
                geom.area * geom.bending_b[row * 9 + node * 3 + 2] * resultants.bending[row];
        }
        forces[node * 6 + 5] += drill_stiffness * u[node * 6 + 5];
    }
    plastic_work_rate *= geom.area;
    Ok(T3ForceOut {
        forces,
        resultants,
        plastic_work_rate,
    })
}

/// Vector difference of two 3-vectors.
fn sub3(a: &[f64; 3], b: &[f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

/// Euclidean norm of a 3-vector.
fn norm3(a: &[f64; 3]) -> f64 {
    (a[0] * a[0] + a[1] * a[1] + a[2] * a[2]).sqrt()
}

/// Dot product of two 3-vectors.
fn dot3(a: &[f64; 3], b: &[f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// Cross product of two 3-vectors.
fn cross3(a: &[f64; 3], b: &[f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh::Point;
    use crate::physics::solid::plasticity::{HardeningLaw, J2Material, PlasticState};
    use crate::physics::solid::shell::ShellSection;

    fn elastic_j2() -> J2Material {
        J2Material::new(200.0e9, 0.3, 1.0e12, HardeningLaw::Perfect)
    }

    fn unit_square() -> Q4CrashGeom {
        Q4CrashGeom::from_corners(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            0.002,
        )
        .unwrap()
    }

    fn unit_triangle() -> T3CrashGeom {
        let section = ShellSection::new(200.0e9, 0.3, 0.002, 7800.0);
        let shell = ShellElement::from_points(
            [0, 1, 2],
            &Point::new_3d(0.0, 0.0, 0.0),
            &Point::new_3d(1.0, 0.0, 0.0),
            &Point::new_3d(0.0, 1.0, 0.0),
            section,
        );
        T3CrashGeom::from_shell(&shell).unwrap()
    }

    #[test]
    fn def_shell_flags_parse_and_validate() {
        let def = DefShell::from_flags(0, 0, 0, 0, 0, 0).unwrap();
        assert_eq!(def, DefShell::explicit_defaults());
        assert!(def.validate().is_ok());
        let bad = DefShell::from_flags(1, 3, 2, 2, 2, 2).unwrap();
        assert!(bad.validate().is_err());
        let ok = DefShell::from_flags(2, 3, 2, 2, 2, 2).unwrap();
        assert!(ok.validate().is_ok());
        assert!(DefShell::from_flags(7, 2, 2, 2, 2, 2).is_err());
        assert!(DefShell::from_flags(1, 2, 2, 2, 29, 2).is_err());
        assert!(!ShellFormulation::Qbat.is_one_point());
        assert!(!TriShellFormulation::Dkt18.is_centroid());
    }

    #[test]
    fn surface_sampled_layers_reproduce_elastic_membrane_and_bending() {
        let geom = unit_square();
        let material = elastic_j2();
        let def = DefShell::explicit_defaults();
        let dt = 1e-3;
        let strain = 1e-5;
        let curvature = 0.01;
        let mut velocity = [0.; 24];
        for n in 0..4 {
            velocity[6 * n] = strain * geom.local[n].0 / dt;
            velocity[6 * n + 4] = curvature * geom.local[n].0 / dt;
        }
        for count in 2..=11 {
            for scale in [1., 1.3] {
                let mut layers = LayerStates::new_piecewise_linear(
                    PlasticState::new_3d(),
                    count,
                    geom.thickness,
                )
                .unwrap();
                layers.thickness_scale = scale;
                let out = q4_crash_forces(
                    &geom,
                    &mut layers,
                    &material,
                    &def,
                    &[0.; 24],
                    &velocity,
                    dt,
                    &HourglassParams::default(),
                    &NoThickShell,
                )
                .unwrap();
                let t = geom.thickness * scale;
                let modulus = 200e9 / (1. - 0.3_f64.powi(2));
                let membrane = modulus * t * strain;
                let bending = modulus * t.powi(3) * curvature / 12.;
                assert!((out.resultants.membrane[0] / membrane - 1.).abs() < 1e-10);
                assert!((out.resultants.bending[0] / bending - 1.).abs() < 1e-10);
                assert!((out.resultants.membrane[1] / membrane - 0.3).abs() < 1e-10);
                assert!((out.resultants.bending[1] / bending - 0.3).abs() < 1e-10);
            }
        }
        for count in [0, 1, 12] {
            assert!(linear_thickness_points(count, 0.002).is_err());
        }
        assert!(linear_thickness_points(3, f64::NAN).is_err());
    }

    #[test]
    fn square_center_operators_are_exact() {
        let geom = unit_square();
        assert!((geom.area - 1.0).abs() < 1e-14);
        // dN/dx = [-0.5, 0.5, 0.5, -0.5], dN/dy = [-0.5, -0.5, 0.5, 0.5].
        let row0: Vec<f64> = geom.membrane_b[0..8].to_vec();
        assert_eq!(row0, vec![-0.5, 0.0, 0.5, 0.0, 0.5, 0.0, -0.5, 0.0]);
        let row1: Vec<f64> = geom.membrane_b[8..16].to_vec();
        assert_eq!(row1, vec![0.0, -0.5, 0.0, -0.5, 0.0, 0.5, 0.0, 0.5]);
        assert_eq!(geom.gamma, [1.0, -1.0, 1.0, -1.0]);
        assert_eq!(
            geom.hg_vector(ShellFormulation::Hallquist),
            [1.0, -1.0, 1.0, -1.0]
        );
        assert_eq!(
            geom.hg_vector(ShellFormulation::WarpOrthogonal),
            [1.0, -1.0, 1.0, -1.0]
        );
    }

    #[test]
    fn warped_quad_frame_preserves_geometry_and_vector_work() {
        // Bilinear saddle: its center tangent plane is exactly xy even
        // though every edge has a nonzero transverse component.
        let corners = [
            [-1., -1., 0.4],
            [1., -1., -0.4],
            [1., 1., 0.4],
            [-1., 1., -0.4],
        ];
        // Also rotate the saddle about two axes so the test cannot rely
        // on zero components in a global coordinate plane.
        for rotated in [false, true] {
            let transform = |p: [f64; 3]| {
                if !rotated {
                    return p;
                }
                let (sy, cy) = 0.37_f64.sin_cos();
                let (sz, cz) = 0.61_f64.sin_cos();
                let x = cy * p[0] + sy * p[2];
                let z = -sy * p[0] + cy * p[2];
                [cz * x - sz * p[1], sz * x + cz * p[1], z]
            };
            let geom = Q4CrashGeom::from_corners(corners.map(transform), 0.002).unwrap();
            let axes = [geom.ex, geom.ey, geom.ez];
            for i in 0..3 {
                for j in 0..3 {
                    let expected = if i == j { 1. } else { 0. };
                    assert!((dot3(&axes[i], &axes[j]) - expected).abs() < 1e-14);
                }
            }
            assert!((geom.area - 4.).abs() < 1e-14);
            for (actual, expected) in
                geom.local
                    .iter()
                    .zip([(0., 0.), (2., 0.), (2., 2.), (0., 2.)])
            {
                assert!((actual.0 - expected.0).abs() < 1e-14);
                assert!((actual.1 - expected.1).abs() < 1e-14);
            }
            let velocity = [0.6, -1.3, 2.1];
            let force = [-0.4, 3.2, 0.7];
            let local_v = axes.map(|axis| dot3(&axis, &velocity));
            let local_f = axes.map(|axis| dot3(&axis, &force));
            assert!((dot3(&local_f, &local_v) - dot3(&force, &velocity)).abs() < 1e-14);
        }
    }

    #[test]
    fn gamma_stays_orthogonal_on_distortion() {
        let geom = Q4CrashGeom::from_corners(
            [
                [0.0, 0.0, 0.0],
                [1.2, 0.1, 0.0],
                [1.0, 1.1, 0.0],
                [0.1, 0.9, 0.0],
            ],
            0.002,
        )
        .unwrap();
        let mut dots = [0.0; 3];
        for i in 0..4 {
            dots[0] += geom.gamma[i];
            dots[1] += geom.gamma[i] * geom.local[i].0;
            dots[2] += geom.gamma[i] * geom.local[i].1;
        }
        assert!(dots.iter().all(|d| d.abs() < 1e-12));
    }

    #[test]
    fn warp_flag_uses_projected_vector() {
        // Center warp slopes vanish identically for normal-consistent
        // frames, so Ishell 4 runs the projected vector (documented on
        // `gamma`); orthogonality still holds on the warped quad.
        use crate::physics::solid::shell::QuadShellElement;
        let section = ShellSection::new(200.0e9, 0.3, 0.002, 7800.0);
        let quad = QuadShellElement::from_points_with_section(
            [0, 1, 2, 3],
            &Point::new_3d(0.0, 0.0, 0.0),
            &Point::new_3d(1.0, 0.0, 0.1),
            &Point::new_3d(1.0, 1.0, 0.0),
            &Point::new_3d(0.0, 1.0, 0.0),
            section,
        );
        let geom = Q4CrashGeom::from_quad(&quad).unwrap();
        assert!(geom.warp.iter().any(|w| w.abs() > 1e-6));
        assert_eq!(geom.hg_vector(ShellFormulation::WarpOrthogonal), geom.gamma);
        let mut dots = [0.0; 3];
        for i in 0..4 {
            dots[0] += geom.gamma[i];
            dots[1] += geom.gamma[i] * geom.local[i].0;
            dots[2] += geom.gamma[i] * geom.local[i].1;
        }
        assert!(dots.iter().all(|d| d.abs() < 1e-9));
    }

    #[test]
    fn plane_stress_recovers_biaxial_law() {
        let material = elastic_j2();
        let state = PlasticState::new_3d();
        // Uniaxial membrane strain exx = 1e-3: plane stress gives
        // sxx = E e / (1 - nu^2), syy = nu E e / (1 - nu^2).
        let (update, dezz) = plane_stress_step(
            &material,
            &state,
            &[1.0e-3, 0.0, 0.0],
            &[0.0; 3],
            0.0,
            1.0e-3,
            ShellPlasticityFlag::RadialReturn,
        )
        .unwrap();
        let e = 200.0e9;
        let nu = 0.3;
        assert!((update.stress[0] - e * 1.0e-3 / (1.0 - nu * nu)).abs() < 1.0e4);
        assert!((update.stress[1] - nu * e * 1.0e-3 / (1.0 - nu * nu)).abs() < 1.0e4);
        assert!(update.stress[2].abs() < 1.0);
        assert!((dezz + nu / (1.0 - nu) * 1.0e-3).abs() < 1.0e-12);
        // Iterative flag converges the same value in 3 passes for linear
        // elasticity (Newton on a linear residual finishes in 2).
        let (iter, _) = plane_stress_step(
            &material,
            &state,
            &[1.0e-3, 0.0, 0.0],
            &[0.0; 3],
            0.0,
            1.0e-3,
            ShellPlasticityFlag::Iterative,
        )
        .unwrap();
        assert!((iter.stress[0] - update.stress[0]).abs() < 1.0);
    }

    #[test]
    fn q4_patch_is_self_equilibrated() {
        let geom = unit_square();
        let material = elastic_j2();
        let mut layers = LayerStates::new(PlasticState::new_3d(), 5, geom.thickness).unwrap();
        let def = DefShell::explicit_defaults();
        let hg = HourglassParams::for_shell(
            def.formulation,
            7800.0,
            4500.0,
            200.0e9 / 2.6,
            geom.area,
            geom.thickness,
            geom.char_length,
            false,
        )
        .unwrap();
        // Uniaxial membrane stretch at 1/s.
        let mut v = [0.0; 24];
        for node in 0..4 {
            v[node * 6] = geom.local[node].0;
        }
        let out = q4_crash_forces(
            &geom,
            &mut layers,
            &material,
            &def,
            &[0.0; 24],
            &v,
            1.0e-3,
            &hg,
            &NoThickShell,
        )
        .unwrap();
        // Linear field: hourglass power ~zero, resultant matches
        // Nxx = t E e / (1 - nu^2) with e = 1e-3.
        assert!(out.hg_power.abs() < 1e-6);
        let expected = 0.002 * 200.0e9 * 1.0e-3 / (1.0 - 0.09);
        assert!((out.resultants.membrane[0] - expected).abs() < 1e-3 * expected);
        assert!(out.resultants.bending.iter().all(|m| m.abs() < 1.0));
        let mut force = [0.0; 3];
        for node in 0..4 {
            force[0] += out.forces[node * 6];
            force[1] += out.forces[node * 6 + 1];
            force[2] += out.forces[node * 6 + 2];
        }
        let scale: f64 = out.forces.iter().map(|f| f.abs()).sum::<f64>().max(1.0e-30);
        for component in force.iter() {
            assert!(component.abs() / scale < 1.0e-12);
        }
    }

    #[test]
    fn q4_rigid_translation_gives_zero_forces() {
        let geom = unit_square();
        let material = elastic_j2();
        let mut layers = LayerStates::new(PlasticState::new_3d(), 2, geom.thickness).unwrap();
        let def = DefShell::explicit_defaults();
        let hg = HourglassParams::for_shell(
            def.formulation,
            7800.0,
            4500.0,
            200.0e9 / 2.6,
            geom.area,
            geom.thickness,
            geom.char_length,
            false,
        )
        .unwrap();
        let mut v = [0.0; 24];
        for node in 0..4 {
            v[node * 6] = 2.0;
            v[node * 6 + 1] = -1.0;
            v[node * 6 + 2] = 0.5;
        }
        let out = q4_crash_forces(
            &geom,
            &mut layers,
            &material,
            &def,
            &[0.0; 24],
            &v,
            1.0e-3,
            &hg,
            &NoThickShell,
        )
        .unwrap();
        assert!(out.forces.iter().all(|f| f.abs() < 1e-6));
        assert_eq!(out.hg_power, 0.0);
    }

    #[test]
    fn q4_transverse_shear_restores_during_hold_and_unloads() {
        let geom = unit_square();
        let material = elastic_j2();
        let def = DefShell::explicit_defaults();
        let hook = ElasticShearHook::new(200e9 / 2.6);
        let hg = HourglassParams {
            membrane_viscous: 0.0,
            membrane_stiffness: 0.0,
            bending_viscous: 0.0,
            bending_stiffness: 0.0,
            drilling: 0.0,
            ..HourglassParams::default()
        };
        let mut layers = LayerStates::new(PlasticState::new_3d(), 3, geom.thickness).unwrap();
        let mut velocity = [0.0; 24];
        for node in 0..4 {
            velocity[6 * node + 2] = geom.local[node].0;
        }
        let loaded = q4_crash_forces(
            &geom,
            &mut layers,
            &material,
            &def,
            &[0.0; 24],
            &velocity,
            0.001,
            &hg,
            &hook,
        )
        .unwrap();
        let expected = (5.0 / 6.0) * (200e9 / 2.6) * geom.thickness * 0.001;
        assert!((loaded.resultants.shear[0] - expected).abs() / expected < 1e-12);
        let held = q4_crash_forces(
            &geom,
            &mut layers,
            &material,
            &def,
            &[0.0; 24],
            &[0.0; 24],
            0.01,
            &hg,
            &hook,
        )
        .unwrap();
        assert_eq!(held.resultants.shear, loaded.resultants.shear);
        assert!(held.forces.iter().any(|f| f.abs() > 1.0));
        for v in &mut velocity {
            *v = -*v;
        }
        let unloaded = q4_crash_forces(
            &geom,
            &mut layers,
            &material,
            &def,
            &[0.0; 24],
            &velocity,
            0.001,
            &hg,
            &hook,
        )
        .unwrap();
        assert!(unloaded.resultants.shear.iter().all(|q| q.abs() < 1e-8));

        let mut split = LayerStates::new(PlasticState::new_3d(), 3, geom.thickness).unwrap();
        for v in &mut velocity {
            *v = -*v;
        }
        let mut last = None;
        for _ in 0..10 {
            last = Some(
                q4_crash_forces(
                    &geom, &mut split, &material, &def, &[0.0; 24], &velocity, 0.0001, &hg, &hook,
                )
                .unwrap(),
            );
        }
        assert!((last.unwrap().resultants.shear[0] - expected).abs() / expected < 1e-12);
    }

    #[test]
    fn q4_hourglass_dissipates_on_spurious_mode() {
        let geom = unit_square();
        let material = elastic_j2();
        let mut layers = LayerStates::new(PlasticState::new_3d(), 2, geom.thickness).unwrap();
        let def = DefShell::explicit_defaults();
        let hg = HourglassParams::for_shell(
            def.formulation,
            7800.0,
            4500.0,
            200.0e9 / 2.6,
            geom.area,
            geom.thickness,
            geom.char_length,
            false,
        )
        .unwrap();
        let mut v = [0.0; 24];
        for node in 0..4 {
            v[node * 6] = QUAD_HG[node];
        }
        let out = q4_crash_forces(
            &geom,
            &mut layers,
            &material,
            &def,
            &[0.0; 24],
            &v,
            1.0e-3,
            &hg,
            &NoThickShell,
        )
        .unwrap();
        assert!(out.hg_power > 0.0);
        // P = c (gamma . v)^2 = c * 16 on the square.
        assert!((out.hg_power - hg.membrane_viscous * 16.0).abs() / out.hg_power < 1e-9);
    }

    #[test]
    fn q4_transverse_hourglass_decays_and_preserves_rigid_rotation() {
        let geom = unit_square();
        let material = elastic_j2();
        let def = DefShell::explicit_defaults();
        let hg = HourglassParams::for_shell(
            def.formulation,
            7800.,
            4500.,
            200e9 / 2.6,
            geom.area,
            geom.thickness,
            geom.char_length,
            false,
        )
        .unwrap();
        assert!(hg.transverse_viscous > 0.);
        let mass = geom.lumped_masses(7800.)[2];
        // The four-node normal checkerboard has modal mass m/4.
        // Compare explicit evolution against its independent damped
        // oscillator solution q(t) = q(0) exp(-4 c t/m).
        let decay_rate = 4. * hg.transverse_viscous / mass;
        let steps = 200;
        let duration = 0.1 / decay_rate;
        let dt = duration / steps as f64;
        let mut v = [0.; 24];
        for n in 0..4 {
            v[6 * n + 2] = QUAD_HG[n];
        }
        let mut layers = LayerStates::new(PlasticState::new_3d(), 3, geom.thickness).unwrap();
        for _ in 0..steps {
            let out = q4_crash_forces(
                &geom,
                &mut layers,
                &material,
                &def,
                &[0.; 24],
                &v,
                dt,
                &hg,
                &NoThickShell,
            )
            .unwrap();
            assert!(out.hg_power > 0.);
            assert!(
                out.resultants
                    .membrane
                    .iter()
                    .chain(&out.resultants.bending)
                    .chain(&out.resultants.shear)
                    .all(|f| f.abs() < 1e-12)
            );
            for n in 0..4 {
                v[6 * n + 2] -= dt * out.forces[6 * n + 2] / mass;
            }
        }
        let expected = (-decay_rate * duration).exp();
        for n in 0..4 {
            assert!((v[6 * n + 2] - QUAD_HG[n] * expected).abs() < 5e-5);
        }
        // Arbitrary affine normal velocity with its matching angular
        // velocity is a rigid rotation plus translation, not a mode
        // that stabilization may resist.
        for n in 0..4 {
            let (x, y) = geom.local[n];
            v[6 * n + 2] = 0.7 + 0.3 * x - 0.2 * y;
            v[6 * n + 3] = -0.2;
            v[6 * n + 4] = -0.3;
        }
        let mut virgin = LayerStates::new(PlasticState::new_3d(), 3, geom.thickness).unwrap();
        let out = q4_crash_forces(
            &geom,
            &mut virgin,
            &material,
            &def,
            &[0.; 24],
            &v,
            dt,
            &hg,
            &NoThickShell,
        )
        .unwrap();
        assert!(out.forces.iter().all(|f| f.abs() < 1e-10));
        assert!(out.hg_power.abs() < 1e-12);
    }

    #[test]
    fn q4_thickness_update_evolves_scale() {
        let geom = unit_square();
        let material = elastic_j2();
        let mut def = DefShell::explicit_defaults();
        def.thickness = ShellThicknessFlag::Update;
        let hg = HourglassParams::for_shell(
            def.formulation,
            7800.0,
            4500.0,
            200.0e9 / 2.6,
            geom.area,
            geom.thickness,
            geom.char_length,
            false,
        )
        .unwrap();
        let mut layers = LayerStates::new(PlasticState::new_3d(), 3, geom.thickness).unwrap();
        let mut v = [0.0; 24];
        for node in 0..4 {
            v[node * 6] = geom.local[node].0;
        }
        q4_crash_forces(
            &geom,
            &mut layers,
            &material,
            &def,
            &[0.0; 24],
            &v,
            1.0e-3,
            &hg,
            &NoThickShell,
        )
        .unwrap();
        // Tensile membrane strain thins the shell.
        assert!(layers.thickness_scale < 1.0);
        let mut frozen = LayerStates::new(PlasticState::new_3d(), 3, geom.thickness).unwrap();
        let constant = DefShell::explicit_defaults();
        q4_crash_forces(
            &geom,
            &mut frozen,
            &material,
            &constant,
            &[0.0; 24],
            &v,
            1.0e-3,
            &hg,
            &NoThickShell,
        )
        .unwrap();
        assert_eq!(frozen.thickness_scale, 1.0);
    }

    #[test]
    fn q4_rejects_unsupported_formulations() {
        let geom = unit_square();
        let material = elastic_j2();
        let mut layers = LayerStates::new(PlasticState::new_3d(), 2, geom.thickness).unwrap();
        let mut def = DefShell::explicit_defaults();
        def.formulation = ShellFormulation::Qbat;
        let hg = HourglassParams {
            membrane_viscous: 1.0,
            membrane_stiffness: 0.0,
            bending_viscous: 1.0,
            bending_stiffness: 0.0,
            drilling: 0.0,
            ..HourglassParams::default()
        };
        let out = q4_crash_forces(
            &geom,
            &mut layers,
            &material,
            &def,
            &[0.0; 24],
            &[0.0; 24],
            1.0e-3,
            &hg,
            &NoThickShell,
        );
        assert!(out.is_err());
        assert!(
            HourglassParams::for_shell(
                ShellFormulation::Qeph,
                7800.0,
                4500.0,
                80.0e9,
                1.0,
                0.002,
                1.0,
                false
            )
            .is_err()
        );
    }

    #[test]
    fn thick_shell_hook_paths() {
        let none = NoThickShell;
        assert_eq!(none.shear_forces(&[0.01, 0.02], 0.002), [0.0, 0.0]);
        let elastic = ElasticShearHook::new(80.0e9);
        let q = elastic.shear_forces(&[0.01, 0.0], 0.002);
        assert!((q[0] - 5.0 / 6.0 * 80.0e9 * 0.002 * 0.01).abs() < 1.0);
        // Hook shear enters the Q4 forces through Bs^T Q.
        let geom = unit_square();
        let material = elastic_j2();
        let mut layers = LayerStates::new(PlasticState::new_3d(), 2, geom.thickness).unwrap();
        let def = DefShell::explicit_defaults();
        let hg = HourglassParams::for_shell(
            def.formulation,
            7800.0,
            4500.0,
            80.0e9,
            geom.area,
            geom.thickness,
            geom.char_length,
            false,
        )
        .unwrap();
        let mut v = [0.0; 24];
        for node in 0..4 {
            v[node * 6 + 4] = 1.0;
        }
        let out = q4_crash_forces(
            &geom,
            &mut layers,
            &material,
            &def,
            &[0.0; 24],
            &v,
            1.0e-3,
            &hg,
            &elastic,
        )
        .unwrap();
        assert!(out.resultants.shear[0].abs() > 0.0);
    }

    #[test]
    fn t3_patch_is_self_equilibrated() {
        let geom = unit_triangle();
        let material = elastic_j2();
        let mut layers = LayerStates::new(PlasticState::new_3d(), 3, geom.thickness).unwrap();
        let def = DefShell::explicit_defaults();
        let mut v = [0.0; 18];
        for node in 0..3 {
            v[node * 6] = geom.local[node].0;
        }
        let out = t3_crash_forces(
            &geom,
            &mut layers,
            &material,
            &def,
            &[0.0; 18],
            &v,
            1.0e-3,
            0.0,
        )
        .unwrap();
        let expected = 0.002 * 200.0e9 * 1.0e-3 / (1.0 - 0.09);
        assert!((out.resultants.membrane[0] - expected).abs() < 1e-3 * expected);
        let mut force = [0.0; 3];
        for node in 0..3 {
            force[0] += out.forces[node * 6];
            force[1] += out.forces[node * 6 + 1];
            force[2] += out.forces[node * 6 + 2];
        }
        let scale: f64 = out.forces.iter().map(|f| f.abs()).sum::<f64>().max(1.0e-30);
        for component in force.iter() {
            assert!(component.abs() / scale < 1.0e-12);
        }
    }

    #[test]
    fn t3_rejects_dkt_variants() {
        let geom = unit_triangle();
        let material = elastic_j2();
        let mut layers = LayerStates::new(PlasticState::new_3d(), 2, geom.thickness).unwrap();
        let mut def = DefShell::explicit_defaults();
        def.tri = TriShellFormulation::Dkt18;
        assert!(
            t3_crash_forces(
                &geom,
                &mut layers,
                &material,
                &def,
                &[0.0; 18],
                &[0.0; 18],
                1.0e-3,
                0.0
            )
            .is_err()
        );
    }

    #[test]
    fn lumped_mass_and_step_match_closed_form() {
        let geom = unit_square();
        let masses = geom.lumped_masses(7800.0);
        let transl: f64 = [0, 1, 2, 6, 7, 8, 12, 13, 14, 18, 19, 20]
            .iter()
            .map(|i| masses[*i])
            .sum();
        assert!((transl - 3.0 * 7800.0 * 0.002).abs() < 1e-12);
        assert!((geom.char_length - 1.0).abs() < 1e-14);
        assert!((geom.critical_step(4500.0) - 1.0 / 4500.0).abs() < 1e-18);
        let tri = unit_triangle();
        assert!((tri.char_length - 0.5 / 2.0_f64.sqrt()).abs() < 1e-14);
    }

    #[test]
    fn native_status_pins_every_flag() {
        use ShellFormulation as Q;
        for flag in [
            Q::BelytschkoTsay,
            Q::Hallquist,
            Q::ElastoPlasticHg,
            Q::WarpOrthogonal,
        ] {
            assert!(flag.is_one_point());
            assert!(flag.native_status().is_implemented());
        }
        for flag in [Q::Qbat, Q::Qeph] {
            assert!(!flag.is_one_point());
            let status = flag.native_status();
            assert!(!status.is_implemented());
            assert!(matches!(
                status,
                NativeStatus::Deferred {
                    since: "2026-10-12",
                    ..
                }
            ));
            assert!(status.to_string().starts_with("deferred since 2026-10-12"));
        }
        use TriShellFormulation as T;
        for flag in [T::C0, T::C0LargeRotation] {
            assert!(flag.is_centroid());
            assert!(flag.native_status().is_implemented());
        }
        for flag in [T::Dkt18, T::DktS3] {
            assert!(!flag.is_centroid());
            assert!(matches!(
                flag.native_status(),
                NativeStatus::Deferred {
                    since: "2026-10-12",
                    ..
                }
            ));
        }
        assert_eq!(
            Q::BelytschkoTsay.native_status().to_string(),
            "implemented (one-point Q4 crash kernel)"
        );
    }

    #[test]
    fn rejections_carry_verdict_reasons() {
        let quad = unit_square();
        let tri = unit_triangle();
        let material = elastic_j2();
        let mut quad_layers = LayerStates::new(PlasticState::new_3d(), 2, quad.thickness).unwrap();
        let mut tri_layers = LayerStates::new(PlasticState::new_3d(), 2, tri.thickness).unwrap();
        let hg = HourglassParams {
            membrane_viscous: 1.0,
            membrane_stiffness: 0.0,
            bending_viscous: 1.0,
            bending_stiffness: 0.0,
            drilling: 0.0,
            ..HourglassParams::default()
        };
        let mut def = DefShell::explicit_defaults();
        def.formulation = ShellFormulation::Qeph;
        let quad_err = q4_crash_forces(
            &quad,
            &mut quad_layers,
            &material,
            &def,
            &[0.0; 24],
            &[0.0; 24],
            1.0e-3,
            &hg,
            &NoThickShell,
        )
        .unwrap_err();
        assert!(quad_err.message.contains("deferred since 2026-10-12"));
        assert!(quad_err.message.contains("Qeph"));
        def.formulation = ShellFormulation::BelytschkoTsay;
        def.tri = TriShellFormulation::DktS3;
        let tri_err = t3_crash_forces(
            &tri,
            &mut tri_layers,
            &material,
            &def,
            &[0.0; 18],
            &[0.0; 18],
            1.0e-3,
            0.0,
        )
        .unwrap_err();
        assert!(tri_err.message.contains("deferred since 2026-10-12"));
        assert!(tri_err.message.contains("DktS3"));
    }

    #[test]
    fn thick_shell_verdict_is_deferred() {
        assert!(!THICK_SHELL_VERDICT.is_implemented());
        assert!(matches!(
            THICK_SHELL_VERDICT,
            NativeStatus::Deferred {
                since: "2026-10-12",
                ..
            }
        ));
        // The thin-shell hooks keep their behavior under the verdict.
        assert_eq!(NoThickShell.shear_forces(&[0.5, -0.25], 0.01), [0.0, 0.0]);
    }

    #[test]
    fn thickness_tables_sum_to_thickness() {
        for n in 1..=5 {
            let points = thickness_points(n, 0.003).unwrap();
            assert_eq!(points.len(), n);
            let total: f64 = points.iter().map(|p| p.weight).sum();
            assert!((total - 0.003).abs() < 1e-15);
            // Symmetric nodes integrate z exactly to zero.
            let moment: f64 = points.iter().map(|p| p.weight * p.z).sum();
            assert!(moment.abs() < 1e-18);
        }
        assert!(thickness_points(0, 0.002).is_err());
        assert!(thickness_points(6, 0.002).is_err());
    }
}
