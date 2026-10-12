//! Native explicit evolution driver for foam material laws.
//!
//! Replays prescribed strain paths through any explicit material point:
//! the driver substeps each stamp interval, advances the law, and books
//! the stress work, plastic flow, and density stamp by stamp. Two path
//! shapes exist: fully prescribed 6-component histories
//! ([`replay_prescribed`]) and axial histories with a solved lateral
//! response ([`replay_uniaxial`]). Corotational helpers
//! ([`polar_decomposition`], [`rotate_voigt6`], [`Corotator`]) support
//! large-rotation replays where the law runs in a rotated frame.
//!
//! Laws plug in through [`ExplicitPoint`]: foam laws run behind
//! [`PointRunner`], which steps through the bundled
//! [`PointStepper`](super::law_foams::PointStepper) advance (trial,
//! commit, and totals), while plain C1 metals run behind [`C1Runner`].
//! Callers needing custom stepping implement [`ExplicitPoint`] directly.
//!
//! # Examples
//!
//! ```rust
//! use sonium_fem::physics::solid::plasticity::foam_explicit::{
//!     C1Runner, PathStamp, replay_prescribed,
//! };
//! use sonium_fem::physics::solid::plasticity::{J2Material, PlasticState};
//!
//! let law = J2Material::mild_steel();
//! let mut point = C1Runner::new(law, PlasticState::new_3d());
//! let path = [
//!     PathStamp::strain(0.0, [0.0; 6]),
//!     PathStamp::strain(1.0, [0.001, -0.0003, -0.0003, 0.0, 0.0, 0.0]),
//! ];
//! let history = replay_prescribed(&mut point, &path, 4, 7800.0);
//! assert_eq!(history.time.len(), 2);
//! assert!(history.energy_density[1] > 0.0);
//! ```
//!
//! # Panics
//!
//! Drivers panic on empty paths, non-positive substep counts, and
//! non-increasing stamp times. These are caller contract violations,
//! not runtime conditions.
//!
//! # Safety
//!
//! Pure safe Rust with no unsafe code.

// Rust guideline compliant 2026-02-21

use super::law_foams::{PointStepper, jacobi_eigen_c1};
use super::update::{StressUpdate, StressUpdateModel};

/// Accepted material-point step in C1 Voigt order.
#[derive(Debug, Clone, Copy)]
pub struct PointStep {
    /// Cauchy stress `[xx, yy, zz, xy, yz, zx]` (Pa).
    pub stress: [f64; 6],
    /// Equivalent plastic strain increment of the step.
    pub delta_equiv_plastic_strain: f64,
    /// True when the step ended on the yield surface.
    pub yielding: bool,
}

impl From<&StressUpdate> for PointStep {
    /// Converts a C1 update into the driver step shape.
    fn from(update: &StressUpdate) -> Self {
        Self {
            stress: update.stress,
            delta_equiv_plastic_strain: update.delta_equiv_plastic_strain,
            yielding: update.yielding,
        }
    }
}

/// Explicit material point behind the replay drivers.
pub trait ExplicitPoint {
    /// Pure trial step without mutating history.
    ///
    /// Drivers call this from lateral Newton iterations, so it must
    /// be side-effect free. Foam trials reuse stored filter rates.
    fn trial_point(&self, dstrain: &[f64; 6], dt: f64) -> PointStep;

    /// Accepted step advancing all history and totals.
    fn step_point_mut(&mut self, dstrain: &[f64; 6], dt: f64) -> PointStep;
}

/// Runner for foam laws with bundled totals advance.
#[derive(Debug, Clone)]
pub struct PointRunner<M: PointStepper>
where
    M::State: core::fmt::Debug + Clone,
{
    /// Material law stepped through [`PointStepper::step_point`].
    pub law: M,
    /// History state at the material point.
    pub state: M::State,
}

impl<M: PointStepper> PointRunner<M>
where
    M::State: core::fmt::Debug + Clone,
{
    /// Builds a runner from a law and its state.
    pub fn new(law: M, state: M::State) -> Self {
        Self { law, state }
    }
}

impl<M: PointStepper> ExplicitPoint for PointRunner<M>
where
    M::State: core::fmt::Debug + Clone,
{
    fn trial_point(&self, dstrain: &[f64; 6], dt: f64) -> PointStep {
        PointStep::from(&self.law.update_with_rate(&self.state, dstrain, dt))
    }

    fn step_point_mut(&mut self, dstrain: &[f64; 6], dt: f64) -> PointStep {
        PointStep::from(&self.law.step_point(&mut self.state, dstrain, dt))
    }
}

/// Runner for plain C1 laws with complete commits.
#[derive(Debug, Clone)]
pub struct C1Runner<M: StressUpdateModel<StrainIncrement = [f64; 6]>>
where
    M::State: core::fmt::Debug + Clone,
{
    /// Material law stepped through update plus commit.
    pub law: M,
    /// History state at the material point.
    pub state: M::State,
}

impl<M: StressUpdateModel<StrainIncrement = [f64; 6]>> C1Runner<M>
where
    M::State: core::fmt::Debug + Clone,
{
    /// Builds a runner from a law and its state.
    pub fn new(law: M, state: M::State) -> Self {
        Self { law, state }
    }
}

impl<M: StressUpdateModel<StrainIncrement = [f64; 6]>> ExplicitPoint for C1Runner<M>
where
    M::State: core::fmt::Debug + Clone,
{
    fn trial_point(&self, dstrain: &[f64; 6], dt: f64) -> PointStep {
        let update = if self.law.is_rate_dependent() {
            self.law.update_with_rate(&self.state, dstrain, dt)
        } else {
            self.law.update(&self.state, dstrain)
        };
        PointStep::from(&update)
    }

    fn step_point_mut(&mut self, dstrain: &[f64; 6], dt: f64) -> PointStep {
        let update = if self.law.is_rate_dependent() {
            self.law.update_with_rate(&self.state, dstrain, dt)
        } else {
            self.law.update(&self.state, dstrain)
        };
        self.law.commit(&mut self.state, &update);
        PointStep::from(&update)
    }
}

/// One prescribed total-strain stamp of a replay path.
#[derive(Debug, Clone, Copy)]
pub struct PathStamp {
    /// Stamp time in s, strictly increasing along a path.
    pub time: f64,
    /// Total strain in C1 Voigt order at the stamp.
    pub strain: [f64; 6],
    /// Current over reference volume, when the caller knows it.
    ///
    /// Corotational replays pass `det F` here; otherwise the driver
    /// falls back to `exp(tr(e) - tr(e0))`, exact for logarithmic
    /// strains and first-order for engineering strains.
    pub volume_ratio: Option<f64>,
}

impl PathStamp {
    /// Builds a stamp with the default volumetric fallback.
    pub fn strain(time: f64, strain: [f64; 6]) -> Self {
        Self {
            time,
            strain,
            volume_ratio: None,
        }
    }

    /// Builds a stamp with a known volume ratio.
    pub fn strained_volume(time: f64, strain: [f64; 6], volume_ratio: f64) -> Self {
        Self {
            time,
            strain,
            volume_ratio: Some(volume_ratio),
        }
    }
}

/// Accumulated history of a prescribed-path replay.
#[derive(Debug, Clone)]
pub struct FoamHistory {
    /// Stamp times in s, matching the input path.
    pub time: Vec<f64>,
    /// Total strain per stamp in C1 Voigt order.
    pub strain: Vec<[f64; 6]>,
    /// Cauchy stress per stamp in C1 Voigt order (Pa).
    pub stress: Vec<[f64; 6]>,
    /// Cumulative stress work density per stamp (J/m^3).
    pub energy_density: Vec<f64>,
    /// Cumulative equivalent plastic strain per stamp.
    pub equiv_plastic: Vec<f64>,
    /// Current mass density per stamp (kg/m^3).
    pub density: Vec<f64>,
    /// Yield flag per stamp (logical OR over substeps).
    pub yielding: Vec<bool>,
}

impl FoamHistory {
    /// Counts the recorded stamps.
    pub fn len(&self) -> usize {
        self.time.len()
    }

    /// Reports whether the history holds no stamps.
    pub fn is_empty(&self) -> bool {
        self.time.is_empty()
    }
}

/// Validates driver inputs shared by every replay entry.
fn check_replay(path_len: usize, substeps: usize, times_increasing: bool) {
    assert!(path_len > 0, "replay needs at least one stamp");
    assert!(substeps > 0, "replay needs at least one substep");
    assert!(
        times_increasing,
        "replay stamp times must strictly increase"
    );
}

/// Density from a stamp with the volumetric fallback.
///
/// Uses the caller volume ratio when present, else the exponential
/// of the trace shift from the path opening strain.
fn stamp_density(density0: f64, stamp: &PathStamp, trace0: f64) -> f64 {
    let ratio = stamp
        .volume_ratio
        .unwrap_or_else(|| (stamp.strain[0] + stamp.strain[1] + stamp.strain[2] - trace0).exp());
    density0 / ratio.max(f64::MIN_POSITIVE)
}

/// Replays a fully prescribed strain history explicitly.
///
/// Advances `point` along `path` with `substeps` uniform substeps per
/// stamp interval, booking trapezoid stress work per substep. The
/// history opens with the virgin state (zero stress and energy) and
/// records one entry per stamp.
///
/// # Examples
///
/// ```rust
/// use sonium_fem::physics::solid::plasticity::foam_explicit::{
///     C1Runner, PathStamp, replay_prescribed,
/// };
/// use sonium_fem::physics::solid::plasticity::{J2Material, PlasticState};
///
/// let law = J2Material::mild_steel();
/// let mut point = C1Runner::new(law, PlasticState::new_3d());
/// let path = [
///     PathStamp::strain(0.0, [0.0; 6]),
///     PathStamp::strain(2.0, [0.002, 0.0, 0.0, 0.0, 0.0, 0.0]),
/// ];
/// let history = replay_prescribed(&mut point, &path, 2, 7800.0);
/// assert!(history.stress[1][0] > 0.0);
/// ```
///
/// # Panics
///
/// Panics on an empty path, zero substeps, or non-increasing times.
pub fn replay_prescribed<P: ExplicitPoint>(
    point: &mut P,
    path: &[PathStamp],
    substeps: usize,
    density0: f64,
) -> FoamHistory {
    let increasing = path.windows(2).all(|w| w[1].time > w[0].time);
    check_replay(path.len(), substeps, increasing);
    let trace0 = path[0].strain[0] + path[0].strain[1] + path[0].strain[2];
    let mut history = FoamHistory {
        time: Vec::with_capacity(path.len()),
        strain: Vec::with_capacity(path.len()),
        stress: Vec::with_capacity(path.len()),
        energy_density: Vec::with_capacity(path.len()),
        equiv_plastic: Vec::with_capacity(path.len()),
        density: Vec::with_capacity(path.len()),
        yielding: Vec::with_capacity(path.len()),
    };
    history.time.push(path[0].time);
    history.strain.push(path[0].strain);
    history.stress.push([0.0; 6]);
    history.energy_density.push(0.0);
    history.equiv_plastic.push(0.0);
    history
        .density
        .push(stamp_density(density0, &path[0], trace0));
    history.yielding.push(false);
    let mut stress = [0.0; 6];
    let mut energy = 0.0;
    let mut ep = 0.0;
    for pair in path.windows(2) {
        let dt = (pair[1].time - pair[0].time) / substeps as f64;
        let dstrain: [f64; 6] =
            core::array::from_fn(|i| (pair[1].strain[i] - pair[0].strain[i]) / substeps as f64);
        let mut stamped_yield = false;
        for _ in 0..substeps {
            let accepted = point.step_point_mut(&dstrain, dt);
            for i in 0..6 {
                energy += 0.5 * (stress[i] + accepted.stress[i]) * dstrain[i];
            }
            stress = accepted.stress;
            ep += accepted.delta_equiv_plastic_strain.max(0.0);
            stamped_yield = stamped_yield || accepted.yielding;
        }
        history.time.push(pair[1].time);
        history.strain.push(pair[1].strain);
        history.stress.push(stress);
        history.energy_density.push(energy);
        history.equiv_plastic.push(ep);
        history
            .density
            .push(stamp_density(density0, &pair[1], trace0));
        history.yielding.push(stamped_yield);
    }
    history
}

/// Newton tolerance on lateral stress residuals (relative).
const LATERAL_RESIDUAL_REL: f64 = 1e-9;

/// Newton iterations cap for the lateral solve.
const LATERAL_MAX_ITERATIONS: usize = 25;

/// Finite-difference step scale for the lateral tangent.
const LATERAL_DIFF_SCALE: f64 = 1.5e-8;

/// Solves transverse strains for zero transverse stress.
///
/// Holds the axial increment fixed and iterates the two lateral
/// increments by Newton on the 2x2 numeric tangent of the trial
/// response. Seeds from `guess` and returns the lateral pair that
/// zeros the transverse stresses within tolerance.
fn solve_lateral<P: ExplicitPoint>(point: &P, axial: f64, dt: f64, guess: [f64; 2]) -> [f64; 2] {
    let mut lateral = guess;
    for _ in 0..LATERAL_MAX_ITERATIONS {
        let base = point.trial_point(&[axial, lateral[0], lateral[1], 0.0, 0.0, 0.0], dt);
        let scale = base.stress[0]
            .abs()
            .max(base.stress[1].abs())
            .max(base.stress[2].abs())
            .max(1.0);
        let r0 = base.stress[1];
        let r1 = base.stress[2];
        if !r0.is_finite() || !r1.is_finite() {
            break;
        }
        if r0.abs() < LATERAL_RESIDUAL_REL * scale && r1.abs() < LATERAL_RESIDUAL_REL * scale {
            break;
        }
        // Numeric 2x2 tangent by forward differences.
        let h0 = LATERAL_DIFF_SCALE * (1.0 + lateral[0].abs());
        let h1 = LATERAL_DIFF_SCALE * (1.0 + lateral[1].abs());
        let ahead0 = point.trial_point(&[axial, lateral[0] + h0, lateral[1], 0.0, 0.0, 0.0], dt);
        let ahead1 = point.trial_point(&[axial, lateral[0], lateral[1] + h1, 0.0, 0.0, 0.0], dt);
        let j00 = (ahead0.stress[1] - r0) / h0;
        let j10 = (ahead0.stress[2] - r1) / h0;
        let j01 = (ahead1.stress[1] - r0) / h1;
        let j11 = (ahead1.stress[2] - r1) / h1;
        let det = j00 * j11 - j01 * j10;
        if !det.is_finite() || det.abs() < 1e-30 * scale * scale {
            break;
        }
        lateral[0] -= (j11 * r0 - j01 * r1) / det;
        lateral[1] -= (j00 * r1 - j10 * r0) / det;
        if !lateral[0].is_finite() || !lateral[1].is_finite() {
            break;
        }
    }
    lateral
}

/// Replays an axial history under uniaxial stress explicitly.
///
/// Solves each substep lateral pair for zero transverse stress by
/// Newton from the previous pair, then accepts the full increment.
/// Records full strain tensors (axial plus solved lateral) per stamp
/// so callers recover thickness histories for force resultants.
///
/// # Examples
///
/// ```rust
/// use sonium_fem::physics::solid::plasticity::foam_explicit::{
///     AxialStamp, C1Runner, replay_uniaxial,
/// };
/// use sonium_fem::physics::solid::plasticity::{J2Material, PlasticState};
///
/// let law = J2Material::mild_steel();
/// let mut point = C1Runner::new(law, PlasticState::new_3d());
/// let path = [AxialStamp::new(0.0, 0.0), AxialStamp::new(1.0, 0.001)];
/// let history = replay_uniaxial(&mut point, &path, 2, 7800.0);
/// assert!(history.stress[1][0] > 0.0);
/// assert!(history.stress[1][1].abs() < 1.0e-6 * history.stress[1][0]);
/// ```
///
/// # Panics
///
/// Panics on an empty path, zero substeps, or non-increasing times.
pub fn replay_uniaxial<P: ExplicitPoint>(
    point: &mut P,
    path: &[AxialStamp],
    substeps: usize,
    density0: f64,
) -> FoamHistory {
    let increasing = path.windows(2).all(|w| w[1].time > w[0].time);
    check_replay(path.len(), substeps, increasing);
    let mut history = FoamHistory {
        time: Vec::with_capacity(path.len()),
        strain: Vec::with_capacity(path.len()),
        stress: Vec::with_capacity(path.len()),
        energy_density: Vec::with_capacity(path.len()),
        equiv_plastic: Vec::with_capacity(path.len()),
        density: Vec::with_capacity(path.len()),
        yielding: Vec::with_capacity(path.len()),
    };
    history.time.push(path[0].time);
    history.strain.push([0.0; 6]);
    history.stress.push([0.0; 6]);
    history.energy_density.push(0.0);
    history.equiv_plastic.push(0.0);
    history.density.push(density0);
    history.yielding.push(false);
    let mut lateral = [0.0, 0.0];
    let mut total = [0.0; 6];
    let mut stress = [0.0; 6];
    let mut energy = 0.0;
    let mut ep = 0.0;
    for pair in path.windows(2) {
        let dt = (pair[1].time - pair[0].time) / substeps as f64;
        let axial = (pair[1].strain_xx - pair[0].strain_xx) / substeps as f64;
        let mut stamped_yield = false;
        for _ in 0..substeps {
            lateral = solve_lateral(point, axial, dt, lateral);
            let dstrain = [axial, lateral[0], lateral[1], 0.0, 0.0, 0.0];
            let accepted = point.step_point_mut(&dstrain, dt);
            for i in 0..6 {
                energy += 0.5 * (stress[i] + accepted.stress[i]) * dstrain[i];
            }
            stress = accepted.stress;
            ep += accepted.delta_equiv_plastic_strain.max(0.0);
            stamped_yield = stamped_yield || accepted.yielding;
            total[0] += axial;
            total[1] += lateral[0];
            total[2] += lateral[1];
        }
        let stamp = PathStamp::strain(pair[1].time, total);
        history.time.push(pair[1].time);
        history.strain.push(total);
        history.stress.push(stress);
        history.energy_density.push(energy);
        history.equiv_plastic.push(ep);
        history.density.push(stamp_density(density0, &stamp, 0.0));
        history.yielding.push(stamped_yield);
    }
    history
}

/// One axial total-strain stamp of a uniaxial path.
#[derive(Debug, Clone, Copy)]
pub struct AxialStamp {
    /// Stamp time in s, strictly increasing along a path.
    pub time: f64,
    /// Total axial strain at the stamp.
    pub strain_xx: f64,
}

impl AxialStamp {
    /// Builds an axial stamp from time and total strain.
    pub fn new(time: f64, strain_xx: f64) -> Self {
        Self { time, strain_xx }
    }
}

/// Polar decomposition of a deformation gradient.
///
/// Factors `f = R U` with rotation `R` and symmetric
/// positive-definite stretch `U`, reusing the tested Jacobi eigensolver
/// on `C = F^T F`. Returns `(R, U)`.
///
/// # Examples
///
/// ```rust
/// use sonium_fem::physics::solid::plasticity::foam_explicit::polar_decomposition;
///
/// let theta = 0.3f64;
/// let (c, s) = (theta.cos(), theta.sin());
/// let f = [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 2.0]];
/// let (rotation, stretch) = polar_decomposition(&f);
/// assert!((stretch[2][2] - 2.0).abs() < 1.0e-12);
/// assert!((rotation[0][0] - c).abs() < 1.0e-12);
/// ```
///
/// # Panics
///
/// Panics when `f` has a non-positive determinant (unphysical or
/// inverted gradients violate the decomposition contract).
pub fn polar_decomposition(f: &[[f64; 3]; 3]) -> ([[f64; 3]; 3], [[f64; 3]; 3]) {
    let det = det3(f);
    assert!(
        det.is_finite() && det > 0.0,
        "polar decomposition needs det F > 0, found {det}"
    );
    // Right Cauchy-Green tensor in C1 Voigt order.
    let mut c = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            c[i][j] = f[0][i] * f[0][j] + f[1][i] * f[1][j] + f[2][i] * f[2][j];
        }
    }
    let voigt = [c[0][0], c[1][1], c[2][2], c[0][1], c[1][2], c[0][2]];
    let (values, vectors) = jacobi_eigen_c1(&voigt, 12);
    // Stretch U = V sqrt(Lambda) V^T; rotation R = F U^-1.
    let mut u = [[0.0; 3]; 3];
    for a in 0..3 {
        let root = values[a].max(0.0).sqrt();
        for i in 0..3 {
            for j in 0..3 {
                u[i][j] += root * vectors[i][a] * vectors[j][a];
            }
        }
    }
    let u_inv = invert3(&u, det3(&u));
    let mut r = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            r[i][j] = f[i][0] * u_inv[0][j] + f[i][1] * u_inv[1][j] + f[i][2] * u_inv[2][j];
        }
    }
    (r, u)
}

/// Determinant of a 3x3 row-major matrix.
fn det3(m: &[[f64; 3]; 3]) -> f64 {
    m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
}

/// Inverse of a 3x3 row-major matrix with known determinant.
fn invert3(m: &[[f64; 3]; 3], det: f64) -> [[f64; 3]; 3] {
    let inv = 1.0 / det;
    [
        [
            (m[1][1] * m[2][2] - m[1][2] * m[2][1]) * inv,
            (m[0][2] * m[2][1] - m[0][1] * m[2][2]) * inv,
            (m[0][1] * m[1][2] - m[0][2] * m[1][1]) * inv,
        ],
        [
            (m[1][2] * m[2][0] - m[1][0] * m[2][2]) * inv,
            (m[0][0] * m[2][2] - m[0][2] * m[2][0]) * inv,
            (m[0][2] * m[1][0] - m[0][0] * m[1][2]) * inv,
        ],
        [
            (m[1][0] * m[2][1] - m[1][1] * m[2][0]) * inv,
            (m[0][1] * m[2][0] - m[0][0] * m[2][1]) * inv,
            (m[0][0] * m[1][1] - m[0][1] * m[1][0]) * inv,
        ],
    ]
}

/// Rotates a C1 Voigt tensor by a rotation matrix.
///
/// Maps `v` to full tensor form, applies `R v R^T` with engineering
/// shear handling, and maps back to C1 order. Serves both corotated
/// strain increments and rotated-back stresses.
///
/// # Examples
///
/// ```rust
/// use sonium_fem::physics::solid::plasticity::foam_explicit::rotate_voigt6;
///
/// let id = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
/// let v = [1.0, 2.0, 3.0, 0.5, 0.25, 0.125];
/// assert_eq!(rotate_voigt6(&id, &v), v);
/// ```
pub fn rotate_voigt6(r: &[[f64; 3]; 3], v: &[f64; 6]) -> [f64; 6] {
    let t = [[v[0], v[3], v[5]], [v[3], v[1], v[4]], [v[5], v[4], v[2]]];
    let mut out = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            let mut sum = 0.0;
            for a in 0..3 {
                for b in 0..3 {
                    sum += r[i][a] * t[a][b] * r[j][b];
                }
            }
            out[i][j] = sum;
        }
    }
    [
        out[0][0], out[1][1], out[2][2], out[0][1], out[1][2], out[0][2],
    ]
}

/// Transpose of a 3x3 row-major matrix.
fn transpose3(m: &[[f64; 3]; 3]) -> [[f64; 3]; 3] {
    [
        [m[0][0], m[1][0], m[2][0]],
        [m[0][1], m[1][1], m[2][1]],
        [m[0][2], m[1][2], m[2][2]],
    ]
}

/// Rotation increment from a spin tensor over a step.
///
/// Returns `exp(W dt)` for the skew-symmetric spin `w` by the
/// Rodrigues formula, stable for small angles. Jaumann replays
/// compose these increments into the spin-integrated rotation.
///
/// # Examples
///
/// ```rust
/// use sonium_fem::physics::solid::plasticity::foam_explicit::spin_increment;
///
/// // Quarter turn about z in one step.
/// let turn = std::f64::consts::FRAC_PI_2;
/// let spin = [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]];
/// let rotation = spin_increment(&spin, turn);
/// assert!((rotation[0][0]).abs() < 1e-12);
/// assert!((rotation[0][1] + 1.0).abs() < 1e-12);
/// ```
pub fn spin_increment(spin: &[[f64; 3]; 3], dt: f64) -> [[f64; 3]; 3] {
    // Axial vector of the skew tensor times dt.
    let axial = [
        0.5 * (spin[2][1] - spin[1][2]) * dt,
        0.5 * (spin[0][2] - spin[2][0]) * dt,
        0.5 * (spin[1][0] - spin[0][1]) * dt,
    ];
    let angle = (axial[0] * axial[0] + axial[1] * axial[1] + axial[2] * axial[2]).sqrt();
    if angle < 1e-15 {
        return [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    }
    let axis = [axial[0] / angle, axial[1] / angle, axial[2] / angle];
    let skew = [
        [0.0, -axis[2], axis[1]],
        [axis[2], 0.0, -axis[0]],
        [-axis[1], axis[0], 0.0],
    ];
    let (sin, cos) = angle.sin_cos();
    let mut out = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            let skew2 = skew[i][0] * skew[0][j] + skew[i][1] * skew[1][j] + skew[i][2] * skew[2][j];
            out[i][j] = sin * skew[i][j] + (1.0 - cos) * skew2 + if i == j { 1.0 } else { 0.0 };
        }
    }
    out
}

/// Corotational strain tracker for large-rotation replays.
///
/// Consumes deformation gradients stamp by stamp, holding the polar
/// rotation per stamp. Each accepted gradient yields the corotated
/// Biot-strain increment `R_mid^T (U - U_prev) R_mid` evaluated with
/// the midpoint rotation, which callers feed to the law; the stored
/// rotations map corotated stresses back to the global frame.
///
/// # Examples
///
/// ```rust
/// use sonium_fem::physics::solid::plasticity::foam_explicit::Corotator;
///
/// let id = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
/// let mut corot = Corotator::new();
/// assert!(corot.push(&id).is_none());
/// let shear = [[1.0, 0.1, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
/// let step = corot.push(&shear).unwrap();
/// assert!(step.corotated_dstrain[3].abs() > 0.0);
/// ```
#[derive(Debug, Clone, Default)]
pub struct Corotator {
    /// Previous stretch tensor, `None` before the first gradient.
    prev_stretch: Option<[[f64; 3]; 3]>,
    /// Previous rotation tensor, `None` before the first gradient.
    prev_rotation: Option<[[f64; 3]; 3]>,
}

/// One accepted corotational step with its frames.
#[derive(Debug, Clone, Copy)]
pub struct CorotatedStep {
    /// Corotated Biot-strain increment in C1 Voigt order.
    pub corotated_dstrain: [f64; 6],
    /// Polar rotation of the new gradient.
    pub rotation: [[f64; 3]; 3],
    /// Determinant of the new gradient (volume ratio).
    pub volume_ratio: f64,
}

impl Corotator {
    /// Builds an empty corotational tracker.
    pub fn new() -> Self {
        Self::default()
    }

    /// Accepts one deformation gradient in row-major order.
    ///
    /// Returns `None` for the opening gradient (no increment yet) and
    /// `Some(step)` afterwards with the corotated increment plus the
    /// new rotation for stress mapping.
    ///
    /// # Panics
    ///
    /// Panics when any gradient has a non-positive determinant.
    pub fn push(&mut self, gradient: &[[f64; 3]; 3]) -> Option<CorotatedStep> {
        let (rotation, stretch) = polar_decomposition(gradient);
        let volume_ratio = det3(gradient);
        let prev_u = self.prev_stretch.replace(stretch);
        let prev_r = self.prev_rotation.replace(rotation);
        let (Some(prev_u), Some(prev_r)) = (prev_u, prev_r) else {
            return None;
        };
        // Midpoint rotation from the averaged gradient polar factor.
        let mid: [[f64; 3]; 3] = core::array::from_fn(|i| {
            core::array::from_fn(|j| 0.5 * (prev_r[i][j] + rotation[i][j]))
        });
        let (rotation_mid, _) = polar_decomposition(&mid);
        let rt = transpose3(&rotation_mid);
        let du: [[f64; 3]; 3] =
            core::array::from_fn(|i| core::array::from_fn(|j| stretch[i][j] - prev_u[i][j]));
        let du_voigt = [du[0][0], du[1][1], du[2][2], du[0][1], du[1][2], du[0][2]];
        // Stretch-difference corotation: R_mid^T dU R_mid in one map.
        // The map is tensor-linear, so engineering shear rides through
        // unchanged; the tensor difference packs to engineering order
        // by doubling the shears for C1 consumers.
        let rotated = rotate_voigt6(&rt, &du_voigt);
        let corotated_dstrain = [
            rotated[0],
            rotated[1],
            rotated[2],
            2.0 * rotated[3],
            2.0 * rotated[4],
            2.0 * rotated[5],
        ];
        Some(CorotatedStep {
            corotated_dstrain,
            rotation,
            volume_ratio,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::super::funct::Funct1D;
    use super::super::law_foams::{HoneycombLaw, HoneycombState};
    use super::*;

    /// Linear isotropic point with explicit stress memory.
    struct LinearPoint {
        e: f64,
        nu: f64,
        stress: [f64; 6],
    }

    impl LinearPoint {
        fn predict(&self, dstrain: &[f64; 6]) -> [f64; 6] {
            let lambda = self.e * self.nu / ((1.0 + self.nu) * (1.0 - 2.0 * self.nu));
            let mu = self.e / (2.0 * (1.0 + self.nu));
            let trace = dstrain[0] + dstrain[1] + dstrain[2];
            core::array::from_fn(|i| {
                let elastic = if i < 3 {
                    lambda * trace + 2.0 * mu * dstrain[i]
                } else {
                    mu * dstrain[i]
                };
                self.stress[i] + elastic
            })
        }
    }

    impl ExplicitPoint for LinearPoint {
        fn trial_point(&self, dstrain: &[f64; 6], _dt: f64) -> PointStep {
            PointStep {
                stress: self.predict(dstrain),
                delta_equiv_plastic_strain: 0.0,
                yielding: false,
            }
        }

        fn step_point_mut(&mut self, dstrain: &[f64; 6], _dt: f64) -> PointStep {
            self.stress = self.predict(dstrain);
            PointStep {
                stress: self.stress,
                delta_equiv_plastic_strain: 0.0,
                yielding: false,
            }
        }
    }

    #[test]
    fn prescribed_linear_energy_matches_closed_form() {
        let mut point = LinearPoint {
            e: 200.0e9,
            nu: 0.3,
            stress: [0.0; 6],
        };
        let path = [
            PathStamp::strain(0.0, [0.0; 6]),
            PathStamp::strain(1.0, [0.001, 0.0, 0.0, 0.0, 0.0, 0.0]),
        ];
        for substeps in [1, 4, 16] {
            point.stress = [0.0; 6];
            let history = replay_prescribed(&mut point, &path, substeps, 7800.0);
            // Uniaxial-strain modulus M = E(1-nu)/((1+nu)(1-2nu)).
            let m = 200.0e9 * 0.7 / (1.3 * 0.4);
            let expected = 0.5 * m * 0.001 * 0.001;
            assert!(
                (history.energy_density[1] - expected).abs() < 1e-9 * expected,
                "substeps {substeps}: {} vs {expected}",
                history.energy_density[1]
            );
            // Substep count cannot change a linear path result.
            assert!((history.stress[1][0] - m * 0.001).abs() < 1e-6);
        }
    }

    #[test]
    fn uniaxial_lateral_solve_recovers_poisson() {
        let mut point = LinearPoint {
            e: 200.0e9,
            nu: 0.3,
            stress: [0.0; 6],
        };
        let path = [AxialStamp::new(0.0, 0.0), AxialStamp::new(1.0, 0.002)];
        let history = replay_uniaxial(&mut point, &path, 4, 7800.0);
        assert!((history.stress[1][0] - 200.0e9 * 0.002).abs() < 1.0);
        assert!(history.stress[1][1].abs() < 1.0e-6 * history.stress[1][0]);
        assert!(history.stress[1][2].abs() < 1.0e-6 * history.stress[1][0]);
        assert!((history.strain[1][1] + 0.3 * 0.002).abs() < 1e-9);
        assert!((history.strain[1][2] + 0.3 * 0.002).abs() < 1e-9);
    }

    #[test]
    fn polar_factors_reconstruct_gradient() {
        let theta = 0.7f64;
        let (c, s) = (theta.cos(), theta.sin());
        let f = [
            [2.0 * c, -0.5 * s, 0.0],
            [2.0 * s, 0.5 * c, 0.0],
            [0.0, 0.0, 1.5],
        ];
        let (r, u) = polar_decomposition(&f);
        // Rotation stays orthogonal with determinant one.
        for i in 0..3 {
            for j in 0..3 {
                let dot = r[0][i] * r[0][j] + r[1][i] * r[1][j] + r[2][i] * r[2][j];
                let want = if i == j { 1.0 } else { 0.0 };
                assert!((dot - want).abs() < 1e-12);
            }
        }
        assert!((det3(&r) - 1.0).abs() < 1e-12);
        // Stretch stays symmetric; factors reconstruct F.
        for i in 0..3 {
            for j in 0..3 {
                assert!((u[i][j] - u[j][i]).abs() < 1e-12);
                let rebuilt = r[i][0] * u[0][j] + r[i][1] * u[1][j] + r[i][2] * u[2][j];
                assert!((rebuilt - f[i][j]).abs() < 1e-11);
            }
        }
    }

    #[test]
    fn corotator_reports_engineering_shear() {
        // Rust guideline compliant 2026-02-21
        // Simple shear F(γ = 0.1): the corotated increment must carry
        // the full engineering shear, not the halved tensor value.
        let mut corotator = Corotator::new();
        let f = [[1.0, 0.1, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let _ = corotator.push(&[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]);
        let step = corotator.push(&f).expect("second push steps");
        assert!(step.corotated_dstrain[3] > 0.09);
        assert!(step.corotated_dstrain[3] < 0.11);
    }

    #[test]
    fn corotator_ignores_rigid_rotation() {
        let mut corot = Corotator::new();
        let id = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        assert!(corot.push(&id).is_none());
        let theta = 0.9f64;
        let (c, s) = (theta.cos(), theta.sin());
        let rotated = [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]];
        let step = corot.push(&rotated).unwrap();
        let norm: f64 = step
            .corotated_dstrain
            .iter()
            .map(|x| x * x)
            .sum::<f64>()
            .sqrt();
        assert!(norm < 1e-12, "rigid spin must not strain: {norm}");
        assert!((step.volume_ratio - 1.0).abs() < 1e-15);
    }

    #[test]
    fn rotate_voigt6_quarter_turn_swaps_axes() {
        // +90 degrees about z maps x to y.
        let r = [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]];
        let back = rotate_voigt6(&r, &[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        assert!((back[0]).abs() < 1e-15);
        assert!((back[1] - 1.0).abs() < 1e-15);
    }

    #[test]
    fn honeycomb_replay_caps_and_books_energy() {
        let cap = Funct1D::new(vec![-1.0, 1.0], vec![0.5, 0.5]).unwrap();
        let tables = core::array::from_fn(|_| cap.clone());
        let law = HoneycombLaw::law28([2.0; 6], tables).unwrap();
        let mut point = PointRunner::new(law, HoneycombState::new());
        let path = [
            PathStamp::strain(0.0, [0.0; 6]),
            PathStamp::strain(1.0, [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        ];
        // Four substeps land the yield kink (0.25) on a boundary.
        let history = replay_prescribed(&mut point, &path, 4, 100.0);
        // Elastic to 0.25 strain (E*eps = 0.5), then capped at 0.5.
        assert!((history.stress[1][0] - 0.5).abs() < 1e-12);
        assert!(history.yielding[1]);
        // Work: triangle 0.5*0.5*0.25 plus rectangle 0.5*0.75.
        let expected = 0.5 * 0.5 * 0.25 + 0.5 * 0.75;
        assert!((history.energy_density[1] - expected).abs() < 1e-9);
        assert!(history.equiv_plastic[1] > 0.0);
    }

    #[test]
    fn penta6_foam_single_step_balances_forces() {
        use crate::physics::solid::nonlinear_solid::{
            Penta6Geom, Penta6State, TetPentaParams, penta6_forces,
        };

        let geom = Penta6Geom::new([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
        ])
        .unwrap();
        let cap = Funct1D::new(vec![-1.0, 1.0], vec![1.0e9, 1.0e9]).unwrap();
        let tables = core::array::from_fn(|_| cap.clone());
        let law = HoneycombLaw::law28([2.0e9; 6], tables).unwrap();
        let params = TetPentaParams::new(1000.0, 1500.0);
        // Uniform dilatation velocities: every point strains equally.
        let mut velocities = [0.0; 18];
        for (node, coords) in geom.coords.iter().enumerate() {
            velocities[node * 3] = coords[0];
            velocities[node * 3 + 1] = coords[1];
            velocities[node * 3 + 2] = coords[2];
        }
        let mut state = Penta6State::new(core::array::from_fn(|_| HoneycombState::new()));
        let out = penta6_forces(&geom, &mut state, &law, &velocities, 1.0e-6, &params).unwrap();
        assert!(!out.yielding);
        assert!(out.dep == 0.0);
        // Self-equilibrated internal forces sum to zero per axis.
        for axis in 0..3 {
            let total: f64 = (0..6).map(|n| out.forces[n * 3 + axis]).sum();
            let scale: f64 = (0..6).map(|n| out.forces[n * 3 + axis].abs()).sum();
            assert!(total.abs() < 1e-9 * scale.max(1.0));
        }
        // Rigid translation produces no forces at all.
        let rigid = [
            1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0,
            3.0,
        ];
        let mut fresh = Penta6State::new(core::array::from_fn(|_| HoneycombState::new()));
        let still = penta6_forces(&geom, &mut fresh, &law, &rigid, 1.0e-6, &params).unwrap();
        assert!(still.forces.iter().all(|f| f.abs() < 1e-9));
    }
}
