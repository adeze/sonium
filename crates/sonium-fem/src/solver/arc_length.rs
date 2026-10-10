//! Riks arc-length continuation for nonlinear equilibrium paths.
//!
//! This module traces `F_int(u) - lambda * F_ext = 0` past limit points with
//! Crisfield cylindrical arc-length control. The driver is generic over
//! residual and tangent-solve closures; the FEM wrapper in [`super`]
//! assembles both from a [`NonlinearForm`](crate::forms::NonlinearForm).

// Rust guideline compliant 2026-02-21

use super::FemNewtonConfig;
use crate::forms::NonlinearForm;
use crate::mesh::Mesh;
use crate::spaces::FunctionSpace;
use sonium_solvers::{solve_real, LinearSolveConfig, NewtonError};

/// Riks continuation settings.
#[derive(Debug, Clone)]
pub struct ArcLengthConfig {
    /// First arc-length radius.
    pub initial_step: f64,
    /// Smallest accepted radius; the trace stops below it.
    pub min_step: f64,
    /// Largest accepted radius.
    pub max_step: f64,
    /// Maximum accepted steps.
    pub max_steps: usize,
    /// Corrector tolerance on the residual 2-norm, relative to the larger
    /// of 1 and the current load level (`|lambda| * ||F_ext||`).
    pub corrector_tolerance: f64,
    /// Corrector tolerance on `|a| / ds^2`, the scaled constraint gap.
    pub constraint_tolerance: f64,
    /// Maximum corrector iterations per step attempt.
    pub corrector_max_iterations: usize,
    /// Step halvings per step before the trace stops.
    pub max_step_halvings: usize,
    /// Load weight `psi` in the arc-length constraint.
    pub psi: f64,
    /// Stop accepting steps past this load factor.
    pub max_lambda: Option<f64>,
    /// Stop accepting steps below this load factor.
    pub min_lambda: Option<f64>,
    /// Iteration count the step-growth heuristic targets.
    pub target_iterations: usize,
}

impl Default for ArcLengthConfig {
    fn default() -> Self {
        Self {
            initial_step: 0.1,
            min_step: 1e-6,
            max_step: 1.0,
            max_steps: 50,
            corrector_tolerance: 1e-10,
            constraint_tolerance: 1e-8,
            corrector_max_iterations: 20,
            max_step_halvings: 10,
            psi: 1.0,
            max_lambda: None,
            min_lambda: None,
            target_iterations: 4,
        }
    }
}

impl ArcLengthConfig {
    /// Validate step sizes, tolerances, and iteration budgets.
    ///
    /// # Errors
    ///
    /// Returns [`NewtonError::InvalidConfig`] for non-positive steps,
    /// non-finite tolerances, or empty iteration budgets.
    pub fn validate(&self) -> Result<(), NewtonError> {
        for (field, value) in [
            ("initial_step", self.initial_step),
            ("min_step", self.min_step),
            ("max_step", self.max_step),
            ("corrector_tolerance", self.corrector_tolerance),
            ("constraint_tolerance", self.constraint_tolerance),
            ("psi", self.psi),
        ] {
            if !value.is_finite() || value <= 0.0 {
                return Err(NewtonError::InvalidConfig(format!(
                    "arc-length {field} must be finite and positive"
                )));
            }
        }
        if self.min_step > self.initial_step || self.initial_step > self.max_step {
            return Err(NewtonError::InvalidConfig(
                "arc-length steps must satisfy min_step <= initial_step <= max_step".into(),
            ));
        }
        if self.max_steps == 0 {
            return Err(NewtonError::InvalidConfig(
                "arc-length max_steps must be non-zero".into(),
            ));
        }
        if self.corrector_max_iterations == 0 {
            return Err(NewtonError::InvalidConfig(
                "arc-length corrector_max_iterations must be non-zero".into(),
            ));
        }
        if self.target_iterations == 0 {
            return Err(NewtonError::InvalidConfig(
                "arc-length target_iterations must be non-zero".into(),
            ));
        }
        Ok(())
    }
}

/// One accepted equilibrium point of a traced path.
#[derive(Debug, Clone)]
pub struct ArcLengthPoint {
    /// Displacement state.
    pub u: Vec<f64>,
    /// Load factor.
    pub lambda: f64,
    /// Final corrector residual norm.
    pub residual: f64,
    /// Corrector iterations used.
    pub newton_iterations: usize,
}

/// Why a trace stopped.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArcLengthStop {
    /// Accepted `max_steps` steps.
    MaxSteps,
    /// Crossed `max_lambda`.
    MaxLambda,
    /// Crossed `min_lambda`.
    MinLambda,
    /// The radius fell below `min_step`.
    MinStep,
}

/// An accepted equilibrium path plus trace statistics.
#[derive(Debug, Clone)]
pub struct ArcLengthPath {
    /// Accepted points, starting with the initial state.
    pub points: Vec<ArcLengthPoint>,
    /// Accepted steps (points beyond the initial state).
    pub steps_accepted: usize,
    /// Rejected step attempts.
    pub steps_rejected: usize,
    /// Why the trace stopped.
    pub stopped: ArcLengthStop,
    /// Final arc-length radius.
    pub final_step: f64,
}

fn norm2(values: &[f64]) -> f64 {
    values.iter().map(|value| value * value).sum::<f64>().sqrt()
}

fn dot(left: &[f64], right: &[f64]) -> f64 {
    left.iter().zip(right).map(|(a, b)| a * b).sum()
}

/// Trace an equilibrium path with cylindrical arc-length control.
///
/// The internal residual closure returns `F_int(u)`; the tangent closure
/// solves `K(u) z = rhs` with the consistent tangent. The external load is
/// the fixed reference direction scaled by `lambda`.
///
/// # Errors
///
/// Returns [`NewtonError::InvalidConfig`] for malformed inputs and
/// [`NewtonError::NumericalError`] when a tangent solve fails inside the
/// corrector after exhausting the step-halving budget.
///
/// # Examples
///
/// ```rust
/// use sonium_fem::solver::nonlinear::arc_length::{ArcLengthConfig, riks_continuation};
/// // Scalar softening spring: F_int(u) = u - u^3 / 3.
/// let path = riks_continuation(
///     |u| vec![u[0] - u[0].powi(3) / 3.0],
///     |u, rhs| Ok(vec![rhs[0] / (1.0 - u[0] * u[0])]),
///     &[1.0],
///     &[0.0],
///     0.0,
///     &ArcLengthConfig::default(),
/// )
/// .unwrap();
/// assert!(path.steps_accepted > 0);
/// ```
pub fn riks_continuation(
    internal_residual: impl Fn(&[f64]) -> Vec<f64>,
    tangent_solve: impl Fn(&[f64], &[f64]) -> Result<Vec<f64>, String>,
    external_load: &[f64],
    u0: &[f64],
    lambda0: f64,
    config: &ArcLengthConfig,
) -> Result<ArcLengthPath, NewtonError> {
    config.validate()?;
    let n = u0.len();
    if n == 0 {
        return Err(NewtonError::InvalidConfig(
            "arc-length system must contain at least one DOF".into(),
        ));
    }
    if external_load.len() != n {
        return Err(NewtonError::InvalidConfig(format!(
            "external load has {} entries; expected {n}",
            external_load.len()
        )));
    }
    if !lambda0.is_finite() || u0.iter().any(|value| !value.is_finite()) {
        return Err(NewtonError::InvalidConfig(
            "arc-length initial state must be finite".into(),
        ));
    }
    if external_load.iter().any(|value| !value.is_finite()) {
        return Err(NewtonError::InvalidConfig(
            "arc-length external load must be finite".into(),
        ));
    }
    let load_norm = norm2(external_load);
    if load_norm == 0.0 {
        return Err(NewtonError::InvalidConfig(
            "arc-length external load must be non-zero".into(),
        ));
    }

    let residual_at = |u: &[f64], lambda: f64| {
        let mut residual = internal_residual(u);
        debug_assert_eq!(residual.len(), n);
        residual.truncate(n);
        for (entry, applied) in residual.iter_mut().zip(external_load) {
            *entry -= lambda * applied;
        }
        residual
    };

    let mut path = ArcLengthPath {
        points: vec![ArcLengthPoint {
            u: u0.to_vec(),
            lambda: lambda0,
            residual: norm2(&residual_at(u0, lambda0)),
            newton_iterations: 0,
        }],
        steps_accepted: 0,
        steps_rejected: 0,
        stopped: ArcLengthStop::MaxSteps,
        final_step: config.initial_step,
    };

    // Previous accepted increment, for direction tracking.
    let mut prev_du = vec![0.0; n];
    let mut prev_dlambda = 0.0_f64;
    let mut step = config.initial_step;

    for _ in 0..config.max_steps {
        let base = path.points.last().expect("path always holds a base point");
        let mut halvings = 0;
        let accepted = loop {
            let trial = corrector_step(&CorrectorInput {
                residual_at: &residual_at,
                tangent_solve: &tangent_solve,
                external_load,
                load_norm,
                base_u: &base.u,
                base_lambda: base.lambda,
                prev_du: &prev_du,
                prev_dlambda,
                step,
                config,
            });
            match trial {
                Ok(point) => break Some(point),
                Err(()) => {
                    path.steps_rejected += 1;
                    halvings += 1;
                    step /= 2.0;
                    if step < config.min_step || halvings > config.max_step_halvings {
                        break None;
                    }
                }
            }
        };
        let Some((point, iterations)) = accepted else {
            path.stopped = ArcLengthStop::MinStep;
            path.final_step = step;
            return Ok(path);
        };
        // Grow or shrink the radius from the corrector effort.
        let growth = (config.target_iterations as f64 / iterations.max(1) as f64)
            .sqrt()
            .clamp(0.5, 2.0);
        step = (step * growth).clamp(config.min_step, config.max_step);
        prev_du = point
            .u
            .iter()
            .zip(&base.u)
            .map(|(next, prev)| next - prev)
            .collect();
        prev_dlambda = point.lambda - base.lambda;
        path.steps_accepted += 1;
        path.points.push(point);
        let tip = path.points.last().expect("just pushed");
        if config.max_lambda.is_some_and(|max| tip.lambda >= max) {
            path.stopped = ArcLengthStop::MaxLambda;
            path.final_step = step;
            return Ok(path);
        }
        if config.min_lambda.is_some_and(|min| tip.lambda <= min) {
            path.stopped = ArcLengthStop::MinLambda;
            path.final_step = step;
            return Ok(path);
        }
    }
    path.final_step = step;
    Ok(path)
}

/// Inputs of one predictor-plus-corrector step attempt.
struct CorrectorInput<'a, R, S> {
    residual_at: &'a R,
    tangent_solve: &'a S,
    external_load: &'a [f64],
    load_norm: f64,
    base_u: &'a [f64],
    base_lambda: f64,
    prev_du: &'a [f64],
    prev_dlambda: f64,
    step: f64,
    config: &'a ArcLengthConfig,
}

/// Attempt one predictor-plus-corrector step, returning the point and cost.
fn corrector_step(
    input: &CorrectorInput<
        impl Fn(&[f64], f64) -> Vec<f64>,
        impl Fn(&[f64], &[f64]) -> Result<Vec<f64>, String>,
    >,
) -> Result<(ArcLengthPoint, usize), ()> {
    let config = input.config;
    let n = input.base_u.len();
    let zt = (input.tangent_solve)(input.base_u, input.external_load).map_err(|_| ())?;
    if zt.len() != n || zt.iter().any(|value| !value.is_finite()) {
        return Err(());
    }
    let scale = (dot(&zt, &zt) + config.psi * config.psi * 1.0).sqrt();
    if scale == 0.0 || !scale.is_finite() {
        return Err(());
    }
    // Predictor along the tangent; keep the previous direction when set.
    let mut dlambda = input.step / scale;
    let mut du: Vec<f64> = zt.iter().map(|value| value * dlambda).collect();
    if input.prev_dlambda != 0.0 || input.prev_du.iter().any(|value| *value != 0.0) {
        let forward =
            dot(input.prev_du, &du) + config.psi * config.psi * input.prev_dlambda * dlambda;
        if forward < 0.0 {
            dlambda = -dlambda;
            du.iter_mut().for_each(|value| *value = -*value);
        }
    }

    for iteration in 0..config.corrector_max_iterations {
        let u: Vec<f64> = input.base_u.iter().zip(&du).map(|(b, d)| b + d).collect();
        let lambda = input.base_lambda + dlambda;
        let residual = (input.residual_at)(&u, lambda);
        if residual.len() != n || residual.iter().any(|value| !value.is_finite()) {
            return Err(());
        }
        let gap =
            dot(&du, &du) + config.psi * config.psi * dlambda * dlambda - input.step * input.step;
        if !gap.is_finite() {
            return Err(());
        }
        let residual_norm = norm2(&residual);
        let level =
            (input.load_norm * input.base_lambda.abs().max(lambda.abs()).max(0.02)).max(1.0);
        if residual_norm <= config.corrector_tolerance * level
            && gap.abs() <= config.constraint_tolerance * input.step * input.step
        {
            return Ok((
                ArcLengthPoint {
                    u,
                    lambda,
                    residual: residual_norm,
                    newton_iterations: iteration,
                },
                iteration + 1,
            ));
        }
        // Bordered solve through two tangent backsolves.
        let z1 = (input.tangent_solve)(&u, &residual).map_err(|_| ())?;
        let z2 = (input.tangent_solve)(&u, input.external_load).map_err(|_| ())?;
        if z1.len() != n || z2.len() != n {
            return Err(());
        }
        if z1.iter().chain(z2.iter()).any(|value| !value.is_finite()) {
            return Err(());
        }
        let denominator = dot(&du, &z2) + config.psi * config.psi * dlambda;
        if denominator == 0.0 || !denominator.is_finite() {
            return Err(());
        }
        let step_lambda = (dot(&du, &z1) - gap / 2.0) / denominator;
        if !step_lambda.is_finite() {
            return Err(());
        }
        for ((slot, minus), plus) in du.iter_mut().zip(&z1).zip(&z2) {
            let next = *slot - minus + plus * step_lambda;
            if !next.is_finite() {
                return Err(());
            }
            *slot = next;
        }
        dlambda += step_lambda;
        if !dlambda.is_finite() {
            return Err(());
        }
    }
    Err(())
}

/// Trace an FEM equilibrium path with Dirichlet BCs and a reference load.
///
/// Assembles `F_int(u)` and the consistent tangent from `form` at every
/// corrector iteration and scales `external_load` by the traced `lambda`,
/// so the first accepted steps reproduce load-controlled Newton at small
/// `lambda` and later steps continue past limit points.
///
/// # Errors
///
/// Returns [`NewtonError`] for malformed meshes, spaces, loads, or
/// arc-length settings, and when the trace cannot advance.
pub fn fem_riks_solve_with_bcs_and_load<F, S>(
    form: &F,
    space: &S,
    mesh: &Mesh,
    dirichlet_dofs: &[(usize, f64)],
    external_load: &[f64],
    initial_guess: &[f64],
    fem_config: &FemNewtonConfig,
    arc_config: &ArcLengthConfig,
) -> Result<ArcLengthPath, NewtonError>
where
    F: NonlinearForm<f64>,
    S: FunctionSpace,
{
    super::validate_space_and_mesh(space, mesh)?;
    let n_dofs = space.num_global_dofs(mesh);
    super::validate_newton_inputs(n_dofs, initial_guess, fem_config)?;
    super::validate_dirichlet_dofs(n_dofs, dirichlet_dofs)?;
    super::validate_external_load(n_dofs, external_load)?;
    if external_load.iter().all(|value| *value == 0.0) {
        return Err(NewtonError::InvalidConfig(
            "arc-length reference load must be non-zero".into(),
        ));
    }
    arc_config.validate()?;

    let is_constrained = {
        let mut flags = vec![false; n_dofs];
        for &(dof, _) in dirichlet_dofs {
            flags[dof] = true;
        }
        flags
    };
    let mut u0 = initial_guess.to_vec();
    for &(dof, value) in dirichlet_dofs {
        u0[dof] = value;
    }

    let quad_order = fem_config.quad_order;
    let inner_config: LinearSolveConfig = fem_config.inner_solver.clone();

    let internal = |u: &[f64]| {
        let mut residual = super::assemble_residual(form, space, mesh, u, quad_order);
        for &(dof, _) in dirichlet_dofs {
            residual[dof] = 0.0;
        }
        residual
    };
    let solve = |u: &[f64], rhs: &[f64]| {
        let (_, mut tangent) = super::assemble_nonlinear(form, space, mesh, u, quad_order);
        super::apply_dirichlet_to_triplet(&mut tangent, &is_constrained);
        let mut modified = rhs.to_vec();
        for &(dof, _) in dirichlet_dofs {
            modified[dof] = 0.0;
        }
        let csr = tangent.to_csr();
        solve_real(&csr, &modified, &inner_config)
            .map(|result| result.x)
            .map_err(|error| error.to_string())
    };

    // Constrained reference entries never act; drop them for a clean trace.
    let mut reference = external_load.to_vec();
    for &(dof, _) in dirichlet_dofs {
        reference[dof] = 0.0;
    }
    riks_continuation(internal, solve, &reference, &u0, 0.0, arc_config)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Von Mises truss residual with unit geometry, driven by arc length.
    fn truss_residual(u: &[f64]) -> Vec<f64> {
        // Equilibrium is P_closed(w) = lambda * 1, so the driver needs
        // F_int(w) = P_closed(w): the downward load balancing the bars.
        let (b, h): (f64, f64) = (1.0, 0.1);
        let l0 = (b * b + h * h).sqrt();
        let h_now = h - u[0];
        let l_now = (b * b + h_now * h_now).sqrt();
        vec![2.0 * (l0 - l_now) / l0 * h_now / l_now]
    }

    fn truss_tangent(u: &[f64], rhs: &[f64]) -> Result<Vec<f64>, String> {
        let h = 1e-8;
        let mut up = u.to_vec();
        let mut down = u.to_vec();
        up[0] += h;
        down[0] -= h;
        let slope = (truss_residual(&up)[0] - truss_residual(&down)[0]) / (2.0 * h);
        if slope == 0.0 {
            return Err("singular truss tangent".to_string());
        }
        Ok(vec![rhs[0] / slope])
    }

    #[test]
    fn softening_spring_traces_past_its_limit_point() {
        let config = ArcLengthConfig {
            initial_step: 0.2,
            max_step: 0.3,
            max_steps: 40,
            ..Default::default()
        };
        let path = riks_continuation(
            |u| vec![u[0] - u[0].powi(3) / 3.0],
            |u, rhs| {
                let slope = 1.0 - u[0] * u[0];
                if slope.abs() < 1e-12 {
                    return Err("singular tangent".to_string());
                }
                Ok(vec![rhs[0] / slope])
            },
            &[1.0],
            &[0.0],
            0.0,
            &config,
        )
        .unwrap();
        // The limit point sits at u = 1; the trace must continue past it
        // while the load factor falls (snap-through branch).
        let peak_lambda = path
            .points
            .iter()
            .map(|point| point.lambda)
            .fold(f64::NEG_INFINITY, f64::max);
        assert!(peak_lambda > 0.6, "missed the 2/3 limit load");
        let tip = path.points.last().unwrap();
        assert!(tip.u[0] > 1.5, "trace stalled before snap-through");
        assert!(tip.lambda < peak_lambda, "load must fall past the limit");
        for point in &path.points {
            assert!(point.residual <= 1e-8, "loose corrector at {:?}", point.u);
        }
    }

    #[test]
    fn von_mises_truss_reaches_its_mirror_branch() {
        let config = ArcLengthConfig {
            initial_step: 0.02,
            max_step: 0.03,
            max_steps: 60,
            psi: 0.1,
            ..Default::default()
        };
        let path =
            riks_continuation(truss_residual, truss_tangent, &[1.0], &[0.0], 0.0, &config).unwrap();
        // Flat position at w = h = 0.1; the trace must snap past it.
        let tip = path.points.last().unwrap();
        assert!(
            tip.u[0] > 0.15,
            "trace stalled near the flat position: {}",
            tip.u[0]
        );
        // First limit load before the flat position: 3.81e-4 for this truss.
        let peak_lambda = path
            .points
            .iter()
            .filter(|point| point.u[0] < 0.12)
            .map(|point| point.lambda)
            .fold(f64::NEG_INFINITY, f64::max);
        assert!(
            (peak_lambda - 3.810872e-4).abs() < 0.01 * 3.810872e-4,
            "limit load {peak_lambda:.6e} misses the closed form"
        );
        // Every point must sit on the equilibrium curve.
        for point in &path.points {
            let closed = truss_residual(&point.u)[0];
            assert!(
                (point.lambda - closed).abs() < 1e-6,
                "point w={} leaves the curve",
                point.u[0]
            );
        }
    }

    #[test]
    fn malformed_arc_length_inputs_are_typed_errors() {
        let config = ArcLengthConfig::default();
        assert!(matches!(
            riks_continuation(
                |u| u.to_vec(),
                |_, rhs| Ok(rhs.to_vec()),
                &[1.0, 2.0],
                &[0.0],
                0.0,
                &config
            ),
            Err(NewtonError::InvalidConfig(_))
        ));
        assert!(matches!(
            riks_continuation(
                |u| u.to_vec(),
                |_, rhs| Ok(rhs.to_vec()),
                &[0.0],
                &[0.0],
                0.0,
                &config
            ),
            Err(NewtonError::InvalidConfig(_))
        ));
        let mut bad = config.clone();
        bad.initial_step = 0.0;
        assert!(matches!(bad.validate(), Err(NewtonError::InvalidConfig(_))));
    }
}
