//! Bounded pseudo-arclength post-buckling reference.
//!
//! This module is an analytic scalar softening-arch continuation fixture. It
//! follows the equilibrium curve
//! `k u - a u^3 - P lambda = 0` through its limit point with a
//! predictor/corrector pseudo-arclength step. It is useful for checking
//! turning-point and snap-back contracts, but is not a product-scale shell or
//! solid post-buckling solver.
//!
//! Fixture scope: the terminal window ends at the zero-load crossing.
//! Studies continuing past it (restiffening, second limit point) are
//! rejected even though the arclength stepper could follow them; extend
//! with an explicit branch policy if deep snap-back is ever needed.

use crate::{StructuralExecutionProvenance, StructuralMechanicsError};
use serde::{Deserialize, Serialize};

/// Schema version for the bounded post-buckling result.
pub const POST_BUCKLING_SCHEMA_VERSION: u32 = 1;

/// Scalar softening-arch continuation inputs.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct PostBucklingStudy {
    pub stiffness_n_per_m: f64,
    pub softening_n_per_m3: f64,
    pub reference_load_n: f64,
    pub terminal_displacement_m: f64,
    pub arc_length_step: f64,
    pub max_steps: usize,
    pub max_newton_iterations: usize,
    pub residual_tolerance_n: f64,
}

impl PostBucklingStudy {
    /// Compact path that crosses the limit point and remains on the positive-load branch.
    pub fn introductory() -> Self {
        Self {
            stiffness_n_per_m: 1_000.0,
            softening_n_per_m3: 20_000.0,
            reference_load_n: 100.0,
            terminal_displacement_m: 0.20,
            arc_length_step: 0.005,
            max_steps: 600,
            max_newton_iterations: 20,
            residual_tolerance_n: 1.0e-10,
        }
    }

    fn limit_point_displacement_m(&self) -> f64 {
        (self.stiffness_n_per_m / (3.0 * self.softening_n_per_m3)).sqrt()
    }

    fn zero_load_displacement_m(&self) -> f64 {
        (self.stiffness_n_per_m / self.softening_n_per_m3).sqrt()
    }

    fn validate(&self) -> Result<(), StructuralMechanicsError> {
        for (field, value) in [
            ("post_buckling.stiffness_n_per_m", self.stiffness_n_per_m),
            ("post_buckling.softening_n_per_m3", self.softening_n_per_m3),
            ("post_buckling.reference_load_n", self.reference_load_n),
            (
                "post_buckling.terminal_displacement_m",
                self.terminal_displacement_m,
            ),
            ("post_buckling.arc_length_step", self.arc_length_step),
            (
                "post_buckling.residual_tolerance_n",
                self.residual_tolerance_n,
            ),
        ] {
            if !value.is_finite() || value <= 0.0 {
                return Err(StructuralMechanicsError::InvalidConfiguration {
                    field,
                    value: value.to_string(),
                });
            }
        }
        if self.max_steps == 0 || self.max_newton_iterations == 0 {
            return Err(StructuralMechanicsError::InvalidConfiguration {
                field: "post_buckling.discretization",
                value: format!(
                    "max_steps={}, max_newton_iterations={}",
                    self.max_steps, self.max_newton_iterations
                ),
            });
        }

        let limit = self.limit_point_displacement_m();
        let zero_load = self.zero_load_displacement_m();
        if self.terminal_displacement_m <= limit || self.terminal_displacement_m >= zero_load {
            return Err(StructuralMechanicsError::InvalidConfiguration {
                field: "post_buckling.terminal_displacement_m",
                value: format!(
                    "{} must be between limit point {} and zero-load displacement {}",
                    self.terminal_displacement_m, limit, zero_load
                ),
            });
        }
        Ok(())
    }
}

/// Branch label for a continuation point.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PostBucklingBranch {
    PrePeak,
    SnapBack,
}

/// Physical idealisation used by this reference.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PostBucklingModelAssumption {
    ScalarSofteningArch,
}

/// One converged pseudo-arclength equilibrium point.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct PostBucklingPoint {
    pub step: usize,
    pub displacement_m: f64,
    pub load_factor: f64,
    pub load_n: f64,
    pub equilibrium_residual_n: f64,
    pub arc_constraint_residual: f64,
    pub newton_iterations: usize,
    pub tangent_displacement_per_arc: f64,
    pub tangent_load_factor_per_arc: f64,
    pub branch: PostBucklingBranch,
}

/// Versioned bounded post-buckling continuation result.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PostBucklingResult {
    pub schema_version: u32,
    pub study: PostBucklingStudy,
    pub points: Vec<PostBucklingPoint>,
    pub turning_point_step: usize,
    pub peak_load_factor: f64,
    pub terminal_displacement_m: f64,
    pub terminal_load_factor: f64,
    pub max_equilibrium_residual_n: f64,
    pub max_arc_constraint_residual: f64,
    pub model_assumption: PostBucklingModelAssumption,
    pub execution: StructuralExecutionProvenance,
}

fn equilibrium(study: &PostBucklingStudy, displacement_m: f64, load_factor: f64) -> f64 {
    study.stiffness_n_per_m * displacement_m
        - study.softening_n_per_m3 * displacement_m.powi(3)
        - study.reference_load_n * load_factor
}

fn tangent(study: &PostBucklingStudy, displacement_m: f64) -> (f64, f64) {
    let dload_du = (study.stiffness_n_per_m
        - 3.0 * study.softening_n_per_m3 * displacement_m.powi(2))
        / study.reference_load_n;
    let norm = (1.0 + dload_du * dload_du).sqrt();
    (1.0 / norm, dload_du / norm)
}

/// Run the bounded scalar pseudo-arclength post-buckling reference.
pub fn run_post_buckling(
    study: &PostBucklingStudy,
) -> Result<PostBucklingResult, StructuralMechanicsError> {
    study.validate()?;

    let mut points = Vec::with_capacity(study.max_steps + 1);
    let mut displacement_m = 0.0;
    let mut load_factor = 0.0;
    let mut path_tangent = tangent(study, displacement_m);
    let mut turning_point_step = None;
    let mut peak_load_factor = load_factor;
    let mut max_equilibrium_residual_n: f64 = 0.0;
    let mut max_arc_constraint_residual: f64 = 0.0;

    points.push(PostBucklingPoint {
        step: 0,
        displacement_m,
        load_factor,
        load_n: 0.0,
        equilibrium_residual_n: 0.0,
        arc_constraint_residual: 0.0,
        newton_iterations: 0,
        tangent_displacement_per_arc: path_tangent.0,
        tangent_load_factor_per_arc: path_tangent.1,
        branch: PostBucklingBranch::PrePeak,
    });

    for step in 1..=study.max_steps {
        let predicted_displacement = displacement_m + study.arc_length_step * path_tangent.0;
        let predicted_load = load_factor + study.arc_length_step * path_tangent.1;
        let mut trial_displacement = predicted_displacement;
        let mut trial_load = predicted_load;
        let mut converged = false;
        let mut equilibrium_residual_n = f64::INFINITY;
        let mut arc_constraint_residual = f64::INFINITY;
        let mut iterations = 0;

        for iteration in 1..=study.max_newton_iterations {
            let residual = equilibrium(study, trial_displacement, trial_load);
            let arc_residual = (trial_displacement - predicted_displacement) * path_tangent.0
                + (trial_load - predicted_load) * path_tangent.1;
            equilibrium_residual_n = residual.abs();
            arc_constraint_residual = arc_residual.abs();
            iterations = iteration;
            if equilibrium_residual_n <= study.residual_tolerance_n
                && arc_constraint_residual
                    <= study.residual_tolerance_n / study.reference_load_n.max(1.0)
            {
                converged = true;
                break;
            }

            let d_residual_du = study.stiffness_n_per_m
                - 3.0 * study.softening_n_per_m3 * trial_displacement.powi(2);
            let determinant =
                d_residual_du * path_tangent.1 + study.reference_load_n * path_tangent.0;
            if !determinant.is_finite() || determinant.abs() <= f64::EPSILON {
                break;
            }
            let delta_displacement =
                (-residual * path_tangent.1 - study.reference_load_n * arc_residual) / determinant;
            let delta_load =
                (path_tangent.0 * residual - d_residual_du * arc_residual) / determinant;
            trial_displacement += delta_displacement;
            trial_load += delta_load;
            if !trial_displacement.is_finite() || !trial_load.is_finite() {
                break;
            }
        }

        if !converged {
            return Err(StructuralMechanicsError::NonConvergence {
                iterations,
                residual: equilibrium_residual_n
                    .max(arc_constraint_residual * study.reference_load_n),
            });
        }

        let branch = if trial_load < load_factor {
            turning_point_step.get_or_insert(step.saturating_sub(1));
            PostBucklingBranch::SnapBack
        } else {
            PostBucklingBranch::PrePeak
        };
        path_tangent = tangent(study, trial_displacement);
        points.push(PostBucklingPoint {
            step,
            displacement_m: trial_displacement,
            load_factor: trial_load,
            load_n: trial_load * study.reference_load_n,
            equilibrium_residual_n,
            arc_constraint_residual,
            newton_iterations: iterations,
            tangent_displacement_per_arc: path_tangent.0,
            tangent_load_factor_per_arc: path_tangent.1,
            branch,
        });
        max_equilibrium_residual_n = max_equilibrium_residual_n.max(equilibrium_residual_n);
        max_arc_constraint_residual = max_arc_constraint_residual.max(arc_constraint_residual);
        peak_load_factor = peak_load_factor.max(trial_load);
        displacement_m = trial_displacement;
        load_factor = trial_load;

        if displacement_m >= study.terminal_displacement_m {
            break;
        }
    }

    let turning_point_step =
        turning_point_step.ok_or(StructuralMechanicsError::NonConvergence {
            iterations: points.len().saturating_sub(1),
            residual: peak_load_factor,
        })?;
    if displacement_m < study.terminal_displacement_m {
        return Err(StructuralMechanicsError::NonConvergence {
            iterations: points.len().saturating_sub(1),
            residual: study.terminal_displacement_m - displacement_m,
        });
    }

    let mut execution = StructuralExecutionProvenance::cpu_reference();
    execution.capability_id =
        "sonium-structural.cpu-reference.f64.post-buckling-scalar.v1".to_string();
    execution.limitations.extend([
        "Scalar softening-arch equilibrium only; no shell/solid mesh, geometric imperfection field, or material plasticity is represented.".to_string(),
        "Pseudo-arclength diagnostics close a turning-point contract but do not establish product-scale, measured, NAFEMS, or external-solver parity.".to_string(),
    ]);

    Ok(PostBucklingResult {
        schema_version: POST_BUCKLING_SCHEMA_VERSION,
        study: *study,
        terminal_displacement_m: displacement_m,
        terminal_load_factor: load_factor,
        turning_point_step,
        peak_load_factor,
        max_equilibrium_residual_n,
        max_arc_constraint_residual,
        points,
        model_assumption: PostBucklingModelAssumption::ScalarSofteningArch,
        execution,
    })
}
