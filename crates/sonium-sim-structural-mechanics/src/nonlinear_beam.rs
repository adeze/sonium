//! Load-stepped 2D corotational beam cantilever studies.
//!
//! Drives [`CorotationalBeamForm`](sonium_fem::physics::solid::beam::CorotationalBeamForm)
//! with load control and dense direct Newton corrections, and ships the two
//! exact references the validations compare against: the elastica (shooting
//! solution of `EI θ'' = P cos θ`) and the closed-form beam-column tip
//! deflection. Sign convention is y-up everywhere: transverse loads are
//! positive upward (matching [`Beam2DStudy`](super::Beam2DStudy), so downward
//! loads are negative) and axial loads are positive tensile.

use super::{
    Beam2DGeometry, StructuralExecutionProvenance, StructuralMaterial, StructuralMechanicsError,
    StructuralModelAssumption,
};
use serde::{Deserialize, Serialize};
use sonium_fem::assembly::assemble_nonlinear_element_form;
use sonium_fem::physics::solid::beam::{BeamElement2D, BeamSection, CorotationalBeamForm};

// Rust guideline compliant 2026-02-21

/// Dense-solve budget: `(MAX_BEAM_ELEMENTS + 1) * 3` DOFs fit comfortably in
/// the cubic dense Newton correction while resolving elastica rotations.
const MAX_BEAM_ELEMENTS: usize = 64;

/// Cantilever study with tip axial and transverse loads.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NonlinearBeamStudy {
    /// Study name for reports.
    pub name: String,
    /// Span, cross-section, and mesh resolution.
    pub geometry: Beam2DGeometry,
    /// Isotropic material (only Young's modulus enters the static response).
    pub material: StructuralMaterial,
    /// Tip axial load in N, positive tensile (x+ at the tip).
    pub tip_axial_load_n: f64,
    /// Tip transverse load in N, positive upward (y+).
    pub tip_transverse_load_n: f64,
    /// Number of uniform load steps from zero to the full load.
    pub load_steps: usize,
    /// Newton iterations allowed per load step.
    pub max_newton_iterations: usize,
    /// Newton residual tolerance, relative to the tip-load magnitude.
    pub newton_residual_tolerance: f64,
}

impl NonlinearBeamStudy {
    /// Slender introductory cantilever (`h/L = 1/50`, Euler-Bernoulli regime).
    pub fn introductory() -> Self {
        Self {
            name: "introductory-nonlinear-beam".to_string(),
            geometry: Beam2DGeometry {
                length_m: 1.0,
                width_m: 0.02,
                height_m: 0.02,
                elements: 32,
            },
            material: StructuralMaterial::default(),
            tip_axial_load_n: 0.0,
            tip_transverse_load_n: 0.0,
            load_steps: 10,
            max_newton_iterations: 50,
            // Matched to the dense-solver floor: axial/bending stiffness
            // disparity (~1e5 conditioning) stalls Newton corrections at a
            // ~1e-8 relative residual, so 1e-9 is unachievable and 1e-7
            // clears it with margin while holding ~1e-6 solution accuracy.
            newton_residual_tolerance: 1.0e-7,
        }
    }

    /// Bending stiffness `EI` in N m² for the rectangular section.
    pub fn bending_stiffness_nm2(&self) -> f64 {
        self.material.youngs_modulus_pa * self.geometry.second_moment_m4()
    }

    fn validate(&self) -> Result<(), StructuralMechanicsError> {
        for (field, value) in [
            ("nonlinear_beam.length_m", self.geometry.length_m),
            ("nonlinear_beam.width_m", self.geometry.width_m),
            ("nonlinear_beam.height_m", self.geometry.height_m),
            (
                "nonlinear_beam.youngs_modulus_pa",
                self.material.youngs_modulus_pa,
            ),
        ] {
            if !value.is_finite() || value <= 0.0 {
                return Err(StructuralMechanicsError::InvalidConfiguration {
                    field,
                    value: value.to_string(),
                });
            }
        }
        if self.geometry.elements == 0 || self.geometry.elements > MAX_BEAM_ELEMENTS {
            return Err(StructuralMechanicsError::InvalidConfiguration {
                field: "nonlinear_beam.elements",
                value: self.geometry.elements.to_string(),
            });
        }
        for (field, value) in [
            ("nonlinear_beam.tip_axial_load_n", self.tip_axial_load_n),
            (
                "nonlinear_beam.tip_transverse_load_n",
                self.tip_transverse_load_n,
            ),
        ] {
            if !value.is_finite() {
                return Err(StructuralMechanicsError::InvalidConfiguration {
                    field,
                    value: value.to_string(),
                });
            }
        }
        if self.load_steps == 0 {
            return Err(StructuralMechanicsError::InvalidConfiguration {
                field: "nonlinear_beam.load_steps",
                value: "0".to_string(),
            });
        }
        if self.max_newton_iterations == 0 {
            return Err(StructuralMechanicsError::InvalidConfiguration {
                field: "nonlinear_beam.max_newton_iterations",
                value: "0".to_string(),
            });
        }
        if !self.newton_residual_tolerance.is_finite() || self.newton_residual_tolerance <= 0.0 {
            return Err(StructuralMechanicsError::InvalidConfiguration {
                field: "nonlinear_beam.newton_residual_tolerance",
                value: self.newton_residual_tolerance.to_string(),
            });
        }
        Ok(())
    }

    /// Euler-Bernoulli elements on `[0, length]` (shear-free, matching the
    /// elastica and beam-column references, which are EB theories).
    fn elements(&self) -> Vec<BeamElement2D> {
        let section = BeamSection::rectangular(
            self.material.youngs_modulus_pa,
            self.geometry.width_m,
            self.geometry.height_m,
            self.material.density_kg_m3,
        );
        let count = self.geometry.elements;
        let dx = self.geometry.length_m / count as f64;
        (0..count)
            .map(|index| {
                BeamElement2D::new(
                    [index, index + 1],
                    index as f64 * dx,
                    0.0,
                    (index + 1) as f64 * dx,
                    0.0,
                    section,
                )
            })
            .collect()
    }
}

/// Converged state of one load step.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NonlinearBeamStep {
    /// 1-based load step index.
    pub step: usize,
    /// Applied load fraction.
    pub load_factor: f64,
    /// Newton iterations used by this step.
    pub newton_iterations: usize,
    /// Max-abs free residual at convergence in N.
    pub residual_n: f64,
    /// Tip axial displacement in m.
    pub tip_ux_m: f64,
    /// Tip transverse displacement in m.
    pub tip_uy_m: f64,
    /// Tip rotation in rad.
    pub tip_rotation_rad: f64,
}

/// Load-stepped corotational cantilever response.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NonlinearBeamResult {
    /// Echoed study.
    pub study: NonlinearBeamStudy,
    /// Global DOF count.
    pub dofs: usize,
    /// Per-step converged states.
    pub steps: Vec<NonlinearBeamStep>,
    /// Final tip axial displacement in m.
    pub tip_ux_m: f64,
    /// Final tip transverse displacement in m.
    pub tip_uy_m: f64,
    /// Final tip rotation in rad.
    pub tip_rotation_rad: f64,
    /// Newton iterations summed over all steps.
    pub total_newton_iterations: usize,
    /// Final max-abs free residual in N.
    pub final_residual_n: f64,
    /// Whether every load step converged.
    pub converged: bool,
    /// Model assumption tag.
    pub model_assumption: StructuralModelAssumption,
    /// Execution provenance.
    pub execution: StructuralExecutionProvenance,
}

/// Exact elastica tip state from the shooting reference.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ElasticaTip {
    /// Deformed tip x-coordinate in m.
    pub x_m: f64,
    /// Deformed tip y-coordinate in m (negative for downward load).
    pub y_m: f64,
    /// Tip tangent slope in rad (negative for downward load).
    pub tip_slope_rad: f64,
}

/// Solve the load-stepped corotational cantilever.
///
/// Root node DOFs `[0, 1, 2]` are clamped; the tip carries the study loads.
/// Each load step runs Newton from the previous converged state with a dense
/// direct correction on the free DOFs.
pub fn run_nonlinear_beam(
    study: &NonlinearBeamStudy,
) -> Result<NonlinearBeamResult, StructuralMechanicsError> {
    study.validate()?;
    let elements = study.elements();
    let dofs = (study.geometry.elements + 1) * 3;
    let form = CorotationalBeamForm::new(elements, dofs);

    let tip_node = study.geometry.elements;
    let mut external = vec![0.0; dofs];
    external[3 * tip_node] = study.tip_axial_load_n;
    external[3 * tip_node + 1] = study.tip_transverse_load_n;
    let load_scale = study
        .tip_axial_load_n
        .abs()
        .max(study.tip_transverse_load_n.abs());
    // Residual gate scale: the tip-load magnitude, floored so zero-load
    // studies still demand a tight absolute residual.
    let force_scale = load_scale.max(1.0e-300);

    let fixed = [0usize, 1, 2];
    let free: Vec<usize> = (0..dofs).filter(|dof| !fixed.contains(dof)).collect();
    let mut solution = vec![0.0; dofs];
    let mut steps = Vec::with_capacity(study.load_steps);
    let mut total_newton_iterations = 0usize;

    for step in 1..=study.load_steps {
        let load_factor = step as f64 / study.load_steps as f64;
        let mut iterations = 0usize;
        let mut residual = f64::INFINITY;
        let mut converged = false;
        for iteration in 1..=study.max_newton_iterations {
            iterations = iteration;
            let (internal, triplets) = assemble_nonlinear_element_form(&form, &solution);
            residual = free
                .iter()
                .map(|&dof| (internal[dof] - load_factor * external[dof]).abs())
                .fold(0.0, f64::max);
            if residual <= study.newton_residual_tolerance * force_scale {
                converged = true;
                break;
            }
            let mut tangent = vec![0.0; dofs * dofs];
            for ((&row, &column), &value) in triplets
                .rows
                .iter()
                .zip(&triplets.cols)
                .zip(&triplets.values)
            {
                tangent[row * dofs + column] += value;
            }
            let n_free = free.len();
            let mut reduced = vec![0.0; n_free * n_free];
            let mut rhs = vec![0.0; n_free];
            for (row, &dof) in free.iter().enumerate() {
                for (column, &other) in free.iter().enumerate() {
                    reduced[row * n_free + column] = tangent[dof * dofs + other];
                }
                rhs[row] = -(internal[dof] - load_factor * external[dof]);
            }
            let correction = solve_dense_system(reduced, rhs).map_err(|message| {
                StructuralMechanicsError::SolverFailure {
                    stage: "nonlinear beam Newton corrector",
                    message: format!("step {step}, Newton iteration {iterations}: {message}"),
                }
            })?;
            for (column, &dof) in free.iter().enumerate() {
                solution[dof] += correction[column];
            }
            if solution.iter().any(|value| !value.is_finite()) {
                return Err(StructuralMechanicsError::ValidationFailure {
                    check: "nonlinear_beam.newton_state_finite",
                    value: f64::NAN,
                });
            }
        }
        if !converged {
            return Err(StructuralMechanicsError::NonlinearNonConvergence {
                model: StructuralModelAssumption::CorotationalBeam,
                load_step: step,
                iterations,
                residual,
            });
        }
        total_newton_iterations += iterations;
        steps.push(NonlinearBeamStep {
            step,
            load_factor,
            newton_iterations: iterations,
            residual_n: residual,
            tip_ux_m: solution[3 * tip_node],
            tip_uy_m: solution[3 * tip_node + 1],
            tip_rotation_rad: solution[3 * tip_node + 2],
        });
    }

    let final_step = steps.last().expect("at least one load step").clone();
    Ok(NonlinearBeamResult {
        study: study.clone(),
        dofs,
        steps,
        tip_ux_m: final_step.tip_ux_m,
        tip_uy_m: final_step.tip_uy_m,
        tip_rotation_rad: final_step.tip_rotation_rad,
        total_newton_iterations,
        final_residual_n: final_step.residual_n,
        converged: true,
        model_assumption: StructuralModelAssumption::CorotationalBeam,
        execution: StructuralExecutionProvenance::cpu_corotational_beam_reference(),
    })
}

/// Dense partial-pivot Gaussian elimination for the Newton correction.
///
/// Beam systems are tiny (`<= 195` DOFs at the element cap); a direct dense
/// solve avoids any iterative-solver tuning inside Newton.
fn solve_dense_system(mut matrix: Vec<f64>, mut rhs: Vec<f64>) -> Result<Vec<f64>, String> {
    let dimension = rhs.len();
    if matrix.len() != dimension * dimension {
        return Err(format!(
            "tangent has {} entries for {dimension} RHS entries",
            matrix.len()
        ));
    }
    for pivot in 0..dimension {
        let (pivot_row, pivot_value) = (pivot..dimension)
            .map(|row| (row, matrix[row * dimension + pivot].abs()))
            .max_by(|(_, left), (_, right)| left.total_cmp(right))
            .ok_or_else(|| "empty tangent matrix".to_string())?;
        if !pivot_value.is_finite() || pivot_value <= 1.0e-14 {
            return Err(format!("singular Newton tangent at pivot {pivot}"));
        }
        if pivot_row != pivot {
            for column in pivot..dimension {
                matrix.swap(pivot * dimension + column, pivot_row * dimension + column);
            }
            rhs.swap(pivot, pivot_row);
        }
        let diagonal = matrix[pivot * dimension + pivot];
        for row in pivot + 1..dimension {
            let factor = matrix[row * dimension + pivot] / diagonal;
            matrix[row * dimension + pivot] = 0.0;
            for column in pivot + 1..dimension {
                matrix[row * dimension + column] -= factor * matrix[pivot * dimension + column];
            }
            rhs[row] -= factor * rhs[pivot];
        }
    }
    let mut solution = vec![0.0; dimension];
    for row in (0..dimension).rev() {
        let mut value = rhs[row];
        for column in row + 1..dimension {
            value -= matrix[row * dimension + column] * solution[column];
        }
        solution[row] = value / matrix[row * dimension + row];
    }
    if solution.iter().any(|value| !value.is_finite()) {
        return Err("Newton tangent produced a non-finite correction".to_string());
    }
    Ok(solution)
}

/// Euler buckling load of a cantilever, `P_cr = π²EI / 4L²`.
pub fn euler_cantilever_buckling_load(length_m: f64, bending_stiffness_nm2: f64) -> f64 {
    std::f64::consts::PI.powi(2) * bending_stiffness_nm2 / (4.0 * length_m * length_m)
}

/// Closed-form tip deflection of a cantilever beam-column.
///
/// Cantilever clamped at `x = 0`, compressive axial load `compressive_n`,
/// downward transverse tip load `transverse_magnitude_n`. With
/// `k² = P/EI`, `v(L) = −Q (tan kL − kL) / Pk` (Timoshenko beam-column
/// theory); the `P → 0` limit recovers `−QL³/3EI` through the linear
/// branch below. Rejects `kL ≥ π/2` (at or above the Euler load, where the
/// tangent stiffness is singular and the formula diverges).
///
/// # Errors
///
/// Returns [`StructuralMechanicsError::InvalidConfiguration`] for
/// non-positive geometry/stiffness, negative loads, or axial loads at or
/// above the Euler buckling load.
pub fn beam_column_cantilever_tip(
    compressive_n: f64,
    transverse_magnitude_n: f64,
    length_m: f64,
    bending_stiffness_nm2: f64,
) -> Result<f64, StructuralMechanicsError> {
    for (field, value) in [
        ("beam_column.length_m", length_m),
        ("beam_column.bending_stiffness_nm2", bending_stiffness_nm2),
    ] {
        if !value.is_finite() || value <= 0.0 {
            return Err(StructuralMechanicsError::InvalidConfiguration {
                field,
                value: value.to_string(),
            });
        }
    }
    for (field, value) in [
        ("beam_column.compressive_n", compressive_n),
        ("beam_column.transverse_magnitude_n", transverse_magnitude_n),
    ] {
        if !value.is_finite() || value < 0.0 {
            return Err(StructuralMechanicsError::InvalidConfiguration {
                field,
                value: value.to_string(),
            });
        }
    }
    let linear_tip = -transverse_magnitude_n * length_m.powi(3) / (3.0 * bending_stiffness_nm2);
    // Below kL = 1e-4 the tan-series truncation differs from the linear
    // limit by O((kL)²) ~ 1e-8 relative; take the limit directly.
    if compressive_n == 0.0 {
        return Ok(linear_tip);
    }
    let kl = (compressive_n / bending_stiffness_nm2).sqrt() * length_m;
    if kl < 1.0e-4 {
        return Ok(linear_tip);
    }
    if kl >= std::f64::consts::FRAC_PI_2 {
        return Err(StructuralMechanicsError::InvalidConfiguration {
            field: "beam_column.compressive_n",
            value: compressive_n.to_string(),
        });
    }
    Ok(-transverse_magnitude_n * (kl.tan() - kl) / (compressive_n * kl / length_m))
}

/// Exact elastica tip state of a cantilever under a downward tip load.
///
/// Solves `EI θ'' = P cos θ` with clamped `θ(0) = 0` and moment-free
/// `θ'(L) = 0` by shooting (RK4 + bisection on the root curvature), then
/// integrates the tip position. At `P → 0` this recovers the linear
/// `−PL³/3EI` tip deflection; the reference test pins that limit.
///
/// # Errors
///
/// Returns [`StructuralMechanicsError::InvalidConfiguration`] for
/// non-positive geometry/stiffness or a negative load magnitude.
pub fn elastica_cantilever_tip(
    transverse_magnitude_n: f64,
    length_m: f64,
    bending_stiffness_nm2: f64,
) -> Result<ElasticaTip, StructuralMechanicsError> {
    if !transverse_magnitude_n.is_finite() || transverse_magnitude_n < 0.0 {
        return Err(StructuralMechanicsError::InvalidConfiguration {
            field: "elastica.transverse_magnitude_n",
            value: transverse_magnitude_n.to_string(),
        });
    }
    for (field, value) in [
        ("elastica.length_m", length_m),
        ("elastica.bending_stiffness_nm2", bending_stiffness_nm2),
    ] {
        if !value.is_finite() || value <= 0.0 {
            return Err(StructuralMechanicsError::InvalidConfiguration {
                field,
                value: value.to_string(),
            });
        }
    }
    if transverse_magnitude_n == 0.0 {
        return Ok(ElasticaTip {
            x_m: length_m,
            y_m: 0.0,
            tip_slope_rad: 0.0,
        });
    }
    let stiffness = transverse_magnitude_n / bending_stiffness_nm2;
    // Residual of the root-curvature guess: integrate θ'' = (P/EI) cos θ
    // and report the tip curvature (zero for the exact root curvature).
    let tip_curvature = |root_curvature: f64| -> (f64, f64, f64, f64) {
        const RK_STEPS: usize = 2048;
        let ds = length_m / RK_STEPS as f64;
        let mut theta = 0.0f64;
        let mut curvature = root_curvature;
        let mut x = 0.0f64;
        let mut y = 0.0f64;
        for _ in 0..RK_STEPS {
            // RK4 on (θ, κ) with κ' = stiffness · cos θ; trapezoid the
            // tip position on the same grid.
            let cos_0 = theta.cos();
            let k1_theta = curvature;
            let k1_kappa = stiffness * cos_0;
            let theta_1 = theta + 0.5 * ds * k1_theta;
            let k2_theta = curvature + 0.5 * ds * k1_kappa;
            let k2_kappa = stiffness * theta_1.cos();
            let theta_2 = theta + 0.5 * ds * k2_theta;
            let k3_theta = curvature + 0.5 * ds * k2_kappa;
            let k3_kappa = stiffness * theta_2.cos();
            let theta_3 = theta + ds * k3_theta;
            let k4_theta = curvature + ds * k3_kappa;
            let k4_kappa = stiffness * theta_3.cos();
            let theta_end =
                theta + ds * (k1_theta + 2.0 * k2_theta + 2.0 * k3_theta + k4_theta) / 6.0;
            x += 0.5 * ds * (cos_0 + theta_end.cos());
            y += 0.5 * ds * (theta.sin() + theta_end.sin());
            theta = theta_end;
            curvature += ds * (k1_kappa + 2.0 * k2_kappa + 2.0 * k3_kappa + k4_kappa) / 6.0;
        }
        (curvature, x, y, theta)
    };
    // Bracket: κ = 0 overshoots (positive tip curvature); doubling the
    // linear root-curvature estimate downward always crosses the root.
    let linear_root = -transverse_magnitude_n * length_m / bending_stiffness_nm2;
    let mut lo = linear_root;
    let mut hi = 0.0;
    let mut residual_hi = tip_curvature(hi).0;
    if residual_hi <= 0.0 {
        return Err(StructuralMechanicsError::SolverFailure {
            stage: "elastica shooting bracket",
            message: "zero root curvature must overshoot".to_string(),
        });
    }
    for _ in 0..64 {
        let residual_lo = tip_curvature(lo).0;
        if residual_lo <= 0.0 {
            break;
        }
        lo *= 2.0;
    }
    let mut root_curvature = lo;
    for _ in 0..100 {
        let mid = 0.5 * (lo + hi);
        let residual_mid = tip_curvature(mid).0;
        if residual_mid > 0.0 {
            hi = mid;
            residual_hi = residual_mid;
        } else {
            lo = mid;
        }
        root_curvature = mid;
        if residual_hi.abs() <= 1.0e-13 * linear_root.abs().max(1.0e-300) {
            break;
        }
    }
    let (_, x, y, theta) = tip_curvature(root_curvature);
    Ok(ElasticaTip {
        x_m: x,
        y_m: y,
        tip_slope_rad: theta,
    })
}
