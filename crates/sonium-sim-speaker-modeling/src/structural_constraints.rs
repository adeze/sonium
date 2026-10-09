//! Optimization-ready structural constraint evaluators (review item C3).
//!
//! These evaluators turn existing simulation outputs into scalar margins
//! suitable for `sonium_solvers::optimization::Constraint` inequalities
//! (`c(x) ≥ 0` feasible). They contain no new physics — only post-processing
//! of solved fields against user-supplied limits:
//!
//! - **Moving-assembly excursion** vs Xmax: peak one-way axial displacement
//!   over the moving-assembly nodes of the axisymmetric distributed model.
//! - **Recovered stress** vs allowables: per-region peak von Mises stress
//!   from [`structural_stress_cells`], which uses each region's own material
//!   (see [`crate::axisymmetric_linear::moving_assembly_material`]).
//! - **Cabinet panel response** vs limits: peak panel-center displacement and
//!   bending stress from the simply-supported plate modal model in
//!   `sonium-fem` (`cabinet_radiation`), driven by the internal cavity
//!   pressure (e.g. `ResolvedAcousticSolution::cavity_pressure_rms_pa`).
//!
//! # Conventions
//!
//! All phasors in this crate are **RMS** (see the crate-level docs). Xmax and
//! panel displacement limits are **peak, one-way** quantities, so every
//! evaluator converts with `peak = √2 · RMS`:
//!
//! - excursion: `x_peak = √2 · max_nodes |u_z,rms|`
//! - stress: `σ_vm,peak = √2 · envelope(σ_rms)`, using the conservative
//!   phase-agnostic triangle-inequality envelope; the exported complex von
//!   Mises field remains available for phase-aligned diagnostics.
//! - panel displacement: `w_peak = √2 · |Σ_mn s_mn X_mn|` with
//!   `s_mn = sin(mπ/2)·sin(nπ/2)` the mode-shape value at the panel center.
//!   Only odd-odd modes are excited by uniform pressure, and this sum is the
//!   exact center displacement of the modal expansion.
//! - panel bending stress: `σ_x = Σ 6D/h²·((mπ/a)² + ν(nπ/b)²)·s_mn X_mn`
//!   (and `σ_y` by symmetry) at the panel center — the thin-plate relation
//!   `σ = 6M/h²` with `M = D(∂²w/∂x² + ν∂²w/∂y²)`. The reported metric is
//!   `√2 · max(|σ_x|, |σ_y|)`.
//!
//! Region keys for stress allowables are the moving-assembly cell markers
//! ([`DIAPHRAGM`], [`COIL_FORMER`], [`SURROUND`], [`SPIDER`],
//! [`COIL_WINDING`]); [`moving_assembly_region_name`] maps them to the
//! physical-group names used by the golden model.
//!
//! Cabinet bracing geometry is validated and lowered by
//! [`crate::cabinet_design::CabinetDesign`]; these evaluators add the panel
//! displacement and bending-stress margins driven by the resolved cavity.

use std::collections::BTreeMap;

use num_complex::Complex64;
use serde::{Deserialize, Serialize};
use sonium_fem::mesh::Mesh;
use sonium_fem::physics::acoustics::electroacoustics::cabinet_radiation::{
    Cabinet, CabinetPanel, PanelModeDisplacement,
};
use sonium_solvers_optimization::{Constraint, ConstraintKind};

use crate::axisymmetric_linear::{
    AxisymmetricLinearSolution, COIL_FORMER, COIL_WINDING, DIAPHRAGM, MOVING_ASSEMBLY_MARKERS,
    SPIDER, SURROUND, conservative_von_mises, structural_stress_cells,
};
use crate::{ModelingError, require_positive};

/// RMS → peak (one-way amplitude) conversion for harmonic phasors.
const RMS_TO_PEAK: f64 = std::f64::consts::SQRT_2;

/// Margin returned for a metric the evaluation did not populate. A missing
/// metric is treated as infeasible (fail-safe) so the optimizer backs away
/// instead of walking through an unevaluated region.
pub const MISSING_METRIC_MARGIN: f64 = -1.0;

/// Errors from structural constraint evaluation.
#[derive(Debug, thiserror::Error)]
pub enum StructuralConstraintError {
    #[error("structural mesh contains no moving-assembly cells")]
    MissingMovingAssembly,
    #[error("solution field layout does not match the structural mesh node count")]
    MeshMismatch,
    #[error("cavity pressure sweep must contain at least one frequency")]
    EmptySweep,
}

/// Moving-assembly limits consumed by the constraint evaluators.
///
/// `xmax_m` is the one-way linear excursion limit; when the driver comes from
/// a [`crate::MovingSystemSpec`] it typically mirrors
/// `MovingSystemSpec::xmax_m`. `stress_allowables_pa` maps a moving-assembly
/// region marker to its peak von Mises allowable (Pa).
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct MovingAssemblyLimits {
    /// One-way excursion limit Xmax (m). `None` = unconstrained.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub xmax_m: Option<f64>,
    /// Peak von Mises stress allowable per region marker (Pa).
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub stress_allowables_pa: BTreeMap<i32, f64>,
}

impl MovingAssemblyLimits {
    pub fn validate(&self) -> Result<(), ModelingError> {
        if let Some(xmax_m) = self.xmax_m {
            require_positive("moving_assembly_limits.xmax_m", xmax_m)?;
        }
        for allowable in self.stress_allowables_pa.values() {
            require_positive("moving_assembly_limits.stress_allowables_pa", *allowable)?;
        }
        Ok(())
    }
}

/// Per-panel cabinet limits consumed by the constraint evaluators.
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
pub struct CabinetPanelLimits {
    /// Peak panel-center displacement limit (m). `None` = unconstrained.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_panel_displacement_m: Option<f64>,
    /// Peak panel-center bending stress limit (Pa). `None` = unconstrained.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_bending_stress_pa: Option<f64>,
}

impl CabinetPanelLimits {
    pub fn validate(&self) -> Result<(), ModelingError> {
        if let Some(limit) = self.max_panel_displacement_m {
            require_positive("cabinet_panel_limits.max_panel_displacement_m", limit)?;
        }
        if let Some(limit) = self.max_bending_stress_pa {
            require_positive("cabinet_panel_limits.max_bending_stress_pa", limit)?;
        }
        Ok(())
    }
}

/// Physical-group name of a moving-assembly region marker, matching the names
/// registered by the golden model's structural mesh.
pub fn moving_assembly_region_name(marker: i32) -> Option<&'static str> {
    match marker {
        DIAPHRAGM => Some("moving diaphragm"),
        COIL_FORMER => Some("coil former"),
        SURROUND => Some("surround"),
        SPIDER => Some("spider"),
        COIL_WINDING => Some("voice coil winding"),
        _ => None,
    }
}

/// Scalar structural metrics consumed by [`to_solver_constraints`].
///
/// Build it with [`StructuralEvaluation::from_axisymmetric_sweep`] and
/// optionally [`StructuralEvaluation::with_cabinet_panels`], or populate the
/// fields directly from any other source (e.g. a lumped response or a cached
/// optimizer evaluation). Every entry that corresponds to an active limit
/// must be populated; a missing metric yields [`MISSING_METRIC_MARGIN`].
#[derive(Debug, Clone, Default)]
pub struct StructuralEvaluation {
    /// Max over frequency of the peak one-way excursion (m).
    pub peak_excursion_m: Option<f64>,
    /// Max over frequency of the peak von Mises stress per region marker (Pa).
    pub peak_stress_pa: BTreeMap<i32, f64>,
    /// Per-panel max over frequency of the peak center displacement (m).
    pub panel_peak_displacement_m: Vec<f64>,
    /// Per-panel max over frequency of the peak center bending stress (Pa).
    pub panel_peak_bending_stress_pa: Vec<f64>,
}

impl StructuralEvaluation {
    /// Evaluate excursion and per-region stress over a solved axisymmetric
    /// frequency sweep.
    pub fn from_axisymmetric_sweep(
        solutions: &[AxisymmetricLinearSolution],
        structural_mesh: &Mesh,
    ) -> Result<Self, StructuralConstraintError> {
        let excursion = moving_assembly_excursion_m(solutions, structural_mesh)?;
        let peak_stress_pa = moving_assembly_peak_stress_pa(solutions, structural_mesh)?;
        Ok(Self {
            peak_excursion_m: excursion.into_iter().reduce(f64::max),
            peak_stress_pa,
            panel_peak_displacement_m: Vec::new(),
            panel_peak_bending_stress_pa: Vec::new(),
        })
    }

    /// Append the cabinet-panel metrics driven by the internal cavity
    /// pressure sweep `(frequency_hz, pressure_rms_pa)`.
    pub fn with_cabinet_panels(
        mut self,
        cavity_pressure_rms_pa: &[(f64, Complex64)],
        cabinet: &Cabinet,
        max_mode_order: usize,
    ) -> Result<Self, StructuralConstraintError> {
        let evaluations = cabinet_panel_response(cavity_pressure_rms_pa, cabinet, max_mode_order)?;
        self.panel_peak_displacement_m = evaluations
            .iter()
            .map(|evaluation| evaluation.peak_displacement_m)
            .collect();
        self.panel_peak_bending_stress_pa = evaluations
            .iter()
            .map(|evaluation| evaluation.peak_bending_stress_pa)
            .collect();
        Ok(self)
    }
}

/// Per-panel worst-case response over a cavity pressure sweep.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CabinetPanelEvaluation {
    /// Index into `Cabinet::panels`.
    pub panel_index: usize,
    /// Frequency at which the peak displacement occurred (Hz).
    pub worst_frequency_hz: f64,
    /// Max over frequency of the peak panel-center displacement (m).
    pub peak_displacement_m: f64,
    /// Max over frequency of the peak panel-center bending stress (Pa).
    pub peak_bending_stress_pa: f64,
}

/// Nodes belonging to the moving-assembly regions of the structural mesh.
fn moving_assembly_nodes(mesh: &Mesh) -> Vec<usize> {
    let mut is_moving = vec![false; mesh.num_nodes()];
    for &marker in &MOVING_ASSEMBLY_MARKERS {
        for cell in mesh.cells_with_marker(marker) {
            for &node in mesh.elements[cell].vertices() {
                is_moving[node] = true;
            }
        }
    }
    (0..mesh.num_nodes())
        .filter(|&node| is_moving[node])
        .collect()
}

fn check_field_layout(
    solutions: &[AxisymmetricLinearSolution],
    structural_mesh: &Mesh,
) -> Result<(), StructuralConstraintError> {
    if solutions
        .iter()
        .any(|solution| solution.displacement_rz_m.len() != structural_mesh.num_nodes())
    {
        return Err(StructuralConstraintError::MeshMismatch);
    }
    Ok(())
}

/// Per-frequency peak one-way excursion (m) of the moving assembly:
/// `√2 · max |u_z,rms|` over the moving-assembly nodes.
pub fn moving_assembly_excursion_m(
    solutions: &[AxisymmetricLinearSolution],
    structural_mesh: &Mesh,
) -> Result<Vec<f64>, StructuralConstraintError> {
    let nodes = moving_assembly_nodes(structural_mesh);
    if nodes.is_empty() && !solutions.is_empty() {
        return Err(StructuralConstraintError::MissingMovingAssembly);
    }
    check_field_layout(solutions, structural_mesh)?;
    Ok(solutions
        .iter()
        .map(|solution| {
            RMS_TO_PEAK
                * nodes
                    .iter()
                    .map(|&node| solution.displacement_rz_m[node][1].norm())
                    .fold(0.0, f64::max)
        })
        .collect())
}

/// Max over the sweep of the per-region peak von Mises stress (Pa):
/// `√2 · conservative_von_mises(σ_rms)` per cell, reduced per region marker.
pub fn moving_assembly_peak_stress_pa(
    solutions: &[AxisymmetricLinearSolution],
    structural_mesh: &Mesh,
) -> Result<BTreeMap<i32, f64>, StructuralConstraintError> {
    check_field_layout(solutions, structural_mesh)?;
    let mut peak: BTreeMap<i32, f64> = BTreeMap::new();
    for solution in solutions {
        let stresses = structural_stress_cells(structural_mesh, &solution.displacement_rz_m);
        for (element, stress) in structural_mesh.elements.iter().zip(&stresses) {
            if !MOVING_ASSEMBLY_MARKERS.contains(&element.marker) {
                continue;
            }
            let value = RMS_TO_PEAK * conservative_von_mises(*stress);
            peak.entry(element.marker)
                .and_modify(|peak| *peak = peak.max(value))
                .or_insert(value);
        }
    }
    Ok(peak)
}

/// Peak panel-center displacement and bending stress of one panel at one
/// frequency, from its modal displacements (RMS phasors).
fn panel_center_response(panel: &CabinetPanel, modes: &[PanelModeDisplacement]) -> (f64, f64) {
    let rigidity = panel.material.flexural_rigidity();
    let thickness = panel.material.thickness;
    let nu = panel.material.poissons_ratio;
    let (a, b) = (panel.width, panel.height);
    let section_factor = 6.0 * rigidity / (thickness * thickness);

    let mut center = Complex64::new(0.0, 0.0);
    let mut sigma_x = Complex64::new(0.0, 0.0);
    let mut sigma_y = Complex64::new(0.0, 0.0);
    for mode in modes {
        // Mode-shape value at the panel center: ±1 for odd m, n.
        let sign = ((mode.m as f64) * std::f64::consts::FRAC_PI_2).sin()
            * ((mode.n as f64) * std::f64::consts::FRAC_PI_2).sin();
        let displacement = mode.displacement_m * sign;
        center += displacement;
        let k_m2 = (mode.m as f64 * std::f64::consts::PI / a).powi(2);
        let k_n2 = (mode.n as f64 * std::f64::consts::PI / b).powi(2);
        sigma_x += section_factor * (k_m2 + nu * k_n2) * displacement;
        sigma_y += section_factor * (nu * k_m2 + k_n2) * displacement;
    }
    (
        RMS_TO_PEAK * center.norm(),
        RMS_TO_PEAK * sigma_x.norm().max(sigma_y.norm()),
    )
}

/// Worst-case panel response over a cavity pressure sweep.
///
/// `cavity_pressure_rms_pa` is `(frequency_hz, pressure_rms_pa)` per sweep
/// point — e.g. collected from
/// `ResolvedAcousticSolution::cavity_pressure_rms_pa`. The retained panel
/// modes span up to `10 × max(max sweep frequency, panel fundamental)` so the
/// quasi-static deflection shape is converged; `max_mode_order` caps the
/// per-axis modal order.
pub fn cabinet_panel_response(
    cavity_pressure_rms_pa: &[(f64, Complex64)],
    cabinet: &Cabinet,
    max_mode_order: usize,
) -> Result<Vec<CabinetPanelEvaluation>, StructuralConstraintError> {
    if cavity_pressure_rms_pa.is_empty() {
        return Err(StructuralConstraintError::EmptySweep);
    }
    let max_frequency = cavity_pressure_rms_pa
        .iter()
        .map(|(frequency, _)| *frequency)
        .fold(0.0, f64::max);
    cabinet
        .panels
        .iter()
        .enumerate()
        .map(|(panel_index, panel)| {
            let cutoff = 10.0 * max_frequency.max(panel.mode_frequency(1, 1));
            let mut evaluation = CabinetPanelEvaluation {
                panel_index,
                worst_frequency_hz: 0.0,
                peak_displacement_m: 0.0,
                peak_bending_stress_pa: 0.0,
            };
            for &(frequency, pressure) in cavity_pressure_rms_pa {
                let modes = panel.modal_displacements(frequency, pressure, cutoff, max_mode_order);
                let (displacement, stress) = panel_center_response(panel, &modes);
                if displacement > evaluation.peak_displacement_m {
                    evaluation.peak_displacement_m = displacement;
                    evaluation.worst_frequency_hz = frequency;
                }
                evaluation.peak_bending_stress_pa = evaluation.peak_bending_stress_pa.max(stress);
            }
            Ok(evaluation)
        })
        .collect()
}

/// Cabinet panel limits plus the number of panels the evaluation covers.
pub type CabinetConstraintSpec = (CabinetPanelLimits, usize);

/// Normalized margins (`limit − value ≥ 0` feasible) for every active limit,
/// ordered as [`constraint_labels`]. A metric missing from the evaluation
/// contributes [`MISSING_METRIC_MARGIN`].
pub fn constraint_margins(
    evaluation: &StructuralEvaluation,
    moving: &MovingAssemblyLimits,
    cabinet: Option<&CabinetConstraintSpec>,
) -> Vec<f64> {
    // Constraint callbacks are an optimizer safety boundary: a NaN metric
    // must never leak through as a NaN margin, because many optimizers treat
    // unordered comparisons as feasible or simply propagate the NaN. Keep
    // the finite negative sentinel for both missing and invalid measurements.
    let margin = |limit: f64, metric: Option<f64>| match metric {
        Some(value) if limit.is_finite() && value.is_finite() => {
            let margin = limit - value;
            if margin.is_finite() {
                margin
            } else {
                MISSING_METRIC_MARGIN
            }
        }
        _ => MISSING_METRIC_MARGIN,
    };
    let mut margins = Vec::new();
    if let Some(xmax_m) = moving.xmax_m {
        margins.push(margin(xmax_m, evaluation.peak_excursion_m));
    }
    for (marker, allowable) in &moving.stress_allowables_pa {
        margins.push(margin(
            *allowable,
            evaluation.peak_stress_pa.get(marker).copied(),
        ));
    }
    if let Some((limits, panel_count)) = cabinet {
        for index in 0..*panel_count {
            if let Some(limit) = limits.max_panel_displacement_m {
                margins.push(margin(
                    limit,
                    evaluation.panel_peak_displacement_m.get(index).copied(),
                ));
            }
            if let Some(limit) = limits.max_bending_stress_pa {
                margins.push(margin(
                    limit,
                    evaluation.panel_peak_bending_stress_pa.get(index).copied(),
                ));
            }
        }
    }
    margins
}

/// Human-readable label per active margin, in the same order as
/// [`constraint_margins`]: excursion first, then stress allowables sorted by
/// region marker, then per-panel displacement and bending stress.
pub fn constraint_labels(
    moving: &MovingAssemblyLimits,
    cabinet: Option<&CabinetConstraintSpec>,
) -> Vec<String> {
    let mut labels = Vec::new();
    if moving.xmax_m.is_some() {
        labels.push("moving_assembly.excursion".to_string());
    }
    for marker in moving.stress_allowables_pa.keys() {
        let region =
            moving_assembly_region_name(*marker).unwrap_or("unknown moving-assembly region");
        labels.push(format!("moving_assembly.stress[{region}]"));
    }
    if let Some((limits, panel_count)) = cabinet {
        for index in 0..*panel_count {
            if limits.max_panel_displacement_m.is_some() {
                labels.push(format!("cabinet.panel[{index}].displacement"));
            }
            if limits.max_bending_stress_pa.is_some() {
                labels.push(format!("cabinet.panel[{index}].bending_stress"));
            }
        }
    }
    labels
}

/// Solver-ready constraints plus the labels of each margin component.
pub struct StructuralConstraintSet {
    /// One inequality constraint (`c(x) ≥ 0` feasible) whose `dim` equals the
    /// number of active limits. Empty when no limit is active.
    pub constraints: Vec<Constraint<f64>>,
    /// Label per component of the constraint output, for reporting.
    pub labels: Vec<String>,
}

/// Bind structural limits into a `sonium_solvers` inequality constraint.
///
/// The caller supplies `evaluate`, a closure mapping design variables `x` to
/// a [`StructuralEvaluation`] — typically re-running (or looking up a cached
/// run of) the driver/cabinet simulation for that candidate, the same way the
/// speaker optimizer's PIR-slope constraint reads its evaluation cache. The
/// closure is invoked once per constraint evaluation and must populate every
/// metric an active limit constrains; missing metrics are reported as
/// infeasible via [`MISSING_METRIC_MARGIN`].
///
/// The produced constraint has `jac: None`, so the solver finite-differences
/// it, and returns all margins as one vector (dim = number of active limits),
/// ordered as [`constraint_labels`]:
///
/// ```ignore
/// let limits = MovingAssemblyLimits {
///     xmax_m: study.moving_system.xmax_m,
///     stress_allowables_pa: [(SURROUND, 2.0e6)].into_iter().collect(),
/// };
/// let set = to_solver_constraints(&limits, None, move |x: &[f64]| {
///     let (solutions, mesh) = rerun_driver_for(x); // caller-owned, cache it
///     StructuralEvaluation::from_axisymmetric_sweep(&solutions, &mesh)
///         .expect("driver evaluation must succeed")
/// });
/// // augmented_lagrangian(&objective, &grad, &x0, &set.constraints, &config)
/// ```
pub fn to_solver_constraints<F>(
    moving: &MovingAssemblyLimits,
    cabinet: Option<&CabinetConstraintSpec>,
    evaluate: F,
) -> StructuralConstraintSet
where
    F: Fn(&[f64]) -> StructuralEvaluation + Send + Sync + 'static,
{
    let labels = constraint_labels(moving, cabinet);
    if labels.is_empty() {
        return StructuralConstraintSet {
            constraints: Vec::new(),
            labels,
        };
    }
    let moving = moving.clone();
    let cabinet = cabinet.copied();
    let dim = labels.len();
    let constraint = Constraint {
        kind: ConstraintKind::Inequality,
        dim,
        fun: Box::new(move |x: &[f64]| {
            let evaluation = evaluate(x);
            constraint_margins(&evaluation, &moving, cabinet.as_ref())
        }),
        jac: None,
    };
    StructuralConstraintSet {
        constraints: vec![constraint],
        labels,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::MovingSystemSpec;
    use crate::axisymmetric_linear::{AxisymmetricLinearSolution, PowerBalance};
    use sonium_fem::mesh::rectangular_mesh_triangles;
    use sonium_fem::physics::acoustics::electroacoustics::CoupledSolveDiagnostics;
    use sonium_fem::physics::acoustics::electroacoustics::cabinet_radiation::PanelMaterial;
    use std::f64::consts::PI;

    fn dummy_solution(
        frequency_hz: f64,
        uz_rms: Complex64,
        node_count: usize,
    ) -> AxisymmetricLinearSolution {
        AxisymmetricLinearSolution {
            frequency_hz,
            voltage_rms_v: Complex64::new(1.0, 0.0),
            current_rms_a: Complex64::new(1.0, 0.0),
            impedance_ohm: Complex64::new(1.0, 0.0),
            pressure_rms_pa: Vec::new(),
            displacement_rz_m: vec![[Complex64::new(0.0, 0.0), uz_rms]; node_count],
            velocity_rz_m_per_s: vec![
                [
                    Complex64::new(0.0, 0.0),
                    Complex64::new(0.0, 2.0 * PI * frequency_hz) * uz_rms
                ];
                node_count
            ],
            observation_spl_db: 0.0,
            power: PowerBalance {
                input_w: 0.0,
                copper_loss_w: 0.0,
                mechanical_loss_w: 0.0,
                acoustic_loss_w: 0.0,
                acoustic_operator_loss_w: 0.0,
                residual_w: 0.0,
            },
            solver: CoupledSolveDiagnostics {
                iterations: 0,
                relative_residual: 0.0,
                converged: true,
                initial_guess_reused: false,
                preconditioner_rebuilt: false,
                defect_corrections: 0,
                residual_before_correction: 0.0,
            },
        }
    }

    fn diaphragm_mesh() -> Mesh {
        let mut mesh = rectangular_mesh_triangles(0.005, 0.035, -0.001, 0.0, 6, 2);
        for element in &mut mesh.elements {
            element.marker = DIAPHRAGM;
        }
        mesh
    }

    #[test]
    fn excursion_matches_lumped_sdof_peak_velocity_over_omega() {
        // SDOF moving system with a known mechanical impedance: v = F / Z_m,
        // x_rms = v / (iω), and the one-way peak excursion is √2·|x_rms|.
        let spec = MovingSystemSpec {
            moving_mass_kg: 14.0e-3,
            mechanical_resistance_kg_per_s: 1.1,
            stiffness_n_per_m: 950.0,
            xmax_m: Some(4.0e-3),
        };
        let force_n = 2.0;
        let mesh = diaphragm_mesh();
        let frequencies = [30.0, 82.0, 300.0];
        let mut expected = Vec::new();
        let solutions: Vec<_> = frequencies
            .iter()
            .map(|&frequency| {
                let omega = 2.0 * PI * frequency;
                let z_m = spec
                    .try_mechanical_impedance(frequency)
                    .expect("valid lumped SDOF parameters");
                let velocity = Complex64::new(force_n, 0.0) / z_m;
                let uz_rms = velocity / Complex64::new(0.0, omega);
                expected.push(RMS_TO_PEAK * uz_rms.norm());
                dummy_solution(frequency, uz_rms, mesh.num_nodes())
            })
            .collect();

        let excursion = moving_assembly_excursion_m(&solutions, &mesh).unwrap();
        for (actual, expected) in excursion.iter().zip(&expected) {
            assert!(
                (actual - expected).abs() / expected < 1e-12,
                "excursion {actual:.6e} vs SDOF {expected:.6e}"
            );
        }
        // The sweep maximum is the 30 Hz point, the closest to this SDOF's
        // fs = sqrt(950/0.014)/(2π) ≈ 41.4 Hz.
        let peak = excursion.iter().copied().fold(0.0, f64::max);
        let max_expected = expected.iter().copied().fold(0.0, f64::max);
        assert!((peak - max_expected).abs() / max_expected < 1e-12);

        // Margin sign flips across the limit.
        let evaluation = StructuralEvaluation::from_axisymmetric_sweep(&solutions, &mesh).unwrap();
        assert!((evaluation.peak_excursion_m.unwrap() - max_expected).abs() < 1e-15);
        let feasible = MovingAssemblyLimits {
            xmax_m: Some(1.1 * max_expected),
            ..MovingAssemblyLimits::default()
        };
        let infeasible = MovingAssemblyLimits {
            xmax_m: Some(0.9 * max_expected),
            ..MovingAssemblyLimits::default()
        };
        assert!(constraint_margins(&evaluation, &feasible, None)[0] > 0.0);
        assert!(constraint_margins(&evaluation, &infeasible, None)[0] < 0.0);
    }

    #[test]
    fn peak_stress_matches_analytic_uniform_strain_and_margin_signs_flip() {
        // Uniform u_z = α·z on a diaphragm-only mesh: εzz = α alone, so
        // σ = [λα, λα, (λ+2μ)α] and |σ_vm,rms| = 2·μ·α exactly.
        let mesh = diaphragm_mesh();
        let alpha = 0.25;
        let material = crate::axisymmetric_linear::moving_assembly_material(DIAPHRAGM).unwrap();
        let expected_rms = 2.0 * material.mu() * alpha;
        let expected_peak = RMS_TO_PEAK * expected_rms;

        let mut solution = dummy_solution(100.0, Complex64::new(0.0, 0.0), mesh.num_nodes());
        for (node, displacement) in solution.displacement_rz_m.iter_mut().enumerate() {
            displacement[1] = Complex64::new(alpha * mesh.nodes[node].y, 0.0);
        }
        let solutions = vec![solution];

        let peak = moving_assembly_peak_stress_pa(&solutions, &mesh).unwrap();
        let actual = peak[&DIAPHRAGM];
        assert!(
            (actual - expected_peak).abs() / expected_peak < 1e-12,
            "peak von Mises {actual:.6e} vs analytic {expected_peak:.6e}"
        );

        let evaluation = StructuralEvaluation::from_axisymmetric_sweep(&solutions, &mesh).unwrap();
        let limits = |scale: f64| MovingAssemblyLimits {
            xmax_m: None,
            stress_allowables_pa: [(DIAPHRAGM, scale * expected_peak)].into_iter().collect(),
        };
        assert!(constraint_margins(&evaluation, &limits(1.1), None)[0] > 0.0);
        assert!(constraint_margins(&evaluation, &limits(0.9), None)[0] < 0.0);
        assert_eq!(
            constraint_labels(&limits(1.0), None),
            vec!["moving_assembly.stress[moving diaphragm]".to_string()]
        );
    }

    #[test]
    fn cabinet_panel_peak_displacement_matches_navier_static_deflection() {
        // Single 0.2×0.3 m MDF panel under quasi-static uniform pressure.
        // Reference: Navier series for the center deflection of a
        // simply-supported plate, w_max = Σ 16·p·s_mn/(π⁶·D·m·n·k⁴).
        let panel = CabinetPanel {
            width: 0.20,
            height: 0.30,
            material: PanelMaterial::mdf_18mm(),
        };
        let cabinet = Cabinet {
            panels: vec![panel.clone()],
        };
        let p = 100.0;
        let cavity = vec![(1.0, Complex64::new(p, 0.0))];
        let evaluations = cabinet_panel_response(&cavity, &cabinet, 12).unwrap();
        assert_eq!(evaluations.len(), 1);
        let actual = evaluations[0].peak_displacement_m;

        let rigidity = panel.material.flexural_rigidity();
        let (a, b) = (panel.width, panel.height);
        let mut navier = 0.0;
        for m in (1..=25).step_by(2) {
            for n in (1..=25).step_by(2) {
                let sign = ((m as f64) * PI / 2.0).sin() * ((n as f64) * PI / 2.0).sin();
                let k2 = (m as f64 / a).powi(2) + (n as f64 / b).powi(2);
                navier += 16.0 * p * sign / (PI.powi(6) * rigidity * m as f64 * n as f64 * k2 * k2);
            }
        }
        let expected_peak = RMS_TO_PEAK * navier;
        assert!(
            (actual - expected_peak).abs() / expected_peak < 0.02,
            "peak center displacement {actual:.6e} vs √2·Navier {expected_peak:.6e}"
        );

        // Hand-computed quasi-static X_11 from the modal mass and force
        // coefficient dominates the response (within a few % for this aspect).
        let f_11 = panel.mode_frequency(1, 1);
        let mode = panel
            .modes_below(2.0 * f_11, 4)
            .into_iter()
            .find(|mode| mode.m == 1 && mode.n == 1)
            .unwrap();
        let omega_11 = 2.0 * PI * mode.frequency;
        let x_11 = mode.force_coefficient * p / (mode.modal_mass * omega_11 * omega_11);
        assert!(
            (actual - RMS_TO_PEAK * x_11).abs() / actual < 0.05,
            "peak {actual:.6e} vs √2·X_11 {:.6e}",
            RMS_TO_PEAK * x_11
        );

        // Bending stress is positive and scales with pressure.
        assert!(evaluations[0].peak_bending_stress_pa > 0.0);
        // Independent Navier/modal stress reconstruction at the same
        // pressure anchors the panel bending metric, not just displacement.
        let modes = panel.modal_displacements(1.0, Complex64::new(p, 0.0), 10.0 * f_11, 12);
        let section_factor = 6.0 * rigidity / panel.material.thickness.powi(2);
        let mut sigma_x = 0.0;
        let mut sigma_y = 0.0;
        for mode in modes {
            let sign = ((mode.m as f64) * PI / 2.0).sin() * ((mode.n as f64) * PI / 2.0).sin();
            let displacement = mode.displacement_m.re * sign;
            let k_m2 = (mode.m as f64 * PI / a).powi(2);
            let k_n2 = (mode.n as f64 * PI / b).powi(2);
            sigma_x +=
                section_factor * (k_m2 + panel.material.poissons_ratio * k_n2) * displacement;
            sigma_y +=
                section_factor * (panel.material.poissons_ratio * k_m2 + k_n2) * displacement;
        }
        let expected_stress = RMS_TO_PEAK * sigma_x.abs().max(sigma_y.abs());
        assert!(
            (evaluations[0].peak_bending_stress_pa - expected_stress).abs()
                / expected_stress.max(1.0)
                < 5.0e-8,
            "bending stress {} vs independent modal sum {expected_stress}",
            evaluations[0].peak_bending_stress_pa
        );
        let doubled =
            cabinet_panel_response(&[(1.0, Complex64::new(2.0 * p, 0.0))], &cabinet, 12).unwrap();
        assert!(
            (doubled[0].peak_displacement_m - 2.0 * actual).abs() / (2.0 * actual) < 1e-12,
            "linear scaling with pressure"
        );

        // Constraint sign flips at the displacement limit.
        let evaluation = StructuralEvaluation::default()
            .with_cabinet_panels(&cavity, &cabinet, 12)
            .unwrap();
        let spec: CabinetConstraintSpec = (
            CabinetPanelLimits {
                max_panel_displacement_m: Some(1.05 * actual),
                max_bending_stress_pa: None,
            },
            1,
        );
        assert!(
            constraint_margins(&evaluation, &MovingAssemblyLimits::default(), Some(&spec))[0] > 0.0
        );
        let spec: CabinetConstraintSpec = (
            CabinetPanelLimits {
                max_panel_displacement_m: Some(0.95 * actual),
                max_bending_stress_pa: None,
            },
            1,
        );
        assert!(
            constraint_margins(&evaluation, &MovingAssemblyLimits::default(), Some(&spec))[0] < 0.0
        );
    }

    #[test]
    fn solver_constraint_is_feasible_below_the_limit_and_violated_above() {
        // 1-DOF check: the design variable is the peak excursion itself.
        let limits = MovingAssemblyLimits {
            xmax_m: Some(4.0e-3),
            stress_allowables_pa: [(SURROUND, 2.0e6)].into_iter().collect(),
        };
        let set = to_solver_constraints(&limits, None, |x: &[f64]| StructuralEvaluation {
            peak_excursion_m: Some(x[0]),
            peak_stress_pa: [(SURROUND, 1.0e6 * x[0] / 4.0e-3)].into_iter().collect(),
            ..StructuralEvaluation::default()
        });
        assert_eq!(set.constraints.len(), 1);
        assert_eq!(set.constraints[0].kind, ConstraintKind::Inequality);
        assert!(set.constraints[0].jac.is_none());
        assert_eq!(set.constraints[0].dim, 2);
        assert_eq!(
            set.labels,
            vec![
                "moving_assembly.excursion".to_string(),
                "moving_assembly.stress[surround]".to_string(),
            ]
        );

        let feasible = (set.constraints[0].fun)(&[2.0e-3]);
        assert!(feasible.iter().all(|c| *c >= 0.0), "feasible: {feasible:?}");
        let violated = (set.constraints[0].fun)(&[1.0e-2]);
        assert!(violated.iter().all(|c| *c < 0.0), "violated: {violated:?}");
        // Mixed: excursion feasible, stress violated.
        let mixed = (set.constraints[0].fun)(&[5.0e-3]);
        assert!(mixed[0] < 0.0 && mixed[1] > 0.0, "mixed: {mixed:?}");

        // Missing metrics fail safe (infeasible).
        let set = to_solver_constraints(&limits, None, |_| StructuralEvaluation::default());
        let margins = (set.constraints[0].fun)(&[0.0]);
        assert!(
            margins
                .iter()
                .all(|c| *c == MISSING_METRIC_MARGIN && *c < 0.0)
        );

        // NaN and infinity must fail safe just like an omitted metric; they
        // must not escape as NaN and bypass an inequality constraint.
        let nan = StructuralEvaluation {
            peak_excursion_m: Some(f64::NAN),
            peak_stress_pa: [(SURROUND, f64::INFINITY)].into_iter().collect(),
            ..StructuralEvaluation::default()
        };
        let nan_margins = constraint_margins(&nan, &limits, None);
        assert_eq!(
            nan_margins,
            vec![MISSING_METRIC_MARGIN, MISSING_METRIC_MARGIN]
        );

        // No active limits → no constraints.
        let set = to_solver_constraints(&MovingAssemblyLimits::default(), None, |_| {
            StructuralEvaluation::default()
        });
        assert!(set.constraints.is_empty() && set.labels.is_empty());
    }

    #[test]
    fn limits_validate_positive_values() {
        let moving = MovingAssemblyLimits {
            xmax_m: Some(-1.0),
            ..MovingAssemblyLimits::default()
        };
        assert!(moving.validate().is_err());
        let moving = MovingAssemblyLimits {
            xmax_m: None,
            stress_allowables_pa: [(SPIDER, 0.0)].into_iter().collect(),
        };
        assert!(moving.validate().is_err());
        let cabinet = CabinetPanelLimits {
            max_panel_displacement_m: Some(f64::NAN),
            max_bending_stress_pa: None,
        };
        assert!(cabinet.validate().is_err());
        let spec = MovingSystemSpec {
            moving_mass_kg: 14.0e-3,
            mechanical_resistance_kg_per_s: 1.1,
            stiffness_n_per_m: 950.0,
            xmax_m: Some(-1.0),
        };
        assert!(spec.validate().is_err());
    }
}
