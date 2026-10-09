//! Density-based topology optimization for the axisymmetric motor steel path.
//!
//! Pole and yoke cells are design variables. SIMP/RAMP interpolates relative
//! permeability between air and low-carbon steel, a magnetostatic adjoint
//! supplies every BL sensitivity from one extra solve, and MMA enforces the
//! requested steel volume fraction.
//!
//! Derivatives are fully analytic:
//! - `d(-BL)/du` is a fixed sparse vector on voice-coil DOFs (BL is linear in
//!   the vector potential up to the `|·|` sign), derived once per mesh.
//! - `dJ/dρ_e = (∂(1/μ)/∂ρ_e)·λᵀK_e^(unit)u` uses per-element unit stiffness
//!   matrices assembled once at startup.
//! - One real sparse Cholesky factorization per iteration serves both the
//!   forward and the adjoint solve (the A-theta operator is symmetric and the
//!   system is SPD under the homogeneous Dirichlet outer boundaries; a dense
//!   real LU is kept as a fallback if the Cholesky pivot check disagrees).
//!
//! Manufacturing and robustness controls (all config-gated, defaults keep
//! the plain volume-constrained run):
//! - **Minimum length scale**: the density filter radius `R`
//!   ([`MotorTopologyConfig::density_filter_radius_m`]) is the minimum
//!   length-scale mechanism. The filtered density at a cell is a weighted
//!   average over its radius-`R` neighborhood, so a binarized (projected)
//!   design cannot sustain steel webs or air pockets thinner than ≈ `2R`
//!   (classical density-filter result; see e.g. Sigmund & Maute 2013 §3.1).
//!   The test suite pins this with an isolated-speck proxy: no steel cell of
//!   a filtered run may lack a steel neighbor within `2R`.
//! - **Symmetry** ([`MotorTopologyConfig::symmetry`]): mirrored design cells
//!   share one design variable (sensitivities summed, densities identical).
//! - **Connectivity** ([`MotorTopologyConfig::bridges`]): pragmatic proxy
//!   for "steel must connect pole to back iron" — a linear lower bound on
//!   the mean density over a caller-specified (r, z) box, enforced through
//!   the multi-constraint MMA. True percolation constraints are nonlinear
//!   and deferred.
//! - **Robust design** ([`MotorTopologyConfig::robust`]): erode/dilate
//!   worst-case formulation over three projection thresholds
//!   (`η−δ, η, η+δ`); MMA sees the worst-case objective and the
//!   sensitivities of the argmax branch (3 forward+adjoint solves per
//!   iteration).
//! - **Steel saturation** ([`MotorTopologyConfig::steel_permeability`]):
//!   with [`SteelPermeability::DifferentialFromBias`], the interpolation
//!   scales each design cell toward its LOCAL differential permeability
//!   `mu_diff_e = 1/(mu_0 * dH/dB)` extracted from a nonlinear B-H bias
//!   solve of the current design, instead of the constant `mu_r = 1000`.
//!   Saturated bottleneck cells then stop attracting steel the real
//!   material cannot carry. The map can be refreshed every `k` iterations;
//!   it is held fixed between refreshes so within-window adjoint
//!   derivatives stay exact.

use crate::axisymmetric_linear::{
    AIR_GAP, AxisymmetricLinearConfig, AxisymmetricLinearModel, MAGNETIC_AIR, MAGNETIC_AXIS,
    MAGNETIC_LOWER_END, MAGNETIC_OUTER_RADIUS, MAGNETIC_UPPER_END, PERMANENT_MAGNET, STEEL_POLE,
    STEEL_YOKE, VOICE_COIL, element_centroid, magnetic_bias_from_potential, triangle_area,
    triangle_gradients,
};
use crate::nonlinear_bias::steel_permeability_map_from_bias;
use serde::Serialize;
use sonium_fem::assembly::{RegionMap, TripletMatrix, assemble_bilinear_by_region};
use sonium_fem::assembly::{assemble_bilinear_where, assemble_linear_where};
use sonium_fem::basis::PolynomialDegree;
use sonium_fem::boundary::generic::{FemSystem, apply_homogeneous_dirichlet};
use sonium_fem::mesh::Mesh;
use sonium_fem::optimization::{
    DensityFilter, HeavisideProjection, InterpolationScheme, MmaConstraint, MmaOptimizer,
};
use sonium_fem::physics::electromagnetics::magnetostatics::{
    AxisymmetricAThetaForm, AzimuthalCurrentDensityForm, BhCurve, MU_0, NonlinearBiasSolveConfig,
};
use sonium_fem::spaces::H1Space;
use sonium_solvers::{LuFactorization, SparseCholeskyFactorization, lu_factorize};
use std::f64::consts::PI;

/// Linear small-signal relative permeability assigned to fully dense steel
/// when [`SteelPermeability::Constant`] is active.
pub(crate) const LINEAR_STEEL_MU_R: f64 = 1_000.0;

/// Heaviside projection + beta-continuation settings.
///
/// When present, the physical density entering the permeability interpolation
/// is `rho_phys = heaviside(rho_filtered; beta, eta)`. `beta` starts at
/// `beta_initial` and doubles every `continuation_interval` MMA iterations
/// until it reaches `beta_max`.
#[derive(Debug, Clone, Copy)]
pub struct ProjectionContinuationConfig {
    /// Initial projection sharpness.
    pub beta_initial: f64,
    /// Maximum projection sharpness reached by continuation.
    pub beta_max: f64,
    /// Projection threshold (typically 0.5).
    pub eta: f64,
    /// Iterations between beta doublings.
    pub continuation_interval: usize,
}

/// Mirror-symmetry reduction: design cells mirrored across the midplane of
/// the design region's bounding box share a single design variable
/// (sensitivities are summed over the group, identical densities assigned).
/// Pairing is by centroid proximity to the mirrored centroid: exact pairs on
/// mirror-symmetric triangulations, approximate equivalence classes on
/// meshes whose triangulation is not mirror-symmetric (e.g. uniform diagonal
/// splits). Unmatched cells keep their own variable.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SymmetryAxis {
    /// Mirror in `z` across the axial midplane `z = z_mid` of the design
    /// bounding box (top/bottom symmetry of the motor cross-section).
    R,
    /// Mirror in `r` across the radial midplane `r = r_mid` of the design
    /// bounding box.
    Z,
}

/// Connectivity proxy: keeps the mean design density over a caller-specified
/// (r, z) box at or above `min_average_density`.
///
/// Members are the design cells whose centroid lies inside the box. This is
/// the pragmatic stand-in for "steel must connect pole to back iron": a path
/// of sufficiently dense cells across the box is not guaranteed (true
/// percolation constraints are nonlinear), but the linear mean bound is
/// exact for the multi-constraint MMA and reliably preserves a steel bridge.
///
/// When the current (uniform) design would violate the floor, the optimizer
/// starts from a pre-densified feasible layout: bridge members are raised to
/// `min_average_density` and the remaining cells rebalanced to meet the
/// steel volume fraction exactly. Since the constraints are linear, every
/// linearized MMA subproblem then stays feasible (feasible-start MMA). If no
/// such initial design exists (floor too high for the volume budget), the
/// run fails with `InvalidConfiguration`.
#[derive(Debug, Clone, Copy)]
pub struct BridgeConstraint {
    /// Minimum centroid radius (m) of member cells.
    pub r_min: f64,
    /// Maximum centroid radius (m) of member cells.
    pub r_max: f64,
    /// Minimum centroid axial coordinate (m) of member cells.
    pub z_min: f64,
    /// Maximum centroid axial coordinate (m) of member cells.
    pub z_max: f64,
    /// Required minimum mean design density across the member cells.
    pub min_average_density: f64,
}

/// Robust (erode/dilate) formulation settings.
///
/// Requires [`MotorTopologyConfig::projection`]. Each iteration evaluates the
/// objective and its sensitivities at three Heaviside projections of the same
/// filtered design — dilated (`eta - delta_eta`), nominal (`eta`), and eroded
/// (`eta + delta_eta`) — and hands MMA the worst-case objective (the maximum;
/// the objective is minimized) with the chain-ruled sensitivities of the
/// argmax branch. Cost: 3 forward+adjoint solves per iteration.
#[derive(Debug, Clone, Copy)]
pub struct RobustConfig {
    /// Projection-threshold offset between the dilated/nominal/eroded designs.
    pub delta_eta: f64,
}

/// Steel permeability model feeding the density interpolation.
///
/// The interpolation maps the physical density of a design cell to a
/// relative permeability `mu_r(rho_e) = 1 + interp(rho_e) * (mu_steel_e - 1)`
/// where `mu_steel_e` is selected by this option. The analytic sensitivity
/// formula is unchanged in form; `c'(rho_e)` simply uses `mu_steel_e` in
/// place of the constant 1000.
#[derive(Debug, Clone, Default)]
pub enum SteelPermeability {
    /// Fixed `mu_r = 1000` low-carbon steel everywhere (the linear default;
    /// identical to the pre-D3 behavior).
    #[default]
    Constant,
    /// Local differential permeability `mu_diff_e = 1/(mu_0 * dH/dB)`
    /// extracted per design element from a nonlinear B-H bias solve of the
    /// current design (see
    /// [`crate::nonlinear_bias::steel_permeability_map_from_bias`]).
    /// Saturated bottleneck elements get a low `mu_diff_e`, so the
    /// interpolation stops rewarding narrow flux paths that real steel
    /// cannot carry.
    ///
    /// The map is held fixed between refreshes, so within a window the
    /// adjoint density derivatives stay exact; across a refresh the model
    /// changes and sensitivities are exact for the new map.
    DifferentialFromBias {
        /// Monotone steel B-H curve of the extraction solve.
        bh_curve: BhCurve,
        /// Newton/load-stepping settings of the extraction solve.
        solve: NonlinearBiasSolveConfig,
        /// Rebuild the map from the current design every `k` MMA iterations
        /// (iteration 0 always extracts). `None` extracts once at startup
        /// from the initial design.
        refresh_interval: Option<usize>,
    },
}

/// Motor topology-optimization settings.
#[derive(Debug, Clone)]
pub struct MotorTopologyConfig {
    /// Maximum MMA iterations.
    pub max_iterations: usize,
    /// Maximum density change used as convergence criterion.
    pub tolerance: f64,
    /// Allowed steel volume fraction inside pole/yoke design cells.
    pub steel_volume_fraction: f64,
    /// Density interpolation applied to relative permeability.
    pub interpolation: InterpolationScheme,
    /// Steel permeability model of the interpolation (linear constant vs
    /// differential permeability from a nonlinear bias solve).
    pub steel_permeability: SteelPermeability,
    /// Optional cone-filter radius (m) applied to the design densities each
    /// iteration. `None` keeps the current unfiltered behavior.
    pub density_filter_radius_m: Option<f64>,
    /// Optional Heaviside projection with beta continuation applied after the
    /// (optional) density filter. `None` keeps the current behavior where the
    /// physical density equals the (filtered) design density.
    pub projection: Option<ProjectionContinuationConfig>,
    /// Optional mirror-symmetry reduction pairing design cells across the
    /// design-region midplane. `None` keeps one variable per design cell.
    pub symmetry: Option<SymmetryAxis>,
    /// Linear connectivity proxies (mean-density floors over (r, z) boxes),
    /// enforced alongside the volume constraint via multi-constraint MMA.
    /// Empty keeps the plain volume-only run.
    pub bridges: Vec<BridgeConstraint>,
    /// Optional erode/dilate robust formulation (requires `projection`).
    /// `None` evaluates only the nominal projection.
    pub robust: Option<RobustConfig>,
}

impl Default for MotorTopologyConfig {
    fn default() -> Self {
        Self {
            max_iterations: 40,
            tolerance: 0.01,
            steel_volume_fraction: 0.65,
            interpolation: InterpolationScheme::Ramp { penalty: 3.0 },
            steel_permeability: SteelPermeability::Constant,
            density_filter_radius_m: None,
            projection: None,
            symmetry: None,
            bridges: Vec::new(),
            robust: None,
        }
    }
}

/// Summary statistics for one density representation in a manufacturability
/// report. `volume_fraction` is volume-weighted; the other statistics are
/// element-wise so they remain useful when the mesh is non-uniform.
#[derive(Debug, Clone, Serialize)]
pub struct MotorTopologyDensityEvidence {
    pub count: usize,
    pub minimum: f64,
    pub maximum: f64,
    pub mean: f64,
    pub volume_fraction: f64,
    pub gray_fraction: f64,
}

/// Projection-continuation evidence for a motor topology result.
#[derive(Debug, Clone, Serialize)]
pub struct MotorTopologyProjectionEvidence {
    pub beta_initial: f64,
    pub beta_final: f64,
    pub beta_max: f64,
    pub eta: f64,
    pub continuation_complete: bool,
}

/// Evidence for one configured bridge-density proxy.
#[derive(Debug, Clone, Serialize)]
pub struct MotorTopologyBridgeEvidence {
    pub r_min: f64,
    pub r_max: f64,
    pub z_min: f64,
    pub z_max: f64,
    pub minimum_average_density: f64,
    pub final_average_density: f64,
    pub margin: f64,
    pub satisfied: bool,
}

/// Robust erode/dilate density evidence at the final continuation beta.
#[derive(Debug, Clone, Serialize)]
pub struct MotorTopologyRobustEvidence {
    pub delta_eta: f64,
    pub dilated: MotorTopologyDensityEvidence,
    pub nominal: MotorTopologyDensityEvidence,
    pub eroded: MotorTopologyDensityEvidence,
}

/// Range summary for a scalar history such as differential permeability.
#[derive(Debug, Clone, Serialize)]
pub struct MotorTopologyScalarEvidence {
    pub count: usize,
    pub minimum: f64,
    pub maximum: f64,
    pub mean: f64,
}

/// Evidence describing the steel permeability model used by the optimizer.
#[derive(Debug, Clone, Serialize)]
pub struct MotorTopologyPermeabilityEvidence {
    /// `constant` or `differential_from_bias`.
    pub model: &'static str,
    pub extraction_count: usize,
    pub final_differential_relative_permeability: Option<MotorTopologyScalarEvidence>,
}

/// Consumer-facing evidence for the axisymmetric motor topology controls.
///
/// This is a manufacturability evidence report, not a 3D manufacturing
/// certification. In particular, bridge constraints are mean-density
/// connectivity proxies and do not prove nonlinear percolation, stress,
/// thermal, or tooling constraints.
#[derive(Debug, Clone, Serialize)]
pub struct MotorTopologyManufacturabilityReport {
    pub scope: &'static str,
    pub design_element_count: usize,
    pub design_variable_count: usize,
    pub design: MotorTopologyDensityEvidence,
    pub filtered: MotorTopologyDensityEvidence,
    pub physical: MotorTopologyDensityEvidence,
    pub density_filter_radius_m: Option<f64>,
    pub projection: Option<MotorTopologyProjectionEvidence>,
    pub symmetry: Option<String>,
    pub bridges: Vec<MotorTopologyBridgeEvidence>,
    pub robust: Option<MotorTopologyRobustEvidence>,
    pub permeability: MotorTopologyPermeabilityEvidence,
    pub volume_constraint_satisfied: bool,
    pub bridge_constraints_satisfied: bool,
}

/// Result of a motor topology run.
#[derive(Debug, Clone)]
pub struct MotorTopologyResult {
    /// Mesh element indices corresponding to `densities`.
    pub design_elements: Vec<usize>,
    /// Optimized iron densities (design variables, pre-filter/projection).
    pub densities: Vec<f64>,
    /// BL history in T·m.
    pub bl_history_tm: Vec<f64>,
    /// Number of completed iterations.
    pub iterations: usize,
    /// Whether the density-change criterion was met.
    pub converged: bool,
    /// Share of final physical densities (post filter/projection) lying in
    /// the gray band (0.1, 0.9). Lower is better: continuation should drive
    /// this toward zero.
    pub gray_fraction: f64,
    /// Differential permeability maps extracted during the run (one entry per
    /// extraction, in order), for
    /// [`SteelPermeability::DifferentialFromBias`]. Empty for
    /// [`SteelPermeability::Constant`].
    pub permeability_map_history: Vec<Vec<f64>>,
    /// Explicit evidence for filter, projection, symmetry, bridge, robust,
    /// volume, and permeability controls used by this result.
    pub manufacturability: MotorTopologyManufacturabilityReport,
}

/// Recoverable motor topology error.
#[derive(Debug, thiserror::Error)]
pub enum MotorTopologyError {
    #[error("invalid motor topology configuration: {0}")]
    InvalidConfiguration(&'static str),
    #[error("motor topology FEM failure: {0}")]
    Fem(String),
}

/// Per-design-element unit stiffness contribution (`inv_mu = 1`), with
/// Dirichlet rows/columns removed to mirror the post-BC system matrix.
struct ElementContribution {
    rows: Vec<usize>,
    cols: Vec<usize>,
    values: Vec<f64>,
}

/// One real factorization of the (symmetric) system matrix, reused for the
/// forward and adjoint solves of one MMA iteration.
pub(crate) enum MotorFactorization {
    SparseCholesky(Box<SparseCholeskyFactorization<f64>>),
    DenseLu(Box<LuFactorization<f64>>),
}

impl MotorFactorization {
    pub(crate) fn factorize(matrix: &TripletMatrix<f64>) -> Result<Self, MotorTopologyError> {
        let csr = matrix.to_csr();
        match SparseCholeskyFactorization::factorize(&csr) {
            Ok(factorization) => Ok(Self::SparseCholesky(Box::new(factorization))),
            Err(_) => {
                // Fallback: the pivot check found a non-SPD pivot — use a
                // pivoted dense real LU instead (still no complex conversion).
                let dense = csr.to_dense();
                let factorization = lu_factorize(&dense)
                    .map_err(|error| MotorTopologyError::Fem(error.to_string()))?;
                Ok(Self::DenseLu(Box::new(factorization)))
            }
        }
    }

    pub(crate) fn solve(&self, rhs: &[f64]) -> Result<Vec<f64>, MotorTopologyError> {
        match self {
            Self::SparseCholesky(factorization) => factorization
                .solve(rhs)
                .map_err(|error| MotorTopologyError::Fem(error.to_string())),
            Self::DenseLu(factorization) => factorization
                .solve(rhs)
                .map_err(|error| MotorTopologyError::Fem(error.to_string())),
        }
    }
}

/// Cached, density-independent data for the motor topology problem: base
/// system (regions + magnet RHS + Dirichlet BCs), per-design-element unit
/// stiffness matrices, and the sparse voice-coil structure of `d(-BL)/du`.
pub(crate) struct MotorTopologyWorkspace {
    driver: AxisymmetricLinearConfig,
    pub(crate) mesh: Mesh,
    pub(crate) design_elements: Vec<usize>,
    volumes: Vec<f64>,
    /// Post-BC base triplets: region assembly with steel at mu_r = 1000.
    base_matrix: TripletMatrix<f64>,
    /// Post-BC magnet-equivalent-current RHS.
    base_rhs: Vec<f64>,
    /// Unit `inv_mu` stiffness triplets per design element (Dirichlet-free).
    contributions: Vec<ElementContribution>,
    /// Voice-coil DOFs with nonzero `g_m` (see [`Self::bl_state_derivative`]).
    coil_dofs: Vec<usize>,
    /// `g_m = sum_{coil cells c containing m} grad(N_m)_z|c * r_c * area_c`.
    coil_gradient_weights: Vec<f64>,
    /// `2 * pi * mean_coil_radius * active_turns / coil_weight`.
    bl_state_scale: f64,
    /// Voice-coil element indices (for the weighted-flux sign).
    coil_cells: Vec<usize>,
    /// Per-design-element steel relative permeability target of the density
    /// interpolation (`mu_r(rho_e) = 1 + interp(rho_e) * (steel_mu_r[e] - 1)`).
    /// `1000` everywhere under [`SteelPermeability::Constant`]; refreshed
    /// from the nonlinear bias map under
    /// [`SteelPermeability::DifferentialFromBias`].
    steel_mu_r: Vec<f64>,
}

impl MotorTopologyWorkspace {
    pub(crate) fn build(driver: &AxisymmetricLinearConfig) -> Result<Self, MotorTopologyError> {
        // Reuse the validated golden-model mesh and marker contract.
        let reference = AxisymmetricLinearModel::build(driver.clone())
            .map_err(|error| MotorTopologyError::Fem(error.to_string()))?;
        let mesh = reference.magnetic_mesh;
        let design_elements = mesh
            .elements
            .iter()
            .enumerate()
            .filter_map(|(index, element)| {
                matches!(element.marker, STEEL_POLE | STEEL_YOKE).then_some(index)
            })
            .collect::<Vec<_>>();
        if design_elements.is_empty() {
            return Err(MotorTopologyError::InvalidConfiguration(
                "magnetic mesh has no pole/yoke design cells",
            ));
        }
        let volumes = design_elements
            .iter()
            .map(|&element| axisymmetric_element_volume(&mesh, element))
            .collect::<Vec<_>>();

        let space = H1Space::new(PolynomialDegree::P1);

        // Density-independent base system: region assembly (steel regions
        // pinned at mu_r = 1000), magnet-equivalent-current RHS, homogeneous
        // Dirichlet outer boundaries — identical to the golden forward model.
        let mut forms = RegionMap::new();
        forms.insert(MAGNETIC_AIR, AxisymmetricAThetaForm::new(MU_0));
        forms.insert(PERMANENT_MAGNET, AxisymmetricAThetaForm::new(1.05 * MU_0));
        forms.insert(STEEL_POLE, AxisymmetricAThetaForm::new(1_000.0 * MU_0));
        forms.insert(STEEL_YOKE, AxisymmetricAThetaForm::new(1_000.0 * MU_0));
        forms.insert(VOICE_COIL, AxisymmetricAThetaForm::new(MU_0));
        forms.insert(AIR_GAP, AxisymmetricAThetaForm::new(MU_0));
        let matrix = assemble_bilinear_by_region(&forms, &space, &space, &mesh, 2)
            .map_err(|error| MotorTopologyError::Fem(error.to_string()))?;
        let equivalent_current_density = driver.remanence_t / (MU_0 * 20.0e-3);
        let source = AzimuthalCurrentDensityForm::new(move |_, _| equivalent_current_density);
        let rhs = assemble_linear_where(&source, &space, &mesh, 2, |_, element| {
            element.marker == PERMANENT_MAGNET
        });
        let mut system = FemSystem::new(matrix, rhs);
        for marker in [
            MAGNETIC_AXIS,
            MAGNETIC_OUTER_RADIUS,
            MAGNETIC_LOWER_END,
            MAGNETIC_UPPER_END,
        ] {
            apply_homogeneous_dirichlet(&mut system, &mesh, &space, marker);
        }
        let dirichlet_dofs = system.dirichlet_dofs.clone();

        // Unit-permeability stiffness of each design element, assembled once.
        // The A-theta form is linear in `inv_mu`, so the element matrix at
        // inverse permeability `nu` is `nu * K_e^(unit)`. Entries in
        // Dirichlet rows/columns are dropped because the BC application would
        // remove them from the system matrix anyway.
        let unit_form = AxisymmetricAThetaForm {
            inv_mu: 1.0,
            axis_epsilon: 1.0e-12,
        };
        let contributions = design_elements
            .iter()
            .map(|&element_index| {
                let triplet = assemble_bilinear_where(&unit_form, &space, &space, &mesh, 2, {
                    |index, _| index == element_index
                });
                let mut contribution = ElementContribution {
                    rows: Vec::with_capacity(triplet.nnz()),
                    cols: Vec::with_capacity(triplet.nnz()),
                    values: Vec::with_capacity(triplet.nnz()),
                };
                for k in 0..triplet.nnz() {
                    let (row, col) = (triplet.rows[k], triplet.cols[k]);
                    if !dirichlet_dofs.contains(&row) && !dirichlet_dofs.contains(&col) {
                        contribution.rows.push(row);
                        contribution.cols.push(col);
                        contribution.values.push(triplet.values[k]);
                    }
                }
                contribution
            })
            .collect::<Vec<_>>();

        // Voice-coil weights of the analytic state derivative. Mirrors the
        // weighted fold in `magnetic_bias_from_potential`: the radial flux
        // density per cell is `B_r = -dA/dz = -sum_l u_l * grad(N_l)_z`, the
        // coil flux functional is `F = sum_coil B_r * r_c * area_c`, and
        // `BL = 2*pi*mean_radius*N*|F|/W` with mesh-constant mean radius
        // `r_bar = (sum_coil <r^2> * area) / W` and weight
        // `W = sum_coil r_c * area_c`.
        let mut coil_gradient_weights = vec![0.0; mesh.num_nodes()];
        let mut weighted_radius_moment = 0.0;
        let mut weight = 0.0;
        let mut coil_cells = Vec::new();
        for (index, element) in mesh.elements.iter().enumerate() {
            if element.marker != VOICE_COIL {
                continue;
            }
            coil_cells.push(index);
            let nodes = element.vertices();
            let (radius, _) = element_centroid(&mesh, nodes);
            let area = triangle_area(&mesh, nodes);
            let gradients = triangle_gradients(&mesh, nodes);
            let radial = [
                mesh.nodes[nodes[0]].x,
                mesh.nodes[nodes[1]].x,
                mesh.nodes[nodes[2]].x,
            ];
            let mean_radius_squared = (radial.iter().map(|r| r * r).sum::<f64>()
                + radial[0] * radial[1]
                + radial[0] * radial[2]
                + radial[1] * radial[2])
                / 6.0;
            weighted_radius_moment += mean_radius_squared * area;
            weight += radius * area;
            for (local, &node) in nodes.iter().enumerate() {
                coil_gradient_weights[node] += gradients[local][1] * radius * area;
            }
        }
        if weight == 0.0 {
            return Err(MotorTopologyError::InvalidConfiguration(
                "magnetic mesh contains no voice-coil cells",
            ));
        }
        let mean_coil_radius = weighted_radius_moment / weight;
        let bl_state_scale = 2.0 * PI * mean_coil_radius * driver.active_turns / weight;
        let coil_dofs = coil_gradient_weights
            .iter()
            .enumerate()
            .filter_map(|(dof, &g)| (g != 0.0).then_some(dof))
            .collect::<Vec<_>>();

        Ok(Self {
            driver: driver.clone(),
            mesh,
            design_elements: design_elements.clone(),
            volumes,
            base_matrix: system.matrix,
            base_rhs: system.rhs,
            contributions,
            coil_dofs,
            coil_gradient_weights,
            bl_state_scale,
            coil_cells,
            steel_mu_r: vec![LINEAR_STEEL_MU_R; design_elements.len()],
        })
    }

    /// Install a per-design-element steel permeability map (relative
    /// differential permeability replacing the constant `mu_r = 1000` target
    /// of the density interpolation).
    fn set_steel_permeability_map(&mut self, map: &[f64]) {
        assert_eq!(
            map.len(),
            self.design_elements.len(),
            "permeability map must cover every design element"
        );
        self.steel_mu_r = map.to_vec();
    }

    /// Assemble `K(rho) = K_base + sum_e correction(rho_e) * K_e^(unit)`,
    /// replicating the region-plus-correction assembly of the golden forward
    /// model (same `mu_r(rho)` RAMP mapping, same clamp).
    fn assemble_matrix(
        &self,
        physical_densities: &[f64],
        interpolation: InterpolationScheme,
    ) -> TripletMatrix<f64> {
        let mut matrix = TripletMatrix::with_capacity(
            self.base_matrix.num_rows,
            self.base_matrix.num_cols,
            self.base_matrix.nnz()
                + self
                    .contributions
                    .iter()
                    .map(|contribution| contribution.rows.len())
                    .sum::<usize>(),
        );
        for k in 0..self.base_matrix.nnz() {
            matrix.push(
                self.base_matrix.rows[k],
                self.base_matrix.cols[k],
                self.base_matrix.values[k],
            );
        }
        let steel_inv_mu = 1.0 / (LINEAR_STEEL_MU_R * MU_0);
        for (position, (contribution, &density)) in self
            .contributions
            .iter()
            .zip(physical_densities)
            .enumerate()
        {
            let mu_r = 1.0
                + interpolation.interpolate(density.clamp(1.0e-3, 1.0))
                    * (self.steel_mu_r[position] - 1.0);
            let correction = 1.0 / (MU_0 * mu_r) - steel_inv_mu;
            for k in 0..contribution.rows.len() {
                matrix.push(
                    contribution.rows[k],
                    contribution.cols[k],
                    correction * contribution.values[k],
                );
            }
        }
        matrix
    }

    /// `d(1/mu)/d(rho)` of the RAMP permeability correction:
    /// `-(mu_steel - 1) * interp'(rho) / (mu_0 * mu_r(rho)^2)` evaluated at
    /// the clamped density (interior-side subgradient at the [1e-3, 1]
    /// bounds, so MMA gradients do not vanish on the boundary). `mu_steel` is
    /// the element's steel permeability target (1000 for the constant model,
    /// the local differential permeability under `DifferentialFromBias`).
    fn inv_mu_correction_derivative(
        density: f64,
        interpolation: InterpolationScheme,
        steel_mu_r: f64,
    ) -> f64 {
        let clamped = density.clamp(1.0e-3, 1.0);
        let mu_r = 1.0 + interpolation.interpolate(clamped) * (steel_mu_r - 1.0);
        -(steel_mu_r - 1.0) * interpolation.interpolate_derivative(clamped) / (MU_0 * mu_r * mu_r)
    }

    /// Coil flux functional `F = sum_coil B_r * r_c * area_c` at `potential`,
    /// evaluated in mesh element order exactly as `magnetic_bias_from_potential`
    /// does (only its sign is consumed).
    fn voice_coil_weighted_flux(&self, potential: &[f64]) -> f64 {
        let mut flux = 0.0;
        for &cell in &self.coil_cells {
            let nodes = self.mesh.elements[cell].vertices();
            let gradients = triangle_gradients(&self.mesh, nodes);
            let mut da_dz = 0.0;
            for (local, &node) in nodes.iter().enumerate() {
                da_dz += potential[node] * gradients[local][1];
            }
            let (radius, _) = element_centroid(&self.mesh, nodes);
            let area = triangle_area(&self.mesh, nodes);
            flux += (-da_dz) * radius * area;
        }
        flux
    }

    /// Analytic `d(-BL)/du` (objective is min `-BL`).
    ///
    /// `BL = 2*pi*r_bar*N*|F|/W` with `F = -sum_m g_m * u_m` linear in the
    /// potential, so `d(-BL)/du_m = scale * sign(F) * g_m` with
    /// `scale = 2*pi*r_bar*N/W` and `g_m` the cached coil gradient weights.
    fn bl_state_derivative(&self, potential: &[f64]) -> Vec<f64> {
        let flux_sign = self.voice_coil_weighted_flux(potential).signum();
        let mut derivative = vec![0.0; potential.len()];
        for &dof in &self.coil_dofs {
            derivative[dof] = self.bl_state_scale * flux_sign * self.coil_gradient_weights[dof];
        }
        derivative
    }

    /// Objective `-BL(physical_densities)` from one forward solve.
    #[cfg(test)]
    fn objective(
        &self,
        physical_densities: &[f64],
        interpolation: InterpolationScheme,
    ) -> Result<f64, MotorTopologyError> {
        let (_, bl_tm) = self.solve_forward(physical_densities, interpolation)?;
        Ok(-bl_tm)
    }

    /// Forward solve helper returning `(potential, BL)`.
    pub(crate) fn solve_forward(
        &self,
        physical_densities: &[f64],
        interpolation: InterpolationScheme,
    ) -> Result<(Vec<f64>, f64), MotorTopologyError> {
        let matrix = self.assemble_matrix(physical_densities, interpolation);
        let factorization = MotorFactorization::factorize(&matrix)?;
        let potential = factorization.solve(&self.base_rhs)?;
        let bias = magnetic_bias_from_potential(&self.driver, &self.mesh, potential.clone())
            .map_err(|error| MotorTopologyError::Fem(error.to_string()))?;
        Ok((potential, bias.distributed_bl_tm))
    }

    /// Small-signal BL and analytic density sensitivities (objective min `-BL`)
    /// from one factorization reused for the forward and adjoint solves.
    ///
    /// With `K(ρ)u = b` (b density-independent) and objective `f = -BL`:
    /// `df/dρ_e = λᵀ (∂K/∂ρ_e) u` where `K λ = dBL/du` (the operator is
    /// symmetric) and `∂K/∂ρ_e = (∂(1/μ)/∂ρ_e) * K_e^(unit)`.
    fn objective_and_sensitivities(
        &self,
        physical_densities: &[f64],
        interpolation: InterpolationScheme,
    ) -> Result<(f64, Vec<f64>), MotorTopologyError> {
        let matrix = self.assemble_matrix(physical_densities, interpolation);
        let factorization = MotorFactorization::factorize(&matrix)?;
        let potential = factorization.solve(&self.base_rhs)?;
        let bias = magnetic_bias_from_potential(&self.driver, &self.mesh, potential.clone())
            .map_err(|error| MotorTopologyError::Fem(error.to_string()))?;

        // Adjoint RHS `dBL/du = -d(-BL)/du` (same sign convention as the
        // previous finite-difference implementation).
        let df_du = self.bl_state_derivative(&potential);
        let adjoint_rhs = df_du.iter().map(|value| -*value).collect::<Vec<_>>();
        let adjoint = factorization.solve(&adjoint_rhs)?;

        let sensitivities = self
            .contributions
            .iter()
            .zip(physical_densities)
            .zip(&self.steel_mu_r)
            .map(|((contribution, &density), &steel_mu_r)| {
                let derivative =
                    Self::inv_mu_correction_derivative(density, interpolation, steel_mu_r);
                let mut quadratic = 0.0;
                for k in 0..contribution.rows.len() {
                    quadratic += adjoint[contribution.rows[k]]
                        * contribution.values[k]
                        * potential[contribution.cols[k]];
                }
                derivative * quadratic
            })
            .collect();
        Ok((bias.distributed_bl_tm, sensitivities))
    }
}

fn axisymmetric_element_volume(mesh: &Mesh, element_index: usize) -> f64 {
    let vertices = mesh.elements[element_index].vertices();
    let points = vertices
        .iter()
        .map(|&node| &mesh.nodes[node])
        .collect::<Vec<_>>();
    let area = 0.5
        * ((points[1].x - points[0].x) * (points[2].y - points[0].y)
            - (points[2].x - points[0].x) * (points[1].y - points[0].y))
            .abs();
    let radius = points.iter().map(|point| point.x).sum::<f64>() / points.len() as f64;
    2.0 * PI * radius * area
}

/// `(r, z)` centroids of the design cells, in design-vector order.
fn design_centroids(mesh: &Mesh, design_elements: &[usize]) -> Vec<[f64; 2]> {
    design_elements
        .iter()
        .map(|&element| {
            let (radius, z) = element_centroid(mesh, mesh.elements[element].vertices());
            [radius, z]
        })
        .collect()
}

/// Group design cells into mirror-symmetric equivalence classes across the
/// midplane of the design bounding box. Each cell is linked to the centroid
/// nearest its mirrored centroid (within a fraction of its nearest-neighbor
/// distance), and groups are the equivalence closure of those links: exact
/// pairs on mirror-symmetric triangulations, small clusters on meshes whose
/// triangulation itself is not mirror-symmetric. Cells without a mirrored
/// counterpart form singleton groups and keep their own design variable.
fn symmetry_groups(centroids: &[[f64; 2]], axis: SymmetryAxis) -> Vec<Vec<usize>> {
    let n = centroids.len();
    // Coordinate index flipped by the mirror: R symmetry flips z (index 1),
    // Z symmetry flips r (index 0).
    let flipped = match axis {
        SymmetryAxis::R => 1,
        SymmetryAxis::Z => 0,
    };
    let mid = {
        let (min, max) = centroids.iter().fold(
            (f64::INFINITY, f64::NEG_INFINITY),
            |(min, max), centroid| (min.min(centroid[flipped]), max.max(centroid[flipped])),
        );
        0.5 * (min + max)
    };
    let distance = |i: usize, j: usize| {
        let dr = centroids[i][0] - centroids[j][0];
        let dz = centroids[i][1] - centroids[j][1];
        (dr * dr + dz * dz).sqrt()
    };
    let nearest_neighbor = |i: usize| {
        (0..n)
            .filter(|&j| j != i)
            .map(|j| distance(i, j))
            .fold(f64::INFINITY, f64::min)
    };

    // Link each cell to the closest centroid of its mirrored location. The
    // tolerance sits between the exact/near-exact mirror-partner distance
    // and the ordinary nearest-neighbor distance (85% of the latter).
    let mut parent: Vec<usize> = (0..n).collect();
    fn find(parent: &mut [usize], mut i: usize) -> usize {
        while parent[i] != i {
            parent[i] = parent[parent[i]];
            i = parent[i];
        }
        i
    }
    for i in 0..n {
        let mut mirror = centroids[i];
        mirror[flipped] = 2.0 * mid - mirror[flipped];
        let tolerance = 0.85 * nearest_neighbor(i);
        let partner = (0..n)
            .filter(|&j| j != i)
            .map(|j| {
                let dr = centroids[j][0] - mirror[0];
                let dz = centroids[j][1] - mirror[1];
                (j, (dr * dr + dz * dz).sqrt())
            })
            .filter(|&(_, gap)| gap <= tolerance)
            .min_by(|(_, a), (_, b)| a.total_cmp(b))
            .map(|(j, _)| j);
        if let Some(j) = partner {
            let (root_i, root_j) = (find(&mut parent, i), find(&mut parent, j));
            if root_i != root_j {
                parent[root_j] = root_i;
            }
        }
    }

    let mut components: std::collections::BTreeMap<usize, Vec<usize>> =
        std::collections::BTreeMap::new();
    for i in 0..n {
        components.entry(find(&mut parent, i)).or_default().push(i);
    }
    components.into_values().collect()
}

/// Design positions whose centroid lies inside the bridge (r, z) box.
fn bridge_members(centroids: &[[f64; 2]], bridge: &BridgeConstraint) -> Vec<usize> {
    centroids
        .iter()
        .enumerate()
        .filter_map(|(position, &[r, z])| {
            (r >= bridge.r_min && r <= bridge.r_max && z >= bridge.z_min && z <= bridge.z_max)
                .then_some(position)
        })
        .collect()
}

/// Map design densities to physical densities: optional density filter, then
/// optional Heaviside projection. Returns `(physical, filtered)`.
pub(crate) fn physical_densities(
    design: &[f64],
    filter: Option<&DensityFilter>,
    projection: Option<&HeavisideProjection>,
) -> (Vec<f64>, Vec<f64>) {
    let filtered = match filter {
        Some(filter) => filter.filter_densities(design),
        None => design.to_vec(),
    };
    let physical = match projection {
        Some(projection) => filtered
            .iter()
            .map(|&rho| projection.project(rho))
            .collect(),
        None => filtered.clone(),
    };
    (physical, filtered)
}

/// Chain-rule physical sensitivities `dJ/dρ_phys` back to design sensitivities
/// `dJ/dρ`: through the Heaviside projection (`∘ H'(ρ_filtered)`), then
/// through the normalized density filter (transpose weight application).
fn design_sensitivities(
    physical_sensitivities: &[f64],
    filtered: &[f64],
    filter: Option<&DensityFilter>,
    projection: Option<&HeavisideProjection>,
) -> Vec<f64> {
    let mut sensitivities = physical_sensitivities.to_vec();
    if let Some(projection) = projection {
        for (sensitivity, &rho) in sensitivities.iter_mut().zip(filtered) {
            *sensitivity *= projection.project_derivative(rho);
        }
    }
    if let Some(filter) = filter {
        sensitivities = filter.filter_sensitivities_transpose(&sensitivities);
    }
    sensitivities
}

/// Share of densities in the gray band (0.1, 0.9).
fn gray_fraction(densities: &[f64]) -> f64 {
    if densities.is_empty() {
        return 0.0;
    }
    let gray = densities
        .iter()
        .filter(|&&rho| (0.1..0.9).contains(&rho))
        .count();
    gray as f64 / densities.len() as f64
}

/// Beta used at the last executed iteration of a run (`iterations` is the
/// number of completed loop passes). `None` when the run used no projection.
pub(crate) fn final_continuation_beta(
    config: &MotorTopologyConfig,
    iterations: usize,
) -> Option<f64> {
    let projection = config.projection?;
    let mut beta = projection.beta_initial;
    for iteration in 0..iterations {
        if iteration > 0 && iteration % projection.continuation_interval == 0 {
            beta = (beta * 2.0).min(projection.beta_max);
        }
    }
    Some(beta)
}

/// Whether the density-change criterion is allowed to terminate optimization.
/// A projected run must complete beta continuation first; otherwise an
/// apparently stationary, fully gray design at `beta_initial` can be reported
/// as converged before projection sharpening has taken effect.
fn projection_continuation_complete(
    projection: Option<ProjectionContinuationConfig>,
    beta: Option<f64>,
) -> bool {
    match (projection, beta) {
        (None, _) => true,
        (Some(projection), Some(beta)) => beta >= projection.beta_max,
        (Some(_), None) => false,
    }
}

fn density_evidence(
    densities: &[f64],
    volumes: &[f64],
    total_volume: f64,
) -> MotorTopologyDensityEvidence {
    let (minimum, maximum, sum) = densities.iter().fold(
        (f64::INFINITY, f64::NEG_INFINITY, 0.0),
        |(minimum, maximum, sum), &density| {
            (minimum.min(density), maximum.max(density), sum + density)
        },
    );
    let weighted_volume = densities
        .iter()
        .zip(volumes)
        .map(|(&density, &volume)| density * volume)
        .sum::<f64>();
    let count = densities.len();
    MotorTopologyDensityEvidence {
        count,
        minimum: if count == 0 { 0.0 } else { minimum },
        maximum: if count == 0 { 0.0 } else { maximum },
        mean: if count == 0 { 0.0 } else { sum / count as f64 },
        volume_fraction: if count == 0 {
            0.0
        } else {
            weighted_volume / total_volume.max(f64::MIN_POSITIVE)
        },
        gray_fraction: gray_fraction(densities),
    }
}

fn scalar_evidence(values: &[f64]) -> MotorTopologyScalarEvidence {
    let (minimum, maximum, sum) = values.iter().fold(
        (f64::INFINITY, f64::NEG_INFINITY, 0.0),
        |(minimum, maximum, sum), &value| (minimum.min(value), maximum.max(value), sum + value),
    );
    let count = values.len();
    MotorTopologyScalarEvidence {
        count,
        minimum: if count == 0 { 0.0 } else { minimum },
        maximum: if count == 0 { 0.0 } else { maximum },
        mean: if count == 0 { 0.0 } else { sum / count as f64 },
    }
}

/// Optimize the pole/yoke steel density for maximum small-signal BL.
pub fn optimize_motor_topology(
    driver: &AxisymmetricLinearConfig,
    config: &MotorTopologyConfig,
) -> Result<MotorTopologyResult, MotorTopologyError> {
    if config.max_iterations == 0
        || !config.tolerance.is_finite()
        || config.tolerance <= 0.0
        || !config.steel_volume_fraction.is_finite()
        || !(1.0e-3..=1.0).contains(&config.steel_volume_fraction)
    {
        return Err(MotorTopologyError::InvalidConfiguration(
            "iterations/tolerance/steel volume fraction are invalid",
        ));
    }
    if let Some(radius) = config.density_filter_radius_m
        && (!radius.is_finite() || radius <= 0.0)
    {
        return Err(MotorTopologyError::InvalidConfiguration(
            "density filter radius must be positive and finite",
        ));
    }
    if let Some(projection) = &config.projection
        && (!projection.beta_initial.is_finite()
            || projection.beta_initial <= 0.0
            || !projection.beta_max.is_finite()
            || projection.beta_max < projection.beta_initial
            || !projection.eta.is_finite()
            || !(0.0..1.0).contains(&projection.eta)
            || projection.continuation_interval == 0)
    {
        return Err(MotorTopologyError::InvalidConfiguration(
            "projection betas/eta/continuation interval are invalid",
        ));
    }
    for bridge in &config.bridges {
        if !bridge.r_min.is_finite()
            || !bridge.r_max.is_finite()
            || bridge.r_min > bridge.r_max
            || !bridge.z_min.is_finite()
            || !bridge.z_max.is_finite()
            || bridge.z_min > bridge.z_max
            || !bridge.min_average_density.is_finite()
            || !(1.0e-3..=1.0).contains(&bridge.min_average_density)
        {
            return Err(MotorTopologyError::InvalidConfiguration(
                "bridge box/minimum average density are invalid",
            ));
        }
    }
    if let Some(robust) = &config.robust {
        let Some(projection) = &config.projection else {
            return Err(MotorTopologyError::InvalidConfiguration(
                "robust formulation requires the Heaviside projection",
            ));
        };
        if !robust.delta_eta.is_finite()
            || robust.delta_eta <= 0.0
            || projection.eta - robust.delta_eta <= 0.0
            || projection.eta + robust.delta_eta >= 1.0
        {
            return Err(MotorTopologyError::InvalidConfiguration(
                "robust delta_eta must be positive with eta +/- delta_eta inside (0, 1)",
            ));
        }
    }
    if let SteelPermeability::DifferentialFromBias {
        refresh_interval, ..
    } = &config.steel_permeability
        && matches!(refresh_interval, Some(0))
    {
        return Err(MotorTopologyError::InvalidConfiguration(
            "permeability map refresh interval must be at least one iteration",
        ));
    }

    let mut workspace = MotorTopologyWorkspace::build(driver)?;
    let filter = config.density_filter_radius_m.map(|radius| {
        DensityFilter::build_from_elements(&workspace.mesh, radius, &workspace.design_elements)
    });
    let projection_config = config.projection;
    let mut beta = projection_config.map(|projection| projection.beta_initial);

    let centroids = design_centroids(&workspace.mesh, &workspace.design_elements);

    // Symmetry reduction: groups of design positions sharing one variable.
    // `reduced[g]` is the design variable of group `g`; the full design
    // vector is its expansion, and design sensitivities aggregate by
    // summation over members (exact chain rule, `d rho_i / d x_g = 1`).
    let groups = match config.symmetry {
        Some(axis) => symmetry_groups(&centroids, axis),
        None => (0..workspace.design_elements.len())
            .map(|position| vec![position])
            .collect(),
    };
    let group_volumes: Vec<f64> = groups
        .iter()
        .map(|group| group.iter().map(|&i| workspace.volumes[i]).sum())
        .collect();
    let num_design = workspace.design_elements.len();
    let expand = |reduced: &[f64]| {
        let mut full = vec![0.0; num_design];
        for (group, &value) in groups.iter().zip(reduced) {
            for &i in group {
                full[i] = value;
            }
        }
        full
    };
    let aggregate = |full_sensitivities: &[f64]| {
        groups
            .iter()
            .map(|group| group.iter().map(|&i| full_sensitivities[i]).sum())
            .collect::<Vec<f64>>()
    };
    // Design position -> group index (for bridge membership in reduced space).
    let mut group_of = vec![0_usize; workspace.design_elements.len()];
    for (group_index, group) in groups.iter().enumerate() {
        for &i in group {
            group_of[i] = group_index;
        }
    }

    // Bridge members (design positions), validated non-empty.
    let bridge_members = config
        .bridges
        .iter()
        .map(|bridge| {
            let members = bridge_members(&centroids, bridge);
            if members.is_empty() {
                Err(MotorTopologyError::InvalidConfiguration(
                    "bridge box contains no design cells",
                ))
            } else {
                Ok(members)
            }
        })
        .collect::<Result<Vec<_>, _>>()?;

    let total_volume = workspace.volumes.iter().sum::<f64>().max(f64::MIN_POSITIVE);
    let build_constraints = |reduced: &[f64]| {
        let volume = reduced
            .iter()
            .zip(&group_volumes)
            .map(|(&x, &v)| x * v)
            .sum::<f64>()
            / total_volume;
        let mut constraints = vec![MmaConstraint {
            grad: group_volumes.iter().map(|&v| v / total_volume).collect(),
            value: volume,
            bound: config.steel_volume_fraction,
        }];
        for (bridge, members) in config.bridges.iter().zip(&bridge_members) {
            let count = members.len() as f64;
            let mean = members.iter().map(|&i| reduced[group_of[i]]).sum::<f64>() / count;
            let mut grad = vec![0.0; groups.len()];
            for &i in members {
                grad[group_of[i]] -= 1.0 / count;
            }
            constraints.push(MmaConstraint {
                grad,
                value: -mean,
                bound: -bridge.min_average_density,
            });
        }
        constraints
    };

    // Feasible start for the bridge floors: pre-densify the bridge members to
    // their floor and rebalance the remaining groups so the steel volume
    // fraction is met exactly. With linear constraints the current design is
    // then feasible for every linearized MMA subproblem, so no subproblem
    // ever goes infeasible (the classic feasible-start MMA pattern; an
    // infeasible start sends the dual multipliers to their cap and wrecks
    // the asymptote state).
    let mut reduced = vec![config.steel_volume_fraction; groups.len()];
    if !config.bridges.is_empty() {
        let mut pinned: std::collections::BTreeMap<usize, f64> = std::collections::BTreeMap::new();
        for (bridge, members) in config.bridges.iter().zip(&bridge_members) {
            for &i in members {
                let value = pinned
                    .entry(group_of[i])
                    .or_insert(config.steel_volume_fraction);
                *value = value.max(bridge.min_average_density);
            }
        }
        for (&group, &value) in &pinned {
            reduced[group] = value;
        }
        let pinned_volume: f64 = pinned.keys().map(|&g| group_volumes[g]).sum();
        let pinned_steel: f64 = pinned
            .iter()
            .map(|(&g, &value)| value * group_volumes[g])
            .sum();
        let remaining_volume = total_volume - pinned_volume;
        let others_value = (config.steel_volume_fraction * total_volume - pinned_steel)
            / remaining_volume.max(f64::MIN_POSITIVE);
        if !(1.0e-3..=1.0).contains(&others_value) {
            return Err(MotorTopologyError::InvalidConfiguration(
                "bridge minimum average density is incompatible with the steel \
                 volume fraction (no feasible initial design)",
            ));
        }
        for (group, value) in reduced.iter_mut().enumerate() {
            if !pinned.contains_key(&group) {
                *value = others_value;
            }
        }
    }
    let mut optimizer = MmaOptimizer::new();
    let mut bl_history_tm = Vec::new();
    let mut permeability_map_history = Vec::new();
    let mut converged = false;

    for iteration in 0..config.max_iterations {
        // Beta continuation: double every `continuation_interval` iterations.
        if let (Some(projection), Some(current_beta)) = (projection_config, beta.as_mut())
            && iteration > 0
            && iteration % projection.continuation_interval == 0
        {
            *current_beta = (*current_beta * 2.0).min(projection.beta_max);
        }
        let nominal_projection = match (projection_config, beta) {
            (Some(projection), Some(current_beta)) => {
                Some(HeavisideProjection::new(current_beta, projection.eta))
            }
            _ => None,
        };

        let densities = expand(&reduced);

        // Differential permeability refresh: rebuild the map from the
        // current design's nonlinear bias solve at iteration 0 and every
        // `refresh_interval` iterations. Between refreshes the map is fixed,
        // so the adjoint derivatives below stay exact for the active map.
        if let SteelPermeability::DifferentialFromBias {
            bh_curve,
            solve,
            refresh_interval,
        } = &config.steel_permeability
        {
            let due = match refresh_interval {
                None => iteration == 0,
                Some(interval) => iteration % interval == 0,
            };
            if due {
                let (physical, _) =
                    physical_densities(&densities, filter.as_ref(), nominal_projection.as_ref());
                let interp_weights = physical
                    .iter()
                    .map(|&rho| config.interpolation.interpolate(rho.clamp(1.0e-3, 1.0)))
                    .collect::<Vec<_>>();
                let map = steel_permeability_map_from_bias(
                    &workspace.mesh,
                    driver.remanence_t,
                    &workspace.design_elements,
                    &interp_weights,
                    bh_curve,
                    solve,
                )
                .map_err(|error| MotorTopologyError::Fem(error.to_string()))?;
                workspace.set_steel_permeability_map(&map.differential_relative_permeability);
                permeability_map_history.push(map.differential_relative_permeability);
            }
        }
        // Objective (min -BL) and design sensitivities. Robust mode takes the
        // worst case over the dilated/nominal/eroded projections.
        let (bl_tm, sensitivities) = match (&config.robust, projection_config) {
            (Some(robust), Some(projection_config)) => {
                // `beta` is `Some` exactly when `projection_config` is (both
                // derive from it at construction), so this is unreachable
                // in practice — but typed, not a panic, if the invariant
                // ever breaks.
                let current_beta = beta.ok_or(MotorTopologyError::InvalidConfiguration(
                    "robust mode requires projection beta",
                ))?;
                let mut worst: Option<(f64, f64, Vec<f64>)> = None;
                for eta in [
                    projection_config.eta - robust.delta_eta,
                    projection_config.eta,
                    projection_config.eta + robust.delta_eta,
                ] {
                    let projection = HeavisideProjection::new(current_beta, eta);
                    let (physical, filtered) =
                        physical_densities(&densities, filter.as_ref(), Some(&projection));
                    let (bl, physical_sensitivities) =
                        workspace.objective_and_sensitivities(&physical, config.interpolation)?;
                    let branch = design_sensitivities(
                        &physical_sensitivities,
                        &filtered,
                        filter.as_ref(),
                        Some(&projection),
                    );
                    let objective = -bl;
                    if worst.as_ref().is_none_or(|(best, _, _)| objective > *best) {
                        worst = Some((objective, bl, branch));
                    }
                }
                let (_, bl, branch) = worst.expect("three projection branches evaluated");
                (bl, branch)
            }
            _ => {
                let (physical, filtered) =
                    physical_densities(&densities, filter.as_ref(), nominal_projection.as_ref());
                let (bl_tm, physical_sensitivities) =
                    workspace.objective_and_sensitivities(&physical, config.interpolation)?;
                let sensitivities = design_sensitivities(
                    &physical_sensitivities,
                    &filtered,
                    filter.as_ref(),
                    nominal_projection.as_ref(),
                );
                (bl_tm, sensitivities)
            }
        };
        bl_history_tm.push(bl_tm);
        #[cfg(test)]
        if std::env::var("MOTOR_TOPOLOGY_DEBUG").is_ok() {
            let volume_fraction = reduced
                .iter()
                .zip(&group_volumes)
                .map(|(&x, &v)| x * v)
                .sum::<f64>()
                / total_volume;
            let bridge_means: Vec<f64> = bridge_members
                .iter()
                .map(|members| {
                    members.iter().map(|&i| reduced[group_of[i]]).sum::<f64>()
                        / members.len() as f64
                })
                .collect();
            eprintln!(
                "iter {iteration}: bl={bl_tm:.6} vol={volume_fraction:.4} bridges={bridge_means:?}"
            );
        }

        let reduced_sensitivities = aggregate(&sensitivities);
        let constraints = build_constraints(&reduced);
        let updated = optimizer.update_multi(
            &reduced_sensitivities,
            &reduced,
            &constraints,
            &group_volumes,
        );
        let max_change = updated
            .iter()
            .zip(&reduced)
            .map(|(&new, &old)| (new - old).abs())
            .fold(0.0, f64::max);
        reduced = updated;
        if max_change < config.tolerance
            && projection_continuation_complete(projection_config, beta)
        {
            converged = true;
            break;
        }
    }

    let densities = expand(&reduced);

    // Gray fraction of the final physical design at the final continuation beta.
    let final_projection = match (projection_config, beta) {
        (Some(projection), Some(current_beta)) => {
            Some(HeavisideProjection::new(current_beta, projection.eta))
        }
        _ => None,
    };
    let (final_physical, final_filtered) =
        physical_densities(&densities, filter.as_ref(), final_projection.as_ref());
    let total_volume = workspace.volumes.iter().sum::<f64>();
    let design_evidence = density_evidence(&densities, &workspace.volumes, total_volume);
    let filtered_evidence = density_evidence(&final_filtered, &workspace.volumes, total_volume);
    let physical_evidence = density_evidence(&final_physical, &workspace.volumes, total_volume);
    // `beta` is `Some` exactly when `projection_config` is (both derive
    // from it at construction): the mismatch arm is unreachable in
    // practice, but typed rather than a panic.
    let projection_evidence = match projection_config {
        Some(projection) => {
            let beta_final = beta.ok_or(MotorTopologyError::InvalidConfiguration(
                "projection evidence requires projection beta",
            ))?;
            Some(MotorTopologyProjectionEvidence {
                beta_initial: projection.beta_initial,
                beta_final,
                beta_max: projection.beta_max,
                eta: projection.eta,
                continuation_complete: projection_continuation_complete(projection_config, beta),
            })
        }
        None => None,
    };
    // Same single-source invariant as above: `projection_config` and
    // `beta` are `Some` together, so the `None` arms are unreachable in
    // practice but typed rather than panics.
    let robust_evidence = match config.robust {
        Some(robust) => {
            let projection = projection_config.ok_or(MotorTopologyError::InvalidConfiguration(
                "robust mode requires projection",
            ))?;
            let current_beta = beta.ok_or(MotorTopologyError::InvalidConfiguration(
                "robust mode requires projection beta",
            ))?;
            let physical_at_eta = |eta: f64| {
                let projection = HeavisideProjection::new(current_beta, eta);
                physical_densities(&densities, filter.as_ref(), Some(&projection)).0
            };
            let dilated = physical_at_eta(projection.eta - robust.delta_eta);
            let eroded = physical_at_eta(projection.eta + robust.delta_eta);
            Some(MotorTopologyRobustEvidence {
                delta_eta: robust.delta_eta,
                dilated: density_evidence(&dilated, &workspace.volumes, total_volume),
                nominal: physical_evidence.clone(),
                eroded: density_evidence(&eroded, &workspace.volumes, total_volume),
            })
        }
        None => None,
    };
    let bridges = config
        .bridges
        .iter()
        .zip(&bridge_members)
        .map(|(bridge, members)| {
            let final_average_density = members
                .iter()
                .map(|&position| densities[position])
                .sum::<f64>()
                / members.len() as f64;
            let margin = final_average_density - bridge.min_average_density;
            MotorTopologyBridgeEvidence {
                r_min: bridge.r_min,
                r_max: bridge.r_max,
                z_min: bridge.z_min,
                z_max: bridge.z_max,
                minimum_average_density: bridge.min_average_density,
                final_average_density,
                margin,
                satisfied: margin >= -1.0e-9,
            }
        })
        .collect::<Vec<_>>();
    let bridge_constraints_satisfied = bridges.iter().all(|bridge| bridge.satisfied);
    let permeability = MotorTopologyPermeabilityEvidence {
        model: match config.steel_permeability {
            SteelPermeability::Constant => "constant",
            SteelPermeability::DifferentialFromBias { .. } => "differential_from_bias",
        },
        extraction_count: permeability_map_history.len(),
        final_differential_relative_permeability: permeability_map_history
            .last()
            .map(|map| scalar_evidence(map)),
    };
    let manufacturability = MotorTopologyManufacturabilityReport {
        scope: "axisymmetric design-cell evidence; bridge constraints are mean-density connectivity proxies, not 3D manufacturing certification",
        design_element_count: workspace.design_elements.len(),
        design_variable_count: groups.len(),
        design: design_evidence.clone(),
        filtered: filtered_evidence,
        physical: physical_evidence,
        density_filter_radius_m: config.density_filter_radius_m,
        projection: projection_evidence,
        symmetry: config.symmetry.map(|axis| format!("{axis:?}")),
        bridges,
        robust: robust_evidence,
        permeability,
        volume_constraint_satisfied: design_evidence.volume_fraction
            <= config.steel_volume_fraction + 1.0e-9,
        bridge_constraints_satisfied,
    };

    Ok(MotorTopologyResult {
        design_elements: workspace.design_elements,
        densities,
        iterations: bl_history_tm.len(),
        bl_history_tm,
        converged,
        gray_fraction: gray_fraction(&final_physical),
        permeability_map_history,
        manufacturability,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture_driver() -> AxisymmetricLinearConfig {
        AxisymmetricLinearConfig {
            magnetic_radial_cells: 8,
            magnetic_axial_cells: 8,
            structural_radial_cells: 6,
            structural_axial_cells: 1,
            acoustic_axial_cells: 4,
            ..AxisymmetricLinearConfig::default()
        }
    }

    /// Reference central finite difference of `-BL` over every state DOF, kept
    /// verbatim from the pre-D1 implementation to pin the analytic
    /// `bl_state_derivative` against it.
    fn fd_bl_state_derivative(
        driver: &AxisymmetricLinearConfig,
        mesh: &Mesh,
        potential: &[f64],
    ) -> Vec<f64> {
        let step = 1.0e-7;
        let mut derivative = vec![0.0; potential.len()];
        for dof in 0..potential.len() {
            let mut plus = potential.to_vec();
            let mut minus = potential.to_vec();
            plus[dof] += step;
            minus[dof] -= step;
            let bl_plus = magnetic_bias_from_potential(driver, mesh, plus)
                .unwrap()
                .distributed_bl_tm;
            let bl_minus = magnetic_bias_from_potential(driver, mesh, minus)
                .unwrap()
                .distributed_bl_tm;
            derivative[dof] = -(bl_plus - bl_minus) / (2.0 * step);
        }
        derivative
    }

    /// D1: analytic `d(-BL)/du` matches the finite-difference reference.
    /// `-BL` is piecewise linear in `u`, so the central FD is exact up to
    /// roundoff (`eps * |BL| / step ~ 1e-9` absolute); a 1e-6 relative bound
    /// against the largest component is comfortable.
    #[test]
    fn analytic_state_derivative_matches_finite_difference() {
        let driver = fixture_driver();
        let workspace = MotorTopologyWorkspace::build(&driver).unwrap();
        let interpolation = MotorTopologyConfig::default().interpolation;
        let densities = vec![0.65; workspace.design_elements.len()];
        let (potential, _) = workspace.solve_forward(&densities, interpolation).unwrap();

        let analytic = workspace.bl_state_derivative(&potential);
        let reference = fd_bl_state_derivative(&driver, &workspace.mesh, &potential);

        let scale = reference
            .iter()
            .map(|value| value.abs())
            .fold(0.0_f64, f64::max)
            .max(1.0e-12);
        let max_relative = analytic
            .iter()
            .zip(&reference)
            .map(|(&analytic, &reference)| (analytic - reference).abs() / scale)
            .fold(0.0_f64, f64::max);
        eprintln!("state derivative max relative error vs FD: {max_relative:e}");
        assert!(
            max_relative < 1.0e-6,
            "analytic d(-BL)/du deviates from FD reference by {max_relative:e}"
        );
    }

    /// F4 regression + D1 tightening: analytic adjoint density sensitivities
    /// (objective min −BL) against central finite differences of the re-solved
    /// objective. Both sides now use the same analytic forward model and the
    /// same factorization, so the error is FD step roundoff only
    /// (`eps * |J| / step ~ 1e-9` relative).
    #[test]
    fn adjoint_density_sensitivities_match_finite_difference() {
        let driver = fixture_driver();
        let workspace = MotorTopologyWorkspace::build(&driver).unwrap();
        assert!(!workspace.design_elements.is_empty());
        let interpolation = MotorTopologyConfig::default().interpolation;
        let densities = vec![0.65; workspace.design_elements.len()];
        let (_, sensitivities) = workspace
            .objective_and_sensitivities(&densities, interpolation)
            .unwrap();

        let objective = |densities: &[f64]| workspace.objective(densities, interpolation).unwrap();
        let step = 1.0e-4;
        let mut worst = 0.0_f64;
        for design_index in [
            0,
            workspace.design_elements.len() / 2,
            workspace.design_elements.len() - 1,
        ] {
            let mut plus = densities.clone();
            let mut minus = densities.clone();
            plus[design_index] += step;
            minus[design_index] -= step;
            let finite_difference = (objective(&plus) - objective(&minus)) / (2.0 * step);
            let adjoint = sensitivities[design_index];
            let scale = finite_difference.abs().max(adjoint.abs()).max(1.0e-12);
            let relative = (adjoint - finite_difference).abs() / scale;
            worst = worst.max(relative);
            eprintln!(
                "design cell {design_index}: adjoint={adjoint:e}, FD={finite_difference:e}, rel={relative:e}"
            );
        }
        assert!(
            worst < 1.0e-5,
            "analytic adjoint sensitivities deviate from FD by {worst:e}"
        );
    }

    /// D1: the sparse real Cholesky path reproduces the dense real LU
    /// forward solution to solver precision on the 8x8 fixture.
    #[test]
    fn sparse_cholesky_matches_dense_lu_on_fixture() {
        let driver = fixture_driver();
        let workspace = MotorTopologyWorkspace::build(&driver).unwrap();
        let interpolation = MotorTopologyConfig::default().interpolation;
        let densities = vec![0.65; workspace.design_elements.len()];
        let matrix = workspace.assemble_matrix(&densities, interpolation);

        let factorization = MotorFactorization::factorize(&matrix).unwrap();
        assert!(
            matches!(factorization, MotorFactorization::SparseCholesky(_)),
            "motor system should factorize as SPD via sparse Cholesky"
        );
        let cholesky = factorization.solve(&workspace.base_rhs).unwrap();

        let dense = matrix.to_csr().to_dense();
        let lu = sonium_solvers::lu_solve(&dense, &workspace.base_rhs).unwrap();

        let scale = lu
            .iter()
            .map(|value| value.abs())
            .fold(0.0_f64, f64::max)
            .max(1.0e-30);
        let max_relative = cholesky
            .iter()
            .zip(&lu)
            .map(|(&cholesky, &lu)| (cholesky - lu).abs() / scale)
            .fold(0.0_f64, f64::max);
        eprintln!("Cholesky vs dense LU max relative difference: {max_relative:e}");
        assert!(
            max_relative < 1.0e-10,
            "sparse Cholesky solution deviates from dense LU by {max_relative:e}"
        );
    }

    #[test]
    fn dense_lu_fallback_solves_symmetric_indefinite_fixture() {
        let mut matrix = TripletMatrix::new(2, 2);
        matrix.push(0, 1, 1.0);
        matrix.push(1, 0, 1.0);

        let factorization = MotorFactorization::factorize(&matrix).unwrap();
        assert!(
            matches!(factorization, MotorFactorization::DenseLu(_)),
            "an indefinite matrix must use the pivoted dense-LU fallback"
        );
        let solution = factorization.solve(&[1.0, 2.0]).unwrap();
        assert!((solution[0] - 2.0).abs() < 1.0e-12);
        assert!((solution[1] - 1.0).abs() < 1.0e-12);
    }

    #[test]
    fn projection_continuation_blocks_early_convergence() {
        let projection = ProjectionContinuationConfig {
            beta_initial: 1.0,
            beta_max: 8.0,
            eta: 0.5,
            continuation_interval: 2,
        };
        assert!(!projection_continuation_complete(
            Some(projection),
            Some(1.0)
        ));
        assert!(projection_continuation_complete(
            Some(projection),
            Some(8.0)
        ));
        assert!(projection_continuation_complete(None, None));
    }

    /// D1: chain rule through the density filter and Heaviside projection
    /// matches central FD of the filtered+projected objective with respect to
    /// the design variables.
    #[test]
    fn filtered_projected_sensitivities_match_finite_difference() {
        let driver = fixture_driver();
        let workspace = MotorTopologyWorkspace::build(&driver).unwrap();
        let interpolation = MotorTopologyConfig::default().interpolation;

        // Filter radius: ~2.5 radial cell widths of the 8-cell fixture mesh.
        let max_radius = workspace
            .mesh
            .nodes
            .iter()
            .map(|node| node.x)
            .fold(0.0_f64, f64::max);
        let filter = DensityFilter::build_from_elements(
            &workspace.mesh,
            2.5 * max_radius / 8.0,
            &workspace.design_elements,
        );
        let projection = HeavisideProjection::new(4.0, 0.5);

        // Non-uniform design densities so the filter mixing is exercised.
        let design = (0..workspace.design_elements.len())
            .map(|index| 0.25 + 0.5 * ((index * 37 % 101) as f64 / 100.0))
            .collect::<Vec<_>>();

        let (physical, filtered) = physical_densities(&design, Some(&filter), Some(&projection));
        let (_, physical_sensitivities) = workspace
            .objective_and_sensitivities(&physical, interpolation)
            .unwrap();
        let sensitivities = design_sensitivities(
            &physical_sensitivities,
            &filtered,
            Some(&filter),
            Some(&projection),
        );

        let objective = |design: &[f64]| {
            let (physical, _) = physical_densities(design, Some(&filter), Some(&projection));
            workspace.objective(&physical, interpolation).unwrap()
        };
        let step = 1.0e-4;
        let mut worst = 0.0_f64;
        for design_index in [
            0,
            workspace.design_elements.len() / 3,
            workspace.design_elements.len() / 2,
            workspace.design_elements.len() - 1,
        ] {
            let mut plus = design.clone();
            let mut minus = design.clone();
            plus[design_index] += step;
            minus[design_index] -= step;
            let finite_difference = (objective(&plus) - objective(&minus)) / (2.0 * step);
            let chain_rule = sensitivities[design_index];
            let scale = finite_difference.abs().max(chain_rule.abs()).max(1.0e-12);
            let relative = (chain_rule - finite_difference).abs() / scale;
            worst = worst.max(relative);
            eprintln!(
                "filtered cell {design_index}: chain={chain_rule:e}, FD={finite_difference:e}, rel={relative:e}"
            );
        }
        assert!(
            worst < 1.0e-5,
            "filter/projection chain rule deviates from FD by {worst:e}"
        );
    }

    /// D1: beta continuation drives the gray-level fraction below that of an
    /// unprojected run on the same fixture.
    #[test]
    fn continuation_reduces_gray_fraction() {
        let driver = fixture_driver();
        let base = MotorTopologyConfig {
            max_iterations: 12,
            tolerance: 1.0e-12, // force all 12 iterations
            steel_volume_fraction: 0.65,
            ..MotorTopologyConfig::default()
        };
        let plain = optimize_motor_topology(&driver, &base).unwrap();
        let projected = optimize_motor_topology(
            &driver,
            &MotorTopologyConfig {
                projection: Some(ProjectionContinuationConfig {
                    beta_initial: 1.0,
                    beta_max: 16.0,
                    eta: 0.5,
                    continuation_interval: 4,
                }),
                ..base.clone()
            },
        )
        .unwrap();
        eprintln!(
            "gray fraction: plain={:.4}, projected={:.4}",
            plain.gray_fraction, projected.gray_fraction
        );
        assert!(
            (0.0..=1.0).contains(&plain.gray_fraction)
                && (0.0..=1.0).contains(&projected.gray_fraction)
        );
        assert!(
            projected.gray_fraction < plain.gray_fraction,
            "continuation should binarize the design: projected gray fraction \
             {:.4} not below unprojected {:.4}",
            projected.gray_fraction,
            plain.gray_fraction
        );
    }

    #[test]
    fn motor_topology_runs_adjoint_mma_on_pole_and_yoke_cells() {
        let driver = fixture_driver();
        let result = optimize_motor_topology(
            &driver,
            &MotorTopologyConfig {
                max_iterations: 2,
                tolerance: 1.0e-12,
                steel_volume_fraction: 0.7,
                ..MotorTopologyConfig::default()
            },
        )
        .unwrap();
        assert_eq!(result.iterations, 2);
        assert!(!result.design_elements.is_empty());
        assert!(
            result
                .bl_history_tm
                .iter()
                .all(|bl| bl.is_finite() && *bl > 0.0)
        );
        assert!(
            result
                .densities
                .iter()
                .all(|density| (1.0e-3..=1.0).contains(density))
        );
        assert!(
            result
                .densities
                .windows(2)
                .any(|pair| (pair[0] - pair[1]).abs() > 1.0e-6),
            "adjoint sensitivities should create a non-uniform steel layout"
        );
        assert_eq!(
            result.manufacturability.design_element_count,
            result.design_elements.len()
        );
        assert_eq!(
            result.manufacturability.design_variable_count,
            result.design_elements.len()
        );
        assert!(result.manufacturability.volume_constraint_satisfied);
        assert!(result.manufacturability.bridge_constraints_satisfied);
        assert_eq!(result.manufacturability.bridges.len(), 0);
        let report_json = serde_json::to_value(&result.manufacturability).unwrap();
        assert_eq!(
            report_json["scope"],
            "axisymmetric design-cell evidence; bridge constraints are mean-density connectivity proxies, not 3D manufacturing certification"
        );
    }

    /// Beta used at the last executed iteration of a run (`iterations` is the
    /// number of completed loop passes).
    fn final_beta(config: &MotorTopologyConfig, iterations: usize) -> Option<f64> {
        final_continuation_beta(config, iterations)
    }

    /// BL of a run's final design projected at `eta + eta_offset` with the
    /// final continuation beta (e.g. the eroded design at `+delta_eta`).
    fn projected_final_bl(
        driver: &AxisymmetricLinearConfig,
        config: &MotorTopologyConfig,
        result: &MotorTopologyResult,
        eta_offset: f64,
    ) -> f64 {
        let workspace = MotorTopologyWorkspace::build(driver).unwrap();
        let filter = config.density_filter_radius_m.map(|radius| {
            DensityFilter::build_from_elements(&workspace.mesh, radius, &workspace.design_elements)
        });
        let projection = final_beta(config, result.iterations).map(|beta| {
            HeavisideProjection::new(beta, config.projection.unwrap().eta + eta_offset)
        });
        let (physical, _) =
            physical_densities(&result.densities, filter.as_ref(), projection.as_ref());
        let (_, bl_tm) = workspace
            .solve_forward(&physical, config.interpolation)
            .unwrap();
        bl_tm
    }

    /// D2 symmetry: mirrored pairs end with identical densities, and the
    /// symmetric run's objective tracks the unrestricted run started from the
    /// same (symmetric) uniform layout. The fixture physics is mildly
    /// z-asymmetric (yoke bottom plate), so the unrestricted run drifts out
    /// of the symmetric subspace; the comparison tolerance reflects that.
    #[test]
    fn symmetry_reduction_preserves_pair_symmetry_and_objective() {
        let driver = fixture_driver();
        let base = MotorTopologyConfig {
            max_iterations: 4,
            tolerance: 1.0e-12,
            steel_volume_fraction: 0.65,
            ..MotorTopologyConfig::default()
        };
        let plain = optimize_motor_topology(&driver, &base).unwrap();
        let symmetric = optimize_motor_topology(
            &driver,
            &MotorTopologyConfig {
                symmetry: Some(SymmetryAxis::R),
                ..base.clone()
            },
        )
        .unwrap();

        let workspace = MotorTopologyWorkspace::build(&driver).unwrap();
        let centroids = design_centroids(&workspace.mesh, &workspace.design_elements);
        let groups = symmetry_groups(&centroids, SymmetryAxis::R);
        let multi: Vec<&Vec<usize>> = groups.iter().filter(|group| group.len() >= 2).collect();
        let grouped_cells = multi.iter().map(|group| group.len()).sum::<usize>();
        let coverage = grouped_cells as f64 / centroids.len() as f64;
        eprintln!(
            "symmetry: {} multi-cell groups covering {grouped_cells}/{} cells (coverage {coverage:.2})",
            multi.len(),
            centroids.len()
        );
        assert!(
            coverage >= 0.4,
            "expected a substantial grouped fraction, got {coverage:.2}"
        );
        for group in &multi {
            let reference = symmetric.densities[group[0]];
            for &i in group.iter() {
                assert!(
                    (symmetric.densities[i] - reference).abs() <= 1.0e-12,
                    "group {group:?} diverged: {} vs {reference}",
                    symmetric.densities[i]
                );
            }
        }

        let bl_plain = plain.bl_history_tm.last().unwrap();
        let bl_symmetric = symmetric.bl_history_tm.last().unwrap();
        let relative = (bl_symmetric - bl_plain).abs() / bl_plain.abs().max(1.0e-12);
        eprintln!(
            "symmetry objective: plain={bl_plain:.6}, symmetric={bl_symmetric:.6}, rel={relative:e}"
        );
        assert!(
            relative <= 0.05,
            "symmetric run objective deviates from unrestricted run by {relative:e}"
        );
    }

    /// D2 bridge: a mean-density floor over the yoke bottom plate keeps the
    /// bridge at or above the bound, where the unconstrained run thins it.
    #[test]
    fn bridge_constraint_enforces_minimum_average_density() {
        let driver = fixture_driver();
        // Yoke bottom plate: steel connecting pole to the outer back iron.
        let bridge = BridgeConstraint {
            r_min: 0.0,
            r_max: 30.0e-3,
            z_min: -20.0e-3,
            z_max: -10.0e-3,
            min_average_density: 0.0, // set after measuring the free run
        };
        let base = MotorTopologyConfig {
            max_iterations: 8,
            tolerance: 1.0e-12,
            steel_volume_fraction: 0.65,
            ..MotorTopologyConfig::default()
        };
        let plain = optimize_motor_topology(&driver, &base).unwrap();

        let workspace = MotorTopologyWorkspace::build(&driver).unwrap();
        let centroids = design_centroids(&workspace.mesh, &workspace.design_elements);
        let members = bridge_members(&centroids, &bridge);
        assert!(!members.is_empty(), "bridge box should select design cells");
        let mean_of = |densities: &[f64]| {
            members.iter().map(|&i| densities[i]).sum::<f64>() / members.len() as f64
        };
        let plain_mean = mean_of(&plain.densities);
        // Require noticeably more bridge steel than the free optimum so the
        // constraint is active at the solution (but stay jointly feasible
        // with the volume cap).
        assert!(
            plain_mean < 0.985,
            "fixture drifted: unconstrained bridge mean {plain_mean:.4} too high for this test"
        );
        let min_required = (plain_mean + 0.03).min(0.99);
        eprintln!("bridge: unconstrained mean={plain_mean:.4}, requiring {min_required:.4}");

        let constrained = optimize_motor_topology(
            &driver,
            &MotorTopologyConfig {
                bridges: vec![BridgeConstraint {
                    min_average_density: min_required,
                    ..bridge
                }],
                ..base.clone()
            },
        )
        .unwrap();
        let constrained_mean = mean_of(&constrained.densities);
        eprintln!("bridge: constrained mean={constrained_mean:.4}");
        assert!(
            constrained_mean >= min_required - 1.0e-6,
            "bridge mean {constrained_mean:.4} below required {min_required:.4}"
        );
        assert!(
            constrained_mean >= plain_mean + 0.01,
            "constraint should actively raise the bridge mean ({constrained_mean:.4} vs {plain_mean:.4})"
        );
        // Volume constraint still holds (volume-weighted mean over all cells).
        let volume_fraction = constrained
            .densities
            .iter()
            .zip(&workspace.volumes)
            .map(|(&density, &volume)| density * volume)
            .sum::<f64>()
            / workspace.volumes.iter().sum::<f64>();
        assert!(volume_fraction <= base.steel_volume_fraction + 1.0e-6);
    }

    /// D2 robust: worst-case (eroded) BL of the robust run beats/matches the
    /// eroded BL of the nominal run, and beta continuation still binarizes.
    #[test]
    fn robust_continuation_improves_eroded_bl() {
        let driver = fixture_driver();
        let projection = ProjectionContinuationConfig {
            beta_initial: 1.0,
            beta_max: 16.0,
            eta: 0.5,
            continuation_interval: 4,
        };
        let delta_eta = 0.15;
        let base = MotorTopologyConfig {
            max_iterations: 12,
            tolerance: 1.0e-12,
            steel_volume_fraction: 0.65,
            projection: Some(projection),
            ..MotorTopologyConfig::default()
        };
        let plain = optimize_motor_topology(&driver, &base).unwrap();
        let start = std::time::Instant::now();
        let robust = optimize_motor_topology(
            &driver,
            &MotorTopologyConfig {
                robust: Some(RobustConfig { delta_eta }),
                ..base.clone()
            },
        )
        .unwrap();
        let robust_elapsed = start.elapsed();
        eprintln!(
            "robust run: {:?} for {} iterations ({:?}/iteration, 3 solves each)",
            robust_elapsed,
            robust.iterations,
            robust_elapsed / robust.iterations.max(1) as u32
        );

        let plain_eroded = projected_final_bl(&driver, &base, &plain, delta_eta);
        let robust_config = MotorTopologyConfig {
            robust: Some(RobustConfig { delta_eta }),
            ..base.clone()
        };
        let robust_eroded = projected_final_bl(&driver, &robust_config, &robust, delta_eta);
        let plain_nominal = projected_final_bl(&driver, &base, &plain, 0.0);
        let robust_nominal = projected_final_bl(&driver, &robust_config, &robust, 0.0);
        eprintln!(
            "eroded BL: plain={plain_eroded:.6}, robust={robust_eroded:.6} \
             (nominal: plain={plain_nominal:.6}, robust={robust_nominal:.6})"
        );
        assert!(
            robust_eroded >= 0.98 * plain_eroded,
            "robust eroded BL {robust_eroded:.6} below plain eroded BL {plain_eroded:.6}"
        );

        // Continuation still binarizes compared to an unprojected run.
        let unprojected = optimize_motor_topology(
            &driver,
            &MotorTopologyConfig {
                projection: None,
                ..base.clone()
            },
        )
        .unwrap();
        eprintln!(
            "gray fraction: unprojected={:.4}, robust={:.4}",
            unprojected.gray_fraction, robust.gray_fraction
        );
        assert!(
            robust.gray_fraction < unprojected.gray_fraction,
            "robust continuation should still reduce the gray fraction"
        );
    }

    /// D2 minimum length scale: with density filter radius R and projection,
    /// the converged layout contains no isolated steel speck — every steel
    /// cell has another steel cell within 2R (proxy for the classical
    /// minimum-feature-size ~2R of density filtering).
    #[test]
    fn filter_radius_sets_minimum_feature_size() {
        let driver = fixture_driver();
        let workspace = MotorTopologyWorkspace::build(&driver).unwrap();
        let max_radius = workspace
            .mesh
            .nodes
            .iter()
            .map(|node| node.x)
            .fold(0.0_f64, f64::max);
        let radius = 2.5 * max_radius / 8.0;
        let base = MotorTopologyConfig {
            max_iterations: 12,
            tolerance: 1.0e-12,
            steel_volume_fraction: 0.65,
            density_filter_radius_m: Some(radius),
            projection: Some(ProjectionContinuationConfig {
                beta_initial: 1.0,
                beta_max: 16.0,
                eta: 0.5,
                continuation_interval: 4,
            }),
            ..MotorTopologyConfig::default()
        };
        let filtered = optimize_motor_topology(&driver, &base).unwrap();
        let unfiltered = optimize_motor_topology(
            &driver,
            &MotorTopologyConfig {
                density_filter_radius_m: None,
                ..base.clone()
            },
        )
        .unwrap();

        let centroids = design_centroids(&workspace.mesh, &workspace.design_elements);
        // Final physical densities (nominal eta, final beta).
        let final_physical = |config: &MotorTopologyConfig, result: &MotorTopologyResult| {
            let filter = config.density_filter_radius_m.map(|r| {
                DensityFilter::build_from_elements(&workspace.mesh, r, &workspace.design_elements)
            });
            let projection = final_beta(config, result.iterations)
                .map(|beta| HeavisideProjection::new(beta, config.projection.unwrap().eta));
            physical_densities(&result.densities, filter.as_ref(), projection.as_ref()).0
        };
        // Count steel cells (rho >= 0.5) without a steel neighbor within 2R.
        let isolated_specks = |physical: &[f64]| {
            let steel = |i: usize| physical[i] >= 0.5;
            (0..physical.len())
                .filter(|&i| steel(i))
                .filter(|&i| {
                    !(0..physical.len()).any(|j| {
                        j != i && steel(j) && {
                            let dr = centroids[i][0] - centroids[j][0];
                            let dz = centroids[i][1] - centroids[j][1];
                            (dr * dr + dz * dz).sqrt() <= 2.0 * radius
                        }
                    })
                })
                .count()
        };
        let filtered_physical = final_physical(&base, &filtered);
        let unfiltered_physical = final_physical(
            &MotorTopologyConfig {
                density_filter_radius_m: None,
                ..base.clone()
            },
            &unfiltered,
        );
        let steel_count = filtered_physical.iter().filter(|&&rho| rho >= 0.5).count();
        let air_count = filtered_physical.iter().filter(|&&rho| rho < 0.5).count();
        let isolated_filtered = isolated_specks(&filtered_physical);
        let isolated_unfiltered = isolated_specks(&unfiltered_physical);
        eprintln!(
            "min-length: radius={radius:.4} m, steel cells={steel_count}, air cells={air_count}, \
             isolated specks: filtered={isolated_filtered}, unfiltered={isolated_unfiltered}"
        );
        assert!(
            steel_count >= 2 && air_count >= 2,
            "layout should contain both steel and air for the proxy to be meaningful"
        );
        assert_eq!(
            isolated_filtered, 0,
            "filtered run has {isolated_filtered} steel cells without a steel neighbor within 2R"
        );
    }

    /// D2 validation: robust mode requires projection, bridges need member
    /// cells, and delta_eta must keep eta +/- delta_eta inside (0, 1).
    #[test]
    fn d2_option_validation() {
        let driver = fixture_driver();
        let projection = ProjectionContinuationConfig {
            beta_initial: 1.0,
            beta_max: 4.0,
            eta: 0.5,
            continuation_interval: 4,
        };
        // Robust without projection is rejected.
        assert!(
            optimize_motor_topology(
                &driver,
                &MotorTopologyConfig {
                    robust: Some(RobustConfig { delta_eta: 0.1 }),
                    ..MotorTopologyConfig::default()
                },
            )
            .is_err()
        );
        // Eroded/dilated thresholds must stay inside (0, 1).
        assert!(
            optimize_motor_topology(
                &driver,
                &MotorTopologyConfig {
                    projection: Some(projection),
                    robust: Some(RobustConfig { delta_eta: 0.6 }),
                    ..MotorTopologyConfig::default()
                },
            )
            .is_err()
        );
        // Bridge box selecting no design cells is rejected.
        assert!(
            optimize_motor_topology(
                &driver,
                &MotorTopologyConfig {
                    bridges: vec![BridgeConstraint {
                        r_min: 1.0,
                        r_max: 2.0,
                        z_min: 1.0,
                        z_max: 2.0,
                        min_average_density: 0.5,
                    }],
                    ..MotorTopologyConfig::default()
                },
            )
            .is_err()
        );
        // Inverted bridge box is rejected.
        assert!(
            optimize_motor_topology(
                &driver,
                &MotorTopologyConfig {
                    bridges: vec![BridgeConstraint {
                        r_min: 2.0,
                        r_max: 1.0,
                        z_min: 0.0,
                        z_max: 1.0,
                        min_average_density: 0.5,
                    }],
                    ..MotorTopologyConfig::default()
                },
            )
            .is_err()
        );
    }

    // ---- D3: differential permeability from the nonlinear bias solve ----

    use crate::nonlinear_bias::{
        default_low_carbon_steel_bh_curve, steel_permeability_map_from_bias,
    };

    /// Fast extraction-solve settings for the 8x8 fixture.
    fn bias_solve_config() -> NonlinearBiasSolveConfig {
        NonlinearBiasSolveConfig {
            continuation_steps: 4,
            ..NonlinearBiasSolveConfig::default()
        }
    }

    /// Straight-line B-H curve with constant relative permeability `mu_r`
    /// (the nonlinear machinery's exact analog of the linear steel model).
    fn linear_bh_curve(mu_r: f64) -> BhCurve {
        let permeability = mu_r * MU_0;
        BhCurve::new(
            [0.0, 0.5, 1.0, 1.5, 2.0, 3.0]
                .into_iter()
                .map(|b| (b, b / permeability))
                .collect(),
        )
        .unwrap()
    }

    /// Max of the per-element mean |B| over the steel cells (rho >= 0.5) of a
    /// design, from one density-aware nonlinear bias evaluation.
    fn max_steel_flux_density(
        workspace: &MotorTopologyWorkspace,
        densities: &[f64],
        interpolation: InterpolationScheme,
        curve: &BhCurve,
    ) -> f64 {
        let weights = densities
            .iter()
            .map(|&rho| interpolation.interpolate(rho.clamp(1.0e-3, 1.0)))
            .collect::<Vec<_>>();
        let map = steel_permeability_map_from_bias(
            &workspace.mesh,
            workspace.driver.remanence_t,
            &workspace.design_elements,
            &weights,
            curve,
            &bias_solve_config(),
        )
        .unwrap();
        densities
            .iter()
            .zip(&map.mean_flux_density_t)
            .filter(|&(&rho, _)| rho >= 0.5)
            .map(|(_, &b)| b)
            .fold(0.0_f64, f64::max)
    }

    /// D3 linear limit: with a straight-line B-H curve at `mu_r = 1000`, the
    /// extracted map is 1000 everywhere and the `DifferentialFromBias` run
    /// reproduces the `Constant` run to solver roundoff.
    #[test]
    fn linear_bh_curve_map_reproduces_constant_run() {
        let driver = fixture_driver();
        let base = MotorTopologyConfig {
            max_iterations: 6,
            tolerance: 1.0e-12,
            steel_volume_fraction: 0.65,
            ..MotorTopologyConfig::default()
        };
        let constant = optimize_motor_topology(&driver, &base).unwrap();
        let differential = optimize_motor_topology(
            &driver,
            &MotorTopologyConfig {
                steel_permeability: SteelPermeability::DifferentialFromBias {
                    bh_curve: linear_bh_curve(LINEAR_STEEL_MU_R),
                    solve: bias_solve_config(),
                    refresh_interval: None,
                },
                ..base.clone()
            },
        )
        .unwrap();

        assert_eq!(differential.permeability_map_history.len(), 1);
        let map = &differential.permeability_map_history[0];
        let max_map_deviation = map
            .iter()
            .map(|mu| (mu - LINEAR_STEEL_MU_R).abs())
            .fold(0.0_f64, f64::max);
        eprintln!("linear-curve map max deviation from 1000: {max_map_deviation:e}");
        assert!(
            max_map_deviation < 1.0e-6 * LINEAR_STEEL_MU_R,
            "linear B-H curve should yield mu_diff = 1000 everywhere, \
             max deviation {max_map_deviation:e}"
        );

        assert_eq!(constant.iterations, differential.iterations);
        let max_bl_relative = constant
            .bl_history_tm
            .iter()
            .zip(&differential.bl_history_tm)
            .map(|(a, b)| (a - b).abs() / a.abs().max(1.0e-12))
            .fold(0.0_f64, f64::max);
        let max_density_deviation = constant
            .densities
            .iter()
            .zip(&differential.densities)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        eprintln!(
            "linear limit: max BL rel deviation={max_bl_relative:e}, \
             max density deviation={max_density_deviation:e}"
        );
        assert!(max_bl_relative < 1.0e-6);
        assert!(max_density_deviation < 1.0e-6);
    }

    /// D3 saturation: the default fixture at its nominal drive already pushes
    /// the bottleneck steel past the B-H knee, so the extracted differential
    /// permeability collapses there. The differential-map optimum then widens
    /// the bottleneck: its peak steel flux density (evaluated with the same
    /// nonlinear solve for both designs) does not exceed the constant-mu
    /// optimum's.
    #[test]
    fn saturation_map_drops_and_shifts_the_optimum() {
        let driver = fixture_driver();
        let curve = default_low_carbon_steel_bh_curve();
        let workspace = MotorTopologyWorkspace::build(&driver).unwrap();
        let interpolation = MotorTopologyConfig::default().interpolation;

        // Map of the uniform starting design.
        let start_weights = vec![interpolation.interpolate(0.65); workspace.design_elements.len()];
        let start_map = steel_permeability_map_from_bias(
            &workspace.mesh,
            driver.remanence_t,
            &workspace.design_elements,
            &start_weights,
            &curve,
            &bias_solve_config(),
        )
        .unwrap();
        let mu = &start_map.differential_relative_permeability;
        let (mu_min, mu_max) = mu.iter().fold((f64::INFINITY, 0.0_f64), |(lo, hi), &m| {
            (lo.min(m), hi.max(m))
        });
        let max_b = start_map
            .mean_flux_density_t
            .iter()
            .copied()
            .fold(0.0_f64, f64::max);
        eprintln!("start map: mu_diff min={mu_min:.1}, max={mu_max:.1}, max mean|B|={max_b:.3} T");
        assert!(
            max_b > 1.4,
            "fixture should drive the bottleneck toward the B-H knee, max |B|={max_b:.3} T"
        );
        assert!(
            mu_min < 0.25 * mu_max,
            "saturated elements should lose most of their differential permeability: \
             min={mu_min:.1}, max={mu_max:.1}"
        );

        let base = MotorTopologyConfig {
            max_iterations: 12,
            tolerance: 1.0e-12,
            steel_volume_fraction: 0.65,
            ..MotorTopologyConfig::default()
        };
        let constant = optimize_motor_topology(&driver, &base).unwrap();
        let differential = optimize_motor_topology(
            &driver,
            &MotorTopologyConfig {
                steel_permeability: SteelPermeability::DifferentialFromBias {
                    bh_curve: curve.clone(),
                    solve: bias_solve_config(),
                    refresh_interval: Some(4),
                },
                ..base.clone()
            },
        )
        .unwrap();

        let designs_differ = constant
            .densities
            .iter()
            .zip(&differential.densities)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        eprintln!("max density difference between optima: {designs_differ:.4}");
        assert!(
            designs_differ > 1.0e-3,
            "saturation-aware model should shift the optimum"
        );

        let constant_max_b =
            max_steel_flux_density(&workspace, &constant.densities, interpolation, &curve);
        let differential_max_b =
            max_steel_flux_density(&workspace, &differential.densities, interpolation, &curve);
        eprintln!(
            "peak steel |B|: constant optimum={constant_max_b:.4} T, \
             differential optimum={differential_max_b:.4} T"
        );
        assert!(
            differential_max_b <= 1.001 * constant_max_b,
            "differential-map optimum should not raise the peak steel flux density: \
             {differential_max_b:.4} T vs {constant_max_b:.4} T"
        );
    }

    /// D3 sensitivity regression: with a fixed non-uniform permeability map
    /// installed, the analytic adjoint sensitivities still match central
    /// finite differences of the re-solved objective.
    #[test]
    fn fixed_permeability_map_preserves_adjoint_sensitivities() {
        let driver = fixture_driver();
        let mut workspace = MotorTopologyWorkspace::build(&driver).unwrap();
        let map = (0..workspace.design_elements.len())
            .map(|index| 200.0 + 1_600.0 * ((index * 37 % 101) as f64 / 100.0))
            .collect::<Vec<_>>();
        workspace.set_steel_permeability_map(&map);
        let interpolation = MotorTopologyConfig::default().interpolation;
        let densities = vec![0.65; workspace.design_elements.len()];
        let (_, sensitivities) = workspace
            .objective_and_sensitivities(&densities, interpolation)
            .unwrap();

        let objective = |densities: &[f64]| workspace.objective(densities, interpolation).unwrap();
        let step = 1.0e-4;
        let mut worst = 0.0_f64;
        for design_index in [
            0,
            workspace.design_elements.len() / 2,
            workspace.design_elements.len() - 1,
        ] {
            let mut plus = densities.clone();
            let mut minus = densities.clone();
            plus[design_index] += step;
            minus[design_index] -= step;
            let finite_difference = (objective(&plus) - objective(&minus)) / (2.0 * step);
            let adjoint = sensitivities[design_index];
            let scale = finite_difference.abs().max(adjoint.abs()).max(1.0e-12);
            let relative = (adjoint - finite_difference).abs() / scale;
            worst = worst.max(relative);
            eprintln!(
                "mapped cell {design_index}: adjoint={adjoint:e}, FD={finite_difference:e}, rel={relative:e}"
            );
        }
        assert!(
            worst < 1.0e-5,
            "adjoint sensitivities with a fixed permeability map deviate from FD by {worst:e}"
        );
    }

    /// D3 refresh: `refresh_interval = Some(2)` re-extracts the map during
    /// the run, the map actually changes between refreshes on the saturating
    /// fixture, and the run stays finite and in-bounds.
    #[test]
    fn refresh_interval_rebuilds_the_saturation_map() {
        let driver = fixture_driver();
        let result = optimize_motor_topology(
            &driver,
            &MotorTopologyConfig {
                max_iterations: 6,
                tolerance: 1.0e-12,
                steel_volume_fraction: 0.65,
                steel_permeability: SteelPermeability::DifferentialFromBias {
                    bh_curve: default_low_carbon_steel_bh_curve(),
                    solve: bias_solve_config(),
                    refresh_interval: Some(2),
                },
                ..MotorTopologyConfig::default()
            },
        )
        .unwrap();
        assert_eq!(result.iterations, 6);
        // Extractions at iterations 0, 2, 4.
        assert_eq!(result.permeability_map_history.len(), 3);
        for window in 1..result.permeability_map_history.len() {
            let drift = result.permeability_map_history[window]
                .iter()
                .zip(&result.permeability_map_history[window - 1])
                .map(|(new, old)| (new - old).abs())
                .fold(0.0_f64, f64::max);
            eprintln!("refresh {window}: max map change = {drift:.2}");
            assert!(
                drift > 1.0,
                "map should measurably change between refreshes on the saturating fixture"
            );
        }
        assert!(
            result
                .bl_history_tm
                .iter()
                .all(|bl| bl.is_finite() && *bl > 0.0)
        );
        assert!(
            result
                .densities
                .iter()
                .all(|density| (1.0e-3..=1.0).contains(density))
        );
    }

    /// D3 validation: a zero refresh interval is rejected.
    #[test]
    fn d3_option_validation() {
        let driver = fixture_driver();
        assert!(
            optimize_motor_topology(
                &driver,
                &MotorTopologyConfig {
                    steel_permeability: SteelPermeability::DifferentialFromBias {
                        bh_curve: default_low_carbon_steel_bh_curve(),
                        solve: bias_solve_config(),
                        refresh_interval: Some(0),
                    },
                    ..MotorTopologyConfig::default()
                },
            )
            .is_err()
        );
    }
}
