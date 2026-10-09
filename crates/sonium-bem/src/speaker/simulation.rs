//! BEM acoustic simulation pipeline for loudspeaker radiation.
//!
//! Runs a frequency sweep with velocity boundary conditions on the
//! driver apertures and evaluates the sound field at measurement points.
//!
//! Key difference from scattering problems: there is no incident field.
//! The RHS comes entirely from the non-zero velocity BCs on the drivers,
//! which `build_tbem_system_with_beta` already handles during assembly.

use num_complex::Complex64;
use sonium_solvers::dense::DenseMatrix;
use std::time::Instant;

use crate::core::assembly::tbem::build_tbem_system_with_beta;
use crate::core::postprocess::pressure::compute_scattered_field;
use crate::core::types::{
    BasisOrder, BoundaryCondition, Element, ElementProperty, ElementType, PhysicsParams,
};

use sonium_solvers::iterative::{GmresConfig, gmres_with_guess};
use sonium_solvers::traits::LinearOperator;

use super::mesh_gen::{MeshRegion, SpeakerMesh};
use super::model::LoudspeakerModel;
use super::model::{admittance_for_material, Z_HARD_PLASTIC, Z_MDF};

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// Result of the BEM simulation across all frequencies and measurement points.
pub struct SimulationResult {
    /// Frequency vector (Hz).
    pub frequencies: Vec<f64>,
    /// SPL at each measurement point for each frequency.
    /// Shape: `[freq_idx][point_idx]`.
    pub spl: Vec<Vec<f64>>,
    /// Phase (radians) at each measurement point for each frequency.
    /// Shape: `[freq_idx][point_idx]`.
    pub phase: Vec<Vec<f64>>,
}

/// Configuration for the BEM solver within the simulation.
pub struct SolverConfig {
    /// GMRES tolerance.
    pub tolerance: f64,
    /// Maximum GMRES iterations per frequency.
    pub max_iterations: usize,
    /// GMRES restart parameter.
    pub restart: usize,
    /// Burton-Miller beta scale factor.
    pub beta_scale: f64,
    /// Whether to recycle the previous solution as initial guess.
    pub recycle_solution: bool,
}

impl Default for SolverConfig {
    fn default() -> Self {
        Self {
            tolerance: 1e-6,
            max_iterations: 200,
            restart: 30,
            beta_scale: 4.0,
            recycle_solution: true,
        }
    }
}

// ---------------------------------------------------------------------------
// Dense operator wrapper (for GMRES)
// ---------------------------------------------------------------------------

struct DenseOperator<'a> {
    matrix: &'a DenseMatrix<Complex64>,
}

impl<'a> LinearOperator<Complex64> for DenseOperator<'a> {
    fn num_rows(&self) -> usize {
        self.matrix.nrows()
    }
    fn num_cols(&self) -> usize {
        self.matrix.ncols()
    }
    fn apply(&self, x: &[Complex64]) -> Vec<Complex64> {
        self.matrix.matvec(x)
    }
    fn apply_into(&self, x: &[Complex64], y: &mut [Complex64]) {
        self.matrix.matvec_into(x, y);
    }
    fn apply_block(&self, x: &DenseMatrix<Complex64>, y: &mut DenseMatrix<Complex64>) {
        // One row-major GEMM (A n×n · X n×k → Y n×k) instead of k matvecs.
        self.matrix.matmul_into(x, y);
    }
    fn apply_transpose(&self, x: &[Complex64]) -> Vec<Complex64> {
        let mut y = vec![Complex64::new(0.0, 0.0); self.matrix.ncols()];
        self.matrix.matvec_transpose_into(x, &mut y);
        y
    }
    fn apply_transpose_into(&self, x: &[Complex64], y: &mut [Complex64]) {
        self.matrix.matvec_transpose_into(x, y);
    }
}

// ---------------------------------------------------------------------------
// Core simulation
// ---------------------------------------------------------------------------

/// Run the full BEM simulation pipeline.
///
/// For each frequency:
/// 1. Update driver element BCs with frequency-dependent piston velocity
/// 2. Assemble BEM system (matrix + RHS from velocity BCs)
/// 3. Solve with GMRES and solution recycling
/// 4. Evaluate scattered field at the given evaluation points
///
/// Returns SPL and phase at each eval point for each frequency.
pub fn run_simulation(
    model: &LoudspeakerModel,
    eval_points: &[[f64; 3]],
) -> SimulationResult {
    run_simulation_with_config(model, eval_points, &SolverConfig::default())
}

/// Run the BEM simulation with custom solver configuration.
pub fn run_simulation_with_config(
    model: &LoudspeakerModel,
    eval_points: &[[f64; 3]],
    solver_config: &SolverConfig,
) -> SimulationResult {
    let frequencies = model.simulation.frequencies();
    let c = model.simulation.speed_of_sound;
    let rho = model.simulation.air_density;

    // Build BEM mesh from speaker mesh
    let speaker_mesh = super::mesh_gen::generate_loudspeaker_mesh(model);
    let (nodes, mut elements) = speaker_mesh_to_bem(&speaker_mesh, model);

    // Build evaluation points array
    let n_eval = eval_points.len();
    let mut eval_array = Array2::zeros((n_eval, 3));
    for (i, pt) in eval_points.iter().enumerate() {
        eval_array[[i, 0]] = pt[0];
        eval_array[[i, 1]] = pt[1];
        eval_array[[i, 2]] = pt[2];
    }

    let mut all_spl = Vec::with_capacity(frequencies.len());
    let mut all_phase = Vec::with_capacity(frequencies.len());
    let mut prev_solution: Option<Array1<Complex64>> = None;

    let total_start = Instant::now();

    for (fi, &freq) in frequencies.iter().enumerate() {
        let freq_start = Instant::now();

        // 1. Update velocity BCs for this frequency
        update_driver_bcs(&mut elements, &speaker_mesh, model, freq);

        // 2. Assemble BEM system
        let physics = PhysicsParams::new(freq, c, rho, false); // exterior problem
        let beta = physics.burton_miller_beta_scaled(solver_config.beta_scale);
        let system = build_tbem_system_with_beta(&elements, &nodes, &physics, beta);

        // 3. Solve (no incident field — RHS is purely from velocity BCs)
        let gmres_config = GmresConfig {
            max_iterations: solver_config.max_iterations,
            restart: solver_config.restart,
            tolerance: solver_config.tolerance,
            print_interval: 0,
            convergence_check_interval: 1,
            exact_residual_check_interval: 0,
            min_restart: None,
            max_restart: None,
        };

        let initial_guess = if solver_config.recycle_solution {
            prev_solution
                .as_ref()
                .map(|solution| solution.as_slice().unwrap())
        } else {
            None
        };

        let operator = DenseOperator {
            matrix: &system.matrix,
        };
        let result = gmres_with_guess(
            &operator,
            system.rhs.as_slice().unwrap(),
            initial_guess,
            &gmres_config,
        );

        let surface_pressure = Array1::from_vec(result.x);

        // 4. Evaluate field at measurement points
        // For radiation: the BEM solves for surface pressure.
        // The Kirchhoff-Helmholtz integral uses both p (from solution) and v_n (from BCs).
        let surface_velocity = build_surface_velocity(&elements);

        let p_scattered = compute_scattered_field(
            &eval_array,
            &elements,
            &nodes,
            &surface_pressure,
            Some(&surface_velocity),
            &physics,
            BasisOrder::Constant,
        );

        // No incident field → total pressure = scattered pressure.
        // The BEM solves for complex pressure amplitudes (peak values); SPL
        // is RMS-referenced, so use the canonical helper that divides by √2.
        let mut spl_vec = Vec::with_capacity(n_eval);
        let mut phase_vec = Vec::with_capacity(n_eval);
        for i in 0..n_eval {
            let p = p_scattered[i];
            spl_vec.push(sonium_xem::pressure_to_spl(p));
            phase_vec.push(p.arg());
        }

        let elapsed = freq_start.elapsed().as_secs_f64();
        log::info!(
            "[{}/{}] f={:.0} Hz, {} iters, converged={}, {:.2}s",
            fi + 1,
            frequencies.len(),
            freq,
            result.iterations,
            result.converged,
            elapsed,
        );

        all_spl.push(spl_vec);
        all_phase.push(phase_vec);
        prev_solution = Some(surface_pressure);
    }

    let total = total_start.elapsed().as_secs_f64();
    log::info!(
        "Simulation complete: {} frequencies in {:.1}s",
        frequencies.len(),
        total
    );

    SimulationResult {
        frequencies,
        spl: all_spl,
        phase: all_phase,
    }
}

// ---------------------------------------------------------------------------
// Mesh conversion: SpeakerMesh → BEM Element/Node arrays
// ---------------------------------------------------------------------------

/// Convert a `SpeakerMesh` to the BEM solver's `Element` + `nodes` representation.
fn speaker_mesh_to_bem(
    mesh: &SpeakerMesh,
    model: &LoudspeakerModel,
) -> (Array2<f64>, Vec<Element>) {
    let c = model.simulation.speed_of_sound;
    let rho = model.simulation.air_density;
    let y_mdf = admittance_for_material(Z_MDF, rho, c);
    let y_plastic = admittance_for_material(Z_HARD_PLASTIC, rho, c);

    let n_nodes = mesh.nodes.len();
    let mut nodes = Array2::zeros((n_nodes, 3));
    for (i, p) in mesh.nodes.iter().enumerate() {
        nodes[i][0] = p[0];
        nodes[i][1] = p[1];
        nodes[i][2] = p[2];
    }

    let n_elem = mesh.triangles.len();
    let mut elements = Vec::with_capacity(n_elem);

    for (idx, (tri, &region)) in mesh.triangles.iter().zip(mesh.region.iter()).enumerate() {
        let bc = match region {
            MeshRegion::CabinetWall => BoundaryCondition::VelocityWithAdmittance {
                velocity: vec![Complex64::new(0.0, 0.0)],
                admittance: y_mdf,
            },
            MeshRegion::Waveguide => BoundaryCondition::VelocityWithAdmittance {
                velocity: vec![Complex64::new(0.0, 0.0)],
                admittance: y_plastic,
            },
            MeshRegion::WooferAperture => {
                // Initial velocity — will be updated per frequency
                BoundaryCondition::Velocity(vec![model.woofer.velocity_at(1000.0)])
            }
            MeshRegion::TweeterAperture => {
                BoundaryCondition::Velocity(vec![model.tweeter.velocity_at(1000.0)])
            }
            MeshRegion::Separator => BoundaryCondition::Velocity(vec![Complex64::new(0.0, 0.0)]),
        };

        let property = match region {
            MeshRegion::Separator => ElementProperty::MidFace,
            _ => ElementProperty::Surface,
        };

        let mut elem = Element {
            connectivity: vec![tri[0], tri[1], tri[2]],
            element_type: ElementType::Tri3,
            property,
            normal: [0.0; 3],
            node_normals: vec![[0.0; 3]; 3],
            center: [0.0; 3],
            area: 0.0,
            boundary_condition: bc,
            group: 0,
            dof_addresses: vec![idx],
        };

        compute_element_geometry(&mut elem, &nodes);
        elements.push(elem);
    }

    (nodes, elements)
}

/// Recompute element center, area, and normal from node coordinates.
fn compute_element_geometry(elem: &mut Element, nodes: &[[f64; 3]]) {
    let n0 = elem.connectivity[0];
    let n1 = elem.connectivity[1];
    let n2 = elem.connectivity[2];

    // Center
    for j in 0..3 {
        elem.center[j] = (nodes[n0][j] + nodes[n1][j] + nodes[n2][j]) / 3.0;
    }

    // Normal and area via cross product
    let v1 = [
        nodes[n1][0] - nodes[n0][0],
        nodes[n1][1] - nodes[n0][1],
        nodes[n1][2] - nodes[n0][2],
    ];
    let v2 = [
        nodes[n2][0] - nodes[n0][0],
        nodes[n2][1] - nodes[n0][1],
        nodes[n2][2] - nodes[n0][2],
    ];
    let cross = [
        v1[1] * v2[2] - v1[2] * v2[1],
        v1[2] * v2[0] - v1[0] * v2[2],
        v1[0] * v2[1] - v1[1] * v2[0],
    ];
    let len = (cross[0] * cross[0] + cross[1] * cross[1] + cross[2] * cross[2]).sqrt();
    elem.area = len / 2.0;
    if len > 1e-15 {
        elem.normal = Array1::from_vec(vec![cross[0] / len, cross[1] / len, cross[2] / len]);
    }

    // Node normals = element normal (constant elements)
    for local in 0..3 {
        for j in 0..3 {
            elem.node_normals[[local, j]] = elem.normal[j];
        }
    }
}

// ---------------------------------------------------------------------------
// Update driver BCs for a specific frequency
// ---------------------------------------------------------------------------

fn update_driver_bcs(
    elements: &mut [Element],
    speaker_mesh: &SpeakerMesh,
    model: &LoudspeakerModel,
    freq: f64,
) {
    let v_woofer = model.woofer.velocity_at(freq);
    let v_tweeter = model.tweeter.velocity_at(freq);

    for (i, elem) in elements.iter_mut().enumerate() {
        match speaker_mesh.region[i] {
            MeshRegion::WooferAperture => {
                elem.boundary_condition = BoundaryCondition::Velocity(vec![v_woofer]);
            }
            MeshRegion::TweeterAperture => {
                elem.boundary_condition = BoundaryCondition::Velocity(vec![v_tweeter]);
            }
            _ => {}
        }
    }
}

// ---------------------------------------------------------------------------
// Build surface velocity vector from element BCs
// ---------------------------------------------------------------------------

fn build_surface_velocity(elements: &[Element]) -> Array1<Complex64> {
    let n = elements.iter().filter(|e| !e.property.is_evaluation()).count();
    let mut v = Array1::zeros(n);
    let mut idx = 0;
    for elem in elements {
        if elem.property.is_evaluation() {
            continue;
        }
        match &elem.boundary_condition {
            BoundaryCondition::Velocity(vals) => {
                v[idx] = vals.first().copied().unwrap_or(Complex64::new(0.0, 0.0));
            }
            BoundaryCondition::VelocityWithAdmittance { velocity, .. } => {
                v[idx] = velocity.first().copied().unwrap_or(Complex64::new(0.0, 0.0));
            }
            _ => {
                v[idx] = Complex64::new(0.0, 0.0);
            }
        }
        idx += 1;
    }
    v
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::model::*;

    fn tiny_model() -> LoudspeakerModel {
        LoudspeakerModel {
            cabinet: CabinetDimensions {
                height: 0.40,
                width: 0.22,
                depth: 0.28,
            },
            woofer: DriverConfig {
                radius: 0.065,
                y_offset: -0.08,
                piston_velocity: 0.001,
                transfer_function: None,
            },
            tweeter: DriverConfig {
                radius: 0.013,
                y_offset: 0.10,
                piston_velocity: 0.001,
                transfer_function: None,
            },
            separator: SeparatorConfig {
                y_position: 0.02,
                thickness: 0.018,
            },
            waveguide: WaveguideConfig {
                horizontal_spline: vec![[0.0, 0.05], [0.5, 0.03], [1.0, 0.013]],
                vertical_spline: vec![[0.0, 0.04], [0.5, 0.025], [1.0, 0.013]],
                depth: 0.025,
            },
            simulation: SimulationConfig {
                freq_start: 1000.0,
                freq_end: 1000.0,
                freq_points: 1,
                speed_of_sound: 343.0,
                air_density: 1.21,
                max_mesh_frequency: Some(2000.0), // coarse mesh
            },
        }
    }

    #[test]
    fn test_speaker_mesh_to_bem() {
        let model = tiny_model();
        let speaker_mesh = super::super::mesh_gen::generate_loudspeaker_mesh(&model);
        let (nodes, elements) = speaker_mesh_to_bem(&speaker_mesh, &model);

        assert!(nodes.len() > 0);
        assert_eq!(elements.len(), speaker_mesh.triangles.len());

        // All elements should have valid geometry
        for elem in &elements {
            assert!(elem.area > 0.0, "element has zero area");
            let n_len = elem.normal.dot(&elem.normal).sqrt();
            assert!(
                (n_len - 1.0).abs() < 1e-6,
                "normal not unit: len={n_len}"
            );
        }
    }

    #[test]
    fn test_surface_velocity_vector() {
        let model = tiny_model();
        let speaker_mesh = super::super::mesh_gen::generate_loudspeaker_mesh(&model);
        let (_nodes, elements) = speaker_mesh_to_bem(&speaker_mesh, &model);

        let v = build_surface_velocity(&elements);
        assert_eq!(v.len(), elements.len());

        // Most elements should have zero velocity (rigid walls)
        let nonzero_count = v.iter().filter(|&&vi| vi.norm() > 1e-15).count();
        let woofer_count = speaker_mesh
            .region
            .iter()
            .filter(|&&r| r == MeshRegion::WooferAperture)
            .count();
        let tweeter_count = speaker_mesh
            .region
            .iter()
            .filter(|&&r| r == MeshRegion::TweeterAperture)
            .count();

        assert_eq!(nonzero_count, woofer_count + tweeter_count);
    }

    #[test]
    fn test_run_simulation_single_freq() {
        let model = tiny_model();
        let eval_points = vec![[0.0, 0.0, 1.14]]; // 1m in front of baffle

        let result = run_simulation(&model, &eval_points);

        assert_eq!(result.frequencies.len(), 1);
        assert_eq!(result.spl.len(), 1);
        assert_eq!(result.spl[0].len(), 1);

        // SPL should be finite and reasonable (not NaN/Inf)
        let spl = result.spl[0][0];
        assert!(spl.is_finite(), "SPL is not finite: {spl}");
        // A 1 mm/s piston at 1m should produce something in the 40-100 dB range
        assert!(spl > 20.0 && spl < 120.0, "SPL out of range: {spl} dB");
    }
}
