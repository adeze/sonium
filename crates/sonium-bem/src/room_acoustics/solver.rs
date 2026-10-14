//! BEM solver for room acoustics
//!
//! Solves the Helmholtz equation in the room interior with rigid boundary conditions.
//!
//! This module supports three build modes:
//! - `native`: Uses native rayon for parallel processing (fastest)
//! - `wasm`: Uses wasm-bindgen-rayon for Web Worker parallelism
//! - Neither: Falls back to sequential processing

use super::*;
use crate::core::parallel::{BEM_ENTRIES_PER_THREAD, gated_map, gated_map_indexed};
use num_complex::Complex64;
use sonium_solvers::dense::DenseMatrix;
use std::f64::consts::PI;

/// Return type for [`build_fmm_system`].
type FmmSystemResult = Result<(SlfmmSystem, Vec<Element>, Vec<[f64; 3]>), String>;

/// Green's function for 3D Helmholtz equation
/// G(r) = exp(ikr) / (4πr)
fn greens_function_3d(r: f64, k: f64) -> Complex64 {
    if r < 1e-10 {
        return Complex64::new(0.0, 0.0);
    }
    sonium_fastmath::exp_i(k * r) / (4.0 * PI * r)
}

/// Derivative of Green's function in normal direction
/// ∂G/∂n = (ikr - 1) * exp(ikr) / (4πr²) * cos(angle)
fn greens_function_derivative(r: f64, k: f64, cos_angle: f64) -> Complex64 {
    if r < 1e-10 {
        return Complex64::new(0.0, 0.0);
    }
    let ikr = Complex64::new(0.0, k * r);
    let factor = (ikr - 1.0) * sonium_fastmath::exp_i(k * r) / (4.0 * PI * r * r);
    factor * cos_angle
}

fn interior_double_layer_boundary_limit() -> Complex64 {
    Complex64::new(0.5, 0.0)
}

const ROOM_INTERIOR_TBEM_SIGN: f64 = -1.0;

/// Calculate element center and normal vector
fn element_center_and_normal(nodes: &[Point3D]) -> (Point3D, Point3D) {
    // Assume quadrilateral element
    let center = Point3D::new(
        nodes.iter().map(|n| n.x).sum::<f64>() / nodes.len() as f64,
        nodes.iter().map(|n| n.y).sum::<f64>() / nodes.len() as f64,
        nodes.iter().map(|n| n.z).sum::<f64>() / nodes.len() as f64,
    );

    // Normal from cross product of edges (works for both triangles and quads)
    let v1 = Point3D::new(
        nodes[1].x - nodes[0].x,
        nodes[1].y - nodes[0].y,
        nodes[1].z - nodes[0].z,
    );
    let v2 = Point3D::new(
        nodes[2].x - nodes[0].x,
        nodes[2].y - nodes[0].y,
        nodes[2].z - nodes[0].z,
    );

    // Cross product
    let nx = v1.y * v2.z - v1.z * v2.y;
    let ny = v1.z * v2.x - v1.x * v2.z;
    let nz = v1.x * v2.y - v1.y * v2.x;

    let norm = (nx * nx + ny * ny + nz * nz).sqrt();
    let normal = Point3D::new(nx / norm, ny / norm, nz / norm);

    (center, normal)
}

/// Calculate element area (supports triangles and quads)
fn element_area(nodes: &[Point3D]) -> f64 {
    if nodes.len() == 3 {
        // Triangle: 0.5 * |v1 × v2|
        let v1 = Point3D::new(
            nodes[1].x - nodes[0].x,
            nodes[1].y - nodes[0].y,
            nodes[1].z - nodes[0].z,
        );
        let v2 = Point3D::new(
            nodes[2].x - nodes[0].x,
            nodes[2].y - nodes[0].y,
            nodes[2].z - nodes[0].z,
        );

        let cross_x = v1.y * v2.z - v1.z * v2.y;
        let cross_y = v1.z * v2.x - v1.x * v2.z;
        let cross_z = v1.x * v2.y - v1.y * v2.x;

        0.5 * (cross_x * cross_x + cross_y * cross_y + cross_z * cross_z).sqrt()
    } else if nodes.len() == 4 {
        // Quadrilateral: split into two triangles and sum areas
        let v1 = Point3D::new(
            nodes[1].x - nodes[0].x,
            nodes[1].y - nodes[0].y,
            nodes[1].z - nodes[0].z,
        );
        let v2 = Point3D::new(
            nodes[2].x - nodes[0].x,
            nodes[2].y - nodes[0].y,
            nodes[2].z - nodes[0].z,
        );

        let cross1_x = v1.y * v2.z - v1.z * v2.y;
        let cross1_y = v1.z * v2.x - v1.x * v2.z;
        let cross1_z = v1.x * v2.y - v1.y * v2.x;
        let area1 = 0.5 * (cross1_x * cross1_x + cross1_y * cross1_y + cross1_z * cross1_z).sqrt();

        let v3 = Point3D::new(
            nodes[3].x - nodes[0].x,
            nodes[3].y - nodes[0].y,
            nodes[3].z - nodes[0].z,
        );

        let cross2_x = v2.y * v3.z - v2.z * v3.y;
        let cross2_y = v2.z * v3.x - v2.x * v3.z;
        let cross2_z = v2.x * v3.y - v2.y * v3.x;
        let area2 = 0.5 * (cross2_x * cross2_x + cross2_y * cross2_y + cross2_z * cross2_z).sqrt();

        area1 + area2
    } else {
        0.0
    }
}

/// Build BEM system matrix for rigid boundaries
pub fn build_bem_matrix(mesh: &RoomMesh, k: f64) -> sonium_solvers::dense::DenseMatrix<Complex64> {
    let n = mesh.elements.len();
    let mut matrix = sonium_solvers::dense::DenseMatrix::<Complex64>::zeros(n, n);

    // Get element centers and normals
    let mut centers = Vec::new();
    let mut normals = Vec::new();
    let mut areas = Vec::new();

    for element in &mesh.elements {
        let nodes: Vec<Point3D> = element.nodes.iter().map(|&i| mesh.nodes[i]).collect();
        let (center, normal) = element_center_and_normal(&nodes);
        let area = element_area(&nodes);

        centers.push(center);
        normals.push(normal);
        areas.push(area);
    }

    // Fill matrix: rigid boundary condition ∂p/∂n = 0
    // This gives: Σ_j (∂G/∂n_i)(r_ij) * p_j * A_j = incident field derivative
    for i in 0..n {
        for j in 0..n {
            let r = centers[i].distance_to(&centers[j]);

            if i == j {
                matrix[[i, j]] = interior_double_layer_boundary_limit();
            } else {
                // Direction from j to i
                let dx = centers[i].x - centers[j].x;
                let dy = centers[i].y - centers[j].y;
                let dz = centers[i].z - centers[j].z;

                // Direct interior double-layer operator: the normal belongs
                // to the integration element `j`, not the observation point
                // `i`.  Using `normal_i` assembles the adjoint operator K'
                // and is inconsistent with a surface-pressure unknown.
                let cos_angle = (dx * normals[j].x + dy * normals[j].y + dz * normals[j].z) / r;

                matrix[[i, j]] = -greens_function_derivative(r, k, cos_angle) * areas[j];
            }
        }
    }

    matrix
}

/// Calculate incident field from sources at element centers
pub fn calculate_incident_field(
    mesh: &RoomMesh,
    sources: &[Source],
    k: f64,
    frequency: f64,
) -> Vec<Complex64> {
    let n = mesh.elements.len();
    let mut incident = vec![Complex64::new(0.0, 0.0); n];

    for (i, element) in mesh.elements.iter().enumerate() {
        let nodes: Vec<Point3D> = element.nodes.iter().map(|&idx| mesh.nodes[idx]).collect();
        let (center, _normal) = element_center_and_normal(&nodes);

        let mut total_pressure = Complex64::new(0.0, 0.0);

        for source in sources {
            let r = center.distance_to(&source.position);
            let amplitude = source.amplitude_towards(&center, frequency);

            // Incident pressure from monopole source
            total_pressure += greens_function_3d(r, k) * amplitude;
        }

        incident[i] = total_pressure;
    }

    incident
}

/// Calculate pressure at field points using double-layer potential representation
///
/// Uses: p(x) = p_inc(x) + ∫∫ (∂G/∂n)(x, y) * p_surface(y) dS(y)
pub fn calculate_field_pressure(
    mesh: &RoomMesh,
    surface_pressure: &[Complex64],
    sources: &[Source],
    field_points: &[Point3D],
    k: f64,
    frequency: f64,
) -> Vec<Complex64> {
    let n_points = field_points.len();
    let mut pressures = vec![Complex64::new(0.0, 0.0); n_points];

    // Get element data
    let mut centers = Vec::new();
    let mut normals = Vec::new();
    let mut areas = Vec::new();

    for element in &mesh.elements {
        let nodes: Vec<Point3D> = element.nodes.iter().map(|&i| mesh.nodes[i]).collect();
        let (center, normal) = element_center_and_normal(&nodes);
        let area = element_area(&nodes);

        centers.push(center);
        normals.push(normal);
        areas.push(area);
    }

    for (ip, point) in field_points.iter().enumerate() {
        // Incident field from sources
        let mut p_incident = Complex64::new(0.0, 0.0);
        for source in sources {
            let r = point.distance_to(&source.position);
            if r < 1e-10 {
                continue;
            }
            let amplitude = source.amplitude_towards(point, frequency);
            p_incident += greens_function_3d(r, k) * amplitude;
        }

        // Scattered field from boundary integral using double-layer potential
        // p_scattered = ∫∫ (∂G/∂n)(x, y) * p_surface(y) dS(y)
        let mut p_scattered = Complex64::new(0.0, 0.0);

        for j in 0..surface_pressure.len() {
            let r = point.distance_to(&centers[j]);
            if r < 1e-10 {
                continue;
            }

            // Direction from element center to field point
            let dx = point.x - centers[j].x;
            let dy = point.y - centers[j].y;
            let dz = point.z - centers[j].z;

            // Cosine of angle between (point - center) direction and outward normal
            let cos_angle = (dx * normals[j].x + dy * normals[j].y + dz * normals[j].z) / r;

            // Normal derivative of Green's function
            let dg_dn = greens_function_derivative(r, k, cos_angle);

            p_scattered += dg_dn * surface_pressure[j] * areas[j];
        }

        // `greens_function_derivative` uses `(x-y)·n_y`, so the accumulated
        // value is `-K p`.  Green's third identity is therefore represented
        // as `p = p_incident + p_scattered` for a rigid boundary.
        pressures[ip] = p_incident + p_scattered;
    }

    pressures
}

// pressure_to_spl is now in sonium_xem::types

/// Simple GMRES solver for complex linear systems
/// Solves Ax = b using restarted GMRES
pub fn gmres_solve(
    a: &DenseMatrix<Complex64>,
    b: &[Complex64],
    max_iter: usize,
    restart: usize,
    tol: f64,
) -> Result<Vec<Complex64>, String> {
    let n = b.len();
    if a.nrows() != n || a.ncols() != n {
        return Err("Matrix dimensions mismatch".to_string());
    }

    let mut x = vec![Complex64::new(0.0, 0.0); n];

    for _cycle in 0..max_iter {
        // Compute initial residual
        let ax = a.matvec(&x);
        let r: Vec<Complex64> = b.iter().zip(ax.iter()).map(|(bi, axi)| bi - axi).collect();
        let beta = r.iter().map(|ri| ri.norm_sqr()).sum::<f64>().sqrt();

        if beta < tol {
            return Ok(x);
        }

        // Arnoldi iteration
        let m = restart.min(n);
        let mut v: Vec<Vec<Complex64>> = vec![vec![Complex64::new(0.0, 0.0); n]; m + 1];
        let mut h = DenseMatrix::<Complex64>::zeros(m + 1, m);

        v[0] = r.iter().map(|ri| ri / Complex64::new(beta, 0.0)).collect();

        for j in 0..m {
            // w = A * v[j]
            let w = a.matvec(&v[j]);
            let mut w_orth = w.clone();

            // Modified Gram-Schmidt orthogonalization
            for i in 0..=j {
                h[[i, j]] = v[i]
                    .iter()
                    .zip(w_orth.iter())
                    .map(|(vi, wi)| vi.conj() * wi)
                    .sum();

                for k in 0..n {
                    w_orth[k] -= h[[i, j]] * v[i][k];
                }
            }

            let h_norm = w_orth.iter().map(|wi| wi.norm_sqr()).sum::<f64>().sqrt();
            h[[j + 1, j]] = Complex64::new(h_norm, 0.0);

            if h_norm > 1e-12 {
                v[j + 1] = w_orth
                    .iter()
                    .map(|wi| wi / Complex64::new(h_norm, 0.0))
                    .collect();
            } else {
                break;
            }
        }

        // Solve least squares problem: minimize ||β*e1 - H*y||
        let mut e1 = vec![Complex64::new(0.0, 0.0); m + 1];
        e1[0] = Complex64::new(beta, 0.0);

        // Use QR decomposition to solve
        let y = solve_least_squares(&h, &e1, m)?;

        // Update solution: x = x + V*y
        for j in 0..m {
            for k in 0..n {
                x[k] += y[j] * v[j][k];
            }
        }

        // Check convergence
        let ax = a.matvec(&x);
        let residual: f64 = b
            .iter()
            .zip(ax.iter())
            .map(|(bi, axi)| (bi - axi).norm_sqr())
            .sum::<f64>()
            .sqrt();

        if residual < tol {
            return Ok(x);
        }
    }

    Ok(x)
}

/// Solve least squares problem using back substitution on upper triangular part
fn solve_least_squares(
    h: &DenseMatrix<Complex64>,
    e1: &[Complex64],
    m: usize,
) -> Result<Vec<Complex64>, String> {
    let mut y = vec![Complex64::new(0.0, 0.0); m];
    let mut rhs: Vec<Complex64> = (0..m).map(|i| e1[i]).collect();

    // Apply Givens rotations to make H upper triangular
    let mut h_tri = DenseMatrix::<Complex64>::zeros(m, m);
    for row in 0..m {
        for col in 0..m {
            h_tri[[row, col]] = h[[row, col]];
        }
    }

    for i in 0..m {
        for j in (i + 1)..m {
            if h_tri[[j, i]].norm() > 1e-12 {
                let a = h_tri[[i, i]];
                let b = h_tri[[j, i]];
                let r = (a.norm_sqr() + b.norm_sqr()).sqrt();
                let c = a.norm() / r;
                let s = b / Complex64::new(r, 0.0);

                // Apply rotation to rows i and j
                for k in i..m {
                    let temp = c * h_tri[[i, k]] + s * h_tri[[j, k]];
                    h_tri[[j, k]] = -s.conj() * h_tri[[i, k]] + c * h_tri[[j, k]];
                    h_tri[[i, k]] = temp;
                }

                let temp = c * rhs[i] + s * rhs[j];
                rhs[j] = -s.conj() * rhs[i] + c * rhs[j];
                rhs[i] = temp;
            }
        }
    }

    // Back substitution
    for i in (0..m).rev() {
        let mut sum = rhs[i];
        for j in (i + 1)..m {
            sum -= h_tri[[i, j]] * y[j];
        }
        if h_tri[[i, i]].norm() > 1e-12 {
            y[i] = sum / h_tri[[i, i]];
        }
    }

    Ok(y)
}

/// Source-directivity convention used by the room TBEM path.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RoomSourceDirectivityMode {
    /// Apply source directivity toward both boundary elements and receivers.
    BoundaryAndReceiver,
    /// Excite the room as an isotropic monopole, then apply receiver-side
    /// directivity to the total field. This mirrors the `sonium-sim-room` FEM
    /// convention used for low-frequency RoomEQ data.
    ///
    /// Convention contrast (SIM-PERF-20261013.md section 1): the excitation
    /// carries amplitude × crossover while the pure pattern multiplies the
    /// total (direct + reverberant) field at each receiver, so reflected
    /// energy inherits the receiver-side pattern. The speaker path instead
    /// applies per-driver crossover transfers at the SOURCE (drive voltages
    /// → velocity BCs) with source-side radiation, keeping crossover
    /// orthogonal to propagation. Unifying the two conventions (e.g.
    /// source-side patterns in the room path) is future work; either change
    /// moves every absorbing-room response and needs revalidation against
    /// the analytical references before it becomes the default.
    ReceiverOnly,
}

/// Configuration for the core TBEM room-acoustics dense assembly path.
#[derive(Debug, Clone)]
pub struct RoomTbemConfig {
    /// Burton-Miller coupling scale. Set to `0.0` to use the CBIE-only room formulation.
    pub beta_scale: f64,
    /// Optional normalized surface admittance (`rho*c / Zs`) per mesh element.
    pub element_admittance: Option<Vec<Complex64>>,
    /// Source-directivity convention used for incident fields.
    pub source_directivity: RoomSourceDirectivityMode,
}

#[derive(Clone, Copy)]
struct RoomFieldElement {
    center: Point3D,
    normal: Point3D,
    area: f64,
}

/// Frequency-specific room TBEM geometry reused across source RHS vectors and
/// receiver field evaluations.
pub struct RoomTbemPreparedGeometry {
    elements: Vec<Element>,
    nodes: Vec<[f64; 3]>,
    field_elements: Vec<RoomFieldElement>,
    physics: PhysicsParams,
    beta: Complex64,
    frequency: f64,
    source_directivity: RoomSourceDirectivityMode,
}

/// Precomputed boundary-to-listener double-layer kernels for a fixed room TBEM
/// geometry and frequency.
pub struct RoomTbemFieldEvaluator {
    field_points: Vec<Point3D>,
    kernels: Vec<Complex64>,
    n_elements: usize,
    wave_number: f64,
    frequency: f64,
    source_directivity: RoomSourceDirectivityMode,
}

impl Default for RoomTbemConfig {
    fn default() -> Self {
        Self {
            beta_scale: 1.0,
            element_admittance: None,
            source_directivity: RoomSourceDirectivityMode::BoundaryAndReceiver,
        }
    }
}

/// Precompute the source-independent scattered-field kernels at receiver points.
///
/// For a fixed mesh and frequency, the double-layer potential contribution is
/// a matrix-vector product between this listener kernel and the source-specific
/// surface pressure. Reusing it avoids recomputing distances and complex
/// exponentials for every source.
pub fn prepare_room_tbem_field_evaluator(
    prepared: &RoomTbemPreparedGeometry,
    field_points: &[Point3D],
) -> RoomTbemFieldEvaluator {
    let n_elements = prepared.field_elements.len();
    let kernels: Vec<Complex64> = field_points
        .iter()
        .flat_map(|point| {
            prepared.field_elements.iter().map(move |element| {
                let r = point.distance_to(&element.center);
                if r < 1e-10 {
                    return Complex64::new(0.0, 0.0);
                }

                let dx = point.x - element.center.x;
                let dy = point.y - element.center.y;
                let dz = point.z - element.center.z;
                let cos_angle =
                    (dx * element.normal.x + dy * element.normal.y + dz * element.normal.z) / r;

                greens_function_derivative(r, prepared.physics.wave_number, cos_angle)
                    * element.area
            })
        })
        .collect();

    RoomTbemFieldEvaluator {
        field_points: field_points.to_vec(),
        kernels,
        n_elements,
        wave_number: prepared.physics.wave_number,
        frequency: prepared.frequency,
        source_directivity: prepared.source_directivity,
    }
}

/// Precompute the geometry and frequency-specific physics used by room TBEM.
///
/// The dense TBEM matrix is source-independent for a fixed mesh, frequency, and
/// boundary configuration. This prepared value lets callers avoid converting
/// the room mesh and rebuilding field-evaluation element data once per source.
pub fn prepare_room_tbem_geometry_with_config(
    mesh: &RoomMesh,
    k: f64,
    frequency: f64,
    config: &RoomTbemConfig,
) -> RoomTbemPreparedGeometry {
    let (elements, nodes) =
        room_mesh_to_core_elements_with_admittance(mesh, k, config.element_admittance.as_deref());
    let physics = room_physics_from_wavenumber(k, frequency);
    let beta = room_burton_miller_beta(&physics, config.beta_scale);
    let field_elements = elements
        .iter()
        .map(|element| RoomFieldElement {
            center: Point3D::new(element.center[0], element.center[1], element.center[2]),
            normal: Point3D::new(element.normal[0], element.normal[1], element.normal[2]),
            area: element.area,
        })
        .collect();

    RoomTbemPreparedGeometry {
        elements,
        nodes,
        field_elements,
        physics,
        beta,
        frequency,
        source_directivity: config.source_directivity,
    }
}

fn room_physics_from_wavenumber(k: f64, frequency: f64) -> PhysicsParams {
    let speed_of_sound = if k.abs() > 1e-12 && frequency > 0.0 {
        2.0 * PI * frequency / k
    } else {
        343.0
    };
    PhysicsParams::new(frequency, speed_of_sound, 1.21, true)
}

fn room_burton_miller_beta(physics: &PhysicsParams, beta_scale: f64) -> Complex64 {
    if beta_scale == 0.0 || physics.wave_number.abs() < 1e-12 {
        Complex64::new(0.0, 0.0)
    } else {
        Complex64::new(
            0.0,
            physics.harmonic_factor * beta_scale / physics.wave_number,
        )
    }
}

fn source_amplitude_for_boundary(
    source: &Source,
    point: &Point3D,
    frequency: f64,
    mode: RoomSourceDirectivityMode,
) -> f64 {
    match mode {
        RoomSourceDirectivityMode::BoundaryAndReceiver => {
            source.amplitude_towards(point, frequency)
        }
        RoomSourceDirectivityMode::ReceiverOnly => source.isotropic_amplitude(frequency),
    }
}

fn room_incident_rhs_tbem(
    elements: &[Element],
    sources: &[Source],
    physics: &PhysicsParams,
    beta: Complex64,
    frequency: f64,
    source_directivity: RoomSourceDirectivityMode,
) -> Vec<Complex64> {
    let gamma = Complex64::new(physics.gamma(), 0.0);
    let tau = Complex64::new(physics.tau, 0.0);
    let mut rhs: Vec<Complex64> = vec![Complex64::new(0.0, 0.0); elements.len()];

    for (i, element) in elements.iter().enumerate() {
        let center = Point3D::new(element.center[0], element.center[1], element.center[2]);
        let normal = Point3D::new(element.normal[0], element.normal[1], element.normal[2]);

        let mut p_inc = Complex64::new(0.0, 0.0);
        let mut dpdn_inc = Complex64::new(0.0, 0.0);

        for source in sources {
            let r = center.distance_to(&source.position);
            if r < 1e-10 {
                continue;
            }

            let amplitude =
                source_amplitude_for_boundary(source, &center, frequency, source_directivity);
            p_inc += greens_function_3d(r, physics.wave_number) * amplitude;

            let dx = center.x - source.position.x;
            let dy = center.y - source.position.y;
            let dz = center.z - source.position.z;
            let cos_angle = (dx * normal.x + dy * normal.y + dz * normal.z) / r;
            dpdn_inc += greens_function_derivative(r, physics.wave_number, cos_angle) * amplitude;
        }

        rhs[i] = ROOM_INTERIOR_TBEM_SIGN * -(gamma * p_inc + beta * tau * dpdn_inc);
    }

    rhs
}

/// Build the dense room-acoustics system with the core TBEM assembler.
///
/// Unlike the legacy `build_bem_matrix*` helpers, this path uses the core singular
/// and regular BEM integration routines and exposes a Burton-Miller coupling scale.
pub fn build_room_tbem_system_with_config(
    mesh: &RoomMesh,
    sources: &[Source],
    k: f64,
    frequency: f64,
    config: &RoomTbemConfig,
) -> crate::core::assembly::tbem::TbemSystem {
    let prepared = prepare_room_tbem_geometry_with_config(mesh, k, frequency, config);
    build_room_tbem_system_with_prepared_geometry(&prepared, sources)
}

/// Build the dense room-acoustics system from prepared TBEM geometry.
pub fn build_room_tbem_system_with_prepared_geometry(
    prepared: &RoomTbemPreparedGeometry,
    sources: &[Source],
) -> crate::core::assembly::tbem::TbemSystem {
    // Room meshes have flat triangular elements with `node_normals = [0;3]`
    // and zero BC values, so the SIMD batched assembler routes ~100 % of
    // far-field pairs through `sonium_compute::bem_cpu::regular_pair_batch`
    // (NEON 2-pair on aarch64). Self / near-field pairs and any pair that
    // doesn't meet the batch preconditions fall back to scalar inside the
    // function, so the matrix is identical to the canonical scalar reference
    // up to the 13-pt → 4-pt quadrature delta at far-field (~1e-9 per entry,
    // well below the GMRES 1e-6 target).
    let mut system = crate::core::assembly::tbem::build_tbem_system_stabilized_with_beta_simd(
        &prepared.elements,
        &prepared.nodes,
        &prepared.physics,
        ROOM_INTERIOR_TBEM_SIGN,
        prepared.beta,
    );
    system.rhs = build_room_tbem_rhs_with_prepared_geometry(prepared, sources);
    system
}

/// Build only the dense room-acoustics TBEM right-hand side.
///
/// The TBEM matrix is source-independent for a fixed mesh, frequency, and
/// boundary configuration. Use this helper when a caller has already assembled
/// and factorized the matrix and only needs additional source RHS vectors.
pub fn build_room_tbem_rhs_with_config(
    mesh: &RoomMesh,
    sources: &[Source],
    k: f64,
    frequency: f64,
    config: &RoomTbemConfig,
) -> Vec<Complex64> {
    let prepared = prepare_room_tbem_geometry_with_config(mesh, k, frequency, config);
    build_room_tbem_rhs_with_prepared_geometry(&prepared, sources)
}

/// Build only the dense room-acoustics TBEM RHS from prepared geometry.
pub fn build_room_tbem_rhs_with_prepared_geometry(
    prepared: &RoomTbemPreparedGeometry,
    sources: &[Source],
) -> Vec<Complex64> {
    room_incident_rhs_tbem(
        &prepared.elements,
        sources,
        &prepared.physics,
        prepared.beta,
        prepared.frequency,
        prepared.source_directivity,
    )
}

/// Build the dense room-acoustics system using the Metal-accelerated TBEM
/// assembler for the far-field regular-pair tail. Self / near-field pairs
/// stay on the CPU through the same curved-normal-capable singular and
/// regular routines as the CPU stabilised path, so the result is
/// numerically equivalent to `build_room_tbem_system_with_config` up to
/// the f32→f64 promotion on the GPU output.
#[cfg(feature = "metal")]
pub fn build_room_tbem_system_with_config_metal(
    mesh: &RoomMesh,
    sources: &[Source],
    k: f64,
    frequency: f64,
    config: &RoomTbemConfig,
    backend: &sonium_compute::MetalBackend,
) -> crate::core::assembly::tbem::TbemSystem {
    let prepared = prepare_room_tbem_geometry_with_config(mesh, k, frequency, config);
    build_room_tbem_system_with_prepared_geometry_metal(&prepared, sources, backend)
}

/// Build the dense room-acoustics system with Metal from prepared TBEM geometry.
#[cfg(feature = "metal")]
pub fn build_room_tbem_system_with_prepared_geometry_metal(
    prepared: &RoomTbemPreparedGeometry,
    sources: &[Source],
    backend: &sonium_compute::MetalBackend,
) -> crate::core::assembly::tbem::TbemSystem {
    let mut system = crate::core::assembly::tbem::build_tbem_system_stabilized_with_beta_metal(
        &prepared.elements,
        &prepared.nodes,
        &prepared.physics,
        ROOM_INTERIOR_TBEM_SIGN,
        prepared.beta,
        backend,
    );
    system.rhs = build_room_tbem_rhs_with_prepared_geometry(prepared, sources);
    system
}

/// Build the dense room-acoustics system using the CUDA-accelerated TBEM
/// assembler for the far-field regular-pair tail. Self / near-field pairs
/// stay on the CPU through the same singular and regular routines as the CPU
/// stabilised path.
#[cfg(feature = "cuda")]
pub fn build_room_tbem_system_with_config_cuda(
    mesh: &RoomMesh,
    sources: &[Source],
    k: f64,
    frequency: f64,
    config: &RoomTbemConfig,
    backend: &sonium_compute::CudaBackend,
) -> crate::core::assembly::tbem::TbemSystem {
    try_build_room_tbem_system_with_config_cuda(mesh, sources, k, frequency, config, backend)
        .unwrap_or_else(|error| panic!("checked CUDA room TBEM assembly failed: {error}"))
}

/// Checked CUDA room-acoustics TBEM assembly.
///
/// The returned error preserves unsupported element layouts, invalid geometry,
/// and CUDA allocation/launch/transfer failures. Use this entry point when a
/// room workload must fail closed instead of taking the compatibility wrapper's
/// panic path.
#[cfg(feature = "cuda")]
pub fn try_build_room_tbem_system_with_config_cuda(
    mesh: &RoomMesh,
    sources: &[Source],
    k: f64,
    frequency: f64,
    config: &RoomTbemConfig,
    backend: &sonium_compute::CudaBackend,
) -> Result<
    crate::core::assembly::tbem::TbemSystem,
    crate::core::assembly::tbem::CudaTbemAssemblyError,
> {
    let prepared = prepare_room_tbem_geometry_with_config(mesh, k, frequency, config);
    try_build_room_tbem_system_with_prepared_geometry_cuda(&prepared, sources, backend)
}

/// Build the dense room-acoustics system with CUDA from prepared TBEM geometry.
#[cfg(feature = "cuda")]
pub fn build_room_tbem_system_with_prepared_geometry_cuda(
    prepared: &RoomTbemPreparedGeometry,
    sources: &[Source],
    backend: &sonium_compute::CudaBackend,
) -> crate::core::assembly::tbem::TbemSystem {
    try_build_room_tbem_system_with_prepared_geometry_cuda(prepared, sources, backend)
        .unwrap_or_else(|error| panic!("checked CUDA room TBEM assembly failed: {error}"))
}

/// Checked CUDA room-acoustics TBEM assembly from prepared geometry.
#[cfg(feature = "cuda")]
pub fn try_build_room_tbem_system_with_prepared_geometry_cuda(
    prepared: &RoomTbemPreparedGeometry,
    sources: &[Source],
    backend: &sonium_compute::CudaBackend,
) -> Result<
    crate::core::assembly::tbem::TbemSystem,
    crate::core::assembly::tbem::CudaTbemAssemblyError,
> {
    let mut system = crate::core::assembly::tbem::build_tbem_system_with_beta_cuda_checked(
        &prepared.elements,
        &prepared.nodes,
        &prepared.physics,
        prepared.beta,
        backend,
    )?;
    if (ROOM_INTERIOR_TBEM_SIGN - 1.0).abs() > f64::EPSILON {
        for row in 0..system.matrix.nrows() {
            for col in 0..system.matrix.ncols() {
                system.matrix[[row, col]] *= ROOM_INTERIOR_TBEM_SIGN;
            }
        }
    }
    system.rhs = build_room_tbem_rhs_with_prepared_geometry(prepared, sources);
    Ok(system)
}

/// Build the dense room-acoustics system with default core TBEM settings.
pub fn build_room_tbem_system(
    mesh: &RoomMesh,
    sources: &[Source],
    k: f64,
    frequency: f64,
) -> crate::core::assembly::tbem::TbemSystem {
    build_room_tbem_system_with_config(mesh, sources, k, frequency, &RoomTbemConfig::default())
}

/// Solve the legacy fast room collocation matrix with dense LU factorization.
///
/// This is the deterministic direct-solver path for small dense room systems.
/// Use [`solve_bem_system`] when restarted GMRES behavior is explicitly wanted.
pub fn solve_bem_system_direct(
    mesh: &RoomMesh,
    sources: &[Source],
    k: f64,
    frequency: f64,
) -> Result<Vec<Complex64>, String> {
    let matrix = build_bem_matrix_parallel(mesh, k);
    let rhs = calculate_incident_field(mesh, sources, k, frequency);

    sonium_solvers::lu_solve(&matrix, &rhs)
        .map_err(|error| format!("Room TBEM LU solve failed: {error}"))
}

/// Solve BEM system using the legacy fast room collocation matrix.
///
/// This path is retained for existing room-data pipelines that need deterministic
/// CI runtimes. For accuracy-critical dense solves, use
/// [`solve_bem_system_core_tbem`], which assembles through the core TBEM singular
/// and regular integration machinery.
pub fn solve_bem_system(
    mesh: &RoomMesh,
    sources: &[Source],
    k: f64,
    frequency: f64,
) -> Result<Vec<Complex64>, String> {
    use crate::core::solver::{GmresConfig, solve_gmres};

    let matrix = build_bem_matrix_parallel(mesh, k);
    let rhs = calculate_incident_field(mesh, sources, k, frequency);

    // Use the core GMRES solver
    let config = GmresConfig {
        max_iterations: 100,
        restart: 50,
        tolerance: 1e-6,
        print_interval: 0,
        convergence_check_interval: 1,
        exact_residual_check_interval: 0,
        min_restart: None,
        max_restart: None,
    };

    // Create matvec closure for dense matrix (implement LinearOperator or use wrapper)
    // solve_gmres expects a LinearOperator.
    // We can use DenseOperator from core::solver
    use crate::core::solver::DenseOperator;
    let op = DenseOperator::new(matrix);

    let solution = solve_gmres(&op, &rhs, &config);

    if solution.converged {
        Ok(solution.x)
    } else {
        Err(format!(
            "Room TBEM GMRES failed to converge after {} iterations, residual {:.3e}",
            solution.iterations, solution.residual
        ))
    }
}

/// Solve BEM system using core TBEM dense assembly and GMRES.
pub fn solve_bem_system_core_tbem(
    mesh: &RoomMesh,
    sources: &[Source],
    k: f64,
    frequency: f64,
) -> Result<Vec<Complex64>, String> {
    solve_bem_system_core_tbem_with_config(mesh, sources, k, frequency, &RoomTbemConfig::default())
}

/// Solve BEM system using core TBEM dense assembly and explicit room settings.
pub fn solve_bem_system_core_tbem_with_config(
    mesh: &RoomMesh,
    sources: &[Source],
    k: f64,
    frequency: f64,
    config: &RoomTbemConfig,
) -> Result<Vec<Complex64>, String> {
    use crate::core::solver::{GmresConfig, solve_gmres};

    let system = build_room_tbem_system_with_config(mesh, sources, k, frequency, config);

    let config = GmresConfig {
        max_iterations: 100,
        restart: 50,
        tolerance: 1e-6,
        print_interval: 0,
        convergence_check_interval: 1,
        exact_residual_check_interval: 0,
        min_restart: None,
        max_restart: None,
    };

    use crate::core::solver::DenseOperator;
    let op = DenseOperator::new(system.matrix);
    let solution = solve_gmres(&op, &system.rhs, &config);

    if solution.converged {
        Ok(solution.x)
    } else {
        Err(format!(
            "Room core TBEM GMRES failed to converge after {} iterations, residual {:.3e}",
            solution.iterations, solution.residual
        ))
    }
}

/// Build BEM matrix with parallel assembly
///
/// Uses portable parallel iteration that works with native rayon, WASM, or sequential fallback.
pub fn build_bem_matrix_parallel(
    mesh: &RoomMesh,
    k: f64,
) -> sonium_solvers::dense::DenseMatrix<Complex64> {
    let n = mesh.elements.len();

    // Precompute element data (rayon above the machine-adaptive gate,
    // serial for small meshes).
    let element_data: Vec<_> = gated_map(&mesh.elements, n, BEM_ENTRIES_PER_THREAD, |element| {
        let nodes: Vec<Point3D> = element.nodes.iter().map(|&i| mesh.nodes[i]).collect();
        let (center, normal) = element_center_and_normal(&nodes);
        let area = element_area(&nodes);
        (center, normal, area)
    });

    // Build matrix rows (rayon above the machine-adaptive gate, serial for
    // small systems). Work is source × field kernel evaluations.
    let rows: Vec<_> = gated_map_indexed(n, n.saturating_mul(n), BEM_ENTRIES_PER_THREAD, |i| {
        let mut row = vec![Complex64::new(0.0, 0.0); n];
        let (center_i, _normal_i, _area_i) = &element_data[i];

        for j in 0..n {
            let (center_j, normal_j, area_j) = &element_data[j];
            let r = center_i.distance_to(center_j);

            if i == j {
                row[j] = interior_double_layer_boundary_limit();
            } else {
                // Off-diagonal
                let dx = center_i.x - center_j.x;
                let dy = center_i.y - center_j.y;
                let dz = center_i.z - center_j.z;

                let cos_angle = (dx * normal_j.x + dy * normal_j.y + dz * normal_j.z) / r;
                row[j] = -greens_function_derivative(r, k, cos_angle) * area_j;
            }
        }
        row
    });

    // Convert row buffers to the BEM array surface.
    let mut matrix = sonium_solvers::dense::DenseMatrix::<Complex64>::zeros(n, n);
    for (i, row) in rows.iter().enumerate() {
        for (j, &val) in row.iter().enumerate() {
            matrix[[i, j]] = val;
        }
    }

    matrix
}

/// Build BEM matrix with adaptive integration for near-singular elements
///
/// Uses adaptive subdivision for self-interaction and nearby elements to improve accuracy.
/// This is especially important at higher frequencies where standard point collocation
/// becomes inaccurate.
pub fn build_bem_matrix_adaptive(
    mesh: &RoomMesh,
    k: f64,
    use_adaptive: bool,
) -> sonium_solvers::dense::DenseMatrix<Complex64> {
    use crate::core::integration::singular::QuadratureParams;
    use crate::core::types::{ElementType, PhysicsParams};

    let n = mesh.elements.len();

    // Physics parameters for singular integration
    let speed_of_sound = 343.0;
    let frequency = k * speed_of_sound / (2.0 * PI);
    let omega = 2.0 * PI * frequency;

    let physics = PhysicsParams {
        wave_number: k,
        density: 1.0,
        speed_of_sound,
        frequency,
        omega,
        wave_length: speed_of_sound / frequency,
        harmonic_factor: 1.0, // exp(+ikr) convention
        pressure_factor: 1.0 * omega * 1.0,
        tau: -1.0, // internal problem (room interior)
    };

    // Precompute element data (rayon above the machine-adaptive gate,
    // serial for small meshes).
    let element_data: Vec<_> = gated_map(&mesh.elements, n, BEM_ENTRIES_PER_THREAD, |element| {
        let nodes: Vec<Point3D> = element.nodes.iter().map(|&i| mesh.nodes[i]).collect();
        let (center, normal) = element_center_and_normal(&nodes);
        let area = element_area(&nodes);
        let char_length = element_characteristic_length(&nodes);
        (center, normal, area, char_length, nodes)
    });

    // Build matrix rows (rayon above the machine-adaptive gate, serial for
    // small systems). Work is source × field kernel evaluations.
    let rows: Vec<_> = gated_map_indexed(n, n.saturating_mul(n), BEM_ENTRIES_PER_THREAD, |i| {
        let mut row = vec![Complex64::new(0.0, 0.0); n];
        let (center_i, normal_i, _area_i, char_length_i, _nodes_i) = &element_data[i];

        for j in 0..n {
            let (center_j, normal_j, area_j, char_length_j, nodes_j) = &element_data[j];
            let r = center_i.distance_to(center_j);

            // Criterion for near-singular: distance < 2 * characteristic length
            let near_threshold = 2.0 * (char_length_i + char_length_j);
            let is_near = r < near_threshold || i == j;

            if use_adaptive && is_near {
                // Use adaptive singular integration for near-field
                let element_coords = nodes_to_array2(nodes_j);
                let source_point = point_to_array1(center_i);
                let source_normal = normal_to_array1(normal_i);

                // Use frequency-adaptive quadrature
                let ka = k * char_length_j;
                let quad_params = QuadratureParams::for_ka(ka);

                let result = crate::core::integration::singular::singular_integration_with_params(
                    &source_point,
                    &source_normal,
                    &element_coords,
                    ElementType::Tri3,
                    &physics,
                    None,
                    0,     // Dirichlet BC type
                    false, // don't compute RHS
                    &quad_params,
                    None, // no curved normals
                );

                // H matrix coefficient (∂G/∂n term) plus the boundary free term
                // for self evaluation. The singular integration returns only the
                // principal-value integral, not the BIE jump term.
                row[j] = result.dg_dn_integral;
                if i == j {
                    row[j] += interior_double_layer_boundary_limit();
                }
            } else {
                // Use point collocation for far-field
                if i == j {
                    row[j] = interior_double_layer_boundary_limit();
                } else {
                    // Off-diagonal: standard collocation
                    let dx = center_i.x - center_j.x;
                    let dy = center_i.y - center_j.y;
                    let dz = center_i.z - center_j.z;

                    let cos_angle = (dx * normal_j.x + dy * normal_j.y + dz * normal_j.z) / r;
                    row[j] = -greens_function_derivative(r, k, cos_angle) * area_j;
                }
            }
        }
        row
    });

    // Convert row buffers to the BEM array surface.
    let mut matrix = sonium_solvers::dense::DenseMatrix::<Complex64>::zeros(n, n);
    for (i, row) in rows.iter().enumerate() {
        for (j, &val) in row.iter().enumerate() {
            matrix[[i, j]] = val;
        }
    }

    matrix
}

/// Helper: compute characteristic length of an element
fn element_characteristic_length(nodes: &[Point3D]) -> f64 {
    if nodes.len() < 3 {
        return 0.0;
    }

    // For triangles: use average edge length
    let d01 = nodes[0].distance_to(&nodes[1]);
    let d12 = nodes[1].distance_to(&nodes[2]);
    let d20 = nodes[2].distance_to(&nodes[0]);

    (d01 + d12 + d20) / 3.0
}

/// Helper: convert Point3D to Array1<f64>
fn point_to_array1(p: &Point3D) -> [f64; 3] {
    [p.x, p.y, p.z]
}

/// Helper: convert normal vector to [f64; 3]
fn normal_to_array1(n: &Point3D) -> [f64; 3] {
    [n.x, n.y, n.z]
}

/// Helper: convert nodes to DenseMatrix<f64> for singular integration
fn nodes_to_array2(nodes: &[Point3D]) -> DenseMatrix<f64> {
    let n = nodes.len();
    let mut coords = DenseMatrix::<f64>::zeros(n, 3);
    for (i, node) in nodes.iter().enumerate() {
        coords[[i, 0]] = node.x;
        coords[[i, 1]] = node.y;
        coords[[i, 2]] = node.z;
    }
    coords
}

/// Calculate incident field normal derivative in parallel
pub fn calculate_incident_field_derivative_parallel(
    mesh: &RoomMesh,
    sources: &[Source],
    k: f64,
    frequency: f64,
) -> Vec<Complex64> {
    // Incident-field precompute (rayon above the machine-adaptive gate,
    // serial for small meshes). Work is elements × sources.
    let element_data: Vec<_> = gated_map(
        &mesh.elements,
        mesh.elements.len().saturating_mul(sources.len()),
        BEM_ENTRIES_PER_THREAD,
        |element| {
            let nodes: Vec<Point3D> = element.nodes.iter().map(|&idx| mesh.nodes[idx]).collect();
            let (center, normal) = element_center_and_normal(&nodes);

            // Compute incident field derivative
            let mut dpdn_inc = Complex64::new(0.0, 0.0);

            for source in sources {
                let r = center.distance_to(&source.position);
                if r < 1e-10 {
                    continue;
                }

                let amplitude = source.amplitude_towards(&center, frequency);

                // Direction from source to point
                let dx = center.x - source.position.x;
                let dy = center.y - source.position.y;
                let dz = center.z - source.position.z;

                // Normal derivative: ∂G/∂n = ∇G · n
                let cos_angle = (dx * normal.x + dy * normal.y + dz * normal.z) / r;

                dpdn_inc += greens_function_derivative(r, k, cos_angle) * amplitude;
            }

            // For rigid boundary condition: ∂p/∂n = 0
            // The BEM formulation gives: H * q = G * p_inc
            // where q = ∂p/∂n is the unknown (should be zero for rigid)
            // We solve: H * q = -∂p_inc/∂n  (to get total field with zero normal derivative)
            -dpdn_inc
        },
    );

    element_data
}

/// Calculate pressure at field points using BEM solution (parallel version)
///
/// Uses the double-layer potential (DLP) representation for field evaluation:
/// p(x) = p_inc(x) + ∫∫ (∂G/∂n)(x, y) * p_surface(y) dS(y)
///
/// This matches the BEM formulation which solves for surface pressure using
/// the H matrix (double-layer potential operator).
pub fn calculate_field_pressure_bem_parallel(
    mesh: &RoomMesh,
    surface_pressure: &[Complex64],
    sources: &[Source],
    field_points: &[Point3D],
    k: f64,
    frequency: f64,
) -> Vec<Complex64> {
    calculate_field_pressure_bem_parallel_with_config(
        mesh,
        surface_pressure,
        sources,
        field_points,
        k,
        frequency,
        &RoomTbemConfig::default(),
    )
}

/// Calculate pressure at field points using the same source-directivity
/// convention as the supplied TBEM config.
pub fn calculate_field_pressure_bem_parallel_with_config(
    mesh: &RoomMesh,
    surface_pressure: &[Complex64],
    sources: &[Source],
    field_points: &[Point3D],
    k: f64,
    frequency: f64,
    config: &RoomTbemConfig,
) -> Vec<Complex64> {
    let prepared = prepare_room_tbem_geometry_with_config(mesh, k, frequency, config);
    calculate_field_pressure_bem_parallel_with_prepared_geometry(
        &prepared,
        surface_pressure,
        sources,
        field_points,
    )
}

/// Calculate receiver pressures from prepared TBEM geometry.
pub fn calculate_field_pressure_bem_parallel_with_prepared_geometry(
    prepared: &RoomTbemPreparedGeometry,
    surface_pressure: &[Complex64],
    sources: &[Source],
    field_points: &[Point3D],
) -> Vec<Complex64> {
    let evaluator = prepare_room_tbem_field_evaluator(prepared, field_points);
    calculate_field_pressure_bem_parallel_with_field_evaluator(
        &evaluator,
        surface_pressure,
        sources,
    )
}

/// Calculate receiver pressures from a precomputed TBEM field evaluator.
pub fn calculate_field_pressure_bem_parallel_with_field_evaluator(
    evaluator: &RoomTbemFieldEvaluator,
    surface_pressure: &[Complex64],
    sources: &[Source],
) -> Vec<Complex64> {
    if evaluator.field_points.len() <= 8 {
        return calculate_field_pressure_bem_with_field_evaluator(
            evaluator,
            surface_pressure,
            sources,
        );
    }

    assert!(
        evaluator.source_directivity != RoomSourceDirectivityMode::ReceiverOnly
            || sources.len() == 1,
        "receiver-only TBEM field evaluation requires a per-source solve; call once per source"
    );
    assert_eq!(
        surface_pressure.len(),
        evaluator.n_elements,
        "surface pressure length mismatch"
    );

    // Calculate pressure at each field point (rayon above the
    // machine-adaptive gate, serial for small point counts).
    let pressures: Vec<_> = gated_map_indexed(
        evaluator.field_points.len(),
        evaluator.field_points.len(),
        BEM_ENTRIES_PER_THREAD,
        |lp_idx| {
            let point = &evaluator.field_points[lp_idx];
            // Incident field from sources
            let mut p_incident = Complex64::new(0.0, 0.0);
            for source in sources {
                let r = point.distance_to(&source.position);
                if r < 1e-10 {
                    continue;
                }
                let amplitude = source_amplitude_for_boundary(
                    source,
                    point,
                    evaluator.frequency,
                    evaluator.source_directivity,
                );
                p_incident += greens_function_3d(r, evaluator.wave_number) * amplitude;
            }

            // Scattered field from boundary integral using double-layer potential
            // p_scattered = ∫∫ (∂G/∂n)(x, y) * p_surface(y) dS(y)
            // where ∂G/∂n is the normal derivative at the surface element (pointing outward)
            let kernel_row = &evaluator.kernels
                [lp_idx * evaluator.n_elements..(lp_idx + 1) * evaluator.n_elements];
            let p_scattered = kernel_row
                .iter()
                .zip(surface_pressure.iter())
                .fold(Complex64::new(0.0, 0.0), |acc, (&kernel, &pressure)| {
                    acc + kernel * pressure
                });

            let pressure = p_incident + p_scattered;
            if evaluator.source_directivity == RoomSourceDirectivityMode::ReceiverOnly
                && sources.len() == 1
            {
                pressure
                    * Complex64::new(
                        sources[0].directivity_factor(point, evaluator.frequency),
                        0.0,
                    )
            } else {
                pressure
            }
        },
    );

    pressures
}

/// Calculate receiver pressures serially from a precomputed TBEM field evaluator.
pub fn calculate_field_pressure_bem_with_field_evaluator(
    evaluator: &RoomTbemFieldEvaluator,
    surface_pressure: &[Complex64],
    sources: &[Source],
) -> Vec<Complex64> {
    assert!(
        evaluator.source_directivity != RoomSourceDirectivityMode::ReceiverOnly
            || sources.len() == 1,
        "receiver-only TBEM field evaluation requires a per-source solve; call once per source"
    );
    assert_eq!(
        surface_pressure.len(),
        evaluator.n_elements,
        "surface pressure length mismatch"
    );

    (0..evaluator.field_points.len())
        .map(|lp_idx| {
            let point = &evaluator.field_points[lp_idx];
            let mut p_incident = Complex64::new(0.0, 0.0);
            for source in sources {
                let r = point.distance_to(&source.position);
                if r < 1e-10 {
                    continue;
                }
                let amplitude = source_amplitude_for_boundary(
                    source,
                    point,
                    evaluator.frequency,
                    evaluator.source_directivity,
                );
                p_incident += greens_function_3d(r, evaluator.wave_number) * amplitude;
            }

            let kernel_row = &evaluator.kernels
                [lp_idx * evaluator.n_elements..(lp_idx + 1) * evaluator.n_elements];
            let p_scattered = kernel_row
                .iter()
                .zip(surface_pressure.iter())
                .fold(Complex64::new(0.0, 0.0), |acc, (&kernel, &pressure)| {
                    acc + kernel * pressure
                });

            let pressure = p_incident + p_scattered;
            if evaluator.source_directivity == RoomSourceDirectivityMode::ReceiverOnly
                && sources.len() == 1
            {
                pressure
                    * Complex64::new(
                        sources[0].directivity_factor(point, evaluator.frequency),
                        0.0,
                    )
            } else {
                pressure
            }
        })
        .collect()
}

// ============================================================================
// FMM Integration for Room Acoustics
// ============================================================================

use crate::core::assembly::slfmm::{SlfmmSystem, build_slfmm_system_stabilized};
use crate::core::mesh::octree::Octree;
use crate::core::solver::{
    GmresConfig, GmresSolution, try_gmres_solve_fmm_batched_with_hilu,
    try_gmres_solve_fmm_batched_with_ilu,
};
use crate::core::types::{
    BoundaryCondition, Cluster, Element, ElementProperty, ElementType, PhysicsParams,
};

/// FMM solver configuration
pub struct FmmSolverConfig {
    /// Maximum elements per octree leaf (affects cluster size)
    pub max_elements_per_leaf: usize,
    /// Maximum octree depth
    pub max_tree_depth: usize,
    /// Number of theta integration points on unit sphere
    pub n_theta: usize,
    /// Number of phi integration points on unit sphere
    pub n_phi: usize,
    /// Number of expansion terms
    pub n_terms: usize,
    /// Separation ratio for near/far field classification
    pub separation_ratio: f64,
}

impl Default for FmmSolverConfig {
    fn default() -> Self {
        Self {
            max_elements_per_leaf: 50,
            max_tree_depth: 8,
            n_theta: 6,
            n_phi: 12,
            n_terms: 6,
            separation_ratio: 1.5, // Standard FMM separation: 2/sqrt(3) ≈ 1.155
        }
    }
}

/// Convert RoomMesh to core Element and nodes arrays for FMM
pub fn room_mesh_to_core_elements(mesh: &RoomMesh, _k: f64) -> (Vec<Element>, Vec<[f64; 3]>) {
    room_mesh_to_core_elements_with_admittance(mesh, _k, None)
}

/// Convert RoomMesh to core Element and nodes arrays, optionally applying per-element admittance.
pub fn room_mesh_to_core_elements_with_admittance(
    mesh: &RoomMesh,
    _k: f64,
    element_admittance: Option<&[Complex64]>,
) -> (Vec<Element>, Vec<[f64; 3]>) {
    let n_nodes = mesh.nodes.len();
    let n_elements = mesh.elements.len();
    if let Some(admittance) = element_admittance {
        assert_eq!(
            admittance.len(),
            n_elements,
            "room element admittance length must match mesh element count"
        );
    }

    // Convert nodes to Vec<[f64; 3]>
    let mut nodes: Vec<[f64; 3]> = vec![[0.0; 3]; n_nodes];
    for (i, node) in mesh.nodes.iter().enumerate() {
        nodes[i][0] = node.x;
        nodes[i][1] = node.y;
        nodes[i][2] = node.z;
    }

    // Convert elements
    let mut elements = Vec::with_capacity(n_elements);
    for (elem_idx, surface_elem) in mesh.elements.iter().enumerate() {
        let elem_nodes: Vec<Point3D> = surface_elem.nodes.iter().map(|&i| mesh.nodes[i]).collect();

        let (center, normal) = element_center_and_normal(&elem_nodes);
        let area = element_area(&elem_nodes);

        // Determine element type
        let elem_type = if surface_elem.nodes.len() == 3 {
            ElementType::Tri3
        } else {
            ElementType::Quad4
        };

        let admittance = element_admittance.map(|values| values[elem_idx]);
        let boundary_condition = match admittance {
            Some(value) if value.norm() > 0.0 => BoundaryCondition::VelocityWithAdmittance {
                velocity: vec![Complex64::new(0.0, 0.0)],
                admittance: value,
            },
            _ => BoundaryCondition::Velocity(vec![Complex64::new(0.0, 0.0)]),
        };

        let element = Element {
            connectivity: surface_elem.nodes.clone(),
            element_type: elem_type,
            property: ElementProperty::Surface,
            normal: [normal.x, normal.y, normal.z],
            node_normals: vec![[0.0; 3]; surface_elem.nodes.len()],
            center: [center.x, center.y, center.z],
            area,
            boundary_condition,
            group: 0,
            dof_addresses: vec![elem_idx],
        };

        elements.push(element);
    }

    (elements, nodes)
}

/// Build clusters from octree for FMM
pub fn build_clusters_from_octree(octree: &Octree, elements: &[Element]) -> Vec<Cluster> {
    let leaves = octree.leaves();
    let mut clusters = Vec::with_capacity(leaves.len());

    for &leaf_idx in &leaves {
        let node = &octree.nodes[leaf_idx];
        if node.element_indices.is_empty() {
            continue;
        }

        let mut cluster = Cluster::new(node.center);
        cluster.element_indices = node.element_indices.clone();
        cluster.num_elements = node.element_indices.len();
        cluster.element_property = ElementProperty::Surface;
        cluster.radius = node.radius();
        cluster.level = node.level;

        // Count DOFs
        cluster.num_dofs = node
            .element_indices
            .iter()
            .filter(|&&i| !elements[i].property.is_evaluation())
            .count();
        cluster.dofs_per_element = 1;

        clusters.push(cluster);
    }

    // Build near/far lists using octree's computed lists
    // Map from octree leaf indices to cluster indices
    let leaf_to_cluster: std::collections::HashMap<usize, usize> = leaves
        .iter()
        .filter(|&&i| !octree.nodes[i].element_indices.is_empty())
        .enumerate()
        .map(|(cluster_idx, &leaf_idx)| (leaf_idx, cluster_idx))
        .collect();

    // Assign near/far cluster indices
    for (cluster_idx, &leaf_idx) in leaves.iter().enumerate() {
        if octree.nodes[leaf_idx].element_indices.is_empty() {
            continue;
        }

        let octree_node = &octree.nodes[leaf_idx];

        // Map near clusters
        let near: Vec<usize> = octree_node
            .near_clusters
            .iter()
            .filter_map(|&near_leaf| leaf_to_cluster.get(&near_leaf).copied())
            .collect();

        // Map far clusters
        let far: Vec<usize> = octree_node
            .far_clusters
            .iter()
            .filter_map(|&far_leaf| leaf_to_cluster.get(&far_leaf).copied())
            .collect();

        if let Some(cluster) = clusters.get_mut(cluster_idx) {
            cluster.near_clusters = near;
            cluster.far_clusters = far;
        }
    }

    clusters
}

/// Build FMM system for room acoustics
pub fn build_fmm_system(
    mesh: &RoomMesh,
    sources: &[Source],
    k: f64,
    frequency: f64,
    fmm_config: &FmmSolverConfig,
) -> FmmSystemResult {
    println!("  Converting room mesh to core elements...");
    let (elements, nodes) = room_mesh_to_core_elements(mesh, k);
    println!("    {} elements, {} nodes", elements.len(), nodes.len());

    // Compute element centers for octree
    println!("  Building octree...");
    let centers: Vec<[f64; 3]> = elements.iter().map(|e| e.center).collect();

    let mut octree = Octree::build(
        &centers,
        fmm_config.max_elements_per_leaf,
        fmm_config.max_tree_depth,
    );

    // Compute interaction lists
    octree.compute_interaction_lists(fmm_config.separation_ratio);

    let stats = octree.stats();
    println!(
        "    {} leaves, {} levels, avg {:.1} elements/leaf",
        stats.num_leaves, stats.num_levels, stats.avg_elements_per_leaf
    );

    // Build clusters from octree
    println!("  Building clusters...");
    let clusters = build_clusters_from_octree(&octree, &elements);
    println!("    {} clusters", clusters.len());

    // Create physics parameters
    let speed_of_sound = 343.0;
    let physics = PhysicsParams::new(frequency, speed_of_sound, 1.21, true);

    // Build SLFMM system
    println!("  Assembling SLFMM system...");
    let mut system = build_slfmm_system_stabilized(
        &elements,
        &nodes,
        &clusters,
        &physics,
        fmm_config.n_theta,
        fmm_config.n_phi,
        fmm_config.n_terms,
        1.0, // sign: exterior problem convention
        1.0, // beta_scale: default
    );

    // Compute RHS
    println!("  Computing RHS...");
    let rhs = calculate_incident_field_derivative_parallel(mesh, sources, k, frequency);
    system.rhs = rhs;

    Ok((system, elements, nodes))
}

/// Solve BEM system using FMM + GMRES + ILU
///
/// This is the recommended solver for large meshes (>1000 elements).
/// Complexity: O(N log N) per iteration vs O(N²) for dense GMRES.
///
/// The near-field matrix for ILU preconditioning is extracted directly
/// from the SLFMM system, avoiding the O(N²) dense matrix assembly.
pub fn solve_bem_fmm_gmres_ilu(
    mesh: &RoomMesh,
    sources: &[Source],
    k: f64,
    frequency: f64,
    fmm_config: &FmmSolverConfig,
    gmres_max_iter: usize,
    gmres_restart: usize,
    gmres_tolerance: f64,
) -> Result<Vec<Complex64>, String> {
    // Build FMM system
    let (system, _elements, _nodes) = build_fmm_system(mesh, sources, k, frequency, fmm_config)?;

    // GMRES configuration
    let gmres_config = GmresConfig {
        max_iterations: gmres_max_iter,
        restart: gmres_restart,
        tolerance: gmres_tolerance,
        print_interval: 0,
        convergence_check_interval: 1,
        exact_residual_check_interval: 0,
        min_restart: None,
        max_restart: None,
    };

    // Get RHS from system
    let rhs = system.rhs.clone();

    // Solve with FMM-accelerated GMRES + ILU preconditioning
    println!("  Solving with FMM + GMRES + ILU...");
    let result = try_gmres_solve_fmm_batched_with_ilu(&system, &rhs, &gmres_config)
        .map_err(|error| error.to_string())?;

    if result.converged {
        println!(
            "    Converged in {} iterations, residual: {:.2e}",
            result.iterations, result.residual
        );
    } else {
        println!(
            "    Warning: Did not converge after {} iterations, residual: {:.2e}",
            result.iterations, result.residual
        );
    }

    Ok(result.x)
}

/// Solve a room BEM system with FMM, GMRES, and checked hierarchical ILU.
///
/// H-ILU is assembled from the exact FMM near field, while far-field actions
/// remain matrix-free. The returned solution is certified against the full
/// SLFMM operator; this is a single-node path and does not claim distributed
/// ownership or communication overlap.
pub fn solve_bem_fmm_gmres_hierarchical(
    mesh: &RoomMesh,
    sources: &[Source],
    k: f64,
    frequency: f64,
    fmm_config: &FmmSolverConfig,
    gmres_max_iter: usize,
    gmres_restart: usize,
    gmres_tolerance: f64,
) -> Result<Vec<Complex64>, String> {
    solve_bem_fmm_gmres_hierarchical_with_result(
        mesh,
        sources,
        k,
        frequency,
        fmm_config,
        gmres_max_iter,
        gmres_restart,
        gmres_tolerance,
    )
    .map(|result| result.x)
}

/// Solve room BEM system with checked hierarchical FMM GMRES and return the
/// complete convergence result.
///
/// The H-ILU is assembled from the exact FMM near field, while far-field
/// actions remain matrix-free. The returned residual is certified against the
/// full SLFMM operator; this is a single-node path and does not claim
/// distributed ownership or communication overlap.
pub fn solve_bem_fmm_gmres_hierarchical_with_result(
    mesh: &RoomMesh,
    sources: &[Source],
    k: f64,
    frequency: f64,
    fmm_config: &FmmSolverConfig,
    gmres_max_iter: usize,
    gmres_restart: usize,
    gmres_tolerance: f64,
) -> Result<GmresSolution<Complex64>, String> {
    // Build FMM system
    let (system, _elements, _nodes) = build_fmm_system(mesh, sources, k, frequency, fmm_config)?;

    // GMRES configuration
    let gmres_config = GmresConfig {
        max_iterations: gmres_max_iter,
        restart: gmres_restart,
        tolerance: gmres_tolerance,
        print_interval: 0,
        convergence_check_interval: 1,
        exact_residual_check_interval: 0,
        min_restart: None,
        max_restart: None,
    };

    // Get RHS
    let rhs = system.rhs.clone();

    println!("  Solving with FMM + GMRES + hierarchical ILU...");
    let result = try_gmres_solve_fmm_batched_with_hilu(&system, &rhs, &gmres_config)
        .map_err(|error| error.to_string())?;

    if result.converged {
        println!(
            "    Converged in {} iterations, residual: {:.2e}",
            result.iterations, result.residual
        );
    } else {
        println!(
            "    Warning: Did not converge after {} iterations, residual: {:.2e}",
            result.iterations, result.residual
        );
    }

    Ok(result)
}

/// Solve BEM system using FMM + GMRES + ILU with full result
///
/// Same as `solve_bem_fmm_gmres_ilu` but returns the full GmresSolution with
/// convergence info.
pub fn solve_bem_fmm_gmres_ilu_with_result(
    mesh: &RoomMesh,
    sources: &[Source],
    k: f64,
    frequency: f64,
    fmm_config: &FmmSolverConfig,
    gmres_max_iter: usize,
    gmres_restart: usize,
    gmres_tolerance: f64,
) -> Result<GmresSolution<Complex64>, String> {
    // Build FMM system
    let (system, _elements, _nodes) = build_fmm_system(mesh, sources, k, frequency, fmm_config)?;

    // GMRES configuration
    let gmres_config = GmresConfig {
        max_iterations: gmres_max_iter,
        restart: gmres_restart,
        tolerance: gmres_tolerance,
        print_interval: 0,
        convergence_check_interval: 1,
        exact_residual_check_interval: 0,
        min_restart: None,
        max_restart: None,
    };

    // Get RHS from system
    let rhs = system.rhs.clone();

    // Solve with FMM-accelerated GMRES + ILU preconditioning
    println!("  Solving with FMM + GMRES + ILU...");
    let result = try_gmres_solve_fmm_batched_with_ilu(&system, &rhs, &gmres_config)
        .map_err(|error| error.to_string())?;

    if result.converged {
        println!(
            "    Converged in {} iterations, residual: {:.2e}",
            result.iterations, result.residual
        );
    } else {
        println!(
            "    Warning: Did not converge after {} iterations, residual: {:.2e}",
            result.iterations, result.residual
        );
    }

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hierarchical_fmm_preconditioner_solves_with_hilu() {
        let mesh = tetrahedron_mesh();
        let source = Source::omnidirectional(Point3D::new(0.2, 0.2, 0.2), 1.0);
        let config = FmmSolverConfig {
            max_elements_per_leaf: 1,
            n_theta: 4,
            n_phi: 8,
            n_terms: 4,
            ..FmmSolverConfig::default()
        };
        let result = solve_bem_fmm_gmres_hierarchical_with_result(
            &mesh,
            &[source],
            1.0,
            100.0,
            &config,
            100,
            10,
            1.0e-6,
        )
        .expect("hierarchical FMM solve");

        assert!(result.converged);
        assert!(result.residual.is_finite() && result.residual <= 1.0e-6);
        let solution = result.x;
        assert_eq!(solution.len(), mesh.elements.len());
        assert!(
            solution
                .iter()
                .all(|value| value.re.is_finite() && value.im.is_finite())
        );
    }

    fn single_flat_triangle_mesh() -> RoomMesh {
        let nodes = vec![
            Point3D::new(0.0, 0.0, 0.0),
            Point3D::new(1.0, 0.0, 0.0),
            Point3D::new(0.0, 1.0, 0.0),
        ];
        let elements = vec![SurfaceElement::triangle(0, 1, 2)];
        RoomMesh { nodes, elements }
    }

    fn two_triangle_mesh() -> RoomMesh {
        let nodes = vec![
            Point3D::new(0.0, 0.0, 0.0),
            Point3D::new(1.0, 0.0, 0.0),
            Point3D::new(0.0, 1.0, 0.0),
            Point3D::new(0.0, 0.0, 1.0),
            Point3D::new(1.0, 0.0, 1.0),
            Point3D::new(0.0, 1.0, 1.0),
        ];
        let elements = vec![
            SurfaceElement::triangle(0, 1, 2),
            SurfaceElement::triangle(3, 4, 5),
        ];
        RoomMesh { nodes, elements }
    }

    fn tetrahedron_mesh() -> RoomMesh {
        let nodes = vec![
            Point3D::new(0.0, 0.0, 0.0),
            Point3D::new(1.0, 0.0, 0.0),
            Point3D::new(0.0, 1.0, 0.0),
            Point3D::new(0.0, 0.0, 1.0),
        ];
        let elements = vec![
            SurfaceElement::triangle(0, 1, 2),
            SurfaceElement::triangle(0, 3, 1),
            SurfaceElement::triangle(0, 2, 3),
            SurfaceElement::triangle(1, 3, 2),
        ];
        RoomMesh { nodes, elements }
    }

    fn assert_self_term_is_boundary_limit(matrix: &sonium_solvers::dense::DenseMatrix<Complex64>) {
        let diag = matrix[[0, 0]];
        assert!(
            (diag - Complex64::new(0.5, 0.0)).norm() < 1e-12,
            "room BEM self term must be the interior double-layer free term only, got {diag:?}"
        );
    }

    #[test]
    fn test_greens_function() {
        let k = 2.0 * PI * 1000.0 / 343.0;
        let r = 1.0;
        let g = greens_function_3d(r, k);
        // Should have magnitude approximately 1/(4πr)
        assert!((g.norm() - 1.0 / (4.0 * PI)).abs() < 0.1);
    }

    #[test]
    fn test_pressure_to_spl() {
        // pressure_to_spl interprets the complex argument as a peak
        // amplitude (what the BEM solves for) and returns RMS-referenced
        // SPL. 1 Pa peak ≈ 0.7071 Pa RMS ≈ 90.97 dB SPL.
        let p = Complex64::new(1.0, 0.0); // 1 Pa peak
        let spl = pressure_to_spl(p);
        assert!(
            (spl - 90.97).abs() < 0.05,
            "1 Pa peak should be ~90.97 dB SPL (RMS-referenced), got {spl}"
        );

        // 1 Pa RMS = sqrt(2) Pa peak should give exactly 94 dB SPL.
        let p_rms_1 = Complex64::new(std::f64::consts::SQRT_2, 0.0);
        let spl_rms = pressure_to_spl(p_rms_1);
        assert!(
            (spl_rms - 94.0).abs() < 0.05,
            "1 Pa RMS (sqrt(2) Pa peak) should be 94 dB SPL, got {spl_rms}"
        );
    }

    #[test]
    fn test_room_bem_self_term_uses_interior_boundary_limit() {
        let mesh = single_flat_triangle_mesh();
        let k = 2.0 * PI * 400.0 / 343.0;

        assert_self_term_is_boundary_limit(&build_bem_matrix(&mesh, k));
        assert_self_term_is_boundary_limit(&build_bem_matrix_parallel(&mesh, k));
        assert_self_term_is_boundary_limit(&build_bem_matrix_adaptive(&mesh, k, false));
        assert_self_term_is_boundary_limit(&build_bem_matrix_adaptive(&mesh, k, true));
    }

    #[test]
    fn test_room_mesh_to_core_elements_applies_element_admittance() {
        let mesh = two_triangle_mesh();
        let admittance = vec![Complex64::new(0.0, 0.0), Complex64::new(0.2, -0.03)];

        let (elements, _) =
            room_mesh_to_core_elements_with_admittance(&mesh, 1.0, Some(&admittance));

        assert!(matches!(
            elements[0].boundary_condition,
            BoundaryCondition::Velocity(_)
        ));
        match &elements[1].boundary_condition {
            BoundaryCondition::VelocityWithAdmittance {
                velocity,
                admittance,
            } => {
                assert_eq!(velocity, &vec![Complex64::new(0.0, 0.0)]);
                assert_eq!(*admittance, Complex64::new(0.2, -0.03));
            }
            other => panic!("expected admittance boundary condition, got {other:?}"),
        }
    }

    #[test]
    fn test_room_tbem_system_has_incident_rhs() {
        let mesh = single_flat_triangle_mesh();
        let frequency = 400.0;
        let k = 2.0 * PI * frequency / 343.0;
        let source = Source::omnidirectional(Point3D::new(0.25, 0.25, 1.0), 1.0);

        let system = build_room_tbem_system(&mesh, &[source], k, frequency);

        assert_eq!(system.num_dofs, 1);
        assert_eq!(system.rhs.len(), 1);
        assert!(
            system.rhs[0].norm() > 1e-12,
            "room TBEM RHS should include the incident pressure field"
        );
    }

    #[test]
    fn test_room_tbem_rhs_only_matches_full_system_rhs() {
        let mesh = two_triangle_mesh();
        let frequency = 400.0;
        let k = 2.0 * PI * frequency / 343.0;
        let source = Source::classical(Point3D::new(0.25, 0.25, 1.0), 10.0, 10.0, 1.0)
            .with_orientation(Point3D::new(1.0, 0.0, 0.0));
        let config = RoomTbemConfig {
            source_directivity: RoomSourceDirectivityMode::ReceiverOnly,
            element_admittance: Some(vec![Complex64::new(0.0, 0.0), Complex64::new(0.2, -0.03)]),
            ..RoomTbemConfig::default()
        };

        let system = build_room_tbem_system_with_config(
            &mesh,
            std::slice::from_ref(&source),
            k,
            frequency,
            &config,
        );
        let rhs = build_room_tbem_rhs_with_config(
            &mesh,
            std::slice::from_ref(&source),
            k,
            frequency,
            &config,
        );
        let prepared = prepare_room_tbem_geometry_with_config(&mesh, k, frequency, &config);
        let prepared_system =
            build_room_tbem_system_with_prepared_geometry(&prepared, std::slice::from_ref(&source));
        let prepared_rhs =
            build_room_tbem_rhs_with_prepared_geometry(&prepared, std::slice::from_ref(&source));

        assert_eq!(rhs, system.rhs);
        assert_eq!(prepared_rhs, system.rhs);
        assert_eq!(prepared_system.rhs, system.rhs);
        assert_eq!(prepared_system.matrix.nrows(), system.matrix.nrows());
        assert_eq!(prepared_system.matrix.ncols(), system.matrix.ncols());
        for row in 0..system.matrix.nrows() {
            for col in 0..system.matrix.ncols() {
                let delta = prepared_system.matrix[[row, col]] - system.matrix[[row, col]];
                assert!(
                    delta.norm() < 1e-12,
                    "prepared matrix drifted at ({row}, {col}): {delta:?}"
                );
            }
        }
    }

    #[test]
    fn test_prepared_tbem_field_matches_direct_field() {
        let mesh = two_triangle_mesh();
        let frequency = 400.0;
        let k = 2.0 * PI * frequency / 343.0;
        let source = Source::classical(Point3D::new(0.25, 0.25, 1.0), 10.0, 10.0, 1.0)
            .with_orientation(Point3D::new(1.0, 0.0, 0.0));
        let field_points = vec![
            Point3D::new(0.25, 0.25, 0.5),
            Point3D::new(0.75, 0.25, 0.75),
        ];
        let surface_pressure = vec![Complex64::new(0.5, -0.25), Complex64::new(-0.1, 0.2)];
        let config = RoomTbemConfig {
            source_directivity: RoomSourceDirectivityMode::ReceiverOnly,
            element_admittance: Some(vec![Complex64::new(0.0, 0.0), Complex64::new(0.2, -0.03)]),
            ..RoomTbemConfig::default()
        };

        let direct = calculate_field_pressure_bem_parallel_with_config(
            &mesh,
            &surface_pressure,
            std::slice::from_ref(&source),
            &field_points,
            k,
            frequency,
            &config,
        );
        let prepared = prepare_room_tbem_geometry_with_config(&mesh, k, frequency, &config);
        let cached = calculate_field_pressure_bem_parallel_with_prepared_geometry(
            &prepared,
            &surface_pressure,
            std::slice::from_ref(&source),
            &field_points,
        );
        let evaluator = prepare_room_tbem_field_evaluator(&prepared, &field_points);
        let cached_from_evaluator = calculate_field_pressure_bem_parallel_with_field_evaluator(
            &evaluator,
            &surface_pressure,
            std::slice::from_ref(&source),
        );

        assert_eq!(cached.len(), direct.len());
        for (idx, (cached, direct)) in cached.iter().zip(direct.iter()).enumerate() {
            assert!(
                (*cached - *direct).norm() < 1e-12,
                "prepared field drifted at point {idx}: cached={cached:?} direct={direct:?}"
            );
        }
        for (idx, (cached, direct)) in cached_from_evaluator.iter().zip(direct.iter()).enumerate() {
            assert!(
                (*cached - *direct).norm() < 1e-12,
                "prepared field evaluator drifted at point {idx}: cached={cached:?} direct={direct:?}"
            );
        }
    }

    #[test]
    fn test_room_tbem_receiver_only_directivity_uses_isotropic_boundary_excitation() {
        let mesh = single_flat_triangle_mesh();
        let frequency = 400.0;
        let k = 2.0 * PI * frequency / 343.0;
        let directional = Source::classical(Point3D::new(0.25, 0.25, 1.0), 10.0, 10.0, 1.0)
            .with_orientation(Point3D::new(1.0, 0.0, 0.0));
        let omni = Source::omnidirectional(Point3D::new(0.25, 0.25, 1.0), 1.0);

        let receiver_only = build_room_tbem_system_with_config(
            &mesh,
            &[directional.clone()],
            k,
            frequency,
            &RoomTbemConfig {
                source_directivity: RoomSourceDirectivityMode::ReceiverOnly,
                ..RoomTbemConfig::default()
            },
        );
        let omni_reference = build_room_tbem_system(&mesh, &[omni], k, frequency);
        let directional_boundary = build_room_tbem_system(&mesh, &[directional], k, frequency);

        assert!((receiver_only.rhs[0] - omni_reference.rhs[0]).norm() < 1.0e-12);
        assert!(
            (directional_boundary.rhs[0] - omni_reference.rhs[0]).norm() > 1.0e-6,
            "default TBEM should still honor boundary-directed source directivity"
        );
    }

    #[test]
    #[should_panic(expected = "receiver-only TBEM field evaluation requires a per-source solve")]
    fn test_room_tbem_receiver_only_field_evaluation_rejects_multi_source_sum() {
        let mesh = single_flat_triangle_mesh();
        let frequency = 400.0;
        let k = 2.0 * PI * frequency / 343.0;
        let sources = vec![
            Source::classical(Point3D::new(0.25, 0.25, 1.0), 10.0, 10.0, 1.0),
            Source::classical(Point3D::new(0.75, 0.25, 1.0), 10.0, 10.0, 1.0),
        ];

        calculate_field_pressure_bem_parallel_with_config(
            &mesh,
            &[Complex64::new(0.0, 0.0)],
            &sources,
            &[Point3D::new(0.5, 0.5, 0.5)],
            k,
            frequency,
            &RoomTbemConfig {
                source_directivity: RoomSourceDirectivityMode::ReceiverOnly,
                ..RoomTbemConfig::default()
            },
        );
    }

    #[test]
    fn test_room_tbem_exposes_burton_miller_control() {
        let mesh = two_triangle_mesh();
        let frequency = 600.0;
        let k = 2.0 * PI * frequency / 343.0;

        let cbie = build_room_tbem_system_with_config(
            &mesh,
            &[],
            k,
            frequency,
            &RoomTbemConfig {
                beta_scale: 0.0,
                ..RoomTbemConfig::default()
            },
        );
        let burton_miller = build_room_tbem_system_with_config(
            &mesh,
            &[],
            k,
            frequency,
            &RoomTbemConfig {
                beta_scale: 1.0,
                ..RoomTbemConfig::default()
            },
        );

        let delta = burton_miller
            .matrix
            .as_slice()
            .iter()
            .zip(cbie.matrix.as_slice().iter())
            .map(|(a, b)| (a - b).norm())
            .fold(0.0, f64::max);
        assert!(
            delta > 1e-12,
            "non-zero beta_scale should change the core room TBEM matrix"
        );
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn test_room_tbem_cuda_matches_cpu() {
        let backend = match sonium_compute::CudaBackend::new_default() {
            Ok(backend) => backend,
            Err(err) => {
                eprintln!("CUDA unavailable, skipping room TBEM CUDA parity test: {err}");
                return;
            }
        };
        if !backend.has_bem_regular_pair_c32() {
            eprintln!("CUDA BEM regular-pair kernel unavailable, skipping room TBEM parity test");
            return;
        }

        let mesh = two_triangle_mesh();
        let frequency = 600.0;
        let k = 2.0 * PI * frequency / 343.0;
        let source = Source::omnidirectional(Point3D::new(0.25, 0.25, 1.0), 1.0);
        let config = RoomTbemConfig::default();

        let cpu =
            build_room_tbem_system_with_config(&mesh, &[source.clone()], k, frequency, &config);
        let gpu = build_room_tbem_system_with_config_cuda(
            &mesh,
            &[source],
            k,
            frequency,
            &config,
            &backend,
        );

        assert_eq!(
            cpu.matrix.shape(),
            gpu.matrix.shape(),
            "matrix dims must agree"
        );
        assert_eq!(cpu.rhs.len(), gpu.rhs.len(), "rhs lens must agree");

        let m_diff: f64 = cpu
            .matrix
            .as_slice()
            .iter()
            .zip(gpu.matrix.as_slice().iter())
            .map(|(a, b)| (a - b).norm_sqr())
            .sum::<f64>()
            .sqrt();
        let m_ref: f64 = cpu
            .matrix
            .as_slice()
            .iter()
            .map(|v| v.norm_sqr())
            .sum::<f64>()
            .sqrt();
        let r_diff: f64 = cpu
            .rhs
            .iter()
            .zip(gpu.rhs.iter())
            .map(|(a, b)| (a - b).norm_sqr())
            .sum::<f64>()
            .sqrt();
        let r_ref: f64 = cpu.rhs.iter().map(|v| v.norm_sqr()).sum::<f64>().sqrt();

        assert!(
            m_diff / m_ref.max(1e-30) < 1e-3,
            "CUDA room matrix relative diff too high"
        );
        assert!(
            r_diff / r_ref.max(1e-30) < 1e-3,
            "CUDA room RHS relative diff too high"
        );
    }

    #[test]
    fn test_room_core_tbem_solve_small_mesh() {
        let mesh = single_flat_triangle_mesh();
        let frequency = 400.0;
        let k = 2.0 * PI * frequency / 343.0;
        let source = Source::omnidirectional(Point3D::new(0.25, 0.25, 1.0), 1.0);

        let solution = solve_bem_system_core_tbem(&mesh, &[source], k, frequency)
            .expect("small core TBEM room solve should converge");

        assert_eq!(solution.len(), mesh.num_elements());
        assert!(
            solution
                .iter()
                .all(|p| p.re.is_finite() && p.im.is_finite())
        );
    }

    #[test]
    fn test_room_direct_solve_small_mesh() {
        let mesh = single_flat_triangle_mesh();
        let frequency = 400.0;
        let k = 2.0 * PI * frequency / 343.0;
        let source = Source::omnidirectional(Point3D::new(0.25, 0.25, 1.0), 1.0);

        let solution = solve_bem_system_direct(&mesh, &[source], k, frequency)
            .expect("small direct room solve should succeed");

        assert_eq!(solution.len(), mesh.num_elements());
        assert!(
            solution
                .iter()
                .all(|p| p.re.is_finite() && p.im.is_finite())
        );
    }

    #[test]
    fn test_room_mesh_to_core_elements() {
        // Create a simple test mesh
        let nodes = vec![
            Point3D::new(0.0, 0.0, 0.0),
            Point3D::new(1.0, 0.0, 0.0),
            Point3D::new(0.5, 1.0, 0.0),
        ];
        let elements = vec![SurfaceElement {
            nodes: vec![0, 1, 2],
        }];
        let mesh = RoomMesh { nodes, elements };

        let k = 2.0 * PI * 100.0 / 343.0;
        let (core_elements, core_nodes) = room_mesh_to_core_elements(&mesh, k);

        assert_eq!(core_elements.len(), 1);
        assert_eq!(core_nodes.len(), 3);
        assert_eq!(core_elements[0].element_type, ElementType::Tri3);
    }
}
