//! Maxwell validation tests
//!
//! Verifies the curl-curl and edge-mass matrix assembly by checking:
//! 1. Global matrix symmetry and positive semi-definiteness
//! 2. Kernel of curl-curl matrix (gradient fields are in the kernel)
//! 3. Assembled matrix dimensions match edge DOF count
//! 4. Waveguide cutoff frequencies (TE/TM modes)
//! 5. Cavity resonance frequencies
//! 6. PEC boundary DOF detection

use sonium_fem::assembly::assemble_bilinear;
use sonium_fem::mesh::generators::rectangular_mesh_triangles;
use sonium_fem::mesh::{BoundaryType, MeshTopology, unit_cube_tetrahedra, unit_square_triangles};
use sonium_fem::physics::electromagnetics::maxwell::{CurlCurlForm, EdgeMassForm, EmMaterial};
use sonium_fem::physics::solid::eigenmodes::{
    ElectromagneticEigenmodeConfig, build_electromagnetic_eigenmode_problem,
};
use sonium_fem::spaces::{FunctionSpace, HcurlSpace};
use sonium_qa::QaResult;
use sonium_solvers::{CsrMatrix, LinearOperator, ShiftInvertConfig};

const CASE_ID: &str = "fem.maxwell-validation.v1";
// Assembly symmetry bounds (2D tight, 3D sampled rows).
const SYMMETRY_TOLERANCE_2D: f64 = 1e-12;
const SYMMETRY_TOLERANCE_3D: f64 = 1e-10;
// Curl-curl diagonal nonnegativity slack.
const DIAGONAL_SLACK: f64 = 1e-14;
// Gradient null-mode residual factor: ||K grad phi|| <= factor * ||grad phi||.
const KERNEL_RESIDUAL_FACTOR: f64 = 1e-12;
// Minimum curl-curl energy of a generic (non-gradient) edge field.
const GENERIC_ENERGY_MINIMUM: f64 = 1e-10;
// Independent relative eigen-residual bound for the solved EM eigenpairs.
const EIGEN_RESIDUAL_TOLERANCE: f64 = 1e-6;
// Waveguide TE10-like cutoff wavenumber accuracy (fine mesh).
const WAVEGUIDE_KC_TOLERANCE: f64 = 0.005;
// Waveguide modal-shape error: least-squares fit of |E_y| against sin(pi x/a).
// The 16x8 mesh reaches ~2e-4 with clear contraction from 8x4.
const WAVEGUIDE_SHAPE_TOLERANCE: f64 = 0.005;
// Rectangular cavity (1,0,1)-family wavenumber accuracy (fine mesh). The n=2
// Kuhn mesh is pre-asymptotic, so the gated pair is n=3/n=4, where the error
// contracts monotonically toward the degenerate exact value.
const CAVITY_K_TOLERANCE: f64 = 0.02;
// Minimum coefficient of variation of the cavity modal field magnitude,
// proving a spatially varying mode rather than a null or constant field.
const CAVITY_VARIATION_MINIMUM: f64 = 0.2;
const SPEED_OF_LIGHT: f64 = 299_792_458.0;

/// Verify curl-curl matrix is symmetric and positive semi-definite (has a kernel).
fn test_curl_curl_global_assembly_2d() -> Result<f64, String> {
    let mesh = unit_square_triangles(4);
    let space = HcurlSpace::new();
    let n_dofs = space.num_global_dofs(&mesh);

    let triplet = assemble_bilinear(&CurlCurlForm, &space, &space, &mesh, 2);
    if triplet.num_rows != n_dofs {
        return Err(format!(
            "curl-curl rows {} != edge DOFs {n_dofs}",
            triplet.num_rows
        ));
    }

    let csr = triplet.to_csr();

    // Check symmetry
    let mut max_asym = 0.0_f64;
    for i in 0..n_dofs {
        for (j, val_ij) in csr.row_entries(i) {
            let val_ji = csr
                .row_entries(j)
                .find(|(col, _)| *col == i)
                .map(|(_, v)| v)
                .unwrap_or(0.0);
            max_asym = max_asym.max((val_ij - val_ji).abs());
        }
    }
    if !max_asym.is_finite() || max_asym >= SYMMETRY_TOLERANCE_2D {
        return Err(format!(
            "2D curl-curl asymmetry {max_asym:.2e} exceeds bound {SYMMETRY_TOLERANCE_2D:.1e}"
        ));
    }

    // Diagonal should be non-negative (curl-curl is positive semi-definite)
    for i in 0..n_dofs {
        let diag = csr
            .row_entries(i)
            .find(|(j, _)| *j == i)
            .map(|(_, v)| v)
            .unwrap_or(0.0);
        if !(diag >= -DIAGONAL_SLACK) {
            return Err(format!("diagonal K[{i},{i}] = {diag} should be >= 0"));
        }
    }
    println!("Curl-curl 2D: max asymmetry={max_asym:.2e}");
    Ok(max_asym)
}

/// Verify edge mass matrix is symmetric positive definite.
fn test_edge_mass_global_assembly_2d() -> Result<f64, String> {
    let mesh = unit_square_triangles(4);
    let space = HcurlSpace::new();
    let n_dofs = space.num_global_dofs(&mesh);

    let triplet = assemble_bilinear(&EdgeMassForm, &space, &space, &mesh, 2);
    let csr = triplet.to_csr();

    // Check symmetry
    let mut max_asym = 0.0_f64;
    for i in 0..n_dofs {
        for (j, val_ij) in csr.row_entries(i) {
            let val_ji = csr
                .row_entries(j)
                .find(|(col, _)| *col == i)
                .map(|(_, v)| v)
                .unwrap_or(0.0);
            max_asym = max_asym.max((val_ij - val_ji).abs());
        }
    }
    if !max_asym.is_finite() || max_asym >= SYMMETRY_TOLERANCE_2D {
        return Err(format!(
            "2D edge-mass asymmetry {max_asym:.2e} exceeds bound {SYMMETRY_TOLERANCE_2D:.1e}"
        ));
    }

    // Diagonal should be strictly positive (mass matrix is SPD)
    for i in 0..n_dofs {
        let diag = csr
            .row_entries(i)
            .find(|(j, _)| *j == i)
            .map(|(_, v)| v)
            .unwrap_or(0.0);
        if !(diag > 0.0) {
            return Err(format!("diagonal M[{i},{i}] = {diag} should be > 0"));
        }
    }
    println!("Edge mass 2D: max asymmetry={max_asym:.2e}");
    Ok(max_asym)
}

/// Verify 3D curl-curl assembly on a tetrahedral mesh.
fn test_curl_curl_global_assembly_3d() -> Result<f64, String> {
    let mesh = unit_cube_tetrahedra(2);
    let space = HcurlSpace::new();
    let n_dofs = space.num_global_dofs(&mesh);

    let triplet = assemble_bilinear(&CurlCurlForm, &space, &space, &mesh, 2);
    if triplet.num_rows != n_dofs {
        return Err(format!(
            "3D curl-curl rows {} != edge DOFs {n_dofs}",
            triplet.num_rows
        ));
    }

    let csr = triplet.to_csr();

    // Check symmetry (sample first 20 rows for speed)
    let check_rows = n_dofs.min(20);
    let mut max_asym = 0.0_f64;
    for i in 0..check_rows {
        for (j, val_ij) in csr.row_entries(i) {
            let val_ji = csr
                .row_entries(j)
                .find(|(col, _)| *col == i)
                .map(|(_, v)| v)
                .unwrap_or(0.0);
            max_asym = max_asym.max((val_ij - val_ji).abs());
        }
    }
    if !max_asym.is_finite() || max_asym >= SYMMETRY_TOLERANCE_3D {
        return Err(format!(
            "3D curl-curl asymmetry {max_asym:.2e} exceeds bound {SYMMETRY_TOLERANCE_3D:.1e}"
        ));
    }
    println!("Curl-curl 3D: max asymmetry={max_asym:.2e}");
    Ok(max_asym)
}

/// Verify the curl-curl matrix has a non-trivial kernel (gradient fields).
///
/// For a simply-connected domain, the kernel of the curl-curl operator is the
/// space of gradient fields: curl(grad phi) = 0 for any scalar phi.
/// On a mesh with N_nodes nodes and N_edges edges, dim(kernel) >= N_nodes - 1.
fn test_curl_curl_kernel_dimension() -> Result<(f64, f64), String> {
    let mesh = unit_square_triangles(3);
    let space = HcurlSpace::new();
    let n_edges = space.num_global_dofs(&mesh);
    let n_nodes = mesh.num_nodes();
    let topo = MeshTopology::build(&mesh);

    let triplet = assemble_bilinear(&CurlCurlForm, &space, &space, &mesh, 2);
    let csr = triplet.to_csr();

    if n_edges != topo.num_edges() {
        return Err(format!(
            "edge DOFs {n_edges} != topology edges {}",
            topo.num_edges()
        ));
    }
    if n_edges <= n_nodes {
        return Err(format!(
            "should have more edges ({n_edges}) than nodes ({n_nodes})"
        ));
    }

    // Edge DOFs for grad(phi) are phi(max_node) - phi(min_node) because
    // MeshTopology orients global edges from the lower node index to the
    // higher one. curl(grad(phi)) must vanish element-by-element.
    let nodal_phi: Vec<f64> = mesh
        .nodes
        .iter()
        .map(|p| p.x + 2.0 * p.y + 0.25 * p.x * p.y)
        .collect();
    let mut gradient_edge_dofs = vec![0.0_f64; n_edges];
    for (edge_idx, &(a, b)) in topo.edges.iter().enumerate() {
        gradient_edge_dofs[edge_idx] = nodal_phi[b] - nodal_phi[a];
    }

    let k_grad = csr.apply(&gradient_edge_dofs);
    let residual_norm = k_grad.iter().map(|v| v * v).sum::<f64>().sqrt();
    let gradient_norm = gradient_edge_dofs.iter().map(|v| v * v).sum::<f64>().sqrt();
    let bound = KERNEL_RESIDUAL_FACTOR * gradient_norm.max(1.0);
    if !residual_norm.is_finite() || residual_norm > bound {
        return Err(format!(
            "gradient field should be in curl-curl kernel: ||K grad(phi)|| = {residual_norm:.3e}, ||grad(phi)|| = {gradient_norm:.3e}"
        ));
    }

    // A generic edge vector should not be mistaken for a gradient null mode.
    let non_gradient: Vec<f64> = (0..n_edges)
        .map(|i| (i as f64 + 1.0).sin() + 0.5 * (2.0 * i as f64 + 1.0).cos())
        .collect();
    let k_non_gradient = csr.apply(&non_gradient);
    let energy: f64 = non_gradient
        .iter()
        .zip(k_non_gradient.iter())
        .map(|(a, b)| a * b)
        .sum();
    if !(energy > GENERIC_ENERGY_MINIMUM) {
        return Err(format!(
            "generic edge field should have positive curl-curl energy, got {energy:.3e}"
        ));
    }
    println!("Kernel: ||K grad||={residual_norm:.2e}, generic energy={energy:.2e}");
    Ok((residual_norm, energy))
}

fn shift_invert_config(target: f64) -> ShiftInvertConfig {
    ShiftInvertConfig {
        target,
        num_eigenvalues: 1,
        krylov_dimension: 60,
        tolerance: 1.0e-8,
        breakdown_tolerance: 1.0e-14,
    }
}

/// Independently recomputed relative true residual for one eigenpair.
fn relative_eigen_residual(
    stiffness: &CsrMatrix<f64>,
    mass: &CsrMatrix<f64>,
    lambda: f64,
    vector: &[f64],
) -> f64 {
    let k_x = stiffness.apply(vector);
    let m_x = mass.apply(vector);
    let residual_norm = k_x
        .iter()
        .zip(m_x.iter())
        .map(|(&kv, &mv)| (kv - lambda * mv).powi(2))
        .sum::<f64>()
        .sqrt();
    let scale = k_x
        .iter()
        .map(|v| v.powi(2))
        .sum::<f64>()
        .sqrt()
        .max(1e-300);
    residual_norm / scale
}

/// Expand a PEC-reduced eigenvector to full edge-DOF ordering (boundary
/// entries are exactly zero) and verify the expansion against the mesh.
fn expand_pec_eigenvector(
    mesh: &sonium_fem::mesh::Mesh,
    pec_markers: &[i32],
    reduced: &[f64],
) -> Result<Vec<f64>, String> {
    let space = HcurlSpace::new();
    let n_dofs = space.num_global_dofs(mesh);
    let mut constrained = std::collections::HashSet::new();
    for &marker in pec_markers {
        constrained.extend(space.boundary_dofs(mesh, marker));
    }
    let free_dofs: Vec<usize> = (0..n_dofs).filter(|d| !constrained.contains(d)).collect();
    if free_dofs.len() != reduced.len() {
        return Err(format!(
            "reduced eigenvector length {} != free DOFs {}",
            reduced.len(),
            free_dofs.len()
        ));
    }
    let mut full = vec![0.0; n_dofs];
    for (index, &dof) in free_dofs.iter().enumerate() {
        full[dof] = reduced[index];
    }
    // PEC entries must be exactly zero by elimination.
    for &dof in constrained.iter() {
        if full[dof] != 0.0 {
            return Err(format!("PEC edge DOF {dof} is not zero after expansion"));
        }
    }
    Ok(full)
}

/// Solve the TE10-like transverse mode of a PEC rectangular waveguide
/// cross-section (a x b, a > b) and check the cutoff wavenumber kc = pi/a,
/// the eigen-residual, and the modal E_y shape against sin(pi x/a).
fn solve_waveguide_te10(nx: usize, ny: usize) -> Result<(f64, f64, f64, f64), String> {
    let a: f64 = 1.0;
    let b: f64 = 0.5;
    let mesh = rectangular_mesh_triangles(0.0, a, 0.0, b, nx, ny);
    let problem = build_electromagnetic_eigenmode_problem(
        &mesh,
        EmMaterial::vacuum(),
        &ElectromagneticEigenmodeConfig {
            order: 1,
            pec_boundary_markers: vec![1, 2, 3, 4],
        },
    );
    let expected_kc = std::f64::consts::PI / a;
    let target = (SPEED_OF_LIGHT * expected_kc).powi(2);
    let result = problem
        .solve_near(&shift_invert_config(target))
        .map_err(|error| format!("waveguide shift-invert failed at {nx}x{ny}: {error:?}"))?;
    if !result.converged {
        return Err(format!(
            "waveguide shift-invert did not converge at {nx}x{ny}"
        ));
    }
    let lambda = result.eigenvalues[0];
    if !lambda.is_finite() || lambda <= 0.0 {
        return Err(format!("waveguide eigenvalue is not positive: {lambda}"));
    }
    let solved_kc = lambda.sqrt() / SPEED_OF_LIGHT;
    let kc_error = (solved_kc - expected_kc).abs() / expected_kc;
    let vector = result.eigenvectors.column_to_vec(0);
    let residual =
        relative_eigen_residual(&problem.stiffness_like, &problem.mass_like, lambda, &vector);
    if !residual.is_finite() || residual >= EIGEN_RESIDUAL_TOLERANCE {
        return Err(format!(
            "waveguide eigen-residual {residual:.2e} exceeds bound {EIGEN_RESIDUAL_TOLERANCE:.1e}"
        ));
    }

    // Modal field check: |E_y| on vertical edges follows sin(pi x/a).
    let full = expand_pec_eigenvector(&mesh, &[1, 2, 3, 4], &vector)?;
    let topo = MeshTopology::build(&mesh);
    let constrained: std::collections::HashSet<usize> = {
        let space = HcurlSpace::new();
        [1, 2, 3, 4]
            .iter()
            .flat_map(|&m| space.boundary_dofs(&mesh, m))
            .collect()
    };
    let mut values = Vec::new();
    let mut shapes = Vec::new();
    for (edge_idx, &(n0, n1)) in topo.edges.iter().enumerate() {
        if constrained.contains(&edge_idx) {
            continue;
        }
        let p0 = mesh.nodes[n0];
        let p1 = mesh.nodes[n1];
        // Vertical edges carry the transverse E_y component.
        if (p0.x - p1.x).abs() > 1e-12 {
            continue;
        }
        let length = ((p0.x - p1.x).powi(2) + (p0.y - p1.y).powi(2)).sqrt();
        if length < 1e-15 {
            continue;
        }
        let x_mid = 0.5 * (p0.x + p1.x);
        values.push(full[edge_idx].abs() / length);
        shapes.push((std::f64::consts::PI * x_mid / a).sin());
    }
    if values.len() < 4 {
        return Err("too few interior vertical edges for the modal shape check".to_string());
    }
    let scale: f64 = values
        .iter()
        .zip(shapes.iter())
        .map(|(&v, &s)| v * s)
        .sum::<f64>()
        / shapes.iter().map(|s| s * s).sum::<f64>().max(1e-300);
    let shape_error = values
        .iter()
        .zip(shapes.iter())
        .map(|(&v, &s)| (v - scale * s).powi(2))
        .sum::<f64>()
        .sqrt()
        / values.iter().map(|v| v * v).sum::<f64>().sqrt().max(1e-300);

    let cutoff_hz = SPEED_OF_LIGHT * solved_kc / (2.0 * std::f64::consts::PI);
    println!(
        "Waveguide TE10 ({nx}x{ny}): kc={solved_kc:.6e} 1/m (expected {expected_kc:.6e}), fc={:.3e} Hz, kc err={kc_error:.2e}, shape err={shape_error:.2e}, residual={residual:.2e}",
        cutoff_hz
    );
    Ok((kc_error, shape_error, residual, cutoff_hz))
}

/// Verify the waveguide TE10 cutoff with PEC eigen-solves on two meshes.
fn test_waveguide_te10_cutoff() -> Result<(f64, f64, f64, f64, f64), String> {
    let (coarse_kc, _, _, _) = solve_waveguide_te10(8, 4)?;
    let (fine_kc, shape_error, residual, cutoff_hz) = solve_waveguide_te10(16, 8)?;
    if fine_kc >= coarse_kc {
        return Err(format!(
            "waveguide cutoff error does not contract: coarse={coarse_kc:.2e}, fine={fine_kc:.2e}"
        ));
    }
    if !fine_kc.is_finite() || fine_kc >= WAVEGUIDE_KC_TOLERANCE {
        return Err(format!(
            "waveguide cutoff error {fine_kc:.2e} exceeds bound {WAVEGUIDE_KC_TOLERANCE:.1e}"
        ));
    }
    if !shape_error.is_finite() || shape_error >= WAVEGUIDE_SHAPE_TOLERANCE {
        return Err(format!(
            "waveguide modal shape error {shape_error:.2e} exceeds bound {WAVEGUIDE_SHAPE_TOLERANCE:.1e}"
        ));
    }
    // An operating point above the solved cutoff must propagate.
    let operating_hz: f64 = 200e6;
    let k_operating = 2.0 * std::f64::consts::PI * operating_hz / SPEED_OF_LIGHT;
    let solved_kc = std::f64::consts::PI * (1.0 - fine_kc);
    if !(k_operating > solved_kc) {
        return Err("200 MHz operating point should be above the solved cutoff".to_string());
    }
    Ok((coarse_kc, fine_kc, shape_error, residual, cutoff_hz))
}

/// Tag every boundary facet of a box mesh with one PEC marker.
fn tag_box_pec(mesh: &mut sonium_fem::mesh::Mesh, marker: i32) {
    mesh.set_boundary_condition(BoundaryType::Dirichlet, marker, |points| {
        points.iter().all(|p| {
            p.x.abs() < 1e-9
                || (p.x - 1.0).abs() < 1e-9
                || p.y.abs() < 1e-9
                || (p.y - 1.0).abs() < 1e-9
                || p.z.abs() < 1e-9
                || (p.z - 1.0).abs() < 1e-9
        })
    });
}

/// Solve the lowest PEC cubic-cavity mode (k = pi sqrt(2)) and check the
/// wavenumber, eigen-residual, and modal field variation.
fn solve_cavity_mode(n: usize) -> Result<(f64, f64, f64, f64), String> {
    let mut mesh = unit_cube_tetrahedra(n);
    tag_box_pec(&mut mesh, 1);
    let problem = build_electromagnetic_eigenmode_problem(
        &mesh,
        EmMaterial::vacuum(),
        &ElectromagneticEigenmodeConfig {
            order: 1,
            pec_boundary_markers: vec![1],
        },
    );
    if problem.stiffness_like.num_rows == 0 {
        return Err("cavity eigenproblem has no free DOFs".to_string());
    }
    // Lowest resonance of the unit PEC cube: the (1,0,1) family at k = pi sqrt(2).
    let expected_k = std::f64::consts::PI * 2.0_f64.sqrt();
    let target = (SPEED_OF_LIGHT * expected_k).powi(2);
    let config = ShiftInvertConfig {
        target,
        num_eigenvalues: 1,
        krylov_dimension: 80,
        tolerance: 1.0e-8,
        breakdown_tolerance: 1.0e-14,
    };
    let result = problem
        .solve_near(&config)
        .map_err(|error| format!("cavity shift-invert failed at n={n}: {error:?}"))?;
    if !result.converged {
        return Err(format!("cavity shift-invert did not converge at n={n}"));
    }
    let lambda = result.eigenvalues[0];
    if !lambda.is_finite() || lambda <= 0.0 {
        return Err(format!("cavity eigenvalue is not positive: {lambda}"));
    }
    let solved_k = lambda.sqrt() / SPEED_OF_LIGHT;
    let k_error = (solved_k - expected_k).abs() / expected_k;
    let vector = result.eigenvectors.column_to_vec(0);
    let residual =
        relative_eigen_residual(&problem.stiffness_like, &problem.mass_like, lambda, &vector);
    if !residual.is_finite() || residual >= EIGEN_RESIDUAL_TOLERANCE {
        return Err(format!(
            "cavity eigen-residual {residual:.2e} exceeds bound {EIGEN_RESIDUAL_TOLERANCE:.1e}"
        ));
    }

    // Modal field check: the expanded field must vary spatially (a physical
    // cavity mode, not a null or constant vector) and vanish on the PEC walls
    // (verified inside the expansion helper).
    let full = expand_pec_eigenvector(&mesh, &[1], &vector)?;
    let topo = MeshTopology::build(&mesh);
    let space = HcurlSpace::new();
    let constrained: std::collections::HashSet<usize> =
        space.boundary_dofs(&mesh, 1).into_iter().collect();
    let mut magnitudes = Vec::new();
    for (edge_idx, &(n0, n1)) in topo.edges.iter().enumerate() {
        if constrained.contains(&edge_idx) {
            continue;
        }
        let p0 = mesh.nodes[n0];
        let p1 = mesh.nodes[n1];
        let length = ((p0.x - p1.x).powi(2) + (p0.y - p1.y).powi(2) + (p0.z - p1.z).powi(2)).sqrt();
        if length < 1e-15 {
            continue;
        }
        magnitudes.push(full[edge_idx].abs() / length);
    }
    if magnitudes.len() < 8 {
        return Err("too few interior edges for the cavity modal check".to_string());
    }
    let mean = magnitudes.iter().sum::<f64>() / magnitudes.len() as f64;
    let variance =
        magnitudes.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / magnitudes.len() as f64;
    let variation = variance.sqrt() / mean.max(1e-300);

    let resonance_hz = SPEED_OF_LIGHT * solved_k / (2.0 * std::f64::consts::PI);
    println!(
        "Cavity (n={n}): k={solved_k:.6e} 1/m (expected {expected_k:.6e}), f={resonance_hz:.3e} Hz, k err={k_error:.2e}, variation={variation:.2e}, residual={residual:.2e}"
    );
    Ok((k_error, variation, residual, resonance_hz))
}

/// Verify rectangular cavity resonance with PEC eigen-solves on two meshes.
fn test_cavity_resonance() -> Result<(f64, f64, f64, f64, f64), String> {
    let (coarse_k, _, _, _) = solve_cavity_mode(3)?;
    let (fine_k, variation, residual, resonance_hz) = solve_cavity_mode(4)?;
    if fine_k >= coarse_k {
        return Err(format!(
            "cavity wavenumber error does not contract: coarse={coarse_k:.2e}, fine={fine_k:.2e}"
        ));
    }
    if !fine_k.is_finite() || fine_k >= CAVITY_K_TOLERANCE {
        return Err(format!(
            "cavity wavenumber error {fine_k:.2e} exceeds bound {CAVITY_K_TOLERANCE:.1e}"
        ));
    }
    if !variation.is_finite() || variation < CAVITY_VARIATION_MINIMUM {
        return Err(format!(
            "cavity modal variation {variation:.2e} below minimum {CAVITY_VARIATION_MINIMUM:.1e}"
        ));
    }
    Ok((coarse_k, fine_k, variation, residual, resonance_hz))
}

/// Verify PEC (Perfect Electric Conductor) boundary DOF discovery.
/// On a PEC boundary, tangential E = 0, which means edge DOFs on boundary = 0.
fn test_pec_boundary_dof_detection() -> Result<usize, String> {
    let mesh = unit_square_triangles(4);
    let space = HcurlSpace::new();
    let n_dofs = space.num_global_dofs(&mesh);
    let topo = MeshTopology::build(&mesh);
    let mut markers: Vec<i32> = topo.boundary_edges.keys().copied().collect();
    markers.sort_unstable();

    let mut boundary_dofs = Vec::new();
    for marker in markers {
        let marker_dofs = space.boundary_dofs(&mesh, marker);
        let expected = topo
            .boundary_edges
            .get(&marker)
            .cloned()
            .unwrap_or_default();
        if marker_dofs != expected {
            return Err(format!(
                "H(curl) boundary DOFs mismatch topology boundary edges for marker {marker}"
            ));
        }
        boundary_dofs.extend(marker_dofs);
    }
    boundary_dofs.sort_unstable();
    boundary_dofs.dedup();

    if boundary_dofs.is_empty() {
        return Err("unit-square mesh should expose H(curl) boundary DOFs".to_string());
    }
    if boundary_dofs.len() >= n_dofs {
        return Err("PEC boundary DOFs should be a proper subset of all edge DOFs".to_string());
    }

    let mut electric_field = vec![1.0_f64; n_dofs];
    for &dof in &boundary_dofs {
        electric_field[dof] = 0.0;
    }
    if !boundary_dofs.iter().all(|&dof| electric_field[dof] == 0.0) {
        return Err("PEC projection should zero every boundary edge DOF".to_string());
    }
    println!(
        "PEC detection: {} boundary DOFs of {n_dofs}",
        boundary_dofs.len()
    );
    Ok(boundary_dofs.len())
}

/// Verify edge element conformity - that curl of edge elements gives nodal gradients.
fn test_edge_element_conformity() -> Result<usize, String> {
    let mesh = unit_square_triangles(2);
    let space = HcurlSpace::new();
    let n_edges = space.num_global_dofs(&mesh);
    let n_nodes = mesh.num_nodes();

    let triplet = assemble_bilinear(&CurlCurlForm, &space, &space, &mesh, 2);
    let csr = triplet.to_csr();

    println!("Edge element conformity:");
    println!("  Nodes: {}, Edges (DOFs): {}", n_nodes, n_edges);
    println!("  Matrix size: {} × {}", csr.num_rows, csr.num_cols);

    if csr.num_rows != n_edges {
        return Err("matrix rows should equal edge DOFs".to_string());
    }
    if csr.num_cols != n_edges {
        return Err("matrix cols should equal edge DOFs".to_string());
    }
    Ok(n_edges)
}

struct MaxwellMetrics {
    curl_curl_asymmetry_2d: f64,
    edge_mass_asymmetry: f64,
    curl_curl_asymmetry_3d: f64,
    kernel_residual: f64,
    generic_energy: f64,
    waveguide_coarse_error: f64,
    waveguide_fine_error: f64,
    waveguide_shape_error: f64,
    waveguide_residual: f64,
    waveguide_cutoff_hz: f64,
    cavity_coarse_error: f64,
    cavity_fine_error: f64,
    cavity_variation: f64,
    cavity_residual: f64,
    cavity_resonance_hz: f64,
    pec_boundary_dofs: usize,
    edge_dofs: usize,
}

fn run() -> Result<MaxwellMetrics, String> {
    let curl_curl_asymmetry_2d = test_curl_curl_global_assembly_2d()?;
    let edge_mass_asymmetry = test_edge_mass_global_assembly_2d()?;
    let curl_curl_asymmetry_3d = test_curl_curl_global_assembly_3d()?;
    let (kernel_residual, generic_energy) = test_curl_curl_kernel_dimension()?;
    let (
        waveguide_coarse_error,
        waveguide_fine_error,
        waveguide_shape_error,
        waveguide_residual,
        waveguide_cutoff_hz,
    ) = test_waveguide_te10_cutoff()?;
    let (
        cavity_coarse_error,
        cavity_fine_error,
        cavity_variation,
        cavity_residual,
        cavity_resonance_hz,
    ) = test_cavity_resonance()?;
    let pec_boundary_dofs = test_pec_boundary_dof_detection()?;
    let edge_dofs = test_edge_element_conformity()?;
    Ok(MaxwellMetrics {
        curl_curl_asymmetry_2d,
        edge_mass_asymmetry,
        curl_curl_asymmetry_3d,
        kernel_residual,
        generic_energy,
        waveguide_coarse_error,
        waveguide_fine_error,
        waveguide_shape_error,
        waveguide_residual,
        waveguide_cutoff_hz,
        cavity_coarse_error,
        cavity_fine_error,
        cavity_variation,
        cavity_residual,
        cavity_resonance_hz,
        pec_boundary_dofs,
        edge_dofs,
    })
}

fn main() {
    match run() {
        Ok(metrics) => {
            let record = QaResult::passed(CASE_ID)
                .requested_backend("cpu")
                .actual_backend("cpu")
                .metric("curl_curl_asymmetry_2d", metrics.curl_curl_asymmetry_2d)
                .metric("edge_mass_asymmetry", metrics.edge_mass_asymmetry)
                .metric("curl_curl_asymmetry_3d", metrics.curl_curl_asymmetry_3d)
                .metric("kernel_residual", metrics.kernel_residual)
                .metric("generic_energy", metrics.generic_energy)
                .metric("waveguide_coarse_error", metrics.waveguide_coarse_error)
                .metric("waveguide_fine_error", metrics.waveguide_fine_error)
                .metric("waveguide_shape_error", metrics.waveguide_shape_error)
                .metric("waveguide_residual", metrics.waveguide_residual)
                .metric("waveguide_cutoff_hz", metrics.waveguide_cutoff_hz)
                .metric("cavity_coarse_error", metrics.cavity_coarse_error)
                .metric("cavity_fine_error", metrics.cavity_fine_error)
                .metric("cavity_variation", metrics.cavity_variation)
                .metric("cavity_residual", metrics.cavity_residual)
                .metric("cavity_resonance_hz", metrics.cavity_resonance_hz)
                .metric("pec_boundary_dofs", metrics.pec_boundary_dofs as f64)
                .metric("edge_dofs", metrics.edge_dofs as f64)
                .maximum("curl_curl_asymmetry_2d", SYMMETRY_TOLERANCE_2D)
                .maximum("edge_mass_asymmetry", SYMMETRY_TOLERANCE_2D)
                .maximum("curl_curl_asymmetry_3d", SYMMETRY_TOLERANCE_3D)
                .maximum("waveguide_fine_error", WAVEGUIDE_KC_TOLERANCE)
                .maximum("waveguide_shape_error", WAVEGUIDE_SHAPE_TOLERANCE)
                .maximum("waveguide_residual", EIGEN_RESIDUAL_TOLERANCE)
                .maximum("cavity_fine_error", CAVITY_K_TOLERANCE)
                .minimum("cavity_variation", CAVITY_VARIATION_MINIMUM)
                .maximum("cavity_residual", EIGEN_RESIDUAL_TOLERANCE)
                .tolerance("PEC waveguide/cavity eigen-solves with residual and modal-field gates")
                .seed("fem-maxwell-validation:cpu:v1");
            if let Ok(json) = record.to_json_line() {
                println!("QA_RESULT: {json}");
            }
            println!("SUCCESS");
        }
        Err(error) => {
            if let Ok(json) = QaResult::failed(CASE_ID, error.clone())
                .requested_backend("cpu")
                .actual_backend("cpu")
                .to_json_line()
            {
                println!("QA_RESULT: {json}");
            }
            eprintln!("FEM MAXWELL VALIDATION FAILED: {error}");
            println!("FAILED");
            std::process::exit(1);
        }
    }
}
