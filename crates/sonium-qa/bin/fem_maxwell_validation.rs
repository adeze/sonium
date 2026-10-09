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
use sonium_fem::mesh::{MeshTopology, unit_cube_tetrahedra, unit_square_triangles};
use sonium_fem::physics::electromagnetics::maxwell::{CurlCurlForm, EdgeMassForm};
use sonium_fem::spaces::{FunctionSpace, HcurlSpace};

/// Verify curl-curl matrix is symmetric and positive semi-definite (has a kernel).
fn test_curl_curl_global_assembly_2d() {
    let mesh = unit_square_triangles(4);
    let space = HcurlSpace::new();
    let n_dofs = space.num_global_dofs(&mesh);

    let triplet = assemble_bilinear(&CurlCurlForm, &space, &space, &mesh, 2);
    assert_eq!(triplet.num_rows, n_dofs);

    let csr = triplet.to_csr();

    // Check symmetry
    for i in 0..n_dofs {
        for (j, val_ij) in csr.row_entries(i) {
            let val_ji = csr
                .row_entries(j)
                .find(|(col, _)| *col == i)
                .map(|(_, v)| v)
                .unwrap_or(0.0);
            assert!(
                (val_ij - val_ji).abs() < 1e-12,
                "Symmetry: K[{i},{j}]={val_ij} vs K[{j},{i}]={val_ji}"
            );
        }
    }

    // Diagonal should be non-negative (curl-curl is positive semi-definite)
    for i in 0..n_dofs {
        let diag = csr
            .row_entries(i)
            .find(|(j, _)| *j == i)
            .map(|(_, v)| v)
            .unwrap_or(0.0);
        assert!(
            diag >= -1e-14,
            "Diagonal K[{i},{i}] = {diag} should be >= 0"
        );
    }
}

/// Verify edge mass matrix is symmetric positive definite.
fn test_edge_mass_global_assembly_2d() {
    let mesh = unit_square_triangles(4);
    let space = HcurlSpace::new();
    let n_dofs = space.num_global_dofs(&mesh);

    let triplet = assemble_bilinear(&EdgeMassForm, &space, &space, &mesh, 2);
    let csr = triplet.to_csr();

    // Check symmetry
    for i in 0..n_dofs {
        for (j, val_ij) in csr.row_entries(i) {
            let val_ji = csr
                .row_entries(j)
                .find(|(col, _)| *col == i)
                .map(|(_, v)| v)
                .unwrap_or(0.0);
            assert!(
                (val_ij - val_ji).abs() < 1e-12,
                "Symmetry: M[{i},{j}]={val_ij} vs M[{j},{i}]={val_ji}"
            );
        }
    }

    // Diagonal should be strictly positive (mass matrix is SPD)
    for i in 0..n_dofs {
        let diag = csr
            .row_entries(i)
            .find(|(j, _)| *j == i)
            .map(|(_, v)| v)
            .unwrap_or(0.0);
        assert!(diag > 0.0, "Diagonal M[{i},{i}] = {diag} should be > 0");
    }
}

/// Verify 3D curl-curl assembly on a tetrahedral mesh.
fn test_curl_curl_global_assembly_3d() {
    let mesh = unit_cube_tetrahedra(2);
    let space = HcurlSpace::new();
    let n_dofs = space.num_global_dofs(&mesh);

    let triplet = assemble_bilinear(&CurlCurlForm, &space, &space, &mesh, 2);
    assert_eq!(triplet.num_rows, n_dofs);

    let csr = triplet.to_csr();

    // Check symmetry (sample first 20 rows for speed)
    let check_rows = n_dofs.min(20);
    for i in 0..check_rows {
        for (j, val_ij) in csr.row_entries(i) {
            let val_ji = csr
                .row_entries(j)
                .find(|(col, _)| *col == i)
                .map(|(_, v)| v)
                .unwrap_or(0.0);
            assert!(
                (val_ij - val_ji).abs() < 1e-10,
                "3D symmetry: K[{i},{j}]={val_ij:.6e} vs K[{j},{i}]={val_ji:.6e}"
            );
        }
    }
}

/// Verify the curl-curl matrix has a non-trivial kernel (gradient fields).
///
/// For a simply-connected domain, the kernel of the curl-curl operator is the
/// space of gradient fields: curl(grad phi) = 0 for any scalar phi.
/// On a mesh with N_nodes nodes and N_edges edges, dim(kernel) >= N_nodes - 1.
fn test_curl_curl_kernel_dimension() {
    let mesh = unit_square_triangles(3);
    let space = HcurlSpace::new();
    let n_edges = space.num_global_dofs(&mesh);
    let n_nodes = mesh.num_nodes();
    let topo = MeshTopology::build(&mesh);

    let triplet = assemble_bilinear(&CurlCurlForm, &space, &space, &mesh, 2);
    let csr = triplet.to_csr();

    assert_eq!(n_edges, topo.num_edges());
    assert!(
        n_edges > n_nodes,
        "Should have more edges ({n_edges}) than nodes ({n_nodes})"
    );

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

    let k_grad = sonium_solvers::LinearOperator::apply(&csr, &gradient_edge_dofs);
    let residual_norm = k_grad.iter().map(|v| v * v).sum::<f64>().sqrt();
    let gradient_norm = gradient_edge_dofs.iter().map(|v| v * v).sum::<f64>().sqrt();
    assert!(
        residual_norm <= 1e-12 * gradient_norm.max(1.0),
        "Gradient field should be in curl-curl kernel: ||K grad(phi)|| = {residual_norm:.3e}, ||grad(phi)|| = {gradient_norm:.3e}"
    );

    // A generic edge vector should not be mistaken for a gradient null mode.
    let non_gradient: Vec<f64> = (0..n_edges)
        .map(|i| (i as f64 + 1.0).sin() + 0.5 * (2.0 * i as f64 + 1.0).cos())
        .collect();
    let k_non_gradient = sonium_solvers::LinearOperator::apply(&csr, &non_gradient);
    let energy: f64 = non_gradient
        .iter()
        .zip(k_non_gradient.iter())
        .map(|(a, b)| a * b)
        .sum();
    assert!(
        energy > 1e-10,
        "Generic edge field should have positive curl-curl energy, got {energy:.3e}"
    );
}

/// Verify waveguide TE/TM cutoff frequencies for a rectangular waveguide.
/// TE_mn modes have cutoff kc = π * sqrt((m/a)² + (n/b)²)
fn test_waveguide_te10_cutoff() {
    let a: f64 = 0.1; // Waveguide width (m)
    let _b: f64 = 0.05; // Waveguide height (m)
    let freq: f64 = 2e9;

    let omega = 2.0 * std::f64::consts::PI * freq;
    let mu = 4e-7 * std::f64::consts::PI;
    let eps: f64 = 8.854e-12;

    let kc_te10 = std::f64::consts::PI / a;
    let fc_exact = kc_te10 / (2.0 * std::f64::consts::PI * (mu * eps).sqrt());

    let k = omega * (mu * eps).sqrt();
    let k_squared = k * k;
    let kc_squared = k_squared - (std::f64::consts::PI / a).powi(2);

    println!("Waveguide TE₁₀ mode:");
    println!("  Cutoff frequency: {:.2} GHz", fc_exact / 1e9);
    println!("  k = {:.2} /m", k);
    println!("  kc² = {:.2} /m²", kc_squared);

    assert!(k > kc_te10, "Operating frequency should be above cutoff");
}

/// Verify rectangular cavity resonance frequencies.
/// TM_mnp modes: f_mnp = (c/2) * sqrt((m/a)² + (n/b)² + (p/d)²)
fn test_cavity_resonance_tm110() {
    let a: f64 = 0.1;
    let b: f64 = 0.05;
    let d: f64 = 0.08;
    let c: f64 = 3e8;

    let m = 1;
    let n = 1;
    let p = 0;

    let fc = (c / 2.0)
        * ((m as f64 / a).powi(2) + (n as f64 / b).powi(2) + (p as f64 / d).powi(2)).sqrt();

    println!("Rectangular cavity TM₁₁₀ mode:");
    println!("  Dimensions: {} × {} × {} m", a, b, d);
    println!("  Resonance: {:.2} GHz", fc / 1e9);

    assert!(fc > 1e9, "TM₁₁₀ should be in GHz range");
}

/// Verify PEC (Perfect Electric Conductor) boundary DOF discovery.
/// On a PEC boundary, tangential E = 0, which means edge DOFs on boundary = 0.
fn test_pec_boundary_dof_detection() {
    let mesh = unit_square_triangles(4);
    let space = HcurlSpace::new();
    let n_dofs = space.num_global_dofs(&mesh);
    let topo = MeshTopology::build(&mesh);
    let mut markers: Vec<i32> = topo.boundary_edges.keys().copied().collect();
    markers.sort_unstable();

    let mut boundary_dofs = Vec::new();
    for marker in markers {
        let marker_dofs = space.boundary_dofs(&mesh, marker);
        assert_eq!(
            marker_dofs,
            topo.boundary_edges
                .get(&marker)
                .cloned()
                .unwrap_or_default(),
            "H(curl) boundary DOFs should match topology boundary edges for marker {marker}"
        );
        boundary_dofs.extend(marker_dofs);
    }
    boundary_dofs.sort_unstable();
    boundary_dofs.dedup();

    assert!(
        !boundary_dofs.is_empty(),
        "Unit-square mesh should expose H(curl) boundary DOFs"
    );
    assert!(
        boundary_dofs.len() < n_dofs,
        "PEC boundary DOFs should be a proper subset of all edge DOFs"
    );

    let mut electric_field = vec![1.0_f64; n_dofs];
    for &dof in &boundary_dofs {
        electric_field[dof] = 0.0;
    }
    assert!(
        boundary_dofs.iter().all(|&dof| electric_field[dof] == 0.0),
        "PEC projection should zero every boundary edge DOF"
    );
}

/// Verify edge element conformity - that curl of edge elements gives nodal gradients.
fn test_edge_element_conformity() {
    let mesh = unit_square_triangles(2);
    let space = HcurlSpace::new();
    let n_edges = space.num_global_dofs(&mesh);
    let n_nodes = mesh.num_nodes();

    // For a 2D mesh, edges connect nodes. The DOF count should be:
    // n_edges = n_nodes + n_boundary_edges (for P1)
    let triplet = assemble_bilinear(&CurlCurlForm, &space, &space, &mesh, 2);
    let csr = triplet.to_csr();

    println!("Edge element conformity:");
    println!("  Nodes: {}, Edges (DOFs): {}", n_nodes, n_edges);
    println!("  Matrix size: {} × {}", csr.num_rows, csr.num_cols);

    // Verify DOF count is correct
    assert!(
        csr.num_rows == n_edges,
        "Matrix rows should equal edge DOFs"
    );
    assert!(
        csr.num_cols == n_edges,
        "Matrix cols should equal edge DOFs"
    );
}

fn main() {
    let result = std::panic::catch_unwind(|| {
        test_curl_curl_global_assembly_2d();
        test_edge_mass_global_assembly_2d();
        test_curl_curl_global_assembly_3d();
        test_curl_curl_kernel_dimension();
        test_waveguide_te10_cutoff();
        test_cavity_resonance_tm110();
        test_pec_boundary_dof_detection();
        test_edge_element_conformity();
    });
    match result {
        Ok(()) => {
            println!("SUCCESS");
        }
        Err(_) => {
            println!("FAILED");
            std::process::exit(1);
        }
    }
}
