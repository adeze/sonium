//! Validation tests comparing FEM solutions against analytical solutions
//!
//! These tests verify that the FEM solver produces correct results by comparing
//! against known analytical solutions from the math-wave crate.
//!
//! Every leg below solves a genuine discrete system: thin-strip cases keep
//! interior rows of free P1 unknowns (a strip with one element across its width
//! would prescribe every node and prove nothing), error norms cover interior
//! nodes and element quadrature points rather than prescribed values alone,
//! and the cylinder far-field leg measures the decay of the numerical FEM
//! scattered field, not of the analytical series.
#![allow(deprecated)]
#![allow(non_snake_case)]

use num_complex::Complex64;
use sonium_fem::assembly::HelmholtzProblem;
use sonium_fem::basis::{Jacobian, PolynomialDegree, evaluate_shape};
use sonium_fem::boundary::{DirichletBC, apply_dirichlet};
use sonium_fem::io::vtk::VtuWriter;
use sonium_fem::mesh::{
    BoundaryType, ElementType, annular_mesh_triangles, box_mesh_tetrahedra,
    rectangular_mesh_triangles, spherical_shell_mesh_tetrahedra, unit_square_triangles,
};
use sonium_fem::quadrature::QuadratureRule;
use sonium_qa::{QaResult, convergence_rate_with_sizes};
use sonium_solvers::{CsrMatrix, GmresConfig, IluPreconditioner, gmres, gmres_preconditioned};
use sonium_wave::analytical::{legendre_p, spherical_bessel_j, spherical_bessel_y};
use sonium_xem::io::binary_output_dir;
use std::collections::HashSet;
use std::f64::consts::PI;

// Named acceptance tolerances for the thin-strip legs. Each strip leg gates
// the interior-node error, and the worst element-quadrature error across all
// strip legs is gated jointly so no leg can hide behind prescribed nodes.
const STRIP_1D_MODE_INTERIOR_TOL: f64 = 1.0e-5;
const STRIP_PLANE_WAVE_INTERIOR_TOL: f64 = 1.0e-5;
const STRIP_HELMHOLTZ_MODE_INTERIOR_TOL: f64 = 1.0e-4;
const STRIP_DAMPED_WAVE_INTERIOR_TOL: f64 = 1.0e-6;
const STRIP_MAX_QUADRATURE_TOL: f64 = 1.0e-2;
// The numerical far-field leg measures the 1/sqrt(r) decay of the solved FEM
// scattered field between consecutive probe radii.
const FAR_FIELD_DECAY_MISMATCH_TOL: f64 = 0.3;

fn write_complex_vtu(name: &str, mesh: &sonium_fem::mesh::Mesh, p: &[Complex64]) {
    let dir = binary_output_dir(env!("CARGO_PKG_NAME"), env!("CARGO_BIN_NAME"));
    let mut w = VtuWriter::new(mesh);
    w.add_point_scalar_complex("pressure", p);
    let _ = w.write(dir.join(format!("{name}.vtu")));
}

/// Convert HelmholtzProblem matrix to CsrMatrix for solver
fn to_csr_matrix(problem: &HelmholtzProblem) -> CsrMatrix<Complex64> {
    let compressed = problem.matrix.to_compressed();
    let n = compressed.dim;

    // Convert triplet format to CSR using from_triplets
    let triplets: Vec<(usize, usize, Complex64)> = (0..compressed.nnz())
        .map(|i| (compressed.rows[i], compressed.cols[i], compressed.values[i]))
        .collect();

    CsrMatrix::from_triplets(n, n, triplets)
}

/// Compute L2 error between FEM solution and analytical solution
fn l2_error<F>(mesh: &sonium_fem::mesh::Mesh, fem_solution: &[Complex64], analytical: F) -> f64
where
    F: Fn(f64, f64, f64) -> Complex64,
{
    let mut error_sq = 0.0;
    let mut norm_sq = 0.0;

    for (i, node) in mesh.nodes.iter().enumerate() {
        let fem_val = fem_solution[i];
        let exact_val = analytical(node.x, node.y, node.z);
        let diff = fem_val - exact_val;
        error_sq += diff.norm_sqr();
        norm_sq += exact_val.norm_sqr();
    }

    if norm_sq > 1e-15 {
        (error_sq / norm_sq).sqrt()
    } else {
        error_sq.sqrt()
    }
}

/// Nodes prescribed by Dirichlet data on the given boundary markers.
fn prescribed_nodes(mesh: &sonium_fem::mesh::Mesh, markers: &[i32]) -> HashSet<usize> {
    let mut prescribed = HashSet::new();
    for boundary in &mesh.boundaries {
        if markers.contains(&boundary.marker) {
            prescribed.extend(boundary.nodes.iter().copied());
        }
    }
    prescribed
}

/// Assert that a Dirichlet problem keeps at least one free unknown.
///
/// A case that prescribes every P1 node (for example a strip with one element
/// across its width and exact data on all four sides) returns the prescribed
/// values verbatim and cannot validate the solver.
fn assert_nontrivial_solve(label: &str, mesh: &sonium_fem::mesh::Mesh, markers: &[i32]) -> usize {
    let free = mesh.num_nodes() - prescribed_nodes(mesh, markers).len();
    assert!(
        free > 0,
        "{label}: every P1 node is prescribed; no free unknowns are solved"
    );
    println!(
        "{label}: {free} free unknowns of {} nodes",
        mesh.num_nodes()
    );
    free
}

/// Relative nodal L2 error restricted to free (non-prescribed) nodes.
fn interior_l2_error<F>(
    mesh: &sonium_fem::mesh::Mesh,
    fem_solution: &[Complex64],
    prescribed: &HashSet<usize>,
    analytical: F,
) -> f64
where
    F: Fn(f64, f64, f64) -> Complex64,
{
    let mut error_sq = 0.0;
    let mut norm_sq = 0.0;
    let mut count = 0usize;

    for (i, node) in mesh.nodes.iter().enumerate() {
        if prescribed.contains(&i) {
            continue;
        }
        let diff = fem_solution[i] - analytical(node.x, node.y, node.z);
        error_sq += diff.norm_sqr();
        norm_sq += analytical(node.x, node.y, node.z).norm_sqr();
        count += 1;
    }

    assert!(count > 0, "interior error needs at least one free node");
    if norm_sq > 1e-15 {
        (error_sq / norm_sq).sqrt()
    } else {
        error_sq.sqrt()
    }
}

/// Relative element-quadrature L2 error of the P1 field over the whole domain.
///
/// Unlike the nodal norm this samples the finite element solution inside the
/// elements through the production shape functions and quadrature rules.
fn quadrature_l2_error<F>(
    mesh: &sonium_fem::mesh::Mesh,
    fem_solution: &[Complex64],
    analytical: F,
) -> f64
where
    F: Fn(f64, f64, f64) -> Complex64,
{
    let mut error_sq = 0.0;
    let mut norm_sq = 0.0;

    for element in &mesh.elements {
        let vertex_count = match element.element_type {
            ElementType::Triangle => 3,
            ElementType::Tetrahedron => 4,
            other => panic!("quadrature L2 norm supports triangles and tetrahedra, got {other:?}"),
        };
        let nodes = &element.nodes[..vertex_count];
        let rule = QuadratureRule::new(element.element_type, 4);

        for point in rule.iter() {
            let shape = evaluate_shape(
                element.element_type,
                PolynomialDegree::P1,
                point.xi(),
                point.eta(),
                point.zeta(),
            );
            let mut x = 0.0;
            let mut y = 0.0;
            let mut z = 0.0;
            let mut discrete = Complex64::new(0.0, 0.0);
            for (i, &node) in nodes.iter().enumerate() {
                let basis = shape.values[i];
                let vertex = &mesh.nodes[node];
                x += basis * vertex.x;
                y += basis * vertex.y;
                z += basis * vertex.z;
                discrete += fem_solution[node] * basis;
            }
            let det = match element.element_type {
                ElementType::Triangle => {
                    let coords: Vec<[f64; 2]> = nodes
                        .iter()
                        .map(|&node| [mesh.nodes[node].x, mesh.nodes[node].y])
                        .collect();
                    Jacobian::from_2d(&shape.gradients, shape.grad_dim, &coords)
                        .det
                        .abs()
                }
                _ => {
                    let coords: Vec<[f64; 3]> = nodes
                        .iter()
                        .map(|&node| [mesh.nodes[node].x, mesh.nodes[node].y, mesh.nodes[node].z])
                        .collect();
                    Jacobian::from_3d(&shape.gradients, shape.grad_dim, &coords)
                        .det
                        .abs()
                }
            };
            let weight = point.weight * det;
            let exact = analytical(x, y, z);
            error_sq += weight * (discrete - exact).norm_sqr();
            norm_sq += weight * exact.norm_sqr();
        }
    }

    if norm_sq > 1e-15 {
        (error_sq / norm_sq).sqrt()
    } else {
        error_sq.sqrt()
    }
}

/// Metrics reported by one thin-strip leg for the structured result.
struct StripLegMetrics {
    interior_error: f64,
    quadrature_error: f64,
    free_dofs: usize,
}

fn assert_helmholtz_l2_convergence(
    label: &str,
    errors: &[f64],
    h_values: &[f64],
    min_pair_rate: f64,
    min_mean_rate: f64,
) -> f64 {
    assert_eq!(
        errors.len(),
        h_values.len(),
        "{label}: errors and h-values length mismatch"
    );
    assert!(
        errors.len() >= 2,
        "{label}: at least two refinements are required"
    );

    for i in 1..errors.len() {
        assert!(
            errors[i] < errors[i - 1],
            "{label}: error should decrease with refinement: h_prev={:.4e}, err_prev={:.4e}; h={:.4e}, err={:.4e}",
            h_values[i - 1],
            errors[i - 1],
            h_values[i],
            errors[i],
        );

        let pair_rate = convergence_rate_with_sizes(&errors[i - 1..=i], &h_values[i - 1..=i])
            .expect("valid adjacent convergence rate");
        assert!(
            pair_rate >= min_pair_rate,
            "{label}: adjacent L2 convergence rate {:.3} below {:.3} between h={:.4e} and h={:.4e}; errors={:.4e}->{:.4e}",
            pair_rate,
            min_pair_rate,
            h_values[i - 1],
            h_values[i],
            errors[i - 1],
            errors[i],
        );
    }

    let mean_rate = convergence_rate_with_sizes(errors, h_values)
        .expect("valid mean Helmholtz FEM convergence rate");
    assert!(
        mean_rate >= min_mean_rate,
        "{label}: mean L2 convergence rate {:.3} below {:.3}; h={h_values:?}, errors={errors:?}",
        mean_rate,
        min_mean_rate,
    );

    println!(
        "{label}: Helmholtz FEM P1 L2 convergence mean rate {:.3}, finest error {:.4e}",
        mean_rate,
        errors.last().copied().unwrap_or_default()
    );

    mean_rate
}

/// Test 1D Helmholtz mode in x-direction using manufactured solution
///
/// Uses a source term to drive the solution to sin(kx). The strip keeps two
/// elements across its width so the middle row stays a free unknown; error
/// norms cover those interior nodes and element quadrature points.
fn test_1d_mode_in_2d_strip() -> StripLegMetrics {
    // Create a thin strip mesh (effectively 1D)
    // Domain: [0, 1] x [0, 0.05]
    let l = 1.0;
    let k = 1.5; // Use a wavenumber that doesn't match the domain

    // Use a thin strip to approximate 1D - finer mesh for accuracy
    let mesh = rectangular_mesh_triangles(0.0, l, 0.0, 0.05, 40, 2);

    // The solution u(x,y) = sin(πx) satisfies -∇²u - k²u = f where:
    // f = (π² - k²) * sin(πx)
    let k_pi = PI;
    let coef = k_pi * k_pi - k * k;
    let source = move |x: f64, _y: f64, _z: f64| Complex64::new(coef * (k_pi * x).sin(), 0.0);
    let exact_u = move |x: f64, _y: f64, _z: f64| Complex64::new((k_pi * x).sin(), 0.0);

    let k_complex = Complex64::new(k, 0.0);
    let mut problem = HelmholtzProblem::assemble(&mesh, PolynomialDegree::P1, k_complex, source);

    // Apply Dirichlet BCs: u = 0 at x=0 (tag 1) and x=L (tag 2)
    // u = exact_u on top/bottom (tags 3, 4)
    let bc_left = DirichletBC::new(1, |_, _, _| Complex64::new(0.0, 0.0));
    let bc_right = DirichletBC::new(2, |_, _, _| Complex64::new(0.0, 0.0));
    let bc_top = DirichletBC::new(4, exact_u);
    let bc_bottom = DirichletBC::new(3, exact_u);

    apply_dirichlet(&mut problem, &mesh, &[bc_left, bc_right, bc_top, bc_bottom]);

    // Solve the system
    let matrix = to_csr_matrix(&problem);
    let rhs = problem.rhs.clone();

    let config = GmresConfig {
        max_iterations: 500,
        restart: 50,
        tolerance: 1e-10,
        print_interval: 0,
        convergence_check_interval: 1,
        min_restart: None,
        max_restart: None,
        exact_residual_check_interval: 0,
    };

    let solution = gmres(&matrix, &rhs, &config);
    assert!(solution.converged, "GMRES should converge");

    write_complex_vtu("1d_mode_2d_strip", &mesh, solution.x.as_slice());

    let markers = [1, 2, 3, 4];
    let free_dofs = assert_nontrivial_solve("1-D mode in 2-D strip", &mesh, &markers);

    // Compare with analytical solution u(x,y) = sin(πx)
    let error = l2_error(&mesh, &solution.x, exact_u);
    let prescribed = prescribed_nodes(&mesh, &markers);
    let interior = interior_l2_error(&mesh, &solution.x, &prescribed, exact_u);
    let quadrature = quadrature_l2_error(&mesh, &solution.x, exact_u);
    println!(
        "1-D mode in 2-D strip: L2 error {error:.4e}, interior {interior:.4e}, quadrature {quadrature:.4e}"
    );

    // P1 elements on fine mesh should achieve good accuracy
    assert!(
        error < 1.0e-5,
        "L2 error {} should be < 1e-5 for P1 elements",
        error
    );
    assert!(
        interior < STRIP_1D_MODE_INTERIOR_TOL,
        "interior L2 error {interior} should be < {STRIP_1D_MODE_INTERIOR_TOL} for P1 elements",
    );

    StripLegMetrics {
        interior_error: interior,
        quadrature_error: quadrature,
        free_dofs,
    }
}

/// Test 2D plane wave solution
///
/// Solves the Helmholtz equation with plane wave boundary conditions
/// u(x,y) = exp(i(k_x * x + k_y * y))
fn test_2d_plane_wave() {
    let k = 2.0; // Wavenumber
    let theta = PI / 4.0; // 45 degree angle
    let kx = k * theta.cos();
    let ky = k * theta.sin();

    // Finer mesh for better accuracy
    let mesh = unit_square_triangles(16);

    // The plane wave satisfies Helmholtz with zero source
    let k_complex = Complex64::new(k, 0.0);
    let mut problem =
        HelmholtzProblem::assemble(&mesh, PolynomialDegree::P1, k_complex, |_, _, _| {
            Complex64::new(0.0, 0.0)
        });

    // Apply plane wave as Dirichlet BC on all boundaries
    let plane_wave = move |x: f64, y: f64, _z: f64| {
        let phase = kx * x + ky * y;
        Complex64::new(phase.cos(), phase.sin())
    };

    let bcs: Vec<DirichletBC> = (1..=4)
        .map(|tag| DirichletBC::new(tag, plane_wave))
        .collect();

    apply_dirichlet(&mut problem, &mesh, &bcs);

    // Solve
    let matrix = to_csr_matrix(&problem);
    let rhs = problem.rhs.clone();

    let config = GmresConfig {
        max_iterations: 500,
        restart: 50,
        tolerance: 1e-10,
        print_interval: 0,
        convergence_check_interval: 1,
        min_restart: None,
        max_restart: None,
        exact_residual_check_interval: 0,
    };

    let solution = gmres(&matrix, &rhs, &config);
    assert!(solution.converged, "GMRES should converge");

    write_complex_vtu("2d_plane_wave", &mesh, solution.x.as_slice());

    // Compare with analytical plane wave
    let error = l2_error(&mesh, &solution.x, plane_wave);
    println!("2-D plane wave: L2 error {error:.4e}");

    // Should be very accurate since plane wave is an exact solution
    assert!(
        error < 1.0e-3,
        "L2 error {} should be < 1e-3 for plane wave",
        error
    );
}

/// Test convergence rate with mesh refinement
///
/// For P1 elements, we expect O(h²) convergence in L2 norm
fn test_convergence_rate() {
    let k = 1.0;
    let kx = k * 0.6; // Not aligned with grid
    let ky = k * 0.8;

    let analytical = move |x: f64, y: f64, _z: f64| {
        let phase = kx * x + ky * y;
        Complex64::new(phase.cos(), phase.sin())
    };

    let mesh_sizes = [4, 8, 16];
    let mut errors = Vec::new();
    let mut h_values = Vec::new();

    for &n in &mesh_sizes {
        let mesh = unit_square_triangles(n);
        let h = 1.0 / (n as f64); // Mesh size

        let k_complex = Complex64::new(k, 0.0);
        let mut problem =
            HelmholtzProblem::assemble(&mesh, PolynomialDegree::P1, k_complex, |_, _, _| {
                Complex64::new(0.0, 0.0)
            });

        // Apply plane wave BCs on all boundaries
        let bcs: Vec<DirichletBC> = (1..=4)
            .map(|tag| DirichletBC::new(tag, analytical))
            .collect();

        apply_dirichlet(&mut problem, &mesh, &bcs);

        let matrix = to_csr_matrix(&problem);
        let rhs = problem.rhs.clone();

        let config = GmresConfig {
            max_iterations: 1000,
            restart: 100,
            tolerance: 1e-12,
            print_interval: 0,
            convergence_check_interval: 1,
            min_restart: None,
            max_restart: None,
            exact_residual_check_interval: 0,
        };

        let solution = gmres(&matrix, &rhs, &config);
        assert!(
            solution.converged,
            "GMRES should converge for mesh size {}",
            n
        );

        let error = l2_error(&mesh, &solution.x, analytical);
        errors.push(error);
        h_values.push(h);
    }

    assert_helmholtz_l2_convergence("plane-wave Dirichlet", &errors, &h_values, 1.5, 1.7);
}

/// Test standing wave solution (2D analog of 1D mode)
///
/// u(x,y) = sin(n*π*x/L_x) * sin(m*π*y/L_y)
/// is an eigenfunction of the Laplacian with eigenvalue k² = (nπ/L_x)² + (mπ/L_y)²
fn test_2d_standing_wave() {
    let lx = 1.0;
    let ly = 1.0;
    let n = 1;
    let m = 1;

    let kx = (n as f64) * PI / lx;
    let ky = (m as f64) * PI / ly;
    let k = (kx * kx + ky * ky).sqrt();

    let mesh = unit_square_triangles(10);

    let k_complex = Complex64::new(k, 0.0);
    let mut problem =
        HelmholtzProblem::assemble(&mesh, PolynomialDegree::P1, k_complex, |_, _, _| {
            Complex64::new(0.0, 0.0)
        });

    // Homogeneous Dirichlet BCs (standing wave is zero on all boundaries)
    let bcs: Vec<DirichletBC> = (1..=4)
        .map(|tag| DirichletBC::new(tag, |_, _, _| Complex64::new(0.0, 0.0)))
        .collect();

    apply_dirichlet(&mut problem, &mesh, &bcs);

    let matrix = to_csr_matrix(&problem);
    let rhs = problem.rhs.clone();

    let config = GmresConfig {
        max_iterations: 500,
        restart: 50,
        tolerance: 1e-10,
        print_interval: 0,
        convergence_check_interval: 1,
        min_restart: None,
        max_restart: None,
        exact_residual_check_interval: 0,
    };

    let solution = gmres(&matrix, &rhs, &config);

    // For the eigenvalue problem with homogeneous BCs and zero source,
    // the solution should be zero (trivial solution)
    // This verifies the system is correctly assembled
    let max_val: f64 = solution.x.iter().map(|v| v.norm()).fold(0.0, f64::max);
    assert!(
        max_val < 1e-8,
        "Solution should be trivial for homogeneous problem"
    );
}

/// Test with non-trivial source term
///
/// Uses manufactured solution: choose u, compute f = -∆u - k²u
fn test_manufactured_solution() {
    let k = 1.5;

    // Choose a smooth solution that's zero on boundaries
    // u(x,y) = sin(πx) * sin(πy)
    let exact_u = |x: f64, y: f64, _z: f64| Complex64::new((PI * x).sin() * (PI * y).sin(), 0.0);

    // For u = sin(πx)*sin(πy):
    // ∆u = -2π²*sin(πx)*sin(πy)
    // f = -∆u - k²u = (2π² - k²)*sin(πx)*sin(πy)
    let coef = 2.0 * PI * PI - k * k;
    let source =
        move |x: f64, y: f64, _z: f64| Complex64::new(coef * (PI * x).sin() * (PI * y).sin(), 0.0);

    let mesh = unit_square_triangles(12);
    let k_complex = Complex64::new(k, 0.0);

    let mut problem = HelmholtzProblem::assemble(&mesh, PolynomialDegree::P1, k_complex, source);

    // Homogeneous Dirichlet BCs (our exact solution is zero on boundary)
    let bcs: Vec<DirichletBC> = (1..=4)
        .map(|tag| DirichletBC::new(tag, |_, _, _| Complex64::new(0.0, 0.0)))
        .collect();

    apply_dirichlet(&mut problem, &mesh, &bcs);

    let matrix = to_csr_matrix(&problem);
    let rhs = problem.rhs.clone();

    let config = GmresConfig {
        max_iterations: 500,
        restart: 50,
        tolerance: 1e-10,
        print_interval: 0,
        convergence_check_interval: 1,
        min_restart: None,
        max_restart: None,
        exact_residual_check_interval: 0,
    };

    let solution = gmres(&matrix, &rhs, &config);
    assert!(solution.converged, "GMRES should converge");

    // Compute error against manufactured solution
    let error = l2_error(&mesh, &solution.x, exact_u);
    println!("2-D standing wave: L2 error {error:.4e}");

    // Should achieve reasonable accuracy
    assert!(
        error < 0.01,
        "L2 error {} should be < 0.01 for manufactured solution",
        error
    );
}

/// Test using math-wave plane_wave_1d function to validate against a known solution
///
/// This test uses the analytical solution from math-wave and compares with FEM.
/// The strip uses four elements across its width so interior rows stay free
/// unknowns; a single element across would prescribe every P1 node.
fn test_with_math_wave_plane_wave_1d() -> StripLegMetrics {
    use sonium_wave::analytical::plane_wave_1d;

    let k = 2.0;
    let x_min = 0.0;
    let x_max = 1.0;

    // Get analytical solution from math-wave
    let analytical_solution = plane_wave_1d(k, x_min, x_max, 50);

    // Create a thin 2D strip to simulate 1D
    let mesh = rectangular_mesh_triangles(x_min, x_max, 0.0, 0.05, 40, 4);

    let k_complex = Complex64::new(k, 0.0);
    let mut problem =
        HelmholtzProblem::assemble(&mesh, PolynomialDegree::P1, k_complex, |_, _, _| {
            Complex64::new(0.0, 0.0)
        });

    // Use the plane wave as Dirichlet BC
    // The plane wave solution is exp(ikx) = cos(kx) + i*sin(kx)
    let plane_wave_bc = move |x: f64, _y: f64, _z: f64| {
        let kx = k * x;
        Complex64::new(kx.cos(), kx.sin())
    };

    let bcs: Vec<DirichletBC> = (1..=4)
        .map(|tag| DirichletBC::new(tag, plane_wave_bc))
        .collect();

    apply_dirichlet(&mut problem, &mesh, &bcs);

    let matrix = to_csr_matrix(&problem);
    let rhs = problem.rhs.clone();

    let config = GmresConfig {
        max_iterations: 500,
        restart: 50,
        tolerance: 1e-10,
        print_interval: 0,
        convergence_check_interval: 1,
        min_restart: None,
        max_restart: None,
        exact_residual_check_interval: 0,
    };

    let solution = gmres(&matrix, &rhs, &config);
    assert!(solution.converged, "GMRES should converge");

    let markers = [1, 2, 3, 4];
    let free_dofs = assert_nontrivial_solve("sonium-wave plane wave", &mesh, &markers);

    // Verify that the FEM solution matches the analytical plane wave
    let error = l2_error(&mesh, &solution.x, plane_wave_bc);
    let prescribed = prescribed_nodes(&mesh, &markers);
    let interior = interior_l2_error(&mesh, &solution.x, &prescribed, plane_wave_bc);
    let quadrature = quadrature_l2_error(&mesh, &solution.x, plane_wave_bc);
    println!(
        "sonium-wave plane wave: L2 error {error:.4e}, interior {interior:.4e}, quadrature {quadrature:.4e}"
    );

    // Also verify that math-wave's solution has the expected properties
    // At x=0: exp(0) = 1
    assert!(
        (analytical_solution.pressure[0].re - 1.0).abs() < 1e-10,
        "Analytical solution at x=0 should be 1"
    );

    // The FEM solution is genuinely solved on interior unknowns, so gate the
    // interior discretization error rather than a vacuous all-node norm.
    assert!(
        interior < STRIP_PLANE_WAVE_INTERIOR_TOL,
        "interior L2 error {interior} should be < {STRIP_PLANE_WAVE_INTERIOR_TOL} for plane wave",
    );

    StripLegMetrics {
        interior_error: interior,
        quadrature_error: quadrature,
        free_dofs,
    }
}

/// Test using math-wave helmholtz_1d_mode function
///
/// The FEM convention is -Δu - k²u = f, so with the source f(x) = sin(nπx/L)
/// the exact mode is u(x) = -sin(nπx/L) / (k² - (nπ/L)²). (The math-wave
/// helper documents the u'' + k²u = +f convention, which carries the opposite
/// sign; the endpoint cross-check below holds either way.)
fn test_with_math_wave_helmholtz_mode() -> StripLegMetrics {
    use sonium_wave::analytical::helmholtz_1d_mode;

    let k = 2.0; // Not at resonance
    let l = 1.0;
    let mode_n = 2;

    // Get analytical solution from math-wave
    let analytical_solution = helmholtz_1d_mode(k, l, mode_n, 50);

    // Thin 2D strip with interior rows of free unknowns
    let mesh = rectangular_mesh_triangles(0.0, l, 0.0, 0.05, 40, 4);

    // The source term for this mode: f(x) = sin(nπx/L)
    let kn = (mode_n as f64) * PI / l;
    let source = move |x: f64, _y: f64, _z: f64| Complex64::new((kn * x).sin(), 0.0);

    let k_complex = Complex64::new(k, 0.0);
    let mut problem = HelmholtzProblem::assemble(&mesh, PolynomialDegree::P1, k_complex, source);

    // The Helmholtz mode has u(0) = u(L) = 0
    // Use analytical values on top/bottom boundaries
    let denom = k * k - kn * kn;
    let exact_u = move |x: f64, _y: f64, _z: f64| Complex64::new(-(kn * x).sin() / denom, 0.0);

    let bc_left = DirichletBC::new(1, |_, _, _| Complex64::new(0.0, 0.0));
    let bc_right = DirichletBC::new(2, |_, _, _| Complex64::new(0.0, 0.0));
    let bc_top = DirichletBC::new(4, exact_u);
    let bc_bottom = DirichletBC::new(3, exact_u);

    apply_dirichlet(&mut problem, &mesh, &[bc_left, bc_right, bc_top, bc_bottom]);

    let matrix = to_csr_matrix(&problem);
    let rhs = problem.rhs.clone();

    let config = GmresConfig {
        max_iterations: 500,
        restart: 50,
        tolerance: 1e-10,
        print_interval: 0,
        convergence_check_interval: 1,
        min_restart: None,
        max_restart: None,
        exact_residual_check_interval: 0,
    };

    let solution = gmres(&matrix, &rhs, &config);
    assert!(solution.converged, "GMRES should converge");

    let markers = [1, 2, 3, 4];
    let free_dofs = assert_nontrivial_solve("sonium-wave Helmholtz mode", &mesh, &markers);

    // Verify the FEM solution matches the analytical
    let error = l2_error(&mesh, &solution.x, exact_u);
    let prescribed = prescribed_nodes(&mesh, &markers);
    let interior = interior_l2_error(&mesh, &solution.x, &prescribed, exact_u);
    let quadrature = quadrature_l2_error(&mesh, &solution.x, exact_u);
    println!(
        "sonium-wave Helmholtz mode: L2 error {error:.4e}, interior {interior:.4e}, quadrature {quadrature:.4e}"
    );

    // Also verify math-wave's solution properties
    // At x=0: sin(0) = 0, so u(0) = 0
    assert!(
        analytical_solution.pressure[0].norm() < 1e-10,
        "Analytical solution at x=0 should be 0"
    );

    assert!(
        interior < STRIP_HELMHOLTZ_MODE_INTERIOR_TOL,
        "interior L2 error {interior} should be < {STRIP_HELMHOLTZ_MODE_INTERIOR_TOL} for helmholtz mode",
    );

    StripLegMetrics {
        interior_error: interior,
        quadrature_error: quadrature,
        free_dofs,
    }
}

/// Test damped wave solution from math-wave
fn test_with_math_wave_damped_wave() -> StripLegMetrics {
    use sonium_wave::analytical::damped_wave_1d;

    let k = 1.0;
    let alpha = 0.5; // Damping coefficient

    // Get analytical solution from math-wave
    let analytical_solution = damped_wave_1d(k, alpha, 0.0, 1.0, 50);

    // The damped wave is: exp(-(α + ik)x) = exp(-αx) * exp(ikx)
    // This solves the Helmholtz equation with complex wavenumber k_complex = k + i*α
    let k_complex = Complex64::new(k, alpha);

    // Thin strip with interior rows of free unknowns
    let mesh = rectangular_mesh_triangles(0.0, 1.0, 0.0, 0.05, 40, 4);

    let mut problem =
        HelmholtzProblem::assemble(&mesh, PolynomialDegree::P1, k_complex, |_, _, _| {
            Complex64::new(0.0, 0.0)
        });

    // The damped wave solution as BC
    let damped_wave = move |x: f64, _y: f64, _z: f64| {
        let damping = (-alpha * x).exp();
        let phase = k * x;
        Complex64::new(damping * phase.cos(), damping * phase.sin())
    };

    let bcs: Vec<DirichletBC> = (1..=4)
        .map(|tag| DirichletBC::new(tag, damped_wave))
        .collect();

    apply_dirichlet(&mut problem, &mesh, &bcs);

    let matrix = to_csr_matrix(&problem);
    let rhs = problem.rhs.clone();

    let config = GmresConfig {
        max_iterations: 500,
        restart: 50,
        tolerance: 1e-10,
        print_interval: 0,
        convergence_check_interval: 1,
        min_restart: None,
        max_restart: None,
        exact_residual_check_interval: 0,
    };

    let solution = gmres(&matrix, &rhs, &config);
    assert!(solution.converged, "GMRES should converge");

    let markers = [1, 2, 3, 4];
    let free_dofs = assert_nontrivial_solve("sonium-wave damped wave", &mesh, &markers);

    let error = l2_error(&mesh, &solution.x, damped_wave);
    let prescribed = prescribed_nodes(&mesh, &markers);
    let interior = interior_l2_error(&mesh, &solution.x, &prescribed, damped_wave);
    let quadrature = quadrature_l2_error(&mesh, &solution.x, damped_wave);
    println!(
        "sonium-wave damped wave: L2 error {error:.4e}, interior {interior:.4e}, quadrature {quadrature:.4e}"
    );

    // Verify math-wave solution
    // At x=0: exp(0) = 1
    assert!(
        (analytical_solution.pressure[0].re - 1.0).abs() < 1e-10,
        "Damped wave at x=0 should be 1"
    );

    assert!(
        interior < STRIP_DAMPED_WAVE_INTERIOR_TOL,
        "interior L2 error {interior} should be < {STRIP_DAMPED_WAVE_INTERIOR_TOL} for damped wave",
    );

    StripLegMetrics {
        interior_error: interior,
        quadrature_error: quadrature,
        free_dofs,
    }
}

// ============================================================================
// Method of Manufactured Solutions (MMS) Tests
// ============================================================================

/// Helper: solve Helmholtz problem with MMS and return L2 error
fn solve_mms_problem<U, F>(mesh: &sonium_fem::mesh::Mesh, k: f64, exact_u: U, source: F) -> f64
where
    U: Fn(f64, f64, f64) -> Complex64 + Copy + 'static,
    F: Fn(f64, f64, f64) -> Complex64 + Sync,
{
    assert_nontrivial_solve("MMS problem", mesh, &[1, 2, 3, 4]);
    let k_complex = Complex64::new(k, 0.0);
    let mut problem = HelmholtzProblem::assemble(mesh, PolynomialDegree::P1, k_complex, source);

    // Apply exact solution as Dirichlet BC on all boundaries
    let bcs: Vec<DirichletBC> = (1..=4).map(|tag| DirichletBC::new(tag, exact_u)).collect();

    apply_dirichlet(&mut problem, mesh, &bcs);

    let matrix = to_csr_matrix(&problem);
    let rhs = problem.rhs.clone();

    let config = GmresConfig {
        max_iterations: 1000,
        restart: 100,
        tolerance: 1e-12,
        print_interval: 0,
        convergence_check_interval: 1,
        min_restart: None,
        max_restart: None,
        exact_residual_check_interval: 0,
    };

    let solution = gmres(&matrix, &rhs, &config);
    assert!(solution.converged, "GMRES should converge");

    l2_error(mesh, &solution.x, exact_u)
}

/// MMS Test 1: u(x,y) = sin(πx) * cos(πy)
///
/// Laplacian: ∆u = -π²sin(πx)cos(πy) - π²sin(πx)cos(πy) = -2π²u
/// Helmholtz: -∆u - k²u = f  =>  f = (2π² - k²) * sin(πx) * cos(πy)
fn test_mms_sin_cos() {
    let k = 1.5;

    // Manufactured solution: u = sin(πx) * cos(πy)
    let exact_u = |x: f64, y: f64, _z: f64| Complex64::new((PI * x).sin() * (PI * y).cos(), 0.0);

    // Source term: f = (2π² - k²) * sin(πx) * cos(πy)
    let coef = 2.0 * PI * PI - k * k;
    let source =
        move |x: f64, y: f64, _z: f64| Complex64::new(coef * (PI * x).sin() * (PI * y).cos(), 0.0);

    // Test on a single mesh first
    let mesh = unit_square_triangles(16);
    let error = solve_mms_problem(&mesh, k, exact_u, source);

    assert!(
        error < 0.01,
        "MMS sin*cos: L2 error {} should be < 0.01",
        error
    );
}

/// MMS Test 2: u(x,y) = sin(πx) * cos(πy) with mesh convergence
///
/// Verifies O(h²) convergence for P1 elements
fn test_mms_sin_cos_convergence() {
    let k = 1.5;

    let exact_u = |x: f64, y: f64, _z: f64| Complex64::new((PI * x).sin() * (PI * y).cos(), 0.0);

    let coef = 2.0 * PI * PI - k * k;
    let source =
        move |x: f64, y: f64, _z: f64| Complex64::new(coef * (PI * x).sin() * (PI * y).cos(), 0.0);

    let mesh_sizes = [4, 8, 16, 32];
    let mut errors = Vec::new();
    let mut h_values = Vec::new();

    for &n in &mesh_sizes {
        let mesh = unit_square_triangles(n);
        let h = 1.0 / (n as f64);
        let error = solve_mms_problem(&mesh, k, exact_u, source);

        errors.push(error);
        h_values.push(h);
    }

    assert_helmholtz_l2_convergence("MMS sin*cos", &errors, &h_values, 1.5, 1.7);
}

/// MMS Test 3: u(x,y) = sin(2πx) * sin(2πy) (higher frequency)
///
/// Tests accuracy with higher frequency modes
fn test_mms_sin_sin_2pi() {
    let k = 2.0;

    // Higher frequency: u = sin(2πx) * sin(2πy)
    // ∆u = -4π²sin(2πx)sin(2πy) - 4π²sin(2πx)sin(2πy) = -8π²u
    let exact_u =
        |x: f64, y: f64, _z: f64| Complex64::new((2.0 * PI * x).sin() * (2.0 * PI * y).sin(), 0.0);

    // f = (8π² - k²) * sin(2πx) * sin(2πy)
    let coef = 8.0 * PI * PI - k * k;
    let source = move |x: f64, y: f64, _z: f64| {
        Complex64::new(coef * (2.0 * PI * x).sin() * (2.0 * PI * y).sin(), 0.0)
    };

    let mesh_sizes = [8, 16, 32];
    let mut errors = Vec::new();
    let mut h_values = Vec::new();

    for &n in &mesh_sizes {
        let mesh = unit_square_triangles(n);
        let h = 1.0 / (n as f64);
        let error = solve_mms_problem(&mesh, k, exact_u, source);

        errors.push(error);
        h_values.push(h);
    }

    assert_helmholtz_l2_convergence("MMS sin(2pi x)*sin(2pi y)", &errors, &h_values, 1.5, 1.7);
}

/// MMS Test 4: u(x,y) = x(1-x) * y(1-y) (polynomial, zero on boundary)
///
/// Simple polynomial that's exactly representable in the solution space
/// but requires proper source term computation
fn test_mms_polynomial() {
    let k = 1.0;

    // u = x(1-x) * y(1-y)
    // ∂²u/∂x² = -2 * y(1-y)
    // ∂²u/∂y² = -2 * x(1-x)
    // ∆u = -2[x(1-x) + y(1-y)]
    let exact_u = |x: f64, y: f64, _z: f64| Complex64::new(x * (1.0 - x) * y * (1.0 - y), 0.0);

    // f = -∆u - k²u = 2[x(1-x) + y(1-y)] - k² * x(1-x) * y(1-y)
    let source = move |x: f64, y: f64, _z: f64| {
        let u_val = x * (1.0 - x) * y * (1.0 - y);
        let laplacian = -2.0 * (x * (1.0 - x) + y * (1.0 - y));
        Complex64::new(-laplacian - k * k * u_val, 0.0)
    };

    let mesh_sizes = [4, 8, 16, 32];
    let mut errors = Vec::new();
    let mut h_values = Vec::new();

    for &n in &mesh_sizes {
        let mesh = unit_square_triangles(n);
        let h = 1.0 / (n as f64);
        let error = solve_mms_problem(&mesh, k, exact_u, source);

        errors.push(error);
        h_values.push(h);
    }

    assert_helmholtz_l2_convergence("MMS polynomial", &errors, &h_values, 1.5, 1.7);
}

/// MMS Test 5: u(x,y) = exp(x) * sin(πy) (exponential in x, periodic in y)
///
/// Mixed exponential-trigonometric solution
fn test_mms_exp_sin() {
    let k = 1.0;

    // u = exp(x) * sin(πy)
    // ∂²u/∂x² = exp(x) * sin(πy) = u
    // ∂²u/∂y² = -π² * exp(x) * sin(πy) = -π²u
    // ∆u = (1 - π²) * u
    let exact_u = |x: f64, y: f64, _z: f64| Complex64::new(x.exp() * (PI * y).sin(), 0.0);

    // f = -∆u - k²u = -(1 - π²)u - k²u = (π² - 1 - k²) * u
    let coef = PI * PI - 1.0 - k * k;
    let source =
        move |x: f64, y: f64, _z: f64| Complex64::new(coef * x.exp() * (PI * y).sin(), 0.0);

    let mesh_sizes = [4, 8, 16, 32];
    let mut errors = Vec::new();
    let mut h_values = Vec::new();

    for &n in &mesh_sizes {
        let mesh = unit_square_triangles(n);
        let h = 1.0 / (n as f64);
        let error = solve_mms_problem(&mesh, k, exact_u, source);

        errors.push(error);
        h_values.push(h);
    }

    assert_helmholtz_l2_convergence("MMS exp*sin", &errors, &h_values, 1.5, 1.7);
}

/// MMS Test 6: Complex-valued solution u(x,y) = (1 + i) * sin(πx) * sin(πy)
///
/// Tests that complex arithmetic is handled correctly
fn test_mms_complex_valued() {
    let k = 1.0;

    // Complex amplitude
    let amp = Complex64::new(1.0, 1.0);

    // u = (1+i) * sin(πx) * sin(πy)
    // ∆u = -2π² * (1+i) * sin(πx) * sin(πy) = -2π² * u
    let exact_u = move |x: f64, y: f64, _z: f64| amp * (PI * x).sin() * (PI * y).sin();

    // f = (2π² - k²) * (1+i) * sin(πx) * sin(πy)
    let coef = 2.0 * PI * PI - k * k;
    let source = move |x: f64, y: f64, _z: f64| amp * coef * (PI * x).sin() * (PI * y).sin();

    let mesh_sizes = [4, 8, 16, 32];
    let mut errors = Vec::new();
    let mut h_values = Vec::new();

    for &n in &mesh_sizes {
        let mesh = unit_square_triangles(n);
        let h = 1.0 / (n as f64);
        let error = solve_mms_problem(&mesh, k, exact_u, source);

        errors.push(error);
        h_values.push(h);
    }

    assert_helmholtz_l2_convergence("MMS complex", &errors, &h_values, 1.5, 1.7);
}

/// MMS Test 7: Verify exact convergence rates numerically
///
/// This test prints actual convergence rates for verification
fn test_mms_convergence_rates_detailed() {
    let k = 1.5;

    // u = sin(πx) * sin(πy) - classic choice with homogeneous BCs
    let exact_u = |x: f64, y: f64, _z: f64| Complex64::new((PI * x).sin() * (PI * y).sin(), 0.0);

    let coef = 2.0 * PI * PI - k * k;
    let source =
        move |x: f64, y: f64, _z: f64| Complex64::new(coef * (PI * x).sin() * (PI * y).sin(), 0.0);

    let mesh_sizes = [4, 8, 16, 32, 64];
    let mut errors = Vec::new();
    let mut h_values = Vec::new();

    for &n in &mesh_sizes {
        let mesh = unit_square_triangles(n);
        let h = 1.0 / (n as f64);
        let error = solve_mms_problem(&mesh, k, exact_u, source);

        errors.push(error);
        h_values.push(h);
    }

    let avg_rate =
        assert_helmholtz_l2_convergence("MMS detailed sin*sin", &errors, &h_values, 1.5, 1.9);
    assert!(
        avg_rate > 1.9 && avg_rate < 2.2,
        "Average convergence rate {} should be in [1.9, 2.2] (theoretical: 2.0)",
        avg_rate
    );

    // Finest mesh should have small error
    // With 64x64 mesh and h=1/64, O(h²) gives ~2.4e-4 theoretical error
    let finest_error = errors.last().unwrap();
    assert!(
        *finest_error < 5e-4,
        "Finest mesh error {} should be < 5e-4",
        finest_error
    );
}

/// MMS Test 8: Different wavenumbers to test dispersion behavior
fn test_mms_varying_wavenumber() {
    // Test with several different wavenumbers
    let wavenumbers = [0.5, 1.0, 2.0, 4.0];

    for &k in &wavenumbers {
        let exact_u =
            |x: f64, y: f64, _z: f64| Complex64::new((PI * x).sin() * (PI * y).sin(), 0.0);

        let coef = 2.0 * PI * PI - k * k;
        let source = move |x: f64, y: f64, _z: f64| {
            Complex64::new(coef * (PI * x).sin() * (PI * y).sin(), 0.0)
        };

        // Need finer mesh for higher wavenumbers (pollution effect)
        let n = if k > 2.0 { 32 } else { 16 };
        let mesh = unit_square_triangles(n);
        let error = solve_mms_problem(&mesh, k, exact_u, source);

        // Error tolerance scales with wavenumber (pollution effect)
        let tol = 0.01 * (1.0 + k * k / 10.0);
        assert!(
            error < tol,
            "MMS with k={}: error {} should be < {}",
            k,
            error,
            tol
        );
    }
}

/// MMS Test 9: Non-square domain (rectangle)
fn test_mms_rectangle() {
    let k = 1.0;

    // Domain [0, 2] x [0, 1]
    let lx = 2.0;
    let ly = 1.0;

    // u = sin(πx/lx) * sin(πy/ly)
    // ∆u = -(π/lx)²u - (π/ly)²u = -[(π/lx)² + (π/ly)²] * u
    let kx = PI / lx;
    let ky = PI / ly;

    let exact_u =
        move |x: f64, y: f64, _z: f64| Complex64::new((kx * x).sin() * (ky * y).sin(), 0.0);

    // f = [(π/lx)² + (π/ly)² - k²] * u
    let laplacian_coef = kx * kx + ky * ky;
    let coef = laplacian_coef - k * k;
    let source =
        move |x: f64, y: f64, _z: f64| Complex64::new(coef * (kx * x).sin() * (ky * y).sin(), 0.0);

    let mesh_sizes = [4, 8, 16, 32];
    let mut errors = Vec::new();
    let mut h_values = Vec::new();

    for &n in &mesh_sizes {
        // Rectangular mesh with aspect ratio 2:1
        let mesh = rectangular_mesh_triangles(0.0, lx, 0.0, ly, 2 * n, n);
        let h = ly / (n as f64); // Use smaller dimension for h
        let error = solve_mms_problem(&mesh, k, exact_u, source);

        errors.push(error);
        h_values.push(h);
    }

    assert_helmholtz_l2_convergence("MMS rectangle", &errors, &h_values, 1.5, 1.7);
}

// ============================================================================
// Scattering Problem: Circular Obstacle (2D)
// ============================================================================

/// Helper: compute the analytical scattered field for a sound-soft cylinder
///
/// For a sound-soft (Dirichlet) cylinder with u=0 on surface:
/// Total field = incident + scattered
/// u_total = exp(ikr cos θ) + Σ aₙ Hₙ(kr) cos(nθ)
///
/// Coefficients: aₙ = -εₙ iⁿ Jₙ(ka) / Hₙ(ka)
fn cylinder_scattering_analytical(
    k: f64,
    cylinder_radius: f64,
    x: f64,
    y: f64,
    num_terms: usize,
) -> Complex64 {
    use std::f64::consts::PI;

    let r = (x * x + y * y).sqrt();
    let theta = y.atan2(x);

    if r < cylinder_radius * 0.999 {
        // Inside cylinder - not physical
        return Complex64::new(0.0, 0.0);
    }

    let ka = k * cylinder_radius;
    let kr = k * r;

    // Incident plane wave: exp(ikx) = exp(ikr cos θ)
    let incident = Complex64::new((kr * theta.cos()).cos(), (kr * theta.cos()).sin());

    // Scattered field using Bessel/Hankel series
    let mut scattered = Complex64::new(0.0, 0.0);

    for n in 0..num_terms {
        let n_f64 = n as f64;
        let epsilon_n = if n == 0 { 1.0 } else { 2.0 };

        // Bessel functions at ka (cylinder surface) and kr (evaluation point)
        let jn_ka = bessel_j(n as i64, ka);
        let yn_ka = bessel_y(n as i64, ka);
        let jn_kr = bessel_j(n as i64, kr);
        let yn_kr = bessel_y(n as i64, kr);

        // Hankel function H_n = J_n + i*Y_n
        let hn_ka = Complex64::new(jn_ka, yn_ka);
        let hn_kr = Complex64::new(jn_kr, yn_kr);

        // i^n = exp(i*n*π/2)
        let i_power_n = Complex64::new((n_f64 * PI / 2.0).cos(), (n_f64 * PI / 2.0).sin());

        // Coefficient for sound-soft cylinder: a_n = -ε_n * i^n * J_n(ka) / H_n(ka)
        let a_n = -epsilon_n * jn_ka / hn_ka * i_power_n;

        // Contribution: a_n * H_n(kr) * cos(nθ)
        let cos_n_theta = (n_f64 * theta).cos();
        scattered += a_n * hn_kr * cos_n_theta;
    }

    incident + scattered
}

/// Bessel function of the first kind using spec_math crate
fn bessel_j(n: i64, x: f64) -> f64 {
    use spec_math::Bessel;
    x.bessel_jv(n as f64)
}

/// Bessel function of the second kind (Neumann function) using spec_math crate
fn bessel_y(n: i64, x: f64) -> f64 {
    use spec_math::Bessel;
    x.bessel_yv(n as f64)
}

/// Scattering by a sound-soft circular cylinder
///
/// Plane wave incident from +x direction scatters off a sound-soft (u=0)
/// cylinder. Uses analytical Bessel/Hankel series solution.
fn test_cylinder_scattering_sound_soft() {
    let cylinder_radius = 1.0;
    let outer_radius = 3.0;
    let k = 2.0; // ka = 2 (resonance regime)

    // Create annular mesh
    let n_radial = 48;
    let n_angular = 96;
    let mesh = annular_mesh_triangles(0.0, 0.0, cylinder_radius, outer_radius, n_radial, n_angular);

    let num_terms = 30; // Enough terms for ka=2

    // Analytical solution as the exact field
    let exact_u = move |x: f64, y: f64, _z: f64| {
        cylinder_scattering_analytical(k, cylinder_radius, x, y, num_terms)
    };

    // For Helmholtz equation in exterior domain: -∆u - k²u = 0 (no source)
    let source = |_x: f64, _y: f64, _z: f64| Complex64::new(0.0, 0.0);

    // Assemble the FEM problem
    let k_complex = Complex64::new(k, 0.0);
    let mut problem = HelmholtzProblem::assemble(&mesh, PolynomialDegree::P1, k_complex, source);

    // Apply Dirichlet BCs:
    // - Inner boundary (tag 1): u = 0 (sound-soft)
    // - Outer boundary (tag 2): u = exact analytical solution
    let bc_inner = DirichletBC::new(1, |_x, _y, _z| Complex64::new(0.0, 0.0));
    let bc_outer = DirichletBC::new(2, exact_u);

    apply_dirichlet(&mut problem, &mesh, &[bc_inner, bc_outer]);
    assert_nontrivial_solve("sound-soft cylinder", &mesh, &[1, 2]);

    // Solve
    let matrix = to_csr_matrix(&problem);
    let rhs = problem.rhs.clone();

    let config = GmresConfig {
        max_iterations: 2000,
        restart: 100,
        tolerance: 1e-10,
        print_interval: 0,
        convergence_check_interval: 1,
        min_restart: None,
        max_restart: None,
        exact_residual_check_interval: 0,
    };

    let solution = gmres(&matrix, &rhs, &config);
    assert!(
        solution.converged,
        "GMRES should converge for scattering problem"
    );

    // Compute L2 error against analytical solution
    let error = l2_error(&mesh, &solution.x, exact_u);
    println!("sound-soft cylinder: L2 error {error:.4e}");

    // For this mesh resolution, expect reasonable accuracy
    // Scattering problems are harder due to oscillatory nature and curved boundary
    assert!(
        error < 0.03,
        "Cylinder scattering L2 error {} should be < 0.03",
        error
    );
}

/// Test mesh convergence for cylinder scattering
fn test_cylinder_scattering_convergence() {
    let cylinder_radius = 1.0;
    let outer_radius = 3.0;
    let k = 1.5; // ka = 1.5

    let num_terms = 25;

    let exact_u = move |x: f64, y: f64, _z: f64| {
        cylinder_scattering_analytical(k, cylinder_radius, x, y, num_terms)
    };

    let source = |_x: f64, _y: f64, _z: f64| Complex64::new(0.0, 0.0);

    // Test with increasing mesh resolution
    let refinements = [(8, 16), (16, 32), (32, 64)];
    let mut errors = Vec::new();
    let mut h_values = Vec::new();

    for &(n_radial, n_angular) in &refinements {
        let mesh =
            annular_mesh_triangles(0.0, 0.0, cylinder_radius, outer_radius, n_radial, n_angular);

        let k_complex = Complex64::new(k, 0.0);
        let mut problem =
            HelmholtzProblem::assemble(&mesh, PolynomialDegree::P1, k_complex, source);

        let bc_inner = DirichletBC::new(1, |_x, _y, _z| Complex64::new(0.0, 0.0));
        let bc_outer = DirichletBC::new(2, exact_u);
        apply_dirichlet(&mut problem, &mesh, &[bc_inner, bc_outer]);
        assert_nontrivial_solve("sound-soft cylinder convergence", &mesh, &[1, 2]);

        let matrix = to_csr_matrix(&problem);
        let rhs = problem.rhs.clone();

        let config = GmresConfig {
            max_iterations: 2000,
            restart: 100,
            tolerance: 1e-10,
            print_interval: 0,
            convergence_check_interval: 1,
            min_restart: None,
            max_restart: None,
            exact_residual_check_interval: 0,
        };

        let solution = gmres(&matrix, &rhs, &config);
        assert!(solution.converged, "GMRES should converge");

        let error = l2_error(&mesh, &solution.x, exact_u);
        println!("sound-soft cylinder n_r={n_radial}, n_theta={n_angular}: L2 error {error:.4e}");
        errors.push(error);

        // Approximate h from radial spacing
        let h = (outer_radius - cylinder_radius) / (n_radial as f64);
        h_values.push(h);
    }

    // Verify errors decrease with refinement
    for i in 1..errors.len() {
        assert!(
            errors[i] < errors[i - 1],
            "Error should decrease: {} should be < {}",
            errors[i],
            errors[i - 1]
        );
    }
    assert!(
        errors.last().is_some_and(|error| *error < 0.03),
        "Finest sound-soft cylinder L2 error {:?} should be < 0.03",
        errors.last()
    );

    // Verify convergence rate (may be lower than 2 due to curved boundary approximation)
    for i in 1..errors.len() {
        let rate = (errors[i - 1] / errors[i]).log2() / (h_values[i - 1] / h_values[i]).log2();
        assert!(
            rate > 1.0,
            "Scattering convergence rate {} should be > 1.0",
            rate
        );
    }
}

/// Test that the numerical scattered field satisfies the Sommerfeld radiation
/// condition qualitatively: for outgoing waves, |u_scattered| decays like
/// 1/√r in 2D.
///
/// This solves the sound-soft cylinder scattering problem with FEM on the
/// annulus, extracts the numerical total field at interior nodes along the
/// θ = 0 ray, subtracts the incident wave, and checks the decay of the
/// resulting numerical scattered field between probe radii.
fn test_cylinder_scattering_far_field_decay() -> f64 {
    let cylinder_radius = 1.0;
    let outer_radius = 3.0;
    let k = 2.0;
    let num_terms = 30;

    // Same mesh as the sound-soft accuracy leg: radial step 1/24, so probe
    // radii 2, 2.25, 2.5, 2.75 land exactly on interior node rings.
    let n_radial = 48;
    let n_angular = 96;
    let mesh = annular_mesh_triangles(0.0, 0.0, cylinder_radius, outer_radius, n_radial, n_angular);

    let exact_u = move |x: f64, y: f64, _z: f64| {
        cylinder_scattering_analytical(k, cylinder_radius, x, y, num_terms)
    };
    let source = |_x: f64, _y: f64, _z: f64| Complex64::new(0.0, 0.0);

    let k_complex = Complex64::new(k, 0.0);
    let mut problem = HelmholtzProblem::assemble(&mesh, PolynomialDegree::P1, k_complex, source);

    let bc_inner = DirichletBC::new(1, |_x, _y, _z| Complex64::new(0.0, 0.0));
    let bc_outer = DirichletBC::new(2, exact_u);
    apply_dirichlet(&mut problem, &mesh, &[bc_inner, bc_outer]);
    assert_nontrivial_solve("cylinder far-field decay", &mesh, &[1, 2]);

    let matrix = to_csr_matrix(&problem);
    let rhs = problem.rhs.clone();

    let config = GmresConfig {
        max_iterations: 2000,
        restart: 100,
        tolerance: 1e-10,
        print_interval: 0,
        convergence_check_interval: 1,
        min_restart: None,
        max_restart: None,
        exact_residual_check_interval: 0,
    };

    let preconditioner = IluPreconditioner::from_csr(&matrix);
    let solution = gmres_preconditioned(&matrix, &preconditioner, &rhs, &config);
    assert!(
        solution.converged,
        "GMRES should converge for far-field scattering problem"
    );

    // Accuracy of the solved field against the analytical series. A large
    // value here means the truncated domain resonates at this wavenumber and
    // the decay check below would measure pollution, not physics. Measured
    // 2.16e-2 on the [1,3] mesh; the bound keeps a 1.8x margin.
    let field_error = l2_error(&mesh, &solution.x, exact_u);
    println!("cylinder far-field solved field: L2 error {field_error:.4e}");
    assert!(
        field_error < 4e-2,
        "far-field solved field L2 error {field_error} should be < 4e-2"
    );

    // Annular nodes are stored ring by ring with angle index 0 on the θ = 0
    // ray, so node (ring * n_angular) sits exactly at (r, 0).
    let probe_rings = [24usize, 30, 36, 42];
    let dr = (outer_radius - cylinder_radius) / n_radial as f64;
    let mut scattered_magnitudes = Vec::with_capacity(probe_rings.len());
    let mut radii = Vec::with_capacity(probe_rings.len());
    for &ring in &probe_rings {
        let r = cylinder_radius + ring as f64 * dr;
        let node = ring * n_angular;
        let total = solution.x[node];

        // Subtract the incident wave to get the numerical scattered field.
        let phase = k * r;
        let incident = Complex64::new(phase.cos(), phase.sin());
        let scattered = (total - incident).norm();
        println!("cylinder far-field probe r={r:.2}: |u_scattered|={scattered:.4e}");
        scattered_magnitudes.push(scattered);
        radii.push(r);
    }

    // For 2D scattering, |u_scattered| ~ 1/√r
    // So |u(r1)| * √r1 ≈ |u(r2)| * √r2
    let mut max_mismatch = 0.0f64;
    for i in 1..radii.len() {
        let r1 = radii[i - 1];
        let r2 = radii[i];
        let expected_ratio = (r1 / r2).sqrt();
        let actual_ratio = scattered_magnitudes[i] / scattered_magnitudes[i - 1];
        let mismatch = (actual_ratio - expected_ratio).abs();
        assert!(
            mismatch.is_finite(),
            "Far-field decay ratio must be finite between r={r1} and r={r2}"
        );
        max_mismatch = max_mismatch.max(mismatch);

        // Allow tolerance since we are not in the true far-field.
        assert!(
            mismatch < FAR_FIELD_DECAY_MISMATCH_TOL,
            "Far-field decay: expected ratio ~{:.3}, got {:.3}",
            expected_ratio,
            actual_ratio
        );
    }
    println!("cylinder far-field decay: maximum ratio mismatch {max_mismatch:.4e}");
    max_mismatch
}

// ============================================================================
// 3D Tests
// ============================================================================

/// Test 3D plane wave solution
fn test_3d_plane_wave() {
    let k = 2.0;
    let theta = PI / 4.0;
    let phi = PI / 3.0;

    // Wave vector
    let kx = k * theta.sin() * phi.cos();
    let ky = k * theta.sin() * phi.sin();
    let kz = k * theta.cos();

    // Use box mesh
    let mut mesh = box_mesh_tetrahedra(0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 4, 4, 4);

    // Set all boundaries to tag 1 (Dirichlet)
    mesh.set_boundary_condition(BoundaryType::Dirichlet, 1, |_| true);

    let k_complex = Complex64::new(k, 0.0);
    let mut problem =
        HelmholtzProblem::assemble(&mesh, PolynomialDegree::P1, k_complex, |_, _, _| {
            Complex64::default()
        });

    let plane_wave = move |x: f64, y: f64, z: f64| {
        let phase = kx * x + ky * y + kz * z;
        Complex64::new(phase.cos(), phase.sin())
    };

    // Apply BC on tag 1
    let bc = DirichletBC::new(1, plane_wave);
    apply_dirichlet(&mut problem, &mesh, &[bc]);

    let matrix = to_csr_matrix(&problem);
    let rhs = problem.rhs.clone();
    let config = GmresConfig {
        max_iterations: 500,
        restart: 50,
        tolerance: 1e-10,
        print_interval: 0,
        convergence_check_interval: 1,
        min_restart: None,
        max_restart: None,
        exact_residual_check_interval: 0,
    };
    let solution = gmres(&matrix, &rhs, &config);
    assert!(solution.converged, "GMRES should converge");

    let error = l2_error(&mesh, &solution.x, plane_wave);
    println!("3-D plane wave: L2 error {error:.4e}");
    assert!(
        error < 0.006,
        "3D plane wave error {} should be < 0.006",
        error
    );
}

/// Helper: 2D Rigid Cylinder scattering
fn cylinder_scattering_rigid_point(k: f64, a: f64, x: f64, y: f64, num_terms: usize) -> Complex64 {
    use spec_math::Bessel;

    let r = (x * x + y * y).sqrt();
    let theta = y.atan2(x);
    let ka = k * a;
    let kr = k * r;

    let incident = Complex64::new((kr * theta.cos()).cos(), (kr * theta.cos()).sin());
    let mut scattered = Complex64::new(0.0, 0.0);

    for n in 0..num_terms {
        let n_f64 = n as f64;
        let epsilon_n = if n == 0 { 1.0 } else { 2.0 };

        // Derivatives J'_n(x) = J_{n-1}(x) - n/x J_n(x)
        let jn_ka = ka.bessel_jv(n_f64);
        let jn_prev = if n == 0 {
            -ka.bessel_jv(1.0)
        } else {
            ka.bessel_jv(n_f64 - 1.0)
        };
        let jn_prime = jn_prev - n_f64 / ka * jn_ka;

        // Y'_n(x)
        let yn_ka = ka.bessel_yv(n_f64);
        let yn_prev = if n == 0 {
            -ka.bessel_yv(1.0)
        } else {
            ka.bessel_yv(n_f64 - 1.0)
        };
        let yn_prime = yn_prev - n_f64 / ka * yn_ka;

        let hn_prime = Complex64::new(jn_prime, yn_prime);
        let hn_kr = Complex64::new(kr.bessel_jv(n_f64), kr.bessel_yv(n_f64));

        let i_pow_n = Complex64::new((n_f64 * PI / 2.0).cos(), (n_f64 * PI / 2.0).sin());

        // Coefficient for rigid cylinder (Neumann)
        let coeff = -epsilon_n * jn_prime / hn_prime * i_pow_n;

        scattered += coeff * hn_kr * (n_f64 * theta).cos();
    }
    incident + scattered
}

fn test_cylinder_scattering_rigid() {
    let cylinder_radius = 1.0;
    let outer_radius = 3.0;
    let k = 2.0;

    let n_radial = 64;
    let n_angular = 128;
    let mesh = annular_mesh_triangles(0.0, 0.0, cylinder_radius, outer_radius, n_radial, n_angular);

    let num_terms = 30;
    let exact_u = move |x: f64, y: f64, _z: f64| {
        cylinder_scattering_rigid_point(k, cylinder_radius, x, y, num_terms)
    };

    let k_complex = Complex64::new(k, 0.0);
    let mut problem =
        HelmholtzProblem::assemble(&mesh, PolynomialDegree::P1, k_complex, |_, _, _| {
            Complex64::default()
        });

    // BCs:
    // Inner (tag 1): Neumann (Rigid) -> Natural BC (do nothing)
    // Outer (tag 2): Dirichlet -> Exact solution
    let bc_outer = DirichletBC::new(2, exact_u);
    apply_dirichlet(&mut problem, &mesh, &[bc_outer]);
    // Only the outer ring is prescribed; the inner ring stays free.
    assert_nontrivial_solve("rigid cylinder", &mesh, &[2]);

    let matrix = to_csr_matrix(&problem);
    let rhs = problem.rhs.clone();
    let config = GmresConfig {
        max_iterations: 2000,
        restart: 100,
        tolerance: 1e-10,
        print_interval: 0,
        convergence_check_interval: 1,
        min_restart: None,
        max_restart: None,
        exact_residual_check_interval: 0,
    };
    let preconditioner = IluPreconditioner::from_csr(&matrix);
    let solution = gmres_preconditioned(&matrix, &preconditioner, &rhs, &config);
    assert!(solution.converged);

    let error = l2_error(&mesh, &solution.x, exact_u);
    println!("rigid cylinder: L2 error {error:.4e}");
    assert!(
        error < 0.03,
        "Rigid cylinder error {} should be < 0.03",
        error
    );
}

/// Helper: 3D Sphere scattering point evaluation
fn sphere_scattering_point(
    k: f64,
    a: f64,
    x: f64,
    y: f64,
    z: f64,
    num_terms: usize,
    bc_type: &str,
) -> Complex64 {
    let r = (x * x + y * y + z * z).sqrt();
    if r < 1e-10 {
        return Complex64::default();
    }
    // Angle theta from z-axis
    let cos_theta = z / r;

    let ka = k * a;
    let kr = k * r;

    let mut total = Complex64::default();

    for n in 0..num_terms {
        let n_f64 = n as f64;
        let prefactor = 2.0 * n_f64 + 1.0;
        let i_pow_n = Complex64::new((n_f64 * PI / 2.0).cos(), (n_f64 * PI / 2.0).sin());

        let jn_ka = spherical_bessel_j(n, ka);
        let yn_ka = spherical_bessel_y(n, ka);
        let hn_ka = Complex64::new(jn_ka, yn_ka);

        let jn_kr = spherical_bessel_j(n, kr);
        let yn_kr = spherical_bessel_y(n, kr);
        let hn_kr = Complex64::new(jn_kr, yn_kr);

        let pn = legendre_p(n, cos_theta);

        let coeff = if bc_type == "dirichlet" {
            // Sound-soft: jn(ka) / hn(ka)
            jn_ka / hn_ka
        } else {
            // Rigid: jn'(ka) / hn'(ka)
            let jn_prev = if n == 0 {
                ka.cos() / ka
            } else {
                spherical_bessel_j(n - 1, ka)
            };
            let jn_prime = jn_prev - (n_f64 + 1.0) / ka * jn_ka;

            let yn_prev = if n == 0 {
                ka.sin() / ka
            } else {
                spherical_bessel_y(n - 1, ka)
            };
            let yn_prime = yn_prev - (n_f64 + 1.0) / ka * yn_ka;

            let hn_prime = Complex64::new(jn_prime, yn_prime);
            Complex64::new(jn_prime, 0.0) / hn_prime
        };

        // Total field = incident + scattered
        // u = Σ (2n+1) i^n [jn(kr) - coeff * hn(kr)] Pn(cos theta)
        total += prefactor * i_pow_n * (jn_kr - coeff * hn_kr) * pn;
    }
    total
}

fn test_sphere_scattering_sound_soft() {
    let a = 1.0;
    let outer_r = 2.0; // Keep domain small for test speed
    let k = 1.0; // ka=1 (Mie)

    let mesh = spherical_shell_mesh_tetrahedra(0.0, 0.0, 0.0, a, outer_r, 2, 4);

    let num_terms = 15;
    let exact_u = move |x: f64, y: f64, z: f64| {
        sphere_scattering_point(k, a, x, y, z, num_terms, "dirichlet")
    };

    let k_complex = Complex64::new(k, 0.0);
    let mut problem =
        HelmholtzProblem::assemble(&mesh, PolynomialDegree::P1, k_complex, |_, _, _| {
            Complex64::default()
        });

    // Inner: Dirichlet u=0
    let bc_inner = DirichletBC::new(1, |_, _, _| Complex64::default());
    // Outer: Exact
    let bc_outer = DirichletBC::new(2, exact_u);

    apply_dirichlet(&mut problem, &mesh, &[bc_inner, bc_outer]);

    let matrix = to_csr_matrix(&problem);
    let rhs = problem.rhs.clone();
    let config = GmresConfig {
        max_iterations: 1000,
        restart: 50,
        tolerance: 1e-8,
        print_interval: 0,
        convergence_check_interval: 1,
        min_restart: None,
        max_restart: None,
        exact_residual_check_interval: 0,
    };
    let solution = gmres(&matrix, &rhs, &config);
    assert!(solution.converged);

    let error = l2_error(&mesh, &solution.x, exact_u);
    println!("sound-soft sphere: L2 error {error:.4e}");
    assert!(
        error < 0.03,
        "Sphere scattering (soft) error {} should be < 0.03",
        error
    );
}

fn test_sphere_scattering_rigid() {
    let a = 1.0;
    let outer_r = 2.0;
    let k = 1.0;

    let mesh = spherical_shell_mesh_tetrahedra(0.0, 0.0, 0.0, a, outer_r, 2, 4);

    let num_terms = 15;
    let exact_u =
        move |x: f64, y: f64, z: f64| sphere_scattering_point(k, a, x, y, z, num_terms, "rigid");

    let k_complex = Complex64::new(k, 0.0);
    let mut problem =
        HelmholtzProblem::assemble(&mesh, PolynomialDegree::P1, k_complex, |_, _, _| {
            Complex64::default()
        });

    // Inner: Natural Neumann (Rigid)
    // Outer: Exact
    let bc_outer = DirichletBC::new(2, exact_u);

    apply_dirichlet(&mut problem, &mesh, &[bc_outer]);

    let matrix = to_csr_matrix(&problem);
    let rhs = problem.rhs.clone();
    let config = GmresConfig {
        max_iterations: 1000,
        restart: 50,
        tolerance: 1e-8,
        print_interval: 0,
        convergence_check_interval: 1,
        min_restart: None,
        max_restart: None,
        exact_residual_check_interval: 0,
    };
    let solution = gmres(&matrix, &rhs, &config);
    assert!(solution.converged);

    let error = l2_error(&mesh, &solution.x, exact_u);
    println!("rigid sphere: L2 error {error:.4e}");
    assert!(
        error < 0.03,
        "Sphere scattering (rigid) error {} should be < 0.03",
        error
    );
}

fn main() {
    let result = std::panic::catch_unwind(|| {
        let strip_1d_mode = test_1d_mode_in_2d_strip();
        test_2d_plane_wave();
        test_convergence_rate();
        test_2d_standing_wave();
        test_manufactured_solution();
        let strip_plane_wave = test_with_math_wave_plane_wave_1d();
        let strip_helmholtz_mode = test_with_math_wave_helmholtz_mode();
        let strip_damped_wave = test_with_math_wave_damped_wave();
        test_mms_sin_cos();
        test_mms_sin_cos_convergence();
        test_mms_sin_sin_2pi();
        test_mms_polynomial();
        test_mms_exp_sin();
        test_mms_complex_valued();
        test_mms_convergence_rates_detailed();
        test_mms_varying_wavenumber();
        test_mms_rectangle();
        test_cylinder_scattering_sound_soft();
        test_cylinder_scattering_convergence();
        let far_field_mismatch = test_cylinder_scattering_far_field_decay();
        test_3d_plane_wave();
        test_cylinder_scattering_rigid();
        test_sphere_scattering_sound_soft();
        test_sphere_scattering_rigid();
        (
            strip_1d_mode,
            strip_plane_wave,
            strip_helmholtz_mode,
            strip_damped_wave,
            far_field_mismatch,
        )
    });
    match result {
        Ok((
            strip_1d_mode,
            strip_plane_wave,
            strip_helmholtz_mode,
            strip_damped_wave,
            far_field_mismatch,
        )) => {
            let strip_max_quadrature = strip_1d_mode
                .quadrature_error
                .max(strip_plane_wave.quadrature_error)
                .max(strip_helmholtz_mode.quadrature_error)
                .max(strip_damped_wave.quadrature_error);
            let strip_min_free_dofs = strip_1d_mode
                .free_dofs
                .min(strip_plane_wave.free_dofs)
                .min(strip_helmholtz_mode.free_dofs)
                .min(strip_damped_wave.free_dofs);
            let record = QaResult::passed("fem.helmholtz-mms-convergence.v1")
                .requested_backend("cpu")
                .actual_backend("cpu")
                .metric("analytical_cases_executed", 24.0)
                .metric("mms_convergence_cases", 5.0)
                .metric("independent_reference", 1.0)
                .metric("strip_1d_mode_interior_error", strip_1d_mode.interior_error)
                .metric(
                    "strip_plane_wave_interior_error",
                    strip_plane_wave.interior_error,
                )
                .metric(
                    "strip_helmholtz_mode_interior_error",
                    strip_helmholtz_mode.interior_error,
                )
                .metric(
                    "strip_damped_wave_interior_error",
                    strip_damped_wave.interior_error,
                )
                .metric("strip_max_quadrature_error", strip_max_quadrature)
                .metric("strip_min_free_dofs", strip_min_free_dofs as f64)
                .metric("far_field_decay_mismatch", far_field_mismatch)
                .maximum("strip_1d_mode_interior_error", STRIP_1D_MODE_INTERIOR_TOL)
                .maximum(
                    "strip_plane_wave_interior_error",
                    STRIP_PLANE_WAVE_INTERIOR_TOL,
                )
                .maximum(
                    "strip_helmholtz_mode_interior_error",
                    STRIP_HELMHOLTZ_MODE_INTERIOR_TOL,
                )
                .maximum(
                    "strip_damped_wave_interior_error",
                    STRIP_DAMPED_WAVE_INTERIOR_TOL,
                )
                .maximum("strip_max_quadrature_error", STRIP_MAX_QUADRATURE_TOL)
                .minimum("strip_min_free_dofs", 1.0)
                .maximum("far_field_decay_mismatch", FAR_FIELD_DECAY_MISMATCH_TOL)
                .seed("helmholtz-p1:strip-40x2/40x4:annulus-48x96:mms")
                .to_json_line()
                .expect("FEM analytical QA result must serialize");
            println!("QA_RESULT: {record}");
            println!("SUCCESS");
        }
        Err(_) => {
            let record = QaResult::failed(
                "fem.helmholtz-mms-convergence.v1",
                "analytical or manufactured-solution FEM assertion failed",
            )
            .to_json_line()
            .expect("FEM analytical QA failure result must serialize");
            println!("QA_RESULT: {record}");
            println!("FAILED");
            std::process::exit(1);
        }
    }
}
