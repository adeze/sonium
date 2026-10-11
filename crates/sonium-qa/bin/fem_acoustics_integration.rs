//! FEM Acoustics Validation Tests
//!
//! Comprehensive validation suite for FEM acoustic solvers.
//!
//! ## Tests
//! 1. Cylinder Scattering (Sound-soft) - 2D
//! 2. Sphere Scattering (Sound-hard) - 3D
//! 3. Helmholtz assembly with sources and impedance BCs
//! 4. Frequency sweep with PrebakedAssembler
//!
//! Usage:
//!     cargo run --bin fem-acoustics --release

use num_complex::Complex64;
use serde::{Deserialize, Serialize};
use sonium_fem::assembly::{HelmholtzProblem, assemble_bilinear};
use sonium_fem::basis::PolynomialDegree;
#[allow(deprecated)]
use sonium_fem::boundary::{DirichletBC, apply_dirichlet};
use sonium_fem::forms::{LaplacianForm, MassForm};
use sonium_fem::io::vtk::VtuWriter;
use sonium_fem::mesh::{Mesh, annular_mesh_triangles, spherical_shell_mesh_tetrahedra};
use sonium_fem::physics::acoustics::pressure::impedance::impedance_to_robin_coefficient;
use sonium_fem::physics::acoustics::pressure::sources::{NeumannFlux, NormalVelocitySource};
use sonium_fem::physics::acoustics::pressure::{
    AcousticsAssemblerExt, build_acoustics_assembler_with_boundaries,
};
use sonium_fem::solver::{GmresConfigF64, SolverConfig, SolverType, solve};
use sonium_fem::spaces::H1Space;
use sonium_solvers::gmres;
use sonium_xem::io::binary_output_dir;

fn write_helmholtz_vtu(name: &str, mesh: &sonium_fem::mesh::Mesh, pressure: &[Complex64]) {
    let dir = binary_output_dir(env!("CARGO_PKG_NAME"), env!("CARGO_BIN_NAME"));
    let mut w = VtuWriter::new(mesh);
    w.add_point_scalar_complex("pressure", pressure);
    let _ = w.write(dir.join(format!("{name}.vtu")));
}
use sonium_wave::special::{legendre_p, spherical_bessel_j, spherical_bessel_y};
use spec_math::Bessel;
use std::collections::HashMap;
use std::path::Path;
use std::time::Instant;

/// Agreement between the iterative and direct solves of the driven system.
const TOL_GMRES_DIRECT_AGREEMENT: f64 = 1e-6;
/// True algebraic residual of the driven iterative solution.
const TOL_TRUE_RESIDUAL: f64 = 1e-8;
/// Driven pressure against the exact quasi-1D reference at 100 Hz.
const TOL_QUASI_1D_100HZ: f64 = 0.02;
/// Driven pressure against the exact quasi-1D reference at 1 Hz.
const TOL_QUASI_1D_1HZ: f64 = 0.01;
/// Cached prebaked sweep matrix against fresh triplet assembly.
const TOL_SWEEP_MATRIX: f64 = 1e-12;
/// Cached sweep solve against the freshly assembled solve.
const TOL_SWEEP_SOLVE: f64 = 1e-9;
/// Frequency for the quasi-static compatibility check.
const LOW_CHECK_FREQUENCY: f64 = 1.0;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationResult {
    pub test_name: String,
    pub solver: String,
    pub mesh_info: String,
    pub dofs: usize,
    pub duration_ms: u64,
    pub l2_error: f64,
    pub iterations: usize,
    pub residual: f64,
    pub passed: bool,
}

fn main() -> anyhow::Result<()> {
    env_logger::init();

    println!("Starting FEM Acoustics QA Suite...");
    println!("====================================");

    let mut results = Vec::new();

    // 1. Integration tests (from fem_acoustics_integration.rs)
    println!("\nRunning Integration Tests...");
    let (source_passed, source_error, source_iterations, source_residual) =
        test_acoustics_helmholtz_with_source_and_impedance();
    results.push(ValidationResult {
        test_name: "Helmholtz with source + impedance".to_string(),
        solver: "GMRES".to_string(),
        mesh_info: "unit square".to_string(),
        dofs: 121,
        duration_ms: 0,
        l2_error: source_error,
        iterations: source_iterations,
        residual: source_residual,
        passed: source_passed,
    });

    let (sweep_passed, sweep_error, sweep_points, sweep_residual) =
        test_acoustics_frequency_sweep();
    results.push(ValidationResult {
        test_name: "Frequency sweep".to_string(),
        solver: "PrebakedAssembler".to_string(),
        mesh_info: "unit square".to_string(),
        dofs: 25,
        duration_ms: 0,
        l2_error: sweep_error,
        iterations: sweep_points,
        residual: sweep_residual,
        passed: sweep_passed,
    });

    // 2. Cylinder Scattering Tests (2D)
    println!("\nRunning Cylinder Scattering Tests...");

    // Keep this integration binary in the deterministic QA tier. Higher-k
    // scattering studies require much finer meshes than the compact P1 cases
    // below and are covered by dedicated convergence/benchmark binaries.
    let cylinder_cases = [(0.5, 8, 16, "Coarse"), (2.0, 48, 96, "Fine")];

    for (k_val, n_radial, n_angular, mesh_name) in cylinder_cases {
        results.push(run_cylinder_scattering_test(
            &format!("Cylinder (k={:.1}) [Direct]", k_val),
            k_val,
            SolverConfig {
                solver_type: SolverType::Direct,
                verbosity: 0,
                ..Default::default()
            },
            n_radial,
            n_angular,
            mesh_name,
        )?);

        // Test GMRES+ILU
        let config = SolverConfig {
            solver_type: SolverType::GmresIlu,
            verbosity: 0,
            wavenumber: Some(k_val),
            gmres: GmresConfigF64 {
                max_iterations: 1000,
                restart: 50,
                tolerance: 1e-10,
                print_interval: 0,
                convergence_check_interval: 1,
                min_restart: None,
                max_restart: None,
                exact_residual_check_interval: 0,
            },
            ..Default::default()
        };
        results.push(run_cylinder_scattering_test(
            &format!("Cylinder (k={:.1}) [GMRES+ILU]", k_val),
            k_val,
            config,
            n_radial,
            n_angular,
            mesh_name,
        )?);
    }

    // 3. Sphere Scattering Tests (3D)
    println!("\nRunning Sphere Scattering Tests...");

    let sphere_cases = [(1.0, 2, 4, "Subdiv 2")];

    for (k, subdiv, layers, name) in &sphere_cases {
        results.push(run_sphere_scattering_test(
            &format!("Sphere (k={:.1}) [Direct]", k),
            *k,
            SolverConfig {
                solver_type: SolverType::Direct,
                verbosity: 0,
                ..Default::default()
            },
            *subdiv,
            *layers,
            name,
        )?);

        let config = SolverConfig {
            solver_type: SolverType::GmresIlu,
            verbosity: 0,
            wavenumber: Some(*k),
            gmres: GmresConfigF64 {
                max_iterations: 2000,
                restart: 60,
                tolerance: 1e-8,
                print_interval: 0,
                convergence_check_interval: 1,
                min_restart: None,
                max_restart: None,
                exact_residual_check_interval: 0,
            },
            ..Default::default()
        };
        results.push(run_sphere_scattering_test(
            &format!("Sphere (k={:.1}) [GMRES+ILU]", k),
            *k,
            config,
            *subdiv,
            *layers,
            name,
        )?);
    }

    // Summary
    print_summary(&results);

    let output_path = binary_output_dir(env!("CARGO_PKG_NAME"), env!("CARGO_BIN_NAME"))
        .join("qa_results_fem_acoustics.json");
    save_results(&results, &output_path)?;
    println!("\nFull results saved to: {}", output_path.display());

    let failed = results.iter().any(|r| !r.passed);
    if failed {
        eprintln!("\nSOME TESTS FAILED");
        std::process::exit(1);
    } else {
        println!("\nALL TESTS PASSED\nSUCCESS");
        Ok(())
    }
}

// =====================================================================
// Integration tests (from fem_acoustics_integration.rs)
// =====================================================================

/// Dense complex Gaussian elimination with partial pivoting.
fn dense_complex_solve(matrix: &mut [Complex64], rhs: &mut [Complex64]) -> bool {
    let n = rhs.len();
    for col in 0..n {
        let mut pivot = col;
        let mut pivot_value = matrix[col * n + col].norm();
        for row in (col + 1)..n {
            let value = matrix[row * n + col].norm();
            if value > pivot_value {
                pivot_value = value;
                pivot = row;
            }
        }
        if pivot_value == 0.0 {
            return false;
        }
        if pivot != col {
            for k in col..n {
                matrix.swap(col * n + k, pivot * n + k);
            }
            rhs.swap(col, pivot);
        }
        let diagonal = matrix[col * n + col];
        for row in (col + 1)..n {
            let factor = matrix[row * n + col] / diagonal;
            matrix[row * n + col] = Complex64::new(0.0, 0.0);
            for k in (col + 1)..n {
                matrix[row * n + k] -= factor * matrix[col * n + k];
            }
            rhs[row] -= factor * rhs[col];
        }
    }
    for row in (0..n).rev() {
        let mut sum = rhs[row];
        for k in (row + 1)..n {
            sum -= matrix[row * n + k] * rhs[k];
        }
        rhs[row] = sum / matrix[row * n + row];
    }
    true
}

fn csr_to_dense(system: &sonium_solvers::CsrMatrix<Complex64>) -> Vec<Complex64> {
    let n = system.num_rows;
    let mut dense = vec![Complex64::new(0.0, 0.0); n * n];
    for row in 0..n {
        for k in system.row_ptrs[row]..system.row_ptrs[row + 1] {
            dense[row * n + system.col_indices[k]] += system.values[k];
        }
    }
    dense
}

fn relative_complex_residual(
    matrix: &[Complex64],
    solution: &[Complex64],
    rhs: &[Complex64],
) -> f64 {
    let n = rhs.len();
    let mut residual = 0.0;
    let mut rhs_norm = 0.0;
    for row in 0..n {
        let mut row_sum = Complex64::new(0.0, 0.0);
        for col in 0..n {
            row_sum += matrix[row * n + col] * solution[col];
        }
        residual += (row_sum - rhs[row]).norm_sqr();
        rhs_norm += rhs[row].norm_sqr();
    }
    (residual / rhs_norm).sqrt()
}

fn relative_complex_difference(first: &[Complex64], second: &[Complex64]) -> f64 {
    let mut difference = 0.0;
    let mut norm = 0.0;
    for (a, b) in first.iter().zip(second.iter()) {
        difference += (a - b).norm_sqr();
        norm += b.norm_sqr();
    }
    (difference / norm).sqrt()
}

/// Assemble the driven source/impedance system at one frequency.
///
/// Returns the Helmholtz matrix and the Neumann-flux load vector for a unit
/// normal velocity on boundary tag 1 with an anechoic impedance on tag 2.
fn assemble_driven_system(
    freq: f64,
) -> (Mesh, sonium_solvers::CsrMatrix<Complex64>, Vec<Complex64>) {
    let mesh = sonium_fem::mesh::unit_square_triangles(10);
    let boundary_tags = vec![2];
    let assembler =
        build_acoustics_assembler_with_boundaries(&mesh, PolynomialDegree::P1, &boundary_tags);

    let rho = 1.204;
    let c = 343.0;
    let k_val = 2.0 * std::f64::consts::PI * freq / c;
    let k = Complex64::new(k_val, 0.0);

    let alpha = impedance_to_robin_coefficient(Complex64::new(rho * c, 0.0), k_val, rho, c);

    let mut boundaries = HashMap::new();
    boundaries.insert(2, alpha);

    let system = assembler.assemble_helmholtz(k, &boundaries);

    let v0 = Complex64::new(1.0, 0.0);
    let source = NormalVelocitySource {
        amplitude: v0,
        omega: 2.0 * std::f64::consts::PI * freq,
        density: rho,
    };
    let flux = source.neumann_flux();

    let b_mass = sonium_fem::assembly::assemble_boundary_mass(&mesh, PolynomialDegree::P1, 1);
    let mut rhs = vec![Complex64::new(0.0, 0.0); mesh.num_nodes()];
    for k in 0..b_mass.nnz() {
        rhs[b_mass.rows[k]] += flux * b_mass.values[k];
    }

    (mesh, system, rhs)
}

fn test_acoustics_helmholtz_with_source_and_impedance() -> (bool, f64, usize, f64) {
    let fail = |message: &str| {
        eprintln!("{message}");
        (false, 1.0, 0, 1.0)
    };

    let (mesh, system, rhs) = assemble_driven_system(100.0);
    let config = sonium_solvers::GmresConfig {
        max_iterations: 1000,
        restart: 50,
        tolerance: 1e-10,
        print_interval: 0,
        convergence_check_interval: 1,
        min_restart: None,
        max_restart: None,
        exact_residual_check_interval: 0,
    };

    let result = gmres(&system, &rhs, &config);

    if !result.converged {
        return fail("GMRES failed to converge");
    }

    let norm = result.x.iter().map(|c| c.norm_sqr()).sum::<f64>().sqrt();
    if norm <= 0.0 {
        return fail("Solution is trivial");
    }

    // Independent direct solve of the same driven system.
    let dense = csr_to_dense(&system);
    let mut direct_matrix = dense.clone();
    let mut direct_solution = rhs.clone();
    if !dense_complex_solve(&mut direct_matrix, &mut direct_solution) {
        return fail("direct solve of the driven system failed");
    }
    let agreement = relative_complex_difference(&result.x, &direct_solution);
    if agreement > TOL_GMRES_DIRECT_AGREEMENT {
        return fail(&format!(
            "GMRES/direct agreement {agreement:.3e} exceeds {TOL_GMRES_DIRECT_AGREEMENT:.1e}"
        ));
    }

    // True algebraic residual of the iterative solution.
    let true_residual = relative_complex_residual(&dense, &result.x, &rhs);
    if true_residual > TOL_TRUE_RESIDUAL {
        return fail(&format!(
            "true residual {true_residual:.3e} exceeds {TOL_TRUE_RESIDUAL:.1e}"
        ));
    }

    // Exact quasi-1D reference: the uniform drive, uniform Robin edge, and
    // zero-flux top/bottom make the solution y-independent, so it equals the
    // 1-D solution p(x) = A cos(kx) + B sin(kx) with -p'(0) = g for the unit
    // normal velocity and p'(1) + alpha p(1) = 0 with alpha = -i k. The flux
    // g = i omega rho and alpha are evaluated from first principles so the
    // reference stays independent of the assembly helpers.
    let rho = 1.204;
    let c = 343.0;
    let quasi_1d_error = |freq: f64, solution: &[Complex64], nodes: &Mesh| -> f64 {
        let omega = 2.0 * std::f64::consts::PI * freq;
        let k = omega / c;
        let flux = Complex64::new(0.0, omega * rho);
        let alpha = Complex64::new(0.0, -k);
        let b = -flux / k;
        let cos_k = k.cos();
        let sin_k = k.sin();
        let a = -b * (k * cos_k + alpha * sin_k) / (alpha * cos_k - k * sin_k);
        let mut error = 0.0;
        let mut norm = 0.0;
        for (node, point) in nodes.nodes.iter().enumerate() {
            let exact = a * (k * point.x).cos() + b * (k * point.x).sin();
            error += (solution[node] - exact).norm_sqr();
            norm += exact.norm_sqr();
        }
        (error / norm).sqrt()
    };
    let exact_100hz = quasi_1d_error(100.0, &result.x, &mesh);
    if exact_100hz > TOL_QUASI_1D_100HZ {
        return fail(&format!(
            "quasi-1D error {exact_100hz:.3e} at 100 Hz exceeds {TOL_QUASI_1D_100HZ:.1e}"
        ));
    }

    let (low_mesh, low_system, low_rhs) = assemble_driven_system(LOW_CHECK_FREQUENCY);
    let low_dense = csr_to_dense(&low_system);
    let mut low_matrix = low_dense.clone();
    let mut low_solution = low_rhs.clone();
    if !dense_complex_solve(&mut low_matrix, &mut low_solution) {
        return fail("direct solve of the low-frequency system failed");
    }
    let exact_1hz = quasi_1d_error(LOW_CHECK_FREQUENCY, &low_solution, &low_mesh);
    if exact_1hz > TOL_QUASI_1D_1HZ {
        return fail(&format!(
            "quasi-1D error {exact_1hz:.3e} at 1 Hz exceeds {TOL_QUASI_1D_1HZ:.1e}"
        ));
    }

    write_helmholtz_vtu("helmholtz_source_impedance", &mesh, &result.x);

    (
        true,
        agreement.max(exact_100hz).max(exact_1hz),
        result.iterations,
        true_residual,
    )
}

fn test_acoustics_frequency_sweep() -> (bool, f64, usize, f64) {
    let fail = |message: &str| {
        eprintln!("{message}");
        (false, 1.0, 0, 1.0)
    };

    let mesh = sonium_fem::mesh::unit_square_triangles(4);
    let assembler = build_acoustics_assembler_with_boundaries(&mesh, PolynomialDegree::P1, &[]);

    // Fresh stiffness and mass triplets with the same quadrature as the
    // prebaked assembler, for the cached-versus-fresh comparison.
    let space = H1Space::new(PolynomialDegree::P1);
    let quad_stiffness =
        sonium_fem::quadrature::required_order_for_stiffness(PolynomialDegree::P1.degree());
    let quad_mass = sonium_fem::quadrature::required_order_for_mass(PolynomialDegree::P1.degree());
    let fresh_stiffness = assemble_bilinear(&LaplacianForm, &space, &space, &mesh, quad_stiffness);
    let fresh_mass = assemble_bilinear(&MassForm, &space, &space, &mesh, quad_mass);
    let n = mesh.num_nodes();

    let frequencies = vec![50.0, 100.0, 200.0];
    let c = 343.0;

    let mut worst_matrix_error = 0.0f64;
    let mut worst_solve_error = 0.0f64;
    for f in &frequencies {
        let k_val = 2.0 * std::f64::consts::PI * f / c;
        let k = Complex64::new(k_val, 0.0);
        let cached = assembler.assemble_helmholtz(k, &HashMap::new());
        if cached.num_rows != n {
            return fail("sweep system dimension mismatch");
        }
        let cached_dense = csr_to_dense(&cached);

        // Fresh assembly of A = K - k^2 M.
        let k_squared = k * k;
        let mut fresh_dense = vec![Complex64::new(0.0, 0.0); n * n];
        for entry in 0..fresh_stiffness.values.len() {
            fresh_dense[fresh_stiffness.rows[entry] * n + fresh_stiffness.cols[entry]] +=
                Complex64::new(fresh_stiffness.values[entry], 0.0);
        }
        for entry in 0..fresh_mass.values.len() {
            fresh_dense[fresh_mass.rows[entry] * n + fresh_mass.cols[entry]] +=
                -k_squared * fresh_mass.values[entry];
        }
        let mut matrix_difference = 0.0f64;
        let mut matrix_norm = 0.0f64;
        for entry in 0..n * n {
            matrix_difference =
                matrix_difference.max((cached_dense[entry] - fresh_dense[entry]).norm());
            matrix_norm = matrix_norm.max(cached_dense[entry].norm());
        }
        let matrix_error = matrix_difference / matrix_norm;
        worst_matrix_error = worst_matrix_error.max(matrix_error);
        if matrix_error > TOL_SWEEP_MATRIX {
            return fail(&format!(
                "sweep matrix error {matrix_error:.3e} at {f} Hz exceeds {TOL_SWEEP_MATRIX:.1e}"
            ));
        }

        // Cached-versus-fresh solves with a deterministic complex load.
        let load: Vec<Complex64> = (0..n)
            .map(|i| Complex64::new((i % 7) as f64 - 3.0, (i % 5) as f64 * 0.5))
            .collect();
        let mut cached_matrix = cached_dense.clone();
        let mut cached_solution = load.clone();
        if !dense_complex_solve(&mut cached_matrix, &mut cached_solution) {
            return fail(&format!("cached sweep solve failed at {f} Hz"));
        }
        let mut fresh_matrix = fresh_dense.clone();
        let mut fresh_solution = load.clone();
        if !dense_complex_solve(&mut fresh_matrix, &mut fresh_solution) {
            return fail(&format!("fresh sweep solve failed at {f} Hz"));
        }
        let solve_error = relative_complex_difference(&cached_solution, &fresh_solution);
        worst_solve_error = worst_solve_error.max(solve_error);
        if solve_error > TOL_SWEEP_SOLVE {
            return fail(&format!(
                "sweep solve error {solve_error:.3e} at {f} Hz exceeds {TOL_SWEEP_SOLVE:.1e}"
            ));
        }
    }

    (
        true,
        worst_solve_error,
        frequencies.len(),
        worst_matrix_error,
    )
}

// =====================================================================
// 2D Cylinder Scattering
// =====================================================================

fn run_cylinder_scattering_test(
    name: &str,
    k: f64,
    config: SolverConfig,
    n_radial: usize,
    n_angular: usize,
    mesh_name: &str,
) -> anyhow::Result<ValidationResult> {
    println!("  Executing: {}...", name);
    let start_time = Instant::now();

    let cylinder_radius = 1.0;
    let outer_radius = 3.0;
    let mesh = annular_mesh_triangles(0.0, 0.0, cylinder_radius, outer_radius, n_radial, n_angular);

    let num_terms = (k * cylinder_radius + 15.0) as usize;
    let exact_u = move |x: f64, y: f64, _z: f64| {
        cylinder_scattering_analytical(k, cylinder_radius, x, y, num_terms)
    };

    let k_complex = Complex64::new(k, 0.0);
    let source = |_x: f64, _y: f64, _z: f64| Complex64::new(0.0, 0.0);

    let problem = HelmholtzProblem::assemble(&mesh, PolynomialDegree::P1, k_complex, source);
    let num_dofs = problem.num_dofs();
    let _b_norm = problem.rhs.iter().map(|x| x.norm_sqr()).sum::<f64>().sqrt();

    let mut problem = problem;
    let bc_inner = DirichletBC::new(1, |_x, _y, _z| Complex64::new(0.0, 0.0));
    let bc_outer = DirichletBC::new(2, exact_u);
    #[allow(deprecated)]
    apply_dirichlet(&mut problem, &mesh, &[bc_inner, bc_outer]);

    let result = solve(&problem, &config);

    let duration = start_time.elapsed().as_millis() as u64;

    let (_solution_vec, iterations, residual, passed, error) = match result {
        Ok(sol) => {
            let error = l2_error(&mesh, &sol.values, exact_u);
            let threshold = if k < 1.0 { 0.01 } else { 0.03 };
            write_helmholtz_vtu(
                &format!("cylinder_k{:.1}_{}", k, mesh_name.to_lowercase()),
                &mesh,
                &sol.values,
            );
            (
                sol.values,
                sol.iterations,
                sol.residual,
                error < threshold,
                error,
            )
        }
        Err(e) => {
            eprintln!("Solver failed: {}", e);
            (vec![Complex64::new(0.0, 0.0); num_dofs], 0, 1.0, false, 1.0)
        }
    };

    Ok(ValidationResult {
        test_name: name.to_string(),
        solver: format!("{:?}", config.solver_type),
        mesh_info: format!("{} ({} DOFs)", mesh_name, num_dofs),
        dofs: num_dofs,
        duration_ms: duration,
        l2_error: error,
        iterations,
        residual,
        passed,
    })
}

// =====================================================================
// 3D Sphere Scattering
// =====================================================================

fn run_sphere_scattering_test(
    name: &str,
    k: f64,
    config: SolverConfig,
    subdivisions: usize,
    layers: usize,
    mesh_name: &str,
) -> anyhow::Result<ValidationResult> {
    println!("  Executing: {}...", name);
    let start_time = Instant::now();

    let sphere_radius = 1.0;
    let outer_radius = 2.0;
    let mesh = spherical_shell_mesh_tetrahedra(
        0.0,
        0.0,
        0.0,
        sphere_radius,
        outer_radius,
        subdivisions,
        layers,
    );

    let num_terms = (k * sphere_radius + 10.0) as usize;
    let coeffs = compute_rigid_sphere_coefficients(k * sphere_radius, num_terms);

    let exact_u = move |x: f64, y: f64, z: f64| {
        let r = (x * x + y * y + z * z).sqrt();
        let kr = k * r;
        let cos_theta = if r > 1e-10 { z / r } else { 0.0 };

        let mut total = Complex64::new(0.0, 0.0);

        for (n, coeff) in coeffs.iter().enumerate() {
            let n_f64 = n as f64;
            let prefactor = 2.0 * n_f64 + 1.0;
            let i_pow = match n % 4 {
                0 => Complex64::new(1.0, 0.0),
                1 => Complex64::new(0.0, 1.0),
                2 => Complex64::new(-1.0, 0.0),
                _ => Complex64::new(0.0, -1.0),
            };

            let j_vals = spherical_bessel_j(n + 1, kr);
            let y_vals = spherical_bessel_y(n + 1, kr);
            let jn = j_vals[n];
            let yn = y_vals[n];
            let hn = Complex64::new(jn, yn);
            let pn = legendre_p(n, cos_theta);

            total += prefactor * i_pow * (jn - coeff * hn) * pn;
        }
        total
    };

    let k_complex = Complex64::new(k, 0.0);
    let source = |_x: f64, _y: f64, _z: f64| Complex64::new(0.0, 0.0);

    let problem = HelmholtzProblem::assemble(&mesh, PolynomialDegree::P1, k_complex, source);
    let num_dofs = problem.num_dofs();

    let mut problem = problem;
    let bc_outer = DirichletBC::new(2, exact_u.clone());
    #[allow(deprecated)]
    apply_dirichlet(&mut problem, &mesh, &[bc_outer]);

    let result = solve(&problem, &config);

    let duration = start_time.elapsed().as_millis() as u64;

    let (_solution_vec, iterations, residual, passed, error) = match result {
        Ok(sol) => {
            let error = l2_error(&mesh, &sol.values, exact_u);
            let threshold = 0.03;
            write_helmholtz_vtu(
                &format!("sphere_k{:.1}_{}", k, mesh_name.to_lowercase()),
                &mesh,
                &sol.values,
            );
            (
                sol.values,
                sol.iterations,
                sol.residual,
                error < threshold,
                error,
            )
        }
        Err(e) => {
            eprintln!("Solver failed: {}", e);
            (vec![Complex64::new(0.0, 0.0); num_dofs], 0, 1.0, false, 1.0)
        }
    };

    Ok(ValidationResult {
        test_name: name.to_string(),
        solver: format!("{:?}", config.solver_type),
        mesh_info: format!("{} ({} DOFs)", mesh_name, num_dofs),
        dofs: num_dofs,
        duration_ms: duration,
        l2_error: error,
        iterations,
        residual,
        passed,
    })
}

// =====================================================================
// Analytical Solutions
// =====================================================================

fn compute_rigid_sphere_coefficients(ka: f64, num_terms: usize) -> Vec<Complex64> {
    let mut coefficients = Vec::with_capacity(num_terms);
    for n in 0..num_terms {
        let n_f64 = n as f64;
        let j_n_vals = spherical_bessel_j(n + 1, ka);
        let y_n_vals = spherical_bessel_y(n + 1, ka);
        let jn = j_n_vals[n];
        let yn = y_n_vals[n];

        let jn_minus_1 = if n > 0 {
            j_n_vals[n - 1]
        } else {
            ka.cos() / ka
        };
        let yn_minus_1 = if n > 0 {
            y_n_vals[n - 1]
        } else {
            // y_{-1}(x) = sin(x)/x.  The negative sign belongs to
            // y_0(x) = -cos(x)/x; carrying it into y_{-1} corrupts the
            // n=0 derivative and produces a mesh-independent sphere error.
            ka.sin() / ka
        };

        let jn_prime = jn_minus_1 - (n_f64 + 1.0) / ka * jn;
        let yn_prime = yn_minus_1 - (n_f64 + 1.0) / ka * yn;

        let hn_prime = Complex64::new(jn_prime, yn_prime);
        coefficients.push(Complex64::new(jn_prime, 0.0) / hn_prime);
    }
    coefficients
}

fn cylinder_scattering_analytical(
    k: f64,
    cylinder_radius: f64,
    x: f64,
    y: f64,
    num_terms: usize,
) -> Complex64 {
    let r = (x * x + y * y).sqrt();
    let theta = y.atan2(x);
    if r < cylinder_radius * 0.999 {
        return Complex64::new(0.0, 0.0);
    }

    let ka = k * cylinder_radius;
    let kr = k * r;
    let incident = Complex64::new((kr * theta.cos()).cos(), (kr * theta.cos()).sin());
    let mut scattered = Complex64::new(0.0, 0.0);

    for n in 0..num_terms {
        let n_f64 = n as f64;
        let epsilon_n = if n == 0 { 1.0 } else { 2.0 };

        let jn_ka = ka.bessel_jv(n_f64);
        let yn_ka = ka.bessel_yv(n_f64);
        let jn_kr = kr.bessel_jv(n_f64);
        let yn_kr = kr.bessel_yv(n_f64);

        let hn_ka = Complex64::new(jn_ka, yn_ka);
        let hn_kr = Complex64::new(jn_kr, yn_kr);

        let i_power_n = match n % 4 {
            0 => Complex64::new(1.0, 0.0),
            1 => Complex64::new(0.0, 1.0),
            2 => Complex64::new(-1.0, 0.0),
            _ => Complex64::new(0.0, -1.0),
        };

        let a_n = -epsilon_n * jn_ka / hn_ka * i_power_n;
        scattered += a_n * hn_kr * (n_f64 * theta).cos();
    }
    incident + scattered
}

fn l2_error<F>(mesh: &Mesh, fem_solution: &[Complex64], analytical: F) -> f64
where
    F: Fn(f64, f64, f64) -> Complex64,
{
    let mut error_sq = 0.0;
    let mut norm_sq = 0.0;
    for (i, node) in mesh.nodes.iter().enumerate() {
        let fem_val = fem_solution[i];
        let exact_val = analytical(node.x, node.y, node.z);
        error_sq += (fem_val - exact_val).norm_sqr();
        norm_sq += exact_val.norm_sqr();
    }
    if norm_sq > 1e-15 {
        (error_sq / norm_sq).sqrt()
    } else {
        error_sq.sqrt()
    }
}

fn print_summary(results: &[ValidationResult]) {
    println!("\nQA Summary:");
    println!(
        "{:<35} | {:<20} | {:<15} | {:<6} | {:<6} | {:<8} | {:<10}",
        "Test Name", "Solver", "Mesh", "Error%", "Iters", "Time(ms)", "Status"
    );
    println!("{:-<120}", "");

    for res in results {
        let l2_err = res.l2_error * 100.0;
        let status = if res.passed { "PASS" } else { "FAIL" };
        let solver_short = res.solver.replace("Gmres", "GMRES");
        println!(
            "{:<35} | {:<20} | {:<15} | {:6.2}% | {:<6} | {:<8} | {}",
            res.test_name,
            solver_short,
            res.mesh_info,
            l2_err,
            res.iterations,
            res.duration_ms,
            status
        );
    }
}

fn save_results(results: &[ValidationResult], path: impl AsRef<Path>) -> anyhow::Result<()> {
    let json = serde_json::to_string_pretty(results)?;
    std::fs::write(path, json)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rigid_sphere_monopole_coefficient_uses_correct_neumann_derivative() {
        let x = 0.7_f64;
        let coefficient = compute_rigid_sphere_coefficients(x, 1)[0];
        let j0_prime = (x * x.cos() - x.sin()) / (x * x);
        let y0_prime = (x * x.sin() + x.cos()) / (x * x);
        let expected = Complex64::new(j0_prime, 0.0) / Complex64::new(j0_prime, y0_prime);

        assert!(
            (coefficient - expected).norm() < 1.0e-12,
            "n=0 rigid-sphere coefficient {coefficient:?} disagrees with {expected:?}"
        );
    }
}
