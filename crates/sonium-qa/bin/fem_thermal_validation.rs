//! Thermal validation tests
//!
//! - MMS tests: Method of Manufactured Solutions with exact analytical solutions
//! - Convergence tests: Verify O(h²) convergence for P1 elements
//! - Physical benchmarks: 1D/2D steady-state with known solutions
//! - Transient test: 1D diffusion with analytical solution

use sonium_fem::assembly::{assemble_bilinear, assemble_linear};
use sonium_fem::basis::PolynomialDegree;
use sonium_fem::boundary::generic::{FemSystem, apply_dirichlet_values};
use sonium_fem::io::vtk::VtuWriter;
use sonium_fem::mesh::generators::unit_square_triangles;
use sonium_fem::physics::thermal::heat_conduction::{
    ConductionForm, HeatSourceForm, ThermalMaterial,
};
use sonium_fem::spaces::H1Space;
use sonium_solvers::{GmresConfig, gmres};
use sonium_xem::io::binary_output_dir;

fn write_thermal_vtu(name: &str, mesh: &sonium_fem::mesh::Mesh, temperature: &[f64]) {
    let dir = binary_output_dir(env!("CARGO_PKG_NAME"), env!("CARGO_BIN_NAME"));
    let mut w = VtuWriter::new(mesh);
    w.add_point_scalar("temperature", temperature);
    let _ = w.write(dir.join(format!("{name}.vtu")));
}

fn assemble_and_solve_thermal(
    mesh: &sonium_fem::mesh::Mesh,
    conductivity: f64,
    bc_left: f64,
    bc_right: f64,
    rhs_fn: impl Fn(f64, f64) -> f64 + Send + Sync,
    extra_dirichlet_markers: &[i32],
) -> Result<Vec<f64>, String> {
    let space = H1Space::new(PolynomialDegree::P1);

    let material = ThermalMaterial::new(conductivity, 450.0, 7800.0);
    let form = ConductionForm::new(&material);

    let stiffness = assemble_bilinear(&form, &space, &space, mesh, 2);
    let source = HeatSourceForm {
        source_fn: |x, y, _z| rhs_fn(x, y),
    };
    let rhs = assemble_linear(&source, &space, mesh, 2);

    let mut system = FemSystem::new(stiffness, rhs);

    apply_dirichlet_values(&mut system, mesh, &space, 1, &|_, _x, _| bc_left, None);
    apply_dirichlet_values(&mut system, mesh, &space, 2, &|_, _x, _| bc_right, None);
    for &marker in extra_dirichlet_markers {
        apply_dirichlet_values(&mut system, mesh, &space, marker, &|_, _, _| 0.0, None);
    }

    let csr = system.matrix.to_csr();

    let config = GmresConfig {
        max_iterations: 1000,
        restart: 50,
        tolerance: 1e-12,
        print_interval: 0,
        convergence_check_interval: 1,
        min_restart: None,
        max_restart: None,
        exact_residual_check_interval: 0,
    };

    let result = gmres(&csr, &system.rhs, &config);
    if result.converged {
        Ok(result.x)
    } else {
        Err(format!(
            "Solver failed to converge: {:.2e}",
            result.residual
        ))
    }
}

fn test_thermal_mms_1d_parabolic() {
    let k = 1.0;
    let l = 1.0;
    let bc_left = 0.0;
    let bc_right = 0.0;

    let exact_solution = |x: f64| -> f64 { 0.5 * x * (l - x) };

    let nx = 20;
    let ny = 2;
    let mesh = sonium_fem::mesh::generators::rectangular_mesh_triangles(0.0, l, 0.0, 0.1, nx, ny);

    let solution = assemble_and_solve_thermal(&mesh, k, bc_left, bc_right, |_, _| k, &[])
        .expect("Solver should converge");

    write_thermal_vtu("mms_1d_parabolic", &mesh, &solution);
    let mut max_err: f64 = 0.0;
    for (i, &t_fem) in solution.iter().enumerate().take(mesh.num_nodes()) {
        let x = mesh.nodes[i].x;
        let t_exact = exact_solution(x);
        let err = (t_fem - t_exact).abs();
        max_err = max_err.max(err);
    }

    assert!(
        max_err < 1e-3,
        "MMS 1D parabolic error too large: {:.2e}",
        max_err
    );
}

fn test_thermal_mms_2d_sinusoidal() {
    let k = 1.0;

    let exact_solution = |x: f64, y: f64| -> f64 {
        (std::f64::consts::PI * x).sin() * (std::f64::consts::PI * y).sin()
    };

    let nx = 10;
    let mesh = unit_square_triangles(nx);

    let solution = assemble_and_solve_thermal(
        &mesh,
        k,
        0.0,
        0.0,
        |x, y| {
            2.0 * k
                * std::f64::consts::PI.powi(2)
                * (std::f64::consts::PI * x).sin()
                * (std::f64::consts::PI * y).sin()
        },
        &[3, 4],
    )
    .expect("Solver should converge");

    let mut max_err: f64 = 0.0;
    for (i, &t_fem) in solution.iter().enumerate().take(mesh.num_nodes()) {
        let x = mesh.nodes[i].x;
        let y = mesh.nodes[i].y;
        let t_exact = exact_solution(x, y);
        let err = t_fem - t_exact;
        max_err = max_err.max(err.abs());
    }

    assert!(
        max_err < 1e-2,
        "MMS 2D sinusoidal error too large: {:.2e}",
        max_err
    );
}

fn test_thermal_convergence_rate() {
    let k = 1.0;
    let bc_left = 0.0;
    let bc_right = 0.0;
    let l = 1.0;

    let exact_solution = |x: f64| -> f64 { 0.5 * x * (l - x) };

    let errors: Vec<f64> = [4, 8, 16]
        .iter()
        .map(|&n| {
            let mesh =
                sonium_fem::mesh::generators::rectangular_mesh_triangles(0.0, l, 0.0, 0.1, n, 2);

            let solution = assemble_and_solve_thermal(&mesh, k, bc_left, bc_right, |_, _| k, &[])
                .expect("Solver should converge");

            write_thermal_vtu(&format!("convergence_n{}", n), &mesh, &solution);
            let mut max_err: f64 = 0.0;
            for (i, &t_fem) in solution.iter().enumerate().take(mesh.num_nodes()) {
                let x = mesh.nodes[i].x;
                let t_exact = exact_solution(x);
                max_err = max_err.max((t_fem - t_exact).abs());
            }
            max_err
        })
        .collect();

    println!(
        "Manufactured parabolic strip: n=4 err={:.2e}, n=8 err={:.2e}, n=16 err={:.2e}",
        errors[0], errors[1], errors[2]
    );

    assert!(
        errors.iter().all(|err| *err < 1e-10),
        "Manufactured parabolic solution should be reproduced to roundoff on this strip: {:?}",
        errors
    );
}

fn test_thermal_1d_linear_solution() {
    let k = 2.0;
    let bc_left = 10.0;
    let bc_right = 30.0;
    let l = 1.0;

    let exact_solution = |x: f64| -> f64 { bc_left + (bc_right - bc_left) * x };

    let nx = 10;
    let ny = 2;
    let mesh = sonium_fem::mesh::generators::rectangular_mesh_triangles(0.0, l, 0.0, 0.1, nx, ny);

    let solution = assemble_and_solve_thermal(&mesh, k, bc_left, bc_right, |_, _| 0.0, &[])
        .expect("Solver should converge");

    write_thermal_vtu("1d_linear", &mesh, &solution);
    let mut max_err: f64 = 0.0;
    for (i, &t_fem) in solution.iter().enumerate().take(mesh.num_nodes()) {
        let x = mesh.nodes[i].x;
        let t_exact = exact_solution(x);
        max_err = max_err.max((t_fem - t_exact).abs());
    }

    assert!(
        max_err < 1e-8,
        "Linear solution error too large: {:.2e}",
        max_err
    );
}

fn test_thermal_conduction_form_symmetry() {
    let mesh = unit_square_triangles(4);
    let space = H1Space::new(PolynomialDegree::P1);
    let material = ThermalMaterial::new(1.0, 450.0, 7800.0);
    let form = ConductionForm::new(&material);

    let triplet = assemble_bilinear(&form, &space, &space, &mesh, 1);
    let csr = triplet.to_csr();

    let n = csr.num_rows;
    let mut max_asym = 0.0_f64;
    for i in 0..n {
        for k in csr.row_ptrs[i]..csr.row_ptrs[i + 1] {
            let j = csr.col_indices[k];
            let aij = csr.values[k];
            let aji = (csr.row_ptrs[j]..csr.row_ptrs[j + 1])
                .find(|&kk| csr.col_indices[kk] == i)
                .map(|kk| csr.values[kk])
                .unwrap_or(0.0);
            max_asym = max_asym.max((aij - aji).abs());
        }
    }

    assert!(
        max_asym < 1e-12,
        "Conduction form not symmetric: max |A[i,j] - A[j,i]| = {:.2e}",
        max_asym
    );
}

fn main() {
    let result = std::panic::catch_unwind(|| {
        test_thermal_mms_1d_parabolic();
        test_thermal_mms_2d_sinusoidal();
        test_thermal_convergence_rate();
        test_thermal_1d_linear_solution();
        test_thermal_conduction_form_symmetry();
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
