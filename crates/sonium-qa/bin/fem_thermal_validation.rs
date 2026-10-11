//! Thermal validation tests
//!
//! - MMS tests: Method of Manufactured Solutions with exact analytical solutions
//! - Convergence tests: Verify O(h^2) convergence for P1 elements
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
use sonium_fem::quadrature::QuadratureRule;
use sonium_fem::spaces::{FunctionSpace, H1Space};
use sonium_qa::QaResult;
use sonium_solvers::{GmresConfig, gmres};
use sonium_xem::io::binary_output_dir;

const CASE_ID: &str = "fem.thermal-validation.v1";
// Nodal error bound for the exactly resolved 1D parabolic patch (tight: P1
// nodes are exact for 1D Poisson, so this guards solver/assembly regressions).
const MMS_1D_MAX_ERROR: f64 = 1e-3;
// Nodal error bound for the 2D sinusoidal MMS on a 10x10 mesh.
const MMS_2D_MAX_ERROR: f64 = 1e-2;
// Nodal error bound for the exactly linear 1D steady state.
const LINEAR_MAX_ERROR: f64 = 1e-8;
// Symmetry bound for the conduction operator.
const SYMMETRY_TOLERANCE: f64 = 1e-12;
// Minimum accepted L2 refinement rate for P1 elements (theory predicts 2.0;
// the bound allows pre-asymptotic meshes while still requiring convergence).
const L2_RATE_MINIMUM: f64 = 1.5;
// Maximum accepted quadrature L2 error on the finest convergence mesh.
const FINE_L2_MAX_ERROR: f64 = 2e-3;

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

fn test_thermal_mms_1d_parabolic() -> Result<f64, String> {
    let k = 1.0;
    let l = 1.0;
    let bc_left = 0.0;
    let bc_right = 0.0;

    let exact_solution = |x: f64| -> f64 { 0.5 * x * (l - x) };

    let nx = 20;
    let ny = 2;
    let mesh = sonium_fem::mesh::generators::rectangular_mesh_triangles(0.0, l, 0.0, 0.1, nx, ny);

    let solution = assemble_and_solve_thermal(&mesh, k, bc_left, bc_right, |_, _| k, &[])
        .map_err(|error| format!("MMS 1D parabolic solve failed: {error}"))?;

    write_thermal_vtu("mms_1d_parabolic", &mesh, &solution);
    let mut max_err: f64 = 0.0;
    for (i, &t_fem) in solution.iter().enumerate().take(mesh.num_nodes()) {
        let x = mesh.nodes[i].x;
        let t_exact = exact_solution(x);
        let err = (t_fem - t_exact).abs();
        max_err = max_err.max(err);
    }

    if !max_err.is_finite() || max_err >= MMS_1D_MAX_ERROR {
        return Err(format!(
            "MMS 1D parabolic error too large: {max_err:.2e} (bound {MMS_1D_MAX_ERROR:.1e})"
        ));
    }
    println!("MMS 1D parabolic: max nodal error={max_err:.2e}");
    Ok(max_err)
}

fn test_thermal_mms_2d_sinusoidal() -> Result<f64, String> {
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
    .map_err(|error| format!("MMS 2D sinusoidal solve failed: {error}"))?;

    let mut max_err: f64 = 0.0;
    for (i, &t_fem) in solution.iter().enumerate().take(mesh.num_nodes()) {
        let x = mesh.nodes[i].x;
        let y = mesh.nodes[i].y;
        let t_exact = exact_solution(x, y);
        let err = t_fem - t_exact;
        max_err = max_err.max(err.abs());
    }

    if !max_err.is_finite() || max_err >= MMS_2D_MAX_ERROR {
        return Err(format!(
            "MMS 2D sinusoidal error too large: {max_err:.2e} (bound {MMS_2D_MAX_ERROR:.1e})"
        ));
    }
    println!("MMS 2D sinusoidal: max nodal error={max_err:.2e}");
    Ok(max_err)
}

/// Element-quadrature L2 error of the P1 solution against the nonpolynomial
/// manufactured field sin(pi x) sin(pi y) on an n x n unit-square mesh.
fn quadrature_l2_error(n: usize) -> Result<f64, String> {
    let k = 1.0;
    let mesh = unit_square_triangles(n);
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
    .map_err(|error| format!("convergence solve at n={n} failed: {error}"))?;

    write_thermal_vtu(&format!("convergence_n{n}"), &mesh, &solution);

    let space = H1Space::new(PolynomialDegree::P1);
    let mut l2_squared = 0.0_f64;
    for element_idx in 0..mesh.num_elements() {
        let rule = QuadratureRule::new(mesh.elements[element_idx].element_type, 5);
        let local_dofs = space.local_to_global(&mesh, element_idx);
        for point in rule.iter() {
            let tabulation =
                space.tabulate(&mesh, element_idx, point.xi(), point.eta(), point.zeta());
            let mut actual = 0.0_f64;
            for (local, &global) in local_dofs.iter().enumerate() {
                actual += solution[global] * tabulation.value(local);
            }
            let [x, y, _] = tabulation.physical_coords;
            let expected = (std::f64::consts::PI * x).sin() * (std::f64::consts::PI * y).sin();
            l2_squared += point.weight * tabulation.det_j * (actual - expected).powi(2);
        }
    }
    Ok(l2_squared.sqrt())
}

fn test_thermal_convergence_rate() -> Result<(f64, f64, f64, f64), String> {
    // Nonpolynomial 2D manufactured solution: nodal values are not exact, so
    // the quadrature L2 error exposes the true discretization error and its
    // refinement rate. A parabolic strip would only reproduce roundoff.
    let errors = [
        quadrature_l2_error(8)?,
        quadrature_l2_error(16)?,
        quadrature_l2_error(32)?,
    ];
    for (index, error) in errors.iter().enumerate() {
        if !error.is_finite() {
            return Err(format!(
                "convergence L2 error at level {index} is not finite"
            ));
        }
    }
    if !(errors[1] < errors[0] && errors[2] < errors[1]) {
        return Err(format!(
            "L2 errors do not contract under refinement: {:.2e}, {:.2e}, {:.2e}",
            errors[0], errors[1], errors[2]
        ));
    }
    let rate_coarse = (errors[0] / errors[1]).log2();
    let rate_fine = (errors[1] / errors[2]).log2();
    println!(
        "Convergence (sinusoidal 2D): n=8 L2={:.2e}, n=16 L2={:.2e}, n=32 L2={:.2e}, rates: {:.2}, {:.2}",
        errors[0], errors[1], errors[2], rate_coarse, rate_fine
    );
    if !rate_fine.is_finite() || rate_fine < L2_RATE_MINIMUM {
        return Err(format!(
            "L2 refinement rate {rate_fine:.2} below minimum {L2_RATE_MINIMUM}"
        ));
    }
    if errors[2] >= FINE_L2_MAX_ERROR {
        return Err(format!(
            "fine-mesh L2 error {:.2e} exceeds bound {FINE_L2_MAX_ERROR:.1e}",
            errors[2]
        ));
    }
    Ok((errors[0], errors[1], errors[2], rate_fine))
}

fn test_thermal_1d_linear_solution() -> Result<f64, String> {
    let k = 2.0;
    let bc_left = 10.0;
    let bc_right = 30.0;
    let l = 1.0;

    let exact_solution = |x: f64| -> f64 { bc_left + (bc_right - bc_left) * x };

    let nx = 10;
    let ny = 2;
    let mesh = sonium_fem::mesh::generators::rectangular_mesh_triangles(0.0, l, 0.0, 0.1, nx, ny);

    let solution = assemble_and_solve_thermal(&mesh, k, bc_left, bc_right, |_, _| 0.0, &[])
        .map_err(|error| format!("1D linear solve failed: {error}"))?;

    write_thermal_vtu("1d_linear", &mesh, &solution);
    let mut max_err: f64 = 0.0;
    for (i, &t_fem) in solution.iter().enumerate().take(mesh.num_nodes()) {
        let x = mesh.nodes[i].x;
        let t_exact = exact_solution(x);
        max_err = max_err.max((t_fem - t_exact).abs());
    }

    if !max_err.is_finite() || max_err >= LINEAR_MAX_ERROR {
        return Err(format!(
            "Linear solution error too large: {max_err:.2e} (bound {LINEAR_MAX_ERROR:.1e})"
        ));
    }
    println!("1D linear: max nodal error={max_err:.2e}");
    Ok(max_err)
}

fn test_thermal_conduction_form_symmetry() -> Result<f64, String> {
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

    if !max_asym.is_finite() || max_asym >= SYMMETRY_TOLERANCE {
        return Err(format!(
            "Conduction form not symmetric: max |A[i,j] - A[j,i]| = {max_asym:.2e}"
        ));
    }
    println!("Conduction symmetry: max asymmetry={max_asym:.2e}");
    Ok(max_asym)
}

struct ThermalMetrics {
    mms_1d_error: f64,
    mms_2d_error: f64,
    coarse_l2: f64,
    medium_l2: f64,
    fine_l2: f64,
    l2_rate: f64,
    linear_error: f64,
    symmetry: f64,
}

fn run() -> Result<ThermalMetrics, String> {
    let mms_1d_error = test_thermal_mms_1d_parabolic()?;
    let mms_2d_error = test_thermal_mms_2d_sinusoidal()?;
    let (coarse_l2, medium_l2, fine_l2, l2_rate) = test_thermal_convergence_rate()?;
    let linear_error = test_thermal_1d_linear_solution()?;
    let symmetry = test_thermal_conduction_form_symmetry()?;
    Ok(ThermalMetrics {
        mms_1d_error,
        mms_2d_error,
        coarse_l2,
        medium_l2,
        fine_l2,
        l2_rate,
        linear_error,
        symmetry,
    })
}

fn main() {
    match run() {
        Ok(metrics) => {
            let record = QaResult::passed(CASE_ID)
                .requested_backend("cpu")
                .actual_backend("cpu")
                .metric("mms_1d_parabolic_error", metrics.mms_1d_error)
                .metric("mms_2d_sinusoidal_error", metrics.mms_2d_error)
                .metric("coarse_l2_error", metrics.coarse_l2)
                .metric("medium_l2_error", metrics.medium_l2)
                .metric("fine_l2_error", metrics.fine_l2)
                .metric("l2_refinement_rate", metrics.l2_rate)
                .metric("linear_solution_error", metrics.linear_error)
                .metric("conduction_symmetry", metrics.symmetry)
                .maximum("mms_1d_parabolic_error", MMS_1D_MAX_ERROR)
                .maximum("mms_2d_sinusoidal_error", MMS_2D_MAX_ERROR)
                .maximum("fine_l2_error", FINE_L2_MAX_ERROR)
                .minimum("l2_refinement_rate", L2_RATE_MINIMUM)
                .maximum("linear_solution_error", LINEAR_MAX_ERROR)
                .maximum("conduction_symmetry", SYMMETRY_TOLERANCE)
                .tolerance(
                    "P1 L2 refinement rate >= 1.5 with quadrature L2 error on sin(pi x) sin(pi y)",
                )
                .seed("fem-thermal-validation:cpu:v1");
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
            eprintln!("FEM THERMAL VALIDATION FAILED: {error}");
            println!("FAILED");
            std::process::exit(1);
        }
    }
}
