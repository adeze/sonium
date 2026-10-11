//! Beam validation tests
//!
//! - Convergence test: Verify O(h⁴) convergence for Euler-Bernoulli beam
//! - Simply supported beam: Analytical solution verification
//! - Cantilever natural frequencies: Eigenvalue problem
//! - Mass matrix symmetry
//! - Axial deformation
//! - Timoshenko beam (shear deformation)
//! - 3-point bending

use sonium_fem::physics::solid::beam::{
    BeamElement2D, BeamSection, assemble_beam_mass, assemble_beam_stiffness,
};
use sonium_qa::QaResult;
use sonium_qa::analytical::{
    beam_euler_cantilever_point, beam_euler_ss_central, beam_euler_ss_uniform,
};
use sonium_solvers::{CsrMatrix, GmresConfig, LinearOperator, TripletMatrix, gmres};

const CASE_ID: &str = "fem.beam-validation.v1";
// Minimum accepted static convergence rate (Hermitian beams give O(h^4);
// cubic exactness may also reproduce the tip deflection to roundoff).
const CONVERGENCE_MIN_RATE: f64 = 3.0;
const CONVERGENCE_EXACT_ERROR: f64 = 1e-9;
// Simply supported uniform-load midpoint error bound.
const SIMPLY_SUPPORTED_MAX_ERROR: f64 = 0.01;
// Consistent mass symmetry bound.
const MASS_SYMMETRY_TOLERANCE: f64 = 1e-10;
// Axial extension error bound.
const AXIAL_MAX_ERROR: f64 = 1e-3;
// Three-point bending midpoint error bound.
const BENDING_MAX_ERROR: f64 = 0.02;
// First cantilever root of cos(beta L) cosh(beta L) = -1.
const BETA1L: f64 = 1.87510406871196;
// Solved first natural frequency must match Euler-Bernoulli theory.
// (Sibling modal analysis reaches ~1e-8 at 40 elements; this bound keeps a
// large margin while still proving the eigen-solve, not just assembly.)
const NATURAL_FREQUENCY_REL_TOLERANCE: f64 = 1e-6;
// Independent relative eigen-residual ||K phi - lambda M phi|| / ||K phi||.
const EIGEN_RESIDUAL_TOLERANCE: f64 = 1e-6;
const INVERSE_ITERATION_MAX: usize = 100;
const INVERSE_ITERATION_TOLERANCE: f64 = 1e-10;

fn apply_dirichlet_csr(
    csr: &mut CsrMatrix<f64>,
    rhs: &mut [f64],
    prescribed_dofs: &[usize],
    prescribed_values: &[f64],
) {
    assert_eq!(
        prescribed_dofs.len(),
        prescribed_values.len(),
        "Dirichlet DOF/value length mismatch"
    );

    for &dof in prescribed_dofs.iter() {
        let range = csr.row_range(dof);
        for i in range {
            csr.values[i] = 0.0;
        }
        let diag_pos = csr.row_entries(dof).position(|(j, _)| j == dof);
        if let Some(idx) = diag_pos {
            let global_idx = csr.row_ptrs[dof] + idx;
            csr.values[global_idx] = 1.0;
        }
    }
    for (dof, &val) in prescribed_dofs.iter().zip(prescribed_values.iter()) {
        rhs[*dof] = val;
    }
}

fn gmres_config(max_iterations: usize) -> GmresConfig<f64> {
    GmresConfig {
        max_iterations,
        restart: 200,
        tolerance: 1e-10,
        print_interval: 0,
        convergence_check_interval: 1,
        min_restart: None,
        max_restart: None,
        exact_residual_check_interval: 0,
    }
}

fn test_beam_cantilever_convergence() -> Result<(f64, f64, f64), String> {
    let l_total: f64 = 1.0;
    let p_load: f64 = 1000.0;
    let e_mod: f64 = 200e9;
    let width: f64 = 0.01;
    let height: f64 = 0.02;
    let section = BeamSection::rectangular(e_mod, width, height, 7800.0);
    let i_inertia = section.moment_of_inertia;

    let exact_delta = beam_euler_cantilever_point(l_total, l_total, e_mod, i_inertia, p_load);

    let mut errors = Vec::new();
    for &n_elements in &[5, 10, 20] {
        let dx = l_total / n_elements as f64;

        let elements: Vec<BeamElement2D> = (0..n_elements)
            .map(|i| {
                let x1 = i as f64 * dx;
                let x2 = (i + 1) as f64 * dx;
                BeamElement2D::new([i, i + 1], x1, 0.0, x2, 0.0, section)
            })
            .collect();

        let n_nodes = n_elements + 1;
        let n_dofs = n_nodes * 3;

        let mut csr = assemble_beam_stiffness(&elements, n_dofs).to_csr();
        let mut rhs = vec![0.0; n_dofs];
        let last_node_v_dof = (n_nodes - 1) * 3 + 1;
        rhs[last_node_v_dof] = -p_load;

        let prescribed_dofs = vec![0, 1, 2];
        let prescribed_values = vec![0.0, 0.0, 0.0];
        apply_dirichlet_csr(&mut csr, &mut rhs, &prescribed_dofs, &prescribed_values);

        let result = gmres(&csr, &rhs, &gmres_config(1000));
        if !result.converged {
            return Err(format!(
                "cantilever convergence solve at {n_elements} elements did not converge"
            ));
        }
        let delta = result.x[last_node_v_dof].abs();
        errors.push((delta - exact_delta).abs() / exact_delta);
    }

    let rate_1 = (errors[1] / errors[0]).log(std::f64::consts::LN_10)
        / (5.0_f64 / 10.0).log(std::f64::consts::LN_10);
    let rate_2 = (errors[2] / errors[1]).log(std::f64::consts::LN_10)
        / (10.0_f64 / 20.0).log(std::f64::consts::LN_10);

    println!(
        "Beam convergence: 5 elem err={:.2e}, 10 elem err={:.2e}, 20 elem err={:.2e}, rates: {:.2}, {:.2}",
        errors[0], errors[1], errors[2], rate_1, rate_2
    );

    let all_exact = errors.iter().all(|&error| error < CONVERGENCE_EXACT_ERROR);
    if !(all_exact || (rate_1 > CONVERGENCE_MIN_RATE && rate_2 > CONVERGENCE_MIN_RATE)) {
        return Err(format!(
            "convergence rate too low: {rate_1:.2}, {rate_2:.2}; errors: {:.2e}, {:.2e}, {:.2e}",
            errors[0], errors[1], errors[2]
        ));
    }
    Ok((errors[2], rate_1, rate_2))
}

fn test_beam_simply_supported() -> Result<f64, String> {
    let l: f64 = 2.0;
    let q_load: f64 = 500.0;
    let e_mod: f64 = 200e9;
    let width: f64 = 0.02;
    let height: f64 = 0.04;
    let section = BeamSection::rectangular(e_mod, width, height, 7800.0);
    let i_inertia = section.moment_of_inertia;

    let n_elements = 20;
    let dx = l / n_elements as f64;

    let elements: Vec<BeamElement2D> = (0..n_elements)
        .map(|i| {
            let x1 = i as f64 * dx;
            let x2 = (i + 1) as f64 * dx;
            BeamElement2D::new([i, i + 1], x1, 0.0, x2, 0.0, section)
        })
        .collect();

    let n_nodes = n_elements + 1;
    let n_dofs = n_nodes * 3;

    let mut csr = assemble_beam_stiffness(&elements, n_dofs).to_csr();
    let mut rhs = vec![0.0; n_dofs];

    for node in 0..n_nodes {
        let v_dof = node * 3 + 1;
        let t_dof = node * 3 + 2;
        if node == 0 {
            rhs[v_dof] = -q_load * dx / 2.0;
            rhs[t_dof] = -q_load * dx * dx / 12.0;
        } else if node == n_nodes - 1 {
            rhs[v_dof] = -q_load * dx / 2.0;
            rhs[t_dof] = q_load * dx * dx / 12.0;
        } else {
            rhs[v_dof] = -q_load * dx;
        }
    }

    let prescribed_dofs: Vec<usize> = vec![0, 1, (n_nodes - 1) * 3 + 1];
    let prescribed_values: Vec<f64> = vec![0.0; 3];
    apply_dirichlet_csr(&mut csr, &mut rhs, &prescribed_dofs, &prescribed_values);

    let result = gmres(&csr, &rhs, &gmres_config(2000));
    if !result.converged {
        return Err(format!(
            "simply supported solve did not converge: residual={:.2e}",
            result.residual
        ));
    }

    let mid_node = n_nodes / 2;
    let mid_delta = result.x[mid_node * 3 + 1].abs();
    let exact_delta = beam_euler_ss_uniform(mid_node as f64 * dx, l, e_mod, i_inertia, q_load);

    let error = (mid_delta - exact_delta).abs() / exact_delta;
    println!(
        "Simply supported: FEM delta={:.6e}, Exact={:.6e}, Error={:.2}%",
        mid_delta,
        exact_delta,
        error * 100.0
    );

    if !error.is_finite() || error >= SIMPLY_SUPPORTED_MAX_ERROR {
        return Err(format!(
            "simply supported error {error:.2e} exceeds bound {SIMPLY_SUPPORTED_MAX_ERROR:.1e}"
        ));
    }
    Ok(error)
}

fn test_beam_mass_matrix_symmetry() -> Result<f64, String> {
    let section = BeamSection::rectangular(200e9, 0.01, 0.02, 7800.0);
    let n_elements = 10;
    let dx = 1.0 / n_elements as f64;

    let elements: Vec<BeamElement2D> = (0..n_elements)
        .map(|i| {
            BeamElement2D::new(
                [i, i + 1],
                i as f64 * dx,
                0.0,
                (i + 1) as f64 * dx,
                0.0,
                section,
            )
        })
        .collect();

    let n_dofs = (n_elements + 1) * 3;
    let mass = assemble_beam_mass(&elements, n_dofs).to_csr();

    let n = mass.num_rows;
    let mut max_asym = 0.0_f64;
    for i in 0..n {
        for k in mass.row_ptrs[i]..mass.row_ptrs[i + 1] {
            let j = mass.col_indices[k];
            let aij = mass.values[k];
            let aji = (mass.row_ptrs[j]..mass.row_ptrs[j + 1])
                .find(|&kk| mass.col_indices[kk] == i)
                .map(|kk| mass.values[kk])
                .unwrap_or(0.0);
            max_asym = max_asym.max((aij - aji).abs());
        }
    }

    println!("Mass symmetry: max asymmetry={max_asym:.2e}");
    if !max_asym.is_finite() || max_asym >= MASS_SYMMETRY_TOLERANCE {
        return Err(format!(
            "mass matrix not symmetric: {max_asym:.2e} (bound {MASS_SYMMETRY_TOLERANCE:.1e})"
        ));
    }
    Ok(max_asym)
}

fn test_beam_axial_deformation() -> Result<f64, String> {
    let l: f64 = 1.0;
    let e_mod: f64 = 200e9;
    let area: f64 = 0.001;
    let section = BeamSection::euler_bernoulli(e_mod, 1e-8, area, 7800.0);

    let n_elements = 5;
    let dx = l / n_elements as f64;

    let elements: Vec<BeamElement2D> = (0..n_elements)
        .map(|i| {
            BeamElement2D::new(
                [i, i + 1],
                i as f64 * dx,
                0.0,
                (i + 1) as f64 * dx,
                0.0,
                section,
            )
        })
        .collect();

    let n_dofs = (n_elements + 1) * 3;
    let mut csr = assemble_beam_stiffness(&elements, n_dofs).to_csr();
    let mut rhs = vec![0.0; n_dofs];

    let pull_force = 1000.0;
    let loaded_u_dof = n_elements * 3;
    rhs[loaded_u_dof] = pull_force;

    apply_dirichlet_csr(&mut csr, &mut rhs, &[0, 1, 2], &[0.0, 0.0, 0.0]);

    let result = gmres(&csr, &rhs, &gmres_config(2000));
    if !result.converged {
        return Err(format!(
            "axial solve did not converge: residual={:.2e}",
            result.residual
        ));
    }

    let extension = result.x[loaded_u_dof].abs();
    let exact_ext = pull_force * l / (e_mod * area);

    let error = (extension - exact_ext).abs() / exact_ext;
    println!(
        "Axial: FEM={:.6e}, Exact={:.6e}, Error={:.2}%",
        extension,
        exact_ext,
        error * 100.0
    );

    if !error.is_finite() || error >= AXIAL_MAX_ERROR {
        return Err(format!(
            "axial error too large: {error:.2e} (bound {AXIAL_MAX_ERROR:.1e})"
        ));
    }
    Ok(error)
}

/// Extract the free-DOF submatrix after clamping the root node.
fn extract_free_submatrix(csr: &CsrMatrix<f64>, free_dofs: &[usize]) -> CsrMatrix<f64> {
    let n = free_dofs.len();
    let mut triplet = TripletMatrix::new(n, n);
    for (new_i, &old_i) in free_dofs.iter().enumerate() {
        for (old_j, val) in csr.row_entries(old_i) {
            if let Some(new_j) = free_dofs.iter().position(|&d| d == old_j) {
                triplet.push(new_i, new_j, val);
            }
        }
    }
    triplet.to_csr()
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum()
}

/// Inverse iteration for the smallest eigenpair of K phi = lambda M phi.
/// Returns (lambda, phi, iterations). The eigenvalue is the Rayleigh quotient
/// of the converged vector, so the residual check below is independent.
fn inverse_iteration_first_mode(
    k: &CsrMatrix<f64>,
    m: &CsrMatrix<f64>,
) -> Result<(f64, Vec<f64>, usize), String> {
    let n = k.num_rows;
    let n_nodes = n / 3;
    // Transverse sine start on the w DOFs only.
    let mut x = vec![0.0; n];
    for j in 0..n_nodes {
        x[3 * j + 1] = (std::f64::consts::PI * (j + 1) as f64 / (n_nodes + 1) as f64).sin();
    }
    let norm = dot(&x, &x).sqrt();
    for value in x.iter_mut() {
        *value /= norm;
    }

    // Inner tolerance matches the sibling modal analysis: the beam stiffness
    // matrix is too ill-conditioned for 1e-10, while 1e-8 still resolves the
    // first eigenpair far below the frequency gate.
    let inner = GmresConfig {
        max_iterations: 1000,
        restart: 300,
        tolerance: 1e-8,
        print_interval: 0,
        convergence_check_interval: 1,
        min_restart: None,
        max_restart: None,
        exact_residual_check_interval: 0,
    };
    for iteration in 0..INVERSE_ITERATION_MAX {
        let mx = m.apply(&x);
        let result = gmres(k, &mx, &inner);
        if !result.converged {
            return Err(format!(
                "inner linear solve failed in inverse iteration {iteration}: residual={:.2e}",
                result.residual
            ));
        }
        let y = result.x;
        let y_norm = dot(&y, &y).sqrt();
        let y_scaled: Vec<f64> = y.iter().map(|v| v / y_norm).collect();
        let change: f64 = x
            .iter()
            .zip(y_scaled.iter())
            .map(|(&a, &b)| (a - b).abs())
            .sum();
        x = y_scaled;
        if change < INVERSE_ITERATION_TOLERANCE {
            // Rayleigh quotient of the converged vector (scale invariant).
            let kx = k.apply(&x);
            let mx = m.apply(&x);
            let denominator = dot(&x, &mx);
            if !denominator.is_finite() || denominator.abs() < 1e-30 {
                return Err("degenerate Rayleigh quotient in inverse iteration".to_string());
            }
            let lambda = dot(&x, &kx) / denominator;
            return Ok((lambda, x, iteration + 1));
        }
    }
    Err("inverse iteration did not converge within the iteration budget".to_string())
}

fn test_beam_natural_frequencies() -> Result<(f64, f64, f64), String> {
    let l: f64 = 1.0;
    let e_mod: f64 = 200e9;
    let width: f64 = 0.01;
    let height: f64 = 0.02;
    let rho: f64 = 7800.0;
    let section = BeamSection::rectangular(e_mod, width, height, rho);
    let i_inertia = section.moment_of_inertia;
    let area = section.area;

    let n_elements = 50;
    let dx = l / n_elements as f64;

    let elements: Vec<BeamElement2D> = (0..n_elements)
        .map(|i| {
            BeamElement2D::new(
                [i, i + 1],
                i as f64 * dx,
                0.0,
                (i + 1) as f64 * dx,
                0.0,
                section,
            )
        })
        .collect();

    let n_dofs = (n_elements + 1) * 3;
    let k_csr = assemble_beam_stiffness(&elements, n_dofs).to_csr();
    let m_csr = assemble_beam_mass(&elements, n_dofs).to_csr();

    if k_csr.nnz() == 0 || m_csr.nnz() == 0 {
        return Err("assembled beam matrices are empty".to_string());
    }

    // Clamp the root node, then solve the free-vibration eigenproblem.
    let constrained = [0, 1, 2];
    let free_dofs: Vec<usize> = (0..n_dofs).filter(|d| !constrained.contains(d)).collect();
    let k_free = extract_free_submatrix(&k_csr, &free_dofs);
    let m_free = extract_free_submatrix(&m_csr, &free_dofs);

    let (lambda, phi, iterations) = inverse_iteration_first_mode(&k_free, &m_free)?;
    let omega_fem = lambda.max(0.0).sqrt();

    // Correct Euler-Bernoulli formula: omega = beta^2 sqrt(EI/(rho A)) / L^2
    // with beta L = 1.8751... (beta enters squared, not fourth-powered).
    let omega_exact = BETA1L.powi(2) * ((e_mod * i_inertia) / (rho * area)).sqrt() / l.powi(2);
    let frequency_error = (omega_fem - omega_exact).abs() / omega_exact;

    // Independent eigen-residual from the assembled operators.
    let k_phi = k_free.apply(&phi);
    let m_phi = m_free.apply(&phi);
    let residual_norm = k_phi
        .iter()
        .zip(m_phi.iter())
        .map(|(&kv, &mv)| (kv - lambda * mv).powi(2))
        .sum::<f64>()
        .sqrt();
    let k_norm = k_phi.iter().map(|v| v.powi(2)).sum::<f64>().sqrt();
    let residual = residual_norm / k_norm;

    println!("Beam natural frequency analysis:");
    println!("  Cantilever analytical: omega_1 = (beta_1 L)^2 sqrt(EI/(rho A L^4))");
    println!("  Expected first frequency: {omega_exact:.2} rad/s");
    println!(
        "  FEM first frequency: {omega_fem:.6e} rad/s, rel error={frequency_error:.2e}, residual={residual:.2e}, iterations={iterations}"
    );

    if !frequency_error.is_finite() || frequency_error >= NATURAL_FREQUENCY_REL_TOLERANCE {
        return Err(format!(
            "first natural frequency error {frequency_error:.2e} exceeds bound {NATURAL_FREQUENCY_REL_TOLERANCE:.1e}"
        ));
    }
    if !residual.is_finite() || residual >= EIGEN_RESIDUAL_TOLERANCE {
        return Err(format!(
            "eigen-residual {residual:.2e} exceeds bound {EIGEN_RESIDUAL_TOLERANCE:.1e}"
        ));
    }
    Ok((frequency_error, residual, omega_fem))
}

fn test_beam_three_point_bending() -> Result<f64, String> {
    let l: f64 = 3.0;
    let p_load: f64 = 1000.0;
    let e_mod: f64 = 200e9;
    let width: f64 = 0.02;
    let height: f64 = 0.04;
    let section = BeamSection::rectangular(e_mod, width, height, 7800.0);
    let i_inertia = section.moment_of_inertia;

    let n_elements = 30;
    let dx = l / n_elements as f64;

    let elements: Vec<BeamElement2D> = (0..n_elements)
        .map(|i| {
            BeamElement2D::new(
                [i, i + 1],
                i as f64 * dx,
                0.0,
                (i + 1) as f64 * dx,
                0.0,
                section,
            )
        })
        .collect();

    let n_nodes = n_elements + 1;
    let n_dofs = n_nodes * 3;

    let mut csr = assemble_beam_stiffness(&elements, n_dofs).to_csr();
    let mut rhs = vec![0.0; n_dofs];

    let mid_node = n_nodes / 2;
    let v_dof = mid_node * 3 + 1;
    rhs[v_dof] = -p_load;

    let prescribed_dofs: Vec<usize> = vec![0, 1, (n_nodes - 1) * 3 + 1];
    let prescribed_values: Vec<f64> = vec![0.0; 3];
    apply_dirichlet_csr(&mut csr, &mut rhs, &prescribed_dofs, &prescribed_values);

    let result = gmres(&csr, &rhs, &gmres_config(2000));
    if !result.converged {
        return Err(format!(
            "three-point bending solve did not converge: residual={:.2e}",
            result.residual
        ));
    }

    let mid_delta = result.x[v_dof].abs();
    let exact_delta = beam_euler_ss_central(l / 2.0, l, e_mod, i_inertia, p_load);

    let error = (mid_delta - exact_delta).abs() / exact_delta;
    println!(
        "3-point bending: FEM={:.6e}, Exact={:.6e}, Error={:.2}%",
        mid_delta,
        exact_delta,
        error * 100.0
    );

    if !error.is_finite() || error >= BENDING_MAX_ERROR {
        return Err(format!(
            "three-point bending error {error:.2e} exceeds bound {BENDING_MAX_ERROR:.1e}"
        ));
    }
    Ok(error)
}

struct BeamMetrics {
    convergence_error: f64,
    convergence_rate_1: f64,
    convergence_rate_2: f64,
    simply_supported_error: f64,
    mass_asymmetry: f64,
    axial_error: f64,
    frequency_error: f64,
    eigen_residual: f64,
    first_frequency: f64,
    bending_error: f64,
}

fn run() -> Result<BeamMetrics, String> {
    let (convergence_error, convergence_rate_1, convergence_rate_2) =
        test_beam_cantilever_convergence()?;
    let simply_supported_error = test_beam_simply_supported()?;
    let mass_asymmetry = test_beam_mass_matrix_symmetry()?;
    let axial_error = test_beam_axial_deformation()?;
    let (frequency_error, eigen_residual, first_frequency) = test_beam_natural_frequencies()?;
    let bending_error = test_beam_three_point_bending()?;
    Ok(BeamMetrics {
        convergence_error,
        convergence_rate_1,
        convergence_rate_2,
        simply_supported_error,
        mass_asymmetry,
        axial_error,
        frequency_error,
        eigen_residual,
        first_frequency,
        bending_error,
    })
}

fn main() {
    match run() {
        Ok(metrics) => {
            let record = QaResult::passed(CASE_ID)
                .requested_backend("cpu")
                .actual_backend("cpu")
                .metric("convergence_error", metrics.convergence_error)
                .metric("convergence_rate_1", metrics.convergence_rate_1)
                .metric("convergence_rate_2", metrics.convergence_rate_2)
                .metric("simply_supported_error", metrics.simply_supported_error)
                .metric("mass_asymmetry", metrics.mass_asymmetry)
                .metric("axial_error", metrics.axial_error)
                .metric("first_frequency_error", metrics.frequency_error)
                .metric("first_frequency_residual", metrics.eigen_residual)
                .metric("first_frequency_rad_s", metrics.first_frequency)
                .metric("bending_error", metrics.bending_error)
                .maximum("simply_supported_error", SIMPLY_SUPPORTED_MAX_ERROR)
                .maximum("mass_asymmetry", MASS_SYMMETRY_TOLERANCE)
                .maximum("axial_error", AXIAL_MAX_ERROR)
                .maximum("first_frequency_error", NATURAL_FREQUENCY_REL_TOLERANCE)
                .maximum("first_frequency_residual", EIGEN_RESIDUAL_TOLERANCE)
                .maximum("bending_error", BENDING_MAX_ERROR)
                .tolerance(
                    "solved first cantilever frequency matches beta^2 Euler-Bernoulli theory",
                )
                .seed("fem-beam-validation:cpu:v1");
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
            eprintln!("FEM BEAM VALIDATION FAILED: {error}");
            println!("FAILED");
            std::process::exit(1);
        }
    }
}
