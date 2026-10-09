use sonium_fem::assembly::{TripletMatrix, assemble_bilinear};
use sonium_fem::basis::PolynomialDegree;
use sonium_fem::boundary::generic::{FemSystem, apply_dirichlet_values};
use sonium_fem::mesh::generators::rectangular_mesh_triangles;
use sonium_fem::physics::fluid::porous_flow::{DarcyMaterial, DarcyPressureForm, DarcyStorageForm};
use sonium_fem::spaces::{FunctionSpace, H1Space};
use sonium_solvers::{GmresConfig, gmres};

fn extend_triplet(target: &mut TripletMatrix<f64>, source: &TripletMatrix<f64>) {
    for i in 0..source.nnz() {
        target.push(source.rows[i], source.cols[i], source.values[i]);
    }
}

/// Exact Terzaghi solution for one-way drainage (top drained, bottom impermeable).
/// y is depth from bottom (0 to h).
fn terzaghi_exact(y: f64, h: f64, cv: f64, t: f64, n_terms: usize) -> f64 {
    let mut sum = 0.0;
    for n in (1..=2 * n_terms - 1).step_by(2) {
        let n_f = n as f64;
        let mode = (n - 1) / 2;
        let sign = if mode % 2 == 0 { 1.0 } else { -1.0 };
        sum += (sign * 4.0 / (n_f * std::f64::consts::PI))
            * (n_f * std::f64::consts::PI * y / (2.0 * h)).cos()
            * (-n_f * n_f * std::f64::consts::PI * std::f64::consts::PI * cv * t / (4.0 * h * h))
                .exp();
    }
    sum
}

/// Solve transient 1D consolidation with backward Euler.
#[allow(clippy::too_many_arguments)]
fn solve_terzaghi(
    permeability: f64,
    viscosity: f64,
    specific_storage: f64,
    h: f64,
    nx: usize,
    ny: usize,
    p0: f64,
    dt: f64,
    n_steps: usize,
) -> (Vec<f64>, Vec<f64>, f64) {
    let l = h / ny as f64; // thin strip
    let mesh = rectangular_mesh_triangles(0.0, l, 0.0, h, nx, ny);
    let space = H1Space::new(PolynomialDegree::P1);
    let n_dofs = space.num_global_dofs(&mesh);

    let material = DarcyMaterial::isotropic(permeability, viscosity, specific_storage);
    let pressure_form = DarcyPressureForm::new(&material);
    let storage_form = DarcyStorageForm::new(&material);

    let k_matrix = assemble_bilinear(&pressure_form, &space, &space, &mesh, 2);
    let s_matrix = assemble_bilinear(&storage_form, &space, &space, &mesh, 2);

    // Time-stepping matrix: S + dt*K
    let mut lhs = TripletMatrix::new(n_dofs, n_dofs);
    extend_triplet(&mut lhs, &s_matrix);
    for i in 0..k_matrix.nnz() {
        lhs.push(k_matrix.rows[i], k_matrix.cols[i], dt * k_matrix.values[i]);
    }

    let mut p = vec![p0; n_dofs];

    for _step in 0..n_steps {
        // RHS: S*p^n
        let s_csr = s_matrix.to_csr();
        let rhs = s_csr.matvec(&p);

        let mut system = FemSystem::new(lhs.clone(), rhs);
        // Dirichlet p=0 at top (marker 4)
        apply_dirichlet_values(&mut system, &mesh, &space, 4, &|_, _, _| 0.0, None);
        // Impermeable at bottom (marker 3) is natural Neumann (no flux)

        let csr = system.matrix.to_csr();
        let result = gmres(
            &csr,
            &system.rhs,
            &GmresConfig {
                max_iterations: 2000,
                restart: 200,
                tolerance: 1e-12,
                print_interval: 0,
                convergence_check_interval: 1,
                min_restart: None,
                max_restart: None,
                exact_residual_check_interval: 0,
            },
        );
        assert!(result.converged, "GMRES did not converge at step {}", _step);
        p = result.x;

        // Rebuild lhs since apply_dirichlet_values modifies it
        lhs = TripletMatrix::new(n_dofs, n_dofs);
        extend_triplet(&mut lhs, &s_matrix);
        for i in 0..k_matrix.nnz() {
            lhs.push(k_matrix.rows[i], k_matrix.cols[i], dt * k_matrix.values[i]);
        }
    }

    let mut y_vals = Vec::new();
    let mut p_vals = Vec::new();
    for (i, node) in mesh.nodes.iter().enumerate() {
        y_vals.push(node.y);
        p_vals.push(p[i]);
    }

    let cv = permeability / (viscosity * specific_storage);
    (y_vals, p_vals, cv)
}

/// Test 1: At t=0, pressure should be uniform (initial condition).
fn test_initial_condition() {
    let permeability = 1e-10;
    let viscosity = 1e-3;
    let specific_storage = 1e-4;
    let h = 1.0;
    let p0 = 100.0;
    let dt = 0.01;
    let n_steps = 0; // Just initial condition

    let (y_vals, p_vals, _) = solve_terzaghi(
        permeability,
        viscosity,
        specific_storage,
        h,
        1,
        20,
        p0,
        dt,
        n_steps,
    );

    let mut max_err: f64 = 0.0;
    for i in 0..y_vals.len() {
        max_err = max_err.max((p_vals[i] - p0).abs());
    }

    println!("Initial condition: max_err={:.4e}", max_err);
    assert!(
        max_err < 1e-10,
        "Initial condition not uniform: max_err={}",
        max_err
    );
}

/// Test 2: After one step, pressure near top should decrease.
fn test_pressure_decay() {
    // Scaled parameters for unit diffusivity
    let permeability = 1.0;
    let viscosity = 1.0;
    let specific_storage = 1.0;
    let h = 1.0;
    let p0 = 100.0;
    let dt = 0.01;
    let n_steps = 1;

    let (y_vals, p_vals, cv) = solve_terzaghi(
        permeability,
        viscosity,
        specific_storage,
        h,
        1,
        40,
        p0,
        dt,
        n_steps,
    );

    let time_factor = cv * dt / (h * h);
    println!("Time factor: {:.4e}", time_factor);

    // Top (y=h) should be close to 0
    let top_idx = y_vals
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .unwrap()
        .0;
    println!("Top pressure: {:.4e} (expected ~0)", p_vals[top_idx]);
    assert!(
        p_vals[top_idx].abs() < 5.0,
        "Top pressure should be near zero"
    );

    // Bottom (y=0) should still be close to p0
    let bot_idx = y_vals
        .iter()
        .enumerate()
        .min_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .unwrap()
        .0;
    println!(
        "Bottom pressure: {:.4e} (expected ~{:.1})",
        p_vals[bot_idx], p0
    );
    assert!(
        p_vals[bot_idx] > 0.8 * p0,
        "Bottom pressure should still be high"
    );
}

/// Test 3: Compare with exact Terzaghi solution at late time.
fn test_exact_solution() {
    // Scaled parameters for unit diffusivity
    let permeability = 1.0;
    let viscosity = 1.0;
    let specific_storage = 1.0;
    let h = 1.0;
    let p0 = 100.0;
    let dt = 0.001;
    let n_steps = 100;

    let (y_vals, p_vals, cv) = solve_terzaghi(
        permeability,
        viscosity,
        specific_storage,
        h,
        1,
        40,
        p0,
        dt,
        n_steps,
    );

    let t = dt * n_steps as f64;
    let time_factor = cv * t / (h * h);
    println!("Time factor (late): {:.4e}", time_factor);

    let mut max_err: f64 = 0.0;
    let mut max_rel_err: f64 = 0.0;
    for i in 0..y_vals.len() {
        let exact = p0 * terzaghi_exact(y_vals[i], h, cv, t, 20);
        let err = (p_vals[i] - exact).abs();
        let rel_err = if exact.abs() > 1.0 {
            err / exact.abs()
        } else {
            err
        };
        max_err = max_err.max(err);
        max_rel_err = max_rel_err.max(rel_err);
    }

    println!(
        "Exact comparison: max_err={:.4e}, max_rel_err={:.4e}",
        max_err, max_rel_err
    );
    assert!(
        max_rel_err < 0.01,
        "Late-time error too large: {:.4e}",
        max_rel_err
    );
}

/// Test 4: Hydraulic diffusivity formula.
fn test_hydraulic_diffusivity() {
    let k = 1e-10;
    let mu = 1e-3;
    let ss = 1e-4;
    let material = DarcyMaterial::isotropic(k, mu, ss);
    let cv = material.hydraulic_diffusivity_isotropic();
    let expected = k / (mu * ss);
    println!(
        "Hydraulic diffusivity: {:.4e} (expected {:.4e})",
        cv, expected
    );
    assert!((cv - expected).abs() < 1e-20);
}

fn main() {
    let result = std::panic::catch_unwind(|| {
        println!("=== QA: porous flow Terzaghi consolidation ===");
        test_initial_condition();
        test_pressure_decay();
        test_exact_solution();
        test_hydraulic_diffusivity();
        println!("SUCCESS");
    });
    if result.is_err() {
        eprintln!("FAILED");
        std::process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn terzaghi_series_recovers_uniform_initial_pressure_in_the_interior() {
        for y in [0.1, 0.25, 0.5, 0.75, 0.9] {
            let normalized_pressure = terzaghi_exact(y, 1.0, 1.0, 0.0, 500);
            assert!(
                (normalized_pressure - 1.0).abs() < 5.0e-3,
                "initial Terzaghi series at y={y} is {normalized_pressure}"
            );
        }
    }
}
