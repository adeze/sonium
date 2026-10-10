//! QA: 45-degree large-rotation cantilever bend vs the elastica.
//!
//! A cantilever under end moment bends to constant curvature; the tip lands
//! on the closed-form elastica circle. Newton iteration with load stepping
//! drives the 3-D corotational form through a dense solve.

// Rust guideline compliant 2026-02-21

use sonium_fem::forms::NonlinearElementForm;
use sonium_fem::physics::solid::beam::{BeamElement3D, CorotationalBeamForm3D, Section3D};

/// Solve a dense system in place with partial pivoting.
fn dense_solve(matrix: &mut [f64], rhs: &mut [f64], n: usize) {
    for col in 0..n {
        let mut pivot = col;
        let mut best = matrix[col * n + col].abs();
        for row in col + 1..n {
            let candidate = matrix[row * n + col].abs();
            if candidate > best {
                best = candidate;
                pivot = row;
            }
        }
        assert!(best > 1.0e-300, "singular Newton matrix");
        if pivot != col {
            for k in col..n {
                matrix.swap(col * n + k, pivot * n + k);
            }
            rhs.swap(col, pivot);
        }
        let diag = matrix[col * n + col];
        for row in col + 1..n {
            let factor = matrix[row * n + col] / diag;
            matrix[row * n + col] = 0.0;
            for k in col + 1..n {
                matrix[row * n + k] -= factor * matrix[col * n + k];
            }
            rhs[row] -= factor * rhs[col];
        }
    }
    for row in (0..n).rev() {
        let mut sum = rhs[row];
        for k in row + 1..n {
            sum -= matrix[row * n + k] * rhs[k];
        }
        rhs[row] = sum / matrix[row * n + row];
    }
}

/// Newton solve with fixed DOFs eliminated, returning the full solution.
fn newton_solve(
    form: &CorotationalBeamForm3D,
    external: &[f64],
    fixed: &[usize],
    initial: &[f64],
    max_iterations: usize,
    tolerance: f64,
) -> Vec<f64> {
    let n = form.num_global_dofs();
    let is_fixed = {
        let mut mask = vec![false; n];
        for &dof in fixed {
            mask[dof] = true;
        }
        mask
    };
    let free: Vec<usize> = (0..n).filter(|dof| !is_fixed[*dof]).collect();
    let mut u = initial.to_vec();
    for _ in 0..max_iterations {
        let mut residual = vec![0.0; n];
        let mut tangent = vec![0.0; n * n];
        for elem in 0..form.num_elements() {
            let map = form.local_to_global(elem);
            let local_r = form.element_residual(elem, &u);
            let local_k = form.element_tangent(elem, &u);
            for (i, &gi) in map.iter().enumerate() {
                residual[gi] += local_r[i];
                for (j, &gj) in map.iter().enumerate() {
                    tangent[gi * n + gj] += local_k[i * map.len() + j];
                }
            }
        }
        for i in 0..n {
            residual[i] -= external[i];
        }
        let free_norm: f64 = free
            .iter()
            .map(|&dof| residual[dof] * residual[dof])
            .sum::<f64>()
            .sqrt();
        let load_norm: f64 = external
            .iter()
            .map(|f| f * f)
            .sum::<f64>()
            .sqrt()
            .max(1.0e-30);
        if free_norm / load_norm < tolerance {
            return u;
        }
        let m = free.len();
        let mut matrix = vec![0.0; m * m];
        let mut rhs = vec![0.0; m];
        for (i, &gi) in free.iter().enumerate() {
            rhs[i] = -residual[gi];
            for (j, &gj) in free.iter().enumerate() {
                matrix[i * m + j] = tangent[gi * n + gj];
            }
        }
        dense_solve(&mut matrix, &mut rhs, m);
        for (i, &gi) in free.iter().enumerate() {
            u[gi] += rhs[i];
        }
    }
    panic!("Newton iteration failed to converge");
}

fn test_45_degree_bend() {
    let length = 10.0;
    let n_elements = 20;
    // Slender Euler-Bernoulli section so shear stays out of the comparison.
    let section = Section3D::rectangular(1.2e6, 0.0, 1.0, 0.1, 1.0, false);
    let dx = length / n_elements as f64;
    let mut positions = Vec::with_capacity(n_elements + 1);
    for i in 0..=n_elements {
        positions.push([i as f64 * dx, 0.0, 0.0]);
    }
    let elements: Vec<BeamElement3D> = (0..n_elements)
        .map(|i| {
            BeamElement3D::new(
                [i, i + 1],
                positions[i],
                positions[i + 1],
                section,
                [0.0, 1.0, 0.0],
            )
        })
        .collect();
    let form = CorotationalBeamForm3D::new(elements, positions, (n_elements + 1) * 6);
    // End moment for 45 degrees of tip rotation: theta = M L / EI.
    let theta = std::f64::consts::FRAC_PI_4;
    let moment = section.eiz() * theta / length;
    let n_dofs = form.num_global_dofs();
    let mut external = vec![0.0; n_dofs];
    external[n_elements * 6 + 5] = moment;
    let fixed = vec![0, 1, 2, 3, 4, 5];
    // Load stepping keeps each Newton solve in its radius of convergence.
    let steps = 10;
    let mut u = vec![0.0; n_dofs];
    for step in 1..=steps {
        let scale = step as f64 / steps as f64;
        let scaled: Vec<f64> = external.iter().map(|f| f * scale).collect();
        u = newton_solve(&form, &scaled, &fixed, &u, 25, 1.0e-10);
    }
    let tip_x = length + u[n_elements * 6];
    let tip_y = u[n_elements * 6 + 1];
    // Elastica circle: constant curvature kappa = theta / L.
    let kappa = theta / length;
    let exact_x = theta.sin() / kappa;
    let exact_y = (1.0 - theta.cos()) / kappa;
    let error_x = (tip_x - exact_x).abs() / exact_x;
    let error_y = (tip_y - exact_y).abs() / exact_y;
    println!(
        "45-degree bend: FEM=({tip_x:.10e}, {tip_y:.10e}) exact=({exact_x:.10e}, {exact_y:.10e})"
    );
    println!(
        "tip rotation={:.6e} target={:.6e}",
        u[n_elements * 6 + 5],
        theta
    );
    assert!(error_x < 5.0e-3, "tip x error {error_x:.3e}");
    assert!(error_y < 5.0e-3, "tip y error {error_y:.3e}");
    println!(
        "GOLDEN beam_45_bend fem_x={tip_x:.17e} fem_y={tip_y:.17e} exact_x={exact_x:.17e} exact_y={exact_y:.17e}"
    );
}

fn main() {
    let result = std::panic::catch_unwind(test_45_degree_bend);
    match result {
        Ok(()) => println!("SUCCESS"),
        Err(_) => {
            println!("FAILED");
            std::process::exit(1);
        }
    }
}
