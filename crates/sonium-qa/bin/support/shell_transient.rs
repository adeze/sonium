//! Full six-DOF native shell histories for matching external elastic fixtures.

// Rust guideline compliant 2026-02-21

use sonium_fem::assembly::TripletMatrix;
use sonium_fem::mesh::Point;
use sonium_fem::physics::solid::shell::{QuadShellElement, ShellElement, ShellSection};
use sonium_fem::solver::implicit_dynamics::{
    ImplicitDynamicsConfig, ImplicitDynamicsIntegrator, ImplicitDynamicsSystem,
};
use sonium_qa::rad::RadExtendedCase;
use sonium_solvers::CsrMatrix;

pub(super) struct ShellHistory {
    pub times: Vec<f64>,
    pub observations: Vec<Vec<f64>>,
    pub max_true_dynamic_residual: f64,
    pub dofs: usize,
    pub steps: usize,
    pub dt: f64,
}

fn add<const N: usize>(target: &mut TripletMatrix<f64>, dofs: &[usize; N], values: &[f64]) {
    for (i, &row) in dofs.iter().enumerate() {
        for (j, &column) in dofs.iter().enumerate() {
            let value = values[i * N + j];
            if value != 0.0 {
                target.push(row, column, value);
            }
        }
    }
}

fn constrain(matrix: &mut CsrMatrix<f64>, fixed: &[bool]) {
    for row in 0..matrix.num_rows {
        for entry in matrix.row_ptrs[row]..matrix.row_ptrs[row + 1] {
            let col = matrix.col_indices[entry];
            // A homogeneous constraint only needs its row/column decoupled.
            // Retain the physical positive diagonal: replacing tiny shell
            // inertia by 1 can make the valid rotary masses numerically singular
            // under a global-scale Cholesky pivot check.
            if col != row && (fixed[row] || fixed[col]) {
                matrix.values[entry] = 0.0;
            }
        }
    }
}

fn curve(points: &[(f64, f64)], time: f64) -> f64 {
    for pair in points.windows(2) {
        if time <= pair[1].0 {
            let fraction = ((time - pair[0].0) / (pair[1].0 - pair[0].0)).clamp(0.0, 1.0);
            return pair[0].1 + fraction * (pair[1].1 - pair[0].1);
        }
    }
    points.last().expect("validated nonempty force function").1
}

fn compensated_action(matrix: &CsrMatrix<f64>, values: &[f64]) -> Vec<f64> {
    (0..matrix.num_rows)
        .map(|row| {
            let mut high = 0.0;
            let mut low = 0.0;
            for entry in matrix.row_ptrs[row]..matrix.row_ptrs[row + 1] {
                let a = matrix.values[entry];
                let b = values[matrix.col_indices[entry]];
                let product = a * b;
                let remainder = a.mul_add(b, -product);
                let sum = high + product;
                let recovered = sum - high;
                low += (high - (sum - recovered)) + (product - recovered) + remainder;
                high = sum;
            }
            high + low
        })
        .collect()
}

fn gravity_loading(
    case: &RadExtendedCase,
    mass: &CsrMatrix<f64>,
) -> Result<Option<Vec<f64>>, Box<dyn std::error::Error>> {
    let Some(gravity) = &case.gravity else {
        return Ok(None);
    };
    let axis = match gravity.direction {
        'X' => 0,
        'Y' => 1,
        'Z' => 2,
        _ => return Err("invalid shell gravity direction".into()),
    };
    let mut field = vec![0.0; mass.num_rows];
    if let Some(set) = gravity.set {
        for &node in &case.base.node_sets[set].nodes {
            field[(node - 1) * 6 + axis] = gravity.value_m_per_s2;
        }
    } else {
        for node in 0..case.base.mesh.nodes.len() {
            field[node * 6 + axis] = gravity.value_m_per_s2;
        }
    }
    // Use the original unconstrained physical mass, before artificial BC diagonals.
    Ok(Some(compensated_action(mass, &field)))
}

fn force_at(
    case: &RadExtendedCase,
    gravity_load: Option<&Vec<f64>>,
    fixed: &[bool],
    time: f64,
) -> Result<Vec<f64>, Box<dyn std::error::Error>> {
    let base = &case.base;
    let mut force = vec![0.0; fixed.len()];
    for load in &base.cloads {
        if !(1..=6).contains(&load.direction) {
            return Err("invalid shell force direction".into());
        }
        let value = load.force_scale_n * curve(&base.functions[load.function].points, time);
        for &node in &base.node_sets[load.set].nodes {
            let index = (node - 1) * 6 + load.direction - 1;
            if !fixed[index] {
                force[index] += value;
            }
        }
    }
    if let Some(load) = gravity_load {
        let gravity = case
            .gravity
            .as_ref()
            .expect("gravity load has a corresponding card");
        let scale = gravity
            .function
            .map_or(1.0, |index| curve(&base.functions[index].points, time));
        for index in 0..force.len() {
            if !fixed[index] {
                force[index] += scale * load[index];
            }
        }
    }
    Ok(force)
}

fn congruence(matrix: &CsrMatrix<f64>, scale: &[f64]) -> CsrMatrix<f64> {
    let mut result = matrix.clone();
    for row in 0..result.num_rows {
        for entry in result.row_ptrs[row]..result.row_ptrs[row + 1] {
            result.values[entry] *= scale[row] * scale[result.col_indices[entry]];
        }
    }
    result
}

/// Solve a full elastic shell history with independent physical equilibrium checks.
///
/// With `equilibrate`, use the exact transformation `u = D q`, `D M D`,
/// `D K D`, and `D F` to balance mixed displacement/rotation units. Every
/// acceptance residual is recomputed in original physical coordinates.
///
/// # Errors
/// Returns an error for invalid fixture data, coordinate scales or observation
/// indices, incompatible timestep/output periods, failed factorization, or
/// physical force imbalance above `1e-8`. No shell DOFs are omitted.
pub(super) fn run(
    case: &RadExtendedCase,
    requested_dt: f64,
    observed_dofs: &[usize],
    equilibrate: bool,
) -> Result<ShellHistory, Box<dyn std::error::Error>> {
    case.validate()?;
    if !requested_dt.is_finite() || requested_dt <= 0.0 {
        return Err("positive finite shell timestep required".into());
    }
    let base = &case.base;
    if !base.mesh.bricks.is_empty() || !base.walls.is_empty() {
        return Err(
            "elastic shell transient requires shell elements and prescribed nodal forces".into(),
        );
    }
    let property = case
        .shell_prop
        .as_ref()
        .ok_or("missing native shell thickness")?;
    let section = ShellSection::new(
        base.material.youngs_modulus_pa,
        base.material.poisson_ratio,
        property.thickness_m,
        base.material.density_kg_per_m3,
    );
    let points = base
        .mesh
        .nodes
        .iter()
        .map(|p| Point::new_3d(p[0], p[1], p[2]))
        .collect::<Vec<_>>();
    let dofs = points.len() * 6;
    if observed_dofs.is_empty() || observed_dofs.iter().any(|&i| i >= dofs) {
        return Err("invalid native shell observation DOF".into());
    }
    let mut k = TripletMatrix::new(dofs, dofs);
    let mut m = TripletMatrix::new(dofs, dofs);
    for nodes in &case.shells.quads {
        let n = nodes.map(|node| node - 1);
        let element = QuadShellElement::from_points_with_section(
            n,
            &points[n[0]],
            &points[n[1]],
            &points[n[2]],
            &points[n[3]],
            section,
        );
        add(&mut k, &element.global_dofs(), &element.stiffness_global());
        add(&mut m, &element.global_dofs(), &element.mass_global());
    }
    for nodes in &case.shells.tris {
        let n = nodes.map(|node| node - 1);
        let element =
            ShellElement::from_points(n, &points[n[0]], &points[n[1]], &points[n[2]], section);
        add(&mut k, &element.global_dofs(), &element.stiffness_global());
        add(&mut m, &element.global_dofs(), &element.mass_global());
    }
    let mut stiffness = k.to_csr();
    let mut mass = m.to_csr();
    let gravity_load = gravity_loading(case, &mass)?;
    let mut fixed = vec![false; dofs];
    for bc in &base.bcs {
        for &node in &base.node_sets[bc.set].nodes {
            for axis in 0..6 {
                fixed[(node - 1) * 6 + axis] |= bc.lock[axis];
            }
        }
    }
    constrain(&mut stiffness, &fixed);
    constrain(&mut mass, &fixed);
    let steps_ratio = base.stop_time_s / requested_dt;
    let nearest_steps = steps_ratio.round();
    // Avoid an extra step when an integral history length rounds upward by
    // one floating-point ULP (for example 2 ms / 1 us). Otherwise round up
    // so the actual timestep never exceeds the requested temporal resolution.
    let steps = if (steps_ratio - nearest_steps).abs()
        <= 32.0 * f64::EPSILON * steps_ratio.abs().max(1.0)
    {
        nearest_steps as usize
    } else {
        steps_ratio.ceil() as usize
    };
    let dt = base.stop_time_s / steps as f64;
    let interval = (base.th_period_s / dt).round() as usize;
    if interval == 0 || (interval as f64 * dt - base.th_period_s).abs() > 1e-10 * base.th_period_s {
        return Err("native timestep must align with original external history period".into());
    }
    let config = ImplicitDynamicsConfig::default();
    let a0 = 1.0 / (config.beta * dt * dt);
    let scale = (0..dofs)
        .map(|row| {
            if equilibrate {
                (stiffness.get(row, row) + a0 * mass.get(row, row))
                    .sqrt()
                    .recip()
            } else {
                1.0
            }
        })
        .collect::<Vec<_>>();
    if scale.iter().any(|s| !s.is_finite() || *s <= 0.0) {
        return Err("invalid shell coordinate scale".into());
    }
    let minimum_scale = scale.iter().copied().fold(f64::INFINITY, f64::min);
    let mut integrator = ImplicitDynamicsIntegrator::new_spd(
        ImplicitDynamicsSystem {
            mass: congruence(&mass, &scale),
            stiffness: congruence(&stiffness, &scale),
            damping: None,
        },
        config,
        dt,
    )?;
    let zero = vec![0.0; dofs];
    let initial_force = force_at(case, gravity_load.as_ref(), &fixed, 0.0)?;
    let initial_force = initial_force
        .iter()
        .zip(&scale)
        .map(|(f, s)| f * s)
        .collect::<Vec<_>>();
    integrator.initialize(&zero, &zero, &initial_force)?;
    let mut history = ShellHistory {
        times: vec![0.0],
        observations: vec![vec![0.0; observed_dofs.len()]],
        max_true_dynamic_residual: 0.0,
        dofs,
        steps,
        dt,
    };
    println!(
        "full native shell transient: {} quads, {dofs} DOFs, {steps} steps, dt={dt:.9e}",
        case.shells.quads.len()
    );
    for step in 1..=steps {
        let time = step as f64 * dt;
        let force = force_at(case, gravity_load.as_ref(), &fixed, time)?;
        let norm = force.iter().map(|v| v * v).sum::<f64>().sqrt().max(1e-12);
        let transformed_force = force
            .iter()
            .zip(&scale)
            .map(|(f, s)| f * s)
            .collect::<Vec<_>>();
        let state =
            integrator.step_checked_accurate(&transformed_force, 0.9e-8 * norm * minimum_scale)?;
        let acceleration = state
            .acceleration
            .iter()
            .zip(&scale)
            .map(|(a, s)| a * s)
            .collect::<Vec<_>>();
        let displacement = state
            .displacement
            .iter()
            .zip(&scale)
            .map(|(u, s)| u * s)
            .collect::<Vec<_>>();
        let inertia = compensated_action(&mass, &acceleration);
        let elastic = compensated_action(&stiffness, &displacement);
        let residual = inertia
            .iter()
            .zip(&elastic)
            .zip(&force)
            .map(|((m, k), f)| (m + k - f).powi(2))
            .sum::<f64>()
            .sqrt()
            / norm;
        if !residual.is_finite() || residual > 1e-8 {
            return Err(format!(
                "step {step} true shell force residual {residual:.12e} exceeds 1e-8"
            )
            .into());
        }
        history.max_true_dynamic_residual = history.max_true_dynamic_residual.max(residual);
        if step.is_multiple_of(interval) {
            history.times.push(time);
            history
                .observations
                .push(observed_dofs.iter().map(|&i| displacement[i]).collect());
        }
    }
    Ok(history)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn coordinate_congruence_preserves_original_physical_matrix_action() {
        let mut matrix = CsrMatrix::new(3, 3);
        matrix.row_ptrs = vec![0, 2, 5, 7];
        matrix.col_indices = vec![0, 1, 0, 1, 2, 1, 2];
        matrix.values = vec![4.0, 1.0, 1.0, 5.0, 2.0, 2.0, 6.0];
        let scale = [1000.0, 2.0, 0.01];
        let u = [0.3, -0.7, 1.2];
        let q = u.iter().zip(&scale).map(|(u, d)| u / d).collect::<Vec<_>>();
        let physical = compensated_action(&matrix, &u);
        let transformed = compensated_action(&congruence(&matrix, &scale), &q);
        for i in 0..3 {
            assert!((transformed[i] / scale[i] - physical[i]).abs() < 1e-13);
        }
    }
    #[test]
    fn zero_constraints_preserve_physical_diagonal_scale_and_free_equations() {
        let mut matrix = CsrMatrix::new(3, 3);
        matrix.row_ptrs = vec![0, 2, 5, 7];
        matrix.col_indices = vec![0, 1, 0, 1, 2, 1, 2];
        matrix.values = vec![1e-11, 1e-16, 1e-16, 1e-18, 2e-19, 2e-19, 3e-18];
        constrain(&mut matrix, &[true, false, false]);
        assert_eq!(
            matrix.values,
            vec![1e-11, 0.0, 0.0, 1e-18, 2e-19, 2e-19, 3e-18]
        );
        let action = compensated_action(&matrix, &[0.0, 2.0, 3.0]);
        assert_eq!(action[0], 0.0);
        assert!((action[1] - 2.6e-18).abs() < 1e-32);
        assert!((action[2] - 9.4e-18).abs() < 1e-32);
    }
    #[test]
    fn body_force_has_correct_mass_gravity_units_and_zero_rotary_force() {
        let angle = 0.3_f64;
        let points = [
            Point::new_3d(0.0, 0.0, 0.0),
            Point::new_3d(2.0, 0.0, 0.0),
            Point::new_3d(2.0, 3.0 * angle.cos(), 3.0 * angle.sin()),
            Point::new_3d(0.0, 3.0 * angle.cos(), 3.0 * angle.sin()),
        ];
        let section = ShellSection::new(200e9, 0.3, 0.02, 1000.0);
        let element = QuadShellElement::from_points_with_section(
            [0, 1, 2, 3],
            &points[0],
            &points[1],
            &points[2],
            &points[3],
            section,
        );
        let mut mass = TripletMatrix::new(24, 24);
        add(&mut mass, &element.global_dofs(), &element.mass_global());
        let mut case = sonium_qa::rad::shell_scordelis_lo_case();
        case.base.mesh.nodes = points.iter().map(|p| [p.x, p.y, p.z]).collect();
        let load = gravity_loading(&case, &mass.to_csr()).unwrap().unwrap();
        let total = (0..4).map(|node| load[node * 6 + 1]).sum::<f64>();
        let expected = 1000.0 * 0.02 * 6.0 * (-9.81);
        assert!((total - expected).abs() < 1e-12 * expected.abs());
        assert!((0..4).all(|node| load[node * 6 + 3..node * 6 + 6].iter().all(|v| *v == 0.0)));
        let fixed = (0..24).map(|index| index < 6).collect::<Vec<_>>();
        let zero = force_at(&case, Some(&load), &fixed, 0.0).unwrap();
        assert!(zero.iter().all(|v| *v == 0.0));
        let half = force_at(&case, Some(&load), &fixed, 2.5).unwrap();
        assert_eq!(half[1], 0.0);
        assert_eq!(half[7], 0.5 * load[7]);
    }
}
