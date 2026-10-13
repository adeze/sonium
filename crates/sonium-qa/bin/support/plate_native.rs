//! Solve the original clamped disk with native mixed triangle/quad shells.
//!
//! Full transient, fixed-location moment recovery, and three-probe parity gate.

// Rust guideline compliant 2026-02-21

use sonium_fem::assembly::TripletMatrix;
use sonium_fem::mesh::Point;
use sonium_fem::physics::solid::shell::{QuadShellElement, ShellElement, ShellSection};
use sonium_fem::solver::implicit_dynamics::{
    ImplicitDynamicsConfig, ImplicitDynamicsIntegrator, ImplicitDynamicsSystem,
};
use sonium_qa::rad::{clamped_plate_shell_mesh, shell_clamped_plate_case};
use sonium_qa::{QaResult, QaStatus};
use sonium_solvers::{CsrMatrix, LinearOperator};

fn natural_coordinates(element: &QuadShellElement, point: [f64; 3]) -> Option<[f64; 2]> {
    let offset = std::array::from_fn::<_, 3, _>(|i| point[i] - element.origin[i]);
    let local = [
        (0..3)
            .map(|i| element.rotation[i * 3] * offset[i])
            .sum::<f64>(),
        (0..3)
            .map(|i| element.rotation[i * 3 + 1] * offset[i])
            .sum::<f64>(),
    ];
    let mut natural = [0.0; 2];
    for _ in 0..12 {
        let [xi, eta] = natural;
        let shapes = [
            (1.0 - xi) * (1.0 - eta),
            (1.0 + xi) * (1.0 - eta),
            (1.0 + xi) * (1.0 + eta),
            (1.0 - xi) * (1.0 + eta),
        ]
        .map(|v| v * 0.25);
        let dx = [-(1.0 - eta), 1.0 - eta, 1.0 + eta, -(1.0 + eta)].map(|v| v * 0.25);
        let dy = [-(1.0 - xi), -(1.0 + xi), 1.0 + xi, 1.0 - xi].map(|v| v * 0.25);
        let residual = std::array::from_fn::<_, 2, _>(|axis| {
            (0..4)
                .map(|i| {
                    shapes[i]
                        * if axis == 0 {
                            element.local_coords[i].0
                        } else {
                            element.local_coords[i].1
                        }
                })
                .sum::<f64>()
                - local[axis]
        });
        let jac = quad_jacobian(element, &dx, &dy);
        let det = jac[0] * jac[3] - jac[1] * jac[2];
        if !det.is_finite() || det.abs() < 1e-20 {
            return None;
        }
        natural[0] -= (jac[3] * residual[0] - jac[1] * residual[1]) / det;
        natural[1] -= (-jac[2] * residual[0] + jac[0] * residual[1]) / det;
        if residual[0].hypot(residual[1]) < 1e-12 {
            return natural
                .iter()
                .all(|v| v.is_finite() && v.abs() <= 1.0 + 1e-9)
                .then_some(natural);
        }
    }
    None
}

fn quad_jacobian(element: &QuadShellElement, dx: &[f64; 4], dy: &[f64; 4]) -> [f64; 4] {
    [
        (0..4).map(|i| element.local_coords[i].0 * dx[i]).sum(),
        (0..4).map(|i| element.local_coords[i].0 * dy[i]).sum(),
        (0..4).map(|i| element.local_coords[i].1 * dx[i]).sum(),
        (0..4).map(|i| element.local_coords[i].1 * dy[i]).sum(),
    ]
}

fn hoop_moment(
    element: &QuadShellElement,
    state: &[f64],
    natural: [f64; 2],
    point: [f64; 3],
) -> f64 {
    let [xi, eta] = natural;
    let dx = [-(1.0 - eta), 1.0 - eta, 1.0 + eta, -(1.0 + eta)].map(|v| v * 0.25);
    let dy = [-(1.0 - xi), -(1.0 + xi), 1.0 + xi, 1.0 - xi].map(|v| v * 0.25);
    let jac = quad_jacobian(element, &dx, &dy);
    let det = jac[0] * jac[3] - jac[1] * jac[2];
    let mut curvature = [0.0; 3];
    for i in 0..4 {
        let gx = (jac[3] * dx[i] - jac[2] * dy[i]) / det;
        let gy = (-jac[1] * dx[i] + jac[0] * dy[i]) / det;
        let node = element.nodes[i];
        let tx = (0..3)
            .map(|axis| element.rotation[axis * 3] * state[node * 6 + 3 + axis])
            .sum::<f64>();
        let ty = (0..3)
            .map(|axis| element.rotation[axis * 3 + 1] * state[node * 6 + 3 + axis])
            .sum::<f64>();
        // Native right-handed rotation convention: [ty,x, -tx,y, ty,y-tx,x].
        curvature[0] += gx * ty;
        curvature[1] -= gy * tx;
        curvature[2] += gy * ty - gx * tx;
    }
    let db = element.section.bending_constitutive();
    let moment = std::array::from_fn::<_, 3, _>(|row| {
        (0..3)
            .map(|col| db[row * 3 + col] * curvature[col])
            .sum::<f64>()
    });
    let radius = point[0].hypot(point[1]);
    let tangent = [-point[1] / radius, point[0] / radius, 0.0];
    let a = (0..3)
        .map(|i| element.rotation[i * 3] * tangent[i])
        .sum::<f64>();
    let b = (0..3)
        .map(|i| element.rotation[i * 3 + 1] * tangent[i])
        .sum::<f64>();
    a * a * moment[0] + b * b * moment[1] + 2.0 * a * b * moment[2]
}

// Cubic patch recovery differentiates the native nodal rotations at the fixed probe.
// The shrinking three-cell patch uses no analytical plate solution or external data.
fn patch_hoop_moment(
    element: &QuadShellElement,
    points: &[Point],
    state: &[f64],
    point: [f64; 3],
    radial_spacing: f64,
) -> Result<f64, Box<dyn std::error::Error>> {
    if state.len() != points.len() * 6 || !radial_spacing.is_finite() || radial_spacing <= 0.0 {
        return Err("invalid patch state or radial spacing".into());
    }
    let mut gram = sonium_solvers::dense::DenseMatrix::<f64>::zeros(10, 10);
    let mut rhs = [vec![0.0; 10], vec![0.0; 10]];
    let mut count = 0;
    for (node, p) in points.iter().enumerate() {
        let offset = [p.x - point[0], p.y - point[1], p.z - point[2]];
        let local: [f64; 2] = std::array::from_fn(|axis| {
            (0..3)
                .map(|i| element.rotation[i * 3 + axis] * offset[i])
                .sum::<f64>()
                / radial_spacing
        });
        let [x, y] = local;
        let distance_squared = x * x + y * y;
        if distance_squared >= 9.0 {
            continue;
        }
        let weight = (1.0 - distance_squared / 9.0).powi(2);
        let basis = [
            1.0,
            x,
            y,
            x * x,
            x * y,
            y * y,
            x * x * x,
            x * x * y,
            x * y * y,
            y * y * y,
        ];
        let rotation: [f64; 2] = std::array::from_fn(|axis| {
            (0..3)
                .map(|i| element.rotation[i * 3 + axis] * state[node * 6 + 3 + i])
                .sum::<f64>()
        });
        for i in 0..10 {
            for j in 0..10 {
                gram[[i, j]] += weight * basis[i] * basis[j];
            }
            for axis in 0..2 {
                rhs[axis][i] += weight * basis[i] * rotation[axis];
            }
        }
        count += 1;
    }
    if count < 20 {
        return Err("insufficient native nodes for cubic patch recovery".into());
    }
    let tx = sonium_solvers::direct::lu_solve(&gram, &rhs[0])?;
    let ty = sonium_solvers::direct::lu_solve(&gram, &rhs[1])?;
    let curvature = [
        ty[1] / radial_spacing,
        -tx[2] / radial_spacing,
        (ty[2] - tx[1]) / radial_spacing,
    ];
    let db = element.section.bending_constitutive();
    let moment: [f64; 3] = std::array::from_fn(|row| {
        (0..3)
            .map(|col| db[row * 3 + col] * curvature[col])
            .sum::<f64>()
    });
    let radius = point[0].hypot(point[1]);
    if radius == 0.0 {
        return Err("hoop direction undefined at the disk center".into());
    }
    let tangent = [-point[1] / radius, point[0] / radius, 0.0];
    let local: [f64; 2] = std::array::from_fn(|axis| {
        (0..3)
            .map(|i| element.rotation[i * 3 + axis] * tangent[i])
            .sum::<f64>()
    });
    let value = local[0].powi(2) * moment[0]
        + local[1].powi(2) * moment[1]
        + 2.0 * local[0] * local[1] * moment[2];
    if !value.is_finite() {
        return Err("non-finite diagnostic patch moment".into());
    }
    Ok(value)
}

fn add_element<const N: usize>(target: &mut TripletMatrix<f64>, dofs: &[usize; N], values: &[f64]) {
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
            let column = matrix.col_indices[entry];
            if fixed[row] {
                matrix.values[entry] = if row == column { 1.0 } else { 0.0 };
            } else if fixed[column] {
                matrix.values[entry] = 0.0;
            }
        }
    }
}

fn amplitude(points: &[(f64, f64)], time: f64) -> f64 {
    for pair in points.windows(2) {
        if time <= pair[1].0 {
            let fraction = ((time - pair[0].0) / (pair[1].0 - pair[0].0)).clamp(0.0, 1.0);
            return pair[0].1 + fraction * (pair[1].1 - pair[0].1);
        }
    }
    points.last().expect("original load curve is nonempty").1
}

fn bending_block(
    matrix: &CsrMatrix<f64>,
    active: &[usize],
) -> Result<CsrMatrix<f64>, Box<dyn std::error::Error>> {
    let mut inverse = vec![usize::MAX; matrix.num_rows];
    for (reduced, &full) in active.iter().enumerate() {
        inverse[full] = reduced;
    }
    let mut block = TripletMatrix::new(active.len(), active.len());
    for row in 0..matrix.num_rows {
        for entry in matrix.row_ptrs[row]..matrix.row_ptrs[row + 1] {
            let column = matrix.col_indices[entry];
            let a = inverse[row];
            let b = inverse[column];
            let value = matrix.values[entry];
            if (a == usize::MAX) != (b == usize::MAX) && value != 0.0 {
                return Err(
                    "flat plate membrane/drilling and bending blocks are not exactly decoupled"
                        .into(),
                );
            }
            if a != usize::MAX && b != usize::MAX && value != 0.0 {
                block.push(a, b, value);
            }
        }
    }
    Ok(block.to_csr())
}

fn expand(values: &[f64], active: &[usize], dimension: usize) -> Vec<f64> {
    let mut full = vec![0.0; dimension];
    for (&index, &value) in active.iter().zip(values) {
        full[index] = value;
    }
    full
}

fn diagonal_congruence(matrix: &CsrMatrix<f64>, scale: &[f64]) -> CsrMatrix<f64> {
    let mut scaled = matrix.clone();
    for row in 0..matrix.num_rows {
        for entry in matrix.row_ptrs[row]..matrix.row_ptrs[row + 1] {
            scaled.values[entry] *= scale[row] * scale[matrix.col_indices[entry]];
        }
    }
    scaled
}

fn compensated_apply(matrix: &CsrMatrix<f64>, values: &[f64]) -> Vec<f64> {
    (0..matrix.num_rows)
        .map(|row| {
            let mut high = 0.0;
            let mut low = 0.0;
            for entry in matrix.row_ptrs[row]..matrix.row_ptrs[row + 1] {
                let a = matrix.values[entry];
                let b = values[matrix.col_indices[entry]];
                let product = a * b;
                let product_error = a.mul_add(b, -product);
                let sum = high + product;
                let recovered = sum - high;
                low += (high - (sum - recovered)) + (product - recovered) + product_error;
                high = sum;
            }
            high + low
        })
        .collect()
}

/// Solve the original full plate history and recover its fixed-location moment.
///
/// # Errors
///
/// Returns an error for invalid resolution, failed factorization or timesteps,
/// or a non-finite field or independently checked residual above `1e-8`.
pub fn solve(
    rings: usize,
    sectors: usize,
    requested_dt: f64,
) -> Result<serde_json::Value, Box<dyn std::error::Error>> {
    solve_impl(rings, sectors, requested_dt, false, true, 0.0)
}

/// Diagnose congruent coordinate scaling on the unchanged physical shell.
///
/// # Errors
///
/// Returns the same input, factorization and true-residual errors as [`solve`].
pub fn solve_scaled(
    rings: usize,
    sectors: usize,
    requested_dt: f64,
) -> Result<serde_json::Value, Box<dyn std::error::Error>> {
    solve_impl(rings, sectors, requested_dt, true, true, 0.0)
}

/// Diagnose LU factorization on the unchanged physical shell.
///
/// # Errors
///
/// Returns the same input, factorization and true-residual errors as [`solve`].
pub fn solve_lu(
    rings: usize,
    sectors: usize,
    requested_dt: f64,
) -> Result<serde_json::Value, Box<dyn std::error::Error>> {
    solve_impl(rings, sectors, requested_dt, false, false, 0.0)
}

/// Diagnose second-order HHT integration of the unchanged full shell history.
///
/// # Errors
///
/// Returns the same input and solve errors as [`solve`], checking the original
/// full six-DOF HHT discrete force balance with the same `1e-8` bound.
pub fn solve_hht(
    rings: usize,
    sectors: usize,
    requested_dt: f64,
) -> Result<serde_json::Value, Box<dyn std::error::Error>> {
    // The standard alpha=-.1 choice damps unresolved stiff modes. Temporal
    // refinement, not this parameter choice, determines physical accuracy.
    solve_impl(rings, sectors, requested_dt, false, true, -0.1)
}

fn relative_force_residual(
    inertia: &[f64],
    elastic: &[f64],
    previous_elastic: &[f64],
    force: &[f64],
    previous_force: &[f64],
    alpha: f64,
) -> f64 {
    let h = 1.0 + alpha;
    let mut residual = 0.0;
    let mut norm = 0.0;
    for i in 0..force.len() {
        let weighted_force = h * force[i] - alpha * previous_force[i];
        residual +=
            (inertia[i] + h * elastic[i] - alpha * previous_elastic[i] - weighted_force).powi(2);
        norm += weighted_force.powi(2);
    }
    residual.sqrt() / norm.sqrt().max(1e-12)
}

fn solve_impl(
    rings: usize,
    sectors: usize,
    requested_dt: f64,
    equilibrate: bool,
    cholesky: bool,
    alpha: f64,
) -> Result<serde_json::Value, Box<dyn std::error::Error>> {
    if rings < 2 || sectors < 3 || !requested_dt.is_finite() || requested_dt <= 0.0 {
        return Err("require rings >=2, sectors >=3 and a positive finite timestep".into());
    }
    let original = shell_clamped_plate_case();
    let case = &original.base;
    let thickness = original
        .shell_prop
        .as_ref()
        .ok_or("missing original shell section")?
        .thickness_m;
    let section = ShellSection::new(
        case.material.youngs_modulus_pa,
        case.material.poisson_ratio,
        thickness,
        case.material.density_kg_per_m3,
    );
    let mesh = clamped_plate_shell_mesh(1.0, rings, sectors);
    let dimension = mesh.nodes.len() * 6;
    let points: Vec<_> = mesh
        .nodes
        .iter()
        .map(|p| Point::new_3d(p[0], p[1], p[2]))
        .collect();
    let mut k = TripletMatrix::new(dimension, dimension);
    let mut m = TripletMatrix::new(dimension, dimension);
    let mut quads = Vec::new();
    for nodes in &mesh.quads {
        let n = nodes.map(|node| node - 1);
        let element = QuadShellElement::from_points_with_section(
            n,
            &points[n[0]],
            &points[n[1]],
            &points[n[2]],
            &points[n[3]],
            section,
        );
        add_element(&mut k, &element.global_dofs(), &element.stiffness_global());
        add_element(&mut m, &element.global_dofs(), &element.mass_global());
        quads.push(element);
    }
    for nodes in &mesh.tris {
        let n = nodes.map(|node| node - 1);
        let element =
            ShellElement::from_points(n, &points[n[0]], &points[n[1]], &points[n[2]], section);
        add_element(&mut k, &element.global_dofs(), &element.stiffness_global());
        add_element(&mut m, &element.global_dofs(), &element.mass_global());
    }
    let mut stiffness = k.to_csr();
    let mut mass = m.to_csr();
    let mut fixed = vec![false; dimension];
    for node in &mesh.rim {
        fixed[(node - 1) * 6..node * 6].fill(true);
    }
    constrain(&mut stiffness, &fixed);
    constrain(&mut mass, &fixed);
    let original_mesh = clamped_plate_shell_mesh(1.0, 8, 32);
    let pressure = -case.cloads[0].force_scale_n / original_mesh.nodal_area_m2[0];
    let mut full_force = vec![0.0; dimension];
    for (node, area) in mesh.nodal_area_m2.iter().enumerate() {
        if !fixed[node * 6 + 2] {
            full_force[node * 6 + 2] = -pressure * area;
        }
    }
    let steps = (case.stop_time_s / requested_dt).ceil() as usize;
    let dt = case.stop_time_s / steps as f64;
    let config = ImplicitDynamicsConfig::hht(alpha);
    println!(
        "original clamped plate transient: rings={rings}, sectors={sectors}, DOFs={dimension}, steps={steps}, dt={dt:.9e}"
    );
    // For this flat shell under normal loading, in-plane and drilling DOFs
    // have zero forcing and exactly zero coupling to bending. Verify that
    // algebraic invariant before removing their homogeneous zero solution.
    // Every residual below still checks the complete original six-DOF system.
    let active: Vec<_> = (0..mesh.nodes.len())
        .flat_map(|node| [node * 6 + 2, node * 6 + 3, node * 6 + 4])
        .collect();
    let reduced_mass = bending_block(&mass, &active)?;
    let reduced_stiffness = bending_block(&stiffness, &active)?;
    // u = D q equilibrates the mixed displacement/rotation units. Multiplying
    // both sides by D preserves exactly M u'' + K u = F, including inertia.
    let mut scale: Vec<_> = (0..active.len())
        .map(|row| {
            let diagonal = |matrix: &CsrMatrix<f64>| {
                (matrix.row_ptrs[row]..matrix.row_ptrs[row + 1])
                    .find(|&entry| matrix.col_indices[entry] == row)
                    .map(|entry| matrix.values[entry])
                    .unwrap_or(0.0)
            };
            ((1.0 + alpha) * diagonal(&reduced_stiffness)
                + 1.0 / (config.beta * dt.powi(2)) * diagonal(&reduced_mass))
            .sqrt()
            .recip()
        })
        .collect();
    if !equilibrate {
        scale.fill(1.0);
    }
    if scale.iter().any(|v| !v.is_finite() || *v <= 0.0) {
        return Err("invalid plate coordinate scale".into());
    }
    let minimum_scale = scale.iter().copied().fold(f64::INFINITY, f64::min);
    let constructor = if cholesky {
        ImplicitDynamicsIntegrator::new_spd
    } else {
        ImplicitDynamicsIntegrator::new
    };
    let mut integrator = constructor(
        ImplicitDynamicsSystem {
            mass: diagonal_congruence(&reduced_mass, &scale),
            stiffness: diagonal_congruence(&reduced_stiffness, &scale),
            damping: None,
        },
        config,
        dt,
    )?;
    let zero = vec![0.0; active.len()];
    integrator.initialize(&zero, &zero, &zero)?;
    let mut max_residual = 0.0_f64;
    let mut previous_displacement = vec![0.0; dimension];
    let mut previous_force = vec![0.0; dimension];
    for step in 1..=steps {
        let time = step as f64 * dt;
        let load_scale = amplitude(&case.functions[0].points, time);
        let force: Vec<_> = full_force.iter().map(|value| value * load_scale).collect();
        let norm = force
            .iter()
            .zip(&previous_force)
            .map(|(next, previous)| ((1.0 + alpha) * next - alpha * previous).powi(2))
            .sum::<f64>()
            .sqrt();
        let previous_elastic = compensated_apply(&stiffness, &previous_displacement);
        // Reserve ten percent for the independently accumulated residual below.
        let reduced_force: Vec<_> = active
            .iter()
            .zip(&scale)
            .map(|(&index, &d)| force[index] * d)
            .collect();
        // ||D r|| <= min(D) epsilon implies ||r|| <= epsilon. The original
        // physical full-system check below remains the final authority.
        let unscale = |values: &[f64]| {
            values
                .iter()
                .zip(&scale)
                .map(|(q, d)| q * d)
                .collect::<Vec<_>>()
        };
        let checked = integrator
            .step_checked_accurate(&reduced_force, 0.9e-8 * norm.max(1e-12) * minimum_scale);
        if let Err(error) = checked {
            // The failed checked step publishes no state. Diagnose its original
            // unrefined candidate, then retain the error and fail this run.
            let state = integrator.step(&reduced_force)?;
            let acceleration = expand(&unscale(&state.acceleration), &active, dimension);
            let displacement = expand(&unscale(&state.displacement), &active, dimension);
            for (label, inertia, elastic, previous_k) in [
                (
                    "ordinary",
                    mass.apply(&acceleration),
                    stiffness.apply(&displacement),
                    stiffness.apply(&previous_displacement),
                ),
                (
                    "compensated",
                    compensated_apply(&mass, &acceleration),
                    compensated_apply(&stiffness, &displacement),
                    previous_elastic.clone(),
                ),
            ] {
                let residual = relative_force_residual(
                    &inertia,
                    &elastic,
                    &previous_k,
                    &force,
                    &previous_force,
                    alpha,
                );
                println!(
                    "failed step {step}, time={time:.9e}, unrefined candidate {label} full physical residual={residual:.12e}"
                );
            }
            return Err(error.into());
        }
        let state = integrator.state();
        let inertia = compensated_apply(
            &mass,
            &expand(&unscale(&state.acceleration), &active, dimension),
        );
        let elastic = compensated_apply(
            &stiffness,
            &expand(&unscale(&state.displacement), &active, dimension),
        );
        let residual = relative_force_residual(
            &inertia,
            &elastic,
            &previous_elastic,
            &force,
            &previous_force,
            alpha,
        );
        if !residual.is_finite() || residual > 1e-8 {
            return Err(format!("step {step} independent residual {residual} exceeds 1e-8").into());
        }
        max_residual = max_residual.max(residual);
        previous_displacement = expand(&unscale(&state.displacement), &active, dimension);
        previous_force = force;
    }
    let physical: Vec<_> = integrator
        .state()
        .displacement
        .iter()
        .zip(&scale)
        .map(|(q, d)| q * d)
        .collect();
    let displacement = expand(&physical, &active, dimension);
    let center = displacement[(mesh.center_node - 1) * 6 + 2];
    // Keep the physical center of the original output shell fixed across meshes.
    let original_probe = original_mesh.quads[original_mesh.probe_shell - 1];
    let probe_point = std::array::from_fn::<_, 3, _>(|axis| {
        original_probe
            .iter()
            .map(|node| original_mesh.nodes[node - 1][axis])
            .sum::<f64>()
            / 4.0
    });
    let (probe_element, natural) = quads
        .iter()
        .find_map(|element| {
            natural_coordinates(element, probe_point).map(|natural| (element, natural))
        })
        .ok_or("original shell probe lies outside the native quad mesh")?;
    let moment = hoop_moment(probe_element, &displacement, natural, probe_point);
    let recovered_m1 = patch_hoop_moment(
        probe_element,
        &points,
        &displacement,
        probe_point,
        1.0 / rings as f64,
    )? / thickness.powi(2);
    let m1 = moment / thickness.powi(2);
    if !m1.is_finite() {
        return Err("native recovered moment is non-finite".into());
    }
    let result = serde_json::json!({
        "scope": "original full transient and fixed-location shell moment; diagnostic only; external parity not yet checked",
        "rings": rings, "sectors": sectors, "dofs": dimension, "steps": steps, "dt": dt,
        "solved_bending_dofs":active.len(), "full_six_dof_equations_checked":true,
        "coordinate_equilibration":equilibrate,"factorization":if cholesky {"cholesky"} else {"lu"},
        "hht_alpha":alpha,"equilibrium":"M a_next + (1+alpha) K u_next - alpha K u_previous = (1+alpha) F_next - alpha F_previous",
        "center_dz": center, "kirchhoff_center_dz": case.probes[0].expected,
        "center_relative_error": (center - case.probes[0].expected).abs() / case.probes[0].expected.abs(),
        "max_independent_equation_residual": max_residual,
        "probe_point": probe_point, "thickness": thickness,
        "hoop_moment_per_length": moment, "m1": m1,
        "recovered_m1":recovered_m1,
        "moment_recovery":"weighted cubic fit of actual native nodal rotations over a shrinking three-radial-cell patch",
        "kirchhoff_m1": case.probes[2].expected,
        "moment_relative_error": (m1-case.probes[2].expected).abs()/case.probes[2].expected.abs(),
    });
    println!("{}", serde_json::to_string_pretty(&result)?);
    Ok(result)
}

fn number(run: &serde_json::Value, key: &str) -> Result<f64, Box<dyn std::error::Error>> {
    run.get(key)
        .and_then(serde_json::Value::as_f64)
        .filter(|v| v.is_finite())
        .ok_or_else(|| format!("missing finite native field {key}").into())
}

/// Compare full native plate histories with the original three external probes.
///
/// # Errors
///
/// Returns an error for changed fixture identity, missing external probes,
/// or a failed native transient or full-system force-balance check.
pub(super) fn parity_result(
    case: &sonium_qa::rad::RadExtendedCase,
    case_id: &str,
    engine: &str,
    external: &[f64],
    reference_passed: bool,
) -> Result<QaResult, Box<dyn std::error::Error>> {
    let original = shell_clamped_plate_case();
    if sonium_qa::rad::write_extended_starter(case)
        != sonium_qa::rad::write_extended_starter(&original)
        || sonium_qa::rad::write_extended_engine(case)
            != sonium_qa::rad::write_extended_engine(&original)
        || case.base.probes.len() != original.base.probes.len()
        || case
            .base
            .probes
            .iter()
            .zip(&original.base.probes)
            .any(|(a, b)| {
                a.channel != b.channel
                    || a.expected != b.expected
                    || a.abs_tol != b.abs_tol
                    || a.rel_tol != b.rel_tol
            })
    {
        return Err("original plate fixture identity differs".into());
    }
    if external.len() != 3 || external.iter().any(|v| !v.is_finite()) {
        return Err("all three finite original external shell probes are required".into());
    }
    // Recover the same fixed physical moment from the native rotation field.
    // The shrinking patch avoids jumps in elementwise curvature near a cell edge.
    let runs = [
        solve(8, 32, 0.00025)?,
        solve(16, 64, 0.00025)?,
        solve(24, 96, 0.00025)?,
        solve(24, 96, 0.000125)?,
    ];
    build_result(case, case_id, engine, external, reference_passed, &runs)
}

fn build_result(
    case: &sonium_qa::rad::RadExtendedCase,
    case_id: &str,
    engine: &str,
    external: &[f64],
    reference_passed: bool,
    runs: &[serde_json::Value; 4],
) -> Result<QaResult, Box<dyn std::error::Error>> {
    let mut result = QaResult::passed(case_id)
        .requested_backend("cpu+opencourant").actual_backend("cpu+opencourant")
        .metric("external_reference_executed",1.0)
        .metric("external_reference_health_passed", if reference_passed {1.0} else {0.0}).minimum("external_reference_health_passed",1.0)
        .metric("native_parity_executed",1.0).minimum("native_parity_executed",1.0)
        .reference(engine)
        .dimension("Original radius 1 m, thickness .02 m circular shell; all six rim DOFs clamped; full 1 s history; 8/16/24 rings and 32/64/96 sectors")
        .formulation("Native DKT hub and MITC4 rings with translational/rotary inertia and compensated checked Newmark; verified exact flat-shell block decoupling, independent full six-DOF force balance; original -1000 Pa tributary nodal load; fixed original shell centroid hoop moment recovered from a shrinking cubic native-rotation patch; raw element moment also requires independent accuracy and external parity")
        .tolerance("Native/external displacement and moment errors <=3%; original Kirchhoff bounds 5% and 15%; spatial changes <=2% then <=.5%, contraction <=.6; time changes <=.3%; every-step true force residual <=1e-8; thickness error <=1e-6");
    for (index, key) in [(0, "center_dz"), (2, "recovered_m1")] {
        let values = runs
            .iter()
            .map(|run| number(run, key))
            .collect::<Result<Vec<_>, _>>()?;
        let first = (values[1] - values[0]).abs() / values[1].abs().max(1e-30);
        let last = (values[2] - values[1]).abs() / values[2].abs().max(1e-30);
        for (name, value, budget) in [
            (
                "native_external_relative_error",
                (values[2] - external[index]).abs() / external[index].abs().max(1e-30),
                0.03,
            ),
            (
                "native_analytical_relative_error",
                (values[2] - case.base.probes[index].expected).abs()
                    / case.base.probes[index].expected.abs(),
                case.base.probes[index].rel_tol,
            ),
            ("spatial_first_relative_change", first, 0.02),
            ("spatial_final_relative_change", last, 0.005),
            ("spatial_contraction", last / first.max(1e-15), 0.6),
            (
                "temporal_relative_change",
                (values[2] - values[3]).abs() / values[3].abs().max(1e-30),
                0.003,
            ),
        ] {
            let name = format!("probe_{index}_{name}");
            result = result.metric(&name, value).maximum(name, budget);
        }
        result = result
            .metric(format!("probe_{index}_native"), values[2])
            .metric(format!("probe_{index}_external"), external[index]);
    }
    // Recovery cannot hide a bad raw native element moment at the original probe.
    let raw_moment = number(&runs[2], "m1")?;
    result = result
        .metric("raw_element_m1", raw_moment)
        .metric(
            "raw_element_moment_external_relative_error",
            (raw_moment - external[2]).abs() / external[2].abs().max(1e-30),
        )
        .maximum("raw_element_moment_external_relative_error", 0.03)
        .metric(
            "raw_element_moment_analytical_relative_error",
            (raw_moment - case.base.probes[2].expected).abs() / case.base.probes[2].expected.abs(),
        )
        .maximum(
            "raw_element_moment_analytical_relative_error",
            case.base.probes[2].rel_tol,
        );
    let thickness = number(&runs[2], "thickness")?;
    result = result
        .metric("native_thickness", thickness)
        .metric(
            "thickness_relative_error",
            (thickness - external[1]).abs() / external[1].abs().max(1e-30),
        )
        .maximum("thickness_relative_error", 1e-6)
        .metric(
            "thickness_analytical_relative_error",
            (thickness - case.base.probes[1].expected).abs() / case.base.probes[1].expected.abs(),
        )
        .maximum(
            "thickness_analytical_relative_error",
            case.base.probes[1].rel_tol,
        );
    for (index, run) in runs.iter().enumerate() {
        let name = format!("mesh_{index}_true_dynamic_residual");
        result = result
            .metric(&name, number(run, "max_independent_equation_residual")?)
            .maximum(name, 1e-8);
    }
    if result.release_eligible().is_err() {
        result.status = QaStatus::Failed;
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn full_hht_equilibrium_requires_the_previous_physical_load_and_field() {
        let inertia = [0.4, -0.7];
        let elastic = [2.0, -3.0];
        let previous_elastic = [-1.0, 4.0];
        let previous_force = [1.0, 0.5];
        let alpha = -0.1;
        let h = 1.0 + alpha;
        let force = std::array::from_fn::<_, 2, _>(|i| {
            (inertia[i] + h * elastic[i] - alpha * previous_elastic[i] + alpha * previous_force[i])
                / h
        });
        assert!(
            relative_force_residual(
                &inertia,
                &elastic,
                &previous_elastic,
                &force,
                &previous_force,
                alpha
            ) < 1e-14
        );
        assert!(
            relative_force_residual(
                &inertia,
                &elastic,
                &[0.0, 0.0],
                &force,
                &previous_force,
                alpha
            ) > 0.01
        );
        assert!(
            relative_force_residual(
                &inertia,
                &elastic,
                &previous_elastic,
                &force,
                &[0.0, 0.0],
                alpha
            ) > 0.01
        );
        assert!(
            relative_force_residual(
                &inertia,
                &elastic,
                &previous_elastic,
                &force,
                &previous_force,
                0.0
            ) > 0.01
        );
    }

    #[test]
    fn original_moment_and_true_residual_are_independent_release_requirements() {
        let case = shell_clamped_plate_case();
        let external: Vec<_> = case.base.probes.iter().map(|p| p.expected).collect();
        let mut runs=[0.98,0.997,1.0,1.001].map(|factor|serde_json::json!({"center_dz":external[0]*factor,"m1":external[2]*factor,"recovered_m1":external[2]*factor,"thickness":external[1],"max_independent_equation_residual":1e-10}));
        assert!(
            build_result(&case, "test", "test", &external, true, &runs)
                .unwrap()
                .release_eligible()
                .is_ok()
        );
        runs[2]["m1"] = serde_json::json!(0.0);
        assert!(
            build_result(&case, "test", "test", &external, true, &runs)
                .unwrap()
                .release_eligible()
                .is_err()
        );
        runs[2]["m1"] = serde_json::json!(external[2]);
        runs[2]["recovered_m1"] = serde_json::json!(0.0);
        assert!(
            build_result(&case, "test", "test", &external, true, &runs)
                .unwrap()
                .release_eligible()
                .is_err()
        );
        runs[2]["recovered_m1"] = serde_json::json!(external[2]);
        runs[2]["max_independent_equation_residual"] = serde_json::json!(1e-4);
        assert!(
            build_result(&case, "test", "test", &external, true, &runs)
                .unwrap()
                .release_eligible()
                .is_err()
        );
    }

    #[test]
    fn coordinate_scaling_preserves_force_balance_in_mixed_units() {
        let mut entries = TripletMatrix::new(3, 3);
        for (row, column, value) in [
            (0, 0, 7e8),
            (0, 1, -2e5),
            (1, 0, -2e5),
            (1, 1, 9e2),
            (1, 2, 3.0),
            (2, 1, 3.0),
            (2, 2, 4.0),
        ] {
            entries.push(row, column, value);
        }
        let matrix = entries.to_csr();
        let scale = [1e-6, 2e-3, 0.1];
        let q = [0.02, -0.3, 0.5];
        let physical: Vec<_> = q.iter().zip(&scale).map(|(q, d)| q * d).collect();
        let full_force = matrix.apply(&physical);
        let transformed = diagonal_congruence(&matrix, &scale).apply(&q);
        for ((scaled, full), d) in transformed.iter().zip(&full_force).zip(&scale) {
            assert!((scaled / d - full).abs() < 1e-12 * full.abs().max(1.0));
        }
    }

    #[test]
    fn flat_shell_block_retains_every_full_equation_and_rejects_coupling() {
        let points = [[1.0, 0.0], [2.0, 0.0], [2.0, 1.0], [1.0, 1.0]]
            .map(|p| Point::new_3d(p[0], p[1], 0.0));
        let element = QuadShellElement::from_points_with_section(
            [0, 1, 2, 3],
            &points[0],
            &points[1],
            &points[2],
            &points[3],
            ShellSection::new(200e9, 0.3, 0.02, 7800.0),
        );
        let active: Vec<_> = (0..4)
            .flat_map(|node| [node * 6 + 2, node * 6 + 3, node * 6 + 4])
            .collect();
        let values: Vec<_> = (0..12).map(|i| ((i + 1) as f64).sin() * 1e-4).collect();
        for entries in [element.stiffness_global(), element.mass_global()] {
            let mut full = TripletMatrix::new(24, 24);
            add_element(&mut full, &element.global_dofs(), &entries);
            let matrix = full.to_csr();
            let block = bending_block(&matrix, &active).unwrap();
            let actual = matrix.apply(&expand(&values, &active, 24));
            let expected = expand(&block.apply(&values), &active, 24);
            for (a, b) in actual.iter().zip(&expected) {
                assert!((a - b).abs() <= 1e-12 * a.abs().max(1.0));
            }
            full.push(0, 2, 1e-20);
            assert!(bending_block(&full.to_csr(), &active).is_err());
        }
    }

    #[test]
    fn moment_recovery_preserves_constant_curvature_axes_and_units() {
        let section = ShellSection::new(200e9, 0.3, 0.02, 7800.0);
        for flip in [1.0, -1.0] {
            let points = [[1.0, 0.0], [2.0, 0.0], [2.0, flip], [1.0, flip]]
                .map(|p| Point::new_3d(p[0], p[1], 0.0));
            let element = QuadShellElement::from_points_with_section(
                [0, 1, 2, 3],
                &points[0],
                &points[1],
                &points[2],
                &points[3],
                section,
            );
            let curvature = [0.7, -0.2, 0.3];
            let mut state = vec![0.0; 24];
            for node in 0..4 {
                let (x, y) = element.local_coords[node];
                let tx = -curvature[1] * y - curvature[2] * x * 0.5;
                let ty = curvature[0] * x + curvature[2] * y * 0.5;
                for axis in 0..3 {
                    state[node * 6 + 3 + axis] =
                        element.rotation[axis * 3] * tx + element.rotation[axis * 3 + 1] * ty;
                }
            }
            let point = [1.5, flip * 0.5, 0.0];
            let natural = natural_coordinates(&element, point).unwrap();
            assert!(natural[0].abs() < 1e-12 && natural[1].abs() < 1e-12);
            let d = section.bending_stiffness();
            let moments = [
                d * (curvature[0] + 0.3 * curvature[1]),
                d * (0.3 * curvature[0] + curvature[1]),
                d * 0.35 * curvature[2],
            ];
            let tangent = [
                -point[1] / point[0].hypot(point[1]),
                point[0] / point[0].hypot(point[1]),
                0.0,
            ];
            let a = (0..3)
                .map(|i| element.rotation[i * 3] * tangent[i])
                .sum::<f64>();
            let b = (0..3)
                .map(|i| element.rotation[i * 3 + 1] * tangent[i])
                .sum::<f64>();
            let expected = a * a * moments[0] + b * b * moments[1] + 2.0 * a * b * moments[2];
            assert!(
                (hoop_moment(&element, &state, natural, point) - expected).abs()
                    < 1e-10 * expected.abs().max(1.0)
            );
            assert_eq!(hoop_moment(&element, &vec![0.0; 24], natural, point), 0.0);
            assert!(natural_coordinates(&element, [4.0, 4.0, 0.0]).is_none());
        }
    }
    #[test]
    fn cubic_patch_recovers_manufactured_rotations_across_original_meshes() {
        let original = clamped_plate_shell_mesh(1.0, 8, 32);
        let shell = original.quads[original.probe_shell - 1];
        let point: [f64; 3] = std::array::from_fn(|axis| {
            shell
                .iter()
                .map(|node| original.nodes[node - 1][axis])
                .sum::<f64>()
                / 4.0
        });
        for rings in [8, 16, 24] {
            let mesh = clamped_plate_shell_mesh(1.0, rings, rings * 4);
            let points = mesh
                .nodes
                .iter()
                .map(|p| Point::new_3d(p[0], p[1], p[2]))
                .collect::<Vec<_>>();
            let section = ShellSection::new(200e9, 0.3, 0.02, 7800.0);
            let element = mesh
                .quads
                .iter()
                .find_map(|nodes| {
                    let ids = nodes.map(|node| node - 1);
                    let element = QuadShellElement::from_points_with_section(
                        ids,
                        &points[ids[0]],
                        &points[ids[1]],
                        &points[ids[2]],
                        &points[ids[3]],
                        section,
                    );
                    natural_coordinates(&element, point).map(|_| element)
                })
                .unwrap();
            let mut state = vec![0.0; points.len() * 6];
            for (node, p) in points.iter().enumerate() {
                let offset = [p.x - point[0], p.y - point[1], p.z - point[2]];
                let x = (0..3)
                    .map(|i| element.rotation[i * 3] * offset[i])
                    .sum::<f64>();
                let y = (0..3)
                    .map(|i| element.rotation[i * 3 + 1] * offset[i])
                    .sum::<f64>();
                let tx = 0.1 + 0.7 * x - 0.2 * y + 0.3 * x * x + 0.6 * x * y - 0.4 * y * y
                    + x * x * x
                    - 0.5 * x * x * y;
                let ty = -0.2 + 0.4 * x + 0.6 * y - 0.1 * x * x
                    + 0.2 * x * y
                    + 0.4 * y * y
                    + 0.3 * x * y * y
                    - 0.2 * y * y * y;
                for axis in 0..3 {
                    state[node * 6 + 3 + axis] =
                        element.rotation[axis * 3] * tx + element.rotation[axis * 3 + 1] * ty;
                }
            }
            let curvature = [0.4, 0.2, -0.1];
            let db = section.bending_constitutive();
            let moment: [f64; 3] = std::array::from_fn(|row| {
                (0..3)
                    .map(|col| db[row * 3 + col] * curvature[col])
                    .sum::<f64>()
            });
            let radius = point[0].hypot(point[1]);
            let tangent = [-point[1] / radius, point[0] / radius, 0.0];
            let local: [f64; 2] = std::array::from_fn(|axis| {
                (0..3)
                    .map(|i| element.rotation[i * 3 + axis] * tangent[i])
                    .sum::<f64>()
            });
            let expected = local[0].powi(2) * moment[0]
                + local[1].powi(2) * moment[1]
                + 2.0 * local[0] * local[1] * moment[2];
            let actual =
                patch_hoop_moment(&element, &points, &state, point, 1.0 / rings as f64).unwrap();
            assert!((actual - expected).abs() <= 1e-9 * expected.abs());
            assert_eq!(
                patch_hoop_moment(
                    &element,
                    &points,
                    &vec![0.0; state.len()],
                    point,
                    1.0 / rings as f64
                )
                .unwrap(),
                0.0
            );
        }
    }
}
