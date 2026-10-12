//! Nonlinear solid mechanics validation tests
//!
//! Tests the full Newton-Raphson pipeline: HyperelasticForm + VectorH1Space
//! with Dirichlet boundary conditions, starting from nonequilibrium states
//! under nonzero load. Additional tests:
//! - NeoHookean strain energy, stress, and tangent consistency
//! - Uniaxial compression against the linear reference field
//! - Traction loading with force balance and equilibrium checks
//! - Finite-strain affine patch with independent solved stress/energy references

use sonium_fem::basis::PolynomialDegree;
use sonium_fem::forms::NonlinearForm;
use sonium_fem::mesh::{ElementType, Mesh, Point};
use sonium_fem::physics::solid::nonlinear_solid::{
    HyperelasticForm, HyperelasticMaterial, NeoHookean,
};
use sonium_fem::solver::{FemNewtonConfig, fem_newton_solve_with_bcs_and_load};
use sonium_fem::spaces::{FunctionSpace, VectorH1Space};
use sonium_qa::{QaResult, QaStatus};

const CASE_ID: &str = "fem.nonlinear-solid-validation.v1";

// Bar geometry and material.
const BAR_LENGTH: f64 = 1.0;
const BAR_HEIGHT: f64 = 0.5;
const CELLS_X: usize = 5;
const CELLS_Y: usize = 4;
const YOUNGS_MODULUS: f64 = 1.0;
const POISSON_RATIO: f64 = 0.3;

// Load levels.
const COMPRESSION: f64 = 0.02;
const TRACTION: f64 = 0.02;

// Acceptance tolerances.
const TOL_BC: f64 = 1e-12;
const TOL_RESIDUAL: f64 = 1e-8;
const TOL_INTERIOR: f64 = 0.05;
const TOL_BALANCE: f64 = 1e-8;
const TOL_TIP: f64 = 0.30;
const TOL_MODEL: f64 = 1e-12;
const TOL_FINITE_DIFFERENCE: f64 = 1e-6;

fn check(condition: bool, message: String) -> Result<(), String> {
    if condition { Ok(()) } else { Err(message) }
}

fn newton_config() -> FemNewtonConfig {
    FemNewtonConfig {
        newton: sonium_solvers::NewtonConfig {
            max_iterations: 25,
            absolute_tolerance: 1e-12,
            ..Default::default()
        },
        quad_order: 2,
        ..Default::default()
    }
}

/// Bar mesh on [0, BAR_LENGTH] x [0, BAR_HEIGHT] with two triangles per cell.
fn bar_mesh() -> Mesh {
    let nx = CELLS_X + 1;
    let ny = CELLS_Y + 1;
    let mut mesh = Mesh::new(2);
    for j in 0..ny {
        for i in 0..nx {
            mesh.add_node(Point::new_2d(
                i as f64 / CELLS_X as f64 * BAR_LENGTH,
                j as f64 / CELLS_Y as f64 * BAR_HEIGHT,
            ));
        }
    }
    for j in 0..CELLS_Y {
        for i in 0..CELLS_X {
            let v0 = j * nx + i;
            let v1 = v0 + 1;
            let v2 = v0 + nx;
            let v3 = v2 + 1;
            mesh.add_element(ElementType::Triangle, vec![v0, v1, v2]);
            mesh.add_element(ElementType::Triangle, vec![v1, v3, v2]);
        }
    }
    mesh.detect_boundaries();
    mesh
}

/// Assemble the internal residual at a trial solution.
fn assemble_residual(
    form: &HyperelasticForm<NeoHookean>,
    space: &VectorH1Space,
    mesh: &Mesh,
    solution: &[f64],
) -> Vec<f64> {
    let n_dofs = space.num_global_dofs(mesh);
    let mut residual = vec![0.0; n_dofs];
    for elem_idx in 0..mesh.num_elements() {
        let dofs = space.local_to_global(mesh, elem_idx);
        let local_sol: Vec<f64> = dofs.iter().map(|&d| solution[d]).collect();
        let quad =
            sonium_fem::quadrature::QuadratureRule::new(mesh.elements[elem_idx].element_type, 2);
        let mut local_vec = sonium_fem::forms::LocalVector::zeros(dofs.len());
        for qp in quad.iter() {
            let tab = space.tabulate(mesh, elem_idx, qp.xi(), qp.eta(), qp.zeta());
            form.element_residual(&tab, &local_sol, qp.weight, &mut local_vec);
        }
        for (i, &dof) in dofs.iter().enumerate() {
            residual[dof] += local_vec.values[i];
        }
    }
    residual
}

#[derive(Debug, Clone, Copy)]
struct Metrics {
    compression_iterations: usize,
    compression_residual: f64,
    compression_bc_error: f64,
    interior_error: f64,
    compression_balance_error: f64,
    traction_iterations: usize,
    traction_residual: f64,
    traction_bc_error: f64,
    traction_equilibrium: f64,
    traction_balance_error: f64,
    tip_displacement: f64,
    tip_error: f64,
    model_checks: usize,
    tangent_error: f64,
    energy_consistency_error: f64,
    affine_patch: AffinePatchMetrics,
}

#[derive(Debug, Clone, Copy)]
struct AffinePatchMetrics {
    displacement_error: f64,
    stress_error: f64,
    energy_error: f64,
    reaction_error: f64,
    residual: f64,
    iterations: usize,
    free_dofs: usize,
}

// A finite-strain affine patch supplies independent closed-form displacement,
// first-Piola stress and stored-energy references for an actually solved field.
fn solve_affine_patch() -> Result<AffinePatchMetrics, String> {
    use sonium_fem::spaces::H1Space;
    let mesh = bar_mesh();
    let scalar = H1Space::new(PolynomialDegree::P1);
    let vector = VectorH1Space::new(PolynomialDegree::P1, 2);
    let material = NeoHookean::from_young(YOUNGS_MODULUS, POISSON_RATIO, 2);
    let form = HyperelasticForm::new(material);
    let stretches: [f64; 2] = [0.92, 1.03];
    let mu = YOUNGS_MODULUS / (2.0 * (1.0 + POISSON_RATIO));
    let lambda =
        YOUNGS_MODULUS * POISSON_RATIO / ((1.0 + POISSON_RATIO) * (1.0 - 2.0 * POISSON_RATIO));
    let log_j = (stretches[0] * stretches[1]).ln();
    let exact_stress = [
        mu * (stretches[0] - 1.0 / stretches[0]) + lambda * log_j / stretches[0],
        0.0,
        0.0,
        mu * (stretches[1] - 1.0 / stretches[1]) + lambda * log_j / stretches[1],
    ];
    let exact_energy_density = mu / 2.0 * (stretches[0].powi(2) + stretches[1].powi(2) - 2.0)
        - mu * log_j
        + lambda / 2.0 * log_j * log_j;
    let mut bcs = Vec::new();
    let count = vector.num_global_dofs(&mesh);
    let mut prescribed = vec![false; count];
    for (node, p) in mesh.nodes.iter().enumerate() {
        if p.x.abs() < 1e-12
            || (p.x - BAR_LENGTH).abs() < 1e-12
            || p.y.abs() < 1e-12
            || (p.y - BAR_HEIGHT).abs() < 1e-12
        {
            for (component, coordinate) in [p.x, p.y].into_iter().enumerate() {
                bcs.push((
                    2 * node + component,
                    (stretches[component] - 1.0) * coordinate,
                ));
                prescribed[2 * node + component] = true;
            }
        }
    }
    let free_dofs = prescribed.iter().filter(|v| !**v).count();
    check(
        free_dofs > 0,
        "affine hyperelastic patch has no free unknowns".into(),
    )?;
    let solution = fem_newton_solve_with_bcs_and_load(
        &form,
        &vector,
        &mesh,
        &bcs,
        &vec![0.0; count],
        &[],
        &newton_config(),
    )
    .map_err(|e| format!("affine hyperelastic patch Newton: {e:?}"))?;
    check(
        solution.converged && solution.iterations > 0,
        "affine hyperelastic patch did not solve from nonequilibrium".into(),
    )?;
    let mut displacement_error = 0.0_f64;
    let mut stress_error = 0.0;
    let mut energy = 0.0;
    for element in 0..mesh.num_elements() {
        let nodes = scalar.local_to_global(&mesh, element);
        for qp in sonium_fem::quadrature::QuadratureRule::new(ElementType::Triangle, 2).iter() {
            let tab = scalar.tabulate(&mesh, element, qp.xi(), qp.eta(), 0.0);
            let mut displacement = [0.0; 2];
            let mut f = [1.0, 0.0, 0.0, 1.0];
            for (local, &node) in nodes.iter().enumerate() {
                for component in 0..2 {
                    displacement[component] += solution.x[2 * node + component] * tab.value(local);
                    for dimension in 0..2 {
                        f[2 * component + dimension] +=
                            solution.x[2 * node + component] * tab.grad(local, dimension);
                    }
                }
            }
            let expected = [
                (stretches[0] - 1.0) * tab.physical_coords[0],
                (stretches[1] - 1.0) * tab.physical_coords[1],
            ];
            displacement_error = displacement_error.max(
                (0..2)
                    .map(|d| (displacement[d] - expected[d]).abs())
                    .fold(0.0_f64, f64::max)
                    / ((stretches[0] - 1.0) * BAR_LENGTH).abs(),
            );
            let stress = material.stress(&f);
            let weight = qp.weight * tab.det_j;
            stress_error += weight
                * (0..4)
                    .map(|d| (stress[d] - exact_stress[d]).powi(2))
                    .sum::<f64>();
            energy += weight * material.energy(&f);
        }
    }
    let area = BAR_LENGTH * BAR_HEIGHT;
    stress_error = (stress_error / (area * exact_stress.iter().map(|v| v * v).sum::<f64>())).sqrt();
    let energy_error = (energy / (area * exact_energy_density) - 1.0).abs();
    let internal = assemble_residual(&form, &vector, &mesh, &solution.x);
    let mut reaction_error = 0.0_f64;
    for axis in 0..2 {
        let side = [BAR_LENGTH, BAR_HEIGHT][axis];
        let face_length = [BAR_HEIGHT, BAR_LENGTH][axis];
        for (position, sign) in [(0.0, -1.0), (side, 1.0)] {
            let reaction = mesh
                .nodes
                .iter()
                .enumerate()
                .filter(|(_, p)| ([p.x, p.y][axis] - position).abs() < 1e-12)
                .map(|(node, _)| internal[2 * node + axis])
                .sum::<f64>();
            let expected = sign * exact_stress[3 * axis] * face_length;
            reaction_error = reaction_error.max((reaction - expected).abs() / expected.abs());
        }
    }
    let residual = internal
        .iter()
        .enumerate()
        .filter(|(i, _)| !prescribed[*i])
        .map(|(_, v)| v * v)
        .sum::<f64>()
        .sqrt()
        / internal
            .iter()
            .enumerate()
            .filter(|(i, _)| prescribed[*i])
            .map(|(_, v)| v * v)
            .sum::<f64>()
            .sqrt();
    for (name, value) in [
        ("displacement", displacement_error),
        ("stress", stress_error),
        ("stored energy", energy_error),
        ("reaction", reaction_error),
        ("residual", residual),
    ] {
        check(
            value.is_finite() && value < 1e-8,
            format!("affine patch {name} error {value:.3e} exceeds 1e-8"),
        )?;
    }
    Ok(AffinePatchMetrics {
        displacement_error,
        stress_error,
        energy_error,
        reaction_error,
        residual,
        iterations: solution.iterations,
        free_dofs,
    })
}

fn run() -> Result<Metrics, String> {
    let material = NeoHookean::from_young(YOUNGS_MODULUS, POISSON_RATIO, 2);
    let form = HyperelasticForm::new(material);
    let space = VectorH1Space::new(PolynomialDegree::P1, 2);
    let mesh = bar_mesh();
    let n_dofs = space.num_global_dofs(&mesh);
    let eps = 1e-6;
    let is_left = |node: usize| mesh.nodes[node].x < eps;
    let is_right = |node: usize| mesh.nodes[node].x > BAR_LENGTH - eps;

    // Compression leg: fixed left edge, prescribed compression on the right
    // edge, solved from the undeformed nonequilibrium state.
    let mut compression_bcs = Vec::new();
    for node in 0..mesh.num_nodes() {
        if is_left(node) {
            compression_bcs.push((node * 2, 0.0));
            compression_bcs.push((node * 2 + 1, 0.0));
        } else if is_right(node) {
            compression_bcs.push((node * 2, -COMPRESSION));
        }
    }
    let initial = vec![0.0; n_dofs];
    let compression = fem_newton_solve_with_bcs_and_load(
        &form,
        &space,
        &mesh,
        &compression_bcs,
        &initial,
        &[],
        &newton_config(),
    )
    .map_err(|error| format!("compression Newton failed: {error:?}"))?;
    check(
        compression.converged,
        "compression Newton did not converge".to_string(),
    )?;
    check(
        compression.iterations >= 1,
        "compression should need nonzero iterations from u=0".to_string(),
    )?;
    check(
        compression.residual <= TOL_RESIDUAL,
        format!(
            "compression residual {:.3e} exceeds {TOL_RESIDUAL:.1e}",
            compression.residual
        ),
    )?;
    let mut compression_bc_error = 0.0f64;
    for &(dof, value) in &compression_bcs {
        compression_bc_error = compression_bc_error.max((compression.x[dof] - value).abs());
    }
    check(
        compression_bc_error <= TOL_BC,
        format!("compression BC error {compression_bc_error:.3e} exceeds {TOL_BC:.1e}"),
    )?;

    // Interior nodes follow the uniform compression field at small strain.
    let mut interior_error = 0.0f64;
    let mut interior_count = 0usize;
    for node in 0..mesh.num_nodes() {
        let point = &mesh.nodes[node];
        if point.x > eps
            && point.x < BAR_LENGTH - eps
            && point.y > eps
            && point.y < BAR_HEIGHT - eps
        {
            let expected = -COMPRESSION * point.x / BAR_LENGTH;
            let error = (compression.x[node * 2] - expected).abs() / COMPRESSION;
            interior_error = interior_error.max(error);
            interior_count += 1;
        }
    }
    check(interior_count > 0, "no interior nodes found".to_string())?;
    check(
        interior_error <= TOL_INTERIOR,
        format!("interior error {interior_error:.3e} exceeds {TOL_INTERIOR:.1e}"),
    )?;

    // Global force balance: internal forces sum to zero without external load.
    let compression_residual = assemble_residual(&form, &space, &mesh, &compression.x);
    let total_force: f64 = compression_residual.iter().sum();
    let max_force: f64 = compression_residual
        .iter()
        .map(|value| value.abs())
        .fold(0.0, f64::max);
    let compression_balance_error = total_force.abs() / max_force;
    check(
        compression_balance_error <= TOL_BALANCE,
        format!("compression imbalance {compression_balance_error:.3e} exceeds {TOL_BALANCE:.1e}"),
    )?;

    // Traction leg: fixed left edge with uniform axial traction on the right
    // edge, solved from the undeformed nonequilibrium state.
    let mut traction_bcs = Vec::new();
    let mut right_nodes = Vec::new();
    for node in 0..mesh.num_nodes() {
        if is_left(node) {
            traction_bcs.push((node * 2, 0.0));
            traction_bcs.push((node * 2 + 1, 0.0));
        } else if is_right(node) {
            right_nodes.push(node);
        }
    }
    right_nodes.sort_by(|&a, &b| {
        mesh.nodes[a]
            .y
            .partial_cmp(&mesh.nodes[b].y)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut external_load = vec![0.0; n_dofs];
    for (position, &node) in right_nodes.iter().enumerate() {
        let below = if position == 0 {
            0.0
        } else {
            mesh.nodes[node].y - mesh.nodes[right_nodes[position - 1]].y
        };
        let above = if position + 1 == right_nodes.len() {
            0.0
        } else {
            mesh.nodes[right_nodes[position + 1]].y - mesh.nodes[node].y
        };
        external_load[node * 2] = TRACTION * (below + above) / 2.0;
    }
    let applied_total: f64 = external_load.iter().sum();
    check(
        (applied_total - TRACTION * BAR_HEIGHT).abs() / (TRACTION * BAR_HEIGHT) < 1e-12,
        "traction load does not integrate to the applied total".to_string(),
    )?;
    let traction = fem_newton_solve_with_bcs_and_load(
        &form,
        &space,
        &mesh,
        &traction_bcs,
        &initial,
        &external_load,
        &newton_config(),
    )
    .map_err(|error| format!("traction Newton failed: {error:?}"))?;
    check(
        traction.converged,
        "traction Newton did not converge".to_string(),
    )?;
    check(
        traction.iterations >= 1,
        "traction should need nonzero iterations from u=0".to_string(),
    )?;
    check(
        traction.residual <= TOL_RESIDUAL,
        format!(
            "traction residual {:.3e} exceeds {TOL_RESIDUAL:.1e}",
            traction.residual
        ),
    )?;
    let mut traction_bc_error = 0.0f64;
    for &(dof, value) in &traction_bcs {
        traction_bc_error = traction_bc_error.max((traction.x[dof] - value).abs());
    }
    check(
        traction_bc_error <= TOL_BC,
        format!("traction BC error {traction_bc_error:.3e} exceeds {TOL_BC:.1e}"),
    )?;

    // Free-DOF equilibrium and global balance with the external load.
    let traction_residual = assemble_residual(&form, &space, &mesh, &traction.x);
    let constrained: std::collections::HashSet<usize> =
        traction_bcs.iter().map(|&(dof, _)| dof).collect();
    let mut traction_equilibrium = 0.0f64;
    for dof in 0..n_dofs {
        if constrained.contains(&dof) {
            continue;
        }
        traction_equilibrium =
            traction_equilibrium.max((traction_residual[dof] - external_load[dof]).abs());
    }
    traction_equilibrium /= applied_total;
    check(
        traction_equilibrium <= TOL_RESIDUAL,
        format!("traction equilibrium {traction_equilibrium:.3e} exceeds {TOL_RESIDUAL:.1e}"),
    )?;
    // Internal nodal forces sum to zero by translation invariance; the
    // constrained reactions carry the applied total with opposite sign.
    let internal_total: f64 = traction_residual.iter().sum();
    let reaction_total: f64 = traction_bcs
        .iter()
        .map(|&(dof, _)| traction_residual[dof])
        .sum();
    let traction_balance_error =
        (internal_total.abs() + (reaction_total + applied_total).abs()) / applied_total;
    check(
        traction_balance_error <= TOL_BALANCE,
        format!("traction imbalance {traction_balance_error:.3e} exceeds {TOL_BALANCE:.1e}"),
    )?;

    // Loaded edge displaces in the traction direction near the linear estimate.
    let tip_displacement = right_nodes
        .iter()
        .map(|&node| traction.x[node * 2])
        .sum::<f64>()
        / right_nodes.len() as f64;
    check(
        tip_displacement > 0.0,
        "traction should pull the loaded edge forward".to_string(),
    )?;
    let linear_estimate = TRACTION * BAR_LENGTH / YOUNGS_MODULUS;
    let tip_error = (tip_displacement - linear_estimate).abs() / linear_estimate;
    check(
        tip_error <= TOL_TIP,
        format!("tip error {tip_error:.3e} exceeds {TOL_TIP:.1e}"),
    )?;

    // Material model checks against the NeoHookean implementation.
    let mut model_checks = 0usize;
    let identity = [1.0, 0.0, 0.0, 1.0];
    check(
        material.energy(&identity).abs() <= TOL_MODEL,
        "strain energy at identity should vanish".to_string(),
    )?;
    model_checks += 1;
    let reference_stress = material.stress(&identity);
    let reference_norm: f64 = reference_stress
        .iter()
        .map(|value| value * value)
        .sum::<f64>()
        .sqrt();
    check(
        reference_norm <= TOL_MODEL,
        "reference configuration should be stress free".to_string(),
    )?;
    model_checks += 1;
    check(
        material.energy(&[1.05, 0.0, 0.0, 1.05]) > 0.0,
        "volumetric stretch should store energy".to_string(),
    )?;
    model_checks += 1;

    // Tangent consistency: central differences of the stress match the
    // analytical tangent at a deformed state.
    let deformed = [1.05, 0.02, -0.01, 0.98];
    let tangent = material.tangent(&deformed);
    let step = 1e-7;
    let mut tangent_error = 0.0f64;
    for direction in 0..4 {
        let mut forward = deformed;
        let mut backward = deformed;
        forward[direction] += step;
        backward[direction] -= step;
        let stress_forward = material.stress(&forward);
        let stress_backward = material.stress(&backward);
        for row in 0..4 {
            let numeric = (stress_forward[row] - stress_backward[row]) / (2.0 * step);
            let analytic = tangent[row * 4 + direction];
            let scale = analytic.abs().max(1.0);
            tangent_error = tangent_error.max((numeric - analytic).abs() / scale);
        }
    }
    check(
        tangent_error <= TOL_FINITE_DIFFERENCE,
        format!("tangent error {tangent_error:.3e} exceeds {TOL_FINITE_DIFFERENCE:.1e}"),
    )?;

    // Energy-stress consistency: the stress is the energy gradient.
    let energy_step = [1e-5, -2e-5, 1.5e-5, 1e-5];
    let mut energy_forward = deformed;
    let mut energy_backward = deformed;
    for component in 0..4 {
        energy_forward[component] += energy_step[component];
        energy_backward[component] -= energy_step[component];
    }
    let energy_difference = material.energy(&energy_forward) - material.energy(&energy_backward);
    let linear_term: f64 = material
        .stress(&deformed)
        .iter()
        .zip(energy_step.iter())
        .map(|(stress, step)| 2.0 * stress * step)
        .sum();
    let energy_consistency_error = (energy_difference - linear_term).abs() / linear_term.abs();
    check(
        energy_consistency_error <= TOL_FINITE_DIFFERENCE,
        format!(
            "energy consistency error {energy_consistency_error:.3e} exceeds {TOL_FINITE_DIFFERENCE:.1e}"
        ),
    )?;

    Ok(Metrics {
        compression_iterations: compression.iterations,
        compression_residual: compression.residual,
        compression_bc_error,
        interior_error,
        compression_balance_error,
        traction_iterations: traction.iterations,
        traction_residual: traction.residual,
        traction_bc_error,
        traction_equilibrium,
        traction_balance_error,
        tip_displacement,
        tip_error,
        model_checks,
        tangent_error,
        energy_consistency_error,
        affine_patch: solve_affine_patch()?,
    })
}

fn main() {
    let result = match run() {
        Ok(metrics) => QaResult::passed(CASE_ID)
            .requested_backend("cpu")
            .actual_backend("cpu")
            .metric(
                "compression_iterations",
                metrics.compression_iterations as f64,
            )
            .minimum("compression_iterations", 1.0)
            .metric("compression_residual", metrics.compression_residual)
            .maximum("compression_residual", TOL_RESIDUAL)
            .metric("compression_bc_error", metrics.compression_bc_error)
            .maximum("compression_bc_error", TOL_BC)
            .metric("interior_error", metrics.interior_error)
            .maximum("interior_error", TOL_INTERIOR)
            .metric(
                "compression_balance_error",
                metrics.compression_balance_error,
            )
            .maximum("compression_balance_error", TOL_BALANCE)
            .metric("traction_iterations", metrics.traction_iterations as f64)
            .minimum("traction_iterations", 1.0)
            .metric("traction_residual", metrics.traction_residual)
            .maximum("traction_residual", TOL_RESIDUAL)
            .metric("traction_bc_error", metrics.traction_bc_error)
            .maximum("traction_bc_error", TOL_BC)
            .metric("traction_equilibrium", metrics.traction_equilibrium)
            .maximum("traction_equilibrium", TOL_RESIDUAL)
            .metric("traction_balance_error", metrics.traction_balance_error)
            .maximum("traction_balance_error", TOL_BALANCE)
            .metric("tip_displacement", metrics.tip_displacement)
            .minimum("tip_displacement", 1e-6)
            .metric("tip_error", metrics.tip_error)
            .maximum("tip_error", TOL_TIP)
            .metric("model_checks", metrics.model_checks as f64)
            .minimum("model_checks", 3.0)
            .metric("tangent_error", metrics.tangent_error)
            .maximum("tangent_error", TOL_FINITE_DIFFERENCE)
            .metric("energy_consistency_error", metrics.energy_consistency_error)
            .maximum("energy_consistency_error", TOL_FINITE_DIFFERENCE)
            .metric("affine_displacement_error",metrics.affine_patch.displacement_error).maximum("affine_displacement_error",1e-8)
            .metric("affine_stress_error",metrics.affine_patch.stress_error).maximum("affine_stress_error",1e-8)
            .metric("affine_energy_error",metrics.affine_patch.energy_error).maximum("affine_energy_error",1e-8)
            .metric("affine_reaction_error",metrics.affine_patch.reaction_error).maximum("affine_reaction_error",1e-8)
            .metric("affine_physical_residual",metrics.affine_patch.residual).maximum("affine_physical_residual",1e-8)
            .metric("affine_newton_iterations",metrics.affine_patch.iterations as f64).minimum("affine_newton_iterations",1.0)
            .metric("affine_free_dofs",metrics.affine_patch.free_dofs as f64).minimum("affine_free_dofs",1.0)
            .reference("uniform compression, traction balance, and finite-strain affine patch with independent NeoHookean stress/energy and face reactions")
            .reference_checksum("analytic/deterministic-fixture:fem-nonlinear-solid-validation-v2")
            .dimension("1.0 by 0.5 bar, 5 by 4 cells, P1 triangles")
            .formulation("NeoHookean Newton solve with Dirichlet BCs from u=0 under load")
            .tolerance("interior 5%; tip 30%; balance 1e-8; tangent FD 1e-6; affine displacement/stress/energy/reaction/residual <= 1e-8")
            .input_checksum("mesh:bar-5x4:neohookean-compression-traction-affine:v2"),
        Err(error) => QaResult::failed(CASE_ID, error)
            .requested_backend("cpu")
            .actual_backend("cpu"),
    };

    match result.to_json_line() {
        Ok(json) => println!("QA_RESULT: {json}"),
        Err(error) => {
            eprintln!("nonlinear solid QA serialization failed: {error}");
            std::process::exit(1);
        }
    }

    if matches!(result.status, QaStatus::Passed) {
        println!("SUCCESS");
    } else {
        println!("FAILED");
        std::process::exit(1);
    }
}
