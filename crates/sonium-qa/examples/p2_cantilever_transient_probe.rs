//! Native transient continuum solve of the original OpenCourant cantilever.

// Rust guideline compliant 2026-02-21

use sonium_fem::assembly::assemble_bilinear;
use sonium_fem::basis::PolynomialDegree;
use sonium_fem::forms::standard::VectorMassForm;
use sonium_fem::mesh::box_mesh_tetrahedra;
use sonium_fem::physics::solid::elasticity::{ElasticityForm, IsotropicMaterial};
use sonium_fem::solver::implicit_dynamics::{
    ImplicitDynamicsConfig, ImplicitDynamicsIntegrator, ImplicitDynamicsSystem,
};
use sonium_fem::spaces::{FunctionSpace, VectorH1Space};
use sonium_qa::rad::cantilever_case;
use sonium_solvers::{CsrMatrix, LinearOperator};

fn constrain_zero(matrix: &mut CsrMatrix<f64>, constrained: &[bool]) {
    for row in 0..matrix.num_rows {
        for entry in matrix.row_ptrs[row]..matrix.row_ptrs[row + 1] {
            let column = matrix.col_indices[entry];
            if constrained[row] {
                matrix.values[entry] = if row == column { 1.0 } else { 0.0 };
            } else if constrained[column] {
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
    points.last().unwrap().1
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let factor = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "1".into())
        .parse::<usize>()?;
    let requested_dt = std::env::args()
        .nth(2)
        .unwrap_or_else(|| "0.001".into())
        .parse::<f64>()?;
    if factor == 0 || !requested_dt.is_finite() || requested_dt <= 0.0 {
        return Err("positive mesh factor and finite positive timestep required".into());
    }
    let case = cantilever_case();
    let mesh = box_mesh_tetrahedra(
        0.0,
        1.0,
        -0.025,
        0.025,
        -0.025,
        0.025,
        10 * factor,
        2 * factor,
        2 * factor,
    );
    let space = VectorH1Space::new(PolynomialDegree::P2, 3);
    let dimension = space.num_global_dofs(&mesh);
    let coordinates = space
        .dof_coordinates(&mesh)
        .ok_or("missing nodal coordinates")?;
    let material =
        IsotropicMaterial::new(case.material.youngs_modulus_pa, case.material.poisson_ratio);
    let mut stiffness =
        assemble_bilinear(&ElasticityForm::new_3d(material), &space, &space, &mesh, 4).to_csr();
    let mut mass = assemble_bilinear(
        &VectorMassForm {
            coefficient: case.material.density_kg_per_m3,
        },
        &space,
        &space,
        &mesh,
        4,
    )
    .to_csr();
    let constrained = coordinates
        .iter()
        .enumerate()
        .filter_map(|(dof, point)| (point[0].abs() < 1e-12).then_some(dof))
        .collect::<Vec<_>>();
    let zero = vec![0.0; dimension];
    let mut is_constrained = vec![false; dimension];
    for dof in constrained {
        is_constrained[dof] = true;
    }
    constrain_zero(&mut stiffness, &is_constrained);
    constrain_zero(&mut mass, &is_constrained);
    let native_dof = |point: [f64; 3], axis: usize| {
        coordinates
            .iter()
            .enumerate()
            .find_map(|(dof, position)| {
                (dof % 3 == axis
                    && position
                        .iter()
                        .zip(point)
                        .all(|(a, b)| (*a - b).abs() < 1e-12))
                .then_some(dof)
            })
            .ok_or("original deck node absent from native nodal space")
    };
    let mut loads = Vec::new();
    for load in &case.cloads {
        for &node in &case.node_sets[load.set].nodes {
            loads.push((
                native_dof(case.mesh.nodes[node - 1], load.direction - 1)?,
                load.force_scale_n,
                load.function,
            ));
        }
    }
    let probe_node = case.probes[0]
        .channel
        .split('/')
        .nth(1)
        .ok_or("missing probe node")?
        .parse::<usize>()?;
    let probe = native_dof(case.mesh.nodes[probe_node - 1], 2)?;
    let steps = (case.stop_time_s / requested_dt).ceil() as usize;
    let dt = case.stop_time_s / steps as f64;
    println!(
        "native P2 transient: {dimension} DOFs, {} tetrahedra, {steps} steps, dt={dt:.6e}",
        mesh.num_elements()
    );
    let mut integrator = ImplicitDynamicsIntegrator::new(
        ImplicitDynamicsSystem {
            mass: mass.clone(),
            stiffness: stiffness.clone(),
            damping: None,
        },
        ImplicitDynamicsConfig::default(),
        dt,
    )?;
    integrator.initialize(&zero, &zero, &zero)?;
    let mut max_true_residual = 0.0_f64;
    for step in 1..=steps {
        let time = step as f64 * dt;
        let mut force = zero.clone();
        for &(dof, scale, function) in &loads {
            force[dof] += scale * curve(&case.functions[function].points, time);
        }
        let force_norm = force.iter().map(|value| value * value).sum::<f64>().sqrt();
        let state = integrator.step_checked(&force, 0.5e-8 * force_norm.max(1e-12))?;
        let inertia = mass.apply(&state.acceleration);
        let elastic = stiffness.apply(&state.displacement);
        let norm = force.iter().map(|value| value * value).sum::<f64>().sqrt();
        let residual = inertia
            .iter()
            .zip(&elastic)
            .zip(&force)
            .map(|((m, k), f)| (m + k - f).powi(2))
            .sum::<f64>()
            .sqrt()
            / norm.max(1e-12);
        if !residual.is_finite() {
            return Err("native equation residual is non-finite".into());
        }
        max_true_residual = max_true_residual.max(residual);
    }
    let actual = integrator.state().displacement[probe];
    let expected = case.probes[0].expected;
    println!(
        "final tip DZ={actual:.12e}, analytical={expected:.12e}, relative error={:.6e}, max independent dynamic residual={max_true_residual:.6e}",
        (actual - expected).abs() / expected.abs()
    );
    if max_true_residual > 1e-8 {
        return Err("native true dynamic residual exceeds 1e-8".into());
    }
    Ok(())
}
