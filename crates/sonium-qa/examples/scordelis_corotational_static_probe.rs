//! Diagnose full-load roof equilibrium using the existing native nonlinear shell.
//!
//! This static investigation cannot replace the original complete8s dynamic
//! gate. It preserves geometry, material, thickness, physical diaphragm
//! constraints and full gravity. Every original quad is split into two native
//! corotational CST/DKT triangles. One axial translation fixes only the global
//! rigid-translation gauge, which a static solve otherwise leaves singular.

// Rust guideline compliant 2026-02-21

use sonium_fem::assembly::assemble_nonlinear_element_form;
use sonium_fem::forms::NonlinearElementForm;
use sonium_fem::mesh::Point;
use sonium_fem::physics::solid::shell::{CorotationalShellForm, ShellElement, ShellSection};
use sonium_qa::rad::{scordelis_lo_shell_mesh, shell_scordelis_lo_case};
use sonium_solvers::direct::SparseLuFactorization;

fn residual(form: &CorotationalShellForm, u: &[f64], load: &[f64], fixed: &[bool]) -> Vec<f64> {
    // Reassemble forces directly, independently of the tangent/linear solve.
    let mut r = load.iter().map(|v| -v).collect::<Vec<_>>();
    for element in 0..form.num_elements() {
        let dofs = form.local_to_global(element);
        for (&dof, force) in dofs.iter().zip(form.element_residual(element, u)) {
            r[dof] += force;
        }
    }
    for (value, &pinned) in r.iter_mut().zip(fixed) {
        if pinned {
            *value = 0.0;
        }
    }
    r
}

fn norm(values: &[f64]) -> f64 {
    values.iter().map(|v| v * v).sum::<f64>().sqrt()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cells: usize = std::env::args().nth(1).map_or(Ok(12), |v| v.parse())?;
    if !(4..=64).contains(&cells) || !cells.is_multiple_of(2) {
        return Err("even roof cells in[4,64] required".into());
    }
    let original = shell_scordelis_lo_case();
    let mesh = scordelis_lo_shell_mesh(cells, cells);
    let material = original.base.material;
    let thickness = original
        .shell_prop
        .as_ref()
        .ok_or("roof thickness missing")?
        .thickness_m;
    let gravity = original
        .gravity
        .as_ref()
        .ok_or("roof gravity missing")?
        .value_m_per_s2;
    let section = ShellSection::new(
        material.youngs_modulus_pa,
        material.poisson_ratio,
        thickness,
        material.density_kg_per_m3,
    );
    let points = mesh
        .nodes
        .iter()
        .map(|p| Point::new_3d(p[0], p[1], p[2]))
        .collect::<Vec<_>>();
    let mut elements = Vec::new();
    for quad in &mesh.quads {
        for local in [[0, 1, 2], [0, 2, 3]] {
            let ids = local.map(|i| quad[i] - 1);
            elements.push(ShellElement::from_points(
                ids,
                &points[ids[0]],
                &points[ids[1]],
                &points[ids[2]],
                section,
            ));
        }
    }
    let dofs = 6 * points.len();
    let mut load = vec![0.0; dofs];
    let mut fixed = vec![false; dofs];
    for element in &elements {
        // Original gravity on the complete physical initial mass, not a tuned
        // edge force. Linear triangles give equal area weights to their nodes.
        for node in element.nodes {
            load[6 * node + 1] +=
                gravity * material.density_kg_per_m3 * thickness * element.area / 3.0;
        }
    }
    for &node in mesh.end_lo.iter().chain(&mesh.end_hi) {
        fixed[6 * (node - 1)] = true;
        fixed[6 * (node - 1) + 1] = true;
    }
    // Remove one rigid axial translation; retain every physical rotation.
    fixed[2] = true;
    let form = CorotationalShellForm::new(elements, dofs);
    let mut u = vec![0.0; dofs];
    let mut levels = Vec::new();
    for step in 1..=10 {
        let fraction = step as f64 / 10.0;
        let force = load.iter().map(|v| fraction * v).collect::<Vec<_>>();
        let denominator = norm(
            &force
                .iter()
                .zip(&fixed)
                .map(|(&v, &p)| if p { 0.0 } else { v })
                .collect::<Vec<_>>(),
        );
        let mut converged = false;
        for iteration in 0..40 {
            let r = residual(&form, &u, &force, &fixed);
            let relative = norm(&r) / denominator;
            println!(
                "roof{cells}: fraction={fraction:.2}, Newton iteration={iteration}, trueR={relative:.12e}"
            );
            if !relative.is_finite() {
                return Err("nonfinite native equilibrium residual".into());
            }
            if relative <= 1e-8 {
                println!(
                    "roof{cells}: full-load fraction={fraction:.2},iterations={iteration},trueR={relative:.6e},edgeDY={:.12e}",
                    u[6 * (mesh.probe_node - 1) + 1]
                );
                levels.push(serde_json::json!({"load_fraction":fraction,"iterations":iteration,"true_relative_residual":relative,"edge_dy_m":u[6*(mesh.probe_node-1)+1]}));
                converged = true;
                break;
            }
            let (_, tangent) = assemble_nonlinear_element_form(&form, &u);
            let mut matrix = tangent.to_csr();
            for row in 0..dofs {
                for entry in matrix.row_ptrs[row]..matrix.row_ptrs[row + 1] {
                    let column = matrix.col_indices[entry];
                    if fixed[row] || fixed[column] {
                        matrix.values[entry] = if row == column { 1.0 } else { 0.0 };
                    }
                }
            }
            let rhs = r.iter().map(|v| -v).collect::<Vec<_>>();
            let delta = SparseLuFactorization::factorize(&matrix)?.solve(&rhs)?;
            let linear_residual = (0..dofs)
                .map(|row| {
                    matrix
                        .row_range(row)
                        .map(|entry| matrix.values[entry] * delta[matrix.col_indices[entry]])
                        .sum::<f64>()
                        - rhs[row]
                })
                .collect::<Vec<_>>();
            let linear_relative = norm(&linear_residual) / norm(&rhs);
            println!("roof{cells}: independent Newton linear-solve R={linear_relative:.12e}");
            if !linear_relative.is_finite() || linear_relative > 1e-8 {
                return Err("native corotational Newton linear solve residual failed".into());
            }
            let mut accepted = None;
            let mut alpha = 1.0;
            for _ in 0..24 {
                let candidate = u
                    .iter()
                    .zip(&delta)
                    .map(|(&v, &d)| v + alpha * d)
                    .collect::<Vec<_>>();
                let next = norm(&residual(&form, &candidate, &force, &fixed));
                if next.is_finite() && next <= (1.0 - 1e-4 * alpha) * norm(&r) {
                    accepted = Some(candidate);
                    break;
                }
                alpha *= 0.5;
            }
            u = accepted.ok_or("native corotational equilibrium line search failed")?;
        }
        if !converged {
            return Err("native corotational equilibrium did not converge".into());
        }
    }
    let result = serde_json::json!({"status":"full-load static nonlinear diagnostic; full8s native/external QA parity unproven",
        "force_model":"elastic virtual work; full corotated-displacement derivative transpose",
        "surface_cells":cells,"triangles":form.num_elements(),"dofs":dofs,"load_levels":levels,
        "edge_dy_m":u[6*(mesh.probe_node-1)+1],"global_solution":u,
        "scope":"Original roof geometry/E/nu/density/thickness/full gravity/physical diaphragm XY supports; native corotational CST/DKT triangles; only global axial translation gauge pinned. Static equilibrium investigation, no original transient completion claim."});
    std::fs::write(
        std::env::var("P2_SCORDELIS_COROTATIONAL_OUTPUT")?,
        serde_json::to_vec_pretty(&result)?,
    )?;
    Ok(())
}
