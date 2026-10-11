//! Wolfram golden replay: tied-interface patch test.
//!
//! Replays `fixtures/solids_tied_contact_patch_golden.json` (derived by
//! `fixtures/solids_tied_contact_patch.wls` at 50-digit precision) through
//! the production tied node-to-surface interface plus H8 assembly and dense
//! solve: two non-matching dissimilar-E blocks transmit uniform pressure.
//! Regenerate with:
//! `wolframscript -script fixtures/solids_tied_contact_patch.wls > fixtures/solids_tied_contact_patch_golden.json`.

// Rust guideline compliant 2026-02-21

use serde_json::Value;
use sonium_fem::physics::solid::contact::explicit::TiedInterface;
use sonium_fem::physics::solid::elasticity::IsotropicMaterial;
use sonium_fem::physics::solid::nonlinear_solid::{
    HexMesh, dense_solve, hex_jacobian, hex_quadrature, hex_shape_derivatives, physical_gradients,
};

const GOLDEN: &str = include_str!("../fixtures/solids_tied_contact_patch_golden.json");

fn golden_values(doc: &Value, name: &str) -> Vec<f64> {
    doc["values"][name]["values"]
        .as_array()
        .unwrap_or_else(|| panic!("golden lacks {name}"))
        .iter()
        .map(|entry| {
            entry
                .as_str()
                .unwrap_or_else(|| panic!("golden lacks {name}"))
                .parse()
                .unwrap()
        })
        .collect()
}

fn check(doc: &Value, name: &str, index: usize, actual: f64) {
    let expected = golden_values(doc, name)[index];
    let abs: f64 = doc["tolerances"][format!("{name}_abs")].as_f64().unwrap();
    let rel: f64 = doc["tolerances"][format!("{name}_rel")].as_f64().unwrap();
    assert!(
        (actual - expected).abs() <= abs + rel * expected.abs(),
        "{name}[{index}]: production {actual:.6e} vs golden {expected:.6e}"
    );
}

/// Linear-elastic H8 element stiffness (24x24, Voigt B^T C B).
fn element_stiffness(coords: &[[f64; 3]; 8], voigt: &[f64; 36]) -> [f64; 576] {
    let mut stiffness = [0.0; 576];
    for (natural, weight) in hex_quadrature() {
        let derivatives = hex_shape_derivatives(&natural);
        let (_, det, inv) = hex_jacobian(coords, &derivatives);
        let grad = physical_gradients(&inv, &derivatives);
        // Strain-displacement rows for Voigt [xx, yy, zz, yz, xz, xy].
        let mut b = [[0.0; 24]; 6];
        for (node, g) in grad.iter().enumerate() {
            b[0][node * 3] = g[0];
            b[1][node * 3 + 1] = g[1];
            b[2][node * 3 + 2] = g[2];
            b[3][node * 3 + 1] = g[2];
            b[3][node * 3 + 2] = g[1];
            b[4][node * 3] = g[2];
            b[4][node * 3 + 2] = g[0];
            b[5][node * 3] = g[1];
            b[5][node * 3 + 1] = g[0];
        }
        let scale = weight * det;
        for i in 0..24 {
            for j in 0..24 {
                let mut sum = 0.0;
                for m in 0..6 {
                    for n in 0..6 {
                        sum += b[m][i] * voigt[m * 6 + n] * b[n][j];
                    }
                }
                stiffness[i * 24 + j] += scale * sum;
            }
        }
    }
    stiffness
}

/// Cauchy stress at a natural point of one element (Voigt 6).
fn element_stress(
    coords: &[[f64; 3]; 8],
    natural: &[f64; 3],
    element_displacements: &[f64; 24],
    voigt: &[f64; 36],
) -> [f64; 6] {
    let derivatives = hex_shape_derivatives(natural);
    let (_, _, inv) = hex_jacobian(coords, &derivatives);
    let grad = physical_gradients(&inv, &derivatives);
    let mut strain = [0.0; 6];
    for (node, g) in grad.iter().enumerate() {
        let u = [
            element_displacements[node * 3],
            element_displacements[node * 3 + 1],
            element_displacements[node * 3 + 2],
        ];
        strain[0] += g[0] * u[0];
        strain[1] += g[1] * u[1];
        strain[2] += g[2] * u[2];
        strain[3] += g[2] * u[1] + g[1] * u[2];
        strain[4] += g[2] * u[0] + g[0] * u[2];
        strain[5] += g[1] * u[0] + g[0] * u[1];
    }
    let mut stress = [0.0; 6];
    for m in 0..6 {
        for n in 0..6 {
            stress[m] += voigt[m * 6 + n] * strain[n];
        }
    }
    stress
}

#[test]
fn tied_contact_patch_matches_wolfram() {
    let doc: Value = serde_json::from_str(GOLDEN).unwrap();
    assert_eq!(doc["meta"]["protocol"], "solids-golden-v1");
    assert_eq!(doc["meta"]["units"], "SI");
    let e_bottom = doc["inputs"]["youngs_bottom_pa"].as_f64().unwrap();
    let e_top = doc["inputs"]["youngs_top_pa"].as_f64().unwrap();
    let nu = doc["inputs"]["poisson_ratio"].as_f64().unwrap();
    let pressure = doc["inputs"]["pressure_pa"].as_f64().unwrap();

    // Non-matching stack: bottom 2x1x1 over z in [0, 1], top 1x1x1 over
    // z in [1, 2]. The tie is load-bearing: without it the top block is
    // a free rigid body under pressure.
    let bottom = HexMesh::brick(2, 1, 1, [0.0, 0.0, 0.0], [2.0, 1.0, 1.0]);
    let top = HexMesh::brick(1, 1, 1, [0.0, 0.0, 1.0], [2.0, 1.0, 2.0]);
    let top_offset = bottom.num_nodes();
    let num_nodes = top_offset + top.num_nodes();
    let mut coords = Vec::with_capacity(3 * num_nodes);
    for mesh in [&bottom, &top] {
        for node in &mesh.coords {
            coords.extend_from_slice(node);
        }
    }

    // Assemble both blocks with their own moduli into one dense system.
    let num_dofs = 3 * num_nodes;
    let mut stiffness = vec![0.0; num_dofs * num_dofs];
    let mut rhs = vec![0.0; num_dofs];
    let materials = [
        IsotropicMaterial::new(e_bottom, nu).voigt_3d(),
        IsotropicMaterial::new(e_top, nu).voigt_3d(),
    ];
    for (mesh, offset, voigt) in [
        (&bottom, 0, &materials[0]),
        (&top, top_offset, &materials[1]),
    ] {
        for elem in 0..mesh.conn.len() {
            let element_coords = mesh.element_coords(elem);
            let element_matrix = element_stiffness(&element_coords, voigt);
            for (slot, node) in mesh.conn[elem].iter().enumerate() {
                for component in 0..3 {
                    let row = 3 * (node + offset) + component;
                    for (other_slot, other) in mesh.conn[elem].iter().enumerate() {
                        for other_component in 0..3 {
                            let col = 3 * (other + offset) + other_component;
                            stiffness[row * num_dofs + col] += element_matrix
                                [(slot * 3 + component) * 24 + other_slot * 3 + other_component];
                        }
                    }
                }
            }
        }
    }

    // Confined sides, fixed base, uniform top pressure.
    let mut fixed = vec![false; num_dofs];
    let node_xyz = |node: usize| [coords[3 * node], coords[3 * node + 1], coords[3 * node + 2]];
    for node in 0..num_nodes {
        let [x, y, z] = node_xyz(node);
        if x == 0.0 || x == 2.0 {
            fixed[3 * node] = true;
        }
        if y == 0.0 || y == 1.0 {
            fixed[3 * node + 1] = true;
        }
        if z == 0.0 {
            fixed[3 * node + 2] = true;
        }
    }
    // Consistent pressure load on the top face via 2x2 Gauss.
    let gauss = 1.0 / 3.0_f64.sqrt();
    for node in 0..num_nodes {
        if node_xyz(node)[2] != 2.0 {
            continue;
        }
        let [x, y, _] = node_xyz(node);
        let (sx, sy) = (
            if x == 0.0 { -1.0 } else { 1.0 },
            if y == 0.0 { -1.0 } else { 1.0 },
        );
        let mut load = 0.0;
        for gx in [-gauss, gauss] {
            for gy in [-gauss, gauss] {
                // Bilinear face shape times the face Jacobian (1 x 0.5).
                load += 0.25 * (1.0 + sx * gx) * (1.0 + sy * gy) * 0.5;
            }
        }
        rhs[3 * node + 2] -= pressure * load;
    }
    let apply_dirichlet = |stiffness: &mut [f64], rhs: &mut [f64]| {
        for dof in 0..num_dofs {
            if !fixed[dof] {
                continue;
            }
            for k in 0..num_dofs {
                stiffness[dof * num_dofs + k] = 0.0;
                stiffness[k * num_dofs + dof] = 0.0;
            }
            stiffness[dof * num_dofs + dof] = 1.0;
            rhs[dof] = 0.0;
        }
    };
    apply_dirichlet(&mut stiffness, &mut rhs);

    // Tie the fine bottom face to the coarse top face, leaving
    // Dirichlet-fixed slave components to their supports.
    let slaves: Vec<usize> = (0..bottom.num_nodes())
        .filter(|node| bottom.coords[*node][2] == 1.0)
        .collect();
    assert_eq!(slaves.len(), 6);
    // Coarse top bottom-quad (local nodes 0, 1, 3 and 0, 3, 2).
    let (t0, t1, t2, t3) = (top_offset, top_offset + 1, top_offset + 2, top_offset + 3);
    let master = [[t0, t1, t3], [t0, t3, t2]];
    let mut tie = TiedInterface::tie_nodes_to_surface(&slaves, &master, &coords, 1.0e12);
    assert_eq!(tie.num_pairs(), 6);
    for pair in tie.pairs_mut() {
        for component in 0..3 {
            if fixed[3 * pair.slave + component] {
                pair.mask[component] = false;
            }
        }
    }
    tie.eliminate(&mut stiffness, &mut rhs, num_nodes);
    apply_dirichlet(&mut stiffness, &mut rhs);
    dense_solve(&mut stiffness, &mut rhs, num_dofs);
    let mut displacements = rhs;
    tie.recover(&mut displacements);

    // The penalty twin of the tie sees no residual at the solution.
    let mut deformed = coords.clone();
    for (slot, value) in displacements.iter().enumerate() {
        deformed[slot] += value;
    }
    let mut tie_forces = vec![0.0; num_dofs];
    tie.apply_penalty(&deformed, &mut tie_forces);
    let max_tie_force = tie_forces.iter().map(|f| f.abs()).fold(0.0, f64::max);
    assert!(
        max_tie_force < 1.0,
        "tied penalty residual {max_tie_force:.3e} N"
    );

    // Axial stress at the golden probe points (Voigt index 2).
    let probes = doc["inputs"]["probe_points_m"].as_array().unwrap();
    assert_eq!(probes.len(), 4);
    for (index, probe) in probes.iter().enumerate() {
        let point = [
            probe[0].as_f64().unwrap(),
            probe[1].as_f64().unwrap(),
            probe[2].as_f64().unwrap(),
        ];
        let (mesh, offset, voigt) = if point[2] < 1.0 {
            (&bottom, 0, &materials[0])
        } else {
            (&top, top_offset, &materials[1])
        };
        let mut found = None;
        for elem in 0..mesh.conn.len() {
            let element_coords = mesh.element_coords(elem);
            let mut min = [f64::INFINITY; 3];
            let mut max = [f64::NEG_INFINITY; 3];
            for node in element_coords.iter() {
                for axis in 0..3 {
                    min[axis] = min[axis].min(node[axis]);
                    max[axis] = max[axis].max(node[axis]);
                }
            }
            if (0..3).all(|axis| point[axis] >= min[axis] && point[axis] <= max[axis]) {
                let natural = [
                    2.0 * (point[0] - min[0]) / (max[0] - min[0]) - 1.0,
                    2.0 * (point[1] - min[1]) / (max[1] - min[1]) - 1.0,
                    2.0 * (point[2] - min[2]) / (max[2] - min[2]) - 1.0,
                ];
                found = Some((elem, natural));
                break;
            }
        }
        let (elem, natural) = found.expect("probe must sit in one element");
        let mut element_displacements = [0.0; 24];
        for (slot, node) in mesh.conn[elem].iter().enumerate() {
            for component in 0..3 {
                element_displacements[slot * 3 + component] =
                    displacements[3 * (node + offset) + component];
            }
        }
        let stress = element_stress(
            &mesh.element_coords(elem),
            &natural,
            &element_displacements,
            voigt,
        );
        check(&doc, "sigma_zz", index, stress[2]);
    }
}
