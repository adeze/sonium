//! Phase-2 gate driver: elastic Taylor-class impact (native run).
//!
//! Mirrors the engine C3 leg `qa-fem-radioss-taylor-bar` (same frame: bar at
//! rest, rigid wall rising at 227 m/s; Neo-Hookean E=117GPa/nu=0.35/rho=8930;
//! 8x6 cylinder mesh; stop 4us) using ONLY Phase-1/2 production APIs:
//! sonium-compute explicit state/dt, 2A `ExplicitStepDriver`, 2B
//! `rigid_wall_contact`, H8 internal forces.
//!
//! Compares the two C3 probes against
//! `crates/sonium-qa/decks/taylor_bar/probes.json` (engine-verified
//! expectations; NO engine needed at gate time):
//!  Probe 0 `FNZ/RWALL/1` plateau, rel tol 25%.
//!  Probe 1 `DZ/{face}` travel,   rel tol  5%.
//! Also asserts bounded energy drift and full-stop stability. Exit 0 = gate
//! PASS, exit 1 = FAIL with per-probe report on stdout.

// Rust guideline compliant 2026-02-21

use sonium_compute::explicit_dt::critical_dt;
use sonium_compute::explicit_state::ExplicitState;
use sonium_fem::physics::solid::contact::explicit::{
    RigidWall, rigid_wall_contact, suggested_penalty,
};
use sonium_fem::physics::solid::nonlinear_solid::{
    HexMesh, NeoHookean, hex_jacobian, hex_quadrature, hex_shape_derivatives, q1_element_residual,
};
use sonium_qa::rad::assert_probe_close;
use sonium_solvers::time_integration::{AdaptiveDtConfig, ExplicitStepDriver};
use std::cell::RefCell;
use std::path::PathBuf;

const LENGTH: f64 = 0.0254;
const RADIUS: f64 = 0.003175;
const VELOCITY: f64 = 227.0;
const YOUNGS: f64 = 117.0e9;
const POISSON: f64 = 0.35;
const DENSITY: f64 = 8930.0;
const NZ: usize = 8;
const N: usize = 6;
/// Run stop matching the C3 deck.
const TSTOP: f64 = 4.0e-6;
/// Wall force plateau average window `[PLATEAU_START, TSTOP]`.
const PLATEAU_START: f64 = 2.0e-6;
/// Documented standard factor for [`suggested_penalty`].
const PENALTY_FACTOR: f64 = 50.0;
/// Gate fails when `|drift| / (KE + |IE|)` exceeds this fraction.
const DRIFT_TOL: f64 = 0.03;
/// Step cap so a stalled run fails instead of hanging.
const MAX_STEPS: u64 = 200_000;
/// Twelve edges of the H8 in [`HEX_NODES`](sonium_fem::physics::solid::nonlinear_solid::HEX_NODES) slot order.
const HEX_EDGES: [(usize, usize); 12] = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 0),
    (4, 5),
    (5, 6),
    (6, 7),
    (7, 4),
    (0, 4),
    (1, 5),
    (2, 6),
    (3, 7),
];

/// Ported cylinder mesh: node coordinates, H8 connectivity, impact-face
/// node set, and face-center node.
type CylinderMesh = (Vec<[f64; 3]>, Vec<[usize; 8]>, Vec<usize>, usize);

/// Bricks a cylinder: exact 0-based port of the C3 deck `cylinder_mesh`.
///
/// Clips an `n` by `n` Cartesian grid per layer to the circular section
/// and projects region-boundary nodes radially; the impact face floats
/// one approach gap above the wall. Returns the node coordinates, the
/// H8 connectivity, the impact-face node set, and the face-center node.
fn cylinder_mesh(nz: usize, n: usize) -> CylinderMesh {
    let gap = VELOCITY * 1.0e-6;
    let step = 2.0 * RADIUS / n as f64;
    let corner = |i: usize| -RADIUS + step * i as f64;
    let mut kept = vec![false; n * n];
    for i in 0..n {
        for j in 0..n {
            kept[i * n + j] = (0..=1).all(|di| {
                (0..=1).all(|dj| {
                    let (x, y) = (corner(i + di), corner(j + dj));
                    x * x + y * y <= RADIUS * RADIUS
                })
            });
        }
    }
    let mut layer_id = vec![0usize; (n + 1) * (n + 1)];
    let mut layer_nodes: Vec<[f64; 2]> = Vec::new();
    for i in 0..=n {
        for j in 0..=n {
            let mut used = false;
            let mut boundary = false;
            for di in 0..=1 {
                for dj in 0..=1 {
                    let (ci, cj) = (i + 1 - di, j + 1 - dj);
                    if ci == 0 || cj == 0 || ci > n || cj > n {
                        continue;
                    }
                    if kept[(ci - 1) * n + (cj - 1)] {
                        used = true;
                    } else {
                        boundary = true;
                    }
                }
            }
            if i == 0 || j == 0 || i == n || j == n {
                boundary = true;
            }
            if !used {
                continue;
            }
            let (mut x, mut y) = (corner(i), corner(j));
            if boundary {
                let r = (x * x + y * y).sqrt();
                if r > 1e-12 {
                    x *= RADIUS / r;
                    y *= RADIUS / r;
                }
            }
            layer_id[i * (n + 1) + j] = layer_nodes.len() + 1;
            layer_nodes.push([x, y]);
        }
    }
    let per_layer = layer_nodes.len();
    let mut nodes = Vec::with_capacity(per_layer * (nz + 1));
    for k in 0..=nz {
        let z = gap + LENGTH * k as f64 / nz as f64;
        for xy in &layer_nodes {
            nodes.push([xy[0], xy[1], z]);
        }
    }
    let mut bricks = Vec::new();
    for k in 0..nz {
        for i in 0..n {
            for j in 0..n {
                if !kept[i * n + j] {
                    continue;
                }
                // 1-based layer ids shifted to 0-based node numbering.
                let at = |di: usize, dj: usize| layer_id[(i + di) * (n + 1) + (j + dj)] - 1;
                let base = k * per_layer;
                bricks.push([
                    base + at(0, 0),
                    base + at(1, 0),
                    base + at(1, 1),
                    base + at(0, 1),
                    base + per_layer + at(0, 0),
                    base + per_layer + at(1, 0),
                    base + per_layer + at(1, 1),
                    base + per_layer + at(0, 1),
                ]);
            }
        }
    }
    let face_set: Vec<usize> = (0..per_layer).collect();
    // Node nearest the section centroid on the impact face.
    let face_center = layer_nodes
        .iter()
        .enumerate()
        .min_by(|a, b| {
            let ra = a.1[0] * a.1[0] + a.1[1] * a.1[1];
            let rb = b.1[0] * b.1[0] + b.1[1] * b.1[1];
            ra.total_cmp(&rb)
        })
        .map(|(index, _)| index)
        .unwrap_or(0);
    (nodes, bricks, face_set, face_center)
}

/// Reference volume of one element by Gauss quadrature of `det J`.
fn element_volume(mesh: &HexMesh, elem: usize) -> f64 {
    let coords = mesh.element_coords(elem);
    let mut volume = 0.0;
    for (natural, weight) in hex_quadrature() {
        let derivatives = hex_shape_derivatives(&natural);
        let (_, det, _) = hex_jacobian(&coords, &derivatives);
        volume += weight * det;
    }
    volume
}

/// Minimum edge length of one element in the reference configuration.
fn min_edge_length(mesh: &HexMesh, elem: usize) -> f64 {
    let coords = mesh.element_coords(elem);
    let mut shortest = f64::INFINITY;
    for (a, b) in HEX_EDGES {
        let dx = coords[a][0] - coords[b][0];
        let dy = coords[a][1] - coords[b][1];
        let dz = coords[a][2] - coords[b][2];
        shortest = shortest.min((dx * dx + dy * dy + dz * dz).sqrt());
    }
    shortest
}

/// One engine-verified probe expectation from `probes.json`.
struct ProbeExpectation {
    label: String,
    channel: String,
    expected: f64,
    abs_tol: f64,
    rel_tol: f64,
}

/// Reads the two C3 probe expectations from the checked-in deck file.
fn read_probe_expectations() -> Result<Vec<ProbeExpectation>, String> {
    let path: PathBuf =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("decks/taylor_bar/probes.json");
    let text = std::fs::read_to_string(&path)
        .map_err(|error| format!("cannot read {}: {error}", path.display()))?;
    let value: serde_json::Value = serde_json::from_str(&text)
        .map_err(|error| format!("cannot parse probes.json: {error}"))?;
    let probes = value
        .get("probes")
        .and_then(serde_json::Value::as_array)
        .ok_or_else(|| "probes.json has no probes array".to_string())?;
    if probes.len() < 2 {
        return Err(format!("probes.json holds {} probes, need 2", probes.len()));
    }
    let mut out = Vec::new();
    for probe in probes.iter().take(2) {
        let get = |key: &str| {
            probe
                .get(key)
                .ok_or_else(|| format!("probe is missing {key}"))
        };
        out.push(ProbeExpectation {
            label: get("label")?
                .as_str()
                .ok_or_else(|| "probe label is not a string".to_string())?
                .to_string(),
            channel: get("channel")?
                .as_str()
                .ok_or_else(|| "probe channel is not a string".to_string())?
                .to_string(),
            expected: get("expected")?
                .as_f64()
                .ok_or_else(|| "probe expected is not a number".to_string())?,
            abs_tol: get("abs_tol")?
                .as_f64()
                .ok_or_else(|| "probe abs_tol is not a number".to_string())?,
            rel_tol: get("rel_tol")?
                .as_f64()
                .ok_or_else(|| "probe rel_tol is not a number".to_string())?,
        });
    }
    Ok(out)
}

fn main() {
    if let Err(error) = run_gate() {
        eprintln!("{error}");
        std::process::exit(1);
    }
}

fn run_gate() -> Result<(), String> {
    let expectations = read_probe_expectations()?;
    if expectations[0].channel != "FNZ/RWALL/1" {
        return Err(format!(
            "probe 0 channel is {}, expected FNZ/RWALL/1",
            expectations[0].channel
        ));
    }
    // The DZ channel names the deck's 1-based face-center node (DZ/13);
    // our 0-based port must resolve to the same node.
    let deck_face_center: usize = expectations[1]
        .channel
        .strip_prefix("DZ/")
        .and_then(|id| id.parse::<usize>().ok())
        .ok_or_else(|| {
            format!(
                "probe 1 channel is {}, expected DZ/<node>",
                expectations[1].channel
            )
        })?;

    let (coords, conn, _face_set, face_center) = cylinder_mesh(NZ, N);
    if face_center + 1 != deck_face_center {
        return Err(format!(
            "mesh port drifted: face center is 0-based {face_center}, deck probes DZ/{deck_face_center}"
        ));
    }
    let mesh = HexMesh::new(coords.clone(), conn);
    let num_nodes = mesh.num_nodes();
    let num_elems = mesh.conn.len();
    println!("mesh: {num_nodes} nodes, {num_elems} bricks, face center node {face_center}");

    // Lumped masses from reference element volumes; every ported node
    // belongs to at least one brick, so every mass stays positive.
    let mut masses = vec![0.0; num_nodes];
    let mut char_lengths = Vec::with_capacity(num_elems);
    for elem in 0..num_elems {
        let volume = element_volume(&mesh, elem);
        if !volume.is_finite() || volume <= 0.0 {
            return Err(format!(
                "element {elem} has non-positive volume {volume:.6e}"
            ));
        }
        for node in mesh.conn[elem] {
            masses[node] += DENSITY * volume / 8.0;
        }
        char_lengths.push(min_edge_length(&mesh, elem));
    }
    let min_mass = masses.iter().cloned().fold(f64::INFINITY, f64::min);
    if !min_mass.is_finite() || min_mass <= 0.0 {
        return Err(format!("non-positive lumped mass {min_mass:.6e}"));
    }
    let min_char = char_lengths.iter().cloned().fold(f64::INFINITY, f64::min);

    let material = NeoHookean::from_young(YOUNGS, POISSON, 3);
    // Dilatational (max) wavespeed for the 3D CFL bound.
    let dilatational = ((material.lambda + 2.0 * material.mu) / DENSITY).sqrt();
    let sound_speeds = vec![dilatational; num_elems];
    let dt_crit = critical_dt(&char_lengths, &sound_speeds);
    println!(
        "dt_crit bulk: {dt_crit:.6e} s (c_dil {dilatational:.6e} m/s, Lc_min {min_char:.6e} m)"
    );

    // Penalty contact is an undamped oscillator on the lumped mass, so
    // the contact CFL `dt <= 2*sqrt(m/k)` caps the adaptive step; the
    // cap below holds a 0.5 margin on that limit.
    let penalty = suggested_penalty(YOUNGS, min_char, PENALTY_FACTOR);
    let dt_contact = 2.0 * (min_mass / penalty).sqrt();
    let dt_max = 0.5 * dt_contact;
    println!("penalty: {penalty:.6e} N/m, dt_max contact: {dt_max:.6e} s");

    let config = AdaptiveDtConfig::new(0.9, 1.0e-12, dt_max, 1.1)
        .map_err(|error| format!("adaptive dt config rejected: {error}"))?;
    let mut driver = ExplicitStepDriver::new(config)
        .map_err(|error| format!("step driver rejected: {error}"))?;

    let mut state = ExplicitState::new(num_nodes);
    for (node, xyz) in coords.iter().enumerate() {
        state.positions[3 * node] = xyz[0];
        state.positions[3 * node + 1] = xyz[1];
        state.positions[3 * node + 2] = xyz[2];
    }
    state.masses.copy_from_slice(&masses);
    // Bar starts at rest: zero initial velocities, no damping anywhere.
    let reference: Vec<f64> = state.positions.clone();
    // Slave set covers the whole bar like the deck's `bar` group, since
    // mushrooming can spread contact past the initial face.
    let slaves: Vec<usize> = (0..num_nodes).collect();
    // Wall-force history `(time, FNZ on wall)` sampled by the force closure.
    let history: RefCell<Vec<(f64, f64)>> = RefCell::new(Vec::new());

    let internal_force = |positions: &[f64], _: &[f64], _: f64| {
        let displacement: Vec<f64> = positions
            .iter()
            .zip(reference.iter())
            .map(|(x, x0)| x - x0)
            .collect();
        let mut forces = vec![0.0; positions.len()];
        for elem in 0..num_elems {
            // `q1_element_residual` is R(u) = dU/du; the driver advances
            // M a = F_int + F_ext, so the internal channel carries -R.
            let residual = q1_element_residual(&material, &mesh, elem, &displacement);
            for (slot, node) in mesh.conn[elem].iter().enumerate() {
                forces[3 * node] -= residual[3 * slot];
                forces[3 * node + 1] -= residual[3 * slot + 1];
                forces[3 * node + 2] -= residual[3 * slot + 2];
            }
        }
        forces
    };
    let external_force = |positions: &[f64], velocities: &[f64], time: f64| {
        // Rising anvil: plane point tracks z = V*t; friction is zero so
        // the wall velocity only enters the (disabled) friction term.
        let mut wall = RigidWall::new([0.0, 0.0, VELOCITY * time], [0.0, 0.0, 1.0], penalty);
        wall.velocity = [0.0, 0.0, VELOCITY];
        let mut forces = vec![0.0; positions.len()];
        let pairs = rigid_wall_contact(&wall, &slaves, positions, velocities, &mut forces);
        let on_wall: f64 = -pairs.iter().map(|pair| pair.force[2]).sum::<f64>();
        history.borrow_mut().push((time, on_wall));
        forces
    };

    let mut steps: u64 = 0;
    while state.time < TSTOP {
        if steps >= MAX_STEPS {
            return Err(format!(
                "stalled: {MAX_STEPS} steps reached at t = {:.6e} s, short of TSTOP {TSTOP:.6e}",
                state.time
            ));
        }
        if let Err(error) = driver.step(
            &mut state,
            internal_force,
            external_force,
            &char_lengths,
            &sound_speeds,
        ) {
            return Err(format!(
                "unstable at t = {:.6e} s after {steps} steps: {error}",
                state.time
            ));
        }
        steps += 1;
    }
    println!(
        "run: reached t = {:.6e} s in {steps} steps, dt_last = {:.6e} s",
        state.time,
        driver.dt()
    );

    let history = history.borrow();
    let mut plateau_sum = 0.0;
    let mut plateau_count = 0usize;
    for (time, force) in history.iter() {
        if *time >= PLATEAU_START {
            plateau_sum += *force;
            plateau_count += 1;
        }
    }
    if plateau_count == 0 {
        return Err("no wall-force samples in the plateau window".to_string());
    }
    let plateau = plateau_sum / plateau_count as f64;
    let face_dz = state.positions[3 * face_center + 2] - reference[3 * face_center + 2];
    let balance = driver.balance().clone();
    let energy_scale = balance.kinetic + balance.internal.abs();
    let drift_frac = if energy_scale > 0.0 {
        balance.drift.abs() / energy_scale
    } else {
        f64::INFINITY
    };
    println!(
        "energy: KE {:.6e} J, IE {:.6e} J, Wext {:.6e} J, drift {:.6e} J ({:.3e} rel)",
        balance.kinetic, balance.internal, balance.external_work, balance.drift, drift_frac
    );

    let mut failed = false;
    println!(
        "probe {} [{}]: actual {plateau:.6e}, expected {:.6e}",
        expectations[0].label, expectations[0].channel, expectations[0].expected
    );
    if let Err(error) = assert_probe_close(
        plateau,
        expectations[0].expected,
        expectations[0].abs_tol,
        expectations[0].rel_tol,
        &expectations[0].label,
    ) {
        println!("FAIL: {error}");
        failed = true;
    }
    println!(
        "probe {} [{}]: actual {face_dz:.6e}, expected {:.6e}",
        expectations[1].label, expectations[1].channel, expectations[1].expected
    );
    if let Err(error) = assert_probe_close(
        face_dz,
        expectations[1].expected,
        expectations[1].abs_tol,
        expectations[1].rel_tol,
        &expectations[1].label,
    ) {
        println!("FAIL: {error}");
        failed = true;
    }
    if !drift_frac.is_finite() || drift_frac > DRIFT_TOL {
        println!("FAIL: energy drift {drift_frac:.3e} exceeds {DRIFT_TOL:.3e}");
        failed = true;
    }
    if failed {
        return Err("FAIL: qa-fem-explicit-taylor-elastic".to_string());
    }
    println!("PASS: qa-fem-explicit-taylor-elastic");
    Ok(())
}
