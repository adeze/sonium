//! Phase 7 lane 7A leg: Taylor-bar impact with contact, plasticity, erosion.
//!
//! Native explicit Johnson-Cook Taylor-bar impact (steel bar at rest, rigid
//! wall rising at 227 m/s; 12x9 cylinder mesh; stop 4 us) on the production
//! H8 plastic path ([`hex_plastic_forces`](sonium_fem::physics::solid::nonlinear_solid::hex_plastic_forces))
//! with penalty wall contact and [`ErosionSet`](sonium_fem::physics::solid::fracture::ductile::ErosionSet)
//! erosion, compared against the vendor engine run of the same deck
//! (`decks/taylor_plastic`, `/MAT/LAW2` with matched Johnson-Cook inputs).
//!
//! Native checks (always run): full-stop stability, bounded energy drift,
//! impact-face yielding, and at least one eroded element. Engine checks
//! (run when `OPENCOURANT_EXE`, or legacy `RADISS_EXE`, is set): deck
//! probes plus native-vs-engine agreement on the wall-force plateau and
//! the impact-face travel. Without an engine the leg reports the native
//! verdict and scopes the skip explicitly. Exit 0 = leg PASS, exit 1 =
//! FAIL with a per-probe report on stdout.

// Rust guideline compliant 2026-02-21

use sonium_compute::explicit_dt::critical_dt;
use sonium_compute::explicit_state::ExplicitState;
use sonium_fem::physics::solid::contact::explicit::{
    RigidWall, rigid_wall_contact, suggested_penalty,
};
use sonium_fem::physics::solid::fracture::ductile::{
    ErosionCriterion, ErosionSet, JohnsonCookDamage,
};
use sonium_fem::physics::solid::nonlinear_solid::{
    HexExplicitGeom, HexExplicitParams, HexExplicitState, HexMesh, hex_jacobian,
    hex_plastic_forces, hex_quadrature, hex_shape_derivatives,
};
use sonium_fem::physics::solid::plasticity::law_metals::{
    Law2, Law2Flow, Law2State, MaxRateEffect,
};
use sonium_qa::rad::{
    OpenCourantCase, RadElasticMaterial, RadExtendedCase, RadFunction, RadJohnsonCook, RadMaterial,
    RadMesh, RadNodeSet, RadPlasticMaterial, RadProbe, RadThRequest, RadWall, assert_probe_close,
    assert_probes_payload_eq, bric_channel, engine_exe, extended_probes_json, run_extended_case,
    write_extended_engine, write_extended_starter,
};
use sonium_solvers::time_integration::{AdaptiveDtConfig, ExplicitStepDriver};
use std::cell::RefCell;
use std::path::PathBuf;

const NAME: &str = "taylor_plastic";
const LENGTH: f64 = 0.0254;
const RADIUS: f64 = 0.003175;
const VELOCITY: f64 = 227.0;
const NZ: usize = 12;
const N: usize = 9;
/// Run stop matching the elastic Taylor leg.
const TSTOP: f64 = 4.0e-6;
/// Wall force plateau average window `[PLATEAU_START, TSTOP]`.
const PLATEAU_START: f64 = 2.0e-6;
/// Penalty factor for [`suggested_penalty`].
///
/// A hundred times below the documented standard 50. At 50 the slam
/// impulse rings the bar bounce mode hard enough to separate the whole
/// face for ~0.5 us mid-run (diagnosed from the engagement history),
/// dragging the plateau mean down; the engine's own face lag (~2e-5 m)
/// shows its wall runs at about this softer compliance scale. Steady
/// penetration stays near 1e-5 m (under 2% of the face travel).
const PENALTY_FACTOR: f64 = 0.5;
/// Contact damping ratio (fraction of critical).
///
/// The production penalty pass is undamped, so the impact transient
/// rings the bar bounce mode hard enough to separate the whole face
/// mid-run. A dashpot in parallel with the penalty spring on each
/// penetrating slave, `c = 2 zeta sqrt(k m)`, counters it. Nodal
/// critical damping barely touches the coherent bar mode (the face to
/// bar mass ratio dilutes it to zeta ~ 0.2 there), so the ratio runs
/// overdamped at 3: the bar mode then sees zeta ~ 0.5 and settles
/// within the run. The dashpot is a numerical stabilizer, so the
/// plateau probe reads the penalty spring force alone while the
/// dashpot mean prints beside it as a stabilization footprint (about
/// 10% here); explicit stability holds (`c dt / m = 1.2 < 2` at
/// omega dt = 0.2).
const CONTACT_DAMPING_RATIO: f64 = 3.0;
/// Gate fails when `|drift| / (KE + |IE|)` exceeds this fraction.
const DRIFT_TOL: f64 = 0.03;
/// Step cap so a stalled run fails instead of hanging.
const MAX_STEPS: u64 = 200_000;
/// Johnson-Cook 4340 flow (Law2::c4340_steel a/b/n/c) on steel elasticity.
const YOUNGS: f64 = 200.0e9;
const POISSON: f64 = 0.3;
const DENSITY: f64 = 7830.0;
const JC_A: f64 = 792.0e6;
const JC_B: f64 = 510.0e6;
const JC_N: f64 = 0.26;
const JC_C: f64 = 0.014;
const JC_REF_RATE: f64 = 1.0;
const JC_MAX_STRESS: f64 = 1.0e30;
const JC_T0: f64 = 298.0;
/// Erosion plastic strain, shared by the native sweep and `/MAT/LAW2` EPS_max.
///
/// Calibrated from the pilot runs: the crushed face core peaks near
/// ep 0.20 on both sides, so 0.18 trips failure there late in the run
/// (native deletion via the sweep, engine `/MAT/LAW2` rupture) while
/// the rest of the bar survives. Both sides then report the failure
/// consistently: native max ep freezes at the threshold, engine PLAS
/// keeps accumulating past it.
const FAIL_EP: f64 = 0.18;
/// Native run must yield the impact face past this plastic strain.
const MIN_FACE_EP: f64 = 0.05;
/// Twelve edges of the H8 in HEX_NODES slot order.
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
/// Engine stamp recorded in `probes.json`.
const ENGINE_STAMP: &str = "OpenCourant linux64_gf 2026-10-12 CommitID 33e68517";
/// Engine wall-force plateau expectation, harvested 2026-10-12.
///
/// Harvested regression bound: the engine run is deterministic, so 10%
/// is a pure tripwire with wide platform-noise margin (not a physics
/// gate; the physics comparison is the native cross-check below).
const PROBE_PLATEAU: f64 = -3.816457e4;
const PROBE_PLATEAU_ABS: f64 = 1.0e3;
const PROBE_PLATEAU_REL: f64 = 0.10;
/// Engine impact-face travel expectation, harvested 2026-10-12.
///
/// Harvested regression bound at 2%: the DZ channel is smooth and the
/// run deterministic, so this trips only on real deck/engine drift.
const PROBE_DZ: f64 = 6.594232e-4;
const PROBE_DZ_ABS: f64 = 1.0e-9;
const PROBE_DZ_REL: f64 = 0.02;
/// Engine impact-brick PLAS expectation, harvested 2026-10-12.
///
/// Harvested regression bound at 10% (same determinism argument as the
/// plateau; the plasticity cross-check is native max ep vs this value).
/// The brick ruptures past `EPS_max` late in the run while PLAS keeps
/// accumulating, so this value sits above the failure strain.
const PROBE_PLAS: f64 = 2.114712e-1;
const PROBE_PLAS_ABS: f64 = 0.02;
const PROBE_PLAS_REL: f64 = 0.10;
/// Native-vs-engine plateau agreement.
///
/// Measured 6% on the 12x9 mesh with the penalty-only probe; 15% holds
/// a 2.5x margin for platform noise in the contact transient.
const NATIVE_PLATEAU_REL: f64 = 0.15;
/// Native-vs-engine travel agreement.
///
/// Measured under 1% with persistent contact; 6% holds wide margin
/// while still catching contact loss (separation overshoots 10%+).
const NATIVE_DZ_REL: f64 = 0.06;
/// Native-vs-engine face crush agreement (loose order check).
///
/// Compares native max equivalent plastic strain against the engine
/// impact-brick PLAS. Measured 16% apart with both sides tripping
/// failure; 50% checks the order, not the digit, since native max ep
/// freezes at the erosion threshold while engine PLAS accumulates on.
const NATIVE_PLAS_REL: f64 = 0.50;

/// Matched Johnson-Cook steel for the native run and the deck writer.
///
/// Isothermal (`m = 0`, no heat capacity): the deck runs without `/HEAT`,
/// so the native leg must not heat or soften either. `MaxRateEffect::Scaled`
/// matches the deck `ICC = 1`; rupture at `FAIL_EP` matches `EPS_max`.
fn jc_steel() -> Law2 {
    let mut steel = Law2::new(
        YOUNGS,
        POISSON,
        Law2Flow::JohnsonCook {
            a: JC_A,
            b: JC_B,
            n: JC_N,
            c: JC_C,
        },
        JC_REF_RATE,
        0.0,
        0.0,
        JC_T0,
        0.0,
        JC_MAX_STRESS,
        MaxRateEffect::Scaled,
        FAIL_EP,
        0.0,
        1.0e30,
    );
    // Tracked for the erosion sweep; Law2 never couples damage back into
    // the stress, so the mechanics still match the deck exactly.
    steel.damage = Some(JohnsonCookDamage::c4340_steel());
    steel
}

/// Ported cylinder mesh: node coordinates, H8 connectivity, impact-face
/// node set, face-center node, and impact-brick element.
type CylinderMesh = (Vec<[f64; 3]>, Vec<[usize; 8]>, Vec<usize>, usize, usize);

/// Bricks a cylinder: exact 0-based port of the Taylor deck `cylinder_mesh`.
///
/// Clips an `n` by `n` Cartesian grid per layer to the circular section
/// and projects region-boundary nodes radially; the impact face floats
/// one approach gap above the wall. Returns the node coordinates, the
/// H8 connectivity, the impact-face node set, the face-center node, and
/// the layer-0 brick holding the face center.
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
    let layer0_count = bricks.len() / nz;
    let impact_brick = bricks[..layer0_count]
        .iter()
        .position(|brick| brick.contains(&face_center))
        .unwrap_or(0);
    (nodes, bricks, face_set, face_center, impact_brick)
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

/// Minimum edge length of one element at the given positions.
fn min_edge_length_at(positions: &[f64], conn: &[usize; 8]) -> f64 {
    let point = |node: usize| {
        [
            positions[3 * node],
            positions[3 * node + 1],
            positions[3 * node + 2],
        ]
    };
    let mut shortest = f64::INFINITY;
    for (a, b) in HEX_EDGES {
        let pa = point(conn[a]);
        let pb = point(conn[b]);
        let dx = pa[0] - pb[0];
        let dy = pa[1] - pb[1];
        let dz = pa[2] - pb[2];
        shortest = shortest.min((dx * dx + dy * dy + dz * dz).sqrt());
    }
    shortest
}

/// Outcome of the native explicit run.
struct NativeOutcome {
    plateau: f64,
    face_dz: f64,
    max_ep: f64,
    max_damage: f64,
    eroded: usize,
    drift_frac: f64,
    steps: u64,
    dt_last: f64,
}

/// Runs the native Johnson-Cook Taylor impact to `TSTOP`.
///
/// Updated-Lagrangian H8: each force evaluation rebuilds the element
/// geometry from the current positions, commits the centroid Law2 point
/// through [`hex_plastic_forces`], and sweeps the [`ErosionSet`]. Inverted
/// or failed elements erode on the spot (standard explicit practice); the
/// lumped masses stay fixed so erosion never destabilizes the step.
fn run_native() -> Result<NativeOutcome, String> {
    let (coords, conn, _face_set, face_center, impact_brick) = cylinder_mesh(NZ, N);
    let mesh = HexMesh::new(coords.clone(), conn);
    let num_nodes = mesh.num_nodes();
    let num_elems = mesh.conn.len();
    println!(
        "mesh: {num_nodes} nodes, {num_elems} bricks, face center node {face_center}, impact brick {impact_brick}"
    );

    // Lumped masses from reference element volumes; every ported node
    // belongs to at least one brick, so every mass stays positive. Masses
    // never change afterward: eroded elements lose stiffness, not inertia.
    let mut masses = vec![0.0; num_nodes];
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
    }
    let min_mass = masses.iter().cloned().fold(f64::INFINITY, f64::min);
    if !min_mass.is_finite() || min_mass <= 0.0 {
        return Err(format!("non-positive lumped mass {min_mass:.6e}"));
    }

    // Dilatational (max) wavespeed for the 3D CFL bound.
    let lame_lambda = YOUNGS * POISSON / ((1.0 + POISSON) * (1.0 - 2.0 * POISSON));
    let shear_mu = YOUNGS / (2.0 * (1.0 + POISSON));
    let dilatational = ((lame_lambda + 2.0 * shear_mu) / DENSITY).sqrt();
    let sound_speeds = vec![dilatational; num_elems];

    // Penalty contact is an undamped oscillator on the lumped mass.
    // Stability alone (`dt <= 2*sqrt(m/k)`) still bounces: one step of
    // penetration at the impact speed reverses the nodal velocity when
    // `(omega dt)^2 ~ 1`, so the face chatters with separation (pilot:
    // force exactly zero at most steps, rare 5e5 N spikes). The cap
    // below resolves the contact at omega dt = 0.2 (dv/V ~ 4% per step),
    // which holds persistent contact; the pilot mean then matches the
    // engine plateau.
    let flat: Vec<f64> = coords.iter().flat_map(|point| *point).collect();
    let min_char = mesh
        .conn
        .iter()
        .map(|conn| min_edge_length_at(&flat, conn))
        .fold(f64::INFINITY, f64::min);
    let penalty = suggested_penalty(YOUNGS, min_char, PENALTY_FACTOR);
    let dt_contact = 2.0 * (min_mass / penalty).sqrt();
    let dt_max = 0.1 * dt_contact;
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
    // Bar starts at rest: zero initial velocities, no bulk damping anywhere
    // (contact damping below is the only dashpot, and only on penetration).
    let nodal_masses = masses.clone();
    let reference: Vec<f64> = state.positions.clone();
    // Slave set covers the whole bar like the deck's `bar` group, since
    // mushrooming can spread contact past the initial face.
    let slaves: Vec<usize> = (0..num_nodes).collect();
    // Wall-force histories sampled by the force closure: the penalty
    // spring force (the physics probe) and the dashpot force (the
    // stabilization footprint, printed beside the probe).
    let history: RefCell<Vec<(f64, f64)>> = RefCell::new(Vec::new());
    let damp_history: RefCell<Vec<(f64, f64)>> = RefCell::new(Vec::new());

    let material = jc_steel();
    let params = HexExplicitParams::new(DENSITY, dilatational);
    let states: RefCell<Vec<HexExplicitState<Law2State>>> = RefCell::new(
        (0..num_elems)
            .map(|_| HexExplicitState::new(Law2State::new(JC_T0)))
            .collect(),
    );
    let erosion = RefCell::new(ErosionSet::new(num_elems));
    let criterion = ErosionCriterion {
        max_equiv_plastic_strain: FAIL_EP,
        max_damage: 1.0,
        max_porosity: f64::INFINITY,
        max_tabular_damage: f64::INFINITY,
    };
    // Current step size for the material update. Fixed-stepping (instead of
    // the driver's adaptive step) threads `dt` into the force closure so
    // the flow stress and damage see the true step rate; the loop below
    // replays the same CFL selection the adaptive driver would apply.
    let step_dt = RefCell::new(dt_max);
    let max_ep = RefCell::new(0.0f64);
    let max_damage = RefCell::new(0.0f64);

    let internal_force = |positions: &[f64], velocities: &[f64], _: f64| {
        let dt = *step_dt.borrow();
        let mut forces = vec![0.0; positions.len()];
        let mut states = states.borrow_mut();
        let mut erosion = erosion.borrow_mut();
        let mut features = Vec::with_capacity(num_elems);
        for (elem, conn) in mesh.conn.iter().enumerate() {
            if !erosion.is_active(elem) {
                // Eroded elements carry no load; the sweep below skips
                // them, so these features are placeholders.
                features.push((0.0, 0.0, 0.0));
                continue;
            }
            let mut elem_coords = [[0.0; 3]; 8];
            for (slot, node) in conn.iter().enumerate() {
                elem_coords[slot] = [
                    positions[3 * node],
                    positions[3 * node + 1],
                    positions[3 * node + 2],
                ];
            }
            let geom = match HexExplicitGeom::new(elem_coords) {
                Ok(geom) => geom,
                Err(_) => {
                    // Inverted element: erode through the sweep below.
                    features.push((f64::INFINITY, 0.0, 0.0));
                    continue;
                }
            };
            let mut vel = [0.0; 24];
            for (slot, node) in conn.iter().enumerate() {
                vel[slot * 3] = velocities[3 * node];
                vel[slot * 3 + 1] = velocities[3 * node + 1];
                vel[slot * 3 + 2] = velocities[3 * node + 2];
            }
            match hex_plastic_forces(&geom, &mut states[elem], &material, &vel, dt, &params) {
                Ok(out) => {
                    for (slot, node) in conn.iter().enumerate() {
                        forces[3 * node] -= out.base.forces[slot * 3];
                        forces[3 * node + 1] -= out.base.forces[slot * 3 + 1];
                        forces[3 * node + 2] -= out.base.forces[slot * 3 + 2];
                    }
                    let mut peak = max_ep.borrow_mut();
                    *peak = peak.max(out.features.0);
                    let mut peak_damage = max_damage.borrow_mut();
                    *peak_damage = peak_damage.max(out.features.1);
                    features.push(out.features);
                }
                Err(_) => {
                    features.push((f64::INFINITY, 0.0, 0.0));
                }
            }
        }
        erosion.sweep(&criterion, &features);
        forces
    };
    let external_force = |positions: &[f64], velocities: &[f64], time: f64| {
        // Rising anvil: plane point tracks z = V*t; friction is zero so
        // the wall velocity only enters the (disabled) friction term.
        let mut wall = RigidWall::new([0.0, 0.0, VELOCITY * time], [0.0, 0.0, 1.0], penalty);
        wall.velocity = [0.0, 0.0, VELOCITY];
        let mut forces = vec![0.0; positions.len()];
        let pairs = rigid_wall_contact(&wall, &slaves, positions, velocities, &mut forces);
        let penalty_wall: f64 = -pairs.iter().map(|pair| pair.force[2]).sum::<f64>();
        // Overdamped dashpot on each penetrating slave (see
        // CONTACT_DAMPING_RATIO): settles the bar bounce mode. Its wall
        // reaction joins the dynamics but stays out of the probe, which
        // reads the penalty spring force alone.
        let mut damp_wall = 0.0;
        for pair in &pairs {
            let node = pair.slave;
            let c = 2.0 * CONTACT_DAMPING_RATIO * (penalty * nodal_masses[node]).sqrt();
            let relative = velocities[3 * node + 2] - VELOCITY;
            let damp = -c * relative;
            forces[3 * node + 2] += damp;
            damp_wall -= damp;
        }
        damp_history.borrow_mut().push((time, damp_wall));
        history.borrow_mut().push((time, penalty_wall));
        forces
    };

    let mut steps: u64 = 0;
    let mut dt_last = dt_max;
    while state.time < TSTOP {
        if steps >= MAX_STEPS {
            return Err(format!(
                "stalled: {MAX_STEPS} steps reached at t = {:.6e} s, short of TSTOP {TSTOP:.6e}",
                state.time
            ));
        }
        // Same CFL selection as the adaptive driver: safety 0.9 on the
        // bulk critical step, contact cap, 1.1 growth limit, exact landing.
        let char_lengths: Vec<f64> = mesh
            .conn
            .iter()
            .map(|conn| min_edge_length_at(&state.positions, conn))
            .collect();
        let dt_crit = critical_dt(&char_lengths, &sound_speeds);
        let mut dt = (0.9 * dt_crit).min(dt_max).min(1.1 * dt_last);
        dt = dt.min(TSTOP - state.time).max(1.0e-12);
        if !dt.is_finite() || dt <= 0.0 {
            return Err(format!(
                "non-positive step {dt:.6e} at t = {:.6e} s",
                state.time
            ));
        }
        *step_dt.borrow_mut() = dt;
        if let Err(error) = driver.step_fixed(&mut state, internal_force, external_force, dt) {
            return Err(format!(
                "unstable at t = {:.6e} s after {steps} steps: {error}",
                state.time
            ));
        }
        dt_last = dt;
        steps += 1;
    }
    println!(
        "run: reached t = {:.6e} s in {steps} steps, dt_last = {dt_last:.6e} s",
        state.time
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
    // Dashpot footprint over the same window (stabilization transparency;
    // the probe above reads the penalty spring alone).
    let damp_mean = {
        let damp_history = damp_history.borrow();
        let (mut sum, mut count) = (0.0, 0usize);
        for (time, force) in damp_history.iter() {
            if *time >= PLATEAU_START {
                sum += *force;
                count += 1;
            }
        }
        sum / count as f64
    };
    println!("contact: dashpot mean {damp_mean:.6e} N over the plateau window");
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
    let eroded = num_elems - erosion.borrow().active_count();
    println!(
        "plastic: max ep {:.6e}, max damage {:.6e}, eroded {eroded}/{num_elems}",
        *max_ep.borrow(),
        *max_damage.borrow()
    );
    Ok(NativeOutcome {
        plateau,
        face_dz,
        max_ep: *max_ep.borrow(),
        max_damage: *max_damage.borrow(),
        eroded,
        drift_frac,
        steps,
        dt_last,
    })
}

/// Builds the Taylor-plastic engine case on the real `sonium_qa::rad` API.
///
/// Same frame and mesh as the native leg (bar at rest, rigid wall rising
/// at the impact speed; 1-based port of [`cylinder_mesh`]) with the matched
/// `/MAT/LAW2` Johnson-Cook steel and `EPS_max = FAIL_EP` erosion.
fn taylor_plastic_case() -> RadExtendedCase {
    let (nodes, bricks, face_set, face_center, impact_brick) = cylinder_mesh(NZ, N);
    // 1-based deck numbering over the identical port.
    let deck_nodes = nodes;
    let deck_bricks: Vec<[usize; 8]> = bricks
        .iter()
        .map(|brick| brick.map(|node| node + 1))
        .collect();
    let mut mesh = RadMesh {
        nodes: deck_nodes,
        bricks: deck_bricks,
    };
    let deck_face_center = face_center + 1;
    let deck_impact_brick = impact_brick + 1;
    let stop = TSTOP;
    // Secondary group covers the whole bar (mushrooming spreads contact past
    // the initial face); the driver node is appended after so it stays out
    // of the contact group.
    let bar_nodes: Vec<usize> = (1..=mesh.num_nodes()).collect();
    let anvil_point = [0.0, 0.0, 0.0];
    let driver = mesh.add_node(anvil_point);
    let base = OpenCourantCase {
        name: NAME.to_string(),
        title: "sonium_qa rad taylor bar johnson-cook plastic impact".to_string(),
        mesh,
        material: RadElasticMaterial {
            youngs_modulus_pa: YOUNGS,
            poisson_ratio: POISSON,
            density_kg_per_m3: DENSITY,
        },
        node_sets: vec![
            RadNodeSet {
                name: "impact face".to_string(),
                nodes: face_set.iter().map(|node| node + 1).collect(),
            },
            RadNodeSet {
                name: "bar".to_string(),
                nodes: bar_nodes,
            },
        ],
        bcs: Vec::new(),
        functions: vec![RadFunction {
            name: "constant wall velocity".to_string(),
            points: vec![(0.0, 1.0), (stop, 1.0)],
        }],
        cloads: Vec::new(),
        walls: vec![RadWall {
            name: "rising anvil".to_string(),
            normal: [0.0, 0.0, 1.0],
            point_m: anvil_point,
            friction: 0.0,
            velocity_m_per_s: Some([0.0, 0.0, VELOCITY]),
            driver_node: Some(driver),
            secondary_set: Some(1),
            search_distance_m: 0.0,
        }],
        th_nodes: vec![deck_face_center],
        probes: vec![
            RadProbe {
                label: "anvil FNZ plateau".to_string(),
                channel: "FNZ/RWALL/1".to_string(),
                expected: PROBE_PLATEAU,
                abs_tol: PROBE_PLATEAU_ABS,
                rel_tol: PROBE_PLATEAU_REL,
            },
            RadProbe {
                label: "impact face DZ".to_string(),
                channel: format!("DZ/{deck_face_center}"),
                expected: PROBE_DZ,
                abs_tol: PROBE_DZ_ABS,
                rel_tol: PROBE_DZ_REL,
            },
            RadProbe {
                label: "impact brick PLAS".to_string(),
                channel: bric_channel("PLAS", deck_impact_brick),
                expected: PROBE_PLAS,
                abs_tol: PROBE_PLAS_ABS,
                rel_tol: PROBE_PLAS_REL,
            },
        ],
        stop_time_s: stop,
        th_period_s: 1.0e-7,
        anim_period_s: 1.0e-6,
    };
    let mut extended = RadExtendedCase::wrap_elastic(base);
    extended.material = RadMaterial::Plastic(RadPlasticMaterial::JohnsonCook(RadJohnsonCook {
        youngs_modulus_pa: YOUNGS,
        poisson_ratio: POISSON,
        density_kg_per_m3: DENSITY,
        yield_pa: JC_A,
        hardening_pa: JC_B,
        hardening_exp: JC_N,
        rate_coeff: JC_C,
        ref_strain_rate: JC_REF_RATE,
        fail_plastic_strain: FAIL_EP,
        max_stress_pa: JC_MAX_STRESS,
    }));
    extended.th_brick = RadThRequest::brick_stress(vec![deck_impact_brick]);
    extended
}

fn decks_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("decks/taylor_plastic")
}

fn main() {
    if let Err(error) = run_leg() {
        eprintln!("{error}");
        std::process::exit(1);
    }
}

fn run_leg() -> Result<(), String> {
    // Native leg first: contact + plasticity + erosion with no engine.
    let native = run_native()?;
    let mut failed = false;
    println!(
        "native plateau: {:.6e} N (engine comparison below)",
        native.plateau
    );
    println!("native face DZ: {:.6e} m", native.face_dz);
    println!(
        "native damage {:.6e}, eroded {}, steps {}",
        native.max_damage, native.eroded, native.steps
    );
    if !native.drift_frac.is_finite() || native.drift_frac > DRIFT_TOL {
        println!(
            "FAIL: native energy drift {:.3e} exceeds {DRIFT_TOL:.3e}",
            native.drift_frac
        );
        failed = true;
    }
    if native.max_ep < MIN_FACE_EP {
        println!(
            "FAIL: impact face never yielded (max ep {:.3e} < {MIN_FACE_EP:.3e})",
            native.max_ep
        );
        failed = true;
    }
    if native.eroded == 0 {
        println!("FAIL: no element eroded (erosion path unexercised)");
        failed = true;
    }
    if native.dt_last <= 0.0 || !native.dt_last.is_finite() {
        println!("FAIL: unstable final step {:.6e}", native.dt_last);
        failed = true;
    }

    // Deck leg: the checked-in trio must match the rad writer.
    let case = taylor_plastic_case();
    case.validate()
        .map_err(|error| format!("taylor_plastic case invalid: {error}"))?;
    let dir = decks_dir();
    let starter_path = dir.join("taylor_plastic_0000.rad");
    let engine_path = dir.join("taylor_plastic_0001.rad");
    let probes_path = dir.join("probes.json");
    let starter = write_extended_starter(&case);
    let engine = write_extended_engine(&case);
    if std::env::var("REGEN_DECKS").as_deref() == Ok("1") {
        std::fs::create_dir_all(&dir).map_err(|error| format!("mkdir decks: {error}"))?;
        std::fs::write(&starter_path, &starter)
            .map_err(|error| format!("write starter: {error}"))?;
        std::fs::write(&engine_path, &engine).map_err(|error| format!("write engine: {error}"))?;
        std::fs::write(&probes_path, extended_probes_json(&case, ENGINE_STAMP))
            .map_err(|error| format!("write probes: {error}"))?;
        println!("regenerated decks for taylor_plastic");
        return Ok(());
    }
    let checked_starter = std::fs::read_to_string(&starter_path).unwrap_or_default();
    let checked_engine = std::fs::read_to_string(&engine_path).unwrap_or_default();
    let checked_probes = std::fs::read_to_string(&probes_path).unwrap_or_default();
    assert_eq!(
        checked_starter, starter,
        "taylor_plastic_0000.rad drifted from the rad writer"
    );
    assert_eq!(
        checked_engine, engine,
        "taylor_plastic_0001.rad drifted from the rad writer"
    );
    assert_probes_payload_eq(
        &checked_probes,
        &extended_probes_json(&case, ENGINE_STAMP),
        "probes.json drifted from the rad writer",
    );
    // Native and deck meshes share one port: the 1-based deck face center
    // must resolve to the same node the native leg tracked.
    let (_, _, _, face_center, _) = cylinder_mesh(NZ, N);
    let deck_face_center: usize = case.base.probes[1]
        .channel
        .strip_prefix("DZ/")
        .and_then(|id| id.parse::<usize>().ok())
        .ok_or_else(|| "probe 1 channel is not DZ/<node>".to_string())?;
    if face_center + 1 != deck_face_center {
        return Err(format!(
            "mesh port drifted: native face center 0-based {face_center}, deck probes DZ/{deck_face_center}"
        ));
    }

    let Some(exe) = engine_exe() else {
        println!("engine comparison skipped: set OPENCOURANT_EXE (or legacy RADISS_EXE)");
        if failed {
            println!("FAILED: qa-fem-explicit-taylor-plastic");
            return Err("FAIL: qa-fem-explicit-taylor-plastic".to_string());
        }
        println!("PASS: qa-fem-explicit-taylor-plastic (native only; engine comparison skipped)");
        println!(
            "SUCCESS: qa-fem-explicit-taylor-plastic (native only; engine comparison skipped)"
        );
        return Ok(());
    };

    // Run in a scratch dir so engine artifacts never pollute the repo.
    let work =
        std::env::temp_dir().join(format!("sonium_rad_taylor_plastic_{}", std::process::id()));
    std::fs::create_dir_all(&work).map_err(|error| format!("mkdir scratch: {error}"))?;
    std::fs::write(work.join("taylor_plastic_0000.rad"), &starter)
        .map_err(|error| format!("write scratch starter: {error}"))?;
    std::fs::write(work.join("taylor_plastic_0001.rad"), &engine)
        .map_err(|error| format!("write scratch engine: {error}"))?;
    let output = run_extended_case(&work, &exe, &case).map_err(|error| {
        let _ = std::fs::remove_dir_all(&work);
        format!("rad taylor_plastic run failed: {error}")
    })?;
    println!("engine: {}", output.engine);

    // Engine probes: plateau averages over the window, DZ/PLAS read final.
    let plateau = output
        .history
        .channel(&case.base.probes[0].channel)
        .and_then(|values| {
            let times = &output.history.times;
            let mut sum = 0.0;
            let mut count = 0usize;
            for (time, value) in times.iter().zip(values.iter()) {
                if *time >= PLATEAU_START {
                    sum += *value;
                    count += 1;
                }
            }
            if count == 0 {
                return Err(sonium_qa::rad::RadError::Parse(format!(
                    "channel {} has no plateau samples",
                    case.base.probes[0].channel
                )));
            }
            Ok(sum / count as f64)
        });
    let dz = output.history.final_value(&case.base.probes[1].channel);
    let plas = output.history.final_value(&case.base.probes[2].channel);
    std::fs::remove_dir_all(&work).ok();
    let (engine_plateau, engine_dz, engine_plas) = match (plateau, dz, plas) {
        (Ok(plateau), Ok(dz), Ok(plas)) => (plateau, dz, plas),
        (Err(error), _, _) | (_, Err(error), _) | (_, _, Err(error)) => {
            println!("FAIL: engine probe unreadable: {error}");
            println!("FAILED: qa-fem-explicit-taylor-plastic");
            return Err("FAIL: qa-fem-explicit-taylor-plastic".to_string());
        }
    };
    for (probe, actual) in [
        (&case.base.probes[0], engine_plateau),
        (&case.base.probes[1], engine_dz),
        (&case.base.probes[2], engine_plas),
    ] {
        println!(
            "probe {} [{}]: engine actual {actual:.6e}, expected {:.6e}",
            probe.label, probe.channel, probe.expected
        );
        if let Err(error) = assert_probe_close(
            actual,
            probe.expected,
            probe.abs_tol,
            probe.rel_tol,
            &probe.label,
        ) {
            println!("FAIL: {error}");
            failed = true;
        }
    }
    if engine_plas < MIN_FACE_EP {
        println!("FAIL: engine impact brick never yielded (PLAS {engine_plas:.3e})");
        failed = true;
    }

    // Native-vs-engine cross-check on the shared probes.
    println!(
        "cross-check plateau: native {:.6e} N vs engine {engine_plateau:.6e} N",
        native.plateau
    );
    if let Err(error) = assert_probe_close(
        native.plateau,
        engine_plateau,
        PROBE_PLATEAU_ABS,
        NATIVE_PLATEAU_REL,
        "native-vs-engine plateau",
    ) {
        println!("FAIL: {error}");
        failed = true;
    }
    println!(
        "cross-check face DZ: native {:.6e} m vs engine {engine_dz:.6e} m",
        native.face_dz
    );
    if let Err(error) = assert_probe_close(
        native.face_dz,
        engine_dz,
        PROBE_DZ_ABS,
        NATIVE_DZ_REL,
        "native-vs-engine face DZ",
    ) {
        println!("FAIL: {error}");
        failed = true;
    }
    println!(
        "cross-check face crush: native max ep {:.6e} vs engine PLAS {engine_plas:.6e}",
        native.max_ep
    );
    if let Err(error) = assert_probe_close(
        native.max_ep,
        engine_plas,
        PROBE_PLAS_ABS,
        NATIVE_PLAS_REL,
        "native-vs-engine face crush",
    ) {
        println!("FAIL: {error}");
        failed = true;
    }

    if failed {
        println!("FAILED: qa-fem-explicit-taylor-plastic");
        return Err("FAIL: qa-fem-explicit-taylor-plastic".to_string());
    }
    println!("PASS: qa-fem-explicit-taylor-plastic");
    println!("SUCCESS: qa-fem-explicit-taylor-plastic");
    Ok(())
}
