//! Lane 7A leg: Johnson-Cook H8 plasticity with damage and erosion.
//!
//! Engine-free proof of the explicit plastic path behind the
//! `hex_explicit_forces<M>` material generic: a single Johnson-Cook
//! element lands on the flow surface, the step rate hardens the response,
//! damage accumulates exactly per the Johnson-Cook rule, erosion features
//! trip an `ErosionSet` sweep, and a miniature Taylor impact with contact
//! runs stably with bounded energy drift while yielding and eroding.

// Rust guideline compliant 2026-02-21

use sonium_compute::explicit_dt::critical_dt;
use sonium_compute::explicit_state::ExplicitState;
use sonium_fem::physics::solid::contact::explicit::{
    RigidWall, rigid_wall_contact, suggested_penalty,
};
use sonium_fem::physics::solid::fracture::ductile::{
    ErosionCriterion, ErosionSet, JohnsonCookDamage, stress_triaxiality_c1,
};
use sonium_fem::physics::solid::nonlinear_solid::{
    ExplicitPlasticMaterial, HexExplicitGeom, HexExplicitParams, HexExplicitState, HexMesh,
    hex_plastic_eroded, hex_plastic_forces,
};
use sonium_fem::physics::solid::plasticity::law_metals::{
    Law2, Law2Flow, Law2State, MaxRateEffect,
};
use sonium_fem::physics::solid::plasticity::update::von_mises_c1;
use sonium_solvers::time_integration::{AdaptiveDtConfig, ExplicitStepDriver};
use std::cell::RefCell;

/// Isothermal rate-free Johnson-Cook steel for closed-form checks.
fn closed_form_jc() -> Law2 {
    Law2::new(
        200.0e9,
        0.3,
        Law2Flow::JohnsonCook {
            a: 400.0e6,
            b: 300.0e6,
            n: 0.5,
            c: 0.0,
        },
        1.0,
        0.0,
        0.0,
        298.0,
        0.0,
        1.0e30,
        MaxRateEffect::Scaled,
        1.0e30,
        0.0,
        1.0e30,
    )
}

/// Uniaxial-strain velocity field at rate `rate` (1/s) on the unit cube.
fn uniaxial_velocities(rate: f64) -> [f64; 24] {
    let mesh = HexMesh::unit_cube(1.0);
    let mut velocities = [0.0; 24];
    for node in 0..8 {
        velocities[node * 3] = rate * mesh.coords[node][0];
    }
    velocities
}

#[test]
fn jc_element_lands_on_flow_surface() {
    let geom = HexExplicitGeom::from_mesh(&HexMesh::unit_cube(1.0), 0).unwrap();
    let material = closed_form_jc();
    let mut state = HexExplicitState::new(Law2State::new(298.0));
    let params = HexExplicitParams::new(7800.0, 4500.0);
    let out = hex_plastic_forces(
        &geom,
        &mut state,
        &material,
        &uniaxial_velocities(1.0),
        1.0e-2,
        &params,
    )
    .unwrap();
    assert!(out.base.yielding);
    let ep = state.centroid.equiv_plastic_strain;
    assert!(ep > 0.0);
    let expected = 400.0e6 + 300.0e6 * ep.powf(0.5);
    let got = von_mises_c1(&out.base.stress);
    assert!(
        (got - expected).abs() / expected < 1e-9,
        "von Mises {got:.6e} vs flow {expected:.6e}"
    );
}

#[test]
fn jc_step_rate_hardens_response() {
    let geom = HexExplicitGeom::from_mesh(&HexMesh::unit_cube(1.0), 0).unwrap();
    let mut material = closed_form_jc();
    material.flow = Law2Flow::JohnsonCook {
        a: 400.0e6,
        b: 300.0e6,
        n: 0.5,
        c: 0.02,
    };
    let params = HexExplicitParams::new(7800.0, 4500.0);
    let mut slow = HexExplicitState::new(Law2State::new(298.0));
    let slow_out = hex_plastic_forces(
        &geom,
        &mut slow,
        &material,
        &uniaxial_velocities(1.0),
        1.0e-2,
        &params,
    )
    .unwrap();
    let mut fast = HexExplicitState::new(Law2State::new(298.0));
    let fast_out = hex_plastic_forces(
        &geom,
        &mut fast,
        &material,
        &uniaxial_velocities(1000.0),
        1.0e-5,
        &params,
    )
    .unwrap();
    assert!(slow_out.base.yielding && fast_out.base.yielding);
    assert!(fast_out.base.strain_rate > 100.0 * slow_out.base.strain_rate);
    let slow_q = von_mises_c1(&slow_out.base.stress);
    let fast_q = von_mises_c1(&fast_out.base.stress);
    assert!(
        fast_q > slow_q,
        "fast {fast_q:.6e} must exceed slow {slow_q:.6e}"
    );
}

#[test]
fn jc_damage_and_erosion_trip_together() {
    let geom = HexExplicitGeom::from_mesh(&HexMesh::unit_cube(1.0), 0).unwrap();
    let mut material = Law2::c4340_steel();
    let damage = JohnsonCookDamage::c4340_steel();
    material.damage = Some(damage);
    let params = HexExplicitParams::new(7800.0, 4500.0);
    // Five 2% uniaxial steps accumulate plastic strain and damage.
    let mut states = [
        HexExplicitState::new(Law2State::new(298.0)),
        HexExplicitState::new(Law2State::new(298.0)),
    ];
    let mut total_damage = 0.0;
    for step in 0..5 {
        let out = hex_plastic_forces(
            &geom,
            &mut states[1],
            &material,
            &uniaxial_velocities(1.0),
            2.0e-2,
            &params,
        )
        .unwrap();
        assert!(out.base.yielding, "step {step} must yield");
        assert!(out.damage_increment >= 0.0);
        total_damage += out.damage_increment;
        // Wrapper damage matches the direct Johnson-Cook rule exactly,
        // including the [0, 1] clamp Law2 applies on saturation.
        let triax = stress_triaxiality_c1(&out.base.stress);
        let raw = damage.increment(
            triax,
            out.base.dep,
            out.base.strain_rate,
            material.homologous_temperature(states[1].centroid.temperature),
        );
        let prev = (total_damage - out.damage_increment).clamp(0.0, 1.0);
        let expected = (prev + raw).clamp(0.0, 1.0) - prev;
        assert!((out.damage_increment - expected).abs() < 1e-15);
    }
    assert!(total_damage > 0.0);
    assert!((states[1].centroid.damage - total_damage.clamp(0.0, 1.0)).abs() < 1e-12);
    // The loaded element trips a strain-or-damage sweep; the virgin one
    // survives and keeps carrying its (zero) load.
    let criterion = ErosionCriterion {
        max_equiv_plastic_strain: 0.05,
        max_damage: 1.0,
        max_porosity: f64::INFINITY,
        max_tabular_damage: f64::INFINITY,
    };
    let mut set = ErosionSet::new(2);
    let features = [
        material.erosion_features(&states[0].centroid),
        material.erosion_features(&states[1].centroid),
    ];
    assert!(features[1].0 > 0.05, "ep {}", features[1].0);
    assert_eq!(set.sweep(&criterion, &features), vec![1]);
    assert!(set.is_active(0));
    assert!(!set.is_active(1));
    let zero = hex_plastic_eroded();
    assert!(zero.base.forces.iter().all(|f| *f == 0.0));
}

/// Miniature Taylor impact: contact + plasticity + erosion stay stable.
///
/// Two-layer 4340 bar against the rising anvil to 4 us with erosion at
/// ep 0.5. Passes when the run completes, the face yields and erodes,
/// and the energy drift stays within 5%.
#[test]
fn mini_taylor_impact_yields_erodes_and_conserves_energy() {
    const LENGTH: f64 = 0.0254;
    const RADIUS: f64 = 0.003175;
    const VELOCITY: f64 = 227.0;
    const TSTOP: f64 = 4.0e-6;
    const FAIL_EP: f64 = 0.1;
    // Coarse clipped-grid cylinder port (4 layers of 4x4 cells).
    let (nz, n) = (4usize, 4usize);
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
            for di in 0..=1 {
                for dj in 0..=1 {
                    let (ci, cj) = (i + 1 - di, j + 1 - dj);
                    if ci == 0 || cj == 0 || ci > n || cj > n {
                        continue;
                    }
                    if kept[(ci - 1) * n + (cj - 1)] {
                        used = true;
                    }
                }
            }
            if !used {
                continue;
            }
            layer_id[i * (n + 1) + j] = layer_nodes.len() + 1;
            layer_nodes.push([corner(i), corner(j)]);
        }
    }
    let per_layer = layer_nodes.len();
    let mut coords = Vec::with_capacity(per_layer * (nz + 1));
    for k in 0..=nz {
        let z = gap + LENGTH * k as f64 / nz as f64;
        for xy in &layer_nodes {
            coords.push([xy[0], xy[1], z]);
        }
    }
    let mut conn = Vec::new();
    for k in 0..nz {
        for i in 0..n {
            for j in 0..n {
                if !kept[i * n + j] {
                    continue;
                }
                let at = |di: usize, dj: usize| layer_id[(i + di) * (n + 1) + (j + dj)] - 1;
                let base = k * per_layer;
                conn.push([
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
    let mesh = HexMesh::new(coords, conn);
    let num_nodes = mesh.num_nodes();
    let num_elems = mesh.conn.len();
    assert!(num_elems >= 4, "mini mesh needs elements");

    let youngs = 200.0e9;
    let poisson = 0.3;
    let density = 7830.0;
    let mut masses = vec![0.0; num_nodes];
    for (elem, brick) in mesh.conn.iter().enumerate() {
        let geom = HexExplicitGeom::from_mesh(&mesh, elem).unwrap();
        for node in brick.iter() {
            masses[*node] += density * geom.volume / 8.0;
        }
    }
    let min_mass = masses.iter().cloned().fold(f64::INFINITY, f64::min);
    assert!(min_mass > 0.0);
    let lame = youngs * poisson / ((1.0 + poisson) * (1.0 - 2.0 * poisson));
    let mu = youngs / (2.0 * (1.0 + poisson));
    let dilatational = ((lame + 2.0 * mu) / density).sqrt();
    let sound_speeds = vec![dilatational; num_elems];
    let min_char = (0..num_elems)
        .map(|elem| {
            HexExplicitGeom::from_mesh(&mesh, elem)
                .map(|geom| geom.char_length)
                .unwrap_or(f64::INFINITY)
        })
        .fold(f64::INFINITY, f64::min);
    let penalty = suggested_penalty(youngs, min_char, 50.0);
    // Resolve the undamped contact oscillator (omega dt = 0.2) so the
    // face holds persistent contact instead of penalty-bouncing.
    let dt_max = 0.2 * (min_mass / penalty).sqrt();

    let config = AdaptiveDtConfig::new(0.9, 1.0e-12, dt_max, 1.1).unwrap();
    let mut driver = ExplicitStepDriver::new(config).unwrap();
    let mut state = ExplicitState::new(num_nodes);
    for (node, xyz) in mesh.coords.iter().enumerate() {
        state.positions[3 * node] = xyz[0];
        state.positions[3 * node + 1] = xyz[1];
        state.positions[3 * node + 2] = xyz[2];
    }
    state.masses.copy_from_slice(&masses);
    let slaves: Vec<usize> = (0..num_nodes).collect();

    let mut material = Law2::c4340_steel();
    material.damage = Some(JohnsonCookDamage::c4340_steel());
    // Isothermal leg: no heating, no thermal softening.
    material.m = 0.0;
    material.volumetric_heat_capacity = 0.0;
    let params = HexExplicitParams::new(density, dilatational);
    let states: RefCell<Vec<HexExplicitState<Law2State>>> = RefCell::new(
        (0..num_elems)
            .map(|_| HexExplicitState::new(Law2State::new(298.0)))
            .collect(),
    );
    let erosion = RefCell::new(ErosionSet::new(num_elems));
    let criterion = ErosionCriterion {
        max_equiv_plastic_strain: FAIL_EP,
        max_damage: 1.0,
        max_porosity: f64::INFINITY,
        max_tabular_damage: f64::INFINITY,
    };
    let step_dt = RefCell::new(dt_max);
    let max_ep = RefCell::new(0.0f64);

    let internal_force = |positions: &[f64], velocities: &[f64], _: f64| {
        let dt = *step_dt.borrow();
        let mut forces = vec![0.0; positions.len()];
        let mut states = states.borrow_mut();
        let mut erosion = erosion.borrow_mut();
        let mut features = Vec::with_capacity(num_elems);
        for (elem, brick) in mesh.conn.iter().enumerate() {
            if !erosion.is_active(elem) {
                features.push((0.0, 0.0, 0.0));
                continue;
            }
            let mut elem_coords = [[0.0; 3]; 8];
            for (slot, node) in brick.iter().enumerate() {
                elem_coords[slot] = [
                    positions[3 * node],
                    positions[3 * node + 1],
                    positions[3 * node + 2],
                ];
            }
            let geom = match HexExplicitGeom::new(elem_coords) {
                Ok(geom) => geom,
                Err(_) => {
                    features.push((f64::INFINITY, 0.0, 0.0));
                    continue;
                }
            };
            let mut vel = [0.0; 24];
            for (slot, node) in brick.iter().enumerate() {
                vel[slot * 3] = velocities[3 * node];
                vel[slot * 3 + 1] = velocities[3 * node + 1];
                vel[slot * 3 + 2] = velocities[3 * node + 2];
            }
            match hex_plastic_forces(&geom, &mut states[elem], &material, &vel, dt, &params) {
                Ok(out) => {
                    for (slot, node) in brick.iter().enumerate() {
                        forces[3 * node] -= out.base.forces[slot * 3];
                        forces[3 * node + 1] -= out.base.forces[slot * 3 + 1];
                        forces[3 * node + 2] -= out.base.forces[slot * 3 + 2];
                    }
                    let peak = out.features.0;
                    let mut running = max_ep.borrow_mut();
                    *running = (*running).max(peak);
                    features.push(out.features);
                }
                Err(_) => features.push((f64::INFINITY, 0.0, 0.0)),
            }
        }
        erosion.sweep(&criterion, &features);
        forces
    };
    let external_force = |positions: &[f64], velocities: &[f64], time: f64| {
        let mut wall = RigidWall::new([0.0, 0.0, VELOCITY * time], [0.0, 0.0, 1.0], penalty);
        wall.velocity = [0.0, 0.0, VELOCITY];
        let mut forces = vec![0.0; positions.len()];
        rigid_wall_contact(&wall, &slaves, positions, velocities, &mut forces);
        forces
    };

    let mut steps = 0u64;
    let mut dt_last = dt_max;
    while state.time < TSTOP {
        assert!(steps < 200_000, "mini run stalled");
        let char_lengths: Vec<f64> = mesh
            .conn
            .iter()
            .map(|brick| {
                let mut elem_coords = [[0.0; 3]; 8];
                for (slot, node) in brick.iter().enumerate() {
                    elem_coords[slot] = [
                        state.positions[3 * node],
                        state.positions[3 * node + 1],
                        state.positions[3 * node + 2],
                    ];
                }
                HexExplicitGeom::new(elem_coords)
                    .map(|geom| geom.char_length)
                    .unwrap_or(min_char)
            })
            .collect();
        let dt_crit = critical_dt(&char_lengths, &sound_speeds);
        let dt = (0.9 * dt_crit)
            .min(dt_max)
            .min(1.1 * dt_last)
            .min(TSTOP - state.time)
            .max(1.0e-12);
        *step_dt.borrow_mut() = dt;
        driver
            .step_fixed(&mut state, internal_force, external_force, dt)
            .expect("mini run must stay stable");
        dt_last = dt;
        steps += 1;
    }
    let balance = driver.balance().clone();
    let scale = balance.kinetic + balance.internal.abs();
    let drift = balance.drift.abs() / scale;
    let eroded = num_elems - erosion.borrow().active_count();
    assert!(
        *max_ep.borrow() > 0.05,
        "face must yield, max ep {}",
        max_ep.borrow()
    );
    assert!(eroded >= 1, "erosion must trip at ep {FAIL_EP}");
    assert!(eroded < num_elems, "the bar must survive");
    assert!(drift < 0.05, "energy drift {drift:.3e} too large");
}
