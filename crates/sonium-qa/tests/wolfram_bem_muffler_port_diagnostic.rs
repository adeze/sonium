//! Reduced-order BEM port diagnostic motivated by Wolfram's classical
//! `Acoustic Muffler` model.
//!
//! Wolfram's proprietary muffler STL and numerical transmission-loss samples
//! are not redistributable. This test therefore does not claim literal
//! geometry parity. It isolates the reusable BEM primitive needed by that
//! example: an internal duct with prescribed inlet velocity, rigid walls, and
//! a characteristic-admittance outlet. Below cutoff, the independent solution
//! is a lossless plane wave with pressure `rho c u` and conserved port power.
//! The test enforces pressure, phase, and power parity with that independent
//! plane-wave solution.

use num_complex::Complex64;
use sonium_bem::{
    core::{
        BoundaryCondition, PhysicsParams,
        assembly::tbem::build_tbem_system_stabilized_with_beta,
        contracts::constant_panel_power,
        solver::{GmresConfig, gmres_solve_tbem_with_ilu},
    },
    room_acoustics::{RectangularRoom, room_mesh_to_core_elements},
};
use sonium_mesher::RoomSurfaceMeshing;
use std::f64::consts::PI;

const LENGTH_M: f64 = 1.0;
const CROSS_SECTION_M: f64 = 0.1;
const FREQUENCY_HZ: f64 = 100.0;
const SOUND_SPEED_M_S: f64 = 343.0;
const DENSITY_KG_M3: f64 = 1.21;
const INLET_PARTICLE_VELOCITY_M_S: f64 = 1.0;

#[derive(Debug)]
struct DuctResult {
    inlet_pressure: Complex64,
    outlet_pressure: Complex64,
    inlet_nonuniformity: f64,
    outlet_nonuniformity: f64,
    inlet_power_w: f64,
    outlet_power_w: f64,
    power_imbalance: f64,
    relative_residual: f64,
    panel_count: usize,
}

fn weighted_port_pressure(
    elements: &[sonium_bem::core::Element],
    pressure: &[Complex64],
    indices: &[usize],
) -> (Complex64, f64) {
    let area = indices
        .iter()
        .map(|&index| elements[index].area)
        .sum::<f64>();
    let average = indices
        .iter()
        .map(|&index| pressure[index] * elements[index].area)
        .sum::<Complex64>()
        / area;
    let variation = (indices
        .iter()
        .map(|&index| (pressure[index] - average).norm_sqr() * elements[index].area)
        .sum::<f64>()
        / area)
        .sqrt()
        / average.norm().max(1.0e-30);
    (average, variation)
}

fn solve_duct(elements_per_meter: usize) -> DuctResult {
    let room = RectangularRoom::new(LENGTH_M, CROSS_SECTION_M, CROSS_SECTION_M);
    let mesh = room.generate_mesh(elements_per_meter);
    let (mut elements, nodes) = room_mesh_to_core_elements(&mesh, 0.0);
    let characteristic_admittance = Complex64::new(1.0 / (DENSITY_KG_M3 * SOUND_SPEED_M_S), 0.0);
    let prescribed_inlet_normal_velocity = Complex64::new(-INLET_PARTICLE_VELOCITY_M_S, 0.0);
    let mut inlet = Vec::new();
    let mut outlet = Vec::new();
    for (index, element) in elements.iter_mut().enumerate() {
        if element.center[0].abs() < 1.0e-12 {
            inlet.push(index);
            element.boundary_condition =
                BoundaryCondition::Velocity(vec![prescribed_inlet_normal_velocity]);
        } else if (element.center[0] - LENGTH_M).abs() < 1.0e-12 {
            outlet.push(index);
            element.boundary_condition = BoundaryCondition::VelocityWithAdmittance {
                velocity: vec![Complex64::new(0.0, 0.0)],
                admittance: characteristic_admittance,
            };
        } else {
            element.boundary_condition =
                BoundaryCondition::Velocity(vec![Complex64::new(0.0, 0.0)]);
        }
        element.dof_addresses = vec![index];
    }
    assert!(!inlet.is_empty() && !outlet.is_empty());

    let physics = PhysicsParams::new(FREQUENCY_HZ, SOUND_SPEED_M_S, DENSITY_KG_M3, true);
    let beta = physics.burton_miller_beta_scaled(4.0);
    let system = build_tbem_system_stabilized_with_beta(&elements, &nodes, &physics, -1.0, beta);
    let solved = gmres_solve_tbem_with_ilu(
        &system.matrix,
        &system.rhs,
        &GmresConfig {
            max_iterations: 1_000,
            restart: 100,
            tolerance: 1.0e-11,
            print_interval: 0,
            convergence_check_interval: 1,
            min_restart: None,
            max_restart: None,
            exact_residual_check_interval: 1,
        },
    );
    assert!(
        solved.converged,
        "mixed-port duct solve did not converge: residual={}",
        solved.residual
    );
    let pressure = solved.x;
    let (inlet_pressure, inlet_nonuniformity) =
        weighted_port_pressure(&elements, &pressure, &inlet);
    let (outlet_pressure, outlet_nonuniformity) =
        weighted_port_pressure(&elements, &pressure, &outlet);

    let inlet_areas: Vec<_> = inlet.iter().map(|&index| elements[index].area).collect();
    let inlet_pressures: Vec<_> = inlet.iter().map(|&index| pressure[index]).collect();
    let inlet_velocities = vec![prescribed_inlet_normal_velocity; inlet.len()];
    let outward_inlet_power =
        constant_panel_power(&inlet_areas, &inlet_pressures, &inlet_velocities)
            .expect("inlet panel power");
    let outlet_areas: Vec<_> = outlet.iter().map(|&index| elements[index].area).collect();
    let outlet_pressures: Vec<_> = outlet.iter().map(|&index| pressure[index]).collect();
    let outlet_velocities: Vec<_> = outlet_pressures
        .iter()
        .map(|pressure| characteristic_admittance * *pressure)
        .collect();
    let outlet_power = constant_panel_power(&outlet_areas, &outlet_pressures, &outlet_velocities)
        .expect("outlet panel power");
    let inlet_power = -outward_inlet_power;
    let power_imbalance =
        (inlet_power - outlet_power).abs() / inlet_power.abs().max(outlet_power.abs()).max(1.0e-30);

    DuctResult {
        inlet_pressure,
        outlet_pressure,
        inlet_nonuniformity,
        outlet_nonuniformity,
        inlet_power_w: inlet_power,
        outlet_power_w: outlet_power,
        power_imbalance,
        relative_residual: solved.residual,
        panel_count: elements.len(),
    }
}

#[test]
fn sonium_bem_mixed_muffler_ports_match_plane_wave_reference() {
    let coarse = solve_duct(10);
    let fine = solve_duct(20);
    eprintln!(
        "Wolfram muffler reduced-order BEM port diagnostic: coarse={coarse:?}, fine={fine:?}"
    );

    let impedance = DENSITY_KG_M3 * SOUND_SPEED_M_S;
    let area = CROSS_SECTION_M.powi(2);
    let exact_pressure = impedance * INLET_PARTICLE_VELOCITY_M_S;
    let exact_power = 0.5 * impedance * INLET_PARTICLE_VELOCITY_M_S.powi(2) * area;
    let wavenumber = 2.0 * PI * FREQUENCY_HZ / SOUND_SPEED_M_S;
    let phase_ratio = fine.outlet_pressure / fine.inlet_pressure;
    let expected_phases = [
        Complex64::new(0.0, wavenumber * LENGTH_M).exp(),
        Complex64::new(0.0, -wavenumber * LENGTH_M).exp(),
    ];
    let phase_error = expected_phases
        .iter()
        .map(|expected| (phase_ratio - expected).norm())
        .fold(f64::INFINITY, f64::min);
    let inlet_pressure_error = (fine.inlet_pressure.norm() - exact_pressure).abs() / exact_pressure;
    let outlet_pressure_error =
        (fine.outlet_pressure.norm() - exact_pressure).abs() / exact_pressure;
    let inlet_power_error = (fine.inlet_power_w - exact_power).abs() / exact_power;
    let outlet_power_error = (fine.outlet_power_w - exact_power).abs() / exact_power;
    eprintln!(
        "interior-BEM ports: inlet/outlet pressure errors={inlet_pressure_error:.6}/{outlet_pressure_error:.6}, inlet/outlet power errors={inlet_power_error:.6}/{outlet_power_error:.6}, imbalance={:.6}",
        fine.power_imbalance
    );

    assert!(fine.relative_residual < 1.0e-9);
    assert!(fine.panel_count > coarse.panel_count);
    assert!(fine.inlet_nonuniformity < 0.08);
    assert!(fine.outlet_nonuniformity < 0.08);
    assert!(inlet_pressure_error < 0.10);
    assert!(outlet_pressure_error < 0.10);
    assert!(phase_error < 0.08);
    assert!(inlet_power_error < 0.10);
    assert!(outlet_power_error < 0.10);
    assert!(fine.power_imbalance < 0.02);
    assert!(
        (fine.outlet_pressure - coarse.outlet_pressure).norm()
            / fine.outlet_pressure.norm().max(1.0e-30)
            < 0.12
    );
}
