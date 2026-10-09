//! Sonium reproduction of Wolfram's oblique-incidence radiation example.
//!
//! A 5025 rad/s plane wave enters a slanted 2D duct at 30 degrees to the
//! bottom normal. The side walls are parallel to propagation and the outlet
//! is perpendicular to it, so the published plane wave is an exact solution.

use sonium_fem::{
    assembly::{PrebakedAssembler, assemble_bilinear, assemble_boundary_mass_generic},
    basis::PolynomialDegree,
    forms::{LaplacianForm, MassForm},
    mesh::{BoundaryType, triangulate::polygon_mesh_triangles},
    spaces::H1Space,
};
use sonium_solvers::SparseLuFactorization;
use std::f64::consts::PI;

const SOUND_SPEED_M_S: f64 = 343.2;
const AIR_DENSITY_KG_M3: f64 = 1.2041;
const ANGULAR_FREQUENCY_RAD_S: f64 = 5025.0;
const TIME_STEP_S: f64 = 2.0e-6;
const END_TIME_S: f64 = 0.002;
const BOTTOM_MARKER: i32 = 1;
const TOP_MARKER: i32 = 3;

fn exact_pressure(time: f64, x: f64, y: f64, wave_x: f64, wave_y: f64) -> f64 {
    (-(wave_x * x + wave_y * y) + ANGULAR_FREQUENCY_RAD_S * time).cos()
}

fn exact_velocity(time: f64, x: f64, y: f64, wave_x: f64, wave_y: f64) -> f64 {
    -ANGULAR_FREQUENCY_RAD_S * (-(wave_x * x + wave_y * y) + ANGULAR_FREQUENCY_RAD_S * time).sin()
}

#[test]
fn sonium_reproduces_wolfram_oblique_radiation_plane_wave() {
    let angle = PI / 6.0;
    let direction_x = angle.sin();
    let direction_y = angle.cos();
    let wavenumber = ANGULAR_FREQUENCY_RAD_S / SOUND_SPEED_M_S;
    let wave_x = wavenumber * direction_x;
    let wave_y = wavenumber * direction_y;

    // Literal four vertices from Wolfram's Polygon expression.
    let vertices = [
        (0.0, 0.0),
        (0.2, 0.0),
        (
            0.2 + direction_x - direction_x.powi(2) / 5.0,
            direction_y - 0.2 * direction_y * direction_x,
        ),
        (direction_x, direction_y),
    ];
    let mut mesh = polygon_mesh_triangles(&vertices, 6, 0)
        .expect("Wolfram oblique-radiation polygon must triangulate");
    for boundary in &mut mesh.boundaries {
        boundary.boundary_type = BoundaryType::Neumann;
        boundary.marker = 0;
    }
    mesh.set_boundary_condition(BoundaryType::Robin, BOTTOM_MARKER, |points| {
        points.iter().all(|point| point.y.abs() < 1.0e-12)
    });
    mesh.set_boundary_condition(BoundaryType::Robin, TOP_MARKER, |points| {
        points
            .iter()
            .all(|point| (point.x * direction_x + point.y * direction_y - 1.0).abs() < 1.0e-12)
    });

    let bottom_edges = mesh
        .boundaries
        .iter()
        .filter(|boundary| boundary.marker == BOTTOM_MARKER)
        .count();
    let top_edges = mesh
        .boundaries
        .iter()
        .filter(|boundary| boundary.marker == TOP_MARKER)
        .count();
    assert!(bottom_edges > 0 && top_edges > 0);

    let space = H1Space::new(PolynomialDegree::P1);
    let laplacian = assemble_bilinear(&LaplacianForm, &space, &space, &mesh, 2);
    let mass = assemble_bilinear(&MassForm, &space, &space, &mesh, 2);
    let bottom = assemble_boundary_mass_generic(&mesh, BOTTOM_MARKER);
    let top = assemble_boundary_mass_generic(&mesh, TOP_MARKER);
    let assembler = PrebakedAssembler::<4>::from_triplets([laplacian, mass, bottom, top], &[]);

    let normal_factor = direction_y;
    let mass_matrix = assembler.assemble_real(&[
        0.0,
        1.0 / (AIR_DENSITY_KG_M3 * SOUND_SPEED_M_S.powi(2)),
        0.0,
        0.0,
    ]);
    let damping_matrix = assembler.assemble_real(&[
        0.0,
        0.0,
        normal_factor / (AIR_DENSITY_KG_M3 * SOUND_SPEED_M_S),
        1.0 / (AIR_DENSITY_KG_M3 * SOUND_SPEED_M_S),
    ]);
    let stiffness_matrix = assembler.assemble_real(&[1.0 / AIR_DENSITY_KG_M3, 0.0, 0.0, 0.0]);
    let bottom_mass = assembler.assemble_real(&[0.0, 0.0, 1.0, 0.0]);

    let beta = 0.25;
    let gamma = 0.5;
    let effective_matrix = assembler.assemble_real(&[
        beta * TIME_STEP_S.powi(2) / AIR_DENSITY_KG_M3,
        1.0 / (AIR_DENSITY_KG_M3 * SOUND_SPEED_M_S.powi(2)),
        gamma * TIME_STEP_S * normal_factor / (AIR_DENSITY_KG_M3 * SOUND_SPEED_M_S),
        gamma * TIME_STEP_S / (AIR_DENSITY_KG_M3 * SOUND_SPEED_M_S),
    ]);
    let mass_factorization =
        SparseLuFactorization::factorize(&mass_matrix).expect("oblique mass factorization");
    let effective_factorization =
        SparseLuFactorization::factorize(&effective_matrix).expect("oblique Newmark factorization");

    let mut pressure = mesh
        .nodes
        .iter()
        .map(|point| exact_pressure(0.0, point.x, point.y, wave_x, wave_y))
        .collect::<Vec<_>>();
    let mut velocity = mesh
        .nodes
        .iter()
        .map(|point| exact_velocity(0.0, point.x, point.y, wave_x, wave_y))
        .collect::<Vec<_>>();

    let boundary_source = |time: f64| {
        let nodal_flux = mesh
            .nodes
            .iter()
            .map(|point| {
                2.0 * normal_factor / (AIR_DENSITY_KG_M3 * SOUND_SPEED_M_S)
                    * exact_velocity(time, point.x, point.y, wave_x, wave_y)
            })
            .collect::<Vec<_>>();
        bottom_mass.matvec(&nodal_flux)
    };
    let initial_source = boundary_source(0.0);
    let damping_velocity = damping_matrix.matvec(&velocity);
    let stiffness_pressure = stiffness_matrix.matvec(&pressure);
    let initial_rhs = initial_source
        .iter()
        .zip(&damping_velocity)
        .zip(&stiffness_pressure)
        .map(|((&source, &damping), &stiffness)| source - damping - stiffness)
        .collect::<Vec<_>>();
    let mut acceleration = mass_factorization
        .solve(&initial_rhs)
        .expect("oblique initial acceleration");

    let steps = (END_TIME_S / TIME_STEP_S).round() as usize;
    for step in 0..steps {
        let time_next = (step + 1) as f64 * TIME_STEP_S;
        let pressure_predictor = pressure
            .iter()
            .zip(&velocity)
            .zip(&acceleration)
            .map(|((&p, &v), &a)| p + TIME_STEP_S * v + (0.5 - beta) * TIME_STEP_S.powi(2) * a)
            .collect::<Vec<_>>();
        let velocity_predictor = velocity
            .iter()
            .zip(&acceleration)
            .map(|(&v, &a)| v + (1.0 - gamma) * TIME_STEP_S * a)
            .collect::<Vec<_>>();
        let source = boundary_source(time_next);
        let damping_velocity = damping_matrix.matvec(&velocity_predictor);
        let stiffness_pressure = stiffness_matrix.matvec(&pressure_predictor);
        let rhs = source
            .iter()
            .zip(&damping_velocity)
            .zip(&stiffness_pressure)
            .map(|((&forcing, &damping), &stiffness)| forcing - damping - stiffness)
            .collect::<Vec<_>>();
        let acceleration_new = effective_factorization
            .solve(&rhs)
            .expect("oblique Newmark step");

        pressure = pressure_predictor
            .iter()
            .zip(&acceleration_new)
            .map(|(&predictor, &a)| predictor + beta * TIME_STEP_S.powi(2) * a)
            .collect();
        velocity = velocity_predictor
            .iter()
            .zip(&acceleration_new)
            .map(|(&predictor, &a)| predictor + gamma * TIME_STEP_S * a)
            .collect();
        acceleration = acceleration_new;
    }

    let errors = mesh
        .nodes
        .iter()
        .zip(&pressure)
        .map(|(point, &computed)| {
            (computed - exact_pressure(END_TIME_S, point.x, point.y, wave_x, wave_y)).abs()
        })
        .collect::<Vec<_>>();
    let maximum_error = errors.iter().copied().fold(0.0_f64, f64::max);
    let rms_error =
        (errors.iter().map(|error| error.powi(2)).sum::<f64>() / errors.len() as f64).sqrt();
    assert!(
        maximum_error < 0.05,
        "oblique radiation maximum plane-wave error {maximum_error:.6} Pa"
    );
    assert!(
        rms_error < 0.02,
        "oblique radiation RMS plane-wave error {rms_error:.6} Pa"
    );
}
