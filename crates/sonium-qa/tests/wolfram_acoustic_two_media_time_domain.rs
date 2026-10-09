//! Sonium reproduction of Wolfram's time-domain two-medium Gaussian-pulse
//! example from `AcousticsTimeDomain`.
//!
//! Wolfram uses a 1D line. Sonium currently has no 1D FEM element type, so the
//! same plane-wave PDE is solved on a thin, rigid-walled 2D strip with an
//! interface-aligned mesh. The field stays uniform across the strip.

use sonium_fem::{
    assembly::{PrebakedAssembler, RegionMap, assemble_bilinear_by_region},
    basis::PolynomialDegree,
    forms::{WeightedLaplacianForm, WeightedMassForm},
    mesh::generators::rectangular_mesh_triangles,
    physics::acoustics::pressure::time_domain::{NewmarkConfig, NewmarkStepper, WaveState},
    spaces::H1Space,
};
use sonium_solvers::SparseLuFactorization;

const LENGTH_M: f64 = 1.5;
const INTERFACE_M: f64 = 0.5;
const STRIP_HEIGHT_M: f64 = 0.01;
const PULSE_WIDTH_M: f64 = 0.05;
const END_TIME_S: f64 = 0.002;
const TIME_STEP_S: f64 = 2.0e-6;
const PRESSURE_PA: f64 = 101_325.0;
const SPECIFIC_GAS_CONSTANT: f64 = 287.05;
const HEAT_CAPACITY_RATIO: f64 = 1.4;
const COLD_TEMPERATURE_K: f64 = 250.0;
const HOT_TEMPERATURE_K: f64 = 600.0;

#[derive(Clone, Copy)]
struct Medium {
    sound_speed: f64,
    density: f64,
}

impl Medium {
    fn ideal_air(temperature: f64) -> Self {
        Self {
            sound_speed: (HEAT_CAPACITY_RATIO * SPECIFIC_GAS_CONSTANT * temperature).sqrt(),
            density: PRESSURE_PA / (SPECIFIC_GAS_CONSTANT * temperature),
        }
    }

    fn impedance(self) -> f64 {
        self.sound_speed * self.density
    }
}

fn energy(
    state: &WaveState,
    mass: &sonium_solvers::CsrMatrix<f64>,
    stiffness: &sonium_solvers::CsrMatrix<f64>,
) -> f64 {
    let mass_velocity = mass.matvec(&state.velocity);
    let stiffness_pressure = stiffness.matvec(&state.pressure);
    0.5 * (state
        .velocity
        .iter()
        .zip(mass_velocity)
        .map(|(left, right)| left * right)
        .sum::<f64>()
        + state
            .pressure
            .iter()
            .zip(stiffness_pressure)
            .map(|(left, right)| left * right)
            .sum::<f64>())
}

#[test]
fn sonium_reproduces_wolfram_two_medium_pulse_scattering() {
    let cold = Medium::ideal_air(COLD_TEMPERATURE_K);
    let hot = Medium::ideal_air(HOT_TEMPERATURE_K);
    let cold_impedance = cold.impedance();
    let hot_impedance = hot.impedance();
    let reflection = (hot_impedance - cold_impedance) / (hot_impedance + cold_impedance);
    let transmission = 2.0 * hot_impedance / (hot_impedance + cold_impedance);
    let reflected_power = reflection * reflection;
    let transmitted_power = transmission * transmission * cold_impedance / hot_impedance;
    assert!((reflected_power + transmitted_power - 1.0).abs() < 1.0e-14);
    assert!(
        reflection < 0.0,
        "hot-air reflection must reverse pressure polarity"
    );

    // dx=2.5 mm gives 20 cells across Wolfram's Gaussian standard scale and
    // places x=0.5 exactly on a node/material interface.
    let mut mesh = rectangular_mesh_triangles(0.0, LENGTH_M, 0.0, STRIP_HEIGHT_M, 600, 1);
    for element in &mut mesh.elements {
        let vertices = element.vertices();
        let centroid_x =
            vertices.iter().map(|&node| mesh.nodes[node].x).sum::<f64>() / vertices.len() as f64;
        element.marker = if centroid_x < INTERFACE_M { 1 } else { 2 };
    }

    let mut mass_forms = RegionMap::new();
    mass_forms.insert(
        1,
        WeightedMassForm {
            coefficient: 1.0 / (cold.density * cold.sound_speed.powi(2)),
        },
    );
    mass_forms.insert(
        2,
        WeightedMassForm {
            coefficient: 1.0 / (hot.density * hot.sound_speed.powi(2)),
        },
    );
    let mut stiffness_forms = RegionMap::new();
    stiffness_forms.insert(
        1,
        WeightedLaplacianForm {
            coefficient: 1.0 / cold.density,
        },
    );
    stiffness_forms.insert(
        2,
        WeightedLaplacianForm {
            coefficient: 1.0 / hot.density,
        },
    );

    let space = H1Space::new(PolynomialDegree::P1);
    let mass_triplet = assemble_bilinear_by_region(&mass_forms, &space, &space, &mesh, 2)
        .expect("two-medium acoustic mass assembly");
    let stiffness_triplet = assemble_bilinear_by_region(&stiffness_forms, &space, &space, &mesh, 2)
        .expect("two-medium acoustic stiffness assembly");
    let assembler = PrebakedAssembler::<2>::from_triplets([stiffness_triplet, mass_triplet], &[]);
    let stiffness = assembler.assemble_real(&[1.0, 0.0]);
    let mass = assembler.assemble_real(&[0.0, 1.0]);

    let stepper = NewmarkStepper::new(NewmarkConfig::trapezoidal(TIME_STEP_S));
    let effective = stepper.effective_stiffness(&stiffness, &mass);
    let effective_factor =
        SparseLuFactorization::factorize(&effective).expect("Newmark effective factorization");
    let mass_factor =
        SparseLuFactorization::factorize(&mass).expect("two-medium acoustic mass factorization");

    let mut state = WaveState::zeros(mesh.num_nodes());
    for (value, point) in state.pressure.iter_mut().zip(&mesh.nodes) {
        *value = 2.0 * (-(point.x / PULSE_WIDTH_M).powi(2)).exp();
    }
    let zero_force = vec![0.0; mesh.num_nodes()];
    let initial_acceleration_rhs =
        stepper.initial_acceleration_rhs(&state.pressure, &zero_force, &stiffness);
    state.acceleration = mass_factor
        .solve(&initial_acceleration_rhs)
        .expect("initial acoustic acceleration");
    let initial_energy = energy(&state, &mass, &stiffness);

    let steps = (END_TIME_S / TIME_STEP_S).round() as usize;
    for _ in 0..steps {
        let rhs = stepper.form_rhs(&state, &zero_force, &stiffness);
        let acceleration = effective_factor.solve(&rhs).expect("Newmark acoustic step");
        stepper.update_state(&mut state, &acceleration);
    }

    let final_energy = energy(&state, &mass, &stiffness);
    let energy_drift = ((final_energy - initial_energy) / initial_energy).abs();
    assert!(
        energy_drift < 2.0e-8,
        "Newmark energy drift {energy_drift:.3e}"
    );

    let reflected = mesh
        .nodes
        .iter()
        .zip(&state.pressure)
        .filter(|(point, _)| point.y.abs() < 1.0e-12 && (0.2..INTERFACE_M).contains(&point.x))
        .min_by(|(_, left), (_, right)| left.total_cmp(right))
        .expect("reflected pulse sample");
    let transmitted = mesh
        .nodes
        .iter()
        .zip(&state.pressure)
        .filter(|(point, _)| point.y.abs() < 1.0e-12 && (INTERFACE_M..0.85).contains(&point.x))
        .max_by(|(_, left), (_, right)| left.total_cmp(right))
        .expect("transmitted pulse sample");

    let reflected_error = ((reflected.1 - reflection) / reflection).abs();
    let transmitted_error = ((transmitted.1 - transmission) / transmission).abs();
    assert!(
        reflected_error < 0.05,
        "reflected amplitude {:.9} versus impedance result {reflection:.9}, relative error {reflected_error:.3e}",
        reflected.1
    );
    assert!(
        transmitted_error < 0.03,
        "transmitted amplitude {:.9} versus impedance result {transmission:.9}, relative error {transmitted_error:.3e}",
        transmitted.1
    );

    let interface_arrival = INTERFACE_M / cold.sound_speed;
    let remaining_time = END_TIME_S - interface_arrival;
    let expected_reflected_x = INTERFACE_M - cold.sound_speed * remaining_time;
    let expected_transmitted_x = INTERFACE_M + hot.sound_speed * remaining_time;
    assert!((reflected.0.x - expected_reflected_x).abs() < 0.006);
    assert!((transmitted.0.x - expected_transmitted_x).abs() < 0.006);
}
