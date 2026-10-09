//! Sonium reproduction of Wolfram's porous-absorber example from
//! `AcousticsTimeDomain`.
//!
//! The test preserves Wolfram's corrected bulk modulus, distributed flow-
//! resistivity damping, 1 kHz pressure source, and matched lossy outlet. The
//! independent reference is the complex wavenumber of the published PDE.

use num_complex::Complex64;
use sonium_fem::{
    assembly::{PrebakedAssembler, assemble_bilinear, assemble_boundary_mass_generic},
    basis::PolynomialDegree,
    forms::{LaplacianForm, MassForm},
    mesh::{BoundaryType, generators::rectangular_mesh_triangles},
    spaces::H1Space,
};
use sonium_solvers::{CsrMatrix, SparseLuFactorization, TripletMatrix};
use std::f64::consts::PI;

const LENGTH_M: f64 = 0.5;
const STRIP_HEIGHT_M: f64 = 0.01;
const SOUND_SPEED_M_S: f64 = 343.2;
const AIR_DENSITY_KG_M3: f64 = 1.2041;
const FREQUENCY_HZ: f64 = 1000.0;
const POROSITY: f64 = 0.8;
const FLOW_RESISTIVITY_PA_S_M2: f64 = 5000.0;
const TIME_STEP_S: f64 = 2.0e-6;
const EXTENDED_END_TIME_S: f64 = 0.008;
const SAMPLE_START_TIME_S: f64 = 0.007;
const OUTLET_MARKER: i32 = 8;

fn reduced_matrix(matrix: &CsrMatrix<f64>, free_dofs: &[usize]) -> CsrMatrix<f64> {
    let mut reduced_index = vec![usize::MAX; matrix.num_rows];
    for (reduced, &full) in free_dofs.iter().enumerate() {
        reduced_index[full] = reduced;
    }
    let mut triplet = TripletMatrix::new(free_dofs.len(), free_dofs.len());
    for (reduced_row, &full_row) in free_dofs.iter().enumerate() {
        for (full_col, value) in matrix.row_entries(full_row) {
            let reduced_col = reduced_index[full_col];
            if reduced_col != usize::MAX {
                triplet.push(reduced_row, reduced_col, value);
            }
        }
    }
    triplet.to_csr()
}

#[test]
fn sonium_reproduces_wolfram_porous_absorber_attenuation() {
    let correction =
        1.0 + 0.0978 * (AIR_DENSITY_KG_M3 * FREQUENCY_HZ / FLOW_RESISTIVITY_PA_S_M2).powf(-0.7);
    let bulk_modulus = AIR_DENSITY_KG_M3 * SOUND_SPEED_M_S.powi(2) / correction.powi(2);
    let omega = 2.0 * PI * FREQUENCY_HZ;
    let damping_rate = POROSITY * FLOW_RESISTIVITY_PA_S_M2 / AIR_DENSITY_KG_M3;

    // For p = exp(i omega t - i k x), Wolfram's PDE gives
    // k^2 = rho omega^2/kappa - i omega phi Rf/kappa.
    let wavenumber_squared = Complex64::new(
        AIR_DENSITY_KG_M3 * omega.powi(2) / bulk_modulus,
        -omega * POROSITY * FLOW_RESISTIVITY_PA_S_M2 / bulk_modulus,
    );
    let wavenumber = wavenumber_squared.sqrt();
    let attenuation_np_m = -wavenumber.im;
    assert!(attenuation_np_m > 0.0, "porous model must be passive");

    // Wolfram's outlet condition uses the lossless sound speed, so it is not
    // exactly matched to the porous medium's complex wavenumber. Retain that
    // small finite-domain reflection in the independent harmonic reference.
    let imaginary_unit = Complex64::new(0.0, 1.0);
    let outlet_admittance = Complex64::new(
        -SOUND_SPEED_M_S * POROSITY * FLOW_RESISTIVITY_PA_S_M2 / bulk_modulus,
        -AIR_DENSITY_KG_M3 * omega * SOUND_SPEED_M_S / bulk_modulus,
    );
    let reflected_to_incident = (imaginary_unit * wavenumber + outlet_admittance)
        / (imaginary_unit * wavenumber - outlet_admittance)
        * (-2.0 * imaginary_unit * wavenumber * LENGTH_M).exp();
    let incident_amplitude =
        Complex64::new(1.0, 0.0) / (Complex64::new(1.0, 0.0) + reflected_to_incident);
    let reflected_amplitude = reflected_to_incident * incident_amplitude;
    let reference_amplitude = |x: f64| {
        (incident_amplitude * (-imaginary_unit * wavenumber * x).exp()
            + reflected_amplitude * (imaginary_unit * wavenumber * x).exp())
        .norm()
    };

    let mut mesh = rectangular_mesh_triangles(0.0, LENGTH_M, 0.0, STRIP_HEIGHT_M, 200, 1);
    mesh.set_boundary_condition(BoundaryType::Robin, OUTLET_MARKER, |points| {
        points
            .iter()
            .all(|point| (point.x - LENGTH_M).abs() < 1.0e-12)
    });
    let space = H1Space::new(PolynomialDegree::P1);
    let laplacian = assemble_bilinear(&LaplacianForm, &space, &space, &mesh, 2);
    let mass = assemble_bilinear(&MassForm, &space, &space, &mesh, 2);
    let outlet = assemble_boundary_mass_generic(&mesh, OUTLET_MARKER);
    let assembler = PrebakedAssembler::<3>::from_triplets([laplacian, mass, outlet], &[]);

    // M p_ddot + C p_dot + K p = 0, including Wolfram's lossy ABC terms.
    let damping_matrix = assembler.assemble_real(&[
        0.0,
        POROSITY * FLOW_RESISTIVITY_PA_S_M2 / (AIR_DENSITY_KG_M3 * bulk_modulus),
        SOUND_SPEED_M_S / bulk_modulus,
    ]);
    let stiffness_matrix = assembler.assemble_real(&[
        1.0 / AIR_DENSITY_KG_M3,
        0.0,
        SOUND_SPEED_M_S * POROSITY * FLOW_RESISTIVITY_PA_S_M2 / (AIR_DENSITY_KG_M3 * bulk_modulus),
    ]);

    let beta = 0.25;
    let gamma = 0.5;
    let effective_matrix = assembler.assemble_real(&[
        beta * TIME_STEP_S.powi(2) / AIR_DENSITY_KG_M3,
        1.0 / bulk_modulus
            + gamma * TIME_STEP_S * POROSITY * FLOW_RESISTIVITY_PA_S_M2
                / (AIR_DENSITY_KG_M3 * bulk_modulus),
        gamma * TIME_STEP_S * SOUND_SPEED_M_S / bulk_modulus
            + beta * TIME_STEP_S.powi(2) * SOUND_SPEED_M_S * POROSITY * FLOW_RESISTIVITY_PA_S_M2
                / (AIR_DENSITY_KG_M3 * bulk_modulus),
    ]);

    let constrained_dofs = mesh
        .nodes
        .iter()
        .enumerate()
        .filter(|(_, point)| point.x.abs() < 1.0e-12)
        .map(|(node, _)| node)
        .collect::<Vec<_>>();
    let free_dofs = (0..mesh.num_nodes())
        .filter(|dof| !constrained_dofs.contains(dof))
        .collect::<Vec<_>>();
    let reduced_effective = reduced_matrix(&effective_matrix, &free_dofs);
    let factorization = SparseLuFactorization::factorize(&reduced_effective)
        .expect("porous reduced Newmark factorization");

    let probe_nodes = [0.25, 0.45].map(|probe_x| {
        mesh.nodes
            .iter()
            .position(|point| (point.x - probe_x).abs() < 1.0e-12 && point.y.abs() < 1.0e-12)
            .expect("porous probe must be a mesh node")
    });
    let mut minima = [f64::INFINITY; 2];
    let mut maxima = [f64::NEG_INFINITY; 2];
    let ndof = mesh.num_nodes();
    let mut pressure = vec![0.0; ndof];
    let mut velocity = vec![0.0; ndof];
    let mut acceleration = vec![0.0; ndof];
    let steps = (EXTENDED_END_TIME_S / TIME_STEP_S).round() as usize;

    for step in 0..steps {
        let time_next = (step + 1) as f64 * TIME_STEP_S;
        let pressure_predictor = (0..ndof)
            .map(|index| {
                pressure[index]
                    + TIME_STEP_S * velocity[index]
                    + (0.5 - beta) * TIME_STEP_S.powi(2) * acceleration[index]
            })
            .collect::<Vec<_>>();
        let velocity_predictor = (0..ndof)
            .map(|index| velocity[index] + (1.0 - gamma) * TIME_STEP_S * acceleration[index])
            .collect::<Vec<_>>();

        let prescribed_pressure = (omega * time_next).sin();
        let mut prescribed_acceleration = vec![0.0; ndof];
        for &dof in &constrained_dofs {
            prescribed_acceleration[dof] =
                (prescribed_pressure - pressure_predictor[dof]) / (beta * TIME_STEP_S.powi(2));
        }
        let stiffness_pressure = stiffness_matrix.matvec(&pressure_predictor);
        let damping_velocity = damping_matrix.matvec(&velocity_predictor);
        let constrained_effective = effective_matrix.matvec(&prescribed_acceleration);
        let reduced_rhs = free_dofs
            .iter()
            .map(|&dof| {
                -stiffness_pressure[dof] - damping_velocity[dof] - constrained_effective[dof]
            })
            .collect::<Vec<_>>();
        let reduced_acceleration = factorization
            .solve(&reduced_rhs)
            .expect("porous Newmark step");
        let mut acceleration_new = prescribed_acceleration;
        for (&full, value) in free_dofs.iter().zip(reduced_acceleration) {
            acceleration_new[full] = value;
        }
        for index in 0..ndof {
            pressure[index] =
                pressure_predictor[index] + beta * TIME_STEP_S.powi(2) * acceleration_new[index];
            velocity[index] =
                velocity_predictor[index] + gamma * TIME_STEP_S * acceleration_new[index];
        }
        acceleration = acceleration_new;

        if time_next >= SAMPLE_START_TIME_S {
            for (probe, &node) in probe_nodes.iter().enumerate() {
                minima[probe] = minima[probe].min(pressure[node]);
                maxima[probe] = maxima[probe].max(pressure[node]);
            }
        }
    }

    let measured_amplitudes = [0, 1].map(|probe| 0.5 * (maxima[probe] - minima[probe]));
    let reference_amplitudes = [0.25_f64, 0.45].map(reference_amplitude);
    for probe in 0..2 {
        let error = ((measured_amplitudes[probe] - reference_amplitudes[probe])
            / reference_amplitudes[probe])
            .abs();
        assert!(
            error < 0.06,
            "porous amplitude at probe {probe}: measured {:.9}, finite-domain harmonic reference {:.9}, relative error {error:.3e}",
            measured_amplitudes[probe],
            reference_amplitudes[probe]
        );
    }
    assert!(measured_amplitudes[1] < measured_amplitudes[0]);
    assert!(damping_rate > 0.0);
}
