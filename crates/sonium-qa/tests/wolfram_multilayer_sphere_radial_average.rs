//! Spherical-mean validation derived from Wolfram's concrete
//! `Multilayer Sphere Heat Transfer` model.
//!
//! The literal model has three concentric material layers and an angularly
//! varying heat flux on the outer sphere. Because the heat equation and all
//! material interfaces are linear and radial, angular averaging exactly
//! decouples its degree-zero spherical-harmonic mode. This test solves that
//! mode with the same radii, materials, initial temperature, inner
//! temperature, outer flux, and 50-second interval.
//!
//! This is deliberately not labelled a literal 3D port: Sonium still needs a
//! conforming marked shell mesh and a spatial boundary-flux adapter to check
//! Wolfram's non-axisymmetric radial cuts. The thin strip below is only a
//! finite-element carrier for the exact spherical measure `4 pi r^2 dr`.

use serde::Deserialize;
use sonium_fem::{
    assembly::{TripletMatrix, assemble_bilinear_per_element},
    basis::PolynomialDegree,
    forms::{BilinearForm, LocalMatrix},
    mesh::{Mesh, rectangular_mesh_triangles},
    spaces::{BasisTabulation, H1Space},
};
use sonium_qa::subprocess::WolframCommandExt;
use sonium_solvers::{CsrMatrix, SparseLuFactorization, traits::Preconditioner};
use std::{env, f64::consts::PI, path::PathBuf, process::Command};

const INNER_RADIUS_M: f64 = 1.0;
const INTERFACE_RADII_M: [f64; 2] = [2.0, 4.0];
const OUTER_RADIUS_M: f64 = 6.0;
const CONDUCTIVITY: [f64; 3] = [1.0, 2.0, 4.0];
const VOLUMETRIC_HEAT_CAPACITY: [f64; 3] = [1.0, 0.5, 4.0 / 9.0];
const FINAL_TIME_S: f64 = 50.0;
const SAMPLE_TIMES_S: [f64; 6] = [5.0, 10.0, 20.0, 30.0, 45.0, 50.0];
const SAMPLE_RADII_M: [f64; 3] = [1.5, 3.0, 5.0];
const STRIP_WIDTH_M: f64 = 0.1;

#[derive(Clone, Copy)]
struct SphericalConductionForm {
    conductivity: f64,
}

impl BilinearForm<f64> for SphericalConductionForm {
    fn integrate_point(
        &self,
        test: &BasisTabulation,
        trial: &BasisTabulation,
        weight: f64,
        local_matrix: &mut LocalMatrix<f64>,
    ) {
        let radius = test.physical_coords[0];
        let measure = weight * test.det_j * 4.0 * PI * radius * radius / STRIP_WIDTH_M;
        for i in 0..test.n_dofs {
            for j in 0..trial.n_dofs {
                let radial_gradient = test.grad(i, 0) * trial.grad(j, 0);
                let transverse_gradient = test.grad(i, 1) * trial.grad(j, 1);
                local_matrix.add(
                    i,
                    j,
                    self.conductivity * (radial_gradient + transverse_gradient) * measure,
                );
            }
        }
    }
}

#[derive(Clone, Copy)]
struct SphericalCapacityForm {
    volumetric_heat_capacity: f64,
}

impl BilinearForm<f64> for SphericalCapacityForm {
    fn integrate_point(
        &self,
        test: &BasisTabulation,
        trial: &BasisTabulation,
        weight: f64,
        local_matrix: &mut LocalMatrix<f64>,
    ) {
        let radius = test.physical_coords[0];
        let measure = weight * test.det_j * 4.0 * PI * radius * radius / STRIP_WIDTH_M;
        for i in 0..test.n_dofs {
            for j in 0..trial.n_dofs {
                local_matrix.add(
                    i,
                    j,
                    self.volumetric_heat_capacity * test.value(i) * trial.value(j) * measure,
                );
            }
        }
    }
}

fn layer_at(radius: f64) -> usize {
    if radius < INTERFACE_RADII_M[0] {
        0
    } else if radius < INTERFACE_RADII_M[1] {
        1
    } else {
        2
    }
}

fn angular_average_flux() -> f64 {
    // Wolfram applies theta^2 (pi-theta)^2 phi^2 (pi-phi)^2 for
    // 0 <= theta,phi <= pi. The remaining azimuthal half-sphere has zero
    // flux. Integrals are
    // integral theta^2(pi-theta)^2 sin(theta)dtheta = 4(12-pi^2)
    // and integral phi^2(pi-phi)^2 dphi = pi^5/30.
    (12.0 - PI * PI) * PI.powi(4) / 30.0
}

fn add_scaled(
    first: &CsrMatrix<f64>,
    first_scale: f64,
    second: &CsrMatrix<f64>,
    second_scale: f64,
) -> CsrMatrix<f64> {
    let mut triplets = TripletMatrix::new(first.num_rows, first.num_cols);
    for row in 0..first.num_rows {
        for entry in first.row_ptrs[row]..first.row_ptrs[row + 1] {
            triplets.push(
                row,
                first.col_indices[entry],
                first_scale * first.values[entry],
            );
        }
        for entry in second.row_ptrs[row]..second.row_ptrs[row + 1] {
            triplets.push(
                row,
                second.col_indices[entry],
                second_scale * second.values[entry],
            );
        }
    }
    triplets.to_csr()
}

fn impose_zero_rows(matrix: &mut CsrMatrix<f64>, dofs: &[usize]) {
    for &dof in dofs {
        for entry in matrix.row_ptrs[dof]..matrix.row_ptrs[dof + 1] {
            matrix.values[entry] = if matrix.col_indices[entry] == dof {
                1.0
            } else {
                0.0
            };
        }
    }
}

fn radial_column_average(mesh: &Mesh, values: &[f64], radius: f64) -> f64 {
    let column: Vec<_> = mesh
        .nodes
        .iter()
        .enumerate()
        .filter_map(|(index, node)| ((node.x - radius).abs() < 1.0e-10).then_some(values[index]))
        .collect();
    assert_eq!(
        column.len(),
        2,
        "radius {radius} must align with a mesh column"
    );
    0.5 * (column[0] + column[1])
}

#[derive(Debug)]
struct ResultSummary {
    samples: [[f64; 3]; 6],
    interface_flux_jump: [f64; 2],
    relative_energy_balance: f64,
    relative_free_residual: f64,
    transverse_spread: f64,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct WolframReference {
    spherical_average_flux: f64,
    spherical_average_samples: [[f64; 3]; 6],
}

fn wolfram_reference() -> Option<WolframReference> {
    let Some(executable) = env::var_os("WOLFRAMSCRIPT") else {
        return Some(
            serde_json::from_str(include_str!(
                "../wolfram/goldens/multilayer_sphere_radial_average.json"
            ))
            .expect("parse checked-in Wolfram multilayer-sphere reference"),
        );
    };
    let script = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("wolfram/multilayer_sphere_radial_average.wls");
    let output = match Command::new(&executable)
        .arg("-file")
        .arg(script)
        .wolfram_output()
    {
        Ok(output) => output,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return None,
        Err(error) => panic!("failed to launch {executable:?}: {error}"),
    };
    assert!(
        output.status.success(),
        "Wolfram multilayer-sphere reference failed:\n{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    let json = stdout
        .lines()
        .rev()
        .find(|line| line.trim_start().starts_with('{'))
        .unwrap_or(&stdout);
    Some(serde_json::from_str(json).unwrap_or_else(|error| {
        panic!("Wolfram multilayer-sphere output was not JSON: {error}\nstdout:\n{stdout}")
    }))
}

fn solve_case(cells: usize, time_step_s: f64) -> ResultSummary {
    assert_eq!(cells % 10, 0);
    let mesh =
        rectangular_mesh_triangles(INNER_RADIUS_M, OUTER_RADIUS_M, 0.0, STRIP_WIDTH_M, cells, 1);
    let space = H1Space::new(PolynomialDegree::P1);
    let layers: Vec<_> = mesh
        .elements
        .iter()
        .map(|element| {
            let radius = element
                .nodes
                .iter()
                .map(|&node| mesh.nodes[node].x)
                .sum::<f64>()
                / element.nodes.len() as f64;
            layer_at(radius)
        })
        .collect();
    let conduction_forms: Vec<_> = layers
        .iter()
        .map(|&layer| SphericalConductionForm {
            conductivity: CONDUCTIVITY[layer],
        })
        .collect();
    let capacity_forms: Vec<_> = layers
        .iter()
        .map(|&layer| SphericalCapacityForm {
            volumetric_heat_capacity: VOLUMETRIC_HEAT_CAPACITY[layer],
        })
        .collect();
    let conduction =
        assemble_bilinear_per_element(&conduction_forms, &space, &space, &mesh, 3).to_csr();
    let capacity =
        assemble_bilinear_per_element(&capacity_forms, &space, &space, &mesh, 3).to_csr();

    let inner_dofs: Vec<_> = mesh
        .nodes
        .iter()
        .enumerate()
        .filter_map(|(index, node)| ((node.x - INNER_RADIUS_M).abs() < 1.0e-12).then_some(index))
        .collect();
    let mut boundary_load = vec![0.0; mesh.num_nodes()];
    let average_flux = angular_average_flux();
    for boundary in mesh
        .boundaries
        .iter()
        .filter(|boundary| boundary.marker == 2)
    {
        assert_eq!(boundary.nodes.len(), 2);
        let first = mesh.nodes[boundary.nodes[0]];
        let second = mesh.nodes[boundary.nodes[1]];
        let edge_length = ((second.x - first.x).powi(2) + (second.y - first.y).powi(2)).sqrt();
        let nodal_load =
            average_flux * 4.0 * PI * OUTER_RADIUS_M.powi(2) * edge_length / (2.0 * STRIP_WIDTH_M);
        boundary_load[boundary.nodes[0]] += nodal_load;
        boundary_load[boundary.nodes[1]] += nodal_load;
    }
    let input_power = boundary_load.iter().sum::<f64>();
    let expected_input_power = average_flux * 4.0 * PI * OUTER_RADIUS_M.powi(2);
    assert!((input_power - expected_input_power).abs() / expected_input_power < 1.0e-13);

    let mut effective = add_scaled(&capacity, 1.0, &conduction, time_step_s);
    impose_zero_rows(&mut effective, &inner_dofs);
    let factorization = SparseLuFactorization::factorize(&effective)
        .expect("multilayer spherical heat operator must factorize");
    let mut temperature = vec![0.0; mesh.num_nodes()];
    let mut removed_heat = 0.0;
    let mut max_free_residual = 0.0_f64;
    let mut samples = [[0.0; 3]; 6];
    let step_count = (FINAL_TIME_S / time_step_s).round() as usize;
    for step in 1..=step_count {
        let previous = temperature.clone();
        let mut rhs = capacity.matvec(&previous);
        for (value, load) in rhs.iter_mut().zip(&boundary_load) {
            *value += time_step_s * load;
        }
        for &dof in &inner_dofs {
            rhs[dof] = 0.0;
        }
        temperature = factorization.apply(&rhs);

        let capacity_increment = capacity.matvec(
            &temperature
                .iter()
                .zip(&previous)
                .map(|(new, old)| new - old)
                .collect::<Vec<_>>(),
        );
        let conductive_step = conduction
            .matvec(&temperature)
            .into_iter()
            .map(|value| time_step_s * value)
            .collect::<Vec<_>>();
        let raw_residual: Vec<_> = capacity_increment
            .iter()
            .zip(&conductive_step)
            .zip(&boundary_load)
            .map(|((capacity_term, conduction_term), load)| {
                capacity_term + conduction_term - time_step_s * load
            })
            .collect();
        removed_heat -= inner_dofs.iter().map(|&dof| raw_residual[dof]).sum::<f64>();
        let free_residual = raw_residual
            .iter()
            .enumerate()
            .filter(|(index, _)| !inner_dofs.contains(index))
            .map(|(_, value)| value * value)
            .sum::<f64>()
            .sqrt();
        max_free_residual = max_free_residual.max(free_residual);

        let time = step as f64 * time_step_s;
        for (sample_time, sample_row) in SAMPLE_TIMES_S.iter().zip(samples.iter_mut()) {
            if (time - sample_time).abs() < 0.5 * time_step_s {
                for (slot, radius) in sample_row.iter_mut().zip(SAMPLE_RADII_M) {
                    *slot = radial_column_average(&mesh, &temperature, radius);
                }
            }
        }
    }

    let spacing = (OUTER_RADIUS_M - INNER_RADIUS_M) / cells as f64;
    let mut interface_flux_jump = [0.0; 2];
    for (slot, &radius) in interface_flux_jump.iter_mut().zip(&INTERFACE_RADII_M) {
        let center = radial_column_average(&mesh, &temperature, radius);
        let left = radial_column_average(&mesh, &temperature, radius - spacing);
        let right = radial_column_average(&mesh, &temperature, radius + spacing);
        let left_layer = layer_at(radius - 0.5 * spacing);
        let right_layer = layer_at(radius + 0.5 * spacing);
        let left_flux = -CONDUCTIVITY[left_layer] * (center - left) / spacing;
        let right_flux = -CONDUCTIVITY[right_layer] * (right - center) / spacing;
        *slot = (left_flux - right_flux).abs() / left_flux.abs().max(right_flux.abs()).max(1.0e-14);
    }

    let stored_heat = capacity.matvec(&temperature).iter().sum::<f64>();
    let input_heat = input_power * FINAL_TIME_S;
    let relative_energy_balance = (stored_heat + removed_heat - input_heat).abs() / input_heat;
    let transverse_spread = (0..=cells)
        .map(|column| {
            let radius = INNER_RADIUS_M + column as f64 * spacing;
            let values: Vec<_> = mesh
                .nodes
                .iter()
                .enumerate()
                .filter_map(|(index, node)| {
                    ((node.x - radius).abs() < 1.0e-10).then_some(temperature[index])
                })
                .collect();
            assert_eq!(values.len(), 2);
            (values[0] - values[1]).abs()
        })
        .fold(0.0, f64::max);

    ResultSummary {
        samples,
        interface_flux_jump,
        relative_energy_balance,
        relative_free_residual: max_free_residual / (time_step_s * input_power),
        transverse_spread,
    }
}

#[test]
fn sonium_reproduces_spherical_mean_of_wolfram_multilayer_heat_case() {
    let coarse = solve_case(100, 0.1);
    let fine = solve_case(200, 0.05);
    eprintln!("Wolfram multilayer spherical mean: coarse={coarse:?}, fine={fine:?}");

    assert!(fine.relative_energy_balance < 1.0e-11);
    assert!(fine.relative_free_residual < 1.0e-11);
    assert!(fine.transverse_spread < 0.01);
    assert!(fine.transverse_spread < 0.6 * coarse.transverse_spread);
    assert!(fine.interface_flux_jump.iter().all(|jump| *jump < 0.03));
    for (time, (fine_row, coarse_row)) in SAMPLE_TIMES_S
        .iter()
        .zip(fine.samples.iter().zip(&coarse.samples))
    {
        for (radius, (&fine_value, &coarse_value)) in
            SAMPLE_RADII_M.iter().zip(fine_row.iter().zip(coarse_row))
        {
            assert!(fine_value > 0.0, "T({time}, {radius}) must be positive");
            assert!(
                (fine_value - coarse_value).abs() / fine_value.max(1.0e-12) < 0.015,
                "T({time}, {radius}): fine={fine_value:.9}, coarse={coarse_value:.9}"
            );
        }
    }

    if let Some(reference) = wolfram_reference() {
        assert!(
            (reference.spherical_average_flux - angular_average_flux()).abs()
                / angular_average_flux()
                < 1.0e-12
        );
        for (time, (sonium_row, wolfram_row)) in SAMPLE_TIMES_S.iter().zip(
            fine.samples
                .iter()
                .zip(&reference.spherical_average_samples),
        ) {
            for (radius, (&sonium, &wolfram)) in SAMPLE_RADII_M
                .iter()
                .zip(sonium_row.iter().zip(wolfram_row))
            {
                assert!(
                    (sonium - wolfram).abs() / wolfram.abs().max(1.0e-12) < 0.02,
                    "T({time}, {radius}): Sonium={sonium:.9}, Wolfram={wolfram:.9}"
                );
            }
        }
    }
}
