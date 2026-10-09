//! Axisymmetric diagnostic derived from Wolfram's first
//! `Magnetostatics for Permanent Magnets` model.
//!
//! The material and magnet geometry are literal: a radius-0.1 m, length-0.2 m
//! uniformly magnetized cylinder with `M_z = 1.2e6 A/m`, surrounded by vacuum
//! out to a nominal 2 m far boundary. The magnet interface is aligned exactly
//! with the meridional mesh. The test uses a rectangular `(r,z)` truncation,
//! rather than Wolfram's spherical outer boundary, and is therefore kept as a
//! diagnostic until Sonium has a conforming half-disk/sphere mesh path.

use serde::Deserialize;
use sonium_fem::{
    assembly::{assemble_bilinear_per_element, assemble_linear_per_element},
    basis::PolynomialDegree,
    forms::{BilinearForm, LinearForm, LocalMatrix, LocalVector},
    mesh::{ElementType, Mesh, Point},
    spaces::{BasisTabulation, H1Space},
};
use sonium_qa::subprocess::WolframCommandExt;
use sonium_solvers::{CsrMatrix, SparseLuFactorization, traits::Preconditioner};
use std::{env, f64::consts::PI, path::PathBuf, process::Command};

const MU_0: f64 = 4.0 * PI * 1.0e-7;
const MAGNET_RADIUS_M: f64 = 0.1;
const MAGNET_HALF_LENGTH_M: f64 = 0.1;
const MAGNETIZATION_A_M: f64 = 1.2e6;
const OUTER_EXTENT_M: f64 = 2.0;
const SAMPLE_Z_M: [f64; 5] = [0.15, 0.2, 0.3, 0.5, 1.0];

#[derive(Clone, Copy)]
struct AxisymmetricPermeabilityForm {
    permeability: f64,
}

impl BilinearForm<f64> for AxisymmetricPermeabilityForm {
    fn integrate_point(
        &self,
        test: &BasisTabulation,
        trial: &BasisTabulation,
        weight: f64,
        local_matrix: &mut LocalMatrix<f64>,
    ) {
        let radius = test.physical_coords[0].max(0.0);
        let measure = weight * test.det_j * 2.0 * PI * radius;
        for i in 0..test.n_dofs {
            for j in 0..trial.n_dofs {
                let gradient_dot = (0..test.spatial_dim)
                    .map(|dimension| test.grad(i, dimension) * trial.grad(j, dimension))
                    .sum::<f64>();
                local_matrix.add(i, j, self.permeability * gradient_dot * measure);
            }
        }
    }
}

#[derive(Clone, Copy)]
struct AxisymmetricMagnetSourceForm {
    remanent_flux_density_z: f64,
}

impl LinearForm<f64> for AxisymmetricMagnetSourceForm {
    fn integrate_point(
        &self,
        test: &BasisTabulation,
        weight: f64,
        local_vector: &mut LocalVector<f64>,
    ) {
        let radius = test.physical_coords[0].max(0.0);
        let measure = weight * test.det_j * 2.0 * PI * radius;
        for i in 0..test.n_dofs {
            local_vector.add(i, self.remanent_flux_density_z * test.grad(i, 1) * measure);
        }
    }
}

fn append_segment(coordinates: &mut Vec<f64>, end: f64, cells: usize) {
    let start = *coordinates
        .last()
        .expect("coordinate sequence must be seeded");
    for index in 1..=cells {
        coordinates.push(start + (end - start) * index as f64 / cells as f64);
    }
}

fn coordinates(refinement: usize) -> (Vec<f64>, Vec<f64>) {
    let mut radius = vec![0.0];
    append_segment(&mut radius, 0.2, 10 * refinement);
    append_segment(&mut radius, 0.5, 6 * refinement);
    append_segment(&mut radius, 1.0, 5 * refinement);
    append_segment(&mut radius, 2.0, 5 * refinement);

    let mut axial = vec![-2.0];
    append_segment(&mut axial, -1.0, 5 * refinement);
    append_segment(&mut axial, -0.5, 5 * refinement);
    append_segment(&mut axial, -0.2, 6 * refinement);
    append_segment(&mut axial, 0.2, 40 * refinement);
    append_segment(&mut axial, 0.5, 6 * refinement);
    append_segment(&mut axial, 1.0, 5 * refinement);
    append_segment(&mut axial, 2.0, 5 * refinement);
    (radius, axial)
}

fn variable_rectangular_mesh(radius: &[f64], axial: &[f64]) -> Mesh {
    let mut mesh = Mesh::new(2);
    for &z in axial {
        for &r in radius {
            mesh.add_node(Point::new_2d(r, z));
        }
    }
    let columns = radius.len();
    for row in 0..axial.len() - 1 {
        for column in 0..radius.len() - 1 {
            let n00 = row * columns + column;
            let n10 = n00 + 1;
            let n01 = n00 + columns;
            let n11 = n01 + 1;
            mesh.add_element(ElementType::Triangle, vec![n00, n10, n11]);
            mesh.add_element(ElementType::Triangle, vec![n00, n11, n01]);
        }
    }
    mesh.detect_boundaries();
    mesh
}

fn impose_zero_rows(matrix: &mut CsrMatrix<f64>, rhs: &mut [f64], dofs: &[usize]) {
    for &dof in dofs {
        for entry in matrix.row_ptrs[dof]..matrix.row_ptrs[dof + 1] {
            matrix.values[entry] = if matrix.col_indices[entry] == dof {
                1.0
            } else {
                0.0
            };
        }
        rhs[dof] = 0.0;
    }
}

fn finite_cylinder_axial_flux_density(z: f64) -> f64 {
    let upper = (z + MAGNET_HALF_LENGTH_M)
        / (MAGNET_RADIUS_M.powi(2) + (z + MAGNET_HALF_LENGTH_M).powi(2)).sqrt();
    let lower = (z - MAGNET_HALF_LENGTH_M)
        / (MAGNET_RADIUS_M.powi(2) + (z - MAGNET_HALF_LENGTH_M).powi(2)).sqrt();
    0.5 * MU_0 * MAGNETIZATION_A_M * (upper - lower)
}

fn axis_derivative(values: &[f64], axial: &[f64], columns: usize, z: f64) -> f64 {
    let row = axial
        .iter()
        .position(|value| (*value - z).abs() < 1.0e-12)
        .expect("sample must align with an axial mesh row");
    assert!(row > 0 && row + 1 < axial.len());
    let h_left = axial[row] - axial[row - 1];
    let h_right = axial[row + 1] - axial[row];
    let previous = values[(row - 1) * columns];
    let center = values[row * columns];
    let next = values[(row + 1) * columns];
    -h_right / (h_left * (h_left + h_right)) * previous
        + (h_right - h_left) / (h_left * h_right) * center
        + h_left / (h_right * (h_left + h_right)) * next
}

#[derive(Debug)]
struct SoniumResult {
    axial_flux_density_t: [f64; 5],
    relative_residual: f64,
    relative_source_sum: f64,
    node_count: usize,
}

fn solve_case(refinement: usize) -> SoniumResult {
    let (radius, axial) = coordinates(refinement);
    let mesh = variable_rectangular_mesh(&radius, &axial);
    let space = H1Space::new(PolynomialDegree::P1);
    let in_magnet: Vec<_> = mesh
        .elements
        .iter()
        .map(|element| {
            let r = element
                .nodes
                .iter()
                .map(|&node| mesh.nodes[node].x)
                .sum::<f64>()
                / element.nodes.len() as f64;
            let z = element
                .nodes
                .iter()
                .map(|&node| mesh.nodes[node].y)
                .sum::<f64>()
                / element.nodes.len() as f64;
            r < MAGNET_RADIUS_M && z.abs() < MAGNET_HALF_LENGTH_M
        })
        .collect();
    assert!(in_magnet.iter().any(|inside| *inside));
    let permeability_forms =
        vec![AxisymmetricPermeabilityForm { permeability: MU_0 }; mesh.num_elements()];
    let source_forms: Vec<_> = in_magnet
        .iter()
        .map(|&inside| AxisymmetricMagnetSourceForm {
            remanent_flux_density_z: if inside {
                MU_0 * MAGNETIZATION_A_M
            } else {
                0.0
            },
        })
        .collect();
    let mut matrix =
        assemble_bilinear_per_element(&permeability_forms, &space, &space, &mesh, 2).to_csr();
    let mut rhs = assemble_linear_per_element(&source_forms, &space, &mesh, 2);
    let rhs_norm = rhs.iter().map(|value| value * value).sum::<f64>().sqrt();
    let source_sum = rhs.iter().sum::<f64>().abs();
    let outer_dofs: Vec<_> = mesh
        .nodes
        .iter()
        .enumerate()
        .filter_map(|(index, point)| {
            ((point.x - OUTER_EXTENT_M).abs() < 1.0e-12
                || (point.y.abs() - OUTER_EXTENT_M).abs() < 1.0e-12)
                .then_some(index)
        })
        .collect();
    impose_zero_rows(&mut matrix, &mut rhs, &outer_dofs);
    let factorization = SparseLuFactorization::factorize(&matrix)
        .expect("axisymmetric permanent-magnet system must factorize");
    let potential = factorization.apply(&rhs);
    let residual = matrix
        .matvec(&potential)
        .iter()
        .zip(&rhs)
        .map(|(lhs, rhs_value)| (lhs - rhs_value).powi(2))
        .sum::<f64>()
        .sqrt();
    let axial_flux_density_t =
        SAMPLE_Z_M.map(|z| -MU_0 * axis_derivative(&potential, &axial, radius.len(), z));

    SoniumResult {
        axial_flux_density_t,
        relative_residual: residual / rhs_norm,
        relative_source_sum: source_sum / rhs_norm,
        node_count: mesh.num_nodes(),
    }
}

#[derive(Debug, Deserialize)]
struct WolframReference {
    schema_version: u32,
    sample_points_m: [[f64; 3]; 5],
    flux_density_t: [[f64; 3]; 5],
    infinite_domain_axial_flux_density_t: [f64; 5],
}

fn wolfram_reference() -> Option<WolframReference> {
    let executable = env::var_os("WOLFRAMSCRIPT")?;
    let script =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("wolfram/permanent_magnet_cylinder.wls");
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
        "Wolfram permanent-magnet reference failed:\n{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    let json = stdout
        .lines()
        .rev()
        .find(|line| line.trim_start().starts_with('{'))
        .unwrap_or(&stdout);
    Some(serde_json::from_str(json).unwrap_or_else(|error| {
        panic!("Wolfram permanent-magnet output was not JSON: {error}\nstdout:\n{stdout}")
    }))
}

#[test]
fn sonium_checks_wolfram_permanent_magnet_axisymmetric_reduction() {
    let coarse = solve_case(1);
    let fine = solve_case(2);
    let analytical = SAMPLE_Z_M.map(finite_cylinder_axial_flux_density);
    eprintln!(
        "Wolfram permanent magnet: coarse={coarse:?}, fine={fine:?}, analytical={analytical:?}"
    );

    assert!(fine.relative_residual < 1.0e-11);
    assert!(fine.relative_source_sum < 1.0e-12);
    assert!(fine.node_count > coarse.node_count);
    for (sample, ((&fine_value, &coarse_value), &exact)) in fine
        .axial_flux_density_t
        .iter()
        .zip(&coarse.axial_flux_density_t)
        .zip(&analytical)
        .enumerate()
    {
        assert!(fine_value > 0.0, "sample {sample} has wrong field sign");
        assert!(
            (fine_value - coarse_value).abs() / fine_value.abs() < 0.08,
            "sample {sample}: fine={fine_value:.9e}, coarse={coarse_value:.9e}"
        );
        assert!(
            (fine_value - exact).abs() / exact.abs() < 0.12,
            "sample {sample}: box FEM={fine_value:.9e}, infinite-domain={exact:.9e}"
        );
    }

    if let Some(reference) = wolfram_reference() {
        assert_eq!(reference.schema_version, 1);
        for (sample, ((point, field), &infinite)) in reference
            .sample_points_m
            .iter()
            .zip(&reference.flux_density_t)
            .zip(&reference.infinite_domain_axial_flux_density_t)
            .enumerate()
        {
            assert!((point[2] - SAMPLE_Z_M[sample]).abs() < 1.0e-12);
            assert!((infinite - analytical[sample]).abs() / analytical[sample] < 1.0e-12);
            assert!(
                (fine.axial_flux_density_t[sample] - field[2]).abs() / field[2].abs().max(1.0e-12)
                    < 0.15,
                "sample {sample}: Sonium={}, Wolfram={}",
                fine.axial_flux_density_t[sample],
                field[2]
            );
        }
    }
}
