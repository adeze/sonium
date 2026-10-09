//! Literal port of the straight current-carrying wire example from Wolfram's
//! `Quasistatic Magnetic Fields` monograph.
//!
//! The published model is magnetostatic: a radius-0.2 conductor with uniform
//! unit current density in a radius-1 circular truncation, unit relative
//! permeability, and zero magnetic vector potential on the outer circle.  It
//! is a prerequisite field check for the monograph's later frequency-domain
//! examples, not an eddy-current or inductive-heating validation.

use serde::Deserialize;
use sonium_fem::{
    assembly::{assemble_bilinear, assemble_linear},
    basis::PolynomialDegree,
    boundary::generic::{FemSystem, apply_homogeneous_dirichlet},
    forms::{LinearForm, LocalVector, WeightedLaplacianForm},
    mesh::generators::circular_mesh_triangles,
    spaces::{BasisTabulation, FunctionSpace, H1Space},
};
use sonium_qa::subprocess::WolframCommandExt;
use sonium_solvers::{GmresConfig, gmres};
use std::{env, f64::consts::PI, path::PathBuf, process::Command};

const MU_0: f64 = 4.0 * PI * 1.0e-7;
const WIRE_RADIUS_M: f64 = 0.2;
const OUTER_RADIUS_M: f64 = 1.0;
const CURRENT_DENSITY_A_M2: f64 = 1.0;
const SAMPLE_RADII_M: [f64; 6] = [0.0, 0.1, 0.2, 0.4, 0.6, 0.8];

struct CurrentDensityForm;

impl LinearForm<f64> for CurrentDensityForm {
    fn integrate_point(
        &self,
        test: &BasisTabulation,
        weight: f64,
        local_vector: &mut LocalVector<f64>,
    ) {
        let radius = test.physical_coords[0].hypot(test.physical_coords[1]);
        let current_density = if radius <= WIRE_RADIUS_M {
            CURRENT_DENSITY_A_M2
        } else {
            0.0
        };
        let integration_weight = weight * test.det_j;
        for local_dof in 0..test.n_dofs {
            local_vector.add(
                local_dof,
                current_density * test.value(local_dof) * integration_weight,
            );
        }
    }
}

#[derive(Debug)]
struct SoniumResult {
    node_count: usize,
    potential_t_m: [f64; SAMPLE_RADII_M.len()],
    relative_field_l2_error: f64,
    relative_residual: f64,
}

fn exact_potential(radius_m: f64) -> f64 {
    let exterior_scale = MU_0 * CURRENT_DENSITY_A_M2 * WIRE_RADIUS_M.powi(2) / 2.0;
    if radius_m <= WIRE_RADIUS_M {
        MU_0 * CURRENT_DENSITY_A_M2 * (WIRE_RADIUS_M.powi(2) - radius_m.powi(2)) / 4.0
            + exterior_scale * (OUTER_RADIUS_M / WIRE_RADIUS_M).ln()
    } else {
        exterior_scale * (OUTER_RADIUS_M / radius_m).ln()
    }
}

fn exact_flux_density(radius_m: f64) -> f64 {
    if radius_m <= WIRE_RADIUS_M {
        MU_0 * CURRENT_DENSITY_A_M2 * radius_m / 2.0
    } else {
        MU_0 * CURRENT_DENSITY_A_M2 * WIRE_RADIUS_M.powi(2) / (2.0 * radius_m)
    }
}

fn solve_case(radial_cells: usize, angular_cells: usize) -> SoniumResult {
    assert_eq!(
        radial_cells % 5,
        0,
        "wire interface must align with a radial ring"
    );
    assert_eq!(angular_cells % 4, 0);
    let mesh = circular_mesh_triangles(0.0, 0.0, OUTER_RADIUS_M, radial_cells, angular_cells);
    let space = H1Space::new(PolynomialDegree::P1);
    let stiffness_form = WeightedLaplacianForm {
        coefficient: 1.0 / MU_0,
    };
    let matrix = assemble_bilinear(&stiffness_form, &space, &space, &mesh, 2);
    let rhs = assemble_linear(&CurrentDensityForm, &space, &mesh, 2);
    let mut system = FemSystem::new(matrix, rhs);
    apply_homogeneous_dirichlet(&mut system, &mesh, &space, 1);

    let csr = system.matrix.to_csr();
    let solved = gmres(
        &csr,
        &system.rhs,
        &GmresConfig {
            max_iterations: 4_000,
            restart: 80,
            // The QA acceptance gate below is 1e-9.  Requesting 1e-12 makes
            // the fine, highly refined Poisson system spend many additional
            // GMRES iterations chasing roundoff without improving the field
            // comparison; its converged true residual is already around 1e-12.
            tolerance: 1.0e-10,
            ..GmresConfig::default()
        },
    );
    assert!(
        solved.converged,
        "GMRES failed: residual={}",
        solved.residual
    );

    let potential_t_m = SAMPLE_RADII_M.map(|radius| {
        let ring = ((radius / OUTER_RADIUS_M) * radial_cells as f64).round() as usize;
        if ring == 0 {
            solved.x[0]
        } else {
            // The circular generator stores the center first, then each ring
            // consecutively; angle zero is the first node on every ring.
            solved.x[1 + (ring - 1) * angular_cells]
        }
    });

    let mut field_error_squared = 0.0;
    let mut field_reference_squared = 0.0;
    for element_index in 0..mesh.num_elements() {
        let tab = space.tabulate(&mesh, element_index, 1.0 / 3.0, 1.0 / 3.0, 0.0);
        let dofs = space.local_to_global(&mesh, element_index);
        let mut da_dx = 0.0;
        let mut da_dy = 0.0;
        for (local_dof, &global_dof) in dofs.iter().enumerate() {
            da_dx += solved.x[global_dof] * tab.grad(local_dof, 0);
            da_dy += solved.x[global_dof] * tab.grad(local_dof, 1);
        }
        let x = tab.physical_coords[0];
        let y = tab.physical_coords[1];
        let radius = x.hypot(y);
        if radius <= 1.0e-14 {
            continue;
        }
        let numerical_b_theta = da_dy * (-y / radius) - da_dx * (x / radius);
        let reference_b_theta = exact_flux_density(radius);
        let area_weight = 0.5 * tab.det_j;
        field_error_squared += (numerical_b_theta - reference_b_theta).powi(2) * area_weight;
        field_reference_squared += reference_b_theta.powi(2) * area_weight;
    }

    let lhs = csr.matvec(&solved.x);
    let residual_norm = lhs
        .iter()
        .zip(&system.rhs)
        .map(|(left, right)| (left - right).powi(2))
        .sum::<f64>()
        .sqrt();
    let rhs_norm = system
        .rhs
        .iter()
        .map(|value| value * value)
        .sum::<f64>()
        .sqrt();

    SoniumResult {
        node_count: mesh.num_nodes(),
        potential_t_m,
        relative_field_l2_error: (field_error_squared / field_reference_squared).sqrt(),
        relative_residual: residual_norm / rhs_norm,
    }
}

#[derive(Debug, Deserialize)]
struct WolframReference {
    schema_version: u32,
    wire_radius_m: f64,
    outer_radius_m: f64,
    current_density_a_m2: f64,
    sample_radii_m: [f64; SAMPLE_RADII_M.len()],
    vector_potential_t_m: [f64; SAMPLE_RADII_M.len()],
    flux_density_t: [f64; SAMPLE_RADII_M.len()],
}

fn wolfram_reference() -> Option<WolframReference> {
    let executable = env::var_os("WOLFRAMSCRIPT")?;
    let script =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("wolfram/quasistatic_straight_wire.wls");
    let output = match Command::new(&executable)
        .arg("-file")
        .arg(script)
        .wolfram_output()
    {
        Ok(output) => output,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            eprintln!("Wolfram executable is unavailable; using the analytical reference");
            return None;
        }
        Err(error) => panic!("failed to launch {executable:?}: {error}"),
    };
    assert!(
        output.status.success(),
        "Wolfram straight-wire reference failed:\n{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    let json = stdout
        .lines()
        .rev()
        .find(|line| line.trim_start().starts_with('{'))
        .unwrap_or(&stdout);
    Some(serde_json::from_str(json).unwrap_or_else(|error| {
        panic!("Wolfram straight-wire output was not JSON: {error}\nstdout:\n{stdout}")
    }))
}

#[test]
fn reproduces_wolfram_quasistatic_straight_wire() {
    let coarse = solve_case(20, 80);
    let fine = solve_case(40, 160);
    let exact_potential_t_m = SAMPLE_RADII_M.map(exact_potential);

    println!(
        "straight wire: coarse={} nodes, field L2={:.6e}; fine={} nodes, field L2={:.6e}, residual={:.3e}",
        coarse.node_count,
        coarse.relative_field_l2_error,
        fine.node_count,
        fine.relative_field_l2_error,
        fine.relative_residual
    );
    println!("straight-wire potential samples: {:?}", fine.potential_t_m);

    assert!(fine.node_count > coarse.node_count);
    assert!(fine.relative_residual < 1.0e-9);
    assert!(fine.relative_field_l2_error < coarse.relative_field_l2_error);
    assert!(fine.relative_field_l2_error < 0.04);

    for (sample, ((&coarse_value, &fine_value), &exact_value)) in coarse
        .potential_t_m
        .iter()
        .zip(&fine.potential_t_m)
        .zip(&exact_potential_t_m)
        .enumerate()
    {
        let scale = exact_potential_t_m[0];
        assert!(
            (fine_value - exact_value).abs() / scale < 2.0e-3,
            "sample {sample}: Sonium={fine_value:.12e}, exact={exact_value:.12e}"
        );
        assert!(
            (fine_value - exact_value).abs() <= (coarse_value - exact_value).abs() + 1.0e-14,
            "sample {sample} did not converge: coarse={coarse_value:.12e}, fine={fine_value:.12e}, exact={exact_value:.12e}"
        );
    }

    if let Some(reference) = wolfram_reference() {
        assert_eq!(reference.schema_version, 1);
        assert!((reference.wire_radius_m - WIRE_RADIUS_M).abs() < 1.0e-14);
        assert!((reference.outer_radius_m - OUTER_RADIUS_M).abs() < 1.0e-14);
        assert!((reference.current_density_a_m2 - CURRENT_DENSITY_A_M2).abs() < 1.0e-14);
        assert_eq!(reference.sample_radii_m, SAMPLE_RADII_M);
        for (sample, &radius) in SAMPLE_RADII_M.iter().enumerate() {
            let potential_scale = exact_potential_t_m[0];
            assert!(
                (fine.potential_t_m[sample] - reference.vector_potential_t_m[sample]).abs()
                    / potential_scale
                    < 5.0e-3
            );
            let exact_b = exact_flux_density(radius);
            assert!(
                (reference.flux_density_t[sample] - exact_b).abs()
                    / exact_b.abs().max(MU_0 * 1.0e-8)
                    < 0.03
            );
        }
    }
}
