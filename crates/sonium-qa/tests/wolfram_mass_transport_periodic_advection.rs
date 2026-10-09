//! Literal coefficients, domain, initial condition, and periodic boundary from
//! Wolfram's one-dimensional pure-advection mass-transport example.
//!
//! Sonium currently has neither a one-dimensional mesh nor a reusable periodic
//! constraint API.  This adapter therefore extrudes the line to a thin 2-D
//! strip and identifies matching left/right P1 nodal degrees of freedom before
//! time integration.

use serde::Deserialize;
use sonium_fem::{
    assembly::assemble_bilinear,
    basis::PolynomialDegree,
    mesh::rectangular_mesh_triangles,
    physics::fluid::transport::{TransportAdvectionForm, TransportCapacityForm, TransportMaterial},
    spaces::{FunctionSpace, H1Space},
};
use sonium_qa::subprocess::WolframCommandExt;
use sonium_solvers::{CsrMatrix, SparseLuFactorization, TripletMatrix};
use std::{env, path::PathBuf, process::Command};

const LEFT_M: f64 = 0.0;
const RIGHT_M: f64 = 1.0;
const STRIP_WIDTH_M: f64 = 0.01;
const VELOCITY_M_S: f64 = 0.02;
const FINAL_TIME_S: f64 = 100.0;
const TIME_STEP_S: f64 = 0.125;
const INITIAL_CENTER_M: f64 = 0.5;
const INITIAL_WIDTH_M: f64 = 0.15;
const INITIAL_AMPLITUDE: f64 = 1000.0;
const SAMPLE_X_M: [f64; 5] = [0.0, 0.25, 0.5, 0.75, 1.0];

fn periodic_displacement(x: f64, center: f64) -> f64 {
    let length = RIGHT_M - LEFT_M;
    let mut displacement = x - center;
    displacement -= length * (displacement / length).round();
    displacement
}

fn exact_concentration(x: f64, time: f64) -> f64 {
    let length = RIGHT_M - LEFT_M;
    let center = LEFT_M + (INITIAL_CENTER_M - LEFT_M + VELOCITY_M_S * time).rem_euclid(length);
    let displacement = periodic_displacement(x, center);
    INITIAL_AMPLITUDE * (-(displacement / INITIAL_WIDTH_M).powi(2)).exp()
}

fn periodic_dof_map(mesh: &sonium_fem::mesh::Mesh) -> (Vec<usize>, Vec<usize>) {
    let mut representatives: Vec<usize> = (0..mesh.nodes.len()).collect();
    for (right_index, right_node) in mesh.nodes.iter().enumerate() {
        if (right_node.x - RIGHT_M).abs() > 1.0e-12 {
            continue;
        }
        let left_index = mesh
            .nodes
            .iter()
            .position(|node| {
                (node.x - LEFT_M).abs() < 1.0e-12 && (node.y - right_node.y).abs() < 1.0e-12
            })
            .expect("every right-boundary node must have a periodic left partner");
        representatives[right_index] = left_index;
    }

    let mut representative_to_reduced = vec![usize::MAX; mesh.nodes.len()];
    let mut reduced_representatives = Vec::new();
    let mut full_to_reduced = vec![usize::MAX; mesh.nodes.len()];
    for (full, &representative) in representatives.iter().enumerate() {
        let reduced = if representative_to_reduced[representative] == usize::MAX {
            let reduced = reduced_representatives.len();
            representative_to_reduced[representative] = reduced;
            reduced_representatives.push(representative);
            reduced
        } else {
            representative_to_reduced[representative]
        };
        full_to_reduced[full] = reduced;
    }
    (full_to_reduced, reduced_representatives)
}

fn reduce_periodic(matrix: &CsrMatrix<f64>, map: &[usize], reduced_size: usize) -> CsrMatrix<f64> {
    let mut reduced = TripletMatrix::new(reduced_size, reduced_size);
    for row in 0..matrix.num_rows {
        for entry in matrix.row_ptrs[row]..matrix.row_ptrs[row + 1] {
            reduced.push(
                map[row],
                map[matrix.col_indices[entry]],
                matrix.values[entry],
            );
        }
    }
    reduced.to_csr()
}

fn linear_combination(
    first: &CsrMatrix<f64>,
    first_scale: f64,
    second: &CsrMatrix<f64>,
    second_scale: f64,
) -> CsrMatrix<f64> {
    assert_eq!(first.num_rows, second.num_rows);
    assert_eq!(first.num_cols, second.num_cols);
    let mut result = TripletMatrix::new(first.num_rows, first.num_cols);
    for row in 0..first.num_rows {
        for entry in first.row_ptrs[row]..first.row_ptrs[row + 1] {
            result.push(
                row,
                first.col_indices[entry],
                first_scale * first.values[entry],
            );
        }
        for entry in second.row_ptrs[row]..second.row_ptrs[row + 1] {
            result.push(
                row,
                second.col_indices[entry],
                second_scale * second.values[entry],
            );
        }
    }
    result.to_csr()
}

#[derive(Debug)]
struct SoniumResult {
    samples: Vec<f64>,
    relative_mass_drift: f64,
    peak_x_m: f64,
    peak_value: f64,
    max_return_error: f64,
}

fn sonium_solution() -> SoniumResult {
    let mesh = rectangular_mesh_triangles(LEFT_M, RIGHT_M, 0.0, STRIP_WIDTH_M, 400, 2);
    let space = H1Space::new(PolynomialDegree::P1);
    let material = TransportMaterial::with_capacity(0.0, [VELOCITY_M_S, 0.0, 0.0], 0.0, 1.0);
    let full_mass = assemble_bilinear(
        &TransportCapacityForm::new(&material),
        &space,
        &space,
        &mesh,
        2,
    )
    .to_csr();
    let full_advection = assemble_bilinear(
        &TransportAdvectionForm::new(&material),
        &space,
        &space,
        &mesh,
        2,
    )
    .to_csr();
    assert_eq!(space.num_global_dofs(&mesh), mesh.nodes.len());

    let (map, representatives) = periodic_dof_map(&mesh);
    let mass = reduce_periodic(&full_mass, &map, representatives.len());
    let advection = reduce_periodic(&full_advection, &map, representatives.len());
    let left = linear_combination(&mass, 1.0, &advection, 0.5 * TIME_STEP_S);
    let right = linear_combination(&mass, 1.0, &advection, -0.5 * TIME_STEP_S);
    let factorization = SparseLuFactorization::factorize(&left)
        .expect("periodic Crank-Nicolson operator must factorize");

    let mut concentration: Vec<f64> = representatives
        .iter()
        .map(|&node| exact_concentration(mesh.nodes[node].x, 0.0))
        .collect();
    let initial = concentration.clone();
    let initial_mass: f64 = mass.matvec(&concentration).iter().sum();
    let steps = (FINAL_TIME_S / TIME_STEP_S).round() as usize;
    assert!((steps as f64 * TIME_STEP_S - FINAL_TIME_S).abs() < f64::EPSILON);
    for _ in 0..steps {
        let rhs = right.matvec(&concentration);
        let result = factorization
            .solve_refined(&left, &rhs, 1.0e-12, 2)
            .expect("reused sparse solve must succeed");
        assert!(result.converged, "sparse refinement failed: {result:?}");
        concentration = result.x;
    }

    let final_mass: f64 = mass.matvec(&concentration).iter().sum();
    let relative_mass_drift = (final_mass - initial_mass).abs() / initial_mass.abs();
    let max_return_error = concentration
        .iter()
        .zip(&initial)
        .map(|(final_value, initial_value)| (final_value - initial_value).abs())
        .fold(0.0_f64, f64::max);
    let (peak_reduced, &peak_value) = concentration
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.total_cmp(b))
        .expect("transport solution is nonempty");
    let peak_x_m = mesh.nodes[representatives[peak_reduced]].x;
    let samples = SAMPLE_X_M
        .iter()
        .map(|&sample_x| {
            let representative = representatives
                .iter()
                .copied()
                .min_by(|&a, &b| {
                    periodic_displacement(mesh.nodes[a].x, sample_x)
                        .abs()
                        .total_cmp(&periodic_displacement(mesh.nodes[b].x, sample_x).abs())
                })
                .expect("transport mesh is nonempty");
            concentration[map[representative]]
        })
        .collect();

    SoniumResult {
        samples,
        relative_mass_drift,
        peak_x_m,
        peak_value,
        max_return_error,
    }
}

#[derive(Deserialize)]
struct WolframReference {
    schema_version: u32,
    velocity_m_s: f64,
    final_time_s: f64,
    sample_x_m: Vec<f64>,
    concentration: Vec<f64>,
}

fn wolfram_reference() -> Option<WolframReference> {
    let executable = env::var_os("WOLFRAMSCRIPT")?;
    let script = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("wolfram/mass_transport_periodic_advection.wls");
    let output = match Command::new(&executable)
        .arg("-file")
        .arg(script)
        .wolfram_output()
    {
        Ok(output) => output,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            eprintln!(
                "Wolfram Engine not available; using exact periodic-advection reference only"
            );
            return None;
        }
        Err(error) => panic!("failed to launch {executable:?}: {error}"),
    };
    assert!(
        output.status.success(),
        "Wolfram periodic-advection reference failed:\n{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    let json = stdout
        .lines()
        .rev()
        .find(|line| line.trim_start().starts_with('{'))
        .unwrap_or(&stdout);
    Some(serde_json::from_str(json).unwrap_or_else(|error| {
        panic!("Wolfram periodic-advection output was not JSON: {error}\nstdout:\n{stdout}")
    }))
}

#[test]
fn sonium_reproduces_wolfram_periodic_pure_advection() {
    let result = sonium_solution();
    let exact = SAMPLE_X_M.map(|x| exact_concentration(x, FINAL_TIME_S));
    eprintln!("periodic pure-advection result: {result:?}");

    for (sample, (&got, &want)) in result.samples.iter().zip(&exact).enumerate() {
        assert!(
            (got - want).abs() < 2.0,
            "sample {sample}: concentration={got:.9}, exact={want:.9}"
        );
    }
    assert!(
        result.relative_mass_drift < 1.0e-11,
        "relative discrete-mass drift {:.3e} exceeds 1e-11",
        result.relative_mass_drift
    );
    assert!(
        result.max_return_error < 2.0,
        "two-circuit return error {:.6} exceeds 2 concentration units",
        result.max_return_error
    );
    assert!((result.peak_x_m - INITIAL_CENTER_M).abs() <= 0.005);
    assert!((result.peak_value - INITIAL_AMPLITUDE).abs() < 2.0);

    if let Some(reference) = wolfram_reference() {
        assert_eq!(reference.schema_version, 1);
        assert_eq!(reference.velocity_m_s, VELOCITY_M_S);
        assert_eq!(reference.final_time_s, FINAL_TIME_S);
        assert_eq!(reference.sample_x_m, SAMPLE_X_M);
        for (sample, ((&got, &wolfram), &want)) in result
            .samples
            .iter()
            .zip(&reference.concentration)
            .zip(&exact)
            .enumerate()
        {
            assert!(
                (wolfram - want).abs() < 1.0,
                "Wolfram sample {sample}: concentration={wolfram:.9}, exact={want:.9}"
            );
            assert!(
                (got - wolfram).abs() < 2.0,
                "sample {sample}: Sonium={got:.9}, Wolfram={wolfram:.9}"
            );
        }
    }
}
