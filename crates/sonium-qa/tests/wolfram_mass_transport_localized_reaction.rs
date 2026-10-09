//! Literal localized-reaction example from Wolfram's mass-transport
//! monograph: an initially uniform species diffuses on the unit square and is
//! consumed at first order only in a small central rectangular region.
//!
//! Sonium can represent the case through its public region-selective low-level
//! assembler. Its high-level transport assembler still accepts only one
//! constant material for the whole mesh.

use serde::Deserialize;
use sonium_fem::{
    assembly::{assemble_bilinear, assemble_bilinear_where},
    basis::PolynomialDegree,
    mesh::{Mesh, rectangular_mesh_triangles},
    physics::fluid::transport::{
        TransportCapacityForm, TransportDiffusionForm, TransportMaterial, TransportReactionForm,
    },
    spaces::{FunctionSpace, H1Space},
};
use sonium_qa::subprocess::WolframCommandExt;
use sonium_solvers::{CsrMatrix, SparseLuFactorization, TripletMatrix};
use std::{env, path::PathBuf, process::Command};

const DIFFUSIVITY_M2_S: f64 = 0.001;
const REACTION_RATE_S_INV: f64 = 1.0;
const INITIAL_CONCENTRATION: f64 = 1000.0;
const FINAL_TIME_S: f64 = 100.0;
const TIME_STEP_S: f64 = 0.5;
const REACTION_X: [f64; 2] = [0.4, 0.6];
const REACTION_Y: [f64; 2] = [0.45, 0.55];
const SAMPLE_POINTS: [[f64; 2]; 5] = [[0.0, 0.0], [0.2, 0.5], [0.4, 0.5], [0.5, 0.5], [0.8, 0.5]];

fn linear_combination(
    first: &CsrMatrix<f64>,
    first_scale: f64,
    second: &CsrMatrix<f64>,
    second_scale: f64,
) -> CsrMatrix<f64> {
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

fn element_is_reactive(mesh: &Mesh, nodes: &[usize]) -> bool {
    let x = nodes.iter().map(|&node| mesh.nodes[node].x).sum::<f64>() / nodes.len() as f64;
    let y = nodes.iter().map(|&node| mesh.nodes[node].y).sum::<f64>() / nodes.len() as f64;
    (REACTION_X[0] - 1.0e-12..=REACTION_X[1] + 1.0e-12).contains(&x)
        && (REACTION_Y[0] - 1.0e-12..=REACTION_Y[1] + 1.0e-12).contains(&y)
}

fn sample_field(mesh: &Mesh, concentration: &[f64]) -> Vec<f64> {
    SAMPLE_POINTS
        .iter()
        .map(|point| {
            let node = mesh
                .nodes
                .iter()
                .enumerate()
                .min_by(|(_, a), (_, b)| {
                    let a_distance = (a.x - point[0]).powi(2) + (a.y - point[1]).powi(2);
                    let b_distance = (b.x - point[0]).powi(2) + (b.y - point[1]).powi(2);
                    a_distance.total_cmp(&b_distance)
                })
                .map(|(index, _)| index)
                .expect("reaction mesh is nonempty");
            concentration[node]
        })
        .collect()
}

#[derive(Debug)]
struct SoniumResult {
    final_integral: f64,
    samples: Vec<f64>,
    relative_reaction_balance_error: f64,
    symmetry_error: f64,
    minimum_concentration: f64,
}

fn sonium_solution(divisions: usize) -> SoniumResult {
    let mesh = rectangular_mesh_triangles(0.0, 1.0, 0.0, 1.0, divisions, divisions);
    let space = H1Space::new(PolynomialDegree::P1);
    let passive = TransportMaterial::with_capacity(DIFFUSIVITY_M2_S, [0.0; 3], 0.0, 1.0);
    let reactive =
        TransportMaterial::with_capacity(DIFFUSIVITY_M2_S, [0.0; 3], REACTION_RATE_S_INV, 1.0);
    let mass = assemble_bilinear(
        &TransportCapacityForm::new(&passive),
        &space,
        &space,
        &mesh,
        2,
    )
    .to_csr();
    let diffusion = assemble_bilinear(
        &TransportDiffusionForm::new(&passive),
        &space,
        &space,
        &mesh,
        2,
    )
    .to_csr();
    let reaction_form = TransportReactionForm::new(&reactive);
    let reaction =
        assemble_bilinear_where(&reaction_form, &space, &space, &mesh, 2, |_, element| {
            element_is_reactive(&mesh, &element.nodes)
        })
        .to_csr();
    assert_eq!(space.num_global_dofs(&mesh), mesh.nodes.len());

    let operator = linear_combination(&diffusion, 1.0, &reaction, 1.0);
    let left = linear_combination(&mass, 1.0, &operator, 0.5 * TIME_STEP_S);
    let right = linear_combination(&mass, 1.0, &operator, -0.5 * TIME_STEP_S);
    let factorization = SparseLuFactorization::factorize(&left)
        .expect("localized-reaction Crank-Nicolson operator must factorize");

    let mut concentration = vec![INITIAL_CONCENTRATION; mesh.nodes.len()];
    let initial_mass: f64 = mass.matvec(&concentration).iter().sum();
    let mut cumulative_reaction = 0.0;
    let steps = (FINAL_TIME_S / TIME_STEP_S).round() as usize;
    for _ in 0..steps {
        let old_rate: f64 = reaction.matvec(&concentration).iter().sum();
        let rhs = right.matvec(&concentration);
        let result = factorization
            .solve_refined(&left, &rhs, 1.0e-11, 2)
            .expect("reused sparse solve must succeed");
        assert!(result.converged, "sparse refinement failed: {result:?}");
        concentration = result.x;
        let new_rate: f64 = reaction.matvec(&concentration).iter().sum();
        cumulative_reaction += 0.5 * TIME_STEP_S * (old_rate + new_rate);
    }

    let final_mass: f64 = mass.matvec(&concentration).iter().sum();
    let samples = sample_field(&mesh, &concentration);
    let symmetry_error = (samples[1] - samples[4]).abs();
    SoniumResult {
        final_integral: final_mass,
        samples,
        relative_reaction_balance_error: (final_mass + cumulative_reaction - initial_mass).abs()
            / initial_mass,
        symmetry_error,
        minimum_concentration: concentration.iter().copied().fold(f64::INFINITY, f64::min),
    }
}

#[derive(Deserialize)]
struct WolframReference {
    schema_version: u32,
    diffusivity_m2_s: f64,
    reaction_rate_s_inv: f64,
    final_time_s: f64,
    sample_points: Vec<Vec<f64>>,
    final_integral: f64,
    samples: Vec<f64>,
}

fn wolfram_reference() -> Option<WolframReference> {
    let Some(executable) = env::var_os("WOLFRAMSCRIPT") else {
        return Some(
            serde_json::from_str(include_str!(
                "../wolfram/goldens/mass_transport_localized_reaction.json"
            ))
            .expect("parse checked-in Wolfram localized-reaction reference"),
        );
    };
    let script = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("wolfram/mass_transport_localized_reaction.wls");
    let output = match Command::new(&executable)
        .arg("-file")
        .arg(script)
        .wolfram_output()
    {
        Ok(output) => output,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            eprintln!(
                "Wolfram Engine not available; using refinement and reaction-balance oracles"
            );
            return None;
        }
        Err(error) => panic!("failed to launch {executable:?}: {error}"),
    };
    assert!(
        output.status.success(),
        "Wolfram localized-reaction reference failed:\n{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    let json = stdout
        .lines()
        .rev()
        .find(|line| line.trim_start().starts_with('{'))
        .unwrap_or(&stdout);
    Some(serde_json::from_str(json).unwrap_or_else(|error| {
        panic!("Wolfram localized-reaction output was not JSON: {error}\nstdout:\n{stdout}")
    }))
}

#[test]
fn sonium_reproduces_wolfram_localized_mass_reaction() {
    let coarse = sonium_solution(40);
    let fine = sonium_solution(60);
    eprintln!("coarse localized-reaction result: {coarse:?}");
    eprintln!("fine localized-reaction result: {fine:?}");

    assert!(fine.relative_reaction_balance_error < 1.0e-9);
    assert!(fine.minimum_concentration >= -1.0e-8);
    assert!(fine.symmetry_error < 0.1);
    assert!((fine.final_integral - coarse.final_integral).abs() / fine.final_integral < 0.005);
    for (sample, (&got, &coarse_value)) in fine.samples.iter().zip(&coarse.samples).enumerate() {
        assert!(
            (got - coarse_value).abs() / got.abs().max(1.0) < 0.07,
            "sample {sample}: fine={got:.9}, coarse={coarse_value:.9}"
        );
    }

    if let Some(reference) = wolfram_reference() {
        assert_eq!(reference.schema_version, 1);
        assert_eq!(reference.diffusivity_m2_s, DIFFUSIVITY_M2_S);
        assert_eq!(reference.reaction_rate_s_inv, REACTION_RATE_S_INV);
        assert_eq!(reference.final_time_s, FINAL_TIME_S);
        assert_eq!(reference.sample_points, SAMPLE_POINTS);
        assert!(
            (fine.final_integral - reference.final_integral).abs() / reference.final_integral
                < 0.01
        );
        for (sample, (&got, &want)) in fine.samples.iter().zip(&reference.samples).enumerate() {
            assert!(
                (got - want).abs() / want.abs().max(1.0) < 0.08,
                "sample {sample}: Sonium={got:.9}, Wolfram={want:.9}"
            );
        }
    }
}
