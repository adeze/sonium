//! Bounded CPU evidence for genuine constant-triangle Galerkin assembly.

use num_complex::Complex64;
use sonium_bem::core::mesh::generate_icosphere_mesh;
use sonium_bem::{PhysicsParams, assemble_galerkin_single_layer};
use sonium_qa::{QaResult, QaStatus};
use sonium_solvers::dense::DenseMatrix;
use std::time::Instant;

const CASE_ID: &str = "bem.galerkin-single-layer.v1";
/// Manifest acceptance bound for `bem.galerkin-single-layer.v1`.
///
/// Both the order-5 matrix and the order-7 refinement must satisfy
/// this symmetry bound; loosening it weakens the physics-gate claim.
const ASYMMETRY_TOLERANCE: f64 = 2.0e-8;

fn matrix_asymmetry(matrix: &DenseMatrix<Complex64>) -> f64 {
    (0..matrix.nrows())
        .flat_map(|row| (0..matrix.ncols()).map(move |column| (row, column)))
        .map(|(row, column)| (matrix[[row, column]] - matrix[[column, row]]).norm())
        .fold(0.0_f64, f64::max)
}

fn maximum_entry_delta(first: &DenseMatrix<Complex64>, second: &DenseMatrix<Complex64>) -> f64 {
    first
        .as_slice()
        .iter()
        .zip(second.as_slice())
        .map(|(coarse, fine)| (*coarse - *fine).norm())
        .fold(0.0_f64, f64::max)
}

fn reject_nonfinite(matrix: &DenseMatrix<Complex64>, order: usize) -> Result<(), String> {
    if matrix
        .as_slice()
        .iter()
        .any(|value| !value.re.is_finite() || !value.im.is_finite())
    {
        return Err(format!(
            "Galerkin single-layer order-{order} matrix has non-finite entries"
        ));
    }
    Ok(())
}

// Independent composite midpoint integration on a square-to-triangle map.
// This uses neither production quadrature rules nor production Green functions.
// Only separated panels are checked here; singular routes retain symmetry checks.
fn midpoint_pair(first: [[f64; 3]; 3], second: [[f64; 3]; 3], k: f64, n: usize) -> Complex64 {
    let samples = |vertices: [[f64; 3]; 3]| {
        let a: [f64; 3] = std::array::from_fn(|d| vertices[1][d] - vertices[0][d]);
        let b: [f64; 3] = std::array::from_fn(|d| vertices[2][d] - vertices[0][d]);
        let cross = [
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        ];
        let jacobian = cross.iter().map(|x| x * x).sum::<f64>().sqrt();
        let mut points = Vec::with_capacity(n * n);
        for i in 0..n {
            let u = (i as f64 + 0.5) / n as f64;
            for j in 0..n {
                let v = (j as f64 + 0.5) / n as f64;
                let point: [f64; 3] =
                    std::array::from_fn(|d| vertices[0][d] + u * a[d] + (1.0 - u) * v * b[d]);
                points.push((point, jacobian * (1.0 - u) / (n * n) as f64));
            }
        }
        points
    };
    let mut integral = Complex64::new(0.0, 0.0);
    let second_samples = samples(second);
    for (x, wx) in samples(first) {
        for (y, wy) in &second_samples {
            let r = (0..3).map(|d| (x[d] - y[d]).powi(2)).sum::<f64>().sqrt();
            integral += Complex64::from_polar(wx * wy / (4.0 * std::f64::consts::PI * r), k * r);
        }
    }
    integral
}

#[derive(Debug)]
struct GalerkinMetrics {
    dimension: usize,
    maximum_asymmetry: f64,
    maximum_order_delta: f64,
    reference_asymmetry: f64,
    reference_order_delta: f64,
    independent_reference_error: f64,
    independent_refinement_error: f64,
}

fn run() -> Result<GalerkinMetrics, String> {
    let mesh = generate_icosphere_mesh(0.2, 0);
    let mut elements = mesh.elements;
    for (index, element) in elements.iter_mut().enumerate() {
        element.dof_addresses = vec![index];
    }
    let physics =
        PhysicsParams::try_new(420.0, 343.0, 1.21, false).map_err(|error| error.to_string())?;
    let coarse = assemble_galerkin_single_layer(&elements, &mesh.nodes, &physics, 3)
        .map_err(|error| error.to_string())?;
    let fine = assemble_galerkin_single_layer(&elements, &mesh.nodes, &physics, 5)
        .map_err(|error| error.to_string())?;
    let reference = assemble_galerkin_single_layer(&elements, &mesh.nodes, &physics, 7)
        .map_err(|error| error.to_string())?;
    reject_nonfinite(&coarse, 3)?;
    reject_nonfinite(&fine, 5)?;
    reject_nonfinite(&reference, 7)?;
    let maximum_asymmetry = matrix_asymmetry(&fine);
    if !maximum_asymmetry.is_finite() || maximum_asymmetry > ASYMMETRY_TOLERANCE {
        return Err(format!(
            "Galerkin single-layer asymmetry {maximum_asymmetry:.3e} exceeds {ASYMMETRY_TOLERANCE:.1e}"
        ));
    }
    let reference_asymmetry = matrix_asymmetry(&reference);
    if !reference_asymmetry.is_finite() || reference_asymmetry > ASYMMETRY_TOLERANCE {
        return Err(format!(
            "Galerkin single-layer order-7 reference asymmetry {reference_asymmetry:.3e} exceeds {ASYMMETRY_TOLERANCE:.1e}"
        ));
    }
    let maximum_order_delta = maximum_entry_delta(&coarse, &fine);
    if !maximum_order_delta.is_finite() {
        return Err("Galerkin single-layer order delta is non-finite".to_string());
    }
    let reference_order_delta = maximum_entry_delta(&fine, &reference);
    if !reference_order_delta.is_finite() {
        return Err("Galerkin single-layer reference delta is non-finite".to_string());
    }
    // Select the farthest pair to keep the independent integrand smooth.
    let (row, column) = (0..elements.len())
        .flat_map(|i| (i + 1..elements.len()).map(move |j| (i, j)))
        .max_by(|&(i, j), &(a, b)| {
            let distance = |i: usize, j: usize| {
                (0..3)
                    .map(|d| (elements[i].center[d] - elements[j].center[d]).powi(2))
                    .sum::<f64>()
            };
            distance(i, j).total_cmp(&distance(a, b))
        })
        .ok_or("independent quadrature requires two panels")?;
    let vertices =
        |index: usize| std::array::from_fn(|d| mesh.nodes[elements[index].connectivity[d]]);
    let midpoint_coarse = midpoint_pair(vertices(row), vertices(column), physics.wave_number, 24);
    let midpoint_fine = midpoint_pair(vertices(row), vertices(column), physics.wave_number, 48);
    let independent_reference_error =
        (fine[[row, column]] - midpoint_fine).norm() / midpoint_fine.norm();
    let independent_refinement_error =
        (midpoint_coarse - midpoint_fine).norm() / midpoint_fine.norm();
    // The second-order midpoint reference is checked for convergence as well
    // as agreement; 0.1% bounds are well below this coarse fixture's accuracy.
    for error in [independent_reference_error, independent_refinement_error] {
        if !error.is_finite() || error > 1.0e-3 {
            return Err(format!(
                "independent quadrature relative error {error:.3e} exceeds 1e-3"
            ));
        }
    }
    Ok(GalerkinMetrics {
        dimension: fine.nrows(),
        maximum_asymmetry,
        maximum_order_delta,
        reference_asymmetry,
        reference_order_delta,
        independent_reference_error,
        independent_refinement_error,
    })
}

fn main() {
    let started = Instant::now();
    let result = match run() {
        Ok(GalerkinMetrics {
            dimension,
            maximum_asymmetry,
            maximum_order_delta,
            reference_asymmetry,
            reference_order_delta,
            independent_reference_error,
            independent_refinement_error,
        }) => QaResult::passed(CASE_ID)
            .metric("independent_reference_error", independent_reference_error)
            .metric("independent_refinement_error", independent_refinement_error)
            .maximum("independent_reference_error", 1.0e-3)
            .maximum("independent_refinement_error", 1.0e-3)
            .metric("dimension", dimension as f64)
            .metric("maximum_matrix_asymmetry", maximum_asymmetry)
            .metric("order_3_to_5_maximum_delta", maximum_order_delta)
            .metric("reference_matrix_asymmetry", reference_asymmetry)
            .metric("order_5_to_7_maximum_delta", reference_order_delta)
            .maximum("maximum_matrix_asymmetry", ASYMMETRY_TOLERANCE)
            .maximum("reference_matrix_asymmetry", ASYMMETRY_TOLERANCE)
            .input_checksum("bem:icosphere:subdivision-0:constant-tri3")
            .reference_checksum("analytic/deterministic-fixture:bem-galerkin-single-layer-v1")
            .dimension("3-D CPU 20-panel constant-triangle Galerkin single layer")
            .formulation("test and trial surface integration with Duffy/Sauter-Schwab pair routes")
            .tolerance("finite entries; matrix asymmetry <= 2e-8 at orders 5 and 7; separated-pair independent reference and refinement errors <= 1e-3")
            .seed("bem-galerkin-single-layer:cpu:v1")
            .elapsed_ms(started.elapsed().as_millis().try_into().unwrap_or(u64::MAX)),
        Err(error) => QaResult::failed(CASE_ID, error),
    };
    println!(
        "QA_RESULT: {}",
        result.to_json_line().expect("serialize QA result")
    );
    if matches!(result.status, QaStatus::Passed) {
        println!("SUCCESS");
    } else {
        println!("FAILED");
        std::process::exit(1);
    }
}
