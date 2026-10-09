//! Release-backed P3 H1 reproduction for triangle and quadrilateral elements.

use sonium_fem::basis::PolynomialDegree;
use sonium_fem::mesh::{Mesh, unit_square_quads, unit_square_triangles};
use sonium_fem::spaces::{FunctionSpace, H1Space};
use sonium_qa::QaResult;
use std::time::Instant;

const CASE_ID: &str = "fem.p3-h1-triangle-quad-reproduction.v1";
const TOLERANCE: f64 = 2.0e-11;

fn polynomial(x: f64, y: f64) -> f64 {
    0.35 * x.powi(3) - 0.8 * x.powi(2) * y + 0.55 * x * y.powi(2) - 0.4 * y.powi(3)
        + 0.7 * x.powi(2)
        - 0.25 * x * y
        + 0.9 * y.powi(2)
        + 0.2 * x
        - 0.6 * y
        + 1.1
}

fn triangle_reference_nodes() -> [[f64; 2]; 10] {
    [
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0 / 3.0, 0.0],
        [2.0 / 3.0, 0.0],
        [2.0 / 3.0, 1.0 / 3.0],
        [1.0 / 3.0, 2.0 / 3.0],
        [0.0, 2.0 / 3.0],
        [0.0, 1.0 / 3.0],
        [1.0 / 3.0, 1.0 / 3.0],
    ]
}

fn quadrilateral_reference_nodes() -> [[f64; 2]; 16] {
    [
        [-1.0, -1.0],
        [1.0, -1.0],
        [1.0, 1.0],
        [-1.0, 1.0],
        [-1.0 / 3.0, -1.0],
        [1.0 / 3.0, -1.0],
        [1.0, -1.0 / 3.0],
        [1.0, 1.0 / 3.0],
        [1.0 / 3.0, 1.0],
        [-1.0 / 3.0, 1.0],
        [-1.0, 1.0 / 3.0],
        [-1.0, -1.0 / 3.0],
        [-1.0 / 3.0, -1.0 / 3.0],
        [1.0 / 3.0, -1.0 / 3.0],
        [1.0 / 3.0, 1.0 / 3.0],
        [-1.0 / 3.0, 1.0 / 3.0],
    ]
}

fn map_reference(mesh: &Mesh, element: usize, xi: f64, eta: f64) -> [f64; 2] {
    let cell = &mesh.elements[element];
    let vertices = &cell.nodes[..cell.element_type.num_vertices()];
    match cell.element_type {
        sonium_fem::mesh::ElementType::Triangle => {
            let p0 = mesh.nodes[vertices[0]];
            let p1 = mesh.nodes[vertices[1]];
            let p2 = mesh.nodes[vertices[2]];
            [
                p0.x + xi * (p1.x - p0.x) + eta * (p2.x - p0.x),
                p0.y + xi * (p1.y - p0.y) + eta * (p2.y - p0.y),
            ]
        }
        sonium_fem::mesh::ElementType::Quadrilateral => {
            let p = [
                mesh.nodes[vertices[0]],
                mesh.nodes[vertices[1]],
                mesh.nodes[vertices[2]],
                mesh.nodes[vertices[3]],
            ];
            let weights = [
                0.25 * (1.0 - xi) * (1.0 - eta),
                0.25 * (1.0 + xi) * (1.0 - eta),
                0.25 * (1.0 + xi) * (1.0 + eta),
                0.25 * (1.0 - xi) * (1.0 + eta),
            ];
            [
                weights
                    .iter()
                    .zip(p)
                    .map(|(weight, point)| weight * point.x)
                    .sum(),
                weights
                    .iter()
                    .zip(p)
                    .map(|(weight, point)| weight * point.y)
                    .sum(),
            ]
        }
        other => panic!("unexpected element type in P3 QA: {other:?}"),
    }
}

fn run_family(mesh: &Mesh, refs: &[[f64; 2]]) -> Result<(f64, usize), String> {
    let space = H1Space::new(PolynomialDegree::P3);
    space
        .validate(mesh)
        .map_err(|error| format!("P3 H1 validation failed: {error}"))?;
    let mut max_error: f64 = 0.0;
    let mut samples = 0;
    for element in 0..mesh.num_elements() {
        let nodal_values: Vec<f64> = refs
            .iter()
            .map(|[xi, eta]| {
                let [x, y] = map_reference(mesh, element, *xi, *eta);
                polynomial(x, y)
            })
            .collect();
        for [xi, eta] in refs {
            let [x, y] = map_reference(mesh, element, *xi, *eta);
            let expected = polynomial(x, y);
            let tabulation = space.tabulate(mesh, element, *xi, *eta, 0.0);
            let actual: f64 = tabulation
                .values
                .iter()
                .zip(&nodal_values)
                .map(|(shape, value)| shape * value)
                .sum();
            max_error = max_error.max((actual - expected).abs());
            samples += 1;
        }
    }
    Ok((max_error, samples))
}

fn run() -> Result<(f64, usize, u64), String> {
    let started = Instant::now();
    let triangle = triangle_reference_nodes();
    let quadrilateral = quadrilateral_reference_nodes();
    let (triangle_error, triangle_samples) = run_family(&unit_square_triangles(2), &triangle)?;
    let (quadrilateral_error, quadrilateral_samples) =
        run_family(&unit_square_quads(2), &quadrilateral)?;
    let max_error = triangle_error.max(quadrilateral_error);
    if max_error > TOLERANCE {
        return Err(format!(
            "P3 reproduction residual {max_error:e} exceeds {TOLERANCE:e}"
        ));
    }
    Ok((
        max_error,
        triangle_samples + quadrilateral_samples,
        started.elapsed().as_millis() as u64,
    ))
}

fn main() {
    match run() {
        Ok((max_error, samples, elapsed_ms)) => {
            let result = QaResult::passed(CASE_ID)
                .requested_backend("cpu")
                .actual_backend("cpu")
                .metric("max_p3_reproduction_residual", max_error)
                .metric("sample_checks", samples as f64)
                .maximum("max_p3_reproduction_residual", TOLERANCE)
                .reference("independent cubic polynomial on affine P3 triangles and quadrilaterals")
                .reference_checksum(
                    "analytic/deterministic-fixture:fem-p3-h1-triangle-quad-reproduction-v1",
                )
                .input_checksum("mesh:unit-square-triangles-and-quads:2")
                .dimension("2-D P3 H1 triangle and Q3 quadrilateral")
                .formulation(
                    "topology-owned vertices, oriented edge nodes, and element-interior nodes",
                )
                .tolerance("cubic interpolation residual <= 2e-11")
                .seed("reference-cubic:unit-square-triangle-quad-v1")
                .timing_ms("p3_h1_reproduction", elapsed_ms)
                .elapsed_ms(elapsed_ms);
            println!(
                "QA_RESULT: {}",
                result.to_json_line().expect("QA result JSON")
            );
            println!("SUCCESS");
        }
        Err(error) => {
            if let Ok(result) = QaResult::failed(CASE_ID, error.clone())
                .requested_backend("cpu")
                .actual_backend("cpu")
                .to_json_line()
            {
                println!("QA_RESULT: {result}");
            }
            eprintln!("FEM P3 H1 TRIANGLE/QUADRILATERAL REPRODUCTION FAILED: {error}");
            println!("FAILED");
            std::process::exit(1);
        }
    }
}
