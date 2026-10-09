//! Release-facing high-order FEM orientation and patch evidence.
//!
//! This bounded case exercises two contracts that are stronger than a
//! partition-of-unity smoke check:
//!
//! * P2 Lagrange interpolation reproduces an arbitrary quadratic polynomial
//!   and its gradient on both reference simplex dimensions.
//! * Tetrahedral order-2 Nédélec edge moments have the required odd/even
//!   parity when each independently reversible adjacent global edge ordering
//!   is reversed.
//!
//! The source is intentionally self-contained so it can be registered as a
//! release QA binary without changing shared QA infrastructure in this packet.

use sonium_fem::basis::{
    nedelec, p2_tetrahedron, p2_tetrahedron_grad, p2_triangle, p2_triangle_grad,
};
use sonium_qa::QaResult;
use std::time::Instant;

const CASE_ID: &str = "fem.high-order-orientation-patch.v1";
const PATCH_TOLERANCE: f64 = 2.0e-12;
const ORIENTATION_TOLERANCE: f64 = 2.0e-13;

#[derive(Debug, Clone, Copy)]
struct Metrics {
    triangle_value_residual: f64,
    triangle_gradient_residual: f64,
    tetrahedron_value_residual: f64,
    tetrahedron_gradient_residual: f64,
    edge_orientation_residual: f64,
    edge_orientation_checks: usize,
    elapsed_ms: u64,
}

fn triangle_quadratic(x: f64, y: f64) -> (f64, [f64; 2]) {
    let value = 0.7 + 1.1 * x - 0.9 * y + 2.0 * x * x - 1.3 * x * y + 1.7 * y * y;
    let gradient = [1.1 + 4.0 * x - 1.3 * y, -0.9 - 1.3 * x + 3.4 * y];
    (value, gradient)
}

fn tetrahedron_quadratic(x: f64, y: f64, z: f64) -> (f64, [f64; 3]) {
    let value =
        0.7 + 1.1 * x - 0.9 * y + 0.5 * z + 2.0 * x * x - 1.3 * x * y + 0.8 * x * z + 1.7 * y * y
            - 0.6 * y * z
            + 0.9 * z * z;
    let gradient = [
        1.1 + 4.0 * x - 1.3 * y + 0.8 * z,
        -0.9 - 1.3 * x + 3.4 * y - 0.6 * z,
        0.5 + 0.8 * x - 0.6 * y + 1.8 * z,
    ];
    (value, gradient)
}

fn max_abs(lhs: f64, rhs: f64) -> f64 {
    (lhs - rhs).abs()
}

fn check_triangle_patch() -> Result<(f64, f64), String> {
    let nodes = [
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [0.5, 0.0],
        [0.5, 0.5],
        [0.0, 0.5],
    ];
    let nodal_values: [f64; 6] = nodes.map(|[x, y]| triangle_quadratic(x, y).0);
    let samples = [[0.17, 0.23], [0.31, 0.19], [0.08, 0.11]];
    let mut value_residual: f64 = 0.0;
    let mut gradient_residual: f64 = 0.0;

    for [x, y] in samples {
        let basis = p2_triangle(x, y);
        let interpolated = basis
            .iter()
            .zip(nodal_values)
            .map(|(&shape, nodal)| shape * nodal)
            .sum::<f64>();
        let (exact, exact_gradient) = triangle_quadratic(x, y);
        value_residual = value_residual.max(max_abs(interpolated, exact));

        let gradients = p2_triangle_grad(x, y);
        for direction in 0..2 {
            let interpolated_gradient = gradients
                .iter()
                .zip(nodal_values)
                .map(|(gradient, nodal)| gradient[direction] * nodal)
                .sum::<f64>();
            gradient_residual =
                gradient_residual.max(max_abs(interpolated_gradient, exact_gradient[direction]));
        }
    }

    if value_residual > PATCH_TOLERANCE || gradient_residual > PATCH_TOLERANCE {
        return Err(format!(
            "P2 triangle quadratic patch failed: value={value_residual:.3e}, gradient={gradient_residual:.3e}"
        ));
    }
    Ok((value_residual, gradient_residual))
}

fn check_tetrahedron_patch() -> Result<(f64, f64), String> {
    let nodes = [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.5, 0.0, 0.0],
        [0.0, 0.5, 0.0],
        [0.0, 0.0, 0.5],
        [0.5, 0.5, 0.0],
        [0.5, 0.0, 0.5],
        [0.0, 0.5, 0.5],
    ];
    let nodal_values: [f64; 10] = nodes.map(|[x, y, z]| tetrahedron_quadratic(x, y, z).0);
    let samples = [[0.10, 0.20, 0.30], [0.18, 0.11, 0.22], [0.20, 0.25, 0.10]];
    let mut value_residual: f64 = 0.0;
    let mut gradient_residual: f64 = 0.0;

    for [x, y, z] in samples {
        let basis = p2_tetrahedron(x, y, z);
        let interpolated = basis
            .iter()
            .zip(nodal_values)
            .map(|(&shape, nodal)| shape * nodal)
            .sum::<f64>();
        let (exact, exact_gradient) = tetrahedron_quadratic(x, y, z);
        value_residual = value_residual.max(max_abs(interpolated, exact));

        let gradients = p2_tetrahedron_grad(x, y, z);
        for direction in 0..3 {
            let interpolated_gradient = gradients
                .iter()
                .zip(nodal_values)
                .map(|(gradient, nodal)| gradient[direction] * nodal)
                .sum::<f64>();
            gradient_residual =
                gradient_residual.max(max_abs(interpolated_gradient, exact_gradient[direction]));
        }
    }

    if value_residual > PATCH_TOLERANCE || gradient_residual > PATCH_TOLERANCE {
        return Err(format!(
            "P2 tetrahedron quadratic patch failed: value={value_residual:.3e}, gradient={gradient_residual:.3e}"
        ));
    }
    Ok((value_residual, gradient_residual))
}

fn check_tetrahedron_edge_orientation() -> Result<(f64, usize), String> {
    let point = (0.19, 0.23, 0.17);
    let canonical = nedelec::nedelec2_tetrahedron_oriented(point.0, point.1, point.2, [0, 1, 2, 3]);
    // These are the three adjacent pairs in TET_EDGES. Reversing an adjacent
    // pair changes only that pair's global orientation in the total order.
    let reversals = [[1, 0, 2, 3], [0, 2, 1, 3], [0, 1, 3, 2]];
    let target_edges = [0usize, 3, 5];
    let mut max_residual: f64 = 0.0;
    let mut checks = 0;

    for (&vertex_ids, edge_idx) in reversals.iter().zip(target_edges) {
        let reversed =
            nedelec::nedelec2_tetrahedron_oriented(point.0, point.1, point.2, vertex_ids);
        let first = 2 * edge_idx;
        for component in 0..3 {
            max_residual = max_residual
                .max((canonical.0[first][component] + reversed.0[first][component]).abs());
            max_residual = max_residual
                .max((canonical.1[first][component] + reversed.1[first][component]).abs());
            max_residual = max_residual
                .max((canonical.0[first + 1][component] - reversed.0[first + 1][component]).abs());
            max_residual = max_residual
                .max((canonical.1[first + 1][component] - reversed.1[first + 1][component]).abs());
            checks += 4;
        }

        // The other edge moments must be invariant under this local edge
        // reversal. Face-bubble transformations are intentionally outside
        // this bounded edge-parity slice.
        for basis in 0..12 {
            if basis == first || basis == first + 1 {
                continue;
            }
            for component in 0..3 {
                max_residual = max_residual
                    .max((canonical.0[basis][component] - reversed.0[basis][component]).abs());
                max_residual = max_residual
                    .max((canonical.1[basis][component] - reversed.1[basis][component]).abs());
                checks += 2;
            }
        }
    }

    if max_residual > ORIENTATION_TOLERANCE {
        return Err(format!(
            "order-2 tetrahedral edge orientation parity failed: residual={max_residual:.3e}"
        ));
    }
    Ok((max_residual, checks))
}

fn run() -> Result<Metrics, String> {
    let started = Instant::now();
    let (triangle_value_residual, triangle_gradient_residual) = check_triangle_patch()?;
    let (tetrahedron_value_residual, tetrahedron_gradient_residual) = check_tetrahedron_patch()?;
    let (edge_orientation_residual, edge_orientation_checks) =
        check_tetrahedron_edge_orientation()?;
    Ok(Metrics {
        triangle_value_residual,
        triangle_gradient_residual,
        tetrahedron_value_residual,
        tetrahedron_gradient_residual,
        edge_orientation_residual,
        edge_orientation_checks,
        elapsed_ms: started.elapsed().as_millis() as u64,
    })
}

fn main() {
    match run() {
        Ok(metrics) => {
            let record = QaResult::passed(CASE_ID)
                .requested_backend("cpu")
                .actual_backend("cpu")
                .metric("triangle_value_residual", metrics.triangle_value_residual)
                .metric("triangle_gradient_residual", metrics.triangle_gradient_residual)
                .metric("tetrahedron_value_residual", metrics.tetrahedron_value_residual)
                .metric("tetrahedron_gradient_residual", metrics.tetrahedron_gradient_residual)
                .metric("edge_orientation_residual", metrics.edge_orientation_residual)
                .metric("edge_orientation_checks", metrics.edge_orientation_checks as f64)
                .maximum("triangle_value_residual", PATCH_TOLERANCE)
                .maximum("triangle_gradient_residual", PATCH_TOLERANCE)
                .maximum("tetrahedron_value_residual", PATCH_TOLERANCE)
                .maximum("tetrahedron_gradient_residual", PATCH_TOLERANCE)
                .maximum("edge_orientation_residual", ORIENTATION_TOLERANCE)
                .reference("analytic quadratic polynomial reproduction and canonical Nedelec edge parity")
                .reference_checksum("analytic/deterministic-fixture")
                .input_checksum("analytic/deterministic-fixture:fem-high-order-orientation-v1")
                .dimension("reference P2 triangle/tetrahedron and order-2 H(curl) tetrahedron")
                .formulation("exact quadratic value/gradient patches plus reversible adjacent-edge orientation matrix")
                .tolerance("P2 value/gradient residuals <= 2e-12; Nedelec parity residual <= 2e-13")
                .seed("reference-simplex-quadratic:tetra-edge-adjacent-reversals-v1")
                .timing_ms("high_order_contracts", metrics.elapsed_ms)
                .elapsed_ms(metrics.elapsed_ms);
            if let Ok(json) = record.to_json_line() {
                println!("QA_RESULT: {json}");
            }
            println!("SUCCESS");
        }
        Err(error) => {
            if let Ok(json) = QaResult::failed(CASE_ID, error.clone())
                .requested_backend("cpu")
                .actual_backend("cpu")
                .to_json_line()
            {
                println!("QA_RESULT: {json}");
            }
            eprintln!("FEM HIGH-ORDER ORIENTATION/PATCH FAILED: {error}");
            println!("FAILED");
            std::process::exit(1);
        }
    }
}
