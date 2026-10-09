//! Cross-space finite-element basis contracts for release QA.

use sonium_fem::basis::{
    nedelec, p1_tetrahedron, p1_tetrahedron_grad, p1_triangle, p1_triangle_grad, p2_tetrahedron,
    p2_tetrahedron_grad, p2_triangle, p2_triangle_grad, raviart_thomas,
};
use sonium_fem::validation::check_basis_partition_of_unity;
use sonium_qa::QaResult;

fn main() {
    match run() {
        Ok(()) => {
            let record = QaResult::passed("fem.element-contracts.v1")
                .requested_backend("cpu")
                .actual_backend("cpu")
                .metric("rt0_triangle_dofs", 3.0)
                .metric("rt0_tetrahedron_dofs", 4.0)
                .metric("nedelec1_triangle_dofs", 3.0)
                .metric("nedelec1_tetrahedron_dofs", 6.0)
                .metric("nedelec_constant_curl_checks", 2.0)
                .metric("scalar_partition_checks", 12.0)
                .metric("p2_triangle_kronecker_nodes", 6.0)
                .metric("nedelec_edge_trace_checks", 9.0)
                .metric("nedelec_orientation_checks", 8.0)
                .metric("exact_sequence_curl_grad_checks", 12.0)
                .reference(
                    "exact nodal partition/unisolvence, lowest-order DOF counts, edge traces, orientation parity, constant-curl property, and independent curl(grad) finite-difference check",
                )
                .reference_checksum("analytic")
                .dimension("triangle and tetrahedron local entities")
                .formulation("H1 partition/unisolvence plus H(div)/H(curl) local finite-element contracts")
                .tolerance("exact counts; 2e-13 identity residual; finite-difference curl(grad) <= 5e-8");
            if let Ok(json) = record.to_json_line() {
                println!("QA_RESULT: {json}");
            }
            println!("SUCCESS");
        }
        Err(error) => {
            if let Ok(json) = QaResult::failed("fem.element-contracts.v1", error.clone())
                .requested_backend("cpu")
                .actual_backend("cpu")
                .to_json_line()
            {
                println!("QA_RESULT: {json}");
            }
            eprintln!("FEM ELEMENT CONTRACT FAILED: {error}");
            println!("FAILED");
            std::process::exit(1);
        }
    }
}

fn run() -> Result<(), String> {
    check_scalar_basis_contracts()?;
    let (rt_tri, div_tri) = raviart_thomas::rt0_triangle(0.2, 0.3);
    let (rt_tet, div_tet) = raviart_thomas::rt0_tetrahedron(0.1, 0.2, 0.3);
    let (ned_tri, curl_tri) = nedelec::nedelec1_triangle(0.2, 0.3);
    let (ned_tet, curl_tet) = nedelec::nedelec1_tetrahedron(0.1, 0.2, 0.3);
    let finite = rt_tri
        .iter()
        .flatten()
        .chain(rt_tet.iter().flatten())
        .chain(ned_tri.iter().flatten())
        .chain(ned_tet.iter().flatten())
        .chain(div_tri.iter())
        .chain(div_tet.iter())
        .chain(curl_tri.iter())
        .chain(curl_tet.iter().flatten())
        .all(|value| value.is_finite());
    if !finite
        || rt_tri.len() != 3
        || div_tri.len() != 3
        || rt_tet.len() != 4
        || div_tet.len() != 4
        || ned_tri.len() != 3
        || curl_tri.len() != 3
        || ned_tet.len() != 6
        || curl_tet.len() != 6
    {
        return Err(
            "basis tabulation returned non-finite values or an invalid local DOF count".into(),
        );
    }
    // Lowest-order Nedelec curls are constant; tabulate a second interior point.
    let (_, curl_tri_second) = nedelec::nedelec1_triangle(0.1, 0.15);
    let (_, curl_tet_second) = nedelec::nedelec1_tetrahedron(0.15, 0.1, 0.2);
    if curl_tri != curl_tri_second || curl_tet != curl_tet_second {
        return Err("lowest-order Nedelec curl is not constant".into());
    }
    check_nedelec_edge_traces()?;
    check_nedelec_orientation_parity()?;
    check_exact_sequence_curl_grad()?;
    println!("spaces=hdiv-rt0,hcurl-nedelec1 local_dofs=3,4,3,6 curls=constant");
    Ok(())
}

fn check_scalar_basis_contracts() -> Result<(), String> {
    let tolerance = 2.0e-13;
    let samples = [(0.17, 0.23), (0.31, 0.19), (0.08, 0.11)];
    for &(xi, eta) in &samples {
        let p1 = p1_triangle(xi, eta);
        let p1_grad = p1_triangle_grad().map(|[dx, dy]| [dx, dy, 0.0]);
        check_basis_partition_of_unity(&p1, &p1_grad, tolerance)
            .map_err(|error| format!("P1 triangle partition contract failed: {error}"))?;

        let p2 = p2_triangle(xi, eta);
        let p2_grad = p2_triangle_grad(xi, eta).map(|[dx, dy]| [dx, dy, 0.0]);
        check_basis_partition_of_unity(&p2, &p2_grad, tolerance)
            .map_err(|error| format!("P2 triangle partition contract failed: {error}"))?;

        let p1_tet = p1_tetrahedron(xi, eta, 0.07);
        check_basis_partition_of_unity(&p1_tet, &p1_tetrahedron_grad(), tolerance)
            .map_err(|error| format!("P1 tetrahedron partition contract failed: {error}"))?;

        let p2_tet = p2_tetrahedron(xi, eta, 0.07);
        check_basis_partition_of_unity(&p2_tet, &p2_tetrahedron_grad(xi, eta, 0.07), tolerance)
            .map_err(|error| format!("P2 tetrahedron partition contract failed: {error}"))?;
    }

    // P2 nodal values must be the identity at the four vertices and six edge
    // midpoints. This catches a node-order or barycentric sign regression
    // that partition-of-unity alone cannot detect.
    let p2_nodes = [
        (0.0, 0.0),
        (1.0, 0.0),
        (0.0, 1.0),
        (0.5, 0.0),
        (0.5, 0.5),
        (0.0, 0.5),
    ];
    for (node, &(xi, eta)) in p2_nodes.iter().enumerate() {
        let values = p2_triangle(xi, eta);
        for (basis, &value) in values.iter().enumerate() {
            let expected = if basis == node { 1.0 } else { 0.0 };
            if (value - expected).abs() > tolerance {
                return Err(format!(
                    "P2 triangle Kronecker contract failed at node {node}, basis {basis}: {value}"
                ));
            }
        }
    }
    Ok(())
}

fn check_nedelec_edge_traces() -> Result<(), String> {
    // The Whitney edge DOF is the oriented line integral. On each affine
    // reference edge the tangential trace is constant, so midpoint evaluation
    // against the edge vector is exact and gives the Kronecker matrix.
    for (edge, &(start, end)) in nedelec::TRIANGLE_EDGES.iter().enumerate() {
        let vertices = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]];
        let midpoint = [
            0.5 * (vertices[start][0] + vertices[end][0]),
            0.5 * (vertices[start][1] + vertices[end][1]),
        ];
        let tangent = [
            vertices[end][0] - vertices[start][0],
            vertices[end][1] - vertices[start][1],
        ];
        let (values, _) = nedelec::nedelec1_triangle(midpoint[0], midpoint[1]);
        for (basis, value) in values.iter().enumerate() {
            let integral = value[0] * tangent[0] + value[1] * tangent[1];
            let expected = if basis == edge { 1.0 } else { 0.0 };
            if (integral - expected).abs() > 2.0e-13 {
                return Err(format!(
                    "Nedelec edge trace contract failed on edge {edge}, basis {basis}: {integral}"
                ));
            }
        }
    }
    Ok(())
}

fn check_nedelec_orientation_parity() -> Result<(), String> {
    let canonical = nedelec::nedelec2_triangle_oriented(0.23, 0.19, [0, 1, 2]);
    let reversed = nedelec::nedelec2_triangle_oriented(0.23, 0.19, [1, 0, 2]);
    // Reversing edge (0,1) changes the odd first moment by a sign while the
    // even higher moment remains unchanged. Other edges are unaffected.
    for component in 0..2 {
        if (canonical.0[0][component] + reversed.0[0][component]).abs() > 2.0e-13
            || (canonical.0[1][component] - reversed.0[1][component]).abs() > 2.0e-13
        {
            return Err("Nedelec order-2 edge orientation parity failed".into());
        }
    }
    for basis in 2..6 {
        for component in 0..2 {
            if (canonical.0[basis][component] - reversed.0[basis][component]).abs() > 2.0e-13 {
                return Err(format!(
                    "Nedelec orientation changed unrelated edge basis {basis}"
                ));
            }
        }
    }
    Ok(())
}

fn check_exact_sequence_curl_grad() -> Result<(), String> {
    // On the reference triangle, the scalar P2 space maps into the vector
    // H(curl) space through the gradient.  The exterior derivative therefore
    // satisfies curl(grad(phi)) = 0.  Use p2_triangle itself for an
    // independent finite-difference derivative reference, rather than merely
    // differentiating the implementation under test twice.
    let epsilon = 1.0e-6;
    let tolerance = 5.0e-8;
    let samples = [(0.17, 0.23), (0.31, 0.19)];
    for &(xi, eta) in &samples {
        let gradients = p2_triangle_grad(xi, eta);
        let gradients_x_plus = p2_triangle_grad(xi + epsilon, eta);
        let gradients_x_minus = p2_triangle_grad(xi - epsilon, eta);
        let gradients_y_plus = p2_triangle_grad(xi, eta + epsilon);
        let gradients_y_minus = p2_triangle_grad(xi, eta - epsilon);
        let values_x_plus = p2_triangle(xi + epsilon, eta);
        let values_x_minus = p2_triangle(xi - epsilon, eta);
        let values_y_plus = p2_triangle(xi, eta + epsilon);
        let values_y_minus = p2_triangle(xi, eta - epsilon);

        for basis in 0..gradients.len() {
            let d_dx_grad_y =
                (gradients_x_plus[basis][1] - gradients_x_minus[basis][1]) / (2.0 * epsilon);
            let d_dy_grad_x =
                (gradients_y_plus[basis][0] - gradients_y_minus[basis][0]) / (2.0 * epsilon);
            let curl_grad = d_dx_grad_y - d_dy_grad_x;
            if curl_grad.abs() > tolerance {
                return Err(format!(
                    "exact-sequence curl(grad) failed at ({xi}, {eta}), basis {basis}: {curl_grad}"
                ));
            }

            let numeric_dx = (values_x_plus[basis] - values_x_minus[basis]) / (2.0 * epsilon);
            let numeric_dy = (values_y_plus[basis] - values_y_minus[basis]) / (2.0 * epsilon);
            if (numeric_dx - gradients[basis][0]).abs() > tolerance
                || (numeric_dy - gradients[basis][1]).abs() > tolerance
            {
                return Err(format!(
                    "P2 gradient/reference mismatch at ({xi}, {eta}), basis {basis}: numeric=({numeric_dx}, {numeric_dy}), tabulated=({:?})",
                    gradients[basis]
                ));
            }
        }
    }
    Ok(())
}
