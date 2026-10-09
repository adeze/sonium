//! Release-facing evidence for the bounded Quad4 quadratic kernel packet.
//!
//! The reference is deliberately independent of the production adaptive
//! regular and edge/Duffy singular schedules.  It does not solve a system or
//! claim that the separate post-processing quadratic field path is complete.

use num_complex::Complex64;
use sonium_bem::core::assembly::tbem::build_tbem_system_quadratic_stabilized;
use sonium_bem::core::integration::regular::regular_integration_linear;
use sonium_bem::core::integration::singular::{
    QuadratureParams, singular_integration_linear_with_params, singular_integration_with_params,
};
use sonium_bem::core::types::{
    BoundaryCondition, Element, ElementProperty, ElementType, PhysicsParams,
};
use sonium_qa::QaResult;
use sonium_solvers::dense::DenseMatrix;

const CASE_ID: &str = "bem.quad4-higher-order-regular-singular-reference.v1";

const PI: f64 = std::f64::consts::PI;
const FREQUENCY: f64 = 80.0;
const SPEED_OF_SOUND: f64 = 343.0;
const DENSITY: f64 = 1.21;
const SOURCE_HEIGHT: f64 = 1.75;

const GL12_X: [f64; 12] = [
    -0.9815606342467192,
    -0.9041172563704749,
    -0.7699026741943047,
    -0.5873179542866175,
    -0.3678314989981802,
    -0.1252334085114689,
    0.1252334085114689,
    0.3678314989981802,
    0.5873179542866175,
    0.7699026741943047,
    0.9041172563704749,
    0.9815606342467192,
];
const GL12_W: [f64; 12] = [
    0.0471753363865118,
    0.1069393259953184,
    0.1600783285433462,
    0.2031674267230659,
    0.2334925365383548,
    0.2491470458134028,
    0.2491470458134028,
    0.2334925365383548,
    0.2031674267230659,
    0.1600783285433462,
    0.1069393259953184,
    0.0471753363865118,
];

fn physics() -> PhysicsParams {
    PhysicsParams::try_new(FREQUENCY, SPEED_OF_SOUND, DENSITY, false).expect("valid QA physics")
}

fn nodes() -> Vec<[f64; 3]> {
    vec![
        [-0.5, -0.5, 0.0],
        [0.5, -0.5, 0.0],
        [0.5, 0.5, 0.0],
        [-0.5, 0.5, 0.0],
    ]
}

fn coords(nodes: &[[f64; 3]]) -> DenseMatrix<f64> {
    DenseMatrix::from_row_major_vec(
        4,
        3,
        nodes.iter().flat_map(|node| node.iter().copied()).collect(),
    )
}

fn element() -> Element {
    Element {
        connectivity: vec![0, 1, 2, 3],
        element_type: ElementType::Quad4,
        property: ElementProperty::Surface,
        normal: [0.0, 0.0, 1.0],
        node_normals: vec![[0.0, 0.0, 1.0]; 4],
        center: [0.0, 0.0, 0.0],
        area: 1.0,
        boundary_condition: BoundaryCondition::Velocity(vec![Complex64::new(0.0, 0.0)]),
        group: 0,
        dof_addresses: (0..9).collect(),
    }
}

fn basis(s: f64, t: f64) -> [f64; 9] {
    let ls = [0.5 * s * (s - 1.0), 1.0 - s * s, 0.5 * s * (s + 1.0)];
    let lt = [0.5 * t * (t - 1.0), 1.0 - t * t, 0.5 * t * (t + 1.0)];
    [
        ls[0] * lt[0],
        ls[2] * lt[0],
        ls[2] * lt[2],
        ls[0] * lt[2],
        ls[1] * lt[0],
        ls[2] * lt[1],
        ls[1] * lt[2],
        ls[0] * lt[1],
        ls[1] * lt[1],
    ]
}

fn green(physics: &PhysicsParams, radius: f64) -> Complex64 {
    Complex64::from_polar(1.0, physics.harmonic_factor * physics.wave_number * radius)
        / (4.0 * PI * radius)
}

fn relative(actual: Complex64, expected: Complex64) -> f64 {
    (actual - expected).norm() / expected.norm().max(1.0e-14)
}

fn regular_reference(physics: &PhysicsParams) -> [Complex64; 9] {
    let mut result = [Complex64::new(0.0, 0.0); 9];
    for (s_index, &s) in GL12_X.iter().enumerate() {
        for (t_index, &t) in GL12_X.iter().enumerate() {
            let point = [0.5 * s, 0.5 * t, 0.0];
            let dx = point[0];
            let dy = point[1];
            let dz = -SOURCE_HEIGHT;
            let radius = (dx * dx + dy * dy + dz * dz).sqrt();
            let kernel = green(physics, radius) * (0.25 * GL12_W[s_index] * GL12_W[t_index]);
            for (index, &value) in basis(s, t).iter().enumerate() {
                result[index] += kernel * value;
            }
        }
    }
    result
}

fn singular_reference(physics: &PhysicsParams) -> [Complex64; 9] {
    let mut result = [Complex64::new(0.0, 0.0); 9];
    for &sign_s in &[-1.0, 1.0] {
        for &sign_t in &[-1.0, 1.0] {
            for transpose in [false, true] {
                for (u_index, &u_node) in GL12_X.iter().enumerate() {
                    let u = 0.5 * (u_node + 1.0);
                    let wu = 0.5 * GL12_W[u_index];
                    for (v_index, &v_node) in GL12_X.iter().enumerate() {
                        let v = 0.5 * (v_node + 1.0);
                        let wv = 0.5 * GL12_W[v_index];
                        let (s, t) = if transpose {
                            (sign_s * u * v, sign_t * u)
                        } else {
                            (sign_s * u, sign_t * u * v)
                        };
                        let radius = 0.5 * u * (1.0 + v * v).sqrt();
                        let kernel = green(physics, radius) * (0.25 * u * wu * wv);
                        for (index, &value) in basis(s, t).iter().enumerate() {
                            result[index] += kernel * value;
                        }
                    }
                }
            }
        }
    }
    result
}

fn run() {
    let mesh_nodes = nodes();
    let element = element();
    let element_coords = coords(&mesh_nodes);
    let physics = physics();
    let source_point = [0.0, 0.0, SOURCE_HEIGHT];
    let source_normal = [0.0, 0.0, 1.0];

    let regular = regular_integration_linear(
        &source_point,
        &source_normal,
        &element_coords,
        ElementType::Quad4,
        1.0,
        &physics,
        9,
    );
    let regular_actual: Vec<_> = regular
        .basis_results
        .iter()
        .map(|result| result.g_integral)
        .collect();
    let regular_expected = regular_reference(&physics);
    let regular_max_relative_error = regular_actual
        .iter()
        .zip(regular_expected)
        .map(|(&actual, expected)| relative(actual, expected))
        .fold(0.0, f64::max);

    let high_accuracy = QuadratureParams::high_accuracy();
    let singular = singular_integration_linear_with_params(
        &[0.0, 0.0, 0.0],
        &source_normal,
        &element_coords,
        ElementType::Quad4,
        &physics,
        9,
        &high_accuracy,
        None,
    );
    let singular_actual: Vec<_> = singular
        .basis_results
        .iter()
        .map(|result| result.g_integral)
        .collect();
    let singular_expected = singular_reference(&physics);
    let singular_max_relative_error = singular_actual
        .iter()
        .zip(singular_expected)
        .map(|(&actual, expected)| relative(actual, expected))
        .fold(0.0, f64::max);
    let singular_scalar = singular_integration_with_params(
        &[0.0, 0.0, 0.0],
        &source_normal,
        &element_coords,
        ElementType::Quad4,
        &physics,
        None,
        0,
        false,
        &high_accuracy,
        None,
    );
    let singular_partition_error = relative(
        singular_actual.iter().copied().sum(),
        singular_scalar.g_integral,
    );

    let system = build_tbem_system_quadratic_stabilized(
        std::slice::from_ref(&element),
        &mesh_nodes,
        &physics,
        1.0,
        1.0,
    );
    let matrix_finite = system
        .matrix
        .as_slice()
        .iter()
        .all(|value| value.re.is_finite() && value.im.is_finite());
    let matrix_nonzero = system
        .matrix
        .as_slice()
        .iter()
        .any(|value| value.norm() > 1.0e-14);

    assert!(regular_max_relative_error < 2.0e-7);
    assert!(singular_max_relative_error < 3.0e-4);
    assert!(singular_partition_error < 2.0e-11);
    assert_eq!(system.matrix.shape(), (9, 9));
    assert_eq!(system.rhs.len(), 9);
    assert!(matrix_finite && matrix_nonzero);

    let qa_result = QaResult::passed(CASE_ID)
        .requested_backend("cpu")
        .actual_backend("cpu")
        .metric("regular_max_relative_error", regular_max_relative_error)
        .metric("singular_max_relative_error", singular_max_relative_error)
        .metric("singular_partition_error", singular_partition_error)
        .metric("assembled_dofs", system.matrix.shape().0 as f64)
        .metric("matrix_finite", if matrix_finite { 1.0 } else { 0.0 })
        .metric("matrix_nonzero", if matrix_nonzero { 1.0 } else { 0.0 })
        .minimum("matrix_finite", 1.0)
        .minimum("matrix_nonzero", 1.0)
        .maximum("regular_max_relative_error", 2.0e-7)
        .maximum("singular_max_relative_error", 3.0e-4)
        .maximum("singular_partition_error", 2.0e-11)
        .input_checksum("analytic/deterministic-fixture:bem-quad4-higher-order-v1")
        .reference("independent 12-point tensor Gauss and quadrant-Duffy kernel references")
        .reference_checksum("analytic/deterministic-fixture:bem-quad4-higher-order-v1")
        .dimension("single regular Quad4 and singular self-panel 9-function quadratic kernel")
        .formulation("CPU higher-order regular/singular TBEM kernel and finite 9x9 assembly")
        .tolerance("regular < 2e-7; singular < 3e-4; partition < 2e-11")
        .seed("bem-quad4-higher-order-reference:cpu:v1");
    println!(
        "QA_RESULT: {}",
        qa_result
            .to_json_line()
            .expect("serialize BEM higher-order QA result")
    );
    println!(
        "BEM_EVIDENCE: {{\"case_id\":\"bem.quad4-higher-order-regular-singular-reference.v1\",\"status\":\"passed\",\"regular_max_relative_error\":{regular_max_relative_error:.6e},\"singular_max_relative_error\":{singular_max_relative_error:.6e},\"singular_partition_error\":{singular_partition_error:.6e},\"assembled_dofs\":{},\"matrix_finite\":{},\"matrix_nonzero\":{}}}",
        system.matrix.shape().0,
        matrix_finite,
        matrix_nonzero,
    );
}

fn main() {
    if std::panic::catch_unwind(run).is_ok() {
        println!("SUCCESS");
    } else {
        println!(
            "QA_RESULT: {}",
            QaResult::failed(CASE_ID, "higher-order kernel reference panicked")
                .requested_backend("cpu")
                .actual_backend("cpu")
                .to_json_line()
                .expect("BEM higher-order failure result must serialize")
        );
        eprintln!("FAILED");
        std::process::exit(1);
    }
}
