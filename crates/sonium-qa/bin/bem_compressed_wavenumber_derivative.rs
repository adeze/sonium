//! Bounded compressed-operator wavenumber derivative reference.

use num_complex::Complex64;
use sonium_bem::core::assembly::hmatrix::{
    AcaVariant, LowRankPrecision, build_hmatrix_stabilized,
    build_hmatrix_stabilized_wavenumber_derivative,
};
use sonium_bem::core::{
    BoundaryCondition, Cluster, Element, ElementProperty, ElementType, PhysicsParams,
};
use sonium_qa::QaResult;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::time::Instant;

const CASE_ID: &str = "bem.compressed-wavenumber-derivative.v1";

#[derive(Debug, Clone, Copy)]
struct Metrics {
    dimension: usize,
    low_rank_blocks: usize,
    relative_error: f64,
}

fn fixture() -> (Vec<Element>, Vec<[f64; 3]>, Vec<Cluster>) {
    let mut nodes = Vec::new();
    let mut elements = Vec::new();
    for patch in 0..2 {
        let x_offset = patch as f64 * 12.0;
        for local in 0..8 {
            let x = x_offset + local as f64 * 1.1;
            let base = nodes.len();
            nodes.extend_from_slice(&[[x, 0.0, 0.0], [x + 0.8, 0.0, 0.0], [x, 0.8, 0.0]]);
            elements.push(Element {
                connectivity: vec![base, base + 1, base + 2],
                element_type: ElementType::Tri3,
                property: ElementProperty::Surface,
                normal: [0.0, 0.0, 1.0],
                node_normals: vec![[0.0; 3]; 3],
                center: [x + 0.8 / 3.0, 0.8 / 3.0, 0.0],
                area: 0.32,
                boundary_condition: BoundaryCondition::Velocity(vec![Complex64::new(0.0, 0.0)]),
                group: 0,
                dof_addresses: vec![patch * 8 + local],
            });
        }
    }

    let mut left = Cluster::new([4.12, 0.27, 0.0]);
    left.element_indices = (0..8).collect();
    left.num_elements = 8;
    left.radius = 4.5;
    left.near_clusters = vec![0];
    left.far_clusters = vec![1];

    let mut right = Cluster::new([16.12, 0.27, 0.0]);
    right.element_indices = (8..16).collect();
    right.num_elements = 8;
    right.radius = 4.5;
    right.near_clusters = vec![1];
    right.far_clusters = vec![0];

    (elements, nodes, vec![left, right])
}

fn relative_error(actual: &[Complex64], expected: &[Complex64]) -> f64 {
    let (numerator, denominator) = actual
        .iter()
        .zip(expected)
        .fold((0.0_f64, 0.0_f64), |(num, den), (a, e)| {
            (num.max((*a - *e).norm()), den.max(e.norm()))
        });
    numerator / denominator.max(1.0e-30)
}

fn run() -> Result<Metrics, String> {
    let (elements, nodes, clusters) = fixture();
    let speed = 343.0;
    let density = 1.21;
    let base = PhysicsParams::new(120.0, speed, density, false);
    let delta_k = 1.0e-4;
    let plus = PhysicsParams::new(
        (base.wave_number + delta_k) * speed / std::f64::consts::TAU,
        speed,
        density,
        false,
    );
    let minus = PhysicsParams::new(
        (base.wave_number - delta_k) * speed / std::f64::consts::TAU,
        speed,
        density,
        false,
    );
    let derivative = build_hmatrix_stabilized_wavenumber_derivative(
        &elements,
        &nodes,
        &clusters,
        &base,
        1.0e-4,
        16,
        AcaVariant::Hca,
        LowRankPrecision::F64,
        1.0,
        1.0,
        6,
    )
    .map_err(|error| error.to_string())?;
    let plus_matrix =
        build_hmatrix_stabilized(&elements, &nodes, &clusters, &plus, 1.0e-4, 16, 1.0, 1.0);
    let minus_matrix =
        build_hmatrix_stabilized(&elements, &nodes, &clusters, &minus, 1.0e-4, 16, 1.0, 1.0);
    let x: Vec<_> = (0..elements.len())
        .map(|index| {
            Complex64::new(
                (0.37 * index as f64 + 0.1).sin(),
                (0.19 * index as f64).cos(),
            )
        })
        .collect();
    let expected: Vec<_> = plus_matrix
        .matvec(&x)
        .iter()
        .zip(minus_matrix.matvec(&x))
        .map(|(p, m)| (*p - m) / (2.0 * delta_k))
        .collect();
    let actual = derivative.matvec(&x);
    let error = relative_error(&actual, &expected);
    if !error.is_finite() || error > 3.0e-4 {
        return Err(format!(
            "compressed wavenumber derivative mismatch: {error:e}"
        ));
    }
    if derivative.low_rank_blocks.is_empty() {
        return Err("derivative assembly produced no compressed far-field block".to_string());
    }
    Ok(Metrics {
        dimension: elements.len(),
        low_rank_blocks: derivative.low_rank_blocks.len(),
        relative_error: error,
    })
}

fn main() {
    let started = Instant::now();
    let result = match catch_unwind(AssertUnwindSafe(run)) {
        Ok(Ok(metrics)) => QaResult::passed(CASE_ID)
            .requested_backend("cpu")
            .actual_backend("cpu")
            .metric("dimension", metrics.dimension as f64)
            .metric("low_rank_blocks", metrics.low_rank_blocks as f64)
            .metric("relative_error", metrics.relative_error)
            .minimum("dimension", 1.0)
            .minimum("low_rank_blocks", 1.0)
            .maximum("relative_error", 3.0e-4)
            .input_checksum(
                "analytic/deterministic-fixture:bem-compressed-wavenumber-derivative-v1",
            )
            .reference("independent central difference of compressed H-matrix actions")
            .reference_checksum(
                "analytic/deterministic-fixture:bem-compressed-wavenumber-derivative-v1",
            )
            .dimension("two separated eight-panel Tri3 clusters; Complex64 action")
            .formulation("compressed stabilized TBEM analytic wavenumber derivative")
            .tolerance("relative action error <= 3e-4 and at least one low-rank block")
            .seed("bem-compressed-wavenumber-derivative:cpu:v1")
            .elapsed_ms(started.elapsed().as_millis().try_into().unwrap()),
        Ok(Err(error)) => QaResult::failed(CASE_ID, error)
            .elapsed_ms(started.elapsed().as_millis().try_into().unwrap()),
        Err(_) => QaResult::failed(
            CASE_ID,
            "panic while running compressed derivative reference",
        )
        .elapsed_ms(started.elapsed().as_millis().try_into().unwrap()),
    };
    println!(
        "QA_RESULT: {}",
        serde_json::to_string(&result).expect("QA JSON serialization")
    );
    if result.status == sonium_qa::QaStatus::Passed {
        println!("SUCCESS");
    } else {
        println!("FAILED");
        std::process::exit(1);
    }
}
