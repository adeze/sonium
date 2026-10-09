//! Release-facing checked boundary for parametric FEM CSR assembly.

use sonium_fem::assembly::{PrebakedAssembler, TripletMatrix};
use sonium_qa::QaResult;
use std::time::Instant;

const CASE_ID: &str = "fem.prebaked-checked-contract.v1";

fn diagonal(size: usize, value: f64) -> TripletMatrix<f64> {
    let mut matrix = TripletMatrix::new(size, size);
    for index in 0..size {
        matrix.push(index, index, value);
    }
    matrix
}

fn main() {
    let started = Instant::now();
    let mut checks = 0.0;
    let mut failures = Vec::new();

    match PrebakedAssembler::<2>::try_from_triplets([diagonal(2, 2.0), diagonal(2, 3.0)], &[]) {
        Ok(assembler) if assembler.num_rows == 2 && assembler.nnz() == 2 => checks += 1.0,
        Ok(_) => failures.push("valid assembler shape"),
        Err(_) => failures.push("valid assembler rejected"),
    }

    let mut nonfinite = diagonal(2, 1.0);
    nonfinite.push(0, 1, f64::NAN);
    if PrebakedAssembler::<2>::try_from_triplets([nonfinite, diagonal(2, 1.0)], &[]).is_err() {
        checks += 1.0;
    } else {
        failures.push("non-finite triplet accepted");
    }

    let mut out_of_bounds = TripletMatrix::new(2, 2);
    out_of_bounds.push(2, 0, 1.0);
    if PrebakedAssembler::<2>::try_from_triplets([out_of_bounds, diagonal(2, 1.0)], &[]).is_err() {
        checks += 1.0;
    } else {
        failures.push("out-of-bounds triplet accepted");
    }

    let boundary = diagonal(2, 1.0);
    if PrebakedAssembler::<2>::try_from_triplets(
        [diagonal(2, 1.0), diagonal(2, 1.0)],
        &[(7, boundary.clone()), (7, boundary)],
    )
    .is_err()
    {
        checks += 1.0;
    } else {
        failures.push("duplicate boundary tag accepted");
    }

    let result = if failures.is_empty() && checks == 4.0 {
        QaResult::passed(CASE_ID)
            .requested_backend("cpu")
            .actual_backend("cpu")
            .metric("boundary_checks", checks)
            .minimum("boundary_checks", 4.0)
            .reference("independent malformed-triplet admission and valid CSR merge contract")
            .reference_checksum("analytic/deterministic-fixture:fem-prebaked-checked-contract-v1")
            .dimension("2x2 square two-component parametric FEM CSR assembler")
        .formulation("valid merge plus typed rejection of non-finite, out-of-bounds, and duplicate-boundary inputs")
        .tolerance("valid merge succeeds; invalid triplets reject")
            .seed("fem-prebaked-checked-contract:cpu:v1")
            .elapsed_ms(started.elapsed().as_millis() as u64)
    } else {
        QaResult::failed(CASE_ID, failures.join("; "))
            .requested_backend("cpu")
            .actual_backend("cpu")
            .metric("boundary_checks", checks)
            .reference("independent malformed-triplet admission and valid CSR merge contract")
            .reference_checksum("analytic/deterministic-fixture:fem-prebaked-checked-contract-v1")
            .dimension("2x2 square two-component parametric FEM CSR assembler")
        .formulation("valid merge plus typed rejection of non-finite, out-of-bounds, and duplicate-boundary inputs")
        .tolerance("valid merge succeeds; invalid triplets reject")
            .seed("fem-prebaked-checked-contract:cpu:v1")
            .elapsed_ms(started.elapsed().as_millis() as u64)
    };

    println!(
        "QA_RESULT: {}",
        result
            .to_json_line()
            .expect("prebaked QA result serializes")
    );
    if failures.is_empty() && checks == 4.0 {
        println!("SUCCESS");
    } else {
        println!("FAILED");
        std::process::exit(1);
    }
}
