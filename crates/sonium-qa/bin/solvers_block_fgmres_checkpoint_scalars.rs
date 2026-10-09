use num_complex::{Complex32, Complex64};
use sonium_qa::QaResult;
use sonium_solvers::AccuracyStatus;
use sonium_solvers::dense::DenseMatrix;
use sonium_solvers::iterative::{BlockFgmresSolution, BlockGmresSolution, KrylovOpCounts};
use std::time::Instant;

const CASE_ID: &str = "solvers.block-fgmres-checkpoint-scalars.v1";
const INPUT_CHECKSUM: &str = "analytic/deterministic-fixture:block-fgmres-checkpoint-scalars-v1";
const TOLERANCE: f64 = 1.0e-12;

fn counts() -> KrylovOpCounts {
    KrylovOpCounts {
        spmv_count: 3,
        preconditioner_count: 2,
        dot_reductions: 4,
        norm_count: 5,
        axpy_passes: 6,
        basis_bytes_touched: 128,
        synchronization_points: 7,
        exact_residual_checks: 8,
    }
}

fn f32_solution() -> BlockFgmresSolution<f32> {
    BlockFgmresSolution {
        x: DenseMatrix::from_row_major_vec(2, 2, vec![1.0, 2.0, -1.0, 0.5]),
        iterations: 6,
        residuals: vec![1.0e-5, 2.0e-5],
        true_residuals: vec![3.0e-6, 4.0e-6],
        backward_errors: vec![5.0e-7, 6.0e-7],
        attainable_accuracy: vec![AccuracyStatus::Attained, AccuracyStatus::NotAttained],
        converged: false,
        op_counts: counts(),
    }
}

fn complex32_solution() -> BlockFgmresSolution<Complex32> {
    BlockFgmresSolution {
        x: DenseMatrix::from_row_major_vec(
            1,
            2,
            vec![Complex32::new(1.0, -2.0), Complex32::new(0.25, 0.5)],
        ),
        iterations: 9,
        residuals: vec![1.0e-4, 2.0e-4],
        true_residuals: vec![3.0e-5, 4.0e-5],
        backward_errors: vec![5.0e-6, 6.0e-6],
        attainable_accuracy: vec![AccuracyStatus::NotChecked, AccuracyStatus::Attained],
        converged: true,
        op_counts: counts(),
    }
}

fn complex64_solution() -> BlockFgmresSolution<Complex64> {
    BlockFgmresSolution {
        x: DenseMatrix::from_row_major_vec(
            2,
            1,
            vec![Complex64::new(1.0, 2.0), Complex64::new(-0.5, 0.25)],
        ),
        iterations: 11,
        residuals: vec![1.0e-8],
        true_residuals: vec![2.0e-9],
        backward_errors: vec![3.0e-10],
        attainable_accuracy: vec![AccuracyStatus::Attained],
        converged: true,
        op_counts: counts(),
    }
}

fn run() -> Result<(f64, f64, f64, f64, f64, f64, f64), String> {
    let f32_source = f32_solution();
    let f32_bytes = f32_source
        .to_typed_checkpoint_bytes()
        .map_err(|error| format!("f32 encode failed: {error}"))?;
    let f32_restored = BlockFgmresSolution::<f32>::from_typed_checkpoint_bytes(&f32_bytes)
        .map_err(|error| format!("f32 decode failed: {error}"))?;
    let f32_error = f32_source
        .x
        .as_slice()
        .iter()
        .zip(f32_restored.x.as_slice())
        .map(|(left, right)| f64::from((*left - *right).abs()))
        .fold(0.0, f64::max);
    let f32_round_trip = f64::from(
        f32_source.iterations == f32_restored.iterations
            && f32_source.op_counts == f32_restored.op_counts
            && f32_source.attainable_accuracy == f32_restored.attainable_accuracy,
    );

    let complex32_source = complex32_solution();
    let complex32_bytes = complex32_source
        .to_typed_checkpoint_bytes()
        .map_err(|error| format!("Complex32 encode failed: {error}"))?;
    let complex32_restored =
        BlockFgmresSolution::<Complex32>::from_typed_checkpoint_bytes(&complex32_bytes)
            .map_err(|error| format!("Complex32 decode failed: {error}"))?;
    let complex32_error = complex32_source
        .x
        .as_slice()
        .iter()
        .zip(complex32_restored.x.as_slice())
        .map(|(left, right)| f64::from((*left - *right).norm()))
        .fold(0.0, f64::max);

    let complex64_source = complex64_solution();
    let complex64_bytes = complex64_source
        .to_typed_checkpoint_bytes()
        .map_err(|error| format!("Complex64 encode failed: {error}"))?;
    let complex64_restored =
        BlockFgmresSolution::<Complex64>::from_typed_checkpoint_bytes(&complex64_bytes)
            .map_err(|error| format!("Complex64 decode failed: {error}"))?;
    let complex64_error = complex64_source
        .x
        .as_slice()
        .iter()
        .zip(complex64_restored.x.as_slice())
        .map(|(left, right)| (*left - *right).norm())
        .fold(0.0, f64::max);

    let gmres_source = BlockGmresSolution {
        x: complex64_source.x.clone(),
        iterations: complex64_source.iterations,
        residuals: complex64_source.residuals.clone(),
        true_residuals: complex64_source.true_residuals.clone(),
        backward_errors: complex64_source.backward_errors.clone(),
        attainable_accuracy: complex64_source.attainable_accuracy.clone(),
        converged: complex64_source.converged,
        op_counts: complex64_source.op_counts,
    };
    let gmres_bytes = gmres_source
        .to_typed_checkpoint_bytes()
        .map_err(|error| format!("block GMRES encode failed: {error}"))?;
    let gmres_restored = BlockGmresSolution::<Complex64>::from_typed_checkpoint_bytes(&gmres_bytes)
        .map_err(|error| format!("block GMRES decode failed: {error}"))?;
    let block_gmres_round_trip = f64::from(
        gmres_source.x.as_slice() == gmres_restored.x.as_slice()
            && gmres_source.true_residuals == gmres_restored.true_residuals
            && gmres_source.op_counts == gmres_restored.op_counts,
    );

    let cross_family_rejection =
        f64::from(BlockFgmresSolution::<f64>::from_typed_checkpoint_bytes(&f32_bytes).is_err());
    let mut trailing = complex64_bytes.clone();
    trailing.push(0xA5);
    let corruption_rejection = f64::from(
        BlockFgmresSolution::<Complex64>::from_typed_checkpoint_bytes(&trailing).is_err(),
    );
    let mut non_finite = f32_solution();
    non_finite.x.as_mut_slice()[0] = f32::NAN;
    let non_finite_rejection = f64::from(non_finite.to_typed_checkpoint_bytes().is_err());

    Ok((
        f32_round_trip,
        f32_error,
        complex32_error,
        complex64_error,
        block_gmres_round_trip,
        cross_family_rejection,
        corruption_rejection + non_finite_rejection,
    ))
}

fn main() {
    let started = Instant::now();
    match run() {
        Ok((
            f32_round_trip,
            f32_error,
            complex32_error,
            complex64_error,
            block_gmres_round_trip,
            cross_family,
            rejects,
        )) => {
            let result = QaResult::passed(CASE_ID)
                .requested_backend("cpu")
                .actual_backend("cpu")
                .reference("independent typed block-GMRES/FGMRES packet fixtures")
                .input_checksum(INPUT_CHECKSUM)
                .reference_checksum(INPUT_CHECKSUM)
                .dimension("2x2, 1x2, and 2x1 block GMRES/FGMRES packets across f32/Complex32/Complex64")
                .tolerance("scalar and block-GMRES round-trip = 1; coordinate errors <= 1e-12; cross-family = 1; corruption/non-finite = 2")
                .metric("f32_round_trip", f32_round_trip)
                .metric("f32_max_error", f32_error)
                .metric("complex32_max_error", complex32_error)
                .metric("complex64_max_error", complex64_error)
                .metric("block_gmres_round_trip", block_gmres_round_trip)
                .metric("cross_family_rejection", cross_family)
                .metric("corruption_and_nonfinite_rejection", rejects)
                .minimum("f32_round_trip", 1.0)
                .maximum("f32_max_error", TOLERANCE)
                .maximum("complex32_max_error", TOLERANCE)
                .maximum("complex64_max_error", TOLERANCE)
                .minimum("block_gmres_round_trip", 1.0)
                .minimum("cross_family_rejection", 1.0)
                .minimum("corruption_and_nonfinite_rejection", 2.0)
                .elapsed_ms(started.elapsed().as_millis() as u64);
            println!("QA_RESULT: {}", result.to_json_line().expect("QA JSON"));
            println!("SUCCESS");
        }
        Err(error) => {
            let result = QaResult::failed(CASE_ID, error)
                .requested_backend("cpu")
                .actual_backend("cpu")
                .input_checksum(INPUT_CHECKSUM)
                .reference_checksum(INPUT_CHECKSUM)
                .elapsed_ms(started.elapsed().as_millis() as u64);
            eprintln!("QA_RESULT: {}", result.to_json_line().expect("QA JSON"));
            std::process::exit(1);
        }
    }
}
