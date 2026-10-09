//! Seeded boundary-fuzz campaign for solver input and wire-state contracts.
//!
//! This is deliberately deterministic and CPU-only.  It exercises the checked
//! sparse constructors, dense/sparse factorization admission, GMRES option admission, and the portable GMRES
//! checkpoint decoder with malformed values generated from one reproducible
//! seed.  The campaign is not a probabilistic correctness proof; it is a
//! release-facing panic/typed-rejection smoke gate for untrusted boundaries.

use sonium_qa::QaResult;
use sonium_solvers::dense::DenseMatrix;
use sonium_solvers::{CsrMatrix, GmresCheckpoint, GmresConfig, TripletMatrix, try_gmres};
use sonium_solvers::{
    DenseDirectConfig, DenseDirectMethod, MatrixClass, SparseDirectConfig, SparseDirectMethod,
    factorize_dense, factorize_sparse,
};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::time::Instant;

const CASE_ID: &str = "solvers.boundary-fuzz.v1";
const SEED: u64 = 0x5eed_f022_2026;
const CHECKSUM: &str = "analytic/seeded-boundary-fuzz:solver-inputs-wire-v1";

#[derive(Debug, Default)]
struct Counters {
    triplet_cases: usize,
    triplet_valid: usize,
    triplet_rejections: usize,
    triplet_panics: usize,
    csr_cases: usize,
    csr_valid: usize,
    csr_rejections: usize,
    csr_panics: usize,
    option_cases: usize,
    option_valid: usize,
    option_rejections: usize,
    option_panics: usize,
    checkpoint_cases: usize,
    checkpoint_round_trip: usize,
    checkpoint_rejections: usize,
    checkpoint_panics: usize,
    factor_cases: usize,
    factor_valid: usize,
    factor_rejections: usize,
    factor_panics: usize,
}

#[derive(Clone, Copy)]
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_f491_4f6c_dd1d)
    }

    fn index(&mut self, upper: usize) -> usize {
        (self.next() as usize) % upper.max(1)
    }
}

fn fuzz_triplets(rng: &mut Rng, counters: &mut Counters, cases: usize) -> usize {
    let mut unexpected = 0;
    for _ in 0..cases {
        let mode = rng.index(8);
        let dimension = 1 + rng.index(4);
        let mut triplets = TripletMatrix::<f64>::new(dimension, dimension);
        let malformed = mode != 0;
        match mode {
            0 => {
                for _ in 0..(1 + rng.index(5)) {
                    triplets.rows.push(rng.index(dimension));
                    triplets.cols.push(rng.index(dimension));
                    triplets.values.push(0.25 + rng.index(7) as f64);
                }
            }
            1 => triplets.rows.push(0),
            2 => {
                triplets.rows.push(0);
                triplets.cols.push(0);
            }
            3 => {
                triplets.rows.push(dimension);
                triplets.cols.push(0);
                triplets.values.push(1.0);
            }
            4 => {
                triplets.rows.push(0);
                triplets.cols.push(dimension);
                triplets.values.push(1.0);
            }
            5 => {
                triplets.rows.push(0);
                triplets.cols.push(0);
                triplets.values.push(f64::NAN);
            }
            6 => {
                triplets.rows.push(0);
                triplets.cols.push(0);
                triplets.values.push(f64::INFINITY);
            }
            7 => {
                triplets.rows.extend_from_slice(&[0, 0]);
                triplets.cols.extend_from_slice(&[0, 0]);
                triplets.values.extend_from_slice(&[f64::MAX, f64::MAX]);
            }
            _ => unreachable!(),
        }
        counters.triplet_cases += 1;
        let outcome = catch_unwind(AssertUnwindSafe(|| triplets.try_to_csr()));
        match outcome {
            Ok(Ok(_)) => {
                counters.triplet_valid += 1;
                if malformed {
                    unexpected += 1;
                }
            }
            Ok(Err(_)) => {
                counters.triplet_rejections += 1;
                if !malformed {
                    unexpected += 1;
                }
            }
            Err(_) => counters.triplet_panics += 1,
        }
    }
    unexpected
}

fn fuzz_csr(rng: &mut Rng, counters: &mut Counters, cases: usize) -> usize {
    let mut unexpected = 0;
    for _ in 0..cases {
        let mode = rng.index(8);
        let malformed = mode != 0;
        let (rows, cols, values, num_rows, num_cols) = match mode {
            0 => (vec![0, 1, 2], vec![0, 1], vec![1.0, 1.0], 2, 2),
            1 => (vec![0, 1], vec![0], vec![1.0], 2, 2),
            2 => (vec![0, 2, 1], vec![0, 1], vec![1.0, 1.0], 2, 2),
            3 => (vec![0, 1, 3], vec![0, 1], vec![1.0, 1.0], 2, 2),
            4 => (vec![0, 1, 2], vec![0, 2], vec![1.0, 1.0], 2, 2),
            5 => (vec![0, 1, 2], vec![0, 1], vec![f64::NAN, 1.0], 2, 2),
            6 => (vec![0, 1, 2], vec![0, 1], vec![f64::INFINITY, 1.0], 2, 2),
            // Checked construction must reject row-count overflow before
            // trying to allocate a row-pointer array.
            7 => (Vec::new(), Vec::new(), Vec::new(), usize::MAX, 0),
            _ => unreachable!(),
        };
        counters.csr_cases += 1;
        let outcome = catch_unwind(AssertUnwindSafe(|| {
            CsrMatrix::<f64>::try_from_raw_parts(num_rows, num_cols, rows, cols, values)
        }));
        match outcome {
            Ok(Ok(_)) => {
                counters.csr_valid += 1;
                if malformed {
                    unexpected += 1;
                }
            }
            Ok(Err(_)) => {
                counters.csr_rejections += 1;
                if !malformed {
                    unexpected += 1;
                }
            }
            Err(_) => counters.csr_panics += 1,
        }
    }
    unexpected
}

fn fuzz_options(rng: &mut Rng, counters: &mut Counters, cases: usize) -> usize {
    let identity = CsrMatrix::<f64>::identity(2);
    let rhs = [1.0, -0.5];
    let mut unexpected = 0;
    for _ in 0..cases {
        let mode = rng.index(7);
        let mut config = GmresConfig::<f64>::default();
        let invalid = match mode {
            0 => false,
            1 => {
                config.max_iterations = 0;
                true
            }
            2 => {
                config.restart = 0;
                true
            }
            3 => {
                config.tolerance = f64::NAN;
                true
            }
            4 => {
                config.tolerance = f64::INFINITY;
                true
            }
            5 => {
                config.min_restart = Some(0);
                true
            }
            6 => {
                config.min_restart = Some(4);
                config.max_restart = Some(2);
                true
            }
            _ => unreachable!(),
        };
        if mode == 0 && rng.next() & 1 == 0 {
            // Zero is explicitly normalized to one by the checked GMRES
            // implementation; keep it in the valid fuzz population.
            config.convergence_check_interval = 0;
        }
        counters.option_cases += 1;
        let outcome = catch_unwind(AssertUnwindSafe(|| try_gmres(&identity, &rhs, &config)));
        match outcome {
            Ok(Ok(_)) => {
                counters.option_valid += 1;
                if invalid {
                    unexpected += 1;
                }
            }
            Ok(Err(_)) => {
                counters.option_rejections += 1;
                if !invalid {
                    unexpected += 1;
                }
            }
            Err(_) => counters.option_panics += 1,
        }
    }
    unexpected
}

fn fuzz_factorizations(rng: &mut Rng, counters: &mut Counters, cases: usize) -> usize {
    let mut unexpected = 0;
    for _ in 0..cases {
        let mode = rng.index(8);
        let malformed = mode != 0;
        let (rows, cols, dense_values, row_ptrs, col_indices, sparse_values) = match mode {
            0 => (
                2,
                2,
                vec![4.0, 1.0, 2.0, 3.0],
                vec![0, 2, 4],
                vec![0, 1, 0, 1],
                vec![4.0, 1.0, 2.0, 3.0],
            ),
            1 => (
                2,
                3,
                vec![4.0, 1.0, 0.0, 2.0, 3.0, 1.0],
                vec![0, 2, 4],
                vec![0, 1, 1, 2],
                vec![4.0, 1.0, 2.0, 3.0],
            ),
            2 => (
                2,
                2,
                vec![1.0, 2.0, 2.0, 4.0],
                vec![0, 2, 4],
                vec![0, 1, 0, 1],
                vec![1.0, 2.0, 2.0, 4.0],
            ),
            3 => (
                2,
                2,
                vec![4.0, 1.0, 2.0, 3.0],
                vec![0, 2, 4],
                vec![0, 1, 0, 1],
                vec![4.0, 1.0, 2.0, 3.0],
            ),
            4 => (
                2,
                2,
                vec![4.0, 1.0, 2.0, 3.0],
                vec![0, 2, 4],
                vec![0, 1, 0, 1],
                vec![4.0, 1.0, 2.0, 3.0],
            ),
            5 => (
                2,
                2,
                vec![4.0, 1.0, 2.0, 3.0],
                vec![0, 2, 4],
                vec![0, 1, 0, 1],
                vec![4.0, 1.0, 2.0, 3.0],
            ),
            6 => (
                2,
                2,
                vec![4.0, 1.0, 2.0, 3.0],
                vec![0, 2, 4],
                vec![0, 1, 0, 1],
                vec![4.0, 1.0, 2.0, 3.0],
            ),
            7 => (0, 2, Vec::new(), vec![0], Vec::new(), Vec::new()),
            _ => unreachable!(),
        };

        let mut dense = DenseMatrix::from_row_major_vec(rows, cols, dense_values);
        let mut dense_config = DenseDirectConfig {
            matrix_class: MatrixClass::General,
            method: DenseDirectMethod::Lu,
            ..Default::default()
        };
        if mode == 5 {
            dense_config.equilibrate = true;
        } else if mode == 6 {
            dense_config.method = DenseDirectMethod::Cholesky;
        }

        let mut sparse =
            CsrMatrix::from_raw_parts(rows, cols, row_ptrs, col_indices, sparse_values);
        if mode == 3 {
            sparse.values[0] = f64::NAN;
            dense.as_mut_slice()[0] = f64::NAN;
        } else if mode == 4 {
            sparse.values[0] = f64::INFINITY;
            dense.as_mut_slice()[0] = f64::INFINITY;
        }
        let mut sparse_config = SparseDirectConfig {
            matrix_class: MatrixClass::General,
            method: SparseDirectMethod::SimplicialLu,
            ..Default::default()
        };
        if mode == 5 {
            sparse_config.memory_budget_bytes = Some(0);
        } else if mode == 6 {
            sparse_config.method = SparseDirectMethod::SimplicialCholesky;
        }

        counters.factor_cases += 1;
        macro_rules! record_factor_outcome {
            ($outcome:expr) => {
                match $outcome {
                    Ok(Ok(())) => {
                        counters.factor_valid += 1;
                        if malformed {
                            unexpected += 1;
                        }
                    }
                    Ok(Err(_)) => {
                        counters.factor_rejections += 1;
                        if !malformed {
                            unexpected += 1;
                        }
                    }
                    Err(_) => counters.factor_panics += 1,
                }
            };
        }
        record_factor_outcome!(catch_unwind(AssertUnwindSafe(|| {
            factorize_dense(&dense, &dense_config).map(|_| ())
        })));
        record_factor_outcome!(catch_unwind(AssertUnwindSafe(|| {
            factorize_sparse(&sparse, &sparse_config, None).map(|_| ())
        })));
    }
    unexpected
}

fn write_u64(bytes: &mut [u8], offset: usize, value: u64) {
    bytes[offset..offset + 8].copy_from_slice(&value.to_le_bytes());
}

fn fuzz_checkpoints(rng: &mut Rng, counters: &mut Counters, cases: usize) -> usize {
    let checkpoint =
        GmresCheckpoint::<f64>::from_iterate(&[0.25, -0.5]).expect("finite checkpoint fixture");
    let bytes = checkpoint
        .to_checkpoint_bytes()
        .expect("checkpoint fixture must serialize");
    let restored = GmresCheckpoint::<f64>::from_checkpoint_bytes(&bytes)
        .expect("checkpoint fixture must round-trip");
    if restored.solution() != checkpoint.solution()
        || restored.dimension() != checkpoint.dimension()
    {
        return 1;
    }
    counters.checkpoint_round_trip = 1;
    let mut unexpected = 0;
    for index in 0..cases {
        let mode = index % 8;
        let mut mutated = bytes.clone();
        match mode {
            0 => mutated.truncate(rng.index(bytes.len())),
            1 => mutated.push(0xa5),
            2 => mutated[0] ^= 0x01,
            3 => mutated[8] = 0xff,
            4 => mutated[13] = 1,
            5 => mutated[10] = 0,
            6 => write_u64(&mut mutated, 14, u64::MAX),
            7 => mutated[38..46].copy_from_slice(&f64::NAN.to_le_bytes()),
            _ => unreachable!(),
        }
        counters.checkpoint_cases += 1;
        let outcome = catch_unwind(AssertUnwindSafe(|| {
            GmresCheckpoint::<f64>::from_checkpoint_bytes(&mutated)
        }));
        match outcome {
            Ok(Ok(_)) => unexpected += 1,
            Ok(Err(_)) => counters.checkpoint_rejections += 1,
            Err(_) => counters.checkpoint_panics += 1,
        }
    }
    unexpected
}

fn run() -> Result<Counters, String> {
    let mut rng = Rng(SEED);
    let mut counters = Counters::default();
    let mut unexpected = 0;
    unexpected += fuzz_triplets(&mut rng, &mut counters, 256);
    unexpected += fuzz_csr(&mut rng, &mut counters, 256);
    unexpected += fuzz_options(&mut rng, &mut counters, 256);
    unexpected += fuzz_checkpoints(&mut rng, &mut counters, 256);
    unexpected += fuzz_factorizations(&mut rng, &mut counters, 256);
    if unexpected != 0 {
        return Err(format!(
            "seeded fuzz campaign observed {unexpected} unexpected acceptances"
        ));
    }
    if counters.triplet_panics
        + counters.csr_panics
        + counters.option_panics
        + counters.checkpoint_panics
        + counters.factor_panics
        != 0
    {
        return Err(format!("seeded fuzz campaign panicked: {counters:?}"));
    }
    Ok(counters)
}

fn main() {
    let started = Instant::now();
    match run() {
        Ok(counters) => {
            let result = QaResult::passed(CASE_ID)
                .seed(format!("{SEED:#x}"))
                .input_checksum(CHECKSUM)
                .reference_checksum(CHECKSUM)
                .metric("triplet_cases", counters.triplet_cases as f64)
                .metric("triplet_rejections", counters.triplet_rejections as f64)
                .metric("csr_cases", counters.csr_cases as f64)
                .metric("csr_rejections", counters.csr_rejections as f64)
                .metric("gmres_option_cases", counters.option_cases as f64)
                .metric("gmres_option_rejections", counters.option_rejections as f64)
                .metric("checkpoint_cases", counters.checkpoint_cases as f64)
                .metric(
                    "checkpoint_rejections",
                    counters.checkpoint_rejections as f64,
                )
                .metric(
                    "checkpoint_round_trip",
                    counters.checkpoint_round_trip as f64,
                )
                .metric("factor_cases", counters.factor_cases as f64)
                .metric("factor_rejections", counters.factor_rejections as f64)
                .metric("panic_count", 0.0)
                .minimum("triplet_rejections", 200.0)
                .minimum("csr_rejections", 200.0)
                .minimum("gmres_option_rejections", 200.0)
                .minimum("checkpoint_rejections", 200.0)
                .minimum("checkpoint_round_trip", 1.0)
                .minimum("factor_rejections", 400.0)
                .minimum("panic_count", 0.0)
                .elapsed_ms(started.elapsed().as_millis() as u64);
            println!("QA_RESULT: {}", result.to_json_line().expect("QA JSON"));
            println!("SUCCESS");
        }
        Err(error) => {
            let result = QaResult::failed(CASE_ID, error.clone())
                .seed(format!("{SEED:#x}"))
                .input_checksum(CHECKSUM)
                .reference_checksum(CHECKSUM)
                .elapsed_ms(started.elapsed().as_millis() as u64);
            println!("QA_RESULT: {}", result.to_json_line().expect("QA JSON"));
            eprintln!("FAILED: {error}");
            std::process::exit(1);
        }
    }
}
