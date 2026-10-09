//! Roofline preflight for solver bandwidth baselines.
//!
//! This bench records simple dense-vector bandwidth and CSR SpMV samples.
//! It is intentionally small and portable: use it before larger solver
//! comparisons to understand the local memory-bandwidth ceiling.

#[path = "matrices/mod.rs"]
mod matrices;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use sonium_solvers::ComplexField;
use std::time::Duration;

#[cfg(feature = "cli")]
use sonium_solvers::perf::{self, CsrSpmvByteModel, PerfRun};

const VECTOR_LEN: usize = 1 << 20;
const SAMPLE_TIME: Duration = Duration::from_millis(100);

#[cfg(feature = "cli")]
struct RooflineMeta<'a> {
    group: &'a str,
    id: &'a str,
    operation: &'a str,
    scalar: &'a str,
    bytes_per_iter: u64,
    n: Option<usize>,
    nnz: Option<usize>,
    csr_byte_model: Option<CsrSpmvByteModel>,
    notes: &'a str,
}

#[cfg(feature = "cli")]
fn record_roofline<F>(meta: RooflineMeta<'_>, mut f: F)
where
    F: FnMut(),
{
    if perf::perf_json_path().is_none() {
        return;
    }

    let sample = perf::measure_for(SAMPLE_TIME, 500, || {
        f();
    });

    let mut run = PerfRun::new("roofline_preflight_bench", meta.group, meta.id)
        .with_bytes(meta.bytes_per_iter)
        .with_timing(sample);
    run.operation = meta.operation.to_string();
    run.scalar = meta.scalar.to_string();
    run.nrows = meta.n;
    run.ncols = meta.n;
    run.nnz = meta.nnz;
    if let Some(model) = meta.csr_byte_model {
        run = run.with_csr_spmv_byte_model(model);
        run.bandwidth_limited_by = Some("csr_spmv_memory_traffic".to_string());
    }
    run.notes = Some(meta.notes.to_string());

    if let Err(err) = perf::append_run_from_env(run) {
        eprintln!("failed to write {} report: {err}", perf::PERF_JSON_ENV);
    }
}

fn bench_dense_roofline(c: &mut Criterion) {
    let mut group = c.benchmark_group("roofline/dense");
    group.sample_size(50);
    group.measurement_time(Duration::from_secs(5));

    let x = vec![1.0_f64; VECTOR_LEN];
    let mut y = vec![2.0_f64; VECTOR_LEN];
    let elem = std::mem::size_of::<f64>();

    let copy_bytes = (2 * VECTOR_LEN * elem) as u64;
    group.throughput(Throughput::Bytes(copy_bytes));
    #[cfg(feature = "cli")]
    record_roofline(
        RooflineMeta {
            group: "roofline/dense",
            id: "copy_f64",
            operation: "copy",
            scalar: "f64",
            bytes_per_iter: copy_bytes,
            n: Some(VECTOR_LEN),
            nnz: None,
            csr_byte_model: None,
            notes: "read x, write y",
        },
        || y.copy_from_slice(&x),
    );
    group.bench_function(BenchmarkId::from_parameter("copy_f64"), |b| {
        b.iter(|| y.copy_from_slice(std::hint::black_box(&x)))
    });

    let fill_bytes = (VECTOR_LEN * elem) as u64;
    group.throughput(Throughput::Bytes(fill_bytes));
    #[cfg(feature = "cli")]
    record_roofline(
        RooflineMeta {
            group: "roofline/dense",
            id: "fill_f64",
            operation: "fill",
            scalar: "f64",
            bytes_per_iter: fill_bytes,
            n: Some(VECTOR_LEN),
            nnz: None,
            csr_byte_model: None,
            notes: "write y",
        },
        || y.fill(3.0),
    );
    group.bench_function(BenchmarkId::from_parameter("fill_f64"), |b| {
        b.iter(|| y.fill(std::hint::black_box(3.0)))
    });

    let axpy_bytes = (3 * VECTOR_LEN * elem) as u64;
    group.throughput(Throughput::Bytes(axpy_bytes));
    #[cfg(feature = "cli")]
    record_roofline(
        RooflineMeta {
            group: "roofline/dense",
            id: "axpy_f64",
            operation: "axpy",
            scalar: "f64",
            bytes_per_iter: axpy_bytes,
            n: Some(VECTOR_LEN),
            nnz: None,
            csr_byte_model: None,
            notes: "read x, read y, write y",
        },
        || <f64 as ComplexField>::vec_axpy(0.5, &x, &mut y),
    );
    group.bench_function(BenchmarkId::from_parameter("axpy_f64"), |b| {
        b.iter(|| <f64 as ComplexField>::vec_axpy(std::hint::black_box(0.5), &x, &mut y))
    });

    let dot_bytes = (2 * VECTOR_LEN * elem) as u64;
    group.throughput(Throughput::Bytes(dot_bytes));
    #[cfg(feature = "cli")]
    record_roofline(
        RooflineMeta {
            group: "roofline/dense",
            id: "dot_f64",
            operation: "dot",
            scalar: "f64",
            bytes_per_iter: dot_bytes,
            n: Some(VECTOR_LEN),
            nnz: None,
            csr_byte_model: None,
            notes: "read x and y",
        },
        || {
            std::hint::black_box(<f64 as ComplexField>::vec_dot(&x, &y));
        },
    );
    group.bench_function(BenchmarkId::from_parameter("dot_f64"), |b| {
        b.iter(|| std::hint::black_box(<f64 as ComplexField>::vec_dot(&x, &y)))
    });

    let nrm2_bytes = (VECTOR_LEN * elem) as u64;
    group.throughput(Throughput::Bytes(nrm2_bytes));
    #[cfg(feature = "cli")]
    record_roofline(
        RooflineMeta {
            group: "roofline/dense",
            id: "nrm2_f64",
            operation: "nrm2",
            scalar: "f64",
            bytes_per_iter: nrm2_bytes,
            n: Some(VECTOR_LEN),
            nnz: None,
            csr_byte_model: None,
            notes: "read x",
        },
        || {
            std::hint::black_box(<f64 as ComplexField>::vec_norm(&x));
        },
    );
    group.bench_function(BenchmarkId::from_parameter("nrm2_f64"), |b| {
        b.iter(|| std::hint::black_box(<f64 as ComplexField>::vec_norm(&x)))
    });

    group.finish();
}

fn bench_spmv_roofline(c: &mut Criterion) {
    let mut group = c.benchmark_group("roofline/spmv");
    group.sample_size(50);
    group.measurement_time(Duration::from_secs(5));

    let mut cases = Vec::from([(
        "poisson_2d",
        matrices::poisson_2d(matrices::grid_size_2d(100_000)),
    )]);
    for (name, row_nnz) in [
        ("synthetic_short_rows_3nnz", 3usize),
        ("synthetic_medium_rows_15nnz", 15usize),
        ("synthetic_long_rows_63nnz", 63usize),
    ] {
        cases.push((name, matrices::fixed_row_nnz(100_000, row_nnz)));
    }

    for (name, matrix) in cases {
        let rows = matrix.num_rows;
        let x = vec![1.0_f64; rows];
        let csr_byte_model = {
            #[cfg(feature = "cli")]
            {
                perf::csr_spmv_byte_model(
                    rows,
                    rows,
                    matrix.nnz(),
                    std::mem::size_of::<f64>(),
                    std::mem::size_of::<usize>(),
                )
            }
            #[cfg(not(feature = "cli"))]
            {
                ()
            }
        };
        let bytes = {
            #[cfg(feature = "cli")]
            {
                csr_byte_model.total_bytes
            }
            #[cfg(not(feature = "cli"))]
            {
                (8 * matrix.nnz() + 8 * matrix.nnz() + 8 * (rows + 1) + 16 * rows) as u64
            }
        };

        group.throughput(Throughput::Bytes(bytes));
        #[cfg(feature = "cli")]
        record_roofline(
            RooflineMeta {
                group: "roofline/spmv",
                id: name,
                operation: "spmv",
                scalar: "f64",
                bytes_per_iter: bytes,
                n: Some(rows),
                nnz: Some(matrix.nnz()),
                csr_byte_model: Some(csr_byte_model.clone()),
                notes: "CSR f64 SpMV byte model includes usize indices; synthetic cases isolate row length effects",
            },
            || {
                std::hint::black_box(matrix.matvec(&x));
            },
        );
        group.bench_with_input(
            BenchmarkId::new(name, rows),
            &(matrix, x),
            |b, (matrix, x)| b.iter(|| std::hint::black_box(matrix.matvec(x))),
        );
    }

    group.finish();
}

criterion_group!(
    name = roofline_preflight;
    config = Criterion::default();
    targets = bench_dense_roofline, bench_spmv_roofline
);
criterion_main!(roofline_preflight);
