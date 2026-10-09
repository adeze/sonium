//! Common bandwidth/roofline preflight for backend development.
//!
//! This benchmark answers the first question before tuning kernels:
//! what does this machine do for plain copy, device copy, BLAS-1
//! (scale/axpy/dot), GEMV, cast, gather, CSR SpMV, BEM regular-pair,
//! and call/sync overhead through the same Backend API used by solvers?

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use std::hint::black_box;

#[cfg(any(feature = "cuda", feature = "metal"))]
use sonium_compute::BemRegularPairIn;
use sonium_compute::backend::CsrRowBucketStats;
use sonium_compute::perf;
use sonium_compute::{Backend, CpuBackend, MemoryPool, MemoryPoolConfig};

const N: usize = 262_144;
const GEMV_M: usize = 1_024;
const GEMV_N: usize = 1_024;
const CSR_ROWS: usize = 16_384;
const GATHER_ROWS: usize = 65_536;
#[cfg(any(feature = "cuda", feature = "metal"))]
const BEM_PAIRS: usize = 16_384;

fn report(backend: &str, op: &str, shape: &str, model: perf::PerfModel) {
    eprintln!(
        "sonium_roofline_preflight backend={backend} op={op} shape={shape} bytes={} flops={} arithmetic_intensity={:.6} metadata=\"{}\"",
        model.bytes,
        model.flops,
        model.arithmetic_intensity(),
        perf::host_metadata()
    );
}

fn ramp(n: usize) -> Vec<f32> {
    (0..n).map(|i| (i as f32 * 0.0123).sin()).collect()
}

fn poisson_1d_csr(rows: usize) -> (Vec<i32>, Vec<i32>, Vec<f32>) {
    let mut row_ptr = Vec::with_capacity(rows + 1);
    let mut col_idx = Vec::with_capacity(rows * 3);
    let mut values = Vec::with_capacity(rows * 3);
    row_ptr.push(0);
    for row in 0..rows {
        if row > 0 {
            col_idx.push(row as i32 - 1);
            values.push(-1.0);
        }
        col_idx.push(row as i32);
        values.push(2.0);
        if row + 1 < rows {
            col_idx.push(row as i32 + 1);
            values.push(-1.0);
        }
        row_ptr.push(col_idx.len() as i32);
    }
    (row_ptr, col_idx, values)
}

#[cfg(any(feature = "cuda", feature = "metal"))]
fn regular_pairs(count: usize) -> Vec<BemRegularPairIn> {
    (0..count)
        .map(|i| {
            let t = i as f32 * 0.001;
            BemRegularPairIn::from_triangle(
                [0.2 + t.sin() * 0.1, 0.3, 1.0],
                [0.0, 0.0, 1.0],
                [2.0, 0.0, 0.0],
                [2.0, 0.5, 0.0],
                [2.0, 0.0, 0.5],
            )
        })
        .collect()
}

fn bench_backend_preflight<B: Backend>(c: &mut Criterion, backend_name: &'static str, backend: B) {
    let pool = MemoryPool::new(backend, MemoryPoolConfig::with_cap(1 << 30)).unwrap();
    let host = ramp(N);
    let mut src = pool.try_alloc::<f32>(N).unwrap();
    let mut dst = pool.try_alloc::<f32>(N).unwrap();
    pool.backend().copy_h2d(&host, src.raw_mut());
    pool.backend().copy_h2d(&host, dst.raw_mut());

    let mut copy = c.benchmark_group("roofline_preflight/copy");
    let copy_model = perf::copy_h2d_model::<f32>(N);
    report(backend_name, "copy_h2d", &N.to_string(), copy_model);
    copy.throughput(Throughput::Bytes(copy_model.bytes));
    copy.bench_function(BenchmarkId::new(backend_name, "h2d_f32"), |b| {
        b.iter(|| pool.backend().copy_h2d(black_box(&host), dst.raw_mut()));
    });

    let d2d_model = perf::copy_d2d_model::<f32>(N);
    report(backend_name, "copy_d2d", &N.to_string(), d2d_model);
    copy.throughput(Throughput::Bytes(d2d_model.bytes));
    copy.bench_function(BenchmarkId::new(backend_name, "d2d_f32"), |b| {
        b.iter(|| pool.backend().copy_d2d(src.raw(), dst.raw_mut()));
    });
    copy.finish();

    let mut blas1 = c.benchmark_group("roofline_preflight/blas1");
    let scale_model = perf::scale_model::<f32>(N);
    report(backend_name, "scale", &N.to_string(), scale_model);
    blas1.throughput(Throughput::Bytes(scale_model.bytes));
    blas1.bench_function(BenchmarkId::new(backend_name, "scale_f32"), |b| {
        b.iter(|| pool.backend().scale(black_box(0.75_f32), dst.raw_mut()));
    });

    let dot_model = perf::dot_model::<f32>(N);
    report(backend_name, "dot", &N.to_string(), dot_model);
    blas1.throughput(Throughput::Bytes(dot_model.bytes));
    blas1.bench_function(BenchmarkId::new(backend_name, "dot_f32"), |b| {
        b.iter(|| black_box(pool.backend().dot(src.raw(), dst.raw())));
    });

    let axpy_model = perf::axpy_model::<f32>(N);
    report(backend_name, "axpy", &N.to_string(), axpy_model);
    blas1.throughput(Throughput::Bytes(axpy_model.bytes));
    blas1.bench_function(BenchmarkId::new(backend_name, "axpy_f32"), |b| {
        b.iter(|| {
            pool.backend()
                .axpy(black_box(0.5_f32), src.raw(), dst.raw_mut())
        });
    });
    blas1.finish();

    let gemv_a_h = ramp(GEMV_M * GEMV_N);
    let gemv_x_h = ramp(GEMV_N);
    let gemv_y_h = ramp(GEMV_M);
    let mut gemv_a = pool.try_alloc::<f32>(GEMV_M * GEMV_N).unwrap();
    let mut gemv_x = pool.try_alloc::<f32>(GEMV_N).unwrap();
    let mut gemv_y = pool.try_alloc::<f32>(GEMV_M).unwrap();
    pool.backend().copy_h2d(&gemv_a_h, gemv_a.raw_mut());
    pool.backend().copy_h2d(&gemv_x_h, gemv_x.raw_mut());
    pool.backend().copy_h2d(&gemv_y_h, gemv_y.raw_mut());
    let gemv_model = perf::gemv_model::<f32>(GEMV_M, GEMV_N);
    report(
        backend_name,
        "gemv",
        &format!("{GEMV_M}x{GEMV_N}"),
        gemv_model,
    );
    let mut gemv = c.benchmark_group("roofline_preflight/gemv");
    gemv.throughput(Throughput::Bytes(gemv_model.bytes));
    gemv.bench_function(BenchmarkId::new(backend_name, "f32"), |b| {
        b.iter(|| {
            pool.backend().gemv(
                sonium_compute::Transpose::NoTrans,
                GEMV_M,
                GEMV_N,
                1.0_f32,
                gemv_a.raw(),
                GEMV_N,
                gemv_x.raw(),
                0.0_f32,
                gemv_y.raw_mut(),
            )
        });
    });
    gemv.finish();

    let mut upper = pool.try_alloc::<f64>(N).unwrap();
    let mut lower = pool.try_alloc::<f32>(N).unwrap();
    let mut roundtrip = pool.try_alloc::<f64>(N).unwrap();
    let upper_h: Vec<f64> = (0..N).map(|i| (i as f64 * 0.007).cos()).collect();
    pool.backend().copy_h2d(&upper_h, upper.raw_mut());
    let mut cast = c.benchmark_group("roofline_preflight/cast");
    let lower_model = perf::cast_model::<f64, f32>(N);
    report(backend_name, "cast_to_lower", &N.to_string(), lower_model);
    cast.throughput(Throughput::Bytes(lower_model.bytes));
    cast.bench_function(BenchmarkId::new(backend_name, "f64_to_f32"), |b| {
        b.iter(|| {
            pool.backend()
                .cast_to_lower::<f64>(upper.raw(), lower.raw_mut())
        });
    });

    let upper_model = perf::cast_model::<f32, f64>(N);
    report(backend_name, "cast_from_lower", &N.to_string(), upper_model);
    cast.throughput(Throughput::Bytes(upper_model.bytes));
    cast.bench_function(BenchmarkId::new(backend_name, "f32_to_f64"), |b| {
        b.iter(|| {
            pool.backend()
                .cast_from_lower::<f64>(lower.raw(), roundtrip.raw_mut())
        });
    });
    cast.finish();

    let value_count = GATHER_ROWS + 3;
    let values_h = ramp(value_count);
    let indices_h: Vec<i32> = (0..GATHER_ROWS)
        .flat_map(|i| [i as i32, i as i32 + 1, i as i32 + 2, i as i32 + 3])
        .collect();
    let weights_h = vec![0.25_f32; GATHER_ROWS * 4];
    let mut values = pool.try_alloc::<f32>(value_count).unwrap();
    let mut indices = pool.try_alloc::<i32>(GATHER_ROWS * 4).unwrap();
    let mut weights = pool.try_alloc::<f32>(GATHER_ROWS * 4).unwrap();
    let mut gathered = pool.try_alloc::<f32>(GATHER_ROWS).unwrap();
    pool.backend().copy_h2d(&values_h, values.raw_mut());
    pool.backend().copy_h2d(&indices_h, indices.raw_mut());
    pool.backend().copy_h2d(&weights_h, weights.raw_mut());
    let gather_model = perf::gather4_weighted_model::<f32>(GATHER_ROWS);
    report(
        backend_name,
        "gather4_weighted",
        &GATHER_ROWS.to_string(),
        gather_model,
    );
    let mut gather = c.benchmark_group("roofline_preflight/gather4");
    gather.throughput(Throughput::Bytes(gather_model.bytes));
    gather.bench_function(BenchmarkId::new(backend_name, "f32"), |b| {
        b.iter(|| {
            pool.backend().gather4_weighted(
                values.raw(),
                indices.raw(),
                weights.raw(),
                gathered.raw_mut(),
            )
        });
    });
    gather.finish();

    let (row_ptr_h, col_idx_h, values_h) = poisson_1d_csr(CSR_ROWS);
    let x_h = vec![1.0_f32; CSR_ROWS];
    let mut row_ptr = pool.try_alloc::<i32>(row_ptr_h.len()).unwrap();
    let mut col_idx = pool.try_alloc::<i32>(col_idx_h.len()).unwrap();
    let mut values = pool.try_alloc::<f32>(values_h.len()).unwrap();
    let mut x = pool.try_alloc::<f32>(CSR_ROWS).unwrap();
    let mut y = pool.try_alloc::<f32>(CSR_ROWS).unwrap();
    pool.backend().copy_h2d(&row_ptr_h, row_ptr.raw_mut());
    pool.backend().copy_h2d(&col_idx_h, col_idx.raw_mut());
    pool.backend().copy_h2d(&values_h, values.raw_mut());
    pool.backend().copy_h2d(&x_h, x.raw_mut());
    let prebaked = pool.backend().prebake_csr(
        CSR_ROWS,
        CSR_ROWS,
        row_ptr.raw(),
        col_idx.raw(),
        values.raw(),
    );
    let stats = CsrRowBucketStats::from_row_ptr(&row_ptr_h);
    let spmv_model = perf::csr_spmv_model_ncols::<f32>(stats, CSR_ROWS);
    report(
        backend_name,
        "csr_spmv",
        &format!("rows={CSR_ROWS} nnz={}", stats.nnz),
        spmv_model,
    );
    let mut spmv = c.benchmark_group("roofline_preflight/spmv");
    spmv.throughput(Throughput::Bytes(spmv_model.bytes));
    spmv.bench_function(BenchmarkId::new(backend_name, "csr_f32_prebaked"), |b| {
        b.iter(|| {
            pool.backend()
                .spmv_prebaked(&prebaked, x.raw(), y.raw_mut())
        });
    });
    spmv.finish();

    let mut overhead = c.benchmark_group("roofline_preflight/overhead");
    report(
        backend_name,
        "synchronize",
        "noop",
        perf::PerfModel::new(0, 0),
    );
    overhead.bench_function(BenchmarkId::new(backend_name, "synchronize"), |b| {
        b.iter(|| pool.backend().synchronize());
    });
    overhead.finish();
}

#[cfg(feature = "cuda")]
fn bench_cuda_bem_preflight(c: &mut Criterion, backend: &sonium_compute::CudaBackend) {
    if !backend.has_bem_regular_pair_c32() {
        eprintln!("CUDA BEM regular-pair kernel unavailable, skipping roofline preflight");
        return;
    }
    let pairs = regular_pairs(BEM_PAIRS);
    let model = perf::bem_regular_pair_c32_model(BEM_PAIRS);
    report(
        "cuda",
        "bem_regular_pair_c32",
        &BEM_PAIRS.to_string(),
        model,
    );
    let mut group = c.benchmark_group("roofline_preflight/bem_regular_pair");
    group.throughput(Throughput::Bytes(model.bytes));
    group.bench_function(BenchmarkId::new("cuda", "c32"), |b| {
        b.iter(|| {
            black_box(backend.bem_regular_pair_c32_batch(
                black_box(&pairs),
                black_box(1.0),
                black_box(1.0),
            ))
        });
    });
    group.finish();
}

#[cfg(feature = "metal")]
fn bench_metal_bem_preflight(c: &mut Criterion, backend: &sonium_compute::MetalBackend) {
    let pairs = regular_pairs(BEM_PAIRS);
    let model = perf::bem_regular_pair_c32_model(BEM_PAIRS);
    report(
        "metal",
        "bem_regular_pair_c32",
        &BEM_PAIRS.to_string(),
        model,
    );
    let mut group = c.benchmark_group("roofline_preflight/bem_regular_pair");
    group.throughput(Throughput::Bytes(model.bytes));
    group.bench_function(BenchmarkId::new("metal", "c32"), |b| {
        b.iter(|| {
            black_box(backend.bem_regular_pair_c32_batch(
                black_box(&pairs),
                black_box(1.0),
                black_box(1.0),
            ))
        });
    });
    group.finish();
}

fn bench_roofline_preflight(c: &mut Criterion) {
    bench_backend_preflight(c, "cpu", CpuBackend);

    #[cfg(feature = "cuda")]
    {
        match sonium_compute::CudaBackend::new_default() {
            Ok(backend) => {
                bench_cuda_bem_preflight(c, &backend);
                bench_backend_preflight(c, "cuda", backend);
            }
            Err(e) => eprintln!("CUDA unavailable, skipping roofline preflight: {e}"),
        }
    }

    #[cfg(feature = "metal")]
    {
        match sonium_compute::MetalBackend::new() {
            Ok(backend) => {
                bench_metal_bem_preflight(c, &backend);
                bench_backend_preflight(c, "metal", backend);
            }
            Err(e) => eprintln!("Metal unavailable, skipping roofline preflight: {e}"),
        }
    }
}

criterion_group!(roofline_preflight, bench_roofline_preflight);
criterion_main!(roofline_preflight);
