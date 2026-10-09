use std::sync::OnceLock;

use criterion::{BenchmarkGroup, Throughput, measurement::WallTime};

pub use sonium_compute::perf::PerfModel;

static HOST_METADATA: OnceLock<String> = OnceLock::new();

pub fn host_metadata() -> &'static str {
    HOST_METADATA
        .get_or_init(sonium_compute::perf::host_metadata)
        .as_str()
}

pub fn report_model(bench: &str, scalar: &str, shape: &str, model: PerfModel) {
    report_model_with_variant(bench, scalar, shape, "generic", "exact", model);
}

pub fn report_model_with_variant(
    bench: &str,
    scalar: &str,
    shape: &str,
    variant: &str,
    precision: &str,
    model: PerfModel,
) {
    eprintln!(
        "sonium_perf_model bench={bench} scalar={scalar} shape={shape} variant={variant} precision={precision} bytes={} flops={} arithmetic_intensity={:.6} metadata=\"{}\"",
        model.bytes,
        model.flops,
        model.arithmetic_intensity(),
        host_metadata()
    );
}

#[allow(dead_code)]
pub fn set_bytes_throughput(
    group: &mut BenchmarkGroup<'_, WallTime>,
    bench: &str,
    scalar: &str,
    shape: &str,
    model: PerfModel,
) {
    report_model(bench, scalar, shape, model);
    group.throughput(Throughput::Bytes(model.bytes));
}

#[allow(dead_code)]
pub fn set_flop_throughput(
    group: &mut BenchmarkGroup<'_, WallTime>,
    bench: &str,
    scalar: &str,
    shape: &str,
    model: PerfModel,
) {
    report_model(bench, scalar, shape, model);
    group.throughput(Throughput::Elements(model.flops));
}

#[allow(dead_code)]
pub fn set_flop_throughput_with_variant(
    group: &mut BenchmarkGroup<'_, WallTime>,
    bench: &str,
    scalar: &str,
    shape: &str,
    variant: &str,
    precision: &str,
    model: PerfModel,
) {
    report_model_with_variant(bench, scalar, shape, variant, precision, model);
    group.throughput(Throughput::Elements(model.flops));
}
