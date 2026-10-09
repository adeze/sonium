//! Bounded W2-03 evidence for reproducible CPU compute timing and memory.
//!
//! This is intentionally an isolated integration-test source. It exercises
//! the public `CpuBackend`/`Backend` API, records a timed repeated GEMM, and
//! replays the same workload so the evidence includes provenance, memory, and
//! deterministic checksum fields without depending on the shared QA runner.

use std::time::Instant;

use sonium_compute::{
    Backend, CapabilityState, CpuBackend, MemoryPool, MemoryPoolConfig, Transpose,
};

const CASE_ID: &str = "w2-03.reproducible-cpu-compute-evidence.v1";
const SCHEMA_VERSION: u32 = 1;
const WORKLOAD: &str = "sonium-compute::CpuBackend::try_gemm<f64>";
const M: usize = 64;
const N: usize = 56;
const K: usize = 48;
const REPETITIONS: usize = 12;
const MEMORY_CAP_BYTES: usize = 1 << 20;

#[derive(Clone, Copy)]
struct Evidence {
    case_id: &'static str,
    schema_version: u32,
    workload: &'static str,
    backend: &'static str,
    gemm_f64_capability: &'static str,
    target_os: &'static str,
    target_arch: &'static str,
    input_fingerprint: u64,
    output_checksum: u64,
    reference_checksum: u64,
    max_abs_error: f64,
    elapsed_ns: u128,
    pool_bytes_in_use: usize,
    pool_high_water_bytes: usize,
    expected_buffer_bytes: usize,
}

fn deterministic_values(len: usize, seed: u64) -> Vec<f64> {
    (0..len)
        .map(|index| {
            let phase = ((index as u64 * 37 + seed) % 101) as f64;
            (phase - 50.0) / 37.0
        })
        .collect()
}

fn fnv1a_f64(values: &[f64]) -> u64 {
    let mut hash = 0xcbf29ce484222325_u64;
    for value in values {
        hash ^= value.to_bits();
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

/// A deliberately different checksum mix used for the scalar reference.
/// This keeps replay identity separate from the backend-output fingerprint.
fn reference_checksum(values: &[f64]) -> u64 {
    let mut hash = 0x9e3779b97f4a7c15_u64;
    for (index, value) in values.iter().enumerate() {
        let mixed = value
            .to_bits()
            .rotate_left((index % 63) as u32)
            .wrapping_add(index as u64);
        hash ^= mixed;
        hash = hash.rotate_left(13).wrapping_mul(0xbf58476d1ce4e5b9);
    }
    hash
}

fn capability_name(state: CapabilityState) -> &'static str {
    match state {
        CapabilityState::Native => "native",
        CapabilityState::VendorNative => "vendor-native",
        CapabilityState::HostSharedFallback => "host-shared-fallback",
        CapabilityState::HostStagedFallback => "host-staged-fallback",
        CapabilityState::Unsupported => "unsupported",
    }
}

fn reference_gemm(a: &[f64], b: &[f64], initial_c: &[f64]) -> Vec<f64> {
    let mut current = initial_c.to_vec();
    let mut next = vec![0.0; M * N];

    for _ in 0..REPETITIONS {
        for column in 0..N {
            for row in 0..M {
                let mut product = 0.0;
                for inner in 0..K {
                    product += a[row * K + inner] * b[inner * N + column];
                }
                let output_index = row * N + column;
                next[output_index] = product + 0.25 * current[output_index];
            }
        }
        std::mem::swap(&mut current, &mut next);
    }

    current
}

fn measure_once(
    a: &[f64],
    b: &[f64],
    initial_c: &[f64],
    expected: &[f64],
    input_fingerprint: u64,
) -> Evidence {
    let pool = MemoryPool::new(CpuBackend, MemoryPoolConfig::with_cap(MEMORY_CAP_BYTES))
        .expect("CPU evidence pool should be constructible");
    let mut a_buffer = pool
        .try_alloc::<f64>(a.len())
        .expect("A allocation should fit the evidence pool");
    let mut b_buffer = pool
        .try_alloc::<f64>(b.len())
        .expect("B allocation should fit the evidence pool");
    let mut c_buffer = pool
        .try_alloc::<f64>(initial_c.len())
        .expect("C allocation should fit the evidence pool");

    pool.backend().copy_h2d(a, a_buffer.raw_mut());
    pool.backend().copy_h2d(b, b_buffer.raw_mut());
    pool.backend().copy_h2d(initial_c, c_buffer.raw_mut());

    // Transfers and allocation are deliberately outside this timer. The
    // timed region is the CPU compute contract plus its synchronization call.
    let start = Instant::now();
    for _ in 0..REPETITIONS {
        pool.backend()
            .try_gemm(
                Transpose::NoTrans,
                Transpose::NoTrans,
                M,
                N,
                K,
                1.0,
                a_buffer.raw(),
                K,
                b_buffer.raw(),
                N,
                0.25,
                c_buffer.raw_mut(),
                N,
            )
            .expect("CPU GEMM should accept the finite evidence workload");
    }
    pool.backend()
        .synchronize()
        .expect("CPU synchronization should succeed");
    let elapsed_ns = start.elapsed().as_nanos();

    let mut output = vec![0.0; M * N];
    pool.backend().copy_d2h(c_buffer.raw(), &mut output);

    let max_abs_error = output
        .iter()
        .zip(expected)
        .map(|(actual, reference)| (actual - reference).abs())
        .fold(0.0, f64::max);
    let capabilities = pool.backend().capabilities();

    Evidence {
        case_id: CASE_ID,
        schema_version: SCHEMA_VERSION,
        workload: WORKLOAD,
        backend: capabilities.backend_name,
        gemm_f64_capability: capability_name(capabilities.gemm.f64),
        target_os: std::env::consts::OS,
        target_arch: std::env::consts::ARCH,
        input_fingerprint,
        output_checksum: fnv1a_f64(&output),
        reference_checksum: reference_checksum(expected),
        max_abs_error,
        elapsed_ns,
        pool_bytes_in_use: pool.bytes_in_use(),
        pool_high_water_bytes: pool.high_water_bytes(),
        expected_buffer_bytes: (a.len() + b.len() + initial_c.len()) * std::mem::size_of::<f64>(),
    }
}

fn assert_finite_evidence(evidence: Evidence) {
    assert_eq!(evidence.case_id, CASE_ID);
    assert_eq!(evidence.schema_version, SCHEMA_VERSION);
    assert_eq!(evidence.workload, WORKLOAD);
    assert_eq!(evidence.backend, "cpu");
    assert_eq!(evidence.gemm_f64_capability, "native");
    assert!(!evidence.target_os.is_empty());
    assert!(!evidence.target_arch.is_empty());
    assert_ne!(evidence.input_fingerprint, 0);
    assert_ne!(evidence.output_checksum, 0);
    assert_ne!(evidence.reference_checksum, 0);
    assert!(evidence.max_abs_error.is_finite());
    assert!(evidence.max_abs_error <= 1.0e-11);
    assert!(evidence.elapsed_ns > 0);
    assert_eq!(evidence.pool_bytes_in_use, evidence.expected_buffer_bytes);
    assert_eq!(
        evidence.pool_high_water_bytes,
        evidence.expected_buffer_bytes
    );
    assert!(evidence.pool_high_water_bytes <= MEMORY_CAP_BYTES);
}

#[test]
fn reproducible_cpu_compute_evidence_v1() {
    let a = deterministic_values(M * K, 11);
    let b = deterministic_values(K * N, 23);
    let initial_c = deterministic_values(M * N, 47);
    let input_fingerprint = fnv1a_f64(&a)
        .wrapping_add(fnv1a_f64(&b).rotate_left(17))
        .wrapping_add(fnv1a_f64(&initial_c).rotate_left(31));
    let expected = reference_gemm(&a, &b, &initial_c);

    // Warm the process once so one-time runtime initialization does not make
    // the replay evidence depend on which run happens to pay that cost.
    let _warmup = measure_once(&a, &b, &initial_c, &expected, input_fingerprint);
    let first = measure_once(&a, &b, &initial_c, &expected, input_fingerprint);
    let replay = measure_once(&a, &b, &initial_c, &expected, input_fingerprint);

    assert_finite_evidence(first);
    assert_finite_evidence(replay);
    assert_eq!(first.input_fingerprint, replay.input_fingerprint);
    assert_eq!(first.output_checksum, replay.output_checksum);
    assert_eq!(first.reference_checksum, replay.reference_checksum);
    assert_eq!(first.pool_bytes_in_use, replay.pool_bytes_in_use);
    assert_eq!(first.pool_high_water_bytes, replay.pool_high_water_bytes);
    assert_eq!(
        first.max_abs_error.to_bits(),
        replay.max_abs_error.to_bits()
    );

    eprintln!(
        "case_id={} schema={} workload={} backend={} capability={} target={}-{} dims={}x{}x{} repetitions={} elapsed_ns={}/{} input_fingerprint=0x{:016x} output_checksum=0x{:016x} reference_checksum=0x{:016x} pool_bytes={} high_water_bytes={} max_abs_error={:.3e}",
        first.case_id,
        first.schema_version,
        first.workload,
        first.backend,
        first.gemm_f64_capability,
        first.target_os,
        first.target_arch,
        M,
        N,
        K,
        REPETITIONS,
        first.elapsed_ns,
        replay.elapsed_ns,
        first.input_fingerprint,
        first.output_checksum,
        first.reference_checksum,
        first.pool_bytes_in_use,
        first.pool_high_water_bytes,
        first.max_abs_error,
    );
}
