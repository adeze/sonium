//! Emit deterministic fastmath throughput and tail-latency measurements.
//!
//! The artifact compares the dispatched slice path, scalar Rust path, and
//! platform `f64::sin_cos` implementation. It is intentionally a small
//! standalone binary so CI can archive one JSON object without parsing
//! Criterion's human-oriented output.

use sonium_fastmath::{
    dispatch_features, reset_slow_path_count, selected_backend, sincos_f64_slice, sincos_scalar,
    slow_path_count,
};
use std::hint::black_box;
use std::time::Instant;

const SAMPLES: usize = 4096;
const BATCHES: usize = 32;
const SCHEMA_VERSION: u32 = 1;

#[derive(Clone, Copy)]
struct Timing {
    ns_per_sample: f64,
    p95_batch_ns: u128,
    p99_batch_ns: u128,
    checksum: u64,
    energy_joules: Option<f64>,
}

fn percentile(values: &mut [u128], percentile: usize) -> u128 {
    values.sort_unstable();
    values[((values.len() - 1) * percentile) / 100]
}

fn json_escape(value: &str) -> String {
    value.replace('\\', "\\\\").replace('"', "\\\"")
}

fn optional_string_json(value: Option<String>) -> String {
    value
        .map(|value| format!("\"{}\"", json_escape(&value)))
        .unwrap_or_else(|| "null".to_owned())
}

fn read_first_line(paths: &[&str]) -> Option<String> {
    paths.iter().find_map(|path| {
        std::fs::read_to_string(path)
            .ok()
            .map(|value| value.trim().to_owned())
            .filter(|value| !value.is_empty())
    })
}

fn cpu_description() -> String {
    #[cfg(target_os = "macos")]
    {
        for key in ["machdep.cpu.brand_string", "hw.model"] {
            if let Ok(output) = std::process::Command::new("sysctl")
                .args(["-n", key])
                .output()
                && output.status.success()
            {
                let value = String::from_utf8_lossy(&output.stdout).trim().to_owned();
                if !value.is_empty() {
                    return value;
                }
            }
        }
    }
    #[cfg(target_os = "linux")]
    {
        if let Ok(contents) = std::fs::read_to_string("/proc/cpuinfo")
            && let Some(model) = contents
                .lines()
                .find_map(|line| line.strip_prefix("model name\t: "))
        {
            return model.to_owned();
        }
    }
    "unknown".to_owned()
}

fn microcode_version() -> Option<String> {
    read_first_line(&[
        "/sys/devices/system/cpu/microcode/version",
        "/sys/devices/system/cpu/cpu0/microcode/version",
    ])
}

fn clock_state() -> Option<String> {
    read_first_line(&[
        "/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq",
        "/sys/devices/system/cpu/cpu0/cpufreq/scaling_setspeed",
    ])
}

fn clock_governor() -> Option<String> {
    read_first_line(&["/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor"])
}

fn clock_min_state() -> Option<String> {
    read_first_line(&["/sys/devices/system/cpu/cpu0/cpufreq/scaling_min_freq"])
}

fn clock_max_state() -> Option<String> {
    read_first_line(&["/sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq"])
}

fn thermal_state() -> Option<String> {
    read_first_line(&["/sys/class/thermal/thermal_zone0/temp"])
}

fn clock_source() -> &'static str {
    if cfg!(target_os = "linux") {
        "/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq"
    } else {
        "platform_clock_unavailable"
    }
}

fn thermal_source() -> &'static str {
    if cfg!(target_os = "linux") {
        "/sys/class/thermal/thermal_zone0/temp"
    } else {
        "platform_thermal_unavailable"
    }
}

fn energy_source() -> &'static str {
    if cfg!(target_os = "linux") {
        "/sys/class/powercap/**/energy_uj"
    } else {
        "platform_energy_unavailable"
    }
}

fn compiler_version() -> String {
    std::process::Command::new("rustc")
        .arg("-vV")
        .output()
        .ok()
        .filter(|output| output.status.success())
        .and_then(|output| String::from_utf8(output.stdout).ok())
        .and_then(|output| output.lines().next().map(str::to_owned))
        .unwrap_or_else(|| "unknown".to_owned())
}

fn vendor_reference_status_json() -> String {
    // The benchmark has no vendor FFI, so even a library discovered on the
    // host is deliberately reported as unavailable-to-this-producer rather
    // than timed with an unverified ABI.  The status is still split out so a
    // future pinned runner can make the comparison explicit.
    let linker_cache = std::process::Command::new("ldconfig")
        .args(["-p"])
        .output()
        .ok()
        .filter(|output| output.status.success())
        .map(|output| String::from_utf8_lossy(&output.stdout).to_ascii_lowercase())
        .unwrap_or_default();
    let available = |needles: &[&str]| needles.iter().any(|needle| linker_cache.contains(needle));
    let status = |name: &str, present: bool| {
        if present {
            format!(
                "{{\"name\":\"{name}\",\"status\":\"available_not_benchmarked\",\"reason\":\"no verified FFI in this producer\"}}"
            )
        } else {
            format!(
                "{{\"name\":\"{name}\",\"status\":\"unavailable\",\"reason\":\"no linker-visible library\"}}"
            )
        }
    };
    [
        status("sleef", available(&["libsleef"])),
        status("svml", available(&["libsvml"])),
        status("oneMKL", available(&["mkl_rt", "libmkl"])),
    ]
    .join(",")
}

fn require_native_avx2() {
    if std::env::var("SONIUM_FASTMATH_REQUIRE_AVX2").as_deref() != Ok("1") {
        return;
    }
    #[cfg(target_arch = "x86_64")]
    assert!(
        std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma"),
        "SONIUM_FASTMATH_REQUIRE_AVX2=1 but native AVX2/FMA is unavailable"
    );
    #[cfg(not(target_arch = "x86_64"))]
    panic!("SONIUM_FASTMATH_REQUIRE_AVX2=1 requires an x86_64 target");
}

fn energy_uj() -> Option<u64> {
    #[cfg(target_os = "linux")]
    {
        let root = std::path::Path::new("/sys/class/powercap");
        let entries = std::fs::read_dir(root).ok()?;
        for entry in entries.flatten() {
            let path = entry.path().join("energy_uj");
            if let Ok(value) = std::fs::read_to_string(path)
                && let Ok(value) = value.trim().parse()
            {
                return Some(value);
            }
        }
    }
    None
}

fn energy_delta(before: Option<u64>, after: Option<u64>) -> Option<f64> {
    Some(after?.saturating_sub(before?) as f64 / 1.0e6)
}

fn output_checksum(output: &[(f64, f64)]) -> u64 {
    let mut checksum = 0xcbf2_9ce4_8422_2325_u64;
    for &(sin, cos) in output {
        for bits in [sin.to_bits(), cos.to_bits()] {
            checksum ^= bits;
            checksum = checksum.wrapping_mul(0x1000_0000_01b3);
        }
    }
    checksum
}

fn input_checksum(input: &[f64]) -> u64 {
    let mut checksum = 0xcbf2_9ce4_8422_2325_u64;
    for &phase in input {
        checksum ^= phase.to_bits();
        checksum = checksum.wrapping_mul(0x1000_0000_01b3);
    }
    checksum
}

fn correctness_gate(input: &[f64]) {
    let mut sins = vec![0.0; input.len()];
    let mut coss = vec![0.0; input.len()];
    sincos_f64_slice(input, &mut sins, &mut coss);
    for (i, &phase) in input.iter().enumerate() {
        let (expected_sin, expected_cos) = phase.sin_cos();
        if expected_sin.is_nan() {
            assert!(
                sins[i].is_nan() && coss[i].is_nan(),
                "correctness gate at {phase}"
            );
            continue;
        }
        let budget = if phase.abs() <= sonium_fastmath::CODY_WAITE_LIMIT {
            8.0 * f64::EPSILON
        } else {
            64.0 * f64::EPSILON
        };
        assert!(
            (sins[i] - expected_sin).abs() <= budget && (coss[i] - expected_cos).abs() <= budget,
            "correctness gate at {phase}: ({}, {}) vs ({expected_sin}, {expected_cos})",
            sins[i],
            coss[i]
        );
    }
}

fn time_fast(input: &[f64]) -> Timing {
    let mut sins = vec![0.0; input.len()];
    let mut coss = vec![0.0; input.len()];
    let mut samples = Vec::with_capacity(BATCHES);
    let energy_before = energy_uj();
    reset_slow_path_count();
    for _ in 0..BATCHES {
        let start = Instant::now();
        sincos_f64_slice(input, &mut sins, &mut coss);
        samples.push(start.elapsed().as_nanos());
        black_box((&sins, &coss));
    }
    let total_ns: u128 = samples.iter().sum();
    let p95_batch_ns = percentile(&mut samples, 95);
    let p99_batch_ns = percentile(&mut samples, 99);
    let output: Vec<_> = sins.iter().copied().zip(coss.iter().copied()).collect();
    Timing {
        ns_per_sample: total_ns as f64 / (BATCHES * input.len()) as f64,
        p95_batch_ns,
        p99_batch_ns,
        checksum: output_checksum(&output),
        energy_joules: energy_delta(energy_before, energy_uj()),
    }
}

fn time_scalar(input: &[f64]) -> Timing {
    let mut output = vec![(0.0, 0.0); input.len()];
    let mut samples = Vec::with_capacity(BATCHES);
    let energy_before = energy_uj();
    for _ in 0..BATCHES {
        let start = Instant::now();
        for (slot, &phase) in output.iter_mut().zip(input) {
            *slot = sincos_scalar(phase);
        }
        samples.push(start.elapsed().as_nanos());
        black_box(&output);
    }
    let total_ns: u128 = samples.iter().sum();
    let p95_batch_ns = percentile(&mut samples, 95);
    let p99_batch_ns = percentile(&mut samples, 99);
    Timing {
        ns_per_sample: total_ns as f64 / (BATCHES * input.len()) as f64,
        p95_batch_ns,
        p99_batch_ns,
        checksum: output_checksum(&output),
        energy_joules: energy_delta(energy_before, energy_uj()),
    }
}

fn time_libm(input: &[f64]) -> Timing {
    let mut output = vec![(0.0, 0.0); input.len()];
    let mut samples = Vec::with_capacity(BATCHES);
    let energy_before = energy_uj();
    for _ in 0..BATCHES {
        let start = Instant::now();
        for (slot, &phase) in output.iter_mut().zip(input) {
            *slot = phase.sin_cos();
        }
        samples.push(start.elapsed().as_nanos());
        black_box(&output);
    }
    let total_ns: u128 = samples.iter().sum();
    let p95_batch_ns = percentile(&mut samples, 95);
    let p99_batch_ns = percentile(&mut samples, 99);
    Timing {
        ns_per_sample: total_ns as f64 / (BATCHES * input.len()) as f64,
        p95_batch_ns,
        p99_batch_ns,
        checksum: output_checksum(&output),
        energy_joules: energy_delta(energy_before, energy_uj()),
    }
}

fn make_inputs(name: &str) -> Vec<f64> {
    (0..SAMPLES)
        .map(|i| {
            let t = i as f64 / SAMPLES as f64;
            match name {
                "reduced" => (t * 2.0 - 1.0) * std::f64::consts::FRAC_PI_4,
                "coherent_ramp" => (t * 20.0 - 10.0) * std::f64::consts::PI,
                "random_full_range" => {
                    // Deterministic LCG; keeping this dependency-free makes
                    // the artifact runnable in minimal release environments.
                    let bits = (i as u64)
                        .wrapping_mul(6_364_136_223_846_793_005)
                        .wrapping_add(1_442_695_040_888_963_407);
                    (bits as f64 / u64::MAX as f64 - 0.5) * 2.0e3
                }
                "reduction_boundaries" => {
                    let k = (i % 257) as f64 - 128.0;
                    k * std::f64::consts::FRAC_PI_2 + (i as f64 % 7.0 - 3.0) * 1.0e-12
                }
                "payne_hanek" => 1.0e12 + i as f64 * 1.0e6,
                _ => unreachable!("unknown fastmath benchmark regime"),
            }
        })
        .collect()
}

fn main() {
    require_native_avx2();
    let regimes = [
        "reduced",
        "coherent_ramp",
        "random_full_range",
        "reduction_boundaries",
        "payne_hanek",
    ];
    let compiler = json_escape(&compiler_version());
    let cpu = json_escape(&cpu_description());
    let microcode = optional_string_json(microcode_version());
    let clock = optional_string_json(clock_state());
    let governor = optional_string_json(clock_governor());
    let clock_min = optional_string_json(clock_min_state());
    let clock_max = optional_string_json(clock_max_state());
    let thermal = optional_string_json(thermal_state());
    let energy_available = energy_uj().is_some();
    let runtime_features = json_escape(&dispatch_features());
    let vendor_references = vendor_reference_status_json();
    let mut reports = Vec::with_capacity(regimes.len());
    for regime in regimes {
        let input = make_inputs(regime);
        // Correctness is a hard gate and deliberately precedes all timing.
        correctness_gate(&input);
        let fast = time_fast(&input);
        let slow_path_rate = slow_path_count() as f64 / (BATCHES * input.len()) as f64;
        let scalar = time_scalar(&input);
        let libm = time_libm(&input);
        reports.push(format!(
            "{{\"name\":\"{regime}\",\"samples\":{},\"input_checksum\":{},\"slow_path_rate\":{slow_path_rate:.17e},\"fast_checksum\":{},\"scalar_checksum\":{},\"libm_checksum\":{},\"fast_ns_per_sample\":{:.6},\"scalar_ns_per_sample\":{:.6},\"libm_ns_per_sample\":{:.6},\"fast_p95_batch_ns\":{},\"fast_p99_batch_ns\":{},\"scalar_p95_batch_ns\":{},\"scalar_p99_batch_ns\":{},\"libm_p95_batch_ns\":{},\"libm_p99_batch_ns\":{},\"fast_energy_joules\":{},\"scalar_energy_joules\":{},\"libm_energy_joules\":{}}}",
            input.len(),
            input_checksum(&input),
            fast.checksum,
            scalar.checksum,
            libm.checksum,
            fast.ns_per_sample,
            scalar.ns_per_sample,
            libm.ns_per_sample,
            fast.p95_batch_ns,
            fast.p99_batch_ns,
            scalar.p95_batch_ns,
            scalar.p99_batch_ns,
            libm.p95_batch_ns,
            libm.p99_batch_ns,
            fast.energy_joules
                .map(|value| format!("{value:.17e}"))
                .unwrap_or_else(|| "null".to_owned()),
            scalar
                .energy_joules
                .map(|value| format!("{value:.17e}"))
                .unwrap_or_else(|| "null".to_owned()),
            libm.energy_joules
                .map(|value| format!("{value:.17e}"))
                .unwrap_or_else(|| "null".to_owned()),
        ));
    }
    println!(
        "{{\"schema_version\":{},\"architecture\":\"{}\",\"compiler\":\"{}\",\"target_features\":\"{}\",\"runtime_features\":\"{}\",\"dispatch\":\"{}\",\"cpu\":\"{}\",\"microcode\":{},\"clock_source\":\"{}\",\"clock_state\":{},\"clock_policy\":{{\"governor\":{},\"min_state\":{},\"max_state\":{}}},\"thermal_source\":\"{}\",\"thermal_state\":{},\"energy_source\":\"{}\",\"energy_available\":{},\"correctness_gate\":\"passed\",\"samples_per_regime\":{},\"batches\":{},\"references\":[\"scalar_rust\",\"platform_f64_sin_cos\"],\"external_references\":[{}],\"regimes\":[{}]}}",
        SCHEMA_VERSION,
        std::env::consts::ARCH,
        compiler,
        json_escape(option_env!("SONIUM_TARGET_FEATURES").unwrap_or("compiler-default")),
        runtime_features,
        selected_backend().as_str(),
        cpu,
        microcode,
        json_escape(clock_source()),
        clock,
        governor,
        clock_min,
        clock_max,
        json_escape(thermal_source()),
        thermal,
        json_escape(energy_source()),
        energy_available,
        SAMPLES,
        BATCHES,
        vendor_references,
        reports.join(",")
    );
}
