#![no_main]

use libfuzzer_sys::fuzz_target;
use num_complex::Complex64;
use sonium_fft::FftPlanner;

fuzz_target!(|bytes: &[u8]| {
    let n = 1 + bytes.first().copied().unwrap_or_default() as usize % 256;
    let input = (0..n)
        .map(|index| {
            let re = bytes.get(1 + 2 * index).copied().unwrap_or_default() as f64 / 127.5 - 1.0;
            let im = bytes.get(2 + 2 * index).copied().unwrap_or_default() as f64 / 127.5 - 1.0;
            Complex64::new(re, im)
        })
        .collect::<Vec<_>>();

    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let planner = FftPlanner::<f64>::new();
        let forward = planner.plan_forward(n);
        let inverse = planner.plan_inverse(n);
        let mut transformed = input.clone();
        forward.process(&mut transformed);
        inverse.process(&mut transformed);
        let scale = n as f64;
        for (actual, expected) in transformed.iter().zip(&input) {
            assert!(((*actual / scale) - *expected).norm() <= 2.0e-9);
        }
    }));
    assert!(result.is_ok(), "FFT planner/round-trip panicked on fuzz input");
});
