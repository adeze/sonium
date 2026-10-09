#![no_main]

use libfuzzer_sys::fuzz_target;
use sonium_fastmath::sincos_f64;

fuzz_target!(|data: &[u8]| {
    // Interpret arbitrary bytes as IEEE-754 values. Limiting each input to a
    // bounded number of lanes keeps a malformed corpus entry from turning
    // one execution into an accidental throughput benchmark.
    for chunk in data.chunks_exact(8).take(64) {
        let x = f64::from_bits(u64::from_le_bytes(
            chunk.try_into().expect("chunks_exact supplies eight bytes"),
        ));
        let (sin, cos) = sincos_f64(x);

        if x.is_finite() {
            assert!(sin.is_finite() && cos.is_finite());
            let identity = sin.mul_add(sin, cos * cos);
            assert!((identity - 1.0).abs() <= 8.0e-14);
        } else {
            assert!(sin.is_nan() && cos.is_nan());
        }
    }
});
