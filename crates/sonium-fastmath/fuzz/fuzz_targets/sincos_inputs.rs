#![no_main]

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    for chunk in data.chunks_exact(8).take(64) {
        let mut bytes = [0_u8; 8];
        bytes.copy_from_slice(chunk);
        let value = f64::from_le_bytes(bytes);
        let _ = sonium_fastmath::sincos_f64(value);
        let _ = sonium_fastmath::exp_i(value);
        let _ = sonium_fastmath::sincos_with_derivatives(value);
        let _ = sonium_fastmath::sincos_f32(value as f32);
    }
});
