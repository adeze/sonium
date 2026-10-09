//! f32 accuracy and slice-dispatch coverage.

use sonium_fastmath::{sincos_f32, sincos_f32_slice};

#[test]
fn f32_scalar_matches_platform_pair() {
    for i in 0..20_000 {
        let x = -1000.0_f32 + 2000.0 * i as f32 / 19_999.0;
        let (sin, cos) = sincos_f32(x);
        let (sin_ref, cos_ref) = x.sin_cos();
        assert!(
            (sin - sin_ref).abs() <= 3e-6,
            "sin error at {x}: {sin} vs {sin_ref}"
        );
        assert!(
            (cos - cos_ref).abs() <= 3e-6,
            "cos error at {x}: {cos} vs {cos_ref}"
        );
    }
}

#[test]
fn f32_slice_matches_scalar_for_simd_widths_and_tail() {
    let input: Vec<f32> = (0..37).map(|i| -17.0 + i as f32 * 0.91).collect();
    let mut sins = vec![0.0; input.len()];
    let mut coss = vec![0.0; input.len()];
    sincos_f32_slice(&input, &mut sins, &mut coss);
    for i in 0..input.len() {
        let (sin, cos) = sincos_f32(input[i]);
        assert_eq!(sins[i], sin);
        assert_eq!(coss[i], cos);
    }
}
